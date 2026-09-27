"""factorio-play: build a factory through `world.*` tool calls, one scene per rollout.

Two tasks. `construct_smelting_line` (the default): each tool call is one
`World` method call on a live factory-sim episode (`servers/world.py`,
`session.py`), and the reward is whether the line verified. `belt_smelting`:
the tools are factory-sim's `WorldV3` (`servers/belt_world.py`,
`belt_tools.py`), where a tool that takes a count or an amount can run several
world actions, and the reward is factory-sim's own score for the task,
min(1, plates delivered / 150), which is 1 exactly when it succeeds.

The model calls `finish` when done; the episode then runs its verification
window. A rollout that never calls `finish` scores 0: its build phase never
ended, so nothing was verified.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field, model_validator
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.harnesses.null import NullHarness

from evolve import evaluate
from factorio_play import belt_tools, prompts
from factorio_play import scenes as scene_plan
from factorio_play.servers.belt_world import BeltToolsetConfig, BeltWorldToolset
from factorio_play.servers.world import WorldState, WorldToolset, WorldToolsetConfig
from fsim.program_api import TASK_PROFILES

logger = logging.getLogger(__name__)


class FactorioPlayData(vf.TaskData):
    sim_task: str
    split: str
    scene_id: str
    family: str
    seed: int
    sample_split: str


class FactorioPlayTaskConfig(vf.TaskConfig):
    tools: WorldToolsetConfig = WorldToolsetConfig()
    """construct_smelting_line's tool server."""
    belt_tools: BeltToolsetConfig = BeltToolsetConfig()
    """belt_smelting's tool server."""
    sim_task: str = evaluate.TASK
    """Which tool server a rollout gets. The taskset sets it from its own `sim_task`."""


class FactorioPlayTask(vf.Task[FactorioPlayData, WorldState, FactorioPlayTaskConfig]):
    @property
    def key(self) -> str:
        return self.data.scene_id

    @classmethod
    def toolsets(cls, config: FactorioPlayTaskConfig) -> list[vf.Toolset]:
        if TASK_PROFILES.get(config.sim_task, "v2") == "v3":
            return [BeltWorldToolset(config.belt_tools)]
        return [WorldToolset(config.tools)]

    @vf.metric
    async def episode(self, trace: vf.Trace) -> dict[str, float]:
        st = trace.state
        out = {
            "finished": float(st.finished),
            "verified_output": float(st.verified_output),
            "decisions": float(st.decisions),
            "refusals": float(st.refusals),
            "tool_calls": float(st.tool_calls),
        }
        if self.data.sim_task == prompts.BELT_TASK:
            out["success"] = float(st.finished and st.success)
        return out

    @vf.reward(weight=1.0)
    async def success(self, trace: vf.Trace) -> float | dict[str, float]:
        """construct_smelting_line: 1 if the line verified. belt_smelting: recorded as
        `score`, factory-sim's min(1, plates / 150)."""
        st = trace.state
        if self.data.sim_task == prompts.BELT_TASK:
            return {"score": belt_tools.score(st.verified_output) if st.finished else 0.0}
        return float(st.finished and st.success)


class FactorioPlayConfig(vf.TasksetConfig):
    sim_task: Literal["construct_smelting_line", "belt_smelting"] = "construct_smelting_line"
    """factory-sim task: construct_smelting_line or belt_smelting."""
    split: Literal["train", "val", "holdout"] = "train"
    """`holdout` is the frozen FactorioRL holdout: evaluation only."""
    num_examples: int = Field(64, ge=1)
    """Scenes (one per task)."""
    seed: int = Field(0, ge=0)
    """First scene index of the split."""
    game_notes: bool = True
    task: FactorioPlayTaskConfig = FactorioPlayTaskConfig()

    @model_validator(mode="after")
    def _tools_follow_sim_task(self):
        """The task config names the tool server `Task.toolsets` launches: this taskset's."""
        if self.task.sim_task != self.sim_task:
            self.task = self.task.model_copy(update={"sim_task": self.sim_task})
        return self


class FactorioPlayTaskset(vf.Taskset[FactorioPlayTask, FactorioPlayConfig]):
    def load(self) -> list[FactorioPlayTask]:
        cfg = self.config
        if cfg.split == "holdout":
            logger.warning("factorio-play: split=holdout is for evaluation only")
        refs = scene_plan.scene_block(cfg.sim_task, cfg.split, cfg.seed, cfg.num_examples)
        system = prompts.system_prompt(cfg.game_notes, cfg.sim_task)
        prompt = prompts.user_prompt(cfg.sim_task)
        tasks = []
        for i, ref in enumerate(refs):
            scene_id = f"{cfg.sim_task}/{cfg.split}/{cfg.seed + i}"
            tasks.append(
                FactorioPlayTask(
                    FactorioPlayData(
                        idx=i,
                        name=scene_id,
                        sim_task=cfg.sim_task,
                        split=cfg.split,
                        scene_id=scene_id,
                        family=ref.family,
                        seed=ref.seed,
                        sample_split=ref.sample_split,
                        system_prompt=system,
                        prompt=prompt,
                    ),
                    cfg.task,
                )
            )
        return tasks


RESPONSES_PROGRAM = (Path(__file__).resolve().parent / "responses_program.py").read_text(
    encoding="utf-8"
)


class FactorioPlayHarness(NullHarness):
    """This taskset's default agent: the `null` loop over the Responses API.

    Everything but the program is the built-in `null` harness: the same
    arguments and MCP wiring, with `responses_program.py` instead of its Chat
    Completions loop. Some reasoning models take function tools only on
    `/responses`."""

    async def setup(self, runtime) -> None:
        await runtime.prepare_uv_script(RESPONSES_PROGRAM, self.config.resolved_env)

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data):
        system_prompt, prompt = self.resolve_prompt(data)
        args = [f"--base-url={endpoint}", f"--api-key={secret}", f"--model={ctx.model}"]
        if system_prompt:
            args.append(f"--system-prompt={system_prompt}")
        if mcp_urls:
            servers = {
                name: {"url": url, "timeout": self.config.tool_timeout}
                for name, url in mcp_urls.items()
            }
            args.append("--mcp-config=" + json.dumps({"mcpServers": servers}))
        if isinstance(prompt, str):
            args.append(f"--prompt={prompt}")
        elif prompt is not None:
            path = f".vf-initial-messages-{trace.id}.json"
            await runtime.write(path, json.dumps([message_to_wire(m) for m in prompt]).encode())
            args.append(f"--initial-messages-file={path}")
        program = await runtime.prepare_uv_script(RESPONSES_PROGRAM, self.config.resolved_env)
        return await runtime.run_program([*program, *args], dict(self.config.resolved_env))


__all__ = ["FactorioPlayHarness", "FactorioPlayTaskset"]
