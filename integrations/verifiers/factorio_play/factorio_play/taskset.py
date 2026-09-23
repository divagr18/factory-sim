"""factorio-play: build a smelting line through `world.*` tool calls, one scene per rollout.

Each tool call is one `World` method call on a live factory-sim episode (see
`servers/world.py` and `session.py`). The model calls `finish` when done; the
episode then runs its verification window, and the reward is whether it
verified. A rollout that never calls `finish` scores 0: its build phase never
ended, so nothing was verified.
"""

from __future__ import annotations

import logging
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field
from verifiers.v1.harnesses.null import NullHarness

from evolve import mutate
from factorio_play import scenes as scene_plan
from factorio_play.servers.world import WorldState, WorldToolset, WorldToolsetConfig

logger = logging.getLogger(__name__)

TOOLS_NOTE = """\
You act through tools that mirror the `world` API one to one: queries (me, tile, \
inventory, ore_tiles, blocked_tiles, entities, patch, decisions_left, last_refused) \
cost nothing; actions (move, place, give, take, mine, wait) each spend a decision. \
give/take/mine name an entity by its current `row` in entities(). Every tool returns \
{"ok": ..., "result": ...} or an error. Call finish() once the line is built and \
fuelled: the episode is scored only after finish()."""


def system_prompt(game_notes: bool = True) -> str:
    parts = [
        "You build factories in a Factorio-like simulator by calling tools.",
        mutate.TASK,
        TOOLS_NOTE,
    ]
    if game_notes and mutate.GAME_NOTES:
        parts.append(mutate.GAME_NOTES)
    return "\n\n".join(parts)


class FactorioPlayData(vf.TaskData):
    sim_task: str
    split: str
    scene_id: str
    family: str
    seed: int
    sample_split: str


class FactorioPlayTaskConfig(vf.TaskConfig):
    tools: WorldToolsetConfig = WorldToolsetConfig()


class FactorioPlayTask(vf.Task[FactorioPlayData, WorldState, FactorioPlayTaskConfig]):
    @property
    def key(self) -> str:
        return self.data.scene_id

    @classmethod
    def toolsets(cls, config: FactorioPlayTaskConfig) -> list[vf.Toolset]:
        return [WorldToolset(config.tools)]

    @vf.metric
    async def episode(self, trace: vf.Trace) -> dict[str, float]:
        st = trace.state
        return {
            "finished": float(st.finished),
            "verified_output": float(st.verified_output),
            "decisions": float(st.decisions),
            "refusals": float(st.refusals),
            "tool_calls": float(st.tool_calls),
        }

    @vf.reward(weight=1.0)
    async def success(self, trace: vf.Trace) -> float:
        st = trace.state
        return float(st.finished and st.success)


class FactorioPlayConfig(vf.TasksetConfig):
    sim_task: str = "construct_smelting_line"
    split: Literal["train", "val", "holdout"] = "train"
    """`holdout` is the frozen FactorioRL holdout: evaluation only."""
    num_examples: int = Field(64, ge=1)
    """Scenes (one per task)."""
    seed: int = Field(0, ge=0)
    """First scene index of the split."""
    game_notes: bool = True
    task: FactorioPlayTaskConfig = FactorioPlayTaskConfig()


class FactorioPlayTaskset(vf.Taskset[FactorioPlayTask, FactorioPlayConfig]):
    def load(self) -> list[FactorioPlayTask]:
        cfg = self.config
        if cfg.split == "holdout":
            logger.warning("factorio-play: split=holdout is for evaluation only")
        refs = scene_plan.scene_block(cfg.sim_task, cfg.split, cfg.seed, cfg.num_examples)
        system = system_prompt(cfg.game_notes)
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
                        prompt="Build the smelting line in this scene with the tools, "
                        "then call finish().",
                    ),
                    cfg.task,
                )
            )
        return tasks


class FactorioPlayHarness(NullHarness):
    """The built-in `null` chat loop (with MCP tools), made this taskset's default."""


__all__ = ["FactorioPlayHarness", "FactorioPlayTaskset"]
