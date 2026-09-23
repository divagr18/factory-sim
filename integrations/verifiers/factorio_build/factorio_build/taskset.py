"""factorio-build as a native verifiers v1 taskset.

One task = one prompt asking for a `def build(world):` program, scored by
running that program on the task's scene subset in factory-sim. The reply is
the whole rollout (single turn): the package exports a `NullHarness` subclass
so an unpinned agent gets a plain chat loop with no tools, rather than
verifiers' default `bash` coding harness.

Scoring is one `@vf.metric` that runs the program (all simulation happens
there, off the event loop) and three `@vf.reward`s that read its numbers:

- `success_rate` (weight 1.0): fraction of the subset's scenes that verify;
- `format` (weight 0.1): +1 sandbox-valid program, -1 refused by the sandbox,
  0 when the reply holds no program at all;
- `refusal_penalty` (weight 0.0, opt-in): minus the share of the program's
  intents the action space refused. Turn it on from config:
  `[env.taskset.task.rewards] refusal_penalty = { weight = 0.05 }`.

None of the signals need a runtime, so `replay` can re-score saved traces.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field
from verifiers.v1.harnesses.null import NullHarness

from factorio_build import core

logger = logging.getLogger(__name__)


class FactorioBuildData(vf.TaskData):
    sim_task: str
    """The factory-sim task the program must solve."""
    split: str
    """train, val or holdout."""
    subset_id: str
    """`<task>/<split>/<first>-<last>`: which of the split's scenes this row runs."""
    scenes: list[tuple[str, int, str]]
    """(family, seed, sample split) per scene; `fsim.scenes.sample` rebuilds each."""


class FactorioBuildTaskConfig(vf.TaskConfig):
    workers: int = Field(4, ge=0)
    """Simulator worker processes per env worker; 0 runs programs in-process (no isolation)."""
    job_timeout_s: float = Field(30.0, gt=0)
    """Per-chunk wall-clock limit before a worker (and its program) is killed."""
    decision_budget: int = Field(600, ge=1)
    """Decisions the program may spend in the build phase of each scene."""


class FactorioBuildTask(vf.Task[FactorioBuildData, vf.State, FactorioBuildTaskConfig]):
    @property
    def key(self) -> str:
        return self.data.subset_id

    def _score(self, text: str | None, scenes=None) -> dict:
        cfg = self.config
        return core.score_completion(
            text,
            self.data.sim_task,
            scenes if scenes is not None else self.data.scenes,
            workers=cfg.workers,
            timeout_s=cfg.job_timeout_s,
            decision_budget=cfg.decision_budget,
        )

    @vf.metric
    async def evaluate(self, trace: vf.Trace) -> dict[str, float]:
        """Run the reply's program on this row's scenes; every reward reads these."""
        result = await asyncio.to_thread(self._score, trace.last_reply)
        trace.info["factorio_build"] = {
            "subset_id": self.data.subset_id,
            "split": self.data.split,
            "error": result["error"],
            "program_hash": result["program_hash"],
        }
        return {k: v for k, v in result.items() if isinstance(v, float)}

    @vf.reward(weight=1.0)
    async def success_rate(self, trace: vf.Trace) -> float:
        return float(trace.metrics.get("success") or 0.0)

    @vf.reward(weight=0.1)
    async def format(self, trace: vf.Trace) -> float:
        return core.format_score(trace.metrics)

    @vf.reward(weight=0.0)
    async def refusal_penalty(self, trace: vf.Trace) -> float:
        return -float(trace.metrics.get("refusal_rate") or 0.0)

    async def validate(self, runtime: vf.Runtime | None = None) -> bool:
        """Model-free check: the seed builder program solves at least one scene."""
        from evolve.seeds import builder

        text = f"```python\n{builder.SOURCE}\n```"
        result = await asyncio.to_thread(self._score, text)
        return result["success"] > 0


class FactorioBuildConfig(vf.TasksetConfig):
    sim_task: str = "construct_smelting_line"
    """factory-sim task. Only construct_smelting_line has a prompt and pinned sets."""
    split: Literal["train", "val", "holdout"] = "train"
    """Scene split. `holdout` is the frozen FactorioRL holdout: evaluation only."""
    n_scenes: int = Field(16, ge=1, le=core.MAX_SCENES)
    """Scenes per row; the reward is the success rate over them."""
    num_examples: int = Field(64, ge=1)
    """Rows (scene subsets) to build."""
    seed: int = Field(0, ge=0)
    """First scene index; row k covers indices seed + k*n_scenes onward."""
    game_notes: bool = True
    """Include `evolve.mutate.GAME_NOTES` (mechanics the API reference leaves out)."""
    task: FactorioBuildTaskConfig = FactorioBuildTaskConfig()


class FactorioBuildTaskset(vf.Taskset[FactorioBuildTask, FactorioBuildConfig]):
    def load(self) -> list[FactorioBuildTask]:
        cfg = self.config
        if cfg.split == "holdout":
            logger.warning(
                "factorio-build: split=holdout is the frozen held-out set; "
                "use it for evaluation only, never as a training reward"
            )
        return [
            FactorioBuildTask(
                FactorioBuildData(
                    idx=i,
                    name=row["subset_id"],
                    sim_task=row["task"],
                    split=row["split"],
                    subset_id=row["subset_id"],
                    scenes=row["scenes"],
                    system_prompt=row["system_prompt"],
                    prompt=row["prompt"],
                ),
                cfg.task,
            )
            for i, row in enumerate(
                core.rows(
                    cfg.sim_task,
                    cfg.split,
                    cfg.n_scenes,
                    cfg.num_examples,
                    cfg.seed,
                    cfg.game_notes,
                )
            )
        ]


class FactorioBuildHarness(NullHarness):
    """The built-in `null` harness (a plain chat loop), made this taskset's default."""


__all__ = ["FactorioBuildHarness", "FactorioBuildTaskset"]
