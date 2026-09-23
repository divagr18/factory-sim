"""Which scenes a task covers: the seed plan `factorio_build.core.scene_block` uses.

Kept here so this package depends on factory-sim alone. The integration test
`test_play_scenes_match_factorio_build` checks that the two pick the same
scenes.
"""

from __future__ import annotations

from dataclasses import dataclass

from evolve import evaluate
from fsim import scenes

SUPPORTED_TASKS = (evaluate.TASK,)
SPLITS = ("train", "val", "holdout")


@dataclass(frozen=True)
class SceneRef:
    """One scene: `scenes.sample(task, sample_split, seed)` gives its blueprint."""

    family: str
    seed: int
    sample_split: str  # "train" (train and val) or "test" (holdout)


def scene_block(task: str, split: str, start: int, n: int) -> list[SceneRef]:
    """Scenes `start .. start+n-1` of a split, in the seed plan `scene_sets` uses."""
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"task must be one of {SUPPORTED_TASKS}, got {task!r}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if start < 0 or n < 1:
        raise ValueError("start must be >= 0 and n >= 1")
    if split == "train":
        if start + n > evaluate.VAL_OFFSET:
            raise ValueError(f"train seeds must stay below {evaluate.VAL_OFFSET}")
        seeds, sample_split = list(range(start, start + n)), "train"
    elif split == "val":
        base = evaluate.VAL_OFFSET + start
        seeds, sample_split = list(range(base, base + n)), "train"
    else:
        base = evaluate.HOLDOUT_START_INDEX + start
        seeds = [evaluate.holdout_seed(base + k) for k in range(n)]
        sample_split = "test"
    return [SceneRef(scenes.sample(task, sample_split, s)[0], s, sample_split) for s in seeds]
