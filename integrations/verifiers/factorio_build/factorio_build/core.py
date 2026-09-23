"""Scene subsets, prompts and scoring for the factorio-build environment.

Nothing here imports `verifiers`: both front ends (the native v1 taskset in
`taskset.py` and the v0 `load_environment` in `legacy.py`) call into this
module, so the two score a completion identically.

Everything that matters is factory-sim's own code:

- scenes come from `fsim.scenes.sample`, over the same seed plan as
  `evolve.evaluate.scene_sets` (train seeds below `VAL_OFFSET`, validation
  seeds from `VAL_OFFSET` on, and the frozen FactorioRL holdout stream);
- the prompt is `evolve.mutate.system_prompt(evolve.evaluate.api_reference())`;
- code is pulled out with `evolve.llm.extract_code`, gated by
  `evolve.sandbox.check`, and run by `evolve.evaluate.worker_job`, which calls
  `fsim.program_api.run_episode` once per scene;
- the per-scene results are summarised by `evolve.evaluate.aggregate`.

Untrusted programs run in `evolve.pool.EvalPool` workers (spawned processes,
one `RlEnv` each, a per-job timeout that kills a hung worker). `worker_job`
also stops any program after `evaluate.PROGRAM_TIME_LIMIT_S` per episode, so a
`while True: pass` ends either way. `workers=0` runs in-process: fine for tests
and trusted code, but with no process isolation a crash or a hang outside
Python bytecode takes the caller down with it.
"""

from __future__ import annotations

import atexit
import math
import threading
from dataclasses import dataclass

from evolve import evaluate, mutate, sandbox
from evolve.llm import extract_code
from fsim import scenes

#: Tasks with a prompt, a contract and pinned scene sets. `evolve.mutate.TASK`
#: describes construct_smelting_line only, so that is the only one exposed.
SUPPORTED_TASKS = (evaluate.TASK,)
SPLITS = ("train", "val", "holdout")
MAX_SCENES = 64


@dataclass(frozen=True)
class SceneRef:
    """One scene: `scenes.sample(task, sample_split, seed)` gives its blueprint."""

    family: str
    seed: int
    sample_split: str  # "train" (train and val) or "test" (holdout)


def _check_task(task: str) -> None:
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"task must be one of {SUPPORTED_TASKS}, got {task!r}")


def scene_block(task: str, split: str, start: int, n: int) -> list[SceneRef]:
    """Scenes `start .. start+n-1` of a split, in the seed plan `scene_sets` uses."""
    _check_task(task)
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


def blueprints(task: str, refs: list[SceneRef]) -> list[tuple[str, int, dict]]:
    """(family, seed, blueprint) triples, the shape `worker_job` takes."""
    out = []
    for ref in refs:
        family, bp = scenes.sample(task, ref.sample_split, ref.seed)
        if family != ref.family:
            raise ValueError(f"seed {ref.seed} draws {family}, not {ref.family}")
        out.append((family, ref.seed, bp))
    return out


def system_prompt(game_notes: bool = True) -> str:
    """Task, program contract, `world` API reference, optionally the game notes."""
    text = mutate.system_prompt(evaluate.api_reference())
    if not game_notes and mutate.GAME_NOTES:
        text = text.replace("\n\n" + mutate.GAME_NOTES, "")
    return text


def user_message(task: str, split: str, subset_id: str, refs: list[SceneRef]) -> str:
    seeds = f"{refs[0].seed}..{refs[-1].seed}" if split != "holdout" else "held-out stream"
    return (
        f"Write a builder program for {task}.\n"
        f"It will be run once on each of the {len(refs)} scenes of scene subset "
        f"{subset_id} (the {split} set, seeds {seeds}). The program never sees the seed; "
        "scenes differ in where the ore patch is, where the character starts and what "
        "stands in the way. Each scene is scored on its own, and your score is the "
        "fraction of scenes where the smelting line verifies.\n\n" + mutate.OUTPUT_FORMAT
    )


def rows(
    task: str = evaluate.TASK,
    split: str = "train",
    n_scenes: int = 16,
    num_examples: int = 64,
    seed: int = 0,
    game_notes: bool = True,
) -> list[dict]:
    """One dict per dataset row: prompts plus the scene subset it is scored on.

    Row k covers the split's scenes `seed + k*n_scenes .. seed + (k+1)*n_scenes - 1`.
    The holdout's first 100 indices are the frozen FactorioRL holdout; indices past
    it continue the same seed stream (unseen, but not what FactorioRL measured)."""
    if not 1 <= n_scenes <= MAX_SCENES:
        raise ValueError(f"n_scenes must be in 1..{MAX_SCENES}")
    if num_examples < 1:
        raise ValueError("num_examples must be >= 1")
    system = system_prompt(game_notes)
    out = []
    for k in range(num_examples):
        start = seed + k * n_scenes
        refs = scene_block(task, split, start, n_scenes)
        subset_id = f"{task}/{split}/{start}-{start + n_scenes - 1}"
        out.append(
            {
                "task": task,
                "split": split,
                "subset_id": subset_id,
                "scenes": [(r.family, r.seed, r.sample_split) for r in refs],
                "system_prompt": system,
                "prompt": user_message(task, split, subset_id, refs),
            }
        )
    return out


# ------------------------------------------------------------------ running

_POOLS: dict[tuple[int, float], object] = {}
_POOL_LOCK = threading.Lock()


def _pool(workers: int, timeout_s: float):
    from evolve.pool import EvalPool

    key = (workers, float(timeout_s))
    with _POOL_LOCK:
        pool = _POOLS.get(key)
        if pool is None:
            pool = EvalPool(
                workers,
                initializer="evolve.evaluate:worker_init",
                job="evolve.evaluate:worker_job",
                timeout_s=timeout_s,
            )
            _POOLS[key] = pool
        return pool


@atexit.register
def close_pools() -> None:
    with _POOL_LOCK:
        for pool in _POOLS.values():
            pool.close()
        _POOLS.clear()


def run_program(
    source: str,
    triples: list[tuple[str, int, dict]],
    *,
    workers: int = 4,
    timeout_s: float = 30.0,
    decision_budget: int = 600,
) -> list[dict]:
    """Per-scene result dicts (`worker_job`'s shape). Blocking: call it off the event loop."""
    if workers <= 0:
        res = evaluate.worker_job(
            {"source": source, "scenes": triples, "decision_budget": decision_budget}
        )
        if res.get("error"):
            return [evaluate._failed(f, s, res["error"]) for f, s, _ in triples]
        return res["results"]
    size = max(1, math.ceil(len(triples) / workers))
    parts = [triples[k : k + size] for k in range(0, len(triples), size)]
    payloads = [{"source": source, "scenes": p, "decision_budget": decision_budget} for p in parts]
    pool = _pool(workers, timeout_s)
    with _POOL_LOCK:  # EvalPool.map is not re-entrant; one candidate at a time per process
        answers = pool.map(payloads)
    out: list[dict] = []
    for part, res in zip(parts, answers, strict=True):
        if res.get("error") or "results" not in res:
            err = res.get("error") or "no results"
            out.extend(evaluate._failed(f, s, err) for f, s, _ in part)
        else:
            out.extend(res["results"])
    return out


# ------------------------------------------------------------------ scoring

NO_CODE = "no ```python block defining `def build(world):` found"


def score_completion(
    text: str | None,
    task: str,
    scene_list: list,
    *,
    workers: int = 4,
    timeout_s: float = 30.0,
    decision_budget: int = 600,
) -> dict:
    """Extract, check and run the program in `text` on `scene_list`.

    Returns floats under the metric names (success, success_macro, has_program,
    sandbox_valid, refusal_rate, n_scenes, program_errors, mean_decisions,
    mean_verified_output) plus "error" and "program_hash" (str or None)."""
    refs = [SceneRef(*s) if not isinstance(s, SceneRef) else s for s in scene_list]
    out = {
        "success": 0.0,
        "success_macro": 0.0,
        "has_program": 0.0,
        "sandbox_valid": 0.0,
        "refusal_rate": 0.0,
        "n_scenes": float(len(refs)),
        "program_errors": 0.0,
        "mean_decisions": 0.0,
        "mean_verified_output": 0.0,
        "error": None,
        "program_hash": None,
    }
    code = extract_code(text)
    if code is None:
        out["error"] = NO_CODE
        return out
    out["has_program"] = 1.0
    try:
        sandbox.check(code)
    except sandbox.SandboxError as e:
        out["error"] = f"sandbox: {e}"
        return out
    out["sandbox_valid"] = 1.0
    out["program_hash"] = sandbox.normalized_hash(code)
    results = run_program(
        code,
        blueprints(task, refs),
        workers=workers,
        timeout_s=timeout_s,
        decision_budget=decision_budget,
    )
    agg = evaluate.aggregate(results, traces=False)
    n = max(1, len(results))
    decisions = sum(r["decisions"] for r in results)
    refusals = sum(r["refusals"] for r in results)
    out.update(
        success=float(agg["success"]),
        success_macro=float(agg["mean"]),
        refusal_rate=refusals / max(1, decisions + refusals),
        program_errors=float(agg["errors"]),
        mean_decisions=decisions / n,
        mean_verified_output=sum(r["verified_output"] for r in results) / n,
        error=agg["first_error"],
    )
    return out


def format_score(metrics: dict) -> float:
    """+1 for a sandbox-valid program, -1 for a program the sandbox refuses, 0 for none."""
    if metrics.get("sandbox_valid"):
        return 1.0
    return -1.0 if metrics.get("has_program") else 0.0
