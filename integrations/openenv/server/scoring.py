"""factory-sim glue: scene selection, program scoring and tool-mode sessions.

Nothing here reimplements the simulator or the scorer. Programs are checked by
`evolve.sandbox`, run by `evolve.evaluate.worker_job` inside an
`evolve.pool.EvalPool` (so a program that hangs or crashes costs a worker, not
the server), and summarised by `evolve.evaluate.Evaluator`. Tool mode drives a
`fsim.program_api.World` directly and scores the episode with
`fsim.program_api.run_episode`.

Splits:
- "train": scene seeds drawn from `[0, VAL_OFFSET)` of `scenes.sample(task, "train", .)`.
- "val": the pinned validation set of `evolve.evaluate.scene_sets()`.
- "holdout": FactorioRL's frozen holdout (`obstructed_patch`). Scoring on it
  never returns a failure trace, and only the holdout server exposes it.
"""

from __future__ import annotations

import atexit
import dataclasses
import inspect
import json
import os
import random
import sys
import threading
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
try:  # an installed factory-sim; otherwise the checkout this file sits in
    import fsim  # noqa: F401
except ImportError:
    if (REPO_ROOT / "fsim").is_dir():
        sys.path.append(str(REPO_ROOT))

from evolve import evaluate, mutate, sandbox  # noqa: E402
from evolve.llm import extract_code  # noqa: E402
from evolve.pool import EvalPool  # noqa: E402
from fsim import lib, scenes  # noqa: E402
from fsim.program_api import BudgetExhausted, World, run_episode  # noqa: E402
from fsim.rl import RlEnv  # noqa: E402

TASK = evaluate.TASK
DECISION_BUDGET = 600
DEFAULT_SCENES = {"train": 16, "val": 16, "holdout": 100}
TOOL_ACTIONS = ("move", "place", "give", "take", "mine", "wait")
TRAIN_SPLITS = ("train", "val")
HOLDOUT_SPLITS = ("holdout",)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


# ------------------------------------------------------------------ prompts


@lru_cache(maxsize=1)
def api_reference() -> str:
    return evaluate.api_reference()


def program_prompt() -> str:
    """The evolution loop's system prompt: task, contract, World API, game notes."""
    return mutate.system_prompt(api_reference())


TOOL_INSTRUCTIONS = """\
Tool mode: you act one decision at a time. Each turn, call exactly one world \
action: move(direction, stride), place(item, x, y, facing), give(entity, item, amount), \
take(entity, item, amount), mine(entity) or wait(). Pass an entity as its integer `row` \
from the latest observation's `entities` (rows renumber as the character moves). The \
observation after every call is the full published world view, so there are no query \
calls. Call finish() when the line is built: the rest of the episode then runs without \
you, and the reward is 1 if at least 10 plates are verified, else 0. A call that returns \
false was refused and cost no decision."""


def tool_prompt() -> str:
    parts = [mutate.TASK, "API reference:\n" + api_reference(), TOOL_INSTRUCTIONS]
    if mutate.GAME_NOTES:
        parts.insert(2, mutate.GAME_NOTES)
    return "\n\n".join(parts)


# ------------------------------------------------------------------ scenes


@lru_cache(maxsize=1)
def _pinned() -> dict:
    return evaluate.scene_sets()


def select_scenes(split: str, n: int, seed: int) -> list[tuple[str, int, dict]]:
    """(family, scene seed, blueprint) for `n` scenes of `split`, deterministic in `seed`."""
    rng = random.Random(seed)
    if split == "train":
        out = []
        for s in rng.sample(range(evaluate.VAL_OFFSET), n):
            family, blueprint = scenes.sample(TASK, "train", s)
            out.append((family, s, blueprint))
        return out
    if split in ("val", "holdout"):
        pool = _pinned()[split]
        if n > len(pool):
            raise ValueError(f"split {split!r} has {len(pool)} scenes; asked for {n}")
        return list(pool[:n]) if n == len(pool) else rng.sample(pool, n)
    raise ValueError(f"unknown split {split!r}")


# ------------------------------------------------------------------ program mode

_POOL: EvalPool | None = None
_POOL_LOCK = threading.Lock()


def _pool() -> EvalPool:
    global _POOL
    if _POOL is None or _POOL._closed:
        _POOL = EvalPool(
            workers=max(1, _env_int("FSIM_OPENENV_WORKERS", min(4, os.cpu_count() or 1))),
            initializer="evolve.evaluate:worker_init",
            job="evolve.evaluate:worker_job",
            timeout_s=_env_float("FSIM_OPENENV_JOB_TIMEOUT_S", 30.0),
        )
    return _POOL


def close_pool() -> None:
    global _POOL
    with _POOL_LOCK:
        if _POOL is not None:
            _POOL.close()
            _POOL = None


atexit.register(close_pool)


def source_from(text: str) -> tuple[str | None, str | None]:
    """(source, error). Fenced completions are unwrapped; bare source passes through."""
    if not isinstance(text, str) or not text.strip():
        return None, "empty program"
    if "```" in text:
        code = extract_code(text)
        if code is None:
            return None, "no ```python block defining `def build(world)` found"
        return code, None
    return text, None


def score_program(text: str, drawn: list, split: str) -> dict:
    """Run one program on the episode's scenes and summarise it.

    Returns {"reward", "success", "rates", "mean", "n", "traces", "error"}; the
    reward is the plain per-scene success rate."""
    source, error = source_from(text)
    if error is None:
        try:
            sandbox.check(source)
        except sandbox.SandboxError as e:
            error = f"sandbox: {e}"
    if error is not None:
        return {"reward": 0.0, "success": 0.0, "rates": {}, "mean": 0.0, "n": 0,
                "traces": {}, "error": error}
    with _POOL_LOCK:
        pool = _pool()
        if split == "holdout":
            ev = evaluate.Evaluator(pool, {"holdout": drawn}, decision_budget=DECISION_BUDGET)
            summary = ev.holdout(source)
        else:
            ev = evaluate.Evaluator(pool, {"train": drawn}, decision_budget=DECISION_BUDGET)
            summary = ev.score([source], "train")[0]
    error = summary.get("error")
    if error is None and summary.get("errors"):
        error = f"program raised on {summary['errors']} of {summary['n']} scenes: " + str(
            summary["first_error"]
        )
    traces = {} if split == "holdout" else summary.get("traces", {})
    return {
        "reward": float(summary["success"]),
        "success": float(summary["success"]),
        "rates": {k: float(v) for k, v in summary["rates"].items()},
        "mean": float(summary["mean"]),
        "n": int(summary["n"]),
        "traces": traces,
        "error": error,
    }


def program_feedback(result: dict, split: str) -> str:
    lines = []
    if result["n"]:
        wins = round(result["success"] * result["n"])
        lines.append(
            f"Success: {wins}/{result['n']} {split} scenes ({result['success']:.2f})."
        )
        per = ", ".join(f"{f} {r:.2f}" for f, r in sorted(result["rates"].items()))
        if per:
            lines.append(f"Per family: {per}.")
    else:
        lines.append("Reward 0: the program was not run on any scene.")
    if result["error"]:
        lines.append(f"Error: {result['error']}")
    # The same trace block the evolution loop shows the model it is improving.
    traces = mutate._traces(result["traces"])
    if traces:
        lines.append(traces)
    return "\n".join(lines)


# ------------------------------------------------------------------ tool mode


def _round(v):
    return round(v, 3) if isinstance(v, float) else v


class ToolSession:
    """One scene, driven one `World` call at a time.

    Every call that reaches `World` is recorded. `finish` then replays the
    recording through `run_episode`, the same scorer program mode uses, on a
    fresh reset of the same scene; the simulator is deterministic, so the replay
    is the episode the agent saw, and the check below says so if it ever is not.
    """

    def __init__(self, family: str, seed: int, blueprint: dict):
        self.family, self.seed, self.blueprint = family, seed, blueprint
        self.env = RlEnv()
        self.env.reset(TASK, blueprint, action_space="v2")
        if self.env.rl.task.action_space != lib.ACTION_SPACE_V2:
            raise RuntimeError("the environment did not reset into the v2 action space")
        self.world = World(self.env, DECISION_BUDGET)
        self.calls: list[tuple[str, list, dict]] = []
        self.done = False
        self.outcome: dict | None = None

    # --- view ---

    def view(self) -> dict:
        w = self.world
        patch = w.patch()
        return {
            "me": [_round(v) for v in w.me()],
            "tile": list(w.tile()),
            "patch": None if patch is None else [_round(v) for v in patch],
            "inventory": {k: v for k, v in w.inventory().items() if v},
            "ore_tiles": [list(t) for t in w.ore_tiles("iron-ore")],
            "blocked_tiles": [list(t) for t in w.blocked_tiles()],
            "entities": [
                {k: _round(v) for k, v in dataclasses.asdict(e).items()} for e in w.entities()
            ],
            "decisions_left": w.decisions_left(),
            "last_refused": w.last_refused(),
            "refusals": w.refusals,
            "failures": w.failures,
        }

    # --- actions ---

    def call(self, method: str, args: list, kwargs: dict) -> tuple[bool | None, str | None]:
        """(World's return value, error). An error leaves the world untouched."""
        if self.done:
            return None, "the episode is over; call reset()"
        if method == "finish":
            if args or kwargs:
                return None, "finish() takes no arguments"
            self.finish()
            return None, None
        if method not in TOOL_ACTIONS:
            expected = ", ".join(TOOL_ACTIONS + ("finish",))
            return None, f"unknown method {method!r}; expected one of {expected}"
        fn = getattr(self.world, method)
        try:
            inspect.signature(fn).bind(*args, **kwargs)
        except TypeError as e:
            return None, f"{method}: {e}"
        if method in ("give", "take", "mine"):
            entity = args[0] if args else kwargs.get("entity")
            if not isinstance(entity, int) or isinstance(entity, bool):
                return None, f"{method}: pass the entity as its integer row"
        self.calls.append((method, list(args), dict(kwargs)))
        try:
            ok = fn(*args, **kwargs)
        except BudgetExhausted:
            self.finish()
            return None, None
        except RuntimeError as e:  # MAX_REFUSALS
            self.finish()
            return None, str(e)
        if self.world.decisions_left() <= 0 or self.env.rl.done:
            self.finish()
        return bool(ok), None

    def finish(self) -> dict:
        if self.outcome is not None:
            return self.outcome
        calls = list(self.calls)

        def replay(world):
            for method, args, kwargs in calls:
                getattr(world, method)(*args, **kwargs)

        live_decisions = self.world.decisions
        r = run_episode(replay, self.blueprint, task=TASK, decision_budget=DECISION_BUDGET,
                        env=self.env)
        error = r.error
        diverged = r.decisions != live_decisions
        if diverged:
            error = f"replay diverged: {r.decisions} decisions, {live_decisions} live"
        self.done = True
        self.outcome = {
            "success": bool(r.success) and not diverged,
            "verified_output": int(r.verified_output),
            "decisions": int(r.decisions),
            "refusals": int(r.refusals),
            "error": error,
        }
        return self.outcome


def dumps(view: dict) -> str:
    return json.dumps(view, separators=(",", ":"))
