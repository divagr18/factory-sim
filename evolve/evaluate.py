"""Score builder programs on fixed scene sets, in a pool of simulator workers.

An evolution loop is only as honest as the numbers it selects on, so the sets
here are pinned rather than drawn per call. Train and validation scenes are
`scenes.sample(task, "train", seed)` over two disjoint seed ranges: the loop
selects on validation, and a program that memorised the training scenes'
quirks is caught there. The holdout is the frozen FactorioRL one
(`docs/evidence/holdout_v3.json`): its episode seeds are
`blake2b(master|run_id|branch|index)`, and fed to `scenes.sample` they
reproduce every one of its blueprint digests, so "holdout" means the scenes
FactorioRL's policies were measured on and not merely the same family.

The holdout is scored only through `Evaluator.holdout`, and never contributes
a trace. A trace is what the next prompt shows the model; one from a held-out
scene would teach the loop the holdout a failure at a time.

A job is one source and a chunk of scenes, so a worker loads a program once
and runs it many times on the `RlEnv` it built at start. Chunks are small
enough that one candidate's evaluation spreads over every worker.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from evolve import sandbox
from fsim import scenes
from fsim.program_api import Entity, run_episode

# 4: the sandbox accepts lambda, :=, _names and set/str methods (3: train-only traces)
EVALUATOR_VERSION = 4
#: The default task: what every function here evaluates when not told otherwise.
TASK = "construct_smelting_line"
FAMILIES_TRAIN = ("open_patch", "offset_patch", "varied_patch", "cluttered_patch")
FAMILIES_HOLDOUT = ("obstructed_patch",)


@dataclass(frozen=True)
class TaskSetup:
    """What evaluating a builder program on one task needs.

    `families_*` restate FactorioRL's layout families (train, then the test ones
    the holdout draws); `decision_budget` is the task's `max_decision_steps`.
    The action space and `World` follow from `program_api.TASK_PROFILES`.
    """

    name: str
    families_train: tuple[str, ...]
    families_holdout: tuple[str, ...]
    decision_budget: int


TASKS: dict[str, TaskSetup] = {
    TASK: TaskSetup(TASK, FAMILIES_TRAIN, FAMILIES_HOLDOUT, 600),
    # FactorioRL `tasks/families/belt_smelting.py` 1.1.0.
    "belt_smelting": TaskSetup(
        "belt_smelting", ("open", "walled", "split_patch"), ("obstructed", "far_chest"), 2500
    ),
}


def task_setup(task: str) -> TaskSetup:
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; known: {', '.join(TASKS)}")
    return TASKS[task]


def scenes_ported(task: str) -> bool:
    """Whether `fsim.scenes` can draw this task's scenes (and the simulator run it)."""
    from fsim.rl import TASKS as SIM_TASKS

    return task in scenes.GENERATORS and task in SIM_TASKS


def require_scenes(task: str) -> TaskSetup:
    """The task's setup, or an error naming what is missing for it to run here."""
    setup = task_setup(task)
    if not scenes_ported(task):
        raise NotImplementedError(
            f"{task}: its scene generator is not ported to fsim.scenes (and the simulator "
            "has no task for it) yet, so there are no scenes to evaluate on"
        )
    return setup


VAL_OFFSET = 100_000
#: The first train scenes, which every candidate runs before anything else. A
#: program that raises on all of them is not worth the rest of the set.
STAGE1_SCENES = 8
CHUNK = 16

# FactorioRL's frozen holdout seed plan (tools/freeze_holdout.py). Restated
# rather than read at import, so the sets exist without a FactorioRL checkout;
# `verify_holdout` checks them against the frozen file when there is one.
HOLDOUT_MASTER = 20260908
HOLDOUT_RUN_ID = "holdout-v3"
HOLDOUT_BRANCH = "eval"
HOLDOUT_START_INDEX = 3000
HOLDOUT_FROZEN = 100
HOLDOUT_FILE = Path(
    os.environ.get(
        "FSIM_HOLDOUT_FILE",
        Path(__file__).resolve().parents[2]
        / "FactorioRL"
        / "docs"
        / "evidence"
        / "holdout_v3.json",
    )
)

SHORT = {
    "burner-mining-drill": "D",
    "stone-furnace": "F",
    "transport-belt": "B",
    "burner-inserter": "I",
    "wooden-chest": "C",
}


# ------------------------------------------------------------------ scene sets


def holdout_seed(index: int) -> int:
    """`factoriorl.seeding.SeedPlan.episode_seed` for the holdout's plan."""
    payload = f"{HOLDOUT_MASTER}|{HOLDOUT_RUN_ID}|{HOLDOUT_BRANCH}|{index}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def scene_digest(blueprint: dict) -> str:
    """FactorioRL's blueprint digest: sha256 of canonical JSON, 16 hex characters.

    The same canonicalisation (`sort_keys`, tight separators) as
    `generator_diagnostics.scene_digest`, so a digest here can be looked up in
    FactorioRL's frozen files directly."""
    canonical = json.dumps(blueprint, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def scene_sets(
    train_n=128, val_n=256, holdout_n=100, holdout_start=0, task: str = TASK
) -> dict[str, list[tuple[str, int, dict]]]:
    """{"train", "val", "holdout"}: lists of (family, seed, blueprint), deterministic.

    Holdout indices past the frozen 100 continue the same seed stream; they are
    unseen, but no FactorioRL result was measured on them. `holdout_start` skips
    into the stream: a method changed after its holdout results were seen has to
    be reported on scenes no run or analysis has touched. Every task draws its
    holdout from the same frozen seed stream, as FactorioRL freezes them."""
    require_scenes(task)

    def draw(split, seeds):
        return [(*scenes.sample(task, split, s), s) for s in seeds]

    holdout_seeds = [
        holdout_seed(HOLDOUT_START_INDEX + holdout_start + k) for k in range(holdout_n)
    ]
    sets = {
        name: [(family, seed, bp) for family, bp, seed in drawn]
        for name, drawn in (
            ("train", draw("train", range(train_n))),
            ("val", draw("train", range(VAL_OFFSET, VAL_OFFSET + val_n))),
            ("holdout", draw("test", holdout_seeds)),
        )
    }
    seen = {scene_digest(bp) for _, _, bp in sets["train"]}
    clash = [s for _, s, bp in sets["val"] if scene_digest(bp) in seen]
    assert not clash, f"validation seeds {clash[:5]} repeat a training scene"
    return sets


def set_digests(sets: dict) -> dict[str, str]:
    """One digest per set, over its scenes' digests in order: what a manifest pins."""
    out = {}
    for name, items in sets.items():
        joined = ",".join(scene_digest(bp) for _, _, bp in items)
        out[name] = hashlib.sha256(joined.encode()).hexdigest()[:16]
    return out


def verify_holdout(
    sets: dict, path: Path | str | None = None, start: int = 0, task: str = TASK
) -> dict:
    """Compare the holdout set with FactorioRL's frozen file (read only).

    Returns {"file", "frozen", "compared", "matched", "mismatched": [index]};
    "file" is None when there is no FactorioRL checkout to compare with."""
    path = Path(path) if path is not None else HOLDOUT_FILE
    if not path.is_file():
        return {"file": None, "frozen": 0, "compared": 0, "matched": 0, "mismatched": []}
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    frozen_tasks = doc["holdout"]["tasks"]
    if task not in frozen_tasks:
        return {"file": str(path), "frozen": 0, "compared": 0, "matched": 0, "mismatched": []}
    episodes = frozen_tasks[task]["episodes"]
    frozen = {e["episode_index"]: e for e in episodes}
    matched, mismatched = 0, []
    for k, (family, _, bp) in enumerate(sets["holdout"]):
        entry = frozen.get(HOLDOUT_START_INDEX + start + k)
        if entry is None:
            continue
        if entry["blueprint_digest"] == scene_digest(bp) and entry["layout_family"] == family:
            matched += 1
        else:
            mismatched.append(HOLDOUT_START_INDEX + start + k)
    return {
        "file": str(path),
        "frozen": len(episodes),
        "compared": matched + len(mismatched),
        "matched": matched,
        "mismatched": mismatched,
    }


# ------------------------------------------------------------- API reference

COUNTERS = {"refusals", "failures"}
UNDOCUMENTED = {"decisions_left": "Decisions the program may still spend before it is stopped."}


def _world_class(task: str):
    from fsim.program_api import world_class

    return world_class(task)


def api_methods(task: str = TASK) -> list[str]:
    """World's methods a program may call, in source order (a subclass's own after)."""
    cls = _world_class(task)
    names: list[str] = []
    for klass in reversed(cls.__mro__):
        for n in vars(klass):
            if n in sandbox.WORLD_API and n not in COUNTERS and n not in names:
                names.append(n)
    return [n for n in names if callable(getattr(cls, n))]


def _first_paragraph(doc: str | None) -> str:
    if not doc:
        return ""
    return " ".join(inspect.cleandoc(doc).split("\n\n")[0].split())


def api_reference(task: str = TASK) -> str:
    """The `world` API as the prompt shows it: signatures and one line each."""
    cls = _world_class(task)
    lines = ["world (the only object a program holds):"]
    for name in api_methods(task):
        sig = str(inspect.signature(getattr(cls, name), eval_str=True)).replace("fsim.obsview.", "")
        sig = sig.replace("(self, ", "(").replace("(self)", "()")
        doc = _first_paragraph(getattr(cls, name).__doc__) or UNDOCUMENTED.get(name, "")
        lines.append(f"  world.{name}{sig}" + (f"  # {doc}" if doc else ""))
    lines.append(
        "  world.refusals, world.failures  # int counters: refused intents, failed actions"
    )
    entity = cls._ENTITY
    fields = ", ".join(
        f"{f.name}: {getattr(f.type, '__name__', f.type)}"
        for f in entity.__dataclass_fields__.values()
    )
    lines.append(f"{entity.__name__} (from world.entities()): {fields}")
    if entity is Entity:
        lines.append(
            '  kind is "furnace" | "mining-drill" | "container" | "wall" | "item-entity" | '
            '"other"; facing is N/E/S/W for drills, None otherwise'
        )
    else:
        lines.extend(cls._ENTITY_NOTES)
    return "\n".join(lines)


# ------------------------------------------------------------------ workers

_ENV = None


def worker_init() -> None:
    """Build this worker's one `RlEnv`; every episode it runs resets it."""
    global _ENV
    from fsim.rl import RlEnv

    _ENV = RlEnv()


def _with_facings(build, facings: list):
    """`build`, with `world.place` recording the facing of each placement that
    lands, which `EpisodeResult.built` does not keep. Same signature, same
    outcome: the program cannot tell."""

    def run(world):
        place = world.place

        def recording_place(item, x, y, facing):
            before = len(world._built)
            ok = place(item, x, y, facing)
            if len(world._built) > before:
                facings.append(facing)
            return ok

        world.place = recording_place
        return build(world)

    return run


#: Wall-clock seconds a program may run per episode. A whole episode takes about
#: 7 ms and a program's own work a few more, so this only ever stops a loop that
#: will not end: queries cost no decision, and the decision budget cannot.
PROGRAM_TIME_LIMIT_S = 2.0
#: Time-limit hits in one chunk after which its remaining scenes are scored as
#: failures without being run. A program that loops forever on one scene will
#: on most; without this one candidate held a run for most of an hour.
MAX_TIME_LIMIT_HITS = 2


def worker_job(payload: dict) -> dict:
    """Run one source on a chunk of scenes: {"results": [per-scene dict], "error": None}."""
    global _ENV
    if _ENV is None:
        worker_init()
    try:
        build = sandbox.load(payload["source"])
    except sandbox.SandboxError as e:
        return {"results": [], "error": f"sandbox: {e}"}
    task = payload.get("task", TASK)
    budget = payload.get("decision_budget", task_setup(task).decision_budget)
    limit = payload.get("time_limit_s", PROGRAM_TIME_LIMIT_S)
    results = []
    hits = 0
    for family, seed, blueprint in payload["scenes"]:
        if hits >= MAX_TIME_LIMIT_HITS:
            results.append(
                _failed(
                    family, seed, f"skipped: the program hit its {limit:g} s limit {hits} times"
                )
            )
            continue
        facings: list = []
        try:
            r = run_episode(
                _with_facings(build, facings),
                blueprint,
                task=task,
                decision_budget=budget,
                env=_ENV,
                time_limit_s=limit,
            )
        except Exception as e:  # the harness, not the program; keep the chunk going
            results.append(_failed(family, seed, f"evaluator: {type(e).__name__}: {e}"))
            worker_init()
            continue
        results.append(
            {
                "family": family,
                "seed": seed,
                "success": bool(r.success),
                "verified_output": int(r.verified_output),
                "decisions": int(r.decisions),
                "refusals": int(r.refusals),
                "failures": int(r.failures),
                "first_plate_tick": r.first_plate_tick,
                "walk_distance": float(r.walk_distance),
                "built": [list(b) for b in r.built],
                "facings": facings if len(facings) == len(r.built) else [],
                "trace": list(r.trace),
                "error": r.error,
            }
        )
        if r.error and r.error.startswith("ProgramTimeLimit"):
            hits += 1
    return {"results": results, "error": None}


def _failed(family: str, seed: int, error: str) -> dict:
    return {
        "family": family,
        "seed": seed,
        "success": False,
        "verified_output": 0,
        "decisions": 0,
        "refusals": 0,
        "failures": 0,
        "first_plate_tick": None,
        "walk_distance": 0.0,
        "built": [],
        "facings": [],
        "trace": [],
        "error": error,
    }


# ---------------------------------------------------------------- aggregation


def layout_signature(result: dict) -> str:
    """What one build looks like: each placement relative to the first drill."""
    built = result["built"]
    if not built:
        return ""
    anchor = next((b for b in built if b[0] == "burner-mining-drill"), built[0])
    facings = result.get("facings") or [""] * len(built)
    parts = []
    for (item, x, y), facing in zip(built, facings, strict=True):
        parts.append(
            f"{SHORT.get(item, item[:1].upper())}{x - anchor[1]:+d}{y - anchor[2]:+d}{facing}"
        )
    return " ".join(parts)


def _trace_for(result: dict) -> list[str]:
    lines = list(result["trace"])
    if result["error"]:
        lines.append(f"program raised {result['error']}")
    lines.append(
        f"episode failed: verified_output {result['verified_output']}, "
        f"{result['decisions']} decisions, {result['refusals']} refusals"
    )
    return lines


def aggregate(results: list[dict], families=None, *, traces: bool = True, error=None) -> dict:
    """Per-family success rates and descriptors over one source's scene results.

    `mean` is the macro average over families, so a family's weight does not
    ride on how often the sampler happened to draw it; `success` is the plain
    per-scene rate."""
    families = list(families) if families else sorted({r["family"] for r in results})
    by_family = {f: [r for r in results if r["family"] == f] for f in families}
    rates = {f: (sum(r["success"] for r in rs) / len(rs)) for f, rs in by_family.items() if rs}
    out_traces = {}
    if traces:
        for f, rs in by_family.items():
            failing = next((r for r in rs if not r["success"]), None)
            if failing is not None:
                out_traces[f] = _trace_for(failing)
    winner = next((r for r in results if r["success"]), None)
    n = len(results)
    errors = [r for r in results if r["error"]]
    return {
        "rates": rates,
        "mean": sum(rates.values()) / len(rates) if rates else 0.0,
        "success": sum(r["success"] for r in results) / n if n else 0.0,
        "n": n,
        "traces": out_traces,
        "descriptors": {
            "layout_signature": layout_signature(winner) if winner else "",
            "mean_decisions": sum(r["decisions"] for r in results) / n if n else 0.0,
            "mean_walk": sum(r["walk_distance"] for r in results) / n if n else 0.0,
        },
        "errors": len(errors),
        "first_error": errors[0]["error"] if errors else None,
        "error": error,
    }


def _zero(families, error: str, n: int = 0) -> dict:
    out = aggregate([], families, error=error)
    out["rates"] = {f: 0.0 for f in families}
    out["n"] = n
    return out


class Evaluator:
    """Scores sources on the pinned sets through an `EvalPool` of `worker_job`s."""

    def __init__(
        self,
        pool,
        sets: dict,
        *,
        decision_budget: int | None = None,
        chunk: int = CHUNK,
        task: str = TASK,
    ):
        self.pool = pool
        self.sets = sets
        self.task = task
        self.setup = task_setup(task)
        self.decision_budget = (
            decision_budget if decision_budget is not None else self.setup.decision_budget
        )
        self.chunk = chunk

    # --- running ---

    def _map(self, jobs: list[tuple[int, list]], sources: list[str]) -> dict[int, list[dict]]:
        """jobs: (source index, scenes). Returns source index -> per-scene results."""
        total = sum(len(sc) for _, sc in jobs)
        workers = max(1, getattr(self.pool, "workers", 1))
        size = max(1, min(self.chunk, math.ceil(total / (2 * workers)))) if total else 1
        payloads, owners = [], []
        for i, sc in jobs:
            for k in range(0, len(sc), size):
                part = sc[k : k + size]
                payload = {
                    "source": sources[i],
                    "scenes": part,
                    "decision_budget": self.decision_budget,
                }
                # Only a task other than the default rides in the payload, so a
                # construct_smelting_line job is the dict it always was.
                if self.task != TASK:
                    payload["task"] = self.task
                payloads.append(payload)
                owners.append((i, part))
        out: dict[int, list[dict]] = {i: [] for i, _ in jobs}
        for (i, part), res in zip(owners, self.pool.map(payloads) if payloads else [], strict=True):
            if res.get("error") or "results" not in res:
                err = res.get("error") or "no results"
                out[i].extend(_failed(f, s, err) for f, s, _ in part)
            else:
                out[i].extend(res["results"])
        return out

    def _run(self, sources: list[str], splits: list[str]) -> list[dict]:
        """Per source: {"error": str | None, split: [per-scene results]}."""
        state = [{"error": None} for _ in sources]
        live = []
        for i, src in enumerate(sources):
            try:
                sandbox.check(src)
                live.append(i)
            except sandbox.SandboxError as e:
                state[i]["error"] = f"sandbox: {e}"
        stage1 = self.sets["train"][:STAGE1_SCENES]
        first = self._map([(i, stage1) for i in live], sources)
        survivors = []
        for i in live:
            rs = first[i]
            if rs and all(r["error"] for r in rs):
                state[i]["error"] = (
                    f"stage 1: every one of {len(rs)} scenes failed: {rs[0]['error']}"
                )
            else:
                survivors.append(i)
        jobs = []
        for i in survivors:
            for split in splits:
                sc = self.sets[split]
                jobs.append((i, split, sc[len(stage1) :] if split == "train" else sc))
        results: dict[tuple[int, str], list] = {}
        grouped = [(k, sc) for k, (_, _, sc) in enumerate(jobs)]
        mapped = self._map(grouped, [sources[i] for i, _, _ in jobs])
        for k, (i, split, _) in enumerate(jobs):
            results[(i, split)] = (first[i] if split == "train" else []) + mapped[k]
        for i in range(len(sources)):
            for split in splits:
                state[i][split] = results.get((i, split))
        return state

    def _families(self, split: str):
        setup = self.setup
        return setup.families_holdout if split == "holdout" else setup.families_train

    def _summarise(self, st: dict, split: str) -> dict:
        fams = self._families(split)
        if st.get(split) is None:
            return _zero(fams, st["error"])
        return aggregate(st[split], fams, traces=split != "holdout", error=st["error"])

    # --- public ---

    def score(self, sources: list[str], split: str = "train") -> list[dict]:
        """One summary per source on the train or val set."""
        if split not in ("train", "val"):
            raise ValueError("score() takes 'train' or 'val'; the holdout has holdout()")
        return [self._summarise(st, split) for st in self._run(list(sources), [split])]

    def full(self, source: str) -> dict:
        """Train and val together, in one pass over the pool.

        "train"/"val" are the per-family rates (the shape `Candidate.scores` and
        a prompt's parent dict take); the whole summaries are under "detail"."""
        st = self._run([source], ["train", "val"])[0]
        train, val = self._summarise(st, "train"), self._summarise(st, "val")
        # Training scenes only. Before version 3 a family whose training scenes
        # all passed showed its first failing validation scene instead: exactly
        # when training saturates, the model was handed the validation failures
        # that selection ranks by, and could patch those scenes one by one.
        traces = dict(train["traces"])
        return {
            "train": train["rates"],
            "val": val["rates"],
            "train_mean": train["mean"],
            "val_mean": val["mean"],
            "traces": traces,
            "descriptors": train["descriptors"],
            "error": st["error"],
            "n": {"train": train["n"], "val": val["n"]},
            "detail": {"train": train, "val": val},
        }

    def holdout(self, source: str) -> dict:
        """The frozen holdout. No stage-1 skip beyond the sandbox, and no traces."""
        st = {"error": None}
        try:
            sandbox.check(source)
        except sandbox.SandboxError as e:
            st["error"] = f"sandbox: {e}"
            return self._summarise(st, "holdout")
        st["holdout"] = self._map([(0, self.sets["holdout"])], [source])[0]
        return self._summarise(st, "holdout")
