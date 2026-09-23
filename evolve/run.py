"""The evolution loop: prompt a model with good programs, keep the better children.

A batch is `concurrency` prompts. Each picks an operator by weight, the next
island in turn and its parents by that island's tournament, then asks the model
for a changed program. Every completion becomes one row in the genealogy, even a
failed one, so a run records what the model wrote and why it went nowhere:

- `extract_failed`: the reply had no python block defining `build`;
- `sandbox_error`: the program broke the contract (the code and the reason are kept);
- a duplicate of a program already stored is not stored again, only counted;
- anything else is evaluated on the train and validation splits, stored under
  the operator that made it, and offered to its island.

The model is the slow part (a minute a completion against about a second to
score one), so the two overlap: batch i+1 is already with the model while batch
i is being scored. Batch i+1's parents are therefore chosen before batch i's
children are admitted. Every choice is drawn on the main thread at a fixed point
in that sequence, from generators seeded by `--seed`, so a run's choices are
reproducible up to what the model writes.

`state.json` (islands, draw positions, counters) and `status.json` (a summary
for a human or a dashboard) are rewritten atomically after every batch. A run
whose directory already holds a genealogy picks up where it stopped: from
`state.json` when it is there, else by replaying the genealogy.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from evolve import mutate, sandbox
from evolve.archive import FACTORY_SIM, Candidate, Islands, Store, write_manifest
from evolve.llm import extract_code
from evolve.seeds.builder import SOURCE

log = logging.getLogger("evolve.run")

DEFAULT_OPERATORS = {"fix": 0.4, "rewrite": 0.3, "crossover": 0.2, "simplify": 0.1}
#: Operators of rows that hold no runnable program; they never join an island.
FAILED = ("extract_failed", "sandbox_error", "eval_error")
STATE_VERSION = 1


def parse_operators(text: str) -> dict[str, float]:
    """`"fix:0.4,rewrite:0.3"` -> weights, checked against `mutate.OPERATORS`."""
    out: dict[str, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        name, sep, weight = part.partition(":")
        name = name.strip()
        if name not in mutate.OPERATORS:
            raise ValueError(f"unknown operator {name!r}; known: {', '.join(mutate.OPERATORS)}")
        w = float(weight) if sep else 1.0
        if w < 0:
            raise ValueError(f"operator weight must be non-negative: {part!r}")
        out[name] = w
    if not out or sum(out.values()) <= 0:
        raise ValueError("need at least one operator with a positive weight")
    return out


@dataclass
class Config:
    name: str
    concurrency: int = 16
    islands: int = 4
    island_size: int = 12
    budget_candidates: int | None = None
    hours: float | None = None
    operators: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_OPERATORS))
    max_tokens: int = 16000
    game_notes: bool = True
    seed: int = 0
    migrate_every: int = 10


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _rates(value) -> dict:
    """Family rates, whether given directly or as `Evaluator.score`'s `{"rates": ...}`."""
    if isinstance(value, dict) and isinstance(value.get("rates"), dict):
        value = value["rates"]
    return {k: float(v) for k, v in (value or {}).items()} if isinstance(value, dict) else {}


def _mean(rates: dict) -> float:
    return sum(rates.values()) / len(rates) if rates else 0.0


def scores_from(result: dict) -> tuple[dict, dict | None]:
    """An `Evaluator.full` result as a Candidate's `scores` and `descriptors`.

    The traces ride in `scores["traces"]`, so a resumed run can still show a
    parent's failures to the model."""
    train, val = _rates(result.get("train")), _rates(result.get("val"))
    traces = result.get("traces")
    if traces is None and isinstance(result.get("train"), dict):
        traces = result["train"].get("traces")
    scores = {
        "train": train,
        "val": val,
        "train_mean": float(result.get("train_mean", _mean(train)) or 0.0),
        "val_mean": float(result.get("val_mean", _mean(val)) or 0.0),
        "traces": {k: [str(x) for x in v] for k, v in (traces or {}).items()},
    }
    if result.get("error"):
        scores["error"] = str(result["error"])
    return scores, result.get("descriptors")


def _failed_scores(error: str, prompt_operator: str | None) -> dict:
    s = {"train": {}, "val": {}, "train_mean": 0.0, "val_mean": 0.0, "error": error}
    if prompt_operator:
        s["prompt_operator"] = prompt_operator
    return s


def parent_dict(c: Candidate) -> dict:
    """A stored candidate in the form `mutate.prompt_*` reads."""
    s = c.scores or {}
    return {
        "code": c.code,
        "scores": {"train": s.get("train", {}), "val": s.get("val", {})},
        "traces": s.get("traces", {}),
    }


def load_islands(store: Store, state_path: Path, n: int, size: int, seed: int) -> Islands:
    """The islands to continue with: exact from `state.json`, else a replay.

    A replay admits only programs that ran, and puts the seeds on every
    island, as seeding did."""
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("islands"):
            return Islands.from_state(state["islands"], store)
    isl = Islands(n, size, seed)
    for c in store.all():
        if c.operator in FAILED:
            continue
        if c.operator == "seed":
            for i in range(n):
                isl.admit(replace(c, island=i))
        else:
            isl.admit(c)
    return isl


@dataclass
class _Job:
    operator: str
    island: int
    parents: list[Candidate]
    messages: list[dict]
    prompt_hash: str


def _blank_op() -> dict:
    return {
        "completions": 0,
        "evaluated": 0,
        "ran": 0,
        "improved": 0,
        "duplicates": 0,
        "sandbox_errors": 0,
        "extract_failures": 0,
        "eval_errors": 0,
        "llm_errors": 0,
    }


class Evolution:
    """The loop, with its collaborators passed in so tests can fake them.

    `client` needs `complete_many(batch, max_tokens=...)`; `evaluator` needs
    `full(source) -> dict`. Only the main thread touches the store, the islands
    and the random generators; the one helper thread only talks to the model.
    """

    def __init__(
        self,
        config: Config,
        client,
        evaluator,
        store: Store,
        islands: Islands,
        status_path,
        *,
        api_reference: str = "",
    ):
        self.config = config
        self.client = client
        self.evaluator = evaluator
        self.store = store
        self.islands = islands
        self.status_path = Path(status_path)
        self.state_path = self.status_path.with_name("state.json")
        self.api_reference = api_reference
        self.rng = random.Random(config.seed)
        self.next_island = 0
        self.batches = 0
        self.llm_calls = 0
        self.llm_errors = 0
        self.latency_total = 0.0
        self.latency_n = 0
        self.duplicates = 0
        self.prior_elapsed = 0.0
        self.per_operator: dict[str, dict] = {}
        self.stopped = "running"
        self._t0 = time.monotonic()
        self._count_store()
        self._load_state()

    # --- bookkeeping ---

    def _op(self, name: str) -> dict:
        return self.per_operator.setdefault(name, _blank_op())

    def _count_store(self) -> None:
        """Counts that the genealogy itself records, so a resume keeps them."""
        self.candidates = self.evaluated = self.ran = self.improved = 0
        self.sandbox_errors = self.extract_failures = self.eval_errors = 0
        rows = self.store.all()
        by_id = {c.id: c for c in rows}
        self.seeds = 0
        for c in rows:
            self.candidates += 1
            if c.operator == "seed":
                self.seeds += 1
                continue
            self._tally(c, [by_id[p] for p in c.parents if p in by_id], restore=True)

    def _tally(self, c: Candidate, parents: list[Candidate], *, restore: bool = False) -> bool:
        """Count one stored candidate. True if it beat its best parent."""
        s = c.scores or {}
        op = self._op(s.get("prompt_operator") or c.operator)
        if not restore:
            self.candidates += 1
        op["completions"] += 1
        if c.operator == "extract_failed":
            self.extract_failures += 1
            op["extract_failures"] += 1
            return False
        if c.operator == "sandbox_error":
            self.sandbox_errors += 1
            op["sandbox_errors"] += 1
            return False
        if c.operator == "eval_error":
            self.eval_errors += 1
            op["eval_errors"] += 1
            return False
        self.evaluated += 1
        op["evaluated"] += 1
        if not s.get("error"):
            self.ran += 1
            op["ran"] += 1
        best_parent = max((p.score() for p in parents), default=float("-inf"))
        better = c.score() > best_parent
        if better:
            self.improved += 1
            op["improved"] += 1
        return better

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        version, internal, gauss = state["rng"]
        self.rng.setstate((version, tuple(internal), gauss))
        self.next_island = int(state.get("next_island", 0))
        self.batches = int(state.get("batches", 0))
        c = state.get("counters", {})
        self.llm_calls = int(c.get("llm_calls", 0))
        self.llm_errors = int(c.get("llm_errors", 0))
        self.latency_total = float(c.get("latency_total", 0.0))
        self.latency_n = int(c.get("latency_n", 0))
        self.duplicates = int(c.get("duplicates", 0))
        self.prior_elapsed = float(c.get("elapsed_s", 0.0))
        for name, extra in c.get("per_operator_extra", {}).items():
            op = self._op(name)
            op["duplicates"] = int(extra.get("duplicates", 0))
            op["llm_errors"] = int(extra.get("llm_errors", 0))
            op["completions"] += op["duplicates"]

    def elapsed(self) -> float:
        return self.prior_elapsed + (time.monotonic() - self._t0)

    def _best(self) -> Candidate | None:
        for c in self.store.best(n=50):
            if c.operator not in FAILED:
                return c
        return None

    def status(self) -> dict:
        elapsed = self.elapsed()
        best = self._best()
        completions = self.candidates - self.seeds + self.duplicates
        per_op = {}
        for name in sorted(self.per_operator):
            o = dict(self.per_operator[name])
            o["improvement_rate"] = o["improved"] / o["evaluated"] if o["evaluated"] else None
            per_op[name] = o
        return {
            "name": self.config.name,
            "state": self.stopped,
            "batches": self.batches,
            "best_val_mean": best.score() if best else None,
            "best_id": best.id if best else None,
            "best_length": best.length if best else None,
            "candidates": self.candidates,
            "evaluated": self.evaluated,
            "ran": self.ran,
            "share_ran": self.ran / completions if completions else None,
            "improved": self.improved,
            "share_improved": self.improved / self.evaluated if self.evaluated else None,
            "duplicates": self.duplicates,
            "sandbox_errors": self.sandbox_errors,
            "extract_failures": self.extract_failures,
            "eval_errors": self.eval_errors,
            "llm_calls": self.llm_calls,
            "llm_errors": self.llm_errors,
            "mean_latency_s": self.latency_total / self.latency_n if self.latency_n else None,
            "elapsed_s": round(elapsed, 3),
            "candidates_per_hour": completions / (elapsed / 3600.0) if elapsed > 0 else None,
            "last_batch_at": _now_iso(),
            "per_operator": per_op,
        }

    def save(self) -> None:
        """Write `state.json`, then `status.json`, each atomically."""
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        version, internal, gauss = self.rng.getstate()
        extra = {
            k: {"duplicates": v["duplicates"], "llm_errors": v["llm_errors"]}
            for k, v in self.per_operator.items()
        }
        state = {
            "version": STATE_VERSION,
            "islands": self.islands.state(),
            "rng": [version, list(internal), gauss],
            "next_island": self.next_island,
            "batches": self.batches,
            "counters": {
                "llm_calls": self.llm_calls,
                "llm_errors": self.llm_errors,
                "latency_total": self.latency_total,
                "latency_n": self.latency_n,
                "duplicates": self.duplicates,
                "elapsed_s": self.elapsed(),
                "per_operator_extra": extra,
            },
        }
        _atomic_json(self.state_path, state)
        _atomic_json(self.status_path, self.status())

    # --- seeding ---

    def seed(self) -> Candidate | None:
        """Evaluate `SOURCE` and put it on every island, unless the run has begun."""
        if self.store.count() > 0:
            log.info("resuming: %d candidates in the genealogy", self.store.count())
            return None
        t0 = time.perf_counter()
        scores, desc = scores_from(self.evaluator.full(SOURCE))
        c = Candidate.new(
            SOURCE,
            code_hash=sandbox.normalized_hash(SOURCE),
            operator="seed",
            island=0,
            scores=scores,
            descriptors=desc,
        )
        self.store.add(c)
        self.candidates += 1
        self.seeds += 1
        for i in range(self.islands.n):
            self.islands.admit(replace(c, island=i))
        log.info(
            "seed %s val_mean=%.3f train_mean=%.3f in %.1fs",
            c.id,
            scores["val_mean"],
            scores["train_mean"],
            time.perf_counter() - t0,
        )
        if scores.get("error"):
            log.warning("the seed program reported an evaluation error")
        return c

    # --- prompts ---

    def _pick_operator(self) -> str:
        names = list(self.config.operators)
        weights = [self.config.operators[n] for n in names]
        return self.rng.choices(names, weights=weights, k=1)[0]

    def build_batch(self, n: int) -> list[_Job]:
        jobs = []
        for _ in range(n):
            op = self._pick_operator()
            island = self.next_island % self.islands.n
            self.next_island = (self.next_island + 1) % self.islands.n
            k = 2 if op == "crossover" else 1
            parents = self.islands.select(island, k=k)
            if not parents:
                continue
            if op == "crossover" and len({p.id for p in parents}) < 2:
                op, parents = "rewrite", parents[:1]  # one member: nothing to cross with
            ref = self.api_reference
            if op == "crossover":
                messages = mutate.prompt_crossover(
                    ref, parent_dict(parents[0]), parent_dict(parents[1])
                )
            elif op == "rewrite":
                messages = mutate.prompt_rewrite(
                    ref, parent_dict(parents[0]), self.rng.choice(mutate.HINTS)
                )
            else:
                messages = mutate.OPERATORS[op](ref, parent_dict(parents[0]))
            jobs.append(_Job(op, island, parents, messages, mutate.prompt_hash(messages)))
        return jobs

    # --- one batch ---

    def _complete(self, jobs: list[_Job]) -> list:
        return self.client.complete_many(
            [j.messages for j in jobs], max_tokens=self.config.max_tokens
        )

    def _store_failure(self, job: _Job, operator: str, code: str, error: str, model) -> Candidate:
        c = Candidate.new(
            code,
            parents=[p.id for p in job.parents],
            operator=operator,
            island=job.island,
            prompt_hash=job.prompt_hash,
            model=model,
            scores=_failed_scores(error, job.operator),
        )
        self.store.add(c)
        self._tally(c, job.parents)
        return c

    def process(self, jobs: list[_Job], results: list) -> None:
        """Turn one batch of completions into stored candidates."""
        t0 = time.perf_counter()
        for job, res in zip(jobs, results, strict=True):
            self.llm_calls += 1
            if isinstance(res, BaseException):
                self.llm_errors += 1
                self._op(job.operator)["llm_errors"] += 1
                log.warning("llm error (%s) for %s prompt", type(res).__name__, job.operator)
                continue
            latency = getattr(res, "latency_s", None)
            if latency is not None:
                self.latency_total += float(latency)
                self.latency_n += 1
            model = getattr(res, "model", None) or None
            code = extract_code(getattr(res, "text", None))
            if code is None:
                self._store_failure(
                    job, "extract_failed", "", "no python block defines build", model
                )
                continue
            try:
                sandbox.check(code)
            except sandbox.SandboxError as e:
                self._store_failure(job, "sandbox_error", code, f"sandbox: {e}", model)
                continue
            h = sandbox.normalized_hash(code)
            if self.store.has_hash(h):
                self.duplicates += 1
                op = self._op(job.operator)
                op["duplicates"] += 1
                op["completions"] += 1
                continue
            try:
                result = self.evaluator.full(code)
            except Exception as e:
                c = Candidate.new(
                    code,
                    code_hash=h,
                    parents=[p.id for p in job.parents],
                    operator="eval_error",
                    island=job.island,
                    prompt_hash=job.prompt_hash,
                    model=model,
                    scores=_failed_scores(f"{type(e).__name__}: {e}", job.operator),
                )
                self.store.add(c)
                self._tally(c, job.parents)
                log.warning("evaluator raised %s", type(e).__name__)
                continue
            scores, desc = scores_from(result)
            c = Candidate.new(
                code,
                code_hash=h,
                parents=[p.id for p in job.parents],
                operator=job.operator,
                island=job.island,
                prompt_hash=job.prompt_hash,
                model=model,
                scores=scores,
                descriptors=desc,
            )
            self.store.add(c)
            better = self._tally(c, job.parents)
            kept = self.islands.admit(c)
            log.info(
                "candidate %s op=%s island=%d parents=%s val_mean=%.3f improved=%s kept=%s",
                c.id,
                job.operator,
                job.island,
                ",".join(p.id for p in job.parents),
                c.score(),
                better,
                kept,
            )
        log.debug("processed %d completions in %.1fs", len(jobs), time.perf_counter() - t0)

    # --- the loop ---

    def _batch_size(self, submitted: int) -> int:
        n = self.config.concurrency
        if self.config.budget_candidates is not None:
            n = min(n, self.config.budget_candidates - submitted)
        if self.config.hours is not None and self.elapsed() >= self.config.hours * 3600.0:
            n = 0
        return max(0, n)

    def run(self) -> dict:
        """Seed if needed, then run batches until the budget or the clock runs out.

        A KeyboardInterrupt (SIGINT) ends the loop after saving what is known;
        the batch that was with the model is dropped, and nothing half-written
        reaches the store, because each candidate is one committed row."""
        submitted = 0
        pending = None
        pending_jobs: list[_Job] = []
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="evolve-llm")
        try:
            self.seed()
            self.save()
            n = self._batch_size(submitted)
            if n:
                pending_jobs = self.build_batch(n)
                submitted += n
                pending = executor.submit(self._complete, pending_jobs)
            while pending is not None:
                t_wait = time.perf_counter()
                results = pending.result()
                jobs, pending, pending_jobs = pending_jobs, None, []
                waited = time.perf_counter() - t_wait
                n = self._batch_size(submitted)
                if n:
                    pending_jobs = self.build_batch(n)
                    submitted += n
                    pending = executor.submit(self._complete, pending_jobs)
                t_eval = time.perf_counter()
                self.process(jobs, results)
                self.batches += 1
                if self.config.migrate_every and self.batches % self.config.migrate_every == 0:
                    self.islands.migrate()
                self.save()
                st = self.status()
                log.info(
                    "batch %d: %d completions, waited %.1fs for the model, scored in %.1fs; "
                    "candidates=%d duplicates=%d best=%s (%.3f)",
                    self.batches,
                    len(jobs),
                    waited,
                    time.perf_counter() - t_eval,
                    st["candidates"],
                    st["duplicates"],
                    st["best_id"],
                    st["best_val_mean"] if st["best_val_mean"] is not None else float("nan"),
                )
            self.stopped = "done"
        except KeyboardInterrupt:
            self.stopped = "interrupted"
            log.warning("interrupted: saving status and stopping")
        finally:
            if pending is not None:
                pending.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            if self.stopped == "running":
                self.stopped = "failed"
            self.save()
        return self.status()


# --- command line ---


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m evolve.run", description=__doc__.split("\n")[0])
    p.add_argument("--name", required=True)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument(
        "--concurrency",
        type=int,
        default=256,
        help="LLM requests in flight. The ceiling is the token-per-minute limit, which a "
        "request charges at its reserved max-tokens: at 2M TPM and ~12k per request that "
        "is ~165 requests/min, ~250 in flight at ~100 s each",
    )
    p.add_argument("--islands", type=int, default=4)
    p.add_argument("--island-size", type=int, default=12)
    p.add_argument("--budget-candidates", type=int, default=None)
    p.add_argument("--hours", type=float, default=None)
    p.add_argument("--config", default=None, help="provider config (default: $FSIM_LLM_CONFIG)")
    p.add_argument(
        "--operators",
        default=",".join(f"{k}:{v}" for k, v in DEFAULT_OPERATORS.items()),
        type=parse_operators,
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=8000,
        help="completion cap, reasoning included. Rate limits reserve it up front, so it "
        "sets how many requests fit in a minute; measured outputs ran 1.7k-4.2k tokens",
    )
    p.add_argument("--no-game-notes", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--migrate-every", type=int, default=10, help="batches between migrations")
    p.add_argument("--job-timeout", type=float, default=300.0, help="seconds per pool job")
    p.add_argument("--runs-dir", default=str(FACTORY_SIM / "runs"))
    p.add_argument("--dry-run", action="store_true")
    return p


def config_from_args(args) -> Config:
    return Config(
        name=args.name,
        concurrency=args.concurrency,
        islands=args.islands,
        island_size=args.island_size,
        budget_candidates=args.budget_candidates,
        hours=args.hours,
        operators=dict(args.operators),
        max_tokens=args.max_tokens,
        game_notes=not args.no_game_notes,
        seed=args.seed,
        migrate_every=args.migrate_every,
    )


def dry_run(config: Config, run_dir: Path, api_reference: str, out=None) -> list[_Job]:
    """Build one batch and print who it would breed, never what it would say.

    Reads an existing run without writing to it. A new run gets an unscored,
    unsaved seed on every island, so nothing is evaluated and no model is called."""
    db = run_dir / "genealogy.sqlite"
    if db.exists():
        store = Store(db, readonly=True)
        islands = load_islands(
            store, run_dir / "state.json", config.islands, config.island_size, config.seed
        )
        store.close()
    else:
        islands = Islands(config.islands, config.island_size, config.seed)
        seed = Candidate.new(SOURCE, operator="seed")
        for i in range(islands.n):
            islands.admit(replace(seed, island=i))

    evo = Evolution(
        config,
        None,
        None,
        Store(":memory:"),
        islands,
        run_dir / "status.json",
        api_reference=api_reference,
    )
    jobs = evo.build_batch(config.concurrency)
    evo.store.close()
    out = out or sys.stdout
    for j in jobs:
        print(
            f"{j.operator:<9} island={j.island} parents={','.join(p.id for p in j.parents)} "
            f"prompt={j.prompt_hash[:12]}",
            file=out,
        )
    return jobs


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = config_from_args(args)
    if not config.game_notes:
        mutate.GAME_NOTES = ""
    run_dir = Path(args.runs_dir) / f"evolve-{args.name}"

    from evolve import evaluate
    from evolve.llm import LLMClient, load_provider
    from evolve.pool import EvalPool

    if args.dry_run:
        dry_run(config, run_dir, evaluate.api_reference())
        return 0

    provider = load_provider(args.config)
    client = LLMClient(provider, concurrency=args.concurrency)
    sets = evaluate.scene_sets()
    digests = {
        split: [evaluate.scene_digest(s[2]) for s in scenes] for split, scenes in sets.items()
    }
    resumed = (run_dir / "genealogy.sqlite").exists()
    write_manifest(
        run_dir,
        name=args.name,
        model=provider.model,
        evaluator_version=evaluate.EVALUATOR_VERSION,
        scene_digests=digests,
        scene_set_digests=evaluate.set_digests(sets),
        game_notes=config.game_notes,
        args={k: v for k, v in vars(args).items() if k != "config"},
        resumed=resumed,
    )
    store = Store(run_dir / "genealogy.sqlite")
    try:
        islands = load_islands(
            store, run_dir / "state.json", config.islands, config.island_size, config.seed
        )
        with EvalPool(
            args.workers,
            initializer="evolve.evaluate:worker_init",
            job="evolve.evaluate:worker_job",
            timeout_s=args.job_timeout,
        ) as pool:
            evaluator = evaluate.Evaluator(pool, sets)
            evo = Evolution(
                config,
                client,
                evaluator,
                store,
                islands,
                run_dir / "status.json",
                api_reference=evaluate.api_reference(),
            )
            status = evo.run()
    finally:
        store.close()
    log.info(
        "%s: %d candidates, best %s (%s)",
        status["state"],
        status["candidates"],
        status["best_id"],
        status["best_val_mean"],
    )
    return 130 if status["state"] == "interrupted" else 0


if __name__ == "__main__":
    sys.exit(main())
