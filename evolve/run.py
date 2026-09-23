"""The evolution loop: prompt a model with good programs, keep the better children.

The loop is a continuous pipeline. Up to `concurrency` prompts are with the model
at all times. Each prompt picks an operator by weight, the next island in turn
and its parents by that island's tournament, then asks the model for a changed
program. Every completion becomes one row in the genealogy, even a failed one,
so a run records what the model wrote and why it went nowhere:

- `extract_failed`: the reply had no python block defining `build`;
- `sandbox_error`: the program broke the contract (the code and the reason are kept);
- a duplicate of a program already stored is not stored again, only counted;
- anything else is evaluated on the train and validation splits, stored under
  the operator that made it, and offered to its island.

The model is the slow part (a minute or more a completion, with a long tail,
against about a second to score one), so there is no batch barrier: each
completion is scored as soon as it arrives, and its slot is refilled at once
with a prompt drawn from the islands as they are then, so children breed from
recent winners. Completions that arrive while another is being scored wait in an
unbounded queue and are scored one after another; free slots are refilled
between scorings, so the model never waits on the evaluator for longer than one
scoring. A request still running after `--request-timeout` seconds is abandoned
and counted in `llm_timeouts`; any late reply is ignored and its slot is
refilled. Abandoned requests keep their daemon thread until the client's own
socket timeout ends them, and at most `concurrency` of them may be pending
before new prompts wait for them, so threads stay bounded.

Requests run on daemon threads that only talk to the model. Every choice is
drawn on the main thread, from generators seeded by `--seed`, so the sequence of
draws is reproducible; which child a draw can see depends on the order the
completions arrive in.

`state.json` (islands, draw positions, counters) and `status.json` (a summary
for a human or a dashboard) are rewritten atomically at least every 30 s and on
exit. `batches` in the status counts those writes; `status_writes` is the same
number under a clearer name. Islands migrate every `--migrate-every` model
replies. A run whose directory already holds a genealogy picks up where it
stopped: from `state.json` when it is there, else by replaying the genealogy.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import random
import sys
import threading
import time
from collections import deque
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
#: The client's per-attempt socket timeout, never longer than `--request-timeout`.
SOCKET_TIMEOUT_S = 300.0


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
    #: Model replies (not timeouts) between migrations; 0 never migrates.
    migrate_every: int = 200
    #: Seconds before an unanswered request is abandoned and its slot refilled.
    request_timeout: float = 600.0
    #: Seconds between writes of `state.json` and `status.json`.
    status_every_s: float = 30.0


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
        "llm_timeouts": 0,
    }


@dataclass
class _Flight:
    job: _Job
    deadline: float


class Evolution:
    """The loop, with its collaborators passed in so tests can fake them.

    `client` needs `complete(messages, max_tokens=...)` and must be safe to call
    from many threads at once; `evaluator` needs `full(source) -> dict`. Only
    the main thread touches the store, the islands and the random generators;
    the request threads only talk to the model and post what it said.
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
        ledger=None,
    ):
        self.config = config
        #: The dollar cap (`evolve.spend.SpendLedger`), or None for an uncapped run.
        self.ledger = ledger
        self.run_spent_usd = 0.0
        self._worst: dict[int, float] = {}  # worst-case cost held per unanswered request
        self._capped = False
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
        self.llm_timeouts = 0
        self.latency_total = 0.0
        self.latency_n = 0
        self.duplicates = 0
        self.prior_elapsed = 0.0
        self.per_operator: dict[str, dict] = {}
        self.stopped = "running"
        self._t0 = time.monotonic()
        # The pipeline, all touched only by the main thread except `_arrivals`,
        # which request threads put their replies on.
        self._arrivals: queue.SimpleQueue = queue.SimpleQueue()
        self._in_flight: dict[int, _Flight] = {}
        self._abandoned: set[int] = set()
        self._ready: deque = deque()
        self._next_token = 0
        self._issued = 0
        self._no_parents = False
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
        self.llm_timeouts = int(c.get("llm_timeouts", 0))
        self.latency_total = float(c.get("latency_total", 0.0))
        self.latency_n = int(c.get("latency_n", 0))
        self.duplicates = int(c.get("duplicates", 0))
        self.prior_elapsed = float(c.get("elapsed_s", 0.0))
        for name, extra in c.get("per_operator_extra", {}).items():
            op = self._op(name)
            op["duplicates"] = int(extra.get("duplicates", 0))
            op["llm_errors"] = int(extra.get("llm_errors", 0))
            op["llm_timeouts"] = int(extra.get("llm_timeouts", 0))
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
            "status_writes": self.batches,
            "in_flight": len(self._in_flight),
            "abandoned_running": len(self._abandoned),
            "queue_depth": len(self._ready) + self._arrivals.qsize(),
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
            "llm_timeouts": self.llm_timeouts,
            "mean_latency_s": self.latency_total / self.latency_n if self.latency_n else None,
            "elapsed_s": round(elapsed, 3),
            "candidates_per_hour": completions / (elapsed / 3600.0) if elapsed > 0 else None,
            "last_batch_at": _now_iso(),
            "per_operator": per_op,
            # Dollars. `spent_usd` is the ledger total across every run that shares
            # it; `run_spent_usd` is this process's share.
            "spent_usd": round(self.ledger.total, 6) if self.ledger else None,
            "run_spent_usd": round(self.run_spent_usd, 6),
            "max_usd": self.ledger.cap_usd if self.ledger else None,
            "held_usd": round(sum(self._worst.values()), 6),
            "spend_capped": self._capped,
        }

    def save(self) -> None:
        """Write `state.json`, then `status.json`, each atomically; count the write."""
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.batches += 1
        version, internal, gauss = self.rng.getstate()
        extra = {
            k: {
                "duplicates": v["duplicates"],
                "llm_errors": v["llm_errors"],
                "llm_timeouts": v["llm_timeouts"],
            }
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
                "llm_timeouts": self.llm_timeouts,
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

    def build_job(self) -> _Job | None:
        """One prompt: an operator, the next island, its parents. None if it is empty."""
        op = self._pick_operator()
        island = self.next_island % self.islands.n
        self.next_island = (self.next_island + 1) % self.islands.n
        k = 2 if op == "crossover" else 1
        parents = self.islands.select(island, k=k)
        if not parents:
            return None
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
        return _Job(op, island, parents, messages, mutate.prompt_hash(messages))

    def build_batch(self, n: int) -> list[_Job]:
        """`n` draws of `build_job`, skipping empty islands (the dry run's view)."""
        return [j for j in (self.build_job() for _ in range(n)) if j is not None]

    # --- one completion ---

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

    def process_one(self, job: _Job, res) -> None:
        """Turn one reply (a completion or the exception its request raised) into a row."""
        if isinstance(res, BaseException) and not isinstance(res, Exception):
            raise res  # a KeyboardInterrupt or SystemExit raised on a request thread
        self.llm_calls += 1
        if isinstance(res, BaseException):
            self.llm_errors += 1
            self._op(job.operator)["llm_errors"] += 1
            log.warning("llm error (%s) for %s prompt", type(res).__name__, job.operator)
            return
        latency = getattr(res, "latency_s", None)
        if latency is not None:
            self.latency_total += float(latency)
            self.latency_n += 1
        model = getattr(res, "model", None) or None
        code = extract_code(getattr(res, "text", None))
        if code is None:
            self._store_failure(job, "extract_failed", "", "no python block defines build", model)
            return
        try:
            sandbox.check(code)
        except sandbox.SandboxError as e:
            self._store_failure(job, "sandbox_error", code, f"sandbox: {e}", model)
            return
        h = sandbox.normalized_hash(code)
        if self.store.has_hash(h):
            self.duplicates += 1
            op = self._op(job.operator)
            op["duplicates"] += 1
            op["completions"] += 1
            return
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
            return
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

    def process(self, jobs: list[_Job], results: list) -> None:
        """`process_one` over paired jobs and replies."""
        for job, res in zip(jobs, results, strict=True):
            self.process_one(job, res)

    # --- the pipeline ---

    def _want_more(self) -> bool:
        """Whether the budget and the clock allow another request."""
        budget = self.config.budget_candidates
        if budget is not None and self._issued >= budget:
            return False
        if self.config.hours is not None and self.elapsed() >= self.config.hours * 3600.0:
            return False
        return not self._no_parents and not self._capped

    def _request(self, token: int, messages: list[dict]) -> None:
        """A request thread: ask the model, post the reply or the exception."""
        try:
            res = self.client.complete(messages, max_tokens=self.config.max_tokens)
        except BaseException as e:  # posted, not raised, so the main thread decides
            res = e
        self._arrivals.put((token, res))

    def _fill(self) -> None:
        """Issue prompts until `concurrency` are in flight, or the budget or clock stop it.

        Abandoned requests still hold a thread; once `concurrency` of them are
        pending, new prompts wait for them to end, so threads stay bounded."""
        c = max(1, self.config.concurrency)
        while (
            len(self._in_flight) < c
            and len(self._in_flight) + len(self._abandoned) < 2 * c
            and self._want_more()
        ):
            job = self.build_job()
            if job is None:
                self._no_parents = True
                log.warning("an island has no members to breed from; issuing no more prompts")
                return
            if self.ledger is not None:
                worst = self.ledger.worst(job.messages, self.config.max_tokens)
                if not self.ledger.allows(sum(self._worst.values()) + worst):
                    self._capped = True
                    log.warning(
                        "spend cap reached: $%.4f spent, $%.4f held for %d unanswered; "
                        "issuing no more",
                        self.ledger.total,
                        sum(self._worst.values()),
                        len(self._worst),
                    )
                    return
            token = self._next_token
            self._next_token += 1
            self._issued += 1
            if self.ledger is not None:
                self._worst[token] = worst
            self._in_flight[token] = _Flight(job, time.monotonic() + self.config.request_timeout)
            threading.Thread(
                target=self._request,
                args=(token, job.messages),
                name=f"evolve-llm-{token}",
                daemon=True,
            ).start()

    def _charge(self, token: int, res) -> None:
        """Bill a reply against the cap and release what was held for it.

        A late reply to an abandoned request is billed too: the provider ran it."""
        self._worst.pop(token, None)
        if self.ledger is None or isinstance(res, BaseException):
            # An error reply is released: a refused or failed request is not billed
            # for output. Its prompt may be, but that is far below what was held.
            return
        self.run_spent_usd += self.ledger.charge(getattr(res, "usage", None))

    def _accept(self, item) -> None:
        """Move one posted reply from in flight to the scoring queue, or drop a late one."""
        token, res = item
        self._charge(token, res)
        if token in self._abandoned:
            self._abandoned.discard(token)
            log.debug("ignored a reply that arrived after its request timed out")
            return
        flight = self._in_flight.pop(token, None)
        if flight is not None:
            self._ready.append((flight.job, res))

    def _drain(self) -> None:
        while True:
            try:
                self._accept(self._arrivals.get_nowait())
            except queue.Empty:
                return

    def _expire(self) -> None:
        now = time.monotonic()
        for token, flight in list(self._in_flight.items()):
            if now < flight.deadline:
                continue
            del self._in_flight[token]
            self._abandoned.add(token)
            self.llm_calls += 1
            self.llm_timeouts += 1
            self._op(flight.job.operator)["llm_timeouts"] += 1
            log.warning(
                "request for a %s prompt timed out after %.0fs; its slot is refilled",
                flight.job.operator,
                self.config.request_timeout,
            )

    def _score_next(self) -> None:
        job, res = self._ready.popleft()
        self.process_one(job, res)
        replies = self.llm_calls - self.llm_timeouts
        if self.config.migrate_every and replies % self.config.migrate_every == 0:
            self.islands.migrate()
            log.info("migrated after %d replies", replies)

    def _checkpoint(self) -> None:
        self.save()
        st = self.status()
        log.info(
            "status %d: in_flight=%d queue=%d candidates=%d duplicates=%d timeouts=%d "
            "best=%s (%.3f)",
            self.batches,
            st["in_flight"],
            st["queue_depth"],
            st["candidates"],
            st["duplicates"],
            st["llm_timeouts"],
            st["best_id"],
            st["best_val_mean"] if st["best_val_mean"] is not None else float("nan"),
        )

    def run(self) -> dict:
        """Seed if needed, then keep the model busy until the budget or the clock runs out.

        `--budget-candidates` counts requests issued in this session; once it is
        spent, or `--hours` has passed, no more are issued and those in flight are
        drained. A KeyboardInterrupt (SIGINT) ends the loop after saving what is
        known; requests in flight are dropped, and nothing half-written reaches
        the store, because each candidate is one committed row."""
        self._issued = 0
        try:
            self.seed()
            self.save()
            next_save = time.monotonic() + self.config.status_every_s
            while True:
                self._fill()
                self._drain()
                self._expire()
                if self._ready:
                    self._score_next()
                elif not self._in_flight and (not self._want_more() or self._no_parents):
                    break
                else:
                    # Nothing to score: wait for a reply, a deadline or the next save.
                    # The cap keeps Ctrl-C prompt on platforms where a blocking get
                    # does not see it.
                    now = time.monotonic()
                    until = min([next_save] + [f.deadline for f in self._in_flight.values()])
                    try:
                        item = self._arrivals.get(timeout=min(max(until - now, 0.0), 0.5))
                    except queue.Empty:
                        pass
                    else:
                        self._accept(item)
                if time.monotonic() >= next_save:
                    self._checkpoint()
                    next_save = time.monotonic() + self.config.status_every_s
            self.stopped = "done"
        except KeyboardInterrupt:
            self.stopped = "interrupted"
            log.warning("interrupted: saving status and stopping")
        finally:
            # Request threads are daemons: whatever is still with the model is
            # dropped here and cannot hold up the process's exit.
            self._abandoned.update(self._in_flight)
            self._in_flight.clear()
            if self.stopped == "running":
                self.stopped = "failed"
            self._checkpoint()
        return self.status()


# --- command line ---


def spend_ledger(provider, args):
    """The dollar cap for this run, or None only when `--no-spend-cap` asked for that.

    A cap without prices cannot be enforced, and an uncapped run on a paid model
    should be a decision rather than a default, so both refuse to start."""
    from evolve.spend import SpendLedger

    if args.no_spend_cap:
        return None
    caps = [c for c in (provider.max_usd, args.max_usd) if c is not None]
    if not caps:
        raise SystemExit(
            "no spend cap: set max_usd in the provider config or pass --max-usd "
            "(or --no-spend-cap to run without one)"
        )
    if not provider.price:
        raise SystemExit(
            "a spend cap needs prices: add price {input, cached_input, output} "
            "(dollars per million tokens) to the provider config"
        )
    return SpendLedger(provider.ledger_path, provider.price, min(caps))


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
    p.add_argument(
        "--migrate-every", type=int, default=200, help="model replies between migrations"
    )
    p.add_argument(
        "--request-timeout",
        type=float,
        default=600.0,
        help="seconds before an unanswered request is abandoned and its slot refilled",
    )
    p.add_argument("--job-timeout", type=float, default=300.0, help="seconds per pool job")
    p.add_argument("--runs-dir", default=str(FACTORY_SIM / "runs"))
    p.add_argument(
        "--max-usd",
        type=float,
        default=None,
        help="dollar cap across every run sharing the ledger; the provider config's "
        "max_usd applies too, and the lower of the two wins",
    )
    p.add_argument(
        "--no-spend-cap",
        action="store_true",
        help="run with no dollar cap. Without it a run refuses to start if no cap and "
        "prices are configured",
    )
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
        request_timeout=args.request_timeout,
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
    ledger = spend_ledger(provider, args)
    # The per-attempt socket timeout also ends an abandoned request's thread.
    client = LLMClient(
        provider,
        concurrency=args.concurrency,
        timeout_s=min(SOCKET_TIMEOUT_S, args.request_timeout),
    )
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
                ledger=ledger,
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
