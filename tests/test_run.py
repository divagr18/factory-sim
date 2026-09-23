"""The evolution loop, with a fake model and a fake evaluator: offline and fast."""

import json
import logging
import random
import sys
import threading
import time
import types

import pytest

import evolve
from evolve import run as evo_run
from evolve import sandbox
from evolve.archive import Islands, Store
from evolve.llm import Completion
from evolve.run import Config, Evolution, load_islands
from evolve.seeds.builder import SOURCE

API_REF = "APIREF-MARKER world.move(direction)"
SEED_VAL = 0.3


def program(waits: int) -> str:
    return "def build(world):\n" + "    world.wait()\n" * waits


def reply(code: str) -> str:
    return f"Plan: wait.\n\n```python\n{code}```\n"


VIOLATION = reply("import os\n\ndef build(world):\n    world.wait()\n")
NO_BLOCK = "I would rather not write code today."


class FakeClient:
    """Replies with `script` in call order, one completion per call, from many threads.

    `latency` is seconds per reply: a number, or a function of the 1-based call
    number. `hold` maps a call number to an Event that call waits on (at most
    5 s) instead. `raise_on` is a call number that raises KeyboardInterrupt."""

    def __init__(self, script, raise_on=None, latency=0.0, hold=None):
        self.script = list(script)
        self.calls = 0
        self.raise_on = raise_on
        self.latency = latency
        self.hold = dict(hold or {})
        self.prompts = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def complete(self, messages, max_tokens=16000):
        with self.lock:
            self.calls += 1
            n = self.calls
            self.prompts.append(messages)
            text = self.script[(n - 1) % len(self.script)]
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.raise_on is not None and n == self.raise_on:
                raise KeyboardInterrupt
            if n in self.hold:
                self.hold[n].wait(5.0)
            else:
                delay = self.latency(n) if callable(self.latency) else self.latency
                if delay:
                    time.sleep(delay)
            if isinstance(text, Exception):
                raise text
            return Completion(text, None, {}, latency_s=2.0, model="fake-model")
        finally:
            with self.lock:
                self.active -= 1


class FakeEvaluator:
    """The seed scores SEED_VAL; any other program scores waits / 10."""

    def __init__(self):
        self.sources = []

    def full(self, source):
        self.sources.append(source)
        v = SEED_VAL if source == SOURCE else source.count("world.wait()") / 10
        rates = {"open_patch": v, "offset_patch": v}
        return {
            "train": rates,
            "val": {"rates": rates, "mean": v},
            "train_mean": v,
            "val_mean": v,
            "traces": {"open_patch": [f"wait -> ok ({v})"]},
            "descriptors": {"layout_signature": f"sig{v}", "mean_decisions": 1.0},
            "error": None,
        }


_open_stores = []


@pytest.fixture(autouse=True)
def _close_stores():
    yield
    while _open_stores:
        _open_stores.pop().close()


def _store(path):
    s = Store(path)
    _open_stores.append(s)
    return s


def make(tmp_path, client, *, evaluator=None, **cfg):
    cfg.setdefault("concurrency", 4)
    cfg.setdefault("islands", 2)
    cfg.setdefault("island_size", 6)
    cfg.setdefault("seed", 7)
    config = Config(name="t", **cfg)
    run_dir = tmp_path / "evolve-t"
    run_dir.mkdir(parents=True, exist_ok=True)
    store = _store(run_dir / "genealogy.sqlite")
    islands = load_islands(
        store, run_dir / "state.json", config.islands, config.island_size, config.seed
    )
    evo = Evolution(
        config,
        client,
        evaluator or FakeEvaluator(),
        store,
        islands,
        run_dir / "status.json",
        api_reference=API_REF,
    )
    return evo, store, run_dir


SCRIPT = [
    reply(program(2)),
    VIOLATION,
    NO_BLOCK,
    reply(
        "def build(world):\n    # a comment the hash ignores\n    world.wait()\n    world.wait()\n"
    ),
    reply(program(3)),
    reply(program(5)),
    reply(program(1)),
    reply(program(4)),
]


def test_every_completion_is_one_candidate_or_one_duplicate(tmp_path):
    evo, store, _ = make(tmp_path, FakeClient(SCRIPT), budget_candidates=8)
    st = evo.run()
    rows = store.all()
    ops = [c.operator for c in rows]
    assert ops.count("seed") == 1
    assert len(rows) - 1 + st["duplicates"] == 8
    assert st["duplicates"] == 1
    assert ops.count("sandbox_error") == 1 and st["sandbox_errors"] == 1
    assert ops.count("extract_failed") == 1 and st["extract_failures"] == 1
    assert st["evaluated"] == 5 and st["ran"] == 5
    assert st["candidates"] == len(rows) == 8
    assert st["llm_calls"] == 8 and st["llm_errors"] == 0

    bad = next(c for c in rows if c.operator == "sandbox_error")
    assert "import os" in bad.code
    assert bad.scores["error"].startswith("sandbox:") and bad.scores["val_mean"] == 0.0
    for c in rows:
        if c.operator == "seed":
            continue
        assert c.parents and c.prompt_hash and c.model == "fake-model"
        assert (
            c.scores["prompt_operator" if c.operator in evo_run.FAILED else "val_mean"] is not None
        )
    hashes = [c.code_hash for c in rows if c.operator not in evo_run.FAILED]
    assert len(hashes) == len(set(hashes))


def test_llm_errors_are_counted_not_stored(tmp_path):
    client = FakeClient([reply(program(2)), RuntimeError("503")])
    evo, store, _ = make(tmp_path, client, budget_candidates=4)
    st = evo.run()
    assert st["llm_calls"] == 4 and st["llm_errors"] == 2
    assert store.count() == 1 + 1  # seed + one program; its clone is a duplicate
    assert st["duplicates"] == 1


def test_parents_feed_prompts_in_parent_dict_format(tmp_path):
    client = FakeClient([reply(program(2))])
    evo, _, _ = make(tmp_path, client, budget_candidates=1, operators={"fix": 1.0})
    evo.run()
    user = client.prompts[0][1]["content"]
    assert "Success rate per scene family" in user
    assert f"wait -> ok ({SEED_VAL})" in user  # the seed's trace
    assert "def build(world):" in user


REQUIRED = {
    "best_val_mean",
    "best_id",
    "best_length",
    "candidates",
    "evaluated",
    "share_ran",
    "share_improved",
    "duplicates",
    "sandbox_errors",
    "extract_failures",
    "llm_calls",
    "llm_errors",
    "llm_timeouts",
    "batches",
    "status_writes",
    "in_flight",
    "queue_depth",
    "mean_latency_s",
    "elapsed_s",
    "candidates_per_hour",
    "last_batch_at",
    "per_operator",
}


def _spy(tmp_path, client, **cfg):
    """An Evolution that keeps every status it writes, in `evo.seen`."""

    class Spy(Evolution):
        seen: list

        def save(self):
            super().save()
            self.seen.append(json.loads(self.status_path.read_text(encoding="utf-8")))

    cfg.setdefault("concurrency", 3)
    cfg.setdefault("islands", 2)
    cfg.setdefault("island_size", 4)
    config = Config(name="t", **cfg)
    run_dir = tmp_path / "evolve-t"
    store = _store(run_dir / "genealogy.sqlite")
    Spy.seen = []  # the constructor does not save, so a fresh list per spy is enough
    evo = Spy(config, client, FakeEvaluator(), store, Islands(2, 4, 0), run_dir / "status.json")
    return evo, store, run_dir


def test_status_file_is_valid_and_written_on_seed_and_exit(tmp_path):
    evo, _, run_dir = _spy(tmp_path, FakeClient(SCRIPT), budget_candidates=8)
    evo.run()
    seen = evo.seen
    # the run is far shorter than status_every_s: once after seeding, once on exit
    assert [s["llm_calls"] for s in seen] == [0, 8]
    assert [s["batches"] for s in seen] == [s["status_writes"] for s in seen] == [1, 2]
    final = seen[-1]
    assert final["in_flight"] == 0 and final["queue_depth"] == 0 and final["llm_timeouts"] == 0
    assert REQUIRED <= set(final)
    assert final["state"] == "done"
    assert final["best_val_mean"] == 0.5 and final["best_length"] == 6
    assert final["mean_latency_s"] == pytest.approx(2.0)
    assert final["candidates_per_hour"] > 0
    assert not list(run_dir.glob("*.tmp"))
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert state["islands"]["n"] == 2 and state["counters"]["duplicates"] == 1


def test_improvement_accounting(tmp_path):
    # One island and one prompt at a time: the first child descends from the
    # seed, and each later prompt is drawn after the previous child was admitted.
    client = FakeClient([reply(program(5)), reply(program(1)), reply(program(4))])
    evo, store, _ = make(
        tmp_path,
        client,
        islands=1,
        concurrency=1,
        budget_candidates=3,
        operators={"fix": 1.0},
    )
    st = evo.run()
    rows = {c.id: c for c in store.all()}
    seed = next(c for c in rows.values() if c.operator == "seed")
    kids = sorted((c for c in rows.values() if c.operator == "fix"), key=lambda c: c.created)
    assert kids[0].parents == [seed.id]
    # 0.5 > 0.3 improves; 0.1 never does; 0.4 improves only if its parent was the seed
    assert kids[1].parents[0] in {seed.id, kids[0].id}
    expected = 1 + (0.4 > rows[kids[2].parents[0]].score())
    assert st["improved"] == expected
    assert st["share_improved"] == pytest.approx(expected / 3)
    fix = st["per_operator"]["fix"]
    assert fix["evaluated"] == 3 and fix["improved"] == expected
    assert fix["improvement_rate"] == pytest.approx(expected / 3)


def test_resume_continues_without_reseeding(tmp_path):
    ev = FakeEvaluator()
    evo, store, run_dir = make(tmp_path, FakeClient(SCRIPT), evaluator=ev, budget_candidates=4)
    first = evo.run()
    store.close()
    ids_before = {m.id for pool in evo.islands.members for m in pool}

    evo2, store2, _ = make(tmp_path, FakeClient(SCRIPT[4:]), evaluator=ev, budget_candidates=4)
    assert {m.id for pool in evo2.islands.members for m in pool} == ids_before
    assert evo2.candidates == first["candidates"]
    second = evo2.run()
    assert ev.sources.count(SOURCE) == 1
    assert [c.operator for c in store2.all()].count("seed") == 1
    assert second["llm_calls"] == 8 and second["duplicates"] == first["duplicates"]
    assert second["candidates"] == store2.count() == first["candidates"] + 4
    assert second["elapsed_s"] >= first["elapsed_s"]
    store2.close()

    # Without state.json the islands are replayed from the genealogy.
    (run_dir / "state.json").unlink()
    store3 = _store(run_dir / "genealogy.sqlite")
    isl = load_islands(store3, run_dir / "state.json", 2, 6, 0)
    seed_id = next(c.id for c in store3.all() if c.operator == "seed")
    for pool in isl.members:
        assert seed_id in {m.id for m in pool}
        assert not any(m.operator in evo_run.FAILED for m in pool)
    store3.close()


def _populated(tmp_path, seed):
    """Islands with several scored members each, via a short real run."""
    script = [reply(program(k)) for k in range(1, 13)]
    evo, store, _ = make(tmp_path, FakeClient(script), budget_candidates=12, seed=seed)
    evo.run()
    return evo


def test_selection_draws_from_the_islands(tmp_path):
    evo = _populated(tmp_path, seed=3)
    evo.config.operators = {"fix": 0.5, "crossover": 0.5}
    jobs = evo.build_batch(8)
    start = jobs[0].island
    assert [j.island for j in jobs] == [(start + i) % 2 for i in range(8)]
    for j in jobs:
        members = {m.id for m in evo.islands.members[j.island]}
        assert {p.id for p in j.parents} <= members
        if j.operator == "crossover":
            assert len({p.id for p in j.parents}) == 2


def test_selection_is_reproducible_given_the_seed(tmp_path):
    def picks(sub):
        evo = _populated(tmp_path / sub, seed=11)
        evo.config.operators = dict(evo_run.DEFAULT_OPERATORS)
        jobs = evo.build_batch(10)
        # ids are fresh per run, so compare parents by what they are
        return [(j.operator, j.island, [p.code for p in j.parents]) for j in jobs]

    assert picks("a") == picks("b")


def test_dry_run_prints_ids_not_prompts(tmp_path, monkeypatch, capsys):
    fake = types.ModuleType("evolve.evaluate")
    fake.api_reference = lambda: API_REF
    monkeypatch.setitem(sys.modules, "evolve.evaluate", fake)
    monkeypatch.setattr(evolve, "evaluate", fake, raising=False)

    def no_provider(*a, **k):
        raise AssertionError("a dry run must not load the provider")

    monkeypatch.setattr("evolve.llm.load_provider", no_provider)
    rc = evo_run.main(
        ["--name", "d", "--dry-run", "--runs-dir", str(tmp_path), "--concurrency", "5"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    lines = out.strip().splitlines()
    assert len(lines) == 5
    assert all("parents=" in line and "island=" in line for line in lines)
    for leak in ("APIREF", "def build", "world.", "Program", "Game notes"):
        assert leak not in out
    assert not (tmp_path / "evolve-d").exists()


def test_main_wires_the_modules_and_resumes(tmp_path, monkeypatch):
    fake = types.ModuleType("evolve.evaluate")
    fake.EVALUATOR_VERSION = 1
    fake.api_reference = lambda: API_REF
    fake.scene_sets = lambda: {"train": [("open_patch", 0, {"a": 1})], "val": []}
    fake.scene_digest = lambda bp: "d" * 64
    fake.set_digests = lambda sets: {k: "s" * 16 for k in sets}
    evaluators = []

    def make_evaluator(pool, sets):
        evaluators.append(FakeEvaluator())
        return evaluators[-1]

    fake.Evaluator = make_evaluator
    monkeypatch.setitem(sys.modules, "evolve.evaluate", fake)
    monkeypatch.setattr(evolve, "evaluate", fake, raising=False)

    class Pool:
        def __init__(self, workers, initializer, job, timeout_s):
            assert initializer == "evolve.evaluate:worker_init"
            assert job == "evolve.evaluate:worker_job"
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.closed = True

    monkeypatch.setattr("evolve.pool.EvalPool", Pool)
    # A provider with a cap and prices: main refuses to start a run on a paid model
    # without them, and the ledger lives in the test's own directory.
    fake_provider = types.SimpleNamespace(
        model="fake-model",
        max_usd=5.0,
        price={"input": 0.05, "cached_input": 0.005, "output": 0.25},
        ledger_path=str(tmp_path / "spend.json"),
    )
    monkeypatch.setattr("evolve.llm.load_provider", lambda path=None: fake_provider)
    timeouts = []

    def make_client(provider, concurrency, timeout_s):
        timeouts.append(timeout_s)
        return FakeClient(SCRIPT)

    monkeypatch.setattr("evolve.llm.LLMClient", make_client)
    argv = ["--name", "m", "--runs-dir", str(tmp_path), "--concurrency", "2"]
    assert evo_run.main(argv + ["--budget-candidates", "4"]) == 0
    run_dir = tmp_path / "evolve-m"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"] == "fake-model" and manifest["resumed"] is False
    assert manifest["scene_set_digests"] == {"train": "s" * 16, "val": "s" * 16}
    first = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert first["state"] == "done" and first["llm_calls"] == 4

    assert evo_run.main(argv + ["--budget-candidates", "2"]) == 0
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["resumed"] is True
    second = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert second["llm_calls"] == 6
    assert SOURCE not in evaluators[1].sources  # not reseeded
    assert timeouts == [evo_run.SOCKET_TIMEOUT_S] * 2

    assert evo_run.main(argv + ["--budget-candidates", "1", "--request-timeout", "20"]) == 0
    assert timeouts[-1] == 20.0


def test_keyboard_interrupt_from_the_client_saves_and_stops(tmp_path):
    # One request at a time, so the third is issued only after two were scored.
    client = FakeClient(SCRIPT, raise_on=3)
    evo, store, run_dir = make(tmp_path, client, concurrency=1, budget_candidates=40)
    st = evo.run()  # does not raise
    assert st["state"] == "interrupted"
    on_disk = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert on_disk["state"] == "interrupted"
    assert client.calls == 3
    # the seed, program(2) and the sandbox violation
    assert store.count() == 3 and st["llm_calls"] == 2 and on_disk["llm_calls"] == 2
    assert (run_dir / "state.json").exists()


def test_keyboard_interrupt_while_scoring_keeps_what_was_stored(tmp_path):
    class Stops(FakeEvaluator):
        def full(self, source):
            if len(self.sources) == 3:
                raise KeyboardInterrupt
            return super().full(source)

    evo, store, run_dir = make(tmp_path, FakeClient(SCRIPT), evaluator=Stops(), budget_candidates=8)
    st = evo.run()
    assert st["state"] == "interrupted"
    assert st["candidates"] == store.count()
    on_disk = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert on_disk["candidates"] == store.count()


def test_logs_hold_no_prompts_or_code(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger="evolve")
    evo, _, _ = make(tmp_path, FakeClient(SCRIPT), budget_candidates=8)
    evo.run()
    text = caplog.text
    assert "status" in text and "candidate" in text
    for leak in ("APIREF", "def build", "world.wait", "import os", "rather not"):
        assert leak not in text


# --- the pipeline ---


def test_fast_completions_are_scored_before_a_slow_one_returns(tmp_path):
    # Call 1 is stuck until scoring releases it: with a batch barrier nothing
    # would be scored, the hold would run out after 5 s and the flag stay unset.
    release = threading.Event()

    class Releases(FakeEvaluator):
        released_by_scoring = False

        def full(self, source):
            out = super().full(source)
            if len([s for s in self.sources if s != SOURCE]) == 5:
                self.released_by_scoring = True
                release.set()
            return out

    script = [reply(program(9))] + [reply(program(k)) for k in range(1, 8)]
    client = FakeClient(script, hold={1: release})
    ev = Releases()
    t0 = time.monotonic()
    evo, store, _ = make(tmp_path, client, evaluator=ev, concurrency=4, budget_candidates=8)
    st = evo.run()
    assert ev.released_by_scoring
    assert time.monotonic() - t0 < 4.0
    assert ev.sources[-1] == program(9)  # the slow child is scored last
    assert st["llm_calls"] == client.calls == 8 and st["evaluated"] == 8
    assert store.count() == 9


def test_in_flight_never_exceeds_concurrency(tmp_path):
    rng = random.Random(5)
    delays = {n: rng.uniform(0.0, 0.02) for n in range(1, 31)}
    client = FakeClient([reply(program(k)) for k in range(1, 31)], latency=lambda n: delays[n])
    evo, _, _ = _spy(tmp_path, client, budget_candidates=30, status_every_s=0.005)
    st = evo.run()
    assert st["llm_calls"] == client.calls == 30
    assert 2 <= client.max_active <= 3
    assert all(s["in_flight"] <= 3 for s in evo.seen)
    assert st["in_flight"] == 0


def test_budget_is_respected_when_concurrency_exceeds_it(tmp_path):
    client = FakeClient([reply(program(k)) for k in range(1, 9)], latency=lambda n: 0.01 * n)
    evo, store, _ = make(tmp_path, client, concurrency=8, budget_candidates=3)
    st = evo.run()
    assert client.calls == st["llm_calls"] == 3
    assert store.count() == 4


def test_request_timeout_abandons_a_stuck_request_and_refills_its_slot(tmp_path):
    # One slot. Call 1 hangs until two later children are scored, then answers
    # with a program that must never be scored: its request had timed out.
    late = threading.Event()

    class Releases(FakeEvaluator):
        def full(self, source):
            out = super().full(source)
            if len([s for s in self.sources if s != SOURCE]) == 2:
                late.set()
            return out

    script = [reply(program(9))] + [reply(program(k)) for k in range(1, 5)]
    client = FakeClient(script, hold={1: late}, latency=0.1)
    ev = Releases()
    t0 = time.monotonic()
    evo, store, run_dir = make(
        tmp_path, client, evaluator=ev, concurrency=1, budget_candidates=5, request_timeout=0.2
    )
    st = evo.run()
    assert time.monotonic() - t0 < 4.0
    assert st["llm_timeouts"] == 1 and st["llm_errors"] == 0
    assert st["llm_calls"] == client.calls == 5  # the stuck one and four refills
    assert program(9) not in ev.sources
    assert store.count() == 1 + 4
    assert st["abandoned_running"] == 0  # its late reply came back and was dropped
    assert sum(o["llm_timeouts"] for o in st["per_operator"].values()) == 1
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert state["counters"]["llm_timeouts"] == 1


def test_a_request_stuck_past_the_run_does_not_hold_it_up(tmp_path):
    stuck = threading.Event()
    client = FakeClient([reply(program(9)), reply(program(1))], hold={1: stuck})
    evo, store, _ = make(tmp_path, client, concurrency=2, budget_candidates=2, request_timeout=0.1)
    try:
        t0 = time.monotonic()
        st = evo.run()
        assert time.monotonic() - t0 < 2.0
        assert st["state"] == "done" and st["llm_timeouts"] == 1 and store.count() == 2
    finally:
        stuck.set()


def test_islands_migrate_every_n_replies(tmp_path):
    client = FakeClient([reply(program(k)) for k in range(1, 8)])
    evo, _, _ = make(tmp_path, client, concurrency=2, budget_candidates=7, migrate_every=3)
    calls = []
    real = evo.islands.migrate
    evo.islands.migrate = lambda: (calls.append(evo.llm_calls), real())
    evo.run()
    assert calls == [3, 6]


def test_status_is_written_periodically(tmp_path):
    client = FakeClient([reply(program(k)) for k in range(1, 7)], latency=0.03)
    evo, _, _ = _spy(tmp_path, client, concurrency=1, budget_candidates=6, status_every_s=0.05)
    evo.run()
    seen = evo.seen
    assert len(seen) >= 3  # seed, at least one periodic write, exit
    assert [s["batches"] for s in seen] == list(range(1, len(seen) + 1))
    assert all(s["status_writes"] == s["batches"] for s in seen)
    assert any(s["in_flight"] == 1 for s in seen[1:-1])
    assert seen[-1]["llm_calls"] == 6 and seen[-1]["state"] == "done"


def test_parse_operators():
    assert evo_run.parse_operators("fix:0.4, rewrite:0.6") == {"fix": 0.4, "rewrite": 0.6}
    with pytest.raises(ValueError):
        evo_run.parse_operators("mutate:1")
    with pytest.raises(ValueError):
        evo_run.parse_operators("fix:0")


def test_sandbox_accepts_the_test_programs():
    sandbox.check(program(3))
    with pytest.raises(sandbox.SandboxError):
        sandbox.check("import os\n\ndef build(world):\n    world.wait()\n")
