"""The evolution loop, with a fake model and a fake evaluator: offline and fast."""

import json
import logging
import sys
import threading
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
    """Returns `script` in order, one completion per prompt; `raise_on` is a
    1-based call number at which it raises KeyboardInterrupt instead."""

    def __init__(self, script, raise_on=None):
        self.script = list(script)
        self.i = 0
        self.calls = 0
        self.raise_on = raise_on
        self.batches = []
        self.lock = threading.Lock()

    def complete_many(self, batch, max_tokens=16000):
        with self.lock:
            self.calls += 1
            self.batches.append(batch)
            if self.raise_on is not None and self.calls == self.raise_on:
                raise KeyboardInterrupt
            out = []
            for _ in batch:
                text = self.script[self.i % len(self.script)]
                self.i += 1
                if isinstance(text, Exception):
                    out.append(text)
                else:
                    out.append(Completion(text, None, {}, latency_s=2.0, model="fake-model"))
            return out


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
    user = client.batches[0][0][1]["content"]
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
    "mean_latency_s",
    "elapsed_s",
    "candidates_per_hour",
    "last_batch_at",
    "per_operator",
}


def test_status_file_is_valid_and_written_after_every_batch(tmp_path):
    seen = []

    class Spy(Evolution):
        def save(self):
            super().save()
            seen.append(json.loads(self.status_path.read_text(encoding="utf-8")))

    config = Config(name="t", concurrency=3, islands=2, island_size=4, budget_candidates=8)
    run_dir = tmp_path / "evolve-t"
    store = _store(run_dir / "genealogy.sqlite")
    islands = Islands(2, 4, 0)
    evo = Spy(config, FakeClient(SCRIPT), FakeEvaluator(), store, islands, run_dir / "status.json")
    evo.run()
    # after seeding, after each of three batches (3 + 3 + 2), and on the way out
    assert [s["llm_calls"] for s in seen] == [0, 3, 6, 8, 8]
    final = seen[-1]
    assert REQUIRED <= set(final)
    assert final["state"] == "done"
    assert final["best_val_mean"] == 0.5 and final["best_length"] == 6
    assert final["mean_latency_s"] == pytest.approx(2.0)
    assert final["candidates_per_hour"] > 0
    assert not list(run_dir.glob("*.tmp"))
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert state["islands"]["n"] == 2 and state["counters"]["duplicates"] == 1


def test_improvement_accounting(tmp_path):
    # One island and one prompt at a time, so the parents are known: batch 2 is
    # prompted while batch 1 is being scored, so both children descend from the seed.
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
    assert [c.parents for c in kids[:2]] == [[seed.id], [seed.id]]
    # 0.5 > 0.3 improves; 0.1 does not; 0.4 improves only if its parent was the seed
    third_parent = rows[kids[2].parents[0]]
    expected = 1 + (0.4 > third_parent.score())
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
    monkeypatch.setattr(
        "evolve.llm.load_provider", lambda path=None: types.SimpleNamespace(model="fake-model")
    )
    monkeypatch.setattr("evolve.llm.LLMClient", lambda provider, concurrency: FakeClient(SCRIPT))
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


def test_keyboard_interrupt_from_the_client_saves_and_stops(tmp_path):
    client = FakeClient(SCRIPT, raise_on=2)
    evo, store, run_dir = make(tmp_path, client, budget_candidates=40)
    st = evo.run()  # does not raise
    assert st["state"] == "interrupted"
    on_disk = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert on_disk["state"] == "interrupted"
    assert client.calls == 2
    # batch 1 was scored while batch 2 was with the model: 3 rows and a duplicate
    assert store.count() == 4 and st["llm_calls"] == 4 and st["duplicates"] == 1
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
    assert "batch" in text
    for leak in ("APIREF", "def build", "world.wait", "import os", "rather not"):
        assert leak not in text


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
