"""The dollar cap: pricing, the shared ledger, and the loop stopping at the cap."""

from __future__ import annotations

import dataclasses
import json
import types

import pytest

from evolve import run as evo_run
from evolve.spend import SpendCapError, SpendLedger

PRICE = {"input": 0.05, "cached_input": 0.005, "output": 0.25}  # dollars per million tokens


def usage(prompt=1000, completion=1000, cached=0):
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "prompt_tokens_details": {"cached_tokens": cached},
    }


def test_cost_bills_cached_input_cheaply_and_reasoning_as_output(tmp_path):
    ledger = SpendLedger(tmp_path / "l.json", PRICE, 5.0)
    # 1M fresh input + 1M output = 0.05 + 0.25
    assert ledger.cost(usage(1_000_000, 1_000_000)) == pytest.approx(0.30)
    # half the prompt cached: 0.5 * 0.05 + 0.5 * 0.005
    assert ledger.cost(usage(1_000_000, 0, cached=500_000)) == pytest.approx(0.0275)
    assert ledger.cost(None) == 0.0 and ledger.cost({}) == 0.0


def test_worst_case_holds_the_whole_output_budget(tmp_path):
    ledger = SpendLedger(tmp_path / "l.json", PRICE, 5.0)
    msgs = [{"role": "user", "content": "x" * 3000}]  # ~1000 tokens at 3 chars/token
    assert ledger.worst(msgs, 8000) == pytest.approx((1000 * 0.05 + 8000 * 0.25) / 1e6)


def test_the_ledger_is_a_total_shared_by_every_run_that_names_it(tmp_path):
    path = tmp_path / "shared.json"
    a = SpendLedger(path, PRICE, 1.0)
    b = SpendLedger(path, PRICE, 1.0)
    a.charge(usage(1_000_000, 1_000_000))
    b.charge(usage(1_000_000, 1_000_000))
    assert a.total == pytest.approx(0.60) == b.total
    assert json.loads(path.read_text())["calls"] == 2
    assert a.allows(0.39) and not a.allows(0.41)


def test_a_cap_needs_prices(tmp_path):
    with pytest.raises(SpendCapError, match="price"):
        SpendLedger(tmp_path / "l.json", {"input": 0.05}, 5.0)


class _Reply:
    def __init__(self, text, usage):
        self.text, self.usage, self.latency_s, self.model = text, usage, 0.0, "m"


GOOD = "def build(world):\n    world.wait()\n"


def _evolution(tmp_path, ledger, *, concurrency=4, budget=50):
    """An Evolution on fakes: every reply costs $0.10, and every child is scored."""
    from evolve.archive import Candidate, Islands, Store

    n = {"i": 0}

    class Client:
        def complete(self, messages, max_tokens=0, **kw):
            n["i"] += 1
            code = GOOD + f"# variant {n['i']}\n" + "    " * 0
            code = code.replace("world.wait()", "world.wait()\n" * (n["i"] % 7 + 1))
            return _Reply(f"```python\n{code}```", usage(200_000, 360_000))  # $0.10

    class Evaluator:
        def full(self, source):
            return {"train": {"f": 0.5}, "val": {"f": 0.5}, "train_mean": 0.5,
                    "val_mean": 0.5, "traces": {}, "descriptors": {}, "error": None}  # fmt: skip

    store = Store(tmp_path / "g.sqlite")
    islands = Islands(2, 4, 0)
    seed = Candidate.new(GOOD, operator="seed", island=0,
                         scores={"train": {"f": 0.4}, "val": {"f": 0.4},
                                 "train_mean": 0.4, "val_mean": 0.4})  # fmt: skip
    store.add(seed)
    for i in range(2):
        islands.admit(dataclasses.replace(seed, island=i))
    # A provider never returns more than max_tokens, so the reply's 360k output
    # tokens must sit inside it -- else the worst case under-holds and the test
    # measures a fake that could not exist.
    config = evo_run.Config(name="t", concurrency=concurrency, budget_candidates=budget,
                            max_tokens=400_000, islands=2, island_size=4)  # fmt: skip
    return evo_run.Evolution(config, Client(), Evaluator(), store, islands,
                             tmp_path / "status.json", ledger=ledger)  # fmt: skip


def test_the_loop_stops_issuing_at_the_cap(tmp_path):
    ledger = SpendLedger(tmp_path / "l.json", PRICE, 0.35)
    evo = _evolution(tmp_path, ledger)
    evo.run()
    status = json.loads((tmp_path / "status.json").read_text())
    # Each reply bills $0.10: the cap admits three, and never a fourth.
    assert ledger.total <= 0.35 + 1e-9
    assert status["spend_capped"] is True
    assert status["llm_calls"] == 3
    assert status["spent_usd"] == pytest.approx(ledger.total)


def test_a_late_reply_to_an_abandoned_request_is_still_billed(tmp_path):
    ledger = SpendLedger(tmp_path / "l.json", PRICE, 5.0)
    evo = _evolution(tmp_path, ledger)
    token = 99
    evo._abandoned.add(token)
    evo._worst[token] = 0.5
    evo._accept((token, _Reply("", usage(200_000, 360_000))))
    assert ledger.total == pytest.approx(0.10)
    assert token not in evo._worst  # its reservation is released


def test_no_cap_and_no_opt_out_refuses_to_start():
    provider = types.SimpleNamespace(max_usd=None, price=PRICE, ledger_path="x")
    args = types.SimpleNamespace(no_spend_cap=False, max_usd=None)
    with pytest.raises(SystemExit, match="no spend cap"):
        evo_run.spend_ledger(provider, args)


def test_a_cap_without_prices_refuses_to_start():
    provider = types.SimpleNamespace(max_usd=5.0, price=None, ledger_path="x")
    args = types.SimpleNamespace(no_spend_cap=False, max_usd=None)
    with pytest.raises(SystemExit, match="prices"):
        evo_run.spend_ledger(provider, args)


def test_the_lower_of_config_and_flag_wins(tmp_path):
    provider = types.SimpleNamespace(max_usd=5.0, price=PRICE, ledger_path=str(tmp_path / "l"))
    args = types.SimpleNamespace(no_spend_cap=False, max_usd=2.0)
    assert evo_run.spend_ledger(provider, args).cap_usd == 2.0


def test_holds_are_shared_between_runs_on_one_ledger(tmp_path):
    """Sixteen runs at once must not each believe the room left is theirs."""
    path = tmp_path / "shared.json"
    a = SpendLedger(path, PRICE, 1.0)
    b = SpendLedger(path, PRICE, 1.0)
    a.reserve("run-a", 0.7)
    assert not b.allows(0.4)  # 0.7 held by a, so b cannot take 0.4
    assert b.allows(0.3)
    a.settle("run-a", 0.7, usage(1_000_000, 1_000_000))  # costs 0.30, releases 0.70
    assert a.total == pytest.approx(0.30)
    assert b.allows(0.7) and not b.allows(0.71)


def test_a_crashed_runs_holds_go_stale(tmp_path, monkeypatch):
    import evolve.spend as spend

    path = tmp_path / "l.json"
    a = SpendLedger(path, PRICE, 1.0)
    a.reserve("dead-run", 0.9)
    assert not a.allows(0.2)
    real = spend.time.time
    monkeypatch.setattr(spend.time, "time", lambda: real() + spend.STALE_S + 1)
    assert a.allows(0.2)


def test_exit_books_unanswered_holds_as_spent(tmp_path):
    ledger = SpendLedger(tmp_path / "l.json", PRICE, 5.0)
    ledger.reserve("r", 0.25)
    assert ledger.close("r") == pytest.approx(0.25)
    assert ledger.total == pytest.approx(0.25)
    assert ledger.held() == 0.0


def test_the_trivial_seed_passes_the_sandbox_and_builds_nothing():
    from evolve import sandbox
    from evolve.seeds.trivial import SOURCE
    from fsim import scenes
    from fsim.program_api import run_episode

    sandbox.check(SOURCE)
    _, scene = scenes.sample("construct_smelting_line", "train", 0)
    result = run_episode(sandbox.load(SOURCE), scene)
    assert not result.success and result.built == [] and result.error is None


def test_concurrent_processes_share_one_ledger_without_errors(tmp_path):
    """Three runs died of this on Windows: a reader held the file open while
    another process's atomic replace ran, and the replace was refused."""
    import multiprocessing as mp

    from tests._pool_jobs import ledger_hammer

    path = str(tmp_path / "shared.json")
    SpendLedger(path, PRICE, 1_000_000.0)  # create it before the processes race
    ctx = mp.get_context("spawn")
    with ctx.Pool(8) as pool:
        done = pool.map(ledger_hammer, [(path, 60)] * 8)
    total = SpendLedger(path, PRICE, 1_000_000.0).total
    assert sum(done) == 480
    assert total == pytest.approx(480 * 0.001)  # every charge landed, none twice


def test_a_refused_replace_is_retried(tmp_path, monkeypatch):
    import os

    import evolve.spend as spend

    ledger = SpendLedger(tmp_path / "l.json", PRICE, 5.0)
    real, fails = os.replace, {"n": 2}

    def flaky(src, dst):
        if fails["n"]:
            fails["n"] -= 1
            raise PermissionError(5, "Access is denied")
        real(src, dst)

    monkeypatch.setattr(spend.os, "replace", flaky)
    ledger.charge(usage(1_000_000, 0))
    assert ledger.total == pytest.approx(0.05)
