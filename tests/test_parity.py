"""The simulator against FactorioRL's golden traces.

- free-running: install the recorded scene, apply the recorded actions, and
  every record -- wire observation, evaluator truth, hidden state, the step's
  action outcome -- must match;
- tick by tick: the mechanics scenarios, replayed one tick at a time, must
  match the engine's hidden state on every tick;
- sync: at every decision, the engine's recorded state is loaded over the
  simulator's, one step is taken, and the next record must match;
- determinism: the same actions give the same state, twice.

Doubles are compared to 1e-12 (see fsim/trace.py), and ore under a drill as a
footprint total (see fsim/parity.py). Everything else is exact.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from fsim.parity import GOLDEN, Replay, compare, free_run, tick_run
from fsim.trace import read_trace

INDEX = json.loads((GOLDEN / "index.json").read_text(encoding="utf-8"))
SCENARIOS = sorted(INDEX)
TICKED = sorted(name for name, entry in INDEX.items() if entry.get("ticks"))


@pytest.mark.parametrize("name", SCENARIOS)
def test_free_running(name):
    found = free_run(name)
    assert not found, found[0]


@pytest.mark.parametrize("name", TICKED)
def test_tick_by_tick(name):
    found = tick_run(name)
    assert found is None, found


@pytest.mark.parametrize("name", SCENARIOS)
def test_one_step_sync(name):
    header, records = read_trace(GOLDEN / f"{name}.jsonl.xz")
    replay = Replay(header)
    replay.reset()
    for previous, record in zip(records, records[1:], strict=False):
        replay.sim.load_hidden(previous["hidden"])
        action = record["transition"]["action"]
        actual = replay.step(action["key"], action["arguments"])
        found = compare(record, actual)
        assert found is None, (record["decision"], found)


@pytest.mark.parametrize("name", ["construct_smelting_line_reference", "masked_random_rollout"])
def test_the_same_actions_give_the_same_state(name):
    header, records = read_trace(GOLDEN / f"{name}.jsonl.xz")

    def run() -> str:
        replay = Replay(header)
        digest = hashlib.sha256()
        digest.update(json.dumps(replay.reset(), sort_keys=True).encode())
        for record in records[1:]:
            action = record["transition"]["action"]
            state = replay.step(action["key"], action["arguments"])
            digest.update(json.dumps(state, sort_keys=True).encode())
        return digest.hexdigest()

    assert run() == run()
