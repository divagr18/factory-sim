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

Scenarios the index marks `"requires": "logistics"` run belts, inserters and
chests. Each reaches a mechanic the simulator does not reproduce yet, so the
strict tests skip them until it does, and `test_logistics_agree_until_known_gap`
holds them to exact agreement up to that point and checks the first difference
is the known one.
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
LOGISTICS = {name for name, entry in INDEX.items() if entry.get("requires") == "logistics"}

#: Where each logistics trace first departs from the simulator: the decision
#: (free-running and sync alike), the start of the differing path, the first
#: differing tick of the per-tick trace, and why.
KNOWN_GAPS = {
    # A waiting inserter starts moving the tick after an ore lands anywhere on
    # its pickup belt's line (t=243, four belts and a turn upstream): the
    # engine's chase of moving belt items, not modelled (inserter_belt_pickup).
    "logistics_smelting_chain": (9, ".remaining_burning_fuel", 244, "belt chase"),
    "logistics_belt_pickup": (2, ".remaining_burning_fuel", 48, "belt chase"),
    # Both feed lanes reach the main belt on the same tick for the first time
    # and the engine moves one of the two items 8/256 further (update_belts).
    "logistics_sideload_merge": (5, ".entities[8].lanes", 127, "first sideload arrival"),
    # After a tick with less than a full tick's energy the hand is drawn part
    # of a step on; the tick itself leaves 875/2^26 J in the buffer.
    "logistics_inserter_fuel_exhaustion": (
        19, ".entities[1].held_stack_position", 555, "part-energy tick"),
    # A loaded turn rotated back to straight: its items are re-placed by a
    # rule the simulator does not have (act_rotate).
    "logistics_belt_rotate_and_mine": (22, ".entities[5].lanes", 631, "rotated loaded belt"),
}  # fmt: skip


def _strict(name: str) -> None:
    if name in LOGISTICS:
        pytest.skip(f"{name}: known gap, {KNOWN_GAPS[name][3]}")


@pytest.mark.parametrize("name", SCENARIOS)
def test_free_running(name):
    _strict(name)
    found = free_run(name)
    assert not found, found[0]


@pytest.mark.parametrize("name", TICKED)
def test_tick_by_tick(name):
    _strict(name)
    found = tick_run(name)
    assert found is None, found


def _sync(name: str):
    """The first record a one-step sync gets wrong, or None."""
    header, records = read_trace(GOLDEN / f"{name}.jsonl.xz")
    replay = Replay(header)
    replay.reset()
    for previous, record in zip(records, records[1:], strict=False):
        replay.load_hidden(previous["hidden"])
        action = record["transition"]["action"]
        actual = replay.step(action["key"], action["arguments"])
        found = compare(record, actual)
        if found:
            return record["decision"], found
    return None


@pytest.mark.parametrize("name", SCENARIOS)
def test_one_step_sync(name):
    _strict(name)
    found = _sync(name)
    assert found is None, found


@pytest.mark.parametrize("name", sorted(KNOWN_GAPS))
def test_logistics_agree_until_known_gap(name):
    decision, path, tick, _ = KNOWN_GAPS[name]
    found = free_run(name)
    assert found and (found[0].decision, found[0].path[: len(path)]) == (decision, path), found
    found = _sync(name)
    assert found and (found[0], found[1][1][: len(path)]) == (decision, path), found
    found = tick_run(name)
    assert found is not None and found.decision == tick, found


def test_every_logistics_scenario_has_a_known_gap():
    assert set(KNOWN_GAPS) == LOGISTICS


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
