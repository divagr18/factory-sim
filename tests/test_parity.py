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
chests. Those in `KNOWN_GAPS` reach a mechanic the simulator does not
reproduce yet, so the strict tests skip them until it does, and
`test_logistics_agree_until_known_gap` holds them to exact agreement up to
that point and checks the first difference is the known one. The others are
held to the strict tests like every other scenario.
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
#: None now: `logistics_smelting_chain` (an inserter watching its belt-line
#: segment falls asleep with ore on the other side of a split) and
#: `logistics_sideload_merge` (two sideloads onto one empty lane in one tick)
#: agree exactly since csrc/fsim.c models segments ("segments").
KNOWN_GAPS: dict = {}


def _strict(name: str) -> None:
    if name in KNOWN_GAPS:
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


def test_known_gaps_are_logistics_scenarios():
    # Agreeing exactly: logistics_belt_rotate_and_mine since loaded belts that
    # change shape re-place their items (rebuild_logistics); logistics_belt_pickup
    # and logistics_inserter_fuel_exhaustion since the arm model (arm_step,
    # inserter_chase) -- the hand's y compared only where its lift is known
    # (fsim.trace.relax_hand_y); logistics_smelting_chain and
    # logistics_sideload_merge since belt-line segments.
    assert set(KNOWN_GAPS) <= LOGISTICS
    assert LOGISTICS - set(KNOWN_GAPS) == {
        "logistics_belt_rotate_and_mine", "logistics_belt_pickup",
        "logistics_inserter_fuel_exhaustion", "logistics_smelting_chain",
        "logistics_sideload_merge"}  # fmt: skip


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
