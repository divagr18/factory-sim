"""A hidden-state load that the activation order cannot follow is refused.

Belt-line segments move in the engine's activation order (csrc/fsim.c,
"segments"), which the engine does not export. A load that changes the items
of a segment that still holds items afterwards would leave the simulator to
guess that segment's place, so `Sim.load_hidden` raises BeltOrderUnknown
before anything is loaded (FactorioRL docs/sim-logistics.md, "A loaded state
and the activation order", decision 2026-09-25). Loads that leave the belts
as the simulator has them, as in every one-step sync of the golden traces
(tests/test_parity.py), and loads that empty a segment, go through.
"""

from __future__ import annotations

import copy

import pytest

from fsim import ITEM_IDS, BeltOrderUnknown, Sim, lib

E = 4


def _world():
    """Four east belts at (200..203, 200), an iron plate put on lane 1 of the
    first, five ticks run."""
    sim = Sim(water=[])
    sim.reset({"character": {"position": [0.5, 0.5]}})
    env = sim.env
    belts = [lib.fsim_add_entity(env, lib.K_BELT, (200 + i) * 256 + 128, 200 * 256 + 128, E)
             for i in range(4)]  # fmt: skip
    lib.fsim_refresh(env)
    assert lib.fsim_belt_insert(env, belts[0], 0, 128, ITEM_IDS["iron-plate"])
    lib.fsim_advance(env, 5)
    return sim, belts


def _belt(hidden: dict, x: int) -> dict:
    return next(r for r in hidden["entities"]
                if r["name"] == "transport-belt" and r["position"][0] == x * 256 + 128)  # fmt: skip


def _lanes(record: dict) -> list:
    lanes = record.get("lanes") or [[], []]
    return [list(lane) if isinstance(lane, list) else [] for lane in lanes]


def _items(sim: Sim, belt: int, lane: int) -> list:
    ln = sim.env.entities[belt].lanes[lane]
    return [(ln.items[k].item, ln.items[k].pos) for k in range(ln.count)]


def test_a_load_that_changes_no_belt_goes_through():
    sim, belts = _world()
    before = [_items(sim, b, lane) for b in belts for lane in (0, 1)]
    sim.load_hidden(sim.hidden())
    assert [_items(sim, b, lane) for b in belts for lane in (0, 1)] == before


def test_an_item_on_an_empty_segment_is_refused():
    sim, belts = _world()
    hidden = copy.deepcopy(sim.hidden())
    record = _belt(hidden, 203)
    lanes = _lanes(record)
    lanes[1] = [["copper-plate", 100, 99]]
    record["lanes"] = lanes
    tick = sim.env.tick
    with pytest.raises(BeltOrderUnknown, match="lane 2 of the belt at"):
        sim.load_hidden(hidden)
    # Refused before anything was loaded.
    assert sim.env.tick == tick and _items(sim, belts[3], 1) == []


def test_moving_an_item_on_a_segment_that_holds_it_is_refused():
    sim, belts = _world()
    hidden = copy.deepcopy(sim.hidden())
    record = _belt(hidden, 200)
    lanes = _lanes(record)
    name, position, *rest = lanes[0][0]
    lanes[0] = [[name, position - 8, *rest]]
    record["lanes"] = lanes
    with pytest.raises(BeltOrderUnknown, match="lane 1 of the belt at"):
        sim.load_hidden(hidden)


def test_emptying_a_segment_goes_through():
    sim, belts = _world()
    hidden = copy.deepcopy(sim.hidden())
    record = _belt(hidden, 200)
    lanes = _lanes(record)
    lanes[0] = []
    record["lanes"] = lanes
    sim.load_hidden(hidden)
    assert all(_items(sim, b, lane) == [] for b in belts for lane in (0, 1))
    lib.fsim_advance(sim.env, 10)
    assert all(_items(sim, b, lane) == [] for b in belts for lane in (0, 1))
