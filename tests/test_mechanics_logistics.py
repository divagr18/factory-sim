"""Belts, burner inserters and chests against the engine, tick by tick.

FactorioRL's logistics probe (tools/probe_logistics.py there) built seventeen
rigs in Factorio 2.0.60 and read them after every tick for 3,000 ticks;
docs/sim-logistics.md there states what it found. tests/logistics_rigs.py
builds the same rigs here, and the engine's readings are in
tests/golden/sim-mechanics-m4-logistics.json.xz (tools/trim_logistics_evidence.py).

Every reading of every compared rig must agree on every tick: belt item
positions exactly, energy and remaining fuel to the parity tolerance (1e-12;
the engine prints 1/240 with an imprecise last digit), everything else
exactly. The exceptions are listed in `GAPS`, each a mechanic the simulator
does not reproduce, and each is held to failing by a strict xfail so that
closing it shows. Rigs where an inserter takes an item still moving on a belt
are not compared at all (see inserter_belt_pickup in csrc/fsim.c).
"""

from __future__ import annotations

import json
import lzma
import math
from pathlib import Path

import pytest
from logistics_rigs import Rigs

from fsim import HAND_Y_UNKNOWN, ITEM_IDS, Sim, lib

GOLDEN = Path(__file__).resolve().parent / "golden" / "sim-mechanics-m4-logistics.json.xz"
EVIDENCE = json.loads(lzma.decompress(GOLDEN.read_bytes()))
TICKS = EVIDENCE["ticks"]

#: (rig, field path or "" for all of it, first tick, last tick): readings
#: known not to match.
GAPS = {
    # Both feed lanes reach the main belt on t=128, and the engine moves the
    # lane-1 item 8/256 further; it catches up by t=138 (update_belts).
    "first sideload arrival": ("side_main", "", 128, 137),
}


def _expand(series: list) -> list:
    out, at = [], 0
    for t in range(TICKS + 1):
        while at + 1 < len(series) and series[at + 1][0] <= t:
            at += 1
        out.append(series[at][1])
    return out


ENGINE = {key: _expand(series) for key, series in EVIDENCE["series"].items()}


@pytest.fixture(scope="module")
def sim():
    return list(Rigs().samples())


def _number(value):
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def same(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    na, nb = _number(a), _number(b)
    if na is not None and nb is not None:
        return math.isclose(na, nb, rel_tol=1e-12, abs_tol=1e-12)
    return a == b


def flatten(value, path: str = "") -> dict:
    """A reading as {field path: value}: nested dicts by dotted key."""
    if not isinstance(value, dict):
        return {path: value}
    out = {}
    for k, v in value.items():
        out.update(flatten(v, f"{path}.{k}" if path else k))
    return out


def _excused(key: str, path: str, t: int) -> bool:
    for gap_key, gap_path, first, last in GAPS.values():
        if gap_key == key and first <= t <= last and path.endswith(gap_path):
            return True
    return False


#: Rigs whose belts were built on tick 0 and whose inserter takes from them
#: while they are young. Accepted on 2026-09-24 (docs/sim-logistics.md,
#: "Inserter belt pickup", rule 6): for about the first 300 ticks after belts
#: are built the engine wakes an inserter asleep on them late, by an amount it
#: does not let us predict, and items on them do not keep it awake; the
#: simulator does not model that. These rigs are compared from t=300 on, the
#: fuel left in the inserter's burner up to the constant offset the young
#: period leaves (its first moves were paid from a different buffer).
YOUNG_BELT_RIGS = {"bend", "bend_belt", "bend_furnace", "flow2", "tick_ins"}
YOUNG_BELT_TICKS = 300


def _unmark_hand_y(engine: dict, ours: dict) -> None:
    """Hand y is not compared where the simulator does not know the lift
    (fsim.trace.relax_hand_y)."""
    for path in [p for p in ours if p.endswith(HAND_Y_UNKNOWN)]:
        del ours[path]
        hand = path[: -len(HAND_Y_UNKNOWN)] + "hand"
        if isinstance(engine.get(hand), list) and isinstance(ours.get(hand), list):
            ours[hand] = [ours[hand][0], engine[hand][1]]


def mismatches(sim, key: str) -> list:
    out = []
    young = key in YOUNG_BELT_RIGS
    offsets: dict = {}
    for t in range(TICKS + 1):
        engine, ours = flatten(ENGINE[key][t]), flatten(sim[t][key])
        _unmark_hand_y(engine, ours)
        if young and t < YOUNG_BELT_TICKS:
            continue
        if young:
            for path in [p for p in engine if p.endswith("remaining")]:
                a, b = _number(engine[path]), _number(ours.get(path))
                if a is None or b is None:
                    continue
                offsets.setdefault(path, b - a)
                engine[path] = repr(a + offsets[path])
        for path in sorted(set(engine) | set(ours)):
            a, b = engine.get(path), ours.get(path)
            if not same(a, b) and not _excused(key, path, t):
                out.append((t, path, a, b))
    return out


def lane(sample: dict, key: str, belt: int, index: int) -> list[int]:
    return [position for _, position in sample[key][belt][index]]


RIGS = sorted(EVIDENCE["series"])


@pytest.mark.parametrize("key", RIGS)
def test_rig_matches_the_engine_every_tick(sim, key):
    found = mismatches(sim, key)
    assert not found, found[:5]


@pytest.mark.parametrize("gap", sorted(GAPS))
@pytest.mark.xfail(strict=True, reason="a mechanic the simulator does not reproduce yet")
def test_known_gap(sim, gap):
    key, gap_path, first, last = GAPS[gap]
    for t in range(first, last + 1):
        engine, ours = flatten(ENGINE[key][t]), flatten(sim[t][key])
        for path in engine:
            if path.endswith(gap_path):
                assert same(engine[path], ours.get(path)), (t, path, engine[path], ours.get(path))


# --------------------------------------------------------------------- belts
# The numbers docs/sim-logistics.md states, read off the simulator.


def test_straight_run_speed_handover_and_compression(sim):
    assert lane(sim[0], "straight", 0, 0) == [248]  # insert_at_back reads 248
    assert lane(sim[1], "straight", 0, 0) == [240]  # 8/256 per tick
    assert lane(sim[31], "straight", 0, 0)[0] == 0
    assert lane(sim[32], "straight", 1, 0) == [248]  # 0 - 8 + 256 on the next belt
    assert lane(sim[319], "straight", 9, 0)[0] == 0  # (248 + 9 * 256) / 8
    for lane_index in (0, 1):
        assert lane(sim[400], "straight", 9, lane_index) == [0, 64, 128, 192]
        assert lane(sim[400], "straight", 8, lane_index) == [0, 64]


def test_right_turn_lane_lengths(sim):
    # Belt 4 of the curve rig is the turn: outer lane 1 is 295 long, inner 106.
    assert lane(sim[128], "curve", 4, 0) == [287]
    assert lane(sim[163], "curve", 4, 0) == [7]
    assert lane(sim[164], "curve", 5, 0) == [255]
    assert lane(sim[128], "curve", 4, 1) == [98]
    assert lane(sim[140], "curve", 4, 1) == [2]
    assert lane(sim[141], "curve", 5, 1) == [250]
    assert lane(sim[260], "curve", 7, 0) == [0] and lane(sim[259], "curve", 7, 0) != [0]
    assert lane(sim[237], "curve", 7, 1) == [0] and lane(sim[236], "curve", 7, 1) != [0]


@pytest.mark.parametrize("k", range(8))
def test_sideload_of_one_item(sim, k):
    # Feed lane 1 joins the main belt's lane 2 at 188, feed lane 2 at 67, each
    # less the rest of the item's move.
    for feed_lane, entry in ((1, 188), (2, 67)):
        rig = f"phase{(feed_lane - 1) * 8 + k + 1}"
        assert sim[15][rig]["feed"][0][feed_lane - 1] == [["copper-plate", k]]
        assert sim[15][rig]["main"][1] == [[], []]
        assert sim[16][rig]["main"][1] == [[], [["copper-plate", entry - (8 - k)]]]


def test_a_blocked_drill_reads_working_once_its_belt_is_extended(sim):
    # The belt goes down at t=1500 between ticks; the engine's status reads
    # `working` at once, before the drill runs again (drill_block_signature).
    assert sim[1499]["drill"]["status"] == "waiting_for_space_in_destination"
    assert sim[1500]["drill"]["status"] == "working"


def test_drill_output_onto_a_belt(sim):
    assert lane(sim[243], "drill_belt", 0, 0) == [120]  # lane 1, 128 moved once
    assert lane(sim[723], "drill_belt", 0, 0) == [0, 64, 128]
    assert sim[963]["drill"]["status"] == "waiting_for_space_in_destination"
    assert lane(sim[1499], "drill_belt", 0, 0) == [0, 64, 128]  # 192 is never used
    assert lane(sim[1501], "drill_belt", 0, 0) == [56, 120, 184]


# ----------------------------------------------------------------- inserters


def _changes(sim, key, pick):
    out, previous = [], object()
    for t, sample in enumerate(sim):
        value = pick(sample[key])
        if value != previous:
            out.append((t, value))
            previous = value
    return out


def test_chest_to_chest_cycle(sim):
    held = _changes(sim, "c2c", lambda s: s["held"])
    assert held[:5] == [(0, None), (9, "iron-plate"), (47, None), (85, "iron-plate"), (123, None)]
    assert _changes(sim, "c2c_dst", lambda n: n)[:3] == [(0, 0), (47, 1), (123, 2)]


def test_energy_per_swing(sim):
    remaining = [float(s["c2c"]["remaining"]) for s in sim]
    # t=85..161: a pickup-to-drop and a drop-to-pickup swing; every tick of it
    # draws 900/2^26 J over the round figure.
    assert remaining[84] - remaining[160] == pytest.approx(66900 + 76 * 900 / 2**26, abs=1e-6)
    draws = [remaining[t - 1] - remaining[t] for t in range(86, 124)]
    assert draws[:5] == pytest.approx([2400] * 5, abs=1e-4)
    assert draws[5:] == pytest.approx([650] * 33, abs=1e-4)


def test_a_waiting_inserter_draws_nothing(sim):
    # order_a: the hand reaches the empty chest at t=9 and waits for the drill.
    energy = {sim[t]["order_a"]["ins"]["energy"] for t in range(9, 244)}
    remaining = {sim[t]["order_a"]["ins"]["remaining"] for t in range(9, 244)}
    assert len(energy) == 1 and len(remaining) == 1


def test_waiting_inserters_react_a_tick_late(sim):
    for key in ("order_a", "order_b"):
        assert _changes(sim, key, lambda r: r["src"])[1] == (243, 1)
        assert _changes(sim, key, lambda r: r["ins"]["held"])[1] == (244, "iron-ore")
    assert _changes(sim, "fout", lambda s: s["held"])[1] == (195, "iron-plate")


def test_fill_limits(sim):
    assert max(s["fill_ore_furnace"]["src"] for s in sim) == 2
    assert max(s["fill_coal_furnace"]["fuel"] for s in sim) == 5
    assert max(s["hot"]["furnace"]["src"] for s in sim) == 2
    # It waits empty-handed at the pickup, reading `waiting_for_source_items`.
    last = sim[TICKS]["fill_ore"]
    assert last["held"] is None and last["status"] == "waiting_for_source_items"


def test_fuel_runs_out(sim):
    wood = _changes(sim, "wood", lambda s: s["status"])
    assert wood[-1] == (2826, "no_fuel")
    assert sim[551]["wood"]["burning"] == "wood" and sim[551]["wood"]["wood"] == 0
    assert sim[2826]["wood_dst"] == 37 and sim[TICKS]["fuel_dst"] == 39


def test_drop_lanes(sim):
    # East: lane 2 at 128; west: lane 1 at 128; south, on the centre line:
    # lane 2 at 77. Each read once moved.
    assert sim[47]["drop1"]["belt"][0] == [[], [["iron-plate", 120]]]
    assert sim[47]["drop2"]["belt"][0] == [[["iron-plate", 120]], []]
    assert sim[47]["drop3"]["belt"][0] == [[], [["iron-plate", 69]]]
    for key in ("drop1", "drop2", "drop3"):
        status = _changes(sim, key, lambda r: r["ins"]["status"])
        assert (883, "waiting_for_space_in_destination") in status


def test_self_refuel(sim):
    coal = _changes(sim, "self", lambda r: r["ins"]["coal"])
    assert coal[:2] == [(0, 0), (37, 1)]
    assert _changes(sim, "self", lambda r: r["dst"])[:2] == [(0, 0), (103, 1)]


# ------------------------------------------------------- not from the probe


def _world() -> Sim:
    sim = Sim(water=[])
    sim.reset({"character": {"position": [0.5, 0.5]}})
    return sim


def test_place_mine_and_transfer_through_the_character():
    sim = Sim(water=[])
    sim.reset(
        {
            "character": {
                "position": [0.5, 0.5],
                "inventory": {"wooden-chest": 1, "transport-belt": 2, "burner-inserter": 1,
                              "coal": 10, "iron-plate": 20},
            }
        }
    )  # fmt: skip
    # A belt goes down under the character, which walks over belts.
    sim.step("place_at", {"item": "transport-belt", "position": [0.5, 0.5], "direction": "east"})
    assert sim.action_outcome() == ("completed", None)
    sim.step("place_at", {"item": "wooden-chest", "position": [2.5, 0.5]})
    chest = sim.env.act.handle
    # Facing west: it waits on the empty belt and drops into the chest.
    sim.step("place_at", {"item": "burner-inserter", "position": [1.5, 0.5], "direction": "west"})
    inserter = sim.env.act.handle
    sim.step("give_to", {"to": f"h{chest}", "item": "iron-plate", "count": 20})
    assert sim.action_outcome() == ("completed", None)
    sim.step("give_to", {"to": f"h{inserter}", "item": "coal", "count": 2})
    assert sim.action_outcome() == ("completed", None)
    sim.step("take_from", {"from": f"h{chest}", "item": "iron-plate", "count": 5})
    assert sim.inventory()["iron-plate"] == 5
    sim.step("mine_at", {"handle": f"h{chest}", "count": 1}, ticks=60)
    assert sim.inventory()["iron-plate"] == 20 and sim.inventory()["wooden-chest"] == 1
    hidden = {e["name"]: e for e in sim.hidden()["entities"]}
    assert hidden["burner-inserter"]["inventories"]["fuel"]["stacks"] == [[1, "coal", 2]]
    assert hidden["transport-belt"]["belt_shape"] == "straight"


def test_a_left_turn_mirrors_the_right():
    """Not measured: a left turn is taken to be the right turn mirrored."""
    sim = _world()
    env = sim.env
    first = lib.fsim_add_entity(env, lib.K_BELT, 10 * 256 + 128, 128, 4)
    turn = lib.fsim_add_entity(env, lib.K_BELT, 11 * 256 + 128, 128, 0)
    lib.fsim_refresh(env)
    assert env.entities[turn].shape == lib.BELT_LEFT
    assert list(env.entities[turn].lane_length) == [106, 295]
    assert env.entities[first].lane_next[1] == turn * 2 + 1
    shapes = {e["position"][0] // 256: e["belt_shape"] for e in sim.hidden()["entities"]}
    assert shapes == {10: "straight", 11: "left"}


def test_a_closed_loop_keeps_its_items():
    """Not measured: four right turns in a square circulate their items."""
    sim = _world()
    env = sim.env
    square = [(10, 0, 4), (11, 0, 8), (11, 1, 12), (10, 1, 0)]
    belts = [lib.fsim_add_entity(env, lib.K_BELT, x * 256 + 128, y * 256 + 128, d)
             for x, y, d in square]  # fmt: skip
    plate = ITEM_IDS["iron-plate"]
    for b in belts:
        assert lib.fsim_belt_insert(env, b, 0, 100, plate)
    lib.fsim_advance(env, 1)
    assert env.chain_count == 2 and env.chain_size[0] < 0  # one loop per lane

    def positions():
        return [(b, it.pos) for b in belts for it in env.entities[b].lanes[0].items[
            0 : env.entities[b].lanes[0].count]]  # fmt: skip

    before = positions()
    lib.fsim_advance(env, 37)
    after = positions()
    assert len(after) == len(before) == 4 and after != before
