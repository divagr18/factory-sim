"""The character on belts, and what mining a chest, a belt or an inserter returns.

FactorioRL's tools/probe_logistics2.py teleports the engine's character onto
belts and reads it every tick (tests/golden/sim-mechanics-m4-logistics2.json.xz,
`character`): standing and walking on a straight belt, on its lane lines, at a
belt end and carried into a chest. The simulator is put on the same belts at
the same places and must read the same position every tick. The turns it was
also put on (t=130..199 and t=500 on) are not compared: carriage round a turn
is not modelled (fsim.c, carry_character).

Mining: the probe mined a chest, a loaded belt and two inserters, one holding
an item, and read the character's slots; the order is asserted here with the
simulator's own items, since several of the probe's are not in its item set.
"""

from __future__ import annotations

import json
import lzma
from pathlib import Path

import pytest

from fsim import ITEM_IDS, ITEM_NAMES, Sim, lib

GOLDEN = Path(__file__).resolve().parent / "golden" / "sim-mechanics-m4-logistics2.json.xz"
EVIDENCE = json.loads(lzma.decompress(GOLDEN.read_bytes()))
OX, OY = EVIDENCE["origin"]
N, E, S, W = 0, 4, 8, 12

#: (tick, what) the probe did to the character, after that tick ran.
PLAN = [
    (40, ("walk", E)), (60, ("walk", W)), (90, ("stop",)), (100, ("walk", N)), (115, ("stop",)),
    (130, ("teleport", "char_turn")), (200, ("teleport", "lane1")), (240, ("teleport", "lane2")),
    (280, ("teleport", "last")), (340, ("teleport", "far")), (420, ("teleport", "chestend")),
]  # fmt: skip
#: Ticks compared: the straight belts, the belt end and the chest.
COMPARED = [*range(0, 130), *range(200, 340), *range(420, 500)]


def tile(base: str, x: int, y: int) -> tuple[int, int]:
    bx, by = EVIDENCE["bases"][base]
    return (OX + bx + x) * 256 + 128, (OY + by + y) * 256 + 128


def _series():
    out, at, cur = [], 0, None
    rows = EVIDENCE["character"]
    for t in range(EVIDENCE["ticks"] + 1):
        while at < len(rows) and rows[at][0] <= t:
            cur = rows[at][1]
            at += 1
        out.append(cur)
    return out


@pytest.fixture(scope="module")
def run():
    sim = Sim(water=[])
    sim.reset({"character": {"position": [0.5, 0.5]}})
    env = sim.env

    def belt(base, x, y, d):
        assert lib.fsim_add_entity(env, lib.K_BELT, *tile(base, x, y), d) >= 0

    for i in range(12):
        belt("char_row", i, 0, E)
    for i in range(9, -1, -1):
        belt("char_north", 0, i, N)
    belt("char_end", 0, 0, E)
    belt("char_end", 1, 0, E)
    belt("char_chestend", 0, 0, E)
    belt("char_chestend", 1, 0, E)
    assert lib.fsim_add_entity(env, lib.K_CHEST, *tile("char_chestend", 2, 0), 0) >= 0
    lib.fsim_refresh(env)
    north = tile("char_north", 0, 8)
    spots = {
        "lane1": (north[0] - 60, north[1]),
        "lane2": (north[0] + 60, north[1]),
        "last": tile("char_end", 1, 0),
        "chestend": tile("char_chestend", 0, 0),
        "char_turn": (0, 0),  # somewhere with no belt; the turn is not compared
        "far": (0, 0),
    }
    env.char_pos.x, env.char_pos.y = tile("char_row", 3, 0)
    plan = dict(PLAN)
    positions = [(env.char_pos.x, env.char_pos.y)]
    for t in range(1, 500):
        lib.fsim_advance(env, 1)
        step = plan.get(t)
        if step and step[0] == "walk":
            env.walk_set, env.walk_set_dir = 1, step[1]
        elif step and step[0] == "stop":
            env.walk_set = 0
        elif step and step[0] == "teleport":
            env.char_pos.x, env.char_pos.y = spots[step[1]]
        positions.append((env.char_pos.x, env.char_pos.y))
    return positions


@pytest.mark.parametrize("t", COMPARED)
def test_position_on_belts_matches_the_engine(run, t):
    engine = _series()[t]["p"]
    assert list(run[t]) == engine


def test_carried_eight_a_tick_and_walking_adds_to_it(run):
    steps = {t: run[t][0] - run[t - 1][0] for t in range(1, 116)}
    assert {steps[t] for t in range(1, 41)} == {8}  # standing
    assert {steps[t] for t in range(41, 61)} == {46}  # walking with the belt
    assert {steps[t] for t in range(61, 91)} == {-30}  # against it


def _world_with(entities) -> tuple[Sim, dict]:
    sim = Sim(water=[])
    sim.reset({"character": {"position": [0.5, 0.5]}})
    env = sim.env
    made = {}
    for key, kind, x, y, d in entities:
        made[key] = lib.fsim_add_entity(env, kind, x * 256 + 128, y * 256 + 128, d)
    lib.fsim_refresh(env)
    return sim, made


def _mine(sim: Sim, index: int) -> list:
    env = sim.env
    env.mining, env.mining_target_entity = 1, index
    env.mined_kind, env.mined_index = 1, index
    env.mining_seconds = env.mining_progress = 0
    env.mining_pos.x, env.mining_pos.y = env.entities[index].pos.x, env.entities[index].pos.y
    for _ in range(40):
        lib.fsim_advance(env, 1)
        if not env.entities[index].alive:
            break
    env.mining = 0
    return [(ITEM_NAMES[s.item], s.count) for s in env.main[0 : lib.FSIM_MAIN_SLOTS] if s.count]


def test_mining_returns_contents_then_the_entity():
    """Chest: its slots in order, then the chest. Belt: lane 1 front to back,
    lane 2 front to back, then the belt. Inserter: its fuel, the inserter, then
    what it held (probe_logistics2 `mine`)."""
    sim, made = _world_with([
        ("chest", lib.K_CHEST, 2, 0, 0),
        ("belt", lib.K_BELT, 4, 0, E),
        ("ins", lib.K_INSERTER, 0, 3, N),
    ])  # fmt: skip
    env = sim.env
    chest = env.entities[made["chest"]].chest
    for slot, name, count in ((0, "iron-plate", 50), (1, "iron-ore", 30), (2, "copper-plate", 10),
                              (4, "stone", 5), (15, "coal", 7)):  # fmt: skip
        chest[slot].item, chest[slot].count = ITEM_IDS[name], count
    order = _mine(sim, made["chest"])
    assert order == [("iron-plate", 50), ("iron-ore", 30), ("copper-plate", 10), ("stone", 5),
                     ("coal", 7), ("wooden-chest", 1)]  # fmt: skip
    b = made["belt"]
    for lane, pos, name in ((0, 0, "wood"), (0, 64, "iron-gear-wheel"), (0, 128, "copper-ore"),
                            (1, 0, "burner-mining-drill"), (1, 64, "stone-wall"),
                            (1, 128, "small-electric-pole")):  # fmt: skip
        assert lib.fsim_belt_insert(env, b, lane, pos, ITEM_IDS[name])
    order = _mine(sim, b)[6:]
    assert order == [("wood", 1), ("iron-gear-wheel", 1), ("copper-ore", 1),
                     ("burner-mining-drill", 1), ("stone-wall", 1), ("small-electric-pole", 1),
                     ("transport-belt", 1)]  # fmt: skip
    ins = made["ins"]
    lib.fsim_entity_insert(env, ins, ITEM_IDS["wood"], 3)
    lib.fsim_inserter_hold(env, ins, ITEM_IDS["stone-furnace"])
    order = _mine(sim, ins)
    assert order[-2:] == [("burner-inserter", 1), ("stone-furnace", 1)]
    assert ("wood", 4) in order  # the fuel joined the belt's wood before the inserter came back
