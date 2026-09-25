"""Hand-mining and reach in the v3 profile, against the engine.

User decisions 1 and 2 (2026-09-25, FactorioRL docs/sim-logistics.md): v3 can
hand-mine a resource tile (`mine_tile`, op 22, the tile named by placement
slot), and its mask is legal exactly where the game accepts. The engine's
numbers come from FactorioRL's tools/probe_handmine.py:

- `handmine-reach.json.xz`: `can_reach_entity` for seven entity types over
  16 x 16 sub-tile character positions x 25 x 25 whole-tile offsets. A v3
  target row is legal exactly where the engine says reachable.
- The mining mechanics themselves (121 ticks an ore, the spill with a full
  inventory, a covered tile mining what covers it) are pinned tick by tick by
  the `hand_mine_rules` parity trace (tests/test_parity.py).
"""

from __future__ import annotations

import json
import lzma
import math

import pytest

from fsim import ITEM_IDS, ffi, lib, scenes
from fsim.parity import GOLDEN
from fsim.rl import NVEC3, RlEnv

OPS = NVEC3[0]
TARGETS = slice(OPS, OPS + NVEC3[1])
PLACEMENTS = slice(OPS + NVEC3[1], OPS + NVEC3[1] + NVEC3[2])
OP_MINE_TILE = 22
OP_MINE_AT = 13
SIDE = 15
TILE = 256

KINDS = {
    "wooden-chest": lib.K_CHEST,
    "transport-belt": lib.K_BELT,
    "burner-inserter": lib.K_INSERTER,
    "stone-wall": lib.K_WALL,
    "stone-furnace": lib.K_FURNACE,
    "burner-mining-drill": lib.K_DRILL,
    "item-on-ground": lib.K_PILE,
}


def _bare(seed: int = 0) -> RlEnv:
    """A v3 env with nothing built and no resource near the origin."""
    _, scene = scenes.sample("construct_smelting_line", "train", seed)
    scene = dict(scene, entities=[], resources=[])
    env = RlEnv()
    env.reset("construct_smelting_line", scene, action_space="v3")
    return env


def test_target_rows_are_legal_exactly_where_the_engine_reaches():
    doc = json.loads(lzma.decompress((GOLDEN / "handmine-reach.json.xz").read_bytes()))
    sub, off = doc["sub"], doc["offsets"]
    checked = 0
    for spec in doc["types"]:
        env = _bare()
        rl = env.rl
        ex, ey = float(spec["position"][0]), float(spec["position"][1])
        index = lib.fsim_add_entity(rl.env, KINDS[spec["name"]], round(ex * TILE),
                                    round(ey * TILE), 0)  # fmt: skip
        if spec["name"] == "item-on-ground":
            rl.env.entities[index].pile.item = ITEM_IDS["iron-plate"]
            rl.env.entities[index].pile.count = 1
        # Every fourth sub-tile position on each axis: 16 of 256, all offsets.
        for si, bits in enumerate(spec["grid"]):
            sx, sy = si % sub, si // sub
            if sx % 4 or sy % 4:
                continue
            k = 0
            for dy in range(-off, off + 1):
                for dx in range(-off, off + 1):
                    cx = (math.floor(ex) + dx) * TILE + sx * TILE // sub
                    cy = (math.floor(ey) + dy) * TILE + sy * TILE // sub
                    rl.env.char_pos.x, rl.env.char_pos.y = cx, cy
                    lib.fsim_observe(rl.env)
                    seen = rl.env.seen_count > 0
                    # `mine_at`'s own row of the per-operation masks: target 1.
                    legal = bool(env.op_masks()[OP_MINE_AT][1]) if seen else False
                    # A ground pile is in reach like the rest, but `mine`
                    # refuses it: its prototype yields nothing.
                    reached = bits[k] == "1" and spec["name"] != "item-on-ground"
                    assert legal == reached, (spec["name"], dx, dy, sx, sy)
                    k += 1
                    checked += 1
    assert checked == 7 * 16 * (2 * off + 1) ** 2


def _ore_env(amount: int = 50) -> RlEnv:
    _, scene = scenes.sample("construct_smelting_line", "train", 0)
    cx, cy = (math.floor(v) for v in scene["character"]["position"])
    scene = dict(scene, entities=[],
                 resources=[{"name": "coal", "position": [cx + 2.5, cy + 0.5], "amount": amount},
                            {"name": "coal", "position": [cx + 4.5, cy + 0.5], "amount": amount}],
                 character={**scene["character"], "position": [cx + 0.5, cy + 0.5]})  # fmt: skip
    env = RlEnv()
    env.reset("construct_smelting_line", scene, action_space="v3")
    return env


def _slot(dx: int, dy: int) -> int:
    return (dx + 7) * SIDE + dy + 7 + 1


def test_mine_tile_mines_the_resource_under_its_slot():
    env = _ore_env()
    assert env.mask[OP_MINE_TILE]
    assert env.mask[PLACEMENTS][_slot(2, 0)]
    # Four tiles away: beyond resource reach. The slot is still legal in the
    # shared dimension (a free tile to build on), so decoding refuses it.
    action = ffi.new("fsim_action *")
    far = ffi.new("int32_t[6]", [OP_MINE_TILE, 0, _slot(4, 0), 0, 0, 1])
    assert lib.fsim_rl_decode(env.rl, far, action) == 1
    coal = ITEM_IDS["coal"]
    before = sum(s.count for s in env.rl.env.main if s.item == coal)
    env.step([OP_MINE_TILE, 0, _slot(2, 0), 0, 0, 2])  # five coal
    assert not env.rl.decode_failure
    for _ in range(25):
        env.step([21, 0, 0, 0, 0, 0])
    held = sum(s.count for s in env.rl.env.main if s.item == coal) - before
    # 121 ticks the first, 120 each after: five in 601 ticks, then it stops.
    assert held == 5 and not env.rl.env.mining


def test_a_covered_tile_mines_the_entity_on_it():
    env = _ore_env()
    chest = ITEM_IDS["wooden-chest"]
    env.rl.env.main[0].item, env.rl.env.main[0].count = chest, 1
    lib.fsim_observe(env.rl.env)
    env.observe()
    env.step([12, 0, _slot(2, 0), 1, 12, 0])  # the chest onto the ore tile
    assert env.rl.env.built[chest] == 1
    coal = ITEM_IDS["coal"]
    before = sum(s.count for s in env.rl.env.main if s.item == coal)
    env.step([OP_MINE_TILE, 0, _slot(2, 0), 0, 0, 1])
    for _ in range(8):
        env.step([21, 0, 0, 0, 0, 0])
    # The chest came back, no coal came, and the mine is still running.
    assert sum(s.count for s in env.rl.env.main if s.item == chest) == 1
    assert sum(s.count for s in env.rl.env.main if s.item == coal) == before
    assert env.rl.env.mining_stalled and env.rl.env.mining


@pytest.mark.parametrize("sub", [(0.1, 0.1), (0.9, 0.5)])
def test_placement_slots_need_build_distance(sub):
    env = _bare()
    rl = env.rl
    here = (math.floor(rl.env.char_pos.x / TILE), math.floor(rl.env.char_pos.y / TILE))
    rl.env.char_pos.x = round((here[0] + sub[0]) * TILE)
    rl.env.char_pos.y = round((here[1] + sub[1]) * TILE)
    lib.fsim_observe(rl.env)
    _, mask = env.observe3()
    px, py = rl.env.char_pos.x / TILE, rl.env.char_pos.y / TILE
    for dx in range(-7, 8):
        for dy in range(-7, 8):
            if (dx, dy) == (0, 0):
                continue
            cx, cy = here[0] + dx + 0.5, here[1] + dy + 0.5
            near = math.sqrt((cx - px) ** 2 + (cy - py) ** 2) <= 10
            assert bool(mask[PLACEMENTS][_slot(dx, dy)]) == near, (dx, dy)
