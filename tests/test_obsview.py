"""The observation decoder against the simulator state it was encoded from."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fsim import expert, ffi, lib, scenes
from fsim.obsview import ITEMS, KINDS, ObsView
from fsim.rl import RlEnv
from fsim.vec import obs_layout

TASK = "construct_smelting_line"
SCENES = [("train", s) for s in range(30)] + [("test", s) for s in range(25)]
KIND_OF = {lib.K_DRILL: "mining-drill", lib.K_FURNACE: "furnace", lib.K_WALL: "wall",
           lib.K_PILE: "item-entity"}  # fmt: skip
ITEM_ID = {
    "iron-ore": lib.IT_IRON_ORE, "copper-ore": lib.IT_COPPER_ORE, "coal": lib.IT_COAL,
    "stone": lib.IT_STONE, "iron-plate": lib.IT_IRON_PLATE,
    "copper-plate": lib.IT_COPPER_PLATE, "stone-furnace": lib.IT_STONE_FURNACE,
    "iron-gear-wheel": lib.IT_IRON_GEAR, "transport-belt": lib.IT_TRANSPORT_BELT,
    "wood": lib.IT_WOOD, "small-electric-pole": lib.IT_SMALL_POLE,
    "wooden-chest": lib.IT_WOODEN_CHEST, "burner-mining-drill": lib.IT_BURNER_DRILL,
    "burner-inserter": lib.IT_BURNER_INSERTER,
}  # fmt: skip


def char(env):
    return env.rl.env.char_pos.x / 256, env.rl.env.char_pos.y / 256


def true_ore(env):
    e = env.rl.env
    return {
        (e.resources[e.tiles[k].resource].tx, e.resources[e.tiles[k].resource].ty)
        for k in range(e.tile_count)
        if e.resources[e.tiles[k].resource].item == lib.IT_IRON_ORE
    }


def true_blocked(env):
    e = env.rl.env
    ox, oy = char(env)
    out = set()
    for k in range(e.blocked_count):
        bx, by = e.blocked[2 * k], e.blocked[2 * k + 1]
        if abs(round(bx - ox)) <= 32 and abs(round(by - oy)) <= 32:
            out.add((bx, by))
    return out


def true_rows(env):
    """(kind, x, y, direction or None, fuel, remembered) per entity-table row."""
    rl, e = env.rl, env.rl.env
    handles = ffi.new("int32_t[32]")
    count = lib.fsim_rl_targets(rl, handles, 32)
    seen = {e.seen[k].handle: e.entities[e.seen[k].entity] for k in range(e.seen_count)}
    memory = {e.memory[i].handle: e.memory[i] for i in range(256) if e.memory[i].used}
    rows = []
    for k in range(count):
        h = handles[k]
        if h in seen:
            ent = seen[h]
            fuel = ent.fuel.count if ent.kind in (lib.K_DRILL, lib.K_FURNACE) else 0
            rows.append((ent.kind, ent.pos.x / 256, ent.pos.y / 256, ent.direction, fuel, False))
        else:
            m = memory[h]
            rows.append((m.kind, m.pos.x / 256, m.pos.y / 256, m.direction, 0, True))
    return rows


def check(env):
    view = ObsView(env.obs)
    ox, oy = char(env)
    assert view.char_pos == (ox, oy)  # exact: a 1/256 position fits float32's mantissa
    assert view.char_tile == (math.floor(ox), math.floor(oy))

    exact = view.exact()
    truth, decoded = true_ore(env), set(view.ore_tiles("iron-ore"))
    if all(exact["ore"]):
        assert decoded == truth
    else:
        # On an ambiguous axis each tile may come with the neighbour it shares a cell with.
        assert truth <= decoded
        ax, ay = (0 if exact["ore"][0] else 1), (0 if exact["ore"][1] else 1)
        for tx, ty in decoded:
            assert any(abs(tx - x) <= ax and abs(ty - y) <= ay for x, y in truth)
    assert view.ore_tiles("copper-ore") == []
    blocked, decoded_b = true_blocked(env), set(view.blocked_tiles())
    if all(exact["blocked"]):
        assert decoded_b == blocked
    else:
        assert blocked <= decoded_b

    rows = true_rows(env)
    entities = view.entities()
    assert [e.row for e in entities] == list(range(len(rows)))
    for e, (kind, x, y, direction, fuel, remembered) in zip(entities, rows, strict=True):
        assert e.kind == KIND_OF[kind]
        assert e.remembered == remembered
        for got, want, origin in ((e.x, x, ox), (e.y, y, oy)):
            if abs(want - origin) <= 32:
                assert got == pytest.approx(want, abs=1e-9)
            else:  # clipped to the edge of the radius
                assert got == origin + 32 * math.copysign(1, want - origin)
        if kind == lib.K_DRILL:
            assert e.facing == "NESW"[direction // 4]
        else:
            assert e.facing is None
        if not remembered:
            assert e.fuel == min(fuel, 200)

    held = view.inventory()
    assert list(held) == list(ITEMS)
    for name, count in held.items():
        total = sum(
            env.rl.env.main[i].count
            for i in range(lib.FSIM_MAIN_SLOTS)
            if env.rl.env.main[i].item == ITEM_ID[name]
        )
        assert count == min(total, 200), name
    return view


@pytest.mark.parametrize(("split", "seed"), SCENES)
def test_decoded_observation_matches_the_simulator(split, seed):
    family, scene = scenes.sample(TASK, split, seed)
    env = RlEnv()
    env.reset(TASK, scene, action_space="v2")
    view = check(env)
    assert view.inventory()["coal"] == 60
    assert view.patch() == pytest.approx(tuple(scene["markers"]["patch"]), abs=1e-9)
    walls = [e for e in view.entities() if e.kind == "wall"]
    assert len(walls) == len(scene["entities"]), family

    def step(vector):
        env.step(vector)
        return bool(env.rl.done)

    expert.advance_to(env.rl, scene["markers"]["patch"], "furnace_fuelled", step)
    for _ in range(20):  # let the drill run, so fuel, contents and working move
        env.step([expert.OP_WAIT, 0, 0, 0, 0, 0])
    view = check(env)
    if not scene["entities"]:  # clear scenes: the expert built both
        kinds = sorted(e.kind for e in view.entities())
        assert kinds == ["furnace", "mining-drill"]
        drill = next(e for e in view.entities() if e.kind == "mining-drill")
        assert drill.working and drill.facing == "S"
        assert view.inventory()["coal"] == 20


def test_entity_offsets_clip_at_the_radius():
    """Beyond 32 tiles the table holds only the direction of an entity, not where it is."""
    _, scene = scenes.sample(TASK, "train", 1)
    env = RlEnv()
    env.reset(TASK, scene, action_space="v2")

    def step(vector):
        env.step(vector)
        return bool(env.rl.done)

    expert.advance_to(env.rl, scene["markers"]["patch"], "furnace", step)
    for _ in range(10):  # ~45 tiles west: the machines are remembered, and out of range
        env.step([3, 0, 0, 0, 0, 0])
    rows = true_rows(env)
    ox, _ = char(env)
    assert rows and all(x - ox > 32 for _, x, *_ in rows)
    table = env.obs["entities"][: len(rows)]
    assert np.all(table[:, 0] == 1.0)
    assert np.all(table[:, 2] == 1.0)  # distance clips too
    check(env)
    for e in ObsView(env.obs).entities():
        assert e.x == ox + 32
        assert e.remembered


def test_whole_number_position_is_reported_as_ambiguous():
    """At x = 0 two ore columns share each even cell; the decode says so and over-reports."""
    _, scene = scenes.sample(TASK, "train", 2)
    scene["character"]["position"] = [0.0, 10.3]
    env = RlEnv()
    env.reset(TASK, scene, action_space="v2")
    view = check(env)
    assert view.exact()["ore"] == (False, True)
    assert set(view.ore_tiles()) > true_ore(env)


def test_the_compact_observation_decodes_the_same():
    _, scene = scenes.sample(TASK, "train", 3)
    env = RlEnv()
    env.reset(TASK, scene, action_space="v2")
    packed = ffi.new("fsim_obs8 *")
    lib.fsim_rl_encode8(env.rl, packed)
    raw = np.frombuffer(ffi.buffer(packed), np.uint8)
    obs8 = {}
    for key, (offset, dtype, shape) in obs_layout(compact=True).items():
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        obs8[key] = raw[offset : offset + size].view(dtype).reshape(shape)
    full, compact = ObsView(env.obs), ObsView(obs8)
    assert compact.ore_tiles() == full.ore_tiles()
    assert compact.blocked_tiles() == full.blocked_tiles()
    assert compact.entities() == full.entities()
    assert compact.inventory() == full.inventory()


def test_type_indices_cover_the_encoder():
    assert set(KINDS.values()) >= set(KIND_OF.values())
