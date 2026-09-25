"""The v3 profile: 96 entity rows of 32 features, 18 items, public markers in the
goal, and a 15x15 placement window (`action_space="v3"`)."""

from __future__ import annotations

import hashlib
import random

import numpy as np
import pytest

from fsim import ITEM_IDS, expert, ffi, lib, scenes
from fsim.obsview import ITEMS_V3, EntityV3, ObsView
from fsim.parity import GOLDEN
from fsim.rl import NVEC, NVEC3, RlEnv
from fsim.trace import read_trace

OPS = NVEC3[0]
TARGETS = slice(OPS, OPS + NVEC3[1])
PLACEMENTS = slice(OPS + NVEC3[1], OPS + NVEC3[1] + NVEC3[2])
ITEM_DIM = slice(sum(NVEC3[:4]), sum(NVEC3[:5]))
CHAIN = "logistics_smelting_chain"
OP_MINE = 13


def digest(obs: dict, mask) -> str:
    h = hashlib.sha256()
    for key in sorted(obs):
        h.update(np.ascontiguousarray(obs[key]).tobytes())
    h.update(np.asarray(mask).tobytes())
    return h.hexdigest()


def chain_env(decisions: int, entity_cap: int = 96) -> RlEnv:
    """The logistics chain, run on its recorded v1 vectors for `decisions` decisions."""
    header, records = read_trace(GOLDEN / f"{CHAIN}.jsonl.xz")
    env = RlEnv()
    env.reset(
        header["task"],
        header["blueprint"],
        decision_ticks=header["decision_ticks"],
        max_steps=header["max_decision_steps"],
        construction_tick_limit=header["construction_tick_limit"],
        entity_cap=entity_cap,
    )
    for record in records[1 : 1 + decisions]:
        env.step(record["transition"]["action"]["vector"])
    return env


def crowded_scene(n: int = 60) -> dict:
    _, scene = scenes.sample("construct_smelting_line", "train", 0)
    x0, y0 = scene["character"]["position"]
    scene = dict(scene)
    scene["entities"] = [
        {"name": "wooden-chest", "position": [round(x0) + 3 + k % 10 + 0.5,
                                              round(y0) + 3 + k // 10 + 0.5],
         "force": "player", "direction": "north"}
        for k in range(n)
    ]  # fmt: skip
    return scene


def test_shapes():
    _, scene = scenes.sample("construct_smelting_line", "train", 1)
    env = RlEnv()
    obs = env.reset("construct_smelting_line", scene, action_space="v3")
    assert obs["grid"].shape == (6, 65, 65)
    assert obs["entities"].shape == (96, 32)
    assert obs["entity_mask"].shape == (96,)
    assert obs["self"].shape == (13,)
    assert obs["inventory"].shape == (18,)
    assert obs["goal"].shape == (30,)
    assert env.mask.shape == (sum(NVEC3),) == (lib.RL3_MASK_SIZE,)
    assert NVEC3 == (25, 97, 226, 5, 19, 4)
    assert env.op_masks().shape == (25, sum(NVEC3[1:])) == (25, lib.RL3_ARG_WIDTH)


def test_v1_unchanged_after_a_v3_episode_in_the_same_env():
    _, scene = scenes.sample("construct_smelting_line", "train", 2)
    rng = random.Random(0)
    moves = [[rng.randrange(12), 0, 0, 0, 0, 0] for _ in range(20)]

    def run(env):
        out = [digest(env.reset("construct_smelting_line", scene), env.mask)]
        for v in moves:
            out.append(digest(env.step(v)[0], env.mask))
        return out

    fresh = run(RlEnv())
    env = RlEnv()
    env.reset("construct_smelting_line", scene, action_space="v3")
    for v in moves[:5]:
        env.step(v)
    assert run(env) == fresh
    assert env.obs["entities"].shape == (32, 16)


def test_entity_cap_is_48_under_v1_and_96_under_v3():
    scene = crowded_scene(60)
    env = RlEnv()
    env.reset("construct_smelting_line", scene, action_space="v3")
    assert env.rl.env.seen_count == 60
    assert int(env.obs["entity_mask"].sum()) == 60
    env.reset("construct_smelting_line", scene)
    assert env.rl.env.seen_count == 48
    assert int(env.obs["entity_mask"].sum()) == 32


def test_v3_rows_and_goal_extend_v1():
    """Features 0-15 of the first rows, the self and grid planes and the first
    12 goal slots are v1's, computed the same way."""
    env = chain_env(40)
    v1 = {k: np.array(v) for k, v in env.observe().items()}
    obs, _ = env.observe3()
    n = int(v1["entity_mask"].sum())
    assert 0 < n <= 32
    assert np.array_equal(obs["entities"][:n, :16], v1["entities"][:n])
    assert np.array_equal(obs["grid"], v1["grid"])
    assert np.array_equal(obs["self"][:12], v1["self"])
    free = sum(1 for i in range(lib.FSIM_MAIN_SLOTS) if env.rl.env.main[i].count == 0)
    assert obs["self"][12] == np.float32(free / lib.FSIM_MAIN_SLOTS)
    assert np.array_equal(obs["goal"][:12], v1["goal"])
    assert np.array_equal(obs["inventory"][:14], v1["inventory"])
    assert not obs["inventory"][14:].any()


def test_placement_slot_names_a_fixed_tile_and_occupied_slots_are_masked():
    env = chain_env(60)
    rl = env.rl
    rl.task.action_space = lib.ACTION_SPACE_V3
    # A furnace to place, so `place_at` is legal and its own row of the
    # per-operation masks shows the tiles (placement: after the 97 targets).
    rl.env.main[0].item, rl.env.main[0].count = ITEM_IDS["stone-furnace"], 1
    row = env.op_masks()[expert.OP_PLACE]
    here = (rl.env.char_pos.x // 256, rl.env.char_pos.y // 256)
    occupied = set()
    for k in range(rl.env.seen_count):
        e = rl.env.entities[rl.env.seen[k].entity]
        if lib.fsim_kind_flags(e.kind) & lib.KF_COLLIDES:
            occupied.add((e.pos.x // 256, e.pos.y // 256))
    mask = row[NVEC3[1] : NVEC3[1] + NVEC3[2]]
    assert mask[0] == 0, "place_at reads the placement: its sentinel is not legal"
    assert row[0] == 1 and not row[1 : NVEC3[1]].any(), "nor a target: sentinel only"
    action = ffi.new("fsim_action *")
    seen_occupied = 0
    for slot in range(225):
        dx, dy = slot // 15 - 7, slot % 15 - 7
        tile = (here[0] + dx, here[1] + dy)
        legal = tile != here and tile not in occupied
        seen_occupied += not legal
        assert bool(mask[slot + 1]) == legal, (dx, dy)
        # Item 1 (iron ore) is not held, so a legal tile still fails on the
        # item; an occupied one fails first either way.
        vector = ffi.new("int32_t[6]", [expert.OP_PLACE, 0, slot + 1, 1, 1, 0])
        assert lib.fsim_rl_decode(rl, vector, action) == 1
    assert seen_occupied > 5


def test_placement_decodes_to_its_tile():
    _, scene = scenes.sample("construct_smelting_line", "train", 4)
    env = RlEnv()
    env.reset("construct_smelting_line", scene, action_space="v3")
    rl = env.rl
    here = (rl.env.char_pos.x // 256, rl.env.char_pos.y // 256)
    furnace = ITEMS_V3.index("stone-furnace") + 1
    assert env.mask[ITEM_DIM][furnace] == 1
    action = ffi.new("fsim_action *")
    for slot in (0, 7, 112, 200, 224):
        if not env.mask[PLACEMENTS][slot + 1]:
            continue
        vector = ffi.new("int32_t[6]", [expert.OP_PLACE, 0, slot + 1, 2, furnace, 0])
        assert lib.fsim_rl_decode(rl, vector, action) == 0
        assert (action.position.x // 256, action.position.y // 256) == (
            here[0] + slot // 15 - 7,
            here[1] + slot % 15 - 7,
        )
        assert action.direction == 1
    # The Stage-2 items are never held, so they are never legal.
    assert not env.mask[ITEM_DIM][15:].any()


def test_a_remembered_row_is_masked_out_of_decoding():
    _, scene = scenes.sample("construct_smelting_line", "train", 5)
    env = RlEnv()
    env.reset("construct_smelting_line", scene, action_space="v3")
    rl = env.rl
    # Place a furnace, then walk far enough that it is only remembered.
    furnace = ITEMS_V3.index("stone-furnace") + 1
    slot = (2 + 7) * 15 + 7  # two tiles east of the character
    assert env.mask[PLACEMENTS][slot + 1]
    env.step([expert.OP_PLACE, 0, slot + 1, 1, furnace, 0])
    assert rl.env.built[lib.IT_STONE_FURNACE] == 1
    for direction in (0, 1, 2, 3):
        for _ in range(12):
            env.step([direction, 0, 0, 0, 0, 0])
        if rl.env.remembered_count and not rl.env.seen_count:
            break
    assert rl.env.remembered_count >= 1 and rl.env.seen_count == 0
    view = ObsView(env.obs)
    rows = view.entities()
    assert rows and all(e.remembered for e in rows)
    action = ffi.new("fsim_action *")
    vector = ffi.new("int32_t[6]", [OP_MINE, rows[0].row + 1, 0, 0, 0, 0])
    assert lib.fsim_rl_decode(rl, vector, action) == 1
    # A remembered row is masked (user decision 2026-09-25: legal exactly where
    # the game accepts), and so are the verbs that need a row.
    assert env.mask[TARGETS][1] == 0
    assert not env.mask[OP_MINE] and not env.mask[expert.OP_GIVE]


def test_logistics_features_match_the_simulator_state():
    env = chain_env(0)
    header, records = read_trace(GOLDEN / f"{CHAIN}.jsonl.xz")
    held_seen = lanes_seen = 0
    for record in records[1:]:
        env.step(record["transition"]["action"]["vector"])
        if env.rl.done:
            break
        obs, _ = env.observe3()
        rl = env.rl
        view = ObsView(obs)
        by_pos = {}
        for k in range(rl.env.seen_count):
            e = rl.env.entities[rl.env.seen[k].entity]
            by_pos[(e.pos.x, e.pos.y)] = e
        for row in view.entities():
            assert isinstance(row, EntityV3)
            e = by_pos[(round(row.x * 256), round(row.y * 256))]
            if row.kind == "transport-belt":
                assert row.lanes == (min(e.lanes[0].count, 8), min(e.lanes[1].count, 8))
                assert row.shape == ("straight", "left", "right")[e.shape]
                lanes_seen += sum(row.lanes)
            elif row.kind == "inserter":
                assert (row.held is not None) == (e.held != lib.IT_NONE)
                held_seen += row.held is not None
                # Pickup one tile along the facing, drop 1.2 tiles the other way.
                fx, fy = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}[row.facing]
                assert row.pickup == (row.x + fx, row.y + fy)
                assert row.drop == (row.x - fx * 307 / 256, row.y - fy * 307 / 256)
            elif row.kind == "mining-drill":
                assert row.drop is not None and row.pickup is None
            elif row.kind == "container":
                assert (row.item is None) == (row.contents == 0)
    assert held_seen > 0 and lanes_seen > 0


def test_markers_follow_the_public_marker_order():
    _, scene = scenes.sample("construct_smelting_line", "train", 6)
    scene = dict(scene)
    scene["markers"] = {**scene["markers"], "output_chest": [40.5, -70.5], "hidden": [1, 1]}
    # "unplaced" is public but has no position: its slot stays empty and the
    # markers after it keep their own slots.
    scene["public_markers"] = ["output_chest", "unplaced", "patch"]
    env = RlEnv()
    obs = env.reset("construct_smelting_line", scene, action_space="v3")
    view = ObsView(obs)
    markers = view.markers()
    assert len(markers) == 6 and markers[3:] == [None] * 3
    assert markers[0] == (40.5, -70.5)
    assert markers[1] is None
    assert markers[2] == tuple(float(v) for v in scene["markers"]["patch"])
    x, y = view.char_pos
    assert obs["goal"][12] == np.float32(np.clip((40.5 - x) / 128, -1, 1))
    assert obs["goal"][14] == 1.0 and obs["goal"][17] == 0.0 and obs["goal"][20] == 1.0


def test_a_marker_on_a_scene_entity_follows_it_until_it_is_gone():
    _, scene = scenes.sample("construct_smelting_line", "train", 8)
    x0, y0 = scene["character"]["position"]
    chest = [round(x0) + 2 + 0.5, round(y0) + 0.5]
    scene = dict(scene)
    scene["entities"] = [
        {"name": "stone-wall", "position": [round(x0) - 20 + 0.5, round(y0) + 0.5]},
        {"name": "wooden-chest", "position": chest, "marker": "output"},
    ]
    # A scene marker of the same name is the fallback once the chest is gone.
    scene["markers"] = {**scene["markers"], "output": [100.5, 100.5]}
    scene["public_markers"] = ["patch", "output"]
    env = RlEnv()
    view = ObsView(env.reset("construct_smelting_line", scene, action_space="v3"))
    assert view.markers()[1] == tuple(chest)
    row = next(e for e in view.entities() if e.kind == "container")
    for _ in range(40):
        env.step([OP_MINE, row.row + 1, 0, 0, 0, 0])
        if not any(e.kind == "container" for e in ObsView(env.obs).entities()):
            break
    view = ObsView(env.obs)
    assert not any(e.kind == "container" and not e.remembered for e in view.entities())
    assert view.markers()[1] == (100.5, 100.5)


def test_v2_views_keep_their_shape():
    _, scene = scenes.sample("construct_smelting_line", "train", 7)
    env = RlEnv()
    obs = env.reset("construct_smelting_line", scene, action_space="v2")
    assert obs["entities"].shape == (32, 16) and env.mask.shape == (sum(NVEC),)
    view = ObsView(obs)
    assert not view.v3 and view.markers() == []
    assert len(view.inventory()) == 14


def test_vectorised_env_speaks_v3_and_refuses_the_unknown():
    """v3 runs batched now (`tests/test_vec_v3.py` pins it to this env)."""
    from fsim.vec import VecEnv

    env = VecEnv(1, "construct_smelting_line", action_space="v3")
    assert env.op_masks.shape == (1, lib.RL3_OPERATIONS, lib.RL3_ARG_WIDTH)
    env.close()
    with pytest.raises(ValueError):
        VecEnv(1, "construct_smelting_line", action_space="v4")
