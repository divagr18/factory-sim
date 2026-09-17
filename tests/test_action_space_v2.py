"""The v2 action-space prototype: entity-table targets, a fixed placement grid."""

from __future__ import annotations

import numpy as np
import pytest

from fsim import expert, ffi, lib, scenes
from fsim.rl import NVEC, RlEnv

OPS = NVEC[0]
TARGETS = slice(OPS, OPS + NVEC[1])
PLACEMENTS = slice(OPS + NVEC[1], OPS + NVEC[1] + NVEC[2])


def built_env(seed=0, stage="furnace"):
    _, scene = scenes.sample("construct_smelting_line", "train", seed)
    env = RlEnv()
    env.reset("construct_smelting_line", scene, action_space="v2")

    def step(vector):
        return env.step(vector)[2]

    expert.advance_to(env.rl, scene["markers"]["patch"], stage, step)
    env.observe()
    return env


def test_placement_slot_names_a_fixed_tile_and_occupied_slots_are_masked():
    env = built_env()
    rl = env.rl
    here = (rl.env.char_pos.x // 256, rl.env.char_pos.y // 256)
    occupied = {(rl.env.entities[rl.env.seen[k].entity].pos.x // 256,
                 rl.env.entities[rl.env.seen[k].entity].pos.y // 256)
                for k in range(rl.env.seen_count)}  # fmt: skip
    mask = env.mask[PLACEMENTS]
    assert mask[0] == 1  # the sentinel
    action = ffi.new("fsim_action *")
    for slot in range(121):
        dx, dy = slot // 11 - 5, slot % 11 - 5
        tile = (here[0] + dx, here[1] + dy)
        legal = tile != here and tile not in occupied
        assert bool(mask[slot + 1]) == legal, (dx, dy)
        vector = ffi.new("int32_t[6]", [expert.OP_PLACE, 0, slot + 1, 1, expert.ITEM_FURNACE, 0])
        failed = lib.fsim_rl_decode(rl, vector, action)
        assert failed == (not legal)
        if legal:
            assert (action.position.x // 256, action.position.y // 256) == tile


def test_target_k_is_row_k_of_the_entity_table():
    env = built_env()
    rl = env.rl
    handles = ffi.new("int32_t[32]")
    count = lib.fsim_rl_targets(rl, handles, 32)
    rows = int(env.obs["entity_mask"].sum())
    assert count == rows == 2
    assert np.array_equal(env.mask[TARGETS][1 : 1 + rows], np.ones(rows, np.uint8))
    assert not env.mask[TARGETS][1 + rows :].any()
    ox, oy = rl.env.char_pos.x / 256, rl.env.char_pos.y / 256
    by_handle = {rl.env.seen[k].handle: rl.env.entities[rl.env.seen[k].entity]
                 for k in range(rl.env.seen_count)}  # fmt: skip
    for k in range(count):
        e = by_handle[handles[k]]
        dx = np.clip((e.pos.x / 256 - ox) / 32, -1, 1)
        dy = np.clip((e.pos.y / 256 - oy) / 32, -1, 1)
        assert env.obs["entities"][k, 0] == pytest.approx(dx, abs=1e-6)
        assert env.obs["entities"][k, 1] == pytest.approx(dy, abs=1e-6)


@pytest.mark.parametrize("seed", range(6))
def test_builder_solves_training_scenes_in_v2(seed):
    family, scene = scenes.sample("construct_smelting_line", "train", seed)
    env = RlEnv()
    env.reset("construct_smelting_line", scene, action_space="v2")
    expert.run_to_completion(env.rl, scene["markers"]["patch"])
    assert env.rl.success and env.rl.decode_failures == 0, family
