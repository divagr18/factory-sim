"""The scripted builder, and the demonstration starts it provides."""

from __future__ import annotations

import numpy as np
import pytest

from fsim import expert, ffi, lib, scenes
from fsim.rl import RlEnv
from fsim.vec import VecEnv


@pytest.mark.parametrize("seed", range(12))
def test_builder_solves_training_scenes(seed):
    family, scene = scenes.sample("construct_smelting_line", "train", seed)
    env = RlEnv()
    env.reset("construct_smelting_line", scene)
    expert.run_to_completion(env.rl, scene["markers"]["patch"])
    assert env.rl.success, family
    assert env.rl.decode_failures == 0


def test_stages_raise_the_line_potential_in_order():
    _, scene = scenes.sample("construct_smelting_line", "train", 5)
    bands = {"walked": (0.09, 0.1), "drill": (0.29, 0.3), "furnace": (0.59, 0.6),
             "drill_fuelled": (0.69, 0.7), "furnace_fuelled": (0.79, 0.8)}  # fmt: skip
    for stage, (low, high) in bands.items():
        env = RlEnv()
        env.reset("construct_smelting_line", scene)

        def step(vector, env=env):
            return env.step(vector)[2]

        taken = expert.advance_to(env.rl, scene["markers"]["patch"], stage, step)
        assert taken >= expert.STAGES.index(stage) + 1
        assert low < lib.fsim_rl_potential(env.rl) < high, stage


def test_placement_index_names_the_tile_the_decoder_places_on():
    _, scene = scenes.sample("construct_smelting_line", "train", 1)
    env = RlEnv()
    env.reset("construct_smelting_line", scene)
    rl = env.rl
    here = (rl.env.char_pos.x // 256, rl.env.char_pos.y // 256)
    action = ffi.new("fsim_action *")
    for dx, dy in ((-5, -5), (0, 1), (3, -2), (5, 5)):
        tile = (here[0] + dx, here[1] + dy)
        index = expert.placement_index(rl, *tile)
        vector = ffi.new("int32_t[6]", [expert.OP_PLACE, 0, index, 1, expert.ITEM_FURNACE, 0])
        assert lib.fsim_rl_decode(rl, vector, action) == 0
        assert (action.position.x // 256, action.position.y // 256) == tile


def test_demo_starts_are_reported_and_evaluation_is_untouched():
    env = VecEnv(16, threads=2, demo_starts=0.5)
    env.reset()
    assert {s for s in env.starts} <= {"scene", *expert.STAGES}
    assert "scene" in env.starts and len(set(env.starts)) > 1
    assert all(
        (length > 0) == (start != "scene")
        for start, length in zip(env.starts, env.lengths, strict=True)
    )
    plain = VecEnv(4, threads=1)
    plain.reset()
    assert set(plain.starts) == {"scene"} and not np.any(plain.lengths)


def test_demo_starts_are_refused_for_build_line():
    with pytest.raises(ValueError):
        VecEnv(2, "build_line", demo_starts=0.5)
