"""The scripted builder, and the demonstration starts it provides."""

from __future__ import annotations

import random

import numpy as np
import pytest

from fsim import expert, ffi, lib, scenes
from fsim.rl import RlEnv
from fsim.vec import VecEnv


@pytest.mark.parametrize("seed", range(12))
def test_builder_solves_the_training_scenes_it_demonstrates(seed):
    """The builder walks in straight lines, so it only claims the clear scenes.

    `VecEnv` starts an obstructed scene from scratch for the same reason; what
    matters there is that a build site exists at all, which
    `test_the_new_training_families_are_buildable` checks.
    """
    family, scene = scenes.sample("construct_smelting_line", "train", seed)
    if scene["entities"]:
        pytest.skip(f"{family} has something in the way")
    env = RlEnv()
    env.reset("construct_smelting_line", scene)
    expert.run_to_completion(env.rl, scene["markers"]["patch"])
    assert env.rl.success, family
    assert env.rl.decode_failures == 0


@pytest.mark.parametrize("seed", range(6))
def test_every_layout_the_builder_may_draw_builds_a_line(seed):
    """The randomised build site is not a source of broken demonstrations."""
    family, scene = scenes.sample("construct_smelting_line", "train", seed)
    env = RlEnv()
    env.reset("construct_smelting_line", scene)
    found = expert.layouts(env.rl, scene["markers"]["patch"])
    assert len(found) >= 8, (family, len(found))
    assert found[0] == (
        (int(scene["markers"]["patch"][0] // 1), int(scene["markers"]["patch"][1] // 1)),
        0,
    ), "the canonical layout comes first"
    for layout in found:
        env = RlEnv()
        env.reset("construct_smelting_line", scene)
        expert.run_to_completion(env.rl, scene["markers"]["patch"], layout=layout)
        assert env.rl.success, (family, layout)
        assert env.rl.decode_failures == 0, (family, layout)


def test_all_four_turns_are_available_and_distinct():
    _, scene = scenes.sample("construct_smelting_line", "train", 3)
    env = RlEnv()
    env.reset("construct_smelting_line", scene)
    found = expert.layouts(env.rl, scene["markers"]["patch"])
    assert {quarters for _, quarters in found} == {0, 1, 2, 3}
    patch = scene["markers"]["patch"]
    anchor = found[0][0]
    # The four turns of one anchor put the builder on four different sides of
    # the drill, facing the way the drill's output must travel.
    sides = set()
    for quarters in range(4):
        builder = expert.Builder(env.rl, patch, layout=(anchor, quarters))
        cx, cy = builder.drill_centre
        sides.add((builder.standing[0] > cx, builder.standing[1] > cy))
        assert builder.drill_facing == expert._turn_direction(expert.DIR_SOUTH, quarters)
    assert len(sides) == 4
    stands = {expert.Builder(env.rl, patch, layout=x).standing for x in found}
    assert len(stands) > len(found) // 2


@pytest.mark.parametrize("family", ["varied_patch", "cluttered_patch"])
def test_the_new_training_families_are_buildable(family):
    """Diversity is worthless if some of it cannot be solved at all."""
    for seed in range(24):
        scene = scenes.GENERATORS["construct_smelting_line"](family, random.Random(seed))
        env = RlEnv()
        env.reset("construct_smelting_line", scene)
        found = expert.layouts(env.rl, scene["markers"]["patch"])
        assert found, (family, seed)
        if scene["entities"]:
            continue  # the walk is not straight; the policy has to find its own
        env = RlEnv()
        env.reset("construct_smelting_line", scene)
        expert.run_to_completion(env.rl, scene["markers"]["patch"], layout=found[0])
        assert env.rl.success, (family, seed, found[0])


def test_obstructed_scenes_are_demonstrated_too():
    """They were not, until 2026-09-20. The builder walks in straight lines so
    refusing looked safe, but it completes the line on 92% of cluttered scenes
    -- and refusing cost a quarter of the demonstrations on the hand-written
    split and a rising share on a generated one, which is what kept a backplay
    gate on its first rung for 20M steps."""
    env = VecEnv(32, demo_starts=1.0, seed=7)
    try:
        env.reset()
        obstructed = [i for i in range(32) if env.families[i] == "cluttered_patch"]
        assert obstructed, "expected the cluttered family in a train split"
        assert any(env.starts[i] != "scene" for i in obstructed), [
            env.starts[i] for i in obstructed
        ]
    finally:
        env.close()


def test_the_old_refusal_is_still_reachable():
    """`--no-demo-obstructed` reproduces a run made before the fix."""
    env = VecEnv(32, demo_starts=1.0, seed=7, demo_obstructed=False)
    try:
        env.reset()
        obstructed = [i for i in range(32) if env.families[i] == "cluttered_patch"]
        assert obstructed
        assert all(env.starts[i] == "scene" for i in obstructed)
    finally:
        env.close()


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


def test_the_builder_solves_build_line_too():
    """Both tasks build the same line, so one builder demonstrates both."""
    for seed in range(8):
        family, scene = scenes.sample("build_line", "train", seed)
        env = RlEnv()
        env.reset("build_line", scene)
        expert.run_to_completion(env.rl, scene["markers"]["patch"])
        assert env.rl.success, (family, seed)


def test_build_line_may_have_demonstration_starts():
    """Every scene the builder can finish gets one, walls or not. The few it
    gets stuck on are rolled back to the scene's own start rather than left
    part-built, so a start is either a full demonstration to its cut or
    nothing at all."""
    env = VecEnv(32, "build_line", demo_starts=1.0, shaping="both", seed=3)
    env.demo_window = (0, 0)
    try:
        env.reset()
        starts = dict(zip(env.families, env.starts, strict=True))
        assert "cluttered_patch" in starts, "expected the cluttered family in a train split"
        assert all(s in ("scene", "back0") for s in env.starts), env.starts
        demonstrated = sum(1 for s in env.starts if s == "back0")
        assert demonstrated >= 28, f"only {demonstrated}/32 demonstrated"
        cluttered = [env.starts[i] for i in range(32) if env.families[i] == "cluttered_patch"]
        assert any(s == "back0" for s in cluttered), cluttered
    finally:
        env.close()
