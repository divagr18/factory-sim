"""Levels as parameters, and the buffer that curates them.

The property the whole design rests on is the first test here: the parameter
space reproduces every hand-written family exactly. If it did not, a UED run
and a hand-written run would be measured against different worlds and none of
the held-out numbers in this project would stay comparable.
"""

from __future__ import annotations

import json
import math
import random

import numpy as np

import pytest

from fsim import scenes, ued


@pytest.mark.parametrize(
    ("task", "family"),
    [(task, family) for task, families in scenes.FAMILIES.items() if task in ued.PATCH_TASKS
     for family in families],  # fmt: skip
)
def test_a_family_is_a_point_in_the_parameter_space(task, family):
    """Byte-identical payloads, not merely similar scenes: same tiles, same
    walls, same start, drawn in the same order from the same seed."""
    for seed in range(40):
        built = ued.build(ued.family_params(task, family, random.Random(seed)))
        reference = scenes.GENERATORS[task](family, random.Random(seed))
        assert json.dumps(built, sort_keys=True) == json.dumps(reference, sort_keys=True), seed


def test_a_level_is_its_parameters_and_nothing_else():
    level = ued.random_level("build_line", random.Random(7))
    assert ued.build(level) == ued.build(level)


def test_generated_levels_are_installable():
    """A level UED invents still has to be a scene the C core accepts."""
    from fsim.rl import RlEnv

    env = RlEnv()
    rng = random.Random(0)
    for _ in range(25):
        level = ued.random_level("construct_smelting_line", rng)
        env.reset("construct_smelting_line", ued.build(level), shaping="both", action_space="v2")


def test_edits_are_small_and_stay_in_bounds():
    rng = random.Random(3)
    level = ued.random_level("construct_smelting_line", rng)
    for _ in range(300):
        level = ued.mutate(level, rng, edits=2)
        assert ued.PATCH_MIN <= level.x_lo <= level.x_hi <= ued.PATCH_MAX
        assert ued.PATCH_MIN <= level.y_lo <= level.y_hi <= ued.PATCH_MAX
        assert abs(level.ox) <= ued.OFFSET_LIMIT and abs(level.oy) <= ued.OFFSET_LIMIT
        assert len(level.walls) <= ued.MAX_WALLS
        for _vertical, away, along, length in level.walls:
            # A screen on top of the patch makes a scene unsolvable rather than
            # hard, and a regret-seeking curriculum would happily find those.
            assert abs(away) >= 4, level.walls
            assert length in ued.WALL_LENGTH
            assert along in ued.WALL_ALONG
        assert 6.0 <= level.rx <= 16.0 and 6.0 <= level.ry <= 16.0
        assert 0.0 <= level.angle < 2 * math.pi


def test_an_edit_keeps_its_parent_in_sight():
    """ACCEL assumes regret varies smoothly with the parameters, so one edit
    has to leave the level recognisably where it was."""
    rng = random.Random(11)
    parent = ued.random_level("build_line", rng)
    for _ in range(50):
        child = ued.mutate(parent, rng, edits=1)
        moved = abs(child.ox - parent.ox) + abs(child.oy - parent.oy)
        resized = (
            abs(child.x_lo - parent.x_lo) + abs(child.x_hi - parent.x_hi)
            + abs(child.y_lo - parent.y_lo) + abs(child.y_hi - parent.y_hi)
        )  # fmt: skip
        assert moved <= 2 and resized <= 1
        assert abs(len(child.walls) - len(parent.walls)) <= 1


def test_the_buffer_prefers_high_regret_and_then_the_stale():
    buffer = ued.LevelBuffer(capacity=8, beta=0.3, rho=0.0, seed=1)
    rng = random.Random(0)
    for i in range(8):
        buffer.consider(ued.random_level("build_line", rng), score=float(i))
    counts = [0] * 8
    for _ in range(4000):
        index, _ = buffer.sample()
        counts[index] += 1
    # Rank-prioritised: the top-scoring level is drawn far more often than the
    # worst, and the ordering is monotone in score.
    assert counts[7] > counts[0] * 10
    assert counts[7] > counts[6] > counts[0]


def test_staleness_rescues_a_level_the_ranking_has_forgotten():
    hot = ued.LevelBuffer(capacity=4, beta=0.1, rho=0.0, seed=2)
    fair = ued.LevelBuffer(capacity=4, beta=0.1, rho=0.9, seed=2)
    for buffer in (hot, fair):
        draw = random.Random(0)
        for i in range(4):
            buffer.consider(ued.random_level("build_line", draw), score=float(i))

    def share_of_worst(buffer):
        counts = [0] * 4
        for _ in range(2000):
            index, _ = buffer.sample()
            counts[index] += 1
        return counts[0] / sum(counts)

    assert share_of_worst(fair) > share_of_worst(hot) * 3


def test_the_buffer_only_takes_a_level_that_beats_its_worst():
    buffer = ued.LevelBuffer(capacity=3, seed=4)
    rng = random.Random(0)
    for score in (0.5, 0.6, 0.7):
        assert buffer.consider(ued.random_level("build_line", rng), score)
    assert not buffer.consider(ued.random_level("build_line", rng), 0.1)
    assert len(buffer) == 3
    assert buffer.consider(ued.random_level("build_line", rng), 0.9)
    assert len(buffer) == 3
    assert min(entry.score for entry in buffer.entries) == pytest.approx(0.6)


def test_the_same_level_is_not_held_twice():
    buffer = ued.LevelBuffer(capacity=16, seed=5)
    level = ued.random_level("build_line", random.Random(0))
    assert buffer.consider(level, 0.5)
    assert not buffer.consider(level, 0.9)
    assert len(buffer) == 1
    # ... and offering it again is an update, not a silent discard.
    assert buffer.entries[0].score == pytest.approx(0.9)


def test_positive_value_loss_reads_as_regret():
    torch = pytest.importorskip("torch")
    # A level the policy is still learning: the critic keeps being surprised.
    learning = torch.tensor([0.4, 0.9, 0.2, 0.7])
    # One it has mastered, or one it never gets anywhere on: no surprise either
    # way, which is why PLR's proxy is a frontier detector rather than a
    # difficulty meter.
    settled = torch.tensor([0.01, -0.02, 0.0, 0.01])
    assert ued.positive_value_loss(learning) > ued.positive_value_loss(settled) * 10
    # Negative advantages do not subtract from regret.
    assert ued.positive_value_loss(torch.tensor([-5.0, -5.0])) == 0.0


def test_parking_a_finished_episode_does_not_draw_a_new_level():
    """Under --whole-episodes a finished environment is reset so its action
    mask stays legal. That reset must not ask the curriculum for a new level:
    the slot's score has not been attributed yet, and replacing the level
    first credits the episode to whichever level happened to arrive next.
    """
    from fsim.vec import VecEnv

    calls: list[int] = []
    levels = ued.Curriculum("build_line", n=4, train_slots=4, mode="plr", seed=0, warm_start=8)

    def source(i, seed):
        calls.append(i)
        return levels.level_for(i, seed)

    env = VecEnv(4, "build_line", threads=1, autoreset=False, level_source=source)
    try:
        env.reset()
        assert len(calls) == 4
        before = list(levels.slot_level)

        rng = np.random.default_rng(0)
        actions = np.zeros((4, 6), dtype=np.int32)
        for _ in range(700):
            for i in range(4):
                legal = np.flatnonzero(env.masks[i][:22])
                actions[i, 0] = rng.choice(legal) if len(legal) else 0
            env.step(actions)
            if not env.alive.any():
                break

        assert not env.alive.any(), "every episode should have ended"
        assert len(calls) == 4, f"the curriculum was asked {len(calls) - 4} extra times"
        assert list(levels.slot_level) == before
    finally:
        env.close()


def test_only_the_replay_slots_are_meant_to_train():
    """The split itself, stated: a Curriculum with 12 of 16 slots training
    replays from the buffer on those and proposes on the rest."""
    levels = ued.Curriculum("build_line", n=16, train_slots=12, mode="accel", seed=0, warm_start=16)
    origins = [levels.level_for(i, i)[0] for i in range(16)]
    assert all(not name.startswith("edit:") for name in origins[:12])
    assert all(name.startswith("edit:") for name in origins[12:])
    assert levels.replayed == 12
    assert levels.edited == 4


def test_a_wall_is_never_laid_across_the_ore():
    """A screen over the patch makes a scene unsolvable rather than hard, and
    a regret-seeking search will find those. A constant minimum offset was not
    enough -- it assumed patches no wider than the hand-written families, and
    resizing grows them to thirteen tiles. Measured before the fix: 688 of
    4,000 mutated levels had walls sitting on ore."""
    rng = random.Random(0)
    for _ in range(600):
        level = ued.random_level("construct_smelting_line", rng)
        for _ in range(12):
            level = ued.mutate(level, rng, edits=2)
        scene = ued.build(level)
        ore = {(math.floor(x), math.floor(y)) for x, y in
               (r["position"] for r in scene["resources"])}  # fmt: skip
        walls = {(math.floor(x), math.floor(y)) for x, y in
                 (e["position"] for e in scene["entities"])}  # fmt: skip
        assert not (ore & walls), (level.walls, sorted(ore & walls))


def test_an_empty_buffer_does_not_get_trained_on():
    """Robust PLR's whole point: a generated level is scored, never learned
    from. A training slot whose buffer was empty fell through to the
    generator and was trained on anyway."""
    levels = ued.Curriculum("build_line", n=8, train_slots=6, mode="plr", seed=0, warm_start=0)
    for i in range(8):
        levels.level_for(i, i)
    assert not any(levels.training_mask()), levels.training_mask()

    warmed = ued.Curriculum("build_line", n=8, train_slots=6, mode="plr", seed=0, warm_start=16)
    for i in range(8):
        warmed.level_for(i, i)
    assert warmed.training_mask() == [True] * 6 + [False] * 2


def test_loading_a_buffer_keeps_its_replay_history(tmp_path):
    buffer = ued.LevelBuffer(capacity=32, seed=0)
    rng = random.Random(0)
    for i in range(6):
        buffer.consider(ued.random_level("build_line", rng), score=i / 6)
    for index in (0, 0, 3):
        buffer.update(index, 0.5)
    buffer.entries[2].staleness = 17.0

    path = tmp_path / "levels.json"
    ued.save_buffer(buffer, path)
    back = ued.load_buffer(path)

    assert [e.seen for e in back.entries] == [e.seen for e in buffer.entries]
    assert [e.staleness for e in back.entries] == [e.staleness for e in buffer.entries]
    assert [e.level.key() for e in back.entries] == [e.level.key() for e in buffer.entries]


def test_a_walled_scene_is_still_demonstrated():
    """The builder walks in straight lines, so scenes containing walls were
    refused a demonstration outright. Measured, it completes the line on 92%
    of `cluttered_patch` and 91% of the walled levels UED generates -- and
    refusing cost the UED arm a rising share of its demonstrations as the
    curriculum filled with walls, which is why its backplay gate never left
    rung 0 in 20M steps.
    """
    from fsim.vec import VecEnv

    rates = {}
    for allow in (False, True):
        env = VecEnv(
            48, "construct_smelting_line", threads=4, shaping="both",
            action_space="v2", demo_starts=0.5, demo_obstructed=allow,
        )  # fmt: skip
        env.demo_window = (0, 2)
        try:
            demonstrated = total = 0
            for _ in range(8):
                env.reset()
                demonstrated += sum(1 for s in env.starts if s != "scene")
                total += len(env.starts)
            rates[allow] = demonstrated / total
        finally:
            env.close()
    # --demo-starts 0.5 should mean about half, and only a stuck builder
    # should cost one.
    assert rates[True] > rates[False] + 0.07, rates
    assert 0.44 < rates[True] < 0.56, rates


def test_a_stuck_demonstration_is_rolled_back():
    """A half-finished demonstration is worse than none: it starts the policy
    from a state the expert never reaches, and labels it as though the expert
    had. Every episode is either a full demonstration to its cut or a plain
    scene start."""
    from fsim.vec import VecEnv

    env = VecEnv(
        64, "construct_smelting_line", threads=4, shaping="both",
        action_space="v2", demo_starts=1.0, demo_obstructed=True,
    )  # fmt: skip
    env.demo_window = (0, 2)
    try:
        for _ in range(6):
            env.reset()
            for i, start in enumerate(env.starts):
                assert start == "scene" or start.startswith("back"), start
                # A rolled-back attempt reports no decisions taken.
                if start == "scene":
                    assert env.lengths[i] == 0, (start, env.lengths[i])
    finally:
        env.close()
