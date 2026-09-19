"""GRPO's two departures from PPO: shared scenes, and a group baseline.

The advantage is checked as arithmetic; the environment side is checked for the
two properties the arithmetic assumes -- that a group really did face the same
scene, and that an episode which ended early stops contributing.
"""

from __future__ import annotations

import numpy as np
import pytest

from fsim.vec import VecEnv

torch = pytest.importorskip("torch")


def _group_advantage(*args):
    from train import group_advantage

    return group_advantage(*args)


def test_advantage_is_relative_to_the_group():
    # Two groups of three. Within each, the middle attempt is the mean.
    rewards = torch.zeros(4, 6)
    rewards[0] = torch.tensor([0.0, 1.0, 2.0, 10.0, 20.0, 30.0])
    live = torch.ones(4, 6)
    adv = _group_advantage(rewards, live, 3)

    assert adv.shape == (4, 6)
    # Every timestep of an episode carries that episode's single number.
    for n in range(6):
        assert torch.allclose(adv[:, n], adv[0, n].expand(4))
    # Each group is centred on itself, so the two groups come out identical
    # even though one scored ten times higher.
    assert torch.allclose(adv[0, :3], adv[0, 3:], atol=1e-5)
    assert adv[0, 1].abs() < 1e-5
    assert adv[0, 0] < 0 < adv[0, 2]
    assert abs(float(adv[0, :3].mean())) < 1e-5


def test_dead_timesteps_neither_count_nor_train():
    rewards = torch.ones(4, 2)
    live = torch.ones(4, 2)
    live[2:, 0] = 0.0  # the first episode ended after two decisions
    adv = _group_advantage(rewards, live, 2)

    # Return 2 against return 4: the shorter episode is the worse one here.
    assert adv[0, 0] < 0 < adv[0, 1]
    # ... and its dead tail is silent rather than negative.
    assert torch.equal(adv[2:, 0], torch.zeros(2))


def test_a_tied_group_has_nothing_to_say():
    rewards = torch.full((3, 4), 0.25)
    for baseline in ("group", "loo"):
        adv = _group_advantage(rewards, torch.ones(3, 4), 4, baseline)
        assert torch.allclose(adv, torch.zeros(3, 4), atol=1e-6), baseline


def test_leave_one_out_judges_against_the_others():
    rewards = torch.zeros(1, 4)
    rewards[0] = torch.tensor([0.0, 0.0, 0.0, 4.0])
    adv = _group_advantage(rewards, torch.ones(1, 4), 4, "loo")
    # The winner beat the other three, who averaged 0; each loser trailed a
    # field averaging 4/3.
    assert adv[0, 3] == pytest.approx(4.0)
    assert adv[0, 0] == pytest.approx(-4.0 / 3.0)
    assert float(adv.sum()) == pytest.approx(0.0, abs=1e-6)


def test_leave_one_out_does_not_inflate_a_near_tie():
    # Three failures and a marginally better failure: GRPO divides by a spread
    # that is nearly zero and reports a full-size advantage; RLOO reports a
    # small one, because the difference really was small.
    rewards = torch.zeros(1, 4)
    rewards[0] = torch.tensor([0.0, 0.0, 0.0, 1e-3])
    standardised = _group_advantage(rewards, torch.ones(1, 4), 4, "group")
    leave_one_out = _group_advantage(rewards, torch.ones(1, 4), 4, "loo")
    assert standardised[0, 3] > 1.0
    assert leave_one_out[0, 3] < 1e-2


def test_a_group_shares_one_scene():
    env = VecEnv(8, "build_line", seed=3, threads=1, group=4, autoreset=False)
    try:
        env.reset()
        seeds = [env._seed_of[i] for i in range(8)]
        assert seeds[:4] == [seeds[0]] * 4
        assert seeds[4:] == [seeds[4]] * 4
        assert seeds[0] != seeds[4]
        assert env.families[:4] == [env.families[0]] * 4
        for i in range(1, 4):
            assert np.array_equal(env.obs["self"][i], env.obs["self"][0])
    finally:
        env.close()


def test_grouping_leaves_the_ungrouped_seed_stream_alone():
    env = VecEnv(4, "build_line", seed=3, threads=1)
    try:
        env.reset()
        assert [env._seed_of[i] for i in range(4)] == [env.seed_base + k for k in range(4)]
    finally:
        env.close()


def test_a_finished_episode_stays_finished():
    env = VecEnv(8, "build_line", seed=3, threads=1, group=4, autoreset=False)
    try:
        env.reset()
        rng = np.random.default_rng(0)
        reported: dict[int, int] = {}
        for t in range(700):
            actions = np.zeros((8, 6), dtype=np.int32)
            for i in range(8):
                legal = np.flatnonzero(env.masks[i][:22])
                actions[i, 0] = rng.choice(legal) if len(legal) else 0
            before = env.alive.copy()
            _, _, reward, terminated, truncated, _ = env.step(actions)
            for i in np.flatnonzero(terminated | truncated).tolist():
                assert before[i], f"environment {i} reported a second time at step {t}"
                reported[i] = t
            assert not np.any(reward[~before]), f"a finished environment was paid at step {t}"
            if not env.alive.any():
                break
        assert len(reported) == 8
    finally:
        env.close()
