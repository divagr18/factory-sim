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


def _togo_advantage(*args):
    from train import togo_advantage

    return togo_advantage(*args)


def test_reward_to_go_credits_the_decision_that_earned_it():
    """Two attempts at one scene. One is paid at its first decision, the other
    at its last. A flat episode return cannot tell them apart at any single
    timestep; reward-to-go can."""
    rewards = torch.zeros(3, 2)
    rewards[0, 0] = 1.0  # paid early
    rewards[2, 1] = 1.0  # paid late
    live = torch.ones(3, 2)

    flat = _group_advantage(rewards, live, 2)
    # Same total, so the episode-return baseline has nothing to say at all.
    assert torch.allclose(flat, torch.zeros(3, 2), atol=1e-5)

    togo = _togo_advantage(rewards, live, 2, 1.0)
    # At the first decision both still have 1.0 to come, so they tie...
    assert togo[0].abs().max() < 1e-5
    # ... but by the second, only the late-paid attempt has anything left.
    assert togo[1, 1] > 0 > togo[1, 0]


def test_reward_to_go_discounts_and_stays_inside_the_episode():
    rewards = torch.zeros(4, 2)
    rewards[3, 0] = 1.0
    live = torch.ones(4, 2)
    live[2:, 1] = 0.0
    togo = _togo_advantage(rewards, live, 2, 0.5)
    # The dead tail of the second episode contributes nothing and is silent.
    assert torch.equal(togo[2:, 1], torch.zeros(2))
    # The first episode's payment is discounted back: 0.5^3, 0.5^2, 0.5, 1.
    assert togo[0, 0] != 0.0
    assert abs(float(togo[3, 0])) > abs(float(togo[0, 0]))


def test_whole_episodes_is_a_rollout_shape_not_a_learner():
    """--whole-episodes parks an environment once its episode ends, so every
    rollout after the first must reset. A PPO arm run with the flag has the
    same dead tail as a GRPO one, and forgetting that trained one control on
    six hundred copies of a frozen state for an entire run -- with no error,
    just metrics that never moved again.
    """
    import train

    args = train.parse(["--run", "x", "--algo", "grpo", "--envs", "8", "--group", "4"])
    assert args.whole_episodes, "grpo cannot compute a return without whole episodes"

    plain = train.parse(["--run", "x"])
    assert not plain.whole_episodes
    assert plain.group == 8  # grouping is available to any learner...

    control = train.parse(["--run", "x", "--whole-episodes", "--group", "4"])
    assert control.algo == "ppo"  # ... and so is the rollout shape
    assert control.whole_episodes
