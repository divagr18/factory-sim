"""The Gymnasium adapters: the checker, spaces, masks, seeding and autoreset."""

from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env  # noqa: E402

from fsim import scenes  # noqa: E402
from fsim.gym_env import (  # noqa: E402
    ENV_IDS,
    MASK_SIZE,
    FactorySimEnv,
    FactorySimVectorEnv,
    sample_masked,
    split_mask,
)
from fsim.rl import NVEC, RlEnv  # noqa: E402
from fsim.vec import OBS_KEYS  # noqa: E402

TASKS = sorted(ENV_IDS.values())


def assert_legal(action, mask):
    for value, part in zip(action, split_mask(mask), strict=True):
        assert part[value], (action, value)


@pytest.mark.parametrize("env_id", sorted(ENV_IDS))
def test_check_env(env_id):
    env = gym.make(env_id, render_mode="rgb_array").unwrapped
    check_env(env, skip_render_check=False)


@pytest.mark.parametrize("env_id", sorted(ENV_IDS))
def test_registered(env_id):
    env = gym.make(env_id, split="test")
    obs, info = env.reset(seed=0)
    assert env.unwrapped.task == ENV_IDS[env_id]
    assert info["family"] in scenes.families(ENV_IDS[env_id], "test")
    assert obs in env.observation_space
    env.close()
    vec = gym.make_vec(env_id, num_envs=2, vectorization_mode="vector_entry_point", threads=1)
    assert isinstance(vec.unwrapped, FactorySimVectorEnv)
    assert vec.metadata["autoreset_mode"] == gym.vector.AutoresetMode.SAME_STEP
    vec.close()


def test_mask_helpers():
    env = FactorySimEnv("build_line")
    _, info = env.reset(seed=3)
    mask = info["action_mask"]
    assert mask.shape == (MASK_SIZE,) and mask.dtype == bool
    assert np.array_equal(mask, env.action_masks())
    parts = split_mask(mask)
    assert [p.size for p in parts] == list(NVEC)
    for part in parts[1:]:
        assert part[0], "index 0 of an argument dimension is the always-legal sentinel"
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert_legal(sample_masked(mask, rng), mask)


@pytest.mark.parametrize("task", TASKS)
def test_masked_rollout_stays_in_spaces(task):
    """A few thousand random legal steps: every observation in its space,
    every sampled action legal, rewards finite."""
    env = FactorySimEnv(task, max_steps=120)
    rng = np.random.default_rng(1)
    obs, info = env.reset(seed=1)
    episodes = 0
    for _ in range(2000):
        assert obs in env.observation_space
        action = sample_masked(info["action_mask"], rng)
        assert action in env.action_space
        assert_legal(action, info["action_mask"])
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward)
        if terminated or truncated:
            episodes += 1
            obs, info = env.reset()
    assert episodes >= 10


def test_seeded_reset_is_deterministic():
    a, b = FactorySimEnv("build_line"), FactorySimEnv("build_line")
    rng = np.random.default_rng(0)
    for seed in (0, 7):
        oa, ia = a.reset(seed=seed)
        ob, ib = b.reset(seed=seed)
        assert ia["scene_seed"] == ib["scene_seed"] and ia["family"] == ib["family"]
        for _ in range(100):
            for key in OBS_KEYS:
                assert np.array_equal(oa[key], ob[key])
            action = sample_masked(ia["action_mask"], rng)
            oa, ra, ta, ua, ia = a.step(action)
            ob, rb, tb, ub, ib = b.step(action)
            assert (ra, ta, ua) == (rb, tb, ub)
        # Unseeded resets continue the seeded RNG identically.
        assert a.reset()[1]["scene_seed"] == b.reset()[1]["scene_seed"]
    # A different seed gives a different scene sequence.
    seeds = {a.reset(seed=s)[1]["scene_seed"] for s in range(5)}
    assert len(seeds) == 5


def test_scene_seed_option_matches_scenes_sample():
    env = FactorySimEnv("plate_line", split="test")
    obs, info = env.reset(options={"scene_seed": 42})
    family, scene = scenes.sample("plate_line", "test", 42)
    assert info["family"] == family
    rl = RlEnv()
    rl.reset("plate_line", scene, max_steps=400, construction_tick_limit=24000, action_space="v2")
    for key in OBS_KEYS:
        assert np.array_equal(obs[key], rl.obs[key])


def test_vector_same_step_autoreset_matches_single_envs():
    """Each slot is exactly an RlEnv; a finished slot reports its last
    observation in `final_obs` and already returns the next episode's."""
    n, max_steps, task = 3, 40, "construct_smelting_line"
    vec = FactorySimVectorEnv(n, task, threads=2, max_steps=max_steps)
    obs, info = vec.reset(seed=5)
    base = 5 * 1_000_003
    singles = []

    def start(env, seed):
        _, scene = scenes.sample(task, "train", seed)
        env.reset(task, scene, max_steps=max_steps, action_space="v2")

    for i in range(n):
        env = RlEnv()
        start(env, base + i)
        singles.append(env)
    started = n
    rng = np.random.default_rng(0)
    finished = 0
    for _ in range(130):
        assert obs in vec.observation_space
        for i, env in enumerate(singles):
            for key in OBS_KEYS:
                assert np.array_equal(obs[key][i], env.obs[key]), key
            assert np.array_equal(info["action_mask"][i], env.mask.astype(bool))
        actions = sample_masked(info["action_mask"], rng)
        obs, rewards, term, trunc, info = vec.step(actions)
        done = term | trunc
        if done.any():
            assert np.array_equal(info["_final_obs"], done)
        else:
            assert "final_obs" not in info
        for i, env in enumerate(singles):
            _, reward, t, u, _ = env.step(actions[i])
            assert (rewards[i], term[i], trunc[i]) == (reward, t, u)
            if t or u:
                finished += 1
                for key in OBS_KEYS:
                    assert np.array_equal(info["final_obs"][i][key], env.obs[key])
                assert info["final_info"][i]["length"] == env.rl.steps
                start(env, base + started)
                started += 1
    assert finished >= n * 3


def test_vector_seeding_and_copy():
    a = FactorySimVectorEnv(4, "build_line", threads=2)
    b = FactorySimVectorEnv(4, "build_line", threads=1, copy=False)
    oa, _ = a.reset(seed=9)
    ob, _ = b.reset(seed=9)
    for key in OBS_KEYS:
        assert np.array_equal(oa[key], ob[key])
        assert oa[key].flags.c_contiguous
    kept = oa["self"].copy()
    a.step(np.zeros((4, 6), dtype=np.int64))
    assert np.array_equal(oa["self"], kept), "copy=True returns arrays step does not overwrite"
    a.close()
    b.close()


def test_vector_masked_rollout():
    vec = FactorySimVectorEnv(16, "plate_line", threads=4, max_steps=80)
    obs, info = vec.reset(seed=0)
    rng = np.random.default_rng(2)
    episodes = 0
    for _ in range(250):  # 4,000 transitions
        actions = sample_masked(info["action_mask"], rng)
        assert actions in vec.action_space
        for row, mask in zip(actions, info["action_mask"], strict=True):
            assert_legal(row, mask)
        obs, rewards, term, trunc, info = vec.step(actions)
        assert obs in vec.observation_space
        assert rewards.shape == (16,) and np.isfinite(rewards).all()
        episodes += int((term | trunc).sum())
    assert episodes >= 16 * 3
    vec.close()
