"""The native PufferLib adapter: buffers, masks, seeding, autoreset, backends.

Skipped unless PufferLib (and its `gym` import) is available. PufferLib 3.0.0
cannot be pip-installed on Windows; see `fsim/puffer_env.py`.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("gymnasium")


def _import_pufferlib():
    # `import pufferlib` symlinks a `resources` directory into the working
    # directory; import it from a scratch directory so the repo stays clean.
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as scratch:
        os.chdir(scratch)
        try:
            module = pytest.importorskip("pufferlib")
            import pufferlib.emulation
            import pufferlib.vector  # noqa: F401

            return module
        finally:
            os.chdir(cwd)


pufferlib = _import_pufferlib()

from fsim import scenes  # noqa: E402
from fsim.gym_env import FactorySimEnv, sample_masked, split_mask  # noqa: E402
from fsim.puffer_env import FactorySimPufferEnv, decode_obs, obs_nbytes  # noqa: E402
from fsim.rl import NVEC, RlEnv  # noqa: E402
from fsim.vec import OBS_KEYS  # noqa: E402


def make(**kwargs):
    return pufferlib.vector.make(
        FactorySimPufferEnv, backend=pufferlib.PufferEnv, env_kwargs=kwargs
    )


def single(task, seed, max_steps=600, split="train"):
    _, scene = scenes.sample(task, split, seed)
    env = RlEnv()
    env.reset(task, scene, max_steps=max_steps, action_space="v2")
    return env


def test_spaces_and_buffers():
    env = make(num_envs=4, threads=2)
    assert env.num_agents == 4
    assert env.single_observation_space.shape == (obs_nbytes(True),)
    assert env.single_observation_space.dtype == np.uint8
    assert tuple(env.single_action_space.nvec) == NVEC
    assert env.observations.shape == (4, obs_nbytes(True))
    assert env.action_masks.shape == (4, sum(NVEC))
    obs, infos = env.reset(seed=0)
    assert obs is env.observations and infos == []
    assert env.observation_space.contains(obs)
    env.close()


@pytest.mark.parametrize("compact", [False, True])
def test_zero_copy_rows_match_single_envs(compact):
    """C writes straight into PufferLib's buffer; decoded, each row is exactly
    the RlEnv observation of the same scene (up to byte rounding if compact)."""
    task, n, seed = "build_line", 3, 4
    env = make(num_envs=n, task=task, threads=1, compact=compact, max_steps=30)
    env.reset(seed=seed)
    singles = [single(task, seed * 1_000_003 + i, max_steps=30) for i in range(n)]
    started = n
    rng = np.random.default_rng(0)
    finished = 0
    for _ in range(70):
        decoded = env.decode()
        for i, ref in enumerate(singles):
            for key in OBS_KEYS:
                if compact and key == "grid":
                    assert np.abs(decoded[key][i] - ref.obs[key]).max() <= 0.5 / 255 + 1e-6
                else:
                    assert np.array_equal(decoded[key][i], ref.obs[key]), key
            assert np.array_equal(env.action_masks[i], ref.mask)
        actions = sample_masked(env.action_masks, rng).astype(np.int32)
        _, rewards, terms, truncs, infos = env.step(actions)
        done = 0
        for i, ref in enumerate(singles):
            _, reward, t, u, _ = ref.step(actions[i])
            assert rewards[i] == np.float32(reward)
            assert (terms[i], truncs[i]) == (t, u)
            if t or u:
                done += 1
                singles[i] = single(task, seed * 1_000_003 + started, max_steps=30)
                started += 1
        assert len(infos) == done
        finished += done
    assert finished >= n * 2
    env.close()


def test_masked_rollout():
    env = make(num_envs=16, task="plate_line", threads=4, max_steps=80)
    env.reset(seed=1)
    rng = np.random.default_rng(1)
    episodes = 0
    for _ in range(250):  # 4,000 transitions
        actions = sample_masked(env.action_masks, rng).astype(np.int32)
        for row, mask in zip(actions, env.action_masks, strict=True):
            for value, part in zip(row, split_mask(mask), strict=True):
                assert part[value]
        _, rewards, _, _, infos = env.step(actions)
        assert np.isfinite(rewards).all()
        for info in infos:
            assert all(np.isscalar(v) for v in info.values())
        episodes += len(infos)
    assert episodes >= 16 * 3
    env.close()


def test_seeding_is_deterministic():
    a, b = make(num_envs=4, threads=2), make(num_envs=4, threads=1)
    a.reset(seed=3)
    b.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert np.array_equal(a.observations, b.observations)
        actions = sample_masked(a.action_masks, rng).astype(np.int32)
        a.step(actions)
        b.step(actions)
        assert np.array_equal(a.rewards, b.rewards)
    a.close()
    b.close()


def test_decode_obs_standalone():
    env = make(num_envs=2, threads=1, compact=False)
    env.reset(seed=0)
    decoded = decode_obs(env.observations, compact=False)
    ref = single("construct_smelting_line", 0)
    for key in OBS_KEYS:
        assert np.array_equal(decoded[key][0], ref.obs[key])
    env.close()


def test_serial_backend():
    vec = pufferlib.vector.make(
        FactorySimPufferEnv, backend=pufferlib.vector.Serial, num_envs=2,
        env_kwargs={"num_envs": 3, "threads": 1},
    )  # fmt: skip
    obs, _ = vec.reset(seed=0)
    assert obs.shape == (6, obs_nbytes(True))
    for _ in range(5):
        obs, rewards, *_ = vec.step(np.zeros((6, 6), dtype=np.int32))
        assert rewards.shape == (6,)
    vec.close()


def test_emulation_route():
    """The alternative: PufferLib's emulation wrapper around the Gymnasium env."""
    env = pufferlib.emulation.GymnasiumPufferEnv(
        env_creator=FactorySimEnv, env_kwargs={"task": "build_line"}
    )
    obs, info = env.reset(seed=0)
    assert obs.shape == (1, *env.single_observation_space.shape)
    rng = np.random.default_rng(0)
    for _ in range(20):
        action = sample_masked(info["action_mask"], rng)
        obs, _, _, _, info = env.step(action.astype(np.int32))
        if not info:  # emulation may drop info on a reset step
            break
    env.close()
