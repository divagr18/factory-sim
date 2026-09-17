"""The packed observation a trainer ships is the float observation, exactly."""

from __future__ import annotations

import numpy as np

from fsim.vec import OBS_KEYS, VecEnv, unpack_grid


def test_packed_observation_unpacks_to_the_rounded_float_one():
    full = VecEnv(12, threads=2, demo_starts=0.5, seed=3)
    packed = VecEnv(12, threads=2, demo_starts=0.5, seed=3, compact=True)
    full.reset()
    packed.reset()
    rng = np.random.default_rng(0)
    for step in range(700):
        grid = unpack_grid(packed.obs["flags"], packed.obs["amount"])
        # np.round is half to even, as the C encoder and torch.round are
        expected = np.round(full.obs["grid"] * np.float32(255)).astype(np.uint8)
        assert np.array_equal(grid, expected), step
        for key in OBS_KEYS[1:]:
            assert np.array_equal(packed.obs[key], full.obs[key]), (step, key)
        assert np.array_equal(packed.masks, full.masks)
        actions = np.zeros((12, 6), np.int32)
        actions[:, 0] = rng.integers(0, 12, 12)  # walking changes the grid
        _, _, r0, t0, _, _ = full.step(actions)
        _, _, r1, t1, _, _ = packed.step(actions)
        assert np.array_equal(r0, r1) and np.array_equal(t0, t1)
        assert np.array_equal(full.potentials, packed.potentials)
    full.close()
    packed.close()


def test_torch_unpacking_matches_numpy():
    import pytest

    torch = pytest.importorskip("torch")
    env = VecEnv(4, threads=1, compact=True)
    env.reset()
    expected = unpack_grid(env.obs["flags"], env.obs["amount"])
    got = unpack_grid(
        torch.from_numpy(env.obs["flags"].copy()), torch.from_numpy(env.obs["amount"].copy()), torch
    )
    assert np.array_equal(got.numpy(), expected)
    assert expected[:, 0].any()  # the ore patch is in view
