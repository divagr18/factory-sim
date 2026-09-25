"""The batched environment under v3 is `RlEnv` under v3, bit for bit.

`VecEnv(action_space="v3")` steps through `fsim_rl_step_range3`, which writes
each environment's v3 tensors, flat mask and per-operation masks in one pass.
Every slot must be exactly the single environment given the same scene and
the same actions: the tensors (`observe3`), both masks (`observe3`,
`op_masks`), the rewards and the episode flags, across autoreset.
"""

from __future__ import annotations

import numpy as np
import pytest

from fsim import lib, scenes
from fsim.rl import NVEC3, RlEnv
from fsim.rl import TASKS as RL_TASKS
from fsim.vec import OBS_KEYS, VecEnv, unpack_grid

OFFSETS = np.cumsum((0, *NVEC3[1:]))


def conditional_actions(rng, masks, op_masks, finish=0.002):
    """A legal v3 action per environment: an operation, then each argument
    from that operation's own row. `finish` ends an episode, so it is drawn
    rarely, which keeps episodes long enough to build something."""
    out = np.zeros((masks.shape[0], 6), np.int32)
    for i in range(masks.shape[0]):
        legal = np.flatnonzero(masks[i, : NVEC3[0]])
        legal = legal[legal != 24] if rng.random() > finish else legal
        op = int(rng.choice(legal))
        out[i, 0] = op
        row = op_masks[i, op]
        for d in range(5):
            choices = np.flatnonzero(row[OFFSETS[d] : OFFSETS[d + 1]])
            out[i, d + 1] = int(rng.choice(choices))
    return out


def _singles(task, n, options):
    envs = []
    for i in range(n):
        env = RlEnv()
        _, scene = scenes.sample(task, "train", i)
        env.reset(task, scene, action_space="v3", **options)
        envs.append(env)
    return envs


TASKS = [
    t
    for t in ("construct_smelting_line", "belt_smelting")
    if t in RL_TASKS and t in scenes.GENERATORS
]


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("shaping", ["none", "both"])
def test_batch_matches_single_environments_v3(task, shaping):
    n, steps, budget = 3, 400, 150  # past two episodes a slot: autoreset
    options = {"shaping": shaping, "max_steps": budget}
    vec = VecEnv(n, task, threads=2, shaping=shaping, max_steps=budget, action_space="v3")
    obs, masks = vec.reset()
    singles = _singles(task, n, options)
    started = n
    rng = np.random.default_rng(0)
    ended = 0
    # `potentials` is the state after the step, so it belongs to the episode
    # that ended there, not to the one autoreset started.
    fresh = np.ones(n, dtype=bool)
    for step in range(steps):
        for i, env in enumerate(singles):
            o, m = env.observe3()
            for key in OBS_KEYS:
                assert np.array_equal(obs[key][i], o[key]), (step, i, key)
            assert np.array_equal(masks[i], m), (step, i)
            assert np.array_equal(vec.op_masks[i], env.op_masks()), (step, i)
            assert fresh[i] or vec.potentials[i] == lib.fsim_rl_potential(env.rl), (step, i)
        actions = conditional_actions(rng, masks, vec.op_masks)
        obs, masks, rewards, term, trunc, _ = vec.step(actions)
        fresh = term | trunc
        for i, env in enumerate(singles):
            _, reward, t, u, _ = env.step(actions[i])
            assert rewards[i] == reward, (step, i)
            assert (term[i], trunc[i]) == (t, u), (step, i)
            assert not vec.flags[i, 3], "a conditional-mask action failed to decode"
            if t or u:
                ended += 1
                _, scene = scenes.sample(task, "train", started)
                started += 1
                env.reset(task, scene, action_space="v3", **options)
    assert ended >= n
    vec.close()


@pytest.mark.parametrize("task", TASKS)
def test_packed_v3_observation_unpacks_to_the_rounded_float_one(task):
    full = VecEnv(4, task, threads=2, seed=3, action_space="v3", max_steps=120)
    packed = VecEnv(4, task, threads=2, seed=3, action_space="v3", max_steps=120, compact=True)
    full.reset()
    packed.reset()
    rng = np.random.default_rng(1)
    for step in range(200):
        grid = unpack_grid(packed.obs["flags"], packed.obs["amount"])
        expected = np.round(full.obs["grid"] * np.float32(255)).astype(np.uint8)
        assert np.array_equal(grid, expected), step
        for key in OBS_KEYS[1:]:
            assert np.array_equal(packed.obs[key], full.obs[key]), (step, key)
        assert np.array_equal(packed.masks, full.masks)
        assert np.array_equal(packed.op_masks, full.op_masks)
        actions = conditional_actions(rng, full.masks, full.op_masks)
        _, _, r0, t0, _, _ = full.step(actions)
        _, _, r1, t1, _, _ = packed.step(actions)
        assert np.array_equal(r0, r1) and np.array_equal(t0, t1)
        assert np.array_equal(full.potentials, packed.potentials)
    full.close()
    packed.close()


def test_v3_mask_shapes():
    vec = VecEnv(2, action_space="v3")
    _, masks = vec.reset()
    assert masks.shape == (2, lib.RL3_MASK_SIZE)
    assert vec.op_masks.shape == (2, lib.RL3_OPERATIONS, lib.RL3_ARG_WIDTH)
    assert vec.obs["entities"].shape == (2, lib.RL3_MAX_ENTITIES, lib.RL3_ENTITY_FEATURES)
    assert vec.obs["goal"].shape == (2, lib.RL3_GOAL_FEATURES)
    vec.close()
