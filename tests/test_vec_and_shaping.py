"""The batched environment, and construct_smelting_line's potential shaping."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fsim import lib, scenes
from fsim.parity import GOLDEN
from fsim.rl import NVEC, RlEnv
from fsim.trace import read_trace
from fsim.vec import OBS_KEYS, VecEnv

REFERENCE = "construct_smelting_line_reference"


def random_actions(rng, masks):
    out = np.zeros((masks.shape[0], 6), np.int32)
    offset = 0
    for d, n in enumerate(NVEC):
        for i in range(masks.shape[0]):
            legal = np.flatnonzero(masks[i, offset : offset + n])
            out[i, d] = rng.choice(legal)
        offset += n
    return out


def reference_vectors():
    header, records = read_trace(GOLDEN / f"{REFERENCE}.jsonl.xz")
    vectors = [r["transition"]["action"]["vector"] for r in records[1:]]
    return header, vectors


@pytest.mark.parametrize("shaping", ["none", "potential", "progress"])
def test_batch_matches_single_environments(shaping):
    """Stepped on threads, with autoreset, each slot is exactly an RlEnv."""
    n, steps = 3, 700  # past one 600-decision episode, so autoreset is covered
    vec = VecEnv(n, threads=2, shaping=shaping, max_steps=600)
    obs, masks = vec.reset()
    singles, seeds = [], []
    for i in range(n):
        env = RlEnv()
        _, scene = scenes.sample("construct_smelting_line", "train", i)
        env.reset("construct_smelting_line", scene, shaping=shaping)
        singles.append(env)
        seeds.append(i)
    started = n
    rng = np.random.default_rng(0)
    for _ in range(steps):
        for i, env in enumerate(singles):
            for key in OBS_KEYS:
                assert np.array_equal(obs[key][i], env.obs[key]), key
            assert np.array_equal(masks[i], env.mask)
        actions = random_actions(rng, masks)
        obs, masks, rewards, term, trunc, _ = vec.step(actions)
        for i, env in enumerate(singles):
            _, reward, t, u, _ = env.step(actions[i])
            assert rewards[i] == reward
            assert (term[i], trunc[i]) == (t, u)
            if t or u:
                _, scene = scenes.sample("construct_smelting_line", "train", started)
                started += 1
                env.reset("construct_smelting_line", scene, shaping=shaping)
    vec.close()


def run_reference(gamma):
    header, vectors = reference_vectors()
    env = RlEnv()
    env.reset(header["task"], header["blueprint"], shaping="potential", gamma=gamma)
    phi0 = lib.fsim_rl_potential(env.rl)
    potentials, shaped, rewards = [phi0], [], []
    for vector in vectors:
        _, reward, terminated, truncated, info = env.step(vector)
        potentials.append(env.rl.potential)
        shaped.append(info["reward_components"]["line_potential"])
        rewards.append(reward)
        if terminated or truncated:
            break
    return info, phi0, potentials, shaped, rewards, terminated


def test_potential_rises_as_the_reference_builds_its_line():
    info, phi0, potentials, _, _, terminated = run_reference(0.999)
    assert terminated and info["success"]
    assert 0.0 < phi0 < 0.1  # only the approach term, before anything is built
    # drill 0.2, line 0.3, both fuelled 0.2, furnace fed 0.1, and the approach
    # term at the patch (the character stands a tile or so from its centre)
    assert 0.89 < max(potentials) < 0.9
    assert potentials[5:9] == pytest.approx([0.299, 0.599, 0.699, 0.799], abs=0.001)
    assert potentials[-1] == 0.0  # zeroed at termination


def test_shaping_telescopes_to_minus_the_initial_potential():
    """With gamma = 1 the shaped terms of an episode sum to -phi(s0), so the
    return differs from the unshaped one by a constant of the start state.

    The last decision carries two of them, as FactorioRL scores it: the
    decision's own transition (its `line_potential` component), then the
    verification window's transition into the absorbing state, -phi(s')."""
    _, phi0, _, shaped, rewards, _ = run_reference(1.0)
    assert sum(rewards) - 1.0 == pytest.approx(-phi0, abs=1e-12)
    into_absorbing = rewards[-1] - shaped[-1] - 1.0  # -phi(s'), s' the built line
    assert -0.9 < into_absorbing < -0.89
    assert sum(shaped) + into_absorbing == pytest.approx(-phi0, abs=1e-12)


def test_shaping_leaves_the_sparse_reward_alone():
    header, vectors = reference_vectors()
    plain, shaped = RlEnv(), RlEnv()
    plain.reset(header["task"], header["blueprint"])
    shaped.reset(header["task"], header["blueprint"], shaping="potential")
    for vector in vectors:
        _, r0, t0, _, i0 = plain.step(vector)
        _, r1, t1, _, i1 = shaped.step(vector)
        extra = r1 - r0 - i1["reward_components"]["line_potential"]
        assert t0 == t1
        if t0:
            assert -0.9 < extra < -0.89  # the verification's -phi(s')
            break
        assert extra == pytest.approx(0.0, abs=1e-12)


def test_shaping_is_refused_for_build_line():
    env = RlEnv()
    _, scene = scenes.sample("build_line", "train", 0)
    with pytest.raises(ValueError):
        env.reset("build_line", scene, shaping="progress")


def test_progress_pays_each_rise_of_the_potential_once():
    header, vectors = reference_vectors()
    env = RlEnv()
    env.reset(header["task"], header["blueprint"], shaping="progress")
    high = lib.fsim_rl_potential(env.rl)
    paid = []
    for vector in vectors:
        _, reward, terminated, _, info = env.step(vector)
        payout = info["reward_components"]["line_progress"]
        phi = lib.fsim_rl_potential(env.rl)
        if not terminated:
            assert payout == pytest.approx(0.5 * max(0.0, phi - high), abs=1e-12)
            high = max(high, phi)
        paid.append(payout)
        if terminated:
            break
    assert all(p >= 0 for p in paid)
    # the reference ends with the full line: 0.5 * (phi_max - phi_0) in total
    assert sum(paid) == pytest.approx(0.5 * (0.899016 - 0.083582), abs=1e-3)
    assert sum(paid) <= 0.45


POTENTIALS = json.loads((GOLDEN / "potentials.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", sorted(POTENTIALS))
def test_potential_matches_factoriorl_on_every_recorded_decision(name):
    """`fsim_rl_potential` against FactorioRL's `line_potential`, evaluated on
    the engine's own observations, bit for bit."""
    header, records = read_trace(GOLDEN / f"{name}.jsonl.xz")
    env = RlEnv()
    env.reset(
        header["task"],
        header["blueprint"],
        decision_ticks=header["decision_ticks"],
        max_steps=header["max_decision_steps"],
        construction_tick_limit=header["construction_tick_limit"],
    )
    expected = POTENTIALS[name]
    assert len(expected) == len(records)
    assert lib.fsim_rl_potential(env.rl) == expected[0]
    for index, record in enumerate(records[1:], start=1):
        env.step(record["transition"]["action"]["vector"])
        assert lib.fsim_rl_potential(env.rl) == expected[index], (name, index)
