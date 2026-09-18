"""Observation, mask and action triples from the scripted builder.

These are labels, which demonstration starts deliberately are not. They exist
for one thing: a behaviour-cloned prior for VPT's KL anchor (Baker et al. 2022,
arXiv:2206.11795), where the fine-tuning loss is

    L = L_ppo + rho * KL(prior, policy)

with rho decayed each iteration and **no entropy bonus**. VPT's argument for
replacing the bonus is the one this task keeps running into: "Blindly exploring
by maximizing entropy is effective when the state and action spaces are
sufficiently small or the reward is sufficiently dense, but becomes infeasible
when the state and action spaces are large and rewards are sparse."

A run that uses a prior is not a from-scratch run, and `train.py` records it.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from fsim import expert, lib, scenes
from fsim.rl import NVEC, RlEnv

#: One sample per builder decision: the state it saw, what it was allowed to
#: do, and what it did.
FIELDS = ("grid", "entities", "entity_mask", "self", "inventory", "goal", "mask", "action")


def _episode(seed: int, task: str, action_space: str, env: RlEnv) -> list[dict] | None:
    """One demonstration, or None when the builder cannot give one."""
    family, scene = scenes.sample(task, "train", seed)
    if scene["entities"]:
        return None  # the builder walks in straight lines; this one has walls
    env.reset(task, scene, action_space=action_space)
    patch = scene["markers"]["patch"]
    layout = expert.choose_layout(env.rl, patch, random.Random(seed * 31 + 7))
    builder = expert.Builder(env.rl, patch, layout=layout)
    out = []
    while (vector := builder.next_vector()) is not None:
        out.append(
            {
                "grid": env.obs["grid"].copy(),
                "entities": env.obs["entities"].copy(),
                "entity_mask": env.obs["entity_mask"].copy(),
                "self": env.obs["self"].copy(),
                "inventory": env.obs["inventory"].copy(),
                "goal": env.obs["goal"].copy(),
                "mask": env.mask.copy(),
                "action": np.asarray(vector, dtype=np.int64),
            }
        )
        _, _, term, trunc, _ = env.step(vector)
        if term or trunc:
            break
    if not env.rl.success and not out:
        return None
    return out


def collect(
    episodes: int,
    task: str = "construct_smelting_line",
    action_space: str = "v2",
    threads: int = 8,
    seed_base: int = 5_000_000,
) -> dict[str, np.ndarray]:
    """`episodes` demonstrations, as one array per field.

    The seeds sit above the training seeds `VecEnv` draws from, so cloning the
    builder does not hand the policy the very episodes it is scored on.
    """
    envs = [RlEnv() for _ in range(threads)]
    samples: list[list[dict]] = [[] for _ in range(threads)]

    def worker(slot: int) -> None:
        for seed in range(seed_base + slot, seed_base + episodes, threads):
            got = _episode(seed, task, action_space, envs[slot])
            if got:
                samples[slot].extend(got)

    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(worker, range(threads)))
    flat = [s for part in samples for s in part]
    if not flat:
        raise RuntimeError("the builder produced no demonstrations")
    out = {f: np.stack([s[f] for s in flat]) for f in FIELDS}
    assert out["action"].shape[1] == len(NVEC)
    assert out["mask"].shape[1] == lib.RL_MASK_SIZE
    return out
