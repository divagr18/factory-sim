"""belt_smelting's demonstrations: recorded builds, and starts cut from them.

A demonstration start replays the builder's decisions in C on a fresh reset
of the scene they were recorded on. The simulator is deterministic, so the
replay must reach exactly the state the build reached -- the same tensors and
masks -- whether the build is replayed from a recorded pool or from the
builder run live.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from fsim import demos, scenes
from fsim.rl import TASKS, RlEnv
from fsim.vec import OBS_KEYS, OP_FINISH, VecEnv

pytestmark = pytest.mark.skipif("belt_smelting" not in TASKS, reason="no belt_smelting")
belt_expert = pytest.importorskip("fsim.belt_expert")

TASK = "belt_smelting"


def _built_state(seed: int, rng=None, split: str = "train"):
    """The v3 observation and masks after the builder's last decision before
    `finish`, and the vectors it took to get there."""
    _family, scene = scenes.sample(TASK, split, seed)
    env = RlEnv()
    env.reset(TASK, scene, action_space="v3")
    builder = belt_expert.BeltBuilder(env.rl, scene, rng=rng)
    vectors = []
    while (vector := builder.next_vector()) is not None and vector[0] != OP_FINISH:
        env.step(vector)
        vectors.append([int(v) for v in vector])
    obs, mask = env.observe3()
    return {k: v.copy() for k, v in obs.items()}, mask.copy(), env.op_masks().copy(), vectors


def _check(vec, i, obs, mask, op_masks):
    for key in OBS_KEYS:
        assert np.array_equal(vec.obs[key][i], obs[key]), key
    assert np.array_equal(vec.masks[i], mask)
    assert np.array_equal(vec.op_masks[i], op_masks)


def test_a_recorded_build_replays_to_the_state_it_reached():
    records: list[dict] = []
    data = demos.collect(2, task=TASK, threads=2, seed_base=0, records=records)
    assert int(data["episodes_kept"]) == len(records) == 2
    assert data["op_masks"].shape[1:] == (25, 351)
    assert data["grid"].dtype == np.uint8
    # Every label is legal under its own operation's mask.
    offsets = np.cumsum((0, 97, 226, 5, 19, 4))
    for action, mask, op_masks in zip(data["action"], data["mask"], data["op_masks"], strict=True):
        assert mask[action[0]]
        for d in range(5):
            assert op_masks[action[0], offsets[d] + action[d + 1]]
    for record in records:
        # demos.collect draws the builder's layout with this generator.
        rng = random.Random(record["seed"] * 31 + 7)
        obs, mask, op_masks, vectors = _built_state(record["seed"], rng)
        assert [v for v in record["vectors"] if v[0] != OP_FINISH] == vectors
        vec = VecEnv(1, TASK, threads=1, action_space="v3", max_steps=2500, demo_starts=1.0)
        vec.demo_pool = [record]
        vec.demo_window = (0, 0)  # the whole build, cut at its end
        vec.reset()
        assert vec.starts[0] == "back0" and vec.lengths[0] == len(vectors)
        _check(vec, 0, obs, mask, op_masks)
        vec.close()


def test_a_live_build_replays_to_the_state_it_reached():
    seed = 7
    obs, mask, op_masks, vectors = _built_state(seed)
    vec = VecEnv(1, TASK, threads=1, action_space="v3", max_steps=2500, demo_starts=1.0)
    vec.demo_window = (0, 0)
    # The slot's first scene is seed 0 of its own counter; point it at `seed`.
    vec.episodes_started = seed
    vec.demo_layouts = False  # the builder's own layout, as _built_state runs it
    vec.reset()
    if vec.starts[0] == "scene":
        pytest.skip("this scene's slot drew no demonstration")
    assert vec.lengths[0] == len(vectors)
    _check(vec, 0, obs, mask, op_masks)
    vec.close()


def test_fractional_windows_cut_the_build_proportionally():
    records: list[dict] = []
    demos.collect(1, task=TASK, threads=1, seed_base=3, records=records)
    length = len([v for v in records[0]["vectors"] if v[0] != OP_FINISH])
    vec = VecEnv(4, TASK, threads=1, action_space="v3", max_steps=2500, demo_starts=1.0)
    vec.demo_pool = records
    vec.demo_window = (0.5, 0.5)
    vec.reset()
    for i in range(4):
        assert vec.starts[i] == f"back{round(0.5 * length)}"
        assert vec.lengths[i] == length - round(0.5 * length)
    vec.demo_window = (1.0, 1.0)  # the last rung: the scene's own start
    vec.reset()
    assert all(start == "scene" for start in vec.starts)
    vec.close()
