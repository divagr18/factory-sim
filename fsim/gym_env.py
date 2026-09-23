"""Gymnasium adapters: one environment, and a vector over the batched C env.

`FactorySimEnv` is a `gymnasium.Env` around `RlEnv`; `FactorySimVectorEnv` is a
`gymnasium.vector.VectorEnv` around `VecEnv`. Both always speak the v2 action
space -- a v2 policy read as v1 fails silently -- and both expose the action
mask the way masked-MultiDiscrete learners expect it: `info["action_mask"]` and
`action_masks()`, the per-dimension masks concatenated in `NVEC` order (201
entries). `split_mask` cuts it back into dimensions.

The vector env uses Gymnasium's **same-step** autoreset (`AutoresetMode.
SAME_STEP`), because that is what `VecEnv` does in C: when an episode ends, the
observation and mask `step` returns already belong to the next episode, and the
finished episode's last observation and summary are in `info["final_obs"]` and
`info["final_info"]` (with `info["_final_obs"]` marking which slots finished).

Importing this module registers `fsim/ConstructSmeltingLine-v0`,
`fsim/BuildLine-v0` and `fsim/PlateLine-v0`; `split="test"` selects the held-out
family, e.g. `gymnasium.make("fsim/BuildLine-v0", split="test")`.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from fsim import scenes
from fsim.rl import NVEC, TASKS, RlEnv
from fsim.vec import OBS_KEYS, VecEnv

ACTION_SPACE = "v2"
MASK_SIZE = int(sum(NVEC))
_SPLITS = np.cumsum(NVEC)[:-1]

#: Per-task episode budgets, as FactorioRL sets them (see `train.py --help`).
TASK_DEFAULTS = {
    "construct_smelting_line": {"max_steps": 600, "tick_limit": None},
    "build_line": {"max_steps": 600, "tick_limit": None},
    "plate_line": {"max_steps": 400, "tick_limit": 24000},
}

ENV_IDS = {
    "fsim/ConstructSmeltingLine-v0": "construct_smelting_line",
    "fsim/BuildLine-v0": "build_line",
    "fsim/PlateLine-v0": "plate_line",
}


def observation_space() -> spaces.Dict:
    """One observation. Every field the C encoder writes is clipped to these bounds."""
    f32 = np.float32
    return spaces.Dict(
        {
            "grid": spaces.Box(0.0, 1.0, (6, 65, 65), f32),
            "entities": spaces.Box(-1.0, 1.0, (32, 16), f32),
            "entity_mask": spaces.Box(0, 1, (32,), np.int8),
            "self": spaces.Box(-1.0, 1.0, (12,), f32),
            "inventory": spaces.Box(0.0, 1.0, (14,), f32),
            "goal": spaces.Box(-1.0, 1.0, (12,), f32),
        }
    )


def action_space() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete(np.array(NVEC, dtype=np.int64))


def split_mask(mask: np.ndarray) -> list[np.ndarray]:
    """The flat (..., 201) mask as one boolean array per action dimension."""
    return np.split(np.asarray(mask, dtype=bool), _SPLITS, axis=-1)


def sample_masked(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A uniformly random legal action per row of a (n, 201) or (201,) mask."""
    mask = np.asarray(mask, dtype=bool)
    single = mask.ndim == 1
    mask = np.atleast_2d(mask)
    out = np.empty((mask.shape[0], len(NVEC)), dtype=np.int64)
    for d, part in enumerate(split_mask(mask)):
        keys = np.where(part, rng.random(part.shape), -1.0)
        out[:, d] = keys.argmax(axis=1)
    return out[0] if single else out


def _options(task: str, max_steps, tick_limit) -> tuple[int, int | None]:
    defaults = TASK_DEFAULTS[task]
    return (
        defaults["max_steps"] if max_steps is None else max_steps,
        defaults["tick_limit"] if tick_limit is None else tick_limit,
    )


_PALETTE = np.array(
    [
        [110, 130, 160],  # iron ore
        [200, 110, 50],  # copper ore
        [30, 30, 30],  # coal
        [170, 150, 110],  # stone
    ],
    dtype=np.float32,
)


def render_obs(obs: dict, scale: int = 4) -> np.ndarray:
    """An RGB picture of one observation: resources by type (brighter where
    richer), blocked tiles white, entities yellow, the character red at the
    centre. (65 * scale, 65 * scale, 3) uint8."""
    grid = np.asarray(obs["grid"], dtype=np.float32)
    img = np.full((65, 65, 3), 60.0, np.float32)
    shade = 0.5 + 0.5 * grid[4]
    for plane in range(4):
        on = grid[plane] > 0
        img[on] = _PALETTE[plane] * shade[on, None]
    img[grid[5] > 0] = 240.0
    for row, present in zip(obs["entities"], obs["entity_mask"], strict=True):
        if present:
            col = int(np.clip(round(row[0] * 32) + 32, 0, 64))
            r = int(np.clip(round(row[1] * 32) + 32, 0, 64))
            img[r, col] = (240, 220, 40)
    img[32, 32] = (230, 30, 30)
    img = img.astype(np.uint8)
    return img.repeat(scale, axis=0).repeat(scale, axis=1)


class FactorySimEnv(gym.Env):
    """One factory-sim episode at a time, drawn from `scenes.sample(task, split)`.

    `reset(seed=s)` seeds Gymnasium's RNG, and every reset draws its scene seed
    from that RNG, so a seeded reset and the resets that follow it are
    reproducible. `options={"scene_seed": k}` runs the scene `scenes.sample`
    draws for seed `k` directly, bypassing the RNG.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        task: str = "construct_smelting_line",
        split: str = "train",
        *,
        max_steps: int | None = None,
        tick_limit: int | None = None,
        shaping: bool | str = False,
        gamma: float = 0.999,
        start_curriculum: float = 0.0,
        render_mode: str | None = None,
    ) -> None:
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}; expected one of {sorted(TASKS)}")
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', not {split!r}")
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.task = task
        self.split = split
        self.max_steps, self.tick_limit = _options(task, max_steps, tick_limit)
        self.shaping = shaping
        self.gamma = gamma
        self.start_curriculum = start_curriculum
        self.render_mode = render_mode
        self.observation_space = observation_space()
        self.action_space = action_space()
        self._rl = RlEnv()
        self.family: str | None = None
        self.scene_seed: int | None = None

    def _obs(self) -> dict:
        return {key: self._rl.obs[key].copy() for key in OBS_KEYS}

    def action_masks(self) -> np.ndarray:
        """The current flat (201,) mask as booleans (sb3-contrib's convention)."""
        return self._rl.mask.astype(bool)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        scene_seed = options.get("scene_seed")
        if scene_seed is None:
            scene_seed = int(self.np_random.integers(0, 2**62))
        self.scene_seed = int(scene_seed)
        self.family, scene = scenes.sample(
            self.task, self.split, self.scene_seed, self.start_curriculum
        )
        self._rl.reset(
            self.task, scene, max_steps=self.max_steps, shaping=self.shaping, gamma=self.gamma,
            construction_tick_limit=self.tick_limit, action_space=ACTION_SPACE,
        )  # fmt: skip
        info = {
            "action_mask": self.action_masks(),
            "family": self.family,
            "scene_seed": self.scene_seed,
        }
        return self._obs(), info

    def step(self, action):
        _, reward, terminated, truncated, info = self._rl.step(action)
        info["action_mask"] = self.action_masks()
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return render_obs(self._rl.obs)
        return None

    def close(self) -> None:
        self._rl = None


class FactorySimVectorEnv(VectorEnv):
    """`num_envs` episodes stepped together in C, on `threads` threads.

    Same-step autoreset (see the module docstring). `reset(seed=s)` restarts
    the scene sequence at `VecEnv`'s seed base for `s` (slot `i`'s first scene is
    `scenes.sample(task, split, s * 1_000_003 + i)`, later scenes continue the
    same counter); `reset()` without a seed keeps drawing fresh scenes.

    `copy=False` returns the C buffers' own (strided) views, which the next
    `step` overwrites -- faster, but a caller that keeps them must copy.
    """

    def __init__(
        self,
        num_envs: int = 8,
        task: str = "construct_smelting_line",
        split: str = "train",
        *,
        threads: int = 8,
        seed: int = 0,
        max_steps: int | None = None,
        tick_limit: int | None = None,
        shaping: bool | str = False,
        gamma: float = 0.999,
        start_curriculum: float = 0.0,
        copy: bool = True,
    ) -> None:
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}; expected one of {sorted(TASKS)}")
        max_steps, tick_limit = _options(task, max_steps, tick_limit)
        self.task = task
        self.split = split
        self.copy = copy
        self.num_envs = num_envs
        self.metadata = {"autoreset_mode": AutoresetMode.SAME_STEP}
        self.render_mode = None
        self.single_observation_space = observation_space()
        self.single_action_space = action_space()
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.vec = VecEnv(
            num_envs, task, split=split, seed=seed, threads=threads, shaping=shaping,
            gamma=gamma, start_curriculum=start_curriculum, max_steps=max_steps,
            tick_limit=tick_limit, action_space=ACTION_SPACE,
        )  # fmt: skip
        # VecEnv resets a finished slot inside `step`; catch each slot's last
        # observation just before that reset overwrites it.
        self._final: dict[int, tuple[dict, np.ndarray]] | None = None
        reset_one = self.vec._reset_one

        def capture(i, *args, **kwargs):
            if self._final is not None:
                obs = {key: self.vec.obs[key][i].copy() for key in OBS_KEYS}
                self._final[i] = (obs, self.vec.masks[i].astype(bool))
            reset_one(i, *args, **kwargs)

        self.vec._reset_one = capture

    def _obs(self) -> dict:
        obs = self.vec.obs
        if self.copy:
            return {key: np.ascontiguousarray(obs[key]) for key in OBS_KEYS}
        return {key: obs[key] for key in OBS_KEYS}

    def action_masks(self) -> np.ndarray:
        """The current (num_envs, 201) masks as booleans."""
        return self.vec.masks.astype(bool)

    def reset(self, *, seed: int | list[int] | None = None, options: dict | None = None):
        if seed is not None:
            if isinstance(seed, (list, tuple)):
                seed = seed[0]
            super().reset(seed=int(seed))
            self.vec.seed_base = int(seed) * 1_000_003
            self.vec.episodes_started = 0
        self._final = None
        self.vec.reset()
        info = {
            "action_mask": self.action_masks(),
            "_action_mask": np.ones(self.num_envs, dtype=bool),
            "family": np.array(self.vec.families, dtype=object),
            "_family": np.ones(self.num_envs, dtype=bool),
        }
        return self._obs(), info

    def step(self, actions):
        actions = np.asarray(actions)
        self._final = {}
        try:
            _, _, rewards, terminated, truncated, episodes = self.vec.step(actions)
        finally:
            final, self._final = self._final, None
        n = self.num_envs
        info: dict[str, Any] = {
            "action_mask": self.action_masks(),
            "_action_mask": np.ones(n, dtype=bool),
        }
        if final:
            finished = np.zeros(n, dtype=bool)
            final_obs = np.full(n, None, dtype=object)
            final_info = np.full(n, None, dtype=object)
            done = np.flatnonzero(terminated | truncated).tolist()
            for i, episode in zip(done, episodes, strict=True):
                obs, mask = final[i]
                finished[i] = True
                final_obs[i] = obs
                final_info[i] = {**episode, "action_mask": mask}
            info.update(
                final_obs=final_obs, _final_obs=finished,
                final_info=final_info, _final_info=finished.copy(),
            )  # fmt: skip
        return self._obs(), rewards, terminated, truncated, info

    def close_extras(self, **kwargs) -> None:
        self.vec.close()


def _register() -> None:
    for env_id, task in ENV_IDS.items():
        if env_id in gym.registry:
            continue
        gym.register(
            id=env_id,
            entry_point="fsim.gym_env:FactorySimEnv",
            vector_entry_point="fsim.gym_env:FactorySimVectorEnv",
            kwargs={"task": task},
            disable_env_checker=False,
        )


_register()
