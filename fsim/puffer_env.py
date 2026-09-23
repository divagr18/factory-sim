"""Native PufferLib (3.x) environment over the batched C env.

`FactorySimPufferEnv` is a `pufferlib.PufferEnv`: one object holding
`num_envs` agents, with PufferLib's shared buffers. It is zero-copy for the
observations and masks: `VecEnv` is handed the address of PufferLib's
`observations` buffer (and of `action_masks`), so the C step writes each
agent's observation straight into the row PufferLib reads.

A native PufferEnv's observation must be a single `Box`, so each row is the C
struct's own bytes -- `fsim_obs8` (`compact=True`, the default: the grid packed
as bits and bytes, 9,100 bytes) or `fsim_obs` (the float grid, 103,632 bytes) -- as
`uint8`. `decode_obs` turns a batch of rows back into the named fields; a
policy can do the same on the GPU with `fsim.vec.obs_layout` and
`fsim.vec.unpack_grid(..., xp=torch)`, as `train.py` does.

Action masks: PufferLib 3.0's trainer does not consume action masks
(`pufferl.py`: "We are not yet handling masks"; its `masks` buffer marks live
agents, not legal actions). The flat (num_envs, 201) legal-action mask is in
`env.action_masks`, refreshed in place by every `reset`/`step`, for a policy
that masks its logits itself. Illegal actions are safe: the simulator counts
them as decode failures.

Resets are internal (same-step autoreset, like `VecEnv`): a finished agent's
row already holds its next episode when `step` returns. Each finished episode
is reported as one numeric dict in the returned info list.

Installing: PufferLib 3.0.0 ships only an sdist whose `setup.py` raises
"Unsupported system: Windows", and pins `numpy<2` and `gymnasium<=0.29.1`,
which conflict with this package's `numpy>=2`. The Python modules this adapter
uses (`pufferlib.PufferEnv`, `pufferlib.vector`, `pufferlib.emulation`) run
fine on numpy 2 and gymnasium 1.x, so on Linux install it with
`pip install --no-deps pufferlib` after `pip install factory-sim[puffer]`.
"""

from __future__ import annotations

import numpy as np
import pufferlib
import pufferlib.spaces  # noqa: F401  (PufferEnv's space checks)
from gymnasium import spaces

from fsim import ffi, lib
from fsim.gym_env import ACTION_SPACE, TASK_DEFAULTS, action_space
from fsim.rl import TASKS
from fsim.vec import OBS_KEYS, PACKED_KEYS, VecEnv, obs_layout, unpack_grid


def obs_nbytes(compact: bool = True) -> int:
    """Bytes in one observation row."""
    return ffi.sizeof("fsim_obs8" if compact else "fsim_obs")


def decode_obs(rows: np.ndarray, compact: bool = True, unpack: bool = True) -> dict:
    """(n, obs_nbytes) uint8 rows -> dict of (n, ...) arrays (copies).

    With `compact` and `unpack`, the packed grid comes back as `grid`, float32
    in [0, 1] -- the same values as the uncompact observation's grid up to the
    byte rounding (`fsim.vec.unpack_grid`). With `unpack=False` a compact batch
    keeps its `flags` and `amount` fields.
    """
    rows = np.ascontiguousarray(rows, dtype=np.uint8)
    n = rows.shape[0]
    out = {}
    for key, (offset, dtype, shape) in obs_layout(compact).items():
        dt = np.dtype(dtype)
        size = int(np.prod(shape)) * dt.itemsize
        out[key] = rows[:, offset : offset + size].copy().view(dt).reshape(n, *shape)
    if compact and unpack:
        grid = unpack_grid(out.pop("flags"), out.pop("amount"))
        out = {"grid": grid.astype(np.float32) / np.float32(255), **out}
        out = {key: out[key] for key in OBS_KEYS}
    return out


class FactorySimPufferEnv(pufferlib.PufferEnv):
    """`num_envs` factory-sim agents as one native PufferEnv.

    `buf` and `seed` are what PufferLib's `Serial`/`Multiprocessing` backends
    pass; with the native backend (`pufferlib.vector.make(FactorySimPufferEnv,
    backend=pufferlib.PufferEnv, env_kwargs=...)`) this one object is the whole
    vector.
    """

    def __init__(
        self,
        num_envs: int = 64,
        task: str = "construct_smelting_line",
        split: str = "train",
        *,
        threads: int = 8,
        compact: bool = True,
        max_steps: int | None = None,
        tick_limit: int | None = None,
        shaping: bool | str = False,
        gamma: float = 0.999,
        start_curriculum: float = 0.0,
        buf=None,
        seed: int | None = 0,
    ) -> None:
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}; expected one of {sorted(TASKS)}")
        defaults = TASK_DEFAULTS[task]
        max_steps = defaults["max_steps"] if max_steps is None else max_steps
        tick_limit = defaults["tick_limit"] if tick_limit is None else tick_limit
        self.task = task
        self.split = split
        self.compact = compact
        self.num_agents = num_envs
        self.single_observation_space = spaces.Box(0, 255, (obs_nbytes(compact),), dtype=np.uint8)
        self.single_action_space = action_space()
        super().__init__(buf)
        if self.observations.dtype != np.uint8 or not self.observations.flags.c_contiguous:
            raise ValueError("observation buffer must be C-contiguous uint8 rows")
        self.action_masks = np.zeros((num_envs, lib.RL_MASK_SIZE), dtype=np.uint8)
        self.vec = VecEnv(
            num_envs, task, split=split, seed=0 if seed is None else seed, threads=threads,
            shaping=shaping, gamma=gamma, start_curriculum=start_curriculum,
            max_steps=max_steps, tick_limit=tick_limit, action_space=ACTION_SPACE,
            compact=compact,
            obs_memory=(self.observations.ctypes.data, self.observations),
            mask_memory=(self.action_masks.ctypes.data, self.action_masks),
        )  # fmt: skip

    @property
    def obs_keys(self) -> tuple[str, ...]:
        return PACKED_KEYS if self.compact else OBS_KEYS

    def decode(self, unpack: bool = True) -> dict:
        """The current observations as named fields (copies)."""
        return decode_obs(self.observations, self.compact, unpack)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.vec.seed_base = int(seed) * 1_000_003
            self.vec.episodes_started = 0
        self.vec.reset()
        self.rewards[:] = 0
        self.terminals[:] = False
        self.truncations[:] = False
        return self.observations, []

    def step(self, actions):
        _, _, rewards, terminated, truncated, episodes = self.vec.step(actions)
        self.rewards[:] = rewards
        self.terminals[:] = terminated
        self.truncations[:] = truncated
        infos = [
            {
                "episode_return": e["return"],
                "episode_length": e["length"],
                "success": float(e["success"]),
                "verified_output": e["verified_output"],
                "decode_failures": e["decode_failures"],
                "peak_potential": e["peak_potential"],
            }
            for e in episodes
        ]
        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def close(self) -> None:
        self.vec.close()
