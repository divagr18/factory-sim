"""A batch of RL environments stepped on a thread pool, with automatic reset.

The C side does the work: `fsim_rl_step_range` steps a contiguous range of
environments and writes each one's observation, mask and transition into shared
arrays, and cffi releases the GIL for the call, so `threads` ranges run in
parallel. Python only draws a scene when an episode ends.

Observations come back as numpy views over one `fsim_obs[n]` block, laid out by
the struct's own offsets. They are overwritten by the next `step`, so a caller
that keeps them copies them (the trainer copies straight to the GPU).

Autoreset follows the same-step convention: when an episode ends, the returned
observation and mask already belong to the next episode, and `step` reports the
finished episode in its `episodes` list.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from fsim import Sim, expert, ffi, lib, scene_struct, scenes
from fsim.rl import task_struct

#: Episode seeds for evaluation start here; training seeds stay below it.
EVAL_SEED_BASE = 1 << 40


OBS_KEYS = ("grid", "entities", "entity_mask", "self", "inventory", "goal")
#: A compact observation's fields: the grid is `flags` and `amount`.
PACKED_KEYS = ("flags", "amount", "entities", "entity_mask", "self", "inventory", "goal")
GRID = 65
FLAG_PLANES = (0, 1, 2, 3, 5)  # the grid planes packed as bits, in bit order


def _obs_dtype(compact: bool = False) -> np.dtype:
    struct = "fsim_obs8" if compact else "fsim_obs"
    grid = (
        {"flags": ("u1", (lib.RL_FLAG_BYTES,)), "amount": ("u1", (GRID, GRID))}
        if compact
        else {"grid": ("<f4", (6, GRID, GRID))}
    )
    fields = {
        **grid,
        "entities": ("<f4", (32, 16)),
        "entity_mask": ("i1", (32,)),
        "self_": ("<f4", (12,)),
        "inventory": ("<f4", (14,)),
        "goal": ("<f4", (12,)),
    }
    return np.dtype(
        {
            "names": list(fields),
            "formats": list(fields.values()),
            "offsets": [ffi.offsetof(struct, name) for name in fields],
            "itemsize": ffi.sizeof(struct),
        }
    )


def obs_layout(compact: bool = False) -> dict[str, tuple[int, str, tuple[int, ...]]]:
    """key -> (byte offset in `fsim_obs`/`fsim_obs8`, numpy dtype, shape): for
    slicing a copied block of observations without going through numpy."""
    dtype = _obs_dtype(compact)
    out = {}
    for key in PACKED_KEYS if compact else OBS_KEYS:
        name = "self_" if key == "self" else key
        sub, offset = dtype.fields[name][:2]
        out[key] = (offset, sub.base.str, sub.shape)
    return out


_BIT_TABLES: dict = {}


def unpack_grid(flags, amount, xp=np):
    """The byte grid, (n, 6, 65, 65) with values round(255 * v), from packed
    fields. `xp` is numpy, or torch for tensors on any device."""
    n = flags.shape[0]
    cells = GRID * GRID
    if xp is np:
        bits = np.unpackbits(flags, axis=1, bitorder="little")[:, : 5 * cells]
        planes = bits.reshape(n, 5, GRID, GRID) * np.uint8(255)
        amount = amount.reshape(n, 1, GRID, GRID)
        return np.concatenate([planes[:, :4], amount, planes[:, 4:]], axis=1)
    table = _BIT_TABLES.get(flags.device)
    if table is None:  # made once, outside any CUDA graph capture
        rows = [[255 if (byte >> k) & 1 else 0 for k in range(8)] for byte in range(256)]
        table = xp.tensor(rows, dtype=xp.uint8, device=flags.device)
        _BIT_TABLES[flags.device] = table
    # One gather: each packed byte becomes its eight cells, already 0 or 255.
    planes = table[flags.long()].view(n, -1)[:, : 5 * cells].view(n, 5, GRID, GRID)
    amount = amount.view(n, 1, GRID, GRID)
    return xp.cat([planes[:, :4], amount, planes[:, 4:]], dim=1)


class VecEnv:
    def __init__(
        self,
        n: int,
        task: str = "construct_smelting_line",
        *,
        split: str = "train",
        seed: int = 0,
        threads: int = 8,
        shaping: bool = False,
        gamma: float = 0.999,
        start_curriculum: float = 0.0,
        max_steps: int | tuple[int, int] = 600,
        eval_seeds: bool = False,
        obs_memory=None,
        demo_starts: float = 0.0,
        compact: bool = False,
        mask_memory=None,
    ) -> None:
        """`obs_memory`, if given, is `(address, owner)`: `n * sizeof(fsim_obs)`
        bytes the observations are written into instead of a fresh block -- a
        trainer passes page-locked memory so the batch reaches the GPU in one
        fast copy. `owner` is kept alive with the environment.

        `demo_starts` is the fraction of episodes that begin partway along the
        scripted build (`fsim.expert`), at a stage drawn uniformly; the rest
        begin at the scene's own start. construct_smelting_line only.

        `mask_memory` is the same for the `n * 201` mask bytes.

        `compact` writes `fsim_obs8` observations: the grid's flag planes as
        bits and its amount plane as bytes, round(255 * value) -- which the
        policy's input rounding makes identical to the float grid as far as
        the network can tell. `obs` then has `flags` and `amount` in place of
        `grid`; `unpack_grid` rebuilds it."""
        self.n = n
        self.task = task
        self.split = split
        self.shaping = shaping
        self.gamma = gamma
        self.start_curriculum = start_curriculum
        self.max_steps = max_steps
        self.seed_base = (EVAL_SEED_BASE if eval_seeds else 0) + seed * 1_000_003
        self.episodes_started = 0
        if demo_starts and task != "construct_smelting_line":
            raise ValueError("demonstration starts exist for construct_smelting_line only")
        self.demo_starts = demo_starts
        self._step_vector = ffi.new("int32_t[6]")

        self._sim = Sim()  # owns the map's water; `env` dies with it
        water = self._sim.env
        self._water = ffi.new("int32_t[]", max(1, 2 * water.water_count))
        ffi.memmove(self._water, water.water, 4 * 2 * water.water_count)
        self.rls = ffi.new("fsim_rl *[]", n)
        for i in range(n):
            self.rls[i] = lib.fsim_rl_new()
            lib.fsim_set_water(self.rls[i].env, self._water, water.water_count)

        self.compact = compact
        struct = "fsim_obs8" if compact else "fsim_obs"
        if obs_memory is None:
            self._obs_c = ffi.new(f"{struct}[]", n)
            self._obs_owner = None
        else:
            address, self._obs_owner = obs_memory
            self._obs_c = ffi.cast(f"{struct} *", address)
        self._encode = lib.fsim_rl_encode8 if compact else lib.fsim_rl_encode
        if mask_memory is None:
            self._masks_c = ffi.new("uint8_t[]", n * lib.RL_MASK_SIZE)
            self._mask_owner = None
        else:
            address, self._mask_owner = mask_memory
            self._masks_c = ffi.cast("uint8_t *", address)
        self._actions_c = ffi.new("int32_t[]", n * 6)
        self._rewards_c = ffi.new("double[]", n)
        self._flags_c = ffi.new("uint8_t[]", n * 4)
        self._verified_c = ffi.new("double[]", n)

        self.obs_nbytes = n * ffi.sizeof(struct)
        block = np.frombuffer(ffi.buffer(self._obs_c, self.obs_nbytes), _obs_dtype(compact))
        keys = PACKED_KEYS if compact else OBS_KEYS
        self.obs = {key: block["self_" if key == "self" else key] for key in keys}
        mask_bytes = n * lib.RL_MASK_SIZE
        self.masks = np.frombuffer(ffi.buffer(self._masks_c, mask_bytes), np.uint8).reshape(n, -1)
        self.actions = np.frombuffer(ffi.buffer(self._actions_c), np.int32).reshape(n, 6)
        self.rewards = np.frombuffer(ffi.buffer(self._rewards_c), np.float64)
        self.flags = np.frombuffer(ffi.buffer(self._flags_c), np.uint8).reshape(n, 4)
        self.verified = np.frombuffer(ffi.buffer(self._verified_c), np.float64)
        self._potentials_c = ffi.new("double[]", n)
        self.potentials = np.frombuffer(ffi.buffer(self._potentials_c), np.float64)

        self.threads = max(1, min(threads, n))
        bounds = np.linspace(0, n, self.threads + 1).astype(int)
        self._ranges = [
            (int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:], strict=True) if b > a
        ]
        self._pool = ThreadPoolExecutor(self.threads) if self.threads > 1 else None
        self._keep: list = [None] * n
        self.families: list[str] = [""] * n
        self.starts: list[str] = [""] * n
        self.returns = np.zeros(n)
        self.peak_potential = np.zeros(n)
        self.lengths = np.zeros(n, dtype=np.int64)

    def close(self) -> None:
        if getattr(self, "_pool", None) is not None:
            self._pool.shutdown()
            self._pool = None
        rls = getattr(self, "rls", None)
        if rls is not None and lib is not None:
            for i in range(self.n):
                lib.fsim_rl_free(rls[i])
            self.rls = None

    def __del__(self) -> None:
        self.close()

    def _reset_one(self, i: int) -> None:
        seed = self.seed_base + self.episodes_started
        self.episodes_started += 1
        family, scene = scenes.sample(self.task, self.split, seed, self.start_curriculum)
        c_scene, keep = scene_struct(scene)
        budget = self.max_steps
        if not isinstance(budget, int):
            # A horizon curriculum: each episode's decision budget is drawn from
            # [lo, hi], so a line built early is verified sooner. Seeded by the
            # episode, like the scene.
            lo, hi = budget
            budget = lo + (seed * 2654435761 + 97) % (hi - lo + 1)
        task = task_struct(
            self.task, scene, max_steps=budget, shaping=self.shaping, gamma=self.gamma
        )
        self._keep[i] = (c_scene, keep, task)
        rl = self.rls[i]
        lib.fsim_rl_reset(rl, task, c_scene)
        start, taken = "scene", 0
        draw = random.Random(seed * 7 + 3)
        if self.demo_starts and draw.random() < self.demo_starts:
            start = draw.choice(expert.STAGES)
            taken = expert.advance_to(rl, scene["markers"]["patch"], start, self._demo_step(rl))
        self._encode(self.rls[i], ffi.addressof(self._obs_c, i))
        lib.fsim_rl_mask(self.rls[i], ffi.addressof(self._masks_c, i * lib.RL_MASK_SIZE))
        self.families[i] = family
        self.starts[i] = start
        self.returns[i] = 0.0
        self.lengths[i] = taken
        self.peak_potential[i] = lib.fsim_rl_potential(self.rls[i])

    def _demo_step(self, rl):
        vector = self._step_vector

        def step(values) -> bool:
            for k, value in enumerate(values):
                vector[k] = value
            lib.fsim_rl_step(rl, vector)
            return bool(rl.done)

        return step

    def reset(self) -> tuple[dict, np.ndarray]:
        for i in range(self.n):
            self._reset_one(i)
        return self.obs, self.masks

    def _run(self, bounds) -> None:
        if self.compact:
            lib.fsim_rl_step_range8(
                self.rls, bounds[0], bounds[1], self._actions_c, self._obs_c, self._masks_c,
                self._rewards_c, self._flags_c, self._verified_c, self._potentials_c,
            )  # fmt: skip
            return
        lib.fsim_rl_step_range(
            self.rls, bounds[0], bounds[1], self._actions_c, self._obs_c, self._masks_c,
            self._rewards_c, self._flags_c, self._verified_c,
        )  # fmt: skip
        potential = lib.fsim_rl_potential
        for i in range(bounds[0], bounds[1]):
            self._potentials_c[i] = potential(self.rls[i])

    def step(self, actions: np.ndarray):
        """-> obs, masks, rewards, terminated, truncated, finished episodes."""
        self.actions[:] = actions
        if self._pool is None:
            self._run((0, self.n))
        else:
            list(self._pool.map(self._run, self._ranges))
        rewards = self.rewards.copy()
        terminated = self.flags[:, 0].astype(bool)
        truncated = self.flags[:, 1].astype(bool)
        self.returns += rewards
        self.lengths += 1
        np.maximum(self.peak_potential, self.potentials, out=self.peak_potential)
        episodes = []
        for i in np.flatnonzero(terminated | truncated).tolist():
            episodes.append(
                {
                    "return": float(self.returns[i]),
                    "length": int(self.lengths[i]),
                    "success": bool(self.flags[i, 2]),
                    "verified_output": float(self.verified[i]),
                    "family": self.families[i],
                    "start": self.starts[i],
                    "decode_failures": int(self.rls[i].decode_failures),
                    # The line potential's peak, whether or not the run shapes
                    # with it: how far towards a working line the episode got.
                    "peak_potential": float(self.peak_potential[i]),
                }
            )
            self._reset_one(i)
        return self.obs, self.masks, rewards, terminated, truncated, episodes
