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

`action_space="v3"` steps through `fsim_rl_step_range3`: the observations are
`fsim_obs3` (or packed, `fsim_obs38`), `masks` is the 376-entry v3 mask, and
`op_masks` holds each environment's per-operation masks, `(n, 25, 351)` --
what `RlEnv.observe3()` and `RlEnv.op_masks()` return, bit for bit.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from fsim import Sim, expert, ffi, lib, scene_struct, scenes
from fsim.rl import task_struct

#: Episode seeds for evaluation start here; training seeds stay below it.
EVAL_SEED_BASE = 1 << 40
#: v3's `finish`: a demonstration is cut before it, never at it.
OP_FINISH = 24
#: Decisions belt_smelting's builder may take before its build is judged stuck.
BELT_DEMO_LIMIT = 2400


OBS_KEYS = ("grid", "entities", "entity_mask", "self", "inventory", "goal")
#: A compact observation's fields: the grid is `flags` and `amount`.
PACKED_KEYS = ("flags", "amount", "entities", "entity_mask", "self", "inventory", "goal")
GRID = 65
FLAG_PLANES = (0, 1, 2, 3, 5)  # the grid planes packed as bits, in bit order


def _struct(compact: bool, v3: bool) -> str:
    if v3:
        return "fsim_obs38" if compact else "fsim_obs3"
    return "fsim_obs8" if compact else "fsim_obs"


def _obs_dtype(compact: bool = False, v3: bool = False) -> np.dtype:
    struct = _struct(compact, v3)
    grid = (
        {"flags": ("u1", (lib.RL_FLAG_BYTES,)), "amount": ("u1", (GRID, GRID))}
        if compact
        else {"grid": ("<f4", (6, GRID, GRID))}
    )
    rows, width = (lib.RL3_MAX_ENTITIES, lib.RL3_ENTITY_FEATURES) if v3 else (32, 16)
    fields = {
        **grid,
        "entities": ("<f4", (rows, width)),
        "entity_mask": ("i1", (rows,)),
        "self_": ("<f4", (lib.RL3_SELF_FEATURES if v3 else 12,)),
        "inventory": ("<f4", (lib.RL3_ITEMS if v3 else 14,)),
        "goal": ("<f4", (lib.RL3_GOAL_FEATURES if v3 else 12,)),
    }
    return np.dtype(
        {
            "names": list(fields),
            "formats": list(fields.values()),
            "offsets": [ffi.offsetof(struct, name) for name in fields],
            "itemsize": ffi.sizeof(struct),
        }
    )


def obs_layout(
    compact: bool = False, v3: bool = False
) -> dict[str, tuple[int, str, tuple[int, ...]]]:
    """key -> (byte offset in `fsim_obs`/`fsim_obs8`, or the v3 structs, numpy
    dtype, shape): for slicing a copied block of observations without going
    through numpy."""
    dtype = _obs_dtype(compact, v3)
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
        tick_limit: int | None = None,
        eval_seeds: bool = False,
        obs_memory=None,
        demo_starts: float = 0.0,
        compact: bool = False,
        mask_memory=None,
        action_space: str = "v1",
        group: int = 1,
        autoreset: bool = True,
        level_source=None,
        demo_obstructed: bool = False,
        opmask_memory=None,
    ) -> None:
        """`obs_memory`, if given, is `(address, owner)`: `n * sizeof(fsim_obs)`
        bytes the observations are written into instead of a fresh block -- a
        trainer passes page-locked memory so the batch reaches the GPU in one
        fast copy. `owner` is kept alive with the environment.

        `demo_starts` is the fraction of episodes that begin partway along the
        scripted build (`fsim.expert`), at a stage drawn uniformly; the rest
        begin at the scene's own start.

        `mask_memory` is the same for the `n * 201` mask bytes (`n * 376`
        under v3), and `opmask_memory` for v3's `n * 25 * 351` per-operation
        mask bytes.

        `group` makes consecutive environments share a scene: `i` and `j` draw
        the same seed when `i // group == j // group`, so a group is `group`
        attempts at one episode. GRPO's baseline is the group's own mean return,
        which only means anything if the group faces the same scene.

        `autoreset=False` leaves a finished environment finished: `step` reports
        it once and then reports nothing more from it, and its rewards read
        zero. A group-relative learner needs one whole episode per environment
        per rollout, not a stream cut at the horizon. The environment is still
        reset underneath (to its own scene again, not a new one) so its action
        mask stays legal -- everything it produces afterwards is discarded.

        `level_source`, if given, is called as `level_source(i, seed)` when
        environment `i` resets and returns `(name, blueprint)` in place of
        drawing one from `scenes.sample`. That is how unsupervised environment
        design feeds the trainer: the curriculum decides which scene a slot
        runs, rather than a seed deciding it (`fsim/ued.py`). `families[i]`
        then reports whatever name the source gave, so metrics can tell a
        replayed level from a freshly generated one.

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
        #: The task's construction tick limit. plate_line's line runs for
        #: about 7,200 ticks before it has made thirty plates, so a task whose
        #: budget is not the default needs to say so.
        self.tick_limit = tick_limit
        self.seed_base = (EVAL_SEED_BASE if eval_seeds else 0) + seed * 1_000_003
        self.episodes_started = 0
        # The builder puts the same line down for either task -- it solves
        # every build_line training scene it is given -- so both may use it.
        self.demo_starts = demo_starts
        #: Demonstrate on scenes that contain walls too. **Off by default, and
        #: the reason is a measurement rather than caution.**
        #:
        #: The builder walks in straight lines, so walled scenes were refused a
        #: demonstration outright. That looked like a bug: the builder in fact
        #: completes the line on 92% of `cluttered_patch` and 91% of the walled
        #: levels UED generates, and refusing cost the hand-written split a
        #: quarter of its demonstrations.
        #:
        #: Turning it on does exactly what it should to the curriculum and the
        #: wrong thing to the result. Over three seeds at the tuned config,
        #: held-out success went 0.836 / 0.486 / 0.383 against 0.920 / 0.856 /
        #: 0.877 / 0.820 for four seeds without it -- thirty points worse, with
        #: two seeds below every run that preceded them. Meanwhile fix-s1
        #: reached backplay rung 4 by 20M steps, solving thirteen-decision cuts
        #: at 67%, where ar-s1 was on rung 2 solving six-decision cuts at 33%.
        #:
        #: So climbing the ladder is not generalising. Every demonstration ends
        #: in the same two-machine arrangement, and more of them is more of the
        #: same: the policy gets better at finishing the expert's build and no
        #: better at starting an unseen one. Kept as an option because the
        #: mechanism is sound and a more varied builder would change the
        #: answer -- see `docs/shaping.md`.
        self.demo_obstructed = demo_obstructed
        #: Backplay's window, in decisions back from the end of the build: an
        #: episode with a demonstration start runs all but `U[lo, hi]` of it.
        #: `None` draws a stage uniformly instead (Backplay's "Uniform").
        self.demo_window: tuple[int, int] | None = None
        #: Draw the builder's layout per episode rather than always building
        #: the one canonical arrangement. False reproduces the single-pose
        #: demonstrations, which a policy memorises (`docs/shaping.md`).
        self.demo_layouts: bool = True
        #: How many of `expert.FURNACE_OFFSETS` a demonstration may use. 1 is
        #: the rotation orbit of the canonical pose, which is what every run
        #: recorded before this option existed used, so leaving it alone keeps
        #: those runs comparable. 2 adds the reflected orbit, which no turn or
        #: translation of a demonstration can reach -- the one axis along which
        #: the builder can be more varied rather than merely re-posed.
        self.demo_variants: int = 1
        #: belt_smelting: recorded builds (`fsim.demos.collect(records=...)`,
        #: `tools/behaviour_clone.py --pool`) to cut demonstration starts from,
        #: as `{"seed", "split", "family", "vectors"}`. The builder re-plans in
        #: Python after every decision; a recorded build replays in C, and the
        #: simulator's determinism makes the replay the build. None runs the
        #: builder live on the slot's own scene.
        self.demo_pool: list[dict] | None = None
        if action_space not in ("v1", "v2", "v3"):
            raise ValueError(f"the vectorised env speaks v1, v2 and v3, not {action_space!r}")
        self.action_space = action_space
        self.v3 = action_space == "v3"
        #: Where a resetting environment gets its scene. None draws from the
        #: task's own families, which is every run that is not a UED run.
        self.level_source = level_source
        if n % group:
            raise ValueError(f"{n} environments do not divide into groups of {group}")
        self.group = group
        self.autoreset = autoreset
        self._group_seeds = [0] * (n // group)
        self._seed_of = [0] * n
        #: The scene each slot is running, so parking can reuse it.
        self._scene_of: list[tuple[str, dict] | None] = [None] * n
        #: False once an episode ends under `autoreset=False`, until `reset`.
        self.alive = np.ones(n, dtype=bool)
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
        struct = _struct(compact, self.v3)
        if obs_memory is None:
            self._obs_c = ffi.new(f"{struct}[]", n)
            self._obs_owner = None
        else:
            address, self._obs_owner = obs_memory
            self._obs_c = ffi.cast(f"{struct} *", address)
        if self.v3:
            self._encode = lib.fsim_rl_encode38 if compact else lib.fsim_rl_encode3
        else:
            self._encode = lib.fsim_rl_encode8 if compact else lib.fsim_rl_encode
        #: Bytes of one environment's flat mask, and of its per-operation masks.
        self.mask_size = lib.RL3_MASK_SIZE if self.v3 else lib.RL_MASK_SIZE
        self.opmask_size = lib.RL3_OPERATIONS * lib.RL3_ARG_WIDTH if self.v3 else 0
        if mask_memory is None:
            self._masks_c = ffi.new("uint8_t[]", n * self.mask_size)
            self._mask_owner = None
        else:
            address, self._mask_owner = mask_memory
            self._masks_c = ffi.cast("uint8_t *", address)
        self._opmask_owner = None
        if not self.v3:
            self._opmasks_c = ffi.NULL
        elif opmask_memory is None:
            self._opmasks_c = ffi.new("uint8_t[]", n * self.opmask_size)
        else:
            address, self._opmask_owner = opmask_memory
            self._opmasks_c = ffi.cast("uint8_t *", address)
        self._actions_c = ffi.new("int32_t[]", n * 6)
        self._rewards_c = ffi.new("double[]", n)
        self._flags_c = ffi.new("uint8_t[]", n * 4)
        self._verified_c = ffi.new("double[]", n)

        self.obs_nbytes = n * ffi.sizeof(struct)
        block = np.frombuffer(
            ffi.buffer(self._obs_c, self.obs_nbytes), _obs_dtype(compact, self.v3)
        )
        keys = PACKED_KEYS if compact else OBS_KEYS
        self.obs = {key: block["self_" if key == "self" else key] for key in keys}
        mask_bytes = n * self.mask_size
        self.masks = np.frombuffer(ffi.buffer(self._masks_c, mask_bytes), np.uint8).reshape(n, -1)
        #: v3 only: `(n, RL3_OPERATIONS, RL3_ARG_WIDTH)`, row o operation o's
        #: legal argument values (`RlEnv.op_masks()`).
        self.op_masks = None
        if self.v3:
            self.op_masks = np.frombuffer(
                ffi.buffer(self._opmasks_c, n * self.opmask_size), np.uint8
            ).reshape(n, lib.RL3_OPERATIONS, lib.RL3_ARG_WIDTH)
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

    def _seed_for(self, i: int) -> int:
        """The scene seed for environment `i`, drawn once per group.

        `reset` walks the environments in order, so the group's first member
        draws and the rest read what it drew. With `group == 1` every member is
        a first member and this is the plain per-episode counter.
        """
        g = i // self.group
        if i % self.group == 0:
            self._group_seeds[g] = self.seed_base + self.episodes_started
            self.episodes_started += 1
        return self._group_seeds[g]

    def _reset_one(self, i: int, seed: int | None = None, reuse: bool = False) -> None:
        if seed is None:
            seed = self._seed_for(i)
        self._seed_of[i] = seed
        if reuse and self._scene_of[i] is not None:
            # Parking a finished environment must not ask the curriculum for a
            # new level: the slot's score has not been attributed yet, and
            # replacing the level first would credit it to the wrong one.
            family, scene = self._scene_of[i]
        elif self.level_source is None:
            family, scene = scenes.sample(self.task, self.split, seed, self.start_curriculum)
        else:
            family, scene = self.level_source(i, seed)
        self._scene_of[i] = (family, scene)
        c_scene, keep = scene_struct(scene)
        budget = self.max_steps
        if not isinstance(budget, int):
            # A horizon curriculum: each episode's decision budget is drawn from
            # [lo, hi], so a line built early is verified sooner. Seeded by the
            # episode, like the scene.
            lo, hi = budget
            budget = lo + (seed * 2654435761 + 97) % (hi - lo + 1)
        task = task_struct(
            self.task, scene, max_steps=budget, shaping=self.shaping, gamma=self.gamma,
            action_space=self.action_space, construction_tick_limit=self.tick_limit,
        )  # fmt: skip
        self._keep[i] = (c_scene, keep, task)
        rl = self.rls[i]
        lib.fsim_rl_reset(rl, task, c_scene)
        start, taken = "scene", 0
        draw = random.Random(seed * 7 + 3)
        # The builder walks in straight lines, so it can only demonstrate a
        # scene with nothing in the way; an obstructed one starts from scratch.
        obstructed = bool(scene["entities"])
        eligible = self.demo_obstructed or not obstructed
        if self.task == "belt_smelting":
            # Its own builder, which plans round walls: every scene has one.
            eligible = False
            if self.demo_starts and draw.random() < self.demo_starts:
                if self.demo_pool:
                    start, taken, family = self._pooled_demo(i, rl, budget, draw)
                else:
                    start, taken = self._belt_demo(rl, task, c_scene, scene, draw)
        if self.demo_starts and eligible and draw.random() < self.demo_starts:
            patch = scene["markers"]["patch"]
            # No buildable arrangement at all: nothing to demonstrate.
            if expert.layouts(rl, patch, self.demo_variants):
                layout = expert.choose_layout(
                    rl, patch, draw if self.demo_layouts else None, self.demo_variants
                )
                if self.demo_window is None:
                    stage = draw.choice(expert.STAGES)
                    taken = expert.advance_to(rl, patch, stage, self._demo_step(rl), layout)
                    start = stage if taken else "scene"
                else:
                    length = expert.plan_length(rl, patch, layout)
                    lo, hi = self.demo_window
                    back = draw.randint(min(lo, length), min(hi, length))
                    wanted = length - back
                    taken = expert.advance_decisions(rl, patch, wanted, self._demo_step(rl), layout)
                    if taken < wanted:
                        # The builder walks in straight lines and something was
                        # in the way. A half-finished demonstration is worse
                        # than none -- it starts the policy from a state the
                        # expert never reaches -- so put the scene back.
                        lib.fsim_rl_reset(rl, task, c_scene)
                        start, taken = "scene", 0
                    else:
                        start = "scene" if back >= length else f"back{back}"
        self._observe_one(i)
        self.families[i] = family
        self.starts[i] = start
        self.returns[i] = 0.0
        self.lengths[i] = taken
        self.peak_potential[i] = lib.fsim_rl_potential(self.rls[i])

    def _cut(self, length: int, draw) -> int:
        """How many decisions back from the end of a `length`-decision build
        a demonstration start begins: `demo_window`, in decisions or (floats)
        fractions of the build, else uniform over the whole build."""
        if self.demo_window is None:
            return draw.randint(0, length)
        lo, hi = self.demo_window
        if isinstance(lo, float) or isinstance(hi, float):
            lo, hi = round(lo * length), round(hi * length)
        return draw.randint(min(lo, length), min(hi, length))

    def _pooled_demo(self, i: int, rl, budget: int, draw) -> tuple[str, int, str]:
        """A demonstration start from `demo_pool` -> (start, decisions, family).

        The slot runs the recorded build's own scene, not the one its seed
        drew: a recorded build only replays on the scene it was recorded on."""
        entry = self.demo_pool[draw.randrange(len(self.demo_pool))]
        family, scene = scenes.sample(self.task, entry["split"], entry["seed"])
        c_scene, keep = scene_struct(scene)
        task = task_struct(
            self.task, scene, max_steps=budget, shaping=self.shaping, gamma=self.gamma,
            action_space=self.action_space, construction_tick_limit=self.tick_limit,
        )  # fmt: skip
        self._keep[i] = (c_scene, keep, task)
        self._scene_of[i] = (family, scene)
        lib.fsim_rl_reset(rl, task, c_scene)
        vectors = [v for v in entry["vectors"] if int(v[0]) != OP_FINISH]
        length = len(vectors)
        back = self._cut(length, draw)
        step = self._demo_step(rl)
        for vector in vectors[: length - back]:
            if step(vector):
                lib.fsim_rl_reset(rl, task, c_scene)
                return "scene", 0, family
        return ("scene" if back >= length else f"back{back}"), length - back, family

    def _belt_demo(self, rl, task, c_scene, scene, draw) -> tuple[str, int]:
        """A belt_smelting demonstration start -> (start name, decisions taken).

        The builder (`fsim/belt_expert.py`) re-plans after every step, so the
        length of its build is only known once it has run: it runs to the
        decision before its `finish`, and the environment is reset and
        replayed through the first `length - back` of its decisions. The
        simulator is deterministic, so the replay reaches the state the build
        did, and the Python builder runs once. `demo_window` is `(lo, hi)`
        decisions back from the end, or fractions of the build's length when
        given as floats (belt_smelting's ladder, `train.py`)."""
        from fsim import belt_expert

        builder = belt_expert.BeltBuilder(rl, scene, rng=draw if self.demo_layouts else None)
        step = self._demo_step(rl)
        vectors: list[tuple[int, ...]] = []
        finished = False
        while len(vectors) < BELT_DEMO_LIMIT:
            vector = builder.next_vector()
            if vector is None or int(vector[0]) == OP_FINISH:
                finished = vector is not None
                break
            vectors.append(tuple(int(v) for v in vector))
            if step(vector):
                break
        lib.fsim_rl_reset(rl, task, c_scene)
        if not finished:
            return "scene", 0  # no finished build to cut
        length = len(vectors)
        back = self._cut(length, draw)
        wanted = length - back
        for vector in vectors[:wanted]:
            if step(vector):
                # The build ended the episode on the way: not a start to use.
                lib.fsim_rl_reset(rl, task, c_scene)
                return "scene", 0
        return ("scene" if back >= length else f"back{back}"), wanted

    def _observe_one(self, i: int) -> None:
        """Environment `i`'s observation and masks, as a step writes them."""
        rl = self.rls[i]
        self._encode(rl, ffi.addressof(self._obs_c, i))
        if self.v3:
            lib.fsim_rl_masks3(
                rl,
                ffi.addressof(self._masks_c, i * self.mask_size),
                ffi.addressof(self._opmasks_c, i * self.opmask_size),
            )
        else:
            lib.fsim_rl_mask(rl, ffi.addressof(self._masks_c, i * self.mask_size))

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
        self.alive[:] = True
        return self.obs, self.masks

    def _run(self, bounds) -> None:
        if self.v3:
            full, packed = (ffi.NULL, self._obs_c) if self.compact else (self._obs_c, ffi.NULL)
            lib.fsim_rl_step_range3(
                self.rls, bounds[0], bounds[1], self._actions_c, full, packed, self._masks_c,
                self._opmasks_c, self._rewards_c, self._flags_c, self._verified_c,
                self._potentials_c,
            )  # fmt: skip
            return
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
        if not self.autoreset:
            # An environment that has already finished keeps being stepped --
            # the C side walks a contiguous range -- but nothing it produces
            # counts. Zero its reward and never report it a second time.
            terminated &= self.alive
            truncated &= self.alive
            rewards[~self.alive] = 0.0
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
            if self.autoreset:
                self._reset_one(i)
            else:
                # Park it on its own scene again rather than drawing the next
                # one: a fresh scene would consume a group's seed out of turn,
                # and the environment only needs a legal mask from here on.
                self.alive[i] = False
                self._reset_one(i, seed=self._seed_of[i], reuse=True)
        return self.obs, self.masks, rewards, terminated, truncated, episodes
