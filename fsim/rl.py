"""The RL contract: tensors, masks, goal and reward, from the C layer.

`RlEnv` is one environment speaking FactorioRL's `parameterized-v1` action
space (`MultiDiscrete[22, 33, 122, 5, 15, 4]`) and its `local-v2` tensor
observation, for `construct_smelting_line` and `build_line`. The arrays it
returns are views over C memory, refreshed by `observe()`.
"""

from __future__ import annotations

import numpy as np

from fsim import ffi, lib, scene_struct

NVEC = (22, 33, 122, 5, 15, 4)
TASKS = {
    "construct_smelting_line": lib.TASK_CONSTRUCT_SMELTING_LINE,
    "build_line": lib.TASK_BUILD_LINE,
}
COMPONENTS = {
    "construct_smelting_line": ("verified_output",),
    "build_line": ("constructed", "plates_produced", "step_cost"),
}


def task_struct(task: str, blueprint: dict, *, decision_ticks=30, max_steps=600,
                construction_tick_limit=None):  # fmt: skip
    t = ffi.new("fsim_task *")
    t.task = TASKS[task]
    t.decision_ticks = decision_ticks
    t.max_steps = max_steps
    if construction_tick_limit is None:
        construction_tick_limit = 18000
    t.construction_tick_limit = construction_tick_limit
    patch = (blueprint.get("markers") or {}).get("patch")
    if patch is not None and "patch" in (blueprint.get("public_markers") or []):
        t.has_patch = 1
        t.patch_x, t.patch_y = float(patch[0]), float(patch[1])
    return t


class RlEnv:
    def __init__(self, water=None) -> None:
        from fsim import Sim

        self.rl = lib.fsim_rl_new()
        self._water = Sim(water)  # loads the map's water once
        lib.fsim_set_water(self.rl.env, self._water_flat(), self._water.env.water_count)
        self.obs_c = ffi.new("fsim_obs *")
        self.mask_c = ffi.new("uint8_t[]", lib.RL_MASK_SIZE)
        self.vector_c = ffi.new("int32_t[6]")
        self.task_name = None
        self._keep = None
        self.obs = {
            "grid": np.frombuffer(ffi.buffer(self.obs_c.grid), np.float32).reshape(6, 65, 65),
            "entities": np.frombuffer(ffi.buffer(self.obs_c.entities), np.float32).reshape(32, 16),
            "entity_mask": np.frombuffer(ffi.buffer(self.obs_c.entity_mask), np.int8),
            "self": np.frombuffer(ffi.buffer(self.obs_c.self_), np.float32),
            "inventory": np.frombuffer(ffi.buffer(self.obs_c.inventory), np.float32),
            "goal": np.frombuffer(ffi.buffer(self.obs_c.goal), np.float32),
        }
        self.mask = np.frombuffer(ffi.buffer(self.mask_c, lib.RL_MASK_SIZE), np.uint8)

    def _water_flat(self):
        env = self._water.env
        self._water_buf = ffi.new("int32_t[]", max(1, 2 * env.water_count))
        ffi.memmove(self._water_buf, env.water, 4 * 2 * env.water_count)
        return self._water_buf

    def __del__(self) -> None:
        rl = getattr(self, "rl", None)
        if rl is not None:
            lib.fsim_rl_free(rl)
            self.rl = None

    def reset(self, task: str, blueprint: dict, **task_options) -> dict:
        self.task_name = task
        scene, keep = scene_struct(blueprint)
        t = task_struct(task, blueprint, **task_options)
        self._keep = (scene, keep, t)
        lib.fsim_rl_reset(self.rl, t, scene)
        return self.observe()

    def observe(self) -> dict:
        lib.fsim_rl_encode(self.rl, self.obs_c)
        lib.fsim_rl_mask(self.rl, self.mask_c)
        return self.obs

    def step(self, vector) -> tuple[dict, float, bool, bool, dict]:
        for i in range(6):
            self.vector_c[i] = int(vector[i])
        reward = lib.fsim_rl_step(self.rl, self.vector_c)
        obs = self.observe()
        names = COMPONENTS[self.task_name]
        info = {
            "success": bool(self.rl.success),
            "decode_failure": bool(self.rl.decode_failure),
            "reward_components": {n: self.rl.components[i] for i, n in enumerate(names)},
        }
        if self.rl.verified:
            info["verified_output"] = self.rl.verified_output
        return obs, reward, bool(self.rl.terminated), bool(self.rl.truncated), info
