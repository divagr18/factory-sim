"""Walking into obstacles, tick by tick, against the engine's corner-slide rigs.

`FactorioRL/tools/probe_corner_slide.py` walked a character at walls, a wall
column, furnaces and pairs of them from a grid of offsets, recording its
position every tick (`sim-mechanics-m5-slide*.json`). Each run is rebuilt here
and walked with `fsim_walk_ticks`.
"""

from __future__ import annotations

import json

import pytest

from fsim import Sim, ffi, lib
from fsim.parity import GOLDEN

KINDS = {"stone-wall": lib.K_WALL, "stone-furnace": lib.K_FURNACE}
DIRS = {"north": 0, "east": 4, "south": 8, "west": 12}
FILES = (
    "sim-mechanics-m5-slide.json",
    "sim-mechanics-m5-slide-gaps.json",
    "sim-mechanics-m5-slide-creep.json",
)


def fixed(v: float) -> int:
    return round(v * 256)


def cases():
    for name in FILES:
        path = GOLDEN / name
        if not path.exists():
            continue
        for rig_name, rig in json.loads(path.read_text(encoding="utf-8"))["rigs"].items():
            for direction, runs in rig["runs"].items():
                yield pytest.param(rig, direction, runs, id=f"{rig_name}-{direction}")


def walk(sim, rig, direction, run):
    env = sim.env
    env.entity_count = 0
    for proto, x, y in rig["placed"]:
        e = env.entities[env.entity_count]
        ffi.memmove(ffi.addressof(e), b"\0" * ffi.sizeof("fsim_entity"), ffi.sizeof("fsim_entity"))
        e.alive, e.kind, e.neutral = 1, KINDS[proto], 1
        e.pos.x, e.pos.y = fixed(x), fixed(y)
        env.entity_count += 1
    lib.fsim_after_load(env)
    env.char_pos.x, env.char_pos.y = fixed(run["start"][1]), fixed(run["start"][2])
    ticks = len(run["path"]) - 1
    out = ffi.new("int32_t[]", 2 * ticks)
    lib.fsim_walk_ticks(env, DIRS[direction], ticks, out)
    return [(out[2 * i], out[2 * i + 1]) for i in range(ticks)]


@pytest.mark.parametrize(("rig", "direction", "runs"), list(cases()))
def test_walking_into_obstacles_matches_the_engine(rig, direction, runs):
    sim = Sim()
    sim.reset({"entities": [], "resources": [], "character": {"position": [0, 0],
               "inventory": {}}, "markers": {}, "radius": 48})  # fmt: skip
    wrong = []
    for run in runs:
        got = walk(sim, rig, direction, run)
        want = [(fixed(p[0]), fixed(p[1])) for p in run["path"][1:]]
        for tick, (g, w) in enumerate(zip(got, want, strict=True), start=1):
            if g != w:
                wrong.append((run["offset_256"], tick, g, w))
                break
    assert not wrong, f"{len(wrong)} of {len(runs)} runs diverge; first: {wrong[:3]}"
