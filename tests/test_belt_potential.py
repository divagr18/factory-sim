"""belt_smelting's line potential (`rl_belt_potential`, docs/shaping.md).

A line is put down piece by piece with the scripted-entity calls -- a drill on
iron ore, the furnace under its drop point, an output inserter, belts to the
chest, a chest inserter, fuel, plates -- and each piece must move phi by its
term's weight and nothing else.
"""

from __future__ import annotations

import math

import pytest

from fsim import lib, scenes
from fsim.rl import TASKS, RlEnv

pytestmark = pytest.mark.skipif("belt_smelting" not in TASKS, reason="no belt_smelting")

TILE = 256
NORTH, EAST, SOUTH, WEST = 0, 4, 8, 12
STEP = {NORTH: (0, -1), EAST: (1, 0), SOUTH: (0, 1), WEST: (-1, 0)}


def _scene(seed=0):
    _family, scene = scenes.sample("belt_smelting", "train", seed)
    env = RlEnv()
    env.reset("belt_smelting", scene, action_space="v3", shaping="potential")
    return env, scene


def _phi(env):
    return lib.fsim_rl_potential(env.rl)


def _add(env, kind, tx, ty, direction=0, lattice=False):
    """An entity on tile (tx, ty): a 1x1 at the tile centre, a 2x2 at the
    lattice point (tx, ty), which is its footprint's bottom-right tile's corner."""
    x, y = (tx * TILE, ty * TILE) if lattice else (tx * TILE + 128, ty * TILE + 128)
    index = lib.fsim_add_entity(env.rl.env, kind, x, y, direction)
    assert index >= 0
    return index


def _toward(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    if dx:
        return EAST if dx > 0 else WEST
    return SOUTH if dy > 0 else NORTH


def test_each_piece_of_a_line_moves_phi_by_its_weight():
    env, scene = _scene()
    iron = [r["position"] for r in scene["resources"] if r["name"] == "iron-ore"]
    chest = next(e["position"] for e in scene["entities"] if e.get("marker") == "output")
    ctx, cty = math.floor(chest[0]), math.floor(chest[1])
    # The approach term alone, and no coal mined yet.
    phi0 = _phi(env)
    ix, iy = scene["markers"]["iron"]
    cx, cy = scene["character"]["position"]
    assert phi0 == pytest.approx(0.05 * max(0.0, 1 - math.hypot(cx - ix, cy - iy) / 64))

    # A drill facing south with iron under it: a 2x2 at lattice point
    # (tx + 1, ty + 1) covers tiles tx..tx + 1, ty..ty + 1.
    tx, ty = math.floor(iron[0][0]), math.floor(iron[0][1])
    lx, ly = tx + 1, ty + 1
    drill = _add(env, lib.K_DRILL, lx, ly, SOUTH, lattice=True)
    phi = _phi(env)
    # Half of the drills term, and the approach term now held at its full 0.05.
    assert phi - phi0 == pytest.approx(0.05 + 0.05 * (1 - phi0 / 0.05))
    # Its drop point is tile (lx, ly + 1); a furnace at lattice (lx, ly + 2) holds it.
    furnace = _add(env, lib.K_FURNACE, lx, ly + 2, lattice=True)
    assert _phi(env) - phi == pytest.approx(0.05)
    phi = _phi(env)
    lib.fsim_entity_insert(env.rl.env, drill, lib.IT_COAL, 5)
    lib.fsim_entity_insert(env.rl.env, furnace, lib.IT_COAL, 5)
    assert _phi(env) - phi == pytest.approx(0.025 + 0.025)
    phi = _phi(env)
    lib.fsim_entity_insert(env.rl.env, furnace, lib.IT_IRON_ORE, 3)
    assert _phi(env) - phi == pytest.approx(0.025)
    phi = _phi(env)

    # An output inserter under the furnace, facing north: pickup (0, -1) in the
    # furnace, drop onto the tile below it.
    out = _add(env, lib.K_INSERTER, lx - 1, ly + 3, NORTH)
    assert _phi(env) - phi == pytest.approx(0.05)
    phi = _phi(env)

    # Belts from under the inserter to the tile two short of the chest, along
    # an L: rows first, then columns.
    start = (lx - 1, ly + 4)
    goal = (ctx, cty - 2) if cty > start[1] else (ctx, cty + 2)
    path = [start]
    while path[-1][1] != goal[1]:
        x, y = path[-1]
        path.append((x, y + (1 if goal[1] > y else -1)))
    while path[-1][0] != goal[0]:
        x, y = path[-1]
        path.append((x + (1 if goal[0] > x else -1), y))
    d0 = abs(start[0] - ctx) + abs(start[1] - cty)
    assert d0 > 2
    for k, tile in enumerate(path):
        nxt = path[k + 1] if k + 1 < len(path) else (ctx, cty)
        _add(env, lib.K_BELT, tile[0], tile[1], _toward(tile, nxt))
        if k == 0:
            assert _phi(env) - phi == pytest.approx(0.025)  # onto a belt, no progress yet
            phi = _phi(env)
    # Every belt is reached from the first, and the last is two from the chest.
    assert _phi(env) - phi == pytest.approx(0.15)
    phi = _phi(env)

    # The chest inserter, between the last belt and the chest, facing the belt.
    facing = _toward((ctx, cty), goal)
    step = STEP[facing]
    inserter = _add(env, lib.K_INSERTER, ctx + step[0], cty + step[1], facing)
    assert _phi(env) - phi == pytest.approx(0.04 + 0.06)
    phi = _phi(env)
    lib.fsim_entity_insert(env.rl.env, out, lib.IT_COAL, 1)
    lib.fsim_entity_insert(env.rl.env, inserter, lib.IT_COAL, 1)
    assert _phi(env) - phi == pytest.approx(0.05 * 2 / 3)
    phi = _phi(env)
    chest_index = env.rl.task.output_entity
    assert chest_index >= 0
    lib.fsim_entity_insert(env.rl.env, chest_index, lib.IT_IRON_PLATE, 10)
    assert _phi(env) - phi == pytest.approx(0.05)
    assert 0.0 < _phi(env) <= 1.0


def test_a_belt_pointing_away_reaches_nothing():
    env, scene = _scene()
    iron = [r["position"] for r in scene["resources"] if r["name"] == "iron-ore"]
    tx, ty = math.floor(iron[0][0]), math.floor(iron[0][1])
    lx, ly = tx + 1, ty + 1
    _add(env, lib.K_DRILL, lx, ly, SOUTH, lattice=True)
    _add(env, lib.K_FURNACE, lx, ly + 2, lattice=True)
    _add(env, lib.K_INSERTER, lx - 1, ly + 3, NORTH)
    _add(env, lib.K_BELT, lx - 1, ly + 4, SOUTH)
    before = _phi(env)
    # A second belt beside the first, not in front of it: nothing runs into it.
    _add(env, lib.K_BELT, lx, ly + 4, SOUTH)
    assert _phi(env) == pytest.approx(before)


def test_the_potential_is_zero_at_termination_and_telescopes():
    """gamma = 1: the shaped terms of an episode sum to -phi(s0)."""
    _family, scene = scenes.sample("belt_smelting", "train", 1)
    plain, shaped = RlEnv(), RlEnv()
    options = {"action_space": "v3", "max_steps": 12, "gamma": 1.0}
    plain.reset("belt_smelting", scene, **options)
    shaped.reset("belt_smelting", scene, shaping="potential", **options)
    phi0 = lib.fsim_rl_potential(shaped.rl)
    wait = [21, 0, 0, 0, 0, 0]
    moves = [[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0], wait]
    total = 0.0
    for k in range(12):
        vector = moves[k % len(moves)]
        _, r0, t0, u0, _ = plain.step(vector)
        _, r1, t1, u1, _ = shaped.step(vector)
        total += r1 - r0
        assert (t0, u0) == (t1, u1)
        if t0 or u0:
            break
    assert t1 or u1
    assert shaped.rl.potential == 0.0
    assert total == pytest.approx(-phi0, abs=1e-12)
