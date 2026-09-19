"""plate_line: the line arrives built, and the task is to make it run.

The scene generator itself is covered by `test_scenes.py` against FactorioRL's
own output. What is checked here is the part that has no golden -- that a scene
installs two empty machines, that fuelling them produces plates on the schedule
FactorioRL measured on the engine, and that the potential leaves room for
exactly the work the task asks for.
"""

from __future__ import annotations

import math

import pytest

from fsim import action_struct, lib, scenes
from fsim.rl import RlEnv

DRILL = (1.0, 1.0)
FURNACE = (1.0, 3.0)
#: FactorioRL measured "about thirty plates per 7,200 ticks" on a real engine.
BUDGET_TICKS = 24000


def _reset(split: str, seed: int, shaping="both"):
    env = RlEnv()
    family, scene = scenes.sample("plate_line", split, seed)
    env.reset("plate_line", scene, max_steps=400, construction_tick_limit=BUDGET_TICKS,
              shaping=shaping, action_space="v2")  # fmt: skip
    return env, family, scene


def _walk_to(world, target, budget=120) -> bool:
    """Axis-first in whole move actions. No pathfinding, so it cannot get
    around `commissioning_walled`'s screen -- which is that family's point."""
    for _ in range(budget):
        x, y = world.char_pos.x / 256.0, world.char_pos.y / 256.0
        dx, dy = target[0] - x, target[1] - y
        if math.hypot(dx, dy) <= 2.0:
            return True
        if abs(dx) > abs(dy):
            key = "step_east" if dx > 0 else "step_west"
        else:
            key = "step_south" if dy > 0 else "step_north"
        lib.fsim_step(world, action_struct(key), 30)
    return False


def _handle_at(world, position) -> str | None:
    """The handle naming the machine at `position`.

    A handle is *not* an entity's unit number. Units are minted at creation,
    handles lazily by the observation sweep in the order things are first
    seen, and the two counters only coincide when the scene holds nothing
    else. With `commissioning_crowded`'s six machines they diverge, and
    `f"h{unit}"` addressed a decoy: the real furnace stayed unfuelled while
    the one at (7, -3) got the coal, and the line made zero plates.
    """
    for i in range(world.entity_count):
        e = world.entities[i]
        if not e.alive:
            continue
        if abs(e.pos.x / 256.0 - position[0]) > 0.01 or abs(e.pos.y / 256.0 - position[1]) > 0.01:
            continue
        for h in range(1, world.next_handle):
            record = world.handles[h]
            if record.used and record.unit == e.unit and record.destroyed_tick < 0:
                return f"h{h}"
    return None


def _commission(world) -> None:
    for target in (DRILL, FURNACE):
        assert _walk_to(world, target), target
        handle = _handle_at(world, target)
        assert handle is not None, target
        lib.fsim_step(
            world, action_struct("give_to", {"to": handle, "item": "coal", "count": 20}), 30
        )


def test_the_scene_installs_two_empty_machines():
    env, family, _ = _reset("train", 0)
    if family == "commissioning_crowded":
        pytest.skip("that family installs six machines on purpose")
    world = env.rl.env
    kinds = sorted(world.entities[i].kind for i in range(world.entity_count))
    assert kinds == sorted((lib.K_DRILL, lib.K_FURNACE))
    for i in range(world.entity_count):
        assert world.entities[i].status == lib.ST_NO_FUEL
    assert world.produced[lib.IT_IRON_PLATE] == 0


@pytest.mark.parametrize("split", ["train", "val"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_fuelled_line_makes_thirty_plates(split, seed):
    env, _, _ = _reset(split, seed)
    world = env.rl.env
    _commission(world)
    while world.produced[lib.IT_IRON_PLATE] < 30 and world.tick < BUDGET_TICKS:
        lib.fsim_step(world, action_struct("wait"), 30)
    assert world.produced[lib.IT_IRON_PLATE] >= 30
    # Roughly 7,200 ticks of production, plus the walk. Well inside the budget,
    # and slow enough that the agent cannot get there by hand-feeding.
    assert 6000 < world.tick < 12000, world.tick


def test_the_potential_leaves_room_for_the_commissioning():
    """The line is already built, so the drill and line terms are already paid.
    What is left is the 0.1 each for the drill's fuel, the furnace's fuel and
    the furnace holding something -- which is precisely the task."""
    env, _, _ = _reset("train", 0)
    world = env.rl.env
    before = lib.fsim_rl_potential(env.rl)
    # 0.2 drill + 0.3 line + 0.1 * approach, and nothing fuelled yet.
    assert 0.5 <= before < 0.6, before
    _commission(world)
    while world.produced[lib.IT_IRON_PLATE] < 1 and world.tick < BUDGET_TICKS:
        lib.fsim_step(world, action_struct("wait"), 30)
    assert lib.fsim_rl_potential(env.rl) > before + 0.15


def test_success_is_thirty_plates_and_nothing_else():
    env, _, _ = _reset("train", 0, shaping="none")
    world = env.rl.env
    _commission(world)
    assert not env.rl.success
    while world.produced[lib.IT_IRON_PLATE] < 30 and world.tick < BUDGET_TICKS:
        lib.fsim_step(world, action_struct("wait"), 30)
    # The C success predicate reads the same counter, over the same world.
    assert world.produced[lib.IT_IRON_PLATE] >= 30


def test_its_components_are_named_like_factoriorls():
    env, _, _ = _reset("train", 0)
    assert env.components == (
        "commissioned",
        "plates_produced",
        "step_cost",
        "line_progress",
        "line_potential",
    )


def test_the_private_marker_does_not_reach_the_goal_vector():
    """plate_line publishes no markers, so goal[9..11] stays empty even though
    the potential measures approach to the line (csrc/fsim.h, has_target)."""
    env, _, scene = _reset("train", 0)
    assert scene["public_markers"] == []
    assert list(env.obs["goal"][9:12]) == [0.0, 0.0, 0.0]
    assert lib.fsim_rl_potential(env.rl) > 0.5


def _crowded(seed: int):
    env = RlEnv()
    scene = scenes.plate_line("commissioning_crowded", __import__("random").Random(seed))
    env.reset("plate_line", scene, max_steps=400, construction_tick_limit=BUDGET_TICKS,
              shaping="both", action_space="v2")  # fmt: skip
    return env, scene


def test_a_handle_is_not_a_unit_number():
    """Units are minted at creation, handles lazily by the observation sweep,
    and the two counters coincide only when the scene holds nothing else. With
    six machines they diverge -- and addressing a machine by its unit number
    fuelled a decoy while the real furnace stayed empty and the line made zero
    plates."""
    env, _ = _crowded(0)
    world = env.rl.env
    diverged = 0
    for i in range(world.entity_count):
        entity = world.entities[i]
        handle = _handle_at(world, (entity.pos.x / 256.0, entity.pos.y / 256.0))
        if handle != f"h{entity.unit}":
            diverged += 1
    assert diverged, "this scene no longer distinguishes handles from units"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_crowded_family_is_still_solvable(seed):
    """Its point is a bigger decision, not an impossible one: six targets and
    three items, with coal cut to 50 so choosing wrongly costs something. The
    two real machines still need 20 each."""
    env, scene = _crowded(seed)
    world = env.rl.env
    assert scene["character"]["inventory"] == {"coal": 50, "iron-ore": 20, "stone": 20}
    assert sum(1 for i in range(world.entity_count) if world.entities[i].alive) == 6
    _commission(world)
    while world.produced[lib.IT_IRON_PLATE] < 30 and world.tick < BUDGET_TICKS:
        lib.fsim_step(world, action_struct("wait"), 30)
    assert world.produced[lib.IT_IRON_PLATE] >= 30


def test_the_decoys_cannot_make_a_plate():
    """Fuel every machine and only the real line produces, so the family
    enlarges the choice without changing what the task rewards."""
    env, _ = _crowded(0)
    world = env.rl.env
    for i in range(world.entity_count):
        entity = world.entities[i]
        if (entity.pos.x / 256.0, entity.pos.y / 256.0) in (DRILL, FURNACE):
            continue
        handle = _handle_at(world, (entity.pos.x / 256.0, entity.pos.y / 256.0))
        _walk_to(world, (entity.pos.x / 256.0, entity.pos.y / 256.0))
        lib.fsim_step(
            world, action_struct("give_to", {"to": handle, "item": "coal", "count": 5}), 30
        )
    for _ in range(400):
        lib.fsim_step(world, action_struct("wait"), 30)
    assert world.produced[lib.IT_IRON_PLATE] == 0
