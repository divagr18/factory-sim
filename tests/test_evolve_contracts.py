"""The seams between the evolution modules, which were built separately.

Each module passed its own tests and the seed program still failed the sandbox:
it used `_` as a throwaway name, which the underscore ban refused. These tests
pin the contracts that only show up when the pieces meet.
"""

from __future__ import annotations

import inspect

import pytest

from evolve import sandbox
from evolve.seeds.builder import SOURCE
from fsim import scenes
from fsim.program_api import World, run_episode


def _public_methods(cls) -> set[str]:
    return {
        name
        for name, member in inspect.getmembers(cls)
        if not name.startswith("_") and (callable(member) or isinstance(member, property))
    }


def test_the_seed_passes_the_sandbox():
    sandbox.check(SOURCE)


def test_the_sandbox_knows_every_world_method_and_nothing_else():
    """`WORLD_API` is readable off any name, so it must be exactly World's surface
    (with `WorldV3`'s, the v3 profile's World).

    A method World gains but the list lacks only works on the literal name
    `world`; a name the list keeps after World drops it is an attribute any value
    may be asked for."""
    from fsim.program_api import WorldV3

    methods = _public_methods(World) | _public_methods(WorldV3)
    counters = {"refusals", "failures"}  # plain int attributes, set in __init__
    assert methods | counters == sandbox.WORLD_API, (
        f"World has {sorted(methods - sandbox.WORLD_API)} not in WORLD_API; "
        f"WORLD_API has {sorted(sandbox.WORLD_API - methods - counters)} World lacks"
    )


def test_no_public_attribute_of_world_reaches_the_simulator():
    """Any public attribute is allowed on the literal `world`, so none may hold the env."""
    from fsim.rl import RlEnv

    env = RlEnv()
    _, scene = scenes.sample("construct_smelting_line", "train", 0)
    env.reset("construct_smelting_line", scene, action_space="v2")
    world = World(env)
    for name in dir(world):
        if name.startswith("_"):
            continue
        value = getattr(world, name)
        if callable(value):
            continue
        assert isinstance(value, (int, float, bool, str, type(None))), (
            f"world.{name} is a {type(value).__name__}; public data must be plain values"
        )


@pytest.mark.parametrize("seed", range(40))
def test_the_seed_loaded_through_the_sandbox_matches_the_expert(seed):
    family, scene = scenes.sample("construct_smelting_line", "train", seed)
    if scene["entities"]:
        pytest.skip(f"{family} has something in the way")
    build = sandbox.load(SOURCE)
    result = run_episode(build, scene)
    assert result.success, (family, result.error, result.trace[-5:])
    assert result.refusals == 0
