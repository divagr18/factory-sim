"""The builder-program API: legality, budgets, errors, and the seed program's parity."""

from __future__ import annotations

import math

import pytest

from evolve.seeds.builder import SOURCE
from fsim import expert, lib, scenes
from fsim.program_api import BudgetExhausted, Entity, World, run_episode
from fsim.rl import RlEnv

TASK = "construct_smelting_line"
CLEAR_SEEDS = [s for s in range(40) if not scenes.sample(TASK, "train", s)[1]["entities"]]


def seed_build():
    ns = {"math": math}
    exec(SOURCE, ns)
    return ns["build"]


def fresh(seed=2):
    _, scene = scenes.sample(TASK, "train", seed)
    env = RlEnv()
    env.reset(TASK, scene, action_space="v2")
    return env, World(env)


def test_run_episode_resets_into_v2_even_after_a_v1_episode():
    env = RlEnv()
    _, scene = scenes.sample(TASK, "train", 1)
    env.reset(TASK, scene)  # the task's default is v1
    assert env.rl.task.action_space == lib.ACTION_SPACE_V1
    seen = []
    run_episode(lambda world: seen.append(world._rl.task.action_space), scene, env=env)
    assert seen == [lib.ACTION_SPACE_V2]


def test_illegal_intents_cost_no_decision():
    env, world = fresh()
    x, y = world.tile()
    illegal = [
        lambda: world.place("stone-furnace", x + 6, y, "N"),  # outside the window
        lambda: world.place("stone-furnace", x, y, "N"),  # the character's own tile
        lambda: world.place("iron-plate", x + 2, y, "N"),  # not held
        lambda: world.place("stone-furnace", x + 2.5, y, "N"),  # not a tile
        lambda: world.place("stone-furnace", x + 2, y, "up"),
        lambda: world.place("rocket", x + 2, y, "N"),
        lambda: world.give(31, "coal", 20),  # no such row
        lambda: world.mine(0),  # nothing in the table on an open patch at the start
        lambda: world.move("NE"),
        lambda: world.move("N", "sprint"),
    ]
    for k, intent in enumerate(illegal):
        assert intent() is False, k
        assert world.refusals == k + 1
        assert world.decisions == 0 and env.rl.steps == 0
    assert env.rl.decode_failures == 0
    assert all("refused" in line for line in world._trace)


def test_legal_intents_step_once_and_build():
    env, world = fresh()
    x, y = world.tile()
    assert world.place("stone-furnace", x + 2, y, "N") is True
    assert world.decisions == 1 and env.rl.steps == 1
    assert world._built == [("stone-furnace", x + 2, y)]
    (furnace,) = world.entities()
    assert furnace.kind == "furnace" and (furnace.x, furnace.y) == (x + 3, y + 1)
    assert world.give(furnace, "coal", 7) is False  # not a transfer amount
    assert world.give(furnace, "coal", 5) is True
    assert world.entities()[0].fuel == 5
    assert world.inventory()["coal"] == 55
    assert world.move("W", "step") and world.move("W", "step")
    # An Entity read before the walk still names the same furnace.
    assert world.take(furnace, "coal", 1) is False  # nothing to take: fuel is not a source
    assert world.give(furnace, "coal", 1) is True
    assert world.entities()[0].fuel == 6
    assert env.rl.decode_failures == 0 and world.refusals == 2
    assert world.walk_distance > 1.5


def test_budget_exhausted_fires_at_the_budget():
    caught = []

    def build(world):
        try:
            while True:
                world.wait()
        except BudgetExhausted:
            caught.append(world.decisions_left())
            raise

    _, scene = scenes.sample(TASK, "train", 1)
    result = run_episode(build, scene, decision_budget=7)
    assert caught == [0]
    assert result.decisions == 7 and result.error is None
    assert not result.success


def test_a_raising_program_is_recorded_and_the_episode_still_finishes():
    def build(world):
        world.wait()
        world.wait()
        raise ValueError("boom")

    env = RlEnv()
    _, scene = scenes.sample(TASK, "train", 1)
    result = run_episode(build, scene, env=env)
    assert result.error == "ValueError: boom"
    assert result.decisions == 2
    assert env.rl.done and env.rl.verified and env.rl.steps == 600
    assert not result.success and result.first_plate_tick is None


def test_endless_refusals_are_stopped():
    def build(world):
        while True:
            world.give(31, "coal", 20)

    _, scene = scenes.sample(TASK, "train", 1)
    result = run_episode(build, scene)
    assert result.decisions == 0 and result.refusals == 1000
    assert "refused intents" in result.error


def test_the_seed_program_passes_the_sandbox_language():
    """No import, lambda, try, with, class or print in the seed source."""
    import ast

    tree = ast.parse(SOURCE)
    banned = (ast.Import, ast.ImportFrom, ast.Lambda, ast.Try, ast.With, ast.ClassDef,
              ast.Global, ast.Nonlocal, ast.Raise, ast.Yield)  # fmt: skip
    assert not [n for n in ast.walk(tree) if isinstance(n, banned)]
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "print" not in names and not [n for n in names if n.startswith("_") and n != "_"]
    assert [n.name for n in tree.body] == ["build"]
    assert len(SOURCE.splitlines()) <= 300 and len(SOURCE) <= 20000


@pytest.mark.parametrize("seed", CLEAR_SEEDS)
def test_seed_program_matches_the_expert(seed):
    family, scene = scenes.sample(TASK, "train", seed)
    reference = RlEnv()
    reference.reset(TASK, scene, action_space="v2")
    patch = scene["markers"]["patch"]
    builder = expert.Builder(reference.rl, patch)
    expert.run_to_completion(reference.rl, patch)

    result = run_episode(seed_build(), scene)
    assert result.success == bool(reference.rl.success), family
    assert result.success, family
    assert result.refusals == 0 and result.failures == 0 and result.error is None
    assert result.built == [
        ("burner-mining-drill", *builder.drill_anchor),
        ("stone-furnace", *builder.furnace_anchor),
    ]
    assert result.verified_output == int(reference.rl.verified_output)
    assert result.first_plate_tick is not None and result.walk_distance > 0


def test_there_are_31_clear_training_scenes_below_40():
    assert len(CLEAR_SEEDS) == 31
    assert isinstance(Entity, type)
