"""v3's `take_fuel` and `finish`, and the inventory facts the per-operation
masks rest on (user decisions, 2026-09-25; FactorioRL docs/sim-logistics.md,
"v3 masks per operation, `finish` and taking fuel")."""

from __future__ import annotations

import json
import lzma

import pytest

from fsim import ITEM_IDS, Sim, lib
from fsim.parity import GOLDEN
from fsim.program_api import SimBackend, WorldV3, play
from fsim.rl import RlEnv
from fsim.trace import read_trace

OP_TAKE_FUEL, OP_FINISH = 23, 24


def _header(name: str) -> dict:
    return read_trace(GOLDEN / f"{name}.jsonl.xz")[0]


def _env(name: str, **overrides) -> RlEnv:
    header = _header(name)
    env = RlEnv()
    env.reset(
        header["task"],
        {**header["blueprint"], **overrides},
        decision_ticks=header["decision_ticks"],
        max_steps=header["max_decision_steps"],
        construction_tick_limit=header["construction_tick_limit"],
        action_space="v3",
    )
    return env


def test_stack_sizes_and_slots_are_the_engines():
    """FactorioRL tools/probe_inventory.py read them off the running engine."""
    path = GOLDEN / "inventory-prototypes.json.xz"
    if not path.is_file():
        pytest.skip("no inventory-prototypes evidence")
    data = json.loads(lzma.decompress(path.read_bytes()))
    assert lib.FSIM_MAIN_SLOTS == data["character"]["main_slots"]
    checked = 0
    for name, record in data["items"].items():
        if name in ITEM_IDS:
            assert lib.fsim_stack_size(ITEM_IDS[name]) == record["stack_size"], name
            checked += 1
    assert checked == 14


@pytest.mark.parametrize("ore", ["iron-ore", "copper-ore", "stone"])
def test_a_furnace_source_takes_54_of_any_ore(ore):
    header = _header("v3_take_fuel")
    blueprint = dict(header["blueprint"])
    furnace = next(e for e in blueprint["entities"] if e["name"] == "stone-furnace")
    blueprint["entities"] = [{**furnace, "contents": {}}]
    blueprint["character"] = {**blueprint["character"], "inventory": {ore: 60}}
    sim = Sim()
    sim.reset(blueprint)
    handle = next(e["h"] for e in sim.observation()["entities"] if e["name"] == "stone-furnace")
    outcomes = []
    for _ in range(3):
        sim.step("give_to", {"to": handle, "item": ore, "count": 20})
        outcomes.append(sim.action_outcome())
    # The third moves 14 and is refused for the 6 that did not fit.
    assert outcomes == [("completed", None), ("completed", None), ("rejected", "no_space")]
    record = next(e for e in sim.observation()["entities"] if e["name"] == "stone-furnace")
    assert record["contents"] == {ore: 54}
    assert sim.inventory() == {ore: 6}


def test_finish_runs_the_verification_and_ends_the_episode():
    env = _env("v3_finish")
    _, reward, terminated, truncated, info = env.step([OP_FINISH, 0, 0, 0, 0, 0])
    assert terminated and not truncated and env.rl.verified
    assert reward == min(env.rl.verified_output / 10, 1.0)
    assert not info["decode_failure"] and env.rl.steps == 1


def test_take_fuel_takes_what_the_fuel_slot_holds():
    env = _env("v3_take_fuel")
    world = WorldV3(SimBackend(env))
    drill = next(e for e in world.entities() if e.kind == "mining-drill")
    before = world.inventory()["coal"]
    assert world.take_fuel(drill, 1)
    assert world.inventory()["coal"] == before + 1
    # A chest has no fuel slot: refused, no decision spent.
    chest = next(e for e in world.entities() if e.kind == "container")
    decisions = world.decisions
    assert not world.take_fuel(chest, 1) and world.decisions == decisions


def test_a_v3_programs_return_is_its_finish():
    env = _env("v3_finish")
    result = play(lambda world: None, SimBackend(env), world_cls=WorldV3)
    assert result.decisions == 1 and result.trace == ["finish -> ok"]
    assert env.rl.done and env.rl.verified and env.rl.steps == 1
    env = _env("v3_finish")
    result = play(lambda world: world.finish(), SimBackend(env), world_cls=WorldV3)
    assert result.decisions == 1 and env.rl.steps == 1


def test_finish_is_not_a_v1_operation():
    header = _header("placement_footprints")
    env = RlEnv()
    env.reset(header["task"], header["blueprint"])
    _, _, terminated, _, info = env.step([OP_FINISH, 0, 0, 0, 0, 0])
    assert info["decode_failure"] and not terminated
