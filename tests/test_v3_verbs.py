"""v3's `take_fuel` and `finish`, and the inventory facts the per-operation
masks rest on (user decisions, 2026-09-25; FactorioRL docs/sim-logistics.md,
"v3 masks per operation, `finish` and taking fuel")."""

from __future__ import annotations

import json
import lzma

import pytest

from fsim import ITEM_IDS, Sim, lib
from fsim.obsview import ObsView
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


# ------------------------------------------------------------------ inspect
#
# User decision (2026-09-30): a chest's contents, item by item, are read by
# opening it with `inspect` within reach; it stays open while visible and in
# reach. The `v3_inspect` engine trace pins every tensor (test_rl_contract);
# these read the same episode through `ObsView` and `WorldV3`.


def _opened_by_decision() -> list:
    header, records = read_trace(GOLDEN / "v3_inspect.jsonl.xz")
    env = _env("v3_inspect")
    out = [ObsView(env.obs).opened()]
    for record in records[1:]:
        env.step(record["transition"]["action"]["vector"])
        out.append(ObsView(env.obs).opened())
    return out


def test_an_inspected_chest_shows_every_item_until_out_of_reach():
    seen = _opened_by_decision()
    contents = [None if s is None else s[1] for s in seen]
    assert contents[:2] == [None, None]
    assert contents[2] == {"coal": 7, "iron-plate": 30}
    assert contents[3] == {"coal": 12, "iron-plate": 30}  # a give, shown at once
    assert contents[4] == contents[5] == contents[6] == {"coal": 12, "iron-plate": 10}
    assert contents[7:12] == [None] * 5  # walked out of reach; back does not reopen
    assert contents[12] == {}  # the empty chest, open
    assert contents[13:] == [{"coal": 12, "iron-plate": 10}] * 3


def test_world_inspect_returns_the_contents_and_costs_a_decision():
    env = _env("v3_inspect")
    world = WorldV3(SimBackend(env))
    chests = [e for e in world.entities() if e.kind == "container"]
    full = min(chests, key=lambda e: abs(e.x - world.me()[0]) + abs(e.y - world.me()[1]))
    assert world.opened() is None
    assert world.inspect(full) == {"coal": 7, "iron-plate": 30}
    assert world.decisions == 1 and world.opened()[1] == {"coal": 7, "iron-plate": 30}
    # A drill or a belt is not a chest: refused, no decision spent.
    furnace = WorldV3(SimBackend(_env("v3_take_fuel")))
    drill = next(e for e in furnace.entities() if e.kind == "mining-drill")
    assert furnace.inspect(drill) is None and furnace.decisions == 0
