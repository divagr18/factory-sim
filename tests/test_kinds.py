"""Entity kinds: the kind table, the capacities, and the furnace and fuel rules
the recorded traces do not reach (they only smelt iron ore on coal)."""

from __future__ import annotations

import pytest

from fsim import KINDS, Sim, lib

SMELT_TICKS = 192  # 3.2 s
FURNACE_DRAW = 1500.0  # J per tick


def _furnace_with(inventory: dict) -> tuple[Sim, int]:
    """A dry map, the character at the origin, a furnace placed beside it."""
    sim = Sim(water=[])
    inventory = {"stone-furnace": 1, **inventory}
    sim.reset({"character": {"position": [0.5, 0.5], "inventory": inventory}})
    sim.step("place_at", {"item": "stone-furnace", "position": [3.0, 0.0]}, ticks=1)
    assert sim.action_outcome() == ("completed", None)
    return sim, sim.env.act.handle


def _furnace(sim: Sim):
    furnaces = [
        sim.env.entities[i]
        for i in range(sim.env.entity_count)
        if sim.env.entities[i].alive and sim.env.entities[i].kind == lib.K_FURNACE
    ]
    assert len(furnaces) == 1
    return furnaces[0]


def _give(sim: Sim, handle: int, item: str, count: int) -> None:
    sim.step("give_to", {"to": f"h{handle}", "item": item, "count": count}, ticks=1)
    assert sim.action_outcome() == ("completed", None)


def test_furnace_smelts_copper_ore_into_copper_plate():
    sim, h = _furnace_with({"copper-ore": 5, "coal": 5})
    _give(sim, h, "coal", 5)
    _give(sim, h, "copper-ore", 5)
    sim.step("wait", ticks=6 * SMELT_TICKS)
    furnace = _furnace(sim)
    assert furnace.result.item == lib.IT_COPPER_PLATE
    assert furnace.result.count == 5
    assert sim.env.produced[lib.IT_COPPER_PLATE] == 5
    assert sim.env.produced[lib.IT_IRON_PLATE] == 0


def test_picking_up_a_furnace_mid_copper_craft_returns_copper_ore():
    sim, h = _furnace_with({"copper-ore": 1, "coal": 1})
    _give(sim, h, "coal", 1)
    _give(sim, h, "copper-ore", 1)
    sim.step("wait", ticks=SMELT_TICKS // 2)
    assert _furnace(sim).crafting and _furnace(sim).source.count == 0
    sim.step("mine_at", {"handle": f"h{h}", "count": 1}, ticks=60)
    assert sim.inventory() == {"copper-ore": 1, "stone-furnace": 1}


@pytest.mark.parametrize(("fuel", "megajoules"), [("coal", 4), ("wood", 2)])
def test_one_item_of_fuel_burns_for_its_fuel_value(fuel, megajoules):
    sim, h = _furnace_with({"iron-ore": 20, fuel: 1})
    _give(sim, h, "iron-ore", 20)
    _give(sim, h, fuel, 1)
    sim.step("wait", ticks=1)
    furnace = _furnace(sim)
    assert furnace.burning == {"coal": lib.IT_COAL, "wood": lib.IT_WOOD}[fuel]
    assert furnace.energy + furnace.remaining == pytest.approx(megajoules * 1e6 - FURNACE_DRAW)
    sim.step("wait", ticks=6000)
    # Whole crafts the one item pays for: 1,333 or 2,666 ticks of draw.
    assert _furnace(sim).result.count == int(megajoules * 1e6 / FURNACE_DRAW) // SMELT_TICKS


def test_every_kind_has_a_row():
    for kind in KINDS:
        assert lib.fsim_kind_mining_time(kind) > 0
    assert lib.fsim_kind_flags(lib.K_DRILL) & lib.KF_DIRECTED
    assert not lib.fsim_kind_flags(lib.K_FURNACE) & lib.KF_DIRECTED
    assert lib.fsim_kind_flags(lib.K_WALL) == lib.KF_COLLIDES | lib.KF_BLOCKS_WALKING
    assert lib.fsim_kind_flags(lib.K_PILE) == 0
    for kind in (lib.K_DRILL, lib.K_FURNACE):
        flags = lib.fsim_kind_flags(kind)
        assert flags & lib.KF_BURNER and flags & lib.KF_MACHINE
    assert lib.fsim_capacity(lib.K_DRILL) == pytest.approx(2500 * 16 / 15)
    assert lib.fsim_capacity(lib.K_FURNACE) == pytest.approx(1500 * 16 / 15)


def _walled(count: int) -> Sim:
    """`count` walls in rows of 20 from (2, 2), the character north of them."""
    sim = Sim(water=[])
    walls = [
        {"name": "stone-wall", "position": [2.5 + i % 20, 2.5 + i // 20]} for i in range(count)
    ]
    sim.reset({"character": {"position": [5.5, 0.5]}, "entities": walls})
    return sim


def test_a_scene_holds_more_than_the_old_128_entities():
    sim = _walled(300)
    assert sim.env.entity_count == 300
    # The observation still publishes the engine's 48 nearest.
    assert sim.env.seen_count == lib.FSIM_MAX_SWEEP
    sim.step("move_south", ticks=30)
    # 30 strides would reach y = 5; the first row (its box from y = 2.21) stops it.
    assert 256 < sim.env.char_pos.y < int(2.25 * 256)


def test_entities_past_the_table_are_dropped_not_written_out_of_bounds():
    sim = _walled(lib.FSIM_MAX_ENTITIES + 40)
    assert sim.env.entity_count == lib.FSIM_MAX_ENTITIES
