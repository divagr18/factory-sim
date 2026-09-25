"""The simulator against FactorioRL's second hand-mining probe, rig by rig.

`tools/probe_handmine2.py` recorded on the engine, every tick:

- `beltpick`..`beltpick6`, `extra`, `beltnext`: a belt built over piles (and
  next to loaded belts) -- where each pile lands on which lane, and which are
  refused and dropped round the belt;
- `resolve`: which of several piles a tile handle resolves to;
- `water`: where ore mined with no room lands when water is near.

Each rig here is rebuilt in the simulator and must come out as the engine's.
"""

from __future__ import annotations

import json
import lzma

import pytest

from fsim import ITEM_IDS, Sim, ffi, lib
from fsim.parity import GOLDEN

TILE = 256
DIRS = {0: "north", 4: "east", 8: "south", 12: "west"}
PICKUP = ("beltpick", "beltpick2", "beltpick3", "beltpick4", "beltpick5", "beltpick6",
          "extra", "beltnext")  # fmt: skip


def _load(family: str) -> dict:
    path = GOLDEN / f"handmine2-{family}.json.xz"
    if not path.is_file():
        pytest.skip(f"no {path.name}")
    return json.loads(lzma.decompress(path.read_bytes()))


def _rigs(families, keep):
    out = []
    for family in families:
        path = GOLDEN / f"handmine2-{family}.json.xz"
        if not path.is_file():
            continue
        doc = json.loads(lzma.decompress(path.read_bytes()))
        for name, rig in doc["rigs"].items():
            if keep(rig):
                out.append(pytest.param(family, name, id=f"{family}/{name}"))
    return out


def _pickup_rig(rig) -> bool:
    kinds = {e["name"] for e in rig["entities"]}
    return (
        any(op[1] == "place" for op in rig["ops"])
        and "item-on-ground" in kinds
        and kinds <= {"item-on-ground", "transport-belt"}
        and not rig["resources"]
    )


class _Items:
    """Engine item names onto distinct simulator items, one rig at a time: the
    probe tells piles apart by item, some of which the simulator lacks."""

    SPARE = ("stone-wall", "small-electric-pole", "transport-belt", "burner-inserter",
             "burner-mining-drill", "stone-furnace", "wooden-chest")  # fmt: skip

    def __init__(self):
        self.to_sim: dict[str, int] = {}

    def __call__(self, name: str) -> int:
        if name not in self.to_sim:
            if name in ITEM_IDS and ITEM_IDS[name] not in self.to_sim.values():
                self.to_sim[name] = ITEM_IDS[name]
            else:
                spare = next(s for s in self.SPARE if ITEM_IDS[s] not in self.to_sim.values()
                             and s not in self.to_sim)  # fmt: skip
                self.to_sim[name] = ITEM_IDS[spare]
        return self.to_sim[name]

    def name(self, item: int) -> str:
        return next(n for n, i in self.to_sim.items() if i == item)


def _sim(rig, water=()) -> Sim:
    sim = Sim(list(water))
    blueprint = {
        "character": {"position": list(rig["start"]), "inventory": {}},
        "entities": [],
        "resources": [
            {"name": r[2], "position": [r[0] + 0.5, r[1] + 0.5], "amount": r[3]}
            for r in rig["resources"]
        ],
        "markers": {},
    }
    sim.reset(blueprint)
    return sim


def _pile(sim, x: float, y: float, item: int, count: int = 1) -> int:
    index = lib.fsim_add_entity(sim.env, lib.K_PILE, round(x * TILE), round(y * TILE), 0)
    sim.env.entities[index].pile.item = item
    sim.env.entities[index].pile.count = count
    return index


def _wait(sim, ticks: int) -> None:
    for _ in range(ticks):
        sim.step("wait", {}, 1)


def _lanes(sim, index: int, items: _Items) -> list:
    e = sim.env.entities[index]
    out = []
    for lane in range(2):
        last = e.lane_length[lane] - 1
        out.append(
            sorted(
                [min(it.pos, last), items.name(it.item)]
                for it in e.lanes[lane].items[0 : e.lanes[lane].count]
            )
        )
    return out


@pytest.mark.parametrize(("family", "name"), _rigs(PICKUP, _pickup_rig))
def test_belt_built_over_piles(family, name):
    doc = _load(family)
    rig, log = doc["rigs"][name], doc["log"][name]
    place = next(op for op in rig["ops"] if op[1] == "place")
    items = _Items()
    sim = _sim(rig)
    env = sim.env
    for i in range(lib.FSIM_MAIN_SLOTS):
        env.main[i].count = 0
    env.main[0].item, env.main[0].count = ITEM_IDS["transport-belt"], 5
    belts = {}
    for spec in rig["entities"]:
        if spec["name"] == "item-on-ground":
            item, count = spec["contents"]["stack"]
            _pile(sim, spec["x"], spec["y"], items(item), count)
        else:
            index = lib.fsim_add_entity(env, lib.K_BELT, round(spec["x"] * TILE),
                                        round(spec["y"] * TILE), spec["d"])  # fmt: skip
            belts[spec["label"]] = index
            for lane, position, item in spec["contents"].get("lanes") or []:
                assert lib.fsim_belt_insert(env, index, lane - 1, position, items(item))
    _wait(sim, place[0])
    count = env.entity_count
    sim.step("place_at", {"item": "transport-belt", "position": [place[4], place[5]],
                          "direction": DIRS[place[6]]}, 0)  # fmt: skip
    assert sim.action_outcome() == ("completed", None)
    belts[place[2]] = next(i for i in range(count, env.entity_count)
                           if env.entities[i].kind == lib.K_BELT)  # fmt: skip
    want = next(r for t, r in log if t == place[0])
    for label, index in belts.items():
        got = _lanes(sim, index, items)
        expected = [sorted([p, it] for p, it in lane) for lane in want["lanes"][label]]
        assert got == expected, (label, got, expected)
    piles = sorted(
        [e.pos.x, e.pos.y, items.name(e.pile.item), e.pile.count]
        for e in (env.entities[i] for i in range(env.entity_count))
        if e.alive and e.kind == lib.K_PILE
    )
    assert piles == sorted(want["piles"]), (piles, want["piles"])


@pytest.mark.parametrize(("family", "name"), _rigs(("resolve",), lambda rig: True))
def test_tile_handle_resolves_to_the_newest_pile(family, name):
    """The engine lists the piles within half a tile of the centre, newest
    first (`handles.resolve` takes the first). Here an older pile on the tile
    centre gives the tile its handle; it is what resolves only when the engine
    found none of the rig's."""
    doc = _load(family)
    rig = doc["rigs"][name]
    note = next(n for n in doc["notes"] if n[0] == name and n[2] == "resolve")
    first = note[4]
    items = _Items()
    sim = _sim({**rig, "start": [2.5, 3.5]})
    anchor = _pile(sim, 2.5, 0.5, items("anchor"))
    for spec in rig["entities"]:
        _pile(sim, spec["x"], spec["y"], items(spec["contents"]["stack"][0]))
    lib.fsim_observe(sim.env)
    handle = next(int(r["h"][1:]) for r in sim.observation()["entities"]
                  if r["name"] == "item-on-ground" and r["p"] == [2.5, 0.5])  # fmt: skip
    kind, index = ffi.new("int32_t *"), ffi.new("int32_t *")
    assert lib.fsim_resolve(sim.env, handle, kind, index) == 0
    e = sim.env.entities[index[0]]
    if first == "none":
        assert index[0] == anchor
    else:
        assert [e.pos.x, e.pos.y, items.name(e.pile.item)] == first


@pytest.mark.parametrize(("family", "name"), _rigs(("water",), lambda rig: True))
def test_water_keeps_a_dropped_item_off(family, name):
    doc = _load(family)
    rig, log = doc["rigs"][name], doc["log"][name]
    sim = _sim(rig, water=[tuple(w) for w in rig["water"]])
    env = sim.env
    for spec in rig["entities"]:
        _pile(sim, spec["x"], spec["y"], ITEM_IDS[spec["contents"]["stack"][0]])
    before = {(e.pos.x, e.pos.y) for e in (env.entities[i] for i in range(env.entity_count))
              if e.alive and e.kind == lib.K_PILE}  # fmt: skip
    # The probe mined with the inventory full; here a slot is left for the
    # mod's check and filled before the first ore arrives.
    for i in range(lib.FSIM_MAIN_SLOTS):
        env.main[i].item, env.main[i].count = ITEM_IDS["wood"], 100
    env.main[79].item, env.main[79].count = 0, 0
    lib.fsim_observe(env)
    tile = next(r["h"] for r in sim.observation()["resources"]["tiles"]
                if r["p"] == [2.5, 0.5])  # fmt: skip
    sim.step("mine_at", {"handle": tile, "count": 20}, 1)
    env.main[79].item, env.main[79].count = ITEM_IDS["wood"], 100
    spills = []
    for _ in range(3):
        sim.step("wait", {}, 121)
        for e in (env.entities[i] for i in range(env.entity_count)):
            if e.alive and e.kind == lib.K_PILE and (e.pos.x, e.pos.y) not in before:
                if [e.pos.x, e.pos.y] not in spills:
                    spills.append([e.pos.x, e.pos.y])
    want = []
    blockers = {(round(s["x"] * TILE), round(s["y"] * TILE)) for s in rig["entities"]}
    for _, r in log:
        for p in r["piles"]:
            if (p[0], p[1]) not in blockers and [p[0], p[1]] not in want:
                want.append([p[0], p[1]])
    assert spills == want
