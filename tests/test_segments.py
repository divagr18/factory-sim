"""Belt-line segments against the engine (csrc/fsim.c, "segments").

FactorioRL's tools/probe_logistics3.py and probe_logistics4.py measured, on
Factorio 2.0.60, when a belt lane's lines on consecutive belts merge into one
segment and split again, the order segments move in, and the per-tile merge
delay the simulator ships as a table (docs/sim-logistics.md there, "Third
probe" and "Fourth probe"). Checked here, every tick:

- which belts' lanes are one segment, in the three parity scenes the probe
  logged (`logistics3-segments.json.xz`: `line_equals` between every pair);
- every item on every belt of 32 one-world sideload rigs
  (`logistics3-sideload.json.xz`): two items reaching one empty lane on the
  same tick, one moved 8/256 further by the second insertion, for feeds of 1
  to 4 young or old belts, placed by script or dropped by two inserters built
  in either order;
- the fourth probe's rigs (`logistics4-*.json.xz`): two inserters acting on
  one tick, built in both orders (`order`: the last built acts first,
  dropping onto a belt as into a chest); a stopped line freed by a script, an
  inserter or a new belt (`sleep`: all of it moves on the next tick); a
  sideload or a drop arriving next to a moving item at every relative
  position and phase (`accept`);
- the table itself against the third probe's separate measurements
  (`logistics3-delays.json.xz`, its `logistics3-delay*.json.xz` merged).
"""

from __future__ import annotations

import json
import lzma
import re
from pathlib import Path

import pytest

from fsim import BELT_DELAY, ITEM_IDS, Sim, lib
from fsim.parity import GOLDEN
from fsim.trace import read_trace

HERE = Path(__file__).resolve().parent / "golden"
SEGMENTS = json.loads(lzma.decompress((HERE / "logistics3-segments.json.xz").read_bytes()))
SIDELOAD = json.loads(lzma.decompress((HERE / "logistics3-sideload.json.xz").read_bytes()))
ORDER = json.loads(lzma.decompress((HERE / "logistics4-order.json.xz").read_bytes()))
SLEEP = json.loads(lzma.decompress((HERE / "logistics4-sleep.json.xz").read_bytes()))
ACCEPT = json.loads(lzma.decompress((HERE / "logistics4-accept.json.xz").read_bytes()))
N, E, S, W = 0, 4, 8, 12


def _rigs(evidence: dict) -> list[str]:
    """The rig names of a probe's evidence file (its other keys are metadata)."""
    return sorted(k for k, v in evidence.items() if isinstance(v, dict) and "log" in v)


def _expand(log: list, last: int) -> list:
    out, at = [], 0
    for t in range(last + 1):
        while at + 1 < len(log) and log[at + 1][0] <= t:
            at += 1
        out.append(log[at][1])
    return out


def _classes(sim: Sim, belts: list[int]) -> list[list[str]]:
    """Per belt, per lane: the 1-based belts whose same lane is its segment,
    as the probe wrote `line_equals` classes."""
    heads = [[lib.fsim_belt_segment(sim.env, b, lane) for lane in (0, 1)] for b in belts]
    out = []
    for i in range(len(belts)):
        row = []
        for lane in (0, 1):
            same = [j + 1 for j in range(len(belts)) if heads[j][lane] == heads[i][lane]]
            row.append(",".join(map(str, same)))
        out.append(row)
    return out


@pytest.mark.parametrize("name", sorted(k for k in SEGMENTS if k.startswith("logistics_")))
def test_parity_scene_segments_every_tick(name):
    header, _ = read_trace(GOLDEN / f"{name}.jsonl.xz")
    sim = Sim()
    sim.reset(header["blueprint"])
    env = sim.env
    belts = sorted(
        (i for i in range(env.entity_count)
         if env.entities[i].alive and env.entities[i].kind == lib.K_BELT),
        key=lambda i: (env.entities[i].pos.y, env.entities[i].pos.x),
    )  # fmt: skip
    layout = SEGMENTS[name]["layout"]["belts"]
    assert [[env.entities[i].pos.x, env.entities[i].pos.y] for i in belts] == [
        [round(x * 256), round(y * 256)] for x, y, *_ in layout
    ]
    log = SEGMENTS[name]["log"]
    last = log[-1][0] + 20
    engine = _expand([[t, belts_] for t, (belts_, _) in log], last)
    for t in range(last + 1):
        if t:
            sim.step("wait", {}, 1)
        assert _classes(sim, belts) == engine[t], (name, t)


# ------------------------------------------------------------ sideload rigs


class SideloadRig:
    """tools/probe_logistics3.py `sideload_rig` / `inserter_rig`, one world."""

    def __init__(self, name: str) -> None:
        self.sim = Sim(water=[])
        self.sim.reset({"character": {"position": [0.5, 0.5]}})
        self.env = self.sim.env
        self.put_at = None
        if name.startswith("ins"):
            feed, order = int(name[5]), name[-2:]
            self._main(200, 200, feed)
            yk = 200 + feed
            for side in order:
                if side == "W":
                    c = self._add(lib.K_CHEST, 199, yk)
                    i = self._add(lib.K_INSERTER, 200, yk, W)
                else:
                    c = self._add(lib.K_CHEST, 203, yk)
                    i = self._add(lib.K_INSERTER, 202, yk, E)
                lib.fsim_entity_insert(self.env, c, ITEM_IDS["copper-plate"], 50)
                lib.fsim_entity_insert(self.env, i, ITEM_IDS["coal"], 2)
            return
        feed, where = (int(v) for v in re.search(r"f(\d)w(\d)", name).groups())
        at = re.search(r"at(\d+)", name)
        ox, oy = (int(at.group(1)), 113) if at else (300, 300)
        first = int(name[-2])
        self._main(ox, oy, feed)
        self.put = (self.belts[3 + where - 1], first)
        self.put_at = 600 if name.startswith("old") else 0

    def _add(self, kind: int, x: int, y: int, d: int = N) -> int:
        index = lib.fsim_add_entity(self.env, kind, x * 256 + 128, y * 256 + 128, d)
        assert index >= 0
        return index

    def _main(self, ox: int, oy: int, feed: int) -> None:
        self.belts = [self._add(lib.K_BELT, ox + i, oy, E) for i in range(3)]
        self.belts += [self._add(lib.K_BELT, ox + 1, oy + k, N) for k in range(1, feed + 1)]
        assert self.env.belt_delay_missing == 0
        lib.fsim_refresh(self.env)

    def _go(self) -> None:
        belt, first = self.put
        for lane in (first, 3 - first):
            assert lib.fsim_belt_insert(self.env, belt, lane - 1, 128, ITEM_IDS["copper-plate"])

    def sample(self) -> list:
        out = []
        for b in self.belts:
            e = self.env.entities[b]
            out.append([[e.lanes[lane].items[k].pos for k in range(e.lanes[lane].count)]
                        for lane in (0, 1)])  # fmt: skip
        return out

    def run(self, ticks: int) -> list:
        rows = []
        for t in range(ticks + 1):
            if t:
                lib.fsim_advance(self.env, 1)
            if t == self.put_at:
                self._go()
            rows.append(self.sample())
        return rows


def _engine_positions(row: list) -> list:
    """The probe's record of one rig: per belt, per lane, item positions."""
    out = []
    for b in row["a"]:
        lanes = []
        for lane in b:
            items = lane if isinstance(lane, list) else []
            lanes.append(sorted(p for p, _ in items))
        out.append(lanes)
    return out


@pytest.mark.parametrize("name", _rigs(SIDELOAD))
def test_sideload_rig_every_tick(name):
    log = SIDELOAD[name]["log"]
    rig = SideloadRig(name)
    ours = rig.run(log[-1][0])
    for t, row in log:
        assert [[sorted(lane) for lane in b] for b in ours[t]] == _engine_positions(row), (name, t)


# ------------------------------------------------------------ fourth probe


def _names_positions(lane) -> list:
    """[position, name] of each item on an engine lane record, sorted."""
    return sorted([p, rest[-1]] for p, *rest in (lane if isinstance(lane, list) else []))


class World:
    """One fresh world for a probe_logistics4 rig."""

    def __init__(self) -> None:
        self.sim = Sim(water=[])
        self.sim.reset({"character": {"position": [0.5, 0.5]}})
        self.env = self.sim.env
        self.belts: list[int] = []
        self.chests: list[int] = []
        self.ins: list[int] = []

    def add(self, kind: int, x: int, y: int, d: int = N) -> int:
        index = lib.fsim_add_entity(self.env, kind, x * 256 + 128, y * 256 + 128, d)
        assert index >= 0 and self.env.belt_delay_missing == 0
        return index

    def belt(self, x: int, y: int, d: int) -> int:
        self.belts.append(self.add(lib.K_BELT, x, y, d))
        return self.belts[-1]

    def chest(self, x: int, y: int, item: str | None = None, count: int = 50) -> int:
        c = self.add(lib.K_CHEST, x, y)
        if item:
            assert lib.fsim_entity_insert(self.env, c, ITEM_IDS[item], count) == count
        self.chests.append(c)
        return c

    def inserter(self, x: int, y: int, d: int, coal: int = 2) -> int:
        i = self.add(lib.K_INSERTER, x, y, d)
        if coal:
            lib.fsim_entity_insert(self.env, i, ITEM_IDS["coal"], coal)
        self.ins.append(i)
        return i

    def put(self, belt: int, lane: int, pos: int, item: str) -> None:
        lib.fsim_refresh(self.env)
        assert lib.fsim_belt_insert(self.env, belt, lane - 1, pos, ITEM_IDS[item])

    def record(self) -> dict:
        names = {v: k for k, v in ITEM_IDS.items()}
        env = self.env
        b = []
        for x in self.belts:
            e = env.entities[x]
            b.append([sorted([e.lanes[lane].items[k].pos, names[e.lanes[lane].items[k].item]]
                             for k in range(e.lanes[lane].count)) for lane in (0, 1)])  # fmt: skip
        c = [[[k + 1, names[st.item], st.count] for k, st in enumerate(env.entities[x].chest)
              if st.count] for x in self.chests]  # fmt: skip
        i = [names[env.entities[x].held] if env.entities[x].held else "" for x in self.ins]
        return {"b": b, "c": c, "i": i}


def _engine_record(row: dict) -> dict:
    return {
        "b": [[_names_positions(lane) for lane in belt] for belt in row["b"]],
        "c": [[list(s) for s in (c if isinstance(c, list) else [])] for c in row["c"]],
        "i": list(row["i"]) if isinstance(row["i"], list) else [],
    }


def _order_world(kind: str, order: str) -> World:
    """tools/probe_logistics4.py `_order_rig`, at (200, 200)."""
    w, ox, oy = World(), 200, 200
    if kind.startswith("belt"):
        for i in range(3):
            w.belt(ox + i, oy, E)
        w.belt(ox + 1, oy + 1, N)
        coal = 0 if kind == "belt_nofuel" else 2
        side = {"W": (lambda: w.chest(ox - 1, oy + 1, "copper-plate"),
                      lambda: w.inserter(ox, oy + 1, W, coal)),
                "E": (lambda: w.chest(ox + 3, oy + 1, "iron-plate"),
                      lambda: w.inserter(ox + 2, oy + 1, E, coal))}  # fmt: skip
        if kind == "belt_pre":
            steps = [side[order[0]][0], side[order[1]][0], side[order[0]][1], side[order[1]][1]]
        else:
            steps = [f for k in order for f in side[k]]
    elif kind == "vert":
        for i in range(3):
            w.belt(ox + i, oy, E)
        side = {"N": (lambda: w.chest(ox + 1, oy - 2, "copper-plate"),
                      lambda: w.inserter(ox + 1, oy - 1, N)),
                "S": (lambda: w.chest(ox + 1, oy + 2, "iron-plate"),
                      lambda: w.inserter(ox + 1, oy + 1, S))}  # fmt: skip
        steps = [f for k in order for f in side[k]]
    elif kind == "chest1":
        w.chest(ox + 1, oy, "stone", 750)
        side = {"W": (lambda: w.chest(ox - 1, oy, "copper-plate"), lambda: w.inserter(ox, oy, W)),
                "E": (lambda: w.chest(ox + 3, oy, "iron-plate"),
                      lambda: w.inserter(ox + 2, oy, E))}  # fmt: skip
        steps = [f for k in order for f in side[k]]
    else:  # pick1, pick1_chest
        w.chest(ox + 1, oy, "iron-plate", 1)
        if kind == "pick1":
            side = {
                "W": (lambda: w.belt(ox - 1, oy, N), lambda: w.inserter(ox, oy, E)),
                "E": (lambda: w.belt(ox + 3, oy, N), lambda: w.inserter(ox + 2, oy, W)),
            }
        else:
            side = {
                "W": (lambda: w.chest(ox - 1, oy), lambda: w.inserter(ox, oy, E)),
                "E": (lambda: w.chest(ox + 3, oy), lambda: w.inserter(ox + 2, oy, W)),
            }
        steps = [f for k in order for f in side[k]]
    for step in steps:
        step()
    lib.fsim_refresh(w.env)
    return w


@pytest.mark.parametrize("name", _rigs(ORDER))
def test_order_rig_every_tick(name):
    kind, order = name.rsplit("_", 1)
    w = _order_world(kind, order)
    for row in ORDER[name]["log"]:
        t = row["t"]
        while w.sim.tick < t:
            lib.fsim_advance(w.env, 1)
        assert w.record() == _engine_record(row), (name, t)


def _sleep_world(how: str):
    w = World()
    for i in range(7):
        w.belt(306 + i, 345, E)
    for b in w.belts:
        for pos in (0, 64, 128, 192):
            w.put(b, 1, pos, "iron-plate")

    def free():
        if how == "script":
            # LuaTransportLine.remove_item on the last belt: its front item
            assert lib.fsim_belt_remove(w.env, w.belts[6], 0, 0)
        elif how == "ins":
            w.chest(312, 347)
            w.inserter(312, 346, N, 5)
        else:
            w.belt(313, 345, E)
        lib.fsim_refresh(w.env)

    return w, free


@pytest.mark.parametrize("name", _rigs(SLEEP))
def test_sleep_rig_every_tick(name):
    how, at = name.split("_")
    w, free = _sleep_world(how)
    for row in SLEEP[name]["log"]:
        t = row["t"]
        while w.sim.tick < t:
            lib.fsim_advance(w.env, 1)
        if t == int(at):
            free()
        assert w.record() == _engine_record(row), (name, t)


def _accept_world(spec: dict) -> World:
    w, x, y = World(), 200, 200
    if spec["kind"] == "side":
        for i in range(3):
            w.belt(x + i, y, E)
        w.belt(x + 1, y + 1, N)
        w.put(w.belts[3], spec["L"], spec["F"], "copper-plate")
        w.put(w.belts[2 - spec["C"] // 256], 2, spec["C"] % 256, "iron-plate")
    else:
        for i in range(4):
            w.belt(x + i, y, E)
        c = w.add(lib.K_CHEST, x + 2, y + 2)
        lib.fsim_entity_insert(w.env, c, ITEM_IDS["copper-plate"], 5)
        i = w.add(lib.K_INSERTER, x + 2, y + 1, S)
        lib.fsim_entity_insert(w.env, i, ITEM_IDS["coal"], 2)
        w.put(w.belts[3 - spec["C"] // 256], 1, spec["C"] % 256, "iron-plate")
    return w


GROUPS = sorted({name.rsplit("_c", 1)[0] for name in ACCEPT["rigs"]})


@pytest.mark.parametrize("group", GROUPS)
def test_accept_rigs_every_tick(group):
    names = [n for n in ACCEPT["rigs"] if n.rsplit("_c", 1)[0] == group]
    assert names
    for name in names:
        w = _accept_world(ACCEPT["rigs"][name])
        for t, belts in ACCEPT["log"][name]:
            while w.sim.tick < t:
                lib.fsim_advance(w.env, 1)
            engine = [[_names_positions(lane) for lane in b] for b in belts]
            assert w.record()["b"] == engine, (name, t)


# ------------------------------------------------------------ the table


def test_table_agrees_with_the_third_probe():
    """Every delay the third probe measured separately (its parity-scene area
    and the areas of the rigs it checked) is the table's -- but where the
    tile north of it, the neighbour that measurement rebuilt every tick, has a
    delay of 1: that neighbour's own timer runs out first and the reading is 1
    (tools/make_belt_delay.py)."""
    table = json.loads(lzma.decompress((HERE / "logistics3-delays.json.xz").read_bytes()))
    assert len(table) > 9000
    artefacts = 0
    for key, d in table.items():
        x, y, lane = (int(v) for v in key.split(","))
        ours = lib.fsim_belt_delay(x, y, lane - 1)
        if ours != d:
            assert d == 1 and lib.fsim_belt_delay(x, y - 1, lane - 1) == 1, (key, d, ours)
            artefacts += 1
    assert artefacts < 50


def test_table_covers_the_scene_area_and_refuses_outside():
    assert BELT_DELAY["engine"] == {"version": "2.0.60", "build": 83512}
    assert [r["rect"] for r in BELT_DELAY["rects"]][0] == [-128, -128, 128, 128]
    for x, y in ((-128, -128), (127, 127), (0, 0), (-128, 127)):
        assert 1 <= lib.fsim_belt_delay(x, y, 0) <= 600
        assert 1 <= lib.fsim_belt_delay(x, y, 1) <= 600
    assert lib.fsim_belt_delay(-129, 0, 0) == -1
    assert lib.fsim_belt_delay(0, 128, 1) == -1
