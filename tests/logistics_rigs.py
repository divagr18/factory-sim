"""FactorioRL's logistics probe, rebuilt in the simulator.

`tools/probe_logistics.py` in FactorioRL builds seventeen rigs of belts,
burner inserters, chests, drills and furnaces far from any scene, fuels them,
and reads every rig after every tick for 3,000 ticks. `Rigs` builds the same
entities, in the same order, at the same places, feeds the belts the way the
probe's `SAMPLE` does, and reads the same fields, so the two can be compared
tick by tick (tests/test_mechanics_logistics.py).

Positions follow the probe: origin (OX, OY) = (160, 160), and a rig offset
(x, y) is the tile whose centre is (OX + x + 0.5, OY + y + 0.5).
"""

from __future__ import annotations

from fsim import ITEM_IDS, ITEM_NAMES, STATUS_NAME, Sim, g, hand_position, lib

OX, OY = 160, 160
TICKS = 3000
UNBLOCK = 1500
N, E, S, W = 0, 4, 8, 12


def tile(x: int, y: int) -> tuple[int, int]:
    """The centre of rig tile (x, y), in 1/256."""
    return (OX + x) * 256 + 128, (OY + y) * 256 + 128


class Rigs:
    def __init__(self) -> None:
        self.ore: list[tuple[int, int]] = []
        self._layout()
        self.sim = Sim(water=[])
        self.sim.reset(
            {
                "character": {"position": [0.5, 0.5]},
                "resources": [
                    {"name": "iron-ore", "position": [x + 0.5, y + 0.5], "amount": 5000}
                    for x, y in self.ore
                ],
            }
        )
        self.env = self.sim.env
        self.fed = {"straight": [0, 0], "bend": [0, 0], "side": [0, 0], "curve": [0, 0]}
        self.flow_next = 0
        self.t = 0
        self._build()
        self._fuel()

    # ---------------------------------------------------------------- layout
    def _layout(self) -> None:
        for x, y in ((-10, -30), (-10, 0), (12, 31), (18, 31), (24, 31)):
            for dx in (-1, 0):
                for dy in (-1, 0):
                    self.ore.append((OX + x + dx, OY + y + dy))

    def _add(self, kind: int, x: int, y: int, direction: int = N) -> int:
        index = lib.fsim_add_entity(self.env, kind, x, y, direction)
        assert index >= 0
        return index

    def belt(self, x: int, y: int, direction: int) -> int:
        return self._add(lib.K_BELT, *tile(x, y), direction)

    def chest(self, x: int, y: int) -> int:
        return self._add(lib.K_CHEST, *tile(x, y))

    def inserter(self, x: int, y: int, direction: int) -> int:
        return self._add(lib.K_INSERTER, *tile(x, y), direction)

    def machine(self, kind: int, x: int, y: int, direction: int = N) -> int:
        """A 2x2 machine centred on the tile corner (OX + x, OY + y)."""
        return self._add(kind, (OX + x) * 256, (OY + y) * 256, direction)

    def insert(self, index: int, item: str, count: int) -> None:
        assert lib.fsim_entity_insert(self.env, index, ITEM_IDS[item], count) == count

    def _build(self) -> None:
        P: dict = {}
        P["straight"] = [self.belt(-30 + i, -30, E) for i in range(10)]
        P["curve"] = [self.belt(-30 + i, -24, E) for i in range(4)]
        P["curve"] += [self.belt(-26, -24 + i, S) for i in range(4)]
        P["side_main"] = [self.belt(-30 + i, -14, E) for i in range(6)]
        P["side_feed"] = [self.belt(-28, -10 - i, N) for i in range(4)]
        P["drill"] = self.machine(lib.K_DRILL, -10, -30, S)
        # Its drop position (150.5, 131.296875) is on rig tile (-10, -29).
        P["drill_belt"] = [self.belt(-10, -29, E)]
        P["c2c_src"] = self.chest(0, -31)
        P["c2c_ins"] = self.inserter(0, -30, N)
        P["c2c_dst"] = self.chest(0, -29)
        self.insert(P["c2c_src"], "iron-plate", 50)
        P["bend_belt"] = [self.belt(8 + i, -30, E) for i in range(4)]
        P["bend_ins"] = self.inserter(11, -29, N)
        P["bend_furnace"] = self.machine(lib.K_FURNACE, 12, -27)
        P["flow_belt"] = [self.belt(-30 + i, 0, E) for i in range(16)]
        P["flow_ins"] = self.inserter(-22, 1, N)
        P["flow_chest"] = self.chest(-22, 2)
        P["flow2_belt"] = [self.belt(-30 + i, 5, E) for i in range(16)]
        P["flow2_ins"] = self.inserter(-22, 6, N)
        P["flow2_chest"] = self.chest(-22, 7)
        P["fout_furnace"] = self.machine(lib.K_FURNACE, 20, 0)
        self.insert(P["fout_furnace"], "iron-ore", 20)
        self.insert(P["fout_furnace"], "coal", 2)
        P["fout_ins"] = self.inserter(19, 1, N)
        P["fout_belt"] = [self.belt(19 + i, 2, E) for i in range(4)]
        P["fill_ore_chest"] = self.chest(0, 10)
        self.insert(P["fill_ore_chest"], "iron-ore", 100)
        P["fill_ore_ins"] = self.inserter(0, 11, N)
        P["fill_ore_furnace"] = self.machine(lib.K_FURNACE, 1, 13)
        P["fill_coal_chest"] = self.chest(8, 10)
        self.insert(P["fill_coal_chest"], "coal", 50)
        P["fill_coal_ins"] = self.inserter(8, 11, N)
        P["fill_coal_furnace"] = self.machine(lib.K_FURNACE, 9, 13)
        P["fuel_src"] = self.chest(16, 10)
        self.insert(P["fuel_src"], "iron-plate", 200)
        P["fuel_ins"] = self.inserter(16, 11, N)
        P["fuel_dst"] = self.chest(16, 12)
        P["wood_src"] = self.chest(22, 10)
        self.insert(P["wood_src"], "iron-plate", 200)
        P["wood_ins"] = self.inserter(22, 11, N)
        P["wood_dst"] = self.chest(22, 12)
        P["tick_drill"] = self.machine(lib.K_DRILL, -10, 0, S)
        P["tick_belt"] = [self.belt(-10, 1, E), self.belt(-9, 1, E)]
        P["tick_ins"] = self.inserter(-9, 2, N)
        P["tick_chest"] = self.chest(-9, 3)
        P["drop"] = []
        for i, direction in enumerate((E, W, S)):
            x = -30 + 6 * i
            c = self.chest(x, 20)
            self.insert(c, "iron-plate", 20)
            r = {"chest": c, "ins": self.inserter(x, 21, N), "belt": []}
            for k in range(3):
                bx, by = {E: (x + k, 22), W: (x - k, 22), S: (x, 22 + k)}[direction]
                r["belt"].append(self.belt(bx, by, direction))
            P["drop"].append(r)
        P["phase"] = []
        for lane in (1, 2):
            for k in range(8):
                x, y = -30 + 4 * k, 25 + 5 * lane
                r = {
                    "main": [self.belt(x - 1, y, E), self.belt(x, y, E), self.belt(x + 1, y, E)],
                    "feed": [self.belt(x, y + 1, N)],
                }
                # insert_at(0.5 + k/256) on the feed's lane.
                item = ITEM_IDS["copper-plate"]
                assert lib.fsim_belt_insert(self.env, r["feed"][0], lane - 1, 128 + k, item)
                P["phase"].append(r)
        P["order_a"] = self._order_rig(12, 31, reverse=False)
        P["order_b"] = self._order_rig(18, 31, reverse=True)
        P["same_drill"] = self.machine(lib.K_DRILL, 24, 31, S)
        P["same_belt"] = [self.belt(24, 32, E)]
        P["same_ins"] = self.inserter(24, 33, N)
        P["same_chest"] = self.chest(24, 34)
        P["self_src"] = self.chest(30, 30)
        self.insert(P["self_src"], "coal", 50)
        P["self_ins"] = self.inserter(30, 31, N)
        P["self_dst"] = self.chest(30, 32)
        P["hot_chest"] = self.chest(34, 30)
        self.insert(P["hot_chest"], "iron-ore", 50)
        P["hot_ins"] = self.inserter(34, 31, N)
        P["hot_furnace"] = self.machine(lib.K_FURNACE, 35, 33)
        self.insert(P["hot_furnace"], "coal", 5)
        self.P = P

    def _order_rig(self, cx: int, cy: int, reverse: bool) -> dict:
        r: dict = {}

        def rest():
            steps = [
                lambda: r.__setitem__("src", self.chest(cx, cy + 1)),
                lambda: r.__setitem__("ins", self.inserter(cx, cy + 2, N)),
                lambda: r.__setitem__("dst", self.chest(cx, cy + 3)),
            ]
            for step in reversed(steps) if reverse else steps:
                step()

        def drill():
            r["drill"] = self.machine(lib.K_DRILL, cx, cy, S)

        if reverse:
            rest()
            drill()
        else:
            drill()
            rest()
        return r

    def _fuel(self) -> None:
        P = self.P
        for key in ("drill", "tick_drill", "same_drill"):
            self.insert(P[key], "coal", 5)
        for key in ("c2c_ins", "bend_ins", "flow_ins", "flow2_ins", "fout_ins", "fill_ore_ins",
                    "fill_coal_ins", "tick_ins", "same_ins", "hot_ins"):  # fmt: skip
            self.insert(P[key], "coal", 5)
        for r in P["drop"]:
            self.insert(r["ins"], "coal", 5)
        for r in (P["order_a"], P["order_b"]):
            self.insert(r["drill"], "coal", 5)
            self.insert(r["ins"], "coal", 5)
        self.insert(P["fuel_ins"], "coal", 1)
        self.insert(P["wood_ins"], "wood", 1)

    # ---------------------------------------------------------------- running
    def _feed(self) -> None:
        """The probe's per-tick feeding, run before each sample is read."""
        P = self.P
        plate, ore, copper = (ITEM_IDS[n] for n in ("iron-plate", "iron-ore", "copper-plate"))
        for lane in (0, 1):
            for key, first, limit, item in (
                ("straight", P["straight"][0], 6, plate),
                ("bend", P["bend_belt"][0], 4, ore),
                ("side", P["side_feed"][0], 3, copper),
                ("curve", P["curve"][0], 1, plate),
            ):
                if self.fed[key][lane] < limit and lib.fsim_belt_insert_back(
                    self.env, first, lane, item
                ):
                    self.fed[key][lane] += 1
        if self.t >= self.flow_next:
            lib.fsim_belt_insert_back(self.env, P["flow_belt"][0], 1, plate)
            lib.fsim_belt_insert_back(self.env, P["flow2_belt"][0], 0, plate)
            self.flow_next = self.t + 20
        if self.t == UNBLOCK and len(P["drill_belt"]) == 1:
            P["drill_belt"].append(self.belt(-9, -29, E))

    def samples(self):
        """Every tick's sample, t = 0..TICKS, as the probe's `SAMPLE` reads it."""
        self._feed()
        lib.fsim_refresh(self.env)
        yield self.sample()
        for self.t in range(1, TICKS + 1):
            lib.fsim_advance(self.env, 1)
            self._feed()
            lib.fsim_refresh(self.env)
            yield self.sample()

    # ---------------------------------------------------------------- reading
    def e(self, index: int):
        return self.env.entities[index]

    def lanes(self, belts: list[int]) -> list:
        """Per belt, both lanes as [name, position] by position (ids dropped)."""
        out = []
        for index in belts:
            b = self.e(index)
            out.append(
                [
                    [[ITEM_NAMES[it.item], it.pos] for it in b.lanes[k].items[0 : b.lanes[k].count]]
                    for k in (0, 1)
                ]
            )
        return out

    def ins(self, index: int) -> dict:
        s = self.e(index)
        pickup, drop = self.sim._inserter_points(s)
        fuel = s.fuel
        return {
            "held": ITEM_NAMES[s.held] if s.held else None,
            "hand": hand_position(s, pickup, drop),
            "energy": g(s.energy),
            "remaining": g(s.remaining),
            "burning": ITEM_NAMES[s.burning] if s.burning else None,
            "coal": fuel.count if fuel.count and fuel.item == ITEM_IDS["coal"] else 0,
            "wood": fuel.count if fuel.count and fuel.item == ITEM_IDS["wood"] else 0,
            "status": STATUS_NAME[s.status],
        }

    def fur(self, index: int) -> dict:
        f = self.e(index)
        ore, coal, plate = ITEM_IDS["iron-ore"], ITEM_IDS["coal"], ITEM_IDS["iron-plate"]
        return {
            "src": f.source.count if f.source.item == ore else 0,
            "fuel": f.fuel.count if f.fuel.item == coal else 0,
            "res": f.result.count if f.result.item == plate else 0,
            "status": STATUS_NAME[f.status],
            "progress": g(f.progress),
        }

    def box(self, index: int, item: str) -> int:
        c = self.e(index)
        return sum(
            c.chest[i].count
            for i in range(lib.FSIM_CHEST_SLOTS)
            if c.chest[i].count and c.chest[i].item == ITEM_IDS[item]
        )

    def drill(self, index: int) -> dict:
        d = self.e(index)
        return {"status": STATUS_NAME[d.status], "progress": g(d.progress), "energy": g(d.energy)}

    def sample(self) -> dict:
        P = self.P
        out = {
            "straight": self.lanes(P["straight"]),
            "curve": self.lanes(P["curve"]),
            "side_main": self.lanes(P["side_main"]),
            "side_feed": self.lanes(P["side_feed"]),
            "drill": self.drill(P["drill"]),
            "drill_belt": self.lanes(P["drill_belt"]),
            "c2c": self.ins(P["c2c_ins"]),
            "c2c_dst": self.box(P["c2c_dst"], "iron-plate"),
            "fout": self.ins(P["fout_ins"]),
            "fout_furnace": self.fur(P["fout_furnace"]),
            "fout_belt": self.lanes(P["fout_belt"]),
            "fill_ore": self.ins(P["fill_ore_ins"]),
            "fill_ore_furnace": self.fur(P["fill_ore_furnace"]),
            "fill_coal": self.ins(P["fill_coal_ins"]),
            "fill_coal_furnace": self.fur(P["fill_coal_furnace"]),
            "fuel": self.ins(P["fuel_ins"]),
            "fuel_dst": self.box(P["fuel_dst"], "iron-plate"),
            "wood": self.ins(P["wood_ins"]),
            "wood_dst": self.box(P["wood_dst"], "iron-plate"),
            "self": {
                "ins": self.ins(P["self_ins"]),
                "src": self.box(P["self_src"], "coal"),
                "dst": self.box(P["self_dst"], "coal"),
            },
            "hot": {"ins": self.ins(P["hot_ins"]), "furnace": self.fur(P["hot_furnace"])},
        }
        for i, r in enumerate(P["drop"], start=1):
            out[f"drop{i}"] = {"ins": self.ins(r["ins"]), "belt": self.lanes(r["belt"])}
        for i, r in enumerate(P["phase"], start=1):
            out[f"phase{i}"] = {"main": self.lanes(r["main"]), "feed": self.lanes(r["feed"])}
        for key in ("order_a", "order_b"):
            r = P[key]
            out[key] = {
                "drill": self.drill(r["drill"]),
                "src": self.box(r["src"], "iron-ore"),
                "ins": self.ins(r["ins"]),
                "dst": self.box(r["dst"], "iron-ore"),
            }
        return out
