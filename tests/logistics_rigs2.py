"""FactorioRL's second logistics probe, rebuilt in the simulator.

`tools/probe_logistics2.py` in FactorioRL builds about 460 rigs far from any
scene and reads each after every tick for 1,500 ticks. `Rigs2` builds the rigs
whose mechanics the simulator reproduces, entity by entity in the probe's
order, at the origins the probe recorded (`bases`), runs the probe's scripted
events and belt feeders, and reads every rig the way the probe's `SAMPLE`
does, so the two compare tick by tick (tests/test_mechanics_logistics2.py).

A rig offset (x, y) is the tile (OX + bx + x, OY + by + y), (bx, by) the
rig's base; a 2x2 machine at (cx, cy) is centred on that tile corner.
"""

from __future__ import annotations

from fsim import HAND_Y_UNKNOWN, ITEM_IDS, ITEM_NAMES, STATUS_NAME, Sim, g, hand_position, lib

N, E, S, W = 0, 4, 8, 12
FACING = {"north": N, "east": E, "south": S, "west": W}


def p256(v: int):
    return v


class Rig:
    def __init__(self, name: str, base: tuple[int, int], w: int, h: int, ground: bool = False,
                 until: int | None = None) -> None:  # fmt: skip
        self.name, self.bx, self.by, self.w, self.h = name, base[0], base[1], w, h
        self.ground, self.until = ground, until
        self.belts: list[int] = []
        self.ins: list[int] = []
        self.chests: list[int] = []
        self.drills: list[int] = []
        self.furnaces: list[int] = []
        self.feed: list[list] = []


class Rigs2:
    def __init__(self, evidence: dict, names: set[str]) -> None:
        self.ox, self.oy = evidence["origin"]
        self.bases = evidence["bases"]
        self.events_t = evidence["events"]
        self.ticks = evidence["ticks"]
        self.names = names
        #: Item piles as the probe created them, read off each rig's first sample.
        self.piles0 = {k: v[0][1].get("g") or [] for k, v in evidence["series"].items()}
        self.rigs: dict[str, Rig] = {}
        self.events: list[tuple[int, str, dict]] = []
        self.ore: list[tuple[int, int]] = []
        self.plan: list = []
        self.rotations: dict = {}
        # Two passes: the first only collects the ore tiles the scene needs.
        self._collect = True
        self._build()
        self._collect = False
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
        self.rigs, self.events = {}, []
        self._build()
        lib.fsim_refresh(self.env)

    # ------------------------------------------------------------- building
    def rig(self, name: str, w: int, h: int, **kw) -> Rig | None:
        if name not in self.names:
            return None
        r = Rig(name, tuple(self.bases[name]), w, h, **kw)
        self.rigs[name] = r
        return r

    def tile(self, r: Rig, x: int, y: int) -> tuple[int, int]:
        return (self.ox + r.bx + x) * 256 + 128, (self.oy + r.by + y) * 256 + 128

    def _add(self, kind: int, x: int, y: int, direction: int = N) -> int:
        if self._collect:
            return -1
        index = lib.fsim_add_entity(self.env, kind, x, y, direction)
        assert index >= 0
        return index

    def belt(self, r: Rig, x: int, y: int, d: int) -> int:
        b = self._add(lib.K_BELT, *self.tile(r, x, y), d)
        r.belts.append(b)
        return b

    def chest(self, r: Rig, x: int, y: int, items=None) -> int:
        c = self._add(lib.K_CHEST, *self.tile(r, x, y))
        r.chests.append(c)
        for it in items or []:
            if self._collect:
                continue
            if len(it) > 2:
                st = self.env.entities[c].chest[it[2] - 1]
                st.item, st.count = ITEM_IDS[it[0]], it[1]
            else:
                assert lib.fsim_entity_insert(self.env, c, ITEM_IDS[it[0]], it[1]) == it[1]
        return c

    def inserter(self, r: Rig, x: int, y: int, d: int, coal: int = 0) -> int:
        e = self._add(lib.K_INSERTER, *self.tile(r, x, y), d)
        r.ins.append(e)
        if coal and not self._collect:
            lib.fsim_entity_insert(self.env, e, ITEM_IDS["coal"], coal)
        return e

    def ore_under(self, r: Rig, cx: int, cy: int) -> None:
        if self._collect:
            for dx in (-1, 0):
                for dy in (-1, 0):
                    self.ore.append((self.ox + r.bx + cx + dx, self.oy + r.by + cy + dy))

    def drill(self, r: Rig, cx: int, cy: int, d: int, coal: int, on_ore: bool = True) -> int:
        if on_ore:
            self.ore_under(r, cx, cy)
        e = self._add(lib.K_DRILL, (self.ox + r.bx + cx) * 256, (self.oy + r.by + cy) * 256, d)
        r.drills.append(e)
        if coal and not self._collect:
            lib.fsim_entity_insert(self.env, e, ITEM_IDS["coal"], coal)
        return e

    def furnace(self, r: Rig, cx: int, cy: int) -> int:
        e = self._add(lib.K_FURNACE, (self.ox + r.bx + cx) * 256, (self.oy + r.by + cy) * 256)
        r.furnaces.append(e)
        return e

    def piles(self, r: Rig) -> None:
        """The rig's piles at the positions the engine gave them."""
        for name, count, x, y in self.piles0.get(r.name, []):
            self.pile(x, y, name, count)

    def pile(self, x: int, y: int, name: str, count: int = 1) -> None:
        """An item-on-ground at map position (x, y), 1/256 tile."""
        if self._collect:
            return
        e = self._add(lib.K_PILE, x, y)
        self.env.entities[e].neutral = 1
        self.env.entities[e].pile.item = ITEM_IDS[name]
        self.env.entities[e].pile.count = count

    def put(self, b: int, lane: int, pos: int, name: str) -> None:
        if not self._collect:
            lib.fsim_belt_insert(self.env, b, lane - 1, pos, ITEM_IDS[name])

    def feeder(self, r: Rig, b: int, lane: int, count: int, name: str) -> None:
        r.feed.append([b, lane, count, name])

    def event(self, t: int, kind: str, **kw) -> None:
        self.events.append((t, kind, kw))

    def _build(self) -> None:  # noqa: C901 -- one block per probe rig family
        release, unblock = self.events_t["release"], self.events_t["unblock"]
        rot_ticks = self.events_t["rotation_ticks"]
        # turns
        if r := self.rig("turn_en_left", 5, 4):
            for i in range(4):
                self.belt(r, i, 3, E)
            self.belt(r, 4, 3, N)
            for i in (2, 1, 0):
                self.belt(r, 4, i, N)
        if r := self.rig("turn_ws_left", 5, 4):
            for i in (4, 3, 2, 1):
                self.belt(r, i, 0, W)
            self.belt(r, 0, 0, S)
            for i in (1, 2, 3):
                self.belt(r, 0, i, S)
        if r := self.rig("turn_ne_right", 4, 5):
            for i in (4, 3, 2, 1):
                self.belt(r, 0, i, N)
            self.belt(r, 0, 0, E)
            for i in (1, 2, 3):
                self.belt(r, i, 0, E)
        if r := self.rig("turn_es_right", 5, 4):
            for i in range(4):
                self.belt(r, i, 0, E)
            self.belt(r, 4, 0, S)
            for i in (1, 2, 3):
                self.belt(r, 4, i, S)
        for key in ("turn_en_left", "turn_ws_left", "turn_ne_right", "turn_es_right"):
            if r := self.rigs.get(key):
                self.feeder(r, r.belts[0], 1, 1, "iron-plate")
                self.feeder(r, r.belts[0], 2, 1, "iron-plate")
        # sideloads from the target's left
        for lane in (1, 2):
            for k in range(8):
                if r := self.rig(f"sl_left_{lane}_{k}", 3, 2):
                    for x in range(3):
                        self.belt(r, x, 1, E)
                    f = self.belt(r, 1, 0, S)
                    self.put(f, lane, 128 + k, "copper-plate")
        # sideload shapes
        if r := self.rig("side_turn_two", 3, 4):
            w = self.belt(r, 0, 0, E)
            self.belt(r, 1, 0, S)
            e = self.belt(r, 2, 0, W)
            for y in (1, 2, 3):
                self.belt(r, 1, y, S)
            for lane in (1, 2):
                self.feeder(r, w, lane, 3, "iron-plate")
                self.feeder(r, e, lane, 3, "copper-plate")
        if r := self.rig("side_turn_behind", 3, 5):
            w = self.belt(r, 0, 1, E)
            self.belt(r, 1, 1, S)
            n = self.belt(r, 1, 0, S)
            for y in (2, 3, 4):
                self.belt(r, 1, y, S)
            for lane in (1, 2):
                self.feeder(r, w, lane, 3, "iron-plate")
                self.feeder(r, n, lane, 3, "copper-plate")
        if r := self.rig("side_end", 1, 4):
            self.belt(r, 0, 0, E)
            for y in (1, 2, 3):
                self.belt(r, 0, y, N)
            for lane in (1, 2):
                self.feeder(r, r.belts[3], lane, 4, "copper-plate")
        if r := self.rig("side_start", 3, 4):
            for x in range(3):
                self.belt(r, x, 0, E)
            for y in (1, 2, 3):
                self.belt(r, 0, y, N)
            for lane in (1, 2):
                self.feeder(r, r.belts[5], lane, 4, "copper-plate")
        if r := self.rig("side_headon", 4, 1):
            for x, d in ((0, E), (1, E), (2, W), (3, W)):
                self.belt(r, x, 0, d)
            for lane in (1, 2):
                self.feeder(r, r.belts[0], lane, 3, "iron-plate")
                self.feeder(r, r.belts[3], lane, 3, "copper-plate")
        if r := self.rig("side_both", 3, 5):
            for x in range(3):
                self.belt(r, x, 2, E)
            self.belt(r, 1, 1, S)
            self.belt(r, 1, 0, S)
            self.belt(r, 1, 3, N)
            self.belt(r, 1, 4, N)
            for lane in (1, 2):
                self.feeder(r, r.belts[4], lane, 3, "iron-plate")
                self.feeder(r, r.belts[6], lane, 3, "copper-plate")
        # simultaneous sideloads (`sim_*`): main of three east belts; a feed of
        # F belts into the side of the middle one, index 1 next to the main;
        # items {lane, feed belt, position} in insertion order
        sims = {
            "sim_l1first": {"items": [(1, 1, 128), (2, 1, 128)]},
            "sim_l2first": {"items": [(2, 1, 128), (1, 1, 128)]},
            "sim_f4_l1first": {"F": 4, "items": [(1, 4, 128), (2, 4, 128)]},
            "sim_f4_l2first": {"F": 4, "items": [(2, 4, 128), (1, 4, 128)]},
            "sim_l1only": {"items": [(1, 1, 128)]},
            "sim_l2only": {"items": [(2, 1, 128)]},
            "sim_l1early": {"items": [(1, 1, 120), (2, 1, 128)]},
            "sim_l2early": {"items": [(1, 1, 128), (2, 1, 120)]},
            "sim_feedfirst_l1": {"feed_first": True, "items": [(1, 1, 128), (2, 1, 128)]},
            "sim_feedfirst_l2": {"feed_first": True, "items": [(2, 1, 128), (1, 1, 128)]},
            "sim_active_l1": {"items": [(1, 1, 128), (2, 1, 128)], "main_items": [(2, 1, 250)]},
            "sim_active_l2": {"items": [(2, 1, 128), (1, 1, 128)], "main_items": [(2, 1, 250)]},
            "sim_other_l1": {"items": [(1, 1, 128), (2, 1, 128)], "main_items": [(1, 1, 250)]},
            "sim_mainfirst_l1": {"items": [(1, 1, 128), (2, 1, 128)], "main_items": [(2, 3, 200)]},
            "sim_n_l1first": {"side": "N", "items": [(1, 1, 128), (2, 1, 128)]},
            "sim_n_l2first": {"side": "N", "items": [(2, 1, 128), (1, 1, 128)]},
            "sim_k3_l1first": {"items": [(1, 1, 131), (2, 1, 131)]},
            "sim_k3_l2first": {"items": [(2, 1, 131), (1, 1, 131)]},
            "sim_three_l1first": {"F": 4, "feeders": [(1, 3), (2, 3)]},
            "sim_three_l2first": {"F": 4, "feeders": [(2, 3), (1, 3)]},
            "sim_pairs_l1": {"F": 2, "items": [(1, 1, 128), (1, 1, 200), (2, 1, 128), (2, 1, 200)]},
            "sim_pairs_l2": {"F": 2, "items": [(2, 1, 128), (2, 1, 200), (1, 1, 128), (1, 1, 200)]},
        }  # fmt: skip
        for name, spec in sims.items():
            f = spec.get("F", 1)
            if not (r := self.rig(name, 3, f + 1)):
                continue
            south = spec.get("side", "S") == "S"
            my = 0 if south else f
            main: list[int] = []
            feed: list[int] = []

            def build_main(r=r, my=my, main=main):
                main.extend(self.belt(r, i, my, E) for i in range(3))

            def build_feed(r=r, my=my, feed=feed, f=f, south=south):
                feed.extend(self.belt(r, 1, my + k, N) if south else self.belt(r, 1, my - k, S)
                            for k in range(1, f + 1))  # fmt: skip

            if spec.get("feed_first"):
                build_feed()
                build_main()
            else:
                build_main()
                build_feed()
            for lane, k, pos in spec.get("main_items", []):
                self.put(main[k - 1], lane, pos, "iron-plate")
            for lane, k, pos in spec.get("items", []):
                self.put(feed[k - 1], lane, pos, "copper-plate")
            for lane, count in spec.get("feeders", []):
                self.feeder(r, feed[f - 1], lane, count, "copper-plate")
        # drops and drill outputs onto turns
        for tag, facing in (("r", S), ("l", N)):
            for side, size in (("n", (4, 5)), ("e", (4, 5)), ("s", (4, 5))):
                if r := self.rig(f"tdrop_{tag}_{side}", *size):
                    self.belt(r, 0, 2, E)
                    self.belt(r, 1, 2, facing)
                    x, y, d = {"n": (1, 1, N), "e": (2, 2, E), "s": (1, 3, S)}[side]
                    cx, cy = {"n": (1, 0), "e": (3, 2), "s": (1, 4)}[side]
                    self.chest(r, cx, cy, [("iron-plate", 20)])
                    self.inserter(r, x, y, d, 5)
            for side, size in (("n", (4, 5)), ("e", (5, 5)), ("s", (4, 6))):
                if r := self.rig(f"tdrill_{tag}_{side}", *size):
                    self.belt(r, 0, 2, E)
                    self.belt(r, 1, 2, facing)
                    cx, cy, d = {"n": (1, 1, S), "e": (3, 2, W), "s": (2, 4, N)}[side]
                    self.drill(r, cx, cy, d, 5)
        # ground
        if r := self.rig("ground_drop", 1, 4, ground=True):
            self.chest(r, 0, 0, [("iron-plate", 20)])
            self.inserter(r, 0, 1, N, 5)
        if r := self.rig("ground_drop_taken", 1, 4, ground=True):
            self.chest(r, 0, 0, [("iron-plate", 20)])
            self.inserter(r, 0, 1, N, 5)
            self.piles(r)
        for key in ("center", "off", "edge", "outside", "multi", "stack"):
            if r := self.rig(f"ground_pick_{key}", 1, 4, ground=True):
                self.piles(r)
                self.inserter(r, 0, 1, N, 5)
                self.chest(r, 0, 2)
        # full chests
        stone = [("stone", 50, i) for i in range(1, 17)]
        if r := self.rig("full_stone", 1, 3):
            self.chest(r, 0, 0, [("iron-plate", 20)])
            self.inserter(r, 0, 1, N, 5)
            self.chest(r, 0, 2, stone)
            self.event(release, "clear_slot", rig=r, chest=1, slot=16)
        room = [("stone", 50, i) for i in range(1, 16)] + [("iron-plate", 99, 16)]
        if r := self.rig("full_room_one", 1, 3):
            self.chest(r, 0, 0, [("iron-plate", 20)])
            self.inserter(r, 0, 1, N, 5)
            self.chest(r, 0, 2, room)
            self.event(release, "take", rig=r, chest=1, name="iron-plate", count=1)
        if r := self.rig("full_plates", 1, 3):
            self.chest(r, 0, 0, [("iron-plate", 20)])
            self.inserter(r, 0, 1, N, 5)
            self.chest(r, 0, 2, [("iron-plate", 100, i) for i in range(1, 17)])
            self.event(release, "take", rig=r, chest=1, name="iron-plate", count=1)
        # fill limits
        for name, item, count, target in (
            ("fill_coal_drill", "coal", 50, "drill"),
            ("fill_coal_drill_ore", "coal", 50, "drill_ore"),
            ("fill_coal_ins", "coal", 50, "inserter"),
            ("fill_wood_furnace", "wood", 50, "furnace"),
            ("fill_wood_drill", "wood", 50, "drill"),
            ("fill_wood_ins", "wood", 50, "inserter"),
            ("fill_coal_chest", "coal", 20, "chest"),
            ("fill_ore_drill", "iron-ore", 50, "drill_ore"),
            ("fill_copper_furnace", "copper-ore", 50, "furnace"),
            ("fill_stone_furnace", "stone", 50, "furnace"),
            ("fill_plate_furnace", "iron-plate", 50, "furnace"),
            ("fill_coal_hot_furnace", "coal", 50, "hot_furnace"),
        ):
            if r := self.rig(name, 3, 4):
                self.chest(r, 0, 0, [(item, count)])
                self.inserter(r, 0, 1, N, 5)
                if target == "drill":
                    self.drill(r, 1, 3, N, 0, on_ore=False)
                elif target == "drill_ore":
                    self.drill(r, 1, 3, N, 0)
                elif target in ("furnace", "hot_furnace"):
                    f = self.furnace(r, 1, 3)
                    if target == "hot_furnace" and not self._collect:
                        lib.fsim_entity_insert(self.env, f, ITEM_IDS["iron-ore"], 30)
                elif target == "inserter":
                    self.inserter(r, 0, 2, E, 0)
                else:
                    self.chest(r, 0, 2)
        # mixed chests
        for name, items, target, coal in (
            ("mix_plate_ore_coal_chest", [("iron-plate", 5, 1), ("iron-ore", 5, 2),
                                          ("coal", 5, 3)], "chest", 5),
            ("mix_ore_plate_chest", [("iron-ore", 5, 1), ("iron-plate", 5, 2)], "chest", 5),
            ("mix_plate_ore_furnace", [("iron-plate", 10, 1), ("iron-ore", 10, 2)], "furnace",
             5),
            ("mix_coal_ore_furnace", [("coal", 10, 1), ("iron-ore", 10, 2)], "furnace", 5),
            ("mix_ore_coal_furnace", [("iron-ore", 10, 1), ("coal", 10, 2)], "furnace", 5),
            ("mix_gaps_chest", [("iron-ore", 5, 3), ("iron-plate", 5, 7)], "chest", 5),
            ("mix_plate_ore_belt", [("iron-plate", 5, 1), ("iron-ore", 5, 2)], "belt", 5),
            ("mix_stone_coal_drill", [("stone", 5, 1), ("coal", 5, 2)], "drill", 5),
            ("mix_plate_coal_self", [("iron-plate", 5, 1), ("coal", 5, 2)], "chest", 0),
            ("mix_split_chest", [("iron-ore", 3, 1), ("iron-plate", 3, 2), ("iron-ore", 3, 3)],
             "chest", 5),
        ):  # fmt: skip
            if r := self.rig(name, 3, 4):
                self.chest(r, 0, 0, items)
                self.inserter(r, 0, 1, N, coal)
                if target == "chest":
                    self.chest(r, 0, 2)
                elif target == "furnace":
                    self.furnace(r, 1, 3)
                elif target == "drill":
                    self.drill(r, 1, 3, N, 0, on_ore=False)
                else:
                    for x in range(3):
                        self.belt(r, x, 2, E)
        # update order
        for name, first in (("order_pick_ab", "a"), ("order_pick_ba", "b")):
            if r := self.rig(name, 3, 5):
                self.chest(r, 1, 2, [("iron-plate", 1)])
                self.chest(r, 1, 0)
                self.chest(r, 1, 4)
                steps = [lambda r=r: self.inserter(r, 1, 1, S, 5),
                         lambda r=r: self.inserter(r, 1, 3, N, 5)]  # fmt: skip
                for step in steps if first == "a" else steps[::-1]:
                    step()
        spots = {"n": (2, 1, S, 2, 0), "s": (2, 3, N, 2, 4), "e": (3, 2, W, 4, 2),
                 "w": (1, 2, E, 0, 2)}  # fmt: skip
        for name, seq, plates in (
            ("order4_nesw_1", "nesw", 1), ("order4_wsen_1", "wsen", 1),
            ("order4_ewns_1", "ewns", 1), ("order4_nesw_3", "nesw", 3),
            ("order4_wsen_3", "wsen", 3), ("order4_ewns_3", "ewns", 3),
        ):  # fmt: skip
            if r := self.rig(name, 5, 5):
                self.chest(r, 2, 2, [("iron-plate", plates)])
                for key in seq:
                    sp = spots[key]
                    self.chest(r, sp[3], sp[4])
                    self.inserter(r, sp[0], sp[1], sp[2], 5)
        for name, first in (("order_drop_ab", "a"), ("order_drop_ba", "b")):
            if r := self.rig(name, 3, 5):
                self.chest(r, 1, 2, room)
                self.chest(r, 1, 0, [("iron-plate", 10)])
                self.chest(r, 1, 4, [("iron-plate", 10)])
                steps = [lambda r=r: self.inserter(r, 1, 1, N, 5),
                         lambda r=r: self.inserter(r, 1, 3, S, 5)]  # fmt: skip
                for step in steps if first == "a" else steps[::-1]:
                    step()
        if r := self.rig("order_furnace3", 4, 5):
            self.furnace(r, 2, 2)
            self.chest(r, 1, -1, [("iron-ore", 10)])
            self.inserter(r, 1, 0, N, 5)
            self.chest(r, 2, -1, [("iron-ore", 10)])
            self.inserter(r, 2, 0, N, 5)
            self.chest(r, -1, 1, [("iron-ore", 10)])
            self.inserter(r, 0, 1, W, 5)

        # rotations and shape changes of loaded belts
        def load(t: int, lengths: tuple[int, int], j: int) -> None:
            for lane, name in ((1, "iron-plate"), (2, "copper-plate")):
                p = j
                while p < lengths[lane - 1]:
                    self.put(t, lane, p, name)
                    p += 64

        def lanes_of(t: int) -> list:
            b = self.env.entities[t]
            return [[[ITEM_NAMES[it.item], it.pos] for it in b.lanes[k].items[0 : b.lanes[k].count]]
                    for k in (0, 1)]  # fmt: skip

        def lengths(t: int) -> tuple[int, int]:
            lib.fsim_refresh(self.env)
            b = self.env.entities[t]
            return b.lane_length[0], b.lane_length[1]

        families = [("rot_r2s_ccw", S, range(64), "ccw"), ("rot_s2r_cw", E, range(64), "cw"),
                    ("rot_r2s_cw", S, range(0, 64, 4), "cw"),
                    ("rot_s2l_ccw", E, range(0, 64, 4), "ccw"),
                    ("rot_l2s_cw", N, range(0, 64, 4), "cw"),
                    ("rot_r_mine", S, range(0, 64, 4), "mine_feeder"),
                    ("rot_r_behind", S, range(0, 64, 4), "add_behind")]  # fmt: skip
        for fam, facing, js, how in families:
            for j in js:
                if not (r := self.rig(f"{fam}_{j}", 2, 1, until=rot_ticks)):
                    continue
                w = self.belt(r, 0, 0, E)
                t = self.belt(r, 1, 0, facing)
                if self._collect:
                    continue
                load(t, lengths(t), j)
                before = lanes_of(t)
                if how in ("cw", "ccw"):
                    self.rotate(t, how == "ccw")
                elif how == "mine_feeder":
                    self.destroy(w)
                else:
                    self.belt(r, 1, -1, facing)
                lib.fsim_refresh(self.env)
                self.rotations[r.name] = {"before": before, "after": lanes_of(t)}
        for j in range(0, 64, 4):
            for how in ("cw", "ccw"):
                if not (r := self.rig(f"rot_ss_{how}_{j}", 1, 1, until=rot_ticks)):
                    continue
                t = self.belt(r, 0, 0, E)
                if self._collect:
                    continue
                load(t, lengths(t), j)
                before = lanes_of(t)
                self.rotate(t, how == "ccw")
                lib.fsim_refresh(self.env)
                self.rotations[r.name] = {"before": before, "after": lanes_of(t)}
        for j in range(0, 64, 4):
            if not (r := self.rig(f"rot_s2r_side_{j}", 2, 1, until=rot_ticks)):
                continue
            t = self.belt(r, 1, 0, S)
            if not self._collect:
                load(t, lengths(t), j)
                before = lanes_of(t)
            w = self.belt(r, 0, 0, E)
            r.belts = [t, w]
            if not self._collect:
                lib.fsim_refresh(self.env)
                self.rotations[r.name] = {"before": before, "after": lanes_of(t)}
        # a waiting inserter whose drop spot is cleared
        if r := self.rig("ground_clear", 1, 4, ground=True):
            self.chest(r, 0, 0, [("iron-plate", 20)])
            self.inserter(r, 0, 1, N, 5)
            self.piles(r)
            self.event(400, "clear_ground", rig=r)
        # two piles on one pickup tile
        for fc, d in (("n", N), ("e", E), ("s", S), ("w", W)):
            for pr, _a, _b in (("x", (77, 0), (-77, 0)), ("y", (0, 77), (0, -77)),
                             ("d", (77, 77), (-77, -77)), ("xr", (-77, 0), (77, 0))):  # fmt: skip
                if r := self.rig(f"gpair_{fc}_{pr}", 3, 5, ground=True):
                    ix, iy = {"n": (1, 3), "s": (1, 1), "e": (0, 2), "w": (2, 2)}[fc]
                    self.piles(r)
                    self.inserter(r, ix, iy, d, 5)
                    self.chest(r, ix + (ix - 1), iy + (iy - 2))
        if r := self.rig("wake3", 5, 5):
            self.chest(r, 2, 2)
            self.chest(r, 2, 0)
            self.inserter(r, 2, 1, S, 5)
            self.chest(r, 4, 2)
            self.inserter(r, 3, 2, W, 5)
            self.chest(r, 2, 4)
            self.inserter(r, 2, 3, N, 5)
            for t in (100, 300, 500):
                self.event(t, "put", rig=r, chest=0, name="iron-plate", count=1)
        for first in ("x", "y"):
            if r := self.rig(f"chain_{first}", 1, 5):
                self.chest(r, 0, 0, [("iron-plate", 5)])
                self.chest(r, 0, 2)
                self.chest(r, 0, 4)
                steps = [lambda r=r: self.inserter(r, 0, 1, N, 5),
                         lambda r=r: self.inserter(r, 0, 3, N, 5)]  # fmt: skip
                for step in steps if first == "x" else steps[::-1]:
                    step()
        for name, t in (("woken_vs_active", 84), ("woken_early", 83)):
            if r := self.rig(name, 3, 5):
                self.chest(r, 1, 2, [("iron-plate", 1)])
                self.chest(r, 1, 4)
                self.inserter(r, 1, 3, N, 5)
                self.chest(r, 1, 0)
                self.inserter(r, 1, 1, S, 5)
                self.event(t, "put", rig=r, chest=0, name="iron-plate", count=1)
        # status corners
        if r := self.rig("stat_fueled_furnace", 3, 4):
            self.chest(r, 0, 0, [("coal", 50)])
            self.inserter(r, 0, 1, N, 5)
            f = self.furnace(r, 1, 3)
            if not self._collect:
                lib.fsim_entity_insert(self.env, f, ITEM_IDS["coal"], 2)
        if r := self.rig("stat_busy_ins", 3, 4):
            self.chest(r, 0, 0, [("coal", 50)])
            self.inserter(r, 0, 1, N, 5)
            self.chest(r, 1, 2, [("iron-plate", 50)])
            self.inserter(r, 0, 2, E, 0)
            self.chest(r, -1, 2)
        part = [("stone", 49, i) for i in range(1, 17)]
        for name, src in (("stat_partial_stone", [("iron-plate", 20)]),
                          ("stat_partial_stone_src",
                           [("iron-plate", 20, 1), ("stone", 5, 2)]),
                          ("stat_room_stone",
                           [("stone", 5, 1), ("iron-plate", 20, 2)])):  # fmt: skip
            if r := self.rig(name, 1, 3):
                self.chest(r, 0, 0, src)
                self.inserter(r, 0, 1, N, 5)
                self.chest(r, 0, 2, part)
        # pickups from items stopped on a straight belt end and on turns
        for side in ("n", "s", "e", "w"):
            if r := self.rig(f"spick_{side}", 5, 5):
                b = self.belt(r, 2, 2, E)
                for k, name in enumerate(("iron-plate", "copper-plate", "iron-ore", "copper-ore")):
                    self.put(b, 1, 64 * k, name)
                for k, name in enumerate(("coal", "wood", "stone", "iron-gear-wheel")):
                    self.put(b, 2, 64 * k, name)
                cx, cy, ix, iy, d = {"n": (2, 0, 2, 1, S), "s": (2, 4, 2, 3, N),
                                     "e": (4, 2, 3, 2, W), "w": (0, 2, 1, 2, E)}[side]  # fmt: skip
                self.chest(r, cx, cy)
                self.inserter(r, ix, iy, d, 5)
        for tag, facing in (("r", S), ("l", N)):
            for side in ("n", "e", "s"):
                if not (r := self.rig(f"tpick_{tag}_{side}", 4, 5)):
                    continue
                self.belt(r, 0, 2, E)
                t = self.belt(r, 1, 2, facing)
                if not self._collect:
                    lib.fsim_refresh(self.env)
                outer = 1 if tag == "r" else 2
                for k, name in enumerate(("iron-plate", "copper-plate", "iron-ore", "copper-ore",
                                          "stone")):  # fmt: skip
                    self.put(t, outer, 64 * k, name)
                self.put(t, 3 - outer, 0, "coal")
                self.put(t, 3 - outer, 64, "wood")
                cx, cy, ix, iy, d = {"n": (1, 0, 1, 1, S), "e": (3, 2, 2, 2, W),
                                     "s": (1, 4, 1, 3, N)}[side]  # fmt: skip
                self.chest(r, cx, cy)
                self.inserter(r, ix, iy, d, 5)
        # energy running out mid-swing, and the redirect to its own fuel slot
        for i in range(24):
            if r := self.rig(f"pe_{i}", 1, 3):
                self.chest(r, 0, 0, [("iron-plate", 50)])
                e = self.inserter(r, 0, 1, N, 0)
                self.chest(r, 0, 2)
                if not self._collect:
                    self.env.entities[e].remaining = 40000 + 2800 * i
                self.event(self.events_t["refuel"], "fuel", rig=r, ins=0, name="coal", count=1)
        for i in range(24):
            if r := self.rig(f"sr_{i}", 1, 3):
                self.chest(r, 0, 0, [("coal", 50)])
                e = self.inserter(r, 0, 1, N, 1)
                self.chest(r, 0, 2)
                if not self._collect:
                    self.env.entities[e].remaining = 30000 + 2800 * i

        # belt line segments and the window an asleep inserter watches
        def seg_rig(name, n_east, turn, n_south, k, reverse):
            if not (r := self.rig(name, n_east + 2, n_south + 4)):
                return
            cells = [(i, 0, E) for i in range(n_east)]
            if turn:
                cells.append((n_east, 0, S))
            cells += [(n_east, j, S) for j in range(1, n_south + 1)]
            made = {}
            for cell in reversed(cells) if reverse else cells:
                made[cell] = self.belt(r, *cell)
            r.belts = [made[cell] for cell in cells]
            last = cells[-1]
            if turn or n_south > 0:
                self.chest(r, last[0], last[1] + 2)
                self.inserter(r, last[0], last[1] + 1, N, 5)
            else:
                self.chest(r, last[0], 2)
                self.inserter(r, last[0], 1, N, 5)
            self.event(400, "put_belt", rig=r, belt=k, lane=1, pos=128)

        for k in range(9):
            seg_rig(f"seg_a_{k}", 5, True, 3, k, False)
            seg_rig(f"seg_ar_{k}", 5, True, 3, k, True)
            seg_rig(f"seg_d_{k}", 3, True, 5, k, False)
        for k in range(8):
            seg_rig(f"seg_b_{k}", 4, True, 3, k, False)
        for k in range(10):
            seg_rig(f"seg_c_{k}", 10, False, 0, k, False)

        def win_rig(name, n_east, turn, n_south, lane, across):
            if not (r := self.rig(name, n_east + 2, n_south + 4)):
                return
            for i in range(n_east):
                self.belt(r, i, 0, E)
            if turn:
                self.belt(r, n_east, 0, S)
            for j in range(1, n_south + 1):
                self.belt(r, n_east, j, S)
            if turn:
                self.chest(r, n_east, n_south + 2)
                self.inserter(r, n_east, n_south + 1, N, 5)
            elif across:
                self.chest(r, n_east - 2, 2)
                self.inserter(r, n_east - 2, 1, N, 5)
            else:
                self.chest(r, n_east + 1, 0)
                self.inserter(r, n_east, 0, W, 5)
            self.event(400, "hold", rig=r, ins=0)
            self.event(400, "put_belt", rig=r, belt=0, lane=lane, pos=255)

        win_rig("win_s_near", 20, False, 0, 2, True)
        win_rig("win_s_far", 20, False, 0, 1, True)
        win_rig("win_s_end", 20, False, 0, 1, False)
        win_rig("win_t_1", 10, True, 3, 1, False)
        win_rig("win_t_2", 10, True, 3, 2, False)
        # drill status after a change
        if r := self.rig("dstat_extend", 3, 3):
            self.drill(r, 1, 1, S, 5)
            self.belt(r, 1, 2, E)
            self.event(unblock, "belt", rig=r, x=2, y=2, d=E)
        if r := self.rig("dstat_rotate", 3, 3):
            self.drill(r, 1, 1, S, 5)
            self.belt(r, 1, 2, E)
            self.event(unblock, "rotate", rig=r, belt=0)
        if r := self.rig("dstat_unrelated", 5, 3):
            self.drill(r, 1, 1, S, 5)
            self.belt(r, 1, 2, E)
            self.event(unblock, "chest", rig=r, x=4, y=2)
        if r := self.rig("dstat_chest", 3, 3):
            self.drill(r, 1, 1, S, 5)
            self.chest(r, 1, 2, stone)
            self.event(unblock, "clear_slot", rig=r, chest=0, slot=16)
        if r := self.rig("dstat_ground", 3, 3, ground=True):
            self.drill(r, 1, 1, S, 5)
            self.event(unblock, "clear_ground", rig=r)

    # ------------------------------------------------------------- scripting
    def rotate(self, t: int, reverse: bool) -> None:
        e = self.env.entities[t]
        e.direction = (e.direction + (12 if reverse else 4)) % 16
        self.env.entities_version += 1

    def destroy(self, index: int) -> None:
        self.env.entities[index].alive = 0
        self.env.entities_version += 1

    def _in_area(self, r: Rig, x: int, y: int) -> bool:
        x0, y0 = (self.ox + r.bx - 1) * 256, (self.oy + r.by - 1) * 256
        x1, y1 = (self.ox + r.bx + r.w + 1) * 256, (self.oy + r.by + r.h + 1) * 256
        return x0 <= x <= x1 and y0 <= y <= y1

    def _run_events(self, t: int) -> None:
        env = self.env
        for when, kind, kw in self.events:
            if when != t:
                continue
            r = kw.get("rig")
            if kind == "clear_slot":
                c = r.chests[kw["chest"]]
                st = env.entities[c].chest[kw["slot"] - 1]
                st.item, st.count = 0, 0
                lib.fsim_script_touched(env, c)
            elif kind == "take":
                c = r.chests[kw["chest"]]
                lib.fsim_entity_remove(env, c, ITEM_IDS[kw["name"]], kw["count"])
            elif kind == "fuel":
                lib.fsim_entity_insert(env, r.ins[kw["ins"]], ITEM_IDS[kw["name"]], kw["count"])
            elif kind == "hold":
                lib.fsim_inserter_hold(env, r.ins[kw["ins"]], ITEM_IDS["iron-plate"])
            elif kind == "put_belt":
                lib.fsim_belt_insert(env, r.belts[kw["belt"]], kw["lane"] - 1, kw["pos"],
                                     ITEM_IDS["iron-plate"])  # fmt: skip
            elif kind == "put":
                lib.fsim_entity_insert(env, r.chests[kw["chest"]], ITEM_IDS[kw["name"]],
                                       kw["count"])  # fmt: skip
            elif kind == "belt":
                self.belt(r, kw["x"], kw["y"], kw["d"])
            elif kind == "rotate":
                self.rotate(r.belts[kw["belt"]], False)
            elif kind == "chest":
                self.chest(r, kw["x"], kw["y"])
            elif kind == "clear_ground":
                for i in range(env.entity_count):
                    e = env.entities[i]
                    if e.alive and e.kind == lib.K_PILE and self._in_area(r, e.pos.x, e.pos.y):
                        lib.fsim_remove_pile(env, i)

    def _feed(self) -> None:
        for r in self.rigs.values():
            for fd in r.feed:
                b, lane, left, name = fd
                if left > 0 and lib.fsim_belt_insert_back(self.env, b, lane - 1, ITEM_IDS[name]):
                    fd[2] -= 1

    def samples(self):
        """Every tick's reading, t = 0..ticks: {rig: record}."""
        self._feed()
        lib.fsim_refresh(self.env)
        yield self.sample(0)
        for t in range(1, self.ticks + 1):
            lib.fsim_advance(self.env, 1)
            self._run_events(t)
            self._feed()
            lib.fsim_refresh(self.env)
            yield self.sample(t)

    # ------------------------------------------------------------- reading
    def _inv(self, stacks) -> list:
        return [[i + 1, ITEM_NAMES[s.item], s.count] for i, s in enumerate(stacks) if s.count > 0]

    def _belt(self, index: int):
        b = self.env.entities[index]
        if not b.alive:
            return "gone"
        shape = {lib.BELT_STRAIGHT: "straight", lib.BELT_LEFT: "left", lib.BELT_RIGHT: "right"}
        return {
            "sh": shape[b.shape],
            "d": b.direction,
            "l": [[[ITEM_NAMES[it.item], it.pos] for it in b.lanes[k].items[0 : b.lanes[k].count]]
                  for k in (0, 1)],
        }  # fmt: skip

    def _ins(self, index: int):
        s = self.env.entities[index]
        if not s.alive:
            return "gone"
        pickup, drop = self.sim._inserter_points(s)
        out = {
            "hp": hand_position(s, pickup, drop),
            "e": g(s.energy),
            "r": g(s.remaining),
            "f": self._inv([s.fuel]),
            "st": STATUS_NAME[s.status],
        }
        if s.held:
            out["h"], out["hc"] = ITEM_NAMES[s.held], 1
        if s.burning:
            out["bu"] = ITEM_NAMES[s.burning]
        if s.lift < 0:
            out[HAND_Y_UNKNOWN] = True
        return out

    def _drill(self, index: int):
        d = self.env.entities[index]
        return {"st": STATUS_NAME[d.status], "pr": g(d.progress), "e": g(d.energy),
                "f": self._inv([d.fuel])}  # fmt: skip

    def _furnace(self, index: int):
        f = self.env.entities[index]
        return {"src": self._inv([f.source]), "f": self._inv([f.fuel]),
                "res": self._inv([f.result]), "st": STATUS_NAME[f.status],
                "pr": g(f.progress)}  # fmt: skip

    def _ground(self, r: Rig) -> list:
        out = []
        for i in range(self.env.entity_count):
            e = self.env.entities[i]
            if e.alive and e.kind == lib.K_PILE and self._in_area(r, e.pos.x, e.pos.y):
                out.append([ITEM_NAMES[e.pile.item], e.pile.count, e.pos.x, e.pos.y])
        out.sort(key=lambda p: (str(p[3]), str(p[2]), p[0]))
        return out

    def sample(self, t: int) -> dict:
        out = {}
        for name, r in self.rigs.items():
            if r.until is not None and t > r.until:
                continue
            rec: dict = {}
            if r.belts:
                rec["b"] = [self._belt(b) for b in r.belts]
            if r.ins:
                rec["i"] = [self._ins(i) for i in r.ins]
            if r.chests:
                rec["c"] = [self._inv(self.env.entities[c].chest) for c in r.chests]
            if r.drills:
                rec["d"] = [self._drill(d) for d in r.drills]
            if r.furnaces:
                rec["f"] = [self._furnace(f) for f in r.furnaces]
            if r.ground:
                rec["g"] = self._ground(r)
            out[name] = rec
        return out


__all__ = ["Rigs2", "FACING", "p256", "Sim"]
