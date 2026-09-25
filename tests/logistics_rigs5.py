"""Replay FactorioRL's fifth logistics probe (tools/probe_logistics5.py) rigs.

A rig there is data: a base tile and a list of timed operations (build a belt,
chest or burner inserter, put an item on a belt, rotate or destroy an entity),
which the probe ran in Factorio 2.0.60 and this module runs in the simulator.
Each rig runs alone in a fresh world, at the probe's own tiles (the belt merge
delay depends on the tile).

A reading, per tick, is the probe's record without the engine's item ids:

- `b`: per belt (in build order), "gone" or its two lanes, each a sorted list
  of [position, item name];
- `s`: per belt and lane, the segment it is in: the index (belt * 2 + lane,
  lane 0 or 1) of the first belt lane of the rig in the same segment, -1 for a
  belt that is gone;
- `i`: per inserter, the item in its hand or "";
- `c`: per chest, its non-empty slots as [slot, name, count].

Operations at tick t happen after the world has run t ticks and before the
reading at t, as the probe's on_tick hook does them.
"""

from __future__ import annotations

from fsim import ITEM_IDS, Sim, lib

NAMES = {v: k for k, v in ITEM_IDS.items()}


class Rig5:
    def __init__(self, rig: dict) -> None:
        self.rig = rig
        self.sim = Sim(water=[])
        self.sim.reset({"character": {"position": [0.5, 0.5]}})
        self.env = self.sim.env
        self.bx, self.by = rig["base"]
        self.labels: dict[str, int] = {}
        self.belts: list[int] = []
        self.ins: list[int] = []
        self.chests: list[int] = []
        self.ops: dict[int, list] = {}
        for op in rig["ops"]:
            self.ops.setdefault(op[0], []).append(op[1:])

    # ------------------------------------------------------------ operations
    def _add(self, kind: int, dx: int, dy: int, d: int) -> int:
        x, y = self.bx + dx, self.by + dy
        index = lib.fsim_add_entity(self.env, kind, x * 256 + 128, y * 256 + 128, d)
        assert index >= 0, (self.rig["name"], kind, x, y)
        assert self.env.belt_delay_missing == 0, (x, y)
        return index

    def _apply(self, op: list) -> None:
        kind, label, *args = op
        env = self.env
        if kind == "belt":
            dx, dy, d = args
            self.labels[label] = self._add(lib.K_BELT, dx, dy, d)
            self.belts.append(self.labels[label])
        elif kind == "chest":
            dx, dy, item, count = args
            c = self._add(lib.K_CHEST, dx, dy, 0)
            if item:
                assert lib.fsim_entity_insert(env, c, ITEM_IDS[item], count) == count
            self.labels[label] = c
            self.chests.append(c)
        elif kind == "ins":
            dx, dy, d, coal = args
            i = self._add(lib.K_INSERTER, dx, dy, d)
            if coal:
                lib.fsim_entity_insert(env, i, ITEM_IDS["coal"], coal)
            self.labels[label] = i
            self.ins.append(i)
        elif kind == "put":
            lane, pos, item = args
            lib.fsim_refresh(env)
            # a refused insert shows in the readings, as in the probe
            lib.fsim_belt_insert(env, self.labels[label], lane - 1, pos, ITEM_IDS[item])
        elif kind == "rotate":
            (reverse,) = args
            assert lib.fsim_script_rotate(env, self.labels[label], 1 if reverse else 0)
        elif kind == "destroy":
            assert lib.fsim_script_destroy(env, self.labels[label])
        else:
            raise ValueError(op)

    # ------------------------------------------------------------ reading
    def reading(self) -> dict:
        env = self.env
        lib.fsim_refresh(env)
        b, keys = [], []
        for x in self.belts:
            e = env.entities[x]
            if not e.alive:
                b.append("gone")
                keys += [None, None]
                continue
            b.append([sorted([e.lanes[k].items[j].pos, NAMES[e.lanes[k].items[j].item]]
                             for j in range(e.lanes[k].count)) for k in (0, 1)])  # fmt: skip
            keys += [lib.fsim_belt_segment(env, x, lane) for lane in (0, 1)]
        s = [-1 if key is None else keys.index(key) for key in keys]
        i = [NAMES[env.entities[x].held] if env.entities[x].held else "" for x in self.ins]
        c = [[[k + 1, NAMES[st.item], st.count] for k, st in enumerate(env.entities[x].chest)
              if st.count] for x in self.chests]  # fmt: skip
        return {"b": b, "s": s, "i": i, "c": c}

    def run(self, ticks: int | None = None):
        """Every tick's reading, t = 0..ticks."""
        last = self.rig["ticks"] if ticks is None else ticks
        for t in range(last + 1):
            if t:
                lib.fsim_advance(self.env, 1)
            for op in self.ops.get(t, []):
                self._apply(op)
            yield self.reading()


def engine_reading(state: dict) -> dict:
    """The probe's record of one tick, ids dropped, as `Rig5.reading`."""
    b = []
    for belt in state["b"]:
        if belt == "gone":
            b.append("gone")
            continue
        b.append([sorted([p, name] for p, _id, name in (lane if isinstance(lane, list) else []))
                  for lane in belt])  # fmt: skip
    s = list(state["s"]) if isinstance(state["s"], list) else []
    i = list(state["i"]) if isinstance(state["i"], list) else []
    c = [[list(x) for x in (ch if isinstance(ch, list) else [])]
         for ch in (state["c"] if isinstance(state["c"], list) else [])]  # fmt: skip
    return {"b": b, "s": s, "i": i, "c": c}
