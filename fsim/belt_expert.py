"""A scripted builder for belt_smelting, on the v3 action space.

A port of FactorioRL's evaluator-only reference for `belt_smelting` 1.1.0
(`tasks/families/belt_smelting.py`: `plan_line`, `fuel_plan`, `_Builder`,
`solve`) that speaks v3 MultiDiscrete vectors, the same action space a policy
uses. It follows the reference step for step:

1. walk to the coal patch and hand-mine `EXTRA_COAL` coal (`mine_tile`);
2. split the coal it then holds with `fuel_plan`;
3. build the planned cell on the iron patch -- two drills facing into two
   furnaces, an inserter under each furnace -- and fuel the inserters and the
   furnaces;
4. lay the belt from the cell to the chest and build the chest inserter, and
   fuel it;
5. walk back and fuel the drills with everything left, then `finish`.

The layout is `plan_line`'s: every cell position and facing whose two drills
sit wholly on iron ore, every side of the chest for the chest inserter, the
belt routed by breadth-first search around walls, and the layout using the
fewest belts kept. What the reference does not know about, the map's water,
is an obstacle here like a wall: the simulator has it, and a route across it
could not be built.

The builder reads the scene from the blueprint (evaluator knowledge a
reference may use) and the live state from `rl.env` between decisions. It
never steps the environment itself: `next_vector()` returns one decision and
the caller steps it, so a batched environment can drive it from a raw
`fsim_rl *` (`fsim.vec`), and demonstrations record exactly what was taken.

Walking is not the reference's greedy axis walk. It is Dijkstra over tiles
(walls, water and built machines block; placed belts cost `BELT_WALK_COST`),
then long, step and nudge strides along the route's straight runs, re-planned
whenever a stride does not move the character. Where the engine's walk ends
inside a tile differs from this one, so a run here and a run there take
different numbers of decisions for the same build.
"""

from __future__ import annotations

import heapq
import math
import random
from collections import deque
from dataclasses import dataclass, field

from fsim import ffi, lib

TILE = 256
SCENE_HALF = 60
BELTS = 40
TARGET_PLATES = 150

# ------------------------------------------------------------------ v3 vector
#: parameterized-v3 operation indices (csrc/fsim_rl.c).
OP_PLACE, OP_GIVE, OP_WAIT, OP_MINE_TILE, OP_FINISH = 12, 16, 21, 22, 24
MOVE_LONG, MOVE_STEP, MOVE_NUDGE = 0, 4, 8  # plus 0 N, 1 E, 2 S, 3 W
#: Tiles each stride covers, from the character's 38/256 tiles a tick.
STRIDE_TILES = {MOVE_LONG: 30 * 38 / 256, MOVE_STEP: 7 * 38 / 256, MOVE_NUDGE: 2 * 38 / 256}
#: `ITEMS_V3` index + 1.
ITEM_ARG = {
    "coal": 3,
    "stone-furnace": 7,
    "transport-belt": 9,
    "burner-mining-drill": 13,
    "burner-inserter": 14,
}
DIRECTION_ARG = {"north": 1, "east": 2, "south": 3, "west": 4}
AMOUNT_ARG = {1: 1, 5: 2, 20: 3}
KIND = {
    "burner-mining-drill": lib.K_DRILL,
    "stone-furnace": lib.K_FURNACE,
    "transport-belt": lib.K_BELT,
    "burner-inserter": lib.K_INSERTER,
}
PLACEMENT_RADIUS = 7  # RL3_PLACEMENT_RADIUS
PLACEMENT_SIDE = 2 * PLACEMENT_RADIUS + 1
MAX_TARGETS = 96  # RL3_TARGETS
RESOURCE_REACH = 2.7
WAIT = [OP_WAIT, 0, 0, 0, 0, 0]
FINISH = [OP_FINISH, 0, 0, 0, 0, 0]

# ------------------------------------------------------------ the reference
#: FactorioRL belt_smelting 1.1.0's constants, restated.
EXTRA_COAL = 40
COAL_J = 4_000_000
DRILL_J_PER_ORE = 600_000
FURNACE_J_PER_PLATE = 288_000
INSERTER_J_PER_SWING = 66_900
INSERTER_BUILT_J = 500_000
INSERTER_J_PER_BELT_PICKUP = 150_000
FUEL_MARGIN = 1.1
CELL_PLATE_CEILING = 280
STAND_TOLERANCE = 0.25
BUILD_REACH = 8.5
BELT_WALK_COST = 60
WALK_ATTEMPTS = 5
FUEL_REACH = 6.0
#: Strides one straight run of a walk may take before it is re-planned.
RUN_BUDGET = 80

Tile = tuple[int, int]
Rect = tuple[int, int, int, int]

# ------------------------------------------------------------------ the scene


@dataclass(frozen=True)
class Scene:
    """What the reference reads off a scene: the generator's decisions."""

    iron: tuple[Tile, ...]
    iron_rect: Rect
    coal: tuple[Tile, ...]
    chest: Tile
    walls: tuple[Tile, ...]
    start: Tile


def _floor(p) -> Tile:
    return (math.floor(p[0]), math.floor(p[1]))


def _bounds(tiles) -> Rect:
    xs = [t[0] for t in tiles]
    ys = [t[1] for t in tiles]
    return (min(xs), min(ys), max(xs), max(ys))


def scene_of(blueprint: dict) -> Scene:
    """The scene a belt_smelting blueprint declares.

    The iron patch's bounding box is the generator's rectangle: a split patch
    loses a strip across its middle and a clipped one its corners, never a
    whole edge."""
    iron = [_floor(r["position"]) for r in blueprint["resources"] if r["name"] == "iron-ore"]
    coal = [_floor(r["position"]) for r in blueprint["resources"] if r["name"] == "coal"]
    chest = next(e for e in blueprint["entities"] if e.get("marker") == "output")
    walls = [_floor(e["position"]) for e in blueprint["entities"] if e["name"] == "stone-wall"]
    return Scene(
        iron=tuple(iron),
        iron_rect=_bounds(iron),
        coal=tuple(coal),
        chest=_floor(chest["position"]),
        walls=tuple(walls),
        start=_floor(blueprint["character"]["position"]),
    )


def water_of(env) -> set[Tile]:
    """The map's water tiles near the scene, from the simulator's terrain."""
    out = set()
    for k in range(env.water_count):
        x, y = env.water[2 * k], env.water[2 * k + 1]
        if -SCENE_HALF - 8 <= x < SCENE_HALF + 8 and -SCENE_HALF - 8 <= y < SCENE_HALF + 8:
            out.add((x, y))
    return out


# ------------------------------------------------------------------ planner

STEP = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}
_CLOCKWISE = ("north", "east", "south", "west")


def _turn(direction: str, quarter_turns: int) -> str:
    return _CLOCKWISE[(_CLOCKWISE.index(direction) + quarter_turns) % 4]


def _rotate_tile(tile: Tile, quarter_turns: int) -> Tile:
    x, y = tile[0] + 0.5, tile[1] + 0.5
    for _ in range(quarter_turns % 4):
        x, y = -y, x
    return (math.floor(x), math.floor(y))


def _rotate_point(point: Tile, quarter_turns: int) -> Tile:
    x, y = point
    for _ in range(quarter_turns % 4):
        x, y = -y, x
    return (x, y)


def footprint(name: str, centre: tuple[float, float]) -> list[Tile]:
    if name in ("burner-mining-drill", "stone-furnace"):
        cx, cy = round(centre[0]), round(centre[1])
        return [(cx - 1, cy - 1), (cx, cy - 1), (cx - 1, cy), (cx, cy)]
    return [(math.floor(centre[0]), math.floor(centre[1]))]


@dataclass(frozen=True)
class Placement:
    item: str
    centre: tuple[float, float]
    direction: str

    @property
    def tiles(self) -> list[Tile]:
        return footprint(self.item, self.centre)

    @property
    def request_tile(self) -> Tile:
        """The tile to name: a 2x2 snaps from `t + 0.5` to `t + 1`."""
        if self.item in ("burner-mining-drill", "stone-furnace"):
            return (round(self.centre[0]) - 1, round(self.centre[1]) - 1)
        return (math.floor(self.centre[0]), math.floor(self.centre[1]))


_CELL_MACHINES = (
    ("burner-mining-drill", (0, 0), "south"),
    ("burner-mining-drill", (2, 0), "south"),
    ("stone-furnace", (0, 2), None),
    ("stone-furnace", (2, 2), None),
)
_CELL_INSERTERS = (((-1, 3), "north"), ((1, 3), "north"))
_CELL_COLLECTION = ((-1, 4), (0, 4), (1, 4))


@dataclass
class LinePlan:
    machines: list[Placement] = field(default_factory=list)
    inserters: list[Placement] = field(default_factory=list)
    belts: list[Placement] = field(default_factory=list)
    chest_inserter: Placement | None = None

    @property
    def drills(self) -> list[Placement]:
        return [p for p in self.machines if p.item == "burner-mining-drill"]

    @property
    def furnaces(self) -> list[Placement]:
        return [p for p in self.machines if p.item == "stone-furnace"]


def _cell(anchor: Tile, turns: int, reverse: bool):
    machines = []
    for item, point, facing in _CELL_MACHINES:
        dx, dy = _rotate_point(point, turns)
        machines.append(
            Placement(
                item,
                (float(anchor[0] + dx), float(anchor[1] + dy)),
                _turn(facing, turns) if facing else "north",
            )
        )
    inserters = []
    for tile, facing in _CELL_INSERTERS:
        rx, ry = _rotate_tile(tile, turns)
        inserters.append(
            Placement(
                "burner-inserter",
                (anchor[0] + rx + 0.5, anchor[1] + ry + 0.5),
                _turn(facing, turns),
            )
        )
    collection = [
        (anchor[0] + _rotate_tile(t, turns)[0], anchor[1] + _rotate_tile(t, turns)[1])
        for t in _CELL_COLLECTION
    ]
    if reverse:
        collection.reverse()
    return machines, inserters, collection


def _direction_between(a: Tile, b: Tile) -> str:
    step = (b[0] - a[0], b[1] - a[1])
    for name, delta in STEP.items():
        if delta == step:
            return name
    raise ValueError(f"{a} and {b} are not adjacent")


def _in_scene(t: Tile) -> bool:
    return -SCENE_HALF <= t[0] < SCENE_HALF and -SCENE_HALF <= t[1] < SCENE_HALF


def _bfs(start: Tile, blocked: set[Tile], goal: Tile | None = None) -> dict:
    parents: dict[Tile, Tile | None] = {start: None}
    queue = deque([start])
    while queue:
        here = queue.popleft()
        if here == goal:
            break
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (here[0] + dx, here[1] + dy)
            if nxt in parents or nxt in blocked or not _in_scene(nxt):
                continue
            parents[nxt] = here
            queue.append(nxt)
    return parents


def _depths(start: Tile, blocked: set[Tile]) -> dict[Tile, int]:
    depth = {start: 0}
    queue = deque([start])
    while queue:
        here = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (here[0] + dx, here[1] + dy)
            if nxt in depth or nxt in blocked or not _in_scene(nxt):
                continue
            depth[nxt] = depth[here] + 1
            queue.append(nxt)
    return depth


def _path(parents: dict, goal: Tile) -> list[Tile]:
    path = [goal]
    while parents[path[-1]] is not None:
        path.append(parents[path[-1]])
    return path[::-1]


def plan_candidates(s: Scene, blocked: set[Tile] = frozenset(), belts: int = BELTS) -> list:
    """Every (belts, side, anchor, turns, reverse) layout that fits, fewest belts first.

    `plan_line`'s search, run to the end instead of stopping at the best: the
    first entry is the reference's layout, and the ones tied with it on belt
    count are the variants `BeltBuilder(rng=...)` draws from."""
    return _plan(s, blocked, belts, exhaustive=True)


def plan_line(s: Scene, blocked: set[Tile] = frozenset(), belts: int = BELTS) -> LinePlan | None:
    """The shortest-belt reference layout (FactorioRL `plan_line`), or None.

    `blocked` adds tiles nothing can be built on -- the map's water -- to the
    scene's walls and chest."""
    found = _plan(s, blocked, belts, exhaustive=False)
    return found[0][-1] if found else None


def _plan(s: Scene, blocked, belts: int, exhaustive: bool) -> list:
    ore = set(s.iron)
    static = set(s.walls) | {s.chest} | set(blocked)
    candidates = []
    for side in _CLOCKWISE:
        sx, sy = STEP[side]
        ins_tile = (s.chest[0] + sx, s.chest[1] + sy)
        pick_tile = (s.chest[0] + 2 * sx, s.chest[1] + 2 * sy)
        if ins_tile in static or pick_tile in static:
            continue
        lower = _depths(pick_tile, static | {ins_tile})
        for ax in range(s.iron_rect[0] - 2, s.iron_rect[2] + 3):
            for ay in range(s.iron_rect[1] - 2, s.iron_rect[3] + 3):
                for turns in range(4):
                    for reverse in (False, True):
                        machines, inserters, collection = _cell((ax, ay), turns, reverse)
                        drill_tiles = [t for m in machines[:2] for t in m.tiles]
                        if not all(t in ore for t in drill_tiles):
                            continue
                        cell_tiles = {t for m in machines for t in m.tiles}
                        cell_tiles |= {t for i in inserters for t in i.tiles}
                        cell_tiles |= set(collection)
                        if cell_tiles & (static | {ins_tile, pick_tile}):
                            continue
                        if collection[-1] not in lower:
                            continue
                        bound = lower[collection[-1]] + len(collection)
                        candidates.append(
                            (bound, side, (ax, ay), turns, reverse, ins_tile, pick_tile)
                        )
    candidates.sort(key=lambda c: (c[0], _CLOCKWISE.index(c[1]), c[2], c[3], c[4]))

    found = []
    best: int | None = None
    for bound, side, anchor, turns, reverse, ins_tile, pick_tile in candidates:
        if bound > belts or (not exhaustive and best is not None and bound >= best):
            break
        if exhaustive and best is not None and bound > best:
            break
        machines, inserters, collection = _cell(anchor, turns, reverse)
        cell_tiles = {t for m in machines for t in m.tiles}
        cell_tiles |= {t for i in inserters for t in i.tiles}
        cell_tiles |= set(collection[:-1])
        parents = _bfs(collection[-1], static | cell_tiles | {ins_tile}, goal=pick_tile)
        if pick_tile not in parents:
            continue
        route = collection[:-1] + _path(parents, pick_tile)
        if len(route) > belts:
            continue
        if best is not None and (len(route) > best or (not exhaustive and len(route) >= best)):
            continue
        plan = LinePlan(machines=machines, inserters=inserters)
        for here, nxt in zip(route, route[1:], strict=False):
            plan.belts.append(
                Placement(
                    "transport-belt", (here[0] + 0.5, here[1] + 0.5), _direction_between(here, nxt)
                )
            )
        plan.belts.append(
            Placement(
                "transport-belt",
                (pick_tile[0] + 0.5, pick_tile[1] + 0.5),
                _direction_between(pick_tile, ins_tile),
            )
        )
        plan.chest_inserter = Placement(
            "burner-inserter", (ins_tile[0] + 0.5, ins_tile[1] + 0.5), side
        )
        if best is None or len(route) < best:
            best = len(route)
            found = [f for f in found if f[0] <= best]
        found.append((len(route), side, anchor, turns, reverse, plan))
    found.sort(key=lambda f: f[0])
    return found


# ------------------------------------------------------------------ fuel


@dataclass(frozen=True)
class FuelPlan:
    plates: int
    drill: int
    furnace: int
    output_inserter: int
    chest_inserter: int


def fuel_plan(coal: int, drills: int = 2, furnaces: int = 2, output_inserters: int = 2) -> FuelPlan:
    """FactorioRL `fuel_plan`: the most plates `coal` buys through the cell, and the split."""

    def coal_for(joules: float) -> int:
        return max(0, math.ceil(joules / COAL_J - 1e-9))

    for plates in range(min(CELL_PLATE_CEILING, coal * COAL_J // DRILL_J_PER_ORE), 0, -1):
        downstream = plates * FUEL_MARGIN
        drill = coal_for(plates / drills * DRILL_J_PER_ORE)
        furnace = coal_for(downstream / furnaces * FURNACE_J_PER_PLATE)
        output = max(
            1, coal_for(downstream / output_inserters * INSERTER_J_PER_SWING - INSERTER_BUILT_J)
        )
        chest = max(1, coal_for(downstream * INSERTER_J_PER_BELT_PICKUP - INSERTER_BUILT_J))
        total = drill * drills + furnace * furnaces + output * output_inserters + chest
        if total <= coal:
            return FuelPlan(plates, drill, furnace, output, chest)
    return FuelPlan(0, 0, 0, 0, 0)


# ------------------------------------------------------------------ builder


class BeltBuilder:
    """Drives one belt_smelting `fsim_rl` through the reference build, one decision at a time.

    `rl` is the `fsim_rl *` of an environment reset under v3 on `blueprint`.
    `next_vector()` is the next decision, or None once the build has ended
    (its last decision is `finish`); the caller steps each one. `stuck` names
    why the build stopped early, if it did; the builder then finishes, so the
    episode still ends on its verification. `rng` draws the layout among those
    tied with the reference's on belt count, instead of taking the
    reference's own, first one.
    """

    def __init__(self, rl, blueprint: dict, rng: random.Random | None = None,
                 extra_coal: int = EXTRA_COAL) -> None:  # fmt: skip
        self.rl = rl
        self.scene = scene_of(blueprint)
        self.water = water_of(rl.env)
        self.extra_coal = extra_coal
        if rng is None:
            self.plan = plan_line(self.scene, self.water)
        else:
            found = plan_candidates(self.scene, self.water)
            self.plan = rng.choice(found)[-1] if found else None
        self.walls = set(self.scene.walls) | {self.scene.chest} | self.water
        self.solid: set[Tile] = set()
        self.belts: set[Tile] = set()
        self.reserved: set[Tile] = set()
        if self.plan is not None:
            self.reserved = {
                t
                for p in [
                    *self.plan.machines,
                    *self.plan.inserters,
                    *self.plan.belts,
                    self.plan.chest_inserter,
                ]  # fmt: skip
                for t in p.tiles
            }
        self.entity_of: dict[tuple[str, tuple[float, float]], int] = {}
        self.fuel: FuelPlan | None = None
        self.stuck: str | None = None
        self.decisions = 0
        self.finished = False
        self._vector = ffi.new("int32_t[6]")
        self._action = ffi.new("fsim_action *")
        self._handles = ffi.new("int32_t[]", MAX_TARGETS)
        self._script = self._run()

    # ---- the interface ----------------------------------------------------

    def next_vector(self) -> list[int] | None:
        if self.finished or self.rl.done:
            return None
        try:
            vector = next(self._script)
        except StopIteration:
            vector = None
        if vector is None:
            self.finished = True
            # The episode still ends on its verification, whatever happened.
            return list(FINISH)
        self.decisions += 1
        if list(vector) == FINISH:
            self.finished = True
        return list(vector)

    # ---- reading the state --------------------------------------------------

    def position(self) -> tuple[float, float]:
        env = self.rl.env
        return (env.char_pos.x / TILE, env.char_pos.y / TILE)

    def here(self) -> Tile:
        return _floor(self.position())

    def held(self, item: int) -> int:
        env = self.rl.env
        return sum(
            env.main[i].count
            for i in range(80)
            if env.main[i].item == item and env.main[i].count > 0
        )

    def _fail(self, reason: str) -> bool:
        if self.stuck is None:
            self.stuck = reason
        return False

    def _legal(self, vector) -> bool:
        for i in range(6):
            self._vector[i] = int(vector[i])
        return not lib.fsim_rl_decode(self.rl, self._vector, self._action)

    def _find_entity(self, placement: Placement) -> int | None:
        """The live entity `placement` built, by kind and centre."""
        env = self.rl.env
        kind = KIND[placement.item]
        best = None
        for i in range(env.entity_count):
            e = env.entities[i]
            if not e.alive or e.kind != kind:
                continue
            d = math.dist((e.pos.x / TILE, e.pos.y / TILE), placement.centre)
            if d <= 0.6 and (best is None or d < best[0]):
                best = (d, i)
        return best[1] if best else None

    def _row(self, entity: int) -> int:
        """The target argument (row + 1) naming a visible entity, or 0."""
        env = self.rl.env
        handle = next(
            (env.seen[k].handle for k in range(env.seen_count) if env.seen[k].entity == entity),
            None,
        )
        if handle is None:
            return 0
        count = lib.fsim_rl_targets(self.rl, self._handles, MAX_TARGETS)
        return next((k + 1 for k in range(count) if self._handles[k] == handle), 0)

    @staticmethod
    def _slot(tile: Tile, here: Tile) -> int:
        dx, dy = tile[0] - here[0], tile[1] - here[1]
        if abs(dx) > PLACEMENT_RADIUS or abs(dy) > PLACEMENT_RADIUS:
            return 0
        return (dx + PLACEMENT_RADIUS) * PLACEMENT_SIDE + dy + PLACEMENT_RADIUS + 1

    # ---- walking --------------------------------------------------------------

    def _costs_from(self, origin: Tile):
        blocked = self.walls | self.solid
        cost = {origin: 0}
        parent: dict[Tile, Tile | None] = {origin: None}
        heap = [(0, origin)]
        while heap:
            c, here = heapq.heappop(heap)
            if c > cost.get(here, 1 << 30):
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (here[0] + dx, here[1] + dy)
                if nxt in blocked or not _in_scene(nxt):
                    continue
                step = BELT_WALK_COST if nxt in self.belts else 1
                if c + step < cost.get(nxt, 1 << 30):
                    cost[nxt] = c + step
                    parent[nxt] = here
                    heapq.heappush(heap, (c + step, nxt))
        return cost, parent

    def _settled_on(self, goal: Tile) -> bool:
        x, y = self.position()
        return (
            abs(x - goal[0] - 0.5) <= STAND_TOLERANCE and abs(y - goal[1] - 0.5) <= STAND_TOLERANCE
        )

    def _walk_to(self, target: tuple[float, float]):
        """Axis moves to within STAND_TOLERANCE of `target`; False when a stride stalls."""
        for _ in range(RUN_BUDGET):
            x, y = self.position()
            dx, dy = target[0] - x, target[1] - y
            if abs(dx) <= STAND_TOLERANCE and abs(dy) <= STAND_TOLERANCE:
                return True
            if abs(dx) >= abs(dy):
                error, direction = abs(dx), (1 if dx > 0 else 3)
            else:
                error, direction = abs(dy), (2 if dy > 0 else 0)
            if error > STRIDE_TILES[MOVE_LONG] + 0.1:
                base = MOVE_LONG
            elif error > STRIDE_TILES[MOVE_STEP] + 0.05:
                base = MOVE_STEP
            else:
                base = MOVE_NUDGE
            yield [base + direction, 0, 0, 0, 0, 0]
            if self.rl.done:
                return False
            if math.dist((x, y), self.position()) < 0.02:
                return False
        return False

    def _walk_tile(self, goal: Tile, parent=None):
        for _attempt in range(WALK_ATTEMPTS):
            if self._settled_on(goal):
                return True
            start = self.here()
            if parent is None or start not in parent:
                _, parent = self._costs_from(start)
            if goal not in parent:
                return self._fail(f"no walkable route from {start} to {goal}")
            route = [goal]
            while parent[route[-1]] is not None and route[-1] != start:
                route.append(parent[route[-1]])
            route.reverse()
            corners = [route[0]]
            for a, b, c in zip(route, route[1:], route[2:], strict=False):
                if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
                    corners.append(b)
            corners.append(route[-1])
            # Settle on the start tile's centre first, so each straight run
            # stays inside its row of tiles.
            ok = True
            for corner in corners:
                if not (yield from self._walk_to((corner[0] + 0.5, corner[1] + 0.5))):
                    ok = False
                    break
            if ok and self._settled_on(goal):
                return True
            if self.rl.done:
                return False
            parent = None
        return self._fail(f"could not walk to {goal}")

    def _stand_for(self, placement: Placement):
        request = placement.request_tile
        cost, parent = self._costs_from(self.here())
        best = None
        for dx in range(-PLACEMENT_RADIUS, PLACEMENT_RADIUS + 1):
            for dy in range(-PLACEMENT_RADIUS, PLACEMENT_RADIUS + 1):
                tile = (request[0] + dx, request[1] + dy)
                if tile not in cost or tile in self.belts or tile in self.reserved:
                    continue
                if any(max(abs(tile[0] - f[0]), abs(tile[1] - f[1])) < 2 for f in placement.tiles):
                    continue
                if math.dist((tile[0] + 0.5, tile[1] + 0.5), placement.centre) > BUILD_REACH:
                    continue
                key = (cost[tile], abs(dx) + abs(dy), tile)
                if best is None or key < best[0]:
                    best = (key, tile)
        return (best[1], parent) if best else (None, None)

    def _stand_near(self, centre: tuple[float, float]):
        cost, parent = self._costs_from(self.here())
        options = [
            (c, t)
            for t, c in cost.items()
            if t not in self.belts
            and t not in self.reserved
            and math.dist((t[0] + 0.5, t[1] + 0.5), centre) <= FUEL_REACH
        ]
        if not options:
            return self._fail(f"nowhere to stand within reach of {centre}")
        return (yield from self._walk_tile(min(options)[1], parent))

    # ---- acting ---------------------------------------------------------------

    def _place(self, placement: Placement):
        stand, parent = self._stand_for(placement)
        if stand is None:
            return self._fail(f"nowhere to stand to build {placement.item} at {placement.centre}")
        if not (yield from self._walk_tile(stand, parent)):
            return False
        slot = self._slot(placement.request_tile, self.here())
        vector = [
            OP_PLACE,
            0,
            slot,
            DIRECTION_ARG[placement.direction],
            ITEM_ARG[placement.item],
            0,
        ]
        if not slot or not self._legal(vector):
            return self._fail(f"tile {placement.request_tile} is not an offered placement")
        yield vector
        entity = self._find_entity(placement)
        if entity is None:
            return self._fail(f"placed {placement.item} but none stands at {placement.centre}")
        self.entity_of[(placement.item, placement.centre)] = entity
        for tile in placement.tiles:
            self.reserved.discard(tile)
            (self.belts if placement.item == "transport-belt" else self.solid).add(tile)
        return True

    def _give_coal(self, placement: Placement, count: int):
        entity = self.entity_of.get((placement.item, placement.centre))
        if entity is None:
            return self._fail(f"no {placement.item} built at {placement.centre}")
        if math.dist(self.position(), placement.centre) > FUEL_REACH:
            if not (yield from self._stand_near(placement.centre)):
                return False
        for amount in (20, 5, 1):
            while count >= amount:
                vector = [OP_GIVE, self._row(entity), 0, 0, ITEM_ARG["coal"], AMOUNT_ARG[amount]]
                if not vector[1] or not self._legal(vector):
                    return self._fail(
                        f"cannot give coal to the {placement.item} at {placement.centre}"
                    )
                yield vector
                count -= amount
        return True

    def _coal_tiles_in_reach(self) -> list[tuple[float, Tile]]:
        env = self.rl.env
        here = self.position()
        out = []
        for k in range(env.tile_count):
            r = env.resources[env.tiles[k].resource]
            if not r.alive or r.item != lib.IT_COAL:
                continue
            d = math.dist(here, (r.tx + 0.5, r.ty + 0.5))
            if d <= RESOURCE_REACH:
                out.append((d, (r.tx, r.ty)))
        return sorted(out)

    def _mine_coal(self, count: int):
        """Stand on the coal patch and hand-mine `count` coal, 20, 5 or 1 a request."""
        coal = set(self.scene.coal) - self.water
        cost, parent = self._costs_from(self.here())
        reachable = [(cost[t], t) for t in coal if t in cost]
        if not reachable:
            return self._fail("no coal tile is reachable")
        if not (yield from self._walk_tile(min(reachable)[1], parent)):
            return False
        goal = self.held(lib.IT_COAL) + count
        while self.held(lib.IT_COAL) < goal:
            held = self.held(lib.IT_COAL)
            tiles = self._coal_tiles_in_reach()
            if not tiles:
                return self._fail("no coal tile in reach at the coal patch")
            batch = next(n for n in (20, 5, 1) if n <= goal - held)
            vector = [
                OP_MINE_TILE,
                0,
                self._slot(tiles[0][1], self.here()),
                0,
                0,
                AMOUNT_ARG[batch],
            ]
            if not self._legal(vector):
                return self._fail(f"mine_tile {tiles[0][1]} is not legal")
            yield vector
            for _ in range(4 * batch + 8):
                if self.held(lib.IT_COAL) >= held + batch:
                    break
                yield WAIT
            else:
                return self._fail(
                    f"hand-mining {batch} coal yielded {self.held(lib.IT_COAL) - held}"
                )
        return True

    def _run(self):
        plan = self.plan
        if plan is None:
            self._fail("no line fits this scene in 40 belts")
            return
        if self.extra_coal and not (yield from self._mine_coal(self.extra_coal)):
            return
        self.fuel = fuel = fuel_plan(self.held(lib.IT_COAL))
        for placement in [*plan.machines, *plan.inserters]:
            if not (yield from self._place(placement)):
                return
        for placement in plan.inserters:
            if not (yield from self._give_coal(placement, fuel.output_inserter)):
                return
        for placement in plan.furnaces:
            if not (yield from self._give_coal(placement, fuel.furnace)):
                return
        for placement in [*plan.belts, plan.chest_inserter]:
            if not (yield from self._place(placement)):
                return
        if not (yield from self._give_coal(plan.chest_inserter, fuel.chest_inserter)):
            return
        share = self.held(lib.IT_COAL) // len(plan.drills)
        for placement in plan.drills:
            if not (yield from self._give_coal(placement, max(fuel.drill, share))):
                return
        yield list(FINISH)


def run(env, blueprint: dict, rng: random.Random | None = None,
        extra_coal: int = EXTRA_COAL) -> dict:  # fmt: skip
    """Reset `env` (an `RlEnv`) to `blueprint` under v3 and run the whole build.

    Returns the outcome: success, the verified output and its two terms, the
    decisions spent, the fuel plan and why the build stopped early, if it did."""
    env.reset("belt_smelting", blueprint, action_space="v3")
    builder = BeltBuilder(env.rl, blueprint, rng=rng, extra_coal=extra_coal)
    vector = ffi.new("int32_t[6]")
    rl = env.rl
    while (v := builder.next_vector()) is not None:
        for i in range(6):
            vector[i] = v[i]
        lib.fsim_rl_step(rl, vector)
        if rl.done:
            break
    return {
        "success": bool(rl.success),
        "verified": bool(rl.verified),
        "plates": float(rl.verified_output),
        "uncapped": float(rl.verified_uncapped),
        "ore_in_window": float(rl.verified_source),
        "decisions": int(rl.steps),
        "tick_at_verify": None,
        "fuel_plan": builder.fuel,
        "stuck": builder.stuck,
    }
