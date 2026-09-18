"""A scripted builder for construct_smelting_line, and the start states it makes.

`Builder` issues MultiDiscrete vectors -- the same action space a policy uses
-- following FactorioRL's evaluator-only reference (`tasks.reference._build_at`):
stand three tiles east of the drill centre, place the drill facing south on the
tile up-left of that centre, place the furnace two tiles below it, fuel both
with twenty coal. It reads argument indices off the state the way the C
domains lay them out (`rl_domains_build`), and `test_expert.py` checks that
what it builds is a line that passes verification.

That arrangement is one of many. `layouts` enumerates every drill anchor the
patch's ore admits and all four turns of the layout about the drill's centre,
and `choose_layout` draws one per episode. Building the same pose every time
teaches a policy that pose rather than the task: measured, a policy trained on
the single canonical layout reaches 84% on unseen scenes of the shape it was
trained on and 0% once a wall stands where it expects to stand.

It exists for one purpose: **demonstration starts**. Resetting an episode to a
state partway along a demonstration, and letting the policy take over from
there, is how Salimans & Chen (2018) learned Montezuma's Revenge from a single
demonstration, and it is the demonstration-driven case of the reverse
curriculum of Florensa et al. (2017). The policy never sees the builder's
actions as labels; it only starts some episodes from where the builder
stopped. A run that uses it is not a from-scratch run, and says so.

`advance_decisions` cuts the build at any decision, which is what Backplay
(Resnick et al. 2018, arXiv:1807.06919) needs: it samples starts from a window
measured backwards from the end of the demonstration and slides that window
back on a fixed schedule. Drawing uniformly from the whole demonstration
instead is their "Uniform" baseline, which they measure as slower.
"""

from __future__ import annotations

import math

from fsim import ffi, lib

TILE = 256
PLACEMENT_RADIUS = 5
MAX_TARGETS = 32
MAX_PLACEMENTS = 121
K_DRILL, K_FURNACE, K_PILE = 1, 2, 4

# parameterized-v1 indices (see fsim/policy.py and csrc/fsim_rl.c)
OP_PLACE, OP_GIVE, OP_WAIT = 12, 16, 21
ITEM_COAL, ITEM_FURNACE, ITEM_DRILL = 3, 7, 13  # encoders.ITEMS position + 1
DIR_NORTH, DIR_SOUTH = 1, 3  # catalog DIRECTIONS position + 1
AMOUNT_20 = 3

#: How far along the build a start state is.
STAGES = ("walked", "drill", "furnace", "drill_fuelled", "furnace_fuelled")


def _floor_tile(v: int) -> int:
    return v // TILE  # Python floor division matches the C floordiv


def _turn(offset: tuple[float, float], quarters: int) -> tuple[float, float]:
    """`offset` turned `quarters` right angles clockwise (y grows southwards)."""
    x, y = offset
    for _ in range(quarters % 4):
        x, y = -y, x
    return x, y


def _turn_direction(direction: int, quarters: int) -> int:
    """A catalog direction (1 north, 2 east, 3 south, 4 west), turned clockwise."""
    return (direction - 1 + quarters) % 4 + 1


def _ore_tiles(env) -> set[tuple[int, int]]:
    return {
        (env.resources[i].tx, env.resources[i].ty)
        for i in range(env.resource_count)
        if env.resources[i].alive
    }


def _taken_tiles(env) -> set[tuple[int, int]]:
    """Tiles nothing can be built on or stood in: obstacles and other machines."""
    taken = {(env.blocked[2 * k], env.blocked[2 * k + 1]) for k in range(env.blocked_count)}
    for k in range(env.seen_count):
        e = env.entities[env.seen[k].entity]
        if e.kind != K_PILE:
            taken.add((_floor_tile(e.pos.x), _floor_tile(e.pos.y)))
    return taken


def _square(anchor: tuple[int, int]) -> list[tuple[int, int]]:
    """The four tiles of a 2x2 machine anchored at its north-west tile."""
    ax, ay = anchor
    return [(ax, ay), (ax + 1, ay), (ax, ay + 1), (ax + 1, ay + 1)]


def layouts(rl, patch) -> list[tuple[tuple[int, int], int]]:
    """Every `(drill anchor, quarter-turns)` this scene can be built with.

    The drill's four tiles must all be ore, the furnace's four and the tile the
    builder stands in must be clear, and both machines must fall inside the
    placement window around that tile. The canonical layout -- the drill on the
    patch centre, the furnace due south, the builder standing to the east -- is
    FactorioRL's `_build_at` reference, and comes first.
    """
    env = rl.env
    ore, taken = _ore_tiles(env), _taken_tiles(env)
    canonical = (math.floor(patch[0]), math.floor(patch[1]))
    anchors = sorted(a for a in ore | {canonical} if set(_square(a)) <= ore)
    anchors.sort(key=lambda a: (a != canonical, a))
    found = []
    for anchor in anchors:
        centre = (anchor[0] + 1, anchor[1] + 1)
        for quarters in range(4):
            fx, fy = _turn((0.0, 2.0), quarters)
            furnace = (round(centre[0] + fx) - 1, round(centre[1] + fy) - 1)
            sx, sy = _turn((3.5, 0.5), quarters)
            stand = (math.floor(centre[0] + sx), math.floor(centre[1] + sy))
            if any(t in taken for t in _square(furnace) + [stand]):
                continue
            if any(
                abs(t[0] - stand[0]) > PLACEMENT_RADIUS or abs(t[1] - stand[1]) > PLACEMENT_RADIUS
                for t in (anchor, furnace)
            ):
                continue
            found.append((anchor, quarters))
    return found


def choose_layout(rl, patch, rng=None) -> tuple[tuple[int, int], int]:
    """One layout for this scene: the canonical one, or a random valid one.

    Drawing a layout per episode is what stops demonstration starts teaching a
    single build pose. A policy trained on one pose memorises it, and any scene
    that blocks that pose -- an obstacle standing on it, an ore patch whose
    shape moves it -- leaves that policy with nothing to fall back on.
    """
    canonical = ((math.floor(patch[0]), math.floor(patch[1])), 0)
    if rng is None:
        return canonical
    found = layouts(rl, patch)
    return rng.choice(found) if found else canonical


def placement_index(rl, tx: int, ty: int) -> int:
    """The placement argument (1-based) that names tile (tx, ty), or 0."""
    env = rl.env
    here_x, here_y = _floor_tile(env.char_pos.x), _floor_tile(env.char_pos.y)
    if rl.task.action_space == lib.ACTION_SPACE_V2:
        dx, dy = tx - here_x, ty - here_y
        if abs(dx) > PLACEMENT_RADIUS or abs(dy) > PLACEMENT_RADIUS:
            return 0
        side = 2 * PLACEMENT_RADIUS + 1
        return (dx + PLACEMENT_RADIUS) * side + dy + PLACEMENT_RADIUS + 1
    occupied = set()
    for k in range(env.seen_count):
        e = env.entities[env.seen[k].entity]
        if e.kind != K_PILE:
            occupied.add((_floor_tile(e.pos.x), _floor_tile(e.pos.y)))
    blocked = {(env.blocked[2 * k], env.blocked[2 * k + 1]) for k in range(env.blocked_count)}
    index = 0
    for dx in range(-PLACEMENT_RADIUS, PLACEMENT_RADIUS + 1):
        for dy in range(-PLACEMENT_RADIUS, PLACEMENT_RADIUS + 1):
            x, y = here_x + dx, here_y + dy
            if (dx == 0 and dy == 0) or (x, y) in occupied or (x, y) in blocked:
                continue
            index += 1
            if (x, y) == (tx, ty):
                return index if index <= MAX_PLACEMENTS else 0
    return 0


def target_index(rl, kind: int, near: tuple[float, float]) -> int:
    """The target argument (1-based) of the visible `kind` nearest `near`, or 0."""
    env = rl.env
    best, best_handle, best_d = 0, 0, math.inf
    for k in range(min(env.seen_count, MAX_TARGETS)):
        e = env.entities[env.seen[k].entity]
        if e.kind != kind:
            continue
        d = math.dist((e.pos.x / TILE, e.pos.y / TILE), near)
        if d < best_d:
            best, best_handle, best_d = k + 1, env.seen[k].handle, d
    if rl.task.action_space != lib.ACTION_SPACE_V2 or not best:
        return best
    handles = ffi.new("int32_t[]", MAX_TARGETS)
    count = lib.fsim_rl_targets(rl, handles, MAX_TARGETS)
    return next((k + 1 for k in range(count) if handles[k] == best_handle), 0)


def move_vector(rl, goal: tuple[float, float], tolerance: float = 0.3) -> list[int] | None:
    """One walking decision towards `goal`, or None once within tolerance.

    Moves the axis with the larger error, with the longest of move (30 ticks,
    about 4.5 tiles), step (7 ticks, about 1 tile) and nudge (2 ticks, about
    0.3 tiles) that does not overshoot.
    """
    x, y = rl.env.char_pos.x / TILE, rl.env.char_pos.y / TILE
    dx, dy = goal[0] - x, goal[1] - y
    if abs(dx) <= tolerance and abs(dy) <= tolerance:
        return None
    if abs(dx) >= abs(dy):
        distance, direction = abs(dx), (1 if dx > 0 else 3)  # east / west
    else:
        distance, direction = abs(dy), (2 if dy > 0 else 0)  # south / north
    stride = 38 / 256
    if distance >= 30 * stride:
        base = 0
    elif distance >= 7 * stride:
        base = 4
    else:
        base = 8
    return [base + direction, 0, 0, 0, 0, 0]


class Builder:
    """Drives one `fsim_rl` through the build, one decision at a time."""

    def __init__(self, rl, patch: tuple[float, float], stop: int = len(STAGES), layout=None) -> None:
        self.rl = rl
        self.stop = stop  # how many stages to complete
        anchor, quarters = layout if layout is not None else choose_layout(rl, patch)
        self.quarters = quarters
        # The drill covers four ore tiles; everything else is placed relative
        # to the point at its centre, turned with it.
        self.drill_anchor = anchor
        self.drill_centre = (anchor[0] + 1, anchor[1] + 1)
        fx, fy = _turn((0.0, 2.0), quarters)
        self.furnace_centre = (self.drill_centre[0] + fx, self.drill_centre[1] + fy)
        self.furnace_anchor = (
            round(self.furnace_centre[0]) - 1,
            round(self.furnace_centre[1]) - 1,
        )
        sx, sy = _turn((3.5, 0.5), quarters)
        self.standing = (self.drill_centre[0] + sx, self.drill_centre[1] + sy)
        self.drill_facing = _turn_direction(DIR_SOUTH, quarters)
        self.furnace_facing = _turn_direction(DIR_NORTH, quarters)
        self.stage = 0  # index into STAGES of the next stage to finish

    def next_vector(self) -> list[int] | None:
        """The next decision, or None when the build is complete."""
        rl = self.rl
        if self.stage == 0:
            vector = move_vector(rl, self.standing)
            if vector is not None:
                return vector
            self.stage = 1
        if self.stage >= self.stop:
            return None
        if self.stage == 1:
            self.stage = 2
            slot = placement_index(rl, *self.drill_anchor)
            return [OP_PLACE, 0, slot, self.drill_facing, ITEM_DRILL, 0]
        if self.stage == 2:
            self.stage = 3
            slot = placement_index(rl, *self.furnace_anchor)
            return [OP_PLACE, 0, slot, self.furnace_facing, ITEM_FURNACE, 0]
        if self.stage == 3:
            self.stage = 4
            target = target_index(rl, K_DRILL, self.drill_centre)
            return [OP_GIVE, target, 0, 0, ITEM_COAL, AMOUNT_20]
        if self.stage == 4:
            self.stage = 5
            target = target_index(rl, K_FURNACE, self.furnace_centre)
            return [OP_GIVE, target, 0, 0, ITEM_COAL, AMOUNT_20]
        return None


def plan_length(rl, patch, layout=None) -> int:
    """How many decisions the whole build takes from here, without stepping.

    The walk is a deterministic function of the character's position, so its
    length is arithmetic: no scene the builder demonstrates has anything
    between the start and the standing spot. Four build decisions follow.
    """
    x, y = rl.env.char_pos.x / TILE, rl.env.char_pos.y / TILE
    goal = Builder(rl, patch, layout=layout).standing
    stride = 38 / 256
    walk = 0
    while walk < 200:
        dx, dy = goal[0] - x, goal[1] - y
        if abs(dx) <= 0.3 and abs(dy) <= 0.3:
            break
        if abs(dx) >= abs(dy):
            distance, axis = abs(dx), 0
        else:
            distance, axis = abs(dy), 1
        ticks = 30 if distance >= 30 * stride else (7 if distance >= 7 * stride else 2)
        moved = min(distance, ticks * stride)
        if axis == 0:
            x += moved if dx > 0 else -moved
        else:
            y += moved if dy > 0 else -moved
        walk += 1
    return walk + 4


def advance_decisions(rl, patch, count: int, step, layout=None) -> int:
    """Run the first `count` decisions of the build; returns how many ran."""
    builder = Builder(rl, patch, layout=layout)
    taken = 0
    while taken < count:
        vector = builder.next_vector()
        if vector is None or step(vector):
            break
        taken += 1
    return taken


def advance_to(rl, patch, stage: str, step, layout=None) -> int:
    """Run the builder until `stage` is complete; returns decisions taken.

    `step(vector)` performs one decision (the caller's, so a batched
    environment can keep its own bookkeeping) and returns whether the episode
    ended. Stops early if it does.
    """
    builder = Builder(rl, patch, stop=STAGES.index(stage) + 1, layout=layout)
    taken = 0
    while (vector := builder.next_vector()) is not None:
        taken += 1
        if step(vector):
            break
    return taken


def run_to_completion(rl, patch, max_waits: int = 1000, layout=None) -> None:
    """The whole build, then waits until the episode ends."""
    builder = Builder(rl, patch, layout=layout)
    ints = ffi.new("int32_t[6]")

    def step(vector):
        for i, v in enumerate(vector):
            ints[i] = v
        lib.fsim_rl_step(rl, ints)
        return bool(rl.done)

    while (vector := builder.next_vector()) is not None:
        if step(vector):
            return
    for _ in range(max_waits):
        if step([OP_WAIT, 0, 0, 0, 0, 0]):
            return
