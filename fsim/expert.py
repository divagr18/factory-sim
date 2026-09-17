"""A scripted builder for construct_smelting_line, and the start states it makes.

`Builder` issues MultiDiscrete vectors -- the same action space a policy uses
-- following FactorioRL's evaluator-only reference (`tasks.reference._build_at`):
stand three tiles east of the drill centre, place the drill facing south on the
tile up-left of that centre, place the furnace two tiles below it, fuel both
with twenty coal. It reads argument indices off the state the way the C
domains lay them out (`rl_domains_build`), and `test_expert.py` checks that
what it builds is a line that passes verification.

It exists for one purpose: **demonstration starts**. Resetting an episode to a
state partway along a demonstration, and letting the policy take over from
there, is how Salimans & Chen (2018) learned Montezuma's Revenge from a single
demonstration, and it is the demonstration-driven case of the reverse
curriculum of Florensa et al. (2017). The policy never sees the builder's
actions as labels; it only starts some episodes from where the builder
stopped. A run that uses it is not a from-scratch run, and says so.
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


def placement_index(rl, tx: int, ty: int) -> int:
    """The placement argument (1-based) that names tile (tx, ty), or 0."""
    env = rl.env
    here_x, here_y = _floor_tile(env.char_pos.x), _floor_tile(env.char_pos.y)
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
    best, best_d = 0, math.inf
    for k in range(min(env.seen_count, MAX_TARGETS)):
        e = env.entities[env.seen[k].entity]
        if e.kind != kind:
            continue
        d = math.dist((e.pos.x / TILE, e.pos.y / TILE), near)
        if d < best_d:
            best, best_d = k + 1, d
    return best


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

    def __init__(self, rl, patch: tuple[float, float], stop: int = len(STAGES)) -> None:
        self.rl = rl
        self.stop = stop  # how many stages to complete
        # The ore tile at the patch centre: its drill covers four ore tiles on
        # every train-family patch.
        tile = (math.floor(patch[0]), math.floor(patch[1]))
        self.drill_centre = (tile[0] + 1, tile[1] + 1)
        self.standing = (self.drill_centre[0] + 3.5, self.drill_centre[1] + 0.5)
        self.stage = 0  # index into STAGES of the next stage to finish

    def next_vector(self) -> list[int] | None:
        """The next decision, or None when the build is complete."""
        rl, cx, cy = self.rl, *self.drill_centre
        if self.stage == 0:
            vector = move_vector(rl, self.standing)
            if vector is not None:
                return vector
            self.stage = 1
        if self.stage >= self.stop:
            return None
        if self.stage == 1:
            self.stage = 2
            return [OP_PLACE, 0, placement_index(rl, cx - 1, cy - 1), DIR_SOUTH, ITEM_DRILL, 0]
        if self.stage == 2:
            self.stage = 3
            return [OP_PLACE, 0, placement_index(rl, cx - 1, cy + 1), DIR_NORTH, ITEM_FURNACE, 0]
        if self.stage == 3:
            self.stage = 4
            return [OP_GIVE, target_index(rl, K_DRILL, (cx, cy)), 0, 0, ITEM_COAL, AMOUNT_20]
        if self.stage == 4:
            self.stage = 5
            target = target_index(rl, K_FURNACE, (cx, cy + 2))
            return [OP_GIVE, target, 0, 0, ITEM_COAL, AMOUNT_20]
        return None


def advance_to(rl, patch, stage: str, step) -> int:
    """Run the builder until `stage` is complete; returns decisions taken.

    `step(vector)` performs one decision (the caller's, so a batched
    environment can keep its own bookkeeping) and returns whether the episode
    ended. Stops early if it does.
    """
    builder = Builder(rl, patch, stop=STAGES.index(stage) + 1)
    taken = 0
    while (vector := builder.next_vector()) is not None:
        taken += 1
        if step(vector):
            break
    return taken


def run_to_completion(rl, patch, max_waits: int = 1000) -> None:
    """The whole build, then waits until the episode ends."""
    builder = Builder(rl, patch)
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
