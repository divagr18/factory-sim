"""The world a builder program drives, and the episode that scores it.

A builder program is a `def build(world): ...` an LLM writes. It is compared
with policies trained on the same task, so it gets exactly what a policy
gets: queries answer from the published observation (`fsim.obsview`) and
nothing else, and every action is one v2 MultiDiscrete vector, the same one a
policy would emit. That is also what lets the same program drive the real game
later: nothing it reads exists only in the simulator.

An intent the action space cannot express -- a tile outside the placement
window, an item not held, an entity row that is not there -- is refused before
it reaches the simulator. It costs no decision and is counted in
`world.refusals`. Legality is `fsim_rl_decode`'s, the check the environment
itself runs, so a legal intent here is never a decode failure there. An intent
that decodes but that the game then refuses (a drill onto a tile something
else covers) does cost its decision, as it would for a policy; `last_refused()`
reports it.

The evaluator side of an episode (success, the verified output, when the first
plate appeared, how far the character walked) may read simulator internals.
The program never can: it holds only a `World`.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from fsim import ffi, lib
from fsim.obsview import FACINGS, ITEMS, Entity, ObsView
from fsim.rl import RlEnv

__all__ = ["BudgetExhausted", "Entity", "EpisodeResult", "World", "run_episode"]

TILE = 256
PLACEMENT_RADIUS = 5
OP_PLACE, OP_MINE, OP_GIVE, OP_TAKE, OP_WAIT = 12, 13, 16, 17, 21
STRIDES = {"long": 0, "step": 4, "nudge": 8}  # ops: base + direction
MOVES = {"N": 0, "E": 1, "S": 2, "W": 3}
AMOUNTS = {1: 1, 5: 2, 20: 3}
TRACE_LENGTH = 20
#: Illegal intents a program may make before it is stopped: a loop that only
#: ever asks for impossible things would otherwise never spend its budget.
MAX_REFUSALS = 1000
WAIT = (OP_WAIT, 0, 0, 0, 0, 0)


class BudgetExhausted(Exception):
    """Raised inside the program when it has no decisions left."""


@dataclass
class EpisodeResult:
    success: bool
    verified_output: int
    decisions: int  # decisions the program spent (the fast-finish waits are not counted)
    refusals: int  # illegal intents: no decision spent
    first_plate_tick: int | None  # at decision granularity (30 ticks)
    walk_distance: float  # tiles, summed over the program's decisions
    built: list[tuple[str, int, int]]  # (item, tile_x, tile_y) placed during the episode
    trace: list[str]  # the last TRACE_LENGTH intents and their outcomes
    error: str | None  # the exception the program raised, if any (not BudgetExhausted)
    failures: int = 0  # legal actions the game refused (these did spend a decision)


class World:
    """What a builder program holds: observation queries and one-decision actions."""

    def __init__(self, env: RlEnv, decision_budget: int = 600) -> None:
        self._env = env
        self._rl = env.rl
        self._budget = decision_budget
        self._view: ObsView | None = None
        self._action = ffi.new("fsim_action *")
        self._vector = ffi.new("int32_t[6]")
        self._baseline_plates = self._rl.env.produced[lib.IT_IRON_PLATE]
        self.decisions = 0
        self.refusals = 0
        self.failures = 0
        self.walk_distance = 0.0
        self._first_plate_tick: int | None = None
        self._built: list[tuple[str, int, int]] = []
        self._trace: deque[str] = deque(maxlen=TRACE_LENGTH)

    # ------------------------------------------------------------ queries

    def _obs(self) -> ObsView:
        if self._view is None:
            self._view = ObsView(self._env.obs)
        return self._view

    def me(self) -> tuple[float, float]:
        """The character's position in tiles."""
        return self._obs().char_pos

    def tile(self) -> tuple[int, int]:
        """The tile the character stands on (floor of `me()`)."""
        return self._obs().char_tile

    def inventory(self) -> dict[str, int]:
        """Every item name -> count held (exact up to 200)."""
        return self._obs().inventory()

    def ore_tiles(self, kind: str = "iron-ore") -> list[tuple[int, int]]:
        """Resource tiles within 12 tiles, sorted.

        When `me()` has a whole-number x (or y), the grid cannot tell two
        neighbouring columns (rows) apart and the list may include a tile beside
        the patch that is not ore; any move off the whole number resolves it.
        """
        return self._obs().ore_tiles(kind)

    def blocked_tiles(self) -> list[tuple[int, int]]:
        """Tiles nothing can stand or be built on (water), sorted."""
        return self._obs().blocked_tiles()

    def entities(self) -> list[Entity]:
        """The entity table, nearest first. Rows renumber as the character moves."""
        return self._obs().entities()

    def patch(self) -> tuple[float, float] | None:
        """The task's public target marker (the ore patch's centre), or None."""
        return self._obs().patch()

    def decisions_left(self) -> int:
        return max(0, min(self._budget - self.decisions, self._rl.task.max_steps - self._rl.steps))

    def last_refused(self) -> bool:
        """Whether the game refused the last action (a legal intent that failed)."""
        return self._obs().last_refused()

    # ------------------------------------------------------------ actions

    def move(self, direction: str, stride: str = "long") -> bool:
        """Walk N/E/S/W: "long" 30 ticks (~4.5 tiles), "step" 7 (~1), "nudge" 2 (~0.3)."""
        intent = f"move {direction} {stride}"
        if direction not in MOVES or stride not in STRIDES:
            return self._refuse(intent, "direction must be N/E/S/W, stride long/step/nudge")
        return self._act(intent, (STRIDES[stride] + MOVES[direction], 0, 0, 0, 0, 0))

    def place(self, item: str, x: int, y: int, facing: str) -> bool:
        """Place `item` on tile (x, y); a 2x2 machine covers x..x+1, y..y+1."""
        intent = f"place {item} ({x},{y}) {facing}"
        tx, ty = _whole(x), _whole(y)
        if item not in ITEMS:
            return self._refuse(intent, "unknown item")
        if facing not in FACINGS:
            return self._refuse(intent, "facing must be N/E/S/W")
        if tx is None or ty is None:
            return self._refuse(intent, "tile coordinates must be whole numbers")
        here_x, here_y = self._obs().char_tile
        dx, dy = tx - here_x, ty - here_y
        if abs(dx) > PLACEMENT_RADIUS or abs(dy) > PLACEMENT_RADIUS:
            return self._refuse(intent, f"more than {PLACEMENT_RADIUS} tiles from tile()")
        if self.inventory()[item] <= 0:
            return self._refuse(intent, "not in inventory")
        side = 2 * PLACEMENT_RADIUS + 1
        slot = (dx + PLACEMENT_RADIUS) * side + dy + PLACEMENT_RADIUS + 1
        vector = (OP_PLACE, 0, slot, FACINGS.index(facing) + 1, ITEMS.index(item) + 1, 0)
        ok = self._act(intent, vector, "tile occupied, or it is the character's own tile")
        if ok and not self.last_refused():
            self._built.append((item, tx, ty))
        return ok

    def give(self, entity, item: str, amount: int) -> bool:
        """Move `amount` (1, 5 or 20) of `item` from the inventory into `entity`."""
        return self._transfer(OP_GIVE, "give", entity, item, amount)

    def take(self, entity, item: str, amount: int) -> bool:
        """Move `amount` (1, 5 or 20) of `item` out of `entity` into the inventory."""
        return self._transfer(OP_TAKE, "take", entity, item, amount)

    def mine(self, entity) -> bool:
        """Mine (pick up) `entity`."""
        intent = f"mine {_describe(entity)}"
        row = self._row(entity)
        if row is None:
            return self._refuse(intent, "no such entity in the table")
        return self._act(intent, (OP_MINE, row + 1, 0, 0, 0, 0))

    def wait(self) -> bool:
        """Let one decision (30 ticks) pass."""
        return self._act("wait", WAIT)

    # ------------------------------------------------------------ internals

    def _transfer(self, op: int, verb: str, entity, item: str, amount: int) -> bool:
        intent = f"{verb} {_describe(entity)} {item} x{amount}"
        if item not in ITEMS:
            return self._refuse(intent, "unknown item")
        if amount not in AMOUNTS or isinstance(amount, bool):
            return self._refuse(intent, "amount must be 1, 5 or 20")
        row = self._row(entity)
        if row is None:
            return self._refuse(intent, "no such entity in the table")
        if op == OP_GIVE and self.inventory()[item] <= 0:
            return self._refuse(intent, "not in inventory")
        vector = (op, row + 1, 0, 0, ITEMS.index(item) + 1, AMOUNTS[amount])
        reason = "not held" if op == OP_GIVE else "no visible entity holds that item"
        return self._act(intent, vector, reason)

    def _row(self, entity) -> int | None:
        """The current table row for `entity`: an `Entity` (matched by kind and
        position, so one read before the character moved still resolves) or a row."""
        current = self.entities()
        if isinstance(entity, Entity):
            for e in current:
                if (
                    e.kind == entity.kind
                    and abs(e.x - entity.x) < 1e-3
                    and abs(e.y - entity.y) < 1e-3
                ):
                    return e.row
            rows = {e.row: e for e in current}
            same = rows.get(entity.row)
            return entity.row if same is not None and same.kind == entity.kind else None
        if isinstance(entity, int) and not isinstance(entity, bool):
            return entity if any(e.row == entity for e in current) else None
        return None

    def _refuse(self, intent: str, reason: str) -> bool:
        self.refusals += 1
        self._trace.append(f"{intent} -> refused ({reason})")
        if self.refusals >= MAX_REFUSALS:
            raise RuntimeError(f"stopped after {MAX_REFUSALS} refused intents")
        return False

    def _act(self, intent: str, vector, illegal: str = "illegal") -> bool:
        if self.decisions_left() <= 0 or self._rl.done:
            raise BudgetExhausted(f"no decisions left ({self.decisions} spent)")
        for i in range(6):
            self._vector[i] = vector[i]
        if vector[0] != OP_WAIT and lib.fsim_rl_decode(self._rl, self._vector, self._action):
            return self._refuse(intent, illegal)
        before = self._rl.env.char_pos.x, self._rl.env.char_pos.y
        lib.fsim_rl_step(self._rl, self._vector)
        lib.fsim_rl_encode(self._rl, self._env.obs_c)
        self._view = None
        self.decisions += 1
        env = self._rl.env
        self.walk_distance += (
            math.hypot(env.char_pos.x - before[0], env.char_pos.y - before[1]) / TILE
        )
        self._note_plate()
        failed = self.last_refused() and vector[0] != OP_WAIT
        self.failures += failed
        self._trace.append(f"{intent} -> {'failed' if failed else 'ok'}")
        return True

    def _note_plate(self) -> None:
        env = self._rl.env
        if (
            self._first_plate_tick is None
            and env.produced[lib.IT_IRON_PLATE] > self._baseline_plates
        ):
            self._first_plate_tick = int(env.tick)


def _whole(v) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return None


def _describe(entity) -> str:
    if isinstance(entity, Entity):
        return f"{entity.kind}@({entity.x:g},{entity.y:g})"
    return f"row {entity}"


def run_episode(
    build,
    scene: dict,
    *,
    task: str = "construct_smelting_line",
    decision_budget: int = 600,
    env: RlEnv | None = None,
) -> EpisodeResult:
    """Reset to `scene` under v2, run `build(world)`, then wait out the episode.

    Whatever the program does, the episode runs to its own end afterwards --
    waits, without encoding observations -- so the task's verification window
    runs and scores what was built.
    """
    env = env if env is not None else RlEnv()
    env.reset(task, scene, action_space="v2")
    rl = env.rl
    if rl.task.action_space != lib.ACTION_SPACE_V2:
        raise RuntimeError("the environment did not reset into the v2 action space")
    world = World(env, decision_budget)
    error = None
    try:
        build(world)
    except BudgetExhausted:
        pass
    except Exception as exc:  # the program's own bug: recorded, and the episode still ends
        error = f"{type(exc).__name__}: {exc}"
    wait = ffi.new("int32_t[6]", list(WAIT))
    for _ in range(rl.task.max_steps + 1):
        if rl.done:
            break
        lib.fsim_rl_step(rl, wait)
        world._note_plate()
    return EpisodeResult(
        success=bool(rl.success),
        verified_output=int(rl.verified_output),
        decisions=world.decisions,
        refusals=world.refusals,
        first_plate_tick=world._first_plate_tick,
        walk_distance=world.walk_distance,
        built=list(world._built),
        trace=list(world._trace),
        error=error,
        failures=world.failures,
    )
