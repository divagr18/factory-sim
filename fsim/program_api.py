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
`world.refusals`. Legality is the backend's own check -- `fsim_rl_decode` in the
simulator, the action mask and decoder on the engine -- so a legal intent here
is never a decode failure there. An intent
that decodes but that the game then refuses (a drill onto a tile something
else covers) does cost its decision, as it would for a policy; `last_refused()`
reports it.

The evaluator side of an episode (success, the verified output, when the first
plate appeared, how far the character walked) may read simulator internals.
The program never can: it holds only a `World`.

`World` talks to a backend: `SimBackend` here, or anything with the same five
members (`obs`, `steps_left`, `done`, `legal`, `step`, plus `plate_tick` and
`finish` for the evaluator). FactorioGym's `tools/program_transfer.py` supplies
one over the real game, and `play` runs a program on either.
"""

from __future__ import annotations

import ctypes
import math
import threading
from collections import deque
from dataclasses import dataclass

from fsim import ffi, lib
from fsim.obsview import FACINGS, ITEMS, Entity, ObsView
from fsim.rl import RlEnv

__all__ = [
    "BudgetExhausted",
    "Entity",
    "EpisodeResult",
    "SimBackend",
    "World",
    "play",
    "run_episode",
]

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


class ProgramTimeLimit(Exception):
    """Raised inside a program that ran past its wall-clock limit.

    The decision budget does not bound a program's time: queries cost no
    decision, so a `while` loop that only reads `world.entities()` -- or a
    search that never terminates without touching `world` at all -- runs
    forever. One such program held evaluation workers at 100% CPU for most of
    an hour, stalling whole evolution runs. The sandbox bans `try`, so a
    program cannot catch this."""


class _Watchdog:
    """Raises `ProgramTimeLimit` in the calling thread after `seconds`, unless left first.

    An asynchronous exception, so it interrupts pure-Python loops between
    bytecodes, which no call counter could. It is armed only around the
    program, never around the simulator's own stepping."""

    def __init__(self, seconds: float | None):
        self.seconds = seconds
        self.fired = False
        self._tid = threading.get_ident()
        self._lock = threading.Lock()
        self._armed = False
        self._timer: threading.Timer | None = None

    def _raise_in(self, exc) -> None:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(self._tid), exc)

    def _fire(self) -> None:
        with self._lock:
            if self._armed:
                self.fired = True
                self._raise_in(ctypes.py_object(ProgramTimeLimit))

    def __enter__(self):
        if self.seconds is not None:
            self._armed = True
            self._timer = threading.Timer(self.seconds, self._fire)
            self._timer.daemon = True
            self._timer.start()
        return self

    def __exit__(self, *exc) -> bool:
        if self._timer is not None:
            with self._lock:
                self._armed = False
            self._timer.cancel()
            # An exception set but not yet raised would surface later, somewhere
            # outside the program: clear it.
            self._raise_in(None)
        return False


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


class SimBackend:
    """A reset `RlEnv` under v2, as `World` drives it: the published observation,
    `fsim_rl_decode` for legality, and one `fsim_rl_step` per decision."""

    def __init__(self, env: RlEnv) -> None:
        self.env = env
        self.rl = env.rl
        self._action = ffi.new("fsim_action *")
        self._vector = ffi.new("int32_t[6]")
        self._baseline_plates = self.rl.env.produced[lib.IT_IRON_PLATE]

    @property
    def obs(self) -> dict:
        return self.env.obs

    @property
    def done(self) -> bool:
        return bool(self.rl.done)

    def steps_left(self) -> int:
        return self.rl.task.max_steps - self.rl.steps

    def legal(self, vector) -> bool:
        for i in range(6):
            self._vector[i] = vector[i]
        return vector[0] == OP_WAIT or not lib.fsim_rl_decode(self.rl, self._vector, self._action)

    def step(self, vector) -> float:
        """One decision; returns the tiles the character moved."""
        for i in range(6):
            self._vector[i] = vector[i]
        env = self.rl.env
        before = env.char_pos.x, env.char_pos.y
        lib.fsim_rl_step(self.rl, self._vector)
        lib.fsim_rl_encode(self.rl, self.env.obs_c)
        return math.hypot(env.char_pos.x - before[0], env.char_pos.y - before[1]) / TILE

    def plate_tick(self) -> int | None:
        env = self.rl.env
        if env.produced[lib.IT_IRON_PLATE] > self._baseline_plates:
            return int(env.tick)
        return None

    def finish(self, on_step) -> tuple[bool, int]:
        """Wait out the episode -- without encoding observations -- so the task's
        verification window runs; `on_step` after each wait."""
        wait = ffi.new("int32_t[6]", list(WAIT))
        for _ in range(self.rl.task.max_steps + 1):
            if self.rl.done:
                break
            lib.fsim_rl_step(self.rl, wait)
            on_step()
        return bool(self.rl.success), int(self.rl.verified_output)


class World:
    """What a builder program holds: observation queries and one-decision actions."""

    def __init__(self, env, decision_budget: int = 600) -> None:
        """`env` is a reset `RlEnv`, or any backend with `SimBackend`'s members."""
        self._backend = env if hasattr(env, "legal") else SimBackend(env)
        if isinstance(self._backend, SimBackend):
            self._env, self._rl = self._backend.env, self._backend.rl
        self._budget = decision_budget
        self._view: ObsView | None = None
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
            self._view = ObsView(self._backend.obs)
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
        return max(0, min(self._budget - self.decisions, self._backend.steps_left()))

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
        if self.decisions_left() <= 0 or self._backend.done:
            raise BudgetExhausted(f"no decisions left ({self.decisions} spent)")
        if not self._backend.legal(vector):
            return self._refuse(intent, illegal)
        self.walk_distance += self._backend.step(vector)
        self._view = None
        self.decisions += 1
        self._note_plate()
        failed = self.last_refused() and vector[0] != OP_WAIT
        self.failures += failed
        self._trace.append(f"{intent} -> {'failed' if failed else 'ok'}")
        return True

    def _note_plate(self) -> None:
        if self._first_plate_tick is None:
            self._first_plate_tick = self._backend.plate_tick()


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
    time_limit_s: float | None = None,
) -> EpisodeResult:
    """Reset to `scene` under v2, run `build(world)`, then wait out the episode.

    `time_limit_s` bounds the program's wall-clock time; past it the program is
    stopped, recorded as an error, and the episode still runs to its end.

    Whatever the program does, the episode runs to its own end afterwards --
    waits, without encoding observations -- so the task's verification window
    runs and scores what was built.
    """
    env = env if env is not None else RlEnv()
    env.reset(task, scene, action_space="v2")
    if env.rl.task.action_space != lib.ACTION_SPACE_V2:
        raise RuntimeError("the environment did not reset into the v2 action space")
    return play(build, SimBackend(env), decision_budget=decision_budget, time_limit_s=time_limit_s)


def play(
    build, backend, *, decision_budget: int = 600, time_limit_s: float | None = None
) -> EpisodeResult:
    """Run `build(world)` on a reset backend, then wait out the episode and score it."""
    world = World(backend, decision_budget)
    error = None
    try:
        with _Watchdog(time_limit_s):
            build(world)
    except ProgramTimeLimit:
        error = f"ProgramTimeLimit: ran past {time_limit_s:g} s"
    except BudgetExhausted:
        pass
    except Exception as exc:  # the program's own bug: recorded, and the episode still ends
        error = f"{type(exc).__name__}: {exc}"
    success, verified_output = backend.finish(world._note_plate)
    return EpisodeResult(
        success=success,
        verified_output=verified_output,
        decisions=world.decisions,
        refusals=world.refusals,
        first_plate_tick=world._first_plate_tick,
        walk_distance=world.walk_distance,
        built=list(world._built),
        trace=list(world._trace),
        error=error,
        failures=world.failures,
    )
