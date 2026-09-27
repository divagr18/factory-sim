"""belt_smelting's tools: factory-sim's `WorldV3`, one tool call at a time, with terse replies.

An expert build of belt_smelting takes about 275 to 370 decisions, some 160
of them waiting on hand-mined coal. One tool call per decision would make a
long, expensive episode, so the tools that take a count or an amount run
several world actions in one call, on the episode thread (`WorldSession.run`):

- `move(direction, stride, count)`: up to `count` strides, stopping after a
  stride that does not move the character;
- `wait(count)`: `count` waits;
- `mine_resource(x, y, amount)`: `amount` split into requests of 20, 5 and 1
  (the amounts `WorldV3.mine_resource` takes), each followed by waits until
  the items have arrived, as `fsim.belt_expert` mines its coal;
- `give`, `take`, `take_fuel`: `amount` split into transfers of 20, 5 and 1.

Each world action inside is one `WorldV3` method call, so the world counts
decisions, refusals and failures exactly as it does for a program. Nothing
here plans: which tiles to walk and build on stays the agent's problem.

Replies are compact JSON: a query answers with its value, an action with an
object whose `ok` says whether every step went through. No `verifiers`
import; the tool server (`servers/belt_world.py`) wraps this.
"""

from __future__ import annotations

import json
import math

from factorio_play.session import _normalise
from fsim.belt_expert import TARGET_PLATES

#: The amounts a v3 transfer or hand-mining request takes, largest first.
STEPS = (20, 5, 1)
MAX_AMOUNT = 200  # the inventory reads exactly up to 200
MAX_MOVES = 50
MAX_WAITS = 100
#: Hand-mining yields one item every 2 s, 4 decisions. `fsim.belt_expert` waits
#: at most 4 per item plus 8 for a request; a request also stops after this
#: many waits with nothing new, which the belt expert never needs.
IDLE_WAITS = 8
#: A stride that moves the character less than this did not move it.
STALL = 0.02
ORE_KINDS = ("iron-ore", "copper-ore", "coal", "stone")
FAILED = "the game refused this action; it still cost a decision"
#: Placed item -> the kind its entity row shows, and where its centre sits
#: relative to the tile named in `place` (a 2x2 machine covers x..x+1, y..y+1).
PLACED = {
    "burner-mining-drill": ("mining-drill", 1.0),
    "stone-furnace": ("furnace", 1.0),
    "transport-belt": ("transport-belt", 0.5),
    "burner-inserter": ("inserter", 0.5),
    "wooden-chest": ("container", 0.5),
}


def score(verified_output: int | float) -> float:
    """factory-sim's belt_smelting reward: the verified plates over the 150 target, capped at 1."""
    return min(1.0, float(verified_output) / TARGET_PLATES)


def split(amount: int) -> list[int]:
    """`amount` as requests of 20, 5 and 1, largest first (23 -> 20, 1, 1, 1)."""
    out = []
    for step in STEPS:
        while amount >= step:
            out.append(step)
            amount -= step
    return out


# ------------------------------------------------------------------ formatting


def _num(v):
    if isinstance(v, bool) or not isinstance(v, float):
        return v
    v = round(v, 2)
    return int(v) if v.is_integer() else v


def _point(p):
    return None if p is None else [_num(float(p[0])), _num(float(p[1]))]


def dumps(value) -> str:
    return json.dumps(value, separators=(",", ":"))


def runs(tiles) -> list[list[int]]:
    """Sorted (x, y) tiles as [y, x_first, x_last] runs of consecutive x in one row."""
    out: list[list[int]] = []
    for x, y in sorted(tiles, key=lambda t: (t[1], t[0])):
        if out and out[-1][0] == y and out[-1][2] == x - 1:
            out[-1][2] = x
        else:
            out.append([y, x, x])
    return out


def entity_row(e) -> dict:
    """One `EntityV3` without the fields that hold nothing."""
    d = {"row": e.row, "kind": e.kind, "x": _num(e.x), "y": _num(e.y)}
    if e.facing:
        d["facing"] = e.facing
    for name in ("fuel", "contents", "output"):
        if getattr(e, name):
            d[name] = int(getattr(e, name))
    if e.working:
        d["working"] = True
    if e.remembered:
        d["remembered"] = True
    if e.kind == "transport-belt":
        d["lanes"] = list(e.lanes)
        d["shape"] = e.shape
    for name in ("held", "item"):
        if getattr(e, name):
            d[name] = getattr(e, name)
    for name in ("pickup", "drop"):
        if getattr(e, name) is not None:
            d[name] = _point(getattr(e, name))
    return d


# ---------------------------------------------------- on the episode thread


def _refusal(world) -> str:
    last = world._trace[-1] if world._trace else ""
    note = last.split("-> refused", 1)[1].strip() if "-> refused" in last else "refused"
    if note.startswith("(") and note.endswith(")"):
        note = note[1:-1]
    return note


def _act(world, method: str, *args) -> tuple[str, str | None]:
    """One world action: ("ok", None), ("refused", why) or ("failed", why)."""
    refusals, failures = world.refusals, world.failures
    getattr(world, method)(*args)
    if world.refusals > refusals:
        return "refused", _refusal(world)
    if world.failures > failures:
        return "failed", FAILED
    return "ok", None


def _reply(world, status: str, reason: str | None, start: int | None = None, **extra) -> dict:
    out: dict = {"ok": status == "ok"}
    if status in ("refused", "failed"):
        out[status] = reason
    out.update(extra)
    if start is not None:
        out["decisions"] = world.decisions - start
    out["left"] = world.decisions_left()
    return out


def _entity(world, row):
    """The entity in `row` (so later steps find it by kind and position), else `row`."""
    return next((e for e in world.entities() if e.row == row), row)


def _move(world, direction, stride, count):
    direction, stride = _normalise("move", (direction, stride))
    start, status, reason, blocked = world.decisions, "ok", None, False
    for _ in range(count):
        before = world.me()
        status, reason = _act(world, "move", direction, stride)
        if status != "ok":
            break
        if math.dist(before, world.me()) < STALL:
            blocked = True
            break
    extra = {"me": _point(world.me())}
    if blocked:
        extra["blocked"] = True
    return _reply(world, status, reason, start, **extra)


def _wait(world, count):
    start = world.decisions
    for _ in range(count):
        world.wait()
    return _reply(world, "ok", None, start)


def _place(world, item, x, y, facing):
    (item, x, y, facing) = _normalise("place", (item, x, y, facing))
    status, reason = _act(world, "place", item, x, y, facing)
    if status != "ok":
        return _reply(world, status, reason)
    kind, offset = PLACED.get(item, (None, 0.0))
    centre = (x + offset, y + offset)
    row = next(
        (
            e.row
            for e in world.entities()
            if e.kind == kind and abs(e.x - centre[0]) < 0.1 and abs(e.y - centre[1]) < 0.1
        ),
        None,
    )
    return _reply(world, status, reason, row=row)


def _rotate(world, row, reverse):
    target = _entity(world, row)
    status, reason = _act(world, "rotate", target, bool(reverse))
    extra = {}
    if status == "ok" and not isinstance(target, int):
        now = next((e for e in world.entities() if _same(e, target)), None)
        if now is not None:
            extra["facing"] = now.facing
    return _reply(world, status, reason, **extra)


def _same(a, b) -> bool:
    return a.kind == b.kind and abs(a.x - b.x) < 1e-3 and abs(a.y - b.y) < 1e-3


def _entities(world, kind):
    return [entity_row(e) for e in world.entities() if kind is None or e.kind == kind]


def _mine(world, row):
    status, reason = _act(world, "mine", row)
    return _reply(world, status, reason)


def _transfer(world, method, row, item, amount):
    target = _entity(world, row)
    start, status, reason = world.decisions, "ok", None
    for part in split(amount):
        args = (target, part) if method == "take_fuel" else (target, item, part)
        status, reason = _act(world, method, *args)
        if status != "ok":
            break
    extra = {"held": world.inventory().get(item, 0)} if item else {}
    return _reply(world, status, reason, start, **extra)


def _mine_resource(world, x, y, amount, wait):
    start = world.decisions
    if not wait:
        status, reason = _act(world, "mine_resource", x, y, amount)
        return _reply(world, status, reason, start)
    tile = (x, y)
    kinds = [k for k in ORE_KINDS if tile in {tuple(t) for t in world.ore_tiles(k)}]

    def held() -> int:
        inv = world.inventory()
        return sum(inv.get(k, 0) for k in kinds) if kinds else sum(inv.values())

    got, status, reason, stopped = 0, "ok", None, None
    for batch in split(amount):
        base = held()
        status, reason = _act(world, "mine_resource", x, y, batch)
        if status != "ok":
            break
        last, idle = base, 0
        for _ in range(4 * batch + 8):
            if held() - base >= batch:
                break
            world.wait()
            now = held()
            idle = 0 if now > last else idle + 1
            last = now
            if idle >= IDLE_WAITS:
                break
        got += max(0, held() - base)
        if held() - base < batch:
            stopped = "stopped: no item arrived for a while (the tile, the reach or the inventory)"
            break
    extra: dict = {"got": got}
    if kinds:
        extra["item"] = kinds[0] if len(kinds) == 1 else kinds
    if stopped and status == "ok":
        return _reply(world, "stopped", None, start, stopped=stopped, **extra)
    return _reply(world, status, reason, start, **extra)


# ------------------------------------------------------------------ the tools


def _whole_amount(amount, low: int, high: int) -> str | None:
    if isinstance(amount, bool) or not isinstance(amount, int) or not low <= amount <= high:
        return f"must be a whole number from {low} to {high}"
    return None


class BeltTools:
    """The belt_smelting tool bodies over one `WorldSession`; each returns the reply text."""

    def __init__(self, session) -> None:
        self.session = session

    def _query(self, name: str, *args, shape=None) -> str:
        out = self.session.call(name, *args)
        if not out["ok"]:
            return dumps({"ok": False, "error": out["error"]})
        value = out["result"]
        return dumps(shape(value) if shape else value)

    def _run(self, fn, *args) -> str:
        return dumps(self.session.run(fn, *args))

    @staticmethod
    def _error(message: str) -> str:
        return dumps({"ok": False, "error": message})

    # ---- queries ----------------------------------------------------------

    def me(self) -> str:
        return self._query("me", shape=_point)

    def tile(self) -> str:
        return self._query("tile", shape=list)

    def inventory(self) -> str:
        return self._query("inventory", shape=lambda inv: {k: v for k, v in inv.items() if v})

    def ore_tiles(self, kind: str = "iron-ore") -> str:
        if kind not in ORE_KINDS:
            return self._error(f"kind must be one of {', '.join(ORE_KINDS)}")
        return self._query("ore_tiles", kind, shape=runs)

    def blocked_tiles(self) -> str:
        return self._query("blocked_tiles", shape=runs)

    def entities(self, kind: str | None = None) -> str:
        return self._run(_entities, kind)

    def marker(self, name: str) -> str:
        return self._query("marker", name, shape=_point)

    def belt_lanes(self, row: int) -> str:
        return self._query("belt_lanes", row, shape=lambda v: None if v is None else list(v))

    def decisions_left(self) -> str:
        return self._query("decisions_left")

    def last_refused(self) -> str:
        return self._query("last_refused")

    # ---- actions ----------------------------------------------------------

    def move(self, direction: str, stride: str = "long", count: int = 1) -> str:
        if err := _whole_amount(count, 1, MAX_MOVES):
            return self._error(f"count {err}")
        return self._run(_move, direction, stride, count)

    def place(self, item: str, x: int, y: int, facing: str) -> str:
        return self._run(_place, item, x, y, facing)

    def rotate(self, row: int, reverse: bool = False) -> str:
        return self._run(_rotate, row, reverse)

    def give(self, row: int, item: str, amount: int) -> str:
        if err := _whole_amount(amount, 1, MAX_AMOUNT):
            return self._error(f"amount {err}")
        return self._run(_transfer, "give", row, item, amount)

    def take(self, row: int, item: str, amount: int) -> str:
        if err := _whole_amount(amount, 1, MAX_AMOUNT):
            return self._error(f"amount {err}")
        return self._run(_transfer, "take", row, item, amount)

    def take_fuel(self, row: int, amount: int) -> str:
        if err := _whole_amount(amount, 1, MAX_AMOUNT):
            return self._error(f"amount {err}")
        return self._run(_transfer, "take_fuel", row, None, amount)

    def mine(self, row: int) -> str:
        return self._run(_mine, row)

    def mine_resource(self, x: int, y: int, amount: int = 1, wait: bool = True) -> str:
        if wait:
            if err := _whole_amount(amount, 1, MAX_AMOUNT):
                return self._error(f"amount {err}")
        elif amount not in STEPS:
            return self._error("without wait, amount must be 1, 5 or 20")
        return self._run(_mine_resource, x, y, amount, bool(wait))

    def wait(self, count: int = 1) -> str:
        if err := _whole_amount(count, 1, MAX_WAITS):
            return self._error(f"count {err}")
        return self._run(_wait, count)

    def finish(self) -> str:
        r = self.session.finish()
        if r is None:
            return self._error(self.session.harness_error or "no result")
        return dumps(
            {
                "ok": True,
                "success": bool(r.success),
                "verified_output": int(r.verified_output),
                "score": _num(score(r.verified_output)),
            }
        )
