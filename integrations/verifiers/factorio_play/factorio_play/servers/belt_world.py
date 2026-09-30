"""belt_smelting's tools as an MCP server: factory-sim's `WorldV3`, one scene per rollout.

The belt_smelting counterpart of `servers/world.py`, which serves
construct_smelting_line and is unchanged. The tool bodies are
`factorio_play.belt_tools.BeltTools`; this module adds the MCP surface and
publishes the episode's counters, and after `finish` its verified result,
into the rollout's shared `WorldState`, where the task's reward reads it.
"""

from __future__ import annotations

import inspect

import verifiers.v1 as vf
from pydantic import Field

from evolve import evaluate
from factorio_play.belt_tools import BeltTools
from factorio_play.servers.world import WorldState
from factorio_play.session import WorldSession
from fsim import scenes

TASK = "belt_smelting"


class BeltToolsetConfig(vf.ToolsetConfig):
    decision_budget: int = Field(evaluate.task_setup(TASK).decision_budget, ge=1)
    """Decisions the build phase allows: the task's own 2,500, as for a builder program."""
    call_timeout_s: float = Field(60.0, gt=0)


class BeltWorldToolset(vf.Toolset[BeltToolsetConfig, WorldState]):
    TOOL_PREFIX = None  # tools are advertised bare: move, place, finish, ...

    session: WorldSession | None = None
    tools: BeltTools | None = None

    async def setup_task(self, task) -> None:
        _, blueprint = scenes.sample(task.sim_task, task.sample_split, task.seed)
        self.session = WorldSession(
            blueprint,
            task=task.sim_task,
            decision_budget=self.config.decision_budget,
            call_timeout_s=self.config.call_timeout_s,
        )
        self.tools = BeltTools(self.session)

    # ------------------------------------------------------------ plumbing

    def _publish(self) -> None:
        s, st = self.session, self.state
        st.tool_calls += 1
        for k, v in s.counters().items():
            setattr(st, k, v)
        if s.finished:
            st.finished = True
            if s.result is not None:
                r = s.result
                st.success = bool(r.success)
                st.verified_output = int(r.verified_output)
                st.decisions, st.refusals, st.failures = r.decisions, r.refusals, r.failures
                st.error = r.error
            else:
                st.error = s.harness_error

    def _call(self, name: str, *args) -> str:
        if self.tools is None:
            return '{"ok":false,"error":"no episode (the server has no task)"}'
        out = getattr(self.tools, name)(*args)
        self._publish()
        return out

    # ------------------------------------------------------------ queries

    @vf.tool
    def me(self) -> str:
        """The character's position in tiles, [x, y]. Costs no decision."""
        return self._call("me")

    @vf.tool
    def tile(self) -> str:
        """The tile the character stands on, [x, y] (floor of me). Costs no decision."""
        return self._call("tile")

    @vf.tool
    def inventory(self) -> str:
        """Items held, name -> count (only those held). Costs no decision."""
        return self._call("inventory")

    @vf.tool
    def ore_tiles(self, kind: str = "iron-ore") -> str:
        """Resource tiles of `kind` (iron-ore, coal, stone, copper-ore) within 12 tiles, as
        runs [y, x_first, x_last]: row y holds x_first..x_last. While me() has a whole-number
        coordinate it may include one tile beside the patch. Costs no decision."""
        return self._call("ore_tiles", kind)

    @vf.tool
    def blocked_tiles(self) -> str:
        """Tiles nothing can stand or be built on (water), as runs [y, x_first, x_last].
        Costs no decision."""
        return self._call("blocked_tiles")

    @vf.tool
    def entities(self, kind: str | None = None) -> str:
        """The entity table, nearest first, optionally only one `kind` (furnace, mining-drill,
        container, transport-belt, inserter, wall, item-entity, other). Each row: row, kind,
        x, y, and only the fields that hold something: facing (drills, belts, inserters; an
        inserter faces its pickup), fuel, contents, output, working, remembered, lanes and
        shape (belts), held, pickup, drop (inserters; a drill's drop point), item. Rows
        renumber as the character moves: read it again before using a row. Costs no
        decision."""
        return self._call("entities", kind)

    @vf.tool
    def marker(self, name: str) -> str:
        """A public marker's position [x, y]: "iron" (the iron patch), "coal" (the coal
        patch) or "output" (the output chest); null for any other name. Costs no decision."""
        return self._call("marker", name)

    @vf.tool
    def belt_lanes(self, row: int) -> str:
        """Items on belt `row`'s lane 1 (left of travel) and lane 2, [lane1, lane2]; null if
        the row is not a belt. Costs no decision."""
        return self._call("belt_lanes", row)

    @vf.tool
    def opened(self) -> str:
        """The chest inspect() opened, {"row": ..., "contents": {item: count}}, or null when
        none is open. Costs no decision."""
        return self._call("opened")

    @vf.tool
    def decisions_left(self) -> str:
        """Decisions the build phase still allows."""
        return self._call("decisions_left")

    @vf.tool
    def last_refused(self) -> str:
        """Whether the game refused the last action (a legal intent that failed)."""
        return self._call("last_refused")

    # ------------------------------------------------------------ actions

    @vf.tool
    def move(self, direction: str, stride: str = "long", count: int = 1) -> str:
        """Walk `count` strides (1-50, one decision each) in `direction` (N/E/S/W, or
        north/east/south/west), stride "long" (~4.5 tiles), "step" (~1) or "nudge" (~0.3).
        Stops early after a stride that did not move the character ("blocked"). Replies with
        the position reached."""
        return self._call("move", direction, stride, count)

    @vf.tool
    def place(self, item: str, x: int, y: int, facing: str) -> str:
        """Place `item` (burner-mining-drill, stone-furnace, transport-belt, burner-inserter,
        wooden-chest) on tile (x, y) facing N/E/S/W; a 2x2 machine covers x..x+1, y..y+1. A
        belt carries toward its facing; an inserter faces its pickup and drops on the
        opposite side; a drill drops ahead of its facing. Reaches tiles within 7 of tile()
        whose centre is within 10 of me(), never the character's own tile. One decision.
        Replies with the new entity's row."""
        return self._call("place", item, x, y, facing)

    @vf.tool
    def rotate(self, row: int, reverse: bool = False) -> str:
        """Turn belt, inserter or drill `row` a quarter turn clockwise (anticlockwise if
        `reverse`). One decision. Replies with its new facing."""
        return self._call("rotate", row, reverse)

    @vf.tool
    def give(self, row: int, item: str, amount: int) -> str:
        """Move `amount` (1-200) of `item` from the inventory into entity `row` (coal goes to
        a machine's fuel slot), as transfers of 20, 5 and 1, one decision each. Replies with
        how many of `item` are still held."""
        return self._call("give", row, item, amount)

    @vf.tool
    def take(self, row: int, item: str, amount: int) -> str:
        """Move `amount` (1-200) of `item` out of entity `row` into the inventory, as transfers
        of 20, 5 and 1, one decision each."""
        return self._call("take", row, item, amount)

    @vf.tool
    def take_fuel(self, row: int, amount: int) -> str:
        """Move `amount` (1-200) of the fuel in entity `row`'s fuel slot into the inventory, as
        transfers of 20, 5 and 1, one decision each; what is burning stays in the machine."""
        return self._call("take_fuel", row, amount)

    @vf.tool
    def mine(self, row: int) -> str:
        """Mine (pick up) entity `row`. One decision."""
        return self._call("mine", row)

    @vf.tool
    def inspect(self, row: int) -> str:
        """Open chest `row`, within 10 tiles, and read what it holds by item. One decision.
        It stays open, readable free with opened(), until you walk out of reach or open
        another; a chest's row otherwise shows only its total and its main item."""
        return self._call("inspect", row)

    @vf.tool
    def mine_resource(self, x: int, y: int, amount: int = 1, wait: bool = True) -> str:
        """Hand-mine `amount` (1-200) ore, coal or stone from resource tile (x, y), whose
        centre must be within 2.7 tiles of me(). Each request of 20, 5 or 1 is one decision;
        with `wait` (the default) it then waits, one decision per wait, until those items
        have arrived (2 s, 4 waits, an item), and stops if none arrive for 8 waits. Without
        `wait`, one request (amount 1, 5 or 20): the mining runs on until that many have
        arrived or a move stops it."""
        return self._call("mine_resource", x, y, amount, wait)

    @vf.tool
    def wait(self, count: int = 1) -> str:
        """Let `count` decisions (30 ticks each, 1-100) pass."""
        return self._call("wait", count)

    @vf.tool
    def finish(self) -> str:
        """End the build phase (one decision): the ten-minute verification window runs at
        once and the episode is scored. Call it once the line is built and fuelled. Nothing
        can be done after."""
        return self._call("finish")


# A tool's docstring is its description; send it as one line, without the source's indentation.
for _fn in vars(BeltWorldToolset).values():
    if callable(_fn) and hasattr(_fn, "tool") and _fn.__doc__:
        _fn.__doc__ = " ".join(inspect.cleandoc(_fn.__doc__).split())
del _fn


if __name__ == "__main__":
    BeltWorldToolset.run()
