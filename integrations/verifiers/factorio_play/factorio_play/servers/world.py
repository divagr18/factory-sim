"""The `world` API as MCP tools: one factory-sim episode per rollout.

Task-scoped (constructed in `Task.toolsets`), so every rollout gets its own
server process, its own `RlEnv` and its own scene, fetched through
`setup_task`. After every call the server publishes the episode's counters,
and after `finish` its verified result, into the rollout's shared
`WorldState`, which is where the task's reward reads it.
"""

from __future__ import annotations

import verifiers.v1 as vf
from pydantic import Field

from factorio_play.session import WorldSession
from fsim import scenes

MAX_WAITS = 100


class WorldState(vf.State):
    finished: bool = False
    success: bool = False
    verified_output: int = 0
    decisions: int = 0
    refusals: int = 0
    failures: int = 0
    tool_calls: int = 0
    error: str | None = None


class WorldToolsetConfig(vf.ToolsetConfig):
    decision_budget: int = Field(600, ge=1)
    """Decisions the build phase allows, as for a builder program."""
    call_timeout_s: float = Field(60.0, gt=0)


class WorldToolset(vf.Toolset[WorldToolsetConfig, WorldState]):
    TOOL_PREFIX = None  # tools are advertised bare: move, place, finish, ...

    session: WorldSession | None = None

    async def setup_task(self, task) -> None:
        _, blueprint = scenes.sample(task.sim_task, task.sample_split, task.seed)
        self.session = WorldSession(
            blueprint,
            task=task.sim_task,
            decision_budget=self.config.decision_budget,
            call_timeout_s=self.config.call_timeout_s,
        )

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

    def _call(self, name: str, *args) -> dict:
        if self.session is None:
            return {"ok": False, "error": "no episode (the server has no task)"}
        out = self.session.call(name, *args)
        self._publish()
        return out

    # ------------------------------------------------------------ queries

    @vf.tool
    def me(self) -> dict:
        """The character's position in tiles, [x, y]. Costs no decision."""
        return self._call("me")

    @vf.tool
    def tile(self) -> dict:
        """The tile the character stands on, [x, y] (floor of me). Costs no decision."""
        return self._call("tile")

    @vf.tool
    def inventory(self) -> dict:
        """Every item name -> count held. Costs no decision."""
        return self._call("inventory")

    @vf.tool
    def ore_tiles(self, kind: str = "iron-ore") -> dict:
        """Resource tiles of `kind` within 12 tiles, sorted. Costs no decision."""
        return self._call("ore_tiles", kind)

    @vf.tool
    def blocked_tiles(self) -> dict:
        """Tiles nothing can stand or be built on (water), sorted. Costs no decision."""
        return self._call("blocked_tiles")

    @vf.tool
    def entities(self) -> dict:
        """The entity table, nearest first: row, kind, x, y, facing, fuel, contents,
        output, working, remembered. Rows renumber as the character moves, so read it
        again before give/take/mine. Costs no decision."""
        return self._call("entities")

    @vf.tool
    def patch(self) -> dict:
        """The ore patch's centre [x, y] (the task's public marker), or null."""
        return self._call("patch")

    @vf.tool
    def decisions_left(self) -> dict:
        """Decisions the build phase still allows."""
        return self._call("decisions_left")

    @vf.tool
    def last_refused(self) -> dict:
        """Whether the game refused the last action (a legal intent that failed)."""
        return self._call("last_refused")

    # ------------------------------------------------------------ actions

    @vf.tool
    def move(self, direction: str, stride: str = "long") -> dict:
        """Walk one decision in `direction` (N/E/S/W, or north/east/south/west) with
        stride "long" (~4.5 tiles), "step" (~1) or "nudge" (~0.3). If it is refused,
        the result says why."""
        return self._call("move", direction, stride)

    @vf.tool
    def place(self, item: str, x: int, y: int, facing: str) -> dict:
        """Place `item` on tile (x, y) facing N/E/S/W (or north/east/south/west); a
        2x2 machine covers x..x+1, y..y+1. Reaches 5 tiles from tile(), never the
        character's own tile. If it is refused, the result says why."""
        return self._call("place", item, x, y, facing)

    @vf.tool
    def give(self, row: int, item: str, amount: int) -> dict:
        """Move `amount` (1, 5 or 20) of `item` from the inventory into entity `row`."""
        return self._call("give", row, item, amount)

    @vf.tool
    def take(self, row: int, item: str, amount: int) -> dict:
        """Move `amount` (1, 5 or 20) of `item` out of entity `row` into the inventory."""
        return self._call("take", row, item, amount)

    @vf.tool
    def mine(self, row: int) -> dict:
        """Mine (pick up) entity `row`."""
        return self._call("mine", row)

    @vf.tool
    def wait(self, count: int = 1) -> dict:
        """Let `count` decisions (30 ticks each, at most 100) pass."""
        out = {"ok": True, "result": 0}
        for _ in range(max(1, min(int(count), MAX_WAITS))):
            out = self._call("wait")
            if not out["ok"]:
                break
        return out

    @vf.tool
    def finish(self) -> dict:
        """End the build phase. The episode then runs its verification window and is
        scored; call this once the line is built and fuelled. Nothing can be done after."""
        if self.session is None:
            return {"ok": False, "error": "no episode (the server has no task)"}
        r = self.session.finish()
        self._publish()
        if r is None:
            return {"ok": False, "error": self.session.harness_error}
        return {"ok": True, "success": r.success, "verified_output": r.verified_output}


if __name__ == "__main__":
    WorldToolset.run()
