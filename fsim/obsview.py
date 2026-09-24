"""The published observation, decoded back into tiles, entities and counts.

A builder program is scored against a policy that sees only the `local-v2`
tensors, so it may see only them too: this module is the whole of what it
knows about the world. It inverts the encoder in `csrc/fsim_rl.c`
(`rl_encode_into`) and reads nothing else.

Most of the encoding inverts exactly. The character's position is stored as
`x / 128` in float32, and a position is a whole number of 1/256 tiles, so the
float holds it without error inside the 128-tile box. Entity offsets are
`dx / 32` of two such positions, exact inside the 32-tile radius and clipped to
+-1 beyond it. Counts are `log1p(n) / log1p(200)`, which float32 separates at
every integer up to the cap of 200.

The grid does not always invert. A resource tile lands in column
`rint(tx + 0.5 - x) + 32`, rounding half to even, so when the character's `x`
is a whole number every value is a half, two neighbouring tiles share an even
column and the odd columns stay empty. Which of the pair is ore is then not in
the observation at all. `ore_tiles` returns both (a superset along that axis)
and `exact()` says which axes were affected. Blocked tiles use `rint(bx - x)`
and are ambiguous in the same way when `x` is a whole number plus a half.
Ore is also only reported within 12 tiles of the character (the simulator's
resource sweep), and entities within its 32-tile sensor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

RADIUS = 32
POSITION_SCALE = 128.0
COUNT_CAP = 200.0
DIRECTIONS16 = 16.0
TYPE_SLOTS = 11.0  # encoders.ENTITY_TYPES has twelve names; index / 11

#: encoders.ITEMS, the inventory vector's order (item argument = index + 1).
ITEMS = (
    "iron-ore", "copper-ore", "coal", "stone", "iron-plate", "copper-plate",
    "stone-furnace", "iron-gear-wheel", "transport-belt", "wood",
    "small-electric-pole", "wooden-chest", "burner-mining-drill", "burner-inserter",
)  # fmt: skip
#: Grid planes 0-3, by resource name.
ORE_PLANES = {"iron-ore": 0, "copper-ore": 1, "coal": 2, "stone": 3}
BLOCKED_PLANE = 5
#: Column 3's type index, as `rl_type_index` writes it.
KINDS = {0: "container", 1: "furnace", 3: "mining-drill", 4: "transport-belt", 5: "inserter",
         10: "wall", 11: "item-entity"}  # fmt: skip
FACINGS = ("N", "E", "S", "W")

_LOG_CAP = math.log1p(COUNT_CAP)


def decode_count(value: float) -> float:
    """A `log1p(n) / log1p(200)` feature back to n (exact up to 200, which it saturates at)."""
    return float(round(math.expm1(float(value) * _LOG_CAP)))


@dataclass(frozen=True)
class Entity:
    """One row of the observation's entity table."""

    row: int  # 0-based; the v2 target argument is row + 1
    kind: str  # "furnace" | "mining-drill" | "container" | "wall" | "item-entity" | "other"
    x: float  # world position in tiles: a 2x2 machine's centre, a 1x1 entity's tile centre
    y: float
    facing: str | None  # drills only; the encoder writes no direction for anything else
    fuel: float
    contents: float
    output: float
    working: bool
    remembered: bool


def _candidates(offset: float, origin: float, half: bool) -> list[int]:
    """The integer coordinates the encoder puts in grid offset `offset`.

    Ore is placed at `rint(t + 0.5 - origin)`, blocked tiles at `rint(t - origin)`.
    The arithmetic here is on the same dyadic values the C does in double, so
    the membership test is exact, not approximate.
    """
    shift = 0.5 if half else 0.0
    low = math.floor(offset + origin - shift - 1)
    return [t for t in range(low, low + 3) if round(t + shift - origin) == offset]


class ObsView:
    """Plain-Python answers about one observation dict (`RlEnv.obs`'s keys).

    The arrays are copied on construction, so a view stays valid after the
    environment steps and overwrites its buffers.
    """

    def __init__(self, obs: dict) -> None:
        if "grid" in obs:
            grid = np.asarray(obs["grid"])
        else:  # a compact observation: bit-packed flags plus an amount byte plane
            from fsim.vec import unpack_grid

            flags = np.asarray(obs["flags"]).reshape(1, -1)
            amount = np.asarray(obs["amount"]).reshape(1, -1)
            grid = unpack_grid(flags, amount)[0]
        if grid.ndim == 4:
            grid = grid[0]
        # Binary planes are 0/1 as float and 0/255 as bytes; either way > half is set.
        threshold = 127 if grid.dtype == np.uint8 else 0.5
        self._planes = grid[[0, 1, 2, 3, BLOCKED_PLANE]] > threshold
        self._entities = np.array(obs["entities"], dtype=np.float64)
        self._mask = np.array(obs["entity_mask"]).astype(bool)
        self._self = np.array(obs["self"], dtype=np.float64)
        self._inventory = np.array(obs["inventory"], dtype=np.float64)
        self._goal = np.array(obs["goal"], dtype=np.float64)
        self.char_pos = (self._self[0] * POSITION_SCALE, self._self[1] * POSITION_SCALE)
        self.char_tile = (math.floor(self.char_pos[0]), math.floor(self.char_pos[1]))

    # ---------------------------------------------------------------- grid

    def exact(self) -> dict[str, tuple[bool, bool]]:
        """Whether each axis of the ore and blocked decodes is exact: {"ore": (x, y), ...}."""
        fx = self.char_pos[0] - math.floor(self.char_pos[0])
        fy = self.char_pos[1] - math.floor(self.char_pos[1])
        return {"ore": (fx != 0.0, fy != 0.0), "blocked": (fx != 0.5, fy != 0.5)}

    def _tiles(self, plane: int, half: bool) -> list[tuple[int, int]]:
        rows, cols = np.nonzero(self._planes[plane])
        if rows.size == 0:
            return []
        ox, oy = self.char_pos
        xs = {c: _candidates(int(c) - RADIUS, ox, half) for c in set(cols.tolist())}
        ys = {r: _candidates(int(r) - RADIUS, oy, half) for r in set(rows.tolist())}
        out = {
            (tx, ty)
            for r, c in zip(rows.tolist(), cols.tolist(), strict=True)
            for tx in xs[c]
            for ty in ys[r]
        }
        return sorted(out)

    def ore_tiles(self, kind: str = "iron-ore") -> list[tuple[int, int]]:
        """Integer tiles holding `kind`, sorted. A superset on an ambiguous axis (see `exact`)."""
        return self._tiles(ORE_PLANES[kind], half=True)

    def blocked_tiles(self) -> list[tuple[int, int]]:
        """Integer tiles marked blocked (water), sorted. A superset on an ambiguous axis."""
        return self._tiles(4, half=False)

    # ---------------------------------------------------------------- rest

    def entities(self) -> list[Entity]:
        """The entity table's rows, in table order (nearest first)."""
        out = []
        ox, oy = self.char_pos
        for i in np.flatnonzero(self._mask).tolist():
            f = self._entities[i]
            kind = KINDS.get(round(f[3] * TYPE_SLOTS), "other")
            facing = None
            if kind == "mining-drill":
                facing = FACINGS[round(f[4] * DIRECTIONS16) // 4 % 4]
            out.append(
                Entity(
                    row=i,
                    kind=kind,
                    x=ox + f[0] * RADIUS,
                    y=oy + f[1] * RADIUS,
                    facing=facing,
                    fuel=decode_count(f[14]),
                    contents=decode_count(f[5]),
                    output=decode_count(f[15]),
                    working=bool(f[12] > 0.5),
                    remembered=bool(f[9] > 0.5),
                )
            )
        return out

    def inventory(self) -> dict[str, int]:
        """Every item in `ITEMS` -> count held (exact up to 200, saturating there)."""
        return {name: int(decode_count(v)) for name, v in zip(ITEMS, self._inventory, strict=True)}

    def last_refused(self) -> bool:
        """Whether the last action the character attempted failed or was cancelled."""
        return bool(self._self[9] > 0.5)

    def patch(self) -> tuple[float, float] | None:
        """The task's public marker (goal slots 9-11), if the task publishes one.

        Exact while the marker is within 32 tiles on each axis; clipped beyond.
        """
        if self._goal[11] < 0.5:
            return None
        return (
            self.char_pos[0] + self._goal[9] * RADIUS,
            self.char_pos[1] + self._goal[10] * RADIUS,
        )
