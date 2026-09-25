"""Belt-heavy throughput: world ticks per second on a scene that is mostly belts.

Six lines of 45 belts, each loaded at its tail by an inserter from a chest,
sideloaded half way by a six-belt feed that has its own inserter, and emptied
by inserters half way and at the end; two closed 6 x 4 loops, each with an
inserter dropping onto it. About 390 entities, 330 of them belts, with items
flowing through merged segments, boundaries, sideloads and loops. The first
700 ticks (every merge delay has run out by 600) are not timed.

Reported: world ticks per second, and belt-lane ticks per second (belt lanes
times ticks: what the segment code is paid per).

    uv run python bench/bench_belts.py [--ticks 20000] [--repeat 3]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsim import ITEM_IDS, Sim, lib  # noqa: E402

N, E, S, W = 0, 4, 8, 12


def build() -> tuple[Sim, int]:
    sim = Sim(water=[])
    sim.reset({"character": {"position": [0.5, 0.5]}})
    env = sim.env
    belts = 0

    def add(kind, x, y, d=N):
        nonlocal belts
        i = lib.fsim_add_entity(env, kind, x * 256 + 128, y * 256 + 128, d)
        assert i >= 0, (kind, x, y)
        belts += kind == lib.K_BELT
        return i

    def chest(x, y, item=None):
        c = add(lib.K_CHEST, x, y)
        if item:
            for _ in range(16):
                lib.fsim_entity_insert(env, c, ITEM_IDS[item], 100)
        return c

    def inserter(x, y, d):
        i = add(lib.K_INSERTER, x, y, d)
        lib.fsim_entity_insert(env, i, ITEM_IDS["coal"], 50)

    for row in range(6):
        y = -100 + 12 * row
        x0 = -60
        for k in range(45):
            add(lib.K_BELT, x0 + k, y, E)
        chest(x0, y + 2, "iron-plate")  # tail: drops on lane 1
        inserter(x0, y + 1, S)
        chest(x0 + 30, y - 2)  # half way: takes
        inserter(x0 + 30, y - 1, S)
        chest(x0 + 44, y - 2)  # the end: takes
        inserter(x0 + 44, y - 1, S)
        for j in range(6, 0, -1):  # feed from the south at k=20
            add(lib.K_BELT, x0 + 20, y + j, N)
        chest(x0 + 21, y + 6, "copper-plate")
        inserter(x0 + 21, y + 5, E)  # drops on the feed's lane 1
    for loop in range(2):
        ox, oy = 20 + 10 * loop, -100
        w, h = 6, 4
        path = [(x, 0, E) for x in range(w - 1)] + [(w - 1, y, S) for y in range(h - 1)]
        path += [(x, h - 1, W) for x in range(w - 1, 0, -1)]
        path += [(0, y, N) for y in range(h - 1, 0, -1)]
        for x, y, d in path:
            add(lib.K_BELT, ox + x, oy + y, d)
        chest(ox + 2, oy - 2, "iron-plate")
        inserter(ox + 2, oy - 1, N)
    assert env.belt_delay_missing == 0
    lib.fsim_refresh(env)
    return sim, belts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ticks", type=int, default=20000)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()
    best = 0.0
    for _ in range(args.repeat):
        sim, belts = build()
        lib.fsim_advance(sim.env, 700)
        start = time.perf_counter()
        lib.fsim_advance(sim.env, args.ticks)
        rate = args.ticks / (time.perf_counter() - start)
        best = max(best, rate)
        print(f"  {rate:,.0f} ticks/s")
    print(f"belts: {belts}, lanes: {2 * belts}")
    print(f"best: {best:,.0f} ticks/s, {best * 2 * belts / 1e6:,.1f} M belt-lane ticks/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
