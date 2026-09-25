"""Build the belt merge-delay table the simulator ships (fsim/data/belt-delay.*).

FactorioRL's tools/probe_logistics4.py (`--family delaymap`) measures, on
Factorio 2.0.60, the merge delay `d` of every tile and both lanes of a
rectangle and writes `runtime/delaymap-<x0>_<y0>_<x1>_<y1>[-f4].json.xz` there
(docs/sim-logistics.md, "Fourth probe"). Each measurement builds a belt at the
tile and rebuilds its downstream neighbour every tick; where the neighbour's
own tile has d = 1 its timer runs out before it is rebuilt and the reading is
1 whatever the tile's d. So every rectangle is measured twice, with the belt
facing north (neighbour at y-1) and facing east (neighbour at x+1), and the
two are combined: equal readings stand; where they differ one of them is 1
and its neighbour's d is 1, and the other is the tile's d. Both readings 1
with both neighbours at 1 would be undecided; the build refuses then.

The result:

- `fsim/data/belt-delay.u16.xz`: xz-compressed little-endian uint16, rectangle
  by rectangle, lane 1's delays then lane 2's, row by row (y, then x);
- `fsim/data/belt-delay.json`: the rectangles, where they came from (engine
  version and build, the method, the probe's command), how many readings the
  combination corrected, and the sha256 of the uncompressed values, which
  `fsim` checks when it loads them.

    uv run python tools/make_belt_delay.py --from ../FactorioRL
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "fsim" / "data"

#: The rectangles (tiles, half-open), in lookup order: the scene area around
#: the origin, then the areas FactorioRL's logistics probes built their rigs in.
RECTS = ((-128, -128, 128, 128), (96, 96, 352, 352), (296, 456, 340, 480))
#: Facing: the neighbour's offset.
NEIGHBOUR = {0: (0, -1), 4: (1, 0)}


def _load(source: Path, rect: tuple, facing: int) -> dict:
    x0, y0, x1, y1 = rect
    tag = "" if facing == 0 else f"-f{facing}"
    path = source / "runtime" / f"delaymap-{x0}_{y0}_{x1}_{y1}{tag}.json.xz"
    doc = json.loads(lzma.decompress(path.read_bytes()))
    assert doc["rect"] == list(rect) and doc["missing"] == 0, (rect, facing, doc["missing"])
    assert doc.get("facing", 0) == facing
    return doc


def combine(source: Path) -> tuple[dict, dict]:
    """{(x, y, lane): d} over every rectangle, and what the combination did."""
    reads: dict = {0: {}, 4: {}}
    docs = []
    for rect in RECTS:
        x0, y0, x1, y1 = rect
        w = x1 - x0
        for facing in (0, 4):
            doc = _load(source, rect, facing)
            docs.append(doc)
            for lane in (1, 2):
                for i, v in enumerate(doc[f"lane{lane}"]):
                    key = (x0 + i % w, y0 + i // w, lane)
                    reads[facing].setdefault(key, v)
    table, fixed = {}, {0: 0, 4: 0}
    for key, n in reads[0].items():
        e = reads[4][key]
        if n == e:
            table[key] = n
        elif n == 1:
            table[key], fixed[0] = e, fixed[0] + 1
        elif e == 1:
            table[key], fixed[4] = n, fixed[4] + 1
        else:
            raise ValueError(f"{key}: north reads {n}, east {e}")
    # Every reading that was overruled had a neighbour at 1 (or outside the
    # table); every 1 kept has a neighbour not at 1 in at least one facing.
    undecided = []
    for (x, y, lane), d in table.items():
        nbs = {}
        for facing, (dx, dy) in NEIGHBOUR.items():
            nbs[facing] = table.get((x + dx, y + dy, lane))
            if reads[facing][(x, y, lane)] != d:
                assert nbs[facing] in (1, None), ((x, y, lane), facing)
        if d == 1 and all(v in (1, None) for v in nbs.values()):
            undecided.append((x, y, lane))
    if undecided:
        raise ValueError(f"undecided (both neighbours at 1): {undecided}")
    engines = {(doc["engine"]["version"], doc["engine"]["build"]) for doc in docs}
    assert len(engines) == 1
    ((version, build),) = engines
    info = {
        "engine": {"version": version, "build": build},
        "methods": sorted({doc["method"] for doc in docs}),
        "corrected": {"north_read_1": fixed[0], "east_read_1": fixed[4]},
        "wall_seconds": sum(doc["wall_seconds"] for doc in docs),
    }
    return table, info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="source", required=True, type=Path)
    args = parser.parse_args()
    table, info = combine(args.source)
    raw = bytearray()
    for x0, y0, x1, y1 in RECTS:
        for lane in (1, 2):
            values = [table[(x, y, lane)] for y in range(y0, y1) for x in range(x0, x1)]
            assert all(1 <= v <= 65535 for v in values)
            raw += struct.pack(f"<{len(values)}H", *values)
    digest = hashlib.sha256(bytes(raw)).hexdigest()
    packed = lzma.compress(bytes(raw), preset=9 | lzma.PRESET_EXTREME)
    (DATA / "belt-delay.u16.xz").write_bytes(packed)
    meta = {
        "what": "belt merge delay d (ticks) of every tile and lane: a belt's lane line merges "
                "with its neighbours' d ticks after it is built",
        "engine": info["engine"],
        "method": "each tile measured twice, facing north and facing east: a belt at the tile, "
                  "its downstream neighbour destroyed and rebuilt every tick, the first on_tick "
                  "t (from the build command) at which their lines are equal; where the two "
                  "differ, the one reading 1 had a neighbour whose own d is 1 and the other "
                  "stands",
        "methods": info["methods"],
        "corrected": info["corrected"],
        "probe": "FactorioRL tools/probe_logistics4.py --family delaymap --rect x0,y0,x1,y1 "
                 "[--facing 4]",
        "layout": "uint16 little-endian; per rectangle lane 1 then lane 2, rows y0..y1-1, "
                  "each x0..x1-1; rectangles in this order, the first containing a tile wins",
        "rects": [{"rect": list(r)} for r in RECTS],
        "values": len(raw) // 2,
        "sha256": digest,
    }  # fmt: skip
    (DATA / "belt-delay.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(f"{len(raw) // 2} values, corrected {info['corrected']}, sha256 {digest[:16]}..., "
          f"{(DATA / 'belt-delay.u16.xz').stat().st_size // 1024} KiB")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
