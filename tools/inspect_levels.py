"""Look at what a UED run's curriculum actually invented.

`walls_mean` going up says a curriculum got harder. It does not say whether it
got harder in a way that means anything, and a regret-seeking search is very
willing to find levels that are hard because they are broken. This prints the
levels themselves.

    python tools/inspect_levels.py runs/ued-pilot-s1/levels.json --top 5
    python tools/inspect_levels.py runs/ued-pilot-s1/levels.json --summary
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fsim import ued  # noqa: E402


def summarise(buffer: ued.LevelBuffer) -> dict:
    entries = buffer.entries
    origins = Counter(
        entry.level.origin.split(":")[0] if entry.level.origin.startswith("edit")
        else entry.level.origin
        for entry in entries
    )  # fmt: skip
    tiles = [len(entry.level.tiles()) for entry in entries]
    walls = [len(entry.level.walls) for entry in entries]
    return {
        "levels": len(entries),
        "origins": dict(origins.most_common()),
        "walls": {"mean": round(sum(walls) / len(walls), 3), "max": max(walls)},
        "tiles": {"mean": round(sum(tiles) / len(tiles), 2), "min": min(tiles), "max": max(tiles)},
        "score": {
            "mean": round(sum(e.score for e in entries) / len(entries), 5),
            "max": round(max(e.score for e in entries), 5),
        },
        # A level nobody has replayed has never been trained on, whatever its
        # score says.
        "never_replayed": sum(1 for e in entries if e.seen == 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("levels", type=Path)
    parser.add_argument("--top", type=int, default=3, help="highest-scoring levels to draw")
    parser.add_argument("--bottom", type=int, default=0, help="lowest-scoring levels to draw")
    parser.add_argument("--summary", action="store_true", help="counts only, no drawings")
    args = parser.parse_args()
    if not args.levels.exists():
        raise SystemExit(f"no such file: {args.levels}")

    buffer = ued.load_buffer(args.levels)
    if not buffer.entries:
        raise SystemExit("the buffer is empty")

    import json

    print(json.dumps(summarise(buffer), indent=1))
    if args.summary:
        return 0

    order = sorted(buffer.entries, key=lambda e: e.score, reverse=True)
    picks = [("highest regret", entry) for entry in order[: args.top]]
    picks += [("lowest regret", entry) for entry in order[len(order) - args.bottom :]]
    for label, entry in picks:
        level = entry.level
        print(
            f"\n--- {label}  score {entry.score:.5f}  replayed {entry.seen}x  "
            f"origin {level.origin}  patch {level.x_hi - level.x_lo + 1}x"
            f"{level.y_hi - level.y_lo + 1} at ({level.ox}, {level.oy})  "
            f"{len(level.walls)} wall(s) ---"
        )
        print(ued.render(level))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
