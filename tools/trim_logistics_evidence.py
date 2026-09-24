"""Cut FactorioRL's logistics probe evidence down to what the tests read.

`docs/evidence/sim-mechanics-m4-logistics.json` (FactorioRL, written by
`tools/probe_logistics.py`) samples seventeen rigs every tick for 3,000 ticks,
delta-encoded, about 5 MB. tests/test_mechanics_logistics.py compares the rigs
the simulator reproduces; this keeps those, drops the engine's belt item ids
(the simulator numbers its own), and stores each rig as the ticks on which it
changed. The result is tests/golden/sim-mechanics-m4-logistics.json.xz.

    uv run python tools/trim_logistics_evidence.py --from ../FactorioRL
"""

from __future__ import annotations

import argparse
import json
import lzma
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "tests" / "golden" / "sim-mechanics-m4-logistics.json.xz"

#: Rigs whose engine readings the simulator is held to. Left out: `bend`,
#: `flow`, `flow2`, `same` and `tick_*`, where an inserter takes items still
#: moving on a belt (not modelled; see inserter_belt_pickup in csrc/fsim.c).
KEYS = (
    "straight", "curve", "side_main", "side_feed", "drill", "drill_belt", "c2c", "c2c_dst",
    "fout", "fout_furnace", "fout_belt", "fill_ore", "fill_ore_furnace", "fill_coal",
    "fill_coal_furnace", "fuel", "fuel_dst", "wood", "wood_dst", "self", "hot", "order_a",
    "order_b", "drop1", "drop2", "drop3",
) + tuple(f"phase{i}" for i in range(1, 17))  # fmt: skip


def undiff(before, change):
    """The probe's `undiff`: apply one delta-encoded sample."""
    out = dict(before)
    for key in change.get("_gone", []):
        out.pop(key, None)
    for key, value in change.items():
        if key == "_gone":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = undiff(out[key], value)
        else:
            out[key] = value
    return out


def is_belts(value) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(b, list) and len(b) == 2 for b in value)
        and all(isinstance(lane, (list, dict)) for b in value for lane in b)
    )


def normalise(value):
    """Belts as [[name, position], ...] per lane, by position; empty Lua tables
    as the lists they are; ids dropped."""
    if is_belts(value):
        return [
            [sorted([item[0], item[1]] for item in (lane if isinstance(lane, list) else []))
             for lane in belt]
            for belt in value
        ]  # fmt: skip
    if isinstance(value, dict):
        return {k: normalise(v) for k, v in value.items()}
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="source", required=True, type=Path)
    args = parser.parse_args()
    path = args.source / "docs" / "evidence" / "sim-mechanics-m4-logistics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    series: dict[str, list] = {key: [] for key in KEYS}
    current: dict = {}
    for row in data["samples"]:
        current = undiff(current, row)
        for key in KEYS:
            value = normalise(current[key])
            if not series[key] or series[key][-1][1] != value:
                series[key].append([current["t"], value])
    out = {
        "source": "FactorioRL docs/evidence/sim-mechanics-m4-logistics.json, trimmed by "
        "tools/trim_logistics_evidence.py",
        "engine": {k: v for k, v in data["engine"].items() if k != "executable"},
        "ticks": data["ticks"],
        "series": series,
    }
    DEST.write_bytes(lzma.compress(json.dumps(out, sort_keys=True).encode()))
    print(f"wrote {DEST} ({DEST.stat().st_size // 1024} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
