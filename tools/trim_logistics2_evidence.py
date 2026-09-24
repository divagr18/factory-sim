"""Cut FactorioRL's second logistics probe evidence down to what the tests read.

`docs/evidence/sim-mechanics-m4-logistics2.json.xz` (FactorioRL, written by
`tools/probe_logistics2.py`) records about 460 rigs for 1,500 ticks, each as the
ticks on which its record changed. tests/test_mechanics_logistics2.py rebuilds
the rigs whose mechanics the simulator reproduces; this keeps those rigs'
series, the rotation reads, the character's series and each rig's origin, and
drops the engine's belt item ids (the simulator numbers its own). The result is
tests/golden/sim-mechanics-m4-logistics2.json.xz.

    uv run python tools/trim_logistics2_evidence.py --from ../FactorioRL
"""

from __future__ import annotations

import argparse
import json
import lzma
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "tests" / "golden" / "sim-mechanics-m4-logistics2.json.xz"

#: Rig name prefixes the tests compare. Left out: `sim_*` (simultaneous
#: sideload arrivals, not reproduced), `char_*` and `mine` (read through
#: `character`).
PREFIXES = (
    "turn_", "sl_left_", "side_", "tdrop_", "tdrill_", "ground_", "gpair_", "full_", "fill_",
    "mix_", "order", "wake3", "chain_", "woken_", "rot_", "stat_", "dstat_", "spick_", "tpick_",
    "pe_", "sr_", "seg_", "win_",
)  # fmt: skip


def normalise(value):
    """Belt items as [name, position]; empty Lua tables as lists."""
    if isinstance(value, dict):
        if not value:
            return []
        out = {k: normalise(v) for k, v in value.items()}
        if "l" in out and "sh" in out:
            out["l"] = [[[it[0], it[1]] for it in lane] for lane in out["l"]]
        return out
    if isinstance(value, list):
        return [normalise(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="source", required=True, type=Path)
    args = parser.parse_args()
    path = args.source / "docs" / "evidence" / "sim-mechanics-m4-logistics2.json.xz"
    data = json.loads(lzma.decompress(path.read_bytes()))
    keep = sorted(k for k in data["series"] if k.startswith(PREFIXES))
    setup = data["setup"]
    rotations = {
        name: {
            side: {"shape": r[side]["shape"], "d": r[side]["d"],
                   "l": [[[it[0], it[1]] for it in lane] for lane in normalise(r[side]["l"])]}
            for side in ("before", "after")
        }
        for name, r in setup["rotations"].items()
    }  # fmt: skip
    out = {
        "source": "FactorioRL docs/evidence/sim-mechanics-m4-logistics2.json.xz, trimmed by "
        "tools/trim_logistics2_evidence.py",
        "engine": {k: v for k, v in data["engine"].items() if k != "executable"},
        "origin": data["origin"],
        "ticks": data["ticks"],
        "events": data["events"],
        "bases": setup["bases"],
        "prototypes": setup["prototypes"],
        "placement": setup["placement"],
        "rotations": rotations,
        "series": {k: [[t, normalise(v)] for t, v in data["series"][k]] for k in keep},
        "character": [[t, normalise(v)] for t, v in data["character"]],
    }
    DEST.write_bytes(lzma.compress(json.dumps(out, sort_keys=True).encode()))
    print(f"wrote {DEST} ({DEST.stat().st_size // 1024} KiB), {len(keep)} rigs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
