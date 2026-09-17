"""Print the first divergence per golden scenario."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsim.parity import GOLDEN, free_run, tick_run  # noqa: E402


def main() -> int:
    index = json.loads((GOLDEN / "index.json").read_text(encoding="utf-8"))
    names = sys.argv[1:] or sorted(index)
    bad = 0
    for name in names:
        found = free_run(name)
        line = f"{name:36} decisions: " + ("ok" if not found else f"{found[0]}")
        if index[name].get("ticks"):
            t = tick_run(name)
            line += " | ticks: " + ("ok" if not t else f"{t}")
            bad += bool(t)
        bad += bool(found)
        print(line[:400], flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
