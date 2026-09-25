"""Belt-line segments against the engine, fifth probe (csrc/fsim.c, "segments").

FactorioRL's tools/probe_logistics5.py ran rigs of timed operations in
Factorio 2.0.60 and logged, every tick, every item on every belt, which belt
lanes are one segment (`line_equals`), every inserter's hand and every chest
(docs/sim-logistics.md there, "Fifth probe"). tests/logistics_rigs5.py runs
the same rigs here; every reading must agree on every tick. Families:

- `order`: where the pieces of a segment go in the activation order when it
  splits at a boundary, loses a belt or has one turned;
- `change`, `feedchg`: turning and removing belts of young and merged lines,
  and of a feed near its sideload;
- `loop`, `loop2`: closed loops, and where their seam is;
- `bound`, `dist`, `drill`: where attachments mark their boundaries, in many
  geometries, drills included;
- `loop3`: 2 x 2 loops, where the boundary search meets the loop's front;
- `trig`, `trig2`, `trig3`: what sets a boundary off, and boundaries close
  together.
"""

from __future__ import annotations

import json
import lzma
from pathlib import Path

import pytest
from logistics_rigs5 import Rig5, engine_reading

HERE = Path(__file__).resolve().parent / "golden"
FAMILIES = (
    "order",
    "change",
    "feedchg",
    "loop",
    "loop2",
    "bound",
    "dist",
    "drill",
    "loop3",
    "trig",
    "trig2",
    "trig3",
)
EVIDENCE = {f: json.loads(lzma.decompress((HERE / f"logistics5-{f}.json.xz").read_bytes()))
            for f in FAMILIES}  # fmt: skip
CASES = [(f, name) for f in FAMILIES for name in sorted(EVIDENCE[f]["rigs"])]


def _expand(log: list, last: int) -> list:
    out, at = [], 0
    for t in range(last + 1):
        while at + 1 < len(log) and log[at + 1][0] <= t:
            at += 1
        out.append(log[at][1])
    return out


@pytest.mark.parametrize(("family", "name"), CASES, ids=[f"{f}-{n}" for f, n in CASES])
def test_rig_matches_the_engine_every_tick(family, name):
    doc = EVIDENCE[family]
    rig = doc["rigs"][name]
    engine = _expand(doc["log"][name], rig["ticks"])
    for t, ours in enumerate(Rig5(rig).run()):
        assert ours == engine_reading(engine[t]), (name, t)
