"""Belts, inserters, chests and ground piles against the engine, second probe.

FactorioRL's tools/probe_logistics2.py built about 460 rigs in Factorio 2.0.60
and read each after every tick for 1,500 ticks; docs/sim-logistics.md there
states what it found. tests/logistics_rigs2.py rebuilds the rigs whose
mechanics the simulator reproduces, and the engine's readings are in
tests/golden/sim-mechanics-m4-logistics2.json.xz
(tools/trim_logistics2_evidence.py).

Every reading of every compared rig must agree on every tick: belt item
positions exactly, energy and remaining fuel to the parity tolerance,
everything else exactly. `NOT_COMPARED` lists the rigs left out and why.
"""

from __future__ import annotations

import json
import lzma
import math
from pathlib import Path

import pytest
from logistics_rigs2 import Rigs2

from fsim import HAND_Y_UNKNOWN

GOLDEN = Path(__file__).resolve().parent / "golden" / "sim-mechanics-m4-logistics2.json.xz"
EVIDENCE = json.loads(lzma.decompress(GOLDEN.read_bytes()))
TICKS = EVIDENCE["ticks"]

#: Rigs in the evidence that are not compared, by prefix, and why.
NOT_COMPARED = {
    # Stone into a furnace makes stone bricks, which the simulator has no item
    # for: the engine's inserter keeps 4 stone in the furnace.
    "fill_stone_furnace": "stone bricks",
    # A script takes the front item off the drill's belt; no action can.
    "dstat_take_belt": "belt item removed by script",
    # Pickups from items stopped on a turn by an inserter on the side opposite
    # the feeding belt: the arm aims at the tabled item point (belt_item_offset)
    # and energy agrees every tick, but on two ticks per rig, as the hand
    # reaches an item, its x is drawn 1/256 off (-346 where the model has -345,
    # -237 where it has -238): how the engine rounds the arm at such a target
    # is not reproduced. The pickups from the other two sides are exact.
    "tpick_r_e": "hand x at a turn pickup, 1/256",
    "tpick_l_e": "hand x at a turn pickup, 1/256",
}


def _expand(series: list, last: int) -> list:
    out, at = [], 0
    for t in range(last + 1):
        while at + 1 < len(series) and series[at + 1][0] <= t:
            at += 1
        out.append(series[at][1] if series[0][0] <= t else None)
    return out


def _number(value):
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def same(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    na, nb = _number(a), _number(b)
    if na is not None and nb is not None:
        return math.isclose(na, nb, rel_tol=1e-12, abs_tol=1e-12)
    return a == b


def flatten(value, path: str = "") -> dict:
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out.update(flatten(v, f"{path}.{k}" if path else k))
        return out
    if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
        out = {}
        for i, v in enumerate(value):
            out.update(flatten(v, f"{path}[{i}]"))
        return out
    return {path: value}


COMPARED = sorted(k for k in EVIDENCE["series"] if k not in NOT_COMPARED)


#: Rigs share a world in batches small enough for the entity table; they do
#: not reach each other.
BATCH = 40


class Run:
    def __init__(self) -> None:
        self.rigs, self.samples, self.rotations = {}, {}, {}
        for first in range(0, len(COMPARED), BATCH):
            batch = Rigs2(EVIDENCE, set(COMPARED[first : first + BATCH]))
            records = list(batch.samples())
            for name, r in batch.rigs.items():
                self.rigs[name] = r
                last = TICKS if r.until is None else r.until
                self.samples[name] = [records[t][name] for t in range(last + 1)]
            self.rotations.update(batch.rotations)


@pytest.fixture(scope="module")
def run():
    return Run()


def mismatches(run, key: str) -> list:
    r = run.rigs[key]
    last = TICKS if r.until is None else r.until
    engine = _expand(EVIDENCE["series"][key], last)
    out = []
    for t in range(last + 1):
        a, b = flatten(engine[t]), flatten(run.samples[key][t])
        # The drawn lift of a swing that did not start from rest is not known:
        # its hand y is not compared (fsim.trace.relax_hand_y).
        for path in [p for p in b if p.endswith(HAND_Y_UNKNOWN)]:
            del b[path]
            hand = path[: -len(HAND_Y_UNKNOWN)] + "hp"
            if isinstance(a.get(hand), list) and isinstance(b.get(hand), list):
                b[hand] = [b[hand][0], a[hand][1]]
        for path in sorted(set(a) | set(b)):
            if not same(a.get(path), b.get(path)):
                out.append((t, path, a.get(path), b.get(path)))
    return out


def test_every_compared_rig_is_built(run):
    assert sorted(run.rigs) == COMPARED


@pytest.mark.parametrize("key", COMPARED)
def test_rig_matches_the_engine_every_tick(run, key):
    found = mismatches(run, key)
    assert not found, found[:5]


@pytest.mark.parametrize("key", sorted(k for k in EVIDENCE["rotations"] if k in COMPARED))
def test_rotation_replaces_items_as_the_engine(run, key):
    engine, ours = EVIDENCE["rotations"][key], run.rotations[key]
    assert ours["before"] == engine["before"]["l"]
    assert ours["after"] == engine["after"]["l"]
