"""Replay a golden trace in the simulator and compare it record by record.

Free-running: the scene is installed from the trace's blueprint and every
recorded action is applied in order; nothing is copied from the recording but
the actions. The episode logic FactorioRL's environment adds around a step --
running the verification window when the decision budget runs out -- is
reproduced here, because the recorded decision includes it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from fsim import Sim
from fsim.trace import (
    Normaliser,
    comparable_hidden,
    first_difference,
    read_trace,
    relax_hand_y,
)

GOLDEN = Path(__file__).resolve().parents[1] / "tests" / "golden"


@dataclass
class Divergence:
    decision: int
    part: str
    path: str


class Replay:
    """Drives a `Sim` the way `FactorioEnv` drives the engine."""

    def __init__(self, header: dict, sim: Sim | None = None) -> None:
        self.header = header
        self.sim = sim or Sim()
        self.normaliser = Normaliser()
        self.truth_extra: dict = {}
        self.over = False

    def reset(self) -> dict:
        self.sim.reset(self.header["blueprint"])
        self.normaliser.begin({"tick": 0, "absolute_tick": 0})
        self.truth_extra = {}
        self.over = False
        self.items_named = 0
        return self.record()

    def load_hidden(self, hidden: dict) -> None:
        """Load a recorded state, keeping belt item names in step with it.

        A recording names belt items 1, 2, ... in the order it first meets
        them. Loaded items keep their recorded names as their ids here, and the
        next item the simulator makes must get the next name, so this side's
        renaming becomes the identity up to the highest name recorded so far.
        """
        self.sim.load_hidden(hidden)
        for record in hidden.get("entities") or []:
            for lane in record.get("lanes") or []:
                for item in lane or []:
                    if len(item) > 2 and item[2] is not None:
                        self.items_named = max(self.items_named, int(item[2]))
        self.normaliser.item_ids = {k: k for k in range(1, self.items_named + 1)}
        self.sim.env.next_item_id = max(self.sim.env.next_item_id, self.items_named)

    def step(self, key: str, arguments: dict) -> dict:
        self.sim.step(key, arguments, self.header["decision_ticks"])
        status, error = self.sim.action_outcome()
        verification = self.header.get("verification")
        truncated = (
            self.sim.steps >= self.header["max_decision_steps"]
            or self.sim.tick >= self.header["construction_tick_limit"]
        )
        if truncated and verification and not self.over:
            self._verify(verification)
            self.over = True
        record = self.record()
        record["transition"] = {"action_status": status, "action_error": error}
        return record

    def _verify(self, verification: dict) -> None:
        item, source = verification["item"], verification.get("source")
        machine = self.sim.truth()["machine_produced"]
        before = float(machine.get(item, 0)) if isinstance(machine, dict) else 0.0
        source_before = float(machine.get(source, 0)) if isinstance(machine, dict) else 0.0
        chunks = verification["ticks"] // self.header["decision_ticks"]
        for _ in range(chunks):
            self.sim.step("wait", {}, self.header["decision_ticks"])
            self.sim.steps -= 1  # evaluator time is not a decision
        machine = self.sim.truth()["machine_produced"]
        machine = machine if isinstance(machine, dict) else {}
        output = max(0.0, float(machine.get(item, 0)) - before)
        produced = output
        if source:
            produced = min(output, max(0.0, float(machine.get(source, 0)) - source_before))
        self.truth_extra = {"verification": {item: produced}}

    def record(self) -> dict:
        observation = self.sim.observation()
        truth = {**self.sim.truth(), **self.truth_extra}
        hidden = self.sim.hidden()
        n = self.normaliser
        n.claim(observation, truth, hidden)
        return {
            "observation": n.observation(observation),
            "truth": n.truth(truth),
            "hidden": n.hidden(hidden),
        }


PARTS = ("observation", "truth", "hidden")


def _footprint_totals(record: dict) -> dict:
    """Ore under each drill, as one total per drill footprint.

    Declared relaxation: which of its four tiles a drill takes the third ten
    ore from follows engine-internal entity order (see the note on
    DRILL_TILES in csrc/fsim.c). The total a drill has taken is exact.
    """
    hidden = record["hidden"]
    tiles = {}
    for entity in hidden.get("entities") or []:
        if entity["name"] != "burner-mining-drill":
            continue
        cx, cy = entity["position"][0] // 256, entity["position"][1] // 256
        key = f"{cx},{cy}"
        for dx in (-1, 0):
            for dy in (-1, 0):
                tiles[(cx + dx, cy + dy)] = key
    return tiles


def _relax(record: dict, tiles: dict) -> dict:
    if not tiles:
        return record
    out = json.loads(json.dumps(record))
    totals: dict[str, int] = {}
    for r in out["hidden"]["resources"]:
        key = tiles.get(((r["position"][0] - 128) // 256, (r["position"][1] - 128) // 256))
        if key:
            totals[key] = totals.get(key, 0) + r["amount"]
    for r in out["hidden"]["resources"]:
        key = tiles.get(((r["position"][0] - 128) // 256, (r["position"][1] - 128) // 256))
        if key:
            r["amount"] = totals[key]
    listed = out["observation"].get("resources", {}).get("tiles")
    if isinstance(listed, list):
        for t in listed:
            key = tiles.get((math.floor(t["p"][0]), math.floor(t["p"][1])))
            if key:
                t["amount"] = totals.get(key, t["amount"])
    return out


def compare(expected: dict, actual: dict, parts=PARTS) -> tuple[str, str] | None:
    tiles = _footprint_totals(expected)
    expected, actual = _relax(expected, tiles), _relax(actual, tiles)
    if "hidden" in actual and "hidden" in expected:
        actual = {**actual, "hidden": relax_hand_y(expected["hidden"], actual["hidden"])}
    for part in parts:
        found = first_difference(expected[part], actual[part])
        if found:
            return part, found
    transition = expected.get("transition")
    if transition and "transition" in actual:
        for key in ("action_status", "action_error"):
            if transition.get(key) != actual["transition"].get(key):
                return (
                    "transition",
                    f".{key}: {transition.get(key)!r} != {actual['transition'].get(key)!r}",
                )
    return None


def free_run(name: str, stop_at_first: bool = True, parts=PARTS) -> list[Divergence]:
    header, records = read_trace(GOLDEN / f"{name}.jsonl.xz")
    replay = Replay(header)
    out = []
    found = compare(records[0], replay.reset(), parts)
    if found:
        out.append(Divergence(0, *found))
        if stop_at_first:
            return out
    for record in records[1:]:
        action = record["transition"]["action"]
        actual = replay.step(action["key"], action["arguments"])
        found = compare(record, actual, parts)
        if found:
            out.append(Divergence(record["decision"], *found))
            if stop_at_first:
                return out
    return out


def tick_run(name: str) -> Divergence | None:
    """Replay a scenario one tick at a time against its tick trace."""
    header, records = read_trace(GOLDEN / f"{name}.jsonl.xz")
    tick_header, rows = read_trace(GOLDEN / f"{name}.ticks.jsonl.xz")
    sim = Sim()
    sim.reset(tick_header["blueprint"])
    normaliser = Normaliser()
    normaliser.begin({"tick": 0, "absolute_tick": 0})

    def hidden():
        raw = sim.hidden()
        normaliser.claim(raw)
        return normaliser.hidden(raw)

    def check(index):
        ours = relax_hand_y(rows[index], hidden())
        found = first_difference(comparable_hidden(rows[index]), comparable_hidden(ours))
        return Divergence(index, "hidden", found) if found else None

    bad = check(0)
    if bad:
        return bad
    index = 0
    per = tick_header["ticks_per_decision"]
    for record in records[1:]:
        action = record["transition"]["action"]
        sim.step(action["key"], action["arguments"], 1)
        index += 1
        bad = check(index)
        if bad:
            return bad
        for _ in range(per - 1):
            sim.step("wait", {}, 1)
            index += 1
            bad = check(index)
            if bad:
                return bad
    return None
