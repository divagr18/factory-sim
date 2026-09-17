"""Reading golden traces, and the normalisation and comparison they need.

The normalisation mirrors what FactorioRL's recorder applies before hashing, so
the simulator's records and the engine's can be compared field by field:
episode-relative ticks, request ids named in session order, empty engine
tables read as the lists they are.

Floats are compared with a tolerance. The engine formats doubles with an
imprecise final digit (0.0041666666666666661 for 1/240), and one of its
progress bars rounds in a way this project matches to about one part in
10^14 rather than exactly; every discrete outcome is still compared exactly.
"""

from __future__ import annotations

import json
import lzma
import math
import re
from pathlib import Path

REQUEST_ID = re.compile(r"^[a-z_]+-[0-9a-f]{8}-\d+(:[a-z_]+)?$")
ABSOLUTE_TICKS = frozenset(
    {"started_tick", "deadline_tick", "cancelled_at_tick", "observed_tick", "last_seen",
     "last_tick", "seen"}
)  # fmt: skip
TRUTH_KEYS = (
    "tick", "markers", "containers", "working", "produced", "working_counts",
    "placed_counts", "stored_energy", "remaining_burning_fuel", "built",
    "machine_produced", "handcrafted", "mined_by_hand", "by_hand_source", "verification",
)  # fmt: skip
HIDDEN_LISTS = ("entities", "ground_items", "resources")
NUMBER = re.compile(r"^-?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$")

#: Relative tolerance for doubles.
REL_TOL = 1e-12
ABS_TOL = 1e-12


def read_trace(path: Path) -> tuple[dict, list[dict]]:
    lines = lzma.decompress(Path(path).read_bytes()).decode().splitlines()
    return json.loads(lines[0]), [json.loads(line) for line in lines[1:]]


def _listify(value):
    return [] if value == {} else value


def _request_order(request_id: str) -> tuple[int, str]:
    head, _, suffix = request_id.partition(":")
    return int(head.rsplit("-", 1)[1]), suffix


class Normaliser:
    def __init__(self) -> None:
        self.base_tick = 0
        self.request_ids: dict[str, str] = {}

    def begin(self, observation: dict) -> None:
        self.base_tick = int(observation.get("absolute_tick") or 0) - int(
            observation.get("tick") or 0
        )

    def claim(self, *values) -> None:
        found: set[str] = set()

        def walk(value):
            if isinstance(value, str) and REQUEST_ID.match(value):
                found.add(value)
            elif isinstance(value, dict):
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        for value in values:
            walk(value)
        for value in sorted(found - set(self.request_ids), key=_request_order):
            self.request_ids[value] = f"r{len(self.request_ids) + 1}"

    def _rename(self, value):
        if isinstance(value, str) and REQUEST_ID.match(value):
            if value not in self.request_ids:
                self.claim(value)
            return self.request_ids[value]
        if isinstance(value, dict):
            return {k: self._rename(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._rename(v) for v in value]
        return value

    def _relative(self, tick):
        return None if tick is None else int(tick) - self.base_tick

    def _ticks(self, value, key=None):
        if isinstance(value, dict):
            return {k: self._ticks(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [self._ticks(v, key) for v in value]
        if key in ABSOLUTE_TICKS and isinstance(value, int) and not isinstance(value, bool):
            return self._relative(value)
        return value

    def observation(self, observation: dict) -> dict:
        body = {k: v for k, v in observation.items() if k not in ("episode_id", "absolute_tick")}
        return self._rename(self._ticks(body))

    def truth(self, truth: dict) -> dict:
        body = {k: truth[k] for k in TRUTH_KEYS if k in truth}
        if "tick" in body:
            body["tick"] = self._relative(body["tick"])
        return self._rename(body)

    def hidden(self, hidden: dict) -> dict:
        body = dict(hidden)
        body["tick"] = self._relative(body.get("tick"))
        for key in HIDDEN_LISTS:
            body[key] = _listify(body.get(key) or [])
        for record in body["entities"]:
            record["inventories"] = record.get("inventories") or {}
            for inventory in record["inventories"].values():
                inventory["stacks"] = _listify(inventory.get("stacks") or [])
        character = body.get("character")
        if character and character.get("main"):
            character["main"]["stacks"] = _listify(character["main"].get("stacks") or [])
        handles = dict(body.get("handles") or {})
        order = []
        for entry in _listify(handles.get("order") or []):
            entry = dict(entry)
            entry["first_seen"] = self._relative(entry.get("first_seen"))
            if entry.get("destroyed_tick") is not None:
                entry["destroyed_tick"] = self._relative(entry["destroyed_tick"])
            order.append(entry)
        handles["order"] = order
        body["handles"] = handles
        inflight = dict(body.get("inflight") or {})
        entries = []
        for entry in _listify(inflight.get("entries") or []):
            entry = dict(entry)
            entry["started_tick"] = self._relative(entry.get("started_tick"))
            if entry.get("deadline_tick") is not None:
                entry["deadline_tick"] = self._relative(entry["deadline_tick"])
            entries.append(entry)
        inflight["entries"] = entries
        body["inflight"] = inflight
        return self._rename(body)


def _as_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and NUMBER.match(value):
        return float(value)
    return None


def first_difference(a, b, path: str = "") -> str | None:
    """The first field two values disagree on, with doubles compared loosely."""
    na, nb = _as_number(a), _as_number(b)
    if na is not None and nb is not None and not (isinstance(a, str) ^ isinstance(b, str)):
        if na == nb or math.isclose(na, nb, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            return None
        return f"{path}: {a!r} != {b!r}"
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f"{path}.{key}: present on one side only ({a.get(key)!r} vs {b.get(key)!r})"
            found = first_difference(a[key], b[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for index, (x, y) in enumerate(zip(a, b, strict=True)):
            found = first_difference(x, y, f"{path}[{index}]")
            if found:
                return found
        return None
    if type(a) is not type(b):
        return f"{path}: {a!r} != {b!r}"
    return None if a == b else f"{path}: {a!r} != {b!r}"


def comparable_hidden(hidden: dict) -> dict:
    """Hidden state without what a one-tick replay changes by construction."""
    body = {k: v for k, v in hidden.items() if k != "event_seq"}
    inflight = dict(body.get("inflight") or {})
    inflight.pop("next_seq", None)
    inflight["entries"] = [
        {k: v for k, v in entry.items() if k not in ("request_id", "seq")}
        for entry in inflight.get("entries") or []
        if entry.get("action") != "advance"
    ]
    body["inflight"] = inflight
    handles = dict(body.get("handles") or {})
    handles["order"] = [
        {k: v for k, v in entry.items() if k not in ("first_seen", "destroyed_tick")}
        for entry in handles.get("order") or []
    ]
    body["handles"] = handles
    return body
