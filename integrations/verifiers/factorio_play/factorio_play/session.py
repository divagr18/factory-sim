"""One factory-sim episode driven one call at a time, for a tool-using model.

A builder program is a function `build(world)` that `run_episode` calls and
then scores. A tool-using model is the same program with its body spread over
many turns. So a `WorldSession` runs the real `fsim.program_api.run_episode`
on a background thread with a `build` that takes its next `world.<method>`
call from a queue: every tool call is one call of that method, and `finish()`
returns from `build`, after which `run_episode` waits out the episode and
verifies it exactly as it does for a program. Nothing about the episode,
its budget or its scoring is reimplemented here.

No `verifiers` import: the tool server wraps this, and tests drive it directly.
"""

from __future__ import annotations

import dataclasses
import queue
import threading

from fsim.obsview import Entity
from fsim.program_api import EpisodeResult, run_episode

#: `World` methods exposed as tools, and whether each spends a decision.
QUERIES = (
    "me tile inventory ore_tiles blocked_tiles entities patch decisions_left last_refused"
).split()
ACTIONS = "move place give take mine wait".split()

_FINISH = object()

#: A model writes "east" as readily as "E". World takes the letters; the tools
#: take either, and any case, for directions and facings.
_COMPASS = {
    "north": "N", "east": "E", "south": "S", "west": "W",
    "n": "N", "e": "E", "s": "S", "w": "W",
}  # fmt: skip


def _normalise(name: str, args: tuple) -> tuple:
    def compass(v):
        return _COMPASS.get(v.strip().lower(), v) if isinstance(v, str) else v

    args = list(args)
    if name == "move":
        if args:
            args[0] = compass(args[0])
        if len(args) > 1 and isinstance(args[1], str):
            args[1] = args[1].strip().lower()
    elif name == "place" and len(args) >= 4:
        args[3] = compass(args[3])
    return tuple(args)


def _plain(value):
    """JSON-ready: dataclasses to dicts, tuples to lists."""
    if isinstance(value, Entity):
        return dataclasses.asdict(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


class WorldSession:
    def __init__(
        self,
        blueprint: dict,
        *,
        task: str = "construct_smelting_line",
        decision_budget: int = 600,
        call_timeout_s: float = 60.0,
    ) -> None:
        self._calls: queue.Queue = queue.Queue()
        self._replies: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._timeout = call_timeout_s
        self._world = None
        self.result: EpisodeResult | None = None
        self.harness_error: str | None = None
        self._thread = threading.Thread(
            target=self._run, args=(blueprint, task, decision_budget), daemon=True
        )
        self._thread.start()
        kind, value = self._replies.get(timeout=call_timeout_s)  # the world is ready
        if kind != "ready":
            raise RuntimeError(f"episode failed to start: {value}")

    # -------------------------------------------------------- episode thread

    def _run(self, blueprint, task, budget) -> None:
        try:
            self.result = run_episode(self._build, blueprint, task=task, decision_budget=budget)
        except Exception as e:  # the simulator or reset, not the model
            self.harness_error = f"{type(e).__name__}: {e}"
            self._replies.put(("error", self.harness_error))
        self._replies.put(("done", None))

    def _build(self, world) -> None:
        self._world = world
        self._replies.put(("ready", None))
        while True:
            item = self._calls.get()
            if item is _FINISH:
                return
            name, args = item
            before = len(world._trace)
            try:
                value = getattr(world, name)(*args)
            except Exception as e:
                # BudgetExhausted, or the refusal cap: the program's episode is over,
                # exactly as when a program raises. run_episode records and scores it.
                self._replies.put(("ended", f"{type(e).__name__}: {e}"))
                raise
            # An action's bool alone gives a model nothing to correct. The world
            # records why an intent was refused, or that the game refused a legal
            # action (which still cost a decision); pass that back.
            note = None
            if name in ACTIONS and len(world._trace) > before:
                last = world._trace[-1]
                if "-> refused" in last:
                    note = last.split("-> refused", 1)[1].strip()
                    if note.startswith("(") and note.endswith(")"):
                        note = note[1:-1]
                elif last.endswith("-> failed"):
                    note = "the game refused this action; it still cost a decision"
            self._replies.put(("ok", (value, note)))

    # -------------------------------------------------------- caller side

    @property
    def finished(self) -> bool:
        return self.result is not None or self.harness_error is not None

    def counters(self) -> dict:
        w = self._world
        if w is None:
            return {"decisions": 0, "refusals": 0, "failures": 0}
        return {"decisions": w.decisions, "refusals": w.refusals, "failures": w.failures}

    def call(self, name: str, *args) -> dict:
        """{"ok": True, "result": ...} or {"ok": False, "error": ...}; never raises."""
        if name not in QUERIES and name not in ACTIONS:
            return {"ok": False, "error": f"unknown method {name!r}"}
        with self._lock:
            if self.finished:
                return {"ok": False, "error": "the episode is over; nothing more can be done"}
            self._calls.put((name, _normalise(name, args)))
            try:
                kind, value = self._replies.get(timeout=self._timeout)
            except queue.Empty:
                return {"ok": False, "error": "the simulator did not answer in time"}
            if kind == "ok":
                value, note = value
                reply = {"ok": True, "result": _plain(value)}
                if note:
                    reply["refused" if value is False else "note"] = note
                return reply
            self._wait_done()
            return {"ok": False, "error": f"episode ended ({value}); it has been scored"}

    def finish(self) -> EpisodeResult | None:
        """End the build phase, run the verification window, return the result."""
        with self._lock:
            if not self.finished:
                self._calls.put(_FINISH)
                self._wait_done()
            return self.result

    def _wait_done(self) -> None:
        while True:
            kind, _ = self._replies.get(timeout=self._timeout * 10)
            if kind == "done":
                self._thread.join(timeout=self._timeout)
                return
