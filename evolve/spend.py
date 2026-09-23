"""A dollar cap on the model, kept in a file so it outlives one run.

The cap is a total, not a per-run allowance: "five dollars for now" means five
dollars across every run on this machine, so spend is appended to a ledger
beside the provider config and every run reads it before asking for more.

A request is priced twice. Before it is sent, at its worst case -- the prompt
at a pessimistic three characters per token, plus the whole `max_tokens` of
output -- and that worst case is held against the cap while the request is in
flight. When the reply arrives, at what the usage block says it cost. So the
loop can never overshoot by the requests it has in flight: it stops issuing
while there is still room for all of them to come back at their worst.

A request abandoned for taking too long still runs to completion on the
provider's side and is still billed, so its worst case stays reserved until
its reply turns up, and a reply that never turns up keeps it reserved for the
rest of the run.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

#: Pessimistic characters per token for sizing a prompt before it is sent.
CHARS_PER_TOKEN = 3.0


class SpendCapError(RuntimeError):
    """The cap cannot be enforced, or has been reached."""


class SpendLedger:
    """Dollars spent, in a JSON file shared by every run that names it.

    `price` is dollars per million tokens: `{"input", "cached_input", "output"}`.
    Reasoning tokens are reported inside `completion_tokens` and billed as
    output, so they are charged as output here too.
    """

    def __init__(self, path: str | os.PathLike, price: dict, cap_usd: float):
        missing = [k for k in ("input", "output") if k not in price]
        if missing:
            raise SpendCapError(f"price is missing {', '.join(missing)}; cannot enforce a cap")
        if cap_usd <= 0:
            raise SpendCapError("the cap must be positive")
        self.path = Path(path)
        self.price = {
            "input": float(price["input"]),
            "cached_input": float(price.get("cached_input", price["input"])),
            "output": float(price["output"]),
        }
        self.cap_usd = float(cap_usd)
        self._lock = self.path.with_suffix(self.path.suffix + ".lock")
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._write({"total_usd": 0.0, "calls": 0})

    # --- pricing ---

    def cost(self, usage: dict | None) -> float:
        """What a reply cost, from its usage block. A missing block costs nothing."""
        if not usage:
            return 0.0
        prompt = int(usage.get("prompt_tokens") or 0)
        cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        cached = min(cached, prompt)
        out = int(usage.get("completion_tokens") or 0)
        p = self.price
        return (
            (prompt - cached) * p["input"] + cached * p["cached_input"] + out * p["output"]
        ) / 1e6

    def worst(self, messages: list[dict], max_tokens: int) -> float:
        """A request's cost if the prompt is large and the reply uses every token allowed."""
        chars = sum(len(str(m.get("content", ""))) for m in messages)
        prompt = chars / CHARS_PER_TOKEN
        return (prompt * self.price["input"] + max_tokens * self.price["output"]) / 1e6

    # --- the ledger ---

    @property
    def total(self) -> float:
        return float(self._read().get("total_usd", 0.0))

    def allows(self, reserved_usd: float) -> bool:
        """Whether spend so far plus `reserved_usd` stays within the cap."""
        return self.total + reserved_usd <= self.cap_usd

    def charge(self, usage: dict | None) -> float:
        """Add a reply's cost to the ledger; returns what it cost."""
        amount = self.cost(usage)
        if amount <= 0:
            return 0.0
        with self._locked():
            data = self._read()
            data["total_usd"] = float(data.get("total_usd", 0.0)) + amount
            data["calls"] = int(data.get("calls", 0)) + 1
            data["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._write(data)
        return amount

    # --- files ---

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"total_usd": 0.0, "calls": 0}

    def _write(self, data: dict) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    class _Lock:
        def __init__(self, path: Path, timeout_s: float = 30.0):
            self.path, self.timeout_s = path, timeout_s

        def __enter__(self):
            deadline = time.monotonic() + self.timeout_s
            while True:
                try:
                    os.close(os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                    return self
                except FileExistsError:
                    # A lock older than the timeout belongs to a process that died.
                    try:
                        if time.time() - self.path.stat().st_mtime > self.timeout_s:
                            self.path.unlink(missing_ok=True)
                            continue
                    except FileNotFoundError:
                        continue
                    if time.monotonic() > deadline:
                        raise SpendCapError(
                            f"could not lock the spend ledger {self.path}"
                        ) from None
                    time.sleep(0.02)

        def __exit__(self, *exc):
            self.path.unlink(missing_ok=True)

    def _locked(self):
        return self._Lock(self._lock)
