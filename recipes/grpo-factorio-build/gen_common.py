"""What rationalize.py and make_replay.py share: an async client for an
OpenAI-compatible chat endpoint (a vLLM server), and an append-only JSONL store
that makes both scripts resumable.

Only the standard library is used, so the scripts run in the pod's prime-rl
venv and in factory-sim's dev env (where the tests run them against a fake
server) alike. Requests are blocking `urllib` calls on a thread pool, one
thread per request in flight; the event loop only orchestrates.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

#: Hosts reached without any system or environment proxy.
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass
class Choice:
    """One sampled reply. `reasoning` is `reasoning_content`, set only by a server
    running a reasoning parser; `text` is the message content."""

    text: str
    reasoning: str | None
    finish_reason: str | None


@dataclass
class Reply:
    """One request's outcome: its choices, or `error` when it failed for good."""

    choices: list[Choice] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None  # summed over the choices
    attempts: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class ChatClient:
    """`/chat/completions` with retries, `concurrency` requests in flight at most.

    429s, 5xx, timeouts, dropped connections and unparseable bodies are retried
    with jittered exponential backoff; any other HTTP error is final. A request
    never raises: failures come back as `Reply.error`, so one bad request costs
    one example, not the run."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "EMPTY",
        concurrency: int = 64,
        timeout_s: float = 1800.0,
        max_attempts: int = 5,
        backoff_s: float = 2.0,
        backoff_max_s: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.concurrency = max(1, int(concurrency))
        self.timeout_s = timeout_s
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_s = backoff_s
        self.backoff_max_s = backoff_max_s
        host = urllib.parse.urlsplit(self.base_url).hostname or ""
        handlers = [urllib.request.ProxyHandler({})] if host in _LOCAL_HOSTS else []
        self._opener = urllib.request.build_opener(*handlers)
        self._pool = ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix="chat")
        self._sleep = time.sleep

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _headers(self) -> dict:
        return {"Content-Type": "application/json", "Authorization": "Bearer " + self._api_key}

    def served_model(self) -> str:
        """The first model id the server lists: vLLM serves exactly one."""
        req = urllib.request.Request(self.base_url + "/models", headers=self._headers())
        with self._opener.open(req, timeout=60) as resp:
            data = json.loads(resp.read())
        models = [m["id"] for m in data.get("data", [])]
        if not models:
            raise RuntimeError(f"{self.base_url}/models lists no model")
        return models[0]

    def _post_once(self, payload: bytes) -> dict:
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=payload,
            method="POST",
            headers=self._headers(),
        )
        with self._opener.open(req, timeout=self.timeout_s) as resp:
            raw = resp.read()
        return json.loads(raw)

    def chat_sync(self, body: dict) -> Reply:
        payload = json.dumps(body).encode("utf-8")
        last = "no attempt made"
        for attempt in range(self.max_attempts):
            try:
                data = self._post_once(payload)
            except urllib.error.HTTPError as e:
                detail = _body_snippet(e)
                last = f"HTTP {e.code}: {detail}".rstrip(": ")
                if e.code != 429 and e.code < 500:
                    return Reply(attempts=attempt + 1, error=last)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last = f"{type(e).__name__}: {getattr(e, 'reason', e)}"
            except ValueError as e:  # the body was not JSON
                last = f"unparseable response: {e}"
            else:
                reply = _parse(data)
                reply.attempts = attempt + 1
                if reply.error is None:
                    return reply
                last = reply.error
            if attempt + 1 < self.max_attempts:
                delay = min(self.backoff_max_s, self.backoff_s * 2**attempt)
                self._sleep(delay * random.uniform(0.5, 1.0))
        return Reply(attempts=self.max_attempts, error=f"gave up: {last}")

    async def chat(self, body: dict) -> Reply:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, self.chat_sync, body)


def _body_snippet(e: urllib.error.HTTPError) -> str:
    try:
        return e.read().decode("utf-8", "replace")[:300]
    except Exception:
        return ""


def _parse(data: dict) -> Reply:
    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        return Reply(error=f"response has no choices: {json.dumps(data)[:300]}")
    out = []
    for c in choices:
        msg = c.get("message") or {}
        out.append(
            Choice(
                text=msg.get("content") or "",
                reasoning=msg.get("reasoning_content") or msg.get("reasoning") or None,
                finish_reason=c.get("finish_reason"),
            )
        )
    usage = data.get("usage") or {}
    return Reply(
        choices=out,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )


def sampling_body(
    model: str,
    messages: list[dict],
    *,
    n: int,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
    seed: int | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> dict:
    """A vLLM chat request. `chat_template_kwargs` switches Qwen3.5's thinking the
    way eval.toml and the RL renderer do; unset top_p/top_k keep the server's."""
    body = {
        "model": model,
        "messages": messages,
        "n": n,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if seed is not None:
        body["seed"] = seed
    if top_p is not None:
        body["top_p"] = top_p
    if top_k is not None:
        body["top_k"] = top_k
    return body


def stable_seed(*parts) -> int:
    """A request seed from its identity, so a resumed run samples what it would have."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


# ------------------------------------------------------------------ storage


def read_jsonl(path: str) -> list[dict]:
    """Rows of a JSONL file; a torn last line (a crash mid-write) is dropped."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f.read().split("\n") if line.strip()]
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return rows


def write_jsonl(path: str, rows: list[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def repair_tail(path: str) -> None:
    """Cut a torn last line off an append-only file before appending to it again."""
    if not os.path.exists(path):
        return
    with open(path, "rb") as f:
        data = f.read()
    if not data or data.endswith(b"\n"):
        return
    cut = data.rfind(b"\n") + 1
    with open(path, "wb") as f:
        f.write(data[:cut])


def check_run_config(path: str, config: dict) -> None:
    """Write the run's generation settings on the first run; refuse a resume that
    would mix samples drawn under different ones into one directory."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
        diff = {k: (old.get(k), v) for k, v in config.items() if old.get(k) != v}
        if diff:
            lines = "\n".join(f"  {k}: was {a!r}, now {b!r}" for k, (a, b) in sorted(diff.items()))
            raise SystemExit(
                f"{path} was written with other settings; use a new --out-dir or the old ones:\n"
                + lines
            )
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summary(values) -> dict:
    """n, mean, median, p10, p90 and max of a list of numbers (empty -> n only)."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {"n": 0}

    def q(p):
        return vals[min(len(vals) - 1, int(p * len(vals)))]

    return {
        "n": len(vals),
        "mean": round(statistics.fmean(vals), 2),
        "median": statistics.median(vals),
        "p10": q(0.1),
        "p90": q(0.9),
        "max": vals[-1],
    }


class TokenCounter:
    """Reply lengths in tokens, with the served model's tokenizer when one loads.

    Stats only. vLLM reports completion tokens per request, not per choice, so a
    choice is counted here; without `transformers` (or with name "none") every
    count is None and only characters and words are reported."""

    def __init__(self, name: str | None):
        self.tok = None
        self.error = None
        if not name or name == "none":
            return
        try:
            from transformers import AutoTokenizer

            self.tok = AutoTokenizer.from_pretrained(name)
        except Exception as e:  # stats only: never fail a run over them
            self.error = f"{type(e).__name__}: {e}"

    def __call__(self, text: str | None) -> int | None:
        if self.tok is None or text is None:
            return None
        return len(self.tok(text, add_special_tokens=False)["input_ids"])
