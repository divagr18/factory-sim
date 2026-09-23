"""A small, sturdy client for an OpenAI-compatible chat endpoint.

The evolution loop is bound by the model, not the evaluator: a candidate costs
about a second to score and a minute or more to write. So the client's job is
throughput and staying up. It runs many requests at once on threads, retries
rate limits and server errors with jittered exponential backoff, and backs off
for a while when the endpoint keeps failing rather than hammering it.

Only the standard library is used (`urllib.request`), so the evolution code adds
no dependency to the simulator.

The API key never leaves `Provider._api_key` except in the Authorization header
of the request itself. Every error message and log line built here passes
through `_Scrubber`, which removes the key and any Authorization or Bearer value,
because an endpoint may echo request headers back in an error body.
"""

from __future__ import annotations

import email.utils
import json
import logging
import os
import random
import re
import textwrap
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

log = logging.getLogger("evolve.llm")

#: The provider config path. The older Qwen-specific name is still read.
CONFIG_ENV = "FSIM_LLM_CONFIG"
LEGACY_CONFIG_ENV = "FSIM_QWEN_CONFIG"

#: Attempts per request, including the first.
MAX_ATTEMPTS = 6
#: Backoff before retry i (0-based) is about BACKOFF_BASE_S * 2**i, capped, with jitter.
BACKOFF_BASE_S = 2.0
BACKOFF_MAX_S = 60.0
#: A server's Retry-After is honoured, but never beyond this.
RETRY_AFTER_MAX_S = 300.0
#: Consecutive failed attempts that open the circuit, and how long it stays open.
BREAKER_THRESHOLD = 10
BREAKER_COOLDOWN_S = 60.0
#: How much of an error body goes into an exception message.
ERROR_BODY_CHARS = 500


class Provider:
    """Where to send requests. The key is private and never shown."""

    __slots__ = (
        "base_url",
        "model",
        "_api_key",
        "token_param",
        "temperature",
        "extra",
        "price",
        "max_usd",
        "ledger_path",
    )

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        token_param: str = "max_tokens",
        temperature: float | None = 0.8,
        extra: dict | None = None,
        price: dict | None = None,
        max_usd: float | None = None,
        ledger_path: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        #: OpenAI's reasoning models refuse `max_tokens` and want
        #: `max_completion_tokens`; DashScope's compatible mode takes either.
        self.token_param = token_param
        #: None leaves `temperature` out of the request: reasoning models reject
        #: any value but their own default.
        self.temperature = temperature
        #: Merged into every request body, e.g. `{"service_tier": "flex"}`.
        self.extra = dict(extra or {})
        #: Dollars per million tokens, `{"input", "cached_input", "output"}`, and the
        #: spend cap. They sit with the key, in the private config, so the cap travels
        #: with the credential rather than living in a public repository.
        self.price = dict(price) if price else None
        self.max_usd = float(max_usd) if max_usd is not None else None
        self.ledger_path = ledger_path

    def __repr__(self) -> str:
        return (
            f"Provider(base_url={self.base_url!r}, model={self.model!r}, "
            f"token_param={self.token_param!r}, extra={self.extra!r}, "
            f"max_usd={self.max_usd!r}, api_key=<redacted>)"
        )

    __str__ = __repr__


def load_provider(path: str | None = None) -> Provider:
    """Read `{"base_url", "api_key", "model"}` from `path`, or from `$FSIM_QWEN_CONFIG`."""
    if path is None:
        path = os.environ.get(CONFIG_ENV) or os.environ.get(LEGACY_CONFIG_ENV)
        if not path:
            raise RuntimeError(
                f"no provider config: pass a path or set {CONFIG_ENV} to a JSON file "
                'holding {"base_url", "api_key", "model"}'
            )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"provider config not found: {path}")
    with open(path, encoding="utf-8-sig") as f:
        try:
            cfg = json.load(f)
        except json.JSONDecodeError as e:
            # The decoder's message quotes no content, but its doc attribute holds the
            # whole file, so drop the original exception rather than chain it.
            raise ValueError(f"provider config is not valid JSON: {path} ({e.msg})") from None
    missing = [k for k in ("base_url", "api_key", "model") if not cfg.get(k)]
    if missing:
        raise ValueError(f"provider config {path} is missing {', '.join(missing)}")
    extra = cfg.get("extra") or {}
    if not isinstance(extra, dict) or "api_key" in extra or "messages" in extra:
        raise ValueError(f"provider config {path}: 'extra' must be a dict of request options")
    return Provider(
        cfg["base_url"],
        cfg["api_key"],
        cfg["model"],
        token_param=cfg.get("token_param", "max_tokens"),
        temperature=cfg.get("temperature", 0.8),
        extra=extra,
        price=cfg.get("price"),
        max_usd=cfg.get("max_usd"),
        ledger_path=cfg.get("ledger") or (str(path) + ".spend.json"),
    )


@dataclass
class Completion:
    text: str
    reasoning: str | None
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    model: str = ""


class LLMError(Exception):
    """A request that failed for good. `status` is the HTTP status, or None."""

    def __init__(self, message: str, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class _Retry(Exception):
    """One failed attempt worth retrying; `after` is a server-requested delay."""

    def __init__(self, message: str, status: int | None = None, after: float | None = None):
        super().__init__(message)
        self.status = status
        self.after = after


class _Scrubber:
    """Removes a secret, and anything that looks like a credential, from text."""

    _AUTH = re.compile(r"(?i)(authorization[\"']?\s*[:=]\s*[\"']?)(?:bearer\s+)?[^\s\"',;}]+")
    _BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
    _SK = re.compile(r"\bsk-[A-Za-z0-9_-]{6,}")

    def __init__(self, secret: str):
        self._secret = secret

    def __call__(self, text: object) -> str:
        s = str(text)
        if self._secret:
            s = s.replace(self._secret, "<redacted>")
        s = self._AUTH.sub(r"\1<redacted>", s)
        s = self._BEARER.sub("Bearer <redacted>", s)
        return self._SK.sub("sk-<redacted>", s)


def _retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header, in seconds or as an HTTP date."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - time.time())


class LLMClient:
    """Chat completions with retries, a circuit breaker and bounded concurrency.

    `concurrency` bounds requests in flight across every thread that uses this
    client, not just one `complete_many` call. The retry and breaker settings are
    attributes so tests (and a cautious orchestrator) can change them.
    """

    def __init__(
        self,
        provider: Provider,
        concurrency: int = 16,
        timeout_s: float = 300,
        *,
        max_attempts: int = MAX_ATTEMPTS,
        backoff_base_s: float = BACKOFF_BASE_S,
        backoff_max_s: float = BACKOFF_MAX_S,
        breaker_threshold: int = BREAKER_THRESHOLD,
        breaker_cooldown_s: float = BREAKER_COOLDOWN_S,
    ):
        self.provider = provider
        self.concurrency = max(1, int(concurrency))
        self.timeout_s = timeout_s
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self.breaker_threshold = max(1, int(breaker_threshold))
        self.breaker_cooldown_s = breaker_cooldown_s
        self._scrub = _Scrubber(provider._api_key)
        self._slots = threading.BoundedSemaphore(self.concurrency)
        self._lock = threading.Lock()
        self._failures = 0
        self._open_until = 0.0
        self.breaker_trips = 0
        # Indirection so tests can run the backoff without waiting for it.
        self._sleep = time.sleep
        self._clock = time.monotonic

    def __repr__(self) -> str:
        return f"LLMClient({self.provider!r}, concurrency={self.concurrency})"

    # -- circuit breaker ---------------------------------------------------

    def _wait_if_open(self) -> None:
        while True:
            with self._lock:
                wait = self._open_until - self._clock()
            if wait <= 0:
                return
            self._sleep(wait)

    def _record(self, ok: bool) -> None:
        with self._lock:
            if ok:
                self._failures = 0
                return
            self._failures += 1
            if self._failures < self.breaker_threshold:
                return
            self._failures = 0
            self._open_until = self._clock() + self.breaker_cooldown_s
            self.breaker_trips += 1
        log.warning(
            "circuit open: %d consecutive failures, pausing new requests for %.0fs",
            self.breaker_threshold,
            self.breaker_cooldown_s,
        )

    # -- one request -------------------------------------------------------

    def _backoff(self, attempt: int, after: float | None) -> float:
        if after is not None:
            return min(after, RETRY_AFTER_MAX_S)
        cap = min(self.backoff_max_s, self.backoff_base_s * 2**attempt)
        return cap * random.uniform(0.5, 1.0)

    def _post(self, payload: bytes) -> dict:
        req = urllib.request.Request(
            self.provider.base_url + "/chat/completions",
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.provider._api_key,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", "replace")
            except Exception:
                body = ""
            body = self._scrub(body[:ERROR_BODY_CHARS])
            msg = f"HTTP {e.code} from {self.provider.base_url}: {body}".rstrip(": ")
            if e.code == 429 or e.code >= 500:
                after = _retry_after(e.headers.get("Retry-After") if e.headers else None)
                raise _Retry(msg, e.code, after) from None
            raise LLMError(msg, e.code) from None
        except TimeoutError as e:
            raise _Retry(f"timeout after {self.timeout_s}s ({type(e).__name__})") from None
        except urllib.error.URLError as e:
            raise _Retry(f"connection error: {self._scrub(e.reason)}") from None
        except (ConnectionError, OSError) as e:
            raise _Retry(f"connection error: {type(e).__name__}: {self._scrub(e)}") from None
        try:
            return json.loads(raw)
        except ValueError:
            snippet = self._scrub(raw[:200].decode("utf-8", "replace"))
            raise _Retry(f"unparseable response body: {snippet!r}") from None

    def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 16000,
        temperature: float | None | object = ...,
    ) -> Completion:
        p = self.provider
        body = {**p.extra, "model": p.model, "messages": messages, p.token_param: max_tokens}
        t = p.temperature if temperature is ... else temperature
        if t is not None:
            body["temperature"] = t
        payload = json.dumps(body).encode("utf-8")
        last: _Retry | None = None
        for attempt in range(self.max_attempts):
            self._wait_if_open()
            t0 = time.perf_counter()
            try:
                with self._slots:
                    body = self._post(payload)
                completion = self._parse(body, time.perf_counter() - t0)
            except _Retry as e:
                self._record(False)
                last = e
                if attempt + 1 == self.max_attempts:
                    break
                delay = self._backoff(attempt, e.after)
                log.warning(
                    "model=%s status=%s attempt %d/%d failed, retrying in %.1fs: %s",
                    self.provider.model,
                    e.status if e.status is not None else "-",
                    attempt + 1,
                    self.max_attempts,
                    delay,
                    self._scrub(e),
                )
                self._sleep(delay)
                continue
            except LLMError as e:
                # A refused request says nothing about the endpoint's health, so it
                # does not count toward the breaker.
                log.warning(
                    "model=%s status=%s not retried: %s",
                    self.provider.model,
                    e.status,
                    self._scrub(e),
                )
                raise
            self._record(True)
            u = completion.usage
            log.info(
                "model=%s status=200 latency=%.1fs prompt_tokens=%s completion_tokens=%s",
                completion.model,
                completion.latency_s,
                u.get("prompt_tokens", "?"),
                u.get("completion_tokens", "?"),
            )
            return completion
        assert last is not None
        msg = self._scrub(f"gave up after {self.max_attempts} attempts: {last}")
        log.error("model=%s status=%s %s", self.provider.model, last.status or "-", msg)
        raise LLMError(msg, last.status, retryable=True)

    def _parse(self, body: dict, latency: float) -> Completion:
        choices = body.get("choices") if isinstance(body, dict) else None
        if not choices:
            detail = self._scrub(json.dumps(body)[:ERROR_BODY_CHARS])
            raise LLMError(f"response has no choices: {detail}")
        message = choices[0].get("message") or {}
        return Completion(
            text=message.get("content") or "",
            reasoning=message.get("reasoning_content") or None,
            usage=body.get("usage") or {},
            latency_s=latency,
            model=body.get("model") or self.provider.model,
        )

    # -- many requests -----------------------------------------------------

    def complete_many(self, batch: list[list[dict]], **kw) -> list[Completion | Exception]:
        """Run `complete` over `batch` concurrently; a failure is returned in its slot."""
        if not batch:
            return []

        def one(messages):
            try:
                return self.complete(messages, **kw)
            except Exception as e:  # returned, not raised, so one bad item costs one slot
                return e

        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(batch))) as pool:
            return list(pool.map(one, batch))


_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", re.DOTALL)
_BUILD = re.compile(r"^[ \t]*def[ \t]+build[ \t]*\(", re.MULTILINE)
_CODE_TAGS = {"", "python", "py", "python3"}


def _last_build_block(text: str) -> str | None:
    found = None
    for m in _FENCE.finditer(text):
        if m.group(1).lower() in _CODE_TAGS and _BUILD.search(m.group(2)):
            found = m.group(2)
    return found


def extract_code(text: str | None) -> str | None:
    """The last fenced python (or untagged) block that defines `build`, else None.

    Reasoning models often draft code before answering, so the last block wins.
    If the text has a `</think>` marker, the answer after it is searched first.
    """
    if not text:
        return None
    text = text.replace("\r\n", "\n").replace("\ufeff", "")
    code = None
    if "</think>" in text:
        code = _last_build_block(text.rsplit("</think>", 1)[1])
    if code is None:
        code = _last_build_block(text)
    if code is None:
        return None
    return textwrap.dedent(code).strip("\n") + "\n"
