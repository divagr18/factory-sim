"""The LLM client against a local fake endpoint: retries, breaker, and no key leaks."""

import json
import logging
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from evolve import llm

KEY = "sk-test-SECRET123"


class FakeServer:
    """A `/chat/completions` stand-in. `script` is a list of (status, body, headers);
    the last entry repeats. A body may be a callable of the request handler."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(n))
                with outer.lock:
                    outer.requests.append((self.path, dict(self.headers), payload))
                    i = len(outer.requests) - 1
                    status, body, headers = outer.script[min(i, len(outer.script) - 1)]
                if callable(body):
                    body = body(self, payload)
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/compatible-mode/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class FakeServerStub:
    url = "http://127.0.0.1:9/v1"


def ok_body(content="hello", reasoning=None):
    msg = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return {
        "model": "fake-model",
        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


@pytest.fixture
def serve():
    servers = []

    def make(script):
        s = FakeServer(script)
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.close()


class FakeTime:
    """Sleeping advances a fake clock instead of waiting."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s

    def clock(self):
        return self.now


def client_for(server, **kw):
    c = llm.LLMClient(llm.Provider(server.url, KEY, "fake-model"), timeout_s=5, **kw)
    t = FakeTime()
    c._sleep, c._clock = t.sleep, t.clock
    return c, t


MSGS = [{"role": "user", "content": "hi"}]


def test_normal_response_with_reasoning(serve):
    s = serve([(200, ok_body("the answer", "thinking..."), None)])
    c, _ = client_for(s)
    out = c.complete(MSGS, max_tokens=123, temperature=0.5)
    assert out.text == "the answer"
    assert out.reasoning == "thinking..."
    assert out.usage["completion_tokens"] == 7
    assert out.model == "fake-model" and out.latency_s >= 0
    path, headers, payload = s.requests[0]
    assert path == "/compatible-mode/v1/chat/completions"
    assert headers["Authorization"] == "Bearer " + KEY
    assert payload == {
        "model": "fake-model",
        "messages": MSGS,
        "max_tokens": 123,
        "temperature": 0.5,
    }


def test_reasoning_absent_is_none(serve):
    s = serve([(200, ok_body("x"), None)])
    c, _ = client_for(s)
    assert c.complete(MSGS).reasoning is None


def test_429_then_200_backs_off(serve):
    s = serve(
        [
            (429, {"error": "slow down"}, {"Retry-After": "7"}),
            (429, {"error": "slow down"}, None),
            (200, ok_body("done"), None),
        ]
    )
    c, t = client_for(s)
    assert c.complete(MSGS).text == "done"
    assert len(s.requests) == 3
    assert t.sleeps[0] == 7.0  # Retry-After honoured
    assert 0 < t.sleeps[1] <= c.backoff_base_s * 2  # jittered backoff for attempt 2


def test_500_exhausts_retries(serve):
    s = serve([(500, {"error": "boom"}, None)])
    c, t = client_for(s, max_attempts=4)
    with pytest.raises(llm.LLMError) as ei:
        c.complete(MSGS)
    assert ei.value.status == 500 and ei.value.retryable
    assert "4 attempts" in str(ei.value)
    assert len(s.requests) == 4
    assert len(t.sleeps) == 3


def test_400_not_retried(serve):
    s = serve([(400, {"error": "bad request"}, None)])
    c, t = client_for(s)
    with pytest.raises(llm.LLMError) as ei:
        c.complete(MSGS)
    assert ei.value.status == 400 and not ei.value.retryable
    assert len(s.requests) == 1 and not t.sleeps


@pytest.mark.parametrize(
    "exc, match",
    [
        (urllib.error.URLError(ConnectionRefusedError("refused")), "connection error"),
        (TimeoutError("timed out"), "timeout"),
        (ConnectionResetError("reset"), "connection error"),
    ],
)
def test_network_errors_are_retried(monkeypatch, exc, match):
    # Patched rather than real: a refused connect takes seconds on Windows.
    calls = []

    def urlopen(req, timeout):
        calls.append(req)
        raise exc

    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    c, t = client_for(FakeServerStub(), max_attempts=3)
    with pytest.raises(llm.LLMError, match=match):
        c.complete(MSGS)
    assert len(calls) == 3 and len(t.sleeps) == 2


def test_key_never_leaks(serve, caplog, tmp_path, monkeypatch):
    def echo(handler, payload):
        auth = handler.headers.get("Authorization")
        return {
            "error": {"message": f"invalid key {auth}", "raw": auth.split()[-1]},
            "headers": {"Authorization": auth},
        }

    caplog.set_level(logging.DEBUG, logger="evolve.llm")
    cfg = tmp_path / "qwen.json"
    s = serve([(401, echo, None)])
    cfg.write_text(json.dumps({"base_url": s.url, "api_key": KEY, "model": "m"}))
    monkeypatch.setenv(llm.CONFIG_ENV, str(cfg))
    provider = llm.load_provider()
    assert KEY not in repr(provider) and KEY not in str(provider)
    assert "api_key=<redacted>" in repr(provider)

    c, _ = client_for(s)
    c.provider = provider
    with pytest.raises(llm.LLMError) as ei:
        c.complete(MSGS)
    assert "invalid key" in str(ei.value)  # the body is echoed, only scrubbed
    assert KEY not in str(ei.value) and KEY not in repr(ei.value)
    assert "SECRET123" not in str(ei.value)
    assert KEY not in repr(c)

    # A retried echo exercises the warning and error log paths too.
    s2 = serve([(503, echo, None)])
    c2, _ = client_for(s2, max_attempts=3)
    with pytest.raises(llm.LLMError) as ei2:
        c2.complete(MSGS)
    assert KEY not in str(ei2.value)
    assert caplog.records
    for rec in caplog.records:
        assert "SECRET123" not in rec.getMessage()
    assert "SECRET123" not in caplog.text


def test_scrubber_patterns():
    scrub = llm._Scrubber("sk-live-abcdef")
    text = 'Authorization: Bearer sk-live-abcdef, "authorization":"Bearer zzz9", bearer tok.en'
    out = scrub(text)
    assert "abcdef" not in out and "zzz9" not in out and "tok.en" not in out


def test_complete_many_order_and_errors(serve):
    def by_content(handler, payload):
        content = payload["messages"][0]["content"]
        return ok_body("echo:" + content)

    s = serve([(200, by_content, None)])
    c, _ = client_for(s, concurrency=4)
    batch = [[{"role": "user", "content": str(i)}] for i in range(10)]
    out = c.complete_many(batch)
    assert [o.text for o in out] == [f"echo:{i}" for i in range(10)]

    # Per-item failures stay in their slots.
    orig = c.complete

    def flaky(messages, **kw):
        if messages[0]["content"] in ("3", "7"):
            raise llm.LLMError("nope", 400)
        return orig(messages, **kw)

    c.complete = flaky
    out = c.complete_many(batch, max_tokens=50)
    assert isinstance(out[3], llm.LLMError) and isinstance(out[7], llm.LLMError)
    assert [o.text for i, o in enumerate(out) if i not in (3, 7)] == [
        f"echo:{i}" for i in range(10) if i not in (3, 7)
    ]
    assert s.requests[-1][2]["max_tokens"] == 50
    assert c.complete_many([]) == []


def test_circuit_breaker_trips(serve, caplog):
    caplog.set_level(logging.WARNING, logger="evolve.llm")
    s = serve([(500, {"error": "down"}, None)] * 3 + [(200, ok_body("back"), None)])
    c, t = client_for(s, breaker_threshold=3, breaker_cooldown_s=42.0, max_attempts=6)
    assert c.complete(MSGS).text == "back"
    assert c.breaker_trips == 1
    assert "circuit open" in caplog.text
    # After the third failure the next attempt waits out the cooldown before sending.
    assert t.now >= 42.0
    assert c._failures == 0


def test_load_provider_errors(tmp_path, monkeypatch):
    monkeypatch.delenv(llm.CONFIG_ENV, raising=False)
    with pytest.raises(RuntimeError, match=llm.CONFIG_ENV):
        llm.load_provider()
    with pytest.raises(FileNotFoundError):
        llm.load_provider(str(tmp_path / "missing.json"))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"base_url": "http://x", "api_key": KEY}))
    with pytest.raises(ValueError, match="model") as ei:
        llm.load_provider(str(bad))
    assert KEY not in str(ei.value)
    broken = tmp_path / "broken.json"
    broken.write_text('{"api_key": "' + KEY + '", ')
    with pytest.raises(ValueError) as ei:
        llm.load_provider(str(broken))
    assert KEY not in str(ei.value) and ei.value.__cause__ is None


def test_load_provider_from_env(tmp_path, monkeypatch):
    cfg = tmp_path / "qwen.json"
    cfg.write_text(
        json.dumps({"base_url": "http://127.0.0.1:1/v1/", "api_key": KEY, "model": "qwen-x"})
    )
    monkeypatch.setenv(llm.CONFIG_ENV, str(cfg))
    p = llm.load_provider()
    assert p.base_url == "http://127.0.0.1:1/v1" and p.model == "qwen-x"
    assert p._api_key == KEY
    assert KEY not in repr(p)


def test_provider_options_shape_the_request(tmp_path):
    """OpenAI reasoning models want `max_completion_tokens`, no temperature, and
    take provider-specific extras; the config decides, not the code."""
    import json as _json

    cfg = tmp_path / "p.json"
    cfg.write_text(
        _json.dumps(
            {
                "base_url": "http://x/v1",
                "api_key": "sk-test-SECRET123",
                "model": "m",
                "token_param": "max_completion_tokens",
                "temperature": None,
                "extra": {"seed": 7},
            }
        )
    )
    provider = llm.load_provider(str(cfg))
    client = llm.LLMClient(provider)
    sent = {}

    def fake_post(payload):
        sent.update(_json.loads(payload))
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    client._post = fake_post
    client.complete([{"role": "user", "content": "hi"}], max_tokens=123)
    assert sent["max_completion_tokens"] == 123 and "max_tokens" not in sent
    assert "temperature" not in sent
    assert sent["seed"] == 7
    assert "sk-test-SECRET123" not in repr(provider)


def test_extra_may_not_smuggle_a_key_or_messages(tmp_path):
    import json as _json

    cfg = tmp_path / "p.json"
    cfg.write_text(
        _json.dumps({"base_url": "u", "api_key": "k", "model": "m", "extra": {"api_key": "x"}})
    )
    with pytest.raises(ValueError, match="extra"):
        llm.load_provider(str(cfg))
