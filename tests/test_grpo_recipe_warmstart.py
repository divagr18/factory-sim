"""The SFT warm-start data pipeline in recipes/grpo-factorio-build.

Covers rationalize.py, make_replay.py and sft_data.py (sft_lora.py's examples,
loss masks and replay mix). No GPU and no model: the chat endpoint is a local
fake OpenAI server that returns canned replies, broken ones included, and
programs are scored in-process by the env's own scorer. The loss-mask tests
load the real Qwen3.5 tokenizer from the Hub (tokenizer files only) and skip
without `transformers`, `jinja2` or network.
"""

from __future__ import annotations

import http.server
import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes" / "grpo-factorio-build"
for path in (RECIPE, ROOT / "integrations" / "verifiers" / "factorio_build"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import gen_common as gc  # noqa: E402
import make_replay  # noqa: E402
import rationalize as rz  # noqa: E402
import sft_data  # noqa: E402
from factorio_build import core  # noqa: E402

from evolve.llm import extract_code  # noqa: E402
from evolve.seeds import builder, trivial  # noqa: E402

TASK = "construct_smelting_line"
MODEL = "Qwen/Qwen3.5-9B"
REF = builder.SOURCE.strip() + "\n"

# ------------------------------------------------------------------ helpers


class FakeServer:
    """A local `/v1` endpoint. `responder(body) -> (status, payload)`, where a
    payload is a dict (sent as JSON) or raw bytes. Every request body is kept."""

    def __init__(self, responder, model: str = MODEL):
        self.requests: list[dict] = []
        lock = threading.Lock()
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, payload):
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.rstrip("/").endswith("/models"):
                    self._send(200, {"data": [{"id": model}]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with lock:
                    server.requests.append(body)
                self._send(*responder(body))

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def serve():
    servers = []

    def start(responder, **kw):
        s = FakeServer(responder, **kw)
        servers.append(s)
        return s

    yield start
    for s in servers:
        s.close()


def completion(choices) -> dict:
    """An OpenAI chat response; `choices` holds texts or (text, finish_reason) pairs."""
    out = []
    for i, c in enumerate(choices):
        text, finish = (c, "stop") if isinstance(c, str) or c is None else c
        out.append(
            {
                "index": i,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish,
            }
        )
    return {"choices": out, "usage": {"prompt_tokens": 100, "completion_tokens": 50 * len(out)}}


def reply(prose: str, code: str, after: str = "") -> str:
    return f"{prose}\n\n```python\n{code.strip()}\n```{after}"


def words(n: int, word: str = "step") -> str:
    return " ".join(f"{word}{i}" for i in range(n))


LONG = "First I find the ore patch, then I pick a drill spot. " + words(60)
SHORT = "I walk to the patch, place a drill and a furnace, and fuel both. " + words(30)
MID = "The line needs a drill over ore and a furnace under its drop point. " + words(40)
TINY = "Here it is."
# Same AST as the reference (a comment only), and a different AST that behaves the same.
REF_COMMENTED = REF.replace(
    "    radius = 5\n", "    # how far to look for ore\n    radius = 5\n", 1
)
REF_DIFFERENT = REF.replace("    radius = 5\n", "    spare = 0\n    radius = 5\n", 1)
SYNTAX_ERROR = "def build(world):\n    x = (\n"
VIOLATION = "import os\n\ndef build(world):\n    os.system('echo hi')\n"


def example_rows(indices=(0, 2), n_scenes: int = 4, notes: bool = True) -> list[dict]:
    """build_sft_data.py rows over train subsets, with the seed builder as the target."""
    rows = core.rows(TASK, "train", n_scenes, max(indices) + 1, 0, notes)
    out = []
    for i in indices:
        r = rows[i]
        prompt = [
            {"role": "system", "content": r["system_prompt"]},
            {"role": "user", "content": r["prompt"]},
        ]
        completion_msg = [{"role": "assistant", "content": f"```python\n{REF.strip()}\n```"}]
        out.append({"prompt": prompt, "completion": completion_msg, "train_mean": 0.9})
    return out


def write_rows(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def subset_of(body: dict) -> str:
    return rz.SUBSET_RE.search(body["messages"][1]["content"]).group(0)


def rz_args(tmp_path: Path, data: Path, server: FakeServer, *extra) -> list[str]:
    return [
        "--data", str(data),
        "--out-dir", str(tmp_path / "out"),
        "--base-url", server.base_url,
        "--score-workers", "0",
        "--tokenizer", "none",
        "--concurrency", "4",
        "--backoff", "0",
        "--log-every", "1",
        *extra,
    ]  # fmt: skip


# ------------------------------------------------------------------ resolving examples


def test_resolve_rebuilds_the_env_row():
    rows = core.rows(TASK, "train", 4, 3, 0, True, rz.PROMPT_VERSION)  # asked with v2
    ex = rz.resolve(example_rows((2,))[0], 7)
    assert ex.subset_id == rows[2]["subset_id"] == f"{TASK}/train/8-11"
    assert ex.scenes == [list(s) for s in rows[2]["scenes"]]
    assert ex.messages == [
        {"role": "system", "content": rows[2]["system_prompt"]},
        {"role": "user", "content": rows[2]["prompt"]},
    ]
    assert ex.reference_code == extract_code(f"```python\n{REF}```")
    assert ex.id.startswith("00007-")
    # The request adds exactly one message, and it holds the reference program.
    msgs = rz.request_messages(ex, hinted=True)
    assert msgs[:2] == ex.messages and len(msgs) == 3 and msgs[2]["role"] == "user"
    assert f"```python\n{ex.reference_code}```" in msgs[2]["content"]
    assert rz.request_messages(ex, hinted=False) == ex.messages


def test_resolve_accepts_prompts_without_game_notes():
    ex = rz.resolve(example_rows((1,), notes=False)[0], 0)
    assert ex.messages[0]["content"] == core.system_prompt(False)


def test_resolve_refuses_prompts_the_env_does_not_serve():
    good = example_rows((0,))[0]
    stale = json.loads(json.dumps(good))
    stale["prompt"][0]["content"] += "\nAn extra line."
    with pytest.raises(rz.PromptMismatch):
        rz.resolve(stale, 0)
    wrong_scenes = json.loads(json.dumps(good))
    wrong_scenes["prompt"][1]["content"] = good["prompt"][1]["content"].replace("0..3", "0..4")
    with pytest.raises(rz.PromptMismatch):
        rz.resolve(wrong_scenes, 0)
    val = json.loads(json.dumps(good))
    val_row = core.rows(TASK, "val", 4, 1, 0, True)[0]
    val["prompt"][1]["content"] = val_row["prompt"]
    with pytest.raises(ValueError, match="train"):
        rz.resolve(val, 0)
    no_code = json.loads(json.dumps(good))
    no_code["completion"][0]["content"] = "no program here"
    with pytest.raises(ValueError, match="no program"):
        rz.resolve(no_code, 0)


# ------------------------------------------------------------------ reply facts


def test_facts_read_the_reply_the_way_the_parser_does():
    f = rz.facts(reply(LONG, REF), REF)
    assert f["n_programs"] == 1 and f["same_as_hint"] == "exact" and not f["hint_leak"]
    assert f["reasoning_words"] == len(LONG.split()) and f["trailing_chars"] == 0
    assert rz.facts(reply(LONG, REF_COMMENTED), REF)["same_as_hint"] == "normalized"
    assert rz.facts(reply(LONG, REF_DIFFERENT), REF)["same_as_hint"] == "different"
    two = reply(LONG, trivial.SOURCE) + "\n\nBetter:\n" + reply(MID, REF, after="\nDone.")
    f2 = rz.facts(two, REF)
    assert f2["n_programs"] == 2 and f2["same_as_hint"] == "exact"  # the last block counts
    assert f2["trailing_chars"] == len("Done.")
    none = rz.facts(LONG, REF)
    assert none["n_programs"] == 0 and none["same_as_hint"] is None
    assert rz.facts("", REF)["reasoning_words"] == 0


@pytest.mark.parametrize(
    "text, leak",
    [
        ("The reference program walks to the patch first.", True),
        ("Like the provided solution, I place the drill.", True),
        ("I adapted the given code slightly.", True),
        ("Following the hint, I fuel both.", True),
        ("Your program only needs one drill.", True),
        ("Given the constraints, I place the drill first.", False),
        ("The program walks to the patch, then places a drill.", False),
        ("I use the drill's drop point as a reference tile.", False),
    ],
)
def test_leak_filter(text, leak):
    assert rz.facts(reply(text + " " + words(40), REF), REF)["hint_leak"] is leak


# ------------------------------------------------------------------ the client


def test_client_retries_server_errors_and_bad_bodies(serve):
    calls = {"n": 0}

    def flaky(body):
        calls["n"] += 1
        if calls["n"] == 1:
            return 500, {"error": "overloaded"}
        if calls["n"] == 2:
            return 200, b"not json"
        if calls["n"] == 3:
            return 200, {"choices": []}
        return 200, completion(["ok"])

    s = serve(flaky)
    client = gc.ChatClient(s.base_url, max_attempts=5, backoff_s=0)
    r = client.chat_sync({"model": MODEL, "messages": []})
    assert r.error is None and r.attempts == 4 and r.choices[0].text == "ok"
    assert r.choices[0].finish_reason == "stop" and r.completion_tokens == 50
    assert client.served_model() == MODEL
    client.close()


def test_client_gives_up_cleanly(serve):
    s = serve(lambda body: (400, {"error": "bad request"}))
    client = gc.ChatClient(s.base_url, max_attempts=5, backoff_s=0)
    r = client.chat_sync({"model": MODEL, "messages": []})
    assert r.attempts == 1 and r.error.startswith("HTTP 400") and not r.choices
    down = serve(lambda body: (503, {"error": "down"}))
    client2 = gc.ChatClient(down.base_url, max_attempts=3, backoff_s=0)
    r2 = client2.chat_sync({"model": MODEL, "messages": []})
    assert r2.attempts == 3 and "gave up" in r2.error and len(down.requests) == 3
    client.close()
    client2.close()


def test_client_unreachable_host_is_an_error_not_an_exception():
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()  # nothing listens on this port now
    client = gc.ChatClient(f"http://127.0.0.1:{port}/v1", max_attempts=2, backoff_s=0)
    r = client.chat_sync({"model": MODEL, "messages": []})
    assert r.attempts == 2 and r.error.startswith("gave up") and not r.choices
    client.close()


# ------------------------------------------------------------------ rationalize end to end

# Example 0 (scenes 0-3, reference success 1.0) and example 1 (scenes 8-11, 0.75).
CANNED = {
    "construct_smelting_line/train/0-3": [
        reply(LONG, REF),  # passes, exact copy
        reply(SHORT, REF),  # passes, exact copy, shortest reasoning: kept
        reply("The reference program walks first. " + words(40), REF),  # hint_leak
        reply(LONG, trivial.SOURCE),  # below_reference, zero_success
        reply(TINY, REF),  # short_reasoning
        (reply(LONG, REF), "length"),  # truncated
    ],
    "construct_smelting_line/train/8-11": [
        LONG,  # no_program
        reply(LONG, SYNTAX_ERROR),  # sandbox
        reply(LONG, VIOLATION),  # sandbox
        reply(LONG, trivial.SOURCE) + "\n\n" + reply(MID, REF),  # several_programs
        reply(LONG, REF_COMMENTED),  # passes, normalized copy
        reply(MID, REF_DIFFERENT),  # passes, a different program, shorter: kept
    ],
}
EXPECTED_REASONS = {
    "construct_smelting_line/train/0-3": [
        [],
        [],
        ["hint_leak"],
        ["below_reference", "zero_success"],
        ["short_reasoning"],
        ["truncated"],
    ],
    "construct_smelting_line/train/8-11": [
        ["no_program"],
        ["sandbox"],
        ["sandbox"],
        ["several_programs"],
        [],
        [],
    ],
}


def canned_responder(body):
    subset = subset_of(body).removeprefix("scene subset ").removesuffix(" (")
    choices = CANNED[subset]
    assert body["n"] == len(choices)
    return 200, completion(choices)


def test_rationalize_end_to_end(tmp_path, serve):
    s = serve(canned_responder)
    data = write_rows(tmp_path / "sft.jsonl", example_rows((0, 2)))
    stats = rz.main(rz_args(tmp_path, data, s, "--n", "6", "--limit", "1"))
    assert len(s.requests) == 1 and stats["examples"] == 1
    stats = rz.main(rz_args(tmp_path, data, s, "--n", "6"))  # resumes: only example 1 is new
    assert len(s.requests) == 2 and stats["examples"] == 2

    examples = [rz.resolve(r, i) for i, r in enumerate(gc.read_jsonl(str(data)))]
    for body, ex in zip(sorted(s.requests, key=subset_of), examples, strict=True):
        assert body["model"] == MODEL and body["temperature"] == 0.7 and body["n"] == 6
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["max_tokens"] == 12288 and "seed" in body
        assert body["messages"] == rz.request_messages(ex, hinted=True)

    out = tmp_path / "out"
    records = {r["subset_id"]: r for r in gc.read_jsonl(str(out / "samples.jsonl"))}
    rules = rz.KeepRules()
    for subset, rec in records.items():
        ref = rec["reference"]["metrics"]["success"]
        got = [rz.reject_reasons(smp, ref, rules) for smp in rec["samples"]]
        assert got == EXPECTED_REASONS[subset], subset
        assert rec["hint_message"] == rz.request_messages(examples[0], True)[2]["content"]
    assert records["construct_smelting_line/train/0-3"]["reference"]["metrics"]["success"] == 1.0
    assert records["construct_smelting_line/train/8-11"]["reference"]["metrics"]["success"] == 0.75

    sft = gc.read_jsonl(str(out / "sft.jsonl"))
    assert len(sft) == 2
    for row, ex in zip(sft, examples, strict=True):
        assert row["messages"][:2] == ex.messages  # the prompt without the hint
        assert len(row["messages"]) == 3 and row["messages"][2]["role"] == "assistant"
    # Among the best-scoring passes, the one whose reasoning is closest to their median length.
    assert sft[0]["messages"][2]["content"] == CANNED["construct_smelting_line/train/0-3"][0]
    assert sft[1]["messages"][2]["content"] == CANNED["construct_smelting_line/train/8-11"][4]
    assert [r["meta"]["same_as_hint"] for r in sft] == ["exact", "normalized"]
    assert [r["meta"]["success"] for r in sft] == [1.0, 0.75]
    # What sft_lora.py reads: thinking off, the reply as the target.
    loaded = sft_data.load_examples(str(out / "sft.jsonl"))
    assert [e["enable_thinking"] for e in loaded] == [False, False]

    assert stats["kept_examples"] == 2 and stats["kept_samples"] == 2
    assert stats["samples"] == 12 and stats["reject_reasons"]["passed"] == 4
    assert stats["reject_reasons"]["sandbox"] == 2
    assert stats["kept_vs_hint"]["exact"] == 1 and stats["kept_vs_hint"]["normalized"] == 1
    assert stats["kept_vs_hint"]["share_identical"] == 1.0  # exact + normalized
    assert stats["hint"]["template"] == rz.HINT_TEMPLATE
    assert stats["reply_tokens"]["all"] == {"n": 0}  # tokenizer off
    assert stats["reasoning_words"]["kept"]["n"] == 2
    run = json.loads((out / "run.json").read_text())
    assert run["hint_template"] == rz.HINT_TEMPLATE and run["model"] == MODEL

    # A rerun asks nothing; a rebuild re-selects without a server.
    rz.main(rz_args(tmp_path, data, s, "--n", "6"))
    assert len(s.requests) == 2
    stats2 = rz.main(rz_args(tmp_path, data, s, "--n", "6", "--rebuild", "--keep-per-example", "2"))
    assert stats2["kept_samples"] == 4
    assert {k: stats2["kept_vs_hint"][k] for k in ("exact", "normalized", "different")} == {
        "exact": 2,
        "normalized": 1,
        "different": 1,
    }
    # Other sampling settings in the same directory are refused.
    with pytest.raises(SystemExit, match="temperature"):
        rz.main(rz_args(tmp_path, data, s, "--n", "6", "--temperature", "0.9"))


def test_rationalize_fallback_mode_and_request_errors(tmp_path, serve):
    """STaR's order: unhinted first, the hint only where nothing unhinted passes."""

    def responder(body):
        subset = subset_of(body)
        hinted = len(body["messages"]) == 3
        if "/train/0-3" in subset:  # unhinted fails, hinted passes
            if hinted:
                return 200, completion([reply(SHORT, REF), reply(LONG, REF_DIFFERENT)])
            return 200, completion([reply(LONG, trivial.SOURCE), LONG])
        if "/train/8-11" in subset:  # unhinted passes: no hinted request
            return 200, completion([reply(MID, REF_DIFFERENT), None])
        return 400, {"error": "context length exceeded"}

    s = serve(responder)
    data = write_rows(tmp_path / "sft.jsonl", example_rows((0, 2, 4)))
    stats = rz.main(
        rz_args(tmp_path, data, s, "--n", "2", "--hint-mode", "fallback", "--round", "2")
    )
    by_subset = {}
    for body in s.requests:
        by_subset.setdefault(subset_of(body), []).append(len(body["messages"]) == 3)
    # 16-19 errors unhinted, so it is asked again with the hint (and errors again).
    assert {k.split("/")[-1][:-2]: v for k, v in by_subset.items()} == {
        "0-3": [False, True],
        "8-11": [False],
        "16-19": [False, True],
    }

    recs = {
        r["subset_id"].split("/")[-1]: r
        for r in gc.read_jsonl(str(tmp_path / "out" / "samples.jsonl"))
    }
    assert [smp["hinted"] for smp in recs["0-3"]["samples"]] == [False, False, True, True]
    assert [smp["k"] for smp in recs["0-3"]["samples"]] == [0, 1, 2, 3]
    assert [r["hinted"] for r in recs["0-3"]["requests"]] == [False, True]
    assert all(smp["error"] for smp in recs["16-19"]["samples"])  # the 400, recorded
    sft = {
        r["meta"]["subset_id"].split("/")[-1]: r
        for r in gc.read_jsonl(str(tmp_path / "out" / "sft.jsonl"))
    }
    assert set(sft) == {"0-3", "8-11"}
    assert sft["0-3"]["meta"]["hinted"] is True and sft["0-3"]["meta"]["k"] == 3
    assert sft["8-11"]["meta"]["hinted"] is False and sft["8-11"]["meta"]["round"] == 2
    assert stats["reject_reasons"]["request_error"] == 4
    assert stats["unhinted"] == {"samples": 6, "kept": 1}
    assert stats["hinted"] == {"samples": 4, "kept": 1}
    assert stats["kept_example_rate"] == round(2 / 3, 4)


# ------------------------------------------------------------------ make_replay


@pytest.mark.parametrize(
    "prompt",
    [
        "I am playing with a set of blocks where I need to arrange the blocks into stacks.",
        "Solve this Blocksworld instance: unstack C from A.",
        "[STATEMENT] As initial conditions I have that, the red block is clear.",
        "Write a PDDL domain for a delivery robot.",
        "Pick up block A and put it on top of block B.",
        "What is the goal state of this planning problem?",
        "Explain the Towers of Hanoi recursion.",
    ],
)
def test_replay_screen_drops_planning_lookalikes(prompt):
    kept, dropped = make_replay.screen([{"id": "x", "prompt": prompt}])
    assert not kept and dropped["planning_like"] == 1


def test_replay_screen_keeps_ordinary_prompts_and_drops_duplicates():
    rows = [
        {"id": "a", "prompt": "Write a haiku about autumn leaves."},
        {"id": "b", "prompt": "  write a haiku about   autumn leaves. "},
        {"id": "c", "prompt": "Hi"},
        {"id": "d", "prompt": "How do I format a code block in Markdown?"},
        {
            "id": "e",
            "prompt": "A train leaves at 3 pm at 60 km/h. When does it arrive 150 km away?",
        },
    ]
    kept, dropped = make_replay.screen(rows)
    assert [p["id"] for p in kept] == ["a", "d", "e"]
    assert dropped == {"duplicate": 1, "length": 1}


def test_oasst_filter():
    base = {
        "role": "prompter",
        "parent_id": None,
        "lang": "en",
        "deleted": False,
        "synthetic": False,
        "review_result": True,
        "labels": {"name": ["spam", "quality"], "value": [0.0, 0.8], "count": [3, 3]},
        "detoxify": {"toxicity": 0.01},
    }
    assert make_replay._oasst_ok(base)
    for change in (
        {"role": "assistant"},
        {"parent_id": "abc"},
        {"lang": "de"},
        {"deleted": True},
        {"synthetic": True},
        {"review_result": False},
        {"labels": {"name": ["spam"], "value": [0.75], "count": [4]}},
        {"detoxify": {"toxicity": 0.9}},
    ):
        assert not make_replay._oasst_ok({**base, **change}), change
    assert make_replay._oasst_ok({**base, "labels": None, "detoxify": None, "parent_id": "None"})


REPLAY_PROMPTS = {
    "p0": "Write a short poem about the sea.",
    "p1": "Explain photosynthesis to a ten-year-old.",
    "p2": "A shop sells pens at 3 dollars each. How much do 7 pens cost?",
    "p3": "Give me three tips for learning a language.",
    "p4": "What causes the seasons on Earth?",
    "p5": "Summarise the plot of a heist film in two sentences.",
    "p6": "Stack block A on top of block B, then unstack it.",  # screened out
}


def replay_responder(body):
    prompt = body["messages"][0]["content"]
    think = body["chat_template_kwargs"]["enable_thinking"]
    key = next(k for k, v in REPLAY_PROMPTS.items() if v == prompt)
    if think:
        if key in ("p0", "p3"):
            return 200, completion([f"Let me think about {key}.\n</think>\n\nAnswer {key}."])
        if key == "p1":  # the server's reasoning parser split it
            msg = {
                "role": "assistant",
                "content": "Answer p1.",
                "reasoning_content": "Thinking p1.",
            }
            return 200, {"choices": [{"message": msg, "finish_reason": "stop"}]}
        return 200, completion([f"Still thinking about {key} and never closing"])
    if key == "p2":
        return 200, completion([("7 pens cost", "length")])
    if key == "p4":
        return 200, completion([""])
    return 200, completion([f"Answer {key}, with thinking off."])


def test_make_replay_end_to_end(tmp_path, serve):
    s = serve(replay_responder)
    prompts = tmp_path / "prompts_in.jsonl"
    write_rows(
        prompts, [{"id": k, "prompt": v, "licence": "CC0-1.0"} for k, v in REPLAY_PROMPTS.items()]
    )
    args = [
        "--out-dir", str(tmp_path / "replay"),
        "--base-url", s.base_url,
        "--prompts", str(prompts),
        "--n-prompts", "6",
        "--think-frac", "0.5",
        "--tokenizer", "none",
        "--backoff", "0",
    ]  # fmt: skip
    stats = make_replay.main(args)
    assert len(s.requests) == 6  # p6 was screened out
    assert stats["selection"]["sources"]["local"]["dropped"] == {"planning_like": 1}
    chosen = gc.read_jsonl(str(tmp_path / "replay" / "prompts.jsonl"))
    think = {p["id"] for p in chosen if p["think"]}
    assert len(think) == 3 and "p6" not in {p["id"] for p in chosen}
    for body in s.requests:
        prompt_id = next(
            k for k, v in REPLAY_PROMPTS.items() if v == body["messages"][0]["content"]
        )
        thinks = body["chat_template_kwargs"]["enable_thinking"]
        assert thinks == (prompt_id in think)
        assert body["temperature"] == (0.6 if thinks else 0.7)
        assert body["max_tokens"] == (16384 if thinks else 4096)

    rows = {r["meta"]["id"]: r for r in gc.read_jsonl(str(tmp_path / "replay" / "replay.jsonl"))}
    expected_kept = set()
    for pid in ("p0", "p1", "p2", "p3", "p4", "p5"):
        if pid in think and pid in ("p0", "p1", "p3"):
            expected_kept.add(pid)
        if pid not in think and pid not in ("p2", "p4"):
            expected_kept.add(pid)
    assert set(rows) == expected_kept
    for pid, row in rows.items():
        user, assistant = row["messages"]
        assert user == {"role": "user", "content": REPLAY_PROMPTS[pid]}
        assert row["enable_thinking"] == (pid in think)
        assert row["meta"]["licence"] == "CC0-1.0"
        if pid in think:
            assert assistant["content"] == f"Answer {pid}."
            assert assistant["reasoning_content"] in (f"Let me think about {pid}.", "Thinking p1.")
        else:
            assert "reasoning_content" not in assistant
    loaded = sft_data.load_examples(str(tmp_path / "replay" / "replay.jsonl"))
    assert sorted(e["enable_thinking"] for e in loaded) == sorted(
        r["enable_thinking"] for r in rows.values()
    )
    assert stats["kept"] == len(expected_kept)
    assert sum(stats["reasons"].values()) == 6

    make_replay.main(args)  # resume: nothing new to ask
    assert len(s.requests) == 6
    with pytest.raises(SystemExit, match="n_prompts"):
        make_replay.main([*args, "--n-prompts", "5"])  # the prompt set is frozen


def test_split_reply():
    assert make_replay.split_reply("plain", None, False) == ("plain", None, None)
    assert make_replay.split_reply("a</think>b", None, False)[2] == "stray_think"
    assert make_replay.split_reply("r\n</think>\n\nans", None, True) == ("ans", "r", None)
    assert make_replay.split_reply("<think>\nr\n</think>\n\nans", None, True) == ("ans", "r", None)
    assert make_replay.split_reply("ans", "r", True) == ("ans", "r", None)
    assert make_replay.split_reply("no end", None, True)[2] == "unclosed_think"


# ------------------------------------------------------------------ sft_data: rows and mix


def test_load_examples_takes_both_formats(tmp_path):
    user = {"role": "user", "content": "q"}
    answer = {"role": "assistant", "content": "a"}
    rows = [
        {"prompt": [user], "completion": [answer]},
        {"messages": [user, answer]},
        {"messages": [user, {**answer, "reasoning_content": "r"}]},
        {"messages": [user, answer], "enable_thinking": True},
    ]
    got = sft_data.load_examples(str(write_rows(tmp_path / "x.jsonl", rows)))
    assert [e["enable_thinking"] for e in got] == [False, False, True, True]
    assert got[0]["messages"] == got[1]["messages"]
    bad = write_rows(tmp_path / "bad.jsonl", [{"messages": [{"role": "user", "content": "q"}]}])
    with pytest.raises(ValueError, match="assistant"):
        sft_data.load_examples(str(bad))


def test_epoch_plan_without_replay_is_the_old_order():
    import random

    rng = random.Random(0)
    old = []
    for _ in range(2):
        order = list(range(10))
        rng.shuffle(order)
        old.append(order)
    plan = sft_data.epoch_plan(10, 0, 0.2, 2, seed=0)
    assert [[i for _, i in epoch] for epoch in plan] == old
    assert sft_data.epoch_plan(10, 50, 0.0, 1)[0] == sft_data.epoch_plan(10, 0, 0.0, 1)[0]


def test_epoch_plan_mixes_replay_at_the_asked_share():
    plan = sft_data.epoch_plan(800, 1000, 0.2, 2, seed=0)
    for epoch in plan:
        sources = [s for s, _ in epoch]
        assert sources.count("task") == 800 and sources.count("replay") == 200
        assert sorted(i for s, i in epoch if s == "task") == list(range(800))
    replay = [i for epoch in plan for s, i in epoch if s == "replay"]
    assert len(set(replay)) == 400  # no repeat before the pool runs out
    small = sft_data.epoch_plan(8, 3, 0.5, 1)[0]
    counts = [sum(1 for s, i in small if s == "replay" and i == k) for k in range(3)]
    assert sum(counts) == 8 and max(counts) - min(counts) <= 1
    with pytest.raises(ValueError):
        sft_data.epoch_plan(10, 10, 1.0, 1)


# ------------------------------------------------------------------ loss masks, real tokenizer


@pytest.fixture(scope="module")
def qwen_tok():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("jinja2")
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL)
    except Exception as e:  # no network, or no Hub access
        pytest.skip(f"{MODEL} tokenizer unavailable: {type(e).__name__}: {e}")


def _task_messages(content: str) -> list[dict]:
    row = example_rows((0,))[0]
    return row["prompt"] + [{"role": "assistant", "content": content}]


def test_empty_think_mask_covers_exactly_the_think_block(qwen_tok):
    tok = qwen_tok
    content = reply(LONG, REF)
    messages = _task_messages(content)
    masked = sft_data.encode(tok, messages, empty_think="mask")
    trained = sft_data.encode(tok, messages, empty_think="train")
    ids = masked["input_ids"]
    assert trained["input_ids"] == ids

    # The inference-time tokenization: the generation prompt on its own, then the reply.
    head = tok.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    assert head.endswith("<|im_start|>assistant\n" + sft_data.EMPTY_THINK)
    head_ids = tok(head, add_special_tokens=False)["input_ids"]
    target_ids = tok(content + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
    assert ids == head_ids + target_ids  # the old sft_lora.py tokenization, unchanged

    start, end = masked["think"]
    assert end == len(head_ids) and end - start == 4
    open_, close = tok.convert_tokens_to_ids(["<think>", "</think>"])
    (nn,) = tok("\n\n", add_special_tokens=False)["input_ids"]
    assert ids[start:end] == [open_, nn, close, nn]
    assert tok.decode(ids[start:end]) == sft_data.EMPTY_THINK

    differ = [
        i
        for i, (a, b) in enumerate(zip(masked["labels"], trained["labels"], strict=True))
        if a != b
    ]
    assert differ == list(range(start, end))  # the mask covers exactly the block
    assert trained["labels"][start:end] == ids[start:end]
    assert masked["labels"][:end] == [-100] * end
    assert trained["labels"][:start] == [-100] * start
    assert masked["labels"][end:] == ids[end:]
    assert tok.decode(masked["labels"][end:]) == content + "<|im_end|>\n"


def test_thinking_on_rows_train_their_reasoning(qwen_tok):
    tok = qwen_tok
    messages = [
        {"role": "user", "content": "What is 6 times 7?"},
        {"role": "assistant", "content": "It is 42.", "reasoning_content": "6 * 7 = 42."},
    ]
    for mode in sft_data.EMPTY_THINK_MODES:
        e = sft_data.encode(tok, messages, enable_thinking=True, empty_think=mode)
        start, end = e["think"]
        assert tok.decode(e["input_ids"][start:end]) == "<think>\n"  # in the prompt: masked
        assert e["labels"][:end] == [-100] * end
        assert tok.decode(e["labels"][end:]) == "6 * 7 = 42.\n</think>\n\nIt is 42.<|im_end|>\n"
    with pytest.raises(ValueError, match="thinking"):
        sft_data.encode(tok, messages, enable_thinking=False)


def test_rationalized_and_replay_rows_encode(qwen_tok, tmp_path):
    rows = [
        {"messages": _task_messages(reply(SHORT, REF)), "meta": {}},
        {
            "messages": [
                {"role": "user", "content": "Name a prime."},
                {
                    "role": "assistant",
                    "content": "7",
                    "reasoning_content": "Small primes: 2, 3, 5, 7.",
                },
            ],
            "enable_thinking": True,
        },
    ]
    for row in sft_data.load_examples(str(write_rows(tmp_path / "rows.jsonl", rows))):
        e = sft_data.encode(qwen_tok, row["messages"], enable_thinking=row["enable_thinking"])
        assert len(e["input_ids"]) == len(e["labels"])
        assert any(lab != -100 for lab in e["labels"])
