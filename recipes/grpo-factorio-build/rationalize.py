"""Rewrite the SFT targets in the model's own voice: rationalisation, as in STaR.

    uv run --no-sync python rationalize.py --data sft.jsonl --out-dir rat_r1 \
        --base-url http://localhost:8000/v1

build_sft_data.py pairs each evolved program with a factorio-build prompt.
SFT on those bare programs taught Qwen3.5-9B the task and cost it its prose
reasoning. Here the model writes its own targets instead. It gets the env's
prompt plus one more user message, `HINT_TEMPLATE`, holding the example's
program as a reference, and is asked to explain the approach step by step
before giving the complete program. The SFT row is the env's prompt WITHOUT
the hint, followed by the model's reply.

Scoring is the env's own code. The program is pulled out with
`evolve.llm.extract_code` and scored by `factorio_build.core.score_completion`
on the example's own scenes, which are rebuilt from the subset id in its
prompt. `core.rows` must reproduce the example's system and user messages
byte for byte, or the run stops before sending a request.

Keep rule, per sample:
- the request succeeded, the reply finished (`finish_reason` "stop"), and
  it holds a sandbox-valid program;
- the program's success on the example's scenes is at least the reference
  program's, and above zero unless `--allow-zero-success` is given;
- the reply does not mention the hint (`LEAK_RE`), holds at most
  `--max-programs` build blocks, and has at least `--min-reasoning-words`
  words of prose before the program.

Of the samples that pass, the best `--keep-per-example` are kept: higher
success first, then the shortest reasoning.

Files in --out-dir:
- `run.json`: the generation settings, including the hint text. A resume
  with other settings is refused.
- `samples.jsonl`: one record per example, with every sample and its scores.
  It is append-only, and the resume key.
- `sft.jsonl`: `{"messages": [system, user, assistant], "meta": ...}`, one row
  per kept sample, for sft_lora.py.
- `stats.json`: kept rates, reject reasons, reply lengths, and how many kept
  programs are the same as the hint or differ from it.

`sft.jsonl` and `stats.json` are rebuilt from `samples.jsonl` at the end of
every run. `--rebuild` rebuilds them under new keep settings without
contacting a server.

Iterating (Iterative-SFT, as in Retaining by Doing): serve the current
checkpoint instead of base and rerun into a new --out-dir with `--round 2`.
`--hint-mode fallback` first samples without the hint and adds hinted samples
only for examples where no unhinted sample passes, which is STaR's order.
The README's runbook has the commands.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "integrations" / "verifiers" / "factorio_build"))
import gen_common as gc  # noqa: E402
from factorio_build import core  # noqa: E402

from evolve import evaluate, sandbox  # noqa: E402
from evolve.llm import _BUILD, _CODE_TAGS, _FENCE, extract_code  # noqa: E402

HINT_VERSION = "hint-v1"
#: The one extra user message. `{program}` is the reference program as the env's
#: parser extracts it, which ends in a newline.
HINT_TEMPLATE = (
    "A reference program that solves this task:\n\n"
    "```python\n{program}```\n\n"
    "Explain the approach step by step in your own words, as your own solution, without "
    "mentioning the reference. Then give the complete program in one ```python fenced "
    "block, with nothing after it. You may adapt or improve the reference program."
)

#: A reply that talks about the hint is not a reply to the hint-free prompt.
LEAK_RE = re.compile(
    r"\b(?:reference|provided|given|supplied|example|original|sample)\s+"
    r"(?:program|solution|code|implementation|script|version)\b"
    r"|\b(?:the|this)\s+(?:reference|hint)\b"
    r"|\byour\s+(?:program|code|solution|implementation)\b",
    re.IGNORECASE,
)

SUBSET_RE = re.compile(r"scene subset (\w+)/(train|val|holdout)/(\d+)-(\d+) \(")
HINT_MODES = {"always": (True,), "never": (False,), "fallback": (False, True)}


class PromptMismatch(ValueError):
    """The example's prompt is not what the env serves today."""


@dataclass
class Example:
    id: str
    index: int
    task: str
    split: str
    subset_id: str
    scenes: list
    messages: list  # the env's prompt: system (if any), then user
    reference_text: str
    reference_code: str


@dataclass(frozen=True)
class KeepRules:
    keep_per_example: int = 1
    min_reasoning_words: int = 30
    max_programs: int = 1
    allow_zero_success: bool = False


# ------------------------------------------------------------------ examples


def resolve(row: dict, index: int) -> Example:
    """An SFT row (build_sft_data.py's prompt/completion, or messages) as the env serves it."""
    if "messages" in row:
        prompt, completion = row["messages"][:-1], row["messages"][-1]["content"]
    else:
        prompt, completion = row["prompt"], row["completion"][-1]["content"]
    system = next((m["content"] for m in prompt if m["role"] == "system"), None)
    users = [m["content"] for m in prompt if m["role"] == "user"]
    if len(users) != 1:
        raise PromptMismatch(f"row {index}: expected one user message, found {len(users)}")
    user = users[0]
    m = SUBSET_RE.search(user)
    if m is None:
        raise PromptMismatch(f"row {index}: no scene subset id in the user message")
    task, split, start, end = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
    if split != "train":
        raise ValueError(f"row {index}: split {split!r}; SFT data must come from train scenes")
    for notes in (True, False):
        env = core.rows(task, split, end - start + 1, 1, start, notes)[0]
        if env["prompt"] == user and (env["system_prompt"] or None) == system:
            break
    else:
        raise PromptMismatch(
            f"row {index} ({task}/{split}/{start}-{end}): its prompt differs from the env's; "
            "rebuild the data with build_sft_data.py"
        )
    code = extract_code(completion)
    if code is None:
        raise ValueError(f"row {index}: the target holds no program")
    messages = ([{"role": "system", "content": env["system_prompt"]}] if system else []) + [
        {"role": "user", "content": env["prompt"]}
    ]
    digest = hashlib.sha256((user + "\0" + code).encode()).hexdigest()[:10]
    return Example(
        id=f"{index:05d}-{digest}",
        index=index,
        task=env["task"],
        split=split,
        subset_id=env["subset_id"],
        scenes=[list(s) for s in env["scenes"]],
        messages=messages,
        reference_text=completion,
        reference_code=code,
    )


def hint_message(reference_code: str) -> dict:
    return {"role": "user", "content": HINT_TEMPLATE.format(program=reference_code)}


def request_messages(ex: Example, hinted: bool) -> list[dict]:
    return ex.messages + ([hint_message(ex.reference_code)] if hinted else [])


# ------------------------------------------------------------------ replies


def _program_blocks(text: str) -> list[re.Match]:
    """Fenced blocks the env's parser would accept as a program (`evolve.llm`'s rule)."""
    return [
        m
        for m in _FENCE.finditer(text)
        if m.group(1).lower() in _CODE_TAGS and _BUILD.search(m.group(2))
    ]


def _same_as(code: str | None, reference: str) -> str | None:
    """ "exact", "normalized" (same AST up to names, comments, layout) or "different"."""
    if code is None:
        return None
    if code.strip() == reference.strip():
        return "exact"
    try:
        if sandbox.normalized_hash(code) == sandbox.normalized_hash(reference):
            return "normalized"
    except (sandbox.SandboxError, SyntaxError, RecursionError, ValueError):
        pass
    return "different"


def facts(text: str, reference_code: str) -> dict:
    """What the keep rule reads off a reply, besides its score."""
    text = (text or "").replace("\r\n", "\n")
    blocks = _program_blocks(text)
    if blocks:
        reasoning, trailing = text[: blocks[-1].start()], text[blocks[-1].end() :]
    else:
        reasoning, trailing = text, ""
    reasoning = reasoning.strip()
    return {
        "n_programs": len(blocks),
        "reasoning_chars": len(reasoning),
        "reasoning_words": len(reasoning.split()),
        "trailing_chars": len(trailing.strip()),
        "reply_chars": len(text),
        "hint_leak": bool(LEAK_RE.search(text)),
        "same_as_hint": _same_as(extract_code(text), reference_code),
    }


def reject_reasons(sample: dict, reference_success: float, rules: KeepRules) -> list[str]:
    """Why a sample cannot be kept; an empty list means it can."""
    if sample.get("error"):
        return ["request_error"]
    reasons = []
    finish = sample.get("finish_reason")
    if finish not in ("stop", None):
        reasons.append("truncated" if finish == "length" else "incomplete")
    m, f = sample["metrics"], sample["facts"]
    if not m["has_program"]:
        reasons.append("no_program")
    elif not m["sandbox_valid"]:
        reasons.append("sandbox")
    else:
        if m["success"] + 1e-9 < reference_success:
            reasons.append("below_reference")
        if m["success"] <= 0 and not rules.allow_zero_success:
            reasons.append("zero_success")
    if f["hint_leak"]:
        reasons.append("hint_leak")
    if f["n_programs"] > rules.max_programs:
        reasons.append("several_programs")
    if f["reasoning_words"] < rules.min_reasoning_words:
        reasons.append("short_reasoning")
    return reasons


def select(record: dict, rules: KeepRules) -> list[dict]:
    """The kept samples: passing ones, higher success first, then shorter reasoning."""
    ref = record["reference"]["metrics"]["success"]
    ok = [s for s in record["samples"] if not reject_reasons(s, ref, rules)]
    ok.sort(key=lambda s: (-s["metrics"]["success"], s["facts"]["reasoning_chars"], s["k"]))
    return ok[: rules.keep_per_example]


# ------------------------------------------------------------------ generation


class Runner:
    def __init__(self, args, client: gc.ChatClient, model: str, rules: KeepRules):
        self.args = args
        self.client = client
        self.model = model
        self.rules = rules
        workers = args.score_workers
        pools = 1 if workers <= 0 else args.score_pools
        # In-process scoring (workers 0) shares one simulator: one program at a time.
        core.MAX_POOLS = pools
        self.score_pool = ThreadPoolExecutor(max_workers=pools, thread_name_prefix="score")
        self.count_tokens = gc.TokenCounter(args.tokenizer if args.tokenizer else model)

    def close(self) -> None:
        self.score_pool.shutdown(wait=True)

    def _score_sync(self, text: str | None, ex: Example) -> dict:
        return core.score_completion(
            text,
            ex.task,
            [tuple(s) for s in ex.scenes],
            workers=self.args.score_workers,
            timeout_s=self.args.score_timeout,
        )

    async def score(self, text: str | None, ex: Example) -> dict:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.score_pool, self._score_sync, text, ex)

    async def _phase(self, ex: Example, hinted: bool, k0: int) -> tuple[dict, list[dict]]:
        a = self.args
        seed = gc.stable_seed(a.seed, ex.id, a.round, int(hinted))
        body = gc.sampling_body(
            self.model,
            request_messages(ex, hinted),
            n=a.n,
            temperature=a.temperature,
            max_tokens=a.max_tokens,
            enable_thinking=False,
            seed=seed,
            top_p=a.top_p,
            top_k=a.top_k,
        )
        reply = await self.client.chat(body)
        request = {
            "hinted": hinted,
            "seed": seed,
            "attempts": reply.attempts,
            "error": reply.error,
            "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens,
        }
        if reply.error:
            return request, [
                {"k": k0 + i, "hinted": hinted, "error": reply.error} for i in range(a.n)
            ]
        metrics = await asyncio.gather(*(self.score(c.text, ex) for c in reply.choices))
        samples = []
        for i, (c, m) in enumerate(zip(reply.choices, metrics, strict=True)):
            samples.append(
                {
                    "k": k0 + i,
                    "hinted": hinted,
                    "error": None,
                    "text": c.text,
                    "reasoning": c.reasoning,
                    "finish_reason": c.finish_reason,
                    "tokens": self.count_tokens(c.text),
                    "metrics": m,
                    "facts": facts(c.text, ex.reference_code),
                }
            )
        return request, samples

    async def process(self, ex: Example) -> dict:
        ref_task = asyncio.ensure_future(self.score(ex.reference_text, ex))
        requests, samples = [], []
        for hinted in HINT_MODES[self.args.hint_mode]:
            request, new = await self._phase(ex, hinted, len(samples))
            requests.append(request)
            samples.extend(new)
            ref = (await ref_task)["success"]
            if any(not reject_reasons(s, ref, self.rules) for s in new):
                break
        ref_metrics = await ref_task
        return {
            "id": ex.id,
            "index": ex.index,
            "task": ex.task,
            "split": ex.split,
            "subset_id": ex.subset_id,
            "scenes": ex.scenes,
            "round": self.args.round,
            "model": self.model,
            "messages": ex.messages,
            "hint_message": hint_message(ex.reference_code)["content"],
            "reference": {
                "program": ex.reference_code,
                "program_hash": ref_metrics["program_hash"],
                "metrics": ref_metrics,
            },
            "requests": requests,
            "samples": samples,
        }


async def generate(args, examples: list[Example], samples_path: str, model: str, rules) -> None:
    client = gc.ChatClient(
        args.base_url,
        api_key=args.api_key,
        concurrency=args.concurrency,
        timeout_s=args.timeout,
        max_attempts=args.max_attempts,
        backoff_s=args.backoff,
    )
    runner = Runner(args, client, model, rules)
    if runner.count_tokens.error:
        print(f"token counts off ({runner.count_tokens.error})", flush=True)
    gc.repair_tail(samples_path)
    t0, done, kept, lock = time.time(), 0, 0, asyncio.Lock()
    try:
        with open(samples_path, "a", encoding="utf-8") as out:

            async def one(ex: Example) -> None:
                nonlocal done, kept
                record = await runner.process(ex)
                async with lock:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    done += 1
                    kept += bool(select(record, rules))
                    if done % args.log_every == 0 or done == len(examples):
                        dt = time.time() - t0
                        eta = dt / done * (len(examples) - done)
                        print(
                            f"{done}/{len(examples)} examples, {kept} with a kept sample, "
                            f"{dt / 60:.1f} min, eta {eta / 60:.1f} min",
                            flush=True,
                        )

            results = await asyncio.gather(*(one(ex) for ex in examples), return_exceptions=True)
    finally:
        runner.close()
        client.close()
    failed = [r for r in results if isinstance(r, BaseException)]
    if failed:
        # Every finished example is on disk; a rerun resumes with the rest.
        raise RuntimeError(f"{len(failed)} examples failed; first: {failed[0]!r}") from failed[0]


# ------------------------------------------------------------------ outputs


def _share(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


def build_outputs(out_dir: str, rules: KeepRules, config: dict | None = None) -> dict:
    """sft.jsonl and stats.json from samples.jsonl, under `rules`."""
    records = sorted(
        gc.read_jsonl(os.path.join(out_dir, "samples.jsonl")), key=lambda r: r["index"]
    )
    rows, kept, all_samples = [], [], []
    reasons = collections.Counter()
    kept_examples = 0
    for rec in records:
        ref = rec["reference"]["metrics"]["success"]
        for s in rec["samples"]:
            all_samples.append(s)
            for r in reject_reasons(s, ref, rules) or ["passed"]:
                reasons[r] += 1
        chosen = select(rec, rules)
        kept_examples += bool(chosen)
        for s in chosen:
            kept.append((rec, s))
            rows.append(
                {
                    "messages": rec["messages"] + [{"role": "assistant", "content": s["text"]}],
                    "meta": {
                        "id": rec["id"],
                        "subset_id": rec["subset_id"],
                        "k": s["k"],
                        "hinted": s["hinted"],
                        "success": s["metrics"]["success"],
                        "reference_success": ref,
                        "same_as_hint": s["facts"]["same_as_hint"],
                        "round": rec["round"],
                        "model": rec["model"],
                        "source": "rationalize",
                    },
                }
            )
    gc.write_jsonl(os.path.join(out_dir, "sft.jsonl"), rows)

    answered = [s for s in all_samples if not s.get("error")]
    same = collections.Counter(s["facts"]["same_as_hint"] for _, s in kept)
    hinted_kept = [s for _, s in kept if s["hinted"]]
    same_hinted = collections.Counter(s["facts"]["same_as_hint"] for s in hinted_kept)

    def lengths(samples, key):
        if key == "tokens":
            return gc.summary(s.get("tokens") for s in samples)
        return gc.summary(s["facts"][key] for s in samples)

    kept_samples = [s for _, s in kept]
    stats = {
        "examples": len(records),
        "kept_examples": kept_examples,
        "kept_example_rate": _share(kept_examples, len(records)),
        "samples": len(all_samples),
        "answered_samples": len(answered),
        "kept_samples": len(kept),
        "sample_pass_rate": _share(reasons["passed"], len(all_samples)),
        "reject_reasons": dict(sorted(reasons.items())),
        # Non-zero under thinking off means the server's reasoning parser moved
        # reply text out of `content`; serve without one for this script.
        "replies_with_reasoning_content": sum(bool(s.get("reasoning")) for s in answered),
        "hinted": {
            "samples": sum(bool(s.get("hinted")) for s in all_samples),
            "kept": len(hinted_kept),
        },
        "unhinted": {
            "samples": sum(not s.get("hinted") for s in all_samples),
            "kept": len(kept) - len(hinted_kept),
        },
        "reference_success": gc.summary(r["reference"]["metrics"]["success"] for r in records),
        "kept_success": gc.summary(s["metrics"]["success"] for s in kept_samples),
        "kept_above_reference": sum(
            s["metrics"]["success"] > r["reference"]["metrics"]["success"] + 1e-9 for r, s in kept
        ),
        "reply_tokens": {
            "all": lengths(answered, "tokens"),
            "kept": lengths(kept_samples, "tokens"),
        },
        "reply_chars": {
            "all": lengths(answered, "reply_chars"),
            "kept": lengths(kept_samples, "reply_chars"),
        },
        "reasoning_words": {
            "all": lengths(answered, "reasoning_words"),
            "kept": lengths(kept_samples, "reasoning_words"),
        },
        "kept_vs_hint": {
            "exact": same["exact"],
            "normalized": same["normalized"],
            "different": same["different"],
            "share_identical": _share(same["exact"] + same["normalized"], len(kept)),
            "share_different": _share(same["different"], len(kept)),
            "hinted_only": {
                "exact": same_hinted["exact"],
                "normalized": same_hinted["normalized"],
                "different": same_hinted["different"],
            },
        },
        "rules": asdict(rules),
        "hint": {"version": HINT_VERSION, "template": HINT_TEMPLATE},
        "run": config,
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


# ------------------------------------------------------------------ main


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="build_sft_data.py output (jsonl)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument(
        "--model", default=None, help="request model name; default: what the server lists"
    )
    ap.add_argument(
        "--round", type=int, default=1, help="1 = base; 2+ = regenerated from a checkpoint"
    )
    ap.add_argument("--hint-mode", choices=sorted(HINT_MODES), default="always")
    ap.add_argument("--n", type=int, default=4, help="samples per request")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=12288)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=64, help="requests in flight")
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds per request")
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--backoff", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=None, help="first N examples only (smoke test)")
    ap.add_argument("--score-workers", type=int, default=4, help="simulator processes per program")
    ap.add_argument(
        "--score-pools",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 4),
        help="programs scored at once",
    )
    ap.add_argument("--score-timeout", type=float, default=30.0)
    ap.add_argument("--keep-per-example", type=int, default=1)
    ap.add_argument("--min-reasoning-words", type=int, default=30)
    ap.add_argument("--max-programs", type=int, default=1)
    ap.add_argument("--allow-zero-success", action="store_true")
    ap.add_argument(
        "--tokenizer", default=None, help="for reply token counts; default: the model; 'none': off"
    )
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--rebuild", action="store_true", help="only rebuild sft.jsonl and stats.json")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    rules = KeepRules(
        keep_per_example=args.keep_per_example,
        min_reasoning_words=args.min_reasoning_words,
        max_programs=args.max_programs,
        allow_zero_success=args.allow_zero_success,
    )
    run_path = os.path.join(args.out_dir, "run.json")
    samples_path = os.path.join(args.out_dir, "samples.jsonl")
    if args.rebuild:
        config = json.load(open(run_path, encoding="utf-8")) if os.path.exists(run_path) else None
        return report(build_outputs(args.out_dir, rules, config))

    rows = gc.read_jsonl(args.data)
    if args.limit is not None:
        rows = rows[: args.limit]
    examples = [resolve(r, i) for i, r in enumerate(rows)]
    if args.model is None:
        probe = gc.ChatClient(args.base_url, api_key=args.api_key, concurrency=1)
        args.model = probe.served_model()
        probe.close()
    config = {
        "data": os.path.abspath(args.data),
        "data_sha256": gc.file_sha256(args.data),
        "model": args.model,
        "round": args.round,
        "hint_mode": args.hint_mode,
        "hint_version": HINT_VERSION,
        "hint_template": HINT_TEMPLATE,
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "enable_thinking": False,
        "score_timeout_s": args.score_timeout,
        "evaluator_version": evaluate.EVALUATOR_VERSION,
    }
    gc.check_run_config(run_path, config)
    done = {r["id"] for r in gc.read_jsonl(samples_path)}
    todo = [ex for ex in examples if ex.id not in done]
    print(
        f"{len(examples)} examples, {len(examples) - len(todo)} already done, model {args.model}, "
        f"hint mode {args.hint_mode}, round {args.round}",
        flush=True,
    )
    if todo:
        asyncio.run(generate(args, todo, samples_path, args.model, rules))
    return report(build_outputs(args.out_dir, rules, config))


def report(stats: dict) -> dict:
    print(
        f"kept {stats['kept_examples']}/{stats['examples']} examples "
        f"({stats['kept_samples']} samples); sample pass rate {stats['sample_pass_rate']}; "
        f"kept vs hint {stats['kept_vs_hint']['share_identical']} identical, "
        f"{stats['kept_vs_hint']['share_different']} different",
        flush=True,
    )
    print("reject reasons:", stats["reject_reasons"], flush=True)
    return stats


if __name__ == "__main__":
    main()
