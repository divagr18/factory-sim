"""Replay data: the BASE model's own answers to general prompts, for
`sft_lora.py --replay`.

    uv run --no-sync python make_replay.py --out-dir replay --n-prompts 1000 \
        --base-url http://localhost:8000/v1

Rehearsal on the model's own outputs limits forgetting during SFT. The rows
here are Qwen3.5's answers, sampled from the served model, to prompts from
small public sets under permissive licences (`SOURCES`):
- OpenAssistant/oasst2 (Apache-2.0): human-written first turns, English,
  not deleted, not flagged as spam, PII or inappropriate.
- openai/gsm8k (MIT): the train split's maths word problems. These draw the
  step-by-step prose the task SFT data lacks.

Prompts that look like planning benchmarks (Blocksworld, PDDL, stacking
blocks, PlanBench's phrasing) are dropped (`PLANNING_RE`), so replay cannot
leak into the PlanBench measurement. `--prompts file.jsonl` replaces the
public sets with local `{"prompt": ...}` rows, which pass the same filters.

Answers are sampled with thinking off at T 0.7. A `--think-frac` share of the
prompts is answered with thinking on at T 0.6, and those rows keep their
reasoning. A reply is kept when it finished (`finish_reason` "stop"), its
answer is not empty and, with thinking on, its think block closed.

Always serve BASE for this script, in every Iterative-SFT round: replay anchors
the tuned model to base.

Files in --out-dir:
- `prompts.jsonl`: the chosen prompts, frozen on the first run.
- `selection.json`: per-source counts and licences.
- `run.json`: sampling settings; a resume with other settings is refused.
- `samples.jsonl`: raw replies, append-only, the resume key.
- `replay.jsonl`: `{"messages", "enable_thinking", "meta"}` rows for sft_lora.py.
- `stats.json`: counts, reject reasons, lengths, licences.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_common as gc  # noqa: E402

#: Public prompt sets. Licences read from each dataset card on 2026-09-25.
SOURCES = {
    "oasst2": {
        "hf_id": "OpenAssistant/oasst2",
        "config": None,
        "split": "train",
        "licence": "Apache-2.0",
        "url": "https://huggingface.co/datasets/OpenAssistant/oasst2",
        "about": "human-written first turns of OpenAssistant conversations, English only",
    },
    "gsm8k": {
        "hf_id": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "licence": "MIT",
        "url": "https://huggingface.co/datasets/openai/gsm8k",
        "about": "grade-school maths word problems, train split",
    },
}
DEFAULT_MIX = "oasst2=0.75,gsm8k=0.25"

#: Planning-benchmark look-alikes (PlanBench Blocksworld and its kin).
_BLOCK_VERBS = r"(?:(?:un)?stack(?:s|ed|ing)?|on top of|pick(?:s|ed|ing)? up|put(?:s|ting)? down)"
PLANNING_RE = re.compile(
    r"blocks?\s*-?\s*world"
    rf"|\bblocks?\b.{{0,120}}\b{_BLOCK_VERBS}\b"
    rf"|\b{_BLOCK_VERBS}\b.{{0,120}}\bblocks?\b"
    r"|\bunstack\b|\bpddl\b|\bplanbench\b|\bplanning (?:problem|domain|task)s?\b"
    r"|\b(?:initial|goal) state\b|\bmy plan is as follows\b|\bas initial conditions i have\b"
    r"|\bmy goal is to have\b|\bgripper\b|\btowers? of hanoi\b|\blogistics domain\b"
    r"|\bsokoban\b",
    re.IGNORECASE | re.DOTALL,
)
#: oasst2 labels (averaged annotator votes) that drop a prompt at >= 0.5.
OASST_BAD_LABELS = (
    "spam",
    "pii",
    "not_appropriate",
    "hate_speech",
    "sexual_content",
    "lang_mismatch",
)
MIN_CHARS, MAX_CHARS = 16, 4000


# ------------------------------------------------------------------ prompts


def _oasst_ok(row: dict) -> bool:
    if row.get("role") != "prompter" or row.get("parent_id") not in (None, "", "None"):
        return False
    if row.get("lang") != "en" or row.get("deleted") or row.get("synthetic"):
        return False
    if row.get("review_result") is False:
        return False
    labels = row.get("labels") or {}
    for name, value in zip(labels.get("name") or [], labels.get("value") or [], strict=False):
        if name in OASST_BAD_LABELS and value is not None and value >= 0.5:
            return False
    tox = (row.get("detoxify") or {}).get("toxicity")
    return tox is None or tox < 0.2


def load_source(name: str) -> list[dict]:
    """Candidate prompts of one public source, as {"id", "prompt", "source", "licence"}."""
    from datasets import load_dataset

    src = SOURCES[name]
    ds = load_dataset(src["hf_id"], src["config"], split=src["split"])
    out = []
    if name == "oasst2":
        for row in ds:
            if _oasst_ok(row):
                out.append({"id": f"oasst2:{row['message_id']}", "prompt": row["text"]})
    elif name == "gsm8k":
        for i, row in enumerate(ds):
            out.append({"id": f"gsm8k:{i}", "prompt": row["question"]})
    else:
        raise ValueError(f"unknown source {name!r}")
    for p in out:
        p["source"], p["licence"] = name, src["licence"]
    return out


def screen(prompts: list[dict]) -> tuple[list[dict], collections.Counter]:
    """Drop planning look-alikes, too short or long prompts, and duplicates."""
    kept, dropped, seen = [], collections.Counter(), set()
    for p in prompts:
        text = (p.get("prompt") or "").strip()
        key = " ".join(text.lower().split())
        if PLANNING_RE.search(text):
            dropped["planning_like"] += 1
        elif not MIN_CHARS <= len(text) <= MAX_CHARS:
            dropped["length"] += 1
        elif key in seen:
            dropped["duplicate"] += 1
        else:
            seen.add(key)
            kept.append({**p, "prompt": text})
    return kept, dropped


def parse_mix(text: str) -> list[tuple[str, float]]:
    mix = []
    for part in text.split(","):
        name, _, w = part.partition("=")
        name = name.strip()
        if name not in SOURCES:
            raise SystemExit(f"--mix: unknown source {name!r}; known: {', '.join(SOURCES)}")
        mix.append((name, float(w) if w else 1.0))
    total = sum(w for _, w in mix)
    return [(n, w / total) for n, w in mix]


def selection_key(args) -> dict:
    """What decides the prompt set; a resume must not change it."""
    return {
        "mix": None if args.prompts else args.mix,
        "prompts_file": os.path.abspath(args.prompts) if args.prompts else None,
        "n_prompts": args.n_prompts,
        "think_frac": args.think_frac,
        "seed": args.seed,
    }


def choose_prompts(args) -> tuple[list[dict], dict]:
    """The prompt set: per-source quotas, shuffled, the first --think-frac share thinking."""
    rng = random.Random(args.seed)
    selection = {"sources": {}, **selection_key(args)}
    chosen: list[dict] = []
    if args.prompts:
        rows = gc.read_jsonl(args.prompts)
        for i, r in enumerate(rows):
            r.setdefault("id", f"local:{i}")
            r.setdefault("source", "local")
            r.setdefault("licence", "unknown")
        groups = [("local", 1.0, rows)]
    else:
        groups = [(name, w, load_source(name)) for name, w in parse_mix(args.mix)]
    left = args.n_prompts
    for k, (name, w, rows) in enumerate(groups):
        kept, dropped = screen(rows)
        rng.shuffle(kept)
        quota = left if k == len(groups) - 1 else min(left, round(args.n_prompts * w))
        take = kept[:quota]
        left -= len(take)
        chosen.extend(take)
        info = SOURCES.get(name, {})
        selection["sources"][name] = {
            "licence": info.get("licence", sorted({r["licence"] for r in rows})),
            "hf_id": info.get("hf_id"),
            "url": info.get("url"),
            "about": info.get("about"),
            "candidates": len(rows),
            "dropped": dict(dropped),
            "eligible": len(kept),
            "taken": len(take),
        }
    rng.shuffle(chosen)
    n_think = round(len(chosen) * args.think_frac)
    for i, p in enumerate(chosen):
        p["think"] = i < n_think
    return chosen, selection


# ------------------------------------------------------------------ replies


def split_reply(
    text: str, reasoning: str | None, think: bool
) -> tuple[str | None, str | None, str | None]:
    """(answer, reasoning, reject reason) of one reply."""
    text = text or ""
    if not think:
        if "</think>" in text:
            return None, None, "stray_think"
        return text, None, None
    if reasoning:
        return text, reasoning, None
    if "</think>" not in text:
        return None, None, "unclosed_think"
    before, after = text.split("</think>", 1)
    return after.lstrip("\n"), before.replace("<think>", "").strip("\n"), None


def judge(sample: dict) -> str | None:
    """Why a sample is not kept, or None."""
    if sample.get("error"):
        return "request_error"
    if sample.get("finish_reason") not in ("stop", None):
        return "truncated" if sample.get("finish_reason") == "length" else "incomplete"
    if sample.get("reject"):
        return sample["reject"]
    if not (sample.get("answer") or "").strip():
        return "empty_answer"
    if sample["think"] and not (sample.get("reasoning") or "").strip():
        return "empty_reasoning"
    return None


async def generate(args, prompts: list[dict], samples_path: str, model: str) -> None:
    client = gc.ChatClient(
        args.base_url,
        api_key=args.api_key,
        concurrency=args.concurrency,
        timeout_s=args.timeout,
        max_attempts=args.max_attempts,
        backoff_s=args.backoff,
    )
    count = gc.TokenCounter(args.tokenizer if args.tokenizer else model)
    gc.repair_tail(samples_path)
    t0, done, lock = time.time(), 0, asyncio.Lock()
    try:
        with open(samples_path, "a", encoding="utf-8") as out:

            async def one(p: dict) -> None:
                nonlocal done
                think = p["think"]
                body = gc.sampling_body(
                    model,
                    [{"role": "user", "content": p["prompt"]}],
                    n=1,
                    temperature=args.think_temperature if think else args.temperature,
                    max_tokens=args.think_max_tokens if think else args.max_tokens,
                    enable_thinking=think,
                    seed=gc.stable_seed(args.seed, p["id"], int(think)),
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
                reply = await client.chat(body)
                rec = {"id": p["id"], "think": think, "error": reply.error}
                if not reply.error:
                    c = reply.choices[0]
                    answer, reasoning, reject = split_reply(c.text, c.reasoning, think)
                    rec.update(
                        text=c.text,
                        answer=answer,
                        reasoning=reasoning,
                        reject=reject,
                        finish_reason=c.finish_reason,
                        completion_tokens=reply.completion_tokens,
                        answer_tokens=count(answer),
                    )
                async with lock:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    done += 1
                    if done % args.log_every == 0 or done == len(prompts):
                        dt = time.time() - t0
                        print(
                            f"{done}/{len(prompts)} prompts, {dt / 60:.1f} min, "
                            f"eta {dt / done * (len(prompts) - done) / 60:.1f} min",
                            flush=True,
                        )

            results = await asyncio.gather(*(one(p) for p in prompts), return_exceptions=True)
    finally:
        client.close()
    failed = [r for r in results if isinstance(r, BaseException)]
    if failed:
        raise RuntimeError(f"{len(failed)} prompts failed; first: {failed[0]!r}") from failed[0]


# ------------------------------------------------------------------ outputs


def build_outputs(out_dir: str, prompts: list[dict], selection: dict, config: dict) -> dict:
    by_id = {p["id"]: p for p in prompts}
    samples = [s for s in gc.read_jsonl(os.path.join(out_dir, "samples.jsonl")) if s["id"] in by_id]
    order = {p["id"]: i for i, p in enumerate(prompts)}
    samples.sort(key=lambda s: order[s["id"]])
    rows, reasons = [], collections.Counter()
    kept = {True: [], False: []}
    for s in samples:
        why = judge(s)
        reasons[why or "kept"] += 1
        if why:
            continue
        p = by_id[s["id"]]
        assistant = {"role": "assistant", "content": s["answer"]}
        if s["think"]:
            assistant["reasoning_content"] = s["reasoning"]
        rows.append(
            {
                "messages": [{"role": "user", "content": p["prompt"]}, assistant],
                "enable_thinking": s["think"],
                "meta": {
                    "id": p["id"],
                    "source": p["source"],
                    "licence": p["licence"],
                    "model": config["model"],
                    "completion_tokens": s.get("completion_tokens"),
                },
            }
        )
        kept[s["think"]].append(s)
    gc.write_jsonl(os.path.join(out_dir, "replay.jsonl"), rows)

    def lengths(group):
        return {
            "answer_chars": gc.summary(len(s["answer"]) for s in group),
            "answer_tokens": gc.summary(s.get("answer_tokens") for s in group),
            "completion_tokens": gc.summary(s.get("completion_tokens") for s in group),
            "reasoning_chars": gc.summary(len(s["reasoning"] or "") for s in group if s["think"]),
        }

    stats = {
        "prompts": len(prompts),
        "answered": len(samples),
        "kept": len(rows),
        "kept_rate": round(len(rows) / len(samples), 4) if samples else None,
        "kept_by_mode": {"thinking_off": len(kept[False]), "thinking_on": len(kept[True])},
        "kept_by_source": dict(collections.Counter(r["meta"]["source"] for r in rows)),
        "reasons": dict(sorted(reasons.items())),
        "lengths": {"thinking_off": lengths(kept[False]), "thinking_on": lengths(kept[True])},
        "selection": selection,
        "run": config,
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


# ------------------------------------------------------------------ main


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument(
        "--model", default=None, help="request model name; default: what the server lists"
    )
    ap.add_argument("--mix", default=DEFAULT_MIX, help="source=weight,... from SOURCES")
    ap.add_argument("--prompts", default=None, help="local {'prompt': ...} jsonl instead of --mix")
    ap.add_argument("--n-prompts", type=int, default=1000)
    ap.add_argument("--think-frac", type=float, default=0.2, help="share answered with thinking on")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--think-temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--think-max-tokens", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=128, help="requests in flight")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--backoff", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=None, help="first N prompts only (smoke test)")
    ap.add_argument(
        "--tokenizer", default=None, help="for token counts; default: the model; 'none'"
    )
    ap.add_argument("--log-every", type=int, default=50)
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    if not 0.0 <= args.think_frac <= 1.0:
        raise SystemExit("--think-frac must be in [0, 1]")
    os.makedirs(args.out_dir, exist_ok=True)
    prompts_path = os.path.join(args.out_dir, "prompts.jsonl")
    selection_path = os.path.join(args.out_dir, "selection.json")
    if os.path.exists(prompts_path):
        prompts = gc.read_jsonl(prompts_path)
        with open(selection_path, encoding="utf-8") as f:
            selection = json.load(f)
        changed = {k: v for k, v in selection_key(args).items() if selection.get(k) != v}
        if changed:
            raise SystemExit(
                f"{prompts_path} was chosen with other settings ({sorted(changed)}); "
                "use a new --out-dir or the old settings"
            )
    else:
        prompts, selection = choose_prompts(args)
        gc.write_jsonl(prompts_path, prompts)
        with open(selection_path, "w", encoding="utf-8") as f:
            json.dump(selection, f, indent=2)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    if args.model is None:
        probe = gc.ChatClient(args.base_url, api_key=args.api_key, concurrency=1)
        args.model = probe.served_model()
        probe.close()
    config = {
        "model": args.model,
        "prompts_sha256": gc.file_sha256(prompts_path),
        "temperature": args.temperature,
        "think_temperature": args.think_temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "think_max_tokens": args.think_max_tokens,
        "seed": args.seed,
    }
    gc.check_run_config(os.path.join(args.out_dir, "run.json"), config)
    samples_path = os.path.join(args.out_dir, "samples.jsonl")
    done = {s["id"] for s in gc.read_jsonl(samples_path)}
    todo = [p for p in prompts if p["id"] not in done]
    print(f"{len(prompts)} prompts, {len(prompts) - len(todo)} already done, model {args.model}")
    if todo:
        asyncio.run(generate(args, todo, samples_path, args.model))
    stats = build_outputs(args.out_dir, prompts, selection, config)
    print(
        f"kept {stats['kept']}/{stats['answered']} "
        f"(thinking off {stats['kept_by_mode']['thinking_off']}, "
        f"on {stats['kept_by_mode']['thinking_on']}); reasons {stats['reasons']}",
        flush=True,
    )
    return stats


if __name__ == "__main__":
    main()
