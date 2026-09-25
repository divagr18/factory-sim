"""LoRA SFT warm start, then merge into a full HF checkpoint.

    CUDA_VISIBLE_DEVICES=0 uv run --no-sync python sft_lora.py
        --data sft.jsonl --out /root/models/sft
        [--replay replay.jsonl --replay-frac 0.2] [--empty-think mask|train]

--data takes build_sft_data.py's prompt/completion rows or rationalize.py's
messages rows. The loss is on the final assistant turn only. The prompt is
rendered exactly as at RL time: Qwen3.5 chat template, thinking off unless a
row asks for it (`sft_data.encode`).

--empty-think: with thinking off, the template's generation prompt ends with an
empty think block (`<think>\\n\\n</think>\\n\\n`). `mask`, the default, keeps it
out of the loss. This script always did so, because the block sat in the
prompt. `train` puts it in the loss, as a trainer that takes the loss over the
whole rendered assistant turn does.

--replay: messages rows (make_replay.py), mixed in so that they make up
--replay-frac of each epoch. Thinking-on rows keep their reasoning in the loss.

--model may be a local checkpoint, for a second Iterative-SFT round
(`--model /root/models/sft`).
"""

import argparse
import math
import os
import shutil
import sys

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sft_data  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
ap.add_argument("--data", default="/root/recipe/sft.jsonl")
ap.add_argument("--out", default="/root/models/sft")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--empty-think", choices=sft_data.EMPTY_THINK_MODES, default="mask")
ap.add_argument("--replay", default=None, help="messages-format jsonl mixed into training")
ap.add_argument("--replay-frac", type=float, default=0.2, help="share of each epoch from --replay")
ap.add_argument("--max-len", type=int, default=16384, help="longer examples are dropped")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

torch.manual_seed(args.seed)
tok = AutoTokenizer.from_pretrained(args.model)


def encode_all(path):
    out, dropped = [], 0
    for row in sft_data.load_examples(path):
        e = sft_data.encode(
            tok,
            row["messages"],
            enable_thinking=row["enable_thinking"],
            empty_think=args.empty_think,
        )
        if len(e["input_ids"]) > args.max_len:
            dropped += 1
            continue
        out.append((e["input_ids"], e["labels"]))
    print(f"{path}: {len(out)} examples, {dropped} longer than {args.max_len} dropped")
    return out


pools = {"task": encode_all(args.data)}
pools["replay"] = encode_all(args.replay) if args.replay else []
plan = sft_data.epoch_plan(
    len(pools["task"]), len(pools["replay"]), args.replay_frac, args.epochs, args.seed
)
print(
    "examples",
    len(pools["task"]),
    "replay per epoch",
    sft_data.replay_per_epoch(len(pools["task"]), len(pools["replay"]), args.replay_frac),
    "max len",
    max(len(ids) for p in pools.values() for ids, _ in p),
    "empty think",
    args.empty_think,
)
first_ids, first_labels = pools["task"][0]
n_prompt = next(i for i, lab in enumerate(first_labels) if lab != -100)
print("prompt tail:", repr(tok.decode(first_ids[max(0, n_prompt - 12) : n_prompt])))

model = AutoModelForImageTextToText.from_pretrained(
    args.model, dtype=torch.bfloat16, device_map={"": 0}
)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
cfg = LoraConfig(
    r=args.rank,
    lora_alpha=2 * args.rank,
    lora_dropout=0.0,
    # language model only: every linear projection, attention and MLP
    target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkvz|in_proj_ba|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)",
)
model = get_peft_model(model, cfg)
model.print_trainable_parameters()

opt = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0
)
total = sum(math.ceil(len(order) / args.accum) for order in plan)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / 3) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total))
)
model.train()
step = 0
for ep, order in enumerate(plan):
    run = 0.0
    for i, (source, idx) in enumerate(order):
        ids, labels = pools[source][idx]
        ids_t = torch.tensor([ids], device="cuda")
        lab_t = torch.tensor([labels], device="cuda")
        loss = model(input_ids=ids_t, labels=lab_t).loss / args.accum
        loss.backward()
        run += loss.item()
        if (i + 1) % args.accum == 0 or i + 1 == len(order):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            print(
                f"epoch {ep} step {step}/{total} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e}",
                flush=True,
            )
            run = 0.0

model = model.merge_and_unload()
os.makedirs(args.out, exist_ok=True)
model.save_pretrained(args.out, safe_serialization=True)
tok.save_pretrained(args.out)
# Tokenizer/processor files vLLM needs; a local checkpoint already holds them.
if os.path.isdir(args.model):
    snap = args.model
else:
    snap = snapshot_download(args.model, local_files_only=True)
for f in os.listdir(snap):
    if (
        not f.endswith(".safetensors")
        and not f.endswith(".index.json")
        and f != "config.json"
        and os.path.isfile(os.path.join(snap, f))
        and not os.path.exists(os.path.join(args.out, f))
    ):
        shutil.copy(os.path.join(snap, f), args.out)
print("SFT_DONE", args.out)
