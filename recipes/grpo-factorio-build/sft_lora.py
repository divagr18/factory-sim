"""LoRA SFT warm start on evolved builder programs, then merge into a full HF checkpoint.

    CUDA_VISIBLE_DEVICES=0 uv run --no-sync python sft_lora.py
        --data sft.jsonl --out /root/models/sft

Loss is on the assistant completion only. The prompt is rendered exactly as it is
at RL time: Qwen3.5 chat template with enable_thinking=False.
"""

import argparse
import json
import math
import os
import random
import shutil

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
ap.add_argument("--data", default="/root/recipe/sft.jsonl")
ap.add_argument("--out", default="/root/models/sft")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--accum", type=int, default=8)
args = ap.parse_args()

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(args.model)
rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]

examples = []
for r in rows:
    prompt = tok.apply_chat_template(
        r["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    completion = r["completion"][0]["content"] + "<|im_end|>\n"
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    c = tok(completion, add_special_tokens=False)["input_ids"]
    examples.append((p + c, [-100] * len(p) + c))
print("examples", len(examples), "max len", max(len(e[0]) for e in examples))
print("prompt tail:", repr(prompt[-80:]))

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
total = math.ceil(len(examples) * args.epochs / args.accum)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / 3) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total))
)
model.train()
step = 0
rng = random.Random(0)
for ep in range(args.epochs):
    order = list(range(len(examples)))
    rng.shuffle(order)
    run = 0.0
    for i, idx in enumerate(order):
        ids, labels = examples[idx]
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
snap = snapshot_download(args.model, local_files_only=True)  # tokenizer/processor files vLLM needs
for f in os.listdir(snap):
    if (
        not f.endswith(".safetensors")
        and not f.endswith(".index.json")
        and f != "config.json"
        and not os.path.exists(os.path.join(args.out, f))
    ):
        shutil.copy(os.path.join(snap, f), args.out)
print("SFT_DONE", args.out)
