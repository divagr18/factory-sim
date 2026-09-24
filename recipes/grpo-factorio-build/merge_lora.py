"""Merge a prime-rl LoRA broadcast (adapter_config.json + adapter_model.safetensors)
into a full HF checkpoint: W += (alpha / r) * B @ A for every adapted module.

    python merge_lora.py --base /root/models/sft --adapter outputs/<run>/broadcasts/step_20 --out /root/models/grpo

prime-rl writes adapter keys as plain module paths (`<module>.lora_A.weight`),
which PEFT's loader can skip without an error; this merges them directly and
fails if any adapted module has no base weight.
"""
import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

cfg = json.load(open(os.path.join(args.adapter, "adapter_config.json")))
scale = cfg["lora_alpha"] / cfg["r"]

base_files = sorted(f for f in os.listdir(args.base) if f.endswith(".safetensors"))
weights = {}
for f in base_files:
    with safe_open(os.path.join(args.base, f), "pt") as h:
        for k in h.keys():
            weights[k] = h.get_tensor(k)

merged = 0
with safe_open(os.path.join(args.adapter, "adapter_model.safetensors"), "pt") as h:
    keys = set(h.keys())
    for a_key in sorted(k for k in keys if k.endswith(".lora_A.weight")):
        module = a_key[: -len(".lora_A.weight")]
        b_key = module + ".lora_B.weight"
        w_key = module + ".weight"
        if b_key not in keys or w_key not in weights:
            raise SystemExit(f"no match for {module}")
        a = h.get_tensor(a_key).float()
        b = h.get_tensor(b_key).float()
        w = weights[w_key]
        weights[w_key] = (w.float() + scale * (b @ a)).to(w.dtype)
        merged += 1
print(f"merged {merged} modules at scale {scale}")

os.makedirs(args.out, exist_ok=True)
save_file(weights, os.path.join(args.out, "model.safetensors"), metadata={"format": "pt"})
for f in os.listdir(args.base):
    if not f.endswith(".safetensors") and not f.endswith(".index.json") and os.path.isfile(os.path.join(args.base, f)):
        shutil.copy(os.path.join(args.base, f), args.out)
print("MERGE_DONE", args.out)
