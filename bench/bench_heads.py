"""Where a v2 update's time goes, against v1, at one minibatch.

    python bench/bench_heads.py --batch 4096

Times forward and backward of the pieces separately -- the extractor, the
pointer head over the entity rows, the placement convolution -- so the v2/v1
gap is attributed rather than guessed at.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsim import lib  # noqa: E402
from fsim.policy import Policy  # noqa: E402
from fsim.rl import NVEC  # noqa: E402


def inputs(batch, device, dtype=torch.float32):
    obs = (
        torch.randint(0, 255, (batch, 6, 65, 65), device=device, dtype=torch.uint8),
        torch.randn(batch, 32, 16, device=device, dtype=dtype),
        torch.ones(batch, 32, device=device, dtype=torch.int8),
        torch.randn(batch, 12, device=device, dtype=dtype),
        torch.randn(batch, 14, device=device, dtype=dtype),
        torch.randn(batch, 12, device=device, dtype=dtype),
    )
    mask = torch.zeros(batch, lib.RL_MASK_SIZE, device=device, dtype=torch.bool)
    offset = 0
    for size in NVEC:
        mask[:, offset : offset + size] = True
        offset += size
    actions = torch.zeros(batch, 6, device=device, dtype=torch.long)
    actions[:, 0] = 12
    return obs, mask, actions


def timed(fn, warmup=3, iters=10, device="cuda"):
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    out = {"batch": args.batch, "device": str(device)}

    for space in ("v1", "v2"):
        policy = Policy(action_space=space).to(device)
        if device.type == "cuda":
            policy.extractor.input_dtype = torch.bfloat16
            policy = policy.to(memory_format=torch.channels_last)
        obs, mask, actions = inputs(args.batch, device)
        autocast = torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda")

        def features():
            with autocast:
                return policy.features(*obs)

        def forward():
            with autocast:
                f = policy.features(*obs)
                return policy.evaluate(f, mask, actions)

        def full():
            with autocast:
                f = policy.features(*obs)
                logp, entropy = policy.evaluate(f, mask, actions)
                loss = -(logp.mean() + entropy.mean()) + policy.value(f).mean()
            policy.zero_grad(set_to_none=True)
            loss.backward()

        out[space] = {
            "extractor_ms": round(timed(features, device=device.type), 2),
            "forward_ms": round(timed(forward, device=device.type), 2),
            "forward_backward_ms": round(timed(full, device=device.type), 2),
        }
        if space == "v2":
            with autocast:
                f = policy.features(*obs)
            context = torch.randn(
                args.batch, policy.features_dim + 22, device=device, dtype=f.dtype
            )
            start = policy.features_dim
            width = policy.rows * policy.row_dim
            rows = f[:, start : start + width].reshape(args.batch, policy.rows, policy.row_dim)
            crop = f[:, start + width :].reshape(args.batch, 6, policy.crop, policy.crop)

            def pointer():
                with autocast:
                    return policy.target_head(rows, context)

            def placement():
                with autocast:
                    return policy.place_head(crop, context)

            out[space]["pointer_head_ms"] = round(timed(pointer, device=device.type), 2)
            out[space]["place_head_ms"] = round(timed(placement, device=device.type), 2)
        del policy, obs, mask, actions
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out["update_ratio_v2_over_v1"] = round(
        out["v2"]["forward_backward_ms"] / out["v1"]["forward_backward_ms"], 2
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
