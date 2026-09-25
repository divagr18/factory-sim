"""Clone the scripted builder into a policy, to anchor a PPO run to.

    python tools/behaviour_clone.py --out runs/prior.pt --episodes 20000

The result is the frozen prior of VPT's fine-tuning loss (Baker et al. 2022):
`L = L_ppo + rho * KL(prior, policy)`, with rho decayed per update and the
entropy bonus removed. `train.py --prior <path>` does that half, and
`train.py --init <path>` starts the policy from it, as VPT fine-tunes its BC
model.

    python tools/behaviour_clone.py --task belt_smelting --out runs/belt-prior.pt \
        --episodes 512 --chunk 32

belt_smelting clones its own builder under parameterized-v3, scoring each
label under the per-operation mask of its operation (`Policy.evaluate` with
`op_masks`), which is the distribution PPO will sample and score it under.
The arguments are cloned drawn in order (`--autoregressive`, train.py's
default): a prior cloned with independent arguments leaves the conditioned
tail untrained, and a run that loads it would draw direction, item and amount
from noise.

Demonstrations are generated in chunks rather than held in one array: a single
observation's grid is 6x65x65 floats, so a hundred thousand of them would be
ten gigabytes. Each chunk is used once and dropped, which also means no two
gradient steps see the same demonstration.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsim import demos  # noqa: E402
from fsim.policy import Policy  # noqa: E402
from fsim.vec import OBS_KEYS  # noqa: E402


def batch_to_device(chunk: dict, device) -> dict:
    out = {
        k: torch.from_numpy(chunk[k]).to(device, non_blocking=True)
        for k in chunk
        if k != "episodes_kept"
    }
    out["mask"] = out["mask"].bool()
    if "op_masks" in out:
        out["op_masks"] = out["op_masks"].bool()
    out["entity_mask"] = out["entity_mask"].to(torch.int8)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("runs/prior.pt"))
    p.add_argument("--episodes", type=int, default=20_000, help="demonstrations in total")
    p.add_argument("--chunk", type=int, default=512, help="demonstrations generated at once")
    p.add_argument("--epochs", type=int, default=4, help="passes over each chunk")
    p.add_argument("--minibatch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--action-space", choices=("v1", "v2", "v3"), default=None)
    p.add_argument("--task", default="construct_smelting_line")
    p.add_argument(
        "--pool",
        type=Path,
        default=None,
        help="belt_smelting: where to write the recorded builds for "
        "train.py --demo-pool (default: next to --out, as <stem>.demos.json)",
    )
    p.add_argument(
        "--independent-arguments",
        dest="autoregressive",
        action="store_false",
        help="clone the five arguments drawn at once, not in order (train.py's flag)",
    )
    p.set_defaults(autoregressive=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    if args.action_space is None:
        args.action_space = "v3" if args.task == "belt_smelting" else "v2"

    device = torch.device(args.device)
    policy = Policy(action_space=args.action_space).to(device)
    policy.autoregressive = args.autoregressive
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    seen, start = 0, time.perf_counter()
    history = []
    records: list[dict] = []
    if args.pool is None and args.task == "belt_smelting":
        args.pool = args.out.with_name(args.out.stem + ".demos.json")
    for chunk_index in range(max(1, args.episodes // args.chunk)):
        raw = demos.collect(
            args.chunk,
            task=args.task,
            action_space=args.action_space,
            threads=args.threads,
            seed_base=5_000_000 + chunk_index * args.chunk,
            records=records,
        )
        kept = int(raw.get("episodes_kept", args.chunk))
        data = batch_to_device(raw, device)
        opm = data.get("op_masks")
        n = data["action"].shape[0]
        losses, hits = [], []
        for _epoch in range(args.epochs):
            order = torch.randperm(n, device=device)
            for s in range(0, n, args.minibatch):
                idx = order[s : s + args.minibatch]
                obs = [data[k][idx] for k in OBS_KEYS]
                actions = data["action"][idx]
                op_masks = None if opm is None else opm[idx]
                f = policy.features(*obs)
                logp, _ = policy.evaluate(f, data["mask"][idx], actions, op_masks)
                loss = -logp.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
                with torch.no_grad():
                    chosen, _ = policy.act(f.detach(), data["mask"][idx], True, op_masks=op_masks)
                    hits.append((chosen == actions).all(1).float().mean().item())
                losses.append(loss.item())
        seen += n
        line = {
            "chunk": chunk_index,
            "episodes_kept": kept,
            "samples": seen,
            "nll": round(float(np.mean(losses)), 4),
            "exact_action_match": round(float(np.mean(hits)), 4),
            "seconds": round(time.perf_counter() - start, 1),
        }
        history.append(line)
        print(json.dumps(line), flush=True)
        if args.pool is not None:
            # Rewritten each chunk, so a run cut short still leaves a pool.
            args.pool.write_text(json.dumps(records), encoding="utf-8")

    torch.save(policy.state_dict(), args.out)
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {"args": {k: str(v) for k, v in vars(args).items()}, "history": history}, indent=2
        ),
        encoding="utf-8",
    )
    print(json.dumps({"saved": str(args.out), "samples": seen}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
