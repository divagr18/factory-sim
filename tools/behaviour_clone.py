"""Clone the scripted builder into a policy, to anchor a PPO run to.

    python tools/behaviour_clone.py --out runs/prior.pt --episodes 20000

The result is the frozen prior of VPT's fine-tuning loss (Baker et al. 2022):
`L = L_ppo + rho * KL(prior, policy)`, with rho decayed per update and the
entropy bonus removed. `train.py --prior <path>` does that half.

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
    out = {k: torch.from_numpy(chunk[k]).to(device, non_blocking=True) for k in chunk}
    out["mask"] = out["mask"].bool()
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
    p.add_argument("--action-space", choices=("v1", "v2"), default="v2")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    policy = Policy(action_space=args.action_space).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    seen, start = 0, time.perf_counter()
    history = []
    for chunk_index in range(max(1, args.episodes // args.chunk)):
        raw = demos.collect(
            args.chunk,
            action_space=args.action_space,
            threads=args.threads,
            seed_base=5_000_000 + chunk_index * args.chunk,
        )
        data = batch_to_device(raw, device)
        n = data["action"].shape[0]
        losses, hits = [], []
        for _epoch in range(args.epochs):
            order = torch.randperm(n, device=device)
            for s in range(0, n, args.minibatch):
                idx = order[s : s + args.minibatch]
                obs = [data[k][idx] for k in OBS_KEYS]
                actions = data["action"][idx]
                f = policy.features(*obs)
                logp, _ = policy.evaluate(f, data["mask"][idx], actions)
                loss = -logp.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
                with torch.no_grad():
                    chosen, _ = policy.act(f.detach(), data["mask"][idx], True)
                    hits.append((chosen == actions).all(1).float().mean().item())
                losses.append(loss.item())
        seen += n
        line = {
            "chunk": chunk_index,
            "samples": seen,
            "nll": round(float(np.mean(losses)), 4),
            "exact_action_match": round(float(np.mean(hits)), 4),
            "seconds": round(time.perf_counter() - start, 1),
        }
        history.append(line)
        print(json.dumps(line), flush=True)

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
