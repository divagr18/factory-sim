"""Evaluate a checkpoint on the whole level space, not one held-out family.

`construct_smelting_line` has a single test family, `obstructed_patch`, and
the training families were chosen partly *because* they lifted it --
`varied_patch` and `cluttered_patch` were added when it sat at 0-1.4%, and
took it to 88%. So "held-out success" on this task measures two things at
once: how well a policy generalises, and how well the training distribution
happens to be fitted to that one family.

That distinction matters as soon as a run stops training on those families. A
UED curriculum drifts away from them by design -- 99.6% of its buffer is
levels it invented -- and would score worse on a family the hand-written set
was tuned toward even if it generalised better everywhere else.

This measures everywhere else: levels drawn uniformly from the parameter
space of `fsim/ued.py`, which contains the hand-written families as points but
is far larger. Neither arm is tuned to it.

It is **not** a harder test, and should not be presented as one. Measured,
`ar-s1` scores 0.953 here against 0.891 on `obstructed_patch`: a random level
is usually an ordinary patch, where the test family is a narrow 3x11 strip
behind a wall. What it gives is a *second, independent axis*. A curriculum
that scores higher here and lower there is one that drifted away from a
family the training set was fitted to; one that scores lower on both is
simply worse. Watch the ceiling -- at 0.95 there is little headroom left to
distinguish arms.

    python tools/eval_broad.py runs/ar-s1/best.pt runs/u2-s1/best.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fsim import ued  # noqa: E402
from fsim.policy import Policy  # noqa: E402
from fsim.vec import VecEnv  # noqa: E402
from train import features, to_device, wilson  # noqa: E402

#: Levels are drawn from a seed disjoint from anything training used.
BROAD_SEED = 987_654_321


def broad_levels(task: str, count: int, seed: int = BROAD_SEED) -> list:
    rng = random.Random(seed)
    return [ued.random_level(task, rng) for _ in range(count)]


def evaluate_levels(policy, device, task: str, levels, action_space="v2",
                    max_steps=600, threads=8, epsilon=0.0) -> dict:  # fmt: skip
    """Success over an explicit list of levels, one episode each."""
    n = min(64, len(levels))
    queue = list(levels)
    handed: list = []

    def source(i, seed):
        level = queue.pop() if queue else handed[i % max(1, len(handed))]
        handed.append(level)
        return level.origin, ued.build(level)

    env = VecEnv(
        n, task, threads=threads, shaping=False, action_space=action_space,
        max_steps=max_steps, level_source=source,
    )  # fmt: skip
    obs, masks = env.reset()
    done: list[dict] = []
    try:
        while len(done) < len(levels):
            tensors = to_device(obs, device)
            mask = torch.from_numpy(masks).to(device).bool()
            actions, _ = policy.act(features(policy, tensors), mask, False, epsilon)
            obs, masks, _, term, trunc, finished = env.step(actions.cpu().numpy())
            done.extend(finished)
            if not queue and len(done) >= len(levels):
                break
    finally:
        env.close()
    done = done[: len(levels)]
    wins = sum(r["success"] for r in done)
    low, high = wilson(wins, len(done))
    return {
        "episodes": len(done),
        "success": wins / max(1, len(done)),
        "success_ci95": [round(low, 4), round(high, 4)],
        "line_built": sum(r["peak_potential"] >= 0.5 for r in done) / max(1, len(done)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--task", default="construct_smelting_line")
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--action-space", default="v2")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--family",
        help="evaluate a named hand-written family instead of the broad space. "
        "Mostly a check on this tool: a checkpoint scored here against its own "
        "test family should reproduce the number train.py reported.",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.family:
        rng = random.Random(BROAD_SEED)
        levels = [ued.family_params(args.task, args.family, rng) for _ in range(args.episodes)]
        print(f"{args.episodes} scenes of {args.family}, seed {BROAD_SEED}")
    else:
        levels = broad_levels(args.task, args.episodes)
        print(f"{args.episodes} levels drawn uniformly from the parameter space, "
              f"seed {BROAD_SEED}")
    print(f"{'checkpoint':28s} {'success':>9s} {'ci95':>18s} {'line_built':>11s}")
    results = {}
    for path in args.checkpoints:
        policy = Policy(action_space=args.action_space).to(device)
        policy.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        policy.autoregressive = True
        policy.eval()
        with torch.no_grad():
            out = evaluate_levels(
                policy, device, args.task, levels, args.action_space,
                threads=args.threads, epsilon=args.epsilon,
            )  # fmt: skip
        name = f"{path.parent.name}/{path.name}"
        results[name] = out
        ci = f"[{out['success_ci95'][0]:.3f}, {out['success_ci95'][1]:.3f}]"
        print(f"{name:28s} {out['success']:9.4f} {ci:>18s} {out['line_built']:11.4f}")
    if args.out:
        args.out.write_text(json.dumps(results, indent=1), "utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
