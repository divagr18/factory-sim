"""Where a checkpoint's episodes fall short of a working line.

Rolls a trained policy (a `train.py` state dict) out on training scenes and
reports, per episode, how far it got -- the line potential's stages, whether
the machines it built are still standing at the end, what it did with its
operations -- so a flat success curve can be read as a specific missing step.

    python tools/diagnose.py runs/progress-s1/best.pt --episodes 128
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsim import lib  # noqa: E402
from fsim.policy import Policy  # noqa: E402
from fsim.vec import OBS_KEYS, VecEnv  # noqa: E402

OP_NAMES = (
    ["move_n", "move_e", "move_s", "move_w", "step_n", "step_e", "step_s", "step_w"]
    + ["nudge_n", "nudge_e", "nudge_s", "nudge_w", "place", "mine", "rotate", "rotate_rev"]
    + ["give", "take", "set_recipe", "craft", "cancel", "wait"]
)
K_DRILL, K_FURNACE = 1, 2


def standing(rl) -> dict:
    env = rl.env
    counts = Counter()
    for i in range(env.entity_count):
        e = env.entities[i]
        if e.alive and not e.neutral:
            counts[e.kind] += 1
    return counts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--device", default="cpu", help="cpu leaves the GPU to a training run")
    p.add_argument("--action-space", choices=("v1", "v2"), default="v1")
    args = p.parse_args()
    device = torch.device(args.device)
    policy = Policy(action_space=args.action_space).to(device)
    policy.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    policy.eval()

    n = min(64, args.episodes)
    env = VecEnv(
        n, threads=8, eval_seeds=True, max_steps=args.max_steps, action_space=args.action_space
    )
    obs, masks = env.reset()
    ops = Counter()
    decode_failures = 0
    steps = 0
    stage_steps: list[dict] = [{} for _ in range(n)]
    results = []
    while len(results) < args.episodes:
        with torch.no_grad():
            t = {k: torch.from_numpy(obs[k]).to(device) for k in OBS_KEYS}
            f = policy.features(*(t[k] for k in OBS_KEYS))
            actions, _ = policy.act(f, torch.from_numpy(masks).to(device).bool(), args.greedy)
        actions = actions.cpu().numpy()
        ops.update(actions[:, 0].tolist())
        # Stage bookkeeping reads the state before the step replaces it.
        before = [standing(env.rls[i]) for i in range(n)]
        obs, masks, _, term, trunc, finished = env.step(actions)
        steps += n
        decode_failures += int(env.flags[:, 3].sum())
        ended = set(np.flatnonzero(term | trunc).tolist())
        for i in range(n):
            if i in ended:
                continue
            phi = lib.fsim_rl_potential(env.rls[i])
            for name, threshold in (("drill", 0.2), ("line", 0.5), ("fuelled", 0.7)):
                if phi >= threshold and name not in stage_steps[i]:
                    stage_steps[i][name] = int(env.lengths[i])
        for i, record in zip(sorted(ended), finished, strict=True):
            record = dict(record)
            record["stages"] = stage_steps[i]
            record["standing_at_end"] = {
                "drills": before[i].get(K_DRILL, 0),
                "furnaces": before[i].get(K_FURNACE, 0),
            }
            results.append(record)
            stage_steps[i] = {}
    env.close()
    results = results[: args.episodes]

    def share(pred):
        return round(sum(1 for r in results if pred(r)) / len(results), 3)

    report = {
        "checkpoint": str(args.checkpoint),
        "episodes": len(results),
        "greedy": args.greedy,
        "success": share(lambda r: r["success"]),
        "reached": {
            s: share(lambda r, s=s: s in r["stages"]) for s in ("drill", "line", "fuelled")
        },
        "peak_potential_mean": round(float(np.mean([r["peak_potential"] for r in results])), 3),
        "peak_at_least_line": share(lambda r: r["peak_potential"] >= 0.5),
        "verified_output_mean": round(
            float(np.mean([max(0.0, r["verified_output"]) for r in results])), 3
        ),
        "no_drill_standing_at_end_after_building_one": share(
            lambda r: "drill" in r["stages"] and r["standing_at_end"]["drills"] == 0
        ),
        "median_step_reached": {
            s: float(np.median([r["stages"][s] for r in results if s in r["stages"]] or [-1]))
            for s in ("drill", "line", "fuelled")
        },
        "decode_failure_rate": round(decode_failures / max(1, steps), 4),
        "operations": {
            OP_NAMES[k]: round(v / sum(ops.values()), 3) for k, v in sorted(ops.items())
        },
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
