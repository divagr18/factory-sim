"""Replay FactorioRL's real-engine transfer episodes in the simulator.

`FactorioRL/tools/sim_transfer.py` writes every episode it plays on the engine
-- the scene payload, the action vectors the policy chose, and the outcome --
to `sim-transfer-m5.episodes.jsonl.xz`. Two checks:

* **open loop**: step the same vectors through the simulator; the verified
  output and success must be the engine's (the simulator predicts what the
  engine did with the same actions);
* **closed loop** (with `--policy`, greedy episodes only): run the exported
  policy in the simulator from the same scene; its action sequence must be the
  one it chose on the engine, since the simulator reproduces the tensors and
  masks bit for bit and greedy decoding is deterministic.

    python tools/replay_transfer.py ../FactorioRL/docs/evidence/sim-transfer-m5.episodes.jsonl.xz \\
        --policy runs/X/policy.ts
"""

from __future__ import annotations

import argparse
import json
import lzma
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsim.rl import RlEnv  # noqa: E402


def reset(env: RlEnv, episode: dict) -> dict:
    return env.reset(
        episode["task"],
        episode["blueprint"],
        decision_ticks=episode["decision_ticks"],
        max_steps=episode["max_decision_steps"],
        construction_tick_limit=episode["construction_tick_limit"],
    )


def open_loop(episode: dict) -> dict:
    env = RlEnv()
    reset(env, episode)
    total = 0.0
    for vector in episode["vectors"]:
        _, reward, terminated, truncated, _ = env.step(vector)
        total += reward
        if terminated or truncated:
            break
    return {
        "success": bool(env.rl.success),
        "verified_output": float(env.rl.verified_output) if env.rl.verified else None,
        "return": round(total, 6),
    }


def closed_loop(episode: dict, module) -> int | None:
    """The first decision at which the simulator's greedy choice differs, or None."""
    import torch

    env = RlEnv()
    obs = reset(env, episode)
    keys = ("grid", "entities", "entity_mask", "self", "inventory", "goal")
    for index, recorded in enumerate(episode["vectors"]):
        tensors = [torch.from_numpy(obs[k].copy()[None]) for k in keys]
        mask = torch.from_numpy(env.mask.astype(bool)[None])
        with torch.no_grad():
            chosen = module(*tensors, mask, True)[0].tolist()
        if chosen != recorded:
            return index
        obs, _, terminated, truncated, _ = env.step(chosen)
        if terminated or truncated:
            break
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("episodes", type=Path)
    parser.add_argument("--policy", type=Path, help="the exported policy, for closed loop")
    args = parser.parse_args()
    lines = lzma.decompress(args.episodes.read_bytes()).decode().splitlines()
    episodes = [json.loads(line) for line in lines]
    module = None
    if args.policy is not None:
        import torch

        module = torch.jit.load(str(args.policy), map_location="cpu").eval()
    rows, mismatches = [], 0
    for episode in episodes:
        if episode["outcome"]["excluded"]:
            continue
        sim = open_loop(episode)
        real = episode["outcome"]
        same = sim["success"] == real["success"] and (
            real["verified_output"] is None
            or abs((sim["verified_output"] or 0.0) - real["verified_output"]) < 1e-9
        )
        row = {"split": episode["split"], "episode": episode["episode"], "real": real,
               "sim": sim, "same_outcome": same}  # fmt: skip
        if module is not None:
            row["closed_loop_first_difference"] = closed_loop(episode, module)
            same = same and row["closed_loop_first_difference"] is None
        mismatches += 0 if same else 1
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = {
        "episodes": len(rows),
        "same_outcome": sum(r["same_outcome"] for r in rows),
        "closed_loop_identical": (
            sum(r["closed_loop_first_difference"] is None for r in rows) if module else None
        ),
    }
    print(json.dumps(summary))
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
