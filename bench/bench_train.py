"""Learner throughput: rollout and update phases of train.py, timed apart.

    python bench/bench_train.py --envs 256 --updates 6
    python bench/bench_train.py --envs 256 --updates 6 --no-graph

Runs `train.main` for a few updates with evaluation off, then reads the
per-phase timings it logs. The first update (graph capture, cudnn autotuning)
is excluded.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--updates", type=int, default=6)
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--minibatch", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        argv = [
            "--run", "bench", "--out", tmp, "--envs", str(args.envs),
            "--horizon", str(args.horizon), "--threads", str(args.threads),
            "--steps", str(args.envs * args.horizon * args.updates),
            "--eval-every", str(10**12), "--no-final", "--shaping", "progress",
            "--minibatches", str(args.envs * args.horizon // args.minibatch),
            "--epochs", str(args.epochs),
        ] + (["--no-graph"] if args.no_graph else [])  # fmt: skip
        train.main(argv)
        rows = [
            json.loads(x) for x in (Path(tmp) / "bench" / "metrics.jsonl").read_text().splitlines()
        ]
    rows = rows[1:]
    rollout = sum(r["time_rollout"] for r in rows)
    update = sum(r["time_update"] for r in rows)
    samples = args.envs * args.horizon * len(rows)
    print(
        json.dumps(
            {
                "envs": args.envs,
                "epochs": args.epochs,
                "graph": not args.no_graph,
                "rollout_steps_per_s": round(samples / rollout),
                "update_samples_per_s": round(samples / update),
                "overall_steps_per_s": round(samples / (rollout + update)),
                "rollout_ms_per_step": round(1000 * rollout / (args.horizon * len(rows)), 2),
                "update_s": round(update / len(rows), 3),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
