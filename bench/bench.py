"""Simulator-only throughput, in decisions per second (one decision = 30 ticks).

Replays the construct_smelting_line reference's 600 actions in a loop, on one
environment and then on N environments on N threads (cffi releases the GIL
around C calls). Rendering to the wire format is off: this measures the C core,
including the observation sweep it takes after every decision.

    uv run python bench/bench.py [--seconds 5] [--threads 8]
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsim import Sim, ffi, lib  # noqa: E402
from fsim.parity import GOLDEN  # noqa: E402
from fsim.trace import read_trace  # noqa: E402

TRACE = "construct_smelting_line_reference"


def worker(header, actions, seconds, out, index):
    sim = Sim()
    batch = ffi.new("fsim_action[]", len(actions))
    for i, (key, args) in enumerate(actions):
        batch[i] = sim.action(key, args)[0]
    ticks = header["decision_ticks"]
    env = sim.env
    count = 0
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        sim.reset(header["blueprint"])
        lib.fsim_run(env, batch, len(actions), ticks)
        count += len(actions)
    out[index] = count


def rl_worker(header, records, seconds, out, index):
    from fsim.rl import RlEnv

    env = RlEnv()
    vectors = [v for r in records[1:] for v in r["transition"]["action"]["vector"]]
    batch = ffi.new("int32_t[]", vectors)
    count = 0
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        env.reset(header["task"], header["blueprint"])
        count += lib.fsim_rl_run(env.rl, batch, len(vectors) // 6, env.obs_c, env.mask_c)
    out[index] = count


def run(threads, seconds, header, payload, target=worker):
    out = [0] * threads
    pool = [
        threading.Thread(target=target, args=(header, payload, seconds, out, i))
        for i in range(threads)
    ]
    start = time.perf_counter()
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    elapsed = time.perf_counter() - start
    return sum(out) / elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    header, records = read_trace(GOLDEN / f"{TRACE}.jsonl.xz")
    actions = [
        (r["transition"]["action"]["key"], r["transition"]["action"]["arguments"])
        for r in records[1:]
    ]
    print(f"cpu count: {os.cpu_count()}")
    n = args.threads
    for label, payload, target in (
        ("simulator only (step + observation sweep)", actions, worker),
        ("RL path (step + reward + tensor encoding + action mask)", records, rl_worker),
    ):
        one = run(1, args.seconds, header, payload, target)
        many = run(n, args.seconds, header, payload, target)
        print(label)
        print(f"  1 env:  {one:,.0f} decisions/s")
        print(f"  {n} envs: {many:,.0f} decisions/s total ({many / n:,.0f} per env)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
