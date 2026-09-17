"""Per-call cost of the RL layer's parts, single-threaded, in microseconds.

Replays the construct_smelting_line reference's decisions and times
`fsim_rl_step` (the simulation and reward), `fsim_rl_encode`,
`fsim_rl_encode8` and `fsim_rl_mask` separately, over the same states.

    python bench/bench_rl_parts.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsim import ffi, lib  # noqa: E402
from fsim.parity import GOLDEN  # noqa: E402
from fsim.rl import RlEnv  # noqa: E402
from fsim.trace import read_trace  # noqa: E402


def main() -> int:
    header, records = read_trace(GOLDEN / "construct_smelting_line_reference.jsonl.xz")
    vectors = [ffi.new("int32_t[6]", r["transition"]["action"]["vector"]) for r in records[1:]]
    env = RlEnv()
    obs = ffi.new("fsim_obs *")
    compact = hasattr(lib, "fsim_rl_encode8")
    obs8 = ffi.new("fsim_obs8 *") if compact else None
    mask = ffi.new("uint8_t[]", lib.RL_MASK_SIZE)
    totals = {"step": 0.0, "encode": 0.0, "encode8": 0.0, "mask": 0.0, "potential": 0.0}
    calls = 0
    clock = time.perf_counter
    for _ in range(3):
        env.reset(header["task"], header["blueprint"])
        for vector in vectors:
            a = clock()
            lib.fsim_rl_step(env.rl, vector)
            b = clock()
            lib.fsim_rl_encode(env.rl, obs)
            c = clock()
            if compact:
                lib.fsim_rl_encode8(env.rl, obs8)
            d = clock()
            lib.fsim_rl_mask(env.rl, mask)
            e = clock()
            lib.fsim_rl_potential(env.rl)
            f = clock()
            for key, dt in zip(totals, (b - a, c - b, d - c, e - d, f - e), strict=True):
                totals[key] += dt
            calls += 1
            if env.rl.done:
                break
    for key, total in totals.items():
        print(f"{key:10s} {1e6 * total / calls:8.1f} us")
    print("(each includes ~0.1 us of cffi call overhead; the step includes a verification "
          "window of 3600 ticks once per episode)")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
