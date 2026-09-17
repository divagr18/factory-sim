# Throughput

`uv run python bench/bench.py --seconds 5 [--threads N]`

This replays the 600 actions of `construct_smelting_line_reference` in a loop
through `fsim_run`, so there is no Python between decisions. Each decision is
30 ticks, followed by the observation sweep the engine takes after every step
(entity and ore sweep, handle minting, memory). It does not include rendering
records for parity or building tensors, which lands in M4.

M3, 2026-09-17, on a 16-thread desktop CPU, Windows, MSVC `/O2`:

| Environments | Threads | Decisions/s | Per environment | Ticks/s |
|---:|---:|---:|---:|---:|
| 1 | 1 | 173,380 | 173,380 | 5.2M |
| 8 | 8 | 850,534 | 106,317 | 25.5M |
| 16 | 16 | 947,423 | 59,214 | 28.4M |

For comparison, the real engine through FactorioRL's harness runs 76
decisions/s on one worker and 261 on eight.

The M3 target was at least 100,000 decisions/s for the simulator alone.
