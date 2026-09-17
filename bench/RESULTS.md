# Throughput

`uv run python bench/bench.py --seconds 5 [--threads N]`

This replays the 600 actions of `construct_smelting_line_reference` in a loop
through `fsim_run`, so there is no Python between decisions. Each decision is
30 ticks, followed by the observation sweep the engine takes after every step
(entity and ore sweep, handle minting, memory). It does not include rendering
records for parity or building tensors, which lands in M4.

M3, 2026-09-17, on a Ryzen 7 5800H laptop (8 cores, 16 threads), Windows, MSVC `/O2`:

| Environments | Threads | Decisions/s | Per environment | Ticks/s |
|---:|---:|---:|---:|---:|
| 1 | 1 | 173,380 | 173,380 | 5.2M |
| 8 | 8 | 850,534 | 106,317 | 25.5M |
| 16 | 16 | 947,423 | 59,214 | 28.4M |

For comparison, the real engine through FactorioRL's harness runs 76
decisions/s on one worker and 261 on eight.

The M3 target was at least 100,000 decisions/s for the simulator alone.

## RL path (M4)

The same 600 decisions through `fsim_rl_run`. Each decision takes an action
vector, runs 30 ticks, computes the reward and termination, then fills the
full `local-v2` tensor observation (a 6x65x65 grid, 32 entity rows, self,
inventory and goal) and the 201-entry action mask.

M4, 2026-09-17, same machine:

| Environments | Threads | Decisions/s | Per environment |
|---:|---:|---:|---:|
| 1 | 1 | 60,467 | 60,467 |
| 8 | 8 | 382,431 | 47,804 |
| 1, after M5's encoder work | 1 | 79,153 | 79,153 |
| 8, after M5's encoder work | 8 | 530,880 | 66,360 |

The encoding dominates: rewriting 25,350 grid cells per decision is a memory
cost, not a simulation cost.

## RL layer parts (M5)

`python bench/bench_rl_parts.py`, one environment, microseconds per call:

| part | before | after |
|---|---:|---:|
| `fsim_rl_step` (30 ticks, reward, termination) | 8.8 | 7.3 |
| `fsim_rl_encode` (float observation) | 11.5 | 3.6 |
| `fsim_rl_encode8` (packed observation) | n/a | 2.8 |
| `fsim_rl_mask` | 1.2 | 0.7 |

Two changes account for most of this:
- **`rint`:** MSVC's `rint` reads the floating-point environment on every
  call. At two calls per ore tile, it was most of the encoder's time; an
  inline round-half-to-even replaced it.
- **Placement domain:** it now marks an 11x11 occupancy window in one pass,
  instead of scanning every entity and water tile once per candidate tile.

## Learner (M5)

`python bench/bench_train.py --envs N --threads 16`: PPO in `train.py`,
64-decision rollouts, 4096-sample minibatches, 2 epochs.

Hardware: RTX 3050 Laptop GPU (4 GB, PCIe 3.0 x8) and Ryzen 7 5800H.

| envs | rollout steps/s | update samples/s | overall steps/s |
|---:|---:|---:|---:|
| 256, before | ~13,000 | ~20,000 | 7,800 (measured in a real run) |
| 256 | 57,372 | 43,048 | 24,594 |
| 512 | 63,461 | 38,418 | 23,931 |
| 256, `--epochs 1` | 62,303 | 81,767 | 35,360 |
| 512, `--epochs 1` | 66,942 | 80,447 | 36,538 |

These are cool-GPU numbers. A sustained 4M-step run on this laptop
averaged **14,500 steps/s**. The GPU reached 83 C and throttled: the update
went from 0.58 s to 1.03 s at about update 180.

What changed, in the order the profile pointed at:

1. **Rollout inference was CPU-bound.** Each decision launched about 830 small
   CUDA kernels, including six `multinomial` calls. Now:
   - the action head builds all five argument masks and distributions as one
     padded tensor, and samples with Gumbel-max (same distribution, no host
     sync);
   - the whole rollout step (slicing the observation block, the extractor,
     sampling, the value) is one CUDA graph replay.

   The graph alone is worth 1.6x on the rollout (38.5k to 61.7k steps/s).
2. **Transfer:** observations went from 104 KB to 9.2 KB per environment:
   - the grid's five flag planes are bits and the amount plane is bytes
     (`fsim_obs8`);
   - the environments write straight into page-locked memory;
   - there is one host-to-device copy per decision.

   The graph unpacks the grid once, and the rollout buffer stores it as bytes,
   so the update never unpacks. Unpacking per minibatch had cost more than
   the smaller copy saved.
3. **The update is GPU-bound.** Capturing it in a CUDA graph measured the
   same (46.6 vs 46.4 ms per minibatch), so only doing less work helps. The
   extractor's first layer (4x4, stride 4) is now the matrix multiply it is
   equivalent to, on `pixel_unshuffle` patches with the same weights. It is
   followed by a channels-last convolution. A byte grid goes straight to bf16
   instead of through five full-size passes. The grid path went from 33 ms to
   25 ms per 4096 samples.

What is left is the GPU:
- about 6 µs per sample of extractor compute at inference;
- about 23 µs per sample, per epoch, in the update.

`--epochs 1` (PufferLib's default) doubles the update rate, but that is a
learning choice, not an optimisation, and the default stays at 2.

### On a desktop: RTX 4060 (8 GB) and Ryzen 5 5600T (6 cores, 12 threads)

`--threads 12`, same commands:

| envs | rollout steps/s | update samples/s | overall steps/s |
|---:|---:|---:|---:|
| 256 | 120,205 | 77,583 | 47,151 |
| 512 | 122,717 | 81,695 | 49,045 |
| 1024 | 91,528 | 82,936 | 43,510 |
| 512, `--epochs 1` | 122,461 | 158,775 | 69,137 |

A 5M-step training run at 512 envs (`--demo-starts 0.5`) held **47,158
steps/s** from start to finish. The GPU stayed at 57 C, and every update
took 0.40 s. That is 3.3 times the laptop's sustained rate.

At this rate a 20M-step run takes about 7 minutes.

Other numbers on the same machine:
- the simulator alone: 169k decisions/s on one core, 981k on 12 threads;
- the RL path: 91k decisions/s on one core, 541k on 12 threads.
