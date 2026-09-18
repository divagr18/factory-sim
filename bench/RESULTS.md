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

## Where a v2 update's time actually goes (2026-09-18, RTX 3050, batch 4096)

`bench/bench_heads.py` times the pieces apart. v2's update is 1.9x v1's, and
the gap is the two extra heads and their backward, not the extractor:

| | v1 | v2 |
|---|---|---|
| extractor | 22.35 ms | 21.31 ms |
| forward | 21.88 ms | 30.57 ms |
| forward + backward | 35.28 ms | 66.89 ms |
| pointer head (forward) | - | 1.86 ms |
| placement head (forward) | - | 5.62 ms |

Inside the extractor, the grid path is 84% of it, and **moving the data costs
more than the arithmetic**:

| piece | ms | |
|---|---|---|
| uint8 -> bf16 and /255 | 5.46 | 2.72 of it is the `/255` pass alone |
| `_grid` | 12.28 | 6.34 of it is unshuffle + permute + copy |
| crop_net | 0.27 | |
| entity_net | 0.84 | |
| vector_net | 0.24 | |

### What did not work, measured end to end

Three rewrites that were faster in isolation and slower, or no better, in the
whole policy step. The isolated numbers are real; they just do not survive
contact with `channels_last`, which the policy is converted to as a whole.

| variant | isolated | whole step |
|---|---|---|
| baseline | 14.99 ms (grid path) | **67.51 ms** |
| `conv2d` stride 4 on the sliced grid | 9.15 ms | 78.65 ms |
| explicit view + permute instead of `pixel_unshuffle` | 10.20 ms | 69.67 ms |
| both, with `channels_last` off | - | 82.88 ms |

The lesson is the one the grid path's own comment already recorded: the linear
formulation is not obviously right, it is *measured* right, and it stays right
only in the layout the rest of the policy runs in. `torch.compile` could fuse
the conversion and the permute, but there is no working Triton on this
platform, so it is not available to measure.

### What did work

Dropping the padding on the placement head's second convolution: only the
middle 11x11 of the 13x13 crop names a slot, and a padded convolution's
interior is exactly an unpadded one's whole output, so it is the same numbers
over 121 positions instead of 169. Isolated, 1.96 ms -> 1.53 ms; in the whole
step, 67.51 ms -> 65.60 ms. The end-to-end confirmation is within this
laptop's thermal noise over a long benchmarking session, so it is claimed on
the arithmetic -- strictly less work for identical output -- rather than on
the wall clock.

### The env side does not need work

At 256 environments and 8 threads, 4.23 ms per vector step without
demonstration starts and 4.63 ms with them at a 120-step episode cap. Real
episodes are 600 steps, so the builder's per-reset cost -- enumerating the
scene's layouts, 0.345 ms -- is about 2% of the rollout. The v2 action space is
*faster* to step than v1 (0.62 ms against 0.68 ms per 64-environment step),
so the entity-table ordering costs nothing.

## A fused kernel for the grid prologue (2026-09-18)

`fsim/patchify.py`. The extractor's first layer is a 4x4 stride-4 projection,
and reaching its input costs three full passes over a 104M-element tensor:
convert to bfloat16, scale by 1/255, unshuffle and permute and copy. One kernel
does all three, reading the 65x65 grid where it already lies:

| at batch 4096 | ms |
|---|---|
| convert + scale + unshuffle + permute | 7.93 |
| the kernel, strided read of the 65x65 | **1.94** |
| the kernel on a contiguous 64x64 copy | 1.92, plus 2.17 for the copy |

What it writes is **bit-identical** to what the portable path builds, at every
batch size tested and at the extremes (0, 1, 254, 255). The features that come
out of the extractor are not: feeding the *same* values to the matmul as a
contiguous tensor rather than a permuted view changes the result by up to 0.125
on logits of order 10, because cuBLAS accumulates it in a different order. That
is a library-version-sized difference, not a change of model, and the kernel is
opt-in (`FSIM_FUSED_GRID=1`) partly so that it never lands in the middle of a
comparison.

Measured on the policy step alone, interleaved within one process: 65.44 and
66.90 ms without, 55.65 and 57.68 ms with -- **14.4% faster**.

**The whole-trainer number is not yet trustworthy.** Four interleaved runs gave
21724, 17715, 12837 and 7196 update samples/s in that order: the kernel wins
within each round, but throughput falls monotonically across them, which is
this laptop thermally throttling after an hour of benchmarking rather than
anything about the kernel. It needs re-measuring on the desktop.

Two things that will not help, measured rather than assumed:

- **A bigger minibatch does nothing.** 1024 -> 54187 samples/s, 2048 -> 59792,
  4096 -> 62800, 8192 -> 62187, 16384 -> 61987. The GPU is saturated at 4096,
  which is what the training configuration already uses. Larger minibatches are
  free to try for *learning* reasons; they are not a throughput lever.
- **`torch.compile`** would be the obvious way to fuse the same three passes
  without writing CUDA, and it cannot run here: there is no working Triton on
  this platform.

## On the desktop, where the runs happen (2026-09-19, RTX 4060, 512 envs)

Interleaved and repeated, so a warming machine could not pass for a result.

| | overall steps/s | update s |
|---|---|---|
| epochs 2, kernel off | 26,804 / 26,798 | 0.907 |
| epochs 2, kernel on | 26,528 / 26,538 | 0.914 |
| epochs 1, kernel off | **41,904 / 41,905** | 0.467 |
| epochs 1, kernel on | 41,756 / 41,880 | 0.466 |

**`--epochs 1` is worth 1.56x**, which is the arithmetic: the update is 16
minibatch steps or 8, and the rollout does not move (103k steps/s either way).
Whether it *learns* as well per environment step is a separate question this
does not answer.

**The fused kernel is worth nothing here, and slightly less than nothing.**
26,804 -> 26,528, repeated. On the laptop's 3050 it took 14.4% off the policy
step in isolation, measured carefully and interleaved; on the 4060 that gain
does not survive into the trainer. The most likely reason is the one the
isolated benchmark could not see: the 4060 has half again the memory bandwidth,
so the three passes the kernel replaces are cheaper there to begin with, and
what is left does not pay for the extra launch.

It stays opt-in and stays off. The lesson is the one this file keeps
recording: a kernel measured on one card, in isolation, against a microbenchmark
is not a speedup until it is measured in the trainer on the card that runs it.
