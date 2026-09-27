# factory-sim

A tick-exact simulator of Factorio's early game, written in C with a Python API.
It is about 3,600× faster than the real game, so agents can train in it and be
tested on the real game afterwards.

```
pip install factory-sim
```

Prebuilt wheels for Python 3.11–3.13 on Linux, Windows and macOS.

## What it simulates

- A character that walks, reaches, hand-mines and carries an inventory
- Burner mining drills, stone furnaces and fuel
- Transport belts, burner inserters and wooden chests
- Items on the ground, walls and water

Every rule and number was measured in Factorio 2.0.60. The simulator is checked
against traces recorded from the real game, at every decision and, for belts
and inserters, at every tick. The measurements live in
[FactorioGym](https://github.com/divagr18/FactorioGym), which drives the real
game through a mod.

## Quick start

```python
import gymnasium as gym
import fsim.gym_env  # registers the environments

env = gym.make("fsim/BuildLine-v0", split="test")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Install the Gymnasium extra with `pip install factory-sim[gym]`. PufferLib,
OpenEnv and Prime Intellect's verifiers format are also supported; see
[docs/interfaces.md](https://github.com/divagr18/factory-sim/blob/main/docs/interfaces.md).

## Speed

| | Decisions per second |
|---|---|
| Real game, 8 workers | ~260 |
| factory-sim, 16 threads | ~950,000 |
| factory-sim, full RL path (observations, masks, rewards), 8 threads | ~530,000 |

A 40-million-decision PPO run takes about two days of nonstop play in the real
game and a couple of hours here.

## Tasks

| Task | Goal |
|---|---|
| `construct_smelting_line` | Find an ore patch, place a drill and a furnace, fuel both, and make 10 iron plates |
| `build_line` | Build a drill-and-furnace line that keeps running after construction |
| `plate_line` | Fuel the right machines in a line that is already built |
| `belt_smelting` | Walk between an iron patch, a coal patch and a chest, and build drills, furnaces, belts and inserters that deliver 150 plates to the chest |

Each task has training layouts and held-out layouts. A seed produces the same
scene here and in the real game.

## Results

Agents trained or searched in the simulator, then run on held-out layouts:

| Method | Task | Simulator | Real game |
|---|---|---|---|
| PPO | `construct_smelting_line` | 92.0% | 90.6% |
| PPO | `build_line` | 99.2% | 90.6% |
| Evolved programs | `construct_smelting_line` | 99.6% | 99.3% |
| Qwen3.5-9B, Evolve & Reinforce | `construct_smelting_line` | 71.9% | not yet run |

The Qwen3.5-9B model was trained with GRPO on the
[`factorio-build`](https://app.primeintellect.ai/dashboard/environments/divagr/factorio-build)
environment. Its general planning stays close to the base model's (PlanBench
Blocksworld 46.4 vs 53.2). Models are in the
[collection on Hugging Face](https://huggingface.co/collections/divagr1925/factoriogym-models-wip).

## What's in the repo

| Path | Contents |
|---|---|
| `fsim/` | The Python API, environments and scene generators |
| `csrc/` | The C simulator |
| `train.py` | Masked PPO in one file |
| `evolve/` | LLM-guided search over `def build(world):` programs |
| `integrations/verifiers/` | The `factorio-build` and `factorio-play` environments for the Prime Environments Hub |
| `recipes/grpo-factorio-build/` | SFT and GRPO recipe for Qwen3.5 with prime-rl |

## Build from source

```
uv sync                        # builds the C extension
uv run pytest                  # includes parity tests against the real game
uv run python bench/bench.py   # throughput
```

## Citing

```bibtex
@software{agrawal2026factorysim,
  author  = {Agrawal, Divyansh},
  title   = {{factory-sim}: a fast, tick-exact simulator of an early-game factory},
  year    = {2026},
  version = {0.2.0},
  url     = {https://github.com/divagr18/factory-sim}
}
```

## License

Apache-2.0. Factorio is a game and trademark of Wube Software Ltd. This project
is independent, not affiliated with or endorsed by Wube, and contains no game
code or assets. See [NOTICE](https://github.com/divagr18/factory-sim/blob/main/NOTICE).
