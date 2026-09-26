# factory-sim

A fast, tick-exact simulator of Factorio's early game: train agents at
simulator speed, then check them against the real game.

```
pip install factory-sim
```

It simulates a small slice of the game exactly: a character walking, reaching
and hand-mining; burner mining drills; stone furnaces; transport belts,
burner inserters and wooden chests; fuel; items on the ground; walls and
water. Every rule and number comes from measurements of Factorio 2.0.60, and
the simulator is checked decision by decision, and tick by tick, against
traces recorded from the running game by
[FactorioGym](https://github.com/divagr18/FactorioGym). A mechanic is added
here only after it has been measured there.

The point is speed. The real game runs about 260 decisions a second across
eight workers. This runs 950,000 on 16 threads, and 530,000 through the full
RL path (observation tensors, masks, rewards) on eight. The 40 million
decisions a PPO run below uses would be about two days of nonstop play on the
game. A belt-heavy scene, 338 belts with about 180 items moving, runs at about
200,000 ticks a second on one laptop core.

## Three ways to solve a task

**Reinforcement learning.** `train.py` is masked PPO in one file: CleanRL's
layout at PufferLib's scale, with the choices and their sources in its
docstring. Policies trained here are evaluated on the real game with
FactorioGym's `tools/sim_transfer.py`. On held-out layouts they score:

| Task | Simulator | Factorio |
|---|---|---|
| `construct_smelting_line` | 92.0% | 90.6% |
| `build_line` | 99.2% | 90.6% |

This is one checkpoint per task, evaluated over 512 simulator episodes and 32
engine episodes. FactorioGym's README has the per-split numbers and the
evidence files.

**Program search.** `evolve/` has a language model write short Python
`def build(world):` programs, scores them in the simulator, and breeds the
better ones: islands, tournament selection, a genealogy of every attempt. A
program sees exactly what a trained policy sees, and every call it makes is one
decision of the same action space. It runs in a static sandbox with no imports
and no reflection. Selection uses a validation set, and the frozen held-out
scenes are touched only for reporting.

On `construct_smelting_line`, 1,000 held-out scenes that no run or analysis
had seen:

| | Held-out success | Simulated decisions |
|---|---|---|
| Evolved programs, 4 runs from a program that does nothing | 0.994, 1.000, 0.990, 0.999 | 0.46–0.69M to pass 0.9 |
| PPO, 4 runs, sampled actions | 0.928, 0.858, 0.837, 0.804 | 40M each |

Sampling is PPO's best mode: played greedily, the same checkpoints score
0.01–0.19. The model knows things PPO has to learn, pathfinding for a start,
so fewer simulated decisions is not less compute. Seeding the search with a
hand-written builder made it generalise *worse*: it kept the builder's
wall-blind walker and patched individual validation scenes. The default seed
is now a program that does nothing. Played on the real game, the four runs'
programs succeed in 397 of 400 held-out episodes (99.3%), and the simulator
agrees on every episode.

**Training a language model.** The same program task is an RL environment for
language models: [`factorio-build`](integrations/verifiers/factorio_build/README.md)
on Prime Intellect's Environments Hub (`prime env install
divagr/factorio-build`). The model writes one program per row, and the reward
is the fraction of the row's unseen scenes it solves.
[`recipes/grpo-factorio-build`](recipes/grpo-factorio-build/README.md) trains
Qwen3.5-9B on it with prime-rl. Scored in the simulator on
`construct_smelting_line`, with PlanBench Blocksworld to measure what the
training costs the model's general planning:

| Qwen3.5-9B | `val` (sim) | `holdout` (sim) | PlanBench generation | PlanBench execution |
|---|---|---|---|---|
| base | 2.1% | 1.6% | 53.2 | 39.8 |
| SFT on 800 evolved programs + GRPO | 85.9% | 84.0% | 2.8 | 3.0 |
| Evolve & Reinforce, round 1 | 85.6% | 71.9% | 46.4 | 27.8 |

Evolve & Reinforce builds the warm start from the model's own explanations of
the evolved programs, with replay of its answers to general prompts, then runs
GRPO. It keeps most of the planning ability that SFT on bare programs
destroyed, and gives up 12 points of holdout. A sample of the SFT + GRPO
model's holdout programs succeeds in 136
of 160 episodes on the real game, again in agreement with the simulator on
every episode. Models:
[`qwen3.5-9b-factorio-build-er-r1`](https://huggingface.co/divagr1925/qwen3.5-9b-factorio-build-er-r1)
and [`-er-r1-sft`](https://huggingface.co/divagr1925/qwen3.5-9b-factorio-build-er-r1-sft);
all of them in the [models collection].

## Use it

As a Gymnasium environment (`pip install factory-sim[gym]`):

```python
import gymnasium as gym
import fsim.gym_env  # registers the ids

env = gym.make("fsim/BuildLine-v0", split="test")   # the held-out family
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

The ids are `fsim/ConstructSmeltingLine-v0`, `fsim/BuildLine-v0` and
`fsim/PlateLine-v0`. The same environment is also available through
**PufferLib**, **OpenEnv** and the **verifiers** / Prime Environments Hub
format; [docs/interfaces.md](docs/interfaces.md) covers each.

Train PPO (`uv sync --group train` for PyTorch):

```
python train.py --run sparse-s1 --seed 1 --steps 20000000
python train.py --run belt-s1 --task belt_smelting    # the v3 profile, its own budget
```

Run program search. It needs an OpenAI-compatible endpoint, set in a JSON
file with `base_url`, `api_key` and `model`, plus optional `price` and
`max_usd` for a hard spending cap:

```
FSIM_LLM_CONFIG=provider.json python -m evolve.run --name demo --budget-candidates 150 --max-usd 1
```

`--task belt_smelting` searches for belt-line programs instead.

## Tasks

| Task | The agent has to |
|---|---|
| `construct_smelting_line` | find an ore patch it can only partly see, place a drill on ore and a furnace under the drill's drop point, fuel both, and get 10 machine-made plates in a verification minute |
| `build_line` | build a drill-and-furnace line and keep it running: success reads the output of a window after construction has settled, so a line that ran once and died fails |
| `plate_line` | commission a line that is already built by fuelling the right machines; one family adds decoy machines and items |
| `belt_smelting` | walk between an iron patch, a coal patch and an output chest that no standing spot reaches two of, and build drills, furnaces, belts and burner inserters that deliver 150 iron plates into the chest in a ten-minute window. The 20 starting coal runs a line for about 79 plates, so the line also needs coal from the coal patch |

Each task has training families and held-out families (a patch behind a wall,
a narrow strip, a chest at the far end of the belt budget), drawn exactly as
FactorioGym draws them, so a seed names the same scene in both projects.
`belt_smelting`'s scripted reference (`fsim/belt_expert.py`, a port of
FactorioGym's reference solver) solves 599 of its first 600 scenes; the one
failure has its whole coal patch on water.

## Action and observation profiles

The first three tasks use `MultiDiscrete([22, 33, 122, 5, 15, 4])`: operation,
target, placement tile, direction, item and amount, in two variants (v1, and
v2, where the target is a table row and the placement a fixed tile).
`belt_smelting` uses v3, `MultiDiscrete([25, 97, 226, 5, 19, 4])`:

- 25 operations: v2's 22 plus hand-mining a resource tile (`mine_tile`),
  `take_fuel` and `finish`, which ends the build phase and runs the
  verification window at once;
- a mask per operation, legal where the game accepts: entities within the
  engine's reach (10 tiles from the character to the entity's box), resource
  tiles within 2.7 tiles, and placements within build distance;
- the observation adds each belt's lane counts and turn, each inserter's hand
  and its pickup and drop points, a drill's drop point, the task's public
  markers, and the free inventory slots.

Programs on `belt_smelting` hold `WorldV3`, which speaks the same profile.

## What "tick-exact" means

- All 24 golden scenarios match the game at every decision, and all 19
  tick-level traces match on every tick. Twelve of the scenarios exercise what
  `belt_smelting` needs: belts, inserters, chests, hand-mining, `take_fuel`
  and `finish`.
- The belt and inserter rules come from five logistics probes and two
  hand-mining probes on the engine: belt-line segments and their merge delays,
  turns, sideloads, loops, and inserters picking from moving belts. FactorioGym's
  [`docs/sim-logistics.md`](https://github.com/divagr18/FactorioGym/blob/main/docs/sim-logistics.md)
  records every measurement.
- Doubles match to within 1e-12, because the game prints some with an
  imprecise last digit.
- Ore under a drill is compared as a total over its four tiles: the order a
  drill visits its tiles follows the engine's internal entity order, which could
  not be reduced to a rule.
- The RL layer matches FactorioGym bit for bit. Under v2 every encoded
  observation hashes identically, and so do the mask, reward, termination and
  success. Under v3 the observation tensors and masks do, at every decision
  of every golden scenario.

## Build from source

```
uv sync                        # installs the checkout editable, compiling csrc/
uv run python build_fsim.py    # rebuilds fsim/_fsim in place after a C change
uv run pytest                  # includes parity against tests/golden
uv run python bench/bench.py   # throughput; results in bench/RESULTS.md
```

`tools/sync_golden.py --from ../FactorioGym` refreshes the golden traces from
a FactorioGym checkout, checking every trace's hash.

## Citing

If you use factory-sim in academic work, please cite it. GitHub's "Cite this
repository" button reads [CITATION.cff](CITATION.cff); in BibTeX:

```bibtex
@software{agrawal2026factorysim,
  author  = {Agrawal, Divyansh},
  title   = {{factory-sim}: a fast, tick-exact simulator of an early-game factory},
  year    = {2026},
  version = {0.2.0},
  url     = {https://github.com/divagr18/factory-sim},
  license = {Apache-2.0}
}
```

## License

Apache-2.0; see [LICENSE](LICENSE). Factorio is a game and trademark of Wube
Software Ltd. This project is independent, and is not affiliated with or
endorsed by Wube. It contains no game code or assets; see [NOTICE](NOTICE).
