# factory-sim

A fast, tick-exact simulator of Factorio's early game: train agents at
simulator speed, then check them against the real game.

```
pip install factory-sim
```

It simulates a small slice of the game exactly: a character walking, reaching
and hand-mining; burner mining drills; stone furnaces; fuel; items on the
ground; walls and water. Every rule and number comes from measurements of
Factorio 2.0.60, and the simulator is checked decision by decision, and tick
by tick, against traces recorded from the running game by
[FactorioGym](https://github.com/divagr18/FactorioGym). A mechanic is added
here only after it has been measured there.

The point is speed. The real game runs about 260 decisions a second across
eight workers. This runs 950,000 on 16 threads, and 530,000 through the full
RL path (observation tensors, masks, rewards) on eight. The 40 million
decisions a PPO run below uses would be about two days of nonstop play on the
game.

## Two ways to solve a task

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
is now a program that does nothing.

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
```

Run program search. It needs an OpenAI-compatible endpoint, set in a JSON
file with `base_url`, `api_key` and `model`, plus optional `price` and
`max_usd` for a hard spending cap:

```
FSIM_LLM_CONFIG=provider.json python -m evolve.run --name demo --budget-candidates 150 --max-usd 1
```

## Tasks

| Task | The agent has to |
|---|---|
| `construct_smelting_line` | find an ore patch it can only partly see, place a drill on ore and a furnace under the drill's drop point, fuel both, and get 10 machine-made plates in a verification minute |
| `build_line` | build a drill-and-furnace line and keep it running: success reads the output of a window after construction has settled, so a line that ran once and died fails |
| `plate_line` | commission a line that is already built by fuelling the right machines; one family adds decoy machines and items |

Each task has training families and a held-out family (a patch behind a wall,
a narrow strip), drawn exactly as FactorioGym draws them, so a seed names the
same scene in both projects.

## What "tick-exact" means

- All 12 golden scenarios match the game at every decision, and all 8
  tick-level traces match on every tick.
- Doubles match to within 1e-12, because the game prints some with an
  imprecise last digit.
- Ore under a drill is compared as a total over its four tiles: the order a
  drill visits its tiles follows the engine's internal entity order, which could
  not be reduced to a rule.
- The RL layer matches FactorioGym bit for bit: every encoded observation hashes
  identically, and so do the mask, reward, termination and success.

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
  version = {0.1.1},
  url     = {https://github.com/divagr18/factory-sim},
  license = {Apache-2.0}
}
```

## License

Apache-2.0; see [LICENSE](LICENSE). Factorio is a game and trademark of Wube
Software Ltd. This project is independent, and is not affiliated with or
endorsed by Wube. It contains no game code or assets; see [NOTICE](NOTICE).
