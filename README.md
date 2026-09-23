# factory-sim

A fast C simulator of a small slice of Factorio's early game: a character
walking, reaching and hand-mining; burner mining drills; stone furnaces; fuel;
ground item piles. It exists to train reinforcement-learning policies quickly,
with the real game kept as the verifier.

It is checked, decision by decision and tick by tick, against golden traces
recorded on Factorio 2.0.60 by [FactorioRL](https://github.com/divagr18/FactorioRL).
A mechanic is added here only after it has been measured there.

## Status

M3, the simulator core, is done:

- All 12 golden scenarios match the real game at every decision, in both
  free-running and one-step sync.
- All 8 tick-level traces match on every tick.
- A single environment runs 173k decisions/s, and 8 threads run 850k in total
  (`bench/RESULTS.md`).

M4, the RL contract, is done too. `fsim.rl.RlEnv` speaks FactorioRL's
`parameterized-v1` action space (`MultiDiscrete[22, 33, 122, 5, 15, 4]`) and its
`local-v2` tensor observation for `construct_smelting_line` and `build_line`:

- It covers the encoder, argument domains, the action mask, vector decoding,
  the goal vector, rewards, termination, and the verification window.
- Driven by each golden trace's recorded action vectors, every decision
  matches FactorioRL:
  - each encoded tensor hashes identically, bit for bit;
  - the mask, goal, reward, reward components, termination, truncation,
    success and decode failures all match.
- The whole RL path runs 60k decisions/s on one environment and 382k on
  eight threads.

Next is M5: a PPO learner trained here, then evaluated on the real game through
FactorioRL.

What it simulates today:

- **Character**: 4-way walking, with collision against walls, machines and
  water. Reach and build distance. Hand-mining ore and picking machines back up.
- **Burner mining drill**: fuel and energy buffer, mining progress, drop
  position per facing, delivery into a furnace or onto the ground, jams.
- **Stone furnace**: iron smelting, fuel, the 54-ore source slot, idle and
  no-fuel states.
- **Transfers**: clamped transfers, with the same choice of slot and the same
  refusals as the game.
- **Bookkeeping**:
  - handles, running actions and the event log, as FactorioRL's mod keeps them;
  - production statistics, and the `steam-power` trigger 24 ticks after the
    50th plate.

How parity is judged:

- Doubles are compared to within 1e-12. The game prints some doubles with an
  imprecise last digit, and one progress bar is matched to about one part in
  10^14.
- Ore under a drill is compared as a total over the drill's four tiles. The
  order a drill visits its tiles follows the engine's internal entity order,
  which could not be reduced to a rule, and the encoder cannot see the
  difference.
- Everything else must match exactly.

## Install

```
pip install git+https://github.com/divagr18/factory-sim   # compiles csrc/: needs a C compiler
```

This installs the simulator (`fsim`) and the program-search loop (`evolve`); the
benchmark map ships inside the package. Extras: `[gym]` for the Gymnasium
adapter, `[puffer]` for PufferLib's (see `docs/interfaces.md`).

## Build and test

```
uv sync                         # installs the checkout editable, compiling csrc/
uv run python build.py          # rebuilds fsim/_fsim in place after a C change
uv run pytest                   # parity against tests/golden
uv run python bench/bench.py
```

`tools/sync_golden.py --from ../FactorioRL` refreshes the golden traces from a
FactorioRL checkout, checking every trace's hash.

See NOTICE for what this project does and does not contain.
