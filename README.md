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

Next is M4: the observation encoder, action masks, reward and goal vector in C,
checked against FactorioRL's fixtures.

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

## Build and test

```
uv sync
uv run python build.py          # compiles csrc/ into fsim/_fsim (MSVC or gcc/clang)
uv run pytest                   # parity against tests/golden
uv run python bench/bench.py
```

`tools/sync_golden.py --from ../FactorioRL` refreshes the golden traces from a
FactorioRL checkout, checking every trace's hash.

See NOTICE for what this project does and does not contain.
