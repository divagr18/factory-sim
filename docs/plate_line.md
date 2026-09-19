# plate_line

FactorioRL's `plate_line` 1.2.0, and the simulator's third task. The first
with no placement in it: the drill and the furnace arrive aligned and empty,
the agent carries 120 coal, and it has to reach each machine and fuel it.
Success is 30 iron plates inside 400 decisions and 24,000 ticks.

Families are `commissioning` (train), `commissioning_far` (val, start 12-16
tiles out instead of 5-9) and `commissioning_walled` (test, a three-tile
screen to walk around). The holdout is **structural** rather than a reshaped
patch, which makes it a stronger test than the construction tasks' holdouts.

## It is solved

20M steps, 512 environments, `--shaping both --action-space v2
--autoregressive`, no demonstration starts.

| seed | train sampled | held-out sampled | held-out eps-greedy | held-out greedy |
|---|---|---|---|---|
| s1 | 0.996 | 0.982 | 0.926 | 0.914 |
| s2 | 0.998 | 1.000 | 1.000 | 1.000 |
| s3 | 0.996 | 0.990 | 0.650 | 0.236 |

Read the sampled column. It is 0.982, 1.000, 0.990 -- the task is solved, and
consistently so.

**Greedy is bimodal and must not be quoted**: 0.914, 1.000, 0.236. Two seeds
made it look like this task was the exception where an argmax policy works,
against 3.9% on `construct_smelting_line`; the third seed says otherwise. The
standing rule in this project applies here as everywhere else -- read the
seeds, never the mean, and report sampled and epsilon-greedy rather than pure
argmax.

One thing is genuinely unusual against the construction tasks:

- **No demonstration starts were needed.** The scripted builder in
  `fsim/expert.py` builds a line and this task is handed one, so it does not
  apply. The potential carries the run by itself: phi starts at 0.59 against
  a 0.9 ceiling, and the whole 0.3 of headroom is the commissioning -- fuel
  the drill, fuel the furnace, get ore into it.

## What it cannot answer

It was added to separate two readings of the autoregressive result (ordered
arguments took held-out success from 60% to 87% on `construct_smelting_line`):
does ordering help with **conjunctions**, or with **placement**? This task has
no placement, so an ordered-versus-independent comparison should isolate it.

It does not, because the task is solved in **0.13M steps** -- the second
update -- against 20M+ for `construct_smelting_line`. The conjunction is too
small: the agent sees two machines and carries one item, so
`give_to(target, item, count)` has roughly six to ten viable combinations
against the other task's 3,960.

Run anyway, three seeds per arm, to say so with numbers rather than a guess:

| arm | steps to 99% train | held-out sampled | held-out eps-greedy |
|---|---|---|---|
| ordered arguments | 0.13 / 0.13 / 0.16M | 0.982 / 1.000 / 0.990 | 0.926 / 1.000 / 0.650 |
| independent arguments | 0.13 / 0.16 / 0.13M | 0.961 / 0.988 / 0.996 | 0.947 / 0.559 / 0.986 |

Indistinguishable on every measure. This does **not** undo the
`construct_smelting_line` result -- ordered arguments took held-out success
there from 60% to 87% over four seeds each, with the give action's reward
component pinned at exactly 0.00 for 7M steps beforehand. It says the effect
disappears when the conjunction is small, which is what the mechanism
predicts and why this task could not test it.

## commissioning_crowded (1.3.0)

The sharper test, added rather than left as a note. Four decoy machines join
the entity table and two decoy items join the inventory, so
`give(target, item, amount)` has 6 x 3 x 4 = 72 combinations against 8. Coal
drops to 50, because with 120 the agent can simply fuel everything and the
choice of target stops mattering. The decoy drills sit off the ore and the
decoy furnaces have nothing feeding them, so none can produce a plate however
it is fuelled: the family enlarges the decision and changes nothing about
what the task rewards. No placement is added.

It is a **train** family, so the 100 held-out scenes are untouched -- their
digests hash to `e8e4b86403c5f11e496a82d11a6fba6f` before and after -- and a
1.2.0 held-out rate and a 1.3.0 one describe the same scenes.

Not yet trained on. The comparison to run is the same one, ordered against
independent, on a task where the conjunction is real.
