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

A sharper test needs a bigger conjunction and still no placement: more
machines in the entity table and several item types in the inventory, so the
target, item and amount each carry real choice. That is a new scene family
rather than a new task, but it has to be added to FactorioRL first and the
holdout re-frozen, as `construct_smelting_line` 1.2.0 was.
