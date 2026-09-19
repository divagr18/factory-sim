# Shaping construct_smelting_line

`construct_smelting_line` (FactorioRL 1.1.1) pays one number: the action-locked
verification score, after the 600-decision budget runs out. This note records
how training on it was settled, and why the task itself was left alone.

## 1. Sparse: no gradient

A uniform random policy over the masked action space was run for 9,000
episodes (5.4M decisions): **no episode produced a single verified plate**.
With the task's only reward identically zero, PPO's policy gradient is
identically zero apart from the entropy bonus. More samples do not change
that. FactorioRL's own reward audit found the same failure on `repair_belt`
("190 episodes of zero signal and 1,900 episodes of zero signal are the same
thing to a policy gradient").

## 2. Potential-based shaping: right idea, fades by design

The textbook fix is potential-based shaping, F = gamma * phi(s') - phi(s)
(Ng, Harada & Russell 1999; *Algorithms for Decision Making* eq. 17.15). It
cannot change which policy is optimal, and it is exempt from FactorioRL's
plateau audit for exactly that reason. The potential is
`fsim_rl_potential`, which is bit for bit FactorioRL's
`tasks.potentials.line_potential`:

| term | weight |
|---|---|
| approach: max(0, 1 - distance to patch / 64) | 0.1 |
| a burner drill is visible | 0.2 |
| a visible furnace holds a drill's drop point (a line) | 0.3 |
| the best line's drill has fuel / its furnace has fuel / the furnace holds ore or plates | 0.1 each |

phi(terminal) = 0 (Grzes 2017). FactorioRL scores the budget's last decision
as two transitions (the decision, then the verification window into the
absorbing state), and the simulator does the same.

Measured (`runs/pilot-potential-s1`, seed 1): lines appeared in **11.5%** of
episodes by 1.6M steps. By 2.4M steps they had fallen back to **2.9%**, while
the value loss went to about 2e-4.

That is the theory working, not a bug. Shaping with phi is equivalent to
initialising the value function with phi (Wiewiora 2003; Sutton & Barto
2nd ed. p. 404). Take a fixed-horizon episode with phi(terminal) = 0 and no
success ever observed. Every policy then has the same discounted shaped
return, -phi(s0).

Early on, the critic has not learned that, so the immediate gamma*phi' - phi
terms push the policy towards building. Once the critic converges on
V = -phi, every advantage is zero, and the push goes with it. Potential
shaping speeds up learning towards a reward the agent can already reach. It
cannot hold the policy at a sub-goal on the way to a reward the agent has
never seen.

## 3. Progress shaping: persistent, and outside the task

`--shaping progress` pays the rise of phi's running maximum, at half weight,
up to 0.45 per episode. This is FactorioRL's `HIGH_WATER` kind, with the cap
that kind requires. It does not telescope, so it keeps paying for progress
the policy has not yet secured. Because it is a high-water mark, a line can
be paid for only once, however often it is torn down and rebuilt.

It does change the objective: an episode can earn 0.45 without succeeding.
FactorioRL's reward audit calls that a plateau. The audit is right to: with
no step cost in this task, building an unfuelled line and idling is a
positive-return strategy that is not success. Success is still worth 1.0 on
top of the plateau, so the plateau is a waypoint on the way to success rather
than a rival to it.

This is why **the task stays 1.1.1**. Its reward is also the agentic bridge's
terminal score and the frozen holdout's contract, and adding a pre-success
term there would change both. So progress shaping is a simulator training
option, like FactorioRL's exploring starts. Evaluation, in the simulator and
on the engine, uses the task's own success predicate.

Measured, seed 1, 256 environments. These runs predate the fix that stopped a
drill feeding another drill from counting as a line.

| run | steps | lines built | success (sampled) |
|---|---|---|---|
| `progress-s1` | 8M | 30% | 0.7% (training episodes) |
| `progress-s1` | 12M | 97% of evaluation episodes | 0 |
| `progress-demo-s1` (`--demo-starts 0.5`) | 20M | 12-19% of scene starts | train split 1.8% (95% CI 0.9-3.3%), greedy 0; held-out family 0 |

Progress shaping does what potential shaping could not: the build keeps
being learned, up to near-certain line construction. It does not deliver
success on its own.

## 4. What the curves say is hard

`tools/diagnose.py` on the 12M `progress-s1` checkpoint (64 evaluation
episodes):
- 98% placed a drill and 97% built a line, but only 5% ever had both machines
  fuelled.
- At step 300, 1 in 32 episodes still had a line. The policy builds, banks
  the one-time bonus, and then breaks the line, most often with
  `rotate_reverse` (16% of its actions at 8M), which moves the drop point.
  The high-water bonus has no price for that; only success does.

Demonstration starts (`fsim/expert.py`) give the policy the missing
experience directly. Episodes that begin with the line already built and
fuelled succeed 25% of the time at 10M steps, which means the other 75%
break a working line in under 600 decisions. What remains hard is not
finding the line but **leaving it alone**, and two properties of the
`parameterized-v1` contract make the sub-steps harder than they look:

- **A `target` index is a position in the sweep order, not a row of the
  entity table.** The policy sees entities sorted by distance, but picks
  targets by the order the engine returned them in. Fuelling "the drill"
  means learning that mapping.
- **A `placement` index skips occupied tiles.** Index *k* names a different
  tile once something is built nearby, so "the tile below the drill" has no
  fixed index.

Both are FactorioRL contract decisions, frozen by `build_line`'s published
numbers. A pointer-style target head over the entity table, and a fixed
121-slot placement grid with occupancy in the mask, are the changes that
would make them learnable. They would be a `parameterized-v2`.

## What the schedule of demonstration starts is worth

`parameterized-v2` and the combined shaping got 14.8% sampled success on the
training families at 20M steps, drawing each episode's demonstration start
uniformly from the five stages of the build. Replacing that draw with
Backplay's sliding window (Resnick et al. 2018) — a start sampled from `U[lo,
hi]` decisions back from the *end* of the demonstration, with the window
sliding backwards on a fixed schedule until, at 65% of training, every episode
starts from the scene — is worth about five times that:

| 40M steps, `--shaping both --action-space v2 --demo-starts 0.5` | 5M | 10M | 20M | 30M | 40M | final |
|---|---|---|---|---|---|---|
| Backplay window, seed 1 | 13.7% | 11.7% | 52.0% | 79.7% | 77.3% | **77.5%** |
| Backplay window, seed 2 | 0.0% | 2.3% | 39.1% | 49.2% | 61.3% | **68.6%** |
| uniform stages, seed 2 (20M run) | — | — | 14.8% | — | — | — |

Backplay's own explanation fits what the curves show: a start one decision from
the end is a task the policy can solve by chance, and every window the schedule
opens is only one decision harder than the one the policy has already learned.
The uniform draw spends a fifth of its episodes on a stage the policy cannot
yet reach the end of, and the reward from those is noise.

Three things are wrong with the resulting policy, and they are the same thing.

**It does not generalise to an unseen family.** `obstructed_patch`, the
held-out family, is 0.0% and 1.4% for the two seeds — below the 3.7-7.2% that
the weaker uniform runs reached. The builder in `fsim/expert.py` can only
demonstrate the two training families; a policy that spends half its episodes
resuming those demonstrations learns their geometry, not the task.

**Its greedy policy is worthless.** Sampled 77.5%, greedy 0.4%. The
environment is deterministic, so an argmax policy that enters a loop never
leaves it. Success comes from sampling, over 600 decisions, from a
distribution that is merely well shaped.

**Its entropy rises as it learns.** 3.32 nats at 5M, 3.60 at 10M, 3.56 at 20M,
3.92 at 30M, 4.03 at the end, with the update KL falling to zero. The policy
stops moving before it sharpens. `--ent 0.01` is the suspect, and this is the
same objection VPT (Baker et al. 2022) raises against an entropy bonus on a
sparse long-horizon task: it is exploration pressure that never expires, and
they replace it outright with a KL term to a frozen behaviour-cloned prior,
decayed 0.9995 per iteration from 0.2.

## The entropy bonus was not the problem

The policy's entropy rising as it learned looked like the bonus outweighing a
shrinking advantage, so the coefficient was swept at 40M steps, everything else
held at the Backplay configuration:

| `--ent` | train sampled | train argmax | held out |
|---|---|---|---|
| 0.01 (the default) | **77.5%** | 0.4% | 0.0% |
| 0.003 | 56.4% | 2.9% | 0.0% |
| 0.001 | 63.5% | 0.0% | 1.8% |
| 0.0 | 37.9% | 0.2% | 0.0% |

Lowering it costs success and buys almost no sharpness: the argmax policy stays
near zero at every value, because what breaks it is cycling in a deterministic
task, not the width of the distribution. The default stays 0.01, and the
epsilon in the evaluation is what makes a near-deterministic policy readable.

## What PufferLib's sparse settings do here

PufferLib 5.0's tuned configs for sparse, long-horizon tasks differ from this
trainer's defaults in four places: rollout 128 rather than 64, GAE lambda 0.90
rather than 0.95, value loss weight 2.0 rather than 0.5, and no advantage
normalisation. Applied as a group, on top of the Backplay schedule, they do not
train at all: 0.0% at every checkpoint of two seeds, `line_built` 0.0, the
potential stuck at its initial 0.25, and a per-update KL of 1e-4 against the
3e-3 of a run that is learning. The policy is frozen.

Advantage normalisation is not the cause on its own — an arm that dropped only
that flag froze the same way, and an arm that kept it and took the other three
also froze. An arm taking the value loss weight alone -- 2.0 on a trunk shared with the
policy head -- froze the same way, which identifies it. The flags stay in the trainer, all defaulting to
this trainer's own values.

## What the policy had actually learned

Rolling the 77.5% policy on the held-out family shows it walking to the patch
and then not building: it reaches the patch on 94-100% of episodes and places
something on 4-9% of them. `place` is 16% of its decisions on a training scene
and 1.7% on a held-out one. The held-out family changes two things at once, so
each was rolled alone, 128 episodes each:

| test-split scenes | success | reached the patch | built a drill | `place` share |
|---|---|---|---|---|
| neither change (7x7 patch, no wall) | **83.6%** | 86.7% | 86.7% | 13.3% |
| the narrow patch alone | 0.8% | 84.4% | 84.4% | 8.6% |
| the wall alone | 0.0% | 93.8% | 8.6% | 1.2% |
| both, the real held-out family | 0.0% | 100% | 3.9% | 1.7% |

**It generalises across unseen scene seeds perfectly well.** 83.6% on seeds it
never trained on. What breaks it is the geometry, and the wall breaks it
completely -- though the wall is three tiles from the ore and never stands on
the canonical build site. What it does is put an entity on the map, and until
then the policy had never seen one before placing its own.

The cause is in `fsim/expert.py`: the builder stood in one place and built one
arrangement, and half of every episode resumed one of its demonstrations. The
policy learned that pose, not the task. Backplay did not make generalisation
worse in any interesting sense -- it made the policy far better at copying a
demonstration, and there was only one demonstration to copy.

Three things follow, and they are the fixes.

**The builder builds anywhere now.** `expert.layouts` enumerates every drill
anchor the ore admits and all four turns of the layout about the drill's
centre -- 144 on a training patch, each of which builds a line that verifies --
and one is drawn per episode.

**There are two more training families.** `varied_patch` draws the patch's
dimensions and position; `cluttered_patch` scatters short walls in the ring
just off the ore, which is the only way the policy meets an entity it did not
build. `obstructed_patch` is untouched, and neither training family reproduces
it, so the held-out family becomes a test of the two variations together.
Cobbe et al. (2019) found agents overfitting CoinRun with 16,000 training
levels; this task had two families and four discrete offsets.

**The evaluation reports an epsilon.** The task is deterministic, so an argmax
policy that enters a cycle never leaves it, and this one emits a single action
for the last 200 decisions of an episode -- 1.1 distinct actions, against 71.4
when sampling. Measured on the same checkpoint: sampled 78.9%, argmax 0.0%,
epsilon 0.05 **28.1%**, epsilon 0.15 **45.3%**. Mnih et al. (2015) evaluated
Atari with an epsilon of 0.05 to prevent exactly this. `train.py` now reports
sampled, epsilon-greedy and argmax, and names the epsilon.

The policy is also flat -- 0.26 probability on its own modal operation -- which
is the entropy bonus still paying out once advantages shrink. `--prior` offers
VPT's alternative: clone the builder (`tools/behaviour_clone.py`), then train
with `rho * KL(prior, policy)`, rho decayed 0.9995 an update from 0.2, and no
entropy bonus at all.

## Drawing the target first

Fuelling the furnace needs an operation, a target, an item and an amount to be
right at once: one action in 22 x 3 x 15 x 4. Drawn independently of each
other, a uniform policy finds it in 15% of six-hundred-step episodes, and a
policy that has committed to anything else finds it almost never. Measured:
`back0`, where the builder finishes the line and the policy need only not break
it, reached 1.00 by four million steps while `back1`, one decision of the
policy's own, sat at 0.00 for seven million.

`--autoregressive` draws the target and then scores direction, item and amount
conditioned on the row it named -- the entity's own embedding, not its index.
Four seeds each, everything else identical, gated schedule:

| held out | s1 | s2 | s3 | s4 | mean | spread |
|---|---|---|---|---|---|---|
| independent arguments | 66.0% | 73.4% | 24.6% | 75.8% | 60.0% | 51.2 |
| **drawn in order** | **92.0%** | **85.5%** | **87.7%** | **82.0%** | **86.8%** | **10.0** |

It moves the mean by 27 points and takes the spread from 51 points to 10: the
worst autoregressive seed beats the best independent one. On the way there,
`back1` reaches 0.39 at 4.1M steps instead of 6.5M and settles at 0.82 rather
than 0.64.

This is AlphaStar's arrangement -- "the action is sampled first, and then
required parameters are sampled one by one from distributions conditioned on
the selected action and previously sampled action parameters" -- and the shape
of action Conditional Action Trees (arXiv:2104.07294) is about. FactorioRL's
`parameterized.py` named it as the fix and called it blocked by stock sb3;
this trainer is not stock sb3, and the blocker had not been true for some time.

What it does not do is fix the masks. The item dimension is still a union over
everything any operation could name, because the masks are built before the
operation is drawn. Conditioning those on the chosen target is the other half
of what Conditional Action Trees does, and it needs a change on the environment
side of both repositories.

## One epoch is not worth 1.56x

The update is sixteen minibatch steps or eight, and `--epochs 1` measured 1.56x
on the desktop (26,804 -> 41,904 steps/s, twice). VPT's wake phase uses each
sample at most once, so it was worth asking what it costs. Four seeds each, at
the same forty-million-step budget, everything else identical:

| held out | s1 | s2 | s3 | s4 | mean |
|---|---|---|---|---|---|
| two epochs | 92.0% | 85.5% | 87.7% | 82.0% | **86.8%** |
| one epoch | 65.6% | 35.0% | 39.1% | 64.8% | 51.1% |

It costs 36 points and doubles the spread. VPT could afford one pass because a
behaviour-cloned prior had already done the work and it had 16.8 billion frames;
here neither is true. The default stays at two.

## Why the held-out family scores higher than the training split

Every seed scores higher on the held-out family than on the training split --
92.0% against 78.3% for the best -- which is the shape a leak makes. It is not
one. The training number is an average over four families, and they are not
equally hard:

| `ar-s1`, by family | |
|---|---|
| cluttered_patch (7x7, walls scattered near it) | 95.6% |
| open_patch (7x7) | 77.8% |
| offset_patch (7x7, displaced) | 76.4% |
| **varied_patch (dimensions drawn, 3x3 to 9x9)** | **61.8%** |
| obstructed_patch, held out (3x11, one wall) | 92.0% |

`varied_patch` is the hard one, by twenty points, and it is a whole
distribution of patch shapes rather than one. The held-out family is a single
fixed shape, which is why it scores like the fixed-shape training families
rather than like `varied_patch`. Walls cost almost nothing once they have been
seen at all: the cluttered family is the *easiest* of the four.

That is worth stating plainly rather than leaving as a headline: the held-out
family is a **compositional** test -- narrow patches and walls are both in
training, their combination is not -- and it is not the hardest thing the
policy is asked to do. `varied_patch` is.

## Forty million steps is where this configuration stops

Two seeds to eighty million, against the four that stopped at forty. Sampled
success on the held-out family, at each checkpoint:

| | 20M | 30M | 40M | 60M | 80M | final |
|---|---|---|---|---|---|---|
| `long-s1` | 0.40 | 0.57 | 0.75 | 0.82 | 0.80 | 87.1% |
| `long-s2` | 0.39 | 0.71 | 0.87 | 0.71 | 0.79 | 89.1% |
| `ar-s1` (40M) | 0.31 | 0.71 | 0.77 | - | - | 92.0% |
| `ar-s2` (40M) | 0.30 | 0.71 | 0.79 | - | - | 85.5% |

Everything interesting happens between ten and forty million. After that the
curve is flat and noisy between 0.71 and 0.87, and the final numbers -- 88.1%
mean over the two long seeds against 86.8% over the four short ones -- differ
by less than the spread between seeds of the same length.

So doubling the compute buys about a point, which is to say nothing. Whatever
comes next has to be a change rather than more of the same, and the place to
look is named in the family breakdown above: `varied_patch`, twenty points
behind every other family, is the only part of the training distribution the
policy has not largely solved.

## The same recipe on a second task

`build_line` asks for the same line and scores it differently: both machines
placed through the action, and ten plates inside a 3600-tick window that begins
after construction. From scratch it has no gradient at all -- measured over
forty million steps, the return is exactly -0.6 at every checkpoint, six
hundred steps of step cost and nothing else, while entropy drifts to 0.41 and
the update KL to 6e-06. Its own reward pays for plates on a high-water mark,
but nothing pays for walking to the patch and putting the machines down, and a
policy that never makes a plate sees one number for every episode it has played.

Both tasks build the same line, so the potential describes both and the builder
demonstrates both -- it solves twelve of twelve build_line training scenes
without a change. With those, three seeds:

| held out | s1 | s2 | s3 | mean | spread |
|---|---|---|---|---|---|
| one training family | 17.2% | 14.3% | **97.9%** | 43.1% | 83.6 |
| **three training families** | **98.8%** | **86.7%** | **99.2%** | **94.9%** | **12.5** |

Training success is 98.4-99.6% in all six runs, so the recipe makes the task
learnable either way. What one family does not do is make *generalisation*
reliable: two seeds of three stayed near zero on the held-out family and the
third reached 97.9%, which is a coin flip rather than a failure. Adding
`varied_patch` and `cluttered_patch` moves the mean by 52 points and takes the
spread from 84 points to 12, and the worst diversified seed beats two of the
three single-family ones outright.

That is the same result the first task gave, from the same change, and it is
worth stating as the general lesson rather than a per-task fix: **one training
shape makes generalisation a matter of luck, and the spread is the number to
watch rather than the mean.** A single seed of either arm above would support
almost any conclusion -- 97.9% from the undiversified arm, 86.7% from the
diversified one -- which is why the seeds are reported here individually.

## Sources

- Ng, Harada, Russell (1999). Policy invariance under reward transformations.
- Wiewiora (2003). Potential-based shaping and Q-value initialization are equivalent.
- Grzes (2017). Reward shaping in episodic reinforcement learning. AAMAS.
- Sutton & Barto (2018), 2nd ed.: section 5.3 (exploring starts), p. 404 (shaping).
- Kochenderfer, Wheeler, Wray (2022). *Algorithms for Decision Making*, eq. 17.15.
- Huang & Ontanon (2020). A closer look at invalid action masking. arXiv:2006.14171.
- Huang et al. (2021). Gym-muRTS. arXiv:2105.13807.
- Salimans & Chen (2018). Learning Montezuma's Revenge from a single demonstration.
- Florensa et al. (2017). Reverse curriculum generation for reinforcement learning. CoRL.
- Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar). Nature 575.
- Bamford & Ovalle (2021). Generalising discrete action spaces with conditional action trees. arXiv:2104.07294.
- Cobbe et al. (2019). Quantifying generalization in RL. arXiv:1812.02341.
- Mnih et al. (2015). Human-level control through deep RL. Nature 518.
- Resnick et al. (2018). Backplay: man muss immer umkehren. arXiv:1807.06919.
- Baker et al. (2022). Video PreTraining (VPT). arXiv:2206.11795.
- PufferLib 5.0 `config/`; CleanRL `ppo_atari.py`, `ppo_multidiscrete_mask.py`.
