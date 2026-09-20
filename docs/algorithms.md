# Learners

`train.py --algo` picks the learner. PPO is the default and everything in
`docs/shaping.md` was measured with it.

## GRPO

`--algo grpo --group G` replaces the critic with the group itself. `G`
environments draw the same scene seed, each runs one whole episode, and an
episode's advantage is its return standardised against the other attempts at
that same scene:

    A_i = (R_i - mean(R_group)) / (std(R_group) + 1e-8)

That one number is then worn by every decision the episode made. No value
head, no GAE, no bootstrap — the group's mean *is* the baseline. The name comes
from DeepSeekMath (Shao et al. 2024), but nothing about it is specific to
language models: it is REINFORCE with a sampled baseline, which is RLOO (Kool
et al. 2019, "Buy 4 REINFORCE samples, get a baseline for free") with a
standard deviation added.

What changes in the rollout:

- **Whole episodes, not segments.** `--horizon` must cover the longest episode
  (600 decisions), and `train.py` refuses to start otherwise. A return is only
  a return if the episode finished.
- **No autoreset inside a rollout.** `VecEnv(autoreset=False)` reports an
  episode once and then stops paying it. The environment is still stepped and
  still reset underneath — to *its own* scene, so it does not consume the next
  group's seed — but none of it trains. `live` in the logged row is how much of
  the horizon was wasted that way.
- **Fewer scenes per update.** At `--envs 128 --group 8` an update sees 16
  distinct scenes, against the hundreds a 512-environment PPO rollout has in
  flight. Held-out generalisation is the number this project is judged on and
  scene diversity is what buys it, so this is the cost to watch, not a detail.

### `--baseline loo`: RLOO, and the division GRPO does not need

`--baseline group` is GRPO as published: centre an episode on its group's mean,
then divide by the group's standard deviation. `--baseline loo` is RLOO
(Kool et al. 2019; Ahmadian et al. 2024): judge the episode against the mean of
the *other* attempts, which is an unbiased baseline, and divide by nothing.

The division is not free. Liu et al. (2025) identify it as a difficulty bias:
a group whose attempts all scored about the same has a spread near zero, so
dividing by it turns a rounding-error difference into a full-size advantage.
`tests/test_grpo.py` pins the case — three failures and a fourth attempt better
by 1e-3 produce an advantage above 1.0 under `group` and below 0.01 under
`loo`. On this task most attempts fail, so most groups are near-ties, which is
exactly the regime where the bias bites.

### The discount is not optional

`group_advantage` discounts the episode return, and with potential-based
shaping it has to. A shaped reward is `gamma * phi(s') - phi(s)`, and those
terms telescope only under the gamma they were written with. Summed flat they
leave `(gamma - 1) * sum_t phi(s_t)` -- a penalty proportional to how much
potential the policy spent the episode in. At gamma 0.999 over 600 decisions:

| episode | undiscounted return | discounted |
|---|---|---|
| never leaves the start (phi ~ 0.05) | -0.0800 | -0.0500 |
| walks in and stays (phi ~ 0.50) | -0.2748 | -0.0000 |
| builds the line (phi -> 0.90) | **-0.4496** | -0.0000 |

Building the line scored 0.37 *worse* than doing nothing: the shaping was not
diluted but reversed. Discounted, each telescopes to exactly `-phi(s_0)`,
which every member of a group shares because they share a scene, so the
baseline removes it. GRPO's published form assumes gamma = 1, which is fine
for a sparse terminal verifier score and wrong here.

### The metric that tells you it has stalled

`group_spread` is the mean within-group standard deviation of returns. When it
reaches zero GRPO has no gradient at all — every attempt at a scene scored the
same, so nothing distinguishes a better one — and it will sit there quietly
rather than erroring. It is the first thing to read in `metrics.jsonl`.

`steps` counts environment steps executed; `decisions` counts the ones that
belonged to an episode and therefore trained. Under PPO they are equal. Under
GRPO they diverge as the policy improves and episodes end sooner, so compare
arms on whichever axis the question calls for — `steps` for compute, `decisions`
for sample efficiency — and say which.

## The critic, and where its gradients go

`--critic-detach` stops the value loss reaching the shared extractor, so the
critic learns on features the policy alone shapes. `--critic-updates N` runs
the value head N times per policy step, on features the trunk has already
produced, through an optimizer that holds the value head and nothing else.

Both come from SAO (Hou et al. 2026, arXiv:2607.07508), which freezes
attention under its value model and updates the critic twice per policy
update. Most of SAO does not apply here -- single-rollout sampling is what
PPO already does, and its asynchrony exists to hide LLM generation latency
that a 26,800 steps/s simulator does not have. These two do apply, because
`Policy.value` reads a slice of the extractor's features and so trains it:
measured, a value backward with the trunk attached puts |grad| 2443 into 20
extractor parameters, and zero with it detached, while the value head learns
identically either way.

The reason to care is local rather than borrowed. `--vf 2.0` froze three runs
here, and the critic was the suspect; this is the knob that tests it.

## UED: curating the scenes instead of hand-writing them

`--ued {plr,accel}` replaces the hand-written training families with a
curriculum that follows the policy's own frontier. `fsim/ued.py` holds it.

A **level is a parameter vector**, not a seed: patch bounds, offset, start
angle and two radii, and up to three wall segments. The five hand-written
families are points in that space, and `tests/test_ued.py` pins the property
the design rests on -- `family_params` reproduces every one of the ten
task-family pairs as a **byte-identical payload**, same tiles, same walls,
same start, drawn in the same order from the same seed. Without that, a UED
run and a hand-written run would be measured against different worlds.

Two radii rather than one distance, because `construct_smelting_line` draws a
fresh `uniform(9, 13)` for x and another for y: its starts lie on an
axis-aligned ellipse. A single distance desynchronised the generator's rng and
silently changed every scene after the start.

**Scoring** is PLR's positive value loss, `mean(clamp(GAE advantage, 0))` over
the episode -- high where the policy is still learning, low both where it has
mastered a level and where it never gets anywhere, which makes it a frontier
detector rather than a difficulty meter.

What the buffer stores is not that number but its **standing among the levels
scored in the same rollout**, in [0, 1]. Raw positive value loss is measured
against a critic that is still learning and its scale drifts: over one
20M-step run the buffer's mean raw score rose from 0.0031 to 0.0172. The
buffer compares scores directly when it decides what to evict, so on raw
values it ranked levels by *when* they were measured rather than by what they
were -- staleness correlated with score at r = -0.170, against -0.064 for wall
count, +0.010 for patch size and +0.000 for start distance, and levels
sampled recently averaged 0.0231 against 0.0168 for those long unsampled. A
standing is comparable across updates; a raw score is not. `raw_mean` in the
logged `ued` block keeps the underlying drift visible.

**Sampling** mixes rank over score with staleness:

    P = (1 - rho) * P_score + rho * P_staleness,  P_score ~ 1 / rank ** (1/beta)

**Robust PLR** is the constraint that matters, and it lives in the trainer:
`--ued-train-frac` of the environment slots replay curated levels and are
trained on; the rest run proposed levels, are scored, and **take no gradient
step at all**. Training on whatever the generator emitted is what biases
vanilla PLR. `decisions` in the metrics row is the check -- it comes out at
exactly the training fraction of `steps`.

**ACCEL** proposes by mutating a level already held rather than drawing a
fresh random one, so complexity compounds from the frontier. `walls_mean`,
`tiles_mean` and `from_families` in the `ued` block are how to see whether it
is working: a curriculum that is doing its job leaves the hand-written
families behind.

UED does **not** imply `--whole-episodes`. It did, so a level's score was the
mean positive advantage over exactly one episode -- but that pinned every UED
run to 128 environments at horizon 600, where three seeds of the *hand-written*
control came back at 0.244, 0.088 and 0.002 held-out. A hundredfold spread
hides any curriculum effect, so the comparison could not have detected one.

Scoring a rollout segment instead is what PLR's own implementation does, and
it runs at the tuned 512 x 64 whose four seeds span ten points. The cost is an
approximation: autoreset hands a finished slot a new level part-way through a
rollout, so `Curriculum.snapshot` is taken before the rollout and the whole
segment is credited to the level the slot started with. At horizon 64 against
600-decision episodes, about a tenth of slots change level inside a rollout.

## UED measured: WITHDRAWN, the arms were not comparable

**The comparison below is confounded and its verdict does not stand.** It is
kept because the confound is the interesting part.

A walled scene gets no demonstration start by default, and an ACCEL buffer
drives the wall fraction from the hand-written split's 25% to **81-87%**.
Measured demonstration rates: control **37.4%**, `u2-s1` **9.5%**, `u2-s2`
**6.7%**, `u2-s3` **8.0%** -- four to fivefold apart. Across the three ACCEL
seeds, held-out success tracks the demonstration rate monotonically:

| seed | demonstrated | held-out |
|---|---|---|
| `u2-s1` | 9.5% | 0.787 |
| `u2-s3` | 8.0% | 0.766 |
| `u2-s2` | 6.7% | 0.389 |

That accounts for the result better than anything about curation, and the
demonstration curriculum is the thing this task is known to depend on --
`build_line` had no gradient at all without it.

Worse, this confound was found and fixed earlier the same night, and then
reintroduced when the fix was reverted: the revert was measured on the
hand-written split, where the wall fraction is 25% and the interaction is
mild, and applied to a curriculum that drives it past 80%.

`u3-s1..s3` rerun both arms under `--demo-obstructed`, where demonstration
eligibility no longer depends on walls and both sit near 50%. That is a worse
absolute setting -- the control scores 0.568 there against 0.868 -- and it is
the same setting for both, which is what the comparison needs.

## The withdrawn comparison

Three seeds each at the tuned config (512 environments, horizon 64, 40M
steps), default settings, identical but for where training scenes come from.
Evaluation draws the frozen hand-written families either way.

| run | held-out `obstructed_patch` | broad parameter space |
|---|---|---|
| `ar-s1` | 0.920 | 0.883 |
| `ar-s2` | 0.856 | 0.859 |
| `ar-s3` | 0.877 | 0.797 |
| `ar-s4` | 0.820 | 0.852 |
| **hand-written mean** | **0.868** | **0.848** |
| `u2-s1` | 0.787 | 0.828 |
| `u2-s2` | 0.389 | 0.398 |
| `u2-s3` | 0.766 | 0.688 |
| **ACCEL mean** | **0.647** | **0.638** |

Twenty-one points behind on both axes, and every ACCEL seed below every
hand-written one.

### The curriculum worked; the policy it produced did not

This is not a case of the machinery failing quietly. Every ACCEL seed ran
clean, held the Robust PLR split at `decisions/steps` 0.750, filled its
4,000-level buffer, drove `from_families` to 0.004-0.006, and kept a mean
standing of 0.64-0.72 where a buffer sampling at random would sit at 0.50.
The policies it produced **build more reliably than the control**:

| | `line_built` | peak phi | held-out |
|---|---|---|---|
| `ar-s1` | 0.806 | 0.732 | 0.920 |
| `u2-s1` | 0.938 | 0.849 | 0.787 |
| `u2-s2` | **0.979** | **0.870** | **0.389** |
| `u2-s3` | 0.959 | 0.856 | 0.766 |

`u2-s2` puts a line down in 98% of episodes, has the highest potential of any
run in this project, and finishes the task in 30% of them. It learned to
build and not to finish. Its curriculum was also the hardest of the three
(`walls_mean` 1.63 against 1.30 and 1.49) and it scored worst -- the generator
outrunning the student, which is the failure ACCEL's regret objective exists
to avoid.

### The excuse that did not survive

`construct_smelting_line` has one test family, and `varied_patch` and
`cluttered_patch` were added to the training split *because* they lifted it
from 0-1.4% to 88%. So a curriculum that stops training on them should score
badly on it whether or not it generalises worse. That is a real effect --
`u2-s1` gains four points moving from the holdout to the broad space, 0.787
to 0.828 -- and it is nowhere near large enough. ACCEL loses by the same
twenty-one points on a space neither arm is tuned to, and `u2-s2` is equally
bad on both (0.389, 0.398). `tools/eval_broad.py` exists for this check.

### What the number is worth

Three bugs were fixed before this comparison could be run at all, each of
which would have produced a null for the wrong reason: demonstrations
starved by a wall-skip rule, a level buffer ranking on recency rather than
regret, and UED forced onto a rollout shape where control seeds ranged from
0.002 to 0.244. The result is that a null now means something.

What it does not license is "UED does not work". The variance is the story:
two of three seeds land at 0.787 and 0.766, within reach of the control's
0.820 floor, and one collapses. A method whose best seed is competitive and
whose worst is catastrophic wants a stabiliser, not abandonment -- and the
obvious suspect is a generator with no notion of how far ahead of the
student it has got.
