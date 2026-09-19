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
