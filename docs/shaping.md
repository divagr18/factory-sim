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
- PufferLib 3.0/4.0 `config/default.ini`; CleanRL `ppo_atari.py`, `ppo_multidiscrete_mask.py`.
