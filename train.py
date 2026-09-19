"""Masked-MultiDiscrete PPO on the simulator, in one file.

CleanRL's layout (one file, one loop, no framework) with PufferLib's scale
(hundreds of C environments per batch). The choices, and where they come from:

* Invalid-action masking on every dimension, operation first and arguments
  conditioned on it (`fsim/policy.py`; Huang & Ontanon 2020, Gym-muRTS 2021).
* GAE (Schulman et al. 2016) with lambda 0.95. gamma is 0.999 by default: an
  episode is 600 decisions and the verification score arrives on the last one,
  so 0.99's 100-decision horizon would discount it to 0.2% of its value from
  the decisions that build the line (0.99^590); 0.999^590 is 55%.
* Clipped surrogate and clipped value loss, advantage normalisation per
  minibatch, gradient-norm clipping and a linearly annealed learning rate: the
  CleanRL `ppo_atari` defaults (clip 0.2 as Huang et al. use for masked PPO).
  PufferLib's tuned configs for sparse, long-horizon tasks differ: horizon
  128-256, minibatch 8192-65536, vf 2-6, lambda 0.8-0.94, and no advantage
  normalisation. --horizon, --minibatches, --vf, --vf-clip, --lam and
  --no-adv-norm exist to test that.
* Entropy 0.01 (CleanRL, Huang et al.) rather than PufferLib's 0.001: the masks
  already remove most of the action space, and early exploration is the
  bottleneck here.
* `--shaping`, over the line potential phi of `fsim_rl_potential`:
  `potential` is gamma*phi(s') - phi(s) with phi(terminal) = 0 (Ng, Harada &
  Russell 1999; Grzes 2017), using this run's gamma; `progress` pays the rise
  of phi's running maximum, half-weighted and capped at 0.45 (FactorioRL's
  HIGH_WATER kind). See docs/shaping.md for why the second exists.

`--demo-starts p` begins that fraction of training episodes partway along a
scripted build (`fsim/expert.py`; Salimans & Chen 2018). Training metrics
report scene-start episodes and demonstration-start ones separately, and
evaluation always starts at the scene's own start.

Truncation is not bootstrapped: construct_smelting_line never truncates (the
budget running out starts verification, a true terminal), and build_line's
600-decision truncation is treated as terminal, a known and small bias.

    python train.py --run sparse-s1 --seed 1 --steps 20000000
    python train.py --run progress-s1 --seed 1 --steps 20000000 --shaping progress
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fsim import ffi, lib
from fsim.policy import EXTRACTOR_VERSION, Policy, export, masked_kl
from fsim.vec import VecEnv, obs_layout, unpack_grid

ROOT = Path(__file__).resolve().parent
KEYS = ("grid", "entities", "entity_mask", "self", "inventory", "goal")

#: Backplay's curriculum (Resnick et al. 2018, arXiv:1807.06919): a
#: demonstration start is drawn from a window measured backwards from the end
#: of the build, and the window slides back on a fixed schedule -- here at
#: fractions of the training budget. The last window is past any build, so
#: every episode starts at the scene's own start. Their practical findings:
#: advancing too fast hurts and advancing slowly does not, adaptive
#: success-thresholded advancement was slower than a fixed schedule, and a dip
#: in success when the window reaches the start is expected.
BACKPLAY_SCHEDULE = (
    (0.00, (0, 2)),
    (0.10, (1, 4)),
    (0.20, (2, 6)),
    (0.30, (4, 9)),
    (0.40, (6, 13)),
    (0.50, (9, 20)),
    (0.65, (99, 99)),
)


def backplay_window(progress: float) -> tuple[int, int]:
    """The window for this fraction of the training budget."""
    window = BACKPLAY_SCHEDULE[0][1]
    for at, value in BACKPLAY_SCHEDULE:
        if progress >= at:
            window = value
    return window


#: The same ladder, climbed on evidence rather than on the clock.
BACKPLAY_LADDER = tuple(window for _at, window in BACKPLAY_SCHEDULE)


class GatedBackplay:
    """Backplay's window, widened only once the policy can finish the last one.

    The fixed schedule is open-loop: it slides the window back at a set
    fraction of the budget whether or not the policy has learned to start from
    where it already is. On an easy scene mix that tracked; on a harder one it
    outruns the policy, which then loses the skill it had -- measured, a run
    reached `line_built` 0.16 by 2M steps and 0.0 by 15M.

    Advancing on measured success instead is the reverse curriculum as
    Florensa et al. (2017) state it: keep starts whose success sits in a band,
    and move outwards from the ones already solved.
    """

    def __init__(self, threshold: float = 0.5, minimum: int = 64, settle: int = 8) -> None:
        self.threshold = threshold
        self.minimum = minimum  # episodes at the deepest cut before it may move
        self.settle = settle  # updates to wait after moving, so the buffer refills
        self.index = 0
        self.waited = settle
        self.advanced_at: list[int] = []

    @property
    def window(self) -> tuple[int, int]:
        return BACKPLAY_LADDER[self.index]

    def update(self, demo: list[dict], steps: int) -> tuple[int, int]:
        """Read the recent demonstration episodes; widen if they are solved.

        Judged on the *deepest* cut into the demonstration the window offers,
        not on the average over it. Averaging is what the first version did,
        and it read a window whose easiest start was solved 100% of the time
        and whose next-easiest was solved 1% of the time as solved: one run
        climbed from the first rung to the last in 400k steps having never
        learned to make a single decision for itself.
        """
        if self.index + 1 >= len(BACKPLAY_LADDER):
            return self.window
        if self.waited < self.settle:  # the buffer still holds the old window
            self.waited += 1
            return self.window
        cuts: dict[str, list[bool]] = {}
        for episode in demo:
            start = episode["start"]
            if start.startswith("back"):
                cuts.setdefault(start, []).append(episode["success"])
        if not cuts:
            return self.window
        deepest = max(cuts, key=lambda name: int(name[4:]))
        if len(cuts[deepest]) >= self.minimum:
            if float(np.mean(cuts[deepest])) >= self.threshold:
                self.index += 1
                self.waited = 0
                self.advanced_at.append(steps)
        return self.window


def parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True)
    p.add_argument(
        "--algo",
        choices=("ppo", "grpo"),
        default="ppo",
        help="grpo drops the critic: --group environments share a scene, each "
        "runs one whole episode, and an episode's advantage is its return "
        "standardised against its group's (Shao et al. 2024, and RLOO before "
        "it). --horizon must then cover a whole episode and --vf is ignored",
    )
    p.add_argument(
        "--group",
        type=int,
        default=8,
        help="environments per scene: i and j draw the same seed when "
        "i // group == j // group. GRPO's baseline is the group's mean, but "
        "the grouping is independent of the learner -- PPO with --group and "
        "--whole-episodes is the control that separates 'GRPO loses' from "
        "'this rollout shape loses'",
    )
    p.add_argument(
        "--whole-episodes",
        action="store_true",
        help="one complete episode per environment per rollout, with no "
        "autoreset inside it. Implied by --algo grpo, which cannot compute a "
        "return without it",
    )
    p.add_argument(
        "--credit",
        choices=("episode", "togo"),
        default="episode",
        help="what --algo grpo hands each decision. episode is GRPO's: the "
        "whole episode's return, the same number on every timestep. togo is "
        "the discounted reward from that decision onward, judged against the "
        "group's at the same decision -- which keeps the group baseline but "
        "stops discarding the shaping's per-decision structure",
    )
    p.add_argument(
        "--baseline",
        choices=("group", "loo"),
        default="group",
        help="how --algo grpo judges an episode against its group. group is "
        "GRPO's: centre on the group mean and divide by its spread. loo is "
        "RLOO's: centre on the mean of the other attempts and do not divide, "
        "which drops the difficulty bias Liu et al. (2025) attribute to that "
        "division (docs/algorithms.md)",
    )
    p.add_argument("--task", default="construct_smelting_line")
    p.add_argument(
        "--max-steps",
        type=int,
        default=600,
        help="decisions per episode. The two construction tasks allow 600; "
        "plate_line allows 400, and a task trained on the wrong budget is a "
        "different task",
    )
    p.add_argument(
        "--tick-limit",
        type=int,
        default=None,
        help="game ticks per episode; defaults to 18000. plate_line's line "
        "needs about 7,200 ticks to make thirty plates, and FactorioRL allows "
        "it 24000",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=20_000_000)
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument(
        "--ent",
        type=float,
        default=None,
        help="entropy bonus; defaults to 0.01, or to 0 with --prior, which is "
        "VPT's arrangement: the KL to the prior replaces the bonus outright",
    )
    p.add_argument(
        "--prior",
        type=Path,
        default=None,
        help="a behaviour-cloned checkpoint to anchor to (tools/behaviour_clone.py). "
        "A run that uses one is not a from-scratch run.",
    )
    p.add_argument("--kl-coef", type=float, default=0.2, help="VPT's rho")
    p.add_argument("--kl-decay", type=float, default=0.9995, help="rho's decay per update")
    p.add_argument(
        "--independent-arguments",
        dest="autoregressive",
        action="store_false",
        help="draw all five arguments at once instead of drawing the target "
        "first and scoring the rest conditioned on it. Measured over four seeds "
        "each, drawing them in order moved held-out success from 60.0%% to "
        "86.8%% and took the spread from 51 points to 10 (docs/shaping.md)",
    )
    p.add_argument("--autoregressive", dest="autoregressive", action="store_true")
    p.set_defaults(autoregressive=True)
    p.add_argument(
        "--fixed-demo-layout",
        action="store_true",
        help="demonstrate the one canonical build pose rather than drawing one, "
        "which is what produced a policy that memorised it (docs/shaping.md)",
    )
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument(
        "--critic-detach",
        action="store_true",
        help="stop the value loss from reaching the shared extractor, so the "
        "critic learns on features the policy alone shapes. SAO (Hou et al. "
        "2026, arXiv:2607.07508) freezes attention under the critic for this "
        "reason -- its gradients are the ones that destabilised full-parameter "
        "training. Here vf 2.0 froze three runs, and the critic is the suspect",
    )
    p.add_argument(
        "--critic-updates",
        type=int,
        default=1,
        help="value-head steps per policy step. SAO decouples the two and runs "
        "the critic twice per policy update to cut variance; the extra steps "
        "reuse the detached features, so they cost a head pass and no trunk",
    )
    p.add_argument(
        "--vf-clip",
        type=float,
        default=None,
        help="value-clip range; defaults to --clip. PufferLib's sparse configs "
        "use 3.5-5 (effectively unclipped) against a 0.1-0.2 policy clip",
    )
    p.add_argument(
        "--no-adv-norm",
        action="store_true",
        help="do not standardise advantages per minibatch, as PufferLib 5.0 does not: "
        "on a terminal-only reward most segments carry no signal, and standardising "
        "inflates those into full-size gradients",
    )
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--shaping", choices=("none", "potential", "progress", "both"), default="none")
    p.add_argument("--start-curriculum", type=float, default=0.0)
    p.add_argument(
        "--demo-schedule",
        choices=("uniform", "backplay", "gated"),
        default="uniform",
        help="uniform: a stage drawn uniformly; backplay: a window sliding back "
        "on the clock; gated: the same ladder, climbed on measured success",
    )
    p.add_argument(
        "--gate",
        type=float,
        default=0.5,
        help="success at the current window needed to widen it, with --demo-schedule gated",
    )
    p.add_argument(
        "--demo-starts",
        type=float,
        default=0.0,
        help="fraction of training episodes started partway along the scripted build",
    )
    p.add_argument(
        "--horizon-curriculum",
        type=int,
        nargs=2,
        metavar=("LO", "HI"),
        help="draw each training episode's decision budget from [LO, HI] (default: 600)",
    )
    p.add_argument(
        "--eval-epsilon",
        type=float,
        default=0.05,
        help="chance a greedy decision samples instead, in the final report's "
        "`epsilon` mode; a deterministic policy in a deterministic task cycles",
    )
    p.add_argument("--eval-episodes", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=2_000_000)
    p.add_argument("--out", type=Path, default=ROOT / "runs")
    p.add_argument("--no-graph", action="store_true", help="rollout inference without CUDA graphs")
    p.add_argument("--no-final", action="store_true", help="skip the final evaluation and export")
    p.add_argument(
        "--action-space",
        choices=("v1", "v2"),
        default="v1",
        help="v1: FactorioRL's parameterized-v1; v2: the simulator prototype",
    )
    args = p.parse_args(argv)
    # A return needs a whole episode, so grpo cannot opt out of the rollout
    # shape. Applied here rather than in main, so anything reading the parsed
    # arguments sees a coherent pair.
    if args.algo == "grpo":
        args.whole_episodes = True
    return args


def to_device(obs: dict, device) -> dict:
    return {k: torch.from_numpy(obs[k]).to(device) for k in KEYS}


class Rollout:
    """Rollout inference with one host-to-device copy and one kernel launch.

    The environments write compact observations (`fsim_obs8`, the grid as
    bytes) and masks straight into page-locked memory. Each decision copies
    both blocks to static device tensors, and a CUDA graph -- slicing the block
    into tensors, the extractor, sampling, the value head -- replays in one
    launch. Measured before this: about 830 kernel launches and 13 MB of
    transfer per decision at 128 environments, most of a 19 ms step.
    """

    DTYPES = {"<f4": torch.float32, "|i1": torch.int8, "|u1": torch.uint8}

    def __init__(self, policy: Policy, n: int, device, graph: bool = True) -> None:
        self.policy, self.n, self.device = policy, n, device
        self.size = ffi.sizeof("fsim_obs8")
        pin = device.type == "cuda"
        self.host = torch.empty(n * self.size, dtype=torch.uint8, pin_memory=pin)
        self.host_mask = torch.empty(n * 201, dtype=torch.uint8, pin_memory=pin)
        self.block = torch.zeros(n * self.size, dtype=torch.uint8, device=device)
        self.mask = torch.zeros((n, 201), dtype=torch.bool, device=device)
        self.layout = obs_layout(compact=True)
        self.graph = None
        self.out = None
        self.use_graph = graph and device.type == "cuda"

    def memories(self) -> dict:
        return {
            "obs_memory": (self.host.data_ptr(), self.host),
            "mask_memory": (self.host_mask.data_ptr(), self.host_mask),
        }

    def decode(self, block: torch.Tensor, n: int) -> dict:
        """Field tensors over a flat block of `n` compact observations."""
        rows = block.view(n, self.size)
        out = {}
        for key, (offset, dtype, shape) in self.layout.items():
            tdtype = self.DTYPES[dtype]
            nbytes = math.prod(shape) * torch.empty((), dtype=tdtype).element_size()
            out[key] = rows[:, offset : offset + nbytes].contiguous().view(tdtype).view(n, *shape)
        return out

    def _infer(self):
        obs = unpacked(self.decode(self.block, self.n))
        with torch.no_grad():
            f = features(self.policy, obs)
            actions, logp = self.policy.act(f, self.mask)
            value = self.policy.value(f)
        return obs, actions, logp, value

    def __call__(self):
        """-> (observation tensors, actions, log-probabilities, values).

        The returned tensors are overwritten by the next call."""
        self.block.copy_(self.host, non_blocking=True)
        self.mask.copy_(self.host_mask.view(self.n, 201), non_blocking=True)
        if not self.use_graph:
            return self._infer()
        if self.graph is None:
            # device constants are made here, outside the capture
            flags = torch.zeros((1, lib.RL_FLAG_BYTES), dtype=torch.uint8, device=self.device)
            amount = torch.zeros((1, 65, 65), dtype=torch.uint8, device=self.device)
            unpack_grid(flags, amount, torch)
            side = torch.cuda.Stream()
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._infer()
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self._infer()
        self.graph.replay()
        return self.out


def unpacked(obs: dict) -> dict:
    """Packed observation tensors as the policy reads them: the byte grid
    rebuilt, the packed fields dropped."""
    grid = unpack_grid(obs.pop("flags"), obs.pop("amount"), torch)
    return {**obs, "grid": grid}


def _group_stats(values: torch.Tensor, live: torch.Tensor, group: int, baseline: str):
    """Centre `values` on their group, counting only the members still running.

    A member whose episode has ended contributes nothing to `values` at that
    timestep, and must not be averaged in either: with a plain mean over the
    block, four dead siblings scoring a nominal zero halve the baseline and
    hand the survivor a large advantage for no reason. Here, success ends an
    episode, so the attempts that run longest are the failures -- averaging the
    dead in pays the policy to fail.
    """
    counts = live.sum(-1, keepdim=True)
    mean = (values * live).sum(-1, keepdim=True) / counts.clamp(min=1.0)
    centred = (values - mean) * live
    if baseline == "loo":
        # R_i against the mean of the others = (n / (n - 1)) * (R_i - mean),
        # over the n that were actually running.
        return centred * torch.where(counts > 1, counts / (counts - 1).clamp(min=1.0), counts * 0)
    variance = (centred * centred).sum(-1, keepdim=True) / counts.clamp(min=1.0)
    return centred / (variance.sqrt() + 1e-8)


def group_advantage(
    rewards: torch.Tensor, live: torch.Tensor, group: int, gamma: float,
    baseline: str = "group",
) -> torch.Tensor:  # fmt: skip
    """A critic-free advantage: one number per episode, worn by all of its
    decisions.

    `rewards` and `live` are `(horizon, envs)`; consecutive blocks of `group`
    environments ran the same scene, so an episode's return is judged against
    its siblings rather than against a critic's guess. Timesteps after an
    episode ended are not part of it and get zero, so they contribute no
    gradient.

    `baseline="group"` is GRPO (Shao et al. 2024): centre on the group's mean,
    then divide by its standard deviation.

    `baseline="loo"` is RLOO (Kool et al. 2019; Ahmadian et al. 2024): judge an
    episode against the mean of the *others*, which is an unbiased baseline,
    and do not divide. The division is what Liu et al. (2025) identify as
    GRPO's difficulty bias -- a group that nearly all failed has a tiny spread,
    so dividing by it turns noise into full-size advantages. On a task where
    most attempts fail that is the case to worry about, which is why both are
    here rather than only one.

    A group whose members all scored the same gets zero advantage under either,
    which is the honest answer: nothing in it distinguishes a better attempt
    from a worse one.

    **The return is discounted**, and with potential-based shaping it has to
    be. A shaped reward is gamma * phi(s') - phi(s), and those terms telescope
    to a constant only under the same gamma they were written with. Summed
    undiscounted they leave a residue of (gamma - 1) * sum_t phi(s_t): a
    penalty proportional to how much potential the policy spent the episode
    in. Measured at gamma 0.999 over 600 decisions, an episode that builds
    the line scores -0.45 against -0.08 for one that never leaves the start,
    so the shaping is not merely diluted but inverted -- the return pays the
    policy to stay away from the patch. Discounted, every one of those
    telescopes to exactly -phi(s_0), which is the same for every member of a
    group because they share a scene, and the baseline removes it outright.
    """
    horizon = rewards.shape[0]
    discount = gamma ** torch.arange(horizon, device=rewards.device, dtype=rewards.dtype)
    totals = (rewards * live * discount.unsqueeze(1)).sum(0).view(-1, group)
    # Every member has a whole episode's return, whenever its episode ended,
    # so every member counts towards the baseline.
    scaled = _group_stats(totals, torch.ones_like(totals), group, baseline)
    return scaled.reshape(1, -1) * live


def togo_advantage(
    rewards: torch.Tensor, live: torch.Tensor, group: int, gamma: float, baseline: str = "group"
) -> torch.Tensor:
    """The group baseline, kept; the flat episode return, dropped.

    Each decision is judged by the discounted reward from that decision
    onward, against what its group's other attempts earned from the *same*
    decision. Still no critic -- the siblings are the baseline, as in
    `group_advantage` -- but a decision that raised the potential is now
    credited for it instead of being averaged into six hundred others.

    This is the control for the obvious reading of GRPO's failure here: our
    reward is shaped per decision, and collapsing an episode to one number
    keeps only its sum. It is also what Sutton & Barto's REINFORCE actually
    uses -- G_t, the return from t, against a baseline that "should vary with
    state" (2nd ed. 13.4). A single episode return is G_0 worn by all six
    hundred decisions, which is neither.

    The baseline at each decision counts only the members still running. A
    member that has finished contributes nothing at that timestep and must not
    be averaged in as a zero: success ends an episode here, so the longest
    survivors are the failures, and averaging the dead in pays for failing.
    """
    horizon = rewards.shape[0]
    masked = rewards * live
    togo = torch.zeros_like(masked)
    running = torch.zeros(masked.shape[1], device=masked.device)
    for t in reversed(range(horizon)):
        running = masked[t] + gamma * running
        togo[t] = running
    scaled = _group_stats(
        togo.view(horizon, -1, group), live.view(horizon, -1, group), group, baseline
    )
    return scaled.reshape(horizon, -1) * live


def features(policy: Policy, obs: dict) -> torch.Tensor:
    """The extractor under bf16 autocast; the heads run in float32.

    The autocast cast cache is off, so a CUDA graph holds no stale casts of the
    weights across optimizer steps."""
    cuda = obs["grid"].is_cuda
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda, cache_enabled=False):
        f = policy.features(*(obs[k] for k in KEYS))
    return f.float()


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a success rate."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


@torch.no_grad()
def evaluate(
    policy, device, args, split: str, episodes: int, greedy: bool, epsilon: float = 0.0
) -> dict:
    """Success over fresh evaluation seeds, disjoint from every training seed."""
    n = min(64, episodes)
    env = VecEnv(
        n, args.task, split=split, seed=args.seed, threads=args.threads,
        shaping=False, gamma=args.gamma, eval_seeds=True, action_space=args.action_space,
        max_steps=args.max_steps, tick_limit=args.tick_limit,
    )  # fmt: skip
    obs, masks = env.reset()
    done: list[dict] = []
    # Each slot runs exactly one episode per round, so the families drawn are
    # the first `episodes` seeds, independent of how long each episode lasts.
    active = np.ones(n, dtype=bool)
    started = n
    while len(done) < episodes:
        t = to_device(obs, device)
        mask = torch.from_numpy(masks).to(device).bool()
        actions, _ = policy.act(features(policy, t), mask, greedy, epsilon)
        obs, masks, _, term, trunc, finished = env.step(actions.cpu().numpy())
        ended = np.flatnonzero(term | trunc)
        for i, record in zip(ended, finished, strict=True):
            if active[i]:
                done.append(record)
                if started >= episodes:
                    active[i] = False
                started += 1
        if not active.any():
            break
    env.close()
    done = done[:episodes]
    wins = sum(r["success"] for r in done)
    low, high = wilson(wins, len(done))
    return {
        "split": split,
        "greedy": greedy,
        "epsilon": epsilon if greedy else None,
        "episodes": len(done),
        "success": wins / max(1, len(done)),
        "success_ci95": [low, high],
        "verified_output_mean": float(np.mean([max(0.0, r["verified_output"]) for r in done])),
        "by_family": {
            f: sum(r["success"] for r in done if r["family"] == f)
            / max(1, sum(r["family"] == f for r in done))
            for f in sorted({r["family"] for r in done})
        },
    }


def main(argv=None) -> int:
    args = parse(argv)
    if args.ent is None:
        args.ent = 0.0 if args.prior else 0.01
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.out / args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps({**vars(args), "out": str(args.out), "extractor_version": EXTRACTOR_VERSION},
                   indent=2, default=str), "utf-8",
    )  # fmt: skip
    log = (out / "metrics.jsonl").open("a", encoding="utf-8")

    torch.backends.cudnn.benchmark = True
    policy = Policy(action_space=args.action_space).to(device)
    if device.type == "cuda":
        # bf16 grid materialisation, and channels-last weights: the grid path's
        # second convolution gets its input in that layout (fsim/policy.py).
        policy.extractor.input_dtype = torch.bfloat16
        policy = policy.to(memory_format=torch.channels_last)
    policy.autoregressive = args.autoregressive
    prior = None
    if args.prior:
        prior = Policy(action_space=args.action_space).to(device)
        prior.load_state_dict(torch.load(args.prior, map_location=device, weights_only=True))
        if device.type == "cuda":
            prior.extractor.input_dtype = torch.bfloat16
            prior = prior.to(memory_format=torch.channels_last)
        prior.autoregressive = args.autoregressive
        prior.eval()
        for parameter in prior.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    # The critic's extra steps get their own optimizer over the value head
    # alone, so they cannot move the trunk even when --critic-detach is off.
    critic_opt = (
        torch.optim.Adam(policy.value_head.parameters(), lr=args.lr, eps=1e-5)
        if args.critic_updates > 1 and args.algo != "grpo"
        else None
    )
    rollout = Rollout(policy, args.envs, device, graph=not args.no_graph)
    env = VecEnv(
        args.envs, args.task, split="train", seed=args.seed, threads=args.threads,
        shaping=args.shaping, gamma=args.gamma, start_curriculum=args.start_curriculum,
        max_steps=tuple(args.horizon_curriculum) if args.horizon_curriculum else args.max_steps,
        tick_limit=args.tick_limit,
        demo_starts=args.demo_starts, compact=True, action_space=args.action_space,
        group=args.group, autoreset=not args.whole_episodes,
        **rollout.memories(),
    )  # fmt: skip
    env.demo_layouts = not args.fixed_demo_layout

    N, T = args.envs, args.horizon
    batch = N * T
    updates = args.steps // batch
    if args.algo == "grpo":
        longest = max(args.horizon_curriculum) if args.horizon_curriculum else args.max_steps
        if T < longest:
            raise SystemExit(
                f"--algo grpo needs --horizon >= {longest}, the longest episode: a "
                "group's returns are its episodes' returns, and an episode cut at "
                f"the horizon has none. Got --horizon {T}."
            )
    buf = {
        # The byte grid, unpacked once in the rollout graph: the update reads it
        # every epoch, and unpacking per minibatch cost more than it saved.
        "grid": torch.zeros((T, N, 6, 65, 65), dtype=torch.uint8, device=device),
        "entities": torch.zeros((T, N, 32, 16), device=device),
        "entity_mask": torch.zeros((T, N, 32), dtype=torch.int8, device=device),
        "self": torch.zeros((T, N, 12), device=device),
        "inventory": torch.zeros((T, N, 14), device=device),
        "goal": torch.zeros((T, N, 12), device=device),
    }
    masks_buf = torch.zeros((T, N, 201), dtype=torch.bool, device=device)
    actions_buf = torch.zeros((T, N, 6), dtype=torch.long, device=device)
    logp_buf = torch.zeros((T, N), device=device)
    rew_buf = torch.zeros((T, N), device=device)
    done_buf = torch.zeros((T, N), device=device)
    val_buf = torch.zeros((T, N), device=device)
    #: GRPO only: whether the timestep belongs to an episode still running. An
    #: environment that finishes early keeps producing observations, and none
    #: of them are part of any episode.
    live_buf = torch.zeros((T, N), device=device)

    env.reset()
    host_step = torch.empty((2, N), dtype=torch.float32, pin_memory=device.type == "cuda")
    step_np = host_step.numpy()
    next_done = torch.zeros(N, device=device)
    episodes: list[dict] = []
    gate = GatedBackplay(threshold=args.gate)
    steps = 0
    decisions = 0
    unfinished = 0.0
    spread = 0.0
    start = time.perf_counter()
    next_eval = args.eval_every
    best = -1.0

    def clock() -> float:
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    for update in range(1, updates + 1):
        began = clock()
        if args.demo_schedule == "backplay":
            env.demo_window = backplay_window((update - 1) / updates)
        elif args.demo_schedule == "gated":
            env.demo_window = gate.window
        frac = 1.0 - (update - 1) / updates
        for group in optimizer.param_groups:
            group["lr"] = frac * args.lr
        if args.whole_episodes:
            # Every rollout is a fresh set of episodes. Required by --algo
            # grpo, whose group returns are only comparable if its members
            # started together -- but it belongs to the rollout shape, not to
            # the learner. Without it, environments that finished in the first
            # rollout are parked for the whole run and every later rollout is
            # six hundred copies of a dead state.
            env.reset()
            next_done = torch.zeros(N, device=device)
        live = torch.ones(N, device=device)
        for t in range(T):
            obs, action, logp, value = rollout()
            for k in KEYS:
                buf[k][t].copy_(obs[k])
            masks_buf[t].copy_(rollout.mask)
            done_buf[t].copy_(next_done)
            live_buf[t].copy_(live)
            actions_buf[t].copy_(action)
            logp_buf[t].copy_(logp)
            val_buf[t].copy_(value)
            _, _, reward, term, trunc, finished = env.step(action.cpu().numpy())
            episodes.extend(finished)
            step_np[0] = reward
            step_np[1] = term | trunc
            host = host_step.to(device, non_blocking=True)
            rew_buf[t].copy_(host[0])
            next_done = host[1]
            live = live * (1.0 - next_done)
        steps += batch
        rolled = clock()

        with torch.no_grad():
            if args.algo == "grpo":
                if args.credit == "togo":
                    adv = togo_advantage(
                        rew_buf, live_buf, args.group, args.gamma, args.baseline
                    )
                else:
                    adv = group_advantage(
                        rew_buf, live_buf, args.group, args.gamma, args.baseline
                    )
                returns = torch.zeros_like(adv)
                unfinished = float(live.sum())
                # The number to watch: a group whose attempts all score the
                # same teaches nothing, and GRPO stalls without ever erroring.
                totals = (rew_buf * live_buf).sum(0).view(-1, args.group)
                spread = float(totals.std(1).mean())
            else:
                next_value = rollout()[3].clone()
                adv = torch.zeros_like(rew_buf)
                last = torch.zeros(N, device=device)
                for t in reversed(range(T)):
                    if t == T - 1:
                        nonterminal = 1.0 - next_done
                        value_next = next_value
                    else:
                        nonterminal = 1.0 - done_buf[t + 1]
                        value_next = val_buf[t + 1]
                    delta = rew_buf[t] + args.gamma * value_next * nonterminal - val_buf[t]
                    last = delta + args.gamma * args.lam * nonterminal * last
                    adv[t] = last
                returns = adv + val_buf
                unfinished = 0.0
                spread = 0.0

        flat = {k: v.reshape(batch, *v.shape[2:]) for k, v in buf.items()}
        b_masks = masks_buf.reshape(batch, -1)
        b_actions = actions_buf.reshape(batch, -1)
        b_logp, b_adv = logp_buf.reshape(-1), adv.reshape(-1)
        b_ret, b_val = returns.reshape(-1), val_buf.reshape(-1)
        # Train on the decisions that were actually part of an episode. The
        # rest are an artefact of holding the horizon open until the last
        # group member finishes, and belong to no episode at all. This follows
        # the rollout shape, not the learner: a PPO arm run with
        # --whole-episodes has exactly the same dead tail.
        usable = torch.nonzero(live_buf.reshape(-1), as_tuple=False).squeeze(1)
        if not args.whole_episodes:
            usable = torch.arange(batch, device=device)
        n_usable = usable.numel()
        decisions += n_usable
        mb = max(1, n_usable // args.minibatches)

        stats = {"pg": [], "v": [], "ent": [], "kl": [], "clipfrac": [], "prior_kl": []}
        # VPT decays rho by a fixed factor each iteration, so the prior protects
        # the policy's skills early and stops constraining it later: "This method
        # protects policy skills in early iterations while guaranteeing that the
        # policy can eventually maximize the reward function."
        rho = args.kl_coef * args.kl_decay ** (update - 1)
        for _epoch in range(args.epochs):
            order = usable[torch.randperm(n_usable, device=device)]
            for s in range(0, n_usable, mb):
                idx = order[s : s + mb]
                f = features(policy, {k: v[idx] for k, v in flat.items()})
                logp, entropy = policy.evaluate(f, b_masks[idx], b_actions[idx])
                ratio_log = logp - b_logp[idx]
                ratio = ratio_log.exp()
                a = b_adv[idx]
                if not args.no_adv_norm and args.algo != "grpo":
                    # GRPO has already standardised, against the group rather
                    # than against whatever else landed in this minibatch.
                    a = (a - a.mean()) / (a.std() + 1e-8)
                pg = torch.max(-a * ratio, -a * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                ent = entropy.mean()
                v_loss = torch.zeros((), device=device)
                if args.algo == "grpo":
                    loss = pg - args.ent * ent
                else:
                    # Detached, the critic reads features the policy alone
                    # shapes, and its gradients never reach the extractor.
                    frozen = f.detach()
                    value = policy.value(frozen if args.critic_detach else f)
                    vf_clip = args.clip if args.vf_clip is None else args.vf_clip
                    v_clipped = b_val[idx] + (value - b_val[idx]).clamp(-vf_clip, vf_clip)
                    v_loss = 0.5 * torch.max(
                        (value - b_ret[idx]) ** 2, (v_clipped - b_ret[idx]) ** 2
                    ).mean()
                    loss = pg - args.ent * ent + args.vf * v_loss
                prior_kl = torch.zeros((), device=device)
                if prior is not None:
                    op = b_actions[idx][:, 0]
                    op_logits, op_mask, arg_logits, pad = policy.head_logits(
                        f, b_masks[idx], op
                    )
                    with torch.no_grad():
                        pf = features(prior, {k: v[idx] for k, v in flat.items()})
                        p_op, _, p_arg, _ = prior.head_logits(pf, b_masks[idx], op)
                    # The joint KL of a factored head is KL(op) plus the KL of
                    # the arguments under each op, weighted by the prior. The
                    # second term is taken at the op the rollout actually chose,
                    # which is one sample of that expectation.
                    prior_kl = (
                        masked_kl(p_op, op_logits, op_mask)
                        + masked_kl(p_arg, arg_logits, pad).sum(-1)
                    ).mean()
                    loss = loss + rho * prior_kl
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()
                for _ in range(args.critic_updates - 1 if critic_opt is not None else 0):
                    # SAO's decoupled frequency: the critic sees the same
                    # minibatch again on the features the trunk already
                    # produced, so an extra step costs a head pass, not a
                    # forward through the extractor. Unclipped, because the
                    # reference it would clip against is this same update's.
                    extra = 0.5 * ((policy.value(frozen) - b_ret[idx]) ** 2).mean()
                    critic_opt.zero_grad(set_to_none=True)
                    extra.backward()
                    nn.utils.clip_grad_norm_(policy.value_head.parameters(), args.max_grad_norm)
                    critic_opt.step()
                with torch.no_grad():
                    stats["pg"].append(pg.detach())
                    stats["v"].append(v_loss.detach())
                    stats["ent"].append(ent.detach())
                    stats["kl"].append(((ratio - 1) - ratio_log).mean())
                    stats["clipfrac"].append(((ratio - 1).abs() > args.clip).float().mean())
                    stats["prior_kl"].append(prior_kl.detach())

        updated = clock()
        loss_means = torch.stack([torch.stack(v).mean() for v in stats.values()]).tolist()
        elapsed = time.perf_counter() - start
        recent = [e for e in episodes[-2048:] if e["start"] == "scene"][-512:]
        demo = [e for e in episodes[-2048:] if e["start"] != "scene"][-512:]
        if args.demo_schedule == "gated":
            gate.update(demo, steps)
        row = {
            "update": update,
            "demo_window": list(env.demo_window) if env.demo_window else None,
            "gate_rung": gate.index if args.demo_schedule == "gated" else None,
            "time_rollout": round(rolled - began, 4),
            "time_update": round(updated - rolled, 4),
            "steps": steps,
            # Under GRPO the horizon is held open until the last member of a
            # group finishes, so some of the steps executed belong to no
            # episode. `decisions` is the count that actually trained.
            "decisions": decisions,
            "unfinished": unfinished,
            "group_spread": round(spread, 5),
            "sps": round(steps / elapsed),
            "lr": optimizer.param_groups[0]["lr"],
            "episodes": len(episodes),
            **dict(zip(stats, loss_means, strict=True)),
        }
        if recent:
            row.update(
                train_success=float(np.mean([e["success"] for e in recent])),
                train_verified=float(np.mean([max(0.0, e["verified_output"]) for e in recent])),
                train_return=float(np.mean([e["return"] for e in recent])),
                peak_potential=float(np.mean([e["peak_potential"] for e in recent])),
                line_built=float(np.mean([e["peak_potential"] >= 0.5 for e in recent])),
                decode_failure_rate=float(
                    np.sum([e["decode_failures"] for e in recent])
                    / max(1, np.sum([e["length"] for e in recent]))
                ),
            )
        if demo:
            row["demo_success"] = {
                stage: float(np.mean([e["success"] for e in demo if e["start"] == stage]))
                for stage in sorted({e["start"] for e in demo})
            }
        if steps >= next_eval or update == updates:
            next_eval += args.eval_every
            policy.eval()
            result = evaluate(policy, device, args, "train", args.eval_episodes, greedy=False)
            row["eval"] = result
            torch.save(policy.state_dict(), out / "last.pt")
            if result["success"] >= best:
                best = result["success"]
                torch.save(policy.state_dict(), out / "best.pt")
        log.write(json.dumps(row) + "\n")
        log.flush()
        if update % 10 == 0 or "eval" in row:
            brief = {k: row[k] for k in ("steps", "sps", "ent", "kl") if k in row}
            brief.update({k: round(row[k], 4) for k in ("train_success", "train_verified",
                          "peak_potential", "line_built") if k in row})  # fmt: skip
            if "demo_success" in row:
                brief["demo"] = {k: round(v, 3) for k, v in row["demo_success"].items()}
            if "eval" in row:
                brief["eval_success"] = row["eval"]["success"]
            print(json.dumps(brief), flush=True)

    env.close()
    log.close()
    if args.no_final:
        return 0
    # The final report: both splits, in all three decision modes, from the best
    # checkpoint. `greedy` is pure argmax and is kept because it is the one that
    # exposes cycling; `epsilon` is the honest deterministic-ish number.
    policy.load_state_dict(torch.load(out / "best.pt", weights_only=True))
    policy.eval()
    modes = (("sampled", False, 0.0), ("epsilon", True, args.eval_epsilon), ("greedy", True, 0.0))
    final = {
        f"{split}_{name}": evaluate(policy, device, args, split, 512, greedy, epsilon)
        for split in ("train", "test")
        for name, greedy, epsilon in modes
    }
    (out / "final.json").write_text(json.dumps(final, indent=2), "utf-8")
    export(policy, out / "policy.ts")
    print(json.dumps({k: v["success"] for k, v in final.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
