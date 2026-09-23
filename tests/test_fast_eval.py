"""The fast evaluator measures what the original one did.

`train.evaluate` picks `best.pt`, so the batched, graph-captured path must run
the same episodes as `--slow-eval`: evaluation seeds 0 .. episodes - 1, in any
round size, and -- in greedy mode, where nothing is drawn -- the same outcomes.
"""

from __future__ import annotations

import numpy as np
import pytest

from fsim.vec import VecEnv

torch = pytest.importorskip("torch")


def _setup(max_steps: int = 12):
    import train
    from fsim.policy import Policy

    args = train.parse(
        ["--run", "x", "--seed", "3", "--max-steps", str(max_steps), "--threads", "4"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    policy = Policy(action_space=args.action_space).to(device)
    if device.type == "cuda":
        policy.extractor.input_dtype = torch.bfloat16
        policy = policy.to(memory_format=torch.channels_last)
    policy.autoregressive = args.autoregressive
    policy.eval()
    return train, args, device, policy


@pytest.fixture
def seeded(monkeypatch):
    """Tag each finished episode with the evaluation seed it ran."""
    original = VecEnv.step

    def step(self, actions):
        before = list(self._seed_of)
        out = original(self, actions)
        for i, record in zip(np.flatnonzero(out[3] | out[4]), out[5], strict=True):
            record["seed"] = before[i] - self.seed_base
        return out

    monkeypatch.setattr(VecEnv, "step", step)


def _outcome(r):
    return (r["family"], r["success"], r["length"], r["return"], r["peak_potential"])


@pytest.mark.parametrize("batch", [512, 48])
def test_fast_eval_runs_the_slow_evals_episodes(seeded, monkeypatch, batch):
    train, args, device, policy = _setup()
    # 48 gives rounds of 48, 48 and a short 4: the round offsets must still
    # land every seed exactly once.
    monkeypatch.setattr(train, "EVAL_BATCH", batch)
    episodes = 100
    with torch.no_grad():
        slow = train._slow_eval_records(policy, device, args, "train", episodes, True)
        fast = train.eval_records(policy, device, args, "train", episodes, True)
    assert sorted(r["seed"] for r in slow) == list(range(episodes))
    assert [r["seed"] for r in fast] == list(range(episodes))
    slow_by_seed = {r["seed"]: _outcome(r) for r in slow}
    assert [_outcome(r) for r in fast] == [slow_by_seed[k] for k in range(episodes)]


def test_evaluate_switches_on_slow_eval(seeded):
    train, args, device, policy = _setup()
    fast = train.evaluate(policy, device, args, "test", 40, greedy=True)
    args.slow_eval = True
    slow = train.evaluate(policy, device, args, "test", 40, greedy=True)
    assert fast == slow
    assert fast["episodes"] == 40


def test_sampled_eval_covers_the_same_scenes(seeded):
    train, args, device, policy = _setup()
    with torch.no_grad():
        fast = train.eval_records(policy, device, args, "train", 70, False)
        again = train.eval_records(policy, device, args, "train", 70, False, 0.0)
    assert [r["seed"] for r in fast] == list(range(70))
    assert [r["family"] for r in fast] == [r["family"] for r in again]
