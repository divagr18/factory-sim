"""The policy's masking. Skipped where torch is not installed (CI)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fsim.policy import OPS, Policy, argument_uses, export  # noqa: E402
from fsim.rl import NVEC  # noqa: E402
from fsim.vec import OBS_KEYS, VecEnv  # noqa: E402


@pytest.fixture(scope="module")
def batch():
    env = VecEnv(16, threads=2)
    obs, masks = env.reset()
    rng = np.random.default_rng(0)
    policy = Policy()
    for _ in range(40):  # walk a little so some envs see targets
        t = {k: torch.from_numpy(obs[k].copy()) for k in OBS_KEYS}
        f = policy.features(*(t[k] for k in OBS_KEYS))
        actions, _ = policy.act(f, torch.from_numpy(masks.copy()).bool())
        obs, masks, *_ = env.step(actions.numpy())
    t = {k: torch.from_numpy(obs[k].copy()) for k in OBS_KEYS}
    out = t, torch.from_numpy(masks.copy()).bool(), rng
    env.close()
    return out


def test_sampled_actions_respect_masks_and_argument_use(batch):
    t, mask, _ = batch
    torch.manual_seed(0)
    policy = Policy()
    uses = argument_uses()
    with torch.no_grad():
        f = policy.features(*(t[k] for k in OBS_KEYS))
        for _ in range(50):
            actions, _ = policy.act(f, mask)
            offsets = np.cumsum((0,) + NVEC)
            for row, a in zip(mask, actions, strict=True):
                for d in range(6):
                    assert row[offsets[d] + a[d]], "sampled a masked entry"
                op = int(a[0])
                for j in range(5):
                    if uses[op, j]:
                        assert a[j + 1] != 0, "used argument left at UNUSED"
                    else:
                        assert a[j + 1] == 0, "unused argument not UNUSED"


def test_evaluate_scores_what_act_sampled(batch):
    t, mask, _ = batch
    policy = Policy()
    with torch.no_grad():
        f = policy.features(*(t[k] for k in OBS_KEYS))
        actions, logp = policy.act(f, mask)
        again, entropy = policy.evaluate(f, mask, actions)
    assert torch.allclose(logp, again, atol=1e-5)
    assert torch.isfinite(entropy).all() and (entropy >= 0).all()


def test_ops_that_read_nothing_contribute_only_their_own_probability(batch):
    t, mask, _ = batch
    policy = Policy()
    with torch.no_grad():
        f = policy.features(*(t[k] for k in OBS_KEYS))
        wait = torch.zeros(len(f), 6, dtype=torch.long)
        wait[:, 0] = OPS - 1
        logp, _ = policy.evaluate(f, mask, wait)
        op_logits = policy.op_head(f).masked_fill(~mask[:, :OPS], -1e8)
        expected = torch.log_softmax(op_logits, -1)[:, OPS - 1]
    assert torch.allclose(logp, expected, atol=1e-5)


def test_export_round_trips(batch, tmp_path):
    t, mask, _ = batch
    policy = Policy().eval()
    export(policy, tmp_path / "policy.ts")
    loaded = torch.jit.load(str(tmp_path / "policy.ts"))
    args = [t[k] for k in OBS_KEYS]
    with torch.no_grad():
        f = policy.features(*args)
        greedy, _ = policy.act(f, mask, True)
        assert torch.equal(loaded(*args, mask, True), greedy)
