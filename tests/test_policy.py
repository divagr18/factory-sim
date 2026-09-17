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


def _reference(policy, features, mask, actions):
    """The per-dimension head the vectorised one replaced, kept as its spec."""
    uses = policy.uses[actions[:, 0]]
    op_mask = mask[:, :OPS]
    op_logits = policy.op_head(features).masked_fill(~op_mask, -1e8)
    logp_all = torch.log_softmax(op_logits, -1)
    logp = logp_all.gather(1, actions[:, :1]).squeeze(1)
    entropy = -torch.where(op_mask, logp_all.exp() * logp_all, torch.zeros_like(logp_all)).sum(-1)
    one_hot = torch.nn.functional.one_hot(actions[:, 0], OPS).float()
    flat = policy.arg_head(torch.cat([features, one_hot], 1))
    offset, greedy = OPS, [actions[:, 0]]
    for j, (size, logits) in enumerate(zip(NVEC[1:], torch.split(flat, NVEC[1:], 1), strict=True)):
        m = mask[:, offset : offset + size].clone()
        offset += size
        others = m[:, 1:].any(1, keepdim=True)
        m[:, :1] &= ~others
        sentinel = torch.zeros_like(m)
        sentinel[:, 0] = True
        m = torch.where(uses[:, j : j + 1], m, sentinel)
        lp = torch.log_softmax(logits.masked_fill(~m, -1e8), -1)
        logp = logp + lp.gather(1, actions[:, j + 1 : j + 2]).squeeze(1)
        entropy = entropy - torch.where(m, lp.exp() * lp, torch.zeros_like(lp)).sum(-1)
        greedy.append(lp.argmax(-1))
    return logp, entropy


def test_vectorised_head_matches_the_per_dimension_reference(batch):
    t, mask, _ = batch
    torch.manual_seed(1)
    policy = Policy()
    with torch.no_grad():
        f = policy.features(*(t[k] for k in OBS_KEYS))
        for _ in range(20):
            actions, logp = policy.act(f, mask)
            ref_logp, ref_entropy = _reference(policy, f, mask, actions)
            again, entropy = policy.evaluate(f, mask, actions)
            assert torch.allclose(logp, ref_logp, atol=1e-5)
            assert torch.allclose(again, ref_logp, atol=1e-5)
            assert torch.allclose(entropy, ref_entropy, atol=1e-4)


def test_gumbel_sampling_follows_the_softmax():
    from fsim.policy import _gumbel_argmax

    torch.manual_seed(2)
    logits = torch.tensor([[0.0, 1.0, -1e8, 2.0]]).repeat(200_000, 1)
    counts = torch.bincount(_gumbel_argmax(logits), minlength=4).float() / len(logits)
    expected = torch.softmax(logits[0], -1)
    assert counts[2] == 0
    assert torch.allclose(counts, expected, atol=0.005)


@pytest.fixture(scope="module")
def batch_v2():
    env = VecEnv(8, threads=2, demo_starts=1.0, action_space="v2", seed=5)
    obs, masks = env.reset()
    out = (
        {k: torch.from_numpy(obs[k].copy()) for k in OBS_KEYS},
        torch.from_numpy(masks.copy()).bool(),
    )
    env.close()
    return out


def test_v2_actions_respect_masks_and_scores_match(batch_v2, tmp_path):
    t, mask = batch_v2
    torch.manual_seed(3)
    policy = Policy(action_space="v2").eval()
    offsets = np.cumsum((0,) + NVEC)
    with torch.no_grad():
        f = policy.features(*(t[k] for k in OBS_KEYS))
        assert f.shape[1] == 256 + 32 * 64 + 6 * 13 * 13
        for _ in range(20):
            actions, logp = policy.act(f, mask)
            again, entropy = policy.evaluate(f, mask, actions)
            assert torch.allclose(logp, again, atol=1e-5)
            assert torch.isfinite(entropy).all()
            for row, a in zip(mask, actions, strict=True):
                for d in range(6):
                    assert row[offsets[d] + a[d]]
    export(policy, tmp_path / "v2.ts")
    loaded = torch.jit.load(str(tmp_path / "v2.ts"))
    with torch.no_grad():
        greedy, _ = policy.act(f, mask, True)
        assert torch.equal(loaded(*(t[k] for k in OBS_KEYS), mask, True), greedy)


def test_v2_target_scores_follow_the_rows(batch_v2):
    """Swapping two entity rows swaps their target scores: a pointer, not a slot."""
    t, mask = batch_v2
    torch.manual_seed(4)
    policy = Policy(action_space="v2").eval()
    counts = t["entity_mask"].sum(1)
    assert (counts >= 2).any()
    i = int(torch.nonzero(counts >= 2)[0])
    swapped = {k: v.clone() for k, v in t.items()}
    swapped["entities"][:, [0, 1]] = swapped["entities"][:, [1, 0]]
    op = torch.full((len(mask),), 16, dtype=torch.long)  # give_to
    with torch.no_grad():
        a, _ = policy._arguments(policy.features(*(t[k] for k in OBS_KEYS)), mask, op)
        b, _ = policy._arguments(policy.features(*(swapped[k] for k in OBS_KEYS)), mask, op)
    # the pooled context is permutation-invariant, so only the pointer moves
    assert torch.allclose(a[i, 0, 1], b[i, 0, 2], atol=1e-5)
    assert torch.allclose(a[i, 0, 2], b[i, 0, 1], atol=1e-5)


def test_v1_checkpoints_still_load():
    state = Policy().state_dict()
    Policy(action_space="v1").load_state_dict(state)
