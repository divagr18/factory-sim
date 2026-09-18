"""The behaviour-cloned prior, its KL, and epsilon-greedy decisions."""

from __future__ import annotations

import numpy as np
import pytest

from fsim import demos, lib
from fsim.rl import NVEC

torch = pytest.importorskip("torch")

from fsim.policy import Policy, masked_kl  # noqa: E402


def test_demonstrations_are_legal_actions_the_builder_actually_took():
    data = demos.collect(64, threads=4)
    n = data["action"].shape[0]
    assert n > 200
    assert data["action"].shape[1] == len(NVEC)
    # Every recorded decision is one the mask allowed at the time.
    rows = np.arange(n)
    offset = 0
    for j, size in enumerate(NVEC):
        chosen = data["action"][:, j]
        assert (chosen < size).all()
        assert data["mask"][rows, offset + chosen].all(), f"argument {j} was masked out"
        offset += size
    assert offset == lib.RL_MASK_SIZE


def test_masked_kl_is_zero_against_itself_and_ignores_masked_entries():
    logits = torch.randn(8, 22)
    valid = torch.zeros(8, 22, dtype=torch.bool)
    valid[:, :5] = True
    assert torch.allclose(masked_kl(logits, logits, valid), torch.zeros(8), atol=1e-6)
    other = logits.clone()
    other[:, 7:] += 9.0  # only masked entries move
    same = masked_kl(logits, other, valid)
    assert torch.allclose(same, torch.zeros(8), atol=1e-6)
    other = logits.clone()
    other[:, :5] += torch.randn(8, 5)
    assert (masked_kl(logits, other, valid) > 0).all()


def _inputs(batch=16):
    obs = (
        torch.rand(batch, 6, 65, 65),
        torch.randn(batch, 32, 16),
        torch.ones(batch, 32, dtype=torch.int8),
        torch.randn(batch, 12),
        torch.randn(batch, 14),
        torch.randn(batch, 12),
    )
    mask = torch.zeros(batch, lib.RL_MASK_SIZE, dtype=torch.bool)
    offset = 0
    for size in NVEC:
        mask[:, offset : offset + size] = True
        offset += size
    return obs, mask


def test_epsilon_zero_is_argmax_and_epsilon_one_is_not():
    torch.manual_seed(0)
    policy = Policy(action_space="v2")
    obs, mask = _inputs()
    features = policy.features(*obs)
    a, _ = policy.act(features, mask, True, 0.0)
    b, _ = policy.act(features, mask, True, 0.0)
    assert torch.equal(a, b), "a greedy decision with no epsilon is deterministic"
    torch.manual_seed(1)
    c, _ = policy.act(features, mask, True, 1.0)
    assert not torch.equal(a, c), "epsilon 1 samples every decision"


def test_the_prior_pulls_the_policy_towards_it():
    torch.manual_seed(0)
    policy = Policy(action_space="v2")
    prior = Policy(action_space="v2")
    for parameter in prior.parameters():
        parameter.requires_grad_(False)
    obs, mask = _inputs()
    op = torch.zeros(16, dtype=torch.long) + 12

    def divergence():
        with torch.no_grad():
            p_op, p_mask, p_arg, pad = prior.head_logits(prior.features(*obs), mask, op)
        q_op, _, q_arg, _ = policy.head_logits(policy.features(*obs), mask, op)
        return (masked_kl(p_op, q_op, p_mask) + masked_kl(p_arg, q_arg, pad).sum(-1)).mean()

    before = float(divergence())
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    for _ in range(20):
        loss = divergence()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert float(divergence()) < before * 0.5


def _autoregressive(batch=32):
    torch.manual_seed(0)
    policy = Policy(action_space="v2")
    policy.autoregressive = True
    obs, mask = _inputs(batch)
    return policy, policy.features(*obs), mask


def test_the_tail_moves_with_the_target_that_was_chosen():
    """The point of the whole thing: item depends on what you are giving to."""
    policy, features, mask = _autoregressive()
    flat, allowed, context = policy._arg_parts(features, mask, torch.full((32,), 16))
    a = policy._retail(features, context, flat, torch.full((32,), 1))
    b = policy._retail(features, context, flat, torch.full((32,), 7))
    head = sum(policy.arg_sizes[:2])
    assert torch.equal(a[:, :head], b[:, :head]), "target and placement must not move"
    assert not torch.allclose(a[:, head:], b[:, head:]), "the tail must move"


def test_evaluate_scores_exactly_what_act_drew():
    """Summed factors are the joint log-probability only under teacher forcing."""
    policy, features, mask = _autoregressive()
    with torch.no_grad():
        actions, logp = policy.act(features, mask)
        scored, _entropy = policy.evaluate(features, mask, actions)
    assert torch.allclose(logp, scored, atol=1e-5)


def test_the_independent_head_still_scores_itself():
    policy, features, mask = _autoregressive()
    policy.autoregressive = False
    with torch.no_grad():
        actions, logp = policy.act(features, mask)
        scored, _entropy = policy.evaluate(features, mask, actions)
    assert torch.allclose(logp, scored, atol=1e-5)


def test_conditioning_leaves_the_action_legal():
    policy, features, mask = _autoregressive(64)
    with torch.no_grad():
        actions, _ = policy.act(features, mask)
    offset = 0
    for j, size in enumerate(NVEC):
        assert (actions[:, j] < size).all()
        assert mask[torch.arange(64), offset + actions[:, j]].all()
        offset += size
