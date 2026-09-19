"""The critic's gradients, and where they are allowed to go.

SAO (Hou et al. 2026, arXiv:2607.07508) keeps the critic off the shared trunk
-- it freezes attention under the value model because those gradients are what
destabilised full-parameter training. We have the same shape: `Policy.value`
reads a slice of the extractor's features, so by default the value loss trains
the extractor too. Here `--vf 2.0` froze three runs, which is why the option to
separate them exists.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from fsim.policy import Policy  # noqa: E402


def _batch(n: int = 4):
    return (
        torch.zeros(n, 6, 65, 65),
        torch.zeros(n, 32, 16),
        torch.zeros(n, 32, dtype=torch.int8),
        torch.zeros(n, 12),
        torch.zeros(n, 14),
        torch.zeros(n, 12),
    )


def _value_backward(policy: Policy, detach: bool) -> tuple[float, float]:
    """-> (total |grad| in the extractor, total |grad| in the value head)."""
    policy.zero_grad(set_to_none=True)
    features = policy.extractor(*_batch())
    value = policy.value(features.detach() if detach else features)
    (0.5 * ((value - torch.ones(4)) ** 2).mean()).backward()

    def total(module):
        return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)

    return total(policy.extractor), total(policy.value_head)


def test_the_value_loss_trains_the_extractor_by_default():
    """Stated so the default is a choice rather than an accident."""
    trunk, head = _value_backward(Policy(action_space="v2"), detach=False)
    assert trunk > 0.0
    assert head > 0.0


def test_detaching_keeps_the_critic_off_the_trunk():
    policy = Policy(action_space="v2")
    attached_trunk, attached_head = _value_backward(policy, detach=False)
    detached_trunk, detached_head = _value_backward(policy, detach=True)

    assert detached_trunk == 0.0, detached_trunk
    # The critic itself is unaffected -- it learns exactly as much either way,
    # which is the point: what changes is only who else the gradient reaches.
    assert detached_head == pytest.approx(attached_head, rel=1e-6)
    assert attached_trunk > 0.0


def test_extra_critic_steps_cannot_move_the_policy():
    """The decoupled-frequency optimizer holds the value head's parameters and
    nothing else, so running the critic more often than the policy is a claim
    about the critic only."""
    policy = Policy(action_space="v2")
    critic_opt = torch.optim.Adam(policy.value_head.parameters(), lr=1e-2)
    before = [p.detach().clone() for p in policy.extractor.parameters()]

    features = policy.extractor(*_batch()).detach()
    for _ in range(3):
        loss = 0.5 * ((policy.value(features) - torch.ones(4)) ** 2).mean()
        critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        critic_opt.step()

    for was, now in zip(before, policy.extractor.parameters(), strict=True):
        assert torch.equal(was, now)
