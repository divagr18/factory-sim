"""The v3 policy head under per-operation masks (user decision "v3 masks: per
operation", option C). Skipped where torch is not installed (CI).

After the operation is sampled, its arguments are drawn under that operation's
own row of `RlEnv.op_masks()`, and `evaluate` scores a stored action under the
row of the operation it stores -- so the PPO ratio is the ratio of two
policies' probabilities of the action under the conditional masks it was
sampled under.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fsim.parity import GOLDEN  # noqa: E402
from fsim.policy import NVEC3, Policy, export  # noqa: E402
from fsim.rl import RlEnv  # noqa: E402
from fsim.trace import read_trace  # noqa: E402

KEYS = ("grid", "entities", "entity_mask", "self", "inventory", "goal")
OPS = NVEC3[0]
ARG = NVEC3[1:]
#: Scenes with most v3 verbs legal somewhere: burners with fuel, chests, ore.
SCENES = ("v3_take_fuel", "v3_finish", "hand_mine_contents", "hand_mine_spills")


def _envs() -> list[RlEnv]:
    out = []
    for name in SCENES:
        header, _ = read_trace(GOLDEN / f"{name}.jsonl.xz")
        env = RlEnv()
        env.reset(
            header["task"],
            header["blueprint"],
            decision_ticks=header["decision_ticks"],
            max_steps=header["max_decision_steps"],
            construction_tick_limit=header["construction_tick_limit"],
            action_space="v3",
        )
        out.append(env)
    return out


def _batch(envs):
    obs = [env.observe3()[0] for env in envs]
    tensors = [torch.from_numpy(np.stack([o[k].copy() for o in obs])) for k in KEYS]
    mask = torch.from_numpy(np.stack([env.observe3()[1].copy() for env in envs])).bool()
    op_masks = torch.from_numpy(np.stack([env.op_masks().copy() for env in envs])).bool()
    return tensors, mask, op_masks


def _reference(policy, features, mask, op_masks, actions):
    """log p(action) the long way: the operation under the flat mask, then each
    argument dimension under the stored operation's own row."""
    op = actions[:, 0]
    op_logits = policy.op_head(features[:, : policy.features_dim]).masked_fill(~mask[:, :OPS], -1e8)
    logp = torch.log_softmax(op_logits, -1).gather(1, op[:, None]).squeeze(1)
    flat, _allowed, _ = policy._arg_parts(features, mask, op, op_masks)
    rows = op_masks[torch.arange(len(op)), op]
    offset = 0
    for d, size in enumerate(ARG):
        legal = rows[:, offset : offset + size]
        logits = flat[:, offset : offset + size].masked_fill(~legal, -1e8)
        logp = logp + torch.log_softmax(logits, -1).gather(1, actions[:, d + 1 : d + 2]).squeeze(1)
        offset += size
    return logp


def test_arguments_come_from_the_sampled_operations_row():
    envs = _envs()
    tensors, mask, op_masks = _batch(envs)
    torch.manual_seed(0)
    policy = Policy(action_space="v3")
    offsets = np.cumsum((0, *ARG))
    seen_ops = set()
    with torch.no_grad():
        f = policy.features(*tensors)
        for _ in range(200):
            actions, _ = policy.act(f, mask, op_masks=op_masks)
            for b, a in enumerate(actions):
                op = int(a[0])
                seen_ops.add(op)
                assert mask[b, op], "sampled an illegal operation"
                for d in range(5):
                    assert op_masks[b, op, offsets[d] + int(a[d + 1])], (op, d, int(a[d + 1]))
    # take_fuel, give_to, take_from and mine_tile all drew arguments here.
    assert {16, 17, 22, 23} <= seen_ops


def test_the_ratio_is_taken_under_the_conditional_masks():
    envs = _envs()
    tensors, mask, op_masks = _batch(envs)
    torch.manual_seed(1)
    old, new = Policy(action_space="v3"), Policy(action_space="v3")
    new.load_state_dict(old.state_dict())
    with torch.no_grad():
        for p in new.parameters():
            p.add_(0.05 * torch.randn_like(p))
        f_old = old.features(*tensors)
        f_new = new.features(*tensors)
        differs = 0
        for _ in range(50):
            actions, logp = old.act(f_old, mask, op_masks=op_masks)
            again, entropy = old.evaluate(f_old, mask, actions, op_masks)
            # The same parameters: ratio 1.
            assert torch.allclose(logp, again, atol=1e-5)
            assert torch.isfinite(entropy).all() and (entropy >= 0).all()
            assert torch.allclose(again, _reference(old, f_old, mask, op_masks, actions), atol=1e-4)
            # New parameters: the ratio is exactly the conditional-mask one.
            new_logp, _ = new.evaluate(f_new, mask, actions, op_masks)
            reference = _reference(new, f_new, mask, op_masks, actions)
            assert torch.allclose(new_logp, reference, atol=1e-4)
            # Scored under the union instead, the same action has another
            # probability: the ratio would compare two different distributions.
            union, _ = old.evaluate(f_old, mask, actions)
            differs += int((union - logp).abs().gt(1e-4).any())
        assert differs > 0


def test_sampled_actions_always_decode():
    """Every action drawn under the per-operation masks is one the decoder takes."""
    envs = _envs()
    torch.manual_seed(2)
    policy = Policy(action_space="v3")
    failures = {"op_masks": 0, "union": 0}
    with torch.no_grad():
        for kind in failures:
            envs = _envs()
            for _ in range(60):
                tensors, mask, op_masks = _batch(envs)
                f = policy.features(*tensors)
                actions, _ = policy.act(f, mask, op_masks=op_masks if kind == "op_masks" else None)
                for env, a in zip(envs, actions.numpy(), strict=True):
                    if env.rl.done:
                        continue
                    _, _, _, _, info = env.step(a)
                    failures[kind] += int(info["decode_failure"])
    assert failures["op_masks"] == 0
    # The flat union offers arguments another operation could use.
    assert failures["union"] > 0


def test_v3_export_takes_the_operation_masks(tmp_path):
    envs = _envs()
    tensors, mask, op_masks = _batch(envs)
    policy = Policy(action_space="v3").eval()
    export(policy, tmp_path / "policy.ts")
    loaded = torch.jit.load(str(tmp_path / "policy.ts"))
    with torch.no_grad():
        f = policy.features(*tensors)
        greedy, _ = policy.act(f, mask, True, op_masks=op_masks)
        assert torch.equal(loaded(*tensors, mask, True, op_masks), greedy)
