"""The policy: a feature extractor over the local-v1 tensors, and an
op-conditioned action head.

The action is `MultiDiscrete[22, 33, 122, 5, 15, 4]`: an operation and five
arguments. Sampling is two-stage. The operation is drawn from the masked
operation logits; the argument head then sees the chosen operation (one-hot)
and draws each argument from a mask narrowed to that operation:

* an argument the operation does not read may only be the UNUSED sentinel 0,
  so it contributes log-probability 0 and entropy 0 -- no gradient noise and no
  entropy bonus from five dimensions the environment ignores;
* an argument it does read may not be 0 while anything else is legal, since 0
  there is a guaranteed decode failure.

Every masked logit is set to -1e8 before the softmax, and the same masked
distribution is used to sample and to score, which is what makes the masked
policy gradient valid (Huang & Ontanon 2020, arXiv:2006.14171). Masked entries
are dropped from the entropy rather than multiplied by -1e8, as in CleanRL's
`ppo_multidiscrete_mask.py`. Gym-muRTS (arXiv:2105.13807) measured that masking
the action type alone and leaving argument logits unmasked collapses
performance, which is why every argument is masked here.

The union masks the environment supplies are the only runtime input the head
needs; which operation reads which argument is a static table of the catalog.
So a policy exported from here runs unchanged against FactorioRL's real-engine
environment, whose `action_masks()` is the same flat 201-entry vector.
"""

from __future__ import annotations

import torch
from torch import nn

NVEC = (22, 33, 122, 5, 15, 4)
ARG_NVEC = NVEC[1:]
OPS = NVEC[0]
MASKED = -1e8

#: parameterized-v1: which of (target, placement, direction, item, amount)
#: each operation reads. Moves (0-11), set_recipe/craft/cancel (18-20, never
#: legal here) and wait (21) read none.
_PLACE, _MINE, _ROTATE, _ROTATE_REVERSE, _GIVE, _TAKE = 12, 13, 14, 15, 16, 17


def argument_uses() -> torch.Tensor:
    uses = torch.zeros(OPS, len(ARG_NVEC), dtype=torch.bool)
    uses[_PLACE, [1, 2, 3]] = True
    for op in (_MINE, _ROTATE, _ROTATE_REVERSE):
        uses[op, 0] = True
    for op in (_GIVE, _TAKE):
        uses[op, [0, 3, 4]] = True
    return uses


EXTRACTOR_VERSION = 1

#: The placement choices cover the 11x11 tiles around the character; rows and
#: columns 26..38 of the grid hold them whichever way the character's position
#: rounds (a tile lands in column dx+32 or dx+33).


class Extractor(nn.Module):
    """Grid + entities + vectors -> one feature vector.

    FactorioRL's `FactorioExtractor` (v7) spends three 3x3 convolutions on the
    whole 65x65 grid, which measured 34 ms per 4096-sample forward pass on the
    development GPU and was most of the training step. This one sees the whole
    grid through a 4x4-patch convolution (a coarse map: where the patch and the
    water are) and the 13x13 cells under the placement choices exactly, through
    a linear layer -- a drill needs ore under it, and that is decided there.
    Entity and vector encoders are `FactorioExtractor`'s.
    """

    def __init__(self, features_dim: int = 256) -> None:
        super().__init__()
        self.grid_net = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
        )
        self.crop_net = nn.Sequential(nn.Flatten(), nn.Linear(6 * 13 * 13, 128), nn.ReLU())
        self.entity_net = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.vector_net = nn.Sequential(
            nn.Linear(12 + 14 + 12, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 128 + 64 + 64 + 128, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, grid, entities, entity_mask, self_, inventory, goal):
        g = self.grid_net(grid)
        c = self.crop_net(grid[:, :, 26:39, 26:39])
        e = self.entity_net(entities)
        mask = entity_mask.float().unsqueeze(-1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        mean_pool = (e * mask).sum(dim=1) / counts
        max_pool = e.masked_fill(mask == 0, -1e9).max(dim=1).values
        max_pool = torch.nan_to_num(max_pool, neginf=0.0)
        v = self.vector_net(torch.cat([self_, inventory, goal], dim=1))
        return self.head(torch.cat([g, c, mean_pool, max_pool, v], dim=1))


def _layer(i: int, o: int, std: float = 2**0.5) -> nn.Linear:
    layer = nn.Linear(i, o)
    nn.init.orthogonal_(layer.weight, std)
    nn.init.zeros_(layer.bias)
    return layer


def _masked_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=-1)
    p_log_p = torch.where(mask, logp.exp() * logp, torch.zeros_like(logp))
    return -p_log_p.sum(-1)


class Policy(nn.Module):
    def __init__(self, features_dim: int = 256) -> None:
        super().__init__()
        self.extractor = Extractor(features_dim)
        self.op_head = nn.Sequential(_layer(features_dim, 256), nn.ReLU(), _layer(256, OPS, 0.01))
        self.arg_head = nn.Sequential(
            _layer(features_dim + OPS, 256), nn.ReLU(), _layer(256, sum(ARG_NVEC), 0.01)
        )
        self.value_head = nn.Sequential(_layer(features_dim, 256), nn.ReLU(), _layer(256, 1, 1.0))
        self.register_buffer("uses", argument_uses(), persistent=False)
        self.arg_sizes: list[int] = list(ARG_NVEC)
        self.ops: int = OPS
        self.masked: float = MASKED

    def features(self, grid, entities, entity_mask, self_, inventory, goal):
        return self.extractor(grid, entities, entity_mask, self_, inventory, goal)

    def value(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_head(features).squeeze(-1)

    def _argument_masks(self, mask: torch.Tensor, op: torch.Tensor) -> list[torch.Tensor]:
        uses = self.uses[op]  # B x 5
        out: list[torch.Tensor] = []
        offset = self.ops
        for j, size in enumerate(self.arg_sizes):
            m = mask[:, offset : offset + size]
            offset += size
            used = uses[:, j : j + 1]
            others = m[:, 1:].any(dim=1, keepdim=True)
            sentinel_only = torch.zeros_like(m)
            sentinel_only[:, 0] = True
            narrowed = m.clone()
            narrowed[:, :1] = m[:, :1] & ~others
            out.append(torch.where(used, narrowed, sentinel_only))
        return out

    def _argument_logits(self, features: torch.Tensor, op: torch.Tensor) -> list[torch.Tensor]:
        one_hot = torch.nn.functional.one_hot(op, self.ops).to(features.dtype)
        logits = self.arg_head(torch.cat([features, one_hot], dim=1))
        return list(torch.split(logits, self.arg_sizes, dim=1))

    def act(self, features: torch.Tensor, mask: torch.Tensor, greedy: bool = False):
        """-> actions (B x 6, int64), log-probabilities (B)."""
        op_mask = mask[:, : self.ops]
        op_logits = self.op_head(features)
        op_logits = torch.where(op_mask, op_logits, torch.full_like(op_logits, self.masked))
        if greedy:
            op = op_logits.argmax(-1)
        else:
            op = torch.multinomial(torch.softmax(op_logits, -1), 1).squeeze(-1)
        logp = torch.log_softmax(op_logits, -1).gather(1, op.unsqueeze(1)).squeeze(1)
        actions = [op]
        masks = self._argument_masks(mask, op)
        for logits, m in zip(self._argument_logits(features, op), masks):  # noqa: B905 (TorchScript has no strict=)
            logits = torch.where(m, logits, torch.full_like(logits, self.masked))
            if greedy:
                a = logits.argmax(-1)
            else:
                a = torch.multinomial(torch.softmax(logits, -1), 1).squeeze(-1)
            logp = logp + torch.log_softmax(logits, -1).gather(1, a.unsqueeze(1)).squeeze(1)
            actions.append(a)
        return torch.stack(actions, dim=1), logp

    def evaluate(self, features: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor):
        """-> log-probabilities and entropies (B) of stored actions."""
        op = actions[:, 0]
        op_mask = mask[:, : self.ops]
        op_logits = self.op_head(features)
        op_logits = torch.where(op_mask, op_logits, torch.full_like(op_logits, self.masked))
        logp = torch.log_softmax(op_logits, -1).gather(1, op.unsqueeze(1)).squeeze(1)
        entropy = _masked_entropy(op_logits, op_mask)
        masks = self._argument_masks(mask, op)
        for j, (logits, m) in enumerate(zip(self._argument_logits(features, op), masks)):  # noqa: B905
            logits = torch.where(m, logits, torch.full_like(logits, self.masked))
            a = actions[:, j + 1]
            logp = logp + torch.log_softmax(logits, -1).gather(1, a.unsqueeze(1)).squeeze(1)
            entropy = entropy + _masked_entropy(logits, m)
        return logp, entropy


class Exported(nn.Module):
    """What a checkpoint exports: observation tensors and mask in, an action out.

    TorchScript, so FactorioRL can run it with torch alone and no import of
    this package. Inputs are the `local-v1` tensors, batched, as float32 (the
    entity mask as any integer type) and the flat 201-entry mask as bool.
    """

    def __init__(self, policy: Policy) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, grid, entities, entity_mask, self_, inventory, goal, mask, greedy: bool):
        features = self.policy.features(grid, entities, entity_mask, self_, inventory, goal)
        actions, _ = self.policy.act(features, mask, greedy)
        return actions


def export(policy: Policy, path) -> None:
    module = Exported(policy).cpu().eval()
    scripted = torch.jit.script(module)
    scripted.save(str(path))
