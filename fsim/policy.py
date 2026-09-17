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

import copy

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


EXTRACTOR_VERSION = 2  # 2: the grid is read at 1/255 resolution

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
        #: The dtype the grid is materialised in; a trainer running the
        #: extractor under bf16 autocast sets bf16, so the cast happens once.
        self.input_dtype: torch.dtype = torch.float32

    def _grid(self, grid: torch.Tensor) -> torch.Tensor:
        """`grid_net`, with its first layer computed as what it is.

        A 4x4 convolution with stride 4 on a 65x65 grid reads the top-left 64x64
        in disjoint 4x4 patches: it is a linear map of `pixel_unshuffle(4)`,
        with the same weights. As a matrix multiply it measured 25 ms against
        cuDNN's 33 ms for the grid path's forward and backward at 4096 samples,
        and its output is already in channels-last layout, which the second
        convolution runs fastest in. Same parameters, same numbers.
        """
        first = self.grid_net[0]
        patches = torch.nn.functional.pixel_unshuffle(grid[:, :, :64, :64], 4)
        h = torch.nn.functional.linear(
            patches.permute(0, 2, 3, 1), first.weight.flatten(1), first.bias
        ).permute(0, 3, 1, 2)
        for index, layer in enumerate(self.grid_net):
            if index >= 1:
                h = layer(h)
        return h

    def forward(self, grid, entities, entity_mask, self_, inventory, goal):
        # Every grid value is read at 1/255 resolution, so the float grid the
        # engine's encoder produces and the byte grid a trainer ships to the GPU
        # (`fsim_obs8`, round(255 * value)) are the same input. A byte grid goes
        # straight to the compute dtype: at a 4096-sample minibatch the grid is
        # 104M elements, and every extra full-size pass over it showed up in the
        # update's wall time.
        if grid.dtype == torch.uint8:
            grid = grid.to(self.input_dtype)
        else:
            grid = torch.round(grid * 255.0).to(self.input_dtype)
        grid = grid.div_(255.0)
        g = self._grid(grid)
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


def _layout() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Where each of the 179 argument entries sits in a (5, 122) padded grid.

    Returns the flat scatter index into 5 * 122, the argument dimension of
    each entry, and which entries are a dimension's UNUSED sentinel (index 0).
    """
    width = max(ARG_NVEC)
    index, dim, sentinel = [], [], []
    for j, size in enumerate(ARG_NVEC):
        for k in range(size):
            index.append(j * width + k)
            dim.append(j)
            sentinel.append(k == 0)
    return torch.tensor(index), torch.tensor(dim), torch.tensor(sentinel)


def _gumbel_argmax(logits: torch.Tensor) -> torch.Tensor:
    """A sample from softmax(logits): argmax(logits + Gumbel noise).

    Same distribution as `torch.multinomial(softmax(logits))`, but with no
    host synchronisation and no data-dependent kernel, so a rollout step can
    be captured in a CUDA graph. Masked entries sit at -1e8 and never win.
    """
    u = torch.rand_like(logits).clamp_(1e-10, 1.0 - 1e-7)
    return (logits - torch.log(-torch.log(u))).argmax(-1)


class Policy(nn.Module):
    def __init__(self, features_dim: int = 256) -> None:
        super().__init__()
        self.extractor = Extractor(features_dim)
        self.op_head = nn.Sequential(_layer(features_dim, 256), nn.ReLU(), _layer(256, OPS, 0.01))
        self.arg_head = nn.Sequential(
            _layer(features_dim + OPS, 256), nn.ReLU(), _layer(256, sum(ARG_NVEC), 0.01)
        )
        self.value_head = nn.Sequential(_layer(features_dim, 256), nn.ReLU(), _layer(256, 1, 1.0))
        index, dim, sentinel = _layout()
        self.register_buffer("uses", argument_uses(), persistent=False)
        self.register_buffer("pad_index", index, persistent=False)
        self.register_buffer("pad_dim", dim, persistent=False)
        self.register_buffer("pad_sentinel", sentinel, persistent=False)
        self.arg_sizes: list[int] = list(ARG_NVEC)
        self.arg_width: int = max(ARG_NVEC)
        self.ops: int = OPS
        self.masked: float = MASKED

    def features(self, grid, entities, entity_mask, self_, inventory, goal):
        return self.extractor(grid, entities, entity_mask, self_, inventory, goal)

    def value(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_head(features).squeeze(-1)

    def _op_logits(self, features: torch.Tensor, mask: torch.Tensor):
        op_mask = mask[:, : self.ops]
        logits = self.op_head(features)
        return torch.where(op_mask, logits, torch.full_like(logits, self.masked)), op_mask

    def _arguments(self, features: torch.Tensor, mask: torch.Tensor, op: torch.Tensor):
        """Masked argument logits and masks for the chosen ops, as (B, 5, 122).

        An argument the op does not read may only be its sentinel; one it does
        read may not be, while anything else in its dimension is legal.
        Padding is masked. One pass for all five dimensions.
        """
        batch = features.shape[0]
        seg = mask[:, self.ops :]  # B x 179
        used = self.uses[op][:, self.pad_dim]  # B x 179
        real = seg & ~self.pad_sentinel
        others = torch.zeros(batch, len(self.arg_sizes), device=seg.device, dtype=features.dtype)
        others = others.index_add(1, self.pad_dim, real.to(features.dtype)) > 0
        narrowed = seg & ~(self.pad_sentinel & others[:, self.pad_dim])
        allowed = torch.where(used, narrowed, self.pad_sentinel.expand_as(seg))

        one_hot = torch.nn.functional.one_hot(op, self.ops).to(features.dtype)
        flat = self.arg_head(torch.cat([features, one_hot], dim=1))  # B x 179
        size = len(self.arg_sizes) * self.arg_width
        logits = torch.full((batch, size), self.masked, device=flat.device, dtype=flat.dtype)
        logits = logits.index_copy(1, self.pad_index, torch.where(allowed, flat, logits[:, :1]))
        pad_mask = torch.zeros(batch, size, device=seg.device, dtype=torch.bool)
        pad_mask = pad_mask.index_copy(1, self.pad_index, allowed)
        shape = (batch, len(self.arg_sizes), self.arg_width)
        return logits.view(shape), pad_mask.view(shape)

    def act(self, features: torch.Tensor, mask: torch.Tensor, greedy: bool = False):
        """-> actions (B x 6, int64), log-probabilities (B)."""
        op_logits, _ = self._op_logits(features, mask)
        op = op_logits.argmax(-1) if greedy else _gumbel_argmax(op_logits)
        logits, _ = self._arguments(features, mask, op)
        args = logits.argmax(-1) if greedy else _gumbel_argmax(logits)
        logp = torch.log_softmax(op_logits, -1).gather(1, op.unsqueeze(1)).squeeze(1)
        arg_logp = torch.log_softmax(logits, -1).gather(2, args.unsqueeze(2)).squeeze(2)
        return torch.cat([op.unsqueeze(1), args], dim=1), logp + arg_logp.sum(1)

    def evaluate(self, features: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor):
        """-> log-probabilities and entropies (B) of stored actions."""
        op = actions[:, 0]
        op_logits, op_mask = self._op_logits(features, mask)
        op_logp = torch.log_softmax(op_logits, -1)
        logp = op_logp.gather(1, op.unsqueeze(1)).squeeze(1)
        zero = torch.zeros_like(op_logp)
        entropy = -torch.where(op_mask, op_logp.exp() * op_logp, zero).sum(-1)
        logits, pad_mask = self._arguments(features, mask, op)
        arg_logp = torch.log_softmax(logits, -1)
        logp = logp + arg_logp.gather(2, actions[:, 1:].unsqueeze(2)).squeeze(2).sum(1)
        p_log_p = torch.where(pad_mask, arg_logp.exp() * arg_logp, torch.zeros_like(arg_logp))
        return logp, entropy - p_log_p.sum((1, 2))


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
    module = Exported(copy.deepcopy(policy)).cpu().eval()
    module.policy.extractor.input_dtype = torch.float32
    scripted = torch.jit.script(module)
    scripted.save(str(path))
