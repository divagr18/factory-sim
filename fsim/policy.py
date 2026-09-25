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

**v3: a mask per operation** (user decision "v3 masks: per operation", option
C). `action_space="v3"` is `MultiDiscrete[25, 97, 226, 5, 19, 4]` over the v3
tensors, and its environment also supplies `op_masks` (`RlEnv.op_masks()`,
FactorioRL `ParameterizedEnv.operation_masks`): per operation, its own legal
values of each argument dimension. Given them, the arguments are drawn under
the sampled operation's row instead of the union, and `evaluate` scores a
stored action under the row of the operation it stores -- the same
conditional mask it was sampled under, which is what keeps the PPO ratio the
ratio of the two policies' probabilities of that action. Without them the head
is exactly what it was.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from fsim import patchify

NVEC = (22, 33, 122, 5, 15, 4)
ARG_NVEC = NVEC[1:]
OPS = NVEC[0]
MASKED = -1e8
#: v3 (FactorioRL parameterized-v3): v1's operations, `mine_tile`,
#: `take_fuel` and `finish`, over the 96-row table and the 15x15 window.
NVEC3 = (25, 97, 226, 5, 19, 4)
_MINE_TILE, _TAKE_FUEL = 22, 23

#: parameterized-v1: which of (target, placement, direction, item, amount)
#: each operation reads. Moves (0-11), set_recipe/craft/cancel (18-20, never
#: legal here) and wait (21) read none.
_PLACE, _MINE, _ROTATE, _ROTATE_REVERSE, _GIVE, _TAKE = 12, 13, 14, 15, 16, 17


def argument_uses(action_space: str = "v1") -> torch.Tensor:
    ops = NVEC3[0] if action_space == "v3" else OPS
    uses = torch.zeros(ops, len(ARG_NVEC), dtype=torch.bool)
    uses[_PLACE, [1, 2, 3]] = True
    for op in (_MINE, _ROTATE, _ROTATE_REVERSE):
        uses[op, 0] = True
    for op in (_GIVE, _TAKE):
        uses[op, [0, 3, 4]] = True
    if action_space == "v3":
        uses[_MINE_TILE, [1, 4]] = True  # a resource tile's slot, and a count
        uses[_TAKE_FUEL, [0, 4]] = True  # a burner's row, and a count
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

    def __init__(
        self,
        features_dim: int = 256,
        entity_features: int = 16,
        vector_dim: int = 12 + 14 + 12,
        crop: int = 13,
    ) -> None:
        """The defaults are v1's; v3 reads 32 features a row, a 13 + 18 + 30
        vector and a 17x17 crop (the 15x15 window and its one-cell margin)."""
        super().__init__()
        #: The crop: `crop` cells a side from grid row and column `crop_start`,
        #: centred on the character's cell 32.
        self.crop: int = crop
        self.crop_start: int = 32 - crop // 2
        self.grid_net = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
        )
        self.crop_net = nn.Sequential(nn.Flatten(), nn.Linear(6 * crop * crop, 128), nn.ReLU())
        self.entity_net = nn.Sequential(
            nn.Linear(entity_features, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.vector_net = nn.Sequential(
            nn.Linear(vector_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 128 + 64 + 64 + 128, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )
        #: The dtype the grid is materialised in; a trainer running the
        #: extractor under bf16 autocast sets bf16, so the cast happens once.
        self.input_dtype: torch.dtype = torch.float32
        #: v2: also return the per-row entity embeddings and the placement
        #: crop, flattened after the features, for the pointer and spatial heads.
        self.expose: bool = False
        #: The fused grid prologue (`fsim/patchify.py`), looked up on the first
        #: forward that could use one so a CPU run never builds a kernel.
        self.fused = None
        self.fused_tried: bool = False

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

    @torch.jit.unused
    def _patched(self, grid: torch.Tensor):
        """`(g, crop)` from the fused kernel, or `(None, None)` without it.

        The kernel builds the projection's input straight from the bytes, so
        the whole grid is never materialised in the compute dtype -- only the
        13x13 the crop reads. What it produces is bit-identical to what the
        portable path builds; the features that come out differ in the last
        bfloat16 places all the same, because the matmul now reads a contiguous
        tensor rather than a permuted view and cuBLAS accumulates it in a
        different order. That is the size of difference a library version bump
        makes, not a change of model.
        """
        if not self.fused_tried:
            self.fused_tried = True
            if grid.is_cuda and grid.dtype == torch.uint8:
                self.fused = patchify.load()
        if self.fused is None or not grid.is_cuda or grid.dtype != torch.uint8:
            return None, None
        first = self.grid_net[0]
        patches = self.fused.patchify(grid, 4)
        batch, cells = grid.shape[0], grid.shape[2] // 4
        h = torch.nn.functional.linear(
            patches.to(self.input_dtype), first.weight.flatten(1), first.bias
        )
        h = h.view(batch, cells, cells, -1).permute(0, 3, 1, 2)
        for index, layer in enumerate(self.grid_net):
            if index >= 1:
                h = layer(h)
        a, b = self.crop_start, self.crop_start + self.crop
        crop = grid[:, :, a:b, a:b].to(self.input_dtype).div_(255.0)
        return h, crop

    def forward(self, grid, entities, entity_mask, self_, inventory, goal):
        # Every grid value is read at 1/255 resolution, so the float grid the
        # engine's encoder produces and the byte grid a trainer ships to the GPU
        # (`fsim_obs8`, round(255 * value)) are the same input. A byte grid goes
        # straight to the compute dtype: at a 4096-sample minibatch the grid is
        # 104M elements, and every extra full-size pass over it showed up in the
        # update's wall time.
        g, crop_raw = None, None
        if not torch.jit.is_scripting():  # an exported policy carries no kernel
            g, crop_raw = self._patched(grid)
        if g is None:
            if grid.dtype == torch.uint8:
                grid = grid.to(self.input_dtype)
            else:
                grid = torch.round(grid * 255.0).to(self.input_dtype)
            grid = grid.div_(255.0)
            g = self._grid(grid)
            a, b = self.crop_start, self.crop_start + self.crop
            crop_raw = grid[:, :, a:b, a:b]
        c = self.crop_net(crop_raw)
        e = self.entity_net(entities)
        mask = entity_mask.float().unsqueeze(-1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        mean_pool = (e * mask).sum(dim=1) / counts
        max_pool = e.masked_fill(mask == 0, -1e9).max(dim=1).values
        max_pool = torch.nan_to_num(max_pool, neginf=0.0)
        v = self.vector_net(torch.cat([self_, inventory, goal], dim=1))
        f = self.head(torch.cat([g, c, mean_pool, max_pool, v], dim=1))
        if not self.expose:
            return f
        rows = (e * mask).flatten(1)
        return torch.cat([f, rows, crop_raw.to(f.dtype).flatten(1)], dim=1)


def _layer(i: int, o: int, std: float = 2**0.5) -> nn.Linear:
    layer = nn.Linear(i, o)
    nn.init.orthogonal_(layer.weight, std)
    nn.init.zeros_(layer.bias)
    return layer


def _layout(arg_nvec=ARG_NVEC) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Where each of the 179 argument entries (351 under v3) sits in a (5, 122)
    padded grid ((5, 226) under v3).

    Returns the flat scatter index into 5 * width, the argument dimension of
    each entry, and which entries are a dimension's UNUSED sentinel (index 0).
    """
    width = max(arg_nvec)
    index, dim, sentinel = [], [], []
    for j, size in enumerate(arg_nvec):
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


class _Unused(nn.Module):
    """Stands in for a v2 head in a v1 policy: no parameters, never called."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a


ROW_DIM = 64
ROWS = 32
CROP = 13
PLACE_CONTEXT = 32


class PointerHead(nn.Module):
    """score(row, context) = w . relu(A row + B context + b) + c.

    The same function as an MLP on the concatenation [row, context], computed
    without repeating the 278-wide context for each of the 32 rows: at a
    4096-sample minibatch that concatenation made v2's update 2.6x slower
    than v1's.
    """

    def __init__(self, row_dim: int, context_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.rows = _layer(row_dim, hidden)
        self.context = _layer(context_dim, hidden)
        self.out = _layer(hidden, 1, 0.01)

    def forward(self, rows: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.rows(rows) + self.context(context).unsqueeze(1))
        return self.out(hidden).squeeze(-1)


class PlacementHead(nn.Module):
    """Per-tile scores over the grid crop, conditioned on the context.

    The context enters as a per-channel bias after the first convolution rather
    than as broadcast input channels, so the convolutions run on the crop's six
    planes alone. The output is the 11x11 of placement slots, not the 13x13 of
    the crop it read.
    """

    def __init__(self, context_dim: int, channels: int = PLACE_CONTEXT) -> None:
        super().__init__()
        self.first = nn.Conv2d(6, channels, 3, padding=1)
        self.context = _layer(context_dim, channels, 1.0)
        # No padding on the second convolution. Only the middle 11x11 of a
        # 13x13 crop names a placement slot, and a padded convolution's
        # interior is exactly an unpadded one's whole output -- every position
        # it keeps reads the same nine inputs -- so this is the same numbers
        # over 121 positions instead of 169.
        self.second = nn.Conv2d(channels, channels, 3)
        self.out = nn.Conv2d(channels, 1, 1)
        nn.init.orthogonal_(self.out.weight, 0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, crop: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        bias = self.context(context).unsqueeze(-1).unsqueeze(-1)
        h = torch.relu(self.first(crop) + bias)
        h = torch.relu(self.second(h))
        return self.out(h)[:, 0]


def masked_kl(prior_logits: torch.Tensor, logits: torch.Tensor, valid: torch.Tensor):
    """KL(prior || policy) over the legal actions, summed over the last axis.

    Both sides are masked here rather than trusted to arrive masked: an illegal
    entry left in the softmax moves the normaliser, and the divergence between
    two policies that agree everywhere it is possible to act would not be zero.
    """
    p = torch.log_softmax(prior_logits.masked_fill(~valid, MASKED), -1)
    q = torch.log_softmax(logits.masked_fill(~valid, MASKED), -1)
    return torch.where(valid, p.exp() * (p - q), torch.zeros_like(p)).sum(-1)


class Policy(nn.Module):
    """`action_space` "v1" is FactorioRL's parameterized-v1. "v2" is the
    simulator prototype (`csrc/fsim.h`, ACTION_SPACE_V2): its `target` names a
    row of the entity table, so it is scored by a pointer head over the rows'
    embeddings, and its `placement` names a fixed tile of the 11x11 window, so
    it is scored by a small convolution over the grid crop whose central 11x11
    cells are those tiles. "v3" is FactorioRL's parameterized-v3, v2's heads
    over the v3 tensors: 96 rows, the 15x15 window from a 17x17 crop, and the
    per-operation masks (see the module docstring)."""

    def __init__(self, features_dim: int = 256, action_space: str = "v1") -> None:
        super().__init__()
        if action_space not in ("v1", "v2", "v3"):
            raise ValueError(f"unknown action space {action_space!r}")
        self.action_space = action_space
        v3 = action_space == "v3"
        nvec = NVEC3 if v3 else NVEC
        ops = nvec[0]
        arg_nvec = nvec[1:]
        #: The pointer and placement heads (v2's, and v3's).
        self.v2: bool = action_space in ("v2", "v3")
        self.features_dim: int = features_dim
        self.rows: int = 96 if v3 else ROWS
        self.row_dim: int = ROW_DIM
        self.crop: int = 17 if v3 else CROP
        #: The placement window's side: the crop less its one-cell margin.
        self.side: int = self.crop - 2
        if v3:
            self.extractor = Extractor(features_dim, 32, 13 + 18 + 30, self.crop)
        else:
            self.extractor = Extractor(features_dim)
        self.extractor.expose = self.v2
        context = features_dim + ops
        if self.v2:
            self.target_head = PointerHead(ROW_DIM, context)
            self.place_head = PlacementHead(context)
        else:
            # Placeholders, so TorchScript compiles the v2 branch; no parameters.
            self.target_head = _Unused()
            self.place_head = _Unused()
        self.op_head = nn.Sequential(_layer(features_dim, 256), nn.ReLU(), _layer(256, ops, 0.01))
        self.arg_head = nn.Sequential(
            _layer(features_dim + ops, 256), nn.ReLU(), _layer(256, sum(arg_nvec), 0.01)
        )
        #: Autoregressive tail: direction, item and amount scored *after* the
        #: target is known, conditioned on the row the policy chose. Five
        #: arguments drawn independently have to hit a conjunction by luck --
        #: giving twenty coal to the furnace is one draw in 22 x 3 x 15 x 4,
        #: which a uniform policy finds in 15% of six-hundred-step episodes and
        #: a committed one almost never. Conditioning is how AlphaStar and
        #: Conditional Action Trees (arXiv:2104.07294) handle the same shape of
        #: action; `train.py --autoregressive` turns it on.
        self.autoregressive: bool = False
        self.tail_sizes: list[int] = list(arg_nvec[2:])
        self.after_target = nn.Sequential(
            _layer(context + ROW_DIM, 128), nn.ReLU(), _layer(128, sum(arg_nvec[2:]), 0.01)
        )
        #: Stands in for the row embedding when no target was named.
        self.absent_target = nn.Parameter(torch.zeros(ROW_DIM))
        self.value_head = nn.Sequential(_layer(features_dim, 256), nn.ReLU(), _layer(256, 1, 1.0))
        index, dim, sentinel = _layout(arg_nvec)
        self.register_buffer("uses", argument_uses(action_space), persistent=False)
        self.register_buffer("pad_index", index, persistent=False)
        self.register_buffer("pad_dim", dim, persistent=False)
        self.register_buffer("pad_sentinel", sentinel, persistent=False)
        self.arg_sizes: list[int] = list(arg_nvec)
        self.arg_width: int = max(arg_nvec)
        self.ops: int = ops
        self.masked: float = MASKED

    def features(self, grid, entities, entity_mask, self_, inventory, goal):
        return self.extractor(grid, entities, entity_mask, self_, inventory, goal)

    def value(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_head(features[:, : self.features_dim]).squeeze(-1)

    def _op_logits(self, features: torch.Tensor, mask: torch.Tensor):
        op_mask = mask[:, : self.ops]
        logits = self.op_head(features[:, : self.features_dim])
        return torch.where(op_mask, logits, torch.full_like(logits, self.masked)), op_mask

    def _arg_parts(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        op: torch.Tensor,
        op_masks: torch.Tensor | None = None,
    ):
        """Unscattered argument logits, their legality, and the op context.

        Split out from `_arguments` because the autoregressive tail rescores
        three of the five dimensions once the target is known, and the pointer
        and placement heads that produce the other two are the expensive part:
        they are computed once and the cheap tail is spliced onto them.

        An argument the op does not read may only be its sentinel; one it does
        read may not be, while anything else in its dimension is legal.
        Padding is masked. One pass for all five dimensions. With `op_masks`
        (B x ops x the argument widths), "legal" is the row of the operation
        `op`, not the union the flat mask carries.
        """
        batch = features.shape[0]
        if op_masks is None:
            seg = mask[:, self.ops :]  # B x 179
        else:
            width = op_masks.shape[2]
            seg = op_masks.gather(1, op.view(batch, 1, 1).expand(batch, 1, width)).squeeze(1)
        used = self.uses[op][:, self.pad_dim]  # B x 179
        real = seg & ~self.pad_sentinel
        others = torch.zeros(batch, len(self.arg_sizes), device=seg.device, dtype=features.dtype)
        others = others.index_add(1, self.pad_dim, real.to(features.dtype)) > 0
        narrowed = seg & ~(self.pad_sentinel & others[:, self.pad_dim])
        allowed = torch.where(used, narrowed, self.pad_sentinel.expand_as(seg))

        f = features[:, : self.features_dim]
        one_hot = torch.nn.functional.one_hot(op, self.ops).to(features.dtype)
        context = torch.cat([f, one_hot], dim=1)
        flat = self.arg_head(context)  # B x 179
        if self.v2:
            flat = self._v2_logits(features, context, flat)
        return flat, allowed, context

    def _target_embedding(self, features: torch.Tensor, target: torch.Tensor):
        """The row the policy named, or a learned stand-in for "none"."""
        batch = features.shape[0]
        absent = self.absent_target.to(features.dtype).expand(batch, self.row_dim)
        if not self.v2:
            return absent
        width = self.rows * self.row_dim
        rows = features[:, self.features_dim : self.features_dim + width]
        rows = rows.reshape(batch, self.rows, self.row_dim)
        index = (target - 1).clamp(min=0, max=self.rows - 1)
        chosen = rows.gather(1, index.view(batch, 1, 1).expand(batch, 1, self.row_dim))
        chosen = chosen.squeeze(1)
        return torch.where((target > 0).unsqueeze(1), chosen, absent)

    def _retail(self, features, context: torch.Tensor, flat: torch.Tensor, target: torch.Tensor):
        """Rescore direction, item and amount now that the target is known."""
        head = sum(self.arg_sizes[:2])
        tail = self.after_target(
            torch.cat([context, self._target_embedding(features, target)], dim=1)
        )
        return torch.cat([flat[:, :head], tail], dim=1)

    def _scatter(self, flat: torch.Tensor, allowed: torch.Tensor):
        batch = flat.shape[0]
        size = len(self.arg_sizes) * self.arg_width
        logits = torch.full((batch, size), self.masked, device=flat.device, dtype=flat.dtype)
        logits = logits.index_copy(1, self.pad_index, torch.where(allowed, flat, logits[:, :1]))
        pad_mask = torch.zeros(batch, size, device=flat.device, dtype=torch.bool)
        pad_mask = pad_mask.index_copy(1, self.pad_index, allowed)
        shape = (batch, len(self.arg_sizes), self.arg_width)
        return logits.view(shape), pad_mask.view(shape)

    def _arguments(self, features: torch.Tensor, mask: torch.Tensor, op: torch.Tensor):
        flat, allowed, _context = self._arg_parts(features, mask, op)
        return self._scatter(flat, allowed)

    def _v2_logits(self, features: torch.Tensor, context: torch.Tensor, flat: torch.Tensor):
        """Replace the target and placement segments with the v2 heads' scores.

        Segments of `flat`: target [0, 33), placement [33, 155), then
        direction, item and amount ([0, 97) and [97, 323) under v3). The
        sentinels (0 and 33, or 0 and 97) stay the MLP's.
        """
        batch = features.shape[0]
        start = self.features_dim
        width = self.rows * self.row_dim
        rows = features[:, start : start + width].reshape(batch, self.rows, self.row_dim)
        crop = features[:, start + width :].reshape(batch, 6, self.crop, self.crop)
        targets = self.target_head(rows, context)
        cells = self.place_head(crop, context)
        # Grid rows are y and columns x; slot (dx + 5) * 11 + (dy + 5), and
        # (dx + 7) * 15 + (dy + 7) under v3.
        placements = cells.transpose(1, 2).reshape(batch, self.side * self.side)
        t = self.rows + 1
        rest = t + self.side * self.side + 1
        return torch.cat(
            [flat[:, :1], targets, flat[:, t : t + 1], placements, flat[:, rest:]], dim=1
        )

    def act(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        greedy: bool = False,
        epsilon: float = 0.0,
        op_masks: torch.Tensor | None = None,
    ):
        """-> actions (B x 6, int64), log-probabilities (B).

        `epsilon` is the chance a greedy decision samples instead. The task is
        deterministic, so a pure argmax policy that enters a cycle never leaves
        it -- measured, one action repeated for the last 200 decisions of an
        episode. Mnih et al. (2015) evaluated with an epsilon of 0.05 for the
        same reason. It has no effect unless `greedy`.

        `op_masks`: v3's per-operation masks. The operation is drawn under the
        flat mask's operation part, and the arguments under its own row.
        """
        op_logits, _ = self._op_logits(features, mask)
        if greedy:
            roll = torch.rand(op_logits.shape[0], device=op_logits.device) < epsilon
            op = torch.where(roll, _gumbel_argmax(op_logits), op_logits.argmax(-1))
        else:
            roll = torch.ones(op_logits.shape[0], device=op_logits.device, dtype=torch.bool)
            op = _gumbel_argmax(op_logits)
        flat, allowed, context = self._arg_parts(features, mask, op, op_masks)
        logits, _ = self._scatter(flat, allowed)
        if self.autoregressive:
            # Draw the target first, then score what may be done with it.
            first = logits[:, 0]
            target = torch.where(roll, _gumbel_argmax(first), first.argmax(-1))
            flat = self._retail(features, context, flat, target)
            logits, _ = self._scatter(flat, allowed)
            rest = torch.where(roll.unsqueeze(1), _gumbel_argmax(logits), logits.argmax(-1))
            args = torch.cat([target.unsqueeze(1), rest[:, 1:]], dim=1)
        else:
            args = torch.where(roll.unsqueeze(1), _gumbel_argmax(logits), logits.argmax(-1))
        logp = torch.log_softmax(op_logits, -1).gather(1, op.unsqueeze(1)).squeeze(1)
        arg_logp = torch.log_softmax(logits, -1).gather(2, args.unsqueeze(2)).squeeze(2)
        return torch.cat([op.unsqueeze(1), args], dim=1), logp + arg_logp.sum(1)

    def head_logits(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        op: torch.Tensor,
        op_masks: torch.Tensor | None = None,
    ):
        """-> operation logits and legality, argument logits and legality.

        What a KL against a frozen prior needs: the distributions themselves,
        not the log-probability of one action.
        """
        op_logits, op_mask = self._op_logits(features, mask)
        flat, allowed, context = self._arg_parts(features, mask, op, op_masks)
        if self.autoregressive:
            # No target has been chosen here, so the tail sees the stand-in.
            flat = self._retail(features, context, flat, torch.zeros_like(op))
        arg_logits, pad_mask = self._scatter(flat, allowed)
        return op_logits, op_mask, arg_logits, pad_mask

    def evaluate(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        actions: torch.Tensor,
        op_masks: torch.Tensor | None = None,
    ):
        """-> log-probabilities and entropies (B) of stored actions.

        With `op_masks`, each action's arguments are scored under the row of
        the operation it stores: the conditional mask `act` sampled them under,
        so the PPO ratio compares like with like. The argument entropy is
        likewise that of the stored operation's conditional distribution.
        """
        op = actions[:, 0]
        op_logits, op_mask = self._op_logits(features, mask)
        op_logp = torch.log_softmax(op_logits, -1)
        logp = op_logp.gather(1, op.unsqueeze(1)).squeeze(1)
        zero = torch.zeros_like(op_logp)
        entropy = -torch.where(op_mask, op_logp.exp() * op_logp, zero).sum(-1)
        flat, allowed, context = self._arg_parts(features, mask, op, op_masks)
        if self.autoregressive:
            # Teacher forcing: score the tail under the target that was taken,
            # which is what makes the summed factors the joint log-probability.
            flat = self._retail(features, context, flat, actions[:, 1])
        logits, pad_mask = self._scatter(flat, allowed)
        arg_logp = torch.log_softmax(logits, -1)
        logp = logp + arg_logp.gather(2, actions[:, 1:].unsqueeze(2)).squeeze(2).sum(1)
        p_log_p = torch.where(pad_mask, arg_logp.exp() * arg_logp, torch.zeros_like(arg_logp))
        return logp, entropy - p_log_p.sum((1, 2))


class Exported(nn.Module):
    """What a checkpoint exports: observation tensors and mask in, an action out.

    TorchScript, so FactorioRL can run it with torch alone and no import of
    this package. Inputs are the `local-v1` tensors, batched, as float32 (the
    entity mask as any integer type) and the flat 201-entry mask as bool. A v3
    policy takes the v3 tensors, the flat 376-entry mask, and the
    per-operation masks (B x 25 x 351, bool) as `op_masks`.
    """

    def __init__(self, policy: Policy) -> None:
        super().__init__()
        self.policy = policy

    def forward(
        self,
        grid,
        entities,
        entity_mask,
        self_,
        inventory,
        goal,
        mask,
        greedy: bool,
        op_masks: torch.Tensor | None = None,
    ):
        features = self.policy.features(grid, entities, entity_mask, self_, inventory, goal)
        actions, _ = self.policy.act(features, mask, greedy, 0.0, op_masks)
        return actions


def export(policy: Policy, path) -> None:
    module = Exported(copy.deepcopy(policy)).cpu().eval()
    module.policy.extractor.input_dtype = torch.float32
    scripted = torch.jit.script(module)
    scripted.save(str(path))
