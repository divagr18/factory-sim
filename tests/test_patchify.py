"""The fused grid prologue writes what the portable path would have built."""

from __future__ import annotations

import pytest

from fsim import patchify

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the kernel needs a device to run on"
)


@pytest.fixture(scope="module")
def module():
    built = patchify.load()
    if built is None:
        pytest.skip(f"no kernel: {patchify.reason}")
    return built


def portable(grid):
    """What `Extractor._grid` feeds its first layer."""
    scaled = grid[:, :, :64, :64].to(torch.bfloat16).div_(255.0)
    patches = torch.nn.functional.pixel_unshuffle(scaled, 4)
    return patches.permute(0, 2, 3, 1).reshape(grid.shape[0], 256, 96)


@pytest.mark.parametrize("batch", [1, 7, 512])
def test_the_kernel_writes_the_same_tensor(module, batch):
    grid = torch.randint(0, 256, (batch, 6, 65, 65), device="cuda", dtype=torch.uint8)
    assert torch.equal(module.patchify(grid, 4), portable(grid))


def test_the_extremes_are_right(module):
    """0 and 255 are the values the encoder actually produces most."""
    for value in (0, 1, 254, 255):
        grid = torch.full((2, 6, 65, 65), value, device="cuda", dtype=torch.uint8)
        assert torch.equal(module.patchify(grid, 4), portable(grid))


def test_the_last_row_and_column_are_never_read(module):
    """`65 // 4` is 16, so the projection's window is the first 64 of each axis."""
    grid = torch.randint(0, 256, (4, 6, 65, 65), device="cuda", dtype=torch.uint8)
    before = module.patchify(grid, 4)
    grid[:, :, 64, :] = 255 - grid[:, :, 64, :]
    grid[:, :, :, 64] = 255 - grid[:, :, :, 64]
    assert torch.equal(module.patchify(grid, 4), before)


def test_the_extractor_agrees_with_itself_either_way():
    """The features differ only in the last bfloat16 places, not in substance."""
    from fsim.policy import Extractor

    if patchify.load() is None:
        pytest.skip(f"no kernel: {patchify.reason}")
    torch.manual_seed(0)
    extractor = Extractor().to("cuda")
    extractor.input_dtype = torch.bfloat16
    grid = torch.randint(0, 256, (64, 6, 65, 65), device="cuda", dtype=torch.uint8)
    rest = (
        torch.randn(64, 32, 16, device="cuda"),
        torch.ones(64, 32, device="cuda", dtype=torch.int8),
        torch.randn(64, 12, device="cuda"),
        torch.randn(64, 14, device="cuda"),
        torch.randn(64, 12, device="cuda"),
    )
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        fast = extractor(grid, *rest)
        extractor.fused = None
        slow = extractor(grid, *rest)
    assert fast.shape == slow.shape
    assert torch.allclose(fast.float(), slow.float(), atol=0.05, rtol=0.05)
