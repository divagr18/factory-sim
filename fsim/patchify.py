"""One CUDA kernel for the extractor's grid prologue, when one can be built.

The extractor's first layer is a 4x4 stride-4 projection of the observation
grid, computed as a matmul over `pixel_unshuffle(grid, 4).permute(0, 2, 3, 1)`.
Reaching that tensor costs three full passes over the grid -- convert to
bfloat16, scale by 1/255, then unshuffle and permute and copy -- and at a
4096-sample minibatch the grid is 104M elements, so those passes measured
7.93 ms against 1.94 ms for one kernel that does all three at once:

    out[b, h * 16 + w, c * 16 + i * 4 + j] = grid[b, c, h * 4 + i, w * 4 + j] / 255

It reads the 65x65 grid where it already lies -- `65 // 4` is 16, and the row
stride is the grid's own -- so the 65th row and column, which the projection
never reads, are simply never touched, and nothing is copied first.

**What it writes is bit-identical to what the portable path builds**, and
`test_patchify.py` asserts exactly that. The features that come out of the
extractor still differ in the last bfloat16 places, because the matmul then
reads a contiguous tensor instead of a permuted view and cuBLAS accumulates it
in a different order -- measured, 0.125 on logits of order 10. That is the size
of difference a library version bump makes, so a run with the kernel and a run
without it are the same experiment, but they are not step-for-step identical
and this does not claim they are.

`export()` never emits it, so a policy transferred to FactorioRL does not
depend on a toolchain being present, and a CPU-only process never builds one.

**Off unless asked for**: set `FSIM_FUSED_GRID=1`. Two reasons to opt in
rather than out. A run that builds it is not numerically the same run as one
that does not, so switching it on under a comparison already in flight would
change two things at once. And building it costs a compiler, a CUDA toolkit
and about a minute the first time, which a machine that only wants to replay a
policy should not be asked for.
"""

from __future__ import annotations

import os
import sys
from typing import Any

SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// One thread per (sample, cell, plane, row-of-the-patch): it reads four bytes
// that are contiguous in the grid and writes the four bfloat16 they become.
__global__ void patchify_kernel(
    const unsigned char* __restrict__ grid,
    at::BFloat16* __restrict__ out,
    int batch, int planes, int side, int cells, int scale) {
  const long long total = (long long)batch * cells * cells * planes * scale;
  for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
       t < total; t += (long long)gridDim.x * blockDim.x) {
    long long rest = t;
    const int i = (int)(rest % scale);   rest /= scale;
    const int c = (int)(rest % planes);  rest /= planes;
    const int w = (int)(rest % cells);   rest /= cells;
    const int h = (int)(rest % cells);   rest /= cells;
    const int b = (int)rest;

    const long long in_base = ((long long)b * planes + c) * side * side
                            + (long long)(h * scale + i) * side + w * scale;
    const long long out_base =
        ((long long)b * cells * cells + h * cells + w) * (planes * scale * scale)
        + (long long)c * scale * scale + i * scale;
    for (int j = 0; j < scale; ++j) {
      out[out_base + j] = at::BFloat16(grid[in_base + j] * (1.0f / 255.0f));
    }
  }
}

torch::Tensor patchify(torch::Tensor grid, int64_t scale) {
  TORCH_CHECK(grid.is_cuda(), "grid must be on the device");
  TORCH_CHECK(grid.scalar_type() == torch::kUInt8, "grid must be uint8");
  TORCH_CHECK(grid.is_contiguous(), "grid must be contiguous");
  TORCH_CHECK(grid.dim() == 4 && grid.size(2) == grid.size(3), "grid must be B x C x S x S");
  const int batch = (int)grid.size(0), planes = (int)grid.size(1), side = (int)grid.size(2);
  const int cells = side / (int)scale;
  auto out = torch::empty({batch, cells * cells, planes * (int)(scale * scale)},
                          grid.options().dtype(torch::kBFloat16));
  const long long needed = (long long)batch * cells * cells * planes * scale;
  const int threads = 256;
  long long blocks = (needed + threads - 1) / threads;
  if (blocks > 65535) blocks = 65535;
  patchify_kernel<<<(int)blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      grid.data_ptr<unsigned char>(), out.data_ptr<at::BFloat16>(),
      batch, planes, side, cells, (int)scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
"""

_MODULE: Any = None
_TRIED = False
#: Why the kernel is not in use, for a bench or a run report to state.
reason: str | None = None


def _vcvars() -> str | None:
    """`vcvars64.bat`, from vswhere or the places the installer uses."""
    import glob
    import subprocess

    program_files = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = os.path.join(program_files, "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if os.path.isfile(vswhere):
        try:
            root = subprocess.run(
                [vswhere, "-latest", "-products", "*", "-property", "installationPath"],
                capture_output=True, text=True, timeout=30, check=True,
            ).stdout.strip()  # fmt: skip
            candidate = os.path.join(root, "VC", "Auxiliary", "Build", "vcvars64.bat")
            if root and os.path.isfile(candidate):
                return candidate
        except Exception:
            pass
    for base in (program_files, os.environ.get("ProgramFiles", r"C:\Program Files")):
        found = glob.glob(
            os.path.join(base, "Microsoft Visual Studio", "*", "*", "VC", "Auxiliary",
                         "Build", "vcvars64.bat")
        )  # fmt: skip
        if found:
            return sorted(found)[-1]
    return None


def _inject_msvc_environment() -> None:
    """Put MSVC on PATH, so the kernel builds outside a developer prompt.

    `torch.utils.cpp_extension` shells out to `where cl`, so without this it
    only works in a developer prompt -- which is not where the training jobs
    run. `vcvars64.bat` is the compiler's own environment script; running it and
    keeping what it exported is what that prompt is.
    """
    import shutil
    import subprocess

    if os.environ.get("VCINSTALLDIR") or shutil.which("cl"):
        return
    script = _vcvars()
    if not script:
        return
    try:
        dumped = subprocess.run(
            f'"{script}" >nul && set', shell=True, capture_output=True, text=True,
            timeout=120, check=True,
        ).stdout  # fmt: skip
    except Exception:
        return
    for line in dumped.splitlines():
        key, _, value = line.partition("=")
        if key and value and key.upper() in ("PATH", "INCLUDE", "LIB", "LIBPATH", "VCINSTALLDIR",
                                             "WINDOWSSDKDIR", "UCRTVERSION", "VCTOOLSVERSION",
                                             "WINDOWSSDKVERSION", "VSINSTALLDIR"):  # fmt: skip
            os.environ[key.upper()] = value


def load():
    """The compiled module, or None with `reason` set. Tried once per process."""
    global _MODULE, _TRIED, reason
    if _TRIED:
        return _MODULE
    _TRIED = True
    if not os.environ.get("FSIM_FUSED_GRID"):
        reason = "not asked for; set FSIM_FUSED_GRID=1"
        return None
    try:
        import torch
    except ImportError:
        reason = "no torch"
        return None
    if not torch.cuda.is_available():
        reason = "no cuda device"
        return None
    try:
        from torch.utils.cpp_extension import load_inline
    except ImportError:  # pragma: no cover
        reason = "no torch.utils.cpp_extension"
        return None
    # `ninja` and the compiler both have to be findable on PATH: ninja ships in
    # the environment's own scripts directory, which a bare interpreter does not
    # add, and MSVC only appears inside a developer prompt.
    scripts = os.path.dirname(sys.executable)
    if scripts and scripts not in os.environ.get("PATH", ""):
        os.environ["PATH"] = scripts + os.pathsep + os.environ.get("PATH", "")
    if sys.platform == "win32":
        _inject_msvc_environment()
    try:
        _MODULE = load_inline(
            name="fsim_patchify",
            cpp_sources="torch::Tensor patchify(torch::Tensor grid, int64_t scale);",
            cuda_sources=SOURCE,
            functions=["patchify"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                # CUDA 13's CCCL headers refuse MSVC's traditional preprocessor.
                "-Xcompiler",
                "/Zc:preprocessor" if sys.platform == "win32" else "-O3",
                "-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING",
            ],
            verbose=False,
        )
    except Exception as error:  # a missing toolchain is ordinary, not fatal
        reason = f"{type(error).__name__}: {str(error).splitlines()[0][:160]}"
        _MODULE = None
    return _MODULE


def available() -> bool:
    return load() is not None
