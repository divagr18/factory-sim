"""Compile csrc/ into the `fsim._fsim` extension (cffi, API mode).

uv run python build.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from cffi import FFI

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"


def cdef_from_header(text: str) -> str:
    """The part of fsim.h between the CFFI markers, which cffi can parse."""
    start = text.index("/* CFFI-BEGIN */") + len("/* CFFI-BEGIN */")
    end = text.index("/* CFFI-END */")
    return text[start:end]


def main() -> int:
    header = (CSRC / "fsim.h").read_text(encoding="utf-8")
    ffi = FFI()
    ffi.cdef(cdef_from_header(header))
    windows = sys.platform == "win32"
    ffi.set_source(
        "fsim._fsim",
        '#include "fsim.h"',
        sources=[str(CSRC / "fsim.c")],
        include_dirs=[str(CSRC)],
        extra_compile_args=["/std:c11", "/O2", "/fp:precise"]
        if windows
        else ["-std=c11", "-O2", "-ffp-contract=off"],
    )
    ffi.compile(tmpdir=str(ROOT / "build"), verbose=False, target=None)
    built = sorted((ROOT / "build" / "fsim").glob("_fsim*"))
    for path in built:
        (ROOT / "fsim" / path.name).write_bytes(path.read_bytes())
    print("built", [p.name for p in built])
    return 0


if __name__ == "__main__":
    sys.exit(main())
