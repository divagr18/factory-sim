"""Copy FactorioRL's golden traces into tests/golden/, checking every hash.

The simulator never reads the FactorioRL checkout at test time: the traces it
is tested against are copied here, with the index that describes them, and
each trace's content hash is re-derived from the file before it is accepted.

    uv run python tools/sync_golden.py --from ../FactorioRL
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "tests" / "golden"


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def trace_hash(path: Path) -> str:
    """The hash FactorioRL's `record_parity_trace.trace_hash` computes."""
    lines = lzma.decompress(path.read_bytes()).decode().splitlines()
    digest = hashlib.sha256()
    digest.update(_canonical(json.loads(lines[0])).encode())
    for line in lines[1:]:
        digest.update(b"\n")
        digest.update(_canonical(json.loads(line)).encode())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="source", required=True, type=Path)
    args = parser.parse_args()
    evidence = args.source / "docs" / "evidence" / "sim-parity"
    index = json.loads((evidence / "index.json").read_text(encoding="utf-8"))
    DEST.mkdir(parents=True, exist_ok=True)
    problems = 0
    for _name, entry in sorted(index.items()):
        wanted = [(entry["trace"], entry["trace_sha256"])]
        if entry.get("ticks"):
            wanted.append((entry["ticks"]["trace"], entry["ticks"]["trace_sha256"]))
        for file_name, expected in wanted:
            source = evidence / file_name
            actual = trace_hash(source)
            if actual != expected:
                print(f"HASH MISMATCH {file_name}: index {expected[:12]} file {actual[:12]}")
                problems += 1
                continue
            shutil.copyfile(source, DEST / file_name)
            print(f"ok {file_name}")
    shutil.copyfile(evidence / "index.json", DEST / "index.json")
    for extra in (
        "sim-mechanics-m1.json",
        "sim-mechanics-m3.json",
        "sim-mechanics-m3-drills.json",
    ):
        path = args.source / "docs" / "evidence" / extra
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            # Where the recording machine keeps its game install is not evidence.
            if isinstance(data.get("engine"), dict):
                data["engine"].pop("executable", None)
            text = json.dumps(data, indent=0, sort_keys=True) + "\n"
            (DEST / extra).write_text(text, "utf-8")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
