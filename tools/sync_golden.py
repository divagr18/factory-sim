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
import subprocess
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
    # The v3 contract (FactorioRL tools/v3_contract_golden.py) and the reach
    # sweep the v3 mask is checked against (tools/probe_handmine.py).
    for name, where in (("v3_contract.json.xz", evidence),
                        ("handmine-reach.json.xz", args.source / "docs" / "evidence")):  # fmt: skip
        if (where / name).exists():
            shutil.copyfile(where / name, DEST / name)
            print(f"ok {name}")
    for extra in (
        "sim-mechanics-m1.json",
        "sim-mechanics-m3.json",
        "sim-mechanics-m3-drills.json",
        "sim-mechanics-m5-slide.json",
        "sim-mechanics-m5-slide-gaps.json",
        "sim-mechanics-m5-slide-creep.json",
    ):
        path = args.source / "docs" / "evidence" / extra
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            # Where the recording machine keeps its game install is not evidence.
            if isinstance(data.get("engine"), dict):
                data["engine"].pop("executable", None)
            text = json.dumps(data, indent=0, sort_keys=True) + "\n"
            (DEST / extra).write_text(text, "utf-8")
    # The installed package's copy of the map: terrain only, see fsim.water_tiles.
    m3 = json.loads((DEST / "sim-mechanics-m3.json").read_text(encoding="utf-8"))
    terrain = {"source": "sim-mechanics-m3.json (FactorioRL docs/evidence), terrain only",
               "terrain": m3["terrain"]}  # fmt: skip
    text = json.dumps(terrain, sort_keys=True) + "\n"
    (ROOT / "fsim" / "data" / "terrain.json").write_text(text, "utf-8")
    write_scenes(args.source)
    write_potentials(args.source)
    return 1 if problems else 0


#: Run inside FactorioRL's own environment, so the scenes are its generators'
#: output and not this project's port of them.
SCENES_SCRIPT = """
import json, random, sys
from factoriorl.tasks import get as get_task
out = {}
for task_id in ("construct_smelting_line", "build_line", "plate_line"):
    task = get_task(task_id)
    for family in task.spec.layout_families:
        for seed in range(int(sys.argv[1])):
            blueprint = task.generate(family, random.Random(seed))
            payload = blueprint.to_dict(public_markers=task.spec.public_markers)
            out[f"{task_id}|{family.name}|{seed}"] = payload
print(json.dumps(out, sort_keys=True))
"""


def write_scenes(source: Path, seeds: int = 16) -> None:
    result = subprocess.run(
        ["uv", "run", "python", "-c", SCENES_SCRIPT, str(seeds)],
        cwd=source, capture_output=True, text=True, check=True,
    )  # fmt: skip
    scenes = json.loads(result.stdout)
    (DEST / "scenes.json").write_text(json.dumps(scenes, sort_keys=True) + "\n", "utf-8")
    print(f"ok scenes.json ({len(scenes)} scenes)")


#: FactorioRL's reference line potential on every recorded decision of every
#: construct_smelting_line trace, for `test_vec_and_shaping.py`.
POTENTIALS_SCRIPT = """
import json, lzma, sys
from pathlib import Path
from factoriorl.tasks.potentials import line_potential
evidence = Path("docs/evidence/sim-parity")
index = json.loads((evidence / "index.json").read_text(encoding="utf-8"))
out = {}
for name in sorted(index):
    lines = lzma.decompress((evidence / index[name]["trace"]).read_bytes()).decode().splitlines()
    header = json.loads(lines[0])
    if header["task"] != "construct_smelting_line":
        continue
    records = [json.loads(line) for line in lines[1:]]
    out[name] = [line_potential(r["observation"], r["truth"], "patch") for r in records]
print(json.dumps(out, sort_keys=True))
"""


def write_potentials(source: Path) -> None:
    result = subprocess.run(
        ["uv", "run", "python", "-c", POTENTIALS_SCRIPT],
        cwd=source, capture_output=True, text=True, check=True,
    )  # fmt: skip
    potentials = json.loads(result.stdout)
    text = json.dumps(potentials, sort_keys=True) + "\n"
    (DEST / "potentials.json").write_text(text, "utf-8")
    print(f"ok potentials.json ({sum(map(len, potentials.values()))} decisions)")


if __name__ == "__main__":
    sys.exit(main())
