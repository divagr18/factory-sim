"""Compare training runs at equal step counts.

Reads `runs/<name>/metrics.jsonl` for each named run and prints, at each
checkpoint in `--at` (millions of steps), the scene-start success rate, the
share of episodes that built a line, and the latest evaluation. With `--out`
the table is written as JSON for docs/.

    python tools/compare_runs.py sparse-s1 potential-s1 progress-s1 --at 1 2 5 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> list[dict]:
    path = ROOT / "runs" / name / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def window(rows: list[dict], steps: int, span: int = 5) -> dict | None:
    """Means over the `span` updates ending at the first row past `steps`."""
    upto = [r for r in rows if r["steps"] <= steps]
    if not upto or upto[-1]["steps"] < steps * 0.95:
        return None
    tail = [r for r in upto[-span:] if "train_success" in r]
    if not tail:
        return None

    def mean(key):
        return round(sum(r.get(key, 0.0) for r in tail) / len(tail), 4)

    evals = [r["eval"] for r in upto if "eval" in r]
    out = {
        "steps": upto[-1]["steps"],
        "train_success": mean("train_success"),
        "train_verified": mean("train_verified"),
        "line_built": mean("line_built"),
        "peak_potential": mean("peak_potential"),
        "entropy": mean("ent"),
    }
    if evals:
        out["eval_success"] = evals[-1]["success"]
        out["eval_success_ci95"] = evals[-1]["success_ci95"]
    demo = [r["demo_success"] for r in upto[-span:] if "demo_success" in r]
    if demo:
        out["demo_success"] = demo[-1]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--at", nargs="+", type=float, default=[1, 2, 5, 10, 20])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    table = {}
    for name in args.runs:
        rows = load(name)
        table[name] = {f"{m:g}M": window(rows, int(m * 1_000_000)) for m in args.at}
        final = ROOT / "runs" / name / "final.json"
        if final.exists():
            table[name]["final"] = {
                k: {x: v[x] for x in ("success", "success_ci95", "by_family")}
                for k, v in json.loads(final.read_text(encoding="utf-8")).items()
            }
    for name, points in table.items():
        print(name)
        for at, row in points.items():
            if at != "final" and row:
                print(
                    f"  {at:>5}: success {row['train_success']:.3f}  "
                    f"line {row['line_built']:.3f}  peak phi {row['peak_potential']:.3f}"
                    + (f"  eval {row['eval_success']:.3f}" if "eval_success" in row else "")
                )
        if "final" in points:
            print("  final:", {k: v["success"] for k, v in points["final"].items()})
    if args.out:
        args.out.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
