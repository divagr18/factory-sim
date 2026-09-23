"""The baseline an evolved program has to beat: one source, and PPO beside it.

Scores a builder program (the seed by default) on the evaluator's pinned
train, validation and holdout sets, per family, and prints the PPO runs'
`final.json` rates in the same columns. The PPO test rows are 512 episodes of
the test split drawn from each run's own seeds, not the frozen holdout, so
they are the same family on different scenes; the row label says so.

    python tools/evolve_report.py
    python tools/evolve_report.py --source cand.py --ppo-runs runs/ar-s1 runs/ar-s2 --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evolve import evaluate as ev  # noqa: E402
from evolve.pool import EvalPool  # noqa: E402

COLUMNS = (*ev.FAMILIES_TRAIN, *ev.FAMILIES_HOLDOUT)
MODES = ("sampled", "epsilon", "greedy")


def program_rows(evaluator: ev.Evaluator, source: str, label: str) -> list[dict]:
    full = evaluator.full(source)
    held = evaluator.holdout(source)
    rows = []
    for split, summary in (
        ("train", full["detail"]["train"]),
        ("val", full["detail"]["val"]),
        ("holdout", held),
    ):
        rows.append(
            {
                "row": f"{label} {split}",
                "rates": summary["rates"],
                "mean": summary["mean"],
                "success": summary["success"],
                "n": summary["n"],
                "error": summary["error"],
                "descriptors": summary["descriptors"],
            }
        )
    return rows


def ppo_rows(run_dir: Path) -> list[dict]:
    with open(run_dir / "final.json", encoding="utf-8") as f:
        final = json.load(f)
    rows = []
    for mode in MODES:
        train, test = final.get(f"train_{mode}"), final.get(f"test_{mode}")
        if train is None and test is None:
            continue
        rates, n = {}, 0
        for part in (train, test):
            if part:
                rates.update(part.get("by_family", {}))
                n += part.get("episodes", 0)
        fams = [f for f in ev.FAMILIES_TRAIN if f in rates]
        rows.append(
            {
                "row": f"ppo {run_dir.name} {mode}",
                "rates": rates,
                "mean": sum(rates[f] for f in fams) / len(fams) if fams else None,
                "success": train.get("success") if train else None,
                "test_success": test.get("success") if test else None,
                "n": n,
                "error": None,
            }
        )
    return rows


def render(rows: list[dict]) -> str:
    width = max(len(r["row"]) for r in rows) + 2
    short = [c.replace("_patch", "") for c in COLUMNS]
    head = (
        "".ljust(width)
        + "".join(c.rjust(11) for c in short)
        + "train-mean".rjust(12)
        + "n".rjust(6)
    )
    out = [head, "-" * len(head)]
    for r in rows:
        cells = "".join(
            (f"{r['rates'][c]:.3f}" if c in r["rates"] else "-").rjust(11) for c in COLUMNS
        )
        mean = "-" if r["mean"] is None or r["row"].endswith("holdout") else f"{r['mean']:.3f}"
        out.append(r["row"].ljust(width) + cells + mean.rjust(12) + str(r["n"]).rjust(6))
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, help="a builder program (default: the seed)")
    ap.add_argument("--ppo-runs", type=Path, nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--train-n", type=int, default=128)
    ap.add_argument("--val-n", type=int, default=256)
    ap.add_argument("--holdout-n", type=int, default=100)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--json", type=Path, help="also write the table here")
    args = ap.parse_args(argv)

    if args.source:
        source, label = args.source.read_text(encoding="utf-8"), args.source.stem
    else:
        from evolve.seeds.builder import SOURCE

        source, label = SOURCE, "seed"
    sets = ev.scene_sets(args.train_n, args.val_n, args.holdout_n)
    check = ev.verify_holdout(sets)
    with EvalPool(
        args.workers, "evolve.evaluate:worker_init", "evolve.evaluate:worker_job", args.timeout
    ) as pool:
        evaluator = ev.Evaluator(pool, sets)
        evaluator.full(source)  # warm every worker, so the timing below is steady state
        start = time.perf_counter()
        rows = program_rows(evaluator, source, label)
        elapsed = time.perf_counter() - start
    for run in args.ppo_runs:
        if (run / "final.json").is_file():
            rows.extend(ppo_rows(run))
        else:
            print(f"skipping {run}: no final.json", file=sys.stderr)

    print(render(rows))
    print()
    print(
        f"{label}: train+val+holdout in {elapsed:.2f}s on {args.workers} workers "
        f"({sum(len(v) for v in sets.values())} episodes)"
    )
    if check["file"]:
        print(
            f"holdout: {check['matched']}/{check['compared']} scenes match the frozen digests "
            f"in {check['file']}"
        )
    else:
        print("holdout: no FactorioRL checkout to verify the frozen digests against")
    if args.ppo_runs:
        print("ppo obstructed column: the run's own test-split draw, not the frozen holdout")
    if args.json:
        doc = {
            "evaluator_version": ev.EVALUATOR_VERSION,
            "task": ev.TASK,
            "set_digests": ev.set_digests(sets),
            "sizes": {k: len(v) for k, v in sets.items()},
            "holdout_check": check,
            "seconds": elapsed,
            "workers": args.workers,
            "rows": rows,
        }
        args.json.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
