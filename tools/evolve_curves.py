"""Learning curves of evolution runs against simulated interaction, scored on the holdout.

Replays each run's genealogy in creation order and follows its best-so-far
program (by `val_mean`, ties to the incumbent; `--tie-break length` replays
runs made before evaluator 3). Every time the best changes, that program is
scored on the frozen holdout, post hoc: the loop never saw these scenes. A
program is scored once per invocation, keyed by the hash of its code, so a seed
shared by several runs costs one holdout pass.

The x axis is simulated decisions: each evaluated row's `scores["episodes"]`
times 600, the decision budget every `construct_smelting_line` episode runs to
the end. Failed rows count their episodes too (missing counts as 0). Each
episode also runs a 3600-tick verification window, 120 decision-equivalents,
which is reported separately and not added to the x axis. PPO's budget is 40M
decisions per run.

Runs are grouped into arms by the letter in their name (`grid-A-s1` is arm A).

    python tools/evolve_curves.py runs/evolve-grid-A-s1 runs/evolve-grid-B-s1 \\
        --holdout-n 1000 --workers 4 --out curves.json --svg curves.svg --ppo ppo.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from hashlib import sha256
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evolve.archive import Store  # noqa: E402

DECISIONS_PER_EPISODE = 600
VERIFY_PER_EPISODE = 120
PPO_BUDGET = 40_000_000
FAILED = ("extract_failed", "sandbox_error", "eval_error")
THRESHOLDS = (0.9, 1.0)
ARM_RE = re.compile(r"(?:^|[-_])([A-Za-z])-s\d+$")
ARM_COLOURS = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#d62728"}
NOTE = (
    "sim_decisions = sum(scores.episodes) * 600 over every row replayed so far; each "
    "episode also runs a 3600-tick verification window (120 decision-equivalents), "
    "reported as verify_decisions and not included. completions = non-seed rows so "
    "far plus status.json duplicates prorated over rows (duplicates are not "
    "timestamped); rows is the exact count. PPO budget: 40M decisions per run."
)


def code_hash(code: str) -> str:
    return sha256(code.encode()).hexdigest()


class HoldoutCache:
    """`score(code)` -> holdout success rate; each distinct code is scored once."""

    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.cache: dict[str, float] = {}
        self.calls = 0

    def score(self, code: str) -> float:
        h = code_hash(code)
        if h not in self.cache:
            self.calls += 1
            self.cache[h] = float(self.evaluator.holdout(code)["mean"])
        return self.cache[h]


def arm_of(name: str) -> str:
    m = ARM_RE.search(name)
    return m.group(1).upper() if m else "?"


def _episodes(c) -> int:
    v = (c.scores or {}).get("episodes")
    try:
        return int(v) if v is not None else 0
    except (TypeError, ValueError):
        return 0


def _val(c) -> float:
    v = (c.scores or {}).get("val_mean")
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return -math.inf
    return float(v)


def _better(c, best, tie_break: str = "incumbent") -> bool:
    """Whether `c` displaces `best` under the run's selection rule.

    "incumbent" (evaluator 3 on): strictly higher validation only. "length":
    ties go to the shorter program, the rule runs made before then selected by."""
    if best is None:
        return True
    if tie_break == "length":
        return (_val(c), -c.length) > (_val(best), -best.length)
    return _val(c) > _val(best)


def first_reach(curve: list[dict], threshold: float):
    """sim_decisions at the first curve point with holdout >= threshold, else 'never'."""
    for p in curve:
        if p["holdout"] >= threshold - 1e-12:
            return p["sim_decisions"]
    return "never"


def replay(rows: list, score, duplicates: int = 0, tie_break: str = "incumbent") -> dict:
    """Walk `rows` (creation order) and score each new best-so-far with `score(code)`."""
    llm_rows = sum(1 for c in rows if c.operator != "seed")
    dup_rate = duplicates / llm_rows if llm_rows else 0.0
    episodes = done = 0
    best = None
    curve: list[dict] = []
    for c in rows:
        episodes += _episodes(c)
        if c.operator != "seed":
            done += 1
        if c.operator in FAILED or not _better(c, best, tie_break):
            continue
        best = c
        curve.append(
            {
                "completions": done + round(done * dup_rate),
                "rows": done,
                "sim_decisions": episodes * DECISIONS_PER_EPISODE,
                "val_mean": _val(c),
                "holdout": score(c.code),
                "candidate_id": c.id,
                "length": c.length,
            }
        )
    final = dict(curve[-1]) if curve else None
    if final is not None:
        final.update(
            {
                "completions": llm_rows + duplicates,
                "rows": llm_rows,
                "sim_decisions": episodes * DECISIONS_PER_EPISODE,
            }
        )
    return {
        "curve": curve,
        "final": final,
        "totals": {
            "rows": len(rows),
            "llm_rows": llm_rows,
            "duplicates": duplicates,
            "completions": llm_rows + duplicates,
            "episodes": episodes,
            "sim_decisions": episodes * DECISIONS_PER_EPISODE,
            "verify_decisions": episodes * VERIFY_PER_EPISODE,
            "failed_rows": sum(1 for c in rows if c.operator in FAILED),
        },
    }


def analyse_run(run_dir, score, tie_break: str = "incumbent") -> dict:
    run_dir = Path(run_dir)
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    name = run_dir.name.removeprefix("evolve-")
    with Store(run_dir / "genealogy.sqlite", readonly=True) as store:
        rows = store.all()
    out = replay(rows, score, int(status.get("duplicates") or 0), tie_break)
    out.update(
        {
            "run": name,
            "dir": str(run_dir),
            "arm": arm_of(name),
            "spent_usd": status.get("spent_usd"),
            "run_spent_usd": status.get("run_spent_usd"),
            "elapsed_s": status.get("elapsed_s"),
            "reach": {str(t): first_reach(out["curve"], t) for t in THRESHOLDS},
        }
    )
    return out


def aggregate_arms(runs: list[dict]) -> dict:
    arms: dict[str, dict] = {}
    for r in sorted(runs, key=lambda r: (r["arm"], r["run"])):
        a = arms.setdefault(r["arm"], {"runs": [], "final_holdout": {}, "curves": {}, "reach": {}})
        a["runs"].append(r["run"])
        a["final_holdout"][r["run"]] = r["final"]["holdout"] if r["final"] else None
        a["curves"][r["run"]] = [[p["sim_decisions"], p["holdout"]] for p in r["curve"]]
        a["reach"][r["run"]] = r["reach"]
    for a in arms.values():
        vals = [v for v in a["final_holdout"].values() if v is not None]
        a["final_holdout_mean"] = sum(vals) / len(vals) if vals else None
        a["final_holdout_min"] = min(vals) if vals else None
        a["final_holdout_max"] = max(vals) if vals else None
        for t in THRESHOLDS:
            hits = [r[str(t)] for r in a["reach"].values() if r[str(t)] != "never"]
            a[f"runs_reaching_{t}"] = f"{len(hits)}/{len(a['reach'])}"
    return arms


# ------------------------------------------------------------------ output


def _fmt_dec(v) -> str:
    if v == "never" or v is None:
        return "never"
    v = float(v)
    for unit, div in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if v >= div:
            return f"{v / div:.3g}{unit}"
    return f"{v:.0f}"


def _fmt(v, spec=".3f") -> str:
    return "-" if v is None else format(v, spec)


def table(runs: list[dict], ppo: list[dict]) -> str:
    head = ("arm", "run", "final val", "final holdout", "dec to 0.9", "dec to 1.0", "$", "wall")
    rows = []
    for r in sorted(runs, key=lambda r: (r["arm"], r["run"])):
        f = r["final"] or {}
        usd = r["run_spent_usd"] if r["run_spent_usd"] is not None else r["spent_usd"]
        wall = r["elapsed_s"]
        rows.append(
            (
                r["arm"],
                r["run"],
                _fmt(f.get("val_mean")),
                _fmt(f.get("holdout")),
                _fmt_dec(r["reach"]["0.9"]),
                _fmt_dec(r["reach"]["1.0"]),
                _fmt(usd, ".2f"),
                "-" if wall is None else f"{wall / 3600:.2f}h",
            )
        )
    for p in ppo:
        end = f"end {_fmt_dec(p.get('steps'))}"
        rows.append(
            (
                "PPO",
                f"{p.get('run')} ({p.get('mode', '')})",
                "-",
                _fmt(p.get("holdout")),
                end if (p.get("holdout") or 0) >= 0.9 else "never",
                end if (p.get("holdout") or 0) >= 1.0 else "never",
                "-",
                "-",
            )
        )
    widths = [max(len(str(x)) for x in col) for col in zip(head, *rows, strict=True)]
    lines = [
        " | ".join(str(x).ljust(w) for x, w in zip(row, widths, strict=True))
        for row in (head, *rows)
    ]
    lines.insert(1, "-+-".join("-" * w for w in widths))
    return "\n".join(line.rstrip() for line in lines)


def svg(runs: list[dict], ppo: list[dict], width=720, height=420) -> str:
    """Holdout vs simulated decisions, log-x, a step polyline per run, PPO as markers."""
    left, right, top, bottom = 60, 110, 20, 45
    xs = [p["sim_decisions"] for r in runs for p in r["curve"] if p["sim_decisions"] > 0]
    xs += [r["totals"]["sim_decisions"] for r in runs if r["totals"]["sim_decisions"] > 0]
    xs += [p["steps"] for p in ppo if p.get("steps")]
    lo = 10 ** math.floor(math.log10(min(xs))) if xs else 1e3
    hi = 10 ** math.ceil(math.log10(max(xs))) if xs else 1e8
    if hi <= lo:
        hi = lo * 10
    pw, ph = width - left - right, height - top - bottom

    def X(v):
        v = max(v, lo)
        return left + pw * (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))

    def Y(v):
        return top + ph * (1 - v)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'font-family="sans-serif" font-size="11">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" stroke="#999"/>',
    ]
    for e in range(int(math.log10(lo)), int(math.log10(hi)) + 1):
        x = X(10**e)
        out.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + ph}" stroke="#eee"/>')
        out.append(f'<text x="{x:.1f}" y="{top + ph + 14}" text-anchor="middle">1e{e}</text>')
    for v in (0, 0.25, 0.5, 0.75, 0.9, 1.0):
        y = Y(v)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + pw}" y2="{y:.1f}" stroke="#eee"/>')
        out.append(f'<text x="{left - 5}" y="{y + 4:.1f}" text-anchor="end">{v:g}</text>')
    out.append(
        f'<text x="{left + pw / 2}" y="{height - 8}" text-anchor="middle">'
        "simulated decisions (log)</text>"
    )
    out.append(
        f'<text x="14" y="{top + ph / 2}" text-anchor="middle" '
        f'transform="rotate(-90 14 {top + ph / 2})">holdout success</text>'
    )
    for r in runs:
        colour = ARM_COLOURS.get(r["arm"], "#7f7f7f")
        pts, prev = [], None
        for p in r["curve"]:
            x, y = X(p["sim_decisions"]), Y(p["holdout"])
            if prev is not None:
                pts.append(f"{x:.1f},{prev:.1f}")
            pts.append(f"{x:.1f},{y:.1f}")
            prev = y
        if prev is not None:
            pts.append(f"{X(r['totals']['sim_decisions']):.1f},{prev:.1f}")
            out.append(
                f'<polyline fill="none" stroke="{colour}" stroke-width="1.5" '
                f'points="{" ".join(pts)}"><title>{r["run"]}</title></polyline>'
            )
    for p in ppo:
        if not p.get("steps") or p.get("holdout") is None:
            continue
        x, y = X(p["steps"]), Y(p["holdout"])
        out.append(
            f'<path d="M{x - 5:.1f},{y:.1f}L{x:.1f},{y - 5:.1f}L{x + 5:.1f},{y:.1f}'
            f'L{x:.1f},{y + 5:.1f}Z" fill="black"><title>PPO {p.get("run")} '
            f"{p.get('mode', '')}</title></path>"
        )
    ly = top + 10
    for arm in sorted({r["arm"] for r in runs}):
        colour = ARM_COLOURS.get(arm, "#7f7f7f")
        lx = left + pw + 15
        out.append(
            f'<line x1="{lx}" y1="{ly}" x2="{lx + 20}" y2="{ly}" stroke="{colour}" '
            'stroke-width="2"/>'
        )
        out.append(f'<text x="{lx + 25}" y="{ly + 4}">arm {arm}</text>')
        ly += 16
    if ppo:
        lx = left + pw + 25
        out.append(f'<path d="M{lx - 5},{ly}L{lx},{ly - 5}L{lx + 5},{ly}L{lx},{ly + 5}Z"/>')
        out.append(f'<text x="{lx + 15}" y="{ly + 4}">PPO</text>')
    out.append("</svg>")
    return "\n".join(out)


def load_ppo(path) -> list[dict]:
    if not path:
        return []
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(doc["rows"] if isinstance(doc, dict) else doc)


def build_report(
    run_dirs, score, ppo: list[dict], holdout_n: int, holdout_start: int = 0,
    tie_break: str = "incumbent",
) -> dict:  # fmt: skip
    runs = [analyse_run(d, score, tie_break) for d in run_dirs]
    return {
        "holdout_n": holdout_n,
        "holdout_start": holdout_start,
        "tie_break": tie_break,
        "decisions_per_episode": DECISIONS_PER_EPISODE,
        "verify_decisions_per_episode": VERIFY_PER_EPISODE,
        "ppo_budget_decisions": PPO_BUDGET,
        "note": NOTE,
        "runs": runs,
        "arms": aggregate_arms(runs),
        "ppo": ppo,
        "holdout_programs_scored": None,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="run directories holding genealogy.sqlite")
    ap.add_argument("--holdout-n", type=int, default=1000)
    ap.add_argument(
        "--holdout-start",
        type=int,
        default=0,
        help="first holdout-stream index; report changed methods on unseen indices (e.g. 1000)",
    )
    ap.add_argument(
        "--tie-break",
        choices=("incumbent", "length"),
        default="incumbent",
        help="the selection rule the runs used: 'length' for runs before evaluator 3",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--svg")
    ap.add_argument("--ppo", help="JSON list of {run, mode, holdout, steps}")
    args = ap.parse_args(argv)

    from evolve import evaluate
    from evolve.pool import EvalPool

    ppo = load_ppo(args.ppo)
    sets = evaluate.scene_sets(
        train_n=0, val_n=0, holdout_n=args.holdout_n, holdout_start=args.holdout_start
    )
    with EvalPool(
        args.workers, "evolve.evaluate:worker_init", "evolve.evaluate:worker_job", timeout_s=600
    ) as pool:
        cache = HoldoutCache(evaluate.Evaluator(pool, sets))
        report = build_report(
            args.runs, cache.score, ppo, args.holdout_n, args.holdout_start, args.tie_break
        )
        report["holdout_programs_scored"] = cache.calls
    Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.svg:
        Path(args.svg).write_text(svg(report["runs"], ppo), encoding="utf-8")
    print(table(report["runs"], ppo))
    for arm, a in sorted(report["arms"].items()):
        finals = ", ".join(f"{k}={_fmt(v)}" for k, v in a["final_holdout"].items())
        print(
            f"arm {arm}: holdout [{finals}] mean {_fmt(a['final_holdout_mean'])} "
            f"min {_fmt(a['final_holdout_min'])} max {_fmt(a['final_holdout_max'])}; "
            f">=0.9 {a['runs_reaching_0.9']}, >=1.0 {a['runs_reaching_1.0']}"
        )
    print(f"{report['holdout_programs_scored']} programs scored on {args.holdout_n} holdout scenes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
