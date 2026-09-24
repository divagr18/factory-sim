"""SFT data for the warm start: evolved builder programs from `evolve` archives,
paired with real factorio-build prompts (train split only).

    uv run python recipes/grpo-factorio-build/build_sft_data.py \
        --archives "runs/*/genealogy.sqlite" --target 800 --out sft.jsonl

Programs must score >= --min-train on training scenes and pass the current
sandbox; duplicates are removed by normalised AST. Every program below 0.9 is
kept; the rest is filled from >= 0.9 programs spread round-robin over
(run, island) lineages, so no single lineage dominates. Selection never reads
validation or holdout scores.
"""
import argparse
import collections
import glob
import json
import random
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "integrations" / "verifiers" / "factorio_build"))
from factorio_build import core  # noqa: E402
from evolve import sandbox  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--archives", required=True, help="glob of genealogy.sqlite files")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--target", type=int, default=800)
ap.add_argument("--min-train", type=float, default=0.5)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = random.Random(args.seed)

progs = {}
for db in sorted(glob.glob(args.archives)):
    run = Path(db).stem if Path(db).stem != "genealogy" else Path(db).parent.name
    for code, tm, island in sqlite3.connect(db).execute("select code, train_mean, island from candidates"):
        if tm is None or tm < args.min_train:
            continue
        try:
            sandbox.check(code)
        except sandbox.SandboxError:
            continue
        progs.setdefault(sandbox.normalized_hash(code), (code, tm, f"{run}/{island}"))
print("distinct sandbox-valid programs:", len(progs))

low = [p for p in progs.values() if p[1] < 0.9]
by_lineage = collections.defaultdict(list)
for p in progs.values():
    if p[1] >= 0.9:
        by_lineage[p[2]].append(p)
for v in by_lineage.values():
    rng.shuffle(v)
rng.shuffle(low)
chosen = low[: args.target]
while len(chosen) < args.target and any(by_lineage.values()):
    for k in list(by_lineage):
        if by_lineage[k] and len(chosen) < args.target:
            chosen.append(by_lineage[k].pop())

rows = list(core.rows("construct_smelting_line", "train", 8, 64, 0, True))
lines = []
for code, tm, _ in chosen:
    r = rng.choice(rows)
    user = r["prompt"]
    user_msgs = user if isinstance(user, list) else [{"role": "user", "content": user}]
    prompt = ([{"role": "system", "content": r["system_prompt"]}] if r["system_prompt"] else []) + user_msgs
    completion = [{"role": "assistant", "content": f"```python\n{code.strip()}\n```"}]
    lines.append({"prompt": prompt, "completion": completion, "train_mean": tm})
rng.shuffle(lines)
args.out.parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    for line in lines:
        f.write(json.dumps(line) + "\n")
hist = collections.Counter(round(line["train_mean"], 1) for line in lines)
print("rows:", len(lines), "lineages:", len(by_lineage), sorted(hist.items()))
