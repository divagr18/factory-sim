"""tools/evolve_curves.py: best-so-far replay, decision accounting, holdout cache, arms."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import evolve_curves as ec  # noqa: E402

from evolve import evaluate as ev  # noqa: E402
from evolve.archive import Candidate, Store  # noqa: E402
from evolve.pool import EvalPool  # noqa: E402
from evolve.seeds.builder import SOURCE as SEED  # noqa: E402

TRIVIAL = "def build(world):\n    world.wait()\n"
LOOP = "def build(world):\n    # loop\n    for _ in range(3):\n        world.wait()\n"
HOLDOUT_N = 8


def _cand(code, op, val, episodes, t, parents=()):
    scores = {"train": {}, "val": {}, "train_mean": val, "val_mean": val}
    if episodes is not None:
        scores["episodes"] = episodes
    c = Candidate.new(code, operator=op, parents=list(parents), scores=scores)
    return replace(c, created=float(t))


def genealogy(seed_code=SEED, best_code=TRIVIAL):
    """seed 0.5; failed row; worse child; child 0.75; shorter tie at 0.75; child 1.0."""
    seed = _cand(seed_code, "seed", 0.5, 392, 1)
    return [
        seed,
        _cand("def build(world) oops", "sandbox_error", 0.0, None, 2, [seed.id]),
        _cand(seed_code + "\n# worse\n", "diff", 0.25, 392, 3, [seed.id]),
        _cand(LOOP + "# padding\n", "diff", 0.75, 392, 4, [seed.id]),
        _cand(LOOP, "rewrite", 0.75, 392, 5, [seed.id]),
        _cand(LOOP + "# same score, longer\n", "diff", 0.75, 392, 6, [seed.id]),
        _cand(best_code, "crossover", 1.0, 8, 7, [seed.id]),
    ]


def write_run(root: Path, name: str, rows, status=None) -> Path:
    d = root / f"evolve-{name}"
    with Store(d / "genealogy.sqlite") as store:
        for c in rows:
            store.add(c)
    if status is not None:
        (d / "status.json").write_text(json.dumps(status), encoding="utf-8")
    return d


# ------------------------------------------------------------------ pure


def test_replay_best_so_far_and_decisions():
    seen = []

    def score(code):
        seen.append(code)
        return {SEED: 0.2, TRIVIAL: 0.0}.get(code, 0.95)

    rows = genealogy()
    out = ec.replay(rows, score, duplicates=3)
    ids = [p["candidate_id"] for p in out["curve"]]
    assert ids == [rows[0].id, rows[3].id, rows[6].id]  # the shorter tie does not displace
    dec = [p["sim_decisions"] for p in out["curve"]]
    assert dec == [392 * 600, 3 * 392 * 600, (5 * 392 + 8) * 600]
    assert [p["rows"] for p in out["curve"]] == [0, 3, 6]
    assert [p["val_mean"] for p in out["curve"]] == [0.5, 0.75, 1.0]
    assert out["totals"]["episodes"] == 5 * 392 + 8
    assert out["totals"]["verify_decisions"] == (5 * 392 + 8) * 120
    assert out["totals"]["failed_rows"] == 1
    assert out["totals"]["completions"] == 6 + 3
    assert out["final"]["candidate_id"] == rows[6].id
    assert out["final"]["completions"] == 9
    assert out["curve"][-1]["completions"] == 6 + 3  # prorated duplicates
    assert len(seen) == 3  # the failed, non-improving and tied rows are never scored
    assert ec.first_reach(out["curve"], 0.9) == 3 * 392 * 600
    assert ec.first_reach(out["curve"], 1.0) == "never"


def test_arm_parsing():
    assert ec.arm_of("grid-A-s1") == "A"
    assert ec.arm_of("grid-d-s12") == "D"
    assert ec.arm_of("smoke2") == "?"


def test_arms_aggregate_across_run_names(tmp_path):
    fake = {SEED: 0.5, TRIVIAL: 0.0}
    runs = [
        ec.analyse_run(write_run(tmp_path, n, genealogy(best_code=b)), lambda c: fake.get(c, 1.0))
        for n, b in (("grid-A-s1", TRIVIAL), ("grid-A-s2", LOOP + "#x\n#y\n"), ("grid-B-s1", LOOP))
    ]
    arms = ec.aggregate_arms(runs)
    assert sorted(arms) == ["A", "B"]
    a = arms["A"]
    assert a["runs"] == ["grid-A-s1", "grid-A-s2"]
    assert a["final_holdout"] == {"grid-A-s1": 0.0, "grid-A-s2": 1.0}
    assert a["final_holdout_mean"] == 0.5
    assert (a["final_holdout_min"], a["final_holdout_max"]) == (0.0, 1.0)
    assert a["runs_reaching_1.0"] == "2/2"
    assert arms["B"]["final_holdout"] == {"grid-B-s1": 1.0}
    assert a["curves"]["grid-A-s1"][0] == [392 * 600, 0.5]


# ------------------------------------------------------------------ real pool


@pytest.fixture(scope="module")
def pool():
    with EvalPool(2, "evolve.evaluate:worker_init", "evolve.evaluate:worker_job", 120) as p:
        yield p


def test_holdout_cache_scores_each_code_once(tmp_path, pool):
    sets = ev.scene_sets(train_n=0, val_n=0, holdout_n=HOLDOUT_N)
    cache = ec.HoldoutCache(ev.Evaluator(pool, sets))
    dirs = [
        write_run(tmp_path, "grid-A-s1", genealogy(), {"duplicates": 2, "run_spent_usd": 0.5}),
        write_run(tmp_path, "grid-A-s2", genealogy()),
        write_run(tmp_path, "grid-C-s1", genealogy(seed_code=TRIVIAL, best_code=SEED)),
    ]
    report = ec.build_report(dirs, cache.score, [], HOLDOUT_N)
    # SEED, LOOP + padding, TRIVIAL: three distinct codes over three runs
    assert cache.calls == 3
    by = {r["run"]: r for r in report["runs"]}
    assert by["grid-A-s1"]["curve"][0]["holdout"] == by["grid-C-s1"]["final"]["holdout"]
    assert by["grid-A-s1"]["final"]["holdout"] == 0.0  # waiting builds nothing
    assert by["grid-A-s1"]["run_spent_usd"] == 0.5
    assert by["grid-A-s1"]["totals"]["completions"] == 8
    assert all(0.0 <= p["holdout"] <= 1.0 for r in report["runs"] for p in r["curve"])


def test_main_writes_json_svg_and_table(tmp_path, capsys):
    a = write_run(tmp_path, "grid-A-s1", genealogy(), {"elapsed_s": 5400, "run_spent_usd": 1.25})
    d = write_run(tmp_path, "grid-D-s1", genealogy(seed_code=TRIVIAL, best_code=SEED))
    ppo = tmp_path / "ppo.json"
    ppo.write_text(
        json.dumps([{"run": "ppo-s1", "mode": "greedy", "holdout": 0.93, "steps": 40_000_000}]),
        encoding="utf-8",
    )
    out, svg = tmp_path / "out.json", tmp_path / "out.svg"
    rc = ec.main(
        [
            str(a),
            str(d),
            "--holdout-n",
            str(HOLDOUT_N),
            "--workers",
            "2",
            "--out",
            str(out),
            "--svg",
            str(svg),
            "--ppo",
            str(ppo),
        ]
    )
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert set(report["arms"]) == {"A", "D"}
    assert report["holdout_programs_scored"] == 3
    assert report["tie_break"] == "incumbent" and report["holdout_start"] == 0
    assert report["ppo"][0]["steps"] == 40_000_000
    text = svg.read_text(encoding="utf-8")
    assert text.startswith("<svg") and text.count("<polyline") == 2 and "PPO" in text
    printed = capsys.readouterr().out
    assert "grid-A-s1" in printed and "ppo-s1" in printed


def test_replay_length_rule_reproduces_runs_before_evaluator_3():
    rows = genealogy()
    out = ec.replay(rows, lambda code: 0.5, tie_break="length")
    ids = [p["candidate_id"] for p in out["curve"]]
    assert ids == [rows[0].id, rows[3].id, rows[4].id, rows[6].id]
