"""The evaluator: pinned scene sets, the stage-1 filter, and what never leaks.

One small pool serves the whole module; the sets are cut down so the file
runs in seconds, and the rates it checks are compared with a plain in-process
`run_episode` loop rather than with numbers written down here.
"""

from __future__ import annotations

import re

import pytest

from evolve import evaluate as ev
from evolve import sandbox
from evolve.pool import EvalPool
from evolve.seeds.builder import SOURCE
from fsim.program_api import run_episode
from fsim.rl import RlEnv

BROKEN = "def build(world):\n    return 1 // 0\n"
IDLE = "def build(world):\n    world.wait()\n"
ESCAPES = "import os\n\ndef build(world):\n    world.wait()\n"


class Recording:
    """An `EvalPool` that remembers every payload it was given."""

    def __init__(self, pool):
        self.pool = pool
        self.workers = pool.workers
        self.payloads = []

    def map(self, payloads):
        self.payloads.extend(payloads)
        return self.pool.map(payloads)


@pytest.fixture(scope="module")
def pool():
    with EvalPool(2, "evolve.evaluate:worker_init", "evolve.evaluate:worker_job", 60) as p:
        yield p


@pytest.fixture(scope="module")
def sets():
    return ev.scene_sets(train_n=24, val_n=16, holdout_n=6)


def digests(items):
    return {ev.scene_digest(bp) for _, _, bp in items}


# ------------------------------------------------------------------ scene sets


def test_scene_sets_are_deterministic_and_disjoint(sets):
    again = ev.scene_sets(train_n=24, val_n=16, holdout_n=6)
    assert again == sets
    assert [s for _, s, _ in sets["train"]] == list(range(24))
    assert [s for _, s, _ in sets["val"]] == list(range(ev.VAL_OFFSET, ev.VAL_OFFSET + 16))
    assert not digests(sets["train"]) & digests(sets["val"])
    assert not digests(sets["holdout"]) & (digests(sets["train"]) | digests(sets["val"]))
    assert {f for f, _, _ in sets["train"] + sets["val"]} <= set(ev.FAMILIES_TRAIN)
    assert {f for f, _, _ in sets["holdout"]} == set(ev.FAMILIES_HOLDOUT)
    assert ev.set_digests(sets) == ev.set_digests(again)


def test_the_holdout_is_factoriorls_frozen_one():
    check = ev.verify_holdout(ev.scene_sets(train_n=0, val_n=0, holdout_n=ev.HOLDOUT_FROZEN))
    if check["file"] is None:
        pytest.skip("no FactorioRL checkout beside this one")
    assert check["matched"] == check["compared"] == ev.HOLDOUT_FROZEN


# ---------------------------------------------------------------------- scores


def test_the_seed_scores_like_a_direct_run_episode_loop(pool, sets):
    [scored] = ev.Evaluator(pool, sets).score([SOURCE], "train")
    env, build = RlEnv(), sandbox.load(SOURCE)
    by_family: dict[str, list[bool]] = {}
    for family, _, blueprint in sets["train"]:
        by_family.setdefault(family, []).append(run_episode(build, blueprint, env=env).success)
    expected = {f: sum(v) / len(v) for f, v in by_family.items()}
    assert scored["rates"] == pytest.approx(expected)
    assert scored["n"] == len(sets["train"]) and scored["error"] is None
    clear = {f for f, _, bp in sets["train"] if not bp["entities"]}
    assert clear and all(scored["rates"][f] == 1.0 for f in clear)
    d = scored["descriptors"]
    assert d["layout_signature"].startswith("D+0+0") and d["mean_decisions"] > 0


def test_full_returns_the_shapes_the_archive_and_prompts_take(pool, sets):
    full = ev.Evaluator(pool, sets).full(SOURCE)
    assert set(full["train"]) <= set(ev.FAMILIES_TRAIN) and set(full["val"]) <= set(
        ev.FAMILIES_TRAIN
    )
    assert full["n"] == {"train": len(sets["train"]), "val": len(sets["val"])}
    assert 0.0 <= full["val_mean"] <= 1.0 and full["error"] is None


def test_a_program_that_always_raises_stops_at_stage_one(pool, sets):
    rec = Recording(pool)
    scored, seed = ev.Evaluator(rec, sets).score([BROKEN, SOURCE], "train")
    assert seed["error"] is None and seed["n"] == len(sets["train"])
    assert scored["error"].startswith("stage 1") and "ZeroDivisionError" in scored["error"]
    assert scored["mean"] == 0.0 and set(scored["rates"].values()) == {0.0}
    stage1 = digests(sets["train"][: ev.STAGE1_SCENES])
    broken_scenes = [
        ev.scene_digest(bp)
        for p in rec.payloads
        if p["source"] == BROKEN
        for _, _, bp in p["scenes"]
    ]
    assert set(broken_scenes) == stage1 and len(broken_scenes) == ev.STAGE1_SCENES


def test_a_sandbox_violation_is_an_error_not_a_crash(pool, sets):
    rec = Recording(pool)
    [scored] = ev.Evaluator(rec, sets).score([ESCAPES], "val")
    assert scored["error"].startswith("sandbox:") and scored["mean"] == 0.0
    assert not any(p["source"] == ESCAPES for p in rec.payloads)
    assert ev.worker_job({"source": ESCAPES, "scenes": sets["train"][:1]})["error"].startswith(
        "sandbox:"
    )
    assert ev.Evaluator(pool, sets).holdout(ESCAPES)["error"].startswith("sandbox:")


def test_traces_never_come_from_the_holdout(pool, sets):
    rec = Recording(pool)
    evaluator = ev.Evaluator(rec, sets)
    full = evaluator.full(IDLE)
    scored = evaluator.score([IDLE], "val")[0]
    held = digests(sets["holdout"])
    assert not any(ev.scene_digest(bp) in held for p in rec.payloads for _, _, bp in p["scenes"])
    assert full["traces"] and set(full["traces"]) <= set(ev.FAMILIES_TRAIN)
    assert all(t[-1].startswith("episode failed") for t in scored["traces"].values())
    holdout = evaluator.holdout(IDLE)
    assert holdout["traces"] == {} and holdout["rates"] == {"obstructed_patch": 0.0}
    assert holdout["n"] == len(sets["holdout"])


def test_a_chunk_the_pool_lost_counts_as_failed_scenes(sets):
    class Lost:
        workers = 3

        def map(self, payloads):
            return [{"error": "timeout after 60s"} for _ in payloads]

    [scored] = ev.Evaluator(Lost(), sets).score([SOURCE], "train")
    assert scored["error"].startswith("stage 1") and "timeout" in scored["error"]


# ------------------------------------------------------------------- reference


def test_api_reference_lists_exactly_the_sandboxs_world_api():
    ref = ev.api_reference()
    listed = set(re.findall(r"world\.(\w+)\(", ref))
    assert listed == sandbox.WORLD_API - ev.COUNTERS
    assert all(f"world.{c}" in ref for c in ev.COUNTERS)
    assert "Entity" in ref and "facing" in ref


def test_layout_signature_is_relative_to_the_first_drill():
    result = {
        "built": [["stone-furnace", 5, 9], ["burner-mining-drill", 5, 7]],
        "facings": ["N", "S"],
    }
    assert ev.layout_signature(result) == "F+0+2N D+0+0S"
    assert ev.layout_signature({"built": [], "facings": []}) == ""
