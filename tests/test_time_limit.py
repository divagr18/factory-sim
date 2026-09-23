"""A program that never ends must cost seconds, not an hour.

The decision budget does not bound a program's time: queries cost no decision.
Evolved programs that looped on `world.entities()`, or in a search that never
touched `world`, held evaluation workers at 100% CPU until the pool's job
timeout, chunk after chunk, and stalled four evolution runs."""

from __future__ import annotations

import time

from evolve import evaluate, sandbox
from evolve.seeds.builder import SOURCE as SEED
from fsim import scenes
from fsim.program_api import run_episode
from fsim.rl import RlEnv

PURE_LOOP = "def build(world):\n    n = 0\n    while True:\n        n = n + 1\n"
QUERY_LOOP = "def build(world):\n    while True:\n        world.entities()\n"


def _scene(seed=0):
    return scenes.sample("construct_smelting_line", "train", seed)[1]


def test_a_pure_python_loop_is_stopped_and_the_episode_still_ends():
    env = RlEnv()
    t0 = time.monotonic()
    r = run_episode(sandbox.load(PURE_LOOP), _scene(), env=env, time_limit_s=0.3)
    assert time.monotonic() - t0 < 3.0
    assert r.error and r.error.startswith("ProgramTimeLimit")
    assert not r.success  # it built nothing, and the episode was scored anyway


def test_a_loop_that_only_queries_is_stopped():
    r = run_episode(sandbox.load(QUERY_LOOP), _scene(), env=RlEnv(), time_limit_s=0.3)
    assert r.error and r.error.startswith("ProgramTimeLimit")


def test_the_limit_leaks_nothing_into_the_next_episode():
    """An asynchronous exception set but not raised would surface later, far from
    the program. After a stopped program, normal episodes must run clean."""
    env = RlEnv()
    run_episode(sandbox.load(PURE_LOOP), _scene(), env=env, time_limit_s=0.2)
    build = sandbox.load(SEED)
    for seed in range(40):
        family, scene = scenes.sample("construct_smelting_line", "train", seed)
        if scene["entities"]:
            continue
        r = run_episode(build, scene, env=env, time_limit_s=2.0)
        assert r.error is None and r.success, (family, r.error)
    time.sleep(0.3)  # past the stopped program's deadline: no stray exception arrives


def test_a_normal_program_is_untouched_by_the_limit():
    env, build = RlEnv(), sandbox.load(SEED)
    for seed in range(12):
        family, scene = scenes.sample("construct_smelting_line", "train", seed)
        if scene["entities"]:
            continue
        with_limit = run_episode(build, scene, env=env, time_limit_s=2.0)
        without = run_episode(build, scene, env=env)
        assert with_limit.success == without.success
        assert with_limit.decisions == without.decisions


def test_a_chunk_stops_running_a_program_after_repeated_time_limits():
    chunk = [
        (f, s, bp)
        for f, s, bp in (
            (*scenes.sample("construct_smelting_line", "train", i), i) for i in range(10)
        )
    ]
    chunk = [(f, s, bp) for f, bp, s in chunk]
    t0 = time.monotonic()
    out = evaluate.worker_job({"source": QUERY_LOOP, "scenes": chunk, "time_limit_s": 0.25})
    elapsed = time.monotonic() - t0
    errors = [r["error"] for r in out["results"]]
    assert len(errors) == 10
    ran = [e for e in errors if e.startswith("ProgramTimeLimit")]
    skipped = [e for e in errors if e.startswith("skipped")]
    assert len(ran) == evaluate.MAX_TIME_LIMIT_HITS
    assert len(skipped) == 10 - evaluate.MAX_TIME_LIMIT_HITS
    assert elapsed < 5.0, elapsed
