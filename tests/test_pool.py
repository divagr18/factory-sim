"""The evaluation pool: ordering, and recovery from every way a job can fail."""

import os
import subprocess
import sys
import time

import pytest

from evolve.pool import EvalPool

JOB = "tests._pool_jobs:dispatch"
INIT = "tests._pool_jobs:init_state"


def _pids(pool):
    return {slot.proc.pid for slot in pool._slots}


@pytest.fixture(scope="module")
def pool():
    with EvalPool(2, INIT, JOB, timeout_s=2.0) as p:
        yield p


def test_order_across_four_workers():
    with EvalPool(4, None, JOB, timeout_s=10.0) as p:
        out = p.map([("jitter", i) for i in range(50)])
    assert [r[0] for r in out] == list(range(50))
    assert len({r[1] for r in out}) > 1  # the work really was spread out


def test_initializer_state_is_seen_by_the_job(pool):
    out = pool.map([("state", i) for i in range(4)])
    assert [r[0] for r in out] == [1000, 1001, 1002, 1003]
    assert all(r[1] != os.getpid() for r in out)


def test_exception_is_an_error_and_keeps_the_worker(pool):
    pids = _pids(pool)
    out = pool.map([("square", 3), ("raise", "bad input"), ("square", 4)])
    assert out == [9, {"error": "ValueError: bad input"}, 16]
    assert _pids(pool) == pids


def test_hard_exit_is_reported_and_replaced(pool):
    pids = _pids(pool)
    out = pool.map([("exit", 3), ("square", 5)])
    assert _pids(pool) != pids
    assert out[0]["error"].startswith("worker crashed")
    assert "3" in out[0]["error"]
    assert out[1] == 25
    assert pool.map([("state", 1), ("state", 2), ("square", 6)])[2] == 36
    assert [r[0] for r in pool.map([("state", 1), ("state", 2)])] == [1001, 1002]


def test_timeout_kills_and_respawns(pool):
    t0 = time.monotonic()
    out = pool.map([("sleep", 30), ("square", 7)])
    assert time.monotonic() - t0 < 15
    assert out == [{"error": "timeout after 2s"}, 49]
    # the replacement ran the initializer again
    assert [r[0] for r in pool.map([("state", 5), ("state", 6)])] == [1005, 1006]


def test_unpicklable_result_and_payload(pool):
    out = pool.map([("unpicklable", 0), (lambda: 0, 1), ("square", 2)])
    assert "not picklable" in out[0]["error"]
    assert "not picklable" in out[1]["error"]
    assert out[2] == 4


def test_many_failures_in_one_map(pool):
    payloads = [("exit", 1), ("sleep", 30), ("raise", "x"), ("exit", 2)] + [
        ("square", i) for i in range(6)
    ]
    out = pool.map(payloads)
    assert out[0]["error"].startswith("worker crashed")
    assert out[1] == {"error": "timeout after 2s"}
    assert out[2] == {"error": "ValueError: x"}
    assert out[3]["error"].startswith("worker crashed")
    assert out[4:] == [i * i for i in range(6)]


def test_hash_seed_is_fixed_in_workers():
    with EvalPool(1, None, JOB) as a:
        ha = a.map([("hash", "abc")])[0]
    with EvalPool(1, None, JOB) as b:
        hb = b.map([("hash", "abc")])[0]
    assert ha == hb
    env = dict(os.environ, PYTHONHASHSEED="0")
    code = "print(hash('abc'))"
    expected = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert ha == int(expected.stdout)


def test_parent_env_is_restored():
    before = os.environ.get("PYTHONHASHSEED")
    with EvalPool(1, None, JOB):
        assert os.environ.get("PYTHONHASHSEED") == before


def test_failing_initializer_reports_per_job():
    with EvalPool(1, "tests._pool_jobs:failing_init", JOB) as p:
        out = p.map([("square", 2), ("square", 3)])
    assert out == [{"error": "initializer failed: RuntimeError: no simulator here"}] * 2


def test_close_twice_and_after_close():
    p = EvalPool(1, None, JOB)
    assert p.map([]) == []
    p.close()
    p.close()
    with pytest.raises(RuntimeError):
        p.map([("square", 1)])
