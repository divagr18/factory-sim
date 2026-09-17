"""The RL contract against FactorioRL's recordings (M4).

Every decision of every golden trace, stepped from its recorded action
*vector*: the encoded tensors must hash identically to the ones FactorioRL's
encoder produced from the engine's observation (bit for bit), and the action
mask, goal vector, reward, termination, truncation and success must match.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from fsim.parity import GOLDEN
from fsim.rl import RlEnv
from fsim.trace import read_trace

INDEX = json.loads((GOLDEN / "index.json").read_text(encoding="utf-8"))
SCENARIOS = sorted(INDEX)


def tensor_hashes(obs: dict) -> dict:
    out = {}
    for key in sorted(obs):
        array = np.ascontiguousarray(obs[key])
        digest = hashlib.sha256()
        digest.update(f"{array.dtype.str}|{array.shape}|".encode())
        digest.update(array.tobytes())
        out[key] = digest.hexdigest()[:16]
    return out


def check(record, env) -> str | None:
    hashes = tensor_hashes(env.obs)
    for key, value in record["tensors"].items():
        if hashes[key] != value:
            return f"tensor {key}"
    mask = "".join("1" if b else "0" for b in env.mask)
    if mask != record["mask"]:
        at = next(i for i, (a, b) in enumerate(zip(mask, record["mask"], strict=True)) if a != b)
        return f"mask bit {at}"
    goal = [float(v) for v in env.obs["goal"]]
    if goal != record["goal"]:
        return f"goal {goal} != {record['goal']}"
    return None


@pytest.mark.parametrize("name", SCENARIOS)
def test_contract(name):
    header, records = read_trace(GOLDEN / f"{name}.jsonl.xz")
    env = RlEnv()
    env.reset(
        header["task"],
        header["blueprint"],
        decision_ticks=header["decision_ticks"],
        max_steps=header["max_decision_steps"],
        construction_tick_limit=header["construction_tick_limit"],
    )
    assert check(records[0], env) is None, (0, check(records[0], env))
    for record in records[1:]:
        t = record["transition"]
        _, reward, terminated, truncated, info = env.step(t["action"]["vector"])
        where = check(record, env)
        assert where is None, (record["decision"], where)
        assert abs(reward - t["reward"]) < 1e-12, (record["decision"], reward, t["reward"])
        assert terminated == t["terminated"], record["decision"]
        assert truncated == t["truncated"], record["decision"]
        assert info["success"] == t["success"], record["decision"]
        for key, value in t["reward_components"].items():
            assert abs(info["reward_components"][key] - value) < 1e-12, (record["decision"], key)
        assert info["decode_failure"] == bool(t["action"]["decode_failure"]), record["decision"]
