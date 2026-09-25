"""The RL contract against FactorioRL's recordings (M4).

Every decision of every golden trace, stepped from its recorded action
*vector*: the encoded tensors must hash identically to the ones FactorioRL's
encoder produced from the engine's observation (bit for bit), and the action
mask, goal vector, reward, termination, truncation and success must match.
"""

from __future__ import annotations

import hashlib
import json
import lzma

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


#: Traces whose tensors are known to depart from the engine: the decision whose
#: tensors first differ, all earlier ones matching. None since inserters chase
#: belt items (fsim.c, inserter_chase); logistics_smelting_chain's remaining
#: gap (tests/test_parity.py) is in state the tensors do not carry.
KNOWN_GAPS: dict[str, int] = {}


@pytest.mark.parametrize("name", SCENARIOS)
def test_contract(name):
    found = first_difference(name)
    if name in KNOWN_GAPS:
        assert found is not None and found[0] == KNOWN_GAPS[name], found
    else:
        assert found is None, found


def first_difference(name) -> tuple | None:
    """The first decision whose tensors, mask, goal or transition differ."""
    header, records = read_trace(GOLDEN / f"{name}.jsonl.xz")
    env = RlEnv()
    env.reset(
        header["task"],
        header["blueprint"],
        decision_ticks=header["decision_ticks"],
        max_steps=header["max_decision_steps"],
        construction_tick_limit=header["construction_tick_limit"],
    )
    if check(records[0], env) is not None:
        return 0, check(records[0], env)
    for record in records[1:]:
        t = record["transition"]
        _, reward, terminated, truncated, info = env.step(t["action"]["vector"])
        where = check(record, env)
        if where is None and abs(reward - t["reward"]) >= 1e-12:
            where = f"reward {reward} != {t['reward']}"
        if where is None and (terminated, truncated, info["success"]) != (
            t["terminated"],
            t["truncated"],
            t["success"],
        ):
            where = "termination"
        if where is None:
            for key, value in t["reward_components"].items():
                if abs(info["reward_components"][key] - value) >= 1e-12:
                    where = f"reward component {key}"
        if where is None and info["decode_failure"] != bool(t["action"]["decode_failure"]):
            where = "decode failure"
        if where is not None:
            return record["decision"], where
    return None


# ------------------------------------------------------------------ v3

#: FactorioRL's v3 encoding of the same recorded observations: per scenario, the
#: v3 tensor hashes and mask at each decision (`tools/v3_contract.py` there).
#: The traces were recorded under local-v2, which carries no belt lanes, hands or
#: pickup and drop points; FactorioRL fills those from each record's engine
#: state the way the local-v3 sensor reports them.
V3_GOLDEN = GOLDEN / "v3_contract.json.xz"
V3 = json.loads(lzma.decompress(V3_GOLDEN.read_bytes())) if V3_GOLDEN.is_file() else None


@pytest.mark.skipif(V3 is None, reason="no tests/golden/v3_contract.json.xz")
@pytest.mark.parametrize("name", sorted((V3 or {}).get("scenarios", {})))
def test_contract_v3(name):
    assert first_difference_v3(name) is None


def first_difference_v3(name) -> tuple | None:
    """The first recorded decision whose v3 tensors or mask differ.

    The run is the recorded v1 one -- its vectors decode as they did -- with
    the sensor's entity cap at local-v3's 96, and each decision is also read
    through `observe3`."""
    header, records = read_trace(GOLDEN / f"{name}.jsonl.xz")
    expected = {entry["decision"]: entry for entry in V3["scenarios"][name]}
    env = RlEnv()
    env.reset(
        header["task"],
        header["blueprint"],
        decision_ticks=header["decision_ticks"],
        max_steps=header["max_decision_steps"],
        construction_tick_limit=header["construction_tick_limit"],
        entity_cap=96,
    )
    for record in records:
        if record["decision"] > 0:
            env.step(record["transition"]["action"]["vector"])
        want = expected.get(record["decision"])
        if want is None:
            continue
        obs, mask = env.observe3()
        hashes = tensor_hashes(obs)
        for key, value in want["tensors"].items():
            if hashes[key] != value:
                return record["decision"], f"tensor {key}"
        bits = "".join("1" if b else "0" for b in mask)
        if bits != want["mask"]:
            at = next(i for i, (a, b) in enumerate(zip(bits, want["mask"], strict=True)) if a != b)
            return record["decision"], f"mask bit {at}"
    return None
