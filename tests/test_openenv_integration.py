"""The OpenEnv environment in integrations/openenv: scored in process, then over the wire.

Skipped entirely when `openenv` is not installed, so the main suite stays
torch- and server-free. Scene counts are kept small.
"""

from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("openenv.core.env_server")

# Read when the shared evaluation pool is first built, so set before any scoring.
os.environ.setdefault("FSIM_OPENENV_WORKERS", "2")
os.environ.setdefault("FSIM_OPENENV_JOB_TIMEOUT_S", "3")

ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = ROOT / "integrations" / "openenv"


def _load_package():
    """Import integrations/openenv as the package `factory_sim_env`, as an install would."""
    if "factory_sim_env" in sys.modules:
        return sys.modules["factory_sim_env"]
    spec = importlib.util.spec_from_file_location(
        "factory_sim_env", ENV_DIR / "__init__.py", submodule_search_locations=[str(ENV_DIR)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["factory_sim_env"] = module
    spec.loader.exec_module(module)
    return module


pkg = _load_package()
from factory_sim_env.models import FactorySimAction  # noqa: E402
from factory_sim_env.server import scoring  # noqa: E402
from factory_sim_env.server.app import app  # noqa: E402
from factory_sim_env.server.factory_sim_environment import (  # noqa: E402
    FactorySimEnvironment,
    FactorySimHoldoutEnvironment,
)

from evolve import sandbox  # noqa: E402
from evolve.seeds.builder import SOURCE as SEED  # noqa: E402
from fsim.obsview import Entity  # noqa: E402
from fsim.program_api import run_episode  # noqa: E402

N = 8


@pytest.fixture(scope="module", autouse=True)
def _pool():
    yield
    scoring.close_pool()


# ------------------------------------------------------------------ program mode


def test_reset_gives_the_task_and_the_api():
    env = FactorySimEnvironment()
    obs = env.reset(seed=0, n_scenes=N)
    assert not obs.done and obs.mode == "program" and obs.split == "train"
    assert "construct_smelting_line" in obs.prompt and "world.place(" in obs.api_reference
    assert len(obs.scenes) == N and env.state.seed == 0
    again = FactorySimEnvironment().reset(seed=0, n_scenes=N)
    assert again.scenes == obs.scenes  # deterministic in the seed


def test_seed_program_scores_high():
    env = FactorySimEnvironment()
    env.reset(seed=0, n_scenes=N)
    obs = env.step(FactorySimAction(program=SEED))
    assert obs.done and obs.error is None
    assert obs.reward >= 0.75 and obs.reward == obs.success_rate
    assert obs.n_scenes == N and obs.family_rates
    assert "Success:" in obs.text
    assert env.state.done and env.state.reward == obs.reward


def test_a_fenced_completion_is_unwrapped():
    env = FactorySimEnvironment()
    env.reset(seed=0, n_scenes=N)
    completion = "Plan: follow the expert.\n```python\n" + SEED.strip("\n") + "\n```\n"
    assert env.step(FactorySimAction(program=completion)).reward >= 0.75


def test_sandbox_violation_gets_zero_and_the_reason():
    env = FactorySimEnvironment()
    env.reset(seed=1, n_scenes=N)
    obs = env.step(FactorySimAction(program="import os\n\ndef build(world):\n    pass\n"))
    assert obs.done and obs.reward == 0.0
    assert obs.error.startswith("sandbox:") and "import is not allowed" in obs.error
    obs = FactorySimEnvironment()
    obs.reset(seed=1, n_scenes=N)
    dunder = "def build(world):\n    return ().__class__\n"
    assert "not allowed" in obs.step(FactorySimAction(program=dunder)).error


def test_a_crashing_program_is_handled():
    env = FactorySimEnvironment()
    env.reset(seed=2, n_scenes=N)
    obs = env.step(FactorySimAction(program="def build(world):\n    return 1 // 0\n"))
    assert obs.done and obs.reward == 0.0 and "ZeroDivisionError" in obs.error


def test_a_hanging_program_times_out():
    env = FactorySimEnvironment()
    env.reset(seed=3, n_scenes=N)
    obs = env.step(FactorySimAction(program="def build(world):\n    while True:\n        pass\n"))
    assert obs.done and obs.reward == 0.0 and "timeout" in obs.error
    # and the pool still serves the next episode
    env.reset(seed=0, n_scenes=N)
    assert env.step(FactorySimAction(program=SEED)).reward >= 0.75


def test_holdout_is_only_on_its_own_server_and_has_no_traces():
    with pytest.raises(ValueError, match="holdout"):
        FactorySimEnvironment().reset(split="holdout")
    env = FactorySimHoldoutEnvironment()
    obs = env.reset(n_scenes=N)
    assert obs.split == "holdout" and {s["family"] for s in obs.scenes} == {"obstructed_patch"}
    trivial = "def build(world):\n    world.wait()\n"
    obs = env.step(FactorySimAction(program=trivial))
    assert obs.reward == 0.0 and obs.failure_traces == {}
    with pytest.raises(ValueError):
        FactorySimHoldoutEnvironment().reset(split="train")


def test_val_split_and_step_after_done():
    env = FactorySimEnvironment()
    env.reset(split="val", n_scenes=4, seed=5)
    env.step(FactorySimAction(program=SEED))
    obs = env.step(FactorySimAction(program=SEED))
    assert obs.done and obs.reward == 0.0 and "call reset" in obs.error


def test_actions_must_pick_one_mode():
    with pytest.raises(ValueError):
        FactorySimAction()
    with pytest.raises(ValueError):
        FactorySimAction(program="x", method="wait")


# ------------------------------------------------------------------ tool mode


def _recorded_calls(blueprint: dict) -> list[tuple[str, list]]:
    """The seed program's World actions on `blueprint`, entities as their row."""
    build = sandbox.load(SEED)
    calls: list[tuple[str, list]] = []

    class Recorder:
        def __init__(self, world):
            self._world = world

        def __getattr__(self, name):
            attr = getattr(self._world, name)
            if name not in scoring.TOOL_ACTIONS:
                return attr

            def act(*args):
                args = list(args)
                if args and isinstance(args[0], Entity):
                    args[0] = self._world._row(args[0])
                calls.append((name, args))
                return attr(*args)

            return act

    result = run_episode(lambda world: build(Recorder(world)), blueprint, task=scoring.TASK)
    assert result.success
    return calls


def test_tool_mode_scripted_build_succeeds():
    seed = 7
    env = FactorySimEnvironment()
    obs = env.reset(mode="tool", seed=seed)
    assert obs.mode == "tool" and obs.world["inventory"]["burner-mining-drill"] == 2
    _, _, blueprint = scoring.select_scenes("train", 1, seed)[0]
    calls = _recorded_calls(blueprint)
    for method, args in calls:
        obs = env.step(FactorySimAction(method=method, args=args))
        assert obs.error is None, obs.error
        if obs.done:
            break
    if not obs.done:
        obs = env.step(FactorySimAction(method="finish"))
    assert obs.done and obs.reward == 1.0 and obs.verified_output >= 10


def test_tool_mode_validates_calls():
    env = FactorySimEnvironment()
    env.reset(mode="tool", seed=4)
    obs = env.step(FactorySimAction(method="teleport", args=[0, 0]))
    assert not obs.done and "unknown method" in obs.error
    obs = env.step(FactorySimAction(method="place", args=["stone-furnace"]))
    assert "missing" in obs.error
    obs = env.step(FactorySimAction(method="move", args=["up", "long"]))
    assert obs.error is None and obs.result is False  # World refused it: no decision
    assert obs.world["refusals"] == 1 and env.state.decisions == 0
    obs = env.step(FactorySimAction(method="finish"))
    assert obs.done and obs.reward == 0.0


# ------------------------------------------------------------------ in-process WebSocket


def test_websocket_session_in_process():
    from starlette.testclient import TestClient

    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "healthy"
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset", "data": {"seed": 0, "n_scenes": 4}})
            reset = ws.receive_json()
            assert reset["type"] == "observation"
            assert len(reset["data"]["observation"]["scenes"]) == 4
            ws.send_json({"type": "step", "data": {"program": SEED}})
            step = ws.receive_json()
            assert step["data"]["done"] and step["data"]["reward"] >= 0.75


# ------------------------------------------------------------------ over the wire


def _free_port() -> int | None:
    try:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except OSError:
        return None


@pytest.fixture(scope="module")
def server_url():
    port = _free_port()
    if port is None:
        pytest.skip("cannot bind a local port")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1",
         "--port", str(port)],
        cwd=ENV_DIR, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.skip(f"server exited: {proc.stderr.read().decode(errors='replace')[-500:]}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=1) as r:
                if r.status == 200:
                    break
        except OSError:
            time.sleep(0.3)
    else:
        proc.kill()
        pytest.skip("server did not come up")
    yield url
    proc.terminate()
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_protocol_round_trip(server_url):
    with pkg.FactorySimEnv(base_url=server_url).sync() as env:
        r = env.reset(seed=0, n_scenes=4)
        assert not r.done and "construct_smelting_line" in r.observation.prompt
        r = env.step(FactorySimAction(program=SEED))
        assert r.done and r.reward >= 0.75 and r.observation.family_rates
        state = env.state()
        assert state.done and state.split == "train" and len(state.scenes) == 4

        r = env.reset(mode="tool", seed=3)
        assert r.observation.world is not None
        r = env.step(FactorySimAction(method="move", args=["E", "long"]))
        assert r.observation.result is True and not r.done
        r = env.step(FactorySimAction(method="finish"))
        assert r.done and r.reward == 0.0


def test_trl_wrappers(server_url, monkeypatch):
    from factory_sim_env import trl_envs

    monkeypatch.setenv("FSIM_OPENENV_URL", server_url)
    env = trl_envs.ProgramToolEnv()
    try:
        prompt = env.reset(seed=0, n_scenes=4)
        assert "def build(world)" in prompt
        feedback = env.submit_program(SEED)
        assert "Success:" in feedback and env.reward >= 0.75
        assert trl_envs.reward_func([env]) == [env.reward]
        with pytest.raises(ValueError):
            env.submit_program(SEED)
    finally:
        env._close()
    tools = trl_envs.WorldToolEnv()
    try:
        tools.reset(seed=3)
        assert "World:" in tools.move("E", "long")
        assert "refused" in tools.mine(99)  # no such row: World refuses, no decision spent
        with pytest.raises(ValueError, match="integer row"):
            tools.give("the furnace", "coal", 5)  # invalid call: raised, as TRL expects
    finally:
        tools._close()
