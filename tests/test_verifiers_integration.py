"""The Prime Intellect `verifiers` environments in integrations/verifiers/.

Offline only: no model is called. Completions are constructed by hand and
scored through verifiers' own scoring entry points (`Rubric.score_rollout`
for the v0 `load_environment`, `Task.score` -- what `replay` runs -- for the
v1 taskset). Skipped entirely when `verifiers` is not installed, and the v1
tests are skipped where `verifiers.v1` cannot import (Windows: it needs fcntl).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("verifiers")
pytest.importorskip("datasets")

ROOT = Path(__file__).resolve().parents[1]
for sub in ("factorio_build", "factorio_play"):
    path = str(ROOT / "integrations" / "verifiers" / sub)
    if path not in sys.path:
        sys.path.insert(0, path)

import factorio_build  # noqa: E402
import factorio_play  # noqa: E402
from factorio_build import core  # noqa: E402
from factorio_play.session import WorldSession  # noqa: E402

from evolve import evaluate, mutate, sandbox  # noqa: E402
from evolve.seeds import builder, trivial  # noqa: E402
from fsim import scenes  # noqa: E402
from fsim.obsview import Entity  # noqa: E402
from fsim.program_api import run_episode  # noqa: E402

needs_v1 = pytest.mark.skipif(
    factorio_build.V1_IMPORT_ERROR is not None,
    reason=f"verifiers.v1 unavailable: {factorio_build.V1_IMPORT_ERROR}",
)

TASK = "construct_smelting_line"
SANDBOX_VIOLATION = "import os\n\ndef build(world):\n    os.system('echo hi')\n"


def fenced(source: str) -> str:
    plan = "Plan: walk to the patch, place a drill and a furnace, fuel both."
    return f"{plan}\n```python\n{source}\n```"


COMPLETIONS = {
    "seed": fenced(builder.SOURCE),
    "nothing": fenced(trivial.SOURCE),
    "violation": fenced(SANDBOX_VIOLATION),
    "no_code": "I would walk to the ore patch and build a drill there.",
}


# ------------------------------------------------------------------ dataset rows


def test_rows_are_well_formed():
    rows = core.rows(TASK, "train", n_scenes=8, num_examples=4, seed=0)
    assert len(rows) == 4
    seen = set()
    for k, row in enumerate(rows):
        assert row["subset_id"] == f"{TASK}/train/{8 * k}-{8 * k + 7}"
        assert len(row["scenes"]) == 8
        for family, seed, sample_split in row["scenes"]:
            assert sample_split == "train"
            assert family in evaluate.FAMILIES_TRAIN
            assert scenes.sample(TASK, sample_split, seed)[0] == family
            seen.add(seed)
        assert "def build(world)" in row["system_prompt"]
        assert "world.place" in row["system_prompt"]  # the API reference
        assert row["subset_id"] in row["prompt"]
    assert seen == set(range(32))  # consecutive, disjoint subsets


def test_load_environment_dataset():
    env = factorio_build.load_environment(num_examples=3, n_scenes=8, workers=0)
    ds = env.dataset
    assert len(ds) == 3
    row = ds[0]
    roles = [m["role"] for m in row["prompt"]]
    assert roles == ["system", "user"]
    assert row["prompt"][0]["content"] == core.system_prompt(True)
    assert row["info"]["subset_id"] == f"{TASK}/train/0-7"
    assert len(row["info"]["scenes"]) == 8


def test_game_notes_toggle():
    assert mutate.GAME_NOTES in core.system_prompt(True)
    without = core.system_prompt(False)
    assert "Game notes:" not in without
    assert "def build(world)" in without


def test_only_construct_smelting_line_is_offered():
    with pytest.raises(ValueError):
        core.rows("plate_line", "train", 8, 1)


# ------------------------------------------------------------------ holdout


def test_holdout_is_opt_in():
    default = factorio_build.load_environment(num_examples=2, n_scenes=8, workers=0)
    for split in ("train", "val"):
        rows = core.rows(TASK, split, 16, 8)
        assert all(ss == "train" for r in rows for _, _, ss in r["scenes"])
        assert all(f not in evaluate.FAMILIES_HOLDOUT for r in rows for f, _, _ in r["scenes"])
    assert all(r["split"] == "train" for r in default.dataset["info"])
    held = core.rows(TASK, "holdout", 10, 2)
    held_scenes = [s for r in held for s in r["scenes"]]
    assert all(f in evaluate.FAMILIES_HOLDOUT and ss == "test" for f, _, ss in held_scenes)
    # The same scenes evaluate.scene_sets pins as the holdout.
    pinned = [s for _, s, _ in evaluate.scene_sets(train_n=1, val_n=1, holdout_n=20)["holdout"]]
    assert [s for r in held for _, s, _ in r["scenes"]] == pinned


def test_val_does_not_repeat_train():
    train = {s for r in core.rows(TASK, "train", 16, 8) for _, s, _ in r["scenes"]}
    val = {s for r in core.rows(TASK, "val", 16, 8) for _, s, _ in r["scenes"]}
    assert not train & val


# ------------------------------------------------------------------ v0 rubric


def _score_v0(env, text: str) -> dict:
    row = env.dataset[0]
    state = {
        "prompt": row["prompt"],
        "completion": [{"role": "assistant", "content": text}],
        "answer": "",
        "info": row["info"],
        "task": row["task"],
        "trajectory": [],
    }
    asyncio.run(env.rubric.score_rollout(state))
    return state


@pytest.fixture(scope="module")
def v0_env():
    return factorio_build.load_environment(num_examples=1, n_scenes=12, workers=0)


def test_v0_seed_program_scores_high(v0_env):
    st = _score_v0(v0_env, COMPLETIONS["seed"])
    assert st["metrics"]["success_rate"] >= 0.75
    assert st["metrics"]["format"] == 1.0
    assert st["reward"] == pytest.approx(st["metrics"]["success_rate"] + 0.1)


def test_v0_program_that_builds_nothing(v0_env):
    st = _score_v0(v0_env, COMPLETIONS["nothing"])
    assert st["metrics"]["success_rate"] == 0.0
    assert st["reward"] == pytest.approx(0.1)  # only the format bonus


def test_v0_sandbox_violation_is_penalised(v0_env):
    st = _score_v0(v0_env, COMPLETIONS["violation"])
    assert st["metrics"]["success_rate"] == 0.0
    assert st["metrics"]["format"] == -1.0
    assert st["reward"] == pytest.approx(-0.1)
    assert "sandbox" in st["factorio_build"]["error"]


def test_v0_no_code_block_scores_zero(v0_env):
    st = _score_v0(v0_env, COMPLETIONS["no_code"])
    assert st["reward"] == 0.0
    assert st["metrics"]["format"] == 0.0


# ------------------------------------------------------------------ pool


def test_pool_matches_in_process_and_kills_runaways():
    scene_list = core.rows(TASK, "train", 8, 1)[0]["scenes"]
    inproc = core.score_completion(COMPLETIONS["seed"], TASK, scene_list, workers=0)
    pooled = core.score_completion(COMPLETIONS["seed"], TASK, scene_list, workers=2)
    assert pooled["success"] == inproc["success"]
    assert pooled["mean_verified_output"] == inproc["mean_verified_output"]
    hang = fenced("def build(world):\n    while True:\n        x = 1\n")
    # Stopped either by the evaluator's per-episode watchdog or by the pool's
    # job timeout, whichever fires first; both score the scene as a failure.
    for workers in (1, 0):
        stuck = core.score_completion(hang, TASK, scene_list[:2], workers=workers, timeout_s=3.0)
        assert stuck["sandbox_valid"] == 1.0
        assert stuck["success"] == 0.0
        assert "timeout" in stuck["error"] or "ProgramTimeLimit" in stuck["error"]
    core.close_pools()


# ------------------------------------------------------------------ tool session


class JsonWorld:
    """What a model sees through MCP: JSON in, JSON out, entities by row."""

    def __init__(self, call):
        self._call = call

    def _r(self, name, *args):
        out = self._call(name, *args)
        if not out["ok"]:
            raise RuntimeError(out["error"])
        return out["result"]

    def me(self):
        return tuple(self._r("me"))

    def tile(self):
        return tuple(self._r("tile"))

    def patch(self):
        p = self._r("patch")
        return tuple(p) if p is not None else None

    def inventory(self):
        return self._r("inventory")

    def ore_tiles(self, kind="iron-ore"):
        return [tuple(t) for t in self._r("ore_tiles", kind)]

    def blocked_tiles(self):
        return [tuple(t) for t in self._r("blocked_tiles")]

    def entities(self):
        return [Entity(**e) for e in self._r("entities")]

    def decisions_left(self):
        return self._r("decisions_left")

    def last_refused(self):
        return self._r("last_refused")

    def move(self, direction, stride="long"):
        return self._r("move", direction, stride)

    def place(self, item, x, y, facing):
        return self._r("place", item, x, y, facing)

    def give(self, entity, item, amount):
        return self._r("give", entity.row if isinstance(entity, Entity) else entity, item, amount)

    def take(self, entity, item, amount):
        return self._r("take", entity.row if isinstance(entity, Entity) else entity, item, amount)

    def mine(self, entity):
        return self._r("mine", entity.row if isinstance(entity, Entity) else entity)

    def wait(self):
        return self._r("wait")


def _drive(call, source):
    try:
        sandbox.load(source)(JsonWorld(call))
    except RuntimeError:
        pass  # the episode ended under the program, as BudgetExhausted would


def test_session_reproduces_run_episode_exactly():
    for seed in range(4):
        _, bp = scenes.sample(TASK, "train", seed)
        direct = run_episode(sandbox.load(builder.SOURCE), bp, task=TASK)
        session = WorldSession(scenes.sample(TASK, "train", seed)[1], task=TASK)
        _drive(session.call, builder.SOURCE)
        r = session.finish()
        assert (r.success, r.verified_output, r.decisions, r.built) == (
            direct.success,
            direct.verified_output,
            direct.decisions,
            direct.built,
        )
        assert session.call("wait")["ok"] is False  # nothing after finish


def test_session_ends_when_budget_runs_out():
    session = WorldSession(scenes.sample(TASK, "train", 0)[1], task=TASK, decision_budget=3)
    assert all(session.call("wait")["ok"] for _ in range(3))
    out = session.call("wait")
    assert out["ok"] is False and "BudgetExhausted" in out["error"]
    assert session.finished and session.result.success is False


# ------------------------------------------------------------------ v1


def _trace(task, text=None, state=None):
    import verifiers.v1 as vf

    nodes = []
    if text is not None:
        nodes = [vf.MessageNode(message=vf.AssistantMessage(content=text), sampled=True)]
    kw = {"state": state} if state is not None else {}
    return vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data, key=task.key, hash=task.hash),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=nodes,
        **kw,
    )


@pytest.fixture(scope="module")
def v1_task():
    from factorio_build.taskset import FactorioBuildConfig, FactorioBuildTaskConfig

    cfg = FactorioBuildConfig(num_examples=2, n_scenes=12, task=FactorioBuildTaskConfig(workers=0))
    tasks = list(factorio_build.FactorioBuildTaskset(cfg))
    assert len(tasks) == 2
    return tasks[0]


@needs_v1
def test_v1_taskset_rows(v1_task):
    from factorio_build.taskset import FactorioBuildConfig

    d = v1_task.data
    assert d.split == "train" and d.subset_id == f"{TASK}/train/0-11"
    assert d.system_prompt == core.system_prompt(True)
    assert len(d.scenes) == 12
    assert FactorioBuildConfig().split == "train"  # holdout only when asked for


@needs_v1
@pytest.mark.parametrize(
    "name, success, fmt",
    [("seed", None, 1.0), ("nothing", 0.0, 1.0), ("violation", 0.0, -1.0), ("no_code", 0.0, 0.0)],
)
def test_v1_rewards(v1_task, name, success, fmt):
    trace = _trace(v1_task, COMPLETIONS[name])
    asyncio.run(v1_task.score(trace))
    rewards = {k: r.score for k, r in trace.rewards.items()}
    if success is None:
        assert rewards["success_rate"] >= 0.75
    else:
        assert rewards["success_rate"] == success
    assert rewards["format"] == fmt
    assert trace.reward == pytest.approx(rewards["success_rate"] + 0.1 * fmt)
    assert trace.info["factorio_build"]["subset_id"] == v1_task.data.subset_id


@needs_v1
def test_v1_validate_and_default_harness(v1_task):
    from verifiers.v1.configs.env import default_agent_harness

    assert asyncio.run(v1_task.validate())
    assert default_agent_harness("factorio-build").id == "factorio-build"


@needs_v1
def test_v1_tools_toolset_offline():
    from factorio_play.servers.world import WorldToolset
    from factorio_play.taskset import FactorioPlayConfig

    tasks = list(factorio_play.FactorioPlayTaskset(FactorioPlayConfig(num_examples=2)))
    task = tasks[0]
    toolset = WorldToolset(task.config.tools)
    asyncio.run(toolset.setup_task(task.data))

    def call(name, *args):
        return getattr(toolset, name)(*args)

    _drive(call, builder.SOURCE)
    done = toolset.finish()
    trace = _trace(task, state=toolset.state)
    asyncio.run(task.score(trace))
    assert trace.rewards["success"].score == float(done["success"])
    assert trace.metrics["finished"] == 1.0
    direct = run_episode(
        sandbox.load(builder.SOURCE), scenes.sample(TASK, "train", task.data.seed)[1], task=TASK
    )
    assert done["success"] == direct.success
    assert done["verified_output"] == direct.verified_output


def test_play_scenes_match_factorio_build():
    """factorio-play carries its own copy of the seed plan so it depends on
    factory-sim alone; it must pick exactly the scenes factorio-build does."""
    from factorio_play import scenes as play_scenes

    for split, start, n in (
        ("train", 0, 5),
        ("val", 3, 5),
        ("holdout", 0, 5),
        ("holdout", 1000, 3),
    ):
        a = core.scene_block(TASK, split, start, n)
        b = play_scenes.scene_block(TASK, split, start, n)
        assert [(r.family, r.seed, r.sample_split) for r in a] == [
            (r.family, r.seed, r.sample_split) for r in b
        ]


def test_session_accepts_compass_words_and_says_why_an_action_is_refused():
    """A live model wrote move("east", "long") and place(..., "S") and got a bare
    False back every time: World takes N/E/S/W only, and the reason never reached
    the model. Words now work, and every refusal carries its reason."""
    session = WorldSession(scenes.sample(TASK, "train", 0)[1], task=TASK)
    start = session.call("me")["result"]
    moved = session.call("move", "East", "Long")
    assert moved == {"ok": True, "result": True}
    assert session.call("me")["result"] != start

    bad = session.call("move", "sideways", "long")
    assert bad["result"] is False and "N/E/S/W" in bad["refused"]

    far = session.call("place", "stone-furnace", 999, 999, "south")
    assert far["result"] is False and "tiles from tile()" in far["refused"]
    session.finish()
