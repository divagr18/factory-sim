"""The v3 builder-program API (`WorldV3`) and the task-aware prompt and evaluator.

`belt_smelting`'s scenes are not ported yet, so the World is exercised on
construct_smelting_line scenes reset into the v3 action space, with belts,
inserters and a chest added to the inventory. What is pinned here is what the
program can do and see; the tensors themselves are pinned by the v3 contract
test against FactorioRL's encoder.
"""

from __future__ import annotations

import pytest

from evolve import evaluate, mutate, sandbox
from fsim import scenes
from fsim.program_api import (
    TASK_PROFILES,
    EntityV3,
    SimBackend,
    World,
    WorldV3,
    play,
    world_class,
)
from fsim.rl import RlEnv

EXTRA = {"transport-belt": 10, "burner-inserter": 2, "wooden-chest": 1}


def _scene(seed=3):
    _, bp = scenes.sample("construct_smelting_line", "train", seed)
    bp = dict(bp)
    bp["character"] = dict(bp["character"])
    bp["character"]["inventory"] = {**bp["character"]["inventory"], **EXTRA}
    return bp


def _run(build, seed=3, budget=40):
    env = RlEnv()
    env.reset("construct_smelting_line", _scene(seed), action_space="v3")
    return play(build, SimBackend(env), decision_budget=budget, world_cls=WorldV3,
                markers=("patch",))  # fmt: skip


def test_task_profiles():
    assert TASK_PROFILES["belt_smelting"] == "v3"
    assert world_class("belt_smelting") is WorldV3
    assert world_class("construct_smelting_line") is World


def test_place_rotate_and_read_a_line():
    seen = {}

    def build(w):
        tx, ty = w.tile()
        assert w.inventory()["boiler"] == 0 and w.inventory()["transport-belt"] == 10
        assert w.place("transport-belt", tx + 6, ty, "E") and not w.last_refused()
        assert w.place("burner-inserter", tx + 6, ty + 1, "N") and not w.last_refused()
        assert w.place("wooden-chest", tx + 6, ty + 2, "N") and not w.last_refused()
        belt = next(e for e in w.entities() if e.kind == "transport-belt")
        seen["belt"] = belt
        seen["lanes"] = w.belt_lanes(belt)
        assert w.rotate(belt)
        seen["after"] = next(e for e in w.entities() if e.kind == "transport-belt").facing
        assert w.rotate(belt, reverse=True)
        seen["back"] = next(e for e in w.entities() if e.kind == "transport-belt").facing
        seen["inserter"] = next(e for e in w.entities() if e.kind == "inserter")
        seen["chest"] = w.belt_lanes(next(e for e in w.entities() if e.kind == "container"))
        seen["far"] = w.place("transport-belt", tx + 8, ty, "E")
        seen["window_edge"] = w.place("transport-belt", tx - 7, ty + 7, "E")

    result = _run(build)
    assert result.error is None, result.error
    belt = seen["belt"]
    assert isinstance(belt, EntityV3)
    assert belt.facing == "E" and belt.shape == "straight" and seen["lanes"] == (0, 0)
    assert (seen["after"], seen["back"]) == ("S", "E")
    ins = seen["inserter"]
    assert ins.facing == "N"
    assert ins.pickup == (belt.x, belt.y)  # it faces its pickup: the belt north of it
    assert ins.drop == (ins.x, ins.y + 1.19921875)
    assert seen["chest"] is None
    assert seen["far"] is False and seen["window_edge"] is True
    assert any("more than 7 tiles" in line for line in result.trace)


def test_markers_by_name():
    seen = {}

    def build(w):
        seen["patch"] = w.marker("patch")
        seen["unknown"] = w.marker("output")
        seen["focus"] = w.patch()

    _run(build)
    assert seen["patch"] is not None and seen["unknown"] is None
    assert seen["patch"] == tuple(float(v) for v in seen["focus"])


def test_the_sandbox_admits_the_v3_api():
    source = """def build(world):
    b = world.entities()[0]
    world.rotate(b)
    world.belt_lanes(b)
    world.marker("iron")
    return (b.lanes, b.shape, b.held, b.pickup, b.drop, b.item)
"""
    sandbox.check(source)


# ------------------------------------------------------------------ prompts


def test_construct_prompts_are_unchanged_by_the_task_argument():
    ref = evaluate.api_reference()
    assert evaluate.api_reference("construct_smelting_line") == ref
    assert mutate.system_prompt(ref) == mutate.system_prompt(ref, "construct_smelting_line")
    assert mutate.TASK in mutate.system_prompt(ref)
    assert "world.rotate" not in ref and "world.marker" not in ref
    assert mutate.hints("construct_smelting_line") is mutate.HINTS


def test_belt_smelting_prompt():
    ref = evaluate.api_reference("belt_smelting")
    for name in ("rotate", "marker", "belt_lanes", "EntityV3", "lanes", "held", "pickup"):
        assert name in ref
    text = mutate.system_prompt(ref, "belt_smelting")
    assert mutate.TASK_BELT_SMELTING in text and mutate.GAME_NOTES_BELT_SMELTING in text
    assert mutate.TASK not in text and mutate.GAME_NOTES not in text
    assert "within 7 tiles" in text and "far lane" in text


def test_game_notes_switch_covers_every_task(monkeypatch):
    monkeypatch.setattr(mutate, "GAME_NOTES", "")
    assert mutate.game_notes("belt_smelting") == ""
    ref = evaluate.api_reference("belt_smelting")
    assert mutate.GAME_NOTES_BELT_SMELTING not in mutate.system_prompt(ref, "belt_smelting")


def test_unknown_task_is_refused():
    with pytest.raises(ValueError):
        mutate.system_prompt("ref", "no_such_task")
    with pytest.raises(ValueError):
        evaluate.task_setup("no_such_task")


# ------------------------------------------------------------------ evaluator


def test_belt_smelting_is_gated_until_its_scenes_exist():
    setup = evaluate.task_setup("belt_smelting")
    assert setup.decision_budget == 2500
    if evaluate.scenes_ported("belt_smelting"):
        pytest.skip("belt_smelting scenes are ported")
    with pytest.raises(NotImplementedError, match="not ported"):
        evaluate.scene_sets(train_n=1, val_n=1, holdout_n=1, task="belt_smelting")


def test_evaluator_payload_names_only_a_non_default_task():
    class Pool:
        workers = 1

        def __init__(self):
            self.payloads = []

        def map(self, payloads):
            self.payloads.extend(payloads)
            return [{"results": []} for _ in payloads]

    pool = Pool()
    sets = {"train": [("open", 0, {})], "val": [], "holdout": []}
    evaluate.Evaluator(pool, sets)._map([(0, sets["train"])], ["src"])
    assert "task" not in pool.payloads[-1] and pool.payloads[-1]["decision_budget"] == 600
    evaluate.Evaluator(pool, sets, task="belt_smelting")._map([(0, sets["train"])], ["src"])
    assert pool.payloads[-1]["task"] == "belt_smelting"
    assert pool.payloads[-1]["decision_budget"] == 2500


def test_the_factorio_build_env_knows_belt_smelting_but_gates_it():
    import sys
    from pathlib import Path

    path = str(
        Path(__file__).resolve().parents[1] / "integrations" / "verifiers" / "factorio_build"
    )
    sys.path.insert(0, path)
    try:
        from factorio_build import core
    except ImportError as exc:  # the package's __init__ needs `verifiers`
        pytest.skip(f"factorio_build unavailable: {exc}")
    finally:
        sys.path.remove(path)
    assert "belt_smelting" in core.KNOWN_TASKS
    if evaluate.scenes_ported("belt_smelting"):
        assert "belt_smelting" in core.SUPPORTED_TASKS
        return
    assert core.SUPPORTED_TASKS == ("construct_smelting_line",)
    with pytest.raises(ValueError, match="not ported"):
        core.rows("belt_smelting", "train", 8, 1)
    text = core.system_prompt(True, "belt_smelting")
    assert "world.belt_lanes" in text and mutate.GAME_NOTES_BELT_SMELTING in text
    assert mutate.GAME_NOTES_BELT_SMELTING not in core.system_prompt(False, "belt_smelting")
    assert core.system_prompt(True) == mutate.system_prompt(evaluate.api_reference())


def test_mine_resource_hand_mines_a_tile():
    _, bp = scenes.sample("construct_smelting_line", "train", 0)
    tx, ty = (int(v // 1) for v in bp["character"]["position"])
    bp = dict(bp, entities=[],
              resources=[{"name": "stone", "position": [tx + 2.5, ty + 0.5], "amount": 9}],
              character={**bp["character"], "position": [tx + 0.5, ty + 0.5]})  # fmt: skip
    seen = {}

    def build(w):
        x, y = w.tile()
        seen["far"] = w.mine_resource(x + 5, y, 1)
        seen["bad"] = w.mine_resource(x + 2, y, 3)
        assert w.mine_resource(x + 2, y, 5)
        for _ in range(22):
            w.wait()
        seen["stone"] = w.inventory()["stone"]

    env = RlEnv()
    env.reset("construct_smelting_line", bp, action_space="v3")
    result = play(build, SimBackend(env), decision_budget=40, world_cls=WorldV3,
                  markers=("patch",))  # fmt: skip
    assert result.error is None, result.error
    assert seen == {"far": False, "bad": False, "stone": 5}
