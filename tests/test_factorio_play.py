"""factorio-play's tool layer without `verifiers`: sessions, the belt_smelting tools, prompts.

These run everywhere, Windows included (`verifiers.v1` does not import there;
the taskset and the MCP servers are tested in test_verifiers_integration.py).
The belt_smelting episode is played by `fsim.belt_expert`'s decisions, sent
through the same tool bodies the MCP server calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLAY = str(ROOT / "integrations" / "verifiers" / "factorio_play")
if PLAY not in sys.path:
    sys.path.insert(0, PLAY)

from factorio_play import belt_tools, prompts  # noqa: E402
from factorio_play import scenes as play_scenes  # noqa: E402
from factorio_play.belt_tools import BeltTools  # noqa: E402
from factorio_play.session import WorldSession  # noqa: E402

from evolve import evaluate, mutate  # noqa: E402
from fsim import belt_expert, ffi, lib, scenes  # noqa: E402
from fsim.obsview import FACINGS, ITEMS_V3  # noqa: E402
from fsim.program_api import run_episode  # noqa: E402
from fsim.rl import RlEnv  # noqa: E402

BELT = "belt_smelting"
TASK = "construct_smelting_line"
GOLDEN = ROOT / "tests" / "golden" / "factorio_play_construct_smelting_line.json"


# ------------------------------------------------------- the expert, as tool calls

STRIDE = {0: "long", 4: "step", 8: "nudge"}
AMOUNT = {1: 1, 2: 5, 3: 20}
SIDE = 15  # the v3 placement window, 2 * 7 + 1


def expert_decisions(blueprint: dict) -> tuple[list[tuple], dict]:
    """`fsim.belt_expert`'s build of a scene, as `WorldV3` calls, and its outcome.

    The builder drives its own environment; each decision is read back as the
    world call that makes it, against the state just before it is stepped."""
    env = RlEnv()
    env.reset(BELT, blueprint, action_space="v3")
    builder = belt_expert.BeltBuilder(env.rl, blueprint)
    vector = ffi.new("int32_t[6]")
    calls = []
    while (v := builder.next_vector()) is not None:
        op = v[0]
        hx, hy = builder.here()
        if op < 12:
            calls.append(("move", FACINGS[op % 4], STRIDE[op - op % 4]))
        elif op == belt_expert.OP_PLACE:
            slot = v[2] - 1
            x, y = hx + slot // SIDE - 7, hy + slot % SIDE - 7
            calls.append(("place", ITEMS_V3[v[4] - 1], x, y, FACINGS[v[3] - 1]))
        elif op == belt_expert.OP_GIVE:
            calls.append(("give", v[1] - 1, ITEMS_V3[v[4] - 1], AMOUNT[v[5]]))
        elif op == belt_expert.OP_MINE_TILE:
            slot = v[2] - 1
            x, y = hx + slot // SIDE - 7, hy + slot % SIDE - 7
            calls.append(("mine_resource", x, y, AMOUNT[v[5]]))
        elif op == belt_expert.OP_WAIT:
            calls.append(("wait",))
        elif op == belt_expert.OP_FINISH:
            calls.append(("finish",))
        else:
            raise AssertionError(f"unexpected op {op}")
        for i in range(6):
            vector[i] = v[i]
        lib.fsim_rl_step(env.rl, vector)
        if env.rl.done:
            break
    outcome = {
        "success": bool(env.rl.success),
        "verified_output": int(env.rl.verified_output),
        "decisions": int(env.rl.steps),
        "stuck": builder.stuck,
    }
    return calls, outcome


def as_tool_calls(decisions: list[tuple]) -> list[tuple]:
    """Runs of decisions folded into the multi-decision tools: k equal moves are one
    `move(count=k)`, a hand-mining request and the waits for its items one
    `mine_resource`, back-to-back requests on one tile one call, and back-to-back
    gives of one item to one row one `give` (both split back 20, 5, 1 as the
    expert split them)."""
    out: list[list] = []
    for d in decisions:
        last = out[-1] if out else None
        name = d[0]
        if name == "move" and last and last[0] == "move" and tuple(last[1:3]) == d[1:]:
            last[3] += 1
        elif name == "move":
            out.append(["move", d[1], d[2], 1])
        elif name == "wait" and last and last[0] in ("mine_resource", "wait"):
            if last[0] == "wait":
                last[1] += 1
        elif name == "wait":
            out.append(["wait", 1])
        elif (
            name == "mine_resource"
            and last
            and last[0] == "mine_resource"
            and tuple(last[1:3]) == d[1:3]
        ):
            last[3] += d[3]
        elif name == "give" and last and last[0] == "give" and tuple(last[1:3]) == d[1:3]:
            last[3] += d[3]
        else:
            out.append(list(d))
    for call in out:
        if call[0] in ("give", "mine_resource"):
            assert belt_tools.split(call[3]) == sorted(belt_tools.split(call[3]), reverse=True)
    return [tuple(c) for c in out]


def play(tools: BeltTools, calls: list[tuple]) -> list[tuple[str, dict, str]]:
    """Send each call through the tools; the transcript is (tool, arguments, reply)."""
    params = {
        "move": ("direction", "stride", "count"),
        "place": ("item", "x", "y", "facing"),
        "give": ("row", "item", "amount"),
        "mine_resource": ("x", "y", "amount"),
        "wait": ("count",),
        "finish": (),
    }
    transcript = []
    for name, *args in calls:
        kwargs = dict(zip(params[name], args, strict=True))
        reply = getattr(tools, name)(**kwargs)
        transcript.append((name, kwargs, reply))
    return transcript


@pytest.fixture(scope="module")
def expert_episode():
    _, blueprint = scenes.sample(BELT, "train", 1)
    decisions, outcome = expert_decisions(blueprint)
    assert outcome["success"] and outcome["stuck"] is None
    session = WorldSession(blueprint, task=BELT, decision_budget=2500)
    calls = as_tool_calls(decisions)
    transcript = play(BeltTools(session), calls)
    return decisions, outcome, calls, transcript, session


def test_the_expert_solves_belt_smelting_through_the_tools(expert_episode):
    decisions, outcome, calls, transcript, session = expert_episode
    name, _, reply = transcript[-1]
    assert name == "finish"
    final = json.loads(reply)
    assert final == {
        "ok": True,
        "success": True,
        "verified_output": outcome["verified_output"],
        "score": 1,
    }
    r = session.result
    # Exactly the expert's episode: the same decisions, spent the same way.
    assert r.decisions == outcome["decisions"] == len(decisions)
    assert r.refusals == 0 and r.failures == 0
    assert r.verified_output >= belt_expert.TARGET_PLATES
    # Every action went through, and the folded tools spent what they folded.
    for tool, kwargs, text in transcript[:-1]:
        reply = json.loads(text)
        assert reply["ok"] is True, (tool, kwargs, reply)
        if tool == "mine_resource":
            assert reply["got"] == kwargs["amount"] and reply["item"] == "coal"
    assert len(calls) < len(decisions) / 2


def test_a_tool_call_costs_what_worldv3_counts():
    _, blueprint = scenes.sample(BELT, "train", 0)
    session = WorldSession(blueprint, task=BELT, decision_budget=2500)
    tools = BeltTools(session)
    left = json.loads(tools.decisions_left())
    assert left == 2500 == evaluate.task_setup(BELT).decision_budget
    moved = json.loads(tools.move("east", "step", 3))
    assert moved["ok"] and moved["decisions"] == 3 and moved["left"] == 2497
    waited = json.loads(tools.wait(4))
    assert waited == {"ok": True, "decisions": 4, "left": 2493}
    # A refusal costs nothing, and says why.
    far = json.loads(tools.place("transport-belt", 999, 999, "N"))
    assert far["ok"] is False and "tiles from tile()" in far["refused"]
    assert far["left"] == 2493
    # Arguments the tool itself cannot take never reach the world.
    assert "error" in json.loads(tools.give(0, "coal", 0))
    assert "error" in json.loads(tools.mine_resource(0, 0, 3, wait=False))
    counts = session.counters()
    assert counts == {"decisions": 7, "refusals": 1, "failures": 0}
    session.finish()


def test_a_tool_that_runs_out_of_budget_ends_and_scores_the_episode():
    _, blueprint = scenes.sample(BELT, "train", 0)
    session = WorldSession(blueprint, task=BELT, decision_budget=5)
    tools = BeltTools(session)
    out = json.loads(tools.wait(10))
    assert out["ok"] is False and "BudgetExhausted" in out["error"]
    assert session.finished and session.result.success is False
    assert session.result.decisions == 5
    assert "over" in json.loads(tools.me())["error"]


def test_belt_queries_are_terse():
    _, blueprint = scenes.sample(BELT, "train", 0)
    session = WorldSession(blueprint, task=BELT)
    tools = BeltTools(session)
    inventory = json.loads(tools.inventory())
    assert inventory == {
        "coal": 20,
        "transport-belt": 40,
        "burner-mining-drill": 4,
        "stone-furnace": 4,
        "burner-inserter": 10,
    }
    for name in ("iron", "coal", "output"):
        x, y = json.loads(tools.marker(name))
        assert (x, y) == tuple(blueprint["markers"][name])
    assert json.loads(tools.marker("nowhere")) is None
    rows = json.loads(tools.entities())
    assert rows and all(set(r) >= {"row", "kind", "x", "y"} for r in rows)
    # Fields that hold nothing are left out.
    assert all(v not in (None, False, 0) for r in rows for k, v in r.items() if k in EMPTY)
    chests = json.loads(tools.entities("container"))
    assert [c["kind"] for c in chests] == ["container"] * len(chests)
    assert " " not in tools.me() and " " not in tools.entities()
    session.finish()


EMPTY = ("facing", "fuel", "contents", "output", "working", "remembered", "held", "item")


def test_ore_tiles_come_as_runs():
    assert belt_tools.runs([(1, 0), (2, 0), (3, 0), (5, 0), (2, 1)]) == [
        [0, 1, 3],
        [0, 5, 5],
        [1, 2, 2],
    ]
    assert belt_tools.split(23) == [20, 1, 1, 1] and belt_tools.split(40) == [20, 20]


def test_belt_score_is_factory_sims():
    assert belt_tools.score(0) == 0.0
    assert belt_tools.score(75) == 0.5
    assert belt_tools.score(150) == belt_tools.score(197) == 1.0


def test_a_v2_session_offers_no_v3_method():
    session = WorldSession(scenes.sample(TASK, "train", 0)[1], task=TASK)
    assert session.v3 is False
    out = session.call("rotate", 0)
    assert out == {"ok": False, "error": "unknown method 'rotate'"}
    assert session.counters()["decisions"] == 0
    session.finish()


# ------------------------------------------------------------------ prompts, scenes


def test_construct_smelting_line_prompts_are_unchanged():
    """0.1.1's prompts, as its own code produced them (tests/golden)."""
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert prompts.system_prompt(True) == golden["system_prompt"]
    assert prompts.system_prompt(False) == golden["system_prompt_no_notes"]
    assert prompts.system_prompt(True, TASK) == golden["system_prompt"]
    assert prompts.user_prompt(TASK) == golden["prompt"]


def test_belt_prompt_reuses_factory_sims_task_text_and_notes():
    text = prompts.system_prompt(True, BELT)
    assert mutate.TASK_BELT_SMELTING in text
    assert mutate.GAME_NOTES_BELT_SMELTING in text
    assert mutate.GAME_NOTES not in text and mutate.TASK not in text
    without = prompts.system_prompt(False, BELT)
    assert mutate.GAME_NOTES_BELT_SMELTING not in without and "Game notes:" not in without
    with pytest.raises(ValueError):
        prompts.system_prompt(True, "plate_line")


def test_belt_scene_blocks():
    assert play_scenes.SUPPORTED_TASKS == (TASK, BELT)
    train = play_scenes.scene_block(BELT, "train", 0, 6)
    assert {r.family for r in train} <= set(evaluate.task_setup(BELT).families_train)
    holdout = play_scenes.scene_block(BELT, "holdout", 0, 6)
    assert {r.family for r in holdout} <= set(evaluate.task_setup(BELT).families_holdout)
    assert all(r.sample_split == "test" for r in holdout)
    val = play_scenes.scene_block(BELT, "val", 0, 3)
    assert [r.seed for r in val] == [evaluate.VAL_OFFSET + k for k in range(3)]
    with pytest.raises(ValueError):
        play_scenes.scene_block("plate_line", "train", 0, 1)


def test_expert_decisions_replay_the_expert_exactly():
    """The adapter reads each vector back as the world call that makes it."""
    _, blueprint = scenes.sample(BELT, "train", 2)
    decisions, outcome = expert_decisions(blueprint)
    it = iter(decisions)

    def build(world):
        for name, *args in it:
            if name == "finish":
                world.finish()
                return
            getattr(world, name)(*args)

    r = run_episode(build, blueprint, task=BELT, decision_budget=2500)
    assert (r.success, r.verified_output, r.decisions, r.refusals) == (
        outcome["success"],
        outcome["verified_output"],
        outcome["decisions"],
        0,
    )


def test_refusal_notes_survive_past_the_trace_cap():
    # World._trace is a capped deque, so its length stops growing after its 20th
    # intent; the reply's `refused` note must still appear after that.
    _, blueprint = scenes.sample(TASK, "train", 0)
    session = WorldSession(blueprint, task=TASK)
    for _ in range(25):
        assert session.call("wait")["ok"]
    far = session.call("place", "stone-furnace", 999, 999, "N")
    assert far["ok"] and far["result"] is False and far["refused"]
    session.finish()
