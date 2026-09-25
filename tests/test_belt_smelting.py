"""belt_smelting: the task, its verification and the scripted builder.

The scenes themselves are pinned seed for seed against FactorioRL's generator
by `test_scenes.py` (`tests/golden/scenes.json`). Here: the frozen holdout,
the task's budget and markers, the container verification of FactorioRL's
`run_verification`, and the builder against the outcomes the reference
solver recorded on the engine (`tests/golden/belt_smelting_reference.json`).
"""

from __future__ import annotations

import hashlib
import json
import random

import pytest

from evolve import evaluate as ev
from fsim import belt_expert, ffi, lib, scenes
from fsim.parity import GOLDEN
from fsim.rl import RlEnv

TASK = "belt_smelting"
FINISH = [24, 0, 0, 0, 0, 0]
WAIT = [21, 0, 0, 0, 0, 0]
REFERENCE = json.loads((GOLDEN / "belt_smelting_reference.json").read_text(encoding="utf-8"))


def _gate_seed(index: int) -> int:
    plan = REFERENCE["seed_plan"]
    payload = f"{plan['master']}|{plan['run_id']}|{plan['branch']}|{index}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _reset(split="train", seed=0, **options):
    family, scene = scenes.sample(TASK, split, seed)
    env = RlEnv()
    env.reset(TASK, scene, **options)
    return env, family, scene


# ------------------------------------------------------------------ scenes


def test_the_holdout_is_factoriorls_frozen_one():
    sets = ev.scene_sets(train_n=0, val_n=0, holdout_n=ev.HOLDOUT_FROZEN, task=TASK)
    check = ev.verify_holdout(sets, task=TASK)
    if check["file"] is None or not check["frozen"]:
        pytest.skip("no FactorioRL checkout (with a frozen belt_smelting holdout) beside this one")
    assert check["matched"] == check["compared"] == ev.HOLDOUT_FROZEN
    assert {f for f, _, _ in sets["holdout"]} == {"obstructed", "far_chest"}


def test_the_splits_draw_their_own_families():
    assert {scenes.sample(TASK, "train", s)[0] for s in range(64)} == {
        "open",
        "walled",
        "split_patch",
    }
    assert {scenes.sample(TASK, "test", s)[0] for s in range(64)} == {"obstructed", "far_chest"}


# ------------------------------------------------------------------ task


def test_the_task_carries_belt_smeltings_budget_and_markers():
    env, _, scene = _reset()
    t = env.rl.task
    assert env.v3 and t.action_space == lib.ACTION_SPACE_V3
    assert (t.max_steps, t.decision_ticks, t.construction_tick_limit) == (2500, 30, 75000)
    # max_game_ticks = construction + the 36,000-tick window, as the spec says.
    assert t.construction_tick_limit + 36000 == 111000
    chest = env.rl.env.entities[t.output_entity]
    assert chest.alive and chest.kind == lib.K_CHEST
    assert (chest.pos.x / 256, chest.pos.y / 256) == tuple(scene["markers"]["output"])
    # Every public marker has its goal slot; goal[9..11] points at "iron".
    assert t.marker_count == 3 and list(t.marker_present)[:3] == [1, 1, 1]
    assert t.has_patch and (t.patch_x, t.patch_y) == tuple(scene["markers"]["iron"])
    held = {
        lib.IT_BURNER_DRILL: 4,
        lib.IT_STONE_FURNACE: 4,
        lib.IT_TRANSPORT_BELT: 40,
        lib.IT_BURNER_INSERTER: 10,
        lib.IT_COAL: 20,
    }
    main = env.rl.env.main
    for item, count in held.items():
        assert sum(main[i].count for i in range(80) if main[i].item == item) == count


def test_nothing_is_delivered_or_verified_at_reset():
    env, _, _ = _reset()
    assert lib.fsim_rl_delivered(env.rl) == 0
    assert not env.rl.verified and not env.rl.success
    assert env.obs3["goal"][1] == 0.0


def test_finish_runs_the_ten_minute_window_and_ends_the_episode():
    env, _, _ = _reset()
    tick = env.rl.env.tick
    _, reward, terminated, truncated, info = env.step(FINISH)
    assert terminated and not truncated
    assert env.rl.verified and env.rl.env.tick == tick + 36000
    assert reward == 0.0 and info["verified_output"] == 0.0 and not info["success"]
    assert info["reward_components"] == {"verified_output": 0.0}


def test_running_out_of_decisions_runs_the_window_too():
    env, _, _ = _reset(max_steps=3)
    for _ in range(2):
        _, _, terminated, truncated, _ = env.step(WAIT)
        assert not terminated and not truncated
    _, _, terminated, truncated, info = env.step(WAIT)
    assert terminated and not truncated and env.rl.verified
    assert env.rl.env.tick == 3 * 30 + 36000


def test_plates_put_in_the_chest_before_the_window_do_not_count():
    """The count is the chest's increase over the window, and a plate there
    already is not an increase."""
    env, _, _ = _reset()
    assert (
        lib.fsim_entity_insert(env.rl.env, env.rl.task.output_entity, lib.IT_IRON_PLATE, 50) == 50
    )
    assert lib.fsim_rl_delivered(env.rl) == 50
    env.step(FINISH)
    assert env.rl.verified_uncapped == 0.0 and env.rl.verified_output == 0.0


def _built_line(extra_coal=belt_expert.EXTRA_COAL, stop_before_finish=True, seed=0):
    env, family, scene = _reset("train", seed)
    builder = belt_expert.BeltBuilder(env.rl, scene, extra_coal=extra_coal)
    vector = ffi.new("int32_t[6]")
    while (v := builder.next_vector()) is not None:
        if v == FINISH and stop_before_finish:
            break
        for i in range(6):
            vector[i] = v[i]
        lib.fsim_rl_step(env.rl, vector)
    return env, builder


def test_a_built_line_is_scored_on_the_chest_and_capped_by_machine_ore():
    env, builder = _built_line()
    assert builder.stuck is None
    before = lib.fsim_rl_delivered(env.rl)
    _, reward, terminated, _, info = env.step(FINISH)
    rl = env.rl
    delivered = lib.fsim_rl_delivered(rl) - before
    assert terminated and rl.verified
    assert rl.verified_uncapped == delivered
    assert rl.verified_output == min(rl.verified_uncapped, rl.verified_source)
    assert rl.verified_output >= 150 and info["success"] and rl.success
    assert reward == pytest.approx(min(1.0, rl.verified_output / 150))
    assert env.obs3["goal"][1] == 1.0


def test_plates_from_hand_fed_ore_are_capped_away():
    """`source="iron-ore"`: a line whose drills mine nothing in the window
    delivers nothing, however many plates reach the chest."""
    env, _ = _built_line()
    rl = env.rl
    # Take the drills' fuel out of the world: no ore is machine-mined in the
    # window, while the furnaces smelt ore put into them by script.
    for i in range(rl.env.entity_count):
        e = rl.env.entities[i]
        if e.alive and e.kind == lib.K_DRILL:
            e.fuel.count = 0
            e.energy = 0.0
            e.remaining = 0.0
            e.burning = lib.IT_NONE
            e.held = lib.IT_NONE
        if e.alive and e.kind == lib.K_FURNACE:
            lib.fsim_entity_insert(rl.env, i, lib.IT_IRON_ORE, 20)
    env.step(FINISH)
    assert rl.verified_source == 0.0
    assert rl.verified_uncapped > 0 and rl.verified_output == 0.0


def test_a_chest_that_was_picked_up_holds_nothing():
    env, _, _ = _reset()
    index = env.rl.task.output_entity
    lib.fsim_entity_insert(env.rl.env, index, lib.IT_IRON_PLATE, 5)
    env.rl.env.entities[index].alive = 0
    assert lib.fsim_rl_delivered(env.rl) == 0


# ------------------------------------------------------------------ builder


def _reference_rows():
    """A spread of the engine episodes: every coal supply, every family."""
    rows = REFERENCE["episodes"]
    picked, seen = [], set()
    for row in rows:
        key = (row["gate"], row["family"])
        if key in seen:
            continue
        seen.add(key)
        picked.append(row)
    return picked


@pytest.mark.parametrize(
    "row", _reference_rows(), ids=lambda r: f"{r['gate']}-{r['split']}-{r['index']}-{r['family']}"
)
def test_the_builder_delivers_what_the_reference_delivered_on_the_engine(row):
    """Same scene, same coal: the same fuel split, the same uncapped delivery,
    and the capped count within one plate.

    Walking differs (see `belt_expert`), so the window opens a few decisions
    earlier or later than on the engine, and one ore more or less is mined in
    it. Over all 303 recorded episodes the simulator matched the plate count
    exactly in 217 and was one above it in 86, and matched `uncapped` in all.
    """
    family, scene = scenes.sample(TASK, row["split"], _gate_seed(row["index"]))
    assert family == row["family"]
    got = belt_expert.run(RlEnv(), scene, extra_coal=row["extra_coal"])
    assert got["stuck"] is None
    assert got["fuel_plan"].__dict__ == row["fuel_plan"]
    assert got["uncapped"] == row["uncapped"]
    assert 0 <= got["plates"] - row["plates"] <= 1
    assert 0 <= got["ore_in_window"] - row["ore_in_window"] <= 1


@pytest.mark.parametrize("split,seed", [("train", 1), ("train", 5), ("test", 0), ("test", 7)])
def test_the_builder_emits_only_legal_vectors_and_succeeds(split, seed):
    env, family, scene = _reset(split, seed)
    builder = belt_expert.BeltBuilder(env.rl, scene)
    while (v := builder.next_vector()) is not None:
        op_masks = env.op_masks()
        assert env.mask[v[0]], (family, v)
        # Every argument the operation reads is one its own mask allows.
        offsets = (0, 97, 97 + 226, 97 + 226 + 5, 97 + 226 + 5 + 19)
        for dim, value in enumerate(v[1:]):
            assert op_masks[v[0], offsets[dim] + value], (family, v, dim)
        env.step(v)
    assert builder.stuck is None
    assert env.rl.success and env.rl.decode_failures == 0


def test_a_drawn_layout_is_one_of_the_shortest():
    env, _, scene = _reset("train", 2)
    s = belt_expert.scene_of(scene)
    water = belt_expert.water_of(env.rl.env)
    found = belt_expert.plan_candidates(s, water)
    reference = belt_expert.plan_line(s, water)
    assert found and found[0][-1] == reference
    assert len({f[0] for f in found}) == 1
    drawn = belt_expert.BeltBuilder(env.rl, scene, rng=random.Random(3)).plan
    assert len(drawn.belts) == len(reference.belts)
