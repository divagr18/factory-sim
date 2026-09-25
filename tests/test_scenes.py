"""The scene generators reproduce FactorioRL's, seed for seed."""

from __future__ import annotations

import json
import random

import pytest

from fsim import lib, scenes
from fsim.parity import GOLDEN
from fsim.rl import RlEnv

GOLDEN_SCENES = json.loads((GOLDEN / "scenes.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("key", sorted(GOLDEN_SCENES))
def test_generator_matches_factoriorl(key):
    task, family, seed = key.split("|")
    scene = scenes.GENERATORS[task](family, random.Random(int(seed)))
    assert scene == GOLDEN_SCENES[key]


def test_every_family_is_covered():
    covered = {tuple(k.split("|")[:2]) for k in GOLDEN_SCENES}
    declared = {(t, f) for t, fams in scenes.FAMILIES.items() for f in fams}
    assert covered == declared


def test_sample_draws_only_the_requested_split():
    seen = {scenes.sample("construct_smelting_line", "train", s)[0] for s in range(64)}
    assert seen == {"open_patch", "offset_patch", "varied_patch", "cluttered_patch"}
    assert {scenes.sample("construct_smelting_line", "test", s)[0] for s in range(8)} == {
        "obstructed_patch"
    }


def test_start_curriculum_moves_the_start_beside_the_patch():
    _, scene = scenes.sample("construct_smelting_line", "train", 3, start_curriculum=1.0)
    px, py = scene["markers"]["patch"]
    x, y = scene["character"]["position"]
    assert abs(x - px) + abs(y - py) <= 3.0


@pytest.mark.parametrize("task", sorted(scenes.FAMILIES))
def test_every_generated_scene_installs(task):
    env = RlEnv()
    for split in ("train", "val", "test"):
        for seed in range(4):
            if not scenes.families(task, split):
                continue
            _, scene = scenes.sample(task, split, seed)
            obs = env.reset(task, scene)
            assert env.rl.env.resource_count == len(scene["resources"])
            if task == "belt_smelting":
                # Its three sites are 20 to 40 tiles apart, so the iron patch
                # can be outside the 32-tile sensor at the start; the chest the
                # verification counts is installed wherever the character is.
                chest = env.rl.env.entities[env.rl.task.output_entity]
                assert chest.alive and chest.kind == lib.K_CHEST
                continue
            assert obs["grid"][0].sum() > 0  # the ore patch is visible


def test_a_missing_water_file_is_an_error_not_a_dry_map(tmp_path, monkeypatch):
    """A packaged simulator without the water file used to run every scene on a
    map with no water -- a different map from the real game's -- in silence."""
    import pytest

    import fsim

    monkeypatch.setattr(fsim, "ROOT", tmp_path / "nowhere")
    monkeypatch.setattr(fsim, "PACKAGE_DATA", tmp_path / "nowhere" / "data")
    with pytest.raises(FileNotFoundError, match="water"):
        fsim.water_tiles()
