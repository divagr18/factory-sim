"""The Backplay window climbs its ladder on evidence, not on the clock."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("torch")

from train import BACKPLAY_LADDER, GatedBackplay, backplay_window  # noqa: E402


def _episodes(n: int, success: float) -> list[dict]:
    wins = round(n * success)
    return [{"start": "back3", "success": i < wins} for i in range(n)]


def test_it_starts_at_the_first_rung():
    assert GatedBackplay().window == BACKPLAY_LADDER[0]


def test_a_solved_window_widens_and_an_unsolved_one_does_not():
    gate = GatedBackplay(threshold=0.5, minimum=64)
    assert gate.update(_episodes(128, 0.2), steps=1000) == BACKPLAY_LADDER[0]
    assert gate.update(_episodes(128, 0.9), steps=2000) == BACKPLAY_LADDER[1]
    assert gate.advanced_at == [2000]


def test_too_few_episodes_never_move_it():
    gate = GatedBackplay(threshold=0.5, minimum=64)
    assert gate.update(_episodes(8, 1.0), steps=10) == BACKPLAY_LADDER[0]
    assert gate.index == 0


def test_scene_starts_are_not_evidence_about_the_window():
    gate = GatedBackplay(threshold=0.5, minimum=8)
    scene_only = [{"start": "scene", "success": True} for _ in range(128)]
    assert gate.update(scene_only, steps=10) == BACKPLAY_LADDER[0]


def test_it_stops_at_the_top_of_the_ladder():
    gate = GatedBackplay(threshold=0.0, minimum=1)
    for step in range(len(BACKPLAY_LADDER) * 2):
        gate.update(_episodes(64, 1.0), steps=step)
    assert gate.window == BACKPLAY_LADDER[-1]
    assert gate.index == len(BACKPLAY_LADDER) - 1


def test_the_clock_schedule_walks_the_same_ladder():
    """The gated ladder is the fixed schedule's windows, in order."""
    seen = [backplay_window(p / 100) for p in range(101)]
    ordered = [w for i, w in enumerate(seen) if i == 0 or w != seen[i - 1]]
    assert ordered == list(BACKPLAY_LADDER)
