"""What an installed copy needs that a checkout gets from tests/golden."""

import json
from pathlib import Path

import fsim

ROOT = Path(__file__).resolve().parents[1]


def test_the_packaged_terrain_is_the_recorded_one():
    packaged = json.loads((ROOT / "fsim" / "data" / "terrain.json").read_text(encoding="utf-8"))
    golden = json.loads(
        (ROOT / "tests" / "golden" / "sim-mechanics-m3.json").read_text(encoding="utf-8")
    )
    assert packaged["terrain"] == golden["terrain"]


def test_water_tiles_read_the_packaged_copy_first():
    data = json.loads((ROOT / "fsim" / "data" / "terrain.json").read_text(encoding="utf-8"))
    assert fsim.water_tiles() == sorted((int(x), int(y)) for x, y in data["terrain"]["water"])
