"""Scene generators for the two tasks, draw for draw with FactorioRL's.

Each generator consumes a `random.Random` in the same order as its FactorioRL
counterpart (`factoriorl.tasks.families.construct_smelting_line.generate` and
`build_line.generate`) and returns the blueprint payload `Blueprint.to_dict`
would, so one seed names one scene in both projects. `tests/golden/scenes.json`
holds FactorioRL's own output for a range of seeds and `test_scenes.py`
compares against it; `tools/sync_golden.py` regenerates it.

`sample` adds what `FactorioEnv.prepare_scene` does around a generator: pick a
family of the requested split, then (optionally) the exploring-starts move of
`_apply_start_curriculum` (Sutton & Barto 2nd ed. section 5.3).
"""

from __future__ import annotations

import math
import random

INVENTORY = {"burner-mining-drill": 2, "stone-furnace": 2, "coal": 60}
UNLOCK = ["burner-mining-drill", "stone-furnace"]

FAMILIES = {
    "construct_smelting_line": {
        "open_patch": "train",
        "offset_patch": "train",
        "obstructed_patch": "test",
        "varied_patch": "train",
        "cluttered_patch": "train",
    },
    "build_line": {
        "square_patch": "train",
        "offset_patch": "val",
        "narrow_patch": "test",
    },
}


def _resource(x, y) -> dict:
    return {"name": "iron-ore", "position": [float(x), float(y)], "amount": 10000}


def _wall(x, y) -> dict:
    return {"name": "stone-wall", "position": [x, y], "direction": "north", "force": "neutral"}


def _payload(entities, tiles, start, marker) -> dict:
    return {
        "public_markers": ["patch"],
        "entities": entities,
        "resources": [_resource(x, y) for x, y in tiles],
        "character": {"position": list(start), "inventory": dict(INVENTORY)},
        "markers": {"patch": list(marker)},
        "unlock_recipes": list(UNLOCK),
        "radius": 48,
    }


def construct_smelting_line(family: str, rng: random.Random) -> dict:
    ox, oy = 0, 0
    if family == "offset_patch":
        ox, oy = rng.choice((-9, 9)), rng.choice((-9, 9))
    if family == "obstructed_patch":
        tiles = [(ox + x, oy + y) for x in range(-1, 2) for y in range(-5, 6)]
    elif family == "varied_patch":
        half_w, half_h = rng.randint(1, 4), rng.randint(1, 4)
        ox, oy = rng.randint(-12, 12), rng.randint(-12, 12)
        tiles = [
            (ox + x, oy + y)
            for x in range(-half_w, half_w + 1)
            for y in range(-half_h, half_h + 1)
        ]
    else:
        if family == "cluttered_patch":
            ox, oy = rng.randint(-9, 9), rng.randint(-9, 9)
        tiles = [(ox + x, oy + y) for x in range(-3, 4) for y in range(-3, 4)]
    cx = sum(x for x, _ in tiles) / len(tiles)
    cy = sum(y for _, y in tiles) / len(tiles)
    angle = rng.uniform(0, 2 * math.pi)
    start = (
        round(cx + math.cos(angle) * rng.uniform(9, 13), 1),
        round(cy + math.sin(angle) * rng.uniform(9, 13), 1),
    )
    entities = []
    if family == "obstructed_patch":
        for y in range(int(cy) - 2, int(cy) + 3):
            entities.append(_wall(cx + 5, float(y)))
    if family == "cluttered_patch":
        seen: set[tuple[float, float]] = set()
        for _ in range(rng.randint(1, 3)):
            vertical = rng.random() < 0.5
            away = rng.choice((-6, -5, -4, 4, 5, 6))
            along = rng.randint(-6, 3)
            length = rng.randint(2, 4)
            for k in range(length):
                x, y = (cx + away, cy + along + k) if vertical else (cx + along + k, cy + away)
                if (x, y) in seen:
                    continue
                seen.add((x, y))
                entities.append(_wall(float(x), float(y)))
    return _payload(entities, tiles, start, (cx, cy))


def build_line(family: str, rng: random.Random) -> dict:
    if family == "narrow_patch":
        vertical = rng.random() < 0.5
        xs = range(-1, 1) if vertical else range(-5, 5)
        ys = range(-5, 5) if vertical else range(-1, 1)
        tiles = [(0.0 + dx, 0.0 + dy) for dx in xs for dy in ys]
    elif family == "offset_patch":
        ox = rng.choice([-8, 8])
        oy = rng.choice([-8, 8])
        tiles = [(0.0 + ox + dx, 0.0 + oy + dy) for dx in range(-3, 4) for dy in range(-3, 4)]
    else:
        tiles = [(0.0 + dx, 0.0 + dy) for dx in range(-3, 4) for dy in range(-3, 4)]
    cx = sum(x for x, _ in tiles) / len(tiles)
    cy = sum(y for _, y in tiles) / len(tiles)
    angle = rng.uniform(0, 2 * math.pi)
    distance = rng.uniform(8.0, 13.0)
    start = (
        round(cx + math.cos(angle) * distance, 1),
        round(cy + math.sin(angle) * distance, 1),
    )
    entities = []
    if family == "narrow_patch":
        side = 1 if math.cos(angle) >= 0 else -1
        for offset in range(-1, 2):
            entities.append(_wall(float(int(cx) + side * 6), float(int(cy) + offset)))
    return _payload(entities, tiles, start, (round(cx, 1), round(cy, 1)))


GENERATORS = {"construct_smelting_line": construct_smelting_line, "build_line": build_line}


def families(task: str, split: str) -> list[str]:
    """A split's families, in declaration order (the order FactorioRL indexes)."""
    return [name for name, s in FAMILIES[task].items() if s == split]


def _start_curriculum(scene: dict, rng: random.Random, fraction: float) -> dict:
    """`FactorioEnv._apply_start_curriculum`: sometimes start beside the patch."""
    if rng.random() >= fraction:
        return scene
    target = scene["markers"]["patch"]
    occupied = {
        (math.floor(e["position"][0]), math.floor(e["position"][1])) for e in scene["entities"]
    }
    for dx, dy in ((0.0, 2.0), (0.0, -2.0), (2.0, 0.0), (-2.0, 0.0), (0.0, 3.0), (0.0, -3.0)):
        candidate = (target[0] + dx, target[1] + dy)
        if (math.floor(candidate[0]), math.floor(candidate[1])) not in occupied:
            scene["character"]["position"] = list(candidate)
            return scene
    return scene


def sample(task: str, split: str, seed: int, start_curriculum: float = 0.0) -> tuple[str, dict]:
    """(family, blueprint) for one episode, as `FactorioEnv.prepare_scene` draws it."""
    rng = random.Random(seed)
    names = families(task, split)
    family = names[rng.randrange(len(names))]
    scene = GENERATORS[task](family, rng)
    if start_curriculum:
        scene = _start_curriculum(scene, rng, start_curriculum)
    return family, scene
