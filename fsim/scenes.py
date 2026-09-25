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
        "varied_patch": "train",
        "cluttered_patch": "train",
    },
    "plate_line": {
        "commissioning": "train",
        "commissioning_crowded": "train",
        "commissioning_far": "val",
        "commissioning_walled": "test",
    },
    "belt_smelting": {
        "open": "train",
        "walled": "train",
        "split_patch": "train",
        "obstructed": "test",
        "far_chest": "test",
    },
}

#: plate_line's machines are pinned, not drawn: the scene is a commissioning
#: scenario, and a drill whose drop tile moved between runs would make two runs
#: incomparable (FactorioRL's plate_line 1.2.0).
DRILL_POSITION = (1.0, 1.0)
FURNACE_POSITION = (1.0, 3.0)
ORE_RADIUS = 3
STARTING_COAL = 120
#: `commissioning_crowded` (plate_line 1.3.0): four decoy machines and two
#: decoy items, so `give(target, item, amount)` is 6 x 3 x 4 = 72 combinations
#: against 8. Coal drops to 50 -- with 120 the agent can fuel everything and
#: the choice of target stops mattering. The decoys cannot produce a plate
#: however they are fuelled, so only the size of the decision changes.
DECOY_MACHINES = (
    ("burner-mining-drill", (-5.0, -4.0), "north"),
    ("burner-mining-drill", (6.0, 5.0), "east"),
    ("stone-furnace", (-6.0, 4.0), "north"),
    ("stone-furnace", (7.0, -3.0), "north"),
)
CROWDED_COAL = 50
CROWDED_DECOY_ITEMS = {"iron-ore": 20, "stone": 20}
#: Tiles the start is pushed out of per step, and how many steps to try.
NUDGE_STEP = 1.0
NUDGE_ATTEMPTS = 8


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
    elif family == "varied_patch":
        half_w, half_h = rng.randint(1, 4), rng.randint(1, 4)
        ox, oy = rng.randint(-12, 12), rng.randint(-12, 12)
        tiles = [
            (0.0 + ox + dx, 0.0 + oy + dy)
            for dx in range(-half_w, half_w + 1)
            for dy in range(-half_h, half_h + 1)
        ]
    else:
        ox = oy = 0
        if family == "cluttered_patch":
            ox, oy = rng.randint(-9, 9), rng.randint(-9, 9)
        tiles = [(0.0 + ox + dx, 0.0 + oy + dy) for dx in range(-3, 4) for dy in range(-3, 4)]
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
    if family == "cluttered_patch":
        seen: set[tuple[float, float]] = set()
        for _ in range(rng.randint(1, 3)):
            vertical = rng.random() < 0.5
            away = rng.choice((-6, -5, -4, 4, 5, 6))
            along = rng.randint(-6, 3)
            length = rng.randint(2, 4)
            for step in range(length):
                if vertical:
                    x, y = cx + away, cy + along + step
                else:
                    x, y = cx + along + step, cy + away
                if (x, y) in seen:
                    continue
                seen.add((x, y))
                entities.append(_wall(float(x), float(y)))
    return _payload(entities, tiles, start, (round(cx, 1), round(cy, 1)))


def _machine(name: str, position, direction: str, marker: str | None) -> dict:
    # An unmarked entity has no `marker` key at all. `EntitySpec.to_dict`
    # omits it rather than writing null, and a null would not compare equal.
    out = {
        "name": name,
        "position": [position[0], position[1]],
        "direction": direction,
        "force": "player",
    }
    if marker is not None:
        out["marker"] = marker
    return out


def _clear_of(start, angle: float, blocked: set[tuple[int, int]]):
    """Push `start` outward along its own bearing until its tile is free.

    Consumes no randomness and is a no-op when the start is already clear, so
    only the scenes that would have started inside the screen differ at all --
    which is what keeps the other blueprint digests identical.
    """
    position = start
    for _ in range(NUDGE_ATTEMPTS):
        if (math.floor(position[0]), math.floor(position[1])) not in blocked:
            return position
        position = (
            round(position[0] + math.cos(angle) * NUDGE_STEP, 1),
            round(position[1] + math.sin(angle) * NUDGE_STEP, 1),
        )
    return position


def plate_line(family: str, rng: random.Random) -> dict:
    """Commissioning, not construction: both machines are placed and aligned
    and both are empty, and the agent has to reach each one and fuel it."""
    angle = rng.uniform(0, 2 * math.pi)
    distance = rng.uniform(12.0, 16.0) if family == "commissioning_far" else rng.uniform(5.0, 9.0)
    start = (
        round(DRILL_POSITION[0] + math.cos(angle) * distance, 1),
        round(DRILL_POSITION[1] + math.sin(angle) * distance, 1),
    )
    resources = [
        (DRILL_POSITION[0] + dx, DRILL_POSITION[1] + dy)
        for dx in range(-ORE_RADIUS, ORE_RADIUS + 1)
        for dy in range(-ORE_RADIUS, ORE_RADIUS + 1)
    ]
    entities = [
        _machine("burner-mining-drill", DRILL_POSITION, "south", "drill"),
        _machine("stone-furnace", FURNACE_POSITION, "north", "furnace"),
    ]
    inventory = {"coal": STARTING_COAL}
    if family == "commissioning_crowded":
        # The start annulus overlaps where the decoys sit, so the start is
        # pushed clear first -- consuming no randomness -- and then the decoys
        # and the decoy items are added.
        occupied = {
            (int(position[0]) + dx, int(position[1]) + dy)
            for _name, position, _direction in DECOY_MACHINES
            for dx in (-1, 0)
            for dy in (-1, 0)
        }
        start = _clear_of(start, angle, occupied)
        for name, position, direction in DECOY_MACHINES:
            entities.append(_machine(name, position, direction, None))
        inventory = {"coal": CROWDED_COAL, **CROWDED_DECOY_ITEMS}
    if family == "commissioning_walled":
        # The structural holdout: a screen to walk around, so the test split
        # differs in layout and not only in where the character starts.
        side = 1 if math.cos(angle) >= 0 else -1
        blocked = {
            (int(DRILL_POSITION[0]) + side * 5, int(DRILL_POSITION[1]) + offset)
            for offset in range(-1, 2)
        }
        for x, y in sorted(blocked, key=lambda t: t[1]):
            entities.append(_wall(float(x), float(y)))
        # The start annulus crosses the screen, so the two draws can collide.
        start = _clear_of(start, angle, blocked)
    return {
        "public_markers": [],
        "entities": entities,
        "resources": [dict(_resource(x, y), amount=5000) for x, y in resources],
        "character": {"position": list(start), "inventory": inventory},
        "markers": {"line": list(DRILL_POSITION)},
        # Nothing to unlock: the machines are already down, and the task never
        # asks for one to be crafted.
        "unlock_recipes": [],
        "radius": 48,
    }


# ------------------------------------------------------------ belt_smelting
#
# FactorioRL `tasks/families/belt_smelting.py` 1.1.0: an iron patch, a coal
# patch and a marked output chest, each at least 21 tiles (edge to edge) from
# the other two, with walls in some families. The draw order is that module's
# docstring's: only `rng.randint`, fixed-count loops and rejection loops that
# redraw the same values in the same order, and every acceptance test is
# integer arithmetic on inclusive tile rectangles (x0, y0, x1, y1).

BELT_INVENTORY = {
    "burner-mining-drill": 4,
    "stone-furnace": 4,
    "transport-belt": 40,
    "burner-inserter": 10,
    "coal": 20,
}
BELT_UNLOCK = ["burner-mining-drill", "stone-furnace", "transport-belt", "burner-inserter"]
BELT_MARKERS = ["iron", "coal", "output"]
BELT_SCENE_HALF = 60
BELT_MIN_GAP_SQ = 21 * 21
BELT_MIN_CENTRE_SQ4 = (2 * 20) ** 2
BELT_MAX_CENTRE_SQ4 = (2 * 40) ** 2
BELT_CHEST_MANHATTAN = {
    "open": (21, 34),
    "walled": (21, 30),
    "split_patch": (21, 34),
    "obstructed": (21, 30),
    "far_chest": (35, 38),
}
BELT_CLEARANCE = 6
BELT_CHEST_CLEARANCE = 4
BELT_CORRIDOR_MARGIN = 3


def _gaps(a, b) -> tuple[int, int]:
    return (
        max(0, b[0] - a[2] - 1, a[0] - b[2] - 1),
        max(0, b[1] - a[3] - 1, a[1] - b[3] - 1),
    )


def _centre_dist_sq4(a, b) -> int:
    dx = (b[0] + b[2]) - (a[0] + a[2])
    dy = (b[1] + b[3]) - (a[1] + a[3])
    return dx * dx + dy * dy


def _centre_tile(r) -> tuple[int, int]:
    return (r[0] + (r[2] - r[0]) // 2, r[1] + (r[3] - r[1]) // 2)


def _in_scene(r) -> bool:
    h = BELT_SCENE_HALF
    return -h <= r[0] and r[2] < h and -h <= r[1] and r[3] < h


def _pair_ok(a, b) -> bool:
    gx, gy = _gaps(a, b)
    return (
        gx * gx + gy * gy >= BELT_MIN_GAP_SQ
        and BELT_MIN_CENTRE_SQ4 <= _centre_dist_sq4(a, b) <= BELT_MAX_CENTRE_SQ4
    )


def _clipped(rect, clip: int) -> list[tuple[int, int]]:
    """A rectangle's tiles, x-major, minus corner triangles of `clip` tiles."""
    x0, y0, x1, y1 = rect
    return [
        (x, y)
        for x in range(x0, x1 + 1)
        for y in range(y0, y1 + 1)
        if min(x - x0, x1 - x) + min(y - y0, y1 - y) >= clip
    ]


def _wall_ok(tile, iron, coal, chest) -> bool:
    t = (tile[0], tile[1], tile[0], tile[1])
    return (
        _in_scene(t)
        and max(_gaps(t, iron)) >= BELT_CLEARANCE
        and max(_gaps(t, coal)) >= BELT_CLEARANCE
        and max(_gaps(t, chest)) >= BELT_CHEST_CLEARANCE
    )


def _belt_iron_patch(family: str, rng: random.Random):
    if family == "split_patch":
        axis = rng.randint(0, 1)
        long = rng.randint(10, 12)
        short = rng.randint(4, 6)
        cut = rng.randint(4, long - 6)
        w, h = (long, short) if axis == 0 else (short, long)
    elif family == "obstructed":
        w = rng.randint(6, 8)
        h = rng.randint(6, 8)
        clip = rng.randint(1, 2)
    else:
        w = rng.randint(4, 7)
        h = rng.randint(4, 7)
    x0 = rng.randint(-8, 8) - w // 2
    y0 = rng.randint(-8, 8) - h // 2
    rect = (x0, y0, x0 + w - 1, y0 + h - 1)
    if family == "split_patch":
        tiles = [
            (x, y)
            for x in range(rect[0], rect[2] + 1)
            for y in range(rect[1], rect[3] + 1)
            if ((x - x0) if axis == 0 else (y - y0)) not in (cut, cut + 1)
        ]
        return tiles, rect
    return _clipped(rect, clip if family == "obstructed" else 0), rect


def belt_smelting_scene(family: str, rng: random.Random) -> dict:
    """The generator's decisions (FactorioRL `belt_smelting.scene`), as tiles."""
    iron, iron_rect = _belt_iron_patch(family, rng)
    coal_w = rng.randint(4, 6)
    coal_h = rng.randint(4, 6)
    ix, iy = _centre_tile(iron_rect)
    low, high = BELT_CHEST_MANHATTAN[family]
    while True:
        cx = ix + rng.randint(-40, 40)
        cy = iy + rng.randint(-40, 40)
        chest_rect = (cx, cy, cx, cy)
        if (
            _in_scene(chest_rect)
            and _pair_ok(iron_rect, chest_rect)
            and low <= sum(_gaps(iron_rect, chest_rect)) <= high
        ):
            break
    while True:
        qx = ix + rng.randint(-40, 40)
        qy = iy + rng.randint(-40, 40)
        coal_rect = (qx, qy, qx + coal_w - 1, qy + coal_h - 1)
        if (
            _in_scene(coal_rect)
            and _pair_ok(iron_rect, coal_rect)
            and _pair_ok(chest_rect, coal_rect)
        ):
            break
    coal = _clipped(coal_rect, 1 if family == "obstructed" else 0)

    walls: list[tuple[int, int]] = []
    taken: set[tuple[int, int]] = set()
    segments = {"walled": (1, 2), "obstructed": (2, 3)}.get(family)
    if segments:
        rects = (iron_rect, coal_rect, chest_rect)
        pairs = ((0, 2), (0, 1), (1, 2))  # iron-chest, iron-coal, coal-chest
        for _ in range(rng.randint(*segments)):
            pair = rng.randint(0, 2)
            t = rng.randint(35, 65)
            length = rng.randint(4, 8) if family == "obstructed" else rng.randint(3, 6)
            offset = rng.randint(-3, 3)
            a = _centre_tile(rects[pairs[pair][0]])
            b = _centre_tile(rects[pairs[pair][1]])
            px = a[0] + (b[0] - a[0]) * t // 100
            py = a[1] + (b[1] - a[1]) * t // 100
            # Across the line between the two sites.
            across_x = abs(b[0] - a[0]) < abs(b[1] - a[1])
            segment = [
                (px + offset - length // 2 + k, py)
                if across_x
                else (px, py + offset - length // 2 + k)
                for k in range(length)
            ]
            if not all(_wall_ok(tile, iron_rect, coal_rect, chest_rect) for tile in segment):
                continue
            for tile in segment:
                if tile not in taken:
                    taken.add(tile)
                    walls.append(tile)

    bx0 = min(iron_rect[0], coal_rect[0], chest_rect[0])
    by0 = min(iron_rect[1], coal_rect[1], chest_rect[1])
    bx1 = max(iron_rect[2], coal_rect[2], chest_rect[2])
    by1 = max(iron_rect[3], coal_rect[3], chest_rect[3])
    clutter = {"obstructed": (6, 10), "far_chest": (18, 26)}.get(family)
    if clutter:
        m = BELT_CORRIDOR_MARGIN
        corridor = (
            min(iron_rect[0], chest_rect[0]) - m,
            min(iron_rect[1], chest_rect[1]) - m,
            max(iron_rect[2], chest_rect[2]) + m,
            max(iron_rect[3], chest_rect[3]) + m,
        )
        for _ in range(rng.randint(*clutter)):
            tile = (rng.randint(bx0, bx1), rng.randint(by0, by1))
            if tile in taken or not _wall_ok(tile, iron_rect, coal_rect, chest_rect):
                continue
            if family == "far_chest" and max(_gaps((*tile, *tile), corridor)) == 0:
                continue
            taken.add(tile)
            walls.append(tile)

    while True:
        start = (rng.randint(bx0, bx1), rng.randint(by0, by1))
        if start not in taken and start != (chest_rect[0], chest_rect[1]):
            break
    return {
        "iron": iron,
        "iron_rect": iron_rect,
        "coal": coal,
        "coal_rect": coal_rect,
        "chest": (chest_rect[0], chest_rect[1]),
        "walls": walls,
        "start": start,
    }


def _patch_centre(tiles) -> list[float]:
    return [
        sum(x for x, _ in tiles) / len(tiles) + 0.5,
        sum(y for _, y in tiles) / len(tiles) + 0.5,
    ]


def belt_smelting(family: str, rng: random.Random) -> dict:
    """FactorioRL `belt_smelting.generate`, as `Blueprint.to_dict` writes it."""
    s = belt_smelting_scene(family, rng)
    chest = [s["chest"][0] + 0.5, s["chest"][1] + 0.5]
    entities = [
        {
            "name": "wooden-chest",
            "position": list(chest),
            "direction": "north",
            "force": "player",
            "marker": "output",
        },
        *(_wall(x + 0.5, y + 0.5) for x, y in s["walls"]),
    ]
    resources = [
        *(_resource(x + 0.5, y + 0.5) for x, y in s["iron"]),
        *(dict(_resource(x + 0.5, y + 0.5), name="coal") for x, y in s["coal"]),
    ]
    return {
        "public_markers": list(BELT_MARKERS),
        "entities": entities,
        "resources": resources,
        "character": {
            "position": [s["start"][0] + 0.5, s["start"][1] + 0.5],
            "inventory": dict(BELT_INVENTORY),
        },
        "markers": {
            "iron": _patch_centre(s["iron"]),
            "coal": _patch_centre(s["coal"]),
            "output": list(chest),
        },
        "unlock_recipes": list(BELT_UNLOCK),
        "radius": 64,
    }


GENERATORS = {
    "construct_smelting_line": construct_smelting_line,
    "build_line": build_line,
    "plate_line": plate_line,
    "belt_smelting": belt_smelting,
}


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
