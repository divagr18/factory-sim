"""Python face of the simulator.

`Sim` drives the C core one decision at a time and, for parity checks,
renders its state in the shapes FactorioRL's engine produces: the `local-v2`
wire observation, the evaluator truth, and the recorder's hidden state. Those
renderings are slow and exist only to compare against the golden traces; the
fast path (tensors, masks) arrives in M4.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from fsim._fsim import ffi, lib

__all__ = ["Sim", "action_struct", "ffi", "lib", "scene_struct"]

ROOT = Path(__file__).resolve().parents[1]
#: Shipped inside the package, so an installed copy has the map without a checkout.
PACKAGE_DATA = Path(__file__).resolve().parent / "data"

ITEM_NAMES = {
    lib.IT_IRON_ORE: "iron-ore",
    lib.IT_COPPER_ORE: "copper-ore",
    lib.IT_COAL: "coal",
    lib.IT_STONE: "stone",
    lib.IT_IRON_PLATE: "iron-plate",
    lib.IT_COPPER_PLATE: "copper-plate",
    lib.IT_STONE_FURNACE: "stone-furnace",
    lib.IT_BURNER_DRILL: "burner-mining-drill",
    lib.IT_STONE_WALL: "stone-wall",
    lib.IT_WOOD: "wood",
    lib.IT_IRON_GEAR: "iron-gear-wheel",
    lib.IT_TRANSPORT_BELT: "transport-belt",
    lib.IT_SMALL_POLE: "small-electric-pole",
    lib.IT_WOODEN_CHEST: "wooden-chest",
    lib.IT_BURNER_INSERTER: "burner-inserter",
}
ITEM_IDS = {name: item for item, name in ITEM_NAMES.items()}

#: Entity kinds: prototype name and type, as the engine reports them. The rest
#: of what a kind is lives in fsim.c's KIND table; `lib.fsim_kind_flags` reads it.
KINDS = {
    lib.K_DRILL: ("burner-mining-drill", "mining-drill"),
    lib.K_FURNACE: ("stone-furnace", "furnace"),
    lib.K_WALL: ("stone-wall", "wall"),
    lib.K_PILE: ("item-on-ground", "item-entity"),
    lib.K_CHEST: ("wooden-chest", "container"),
    lib.K_BELT: ("transport-belt", "transport-belt"),
    lib.K_INSERTER: ("burner-inserter", "inserter"),
}
KIND_NAME = {kind: name for kind, (name, _) in KINDS.items()}
KIND_TYPE = {kind: type_ for kind, (_, type_) in KINDS.items()}
STATUS_NAME = {
    lib.ST_WORKING: "working",
    lib.ST_NORMAL: "normal",
    lib.ST_WAITING_FOR_SOURCE: "waiting_for_source_items",
    lib.ST_NO_INGREDIENTS: "no_ingredients",
    lib.ST_WAITING_FOR_SPACE: "waiting_for_space_in_destination",
    lib.ST_NO_FUEL: "no_fuel",
    lib.ST_NO_MINABLE: "no_minable_resources",
}
RESULT_NAME = {
    lib.R_COMPLETED: "completed",
    lib.R_RUNNING: "running",
    lib.R_REJECTED: "rejected",
    lib.R_FAILED: "failed",
    lib.R_CANCELLED: "cancelled",
}
ERROR_NAME = {
    lib.E_PRECONDITION: "precondition",
    lib.E_OUT_OF_REACH: "out_of_reach",
    lib.E_COLLISION: "collision",
    lib.E_NO_ITEMS: "no_items",
    lib.E_NO_SPACE: "no_space",
    lib.E_BUSY: "busy",
    lib.E_UNKNOWN_HANDLE: "unknown_handle",
    lib.E_TARGET_MISSING: "target_missing",
    lib.E_NOT_MINEABLE: "not_mineable",
    lib.E_INVALID_TARGET: "invalid_target",
    lib.E_TECH_LOCKED: "tech_locked",
    lib.E_ENGINE: "engine",
}
VERB_NAME = {
    lib.V_WAIT: "wait",
    lib.V_MOVE: "move",
    lib.V_MINE: "mine",
    lib.V_PLACE: "place",
    lib.V_ROTATE: "rotate",
    lib.V_TRANSFER: "transfer",
}
DIRECTIONS = ("north", "east", "south", "west")
#: Entities a scene may place itself, rather than the agent building them --
#: plate_line is handed an aligned drill and furnace, both empty; the logistics
#: scenes a rig of belts, inserters and chests, with fuel and contents.
SCENE_MACHINES = {
    "burner-mining-drill": lib.K_DRILL,
    "stone-furnace": lib.K_FURNACE,
    "wooden-chest": lib.K_CHEST,
    "transport-belt": lib.K_BELT,
    "burner-inserter": lib.K_INSERTER,
}
BELT_SHAPES = {lib.BELT_STRAIGHT: "straight", lib.BELT_LEFT: "left", lib.BELT_RIGHT: "right"}
STRIDES = {"move": 30, "step": 7, "nudge": 2}

BASE_RECIPES = (
    "burner-inserter", "burner-mining-drill", "copper-plate", "firearm-magazine",
    "iron-chest", "iron-gear-wheel", "iron-plate", "light-armor", "stone-brick",
    "stone-furnace", "transport-belt", "wooden-chest",
)  # fmt: skip
STEAM_POWER_RECIPES = ("boiler", "offshore-pump", "pipe", "pipe-to-ground", "steam-engine")
PROFILES = {
    "action": "primitive-v1",
    "action_version": 1,
    "assistance": "none",
    "drivers": {"crafting": "native", "mining": "native", "placement": "assembled"},
    "observation": "local-v2",
    "observation_version": 7,
}
EVENT_WINDOW = 8
#: Neutral entities are reported inside the scene's box only.
SCENE_RADIUS = 48


def fixed(value: float) -> int:
    """A tile coordinate in 1/256, truncated toward zero as a teleport does."""
    return int(value * 256)


def tiles(value: int) -> float:
    return value / 256


def luaize(value):
    """The shape `helpers.table_to_json` gives: integral doubles as integers,
    empty tables as objects."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, dict):
        return {k: luaize(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [luaize(v) for v in value] if value else {}
    return value


def g(value: float) -> str:
    return f"{value:.17g}"


def water_tiles() -> list[tuple[int, int]]:
    """The benchmark map's water, recorded by FactorioRL's probe.

    Missing, this raises rather than returning no water. It used to return an
    empty list, and a packaged copy of the simulator without the file would
    then have run every scene -- the frozen holdout included -- on a map with
    no water: a different map from the real game's, with nothing to say so.
    `Sim(water=[])` is still available for a deliberately dry map.
    """
    candidates = (
        PACKAGE_DATA / "terrain.json",
        ROOT / "tests" / "golden" / "sim-mechanics-m3.json",
        ROOT.parent / "FactorioRL" / "docs" / "evidence" / "sim-mechanics-m3.json",
    )
    for candidate in candidates:
        if candidate.exists():
            data = json.loads(candidate.read_text(encoding="utf-8"))
            return sorted((int(x), int(y)) for x, y in data["terrain"]["water"])
    raise FileNotFoundError(
        "the benchmark map's water is missing (sim-mechanics-m3.json); looked in "
        + ", ".join(str(c) for c in candidates)
        + ". Ship it with the simulator, or pass water=[] for a deliberately dry map."
    )


def _handle(value) -> int:
    if isinstance(value, str) and value.startswith("h") and value[1:].isdigit():
        return int(value[1:])
    return -1


def _endpoint(value) -> int:
    return 0 if value == "character" else _handle(value)


def scene_struct(blueprint: dict):
    """The C scene for a FactorioRL blueprint payload, and what keeps it alive."""
    resources = blueprint.get("resources") or []
    walls, machines = [], []
    for e in blueprint.get("entities") or []:
        if e["name"] == "stone-wall":
            walls.append(e)
        elif e["name"] in SCENE_MACHINES:
            machines.append(e)
        else:
            raise NotImplementedError(f"scene entity: {e!r}")
    keep = []
    scene = ffi.new("fsim_scene *")
    keep.append(scene)

    def array(values):
        data = ffi.new("int32_t[]", list(values) or [0])
        keep.append(data)
        return data

    scene.resource_count = len(resources)
    scene.resource_item = array(ITEM_IDS[r["name"]] for r in resources)
    scene.resource_tx = array(math.floor(r["position"][0]) for r in resources)
    scene.resource_ty = array(math.floor(r["position"][1]) for r in resources)
    scene.resource_amount = array(int(r.get("amount", 1000)) for r in resources)
    scene.wall_count = len(walls)
    scene.wall_x = array(fixed(w["position"][0]) for w in walls)
    scene.wall_y = array(fixed(w["position"][1]) for w in walls)
    # A 2x2 machine's declared position is already the integer centre it snaps
    # to, so it is used as given rather than pushed to a tile centre.
    scene.machine_count = len(machines)
    scene.machine_kind = array(SCENE_MACHINES[m["name"]] for m in machines)
    scene.machine_x = array(fixed(m["position"][0]) for m in machines)
    scene.machine_y = array(fixed(m["position"][1]) for m in machines)
    scene.machine_dir = array(DIRECTIONS.index(m.get("direction") or "north") * 4 for m in machines)
    # What each is given, as the mod's `LuaEntity.insert` calls: item names in
    # sorted order, one entity at a time.
    contents = [
        (index, ITEM_IDS[item], int(count))
        for index, m in enumerate(machines)
        for item, count in sorted((m.get("contents") or {}).items())
    ]
    scene.content_count = len(contents)
    scene.content_machine = array(c[0] for c in contents)
    scene.content_item = array(c[1] for c in contents)
    scene.content_amount = array(c[2] for c in contents)
    character = blueprint.get("character") or {}
    position = character.get("position") or [0, 0]
    scene.character.x = fixed(position[0])
    scene.character.y = fixed(position[1])
    inventory = sorted((character.get("inventory") or {}).items())
    scene.inventory_count = len(inventory)
    scene.inventory_item = array(ITEM_IDS[name] for name, _ in inventory)
    scene.inventory_amount = array(count for _, count in inventory)
    return scene, keep


def action_struct(key: str, arguments: dict | None = None):
    """A catalog key and its arguments as a C action."""
    arguments = arguments or {}
    a = ffi.new("fsim_action *")
    prefix, _, direction = key.partition("_")
    if prefix in STRIDES and direction in DIRECTIONS:
        a.verb = lib.V_MOVE
        a.direction = DIRECTIONS.index(direction)
        a.ticks = STRIDES[prefix]
    elif key == "place_at":
        a.verb = lib.V_PLACE
        a.item = ITEM_IDS.get(arguments["item"], lib.IT_NONE)
        a.direction = DIRECTIONS.index(arguments.get("direction", "north"))
        a.position.x = round(arguments["position"][0] * 256)
        a.position.y = round(arguments["position"][1] * 256)
    elif key == "mine_at":
        a.verb = lib.V_MINE
        a.handle = _handle(arguments["handle"])
        a.count = int(arguments.get("count", 1))
    elif key in ("rotate_at", "rotate_at_reverse"):
        a.verb = lib.V_ROTATE
        a.handle = _handle(arguments["handle"])
        a.reverse = 1 if key == "rotate_at_reverse" else 0
    elif key in ("give_to", "take_from"):
        a.verb = lib.V_TRANSFER
        a.from_handle = _endpoint(arguments["from"] if key == "take_from" else "character")
        a.to_handle = _endpoint(arguments["to"] if key == "give_to" else "character")
        a.item = ITEM_IDS.get(arguments["item"], lib.IT_NONE)
        a.count = int(arguments["count"])
    elif key == "wait":
        a.verb = lib.V_WAIT
    else:
        raise ValueError(f"unsupported action {key}")
    return a


#: Where a burner inserter's hand is drawn (`held_stack_position`), per tick of
#: each move, relative to the inserter in 1/256 tile, for one facing north
#: (pickup at (0, -256), drop at (0, 307)). Read off the engine by FactorioRL's
#: logistics probe (docs/evidence/sim-mechanics-m4-logistics.json, rigs `c2c`
#: and `self`); every chest, furnace and belt-drop swing it recorded follows
#: these exactly. It is a drawing position, not a path the hand's timing
#: depends on. Row k is the hand after k ticks of the move.
#:
#: The drawing is the hand's place on the ground raised by `HAND_LIFT` on
#: screen (y up), and the ground path of another facing is the north one turned
#: (east) or mirrored (south, west), so the hand swings through the west side
#: whether the inserter faces north or south. The lift and the facings were
#: read off FactorioRL's logistics parity traces, which have inserters facing
#: all four ways and match this on every chest, furnace and belt-drop swing.
HAND_APPROACH = ((0, -179), (0, -188), (0, -197), (0, -206), (0, -215), (0, -224), (0, -232),
                 (0, -241), (0, -256))  # fmt: skip
HAND_TO_DROP = (
    (0, -256), (-21, -279), (-44, -300), (-68, -318), (-93, -333), (-122, -351), (-144, -352),
    (-166, -350), (-186, -344), (-206, -337), (-223, -328), (-240, -316), (-255, -303),
    (-268, -286), (-279, -269), (-289, -249), (-296, -228), (-302, -205), (-305, -181),
    (-307, -156), (-306, -132), (-303, -105), (-299, -79), (-292, -50), (-284, -23), (-273, 5),
    (-261, 33), (-247, 60), (-231, 87), (-214, 113), (-195, 140), (-175, 165), (-154, 190),
    (-132, 214), (-109, 237), (-85, 258), (-61, 279), (-36, 298), (0, 307),
)  # fmt: skip
HAND_TO_PICKUP = (
    (0, 307), (-24, 282), (-47, 255), (-68, 227), (-87, 200), (-101, 164), (-120, 144),
    (-138, 123), (-155, 102), (-171, 79), (-186, 57), (-200, 34), (-212, 10), (-223, -13),
    (-232, -36), (-240, -59), (-247, -82), (-251, -104), (-254, -126), (-255, -147),
    (-255, -167), (-253, -185), (-249, -204), (-243, -220), (-236, -236), (-228, -250),
    (-217, -262), (-206, -273), (-193, -282), (-178, -289), (-163, -293), (-146, -295),
    (-128, -296), (-110, -293), (-91, -289), (-71, -282), (-51, -272), (-30, -261), (0, -256),
)  # fmt: skip
HAND_TO_SELF = (
    (0, -256), (20, -253), (38, -248), (55, -241), (70, -232), (83, -221), (95, -209),
    (104, -196), (112, -181), (117, -165), (121, -148), (123, -132), (123, -114), (121, -96),
    (118, -79), (114, -61), (108, -44), (101, -27), (94, -10), (85, -1), (76, 4), (67, 9),
    (57, 13), (47, 15), (37, 15), (28, 14), (19, 12), (11, 8), (2, 2),
)  # fmt: skip
HAND_SELF_BACK = (
    (2, 2), (9, 1), (17, -2), (26, -4), (35, -7), (45, -11), (54, -15), (64, -20), (74, -26),
    (84, -31), (93, -37), (101, -45), (109, -53), (115, -61), (120, -70), (124, -78),
    (127, -88), (128, -97), (127, -105), (125, -120), (120, -137), (114, -153), (106, -170),
    (96, -186), (84, -201), (70, -216), (54, -230), (36, -242), (0, -256),
)  # fmt: skip
#: How high the hand is drawn above the ground, per tick of a swing (both
#: ways) and of a swing to its own fuel slot (both ways); 0 while extending.
HAND_LIFT = (
    0, 15, 30, 44, 57, 70, 81, 92, 101, 110, 118, 125, 132, 137, 142, 145, 148, 150, 151, 151,
    151, 149, 147, 143, 139, 134, 128, 122, 114, 106, 96, 86, 75, 63, 50, 37, 22, 7, 0,
)  # fmt: skip
HAND_SELF_LIFT = (
    0, 7, 14, 19, 24, 28, 31, 34, 35, 35, 35, 34, 32, 29, 25, 20, 15, 9, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0,
)  # fmt: skip
HAND_TABLES = {
    lib.INS_APPROACH: (HAND_APPROACH, (0,) * len(HAND_APPROACH)),
    lib.INS_TO_DROP: (HAND_TO_DROP, HAND_LIFT),
    lib.INS_TO_PICKUP: (HAND_TO_PICKUP, HAND_LIFT),
    lib.INS_TO_SELF: (HAND_TO_SELF, HAND_SELF_LIFT),
    lib.INS_SELF_BACK: (HAND_SELF_BACK, HAND_SELF_LIFT),
}


def _face(offset: tuple[int, int], lift: int, direction: int) -> tuple[int, int]:
    """A north-facing drawn offset as an inserter facing `direction` draws it."""
    x, y = offset[0], offset[1] + lift
    x, y = {0: (x, y), 4: (-y, x), 8: (x, -y), 12: (y, x)}[direction]
    return x, y - lift


def _load_hand(e, record: dict) -> None:
    """Where inserter `e` is in its cycle, read back from a recording.

    The engine exports the hand's drawn position, not the cycle; the tables
    above map one to the other wherever the hand is on a drawn swing. A hand
    they do not place (see `hand_position`) keeps the simulator's own cycle.
    """
    status = record.get("status")
    if status == "waiting_for_source_items":
        e.phase, e.swing = lib.INS_WAIT_PICKUP, 0.0
        return
    if status == "waiting_for_space_in_destination":
        e.phase, e.swing = lib.INS_WAIT_DROP, 0.0
        return
    hand = record.get("held_stack_position")
    if not hand:
        return
    offset = (hand[0] - e.pos.x, hand[1] - e.pos.y)
    holding = bool(e.held)
    phases = (
        (lib.INS_TO_DROP, lib.INS_TO_SELF)
        if holding
        else (lib.INS_TO_PICKUP, lib.INS_SELF_BACK, lib.INS_APPROACH)
    )
    for phase in phases:
        table, lift = HAND_TABLES[phase]
        # The last row is the arrival, which ends the move on the same tick.
        for step, row in enumerate(table[:-1]):
            if _face(row, lift[step], e.direction) == offset:
                if (
                    phase == lib.INS_TO_DROP
                    and step == 0
                    and e.fuel.count == 0
                    and e.held in (lib.IT_COAL, lib.IT_WOOD)
                ):
                    phase = lib.INS_TO_SELF
                e.phase, e.swing = phase, float(step)
                return


def hand_position(e, pickup: list, drop: list) -> list:
    """`held_stack_position` for inserter `e`, from where it is in its cycle.

    Exact for every swing between chests, furnaces and belt drops the engine
    recorded, in all four facings. Not reproduced: a hand that took an item
    from a belt (it is drawn at the item, and the swing from there differs)
    and a tick with less than a full tick's energy (the hand moves part of a
    step), where this reads the step it is on.
    """
    if e.phase == lib.INS_WAIT_PICKUP:
        return list(pickup)
    if e.phase == lib.INS_WAIT_DROP:
        return list(drop)
    table, lift = HAND_TABLES[e.phase]
    step = min(int(e.swing), len(table) - 1)
    x, y = _face(table[step], lift[step], e.direction)
    return [e.pos.x + x, e.pos.y + y]


class Sim:
    def __init__(self, water: list[tuple[int, int]] | None = None) -> None:
        self.env = lib.fsim_new()
        water = water_tiles() if water is None else water
        water = sorted(water, key=lambda p: (p[1], p[0]))
        flat = ffi.new("int32_t[]", [v for p in water for v in p] or [0])
        lib.fsim_set_water(self.env, flat, len(water))
        self.blueprint: dict = {}
        self.steps = 0

    def __del__(self) -> None:
        env = getattr(self, "env", None)
        if env is not None:
            lib.fsim_free(env)
            self.env = None

    # ------------------------------------------------------------- lifecycle
    def reset(self, blueprint: dict) -> None:
        self.blueprint = blueprint
        self.steps = 0
        scene, self._scene_arrays = scene_struct(blueprint)
        lib.fsim_reset(self.env, scene)

    # ------------------------------------------------------------- stepping
    def action(self, key: str, arguments: dict | None = None):
        return action_struct(key, arguments)

    def step(self, key: str, arguments: dict | None = None, ticks: int = 30) -> None:
        lib.fsim_step(self.env, self.action(key, arguments), ticks)
        self.steps += 1

    @property
    def tick(self) -> int:
        return int(self.env.tick)

    # ------------------------------------------------------------- rendering
    def request_id(self, step: int, is_act: bool) -> str:
        return f"step-00000000-{step}" + (":act" if is_act else "")

    def events(self) -> list[dict]:
        env = self.env
        out = []
        for k in range(env.event_count):
            e = env.events[(env.event_head + k) % lib.FSIM_EVENT_LIMIT]
            record = {
                "seq": e.seq,
                "request_id": self.request_id(e.step, bool(e.is_act)),
                "tick": int(e.tick),
                "status": RESULT_NAME[e.status],
            }
            if e.action >= 0:
                record["action"] = VERB_NAME[e.action]
            if e.result != lib.R_NONE:
                record["result"] = RESULT_NAME[e.result]
            if e.error != lib.E_NONE:
                record["error"] = ERROR_NAME[e.error]
            out.append(record)
        return out

    def inventory(self) -> dict:
        totals: dict[str, int] = {}
        for i in range(lib.FSIM_MAIN_SLOTS):
            s = self.env.main[i]
            if s.count > 0:
                name = ITEM_NAMES[s.item]
                totals[name] = totals.get(name, 0) + s.count
        return totals

    def _entity_record(self, handle: int, e) -> dict:
        record = {
            "h": f"h{handle}",
            "name": KIND_NAME[e.kind],
            "type": KIND_TYPE[e.kind],
            "p": [tiles(e.pos.x), tiles(e.pos.y)],
        }
        flags = lib.fsim_kind_flags(e.kind)
        if flags & lib.KF_DIRECTED:
            record["d"] = e.direction
        if e.kind == lib.K_PILE:
            record["contents"] = {ITEM_NAMES[e.pile.item]: e.pile.count}
        if e.kind == lib.K_FURNACE and e.source.count > 0:
            record["contents"] = {ITEM_NAMES[e.source.item]: e.source.count}
        if e.kind == lib.K_CHEST:
            contents = self._chest_totals(e)
            if contents:
                record["contents"] = contents
        if e.kind != lib.K_PILE:
            record["status"] = e.status
            record["st"] = STATUS_NAME[e.status]
            record["working"] = e.status == lib.ST_WORKING
        if flags & lib.KF_BURNER:
            record["burns"] = True
            if e.fuel.count > 0:
                record["fuel"] = {ITEM_NAMES[e.fuel.item]: e.fuel.count}
        if e.kind == lib.K_FURNACE and e.result.count > 0:
            record["output"] = {ITEM_NAMES[e.result.item]: e.result.count}
        return record

    def observation(self) -> dict:
        env = self.env
        character = {
            "present": True,
            "position": [tiles(env.char_pos.x), tiles(env.char_pos.y)],
            "walking": bool(env.walk_pub),
            "direction": env.walk_pub_dir,
        }
        if env.mining:
            character["mining"] = {"active": True, "progress": env.mining_progress}
        entities = [
            self._entity_record(env.seen[k].handle, env.entities[env.seen[k].entity])
            for k in range(env.seen_count)
        ]
        tiles_out = []
        for k in range(env.tile_count):
            r = env.resources[env.tiles[k].resource]
            tiles_out.append(
                {
                    "h": f"h{env.tiles[k].handle}",
                    "name": ITEM_NAMES[r.item],
                    "p": [r.tx + 0.5, r.ty + 0.5],
                    "amount": r.amount,
                }
            )
        remembered = []
        for k in range(env.remembered_count):
            m = env.memory[env.remembered[k]]
            record = {
                "h": f"h{m.handle}",
                "name": KIND_NAME[m.kind],
                "type": KIND_TYPE[m.kind],
                "p": [tiles(m.pos.x), tiles(m.pos.y)],
                "age": int(env.tick - m.last_seen),
                "last_tick": int(m.last_seen),
            }
            if m.has_dir:
                record["d"] = m.direction
            if m.contents.count > 0:
                record["contents"] = {ITEM_NAMES[m.contents.item]: m.contents.count}
            remembered.append(record)
        inflight = []
        for i in range(lib.FSIM_MAX_INFLIGHT):
            f = env.inflight[i]
            if not f.used or f.terminal or f.verb < 0:
                continue
            record = {
                "request_id": self.request_id(f.step, True),
                "action": VERB_NAME[f.verb],
                "started_tick": int(f.started_tick),
                "_seq": f.seq,
            }
            if f.verb == lib.V_MINE:
                record["progress"] = env.mining_progress
                record["target"] = f"h{f.target}"
            elif f.deadline_tick >= 0 and f.deadline_tick > f.started_tick:
                record["progress"] = min(
                    1.0, (env.tick - f.started_tick) / (f.deadline_tick - f.started_tick)
                )
            inflight.append(record)
        inflight.sort(key=lambda r: r.pop("_seq"))
        events = self.events()
        counts = {
            "settled": len(events),
            "refused": sum(1 for e in events if e["status"] in ("failed", "cancelled")),
        }
        blocked = [[env.blocked[2 * k], env.blocked[2 * k + 1]] for k in range(env.blocked_count)]
        recipes = sorted(BASE_RECIPES + (STEAM_POWER_RECIPES if env.steam_power else ()))
        markers = self.blueprint.get("markers") or {}
        public = self.blueprint.get("public_markers") or []
        goal = {name: markers[name] for name in public if name in markers}
        body = {
            "episode_id": "ep-1",
            "tick": int(env.tick),
            "absolute_tick": int(env.tick),
            "profiles": PROFILES,
            "character": character,
            "inventory": self.inventory(),
            "sensor": {"radius": 32, "origin": [tiles(env.origin.x), tiles(env.origin.y)]},
            "terrain": {"blocked": blocked},
            "resources": {"tiles": tiles_out},
            "entities": entities,
            "remembered": remembered,
            "task": {"transfers": env.transfers, "items_moved": env.items_moved},
            "goal": goal,
            "inflight": inflight,
            "events": events[-EVENT_WINDOW:],
            "event_counts": counts,
            "recipes": recipes,
        }
        return luaize(body)

    def _player_entities(self):
        env = self.env
        for i in range(env.entity_count):
            e = env.entities[i]
            if e.alive and not e.neutral and e.kind != lib.K_PILE:
                yield e

    def truth(self) -> dict:
        env = self.env
        produced = {ITEM_NAMES[i]: env.produced[i] for i in ITEM_NAMES if env.produced[i] > 0}
        mined = {
            ITEM_NAMES[i]: env.mined_by_action[i] for i in ITEM_NAMES if env.mined_by_action[i] > 0
        }
        machine = {}
        for name, count in produced.items():
            value = count - mined.get(name, 0)
            if value > 0:
                machine[name] = value
        placed: dict[str, int] = {"character": 1}
        working: dict[str, int] = {}
        stored: dict[str, float] = {}
        burning: dict[str, float] = {}
        for e in self._player_entities():
            name = KIND_NAME[e.kind]
            placed[name] = placed.get(name, 0) + 1
            if e.status == lib.ST_WORKING:
                working[name] = working.get(name, 0) + 1
            if e.energy > 0:
                stored[name] = stored.get(name, 0) + e.energy
            if e.remaining > 0:
                burning[name] = burning.get(name, 0) + e.remaining
        built = {ITEM_NAMES[i]: env.built[i] for i in ITEM_NAMES if env.built[i] > 0}
        body = {
            "tick": int(env.tick),
            "markers": self.blueprint.get("markers") or {},
            "containers": {},
            "working": {},
            "produced": produced,
            "working_counts": working,
            "placed_counts": placed,
            "stored_energy": stored,
            "remaining_burning_fuel": burning,
            "built": built,
            "machine_produced": machine,
            "handcrafted": {},
            "mined_by_hand": mined,
            "by_hand_source": "mod_actions",
        }
        return luaize(body)

    @staticmethod
    def _slots(stack, size: int) -> dict:
        stacks = [[1, ITEM_NAMES[stack.item], stack.count]] if stack.count > 0 else []
        return {"size": size, "stacks": stacks}

    @staticmethod
    def _chest_totals(e) -> dict:
        totals: dict[str, int] = {}
        for i in range(lib.FSIM_CHEST_SLOTS):
            s = e.chest[i]
            if s.count > 0:
                totals[ITEM_NAMES[s.item]] = totals.get(ITEM_NAMES[s.item], 0) + s.count
        return totals

    @staticmethod
    def _lanes(e) -> list:
        """A belt's lanes as `[name, position, id]`, ascending position then id;
        ids are the simulator's own, for the trace normaliser to rename."""
        out = []
        for lane in range(2):
            items = [
                [ITEM_NAMES[it.item], it.pos, it.id]
                for it in e.lanes[lane].items[0 : e.lanes[lane].count]
            ]
            out.append(sorted(items, key=lambda item: (item[1], item[2])))
        return out

    @staticmethod
    def _inserter_points(e) -> tuple[list, list]:
        """Pickup and drop position, 1/256 tile: 1 and 1.19921875 tiles out
        along the inserter's direction, which points at the pickup."""
        ux, uy = {0: (0, -1), 4: (1, 0), 8: (0, 1), 12: (-1, 0)}[e.direction]
        pickup = [e.pos.x + 256 * ux, e.pos.y + 256 * uy]
        drop = [e.pos.x - 307 * ux, e.pos.y - 307 * uy]
        return pickup, drop

    def hidden(self) -> dict:
        env = self.env
        lib.fsim_refresh(env)
        entities = []
        for i in range(env.entity_count):
            e = env.entities[i]
            if not e.alive or e.kind == lib.K_PILE:
                continue
            if e.neutral and (
                abs(tiles(e.pos.x)) > SCENE_RADIUS or abs(tiles(e.pos.y)) > SCENE_RADIUS
            ):
                continue
            record = {
                "name": KIND_NAME[e.kind],
                "force": "neutral" if e.neutral else "player",
                "position": [e.pos.x, e.pos.y],
                "direction": e.direction,
                "status": STATUS_NAME.get(e.status),
                "energy": g(e.energy),
            }
            if e.kind == lib.K_WALL:
                record["inventories"] = {}
            elif e.kind == lib.K_CHEST:
                stacks = [
                    [i + 1, ITEM_NAMES[e.chest[i].item], e.chest[i].count]
                    for i in range(lib.FSIM_CHEST_SLOTS)
                    if e.chest[i].count > 0
                ]
                record["inventories"] = {"chest": {"size": lib.FSIM_CHEST_SLOTS, "stacks": stacks}}
            elif e.kind == lib.K_BELT:
                record["inventories"] = {}
                record["belt_shape"] = BELT_SHAPES[e.shape]
                record["lanes"] = self._lanes(e)
            else:
                record["remaining_burning_fuel"] = g(e.remaining)
                if e.burning:
                    record["currently_burning"] = ITEM_NAMES[e.burning]
                inventories = {
                    "fuel": self._slots(e.fuel, 1),
                    "burnt_result": {"size": 0, "stacks": []},
                }
                if e.kind == lib.K_DRILL:
                    record["mining_progress"] = g(e.progress)
                    record["bonus_mining_progress"] = "0"
                elif e.kind == lib.K_INSERTER:
                    pickup, drop = self._inserter_points(e)
                    record["pickup_position"] = pickup
                    record["drop_position"] = drop
                    record["held_stack_position"] = hand_position(e, pickup, drop)
                    if e.held:
                        record["held"] = {"name": ITEM_NAMES[e.held], "count": 1}
                else:
                    record["crafting_progress"] = g(e.progress)
                    record["bonus_progress"] = "0"
                    record["products_finished"] = e.products_finished
                    inventories["furnace_source"] = self._slots(e.source, 1)
                    inventories["furnace_result"] = self._slots(e.result, 1)
                record["inventories"] = inventories
            entities.append(record)
        entities.sort(key=lambda r: (r["position"][1], r["position"][0], r["name"]))
        ground = []
        for i in range(env.entity_count):
            e = env.entities[i]
            if e.alive and e.kind == lib.K_PILE:
                ground.append(
                    {
                        "name": ITEM_NAMES[e.pile.item],
                        "count": e.pile.count,
                        "position": [e.pos.x, e.pos.y],
                    }
                )
        ground.sort(key=lambda r: (r["position"][1], r["position"][0], r["name"]))
        resources = []
        for i in range(env.resource_count):
            r = env.resources[i]
            if r.alive:
                resources.append(
                    {
                        "name": ITEM_NAMES[r.item],
                        "position": [r.tx * 256 + 128, r.ty * 256 + 128],
                        "amount": r.amount,
                    }
                )
        resources.sort(key=lambda r: (r["position"][1], r["position"][0], r["name"]))
        main = [
            [i + 1, ITEM_NAMES[env.main[i].item], env.main[i].count]
            for i in range(lib.FSIM_MAIN_SLOTS)
            if env.main[i].count > 0
        ]
        character = {
            "position": [env.char_pos.x, env.char_pos.y],
            "direction": env.char_dir8,
            "mining": bool(env.mining),
            "mining_position": [env.mining_pos.x, env.mining_pos.y],
            "mining_progress": g(env.mining_progress if env.mining else 0.0),
            "walking": bool(env.walk_pub),
            "walking_direction": env.walk_pub_dir,
            "crafting_queue_size": 0,
            "main": {"size": lib.FSIM_MAIN_SLOTS, "stacks": main},
        }
        if env.selected_kind == 1:
            e = env.entities[env.selected_index]
            if e.alive:
                character["selected"] = {"name": KIND_NAME[e.kind], "position": [e.pos.x, e.pos.y]}
        elif env.selected_kind == 2:
            r = env.resources[env.selected_index]
            if r.alive:
                character["selected"] = {
                    "name": ITEM_NAMES[r.item],
                    "position": [r.tx * 256 + 128, r.ty * 256 + 128],
                }
        order = []
        for h in range(1, env.next_handle):
            rec = env.handles[h]
            if not rec.used:
                continue
            entry = {
                "handle": f"h{h}",
                "kind": "unit" if rec.kind == lib.H_UNIT else "tile",
                "first_seen": int(rec.first_seen),
            }
            if rec.destroyed_tick >= 0:
                entry["destroyed_tick"] = int(rec.destroyed_tick)
            if rec.kind == lib.H_UNIT:
                entry["name"] = KIND_NAME[rec.name]
                for i in range(env.entity_count):
                    e = env.entities[i]
                    if e.alive and e.unit == rec.unit:
                        entry["position"] = [e.pos.x, e.pos.y]
            else:
                entry["name"] = (
                    "item-on-ground" if rec.tile_type == lib.TT_PILE else ITEM_NAMES[rec.name]
                )
                entry["tile"] = [rec.tx, rec.ty]
            order.append(entry)
        entries = []
        for i in range(lib.FSIM_MAX_INFLIGHT):
            f = env.inflight[i]
            if not f.used or f.terminal:
                continue
            record = {
                "request_id": self.request_id(f.step, f.verb >= 0),
                "seq": f.seq,
                "action": VERB_NAME[f.verb] if f.verb >= 0 else "advance",
                "kind": VERB_NAME[f.verb] if f.verb >= 0 else "advance",
                "started_tick": int(f.started_tick),
            }
            if f.deadline_tick >= 0:
                record["deadline_tick"] = int(f.deadline_tick)
            if f.verb == lib.V_MINE:
                record["target"] = f"h{f.target}"
                record["goal"] = {"count": f.goal_count, "item": ITEM_NAMES[f.goal_item]}
                record["baseline"] = f.baseline
                record["queued"] = f.queued
            entries.append(record)
        entries.sort(key=lambda r: r["seq"])
        body = {
            "tick": int(env.tick),
            "entities": entities,
            "ground_items": ground,
            "resources": resources,
            "character": character,
            "handles": {"next_id": env.next_handle, "order": order},
            "inflight": {"next_seq": env.next_seq, "entries": entries},
            "event_seq": env.event_seq,
        }
        return luaize(body)

    # ------------------------------------------------------------- sync mode
    def load_hidden(self, hidden: dict) -> None:
        """Overwrite state with an engine recording of the same world.

        Everything the engine exports is taken from it: the character, every
        machine's energy, fuel, slots and progress, chests' slots, belt lanes,
        what an inserter holds, ore amounts and ground piles. Where an inserter
        is in its cycle is read back from its drawn hand (`_load_hand`). What
        the engine does not export -- which tile a drill is on in its cycle,
        whether it has delivered to its target before, whether a furnace has
        consumed its current ingredient, which belt items moved last tick --
        stays the simulator's.
        Entities are matched by name and position; a world whose entities
        differ is not the same world, and the comparison after the step says
        so.
        """
        env = self.env
        env.tick = hidden["tick"]
        c = hidden["character"]
        env.char_pos.x, env.char_pos.y = c["position"]
        env.char_dir8 = c["direction"]
        env.walk_pub = int(bool(c["walking"]))
        env.walk_pub_dir = c["walking_direction"]
        for i in range(lib.FSIM_MAIN_SLOTS):
            env.main[i].item = 0
            env.main[i].count = 0
        for index, name, count in c["main"]["stacks"] or []:
            env.main[index - 1].item = ITEM_IDS[name]
            env.main[index - 1].count = count
        if c["mining"]:
            progress = float(c["mining_progress"])
            duration = 1.0
            if env.mining_target_entity >= 0:
                duration = lib.fsim_kind_mining_time(env.entities[env.mining_target_entity].kind)
            env.mining_progress = progress
            env.mining_seconds = progress * duration

        by_place = {}
        for i in range(env.entity_count):
            e = env.entities[i]
            if e.alive and e.kind != lib.K_PILE:
                by_place[(KIND_NAME[e.kind], e.pos.x, e.pos.y)] = e
        for record in hidden["entities"]:
            e = by_place.get((record["name"], record["position"][0], record["position"][1]))
            if e is None:
                continue
            e.direction = record["direction"]
            e.status = {v: k for k, v in STATUS_NAME.items()}.get(record.get("status"), e.status)
            if record["name"] == "stone-wall":
                continue
            if record["name"] == "wooden-chest":
                for i in range(lib.FSIM_CHEST_SLOTS):
                    e.chest[i].item = 0
                    e.chest[i].count = 0
                for index, name, count in (record["inventories"].get("chest") or {}).get(
                    "stacks"
                ) or []:
                    e.chest[index - 1].item = ITEM_IDS[name]
                    e.chest[index - 1].count = count
                continue
            if record["name"] == "transport-belt":
                for lane, items in enumerate(record.get("lanes") or [[], []]):
                    items = items or []
                    e.lanes[lane].count = len(items)
                    for k, (name, position, *rest) in enumerate(items):
                        it = e.lanes[lane].items[k]
                        it.item = ITEM_IDS[name]
                        it.pos = position
                        it.id = rest[0] if rest else 0
                        it.moved = 1
                        env.next_item_id = max(env.next_item_id, it.id)
                continue
            e.energy = float(record["energy"])
            e.remaining = float(record["remaining_burning_fuel"])
            e.burning = ITEM_IDS.get(record.get("currently_burning"), 0)
            inventories = record["inventories"]

            def load(stack, key, inventories=inventories):
                stacks = (inventories.get(key) or {}).get("stacks") or []
                stack.item = ITEM_IDS[stacks[0][1]] if stacks else 0
                stack.count = stacks[0][2] if stacks else 0

            load(e.fuel, "fuel")
            if record["name"] == "burner-mining-drill":
                e.progress = float(record["mining_progress"])
                e.seconds = e.progress * 1.0
            elif record["name"] == "burner-inserter":
                held = record.get("held")
                e.held = ITEM_IDS[held["name"]] if held else 0
                _load_hand(e, record)
            else:
                load(e.source, "furnace_source")
                load(e.result, "furnace_result")
                e.progress = float(record["crafting_progress"])
                e.seconds = e.progress * 3.2
                e.products_finished = record["products_finished"]

        amounts = {tuple(r["position"]): r["amount"] for r in hidden["resources"]}
        for i in range(env.resource_count):
            r = env.resources[i]
            key = (r.tx * 256 + 128, r.ty * 256 + 128)
            if r.alive and key in amounts:
                r.amount = amounts[key]
        # Belt links and inserter targets follow what was loaded.
        env.entities_version += 1
        piles = {tuple(g["position"]): g for g in hidden["ground_items"]}
        for i in range(env.entity_count):
            e = env.entities[i]
            if e.alive and e.kind == lib.K_PILE and (e.pos.x, e.pos.y) in piles:
                e.pile.count = piles[(e.pos.x, e.pos.y)]["count"]

    def action_outcome(self) -> tuple[str | None, str | None]:
        act = self.env.act
        status = RESULT_NAME.get(act.status)
        error = ERROR_NAME.get(act.error) if act.status == lib.R_REJECTED else None
        return status, error
