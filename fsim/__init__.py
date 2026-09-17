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

KIND_NAME = {
    lib.K_DRILL: "burner-mining-drill",
    lib.K_FURNACE: "stone-furnace",
    lib.K_WALL: "stone-wall",
    lib.K_PILE: "item-on-ground",
}
KIND_TYPE = {
    lib.K_DRILL: "mining-drill",
    lib.K_FURNACE: "furnace",
    lib.K_WALL: "wall",
    lib.K_PILE: "item-entity",
}
STATUS_NAME = {
    lib.ST_WORKING: "working",
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
    """The benchmark map's water, recorded by FactorioRL's probe."""
    for candidate in (
        ROOT / "tests" / "golden" / "sim-mechanics-m3.json",
        ROOT.parent / "FactorioRL" / "docs" / "evidence" / "sim-mechanics-m3.json",
    ):
        if candidate.exists():
            data = json.loads(candidate.read_text(encoding="utf-8"))
            return sorted((int(x), int(y)) for x, y in data["terrain"]["water"])
    return []


def _handle(value) -> int:
    if isinstance(value, str) and value.startswith("h") and value[1:].isdigit():
        return int(value[1:])
    return -1


def _endpoint(value) -> int:
    return 0 if value == "character" else _handle(value)


def scene_struct(blueprint: dict):
    """The C scene for a FactorioRL blueprint payload, and what keeps it alive."""
    resources = blueprint.get("resources") or []
    walls = [e for e in blueprint.get("entities") or [] if e["name"] == "stone-wall"]
    others = [e for e in blueprint.get("entities") or [] if e["name"] != "stone-wall"]
    if others:
        raise NotImplementedError(f"scene entities beyond walls: {others[:1]}")
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
        if e.kind == lib.K_DRILL:
            record["d"] = e.direction
        if e.kind == lib.K_PILE:
            record["contents"] = {ITEM_NAMES[e.pile.item]: e.pile.count}
        if e.kind == lib.K_FURNACE and e.source.count > 0:
            record["contents"] = {ITEM_NAMES[e.source.item]: e.source.count}
        if e.kind != lib.K_PILE:
            record["status"] = e.status
            record["st"] = STATUS_NAME[e.status]
            record["working"] = e.status == lib.ST_WORKING
        if e.kind in (lib.K_DRILL, lib.K_FURNACE):
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

    def hidden(self) -> dict:
        env = self.env
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
        machine's energy, fuel, slots and progress, ore amounts and ground
        piles. What it does not export -- which tile a drill is on in its
        cycle, whether it has delivered to its target before, whether a
        furnace has consumed its current ingredient -- stays the simulator's.
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
                kind = env.entities[env.mining_target_entity].kind
                duration = {lib.K_DRILL: 0.3, lib.K_FURNACE: 0.2, lib.K_WALL: 0.2}.get(kind, 0.025)
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
