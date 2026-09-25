/* factory-sim core. See fsim.h for units and provenance. */
#include "fsim.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ constants */

#define TILE 256
#define STRIDE 38                 /* 0.15 tiles per tick, truncated to 1/256 */
#define CHAR_BOX 51               /* 0.19921875 */
#define MACHINE_BOX 179           /* 0.69921875: drill and furnace */
#define WALL_BOX 74               /* 0.2890625 */
#define PILE_BOX 35               /* 0.13671875 */
#define SENSOR_RADIUS 32
#define RESOURCE_RADIUS 12
#define BUILD_DISTANCE 10.0
#define REACH_DISTANCE 10.0
#define RESOURCE_REACH 2.7
#define DRILL_AREA 253            /* mining radius 0.99 */
#define FURNACE_SOURCE_CAP 54     /* measured: 49 ore + an insert of 20 takes 5 */
#define FUEL_VALUE_COAL 4000000.0
#define FUEL_VALUE_WOOD 2000000.0
#define DRILL_USAGE 2500.0
#define FURNACE_USAGE 1500.0
#define DRILL_SPEED 0.25
#define CHAR_MINING_SPEED 0.5
#define SMELT_SECONDS 3.2
#define STEAM_POWER_PLATES 50
#define STEAM_POWER_LAG 24        /* ticks from the 50th plate to the research */

/* Belts and burner inserters, measured tick by tick on the engine by
 * FactorioRL tools/probe_logistics.py (docs/sim-logistics.md,
 * docs/evidence/sim-mechanics-m4-logistics.json). */
#define BELT_BOX 102              /* 0.3984375, the collision box the probe read */
#define INSERTER_BOX 38           /* 0.1484375 */
#define CHEST_BOX 89              /* 0.35 truncated to 1/256 like the others: not measured */
#define BELT_MINING_TIME 0.1      /* not measured: the prototypes' value */
#define CHEST_MINING_TIME 0.1     /* not measured */
#define INSERTER_MINING_TIME 0.1  /* not measured */
#define BELT_SPEED 8              /* 0.03125 tiles per tick */
#define BELT_GAP 64               /* the closest two items on one lane come */
#define BELT_CURVE_OUTER 295      /* lane lengths on a turn */
#define BELT_CURVE_INNER 106
#define SIDELOAD_FAR 188          /* where a sideloading feed lane joins: the lane upstream... */
#define SIDELOAD_NEAR 67          /* ...and the one downstream (68 by geometry; 67 measured) */
#define INSERTER_PICKUP 256       /* pickup (0, -1) and drop (0, 1.19921875) from the */
#define INSERTER_DROP 307         /* inserter, along its direction, which points at the pickup */
#define INSERTER_USAGE 2400.0     /* max_energy_usage; the buffer holds 16/15 of it */
#define INSERTER_BUILT_FUEL 500000.0  /* a new inserter burns a quarter of a wood */
/* The hand's motion and its energy are the arm's (fsim.c, arm_step): pickup
 * to drop and drop to pickup take 38 ticks, the first 5 extending (2,400 J)
 * and the rest turning (650 J), 50 kJ a turn in single precision
 * (650 + 900/2^26 J a step). */
#define INSERTER_SOURCE_LIMIT 2   /* ore it keeps in a furnace's source */
#define INSERTER_FUEL_LIMIT 5     /* fuel it keeps in a fuel slot */


static const int32_t STACK_SIZE[IT_COUNT] = {
    0, 50, 50, 50, 50, 100, 100, 50, 50, 100, 100, 100, 100, 50, 50, 50,
};

static double mining_time_of_item(int32_t item) {
    switch (item) {
    case IT_IRON_ORE: case IT_COPPER_ORE: case IT_COAL: case IT_STONE: return 1.0;
    default: return 1.0;
    }
}

/* Energy an item gives when burnt, or 0 for an item that is not fuel. */
static double fuel_value(int32_t item) {
    switch (item) {
    case IT_COAL: return FUEL_VALUE_COAL;
    case IT_WOOD: return FUEL_VALUE_WOOD;
    default: return 0.0;
    }
}

static int is_fuel(int32_t item) { return fuel_value(item) > 0.0; }

/* ------------------------------------------------------------------ kinds
 *
 * What is constant about an entity kind, in one row per K_*. A new kind is a
 * K_* in fsim.h, a row here, a case in update_world if it runs, and its name
 * in fsim/__init__.py's KINDS. What its inventories accept is machine_accepts
 * and machine_slot. */

typedef struct {
    int32_t flags;          /* KF_* */
    int32_t box;            /* collision half-size, 1/256 tiles */
    int32_t size;           /* footprint side in tiles: odd snaps to a tile centre */
    int32_t item;           /* the item that places it and mining it returns */
    int32_t name_rank;      /* sort rank of its prototype name, by string order */
    int32_t status;         /* status when created */
    double mining_time;     /* seconds for the character to mine it */
    double usage;           /* burner draw per tick, J */
    int32_t rl_type;        /* encoders.ENTITY_TYPES index */
} kind_info;

static const kind_info KIND[K_COUNT] = {
    [K_NONE] = {0, 0, 0, IT_NONE, 9, ST_NONE, 1.0, 0.0, 11},
    /* burner-mining-drill */
    [K_DRILL] = {KF_COLLIDES | KF_BLOCKS_WALKING | KF_BURNER | KF_MACHINE | KF_DIRECTED,
                 MACHINE_BOX, 2, IT_BURNER_DRILL, 1, ST_NO_FUEL, 0.3, DRILL_USAGE, 3},
    /* stone-furnace */
    [K_FURNACE] = {KF_COLLIDES | KF_BLOCKS_WALKING | KF_BURNER | KF_MACHINE,
                   MACHINE_BOX, 2, IT_STONE_FURNACE, 3, ST_NO_FUEL, 0.2, FURNACE_USAGE, 1},
    /* stone-wall */
    [K_WALL] = {KF_COLLIDES | KF_BLOCKS_WALKING,
                WALL_BOX, 1, IT_STONE_WALL, 4, ST_WORKING, 0.2, 0.0, 10},
    /* item-on-ground: never placed, and mining it returns its pile */
    [K_PILE] = {0, PILE_BOX, 1, IT_NONE, 2, ST_NONE, 0.025, 0.0, 11},
    /* wooden-chest: the engine reports it `normal` */
    [K_CHEST] = {KF_COLLIDES | KF_BLOCKS_WALKING | KF_MACHINE,
                 CHEST_BOX, 1, IT_WOODEN_CHEST, 6, ST_NORMAL, CHEST_MINING_TIME, 0.0, 0},
    /* transport-belt: the character walks over it */
    [K_BELT] = {KF_COLLIDES | KF_DIRECTED,
                BELT_BOX, 1, IT_TRANSPORT_BELT, 5, ST_WORKING, BELT_MINING_TIME, 0.0, 4},
    /* burner-inserter: its direction points at its pickup tile */
    [K_INSERTER] = {KF_COLLIDES | KF_BLOCKS_WALKING | KF_BURNER | KF_MACHINE | KF_DIRECTED,
                    INSERTER_BOX, 1, IT_BURNER_INSERTER, 0, ST_WORKING, INSERTER_MINING_TIME,
                    INSERTER_USAGE, 5},
};

static const kind_info *kind_of(int32_t kind) {
    return &KIND[kind > K_NONE && kind < K_COUNT ? kind : K_NONE];
}

static int has_flag(int32_t kind, int32_t flag) { return (kind_of(kind)->flags & flag) != 0; }

int32_t fsim_kind_flags(int32_t kind) { return kind_of(kind)->flags; }

double fsim_kind_mining_time(int32_t kind) { return kind_of(kind)->mining_time; }

double fsim_capacity(int32_t kind) {
    /* A burner's buffer holds 16/15 of its per-tick draw. */
    return kind_of(kind)->usage * 16.0 / 15.0;
}

static double entity_mining_time(int32_t kind) { return kind_of(kind)->mining_time; }

static int32_t entity_item(int32_t kind) { return kind_of(kind)->item; }

static int32_t box_of(int32_t kind) { return kind_of(kind)->box; }

static int32_t name_rank(int32_t kind) { return kind_of(kind)->name_rank; }

/* The kind an item places, or K_NONE. */
static int32_t kind_placed_by(int32_t item) {
    if (item == IT_NONE) return K_NONE;
    for (int32_t k = K_NONE + 1; k < K_COUNT; k++)
        if (KIND[k].item == item) return k;
    return K_NONE;
}

/* The field sizes in fsim.h are literals (cffi); keep them to the limits. */
#define FIELD_COUNT(field) (sizeof(((fsim_env *)0)->field) / sizeof(((fsim_env *)0)->field[0]))
_Static_assert(FIELD_COUNT(events) == FSIM_EVENT_LIMIT, "events");
_Static_assert(FIELD_COUNT(main) == FSIM_MAIN_SLOTS, "main");
_Static_assert(FIELD_COUNT(entities) == FSIM_MAX_ENTITIES, "entities");
_Static_assert(FIELD_COUNT(fillers) == 4 * FSIM_MAX_FILLERS, "fillers");
_Static_assert(FIELD_COUNT(resources) == FSIM_MAX_RESOURCES, "resources");
_Static_assert(FIELD_COUNT(handles) == FSIM_MAX_HANDLES, "handles");
_Static_assert(FIELD_COUNT(inflight) == FSIM_MAX_INFLIGHT, "inflight");
_Static_assert(FIELD_COUNT(superseded) == FSIM_MAX_SUPERSEDED, "superseded");
_Static_assert(FIELD_COUNT(memory) == FSIM_MAX_MEMORY, "memory");
_Static_assert(FIELD_COUNT(produced) == IT_COUNT, "produced");
_Static_assert(FIELD_COUNT(mined_by_action) == IT_COUNT, "mined_by_action");
_Static_assert(FIELD_COUNT(built) == IT_COUNT, "built");
_Static_assert(FIELD_COUNT(water) == 2 * FSIM_MAX_WATER, "water");
_Static_assert(FIELD_COUNT(seen) == FSIM_MAX_SWEEP, "seen");
_Static_assert(FIELD_COUNT(tiles) == FSIM_MAX_TILES, "tiles");
_Static_assert(FIELD_COUNT(blocked) == 2 * FSIM_MAX_BLOCKED, "blocked");
_Static_assert(FIELD_COUNT(remembered) == FSIM_MAX_MEMORY, "remembered");
_Static_assert(FIELD_COUNT(inserters) == FSIM_MAX_ENTITIES, "inserters");
_Static_assert(FIELD_COUNT(chain_first) == FSIM_MAX_LANES, "chain_first");
_Static_assert(FIELD_COUNT(chain_size) == FSIM_MAX_LANES, "chain_size");
_Static_assert(FIELD_COUNT(chain_lanes) == FSIM_MAX_LANES, "chain_lanes");
_Static_assert(FSIM_MAX_LANES == 2 * FSIM_MAX_ENTITIES, "lanes");
_Static_assert(sizeof(((fsim_entity *)0)->chest) / sizeof(fsim_stack) == FSIM_CHEST_SLOTS, "chest");
_Static_assert(sizeof(((fsim_lane *)0)->items) / sizeof(fsim_belt_item) == FSIM_LANE_ITEMS, "lane");

/* ------------------------------------------------------------------ helpers */

static void wake(fsim_env *env, int32_t index);
static void belt_line_added(fsim_env *env, int32_t ref);
static void arm_draw(fsim_entity *s);

static int64_t floordiv(int64_t a, int64_t b) {
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) q -= 1;
    return q;
}

static double tiles(int32_t v) { return (double)v / TILE; }

/* The centre an entity of `kind` snaps to from a requested coordinate: the
 * tile centre for an odd footprint, the nearest tile corner for an even one. */
static int32_t snap(int32_t kind, int32_t v) {
    if (kind_of(kind)->size % 2) return (int32_t)floordiv(v, TILE) * TILE + TILE / 2;
    return (int32_t)floordiv(v + TILE / 2, TILE) * TILE;
}

static int32_t count_main(const fsim_env *env, int32_t item) {
    int32_t total = 0;
    for (int i = 0; i < FSIM_MAIN_SLOTS; i++)
        if (env->main[i].item == item) total += env->main[i].count;
    return total;
}

static int32_t empty_slots(const fsim_env *env) {
    int32_t n = 0;
    for (int i = 0; i < FSIM_MAIN_SLOTS; i++)
        if (env->main[i].count == 0) n++;
    return n;
}

/* Room for `item` in the main inventory. */
static int32_t main_room(const fsim_env *env, int32_t item) {
    int32_t room = 0;
    int32_t stack = STACK_SIZE[item];
    for (int i = 0; i < FSIM_MAIN_SLOTS; i++) {
        if (env->main[i].count == 0) room += stack;
        else if (env->main[i].item == item) room += stack - env->main[i].count;
    }
    return room;
}

/* Insert into the main inventory: existing stacks first, then empty slots. */
static int32_t insert_main(fsim_env *env, int32_t item, int32_t count) {
    int32_t left = count;
    int32_t stack = STACK_SIZE[item];
    for (int i = 0; i < FSIM_MAIN_SLOTS && left > 0; i++) {
        fsim_stack *s = &env->main[i];
        if (s->count > 0 && s->item == item && s->count < stack) {
            int32_t take = stack - s->count;
            if (take > left) take = left;
            s->count += take;
            left -= take;
        }
    }
    for (int i = 0; i < FSIM_MAIN_SLOTS && left > 0; i++) {
        fsim_stack *s = &env->main[i];
        if (s->count == 0) {
            int32_t take = left > stack ? stack : left;
            s->item = item;
            s->count = take;
            left -= take;
        }
    }
    return count - left;
}

/* Remove from the main inventory, first stacks first. */
static int32_t remove_main(fsim_env *env, int32_t item, int32_t count) {
    int32_t left = count;
    for (int i = 0; i < FSIM_MAIN_SLOTS && left > 0; i++) {
        fsim_stack *s = &env->main[i];
        if (s->count > 0 && s->item == item) {
            int32_t take = s->count > left ? left : s->count;
            s->count -= take;
            left -= take;
            if (s->count == 0) s->item = IT_NONE;
        }
    }
    return count - left;
}

/* A single-slot inventory. */
static int32_t slot_room(const fsim_stack *s, int32_t item, int32_t cap) {
    if (s->count == 0) return cap;
    if (s->item != item) return 0;
    return cap - s->count;
}

static int32_t slot_insert(fsim_stack *s, int32_t item, int32_t count, int32_t cap) {
    int32_t room = slot_room(s, item, cap);
    int32_t take = count > room ? room : count;
    if (take <= 0) return 0;
    s->item = item;
    s->count += take;
    return take;
}

static int32_t slot_remove(fsim_stack *s, int32_t item, int32_t count) {
    if (s->count == 0 || s->item != item) return 0;
    int32_t take = s->count > count ? count : s->count;
    s->count -= take;
    if (s->count == 0) s->item = IT_NONE;
    return take;
}

/* A chest: sixteen slots, filled as the main inventory is. */
static int32_t chest_count(const fsim_entity *c, int32_t item) {
    int32_t total = 0;
    for (int i = 0; i < FSIM_CHEST_SLOTS; i++)
        if (c->chest[i].count > 0 && c->chest[i].item == item) total += c->chest[i].count;
    return total;
}

static int32_t chest_room(const fsim_entity *c, int32_t item) {
    int32_t room = 0;
    for (int i = 0; i < FSIM_CHEST_SLOTS; i++) {
        if (c->chest[i].count == 0) room += STACK_SIZE[item];
        else if (c->chest[i].item == item) room += STACK_SIZE[item] - c->chest[i].count;
    }
    return room;
}

static int32_t chest_insert(fsim_entity *c, int32_t item, int32_t count) {
    int32_t left = count;
    for (int i = 0; i < FSIM_CHEST_SLOTS && left > 0; i++)
        if (c->chest[i].count > 0 && c->chest[i].item == item)
            left -= slot_insert(&c->chest[i], item, left, STACK_SIZE[item]);
    for (int i = 0; i < FSIM_CHEST_SLOTS && left > 0; i++)
        if (c->chest[i].count == 0) left -= slot_insert(&c->chest[i], item, left, STACK_SIZE[item]);
    return count - left;
}

static int32_t chest_remove(fsim_entity *c, int32_t item, int32_t count) {
    int32_t left = count;
    for (int i = 0; i < FSIM_CHEST_SLOTS && left > 0; i++)
        left -= slot_remove(&c->chest[i], item, left);
    return count - left;
}

/* ------------------------------------------------------------------ events */

static void record_event(fsim_env *env, int32_t step, int32_t is_act, int32_t status,
                         int32_t action, int32_t result, int32_t error) {
    env->event_seq += 1;
    int32_t index;
    if (env->event_count < FSIM_EVENT_LIMIT) {
        index = (env->event_head + env->event_count) % FSIM_EVENT_LIMIT;
        env->event_count += 1;
    } else {
        index = env->event_head;
        env->event_head = (env->event_head + 1) % FSIM_EVENT_LIMIT;
    }
    fsim_event *e = &env->events[index];
    e->seq = env->event_seq;
    e->step = step;
    e->is_act = is_act;
    e->tick = env->tick;
    e->status = status;
    e->action = action;
    e->result = result;
    e->error = error;
}

/* ------------------------------------------------------------------ handles */

/* Handle 0 names nothing: minting returns it when the table is full. */
static int32_t mint_unit(fsim_env *env, int32_t entity_index) {
    const fsim_entity *e = &env->entities[entity_index];
    for (int32_t h = 1; h < env->next_handle; h++) {
        fsim_handle *rec = &env->handles[h];
        if (rec->used && rec->kind == H_UNIT && rec->unit == e->unit && rec->destroyed_tick < 0)
            return h;
    }
    if (env->next_handle >= FSIM_MAX_HANDLES) return 0;
    int32_t h = env->next_handle++;
    fsim_handle *rec = &env->handles[h];
    memset(rec, 0, sizeof(*rec));
    rec->used = 1;
    rec->kind = H_UNIT;
    rec->unit = e->unit;
    rec->name = e->kind;
    rec->first_seen = env->tick;
    rec->destroyed_tick = -1;
    rec->last_pos_x = e->pos.x;
    rec->last_pos_y = e->pos.y;
    return h;
}

/* Tile handles are keyed on the tile alone, as the mod keys them: a pile on an
 * ore tile resolves to the ore tile's handle. */
static int32_t mint_tile(fsim_env *env, int32_t tx, int32_t ty, int32_t tile_type, int32_t name) {
    for (int32_t h = 1; h < env->next_handle; h++) {
        fsim_handle *rec = &env->handles[h];
        if (rec->used && rec->kind == H_TILE && rec->tx == tx && rec->ty == ty &&
            rec->destroyed_tick < 0)
            return h;
    }
    if (env->next_handle >= FSIM_MAX_HANDLES) return 0;
    int32_t h = env->next_handle++;
    fsim_handle *rec = &env->handles[h];
    memset(rec, 0, sizeof(*rec));
    rec->used = 1;
    rec->kind = H_TILE;
    rec->tx = tx;
    rec->ty = ty;
    rec->tile_type = tile_type;
    rec->name = name;
    rec->first_seen = env->tick;
    rec->destroyed_tick = -1;
    return h;
}

static int32_t find_unit(const fsim_env *env, int32_t unit) {
    for (int32_t i = 0; i < env->entity_count; i++)
        if (env->entities[i].alive && env->entities[i].unit == unit) return i;
    return -1;
}

static int32_t find_resource_at(const fsim_env *env, int32_t tx, int32_t ty) {
    for (int32_t i = 0; i < env->resource_count; i++) {
        const fsim_resource *r = &env->resources[i];
        if (r->alive && r->tx == tx && r->ty == ty) return i;
    }
    return -1;
}

static int32_t find_pile_near_tile(const fsim_env *env, int32_t tx, int32_t ty) {
    int64_t cx = (int64_t)tx * TILE + TILE / 2;
    int64_t cy = (int64_t)ty * TILE + TILE / 2;
    for (int32_t i = 0; i < env->entity_count; i++) {
        const fsim_entity *e = &env->entities[i];
        if (!e->alive || e->kind != K_PILE) continue;
        int64_t dx = e->pos.x - cx, dy = e->pos.y - cy;
        if (dx * dx + dy * dy <= (int64_t)(TILE / 2) * (TILE / 2)) return i;
    }
    return -1;
}

/* kind: 1 entity, 2 resource. Returns 0 on success or an error code. */
int32_t fsim_resolve(fsim_env *env, int32_t h, int32_t *kind, int32_t *index) {
    *kind = 0;
    *index = -1;
    if (h <= 0 || h >= env->next_handle || !env->handles[h].used) return E_UNKNOWN_HANDLE;
    fsim_handle *rec = &env->handles[h];
    if (rec->destroyed_tick >= 0) return E_TARGET_MISSING;
    if (rec->kind == H_UNIT) {
        int32_t i = find_unit(env, rec->unit);
        if (i >= 0) {
            *kind = 1;
            *index = i;
            return 0;
        }
        rec->destroyed_tick = env->tick;
        return E_TARGET_MISSING;
    }
    if (rec->tile_type == TT_RESOURCE) {
        int32_t i = find_resource_at(env, rec->tx, rec->ty);
        if (i >= 0 && env->resources[i].item == rec->name) {
            *kind = 2;
            *index = i;
            return 0;
        }
    } else {
        int32_t i = find_pile_near_tile(env, rec->tx, rec->ty);
        if (i >= 0) {
            *kind = 1;
            *index = i;
            return 0;
        }
    }
    rec->destroyed_tick = env->tick;
    return E_TARGET_MISSING;
}

/* A unit destroyed during an update: the destroy event reports the tick
 * before, and the registry forgets the unit at once. */
static void unit_destroyed(fsim_env *env, int32_t unit) {
    for (int32_t h = 1; h < env->next_handle; h++) {
        fsim_handle *rec = &env->handles[h];
        if (rec->used && rec->kind == H_UNIT && rec->unit == unit && rec->destroyed_tick < 0) {
            rec->destroyed_tick = env->tick - 1;
        }
    }
}

/* ------------------------------------------------------------------ geometry */

static int water_at(const fsim_env *env, int32_t tx, int32_t ty) {
    /* sorted by (y, x): binary search */
    int32_t lo = 0, hi = env->water_count - 1;
    while (lo <= hi) {
        int32_t mid = (lo + hi) / 2;
        int32_t my = env->water[2 * mid + 1], mx = env->water[2 * mid];
        if (my == ty && mx == tx) return 1;
        if (my < ty || (my == ty && mx < tx)) lo = mid + 1;
        else hi = mid - 1;
    }
    return 0;
}

/* Does a box of half-size `r` at `p` touch water? `inclusive` counts touching. */
static int box_hits_water(const fsim_env *env, fsim_pos p, int32_t r, int inclusive) {
    int32_t x0 = p.x - r, x1 = p.x + r, y0 = p.y - r, y1 = p.y + r;
    int64_t tx0 = floordiv(x0, TILE), tx1 = floordiv(x1, TILE);
    int64_t ty0 = floordiv(y0, TILE), ty1 = floordiv(y1, TILE);
    for (int64_t ty = ty0; ty <= ty1; ty++) {
        for (int64_t tx = tx0; tx <= tx1; tx++) {
            if (!water_at(env, (int32_t)tx, (int32_t)ty)) continue;
            int64_t lx = tx * TILE, hx = lx + TILE, ly = ty * TILE, hy = ly + TILE;
            int hit = inclusive ? (x1 >= lx && x0 <= hx && y1 >= ly && y0 <= hy)
                                : (x1 > lx && x0 < hx && y1 > ly && y0 < hy);
            if (hit) return 1;
        }
    }
    return 0;
}

/* ------------------------------------------------------------ character motion
 *
 * Measured on the engine (FactorioRL tools/probe_corner_slide.py,
 * docs/evidence/sim-mechanics-m5-slide*.json):
 *
 * - Two aligned obstacles whose boxes are closer than GAP_SOLID block the
 *   character between them, although it would fit: two adjacent walls (108/256
 *   apart) stop a 102/256 character mid-gap, while a wall beside a furnace
 *   (131/256) and adjacent furnaces (154/256) let it through. The gap is
 *   modelled as a box of its own.
 * - A blocked stride turns into a slide when the obstacle can be cleared
 *   sideways: towards the side the character's centre is on (a dead-centre
 *   walker tries +y first when walking east or west, -x when walking north or
 *   south, then the other side), by up to a stride per tick with no forward
 *   progress, only if the whole clearance is at most SLIDE_LIMIT and a stride
 *   from the cleared position is free.
 * - Otherwise the character creeps forward to contact (see walk_one_tick).
 */

#define GAP_SOLID 128             /* 108 blocks, 131 passes: bounded, not pinned */
#define SLIDE_LIMIT 179           /* 175 slides, 183 does not: bounded, not pinned */

typedef struct {
    int32_t x0, y0, x1, y1;
} fsim_box;

static int solid_entity(const fsim_entity *e) {
    return e->alive && has_flag(e->kind, KF_BLOCKS_WALKING);
}

static fsim_box entity_box(const fsim_entity *e) {
    int32_t r = box_of(e->kind);
    fsim_box b = {e->pos.x - r, e->pos.y - r, e->pos.x + r, e->pos.y + r};
    return b;
}

static void rebuild_fillers(fsim_env *env) {
    env->filler_count = 0;
    for (int32_t i = 0; i < env->entity_count; i++) {
        if (!solid_entity(&env->entities[i])) continue;
        fsim_box a = entity_box(&env->entities[i]);
        for (int32_t j = 0; j < env->entity_count; j++) {
            if (j == i || !solid_entity(&env->entities[j])) continue;
            fsim_box b = entity_box(&env->entities[j]);
            fsim_box f;
            int32_t gap;
            if (a.x1 < b.x0 && a.y0 <= b.y1 && b.y0 <= a.y1) {   /* b east of a */
                gap = b.x0 - a.x1;
                f.x0 = a.x1; f.x1 = b.x0;
                f.y0 = a.y0 > b.y0 ? a.y0 : b.y0;
                f.y1 = a.y1 < b.y1 ? a.y1 : b.y1;
            } else if (a.y1 < b.y0 && a.x0 <= b.x1 && b.x0 <= a.x1) {  /* b south of a */
                gap = b.y0 - a.y1;
                f.y0 = a.y1; f.y1 = b.y0;
                f.x0 = a.x0 > b.x0 ? a.x0 : b.x0;
                f.x1 = a.x1 < b.x1 ? a.x1 : b.x1;
            } else {
                continue;
            }
            if (gap >= GAP_SOLID || env->filler_count >= FSIM_MAX_FILLERS) continue;
            int32_t *out = &env->fillers[4 * env->filler_count++];
            out[0] = f.x0; out[1] = f.y0; out[2] = f.x1; out[3] = f.y1;
        }
    }
    env->fillers_version = env->entities_version;
}

static int box_blocks(fsim_box b, fsim_pos p) {
    /* Boxes that only touch still collide. */
    return p.x + CHAR_BOX >= b.x0 && p.x - CHAR_BOX <= b.x1 &&
           p.y + CHAR_BOX >= b.y0 && p.y - CHAR_BOX <= b.y1;
}

/* The first obstacle the character at `p` collides with: an entity, then a
 * gap filler. Returns 0 when only water (or nothing) is in the way. */
static int blocking_box(fsim_env *env, fsim_pos p, fsim_box *out) {
    if (env->fillers_version != env->entities_version) rebuild_fillers(env);
    for (int32_t i = 0; i < env->entity_count; i++) {
        const fsim_entity *e = &env->entities[i];
        if (!solid_entity(e)) continue;
        fsim_box b = entity_box(e);
        if (box_blocks(b, p)) {
            *out = b;
            return 1;
        }
    }
    for (int32_t k = 0; k < env->filler_count; k++) {
        const int32_t *f = &env->fillers[4 * k];
        fsim_box b = {f[0], f[1], f[2], f[3]};
        if (box_blocks(b, p)) {
            *out = b;
            return 1;
        }
    }
    return 0;
}

static int character_blocked(fsim_env *env, fsim_pos p) {
    fsim_box b;
    if (blocking_box(env, p, &b)) return 1;
    return box_hits_water(env, p, CHAR_BOX, 1);
}

/* Slide around the obstacle `next` runs into, if the engine would; `stride`
 * is the move's length (a walking stride, or a belt's carry). */
static int try_slide(fsim_env *env, int32_t ux, int32_t uy, fsim_pos next, int32_t stride) {
    fsim_box b;
    if (!blocking_box(env, next, &b)) return 0;
    int horizontal = ux != 0;
    int32_t c = horizontal ? next.y : next.x;
    int32_t lo = horizontal ? b.y0 : b.x0, hi = horizontal ? b.y1 : b.x1;
    int32_t rel2 = 2 * c - (lo + hi);
    int32_t sides[2];
    int32_t n = 0;
    if (rel2 > 0) {
        sides[n++] = 1;
    } else if (rel2 < 0) {
        sides[n++] = -1;
    } else {
        sides[n++] = horizontal ? 1 : -1;
        sides[n++] = horizontal ? -1 : 1;
    }
    for (int32_t k = 0; k < n; k++) {
        int32_t side = sides[k];
        int32_t need = side > 0 ? hi + CHAR_BOX + 1 - c : c - (lo - CHAR_BOX - 1);
        if (need <= 0 || need > SLIDE_LIMIT) continue;
        fsim_pos cleared = env->char_pos;
        if (horizontal) cleared.y += side * need;
        else cleared.x += side * need;
        fsim_pos ahead = {cleared.x + ux * stride, cleared.y + uy * stride};
        if (character_blocked(env, ahead)) continue;
        int32_t step = need < stride ? need : stride;
        fsim_pos moved = env->char_pos;
        if (horizontal) moved.y += side * step;
        else moved.x += side * step;
        if (character_blocked(env, moved)) continue;
        env->char_pos = moved;
        return 1;
    }
    return 0;
}

/* One move of `stride` in `dir16`, as the engine moves the character: the
 * whole move if it is clear, else a slide, else a creep to contact. */
static void move_character(fsim_env *env, int32_t dir16, int32_t stride) {
    int32_t ux = 0, uy = 0;
    if (dir16 == 0) uy = -1;
    else if (dir16 == 4) ux = 1;
    else if (dir16 == 8) uy = 1;
    else if (dir16 == 12) ux = -1;
    else return;
    fsim_pos next = {env->char_pos.x + ux * stride, env->char_pos.y + uy * stride};
    if (!character_blocked(env, next)) {
        env->char_pos = next;
        return;
    }
    if (try_slide(env, ux, uy, next, stride)) return;
    /* Creep: half a stride, then 1/32 of a tile, then exactly to contact --
     * measured walking, from every start phase against a furnace (a gap of 37
     * goes 19, 8, 8, 2; a gap of 13 goes 8, 5; a gap of 7 goes 7). A belt's
     * carry was only seen to slide (probe_logistics2 `char_chestend`). */
    const int32_t CREEP[2] = {stride / 2, 8};
    for (int32_t k = 0; k < 2; k++) {
        if (CREEP[k] >= stride) continue;
        fsim_pos p = {env->char_pos.x + ux * CREEP[k], env->char_pos.y + uy * CREEP[k]};
        if (!character_blocked(env, p)) {
            env->char_pos = p;
            return;
        }
    }
    for (int32_t step = 7; step >= 1; step--) {
        fsim_pos p = {env->char_pos.x + ux * step, env->char_pos.y + uy * step};
        if (!character_blocked(env, p)) {
            env->char_pos = p;
            return;
        }
    }
}

static void walk_one_tick(fsim_env *env, int32_t dir16) { move_character(env, dir16, STRIDE); }

static double centre_distance(fsim_pos a, fsim_pos b) {
    double dx = tiles(a.x - b.x), dy = tiles(a.y - b.y);
    return sqrt(dx * dx + dy * dy);
}

static fsim_pos resource_pos(const fsim_resource *r) {
    fsim_pos p = {r->tx * TILE + TILE / 2, r->ty * TILE + TILE / 2};
    return p;
}

/* Distance from the character to a bounding box. */
static double box_distance(fsim_pos c, fsim_pos centre, int32_t r) {
    double bx = fmax(fmax(tiles(centre.x - r - c.x), 0.0), tiles(c.x - centre.x - r));
    double by = fmax(fmax(tiles(centre.y - r - c.y), 0.0), tiles(c.y - centre.y - r));
    return sqrt(bx * bx + by * by);
}

#define RESOURCE_BOX 25

/* `can_reach_entity`: box distance within reach, resource reach for ore. */
static int can_reach(const fsim_env *env, int32_t kind, int32_t index) {
    if (kind == 2) {
        fsim_pos p = resource_pos(&env->resources[index]);
        return box_distance(env->char_pos, p, RESOURCE_BOX) <= RESOURCE_REACH;
    }
    const fsim_entity *e = &env->entities[index];
    return box_distance(env->char_pos, e->pos, box_of(e->kind)) <= REACH_DISTANCE;
}

/* 16-way direction to a point, and the 8-way one, as the character turns
 * to face what it mines. */
static void face(fsim_env *env, fsim_pos target) {
    double dx = tiles(target.x - env->char_pos.x);
    double dy = tiles(target.y - env->char_pos.y);
    double angle = atan2(dx, -dy) * 180.0 / 3.141592653589793;
    if (angle < 0) angle += 360.0;
    int32_t d16 = (int32_t)floor(angle / 22.5 + 0.5) % 16;
    int32_t d8 = ((int32_t)floor(angle / 45.0 + 0.5) % 8) * 2;
    env->walk_set_dir = d16;
    env->walk_pub_dir = d16;
    env->char_dir8 = d8;
}

/* ------------------------------------------------------------------ entities */

/* The new entity's index, or -1 when the table is full. */
static int32_t new_entity(fsim_env *env, int32_t kind, fsim_pos pos, int32_t direction,
                          int32_t neutral) {
    if (env->entity_count >= FSIM_MAX_ENTITIES) return -1;
    int32_t i = env->entity_count++;
    env->entities_version++;
    fsim_entity *e = &env->entities[i];
    memset(e, 0, sizeof(*e));
    e->alive = 1;
    e->kind = kind;
    e->neutral = neutral;
    e->unit = ++env->next_unit;
    e->pos = pos;
    e->direction = direction;
    e->status = kind_of(kind)->status;
    e->pickup_target = e->drop_target = -1;
    for (int lane = 0; lane < 2; lane++) {
        /* 0 until rebuild_logistics sets it, so the first shape a belt gets
         * (built, or loaded with its items) re-places nothing. */
        e->lane_length[lane] = 0;
        e->lane_next[lane] = e->lane_side[lane] = -1;
        if (i * 2 + lane < FSIM_MAX_LANES) {
            env->seg_join[i * 2 + lane] = env->seg_head[i * 2 + lane] = -1;
            env->lane_pos[i * 2 + lane] = 0;
        }
    }
    if (kind == K_BELT) {
        /* Its merge timers start at the next rebuild_logistics ("segments"). */
        env->belts_changed = 1;
        int32_t tx = (int32_t)floordiv(pos.x, TILE), ty = (int32_t)floordiv(pos.y, TILE);
        e->built_tick = env->tick;
        e->seg_new = 1;
        for (int lane = 0; lane < 2; lane++) {
            e->delay[lane] = fsim_belt_delay(tx, ty, lane);
            if (e->delay[lane] < 0 && lane == 0) {
                if (env->belt_delay_missing++ == 0) {
                    env->belt_delay_missing_x = tx;
                    env->belt_delay_missing_y = ty;
                }
            }
        }
    }
    if (kind == K_INSERTER) {
        /* As built: empty fuel slot, but already burning a quarter of a wood,
         * the hand retracted on the pickup side. */
        e->burning = IT_WOOD;
        e->remaining = INSERTER_BUILT_FUEL;
        e->phase = INS_APPROACH;
        e->arm_w = (float)direction / 16.0f;
        e->arm_len = 0.7;          /* starting_distance */
        arm_draw(e);
        env->inserters[env->inserter_count++] = i;
    }
    return i;
}

/* Take inserter `index` out of the update list, keeping the others' order. */
static void inserter_unlist(fsim_env *env, int32_t index) {
    for (int32_t k = 0; k < env->inserter_count; k++) {
        if (env->inserters[k] != index) continue;
        for (int32_t j = k; j + 1 < env->inserter_count; j++)
            env->inserters[j] = env->inserters[j + 1];
        env->inserter_count--;
        return;
    }
}

static void destroy_entity(fsim_env *env, int32_t index) {
    fsim_entity *e = &env->entities[index];
    if (e->kind == K_INSERTER) {
        if (e->sleep_seq) env->sleeper_count--;
        else inserter_unlist(env, index);
        e->sleep_seq = 0;
    }
    e->alive = 0;
    env->entities_version++;
    if (e->kind == K_BELT) env->belts_changed = 1;
    if (e->kind != K_PILE) unit_destroyed(env, e->unit);   /* piles have tile handles */
    if (env->selected_kind == 1 && env->selected_index == index) env->selected_kind = 0;
}

static fsim_pos drop_position(const fsim_entity *d) {
    fsim_pos p = d->pos;
    switch (d->direction) {
    case 0: p.x += -128; p.y += -332; break;
    case 4: p.x += 332; p.y += -128; break;
    case 8: p.x += 128; p.y += 332; break;
    default: p.x += -332; p.y += 128; break;
    }
    return p;
}

/* The machine whose footprint holds `p`, or -1. */
static int32_t machine_at(const fsim_env *env, fsim_pos p, int32_t except) {
    for (int32_t i = 0; i < env->entity_count; i++) {
        const fsim_entity *e = &env->entities[i];
        if (!e->alive || i == except || !has_flag(e->kind, KF_MACHINE)) continue;
        int32_t half = kind_of(e->kind)->size * TILE / 2;
        if (p.x >= e->pos.x - half && p.x < e->pos.x + half &&
            p.y >= e->pos.y - half && p.y < e->pos.y + half)
            return i;
    }
    return -1;
}

static int pile_blocks(const fsim_env *env, fsim_pos p) {
    for (int32_t i = 0; i < env->entity_count; i++) {
        const fsim_entity *e = &env->entities[i];
        if (!e->alive || e->kind != K_PILE) continue;
        if (abs(e->pos.x - p.x) <= 2 * PILE_BOX && abs(e->pos.y - p.y) <= 2 * PILE_BOX) return 1;
    }
    return 0;
}

/* A drill takes ten ore from one of its four tiles, then moves to the next:
 * top-right, bottom-left, then usually bottom-right before top-left. The
 * last two swap for some drills in a larger patch -- it follows the engine's
 * internal entity order, which docs/evidence/sim-mechanics-m3-drills.json
 * could not reduce to a rule -- so parity compares ore under a drill as a
 * footprint total. The encoder cannot see the difference: it caps amounts
 * at 4,000. */
#define ORE_PER_TILE 10
static const int32_t DRILL_TILES[4][2] = {{0, -1}, {-1, 0}, {0, 0}, {-1, -1}};

static int32_t drill_resource(const fsim_env *env, fsim_entity *d) {
    int32_t cx = (int32_t)floordiv(d->pos.x, TILE), cy = (int32_t)floordiv(d->pos.y, TILE);
    for (int k = 0; k < 4; k++) {
        int32_t cursor = (d->mine_cursor + k) % 4;
        int32_t r = find_resource_at(env, cx + DRILL_TILES[cursor][0], cy + DRILL_TILES[cursor][1]);
        if (r >= 0) {
            if (k != 0) {
                d->mine_cursor = cursor;
                d->mine_count = 0;
            }
            return r;
        }
    }
    return -1;
}

/* An output dropped into a machine: into a chest, fuel into a burner's fuel
 * slot, anything else into a furnace's source. */
static int32_t machine_accepts(fsim_entity *m, int32_t item, int32_t count, int do_insert) {
    if (m->kind == K_CHEST) {
        if (do_insert) return chest_insert(m, item, count);
        return chest_room(m, item) >= count ? count : 0;
    }
    if (has_flag(m->kind, KF_BURNER) && is_fuel(item)) {
        if (do_insert) return slot_insert(&m->fuel, item, count, STACK_SIZE[item]);
        return slot_room(&m->fuel, item, STACK_SIZE[item]) >= count ? count : 0;
    }
    if (m->kind == K_FURNACE) {
        int32_t cap = item == IT_IRON_ORE ? FURNACE_SOURCE_CAP : STACK_SIZE[item];
        if (do_insert) return slot_insert(&m->source, item, count, cap);
        return slot_room(&m->source, item, cap) >= count ? count : 0;
    }
    return 0;
}

/* Refill the energy buffer from the fuel being burnt, taking a new item when
 * that runs out. */
static void burner_refill(fsim_env *env, fsim_entity *e) {
    double capacity = fsim_capacity(e->kind);
    for (;;) {
        double needed = capacity - e->energy;
        if (needed <= 0) break;
        if (e->remaining <= 0) {
            if (e->fuel.count > 0) {
                e->burning = e->fuel.item;
                slot_remove(&e->fuel, e->fuel.item, 1);
                e->remaining = fuel_value(e->burning);
                wake(env, (int32_t)(e - env->entities));
            } else {
                break;
            }
        }
        double take = needed < e->remaining ? needed : e->remaining;
        e->energy += take;
        e->remaining -= take;
    }
    if (e->remaining <= 0) {
        e->remaining = 0;
        e->burning = IT_NONE;
    }
}

/* Spend one tick of draw from the buffer; the fraction of a full tick done. */
static double burner_work(fsim_entity *e, double usage) {
    if (e->energy <= 0) return 0.0;
    if (e->energy >= usage) {
        e->energy -= usage;
        return 1.0;
    }
    double fraction = e->energy / usage;
    e->energy = 0;
    return fraction;
}

/* ------------------------------------------------------------------ belts
 *
 * Measured on the engine (docs/sim-logistics.md):
 *
 * - Each belt has one line per lane: lane 1 left of the direction of travel,
 *   lane 2 right, their centre lines 60/256 either side of the belt's.
 *   Positions are 1/256 tile from the belt's downstream edge; a straight lane
 *   is 256 long, a turn's outer lane 295 and its inner lane 106.
 * - Lines of consecutive belts chain: an item at p goes to p - 8 each tick,
 *   and below 0 it is on the next belt at p - 8 + that belt's lane length.
 *   The front item of a chain stops at exactly 0; behind it items keep 64
 *   apart and never move backwards.
 * - A belt pointing into the side of another feeds both its lanes onto the
 *   target's lane on that side, joining 188 from the target's downstream edge
 *   for the feed lane upstream and 67 for the other.
 * - Anything put on a lane (a drill's output, an inserter's drop, a
 *   sideloaded item) goes at the first place at or behind its target that is
 *   64 clear of every item ahead of it -- items behind are not consulted --
 *   and only if that is less than 64 behind the target. It then moves once
 *   at once, as if it had been there when the line moved: an output aimed at
 *   128 reads 120 on the tick it lands, or stays at 128 behind a stopped item
 *   at 64 (see lane_insert).
 *
 * A lane is named by a ref, entity index * 2 + lane (0 for lane 1). Chains of
 * lanes are found once per change of the entities (rebuild_logistics); what
 * moves, and in what order, is their segments ("segments" below).
 */

static fsim_lane *lane_of(fsim_env *env, int32_t ref) {
    return &env->entities[ref >> 1].lanes[ref & 1];
}

static int32_t lane_length_of(const fsim_env *env, int32_t ref) {
    return env->entities[ref >> 1].lane_length[ref & 1];
}

/* The unit vector of a 16-way direction: north, east, south, west. */
static void dir_vec(int32_t dir16, int32_t *ux, int32_t *uy) {
    switch (dir16) {
    case 0: *ux = 0; *uy = -1; break;
    case 4: *ux = 1; *uy = 0; break;
    case 8: *ux = 0; *uy = 1; break;
    default: *ux = -1; *uy = 0; break;
    }
}

static void lane_put(fsim_lane *lane, int32_t at, int32_t pos, int32_t item, int32_t id,
                     int32_t moved) {
    for (int32_t k = lane->count; k > at; k--) lane->items[k] = lane->items[k - 1];
    lane->items[at].pos = (int16_t)pos;
    lane->items[at].item = (uint8_t)item;
    lane->items[at].moved = (uint8_t)moved;
    lane->items[at].id = id;
    lane->count++;
}

static void lane_take(fsim_lane *lane, int32_t at) {
    for (int32_t k = at; k + 1 < lane->count; k++) lane->items[k] = lane->items[k + 1];
    lane->count--;
}

/* ------------------------------------------------------------------ segments
 *
 * FactorioRL docs/sim-logistics.md, "Third probe" and "Fourth probe": the
 * engine keeps a lane's line on each belt as its own object at first and
 * merges consecutive ones into one, a *segment*, later. Segments are what
 * moves (in activation order) and what a waiting inserter watches.
 *
 * - Merge. Every tile and lane has a delay d, 1 to 600 ticks, measured over
 *   the whole area scenes use and shipped as a table (fsim/data/belt-delay.*,
 *   fsim_set_belt_delay). A belt's lane starts a timer of d when it is built;
 *   when any timer on a lane chain runs out the whole chain merges, and every
 *   timer on it stops. A lane whose timer is not running starts it again
 *   when a belt is built next to it on its chain.
 * - Boundaries. An entity that works on a lane marks a boundary between the
 *   belts two and three downstream of its own (k+2|k+3): an inserter picking
 *   up (both lanes), an inserter dropping or a drill putting out (the lane it
 *   drops on); a feed sideloading onto belt k marks k+1|k+2 on the target
 *   lane. Every attached entity's boundary counts, whether it has worked yet
 *   or not (`logistics_belt_pickup`: pickups not yet made split at t=133).
 * - A chain an entity has put an item on or taken one off (`lane_dirty`)
 *   merges into its pieces between boundaries. A merged segment such an
 *   interaction happens on splits at its boundaries d (of its head's tile and
 *   lane) after the first one. Script inserts are not interactions.
 * - Order. Segments that hold items move each tick, last activated first: a
 *   segment that gets an item while it has none goes to the end of
 *   seg_order, which runs from the end. A segment moves the segment ahead of
 *   it, or the one it sideloads into, first when that one holds items and has
 *   not moved this tick -- so the second of two sideloads onto an empty
 *   target in one tick moves the first 8/256 on. Merged pieces keep the
 *   downstream piece's place when it held items, and otherwise go first in
 *   seg_order (move last).
 *
 * Not measured, and chosen: the place in seg_order of pieces a split, a
 * removed belt or a loaded state leaves holding items (first, like a merge);
 * what rotating a belt does to its timers (nothing); closed loops (they move
 * as a whole, before the segments, as before).
 */

static int32_t DELAY_COUNT = 0;
static int32_t *DELAY_RECTS = NULL;
static int64_t *DELAY_OFFSETS = NULL;
static uint16_t *DELAY_VALUES = NULL;

int32_t fsim_set_belt_delay(int32_t count, const int32_t *rects, const uint16_t *values) {
    free(DELAY_RECTS);
    free(DELAY_OFFSETS);
    free(DELAY_VALUES);
    DELAY_RECTS = NULL;
    DELAY_OFFSETS = NULL;
    DELAY_VALUES = NULL;
    DELAY_COUNT = 0;
    if (count <= 0) return 0;
    int64_t total = 0;
    for (int32_t r = 0; r < count; r++) {
        int64_t w = rects[4 * r + 2] - rects[4 * r], h = rects[4 * r + 3] - rects[4 * r + 1];
        if (w <= 0 || h <= 0) return -1;
        total += 2 * w * h;
    }
    DELAY_RECTS = (int32_t *)malloc(sizeof(int32_t) * 4 * (size_t)count);
    DELAY_OFFSETS = (int64_t *)malloc(sizeof(int64_t) * (size_t)count);
    DELAY_VALUES = (uint16_t *)malloc(sizeof(uint16_t) * (size_t)total);
    if (!DELAY_RECTS || !DELAY_OFFSETS || !DELAY_VALUES) {
        fsim_set_belt_delay(0, NULL, NULL);
        return -1;
    }
    memcpy(DELAY_RECTS, rects, sizeof(int32_t) * 4 * (size_t)count);
    memcpy(DELAY_VALUES, values, sizeof(uint16_t) * (size_t)total);
    int64_t at = 0;
    for (int32_t r = 0; r < count; r++) {
        DELAY_OFFSETS[r] = at;
        at += 2 * (int64_t)(rects[4 * r + 2] - rects[4 * r]) * (rects[4 * r + 3] - rects[4 * r + 1]);
    }
    DELAY_COUNT = count;
    return 0;
}

int32_t fsim_belt_delay(int32_t tx, int32_t ty, int32_t lane) {
    if (lane < 0 || lane > 1) return -1;
    for (int32_t r = 0; r < DELAY_COUNT; r++) {
        const int32_t *b = &DELAY_RECTS[4 * r];
        if (tx < b[0] || tx >= b[2] || ty < b[1] || ty >= b[3]) continue;
        int64_t w = b[2] - b[0], h = b[3] - b[1];
        uint16_t v = DELAY_VALUES[DELAY_OFFSETS[r] + lane * w * h + (ty - b[1]) * w + (tx - b[0])];
        return v ? (int32_t)v : -1;
    }
    return -1;
}

static void seg_timer(fsim_env *env, int64_t at) {
    if (at > 0 && (env->seg_next_timer == 0 || at < env->seg_next_timer)) env->seg_next_timer = at;
}

/* The lanes of the segment headed by `head`, front first. */
static int32_t seg_lanes(const fsim_env *env, int32_t head, const int32_t **refs) {
    int32_t c = env->lane_chain[head];
    const int32_t *all = &env->chain_lanes[env->chain_first[c]];
    int32_t size = env->chain_size[c];
    int32_t p = env->lane_pos[head], n = 1;
    while (p + n < size && env->seg_join[all[p + n]] == all[p + n - 1]) n++;
    *refs = all + p;
    return n;
}

static int seg_empty(const fsim_env *env, int32_t head) {
    const int32_t *refs;
    int32_t n = seg_lanes(env, head, &refs);
    for (int32_t k = 0; k < n; k++)
        if (env->entities[refs[k] >> 1].lanes[refs[k] & 1].count) return 0;
    return 1;
}

static void seg_compact(fsim_env *env) {
    int32_t out = 0;
    for (int32_t i = 0; i < env->seg_count; i++)
        if (env->seg_order[i] >= 0) env->seg_order[out++] = env->seg_order[i];
    env->seg_count = out;
}

/* Activated: to the end of the order, to move first. */
static void seg_list_add(fsim_env *env, int32_t head) {
    env->seg_sleep[head] = 0;
    if (env->seg_listed[head]) return;
    if (env->seg_count >= 2048) seg_compact(env);
    if (env->seg_count >= 2048) return;
    env->seg_order[env->seg_count++] = head;
    env->seg_listed[head] = 1;
}

/* To the front of the order, to move last. */
static void seg_list_front(fsim_env *env, int32_t head) {
    env->seg_sleep[head] = 0;
    if (env->seg_listed[head]) return;
    seg_compact(env);
    if (env->seg_count >= 2048) return;
    for (int32_t i = env->seg_count; i > 0; i--) env->seg_order[i] = env->seg_order[i - 1];
    env->seg_order[0] = head;
    env->seg_count++;
    env->seg_listed[head] = 1;
}

static void seg_list_remove(fsim_env *env, int32_t head) {
    if (!env->seg_listed[head]) return;
    env->seg_listed[head] = 0;
    for (int32_t i = env->seg_count - 1; i >= 0; i--)
        if (env->seg_order[i] == head) {
            env->seg_order[i] = -1;
            return;
        }
}

/* An item went onto lane `ref`: its segment is awake. */
static void seg_touch(fsim_env *env, int32_t ref) {
    int32_t h = env->seg_head[ref];
    if (h >= 0) seg_list_add(env, h);
}

/* An entity took an item off lane `ref`: its segment rests when empty, and
 * wakes if it was asleep (FactorioRL probe_logistics4 `sleep`: an inserter's
 * pickup from a stopped line moves everything behind the next tick). */
static void seg_check(fsim_env *env, int32_t ref) {
    int32_t h = env->seg_head[ref];
    if (h < 0) return;
    if (seg_empty(env, h)) {
        seg_list_remove(env, h);
        env->seg_sleep[h] = 0;
    } else {
        seg_list_add(env, h);
    }
}

/* An entity put an item on lane `ref` or took one off, in the engine's tick
 * `when`: a drop, a drill's output or a sideload goes on a tick before the
 * simulator places it, already moved once (lane_insert), a pickup on the same
 * tick (`logistics_belt_pickup`: drops read at t=47 split the line at
 * 46 + 87; `logistics_smelting_chain`: the drill's first ore, read at 243,
 * at 242 + 492). */
static void seg_interact(fsim_env *env, int32_t ref, int64_t when) {
    env->lane_dirty[ref] = 1;
    int32_t h = env->seg_head[ref];
    if (h < 0 || env->seg_split_at[h]) return;
    const int32_t *refs;
    int32_t n = seg_lanes(env, h, &refs);
    for (int32_t k = 1; k < n; k++) {
        if (!env->seg_cut[refs[k]]) continue;
        int32_t d = env->entities[h >> 1].delay[h & 1];
        if (d < 1) return;
        env->seg_split_at[h] = when + d;
        seg_timer(env, env->seg_split_at[h]);
        return;
    }
}

/* Heads of chain `c`'s lanes from its joins. */
static void seg_heads(fsim_env *env, int32_t c) {
    int32_t size = env->chain_size[c];
    const int32_t *refs = &env->chain_lanes[env->chain_first[c]];
    if (size < 0) {
        for (int32_t k = 0; k < -size; k++) env->seg_head[refs[k]] = -1;
        return;
    }
    int32_t head = -1;
    for (int32_t k = 0; k < size; k++) {
        if (k == 0 || env->seg_join[refs[k]] != refs[k - 1]) head = refs[k];
        env->seg_head[refs[k]] = head;
    }
}

/* After chain `c` merged: an inserter asleep on one of its lanes wakes if its
 * segment now holds an item. */
static void seg_wake_watchers(fsim_env *env, int32_t c) {
    if (env->belt_sleepers == 0) return;
    for (int32_t k = 0; k < env->inserter_count; k++) {
        fsim_entity *s = &env->entities[env->inserters[k]];
        if (!s->belt_asleep || s->pickup_target < 0) continue;
        int32_t b = s->pickup_target;
        for (int lane = 0; lane < 2; lane++) {
            int32_t h = env->seg_head[b * 2 + lane];
            if (h < 0 || env->lane_chain[b * 2 + lane] != c || seg_empty(env, h)) continue;
            /* Before the inserters run: it acts this tick (probe_logistics
             * `flow2`: lane 1 merges at t=76 and the inserter refills then). */
            s->belt_asleep = 0;
            env->belt_sleepers--;
            break;
        }
    }
}

/* Chain `c` merges: whole, or into its pieces between boundaries when an
 * entity has worked on it. */
static void seg_merge_chain(fsim_env *env, int32_t c) {
    int32_t size = env->chain_size[c];
    const int32_t *refs = &env->chain_lanes[env->chain_first[c]];
    int32_t n = size < 0 ? -size : size;
    for (int32_t k = 0; k < n; k++) env->entities[refs[k] >> 1].merge_at[refs[k] & 1] = 0;
    if (size <= 1) return;
    int dirty = 0;
    for (int32_t k = 0; k < size && !dirty; k++) dirty = env->lane_dirty[refs[k]];
    int32_t old_head[FSIM_MAX_LANES];
    uint8_t awake[FSIM_MAX_LANES];
    for (int32_t k = 0; k < size; k++) {
        old_head[k] = env->seg_head[refs[k]];
        awake[k] = old_head[k] == refs[k] && env->seg_listed[refs[k]];
    }
    for (int32_t k = 1; k < size; k++)
        if (!(dirty && env->seg_cut[refs[k]])) env->seg_join[refs[k]] = refs[k - 1];
    seg_heads(env, c);
    for (int32_t k = 0; k < size; k++) {
        int32_t h = refs[k];
        if (env->seg_head[h] == h || old_head[k] != h) continue;
        seg_list_remove(env, h);            /* was a head, now merged into another */
        env->seg_split_at[h] = 0;
        env->seg_sleep[h] = 0;
    }
    for (int32_t k = 0; k < size; k++) {
        int32_t h = refs[k];
        if (env->seg_head[h] != h || env->seg_listed[h]) continue;
        int any = 0;
        for (int32_t j = k + 1; j < size && env->seg_head[refs[j]] == h; j++) any |= awake[j];
        if (any) seg_list_front(env, h);
        else env->seg_sleep[h] = !seg_empty(env, h);
    }
    seg_wake_watchers(env, c);
}

/* Segment `head` splits at its boundaries. */
static void seg_split(fsim_env *env, int32_t head) {
    const int32_t *refs;
    int32_t n = seg_lanes(env, head, &refs);
    env->seg_split_at[head] = 0;
    int32_t pieces[FSIM_MAX_LANES];
    int32_t np = 0;
    for (int32_t k = 1; k < n; k++)
        if (env->seg_cut[refs[k]]) {
            env->seg_join[refs[k]] = -1;
            pieces[np++] = refs[k];
        }
    if (!np) return;
    int awake = env->seg_listed[head];
    seg_heads(env, env->lane_chain[head]);
    if (seg_empty(env, head)) {
        seg_list_remove(env, head);
        env->seg_sleep[head] = 0;
    }
    for (int32_t k = 0; k < np; k++) {
        if (seg_empty(env, pieces[k])) continue;
        if (awake) seg_list_front(env, pieces[k]);
        else env->seg_sleep[pieces[k]] = 1;
    }
}

/* Merges and splits due by now. */
static void seg_timers(fsim_env *env) {
    if (!env->seg_next_timer || env->tick < env->seg_next_timer) return;
    int64_t now = env->tick;
    for (int32_t c = 0; c < env->chain_count; c++) {
        int32_t size = env->chain_size[c];
        const int32_t *refs = &env->chain_lanes[env->chain_first[c]];
        int32_t n = size < 0 ? -size : size;
        for (int32_t k = 0; k < n; k++) {
            int64_t m = env->entities[refs[k] >> 1].merge_at[refs[k] & 1];
            if (m && m <= now) {
                seg_merge_chain(env, c);
                break;
            }
        }
    }
    for (int32_t c = 0; c < env->chain_count; c++) {
        int32_t size = env->chain_size[c];
        const int32_t *refs = &env->chain_lanes[env->chain_first[c]];
        for (int32_t k = 0; k < size; k++) {
            int32_t h = refs[k];
            if (env->seg_head[h] == h && env->seg_split_at[h] && env->seg_split_at[h] <= now)
                seg_split(env, h);
        }
    }
    env->seg_next_timer = 0;
    for (int32_t c = 0; c < env->chain_count; c++) {
        int32_t size = env->chain_size[c];
        const int32_t *refs = &env->chain_lanes[env->chain_first[c]];
        int32_t n = size < 0 ? -size : size;
        for (int32_t k = 0; k < n; k++) {
            seg_timer(env, env->entities[refs[k] >> 1].merge_at[refs[k] & 1]);
            if (env->seg_head[refs[k]] == refs[k]) seg_timer(env, env->seg_split_at[refs[k]]);
        }
    }
}

int32_t fsim_belt_segment(fsim_env *env, int32_t index, int32_t lane) {
    if (index < 0 || index >= env->entity_count || lane < 0 || lane > 1) return -1;
    if (!env->entities[index].alive || env->entities[index].kind != K_BELT) return -1;
    if (env->logistics_version != env->entities_version) fsim_refresh(env);
    return env->seg_head[index * 2 + lane];
}

/* Put `item` on lane `ref` aimed at `target`, judged from `from`. Returns 1
 * when it went on. Items at or ahead of `from` (in the lane's position now)
 * are ahead of it; it goes at the first place at or behind `target` that is
 * 64 clear of every item ahead, and only if that is less than 64 behind
 * `from`. `from` is `target` for a drop or a script insert; for a sideload
 * it is the entry point, and `target` the entry plus the part of the feed
 * item's move not yet made (FactorioRL probe_logistics4 `accept`: 1,296
 * sideloads onto a lane with an item passing the entry, every phase; an item
 * exactly at the entry refuses the sideload, one a step behind it is not
 * consulted). `exact`: only at `target` itself (insert_at_back, whose target
 * is the lane's upstream edge). The placement is measured for drops at 128
 * (the drill on a free, a stopped and a just-extended line, and `accept`'s
 * 81 inserter drops past a moving item) and for sideloads; `exact` is the
 * script call's behaviour and fits the probe's feed timing. */
static int lane_insert_from(fsim_env *env, int32_t ref, int32_t from, int32_t target, int32_t item,
                            int32_t id, int exact) {
    fsim_lane *lane = lane_of(env, ref);
    int32_t length = lane_length_of(env, ref);
    int32_t next = env->entities[ref >> 1].lane_next[ref & 1];
    /* The nearest item ahead, in this lane's coordinates: the back of the
     * lane it runs into, then this lane's own items up to the insertion. */
    int32_t q = from, ahead = INT32_MIN;
    if (next >= 0) {
        const fsim_lane *down = lane_of(env, next);
        if (down->count > 0) {
            ahead = down->items[down->count - 1].pos - lane_length_of(env, next);
            if (ahead + BELT_GAP > q) q = ahead + BELT_GAP;
        }
    }
    int32_t at = 0;
    for (; at < lane->count; at++) {
        int32_t p = lane->items[at].pos;
        if (p > q) break;
        ahead = p;
        if (p + BELT_GAP > q) q = p + BELT_GAP;
    }
    if (q - from >= BELT_GAP) return 0;
    if (q < target) q = target;
    if (exact && q != target) return 0;
    if (lane->count >= FSIM_LANE_ITEMS) return 0;
    /* The move it makes at once, never past the lane's downstream edge: a
     * script insert within 8 of the edge reads 0 on its own belt (`accept`,
     * `drop_c768`..`c775`), whatever lies beyond. */
    int32_t moved_to = q - BELT_SPEED;
    if (ahead != INT32_MIN && ahead + BELT_GAP > moved_to) moved_to = ahead + BELT_GAP;
    if (moved_to < 0) moved_to = 0;
    if (moved_to > q) moved_to = q;
    if (moved_to >= length) return 0;
    lane_put(lane, at, moved_to, item, id, moved_to != q);
    seg_touch(env, ref);
    belt_line_added(env, ref);
    return 1;
}

static int lane_insert(fsim_env *env, int32_t ref, int32_t target, int32_t item, int32_t id,
                       int exact) {
    return lane_insert_from(env, ref, target, target, item, id, exact);
}

static int32_t new_item_id(fsim_env *env) { return ++env->next_item_id; }

/* Belt `index` under point `p`: which lane's half holds it, and the target
 * position, the point's distance from the belt's downstream edge. On the
 * centre line it is lane 2 (measured once: a belt running south, away from
 * the inserter).
 *
 * On a turn (FactorioRL tools/probe_logistics2.py, `tdrop_*` and `tdrill_*`:
 * inserters and drills dropping from every free side of a right and a left
 * turn) the item goes on the inner lane when the point is nearer the turn's
 * inner corner than the opposite corner, else on the outer lane, and always
 * at the middle of that lane: 53 of 106, 147 of 295. Every drop point an
 * inserter or a drill has lies 51 or 52/256 off the tile centre along one
 * axis, so none is on the diagonal between the two. */
static int belt_drop_target(const fsim_env *env, int32_t index, fsim_pos p, int32_t *ref,
                            int32_t *target) {
    const fsim_entity *b = &env->entities[index];
    int32_t ux, uy;
    dir_vec(b->direction, &ux, &uy);
    int32_t dx = p.x - b->pos.x, dy = p.y - b->pos.y;
    if (b->shape != BELT_STRAIGHT) {
        /* The inner corner lies ahead and on the side the turn is fed from:
         * the right, (-uy, ux), for a right turn. */
        int32_t side = b->shape == BELT_RIGHT ? 1 : -1;
        int32_t kx = ux - side * uy, ky = uy + side * ux;
        int inner = dx * kx + dy * ky > 0;
        int32_t lane = b->shape == BELT_RIGHT ? (inner ? 1 : 0) : (inner ? 0 : 1);
        *ref = index * 2 + lane;
        *target = b->lane_length[lane] / 2;
        return 1;
    }
    int32_t along = dx * ux + dy * uy;
    int32_t lateral = -dx * uy + dy * ux;   /* towards the right of travel */
    *ref = index * 2 + (lateral < 0 ? 0 : 1);
    *target = TILE / 2 - along;
    return 1;
}

static int belt_drop(fsim_env *env, int32_t index, fsim_pos p, int32_t item) {
    int32_t ref, target;
    if (!belt_drop_target(env, index, p, &ref, &target)) return 0;
    if (!lane_insert(env, ref, target, item, new_item_id(env), 0)) return 0;
    seg_interact(env, ref, env->tick - 1);
    return 1;
}

/* The belt whose tile holds `p`, or -1. */
static int32_t belt_at(const fsim_env *env, fsim_pos p) {
    int64_t tx = floordiv(p.x, TILE), ty = floordiv(p.y, TILE);
    for (int32_t i = 0; i < env->entity_count; i++) {
        const fsim_entity *e = &env->entities[i];
        if (e->alive && e->kind == K_BELT && floordiv(e->pos.x, TILE) == tx &&
            floordiv(e->pos.y, TILE) == ty)
            return i;
    }
    return -1;
}

static void seg_wake_move(fsim_env *env, int32_t head);

/* One tick of segment `head`: what it runs into first, then its own lanes
 * front first, as one line. */
static void move_segment(fsim_env *env, int32_t head) {
    int64_t now = env->tick;
    env->seg_moved[head] = now;
    int32_t c = env->lane_chain[head];
    const int32_t *all = &env->chain_lanes[env->chain_first[c]];
    int32_t p = env->lane_pos[head];
    const fsim_entity *front = &env->entities[head >> 1];
    int32_t side = p == 0 ? front->lane_side[head & 1] : -1;
    int32_t entry = front->lane_entry[head & 1];
    int32_t down = p > 0 ? all[p - 1] : side;
    if (down >= 0) {
        int32_t dh = env->seg_head[down];
        if (dh >= 0 && env->seg_listed[dh] && env->seg_moved[dh] != now) move_segment(env, dh);
    }
    /* Chain coordinates, from the chain's front: this segment's downstream
     * edge, and the back item ahead of it after its move. */
    int32_t offset = 0;
    for (int32_t k = 0; k < p; k++) offset += lane_length_of(env, all[k]);
    int has_ahead = 0;
    int32_t ahead = 0;
    for (int32_t k = p - 1, edge = offset; k >= 0; k--) {
        edge -= lane_length_of(env, all[k]);
        const fsim_lane *l = lane_of(env, all[k]);
        if (l->count) {
            has_ahead = 1;
            ahead = edge + l->items[l->count - 1].pos;
            break;
        }
    }
    const int32_t *refs;
    int32_t n = seg_lanes(env, head, &refs);
    int moved = 0;
    for (int32_t k = 0; k < n; k++) {
        fsim_lane *lane = lane_of(env, refs[k]);
        int32_t i = 0;
        while (i < lane->count) {
            fsim_belt_item *it = &lane->items[i];
            int32_t old = offset + it->pos;
            int32_t want = old - BELT_SPEED;
            if (has_ahead) {
                if (ahead + BELT_GAP > want) want = ahead + BELT_GAP;
            } else if (want < 0) {
                /* The front of the chain at its end: across into the side of
                 * another belt if it sideloads, else it stops at 0. */
                if (side >= 0 && lane_insert_from(env, side, entry, entry + old, it->item, it->id, 0)) {
                    seg_interact(env, side, now - 1);
                    lane_take(lane, i);
                    moved = 1;
                    continue;
                }
                want = 0;
            }
            if (want > old) want = old;
            has_ahead = 1;
            ahead = want;
            if (want < offset) {
                /* Onto the lane ahead, behind everything already there: in
                 * this segment, or across into the next one. */
                int32_t dref = k > 0 ? refs[k - 1] : all[p - 1];
                fsim_lane *dl = lane_of(env, dref);
                int32_t down_length = lane_length_of(env, dref);
                if (dl->count < FSIM_LANE_ITEMS) {
                    fsim_belt_item moving = *it;
                    lane_take(lane, i);
                    lane_put(dl, dl->count, want - (offset - down_length), moving.item, moving.id,
                             1);
                    moved = 1;
                    if (k == 0) {
                        seg_touch(env, dref);
                        belt_line_added(env, dref);
                    }
                    continue;
                }
                want = offset;      /* no room: cannot happen at 64 apart */
                ahead = want;
            }
            it->moved = (uint8_t)(want != old);
            moved |= want != old;
            it->pos = (int16_t)(want - offset);
            i++;
        }
        offset += lane_length_of(env, refs[k]);
    }
    /* Nothing could move: asleep until an item goes on or comes off, the
     * belts change, or what it runs into moves (FactorioRL probe_logistics4
     * `sleep`, and `ins_*` of the third probe: a stopped target the first of
     * two sideloads wakes, the second moves). */
    if (!moved) {
        seg_list_remove(env, head);
        env->seg_sleep[head] = !seg_empty(env, head);
    }
    /* What runs into it, asleep, wakes and moves now: the lane behind on the
     * chain, and the chains sideloading onto it (a release moves the whole
     * stopped line on the same tick, `sleep`). */
    if (p + n < env->chain_size[c]) seg_wake_move(env, env->seg_head[all[p + n]]);
    for (int32_t k = 0; k < n; k++)
        for (int32_t f = env->side_first[refs[k]]; f >= 0; f = env->side_link[f])
            seg_wake_move(env, env->seg_head[f]);
}

/* A segment asleep wakes, to the end of the order, and moves this tick. */
static void seg_wake_move(fsim_env *env, int32_t head) {
    if (head < 0 || !env->seg_sleep[head]) return;
    seg_list_add(env, head);
    if (env->seg_moved[head] != env->tick) move_segment(env, head);
}

typedef struct {
    int32_t pos;                /* chain coordinate */
    int32_t move;
    fsim_belt_item it;
} ring_item;

/* One tick of a closed loop of lanes. Not measured: every item moves 8 unless
 * that brings it within 64 of the item ahead after that one's own move -- the
 * largest movement that keeps the straight-line rule, found by relaxation. */
static void update_ring(fsim_env *env, const int32_t *refs, int32_t size) {
    int32_t total = 0, length = 0;
    for (int32_t k = 0; k < size; k++) total += lane_of(env, refs[k])->count;
    if (total == 0) return;
    ring_item *items = (ring_item *)malloc(sizeof(ring_item) * (size_t)total);
    int32_t *offsets = (int32_t *)malloc(sizeof(int32_t) * (size_t)size);
    if (!items || !offsets) {
        free(items);
        free(offsets);
        return;
    }
    int32_t n = 0;
    for (int32_t k = 0; k < size; k++) {
        fsim_lane *lane = lane_of(env, refs[k]);
        offsets[k] = length;
        for (int32_t i = 0; i < lane->count; i++) {
            items[n].pos = length + lane->items[i].pos;
            items[n].move = BELT_SPEED;
            items[n].it = lane->items[i];
            n++;
        }
        length += lane_length_of(env, refs[k]);
        lane->count = 0;
    }
    for (int changed = 1; changed;) {
        changed = 0;
        for (int32_t i = 0; i < n; i++) {
            int32_t a = i > 0 ? i - 1 : n - 1;
            int32_t gap = items[i].pos - items[a].pos + (i > 0 ? 0 : length);
            int32_t move = n > 1 ? gap + items[a].move - BELT_GAP : BELT_SPEED;
            if (move > BELT_SPEED) move = BELT_SPEED;
            if (move < 0) move = 0;
            if (move < items[i].move) {
                items[i].move = move;
                changed = 1;
            }
        }
    }
    for (int32_t i = 0; i < n; i++) {
        int32_t pos = items[i].pos - items[i].move;
        if (pos < 0) pos += length;
        int32_t k = size - 1;
        while (k > 0 && offsets[k] > pos) k--;
        fsim_lane *lane = lane_of(env, refs[k]);
        int32_t at = lane->count;
        while (at > 0 && lane->items[at - 1].pos > pos - offsets[k]) at--;
        if (lane->count < FSIM_LANE_ITEMS)
            lane_put(lane, at, pos - offsets[k], items[i].it.item, items[i].it.id,
                     items[i].move != 0);
    }
    free(items);
    free(offsets);
}

/* One tick of the belts: merges and splits due, closed loops, then the
 * segments holding items, last activated first (see "segments"). */
static void update_belts(fsim_env *env) {
    seg_timers(env);
    for (int32_t c = 0; c < env->chain_count; c++) {
        int32_t size = env->chain_size[c];
        if (size < 0) update_ring(env, &env->chain_lanes[env->chain_first[c]], -size);
    }
    seg_compact(env);
    int32_t n = env->seg_count;
    if (n == 0) return;
    int32_t order[2048];
    memcpy(order, env->seg_order, sizeof(int32_t) * (size_t)n);
    env->belt_phase = 1;
    for (int32_t i = n - 1; i >= 0; i--) {
        int32_t h = order[i];
        if (env->seg_listed[h] && env->seg_head[h] == h && env->seg_moved[h] != env->tick)
            move_segment(env, h);
    }
    env->belt_phase = 0;
}

typedef struct {
    int64_t key;
    int32_t index;
} tile_entry;

static int64_t tile_key(int64_t tx, int64_t ty) { return ty * 4294967296LL + tx; }

static int tile_cmp(const void *pa, const void *pb) {
    const tile_entry *a = pa, *b = pb;
    if (a->key != b->key) return a->key < b->key ? -1 : 1;
    return a->index < b->index ? -1 : (a->index > b->index);
}

/* The belt on tile (tx, ty) facing `dir16`, or -1 (any facing: dir16 < 0). */
static int32_t belt_on(const fsim_env *env, const tile_entry *belts, int32_t n, int64_t tx,
                       int64_t ty, int32_t dir16) {
    int64_t key = tile_key(tx, ty);
    int32_t lo = 0, hi = n - 1;
    while (lo <= hi) {
        int32_t mid = (lo + hi) / 2;
        if (belts[mid].key < key) lo = mid + 1;
        else hi = mid - 1;
    }
    if (lo >= n || belts[lo].key != key) return -1;
    int32_t index = belts[lo].index;
    if (dir16 >= 0 && env->entities[index].direction != dir16) return -1;
    return index;
}

/* The direction of a unit vector. */
static int32_t vec_dir(int32_t ux, int32_t uy) {
    if (uy < 0) return 0;
    if (ux > 0) return 4;
    if (uy > 0) return 8;
    return 12;
}

/* The entity at an inserter's pickup or drop point: a machine, else a belt. */
static int32_t point_target(const fsim_env *env, fsim_pos p, int32_t except) {
    int32_t m = machine_at(env, p, except);
    return m >= 0 ? m : belt_at(env, p);
}

static fsim_pos inserter_point(const fsim_entity *s, int32_t distance) {
    int32_t ux, uy;
    dir_vec(s->direction, &ux, &uy);
    fsim_pos p = {s->pos.x + ux * distance, s->pos.y + uy * distance};
    return p;
}

static int drill_blocked(const fsim_entity *d);
static int32_t drill_block_signature(const fsim_env *env, int32_t belt, fsim_pos drop);
static void unblock_drills(fsim_env *env, int32_t index, fsim_pos where);

/* Mark the boundary `steps` lanes downstream of lane `ref`: between that
 * lane and the one after it ("segments"). */
static void seg_mark_cut(fsim_env *env, int32_t ref, int32_t steps) {
    for (int32_t i = 0; i < steps && ref >= 0; i++)
        ref = env->entities[ref >> 1].lane_next[ref & 1];
    if (ref >= 0 && env->entities[ref >> 1].lane_next[ref & 1] >= 0) env->seg_cut[ref] = 1;
}

/* Segments after the links and chains changed (rebuild_logistics): joins
 * whose link is gone are dropped, new belts start their merge timers and
 * wake their neighbours', boundaries follow the entities, and every
 * segment holding items is in the activation order. `pred` is each lane's
 * feeder on its chain. */
static void rebuild_segments(fsim_env *env, const int32_t *pred) {
    int32_t lanes = env->entity_count * 2;
    if (lanes > FSIM_MAX_LANES) lanes = FSIM_MAX_LANES;
    for (int32_t c = 0; c < env->chain_count; c++) {
        int32_t size = env->chain_size[c] < 0 ? -env->chain_size[c] : env->chain_size[c];
        for (int32_t j = 0; j < size; j++) env->lane_pos[env->chain_lanes[env->chain_first[c] + j]] = j;
    }
    for (int32_t r = 0; r < lanes; r++) {
        const fsim_entity *e = &env->entities[r >> 1];
        env->seg_cut[r] = 0;
        if (!e->alive || e->kind != K_BELT) {
            env->seg_join[r] = env->seg_head[r] = -1;
            if (env->seg_listed[r]) seg_list_remove(env, r);
            env->seg_split_at[r] = 0;
            continue;
        }
        if (env->seg_join[r] >= 0 && env->seg_join[r] != e->lane_next[r & 1]) env->seg_join[r] = -1;
    }
    /* New belts: their own timers, and a neighbour's whose timer had stopped. */
    for (int32_t i = 0; i < env->entity_count && i * 2 + 1 < FSIM_MAX_LANES; i++) {
        fsim_entity *b = &env->entities[i];
        if (!b->alive || b->kind != K_BELT || !b->seg_new) continue;
        for (int lane = 0; lane < 2; lane++) {
            if (b->delay[lane] >= 1) b->merge_at[lane] = b->built_tick + b->delay[lane];
            int32_t near[2] = {b->lane_next[lane], pred[i * 2 + lane]};
            for (int k = 0; k < 2; k++) {
                int32_t r = near[k];
                if (r < 0) continue;
                fsim_entity *nb = &env->entities[r >> 1];
                if (nb->seg_new || nb->merge_at[r & 1] || nb->delay[r & 1] < 1) continue;
                nb->merge_at[r & 1] = b->built_tick + nb->delay[r & 1];
            }
        }
    }
    for (int32_t i = 0; i < env->entity_count; i++) env->entities[i].seg_new = 0;
    /* Boundaries. */
    for (int32_t i = 0; i < env->entity_count; i++) {
        const fsim_entity *e = &env->entities[i];
        if (!e->alive) continue;
        if (e->kind == K_INSERTER) {
            if (e->pickup_target >= 0 && env->entities[e->pickup_target].kind == K_BELT)
                for (int lane = 0; lane < 2; lane++) seg_mark_cut(env, e->pickup_target * 2 + lane, 2);
            if (e->drop_target >= 0 && env->entities[e->drop_target].kind == K_BELT) {
                int32_t ref, target;
                if (belt_drop_target(env, e->drop_target, inserter_point(e, -INSERTER_DROP), &ref,
                                     &target))
                    seg_mark_cut(env, ref, 2);
            }
        } else if (e->kind == K_DRILL) {
            fsim_pos drop = drop_position(e);
            if (machine_at(env, drop, i) >= 0) continue;
            int32_t belt = belt_at(env, drop), ref, target;
            if (belt >= 0 && belt_drop_target(env, belt, drop, &ref, &target))
                seg_mark_cut(env, ref, 2);
        } else if (e->kind == K_BELT) {
            for (int lane = 0; lane < 2; lane++)
                if (e->lane_side[lane] >= 0) seg_mark_cut(env, e->lane_side[lane], 1);
        }
    }
    /* The chain fronts sideloading onto each lane. */
    for (int32_t r = 0; r < lanes; r++) env->side_first[r] = env->side_link[r] = -1;
    for (int32_t r = lanes - 1; r >= 0; r--) {
        const fsim_entity *e = &env->entities[r >> 1];
        if (!e->alive || e->kind != K_BELT || e->lane_side[r & 1] < 0) continue;
        int32_t t = e->lane_side[r & 1];
        env->side_link[r] = env->side_first[t];
        env->side_first[t] = r;
    }
    /* Heads, and the activation order: what no longer heads a segment or
     * holds nothing leaves it; a belt built, removed or turned wakes every
     * segment asleep; what holds items and is neither in it nor asleep (a
     * piece a removed belt left) goes first. */
    for (int32_t c = 0; c < env->chain_count; c++) seg_heads(env, c);
    for (int32_t r = 0; r < lanes; r++) {
        if (env->seg_head[r] == r) continue;
        if (env->seg_listed[r]) seg_list_remove(env, r);
        env->seg_split_at[r] = 0;
        env->seg_sleep[r] = 0;
    }
    for (int32_t r = 0; r < lanes; r++)
        if (env->seg_head[r] == r && seg_empty(env, r)) {
            seg_list_remove(env, r);
            env->seg_sleep[r] = 0;
        }
    if (env->belts_changed)
        for (int32_t r = 0; r < lanes; r++)
            if (env->seg_head[r] == r && env->seg_sleep[r]) seg_list_add(env, r);
    env->belts_changed = 0;
    for (int32_t r = 0; r < lanes; r++)
        if (env->seg_head[r] == r && !env->seg_listed[r] && !env->seg_sleep[r] && !seg_empty(env, r))
            seg_list_front(env, r);
    env->seg_next_timer = 0;
    for (int32_t r = 0; r < lanes; r++) {
        const fsim_entity *e = &env->entities[r >> 1];
        if (!e->alive || e->kind != K_BELT) continue;
        seg_timer(env, e->merge_at[r & 1]);
        if (env->seg_head[r] == r) seg_timer(env, env->seg_split_at[r]);
    }
}

/* Belt shapes, lane links, the chains and the order they run in, and every
 * inserter's pickup and drop target. */
static void rebuild_logistics(fsim_env *env) {
    tile_entry belts[FSIM_MAX_ENTITIES];
    int32_t n = 0;
    for (int32_t i = 0; i < env->entity_count; i++) {
        const fsim_entity *e = &env->entities[i];
        if (!e->alive) continue;
        if (e->kind == K_BELT) {
            belts[n].key = tile_key(floordiv(e->pos.x, TILE), floordiv(e->pos.y, TILE));
            belts[n].index = i;
            n++;
        }
    }
    qsort(belts, (size_t)n, sizeof(belts[0]), tile_cmp);

    /* Shapes: a belt fed from behind is straight; fed from exactly one side
     * and not from behind, it turns (a right turn when that side is its
     * right). A left turn mirrors the measured right turn: not measured. */
    for (int32_t k = 0; k < n; k++) {
        fsim_entity *b = &env->entities[belts[k].index];
        int32_t ux, uy;
        dir_vec(b->direction, &ux, &uy);
        int64_t tx = floordiv(b->pos.x, TILE), ty = floordiv(b->pos.y, TILE);
        int32_t rx = -uy, ry = ux;
        int behind = belt_on(env, belts, n, tx - ux, ty - uy, b->direction) >= 0;
        int from_right = belt_on(env, belts, n, tx + rx, ty + ry, vec_dir(-rx, -ry)) >= 0;
        int from_left = belt_on(env, belts, n, tx - rx, ty - ry, vec_dir(rx, ry)) >= 0;
        int32_t old_lengths[2] = {b->lane_length[0], b->lane_length[1]};
        b->shape = BELT_STRAIGHT;
        if (!behind && from_right != from_left) b->shape = from_right ? BELT_RIGHT : BELT_LEFT;
        b->lane_length[0] = b->shape == BELT_RIGHT ? BELT_CURVE_OUTER
                          : b->shape == BELT_LEFT ? BELT_CURVE_INNER : TILE;
        b->lane_length[1] = b->shape == BELT_RIGHT ? BELT_CURVE_INNER
                          : b->shape == BELT_LEFT ? BELT_CURVE_OUTER : TILE;
        /* A belt that became or stopped being a turn keeps its items on the
         * same lanes, each at floor(p * (new length - 1) / old length):
         * FactorioRL tools/probe_logistics2.py `rot_*`, every position of a
         * right and a left turn rotated to straight and back, a turn's
         * feeder removed or a feeder added behind it (and 64 -> 55, 263 ->
         * 227, inner 64 -> 153 in the logistics_belt_rotate_and_mine
         * trace). Items can end up closer than 64. A belt that stays
         * straight keeps its positions (`rot_ss_*`). */
        for (int lane = 0; lane < 2; lane++) {
            int32_t old_length = old_lengths[lane], new_length = b->lane_length[lane];
            if (old_length == new_length || old_length <= 0) continue;
            fsim_lane *l = &b->lanes[lane];
            for (int32_t j = 0; j < l->count; j++)
                l->items[j].pos = (int16_t)((int64_t)l->items[j].pos * (new_length - 1) /
                                            old_length);
        }
    }

    /* Links: into the belt ahead, lane to lane, unless it faces back at this
     * one; into its side when that belt is straight. */
    int32_t pred[FSIM_MAX_LANES];
    for (int32_t r = 0; r < FSIM_MAX_LANES; r++) pred[r] = -1;
    for (int32_t k = 0; k < n; k++) {
        int32_t index = belts[k].index;
        fsim_entity *b = &env->entities[index];
        int32_t ux, uy;
        dir_vec(b->direction, &ux, &uy);
        int64_t tx = floordiv(b->pos.x, TILE), ty = floordiv(b->pos.y, TILE);
        for (int lane = 0; lane < 2; lane++) b->lane_next[lane] = b->lane_side[lane] = -1;
        int32_t ahead = belt_on(env, belts, n, tx + ux, ty + uy, -1);
        if (ahead < 0) continue;
        const fsim_entity *a = &env->entities[ahead];
        if (a->direction == (b->direction + 8) % 16) continue;
        if (a->direction == b->direction || a->shape != BELT_STRAIGHT) {
            for (int lane = 0; lane < 2; lane++) {
                b->lane_next[lane] = ahead * 2 + lane;
                pred[ahead * 2 + lane] = index * 2 + lane;
            }
            continue;
        }
        /* Sideload: onto the target's lane on the side this belt comes from.
         * Only a feed into the right side of the target was measured. */
        int32_t ax, ay;
        dir_vec(a->direction, &ax, &ay);
        int from_right = tx - floordiv(a->pos.x, TILE) == -ay &&
                         ty - floordiv(a->pos.y, TILE) == ax;
        int32_t target = ahead * 2 + (from_right ? 1 : 0);
        for (int lane = 0; lane < 2; lane++) {
            /* Lane 1 lies to the left of this belt's travel, (uy, -ux). */
            int32_t sx = lane == 0 ? uy : -uy, sy = lane == 0 ? -ux : ux;
            int upstream = sx * ax + sy * ay < 0;
            b->lane_side[lane] = target;
            b->lane_entry[lane] = upstream ? SIDELOAD_FAR : SIDELOAD_NEAR;
        }
    }

    /* Chains: from each lane that runs into nothing, back along its feeders;
     * what is left over is closed loops. By ref, so the order does not depend
     * on where the belts are. */
    int32_t chain_of[FSIM_MAX_LANES];
    int32_t first[FSIM_MAX_LANES], size[FSIM_MAX_LANES], lanes[FSIM_MAX_LANES];
    for (int32_t r = 0; r < FSIM_MAX_LANES; r++) chain_of[r] = -1;
    int32_t chains = 0, used = 0;
    int32_t refs[FSIM_MAX_LANES];
    int32_t nrefs = 0;
    for (int32_t k = 0; k < n; k++) {
        refs[nrefs++] = belts[k].index * 2;
        refs[nrefs++] = belts[k].index * 2 + 1;
    }
    /* By entity, lane 2 first. Chain order no longer decides what moves first
     * (segments do); it only numbers the chains. */
    for (int32_t i = 1; i < nrefs; i++) {
        int32_t v = refs[i], j = i - 1;
        while (j >= 0 && (refs[j] ^ 1) > (v ^ 1)) {
            refs[j + 1] = refs[j];
            j--;
        }
        refs[j + 1] = v;
    }
    for (int32_t pass = 0; pass < 2; pass++) {
        for (int32_t i = 0; i < nrefs; i++) {
            int32_t r = refs[i];
            if (chain_of[r] >= 0) continue;
            if (pass == 0 && env->entities[r >> 1].lane_next[r & 1] >= 0) continue;
            first[chains] = used;
            int32_t count = 0;
            for (int32_t at = r; at >= 0 && chain_of[at] < 0; at = pred[at]) {
                chain_of[at] = chains;
                lanes[used++] = at;
                count++;
            }
            size[chains] = pass == 0 ? count : -count;
            chains++;
        }
    }

    /* Order: a chain after the chain it sideloads into. Depth first; a cycle
     * of sideloads runs in chain order. */
    int32_t state[FSIM_MAX_LANES], order[FSIM_MAX_LANES], stack[FSIM_MAX_LANES];
    int32_t done = 0;
    for (int32_t c = 0; c < chains; c++) state[c] = 0;
    for (int32_t c = 0; c < chains; c++) {
        int32_t depth = 0;
        stack[depth++] = c;
        while (depth > 0) {
            int32_t at = stack[depth - 1];
            if (state[at] == 2) {
                depth--;
                continue;
            }
            int32_t target = -1;
            if (size[at] > 0) {
                int32_t front = lanes[first[at]];
                int32_t side = env->entities[front >> 1].lane_side[front & 1];
                if (side >= 0) target = chain_of[side];
            }
            if (state[at] == 0 && target >= 0 && state[target] == 0) {
                state[at] = 1;
                stack[depth++] = target;
                continue;
            }
            state[at] = 2;
            order[done++] = at;
            depth--;
        }
    }
    int32_t out = 0;
    for (int32_t r = 0; r < FSIM_MAX_LANES; r++) env->lane_chain[r] = -1;
    for (int32_t k = 0; k < done; k++) {
        int32_t c = order[k];
        env->chain_first[k] = out;
        env->chain_size[k] = size[c];
        int32_t count = size[c] < 0 ? -size[c] : size[c];
        for (int32_t j = 0; j < count; j++) {
            env->lane_chain[lanes[first[c] + j]] = k;
            env->chain_lanes[out++] = lanes[first[c] + j];
        }
    }
    env->chain_count = done;

    /* Every inserter's targets. One asleep wakes when a target changes (a
     * machine built or removed at its pickup or drop point), or when it is
     * no longer waiting (a hidden-state load put it mid-swing). */
    for (int32_t index = 0; index < env->entity_count; index++) {
        fsim_entity *s = &env->entities[index];
        if (!s->alive || s->kind != K_INSERTER) continue;
        int32_t pickup = point_target(env, inserter_point(s, INSERTER_PICKUP), index);
        int32_t drop = point_target(env, inserter_point(s, -INSERTER_DROP), index);
        int changed = pickup != s->pickup_target || drop != s->drop_target;
        s->pickup_target = pickup;
        s->drop_target = drop;
        if (s->sleep_seq &&
            (changed || (s->phase != INS_WAIT_PICKUP && s->phase != INS_WAIT_DROP))) {
            s->sleep_seq = 0;
            env->sleeper_count--;
            env->inserters[env->inserter_count++] = index;
        }
    }
    for (int32_t i = 0; i < env->entity_count; i++) {
        fsim_entity *d = &env->entities[i];
        if (!drill_blocked(d)) continue;
        fsim_pos drop = drop_position(d);
        if (machine_at(env, drop, i) >= 0) continue;
        int32_t belt = belt_at(env, drop);
        if (belt >= 0 && drill_block_signature(env, belt, drop) != d->block_sig)
            d->status = ST_WORKING;
    }
    rebuild_segments(env, pred);
    env->logistics_version = env->entities_version;
}

/* ------------------------------------------------------------------ inserters
 *
 * Measured on the engine (docs/sim-logistics.md), burner inserters only:
 *
 * - As built the hand extends to the pickup for 8 ticks, 1,750 J each, and
 *   takes an item on the 8th; from then on pickup to drop and drop to pickup
 *   are 38 ticks each, the first 5 at 2,400 J and the rest at 650 J.
 * - An item goes into the drop target on the 38th tick; the hand then
 *   swings back empty, the same 38 ticks.
 * - With nothing to take, or nowhere to put it, the hand waits and draws
 *   nothing -- the tick it arrived is not even refilled. Inserters run before
 *   drills and furnaces, so a change there is acted on the tick after; the
 *   tick it acts, it refills the buffer and moves on the next.
 * - Before it picks up it checks that the drop target would take the item: a
 *   chest with room for it, a furnace below 2 ore and 5 fuel, a drill or an
 *   inserter below 5 fuel (FactorioRL tools/probe_logistics2.py, `full_*`,
 *   `fill_*`). A hand that would overfill waits empty at the pickup. From a
 *   chest it takes from the last slot holding an item the target wants
 *   (`mix_*`).
 * - Holding fuel with its own fuel slot empty, it swings to itself instead
 *   (28 ticks, 2,400 J each), fills the slot and swings back (28 more).
 * - With nothing at its drop point it drops on the ground, an item pile at
 *   the drop point, unless a pile is already there; with nothing at its
 *   pickup point it takes from the piles on that tile (`ground_*`,
 *   `gpair_*`).
 * - Update order (`order_*`, `wake3`, `chain_*`, `woken_*`, and
 *   probe_logistics4 `order`, dropping onto a belt too): inserters run from
 *   the last in the update list to the first, and a new inserter joins the
 *   end. One that waits on a machine falls asleep, leaving the list, and
 *   is woken by any change to that machine's contents or to its drop
 *   target's, rejoining the end: on the next tick those that fell asleep
 *   first run first, ahead of every inserter that stayed awake, and never on
 *   the tick they were woken in (a drop into a chest wakes the inserter
 *   waiting on it for the next tick, whichever of the two runs first).
 *
 * - The arm itself (rotation, extension, energy, the chase of a moving belt
 *   item, the redirect to its own fuel slot when that empties mid-swing) is
 *   the arm section below. An inserter waiting on a belt line sleeps on the
 *   line (belt_asleep) and stays in the list.
 *
 * Not modelled (FactorioRL docs/sim-logistics.md): the hand's drawn lift
 * (hand y) of a swing that did not start from rest. The young-belt wake delay
 * is the inserter watching its pickup belt's segment ("segments").
 */

static int32_t smelt_product(int32_t item);

/* Whether `s` would take `item` now: for its own empty fuel slot, or for a
 * drop target that would take it without passing the fill limits. A belt or
 * the ground is checked at the drop instead, where the hand waits. */
static int inserter_wants(const fsim_env *env, const fsim_entity *s, int32_t item) {
    if (is_fuel(item) && s->fuel.count == 0) return 1;
    if (s->drop_target < 0) return 1;   /* the ground */
    const fsim_entity *d = &env->entities[s->drop_target];
    switch (d->kind) {
    case K_CHEST:
        return chest_room(d, item) > 0;
    case K_BELT:
        return 1;
    case K_FURNACE:
        if (is_fuel(item))
            return d->fuel.count < INSERTER_FUEL_LIMIT &&
                   slot_room(&d->fuel, item, STACK_SIZE[item]) > 0;
        if (smelt_product(item) == IT_NONE) return 0;
        return d->source.count < INSERTER_SOURCE_LIMIT &&
               slot_room(&d->source, item, STACK_SIZE[item]) > 0;
    case K_DRILL:
    case K_INSERTER:
        return is_fuel(item) && d->fuel.count < INSERTER_FUEL_LIMIT &&
               slot_room(&d->fuel, item, STACK_SIZE[item]) > 0;
    default:
        return 0;
    }
}

/* Whether the drop target is full, which the inserter reports as
 * `waiting_for_space_in_destination` whatever its hand is doing, from the
 * tick the target fills (`full_*`, `fill_*`, `mix_*`, `stat_*`): a chest with
 * a full stack in every slot (one holding part stacks of other items is
 * not); a furnace at both its ore and its fuel limit; a drill at its fuel
 * limit. An inserter as the target never is, nor a belt. */
static int inserter_target_full(const fsim_env *env, const fsim_entity *s) {
    if (s->drop_target < 0) return 0;
    const fsim_entity *d = &env->entities[s->drop_target];
    switch (d->kind) {
    case K_CHEST:
        for (int i = 0; i < FSIM_CHEST_SLOTS; i++)
            if (d->chest[i].count == 0 || d->chest[i].count < STACK_SIZE[d->chest[i].item])
                return 0;
        return 1;
    case K_FURNACE:
        return d->fuel.count >= INSERTER_FUEL_LIMIT && d->source.count >= INSERTER_SOURCE_LIMIT;
    case K_DRILL:
        return d->fuel.count >= INSERTER_FUEL_LIMIT;
    default:
        return 0;
    }
}

/* The status of a waiting inserter. */
static int32_t inserter_wait_status(const fsim_env *env, const fsim_entity *s) {
    if (s->phase == INS_WAIT_DROP) return ST_WAITING_FOR_SPACE;
    return inserter_target_full(env, s) ? ST_WAITING_FOR_SPACE : ST_WAITING_FOR_SOURCE;
}

/* What a waiting inserter waits on: its drop target when it holds an item
 * or its source has something (which the target will not take now), else its
 * source; -1 for the ground. */
static int32_t inserter_waits_on(const fsim_env *env, const fsim_entity *s) {
    if (s->phase == INS_WAIT_DROP) return s->drop_target;
    int32_t src = s->pickup_target;
    int has = 0;
    if (src >= 0) {
        const fsim_entity *e = &env->entities[src];
        if (e->kind == K_CHEST)
            for (int i = 0; i < FSIM_CHEST_SLOTS && !has; i++) has = e->chest[i].count > 0;
        else if (e->kind == K_FURNACE) has = e->result.count > 0;
    }
    return has ? s->drop_target : src;
}

/* How it waits (probe_logistics2 `fill_*`, `ground_*`, `stat_busy_ins`):
 * 2, asleep, on a chest, furnace or drill, which wake it when their contents
 * change; 1 on a belt, where it keeps its buffer as its last move left it and
 * looks again every tick; 0 awake, on the ground or on an inserter, refilling
 * its buffer every tick and looking again. */
static int inserter_wait_mode(const fsim_env *env, const fsim_entity *s) {
    int32_t t = inserter_waits_on(env, s);
    if (t < 0) return 0;
    int32_t kind = env->entities[t].kind;
    if (kind == K_CHEST || kind == K_FURNACE || kind == K_DRILL) return 2;
    return kind == K_BELT ? 1 : 0;
}

static void inserter_sleep(fsim_env *env, int32_t index) {
    fsim_entity *s = &env->entities[index];
    inserter_unlist(env, index);
    s->sleep_seq = ++env->sleep_counter;
    env->sleeper_count++;
}

/* The contents of entity `index` changed: wake the inserters asleep on it.
 * They rejoin the update list latest asleep first, so the first asleep runs
 * first. */
static void wake(fsim_env *env, int32_t index) {
    if (env->sleeper_count == 0) return;
    int32_t found[FSIM_MAX_ENTITIES];
    int32_t n = 0;
    for (int32_t i = 0; i < env->entity_count; i++) {
        const fsim_entity *e = &env->entities[i];
        if (e->alive && e->kind == K_INSERTER && e->sleep_seq &&
            (e->pickup_target == index || e->drop_target == index))
            found[n++] = i;
    }
    for (int32_t a = 1; a < n; a++) {
        int32_t v = found[a], b = a - 1;
        while (b >= 0 && env->entities[found[b]].sleep_seq < env->entities[v].sleep_seq) {
            found[b + 1] = found[b];
            b--;
        }
        found[b + 1] = v;
    }
    for (int32_t k = 0; k < n; k++) {
        fsim_entity *e = &env->entities[found[k]];
        e->sleep_seq = 0;
        env->sleeper_count--;
        env->inserters[env->inserter_count++] = found[k];
        e->status = inserter_wait_status(env, e);
    }
}

/* Taking from the ground: a pile on the pickup tile, the one furthest along
 * the inserter's direction first, then the northmost, then the westmost
 * (`gpair_*`: two piles at mirrored offsets, all four facings). One item at a
 * time, like a chest (`ground_pick_stack`). */
static int inserter_ground_pickup(fsim_env *env, const fsim_entity *s, int32_t *item) {
    fsim_pos p = inserter_point(s, INSERTER_PICKUP);
    int64_t tx = floordiv(p.x, TILE), ty = floordiv(p.y, TILE);
    int32_t ux, uy;
    dir_vec(s->direction, &ux, &uy);
    int32_t best = -1;
    int64_t best_along = 0;
    for (int32_t i = 0; i < env->entity_count; i++) {
        const fsim_entity *e = &env->entities[i];
        if (!e->alive || e->kind != K_PILE || e->pile.count <= 0) continue;
        if (floordiv(e->pos.x, TILE) != tx || floordiv(e->pos.y, TILE) != ty) continue;
        if (!inserter_wants(env, s, e->pile.item)) continue;
        int64_t along = (int64_t)(e->pos.x - s->pos.x) * ux + (int64_t)(e->pos.y - s->pos.y) * uy;
        if (best >= 0) {
            const fsim_entity *b = &env->entities[best];
            if (along < best_along) continue;
            if (along == best_along &&
                (e->pos.y > b->pos.y || (e->pos.y == b->pos.y && e->pos.x >= b->pos.x)))
                continue;
        }
        best = i;
        best_along = along;
    }
    if (best < 0) return 0;
    fsim_entity *pile = &env->entities[best];
    *item = pile->pile.item;
    if (--pile->pile.count <= 0) destroy_entity(env, best);
    return 1;
}

/* The hand at the pickup: take one item if there is one it wants. */
static int inserter_pickup(fsim_env *env, int32_t index) {
    fsim_entity *s = &env->entities[index];
    int32_t item = IT_NONE;
    if (s->pickup_target < 0) {
        if (!inserter_ground_pickup(env, s, &item)) item = IT_NONE;
    } else {
        fsim_entity *src = &env->entities[s->pickup_target];
        if (src->kind == K_CHEST) {
            for (int i = FSIM_CHEST_SLOTS - 1; i >= 0 && item == IT_NONE; i--)
                if (src->chest[i].count > 0 && inserter_wants(env, s, src->chest[i].item)) {
                    item = src->chest[i].item;
                    slot_remove(&src->chest[i], item, 1);
                }
        } else if (src->kind == K_FURNACE) {
            if (src->result.count > 0 && inserter_wants(env, s, src->result.item)) {
                item = src->result.item;
                slot_remove(&src->result, item, 1);
            }
        }
        if (item != IT_NONE) {
            wake(env, s->pickup_target);
            if (src->kind == K_CHEST) unblock_drills(env, s->pickup_target, src->pos);
        }
    }
    if (item == IT_NONE) return 0;
    s = &env->entities[index];
    s->held = item;
    s->phase = is_fuel(item) && s->fuel.count == 0 ? INS_TO_SELF : INS_TO_DROP;
    s->swing = 0;
    return 1;
}

/* The hand at the drop: put the item in -- on the ground when nothing is at
 * the drop point, unless a pile already lies there (`ground_drop*`). */
static int inserter_drop(fsim_env *env, int32_t index) {
    fsim_entity *s = &env->entities[index];
    fsim_pos at = inserter_point(s, -INSERTER_DROP);
    int done;
    if (s->drop_target < 0) {
        done = 0;
        if (!pile_blocks(env, at)) {
            int32_t p = new_entity(env, K_PILE, at, 0, 1);
            if (p >= 0) {
                env->entities[p].pile.item = env->entities[index].held;
                env->entities[p].pile.count = 1;
                done = 1;
            }
        }
    } else {
        fsim_entity *d = &env->entities[s->drop_target];
        if (d->kind == K_BELT) {
            done = belt_drop(env, s->drop_target, at, s->held);
        } else {
            done = machine_accepts(d, s->held, 1, 1) == 1;
            if (done) wake(env, s->drop_target);
        }
    }
    if (!done) return 0;
    s = &env->entities[index];
    s->held = IT_NONE;
    s->phase = INS_TO_PICKUP;
    s->swing = 0;
    return 1;
}

/* ------------------------------------------------------------ the arm
 *
 * FactorioRL docs/sim-logistics.md, "Inserter belt pickup" (tools/
 * inserter_model.py replays 1,154 engine rigs with it), in the engine's
 * arithmetic:
 *
 * - The arm is an orientation and a length. The orientation is kept in
 *   world terms, turns clockwise from north in [0, 1), in single precision;
 *   a target's orientation is atan2 of its offset over a full turn, rounded
 *   to single, plus one when negative. Every energy the chase probe recorded
 *   agrees to the last bit of a double only this way (to 3 mJ otherwise).
 * - Each tick it moves toward a target: extension at most 0.035 tile, set
 *   outright within that and free within 0.001; if less than a step is left
 *   after a step it is set too. Rotation at most 0.013 turn (single) the
 *   short way, an exact half turn going anticlockwise for +0.5 and clockwise
 *   for -0.5; set outright within a step, or when the extension got there
 *   this tick and less than a step is left after the step.
 * - Energy: 50 kJ a tile of extension and 50 kJ a turn of rotation, as
 *   charged. Short of that, the extension is paid first and moves in
 *   proportion; the rotation gets the rest as a single-precision turn, and
 *   what that costs is taken, the buffer never going below zero
 *   (probe_logistics2 `pe_*`: the buffer left after the short tick is the
 *   rest less 50 kJ times the single-precision turn, to the bit, on 26
 *   ticks). With next to nothing left it does not move at all: it keeps the
 *   crumb and reads `working` (`pe_1` and five more: crumbs of 2e-14 to
 *   6e-13 J stayed; 9e-7 J and more were spent). ARM_MIN_ENERGY lies between.
 * - The hand is drawn at the arm's end, truncated toward zero to 1/256 tile
 *   in each world axis; on its target it is drawn exactly on it (an item's
 *   lane line reads 60, not 59.99999999). The drawing also lifts the hand
 *   during a swing, by the tables below, read off the chest-to-chest swings;
 *   for a swing that did not start from rest at the pickup, the drop point or
 *   its own fuel slot the lift is not known (`lift` -1).
 */

#define ARM_ROT 0.013f
#define ARM_EXT 0.035
#define ARM_DEADZONE 0.001
#define ARM_START 0.7
#define ARM_PICKUP 1.0
#define ARM_DROP 1.2
#define ARM_ENERGY 50000.0
#define ARM_MIN_ENERGY 1e-9
/* Its own fuel slot: 0.01 tile right and 0.01 behind for a north-facing
 * inserter (the swing to it and back, `self`, every tick). */
#define ARM_SELF 2.56

static const int32_t HAND_LIFT[39] = {
    0, 15, 30, 44, 57, 70, 81, 92, 101, 110, 118, 125, 132, 137, 142, 145, 148, 150, 151, 151,
    151, 149, 147, 143, 139, 134, 128, 122, 114, 106, 96, 86, 75, 63, 50, 37, 22, 7, 0,
};
static const int32_t HAND_SELF_LIFT[29] = {
    0, 7, 14, 19, 24, 28, 31, 34, 35, 35, 35, 34, 32, 29, 25, 20, 15, 9, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0,
};

typedef struct {
    float w;
    double len;             /* tiles */
    double vx, vy;          /* offset from the inserter, world, 1/256 */
} arm_target;

static float arm_orient(double vx, double vy) {
    float w = (float)(atan2(vx, -vy) / (2.0 * 3.14159265358979323846));
    if (w < 0.0f) w = w + 1.0f;
    return w;
}

static arm_target arm_toward(double vx, double vy) {
    arm_target t = {arm_orient(vx, vy), sqrt(vx * vx + vy * vy) / TILE, vx, vy};
    return t;
}

/* A point at (lx, ly) in the inserter's own frame (pickup ahead at (0, -1)),
 * 1/256 tile, turned to the world. */
static arm_target arm_local(const fsim_entity *s, double lx, double ly) {
    switch (s->direction) {
    case 4: return arm_toward(-ly, lx);
    case 8: return arm_toward(-lx, -ly);
    case 12: return arm_toward(ly, -lx);
    default: return arm_toward(lx, ly);
    }
}

static arm_target arm_pickup_point(const fsim_entity *s) {
    return arm_local(s, 0.0, -ARM_PICKUP * TILE);
}

static arm_target arm_drop_point(const fsim_entity *s) {
    return arm_local(s, 0.0, ARM_DROP * TILE);
}

/* Its own fuel slot: the swing to it is the north one turned for an inserter
 * facing east and mirrored for south and west, as its drawn hand is. */
static arm_target arm_self_point(const fsim_entity *s) {
    double x = ARM_SELF, y = ARM_SELF;
    switch (s->direction) {
    case 4: return arm_toward(-y, x);
    case 8: return arm_toward(x, -y);
    case 12: return arm_toward(y, x);
    default: return arm_toward(x, y);
    }
}

/* One tick toward `t` with `budget` J in the buffer. Returns 1 when the arm
 * is on the target; `spent[0]` and `spent[1]` are what the extension and the
 * rotation cost, which the buffer loses in that order (`pe_19`: the other
 * order leaves 2e-13 J more after the tick), `*short_tick` whether the
 * budget cut it short. */
static int arm_step(fsim_entity *s, arm_target t, double budget, double *spent, int *short_tick) {
    spent[0] = spent[1] = 0.0;
    *short_tick = 0;
    if (budget < ARM_MIN_ENERGY) {
        *short_tick = 1;
        return 0;
    }
    float d = t.w - s->arm_w;
    if (d >= 0.5f) d = d - 1.0f;
    else if (d <= -0.5f) d = d + 1.0f;
    double dl = t.len - s->arm_len, ext, len;
    int rl;
    if (fabs(dl) < ARM_DEADZONE) {
        ext = 0.0;
        len = t.len;
        rl = 1;
    } else if (fabs(dl) <= ARM_EXT) {
        ext = fabs(dl);
        len = t.len;
        rl = 1;
    } else {
        ext = ARM_EXT;
        len = s->arm_len + (dl > 0 ? ARM_EXT : -ARM_EXT);
        rl = 0;
        if (fabs(t.len - len) < ARM_EXT) {
            len = t.len;
            rl = 1;
        }
    }
    float ad = fabsf(d), rot, w;
    int ra;
    if (ad <= ARM_ROT) {
        rot = ad;
        w = t.w;
        ra = 1;
    } else {
        rot = ARM_ROT;
        w = s->arm_w + (d > 0 ? ARM_ROT : -ARM_ROT);
        if (w >= 1.0f) w = w - 1.0f;
        else if (w < 0.0f) w = w + 1.0f;
        ra = 0;
        if (rl && (double)ad - (double)ARM_ROT < (double)ARM_ROT) {
            w = t.w;
            ra = 1;
        }
    }
    double e_ext = ARM_ENERGY * ext, e_rot = ARM_ENERGY * (double)rot;
    if (e_ext + e_rot > budget) {
        *short_tick = 1;
        if (e_ext >= budget) {
            len = s->arm_len + (dl > 0 ? 1.0 : -1.0) * ext * (budget / e_ext);
            s->arm_len = len;
            spent[0] = budget;
            return 0;
        }
        float part = (float)((budget - e_ext) / ARM_ENERGY);
        w = s->arm_w + (d > 0 ? part : -part);
        if (w >= 1.0f) w = w - 1.0f;
        else if (w < 0.0f) w = w + 1.0f;
        s->arm_w = w;
        s->arm_len = len;
        spent[0] = e_ext;
        spent[1] = ARM_ENERGY * (double)part;
        return 0;
    }
    s->arm_w = w;
    s->arm_len = len;
    spent[0] = e_ext;
    spent[1] = e_rot;
    return ra && rl;
}

/* The buffer pays for a step. */
static void arm_pay(fsim_entity *s, const double *spent) {
    s->energy -= spent[0];
    s->energy -= spent[1];
    if (s->energy < 0) s->energy = 0;
}

/* The drawn hand, from the arm and the lift in use. */
static void arm_draw(fsim_entity *s) {
    double a = 2.0 * 3.14159265358979323846 * (double)s->arm_w, r = s->arm_len * TILE;
    double vx = r * sin(a), vy = -r * cos(a);
    if (s->arm_at && s->lift >= 0) {
        /* At rest on the pickup, the drop or its fuel slot. */
        vx = s->arm_vx;
        vy = s->arm_vy;
    }
    int32_t lift = 0;
    if (s->lift == 1 && s->lift_step >= 0 && s->lift_step < 39) lift = HAND_LIFT[s->lift_step];
    else if (s->lift == 2 && s->lift_step >= 0 && s->lift_step < 29)
        lift = HAND_SELF_LIFT[s->lift_step];
    s->hand_x = (int32_t)vx;
    s->hand_y = (int32_t)vy - lift;
}

/* Rest the arm exactly on `t`, as a hidden-state load or construction does. */
static void arm_place(fsim_entity *s, arm_target t) {
    s->arm_w = t.w;
    s->arm_len = t.len;
    s->arm_at = 1;
    s->arm_vx = t.vx;
    s->arm_vy = t.vy;
}

/* A move starts: `lift` names its drawn lift (see arm_draw). The move after
 * one cut short by the buffer is drawn with a lift that is not known either
 * (`pe_11`: 1/256 off the table at two ticks of the next swing, then exact). */
static void arm_begin(fsim_entity *s, int32_t lift) {
    s->lift = s->lift == -2 ? -1 : lift;
    s->lift_step = 0;
}

/* A tick cut short: this move's lift, and the next one's, are not known. */
static void arm_taint(fsim_entity *s) { s->lift = -2; }

/* ------------------------------------------------------------ taking from a belt
 *
 * Rules 4 to 6 of "Inserter belt pickup":
 *
 * - The hand chases the item it chose as long as that item is on the pickup
 *   belt (either lane, this belt's own positions). Otherwise it chooses among
 *   the items there, after this tick's belt move: the lane nearer the
 *   inserter first (for a belt running along the arm, lane 1), then the item
 *   furthest upstream. The choice sticks. With none it heads for the pickup
 *   point. Arriving on an item it takes it, that tick.
 * - If the item it chased left the pickup belt while the hand is over the
 *   pickup belt's tile, it does nothing that tick and chooses again the next.
 * - At rest on the pickup point with nothing to chase and no item anywhere
 *   on its belt's line (the chains of both lanes, upstream and downstream),
 *   it falls asleep, keeping its buffer. An item added to the line anywhere
 *   but on the pickup belt wakes it and refills it; an item on the pickup
 *   belt wakes it into a move paid from what it kept. With items on the line
 *   it waits awake, refilled every tick.
 *
 * A turn as the pickup belt: the same rules, with the item where the engine
 * puts it on the turn's arc (belt_item_offset; exact from the north and south
 * of a turn fed from the west, 1/256 off in hand x on two ticks from the east,
 * FactorioRL `tpick_*`). "The line" is the pickup belt's segment: on young
 * belts, before their chain merges, an item upstream does not keep the
 * inserter awake (the young-belt wake delay, "segments").
 */

/* Where an item sits on a turn, measured (FactorioRL tools/probe_logistics3.py,
 * `turn_points`: LuaTransportLine.get_line_item_position at every position of
 * both lanes of all eight turns). The points lie on quarter circles about the
 * inner corner, radius 188 and 67, but on the 1/256 grid the engine keeps, so
 * they are tabled rather than computed. Offsets from the belt's centre, 1/256
 * tile, for a right turn facing south (fed from the west): x is the left of
 * travel (east), y the direction of travel (south). Index: the position on
 * the lane, 0 at the exit. A left turn is the mirror image, its lanes swapped;
 * every facing is a rotation of these, exactly. */
static const int16_t TURN_OUTER[296][2] = {
    {60, 128}, {60, 127}, {60, 126}, {60, 125}, {60, 124}, {60, 123}, {60, 122}, {60, 121},
    {59, 120}, {59, 119}, {59, 118}, {59, 117}, {59, 116}, {59, 115}, {59, 114}, {59, 113},
    {59, 112}, {59, 111}, {59, 110}, {59, 109}, {59, 108}, {58, 108}, {58, 107}, {58, 106},
    {58, 105}, {58, 104}, {58, 103}, {58, 102}, {58, 101}, {57, 100}, {57, 99}, {57, 98},
    {57, 97}, {57, 96}, {57, 95}, {56, 94}, {56, 93}, {56, 92}, {56, 91}, {56, 90},
    {55, 89}, {55, 88}, {55, 87}, {55, 86}, {55, 85}, {54, 84}, {54, 83}, {54, 82},
    {54, 81}, {53, 80}, {53, 79}, {53, 78}, {52, 77}, {52, 76}, {52, 75}, {52, 74},
    {51, 73}, {51, 72}, {51, 71}, {50, 70}, {50, 69}, {50, 68}, {49, 68}, {49, 67},
    {49, 66}, {49, 65}, {48, 64}, {48, 63}, {47, 62}, {47, 61}, {47, 60}, {46, 59},
    {46, 58}, {46, 57}, {45, 56}, {45, 55}, {44, 54}, {44, 53}, {44, 53}, {43, 52},
    {43, 51}, {42, 50}, {42, 49}, {42, 48}, {41, 47}, {41, 46}, {40, 45}, {40, 44},
    {39, 44}, {39, 43}, {38, 42}, {38, 41}, {38, 40}, {37, 39}, {37, 38}, {36, 37},
    {36, 36}, {35, 36}, {35, 35}, {34, 34}, {34, 33}, {33, 32}, {33, 31}, {32, 30},
    {32, 30}, {31, 29}, {30, 28}, {30, 27}, {29, 26}, {29, 25}, {28, 24}, {28, 24},
    {27, 23}, {27, 22}, {26, 21}, {25, 20}, {25, 20}, {24, 19}, {24, 18}, {23, 17},
    {23, 16}, {22, 15}, {21, 15}, {21, 14}, {20, 13}, {19, 12}, {19, 12}, {18, 11},
    {18, 10}, {17, 9}, {16, 8}, {16, 8}, {15, 7}, {14, 6}, {14, 5}, {13, 5},
    {12, 4}, {12, 3}, {11, 2}, {10, 2}, {10, 1}, {9, 0}, {8, -1}, {8, -1},
    {7, -2}, {6, -3}, {6, -3}, {5, -4}, {4, -5}, {3, -6}, {3, -6}, {2, -7},
    {1, -8}, {1, -8}, {0, -9}, {-1, -10}, {-2, -10}, {-2, -11}, {-3, -12}, {-4, -12},
    {-5, -13}, {-5, -14}, {-6, -14}, {-7, -15}, {-8, -16}, {-8, -16}, {-9, -17}, {-10, -18},
    {-11, -18}, {-12, -19}, {-12, -19}, {-13, -20}, {-14, -21}, {-15, -21}, {-15, -22}, {-16, -23},
    {-17, -23}, {-18, -24}, {-19, -24}, {-20, -25}, {-20, -25}, {-21, -26}, {-22, -27}, {-23, -27},
    {-24, -28}, {-24, -28}, {-25, -29}, {-26, -29}, {-27, -30}, {-28, -30}, {-29, -31}, {-30, -32},
    {-30, -32}, {-31, -33}, {-32, -33}, {-33, -34}, {-34, -34}, {-35, -35}, {-36, -35}, {-36, -36},
    {-37, -36}, {-38, -37}, {-39, -37}, {-40, -38}, {-41, -38}, {-42, -38}, {-43, -39}, {-44, -39},
    {-44, -40}, {-45, -40}, {-46, -41}, {-47, -41}, {-48, -42}, {-49, -42}, {-50, -42}, {-51, -43},
    {-52, -43}, {-53, -44}, {-53, -44}, {-54, -44}, {-55, -45}, {-56, -45}, {-57, -46}, {-58, -46},
    {-59, -46}, {-60, -47}, {-61, -47}, {-62, -47}, {-63, -48}, {-64, -48}, {-65, -49}, {-66, -49},
    {-67, -49}, {-68, -49}, {-68, -50}, {-69, -50}, {-70, -50}, {-71, -51}, {-72, -51}, {-73, -51},
    {-74, -52}, {-75, -52}, {-76, -52}, {-77, -52}, {-78, -53}, {-79, -53}, {-80, -53}, {-81, -54},
    {-82, -54}, {-83, -54}, {-84, -54}, {-85, -55}, {-86, -55}, {-87, -55}, {-88, -55}, {-89, -55},
    {-90, -56}, {-91, -56}, {-92, -56}, {-93, -56}, {-94, -56}, {-95, -57}, {-96, -57}, {-97, -57},
    {-98, -57}, {-99, -57}, {-100, -57}, {-101, -58}, {-102, -58}, {-103, -58}, {-104, -58}, {-105, -58},
    {-106, -58}, {-107, -58}, {-108, -58}, {-108, -59}, {-109, -59}, {-110, -59}, {-111, -59}, {-112, -59},
    {-113, -59}, {-114, -59}, {-115, -59}, {-116, -59}, {-117, -59}, {-118, -59}, {-119, -59}, {-120, -59},
    {-121, -60}, {-122, -60}, {-123, -60}, {-124, -60}, {-125, -60}, {-126, -60}, {-127, -60}, {-127, -60},
};
static const int16_t TURN_INNER[107][2] = {
    {-61, 128}, {-61, 127}, {-61, 126}, {-61, 125}, {-61, 124}, {-61, 123}, {-61, 122}, {-61, 121},
    {-61, 120}, {-61, 119}, {-61, 118}, {-62, 117}, {-62, 116}, {-62, 116}, {-62, 115}, {-62, 114},
    {-63, 113}, {-63, 112}, {-63, 111}, {-63, 110}, {-64, 109}, {-64, 108}, {-64, 107}, {-65, 106},
    {-65, 105}, {-65, 104}, {-66, 103}, {-66, 102}, {-66, 101}, {-67, 100}, {-67, 99}, {-68, 98},
    {-68, 98}, {-69, 97}, {-69, 96}, {-70, 95}, {-70, 94}, {-71, 93}, {-71, 92}, {-72, 91},
    {-72, 91}, {-73, 90}, {-73, 89}, {-74, 88}, {-75, 87}, {-75, 87}, {-76, 86}, {-76, 85},
    {-77, 84}, {-78, 83}, {-78, 83}, {-79, 82}, {-80, 81}, {-81, 81}, {-81, 80}, {-82, 79},
    {-83, 78}, {-83, 78}, {-84, 77}, {-85, 76}, {-86, 76}, {-87, 75}, {-87, 75}, {-88, 74},
    {-89, 73}, {-90, 73}, {-91, 72}, {-91, 72}, {-92, 71}, {-93, 71}, {-94, 70}, {-95, 70},
    {-96, 69}, {-97, 69}, {-98, 68}, {-98, 68}, {-99, 67}, {-100, 67}, {-101, 66}, {-102, 66},
    {-103, 66}, {-104, 65}, {-105, 65}, {-106, 65}, {-107, 64}, {-108, 64}, {-109, 64}, {-110, 63},
    {-111, 63}, {-112, 63}, {-113, 63}, {-114, 62}, {-115, 62}, {-116, 62}, {-116, 62}, {-117, 62},
    {-118, 61}, {-119, 61}, {-120, 61}, {-121, 61}, {-122, 61}, {-123, 61}, {-124, 61}, {-125, 61},
    {-126, 61}, {-127, 61}, {-127, 61},
};

/* Where item `pos` of lane `lane` on belt `b` is, as an offset from `s`,
 * 1/256 tile: on a straight belt on the lane's line, 60 either side of the
 * centre, at its distance from the downstream edge; on a turn the tabled
 * point. */
static void belt_item_offset(const fsim_entity *b, int32_t lane, int32_t pos, const fsim_entity *s,
                             double *vx, double *vy) {
    int32_t ux, uy;
    dir_vec(b->direction, &ux, &uy);
    double lx = uy, ly = -ux;            /* left of travel */
    double cx = b->pos.x - s->pos.x, cy = b->pos.y - s->pos.y;
    if (b->shape == BELT_STRAIGHT) {
        double along = TILE / 2 - pos, side = lane == 0 ? 60.0 : -60.0;
        *vx = cx + ux * along + lx * side;
        *vy = cy + uy * along + ly * side;
        return;
    }
    int right = b->shape == BELT_RIGHT;
    int inner = right == (lane == 1);
    int32_t n = inner ? 107 : 296;
    int32_t at = pos < 0 ? 0 : pos >= n ? n - 1 : pos;
    const int16_t *pt = inner ? TURN_INNER[at] : TURN_OUTER[at];
    double side = right ? pt[0] : -pt[0], fwd = pt[1];
    *vx = cx + lx * side + ux * fwd;
    *vy = cy + ly * side + uy * fwd;
}

typedef struct {
    int32_t lane, at, id;
    double vx, vy;
} belt_pick;

/* The item on the pickup belt to chase, or 0 when there is none. */
static int belt_choose(fsim_env *env, const fsim_entity *s, belt_pick *out) {
    const fsim_entity *b = &env->entities[s->pickup_target];
    int32_t ux, uy, bx, by;
    dir_vec(s->direction, &ux, &uy);
    dir_vec(b->direction, &bx, &by);
    int along = ux * bx + uy * by != 0;   /* the belt runs along the arm */
    int found = 0;
    int64_t best_key = 0;
    for (int32_t lane = 0; lane < 2; lane++) {
        const fsim_lane *l = &b->lanes[lane];
        for (int32_t at = 0; at < l->count; at++) {
            const fsim_belt_item *it = &l->items[at];
            if (it->pos >= b->lane_length[lane] || !inserter_wants(env, s, it->item)) continue;
            double vx, vy;
            belt_item_offset(b, lane, it->pos, s, &vx, &vy);
            /* Distance of the lane from the inserter: across the arm for a
             * belt crossing in front, sideways for one along it. */
            double fwd = -(vx * ux + vy * uy), lat = vx * uy - vy * ux;
            int64_t dist = (int64_t)floor((along ? fabs(lat) : fabs(fwd)) + 0.5);
            /* On a turn, lane 1 first from every side (probe_logistics2
             * `tpick_*`: the outer lane of a right turn, the inner of a left). */
            if (b->shape != BELT_STRAIGHT) dist = 0;
            int64_t key = (dist * 2 + lane) * 4096 - it->pos;
            if (found && key >= best_key) continue;
            found = 1;
            best_key = key;
            out->lane = lane;
            out->at = at;
            out->id = it->id;
            out->vx = vx;
            out->vy = vy;
        }
    }
    return found;
}

/* The chased item, if it is still on the pickup belt. */
static int belt_find(fsim_env *env, const fsim_entity *s, int32_t id, belt_pick *out) {
    const fsim_entity *b = &env->entities[s->pickup_target];
    for (int32_t lane = 0; lane < 2; lane++) {
        const fsim_lane *l = &b->lanes[lane];
        for (int32_t at = 0; at < l->count; at++) {
            if (l->items[at].id != id || l->items[at].pos >= b->lane_length[lane]) continue;
            out->lane = lane;
            out->at = at;
            out->id = id;
            belt_item_offset(b, lane, l->items[at].pos, s, &out->vx, &out->vy);
            return 1;
        }
    }
    return 0;
}

/* Whether the hand is over the pickup belt's tile. */
static int hand_over_pickup(const fsim_entity *s) {
    double a = 2.0 * 3.14159265358979323846 * (double)s->arm_w, r = s->arm_len * TILE;
    double vx = s->arm_at ? s->arm_vx : r * sin(a), vy = s->arm_at ? s->arm_vy : -r * cos(a);
    int32_t ux, uy;
    dir_vec(s->direction, &ux, &uy);
    double fwd = vx * ux + vy * uy, lat = vx * uy - vy * ux;
    return fabs(lat) <= TILE / 2 && fwd >= TILE / 2 && fwd <= 3 * TILE / 2;
}

/* The lanes an inserter picking from belt lane `ref` watches: its segment
 * ("segments"), or the whole loop it is on. */
static int32_t watched_lanes(const fsim_env *env, int32_t ref, const int32_t **refs) {
    int32_t h = env->seg_head[ref];
    if (h >= 0) return seg_lanes(env, h, refs);
    int32_t c = env->lane_chain[ref];
    if (c < 0) return 0;
    *refs = &env->chain_lanes[env->chain_first[c]];
    return env->chain_size[c] < 0 ? -env->chain_size[c] : env->chain_size[c];
}

/* Whether lanes `a` and `b` are watched together: one segment, or one loop. */
static int same_watch(const fsim_env *env, int32_t a, int32_t b) {
    int32_t ha = env->seg_head[a], hb = env->seg_head[b];
    if (ha >= 0 || hb >= 0) return ha == hb;
    return env->lane_chain[a] >= 0 && env->lane_chain[a] == env->lane_chain[b];
}

/* Whether any item is on the segments of `s`'s pickup belt. */
static int belt_line_busy(const fsim_env *env, const fsim_entity *s) {
    int32_t b = s->pickup_target;
    for (int32_t lane = 0; lane < 2; lane++) {
        const int32_t *refs;
        int32_t n = watched_lanes(env, b * 2 + lane, &refs);
        for (int32_t k = 0; k < n; k++)
            if (lane_of((fsim_env *)env, refs[k])->count > 0) return 1;
    }
    return 0;
}

/* An item was added to lane `ref`: wake the inserters asleep on its segment,
 * unless the lane is their own pickup belt's. A woken inserter refills its
 * buffer in its next update, which moves it only if there is something on
 * its pickup belt by then (probe_logistics tick_ins: a drill's output onto
 * the line at t=483, after the inserters ran, shows as a refill at t=484). */
static void belt_line_added(fsim_env *env, int32_t ref) {
    if (env->belt_sleepers == 0) return;
    for (int32_t k = 0; k < env->inserter_count; k++) {
        fsim_entity *s = &env->entities[env->inserters[k]];
        if (!s->belt_asleep || s->pickup_target < 0 || s->pickup_target == ref >> 1) continue;
        int32_t b = s->pickup_target;
        if (!same_watch(env, b * 2, ref) && !same_watch(env, b * 2 + 1, ref)) continue;
        s->belt_asleep = 0;
        env->belt_sleepers--;
        /* Woken while the belts move, before the inserters run, it acts this
         * tick (logistics_smelting_chain, t=803: ore crossing onto its
         * segment); woken by another entity, from the next. */
        if (!env->belt_phase) s->woke_tick = env->tick;
    }
}

/* Whether any item on the segments of `s`'s pickup belt is one it would take. */
static int belt_line_wanted(const fsim_env *env, const fsim_entity *s) {
    int32_t b = s->pickup_target;
    for (int32_t lane = 0; lane < 2; lane++) {
        const int32_t *refs;
        int32_t n = watched_lanes(env, b * 2 + lane, &refs);
        for (int32_t k = 0; k < n; k++) {
            const fsim_lane *l = lane_of((fsim_env *)env, refs[k]);
            for (int32_t at = 0; at < l->count; at++)
                if (inserter_wants(env, s, l->items[at].item)) return 1;
        }
    }
    return 0;
}

/* ------------------------------------------------------------ the cycle */

/* Arrived at the drop point holding an item, at its own fuel slot, or at the
 * pickup point empty: act. Returns 0 when it now waits. */
static int inserter_arrive(fsim_env *env, int32_t index) {
    fsim_entity *s = &env->entities[index];
    switch (s->phase) {
    case INS_TO_DROP:
        if (inserter_drop(env, index)) {
            s = &env->entities[index];
            arm_begin(s, 1);
            return 1;
        }
        env->entities[index].phase = INS_WAIT_DROP;
        return 0;
    case INS_TO_SELF:
        if (slot_insert(&s->fuel, s->held, 1, STACK_SIZE[s->held]) == 1) {
            s->held = IT_NONE;
            s->phase = INS_SELF_BACK;
            arm_begin(s, s->lift == 2 ? 2 : -1);
            wake(env, index);
        } else {
            s->phase = INS_TO_DROP;   /* its slot filled meanwhile: not measured */
        }
        return 1;
    default:
        if (inserter_pickup(env, index)) {
            s = &env->entities[index];
            arm_begin(s, s->phase == INS_TO_SELF ? 2 : 1);
            return 1;
        }
        env->entities[index].phase = INS_WAIT_PICKUP;
        return 0;
    }
}

/* It cannot act: report why, and sleep or stay awake as it would. */
static void inserter_wait(fsim_env *env, int32_t index) {
    fsim_entity *s = &env->entities[index];
    int mode = inserter_wait_mode(env, s);
    if (mode == 0) burner_refill(env, s);
    s = &env->entities[index];
    s->status = inserter_wait_status(env, s);
    if (mode == 2) inserter_sleep(env, index);
}

static void inserter_status(fsim_env *env, fsim_entity *s) {
    if (s->energy <= 0) s->status = ST_NO_FUEL;
    else s->status = inserter_target_full(env, s) ? ST_WAITING_FOR_SPACE : ST_WORKING;
}

/* One tick of an inserter whose pickup is a belt and whose hand is empty. */
static void inserter_chase(fsim_env *env, int32_t index) {
    fsim_entity *s = &env->entities[index];
    belt_pick pick;
    int have = 0;
    if (s->belt_asleep) {
        /* An item on the pickup belt wakes it, into a move on what it kept. */
        if (!belt_choose(env, s, &pick)) return;
        s->belt_asleep = 0;
        env->belt_sleepers--;
        have = 1;
        s->chase_id = pick.id;
    } else if (s->chase_id) {
        have = belt_find(env, s, s->chase_id, &pick);
        if (!have && hand_over_pickup(s)) {
            /* Lost over the belt: this tick is spent. */
            s->chase_id = 0;
            burner_refill(env, s);
            inserter_status(env, &env->entities[index]);
            return;
        }
    }
    if (!have && belt_choose(env, s, &pick)) {
        have = 1;
        s->chase_id = pick.id;
    }
    if (!have) s->chase_id = 0;
    arm_target t = have ? arm_toward(pick.vx, pick.vy) : arm_pickup_point(s);
    int at_rest = !have && s->arm_at && s->phase == INS_WAIT_PICKUP;
    double spent[2] = {0.0, 0.0};
    int short_tick = 0, arrived = 1;
    if (!at_rest) {
        if (have && s->lift == 1 && s->phase == INS_TO_PICKUP) s->lift = -1;
        arrived = arm_step(s, t, s->energy, spent, &short_tick);
        arm_pay(s, spent);
        if (short_tick) arm_taint(s);
        s->lift_step++;
        s->arm_at = arrived;
        if (arrived) {
            s->arm_vx = t.vx;
            s->arm_vy = t.vy;
        }
    }
    if (arrived && have) {
        /* On the item: take it. */
        fsim_entity *b = &env->entities[s->pickup_target];
        s->held = b->lanes[pick.lane].items[pick.at].item;
        lane_take(&b->lanes[pick.lane], pick.at);
        seg_interact(env, s->pickup_target * 2 + pick.lane, env->tick);
        seg_check(env, s->pickup_target * 2 + pick.lane);
        s->chase_id = 0;
        s->phase = is_fuel(s->held) && s->fuel.count == 0 ? INS_TO_SELF : INS_TO_DROP;
        arm_begin(s, -1);
    } else if (arrived && !have) {
        s->phase = INS_WAIT_PICKUP;
        arm_begin(s, 0);
        if (!belt_line_busy(env, s)) {
            /* Asleep, keeping what its last move left in the buffer. */
            s->belt_asleep = 1;
            env->belt_sleepers++;
            s->status = inserter_wait_status(env, s);
            return;
        }
        if (!belt_line_wanted(env, s) && s->drop_target >= 0 &&
            has_flag(env->entities[s->drop_target].kind, KF_MACHINE)) {
            /* Nothing on the line its target would take: asleep on the target
             * (probe_logistics `bend`: ore queued behind a furnace at its
             * limit). */
            s->status = inserter_wait_status(env, s);
            inserter_sleep(env, index);
            return;
        }
        burner_refill(env, s);
        s->status = inserter_wait_status(env, &env->entities[index]);
        return;
    } else if (s->phase == INS_WAIT_PICKUP) {
        s->phase = INS_TO_PICKUP;
    }
    burner_refill(env, s);
    inserter_status(env, &env->entities[index]);
}

static void update_inserter(fsim_env *env, int32_t index) {
    fsim_entity *s = &env->entities[index];
    /* Woken this tick by an item another entity put on its line: it runs
     * from the next (logistics_smelting_chain, t=730: a plate dropped onto
     * the line, the inserter at its end refilled at t=731). */
    if (s->woke_tick == env->tick && env->tick > 0) return;
    if (s->belt_asleep && (s->pickup_target < 0 || env->entities[s->pickup_target].kind != K_BELT
                           || s->held)) {
        s->belt_asleep = 0;
        env->belt_sleepers--;
    }
    if (!s->held && s->pickup_target >= 0 && env->entities[s->pickup_target].kind == K_BELT &&
        s->phase != INS_WAIT_DROP) {
        inserter_chase(env, index);
        arm_draw(&env->entities[index]);
        return;
    }
    if (s->belt_asleep) return;
    if (s->phase == INS_WAIT_PICKUP || s->phase == INS_WAIT_DROP) {
        int picking = s->phase == INS_WAIT_PICKUP;
        int acted = picking ? inserter_pickup(env, index) : inserter_drop(env, index);
        if (!acted) {
            inserter_wait(env, index);
            arm_draw(&env->entities[index]);
            return;
        }
        s = &env->entities[index];
        if (picking) arm_begin(s, s->phase == INS_TO_SELF ? 2 : 1);
        else arm_begin(s, 1);
    } else {
        /* Holding fuel with its own slot empty it swings to itself, from
         * wherever the slot empties (`sr_*`); otherwise to the drop. */
        if (s->held) {
            int32_t phase = is_fuel(s->held) && s->fuel.count == 0 ? INS_TO_SELF : INS_TO_DROP;
            if (phase != s->phase) {
                s->phase = phase;
                if (s->lift >= 0) s->lift = -1;
            }
        }
        arm_target t = s->phase == INS_TO_DROP  ? arm_drop_point(s)
                     : s->phase == INS_TO_SELF ? arm_self_point(s)
                                               : arm_pickup_point(s);
        double spent[2];
        int short_tick;
        int arrived = arm_step(s, t, s->energy, spent, &short_tick);
        arm_pay(s, spent);
        if (short_tick) arm_taint(s);
        s->lift_step++;
        s->arm_at = arrived;
        if (arrived) {
            s->arm_vx = t.vx;
            s->arm_vy = t.vy;
        }
        if (arrived && !inserter_arrive(env, index)) {
            inserter_wait(env, index);
            arm_draw(&env->entities[index]);
            return;
        }
    }
    s = &env->entities[index];
    burner_refill(env, s);
    s = &env->entities[index];
    inserter_status(env, s);
    arm_draw(s);
}

/* A drill whose output was refused reads `working` again as soon as its
 * output target changes, before it runs again (FactorioRL
 * tools/probe_logistics2.py `dstat_*`, read in the same script call as the
 * change): its drop belt extended or rotated, room made in its chest, the
 * pile at its drop point removed -- not a chest built nearby. The belt's
 * part is kept as a signature of its drop lane's links, checked when the
 * links are rebuilt. */
static int32_t drill_block_signature(const fsim_env *env, int32_t belt, fsim_pos drop) {
    int32_t ref, target;
    if (!belt_drop_target(env, belt, drop, &ref, &target)) return 0;
    const fsim_entity *b = &env->entities[belt];
    int32_t lane = ref & 1;
    return ((ref * 16 + b->direction) * 2053 + b->lane_next[lane] + 2) * 2053 + b->lane_side[lane] + 2;
}

static int drill_blocked(const fsim_entity *d) {
    return d->alive && d->kind == K_DRILL && d->held != IT_NONE &&
           d->status == ST_WAITING_FOR_SPACE;
}

/* Blocked drills that drop into machine `index`, or onto the ground within a
 * pile's reach of `where` when `index` < 0, read working. */
static void unblock_drills(fsim_env *env, int32_t index, fsim_pos where) {
    for (int32_t i = 0; i < env->entity_count; i++) {
        fsim_entity *d = &env->entities[i];
        if (!drill_blocked(d)) continue;
        fsim_pos drop = drop_position(d);
        int32_t target = machine_at(env, drop, i);
        if (index >= 0 ? target == index
                       : (target < 0 && belt_at(env, drop) < 0 &&
                          abs(drop.x - where.x) <= 2 * PILE_BOX &&
                          abs(drop.y - where.y) <= 2 * PILE_BOX))
            d->status = ST_WORKING;
    }
}

static void update_drill(fsim_env *env, int32_t index) {
    fsim_entity *d = &env->entities[index];
    fsim_pos drop = drop_position(d);
    int32_t target = machine_at(env, drop, index);
    int32_t belt = target < 0 ? belt_at(env, drop) : -1;
    if (d->held != IT_NONE) {
        if (target >= 0) {
            if (machine_accepts(&env->entities[target], d->held, 1, 1) == 1) {
                d->held = IT_NONE;
                d->linked_unit = env->entities[target].unit;
                wake(env, target);
            } else {
                d->status = ST_WAITING_FOR_SPACE;
                return;
            }
        } else if (belt >= 0) {
            /* Onto the lane whose half holds the drop point: lane 1 at 128
             * for a drill facing south onto a belt running east. */
            if (!belt_drop(env, belt, drop, d->held)) {
                d->status = ST_WAITING_FOR_SPACE;
                d->block_sig = drill_block_signature(env, belt, drop);
                return;
            }
            d = &env->entities[index];
            d->held = IT_NONE;
        } else {
            if (pile_blocks(env, drop)) {
                d->status = ST_WAITING_FOR_SPACE;
                return;
            }
            int32_t p = new_entity(env, K_PILE, drop, 0, 1);
            if (p < 0) {
                d->status = ST_WAITING_FOR_SPACE;
                return;
            }
            env->entities[p].pile.item = d->held;
            env->entities[p].pile.count = 1;
            d = &env->entities[index];
            d->held = IT_NONE;
        }
    }
    int32_t resource = drill_resource(env, d);
    if (resource < 0) {
        /* No fuel shows before no ore (probe_logistics2 `fill_coal_drill`). */
        d->status = d->energy <= 0 && d->remaining <= 0 && d->fuel.count == 0 ? ST_NO_FUEL
                                                                                 : ST_NO_MINABLE;
        return;
    }
    double fraction = burner_work(d, DRILL_USAGE);
    int produced = 0;
    if (fraction > 0) {
        fsim_resource *r = &env->resources[resource];
        double duration = mining_time_of_item(r->item);
        d->seconds += fraction * DRILL_SPEED / 60.0;
        d->progress = d->seconds / duration;
        if (d->progress >= 1.0) {
            d->progress -= 1.0;
            d->seconds = d->progress * duration;
            d->held = r->item;
            env->produced[r->item] += 1;
            r->amount -= 1;
            if (r->amount <= 0) r->alive = 0;
            produced = 1;
            if (++d->mine_count >= ORE_PER_TILE) {
                d->mine_count = 0;
                d->mine_cursor = (d->mine_cursor + 1) % 4;
            }
        }
    }
    burner_refill(env, d);
    if (d->energy <= 0) {
        d->status = ST_NO_FUEL;
    } else if (produced && target >= 0 && env->entities[target].unit != d->linked_unit) {
        /* Only an output into a machine it has not delivered to before shows
         * as waiting; after that the drill keeps working while it delivers. */
        d->status = ST_WAITING_FOR_SPACE;
    } else if (produced && target < 0 && belt < 0 && pile_blocks(env, drop)) {
        d->status = ST_WAITING_FOR_SPACE;
    } else {
        d->status = ST_WORKING;
    }
}

/* What a furnace makes from `item`, or IT_NONE. */
static int32_t smelt_product(int32_t item) {
    switch (item) {
    case IT_IRON_ORE: return IT_IRON_PLATE;
    case IT_COPPER_ORE: return IT_COPPER_PLATE;
    default: return IT_NONE;
    }
}

static int furnace_can_start(const fsim_entity *f) {
    if (f->source.count < 1) return 0;
    int32_t product = smelt_product(f->source.item);
    if (product == IT_NONE) return 0;
    return slot_room(&f->result, product, STACK_SIZE[product]) >= 1;
}

static void update_furnace(fsim_env *env, int32_t index) {
    fsim_entity *f = &env->entities[index];
    if (!f->crafting) {
        if (!furnace_can_start(f)) {
            f->status = (f->energy <= 0 && f->remaining <= 0 && f->fuel.count == 0)
                        ? ST_NO_FUEL : ST_NO_INGREDIENTS;
            return;
        }
        if (f->energy <= 0) {
            /* The first tick with something to smelt only loads fuel. */
            burner_refill(env, f);
            f->status = f->energy > 0 ? ST_WORKING : ST_NO_FUEL;
            return;
        }
        f->ingredient = f->source.item;
        slot_remove(&f->source, f->source.item, 1);
        f->crafting = 1;
        wake(env, index);
    }
    double fraction = burner_work(f, FURNACE_USAGE);
    if (fraction > 0) {
        f->seconds += fraction / 60.0;
        f->progress = f->seconds / SMELT_SECONDS;
        if (f->progress >= 1.0) {
            int32_t product = smelt_product(f->ingredient);
            slot_insert(&f->result, product, 1, STACK_SIZE[product]);
            env->produced[product] += 1;
            env->plates_crafted += 1;
            if (env->plates_crafted == STEAM_POWER_PLATES && !env->steam_power_at)
                env->steam_power_at = env->tick + STEAM_POWER_LAG;
            f->products_finished += 1;
            f->progress -= 1.0;
            f->seconds = f->progress * SMELT_SECONDS;
            if (furnace_can_start(f)) {
                f->ingredient = f->source.item;
                slot_remove(&f->source, f->source.item, 1);
            } else {
                f->crafting = 0;
                f->progress = 0;
                f->seconds = 0;
            }
            wake(env, index);
        }
    }
    if (f->crafting) {
        burner_refill(env, f);
        f->status = f->energy > 0 ? ST_WORKING : ST_NO_FUEL;
    } else {
        f->status = ST_NO_INGREDIENTS;
    }
}

/* Return a picked-up machine's contents, then the machine. */
static void pick_up(fsim_env *env, int32_t index) {
    fsim_entity *e = &env->entities[index];
    if (e->fuel.count > 0) insert_main(env, e->fuel.item, e->fuel.count);
    if (e->kind == K_FURNACE) {
        /* The ingredient of a craft in progress comes back with the source. */
        int32_t source = e->source.count;
        int32_t source_item = e->source.count ? e->source.item : e->ingredient;
        if (e->crafting) {
            if (source > 0 && e->source.item != e->ingredient) insert_main(env, e->ingredient, 1);
            else source += 1;
        }
        if (source > 0) insert_main(env, source_item, source);
        if (e->result.count > 0) insert_main(env, e->result.item, e->result.count);
    }
    /* The order FactorioRL's probe_logistics2 read off the character's slots
     * (`mine`): a chest's slots in order, then the chest; a belt's lane 1
     * then lane 2, front to back, then the belt; an inserter's fuel, the
     * inserter, then what it held. */
    if (e->kind == K_CHEST)
        for (int i = 0; i < FSIM_CHEST_SLOTS; i++)
            if (e->chest[i].count > 0) insert_main(env, e->chest[i].item, e->chest[i].count);
    if (e->kind == K_BELT)
        for (int lane = 0; lane < 2; lane++)
            for (int32_t k = 0; k < e->lanes[lane].count; k++)
                insert_main(env, e->lanes[lane].items[k].item, 1);
    if (e->kind == K_PILE) {
        insert_main(env, e->pile.item, e->pile.count);
        unblock_drills(env, -1, e->pos);
    } else {
        insert_main(env, entity_item(e->kind), 1);
    }
    /* An inserter's hand after the inserter (probe_logistics2 `mine`). */
    if (e->kind == K_INSERTER && e->held != IT_NONE) insert_main(env, e->held, 1);
    destroy_entity(env, index);
}

/* A belt carries the character standing or walking on it: after its own
 * step, if its centre is on a straight belt's tile, it moves 8/256 the belt's
 * way, sliding round what it runs into like a walking step
 * (probe_logistics2: standing 8 a tick, walking with the belt 38 + 8, against
 * it 38 - 8, across it 8 sideways until the step has left the tile, lanes
 * alike; carried off a belt end it stops once its centre is on the edge; into
 * a chest it slides along it, 8 a tick). On a turn it swings round the inner
 * corner at a rate that depends on where it stands, which is not reduced to a
 * rule yet (`char_rturn`, `char_lturn`): not modelled. */
static void carry_character(fsim_env *env) {
    int32_t b = belt_at(env, env->char_pos);
    if (b < 0 || env->entities[b].shape != BELT_STRAIGHT) return;
    move_character(env, env->entities[b].direction, BELT_SPEED);
}

static void update_character(fsim_env *env) {
    env->walk_pub = env->walk_set;
    if (env->walk_set) env->walk_pub_dir = env->walk_set_dir;
    if (env->walk_pub) {
        env->char_dir8 = env->walk_pub_dir;
        walk_one_tick(env, env->walk_pub_dir);
    }
    if (env->chain_count) carry_character(env);
    if (!env->mining) return;
    double duration;
    int32_t item;
    if (env->mining_target_resource >= 0) {
        fsim_resource *r = &env->resources[env->mining_target_resource];
        if (!r->alive) return;
        duration = mining_time_of_item(r->item);
        item = r->item;
    } else if (env->mining_target_entity >= 0) {
        fsim_entity *e = &env->entities[env->mining_target_entity];
        if (!e->alive) return;
        duration = entity_mining_time(e->kind);
        item = IT_NONE;
    } else {
        return;
    }
    face(env, env->mining_pos);
    env->mining_seconds += CHAR_MINING_SPEED / 60.0;
    env->mining_progress = env->mining_seconds / duration;
    if (env->mining_progress < 1.0) return;
    /* A finished item starts the next from nothing. */
    env->mining_progress = 0;
    env->mining_seconds = 0;
    if (env->mining_target_resource >= 0) {
        fsim_resource *r = &env->resources[env->mining_target_resource];
        insert_main(env, item, 1);
        env->produced[item] += 1;
        r->amount -= 1;
        if (r->amount <= 0) {
            r->alive = 0;
            if (env->selected_kind == 2 && env->selected_index == env->mining_target_resource)
                env->selected_kind = 0;
        }
    } else {
        pick_up(env, env->mining_target_entity);
    }
}

/* One tick, in the engine's order as the probes read it off same-tick
 * evidence (docs/sim-logistics.md):
 *
 * 1. the character;
 * 2. belts: an item a drill or an inserter puts on a lane moves once as it
 *    lands, and the line it lands on has already moved (a drill output onto a
 *    just-extended line lands 64 behind the item ahead as it stands after the
 *    move); an inserter reacts to an item arriving on its pickup tile in the
 *    tick it arrives;
 * 3. inserters, the last built first (FactorioRL tools/probe_logistics2.py,
 *    `order_*`: two and four inserters reaching one plate, or one free slot,
 *    on the same tick), and all before drills and furnaces, since a drill's
 *    output into a chest, a furnace's product and room made in a furnace's
 *    source are all acted on a tick later, whichever was built first;
 * 4. drills and furnaces in reverse creation order: a furnace sees ore a
 *    drill delivered this tick only on the next one. */
static void update_world(fsim_env *env) {
    if (env->steam_power_at && env->tick >= env->steam_power_at) env->steam_power = 1;
    /* Belt shapes first: a belt built this step carries the character. */
    if (env->logistics_version != env->entities_version) rebuild_logistics(env);
    update_character(env);
    if (env->logistics_version != env->entities_version) rebuild_logistics(env);
    if (env->chain_count) update_belts(env);
    for (int32_t k = env->inserter_count - 1; k >= 0; k--) {
        int32_t i = env->inserters[k];
        if (env->entities[i].alive) update_inserter(env, i);
    }
    for (int32_t i = env->entity_count - 1; i >= 0; i--) {
        fsim_entity *e = &env->entities[i];
        if (!e->alive) continue;
        switch (e->kind) {
        case K_FURNACE: update_furnace(env, i); break;
        case K_DRILL: update_drill(env, i); break;
        default: break;
        }
    }
}

/* ------------------------------------------------------------------ inflight */

static int32_t inflight_start(fsim_env *env, int32_t verb, int32_t step) {
    int32_t slot = -1;
    for (int32_t i = 0; i < FSIM_MAX_INFLIGHT; i++) {
        if (!env->inflight[i].used || env->inflight[i].terminal) {
            slot = i;
            break;
        }
    }
    if (slot < 0) abort();
    fsim_inflight *f = &env->inflight[slot];
    memset(f, 0, sizeof(*f));
    f->used = 1;
    f->step = step;
    f->seq = ++env->next_seq;
    f->verb = verb;
    f->started_tick = env->tick;
    f->deadline_tick = -1;
    if (verb == V_MOVE) env->slot_move = slot;
    else if (verb == V_MINE) env->slot_mine = slot;
    else env->slot_advance = slot;
    return slot;
}

static int32_t occupant(fsim_env *env, int32_t verb) {
    int32_t slot = verb == V_MOVE ? env->slot_move : verb == V_MINE ? env->slot_mine : env->slot_advance;
    if (slot < 0) return -1;
    if (env->inflight[slot].terminal || !env->inflight[slot].used) {
        if (verb == V_MOVE) env->slot_move = -1;
        else if (verb == V_MINE) env->slot_mine = -1;
        else env->slot_advance = -1;
        return -1;
    }
    return slot;
}

static void finish(fsim_env *env, int32_t slot) {
    env->inflight[slot].terminal = 1;
    if (env->slot_move == slot) env->slot_move = -1;
    if (env->slot_mine == slot) env->slot_mine = -1;
    if (env->slot_advance == slot) env->slot_advance = -1;
}

/* Stopping keeps the progress: mining the same target again resumes it. */
static void stop_mining(fsim_env *env) {
    env->mining = 0;
    env->mining_pos.x = 0;
    env->mining_pos.y = 0;
    env->mining_target_entity = -1;
    env->mining_target_resource = -1;
}

static void start_mining(fsim_env *env, int32_t kind, int32_t index, fsim_pos position) {
    if (kind != env->mined_kind || index != env->mined_index) {
        env->mining_seconds = 0;
        env->mining_progress = 0;
    }
    env->mined_kind = kind;
    env->mined_index = index;
    env->mining = 1;
    env->mining_pos = position;
    env->mining_target_resource = kind == 2 ? index : -1;
    env->mining_target_entity = kind == 1 ? index : -1;
}

static void cancel(fsim_env *env, int32_t slot) {
    fsim_inflight *f = &env->inflight[slot];
    if (f->verb == V_MOVE) env->walk_set = 0;
    else if (f->verb == V_MINE) stop_mining(env);
    f->result = R_CANCELLED;
    finish(env, slot);
}

/* ------------------------------------------------------------------ actions */

static int32_t reject(fsim_env *env, int32_t error) {
    env->act.status = R_REJECTED;
    env->act.error = error;
    return error;
}

static int32_t act_move(fsim_env *env, const fsim_action *a) {
    for (int k = 0; k < 2; k++) {
        int32_t slot = occupant(env, k == 0 ? V_MOVE : V_MINE);
        if (slot >= 0) {
            cancel(env, slot);
            /* Kept by value: the slot is free for the move about to start. */
            if (env->superseded_count < FSIM_MAX_SUPERSEDED)
                env->superseded[env->superseded_count++] =
                    env->inflight[slot].step * 16 + env->inflight[slot].verb;
        }
    }
    int32_t dir16 = a->direction * 4;
    env->walk_set = 1;
    env->walk_set_dir = dir16;
    int32_t slot = inflight_start(env, V_MOVE, env->act_step);
    env->inflight[slot].deadline_tick = env->tick + a->ticks;
    env->inflight[slot].move_dir = dir16;
    env->act_inflight = slot;
    env->act.status = R_RUNNING;
    return 0;
}

static int32_t act_mine(fsim_env *env, const fsim_action *a) {
    if (occupant(env, V_MINE) >= 0) return reject(env, E_BUSY);
    int32_t kind, index;
    int32_t code = fsim_resolve(env, a->handle, &kind, &index);
    if (code) return reject(env, code);
    int32_t item;
    fsim_pos position;
    if (kind == 2) {
        const fsim_resource *r = &env->resources[index];
        position = resource_pos(r);
        if (centre_distance(env->char_pos, position) > RESOURCE_REACH)
            return reject(env, E_OUT_OF_REACH);
        item = r->item;
    } else {
        const fsim_entity *e = &env->entities[index];
        if (!can_reach(env, 1, index)) return reject(env, E_OUT_OF_REACH);
        position = e->pos;
        item = e->kind == K_PILE ? e->pile.item : entity_item(e->kind);
    }
    if (empty_slots(env) == 0) return reject(env, E_NO_SPACE);
    env->selected_kind = kind;
    env->selected_index = index;
    start_mining(env, kind, index, position);
    int32_t slot = inflight_start(env, V_MINE, env->act_step);
    fsim_inflight *f = &env->inflight[slot];
    f->target = a->handle;
    f->goal_item = item;
    f->goal_count = a->count;
    f->baseline = count_main(env, item);
    f->queued = 0;
    env->act_inflight = slot;
    env->act.status = R_RUNNING;
    return 0;
}

static int placeable(const fsim_env *env, int32_t kind, fsim_pos centre) {
    int32_t r = box_of(kind);
    for (int32_t i = 0; i < env->entity_count; i++) {
        const fsim_entity *e = &env->entities[i];
        if (!e->alive || !has_flag(e->kind, KF_COLLIDES)) continue;
        int32_t reach = box_of(e->kind) + r;
        if (abs(e->pos.x - centre.x) < reach && abs(e->pos.y - centre.y) < reach) return 0;
    }
    int32_t reach = CHAR_BOX + r;
    if (has_flag(kind, KF_BLOCKS_WALKING) && abs(env->char_pos.x - centre.x) < reach &&
        abs(env->char_pos.y - centre.y) < reach)
        return 0;
    if (box_hits_water(env, centre, r, 0)) return 0;
    if (kind == K_DRILL) {
        int found = 0;
        for (int32_t i = 0; i < env->resource_count && !found; i++) {
            const fsim_resource *res = &env->resources[i];
            if (!res->alive) continue;
            fsim_pos p = resource_pos(res);
            if (abs(p.x - centre.x) <= DRILL_AREA && abs(p.y - centre.y) <= DRILL_AREA) found = 1;
        }
        if (!found) return 0;
    }
    return 1;
}

static int32_t act_place(fsim_env *env, const fsim_action *a) {
    if (count_main(env, a->item) < 1) return reject(env, E_NO_ITEMS);
    int32_t kind = kind_placed_by(a->item);
    if (kind == K_NONE) return reject(env, E_INVALID_TARGET);
    if (centre_distance(env->char_pos, a->position) > BUILD_DISTANCE)
        return reject(env, E_OUT_OF_REACH);
    fsim_pos centre = {snap(kind, a->position.x), snap(kind, a->position.y)};
    if (!placeable(env, kind, centre)) return reject(env, E_COLLISION);
    /* Not the engine's limit but the simulator's (fsim.h). */
    if (env->entity_count >= FSIM_MAX_ENTITIES) return reject(env, E_ENGINE);
    int32_t direction = has_flag(kind, KF_DIRECTED) ? a->direction * 4 : 0;
    int32_t index = new_entity(env, kind, centre, direction, 0);
    remove_main(env, a->item, 1);
    env->built[a->item] += 1;
    env->act.handle = mint_unit(env, index);
    env->act.position = centre;
    env->act.status = R_COMPLETED;
    return 0;
}

static int32_t act_rotate(fsim_env *env, const fsim_action *a) {
    int32_t kind, index;
    int32_t code = fsim_resolve(env, a->handle, &kind, &index);
    if (code) return reject(env, code);
    if (!can_reach(env, kind, index)) return reject(env, E_OUT_OF_REACH);
    if (kind != 1 || !has_flag(env->entities[index].kind, KF_DIRECTED))
        return reject(env, E_INVALID_TARGET);
    fsim_entity *d = &env->entities[index];
    d->direction = (d->direction + (a->reverse ? 12 : 4)) % 16;
    if (d->kind == K_BELT || d->kind == K_INSERTER) {
        /* New links and targets (rebuild_logistics). Not measured: where a
         * turned belt's items go when its lane lengths change -- the engine
         * re-places them (a loaded turn rotated back to straight in
         * FactorioRL's logistics_belt_rotate_and_mine trace) by a rule this
         * does not have -- and what a turned inserter does with a swing in
         * progress: it carries on. */
        env->entities_version++;
        if (d->kind == K_BELT) env->belts_changed = 1;
    }
    env->act.status = R_COMPLETED;
    return 0;
}

/* Which of a machine's slots a transfer reads from or writes to, in the
 * mod's order: fuel, then source, then result. */
static fsim_stack *machine_slot(fsim_entity *m, int32_t item, int removing, int32_t *cap) {
    fsim_stack *order[3];
    int32_t caps[3];
    int n = 0;
    if (has_flag(m->kind, KF_BURNER)) {
        order[n] = &m->fuel;
        caps[n++] = STACK_SIZE[item];
    }
    if (m->kind == K_FURNACE) {
        order[n] = &m->source;
        caps[n++] = item == IT_IRON_ORE ? FURNACE_SOURCE_CAP : STACK_SIZE[item];
        order[n] = &m->result;
        caps[n++] = STACK_SIZE[item];
    }
    for (int i = 0; i < n; i++) {
        fsim_stack *s = order[i];
        int accepts;
        if (removing) {
            accepts = s->count > 0 && s->item == item;
        } else if (s == &m->fuel) {
            accepts = is_fuel(item) && slot_room(s, item, caps[i]) > 0;
        } else if (s == &m->source) {
            accepts = (item == IT_IRON_ORE || item == IT_COPPER_ORE || item == IT_STONE) &&
                      slot_room(s, item, caps[i]) > 0;
        } else {
            accepts = slot_room(s, item, caps[i]) > 0;
        }
        if (accepts) {
            *cap = caps[i];
            return s;
        }
    }
    *cap = STACK_SIZE[item];
    return &m->fuel;
}

static int32_t act_transfer(fsim_env *env, const fsim_action *a) {
    if (a->from_handle == 0 && a->to_handle == 0) return reject(env, E_PRECONDITION);
    int32_t ends[2] = {a->from_handle, a->to_handle};
    int32_t kinds[2] = {0, 0}, indices[2] = {-1, -1};
    for (int k = 0; k < 2; k++) {
        if (ends[k] == 0) continue;
        int32_t code = fsim_resolve(env, ends[k], &kinds[k], &indices[k]);
        if (code) return reject(env, code);
        if (!can_reach(env, kinds[k], indices[k])) return reject(env, E_OUT_OF_REACH);
    }
    for (int k = 0; k < 2; k++) {
        if (ends[k] == 0) continue;
        if (kinds[k] != 1) return reject(env, E_PRECONDITION);
        if (!has_flag(env->entities[indices[k]].kind, KF_MACHINE))
            return reject(env, E_PRECONDITION);
    }
    int32_t item = a->item;
    int32_t available;
    fsim_stack *from_slot = NULL;
    int32_t from_cap = 0;
    fsim_entity *chests[2] = {NULL, NULL};
    for (int k = 0; k < 2; k++)
        if (ends[k] != 0 && env->entities[indices[k]].kind == K_CHEST)
            chests[k] = &env->entities[indices[k]];
    if (ends[0] == 0) {
        available = count_main(env, item);
    } else if (chests[0]) {
        available = chest_count(chests[0], item);
    } else {
        from_slot = machine_slot(&env->entities[indices[0]], item, 1, &from_cap);
        available = (from_slot->item == item) ? from_slot->count : 0;
    }
    if (available <= 0) return reject(env, E_NO_ITEMS);
    int32_t wanted = a->count < available ? a->count : available;
    fsim_stack *to_slot = NULL;
    int32_t to_cap = 0;
    if (ends[1] == 0) {
        if (main_room(env, item) <= 0) return reject(env, E_NO_SPACE);
    } else if (chests[1]) {
        if (chest_room(chests[1], item) <= 0) return reject(env, E_NO_SPACE);
    } else {
        to_slot = machine_slot(&env->entities[indices[1]], item, 0, &to_cap);
        int accepts = slot_room(to_slot, item, to_cap) > 0;
        if (to_slot == &env->entities[indices[1]].fuel && !is_fuel(item))
            accepts = 0;
        if (!accepts) return reject(env, E_NO_SPACE);
    }
    int32_t removed = ends[0] == 0 ? remove_main(env, item, wanted)
                    : chests[0]    ? chest_remove(chests[0], item, wanted)
                                   : slot_remove(from_slot, item, wanted);
    if (removed <= 0) return reject(env, E_NO_ITEMS);
    int32_t inserted = ends[1] == 0 ? insert_main(env, item, removed)
                     : chests[1]    ? chest_insert(chests[1], item, removed)
                                    : slot_insert(to_slot, item, removed, to_cap);
    if (inserted < removed) {
        /* What did not fit goes back; a chest's may land in other slots than
         * it left (not measured). */
        int32_t back = removed - inserted;
        if (ends[0] == 0) insert_main(env, item, back);
        else if (chests[0]) chest_insert(chests[0], item, back);
        else slot_insert(from_slot, item, back, from_cap);
        return reject(env, E_NO_SPACE);
    }
    for (int k = 0; k < 2; k++)
        if (ends[k] != 0) wake(env, indices[k]);
    if (chests[0]) unblock_drills(env, indices[0], chests[0]->pos);
    env->transfers += 1;
    env->items_moved += inserted;
    env->act.count = inserted;
    if (inserted != a->count) {
        env->act.requested = a->count;
        env->act.available = available;
    }
    env->act.status = R_COMPLETED;
    return 0;
}

static void dispatch(fsim_env *env, const fsim_action *a) {
    int32_t guard_slot = -1;
    int32_t guard_held = 0;
    if (a->verb != V_MINE) {
        int32_t slot = occupant(env, V_MINE);
        if (slot >= 0) {
            guard_slot = slot;
            guard_held = count_main(env, env->inflight[slot].goal_item);
        }
    }
    switch (a->verb) {
    case V_MOVE: act_move(env, a); break;
    case V_MINE: act_mine(env, a); break;
    case V_PLACE: act_place(env, a); break;
    case V_ROTATE: act_rotate(env, a); break;
    case V_TRANSFER: act_transfer(env, a); break;
    default: env->act.status = R_COMPLETED; break;
    }
    if (guard_slot >= 0 && !env->inflight[guard_slot].terminal) {
        fsim_inflight *f = &env->inflight[guard_slot];
        f->baseline += count_main(env, f->goal_item) - guard_held;
    }
}

/* ------------------------------------------------------------------ polling */

typedef struct {
    int32_t slot;
    int32_t status;
    int32_t error;
} settled_t;

static int poll(fsim_env *env, int32_t slot, settled_t *out) {
    fsim_inflight *f = &env->inflight[slot];
    out->slot = slot;
    out->error = E_NONE;
    if (f->verb == V_MOVE) {
        if (env->tick >= f->deadline_tick) {
            env->walk_set = 0;
            out->status = R_COMPLETED;
            return 1;
        }
        return 0;
    }
    if (f->verb == V_MINE) {
        int32_t gained = count_main(env, f->goal_item) - f->baseline;
        int32_t kind, index;
        int32_t code = fsim_resolve(env, f->target, &kind, &index);
        if (code) {
            stop_mining(env);
            if (gained > 0) env->mined_by_action[f->goal_item] += gained;
            if (gained >= f->goal_count) {
                out->status = R_COMPLETED;
            } else {
                out->status = R_FAILED;
                out->error = code;
            }
            return 1;
        }
        if (gained >= f->goal_count) {
            stop_mining(env);
            if (gained > 0) env->mined_by_action[f->goal_item] += gained;
            out->status = R_COMPLETED;
            return 1;
        }
        env->selected_kind = kind;
        env->selected_index = index;
        start_mining(env, kind, index,
                     kind == 2 ? resource_pos(&env->resources[index]) : env->entities[index].pos);
        return 0;
    }
    if (env->tick >= f->deadline_tick) {
        out->status = R_COMPLETED;
        return 1;
    }
    return 0;
}

static void poll_pass(fsim_env *env);

static void tick_once(fsim_env *env) {
    env->tick += 1;
    update_world(env);
    poll_pass(env);
}

/* The mod's per-tick pass: poll running operations in start order, settle
 * what finished, then settle what an action superseded. It also runs once on
 * the tick a step arrives, before the world moves. */
static void poll_pass(fsim_env *env) {

    int32_t order[FSIM_MAX_INFLIGHT];
    int32_t n = 0;
    for (int32_t i = 0; i < FSIM_MAX_INFLIGHT; i++)
        if (env->inflight[i].used && !env->inflight[i].terminal) order[n++] = i;
    for (int32_t i = 1; i < n; i++) {
        int32_t v = order[i];
        int32_t j = i - 1;
        while (j >= 0 && env->inflight[order[j]].seq > env->inflight[v].seq) {
            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = v;
    }
    settled_t settled[FSIM_MAX_INFLIGHT];
    int32_t m = 0;
    for (int32_t i = 0; i < n; i++) {
        if (poll(env, order[i], &settled[m])) m++;
    }
    for (int32_t i = 0; i < m; i++) finish(env, settled[i].slot);

    int advance_done = 0;
    for (int32_t i = 0; i < m; i++) {
        fsim_inflight *f = &env->inflight[settled[i].slot];
        f->result = settled[i].status;
        if (f->verb < 0) {
            advance_done = 1;
            /* The step's own event: its action as it stands now. */
            int32_t action = env->act.status == R_REJECTED ? -1 : env->act.verb;
            record_event(env, f->step, 0, R_COMPLETED, action, env->act.status,
                         env->act.status == R_REJECTED ? env->act.error : E_NONE);
        } else {
            if (settled[i].slot == env->act_inflight && f->step == env->act_step)
                env->act.status = settled[i].status;
            record_event(env, f->step, 1, settled[i].status, f->verb, R_NONE, settled[i].error);
        }
    }
    for (int32_t i = 0; i < env->superseded_count; i++) {
        int32_t packed = env->superseded[i];
        record_event(env, packed / 16, 1, R_CANCELLED, packed % 16, R_NONE, E_NONE);
    }
    env->superseded_count = 0;
    if (advance_done) fsim_observe(env);
}

int64_t fsim_step(fsim_env *env, const fsim_action *action, int32_t ticks) {
    env->step_counter += 1;
    env->act_step = env->step_counter;
    memset(&env->act, 0, sizeof(env->act));
    env->act.verb = action->verb;
    env->act_inflight = -1;
    dispatch(env, action);
    int32_t slot = inflight_start(env, -1, env->act_step);
    env->inflight[slot].deadline_tick = env->tick + ticks;
    poll_pass(env);
    for (int32_t i = 0; i < ticks; i++) tick_once(env);
    return env->tick;
}

/* ------------------------------------------------------------------ observe */

typedef struct {
    int64_t d2;
    int32_t y, x, rank, index;
} sweep_item;

static int sweep_cmp(const void *pa, const void *pb) {
    const sweep_item *a = pa, *b = pb;
    if (a->d2 != b->d2) return a->d2 < b->d2 ? -1 : 1;
    if (a->y != b->y) return a->y < b->y ? -1 : 1;
    if (a->x != b->x) return a->x < b->x ? -1 : 1;
    if (a->rank != b->rank) return a->rank < b->rank ? -1 : 1;
    return a->index < b->index ? -1 : (a->index > b->index);
}

/* "h12" against "h9", as Lua compares strings. */
static int handle_string_cmp(int32_t a, int32_t b) {
    char sa[16], sb[16];
    int la = 0, lb = 0;
    int32_t v = a;
    char tmp[16];
    int t = 0;
    do { tmp[t++] = (char)('0' + v % 10); v /= 10; } while (v);
    while (t) sa[la++] = tmp[--t];
    sa[la] = 0;
    v = b;
    t = 0;
    do { tmp[t++] = (char)('0' + v % 10); v /= 10; } while (v);
    while (t) sb[lb++] = tmp[--t];
    sb[lb] = 0;
    return strcmp(sa, sb);
}

static int remembered_before(const fsim_memory *a, const fsim_memory *b) {
    if (a->pos.y != b->pos.y) return a->pos.y < b->pos.y;
    if (a->pos.x != b->pos.x) return a->pos.x < b->pos.x;
    int ra = name_rank(a->kind), rb = name_rank(b->kind);
    if (ra != rb) return ra < rb;
    return handle_string_cmp(a->handle, b->handle) < 0;
}

/* The one stack an observation shows as an entity's contents. */
static fsim_stack shown_contents(const fsim_entity *e) {
    fsim_stack none = {IT_NONE, 0};
    switch (e->kind) {
    case K_FURNACE: return e->source;
    case K_PILE: return e->pile;
    case K_CHEST: {
        /* The first item and how many of it; the rest is in a memory
         * record's `amounts` and in contents_total. */
        for (int i = 0; i < FSIM_CHEST_SLOTS; i++)
            if (e->chest[i].count > 0) {
                fsim_stack first = {e->chest[i].item, chest_count(e, e->chest[i].item)};
                return first;
            }
        return none;
    }
    default: return none;
    }
}

/* Everything an observation's `contents` counts: a chest's whole inventory. */
static int32_t contents_total(const fsim_entity *e) {
    if (e->kind != K_CHEST) return shown_contents(e).count;
    int32_t total = 0;
    for (int i = 0; i < FSIM_CHEST_SLOTS; i++) total += e->chest[i].count;
    return total;
}

static void remember(fsim_env *env, int32_t handle, const fsim_entity *e) {
    /* Slots from memory_top up are unused, so the scan stops there. */
    int32_t free_slot = -1;
    for (int32_t i = 0; i < env->memory_top; i++) {
        if (env->memory[i].used && env->memory[i].handle == handle) {
            free_slot = i;
            break;
        }
        if (!env->memory[i].used && free_slot < 0) free_slot = i;
    }
    if (free_slot < 0) {
        if (env->memory_top >= FSIM_MAX_MEMORY) return;
        free_slot = env->memory_top++;
    }
    fsim_memory *m = &env->memory[free_slot];
    m->used = 1;
    m->handle = handle;
    m->kind = e->kind;
    m->pos = e->pos;
    m->has_dir = has_flag(e->kind, KF_DIRECTED);
    m->direction = e->direction;
    m->contents = shown_contents(e);
    /* The mod remembers a record's whole `contents`: for a chest every item
     * it holds, as `get_contents` totals them (mod/factoriorl/memory.lua). */
    memset(m->amounts, 0, sizeof(m->amounts));
    if (e->kind == K_CHEST)
        for (int i = 0; i < FSIM_CHEST_SLOTS; i++)
            if (e->chest[i].count > 0) m->amounts[e->chest[i].item] += (uint16_t)e->chest[i].count;
    m->last_seen = env->tick;
}

void fsim_observe(fsim_env *env) {
    fsim_pos o = env->char_pos;
    env->origin = o;
    const int64_t r_entities = (int64_t)SENSOR_RADIUS * TILE;
    sweep_item items[FSIM_MAX_ENTITIES > FSIM_MAX_RESOURCES ? FSIM_MAX_ENTITIES : FSIM_MAX_RESOURCES];
    int32_t n = 0;
    for (int32_t i = 0; i < env->entity_count; i++) {
        const fsim_entity *e = &env->entities[i];
        if (!e->alive) continue;
        int64_t dx = e->pos.x - o.x, dy = e->pos.y - o.y;
        int64_t d2 = dx * dx + dy * dy;
        if (d2 > r_entities * r_entities) continue;
        items[n].d2 = d2;
        items[n].y = e->pos.y;
        items[n].x = e->pos.x;
        items[n].rank = name_rank(e->kind);
        items[n].index = i;
        n++;
    }
    qsort(items, (size_t)n, sizeof(items[0]), sweep_cmp);
    if (n > FSIM_MAX_SWEEP) n = FSIM_MAX_SWEEP;
    env->seen_count = n;
    for (int32_t k = 0; k < n; k++) {
        int32_t i = items[k].index;
        const fsim_entity *e = &env->entities[i];
        int32_t h = e->kind == K_PILE
            ? mint_tile(env, (int32_t)floordiv(e->pos.x, TILE), (int32_t)floordiv(e->pos.y, TILE),
                        TT_PILE, K_PILE)
            : mint_unit(env, i);
        env->seen[k].handle = h;
        env->seen[k].entity = i;
        env->seen[k].d2 = items[k].d2;
    }

    /* memory */
    for (int32_t k = 0; k < n; k++)
        remember(env, env->seen[k].handle, &env->entities[env->seen[k].entity]);
    env->remembered_count = 0;
    for (int32_t i = 0; i < env->memory_top; i++) {
        fsim_memory *m = &env->memory[i];
        if (!m->used) continue;
        int seen = 0;
        for (int32_t k = 0; k < n; k++)
            if (env->seen[k].handle == m->handle) seen = 1;
        if (seen) continue;
        double d = centre_distance(m->pos, o);
        if (d <= SENSOR_RADIUS) {
            m->used = 0;
        } else {
            env->remembered[env->remembered_count++] = i;
        }
    }
    /* Insertion sort: a handful of records, and no shared comparator state,
     * so environments can step on separate threads. */
    for (int32_t i = 1; i < env->remembered_count; i++) {
        int32_t v = env->remembered[i];
        int32_t j = i - 1;
        while (j >= 0 && remembered_before(&env->memory[v], &env->memory[env->remembered[j]])) {
            env->remembered[j + 1] = env->remembered[j];
            j--;
        }
        env->remembered[j + 1] = v;
    }

    /* resource tiles */
    const int64_t r_tiles = (int64_t)RESOURCE_RADIUS * TILE;
    int32_t t = 0;
    for (int32_t i = 0; i < env->resource_count; i++) {
        const fsim_resource *r = &env->resources[i];
        if (!r->alive) continue;
        fsim_pos p = resource_pos(r);
        int64_t dx = p.x - o.x, dy = p.y - o.y;
        int64_t d2 = dx * dx + dy * dy;
        if (d2 > r_tiles * r_tiles) continue;
        items[t].d2 = d2;
        items[t].y = p.y;
        items[t].x = p.x;
        items[t].rank = r->item;
        items[t].index = i;
        t++;
    }
    qsort(items, (size_t)t, sizeof(items[0]), sweep_cmp);
    if (t > FSIM_MAX_TILES) t = FSIM_MAX_TILES;
    env->tile_count = t;
    for (int32_t k = 0; k < t; k++) {
        const fsim_resource *r = &env->resources[items[k].index];
        env->tiles[k].handle = mint_tile(env, r->tx, r->ty, TT_RESOURCE, r->item);
        env->tiles[k].resource = items[k].index;
    }

    /* water in the sensor square */
    int64_t tx0 = floordiv(o.x - SENSOR_RADIUS * TILE, TILE);
    int64_t tx1 = -floordiv(-(int64_t)(o.x + SENSOR_RADIUS * TILE), TILE) - 1;
    int64_t ty0 = floordiv(o.y - SENSOR_RADIUS * TILE, TILE);
    int64_t ty1 = -floordiv(-(int64_t)(o.y + SENSOR_RADIUS * TILE), TILE) - 1;
    int32_t b = 0;
    for (int32_t i = 0; i < env->water_count && b < FSIM_MAX_BLOCKED; i++) {
        int32_t wx = env->water[2 * i], wy = env->water[2 * i + 1];
        if (wx < tx0 || wx > tx1 || wy < ty0 || wy > ty1) continue;
        env->blocked[2 * b] = wx;
        env->blocked[2 * b + 1] = wy;
        b++;
    }
    env->blocked_count = b;
}

/* ------------------------------------------------------------------ lifecycle */

fsim_env *fsim_new(void) {
    fsim_env *env = (fsim_env *)calloc(1, sizeof(fsim_env));
    return env;
}

void fsim_free(fsim_env *env) { free(env); }

void fsim_set_water(fsim_env *env, const int32_t *xy, int32_t count) {
    if (count > FSIM_MAX_WATER) count = FSIM_MAX_WATER;
    memcpy(env->water, xy, sizeof(int32_t) * 2 * (size_t)count);
    env->water_count = count;
}

void fsim_walk_ticks(fsim_env *env, int32_t dir16, int32_t ticks, int32_t *xy) {
    for (int32_t i = 0; i < ticks; i++) {
        walk_one_tick(env, dir16);
        xy[2 * i] = env->char_pos.x;
        xy[2 * i + 1] = env->char_pos.y;
    }
}

void fsim_after_load(fsim_env *env) {
    env->entities_version++;
    if (env->mining) {
        int32_t kind, index;
        env->mining_target_entity = -1;
        env->mining_target_resource = -1;
        int32_t slot = occupant(env, V_MINE);
        if (slot >= 0 && fsim_resolve(env, env->inflight[slot].target, &kind, &index) == 0) {
            if (kind == 2) env->mining_target_resource = index;
            else env->mining_target_entity = index;
        }
    }
}

void fsim_reset(fsim_env *env, const fsim_scene *scene) {
    /* Everything but the map's water, which outlives episodes. */
    int32_t water_count = env->water_count;
    size_t start = offsetof(fsim_env, water);
    size_t end = start + sizeof(env->water);
    memset(env, 0, start);
    memset((char *)env + end, 0, sizeof(*env) - end);
    env->water_count = water_count;
    env->next_handle = 1;
    env->slot_move = env->slot_mine = env->slot_advance = -1;
    env->mining_target_entity = env->mining_target_resource = -1;
    env->act_inflight = -1;

    for (int32_t i = 0; i < scene->resource_count; i++) {
        fsim_resource *r = &env->resources[env->resource_count++];
        r->alive = 1;
        r->item = scene->resource_item[i];
        r->tx = scene->resource_tx[i];
        r->ty = scene->resource_ty[i];
        r->amount = scene->resource_amount[i];
    }
    for (int32_t i = 0; i < scene->wall_count; i++) {
        fsim_pos p = {
            (int32_t)floordiv(scene->wall_x[i], TILE) * TILE + TILE / 2,
            (int32_t)floordiv(scene->wall_y[i], TILE) * TILE + TILE / 2,
        };
        new_entity(env, K_WALL, p, 0, 1);
    }
    /* Entities the scene places rather than the agent (fsim.h), then what
     * they are given. Empty, `new_entity` makes a drill or a furnace
     * ST_NO_FUEL. */
    int32_t placed[FSIM_MAX_ENTITIES];
    for (int32_t i = 0; i < scene->machine_count && i < FSIM_MAX_ENTITIES; i++) {
        fsim_pos p = {scene->machine_x[i], scene->machine_y[i]};
        placed[i] = new_entity(env, scene->machine_kind[i], p, scene->machine_dir[i], 0);
    }
    for (int32_t i = 0; i < scene->content_count; i++) {
        int32_t m = scene->content_machine[i];
        if (m >= 0 && m < scene->machine_count && m < FSIM_MAX_ENTITIES && placed[m] >= 0)
            fsim_entity_insert(env, placed[m], scene->content_item[i], scene->content_amount[i]);
    }
    env->char_pos = scene->character;
    for (int32_t i = 0; i < scene->inventory_count; i++)
        insert_main(env, scene->inventory_item[i], scene->inventory_amount[i]);
    fsim_refresh(env);
    fsim_observe(env);
}

/* ------------------------------------------------------------------ script */

int32_t fsim_add_entity(fsim_env *env, int32_t kind, int32_t x, int32_t y, int32_t dir16) {
    if (kind <= K_NONE || kind >= K_COUNT) return -1;
    fsim_pos p = {x, y};
    return new_entity(env, kind, p, has_flag(kind, KF_DIRECTED) ? dir16 : 0, 0);
}

int32_t fsim_entity_insert(fsim_env *env, int32_t index, int32_t item, int32_t count) {
    if (index < 0 || index >= env->entity_count || item <= IT_NONE || item >= IT_COUNT)
        return 0;
    fsim_entity *e = &env->entities[index];
    if (!e->alive || count <= 0) return 0;
    int32_t done;
    if (e->kind == K_CHEST) {
        done = chest_insert(e, item, count);
    } else if (has_flag(e->kind, KF_BURNER) && is_fuel(item)) {
        done = slot_insert(&e->fuel, item, count, STACK_SIZE[item]);
    } else if (e->kind == K_FURNACE && smelt_product(item) != IT_NONE) {
        int32_t cap = item == IT_IRON_ORE ? FURNACE_SOURCE_CAP : STACK_SIZE[item];
        done = slot_insert(&e->source, item, count, cap);
    } else if (e->kind == K_FURNACE && (item == IT_IRON_PLATE || item == IT_COPPER_PLATE)) {
        /* The mod names the result slot for a declared product. */
        done = slot_insert(&e->result, item, count, STACK_SIZE[item]);
    } else {
        return 0;
    }
    /* Status as the engine reads it before the first tick, where measured: a
     * drill given fuel is working (FactorioRL's logistics traces), a furnace
     * given fuel has no ingredients, with ore in it or not (those traces and
     * the logistics probe's `fout` furnace). */
    if (e->kind == K_DRILL && e->fuel.count > 0) e->status = ST_WORKING;
    if (e->kind == K_FURNACE && e->fuel.count > 0) e->status = ST_NO_INGREDIENTS;
    /* An inserter out of fuel reads working as soon as it has some
     * (probe_logistics2 `pe_*`, refuelled at t=1000). */
    if (e->kind == K_INSERTER && e->status == ST_NO_FUEL && e->fuel.count > 0)
        e->status = ST_WORKING;
    if (done > 0) wake(env, index);
    return done;
}

static int32_t script_lane(fsim_env *env, int32_t index, int32_t lane) {
    if (index < 0 || index >= env->entity_count || lane < 0 || lane > 1) return -1;
    if (!env->entities[index].alive || env->entities[index].kind != K_BELT) return -1;
    if (env->logistics_version != env->entities_version) rebuild_logistics(env);
    return index * 2 + lane;
}

int32_t fsim_belt_insert(fsim_env *env, int32_t index, int32_t lane, int32_t position,
                         int32_t item) {
    int32_t ref = script_lane(env, index, lane);
    if (ref < 0 || item <= IT_NONE || item >= IT_COUNT) return 0;
    return lane_insert(env, ref, position, item, new_item_id(env), 0);
}

int32_t fsim_belt_insert_back(fsim_env *env, int32_t index, int32_t lane, int32_t item) {
    int32_t ref = script_lane(env, index, lane);
    if (ref < 0 || item <= IT_NONE || item >= IT_COUNT) return 0;
    return lane_insert(env, ref, lane_length_of(env, ref), item, new_item_id(env), 1);
}

int32_t fsim_belt_remove(fsim_env *env, int32_t index, int32_t lane, int32_t at) {
    int32_t ref = script_lane(env, index, lane);
    if (ref < 0) return 0;
    fsim_lane *l = lane_of(env, ref);
    if (at < 0 || at >= l->count) return 0;
    lane_take(l, at);
    seg_check(env, ref);
    return 1;
}

int32_t fsim_entity_remove(fsim_env *env, int32_t index, int32_t item, int32_t count) {
    if (index < 0 || index >= env->entity_count || item <= IT_NONE || item >= IT_COUNT) return 0;
    fsim_entity *e = &env->entities[index];
    if (!e->alive || e->kind != K_CHEST || count <= 0) return 0;
    int32_t done = chest_remove(e, item, count);
    if (done > 0) fsim_script_touched(env, index);
    return done;
}

void fsim_inserter_set(fsim_env *env, int32_t index, int32_t phase, int32_t step) {
    if (index < 0 || index >= env->entity_count) return;
    fsim_entity *s = &env->entities[index];
    if (!s->alive || s->kind != K_INSERTER) return;
    arm_target from, to;
    int32_t lift = 1;
    switch (phase) {
    case INS_APPROACH:
        s->arm_w = (float)s->direction / 16.0f;
        s->arm_len = ARM_START;
        s->arm_at = 0;
        to = arm_pickup_point(s);
        lift = 0;
        break;
    case INS_TO_DROP: from = arm_pickup_point(s); to = arm_drop_point(s); arm_place(s, from); break;
    case INS_TO_PICKUP: from = arm_drop_point(s); to = arm_pickup_point(s); arm_place(s, from); break;
    case INS_TO_SELF:
        from = arm_pickup_point(s); to = arm_self_point(s); arm_place(s, from); lift = 2; break;
    case INS_SELF_BACK:
        from = arm_self_point(s); to = arm_pickup_point(s); arm_place(s, from); lift = 2; break;
    case INS_WAIT_DROP:
        arm_place(s, arm_drop_point(s));
        s->phase = phase;
        arm_begin(s, 0);
        arm_draw(s);
        return;
    default:
        arm_place(s, arm_pickup_point(s));
        s->phase = phase;
        arm_begin(s, 0);
        arm_draw(s);
        return;
    }
    s->phase = phase;
    arm_begin(s, lift);
    for (int32_t k = 0; k < step; k++) {
        double spent[2];
        int short_tick;
        int arrived = arm_step(s, to, 1e18, spent, &short_tick);
        s->arm_at = arrived;
        if (arrived) {
            s->arm_vx = to.vx;
            s->arm_vy = to.vy;
        }
        s->lift_step++;
    }
    s->swing = step;
    arm_draw(s);
}

void fsim_inserter_hold(fsim_env *env, int32_t index, int32_t item) {
    if (index < 0 || index >= env->entity_count || item <= IT_NONE || item >= IT_COUNT) return;
    fsim_entity *s = &env->entities[index];
    if (!s->alive || s->kind != K_INSERTER) return;
    if (s->belt_asleep) {
        s->belt_asleep = 0;
        env->belt_sleepers--;
    }
    if (s->sleep_seq) {
        s->sleep_seq = 0;
        env->sleeper_count--;
        env->inserters[env->inserter_count++] = index;
    }
    s->held = item;
    s->chase_id = 0;
    s->phase = is_fuel(item) && s->fuel.count == 0 ? INS_TO_SELF : INS_TO_DROP;
    inserter_status(env, s);
    arm_begin(s, s->arm_at && s->arm_vx == arm_pickup_point(s).vx &&
                     s->arm_vy == arm_pickup_point(s).vy ? (s->phase == INS_TO_SELF ? 2 : 1) : -1);
}

void fsim_script_touched(fsim_env *env, int32_t index) {
    if (index < 0 || index >= env->entity_count || !env->entities[index].alive) return;
    wake(env, index);
    unblock_drills(env, index, env->entities[index].pos);
}

void fsim_remove_pile(fsim_env *env, int32_t index) {
    if (index < 0 || index >= env->entity_count) return;
    fsim_entity *e = &env->entities[index];
    if (!e->alive || e->kind != K_PILE) return;
    destroy_entity(env, index);
    unblock_drills(env, -1, e->pos);
}

void fsim_refresh(fsim_env *env) {
    if (env->logistics_version != env->entities_version) rebuild_logistics(env);
}

void fsim_advance(fsim_env *env, int32_t ticks) {
    for (int32_t i = 0; i < ticks; i++) {
        env->tick += 1;
        update_world(env);
    }
}

/* Many decisions in one call, for throughput: no Python between them. */
int64_t fsim_run(fsim_env *env, const fsim_action *actions, int32_t count, int32_t ticks) {
    for (int32_t i = 0; i < count; i++) fsim_step(env, &actions[i], ticks);
    return env->tick;
}

#include "fsim_rl.c"
