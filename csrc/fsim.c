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
#define INSERTER_APPROACH_TICKS 8 /* as built, the hand reaches the pickup at t=9 */
/* Pickup to drop and drop to pickup take 38 ticks: a swing is half a turn at
 * 0.013 turn per tick, and the tick that picks up or drops is its first step,
 * so it ends once swing + 1 reaches this. A hand restarted after a tick short
 * of energy arrives when the fractions add up (FactorioRL's
 * logistics_inserter_fuel_exhaustion trace, the refuelled inserter's drop at
 * t=3025). */
#define INSERTER_SWING_STEPS (0.5 / 0.013)
#define INSERTER_FAST_TICKS 5     /* the first five ticks of a swing draw more */
#define INSERTER_SELF_TICKS 28    /* pickup to its own fuel slot, and back */
#define INSERTER_SOURCE_LIMIT 2   /* ore it keeps in a furnace's source */
#define INSERTER_FUEL_LIMIT 5     /* fuel it keeps in a fuel slot */
/* Burner draw per tick, from remaining_burning_fuel differences. Every swing
 * tick draws 900/2^26 J above the round figure, the approach one ulp. */
#define INSERTER_DRAW_APPROACH 0x1.b580000000001p+10   /* 1750.0000000000002 */
#define INSERTER_DRAW_FAST 0x1.2c00001c2p+11           /* 2400 + 900/2^26 */
#define INSERTER_DRAW_SLOW 0x1.450000708p+9            /* 650 + 900/2^26 */

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

/* Slide around the obstacle `next` runs into, if the engine would. */
static int try_slide(fsim_env *env, int32_t ux, int32_t uy, fsim_pos next) {
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
        fsim_pos ahead = {cleared.x + ux * STRIDE, cleared.y + uy * STRIDE};
        if (character_blocked(env, ahead)) continue;
        int32_t step = need < STRIDE ? need : STRIDE;
        fsim_pos moved = env->char_pos;
        if (horizontal) moved.y += side * step;
        else moved.x += side * step;
        if (character_blocked(env, moved)) continue;
        env->char_pos = moved;
        return 1;
    }
    return 0;
}

static void walk_one_tick(fsim_env *env, int32_t dir16) {
    int32_t ux = 0, uy = 0;
    if (dir16 == 0) uy = -1;
    else if (dir16 == 4) ux = 1;
    else if (dir16 == 8) uy = 1;
    else if (dir16 == 12) ux = -1;
    else return;
    fsim_pos next = {env->char_pos.x + ux * STRIDE, env->char_pos.y + uy * STRIDE};
    if (!character_blocked(env, next)) {
        env->char_pos = next;
        return;
    }
    if (try_slide(env, ux, uy, next)) return;
    /* Creep: half a stride, then 1/32 of a tile, then exactly to contact --
     * measured from every start phase against a furnace (a gap of 37 goes
     * 19, 8, 8, 2; a gap of 13 goes 8, 5; a gap of 7 goes 7). */
    static const int32_t CREEP[2] = {STRIDE / 2, 8};
    for (int32_t k = 0; k < 2; k++) {
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
        e->lane_length[lane] = TILE;
        e->lane_next[lane] = e->lane_side[lane] = -1;
    }
    if (kind == K_INSERTER) {
        /* As built: empty fuel slot, but already burning a quarter of a wood,
         * the hand retracted on the pickup side. */
        e->burning = IT_WOOD;
        e->remaining = INSERTER_BUILT_FUEL;
        e->phase = INS_APPROACH;
    }
    return i;
}

static void destroy_entity(fsim_env *env, int32_t index) {
    fsim_entity *e = &env->entities[index];
    e->alive = 0;
    env->entities_version++;
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
static void burner_refill(fsim_entity *e) {
    double capacity = fsim_capacity(e->kind);
    for (;;) {
        double needed = capacity - e->energy;
        if (needed <= 0) break;
        if (e->remaining <= 0) {
            if (e->fuel.count > 0) {
                e->burning = e->fuel.item;
                slot_remove(&e->fuel, e->fuel.item, 1);
                e->remaining = fuel_value(e->burning);
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
 * lanes are found once per change of the entities (rebuild_logistics) and run
 * front first, a chain before the chains that sideload into it, so a
 * sideloaded item joins a line that has already moved this tick.
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

/* Put `item` on lane `ref` aimed at `target`. Returns 1 when it went on.
 * `exact`: only at `target` itself (insert_at_back, whose target is the lane's
 * upstream edge). The placement is measured for drops at 128 (the drill on a
 * free, a stopped and a just-extended line) and for sideloads; `exact` is the
 * script call's behaviour and fits the probe's feed timing. */
static int lane_insert(fsim_env *env, int32_t ref, int32_t target, int32_t item, int32_t id,
                       int exact) {
    fsim_lane *lane = lane_of(env, ref);
    int32_t length = lane_length_of(env, ref);
    int32_t next = env->entities[ref >> 1].lane_next[ref & 1];
    /* The nearest item ahead, in this lane's coordinates: the back of the
     * lane it runs into, then this lane's own items up to the insertion. */
    int32_t q = target, ahead = INT32_MIN;
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
    if (q - target >= BELT_GAP || (exact && q != target)) return 0;
    if (lane->count >= FSIM_LANE_ITEMS) return 0;
    /* The move it makes at once. With nothing ahead on a line that ends here
     * it stops at 0 -- also on a lane that sideloads onward, where an item
     * put within 8 of the edge would otherwise transfer mid-insert; no drop or
     * sideload aims that close. */
    int32_t moved_to = q - BELT_SPEED;
    if (ahead != INT32_MIN) {
        if (ahead + BELT_GAP > moved_to) moved_to = ahead + BELT_GAP;
    } else if (next < 0 && moved_to < 0) {
        moved_to = 0;
    }
    if (moved_to > q) moved_to = q;
    if (moved_to >= length) return 0;
    if (moved_to < 0) {
        fsim_lane *down = lane_of(env, next);
        if (down->count >= FSIM_LANE_ITEMS) return 0;
        lane_put(down, down->count, moved_to + lane_length_of(env, next), item, id, 1);
        return 1;
    }
    lane_put(lane, at, moved_to, item, id, moved_to != q);
    return 1;
}

static int32_t new_item_id(fsim_env *env) { return ++env->next_item_id; }

/* Belt `index` under point `p`: which lane's half holds it, and the target
 * position, the point's distance from the belt's downstream edge. On the
 * centre line it is lane 2 (measured once: a belt running south, away from
 * the inserter). Returns 0 on a turn, where no drop was measured. */
static int belt_drop_target(const fsim_env *env, int32_t index, fsim_pos p, int32_t *ref,
                            int32_t *target) {
    const fsim_entity *b = &env->entities[index];
    if (b->shape != BELT_STRAIGHT) return 0;
    int32_t ux, uy;
    dir_vec(b->direction, &ux, &uy);
    int32_t dx = p.x - b->pos.x, dy = p.y - b->pos.y;
    int32_t along = dx * ux + dy * uy;
    int32_t lateral = -dx * uy + dy * ux;   /* towards the right of travel */
    *ref = index * 2 + (lateral < 0 ? 0 : 1);
    *target = TILE / 2 - along;
    return 1;
}

static int belt_drop(fsim_env *env, int32_t index, fsim_pos p, int32_t item) {
    int32_t ref, target;
    if (!belt_drop_target(env, index, p, &ref, &target)) return 0;
    return lane_insert(env, ref, target, item, new_item_id(env), 0);
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

/* One tick of a chain of lanes, front lane first. */
static void update_chain(fsim_env *env, const int32_t *refs, int32_t size) {
    int32_t total = 0;
    for (int32_t k = 0; k < size; k++) total += lane_of(env, refs[k])->count;
    if (total == 0) return;
    const fsim_entity *front = &env->entities[refs[0] >> 1];
    int32_t side = front->lane_side[refs[0] & 1];
    int32_t entry = front->lane_entry[refs[0] & 1];
    int has_ahead = 0;
    int32_t ahead = 0;          /* the item ahead after its move, in chain coordinates */
    int32_t offset = 0;         /* chain coordinate of this lane's downstream edge */
    for (int32_t k = 0; k < size; k++) {
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
                if (side >= 0 && lane_insert(env, side, entry + old, it->item, it->id, 0)) {
                    lane_take(lane, i);
                    continue;
                }
                want = 0;
            }
            if (want > old) want = old;
            has_ahead = 1;
            ahead = want;
            if (want < offset) {
                /* Onto the lane ahead, behind everything already there. */
                fsim_lane *down = lane_of(env, refs[k - 1]);
                int32_t down_length = lane_length_of(env, refs[k - 1]);
                if (down->count < FSIM_LANE_ITEMS) {
                    fsim_belt_item moving = *it;
                    lane_take(lane, i);
                    lane_put(down, down->count, want - (offset - down_length), moving.item,
                             moving.id, 1);
                    continue;
                }
                want = offset;      /* no room: cannot happen at 64 apart */
                ahead = want;
            }
            it->moved = (uint8_t)(want != old);
            it->pos = (int16_t)(want - offset);
            i++;
        }
        offset += lane_length_of(env, refs[k]);
    }
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

/* Chains run in the order rebuild_logistics found. Not settled by the
 * evidence: when both lanes of a feed reach a target on the same tick for the
 * first time, the engine moves one of the two items 8 further than this does
 * (feed lane 1 in the probe's sideload rig, t=128; feed lane 2 in FactorioRL's
 * logistics_sideload_merge trace, t=127) -- apparently whichever lane the
 * engine updates first, which looks like the order the lanes first got
 * items. Every later transfer in both recordings matches. */
static void update_belts(fsim_env *env) {
    for (int32_t c = 0; c < env->chain_count; c++) {
        int32_t size = env->chain_size[c];
        const int32_t *refs = &env->chain_lanes[env->chain_first[c]];
        if (size < 0) update_ring(env, refs, -size);
        else update_chain(env, refs, size);
    }
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

/* Belt shapes, lane links, the chains and the order they run in, and every
 * inserter's pickup and drop target. */
static void rebuild_logistics(fsim_env *env) {
    tile_entry belts[FSIM_MAX_ENTITIES];
    int32_t n = 0;
    env->inserter_count = 0;
    for (int32_t i = 0; i < env->entity_count; i++) {
        const fsim_entity *e = &env->entities[i];
        if (!e->alive) continue;
        if (e->kind == K_BELT) {
            belts[n].key = tile_key(floordiv(e->pos.x, TILE), floordiv(e->pos.y, TILE));
            belts[n].index = i;
            n++;
        } else if (e->kind == K_INSERTER) {
            env->inserters[env->inserter_count++] = i;
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
        b->shape = BELT_STRAIGHT;
        if (!behind && from_right != from_left) b->shape = from_right ? BELT_RIGHT : BELT_LEFT;
        b->lane_length[0] = b->shape == BELT_RIGHT ? BELT_CURVE_OUTER
                          : b->shape == BELT_LEFT ? BELT_CURVE_INNER : TILE;
        b->lane_length[1] = b->shape == BELT_RIGHT ? BELT_CURVE_INNER
                          : b->shape == BELT_LEFT ? BELT_CURVE_OUTER : TILE;
        /* A belt that became or stopped being a turn keeps its items, clipped
         * to the new lane: how the engine re-places them is not measured. */
        for (int lane = 0; lane < 2; lane++) {
            fsim_lane *l = &b->lanes[lane];
            int32_t limit = b->lane_length[lane] - 1;
            for (int32_t j = l->count - 1; j >= 0; j--) {
                if (l->items[j].pos > limit) l->items[j].pos = (int16_t)limit;
                if (j + 1 < l->count && l->items[j].pos > l->items[j + 1].pos)
                    l->items[j].pos = l->items[j + 1].pos;
                limit = l->items[j].pos;
            }
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
    /* By entity, lane 2 first: of the two lanes of one feed, lane 2 joins the
     * target first (a stuck target in FactorioRL's logistics_sideload_merge
     * trace, t=583). Measured once; see the note on update_belts. */
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
    for (int32_t k = 0; k < done; k++) {
        int32_t c = order[k];
        env->chain_first[k] = out;
        env->chain_size[k] = size[c];
        int32_t count = size[c] < 0 ? -size[c] : size[c];
        for (int32_t j = 0; j < count; j++) env->chain_lanes[out++] = lanes[first[c] + j];
    }
    env->chain_count = done;

    for (int32_t k = 0; k < env->inserter_count; k++) {
        int32_t index = env->inserters[k];
        fsim_entity *s = &env->entities[index];
        s->pickup_target = point_target(env, inserter_point(s, INSERTER_PICKUP), index);
        s->drop_target = point_target(env, inserter_point(s, -INSERTER_DROP), index);
    }
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
 * - It keeps a furnace at 2 ore and 5 fuel, checked before it picks up: a hand
 *   that would overfill waits empty at the pickup.
 * - Holding fuel with its own fuel slot empty, it swings to itself instead
 *   (28 ticks, 2,400 J each), fills the slot and swings back (28 more).
 *
 * Not modelled (see docs/sim-logistics.md, open questions): taking an item
 * that is still moving on a belt (inserter_belt_pickup), the redirect to its
 * own fuel slot when that empties mid-swing, and the draw of a hand that got
 * less than a full tick of energy, which is known only to stop it.
 */

static int32_t smelt_product(int32_t item);

/* Whether `s` would take `item` now: for its own empty fuel slot, or for a
 * drop target that would take it without passing the fill limits. A chest or
 * a belt is checked at the drop instead, where the hand waits (not measured
 * for a full chest). */
static int inserter_wants(const fsim_env *env, const fsim_entity *s, int32_t item) {
    if (is_fuel(item) && s->fuel.count == 0) return 1;
    if (s->drop_target < 0) return 0;
    const fsim_entity *d = &env->entities[s->drop_target];
    switch (d->kind) {
    case K_CHEST:
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
        /* The furnace's fuel limit; not measured for these. */
        return is_fuel(item) && d->fuel.count < INSERTER_FUEL_LIMIT &&
               slot_room(&d->fuel, item, STACK_SIZE[item]) > 0;
    default:
        return 0;
    }
}

/* Taking from a belt. The engine chases moving items (`chases_belt_items`):
 * the hand follows an item along the belt and takes it where they meet, which
 * the probe recorded tick by tick (rigs `same`, `tick_*`, `flow*`, `bend`) but
 * did not reduce to a rule. That model goes here.
 *
 * In: the inserter `s` (its hand is at the pickup this tick: arriving there,
 * or waiting) and the belt `belt` under its pickup point, after this tick's
 * belt update. Out: 1 with the item removed from its lane and returned in
 * `*item`, or 0 to wait.
 *
 * TODO(chase): implemented is only what the probe pinned down -- items that
 * did not move in this tick's belt update are taken like a chest's
 * (`bend`: the pickup 38 ticks after the drop), the one nearest the tile
 * centre, lane 2 first on a tie (measured twice, with the inserter on lane
 * 2's side, so "near lane" is as likely), then the downstream one. A moving
 * item is not taken: the hand waits until it stops. */
static int inserter_belt_pickup(fsim_env *env, const fsim_entity *s, int32_t belt,
                                int32_t *item) {
    fsim_entity *b = &env->entities[belt];
    int32_t best_lane = -1, best_at = -1, best_distance = 0;
    for (int32_t lane = 1; lane >= 0; lane--) {
        const fsim_lane *l = &b->lanes[lane];
        for (int32_t at = 0; at < l->count; at++) {
            const fsim_belt_item *it = &l->items[at];
            if (it->moved || it->pos >= TILE || !inserter_wants(env, s, it->item)) continue;
            int32_t distance = abs(it->pos - TILE / 2);
            if (best_lane < 0 || distance < best_distance) {
                best_lane = lane;
                best_at = at;
                best_distance = distance;
            }
        }
    }
    if (best_lane < 0) return 0;
    *item = b->lanes[best_lane].items[best_at].item;
    lane_take(&b->lanes[best_lane], best_at);
    return 1;
}

/* The hand at the pickup: take one item if there is one it wants. */
static int inserter_pickup(fsim_env *env, int32_t index) {
    fsim_entity *s = &env->entities[index];
    if (s->pickup_target < 0) return 0;
    fsim_entity *src = &env->entities[s->pickup_target];
    int32_t item = IT_NONE;
    if (src->kind == K_CHEST) {
        for (int i = 0; i < FSIM_CHEST_SLOTS && item == IT_NONE; i++)
            if (src->chest[i].count > 0 && inserter_wants(env, s, src->chest[i].item)) {
                item = src->chest[i].item;
                slot_remove(&src->chest[i], item, 1);
            }
    } else if (src->kind == K_FURNACE) {
        if (src->result.count > 0 && inserter_wants(env, s, src->result.item)) {
            item = src->result.item;
            slot_remove(&src->result, item, 1);
        }
    } else if (src->kind == K_BELT) {
        if (!inserter_belt_pickup(env, s, s->pickup_target, &item)) item = IT_NONE;
    }
    if (item == IT_NONE) return 0;
    s->held = item;
    s->phase = is_fuel(item) && s->fuel.count == 0 ? INS_TO_SELF : INS_TO_DROP;
    s->swing = 0;
    return 1;
}

/* The hand at the drop: put the item in. With nothing there it waits (the
 * engine drops it on the ground: not measured). */
static int inserter_drop(fsim_env *env, int32_t index) {
    fsim_entity *s = &env->entities[index];
    if (s->drop_target < 0) return 0;
    fsim_entity *d = &env->entities[s->drop_target];
    int done;
    if (d->kind == K_BELT)
        done = belt_drop(env, s->drop_target, inserter_point(s, -INSERTER_DROP), s->held);
    else
        done = machine_accepts(d, s->held, 1, 1) == 1;
    if (!done) return 0;
    s->held = IT_NONE;
    s->phase = INS_TO_PICKUP;
    s->swing = 0;
    return 1;
}

/* Whether the move in progress has ended. The approach and the swing to its
 * own slot were only seen in whole ticks. */
static int inserter_arrived(const fsim_entity *s) {
    switch (s->phase) {
    case INS_APPROACH: return s->swing >= INSERTER_APPROACH_TICKS;
    case INS_TO_SELF: case INS_SELF_BACK: return s->swing >= INSERTER_SELF_TICKS;
    default: return s->swing + 1.0 >= INSERTER_SWING_STEPS;
    }
}

static double inserter_draw(const fsim_entity *s) {
    switch (s->phase) {
    case INS_APPROACH: return INSERTER_DRAW_APPROACH;
    case INS_TO_SELF: case INS_SELF_BACK: return INSERTER_DRAW_FAST;
    default: return s->swing < INSERTER_FAST_TICKS ? INSERTER_DRAW_FAST : INSERTER_DRAW_SLOW;
    }
}

/* The hand has reached the end of its move. Returns 0 when it now waits. */
static int inserter_arrive(fsim_env *env, int32_t index) {
    fsim_entity *s = &env->entities[index];
    switch (s->phase) {
    case INS_TO_DROP:
        if (inserter_drop(env, index)) return 1;
        s->phase = INS_WAIT_DROP;
        s->status = ST_WAITING_FOR_SPACE;
        return 0;
    case INS_TO_SELF:
        s->swing = 0;
        if (slot_insert(&s->fuel, s->held, 1, STACK_SIZE[s->held]) == 1) {
            s->held = IT_NONE;
            s->phase = INS_SELF_BACK;
        } else {
            s->phase = INS_TO_DROP;   /* its slot filled meanwhile: not measured */
        }
        return 1;
    default:
        if (inserter_pickup(env, index)) return 1;
        s->phase = INS_WAIT_PICKUP;
        s->status = ST_WAITING_FOR_SOURCE;
        return 0;
    }
}

static void update_inserter(fsim_env *env, int32_t index) {
    fsim_entity *s = &env->entities[index];
    if (s->phase == INS_WAIT_PICKUP || s->phase == INS_WAIT_DROP) {
        int picking = s->phase == INS_WAIT_PICKUP;
        int acted = picking ? inserter_pickup(env, index) : inserter_drop(env, index);
        if (!acted) {
            s->status = picking ? ST_WAITING_FOR_SOURCE : ST_WAITING_FOR_SPACE;
            return;
        }
    } else {
        double fraction = burner_work(s, inserter_draw(s));
        s->swing += fraction;
        if (fraction > 0 && inserter_arrived(s) && !inserter_arrive(env, index))
            return;
    }
    burner_refill(s);
    s->status = s->energy > 0 ? ST_WORKING : ST_NO_FUEL;
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
            } else {
                d->status = ST_WAITING_FOR_SPACE;
                return;
            }
        } else if (belt >= 0) {
            /* Onto the lane whose half holds the drop point: lane 1 at 128
             * for a drill facing south onto a belt running east. */
            if (!belt_drop(env, belt, drop, d->held)) {
                d->status = ST_WAITING_FOR_SPACE;
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
        d->status = ST_NO_MINABLE;
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
    burner_refill(d);
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
            burner_refill(f);
            f->status = f->energy > 0 ? ST_WORKING : ST_NO_FUEL;
            return;
        }
        f->ingredient = f->source.item;
        slot_remove(&f->source, f->source.item, 1);
        f->crafting = 1;
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
        }
    }
    if (f->crafting) {
        burner_refill(f);
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
    /* Not measured: the order a chest's, a belt's and an inserter's contents
     * come back in, which only decides the slots they land in. */
    if (e->kind == K_CHEST)
        for (int i = 0; i < FSIM_CHEST_SLOTS; i++)
            if (e->chest[i].count > 0) insert_main(env, e->chest[i].item, e->chest[i].count);
    if (e->kind == K_BELT)
        for (int lane = 0; lane < 2; lane++)
            for (int32_t k = 0; k < e->lanes[lane].count; k++)
                insert_main(env, e->lanes[lane].items[k].item, 1);
    if (e->kind == K_INSERTER && e->held != IT_NONE) insert_main(env, e->held, 1);
    if (e->kind == K_PILE) {
        insert_main(env, e->pile.item, e->pile.count);
    } else {
        insert_main(env, entity_item(e->kind), 1);
    }
    destroy_entity(env, index);
}

static void update_character(fsim_env *env) {
    env->walk_pub = env->walk_set;
    if (env->walk_set) env->walk_pub_dir = env->walk_set_dir;
    if (env->walk_pub) {
        env->char_dir8 = env->walk_pub_dir;
        walk_one_tick(env, env->walk_pub_dir);
    }
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
 * 3. inserters, in creation order (not measured between inserters): before
 *    drills and furnaces, since a drill's output into a chest, a furnace's
 *    product and room made in a furnace's source are all acted on a tick
 *    later, whichever was built first;
 * 4. drills and furnaces in reverse creation order: a furnace sees ore a
 *    drill delivered this tick only on the next one. */
static void update_world(fsim_env *env) {
    if (env->steam_power_at && env->tick >= env->steam_power_at) env->steam_power = 1;
    update_character(env);
    if (env->logistics_version != env->entities_version) rebuild_logistics(env);
    if (env->chain_count) update_belts(env);
    for (int32_t k = 0; k < env->inserter_count; k++) {
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
        /* The first item and how many of it; a chest of several items is
         * more than a remembered record holds. */
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
