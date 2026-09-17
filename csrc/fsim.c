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
#define DRILL_USAGE 2500.0
#define FURNACE_USAGE 1500.0
#define DRILL_SPEED 0.25
#define CHAR_MINING_SPEED 0.5
#define SMELT_SECONDS 3.2
#define STEAM_POWER_PLATES 50
#define STEAM_POWER_LAG 24        /* ticks from the 50th plate to the research */

static const int32_t STACK_SIZE[IT_COUNT] = {
    0, 50, 50, 50, 50, 100, 100, 50, 50, 100, 100, 100, 100, 50, 50, 50,
};

static double mining_time_of_item(int32_t item) {
    switch (item) {
    case IT_IRON_ORE: case IT_COPPER_ORE: case IT_COAL: case IT_STONE: return 1.0;
    default: return 1.0;
    }
}

double fsim_capacity(int32_t kind) {
    /* A burner's buffer holds 16/15 of its per-tick draw. */
    return kind == K_DRILL ? DRILL_USAGE * 16.0 / 15.0 : FURNACE_USAGE * 16.0 / 15.0;
}

static double entity_mining_time(int32_t kind) {
    switch (kind) {
    case K_DRILL: return 0.3;
    case K_FURNACE: return 0.2;
    case K_WALL: return 0.2;
    case K_PILE: return 0.025;
    default: return 1.0;
    }
}

static int32_t entity_item(int32_t kind) {
    switch (kind) {
    case K_DRILL: return IT_BURNER_DRILL;
    case K_FURNACE: return IT_STONE_FURNACE;
    case K_WALL: return IT_STONE_WALL;
    default: return IT_NONE;
    }
}

static int32_t box_of(int32_t kind) {
    switch (kind) {
    case K_DRILL: case K_FURNACE: return MACHINE_BOX;
    case K_WALL: return WALL_BOX;
    case K_PILE: return PILE_BOX;
    default: return 0;
    }
}

/* Sort rank of an entity's prototype name, by string order. */
static int32_t name_rank(int32_t kind) {
    switch (kind) {
    case K_DRILL: return 0;   /* burner-mining-drill */
    case K_PILE: return 1;    /* item-on-ground */
    case K_FURNACE: return 2; /* stone-furnace */
    case K_WALL: return 3;    /* stone-wall */
    default: return 9;
    }
}

/* ------------------------------------------------------------------ helpers */

static int64_t floordiv(int64_t a, int64_t b) {
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) q -= 1;
    return q;
}

static double tiles(int32_t v) { return (double)v / TILE; }

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

static int32_t mint_unit(fsim_env *env, int32_t entity_index) {
    const fsim_entity *e = &env->entities[entity_index];
    for (int32_t h = 1; h < env->next_handle; h++) {
        fsim_handle *rec = &env->handles[h];
        if (rec->used && rec->kind == H_UNIT && rec->unit == e->unit && rec->destroyed_tick < 0)
            return h;
    }
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

static int character_blocked(const fsim_env *env, fsim_pos p) {
    for (int32_t i = 0; i < env->entity_count; i++) {
        const fsim_entity *e = &env->entities[i];
        if (!e->alive || e->kind == K_PILE) continue;
        int32_t reach = box_of(e->kind) + CHAR_BOX;
        int32_t dx = abs(e->pos.x - p.x), dy = abs(e->pos.y - p.y);
        /* Boxes that only touch still collide. */
        if (dx <= reach && dy <= reach) return 1;
    }
    return box_hits_water(env, p, CHAR_BOX, 1);
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
    /* Blocked: the largest power-of-two step that still clears. */
    for (int32_t step = 32; step >= 1; step /= 2) {
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

static int32_t new_entity(fsim_env *env, int32_t kind, fsim_pos pos, int32_t direction,
                          int32_t neutral) {
    int32_t i = env->entity_count++;
    fsim_entity *e = &env->entities[i];
    memset(e, 0, sizeof(*e));
    e->alive = 1;
    e->kind = kind;
    e->neutral = neutral;
    e->unit = ++env->next_unit;
    e->pos = pos;
    e->direction = direction;
    e->status = (kind == K_DRILL || kind == K_FURNACE) ? ST_NO_FUEL
               : (kind == K_WALL ? ST_WORKING : ST_NONE);
    return i;
}

static void destroy_entity(fsim_env *env, int32_t index) {
    fsim_entity *e = &env->entities[index];
    e->alive = 0;
    if (e->kind != K_PILE) unit_destroyed(env, e->unit);
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
        if (!e->alive || i == except) continue;
        if (e->kind != K_FURNACE && e->kind != K_DRILL) continue;
        if (p.x >= e->pos.x - TILE && p.x < e->pos.x + TILE &&
            p.y >= e->pos.y - TILE && p.y < e->pos.y + TILE)
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

static int32_t machine_accepts(fsim_entity *m, int32_t item, int32_t count, int do_insert) {
    if (m->kind == K_FURNACE) {
        if (item == IT_COAL || item == IT_WOOD) {
            if (do_insert) return slot_insert(&m->fuel, item, count, STACK_SIZE[item]);
            return slot_room(&m->fuel, item, STACK_SIZE[item]) >= count ? count : 0;
        }
        int32_t cap = item == IT_IRON_ORE ? FURNACE_SOURCE_CAP : STACK_SIZE[item];
        if (do_insert) return slot_insert(&m->source, item, count, cap);
        return slot_room(&m->source, item, cap) >= count ? count : 0;
    }
    if (m->kind == K_DRILL && (item == IT_COAL || item == IT_WOOD)) {
        if (do_insert) return slot_insert(&m->fuel, item, count, STACK_SIZE[item]);
        return slot_room(&m->fuel, item, STACK_SIZE[item]) >= count ? count : 0;
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
                e->remaining = FUEL_VALUE_COAL;
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

static void update_drill(fsim_env *env, int32_t index) {
    fsim_entity *d = &env->entities[index];
    fsim_pos drop = drop_position(d);
    int32_t target = machine_at(env, drop, index);
    if (d->held != IT_NONE) {
        if (target >= 0) {
            if (machine_accepts(&env->entities[target], d->held, 1, 1) == 1) {
                d->held = IT_NONE;
                d->linked_unit = env->entities[target].unit;
            } else {
                d->status = ST_WAITING_FOR_SPACE;
                return;
            }
        } else {
            if (pile_blocks(env, drop)) {
                d->status = ST_WAITING_FOR_SPACE;
                return;
            }
            int32_t p = new_entity(env, K_PILE, drop, 0, 1);
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
    } else if (produced && target < 0 && pile_blocks(env, drop)) {
        d->status = ST_WAITING_FOR_SPACE;
    } else {
        d->status = ST_WORKING;
    }
}

static int furnace_can_start(const fsim_entity *f) {
    if (f->source.count < 1) return 0;
    int32_t product = f->source.item == IT_IRON_ORE ? IT_IRON_PLATE
                    : f->source.item == IT_COPPER_ORE ? IT_COPPER_PLATE : IT_NONE;
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
        slot_remove(&f->source, f->source.item, 1);
        f->crafting = 1;
    }
    double fraction = burner_work(f, FURNACE_USAGE);
    if (fraction > 0) {
        f->seconds += fraction / 60.0;
        f->progress = f->seconds / SMELT_SECONDS;
        if (f->progress >= 1.0) {
            slot_insert(&f->result, IT_IRON_PLATE, 1, STACK_SIZE[IT_IRON_PLATE]);
            env->produced[IT_IRON_PLATE] += 1;
            env->plates_crafted += 1;
            if (env->plates_crafted == STEAM_POWER_PLATES && !env->steam_power_at)
                env->steam_power_at = env->tick + STEAM_POWER_LAG;
            f->products_finished += 1;
            f->progress -= 1.0;
            f->seconds = f->progress * SMELT_SECONDS;
            if (furnace_can_start(f)) {
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
        int32_t source = e->source.count;
        int32_t source_item = e->source.count ? e->source.item : IT_IRON_ORE;
        if (e->crafting) source += 1;
        if (source > 0) insert_main(env, source_item, source);
        if (e->result.count > 0) insert_main(env, e->result.item, e->result.count);
    }
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

static void update_world(fsim_env *env) {
    if (env->steam_power_at && env->tick >= env->steam_power_at) env->steam_power = 1;
    update_character(env);
    /* Machines in reverse creation order: a furnace sees ore a drill
     * delivered this tick only on the next one. */
    for (int32_t i = env->entity_count - 1; i >= 0; i--) {
        fsim_entity *e = &env->entities[i];
        if (!e->alive) continue;
        if (e->kind == K_FURNACE) update_furnace(env, i);
        else if (e->kind == K_DRILL) update_drill(env, i);
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
        if (!e->alive || e->kind == K_PILE) continue;
        int32_t reach = box_of(e->kind) + r;
        if (abs(e->pos.x - centre.x) < reach && abs(e->pos.y - centre.y) < reach) return 0;
    }
    int32_t reach = CHAR_BOX + r;
    if (abs(env->char_pos.x - centre.x) < reach && abs(env->char_pos.y - centre.y) < reach)
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
    int32_t kind = a->item == IT_BURNER_DRILL ? K_DRILL : a->item == IT_STONE_FURNACE ? K_FURNACE
                 : a->item == IT_STONE_WALL ? K_WALL : K_NONE;
    if (kind == K_NONE) return reject(env, E_INVALID_TARGET);
    if (centre_distance(env->char_pos, a->position) > BUILD_DISTANCE)
        return reject(env, E_OUT_OF_REACH);
    fsim_pos centre;
    if (kind == K_WALL) {
        centre.x = (int32_t)floordiv(a->position.x, TILE) * TILE + TILE / 2;
        centre.y = (int32_t)floordiv(a->position.y, TILE) * TILE + TILE / 2;
    } else {
        centre.x = (int32_t)floordiv(a->position.x + TILE / 2, TILE) * TILE;
        centre.y = (int32_t)floordiv(a->position.y + TILE / 2, TILE) * TILE;
    }
    if (!placeable(env, kind, centre)) return reject(env, E_COLLISION);
    int32_t direction = kind == K_DRILL ? a->direction * 4 : 0;
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
    if (kind != 1 || env->entities[index].kind != K_DRILL) return reject(env, E_INVALID_TARGET);
    fsim_entity *d = &env->entities[index];
    d->direction = (d->direction + (a->reverse ? 12 : 4)) % 16;
    env->act.status = R_COMPLETED;
    return 0;
}

/* Which of a machine's slots a transfer reads from or writes to, in the
 * mod's order: fuel, then source, then result. */
static fsim_stack *machine_slot(fsim_entity *m, int32_t item, int removing, int32_t *cap) {
    fsim_stack *order[3];
    int32_t caps[3];
    int n = 0;
    order[n] = &m->fuel;
    caps[n++] = STACK_SIZE[item];
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
            accepts = (item == IT_COAL || item == IT_WOOD) && slot_room(s, item, caps[i]) > 0;
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
        int32_t kind = env->entities[indices[k]].kind;
        if (kind != K_DRILL && kind != K_FURNACE) return reject(env, E_PRECONDITION);
    }
    int32_t item = a->item;
    int32_t available;
    fsim_stack *from_slot = NULL;
    int32_t from_cap = 0;
    if (ends[0] == 0) {
        available = count_main(env, item);
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
    } else {
        to_slot = machine_slot(&env->entities[indices[1]], item, 0, &to_cap);
        int accepts = slot_room(to_slot, item, to_cap) > 0;
        if (to_slot == &env->entities[indices[1]].fuel && !(item == IT_COAL || item == IT_WOOD))
            accepts = 0;
        if (!accepts) return reject(env, E_NO_SPACE);
    }
    int32_t removed = ends[0] == 0 ? remove_main(env, item, wanted)
                                   : slot_remove(from_slot, item, wanted);
    if (removed <= 0) return reject(env, E_NO_ITEMS);
    int32_t inserted = ends[1] == 0 ? insert_main(env, item, removed)
                                    : slot_insert(to_slot, item, removed, to_cap);
    if (inserted < removed) {
        int32_t back = removed - inserted;
        if (ends[0] == 0) insert_main(env, item, back);
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

static void remember(fsim_env *env, int32_t handle, const fsim_entity *e) {
    int32_t free_slot = -1;
    for (int32_t i = 0; i < FSIM_MAX_MEMORY; i++) {
        if (env->memory[i].used && env->memory[i].handle == handle) {
            free_slot = i;
            break;
        }
        if (!env->memory[i].used && free_slot < 0) free_slot = i;
    }
    if (free_slot < 0) return;
    fsim_memory *m = &env->memory[free_slot];
    m->used = 1;
    m->handle = handle;
    m->kind = e->kind;
    m->pos = e->pos;
    m->has_dir = e->kind == K_DRILL;
    m->direction = e->direction;
    if (e->kind == K_FURNACE) m->contents = e->source;
    else if (e->kind == K_PILE) m->contents = e->pile;
    else m->contents.item = m->contents.count = 0;
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
    for (int32_t i = 0; i < FSIM_MAX_MEMORY; i++) {
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

void fsim_after_load(fsim_env *env) {
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
    env->char_pos = scene->character;
    for (int32_t i = 0; i < scene->inventory_count; i++)
        insert_main(env, scene->inventory_item[i], scene->inventory_amount[i]);
    fsim_observe(env);
}

/* Many decisions in one call, for throughput: no Python between them. */
int64_t fsim_run(fsim_env *env, const fsim_action *actions, int32_t count, int32_t ticks) {
    for (int32_t i = 0; i < count; i++) fsim_step(env, &actions[i], ticks);
    return env->tick;
}
