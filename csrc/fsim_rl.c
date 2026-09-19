/* The RL contract FactorioRL's policies see, computed from simulator state.
 *
 * Mirrors, in order: `encoders.encode` (local-v1 tensor layout, used for
 * local-v2 tasks), `FactorioEnv.argument_domains` and `_placement_candidates`,
 * `FactorioEnv.action_masks`, `ParameterizedEnv.action_masks` and `decode`,
 * `FactorioEnv._goal_vector`, `RewardAccountant` and the episode logic of
 * `FactorioEnv.step_payload`, including the verification window. Checked
 * against the tensor hashes, masks, goals and rewards FactorioRL recorded.
 *
 * Included from fsim.c: it reads the simulator's internals directly.
 */

#define RL_RADIUS 32
#define RL_PLACEMENT_RADIUS 5
#define RL_TYPE_SLOTS 12            /* encoders.ENTITY_TYPES */
#define RL_STATUS_SLOTS 13          /* encoders.ENTITY_STATUS */
#define RL_DIRECTIONS 16.0

/* encoders.ITEMS, as simulator item ids (IT_NONE for items it lacks). */
static const int32_t RL_ITEM_IDS[RL_ITEMS] = {
    IT_IRON_ORE, IT_COPPER_ORE, IT_COAL, IT_STONE, IT_IRON_PLATE, IT_COPPER_PLATE,
    IT_STONE_FURNACE, IT_IRON_GEAR, IT_TRANSPORT_BELT, IT_WOOD, IT_SMALL_POLE,
    IT_WOODEN_CHEST, IT_BURNER_DRILL, IT_BURNER_INSERTER,
};

/* Catalog operation indices (parameterized-v1). */
#define OP_PLACE_AT 12
#define OP_MINE_AT 13
#define OP_ROTATE_AT 14
#define OP_ROTATE_REVERSE 15
#define OP_GIVE_TO 16
#define OP_TAKE_FROM 17
#define OP_SET_RECIPE 18
#define OP_CRAFT 19
#define OP_CANCEL 20
#define OP_WAIT 21

static const int32_t TRANSFER_AMOUNTS[3] = {1, 5, 20};

static double rl_clip(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static double rl_log_count(double count, double cap) {
    return rl_clip(log1p(count > 0.0 ? count : 0.0) / log1p(cap), 0.0, 1.0);
}

static int32_t rl_type_index(int32_t kind) {
    switch (kind) {
    case K_FURNACE: return 1;   /* furnace */
    case K_DRILL: return 3;     /* mining-drill */
    case K_WALL: return 10;     /* wall */
    default: return 11;         /* other: item-entity */
    }
}

/* ENTITY_STATUS index + 1, or 0 for a name not in the list. */
static int32_t rl_status_slot(int32_t status) {
    switch (status) {
    case ST_WORKING: return 1;
    case ST_NO_FUEL: return 5;
    case ST_NO_MINABLE: return 6;
    case ST_WAITING_FOR_SPACE: return 8;
    default: return 0;           /* no_ingredients is not in the vocabulary */
    }
}

/* ------------------------------------------------------------------ rows */

typedef struct {
    double distance;
    int32_t remembered;
    int32_t index;       /* seen index, or memory index */
    int32_t order;
} rl_row;

/* The entity table's rows: visible entities then remembered ones, stably by
 * distance from the character, at most RL_MAX_ENTITIES. */
static int32_t rl_rows(const fsim_env *env, rl_row *rows) {
    double ox = tiles(env->char_pos.x), oy = tiles(env->char_pos.y);
    int32_t n = 0;
    for (int32_t k = 0; k < env->seen_count; k++) {
        const fsim_entity *e = &env->entities[env->seen[k].entity];
        rows[n].distance = hypot(tiles(e->pos.x) - ox, tiles(e->pos.y) - oy);
        rows[n].remembered = 0;
        rows[n].index = k;
        rows[n].order = n;
        n++;
    }
    for (int32_t k = 0; k < env->remembered_count; k++) {
        const fsim_memory *m = &env->memory[env->remembered[k]];
        rows[n].distance = hypot(tiles(m->pos.x) - ox, tiles(m->pos.y) - oy);
        rows[n].remembered = 1;
        rows[n].index = env->remembered[k];
        rows[n].order = n;
        n++;
    }
    for (int32_t i = 1; i < n; i++) {
        rl_row v = rows[i];
        int32_t j = i - 1;
        while (j >= 0 && rows[j].distance > v.distance) {
            rows[j + 1] = rows[j];
            j--;
        }
        rows[j + 1] = v;
    }
    return n > RL_MAX_ENTITIES ? RL_MAX_ENTITIES : n;
}

/* ------------------------------------------------------------------ domains */

typedef struct {
    int32_t targets[RL_MAX_ENTITIES + FSIM_MAX_TILES + 64];
    int32_t target_count;
    int32_t placements[2 * RL_PLACEMENTS];   /* tile x, y */
    int32_t placement_count;
    int32_t placement_legal[RL_PLACEMENTS];  /* v2: slot k is a legal tile */
    int32_t held[IT_COUNT];                  /* item held with a count */
    int32_t source[IT_COUNT];                /* item some visible entity holds */
} rl_domains;

static void rl_domains_build(const fsim_rl *rl, rl_domains *d) {
    const fsim_env *env = rl->env;
    int v2 = rl->task.action_space == ACTION_SPACE_V2;
    memset(d, 0, sizeof(*d));
    if (v2) {
        rl_row rows[FSIM_MAX_SWEEP + FSIM_MAX_MEMORY];
        int32_t n = rl_rows(env, rows);
        for (int32_t k = 0; k < n; k++) {
            d->targets[d->target_count++] = rows[k].remembered
                ? env->memory[rows[k].index].handle
                : env->seen[rows[k].index].handle;
        }
    } else {
        for (int32_t k = 0; k < env->seen_count; k++)
            d->targets[d->target_count++] = env->seen[k].handle;
        for (int32_t k = 0; k < env->tile_count; k++)
            d->targets[d->target_count++] = env->tiles[k].handle;
    }

    int32_t here_x = (int32_t)floordiv(env->char_pos.x, TILE);
    int32_t here_y = (int32_t)floordiv(env->char_pos.y, TILE);
    /* Occupancy of the (2R+1)^2 window, marked in one pass over the sweep
     * and the blocked tiles instead of once per candidate. */
    enum { SIDE = 2 * RL_PLACEMENT_RADIUS + 1 };
    uint8_t occupied_at[SIDE * SIDE];
    memset(occupied_at, 0, sizeof(occupied_at));
    occupied_at[RL_PLACEMENT_RADIUS * SIDE + RL_PLACEMENT_RADIUS] = 1; /* own tile */
    for (int32_t k = 0; k < env->seen_count; k++) {
        const fsim_entity *e = &env->entities[env->seen[k].entity];
        if (e->kind == K_PILE) continue;
        int64_t dx = floordiv(e->pos.x, TILE) - here_x + RL_PLACEMENT_RADIUS;
        int64_t dy = floordiv(e->pos.y, TILE) - here_y + RL_PLACEMENT_RADIUS;
        if (dx >= 0 && dx < SIDE && dy >= 0 && dy < SIDE) occupied_at[dx * SIDE + dy] = 1;
    }
    for (int32_t k = 0; k < env->blocked_count; k++) {
        int64_t dx = (int64_t)env->blocked[2 * k] - here_x + RL_PLACEMENT_RADIUS;
        int64_t dy = (int64_t)env->blocked[2 * k + 1] - here_y + RL_PLACEMENT_RADIUS;
        if (dx >= 0 && dx < SIDE && dy >= 0 && dy < SIDE) occupied_at[dx * SIDE + dy] = 1;
    }
    for (int32_t dx = -RL_PLACEMENT_RADIUS; dx <= RL_PLACEMENT_RADIUS; dx++) {
        for (int32_t dy = -RL_PLACEMENT_RADIUS; dy <= RL_PLACEMENT_RADIUS; dy++) {
            int32_t tx = here_x + dx, ty = here_y + dy;
            int32_t slot = (dx + RL_PLACEMENT_RADIUS) * SIDE + dy + RL_PLACEMENT_RADIUS;
            if (v2) {
                d->placements[2 * slot] = tx;
                d->placements[2 * slot + 1] = ty;
                d->placement_legal[slot] = !occupied_at[slot];
                d->placement_count += !occupied_at[slot];
                continue;
            }
            if (occupied_at[slot]) continue;
            d->placements[2 * d->placement_count] = tx;
            d->placements[2 * d->placement_count + 1] = ty;
            d->placement_count++;
        }
    }
    for (int i = 0; i < FSIM_MAIN_SLOTS; i++)
        if (env->main[i].count > 0) d->held[env->main[i].item] = 1;
    for (int32_t k = 0; k < env->seen_count; k++) {
        const fsim_entity *e = &env->entities[env->seen[k].entity];
        if (e->kind == K_FURNACE) {
            if (e->source.count > 0) d->source[e->source.item] = 1;
            if (e->result.count > 0) d->source[e->result.item] = 1;
        } else if (e->kind == K_PILE && e->pile.count > 0) {
            d->source[e->pile.item] = 1;
        }
    }
}

static int rl_any(const int32_t *flags) {
    for (int i = 0; i < IT_COUNT; i++)
        if (flags[i]) return 1;
    return 0;
}

void fsim_rl_mask(fsim_rl *rl, uint8_t *mask) {
    rl_domains d;
    rl_domains_build(rl, &d);
    int v2 = rl->task.action_space == ACTION_SPACE_V2;
    memset(mask, 0, RL_MASK_SIZE);
    int has_targets = d.target_count > 0;
    int has_items = rl_any(d.held);
    int has_sources = rl_any(d.source);
    for (int op = 0; op < RL_OPERATIONS; op++) {
        int legal;
        switch (op) {
        case OP_PLACE_AT: legal = d.placement_count > 0 && has_items; break;
        case OP_MINE_AT: case OP_ROTATE_AT: case OP_ROTATE_REVERSE: legal = has_targets; break;
        case OP_GIVE_TO: legal = has_targets && has_items; break;
        case OP_TAKE_FROM: legal = has_targets && has_sources; break;
        case OP_SET_RECIPE: case OP_CRAFT: case OP_CANCEL: legal = 0; break; /* no dimension */
        default: legal = 1; break;
        }
        mask[op] = (uint8_t)legal;
    }
    int32_t offset = RL_OPERATIONS;
    mask[offset] = 1;
    for (int32_t k = 0; k < d.target_count && k < RL_TARGETS; k++) mask[offset + 1 + k] = 1;
    offset += RL_TARGETS + 1;
    mask[offset] = 1;
    if (v2) {
        for (int32_t k = 0; k < RL_PLACEMENTS; k++) mask[offset + 1 + k] = (uint8_t)d.placement_legal[k];
    } else {
        for (int32_t k = 0; k < d.placement_count && k < RL_PLACEMENTS; k++) mask[offset + 1 + k] = 1;
    }
    offset += RL_PLACEMENTS + 1;
    for (int k = 0; k <= 4; k++) mask[offset + k] = 1;
    offset += 5;
    mask[offset] = 1;
    for (int k = 0; k < RL_ITEMS; k++) {
        int32_t item = RL_ITEM_IDS[k];
        mask[offset + 1 + k] = (uint8_t)(item != IT_NONE && (d.held[item] || d.source[item]));
    }
    offset += RL_ITEMS + 1;
    for (int k = 0; k <= 3; k++) mask[offset + k] = 1;
}

/* ------------------------------------------------------------------ decode */

int32_t fsim_rl_decode(fsim_rl *rl, const int32_t *v, fsim_action *out) {
    memset(out, 0, sizeof(*out));
    out->verb = V_WAIT;
    int32_t op = v[0];
    if (op < 0 || op >= RL_OPERATIONS) return 1;
    if (op < 12) {
        out->verb = V_MOVE;
        out->direction = op % 4;
        out->ticks = op < 4 ? 30 : (op < 8 ? 7 : 2);
        return 0;
    }
    if (op == OP_WAIT) return 0;
    if (op == OP_SET_RECIPE || op == OP_CRAFT || op == OP_CANCEL) return 1;

    rl_domains d;
    rl_domains_build(rl, &d);
    int v2 = rl->task.action_space == ACTION_SPACE_V2;
    int32_t targets = d.target_count < RL_TARGETS ? d.target_count : RL_TARGETS;
    int32_t placements = d.placement_count < RL_PLACEMENTS ? d.placement_count : RL_PLACEMENTS;
    int32_t target = v[1], placement = v[2], direction = v[3], item_index = v[4], amount = v[5];

    /* Arguments in payload order, as `decode` walks them: a missing or
     * out-of-range one is a failure, whichever it is. */
    switch (op) {
    case OP_PLACE_AT:
        if (item_index == 0 || item_index - 1 >= RL_ITEMS) return 1;
        if (v2) {
            if (placement < 1 || placement > RL_PLACEMENTS || !d.placement_legal[placement - 1])
                return 1;
        } else if (placement == 0 || placement - 1 >= placements) {
            return 1;
        }
        if (direction == 0 || direction - 1 >= 4) return 1;
        break;
    case OP_MINE_AT: case OP_ROTATE_AT: case OP_ROTATE_REVERSE:
        if (target == 0 || target - 1 >= targets) return 1;
        break;
    case OP_GIVE_TO: case OP_TAKE_FROM:
        if (target == 0 || target - 1 >= targets) return 1;
        if (item_index == 0 || item_index - 1 >= RL_ITEMS) return 1;
        if (amount == 0 || amount - 1 >= 3) return 1;
        break;
    default:
        return 1;
    }
    int32_t item = item_index > 0 ? RL_ITEM_IDS[item_index - 1] : IT_NONE;
    /* `step_arguments` re-checks items against the domain the verb reads. */
    if (op == OP_PLACE_AT || op == OP_GIVE_TO) {
        if (item == IT_NONE || !d.held[item]) return 1;
    } else if (op == OP_TAKE_FROM) {
        if (item == IT_NONE || !d.source[item]) return 1;
    }
    switch (op) {
    case OP_PLACE_AT:
        out->verb = V_PLACE;
        out->item = item;
        out->direction = direction - 1;
        out->position.x = d.placements[2 * (placement - 1)] * TILE + TILE / 2;
        out->position.y = d.placements[2 * (placement - 1) + 1] * TILE + TILE / 2;
        break;
    case OP_MINE_AT:
        out->verb = V_MINE;
        out->handle = d.targets[target - 1];
        out->count = 1;
        break;
    case OP_ROTATE_AT: case OP_ROTATE_REVERSE:
        out->verb = V_ROTATE;
        out->handle = d.targets[target - 1];
        out->reverse = op == OP_ROTATE_REVERSE;
        break;
    case OP_GIVE_TO:
        out->verb = V_TRANSFER;
        out->from_handle = 0;
        out->to_handle = d.targets[target - 1];
        out->item = item;
        out->count = TRANSFER_AMOUNTS[amount - 1];
        break;
    case OP_TAKE_FROM:
        out->verb = V_TRANSFER;
        out->from_handle = d.targets[target - 1];
        out->to_handle = 0;
        out->item = item;
        out->count = TRANSFER_AMOUNTS[amount - 1];
        break;
    }
    return 0;
}

/* ------------------------------------------------------------------ encode */

static void rl_goal(fsim_rl *rl, float *goal);
double fsim_rl_potential(const fsim_rl *rl);

/* The fields of one observation, wherever they live: `fsim_obs` has a float
 * grid, `fsim_obs8` a packed one (`grid` is NULL exactly when `flags` is set).
 * The caller has zeroed every field. */
typedef struct {
    float *grid;
    uint8_t *flags;       /* packed planes 0-3 and 5 */
    uint8_t *amount;      /* plane 4 as bytes */
    float *entities;
    int8_t *entity_mask;
    float *self_;
    float *inventory;
    float *goal;
} rl_fields;

/* Round half to even, as `rint` does in the default rounding mode, without
 * the library call: MSVC's `rint` reads the floating-point environment on
 * every call, and two per resource tile made it most of the encoder's time.
 * Exact for |x| < 2^53 (the integer part and the fraction are both exact). */
static double rl_rint(double x) {
    double t = (double)(int64_t)x;      /* toward zero */
    if (t > x) t -= 1.0;                /* now floor(x) */
    double frac = x - t;
    if (frac > 0.5) return t + 1.0;
    if (frac < 0.5) return t;
    return ((int64_t)t & 1) ? t + 1.0 : t;
}

/* round(255 * v) for v in [0, 1], half to even: what `fsim_obs8` stores, and
 * what `torch.round(grid * 255)` gives on the float grid. */
static uint8_t rl_byte(float v) {
    if (v <= 0.0f) return 0;
    if (v >= 1.0f) return 255;
    return (uint8_t)rl_rint((double)(v * 255.0f));
}

/* Sets flag plane `k`'s bit for `cell` in a packed grid. */
static void rl_set_flag(uint8_t *flags, int32_t k, int32_t cell) {
    int32_t bit = k * RL_GRID_SIZE * RL_GRID_SIZE + cell;
    flags[bit >> 3] |= (uint8_t)(1u << (bit & 7));
}

static void rl_encode_into(fsim_rl *rl, rl_fields *obs);

void fsim_rl_encode(fsim_rl *rl, fsim_obs *obs) {
    memset(obs, 0, sizeof(*obs));
    rl_fields fields = {obs->grid, NULL, NULL, obs->entities, obs->entity_mask, obs->self_,
                        obs->inventory, obs->goal};
    rl_encode_into(rl, &fields);
}

static void rl_encode_into(fsim_rl *rl, rl_fields *obs) {
    const fsim_env *env = rl->env;
    double ox = tiles(env->char_pos.x), oy = tiles(env->char_pos.y);
    const int32_t span = 2 * RL_RADIUS;
    const int32_t plane_size = RL_GRID_SIZE * RL_GRID_SIZE;

    for (int32_t k = 0; k < env->tile_count; k++) {
        const fsim_resource *r = &env->resources[env->tiles[k].resource];
        double px = r->tx + 0.5, py = r->ty + 0.5;
        int32_t col = (int32_t)rl_rint(px - ox) + RL_RADIUS;
        int32_t row = (int32_t)rl_rint(py - oy) + RL_RADIUS;
        if (row < 0 || row > span || col < 0 || col > span) continue;
        int32_t plane = r->item == IT_IRON_ORE ? 0 : r->item == IT_COPPER_ORE ? 1
                      : r->item == IT_COAL ? 2 : r->item == IT_STONE ? 3 : -1;
        int32_t cell = row * RL_GRID_SIZE + col;
        float amount = (float)rl_log_count((double)r->amount, 4000.0);
        if (obs->grid) {
            if (plane >= 0) obs->grid[plane * plane_size + cell] = 1.0f;
            float *slot = &obs->grid[4 * plane_size + cell];
            if (amount > *slot) *slot = amount;
        } else {
            if (plane >= 0) rl_set_flag(obs->flags, plane, cell);
            uint8_t byte = rl_byte(amount);
            if (byte > obs->amount[cell]) obs->amount[cell] = byte;
        }
    }
    for (int32_t k = 0; k < env->blocked_count; k++) {
        double px = env->blocked[2 * k], py = env->blocked[2 * k + 1];
        int32_t col = (int32_t)rl_rint(px - ox) + RL_RADIUS;
        int32_t row = (int32_t)rl_rint(py - oy) + RL_RADIUS;
        if (row < 0 || row > span || col < 0 || col > span) continue;
        int32_t cell = 5 * plane_size + row * RL_GRID_SIZE + col;
        if (obs->grid) obs->grid[cell] = 1.0f;
        else rl_set_flag(obs->flags, 4, row * RL_GRID_SIZE + col);
    }

    /* Rows: visible entities then remembered ones, stably by distance. */
    rl_row rows[FSIM_MAX_SWEEP + FSIM_MAX_MEMORY];
    int32_t n = rl_rows(env, rows);
    for (int32_t i = 0; i < n; i++) {
        float *f = &obs->entities[i * RL_ENTITY_FEATURES];
        int32_t kind, direction, status, working_known, working;
        int32_t contents = 0, fuel = 0, output = 0;
        double px, py, age = 0.0;
        if (!rows[i].remembered) {
            const fsim_entity *e = &env->entities[env->seen[rows[i].index].entity];
            kind = e->kind;
            px = tiles(e->pos.x);
            py = tiles(e->pos.y);
            direction = e->kind == K_DRILL ? e->direction : 0;
            status = e->kind == K_PILE ? ST_NONE : e->status;
            working_known = e->kind != K_PILE;
            working = e->status == ST_WORKING;
            if (e->kind == K_FURNACE) {
                contents = e->source.count;
                output = e->result.count;
            } else if (e->kind == K_PILE) {
                contents = e->pile.count;
            }
            if (e->kind == K_DRILL || e->kind == K_FURNACE) fuel = e->fuel.count;
        } else {
            const fsim_memory *m = &env->memory[rows[i].index];
            kind = m->kind;
            px = tiles(m->pos.x);
            py = tiles(m->pos.y);
            direction = m->has_dir ? m->direction : 0;
            status = ST_NONE;
            working_known = 0;
            working = 0;
            contents = m->contents.count;
            age = (double)(env->tick - m->last_seen);
        }
        f[0] = (float)rl_clip((px - ox) / RL_RADIUS, -1.0, 1.0);
        f[1] = (float)rl_clip((py - oy) / RL_RADIUS, -1.0, 1.0);
        f[2] = (float)rl_clip(rows[i].distance / RL_RADIUS, -1.0, 1.0);
        f[3] = (float)((double)rl_type_index(kind) / (double)(RL_TYPE_SLOTS - 1));
        f[4] = (float)((double)direction / RL_DIRECTIONS);
        f[5] = (float)rl_log_count((double)contents, 200.0);
        f[6] = 0.0f;
        f[7] = status != ST_NONE ? 1.0f : 0.0f;
        f[8] = 1.0f;
        f[9] = rows[i].remembered ? 1.0f : 0.0f;
        f[10] = rows[i].remembered ? (float)rl_log_count(age, 3600.0) : 0.0f;
        int32_t slot = rl_status_slot(status);
        f[11] = slot ? (float)((double)slot / (double)RL_STATUS_SLOTS) : 0.0f;
        f[12] = working_known && working ? 1.0f : 0.0f;
        f[13] = working_known ? 1.0f : 0.0f;
        f[14] = (float)rl_log_count((double)fuel, 200.0);
        f[15] = (float)rl_log_count((double)output, 200.0);
        obs->entity_mask[i] = 1;
    }

    obs->self_[0] = (float)rl_clip(ox / 128.0, -1.0, 1.0);
    obs->self_[1] = (float)rl_clip(oy / 128.0, -1.0, 1.0);
    obs->self_[2] = env->walk_pub ? 1.0f : 0.0f;
    obs->self_[3] = (float)((double)env->walk_pub_dir / RL_DIRECTIONS);
    obs->self_[4] = env->mining ? 1.0f : 0.0f;
    obs->self_[5] = env->mining ? (float)env->mining_progress : 0.0f;
    int running = 0;
    for (int32_t i = 0; i < FSIM_MAX_INFLIGHT; i++) {
        const fsim_inflight *f = &env->inflight[i];
        if (f->used && !f->terminal && f->verb >= 0) running = 1;
    }
    obs->self_[8] = running ? 1.0f : 0.0f;
    if (env->event_count > 0) {
        const fsim_event *last =
            &env->events[(env->event_head + env->event_count - 1) % FSIM_EVENT_LIMIT];
        int refused_last = last->status == R_FAILED || last->status == R_CANCELLED;
        obs->self_[9] = refused_last ? 1.0f : 0.0f;
        obs->self_[10] = last->status == R_COMPLETED ? 1.0f : 0.0f;
        int32_t refused = 0;
        for (int32_t k = 0; k < env->event_count; k++) {
            const fsim_event *e = &env->events[(env->event_head + k) % FSIM_EVENT_LIMIT];
            if (e->status == R_FAILED || e->status == R_CANCELLED) refused++;
        }
        obs->self_[11] = (float)((double)refused / (double)env->event_count);
    }
    for (int k = 0; k < RL_ITEMS; k++) {
        int32_t item = RL_ITEM_IDS[k];
        obs->inventory[k] = item == IT_NONE ? 0.0f
                          : (float)rl_log_count((double)count_main(env, item), 200.0);
    }
    rl_goal(rl, obs->goal);
}

/* ------------------------------------------------------------------ task */

static int32_t rl_machine_produced(const fsim_env *env, int32_t item) {
    int32_t v = env->produced[item] - env->mined_by_action[item];
    return v > 0 ? v : 0;
}

#define WINDOW_PLATES 10
#define WINDOW_TICKS 3600
#define SETTLE_TICKS 7200
#define SAMPLE_GAP 30
#define VERIFY_TARGET 10
/* plate_line's goal: thirty plates, about an eighth of its tick budget spent
 * producing, which is more than a single hand-fed smelt can reach. */
#define PLATE_TARGET 30
#define VERIFY_TICKS 3600
#define PROGRESS_WEIGHT 0.5
#define PROGRESS_CAP 0.45

static void rl_window_record(fsim_rl *rl) {
    int64_t tick = rl->env->tick;
    int32_t plates = rl->env->produced[IT_IRON_PLATE];
    if (rl->window_count > 0) {
        int32_t last = (rl->window_head + rl->window_count - 1) % RL_WINDOW_SAMPLES;
        if (rl->window_tick[last] == tick) {
            rl->window_plates[last] = plates;
            return;
        }
    }
    int32_t index;
    if (rl->window_count < RL_WINDOW_SAMPLES) {
        index = (rl->window_head + rl->window_count) % RL_WINDOW_SAMPLES;
        rl->window_count++;
    } else {
        index = rl->window_head;
        rl->window_head = (rl->window_head + 1) % RL_WINDOW_SAMPLES;
    }
    rl->window_tick[index] = tick;
    rl->window_plates[index] = plates;
}

static int rl_sustained(const fsim_rl *rl) {
    int32_t n = rl->window_count;
    if (n < 2) return 0;
    #define AT(i) ((rl->window_head + (i)) % RL_WINDOW_SAMPLES)
    int64_t first = rl->window_tick[AT(0)];
    int64_t latest = rl->window_tick[AT(n - 1)];
    if (latest - first < WINDOW_TICKS) return 0;
    int64_t cutoff = latest - WINDOW_TICKS;
    int32_t start = 0;
    for (int32_t i = 0; i < n; i++) {
        if (rl->window_tick[AT(i)] <= cutoff) start = i;
        else break;
    }
    if (n - start < 2) return 0;
    for (int32_t i = start + 1; i < n; i++)
        if (rl->window_tick[AT(i)] - rl->window_tick[AT(i - 1)] > SAMPLE_GAP) return 0;
    if (latest < SETTLE_TICKS) return 0;
    int32_t output = rl->window_plates[AT(n - 1)] - rl->window_plates[AT(start)];
    return output >= WINDOW_PLATES;
    #undef AT
}

static int rl_succeeded(const fsim_rl *rl) {
    if (rl->task.task == TASK_PLATE_LINE) {
        /* The line is already built, so nothing is verified and nothing is
         * counted as constructed: the only question is whether it ran. */
        return rl->env->produced[IT_IRON_PLATE] >= PLATE_TARGET;
    }
    if (rl->task.task == TASK_BUILD_LINE) {
        return rl->env->built[IT_BURNER_DRILL] >= 1 && rl->env->built[IT_STONE_FURNACE] >= 1 &&
               rl_sustained(rl);
    }
    return rl->verified && rl->verified_output >= VERIFY_TARGET;
}

static void rl_goal(fsim_rl *rl, float *goal) {
    const fsim_env *env = rl->env;
    memset(goal, 0, sizeof(float) * RL_GOAL_FEATURES);
    double fraction = (double)rl->steps / (double)(rl->task.max_steps > 1 ? rl->task.max_steps : 1);
    goal[0] = (float)(fraction < 1.0 ? fraction : 1.0);
    if (rl->task.task == TASK_PLATE_LINE) {
        /* Nothing is built and nothing is verified here, so the goal reports
         * the one thing that moves: how far along the plate count is. */
        double plates = (double)env->produced[IT_IRON_PLATE] / (double)PLATE_TARGET;
        goal[1] = (float)(plates < 1.0 ? plates : 1.0);
        goal[2] = env->produced[IT_IRON_PLATE] >= PLATE_TARGET ? 1.0f : 0.0f;
        /* goal[3]: the landmark "produced >= 1". */
        goal[3] = env->produced[IT_IRON_PLATE] >= 1 ? 1.0f : 0.0f;
    } else if (rl->task.task == TASK_BUILD_LINE) {
        goal[1] = env->built[IT_BURNER_DRILL] >= 1 ? 1.0f : 0.0f;
        goal[2] = env->built[IT_STONE_FURNACE] >= 1 ? 1.0f : 0.0f;
        goal[3] = rl_sustained(rl) ? 1.0f : 0.0f;
        /* goal[4]: the landmark "produced >= 1", read with empty truth: 0. */
    } else {
        goal[1] = rl->verified && rl->verified_output >= VERIFY_TARGET ? 1.0f : 0.0f;
    }
    if (rl->task.has_patch) {
        double px = tiles(env->char_pos.x), py = tiles(env->char_pos.y);
        goal[9] = (float)rl_clip((rl->task.patch_x - px) / RL_RADIUS, -1.0, 1.0);
        goal[10] = (float)rl_clip((rl->task.patch_y - py) / RL_RADIUS, -1.0, 1.0);
        goal[11] = 1.0f;
    }
}

/* The line potential (construct_smelting_line 1.2.0), phi(s) in [0, 0.9].
 *
 * Read from the published observation only -- the entities the last sweep saw,
 * the character's position and the public patch marker -- so FactorioRL's
 * `line_potential` computes the same number from the same payload:
 *
 *   0.1 * approach   max(0, 1 - |character - patch| / 64)  (CHARACTER_WITHIN)
 *   0.2 * drill      a burner drill is visible
 *   0.3 * line       a visible furnace's footprint holds a drill's drop point
 *   0.1 * each of    the best line's drill has fuel, its furnace has fuel,
 *                    its furnace holds ore or plates
 *
 * A potential of the state, so any loop through states pays nothing (Ng,
 * Harada & Russell 1999) and it is zeroed on termination (Grzes 2017). */
double fsim_rl_potential(const fsim_rl *rl) {
    const fsim_env *env = rl->env;
    double phi = 0.0;
    /* The potential's own marker, which may be private (fsim.h). */
    if (rl->task.has_target) {
        double dx = tiles(env->char_pos.x) - rl->task.target_x;
        double dy = tiles(env->char_pos.y) - rl->task.target_y;
        double approach = 1.0 - sqrt(dx * dx + dy * dy) / 64.0;
        phi += 0.1 * (approach > 0.0 ? approach : 0.0);
    }
    int drill = 0, line = 0, best = 0;
    for (int32_t i = 0; i < env->seen_count; i++) {
        const fsim_entity *d = &env->entities[env->seen[i].entity];
        if (d->kind != K_DRILL) continue;
        drill = 1;
        fsim_pos drop = drop_position(d);
        for (int32_t j = 0; j < env->seen_count; j++) {
            const fsim_entity *f = &env->entities[env->seen[j].entity];
            /* Only a furnace makes a line: a drill's fuel slot takes no ore, so
             * one drill dropping into another is a jam, and counting it let a
             * policy bank 0.3 for placing its two drills side by side. */
            if (j == i || f->kind != K_FURNACE) continue;
            if (!(drop.x >= f->pos.x - TILE && drop.x < f->pos.x + TILE &&
                  drop.y >= f->pos.y - TILE && drop.y < f->pos.y + TILE)) continue;
            line = 1;
            int score = (d->fuel.count > 0) + (f->fuel.count > 0) +
                        (f->source.count > 0 || f->result.count > 0);
            if (score > best) best = score;
        }
    }
    return phi + 0.2 * drill + 0.3 * line + 0.1 * best;
}

/* Components, by task:
 *   construct_smelting_line: [verified_output], and with shaping
 *                            [verified_output, line_potential]
 *   build_line: [constructed, plates_produced, step_cost] */
static double rl_rewards(fsim_rl *rl, int succeeded) {
    memset(rl->components, 0, sizeof(rl->components));
    if (rl->task.task == TASK_BUILD_LINE || rl->task.task == TASK_PLATE_LINE) {
        /* Same three components either way -- sparse success, a capped
         * high-water bonus on plates, a step cost -- graded to each task's own
         * plate count so neither can reach the cap while unfinished. */
        double goal_plates =
            rl->task.task == TASK_PLATE_LINE ? (double)PLATE_TARGET : (double)WINDOW_PLATES;
        rl->components[0] = succeeded ? 1.0 : 0.0;
        double value = (double)rl->env->produced[IT_IRON_PLATE];
        double previous = rl->high_water;
        double gain = value - previous > 0.0 ? value - previous : 0.0;
        rl->high_water = previous > value ? previous : value;
        double payout = (0.15 / goal_plates) * gain * 1.0;
        double room = 0.15 - rl->paid;
        payout = payout < room ? payout : room;
        if (payout < 0.0) payout = 0.0;
        rl->paid += payout;
        rl->components[1] = payout;
        rl->components[2] = -0.001;
        return rl->components[0] + rl->components[1] + rl->components[2];
    }
    rl->components[0] = succeeded ? 1.0 : 0.0;
    return rl->components[0];
}

/* One transition's shaping terms, mirroring `RewardAccountant.step`:
 * parts[0] the progress (HIGH_WATER) term, parts[1] the potential term. */
static double rl_shaping(fsim_rl *rl, int32_t mode, int terminated, double *parts) {
    double value = fsim_rl_potential(rl);
    parts[0] = parts[1] = 0.0;
    if (mode == SHAPING_POTENTIAL || mode == SHAPING_BOTH) {
        /* Grzes 2017: phi(terminal) = 0, or the sum does not telescope. */
        double next = terminated ? 0.0 : value;
        parts[1] = rl->task.gamma * next - rl->potential;
        rl->potential = next;
    }
    if (mode == SHAPING_PROGRESS || mode == SHAPING_BOTH) {
        /* HIGH_WATER: pay the rise of the running maximum, never the level, up
         * to a cumulative cap -- a line pays once however often it is rebuilt. */
        double previous = rl->progress_high;
        double gain = value - previous > 0.0 ? value - previous : 0.0;
        rl->progress_high = previous > value ? previous : value;
        double payout = PROGRESS_WEIGHT * gain * 1.0;
        double room = PROGRESS_CAP - rl->progress_paid;
        payout = payout < room ? payout : room;
        if (payout < 0.0) payout = 0.0;
        rl->progress_paid += payout;
        parts[0] = payout;
    }
    return parts[0] + parts[1];
}

static void rl_verify(fsim_rl *rl) {
    fsim_env *env = rl->env;
    double before = (double)rl_machine_produced(env, IT_IRON_PLATE);
    double source_before = (double)rl_machine_produced(env, IT_IRON_ORE);
    fsim_action wait;
    memset(&wait, 0, sizeof(wait));
    wait.verb = V_WAIT;
    int32_t chunks = VERIFY_TICKS / rl->task.decision_ticks;
    for (int32_t i = 0; i < chunks; i++) {
        fsim_step(env, &wait, rl->task.decision_ticks);
        rl_window_record(rl);
    }
    double output = (double)rl_machine_produced(env, IT_IRON_PLATE) - before;
    if (output < 0.0) output = 0.0;
    double sourced = (double)rl_machine_produced(env, IT_IRON_ORE) - source_before;
    if (sourced < 0.0) sourced = 0.0;
    rl->verified = 1;
    rl->verified_output = output < sourced ? output : sourced;
}

double fsim_rl_step(fsim_rl *rl, const int32_t *vector) {
    fsim_action action;
    rl->decode_failure = fsim_rl_decode(rl, vector, &action);
    if (rl->decode_failure) rl->decode_failures++;
    rl->steps++;
    fsim_step(rl->env, &action, rl->task.decision_ticks);
    rl_window_record(rl);
    int succeeded = rl_succeeded(rl);
    int terminated = succeeded;
    double reward = rl_rewards(rl, succeeded);
    /* Both tasks build the same line, so the same potential describes both.
     * build_line's own reward already pays for plates on a high-water mark,
     * but nothing pays for walking to the patch and putting the machines down,
     * and a policy that never produces a plate sees a constant return: measured
     * over forty million steps from scratch, exactly -0.6 at every checkpoint,
     * which is six hundred steps of step cost and no gradient at all. */
    int shaping = rl->task.shaping;
    /* The transition is scored before the verification window runs, as
     * `FactorioEnv.step` scores it; `run_verification` then scores a second
     * transition, from s' to the state after the window, which is terminal. */
    if (shaping) {
        double parts[2];
        double shaped = rl_shaping(rl, shaping, terminated, parts);
        /* components: [verified_output, line_progress or line_potential]
         * for one mode, [verified_output, line_progress, line_potential] for
         * both. */
        /* After the task's own components: build_line fills three, the other
         * task fills one. */
        int32_t at = rl->task.task == TASK_CONSTRUCT_SMELTING_LINE ? 1 : 3;
        if (shaping == SHAPING_BOTH) {
            rl->components[at] = parts[0];
            rl->components[at + 1] = parts[1];
        } else {
            rl->components[at] = shaped;
        }
        reward += shaped;
    }
    int truncated = !terminated && (rl->steps >= rl->task.max_steps ||
                                    rl->env->tick >= rl->task.construction_tick_limit);
    if (truncated && rl->task.task == TASK_CONSTRUCT_SMELTING_LINE && !rl->verified) {
        rl_verify(rl);
        double score = rl->verified_output / VERIFY_TARGET;
        reward += score < 1.0 ? score : 1.0;
        if (shaping) {
            double parts[2];
            reward += rl_shaping(rl, shaping, 1, parts);
        }
        succeeded = rl_succeeded(rl);
        terminated = 1;
        truncated = 0;
    }
    rl->reward = reward;
    rl->terminated = terminated;
    rl->truncated = truncated;
    rl->success = succeeded;
    rl->done = terminated || truncated;
    return reward;
}

fsim_rl *fsim_rl_new(void) {
    fsim_rl *rl = (fsim_rl *)calloc(1, sizeof(fsim_rl));
    rl->env = fsim_new();
    return rl;
}

void fsim_rl_free(fsim_rl *rl) {
    if (!rl) return;
    fsim_free(rl->env);
    free(rl);
}

void fsim_rl_reset(fsim_rl *rl, const fsim_task *task, const fsim_scene *scene) {
    fsim_env *env = rl->env;
    memset((char *)rl + sizeof(rl->env), 0, sizeof(*rl) - sizeof(rl->env));
    rl->task = *task;
    fsim_reset(env, scene);
    rl_window_record(rl);
    rl->high_water = (double)env->produced[IT_IRON_PLATE];
    if (rl->task.shaping) {
        rl->potential = fsim_rl_potential(rl);
        rl->progress_high = rl->potential;
    }
}

/* Decisions, each followed by the encoding and mask a policy would read. */
int32_t fsim_rl_run(fsim_rl *rl, const int32_t *vectors, int32_t count, fsim_obs *obs,
                    uint8_t *mask) {
    int32_t done = 0;
    for (int32_t i = 0; i < count && !rl->done; i++) {
        fsim_rl_step(rl, &vectors[6 * i]);
        fsim_rl_encode(rl, obs);
        fsim_rl_mask(rl, mask);
        done++;
    }
    return done;
}

void fsim_rl_step_range(fsim_rl **rls, int32_t first, int32_t last, const int32_t *actions,
                        fsim_obs *obs, uint8_t *masks, double *rewards, uint8_t *flags,
                        double *verified) {
    for (int32_t i = first; i < last; i++) {
        fsim_rl *rl = rls[i];
        rewards[i] = fsim_rl_step(rl, &actions[6 * i]);
        flags[4 * i] = (uint8_t)rl->terminated;
        flags[4 * i + 1] = (uint8_t)rl->truncated;
        flags[4 * i + 2] = (uint8_t)rl->success;
        flags[4 * i + 3] = (uint8_t)rl->decode_failure;
        verified[i] = rl->verified ? rl->verified_output : -1.0;
        fsim_rl_encode(rl, &obs[i]);
        fsim_rl_mask(rl, &masks[RL_MASK_SIZE * i]);
    }
}

void fsim_rl_encode8(fsim_rl *rl, fsim_obs8 *out) {
    memset(out, 0, sizeof(*out));
    rl_fields fields = {NULL, out->flags, out->amount, out->entities, out->entity_mask,
                        out->self_, out->inventory, out->goal};
    rl_encode_into(rl, &fields);
}

void fsim_rl_step_range8(fsim_rl **rls, int32_t first, int32_t last, const int32_t *actions,
                         fsim_obs8 *obs, uint8_t *masks, double *rewards, uint8_t *flags,
                         double *verified, double *potentials) {
    for (int32_t i = first; i < last; i++) {
        fsim_rl *rl = rls[i];
        rewards[i] = fsim_rl_step(rl, &actions[6 * i]);
        flags[4 * i] = (uint8_t)rl->terminated;
        flags[4 * i + 1] = (uint8_t)rl->truncated;
        flags[4 * i + 2] = (uint8_t)rl->success;
        flags[4 * i + 3] = (uint8_t)rl->decode_failure;
        verified[i] = rl->verified ? rl->verified_output : -1.0;
        potentials[i] = fsim_rl_potential(rl);
        fsim_rl_encode8(rl, &obs[i]);
        fsim_rl_mask(rl, &masks[RL_MASK_SIZE * i]);
    }
}

int32_t fsim_rl_targets(fsim_rl *rl, int32_t *handles, int32_t cap) {
    rl_domains d;
    rl_domains_build(rl, &d);
    int32_t n = d.target_count < cap ? d.target_count : cap;
    for (int32_t k = 0; k < n; k++) handles[k] = d.targets[k];
    return n;
}
