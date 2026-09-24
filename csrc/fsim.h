/* factory-sim: a tick-exact simulator of a small early-game factory.
 *
 * Every rule here was measured on Factorio 2.0.60 by FactorioRL
 * (docs/evidence/sim-mechanics-m1.json, sim-mechanics-m3.json and the golden
 * traces under tests/golden). Nothing is copied from the game.
 *
 * Units: positions are int32 in 1/256 tiles; ticks are int64; energy is
 * joules as double; progress bars accumulate seconds and are reported as
 * seconds / duration, which is how the measured values round.
 *
 * The block between the CFFI markers is parsed by cffi and must stay plain
 * C declarations with integer #defines only.
 */
#ifndef FSIM_H
#define FSIM_H

#include <stdint.h>

/* CFFI-BEGIN */
/* Capacities. cffi cannot size an array with a #define, so the struct fields
 * below repeat these as literals; change both together. FSIM_EVENT_LIMIT,
 * FSIM_MAX_SWEEP and FSIM_MAX_TILES are the engine's own limits (the mod's
 * EVENT_LIMIT, `entity_cap` and `resource_cap`) and part of what parity
 * compares; FSIM_MAX_MEMORY is the mod's MEMORY_LIMIT. The others are the
 * simulator's own room: entity slots are never reused within an episode, so
 * FSIM_MAX_ENTITIES counts every entity created, ground piles included. */
#define FSIM_MAX_ENTITIES 512
#define FSIM_MAX_RESOURCES 2048
#define FSIM_MAX_WATER 8192
#define FSIM_MAIN_SLOTS 80
#define FSIM_MAX_HANDLES 8192
#define FSIM_MAX_INFLIGHT 16
#define FSIM_EVENT_LIMIT 256
#define FSIM_MAX_SUPERSEDED 4
#define FSIM_MAX_SWEEP 48
#define FSIM_MAX_TILES 512
#define FSIM_MAX_MEMORY 2048
#define FSIM_MAX_BLOCKED 4225
#define FSIM_MAX_FILLERS 1024
#define FSIM_MAX_LANES 1024       /* two per entity */
#define FSIM_CHEST_SLOTS 16
#define FSIM_LANE_ITEMS 8         /* items one belt lane can hold (fsim.c, lane_insert) */

/* Items. Order is fixed: it is the simulator's own numbering, not the
 * encoder's. Names live in fsim/items.py. */
#define IT_NONE 0
#define IT_IRON_ORE 1
#define IT_COPPER_ORE 2
#define IT_COAL 3
#define IT_STONE 4
#define IT_IRON_PLATE 5
#define IT_COPPER_PLATE 6
#define IT_STONE_FURNACE 7
#define IT_BURNER_DRILL 8
#define IT_STONE_WALL 9
#define IT_WOOD 10
#define IT_IRON_GEAR 11
#define IT_TRANSPORT_BELT 12
#define IT_SMALL_POLE 13
#define IT_WOODEN_CHEST 14
#define IT_BURNER_INSERTER 15
#define IT_COUNT 16

/* Entity kinds. */
#define K_NONE 0
#define K_DRILL 1
#define K_FURNACE 2
#define K_WALL 3
#define K_PILE 4
#define K_CHEST 5
#define K_BELT 6
#define K_INSERTER 7
#define K_COUNT 8

/* What a kind is, as flags (fsim_kind_flags). Everything else constant about
 * a kind is its row in fsim.c's KIND table. */
#define KF_COLLIDES 1         /* occupies its footprint: placement collides with it */
#define KF_BLOCKS_WALKING 2   /* the character collides with it */
#define KF_BURNER 4           /* a fuel slot and an energy buffer */
#define KF_MACHINE 8          /* has inventories: transfers and a drill's output reach it */
#define KF_DIRECTED 16        /* its direction matters: rotation, memory, encoder */

/* Engine status names, by the codes the game reports. */
#define ST_NONE 0
#define ST_WORKING 1
#define ST_NORMAL 2
#define ST_NO_INGREDIENTS 18
#define ST_WAITING_FOR_SPACE 34
#define ST_NO_FUEL 53
#define ST_NO_MINABLE 30
#define ST_WAITING_FOR_SOURCE 32

/* Verbs, as the mod's action names. */
#define V_WAIT 0
#define V_MOVE 1
#define V_MINE 2
#define V_PLACE 3
#define V_ROTATE 4
#define V_TRANSFER 5

/* Belt shapes, as the engine's belt_shape. */
#define BELT_STRAIGHT 0
#define BELT_LEFT 1
#define BELT_RIGHT 2

/* Where a burner inserter's hand is in its cycle (fsim.c, update_inserter). */
#define INS_APPROACH 0        /* as built: the hand extends to the pickup */
#define INS_TO_DROP 1         /* holding, swinging to the drop */
#define INS_TO_PICKUP 2       /* empty, swinging back */
#define INS_WAIT_PICKUP 3     /* at the pickup, nothing to take */
#define INS_WAIT_DROP 4       /* at the drop, nowhere to put it */
#define INS_TO_SELF 5         /* holding fuel for its own empty fuel slot */
#define INS_SELF_BACK 6       /* from its own fuel slot back to the pickup */

/* Request results. */
#define R_NONE 0
#define R_COMPLETED 1
#define R_RUNNING 2
#define R_REJECTED 3
#define R_FAILED 4
#define R_CANCELLED 5

/* Error codes, as protocol.lua names them. */
#define E_NONE 0
#define E_PRECONDITION 1
#define E_OUT_OF_REACH 2
#define E_COLLISION 3
#define E_NO_ITEMS 4
#define E_NO_SPACE 5
#define E_BUSY 6
#define E_UNKNOWN_HANDLE 7
#define E_TARGET_MISSING 8
#define E_NOT_MINEABLE 9
#define E_INVALID_TARGET 10
#define E_TECH_LOCKED 11
#define E_ENGINE 12

/* Handle kinds. */
#define H_UNIT 1
#define H_TILE 2

/* Tile handle types. */
#define TT_RESOURCE 1
#define TT_PILE 2

typedef struct {
    int32_t x;
    int32_t y;
} fsim_pos;

typedef struct {
    int32_t item;
    int32_t count;
} fsim_stack;

typedef struct {
    int32_t alive;
    int32_t item;     /* the ore it yields */
    int32_t tx, ty;   /* tile; the entity sits at the tile centre */
    int32_t amount;
} fsim_resource;

/* One item on a belt lane. Positions are 1/256 tile from the downstream end
 * of this belt's own lane, as the engine's get_line_item_position reads them. */
typedef struct {
    int16_t pos;
    uint8_t item;
    uint8_t moved;        /* it moved in this tick's belt update */
    int32_t id;           /* the simulator's own item number, not the engine's */
} fsim_belt_item;

typedef struct {
    int32_t count;
    fsim_belt_item items[8];  /* FSIM_LANE_ITEMS, front (lowest position) first */
} fsim_lane;

typedef struct {
    int32_t alive;
    int32_t kind;
    int32_t neutral;      /* force: 0 player, 1 neutral */
    int32_t unit;         /* creation number; the unit identity handles key on */
    fsim_pos pos;
    int32_t direction;    /* 16-way */
    int32_t status;
    /* burner */
    double energy;
    double remaining;
    int32_t burning;      /* item mid-burn, or IT_NONE */
    fsim_stack fuel;
    /* furnace */
    fsim_stack source;
    fsim_stack result;
    int32_t crafting;     /* an ingredient has been consumed for the current craft */
    int32_t ingredient;   /* ...and which item it was */
    int32_t products_finished;
    /* progress: seconds accumulated towards the current craft or ore */
    double seconds;
    double progress;
    /* drill; inserter */
    int32_t held;         /* item mined and not yet delivered; the item in an inserter's hand */
    int32_t linked_unit;  /* the machine it has delivered into before, or 0 */
    int32_t mine_cursor;  /* which of its four tiles it is mining */
    int32_t mine_count;   /* ore taken from that tile so far */
    /* pile */
    fsim_stack pile;
    /* chest */
    fsim_stack chest[16];         /* FSIM_CHEST_SLOTS */
    /* inserter */
    int32_t phase;                /* INS_* */
    double swing;                 /* ticks of the current move done; fractional after a short tick */
    /* belt */
    fsim_lane lanes[2];           /* lane 1 (left of travel), lane 2 */
    /* Derived from the neighbours, rebuilt whenever entities change
     * (fsim.c, rebuild_logistics); a hidden-state load rebuilds them too. */
    int32_t shape;                /* BELT_* */
    int32_t lane_length[2];       /* 256 straight; 295 / 106 on a turn */
    int32_t lane_next[2];         /* lane ref (entity * 2 + lane) it runs into, or -1 */
    int32_t lane_side[2];         /* lane ref it sideloads into, or -1 */
    int32_t lane_entry[2];        /* ...at this position on that lane */
    int32_t pickup_target;        /* inserter: entity at its pickup point, or -1 */
    int32_t drop_target;          /* ...and at its drop point */
} fsim_entity;

typedef struct {
    int32_t used;
    int32_t kind;         /* H_UNIT / H_TILE */
    int32_t unit;         /* H_UNIT */
    int32_t tx, ty;       /* H_TILE */
    int32_t tile_type;    /* TT_* */
    int32_t name;         /* item id of a resource, or entity kind, see name_code */
    int64_t first_seen;
    int64_t destroyed_tick;  /* -1: not destroyed */
    int32_t last_pos_x, last_pos_y; /* units: last known position */
} fsim_handle;

typedef struct {
    int32_t used;
    int32_t terminal;
    int32_t step;         /* the step request number this belongs to */
    int32_t seq;
    int32_t verb;         /* V_MOVE, V_MINE, or -1 for advance */
    int64_t started_tick;
    int64_t deadline_tick;/* -1 if none */
    int32_t target;       /* handle number (mine) */
    int32_t goal_item;
    int32_t goal_count;
    int32_t baseline;
    int32_t queued;
    int32_t result;       /* R_* once settled */
    int32_t move_dir;     /* for moves */
} fsim_inflight;

typedef struct {
    int32_t seq;
    int32_t step;         /* request number */
    int32_t is_act;       /* request id carries ":act" */
    int64_t tick;
    int32_t status;       /* R_* */
    int32_t action;       /* V_* or -1 when absent */
    int32_t result;       /* R_* or R_NONE when absent */
    int32_t error;        /* E_* */
} fsim_event;

typedef struct {
    int32_t handle;
    int32_t entity;       /* index into entities */
    int64_t d2;           /* squared distance in 1/65536 tiles^2 */
} fsim_seen;

typedef struct {
    int32_t handle;
    int32_t resource;     /* index into resources */
} fsim_seen_tile;

typedef struct {
    int32_t used;
    int32_t handle;
    int32_t kind;
    int32_t name;
    fsim_pos pos;
    int32_t has_dir;
    int32_t direction;
    fsim_stack contents;  /* the one item a record of these kinds can hold */
    int64_t last_seen;
} fsim_memory;

typedef struct {
    int32_t verb;
    int32_t direction;    /* 0..3: north, east, south, west */
    int32_t ticks;
    int32_t handle;       /* 0: none; -1: a string that names no handle */
    int32_t from_handle;  /* 0: character; -1: unknown; else handle number */
    int32_t to_handle;
    int32_t item;
    int32_t count;
    int32_t reverse;
    fsim_pos position;
} fsim_action;

typedef struct {
    /* what the step's response reports about its action */
    int32_t status;       /* R_* */
    int32_t error;        /* E_* */
    int32_t verb;
    int32_t count;        /* transfer: inserted */
    int32_t requested;    /* transfer: set when clamped */
    int32_t available;
    int32_t handle;       /* place: the new entity's handle */
    fsim_pos position;    /* place: where it landed */
} fsim_act_result;

typedef struct {
    int64_t tick;               /* episode tick */
    int32_t step_counter;       /* request numbers */
    int32_t next_seq;
    int32_t event_seq;
    int32_t event_count;
    int32_t event_head;
    fsim_event events[256];

    /* character */
    fsim_pos char_pos;
    int32_t char_dir8;          /* LuaEntity.direction */
    int32_t walk_set;           /* walking_state as the script last wrote it */
    int32_t walk_set_dir;
    int32_t walk_pub;           /* ...and as reads return it */
    int32_t walk_pub_dir;
    int32_t mining;
    fsim_pos mining_pos;
    int32_t mining_target_entity;   /* entity index, or -1 */
    int32_t mining_target_resource; /* resource index, or -1 */
    double mining_seconds;
    double mining_progress;
    int32_t mined_kind;         /* what the kept progress belongs to: 1 entity, 2 resource */
    int32_t mined_index;
    int32_t selected_kind;      /* 0 none, 1 entity, 2 resource */
    int32_t selected_index;
    fsim_stack main[80];

    fsim_entity entities[512];
    int32_t entity_count;
    int32_t next_unit;
    /* Gap fillers: the space between two aligned obstacles too close for the
     * character to pass, as boxes [x0, x1] x [y0, y1]. Rebuilt when
     * `entities_version` moves past `fillers_version`. */
    int32_t entities_version;
    int32_t fillers_version;
    int32_t filler_count;
    int32_t fillers[4096];      /* FSIM_MAX_FILLERS boxes, 4 values each */
    fsim_resource resources[2048];
    int32_t resource_count;

    fsim_handle handles[8192];
    int32_t next_handle;

    fsim_inflight inflight[16];
    int32_t slot_move;          /* inflight index, -1 */
    int32_t slot_mine;
    int32_t slot_advance;
    int32_t superseded[4];
    int32_t superseded_count;

    fsim_memory memory[2048];
    int32_t memory_top;         /* slots at and above this are unused */

    /* truth */
    int32_t produced[16];
    int32_t mined_by_action[16];
    int32_t built[16];
    int32_t transfers;
    int32_t items_moved;
    int32_t steam_power;        /* the trigger technology has been researched */
    int32_t plates_crafted;     /* its counter */
    int64_t steam_power_at;     /* the tick it will be researched, or 0 */

    /* terrain: water tiles, sorted (y, x) */
    int32_t water_count;
    int32_t water[16384];

    /* last observation */
    int32_t seen_count;
    fsim_seen seen[48];
    int32_t tile_count;
    fsim_seen_tile tiles[512];
    int32_t blocked_count;
    int32_t blocked[8450];
    int32_t remembered_count;
    int32_t remembered[2048];   /* memory indices, in published order */
    fsim_pos origin;

    /* the step in progress */
    int32_t act_step;           /* request number of the current step */
    fsim_act_result act;
    int32_t act_inflight;       /* inflight index of its ongoing action, -1 */

    /* Belts and inserters: the update schedule, rebuilt when `entities_version`
     * moves past `logistics_version` (fsim.c, rebuild_logistics). */
    int32_t logistics_version;
    int32_t inserter_count;
    int32_t inserters[512];     /* entity indices, in creation order */
    int32_t chain_count;
    int32_t chain_first[1024];  /* FSIM_MAX_LANES: a chain's first lane in chain_lanes */
    int32_t chain_size[1024];   /* its lanes; negative for a closed loop */
    int32_t chain_lanes[1024];  /* lane refs, each chain front (downstream) first */
    int32_t next_item_id;
} fsim_env;

typedef struct {
    int32_t resource_count;
    int32_t *resource_item;
    int32_t *resource_tx;
    int32_t *resource_ty;
    int32_t *resource_amount;
    int32_t wall_count;
    int32_t *wall_x;            /* declared positions, 1/256 */
    int32_t *wall_y;
    /* Entities the scene places rather than the agent: plate_line is handed a
     * drill and a furnace already aligned, both empty, which is what
     * `new_entity` gives them (ST_NO_FUEL); logistics scenes place belts,
     * chests and inserters. Declared positions are already the centres the
     * entity snaps to, so they are used as given. */
    int32_t machine_count;
    int32_t *machine_kind;
    int32_t *machine_x;         /* declared positions, 1/256 */
    int32_t *machine_y;
    int32_t *machine_dir;
    /* What those entities are given once built, as `LuaEntity.insert` calls in
     * this order (fsim_entity_insert): fuel, ore, a chest's contents. */
    int32_t content_count;
    int32_t *content_machine;   /* index into the machine arrays */
    int32_t *content_item;
    int32_t *content_amount;
    fsim_pos character;         /* already truncated to 1/256 */
    int32_t inventory_count;    /* already in insertion order */
    int32_t *inventory_item;
    int32_t *inventory_amount;
} fsim_scene;

fsim_env *fsim_new(void);
void fsim_free(fsim_env *env);
void fsim_set_water(fsim_env *env, const int32_t *xy, int32_t count);
void fsim_reset(fsim_env *env, const fsim_scene *scene);
/* One decision: apply the action at the current tick, run `ticks` ticks,
 * then take the observation. Returns the episode tick. */
int64_t fsim_step(fsim_env *env, const fsim_action *action, int32_t ticks);
/* The observation a reset or a step takes. */
void fsim_observe(fsim_env *env);
int64_t fsim_run(fsim_env *env, const fsim_action *actions, int32_t count, int32_t ticks);
/* Recompute derived character state after a hidden-state load. */
void fsim_after_load(fsim_env *env);
/* Walk `ticks` ticks in `dir16` (0, 4, 8, 12) with nothing else running, and
 * write the position after each into `xy` (2 per tick). For collision tests. */
void fsim_walk_ticks(fsim_env *env, int32_t dir16, int32_t ticks, int32_t *xy);
int32_t fsim_resolve(fsim_env *env, int32_t handle, int32_t *kind, int32_t *index);
double fsim_capacity(int32_t kind);
/* Script-side construction, as LuaSurface.create_entity and friends: what
 * scenes and the mechanics tests build worlds with. An entity at `x, y`
 * (1/256, used as given) facing `dir16`; its index, or -1 when full. */
int32_t fsim_add_entity(fsim_env *env, int32_t kind, int32_t x, int32_t y, int32_t dir16);
/* LuaEntity.insert: fuel to a burner's fuel slot, anything else to a
 * furnace's source or a chest. Returns what went in. */
int32_t fsim_entity_insert(fsim_env *env, int32_t index, int32_t item, int32_t count);
/* LuaTransportLine.insert_at / insert_at_back on belt `index`, lane 1 or 2
 * (0-based here): 1 when the item went on. Both see the line as it stands
 * between ticks and move the item once, as the engine's readback does. */
int32_t fsim_belt_insert(fsim_env *env, int32_t index, int32_t lane, int32_t position,
                         int32_t item);
int32_t fsim_belt_insert_back(fsim_env *env, int32_t index, int32_t lane, int32_t item);
/* Bring belt shapes and links and inserter targets up to date with the
 * entities; ticks do this themselves, a render between them calls it. */
void fsim_refresh(fsim_env *env);
/* Run `ticks` world ticks with no request in flight. */
void fsim_advance(fsim_env *env, int32_t ticks);
/* KF_* flags of an entity kind, and the seconds the character takes to mine one. */
int32_t fsim_kind_flags(int32_t kind);
double fsim_kind_mining_time(int32_t kind);
/* ---- RL layer (fsim_rl.c): the tensors, masks, goal and reward of
 * FactorioRL's parameterized-v1 / local-v2 contract. */
#define RL_GRID_PLANES 6
#define RL_GRID_SIZE 65
#define RL_MAX_ENTITIES 32
#define RL_ENTITY_FEATURES 16
#define RL_SELF_FEATURES 12
#define RL_ITEMS 14
#define RL_GOAL_FEATURES 12
#define RL_OPERATIONS 22
#define RL_TARGETS 32
#define RL_PLACEMENTS 121
#define RL_MASK_SIZE 201
#define RL_WINDOW_SAMPLES 512

#define TASK_CONSTRUCT_SMELTING_LINE 1
#define TASK_BUILD_LINE 2
/* Commissioning rather than construction: the line is already down and both
 * machines are empty, and the agent has to reach each one and fuel it. */
#define TASK_PLATE_LINE 3

#define ACTION_SPACE_V1 0
#define ACTION_SPACE_V2 2

#define SHAPING_NONE 0
#define SHAPING_POTENTIAL 1
#define SHAPING_PROGRESS 2
#define SHAPING_BOTH 3

typedef struct {
    float grid[25350];          /* 6 x 65 x 65 */
    float entities[512];        /* 32 x 16 */
    int8_t entity_mask[32];
    float self_[12];
    float inventory[14];
    float goal[12];
} fsim_obs;

/* The same observation, packed: what a trainer copies to the GPU every
 * decision, a third of the size. Grid planes 0-3 (resources) and 5 (blocked)
 * only ever hold 0 or 1, so they are bits -- plane p's cell c is bit
 * (k * 4225 + c) of `flags`, LSB first, for k = 0..4 over planes 0, 1, 2, 3, 5.
 * Plane 4 (the log amount, in [0, 1]) is `amount`, round(255 * value), half to
 * even. */
#define RL_FLAG_BYTES 2641
typedef struct {
    uint8_t flags[2641];
    uint8_t amount[4225];
    float entities[512];
    int8_t entity_mask[32];
    float self_[12];
    float inventory[14];
    float goal[12];
} fsim_obs8;

typedef struct {
    int32_t task;
    int32_t decision_ticks;
    int32_t max_steps;
    int32_t construction_tick_limit;
    /* The marker the goal vector points at. Only a *public* marker belongs
     * here: goal[9..11] is shown to the policy. */
    int32_t has_patch;
    double patch_x;
    double patch_y;
    /* The marker the potential measures approach to, read from the scene's
     * truth and not necessarily public -- FactorioRL's
     * `line_potential(observation, truth, marker)` reads truth for the same
     * reason. Potential-based shaping cannot change which policy is optimal,
     * so privileged information in phi is sound where the same information in
     * the observation would be a leak. For the two construction tasks these
     * are the same public "patch" marker. */
    int32_t has_target;
    double target_x;
    double target_y;
    /* construct_smelting_line shaping over the line potential phi:
     *   SHAPING_NONE      1.1.1, the verification score alone
     *   SHAPING_POTENTIAL gamma * phi(s') - phi(s), phi(terminal) = 0
     *   SHAPING_PROGRESS  a HIGH_WATER component on phi, weight 0.5, cap 0.45
     *   SHAPING_BOTH      both: progress pays for reaching a stage once, the
     *                     potential charges for leaving it */
    int32_t shaping;
    double gamma;
    /* ACTION_SPACE_V1: FactorioRL's parameterized-v1 (the default).
     * ACTION_SPACE_V2: a simulator prototype with the same vector shape, where
     *   `target` k names row k-1 of the encoded entity table and `placement` p
     *   names the fixed tile ((p-1) / 11 - 5, (p-1) % 11 - 5) from the
     *   character's tile, masked when occupied instead of skipped. */
    int32_t action_space;
} fsim_task;

typedef struct {
    fsim_env *env;
    fsim_task task;
    int32_t steps;
    int32_t done;
    int32_t window_count;
    int32_t window_head;
    int64_t window_tick[512];
    int32_t window_plates[512];
    double high_water;
    double paid;
    int32_t verified;
    double verified_output;
    int32_t decode_failures;
    /* the last transition */
    double reward;
    int32_t terminated;
    int32_t truncated;
    int32_t success;
    /* Task-specific, then the shaping terms after them: one component for
     * construct_smelting_line and three for build_line, so the shaped terms
     * start at slot 1 or slot 3. See `rl_rewards` and `fsim_rl_step`. */
    double components[5];
    int32_t decode_failure;
    double potential;           /* SHAPING_POTENTIAL: phi of the current state */
    double progress_high;       /* SHAPING_PROGRESS: highest phi so far */
    double progress_paid;       /* ...and what it has paid */
} fsim_rl;

fsim_rl *fsim_rl_new(void);
void fsim_rl_free(fsim_rl *rl);
void fsim_rl_reset(fsim_rl *rl, const fsim_task *task, const fsim_scene *scene);
/* One decision from a MultiDiscrete vector [op, target, placement, direction,
 * item, amount]. Returns the reward; flags are in `rl`. */
double fsim_rl_step(fsim_rl *rl, const int32_t *vector);
void fsim_rl_encode(fsim_rl *rl, fsim_obs *obs);
void fsim_rl_mask(fsim_rl *rl, uint8_t *mask);
/* Decode a vector into an action; returns 0, or 1 for a decode failure (the
 * action is then a wait). */
int32_t fsim_rl_decode(fsim_rl *rl, const int32_t *vector, fsim_action *out);
int32_t fsim_rl_run(fsim_rl *rl, const int32_t *vectors, int32_t count, fsim_obs *obs,
                    uint8_t *mask);
/* The line potential phi(s) in [0, 1], from the published observation. */
double fsim_rl_potential(const fsim_rl *rl);
/* Environments [first, last) of a batch: step each with its row of `actions`
 * (6 per env), then write its observation, mask and transition. A caller runs
 * disjoint ranges on separate threads. */
void fsim_rl_step_range(fsim_rl **rls, int32_t first, int32_t last, const int32_t *actions,
                        fsim_obs *obs, uint8_t *masks, double *rewards, uint8_t *flags,
                        double *verified);
void fsim_rl_encode8(fsim_rl *rl, fsim_obs8 *obs);
/* The target argument's domain, in order: handle of target k+1. Returns the
 * count (at most `cap`). */
int32_t fsim_rl_targets(fsim_rl *rl, int32_t *handles, int32_t cap);
/* As fsim_rl_step_range, writing compact observations and each environment's
 * line potential after the step. */
void fsim_rl_step_range8(fsim_rl **rls, int32_t first, int32_t last, const int32_t *actions,
                         fsim_obs8 *obs, uint8_t *masks, double *rewards, uint8_t *flags,
                         double *verified, double *potentials);
/* CFFI-END */

#endif
