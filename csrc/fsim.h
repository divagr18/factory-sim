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
#define FSIM_MAX_ENTITIES 128
#define FSIM_MAX_RESOURCES 512
#define FSIM_MAX_WATER 8192
#define FSIM_MAIN_SLOTS 80
#define FSIM_MAX_HANDLES 2048
#define FSIM_MAX_INFLIGHT 16
#define FSIM_EVENT_LIMIT 256
#define FSIM_MAX_SUPERSEDED 4
#define FSIM_MAX_SWEEP 48
#define FSIM_MAX_TILES 512
#define FSIM_MAX_MEMORY 256
#define FSIM_MAX_BLOCKED 4225

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

/* Engine status names, by the codes the game reports. */
#define ST_NONE 0
#define ST_WORKING 1
#define ST_NO_INGREDIENTS 18
#define ST_WAITING_FOR_SPACE 34
#define ST_NO_FUEL 53
#define ST_NO_MINABLE 30

/* Verbs, as the mod's action names. */
#define V_WAIT 0
#define V_MOVE 1
#define V_MINE 2
#define V_PLACE 3
#define V_ROTATE 4
#define V_TRANSFER 5

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
    int32_t products_finished;
    /* progress: seconds accumulated towards the current craft or ore */
    double seconds;
    double progress;
    /* drill */
    int32_t held;         /* item mined and not yet delivered */
    int32_t linked_unit;  /* the machine it has delivered into before, or 0 */
    int32_t mine_cursor;  /* which of its four tiles it is mining */
    int32_t mine_count;   /* ore taken from that tile so far */
    /* pile */
    fsim_stack pile;
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

    fsim_entity entities[128];
    int32_t entity_count;
    int32_t next_unit;
    fsim_resource resources[512];
    int32_t resource_count;

    fsim_handle handles[2048];
    int32_t next_handle;

    fsim_inflight inflight[16];
    int32_t slot_move;          /* inflight index, -1 */
    int32_t slot_mine;
    int32_t slot_advance;
    int32_t superseded[4];
    int32_t superseded_count;

    fsim_memory memory[256];

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
    int32_t remembered[256];    /* memory indices, in published order */
    fsim_pos origin;

    /* the step in progress */
    int32_t act_step;           /* request number of the current step */
    fsim_act_result act;
    int32_t act_inflight;       /* inflight index of its ongoing action, -1 */
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
int32_t fsim_resolve(fsim_env *env, int32_t handle, int32_t *kind, int32_t *index);
double fsim_capacity(int32_t kind);
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

#define SHAPING_NONE 0
#define SHAPING_POTENTIAL 1
#define SHAPING_PROGRESS 2

typedef struct {
    float grid[25350];          /* 6 x 65 x 65 */
    float entities[512];        /* 32 x 16 */
    int8_t entity_mask[32];
    float self_[12];
    float inventory[14];
    float goal[12];
} fsim_obs;

typedef struct {
    int32_t task;
    int32_t decision_ticks;
    int32_t max_steps;
    int32_t construction_tick_limit;
    int32_t has_patch;
    double patch_x;
    double patch_y;
    /* construct_smelting_line shaping over the line potential phi:
     *   SHAPING_NONE      1.1.1, the verification score alone
     *   SHAPING_POTENTIAL gamma * phi(s') - phi(s), phi(terminal) = 0
     *   SHAPING_PROGRESS  a HIGH_WATER component on phi, weight 0.5, cap 0.45 */
    int32_t shaping;
    double gamma;
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
    double components[3];       /* task-specific; see fsim_rl.c */
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
/* CFFI-END */

#endif
