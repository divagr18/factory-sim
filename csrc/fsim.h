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
#define FSIM_MAX_SWEEP 96
/* The sweep's cap unless an env sets its own (`sweep_cap`): local-v2's
 * entity_cap. local-v3 raises it to 96. */
#define FSIM_SWEEP_DEFAULT 48
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
    uint8_t entered;      /* crossed this tick into a segment still to move: how far */
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
    double swing;                 /* ticks of the current move done */
    /* The arm (fsim.c, arm_step): orientation, turns clockwise from north in
     * [0, 1), kept in single precision as the engine keeps it, and length in
     * tiles. While the hand rests on its target, the target's offset from the
     * inserter (1/256 tile) is where it is drawn. */
    float arm_w;
    double arm_len;
    int32_t arm_at;
    double arm_vx, arm_vy;
    int32_t chase_id;             /* the belt item it is chasing, or 0 */
    int32_t lift;                 /* the drawn lift in use: 0 none, 1 a swing, 2 to itself, -1 unknown */
    int32_t lift_step;            /* ticks into that move */
    int32_t hand_x, hand_y;       /* held_stack_position less the position, 1/256 */
    int32_t belt_asleep;          /* asleep on its pickup belt's segment (rule 6) */
    int64_t woke_tick;            /* the tick an item on that segment woke it */
    int64_t look_when;            /* tick + 1 an item added to its segment made it look, or 0 */
    /* belt */
    fsim_lane lanes[2];           /* lane 1 (left of travel), lane 2 */
    /* Belt-line segments (fsim.c, "segments"): when it was built, each lane's
     * merge delay from the measured table (-1: the table has no entry), and
     * the tick each lane's merge timer runs out, or 0 when none is running. */
    int64_t built_tick;
    /* Its boundaries (fsim.c, "boundaries"): per slot -- an inserter's
     * pickup belt lanes 0 and 1 and its drop lane 2, a drill's output 2, a
     * feed front's sideload 0 -- the tick + 1 it was first set off, or 0,
     * and the tick its split is due, or 0. */
    int64_t bnd_trig[3];
    int64_t bnd_split[3];
    int64_t merge_at[2];
    int32_t delay[2];
    int32_t seg_new;              /* built since the last rebuild_logistics */
    int32_t seg_rot;              /* turned since the last rebuild_logistics, at rot_tick */
    int64_t rot_tick;
    /* Derived from the neighbours, rebuilt whenever entities change
     * (fsim.c, rebuild_logistics); a hidden-state load rebuilds them too. */
    int32_t shape;                /* BELT_* */
    int32_t lane_length[2];       /* 256 straight; 295 / 106 on a turn */
    int32_t lane_next[2];         /* lane ref (entity * 2 + lane) it runs into, or -1 */
    int32_t lane_side[2];         /* lane ref it sideloads into, or -1 */
    int32_t lane_entry[2];        /* ...at this position on that lane */
    int32_t pickup_target;        /* inserter: entity at its pickup point, or -1 */
    int32_t drop_target;          /* ...and at its drop point */
    /* inserter: 0 while in the update list; else when it fell asleep, waiting
     * on a machine, until that machine's contents change (fsim.c, wake) */
    int64_t sleep_seq;
    /* drill: its drop belt's links when its output was refused (fsim.c,
     * drill_block_signature) */
    int32_t block_sig;
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
    fsim_stack contents;  /* a pile's or a furnace's one stack; a chest's first item */
    uint16_t amounts[16]; /* IT_COUNT: a chest's whole contents, item by item */
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
    fsim_seen seen[96];         /* FSIM_MAX_SWEEP; the first sweep_cap are used */
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
    int32_t inserters[512];     /* the inserters awake, run last first (fsim.c, update_world) */
    int64_t sleep_counter;
    int32_t sleeper_count;      /* inserters asleep */
    int32_t chain_count;
    int32_t chain_first[1024];  /* FSIM_MAX_LANES: a chain's first lane in chain_lanes */
    int32_t chain_size[1024];   /* its lanes; negative for a closed loop */
    int32_t chain_lanes[1024];  /* lane refs, each chain front (downstream) first */
    int32_t lane_chain[1024];   /* the chain a lane ref is in, or -1 */
    int32_t belt_sleepers;      /* inserters asleep on a belt segment */
    int32_t next_item_id;
    /* Belt-line segments (fsim.c, "segments"), by lane ref (entity * 2 + lane).
     * A segment is named by its head, its downstream-most lane; the arrays
     * marked "by head" mean something only at a head. */
    int32_t lane_pos[1024];     /* a lane's index in its chain, the front 0 */
    int32_t seg_head[1024];     /* the head of the segment a lane is in, or -1 */
    int32_t seg_join[1024];     /* the lane downstream it is merged with, or -1 */
    uint8_t seg_cut[1024];      /* a boundary in force lies between it and the lane downstream */
    uint8_t seg_listed[1024];   /* by head: in seg_order (awake) */
    uint8_t seg_sleep[1024];    /* by head: holds items, none of which could move */
    uint8_t seg_busy[1024];     /* by head: moving now (move_segment) */
    uint8_t seg_wrap[1024];     /* its join is a closed loop's seam, joined all round */
    int32_t side_first[1024];   /* the first chain front that sideloads into a lane, or -1 */
    int32_t side_link[1024];    /* the next one after a front, or -1 */
    int32_t belts_changed;      /* a belt was built, removed or turned since the last rebuild */
    int64_t seg_moved[1024];    /* by head: the tick it last moved */
    int32_t seg_count;          /* entries in seg_order */
    int32_t seg_order[2048];    /* awake segments, by head, oldest activation first; -1 gone */
    int64_t seg_next_timer;     /* the earliest merge or split due, or 0 */
    int32_t belt_phase;         /* the belts are moving (update_belts) */
    int32_t updating;           /* inside a world tick (update_world) */
    /* Edges a belt built, removed or turned cuts at the next rebuild
     * (fsim.c, seg_cut_radius): the lane, flags (1: it was merged across the
     * edge, 2: the edge is this lane's downstream one, to cut), the tick. */
    int32_t seg_pend_count;
    int32_t seg_pend[4 * FSIM_MAX_LANES];
    uint8_t seg_pend_joined[4 * FSIM_MAX_LANES];
    int64_t seg_pend_tick[4 * FSIM_MAX_LANES];
    /* Belts built on a tile the merge-delay table does not cover: counted,
     * and the first such tile kept. Their lanes never merge; the Python layer
     * refuses to go on (fsim.BeltDelayMissing). */
    int32_t belt_delay_missing;
    int32_t belt_delay_missing_x;
    int32_t belt_delay_missing_y;
    /* The observation's entity cap (the profile's entity_cap), at most
     * FSIM_MAX_SWEEP; 0 means FSIM_SWEEP_DEFAULT. Kept across fsim_reset. */
    int32_t sweep_cap;
    /* Hand-mining picked an entity up while mining stays asked for: nothing
     * is selected again until mining stops (fsim.c, update_character). */
    int32_t mining_stalled;
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
/* A script takes item `at` (0: the front) off belt `index`'s lane (0-based):
 * 1 when there was one. Its segment wakes, as after an inserter's pickup. */
int32_t fsim_belt_remove(fsim_env *env, int32_t index, int32_t lane, int32_t at);
/* LuaEntity.remove_item on a chest (slot 1 first); returns what came out. */
int32_t fsim_entity_remove(fsim_env *env, int32_t index, int32_t item, int32_t count);
/* Put inserter `index` `step` ticks into move `phase` (INS_*) from where that
 * move starts, as a hidden-state load reads it off the drawn hand. */
void fsim_inserter_set(fsim_env *env, int32_t index, int32_t phase, int32_t step);
/* LuaEntity.held_stack.set_stack on inserter `index`: `item` in its hand. */
void fsim_inserter_hold(fsim_env *env, int32_t index, int32_t item);
/* A script changed entity `index`'s contents directly: what the engine
 * notifies (waiting inserters, blocked drills) is told. */
void fsim_script_touched(fsim_env *env, int32_t index);
/* LuaEntity.destroy on the item pile `index`. */
void fsim_remove_pile(fsim_env *env, int32_t index);
/* LuaEntity.rotate{reverse} on belt or inserter `index`: a quarter turn
 * clockwise, or anticlockwise when `reverse`. 1 when it turned. */
int32_t fsim_script_rotate(fsim_env *env, int32_t index, int32_t reverse);
/* LuaEntity.destroy on belt `index`: its items go with it. 1 when destroyed. */
int32_t fsim_script_destroy(fsim_env *env, int32_t index);
/* Bring belt shapes and links and inserter targets up to date with the
 * entities; ticks do this themselves, a render between them calls it. */
void fsim_refresh(fsim_env *env);
/* Run `ticks` world ticks with no request in flight. */
void fsim_advance(fsim_env *env, int32_t ticks);
/* The measured belt merge delay, shared by every env: `count` rectangles, each
 * x0, y0, x1, y1 (tiles, half-open) in `rects`, and `values` holding, rectangle
 * by rectangle, lane 1's delays then lane 2's, row by row (y, then x). The
 * table is copied. Returns 0, or -1 when out of memory. */
int32_t fsim_set_belt_delay(int32_t count, const int32_t *rects, const uint16_t *values);
/* The delay of tile (tx, ty), lane 0 or 1, or -1 where the table has none. */
int32_t fsim_belt_delay(int32_t tx, int32_t ty, int32_t lane);
/* The segment belt `index`'s lane (0 or 1) is in, named by its head's lane
 * ref, or -1 (not a belt, or on a closed loop). */
int32_t fsim_belt_segment(fsim_env *env, int32_t index, int32_t lane);
/* KF_* flags of an entity kind, and the seconds the character takes to mine one. */
int32_t fsim_kind_flags(int32_t kind);
double fsim_kind_mining_time(int32_t kind);
/* An item's stack size (0 for none). */
int32_t fsim_stack_size(int32_t item);
/* Whether the character's mining target is in reach from where it stands (1
 * or 0), or -1 when it mines nothing. */
int32_t fsim_mining_in_reach(const fsim_env *env);
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

/* v3 (FactorioRL parameterized-v3 / local-v3): 96 rows of 32 features, the
 * items grown by the Stage-2 four, 6 public-marker triples after the goal,
 * a 15x15 placement window, and a 13th self feature, the free share of the
 * main inventory. MultiDiscrete[25, 97, 226, 5, 19, 4]. */
#define RL3_MAX_ENTITIES 96
#define RL3_ENTITY_FEATURES 32
#define RL3_ITEMS 18
#define RL3_GOAL_FEATURES 30
#define RL3_MARKERS 6
#define RL3_TARGETS 96
#define RL3_PLACEMENTS 225
#define RL3_PLACEMENT_RADIUS 7
/* v3's catalog is parameterized-v1's 22 operations, `mine_tile` (22),
 * `take_fuel` (23) and `finish` (24). */
#define RL3_OPERATIONS 25
#define RL3_SELF_FEATURES 13
/* The argument dimensions (97 + 226 + 5 + 19 + 4): one row of the
 * per-operation masks (fsim_rl_opmask3), and the flat mask after the ops. */
#define RL3_ARG_WIDTH 351
#define RL3_MASK_SIZE 376

#define TASK_CONSTRUCT_SMELTING_LINE 1
#define TASK_BUILD_LINE 2
/* Commissioning rather than construction: the line is already down and both
 * machines are empty, and the agent has to reach each one and fuel it. */
#define TASK_PLATE_LINE 3

#define ACTION_SPACE_V1 0
#define ACTION_SPACE_V2 2
#define ACTION_SPACE_V3 3

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

/* The v3 observation (RL3_*). */
typedef struct {
    float grid[25350];          /* 6 x 65 x 65 */
    float entities[3072];       /* 96 x 32 */
    int8_t entity_mask[96];
    float self_[13];
    float inventory[18];
    float goal[30];
} fsim_obs3;

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
     *   character's tile, masked when occupied instead of skipped.
     * ACTION_SPACE_V3: v2's meanings over the v3 sizes, MultiDiscrete[25, 97,
     *   226, 5, 19, 4]: target k is row k-1 of the 96-row table (refused
     *   unless visible and in reach), placement p the tile
     *   ((p-1) / 15 - 7, (p-1) % 15 - 7), with `mine_tile` (22), `take_fuel`
     *   (23) and `finish` (24); masked per operation (fsim_rl_opmask3). */
    int32_t action_space;
    /* The observation's entity cap (env->sweep_cap): 0 keeps local-v2's 48;
     * local-v3 is 96. */
    int32_t entity_cap;
    /* v3: goal slots 12.. hold public_markers[k] for k < marker_count (at
     * most RL3_MARKERS), each slot its own marker. Slot k is published at
     * entity `marker_entity[k]` while that entity is alive (a marker that
     * names a scene entity: the mod publishes its live position and drops it
     * once the entity is gone), else at (marker_x, marker_y) when
     * `marker_present[k]` (a scene marker), else not at all. */
    int32_t marker_count;
    double marker_x[6];
    double marker_y[6];
    int32_t marker_present[6];
    int32_t marker_entity[6];   /* entity index, or -1 */
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
/* The v3 tensors and mask, whatever the env's action space: a v1 run can be
 * read in v3 too (the contract test does). */
void fsim_rl_encode3(fsim_rl *rl, fsim_obs3 *obs);
void fsim_rl_mask3(fsim_rl *rl, uint8_t *mask);
/* The v3 per-operation masks (FactorioRL `ParameterizedEnv.operation_masks`):
 * RL3_OPERATIONS rows of RL3_ARG_WIDTH, row o operation o's legal values of
 * each argument dimension (target, placement, direction, item, amount). */
void fsim_rl_opmask3(fsim_rl *rl, uint8_t *masks);
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
