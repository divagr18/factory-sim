"""Unsupervised environment design: levels as parameters, curated by regret.

Every scene family in this project was hand-written, which makes "held-out
generalisation" partly a claim about which family was chosen to hold out. UED
replaces the hand-written curriculum with one that follows the policy's own
frontier. The simulator's speed is what makes it affordable -- the reason
nobody has run it on this game before.

The frame is old and the books state it plainly (Yannakakis & Togelius,
*Artificial Intelligence and Games* 5.3): procedurally generated levels whose
difficulty tracks the policy's strength are what stopped RL overfitting to
single levels, citing Justesen et al. 2018. What PLR and ACCEL add is *which*
levels to keep.

Three pieces here:

`Level` and `build`
    A level is an explicit parameter vector rather than a seed, so it can be
    mutated. The parameter space is a strict superset of the five hand-written
    families -- `family_params` reproduces each of them exactly -- so the
    frozen holdout stays expressible and every number measured against it
    stays comparable.

`LevelBuffer`
    Prioritised Level Replay (Jiang et al. 2021). Levels are scored by mean
    positive value loss over an episode, `mean(clamp(GAE advantage, 0))`,
    which is the regret proxy both PLR and ACCEL use. Sampling mixes a rank
    prioritisation over score with staleness:

        P = (1 - rho) * P_score + rho * P_staleness
        P_score proportional to 1 / rank ** (1 / beta)

`mutate`
    ACCEL's editor (Parker-Holder et al. 2022): perturb a level the policy
    nearly handles, rather than drawing an unrelated random one, so complexity
    compounds from the frontier instead of restarting at it.

**Robust PLR is a property of the trainer, not of this module**, and it is the
part that matters for correctness: a level that came from the generator or the
editor is rolled out to be scored and is *not* trained on. Only replayed levels
take a gradient step. `train.py` enforces that by excluding evaluation slots
from the update.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace

from fsim import scenes

#: Tasks whose scenes are one ore patch, optionally screened by walls.
PATCH_TASKS = ("construct_smelting_line", "build_line")

#: A wall segment: (vertical, away, along, length). `away` is its offset from
#: the patch's *anchor* on the axis the wall does not run along.
#:
#: A screen laid across the ore makes a scene unsolvable rather than hard, and
#: a regret-seeking search will happily find those. A constant minimum of 4
#: was not enough: it assumed patches no wider than the hand-written families,
#: and `_resize` grows them to thirteen tiles, so `away = 4` landed inside.
#: Measured before the fix, 688 of 4,000 mutated levels had walls sitting on
#: ore. `clear_walls` pushes each wall out past the patch's actual extent.
WALL_AWAY = tuple(v for v in range(-6, 7) if abs(v) >= 4)
#: Tiles of clearance between the patch's edge and a wall, so a machine still
#: fits beside the ore it is mining.
WALL_MARGIN = 1
WALL_ALONG = range(-6, 4)
WALL_LENGTH = range(2, 6)

#: Patch bounds, as (x_lo, x_hi, y_lo, y_hi) offsets from its centre. Not
#: half-extents: build_line's narrow_patch is 2x10 and not centred on a tile,
#: so symmetric extents cannot express it.
PATCH_MIN, PATCH_MAX = -6, 6
OFFSET_LIMIT = 12
MAX_WALLS = 3


@dataclass(frozen=True)
class Level:
    """One scene, as the numbers that generate it."""

    task: str
    x_lo: int
    x_hi: int
    y_lo: int
    y_hi: int
    ox: int
    oy: int
    angle: float
    #: The start's offset from the patch centre, per axis. Two radii rather
    #: than one because construct_smelting_line draws a fresh uniform(9, 13)
    #: for x and another for y -- its starts lie on an axis-aligned ellipse,
    #: not a circle. build_line draws one distance and uses it for both.
    rx: float
    ry: float
    walls: tuple[tuple[int, int, int, int], ...] = ()
    #: The hand-written family this came from, when it came from one. Kept for
    #: reporting: a curriculum that rediscovers `obstructed_patch` is worth
    #: being able to say so.
    origin: str = "generated"

    def tiles(self) -> list[tuple[float, float]]:
        return [
            (float(self.ox + x), float(self.oy + y))
            for x in range(self.x_lo, self.x_hi + 1)
            for y in range(self.y_lo, self.y_hi + 1)
        ]

    def clear_walls(self) -> Level:
        """The same level with every wall pushed clear of the ore.

        `build` anchors walls at `int(cx), int(cy)`, so a wall on the axis it
        crosses is clear when `|away|` exceeds that axis's half-extent plus a
        margin. Pushing outward rather than dropping the wall keeps the edit
        that produced it meaningful -- the screen stays, it just stops being
        laid over the patch.
        """
        if not self.walls:
            return self
        half_x = max(abs(self.x_lo), abs(self.x_hi)) + WALL_MARGIN
        half_y = max(abs(self.y_lo), abs(self.y_hi)) + WALL_MARGIN
        walls = []
        for vertical, away, along, length in self.walls:
            # A vertical wall runs along y and is offset in x.
            floor = half_x if vertical else half_y
            if abs(away) < floor:
                away = floor if away >= 0 else -floor
            walls.append((vertical, away, along, length))
        return replace(self, walls=tuple(walls))

    def key(self) -> tuple:
        """Identity for deduplication: the parameters, not the origin."""
        return (
            self.task, self.x_lo, self.x_hi, self.y_lo, self.y_hi, self.ox, self.oy,
            round(self.angle, 4), round(self.rx, 3), round(self.ry, 3), self.walls,
        )  # fmt: skip


def build(level: Level) -> dict:
    """The blueprint payload, deterministically -- no rng, so a level is its
    parameters and nothing else."""
    tiles = level.tiles()
    cx = sum(x for x, _ in tiles) / len(tiles)
    cy = sum(y for _, y in tiles) / len(tiles)
    start = (
        round(cx + math.cos(level.angle) * level.rx, 1),
        round(cy + math.sin(level.angle) * level.ry, 1),
    )
    entities = []
    seen: set[tuple[float, float]] = set()
    # Walls sit on the tile grid, anchored at int(cx), int(cy). Every
    # hand-written family does this -- build_line's narrow_patch explicitly,
    # the others because their patch centres are already integers -- and a
    # wall half a tile off the grid is a different obstacle.
    ax, ay = float(int(cx)), float(int(cy))
    for vertical, away, along, length in level.walls:
        for k in range(length):
            x, y = (ax + away, ay + along + k) if vertical else (ax + along + k, ay + away)
            if (x, y) in seen:
                continue
            seen.add((x, y))
            entities.append(scenes._wall(float(x), float(y)))
    marker = (cx, cy) if level.task == "construct_smelting_line" else (round(cx, 1), round(cy, 1))
    return scenes._payload(entities, tiles, start, marker)


def random_level(task: str, rng: random.Random) -> Level:
    """A level from the whole space, which is what PLR's generator draws."""
    width = rng.randint(1, 4)
    height = rng.randint(1, 5)
    walls = tuple(
        (
            int(rng.random() < 0.5),
            rng.choice(WALL_AWAY),
            rng.choice(list(WALL_ALONG)),
            rng.choice(list(WALL_LENGTH)),
        )
        for _ in range(rng.randint(0, MAX_WALLS))
    )
    lo, hi = (9.0, 13.0) if task == "construct_smelting_line" else (8.0, 13.0)
    angle = rng.uniform(0, 2 * math.pi)
    rx = rng.uniform(lo, hi)
    ry = rng.uniform(lo, hi) if task == "construct_smelting_line" else rx
    return Level(
        task=task,
        x_lo=-width, x_hi=width, y_lo=-height, y_hi=height,
        ox=rng.randint(-OFFSET_LIMIT, OFFSET_LIMIT),
        oy=rng.randint(-OFFSET_LIMIT, OFFSET_LIMIT),
        angle=angle, rx=rx, ry=ry, walls=walls,
    ).clear_walls()  # fmt: skip


#: How `mutate` may change a level. Each is a small step, because ACCEL assumes
#: regret varies smoothly with the parameters -- an edit that jumps across the
#: space is a fresh random level wearing a parent's name.
def _resize(level: Level, rng: random.Random) -> Level:
    field_name = rng.choice(("x_lo", "x_hi", "y_lo", "y_hi"))
    step = rng.choice((-1, 1))
    value = getattr(level, field_name) + step
    value = max(PATCH_MIN, min(PATCH_MAX, value))
    candidate = replace(level, **{field_name: value})
    # An empty patch is not a hard level, it is a broken one.
    if candidate.x_hi < candidate.x_lo or candidate.y_hi < candidate.y_lo:
        return level
    return candidate


def _shift(level: Level, rng: random.Random) -> Level:
    axis = "ox" if rng.random() < 0.5 else "oy"
    value = getattr(level, axis) + rng.choice((-2, -1, 1, 2))
    return replace(level, **{axis: max(-OFFSET_LIMIT, min(OFFSET_LIMIT, value))})


def _move_start(level: Level, rng: random.Random) -> Level:
    def step(value: float) -> float:
        return min(16.0, max(6.0, value + rng.uniform(-1.5, 1.5)))

    return replace(
        level,
        angle=(level.angle + rng.uniform(-0.6, 0.6)) % (2 * math.pi),
        rx=step(level.rx),
        ry=step(level.ry),
    )


def _add_wall(level: Level, rng: random.Random) -> Level:
    if len(level.walls) >= MAX_WALLS:
        return level
    wall = (
        int(rng.random() < 0.5),
        rng.choice(WALL_AWAY),
        rng.choice(list(WALL_ALONG)),
        rng.choice(list(WALL_LENGTH)),
    )
    return replace(level, walls=level.walls + (wall,))


def _drop_wall(level: Level, rng: random.Random) -> Level:
    if not level.walls:
        return level
    at = rng.randrange(len(level.walls))
    return replace(level, walls=level.walls[:at] + level.walls[at + 1 :])


def _nudge_wall(level: Level, rng: random.Random) -> Level:
    if not level.walls:
        return _add_wall(level, rng)
    at = rng.randrange(len(level.walls))
    vertical, away, along, length = level.walls[at]
    which = rng.randrange(3)
    if which == 0:
        step = rng.choice((-1, 1))
        moved = away + step
        # Step over the forbidden band rather than into it.
        if abs(moved) < 4:
            moved = away + 2 * step
        away = max(-6, min(6, moved)) if abs(moved) >= 4 else away
    elif which == 1:
        along = max(WALL_ALONG.start, min(WALL_ALONG.stop - 1, along + rng.choice((-1, 1))))
    else:
        length = max(WALL_LENGTH.start, min(WALL_LENGTH.stop - 1, length + rng.choice((-1, 1))))
    walls = list(level.walls)
    walls[at] = (vertical, away, along, length)
    return replace(level, walls=tuple(walls))


EDITS = (_resize, _shift, _move_start, _add_wall, _drop_wall, _nudge_wall)


def mutate(level: Level, rng: random.Random, edits: int = 2) -> Level:
    """ACCEL's editor: a handful of small changes to a level already in hand."""
    out = level
    for _ in range(edits):
        out = rng.choice(EDITS)(out, rng)
    # After the edits, not between them: resizing the patch can swallow a wall
    # that was clear when it was placed.
    out = out.clear_walls()
    return replace(out, origin=f"edit:{level.origin.split(':')[-1]}")


def family_params(task: str, family: str, rng: random.Random) -> Level:
    """A hand-written family, expressed in the parameter space.

    This is what keeps the holdout comparable: `obstructed_patch` is a point in
    the same space UED searches, not a separate kind of thing, so a curriculum
    can rediscover it and an evaluation can still name it.
    """
    lo, hi = (9.0, 13.0) if task == "construct_smelting_line" else (8.0, 13.0)
    ox = oy = 0
    x_lo, x_hi, y_lo, y_hi = -3, 3, -3, 3
    walls: tuple[tuple[int, int, int, int], ...] = ()
    if family == "offset_patch":
        step = 9 if task == "construct_smelting_line" else 8
        ox, oy = rng.choice((-step, step)), rng.choice((-step, step))
    elif family == "obstructed_patch":
        x_lo, x_hi, y_lo, y_hi = -1, 1, -5, 5
        walls = ((1, 5, -2, 5),)
    elif family == "narrow_patch":
        if rng.random() < 0.5:
            x_lo, x_hi, y_lo, y_hi = -1, 0, -5, 4
        else:
            x_lo, x_hi, y_lo, y_hi = -5, 4, -1, 0
    elif family == "varied_patch":
        width, height = rng.randint(1, 4), rng.randint(1, 4)
        x_lo, x_hi, y_lo, y_hi = -width, width, -height, height
        ox, oy = rng.randint(-12, 12), rng.randint(-12, 12)
    elif family == "cluttered_patch":
        ox, oy = rng.randint(-9, 9), rng.randint(-9, 9)
    # The generators draw the start before the walls, and narrow_patch's
    # screen depends on which side the start landed, so the order matters as
    # much as the values.
    angle = rng.uniform(0, 2 * math.pi)
    rx = rng.uniform(lo, hi)
    ry = rng.uniform(lo, hi) if task == "construct_smelting_line" else rx
    if family == "cluttered_patch":
        walls = tuple(
            (
                int(rng.random() < 0.5),
                rng.choice((-6, -5, -4, 4, 5, 6)),
                rng.randint(-6, 3),
                rng.randint(2, 4),
            )
            for _ in range(rng.randint(1, 3))
        )
    elif family == "narrow_patch":
        side = 1 if math.cos(angle) >= 0 else -1
        walls = ((1, side * 6, -1, 3),)
    return Level(
        task=task, x_lo=x_lo, x_hi=x_hi, y_lo=y_lo, y_hi=y_hi, ox=ox, oy=oy,
        angle=angle, rx=rx, ry=ry, walls=walls, origin=family,
    )  # fmt: skip


@dataclass
class _Entry:
    level: Level
    #: The level's standing among the levels scored in the same rollout, in
    #: [0, 1]. Not the raw positive value loss: that is measured against a
    #: moving critic and drifts, so two raw scores from different updates are
    #: not comparable -- and the buffer compares them directly when it decides
    #: what to evict. Measured on a 20M-step run, raw score tracked *recency*
    #: (r = -0.170 against staleness, recently scored levels averaging 0.0231
    #: against 0.0168) more strongly than it tracked anything about the level:
    #: walls r = -0.064, patch size r = +0.010, start distance r = +0.000.
    score: float = 0.0
    #: What that standing was computed from, kept only for reporting.
    raw: float = 0.0
    staleness: float = 0.0
    seen: int = 0


class LevelBuffer:
    """Prioritised Level Replay's buffer (Jiang et al. 2021).

    Levels are ranked by score -- mean positive value loss, the regret proxy --
    and sampled by rank mixed with staleness, so a level that scored well long
    ago is revisited before the ranking calcifies around whatever was measured
    first.
    """

    def __init__(
        self,
        capacity: int = 4000,
        beta: float = 0.3,
        rho: float = 0.3,
        seed: int = 0,
    ) -> None:
        #: Sampling temperature over ranks: P proportional to 1/rank**(1/beta).
        self.beta = beta
        #: How much of the distribution is staleness rather than score.
        self.rho = rho
        self.capacity = capacity
        self.entries: list[_Entry] = []
        self._index: dict[tuple, int] = {}
        self.rng = random.Random(seed)
        self.inserted = 0
        self.rejected = 0

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def levels(self) -> list[Level]:
        return [entry.level for entry in self.entries]

    def consider(self, level: Level, score: float, raw: float | None = None) -> bool:
        """Offer a scored level. Kept if the buffer has room or it beats the
        worst level in it -- which is how ACCEL's edits compound: an edit only
        survives if it is harder than something already held."""
        raw = score if raw is None else raw
        key = level.key()
        if key in self._index:
            self.update(self._index[key], score, raw)
            return False
        if len(self.entries) < self.capacity:
            self._index[key] = len(self.entries)
            self.entries.append(_Entry(level, score, raw))
            self.inserted += 1
            return True
        worst = min(range(len(self.entries)), key=lambda i: self.entries[i].score)
        if score <= self.entries[worst].score:
            self.rejected += 1
            return False
        del self._index[self.entries[worst].level.key()]
        self.entries[worst] = _Entry(level, score, raw)
        self._index[key] = worst
        self.inserted += 1
        return True

    def update(self, index: int, score: float, raw: float | None = None) -> None:
        """A replayed level's score, after it was trained on."""
        entry = self.entries[index]
        entry.score = score
        entry.raw = score if raw is None else raw
        entry.seen += 1

    def weights(self) -> list[float]:
        n = len(self.entries)
        if n == 0:
            return []
        order = sorted(range(n), key=lambda i: self.entries[i].score, reverse=True)
        score_w = [0.0] * n
        for rank, i in enumerate(order, start=1):
            score_w[i] = 1.0 / rank ** (1.0 / self.beta)
        total = sum(score_w) or 1.0
        score_w = [w / total for w in score_w]
        if self.rho <= 0.0:
            return score_w
        stale = [entry.staleness for entry in self.entries]
        stale_total = sum(stale)
        if stale_total <= 0.0:
            stale_w = [1.0 / n] * n
        else:
            stale_w = [s / stale_total for s in stale]
        return [(1.0 - self.rho) * s + self.rho * t for s, t in zip(score_w, stale_w, strict=True)]

    def sample(self) -> tuple[int, Level]:
        """A level to replay, and its index so its score can be updated."""
        weights = self.weights()
        index = self.rng.choices(range(len(self.entries)), weights=weights, k=1)[0]
        for entry in self.entries:
            entry.staleness += 1.0
        self.entries[index].staleness = 0.0
        return index, self.entries[index].level

    def stats(self) -> dict:
        if not self.entries:
            return {"size": 0, "score_mean": 0.0, "score_max": 0.0, "walls_mean": 0.0}
        scores = [entry.score for entry in self.entries]
        raws = [entry.raw for entry in self.entries]
        return {
            "size": len(self.entries),
            "score_mean": round(sum(scores) / len(scores), 5),
            "score_max": round(max(scores), 5),
            # What the standings were computed from, so the drift that made
            # raw scores incomparable stays visible.
            "raw_mean": round(sum(raws) / len(raws), 5),
            # Complexity, as ACCEL reports it: does the curriculum get harder?
            "walls_mean": round(
                sum(len(e.level.walls) for e in self.entries) / len(self.entries), 3
            ),
            "tiles_mean": round(
                sum(len(e.level.tiles()) for e in self.entries) / len(self.entries), 2
            ),
            "inserted": self.inserted,
            "rejected": self.rejected,
        }


def positive_value_loss(advantages) -> float:
    """PLR's score: the mean of the positive part of the GAE advantage.

    `returns - value_preds` is the GAE advantage, so this is exactly
    `mean(max(sum_k (gamma*lambda)^(k-t) delta_k, 0))` as ACCEL writes it. It
    reads as "how much better the episode went than the critic expected",
    which is high where the policy is still learning and low both where it has
    mastered the level and where it never gets anywhere at all.
    """
    clipped = advantages.clamp(min=0.0)
    return float(clipped.mean())


class Curriculum:
    """Which scene each environment slot runs, and what happens to its score.

    Slots are split in two. The first `train_slots` **replay** levels drawn
    from the buffer, and the trainer takes gradient steps on them. The rest
    run levels that came from the generator or the editor; they are scored and
    then either kept or discarded, and **nothing they produce is trained on**.

    That split is Robust PLR (Jiang et al. 2021) and it is the part that
    matters. Training on freshly generated levels is what biases vanilla PLR's
    curriculum -- the policy is updated on whatever the generator happened to
    produce, rather than on what the curation decided was worth learning.

    A UED run uses whole episodes, one per slot per rollout, so a level's
    score is unambiguous: the mean positive advantage over that episode, and
    no attribution across episode boundaries to get wrong.
    """

    #: How the non-training slots get their levels.
    MODES = ("plr", "accel")

    def __init__(
        self,
        task: str,
        n: int,
        train_slots: int,
        mode: str = "accel",
        buffer: LevelBuffer | None = None,
        seed: int = 0,
        edits: int = 2,
        warm_start: int = 0,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode {mode!r} is not one of {self.MODES}")
        self.task = task
        self.n = n
        self.train_slots = train_slots
        self.mode = mode
        self.edits = edits
        self.buffer = buffer if buffer is not None else LevelBuffer(seed=seed)
        self.rng = random.Random(seed + 1)
        #: The level each slot is running, and its buffer index when replayed.
        self.slot_level: list[Level | None] = [None] * n
        self.slot_index: list[int | None] = [None] * n
        #: Whether each slot may be trained on this rollout. A training slot
        #: whose buffer was empty fell through to the generator, and training
        #: on a generated level is exactly what Robust PLR forbids -- so the
        #: mask is read per rollout rather than fixed at the slot split.
        self.slot_trains: list[bool] = [False] * n
        self.replayed = 0
        self.generated = 0
        self.edited = 0
        if warm_start:
            self.seed_from_families(warm_start)

    def seed_from_families(self, count: int) -> None:
        """Start the buffer from the hand-written training families.

        ACCEL starts from simple levels and compounds; here the simple levels
        already exist and are the ones every earlier result was measured on,
        so the curriculum begins where the project is rather than at noise.
        """
        names = scenes.families(self.task, "train")
        for k in range(count):
            family = names[k % len(names)]
            level = family_params(self.task, family, random.Random(1_000_000 + k))
            self.buffer.consider(level, score=0.0)

    def level_for(self, i: int, seed: int) -> tuple[str, dict]:
        """`VecEnv`'s level source: the scene slot `i` runs next."""
        replay = i < self.train_slots and len(self.buffer) > 0
        if replay:
            index, level = self.buffer.sample()
            self.replayed += 1
        else:
            index = None
            level = self._propose(seed)
        self.slot_level[i] = level
        self.slot_index[i] = index
        self.slot_trains[i] = index is not None
        return level.origin, build(level)

    def _propose(self, seed: int) -> Level:
        if self.mode == "accel" and len(self.buffer) > 0:
            _index, parent = self.buffer.sample()
            self.edited += 1
            return mutate(parent, self.rng, self.edits)
        self.generated += 1
        return random_level(self.task, self.rng)

    @staticmethod
    def standings(scores) -> list[float]:
        """Each score's standing among the scores measured beside it, in [0, 1].

        Raw positive value loss is measured against a critic that is still
        learning, so its scale drifts: over one 20M-step run the mean rose
        from 0.0031 to 0.0172. The buffer compares scores directly when it
        decides what to keep, so on raw values it was ranking levels by *when*
        they happened to be measured -- staleness correlated with score at
        r = -0.170, more strongly than walls at -0.064 or patch size at
        +0.010. A standing is comparable across updates because it is relative
        to the levels that ran in the same one.
        """
        order = sorted(range(len(scores)), key=lambda i: scores[i])
        out = [0.0] * len(scores)
        last = len(scores) - 1
        for rank, i in enumerate(order):
            out[i] = rank / last if last else 0.5
        return out

    def report(self, scores, at: dict | None = None) -> None:
        """One score per slot, after the rollout that produced them.

        `at` is a `snapshot` taken before the rollout, for the case where a
        rollout is shorter than an episode and slots changed level inside it.

        A replayed level's standing is updated in place. A proposed level is
        offered to the buffer, and kept only if it beats something already
        held -- which is how ACCEL's edits compound instead of drifting, and
        why the comparison has to be between comparable numbers.
        """
        ranked = self.standings(list(scores))
        levels = at["level"] if at else self.slot_level
        indices = at["index"] if at else self.slot_index
        for i, (standing, raw) in enumerate(zip(ranked, scores, strict=True)):
            level = levels[i]
            if level is None:
                continue
            index = indices[i]
            if index is not None:
                self.buffer.update(index, float(standing), float(raw))
            else:
                self.buffer.consider(level, float(standing), float(raw))

    def training_mask(self) -> list[bool]:
        """Which slots this rollout may train on, after levels were assigned."""
        return list(self.slot_trains)

    def snapshot(self) -> dict:
        """What each slot was running at this moment.

        Needed when the rollout is shorter than an episode. Autoreset hands a
        finished slot a new level part-way through, so by the time advantages
        are known `slot_level` no longer says which level earned them. The
        snapshot is taken before the rollout and scored afterwards; a slot that
        changed level inside the rollout has its segment credited to the level
        it started with, which is the approximation PLR's own implementation
        makes for partial segments.
        """
        return {
            "level": list(self.slot_level),
            "index": list(self.slot_index),
            "trains": list(self.slot_trains),
        }

    def stats(self) -> dict:
        out = self.buffer.stats()
        out.update(
            {
                "replayed": self.replayed,
                "generated": self.generated,
                "edited": self.edited,
                # What fraction of the buffer still comes from a hand-written
                # family: a curriculum that has left them behind says so here.
                "from_families": round(
                    sum(1 for e in self.buffer.entries if e.level.origin != "generated"
                        and not e.level.origin.startswith("edit"))
                    / max(1, len(self.buffer.entries)),
                    3,
                ),
            }
        )  # fmt: skip
        return out


def save_buffer(buffer: LevelBuffer, path) -> None:
    """Write the curriculum out. Without this a run's curriculum dies with the
    process: it could not be inspected afterwards, compared between runs, or
    used to warm-start the next one, which is most of what ACCEL's compounding
    is for."""
    import json
    from pathlib import Path

    rows = [
        {
            "score": entry.score,
            "raw": entry.raw,
            "seen": entry.seen,
            "staleness": entry.staleness,
            "level": {
                "task": entry.level.task,
                "x_lo": entry.level.x_lo, "x_hi": entry.level.x_hi,
                "y_lo": entry.level.y_lo, "y_hi": entry.level.y_hi,
                "ox": entry.level.ox, "oy": entry.level.oy,
                "angle": entry.level.angle, "rx": entry.level.rx, "ry": entry.level.ry,
                "walls": [list(w) for w in entry.level.walls],
                "origin": entry.level.origin,
            },
        }  # fmt: skip
        for entry in buffer.entries
    ]
    Path(path).write_text(
        json.dumps(
            {"beta": buffer.beta, "rho": buffer.rho, "capacity": buffer.capacity, "levels": rows},
            indent=1,
        ),
        "utf-8",
    )


def load_buffer(path, seed: int = 0) -> LevelBuffer:
    import json
    from pathlib import Path

    blob = json.loads(Path(path).read_text("utf-8"))
    buffer = LevelBuffer(
        capacity=blob["capacity"], beta=blob["beta"], rho=blob["rho"], seed=seed
    )
    for row in blob["levels"]:
        spec = dict(row["level"])
        spec["walls"] = tuple(tuple(w) for w in spec["walls"])
        buffer.consider(Level(**spec), row["score"], row.get("raw", row["score"]))
        # consider() only carries the score. Restoring seen and staleness is
        # the point of loading at all: without them a resumed run reports
        # every level as never replayed and restarts the staleness clock.
        buffer.entries[-1].seen = int(row.get("seen", 0))
        buffer.entries[-1].staleness = float(row.get("staleness", 0.0))
    return buffer


def render(level: Level, width: int = 33, height: int = 25) -> str:
    """A level as text, centred on its patch: `#` wall, `o` ore, `@` start.

    Aggregate statistics say a curriculum got harder; they do not say whether
    it got harder in a way that means anything. This is for looking.
    """
    scene = build(level)
    ore = {(math.floor(x), math.floor(y)) for x, y in (r["position"] for r in scene["resources"])}
    walls = {(math.floor(x), math.floor(y)) for x, y in (e["position"] for e in scene["entities"])}
    sx, sy = scene["character"]["position"]
    start = (math.floor(sx), math.floor(sy))
    cx = (level.ox * 2 + level.x_lo + level.x_hi) // 2
    cy = (level.oy * 2 + level.y_lo + level.y_hi) // 2
    lines = []
    for row in range(cy - height // 2, cy + height // 2 + 1):
        out = []
        for col in range(cx - width // 2, cx + width // 2 + 1):
            cell = (col, row)
            out.append(
                "@" if cell == start else "#" if cell in walls else "o" if cell in ore else "."
            )
        lines.append("".join(out))
    return "\n".join(lines)
