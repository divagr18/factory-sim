"""Every candidate an evolution run makes, where it came from, and which ones breed.

`Store` is the run's genealogy: one SQLite row per candidate, written once and
never changed, so a run can be stopped and picked up again and every program can
be traced back through its first parents to a seed. It is the only durable
state. `Islands` and `MapElites` are selection views over it and can be rebuilt
from it.

Scores are rates per scene family, on a training split and a validation split
(`scores["train"]`, `scores["val"]`), with their means cached as `train_mean`
and `val_mean`. Selection ranks by `val_mean`. Ties go to the shorter program,
because a shorter program that does as well is easier for the next prompt to
read and less likely to be overfitted to the training scenes, and then to the
older one, so that a resubmitted clone never displaces the original.

Islands keep populations apart so that one early lucky layout cannot take over
the whole run (the island model of FunSearch, Romera-Paredes et al. 2024).
Ranking by score alone still lets one layout take over each island. So each
island reserves one slot for diversity, described in `Islands`.
"""

from __future__ import annotations

import json
import math
import os
import random
import sqlite3
import subprocess
import time
import uuid
from bisect import bisect_right
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

FACTORY_SIM = Path(__file__).resolve().parent.parent

_COLUMNS = (
    "id",
    "code",
    "code_hash",
    "parents",
    "operator",
    "island",
    "prompt_hash",
    "model",
    "scores",
    "descriptors",
    "length",
    "created",
    "train_mean",
    "val_mean",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    id          TEXT PRIMARY KEY,
    code        TEXT NOT NULL,
    code_hash   TEXT NOT NULL,
    parents     TEXT NOT NULL,      -- JSON list of ids, first parent first
    operator    TEXT NOT NULL,
    island      INTEGER NOT NULL,
    prompt_hash TEXT,
    model       TEXT,
    scores      TEXT NOT NULL,      -- JSON {"train": {...}, "val": {...}, "train_mean", "val_mean"}
    descriptors TEXT,               -- JSON or NULL
    length      INTEGER NOT NULL,
    created     REAL NOT NULL,
    train_mean  REAL,               -- copied out of scores so SQL can rank by it
    val_mean    REAL
);
CREATE INDEX IF NOT EXISTS candidates_code_hash ON candidates (code_hash);
CREATE INDEX IF NOT EXISTS candidates_val_mean ON candidates (val_mean);
"""


def source_length(code: str) -> int:
    """Non-blank source lines: what the length tie-break counts."""
    return sum(1 for line in code.splitlines() if line.strip())


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def _mean(scores: dict, key: str) -> float | None:
    v = scores.get(key) if scores else None
    return None if v is None else float(v)


@dataclass
class Candidate:
    id: str
    code: str
    code_hash: str
    parents: list[str]
    operator: str
    island: int
    prompt_hash: str | None
    model: str | None
    scores: dict
    descriptors: dict | None
    length: int
    created: float

    @classmethod
    def new(
        cls,
        code: str,
        *,
        code_hash: str | None = None,
        parents: list[str] | None = None,
        operator: str = "seed",
        island: int = 0,
        prompt_hash: str | None = None,
        model: str | None = None,
        scores: dict | None = None,
        descriptors: dict | None = None,
    ) -> Candidate:
        """Fill in id, length and created. Pass `code_hash` from
        `sandbox.normalized_hash`; the fallback hashes the raw text, which
        misses clones that differ only in formatting."""
        return cls(
            id=new_id(),
            code=code,
            code_hash=code_hash or sha256(code.encode()).hexdigest(),
            parents=list(parents or []),
            operator=operator,
            island=island,
            prompt_hash=prompt_hash,
            model=model,
            scores=dict(scores or {}),
            descriptors=descriptors,
            length=source_length(code),
            created=time.time(),
        )

    def score(self, key: str = "val_mean") -> float:
        """The value to rank by; a missing score ranks below every real one."""
        v = _mean(self.scores, key)
        return -math.inf if v is None or math.isnan(v) else v

    def rank_key(self, key: str = "val_mean") -> tuple:
        """Ascending sort key: best first."""
        return (-self.score(key), self.length, self.created, self.id)

    @property
    def layout_signature(self) -> str | None:
        return (self.descriptors or {}).get("layout_signature")

    def to_row(self) -> tuple:
        return (
            self.id,
            self.code,
            self.code_hash,
            json.dumps(self.parents),
            self.operator,
            int(self.island),
            self.prompt_hash,
            self.model,
            json.dumps(self.scores, sort_keys=True),
            None if self.descriptors is None else json.dumps(self.descriptors, sort_keys=True),
            int(self.length),
            float(self.created),
            _mean(self.scores, "train_mean"),
            _mean(self.scores, "val_mean"),
        )

    @classmethod
    def from_row(cls, row) -> Candidate:
        d = dict(zip(_COLUMNS, row, strict=False))
        return cls(
            id=d["id"],
            code=d["code"],
            code_hash=d["code_hash"],
            parents=json.loads(d["parents"]),
            operator=d["operator"],
            island=d["island"],
            prompt_hash=d["prompt_hash"],
            model=d["model"],
            scores=json.loads(d["scores"]),
            descriptors=None if d["descriptors"] is None else json.loads(d["descriptors"]),
            length=d["length"],
            created=d["created"],
        )


class Store:
    """The genealogy database. One writer process; any number of readers.

    WAL mode lets readers see every committed row while the writer keeps
    going, without either blocking the other. Each `add` commits on its own, so
    a crash loses at most the candidate being written. Open with
    `readonly=True` from a process that only polls (the orchestrator's
    dashboard): it cannot take the write lock by accident.
    """

    def __init__(self, path, *, readonly: bool = False, timeout_s: float = 30.0):
        self.path = Path(path)
        self.readonly = readonly
        if readonly:
            uri = f"{self.path.resolve().as_uri()}?mode=ro"
            self.conn = sqlite3.connect(uri, uri=True, timeout=timeout_s)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.path), timeout=timeout_s)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.executescript(SCHEMA)
            self.conn.commit()
        self.conn.execute(f"PRAGMA busy_timeout={int(timeout_s * 1000)}")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _select(self, where: str = "", args: tuple = (), tail: str = "") -> list[Candidate]:
        sql = f"SELECT {', '.join(_COLUMNS)} FROM candidates {where} {tail}"
        return [Candidate.from_row(r) for r in self.conn.execute(sql, args)]

    def add(self, c: Candidate) -> None:
        marks = ", ".join("?" * len(_COLUMNS))
        with self.conn:
            self.conn.execute(
                f"INSERT INTO candidates ({', '.join(_COLUMNS)}) VALUES ({marks})", c.to_row()
            )

    def get(self, id: str) -> Candidate | None:
        rows = self._select("WHERE id = ?", (id,))
        return rows[0] if rows else None

    def has_hash(self, h: str) -> bool:
        q = "SELECT 1 FROM candidates WHERE code_hash = ? LIMIT 1"
        return self.conn.execute(q, (h,)).fetchone() is not None

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]

    def all(self, island: int | None = None) -> list[Candidate]:
        """In creation order, which is the order a rebuild replays them."""
        if island is None:
            return self._select(tail="ORDER BY created, rowid")
        return self._select("WHERE island = ?", (island,), "ORDER BY created, rowid")

    def best(self, n: int = 10, key: str = "val_mean") -> list[Candidate]:
        """Top `n` by `key`, then shorter, then older. Unscored rows come last."""
        if key in ("val_mean", "train_mean"):
            tail = f"ORDER BY ({key} IS NULL), {key} DESC, length, created, id LIMIT ?"
            return self._select(tail=tail, args=(n,))
        return sorted(self.all(), key=lambda c: c.rank_key(key))[:n]

    def lineage(self, id: str) -> list[Candidate]:
        """First parents back to the root, root first. Stops at a parent the
        store does not hold, so a lineage is never longer than what is known."""
        chain: list[Candidate] = []
        seen: set[str] = set()
        cur = self.get(id)
        while cur is not None and cur.id not in seen:
            chain.append(cur)
            seen.add(cur.id)
            cur = self.get(cur.parents[0]) if cur.parents else None
        chain.reverse()
        return chain


class Islands:
    """`n` separate populations of at most `size` each, ranked by `val_mean`.

    Diversity slot. When `size >= 2`, one of an island's slots is held back for
    a program whose `descriptors["layout_signature"]` differs from that of the
    island's best. The top `size - 1` members are chosen by rank alone. If none
    of them has a signature different from the best's, the last slot goes to
    the best-ranked remaining candidate that does. If one of them already does,
    or no candidate does, the last slot goes by rank like the others. So an
    island that has ever held a second layout keeps one for as long as the
    candidate survives, even when a single layout's variants outscore it. A
    candidate with no signature never counts as different, because nothing
    shows that it is. Candidates that have been evicted are gone, so the slot
    only ever chooses among the current members and the newcomer.

    `select` draws from `random.Random(seed)`, so a run's parent choices are
    reproducible given the same sequence of calls. `migrate` copies with
    `dataclasses.replace(c, island=j)`: the copy keeps its id, and the Store
    still records the island the candidate was born on.
    """

    TOURNAMENT = 3

    def __init__(self, n: int, size: int, seed: int):
        if n < 1 or size < 1:
            raise ValueError("need n >= 1 and size >= 1")
        self.n, self.size, self.seed = n, size, seed
        self.members: list[list[Candidate]] = [[] for _ in range(n)]
        self.rng = random.Random(seed)

    def _rank(self, pool: list[Candidate]) -> list[Candidate]:
        pool = sorted(pool, key=lambda c: c.rank_key())
        if len(pool) <= self.size:
            return pool
        top = pool[: self.size - 1] if self.size >= 2 else pool[:1]
        if self.size == 1:
            return top
        rest = pool[self.size - 1 :]
        best_sig = top[0].layout_signature

        def differs(c: Candidate) -> bool:
            return c.layout_signature is not None and c.layout_signature != best_sig

        if not any(differs(c) for c in top[1:]):
            other = next((c for c in rest if differs(c)), None)
            if other is not None:
                return top + [other]
        return top + rest[:1]

    def admit(self, c: Candidate) -> bool:
        """Offer `c` to island `c.island`. True if it is a member afterwards."""
        i = c.island % self.n
        pool = self.members[i]
        if any(m.id == c.id for m in pool):
            return True
        self.members[i] = self._rank(pool + [c])
        return any(m.id == c.id for m in self.members[i])

    def best(self, island: int) -> Candidate | None:
        pool = self.members[island % self.n]
        return min(pool, key=lambda c: c.rank_key()) if pool else None

    def select(self, island: int, k: int = 2) -> list[Candidate]:
        """`k` tournament winners (tournaments of 3). Distinct while the island
        has more than `k` members, so crossover gets two different parents."""
        pool = sorted(self.members[island % self.n], key=lambda c: c.rank_key())
        if not pool:
            return []
        chosen: list[Candidate] = []
        for _ in range(k):
            taken = {c.id for c in chosen}
            left = [c for c in pool if c.id not in taken] or pool
            entrants = self.rng.sample(left, min(self.TOURNAMENT, len(left)))
            chosen.append(min(entrants, key=lambda c: c.rank_key()))
        return chosen

    def migrate(self) -> None:
        """Ring migration: each island's best, as it was before any copy
        arrived, is offered to the next island."""
        bests = [self.best(i) for i in range(self.n)]
        for i, b in enumerate(bests):
            if b is not None and self.n > 1:
                self.admit(replace(b, island=(i + 1) % self.n))

    def state(self) -> dict:
        """JSON-able: the member ids of each island and the draw state."""
        version, internal, gauss = self.rng.getstate()
        return {
            "n": self.n,
            "size": self.size,
            "seed": self.seed,
            "members": [[c.id for c in pool] for pool in self.members],
            "rng": [version, list(internal), gauss],
        }

    @classmethod
    def from_state(cls, state: dict, store: Store) -> Islands:
        """Exact resume, including migrated copies and the draw position."""
        isl = cls(state["n"], state["size"], state["seed"])
        for i, ids in enumerate(state["members"]):
            for cid in ids:
                c = store.get(cid)
                if c is not None:
                    isl.members[i].append(replace(c, island=i))
            isl.members[i].sort(key=lambda c: c.rank_key())
        version, internal, gauss = state["rng"]
        isl.rng.setstate((version, tuple(internal), gauss))
        return isl

    @classmethod
    def rebuild(cls, store: Store, n: int, size: int, seed: int) -> Islands:
        """Replay every stored candidate in creation order. Unlike
        `from_state`, this does not replay migrations or restore the draw
        position; it gives the populations a run would have without them."""
        isl = cls(n, size, seed)
        for c in store.all():
            isl.admit(c)
        return isl


class MapElites:
    """The best program per cell of a descriptor grid (Mouret & Clune 2015).

    `bins` maps a descriptor name to sorted edges; `e` edges make `e + 1` bins,
    and a value equal to an edge falls in the bin above it. A candidate that
    lacks one of the descriptors has no cell and is not admitted.
    """

    def __init__(self, bins: dict[str, list[float]]):
        self.bins = {k: list(v) for k, v in bins.items()}
        for name, edges in self.bins.items():
            if edges != sorted(edges):
                raise ValueError(f"bin edges for {name!r} are not sorted")
        self.grid: dict[tuple[int, ...], Candidate] = {}

    def cell(self, c: Candidate) -> tuple[int, ...] | None:
        d = c.descriptors or {}
        if any(d.get(name) is None for name in self.bins):
            return None
        return tuple(bisect_right(edges, float(d[name])) for name, edges in self.bins.items())

    def admit(self, c: Candidate) -> bool:
        cell = self.cell(c)
        if cell is None:
            return False
        cur = self.grid.get(cell)
        if cur is None or c.rank_key() < cur.rank_key():
            self.grid[cell] = c
            return True
        return False

    def cells(self) -> dict[tuple[int, ...], Candidate]:
        return dict(self.grid)

    def coverage(self) -> float:
        total = math.prod(len(e) + 1 for e in self.bins.values())
        return len(self.grid) / total


def factory_sim_commit(repo=FACTORY_SIM) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def write_manifest(run_dir, **fields) -> dict:
    """Write or merge `run_dir/manifest.json` and return what was written.

    A resumed run merges into the manifest rather than replacing it: new
    values win, `created` keeps the first write's time, and if the simulator's
    commit has changed since, the earlier commits are kept in
    `previous_factory_sim_commits` so a run built across two commits says so.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    merged = {**old, **fields}
    merged["created"] = old.get("created") or datetime.now(UTC).isoformat()
    commit = factory_sim_commit()
    if "factory_sim_commit" in old and old["factory_sim_commit"] != commit:
        prev = list(old.get("previous_factory_sim_commits", []))
        prev.append(old["factory_sim_commit"])
        merged["previous_factory_sim_commits"] = prev
    merged["factory_sim_commit"] = commit
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(merged, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return merged


__all__ = [
    "Candidate",
    "Islands",
    "MapElites",
    "SCHEMA",
    "Store",
    "factory_sim_commit",
    "source_length",
    "write_manifest",
]
