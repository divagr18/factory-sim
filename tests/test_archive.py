"""The genealogy store, the islands and the MAP-Elites grid."""

import json
import sqlite3
import threading

import pytest

from evolve import archive
from evolve.archive import Candidate, Islands, MapElites, Store, source_length, write_manifest

_clock = iter(range(1, 10**6))


def cand(val=0.5, length=5, *, island=0, sig=None, parents=(), code=None, desc=None, **kw):
    """A candidate with a controlled length and a strictly increasing `created`."""
    code = code or "def build(world):\n" + "    world.wait()\n" * (length - 1)
    c = Candidate.new(
        code,
        parents=list(parents),
        island=island,
        scores={
            "train": {"open_patch": val},
            "val": {"open_patch": val},
            "val_mean": val,
            "train_mean": val,
        },
        descriptors=desc if desc is not None else ({"layout_signature": sig} if sig else None),
        **kw,
    )
    c.created = float(next(_clock))
    return c


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "runs" / "evolve-t" / "genealogy.sqlite")
    yield s
    s.close()


# ---------------------------------------------------------------- Store


def test_length_counts_non_blank_lines():
    assert source_length("def build(world):\n\n    \n    world.wait()\n") == 2
    assert cand(length=7).length == 7


def test_round_trip_keeps_json_fields(store):
    c = cand(
        0.75,
        parents=["aaa", "bbb"],
        desc={"layout_signature": "S", "walk": 12.5},
        prompt_hash="ph",
        model="m",
    )
    c.scores["val"] = {"open_patch": 1.0, "offset_patch": 0.5}
    store.add(c)
    got = store.get(c.id)
    assert got == c
    assert got.parents == ["aaa", "bbb"]
    assert got.descriptors == {"layout_signature": "S", "walk": 12.5}
    assert got.scores["val"]["offset_patch"] == 0.5
    assert store.get("missing") is None
    assert store.count() == 1
    assert len(c.id) == 12


def test_round_trip_with_null_descriptors(store):
    c = cand(desc={})
    c.descriptors = None
    store.add(c)
    assert store.get(c.id).descriptors is None


def test_has_hash(store):
    c = cand()
    assert not store.has_hash(c.code_hash)
    store.add(c)
    assert store.has_hash(c.code_hash)
    assert not store.has_hash("0" * 64)


def test_duplicate_id_is_refused(store):
    c = cand()
    store.add(c)
    with pytest.raises(sqlite3.IntegrityError):
        store.add(c)
    assert store.count() == 1


def test_best_orders_by_score_then_length_then_age(store):
    a = cand(0.9, length=10)
    b = cand(0.9, length=4)
    c = cand(0.9, length=4)  # same score and length as b, but younger
    d = cand(1.0, length=30)
    e = cand(0.1)
    unscored = cand()
    unscored.scores = {}
    for x in (e, unscored, a, c, b, d):
        store.add(x)
    ids = [x.id for x in store.best(10)]
    assert ids == [d.id, b.id, c.id, a.id, e.id, unscored.id]
    assert [x.id for x in store.best(2)] == [d.id, b.id]
    assert [x.id for x in store.best(10, key="train_mean")][:2] == [d.id, b.id]


def test_best_by_a_key_outside_the_columns(store):
    a, b = cand(0.1), cand(0.2)
    a.scores["obstructed"] = 0.9
    b.scores["obstructed"] = 0.3
    store.add(a)
    store.add(b)
    assert [x.id for x in store.best(2, key="obstructed")] == [a.id, b.id]


def test_lineage_walks_first_parents_root_first(store):
    root = cand()
    other = cand()
    g1 = cand(parents=[root.id, other.id])
    g2 = cand(parents=[g1.id])
    g3 = cand(parents=[g2.id, other.id])
    for x in (root, other, g1, g2, g3):
        store.add(x)
    assert [x.id for x in store.lineage(g3.id)] == [root.id, g1.id, g2.id, g3.id]
    assert [x.id for x in store.lineage(root.id)] == [root.id]
    assert store.lineage("nope") == []


def test_lineage_stops_at_an_unknown_parent(store):
    orphan = cand(parents=["gone"])
    store.add(orphan)
    assert [x.id for x in store.lineage(orphan.id)] == [orphan.id]


def test_reopen_resumes(tmp_path):
    path = tmp_path / "g.sqlite"
    with Store(path) as s:
        first = [cand(i / 10) for i in range(5)]
        for c in first:
            s.add(c)
    with Store(path) as s:
        assert s.count() == 5
        s.add(cand(0.95))
        assert s.count() == 6
        assert s.best(1)[0].scores["val_mean"] == 0.95
        assert [c.id for c in s.all()][:5] == [c.id for c in first]


def test_all_filters_by_island(store):
    for i in range(6):
        store.add(cand(island=i % 3))
    assert len(store.all()) == 6
    assert {c.island for c in store.all(island=2)} == {2}
    assert len(store.all(island=2)) == 2


def test_reader_sees_commits_while_the_writer_writes(tmp_path):
    path = tmp_path / "g.sqlite"
    writer = Store(path)
    reader = Store(path, readonly=True)
    seen: list[int] = []
    stop = threading.Event()

    def poll():
        r = Store(path, readonly=True)
        while not stop.is_set():
            seen.append(r.count())
        seen.append(r.count())
        r.close()

    t = threading.Thread(target=poll)
    t.start()
    for i in range(40):
        writer.add(cand(i / 40))
        assert reader.count() == i + 1  # each add is committed and visible at once
    stop.set()
    t.join()
    assert seen == sorted(seen)  # never goes backwards
    assert seen[-1] == 40
    with pytest.raises(sqlite3.OperationalError):
        reader.add(cand())
    reader.close()
    writer.close()


# ---------------------------------------------------------------- Islands


def test_admit_keeps_the_best_size_and_evicts():
    isl = Islands(n=2, size=3, seed=0)
    a, b, c = cand(0.5, island=1), cand(0.6, island=1), cand(0.7, island=1)
    assert all(isl.admit(x) for x in (a, b, c))
    assert not isl.admit(cand(0.1, island=1))
    assert isl.admit(cand(0.8, island=1))
    assert a.id not in {m.id for m in isl.members[1]}
    assert isl.members[0] == []
    assert isl.admit(b)  # already a member: stays, is not duplicated
    assert len(isl.members[1]) == 3


def test_admit_breaks_ties_by_length():
    isl = Islands(n=1, size=2, seed=0)
    long1, long2 = cand(0.5, length=20), cand(0.5, length=20)
    isl.admit(long1)
    isl.admit(long2)
    short = cand(0.5, length=3)
    assert isl.admit(short)
    assert {m.id for m in isl.members[0]} == {short.id, long1.id}
    assert not isl.admit(cand(0.5, length=25))


def test_select_is_a_deterministic_tournament():
    def picks(seed):
        isl = Islands(n=1, size=10, seed=seed)
        for i in range(10):
            isl.admit(cand(i / 10))
        return [[c.scores["val_mean"] for c in isl.select(0, k=2)] for _ in range(20)]

    runs = picks(7)
    assert runs == picks(7)
    assert runs != picks(8)
    for a, b in runs:
        assert a != b  # two distinct parents
    # tournaments of 3 from 10 can never return the two worst
    assert min(min(p) for p in runs) >= 0.2


def test_select_with_few_members():
    isl = Islands(n=2, size=5, seed=0)
    assert isl.select(0) == []
    only = cand()
    isl.admit(only)
    assert [c.id for c in isl.select(0, k=2)] == [only.id, only.id]


def test_migrate_is_a_ring():
    isl = Islands(n=3, size=4, seed=0)
    bests = [cand(0.9, island=i) for i in range(3)]
    for i in range(3):
        isl.admit(bests[i])
        isl.admit(cand(0.1, island=i))
    isl.migrate()
    for i in range(3):
        ids = {m.id for m in isl.members[i]}
        assert bests[(i - 1) % 3].id in ids  # got the previous island's best
        assert bests[i].id in ids
        assert all(m.island == i for m in isl.members[i])
    assert bests[0].island == 0  # the original is not moved, only copied


def test_migrate_does_not_chain_within_one_call():
    isl = Islands(n=3, size=4, seed=0)
    star = cand(1.0, island=0)
    isl.admit(star)
    isl.admit(cand(0.2, island=1))
    isl.admit(cand(0.2, island=2))
    isl.migrate()
    assert star.id in {m.id for m in isl.members[1]}
    assert star.id not in {m.id for m in isl.members[2]}


def test_diversity_slot_keeps_one_other_layout():
    isl = Islands(n=1, size=3, seed=0)
    other = cand(0.3, sig="B")
    isl.admit(other)
    for v in (0.9, 0.8, 0.7, 0.6):
        isl.admit(cand(v, sig="A"))
    members = isl.members[0]
    assert other.id in {m.id for m in members}
    assert [m.scores["val_mean"] for m in members] == [0.9, 0.8, 0.3]
    # a better "B" takes the slot from the worse one
    better_b = cand(0.4, sig="B")
    assert isl.admit(better_b)
    assert other.id not in {m.id for m in isl.members[0]}
    # an unsigned candidate is not a different layout
    assert not isl.admit(cand(0.5))


def test_diversity_slot_goes_by_rank_when_the_top_is_already_mixed():
    isl = Islands(n=1, size=3, seed=0)
    for v, s in ((0.9, "A"), (0.8, "B"), (0.7, "A"), (0.2, "C")):
        isl.admit(cand(v, sig=s))
    assert [m.scores["val_mean"] for m in isl.members[0]] == [0.9, 0.8, 0.7]


def test_diversity_slot_follows_the_best_signature():
    isl = Islands(n=1, size=2, seed=0)
    isl.admit(cand(0.5, sig="A"))
    isl.admit(cand(0.4, sig="A"))
    b = cand(0.9, sig="B")
    isl.admit(b)  # B is now the best; the slot now wants something other than B
    assert [m.layout_signature for m in isl.members[0]] == ["B", "A"]
    isl.admit(cand(0.8, sig="B"))
    assert [m.layout_signature for m in isl.members[0]] == ["B", "A"]


def test_rebuild_and_state_resume(store):
    isl = Islands(n=2, size=3, seed=5)
    for i in range(12):
        c = cand(((i * 7) % 12) / 12, island=i % 2, sig="AB"[i % 3 == 0])
        store.add(c)
        isl.admit(c)
    rebuilt = Islands.rebuild(store, 2, 3, 5)
    assert [[m.id for m in p] for p in rebuilt.members] == [[m.id for m in p] for p in isl.members]

    isl.migrate()
    isl.select(0)
    state = json.loads(json.dumps(isl.state()))
    resumed = Islands.from_state(state, store)
    assert [[m.id for m in p] for p in resumed.members] == [[m.id for m in p] for p in isl.members]
    assert [c.id for c in resumed.select(1, k=2)] == [c.id for c in isl.select(1, k=2)]


# ---------------------------------------------------------------- MAP-Elites


def test_map_elites_bins_and_replaces():
    me = MapElites({"walk": [10.0, 20.0], "entities": [3.0]})
    assert me.coverage() == 0
    a = cand(0.5, desc={"walk": 5.0, "entities": 2})
    assert me.cell(a) == (0, 0)
    assert me.cell(cand(desc={"walk": 10.0, "entities": 3})) == (1, 1)  # edge goes up
    assert me.cell(cand(desc={"walk": 99.0, "entities": 0})) == (2, 0)
    assert me.admit(a)
    assert not me.admit(cand(0.4, desc={"walk": 7.0, "entities": 1}))
    assert not me.admit(cand(0.5, length=9, desc={"walk": 7.0, "entities": 1}))
    shorter = cand(0.5, length=2, desc={"walk": 7.0, "entities": 1})
    assert me.admit(shorter)
    better = cand(0.8, length=40, desc={"walk": 1.0, "entities": 0})
    assert me.admit(better)
    assert me.cells() == {(0, 0): better}
    assert me.admit(cand(0.1, desc={"walk": 15.0, "entities": 5}))
    assert me.coverage() == pytest.approx(2 / 6)
    assert not me.admit(cand(0.9, desc={"walk": 1.0}))  # missing descriptor
    assert not me.admit(cand(0.9))


def test_map_elites_rejects_unsorted_edges():
    with pytest.raises(ValueError):
        MapElites({"walk": [5.0, 1.0]})


# ---------------------------------------------------------------- manifest


def test_manifest_merges_and_keeps_created(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "evolve-x"
    monkeypatch.setattr(archive, "factory_sim_commit", lambda: "c1")
    m1 = write_manifest(run, model="qwen", evaluator_version="1", scenes={"open_patch": "d1"})
    assert m1["factory_sim_commit"] == "c1"
    created = m1["created"]

    monkeypatch.setattr(archive, "factory_sim_commit", lambda: "c2")
    m2 = write_manifest(run, evaluator_version="2", resumed=True)
    on_disk = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == m2
    assert m2["created"] == created
    assert m2["model"] == "qwen"  # kept from the first write
    assert m2["evaluator_version"] == "2"  # new value wins
    assert m2["resumed"] is True
    assert m2["factory_sim_commit"] == "c2"
    assert m2["previous_factory_sim_commits"] == ["c1"]

    m3 = write_manifest(run, created="ignored")
    assert m3["created"] == created
    assert "previous_factory_sim_commits" in m3 and m3["previous_factory_sim_commits"] == ["c1"]


def test_factory_sim_commit_reads_git(tmp_path):
    commit = archive.factory_sim_commit()
    assert commit is None or len(commit) == 40
    assert archive.factory_sim_commit(tmp_path / "not-a-repo") is None
