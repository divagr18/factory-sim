"""The builder-program sandbox: what it refuses, what it runs, and how it hashes."""

import pytest

from evolve.sandbox import SAFE_BUILTINS, SandboxError, check, load, normalized_hash


def _in_build(body: str) -> str:
    lines = "\n".join("    " + line for line in body.splitlines())
    return f"def build(world):\n{lines}\n"


REJECT = [
    # imports, anywhere
    ("import os\ndef build(world):\n    pass\n", "import"),
    ("from os import path\ndef build(world):\n    pass\n", "import"),
    (_in_build("import os"), "import"),
    (_in_build("def helper():\n    from os import system\n    return system"), "import"),
    # dunder and private escapes
    (_in_build("return ().__class__.__bases__"), "attribute '__class__'"),
    (_in_build("return world._env"), "attribute '_env'"),
    (_in_build("return math.__dict__"), "attribute '__dict__'"),
    (_in_build("return world.inventory().__class__"), "attribute '__class__'"),
    (_in_build("return (x for x in []).gi_frame"), "attribute 'gi_frame'"),
    (_in_build('return "{0.__class__}".format(world)'), "attribute 'format'"),
    (_in_build("return __builtins__"), "name '__builtins__'"),
    (_in_build("__x = 1"), "name '__x'"),
    (_in_build("return world.me(__n=1)"), "keyword '__n'"),
    ("def __helper(world):\n    pass\ndef build(world):\n    pass\n", "name '__helper'"),
    # the freedoms below open nothing: attributes are still checked everywhere
    (_in_build("_x = world\nreturn _x._env"), "attribute '_env'"),
    (_in_build("f = lambda: ().__class__\nreturn f()"), "attribute '__class__'"),
    (_in_build("return (lambda w: w._env)(world)"), "attribute '_env'"),
    (_in_build("return (lambda *a: a)"), "plain positional"),
    (_in_build("return (lambda: getattr)(world)"), "'getattr'"),
    (_in_build("if (__b := 1):\n    pass"), "name '__b'"),
    (_in_build("return set().__init__"), "attribute '__init__'"),
    (_in_build('return "{0.x}".format(world)'), "attribute 'format'"),
    # forbidden builtins
    (_in_build('return getattr(world, "_env")'), "'getattr'"),
    (_in_build('setattr(world, "x", 1)'), "'setattr'"),
    (_in_build('print("hi")'), "'print'"),
    (_in_build('exec("1")'), "'exec'"),
    (_in_build('return eval("1")'), "'eval'"),
    (_in_build('return compile("1", "f", "eval")'), "'compile'"),
    (_in_build('return open("x")'), "'open'"),
    (_in_build("return globals()"), "'globals'"),
    (_in_build("return locals()"), "'locals'"),
    (_in_build("return vars(world)"), "'vars'"),
    (_in_build('return __import__("os")'), "name '__import__'"),
    (_in_build("return type(world)"), "'type'"),
    (_in_build("return os.getcwd()"), "name 'os' is not available"),
    (_in_build("return math.nope(1)"), "math.nope"),
    (_in_build("xs = []\nreturn xs.frobnicate"), "attribute 'frobnicate'"),
    # banned statements and expressions
    (_in_build("try:\n    pass\nexcept Exception:\n    pass"), "try"),
    (_in_build("raise ValueError()"), "raise"),
    (_in_build("assert world"), "assert"),
    (_in_build("with world:\n    pass"), "with"),
    ("class A:\n    pass\ndef build(world):\n    pass\n", "class"),
    (_in_build("class A:\n    pass"), "class"),
    (_in_build("yield 1"), "yield"),
    (_in_build("yield from []"), "yield"),
    ("async def build(world):\n    pass\n", "async def"),
    (_in_build("async def f():\n    await f()"), "async def"),
    ("x = 1\ndef build(world):\n    global x\n", "only def"),
    (_in_build("global x"), "global"),
    (_in_build("x = 1\ndef f():\n    nonlocal x"), "nonlocal"),
    (_in_build("match world:\n    case _:\n        pass"), "match"),
    ("x = 1\ndef build(world):\n    pass\n", "only def statements"),
    ("x = 1\ndef build(world):\n    pass\ndel x\n", "only def statements"),
    # f-strings, literals, unpacking, signatures, decorators
    (_in_build('return f"{world:>10}"'), "format spec"),
    (_in_build('return b"x"'), "bytes"),
    (_in_build("d = {}\nreturn max(**d)"), "'**'"),
    (_in_build("def f(*a):\n    pass"), "plain positional"),
    (_in_build("def f(**k):\n    pass"), "plain positional"),
    (_in_build("def f(a, *, b):\n    pass"), "plain positional"),
    (_in_build("@max\ndef f():\n    pass"), "decorators"),
    # program shape
    ("def helper(world):\n    pass\n", "no top-level 'def build(world)'"),
    ("def build(world, extra):\n    pass\n", "exactly one argument"),
    ("def build():\n    pass\n", "exactly one argument"),
    ("def build(world):\n    pass\ndef build(world):\n    pass\n", "defined twice"),
    ("def build(world):\n    return (\n", "syntax error"),
    (_in_build("pass\n" * 300), "lines (limit 300)"),
    (_in_build("x = '" + "a" * 20_000 + "'"), "characters (limit 20000)"),
]


@pytest.mark.parametrize("source,reason", REJECT)
def test_rejects(source, reason):
    with pytest.raises(SandboxError) as e:
        check(source)
    assert reason in str(e.value)
    with pytest.raises(SandboxError):
        load(source)


def test_error_names_the_line():
    source = "def build(world):\n    x = 1\n    return x.__class__\n"
    with pytest.raises(SandboxError, match=r"^line 3: attribute '__class__' is not allowed$"):
        check(source)


def test_builtins_are_exactly_the_spec():
    spec = (
        "abs all any bool dict enumerate filter float int isinstance len list map max min "
        "range reversed round set sorted sum tuple zip"
    ).split()
    assert sorted(SAFE_BUILTINS) == sorted(spec)


class _Entity:
    def __init__(self, row, kind, x, y):
        self.row, self.kind, self.x, self.y = row, kind, x, y
        self.facing, self.fuel, self.contents, self.output = None, 0.0, 0.0, 0.0
        self.working = self.remembered = False


class _FakeWorld:
    def __init__(self):
        self.log = []
        self.refusals = 0

    def me(self):
        return (0.5, 0.5)

    def ore_tiles(self, kind="iron-ore"):
        return [(3, 4), (-2, 1), (5, 5), (1, 1)]

    def entities(self):
        return [_Entity(0, "furnace", 2.0, 2.0), _Entity(1, "mining-drill", 9.0, 9.0)]

    def inventory(self):
        return {"coal": 10, "burner-mining-drill": 2}

    def place(self, item, x, y, facing):
        self.log.append(("place", item, x, y, facing))
        return True

    def give(self, entity, item, amount):
        self.log.append(("give", entity.row, item, amount))
        return True

    def move(self, direction, stride="long"):
        self.log.append(("move", direction, stride))
        return True


VALID = '''\
"""Place drills on the ore nearest to me, then fuel them."""


def nearest_first(world, tiles):
    """A top-level helper."""
    mx, my = world.me()

    def dist(t):
        return math.hypot(t[0] - mx, t[1] - my)

    return sorted(tiles, key=dist)


def build(world: "World") -> None:
    tiles = nearest_first(world, world.ore_tiles())
    have: int = world.inventory().get("burner-mining-drill", 0)
    placed = []
    for i, (x, y) in enumerate(tiles[:have]):
        facing = "S" if i % 2 == 0 else "N"
        if world.place("burner-mining-drill", x, y, facing):
            placed.append((x, y))
    furnaces = {e.row: e for e in world.entities() if e.kind == "furnace"}
    for row in sorted(furnaces.keys()):
        world.give(furnaces[row], "coal", min(5, world.inventory()["coal"]))
    steps = 0
    while steps < 2 and not isinstance(steps, float):
        world.move("N", stride="step")
        steps += 1
    counts = {kind: len([p for p in placed if p[0] > 0]) for kind in ("a", "b")}
    label = f"placed {len(placed)} of {have}"
    total = sum(abs(x) + abs(y) for x, y in zip([1, 2], [3, 4]))
    del counts["a"]
    return label, total, counts, list(reversed(placed)), round(max(1.5, 2.25), 1)
'''


def test_accepts_and_runs_a_representative_program():
    check(VALID)
    build = load(VALID)
    world = _FakeWorld()
    label, total, counts, rev, r = build(world)
    assert label == "placed 2 of 2"
    assert total == 10
    assert counts == {"b": 1}
    assert rev == [(-2, 1), (1, 1)]
    assert r == 2.2
    assert world.log[:2] == [
        ("place", "burner-mining-drill", 1, 1, "S"),
        ("place", "burner-mining-drill", -2, 1, "N"),
    ]
    assert ("give", 0, "coal", 5) in world.log
    assert world.log[-1] == ("move", "N", "step")


def test_world_api_works_through_an_alias():
    check(_in_build("def walk(w):\n    return w.move('N')\nreturn walk(world)"))


def test_loaded_program_sees_only_safe_builtins():
    build = load(_in_build("return len([1, 2]), math.pi"))
    assert build(None)[0] == 2
    g = build.__globals__
    assert set(g) == {"__builtins__", "math", "build"}
    assert set(g["__builtins__"]) == set(SAFE_BUILTINS)


# --- normalized_hash -------------------------------------------------------

BASE = """\
def helper(tiles, limit):
    out = []
    for t in tiles:
        if len(out) < limit:
            out.append(t)
    return out


def build(world):
    tiles = world.ore_tiles()
    chosen = helper(tiles, 3)
    for x, y in chosen:
        world.place("burner-mining-drill", x, y, "S")
    return [t for t in chosen if t[0] > 0]
"""


def test_hash_ignores_layout_comments_and_docstrings():
    variant = '''\
"""Module docstring."""


def helper(tiles,limit):  # trailing comment
    """Helper docstring."""
    out=[]
    for t in tiles:
        # a comment
        if len(out)<limit: out.append(t)
    return out
def build(world) -> None:
    tiles = world.ore_tiles()
    chosen = helper(
        tiles,
        3,
    )
    for (x, y) in chosen:
        world.place("burner-mining-drill", x, y, "S")
    return [t for t in chosen if t[0] > 0]
'''
    check(variant)
    assert normalized_hash(variant) == normalized_hash(BASE)


def test_hash_ignores_consistent_renames():
    renamed = """def pick(candidates, cap):
    acc = []
    for tile in candidates:
        if len(acc) < cap:
            acc.append(tile)
    return acc


def build(w):
    found = w.ore_tiles()
    sel = pick(found, 3)
    for a, b in sel:
        w.place("burner-mining-drill", a, b, "S")
    return [c for c in sel if c[0] > 0]
"""
    check(renamed)
    assert normalized_hash(renamed) == normalized_hash(BASE)


def test_hash_sees_real_changes():
    h = normalized_hash(BASE)
    assert normalized_hash(BASE.replace("3)", "4)")) != h
    assert normalized_hash(BASE.replace('"S"', '"N"')) != h
    assert normalized_hash(BASE.replace("t[0] > 0", "t[1] > 0")) != h


def test_hash_does_not_merge_swapped_variables():
    a = _in_build("p = 1\nq = 2\nreturn p - q")
    b = _in_build("p = 1\nq = 2\nreturn q - p")
    assert normalized_hash(a) != normalized_hash(b)


def test_hash_respects_scopes():
    # `n` in the helper refers to the enclosing local in one program and to its
    # own parameter in the other; the two must not collide.
    a = _in_build("n = 1\ndef f(m):\n    return n\nreturn f(2)")
    b = _in_build("n = 1\ndef f(m):\n    return m\nreturn f(2)")
    assert normalized_hash(a) != normalized_hash(b)
    c = _in_build("k = 1\ndef g(z):\n    return k\nreturn g(2)")
    assert normalized_hash(a) == normalized_hash(c)


def test_hash_keeps_parameters_named_by_keyword():
    a = "def h(n):\n    return n\ndef build(world):\n    return h(n=1)\n"
    b = "def h(m):\n    return m\ndef build(world):\n    return h(n=1)\n"
    assert normalized_hash(a) != normalized_hash(b)


def test_hash_of_refused_program_is_stable():
    bad = "import os\ndef build(world):\n    pass\n"
    assert normalized_hash(bad) == normalized_hash("import os\n\ndef build(world):  # hi\n  pass\n")


#: Ordinary Python that models wrote and the sandbox used to refuse, without the
#: prompt ever saying so. Each must pass check() and run.
ACCEPT = [
    "_blocked = set()\n_blocked.add((1, 2))\nreturn len(_blocked)",
    "def _walk(n):\n    return n + 1\nreturn _walk(1)",
    "return sorted([(2, 'b'), (1, 'a')], key=lambda p: p[0])[0][0]",
    "a = {1, 2, 3}\nb = {2, 3, 4}\nreturn len(a.intersection(b) | a.union(b) - a.difference(b))",
    "a = {1, 2}\na.difference_update({1})\nreturn a.issubset({2, 3}) and a.isdisjoint({5})",
    "if (n := 3) > 2:\n    return n\nreturn 0",
    "return list(map(abs, filter(lambda v: v < 0, [-1, 2, -3])))",
    "return ','.join(['a', 'b']).split(',')[0].strip().upper()",
]


@pytest.mark.parametrize("body", ACCEPT)
def test_accepts_ordinary_python(body):
    source = _in_build(body)
    check(source)
    assert load(source)(None) is not None


def test_hash_is_stable_for_lambda_and_walrus():
    a = _in_build("return sorted([3, 1], key=lambda v: -v)")
    assert normalized_hash(a) == normalized_hash(a + "\n")
