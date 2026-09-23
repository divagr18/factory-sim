"""Check and load LLM-written builder programs before anything runs them.

A candidate is source text defining `def build(world)`. It is executed thousands
of times an hour, so the gate is static and conservative: the source is parsed
and every node is checked against an allowlist. Anything the allowlist does not
name is refused, so a Python release that adds syntax cannot open a hole.

The escapes that matter in CPython all go through attributes (`().__class__`,
`gen.gi_frame`, `"{0.__class__}".format(x)`) or through builtins (`getattr`,
`__import__`, `type`). So names and attributes starting with `_` are refused
everywhere, the builtins are a short list of pure functions and types, and an
attribute is allowed only on `world` (any public name), on `math` (its public
names), or from `SAFE_ATTRS`, which covers the entity fields and the container
methods that ordinary data code needs. `load` compiles the checked tree itself,
not the source again, and runs it with `SAFE_BUILTINS` as its only builtins.

What this does not stop is resource use: `range(10**12)` or `"a" * 10**10`
passes the check. The evaluation pool's per-job timeout and crash recovery
handle that, which is why programs only ever run inside a pool worker.

Error messages are short and name a line, because they go back to the LLM.
"""

from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import math

MAX_LINES = 300
MAX_CHARS = 20_000

#: The builtins a program may call: pure functions and plain types.
SAFE_BUILTIN_NAMES = (
    "abs all any bool dict enumerate filter float int isinstance len list map max min range "
    "reversed round set sorted sum tuple zip"
).split()
SAFE_BUILTINS = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES}

#: `World`'s public API. These may also be read off a name other than `world`,
#: so `def walk(w): w.move("N")` works; they are not attributes of any
#: builtin type a program can reach.
WORLD_API = frozenset(
    "me tile patch inventory ore_tiles blocked_tiles entities decisions_left last_refused "
    "move place give take mine wait refusals failures".split()
)

#: Attributes allowed on any value: `Entity` fields, then list, dict, set and str
#: methods that only read, build or mutate plain data. `format` and `format_map`
#: are left out on purpose: a format string can walk attributes.
SAFE_ATTRS = (
    frozenset(
        "x y row kind facing fuel contents output working remembered "
        # list and dict
        "append extend insert pop remove index count sort reverse copy clear "
        "get items keys values update setdefault "
        # set
        "add discard union intersection difference symmetric_difference issubset "
        "issuperset isdisjoint intersection_update difference_update "
        "symmetric_difference_update "
        # str
        "startswith endswith join split strip lstrip rstrip lower upper replace".split()
    )
    | WORLD_API
)

MATH_NAMES = frozenset(n for n in dir(math) if not n.startswith("_"))

#: Refused by name even when a program binds them itself, so the message is clear.
FORBIDDEN_NAMES = frozenset(
    "print exec eval compile open getattr setattr delattr hasattr globals locals vars dir "
    "type object super input breakpoint help exit quit memoryview bytearray classmethod "
    "staticmethod property id".split()
)

_ALLOWED_NODES = frozenset(
    getattr(ast, n)
    for n in (
        # module and functions
        "Module FunctionDef arguments arg keyword "
        # statements
        "Return Assign AugAssign AnnAssign For While If Expr Pass Break Continue Delete "
        # expressions
        "BoolOp BinOp UnaryOp IfExp Dict Set ListComp SetComp DictComp GeneratorExp "
        "Lambda NamedExpr "
        "comprehension Compare Call JoinedStr FormattedValue Constant Attribute Subscript "
        "Starred Name List Tuple Slice "
        # contexts and operators
        "Load Store Del And Or Add Sub Mult Div FloorDiv Mod Pow LShift RShift BitOr BitXor "
        "BitAnd Invert Not UAdd USub Eq NotEq Lt LtE Gt GtE Is IsNot In NotIn"
    ).split()
)

_BANNED_WORDS = {
    "Import": "import",
    "ImportFrom": "import",
    "Global": "global",
    "Nonlocal": "nonlocal",
    "Try": "try",
    "TryStar": "try",
    "Raise": "raise",
    "Assert": "assert",
    "With": "with",
    "AsyncWith": "async with",
    "ClassDef": "class",
    "Yield": "yield",
    "YieldFrom": "yield",
    "Await": "await",
    "AsyncFunctionDef": "async def",
    "AsyncFor": "async for",
    "Match": "match",
}

_CONSTANT_TYPES = (bool, int, float, str, type(None))
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


class SandboxError(Exception):
    """A program was refused; the message is short enough to show the LLM."""


def _err(node: ast.AST | None, msg: str) -> SandboxError:
    line = getattr(node, "lineno", None)
    return SandboxError(f"line {line}: {msg}" if line else msg)


def _is_docstring(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _bound_names(tree: ast.Module) -> set[str]:
    """Every name the program binds anywhere: parameters, targets, defs.

    One module-wide set is looser than Python's scoping, but a name bound in
    another function only turns into a NameError at run time; it can never reach
    anything outside the program's own globals.
    """
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
    return names


class _Checker(ast.NodeVisitor):
    def __init__(self, visible: set[str]):
        self.visible = visible

    def visit(self, node: ast.AST) -> None:
        kind = type(node)
        if kind not in _ALLOWED_NODES:
            word = _BANNED_WORDS.get(kind.__name__)
            if word:
                raise _err(node, f"{word} is not allowed")
            raise _err(node, f"{kind.__name__} is not allowed")
        method = getattr(self, "visit_" + kind.__name__, None)
        if method:
            method(node)
        else:
            self.generic_visit(node)

    def _ident(self, node: ast.AST, name: str, what: str) -> None:
        # A program's own names may start with one underscore (`_blocked`,
        # `_walk`): a name only ever resolves to the program's own bindings or
        # SAFE_BUILTINS, so it reaches nothing. A double underscore is refused,
        # because dunders are how an escape starts. `World`'s internals are
        # single-underscore *attributes*, and visit_Attribute refuses every
        # attribute that starts with `_`.
        if name.startswith("__"):
            raise _err(node, f"{what} '{name}' is not allowed")
        if name in FORBIDDEN_NAMES:
            raise _err(node, f"'{name}' is not allowed")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._ident(node, node.name, "name")
        if node.decorator_list:
            raise _err(node, "decorators are not allowed")
        a = node.args
        if a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg:
            raise _err(node, f"'{node.name}' may only have plain positional parameters")
        for arg in a.args:
            self._ident(arg, arg.arg, "parameter")
        for default in a.defaults:
            self.visit(default)
        for stmt in node.body:
            self.visit(stmt)
        # Annotations are never checked or run: load() strips them.

    def visit_Lambda(self, node: ast.Lambda) -> None:
        a = node.args
        if a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg:
            raise _err(node, "a lambda may only have plain positional parameters")
        for arg in a.args:
            self._ident(arg, arg.arg, "parameter")
        for default in a.defaults:
            self.visit(default)
        self.visit(node.body)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.target)
        if node.value is not None:
            self.visit(node.value)

    def visit_Name(self, node: ast.Name) -> None:
        self._ident(node, node.id, "name")
        if isinstance(node.ctx, ast.Load) and node.id not in self.visible:
            raise _err(node, f"name '{node.id}' is not available")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        attr = node.attr
        base = node.value
        self.visit(base)  # innermost first, so the message names the first problem
        if attr.startswith("_"):
            raise _err(node, f"attribute '{attr}' is not allowed")
        if isinstance(base, ast.Name) and base.id == "math":
            if attr not in MATH_NAMES:
                raise _err(node, f"math.{attr} is not available")
        elif not (isinstance(base, ast.Name) and base.id == "world") and attr not in SAFE_ATTRS:
            raise _err(node, f"attribute '{attr}' is not allowed")

    def visit_keyword(self, node: ast.keyword) -> None:
        if node.arg is None:
            raise _err(node, "'**' unpacking in calls is not allowed")
        self._ident(node, node.arg, "keyword")
        self.visit(node.value)

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, _CONSTANT_TYPES):
            raise _err(node, f"{type(node.value).__name__} literals are not allowed")

    def visit_FormattedValue(self, node: ast.FormattedValue) -> None:
        if node.format_spec is not None:
            raise _err(node, "f-string format specs are not allowed")
        self.visit(node.value)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        if node.is_async:
            raise _err(node.target, "async comprehensions are not allowed")
        self.generic_visit(node)


def _check_size(source: str) -> None:
    if len(source) > MAX_CHARS:
        raise SandboxError(f"program is {len(source)} characters (limit {MAX_CHARS})")
    lines = len(source.splitlines())
    if lines > MAX_LINES:
        raise SandboxError(f"program is {lines} lines (limit {MAX_LINES})")


def _check_tree(tree: ast.Module) -> None:
    body = tree.body[1:] if tree.body and _is_docstring(tree.body[0]) else tree.body
    seen = set()
    for stmt in body:
        if not isinstance(stmt, ast.FunctionDef):
            word = _BANNED_WORDS.get(type(stmt).__name__)
            if word:
                raise _err(stmt, f"{word} is not allowed")
            raise _err(stmt, "only def statements are allowed at top level")
        if stmt.name in seen:
            raise _err(stmt, f"function '{stmt.name}' is defined twice")
        seen.add(stmt.name)
    build = next((s for s in body if s.name == "build"), None)
    if build is None:
        raise SandboxError("no top-level 'def build(world)' found")
    a = build.args
    if len(a.args) != 1 or a.defaults or a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg:
        raise _err(build, "build must take exactly one argument: def build(world)")
    visible = _bound_names(tree) | set(SAFE_BUILTIN_NAMES) | {"math", "world"}
    checker = _Checker(visible)
    for stmt in body:
        checker.visit(stmt)


def _parse(source: str) -> ast.Module:
    if not isinstance(source, str):
        raise SandboxError("program must be a string")
    _check_size(source)
    try:
        return ast.parse(source, "<program>")
    except SyntaxError as e:
        raise SandboxError(f"line {e.lineno}: syntax error: {e.msg}") from None
    except (ValueError, RecursionError, MemoryError) as e:
        raise SandboxError(f"cannot parse: {type(e).__name__}") from None


def _parse_checked(source: str) -> ast.Module:
    tree = _parse(source)
    try:
        _check_tree(tree)
    except RecursionError:
        raise SandboxError("program is nested too deeply") from None
    return tree


def check(source: str) -> None:
    """Raise `SandboxError` if `source` is not an acceptable builder program."""
    _parse_checked(source)


class _StripAnnotations(ast.NodeTransformer):
    """Drop annotations, which would otherwise be evaluated at def time unchecked."""

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.annotation = None
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.returns = None
        self.generic_visit(node)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.stmt:
        if node.value is None:
            return ast.copy_location(ast.Pass(), node)
        return ast.copy_location(ast.Assign(targets=[node.target], value=node.value), node)


def load(source: str):
    """Check `source`, run it in restricted globals and return its `build`."""
    tree = _StripAnnotations().visit(_parse_checked(source))
    ast.fix_missing_locations(tree)
    code = compile(tree, "<program>", "exec")
    env = {"__builtins__": dict(SAFE_BUILTINS), "math": math}
    exec(code, env)
    return env["build"]


# --- normalised hashing ---------------------------------------------------


def _strip_docstrings(tree: ast.Module) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef)) and node.body:
            if _is_docstring(node.body[0]):
                node.body = node.body[1:] or [ast.Pass()]


def _local_binds(nodes: list[ast.AST]) -> set[str]:
    """Names bound directly in one function scope, not in nested ones."""
    out = set()
    stack = list(nodes)
    while stack:
        n = stack.pop()
        if isinstance(n, ast.FunctionDef):
            out.add(n.name)
            stack.extend(n.args.defaults)
            continue
        if isinstance(n, _COMPREHENSIONS):
            stack.append(n.generators[0].iter)
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        stack.extend(ast.iter_child_nodes(n))
    return out


class _Scope:
    def __init__(self, bound: set[str], depth: int, keep: frozenset[str] = frozenset()):
        self.bound, self.keep, self.prefix = bound, keep, f"_{depth}_"
        self.names: dict[str, str] = {}

    def name(self, n: str) -> str:
        if n in self.keep:
            return n
        if n not in self.names:
            self.names[n] = f"{self.prefix}{len(self.names)}"
        return self.names[n]


class _Renamer:
    """Alpha-rename every program-bound name, scope by scope, in first-use order.

    Canonical names start with `_`, which programs cannot use, so they never
    collide with `math`, `world` methods or builtins. A parameter whose name is
    also used as a keyword argument somewhere keeps its name: renaming it would
    make `h(n=1)` look equal to a program where `h` takes no `n`.
    """

    def __init__(self, keywords: frozenset[str]):
        self.keywords = keywords

    def resolve(self, name: str, scopes: list[_Scope]) -> str:
        for scope in reversed(scopes):
            if name in scope.bound:
                return scope.name(name)
        return name

    def run(self, node: ast.AST, scopes: list[_Scope]) -> None:
        if isinstance(node, ast.FunctionDef):
            node.name = self.resolve(node.name, scopes)
            for d in node.args.defaults:
                self.run(d, scopes)
            params = {a.arg for a in node.args.args}
            inner = _Scope(_local_binds(node.body) | params, len(scopes), self.keywords & params)
            for a in node.args.args:
                a.arg = inner.name(a.arg)
            for stmt in node.body:
                self.run(stmt, scopes + [inner])
        elif isinstance(node, _COMPREHENSIONS):
            gens = node.generators
            self.run(gens[0].iter, scopes)
            targets = {n.id for g in gens for n in ast.walk(g.target) if isinstance(n, ast.Name)}
            inner = scopes + [_Scope(targets, len(scopes))]
            for i, g in enumerate(gens):
                self.run(g.target, inner)
                if i:
                    self.run(g.iter, inner)
                for cond in g.ifs:
                    self.run(cond, inner)
            if isinstance(node, ast.DictComp):
                self.run(node.key, inner)
                self.run(node.value, inner)
            else:
                self.run(node.elt, inner)
        else:
            if isinstance(node, ast.Name):
                node.id = self.resolve(node.id, scopes)
            for child in ast.iter_child_nodes(node):
                self.run(child, scopes)


def normalized_hash(source: str) -> str:
    """sha256 of the program's AST, blind to layout, comments, docstrings,
    annotations and consistent renaming of its own functions and variables.

    Renaming is applied only to programs that pass `check`, because the scope
    model assumes no `global`, `nonlocal`, `lambda`, `:=` or `class`. A refused
    program, or one using `lambda` or `:=`, still gets a stable hash, just
    without renaming.
    """
    tree = _parse(source)
    tree = _StripAnnotations().visit(tree)
    _strip_docstrings(tree)
    try:
        _check_tree(copy.deepcopy(tree))
    except (SandboxError, RecursionError):
        pass
    else:
        if any(isinstance(n, (ast.Lambda, ast.NamedExpr)) for n in ast.walk(tree)):
            return hashlib.sha256(
                ast.dump(tree, annotate_fields=True, include_attributes=False).encode()
            ).hexdigest()
        keywords = frozenset(n.arg for n in ast.walk(tree) if isinstance(n, ast.keyword) and n.arg)
        top = {s.name for s in tree.body if isinstance(s, ast.FunctionDef)}
        module = _Scope(top, 0, frozenset({"build"}))
        renamer = _Renamer(keywords)
        for stmt in tree.body:
            renamer.run(stmt, [module])
    dump = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(dump.encode()).hexdigest()
