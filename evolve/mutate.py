"""Prompts that ask the model to change a builder program.

Each operator returns an OpenAI-style message list: one system message that
fixes the task, the program contract, the API and the game's mechanics, and one
user message that carries the parent (or parents) and says what to change.

The system message is the same for every operator on a given API reference, so a
provider that caches prompt prefixes can reuse it. `GAME_NOTES` is read when a
prompt is built, so the orchestrator can set it to "" to measure what the notes
are worth.
"""

from __future__ import annotations

import hashlib
import json

TASK = """\
Task: construct_smelting_line.
The scene has an iron ore patch. The character starts holding 2 burner mining drills, \
2 stone furnaces and 60 coal. Build a working smelting line: a burner mining drill on ore \
whose output drops into a stone furnace, with both machines fuelled, so that at least 10 \
iron plates are machine-made during a verification window after the build phase. Some \
scenes have walls; walking straight into a wall wastes decisions."""

ALLOWED_BUILTINS = (
    "abs all any bool dict enumerate filter float int isinstance len list map max min range "
    "reversed round set sorted sum tuple zip"
)

CONTRACT = f"""\
Program contract:
- Exactly one top-level `def build(world):` (helper functions, nested or top-level, are fine).
- Use only `world.*` methods, `math.*` (`math` is already global; never import it), these \
builtins: {ALLOWED_BUILTINS}, and True/False/None.
- Attributes: `world.*`, `math.*`, entity fields, and ordinary list, dict, set and str \
methods (no `format`). No attribute may start with `_`; your own names may, but not with `__`.
- Banned: import, open, exec, eval, compile, getattr, setattr, globals, locals, vars, print, \
global/nonlocal, try/except/raise/assert, with, class, yield, async, match.
- At most 300 lines (and 20,000 characters)."""

#: Mechanics the API reference does not state. Set to "" to leave them out.
GAME_NOTES = """\
Game notes:
- Every action is one decision of about 30 ticks.
- The build phase has a 600-decision budget; running out ends it.
- `place` only reaches tiles within 5 tiles (Chebyshev distance) of the character's own \
tile, never the tile the character stands on, and never an occupied or blocked tile.
- A 2x2 machine placed at (x, y) covers tiles x..x+1, y..y+1; its centre is (x+1, y+1).
- A burner mining drill facing S drops its ore at about (cx+0.5, cy+1.3) from its centre \
(cx, cy); rotating the facing rotates that point about the centre.
- A furnace catches the ore if its 2x2 footprint contains the drop point.
- Both machines need fuel (coal) to run.
- A drill only mines if its footprint is on ore."""

OUTPUT_FORMAT = """\
Output format: first a brief plan of at most 5 lines, then exactly one ```python fenced \
block containing the complete program. Nothing after the block."""

HINTS = [
    "Route around blocked tiles instead of walking in a straight line toward the target.",
    "Choose a stand tile from which both machines' placement tiles are within reach, so "
    "the whole line is built without moving again.",
    "Verify each placement succeeded, and if it did not, try the next candidate layout "
    "instead of carrying on.",
    "Enumerate candidate layouts and prefer the one whose stand tile is closest to the start.",
    "Fuel the drill before the furnace, so mining starts as early as possible.",
    "Check entities() after placing to find the new machine's row rather than assuming it.",
    "Compute the drill's drop point from its facing and place the furnace so its footprint "
    "covers that point; pick the facing that points toward free, unblocked tiles.",
    "Require all four tiles under the drill to be ore, not just the placement tile.",
    "Walk with long strides while far away and switch to steps or nudges near the stand "
    "tile, so you do not overshoot.",
    "Spend coal deliberately: give both machines enough fuel to last the verification "
    "window, and keep the rest in reserve.",
    "Watch decisions_left() and fall back to the simplest working layout when the budget runs low.",
    "Build the second drill-furnace pair as a backup only after the first is fuelled.",
    "When a move is refused or does not change your tile, treat that direction as blocked "
    "and try a detour.",
    "Keep the program short and deterministic: fewer branches means fewer ways to fail.",
]


def system_prompt(api_reference: str) -> str:
    parts = [
        "You write and improve short Python programs that build factories in a Factorio-like "
        "simulator. A program is run once per scene; it acts only through the `world` API.",
        TASK,
        CONTRACT,
        "API reference:\n" + api_reference.strip(),
    ]
    if GAME_NOTES:
        parts.append(GAME_NOTES)
    parts.append(OUTPUT_FORMAT)
    return "\n\n".join(parts)


def _scores_table(scores: dict | None) -> str:
    """Rates by split and family, e.g. `train  open_patch   0.85`, plus a mean per split."""
    if not scores:
        return ""
    rows = []
    for split in ("train", "val"):
        fams = scores.get(split)
        if not isinstance(fams, dict) or not fams:
            continue
        for fam in sorted(fams):
            rows.append((split, fam, f"{float(fams[fam]):.2f}"))
        mean = sum(float(v) for v in fams.values()) / len(fams)
        rows.append((split, "(mean)", f"{mean:.2f}"))
    if not rows:
        return ""
    w = max(len(r[1]) for r in rows)
    lines = [f"{'split':<6} {'family':<{w}}  success"]
    lines += [f"{s:<6} {f:<{w}}  {r}" for s, f, r in rows]
    return "Success rate per scene family:\n" + "\n".join(lines)


def _traces(traces: dict | None, max_lines: int = 20, max_chars: int = 160) -> str:
    """The last intents from a failing episode of each family, as short blocks."""
    if not traces:
        return ""
    blocks = []
    for fam in sorted(traces):
        lines = [str(x) for x in (traces[fam] or [])][-max_lines:]
        if not lines:
            continue
        body = "\n".join(x if len(x) <= max_chars else x[: max_chars - 3] + "..." for x in lines)
        blocks.append(f"[{fam}]\n{body}")
    if not blocks:
        return ""
    return "Last actions of a failing episode, per family:\n" + "\n\n".join(blocks)


def _render(parent: dict, name: str = "Program") -> str:
    parts = []
    code = (parent.get("code") or "").strip("\n")
    if code:
        parts.append(f"{name}:\n```python\n{code}\n```")
    for section in (_scores_table(parent.get("scores")), _traces(parent.get("traces"))):
        if section:
            parts.append(section)
    return "\n\n".join(parts)


def _messages(api_reference: str, user: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt(api_reference)},
        {"role": "user", "content": user.strip() + "\n\n" + OUTPUT_FORMAT},
    ]


def prompt_fix(api_reference: str, parent: dict) -> list[dict]:
    user = (
        _render(parent)
        + "\n\nFind the most likely cause of the failures shown in the scores and traces, "
        "and make a targeted fix. Keep what already works; change as little as needed."
    )
    return _messages(api_reference, user)


def prompt_rewrite(api_reference: str, parent: dict, hint: str) -> list[dict]:
    user = (
        _render(parent)
        + "\n\nRewrite this program, following this strategy hint:\n"
        + hint.strip()
        + "\nYou may restructure freely, but the result must still satisfy the contract."
    )
    return _messages(api_reference, user)


def prompt_crossover(api_reference: str, a: dict, b: dict) -> list[dict]:
    user = (
        _render(a, "Program A")
        + "\n\n"
        + _render(b, "Program B")
        + "\n\nWrite one program that combines the strengths of A and B: take from each "
        "the parts that make it succeed on the families where it scores higher."
    )
    return _messages(api_reference, user)


def prompt_simplify(api_reference: str, parent: dict) -> list[dict]:
    user = (
        _render(parent)
        + "\n\nSimplify this program: fewer lines, clearer names, no dead code. It must "
        "behave exactly the same, issuing the same actions in the same order on every "
        "scene. Do not change behaviour, even to fix a failure."
    )
    return _messages(api_reference, user)


OPERATORS = {
    "fix": prompt_fix,
    "rewrite": prompt_rewrite,
    "crossover": prompt_crossover,
    "simplify": prompt_simplify,
}


def prompt_hash(messages: list[dict]) -> str:
    """sha256 of the messages as canonical JSON, to tie a candidate to its prompt."""
    canon = json.dumps(messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()
