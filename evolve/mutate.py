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

#: `belt_smelting` (FactorioRL `tasks/families/belt_smelting.py` 1.0.0). The
#: numbers restate that module's constants: starting inventory, the 60-plate
#: target, the 36,000-tick window, the 2,500-decision budget, site spacing.
TASK_BELT_SMELTING = """\
Task: belt_smelting.
The scene has an iron ore patch, a coal patch and a wooden chest, each more than 20 tiles \
from the other two, so no single standing tile reaches two of them. `world.marker("iron")`, \
`world.marker("coal")` and `world.marker("output")` give their centres. The character starts \
holding 4 burner mining drills, 4 stone furnaces, 40 transport belts, 8 burner inserters and \
20 coal. Build a line that mines iron ore, smelts it and delivers the plates into the output \
chest: at least 60 iron plates must arrive in that chest during a 36,000-tick (ten-minute) \
verification window that starts after the build phase, with no actions allowed during it. \
Plates smelted from hand-mined ore do not count. Coal is short: 20 is enough only if it is \
not wasted before the window. Some scenes have walls."""

#: Mechanics for `belt_smelting`, measured on the engine (FactorioRL
#: `docs/sim-logistics.md`). Setting `GAME_NOTES` to "" leaves every task's out.
GAME_NOTES_BELT_SMELTING = """\
Game notes:
- Every action is one decision of about 30 ticks (60 ticks = 1 s).
- The build phase has a 2,500-decision budget; running out ends it.
- `place` only reaches tiles within 7 tiles (Chebyshev distance) of the character's own \
tile whose centre is also within 10 tiles (straight line) of `me()`, never the tile the \
character stands on, and never an occupied or blocked tile. The sites are 20 to 40 tiles \
apart, so the character has to walk between them.
- `give`, `take`, `mine` and `rotate` need the entity within 10 tiles of `me()`, measured \
to the edge of the entity's box; `mine_resource` needs the tile's centre within 2.7 tiles.
- Hand-mining takes 2 s per item for ore, coal and stone, and keeps going until the asked \
amount has arrived; a `move` stops it. If an entity stands on the tile, `mine_resource` \
mines that entity instead and then nothing more until you move. With no room left in the \
inventory, a mined item drops on the ground. A `mine_resource` with no free inventory slot \
at all is refused.
- A 2x2 machine placed at (x, y) covers tiles x..x+1, y..y+1; its centre is (x+1, y+1).
- Returning from `build` ends the build phase at once, and so does `world.finish()`: the \
verification window starts then, not when the budget runs out.
- `take_fuel` empties a machine's fuel slot into the inventory; the fuel already burning \
stays in the machine and burns on.
- A belt carries items toward its facing, 1.875 tiles per second. Each belt tile has two \
lanes (lane 1 on the left of travel, lane 2 on the right), each holding at most 4 items. A \
belt whose end meets another belt's side feeds the lane on the side it comes from; a belt \
that feeds straight into the back of another belt continues the line, turning if the next \
belt faces a different way.
- A burner mining drill facing S drops its ore at about (cx+0.5, cy+1.3) from its centre \
(cx, cy); rotating the facing rotates that point about the centre. It mines 0.25 ore/s if its \
footprint is on ore and it has fuel. Onto a belt, the ore lands on the lane nearest the drill.
- A burner inserter faces its pickup: facing N it takes from the tile north of it and drops \
onto the tile south of it (and likewise for the other facings). Onto a belt it drops on the \
far lane (the one away from the inserter). It takes from belts, chests, furnace output and \
the ground, and puts into belts, chests, and machine fuel or input slots.
- An inserter fills a stone furnace to at most 2 ore and at most 5 fuel, and waits instead of \
overfilling. It keeps its own fuel slot stocked from any fuel it moves; a newly placed burner \
inserter already has a little fuel of its own.
- A stone furnace makes one iron plate from one iron ore every 3.2 s while fuelled.
- Drills, furnaces and inserters all burn coal; `world.give` puts coal into a machine's fuel \
slot.
- `world.entities()` reports each belt's lane counts and turn, each inserter's held item, \
pickup and drop points, and each drill's drop point."""

HINTS_BELT_SMELTING = [
    "Put the drills on the iron patch facing a belt, and route the belt to furnaces placed "
    "near the chest, with inserters from the belt into the furnaces and from the furnaces "
    "into the chest.",
    "Smelt next to the ore instead: drills drop straight into furnaces, inserters take the "
    "plates out onto a belt, and the belt runs to an inserter that feeds the chest.",
    "Count belt tiles before building: plan the route as a list of tiles and facings, and "
    "check it fits in the belts you hold before placing the first one.",
    "Place each belt facing the next tile of the route, and turn only at corners.",
    "Check each inserter's pickup and drop points in entities() after placing it.",
    "Walk to each site with long strides and finish with steps or nudges.",
    "Spend coal only on machines that will run during the window, and give each just enough.",
    "Route belts around blocked tiles instead of through them.",
    "Watch decisions_left() and finish a working line before adding a second one.",
]

DEFAULT_TASK = "construct_smelting_line"
#: Per-task prompt text. `construct_smelting_line` reads the module globals
#: (`TASK`, `GAME_NOTES`, `HINTS`), so its prompts are exactly what they were;
#: `GAME_NOTES = ""` switches every task's notes off.
TASK_TEXT = {"belt_smelting": TASK_BELT_SMELTING}
TASK_NOTES = {"belt_smelting": GAME_NOTES_BELT_SMELTING}
TASK_HINTS = {"belt_smelting": HINTS_BELT_SMELTING}
TASKS = (DEFAULT_TASK, *TASK_TEXT)


def _known(task: str | None) -> str:
    task = task or DEFAULT_TASK
    if task not in TASKS:
        raise ValueError(f"no prompt for task {task!r}; known: {', '.join(TASKS)}")
    return task


def task_text(task: str | None = None) -> str:
    return TASK_TEXT.get(_known(task), TASK)


def game_notes(task: str | None = None) -> str:
    """The task's game notes, or "" once `GAME_NOTES` is switched off."""
    task = _known(task)
    if not GAME_NOTES:
        return ""
    return TASK_NOTES.get(task, GAME_NOTES)


def hints(task: str | None = None) -> list[str]:
    return TASK_HINTS.get(_known(task), HINTS)


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


def system_prompt(api_reference: str, task: str | None = None) -> str:
    parts = [
        "You write and improve short Python programs that build factories in a Factorio-like "
        "simulator. A program is run once per scene; it acts only through the `world` API.",
        task_text(task),
        CONTRACT,
        "API reference:\n" + api_reference.strip(),
    ]
    notes = game_notes(task)
    if notes:
        parts.append(notes)
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


def _messages(api_reference: str, user: str, task: str | None = None) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt(api_reference, task)},
        {"role": "user", "content": user.strip() + "\n\n" + OUTPUT_FORMAT},
    ]


def prompt_fix(api_reference: str, parent: dict, task: str | None = None) -> list[dict]:
    user = (
        _render(parent)
        + "\n\nFind the most likely cause of the failures shown in the scores and traces, "
        "and make a targeted fix. Keep what already works; change as little as needed."
    )
    return _messages(api_reference, user, task)


def prompt_rewrite(
    api_reference: str, parent: dict, hint: str, task: str | None = None
) -> list[dict]:
    user = (
        _render(parent)
        + "\n\nRewrite this program, following this strategy hint:\n"
        + hint.strip()
        + "\nYou may restructure freely, but the result must still satisfy the contract."
    )
    return _messages(api_reference, user, task)


def prompt_crossover(api_reference: str, a: dict, b: dict, task: str | None = None) -> list[dict]:
    user = (
        _render(a, "Program A")
        + "\n\n"
        + _render(b, "Program B")
        + "\n\nWrite one program that combines the strengths of A and B: take from each "
        "the parts that make it succeed on the families where it scores higher."
    )
    return _messages(api_reference, user, task)


def prompt_simplify(api_reference: str, parent: dict, task: str | None = None) -> list[dict]:
    user = (
        _render(parent)
        + "\n\nSimplify this program: fewer lines, clearer names, no dead code. It must "
        "behave exactly the same, issuing the same actions in the same order on every "
        "scene. Do not change behaviour, even to fix a failure."
    )
    return _messages(api_reference, user, task)


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
