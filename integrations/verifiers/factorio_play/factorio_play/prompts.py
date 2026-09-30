"""The system and user prompts, per task.

No `verifiers` import, so the prompts can be checked where `verifiers.v1`
cannot load. `construct_smelting_line`'s prompts are exactly 0.1.1's; the
test `test_construct_smelting_line_prompts_are_unchanged` pins them.

The task text and the game notes are factory-sim's own (`evolve.mutate`),
the same text `factorio-build` shows a program writer. Only the note on how
tools map onto the `world` API is this package's.
"""

from __future__ import annotations

from evolve import evaluate, mutate

BELT_TASK = "belt_smelting"

TOOLS_NOTE = """\
You act through tools that mirror the `world` API one to one: queries (me, tile, \
inventory, ore_tiles, blocked_tiles, entities, patch, decisions_left, last_refused) \
cost nothing; actions (move, place, give, take, mine, wait) each spend a decision. \
give/take/mine name an entity by its current `row` in entities(). Every tool returns \
{"ok": ..., "result": ...} or an error. Call finish() once the line is built and \
fuelled: the episode is scored only after finish()."""

TOOLS_NOTE_V3 = """\
You act through tools that mirror the `world` API: where the text says `world.<name>`, \
call the tool <name>. Queries (me, tile, inventory, ore_tiles, blocked_tiles, entities, \
marker, belt_lanes, opened, decisions_left, last_refused) cost nothing. Actions spend \
decisions, one per world action: move, place, rotate, give, take, take_fuel, mine, \
mine_resource, inspect, wait and finish. A tool that takes a count or an amount can run \
several world actions in one call: move and wait one per count; give, take and take_fuel \
transfers of 20, 5 and 1 (giving 23 coal is four: 20, 1, 1 and 1); mine_resource requests \
of 20, 5 and 1, each followed by the waits for its items. Its reply says how many \
decisions it spent. rotate, give, take, take_fuel, mine, inspect and belt_lanes name an \
entity by its current `row` in entities(); rows renumber as the character moves. Replies \
are compact JSON. An action's reply has "ok": true when every step went through, \
"refused" with the reason when the action space refused a step (no decision spent on it), \
and "failed" when the game refused a legal step (it still cost a decision). There is no \
program here: calling finish() is what returning from `build` does. It ends the build \
phase, runs the verification window and scores the episode; nothing can be done after it."""

#: The user message, per task.
PROMPTS = {
    evaluate.TASK: "Build the smelting line in this scene with the tools, then call finish().",
    BELT_TASK: "Build a line that delivers iron plates into the output chest in this scene "
    "with the tools, then call finish().",
}


def system_prompt(game_notes: bool = True, task: str = evaluate.TASK) -> str:
    if task == evaluate.TASK:
        parts = [
            "You build factories in a Factorio-like simulator by calling tools.",
            mutate.TASK,
            TOOLS_NOTE,
        ]
        if game_notes and mutate.GAME_NOTES:
            parts.append(mutate.GAME_NOTES)
        return "\n\n".join(parts)
    if task != BELT_TASK:
        raise ValueError(f"no prompt for task {task!r}")
    parts = [
        "You build factories in a Factorio-like simulator by calling tools.",
        mutate.task_text(task),
        TOOLS_NOTE_V3,
    ]
    notes = mutate.game_notes(task)
    if game_notes and notes:
        parts.append(notes)
    return "\n\n".join(parts)


def user_prompt(task: str = evaluate.TASK) -> str:
    return PROMPTS[task]
