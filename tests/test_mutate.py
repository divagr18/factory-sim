"""Mutation prompts carry what the model needs, and code comes back out of its answer."""

import pytest

from evolve import mutate
from evolve.llm import extract_code

API = "World.move(direction, stride='long') -> bool\n    Walk one decision. (API_SENTINEL)"

PARENT = {
    "code": "def build(world):\n    world.wait()  # PARENT_SENTINEL\n",
    "scores": {
        "train": {"open_patch": 0.9, "cluttered_patch": 0.25},
        "val": {"obstructed_patch": 0.125},
    },
    "traces": {"cluttered_patch": ["move N long -> ok", "place stone-furnace (3,4) S -> refused"]},
}
OTHER = {
    "code": "def build(world):\n    world.move('E')  # OTHER_SENTINEL\n",
    "scores": {"train": {"open_patch": 0.5}},
    "traces": {"offset_patch": ["move E long -> ok"]},
}


def text_of(messages):
    assert [m["role"] for m in messages] == ["system", "user"]
    return messages[0]["content"], messages[1]["content"]


def all_prompts():
    return {
        "fix": mutate.prompt_fix(API, PARENT),
        "rewrite": mutate.prompt_rewrite(API, PARENT, mutate.HINTS[0]),
        "crossover": mutate.prompt_crossover(API, PARENT, OTHER),
        "simplify": mutate.prompt_simplify(API, PARENT),
    }


@pytest.mark.parametrize("op", ["fix", "rewrite", "crossover", "simplify"])
def test_builder_includes_everything(op):
    system, user = text_of(all_prompts()[op])
    assert "API_SENTINEL" in system
    assert "construct_smelting_line" in system and "def build(world):" in system
    assert "at most 5 lines" in system and "```python" in system
    assert "at most 5 lines" in user  # the format is restated next to the parent
    assert "PARENT_SENTINEL" in user
    assert "cluttered_patch" in user and "0.25" in user and "0.12" in user
    assert "obstructed_patch" in user
    assert "place stone-furnace (3,4) S -> refused" in user


def test_operator_specific_text():
    p = all_prompts()
    assert mutate.HINTS[0] in p["rewrite"][1]["content"]
    assert "OTHER_SENTINEL" in p["crossover"][1]["content"]
    assert "Program A" in p["crossover"][1]["content"]
    assert "Program B" in p["crossover"][1]["content"]
    assert "exactly the same" in p["simplify"][1]["content"]
    assert set(mutate.OPERATORS) == {"fix", "rewrite", "crossover", "simplify"}


def test_missing_parts_are_omitted():
    _, user = text_of(mutate.prompt_fix(API, {"code": "def build(world):\n    pass\n"}))
    assert "Success rate" not in user and "Last actions" not in user
    _, user = text_of(mutate.prompt_fix(API, {"code": "def build(world): pass", "scores": {}}))
    assert "Success rate" not in user


def test_system_prompt_states_rules_and_notes():
    system = mutate.system_prompt(API)
    for word in ("2 burner mining drills", "2 stone furnaces", "60 coal", "10", "walls"):
        assert word in system
    for banned in ("import", "lambda", "getattr", "try", "300 lines"):
        assert banned in system
    assert "sorted" in system and "never import it" in system
    assert mutate.GAME_NOTES in system
    for fact in ("30 ticks", "600-decision", "Chebyshev", "(x+1, y+1)", "(cx+0.5, cy+1.3)"):
        assert fact in mutate.GAME_NOTES


def test_game_notes_can_be_toggled(monkeypatch):
    monkeypatch.setattr(mutate, "GAME_NOTES", "")
    system = mutate.system_prompt(API)
    assert "Chebyshev" not in system and "API_SENTINEL" in system


def test_hints():
    assert 10 <= len(mutate.HINTS) <= 15
    assert len(set(mutate.HINTS)) == len(mutate.HINTS)


def test_prompt_hash_stable():
    a = mutate.prompt_fix(API, PARENT)
    b = mutate.prompt_fix(API, dict(PARENT))
    assert mutate.prompt_hash(a) == mutate.prompt_hash(b)
    assert len(mutate.prompt_hash(a)) == 64
    reordered = [{"content": m["content"], "role": m["role"]} for m in a]
    assert mutate.prompt_hash(reordered) == mutate.prompt_hash(a)
    assert mutate.prompt_hash(mutate.prompt_simplify(API, PARENT)) != mutate.prompt_hash(a)


def test_extract_code_last_build_block():
    text = (
        "Plan: walk, place.\n"
        "```python\ndef build(world):\n    return 1\n```\n"
        "Helper, not a program:\n```python\ndef helper():\n    pass\n```\n"
        "Final:\n```\ndef build(world):\n    return 2\n```\n"
        '```json\n{"def build(": 1}\n```\n'
    )
    assert extract_code(text) == "def build(world):\n    return 2\n"


def test_extract_code_none_and_reasoning():
    assert extract_code("") is None
    assert extract_code(None) is None
    assert extract_code("```python\ndef helper():\n    pass\n```") is None
    assert extract_code("no code at all") is None
    text = (
        "\ufeff  <think>Maybe:\n```python\ndef build(world):\n    draft()\n```\n"
        "Hmm, better idea.</think>\n"
        "Plan: one line.\n```python\r\n\ufeffdef build(world):\r\n    final()\r\n```"
    )
    assert extract_code(text) == "def build(world):\n    final()\n"
    # Reasoning drafts without a </think> marker: the answer still comes last.
    text = "Let me think... ```python\ndef build(world):\n    draft()\n```\nAnswer:\n" + (
        "```python\n    def build(world):\n        final()\n```"
    )
    assert extract_code(text) == "def build(world):\n    final()\n"
