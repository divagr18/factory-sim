"""Examples, loss masks and the replay mix for sft_lora.py.

Nothing here imports torch, so the tests check the masks with the real Qwen3.5
tokenizer and no model.

Qwen3.5's chat template (checked on Qwen/Qwen3.5-9B, 2026-09-25):
- The generation prompt is `<|im_start|>assistant\\n`, followed by
  `<think>\\n\\n</think>\\n\\n` when `enable_thinking=False`, and by `<think>\\n`
  otherwise.
- The final assistant turn renders as
  `<|im_start|>assistant\\n<think>\\n{reasoning}\\n</think>\\n\\n{content}<|im_end|>\\n`.
  With no reasoning, that is the same empty block the thinking-off generation
  prompt ends with.
- `<think>` and `</think>` are single added tokens, so the block is four
  tokens: `<think>`, `\\n\\n`, `</think>`, `\\n\\n`.
"""

from __future__ import annotations

import json
import random

OPENER = "<|im_start|>assistant\n"
EMPTY_THINK = "<think>\n\n</think>\n\n"
EMPTY_THINK_MODES = ("mask", "train")


def load_examples(path: str) -> list[dict]:
    """Rows as {"messages", "enable_thinking"}, from either format:
    build_sft_data.py's {"prompt", "completion"}, or the {"messages"} rows that
    rationalize.py and make_replay.py write. A row thinks if it says so
    (`enable_thinking`) or if its assistant message carries `reasoning_content`."""
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if "messages" in row:
                messages = row["messages"]
            else:
                messages = list(row["prompt"]) + list(row["completion"])
            if not messages or messages[-1]["role"] != "assistant":
                raise ValueError(f"{path}:{n + 1}: the last message must be the assistant's")
            thinks = row.get("enable_thinking")
            if thinks is None:
                thinks = bool(messages[-1].get("reasoning_content"))
            out.append({"messages": messages, "enable_thinking": bool(thinks)})
    return out


def encode(tok, messages: list[dict], *, enable_thinking: bool = False, empty_think: str = "mask"):
    """Token ids and labels for one example, with the loss on its final assistant turn.

    The prompt is tokenized as vLLM tokenizes it at inference: the template's
    generation prompt on its own, then the reply. With thinking off the
    generation prompt ends with the empty think block, so the model never
    writes it. `mask` keeps those four tokens out of the loss, which is what
    sft_lora.py has always done, because they sat in the prompt. `train` puts
    them in, as a trainer does when it takes the loss over the whole rendered
    assistant turn. With thinking on, the `<think>\\n` opener is always masked;
    the reasoning, `</think>` and the answer are trained.

    Returns {"input_ids", "labels", "think": (start, end)}, where `think` is the
    token span of what follows the assistant opener in the generation prompt."""
    if empty_think not in EMPTY_THINK_MODES:
        raise ValueError(f"empty_think must be one of {EMPTY_THINK_MODES}")
    if messages[-1]["role"] != "assistant":
        raise ValueError("the last message must be the assistant's")
    head = tok.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    full = tok.apply_chat_template(messages, tokenize=False, enable_thinking=enable_thinking)
    if not full.startswith(head):
        raise ValueError(
            "the rendered conversation does not start with the generation prompt; a thinking-off "
            "example must have no reasoning, and a thinking-on one needs enable_thinking"
        )
    cut = head.rfind(OPENER)
    if cut < 0:
        raise ValueError("no assistant opener in the generation prompt")
    cut += len(OPENER)
    pre, think, target = head[:cut], head[cut:], full[len(head) :]
    if not enable_thinking and think != EMPTY_THINK:
        raise ValueError(f"unexpected thinking-off generation prompt tail {think!r}")

    def ids(text: str) -> list[int]:
        return list(tok(text, add_special_tokens=False)["input_ids"])

    ids_pre, ids_think, ids_target = ids(pre), ids(think), ids(target)
    if ids_pre + ids_think != ids(head):
        raise ValueError("the prompt tokenizes differently when split at the assistant opener")
    train_think = empty_think == "train" and not enable_thinking
    labels = (
        [-100] * len(ids_pre)
        + (list(ids_think) if train_think else [-100] * len(ids_think))
        + list(ids_target)
    )
    start = len(ids_pre)
    return {
        "input_ids": ids_pre + ids_think + ids_target,
        "labels": labels,
        "think": (start, start + len(ids_think)),
    }


def replay_per_epoch(n_task: int, n_replay: int, frac: float) -> int:
    """Replay examples per epoch so that they are `frac` of it."""
    if not 0.0 <= frac < 1.0:
        raise ValueError("replay fraction must be in [0, 1)")
    if n_replay == 0 or frac == 0.0:
        return 0
    return round(n_task * frac / (1.0 - frac))


def epoch_plan(
    n_task: int, n_replay: int, frac: float, epochs: int, seed: int = 0
) -> list[list[tuple[str, int]]]:
    """Per epoch, the order of ("task" | "replay", index) pairs.

    Every task example appears once per epoch. Replay is drawn without
    replacement across epochs and reshuffled when the pool runs out, so a
    replay row repeats only once all the others have been used. With no
    replay, the order is the one sft_lora.py always used (`random.Random(seed)`,
    one shuffle per epoch)."""
    per_epoch = replay_per_epoch(n_task, n_replay, frac)
    rng = random.Random(seed)
    pool: list[int] = []
    plan = []
    for _ in range(epochs):
        order = [("task", i) for i in range(n_task)]
        for _ in range(per_epoch):
            if not pool:
                pool = list(range(n_replay))
                rng.shuffle(pool)
            order.append(("replay", pool.pop()))
        rng.shuffle(order)
        plan.append(order)
    return plan
