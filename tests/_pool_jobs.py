"""Jobs for test_pool.py. Spawned workers import them by dotted path."""

import os
import time

STATE = None


def init_state():
    global STATE
    STATE = {"base": 1000, "pid": os.getpid()}


def failing_init():
    raise RuntimeError("no simulator here")


def square(x):
    return x * x


def dispatch(payload):
    """One job that can misbehave in every way, so one pool covers all the cases."""
    op, arg = payload
    if op == "square":
        return arg * arg
    if op == "raise":
        raise ValueError(arg)
    if op == "exit":
        os._exit(arg)
    if op == "sleep":
        time.sleep(arg)
        return "woke"
    if op == "hash":
        return hash(arg)
    if op == "state":
        return STATE["base"] + arg, STATE["pid"]
    if op == "jitter":
        time.sleep((arg * 7919 % 13) / 1000)
        return arg, os.getpid()
    if op == "unpicklable":
        return lambda: None
    raise KeyError(op)


def ledger_hammer(args):
    """Many mixed ledger operations from one process (see test_spend.py)."""
    import os as _os

    from evolve.spend import SpendLedger

    path, n = args
    price = {"input": 0.05, "cached_input": 0.005, "output": 0.25}
    ledger = SpendLedger(path, price, 1_000_000.0)
    key = f"hammer:{_os.getpid()}"
    for _ in range(n):
        ledger.allows(0.01)
        ledger.reserve(key, 0.01)
        _ = ledger.total
        ledger.settle(key, 0.01, {"prompt_tokens": 0, "completion_tokens": 4000})  # $0.001
        ledger.heartbeat(key)
    return n
