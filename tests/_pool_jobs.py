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
