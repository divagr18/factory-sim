"""A persistent process pool that survives the jobs it runs.

Candidate programs can hang, allocate without bound, or take the simulator down
with them, and `multiprocessing.Pool` cannot kill one stuck task. So each worker
here is a plain spawned `Process` with its own pipe, the parent hands it one
payload at a time, and the parent keeps the clock. A job that overruns its
timeout gets its worker terminated; a worker that dies is noticed through its
process sentinel. Either way the payload's result becomes `{"error": ...}` and a
fresh worker, with a freshly run initializer, takes the slot.

A private pipe per worker, rather than one shared result queue, means killing a
worker can never leave a shared lock held or a shared pipe half written.

`initializer` and `job` are `"module.path:function"` strings resolved inside the
worker, so nothing unpicklable (a cffi simulator, say) crosses a process
boundary. The initializer runs once per worker process and typically builds
that worker's environment into module state that the job reads.

Workers start with `PYTHONHASHSEED=0`, so anything that iterates a set of
strings behaves the same in every worker and every run.
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import time
from collections import deque
from multiprocessing.connection import wait
from multiprocessing.reduction import ForkingPickler

_CTX = mp.get_context("spawn")
#: A slot whose worker dies this many times in a row before it is ready
#: breaks the pool, instead of respawning forever.
MAX_START_FAILURES = 3


def _resolve(path: str):
    module, _, name = path.partition(":")
    if not module or not name:
        raise ValueError(f"expected 'module.path:function', got {path!r}")
    obj = importlib.import_module(module)
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def _describe(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


def _worker_main(conn, initializer: str | None, job: str) -> None:
    init_error = None
    fn = None
    try:
        if initializer:
            _resolve(initializer)()
        fn = _resolve(job)
    except BaseException as e:  # reported per job, so the pool does not respawn in a loop
        init_error = _describe(e)
    conn.send(("ready", init_error))
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        idx, payload = msg
        if init_error is not None:
            result = {"error": f"initializer failed: {init_error}"}
        else:
            try:
                result = fn(payload)
            except KeyboardInterrupt:
                return
            except BaseException as e:
                result = {"error": _describe(e)}
        try:
            conn.send(("result", idx, result))
        except (EOFError, OSError):
            return
        except Exception as e:  # the result did not pickle
            conn.send(("result", idx, {"error": f"result not picklable: {_describe(e)}"}))


class _Slot:
    __slots__ = ("proc", "conn", "ready", "job", "deadline", "failures")

    def __init__(self):
        self.proc = self.conn = None
        self.ready = False
        self.job = None  # (index, payload) while busy
        self.deadline = 0.0
        self.failures = 0


class EvalPool:
    """`workers` processes that each run `initializer()` once, then `job(payload)`."""

    def __init__(
        self,
        workers: int,
        initializer: str | None,
        job: str,
        timeout_s: float = 60.0,
        start_timeout_s: float = 300.0,
    ):
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self.workers = workers
        self.initializer = initializer
        self.job = job
        self.timeout_s = float(timeout_s)
        self.start_timeout_s = float(start_timeout_s)
        self._closed = False
        self._slots = [_Slot() for _ in range(workers)]
        try:
            for slot in self._slots:
                self._spawn(slot)
        except BaseException:
            self.close()
            raise

    # --- process management ---

    def _spawn(self, slot: _Slot) -> None:
        parent, child = _CTX.Pipe(duplex=True)
        proc = _CTX.Process(
            target=_worker_main, args=(child, self.initializer, self.job), daemon=True
        )
        # The seed must be in the environment at interpreter start, so it is set
        # around start() and the parent's own value is put back.
        old = os.environ.get("PYTHONHASHSEED")
        os.environ["PYTHONHASHSEED"] = "0"
        try:
            proc.start()
        finally:
            if old is None:
                del os.environ["PYTHONHASHSEED"]
            else:
                os.environ["PYTHONHASHSEED"] = old
        child.close()  # so the parent sees EOF when the worker dies
        slot.proc, slot.conn = proc, parent
        slot.ready, slot.job = False, None
        slot.deadline = time.monotonic() + self.start_timeout_s

    def _kill(self, slot: _Slot) -> None:
        proc, conn = slot.proc, slot.conn
        slot.proc = slot.conn = None
        if conn is not None:
            conn.close()
        if proc is not None:
            if proc.is_alive():
                proc.terminate()
            proc.join(5)
            if proc.is_alive():
                proc.kill()
                proc.join(5)
            proc.close()

    def _replace(self, slot: _Slot, results: list, error: str) -> None:
        """Record `error` for the slot's job (if any) and put a fresh worker in its place."""
        if slot.job is not None:
            results[slot.job[0]] = {"error": error}
        elif not slot.ready:
            slot.failures += 1
            if slot.failures >= MAX_START_FAILURES:
                self._kill(slot)
                raise RuntimeError(f"worker failed to start {slot.failures} times: {error}")
        self._kill(slot)
        self._spawn(slot)

    # --- public API ---

    def map(self, payloads: list) -> list:
        """One result per payload, in payload order, using every worker at once."""
        if self._closed:
            raise RuntimeError("pool is closed")
        payloads = list(payloads)
        results: list = [None] * len(payloads)
        todo = deque(enumerate(payloads))
        try:
            self._run(todo, results)
        except BaseException:
            # An interrupted map leaves jobs in flight; the pool is not reusable.
            self.close()
            raise
        return results

    def _run(self, todo: deque, results: list) -> None:
        slots = self._slots
        while todo or any(s.job is not None for s in slots):
            for slot in slots:
                while slot.ready and slot.job is None and todo:
                    item = todo.popleft()
                    try:
                        data = ForkingPickler.dumps(item)
                    except Exception as e:
                        results[item[0]] = {"error": f"payload not picklable: {_describe(e)}"}
                        continue
                    try:
                        slot.conn.send_bytes(data)
                    except OSError:
                        todo.appendleft(item)  # it never ran; retry on a fresh worker
                        self._replace(slot, results, "worker crashed before the job")
                        break
                    slot.job = item
                    slot.deadline = time.monotonic() + self.timeout_s

            watched = [s for s in slots if s.job is not None or not s.ready]
            if not watched:
                continue
            now = time.monotonic()
            timeout = max(0.0, min(s.deadline for s in watched) - now)
            handles = {}
            for s in watched:
                handles[s.conn] = s
                handles[s.proc.sentinel] = s
            fired = {id(handles[h]): handles[h] for h in wait(list(handles), timeout)}
            for slot in fired.values():
                self._service(slot, results)

            now = time.monotonic()
            for slot in slots:
                if (slot.job is not None or not slot.ready) and now >= slot.deadline:
                    if slot.job is not None:
                        msg = f"timeout after {self.timeout_s:g}s"
                    else:
                        msg = f"worker did not start within {self.start_timeout_s:g}s"
                    self._replace(slot, results, msg)

    def _service(self, slot: _Slot, results: list) -> None:
        try:
            while slot.conn.poll():
                msg = slot.conn.recv()
                if msg[0] == "ready":
                    slot.ready, slot.failures = True, 0
                    slot.deadline = 0.0
                else:
                    _, idx, result = msg
                    results[idx] = result
                    slot.job = None
            if slot.proc.is_alive():
                return
            error = f"worker crashed: exit code {slot.proc.exitcode}"
        except (EOFError, OSError) as e:
            slot.proc.join(1)
            error = f"worker crashed: exit code {slot.proc.exitcode} ({type(e).__name__})"
        self._replace(slot, results, error)

    def close(self) -> None:
        """Stop every worker. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        for slot in self._slots:
            if slot.conn is not None and slot.job is None:
                try:
                    slot.conn.send(None)
                except (OSError, EOFError):
                    pass
        end = time.monotonic() + 2.0
        for slot in self._slots:
            if slot.proc is not None and slot.job is None:
                slot.proc.join(max(0.0, end - time.monotonic()))
        for slot in self._slots:
            slot.job = None
            self._kill(slot)

    def __enter__(self) -> EvalPool:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
