"""The background worker: ordering, latest-wins replacement, error isolation and the synchronous mode."""

import threading
import time
from collections.abc import Callable

from pixelforge.gui.worker import Worker


class Inbox:
    """Collects GUI-thread callbacks the way a shell's ``call_soon`` would, to be run explicitly."""

    def __init__(self) -> None:
        self.items: list[Callable[[], None]] = []
        self.lock = threading.Lock()

    def deliver(self, fn: Callable[[], None]) -> None:
        with self.lock:
            self.items.append(fn)

    def run_all(self) -> int:
        with self.lock:
            items, self.items = self.items, []
        for fn in items:
            fn()
        return len(items)


def test_jobs_run_and_deliver_results() -> None:
    inbox = Inbox()
    worker = Worker(inbox.deliver)
    results: list[int] = []
    worker.submit("a", lambda: 1 + 1, results.append)
    worker.submit("b", lambda: 40 + 2, results.append)
    assert worker.wait(5)
    assert not worker.busy
    inbox.run_all()
    assert sorted(results) == [2, 42]
    worker.stop()
    worker.submit("late", lambda: 99, results.append)  # ignored after stop
    assert not worker.pending("late")


def test_latest_request_wins_per_key() -> None:
    inbox = Inbox()
    worker = Worker(inbox.deliver)
    gate = threading.Event()
    seen: list[str] = []

    def slow() -> str:
        gate.wait(5)
        return "first"

    worker.submit("frame", slow, seen.append)
    time.sleep(0.05)  # the first job is running now
    assert worker.pending("frame") and worker.busy
    worker.submit("frame", lambda: "second", seen.append)
    worker.submit("frame", lambda: "third", seen.append)  # replaces "second" while queued
    gate.set()
    assert worker.wait(5)
    inbox.run_all()
    assert seen == ["third"]  # the superseded first result was dropped, "second" never ran
    worker.submit("x", lambda: "x", seen.append)
    worker.cancel("x") if worker.pending("x") else None
    worker.wait(5)
    inbox.run_all()
    worker.stop()


def test_errors_are_isolated_and_sync_mode_runs_inline() -> None:
    inbox = Inbox()
    worker = Worker(inbox.deliver)
    out: list[object] = []

    def boom() -> None:
        raise RuntimeError("kaboom")

    worker.submit("bad", boom, out.append)
    worker.submit("good", lambda: "ok", out.append)
    assert worker.wait(5)
    inbox.run_all()
    assert out == ["ok"] and worker.errors and "kaboom" in worker.errors[0]
    worker.stop()
    sync = Worker(inbox.deliver, sync=True)
    sync.submit("s", lambda: 7, out.append)
    assert out[-1] == 7 and not inbox.items  # delivered inline, nothing queued
    sync.submit("bad", boom, out.append)
    assert len(sync.errors) == 1
