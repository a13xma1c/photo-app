"""A background worker for expensive picture operations (compositing, filter previews).

Jobs are keyed: submitting a new job with a key that is still *queued* replaces
the old one ("latest request wins"), so dragging a slider never piles up
work.  Results are handed back on the GUI thread through a thread-safe
``deliver`` callable supplied by the shell, which keeps the widgets oblivious
to threads.  ``Worker(sync=True)`` runs everything inline - handy for tests.
"""

from __future__ import annotations

import threading
import traceback
from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from typing import Any

Job = tuple[Callable[[], Any], Callable[[Any], None] | None]


class Worker:
    def __init__(self, deliver: Callable[[Callable[[], None]], None], sync: bool = False) -> None:
        self._deliver = deliver
        self.sync = sync
        self._queue: OrderedDict[str, Job] = OrderedDict()
        self._cv = threading.Condition()
        self._running: str | None = None
        self._stopped = False
        self._thread: threading.Thread | None = None
        self.errors: list[str] = []

    # ------------------------------------------------------------------ api --
    def submit(self, key: str, fn: Callable[[], Any], on_done: Callable[[Any], None] | None = None) -> None:
        """Run ``fn()`` in the background and call ``on_done(result)`` on the GUI thread (unless superseded)."""
        if self.sync:
            try:
                result = fn()
            except Exception:  # noqa: BLE001 - report like the threaded path does
                self.errors.append(traceback.format_exc())
                return
            if on_done is not None:
                on_done(result)
            return
        with self._cv:
            if self._stopped:
                return
            self._queue[key] = (fn, on_done)  # replaces a queued job with the same key
            self._queue.move_to_end(key)
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="pixelforge-worker", daemon=True)
                self._thread.start()
            self._cv.notify()

    def cancel(self, key: str) -> None:
        with self._cv:
            self._queue.pop(key, None)

    @property
    def busy(self) -> bool:
        with self._cv:
            return bool(self._queue) or self._running is not None

    def pending(self, key: str) -> bool:
        with self._cv:
            return key in self._queue or self._running == key

    def wait(self, timeout: float = 5.0) -> bool:
        """Block until no job is queued or running (for tests); True when idle."""
        with self._cv:
            return self._cv.wait_for(lambda: not self._queue and self._running is None, timeout)

    def stop(self) -> None:
        with self._cv:
            self._stopped = True
            self._queue.clear()
            self._cv.notify_all()

    # --------------------------------------------------------------- thread --
    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._queue and not self._stopped:
                    self._cv.wait()
                if self._stopped:
                    self._running = None
                    self._cv.notify_all()
                    return
                key, (fn, on_done) = self._queue.popitem(last=False)
                self._running = key
            try:
                result = fn()
            except Exception:  # noqa: BLE001 - a failing job must not kill the worker
                self.errors.append(traceback.format_exc())
                result, on_done = None, None
            with self._cv:
                superseded = key in self._queue  # a newer request for the same key arrived meanwhile
                self._running = None
                self._cv.notify_all()
            if on_done is not None and not superseded and not self._stopped:
                self._deliver(partial(on_done, result))
