"""Generic, capacity-bounded undo/redo history with named states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")

DEFAULT_CAPACITY = 40


@dataclass(frozen=True)
class HistoryState(Generic[T]):
    label: str
    snapshot: T


class History(Generic[T]):
    """A linear list of snapshots plus a cursor.

    ``push`` discards any redo states, appends the new one and trims the oldest
    states when the capacity is exceeded.  ``undo``/``redo``/``go_to`` move the
    cursor and return the snapshot that should become current.
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._states: list[HistoryState[T]] = []
        self._index = -1

    # ------------------------------------------------------------ queries --
    def __len__(self) -> int:
        return len(self._states)

    @property
    def index(self) -> int:
        return self._index

    @property
    def labels(self) -> list[str]:
        return [s.label for s in self._states]

    @property
    def states(self) -> list[HistoryState[T]]:
        return list(self._states)

    @property
    def current(self) -> T | None:
        return self._states[self._index].snapshot if self._index >= 0 else None

    @property
    def can_undo(self) -> bool:
        return self._index > 0

    @property
    def can_redo(self) -> bool:
        return self._index < len(self._states) - 1

    # ----------------------------------------------------------- mutation --
    def clear(self) -> None:
        self._states.clear()
        self._index = -1

    def push(self, label: str, snapshot: T) -> None:
        del self._states[self._index + 1 :]
        self._states.append(HistoryState(label, snapshot))
        overflow = len(self._states) - max(1, self.capacity)
        if overflow > 0:
            del self._states[:overflow]
        self._index = len(self._states) - 1

    def at(self, index: int) -> T | None:
        """The snapshot at *index* (None when out of range)."""
        return self._states[index].snapshot if 0 <= index < len(self._states) else None

    def drop_oldest(self, count: int = 1) -> None:
        """Forget the *count* oldest states (never the current one)."""
        count = max(0, min(count, self._index))
        if count:
            del self._states[:count]
            self._index -= count

    def replace_current(self, label: str, snapshot: T) -> None:
        """Overwrite the current state (used to coalesce rapid edits)."""
        if self._index < 0:
            self.push(label, snapshot)
        else:
            self._states[self._index] = HistoryState(label, snapshot)

    def undo(self) -> T | None:
        if not self.can_undo:
            return None
        self._index -= 1
        return self.current

    def redo(self) -> T | None:
        if not self.can_redo:
            return None
        self._index += 1
        return self.current

    def go_to(self, index: int) -> T | None:
        if not 0 <= index < len(self._states):
            raise IndexError(index)
        self._index = index
        return self.current
