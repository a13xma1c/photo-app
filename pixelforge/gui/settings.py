"""Small JSON-backed store for user preferences (recent files, export options)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MAX_RECENT = 8


class Settings:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._data: dict[str, Any] = {}
        if path is not None and path.exists():
            try:
                loaded = json.loads(path.read_text("utf-8"))
                if isinstance(loaded, dict):
                    self._data = loaded
            except (OSError, ValueError):
                self._data = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._flush()

    # recent files -------------------------------------------------------------
    def recent_files(self) -> list[str]:
        return [p for p in self._data.get("recent", []) if isinstance(p, str) and Path(p).exists()]

    def add_recent(self, path: str) -> None:
        resolved = str(Path(path).resolve())
        recent = [resolved] + [p for p in self._data.get("recent", []) if p != resolved]
        self.set("recent", recent[:MAX_RECENT])

    def clear_recent(self) -> None:
        self.set("recent", [])

    def _flush(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, indent=2), "utf-8")
        except OSError:
            pass
