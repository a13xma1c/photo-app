"""Named adjustment presets ("looks") and user preset persistence."""

from __future__ import annotations

import json
from pathlib import Path

from .adjustments import default_adjustments, normalize_adjustments

PRESETS: dict[str, dict[str, int]] = {
    "None": {},
    "Auto Pop": {"contrast": 12, "vibrance": 25, "sharpen": 15},
    "Vivid": {"contrast": 20, "saturation": 40},
    "Warm": {"temperature": 40, "saturation": 10},
    "Cool": {"temperature": -40, "saturation": 5},
    "Golden Hour": {"temperature": 35, "exposure": 8, "shadows": 20, "vibrance": 20},
    "B&W": {"grayscale": 100, "contrast": 15},
    "Film Noir": {"grayscale": 100, "contrast": 40, "vignette": 55, "shadows": -10},
    "Sepia": {"sepia": 100},
    "Vintage": {"sepia": 45, "contrast": -10, "vignette": 45, "saturation": -20},
    "Fade": {"contrast": -25, "brightness": 10, "saturation": -20},
    "Matte": {"shadows": 35, "contrast": -15, "saturation": -10},
    "Dramatic": {"contrast": 45, "saturation": -15, "brightness": -10},
    "HDR": {"highlights": -40, "shadows": 45, "contrast": 20, "sharpen": 30, "vibrance": 20},
    "Soft Glow": {"blur": 2, "brightness": 8, "contrast": -8},
    "Cross Process": {"hue": 12, "contrast": 25, "saturation": 25, "tint": 20},
}


def preset_adjustments(name: str) -> dict[str, int]:
    """Full adjustment dict for a built-in preset."""
    return {**default_adjustments(), **PRESETS[name]}


class PresetStore:
    """User presets saved as JSON in a file (built-in ones stay read-only)."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.user: dict[str, dict[str, int]] = {}
        if self.path and self.path.exists():
            try:
                data = json.loads(self.path.read_text("utf-8"))
                self.user = {k: normalize_adjustments(v) for k, v in data.items()}
            except (OSError, ValueError):
                self.user = {}

    def names(self) -> list[str]:
        return list(PRESETS) + sorted(self.user)

    def get(self, name: str) -> dict[str, int]:
        if name in PRESETS:
            return preset_adjustments(name)
        return dict(self.user[name])

    def save(self, name: str, adj: dict[str, int]) -> None:
        if name in PRESETS:
            raise ValueError(f"'{name}' is a built-in preset")
        self.user[name] = normalize_adjustments(adj)
        self._flush()

    def delete(self, name: str) -> None:
        if name in PRESETS:
            raise ValueError(f"'{name}' is a built-in preset")
        self.user.pop(name, None)
        self._flush()

    def _flush(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.user, indent=2), "utf-8")
