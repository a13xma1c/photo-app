"""Brush presets: tip shapes, dynamics and the preset library.

A :class:`BrushPreset` describes how a stroke is built from dabs: the tip
shape (round, square, chalk, spatter or a custom picture), its size,
hardness, angle and roundness, the spacing between dabs, and dynamics -
scatter, size and opacity jitter, flow (how much one dab adds; low flow
builds up like an airbrush) and the number of dabs per step.  Presets are
plain data, so user presets - including custom tips defined from a picture -
live in a JSON file next to the adjustment presets.
"""

from __future__ import annotations

import base64
import io
import json
import math
import random
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

TIPS: tuple[str, ...] = ("round", "square", "chalk", "spatter", "custom")
MAX_SIZE = 500


@dataclass(frozen=True)
class BrushPreset:
    name: str = "Brush"
    tip: str = "round"
    size: int = 24
    hardness: int = 80  # % - 100 is a crisp edge, 0 fully soft
    spacing: int = 15  # % of the size between dabs
    angle: float = 0.0  # degrees, rotates the tip
    roundness: int = 100  # % - 100 is circular, smaller squashes the tip
    scatter: int = 0  # % of the size the dabs wander off the path
    size_jitter: int = 0  # % random size variation per dab
    opacity_jitter: int = 0  # % random opacity variation per dab
    flow: int = 100  # % - what one dab adds; overlapping dabs build up
    count: int = 1  # dabs per spacing step
    texture: str | None = None  # base64 PNG of a custom tip (mode L)
    builtin: bool = field(default=False, compare=False)

    def normalized(self) -> BrushPreset:
        return replace(
            self,
            tip=self.tip if self.tip in TIPS else "round",
            size=int(min(MAX_SIZE, max(1, self.size))),
            hardness=int(min(100, max(0, self.hardness))),
            spacing=int(min(1000, max(1, self.spacing))),
            angle=float(self.angle) % 360,
            roundness=int(min(100, max(5, self.roundness))),
            scatter=int(min(1000, max(0, self.scatter))),
            size_jitter=int(min(100, max(0, self.size_jitter))),
            opacity_jitter=int(min(100, max(0, self.opacity_jitter))),
            flow=int(min(100, max(1, self.flow))),
            count=int(min(16, max(1, self.count))),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("builtin", None)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any], builtin: bool = False) -> BrushPreset:
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "builtin"}
        return cls(builtin=builtin, **fields).normalized()

    # ------------------------------------------------------------------ tips --
    def tip_image(self) -> Image.Image | None:
        """The custom tip as an ``L`` image (None for built-in shapes or unreadable data)."""
        if self.tip != "custom" or not self.texture:
            return None
        try:
            img = Image.open(io.BytesIO(base64.b64decode(self.texture)))
            img.load()
            return img.convert("L")
        except Exception:  # noqa: BLE001 - a broken texture falls back to a round tip
            return None

    def stamp(self, size: int | None = None, angle: float | None = None) -> np.ndarray:
        """The dab as a float32 coverage array for *size* pixels (defaults to the preset's size and angle)."""
        size = self.size if size is None else size
        angle = self.angle if angle is None else angle
        return _stamp(self.tip, max(1, int(round(size))), self.hardness / 100, round(angle % 360, 1), self.roundness / 100, self.texture)

    def description(self) -> str:
        bits = [f"{self.size} px", f"{self.hardness} %"]
        if self.tip != "round":
            bits.insert(0, self.tip)
        if self.flow < 100:
            bits.append(f"flow {self.flow} %")
        if self.scatter:
            bits.append("scatter")
        return " · ".join(bits)


def _base_shape(tip: str, diameter: int, hardness: float, texture: str | None) -> np.ndarray:
    """Axis-aligned dab of *diameter* pixels before roundness / rotation."""
    if diameter == 1:
        return np.ones((1, 1), np.float32)
    radius = diameter / 2
    coords = (np.arange(diameter, dtype=np.float32) + 0.5) - radius
    xs, ys = coords[None, :], coords[:, None]
    inner = hardness * 0.999

    def falloff(dist: np.ndarray) -> np.ndarray:
        t = np.clip((dist - inner) / max(1 - inner, 1e-6), 0.0, 1.0)
        return (1 - t * t * (3 - 2 * t)).astype(np.float32)

    if tip == "square":
        dist = np.maximum(np.abs(xs), np.abs(ys)) / radius
        return falloff(dist)
    if tip == "chalk":
        rng = np.random.default_rng(12345)
        grain = rng.random((diameter, diameter)).astype(np.float32)
        grain = (grain * 0.7 + 0.3) * (rng.random((diameter, diameter)) > 0.25)
        dist = np.sqrt(xs * xs + ys * ys) / radius
        return (falloff(dist) * grain).astype(np.float32)
    if tip == "spatter":
        rng = np.random.default_rng(777)
        out = np.zeros((diameter, diameter), np.float32)
        dots = max(6, diameter // 2)
        for _ in range(dots):
            a, r = rng.random() * 2 * math.pi, math.sqrt(rng.random()) * radius * 0.9
            cx, cy = radius + math.cos(a) * r, radius + math.sin(a) * r
            dr = max(0.6, radius * (0.04 + 0.12 * rng.random()))
            dist = np.sqrt((xs + radius - cx) ** 2 + (ys + radius - cy) ** 2)
            np.maximum(out, np.clip(dr + 0.5 - dist, 0, 1), out=out)
        return out
    if tip == "custom" and texture:
        try:
            img = Image.open(io.BytesIO(base64.b64decode(texture))).convert("L")
            img = img.resize((diameter, diameter), Image.Resampling.BILINEAR)
            return np.asarray(img, dtype=np.float32) / 255.0
        except Exception:  # noqa: BLE001
            pass
    dist = np.sqrt(xs * xs + ys * ys) / radius
    return falloff(dist)


@lru_cache(maxsize=256)
def _stamp(tip: str, diameter: int, hardness: float, angle: float, roundness: float, texture: str | None) -> np.ndarray:
    base = _base_shape(tip, diameter, hardness, texture)
    if roundness >= 0.999 and (angle % 360 < 1e-6 or tip in ("round", "chalk", "spatter") and roundness >= 0.999):
        return base
    img = Image.fromarray(np.clip(base * 255 + 0.5, 0, 255).astype(np.uint8), "L")
    if roundness < 0.999:
        squashed_h = max(1, int(round(diameter * roundness)))
        img = img.resize((diameter, squashed_h), Image.Resampling.BILINEAR)
        canvas = Image.new("L", (diameter, diameter), 0)
        canvas.paste(img, (0, (diameter - squashed_h) // 2))
        img = canvas
    if angle % 360 > 1e-6:
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)
    return np.asarray(img, dtype=np.float32) / 255.0


def texture_from_image(image: Image.Image, size: int = 96) -> str:
    """Encode a picture as a custom tip: its alpha × luminance, fitted into *size* × *size*, as base64 PNG."""
    rgba = image.convert("RGBA")
    bbox = rgba.getchannel("A").getbbox()
    if bbox:
        rgba = rgba.crop(bbox)
    lum = rgba.convert("L")
    alpha = rgba.getchannel("A")
    tip = Image.fromarray((np.asarray(lum, dtype=np.float32) * np.asarray(alpha, dtype=np.float32) / 255).astype(np.uint8), "L")
    scale = min(size / max(1, tip.width), size / max(1, tip.height))
    tip = tip.resize((max(1, round(tip.width * scale)), max(1, round(tip.height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(tip, ((size - tip.width) // 2, (size - tip.height) // 2))
    buf = io.BytesIO()
    canvas.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


BUILTIN_PRESETS: tuple[BrushPreset, ...] = tuple(
    p.normalized()
    for p in (
        BrushPreset("Hard Round", "round", 24, 100, 12, builtin=True),
        BrushPreset("Soft Round", "round", 40, 0, 12, builtin=True),
        BrushPreset("Airbrush", "round", 80, 0, 8, flow=12, builtin=True),
        BrushPreset("Pencil", "round", 3, 100, 10, builtin=True),
        BrushPreset("Marker", "round", 18, 90, 10, roundness=45, angle=45, builtin=True),
        BrushPreset("Calligraphy", "round", 22, 100, 6, roundness=20, angle=35, builtin=True),
        BrushPreset("Square", "square", 20, 100, 14, builtin=True),
        BrushPreset("Chalk", "chalk", 36, 60, 18, opacity_jitter=30, builtin=True),
        BrushPreset("Spatter", "spatter", 60, 100, 40, scatter=60, size_jitter=50, opacity_jitter=40, count=2, builtin=True),
        BrushPreset("Watercolor", "round", 60, 20, 25, flow=25, opacity_jitter=35, scatter=10, builtin=True),
        BrushPreset("Stipple", "round", 6, 100, 120, scatter=150, size_jitter=60, count=3, builtin=True),
    )
)


class BrushLibrary:
    """Built-in brush presets plus the user's own, saved as JSON."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.user: dict[str, BrushPreset] = {}
        if self.path and self.path.exists():
            try:
                data = json.loads(self.path.read_text("utf-8"))
                self.user = {name: BrushPreset.from_dict({**entry, "name": name}) for name, entry in data.items()}
            except (OSError, ValueError, TypeError):
                self.user = {}

    def names(self) -> list[str]:
        return [p.name for p in BUILTIN_PRESETS] + sorted(self.user)

    def presets(self) -> list[BrushPreset]:
        return list(BUILTIN_PRESETS) + [self.user[n] for n in sorted(self.user)]

    def get(self, name: str) -> BrushPreset:
        for preset in BUILTIN_PRESETS:
            if preset.name == name:
                return preset
        return self.user[name]

    def is_builtin(self, name: str) -> bool:
        return any(p.name == name for p in BUILTIN_PRESETS)

    def save(self, preset: BrushPreset) -> BrushPreset:
        if self.is_builtin(preset.name):
            raise ValueError(f"'{preset.name}' is a built-in brush")
        saved = replace(preset.normalized(), builtin=False)
        self.user[saved.name] = saved
        self._flush()
        return saved

    def delete(self, name: str) -> None:
        if self.is_builtin(name):
            raise ValueError(f"'{name}' is a built-in brush")
        self.user.pop(name, None)
        self._flush()

    def _flush(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {name: {k: v for k, v in p.to_dict().items() if k != "name"} for name, p in self.user.items()}
            self.path.write_text(json.dumps(data, indent=2), "utf-8")


def stroke_preview(preset: BrushPreset, size: tuple[int, int] = (160, 40), seed: int = 1) -> np.ndarray:
    """Coverage of a sample S-curve stroke with *preset* scaled to fit *size* (for thumbnails)."""
    from .painting import Stroke  # local import: painting imports this module

    w, h = size
    scale = min(1.0, (h * 0.6) / max(1, preset.size))
    small = replace(preset, size=max(1, int(round(preset.size * scale))))
    stroke = Stroke(size, preset=small, seed=seed)
    for i in range(41):
        t = i / 40
        x = 8 + t * (w - 16)
        y = h / 2 + math.sin(t * math.pi * 2) * h * 0.22
        stroke.add_point(x, y)
    return stroke.coverage


def seeded_random(seed: int | None) -> random.Random:
    return random.Random(seed)
