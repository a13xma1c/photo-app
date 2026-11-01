"""Layer effects (layer styles): drop shadow, outer glow and stroke.

Effects are stored in ``layer.meta["effects"]`` and rendered live on top of
the layer's (masked) pixels, so they follow every edit and can be changed at
any time.  Dilation / erosion for glow and stroke use a blurred coverage with
an iso-distance threshold - exact for straight edges, pleasantly rounded at
corners and, unlike rank filters, fast on large pictures.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .color import parse_color

EFFECT_NAMES: tuple[str, ...] = ("shadow", "glow", "stroke")
STROKE_POSITIONS: tuple[str, ...] = ("outside", "inside", "center")

DEFAULTS: dict[str, dict[str, Any]] = {
    "shadow": {"enabled": False, "dx": 6, "dy": 6, "blur": 8, "opacity": 60, "color": [0, 0, 0]},
    "glow": {"enabled": False, "size": 10, "opacity": 75, "color": [255, 240, 170]},
    "stroke": {"enabled": False, "size": 3, "opacity": 100, "color": [255, 255, 255], "position": "outside"},
}

_ISO = 0.5 * math.erfc(1 / math.sqrt(2))  # value of a blurred straight edge one sigma outside it


def default_effects() -> dict[str, dict[str, Any]]:
    return deepcopy(DEFAULTS)


def normalize_effects(effects: Mapping[str, Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Fill in defaults, clamp values and drop unknown keys."""
    out = default_effects()
    for name, settings in (effects or {}).items():
        if name not in out:
            continue
        for key, value in settings.items():
            if key not in out[name]:
                continue
            if key == "enabled":
                out[name][key] = bool(value)
            elif key == "color":
                out[name][key] = list(parse_color(value)[:3])
            elif key == "position":
                out[name][key] = value if value in STROKE_POSITIONS else "outside"
            elif key == "opacity":
                out[name][key] = int(min(100, max(0, round(float(value)))))
            elif key in ("dx", "dy"):
                out[name][key] = int(min(500, max(-500, round(float(value)))))
            else:
                out[name][key] = int(min(250, max(0, round(float(value)))))
    return out


def is_empty(effects: Mapping[str, Mapping[str, Any]] | None) -> bool:
    """True when no effect is switched on."""
    return not effects or not any(settings.get("enabled") for settings in effects.values())


def padding(effects: Mapping[str, Mapping[str, Any]] | None, scale: float = 1.0) -> int:
    """How far (in scaled pixels) the effects can reach beyond the layer's pixels."""
    if is_empty(effects):
        return 0
    fx = normalize_effects(effects)
    reach = 0.0
    if fx["shadow"]["enabled"]:
        reach = max(reach, math.hypot(fx["shadow"]["dx"], fx["shadow"]["dy"]) + fx["shadow"]["blur"] * 3)
    if fx["glow"]["enabled"]:
        reach = max(reach, fx["glow"]["size"] * 3)
    if fx["stroke"]["enabled"]:
        reach = max(reach, fx["stroke"]["size"] * 3)
    return int(math.ceil(reach * scale)) + 2


def _coverage(alpha: Image.Image) -> np.ndarray:
    return np.asarray(alpha, dtype=np.float32) / 255.0


def _blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.05:
        return arr
    img = Image.fromarray(np.clip(arr * 255 + 0.5, 0, 255).astype(np.uint8), "L")
    return np.asarray(img.filter(ImageFilter.GaussianBlur(sigma)), dtype=np.float32) / 255.0


def dilate(alpha: Image.Image, distance: float) -> np.ndarray:
    """Coverage grown by *distance* pixels (anti-aliased, rounded corners)."""
    cov = _coverage(alpha)
    if distance <= 0:
        return cov
    hard = (cov > 0.02).astype(np.float32)
    blurred = _blur(hard, distance)
    slope = 0.242 / distance  # d/dx of a blurred edge one sigma out
    grown = np.clip((blurred - _ISO) / max(slope, 1e-6) + 0.5, 0, 1)
    return np.maximum(grown, cov)


def erode(alpha: Image.Image, distance: float) -> np.ndarray:
    """Coverage shrunk by *distance* pixels."""
    cov = _coverage(alpha)
    if distance <= 0:
        return cov
    hard = (cov > 0.98).astype(np.float32)
    blurred = _blur(hard, distance)
    slope = 0.242 / distance
    shrunk = np.clip((blurred - (1 - _ISO)) / max(slope, 1e-6) + 0.5, 0, 1)
    return np.minimum(shrunk, cov)


def _colour_layer(size: tuple[int, int], color: Any, coverage: np.ndarray, opacity: float) -> Image.Image:
    r, g, b, _ = parse_color(color)
    layer = Image.new("RGBA", size, (r, g, b, 0))
    layer.putalpha(Image.fromarray(np.clip(coverage * opacity * 255 + 0.5, 0, 255).astype(np.uint8), "L"))
    return layer


def apply_effects(img: Image.Image, effects: Mapping[str, Mapping[str, Any]] | None, scale: float = 1.0) -> Image.Image:
    """Render *effects* around/over the RGBA *img* (which is drawn at *scale* pixels per document pixel)."""
    if is_empty(effects):
        return img
    fx = normalize_effects(effects)
    img = img if img.mode == "RGBA" else img.convert("RGBA")
    alpha = img.getchannel("A")
    below: list[Image.Image] = []
    if fx["shadow"]["enabled"]:
        s = fx["shadow"]
        shifted = Image.new("L", alpha.size, 0)
        shifted.paste(alpha, (int(round(s["dx"] * scale)), int(round(s["dy"] * scale))))
        cov = _blur(_coverage(shifted), s["blur"] * scale)
        below.append(_colour_layer(img.size, s["color"], cov, s["opacity"] / 100))
    if fx["glow"]["enabled"]:
        g = fx["glow"]
        grown = dilate(alpha, g["size"] * scale * 0.6)
        cov = _blur(grown, g["size"] * scale * 0.5)
        below.append(_colour_layer(img.size, g["color"], cov, g["opacity"] / 100))
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    for layer in below:
        out = Image.alpha_composite(out, layer)
    out = Image.alpha_composite(out, img) if below else img.copy()
    if fx["stroke"]["enabled"]:
        st = fx["stroke"]
        size = max(0.5, st["size"] * scale)
        cov = _coverage(alpha)
        if st["position"] == "outside":
            ring = dilate(alpha, size) - cov
        elif st["position"] == "inside":
            ring = cov - erode(alpha, size)
        else:
            ring = dilate(alpha, size / 2) - erode(alpha, size / 2)
        ring = np.clip(ring, 0, 1)
        out = Image.alpha_composite(out, _colour_layer(img.size, st["color"], ring, st["opacity"] / 100))
    return out
