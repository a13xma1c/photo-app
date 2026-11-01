"""Curves and Levels: tonal remapping through lookup tables.

A *curve* is a list of ``(input, output)`` control points in 0..255.  The
smooth curve through them is a monotone cubic (Fritsch–Carlson) so it never
overshoots or creates reversals between points - exactly what photo editors
do.  ``apply_curves`` takes one curve per channel (``rgb`` for all three plus
optional ``r``, ``g``, ``b``) and applies the combined table to the RGB
channels, leaving alpha untouched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from PIL import Image

Point = tuple[float, float]
CHANNELS: tuple[str, ...] = ("rgb", "r", "g", "b")
IDENTITY_POINTS: tuple[Point, ...] = ((0.0, 0.0), (255.0, 255.0))
Curves = dict[str, list[Point]]


def default_curves() -> Curves:
    return {channel: list(IDENTITY_POINTS) for channel in CHANNELS}


def normalize_points(points: Sequence[Sequence[float]]) -> list[Point]:
    """Sorted, clamped control points with distinct inputs (later duplicates win)."""
    by_x: dict[float, float] = {}
    for x, y in points:
        by_x[min(255.0, max(0.0, float(x)))] = min(255.0, max(0.0, float(y)))
    return sorted(by_x.items())


def is_identity_curve(points: Sequence[Sequence[float]]) -> bool:
    pts = normalize_points(points)
    return all(abs(x - y) < 1e-9 for x, y in pts) and (len(pts) < 2 or (pts[0][0] == 0 and pts[-1][0] == 255))


def is_identity(curves: Mapping[str, Sequence[Sequence[float]]] | None) -> bool:
    return curves is None or all(is_identity_curve(points) for points in curves.values())


def _monotone_slopes(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Fritsch–Carlson tangents: the interpolant stays monotone wherever the data is."""
    n = len(xs)
    dx = np.diff(xs)
    delta = np.diff(ys) / dx
    m = np.empty(n)
    m[0], m[-1] = delta[0], delta[-1]
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0 or min(abs(delta[i - 1]), abs(delta[i])) < 1e-9:
            m[i] = 0.0  # extremum or a flat piece: a horizontal tangent keeps the curve monotone
        else:
            w1, w2 = 2 * dx[i] + dx[i - 1], dx[i] + 2 * dx[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
    return m


def curve_values(points: Sequence[Sequence[float]], xs: np.ndarray) -> np.ndarray:
    """Evaluate the smooth curve through *points* at *xs* (flat beyond the end points)."""
    pts = normalize_points(points)
    if not pts:
        return xs.astype(np.float64)
    if len(pts) == 1:
        return np.full_like(xs, pts[0][1], dtype=np.float64)
    px_ = np.array([p[0] for p in pts])
    py = np.array([p[1] for p in pts])
    m = _monotone_slopes(px_, py)
    idx = np.clip(np.searchsorted(px_, xs, side="right") - 1, 0, len(pts) - 2)
    x0, x1 = px_[idx], px_[idx + 1]
    h = x1 - x0
    t = np.clip((xs - x0) / h, 0, 1)
    t2, t3 = t * t, t * t * t
    h00, h10, h01, h11 = 2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + t, -2 * t3 + 3 * t2, t3 - t2
    y = h00 * py[idx] + h10 * h * m[idx] + h01 * py[idx + 1] + h11 * h * m[idx + 1]
    y = np.where(xs < px_[0], py[0], np.where(xs > px_[-1], py[-1], y))
    return np.clip(y, 0, 255)


def curve_lut(points: Sequence[Sequence[float]]) -> list[int]:
    """256-entry lookup table for the curve through *points*."""
    # floor(v + ½) rather than round(): monotone values must give a monotone table (no banker's rounding flips)
    return [int(np.floor(v + 0.5 + 1e-6)) for v in curve_values(points, np.arange(256, dtype=np.float64))]


def channel_luts(curves: Mapping[str, Sequence[Sequence[float]]]) -> tuple[list[int], list[int], list[int]]:
    """Per-channel tables with the master ``rgb`` curve applied first."""
    master = curve_lut(curves.get("rgb", IDENTITY_POINTS))
    out = []
    for channel in ("r", "g", "b"):
        own = curve_lut(curves.get(channel, IDENTITY_POINTS))
        out.append([own[master[i]] for i in range(256)])
    return out[0], out[1], out[2]


def apply_curves(img: Image.Image, curves: Mapping[str, Sequence[Sequence[float]]]) -> Image.Image:
    """Apply *curves* to the RGB channels of *img* (alpha is preserved)."""
    if is_identity(curves):
        return img
    lut_r, lut_g, lut_b = channel_luts(curves)
    if img.mode == "RGBA":
        r, g, b, a = img.split()
        return Image.merge("RGBA", (r.point(lut_r), g.point(lut_g), b.point(lut_b), a))
    rgb = img.convert("RGB")
    r, g, b = rgb.split()
    return Image.merge("RGB", (r.point(lut_r), g.point(lut_g), b.point(lut_b)))


def levels_points(in_black: int, in_white: int, gamma: float, out_black: int, out_white: int) -> list[Point]:
    """Levels expressed as curve points: black/white clip plus the midpoint moved by *gamma*."""
    in_white = max(in_white, in_black + 1)
    mid_in = (in_black + in_white) / 2
    mid_out = out_black + (0.5 ** (1.0 / max(gamma, 1e-3))) * (out_white - out_black)
    return [(float(in_black), float(out_black)), (mid_in, mid_out), (float(in_white), float(out_white))]


def histogram(img: Image.Image, bins: int = 256) -> np.ndarray:
    """Per-channel RGB histogram of the opaque pixels, normalised so the 98th percentile is 1 (shape 3×256)."""
    arr = np.asarray(img.convert("RGBA"))
    alpha = arr[..., 3] > 0
    if not alpha.any():
        return np.zeros((3, bins), np.float32)
    data = np.stack([np.bincount(arr[..., c][alpha].ravel(), minlength=bins)[:bins] for c in range(3)]).astype(np.float32)
    nonzero = data[data > 0]
    peak = float(np.percentile(nonzero, 98)) if nonzero.size else 0.0
    return np.clip(data / peak, 0, 1) if peak > 0 else data


def serializable(curves: Mapping[str, Sequence[Sequence[float]]]) -> dict[str, list[list[float]]]:
    return {channel: [[float(x), float(y)] for x, y in points] for channel, points in curves.items()}
