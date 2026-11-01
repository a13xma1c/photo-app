"""Painting primitives: brushes, strokes, fills, gradients, text and shapes.

All functions are pure: they take an RGBA layer image and return a *new* image.
The :class:`Stroke` class accumulates a soft-brush coverage mask incrementally
so the GUI can show a live preview while the mouse moves.
"""

from __future__ import annotations

import math
import os
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from .brushes import BrushPreset
from .color import parse_color

Point = tuple[float, float]

GRADIENT_TYPES: tuple[str, ...] = ("linear", "radial", "reflected")
SHAPE_TYPES: tuple[str, ...] = ("rectangle", "ellipse", "line")


# ------------------------------------------------------------------ brush --
@lru_cache(maxsize=64)
def brush_stamp(diameter: int, hardness: float = 0.8) -> np.ndarray:
    """Return a ``(d, d)`` float32 coverage stamp for a round soft brush.

    ``hardness`` in 0-1: 1 gives a crisp disc, 0 a fully soft gradient.
    """
    diameter = max(1, int(diameter))
    hardness = float(min(1.0, max(0.0, hardness)))
    if diameter == 1:
        return np.ones((1, 1), np.float32)
    radius = diameter / 2
    coords = (np.arange(diameter, dtype=np.float32) + 0.5) - radius
    dist = np.sqrt(coords[None, :] ** 2 + coords[:, None] ** 2) / radius
    inner = hardness * 0.999
    t = np.clip((dist - inner) / max(1 - inner, 1e-6), 0.0, 1.0)
    stamp = 1 - t * t * (3 - 2 * t)  # smoothstep falloff
    return stamp.astype(np.float32)


class Stroke:
    """Incrementally accumulates brush coverage for one mouse-down/up gesture.

    Dabs follow a :class:`~pixelforge.core.brushes.BrushPreset`: tip shape,
    spacing, scatter, size / opacity jitter and flow.  ``Stroke(size, diameter,
    hardness)`` still works and means a plain round brush.
    """

    def __init__(
        self,
        size: tuple[int, int],
        diameter: int | None = None,
        hardness: float = 0.8,
        spacing: float = 0.15,
        preset: BrushPreset | None = None,
        seed: int | None = None,
    ) -> None:
        self.width, self.height = size
        if preset is None:
            preset = BrushPreset(
                "Round", "round", max(1, int(diameter or 24)), int(round(hardness * 100)), max(1, int(round(spacing * 100)))
            )
        self.preset = preset.normalized()
        self.diameter = self.preset.size
        self.hardness = self.preset.hardness / 100
        self.spacing = max(0.5, self.diameter * self.preset.spacing / 100)
        self.coverage = np.zeros((self.height, self.width), np.float32)
        self._rng = random.Random(seed)
        self._last: Point | None = None
        self._carry = 0.0
        self.dirty: tuple[int, int, int, int] | None = None  # bbox of touched pixels

    # --------------------------------------------------------------------
    def _dab(self, x: float, y: float, stamp: np.ndarray, strength: float) -> None:
        d = stamp.shape[0]
        left, top = int(round(x - d / 2)), int(round(y - d / 2))
        x0, y0 = max(left, 0), max(top, 0)
        x1, y1 = min(left + d, self.width), min(top + d, self.height)
        if x1 <= x0 or y1 <= y0:
            return
        region = self.coverage[y0:y1, x0:x1]
        piece = stamp[y0 - top : y1 - top, x0 - left : x1 - left]
        if strength >= 0.999:
            np.maximum(region, piece, out=region)
        else:
            region += piece * np.float32(strength) * (1 - region)  # flow: dabs build up where they overlap
        if self.dirty is None:
            self.dirty = (x0, y0, x1, y1)
        else:
            a, b, c, e = self.dirty
            self.dirty = (min(a, x0), min(b, y0), max(c, x1), max(e, y1))

    def _stamp(self, x: float, y: float) -> None:
        p = self.preset
        rng = self._rng
        for _ in range(p.count):
            size = p.size
            if p.size_jitter:
                size = max(1, int(round(p.size * (1 - rng.random() * p.size_jitter / 100))))
            dx = dy = 0.0
            if p.scatter:
                reach = p.size * p.scatter / 100
                dx, dy = (rng.random() * 2 - 1) * reach, (rng.random() * 2 - 1) * reach
            strength = p.flow / 100
            if p.opacity_jitter:
                strength *= 1 - rng.random() * p.opacity_jitter / 100
            self._dab(x + dx, y + dy, p.stamp(size), strength)

    def add_point(self, x: float, y: float) -> None:
        """Extend the stroke to (x, y), stamping along the segment at fixed spacing."""
        if self._last is None:
            self._stamp(x, y)
            self._last = (x, y)
            return
        lx, ly = self._last
        dist = math.hypot(x - lx, y - ly)
        if dist == 0:
            return
        pos = self.spacing - self._carry
        while pos <= dist:
            t = pos / dist
            self._stamp(lx + (x - lx) * t, ly + (y - ly) * t)
            pos += self.spacing
        self._carry = dist - (pos - self.spacing)
        self._last = (x, y)

    def take_dirty(self) -> tuple[int, int, int, int] | None:
        """Return and reset the bbox touched since the last call (for incremental previews)."""
        box, self.dirty = self.dirty, None
        return box

    def add_points(self, points: Sequence[Point]) -> None:
        for x, y in points:
            self.add_point(x, y)

    def mask_image(self) -> Image.Image:
        return Image.fromarray((self.coverage * 255 + 0.5).astype(np.uint8), "L")


def _coverage_array(coverage: np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(coverage, Image.Image):
        return np.asarray(coverage.convert("L"), dtype=np.float32) / 255.0
    return coverage.astype(np.float32)


def _selection_array(selection: Image.Image | None, shape: tuple[int, int]) -> np.ndarray | None:
    if selection is None:
        return None
    arr = np.asarray(selection.convert("L"), dtype=np.float32) / 255.0
    if arr.shape != shape:
        raise ValueError("selection mask size does not match layer size")
    return arr


def _touched_bbox(cov: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.flatnonzero(cov.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(cov.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def paint(
    layer: Image.Image,
    coverage: np.ndarray | Image.Image,
    color,
    opacity: float = 1.0,
    selection: Image.Image | None = None,
) -> Image.Image:
    """Paint *color* through the *coverage* mask onto *layer* (returns a new image).

    Only the bounding box of the touched pixels is processed, so short strokes
    on large layers stay cheap.
    """
    layer = layer.convert("RGBA")
    cov = _coverage_array(coverage)
    if cov.shape != (layer.height, layer.width):
        raise ValueError("coverage size does not match layer size")
    sel = _selection_array(selection, cov.shape)
    if sel is not None:
        cov = cov * sel
    bbox = _touched_bbox(cov)
    if bbox is None:
        return layer.copy()
    x0, y0, x1, y1 = bbox
    cov = cov[y0:y1, x0:x1] * np.float32(min(1.0, max(0.0, opacity)))
    base = np.asarray(layer.crop(bbox)).astype(np.float32)
    base *= np.float32(1 / 255)
    r, g, b, a = parse_color(color)
    src_rgb = np.array([r, g, b], np.float32) / 255.0
    src_a = cov[..., None] * np.float32(a / 255.0)
    dst_rgb, dst_a = base[..., :3], base[..., 3:4]
    out_a = src_a + dst_a * (1 - src_a)
    with np.errstate(divide="ignore", invalid="ignore"):
        out_rgb = np.where(out_a > 0, (src_rgb * src_a + dst_rgb * dst_a * (1 - src_a)) / np.maximum(out_a, 1e-9), 0)
    patch = np.concatenate([out_rgb, out_a], axis=2)
    out = layer.copy()
    out.paste(Image.fromarray((np.clip(patch, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA"), (x0, y0))
    return out


def erase(
    layer: Image.Image,
    coverage: np.ndarray | Image.Image,
    opacity: float = 1.0,
    selection: Image.Image | None = None,
) -> Image.Image:
    """Reduce alpha through the *coverage* mask (returns a new image)."""
    layer = layer.convert("RGBA")
    cov = _coverage_array(coverage)
    if cov.shape != (layer.height, layer.width):
        raise ValueError("coverage size does not match layer size")
    sel = _selection_array(selection, cov.shape)
    if sel is not None:
        cov = cov * sel
    bbox = _touched_bbox(cov)
    if bbox is None:
        return layer.copy()
    x0, y0, x1, y1 = bbox
    cov = cov[y0:y1, x0:x1] * np.float32(min(1.0, max(0.0, opacity)))
    patch = np.asarray(layer.crop(bbox)).copy()
    alpha = patch[..., 3].astype(np.float32) * (1 - cov)
    patch[..., 3] = (alpha + 0.5).astype(np.uint8)
    out = layer.copy()
    out.paste(Image.fromarray(patch, "RGBA"), (x0, y0))
    return out


def _source_over(layer: Image.Image, cov: np.ndarray, bbox: tuple[int, int, int, int], src: np.ndarray) -> Image.Image:
    """Composite the float RGBA *src* patch (0..1, same size as *bbox*) over *layer* through *cov* (already clipped to bbox)."""
    x0, y0, x1, y1 = bbox
    base = np.asarray(layer.crop(bbox)).astype(np.float32)
    base *= np.float32(1 / 255)
    src_a = src[..., 3:4] * cov[..., None]
    dst_rgb, dst_a = base[..., :3], base[..., 3:4]
    out_a = src_a + dst_a * (1 - src_a)
    with np.errstate(divide="ignore", invalid="ignore"):
        out_rgb = np.where(out_a > 0, (src[..., :3] * src_a + dst_rgb * dst_a * (1 - src_a)) / np.maximum(out_a, 1e-9), 0)
    patch = np.concatenate([out_rgb, out_a], axis=2)
    out = layer.copy()
    out.paste(Image.fromarray((np.clip(patch, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA"), (x0, y0))
    return out


def _blur_array(arr: np.ndarray, radius: float) -> np.ndarray:
    """Gaussian blur of a float array (H×W or H×W×C) via Pillow."""
    if radius <= 0:
        return arr
    if arr.ndim == 2:
        img = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8), "L")
        return np.asarray(img.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0
    channels = [_blur_array(arr[..., c], radius) for c in range(arr.shape[2])]
    return np.stack(channels, axis=2)


def clone(
    layer: Image.Image,
    coverage: np.ndarray | Image.Image,
    source: Image.Image | None,
    offset: tuple[int, int] = (0, 0),
    opacity: float = 1.0,
    selection: Image.Image | None = None,
    heal: bool = False,
) -> Image.Image:
    """Clone stamp / healing brush: paint pixels of *source* (shifted by *offset*) through *coverage*.

    Destination ``(x, y)`` receives ``source[x + dx, y + dy]``.  With *heal*
    the source only contributes its texture (high frequencies) while the
    destination keeps its own colour and lighting - blemishes disappear
    instead of being stamped over.  ``source=None`` with *heal* is *spot
    healing*: the covered pixels are rebuilt from their uncovered surroundings.
    """
    layer = layer.convert("RGBA")
    cov = _coverage_array(coverage)
    if cov.shape != (layer.height, layer.width):
        raise ValueError("coverage size does not match layer size")
    sel = _selection_array(selection, cov.shape)
    if sel is not None:
        cov = cov * sel
    bbox = _touched_bbox(cov)
    if bbox is None:
        return layer.copy()
    if source is None and not heal:
        raise ValueError("the clone stamp needs a source")
    x0, y0, x1, y1 = bbox
    radius = max(2.0, min(x1 - x0, y1 - y0) / 4)
    if heal:  # healing looks at the surroundings, so work on a padded box
        pad = int(radius * 3) + 2
        x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(layer.width, x1 + pad), min(layer.height, y1 + pad)
        bbox = (x0, y0, x1, y1)
    cov = cov[y0:y1, x0:x1] * np.float32(min(1.0, max(0.0, opacity)))
    dst = np.asarray(layer.crop(bbox)).astype(np.float32) / 255.0
    if source is not None:
        if source.size != layer.size:
            raise ValueError("source size does not match layer size")
        dx, dy = int(offset[0]), int(offset[1])
        shifted = offset_image(source.convert("RGBA"), -dx, -dy)
        src = np.asarray(shifted.crop(bbox)).astype(np.float32) / 255.0
    else:
        src = dst.copy()
    if heal:
        # colour and lighting come from the *uncovered* surroundings of the destination
        # (normalised convolution: the blemish itself carries no weight) ...
        weight = np.clip(1.0 - cov, 0, 1) * dst[..., 3]
        smooth = _blur_array(dst[..., :3] * weight[..., None], radius * 2)
        norm = _blur_array(weight, radius * 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            surroundings = np.where(norm[..., None] > 1e-4, smooth / np.maximum(norm[..., None], 1e-4), dst[..., :3])
        if source is None:  # spot healing: the surroundings alone rebuild the area
            src = np.concatenate([surroundings, np.maximum(dst[..., 3:4], cov[..., None])], axis=2)
        else:  # ... while the source contributes only its texture
            detail = src[..., :3] - _blur_array(src[..., :3], radius)
            src = np.concatenate([np.clip(surroundings + detail, 0, 1), src[..., 3:4]], axis=2)
    return _source_over(layer, cov, bbox, src)


# ------------------------------------------------------------------- fill --
def fill(layer: Image.Image, color, mask: Image.Image | None = None, opacity: float = 1.0) -> Image.Image:
    """Fill the layer (or the masked region) with a solid colour."""
    layer = layer.convert("RGBA")
    rgba = parse_color(color)
    solid = Image.new("RGBA", layer.size, rgba)
    if mask is None and opacity >= 1.0:
        return solid
    mask_img = mask.convert("L") if mask is not None else Image.new("L", layer.size, 255)
    if opacity < 1.0:
        mask_img = mask_img.point(lambda v: int(v * opacity))
    out = layer.copy()
    out.paste(solid, (0, 0), mask_img)
    return out


def clear(layer: Image.Image, mask: Image.Image | None = None) -> Image.Image:
    """Make the masked region (or the whole layer) transparent."""
    return fill(layer, (0, 0, 0, 0), mask)


# --------------------------------------------------------------- gradient --
def gradient_image(size: tuple[int, int], start: Point, end: Point, color_a, color_b, kind: str = "linear") -> Image.Image:
    """RGBA gradient from *color_a* at *start* to *color_b* at *end*."""
    if kind not in GRADIENT_TYPES:
        raise ValueError(f"unknown gradient type {kind!r}")
    w, h = size
    ca = np.array(parse_color(color_a), np.float32) / 255.0
    cb = np.array(parse_color(color_b), np.float32) / 255.0
    xs = np.arange(w, dtype=np.float32) + 0.5
    ys = np.arange(h, dtype=np.float32) + 0.5
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    t: np.ndarray
    if length_sq == 0:
        t = np.ones((h, w), np.float32)
    elif kind == "radial":
        t = np.sqrt((xs[None, :] - start[0]) ** 2 + (ys[:, None] - start[1]) ** 2) / math.sqrt(length_sq)
    else:
        t = ((xs[None, :] - start[0]) * dx + (ys[:, None] - start[1]) * dy) / length_sq
        if kind == "reflected":
            t = np.abs(t)
    t = np.clip(t, 0.0, 1.0)[..., None]
    out = ca * (1 - t) + cb * t
    return Image.fromarray((out * 255 + 0.5).astype(np.uint8), "RGBA")


def apply_gradient(
    layer: Image.Image,
    start: Point,
    end: Point,
    color_a,
    color_b,
    kind: str = "linear",
    selection: Image.Image | None = None,
) -> Image.Image:
    """Draw a gradient over the layer (within the selection, if any)."""
    grad = gradient_image(layer.size, start, end, color_a, color_b, kind)
    if selection is None:
        return grad
    out = layer.convert("RGBA").copy()
    out.paste(grad, (0, 0), selection.convert("L"))
    return out


# ------------------------------------------------------------------- text --
_FONT_DIRS = {
    "linux": ["/usr/share/fonts", "/usr/local/share/fonts", "~/.fonts", "~/.local/share/fonts"],
    "darwin": ["/System/Library/Fonts", "/Library/Fonts", "~/Library/Fonts"],
    "win32": [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")],
}


@lru_cache(maxsize=1)
def system_fonts() -> dict[str, str]:
    """Map of font display names to file paths for TrueType/OpenType fonts found on this machine."""
    fonts: dict[str, str] = {}
    for base in _FONT_DIRS.get(sys.platform, _FONT_DIRS["linux"]):
        root = Path(os.path.expanduser(base))
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() in (".ttf", ".otf") and path.is_file():
                fonts.setdefault(path.stem, str(path))
    return fonts


def load_font(name: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a font by display name or path, falling back to Pillow's bundled font."""
    size = max(1, int(size))
    if name:
        path = system_fonts().get(name, name)
        try:
            return ImageFont.truetype(path, size)
        except (OSError, ValueError):
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def render_text(
    size: tuple[int, int],
    position: Point,
    text: str,
    color,
    font_name: str | None = None,
    font_size: int = 32,
    anchor: str = "la",
    align: str = "left",
    stroke_width: int = 0,
    stroke_color=None,
) -> Image.Image:
    """Render *text* onto a new transparent layer image."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    if not text:
        return img
    font = load_font(font_name, font_size)
    draw = ImageDraw.Draw(img)
    kwargs: dict[str, Any] = {}
    if stroke_width > 0:
        kwargs.update(stroke_width=int(stroke_width), stroke_fill=parse_color(stroke_color or (0, 0, 0, 255)))
    draw.multiline_text(
        (float(position[0]), float(position[1])),
        text,
        fill=parse_color(color),
        font=font,
        anchor=anchor if "\n" not in text else None,
        align=align,
        **kwargs,
    )
    return img


def text_bbox(text: str, font_name: str | None, font_size: int) -> tuple[int, int, int, int]:
    font = load_font(font_name, font_size)
    dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    left, top, right, bottom = dummy.multiline_textbbox((0, 0), text or " ", font=font)
    return int(left), int(top), int(right), int(bottom)


# ----------------------------------------------------------------- shapes --
def draw_shape(
    layer: Image.Image,
    kind: str,
    box: Sequence[float],
    fill_color=None,
    outline_color=None,
    width: int = 0,
    selection: Image.Image | None = None,
    antialias: bool = True,
) -> Image.Image:
    """Draw a rectangle, ellipse or line onto the layer (returns a new image)."""
    if kind not in SHAPE_TYPES:
        raise ValueError(f"unknown shape {kind!r}")
    ss = 3 if antialias else 1
    w, h = layer.size
    overlay = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x0, y0, x1, y1 = (float(v) * ss for v in box)
    fill_rgba = parse_color(fill_color) if fill_color is not None else None
    outline_rgba = parse_color(outline_color) if outline_color is not None and width > 0 else None
    line_w = max(1, int(round(width * ss))) if outline_rgba or kind == "line" else 0
    if kind == "line":
        draw.line((x0, y0, x1, y1), fill=outline_rgba or fill_rgba or (0, 0, 0, 255), width=line_w or ss)
    else:
        bx = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        fn = draw.rectangle if kind == "rectangle" else draw.ellipse
        if fill_rgba is None and outline_rgba is None:
            outline_rgba, line_w = (0, 0, 0, 255), ss
        fn(bx, fill=fill_rgba, outline=outline_rgba, width=line_w if outline_rgba else 0)
    if ss > 1:
        overlay = overlay.resize((w, h), Image.Resampling.LANCZOS)
    out = layer.convert("RGBA").copy()
    if selection is not None:
        alpha = overlay.getchannel("A")
        sel = selection.convert("L")
        overlay.putalpha(ImageChops.multiply(alpha, sel))
    return Image.alpha_composite(out, overlay)


# ----------------------------------------------------------------- offset --
def offset_image(image: Image.Image, dx: int, dy: int) -> Image.Image:
    """Shift image content without wrapping; exposed areas become transparent/black.

    Works for RGBA layers and for ``L`` selection masks alike.
    """
    out = Image.new(image.mode, image.size, (0, 0, 0, 0) if image.mode == "RGBA" else 0)
    out.paste(image, (int(dx), int(dy)))
    return out


# -------------------------------------------------------------- transform --
Matrix = tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class Transform:
    """An affine transform of a rectangular piece of a layer (Free Transform).

    The source rectangle *box* is scaled by ``scale_x``/``scale_y`` about
    *pivot* (its centre by default), rotated by *angle* degrees
    (counter-clockwise on screen, like :meth:`PIL.Image.Image.rotate`) and
    finally moved by ``dx``/``dy``.  Negative scales flip.
    """

    box: tuple[float, float, float, float]
    scale_x: float = 1.0
    scale_y: float = 1.0
    angle: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    pivot: tuple[float, float] | None = None

    @property
    def center(self) -> tuple[float, float]:
        if self.pivot is not None:
            return self.pivot
        x0, y0, x1, y1 = self.box
        return (x0 + x1) / 2, (y0 + y1) / 2

    @property
    def is_identity(self) -> bool:
        return (
            abs(self.scale_x - 1) < 1e-9
            and abs(self.scale_y - 1) < 1e-9
            and abs(self.angle) % 360 < 1e-9
            and abs(self.dx) < 1e-9
            and abs(self.dy) < 1e-9
        )

    def matrix(self) -> Matrix:
        """Forward mapping ``(x, y) -> (a*x + b*y + c, d*x + e*y + f)`` from source to destination."""
        px_, py_ = self.center
        rad = math.radians(self.angle)
        cos, sin = math.cos(rad), math.sin(rad)
        a, b = self.scale_x * cos, self.scale_y * sin
        d, e = -self.scale_x * sin, self.scale_y * cos
        c = px_ + self.dx - (a * px_ + b * py_)
        f = py_ + self.dy - (d * px_ + e * py_)
        return a, b, c, d, e, f

    def inverse(self) -> Matrix:
        """Destination → source mapping (what :meth:`PIL.Image.Image.transform` wants)."""
        a, b, c, d, e, f = self.matrix()
        det = a * e - b * d
        if abs(det) < 1e-12:
            raise ValueError("transform is degenerate")
        ia, ib, id_, ie = e / det, -b / det, -d / det, a / det
        return ia, ib, -(ia * c + ib * f), id_, ie, -(id_ * c + ie * f)

    def apply(self, x: float, y: float) -> tuple[float, float]:
        a, b, c, d, e, f = self.matrix()
        return a * x + b * y + c, d * x + e * y + f

    def unapply(self, x: float, y: float) -> tuple[float, float]:
        a, b, c, d, e, f = self.inverse()
        return a * x + b * y + c, d * x + e * y + f

    def corners(self) -> list[tuple[float, float]]:
        """Destination positions of the source box corners (nw, ne, se, sw)."""
        x0, y0, x1, y1 = self.box
        return [self.apply(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]

    def bounds(self) -> tuple[float, float, float, float]:
        pts = self.corners()
        xs, ys = [x for x, _ in pts], [y for _, y in pts]
        return min(xs), min(ys), max(xs), max(ys)


def transform_image(image: Image.Image, transform: Transform, resample: Image.Resampling | None = None) -> Image.Image:
    """Apply *transform* to the whole *image* (size unchanged; exposed areas become transparent / black).

    RGBA images are resampled premultiplied so transparent edges do not bleed dark fringes.
    """
    if transform.is_identity:
        return image
    coeffs = transform.inverse()
    if resample is None:
        resample = Image.Resampling.BICUBIC if image.mode == "RGBA" else Image.Resampling.BILINEAR
    if image.mode != "RGBA":
        return image.transform(image.size, Image.Transform.AFFINE, coeffs, resample=resample, fillcolor=0)
    arr = np.asarray(image, dtype=np.float32)
    alpha = arr[..., 3:4] / 255.0
    pre = np.empty_like(arr)
    pre[..., :3] = arr[..., :3] * alpha
    pre[..., 3] = arr[..., 3]
    moved = Image.fromarray(np.clip(pre + 0.5, 0, 255).astype(np.uint8), "RGBA").transform(
        image.size, Image.Transform.AFFINE, coeffs, resample=resample, fillcolor=(0, 0, 0, 0)
    )
    out = np.asarray(moved, dtype=np.float32)
    a = out[..., 3:4]
    rgb = np.where(a > 0, out[..., :3] * 255.0 / np.maximum(a, 1e-6), 0)
    result = np.empty_like(out)
    result[..., :3] = np.clip(rgb + 0.5, 0, 255)
    result[..., 3] = out[..., 3]
    return Image.fromarray(result.astype(np.uint8), "RGBA")
