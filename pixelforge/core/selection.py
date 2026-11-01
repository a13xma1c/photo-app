"""Selections are 8-bit masks (mode ``L``) the size of the document.

255 means fully selected, 0 not selected; intermediate values come from
feathering or anti-aliased ellipses/lassos and act as partial selection.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

Box = tuple[int, int, int, int]
Point = tuple[float, float]

COMBINE_MODES: tuple[str, ...] = ("replace", "add", "subtract", "intersect")


def _normalized_box(box: Sequence[float]) -> Box:
    x0, y0, x1, y1 = box
    return (int(round(min(x0, x1))), int(round(min(y0, y1))), int(round(max(x0, x1))), int(round(max(y0, y1))))


@dataclass(frozen=True)
class Selection:
    mask: Image.Image

    def __post_init__(self) -> None:
        if self.mask.mode != "L":
            object.__setattr__(self, "mask", self.mask.convert("L"))

    # ------------------------------------------------------------ factories --
    @classmethod
    def empty(cls, size: tuple[int, int]) -> Selection:
        return cls(Image.new("L", size, 0))

    @classmethod
    def all(cls, size: tuple[int, int]) -> Selection:
        return cls(Image.new("L", size, 255))

    @classmethod
    def rectangle(cls, size: tuple[int, int], box: Sequence[float]) -> Selection:
        mask = Image.new("L", size, 0)
        x0, y0, x1, y1 = _normalized_box(box)
        if x1 > x0 and y1 > y0:
            ImageDraw.Draw(mask).rectangle((x0, y0, x1 - 1, y1 - 1), fill=255)
        return cls(mask)

    @classmethod
    def ellipse(cls, size: tuple[int, int], box: Sequence[float], antialias: bool = True) -> Selection:
        x0, y0, x1, y1 = _normalized_box(box)
        if x1 <= x0 or y1 <= y0:
            return cls.empty(size)
        if not antialias:
            mask = Image.new("L", size, 0)
            ImageDraw.Draw(mask).ellipse((x0, y0, x1 - 1, y1 - 1), fill=255)
            return cls(mask)
        ss = 4  # supersample for smooth edges
        big = Image.new("L", ((x1 - x0) * ss, (y1 - y0) * ss), 0)
        ImageDraw.Draw(big).ellipse((0, 0, big.width - 1, big.height - 1), fill=255)
        small = big.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
        mask = Image.new("L", size, 0)
        mask.paste(small, (x0, y0))
        return cls(mask)

    @classmethod
    def polygon(cls, size: tuple[int, int], points: Sequence[Point]) -> Selection:
        mask = Image.new("L", size, 0)
        if len(points) >= 3:
            ImageDraw.Draw(mask).polygon([(float(x), float(y)) for x, y in points], fill=255)
        return cls(mask)

    @classmethod
    def magic_wand(cls, image: Image.Image, seed: tuple[int, int], tolerance: int = 32, contiguous: bool = True) -> Selection:
        """Select pixels similar to the one under *seed* (tolerance 0-255)."""
        arr = np.asarray(image.convert("RGBA"))
        mask = flood_mask(arr, seed, tolerance, contiguous)
        return cls(Image.fromarray(mask.astype(np.uint8) * 255, "L"))

    @classmethod
    def from_alpha(cls, image: Image.Image) -> Selection:
        """Select the opaque pixels of *image* (like ctrl-clicking a layer)."""
        return cls(image.convert("RGBA").getchannel("A"))

    # ------------------------------------------------------------- queries --
    @property
    def size(self) -> tuple[int, int]:
        return self.mask.size

    @property
    def bbox(self) -> Box | None:
        return self.mask.getbbox()

    @property
    def is_empty(self) -> bool:
        return self.mask.getbbox() is None

    @property
    def is_full(self) -> bool:
        return ImageChops.invert(self.mask).getbbox() is None

    def coverage(self) -> float:
        """Fraction of the document selected (0-1, counting partial selection)."""
        arr = np.asarray(self.mask, dtype=np.float64)
        return float(arr.mean() / 255) if arr.size else 0.0

    def contains(self, x: float, y: float) -> bool:
        ix, iy = int(x), int(y)
        if not (0 <= ix < self.mask.width and 0 <= iy < self.mask.height):
            return False
        value = self.mask.getpixel((ix, iy))
        return isinstance(value, (int, float)) and value > 127

    def as_array(self) -> np.ndarray:
        """Mask as float32 ``(H, W, 1)`` in 0-1."""
        return (np.asarray(self.mask, dtype=np.float32) / 255.0)[..., None]

    # --------------------------------------------------------------- edits --
    def combine(self, other: Selection, mode: str = "replace") -> Selection:
        if other.size != self.size:
            raise ValueError("selection sizes differ")
        if mode == "replace":
            return other
        if mode == "add":
            return Selection(ImageChops.lighter(self.mask, other.mask))
        if mode == "subtract":
            return Selection(ImageChops.darker(self.mask, ImageChops.invert(other.mask)))
        if mode == "intersect":
            return Selection(ImageChops.darker(self.mask, other.mask))
        raise ValueError(f"unknown combine mode {mode!r}")

    def invert(self) -> Selection:
        return Selection(ImageChops.invert(self.mask))

    def feather(self, radius: float) -> Selection:
        if radius <= 0:
            return self
        return Selection(self.mask.filter(ImageFilter.GaussianBlur(radius)))

    def grow(self, pixels: int) -> Selection:
        if pixels <= 0:
            return self
        return Selection(self.mask.filter(ImageFilter.MaxFilter(2 * pixels + 1)))

    def shrink(self, pixels: int) -> Selection:
        if pixels <= 0:
            return self
        return Selection(self.mask.filter(ImageFilter.MinFilter(2 * pixels + 1)))

    def border(self, width: int) -> Selection:
        grown, shrunk = self.grow(width), self.shrink(width)
        return Selection(ImageChops.subtract(grown.mask, shrunk.mask))

    def transformed(self, fn) -> Selection:
        """Apply an ``Image -> Image`` function (crop, resize, transpose...) to the mask."""
        return Selection(fn(self.mask))

    def outline(self, scale: float = 1.0) -> Image.Image:
        """1-pixel edge mask of the selection at the given display scale (mode L)."""
        mask = self.mask
        if scale != 1.0:
            size = (max(1, round(mask.width * scale)), max(1, round(mask.height * scale)))
            mask = mask.resize(size, Image.Resampling.NEAREST if scale > 1 else Image.Resampling.BILINEAR)
        binary = mask.point(lambda v: 255 if v > 127 else 0)
        eroded = binary.filter(ImageFilter.MinFilter(3))
        return ImageChops.subtract(binary, eroded)

    @staticmethod
    def marching_ants(edge: Image.Image, phase: int = 0, period: int = 8) -> Image.Image:
        """Turn an edge mask into an RGBA 'marching ants' overlay image."""
        e = np.asarray(edge) > 0
        h, w = e.shape
        diag = _diagonal_index((h, w))
        stripe = ((diag + phase) // (period // 2)) % 2 == 0
        rgba = np.zeros((h, w, 4), np.uint8)
        rgba[e & stripe] = (255, 255, 255, 255)
        rgba[e & ~stripe] = (0, 0, 0, 255)
        return Image.fromarray(rgba, "RGBA")


_DIAG_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _diagonal_index(shape: tuple[int, int]) -> np.ndarray:
    """``x + y`` for every pixel, memoised per shape (the ants animation calls this every tick)."""
    diag = _DIAG_CACHE.get(shape)
    if diag is None:
        if len(_DIAG_CACHE) > 8:
            _DIAG_CACHE.clear()
        h, w = shape
        diag = np.arange(w, dtype=np.int32)[None, :] + np.arange(h, dtype=np.int32)[:, None]
        _DIAG_CACHE[shape] = diag
    return diag


# ------------------------------------------------------------- flood fill --
def color_match_mask(arr: np.ndarray, seed: tuple[int, int], tolerance: int) -> np.ndarray:
    """Boolean mask of pixels whose max channel difference to the seed is <= tolerance."""
    x, y = int(seed[0]), int(seed[1])
    h, w = arr.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError("seed outside image")
    target = arr[y, x].astype(np.int16)
    diff = np.abs(arr.astype(np.int16) - target).max(axis=2)
    return diff <= tolerance


def flood_mask(arr: np.ndarray, seed: tuple[int, int], tolerance: int = 32, contiguous: bool = True) -> np.ndarray:
    """Scanline flood fill returning a boolean mask of the filled region.

    Works on any ``(H, W, C)`` uint8 array.  Rows are processed span by span
    with NumPy so the Python-level loop only runs once per *span*, which keeps
    it fast even for multi-megapixel images.
    """
    match = color_match_mask(arr, seed, tolerance)
    if not contiguous:
        return match
    h, w = match.shape
    filled = np.zeros_like(match)
    x0, y0 = int(seed[0]), int(seed[1])
    stack: list[tuple[int, int]] = [(x0, y0)]
    while stack:
        x, y = stack.pop()
        if filled[y, x] or not match[y, x]:
            continue
        row = match[y]
        blockers = np.flatnonzero(~row[: x + 1])
        left = int(blockers[-1]) + 1 if blockers.size else 0
        blockers = np.flatnonzero(~row[x:])
        right = x + int(blockers[0]) - 1 if blockers.size else w - 1
        filled[y, left : right + 1] = True
        for ny in (y - 1, y + 1):
            if 0 <= ny < h:
                seg = match[ny, left : right + 1] & ~filled[ny, left : right + 1]
                if not seg.any():
                    continue
                starts = np.flatnonzero(np.diff(np.concatenate(([False], seg)).astype(np.int8)) == 1)
                stack.extend((left + int(s), ny) for s in starts)
    return filled
