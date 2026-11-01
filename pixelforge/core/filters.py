"""Destructive, one-shot filters (Filter menu) with self-describing parameters.

Each filter is a function ``(RGBA image, **params) -> RGBA image`` registered
via :func:`register` with a list of :class:`Param` describing its arguments.
The GUI builds its parameter dialogs from these specs, and the CLI exposes them
as ``--filter name:param=value``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageOps


@dataclass(frozen=True)
class Param:
    name: str
    label: str
    min: float
    max: float
    default: float | str
    step: float = 1.0
    kind: str = "float"  # "int" | "float" | "bool" | "choice"
    choices: tuple[str, ...] = ()

    def coerce(self, value):
        if self.kind == "bool":
            return bool(value) if not isinstance(value, str) else value.lower() in ("1", "true", "yes", "on")
        if self.kind == "choice":
            if value not in self.choices:
                raise ValueError(f"{self.name} must be one of {self.choices}")
            return value
        num = float(value)
        num = min(self.max, max(self.min, num))
        return int(round(num)) if self.kind == "int" else num


@dataclass(frozen=True)
class FilterSpec:
    name: str
    category: str
    func: Callable[..., Image.Image]
    params: tuple[Param, ...] = field(default_factory=tuple)
    description: str = ""

    def defaults(self) -> dict:
        return {p.name: p.coerce(p.default) for p in self.params}

    def apply(self, img: Image.Image, **params) -> Image.Image:
        values = self.defaults()
        for key, value in params.items():
            spec = next((p for p in self.params if p.name == key), None)
            if spec is None:
                raise ValueError(f"filter {self.name!r} has no parameter {key!r}")
            values[key] = spec.coerce(value)
        return self.func(img.convert("RGBA"), **values)


FILTERS: dict[str, FilterSpec] = {}
CATEGORIES: tuple[str, ...] = ("Adjust", "Blur", "Sharpen", "Stylize", "Noise", "Distort")


def register(name: str, category: str, *params: Param, description: str = ""):
    def deco(fn):
        FILTERS[name] = FilterSpec(name, category, fn, tuple(params), description)
        return fn

    return deco


def filters_in(category: str) -> list[FilterSpec]:
    return [f for f in FILTERS.values() if f.category == category]


def apply_filter(img: Image.Image, name: str, **params) -> Image.Image:
    try:
        spec = FILTERS[name]
    except KeyError:
        raise KeyError(f"unknown filter {name!r}") from None
    return spec.apply(img, **params)


# ----------------------------------------------------------------- helpers --
def _rgb_op(img: Image.Image, fn: Callable[[Image.Image], Image.Image]) -> Image.Image:
    """Apply *fn* to the RGB channels, keeping alpha untouched."""
    rgb, alpha = img.convert("RGB"), img.getchannel("A")
    out = fn(rgb).convert("RGB")
    return Image.merge("RGBA", (*out.split(), alpha))


def _premultiplied(img: Image.Image, fn: Callable[[Image.Image], Image.Image]) -> Image.Image:
    """Apply a spatial filter to the premultiplied image so edges stay clean."""
    return fn(img.convert("RGBa")).convert("RGBA")


def _to_float(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float32) / 255.0


def _from_float(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA")


# ------------------------------------------------------------------ adjust --
@register("Invert", "Adjust", description="Invert colours (keeps transparency)")
def invert(img: Image.Image) -> Image.Image:
    return _rgb_op(img, ImageOps.invert)


@register("Desaturate", "Adjust", description="Convert to shades of grey")
def desaturate(img: Image.Image) -> Image.Image:
    return _rgb_op(img, lambda im: ImageOps.grayscale(im).convert("RGB"))


@register("Auto Contrast", "Adjust", Param("cutoff", "Cutoff %", 0, 20, 1, 0.5), description="Stretch the histogram")
def auto_contrast(img: Image.Image, cutoff: float = 1) -> Image.Image:
    return _rgb_op(img, lambda im: ImageOps.autocontrast(im, cutoff=cutoff))


@register("Equalize", "Adjust", description="Histogram equalisation")
def equalize(img: Image.Image) -> Image.Image:
    return _rgb_op(img, ImageOps.equalize)


@register("Posterize", "Adjust", Param("bits", "Bits per channel", 1, 8, 4, 1, "int"))
def posterize(img: Image.Image, bits: int = 4) -> Image.Image:
    return _rgb_op(img, lambda im: ImageOps.posterize(im, int(bits)))


@register("Solarize", "Adjust", Param("threshold", "Threshold", 0, 255, 128, 1, "int"))
def solarize(img: Image.Image, threshold: int = 128) -> Image.Image:
    return _rgb_op(img, lambda im: ImageOps.solarize(im, int(threshold)))


@register("Threshold", "Adjust", Param("level", "Level", 0, 255, 128, 1, "int"))
def threshold(img: Image.Image, level: int = 128) -> Image.Image:
    return _rgb_op(img, lambda im: im.convert("L").point(lambda v: 255 if v >= level else 0).convert("RGB"))


@register(
    "Levels",
    "Adjust",
    Param("in_black", "Input black", 0, 254, 0, 1, "int"),
    Param("in_white", "Input white", 1, 255, 255, 1, "int"),
    Param("gamma", "Gamma", 0.1, 5.0, 1.0, 0.05),
    Param("out_black", "Output black", 0, 255, 0, 1, "int"),
    Param("out_white", "Output white", 0, 255, 255, 1, "int"),
    description="Remap tonal range",
)
def levels(
    img: Image.Image, in_black: int = 0, in_white: int = 255, gamma: float = 1.0, out_black: int = 0, out_white: int = 255
) -> Image.Image:
    in_white = max(in_white, in_black + 1)
    lut = []
    for i in range(256):
        v = (i - in_black) / (in_white - in_black)
        v = min(1.0, max(0.0, v)) ** (1.0 / max(gamma, 1e-3))
        lut.append(int(round(out_black + v * (out_white - out_black))))
    return _rgb_op(img, lambda im: im.point(lut * 3))


@register(
    "Color Balance",
    "Adjust",
    Param("red", "Cyan / Red", -100, 100, 0, 1, "int"),
    Param("green", "Magenta / Green", -100, 100, 0, 1, "int"),
    Param("blue", "Yellow / Blue", -100, 100, 0, 1, "int"),
)
def color_balance(img: Image.Image, red: int = 0, green: int = 0, blue: int = 0) -> Image.Image:
    def op(im: Image.Image) -> Image.Image:
        channels = []
        for ch, shift in zip(im.split(), (red, green, blue), strict=True):
            channels.append(ch.point([min(255, max(0, i + int(shift))) for i in range(256)]))
        return Image.merge("RGB", channels)

    return _rgb_op(img, op)


@register(
    "Colorize",
    "Adjust",
    Param("hue", "Hue", 0, 360, 30, 1, "int"),
    Param("saturation", "Saturation", 0, 100, 50, 1, "int"),
    description="Tint the image with a single hue",
)
def colorize(img: Image.Image, hue: int = 30, saturation: int = 50) -> Image.Image:
    def op(im: Image.Image) -> Image.Image:
        lum = im.convert("L")
        h = Image.new("L", im.size, int(hue / 360 * 255))
        s = Image.new("L", im.size, int(saturation / 100 * 255))
        return Image.merge("HSV", (h, s, lum)).convert("RGB")

    return _rgb_op(img, op)


# -------------------------------------------------------------------- blur --
@register("Gaussian Blur", "Blur", Param("radius", "Radius", 0.1, 100, 3, 0.1))
def gaussian_blur(img: Image.Image, radius: float = 3) -> Image.Image:
    return _premultiplied(img, lambda im: im.filter(ImageFilter.GaussianBlur(radius)))


@register("Box Blur", "Blur", Param("radius", "Radius", 1, 100, 3, 1, "int"))
def box_blur(img: Image.Image, radius: int = 3) -> Image.Image:
    return _premultiplied(img, lambda im: im.filter(ImageFilter.BoxBlur(radius)))


@register(
    "Motion Blur",
    "Blur",
    Param("length", "Length", 1, 200, 15, 1, "int"),
    Param("angle", "Angle", 0, 180, 0, 1, "int"),
)
def motion_blur(img: Image.Image, length: int = 15, angle: int = 0) -> Image.Image:
    size = max(3, int(length) | 1)
    kernel = np.zeros((size, size), np.float32)
    c = size // 2
    rad = math.radians(angle)
    dx, dy = math.cos(rad), -math.sin(rad)
    for i in range(size):
        t = i - c
        x, y = int(round(c + t * dx)), int(round(c + t * dy))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1
    kernel /= kernel.sum()
    if size <= 5:  # Pillow's built-in kernels are limited to 3x3 and 5x5
        flt = ImageFilter.Kernel((size, size), kernel.flatten().tolist(), scale=1.0)
        return _premultiplied(img, lambda im: im.filter(flt))
    return _convolve_rgba(img, kernel)


def _convolve_rgba(img: Image.Image, kernel: np.ndarray) -> Image.Image:
    """Convolve an image with an arbitrary kernel using NumPy (edge-clamped)."""
    arr = np.asarray(img.convert("RGBa"), dtype=np.float32)  # premultiplied
    kh, kw = kernel.shape
    py, px = kh // 2, kw // 2
    padded = np.pad(arr, ((py, py), (px, px), (0, 0)), mode="edge")
    out = np.zeros_like(arr)
    ys, xs = np.nonzero(kernel)
    for ky, kx in zip(ys, xs, strict=True):
        out += kernel[ky, kx] * padded[ky : ky + arr.shape[0], kx : kx + arr.shape[1]]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBa").convert("RGBA")


@register("Median", "Blur", Param("size", "Size", 3, 15, 3, 2, "int"), description="Noise reduction")
def median(img: Image.Image, size: int = 3) -> Image.Image:
    size = int(size) | 1
    return img.filter(ImageFilter.MedianFilter(size))


@register("Smooth", "Blur")
def smooth(img: Image.Image) -> Image.Image:
    return _premultiplied(img, lambda im: im.filter(ImageFilter.SMOOTH_MORE))


# ----------------------------------------------------------------- sharpen --
@register("Sharpen", "Sharpen", Param("amount", "Amount", 0, 500, 100, 5, "int"))
def sharpen(img: Image.Image, amount: int = 100) -> Image.Image:
    return _rgb_op(img, lambda im: im.filter(ImageFilter.UnsharpMask(radius=1.5, percent=int(amount), threshold=0)))


@register(
    "Unsharp Mask",
    "Sharpen",
    Param("radius", "Radius", 0.1, 50, 2, 0.1),
    Param("percent", "Amount %", 1, 500, 150, 1, "int"),
    Param("threshold", "Threshold", 0, 255, 3, 1, "int"),
)
def unsharp_mask(img: Image.Image, radius: float = 2, percent: int = 150, threshold: int = 3) -> Image.Image:
    return _rgb_op(img, lambda im: im.filter(ImageFilter.UnsharpMask(radius, int(percent), int(threshold))))


@register("Edge Enhance", "Sharpen")
def edge_enhance(img: Image.Image) -> Image.Image:
    return _rgb_op(img, lambda im: im.filter(ImageFilter.EDGE_ENHANCE_MORE))


@register("Detail", "Sharpen")
def detail(img: Image.Image) -> Image.Image:
    return _rgb_op(img, lambda im: im.filter(ImageFilter.DETAIL))


# ----------------------------------------------------------------- stylize --
@register("Find Edges", "Stylize")
def find_edges(img: Image.Image) -> Image.Image:
    return _rgb_op(img, lambda im: im.filter(ImageFilter.FIND_EDGES))


@register("Emboss", "Stylize")
def emboss(img: Image.Image) -> Image.Image:
    return _rgb_op(img, lambda im: im.filter(ImageFilter.EMBOSS))


@register("Contour", "Stylize")
def contour(img: Image.Image) -> Image.Image:
    return _rgb_op(img, lambda im: im.filter(ImageFilter.CONTOUR))


@register("Pixelate", "Stylize", Param("block", "Block size", 2, 200, 8, 1, "int"))
def pixelate(img: Image.Image, block: int = 8) -> Image.Image:
    block = max(2, int(block))
    w, h = img.size
    small = img.resize((max(1, math.ceil(w / block)), max(1, math.ceil(h / block))), Image.Resampling.BOX)
    return small.resize((w, h), Image.Resampling.NEAREST)


@register(
    "Oil Paint",
    "Stylize",
    Param("radius", "Radius", 1, 12, 3, 1, "int"),
    Param("levels", "Intensity levels", 4, 64, 20, 1, "int"),
)
def oil_paint(img: Image.Image, radius: int = 3, levels: int = 20) -> Image.Image:
    """Classic oil-painting effect: each pixel takes the mean colour of the most common intensity bucket around it.

    Two passes over the intensity levels using Pillow's box blur as a fast
    windowed sum: first find the dominant level per pixel, then fetch that
    level's mean colour.  Cost is O(levels × pixels) regardless of the radius.
    """
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba)
    rgb = arr[..., :3]
    h, w = rgb.shape[:2]
    levels, r = int(levels), int(radius)
    lum = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    buckets = np.clip((lum / 255.0 * (levels - 1) + 0.5).astype(np.int32), 0, levels - 1)
    present = np.unique(buckets)
    # pass 1: dominant bucket per pixel (window counts via box blur of the bucket mask)
    best_count = np.zeros((h, w), np.uint8)
    best_level = np.zeros((h, w), np.int32)
    masks = {}
    for k in present:
        mask = buckets == k
        masks[k] = mask
        count = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255, "L").filter(ImageFilter.BoxBlur(r)))
        better = count > best_count
        best_count = np.where(better, count, best_count)
        best_level[better] = k
    # pass 2: mean colour of the dominant bucket, only for the pixels that picked it
    out_rgb = rgb.copy()
    for k in np.unique(best_level):
        sel = best_level == k
        packed = np.zeros((h, w, 4), np.uint8)
        packed[..., :3] = rgb * masks[k][..., None]
        packed[..., 3] = masks[k] * 255
        blurred = np.asarray(Image.fromarray(packed, "RGBA").filter(ImageFilter.BoxBlur(r)))[sel].astype(np.float32)
        mean = blurred[:, :3] / np.maximum(blurred[:, 3:4], 1e-3) * 255.0
        out_rgb[sel] = np.clip(mean + 0.5, 0, 255).astype(np.uint8)
    out = arr.copy()
    out[..., :3] = out_rgb
    return Image.fromarray(out, "RGBA")


@register(
    "Vignette",
    "Stylize",
    Param("strength", "Strength", 0, 100, 50, 1, "int"),
    Param("softness", "Softness", 0, 100, 50, 1, "int"),
)
def vignette(img: Image.Image, strength: int = 50, softness: int = 50) -> Image.Image:
    arr = _to_float(img)
    h, w = arr.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xs + 0.5 - w / 2) / (w / 2)
    ny = (ys + 0.5 - h / 2) / (h / 2)
    r = np.sqrt(nx * nx + ny * ny) / math.sqrt(2)
    inner = 1.0 - softness / 100 * 0.9
    t = np.clip((r - inner * 0.6) / max(1.05 - inner * 0.6, 1e-6), 0, 1)
    t = t * t * (3 - 2 * t)
    arr[..., :3] *= (1 - t * strength / 100)[..., None]
    return _from_float(arr)


@register(
    "Drop Shadow",
    "Stylize",
    Param("offset_x", "Offset X", -100, 100, 6, 1, "int"),
    Param("offset_y", "Offset Y", -100, 100, 6, 1, "int"),
    Param("blur", "Blur", 0, 50, 6, 0.5),
    Param("opacity", "Opacity %", 0, 100, 60, 1, "int"),
    description="Shadow behind the opaque pixels of the layer",
)
def drop_shadow(img: Image.Image, offset_x: int = 6, offset_y: int = 6, blur: float = 6, opacity: int = 60) -> Image.Image:
    alpha = img.getchannel("A")
    shadow_alpha = Image.new("L", img.size, 0)
    shadow_alpha.paste(alpha, (int(offset_x), int(offset_y)))
    if blur > 0:
        shadow_alpha = shadow_alpha.filter(ImageFilter.GaussianBlur(blur))
    shadow_alpha = shadow_alpha.point(lambda v: int(v * opacity / 100))
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shadow.putalpha(shadow_alpha)
    return Image.alpha_composite(shadow, img)


@register(
    "Outline",
    "Stylize",
    Param("width", "Width", 1, 50, 3, 1, "int"),
    Param("color", "Colour", 0, 0, "black", 1, "choice", ("black", "white", "red", "yellow", "blue")),
    description="Stroke around the opaque pixels of the layer",
)
def outline(img: Image.Image, width: int = 3, color: str = "black") -> Image.Image:
    colors = {"black": (0, 0, 0), "white": (255, 255, 255), "red": (230, 40, 40), "yellow": (250, 220, 40), "blue": (40, 110, 240)}
    alpha = img.getchannel("A")
    grown = alpha.filter(ImageFilter.MaxFilter(2 * int(width) + 1))
    ring = Image.new("RGBA", img.size, (*colors[color], 0))
    ring.putalpha(grown)
    return Image.alpha_composite(ring, img)


# ------------------------------------------------------------------- noise --
@register(
    "Add Noise",
    "Noise",
    Param("amount", "Amount", 0, 100, 20, 1, "int"),
    Param("monochrome", "Monochrome", 0, 1, 1, 1, "bool"),
    Param("seed", "Seed", 0, 9999, 0, 1, "int"),
)
def add_noise(img: Image.Image, amount: int = 20, monochrome: bool = True, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(int(seed) or None)
    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape[:2]
    sigma = amount / 100 * 64
    if monochrome:
        noise = rng.normal(0, sigma, (h, w, 1)).repeat(3, axis=2)
    else:
        noise = rng.normal(0, sigma, (h, w, 3))
    arr[..., :3] += noise
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")


@register("Despeckle", "Noise")
def despeckle(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.MedianFilter(3))


@register(
    "Film Grain",
    "Noise",
    Param("amount", "Amount", 0, 100, 25, 1, "int"),
    Param("size", "Grain size", 1, 5, 1, 1, "int"),
)
def film_grain(img: Image.Image, amount: int = 25, size: int = 1) -> Image.Image:
    rng = np.random.default_rng(12345)
    w, h = img.size
    gw, gh = max(1, w // size), max(1, h // size)
    grain = rng.normal(0, amount / 100 * 48, (gh, gw)).astype(np.float32)
    grain_img = Image.fromarray(np.clip(grain + 128, 0, 255).astype(np.uint8), "L").resize((w, h), Image.Resampling.BILINEAR)
    g = np.asarray(grain_img, dtype=np.float32) - 128
    arr = np.asarray(img, dtype=np.float32)
    lum = arr[..., :3].mean(axis=2) / 255.0
    weight = 1 - np.abs(lum - 0.5) * 1.2  # grain shows most in mid-tones
    arr[..., :3] += (g * np.clip(weight, 0.2, 1))[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")


# ----------------------------------------------------------------- distort --
@register(
    "Wave",
    "Distort",
    Param("amplitude", "Amplitude", 0, 100, 10, 1, "int"),
    Param("wavelength", "Wavelength", 4, 500, 60, 1, "int"),
    Param("direction", "Direction", 0, 0, "horizontal", 1, "choice", ("horizontal", "vertical")),
)
def wave(img: Image.Image, amplitude: int = 10, wavelength: int = 60, direction: str = "horizontal") -> Image.Image:
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    if direction == "horizontal":
        src_x = xs + amplitude * np.sin(2 * np.pi * ys / wavelength)
        src_y = ys
    else:
        src_x = xs
        src_y = ys + amplitude * np.sin(2 * np.pi * xs / wavelength)
    sx = np.clip(np.round(src_x).astype(np.int64), 0, w - 1)
    sy = np.clip(np.round(src_y).astype(np.int64), 0, h - 1)
    return Image.fromarray(arr[sy, sx], "RGBA")


@register(
    "Swirl",
    "Distort",
    Param("angle", "Angle", -720, 720, 90, 1, "int"),
    Param("radius", "Radius %", 10, 100, 100, 1, "int"),
)
def swirl(img: Image.Image, angle: int = 90, radius: int = 100) -> Image.Image:
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    dx, dy = xs - cx, ys - cy
    r = np.sqrt(dx * dx + dy * dy)
    max_r = min(cx, cy) * radius / 100
    theta = np.arctan2(dy, dx) + math.radians(angle) * np.clip(1 - r / max(max_r, 1e-6), 0, 1)
    sx = np.clip(np.round(cx + r * np.cos(theta)).astype(np.int64), 0, w - 1)
    sy = np.clip(np.round(cy + r * np.sin(theta)).astype(np.int64), 0, h - 1)
    return Image.fromarray(arr[sy, sx], "RGBA")


def difference(a: Image.Image, b: Image.Image) -> float:
    """Mean absolute difference between two images (testing helper)."""
    return float(np.asarray(ImageChops.difference(a.convert("RGBA"), b.convert("RGBA")), dtype=np.float32).mean())
