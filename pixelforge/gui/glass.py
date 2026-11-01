"""Pillow drawing primitives for the glass UI: anti-aliased shapes, frosted panels, vector icons.

All shape helpers render at a super-sampling factor and downscale, because
``ImageDraw`` itself does not anti-alias.  Results are cached by their
parameters, so repeated renders are cheap.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .theme import METRICS, PALETTE, RGBA, AnyFont, font_runs, font_size, px, text_width

SS = 3  # super-sampling factor


# ------------------------------------------------------------------ shapes --
@lru_cache(maxsize=256)
def rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    """Anti-aliased rounded-rectangle coverage mask (mode L)."""
    w, h = max(1, size[0]), max(1, size[1])
    radius = max(0, min(radius, w // 2, h // 2))
    big = Image.new("L", (w * SS, h * SS), 0)
    ImageDraw.Draw(big).rounded_rectangle((0, 0, w * SS - 1, h * SS - 1), radius * SS, fill=255)
    return big.resize((w, h), Image.Resampling.LANCZOS)


@lru_cache(maxsize=256)
def rounded_rect(size: tuple[int, int], radius: int, fill: RGBA, outline: RGBA | None = None, width: int = 1) -> Image.Image:
    """Anti-aliased filled (and optionally outlined) rounded rectangle (RGBA)."""
    w, h = max(1, size[0]), max(1, size[1])
    radius = max(0, min(radius, w // 2, h // 2))
    big = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(big)
    draw.rounded_rectangle((0, 0, w * SS - 1, h * SS - 1), radius * SS, fill=fill, outline=outline, width=width * SS if outline else 0)
    return big.resize((w, h), Image.Resampling.LANCZOS)


@lru_cache(maxsize=128)
def circle(diameter: int, fill: RGBA, outline: RGBA | None = None, width: int = 1) -> Image.Image:
    d = max(1, diameter)
    big = Image.new("RGBA", (d * SS, d * SS), (0, 0, 0, 0))
    ImageDraw.Draw(big).ellipse((0, 0, d * SS - 1, d * SS - 1), fill=fill, outline=outline, width=width * SS if outline else 0)
    return big.resize((d, d), Image.Resampling.LANCZOS)


@lru_cache(maxsize=64)
def shadow(size: tuple[int, int], radius: int, blur: int, alpha: int) -> Image.Image:
    """Soft drop shadow for a rounded rectangle of *size* (image is larger by 2*blur on each side)."""
    w, h = size
    pad = blur * 2
    img = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle((pad, pad, pad + w - 1, pad + h - 1), radius, fill=(0, 0, 0, alpha))
    return img.filter(ImageFilter.GaussianBlur(blur))


def paste(dst: Image.Image, src: Image.Image, xy: tuple[int, int]) -> None:
    """Alpha-composite *src* onto *dst* at *xy* (clipped)."""
    x, y = int(xy[0]), int(xy[1])
    if x >= dst.width or y >= dst.height or x + src.width <= 0 or y + src.height <= 0:
        return
    crop = src
    if x < 0 or y < 0 or x + src.width > dst.width or y + src.height > dst.height:
        cx0, cy0 = max(0, -x), max(0, -y)
        cx1, cy1 = min(src.width, dst.width - x), min(src.height, dst.height - y)
        crop = src.crop((cx0, cy0, cx1, cy1))
        x, y = max(0, x), max(0, y)
    dst.alpha_composite(crop, (x, y))


def fill_rounded(dst: Image.Image, box: tuple[int, int, int, int], radius: int, fill: RGBA, outline: RGBA | None = None) -> None:
    x0, y0, x1, y1 = (int(v) for v in box)
    if x1 <= x0 or y1 <= y0:
        return
    paste(dst, rounded_rect((x1 - x0, y1 - y0), radius, fill, outline), (x0, y0))


def fill_circle(dst: Image.Image, center: tuple[float, float], diameter: int, fill: RGBA, outline: RGBA | None = None) -> None:
    x, y = center
    paste(dst, circle(diameter, fill, outline), (int(round(x - diameter / 2)), int(round(y - diameter / 2))))


def text(dst: Image.Image, xy: tuple[float, float], string: str, fnt: AnyFont, fill: RGBA, anchor: str = "la") -> None:
    """Draw *string* with *fnt*; characters the font lacks (⌘ × • …) fall back to a symbol font.

    Multi-line strings are only supported with the default top-left anchor.
    """
    draw = ImageDraw.Draw(dst)
    if "\n" in string:
        line_h = int(font_size(fnt) * 1.45)
        for i, line in enumerate(string.split("\n")):
            text(dst, (xy[0], xy[1] + i * line_h), line, fnt, fill, "la")
        return
    runs = font_runs(string, fnt)
    if len(runs) <= 1:
        draw.text((xy[0], xy[1]), string, font=fnt, fill=fill, anchor=anchor)
        return
    # Mixed fonts: lay the runs out left-to-right on the primary font's baseline.
    widths = [run_font.getlength(run) for run, run_font in runs]
    total = sum(widths)
    x = xy[0] - {"l": 0.0, "m": total / 2, "r": total}[anchor[0]]
    ascent, descent = fnt.getmetrics() if isinstance(fnt, ImageFont.FreeTypeFont) else (font_size(fnt), 0)
    baseline = xy[1] + {"a": ascent, "m": (ascent - descent) / 2, "s": 0.0, "d": -descent}[anchor[1]]
    for (run, run_font), width in zip(runs, widths, strict=True):
        draw.text((x, baseline), run, font=run_font, fill=fill, anchor="ls")
        x += width


def ellipsize(string: str, fnt: AnyFont, max_width: int) -> str:
    """Shorten *string* with an ellipsis so it fits in *max_width* pixels."""
    if text_width(string, fnt) <= max_width:
        return string
    ell = "…"
    lo, hi = 0, len(string)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if text_width(string[:mid] + ell, fnt) <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return string[:lo].rstrip() + ell


# ------------------------------------------------------------------- glass --
def glass(backdrop: Image.Image, radius: int | None = None, blur: int | None = None, tint: RGBA | None = None) -> Image.Image:
    """Frosted glass panel the size of *backdrop* (RGB image of what lies behind it).

    The backdrop is blurred, tinted, given a soft top sheen and a 1px edge
    highlight that fades towards the bottom, then clipped to a rounded shape.
    """
    radius = METRICS.panel_radius if radius is None else radius
    blur = METRICS.blur if blur is None else blur
    tint = PALETTE.glass_tint if tint is None else tint
    w, h = backdrop.size
    base = backdrop.convert("RGBA")
    if blur > 0:
        # Blur at half resolution (it is low-frequency anyway) and pad with a
        # stretched copy first so the edges don't darken against the border.
        hw, hh, hb = max(1, w // 2), max(1, h // 2), max(1, blur // 2)
        small = base.resize((hw, hh), Image.Resampling.BILINEAR)
        padded = small.resize((hw + hb * 2, hh + hb * 2), Image.Resampling.BILINEAR)
        padded.paste(small, (hb, hb))
        blurred = padded.filter(ImageFilter.GaussianBlur(hb)).crop((hb, hb, hb + hw, hb + hh))
        base = blurred.resize((w, h), Image.Resampling.BILINEAR)
    base.alpha_composite(Image.new("RGBA", (w, h), tint))

    sheen_h = max(1, h // 3)
    sheen = Image.new("RGBA", (w, sheen_h), (255, 255, 255, 0))
    sheen.putalpha(Image.linear_gradient("L").resize((w, sheen_h)).point(lambda v: int((255 - v) / 255 * PALETTE.glass_sheen)))
    base.alpha_composite(sheen)

    outer = np.asarray(rounded_mask((w, h), radius), dtype=np.int32)
    inner = np.zeros((h, w), np.int32)
    t = max(1, px(1))  # edge highlight thickness
    if w > 2 * t and h > 2 * t:
        inner[t:-t, t:-t] = np.asarray(rounded_mask((w - 2 * t, h - 2 * t), max(0, radius - t)), dtype=np.int32)
    ring = np.clip(outer - inner, 0, 255)
    fade = np.linspace(1.0, 0.3, h, dtype=np.float32)[:, None]  # brighter edge at the top
    ring_alpha = (ring * fade * (PALETTE.glass_border / 255)).astype(np.uint8)
    highlight = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    highlight.putalpha(Image.fromarray(ring_alpha, "L"))
    base.alpha_composite(highlight)

    alpha = (np.asarray(base.getchannel("A"), dtype=np.int32) * outer // 255).astype(np.uint8)
    base.putalpha(Image.fromarray(alpha, "L"))
    return base


# ------------------------------------------------------------------- icons --
IconFn = Callable[[ImageDraw.ImageDraw, float, float, float, RGBA], None]
ICONS: dict[str, IconFn] = {}


def _icon(name: str):
    def deco(fn: IconFn) -> IconFn:
        ICONS[name] = fn
        return fn

    return deco


def icon(name: str, size: int | None = None, color: RGBA | None = None) -> Image.Image:
    """Render a vector icon to an RGBA image (cached); the colour defaults to the theme's text colour."""
    return _icon_cached(name, METRICS.icon_size if size is None else size, PALETTE.fg if color is None else color)


@lru_cache(maxsize=512)
def _icon_cached(name: str, size: int, color: RGBA) -> Image.Image:
    big = Image.new("RGBA", (size * SS, size * SS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(big)
    fn = ICONS.get(name)
    if fn is None:
        draw.rectangle((size, size, size * SS - size, size * SS - size), outline=color, width=SS)
    else:
        fn(draw, size * SS / 2, size * SS / 2, size * SS, color)
    return big.resize((size, size), Image.Resampling.LANCZOS)


def _w(s: float) -> int:
    return max(1, int(round(s * 0.085)))


def _line(d, pts, s, col, width=None):
    d.line([(float(x), float(y)) for x, y in pts], fill=col, width=width or _w(s), joint="curve")


def _arrow_head(d, tip, angle, s, col, length=None):
    length = length or s * 0.17
    for off in (math.radians(150), math.radians(-150)):
        a = angle + off
        _line(d, [tip, (tip[0] + math.cos(a) * length, tip[1] + math.sin(a) * length)], s, col)


@_icon("hand")
def _hand(d, x, y, s, col):
    r = s * 0.34
    d.rounded_rectangle((x - r * 0.7, y - r * 0.1, x + r * 0.7, y + r * 1.0), radius=r * 0.3, outline=col, width=_w(s))
    for fx, top in ((-0.5, -0.55), (-0.17, -0.95), (0.17, -0.85), (0.5, -0.5)):
        _line(d, [(x + r * fx, y - r * 0.05), (x + r * fx, y + r * top)], s, col)
    _line(d, [(x - r * 0.7, y + r * 0.45), (x - r * 1.1, y - r * 0.1)], s, col)


@_icon("zoom")
def _zoom(d, x, y, s, col):
    r = s * 0.24
    cx, cy = x - r * 0.3, y - r * 0.3
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=col, width=_w(s))
    _line(d, [(cx + r * 0.72, cy + r * 0.72), (x + r * 1.35, y + r * 1.35)], s, col, _w(s) + SS // 2)


@_icon("zoom-in")
def _zoom_in(d, x, y, s, col):
    r = s * 0.26
    _line(d, [(x - r, y), (x + r, y)], s, col)
    _line(d, [(x, y - r), (x, y + r)], s, col)


@_icon("zoom-out")
def _zoom_out(d, x, y, s, col):
    r = s * 0.26
    _line(d, [(x - r, y), (x + r, y)], s, col)


@_icon("fit")
def _fit(d, x, y, s, col):
    r = s * 0.32
    k = r * 0.45
    for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        _line(d, [(x + sx * r, y + sy * (r - k)), (x + sx * r, y + sy * r), (x + sx * (r - k), y + sy * r)], s, col)


@_icon("crop")
def _crop(d, x, y, s, col):
    r = s * 0.32
    _line(d, [(x - r * 0.45, y - r), (x - r * 0.45, y + r * 0.45), (x + r, y + r * 0.45)], s, col)
    _line(d, [(x - r, y - r * 0.45), (x + r * 0.45, y - r * 0.45), (x + r * 0.45, y + r)], s, col)


@_icon("rotate-cw")
def _rotate_cw(d, x, y, s, col):
    r = s * 0.28
    d.arc((x - r, y - r, x + r, y + r), start=-60, end=200, fill=col, width=_w(s))
    tip = (x + r * math.cos(math.radians(-60)), y + r * math.sin(math.radians(-60)))
    _arrow_head(d, tip, math.radians(-60 + 90), s, col)


@_icon("rotate-ccw")
def _rotate_ccw(d, x, y, s, col):
    r = s * 0.28
    d.arc((x - r, y - r, x + r, y + r), start=-20, end=240, fill=col, width=_w(s))
    tip = (x + r * math.cos(math.radians(240)), y + r * math.sin(math.radians(240)))
    _arrow_head(d, tip, math.radians(240 - 90), s, col)


@_icon("flip-h")
def _flip_h(d, x, y, s, col):
    r = s * 0.3
    d.line([(x, y - r * 1.15), (x, y + r * 1.15)], fill=col, width=_w(s))
    d.polygon([(x - r * 0.3, y - r * 0.75), (x - r * 0.3, y + r * 0.75), (x - r * 1.1, y)], outline=col, width=_w(s))
    d.polygon([(x + r * 0.3, y - r * 0.75), (x + r * 0.3, y + r * 0.75), (x + r * 1.1, y)], fill=col)


@_icon("flip-v")
def _flip_v(d, x, y, s, col):
    r = s * 0.3
    d.line([(x - r * 1.15, y), (x + r * 1.15, y)], fill=col, width=_w(s))
    d.polygon([(x - r * 0.75, y - r * 0.3), (x + r * 0.75, y - r * 0.3), (x, y - r * 1.1)], outline=col, width=_w(s))
    d.polygon([(x - r * 0.75, y + r * 0.3), (x + r * 0.75, y + r * 0.3), (x, y + r * 1.1)], fill=col)


@_icon("undo")
def _undo(d, x, y, s, col):
    r = s * 0.3
    _line(d, [(x - r, y - r * 0.1), (x + r * 0.3, y - r * 0.1)], s, col)
    d.arc((x - r * 0.3, y - r * 0.1, x + r * 0.9, y + r * 1.0), start=-90, end=90, fill=col, width=_w(s))
    _line(d, [(x + r * 0.3, y + r * 1.0), (x - r * 0.1, y + r * 1.0)], s, col)
    _arrow_head(d, (x - r, y - r * 0.1), math.pi, s, col)


@_icon("redo")
def _redo(d, x, y, s, col):
    r = s * 0.3
    _line(d, [(x + r, y - r * 0.1), (x - r * 0.3, y - r * 0.1)], s, col)
    d.arc((x - r * 0.9, y - r * 0.1, x + r * 0.3, y + r * 1.0), start=90, end=270, fill=col, width=_w(s))
    _line(d, [(x - r * 0.3, y + r * 1.0), (x + r * 0.1, y + r * 1.0)], s, col)
    _arrow_head(d, (x + r, y - r * 0.1), 0, s, col)


@_icon("revert")
def _revert(d, x, y, s, col):
    r = s * 0.28
    d.arc((x - r, y - r, x + r, y + r), start=30, end=330, fill=col, width=_w(s))
    tip = (x + r * math.cos(math.radians(30)), y + r * math.sin(math.radians(30)))
    _arrow_head(d, tip, math.radians(30 - 90), s, col)


@_icon("open")
def _open(d, x, y, s, col):
    r = s * 0.32
    _line(
        d,
        [
            (x - r, y + r * 0.7),
            (x - r, y - r * 0.6),
            (x - r * 0.35, y - r * 0.6),
            (x - r * 0.1, y - r * 0.3),
            (x + r, y - r * 0.3),
            (x + r, y + r * 0.7),
            (x - r, y + r * 0.7),
        ],
        s,
        col,
    )


@_icon("export")
def _export(d, x, y, s, col):
    r = s * 0.32
    _line(d, [(x - r, y + r * 0.3), (x - r, y + r * 0.8), (x + r, y + r * 0.8), (x + r, y + r * 0.3)], s, col)
    _line(d, [(x, y - r * 0.9), (x, y + r * 0.35)], s, col)
    _arrow_head(d, (x, y + r * 0.35), math.pi / 2, s, col)


@_icon("more")
def _more(d, x, y, s, col):
    r = s * 0.07
    for dx in (-0.3, 0, 0.3):
        cx = x + dx * s
        d.ellipse((cx - r, y - r, cx + r, y + r), fill=col)


@_icon("check")
def _check(d, x, y, s, col):
    r = s * 0.28
    _line(d, [(x - r, y), (x - r * 0.25, y + r * 0.7), (x + r, y - r * 0.6)], s, col)


@_icon("close")
def _close(d, x, y, s, col):
    r = s * 0.24
    _line(d, [(x - r, y - r), (x + r, y + r)], s, col)
    _line(d, [(x - r, y + r), (x + r, y - r)], s, col)


@_icon("image")
def _image(d, x, y, s, col):
    r = s * 0.34
    d.rounded_rectangle((x - r, y - r * 0.8, x + r, y + r * 0.8), radius=r * 0.2, outline=col, width=_w(s))
    _line(
        d,
        [
            (x - r * 0.8, y + r * 0.55),
            (x - r * 0.2, y - r * 0.1),
            (x + r * 0.2, y + r * 0.3),
            (x + r * 0.45, y + r * 0.05),
            (x + r * 0.85, y + r * 0.55),
        ],
        s,
        col,
    )
    d.ellipse((x + r * 0.3, y - r * 0.55, x + r * 0.6, y - r * 0.25), fill=col)


@_icon("resize")
def _resize(d, x, y, s, col):
    r = s * 0.32
    d.rectangle((x - r, y - r * 0.75, x + r * 0.2, y + r * 0.35), outline=col, width=_w(s))
    _line(
        d,
        [
            (x + r * 0.2, y - r * 0.75),
            (x + r, y - r * 0.75),
            (x + r, y + r * 0.85),
            (x - r * 0.4, y + r * 0.85),
            (x - r * 0.4, y + r * 0.35),
        ],
        s,
        col,
    )


@_icon("chevron-down")
def _chevron(d, x, y, s, col):
    r = s * 0.2
    _line(d, [(x - r, y - r * 0.4), (x, y + r * 0.5), (x + r, y - r * 0.4)], s, col)


@_icon("keyboard")
def _keyboard(d, x, y, s, col):
    r = s * 0.34
    d.rounded_rectangle((x - r, y - r * 0.6, x + r, y + r * 0.6), radius=r * 0.2, outline=col, width=_w(s))
    for row, ys in ((-0.25, 3), (0.1, 3)):
        for i in range(ys):
            cx = x + (i - 1) * r * 0.5
            d.rectangle((cx - r * 0.1, y + r * row - r * 0.08, cx + r * 0.1, y + r * row + r * 0.08), fill=col)
    d.rectangle((x - r * 0.45, y + r * 0.3, x + r * 0.45, y + r * 0.42), fill=col)


@_icon("info")
def _info(d, x, y, s, col):
    r = s * 0.32
    d.ellipse((x - r, y - r, x + r, y + r), outline=col, width=_w(s))
    _line(d, [(x, y - r * 0.1), (x, y + r * 0.55)], s, col)
    d.ellipse((x - r * 0.12, y - r * 0.6, x + r * 0.12, y - r * 0.36), fill=col)


@_icon("eye")
def _eye(d, x, y, s, col):
    r = s * 0.36
    pts = [(x - r, y), (x - r * 0.5, y - r * 0.55), (x, y - r * 0.7), (x + r * 0.5, y - r * 0.55), (x + r, y)]
    _line(d, pts, s, col)
    pts = [(x - r, y), (x - r * 0.5, y + r * 0.55), (x, y + r * 0.7), (x + r * 0.5, y + r * 0.55), (x + r, y)]
    _line(d, pts, s, col)
    d.ellipse((x - r * 0.28, y - r * 0.28, x + r * 0.28, y + r * 0.28), fill=col)


@_icon("straighten")
def _straighten(d, x, y, s, col):
    r = s * 0.34
    d.rectangle((x - r, y - r * 0.7, x + r, y + r * 0.7), outline=col, width=_w(s))
    _line(d, [(x - r * 1.1, y + r * 0.35), (x + r * 1.1, y - r * 0.35)], s, col)


@_icon("canvas")
def _canvas(d, x, y, s, col):
    r = s * 0.34
    d.rectangle((x - r, y - r * 0.8, x + r, y + r * 0.8), outline=col, width=_w(s))
    d.rectangle((x - r * 0.45, y - r * 0.35, x + r * 0.45, y + r * 0.35), fill=col)


@_icon("trim")
def _trim(d, x, y, s, col):
    r = s * 0.34
    d.rectangle((x - r, y - r * 0.8, x + r, y + r * 0.8), outline=col, width=_w(s))
    _line(d, [(x - r * 0.5, y), (x + r * 0.5, y)], s, col)
    _arrow_head(d, (x - r * 0.5, y), math.pi, s, col, s * 0.12)
    _arrow_head(d, (x + r * 0.5, y), 0, s, col, s * 0.12)


@_icon("trash")
def _trash(d, x, y, s, col):
    r = s * 0.34
    _line(d, [(x - r, y - r * 0.6), (x + r, y - r * 0.6)], s, col)
    _line(d, [(x - r * 0.35, y - r * 0.6), (x - r * 0.35, y - r), (x + r * 0.35, y - r), (x + r * 0.35, y - r * 0.6)], s, col)
    d.polygon(
        [(x - r * 0.75, y - r * 0.6), (x + r * 0.75, y - r * 0.6), (x + r * 0.55, y + r), (x - r * 0.55, y + r)], outline=col, width=_w(s)
    )


@_icon("plus")
def _plus(d, x, y, s, col):
    r = s * 0.3
    _line(d, [(x - r, y), (x + r, y)], s, col)
    _line(d, [(x, y - r), (x, y + r)], s, col)


@_icon("copy")
def _copy(d, x, y, s, col):
    r = s * 0.32
    d.rounded_rectangle((x - r, y - r * 0.4, x + r * 0.4, y + r), radius=r * 0.15, outline=col, width=_w(s))
    _line(d, [(x - r * 0.4, y - r * 0.4), (x - r * 0.4, y - r), (x + r, y - r), (x + r, y + r * 0.4)], s, col)


@_icon("clock")
def _clock(d, x, y, s, col):
    r = s * 0.34
    d.ellipse((x - r, y - r, x + r, y + r), outline=col, width=_w(s))
    _line(d, [(x, y - r * 0.55), (x, y), (x + r * 0.4, y + r * 0.25)], s, col)


@_icon("filter")
def _filter(d, x, y, s, col):
    r = s * 0.34
    for cx, cy in ((x - r * 0.35, y - r * 0.2), (x + r * 0.35, y - r * 0.2), (x, y + r * 0.4)):
        d.ellipse((cx - r * 0.55, cy - r * 0.55, cx + r * 0.55, cy + r * 0.55), outline=col, width=_w(s))


@_icon("move")
def _move(d, x, y, s, col):
    r = s * 0.36
    _line(d, [(x - r, y), (x + r, y)], s, col)
    _line(d, [(x, y - r), (x, y + r)], s, col)
    for tip, angle in (((x - r, y), math.pi), ((x + r, y), 0.0), ((x, y - r), -math.pi / 2), ((x, y + r), math.pi / 2)):
        _arrow_head(d, tip, angle, s, col, s * 0.13)


@_icon("marquee")
def _marquee(d, x, y, s, col):
    r = s * 0.34
    w = _w(s)
    dash = r * 0.32
    for i in range(4):
        sx = x - r + i * (2 * r) / 4
        d.line([(sx, y - r * 0.8), (sx + dash, y - r * 0.8)], fill=col, width=w)
        d.line([(sx, y + r * 0.8), (sx + dash, y + r * 0.8)], fill=col, width=w)
    for i in range(3):
        sy = y - r * 0.8 + i * (1.6 * r) / 3
        d.line([(x - r, sy), (x - r, sy + dash)], fill=col, width=w)
        d.line([(x + r, sy), (x + r, sy + dash)], fill=col, width=w)


@_icon("lasso")
def _lasso(d, x, y, s, col):
    r = s * 0.32
    pts = []
    for i in range(13):
        a = math.pi * 2 * i / 12
        pts.append((x + math.cos(a) * r, y - r * 0.25 + math.sin(a) * r * 0.6))
    _line(d, pts, s, col)
    _line(d, [(x - r * 0.15, y + r * 0.35), (x - r * 0.45, y + r * 1.1)], s, col)


@_icon("wand")
def _wand(d, x, y, s, col):
    r = s * 0.36
    _line(d, [(x - r, y + r), (x + r * 0.15, y - r * 0.15)], s, col, _w(s) + SS // 2)
    cx, cy = x + r * 0.45, y - r * 0.45
    for a in range(0, 360, 45):
        rad = math.radians(a)
        length = r * 0.42 if a % 90 == 0 else r * 0.22
        d.line([(cx, cy), (cx + math.cos(rad) * length, cy + math.sin(rad) * length)], fill=col, width=max(1, _w(s) // 2))


@_icon("brush")
def _brush(d, x, y, s, col):
    r = s * 0.36
    _line(d, [(x + r, y - r), (x - r * 0.1, y + r * 0.1)], s, col)
    d.polygon(
        [(x - r * 0.1, y + r * 0.1), (x - r * 0.5, y), (x - r * 0.95, y + r * 0.8), (x - r * 0.25, y + r * 0.6)],
        fill=col,
    )


@_icon("eraser")
def _eraser(d, x, y, s, col):
    r = s * 0.34
    d.polygon([(x - r, y + r * 0.3), (x + r * 0.3, y - r), (x + r, y - r * 0.3), (x - r * 0.3, y + r)], outline=col, width=_w(s))
    _line(d, [(x - r * 0.45, y - r * 0.25), (x + r * 0.2, y + r * 0.4)], s, col)
    _line(d, [(x - r, y + r * 1.05), (x + r, y + r * 1.05)], s, col)


@_icon("fill")
def _fill(d, x, y, s, col):
    r = s * 0.34
    d.polygon(
        [(x - r * 0.7, y - r * 0.1), (x - r * 0.1, y - r * 0.9), (x + r * 0.7, y - r * 0.1), (x + r * 0.1, y + r * 0.7)],
        outline=col,
        width=_w(s),
    )
    _line(d, [(x - r * 0.4, y - r * 0.5), (x - r * 0.4, y - r * 1.05)], s, col)
    d.ellipse((x + r * 0.55, y + r * 0.3, x + r * 1.0, y + r * 0.9), fill=col)


@_icon("gradient")
def _gradient(d, x, y, s, col):
    r = s * 0.34
    steps = 6
    for i in range(steps):
        alpha = int(255 * (1 - i / steps))
        sx0 = x - r + i * (2 * r) / steps
        d.rectangle((sx0, y - r * 0.7, sx0 + (2 * r) / steps + 1, y + r * 0.7), fill=(*col[:3], alpha))
    d.rectangle((x - r, y - r * 0.7, x + r, y + r * 0.7), outline=col, width=max(1, _w(s) // 2))


@_icon("text")
def _text_icon(d, x, y, s, col):
    r = s * 0.34
    _line(d, [(x - r, y - r * 0.8), (x + r, y - r * 0.8)], s, col)
    _line(d, [(x, y - r * 0.8), (x, y + r * 0.9)], s, col)
    _line(d, [(x - r * 0.35, y + r * 0.9), (x + r * 0.35, y + r * 0.9)], s, col)


@_icon("shape")
def _shape(d, x, y, s, col):
    r = s * 0.34
    d.rectangle((x - r, y - r, x + r * 0.3, y + r * 0.3), outline=col, width=_w(s))
    d.ellipse((x - r * 0.3, y - r * 0.3, x + r, y + r), outline=col, width=_w(s))


@_icon("eyedropper")
def _eyedropper(d, x, y, s, col):
    r = s * 0.36
    _line(d, [(x - r * 0.9, y + r * 0.9), (x + r * 0.3, y - r * 0.3)], s, col)
    _line(d, [(x + r * 0.05, y - r * 0.55), (x + r * 0.55, y - r * 0.05)], s, col, _w(s) + SS)
    d.ellipse((x + r * 0.25, y - r * 1.0, x + r * 1.0, y - r * 0.25), fill=col)


@_icon("duplicate")
def _duplicate(d, x, y, s, col):
    r = s * 0.32
    d.rectangle((x - r, y - r * 0.4, x + r * 0.4, y + r), outline=col, width=_w(s))
    _line(d, [(x - r * 0.4, y - r * 0.4), (x - r * 0.4, y - r), (x + r, y - r), (x + r, y + r * 0.4)], s, col)


@_icon("merge")
def _merge(d, x, y, s, col):
    r = s * 0.34
    _line(d, [(x, y - r), (x, y + r * 0.25)], s, col)
    _arrow_head(d, (x, y + r * 0.25), math.pi / 2, s, col, s * 0.14)
    _line(d, [(x - r, y + r * 0.7), (x + r, y + r * 0.7)], s, col)


@_icon("up")
def _up(d, x, y, s, col):
    r = s * 0.3
    _line(d, [(x - r, y + r * 0.45), (x, y - r * 0.45), (x + r, y + r * 0.45)], s, col)


@_icon("down")
def _down(d, x, y, s, col):
    r = s * 0.3
    _line(d, [(x - r, y - r * 0.45), (x, y + r * 0.45), (x + r, y - r * 0.45)], s, col)


@_icon("lock")
def _lock(d, x, y, s, col):
    r = s * 0.3
    d.rounded_rectangle((x - r, y - r * 0.1, x + r, y + r), radius=r * 0.2, outline=col, width=_w(s))
    d.arc((x - r * 0.6, y - r * 1.0, x + r * 0.6, y + r * 0.2), start=180, end=360, fill=col, width=_w(s))


@_icon("unlock")
def _unlock(d, x, y, s, col):
    r = s * 0.3
    d.rounded_rectangle((x - r, y - r * 0.1, x + r, y + r), radius=r * 0.2, outline=col, width=_w(s))
    d.arc((x + r * 0.1, y - r * 1.0, x + r * 1.3, y + r * 0.2), start=180, end=300, fill=col, width=_w(s))


@_icon("eye-off")
def _eye_off(d, x, y, s, col):
    r = s * 0.36
    pts = [(x - r, y), (x - r * 0.5, y + r * 0.55), (x, y + r * 0.7), (x + r * 0.5, y + r * 0.55), (x + r, y)]
    _line(d, pts, s, col)
    _line(d, [(x - r * 0.8, y + r * 0.8), (x + r * 0.8, y - r * 0.8)], s, col)


@_icon("swap")
def _swap(d, x, y, s, col):
    r = s * 0.3
    d.arc((x - r, y - r, x + r, y + r), start=200, end=340, fill=col, width=_w(s))
    tip = (x + r * math.cos(math.radians(340)), y + r * math.sin(math.radians(340)))
    _arrow_head(d, tip, math.radians(340 + 90), s, col, s * 0.12)


@_icon("save")
def _save(d, x, y, s, col):
    r = s * 0.34
    d.polygon(
        [(x - r, y - r), (x + r * 0.6, y - r), (x + r, y - r * 0.6), (x + r, y + r), (x - r, y + r)],
        outline=col,
        width=_w(s),
    )
    d.rectangle((x - r * 0.5, y - r, x + r * 0.35, y - r * 0.35), outline=col, width=_w(s))
    d.rectangle((x - r * 0.55, y + r * 0.2, x + r * 0.55, y + r), outline=col, width=_w(s))


@_icon("mask")
def _mask(d, x, y, s, col):
    r = s * 0.34
    d.rounded_rectangle((x - r, y - r, x + r, y + r), radius=r * 0.25, outline=col, width=_w(s))
    d.ellipse((x - r * 0.5, y - r * 0.5, x + r * 0.5, y + r * 0.5), fill=col)


@_icon("link")
def _link(d, x, y, s, col):
    r = s * 0.3
    w = _w(s)
    d.rounded_rectangle((x - r, y - r * 0.35, x + r * 0.1, y + r * 0.35), radius=r * 0.35, outline=col, width=w)
    d.rounded_rectangle((x - r * 0.1, y - r * 0.35, x + r, y + r * 0.35), radius=r * 0.35, outline=col, width=w)


@_icon("transform")
def _transform(d, x, y, s, col):
    r = s * 0.3
    w = _w(s)
    d.rectangle((x - r, y - r, x + r, y + r), outline=col, width=w)
    h = s * 0.07
    for cx, cy in ((x - r, y - r), (x + r, y - r), (x - r, y + r), (x + r, y + r)):
        d.rectangle((cx - h, cy - h, cx + h, cy + h), fill=col)
    d.arc((x - r * 0.55, y - r * 0.55, x + r * 0.55, y + r * 0.55), start=200, end=340, fill=col, width=w)
    tip = (x + r * 0.55 * math.cos(math.radians(340)), y + r * 0.55 * math.sin(math.radians(340)))
    _arrow_head(d, tip, math.radians(340 + 90), s, col, s * 0.1)


@_icon("stamp")
def _stamp(d, x, y, s, col):
    r = s * 0.3
    w = _w(s)
    d.rounded_rectangle((x - r * 0.3, y - r, x + r * 0.3, y - r * 0.1), radius=r * 0.15, outline=col, width=w)
    d.rounded_rectangle((x - r, y - r * 0.1, x + r, y + r * 0.45), radius=r * 0.2, outline=col, width=w)
    _line(d, [(x - r * 0.8, y + r), (x + r * 0.8, y + r)], s, col)


@_icon("heal")
def _heal(d, x, y, s, col):
    r = s * 0.3
    w = _w(s)
    d.rounded_rectangle((x - r * 0.35, y - r, x + r * 0.35, y + r), radius=r * 0.35, outline=col, width=w)
    d.rounded_rectangle((x - r, y - r * 0.35, x + r, y + r * 0.35), radius=r * 0.35, outline=col, width=w)
    d.rectangle((x - r * 0.33, y - r * 0.33, x + r * 0.33, y + r * 0.33), fill=col)


@_icon("adjust")
def _adjust(d, x, y, s, col):
    r = s * 0.32
    d.ellipse((x - r, y - r, x + r, y + r), outline=col, width=_w(s))
    d.pieslice((x - r, y - r, x + r, y + r), start=270, end=90, fill=col)


@_icon("fx")
def _fx(d, x, y, s, col):
    fnt = None
    try:
        from .theme import font

        fnt = font(int(s * 0.55), "medium")
    except Exception:  # noqa: BLE001 - the icon degrades to a box
        pass
    if fnt is not None:
        d.text((x, y), "fx", font=fnt, fill=col, anchor="mm")
    else:
        d.rectangle((x - s * 0.3, y - s * 0.3, x + s * 0.3, y + s * 0.3), outline=col, width=_w(s))


@_icon("chevron-right")
def _chevron_right(d, x, y, s, col):
    r = s * 0.18
    _line(d, [(x - r * 0.6, y - r), (x + r * 0.6, y), (x - r * 0.6, y + r)], s, col)


@_icon("folder")
def _folder(d, x, y, s, col):
    r = s * 0.32
    w = _w(s)
    d.rounded_rectangle((x - r, y - r * 0.55, x + r, y + r * 0.7), radius=r * 0.2, outline=col, width=w)
    _line(d, [(x - r, y - r * 0.55), (x - r, y - r * 0.85), (x - r * 0.3, y - r * 0.85), (x - r * 0.05, y - r * 0.55)], s, col)


@_icon("cube")
def _cube(d, x, y, s, col):
    r = s * 0.4
    w = _w(s)
    top = (x, y - r)
    left, right, bottom = (x - r * 0.87, y - r * 0.5), (x + r * 0.87, y - r * 0.5), (x, y)
    lb, rb, bb = (x - r * 0.87, y + r * 0.5), (x + r * 0.87, y + r * 0.5), (x, y + r)
    d.polygon([top, right, bottom, left], outline=col, width=w)
    d.polygon([left, bottom, bb, lb], outline=col, width=w)
    d.polygon([bottom, right, rb, bb], outline=col, width=w)
