"""Design tokens: colours, typography and metrics for the glass UI.

Everything in the interface is rendered with Pillow onto a single Tk canvas,
so colours are RGB(A) tuples rather than Tk colour strings.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

RGBA = tuple[int, int, int, int]
AnyFont = ImageFont.FreeTypeFont | ImageFont.ImageFont


@dataclass
class Palette:
    """Colours of one theme. ``PALETTE`` is the live instance; :func:`set_theme` copies a theme's values into it."""

    # workspace
    bg: tuple[int, int, int] = (20, 21, 25)
    checker: tuple[tuple[int, int, int], tuple[int, int, int]] = ((52, 53, 58), (40, 41, 46))
    # glass panels
    glass_tint: RGBA = (30, 32, 38, 168)  # colour laid over the blurred backdrop
    glass_sheen: int = 22  # alpha of the top highlight sweep
    glass_border: int = 70  # alpha of the 1px edge highlight (top), fades downwards
    glass_shadow: int = 110
    # controls
    fill: RGBA = (255, 255, 255, 20)  # neutral control fill
    fill_hover: RGBA = (255, 255, 255, 36)
    fill_pressed: RGBA = (255, 255, 255, 56)
    fill_selected: RGBA = (255, 255, 255, 48)
    accent: RGBA = (82, 140, 255, 255)
    accent_hover: RGBA = (110, 160, 255, 255)
    accent_soft: RGBA = (82, 140, 255, 70)
    danger: RGBA = (240, 90, 96, 255)
    track: RGBA = (255, 255, 255, 38)
    knob: RGBA = (250, 250, 252, 255)
    knob_shadow: RGBA = (0, 0, 0, 90)
    # text
    fg: RGBA = (244, 245, 248, 255)
    fg_muted: RGBA = (168, 172, 182, 255)
    fg_dim: RGBA = (110, 114, 124, 255)
    on_accent: RGBA = (255, 255, 255, 255)
    # overlays
    crop_dim: RGBA = (0, 0, 0, 120)


@dataclass
class Metrics:
    """Pixel sizes of the UI at the current scale (see :func:`set_scale`).

    The defaults are the 1× values; ``set_scale`` multiplies every field in
    place so all modules that imported ``METRICS`` see the scaled numbers.
    """

    panel_radius: int = 22
    panel_pad: int = 16
    panel_gap: int = 14  # distance between floating panels and the window edge
    control_radius: int = 12
    pill_radius: int = 18
    button_h: int = 34
    icon_button: int = 38
    icon_size: int = 18
    row_h: int = 32
    slider_h: int = 48
    caption_h: int = 26
    segmented_h: int = 34
    blur: int = 16
    font_size: int = 13  # logical points - scaled by font()
    small_font_size: int = 11
    title_font_size: int = 15


DARK = Palette()
LIGHT = Palette(
    bg=(234, 235, 239),
    checker=((214, 215, 220), (199, 200, 206)),
    glass_tint=(250, 250, 252, 190),
    glass_sheen=36,
    glass_border=150,
    glass_shadow=70,
    fill=(0, 0, 0, 14),
    fill_hover=(0, 0, 0, 26),
    fill_pressed=(0, 0, 0, 42),
    fill_selected=(0, 0, 0, 34),
    accent=(36, 104, 236, 255),
    accent_hover=(60, 128, 255, 255),
    accent_soft=(36, 104, 236, 60),
    danger=(214, 58, 70, 255),
    track=(0, 0, 0, 40),
    knob=(255, 255, 255, 255),
    knob_shadow=(0, 0, 0, 70),
    fg=(26, 28, 34, 255),
    fg_muted=(92, 96, 106, 255),
    fg_dim=(150, 154, 164, 255),
    on_accent=(255, 255, 255, 255),
    crop_dim=(0, 0, 0, 100),
)
HIGH_CONTRAST = Palette(
    bg=(0, 0, 0),
    checker=((70, 70, 70), (40, 40, 40)),
    glass_tint=(10, 10, 12, 240),
    glass_sheen=0,
    glass_border=200,
    glass_shadow=0,
    fill=(255, 255, 255, 40),
    fill_hover=(255, 255, 255, 70),
    fill_pressed=(255, 255, 255, 100),
    fill_selected=(255, 255, 255, 90),
    accent=(255, 210, 0, 255),
    accent_hover=(255, 230, 80, 255),
    accent_soft=(255, 210, 0, 110),
    danger=(255, 90, 90, 255),
    track=(255, 255, 255, 90),
    knob=(255, 255, 255, 255),
    knob_shadow=(0, 0, 0, 0),
    fg=(255, 255, 255, 255),
    fg_muted=(230, 230, 230, 255),
    fg_dim=(180, 180, 180, 255),
    on_accent=(0, 0, 0, 255),
    crop_dim=(0, 0, 0, 170),
)
THEMES: dict[str, Palette] = {"dark": DARK, "light": LIGHT, "high-contrast": HIGH_CONTRAST}
THEME_NAMES: dict[str, str] = {"dark": "Dark Theme", "light": "Light Theme", "high-contrast": "High Contrast"}

_BASE_METRICS = Metrics()
PALETTE = Palette()
METRICS = Metrics()
THEME = "dark"


def set_theme(name: str) -> str:
    """Switch the live palette to the theme *name* (unknown names fall back to dark); returns the name in effect."""
    global THEME
    if name not in THEMES:
        name = "dark"
    source = THEMES[name]
    for field_name in Palette.__dataclass_fields__:
        setattr(PALETTE, field_name, getattr(source, field_name))
    THEME = name
    return name


def is_light_theme() -> bool:
    return sum(PALETTE.bg) > 3 * 128


SCALE = 1.0
MIN_SCALE, MAX_SCALE = 1.0, 4.0


def set_scale(scale: float) -> float:
    """Set the UI scale factor (1.0 = 96 dpi, 2.0 = HiDPI/"Retina") and rescale all metrics."""
    global SCALE
    scale = min(MAX_SCALE, max(MIN_SCALE, float(scale)))
    SCALE = scale
    for name in Metrics.__dataclass_fields__:
        if name.endswith("font_size"):
            setattr(METRICS, name, getattr(_BASE_METRICS, name))  # fonts scale when loaded
        else:
            setattr(METRICS, name, max(1, round(getattr(_BASE_METRICS, name) * scale)))
    return scale


def px(value: float) -> int:
    """Convert a 1× design pixel value to device pixels at the current scale."""
    return int(round(value * SCALE))


BUNDLED_FONT_DIR = Path(__file__).with_name("fonts")
BUNDLED_FONTS = {"regular": "DejaVuSans", "medium": "DejaVuSans-Bold"}  # shipped with PixelForge (see fonts/LICENSE-DejaVu.txt)
# Set PIXELFORGE_SYSTEM_FONTS=1 to prefer the platform's UI font over the bundled one.
USE_SYSTEM_FONTS = os.environ.get("PIXELFORGE_SYSTEM_FONTS", "") not in ("", "0", "false", "no")

_FONT_CANDIDATES = {
    "regular": [
        "Inter-Regular",
        "Inter",
        "SFProText-Regular",
        "SF-Pro-Text-Regular",
        "HelveticaNeue",
        "Helvetica Neue",
        "segoeui",
        "Ubuntu-R",
        "Cantarell-Regular",
        "NotoSans-Regular",
        "DejaVuSans",
        "LiberationSans-Regular",
        "Poppins-Regular",
    ],
    "medium": [
        "Inter-Medium",
        "Inter-SemiBold",
        "SFProText-Semibold",
        "HelveticaNeue-Medium",
        "segoeuisb",
        "seguisb",
        "Ubuntu-M",
        "Cantarell-Bold",
        "NotoSans-Medium",
        "DejaVuSans-Bold",
        "LiberationSans-Bold",
        "Poppins-Medium",
    ],
}
_FONT_DIRS = {
    "darwin": ["/System/Library/Fonts", "/Library/Fonts", "~/Library/Fonts"],
    "win32": [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")],
}.get(sys.platform, ["/usr/share/fonts", "/usr/local/share/fonts", "~/.fonts", "~/.local/share/fonts"])


@lru_cache(maxsize=1)
def _font_files() -> dict[str, str]:
    """Font files by lower-case stem: the bundled fonts first (under ``bundled-<weight>`` too), then the system's."""
    files: dict[str, str] = {}
    for weight, stem in BUNDLED_FONTS.items():
        path = BUNDLED_FONT_DIR / f"{stem}.ttf"
        if path.is_file():
            files[f"bundled-{weight}"] = str(path)
            files.setdefault(stem.lower(), str(path))
    for base in _FONT_DIRS:
        root = Path(os.path.expanduser(base))
        if root.is_dir():
            for path in root.rglob("*"):
                if path.suffix.lower() in (".ttf", ".otf", ".ttc") and path.is_file():
                    files.setdefault(path.stem.lower(), str(path))
    return files


def _candidates(weight: str) -> list[str]:
    names = list(_FONT_CANDIDATES.get(weight, _FONT_CANDIDATES["regular"]))
    bundled = f"bundled-{weight if weight in BUNDLED_FONTS else 'regular'}"
    return names + [bundled] if USE_SYSTEM_FONTS else [bundled] + names


def font(size: int, weight: str = "regular") -> AnyFont:
    """Best available UI font at *size* logical points (scaled by the UI scale)."""
    return _load_font(max(1, int(round(size * SCALE))), weight)


def font_size(fnt: AnyFont) -> int:
    """Pixel size of a font (Pillow's bitmap fallback font has no size attribute)."""
    return int(getattr(fnt, "size", METRICS.font_size))


@lru_cache(maxsize=64)
def _load_font(size: int, weight: str) -> AnyFont:
    files = _font_files()
    for name in _candidates(weight):
        path = files.get(name.lower())
        if not path:
            continue
        try:
            fnt = ImageFont.truetype(path, size)
        except OSError:
            continue
        if weight == "medium":  # variable fonts (SF, Inter) carry their weights as named instances
            for variation in (b"Semibold", b"Medium", b"Bold"):
                try:
                    fnt.set_variation_by_name(variation)
                    break
                except (OSError, ValueError, AttributeError):
                    continue
        return fnt
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


# Fonts tried, in order, for characters the UI font cannot display (⌘ × • − …).
_SYMBOL_CANDIDATES = [
    "bundled-regular",
    "Apple Symbols",
    "AppleSymbols",
    "Menlo",
    "SFNSMono",
    "seguisym",
    "SegoeUISymbol",
    "segoeui",
    "DejaVuSans",
    "NotoSansSymbols2-Regular",
    "NotoSansSymbols-Regular",
    "NotoSansMath-Regular",
    "Symbola",
    "FreeSans",
    "Arial Unicode",
    "LiberationSans-Regular",
]
_glyph_cache: dict[tuple[int, str], bool] = {}
_notdef_cache: dict[int, bytes | None] = {}


def _notdef(fnt: AnyFont) -> bytes | None:
    """Pixels of the font's "missing glyph" box (None when the font draws nothing for it)."""
    key = id(fnt)
    if key not in _notdef_cache:
        try:
            mask = fnt.getmask("\U0010fffd")
            _notdef_cache[key] = bytes(mask) if mask.getbbox() else None
        except Exception:  # bitmap fallback font
            _notdef_cache[key] = None
    return _notdef_cache[key]


def has_glyph(fnt: AnyFont, char: str) -> bool:
    """True when *fnt* can draw *char* (rather than a blank or the .notdef box)."""
    if not char or char.isspace():
        return True
    key = (id(fnt), char)
    cached = _glyph_cache.get(key)
    if cached is None:
        try:
            mask = fnt.getmask(char)
            if not mask.getbbox():
                cached = False
            else:
                notdef = _notdef(fnt)
                cached = notdef is None or bytes(mask) != notdef
        except Exception:
            cached = True
        _glyph_cache[key] = cached
    return cached


@lru_cache(maxsize=32)
def _symbol_fonts(size: int) -> tuple[AnyFont, ...]:
    files = _font_files()
    fonts: list[AnyFont] = []
    for name in _SYMBOL_CANDIDATES:
        path = files.get(name.lower())
        if path:
            try:
                fonts.append(ImageFont.truetype(path, size))
            except OSError:
                continue
    return tuple(fonts)


def font_for(char: str, fnt: AnyFont) -> AnyFont:
    """The font to draw *char* with: *fnt* itself, or the first symbol font that has the glyph."""
    if has_glyph(fnt, char):
        return fnt
    for candidate in _symbol_fonts(font_size(fnt)):
        if has_glyph(candidate, char):
            return candidate
    return fnt


def font_runs(text: str, fnt: AnyFont) -> list[tuple[str, AnyFont]]:
    """Split *text* into runs that share a font, substituting symbol fonts for missing glyphs."""
    runs: list[tuple[str, AnyFont]] = []
    for char in text:
        use = font_for(char, fnt)
        if runs and runs[-1][1] is use:
            runs[-1] = (runs[-1][0] + char, use)
        else:
            runs.append((char, use))
    return runs


def text_width(text: str, fnt: AnyFont) -> float:
    """Advance width of *text* in pixels, honouring glyph fallback."""
    return sum(run_font.getlength(run) for run, run_font in font_runs(text, fnt))


def text_size(text: str, fnt: AnyFont) -> tuple[int, int]:
    """Width and height of *text* in pixels."""
    _left, top, _right, bottom = fnt.getbbox(text or " ")
    return int(round(text_width(text or " ", fnt))), int(bottom - top)


def with_alpha(color: RGBA, alpha: int) -> RGBA:
    return (color[0], color[1], color[2], max(0, min(255, alpha)))


def mix(a: RGBA, b: RGBA, t: float) -> RGBA:
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(4))  # type: ignore[return-value]
