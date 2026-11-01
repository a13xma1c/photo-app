"""Small colour helpers shared by the core and the GUI."""

from __future__ import annotations

from collections.abc import Sequence

RGB = tuple[int, int, int]
RGBA = tuple[int, int, int, int]


def clamp(value: float, lo: float = 0, hi: float = 255) -> float:
    """Clamp *value* into the closed interval [lo, hi]."""
    return lo if value < lo else hi if value > hi else value


def parse_color(value: str | Sequence[float]) -> RGBA:
    """Return an RGBA tuple from ``#rgb``, ``#rrggbb``, ``#rrggbbaa`` or a tuple."""
    if isinstance(value, str):
        text = value.strip().lstrip("#")
        if len(text) == 3:
            text = "".join(ch * 2 for ch in text)
        if len(text) == 6:
            text += "ff"
        if len(text) != 8:
            raise ValueError(f"invalid colour {value!r}")
        return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4, 6))  # type: ignore[return-value]
    if len(value) == 3:
        r, g, b = value
        return int(r), int(g), int(b), 255
    if len(value) == 4:
        r, g, b, a = value
        return int(r), int(g), int(b), int(a)
    raise ValueError(f"invalid colour {value!r}")


def to_hex(color: str | Sequence[float], alpha: bool = False) -> str:
    """Format a colour tuple as ``#rrggbb`` (or ``#rrggbbaa`` when *alpha*)."""
    rgba = parse_color(color)
    return "#" + "".join(f"{c:02x}" for c in (rgba if alpha else rgba[:3]))


def luminance(color: tuple) -> float:
    """Relative luminance (0-255) using the Rec. 601 weights."""
    r, g, b = parse_color(color)[:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def contrast_text_color(background: tuple | str) -> str:
    """Return black or white, whichever is more readable on *background*."""
    return "#000000" if luminance(parse_color(background)) > 150 else "#ffffff"
