"""A small retained-mode widget kit rendered with Pillow into glass panels.

Widgets know how tall they want to be (``measure``), where they ended up
(``rect``, in panel-content coordinates), how to draw themselves and how to
react to pointer/keyboard events delivered by the :class:`View`.  Because the
whole UI is drawn by us, alignment, hover states, tooltips and scrolling behave
identically on every platform.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..core.effects import is_empty as effects_empty
from . import glass
from .theme import METRICS, PALETTE, RGBA, font, font_size, px, text_size, text_width

Rect = tuple[int, int, int, int]


@dataclass
class DrawContext:
    """Per-render state shared by all widgets of a panel."""

    hover: Widget | None = None
    pressed: Widget | None = None
    focused: Widget | None = None
    offset: tuple[int, int] = (0, 0)  # translation applied to widget rects (scrolling)

    def box(self, widget: Widget) -> Rect:
        x0, y0, x1, y1 = widget.rect
        dx, dy = self.offset
        return x0 + dx, y0 + dy, x1 + dx, y1 + dy

    def is_hover(self, widget: Widget) -> bool:
        return self.hover is widget and widget.enabled

    def is_pressed(self, widget: Widget) -> bool:
        return self.pressed is widget and widget.enabled


class Widget:
    """Base class: a rectangle that can measure, draw and receive events."""

    width: int | None = None  # fixed width inside a Row (None = flexible)
    flex: int = 1
    tooltip: str = ""
    cursor: str = ""
    focusable: bool = False

    def __init__(self) -> None:
        self.rect: Rect = (0, 0, 0, 0)
        self.panel: Panel | None = None
        self.enabled = True
        self.visible = True
        self.height = METRICS.row_h

    # layout ----------------------------------------------------------------
    def measure(self, width: int) -> int:
        return self.height

    def preferred_width(self) -> int:
        """Natural width, used by panels that shrink-wrap their content."""
        return self.width or 0

    def layout(self, x: int, y: int, w: int, h: int) -> None:
        self.rect = (x, y, x + w, y + h)

    def attach(self, panel: Panel) -> None:
        self.panel = panel
        for child in self.children():
            child.attach(panel)

    def children(self) -> Sequence[Widget]:
        return ()

    def descendants(self):
        yield self
        for child in self.children():
            yield from child.descendants()

    def contains(self, x: float, y: float) -> bool:
        x0, y0, x1, y1 = self.rect
        return x0 <= x < x1 and y0 <= y < y1

    def hit(self, x: float, y: float) -> Widget | None:
        if not self.visible or not self.contains(x, y):
            return None
        for child in reversed(self.children()):
            found = child.hit(x, y)
            if found is not None:
                return found
        return self

    def invalidate(self) -> None:
        if self.panel is not None:
            self.panel.invalidate()

    # drawing ----------------------------------------------------------------
    def draw(self, img: Image.Image, ctx: DrawContext) -> None:
        for child in self.children():
            if child.visible:
                child.draw(img, ctx)

    # events (coordinates are panel-content coordinates) --------------------
    def on_press(self, x: float, y: float) -> None:
        pass

    def on_drag(self, x: float, y: float) -> None:
        pass

    def on_release(self, x: float, y: float, inside: bool) -> None:
        pass

    def on_double(self, x: float, y: float) -> None:
        pass

    def on_wheel(self, dx: float, dy: float) -> bool:
        return False

    def on_key(self, keysym: str, char: str, mods: frozenset[str] = frozenset()) -> bool:
        return False

    def on_focus(self, focused: bool) -> None:
        pass


# -------------------------------------------------------------- containers --
class Column(Widget):
    """Stacks children vertically."""

    def __init__(self, children: Sequence[Widget] = (), gap: int = 0) -> None:
        super().__init__()
        self._children: list[Widget] = list(children)
        self.gap = px(gap)

    def children(self) -> Sequence[Widget]:
        return self._children

    def add(self, *widgets: Widget) -> None:
        self._children.extend(widgets)
        if self.panel is not None:
            for w in widgets:
                w.attach(self.panel)
        self.invalidate()

    def clear(self) -> None:
        self._children.clear()
        self.invalidate()

    def measure(self, width: int) -> int:
        visible = [c for c in self._children if c.visible]
        return sum(c.measure(width) for c in visible) + self.gap * max(0, len(visible) - 1)

    def preferred_width(self) -> int:
        return max((c.preferred_width() for c in self._children if c.visible), default=0)

    def layout(self, x: int, y: int, w: int, h: int) -> None:
        super().layout(x, y, w, h)
        cy = y
        for child in self._children:
            if not child.visible:
                continue
            ch = child.measure(w)
            child.layout(x, cy, w, ch)
            cy += ch + self.gap


class Row(Widget):
    """Places children side by side; fixed-width children keep their width, the rest share the remainder."""

    def __init__(self, children: Sequence[Widget] = (), gap: int = 8, height: int | None = None, align: str = "fill") -> None:
        super().__init__()
        self._children: list[Widget] = list(children)
        self.gap = px(gap)
        self._height = None if height is None else px(height)
        self.align = align  # "fill" | "left" | "center" | "right"

    def children(self) -> Sequence[Widget]:
        return self._children

    def add(self, *widgets: Widget) -> None:
        self._children.extend(widgets)
        if self.panel is not None:
            for w in widgets:
                w.attach(self.panel)
        self.invalidate()

    def clear(self) -> None:
        self._children.clear()
        self.invalidate()

    def measure(self, width: int) -> int:
        if self._height is not None:
            return self._height
        visible = [c for c in self._children if c.visible]
        return max((c.measure(width) for c in visible), default=0)

    def preferred_width(self) -> int:
        visible = [c for c in self._children if c.visible]
        return sum(c.preferred_width() for c in visible) + self.gap * max(0, len(visible) - 1)

    def layout(self, x: int, y: int, w: int, h: int) -> None:
        super().layout(x, y, w, h)
        visible = [c for c in self._children if c.visible]
        if not visible:
            return
        fixed = sum(c.width for c in visible if c.width is not None)
        flex_total = sum(c.flex for c in visible if c.width is None)
        gaps = self.gap * (len(visible) - 1)
        remaining = max(0, w - fixed - gaps)
        if self.align == "fill" or flex_total:
            cx = x
        else:
            total = fixed + gaps
            cx = x if self.align == "left" else x + (w - total) // 2 if self.align == "center" else x + w - total
        for child in visible:
            cw = child.width if child.width is not None else (remaining * child.flex // flex_total if flex_total else 0)
            ch = child.measure(cw)
            child.layout(cx, y + (h - ch) // 2, cw, ch)
            cx += cw + self.gap


class FlowRow(Widget):
    """Like :class:`Row`, but wraps children onto new lines when they don't fit; each line is centred."""

    def __init__(self, children: Sequence[Widget] = (), gap: int = 10, line_gap: int = 6, align: str = "center") -> None:
        super().__init__()
        self._children: list[Widget] = list(children)
        self.gap = px(gap)
        self.line_gap = px(line_gap)
        self.align = align

    def children(self) -> Sequence[Widget]:
        return self._children

    def add(self, *widgets: Widget) -> None:
        self._children.extend(widgets)
        if self.panel is not None:
            for w in widgets:
                w.attach(self.panel)
        self.invalidate()

    def clear(self) -> None:
        self._children.clear()
        self.invalidate()

    def _lines(self, width: int) -> list[list[Widget]]:
        lines: list[list[Widget]] = [[]]
        used = 0
        for child in self._children:
            if not child.visible:
                continue
            w = child.preferred_width()
            if lines[-1] and used + self.gap + w > width:
                lines.append([])
                used = 0
            if lines[-1]:
                used += self.gap
            lines[-1].append(child)
            used += w
        return [line for line in lines if line]

    def _line_height(self, line: list[Widget], width: int) -> int:
        return max((c.measure(c.preferred_width() or width) for c in line), default=0)

    def measure(self, width: int) -> int:
        lines = self._lines(width)
        return sum(self._line_height(line, width) for line in lines) + self.line_gap * max(0, len(lines) - 1)

    def preferred_width(self) -> int:
        visible = [c for c in self._children if c.visible]
        return sum(c.preferred_width() for c in visible) + self.gap * max(0, len(visible) - 1)

    def layout(self, x: int, y: int, w: int, h: int) -> None:
        super().layout(x, y, w, h)
        cy = y
        for line in self._lines(w):
            lh = self._line_height(line, w)
            total = sum(min(c.preferred_width(), w) for c in line) + self.gap * (len(line) - 1)
            cx = x if self.align == "left" else x + max(0, (w - total) // 2)
            for child in line:
                cw = min(child.preferred_width() or w, w)  # a single oversized child shrinks to the available width
                ch = child.measure(cw)
                child.layout(cx, cy + (lh - ch) // 2, cw, ch)
                cx += cw + self.gap
            cy += lh + self.line_gap


class Spacer(Widget):
    def __init__(self, height: int = 8, width: int | None = None) -> None:
        super().__init__()
        self.height = px(height)
        self.width = None if width is None else px(width)
        self.flex = 1 if width is None else 0


class Separator(Widget):
    def __init__(self, margin: int = 6) -> None:
        super().__init__()
        self.height = px(margin) * 2 + px(1)

    def draw(self, img: Image.Image, ctx: DrawContext) -> None:
        x0, y0, x1, y1 = ctx.box(self)
        y = (y0 + y1) // 2
        glass.fill_rounded(img, (x0 + px(4), y, x1 - px(4), y + px(1)), 0, (255, 255, 255, 28))


# ------------------------------------------------------------------- text --
class Label(Widget):
    @staticmethod
    def styles() -> dict[str, tuple[int, str, RGBA]]:
        return {
            "body": (METRICS.font_size, "regular", PALETTE.fg),
            "muted": (METRICS.font_size, "regular", PALETTE.fg_muted),
            "caption": (METRICS.small_font_size, "medium", PALETTE.fg_dim),
            "title": (METRICS.title_font_size, "medium", PALETTE.fg),
            "small": (METRICS.small_font_size, "regular", PALETTE.fg_muted),
        }

    def __init__(
        self,
        text: str = "",
        style: str = "body",
        align: str = "left",
        height: int | None = None,
        uppercase: bool = False,
        width: int | None = None,
    ) -> None:
        super().__init__()
        self.width = None if width is None else px(width)
        self.flex = 0 if width else 1
        self.text = text
        self.style = style
        self.align = align
        self.uppercase = uppercase
        size, weight, self.color = self.styles()[style]
        self.font = font(size, weight)
        self.height = px(height) if height else (METRICS.caption_h if style == "caption" else METRICS.row_h)

    def set_text(self, text: str) -> None:
        if text != self.text:
            self.text = text
            self.invalidate()

    def set_color(self, color: RGBA | None) -> None:
        color = color or self.styles()[self.style][2]
        if color != self.color:
            self.color = color
            self.invalidate()

    def preferred_width(self) -> int:
        if self.width is not None:
            return self.width
        return max((text_size(line, self.font)[0] for line in self.text.split("\n")), default=0) + px(8)

    def measure(self, width: int) -> int:
        lines = self.text.count("\n") + 1
        if lines == 1:
            return self.height
        return lines * int(font_size(self.font) * 1.45) + px(8)

    def draw(self, img: Image.Image, ctx: DrawContext) -> None:
        x0, y0, x1, y1 = ctx.box(self)
        txt = self.text.upper() if self.uppercase else self.text
        if "\n" in txt:
            glass.text(img, (x0, y0 + px(4)), txt, self.font, self.color, "la")
            return
        txt = glass.ellipsize(txt, self.font, x1 - x0)
        cy = (y0 + y1) / 2
        if self.align == "center":
            glass.text(img, ((x0 + x1) / 2, cy), txt, self.font, self.color, "mm")
        elif self.align == "right":
            glass.text(img, (x1, cy), txt, self.font, self.color, "rm")
        else:
            glass.text(img, (x0, cy), txt, self.font, self.color, "lm")


# ---------------------------------------------------------------- buttons --
class Button(Widget):
    """Pill-shaped push button with optional icon; ``primary`` uses the accent colour."""

    cursor = "hand"

    def __init__(
        self,
        text: str,
        command: Callable[[], None] | None = None,
        icon: str | None = None,
        primary: bool = False,
        width: int | None = None,
        tooltip: str = "",
        danger: bool = False,
        small: bool = False,
    ) -> None:
        super().__init__()
        self.text, self.command, self.icon, self.primary, self.danger = text, command, icon, primary, danger
        self.tooltip = tooltip
        self.small = small
        self.font = font(METRICS.small_font_size if small else METRICS.font_size, "medium")
        self.height = px(22) if small else METRICS.button_h
        self.icon_px = px(12) if small else METRICS.icon_size
        if width is None:
            tw = text_size(text, self.font)[0] if text else 0
            width = tw + (px(16) if small else px(28)) + (self.icon_px + px(6) if icon else 0)
        else:
            width = px(width)
        self.width = width
        self.flex = 0

    def draw(self, img: Image.Image, ctx: DrawContext) -> None:
        x0, y0, x1, y1 = ctx.box(self)
        if self.primary:
            fill = PALETTE.accent_hover if ctx.is_hover(self) else PALETTE.accent
            if ctx.is_pressed(self):
                fill = (60, 110, 220, 255)
            color = PALETTE.on_accent
        else:
            fill = PALETTE.fill_pressed if ctx.is_pressed(self) else PALETTE.fill_hover if ctx.is_hover(self) else PALETTE.fill
            color = PALETTE.danger if self.danger else PALETTE.fg
        if not self.enabled:
            fill = (255, 255, 255, 10)
            color = PALETTE.fg_dim
        glass.fill_rounded(img, (x0, y0, x1, y1), (y1 - y0) // 2, fill)
        cy = (y0 + y1) / 2
        content_w = (text_size(self.text, self.font)[0] if self.text else 0) + (self.icon_px + px(6) if self.icon else 0)
        cx = (x0 + x1) / 2 - content_w / 2
        if self.icon:
            glass.paste(img, glass.icon(self.icon, self.icon_px, color), (int(cx), int(cy - self.icon_px / 2)))
            cx += self.icon_px + px(6)
        if self.text:
            glass.text(img, (cx, cy), self.text, self.font, color, "lm")

    def on_release(self, x: float, y: float, inside: bool) -> None:
        if inside and self.enabled and self.command:
            self.command()


class IconButton(Widget):
    """Square rounded button showing only an icon; ``active`` renders the selected (tool) look."""

    cursor = "hand"

    def __init__(
        self,
        icon: str,
        command: Callable[[], None] | None = None,
        tooltip: str = "",
        size: int | None = None,
        icon_size: int | None = None,
        active: bool = False,
    ) -> None:
        super().__init__()
        self.icon, self.command, self.tooltip = icon, command, tooltip
        self.active = active
        self.height = METRICS.icon_button if size is None else px(size)
        self.width = self.height
        self.flex = 0
        self.icon_size = METRICS.icon_size if icon_size is None else px(icon_size)

    def draw(self, img: Image.Image, ctx: DrawContext) -> None:
        x0, y0, x1, y1 = ctx.box(self)
        if self.active:
            fill: RGBA | None = PALETTE.accent if not ctx.is_pressed(self) else (60, 110, 220, 255)
            color = PALETTE.on_accent
        else:
            fill = PALETTE.fill_pressed if ctx.is_pressed(self) else PALETTE.fill_hover if ctx.is_hover(self) else None
            color = PALETTE.fg if self.enabled else PALETTE.fg_dim
        if fill:
            glass.fill_rounded(img, (x0, y0, x1, y1), METRICS.control_radius, fill)
        ico = glass.icon(self.icon, self.icon_size, color)
        glass.paste(img, ico, ((x0 + x1 - ico.width) // 2, (y0 + y1 - ico.height) // 2))

    def on_release(self, x: float, y: float, inside: bool) -> None:
        if inside and self.enabled and self.command:
            self.command()


class Segmented(Widget):
    """Segmented control (tabs): one pill with the selected segment highlighted."""

    cursor = "hand"

    def __init__(
        self, options: Sequence[str], index: int = 0, command: Callable[[int], None] | None = None, width: int | None = None
    ) -> None:
        super().__init__()
        self.options = list(options)
        self.index = index
        self.command = command
        self.font = font(METRICS.font_size, "medium")
        self.height = METRICS.segmented_h
        self.width = None if width is None else px(width)
        self.flex = 0 if width else 1
        self._hover_index: int | None = None

    def _segment_at(self, x: float) -> int:
        x0, _y0, x1, _y1 = self.rect
        inset = px(2)
        seg_w = (x1 - x0 - 2 * inset) / max(1, len(self.options))
        return int(min(len(self.options) - 1, max(0, (x - x0 - inset) // seg_w)))

    def draw(self, img: Image.Image, ctx: DrawContext) -> None:
        x0, y0, x1, y1 = ctx.box(self)
        inset = px(2)
        glass.fill_rounded(img, (x0, y0, x1, y1), (y1 - y0) // 2, PALETTE.fill)
        seg_w = (x1 - x0 - 2 * inset) / max(1, len(self.options))
        for i, label in enumerate(self.options):
            sx0 = x0 + inset + int(i * seg_w)
            sx1 = x0 + inset + int((i + 1) * seg_w)
            if i == self.index:
                glass.fill_rounded(img, (sx0, y0 + inset, sx1, y1 - inset), (y1 - y0) // 2 - inset, PALETTE.fill_selected)
            elif ctx.is_hover(self) and self._hover_index == i:
                glass.fill_rounded(img, (sx0, y0 + inset, sx1, y1 - inset), (y1 - y0) // 2 - inset, (255, 255, 255, 14))
            color = PALETTE.fg if i == self.index else PALETTE.fg_muted
            glass.text(img, ((sx0 + sx1) / 2, (y0 + y1) / 2), label, self.font, color, "mm")

    def set_hover_position(self, x: float) -> None:
        idx = self._segment_at(x)
        if idx != self._hover_index:
            self._hover_index = idx
            self.invalidate()

    def select(self, index: int) -> None:
        if index != self.index:
            self.index = index
            self.invalidate()
            if self.command:
                self.command(index)

    def on_release(self, x: float, y: float, inside: bool) -> None:
        if inside:
            self.select(self._segment_at(x))


class Toggle(Widget):
    """Label with a switch on the right."""

    cursor = "hand"

    def __init__(self, text: str, value: bool = False, command: Callable[[bool], None] | None = None) -> None:
        super().__init__()
        self.text, self.value, self.command = text, value, command
        self.font = font(METRICS.font_size)
        self.height = METRICS.row_h

    def draw(self, img: Image.Image, ctx: DrawContext) -> None:
        x0, y0, x1, y1 = ctx.box(self)
        cy = (y0 + y1) / 2
        glass.text(img, (x0, cy), self.text, self.font, PALETTE.fg, "lm")
        w, h = px(38), px(22)
        sx0, sy0 = x1 - w, int(cy - h / 2)
        fill = PALETTE.accent if self.value else PALETTE.fill_hover if ctx.is_hover(self) else PALETTE.fill
        glass.fill_rounded(img, (sx0, sy0, sx0 + w, sy0 + h), h // 2, fill)
        kx = sx0 + w - h / 2 if self.value else sx0 + h / 2
        glass.fill_circle(img, (kx, cy), h - px(6), PALETTE.knob)

    def on_release(self, x: float, y: float, inside: bool) -> None:
        if inside:
            self.value = not self.value
            self.invalidate()
            if self.command:
                self.command(self.value)


# ----------------------------------------------------------------- slider --
class Slider(Widget):
    """Labelled slider: value on the right, drag anywhere on the track, double-click resets.

    ``on_change`` fires continuously, ``on_commit`` once when the gesture ends.
    """

    cursor = "resize-h"

    def __init__(
        self,
        label: str,
        lo: float,
        hi: float,
        value: float | None = None,
        default: float = 0,
        on_change: Callable[[float], None] | None = None,
        on_commit: Callable[[], None] | None = None,
        integer: bool = True,
        fmt: str | None = None,
        tooltip: str = "",
        compact: bool = False,
    ) -> None:
        super().__init__()
        self.compact = compact  # single line: label · track · value (for tool option bars)
        self.label, self.lo, self.hi, self.default, self.integer = label, lo, hi, default, integer
        self.value = default if value is None else value
        self.on_change, self.on_commit = on_change, on_commit
        self.fmt = fmt or ("{:+d}" if integer and lo < 0 else "{:d}" if integer else "{:.2f}")
        self.tooltip = tooltip
        self.height = px(30) if compact else METRICS.slider_h
        self.font = font(METRICS.font_size)
        self.value_font = font(METRICS.font_size, "medium")
        self._dragging = False
        self._changed = False
        self._label_w = int(text_width(label, self.font)) + px(10) if compact else 0
        self._value_w = max(
            int(text_width(self.fmt.format(int(hi) if integer else hi), self.value_font)),
            int(text_width(self.fmt.format(int(lo) if integer else lo), self.value_font)),
        ) + px(8)

    # geometry ----------------------------------------------------------------
    def _track(self) -> tuple[int, int, int]:
        """(x0, x1, y) of the track line in content coordinates."""
        x0, y0, x1, y1 = self.rect
        if self.compact:
            return x0 + self._label_w + px(8), x1 - self._value_w - px(8), (y0 + y1) // 2
        return x0 + px(8), x1 - px(8), y1 - px(14)

    def _fraction(self) -> float:
        return 0.0 if self.hi == self.lo else (self.value - self.lo) / (self.hi - self.lo)

    def value_at(self, x: float) -> float:
        tx0, tx1, _ = self._track()
        frac = min(1.0, max(0.0, (x - tx0) / max(1, tx1 - tx0)))
        value = self.lo + frac * (self.hi - self.lo)
        return int(round(value)) if self.integer else value

    def format_value(self) -> str:
        value = int(self.value) if self.integer else self.value
        return self.fmt.format(value)

    # state -------------------------------------------------------------------
    def set(self, value: float, notify: bool = False) -> None:
        value = min(self.hi, max(self.lo, value))
        value = int(round(value)) if self.integer else value
        if value != self.value:
            self.value = value
            self.invalidate()
            if notify and self.on_change:
                self._changed = True
                self.on_change(value)

    def reset(self) -> None:
        self.set(self.default, notify=True)
        self._finish()

    def _finish(self) -> None:
        if self._changed and self.on_commit:
            self.on_commit()
        self._changed = False

    # drawing -----------------------------------------------------------------
    def draw(self, img: Image.Image, ctx: DrawContext) -> None:
        x0, y0, x1, y1 = ctx.box(self)
        dx, dy = ctx.offset
        tx0, tx1, ty = self._track()
        tx0, tx1, ty = tx0 + dx, tx1 + dx, ty + dy
        label_color = PALETTE.fg if self.enabled else PALETTE.fg_dim
        label_y = (y0 + y1) / 2 if self.compact else y0 + px(14)
        glass.text(img, (x0 + (0 if self.compact else px(8)), label_y), self.label, self.font, label_color, "lm")
        at_default = self.value == self.default
        value_color = PALETTE.fg_muted if at_default else PALETTE.fg
        glass.text(
            img,
            (x1 - (0 if self.compact else px(8)), label_y),
            self.format_value(),
            self.value_font,
            value_color if self.enabled else PALETTE.fg_dim,
            "rm",
        )
        # track
        half = max(1, px(2))
        glass.fill_rounded(img, (tx0, ty - half, tx1, ty + half), half, PALETTE.track)
        kx = tx0 + (tx1 - tx0) * self._fraction()
        # filled portion from the default position (centre for bipolar sliders) to the knob
        zero_x = tx0 + (tx1 - tx0) * (0.0 if self.hi == self.lo else (self.default - self.lo) / (self.hi - self.lo))
        if self.enabled and abs(kx - zero_x) >= 1:
            glass.fill_rounded(img, (int(min(kx, zero_x)), ty - half, int(max(kx, zero_x)), ty + half), half, PALETTE.accent)
        # knob
        active = ctx.is_hover(self) or self._dragging
        d = px(16) if active else px(14)
        glass.fill_circle(img, (kx, ty + px(1)), d + px(2), PALETTE.knob_shadow)
        glass.fill_circle(img, (kx, ty), d, PALETTE.knob if self.enabled else PALETTE.fg_dim)

    # events ------------------------------------------------------------------
    def in_value_zone(self, x: float, y: float) -> bool:
        """The value text (top right, or right end in compact mode): clicking it resets the slider."""
        x0, y0, x1, _y1 = self.rect
        if self.compact:
            return x > x1 - self._value_w
        return y < y0 + px(26) and x > x1 - px(64)

    def on_press(self, x: float, y: float) -> None:
        if not self.enabled:
            return
        x0, y0, _x1, _y1 = self.rect
        if self.compact:
            if self.in_value_zone(x, y):
                if self.value != self.default:
                    self.reset()
                return
            if x < x0 + self._label_w:
                return
        elif y < y0 + px(26):  # label row: only the value is interactive
            if self.in_value_zone(x, y) and self.value != self.default:
                self.reset()
            return
        self._dragging = True
        self.set(self.value_at(x), notify=True)
        self.invalidate()

    def on_drag(self, x: float, y: float) -> None:
        if self._dragging:
            self.set(self.value_at(x), notify=True)

    def on_release(self, x: float, y: float, inside: bool) -> None:
        if self._dragging:
            self._dragging = False
            self.invalidate()
            self._finish()

    def on_double(self, x: float, y: float) -> None:
        if self.enabled:
            self._dragging = False
            self.reset()

    def on_key(self, keysym: str, char: str, mods: frozenset[str] = frozenset()) -> bool:
        if not self.enabled:
            return False
        step = 1 if self.integer else (self.hi - self.lo) / 100
        if "shift" in mods:
            step *= 10
        if keysym in ("Left", "Down"):
            self.set(self.value - step, notify=True)
        elif keysym in ("Right", "Up"):
            self.set(self.value + step, notify=True)
        else:
            return False
        self._finish()
        return True


# -------------------------------------------------------------- list rows --
class ListItem(Widget):
    """Selectable row with text, optional leading icon and check mark."""

    cursor = "hand"

    def __init__(
        self,
        text: str,
        command: Callable[[], None] | None = None,
        selected: bool = False,
        dim: bool = False,
        icon: str | None = None,
        checked: bool = False,
        subtitle: str = "",
        trailing_icon: str | None = None,
        on_trailing: Callable[[], None] | None = None,
        trailing_tooltip: str = "",
    ) -> None:
        super().__init__()
        self.text, self.command, self.selected, self.dim = text, command, selected, dim
        self.icon, self.checked, self.subtitle = icon, checked, subtitle
        self.trailing_icon, self.on_trailing, self.trailing_tooltip = trailing_icon, on_trailing, trailing_tooltip
        self.font = font(METRICS.font_size, "medium" if selected else "regular")
        self.sub_font = font(METRICS.small_font_size)
        self.height = METRICS.row_h + (px(14) if subtitle else 0)

    def draw(self, img: Image.Image, ctx: DrawContext) -> None:
        x0, y0, x1, y1 = ctx.box(self)
        inset, radius = px(1), METRICS.control_radius - px(2)
        if self.selected:
            glass.fill_rounded(img, (x0, y0 + inset, x1, y1 - inset), radius, PALETTE.fill_selected)
        elif ctx.is_hover(self):
            glass.fill_rounded(img, (x0, y0 + inset, x1, y1 - inset), radius, PALETTE.fill)
        color = PALETTE.fg_dim if self.dim else PALETTE.fg
        tx = x0 + px(12)
        cy = (y0 + y1) / 2 if not self.subtitle else y0 + METRICS.row_h / 2 + px(2)
        small = px(16)
        if self.icon:
            ico = glass.icon(self.icon, small, color)
            glass.paste(img, ico, (tx, int(cy - small / 2)))
            tx += px(24)
        right = x1 - px(12) - (px(20) if self.checked else 0) - (px(28) if self.trailing_icon else 0)
        glass.text(img, (tx, cy), glass.ellipsize(self.text, self.font, right - tx), self.font, color, "lm")
        if self.trailing_icon and (ctx.is_hover(self) or self.selected):
            tx0, ty0, tx1, ty1 = self._trailing_rect()
            dx, dy = ctx.offset
            glass.fill_rounded(img, (tx0 + dx, ty0 + dy, tx1 + dx, ty1 + dy), px(8), PALETTE.fill)
            ico = glass.icon(self.trailing_icon, small, PALETTE.fg_muted)
            glass.paste(img, ico, ((tx0 + tx1) // 2 - small // 2 + dx, (ty0 + ty1) // 2 - small // 2 + dy))
        if self.subtitle:
            glass.text(
                img, (tx, cy + px(15)), glass.ellipsize(self.subtitle, self.sub_font, right - tx), self.sub_font, PALETTE.fg_muted, "lm"
            )
        if self.checked:
            check_x = x1 - px(28) - (px(28) if self.trailing_icon else 0)
            glass.paste(img, glass.icon("check", small, PALETTE.accent), (check_x, int(cy - small / 2)))

    def _trailing_rect(self) -> Rect:
        x0, y0, x1, y1 = self.rect
        size = px(24)
        cy = (y0 + y1) // 2 if not self.subtitle else y0 + METRICS.row_h // 2 + px(2)
        return x1 - px(6) - size, cy - size // 2, x1 - px(6), cy + size // 2

    def in_trailing_zone(self, x: float, y: float) -> bool:
        if not self.trailing_icon:
            return False
        tx0, ty0, tx1, ty1 = self._trailing_rect()
        return tx0 <= x < tx1 and ty0 <= y < ty1

    def on_release(self, x: float, y: float, inside: bool) -> None:
        if not (inside and self.enabled):
            return
        if self.in_trailing_zone(x, y) and self.on_trailing is not None:
            self.on_trailing()
        elif self.command:
            self.command()


# ------------------------------------------------------------- text entry --
class TextField(Widget):
    """Labelled single-line text field with inline keyboard editing (numeric when ``numeric=(lo, hi)``)."""

    cursor = "text"
    focusable = True

    def __init__(
        self,
        label: str,
        value: str | int = "",
        on_change: Callable[[str], None] | None = None,
        placeholder: str = "",
        suffix: str = "",
        field_width: int = 120,
        max_length: int = 64,
        numeric: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.label, self.on_change, self.placeholder, self.suffix = label, on_change, placeholder, suffix
        self.numeric = numeric
        self.text = str(value)
        self.max_length = max_length
        self.field_width = field_width
        self.font = font(METRICS.font_size)
        self.mono = font(METRICS.font_size, "medium")
        self.height = METRICS.row_h + px(6)
        self.focused = False

    @property
    def value(self) -> str:
        return self.text

    def set_text(self, text: str) -> None:
        if text != self.text:
            self.text = text
            self.invalidate()

    def _field_rect(self) -> Rect:
        x0, y0, x1, y1 = self.rect
        left = x1 - px(self.field_width) if self.label else x0
        return left, y0 + px(3), x1, y1 - px(3)

    def draw(self, img: Image.Image, ctx: DrawContext) -> None:
        x0, y0, x1, y1 = ctx.box(self)
        dx, dy = ctx.offset
        fx0, fy0, fx1, fy1 = self._field_rect()
        fx0, fy0, fx1, fy1 = fx0 + dx, fy0 + dy, fx1 + dx, fy1 + dy
        cy = (y0 + y1) / 2
        if self.label:
            glass.text(img, (x0, cy), self.label, self.font, PALETTE.fg, "lm")
        outline = PALETTE.accent if self.focused else None
        glass.fill_rounded(img, (fx0, fy0, fx1, fy1), METRICS.control_radius, PALETTE.fill, outline)
        avail = fx1 - fx0 - px(24) - (int(text_width(self.suffix, self.font)) + px(6) if self.suffix else 0)
        if self.text or self.focused:
            shown = glass.ellipsize(self.text, self.mono, avail) + ("|" if self.focused else "")
            glass.text(img, (fx0 + px(12), cy), shown, self.mono, PALETTE.fg, "lm")
        elif self.placeholder:
            glass.text(img, (fx0 + px(12), cy), glass.ellipsize(self.placeholder, self.font, avail), self.font, PALETTE.fg_dim, "lm")
        if self.suffix:
            glass.text(img, (fx1 - px(12), cy), self.suffix, self.font, PALETTE.fg_muted, "rm")

    def _accepts(self, char: str) -> bool:
        if self.numeric is not None:
            return char.isdigit()
        return char.isprintable()

    def _clamped(self) -> int:
        lo, hi = self.numeric or (0, 10**9)
        try:
            return min(hi, max(lo, int(self.text or "0")))
        except ValueError:
            return lo

    def on_focus(self, focused: bool) -> None:
        self.focused = focused
        if not focused and self.numeric is not None:
            clamped = str(self._clamped())
            if clamped != self.text:
                self.text = clamped
                if self.on_change:
                    self.on_change(self.text)
        self.invalidate()

    def on_key(self, keysym: str, char: str, mods: frozenset[str] = frozenset()) -> bool:
        if keysym == "BackSpace":
            self.text = self.text[:-1]
        elif keysym in ("Return", "KP_Enter", "Tab", "Escape"):
            return False  # let the panel handle confirm / focus change / dismiss
        elif char and self._accepts(char) and len(self.text) < self.max_length:
            self.text += char
        else:
            return False
        self.invalidate()
        if self.on_change:
            self.on_change(self.text)
        return True


class NumberField(TextField):
    """Labelled integer field (clamped to ``lo``..``hi`` when focus leaves)."""

    def __init__(
        self,
        label: str,
        value: int,
        lo: int = 1,
        hi: int = 20000,
        on_change: Callable[[int], None] | None = None,
        suffix: str = "px",
        field_width: int = 120,
    ) -> None:
        self._on_int_change = on_change
        super().__init__(label, str(int(value)), self._text_changed, suffix=suffix, max_length=6, numeric=(lo, hi), field_width=field_width)

    @property
    def value(self) -> int:  # type: ignore[override]
        return self._clamped()

    def set_value(self, value: int) -> None:
        self.set_text(str(int(value)))

    def _text_changed(self, text: str) -> None:
        if text and self._on_int_change:
            self._on_int_change(self.value)


class AnchorPicker(Widget):
    """3×3 grid to choose where the existing picture sits when the canvas changes size."""

    cursor = "hand"
    ANCHORS = ("nw", "n", "ne", "w", "center", "e", "sw", "s", "se")

    def __init__(self, label: str = "Anchor", value: str = "center", command: Callable[[str], None] | None = None) -> None:
        super().__init__()
        self.label, self.value, self.command = label, value, command
        self.cell = px(22)
        self.gap = px(4)
        self.font = font(METRICS.font_size)
        self.height = self.cell * 3 + self.gap * 2 + px(6)

    def _grid_origin(self) -> tuple[int, int]:
        x0, y0, x1, y1 = self.rect
        size = self.cell * 3 + self.gap * 2
        return x1 - size, y0 + px(3)

    def _cell_at(self, x: float, y: float) -> str | None:
        gx, gy = self._grid_origin()
        step = self.cell + self.gap
        if x < gx or y < gy:
            return None
        col, row = int((x - gx) // step), int((y - gy) // step)
        if 0 <= col < 3 and 0 <= row < 3 and (x - gx) % step < self.cell and (y - gy) % step < self.cell:
            return self.ANCHORS[row * 3 + col]
        return None

    def draw(self, img: Image.Image, ctx: DrawContext) -> None:
        x0, y0, x1, y1 = ctx.box(self)
        dx, dy = ctx.offset
        glass.text(img, (x0, (y0 + y1) / 2), self.label, self.font, PALETTE.fg, "lm")
        gx, gy = self._grid_origin()
        gx, gy = gx + dx, gy + dy
        step = self.cell + self.gap
        for i, name in enumerate(self.ANCHORS):
            cx0, cy0 = gx + (i % 3) * step, gy + (i // 3) * step
            fill = PALETTE.accent if name == self.value else PALETTE.fill_hover if ctx.is_hover(self) else PALETTE.fill
            glass.fill_rounded(img, (cx0, cy0, cx0 + self.cell, cy0 + self.cell), px(5), fill)

    def on_release(self, x: float, y: float, inside: bool) -> None:
        if not inside:
            return
        cell = self._cell_at(x, y)
        if cell and cell != self.value:
            self.value = cell
            self.invalidate()
            if self.command:
                self.command(cell)


class Histogram(Widget):
    """RGB histogram of the picture (square-root scaled, like most photo editors)."""

    def __init__(self, height: int = 88) -> None:
        super().__init__()
        self.height = px(height)
        self.data: np.ndarray | None = None  # shape (3, 256), normalised 0..1
        self.clipping: tuple[bool, bool] = (False, False)

    def set_data(self, data: np.ndarray | None) -> None:
        if data is None and self.data is None:
            return
        self.data = data
        if data is not None:
            self.clipping = (bool(data[:, 0].max() > 0.02), bool(data[:, 255].max() > 0.02))
        self.invalidate()

    def draw(self, img: Image.Image, ctx: DrawContext) -> None:
        x0, y0, x1, y1 = ctx.box(self)
        glass.fill_rounded(img, (x0, y0, x1, y1), METRICS.control_radius, (0, 0, 0, 70))
        if self.data is None:
            return
        inner_x0, inner_x1 = x0 + px(6), x1 - px(6)
        inner_y0, inner_y1 = y0 + px(6), y1 - px(6)
        w, h = inner_x1 - inner_x0, inner_y1 - inner_y0
        if w < 8 or h < 8:
            return
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        xs = np.linspace(0, w - 1, 256)
        for channel, color in ((0, (255, 90, 90, 150)), (1, (110, 230, 120, 150)), (2, (100, 150, 255, 150))):
            curve = np.sqrt(np.clip(self.data[channel], 0, 1))
            points = [(0, h)] + [(float(x), float(h - v * (h - 1))) for x, v in zip(xs, curve, strict=True)] + [(w - 1, h)]
            draw.polygon(points, fill=color)
        img.alpha_composite(layer, (inner_x0, inner_y0))
        for shadow, side_x in ((self.clipping[0], inner_x0), (self.clipping[1], inner_x1 - px(6))):
            if shadow:
                glass.fill_rounded(img, (side_x, inner_y0, side_x + px(6), inner_y0 + px(6)), px(2), (255, 255, 255, 200))


class CurveEditor(Widget):
    """Interactive tone curve: drag points, click the grid to add one, drag a point far outside to remove it.

    ``curves`` maps channel names (``rgb``, ``r``, ``g``, ``b``) to control
    points in 0..255; *channel* selects which one is edited.  The histogram of
    the picture (shape 3×256) is drawn behind the curve.
    """

    cursor = "cross"
    focusable = True
    REACH = 9  # logical px around a point that grabs it
    DROP_DISTANCE = 40  # logical px outside the grid that deletes a dragged point

    def __init__(
        self,
        curves: dict[str, list[tuple[float, float]]],
        channel: str = "rgb",
        on_change: Callable[[], None] | None = None,
        histogram: np.ndarray | None = None,
        size: int = 256,
    ) -> None:
        super().__init__()
        self.curves, self.channel, self.on_change, self.histogram = curves, channel, on_change, histogram
        self.height = px(size)
        self.selected: int | None = None
        self._dragging = False
        self.pointer: tuple[float, float] | None = None  # last press/drag position in 0..255 curve space

    # geometry ----------------------------------------------------------------
    def _grid(self) -> Rect:
        x0, y0, x1, y1 = self.rect
        side = min(x1 - x0, y1 - y0) - px(2)
        gx0 = (x0 + x1 - side) // 2
        gy0 = (y0 + y1 - side) // 2
        return gx0, gy0, gx0 + side, gy0 + side

    def _to_curve(self, x: float, y: float) -> tuple[float, float]:
        gx0, gy0, gx1, gy1 = self._grid()
        side = max(1, gx1 - gx0)
        return (x - gx0) / side * 255, (gy1 - y) / side * 255

    def _to_widget(self, cx: float, cy: float) -> tuple[float, float]:
        gx0, gy0, gx1, gy1 = self._grid()
        side = gx1 - gx0
        return gx0 + cx / 255 * side, gy1 - cy / 255 * side

    @property
    def points(self) -> list[tuple[float, float]]:
        return self.curves.setdefault(self.channel, [(0.0, 0.0), (255.0, 255.0)])

    def set_channel(self, channel: str) -> None:
        self.channel = channel
        self.selected = None
        self.invalidate()

    def reset(self, all_channels: bool = False) -> None:
        for name in list(self.curves) if all_channels else [self.channel]:
            self.curves[name] = [(0.0, 0.0), (255.0, 255.0)]
        self.selected = None
        self._changed()

    def _changed(self) -> None:
        self.invalidate()
        if self.on_change is not None:
            self.on_change()

    def _point_at(self, x: float, y: float) -> int | None:
        best, best_d = None, float(px(self.REACH))
        for i, (cx, cy) in enumerate(self.points):
            wx, wy = self._to_widget(cx, cy)
            d = math.hypot(wx - x, wy - y)
            if d <= best_d:
                best, best_d = i, d
        return best

    # drawing -----------------------------------------------------------------
    def draw(self, img: Image.Image, ctx: DrawContext) -> None:
        from ..core.curves import curve_values

        dx, dy = ctx.offset
        gx0, gy0, gx1, gy1 = self._grid()
        gx0, gy0, gx1, gy1 = gx0 + dx, gy0 + dy, gx1 + dx, gy1 + dy
        side = gx1 - gx0
        if side < 16:
            return
        glass.fill_rounded(
            img, (gx0, gy0, gx1, gy1), METRICS.control_radius, (0, 0, 0, 80), PALETTE.accent if ctx.focused is self else None
        )
        layer = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        if self.histogram is not None:
            channel_index = {"r": 0, "g": 1, "b": 2}.get(self.channel)
            data = self.histogram[channel_index] if channel_index is not None else self.histogram.mean(axis=0)
            curve = np.sqrt(np.clip(data, 0, 1))
            xs = np.linspace(0, side - 1, 256)
            pts = [(0.0, float(side))] + [(float(x), float(side - v * (side - 1) * 0.9)) for x, v in zip(xs, curve, strict=True)]
            pts.append((float(side - 1), float(side)))
            draw.polygon(pts, fill=(255, 255, 255, 38))
        for i in range(1, 4):
            q = side * i / 4
            draw.line((q, 0, q, side), fill=(255, 255, 255, 28), width=1)
            draw.line((0, q, side, q), fill=(255, 255, 255, 28), width=1)
        draw.line((0, side - 1, side - 1, 0), fill=(255, 255, 255, 45), width=1)
        xs_curve = np.linspace(0, 255, side)
        colors = {"rgb": (255, 255, 255, 235), "r": (255, 110, 110, 235), "g": (120, 230, 130, 235), "b": (120, 160, 255, 235)}
        for name, points in self.curves.items():
            if name == self.channel or len(points) < 2 or all(abs(x - y) < 1e-9 for x, y in points):
                continue
            ys = curve_values(points, xs_curve)
            faint = (*colors.get(name, (255, 255, 255))[:3], 70)
            draw.line([(i, side - 1 - y / 255 * (side - 1)) for i, y in enumerate(ys)], fill=faint, width=max(1, px(1)))
        ys = curve_values(self.points, xs_curve)
        line = [(i, side - 1 - y / 255 * (side - 1)) for i, y in enumerate(ys)]
        draw.line(line, fill=(0, 0, 0, 120), width=max(1, px(1)) + 2)
        draw.line(line, fill=colors.get(self.channel, (255, 255, 255, 235)), width=max(1, px(1)))
        r = px(4)
        for i, (cx, cy) in enumerate(self.points):
            wx, wy = cx / 255 * (side - 1), side - 1 - cy / 255 * (side - 1)
            fill = PALETTE.accent if i == self.selected else (255, 255, 255, 255)
            draw.ellipse((wx - r, wy - r, wx + r, wy + r), fill=fill, outline=(0, 0, 0, 150), width=1)
        img.alpha_composite(layer, (gx0, gy0))

    # interaction -------------------------------------------------------------
    def on_press(self, x: float, y: float) -> None:
        self.pointer = self._to_curve(x, y)
        index = self._point_at(x, y)
        if index is None:
            cx, cy = self.pointer
            cx, cy = min(255.0, max(0.0, cx)), min(255.0, max(0.0, cy))
            pts = self.points
            if any(abs(px_ - cx) < 1 for px_, _ in pts):
                return
            pts.append((cx, cy))
            pts.sort()
            index = pts.index((cx, cy))
            self._changed()
        self.selected = index
        self._dragging = True
        self.invalidate()

    def on_drag(self, x: float, y: float) -> None:
        if not self._dragging or self.selected is None:
            return
        gx0, gy0, gx1, gy1 = self._grid()
        margin = px(self.DROP_DISTANCE)
        pts = self.points
        if len(pts) > 2 and (x < gx0 - margin or x > gx1 + margin or y < gy0 - margin or y > gy1 + margin):
            del pts[self.selected]
            self.selected = None
            self._dragging = False
            self._changed()
            return
        self._move_selected(*self._to_curve(x, y))

    def _move_selected(self, cx: float, cy: float) -> None:
        """Move the selected point to (cx, cy) in curve space, keeping the order of the points."""
        if self.selected is None:
            return
        pts = self.points
        lo = pts[self.selected - 1][0] + 1 if self.selected > 0 else 0.0
        hi = pts[self.selected + 1][0] - 1 if self.selected < len(pts) - 1 else 255.0
        cx = min(hi, max(lo, cx))
        cy = min(255.0, max(0.0, cy))
        pts[self.selected] = (cx, cy)
        self.pointer = (cx, cy)
        self._changed()

    def on_release(self, x: float, y: float, inside: bool) -> None:
        self._dragging = False

    def on_double(self, x: float, y: float) -> None:
        index = self._point_at(x, y)
        if index is not None and len(self.points) > 2:
            del self.points[index]
            self.selected = None
            self._changed()

    def on_key(self, keysym: str, char: str, mods: frozenset[str] = frozenset()) -> bool:
        if self.selected is None:
            return False
        deltas = {"Left": (-1, 0), "Right": (1, 0), "Up": (0, 1), "Down": (0, -1)}
        if keysym in deltas:
            step = 10 if "shift" in mods else 1
            ddx, ddy = deltas[keysym]
            cx, cy = self.points[self.selected]
            self._move_selected(cx + ddx * step, cy + ddy * step)
            return True
        if keysym in ("Delete", "BackSpace") and len(self.points) > 2:
            del self.points[self.selected]
            self.selected = None
            self._changed()
            return True
        return False


class LevelsEditor(Widget):
    """Levels: histogram with draggable black / grey / white input markers and an output range bar."""

    cursor = "hand"
    focusable = True
    REACH = 10

    def __init__(self, histogram: np.ndarray | None = None, on_change: Callable[[], None] | None = None, height: int = 150) -> None:
        super().__init__()
        self.histogram, self.on_change = histogram, on_change
        self.height = px(height)
        self.in_black, self.in_white, self.gamma = 0, 255, 1.0
        self.out_black, self.out_white = 0, 255
        self._drag: str | None = None

    def values(self) -> dict[str, float]:
        return {
            "in_black": self.in_black,
            "in_white": self.in_white,
            "gamma": round(self.gamma, 2),
            "out_black": self.out_black,
            "out_white": self.out_white,
        }

    def set_values(self, **values: float) -> None:
        for key, value in values.items():
            setattr(self, key, value)
        self._changed()

    def reset(self) -> None:
        self.in_black, self.in_white, self.gamma, self.out_black, self.out_white = 0, 255, 1.0, 0, 255
        self._changed()

    def _changed(self) -> None:
        self.invalidate()
        if self.on_change is not None:
            self.on_change()

    # geometry ----------------------------------------------------------------
    BAR_H = 10

    def _bands(self) -> tuple[Rect, int, int]:
        """(histogram rect, input marker row y, output marker row y)."""
        x0, y0, x1, y1 = self.rect
        marker_h = px(14)
        below = px(8) + marker_h + px(10) + px(self.BAR_H) + px(10) + marker_h + px(2)
        hist = (x0 + px(8), y0 + px(4), x1 - px(8), y1 - below)
        in_y = hist[3] + px(8) + marker_h // 2
        out_y = y1 - px(2) - marker_h // 2
        return hist, in_y, out_y

    def _bar(self) -> Rect:
        """The black → white output bar sits between the two marker rows."""
        hist, in_y, out_y = self._bands()
        top = in_y + px(7) + px(10)
        return hist[0], top, hist[2], top + px(self.BAR_H)

    def _x_of(self, value: float) -> float:
        hist, _, _ = self._bands()
        return hist[0] + value / 255 * (hist[2] - hist[0])

    def _value_of(self, x: float) -> float:
        hist, _, _ = self._bands()
        return min(255.0, max(0.0, (x - hist[0]) / max(1, hist[2] - hist[0]) * 255))

    @property
    def mid_input(self) -> float:
        return self.in_black + (self.in_white - self.in_black) * (0.5**self.gamma)

    def _markers(self) -> dict[str, tuple[float, float]]:
        _, in_y, out_y = self._bands()
        return {
            "in_black": (self._x_of(self.in_black), in_y),
            "gamma": (self._x_of(self.mid_input), in_y),
            "in_white": (self._x_of(self.in_white), in_y),
            "out_black": (self._x_of(self.out_black), out_y),
            "out_white": (self._x_of(self.out_white), out_y),
        }

    def _marker_at(self, x: float, y: float) -> str | None:
        best, best_d = None, float(px(self.REACH))
        for name, (mx, my) in self._markers().items():
            d = math.hypot(mx - x, my - y)
            if d <= best_d:
                best, best_d = name, d
        return best

    # drawing -----------------------------------------------------------------
    def draw(self, img: Image.Image, ctx: DrawContext) -> None:
        dx, dy = ctx.offset
        hist, in_y, out_y = self._bands()
        hx0, hy0, hx1, hy1 = hist[0] + dx, hist[1] + dy, hist[2] + dx, hist[3] + dy
        glass.fill_rounded(
            img, (hx0, hy0, hx1, hy1), METRICS.control_radius, (0, 0, 0, 80), PALETTE.accent if ctx.focused is self else None
        )
        w, h = hx1 - hx0, hy1 - hy0
        if w > 16 and h > 8 and self.histogram is not None:
            layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer)
            data = np.sqrt(np.clip(self.histogram.mean(axis=0), 0, 1))
            xs = np.linspace(0, w - 1, 256)
            pts = (
                [(0.0, float(h))] + [(float(x), float(h - v * (h - 2))) for x, v in zip(xs, data, strict=True)] + [(float(w - 1), float(h))]
            )
            draw.polygon(pts, fill=(255, 255, 255, 110))
            img.alpha_composite(layer, (hx0, hy0))
        draw = ImageDraw.Draw(img)
        # output bar: a black → white gradient
        bx0, by0, bx1, by1 = self._bar()
        by0, by1 = by0 + dy, by1 + dy
        for i in range(256):
            bx = hx0 + i / 255 * (hx1 - hx0)
            draw.line((bx, by0, bx, by1), fill=(i, i, i, 255))
        r = px(6)
        fills = {
            "in_black": (20, 20, 20),
            "gamma": (128, 128, 128),
            "in_white": (245, 245, 245),
            "out_black": (20, 20, 20),
            "out_white": (245, 245, 245),
        }
        for name, (mx, my) in self._markers().items():
            mx, my = mx + dx, my + dy
            draw.polygon([(mx, my - r), (mx - r, my + r), (mx + r, my + r)], fill=(*fills[name], 255), outline=(255, 255, 255, 200))

    # interaction -------------------------------------------------------------
    def on_press(self, x: float, y: float) -> None:
        self._drag = self._marker_at(x, y)
        self.invalidate()

    def on_drag(self, x: float, y: float) -> None:
        if self._drag is None:
            return
        v = self._value_of(x)
        if self._drag == "in_black":
            self.in_black = int(min(v, self.in_white - 2))
        elif self._drag == "in_white":
            self.in_white = int(max(v, self.in_black + 2))
        elif self._drag == "gamma":
            span = max(1, self.in_white - self.in_black)
            t = min(0.99, max(0.01, (v - self.in_black) / span))
            self.gamma = min(9.99, max(0.1, math.log(t) / math.log(0.5)))
        elif self._drag == "out_black":
            self.out_black = int(min(v, self.out_white))
        elif self._drag == "out_white":
            self.out_white = int(max(v, self.out_black))
        self._changed()

    def on_release(self, x: float, y: float, inside: bool) -> None:
        self._drag = None

    def on_double(self, x: float, y: float) -> None:
        self.reset()


class ColorDot(Widget):
    """A small colour swatch (used in the info pill for the colour under the cursor)."""

    def __init__(self, color: RGBA | None = None) -> None:
        super().__init__()
        self.color = color
        self.width = px(18)
        self.flex = 0

    def set_color(self, color: RGBA | None) -> None:
        if color != self.color:
            self.color = color
            self.invalidate()

    def draw(self, img: Image.Image, ctx: DrawContext) -> None:
        x0, y0, x1, y1 = ctx.box(self)
        size = px(14)
        cy = (y0 + y1) // 2
        box = (x0, cy - size // 2, x0 + size, cy + size // 2)
        if self.color is None:
            glass.fill_rounded(img, box, px(4), (255, 255, 255, 0), (255, 255, 255, 50))
        else:
            r, g, b, _a = self.color
            glass.fill_rounded(img, box, px(4), (r, g, b, 255), (255, 255, 255, 60))


# ------------------------------------------------------------------ panel --
PlaceFn = Callable[[int, int, int, int], Rect]


class Panel:
    """A floating glass surface hosting a widget tree.

    ``place(viewport_w, viewport_h, preferred_w, preferred_h) -> (x, y, w, h)``
    positions the panel; with ``fit_width`` / ``fit_height`` the preferred
    size of the content wins over the returned ``w`` / ``h``.
    """

    def __init__(
        self,
        content: Widget,
        place: PlaceFn,
        *,
        padding: int | tuple[int, int, int, int] | None = None,
        radius: int | None = None,
        scrollable: bool = False,
        fit_height: bool = False,
        fit_width: bool = False,
        max_width: int | None = None,
        name: str = "",
        modal: bool = False,
        dismiss_on_outside: bool = False,
        blur: int | None = None,
        shadow: bool = True,
    ) -> None:
        self.content = content
        self.place = place
        if padding is None:
            self.pad: tuple[int, int, int, int] = (METRICS.panel_pad,) * 4
        elif isinstance(padding, int):
            self.pad = (px(padding),) * 4
        else:
            left, top, right, bottom = padding  # left, top, right, bottom
            self.pad = (px(left), px(top), px(right), px(bottom))
        self.radius = METRICS.panel_radius if radius is None else px(radius)
        self.scrollable = scrollable
        self.fit_height = fit_height
        self.fit_width = fit_width
        self.max_width = max_width
        self.name = name
        self.modal = modal
        self.dismiss_on_outside = dismiss_on_outside
        self.blur = METRICS.blur if blur is None else px(blur)
        self.shadow = shadow
        self.visible = True
        self.rect: Rect = (0, 0, 1, 1)
        self.scroll_y = 0
        self.content_height = 0
        self._image: Image.Image | None = None
        self._dirty = True
        self._backdrop_key: object = None  # what the glass last blurred; set by the surface
        self.view: Any = None  # the owning Surface
        content.attach(self)

    # geometry ----------------------------------------------------------------
    @property
    def x(self) -> int:
        return self.rect[0]

    @property
    def y(self) -> int:
        return self.rect[1]

    @property
    def width(self) -> int:
        return self.rect[2] - self.rect[0]

    @property
    def height(self) -> int:
        return self.rect[3] - self.rect[1]

    @property
    def viewport_height(self) -> int:
        return self.height - self.pad[1] - self.pad[3]

    @property
    def max_scroll(self) -> int:
        return max(0, self.content_height - self.viewport_height)

    def layout(self, vw: int, vh: int) -> None:
        pad_x, pad_y = self.pad[0] + self.pad[2], self.pad[1] + self.pad[3]
        pref_w = self.content.preferred_width() + pad_x
        pref_h = self.content.measure(max(1, (pref_w if self.fit_width else 300) - pad_x)) + pad_y
        x, y, w, h = self.place(vw, vh, pref_w, pref_h)
        if self.fit_width:
            w = pref_w if self.max_width is None else min(pref_w, px(self.max_width))
        cw = max(1, w - pad_x)
        self.content_height = self.content.measure(cw)
        if self.fit_height:
            h = self.content_height + pad_y
        self.rect = (int(x), int(y), int(x + w), int(y + h))
        self.content.layout(0, 0, cw, self.content_height)
        self.scroll_y = min(self.scroll_y, self.max_scroll)
        self._dirty = True

    def contains(self, vx: float, vy: float) -> bool:
        x0, y0, x1, y1 = self.rect
        return self.visible and x0 <= vx < x1 and y0 <= vy < y1

    def to_content(self, vx: float, vy: float) -> tuple[float, float]:
        return vx - self.rect[0] - self.pad[0], vy - self.rect[1] - self.pad[1] + self.scroll_y

    def widget_at(self, vx: float, vy: float) -> Widget | None:
        if not self.contains(vx, vy):
            return None
        cx, cy = self.to_content(vx, vy)
        if cy < self.scroll_y or cy >= self.scroll_y + self.viewport_height:
            return None
        found = self.content.hit(cx, cy)
        return None if found is self.content else found

    def widget_rect_in_view(self, widget: Widget) -> Rect:
        x0, y0, x1, y1 = widget.rect
        ox, oy = self.rect[0] + self.pad[0], self.rect[1] + self.pad[1] - self.scroll_y
        return x0 + ox, y0 + oy, x1 + ox, y1 + oy

    def scroll_to_widget(self, widget: Widget) -> None:
        """Scroll just enough to make *widget* visible."""
        if not self.scrollable:
            return
        y0, y1 = widget.rect[1], widget.rect[3]
        if y0 < self.scroll_y:
            self.scroll_y = y0
        elif y1 > self.scroll_y + self.viewport_height:
            self.scroll_y = min(self.max_scroll, y1 - self.viewport_height)
        self.invalidate()

    def scroll(self, dy: float) -> bool:
        if not self.scrollable or self.max_scroll == 0:
            return False
        new = int(min(self.max_scroll, max(0, self.scroll_y + dy)))
        if new != self.scroll_y:
            self.scroll_y = new
            self.invalidate()
        return True

    # rendering ---------------------------------------------------------------
    def invalidate(self) -> None:
        self._dirty = True
        if self.view is not None:
            self.view.panel_changed(self)

    @property
    def dirty(self) -> bool:
        return self._dirty

    def render(self, backdrop: Image.Image, ctx: DrawContext) -> Image.Image:
        """Draw the panel over *backdrop* (RGB crop of what's behind it)."""
        w, _h = self.width, self.height
        img = glass.glass(backdrop, self.radius, self.blur)
        cw = max(1, w - self.pad[0] - self.pad[2])
        ch = max(1, self.viewport_height)
        content = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        ctx.offset = (0, -self.scroll_y)
        self.content.draw(content, ctx)
        img.alpha_composite(content, (self.pad[0], self.pad[1]))
        if self.scrollable and self.max_scroll > 0:
            track_h = ch - px(8)
            thumb_h = max(px(24), int(track_h * ch / max(ch, self.content_height)))
            thumb_y = self.pad[1] + px(4) + int((track_h - thumb_h) * self.scroll_y / self.max_scroll)
            glass.fill_rounded(img, (w - px(8), thumb_y, w - px(4), thumb_y + thumb_h), px(2), (255, 255, 255, 60))
        self._image = img
        self._dirty = False
        return img


# ------------------------------------------------------------------ layers --
class LayerRow(Widget):
    """One row of the Layers list: visibility eye, thumbnail (plus mask thumbnail), name and details.

    On the active layer the edit target - pixels or mask - is outlined; clicking
    the mask thumbnail targets the mask, clicking the picture thumbnail the pixels.
    """

    cursor = "hand"
    THUMB = (44, 34)

    def __init__(
        self,
        layer,
        active: bool,
        thumbnail: Image.Image,
        on_select: Callable[[], None],
        on_toggle: Callable[[], None],
        on_edit: Callable[[], None] | None = None,
        mask_thumbnail: Image.Image | None = None,
        mask_active: bool = False,
        on_select_mask: Callable[[], None] | None = None,
        depth: int = 0,
        collapsed: bool | None = None,
        on_collapse: Callable[[], None] | None = None,
        member_count: int = 0,
    ) -> None:
        super().__init__()
        self.layer, self.active, self.thumb = layer, active, thumbnail
        self.mask_thumb, self.mask_active, self.on_select_mask = mask_thumbnail, mask_active, on_select_mask
        self.on_select, self.on_toggle, self.on_edit = on_select, on_toggle, on_edit
        self.depth, self.collapsed, self.on_collapse, self.member_count = depth, collapsed, on_collapse, member_count
        self.font = font(METRICS.font_size, "medium" if active else "regular")
        self.sub_font = font(METRICS.small_font_size)
        self.height = px(46)

    INDENT = 18

    def _left(self) -> int:
        return self.rect[0] + px(self.INDENT) * self.depth

    def _eye_zone(self, x: float) -> bool:
        return x < self._left() + px(30)

    def _thumb_zone(self, x: float) -> bool:
        start = self._left() + px(32)
        return start <= x < start + self.thumb.width

    def _mask_zone(self, x: float) -> bool:
        if self.mask_thumb is None:
            return False
        start = self._left() + px(32) + self.thumb.width + px(4)
        return start <= x < start + self.mask_thumb.width + px(4)

    def details(self) -> str:
        parts = []
        if self.layer.blend_mode != "normal":
            parts.append(self.layer.blend_mode)
        if self.layer.opacity < 1:
            parts.append(f"{self.layer.opacity * 100:.0f}%")
        if self.layer.has_adjustments:
            parts.append("adjusted")
        if not effects_empty(self.layer.meta.get("effects")):
            parts.append("fx")
        if self.layer.kind == "text":
            parts.append("text")
        elif self.layer.kind == "adjustment":
            parts.append("adjustment")
        elif self.layer.kind == "group":
            parts.insert(0, f"{self.member_count} layers" if self.member_count != 1 else "1 layer")
        if self.layer.mask is not None and not self.layer.mask_enabled:
            parts.append("mask off")
        if self.layer.locked:
            parts.append("locked")
        return "  ·  ".join(parts)

    def draw(self, img: Image.Image, ctx: DrawContext) -> None:
        x0, y0, x1, y1 = ctx.box(self)
        if self.active:
            glass.fill_rounded(img, (x0, y0 + px(1), x1, y1 - px(1)), METRICS.control_radius - px(2), PALETTE.accent_soft)
        elif ctx.is_hover(self):
            glass.fill_rounded(img, (x0, y0 + px(1), x1, y1 - px(1)), METRICS.control_radius - px(2), PALETTE.fill)
        cy = (y0 + y1) // 2
        x0 += px(self.INDENT) * self.depth
        eye = glass.icon("eye" if self.layer.visible else "eye-off", px(16), PALETTE.fg if self.layer.visible else PALETTE.fg_dim)
        glass.paste(img, eye, (x0 + px(7), cy - px(8)))
        tx = x0 + px(32)
        ty = cy - self.thumb.height // 2
        glass.paste(img, self.thumb, (tx, ty))
        if self.collapsed is not None:  # group header: expand / collapse chevron on the thumbnail
            chevron = glass.icon("chevron-right" if self.collapsed else "chevron-down", px(14), PALETTE.fg)
            glass.paste(img, chevron, (tx + self.thumb.width - px(16), ty + self.thumb.height - px(16)))
        if self.active and self.mask_thumb is not None and not self.mask_active:
            self._outline(img, (tx, ty, tx + self.thumb.width, ty + self.thumb.height))
        tx += self.thumb.width + px(4)
        if self.mask_thumb is not None:
            my = cy - self.mask_thumb.height // 2
            glass.paste(img, self.mask_thumb, (tx + px(2), my))
            if self.active and self.mask_active:
                self._outline(img, (tx + px(2), my, tx + px(2) + self.mask_thumb.width, my + self.mask_thumb.height))
            tx += self.mask_thumb.width + px(4)
        tx += px(6)
        color = PALETTE.fg if self.layer.visible else PALETTE.fg_muted
        details = self.details()
        name_y = cy - (px(8) if details else 0)
        glass.text(img, (tx, name_y), glass.ellipsize(self.layer.name, self.font, x1 - tx - px(8)), self.font, color, "lm")
        if details:
            glass.text(
                img, (tx, cy + px(9)), glass.ellipsize(details, self.sub_font, x1 - tx - px(8)), self.sub_font, PALETTE.fg_muted, "lm"
            )

    @staticmethod
    def _outline(img: Image.Image, box: tuple[int, int, int, int]) -> None:
        x0, y0, x1, y1 = box
        ImageDraw.Draw(img).rectangle((x0 - px(2), y0 - px(2), x1 + px(1), y1 + px(1)), outline=PALETTE.accent, width=max(1, px(2)))

    def on_release(self, x: float, y: float, inside: bool) -> None:
        if not inside:
            return
        if self._eye_zone(x):
            self.on_toggle()
        elif self._mask_zone(x) and self.on_select_mask is not None:
            self.on_select_mask()
        elif self._thumb_zone(x) and self.on_collapse is not None:
            self.on_collapse()
        else:
            self.on_select()

    def on_double(self, x: float, y: float) -> None:
        if not self._eye_zone(x) and not self._mask_zone(x) and self.on_edit is not None:
            self.on_edit()


class ColorSwatches(Widget):
    """Foreground / background colour squares (click to pick, small arrow swaps)."""

    cursor = "hand"

    def __init__(self, get_colors: Callable[[], tuple[RGBA, RGBA]], on_pick: Callable[[str], None], on_swap: Callable[[], None]) -> None:
        super().__init__()
        self.get_colors, self.on_pick, self.on_swap = get_colors, on_pick, on_swap
        self.height = px(56)
        self.tooltip = "Foreground / background colour · click to change · X swaps"

    def _zones(self) -> dict[str, Rect]:
        x0, y0, x1, y1 = self.rect
        size = px(30)
        w = x1 - x0
        left = x0 + (w - px(44)) // 2
        return {
            "background": (left + px(14), y0 + px(20), left + px(14) + size, y0 + px(20) + size),
            "foreground": (left, y0 + px(6), left + size, y0 + px(6) + size),
            "swap": (left + px(32), y0 + px(2), left + px(46), y0 + px(16)),
        }

    def draw(self, img: Image.Image, ctx: DrawContext) -> None:
        dx, dy = ctx.offset
        fg, bg = self.get_colors()
        zones = self._zones()
        for name, color in (("background", bg), ("foreground", fg)):
            zx0, zy0, zx1, zy1 = zones[name]
            glass.fill_rounded(
                img, (zx0 + dx, zy0 + dy, zx1 + dx, zy1 + dy), px(6), (color[0], color[1], color[2], 255), (255, 255, 255, 90)
            )
        sx0, sy0, sx1, sy1 = zones["swap"]
        glass.paste(img, glass.icon("swap", px(12), PALETTE.fg_muted), (sx0 + dx, sy0 + dy))

    def on_release(self, x: float, y: float, inside: bool) -> None:
        if not inside:
            return
        for name, (zx0, zy0, zx1, zy1) in self._zones().items():
            if zx0 <= x < zx1 and zy0 <= y < zy1:
                if name == "swap":
                    self.on_swap()
                else:
                    self.on_pick(name)
                return
