"""The editing surface: picture, tool overlay and floating glass panels, independent of the toolkit.

:class:`Surface` owns the zoom/pan state, composes the *frame* (workspace
background, picture, tool overlay) with Pillow, renders every
:class:`~pixelforge.gui.ui.Panel` over a crop of that frame (so the glass blur
shows what is really behind it) and routes pointer/keyboard events to panels
or the active tool.  A small :class:`Backend` object supplied by the toolkit
layer does the only platform-specific work: reporting the viewport size,
painting the produced images, setting the cursor and running timers.

All coordinates are **device pixels**; the backend converts from logical
(point) coordinates on HiDPI displays.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from ..core.document import DocumentError
from . import glass, theme
from .theme import METRICS, PALETTE, font, px
from .ui import DrawContext, Panel, Segmented, Widget
from .worker import Worker

ZOOM_LEVELS: tuple[float, ...] = (0.05, 0.1, 0.15, 0.25, 0.33, 0.5, 0.67, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)
MIN_ZOOM, MAX_ZOOM = 0.02, 32.0
CHECKER_BASE = 12
TOOLTIP_DELAY = 450

Layer = tuple[str, int, int, Image.Image]  # kind ("frame" | "shadow" | "panel" | "tooltip"), x, y, image


@dataclass(frozen=True)
class PointerEvent:
    """A pointer event in document coordinates (plus the raw surface position in device pixels)."""

    x: float
    y: float
    wx: int
    wy: int
    shift: bool = False
    ctrl: bool = False
    alt: bool = False

    def clamped(self, width: int, height: int) -> tuple[float, float]:
        return min(max(self.x, 0.0), float(width)), min(max(self.y, 0.0), float(height))


class Backend(Protocol):
    """What the toolkit layer must provide to a :class:`Surface`."""

    def viewport_size(self) -> tuple[int, int]: ...  # device pixels

    def present(self, layers: list[Layer]) -> None: ...

    def set_cursor(self, name: str) -> None: ...

    def after(self, ms: int, fn: Callable[[], None]) -> object: ...

    def after_cancel(self, handle: object) -> None: ...


ASYNC_PIXELS = 1_500_000  # pictures larger than this are composited on the background worker


class Surface:
    def __init__(self, app: Any, backend: Backend) -> None:
        self.app: Any = app
        self.backend = backend
        self.zoom = 1.0
        self.ox = 0
        self.oy = 0
        self.panels: list[Panel] = []
        self._frame: Image.Image | None = None
        self._frame_dirty = True
        self._render_job: object | None = None
        self.frame_pending = False  # a background composite for the current view is on its way
        self._last_picture: tuple[Image.Image, float, tuple[int, int, int, int]] | None = None  # picture, zoom, view box
        self._force_sync = False
        self._checker_cache: dict[tuple[int, int, int, int], Image.Image] = {}
        self._panel_layers: dict[int, tuple[Layer, Layer | None]] = {}
        self._fit_pending = False
        self.render_count = 0

        self.hover_panel: Panel | None = None
        self.hover_widget: Widget | None = None
        self.pressed_widget: Widget | None = None
        self.focused_widget: Widget | None = None
        self.pointer: tuple[int, int] | None = None
        self._pan_anchor: tuple[int, int, int, int] | None = None
        self._space = False
        self._tooltip_job: object | None = None
        self._tooltip_layer: Layer | None = None
        self.tooltip_widget: Widget | None = None
        self.cursor = "arrow"
        self.show_grid = False
        self.show_rulers = False
        self._ants_phase = 0
        self._ants_job: object | None = None
        self._ants_cache: tuple[tuple, Image.Image] | None = None

    # ------------------------------------------------------------- panels --
    def add_panel(self, panel: Panel) -> Panel:
        panel.view = self
        self.panels.append(panel)
        self.layout_panels()
        return panel

    def remove_panel(self, panel: Panel) -> None:
        if panel in self.panels:
            self.panels.remove(panel)
            self._panel_layers.pop(id(panel), None)
            if self.hover_widget is not None and self.hover_widget.panel is panel:
                self._set_hover(None, None)
            if self.focused_widget is not None and self.focused_widget.panel is panel:
                self.set_focus(None)
            self.schedule()

    def clear_panels(self) -> None:
        for panel in list(self.panels):
            self.remove_panel(panel)

    def layout_panels(self) -> None:
        vw, vh = self.backend.viewport_size()
        for panel in self.panels:
            panel.layout(vw, vh)
        self.schedule()

    def panel_changed(self, _panel: Panel) -> None:
        self.schedule()

    def modal_panel(self) -> Panel | None:
        for panel in reversed(self.panels):
            if panel.visible and panel.modal:
                return panel
        return None

    def panel_at(self, vx: float, vy: float) -> Panel | None:
        modal = self.modal_panel()
        if modal is not None:
            return modal if modal.contains(vx, vy) else None
        for panel in reversed(self.panels):
            if panel.contains(vx, vy):
                return panel
        return None

    def set_focus(self, widget: Widget | None) -> None:
        if widget is self.focused_widget:
            return
        if self.focused_widget is not None:
            self.focused_widget.on_focus(False)
        self.focused_widget = widget
        if widget is not None:
            widget.on_focus(True)
        focus_changed = getattr(self.backend, "focus_changed", None)
        if focus_changed is not None:
            focus_changed()

    # ----------------------------------------------------------- geometry --
    @property
    def doc(self):
        return self.app.doc

    def viewport_size(self) -> tuple[int, int]:
        w, h = self.backend.viewport_size()
        return max(1, w), max(1, h)

    def to_doc(self, wx: float, wy: float) -> tuple[float, float]:
        return (wx - self.ox) / self.zoom, (wy - self.oy) / self.zoom

    def to_view(self, x: float, y: float) -> tuple[float, float]:
        return self.ox + x * self.zoom, self.oy + y * self.zoom

    def image_rect(self) -> tuple[int, int, int, int]:
        w, h = math.ceil(self.doc.width * self.zoom), math.ceil(self.doc.height * self.zoom)
        return self.ox, self.oy, self.ox + w, self.oy + h

    def content_area(self) -> tuple[int, int, int, int]:
        return self.app.content_area(*self.viewport_size())

    def set_zoom(self, zoom: float, anchor: tuple[float, float] | None = None) -> None:
        zoom = min(MAX_ZOOM, max(MIN_ZOOM, float(zoom)))
        if self.doc is None:
            self.zoom = zoom
            return
        if anchor is None:
            x0, y0, x1, y1 = self.content_area()
            anchor = ((x0 + x1) / 2, (y0 + y1) / 2)
        dx, dy = self.to_doc(*anchor)
        self.zoom = zoom
        self.ox = int(round(anchor[0] - dx * zoom))
        self.oy = int(round(anchor[1] - dy * zoom))
        self._clamp_offset()
        self.app.view_changed()
        self.invalidate_frame()

    def zoom_in(self, anchor=None) -> None:
        self.set_zoom(next((z for z in ZOOM_LEVELS if z > self.zoom * 1.001), self.zoom * 1.5), anchor)

    def zoom_out(self, anchor=None) -> None:
        self.set_zoom(next((z for z in reversed(ZOOM_LEVELS) if z < self.zoom * 0.999), self.zoom / 1.5), anchor)

    def fit(self, allow_upscale: bool = False) -> None:
        if self.doc is None:
            return
        vw, vh = self.viewport_size()
        if vw < 50 or vh < 50:
            self._fit_pending = True
            return
        x0, y0, x1, y1 = self.content_area()
        margin = px(28)
        zoom = min((x1 - x0 - margin * 2) / self.doc.width, (y1 - y0 - margin * 2) / self.doc.height)
        if not allow_upscale:
            zoom = min(zoom, 1.0)
        self.zoom = min(MAX_ZOOM, max(MIN_ZOOM, zoom))
        self.ox = int(round((x0 + x1) / 2 - self.doc.width * self.zoom / 2))
        self.oy = int(round((y0 + y1) / 2 - self.doc.height * self.zoom / 2))
        self.app.view_changed()
        self.invalidate_frame()

    def pan(self, dx: float, dy: float) -> None:
        self.ox += int(round(dx))
        self.oy += int(round(dy))
        self._clamp_offset()
        self.invalidate_frame()

    def _clamp_offset(self) -> None:
        if self.doc is None:
            return
        vw, vh = self.viewport_size()
        w, h = self.doc.width * self.zoom, self.doc.height * self.zoom
        keep = px(64)
        self.ox = int(min(vw - keep, max(keep - w, self.ox)))
        self.oy = int(min(vh - keep, max(keep - h, self.oy)))

    def visible_doc_box(self) -> tuple[float, float, float, float] | None:
        if self.doc is None:
            return None
        vw, vh = self.viewport_size()
        x0, y0 = self.to_doc(0, 0)
        x1, y1 = self.to_doc(vw, vh)
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(self.doc.width), x1), min(float(self.doc.height), y1)
        return None if x1 <= x0 or y1 <= y0 else (x0, y0, x1, y1)

    # ------------------------------------------------------------ render --
    def invalidate_frame(self) -> None:
        self._frame_dirty = True
        self.schedule()

    def schedule(self, delay: int = 8) -> None:
        if self._render_job is None:
            self._render_job = self.backend.after(delay, self.render)

    def clear_caches(self) -> None:
        """Forget rendered backdrops, panels and checkers (after a theme change)."""
        self._checker_cache.clear()
        self._panel_layers.clear()
        self._last_picture = None
        for panel in self.panels:
            panel.invalidate()

    def resized(self) -> None:
        """The backend's viewport changed size."""
        self._checker_cache.clear()
        self.layout_panels()
        if self._fit_pending and self.doc is not None:
            self._fit_pending = False
            self.fit()
        self.invalidate_frame()

    def _checker(self, box: tuple[int, int, int, int]) -> Image.Image:
        x0, y0, x1, y1 = box
        checker = px(CHECKER_BASE)
        key = (x0 % (checker * 2), y0 % (checker * 2), x1 - x0, y1 - y0)
        img = self._checker_cache.get(key)
        if img is None:
            if len(self._checker_cache) > 8:
                self._checker_cache.clear()
            w, h = x1 - x0, y1 - y0
            ys, xs = np.mgrid[0:h, 0:w]
            pattern = (((xs + key[0]) // checker + (ys + key[1]) // checker) % 2).astype(bool)
            arr = np.empty((h, w, 3), np.uint8)
            light, dark = PALETTE.checker
            arr[...] = dark
            arr[pattern] = light
            img = Image.fromarray(arr, "RGB")
            self._checker_cache[key] = img
        return img

    def compose_frame(self) -> Image.Image:
        vw, vh = self.viewport_size()
        frame = Image.new("RGB", (vw, vh), PALETTE.bg)
        if self.doc is None:
            return frame
        ix0, iy0, ix1, iy1 = self.image_rect()
        vx0, vy0, vx1, vy1 = max(0, ix0), max(0, iy0), min(vw, ix1), min(vh, iy1)
        if vx1 <= vx0 or vy1 <= vy0:
            return frame
        iw, ih = ix1 - ix0, iy1 - iy0
        if iw * ih <= 4_000_000 and iw < vw * 2 and ih < vh * 2:
            frame_rgba = frame.convert("RGBA")
            blur = px(18)
            glass.paste(frame_rgba, glass.shadow((iw, ih), 0, blur, PALETTE.glass_shadow), (ix0 - blur * 2, iy0 - blur * 2 + px(10)))
            frame = frame_rgba.convert("RGB")
        checker = self._checker((vx0, vy0, vx1, vy1))
        view_box = (vx0 - ix0, vy0 - iy0, vx1 - ix0, vy1 - iy0)
        picture = self._picture(view_box)
        region = checker.copy()
        if picture is not None:
            region.paste(picture, (0, 0), picture)
        frame.paste(region, (vx0, vy0))
        tool = self.app.tool
        if tool is not None:
            frame = tool.paint_overlay(frame, self)
        paint_overlay = getattr(self.app, "paint_overlay", None)
        if paint_overlay is not None:
            frame = paint_overlay(frame, self)
        if self.show_grid:
            frame = self._paint_grid(frame)
        if self.show_rulers:
            frame = self._paint_rulers(frame)
        return frame

    def _picture(self, view_box: tuple[int, int, int, int]) -> Image.Image | None:
        """The composite for *view_box*: rendered inline for small pictures, otherwise fetched from the cache
        while a background job renders it - showing the previous picture re-projected in the meantime."""
        doc = self.doc
        assert doc is not None
        params: dict[str, Any] = {
            "scale": self.zoom,
            "view_box": view_box,
            "overrides": self.app.preview_overrides or None,
            "neutral": bool(getattr(self.app, "compare", False)),
            "mask_overrides": getattr(self.app, "preview_mask_overrides", None) or None,
        }
        worker: Worker | None = getattr(self.app, "worker", None)
        self._force_sync, force_sync = False, self._force_sync
        if worker is not None and doc.width * doc.height > ASYNC_PIXELS and not force_sync:
            picture = doc.cached_composite(**params)
            if picture is None:
                self.frame_pending = True
                worker.submit("frame", lambda: doc.composite(**params), self._frame_ready)
                return self._interim_picture(view_box)
        else:
            picture = doc.composite(**params)
        self.frame_pending = False
        self._last_picture = (picture, self.zoom, view_box)
        return picture

    def _frame_ready(self, result: Image.Image | None) -> None:
        """A background composite finished (GUI thread): draw again - the cache now has it."""
        if result is None:
            self._force_sync = True  # the job failed: let the inline path surface the problem
        self.invalidate_frame()

    def _interim_picture(self, view_box: tuple[int, int, int, int]) -> Image.Image | None:
        """The last rendered picture re-projected onto the new zoom / view box (a quick stand-in)."""
        if self._last_picture is None or self.doc is None:
            return None
        picture, old_zoom, old_box = self._last_picture
        w, h = view_box[2] - view_box[0], view_box[3] - view_box[1]
        # both boxes in document coordinates
        old = (old_box[0] / old_zoom, old_box[1] / old_zoom, old_box[2] / old_zoom, old_box[3] / old_zoom)
        new = (view_box[0] / self.zoom, view_box[1] / self.zoom, view_box[2] / self.zoom, view_box[3] / self.zoom)
        ix0, iy0 = max(old[0], new[0]), max(old[1], new[1])
        ix1, iy1 = min(old[2], new[2]), min(old[3], new[3])
        if ix1 - ix0 < 1 / self.zoom or iy1 - iy0 < 1 / self.zoom:
            return None
        src_box = ((ix0 - old[0]) * old_zoom, (iy0 - old[1]) * old_zoom, (ix1 - old[0]) * old_zoom, (iy1 - old[1]) * old_zoom)
        dst_w, dst_h = max(1, round((ix1 - ix0) * self.zoom)), max(1, round((iy1 - iy0) * self.zoom))
        resample = Image.Resampling.NEAREST if self.zoom > old_zoom else Image.Resampling.BILINEAR
        piece = picture.resize((dst_w, dst_h), resample, box=src_box)
        out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        out.paste(piece, (round((ix0 - new[0]) * self.zoom), round((iy0 - new[1]) * self.zoom)))
        return out

    def grid_spacing(self) -> int:
        """Grid spacing in document pixels chosen so lines are at least ~40 device px apart."""
        for step in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000):
            if step * self.zoom >= px(40):
                return step
        return 10000

    def _paint_grid(self, frame: Image.Image) -> Image.Image:
        ix0, iy0, ix1, iy1 = self.image_rect()
        vw, vh = frame.size
        step = self.grid_spacing()
        base = frame.convert("RGBA")
        draw = ImageDraw.Draw(base)
        color = (255, 255, 255, 70)
        x = 0
        while x <= self.doc.width:
            vx = int(round(self.ox + x * self.zoom))
            if 0 <= vx < vw:
                draw.line((vx, max(0, iy0), vx, min(vh, iy1)), fill=color, width=1)
            x += step
        y = 0
        while y <= self.doc.height:
            vy = int(round(self.oy + y * self.zoom))
            if 0 <= vy < vh:
                draw.line((max(0, ix0), vy, min(vw, ix1), vy), fill=color, width=1)
            y += step
        return base.convert("RGB")

    def _paint_rulers(self, frame: Image.Image) -> Image.Image:
        vw, vh = frame.size
        size = px(20)
        base = frame.convert("RGBA")
        draw = ImageDraw.Draw(base)
        draw.rectangle((0, 0, vw, size), fill=(24, 25, 29, 230))
        draw.rectangle((0, 0, size, vh), fill=(24, 25, 29, 230))
        fnt = font(9)
        step = self.grid_spacing()
        minor = max(1, step // 5)
        text_color = (200, 203, 210, 255)
        tick_color = (150, 153, 160, 255)
        start_x = int((0 - self.ox) / self.zoom // minor * minor) - minor
        x = start_x
        while True:
            vx = self.ox + x * self.zoom
            if vx > vw:
                break
            if vx >= size:
                major = x % step == 0
                draw.line((vx, size - (px(8) if major else px(4)), vx, size), fill=tick_color, width=1)
                if major:
                    glass.text(base, (vx + px(3), px(2)), str(int(x)), fnt, text_color, "la")
            x += minor
        start_y = int((0 - self.oy) / self.zoom // minor * minor) - minor
        y = start_y
        while True:
            vy = self.oy + y * self.zoom
            if vy > vh:
                break
            if vy >= size:
                major = y % step == 0
                draw.line((size - (px(8) if major else px(4)), vy, size, vy), fill=tick_color, width=1)
                if major:
                    label = Image.new("RGBA", (px(60), px(12)), (0, 0, 0, 0))
                    glass.text(label, (0, 0), str(int(y)), fnt, text_color, "la")
                    label = label.crop(label.getbbox() or (0, 0, 1, 1)).rotate(90, expand=True)
                    glass.paste(base, label, (px(2), int(vy) - label.height - px(2)))
            y += minor
        draw.rectangle((0, 0, size, size), fill=(24, 25, 29, 255))
        return base.convert("RGB")

    # ---------------------------------------------------------- selection --
    def _ants_layer(self) -> Layer | None:
        """Marching-ants outline of the selection for the visible part of the picture."""
        doc = self.doc
        if doc is None or doc.selection is None:
            self._stop_ants()
            return None
        vw, vh = self.viewport_size()
        ix0, iy0, ix1, iy1 = self.image_rect()
        vx0, vy0, vx1, vy1 = max(0, ix0), max(0, iy0), min(vw, ix1), min(vh, iy1)
        if vx1 <= vx0 or vy1 <= vy0:
            return None
        key = (id(doc.selection), self.zoom, self.ox, self.oy, vw, vh)
        if self._ants_cache is None or self._ants_cache[0] != key:
            z = self.zoom
            doc_box = ((vx0 - ix0) / z, (vy0 - iy0) / z, min((vx1 - ix0) / z, float(doc.width)), min((vy1 - iy0) / z, float(doc.height)))
            resample = Image.Resampling.NEAREST if z >= 1 else Image.Resampling.BILINEAR
            mask = doc.selection.mask.resize((vx1 - vx0, vy1 - vy0), resample, box=doc_box)
            binary = mask.point(lambda v: 255 if v > 127 else 0)
            edge = ImageChops.subtract(binary, binary.filter(ImageFilter.MinFilter(3)))
            arr = np.asarray(binary)
            e = np.asarray(edge).copy()
            if vx0 == ix0:
                e[:, 0] = np.maximum(e[:, 0], arr[:, 0])
            if vy0 == iy0:
                e[0, :] = np.maximum(e[0, :], arr[0, :])
            if vx1 == ix1:
                e[:, -1] = np.maximum(e[:, -1], arr[:, -1])
            if vy1 == iy1:
                e[-1, :] = np.maximum(e[-1, :], arr[-1, :])
            self._ants_cache = (key, Image.fromarray(e, "L"))
        edge = self._ants_cache[1]
        if edge.getbbox() is None:
            return None
        from ..core.selection import Selection

        ants = Selection.marching_ants(edge, self._ants_phase, period=max(8, px(8)))
        if self._ants_job is None:
            self._ants_job = self.backend.after(200, self._animate_ants)
        return ("ants", vx0, vy0, ants)

    def _animate_ants(self) -> None:
        self._ants_job = None
        if self.doc is not None and self.doc.selection is not None:
            self._ants_phase = (self._ants_phase + 1) % 8
            self.schedule(0)

    def _stop_ants(self) -> None:
        if self._ants_job is not None:
            self.backend.after_cancel(self._ants_job)
            self._ants_job = None

    def render(self) -> None:
        """Compose everything that changed and hand the layers to the backend."""
        self._render_job = None
        self.render_count += 1
        vw, vh = self.viewport_size()
        for panel in self.panels:
            if panel.needs_layout:
                panel.layout(vw, vh)
        frame_changed = self._frame_dirty or self._frame is None or self._frame.size != (vw, vh)
        if frame_changed:
            self._frame = self.compose_frame()
            self._frame_dirty = False
        assert self._frame is not None
        ctx = DrawContext(self.hover_widget, self.pressed_widget, self.focused_widget)
        layers: list[Layer] = [("frame", 0, 0, self._frame)]
        ants = self._ants_layer()
        if ants is not None:
            layers.append(ants)
        tool = self.app.tool
        cursor_layer = tool.cursor_layer(self) if tool is not None and self.pointer is not None and self.hover_panel is None else None
        if cursor_layer is not None:
            layers.append(cursor_layer)
        for panel in self.panels:
            if not panel.visible:
                self._panel_layers.pop(id(panel), None)
                continue
            x0, y0, x1, y1 = panel.rect
            backdrop = Image.new("RGB", (panel.width, panel.height), PALETTE.bg)
            cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(vw, x1), min(vh, y1)
            if cx1 > cx0 and cy1 > cy0:
                backdrop.paste(self._frame.crop((cx0, cy0, cx1, cy1)), (cx0 - x0, cy0 - y0))
            backdrop_key = (panel.rect, hash(backdrop.tobytes()))
            cached = self._panel_layers.get(id(panel))
            if cached is None or panel.dirty or backdrop_key != getattr(panel, "_backdrop_key", None):
                panel._backdrop_key = backdrop_key
                shadow_layer: Layer | None = None
                if panel.shadow:
                    blur = px(14)
                    shadow_img = glass.shadow((panel.width, panel.height), panel.radius, blur, 90)
                    shadow_layer = ("shadow", x0 - blur * 2, y0 - blur * 2 + px(6), shadow_img)
                panel_layer: Layer = ("panel", x0, y0, panel.render(backdrop, ctx))
                cached = (panel_layer, shadow_layer)
                self._panel_layers[id(panel)] = cached
            panel_layer, shadow_layer = cached
            if shadow_layer is not None:
                layers.append(shadow_layer)
            layers.append(panel_layer)
        if self._tooltip_layer is not None:
            layers.append(self._tooltip_layer)
        self.backend.present(layers)

    # ------------------------------------------------------------ events --
    def _pointer_event(self, wx: int, wy: int, shift: bool, ctrl: bool, alt: bool) -> PointerEvent:
        x, y = self.to_doc(wx, wy)
        return PointerEvent(x, y, wx, wy, shift, ctrl, alt)

    def set_space(self, down: bool) -> None:
        if down != self._space:
            self._space = down
            self.update_cursor()

    def update_cursor(self) -> None:
        if self._space and self.doc is not None:
            cursor = "grab"
        elif self.hover_widget is not None:
            cursor = self.hover_widget.cursor if self.hover_widget.enabled else "arrow"
        elif self.hover_panel is not None:
            cursor = "arrow"
        elif self.doc is not None and self.app.tool is not None:
            cursor = self.app.tool.cursor_at(self.pointer) if self.pointer else self.app.tool.cursor
        else:
            cursor = "arrow"
        if cursor != self.cursor:
            self.cursor = cursor
            self.backend.set_cursor(cursor)

    def _guard(self, fn: Callable[[], None]) -> None:
        """Run a tool/widget callback; unexpected errors are reported instead of killing the event loop."""
        try:
            fn()
        except DocumentError as exc:  # an edit the document refuses: tell the user, no traceback
            notify = getattr(self.app, "notify", None)
            if notify is None:
                raise
            notify(str(exc))
        except Exception as exc:  # noqa: BLE001 - the app must survive any tool bug
            report = getattr(self.app, "report_error", None)
            if report is None:
                raise
            report(exc)

    def press(self, wx: int, wy: int, shift: bool = False, ctrl: bool = False, alt: bool = False, button: int = 1) -> None:
        self.hide_tooltip()
        if button == 2:  # middle button pans
            if self.doc is not None and self.panel_at(wx, wy) is None:
                self._pan_anchor = (wx, wy, self.ox, self.oy)
            return
        panel = self.panel_at(wx, wy)
        modal = self.modal_panel()
        if panel is None and modal is not None:
            if modal.dismiss_on_outside:
                self.app.dismiss_modal()
            return
        if panel is not None:
            widget = panel.widget_at(wx, wy)
            self.set_focus(widget if widget is not None and widget.focusable else None)
            if widget is not None and widget.enabled:
                self.pressed_widget = widget
                self._guard(lambda: widget.on_press(*panel.to_content(wx, wy)))
                panel.invalidate()
            return
        self.set_focus(None)
        if self.doc is None:
            return
        if self._space:
            self._pan_anchor = (wx, wy, self.ox, self.oy)
            return
        if self.app.tool is not None:
            self._guard(lambda: self.app.tool.on_press(self._pointer_event(wx, wy, shift, ctrl, alt)))

    def drag(self, wx: int, wy: int, shift: bool = False, ctrl: bool = False, alt: bool = False) -> None:
        self.pointer = (wx, wy)
        if self.pressed_widget is not None:
            panel = self.pressed_widget.panel
            if panel is not None:
                widget = self.pressed_widget
                self._guard(lambda: widget.on_drag(*panel.to_content(wx, wy)))
            return
        if self._pan_anchor is not None:
            ax, ay, ox, oy = self._pan_anchor
            self.ox, self.oy = ox + wx - ax, oy + wy - ay
            self._clamp_offset()
            self.invalidate_frame()
            return
        if self.doc is not None and self.app.tool is not None:
            self._guard(lambda: self.app.tool.on_drag(self._pointer_event(wx, wy, shift, ctrl, alt)))

    def release(self, wx: int, wy: int, shift: bool = False, ctrl: bool = False, alt: bool = False, button: int = 1) -> None:
        if button == 2:
            self._pan_anchor = None
            return
        if self.pressed_widget is not None:
            widget, self.pressed_widget = self.pressed_widget, None
            panel = widget.panel
            if panel is not None:
                cx, cy = panel.to_content(wx, wy)
                self._guard(lambda: widget.on_release(cx, cy, widget.contains(cx, cy)))
                panel.invalidate()
            self._refresh_hover(wx, wy)
            return
        if self._pan_anchor is not None:
            self._pan_anchor = None
            return
        if self.doc is not None and self.app.tool is not None:
            self._guard(lambda: self.app.tool.on_release(self._pointer_event(wx, wy, shift, ctrl, alt)))

    def double(self, wx: int, wy: int, shift: bool = False, ctrl: bool = False, alt: bool = False) -> None:
        panel = self.panel_at(wx, wy)
        if panel is not None:
            widget = panel.widget_at(wx, wy)
            if widget is not None and widget.enabled:
                self._guard(lambda: widget.on_double(*panel.to_content(wx, wy)))
                panel.invalidate()
            return
        if self.doc is not None and self.app.tool is not None and self.modal_panel() is None:
            self._guard(lambda: self.app.tool.on_double(self._pointer_event(wx, wy, shift, ctrl, alt)))

    def motion(self, wx: int, wy: int) -> None:
        self.pointer = (wx, wy)
        self._refresh_hover(wx, wy)
        if self.hover_panel is None and self.doc is not None:
            self.app.pointer_moved(*self.to_doc(wx, wy))
            if self.app.tool is not None:
                self.app.tool.on_motion(self._pointer_event(wx, wy, False, False, False))
                if self.app.tool.has_cursor_layer:
                    self.schedule(0)
        else:
            self.app.pointer_moved(None, None)
            if self.app.tool is not None and self.app.tool.has_cursor_layer:
                self.schedule(0)
        self.update_cursor()

    def _refresh_hover(self, vx: float, vy: float) -> None:
        panel = self.panel_at(vx, vy)
        widget = panel.widget_at(vx, vy) if panel is not None else None
        if isinstance(widget, Segmented) and panel is not None:
            widget.set_hover_position(panel.to_content(vx, vy)[0])
        self._set_hover(panel, widget)

    def _set_hover(self, panel: Panel | None, widget: Widget | None) -> None:
        if widget is not self.hover_widget:
            old = self.hover_widget
            self.hover_widget = widget
            if old is not None:
                old.invalidate()
            if widget is not None:
                widget.invalidate()
            self._schedule_tooltip(widget)
        self.hover_panel = panel
        self.update_cursor()

    def leave(self) -> None:
        self.pointer = None
        self._set_hover(None, None)
        self.app.pointer_moved(None, None)
        if self.app.tool is not None and self.app.tool.has_cursor_layer:
            self.schedule(0)

    def wheel(self, wx: int, wy: int, dx: float, dy: float, ctrl: bool = False) -> None:
        """Scroll by (dx, dy) device pixels; positive dy = content moves down (wheel up)."""
        panel = self.panel_at(wx, wy)
        if panel is not None:
            widget = panel.widget_at(wx, wy)
            if widget is not None and widget.on_wheel(dx, dy):
                return
            panel.scroll(-dy)
            self._refresh_hover(wx, wy)
            return
        if self.doc is None or self.modal_panel() is not None:
            return
        if ctrl and dy:
            self.set_zoom(self.zoom * (1.12 if dy > 0 else 1 / 1.12), (wx, wy))
        else:
            self.pan(dx, dy)

    def key(self, keysym: str, char: str, mods: frozenset[str] = frozenset()) -> bool:
        """Offer a key press to the focused (or hovered) widget; True when consumed."""
        if self.focused_widget is not None and self.focused_widget.on_key(keysym, char, mods):
            return True
        if self.hover_widget is not None and not self.focused_widget and self.hover_widget.on_key(keysym, char, mods):
            return True
        return False

    # ----------------------------------------------------------- tooltip --
    def _schedule_tooltip(self, widget: Widget | None) -> None:
        self.hide_tooltip()
        if widget is not None and widget.tooltip and widget.enabled:
            self._tooltip_job = self.backend.after(TOOLTIP_DELAY, lambda: self.show_tooltip(widget))

    def tooltip_pending(self) -> bool:
        return self._tooltip_job is not None

    def show_tooltip(self, widget: Widget) -> None:
        self._tooltip_job = None
        if widget is not self.hover_widget or widget.panel is None or self.pressed_widget is not None:
            return
        fnt = font(METRICS.small_font_size, "medium")
        tw = int(theme.text_width(widget.tooltip, fnt))
        w, h = tw + px(24), px(28)
        img = glass.rounded_rect((w, h), h // 2, (24, 26, 31, 236), (255, 255, 255, 40), width=max(1, px(1)))
        glass.text(img, (w / 2, h / 2), widget.tooltip, fnt, PALETTE.fg, "mm")
        x0, y0, x1, y1 = widget.panel.widget_rect_in_view(widget)
        vw, vh = self.viewport_size()
        if x1 < px(120):  # left tool strip: show to the right
            tx, ty = x1 + px(10), (y0 + y1) // 2 - h // 2
        elif y0 - h - px(8) >= px(4) and y0 > px(80):  # prefer above, so the tooltip never covers the next control
            tx, ty = (x0 + x1) // 2 - w // 2, y0 - h - px(6)
        else:
            tx, ty = (x0 + x1) // 2 - w // 2, y1 + px(8)
        tx = max(px(4), min(vw - w - px(4), tx))
        self._tooltip_layer = ("tooltip", int(tx), int(ty), img)
        self.tooltip_widget = widget
        self.schedule(0)

    def hide_tooltip(self) -> None:
        if self._tooltip_job is not None:
            self.backend.after_cancel(self._tooltip_job)
            self._tooltip_job = None
        if self.tooltip_widget is not None:
            self.tooltip_widget = None
            self._tooltip_layer = None
            self.schedule(0)

    def shutdown(self) -> None:
        self._stop_ants()
        for job in (self._render_job, self._tooltip_job):
            if job is not None:
                self.backend.after_cancel(job)
        self._render_job = self._tooltip_job = None


def wheel_pixels_from_notches(notches: float) -> float:
    """Pixels to scroll for a number of mouse-wheel notches."""
    return notches * 48.0 * theme.SCALE
