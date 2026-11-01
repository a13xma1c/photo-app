"""Interactive tools: navigation, crop, move, selections, painting, gradient, text, shapes, eyedropper.

A tool receives :class:`PointerEvent` objects in document coordinates, may
paint a transient overlay into the frame (``paint_overlay``), may provide a
lightweight cursor layer that follows the pointer (``cursor_layer``) and may
use ``app.begin_preview`` / ``app.end_preview`` to show uncommitted results
while dragging.  Each tool also describes its option widgets for the tool
options bar (``build_options``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageDraw

from ..core import painting
from ..core.brushes import BrushPreset
from ..core.document import Document, DocumentError
from ..core.painting import GRADIENT_TYPES, SHAPE_TYPES, Stroke, Transform
from ..core.selection import COMBINE_MODES, Selection
from ..core.three_d import SHAPES, Scene, bounding_box
from ..core.three_d import render as render_scene
from .i18n import tr
from .surface import Layer, PointerEvent, Surface
from .theme import PALETTE, px, text_width
from .ui import Button, Label, NumberField, Segmented, Slider, Spacer, Toggle, Widget

if TYPE_CHECKING:
    from .editor import Editor

HANDLE_BASE = 7  # half size of a crop handle, in 1x view pixels
ASPECTS: dict[str, float | None] = {"Free": None, "1:1": 1.0, "4:3": 4 / 3, "3:2": 3 / 2, "16:9": 16 / 9}
SNAP_BASE = 7  # snapping distance in 1x view pixels


def _selection_mode(e: PointerEvent, default: str) -> str:
    if e.shift and e.alt:
        return "intersect"
    if e.shift:
        return "add"
    if e.alt:
        return "subtract"
    return default


def _mode_option(var: dict[str, Any], key: str = "mode") -> Widget:
    labels = [tr("Replace"), tr("Add"), tr("Subtract"), tr("Intersect")]
    seg = Segmented(labels, COMBINE_MODES.index(var[key]), lambda i: var.__setitem__(key, COMBINE_MODES[i]), width=280)
    seg.tooltip = tr("Shift adds, Alt subtracts, Shift+Alt intersects")
    return seg


def _option_slider(
    label: str, lo: int, hi: int, var: dict[str, Any], key: str, on_change: Callable[[int], None] | None = None, width: int = 220
) -> Slider:
    def changed(value: float) -> None:
        var[key] = int(value)
        if on_change:
            on_change(int(value))

    slider = Slider(label, lo, hi, var[key], var[key], on_change=changed, fmt="{:d}", compact=True)
    slider.width = px(width)
    slider.flex = 0
    return slider


def _option_toggle(label: str, var: dict[str, Any], key: str, width: int = 150) -> Toggle:
    toggle = Toggle(label, var[key], lambda v: var.__setitem__(key, v))
    toggle.width = px(width)
    toggle.flex = 0
    return toggle


def _dashed_rect(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float]) -> None:
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    draw.rectangle((x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)), outline=(0, 0, 0, 160), width=max(1, px(1)) + 2)
    draw.rectangle((x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)), outline=(255, 255, 255, 230), width=max(1, px(1)))


class Tool:
    id = "tool"
    name = "Tool"
    icon = "hand"
    key = ""  # single-letter shortcut (lower case: plain key, upper case: Shift+key)
    shortcut_label = ""  # shown in the tooltip when the tool has no single-letter key
    cursor = "arrow"
    hint = ""
    has_cursor_layer = False
    options_follow_layer = False  # rebuild the options bar when the active layer changes
    edits_smart_layers = False  # may work on 3D / text layers (whose pixels are re-rendered from settings)

    def __init__(self, app: Editor) -> None:
        self.app = app
        self.options: dict[str, Any] = {}  # persisted per tool, edited through the options bar

    @property
    def doc(self) -> Document:
        """The open document - tools only receive events while one is open."""
        doc = self.app.doc
        if doc is None:
            raise DocumentError("No picture is open")
        return doc

    @property
    def view(self) -> Surface:
        return self.app.view

    def activate(self) -> None:
        pass

    def deactivate(self) -> None:
        self.cancel()

    def build_options(self) -> list[Widget]:
        return []

    def cursor_at(self, pointer: tuple[int, int] | None) -> str:
        return self.cursor

    def on_press(self, e: PointerEvent) -> None:
        pass

    def on_drag(self, e: PointerEvent) -> None:
        pass

    def on_release(self, e: PointerEvent) -> None:
        pass

    def on_double(self, e: PointerEvent) -> None:
        pass

    def on_motion(self, e: PointerEvent) -> None:
        pass

    def on_key(self, keysym: str, mods: frozenset[str] = frozenset()) -> bool:
        return False

    def cancel(self) -> None:
        pass

    def paint_overlay(self, frame: Image.Image, view: Surface) -> Image.Image:
        return frame

    def cursor_layer(self, view: Surface) -> Layer | None:
        return None

    def _editable(self) -> bool:
        """Whether the active layer can take pixel (or mask) edits right now; otherwise the reason is shown."""
        doc = self.app.doc
        if doc is None:
            return False
        if doc.active_layer.locked:
            self.app.notify(tr("The layer is locked - unlock it in the Layers tab to edit"))
            return False
        if doc.active_layer.kind == "group" and not doc.editing_mask:
            self.app.notify(tr("A group has no pixels - select a layer inside it"))
            return False
        if doc.is_smart() and not doc.editing_mask and not self.edits_smart_layers:
            what = tr("3D object") if doc.active_layer.kind == "3d" else tr("text layer")
            self.app.notify(tr("This is a {what} - rasterize it (Layer ▸ Rasterize Layer) to paint on its pixels").format(what=what))
            return False
        return True


# ------------------------------------------------------------ navigation --
class HandTool(Tool):
    id, name, icon, key, cursor = "hand", "Hand", "hand", "h", "grab"
    hint = "Drag to pan · double-click to fit"

    def __init__(self, app) -> None:
        super().__init__(app)
        self._anchor: tuple[int, int, int, int] | None = None

    def on_press(self, e: PointerEvent) -> None:
        self._anchor = (e.wx, e.wy, self.view.ox, self.view.oy)

    def on_drag(self, e: PointerEvent) -> None:
        if self._anchor is None:
            return
        ax, ay, ox, oy = self._anchor
        self.view.ox, self.view.oy = ox + e.wx - ax, oy + e.wy - ay
        self.view._clamp_offset()
        self.view.invalidate_frame()

    def on_release(self, e: PointerEvent) -> None:
        self._anchor = None

    def on_double(self, e: PointerEvent) -> None:
        self.view.fit()


class ZoomTool(Tool):
    id, name, icon, key, cursor = "zoom", "Zoom", "zoom", "z", "zoom-in"
    hint = "Click to zoom in · Alt-click to zoom out"

    def on_press(self, e: PointerEvent) -> None:
        if e.alt:
            self.view.zoom_out((e.wx, e.wy))
        else:
            self.view.zoom_in((e.wx, e.wy))

    def on_double(self, e: PointerEvent) -> None:
        self.view.fit()


# ------------------------------------------------------------------ crop --
class CropTool(Tool):
    id, name, icon, key, cursor = "crop", "Crop", "crop", "c", "cross"
    hint = "Drag to choose an area · Enter applies · Esc cancels"

    def __init__(self, app) -> None:
        super().__init__(app)
        self.rect: tuple[float, float, float, float] | None = None
        self.aspect: float | None = None
        self._mode: str | None = None  # "new" | "move" | handle name like "nw", "e"
        self._start: tuple[float, float] | None = None
        self._orig: tuple[float, float, float, float] | None = None

    def activate(self) -> None:
        self.app.crop_changed()

    def set_aspect(self, aspect: float | None) -> None:
        self.aspect = aspect
        if self.rect is not None and aspect:
            x0, y0, x1, y1 = self.rect
            w = x1 - x0
            h = w / aspect
            if y0 + h > self.doc.height:
                h = self.doc.height - y0
                w = h * aspect
            self.rect = (x0, y0, x0 + w, y0 + h)
            self.view.invalidate_frame()
            self.app.crop_changed()

    # snapping ----------------------------------------------------------------
    def _snap_targets(self, horizontal: bool) -> list[float]:
        size = self.doc.width if horizontal else self.doc.height
        targets = [0.0, size / 3, size / 2, size * 2 / 3, float(size)]
        if self.view.show_grid:
            step = self.view.grid_spacing()
            targets += [float(v) for v in range(0, int(size) + 1, step)]
        return targets

    def _snap(self, value: float, horizontal: bool) -> float:
        if not getattr(self.app, "snap_enabled", True) or self.aspect:
            return value
        tolerance = px(SNAP_BASE) / max(self.view.zoom, 1e-6)
        best, best_dist = value, tolerance
        for target in self._snap_targets(horizontal):
            dist = abs(target - value)
            if dist < best_dist:
                best, best_dist = target, dist
        return best

    def _handles(self) -> dict[str, tuple[float, float]]:
        if self.rect is None:
            return {}
        x0, y0 = self.view.to_view(self.rect[0], self.rect[1])
        x1, y1 = self.view.to_view(self.rect[2], self.rect[3])
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        return {"nw": (x0, y0), "n": (xm, y0), "ne": (x1, y0), "e": (x1, ym), "se": (x1, y1), "s": (xm, y1), "sw": (x0, y1), "w": (x0, ym)}

    def _handle_at(self, wx: float, wy: float) -> str | None:
        reach = px(HANDLE_BASE + 3)
        for name, (hx, hy) in self._handles().items():
            if abs(wx - hx) <= reach and abs(wy - hy) <= reach:
                return name
        return None

    def _inside(self, x: float, y: float) -> bool:
        if self.rect is None:
            return False
        x0, y0, x1, y1 = self.rect
        return x0 <= x <= x1 and y0 <= y <= y1

    def cursor_at(self, pointer: tuple[int, int] | None) -> str:
        if pointer is None or self.rect is None:
            return self.cursor
        handle = self._handle_at(*pointer)
        if handle:
            cursors = {"n": "resize-v", "s": "resize-v", "e": "resize-h", "w": "resize-h", "nw": "resize-nwse", "se": "resize-nwse"}
            return cursors.get(handle, "resize-nesw")
        if self._inside(*self.view.to_doc(*pointer)):
            return "move"
        return self.cursor

    def on_press(self, e: PointerEvent) -> None:
        handle = self._handle_at(e.wx, e.wy)
        if handle and self.rect is not None:
            self._mode = handle
        elif self._inside(e.x, e.y):
            self._mode = "move"
        else:
            self._mode = "new"
            self.rect = None
        # A drag may start outside the picture: the crop area is the part that overlaps it.
        self._start = e.clamped(self.doc.width, self.doc.height) if self._mode == "new" else (e.x, e.y)
        self._orig = self.rect

    def on_drag(self, e: PointerEvent) -> None:
        if self._mode is None or self._start is None:
            return
        w_, h_ = self.doc.width, self.doc.height
        sx, sy = self._start
        if self._mode == "new":
            x1, y1 = e.clamped(w_, h_)
            x1, y1 = self._snap(x1, True), self._snap(y1, False)
            x0, y0 = self._snap(sx, True), self._snap(sy, False)
            rect = self._constrain((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), anchor=(x0, y0))
            self.rect = rect if rect[2] - rect[0] >= 1 and rect[3] - rect[1] >= 1 else None
        elif self._mode == "move" and self._orig is not None:
            ox0, oy0, ox1, oy1 = self._orig
            dx = min(max(e.x - sx, -ox0), w_ - ox1)
            dy = min(max(e.y - sy, -oy0), h_ - oy1)
            self.rect = (ox0 + dx, oy0 + dy, ox1 + dx, oy1 + dy)
        elif self._orig is not None:
            px_, py_ = e.clamped(w_, h_)
            self.rect = self._resize(self._orig, self._mode, (self._snap(px_, True), self._snap(py_, False)))
        self.view.invalidate_frame()
        self.app.crop_changed()

    def _resize(self, orig, handle: str, point: tuple[float, float]) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = orig
        px_, py_ = point
        if "w" in handle:
            x0 = min(px_, x1 - 1)
        if "e" in handle:
            x1 = max(px_, x0 + 1)
        if "n" in handle:
            y0 = min(py_, y1 - 1)
        if "s" in handle:
            y1 = max(py_, y0 + 1)
        rect = (x0, y0, x1, y1)
        if self.aspect:
            ax = x1 if "w" in handle else x0
            ay = y1 if "n" in handle else y0
            rect = self._constrain(rect, anchor=(ax, ay), prefer_width=handle not in ("n", "s"))
        return rect

    def _constrain(self, rect, anchor: tuple[float, float], prefer_width: bool = True) -> tuple[float, float, float, float]:
        if not self.aspect:
            return rect
        x0, y0, x1, y1 = rect
        w, h = x1 - x0, y1 - y0
        if prefer_width:
            h = w / self.aspect
        else:
            w = h * self.aspect
        ax, ay = anchor
        w_, h_ = self.doc.width, self.doc.height
        right = ax <= (x0 + x1) / 2
        down = ay <= (y0 + y1) / 2
        max_w = (w_ - ax) if right else ax
        max_h = (h_ - ay) if down else ay
        if w > max_w:
            w = max_w
            h = w / self.aspect
        if h > max_h:
            h = max_h
            w = h * self.aspect
        nx0 = ax if right else ax - w
        ny0 = ay if down else ay - h
        return (nx0, ny0, nx0 + w, ny0 + h)

    def on_release(self, e: PointerEvent) -> None:
        self._mode = self._start = self._orig = None
        if self.rect is not None and (self.rect[2] - self.rect[0] < 2 or self.rect[3] - self.rect[1] < 2):
            self.rect = None
            self.view.invalidate_frame()
        self.app.crop_changed()

    def on_double(self, e: PointerEvent) -> None:
        if self.rect is not None and self._inside(e.x, e.y):
            self.apply()

    def on_key(self, keysym: str, mods: frozenset[str] = frozenset()) -> bool:
        if keysym in ("Return", "KP_Enter") and self.rect is not None:
            self.apply()
            return True
        return False

    def apply(self) -> None:
        if self.rect is None or self.app.doc is None:
            return
        rect = self.rect
        self.rect = None
        self.app.crop_changed()
        try:
            self.doc.crop(rect)
        except DocumentError as exc:
            self.app.notify(str(exc))
            return
        self.app.after_edit(fit=True)

    def cancel(self) -> None:
        had = self.rect is not None
        self.rect = None
        self._mode = self._start = self._orig = None
        if had and self.view is not None:
            self.view.invalidate_frame()
        self.app.crop_changed()

    def size_text(self) -> str:
        if self.rect is None:
            return ""
        return f"{round(self.rect[2] - self.rect[0])} × {round(self.rect[3] - self.rect[1])} px"

    def paint_overlay(self, frame: Image.Image, view: Surface) -> Image.Image:
        if self.rect is None:
            return frame
        x0, y0 = view.to_view(self.rect[0], self.rect[1])
        x1, y1 = view.to_view(self.rect[2], self.rect[3])
        x0, y0 = int(round(x0)), int(round(y0))
        x1, y1 = max(int(round(x1)), x0 + 1), max(int(round(y1)), y0 + 1)
        ix0, iy0, ix1, iy1 = view.image_rect()
        base = frame.convert("RGBA")
        dim = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(dim)
        draw.rectangle((ix0, iy0, ix1, iy1), fill=PALETTE.crop_dim)
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=(0, 0, 0, 0))
        base.alpha_composite(dim)
        draw = ImageDraw.Draw(base)
        for i in (1, 2):  # rule of thirds
            gx = x0 + (x1 - x0) * i // 3
            gy = y0 + (y1 - y0) * i // 3
            draw.line((gx, y0, gx, y1), fill=(255, 255, 255, 70), width=max(1, px(1)))
            draw.line((x0, gy, x1, gy), fill=(255, 255, 255, 70), width=max(1, px(1)))
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 255, 255, 230), width=max(1, px(1)))
        handle = px(HANDLE_BASE)
        for hx, hy in self._handles().values():
            hx, hy = int(round(hx)), int(round(hy))
            box = (hx - handle, hy - handle, hx + handle, hy + handle)
            draw.rounded_rectangle(box, px(3), fill=(255, 255, 255, 240), outline=(0, 0, 0, 90), width=max(1, px(1)))
        return base.convert("RGB")


# ------------------------------------------------------------------ move --
class MoveTool(Tool):
    id, name, icon, key, cursor = "move", "Move", "move", "v", "move"
    hint = "Drag to move the layer · arrow keys nudge (Shift: 10 px)"
    edits_smart_layers = True

    def __init__(self, app) -> None:
        super().__init__(app)
        self._start: tuple[float, float] | None = None
        self._delta = (0, 0)
        self._image: Image.Image | None = None
        self.options = {"auto_select": False}

    def build_options(self) -> list[Widget]:
        return [_option_toggle(tr("Auto-select layer"), self.options, "auto_select", 190)]

    def on_press(self, e: PointerEvent) -> None:
        if self.app.doc is None:
            return
        if self.options["auto_select"]:
            self._auto_select(e)
        if not self._editable():
            return
        self._start = (e.x, e.y)
        self._delta = (0, 0)
        self._image = self.doc.edit_image()

    def _auto_select(self, e: PointerEvent) -> None:
        ix, iy = int(e.x), int(e.y)
        if not (0 <= ix < self.doc.width and 0 <= iy < self.doc.height):
            return
        for index in range(len(self.doc.layers) - 1, -1, -1):
            layer = self.doc.layers[index]
            if layer.visible and layer.image.getchannel("A").getpixel((ix, iy)):
                if index != self.doc.active_index:
                    self.doc.set_active(index)
                    self.app.refresh_all()
                return

    def on_drag(self, e: PointerEvent) -> None:
        if self._start is None or self._image is None:
            return
        dx, dy = int(round(e.x - self._start[0])), int(round(e.y - self._start[1]))
        if (dx, dy) == self._delta:
            return
        self._delta = (dx, dy)
        self.app.begin_preview(self.doc.active_layer.id, painting.offset_image(self._image, dx, dy))

    def on_release(self, e: PointerEvent) -> None:
        if self._start is None:
            return
        dx, dy = self._delta
        self._start = self._image = None
        self.app.end_preview()
        if dx or dy:
            self.doc.offset_active(dx, dy)
            self.app.after_edit()

    def on_key(self, keysym: str, mods: frozenset[str] = frozenset()) -> bool:
        deltas = {"Left": (-1, 0), "Right": (1, 0), "Up": (0, -1), "Down": (0, 1)}
        if keysym not in deltas or self.app.doc is None or not self._editable():
            return False
        step = 10 if "shift" in mods else 1
        dx, dy = deltas[keysym]
        self.doc.offset_active(dx * step, dy * step, label="Nudge")
        self.app.after_edit()
        return True

    def cancel(self) -> None:
        self._start = self._image = None
        self.app.end_preview()


# -------------------------------------------------------------- transform --
def _box_points(box: tuple[float, float, float, float]) -> dict[str, tuple[float, float]]:
    """Corner and edge-midpoint handle positions of *box*, keyed nw, n, ne, e, se, s, sw, w."""
    x0, y0, x1, y1 = box
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    return {"nw": (x0, y0), "n": (xm, y0), "ne": (x1, y0), "e": (x1, ym), "se": (x1, y1), "s": (xm, y1), "sw": (x0, y1), "w": (x0, ym)}


class TransformTool(Tool):
    """Free Transform: scale, rotate and move the selected pixels of the active layer with handles."""

    id, name, icon, key, cursor = "transform", "Free Transform", "transform", "", "arrow"
    shortcut_label = "Mod+T"
    hint = "Drag handles to scale (Shift keeps the aspect) · drag outside a corner to rotate · drag inside to move · Enter applies"
    edits_smart_layers = True
    ROTATE_REACH = 26  # device pixels beyond a corner handle that still rotate

    def __init__(self, app: Editor) -> None:
        super().__init__(app)
        self.options = {"keep_aspect": True}
        self.transform: Transform | None = None
        self._mode: str | None = None  # "move" | "rotate" | handle name
        self._start: tuple[float, float] | None = None
        self._orig: Transform | None = None
        self._start_angle = 0.0
        self._fields: dict[str, NumberField] = {}
        self._syncing = False

    # lifecycle ---------------------------------------------------------------
    def activate(self) -> None:
        self.transform = None
        if self.app.doc is None:
            return
        box = self.doc.content_box()
        if box is None:
            self.app.notify(tr("The layer is empty - nothing to transform"))
            return
        if self.doc.active_layer.locked:
            self.app.notify(tr("The layer is locked - unlock it in the Layers tab to edit"))
            return
        self.transform = Transform(tuple(float(v) for v in box))  # type: ignore[arg-type]
        self.view.invalidate_frame()

    def build_options(self) -> list[Widget]:
        fw = 78
        self._fields = {
            "w": NumberField(tr("W"), 100, -2000, 2000, lambda v: self._field_changed("scale_x", v / 100), suffix="%", field_width=fw),
            "h": NumberField(tr("H"), 100, -2000, 2000, lambda v: self._field_changed("scale_y", v / 100), suffix="%", field_width=fw),
            "angle": NumberField(tr("Angle"), 0, -360, 360, lambda v: self._field_changed("angle", float(v)), suffix="°", field_width=fw),
        }
        for field in self._fields.values():
            field.width = px(fw + 8) + int(text_width(field.label, field.font))
            field.flex = 0
        apply = Button(tr("Apply"), self.apply, primary=True, tooltip="Enter", small=True)
        cancel = Button(tr("Cancel"), self.cancel, tooltip="Esc", small=True)
        flip_h = Button(tr("Flip H"), lambda: self._flip(True), small=True)
        flip_v = Button(tr("Flip V"), lambda: self._flip(False), small=True)
        self._sync_fields()
        return [
            self._fields["w"],
            self._fields["h"],
            self._fields["angle"],
            _option_toggle(tr("Keep aspect"), self.options, "keep_aspect", 130),
            flip_h,
            flip_v,
            Spacer(width=6),
            cancel,
            apply,
        ]

    def _sync_fields(self, skip: str | None = None) -> None:
        if not self._fields or self.transform is None:
            return
        values = {
            "scale_x": ("w", round(self.transform.scale_x * 100)),
            "scale_y": ("h", round(self.transform.scale_y * 100)),
            "angle": ("angle", round(self.transform.angle)),
        }
        self._syncing = True
        try:
            for attr, (name, value) in values.items():
                if attr != skip:
                    self._fields[name].set_value(value)
        finally:
            self._syncing = False

    def _field_changed(self, attr: str, value: float) -> None:
        if self._syncing or self.transform is None:
            return
        if attr in ("scale_x", "scale_y") and abs(value) < 0.01:
            return
        t = self.transform
        if attr == "angle":
            new = replace(t, angle=value)
        else:
            sx = value if attr == "scale_x" else t.scale_x
            sy = value if attr == "scale_y" else t.scale_y
            if self.options["keep_aspect"]:
                sx, sy = (sx, math.copysign(abs(sx), sy)) if attr == "scale_x" else (math.copysign(abs(sy), sx), sy)
            new = replace(t, scale_x=sx, scale_y=sy)
        # while typing, only the *other* fields are refreshed so the caret keeps its place
        self._set(new, sync=False)
        self._sync_fields(skip=attr)

    def _flip(self, horizontal: bool) -> None:
        if self.transform is None:
            return
        t = self.transform
        self._set(replace(t, scale_x=-t.scale_x) if horizontal else replace(t, scale_y=-t.scale_y))

    def _set(self, transform: Transform, sync: bool = True) -> None:
        self.transform = transform
        if sync:
            self._sync_fields()
        self.app.begin_preview(self.doc.active_layer.id, self.doc.preview_transform(transform))

    # geometry ----------------------------------------------------------------
    def _handles(self) -> dict[str, tuple[float, float]]:
        """Handle positions in view pixels, keyed like the crop tool's (nw, n, ne, e, se, s, sw, w)."""
        if self.transform is None:
            return {}
        points = _box_points(self.transform.box)
        return {name: self.view.to_view(*self.transform.apply(x, y)) for name, (x, y) in points.items()}

    def _handle_at(self, wx: float, wy: float) -> str | None:
        reach = px(HANDLE_BASE + 3)
        for name, (hx, hy) in self._handles().items():
            if abs(wx - hx) <= reach and abs(wy - hy) <= reach:
                return name
        return None

    def _inside(self, x: float, y: float) -> bool:
        if self.transform is None:
            return False
        sx, sy = self.transform.unapply(x, y)
        x0, y0, x1, y1 = self.transform.box
        return x0 <= sx <= x1 and y0 <= sy <= y1

    def _near_corner(self, wx: float, wy: float) -> bool:
        reach = px(self.ROTATE_REACH + HANDLE_BASE)
        handles = self._handles()
        return any(math.hypot(wx - handles[n][0], wy - handles[n][1]) <= reach for n in ("nw", "ne", "se", "sw"))

    def cursor_at(self, pointer: tuple[int, int] | None) -> str:
        if pointer is None or self.transform is None:
            return self.cursor
        handle = self._handle_at(*pointer)
        if handle:
            cursors = {"n": "resize-v", "s": "resize-v", "e": "resize-h", "w": "resize-h", "nw": "resize-nwse", "se": "resize-nwse"}
            return cursors.get(handle, "resize-nesw")
        if self._inside(*self.view.to_doc(*pointer)):
            return "move"
        if self._near_corner(*pointer):
            return "rotate"
        return self.cursor

    # interaction -------------------------------------------------------------
    def on_press(self, e: PointerEvent) -> None:
        if self.transform is None:
            return
        handle = self._handle_at(e.wx, e.wy)
        if handle:
            self._mode = handle
        elif self._inside(e.x, e.y):
            self._mode = "move"
        elif self._near_corner(e.wx, e.wy):
            self._mode = "rotate"
            cx, cy = self.transform.apply(*self.transform.center)
            self._start_angle = math.degrees(math.atan2(cy - e.y, e.x - cx))
        else:
            self._mode = None
            return
        self._start = (e.x, e.y)
        self._orig = self.transform

    def on_drag(self, e: PointerEvent) -> None:
        if self._mode is None or self._start is None or self._orig is None:
            return
        orig = self._orig
        if self._mode == "move":
            new = replace(orig, dx=orig.dx + e.x - self._start[0], dy=orig.dy + e.y - self._start[1])
        elif self._mode == "rotate":
            cx, cy = orig.apply(*orig.center)
            angle = orig.angle + math.degrees(math.atan2(cy - e.y, e.x - cx)) - self._start_angle
            if e.shift:
                angle = round(angle / 15) * 15
            new = replace(orig, angle=(angle + 180) % 360 - 180)
        else:
            new = self._scaled(orig, self._mode, (e.x, e.y), keep_aspect=e.shift or self.options["keep_aspect"], centered=e.alt)
        self._set(new)

    def _scaled(self, orig: Transform, handle: str, point: tuple[float, float], keep_aspect: bool, centered: bool) -> Transform:
        """Scale so the dragged handle follows *point* while the opposite handle (or the centre) stays put."""
        x0, y0, x1, y1 = orig.box
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        hx, hy = _box_points(orig.box)[handle]
        ax, ay = (xm, ym) if centered else (x0 + x1 - hx, y0 + y1 - hy)  # anchor: centre or opposite point
        # pointer in the rotated frame, relative to the anchor's screen position
        a_dest = orig.apply(ax, ay)
        rad = math.radians(orig.angle)
        cos, sin = math.cos(rad), math.sin(rad)
        vx, vy = point[0] - a_dest[0], point[1] - a_dest[1]
        ux, uy = vx * cos - vy * sin, vx * sin + vy * cos  # undo the rotation
        sx, sy = orig.scale_x, orig.scale_y
        if hx != ax:
            sx = ux / (hx - ax)
        if hy != ay:
            sy = uy / (hy - ay)
        if keep_aspect:
            if handle in ("n", "s"):
                sx = math.copysign(abs(sy), sx)
            elif handle in ("e", "w"):
                sy = math.copysign(abs(sx), sy)
            else:
                mag = max(abs(sx), abs(sy)) if abs(ux) > abs(uy) * 1.5 or abs(uy) > abs(ux) * 1.5 else (abs(sx) + abs(sy)) / 2
                sx, sy = math.copysign(mag, sx), math.copysign(mag, sy)
        sx = math.copysign(max(abs(sx), 0.01), sx or 1)
        sy = math.copysign(max(abs(sy), 0.01), sy or 1)
        scaled = replace(orig, scale_x=sx, scale_y=sy, dx=0.0, dy=0.0)
        nx, ny = scaled.apply(ax, ay)
        return replace(scaled, dx=a_dest[0] - nx, dy=a_dest[1] - ny)

    def on_release(self, e: PointerEvent) -> None:
        self._mode = self._start = self._orig = None

    def on_double(self, e: PointerEvent) -> None:
        if self.transform is not None and self._inside(e.x, e.y):
            self.apply()

    def on_key(self, keysym: str, mods: frozenset[str] = frozenset()) -> bool:
        if self.transform is None:
            return False
        if keysym in ("Return", "KP_Enter"):
            self.apply()
            return True
        deltas = {"Left": (-1, 0), "Right": (1, 0), "Up": (0, -1), "Down": (0, 1)}
        if keysym in deltas:
            step = 10 if "shift" in mods else 1
            dx, dy = deltas[keysym]
            self._set(replace(self.transform, dx=self.transform.dx + dx * step, dy=self.transform.dy + dy * step))
            return True
        return False

    def apply(self) -> None:
        transform = self.transform
        if transform is None or self.app.doc is None:
            return
        self.transform = None
        self.app.end_preview()
        try:
            self.doc.transform_active(transform)
        except DocumentError as exc:
            self.app.notify(str(exc))
        self.app.after_edit()
        self.activate()  # ready for the next transform of the same layer

    def cancel(self) -> None:
        had = self.transform is not None
        self.transform = None
        self._mode = self._start = self._orig = None
        self.app.end_preview()
        if had:
            self.view.invalidate_frame()
            self.app.refresh_all()

    def deactivate(self) -> None:
        self.cancel()

    def paint_overlay(self, frame: Image.Image, view: Surface) -> Image.Image:
        if self.transform is None:
            return frame
        base = frame.convert("RGBA")
        draw = ImageDraw.Draw(base)
        pts = [view.to_view(x, y) for x, y in self.transform.corners()]
        draw.polygon(pts, outline=(0, 0, 0, 160), width=max(1, px(1)) + 2)
        draw.polygon(pts, outline=(255, 255, 255, 230), width=max(1, px(1)))
        handle = px(HANDLE_BASE)
        for hx, hy in self._handles().values():
            hx, hy = int(round(hx)), int(round(hy))
            box = (hx - handle, hy - handle, hx + handle, hy + handle)
            draw.rounded_rectangle(box, px(3), fill=(255, 255, 255, 240), outline=(0, 0, 0, 90), width=max(1, px(1)))
        cx, cy = view.to_view(*self.transform.apply(*self.transform.center))
        r = px(5)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(255, 255, 255, 230), width=max(1, px(1)))
        draw.line((cx - r * 1.6, cy, cx + r * 1.6, cy), fill=(255, 255, 255, 230), width=max(1, px(1)))
        draw.line((cx, cy - r * 1.6, cx, cy + r * 1.6), fill=(255, 255, 255, 230), width=max(1, px(1)))
        return base.convert("RGB")


# ------------------------------------------------------------ selections --
class MarqueeTool(Tool):
    id, name, icon, key, cursor = "marquee", "Marquee", "marquee", "m", "cross"
    hint = "Drag to select · Shift adds · Alt subtracts · click to deselect"

    def __init__(self, app) -> None:
        super().__init__(app)
        self.options = {"shape": "rectangle", "mode": "replace", "feather": 0}
        self._start: tuple[float, float] | None = None
        self._rect: tuple[float, float, float, float] | None = None
        self._press_mode = "replace"

    def build_options(self) -> list[Widget]:
        shape = Segmented(
            [tr("Rectangle"), tr("Ellipse")],
            0 if self.options["shape"] == "rectangle" else 1,
            lambda i: self.options.__setitem__("shape", ["rectangle", "ellipse"][i]),
            width=180,
        )
        return [shape, _mode_option(self.options), _option_slider(tr("Feather"), 0, 100, self.options, "feather", width=190)]

    def on_press(self, e: PointerEvent) -> None:
        self._press_mode = _selection_mode(e, self.options["mode"])
        self._start = e.clamped(self.doc.width, self.doc.height)
        self._rect = None

    def on_drag(self, e: PointerEvent) -> None:
        if self._start is None:
            return
        x0, y0 = self._start
        x1, y1 = e.clamped(self.doc.width, self.doc.height)
        if e.ctrl:
            side = max(abs(x1 - x0), abs(y1 - y0))
            x1 = x0 + math.copysign(side, x1 - x0 or 1)
            y1 = y0 + math.copysign(side, y1 - y0 or 1)
            x1, y1 = min(max(x1, 0), self.doc.width), min(max(y1, 0), self.doc.height)
        self._rect = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        self.view.invalidate_frame()

    def on_release(self, e: PointerEvent) -> None:
        if self._start is None:
            return
        rect, mode = self._rect, self._press_mode
        self._start = self._rect = None
        if rect is None or rect[2] - rect[0] < 1 or rect[3] - rect[1] < 1:
            if mode == "replace" and self.doc.selection is not None:
                self.doc.deselect()
                self.app.after_edit()
            else:
                self.view.invalidate_frame()
            return
        factory = Selection.rectangle if self.options["shape"] == "rectangle" else Selection.ellipse
        sel = factory(self.doc.size, rect)
        if self.options["feather"] > 0:
            sel = sel.feather(self.options["feather"])
        self.doc.set_selection(sel, mode)
        self.app.after_edit()

    def paint_overlay(self, frame: Image.Image, view: Surface) -> Image.Image:
        if self._rect is None:
            return frame
        base = frame.convert("RGBA")
        draw = ImageDraw.Draw(base)
        box = view.to_view(self._rect[0], self._rect[1]) + view.to_view(self._rect[2], self._rect[3])
        if self.options["shape"] == "rectangle":
            _dashed_rect(draw, box)
        else:
            x0, y0, x1, y1 = (int(round(v)) for v in box)
            draw.ellipse((x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)), outline=(255, 255, 255, 230), width=max(1, px(1)))
        return base.convert("RGB")

    def cancel(self) -> None:
        had = self._rect is not None
        self._start = self._rect = None
        if had:
            self.view.invalidate_frame()


class LassoTool(Tool):
    id, name, icon, key, cursor = "lasso", "Lasso", "lasso", "l", "cross"
    hint = "Drag a freehand outline · release to close the selection"

    def __init__(self, app) -> None:
        super().__init__(app)
        self.options = {"mode": "replace", "feather": 0}
        self._points: list[tuple[float, float]] = []
        self._press_mode = "replace"

    def build_options(self) -> list[Widget]:
        return [_mode_option(self.options), _option_slider(tr("Feather"), 0, 100, self.options, "feather", width=190)]

    def on_press(self, e: PointerEvent) -> None:
        self._press_mode = _selection_mode(e, self.options["mode"])
        self._points = [e.clamped(self.doc.width, self.doc.height)]

    def on_drag(self, e: PointerEvent) -> None:
        if not self._points:
            return
        pt = e.clamped(self.doc.width, self.doc.height)
        if math.hypot(pt[0] - self._points[-1][0], pt[1] - self._points[-1][1]) * self.view.zoom >= 2:
            self._points.append(pt)
            self.view.invalidate_frame()

    def on_release(self, e: PointerEvent) -> None:
        points, mode = self._points, self._press_mode
        self._points = []
        if len(points) < 3:
            if mode == "replace" and self.doc.selection is not None:
                self.doc.deselect()
                self.app.after_edit()
            else:
                self.view.invalidate_frame()
            return
        sel = Selection.polygon(self.doc.size, points)
        if self.options["feather"] > 0:
            sel = sel.feather(self.options["feather"])
        self.doc.set_selection(sel, mode)
        self.app.after_edit()

    def paint_overlay(self, frame: Image.Image, view: Surface) -> Image.Image:
        if len(self._points) < 2:
            return frame
        base = frame.convert("RGBA")
        draw = ImageDraw.Draw(base)
        pts = [view.to_view(x, y) for x, y in self._points]
        draw.line(pts, fill=(0, 0, 0, 160), width=max(1, px(1)) + 2)
        draw.line(pts, fill=(255, 255, 255, 230), width=max(1, px(1)))
        return base.convert("RGB")

    def cancel(self) -> None:
        had = bool(self._points)
        self._points = []
        if had:
            self.view.invalidate_frame()


class WandTool(Tool):
    id, name, icon, key, cursor = "wand", "Magic Wand", "wand", "w", "cross"
    hint = "Click to select similar colours · Shift adds · Alt subtracts"

    def __init__(self, app) -> None:
        super().__init__(app)
        self.options = {"mode": "replace", "tolerance": 32, "contiguous": True, "merged": True}

    def build_options(self) -> list[Widget]:
        return [
            _option_slider(tr("Tolerance"), 0, 255, self.options, "tolerance", width=200),
            _option_toggle(tr("Contiguous"), self.options, "contiguous", 130),
            _option_toggle(tr("All layers"), self.options, "merged", 120),
            _mode_option(self.options),
        ]

    def on_press(self, e: PointerEvent) -> None:
        if not (0 <= e.x < self.doc.width and 0 <= e.y < self.doc.height):
            return
        self.doc.magic_wand(
            e.x,
            e.y,
            self.options["tolerance"],
            self.options["contiguous"],
            self.options["merged"],
            _selection_mode(e, self.options["mode"]),
        )
        self.app.after_edit()


# --------------------------------------------------------------- painting --
class BrushTool(Tool):
    id, name, icon, key, cursor = "brush", "Brush", "brush", "b", "none"
    hint = "Paint with the foreground colour · [ and ] change the size · Shift-click for straight lines"
    has_cursor_layer = True
    erase = False
    default_preset = "Hard Round"

    def __init__(self, app: Editor) -> None:
        super().__init__(app)
        self.preset: BrushPreset = app.brushes.get(self.default_preset)
        self.options = {"size": self.preset.size, "hardness": self.preset.hardness, "opacity": 100, "flow": self.preset.flow}
        self._stroke: Stroke | None = None
        self._base: Image.Image | None = None
        self._preview: Image.Image | None = None
        self._last_point: tuple[float, float] | None = None
        self._size_slider: Slider | None = None
        self._brush_button: Button | None = None

    def build_options(self) -> list[Widget]:
        self._size_slider = _option_slider(tr("Size"), 1, 500, self.options, "size", width=200)
        self._brush_button = Button(self.preset.name, self.open_brushes, icon="brush", small=True, tooltip=tr("Brush presets and settings"))
        return [
            self._brush_button,
            self._size_slider,
            _option_slider(tr("Hardness"), 0, 100, self.options, "hardness", width=170),
            _option_slider(tr("Opacity"), 1, 100, self.options, "opacity", width=170),
            _option_slider(tr("Flow"), 1, 100, self.options, "flow", width=150),
        ]

    def open_brushes(self) -> None:
        from . import sheets

        sheets.brush_sheet(self.app, self)

    def current_preset(self) -> BrushPreset:
        """The preset as the options bar has tuned it (size, hardness and flow live in the options)."""
        return replace(
            self.preset, size=int(self.options["size"]), hardness=int(self.options["hardness"]), flow=int(self.options["flow"])
        ).normalized()

    def set_preset(self, preset: BrushPreset) -> None:
        """Switch to *preset*: its size, hardness and flow become the options bar values."""
        self.preset = preset
        self.options.update(size=preset.size, hardness=preset.hardness, flow=preset.flow)
        if self._size_slider is not None:
            self._size_slider.set(preset.size)
        if self._brush_button is not None:
            self._brush_button.text = preset.name
            self._brush_button.invalidate()
        self.app.refresh_tool_options()
        self.view.schedule(0)

    def _color(self):
        return self.app.foreground

    def on_press(self, e: PointerEvent) -> None:
        if not self._editable():
            return
        target = self.doc.edit_image()  # the layer's pixels, or its mask while the mask is being edited
        self._stroke = Stroke(self.doc.size, preset=self.current_preset())
        self._base = target
        self._preview = target.copy()
        if e.shift and self._last_point is not None:
            self._stroke.add_point(*self._last_point)
        self._stroke.add_point(e.x, e.y)
        self._last_point = (e.x, e.y)
        self._update_preview()

    def on_drag(self, e: PointerEvent) -> None:
        if self._stroke is None:
            return
        self._stroke.add_point(e.x, e.y)
        self._last_point = (e.x, e.y)
        self._update_preview()

    def _update_preview(self) -> None:
        assert self._stroke is not None and self._base is not None and self._preview is not None
        box = self._stroke.take_dirty()
        if box is None:
            return
        x0, y0, x1, y1 = box
        cov = self._stroke.coverage[y0:y1, x0:x1]
        sel = self.doc.selection_mask()
        sel_crop = sel.crop(box) if sel is not None else None
        patch = self._patch(self._base.crop(box), cov, sel_crop, box)
        self._preview.paste(patch, (x0, y0))
        self.app.begin_preview(self.doc.active_layer.id, self._preview)

    def _patch(self, base: Image.Image, cov: np.ndarray, sel: Image.Image | None, box: tuple[int, int, int, int]) -> Image.Image:
        """Preview of the stroke on the *box* region of the layer."""
        opacity = self.options["opacity"] / 100
        if self.erase:
            return painting.erase(base, cov, opacity, sel)
        return painting.paint(base, cov, self._color(), opacity, sel)

    def _commit(self, stroke: Stroke) -> None:
        opacity = self.options["opacity"] / 100
        if self.erase:
            self.doc.erase(stroke.coverage, opacity)
        else:
            self.doc.paint(stroke.coverage, self._color(), opacity)

    def on_release(self, e: PointerEvent) -> None:
        if self._stroke is None:
            return
        stroke, self._stroke = self._stroke, None
        self._base = self._preview = None
        self.app.end_preview()
        self._commit(stroke)
        self.app.after_edit()

    def on_key(self, keysym: str, mods: frozenset[str] = frozenset()) -> bool:
        if keysym == "bracketleft":
            self.options["size"] = max(1, int(self.options["size"] / 1.25))
        elif keysym == "bracketright":
            self.options["size"] = min(500, max(self.options["size"] + 1, int(self.options["size"] * 1.25)))
        else:
            return False
        if self._size_slider is not None:
            self._size_slider.set(self.options["size"])
        self.view.schedule(0)
        return True

    def cursor_layer(self, view: Surface) -> Layer | None:
        if view.pointer is None:
            return None
        radius = max(2.0, self.options["size"] * view.zoom / 2)
        d = int(radius * 2) + px(4)
        img = Image.new("RGBA", (d, d), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        c = d / 2
        ry = max(1.0, radius * self.preset.roundness / 100)  # the outline mirrors the tip's roundness and angle
        for extra, color in ((1, (0, 0, 0, 150)), (0, (255, 255, 255, 230))):
            draw.ellipse((c - radius - extra, c - ry - extra, c + radius + extra, c + ry + extra), outline=color, width=max(1, px(1)))
        if self.preset.angle % 180 > 1e-6 and self.preset.roundness < 100:
            img = img.rotate(self.preset.angle, resample=Image.Resampling.BILINEAR)
        wx, wy = view.pointer
        return ("cursor", int(wx - c), int(wy - c), img)

    def cancel(self) -> None:
        self._stroke = self._base = self._preview = None
        self.app.end_preview()


class CloneTool(BrushTool):
    """Clone stamp: Alt-click to pick the source, then paint copies of it."""

    id, name, icon, key, cursor = "clone", "Clone Stamp", "stamp", "s", "none"
    hint = "Alt-click to set the source, then paint · [ and ] change the size"
    heal = False

    default_preset = "Soft Round"

    def __init__(self, app: Editor) -> None:
        super().__init__(app)
        self.options.update(size=40, hardness=60, aligned=True, merged=True)
        self.source_point: tuple[float, float] | None = None  # where the user Alt-clicked (document coordinates)
        self._source: Image.Image | None = None  # pixels sampled when the stroke began
        self._offset: tuple[int, int] | None = None  # destination → source offset of the current / aligned stroke
        self._stroke_start: tuple[float, float] | None = None

    def build_options(self) -> list[Widget]:
        widgets = super().build_options()
        widgets.append(_option_toggle(tr("Aligned"), self.options, "aligned", 110))
        widgets.append(_option_toggle(tr("All layers"), self.options, "merged", 120))
        return widgets

    def activate(self) -> None:
        self._offset = None

    def _sample(self) -> Image.Image:
        return self.doc.flatten() if self.options["merged"] else self.doc.edit_image()

    def on_press(self, e: PointerEvent) -> None:
        if e.alt:
            self.source_point = (e.x, e.y)
            self._offset = None
            self.view.invalidate_frame()
            return
        if self.source_point is None and not self.heal:
            self.app.notify(tr("Alt-click the picture to choose a source first"))
            return
        if not self._editable():
            return
        if self.source_point is not None and (self._offset is None or not self.options["aligned"]):
            self._offset = (int(round(self.source_point[0] - e.x)), int(round(self.source_point[1] - e.y)))
        self._source = self._sample() if self.source_point is not None else None
        self._stroke_start = (e.x, e.y)
        super().on_press(e)

    def _patch(self, base: Image.Image, cov: np.ndarray, sel: Image.Image | None, box: tuple[int, int, int, int]) -> Image.Image:
        assert self._base is not None and self._stroke is not None
        opacity = self.options["opacity"] / 100
        # healing looks at the surroundings: work on a padded region around the dirty box, not the whole layer
        x0, y0, x1, y1 = box
        pad = int(max(x1 - x0, y1 - y0) * 0.75) + 4
        rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
        rx1, ry1 = min(self._base.width, x1 + pad), min(self._base.height, y1 + pad)
        region = (rx0, ry0, rx1, ry1)
        region_cov = self._stroke.coverage[ry0:ry1, rx0:rx1].copy()
        region_cov[: y0 - ry0, :] = 0  # only the dirty box is painted in this update
        region_cov[y1 - ry0 :, :] = 0
        region_cov[:, : x0 - rx0] = 0
        region_cov[:, x1 - rx0 :] = 0
        full_sel = self.doc.selection_mask()
        region_sel = full_sel.crop(region) if full_sel is not None else None
        source = None
        if self._source is not None:
            dx, dy = self._offset or (0, 0)
            source = self._source.crop((rx0 + dx, ry0 + dy, rx1 + dx, ry1 + dy))
        result = painting.clone(self._base.crop(region), region_cov, source, (0, 0), opacity, region_sel, self.heal)
        return result.crop((x0 - rx0, y0 - ry0, x1 - rx0, y1 - ry0))

    def _commit(self, stroke: Stroke) -> None:
        self.doc.clone(stroke.coverage, self._source, self._offset or (0, 0), self.options["opacity"] / 100, self.heal)
        self._source = None

    def paint_overlay(self, frame: Image.Image, view: Surface) -> Image.Image:
        """Crosshair on the source: at the Alt-clicked point, or following the pointer while painting."""
        if self.source_point is None:
            return frame
        if self._stroke is not None and self._offset is not None and view.pointer is not None:
            px_, py_ = view.to_doc(*view.pointer)
            sx, sy = px_ + self._offset[0], py_ + self._offset[1]
        else:
            sx, sy = self.source_point
        vx, vy = view.to_view(sx, sy)
        base = frame.convert("RGBA")
        draw = ImageDraw.Draw(base)
        r = px(7)
        w = max(1, px(1))
        for color, extra in (((0, 0, 0, 160), 1), ((255, 255, 255, 230), 0)):
            draw.ellipse((vx - r - extra, vy - r - extra, vx + r + extra, vy + r + extra), outline=color, width=w + extra)
            draw.line((vx - r * 1.8, vy, vx + r * 1.8, vy), fill=color, width=w + extra)
            draw.line((vx, vy - r * 1.8, vx, vy + r * 1.8), fill=color, width=w + extra)
        return base.convert("RGB")

    def cancel(self) -> None:
        super().cancel()
        self._source = None


class HealTool(CloneTool):
    """Healing brush: like the clone stamp but keeps the destination's colour and lighting; without a source it spot-heals."""

    id, name, icon, key, cursor = "heal", "Healing Brush", "heal", "j", "none"
    hint = "Paint over blemishes to heal them · Alt-click to heal with texture from another spot"
    heal = True


class EraserTool(BrushTool):
    id, name, icon, key = "eraser", "Eraser", "eraser", "e"
    hint = "Erase to transparency · [ and ] change the size"
    erase = True


class FillTool(Tool):
    id, name, icon, key, cursor = "fill", "Paint Bucket", "fill", "g", "cross"
    hint = "Click to fill similar colours with the foreground colour (Alt: background colour)"

    def __init__(self, app) -> None:
        super().__init__(app)
        self.options = {"tolerance": 32, "contiguous": True, "merged": False, "whole": False}

    def build_options(self) -> list[Widget]:
        return [
            _option_slider(tr("Tolerance"), 0, 255, self.options, "tolerance", width=200),
            _option_toggle(tr("Contiguous"), self.options, "contiguous", 130),
            _option_toggle(tr("All layers"), self.options, "merged", 120),
            _option_toggle(tr("Fill whole selection"), self.options, "whole", 180),
        ]

    def on_press(self, e: PointerEvent) -> None:
        if not self._editable():
            return
        color = self.app.background if e.alt else self.app.foreground
        if self.options["whole"]:
            self.doc.fill(color)
        else:
            if not (0 <= e.x < self.doc.width and 0 <= e.y < self.doc.height):
                return
            self.doc.bucket_fill(e.x, e.y, color, self.options["tolerance"], self.options["contiguous"], self.options["merged"])
        self.app.after_edit()


class GradientTool(Tool):
    id, name, icon, key, cursor = "gradient", "Gradient", "gradient", "G", "cross"
    hint = "Drag to draw a foreground → background gradient · Ctrl constrains the angle"

    def __init__(self, app) -> None:
        super().__init__(app)
        self.options = {"kind": "linear", "reverse": False}
        self._start: tuple[float, float] | None = None
        self._end: tuple[float, float] | None = None

    def build_options(self) -> list[Widget]:
        kinds = list(GRADIENT_TYPES)
        seg = Segmented(
            [tr(k.title()) for k in kinds],
            kinds.index(self.options["kind"]),
            lambda i: self.options.__setitem__("kind", kinds[i]),
            width=260,
        )
        return [seg, _option_toggle(tr("Reverse"), self.options, "reverse", 120)]

    def on_press(self, e: PointerEvent) -> None:
        if not self._editable():
            return
        self._start = (e.x, e.y)
        self._end = None

    def on_drag(self, e: PointerEvent) -> None:
        if self._start is None:
            return
        x, y = e.x, e.y
        if e.ctrl:
            dx, dy = x - self._start[0], y - self._start[1]
            angle = round(math.atan2(dy, dx) / (math.pi / 4)) * (math.pi / 4)
            length = math.hypot(dx, dy)
            x, y = self._start[0] + math.cos(angle) * length, self._start[1] + math.sin(angle) * length
        self._end = (x, y)
        self.view.invalidate_frame()

    def on_release(self, e: PointerEvent) -> None:
        start, end = self._start, self._end
        self._start = self._end = None
        if start is None or end is None or math.hypot(end[0] - start[0], end[1] - start[1]) < 1:
            self.view.invalidate_frame()
            return
        a, b = self.app.foreground, self.app.background
        if self.options["reverse"]:
            a, b = b, a
        self.doc.gradient(start, end, a, b, self.options["kind"])
        self.app.after_edit()

    def paint_overlay(self, frame: Image.Image, view: Surface) -> Image.Image:
        if self._start is None or self._end is None:
            return frame
        base = frame.convert("RGBA")
        draw = ImageDraw.Draw(base)
        x0, y0 = view.to_view(*self._start)
        x1, y1 = view.to_view(*self._end)
        draw.line((x0, y0, x1, y1), fill=(0, 0, 0, 160), width=max(1, px(1)) + 2)
        draw.line((x0, y0, x1, y1), fill=(255, 255, 255, 230), width=max(1, px(1)))
        r = px(5)
        for x, y in ((x0, y0), (x1, y1)):
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 0, 0, 200), outline=(255, 255, 255, 230), width=max(1, px(1)))
        return base.convert("RGB")

    def cancel(self) -> None:
        had = self._end is not None
        self._start = self._end = None
        if had:
            self.view.invalidate_frame()


# ------------------------------------------------------------------ text --
class TextTool(Tool):
    id, name, icon, key, cursor = "text", "Text", "text", "t", "text"
    hint = "Click where the text should start · double-click a text layer to edit it"

    def on_press(self, e: PointerEvent) -> None:
        if self.app.doc is None:
            return
        self.app.open_text_sheet(position=(e.x, e.y))


# ---------------------------------------------------------------- shapes --
class ShapeTool(Tool):
    id, name, icon, key, cursor = "shape", "Shape", "shape", "u", "cross"
    hint = "Drag to draw · fill uses the foreground, stroke the background colour · Ctrl constrains"

    def __init__(self, app) -> None:
        super().__init__(app)
        self.options = {"kind": "rectangle", "fill": True, "stroke": 0}
        self._start: tuple[float, float] | None = None
        self._box: tuple[float, float, float, float] | None = None

    def build_options(self) -> list[Widget]:
        kinds = list(SHAPE_TYPES)
        seg = Segmented(
            [tr(k.title()) for k in kinds],
            kinds.index(self.options["kind"]),
            lambda i: self.options.__setitem__("kind", kinds[i]),
            width=260,
        )
        return [
            seg,
            _option_toggle(tr("Fill"), self.options, "fill", 100),
            _option_slider(tr("Stroke"), 0, 100, self.options, "stroke", width=190),
        ]

    def on_press(self, e: PointerEvent) -> None:
        if not self._editable():
            return
        self._start = (e.x, e.y)
        self._box = None

    def on_drag(self, e: PointerEvent) -> None:
        if self._start is None:
            return
        x0, y0 = self._start
        x1, y1 = e.x, e.y
        if e.ctrl:
            side = max(abs(x1 - x0), abs(y1 - y0))
            x1 = x0 + math.copysign(side, x1 - x0 or 1)
            y1 = y0 + math.copysign(side, y1 - y0 or 1)
        self._box = (x0, y0, x1, y1)
        self.view.invalidate_frame()

    def on_release(self, e: PointerEvent) -> None:
        box = self._box
        self._start = self._box = None
        if box is None or (abs(box[2] - box[0]) < 1 and abs(box[3] - box[1]) < 1):
            self.view.invalidate_frame()
            return
        kind = self.options["kind"]
        fill_color = self.app.foreground if (self.options["fill"] and kind != "line") else None
        width = self.options["stroke"]
        outline = self.app.background if width > 0 else None
        if kind == "line":
            outline, width = self.app.foreground, max(1, width)
        self.doc.draw_shape(kind, box, fill_color, outline, width)
        self.app.after_edit()

    def paint_overlay(self, frame: Image.Image, view: Surface) -> Image.Image:
        if self._box is None:
            return frame
        base = frame.convert("RGBA")
        draw = ImageDraw.Draw(base)
        x0, y0 = view.to_view(self._box[0], self._box[1])
        x1, y1 = view.to_view(self._box[2], self._box[3])
        kind = self.options["kind"]
        if kind == "line":
            draw.line((x0, y0, x1, y1), fill=(255, 255, 255, 230), width=max(1, px(1)))
        elif kind == "rectangle":
            _dashed_rect(draw, (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
        else:
            bx = (int(min(x0, x1)), int(min(y0, y1)), int(max(x0, x1)) + 1, int(max(y0, y1)) + 1)
            draw.ellipse(bx, outline=(255, 255, 255, 230), width=max(1, px(1)))
        return base.convert("RGB")

    def cancel(self) -> None:
        had = self._box is not None
        self._start = self._box = None
        if had:
            self.view.invalidate_frame()


# -------------------------------------------------------------------- 3D --
class ThreeDTool(Tool):
    """Rotate (drag), move (Shift-drag) and scale (Alt-drag) the 3D object of the active layer."""

    id, name, icon, key, cursor = "3d", "3D", "cube", "k", "move"
    hint = "Drag to rotate the 3D object · Shift-drag moves · Alt-drag scales · double-click for all settings"
    edits_smart_layers = True
    options_follow_layer = True
    PREVIEW_SCALE = 0.5

    def __init__(self, app: Editor) -> None:
        super().__init__(app)
        self.options = {"wireframe": False}
        self._start: tuple[float, float] | None = None
        self._mode = "rotate"
        self._origin: Scene | None = None
        self._pending: Scene | None = None
        self._generation = 0

    def _scene(self) -> Scene | None:
        return self.doc.scene_of() if self.app.doc is not None else None

    def build_options(self) -> list[Widget]:
        scene = self._scene()
        widgets: list[Widget] = [Button(tr("3D Settings…"), self.open_settings, icon="cube", small=True)]
        if scene is not None:
            shapes = [sh for sh in SHAPES if sh != "extrusion"]
            current = shapes.index(scene.shape) if scene.shape in shapes else 0
            widgets.append(Segmented([tr(sh.title()) for sh in shapes], current, self._pick_shape, width=420))
            self.options["wireframe"] = scene.wireframe
            widgets.append(_option_toggle(tr("Wireframe"), self.options, "wireframe", 120))
            toggle = widgets[-1]
            assert isinstance(toggle, Toggle)
            toggle.command = self._toggle_wireframe
        return widgets

    def _pick_shape(self, index: int) -> None:
        shapes = [sh for sh in SHAPES if sh != "extrusion"]
        if self._scene() is None:
            return
        self.app.doc_op(lambda d: d.update_3d_layer(shape=shapes[index], label="3D Shape"))

    def _toggle_wireframe(self, value: bool) -> None:
        self.options["wireframe"] = value
        if self._scene() is not None:
            self.app.doc_op(lambda d: d.update_3d_layer(wireframe=value, label="3D Wireframe"))

    def open_settings(self) -> None:
        from . import sheets

        if self.app.doc is None:
            return
        if self._scene() is None:
            self.app.notify(tr("Select a 3D layer, or add one with Layer ▸ New 3D Object…"))
            return
        sheets.three_d_sheet(self.app, self.doc.active_index)

    # interaction -------------------------------------------------------------
    def on_press(self, e: PointerEvent) -> None:
        scene = self._scene()
        if scene is None:
            if self.app.doc is not None:
                self.app.notify(tr("Select a 3D layer, or add one with Layer ▸ New 3D Object…"))
            return
        if not self._editable():
            return
        self._start = (e.x, e.y)
        self._origin = scene
        self._mode = "move" if e.shift else "scale" if e.alt else "rotate"

    def on_drag(self, e: PointerEvent) -> None:
        if self._start is None or self._origin is None:
            return
        dx, dy = e.x - self._start[0], e.y - self._start[1]
        o = self._origin
        if self._mode == "move":
            scene = replace(o, position=(o.position[0] + dx, o.position[1] + dy))
        elif self._mode == "scale":
            scene = replace(o, size=max(4.0, o.size * (1 + dx / 200.0)))
        else:
            rx, ry, rz = o.rotation
            scene = replace(o, rotation=(rx + dy * 0.5, ry + dx * 0.5, rz))
        self._pending = scene.normalized()
        self._preview(self._pending)

    def _preview(self, scene: Scene) -> None:
        """Render a quick half-resolution preview on the worker; only the newest result is shown."""
        doc = self.doc
        layer_id = doc.active_layer.id
        self._generation += 1
        generation = self._generation
        size = doc.size

        def job() -> Image.Image:
            small = render_scene(scene, size, supersample=1, scale=self.PREVIEW_SCALE)
            return small.resize(size, Image.Resampling.BILINEAR)

        def show(image: Image.Image) -> None:
            if generation == self._generation and self.app.doc is doc and doc.active_layer.id == layer_id:
                self.app.begin_preview(layer_id, image)

        self.app.worker.submit("3d-preview", job, show)

    def on_release(self, e: PointerEvent) -> None:
        pending, self._pending = self._pending, None
        self._start = self._origin = None
        self._generation += 1
        self.app.worker.cancel("3d-preview")
        self.app.end_preview()
        if pending is None:
            return
        self.app.doc_op(
            lambda d: d.update_3d_layer(scene=pending, label="Rotate 3D Object" if self._mode == "rotate" else "Move 3D Object")
        )

    def on_double(self, e: PointerEvent) -> None:
        self.open_settings()

    def on_key(self, keysym: str, mods: frozenset[str] = frozenset()) -> bool:
        scene = self._scene()
        if scene is None:
            return False
        deltas = {"Left": (0, -15), "Right": (0, 15), "Up": (-15, 0), "Down": (15, 0)}
        if keysym not in deltas:
            return False
        drx, dry = deltas[keysym]
        rx, ry, rz = scene.rotation
        self.app.doc_op(lambda d: d.update_3d_layer(rotation=(rx + drx, ry + dry, rz), label="Rotate 3D Object"))
        return True

    def paint_overlay(self, frame: Image.Image, view: Surface) -> Image.Image:
        scene = self._pending or self._scene()
        if scene is None or self.app.doc is None:
            return frame
        x0, y0, x1, y1 = bounding_box(scene, self.doc.size)
        vx0, vy0 = view.to_view(x0, y0)
        vx1, vy1 = view.to_view(x1, y1)
        base = frame.convert("RGBA")
        _dashed_rect(ImageDraw.Draw(base), (vx0, vy0, vx1, vy1))
        return base.convert("RGB")

    def cancel(self) -> None:
        self._start = self._origin = self._pending = None
        self._generation += 1
        self.app.end_preview()


# ------------------------------------------------------------ eyedropper --
class EyedropperTool(Tool):
    id, name, icon, key, cursor = "eyedropper", "Eyedropper", "eyedropper", "i", "cross"
    hint = "Click to pick the foreground colour (Alt: background colour)"

    def __init__(self, app) -> None:
        super().__init__(app)
        self.options = {"merged": True}

    def build_options(self) -> list[Widget]:
        return [
            _option_toggle(tr("All layers"), self.options, "merged", 120),
            Spacer(width=8),
            Label(tr("Click the picture to sample a colour"), style="small"),
        ]

    def _pick(self, e: PointerEvent) -> None:
        color = self.doc.pick_color(e.x, e.y, merged=self.options["merged"])
        if color is None or color[3] == 0:
            return
        self.app.set_color("background" if e.alt else "foreground", (*color[:3], 255))

    on_press = _pick
    on_drag = _pick


TOOL_CLASSES: tuple[type[Tool], ...] = (
    MoveTool,
    TransformTool,
    MarqueeTool,
    LassoTool,
    WandTool,
    CropTool,
    BrushTool,
    EraserTool,
    CloneTool,
    HealTool,
    FillTool,
    GradientTool,
    TextTool,
    ShapeTool,
    ThreeDTool,
    EyedropperTool,
    ZoomTool,
    HandTool,
)
