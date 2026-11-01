"""Glass sheets and popover menus opened by the editor (resize, straighten, text, filters, …).

Every function here builds a :class:`Panel`, wires its widgets to the editor
and opens it via ``editor.open_sheet``.  Keeping them out of ``editor.py``
keeps that module focused on state and commands.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

from PIL import Image

from .. import APP_NAME, __version__
from ..core import brushes as brushes_mod
from ..core import curves as curves_mod
from ..core import effects as effects_mod
from ..core import three_d
from ..core.brushes import BrushPreset
from ..core.document import inscribed_box
from ..core.filters import CATEGORIES, FILTERS, FilterSpec, filters_in
from ..core.painting import system_fonts
from ..core.presets import PRESETS
from ..core.three_d import Scene
from .i18n import tr
from .theme import PALETTE, px
from .ui import (
    AnchorPicker,
    BrushRow,
    Button,
    ColorDot,
    Column,
    CurveEditor,
    Label,
    LevelsEditor,
    ListItem,
    NumberField,
    Panel,
    Row,
    Segmented,
    Slider,
    Spacer,
    TextField,
    Toggle,
    Widget,
)

ABOUT_TEXT = (
    "A layer-based picture editor built with Python, Pillow and NumPy.\n"
    "Non-destructive adjustments, presets, selections, brushes, text,\n"
    "filters, crop and straighten, snapshots and project files."
)


def _buttons(editor, ok_label: str, ok: Callable[[], None], cancel_label: str | None = None) -> Row:
    cancel = Button(cancel_label or tr("Cancel"), editor.dismiss_modal)
    return Row([Spacer(), cancel, Button(ok_label, ok, primary=True)], gap=8, height=34)


def _sheet(editor, name: str, items: list[Widget], width: int = 360, default: Callable[[], None] | None = None, **kw) -> Panel:
    panel = Panel(Column(items), editor.sheet_place(width), padding=22, fit_height=True, modal=True, name=name, **kw)
    if default is not None:
        panel.default_action = default  # type: ignore[attr-defined]
    editor.open_sheet(panel)
    return panel


def _popover(editor, name: str, items: list[Widget], place, width: int = 240, scrollable: bool = False) -> Panel:
    panel = Panel(
        Column(items),
        place,
        padding=8,
        radius=16,
        fit_height=not scrollable,
        modal=True,
        dismiss_on_outside=True,
        name=name,
        scrollable=scrollable,
    )
    editor.open_sheet(panel)
    return panel


# ------------------------------------------------------------------ image --
def resize_sheet(editor) -> None:
    doc = editor.doc
    if doc is None:
        return
    w0, h0 = doc.size
    state = {"lock": True, "syncing": False}

    def width_changed(value: int) -> None:
        if state["lock"] and not state["syncing"]:
            state["syncing"] = True
            height_field.set_value(max(1, round(value * h0 / w0)))
            state["syncing"] = False

    def height_changed(value: int) -> None:
        if state["lock"] and not state["syncing"]:
            state["syncing"] = True
            width_field.set_value(max(1, round(value * w0 / h0)))
            state["syncing"] = False

    width_field = NumberField(tr("Width"), w0, on_change=width_changed)
    height_field = NumberField(tr("Height"), h0, on_change=height_changed)
    lock = Toggle(tr("Keep proportions"), True, lambda v: state.__setitem__("lock", v))

    def apply() -> None:
        w, h = width_field.value, height_field.value
        editor.dismiss_modal()
        editor.doc_op(lambda d: d.resize(w, h), fit=True)

    _sheet(
        editor,
        "resize",
        [
            Label(tr("Resize picture"), style="title"),
            Label(tr("Currently {w} × {h} px").format(w=w0, h=h0), style="muted"),
            Spacer(8),
            width_field,
            height_field,
            Spacer(4),
            lock,
            Spacer(12),
            _buttons(editor, tr("Resize"), apply),
        ],
        340,
        apply,
    )
    editor.view.set_focus(width_field)


def canvas_sheet(editor) -> None:
    doc = editor.doc
    if doc is None:
        return
    w0, h0 = doc.size
    width_field = NumberField(tr("Width"), w0)
    height_field = NumberField(tr("Height"), h0)
    anchor = AnchorPicker(tr("Anchor"), "center")
    fill = Segmented([tr("Transparent"), tr("White"), tr("Black")], 0)

    def apply() -> None:
        w, h = width_field.value, height_field.value
        color = [None, (255, 255, 255, 255), (0, 0, 0, 255)][fill.index]
        editor.dismiss_modal()
        editor.doc_op(lambda d: d.canvas_size(w, h, anchor.value, color), fit=True)

    _sheet(
        editor,
        "canvas",
        [
            Label(tr("Canvas size"), style="title"),
            Label(tr("Currently {w} × {h} px · the picture keeps its size").format(w=w0, h=h0), style="muted"),
            Spacer(8),
            width_field,
            height_field,
            Spacer(4),
            anchor,
            Spacer(8),
            Label(tr("New area"), style="caption", uppercase=True),
            fill,
            Spacer(12),
            _buttons(editor, tr("Apply"), apply),
        ],
        380,
        apply,
    )
    editor.view.set_focus(width_field)


def straighten_sheet(editor) -> None:
    doc = editor.doc
    if doc is None:
        return
    layer = doc.active_layer
    source = layer.image
    limit = 1_600_000  # preview on a reduced copy for big pictures so the slider stays fluid
    if source.width * source.height > limit:
        factor = (limit / (source.width * source.height)) ** 0.5
        small = source.resize((max(1, int(source.width * factor)), max(1, int(source.height * factor))), Image.Resampling.BILINEAR)
    else:
        small = source
    state = {"angle": 0.0, "crop": True}

    def preview() -> None:
        angle = state["angle"]
        if abs(angle) < 1e-6:
            editor.preview_overrides = {}
        else:
            rotated = small.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(0, 0, 0, 0))
            if rotated.size != source.size:
                rotated = rotated.resize(source.size, Image.Resampling.BILINEAR)
            editor.preview_overrides = {layer.id: rotated}
        editor.straighten_preview = (angle, state["crop"])
        editor.view.invalidate_frame()

    def on_angle(value: float) -> None:
        state["angle"] = round(value, 1)
        preview()

    def on_crop(value: bool) -> None:
        state["crop"] = value
        preview()

    def apply() -> None:
        angle, crop = state["angle"], state["crop"]
        editor.dismiss_modal()
        if abs(angle) >= 0.05:
            editor.doc_op(lambda d: d.straighten(angle, crop=crop), fit=True)

    slider = Slider(tr("Angle"), -45, 45, 0, 0, on_change=on_angle, integer=False, fmt="{:+.1f}°", tooltip=tr("Drag to level the horizon"))
    panel = Panel(
        Column(
            [
                Label(tr("Straighten"), style="title"),
                Label(tr("Use the grid to level the horizon."), style="muted"),
                Spacer(8),
                slider,
                Toggle(tr("Crop away the empty corners"), True, on_crop),
                Spacer(12),
                _buttons(editor, tr("Apply"), apply),
            ]
        ),
        editor.bottom_sheet_place(380),
        padding=22,
        fit_height=True,
        modal=True,
        name="straighten",
    )
    panel.default_action = apply  # type: ignore[attr-defined]
    editor.open_sheet(panel)
    preview()


def paint_straighten_overlay(editor, frame: Image.Image, view) -> Image.Image:
    """Grid plus the crop that will remain, drawn while the straighten sheet is open."""
    from PIL import ImageDraw

    angle, crop = editor.straighten_preview
    base = frame.convert("RGBA")
    draw = ImageDraw.Draw(base)
    ix0, iy0, ix1, iy1 = view.image_rect()
    if crop and abs(angle) > 1e-6:
        bx0, by0, bx1, by1 = inscribed_box(editor.doc.width, editor.doc.height, angle)
        cx0, cy0 = view.to_view(bx0, by0)
        cx1, cy1 = view.to_view(bx1, by1)
        box = (int(cx0), int(cy0), max(int(cx1) - 1, int(cx0)), max(int(cy1) - 1, int(cy0)))
        dim = Image.new("RGBA", base.size, (0, 0, 0, 0))
        ddraw = ImageDraw.Draw(dim)
        ddraw.rectangle((ix0, iy0, ix1, iy1), fill=PALETTE.crop_dim)
        ddraw.rectangle(box, fill=(0, 0, 0, 0))
        base.alpha_composite(dim)
        draw.rectangle(box, outline=(255, 255, 255, 220), width=max(1, px(1)))
    steps = 8
    for i in range(1, steps):
        gx = ix0 + (ix1 - ix0) * i / steps
        gy = iy0 + (iy1 - iy0) * i / steps
        draw.line((gx, iy0, gx, iy1), fill=(255, 255, 255, 60), width=max(1, px(1)))
        draw.line((ix0, gy, ix1, gy), fill=(255, 255, 255, 60), width=max(1, px(1)))
    return base.convert("RGB")


# ------------------------------------------------------------------ export --
def export_sheet(editor, path: str, fmt: str, has_alpha: bool) -> None:
    quality = Slider(tr("Quality"), 1, 100, int(editor.settings.get("export_quality", 92)), 92, fmt="{:d}")
    is_white = tuple(editor.settings.get("export_background", (255, 255, 255))) == (255, 255, 255)
    background = Segmented([tr("White"), tr("Black")], 0 if is_white else 1)
    rows: list[Widget] = [Label(tr("Export as {fmt}").format(fmt=fmt), style="title"), Label(Path(path).name, style="muted"), Spacer(8)]
    if fmt in ("JPEG", "WEBP"):
        rows.append(quality)
    if not has_alpha:
        rows += [Label(tr("Transparent areas become"), style="caption", uppercase=True), background, Spacer(6)]
    keep_meta = Toggle(tr("Keep colour profile and EXIF"), bool(editor.settings.get("export_metadata", True)))
    if fmt in ("JPEG", "PNG", "WEBP", "TIFF"):
        rows.append(keep_meta)
        described = editor.doc.metadata.describe() if editor.doc is not None else ""
        if described:
            rows.append(Label(described, style="small"))

    def save() -> None:
        bg = (255, 255, 255) if background.index == 0 else (0, 0, 0)
        editor.settings.set("export_quality", int(quality.value))
        editor.settings.set("export_background", list(bg))
        editor.settings.set("export_metadata", bool(keep_meta.value))
        editor.dismiss_modal()
        editor.write_image(path, int(quality.value), bg, metadata=bool(keep_meta.value))

    rows += [Spacer(8), _buttons(editor, tr("Export"), save)]
    _sheet(editor, "export", rows, 360, save)


# ------------------------------------------------------------- presets etc --
def save_preset_sheet(editor) -> None:
    doc = editor.doc
    if doc is None:
        return
    field = TextField("", placeholder=tr("Preset name"), field_width=0, max_length=40)
    error = Label("", style="small")
    error.color = PALETTE.danger

    def save() -> None:
        name = field.value.strip()
        if not name:
            error.set_text(tr("Please enter a name"))
            return
        if name in PRESETS:
            error.set_text(tr("That name belongs to a built-in preset"))
            return
        editor.presets.save(name, doc.active_layer.adjustments)
        editor.dismiss_modal()
        editor.refresh_all()
        editor.notify(tr("Saved preset “{name}”").format(name=name), error=False, timeout=2500)

    _sheet(
        editor,
        "save-preset",
        [
            Label(tr("Save preset"), style="title"),
            Label(tr("The current slider values will be stored under this name."), style="muted"),
            Spacer(8),
            field,
            error,
            Spacer(8),
            _buttons(editor, tr("Save"), save),
        ],
        360,
        save,
    )
    editor.view.set_focus(field)


def snapshot_sheet(editor) -> None:
    doc = editor.doc
    if doc is None:
        return
    field = TextField("", tr("Snapshot {n}").format(n=len(doc.snapshots) + 1), field_width=0, max_length=40)

    def take() -> None:
        name = field.value.strip()
        editor.dismiss_modal()
        doc.take_snapshot(name)
        editor.refresh_all()
        editor.notify(tr("Snapshot “{name}” saved").format(name=name or "Snapshot"), error=False, timeout=2500)

    _sheet(
        editor,
        "snapshot",
        [
            Label(tr("Take snapshot"), style="title"),
            Label(tr("Bookmarks the current state so you can come back to it later."), style="muted"),
            Spacer(8),
            field,
            Spacer(8),
            _buttons(editor, tr("Save"), take),
        ],
        360,
        take,
    )
    editor.view.set_focus(field)


def rename_layer_sheet(editor, index: int) -> None:
    doc = editor.doc
    if doc is None:
        return
    field = TextField("", doc.layers[index].name, field_width=0, max_length=60)

    def rename() -> None:
        name = field.value.strip()
        editor.dismiss_modal()
        if name:
            editor.doc_op(lambda d: d.set_layer_props(index, name=name))

    _sheet(
        editor,
        "rename",
        [Label(tr("Rename layer"), style="title"), Spacer(8), field, Spacer(8), _buttons(editor, tr("Rename"), rename)],
        340,
        rename,
    )
    editor.view.set_focus(field)


def number_sheet(editor, title: str, label: str, initial: int, lo: int, hi: int, on_ok: Callable[[int], None]) -> None:
    field = NumberField(label, initial, lo, hi)

    def ok() -> None:
        value = field.value
        editor.dismiss_modal()
        on_ok(value)

    _sheet(editor, "number", [Label(title, style="title"), Spacer(8), field, Spacer(8), _buttons(editor, tr("Apply"), ok)], 340, ok)
    editor.view.set_focus(field)


# -------------------------------------------------------------------- text --
def text_sheet(editor, position: tuple[float, float] | None = None, layer_index: int | None = None) -> None:
    doc = editor.doc
    if doc is None:
        return
    meta = dict(doc.layers[layer_index].meta) if layer_index is not None else {}
    state: dict[str, Any] = {
        "font": meta.get("font_name"),
        "color": tuple(meta.get("color") or editor.foreground),
        "stroke_color": tuple(meta.get("stroke_color") or (0, 0, 0, 255)),
    }
    text_field = TextField("", meta.get("text", ""), placeholder=tr("Your text"), field_width=0, max_length=200)
    size_field = NumberField(tr("Size"), int(meta.get("font_size", 48)), 4, 1000)
    stroke_field = NumberField(tr("Outline"), int(meta.get("stroke_width", 0)), 0, 50)
    font_button = Button(str(state["font"] or tr("Default font")), None, width=200)
    color_dot, stroke_dot = ColorDot(state["color"]), ColorDot(state["stroke_color"])

    def pick_font() -> None:
        names = [None] + sorted(system_fonts())

        def choose(name: str | None) -> None:
            state["font"] = name
            font_button.text = name or tr("Default font")
            font_button.invalidate()
            editor.close_popover()

        items: list[Widget] = [ListItem(n or tr("Default font"), partial(choose, n), checked=n == state["font"]) for n in names]
        editor.open_popover("fonts", items, editor.sheet_place(320), scrollable=True)

    def pick(which: str, dot: ColorDot) -> None:
        color = editor.shell.ask_color(state[which])
        if color is not None:
            state[which] = color
            dot.set_color(color)

    font_button.command = pick_font

    def apply() -> None:
        text = text_field.value
        editor.dismiss_modal()
        if not text.strip():
            return
        kwargs = {
            "font_name": state["font"],
            "font_size": size_field.value,
            "color": state["color"],
            "stroke_width": stroke_field.value,
            "stroke_color": state["stroke_color"] if stroke_field.value > 0 else None,
        }
        if layer_index is not None:
            editor.doc_op(lambda d: d.update_text_layer(layer_index, text=text, **kwargs))
        else:
            pos = position or (doc.width / 2, doc.height / 2)
            editor.doc_op(lambda d: d.add_text_layer(text, pos, **kwargs))

    _sheet(
        editor,
        "text",
        [
            Label(tr("Edit text") if layer_index is not None else tr("Add text"), style="title"),
            Spacer(8),
            text_field,
            Spacer(4),
            Row([Label(tr("Font")), font_button], gap=8, height=34),
            size_field,
            Row(
                [Label(tr("Colour")), Spacer(), color_dot, Button(tr("Change…"), lambda: pick("color", color_dot), small=True)],
                gap=8,
                height=30,
            ),
            stroke_field,
            Row(
                [
                    Label(tr("Outline colour")),
                    Spacer(),
                    stroke_dot,
                    Button(tr("Change…"), lambda: pick("stroke_color", stroke_dot), small=True),
                ],
                gap=8,
                height=30,
            ),
            Spacer(10),
            _buttons(editor, tr("Apply"), apply),
        ],
        400,
        apply,
    )
    editor.view.set_focus(text_field)


# ---------------------------------------------------------------- filters --
def filter_picker(editor, category_index: int = 0) -> None:
    if editor.doc is None:
        return
    state = {"category": category_index}
    listing = Column()

    def fill() -> None:
        listing.clear()
        for spec in filters_in(CATEGORIES[state["category"]]):
            subtitle = ", ".join(p.label for p in spec.params) or (spec.description or tr("No options"))
            listing.add(ListItem(spec.name, partial(_run_filter_from_picker, editor, spec), subtitle=subtitle))

    def pick(index: int) -> None:
        state["category"] = index
        fill()
        editor.view.layout_panels()

    fill()
    panel = Panel(
        Column([Label(tr("Filters"), style="title"), Spacer(6), Segmented(list(CATEGORIES), category_index, pick), Spacer(8), listing]),
        editor.sheet_place(450),
        padding=18,
        fit_height=True,
        modal=True,
        dismiss_on_outside=True,
        name="filters",
    )
    editor.open_sheet(panel)


def _run_filter_from_picker(editor, spec: FilterSpec) -> None:
    editor.menu_action(lambda: editor.run_filter(spec))


def filter_sheet(editor, spec: FilterSpec) -> None:
    doc = editor.doc
    if doc is None:
        return
    values = spec.defaults()
    controls: list[Widget] = []
    preview = LivePreview(editor, f"filter:{spec.name}", lambda: lambda im: spec.apply(im, **values))
    schedule_preview = preview.schedule

    for param in spec.params:
        if param.kind == "bool":

            def on_toggle(value: bool, p=param) -> None:
                values[p.name] = value
                schedule_preview()

            controls.append(Toggle(param.label, bool(param.default), on_toggle))
        elif param.kind == "choice":
            controls.append(Label(param.label, style="caption", uppercase=True))
            index = param.choices.index(param.default) if param.default in param.choices else 0

            def on_choice(i: int, p=param) -> None:
                values[p.name] = p.choices[i]
                schedule_preview()

            controls.append(Segmented(list(param.choices), index, on_choice))
            controls.append(Spacer(4))
        else:
            integer = param.kind == "int"
            default = float(param.default)

            def on_slide(v: float, p=param) -> None:
                values[p.name] = p.coerce(v)
                schedule_preview()

            controls.append(
                Slider(
                    param.label,
                    param.min,
                    param.max,
                    default,
                    default,
                    on_change=on_slide,
                    integer=integer,
                    fmt="{:d}" if integer else "{:.1f}",
                )
            )

    def apply() -> None:
        preview.stop()
        chosen = dict(values)
        editor.dismiss_modal()
        editor.doc_op(lambda d: d.apply_filter(spec.name, **chosen))

    panel = Panel(
        Column(
            [
                Label(spec.name, style="title"),
                Label(spec.description or tr("{category} filter").format(category=spec.category), style="muted"),
                Spacer(8),
                *controls,
                Spacer(10),
                _buttons(editor, tr("Apply"), apply),
            ]
        ),
        editor.bottom_sheet_place(380),
        padding=22,
        fit_height=True,
        modal=True,
        name=f"filter:{spec.name}",
    )
    panel.default_action = apply  # type: ignore[attr-defined]
    panel.on_dismiss = preview.stop  # type: ignore[attr-defined]
    editor.open_sheet(panel)
    preview.now()


# ----------------------------------------------------------------- tones --
class LivePreview:
    """Debounced live preview while a sheet is open.

    On pixel layers the operation from *make_fn* is previewed on the layer's
    pixels; on adjustment layers the settings from *make_meta* are written to
    the layer's ``meta`` (uncommitted) so the whole composite updates.
    """

    def __init__(
        self,
        editor,
        sheet_name: str,
        make_fn: Callable[[], Callable[[Image.Image], Image.Image]] | None,
        make_meta: Callable[[], dict[str, Any]] | None = None,
        meta_only: bool = False,
    ) -> None:
        self.editor, self.sheet_name, self.make_fn, self.make_meta = editor, sheet_name, make_fn, make_meta
        self.doc = editor.doc
        layer = editor.doc.active_layer
        self.layer_id = layer.id
        self.adjustment = make_meta is not None and (layer.kind == "adjustment" or meta_only)
        self.original_meta = dict(layer.meta)
        self.job: object | None = None
        self.generation = 0  # only the newest background result is shown

    def _open(self) -> bool:
        editor = self.editor
        return editor.doc is self.doc and editor.sheet is not None and editor.sheet.name == self.sheet_name

    def now(self) -> None:
        self.job = None
        editor = self.editor
        if not self._open():
            return
        if self.adjustment:
            assert self.make_meta is not None
            index = self.doc.layer_index(self.layer_id)
            if index is None:
                return
            self.doc.set_layer_meta(index, commit=False, **self.make_meta())
            editor.view.invalidate_frame()
            editor.update_histogram()
            return
        assert self.make_fn is not None
        fn = editor.preview_region_fn(self.make_fn())
        doc = self.doc
        self.generation += 1
        generation = self.generation

        def compute() -> Image.Image | Exception:
            try:
                return doc.preview_active(fn)
            except Exception as exc:  # noqa: BLE001 - a bad parameter combination must not kill the sheet
                return exc

        def show(result: Image.Image | Exception) -> None:
            if generation != self.generation or not self._open():
                return
            if isinstance(result, Exception):
                editor.notify(str(result))
                return
            editor.begin_preview(self.layer_id, result)
            editor.update_histogram()

        editor.worker.submit(f"preview:{self.sheet_name}", compute, show)

    def restore(self) -> None:
        """Undo an uncommitted adjustment-layer preview (the sheet was cancelled)."""
        self.stop()
        if not self.adjustment:
            return
        index = self.doc.layer_index(self.layer_id)
        if index is not None and self.doc.layers[index].meta != self.original_meta:
            self.doc.layers[index] = self.doc.layers[index].copy(meta=dict(self.original_meta))
            self.doc.touch()
            self.editor.view.invalidate_frame()
            self.editor.update_histogram()

    def schedule(self) -> None:
        if self.job is not None:
            self.editor.shell.after_cancel(self.job)
        self.job = self.editor.shell.after(120, self.now)

    def stop(self) -> None:
        self.generation += 1  # results still in flight are ignored
        self.editor.worker.cancel(f"preview:{self.sheet_name}")
        if self.job is not None:
            self.editor.shell.after_cancel(self.job)
            self.job = None


NEUTRAL_LEVELS: dict[str, float] = {"in_black": 0, "in_white": 255, "gamma": 1.0, "out_black": 0, "out_white": 255}


def _layer_histogram(editor) -> Any:
    """Histogram behind curve/levels editors: the layer's pixels, or what lies below an adjustment layer."""
    doc = editor.doc
    scale = min(1.0, 320 / max(doc.width, doc.height))
    if doc.active_layer.kind == "adjustment":
        return curves_mod.histogram(doc.composite_below(doc.active_index, scale))
    source = doc.edit_image()
    small = source.resize((max(1, round(doc.width * scale)), max(1, round(doc.height * scale))), Image.Resampling.BILINEAR)
    return curves_mod.histogram(small)


def curves_sheet(editor) -> None:
    """Photoshop-style Curves: a master RGB curve plus one per channel, with live preview."""
    doc = editor.doc
    if doc is None:
        return
    layer = doc.active_layer
    adjustment = layer.kind == "adjustment"
    curves = curves_mod.default_curves()
    for channel, points in (layer.meta.get("curves") or {}).items() if adjustment else ():
        curves[channel] = [(float(x), float(y)) for x, y in points]
    name = "curves"
    preview = LivePreview(
        editor,
        name,
        lambda: partial(curves_mod.apply_curves, curves=curves_mod.serializable(curves)),
        lambda: {"curves": None if curves_mod.is_identity(curves) else curves_mod.serializable(curves)},
    )
    readout = Label("", style="muted", width=200)

    def changed() -> None:
        if curve_editor.selected is not None and curve_editor.selected < len(curve_editor.points):
            x, y = curve_editor.points[curve_editor.selected]
            readout.set_text(tr("Input {x}   Output {y}").format(x=round(x), y=round(y)))
        else:
            readout.set_text("")
        preview.schedule()

    curve_editor = CurveEditor(curves, "rgb", changed, _layer_histogram(editor), size=260)
    channels = list(curves_mod.CHANNELS)

    def pick_channel(index: int) -> None:
        curve_editor.set_channel(channels[index])
        changed()

    def reset() -> None:
        curve_editor.reset(all_channels=True)

    def apply() -> None:
        preview.stop()
        chosen = curves_mod.serializable(curves)
        if adjustment:
            original = preview.original_meta.get("curves")
            preview.restore()
            editor.dismiss_modal()
            new_value = None if curves_mod.is_identity(chosen) else chosen
            if new_value != original:
                editor.doc_op(lambda d: d.set_layer_meta(label="Curves", curves=new_value))
            return
        editor.dismiss_modal()
        if not curves_mod.is_identity(chosen):
            editor.doc_op(lambda d: d.apply_curves(chosen))

    items: list[Widget] = [
        Label(tr("Curves"), style="title"),
        Label(tr("Click to add a point · drag to shape · drag away to remove"), style="muted"),
        Spacer(8),
        Segmented([tr("RGB"), tr("Red"), tr("Green"), tr("Blue")], 0, pick_channel),
        Spacer(6),
        curve_editor,
        Row([readout, Spacer(), Button(tr("Reset"), reset, small=True)], gap=8, height=30),
        Spacer(6),
        _buttons(editor, tr("Apply"), apply),
    ]
    panel = _sheet(editor, name, items, width=340, default=apply)
    panel.on_dismiss = preview.restore  # type: ignore[attr-defined]
    editor.view.set_focus(curve_editor)
    preview.now()


def levels_sheet(editor) -> None:
    """Graphical Levels: input black / grey / white and output range over the layer's histogram."""
    doc = editor.doc
    if doc is None:
        return
    name = "levels"
    layer = doc.active_layer
    adjustment = layer.kind == "adjustment"
    levels_editor = LevelsEditor(_layer_histogram(editor), height=170)
    if adjustment and layer.meta.get("levels"):
        for key, value in layer.meta["levels"].items():
            setattr(levels_editor, key, value)

    def levels_fn() -> Callable[[Image.Image], Image.Image]:
        values = levels_editor.values()
        return lambda im: FILTERS["Levels"].apply(im, **values)

    def levels_meta() -> dict[str, Any]:
        values = levels_editor.values()
        return {"levels": None if values == NEUTRAL_LEVELS else values}

    preview = LivePreview(editor, name, levels_fn, levels_meta)
    fields: dict[str, NumberField] = {}
    state = {"syncing": False}

    def sync_fields() -> None:
        state["syncing"] = True
        try:
            fields["in_black"].set_value(levels_editor.in_black)
            fields["in_white"].set_value(levels_editor.in_white)
            fields["gamma"].set_value(round(levels_editor.gamma * 100))
            fields["out_black"].set_value(levels_editor.out_black)
            fields["out_white"].set_value(levels_editor.out_white)
        finally:
            state["syncing"] = False

    def changed() -> None:
        sync_fields()
        preview.schedule()

    levels_editor.on_change = changed

    def field_changed(key: str, value: int) -> None:
        if state["syncing"]:
            return
        if key == "gamma":
            levels_editor.gamma = max(0.1, value / 100)
        elif key == "in_black":
            levels_editor.in_black = min(value, levels_editor.in_white - 2)
        elif key == "in_white":
            levels_editor.in_white = max(value, levels_editor.in_black + 2)
        else:
            setattr(levels_editor, key, value)
        levels_editor.invalidate()
        preview.schedule()

    def number(key: str, label: str, value: int, lo: int, hi: int) -> NumberField:
        field = NumberField(label, value, lo, hi, partial(field_changed, key), suffix="", field_width=60)
        fields[key] = field
        return field

    def apply() -> None:
        preview.stop()
        values = levels_editor.values()
        if adjustment:
            original = preview.original_meta.get("levels")
            preview.restore()
            editor.dismiss_modal()
            new_value = None if values == NEUTRAL_LEVELS else values
            if new_value != original:
                editor.doc_op(lambda d: d.set_layer_meta(label="Levels", levels=new_value))
            return
        editor.dismiss_modal()
        if values != NEUTRAL_LEVELS:
            editor.doc_op(lambda d: d.apply_filter("Levels", **values))

    items: list[Widget] = [
        Label(tr("Levels"), style="title"),
        Label(tr("Drag the input markers and the output range"), style="muted"),
        Spacer(8),
        levels_editor,
        Spacer(4),
        Row(
            [
                number("in_black", tr("In"), 0, 0, 253),
                number("gamma", tr("Gamma"), 100, 10, 999),
                number("in_white", tr("White"), 255, 2, 255),
            ],
            gap=8,
            height=40,
        ),
        Row([number("out_black", tr("Out"), 0, 0, 255), number("out_white", tr("White"), 255, 0, 255), Spacer()], gap=8, height=40),
        Row([Spacer(), Button(tr("Reset"), levels_editor.reset, small=True)], gap=8, height=30),
        Spacer(6),
        _buttons(editor, tr("Apply"), apply),
    ]
    panel = _sheet(editor, name, items, width=400, default=apply)
    panel.on_dismiss = preview.restore  # type: ignore[attr-defined]
    editor.view.set_focus(levels_editor)
    sync_fields()
    preview.now()


# ---------------------------------------------------------------- effects --
def effects_sheet(editor) -> None:
    """Layer effects (drop shadow, outer glow, stroke) - live, re-editable, stored with the layer."""
    doc = editor.doc
    if doc is None:
        return
    layer = doc.active_layer
    if layer.kind == "adjustment":
        editor.notify(tr("Effects need a layer with pixels"))
        return
    fx = effects_mod.normalize_effects(layer.meta.get("effects"))
    name = "effects"
    preview = LivePreview(
        editor, name, None, lambda: {"effects": None if effects_mod.is_empty(fx) else effects_mod.normalize_effects(fx)}, meta_only=True
    )
    sections: list[Widget] = [Label(tr("Layer Effects"), style="title"), Label(layer.name, style="muted"), Spacer(6)]

    def slider(effect: str, key: str, label: str, lo: int, hi: int) -> Slider:
        def changed(value: float) -> None:
            fx[effect][key] = int(value)
            preview.schedule()

        return Slider(label, lo, hi, fx[effect][key], effects_mod.DEFAULTS[effect][key], on_change=changed, fmt="{:d}", compact=True)

    def toggle(effect: str, label: str) -> Toggle:
        def changed(value: bool) -> None:
            fx[effect]["enabled"] = value
            preview.schedule()

        return Toggle(label, fx[effect]["enabled"], changed)

    def colour(effect: str) -> Row:
        dot = ColorDot((*fx[effect]["color"], 255))

        def pick() -> None:
            chosen = editor.shell.ask_color((*fx[effect]["color"], 255))
            if chosen is not None:
                fx[effect]["color"] = list(chosen[:3])
                dot.set_color(chosen)
                preview.schedule()

        return Row([Label(tr("Colour"), width=70), dot, Button(tr("Pick…"), pick, small=True), Spacer()], gap=8, height=30)

    positions = list(effects_mod.STROKE_POSITIONS)

    def position_changed(index: int) -> None:
        fx["stroke"]["position"] = positions[index]
        preview.schedule()

    sections += [
        toggle("shadow", tr("Drop Shadow")),
        slider("shadow", "dx", tr("Offset X"), -100, 100),
        slider("shadow", "dy", tr("Offset Y"), -100, 100),
        slider("shadow", "blur", tr("Blur"), 0, 100),
        slider("shadow", "opacity", tr("Opacity"), 0, 100),
        colour("shadow"),
        Spacer(8),
        toggle("glow", tr("Outer Glow")),
        slider("glow", "size", tr("Size"), 1, 100),
        slider("glow", "opacity", tr("Opacity"), 0, 100),
        colour("glow"),
        Spacer(8),
        toggle("stroke", tr("Stroke")),
        slider("stroke", "size", tr("Size"), 1, 50),
        slider("stroke", "opacity", tr("Opacity"), 0, 100),
        Segmented([tr(p.title()) for p in positions], positions.index(fx["stroke"]["position"]), position_changed),
        colour("stroke"),
        Spacer(10),
    ]
    Slider.align([w for w in sections if isinstance(w, Slider)])

    def clear() -> None:
        for effect in fx.values():
            effect["enabled"] = False
        apply()

    def apply() -> None:
        original = preview.original_meta.get("effects")
        preview.restore()
        editor.dismiss_modal()
        new_value = None if effects_mod.is_empty(fx) else effects_mod.normalize_effects(fx)
        if new_value != original:
            editor.doc_op(lambda d: d.set_layer_meta(label="Layer Effects", effects=new_value))

    sections.append(
        Row(
            [
                Button(tr("Remove All"), clear, small=True),
                Spacer(),
                Button(tr("Cancel"), editor.dismiss_modal),
                Button(tr("OK"), apply, primary=True),
            ],
            gap=8,
            height=34,
        )
    )
    panel = Panel(
        Column(sections),
        editor.sheet_place(360),
        padding=22,
        fit_height=False,
        modal=True,
        name=name,
        scrollable=True,
    )
    panel.default_action = apply  # type: ignore[attr-defined]
    panel.on_dismiss = preview.restore  # type: ignore[attr-defined]
    editor.open_sheet(panel)
    preview.now()


# ----------------------------------------------------------------- brushes --
def brush_sample(preset: BrushPreset, size: tuple[int, int] = (120, 36)) -> Image.Image:
    """A stroke sample of *preset* drawn in the theme's text colour (for the picker)."""
    import numpy as np

    cov = brushes_mod.stroke_preview(preset, size)
    r, g, b, _a = PALETTE.fg
    sample = Image.new("RGBA", size, (r, g, b, 0))
    sample.putalpha(Image.fromarray(np.clip(cov * 255 + 0.5, 0, 255).astype(np.uint8), "L"))
    return sample


def brush_sheet(editor, tool) -> None:
    """Brush presets and settings for a brush-like tool (brush, eraser, clone stamp, healing brush)."""
    library: brushes_mod.BrushLibrary = editor.brushes
    name = "brushes"
    state: dict[str, Any] = {"preset": tool.current_preset()}
    rows_column = Column()
    settings_column = Column()
    name_field = TextField(tr("Name"), state["preset"].name, field_width=200, max_length=40)

    def select(preset: BrushPreset) -> None:
        state["preset"] = preset
        tool.set_preset(preset)
        name_field.set_text(preset.name)
        rebuild_rows()
        rebuild_settings()

    def rebuild_rows() -> None:
        rows_column.clear()
        current = state["preset"]
        for preset in library.presets():
            selected = preset.name == current.name
            rows_column.add(BrushRow(preset.name, preset.description(), brush_sample(preset), partial(select, preset), selected))
        rows_column.invalidate()

    def tweak(**changes: Any) -> None:
        state["preset"] = replace(state["preset"], **changes).normalized()
        tool.set_preset(state["preset"])  # the list shows the saved presets, so it needs no rebuild here

    def slider(key: str, label: str, lo: int, hi: int, fmt: str = "{:d}") -> Slider:
        value = getattr(state["preset"], key)
        widget = Slider(label, lo, hi, value, value, on_change=lambda v: tweak(**{key: int(v)}), fmt=fmt, compact=True)
        return widget

    def rebuild_settings() -> None:
        settings_column.clear()
        preset = state["preset"]
        tip_names = [t for t in brushes_mod.TIPS if t != "custom" or preset.texture]
        index = tip_names.index(preset.tip) if preset.tip in tip_names else 0
        sliders = [
            slider("spacing", tr("Spacing"), 1, 200, "{:d} %"),
            slider("angle", tr("Angle"), 0, 359, "{:d}°"),
            slider("roundness", tr("Roundness"), 5, 100, "{:d} %"),
            slider("scatter", tr("Scatter"), 0, 300, "{:d} %"),
            slider("size_jitter", tr("Size jitter"), 0, 100, "{:d} %"),
            slider("opacity_jitter", tr("Opacity jitter"), 0, 100, "{:d} %"),
            slider("count", tr("Count"), 1, 8),
        ]
        Slider.align(sliders)
        settings_column.add(
            Label(tr("Tip"), style="caption", uppercase=True),
            Segmented([tr(t.title()) for t in tip_names], index, lambda i: tweak(tip=tip_names[i])),
            Spacer(4),
            *sliders,
        )
        settings_column.invalidate()

    def define_tip() -> None:
        doc = editor.doc
        if doc is None:
            return
        piece = doc.copy_selection()
        if piece is None or piece[0].getbbox() is None:
            editor.notify(tr("Select something with pixels to use as a brush tip"))
            return
        tweak(tip="custom", texture=brushes_mod.texture_from_image(piece[0]))
        rebuild_settings()
        editor.notify(tr("Brush tip taken from the selection"), error=False)

    def save() -> None:
        new_name = name_field.value.strip()
        if not new_name:
            editor.notify(tr("Give the brush a name first"))
            return
        if library.is_builtin(new_name):
            editor.notify(tr("'{name}' is a built-in brush - choose another name").format(name=new_name))
            return
        saved = library.save(replace(state["preset"], name=new_name))
        select(saved)
        editor.notify(tr("Saved brush '{name}'").format(name=new_name), error=False)

    def delete() -> None:
        current = state["preset"]
        if library.is_builtin(current.name) or current.name not in library.user:
            editor.notify(tr("Built-in brushes cannot be deleted"))
            return
        library.delete(current.name)
        select(library.get(tool.default_preset))

    rebuild_rows()
    rebuild_settings()
    items: list[Widget] = [
        Label(tr("Brushes"), style="title"),
        Label(tr("Pick a preset, then tune it below - changes apply to the tool right away"), style="muted"),
        Spacer(6),
        rows_column,
        Spacer(8),
        settings_column,
        Spacer(8),
        Row([Button(tr("Tip from selection"), define_tip, small=True), Spacer()], gap=8, height=30),
        Spacer(4),
        Row([name_field, Button(tr("Save"), save, small=True), Button(tr("Delete"), delete, small=True, danger=True)], gap=8, height=40),
        Spacer(8),
        Row([Spacer(), Button(tr("Done"), editor.dismiss_modal, primary=True)], gap=8, height=34),
    ]
    panel = Panel(Column(items), editor.sheet_place(420), padding=22, fit_height=False, modal=True, name=name, scrollable=True)
    panel.default_action = editor.dismiss_modal  # type: ignore[attr-defined]
    editor.open_sheet(panel)


# ---------------------------------------------------------------------- 3D --
def three_d_sheet(editor, index: int, created: bool = False) -> None:
    """All settings of a 3D object layer with a quick live preview; Cancel on a freshly *created* layer removes it again."""
    doc = editor.doc
    if doc is None:
        return
    original = doc.scene_of(index)
    if original is None:
        return
    layer_id = doc.layers[index].id
    state: dict[str, Any] = {"scene": original, "generation": 0, "job": None}
    name = "3d"

    def preview_now() -> None:
        state["job"] = None
        if editor.doc is not doc or editor.sheet is None or editor.sheet.name != name:
            return
        scene: Scene = state["scene"]
        state["generation"] += 1
        generation = state["generation"]
        size = doc.size

        def job() -> Image.Image:
            return three_d.render(scene, size, supersample=1, scale=0.5).resize(size, Image.Resampling.BILINEAR)

        def show(image: Image.Image) -> None:
            if generation == state["generation"] and editor.sheet is not None and editor.sheet.name == name:
                editor.begin_preview(layer_id, image)

        editor.worker.submit("3d-preview", job, show)

    def schedule() -> None:
        if state["job"] is not None:
            editor.shell.after_cancel(state["job"])
        state["job"] = editor.shell.after(80, preview_now)

    def tweak(**changes: Any) -> None:
        state["scene"] = replace(state["scene"], **changes).normalized()
        schedule()

    def slider(label: str, lo: int, hi: int, value: float, setter: Callable[[float], None], fmt: str = "{:d}") -> Slider:
        return Slider(label, lo, hi, value, value, on_change=setter, fmt=fmt, compact=True)

    def signed(degrees: float) -> float:
        """Angles are shown as -180 … 180 so the neutral 0° sits in the middle of the track."""
        return (degrees + 180) % 360 - 180

    def rot(axis: int) -> Callable[[float], None]:
        def set_rotation(v: float) -> None:
            rotation = list(state["scene"].rotation)
            rotation[axis] = float(v) % 360
            tweak(rotation=tuple(rotation))

        return set_rotation

    def light(axis: int) -> Callable[[float], None]:
        def set_light(v: float) -> None:
            current = list(state["scene"].light)
            current[axis] = float(v) % 360 if axis == 0 else float(v)
            tweak(light=tuple(current))

        return set_light

    dot = ColorDot((*original.color, 255))

    def pick_colour() -> None:
        chosen = editor.shell.ask_color((*state["scene"].color, 255))
        if chosen is not None:
            dot.set_color(chosen)
            tweak(color=tuple(chosen[:3]))

    shapes = [sh for sh in three_d.SHAPES if sh != "extrusion" or original.shape == "extrusion"]
    scene0 = original
    sliders = [
        slider(tr("Size"), 10, max(2000, int(scene0.size)), scene0.size, lambda v: tweak(size=float(v)), "{:d} px"),
        slider(tr("Depth"), 1, 200, scene0.depth, lambda v: tweak(depth=int(v)), "{:d} %"),
        slider(tr("Rotate X"), -180, 180, signed(scene0.rotation[0]), rot(0), "{:+d}°"),
        slider(tr("Rotate Y"), -180, 180, signed(scene0.rotation[1]), rot(1), "{:+d}°"),
        slider(tr("Rotate Z"), -180, 180, signed(scene0.rotation[2]), rot(2), "{:+d}°"),
        slider(tr("Light direction"), -180, 180, signed(scene0.light[0]), light(0), "{:+d}°"),
        slider(tr("Light height"), -89, 89, scene0.light[1], light(1), "{:+d}°"),
        slider(tr("Ambient"), 0, 100, scene0.ambient, lambda v: tweak(ambient=int(v)), "{:d} %"),
        slider(tr("Shininess"), 1, 128, scene0.shininess, lambda v: tweak(shininess=int(v))),
        slider(tr("Specular"), 0, 100, scene0.specular, lambda v: tweak(specular=int(v)), "{:d} %"),
        slider(tr("Detail"), 6, 96, scene0.segments, lambda v: tweak(segments=int(v))),
    ]
    Slider.align(sliders)
    items: list[Widget] = [
        Label(tr("3D Object"), style="title"),
        Label(tr("Rotate, light and colour the object - it stays editable on its layer"), style="muted"),
        Spacer(6),
        Segmented([tr(sh.title()) for sh in shapes], shapes.index(scene0.shape), lambda i: tweak(shape=shapes[i])),
        Spacer(4),
        *sliders,
        Row([Label(tr("Colour"), width=70), dot, Button(tr("Pick…"), pick_colour, small=True), Spacer()], gap=8, height=30),
        Toggle(tr("Wireframe"), scene0.wireframe, lambda v: tweak(wireframe=bool(v))),
        Spacer(10),
    ]

    def stop() -> None:
        if state["job"] is not None:
            editor.shell.after_cancel(state["job"])
            state["job"] = None
        state["generation"] += 1
        editor.worker.cancel("3d-preview")

    def cancel() -> None:
        stop()
        editor.dismiss_modal()
        if created:
            layer_index = doc.layer_index(layer_id)
            if layer_index is not None and len(doc.layers) > 1:
                doc.set_active(layer_index)
                doc.remove_layer(layer_index)
                editor.after_edit()

    def apply() -> None:
        stop()
        scene: Scene = state["scene"]
        editor.dismiss_modal()
        layer_index = doc.layer_index(layer_id)
        if layer_index is None or scene == original:
            return
        editor.doc_op(lambda d: d.update_3d_layer(layer_index, scene=scene, label="Edit 3D Object"))

    items.append(Row([Spacer(), Button(tr("Cancel"), cancel), Button(tr("OK"), apply, primary=True)], gap=8, height=34))
    panel = Panel(Column(items), editor.sheet_place(440), padding=22, fit_height=False, modal=True, name=name, scrollable=True)
    panel.default_action = apply  # type: ignore[attr-defined]
    panel.on_dismiss = stop  # type: ignore[attr-defined]
    editor.open_sheet(panel)


# ------------------------------------------------------------------- help --
def shortcuts_sheet(editor, shortcuts) -> None:
    rows: list[Widget] = [Label(tr("Keyboard shortcuts"), style="title"), Spacer(6)]
    for keys, desc in shortcuts:
        k = Label(keys, style="body", width=210)
        k.color = PALETTE.accent_hover
        rows.append(Row([k, Label(tr(desc), style="muted")], gap=10, height=26))
    rows += [Spacer(10), Row([Spacer(), Button(tr("Close"), editor.dismiss_modal, primary=True)], height=34)]
    _sheet(editor, "shortcuts", rows, 520, editor.dismiss_modal, dismiss_on_outside=True)


def about_sheet(editor) -> None:
    _sheet(
        editor,
        "about",
        [
            Label(APP_NAME, style="title"),
            Label(tr("Version {version}").format(version=__version__), style="muted"),
            Spacer(8),
            Label(ABOUT_TEXT, style="body"),
            Spacer(12),
            Row([Spacer(), Button(tr("Close"), editor.dismiss_modal, primary=True)], height=34),
        ],
        420,
        editor.dismiss_modal,
        dismiss_on_outside=True,
    )
