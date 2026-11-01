"""The editor: panels, tools, commands and document actions - independent of the GUI toolkit.

:class:`Editor` builds the floating glass panels on a :class:`Surface` and
maps everything the user can do onto :class:`pixelforge.core.document.Document`.
The toolkit layer (:mod:`pixelforge.gui.qt`) supplies a :class:`Shell`
with the window, native menus/dialogs and timers.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, ImageChops

from .. import APP_NAME
from ..core import io as pfio
from ..core.adjustments import GROUPS, SLIDERS, auto_enhance, is_neutral
from ..core.blend import BLEND_MODE_NAMES
from ..core.document import Document, DocumentError
from ..core.filters import CATEGORIES, FilterSpec, filters_in
from ..core.layers import Layer, rgba_to_mask
from ..core.presets import PRESETS, PresetStore
from . import glass, sheets
from .i18n import available_languages, current_language, set_language, tr
from .settings import Settings
from .surface import Surface
from .theme import PALETTE, RGBA, THEME_NAMES, px, set_scale, set_theme
from .tools import ASPECTS, TOOL_CLASSES, CropTool, Tool
from .ui import (
    Button,
    ColorDot,
    ColorSwatches,
    Column,
    FlowRow,
    Histogram,
    IconButton,
    Label,
    LayerRow,
    ListItem,
    Panel,
    PlaceFn,
    Row,
    Segmented,
    Separator,
    Slider,
    Spacer,
    Widget,
)
from .worker import Worker

IS_MAC = sys.platform == "darwin"
MASK_OVERLAY_COLOR = (255, 40, 40)
MOD_LABEL = "⌘" if IS_MAC else "Ctrl+"
CONFIG_DIR = Path(os.environ.get("PIXELFORGE_CONFIG_DIR", Path.home() / ".pixelforge"))

# layout constants in 1x design pixels (converted with px() at layout time)
GAP = 14
TOPBAR_H = 56
SIDE_W = 300
STRIP_W = 38 * 2 + 2 + 16
BOTTOM_H = 50
THUMB = (44, 34)

OPEN_FILETYPES = [
    ("Pictures and projects", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp *.ico *.pfp"),
    ("All files", "*"),
]
EXPORT_FILETYPES = [("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("WebP", "*.webp"), ("TIFF", "*.tif *.tiff"), ("BMP", "*.bmp")]
PROJECT_FILETYPES = [("PixelForge project", "*.pfp")]

SHORTCUTS: tuple[tuple[str, str], ...] = (
    (f"{MOD_LABEL}O  /  {MOD_LABEL}S  /  {MOD_LABEL}E", "Open / save project / export picture"),
    (f"{MOD_LABEL}Z  /  {MOD_LABEL}Shift+Z", "Undo / redo"),
    ("V  M  L  W  C", "Move, Marquee, Lasso, Magic Wand, Crop"),
    ("B  E  G  Shift+G", "Brush, Eraser, Paint Bucket, Gradient"),
    ("S  J", "Clone Stamp, Healing Brush (Alt-click picks the source)"),
    ("T  U  I  Z  H", "Text, Shape, Eyedropper, Zoom, Hand"),
    ("[  /  ]", "Brush size"),
    ("X  /  D", "Swap / reset colours"),
    (f"{MOD_LABEL}A  /  {MOD_LABEL}D  /  {MOD_LABEL}Shift+I", "Select all / deselect / invert"),
    ("Delete  ·  Alt+Backspace  ·  " + MOD_LABEL + "Backspace", "Clear / fill with foreground / background"),
    (f"{MOD_LABEL}Shift+N  /  {MOD_LABEL}J  /  {MOD_LABEL}Shift+E", "New layer / duplicate / merge down"),
    (f"{MOD_LABEL}G  /  {MOD_LABEL}Shift+G", "Group layer / ungroup"),
    (f"{MOD_LABEL}\\", "Edit the layer mask / the layer pixels"),
    (f"{MOD_LABEL}[  /  {MOD_LABEL}]", "Rotate left / right"),
    (f"{MOD_LABEL}T", "Free Transform the layer (Shift: keep aspect · Alt: from the centre)"),
    (f"{MOD_LABEL}L  /  {MOD_LABEL}M", "Levels / Curves"),
    ("Enter  /  Esc", "Apply / cancel the current tool action"),
    (f"{MOD_LABEL}+  /  {MOD_LABEL}−  /  {MOD_LABEL}0  /  {MOD_LABEL}1  /  {MOD_LABEL}2", "Zoom in / out / fit / 100 % / fit width"),
    ("Space + drag  ·  pinch  ·  " + MOD_LABEL + "wheel", "Pan / zoom"),
    ("\\  (hold)", "Compare with the original"),
    ("Double-click a slider  ·  ↑ ↓", "Reset it · nudge (Shift: 10)"),
    (f"Tab  /  {MOD_LABEL}.", "Hide / show the panels"),
)


class Shell(Protocol):
    """Window services the toolkit layer provides to the editor."""

    surface: Surface

    def set_title(self, title: str) -> None: ...

    def ask_open_file(self, filetypes: list[tuple[str, str]]) -> str | None: ...

    def ask_open_files(self, filetypes: list[tuple[str, str]]) -> list[str]: ...

    def ask_save_file(self, suggested: str, filetypes: list[tuple[str, str]]) -> str | None: ...

    def ask_directory(self, title: str) -> str | None: ...

    def ask_color(self, initial: RGBA) -> RGBA | None: ...

    def show_error(self, title: str, message: str) -> None: ...

    def ask_save_changes(self, name: str) -> bool | None: ...  # True = save, False = discard, None = cancel

    def after(self, ms: int, fn: Callable[[], None]) -> object: ...

    def after_cancel(self, handle: object) -> None: ...

    def call_soon(self, fn: Callable[[], None]) -> None: ...  # thread-safe hand-off to the GUI thread

    def announce(self, message: str) -> None: ...  # speak a status message through assistive technology

    def focus_changed(self) -> None: ...

    def commands_changed(self) -> None: ...

    def menus_changed(self) -> None: ...

    def close_window(self) -> None: ...

    def copy_image(self, image: Image.Image) -> None: ...

    def paste_image(self) -> Image.Image | None: ...


@dataclass(frozen=True)
class Command:
    """A menu/keyboard command (the shell turns these into native menu items)."""

    id: str
    label: str
    menu: str
    run: Callable[[], object]
    shortcut: str = ""  # "Mod+O", "Mod+Shift+Z", "Mod++" ...
    enabled: Callable[[], bool] = lambda: True
    role: str = ""  # "quit" | "about" for platform menus
    submenu: str = ""  # optional second level (Filter ▸ Blur ▸ …)
    checked: Callable[[], bool] | None = None  # toggle commands show a check mark


class Editor:
    def __init__(self, shell: Shell, config_dir: Path | None = None, scale: float = 1.0) -> None:
        self.shell = shell
        self.view = shell.surface
        self.view.app = self
        self.scale = set_scale(scale)
        self.config_dir = Path(config_dir) if config_dir else CONFIG_DIR
        self.presets = PresetStore(self.config_dir / "presets.json")
        self.settings = Settings(self.config_dir / "settings.json")
        self.language = set_language(self.settings.get("language"))
        self.theme = set_theme(str(self.settings.get("theme", "dark")))
        self.doc: Document | None = None
        self.compare = False  # show the picture without adjustments ("before")
        self.foreground: RGBA = (0, 0, 0, 255)
        self.background: RGBA = (255, 255, 255, 255)
        self.snap_enabled = bool(self.settings.get("snap", True))
        self.view.show_grid = bool(self.settings.get("grid", False))
        self.view.show_rulers = bool(self.settings.get("rulers", False))
        self._notify_error = False
        self._zoom_memory: dict[str, float] = dict(self.settings.get("zoom", {}))
        self.straighten_preview: tuple[float, bool] | None = None
        self.preview_overrides: dict[int, Image.Image] = {}
        self.preview_mask_overrides: dict[int, Image.Image] = {}
        self.collapsed_groups: set[int] = set()  # layer ids of groups folded in the Layers tab
        self.worker = Worker(shell.call_soon)  # heavy renders / previews run here, results come back on the GUI thread
        self.mask_overlay = bool(self.settings.get("mask_overlay", True))  # red tint over masked-out areas while editing a mask
        self.tools: dict[str, Tool] = {cls.id: cls(self) for cls in TOOL_CLASSES}
        crop_tool = self.tools["crop"]
        assert isinstance(crop_tool, CropTool)
        self.crop_tool: CropTool = crop_tool
        self.tool: Tool | None = None
        self.sheet: Panel | None = None
        self._popover: Panel | None = None
        self._notify_job: object | None = None
        self._panels_hidden = False
        self._tab = 0
        self._thumb_cache: dict[tuple[object, ...], Image.Image] = {}
        self.autosave_path = self.config_dir / "autosave.pfp"
        self.autosave_interval_ms = 120_000
        self._autosave_job: object | None = None
        self.commands = self._build_commands()
        self._build_panels()
        self.select_tool("hand")
        self.refresh_all()
        self._schedule_autosave()
        if self.autosave_path.exists():
            self.shell.after(300, self.offer_recovery)

    # ------------------------------------------------------------ autosave --
    def _schedule_autosave(self) -> None:
        if self._autosave_job is not None:
            self.shell.after_cancel(self._autosave_job)
        self._autosave_job = self.shell.after(self.autosave_interval_ms, self._autosave_tick)

    def _autosave_tick(self) -> None:
        self._autosave_job = None
        self.autosave_now()
        self._schedule_autosave()

    def autosave_now(self) -> bool:
        """Write unsaved work to the autosave file (without touching the document's own path)."""
        if self.doc is None or not self.doc.dirty:
            return False
        try:
            self.autosave_path.parent.mkdir(parents=True, exist_ok=True)
            pfio.save_project(self.doc, self.autosave_path, record=False)
            return True
        except Exception as exc:  # noqa: BLE001 - autosave must never interrupt editing
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
            return False

    def clear_autosave(self) -> None:
        try:
            self.autosave_path.unlink(missing_ok=True)
        except OSError:
            pass

    def offer_recovery(self) -> None:
        """Ask whether to restore the autosaved document found from a previous session."""
        if not self.autosave_path.exists() or self.doc is not None:
            return
        when = time.strftime("%H:%M", time.localtime(self.autosave_path.stat().st_mtime))

        def recover() -> None:
            self.dismiss_modal()
            try:
                doc = pfio.load_project(self.autosave_path)
            except Exception as exc:  # noqa: BLE001
                self.clear_autosave()
                self.notify(tr("The autosaved work could not be read ({error})").format(error=exc))
                return
            doc.path = None
            doc.name = tr("Recovered picture")
            doc.mark_dirty()
            self.set_document(doc)
            self.notify(tr("Recovered your unsaved work - remember to save it"), error=False, timeout=5000)

        def discard() -> None:
            self.dismiss_modal()
            self.clear_autosave()

        sheet = Panel(
            Column(
                [
                    Label(tr("Recover unsaved work?"), style="title"),
                    Label(tr("PixelForge was closed with unsaved changes (autosaved at {time}).").format(time=when), style="muted"),
                    Spacer(12),
                    Row([Spacer(), Button(tr("Discard"), discard), Button(tr("Recover"), recover, primary=True)], gap=8, height=34),
                ]
            ),
            self.sheet_place(420),
            padding=22,
            fit_height=True,
            modal=True,
            name="recover",
        )
        sheet.default_action = recover  # type: ignore[attr-defined]
        self.open_sheet(sheet)

    # ------------------------------------------------------------ rebuild --
    def rebuild(self) -> None:
        """Recreate every panel (after a scale or language change); the document is untouched."""
        self.dismiss_modal()
        if self.tool is not None:
            self.tool.cancel()
        self.view.clear_panels()
        self._thumb_cache.clear()
        self.commands = self._build_commands()
        self._build_panels()
        for tid, btn in self.tool_buttons.items():
            btn.active = self.tool is not None and tid == self.tool.id
        self._show_tool_options()
        self._set_info(tr(self.tool.hint) if self.tool else "")
        self.shell.menus_changed()
        self.refresh_all()

    def rescale(self, scale: float) -> None:
        """Rebuild the interface for a new device pixel ratio (e.g. window moved to another screen)."""
        if abs(scale - self.scale) < 1e-6:
            return
        self.scale = set_scale(scale)
        self.rebuild()
        self.view.fit()

    def set_ui_language(self, code: str) -> None:
        self.language = set_language(code)
        self.settings.set("language", self.language)
        self.rebuild()

    def set_theme(self, name: str) -> None:
        """Switch between the dark, light and high-contrast looks."""
        self.theme = set_theme(name)
        self.settings.set("theme", self.theme)
        self.view.clear_caches()
        self.rebuild()
        self.view.invalidate_frame()

    # ---------------------------------------------------------- commands --
    def _build_commands(self) -> list[Command]:
        has_doc = lambda: self.doc is not None  # noqa: E731
        has_sel = lambda: bool(self.doc and self.doc.selection is not None)  # noqa: E731
        multi = lambda: bool(self.doc and len(self.doc.layers) > 1)  # noqa: E731
        above = lambda: bool(self.doc and self.doc.active_index > 0)  # noqa: E731
        filter_commands = [
            Command(
                f"filter:{spec.name}",
                spec.name + ("…" if spec.params else ""),
                tr("Filter"),
                partial(self.run_filter, spec),
                "",
                has_doc,
                submenu=spec.category,
            )
            for category in CATEGORIES
            for spec in filters_in(category)
        ]
        language_commands = [
            Command(
                f"lang:{code}",
                name,
                tr("View"),
                partial(self.set_ui_language, code),
                "",
                submenu=tr("Language"),
                checked=partial(self._is_language, code),
            )
            for code, name in available_languages().items()
        ]
        theme_commands = [
            Command(
                f"theme:{code}",
                tr(name),
                tr("View"),
                partial(self.set_theme, code),
                "",
                submenu=tr("Theme"),
                checked=partial(self._is_theme, code),
            )
            for code, name in THEME_NAMES.items()
        ]
        op = self.doc_op
        file_menu, edit, select, image, layer, view, helpm = (tr(n) for n in ("File", "Edit", "Select", "Image", "Layer", "View", "Help"))
        return [
            Command("open", tr("Open…"), file_menu, self.open_file, "Mod+O"),
            Command("clear-recent", tr("Clear Recent Files"), file_menu, self.clear_recent, "", self._has_recent),
            Command("save", tr("Save Project"), file_menu, self.save_project, "Mod+S", has_doc),
            Command("save-as", tr("Save Project As…"), file_menu, lambda: self.save_project(ask=True), "Mod+Shift+S", has_doc),
            Command("export", tr("Export Picture…"), file_menu, self.export_picture, "Mod+E", has_doc),
            Command("batch", tr("Apply This Look to Other Pictures…"), file_menu, self.batch_export, "Mod+Shift+B", has_doc),
            Command("close", tr("Close Picture"), file_menu, self.close_document, "Mod+W", has_doc),
            Command("quit", tr("Quit"), file_menu, self.quit_app, "Mod+Q", role="quit"),
            Command("undo", tr("Undo"), edit, self.undo, "Mod+Z", lambda: bool(self.doc and self.doc.can_undo)),
            Command("redo", tr("Redo"), edit, self.redo, "Mod+Shift+Z", lambda: bool(self.doc and self.doc.can_redo)),
            Command("revert", tr("Revert to Original"), edit, self.revert, "", lambda: bool(self.doc and self.doc.can_undo)),
            Command("copy", tr("Copy Picture"), edit, self.copy_image, "Mod+C", has_doc),
            Command("paste", tr("Paste as New Picture"), edit, self.paste_image, "Mod+V"),
            Command("clear", tr("Clear"), edit, self.clear_selection, "", has_sel),
            Command("fill-fg", tr("Fill with Foreground"), edit, lambda: self.fill("foreground"), "Alt+Backspace", has_doc),
            Command("fill-bg", tr("Fill with Background"), edit, lambda: self.fill("background"), "Mod+Backspace", has_doc),
            Command("select-all", tr("Select All"), select, lambda: op(lambda d: d.select_all()), "Mod+A", has_doc),
            Command("deselect", tr("Deselect"), select, lambda: op(lambda d: d.deselect()), "Mod+D", has_sel),
            Command("invert-sel", tr("Invert Selection"), select, lambda: op(lambda d: d.invert_selection()), "Mod+Shift+I", has_doc),
            Command("select-layer", tr("Select Layer Pixels"), select, lambda: op(lambda d: d.select_layer_alpha()), "", has_doc),
            Command("feather", tr("Feather…"), select, lambda: self.modify_selection("feather"), "", has_sel),
            Command("grow", tr("Grow…"), select, lambda: self.modify_selection("grow"), "", has_sel),
            Command("shrink", tr("Shrink…"), select, lambda: self.modify_selection("shrink"), "", has_sel),
            Command("border", tr("Border…"), select, lambda: self.modify_selection("border"), "", has_sel),
            Command("rotate-left", tr("Rotate Left"), image, lambda: op(lambda d: d.rotate90(False), fit=True), "Mod+[", has_doc),
            Command("rotate-right", tr("Rotate Right"), image, lambda: op(lambda d: d.rotate90(True), fit=True), "Mod+]", has_doc),
            Command("flip-h", tr("Flip Horizontal"), image, lambda: op(lambda d: d.flip(True)), "", has_doc),
            Command("flip-v", tr("Flip Vertical"), image, lambda: op(lambda d: d.flip(False)), "", has_doc),
            Command("resize", tr("Resize…"), image, lambda: sheets.resize_sheet(self), "Mod+Alt+I", has_doc),
            Command("straighten", tr("Straighten…"), image, lambda: sheets.straighten_sheet(self), "Mod+Alt+R", has_doc),
            Command("canvas", tr("Canvas Size…"), image, lambda: sheets.canvas_sheet(self), "Mod+Alt+C", has_doc),
            Command("trim", tr("Trim Transparent Edges"), image, self.trim, "", has_doc),
            Command("crop-sel", tr("Crop to Selection"), image, lambda: op(lambda d: d.crop_to_selection(), fit=True), "", has_sel),
            Command("flatten", tr("Flatten Image"), image, lambda: op(lambda d: d.flatten_layers()), "", multi),
            Command("auto", tr("Auto Enhance"), image, self.auto_enhance, "Mod+Shift+A", has_doc),
            Command("levels", tr("Levels…"), image, lambda: sheets.levels_sheet(self), "Mod+L", has_doc),
            Command("curves", tr("Curves…"), image, lambda: sheets.curves_sheet(self), "Mod+M", has_doc),
            Command("transform", tr("Free Transform"), edit, lambda: self.select_tool("transform"), "Mod+T", has_doc),
            Command("new-layer", tr("New Layer"), layer, lambda: op(lambda d: d.add_layer()), "Mod+Shift+N", has_doc),
            Command("new-adjustment", tr("New Adjustment Layer"), layer, lambda: op(lambda d: d.add_adjustment_layer()), "", has_doc),
            Command("group", tr("Group Layer"), layer, lambda: op(lambda d: d.group_active()), "Mod+G", has_doc),
            Command(
                "ungroup",
                tr("Ungroup"),
                layer,
                lambda: op(lambda d: d.ungroup()),
                "Mod+Shift+G",
                lambda: bool(self.doc and (self.doc.active_layer.kind == "group" or self.doc.group_of(self.doc.active_index) is not None)),
            ),
            Command("dup-layer", tr("Duplicate Layer"), layer, lambda: op(lambda d: d.duplicate_layer()), "Mod+J", has_doc),
            Command("del-layer", tr("Delete Layer"), layer, lambda: op(lambda d: d.remove_layer()), "", multi),
            Command("rename-layer", tr("Rename Layer…"), layer, lambda: self.edit_layer(None), "", has_doc),
            Command(
                "layer-up",
                tr("Move Layer Up"),
                layer,
                lambda: op(lambda d: d.raise_layer()),
                "",
                lambda: bool(self.doc and self.doc.active_index < len(self.doc.layers) - 1),
            ),
            Command("layer-down", tr("Move Layer Down"), layer, lambda: op(lambda d: d.lower_layer()), "", above),
            Command("merge-down", tr("Merge Down"), layer, lambda: op(lambda d: d.merge_down()), "Mod+Shift+E", above),
            Command("merge-visible", tr("Merge Visible"), layer, lambda: op(lambda d: d.merge_visible()), "", multi),
            Command(
                "bake",
                tr("Apply Adjustments to Pixels"),
                layer,
                lambda: op(lambda d: d.bake_adjustments()),
                "",
                lambda: bool(self.doc and self.doc.active_layer.has_adjustments),
            ),
            Command(
                "lock",
                tr("Lock / Unlock Layer"),
                layer,
                self.toggle_lock,
                "",
                has_doc,
                checked=lambda: bool(self.doc and self.doc.active_layer.locked),
            ),
            Command(
                "effects",
                tr("Layer Effects…"),
                layer,
                lambda: sheets.effects_sheet(self),
                "",
                lambda: bool(self.doc and self.doc.active_layer.kind != "adjustment"),
            ),
            *self._mask_commands(layer),
            Command("zoom-in", tr("Zoom In"), view, self.view.zoom_in, "Mod++", has_doc),
            Command("zoom-out", tr("Zoom Out"), view, self.view.zoom_out, "Mod+-", has_doc),
            Command("fit", tr("Fit to Window"), view, self.view.fit, "Mod+0", has_doc),
            Command("actual", tr("Actual Size"), view, lambda: self.view.set_zoom(1.0), "Mod+1", has_doc),
            Command("fit-width", tr("Fit Width"), view, self.fit_width, "Mod+2", has_doc),
            Command("compare", tr("Compare with Original"), view, self.toggle_compare, "", has_doc, checked=lambda: self.compare),
            Command("grid", tr("Show Grid"), view, self.toggle_grid, "Mod+'", checked=lambda: self.view.show_grid),
            Command("rulers", tr("Show Rulers"), view, self.toggle_rulers, "Mod+R", checked=lambda: self.view.show_rulers),
            Command("snap", tr("Snap Crop to Guides"), view, self.toggle_snap, "", checked=lambda: self.snap_enabled),
            Command("panels", tr("Hide / Show Panels"), view, self.toggle_panels, "Mod+."),
            *theme_commands,
            *language_commands,
            *filter_commands,
            Command("shortcuts", tr("Keyboard Shortcuts"), helpm, lambda: sheets.shortcuts_sheet(self, SHORTCUTS), ""),
            Command("about", tr("About {app}").format(app=APP_NAME), helpm, lambda: sheets.about_sheet(self), "", role="about"),
        ]

    def _mask_commands(self, menu: str) -> list[Command]:
        op = self.doc_op
        sub = tr("Layer Mask")
        no_mask = lambda: bool(self.doc and self.doc.active_layer.mask is None)  # noqa: E731
        has_mask = lambda: bool(self.doc and self.doc.active_layer.mask is not None)  # noqa: E731
        return [
            Command(
                "mask-reveal", tr("Add Mask (Reveal All)"), menu, lambda: op(lambda d: d.add_mask(mode="reveal")), "", no_mask, submenu=sub
            ),
            Command("mask-hide", tr("Add Mask (Hide All)"), menu, lambda: op(lambda d: d.add_mask(mode="hide")), "", no_mask, submenu=sub),
            Command(
                "mask-edit",
                tr("Edit Mask"),
                menu,
                self.toggle_mask_editing,
                "Mod+\\",
                has_mask,
                submenu=sub,
                checked=lambda: bool(self.doc and self.doc.editing_mask),
            ),
            Command(
                "mask-enabled",
                tr("Enable Mask"),
                menu,
                lambda: op(lambda d: d.set_mask_enabled(not d.active_layer.mask_enabled)),
                "",
                has_mask,
                submenu=sub,
                checked=lambda: bool(self.doc and self.doc.active_layer.mask is not None and self.doc.active_layer.mask_enabled),
            ),
            Command(
                "mask-overlay",
                tr("Show Mask Overlay"),
                menu,
                self.toggle_mask_overlay,
                "",
                has_mask,
                submenu=sub,
                checked=lambda: self.mask_overlay,
            ),
            Command("mask-invert", tr("Invert Mask"), menu, lambda: op(lambda d: d.invert_mask()), "", has_mask, submenu=sub),
            Command("mask-select", tr("Load Mask as Selection"), menu, lambda: op(lambda d: d.select_mask()), "", has_mask, submenu=sub),
            Command("mask-apply", tr("Apply Mask"), menu, lambda: op(lambda d: d.remove_mask(apply=True)), "", has_mask, submenu=sub),
            Command("mask-delete", tr("Delete Mask"), menu, lambda: op(lambda d: d.remove_mask()), "", has_mask, submenu=sub),
        ]

    def toggle_mask_editing(self) -> None:
        if self.doc is None or self.doc.active_layer.mask is None:
            return
        self.doc.set_mask_editing(not self.doc.editing_mask)
        if self.tool is not None:
            self.tool.cancel()
        self.after_edit()

    def toggle_mask_overlay(self) -> None:
        self.mask_overlay = not self.mask_overlay
        self.settings.set("mask_overlay", self.mask_overlay)
        self.view.invalidate_frame()
        self.shell.commands_changed()

    def _is_language(self, code: str) -> bool:
        return current_language() == code

    def _is_theme(self, code: str) -> bool:
        return self.theme == code

    def _has_recent(self) -> bool:
        return bool(self.recent_files())

    def _save_clicked(self) -> None:
        self.save_project()

    def _export_clicked(self) -> None:
        self.export_picture()

    def command(self, command_id: str) -> Command:
        return next(c for c in self.commands if c.id == command_id)

    # ================================================================ UI ==
    def _build_panels(self) -> None:
        v = self.view
        m = MOD_LABEL
        # --- top-left: file & history actions ------------------------------
        self.btn_open = IconButton("open", self.open_file, f"{tr('Open…')}  {m}O")
        self.btn_save = IconButton("save", self._save_clicked, f"{tr('Save Project')}  {m}S")
        self.btn_export = IconButton("export", self._export_clicked, f"{tr('Export Picture…')}  {m}E")
        self.btn_undo = IconButton("undo", self.undo, f"{tr('Undo')}  {m}Z")
        self.btn_redo = IconButton("redo", self.redo, f"{tr('Redo')}  {m}Shift+Z")
        self.btn_revert = IconButton("revert", self.revert, tr("Revert to Original"))
        self.btn_compare = IconButton("eye", self.toggle_compare, tr("Compare with Original") + "  (\\)")
        self.topbar = v.add_panel(
            Panel(
                Row(
                    [
                        self.btn_open,
                        self.btn_save,
                        self.btn_export,
                        Spacer(width=6),
                        self.btn_undo,
                        self.btn_redo,
                        self.btn_revert,
                        Spacer(width=6),
                        self.btn_compare,
                    ],
                    gap=2,
                ),
                lambda vw, vh, pw, ph: (px(GAP), px(GAP), pw, px(TOPBAR_H)),
                padding=(10, 9, 10, 9),
                fit_width=True,
                name="topbar",
            )
        )

        # --- top-centre: tool options ----------------------------------------
        self.options_row = FlowRow([], gap=12)
        self.optionsbar = v.add_panel(
            Panel(
                self.options_row,
                self._options_place,
                padding=(14, 10, 14, 10),
                fit_height=True,
                name="options",
            )
        )

        # --- top-right: file name & more menu ------------------------------
        self.title_label = Label("", style="title")
        self.subtitle_label = Label("", style="small", width=92, align="right")
        self.btn_more = IconButton("more", self.open_more_menu, tr("More…"))
        self.namebar = v.add_panel(
            Panel(
                Row([self.title_label, self.subtitle_label, self.btn_more], gap=6),
                lambda vw, vh, pw, ph: (vw - px(GAP + SIDE_W), px(GAP), px(SIDE_W), px(TOPBAR_H)),
                padding=(18, 9, 10, 9),
                name="namebar",
            )
        )

        # --- left: tools, colours, transforms ---------------------------------
        self.tool_buttons: dict[str, IconButton] = {}
        for tid, t in self.tools.items():
            if t.key:
                shortcut = t.key.upper() if t.key.islower() else f"Shift+{t.key}"
            else:
                shortcut = t.shortcut_label.replace("Mod+", m)
            self.tool_buttons[tid] = IconButton(t.icon, partial(self.select_tool, tid), f"{tr(t.name)}  {shortcut}".rstrip())
        buttons = list(self.tool_buttons.values())
        strip_items: list[Widget] = [Row(buttons[i : i + 2], gap=2, align="left") for i in range(0, len(buttons), 2)]
        self.swatches = ColorSwatches(lambda: (self.foreground, self.background), self.pick_color, self.swap_colors)
        strip_items += [Separator(4), self.swatches, Separator(4)]
        self.transform_buttons = [
            IconButton("rotate-ccw", lambda: self.doc_op(lambda d: d.rotate90(False), fit=True), tr("Rotate Left")),
            IconButton("rotate-cw", lambda: self.doc_op(lambda d: d.rotate90(True), fit=True), tr("Rotate Right")),
            IconButton("flip-h", lambda: self.doc_op(lambda d: d.flip(True)), tr("Flip Horizontal")),
            IconButton("flip-v", lambda: self.doc_op(lambda d: d.flip(False)), tr("Flip Vertical")),
        ]
        strip_items += [Row(self.transform_buttons[:2], gap=2, align="left"), Row(self.transform_buttons[2:], gap=2, align="left")]
        self.strip = v.add_panel(
            Panel(
                Column(strip_items, gap=2),
                lambda vw, vh, pw, ph: (px(GAP), max(px(GAP + TOPBAR_H + GAP), (vh - ph) // 2), px(STRIP_W), ph),
                padding=8,
                fit_height=True,
                name="strip",
            )
        )

        # --- right: adjust / presets / layers / history ------------------------
        self.tabs = Segmented([tr("Adjust"), tr("Presets"), tr("Layers"), tr("History")], self._tab, self._select_tab)
        self.tab_body = Column()
        self.sliders: dict[str, Slider] = {}
        self.adjust_content = self._build_adjust_content()
        self.side = v.add_panel(
            Panel(
                Column([self.tabs, Spacer(10), self.tab_body]),
                lambda vw, vh, pw, ph: (vw - px(GAP + SIDE_W), px(GAP + TOPBAR_H + GAP), px(SIDE_W), vh - px(2 * GAP + TOPBAR_H + GAP)),
                padding=(14, 14, 14, 10),
                scrollable=True,
                name="side",
            )
        )

        # --- bottom: zoom pill & info pill -----------------------------------
        self.zoom_label = Button("100%", self._zoom_menu_toggle, width=64, tooltip=tr("Zoom level"))
        self.zoombar = v.add_panel(
            Panel(
                Row(
                    [
                        IconButton("zoom-out", self.view.zoom_out, f"{tr('Zoom Out')}  {m}−"),
                        self.zoom_label,
                        IconButton("zoom-in", self.view.zoom_in, f"{tr('Zoom In')}  {m}+"),
                        Spacer(width=4),
                        IconButton("fit", self.view.fit, f"{tr('Fit to Window')}  {m}0"),
                        Button("1:1", lambda: self.view.set_zoom(1.0), width=44, tooltip=f"{tr('Actual Size')}  {m}1"),
                    ],
                    gap=2,
                ),
                lambda vw, vh, pw, ph: (self._bottom_center(vw, pw), vh - px(GAP + BOTTOM_H), pw, px(BOTTOM_H)),
                padding=(8, 6, 8, 6),
                fit_width=True,
                name="zoombar",
            )
        )
        self.info_label = Label("", style="small")
        self.info_dot = ColorDot()
        self.info_dot.visible = False
        self.infobar = v.add_panel(
            Panel(
                Row([self.info_dot, self.info_label], gap=6),
                lambda vw, vh, pw, ph: (px(GAP), vh - px(GAP + 38), pw, px(38)),
                padding=(16, 4, 16, 4),
                fit_width=True,
                max_width=460,
                name="infobar",
            )
        )

        # --- crop bar (shown while cropping) ---------------------------------
        self.crop_aspect = Segmented(list(ASPECTS), 0, self._crop_aspect_changed, width=250)
        self.crop_size = Label("", style="muted", width=110, align="center")
        self.btn_crop_apply = Button(tr("Apply"), self.crop_tool.apply, primary=True, tooltip="Enter")
        self.cropbar = v.add_panel(
            Panel(
                Row(
                    [
                        self.crop_aspect,
                        self.crop_size,
                        Button(tr("Cancel"), self.crop_tool.cancel, tooltip="Esc"),
                        self.btn_crop_apply,
                    ],
                    gap=8,
                ),
                lambda vw, vh, pw, ph: (self._bottom_center(vw, pw), vh - px(GAP + BOTTOM_H + GAP + 50), pw, px(50)),
                padding=(10, 8, 10, 8),
                fit_width=True,
                name="cropbar",
            )
        )
        self.cropbar.visible = False

        # --- empty state -------------------------------------------------------
        self.recent_column = Column()
        self.empty_card = v.add_panel(
            Panel(
                Column(
                    [
                        Spacer(6),
                        Label(tr("Open a picture to start"), style="title", align="center"),
                        Label("PNG · JPEG · WebP · TIFF · BMP · GIF · PFP", style="muted", align="center"),
                        Spacer(10),
                        Row([Button(tr("Open…"), self.open_file, icon="open", primary=True, width=150)], align="center", height=34),
                        Spacer(4),
                        Label(
                            f"{MOD_LABEL}O  ·  {tr('drop a file anywhere')}  ·  {MOD_LABEL}V {tr('to paste')}",
                            style="small",
                            align="center",
                        ),
                        self.recent_column,
                    ]
                ),
                lambda vw, vh, pw, ph: ((vw - px(380)) // 2, (vh - ph) // 2 - px(20), px(380), ph),
                padding=20,
                fit_height=True,
                name="empty",
            )
        )
        self._fill_recent()

    def _build_adjust_content(self) -> Column:
        self.histogram = Histogram()
        self.btn_auto = Button(tr("Auto"), self.auto_enhance, small=True, tooltip=tr("Auto Enhance") + f"  {MOD_LABEL}Shift+A")
        self.adjust_note = Label(tr("Adjustment layer - affects every layer below it"), style="small", align="center")
        self.adjust_note.visible = False
        items: list[Widget] = [self.histogram, Spacer(6), Row([Spacer(), self.btn_auto], height=24), self.adjust_note, Spacer(2)]
        self.group_reset_buttons: dict[str, Button] = {}
        for group in GROUPS:
            reset = Button(tr("Reset"), partial(self.reset_group, group), small=True, tooltip=tr("Reset these sliders"))
            self.group_reset_buttons[group] = reset
            items.append(Row([Label(tr(group), style="caption", uppercase=True), reset], gap=6, height=26))
            for key, spec in SLIDERS.items():
                if spec.group != group:
                    continue
                slider = Slider(
                    tr(spec.label),
                    spec.min,
                    spec.max,
                    spec.default,
                    spec.default,
                    on_change=partial(self._adjust_live_value, key),
                    on_commit=partial(self._adjust_commit, key),
                    tooltip=spec.tooltip,
                )
                self.sliders[key] = slider
                items.append(slider)
            items.append(Spacer(6))
        self.btn_reset_adjust = Button(tr("Reset all adjustments"), self.reset_adjustments, width=SIDE_W - 28 - 8)
        items.append(Row([self.btn_reset_adjust], align="center", height=34))
        return Column(items)

    def _options_place(self, vw: int, vh: int, pw: int, ph: int) -> tuple[int, int, int, int]:
        """The tool options bar sits centred in the gap between the top-left bar and the name bar, never overlapping them."""
        left = self.topbar.rect[2] + px(GAP)
        right = vw - px(GAP + SIDE_W) - px(GAP)
        available = max(px(120), right - left)
        width = min(self.options_row.preferred_width() + px(28), available)
        x = left + (available - width) // 2
        y = px(GAP) + (px(TOPBAR_H) - ph) // 2 if ph <= px(TOPBAR_H) else px(GAP)
        return x, y, width, ph

    @staticmethod
    def _bottom_center(vw: int, pw: int) -> int:
        return (vw - px(SIDE_W + GAP) + px(STRIP_W + GAP)) // 2 - pw // 2

    def content_area(self, vw: int, vh: int) -> tuple[int, int, int, int]:
        """Viewport region between the floating panels, used to fit the picture."""
        if self._panels_hidden:
            return 0, 0, vw, vh
        top = px(GAP + TOPBAR_H + GAP)
        if getattr(self, "optionsbar", None) is not None and self.optionsbar.visible:
            top = max(top, self.optionsbar.rect[3] + px(GAP))
        return px(GAP + STRIP_W + GAP), top, vw - px(GAP + SIDE_W + GAP), vh - px(GAP + BOTTOM_H + GAP)

    def _content_center_x(self, vw: int, vh: int) -> int:
        x0, _y0, x1, _y1 = self.content_area(vw, vh) if self.doc is not None else (0, 0, vw, vh)
        return (x0 + x1) // 2

    def sheet_place(self, width: int):
        """Centred on the picture area (or the window when nothing is open)."""

        def place(vw: int, vh: int, pw: int, ph: int) -> tuple[int, int, int, int]:
            w = min(px(width), vw - 2 * px(GAP))
            x = min(max(px(GAP), self._content_center_x(vw, vh) - w // 2), vw - w - px(GAP))
            return x, max(px(GAP), (vh - ph) // 2 - px(30)), w, ph

        return place

    def bottom_sheet_place(self, width: int):
        def place(vw: int, vh: int, pw: int, ph: int) -> tuple[int, int, int, int]:
            w = min(px(width), vw - 2 * px(GAP))
            x = min(max(px(GAP), self._content_center_x(vw, vh) - w // 2), vw - w - px(GAP))
            return x, vh - ph - px(GAP + BOTTOM_H + GAP + 8), w, ph

        return place

    # ------------------------------------------------------------ tabs --
    def _select_tab(self, index: int) -> None:
        self._tab = index
        self.side.scroll_y = 0
        self._fill_tab()
        self.view.layout_panels()

    def _fill_tab(self) -> None:
        self.tab_body.clear()
        if self._tab == 0:
            self.tab_body.add(self.adjust_content)
            self._sync_sliders()
        elif self._tab == 1:
            self.tab_body.add(*self._preset_items())
        elif self._tab == 2:
            self.tab_body.add(*self._layer_items())
        else:
            self.tab_body.add(*self._history_items())

    def _preset_items(self) -> list[Widget]:
        current = self.doc.active_layer.adjustments if self.doc else None
        save = Button(tr("Save current look as preset…"), lambda: sheets.save_preset_sheet(self), icon="plus", width=SIDE_W - 28 - 8)
        save.enabled = current is not None and not is_neutral(current)
        items: list[Widget] = [Row([save], align="center", height=34), Spacer(6)]

        def describe(adj: dict[str, int]) -> str:
            return ", ".join(f"{tr(SLIDERS[k].label)} {v:+d}" for k, v in adj.items() if v != SLIDERS[k].default) or tr("Neutral")

        for name in PRESETS:
            adj = self.presets.get(name)
            checked = current is not None and all(current.get(k) == v for k, v in adj.items())
            items.append(ListItem(name, partial(self.apply_preset, name), checked=checked, subtitle=describe(PRESETS[name])))
        if self.presets.user:
            items += [Spacer(4), Label(tr("My presets"), style="caption", uppercase=True)]
            for name in sorted(self.presets.user):
                adj = self.presets.get(name)
                checked = current is not None and all(current.get(k) == v for k, v in adj.items())
                items.append(
                    ListItem(
                        name,
                        partial(self.apply_preset, name),
                        checked=checked,
                        subtitle=describe(adj),
                        trailing_icon="trash",
                        on_trailing=partial(self.delete_preset, name),
                        trailing_tooltip=tr("Delete preset"),
                    )
                )
        return items

    def _history_items(self) -> list[Widget]:
        if self.doc is None:
            return [Label(tr("No picture open"), style="muted", align="center")]
        items: list[Widget] = [
            Row(
                [Button(tr("Take snapshot…"), lambda: sheets.snapshot_sheet(self), icon="plus", width=SIDE_W - 28 - 8)],
                align="center",
                height=34,
            ),
            Spacer(6),
        ]
        if self.doc.snapshots:
            items.append(Label(tr("Snapshots"), style="caption", uppercase=True))
            for i, (name, _snap) in enumerate(self.doc.snapshots):
                items.append(
                    ListItem(
                        name,
                        partial(self.restore_snapshot, i),
                        icon="clock",
                        trailing_icon="trash",
                        on_trailing=partial(self.delete_snapshot, i),
                        trailing_tooltip=tr("Delete snapshot"),
                    )
                )
            items += [Spacer(6), Label(tr("Steps"), style="caption", uppercase=True)]
        current = self.doc.history.index
        for i, label in enumerate(self.doc.history.labels):
            items.append(ListItem(label, partial(self.go_to_history, i), selected=i == current, dim=i > current))
        return items

    # layers tab ----------------------------------------------------------------
    def _thumbnail(self, layer: Layer) -> Image.Image:
        key = (layer.id, id(layer.image), layer.kind)
        thumb = self._thumb_cache.get(key)
        if thumb is None:
            if len(self._thumb_cache) > 64:
                self._thumb_cache.clear()
            tw, th = px(THUMB[0]), px(THUMB[1])
            if layer.kind in ("adjustment", "group"):
                thumb = Image.new("RGBA", (tw, th), (*PALETTE.fg_dim[:3], 255))
                icon = glass.icon("adjust" if layer.kind == "adjustment" else "folder", px(20), (255, 255, 255, 255))
                thumb.alpha_composite(icon, ((tw - icon.width) // 2, (th - icon.height) // 2))
                self._thumb_cache[key] = thumb
                return thumb
            scale = min(tw / layer.image.width, th / layer.image.height)
            size = (max(1, round(layer.image.width * scale)), max(1, round(layer.image.height * scale)))
            small = layer.image.resize(size, Image.Resampling.BILINEAR, reducing_gap=2.0)
            thumb = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
            light, dark = PALETTE.checker
            cell = max(2, px(5))
            for yy in range(0, th, cell):
                for xx in range(0, tw, cell):
                    thumb.paste((*(light if (xx // cell + yy // cell) % 2 else dark), 255), (xx, yy, xx + cell, yy + cell))
            thumb.alpha_composite(small, ((tw - size[0]) // 2, (th - size[1]) // 2))
            self._thumb_cache[key] = thumb
        return thumb

    def _mask_thumbnail(self, layer: Layer) -> Image.Image | None:
        if layer.mask is None:
            return None
        key = (layer.id, id(layer.mask), "mask")
        thumb = self._thumb_cache.get(key)
        if thumb is None:
            tw, th = px(THUMB[0]), px(THUMB[1])
            scale = min(tw / layer.mask.width, th / layer.mask.height)
            size = (max(1, round(layer.mask.width * scale)), max(1, round(layer.mask.height * scale)))
            small = layer.mask.resize(size, Image.Resampling.BILINEAR, reducing_gap=2.0).convert("RGBA")
            thumb = Image.new("RGBA", (tw, th), (*PALETTE.fg_dim[:3], 255))
            thumb.paste(small, ((tw - size[0]) // 2, (th - size[1]) // 2))
            self._thumb_cache[key] = thumb
        return thumb

    def select_mask_target(self, index: int) -> None:
        """Click on a mask thumbnail: make the layer active and edit its mask."""
        if self.doc is None:
            return
        self.doc.set_active(index)
        self.doc.set_mask_editing(True)
        if self.tool is not None:
            self.tool.cancel()
        self.after_edit()

    def _layer_items(self) -> list[Widget]:
        doc = self.doc
        if doc is None:
            return [Label(tr("No picture open"), style="muted", align="center")]
        active = doc.active_layer
        self.blend_button = Button(tr(active.blend_mode.title()), self.open_blend_menu, width=150)
        self.layer_opacity = Slider(
            tr("Opacity"),
            0,
            100,
            round(active.opacity * 100),
            100,
            on_change=self._opacity_live,
            on_commit=self._opacity_commit,
            fmt="{:d}",
        )
        items: list[Widget] = [Row([Label(tr("Blend"), width=50), self.blend_button], gap=8, height=34), self.layer_opacity, Spacer(4)]
        for index in range(len(doc.layers) - 1, -1, -1):
            layer = doc.layers[index]
            depth, hidden = self._group_depth(index)
            if hidden:
                continue
            is_group = layer.kind == "group"
            items.append(
                LayerRow(
                    layer,
                    index == doc.active_index,
                    self._thumbnail(layer),
                    on_select=partial(self.select_layer, index),
                    on_toggle=partial(self.toggle_layer_visible, index),
                    on_edit=partial(self.edit_layer, index),
                    mask_thumbnail=self._mask_thumbnail(layer),
                    mask_active=index == doc.active_index and doc.editing_mask,
                    on_select_mask=partial(self.select_mask_target, index),
                    depth=depth,
                    collapsed=(layer.id in self.collapsed_groups) if is_group else None,
                    on_collapse=partial(self.toggle_group_collapsed, layer.id) if is_group else None,
                    member_count=len(doc.group_members(index)) if is_group else 0,
                )
            )
        op = self.doc_op
        self.layer_buttons = {
            "plus": IconButton("plus", lambda: op(lambda d: d.add_layer()), tr("New Layer"), size=30, icon_size=15),
            "adjust": IconButton(
                "adjust", lambda: op(lambda d: d.add_adjustment_layer()), tr("New Adjustment Layer"), size=30, icon_size=15
            ),
            "duplicate": IconButton("duplicate", lambda: op(lambda d: d.duplicate_layer()), tr("Duplicate Layer"), size=30, icon_size=15),
            "merge": IconButton("merge", lambda: op(lambda d: d.merge_down()), tr("Merge Down"), size=30, icon_size=15),
            "up": IconButton("up", lambda: op(lambda d: d.raise_layer()), tr("Move Layer Up"), size=30, icon_size=15),
            "down": IconButton("down", lambda: op(lambda d: d.lower_layer()), tr("Move Layer Down"), size=30, icon_size=15),
            "group": IconButton("folder", lambda: op(lambda d: d.group_active()), tr("Group Layer"), size=30, icon_size=15),
            "mask": IconButton("mask", self.open_mask_menu, tr("Layer Mask"), size=30, icon_size=15),
            "fx": IconButton("fx", lambda: sheets.effects_sheet(self), tr("Layer Effects…"), size=30, icon_size=15),
            "lock": IconButton("unlock" if active.locked else "lock", self.toggle_lock, tr("Lock / Unlock Layer"), size=30, icon_size=15),
            "trash": IconButton("trash", lambda: op(lambda d: d.remove_layer()), tr("Delete Layer"), size=30, icon_size=15),
        }
        self.layer_buttons["merge"].enabled = doc.active_index > 0
        self.layer_buttons["down"].enabled = doc.active_index > 0
        self.layer_buttons["up"].enabled = doc.active_index < len(doc.layers) - 1
        self.layer_buttons["trash"].enabled = len(doc.layers) > 1
        first = [self.layer_buttons[k] for k in ("plus", "adjust", "group", "duplicate", "merge", "trash")]
        second = [self.layer_buttons[k] for k in ("up", "down", "mask", "fx", "lock")]
        items += [Spacer(6), Row(first, gap=4, align="center", height=30), Spacer(2), Row(second, gap=4, align="center", height=30)]
        return items

    def _group_depth(self, index: int) -> tuple[int, bool]:
        """Nesting depth of a layer in the Layers list and whether a collapsed group hides it."""
        assert self.doc is not None
        depth, hidden = 0, False
        parent = self.doc.group_of(index)
        seen = 0
        while parent is not None and seen < len(self.doc.layers):
            depth += 1
            if self.doc.layers[parent].id in self.collapsed_groups:
                hidden = True
            parent = self.doc.group_of(parent)
            seen += 1
        return depth, hidden

    def toggle_group_collapsed(self, layer_id: int) -> None:
        if layer_id in self.collapsed_groups:
            self.collapsed_groups.discard(layer_id)
        else:
            self.collapsed_groups.add(layer_id)
        self.refresh_all()

    def select_layer(self, index: int) -> None:
        """Click on a layer row (or its picture thumbnail): activate the layer and target its pixels."""
        if self.doc is None:
            return
        changed = index != self.doc.active_index or self.doc.editing_mask
        self.doc.set_active(index)
        self.doc.set_mask_editing(False)
        if changed:
            if self.tool is not None:
                self.tool.cancel()
            self.view.invalidate_frame()
            self.refresh_all()

    def open_mask_menu(self) -> None:
        """The Layer Mask commands as a popover next to the mask button."""
        if self.doc is None:
            return
        items: list[Widget] = []
        for cmd in self.commands:
            if not cmd.id.startswith("mask-") or not cmd.enabled():
                continue
            checked = cmd.checked() if cmd.checked is not None else False
            items.append(ListItem(cmd.label, partial(self.menu_action, cmd.run), checked=checked))
        x0, y0, x1, y1 = self.side.widget_rect_in_view(self.layer_buttons["mask"])
        self.open_popover(
            "mask",
            items,
            lambda vw, vh, pw, ph: (min(x0, vw - px(250) - px(GAP)), max(px(GAP), y0 - ph - px(6)), px(250), ph),
            width=250,
        )

    def toggle_layer_visible(self, index: int) -> None:
        if self.doc:
            self.doc.set_layer_props(index, visible=not self.doc.layers[index].visible)
            self.after_edit()

    def toggle_lock(self) -> None:
        self.doc_op(lambda d: d.set_layer_props(locked=not d.active_layer.locked))

    def edit_layer(self, index: int | None) -> None:
        """Double-click on a layer: edit text layers, rename others."""
        if self.doc is None:
            return
        index = self.doc.active_index if index is None else index
        if self.doc.layers[index].kind == "text":
            self.doc.set_active(index)
            sheets.text_sheet(self, layer_index=index)
        else:
            sheets.rename_layer_sheet(self, index)

    def open_blend_menu(self) -> None:
        if self.doc is None:
            return
        current = self.doc.active_layer.blend_mode

        def choose(mode: str) -> None:
            self.close_popover()
            self.doc_op(lambda d: d.set_layer_props(blend_mode=mode))

        items = [ListItem(tr(m.title()), partial(choose, m), checked=m == current) for m in BLEND_MODE_NAMES]
        x0, y0, x1, y1 = self.side.widget_rect_in_view(self.blend_button)
        self.open_popover(
            "blend", items, lambda vw, vh, pw, ph: (x0, min(y1 + px(4), vh - px(380) - px(GAP)), px(200), px(380)), scrollable=True
        )

    def _opacity_live(self, value: float) -> None:
        if self.doc and abs(self.doc.active_layer.opacity - value / 100) > 1e-6:
            self.doc.set_layer_props(commit=False, opacity=value / 100)
            self.view.invalidate_frame()

    def _opacity_commit(self) -> None:
        doc = self.doc
        if doc is None:
            return
        current = doc.history.current
        if current is None or current.layers[doc.active_index].opacity != doc.active_layer.opacity:
            doc.commit(tr("Layer Opacity"))
            self.refresh_all()

    def _sync_sliders(self) -> None:
        adj = self.doc.active_layer.adjustments if self.doc else {}
        self.adjust_note.visible = bool(self.doc and self.doc.active_layer.kind == "adjustment")
        for key, slider in self.sliders.items():
            slider.set(adj.get(key, SLIDERS[key].default))
            slider.enabled = self.doc is not None
        self.btn_reset_adjust.enabled = bool(self.doc) and not is_neutral(adj)
        self.btn_auto.enabled = bool(self.doc)
        for group, button in self.group_reset_buttons.items():
            dirty = any(adj.get(k, sp.default) != sp.default for k, sp in SLIDERS.items() if sp.group == group)
            button.enabled = bool(self.doc) and dirty
            button.visible = button.enabled

    # ---------------------------------------------------------- refresh --
    def refresh_all(self) -> None:
        has = self.doc is not None
        for panel in (self.strip, self.side, self.zoombar, self.infobar):
            panel.visible = has and not self._panels_hidden
        self.namebar.visible = has and not self._panels_hidden
        self.topbar.visible = not self._panels_hidden
        self.optionsbar.visible = has and not self._panels_hidden and bool(self.options_row.children())
        self.empty_card.visible = not has
        self.cropbar.visible = has and not self._panels_hidden and self.tool is self.crop_tool and self.crop_tool.rect is not None
        self.btn_save.enabled = self.btn_export.enabled = has
        doc = self.doc
        self.btn_undo.enabled = doc is not None and doc.can_undo
        self.btn_redo.enabled = doc is not None and doc.can_redo
        self.btn_revert.enabled = doc is not None and doc.can_undo
        self.btn_compare.enabled = has
        for b in self.transform_buttons:
            b.enabled = has
        for b in self.tool_buttons.values():
            b.enabled = has
        if doc is not None:
            name = Path(doc.path).name if doc.path else Path(doc.source_path).name if doc.source_path else doc.name
            self.title_label.set_text(("• " if doc.dirty else "") + name)
            meta = doc.metadata.describe()
            self.title_label.tooltip = (doc.path or doc.source_path or "") + (f"\n{meta}" if meta else "")
            size_text = f"{doc.width} × {doc.height}"
            self.subtitle_label.set_text(f"{size_text}  ·  {tr('editing mask')}" if doc.editing_mask else size_text)
            self.shell.set_title(f"{'• ' if doc.dirty else ''}{name} — {APP_NAME}")
        else:
            self.shell.set_title(APP_NAME)
        self._fill_tab()
        self.update_histogram()
        self.view_changed()
        self.view.layout_panels()
        if self._tab == 3 and self.doc is not None:
            selected = next((w for w in self.tab_body.children() if isinstance(w, ListItem) and w.selected), None)
            if selected is not None:
                self.side.scroll_to_widget(selected)
        for panel in self.view.panels:
            panel.invalidate()
        self.shell.commands_changed()

    def after_edit(self, fit: bool = False) -> None:
        if fit:
            self.view.layout_panels()
            self.view.fit()
        self.view.invalidate_frame()
        self.refresh_all()

    def update_histogram(self) -> None:
        """Recompute the RGB histogram from a small composite (cheap enough to run while dragging)."""
        if self.doc is None:
            self.histogram.set_data(None)
            return
        scale = min(1.0, 320 / max(self.doc.width, self.doc.height))
        small = self.doc.composite(scale, neutral=self.compare, overrides=self.preview_overrides or None)
        arr = np.asarray(small)
        alpha = arr[..., 3] > 0
        if not alpha.any():
            self.histogram.set_data(None)
            return
        data = np.stack([np.bincount(arr[..., c][alpha].ravel(), minlength=256)[:256] for c in range(3)]).astype(np.float32)
        nonzero = data[data > 0]
        peak = float(np.percentile(nonzero, 98)) if nonzero.size else 0.0
        self.histogram.set_data(np.clip(data / peak, 0, 1) if peak > 0 else data)

    def view_changed(self) -> None:
        z = self.view.zoom
        self.zoom_label.text = f"{z * 100:.0f}%" if z >= 0.1 else f"{z * 100:.1f}%"
        self.zoom_label.invalidate()
        if self.doc is not None and self.doc.source_path:
            self._zoom_memory[self.doc.source_path] = round(z, 4)

    def fit_width(self) -> None:
        if self.doc is None:
            return
        x0, _y0, x1, _y1 = self.view.content_area()
        self.view.set_zoom((x1 - x0 - px(56)) / self.doc.width)

    def fit_height(self) -> None:
        if self.doc is None:
            return
        _x0, y0, _x1, y1 = self.view.content_area()
        self.view.set_zoom((y1 - y0 - px(56)) / self.doc.height)

    def _remember_zoom(self) -> None:
        if self._zoom_memory != self.settings.get("zoom", {}):
            self.settings.set("zoom", dict(list(self._zoom_memory.items())[-50:]))

    # info pill -------------------------------------------------------------
    def _set_info(self, text: str, color=None, swatch=None) -> None:
        if color is None and self._notify_job is not None:
            self.shell.after_cancel(self._notify_job)
            self._notify_job = None
        self.info_label.set_color(color)
        show_swatch = swatch is not None
        changed = text != self.info_label.text or show_swatch != self.info_dot.visible
        self.info_dot.set_color(swatch)
        self.info_dot.visible = show_swatch
        if changed:
            self.info_label.set_text(text)
            self.infobar.layout(*self.view.viewport_size())
            self.view.schedule()

    def pointer_moved(self, x: float | None, y: float | None) -> None:
        if self.compare or (self._notify_job is not None and self._notify_error):
            return
        color = None
        if x is None or y is None or self.doc is None or not (0 <= x < self.doc.width and 0 <= y < self.doc.height):
            text = tr(self.tool.hint) if self.tool else ""
        else:
            color = self.doc.pick_color(x, y)
            text = f"{int(x)}, {int(y)}"
            if color is not None:
                r, g, b, a = color
                text += f"   ·   #{r:02X}{g:02X}{b:02X}   ·   {r} {g} {b}" + (f"   ·   α {a}" if a < 255 else "")
        self._set_info(text, swatch=color)

    def notify(self, message: str, error: bool = True, timeout: int = 3500) -> None:
        if self._notify_job is not None:
            self.shell.after_cancel(self._notify_job)
        self._notify_error = error
        self._set_info(message, PALETTE.danger if error else PALETTE.accent_hover)
        self._notify_job = self.shell.after(timeout, self._clear_notify)
        announce = getattr(self.shell, "announce", None)
        if announce is not None:
            announce(message)

    def _clear_notify(self) -> None:
        self._notify_job = None
        self._set_info(tr(self.tool.hint) if self.tool else "")

    def crop_changed(self) -> None:
        crop = self.crop_tool
        show = self.doc is not None and self.tool is crop and crop.rect is not None and not self._panels_hidden
        if show != self.cropbar.visible:
            self.cropbar.visible = show
            self.view.layout_panels()
        self.crop_size.set_text(crop.size_text())
        self.btn_crop_apply.enabled = crop.rect is not None

    def _crop_aspect_changed(self, index: int) -> None:
        self.crop_tool.set_aspect(list(ASPECTS.values())[index])

    # ----------------------------------------------------------- tools --
    def select_tool(self, tool_id: str) -> None:
        tool = self.tools[tool_id]
        if tool is self.tool:
            return
        if self.tool is not None:
            self.tool.deactivate()
        self.tool = tool
        for tid, btn in self.tool_buttons.items():
            btn.active = tid == tool_id
            btn.invalidate()
        self._set_info(tr(tool.hint))
        tool.activate()  # may replace the hint with a notice (e.g. "nothing to transform")
        self._show_tool_options()
        self.view.update_cursor()
        self.view.invalidate_frame()

    def _show_tool_options(self) -> None:
        self.options_row.clear()
        if self.tool is not None:
            widgets = self.tool.build_options()
            if widgets:
                self.options_row.add(*widgets)
        self.optionsbar.visible = self.doc is not None and not self._panels_hidden and bool(self.options_row.children())
        self.view.layout_panels()

    # colours -------------------------------------------------------------------
    def set_color(self, which: str, color: RGBA) -> None:
        setattr(self, which, tuple(color))
        self.swatches.invalidate()

    def pick_color(self, which: str) -> None:
        chosen = self.shell.ask_color(getattr(self, which))
        if chosen is not None:
            self.set_color(which, chosen)

    def swap_colors(self) -> None:
        self.foreground, self.background = self.background, self.foreground
        self.swatches.invalidate()

    def reset_colors(self) -> None:
        self.foreground, self.background = (0, 0, 0, 255), (255, 255, 255, 255)
        self.swatches.invalidate()

    # live previews ----------------------------------------------------------------
    def begin_preview(self, layer_id: int, image: Image.Image) -> None:
        """Show *image* in place of a layer (or, while editing its mask, in place of the mask) without committing."""
        if self.doc is not None and self.doc.editing_mask and layer_id == self.doc.active_layer.id:
            self.preview_mask_overrides = {layer_id: rgba_to_mask(image)}
            self.preview_overrides = {}
        else:
            self.preview_overrides = {layer_id: image}
            self.preview_mask_overrides = {}
        self.view.invalidate_frame()

    def end_preview(self) -> None:
        if self.preview_overrides or self.preview_mask_overrides:
            self.preview_overrides = {}
            self.preview_mask_overrides = {}
            self.view.invalidate_frame()

    # ------------------------------------------------------------- keys --
    def key_pressed(self, key: str, char: str, mods: frozenset[str]) -> bool:
        """Handle a key that no menu command claimed. ``key`` is a normalised name, ``mods`` ⊆ {mod, shift, alt}."""
        if key == "space":
            self.view.set_space(True)
            return True
        if key == "backslash" and "mod" not in mods and self.doc is not None and self.sheet is None:
            self.set_compare(True)
            return True
        if self.view.key(key, char, mods):
            return True
        if key == "Escape":
            if self._popover is not None:
                self.close_popover()
            elif self.sheet is not None:
                self.dismiss_modal()
            elif self.tool is not None:
                self.tool.cancel()
            return True
        if key in ("Return", "KP_Enter"):
            if self.sheet is not None and getattr(self.sheet, "default_action", None):
                self.sheet.default_action()  # type: ignore[attr-defined]
                return True
            return bool(self.tool is not None and self.tool.on_key(key, mods))
        if "mod" in mods:
            return self._run_shortcut(key, mods)
        if key in ("Delete", "BackSpace") and self.sheet is None and self.doc is not None and self.view.focused_widget is None:
            if "alt" in mods:
                self.fill("foreground")
            else:
                self.clear_selection()
            return True
        if key == "Tab":
            if self.sheet is not None:
                self._cycle_focus(backwards="shift" in mods)
            else:
                self.toggle_panels()
            return True
        if self.doc is not None and self.sheet is None and self.tool is not None and self.tool.on_key(key, mods):
            return True
        if self.doc is not None and self.sheet is None and char:
            if char == "x":
                self.swap_colors()
                return True
            if char == "d":
                self.reset_colors()
                return True
            for tool in self.tools.values():
                if tool.key and char == tool.key:
                    self.select_tool(tool.id)
                    return True
        return False

    def _cycle_focus(self, backwards: bool = False) -> None:
        """Move keyboard focus to the next (or previous) text field of the open sheet."""
        if self.sheet is None:
            return
        fields = [w for w in self.sheet.content.descendants() if w.focusable and w.visible and w.enabled]
        if not fields:
            return
        current = self.view.focused_widget
        index = fields.index(current) if current in fields else -1
        step = -1 if backwards else 1
        self.view.set_focus(fields[(index + step) % len(fields)])
        self.sheet.invalidate()

    def key_released(self, key: str) -> None:
        if key == "space":
            self.view.set_space(False)
        elif key == "backslash":
            self.set_compare(False)

    def _run_shortcut(self, key: str, mods: frozenset[str]) -> bool:
        """Fallback for shells without native menu shortcuts: match ``Mod+...`` command shortcuts."""
        names = {
            "plus": "+",
            "equal": "+",
            "minus": "-",
            "bracketleft": "[",
            "bracketright": "]",
            "apostrophe": "'",
            "backslash": "\\",
            "BackSpace": "Backspace",
        }
        base = names.get(key, key.upper() if len(key) == 1 else key)
        wanted = "Mod+" + ("Shift+" if "shift" in mods else "") + ("Alt+" if "alt" in mods else "") + base
        for cmd in self.commands:
            if cmd.shortcut and cmd.shortcut.upper() == wanted.upper() and cmd.enabled():
                cmd.run()
                return True
        return False

    def toggle_panels(self) -> None:
        self._panels_hidden = not self._panels_hidden
        self.refresh_all()

    def toggle_grid(self) -> None:
        self.view.show_grid = not self.view.show_grid
        self.settings.set("grid", self.view.show_grid)
        self.view.invalidate_frame()
        self.shell.commands_changed()

    def toggle_rulers(self) -> None:
        self.view.show_rulers = not self.view.show_rulers
        self.settings.set("rulers", self.view.show_rulers)
        self.view.invalidate_frame()
        self.shell.commands_changed()

    def toggle_snap(self) -> None:
        self.snap_enabled = not self.snap_enabled
        self.settings.set("snap", self.snap_enabled)
        self.shell.commands_changed()

    # ------------------------------------------------------------ sheets --
    def open_sheet(self, panel: Panel) -> None:
        self.dismiss_modal()
        self.sheet = panel
        self.view.add_panel(panel)

    def open_popover(self, name: str, items: Sequence[Widget], place: PlaceFn, width: int = 240, scrollable: bool = False) -> Panel:
        """A light menu-like panel; closes the previous popover but leaves an open sheet in place."""
        self.close_popover()
        panel = Panel(
            Column(list(items)),
            place,
            padding=8,
            radius=16,
            fit_height=not scrollable,
            modal=True,
            dismiss_on_outside=True,
            name=name,
            scrollable=scrollable,
        )
        self._popover = panel
        self.view.add_panel(panel)
        return panel

    def close_popover(self) -> None:
        if self._popover is not None:
            panel, self._popover = self._popover, None
            self.view.remove_panel(panel)

    def dismiss_modal(self) -> None:
        """Close the popover if one is open, otherwise the sheet."""
        if self._popover is not None:
            self.close_popover()
            return
        if self.sheet is not None:
            panel, self.sheet = self.sheet, None
            self.view.remove_panel(panel)
            on_dismiss = getattr(panel, "on_dismiss", None)
            if on_dismiss is not None:
                on_dismiss()
        if self.straighten_preview is not None or self.preview_overrides or self.preview_mask_overrides:
            self.straighten_preview = None
            self.preview_overrides = {}
            self.preview_mask_overrides = {}
            self.view.invalidate_frame()
            self.update_histogram()

    def menu_action(self, fn: Callable[[], object]) -> None:
        self.dismiss_modal()
        fn()

    def open_more_menu(self) -> None:
        if self.sheet is not None or self._popover is not None:
            self.dismiss_modal()
            return
        act = self.menu_action
        doc_items = [
            ListItem(tr("Straighten…"), lambda: act(lambda: sheets.straighten_sheet(self)), icon="straighten"),
            ListItem(tr("Resize…"), lambda: act(lambda: sheets.resize_sheet(self)), icon="resize"),
            ListItem(tr("Canvas Size…"), lambda: act(lambda: sheets.canvas_sheet(self)), icon="canvas"),
            ListItem(tr("Trim Transparent Edges"), lambda: act(self.trim), icon="trim"),
            Separator(4),
            ListItem(tr("Filters…"), lambda: act(lambda: sheets.filter_picker(self)), icon="filter"),
            ListItem(tr("Auto Enhance"), lambda: act(self.auto_enhance), icon="wand"),
            ListItem(tr("Apply This Look to Other Pictures…"), lambda: act(self.batch_export), icon="export"),
            Separator(4),
            ListItem(tr("Copy Picture"), lambda: act(self.copy_image), icon="copy"),
            ListItem(tr("Revert to Original"), lambda: act(self.revert), icon="revert"),
        ]
        items = doc_items + [
            Separator(4),
            ListItem(tr("Show Grid"), lambda: act(self.toggle_grid), checked=self.view.show_grid),
            ListItem(tr("Show Rulers"), lambda: act(self.toggle_rulers), checked=self.view.show_rulers),
            Separator(4),
            ListItem(tr("Keyboard Shortcuts"), lambda: act(lambda: sheets.shortcuts_sheet(self, SHORTCUTS)), icon="keyboard"),
            ListItem(tr("About {app}").format(app=APP_NAME), lambda: act(lambda: sheets.about_sheet(self)), icon="info"),
        ]
        if self.doc is None:
            for item in doc_items:
                item.enabled = False
        self.open_popover("menu", items, lambda vw, vh, pw, ph: (vw - px(GAP + 260), px(GAP + TOPBAR_H + 6), px(260), ph), width=260)

    def _zoom_to(self, zoom: float) -> None:
        self.menu_action(lambda: self.view.set_zoom(zoom))

    def _zoom_menu_toggle(self) -> None:
        if self._popover is not None:
            self.close_popover()
            return
        act = self.menu_action
        items: list[Widget] = [
            ListItem(f"{int(z * 100)}%", partial(self._zoom_to, z), checked=abs(self.view.zoom - z) < 1e-6)
            for z in (0.25, 0.5, 1.0, 2.0, 4.0)
        ]
        items.append(ListItem(tr("Fit to Window"), lambda: act(self.view.fit), icon="fit"))
        items.append(ListItem(tr("Fit Width"), lambda: act(self.fit_width)))
        items.append(ListItem(tr("Fit Height"), lambda: act(self.fit_height)))
        x0, y0, x1, y1 = self.zoombar.widget_rect_in_view(self.zoom_label)
        self.open_popover("zoommenu", items, lambda vw, vh, pw, ph: ((x0 + x1) // 2 - px(80), y0 - ph - px(8), px(160), ph), width=160)

    def open_text_sheet(self, position: tuple[float, float] | None = None) -> None:
        sheets.text_sheet(self, position=position)

    # ------------------------------------------------------------- file --
    def _fill_recent(self) -> None:
        self.recent_column.clear()
        recent = self.recent_files()
        if not recent:
            return
        self.recent_column.add(Spacer(12), Separator(2), Spacer(4), Label(tr("Recent"), style="caption", uppercase=True))
        for path in recent[:5]:
            self.recent_column.add(ListItem(Path(path).name, partial(self.open_file, path), icon="clock", subtitle=str(Path(path).parent)))

    def recent_files(self) -> list[str]:
        return self.settings.recent_files()

    def clear_recent(self) -> None:
        self.settings.clear_recent()
        self._fill_recent()
        self.refresh_all()

    def _confirm_discard(self) -> bool:
        if self.doc is None or not self.doc.dirty:
            return True
        answer = self.shell.ask_save_changes(self.doc.name)
        if answer is None:
            return False
        return self.save_project() if answer else True

    def set_document(self, doc: Document | None) -> None:
        if self.tool is not None:
            self.tool.cancel()
        self.dismiss_modal()
        self.preview_overrides = {}
        self.straighten_preview = None
        self._remember_zoom()
        self._thumb_cache.clear()
        self.doc = doc
        if doc is None:
            self.compare = False
            self.btn_compare.active = False
        self._show_tool_options()
        self.refresh_all()
        if doc is not None:
            remembered = self._zoom_memory.get(doc.source_path or "")  # read before fit() records the new zoom
            self.view.fit()
            if remembered and abs(remembered - self.view.zoom) > 1e-3:
                self.view.set_zoom(remembered)
        self.view.invalidate_frame()

    def open_file(self, path: str | None = None) -> None:
        if not self._confirm_discard():
            return
        path = path or self.shell.ask_open_file(OPEN_FILETYPES)
        if not path:
            return
        try:
            doc = pfio.open_document(path)
        except Exception as exc:  # unreadable / unsupported file
            self.shell.show_error(tr("Could not open picture"), f"{Path(path).name}\n\n{exc}")
            return
        self.settings.add_recent(path)
        self._fill_recent()
        self.set_document(doc)
        self.notify(tr("Opened {name}").format(name=Path(path).name), error=False, timeout=2500)

    def save_project(self, ask: bool = False) -> bool:
        """Save the layered document as a .pfp project (asks for a location the first time or when *ask*)."""
        if self.doc is None:
            return False
        path = self.doc.path
        if ask or not path:
            stem = Path(self.doc.source_path).stem if self.doc.source_path else self.doc.name
            path = self.shell.ask_save_file(f"{stem}{pfio.PROJECT_EXT}", PROJECT_FILETYPES)
            if not path:
                return False
            if Path(path).suffix.lower() != pfio.PROJECT_EXT:
                path += pfio.PROJECT_EXT
        try:
            pfio.save_project(self.doc, path)
        except Exception as exc:
            self.shell.show_error(tr("Could not save"), str(exc))
            return False
        self.settings.add_recent(path)
        self._fill_recent()
        self.clear_autosave()
        self.refresh_all()
        self.notify(tr("Project saved as {name}").format(name=Path(path).name), error=False, timeout=2500)
        return True

    def export_picture(self, ask_options: bool = True) -> bool:
        """Export the flattened picture; JPEG/WebP and opaque formats get an options sheet first."""
        if self.doc is None:
            return False
        stem = Path(self.doc.source_path).stem if self.doc.source_path else self.doc.name
        path = self.shell.ask_save_file(f"{stem}-edited.png", EXPORT_FILETYPES)
        if not path:
            return False
        if not Path(path).suffix:
            path += ".png"
        fmt, has_alpha = pfio.EXPORT_FORMATS.get(Path(path).suffix.lower(), ("PNG", True))
        has_metadata = bool(self.doc.metadata.describe()) and fmt in ("JPEG", "PNG", "WEBP", "TIFF")
        if ask_options and (fmt in ("JPEG", "WEBP") or not has_alpha or has_metadata):
            sheets.export_sheet(self, path, fmt, has_alpha)
            return True
        return self.write_image(
            path, self.settings.get("export_quality", 92), tuple(self.settings.get("export_background", (255, 255, 255)))
        )

    def write_image(self, path: str, quality: int, background: tuple, metadata: bool | None = None) -> bool:
        assert self.doc is not None
        if metadata is None:
            metadata = bool(self.settings.get("export_metadata", True))
        try:
            pfio.save_image(self.doc, path, quality=int(quality), background=background, metadata=metadata)
        except Exception as exc:
            self.shell.show_error(tr("Could not save"), str(exc))
            return False
        size = Path(path).stat().st_size
        human = f"{size / 1024:.0f} KB" if size < 1024 * 1024 else f"{size / 1024 / 1024:.1f} MB"
        self.notify(tr("Exported {name}").format(name=Path(path).name) + f"  ·  {human}", error=False, timeout=3000)
        return True

    def close_document(self) -> None:
        if self.doc is not None and self._confirm_discard():
            self.clear_autosave()
            self.set_document(None)

    def quit_app(self) -> None:
        if self._confirm_discard():
            self.clear_autosave()
            self.shutdown()
            self.shell.close_window()

    def shutdown(self) -> None:
        self.worker.stop()
        self._remember_zoom()
        for attr in ("_notify_job", "_autosave_job"):
            job = getattr(self, attr)
            if job is not None:
                self.shell.after_cancel(job)
                setattr(self, attr, None)
        self.view.shutdown()

    # ----------------------------------------------------------- clipboard --
    def copy_image(self) -> None:
        if self.doc is None:
            return
        image, _pos = self.doc.copy_selection(merged=True) or (self.doc.export_image("RGBA"), (0, 0))
        self.shell.copy_image(image)
        self.notify(tr("Picture copied to the clipboard"), error=False, timeout=2500)

    def paste_image(self) -> None:
        image = self.shell.paste_image()
        if image is None:
            self.notify(tr("The clipboard has no picture"))
            return
        if self.doc is not None:
            self.doc.paste(image, None if image.size != self.doc.size else (0, 0))
            self.after_edit()
            self.notify(tr("Pasted as a new layer"), error=False, timeout=2500)
            return
        self.set_document(Document.from_image(image, tr("Pasted picture")))
        self.notify(tr("Pasted a {w} × {h} picture").format(w=image.width, h=image.height), error=False, timeout=2500)

    # ------------------------------------------------------------ edits --
    def doc_op(self, fn: Callable[[Document], object], fit: bool = False) -> None:
        if self.doc is None:
            return
        try:
            fn(self.doc)
        except DocumentError as exc:
            self.notify(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - never let an edit take the session down
            self.report_error(exc)
            return
        self.after_edit(fit=fit)

    def report_error(self, exc: BaseException) -> None:
        """Show an unexpected error in the info pill and log the traceback (the app keeps running)."""
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        self.end_preview()
        if self.tool is not None:
            try:
                self.tool.cancel()
            except Exception:  # noqa: BLE001
                pass
        self.notify(tr("Something went wrong: {error}").format(error=f"{type(exc).__name__}: {exc}"), timeout=6000)

    _doc_op = doc_op

    def undo(self) -> None:
        if self.doc and self.doc.undo():
            if self.tool:
                self.tool.cancel()
            self.after_edit()

    def redo(self) -> None:
        if self.doc and self.doc.redo():
            if self.tool:
                self.tool.cancel()
            self.after_edit()

    def revert(self) -> None:
        self.doc_op(lambda d: d.revert(), fit=True)

    def go_to_history(self, index: int) -> None:
        if self.doc and index != self.doc.history.index:
            self.doc.go_to_history(index)
            self.after_edit()

    def restore_snapshot(self, index: int) -> None:
        if self.doc is None or not 0 <= index < len(self.doc.snapshots):
            return
        self.doc.restore_snapshot(index)
        self.after_edit(fit=True)

    def delete_snapshot(self, index: int) -> None:
        if self.doc is None or not 0 <= index < len(self.doc.snapshots):
            return
        self.doc.delete_snapshot(index)
        self.refresh_all()

    def clear_selection(self) -> None:
        if self.doc is None:
            return
        if self.doc.selection is None:
            self.notify(tr("Nothing selected - make a selection first"))
            return
        self.doc_op(lambda d: d.clear())

    def fill(self, which: str) -> None:
        color = self.foreground if which == "foreground" else self.background
        self.doc_op(lambda d: d.fill(color, label=tr("Fill")))

    def modify_selection(self, op: str) -> None:
        if self.doc is None or self.doc.selection is None:
            self.notify(tr("Nothing selected - make a selection first"))
            return
        titles = {
            "feather": tr("Feather selection"),
            "grow": tr("Grow selection"),
            "shrink": tr("Shrink selection"),
            "border": tr("Border selection"),
        }
        sheets.number_sheet(self, titles[op], tr("Pixels"), 5, 0, 500, lambda amount: self.doc_op(lambda d: d.modify_selection(op, amount)))

    def _adjust_live_value(self, key: str, value: float) -> None:
        self._adjust_live(key, int(value))

    def _adjust_live(self, key: str, value: int) -> None:
        if self.doc is None:
            return
        self.doc.set_adjustment(key, value)
        self.view.invalidate_frame()
        self.update_histogram()

    def _adjust_commit(self, key: str) -> None:
        if self.doc is None:
            return
        current = self.doc.history.current
        if current is not None and current.layers[self.doc.active_index].adjustments == self.doc.active_layer.adjustments:
            return
        self.doc.commit(f"{tr('Adjust')} {tr(SLIDERS[key].label)}")
        self.refresh_all()

    def reset_adjustments(self) -> None:
        if self.doc and not is_neutral(self.doc.active_layer.adjustments):
            self.doc_op(lambda d: d.reset_adjustments())

    def reset_group(self, group: str) -> None:
        if self.doc is None:
            return
        adj = dict(self.doc.active_layer.adjustments)
        changed = False
        for key, spec in SLIDERS.items():
            if spec.group == group and adj.get(key, spec.default) != spec.default:
                adj[key] = spec.default
                changed = True
        if changed:
            self.doc.set_adjustments(adj, commit=True, label=f"{tr('Reset')} {tr(group)}")
            self.after_edit()

    def auto_enhance(self) -> None:
        """Compute adjustments from the picture's histogram and colour cast and apply them to the active layer."""
        if self.doc is None:
            return
        adj = auto_enhance(self.doc.composite(min(1.0, 512 / max(self.doc.size)), neutral=True))
        if adj == self.doc.active_layer.adjustments:
            self.notify(tr("Auto enhance found nothing to change"), error=False)
            return
        self.doc.set_adjustments(adj, commit=True, label=tr("Auto Enhance"))
        self.after_edit()

    def apply_preset(self, name: str) -> None:
        if self.doc is None:
            return
        adj = self.presets.get(name)
        if adj == self.doc.active_layer.adjustments:
            return
        self.doc.set_adjustments(adj, commit=True, label=f"{tr('Preset')}: {name}" if name != "None" else tr("Reset Adjustments"))
        self.after_edit()

    def delete_preset(self, name: str) -> None:
        if name in self.presets.user:
            self.presets.delete(name)
            self.refresh_all()
            self.notify(tr("Deleted preset “{name}”").format(name=name), error=False, timeout=2500)

    # ------------------------------------------------------------ compare --
    def set_compare(self, on: bool) -> None:
        on = bool(on and self.doc is not None)
        if on == self.compare:
            return
        self.compare = on
        self.btn_compare.active = on
        self.btn_compare.invalidate()
        self.update_histogram()
        self._set_info(tr("Original (before adjustments)") if on else (tr(self.tool.hint) if self.tool else ""))
        self.view.invalidate_frame()
        self.shell.commands_changed()

    def toggle_compare(self) -> None:
        self.set_compare(not self.compare)

    # ---------------------------------------------------------------- trim --
    def trim(self) -> None:
        if self.doc is None:
            return
        if self.doc.trim():
            self.after_edit(fit=True)
        else:
            self.notify(tr("Nothing to trim - the picture has no transparent edges"), error=False)

    # ------------------------------------------------------------- filters --
    def run_filter(self, spec: FilterSpec) -> None:
        if self.doc is None:
            return
        if self.doc.active_layer.locked:
            self.notify(tr("The layer is locked"))
            return
        if not spec.params:
            self.doc_op(lambda d: d.apply_filter(spec.name))
            return
        sheets.filter_sheet(self, spec)

    def preview_filter_fn(self, spec: FilterSpec, params: dict) -> Callable[[Image.Image], Image.Image]:
        """Filter function for previews: large pictures are only filtered where they are visible."""
        return self.preview_region_fn(lambda im: spec.apply(im, **params))

    def preview_region_fn(self, fn: Callable[[Image.Image], Image.Image]) -> Callable[[Image.Image], Image.Image]:
        """Wrap a pixel operation so previews of large pictures only process the visible region."""
        assert self.doc is not None
        box = self.view.visible_doc_box()
        if box is None or self.doc.width * self.doc.height <= 1_500_000:
            return fn
        margin = 32
        x0, y0 = max(0, int(box[0]) - margin), max(0, int(box[1]) - margin)
        x1, y1 = min(self.doc.width, int(box[2]) + margin + 1), min(self.doc.height, int(box[3]) + margin + 1)

        def region_fn(im: Image.Image) -> Image.Image:
            patch = fn(im.crop((x0, y0, x1, y1)))
            out = im.copy()
            out.paste(patch, (x0, y0))
            return out

        return region_fn

    # ---------------------------------------------------------- batch export --
    def batch_export(self) -> None:
        if self.doc is None:
            return
        adjustments = dict(self.doc.active_layer.adjustments)
        sources = self.shell.ask_open_files(OPEN_FILETYPES)
        if not sources:
            return
        out_dir = self.shell.ask_directory(tr("Choose the folder for the exported pictures"))
        if not out_dir:
            return
        done, failed = self.export_look(sources, out_dir, adjustments)
        if failed:
            names = ", ".join(Path(f).name for f in failed[:3])
            self.notify(tr("Exported {n} picture(s); {m} failed: {names}").format(n=done, m=len(failed), names=names))
        else:
            self.notify(tr("Exported {n} picture(s) to {folder}").format(n=done, folder=Path(out_dir).name), error=False, timeout=4000)

    def export_look(self, sources: list[str], out_dir: str, adjustments: dict[str, int]) -> tuple[int, list[str]]:
        quality = int(self.settings.get("export_quality", 92))
        background = tuple(self.settings.get("export_background", (255, 255, 255)))
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        done, failed = 0, []
        for index, src in enumerate(sources, 1):
            self._set_info(tr("Exporting {i} / {n}:  {name}").format(i=index, n=len(sources), name=Path(src).name))
            self.view.render()
            try:
                doc = pfio.open_document(src)
                for layer_index in range(len(doc.layers)):
                    doc.set_adjustments(adjustments, layer_index)
                suffix = Path(src).suffix.lower()
                if suffix not in pfio.EXPORT_FORMATS:
                    suffix = ".png"
                pfio.save_image(doc, out / f"{Path(src).stem}-edited{suffix}", quality=quality, background=background)
                done += 1
            except Exception:
                failed.append(src)
        return done, failed

    # ------------------------------------------------------------- overlay --
    def paint_overlay(self, frame: Image.Image, view: Surface) -> Image.Image:
        if self.doc is None:
            return frame
        if self.mask_overlay and self.doc.editing_mask and not self.compare:
            frame = self._paint_mask_overlay(frame, view)
        if self.straighten_preview is not None:
            frame = sheets.paint_straighten_overlay(self, frame, view)
        return frame

    def _paint_mask_overlay(self, frame: Image.Image, view: Surface) -> Image.Image:
        """Rubylith: tint the areas the active layer's mask hides."""
        assert self.doc is not None
        ix0, iy0, ix1, iy1 = view.image_rect()
        vw, vh = frame.size
        vx0, vy0, vx1, vy1 = max(0, ix0), max(0, iy0), min(vw, ix1), min(vh, iy1)
        if vx1 <= vx0 or vy1 <= vy0:
            return frame
        override = self.preview_mask_overrides.get(self.doc.active_layer.id)
        mask = self.doc.render_mask(view.zoom, (vx0 - ix0, vy0 - iy0, vx1 - ix0, vy1 - iy0), override)
        if mask is None:
            return frame
        alpha = ImageChops.invert(mask).point(lambda v: v * 0.55)
        tint = Image.new("RGBA", mask.size, (*MASK_OVERLAY_COLOR, 0))
        tint.putalpha(alpha)
        region = frame.crop((vx0, vy0, vx1, vy1)).convert("RGBA")
        region.alpha_composite(tint)
        frame.paste(region.convert("RGB"), (vx0, vy0))
        return frame
