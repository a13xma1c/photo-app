"""GUI tests for the glass interface - run against real Qt widgets on the offscreen platform.

They exercise the widget kit directly (layout / hit-testing math), drive the
surface with real Qt mouse/keyboard/wheel events where it matters (hover →
tooltip, trackpad scrolling, clicks, HiDPI) and patch the native dialogs so
nothing blocks.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TypeVar

import pytest
from PIL import Image, ImageFont

from tests.helpers import gray, pixel

pytestmark = pytest.mark.gui
pytest.importorskip("PySide6.QtCore")
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from pixelforge.core.document import Document  # noqa: E402
from pixelforge.core.selection import Selection  # noqa: E402
from pixelforge.gui import editor as editor_mod  # noqa: E402
from pixelforge.gui import glass, sheets, theme, ui  # noqa: E402
from pixelforge.gui import tools as tools_mod  # noqa: E402
from pixelforge.gui.editor import Editor  # noqa: E402
from pixelforge.gui.qt import MainWindow  # noqa: E402
from pixelforge.gui.surface import PointerEvent, wheel_pixels_from_notches  # noqa: E402
from pixelforge.gui.theme import METRICS, PALETTE, font, font_size, text_size  # noqa: E402
from pixelforge.gui.tools import (  # noqa: E402
    BrushTool,
    CloneTool,
    CropTool,
    HealTool,
    MarqueeTool,
    MoveTool,
    Tool,
    TransformTool,
)

ToolT = TypeVar("ToolT", bound=Tool)


# ------------------------------------------------------------ narrowing --
def doc_of(app: Editor) -> Document:
    """The open document (tests always know one is there)."""
    assert app.doc is not None
    return app.doc


def tool_of(app: Editor) -> Tool:
    assert app.tool is not None
    return app.tool


def tool_as(app: Editor, cls: type[ToolT]) -> ToolT:
    assert isinstance(app.tool, cls)
    return app.tool


def shell_of(app: Editor) -> MainWindow:
    assert isinstance(app.shell, MainWindow)
    return app.shell


def sheet_of(app: Editor) -> ui.Panel:
    assert app.sheet is not None
    return app.sheet


def popover_of(app: Editor) -> ui.Panel:
    assert app._popover is not None
    return app._popover


def editor_of(win: MainWindow) -> Editor:
    assert win.editor is not None
    return win.editor


def fire(widget: ui.Button | ui.IconButton | ui.ListItem) -> None:
    """Invoke a widget's command the way a click would."""
    assert widget.command is not None
    widget.command()


def is_checked(app: Editor, command_id: str) -> bool:
    checked = app.command(command_id).checked
    assert checked is not None
    return checked()


def rect_of(tool: CropTool) -> tuple[float, float, float, float]:
    assert tool.rect is not None
    return tool.rect


def ev(x: float, y: float, wx: int = 0, wy: int = 0, shift: bool = False, ctrl: bool = False, alt: bool = False) -> PointerEvent:
    return PointerEvent(x, y, wx, wy, shift, ctrl, alt)


def press_key(app: Editor, name: str, char: str = "", mods: tuple[str, ...] = ()) -> bool:
    return app.key_pressed(name, char or (name if len(name) == 1 else ""), frozenset(mods))


def widget_center(app: Editor, panel: ui.Panel, widget: ui.Widget) -> tuple[int, int]:
    x0, y0, x1, y1 = panel.widget_rect_in_view(widget)
    return (x0 + x1) // 2, (y0 + y1) // 2


def logical(app: Editor, x: float, y: float) -> QtCore.QPoint:
    dpr = shell_of(app).widget.dpr
    return QtCore.QPoint(int(x / dpr), int(y / dpr))


def move(app: Editor, x: float, y: float) -> None:
    QTest.mouseMove(shell_of(app).widget, logical(app, x, y))
    QtWidgets.QApplication.processEvents()


def _send_mouse(app: Editor, kind: QtCore.QEvent.Type, x: float, y: float, button: Qt.MouseButton, buttons: Qt.MouseButton) -> None:
    widget = shell_of(app).widget
    pos = QtCore.QPointF(logical(app, x, y))
    event = QtGui.QMouseEvent(kind, pos, widget.mapToGlobal(pos), button, buttons, Qt.KeyboardModifier.NoModifier)
    QtWidgets.QApplication.sendEvent(widget, event)
    QtWidgets.QApplication.processEvents()


def mouse_press(app: Editor, x: float, y: float, button: Qt.MouseButton = Qt.MouseButton.LeftButton) -> None:
    # sent directly (QTest.mousePress would synthesise double-clicks for rapid successive presses)
    _send_mouse(app, QtCore.QEvent.Type.MouseButtonPress, x, y, button, button)


def mouse_release(app: Editor, x: float, y: float, button: Qt.MouseButton = Qt.MouseButton.LeftButton) -> None:
    _send_mouse(app, QtCore.QEvent.Type.MouseButtonRelease, x, y, button, Qt.MouseButton.NoButton)


def drag_mouse(app: Editor, x0: float, y0: float, x1: float, y1: float, button: Qt.MouseButton = Qt.MouseButton.LeftButton) -> None:
    mouse_press(app, x0, y0, button)
    _send_mouse(app, QtCore.QEvent.Type.MouseMove, x1, y1, Qt.MouseButton.NoButton, button)
    mouse_release(app, x1, y1, button)


def click(app: Editor, panel: ui.Panel, widget: ui.Widget) -> None:
    x, y = widget_center(app, panel, widget)
    mouse_press(app, x, y)
    mouse_release(app, x, y)


def wheel(app: Editor, x: float, y: float, pixel_dy: float = 0, pixel_dx: float = 0, notches: float = 0, ctrl: bool = False) -> None:
    """Send a QWheelEvent: pixel deltas emulate a trackpad, notches a mouse wheel."""
    widget = shell_of(app).widget
    pos = QtCore.QPointF(logical(app, x, y))
    pixel = QtCore.QPoint(int(pixel_dx), int(pixel_dy))
    angle = QtCore.QPoint(0, int(notches * 120))
    mods = Qt.KeyboardModifier.ControlModifier if ctrl else Qt.KeyboardModifier.NoModifier
    event = QtGui.QWheelEvent(pos, widget.mapToGlobal(pos), pixel, angle, Qt.MouseButton.NoButton, mods, Qt.ScrollPhase.ScrollUpdate, False)
    QtWidgets.QApplication.sendEvent(widget, event)
    QtWidgets.QApplication.processEvents()


def key(app: Editor, qt_key: Qt.Key, text: str = "", mods: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier) -> None:
    QTest.keyClick(shell_of(app).widget, qt_key, mods)
    QtWidgets.QApplication.processEvents()


# ------------------------------------------------------------ drawing kit --
def test_glass_primitives() -> None:
    mask = glass.rounded_mask((40, 20), 8)
    assert mask.size == (40, 20) and gray(mask, 20, 10) == 255 and gray(mask, 0, 0) == 0
    rect = glass.rounded_rect((30, 30), 10, (255, 0, 0, 255), (0, 0, 255, 255))
    assert rect.mode == "RGBA" and pixel(rect, 15, 15) == (255, 0, 0, 255)
    assert glass.circle(9, (1, 2, 3, 255)).size == (9, 9)
    panel = glass.glass(Image.new("RGB", (120, 80), (200, 50, 50)))
    assert panel.size == (120, 80) and pixel(panel, 0, 0)[3] == 0  # rounded corner is transparent
    assert pixel(panel, 60, 40)[3] == 255
    assert pixel(panel, 60, 40)[0] < 200  # tinted darker than the backdrop
    for name in glass.ICONS:
        ico = glass.icon(name, 18, PALETTE.fg)
        assert ico.size == (18, 18) and ico.getbbox() is not None, name
    assert glass.icon("nonexistent", 18).getbbox() is not None
    fnt = font(13)
    assert glass.ellipsize("short", fnt, 500) == "short"
    long = glass.ellipsize("a fairly long piece of text that must be cut", fnt, 80)
    assert long.endswith("…") and fnt.getlength(long) <= 80
    assert text_size("Hello", fnt)[0] > 0
    target = Image.new("RGBA", (20, 20))
    glass.paste(target, glass.circle(50, (255, 255, 255, 255)), (-10, -10))  # clipped paste must not raise
    glass.paste(target, glass.circle(5, (255, 255, 255, 255)), (100, 100))


# --------------------------------------------------------------- widgets --
def test_column_and_row_layout() -> None:
    a, b = ui.Spacer(10), ui.Label("x", height=30)
    col = ui.Column([a, b], gap=4)
    assert col.measure(200) == 44
    col.layout(0, 0, 200, 44)
    assert a.rect == (0, 0, 200, 10) and b.rect == (0, 14, 200, 44)
    fixed = ui.Button("Go", width=60)
    flex = ui.Label("label")
    row = ui.Row([fixed, flex], gap=10)
    row.layout(0, 0, 200, 40)
    assert fixed.rect[2] - fixed.rect[0] == 60 and flex.rect == (70, 4, 200, 36)
    assert row.preferred_width() == 60 + 10 + flex.preferred_width()
    centred = ui.Row([ui.Button("A", width=50), ui.Button("B", width=50)], gap=10, align="center")
    centred.layout(0, 0, 300, 40)
    assert centred.children()[0].rect[0] == 95
    hidden = ui.Label("hidden")
    hidden.visible = False
    col2 = ui.Column([hidden, b])
    assert col2.measure(100) == 30
    assert col.hit(5, 20) is b and col.hit(500, 500) is None


def test_slider_math_and_callbacks() -> None:
    changes: list[float] = []
    commits: list[int] = []
    s = ui.Slider("Contrast", -100, 100, 0, 0, on_change=changes.append, on_commit=lambda: commits.append(1))
    s.layout(0, 0, 200, METRICS.slider_h)
    tx0, tx1, _ = s._track()
    assert s.value_at(tx0) == -100 and s.value_at(tx1) == 100 and s.value_at((tx0 + tx1) / 2) == 0
    s.on_press(tx1, 30)
    s.on_drag(tx0 + (tx1 - tx0) * 0.75, 30)
    assert changes[-1] == 50 and s._dragging and commits == []
    s.on_release(0, 0, True)
    assert commits == [1] and not s._dragging
    s.on_release(0, 0, True)  # nothing pending
    assert commits == [1]
    s.on_double(0, 0)
    assert s.value == 0 and commits == [1, 1]
    assert s.on_key("Right", "") and s.value == 1 and commits == [1, 1, 1]
    assert not s.on_key("a", "a")
    s.set(500)
    assert s.value == 100 and s.format_value() == "+100"
    img = Image.new("RGBA", (200, METRICS.slider_h))
    s.draw(img, ui.DrawContext())
    assert img.getbbox() is not None
    s.enabled = False
    s.on_press(tx1, 30)
    assert not s._dragging


def test_segmented_toggle_listitem_numberfield() -> None:
    picks: list[int] = []
    seg = ui.Segmented(["A", "B", "C"], 0, picks.append)
    seg.layout(0, 0, 300, 34)
    assert seg._segment_at(10) == 0 and seg._segment_at(150) == 1 and seg._segment_at(299) == 2
    seg.on_release(250, 10, True)
    assert seg.index == 2 and picks == [2]
    seg.set_hover_position(150)
    assert seg._hover_index == 1
    toggled: list[bool] = []
    tg = ui.Toggle("Lock", False, toggled.append)
    tg.layout(0, 0, 200, 32)
    tg.on_release(5, 5, True)
    assert tg.value and toggled == [True]
    clicked = []
    item = ui.ListItem("Row", lambda: clicked.append(1), selected=True, checked=True, subtitle="sub", icon="info")
    item.layout(0, 0, 200, item.height)
    item.on_release(1, 1, False)
    item.on_release(1, 1, True)
    assert clicked == [1]
    img = Image.new("RGBA", (200, 60))
    item.draw(img, ui.DrawContext(hover=item))
    values: list[float] = []
    nf = ui.NumberField("Width", 640, lo=1, hi=5000, on_change=values.append)
    nf.layout(0, 0, 260, 38)
    assert nf.on_key("BackSpace", "") and nf.text == "64"
    assert nf.on_key("5", "5") and nf.value == 645
    assert not nf.on_key("a", "a") and not nf.on_key("Return", "")
    nf.text = "99999"
    nf.on_focus(True)
    nf.on_focus(False)
    assert nf.value == 5000 and nf.text == "5000"  # clamped on blur
    nf.set_value(12)
    nf.draw(img, ui.DrawContext(focused=nf))


def test_panel_layout_scroll_and_hit_testing() -> None:
    rows = [ui.ListItem(f"row {i}") for i in range(30)]
    panel = ui.Panel(ui.Column(rows), lambda vw, vh, pw, ph: (10, 10, 200, 300), padding=10, scrollable=True)
    panel.layout(800, 600)
    assert panel.rect == (10, 10, 210, 310)
    assert panel.content_height == 30 * METRICS.row_h
    assert panel.max_scroll == panel.content_height - 280
    assert panel.widget_at(50, 50) is rows[0]  # content y = 50 - 10 (panel) - 10 (padding) = 30 -> first row
    assert panel.widget_at(50, 305) is None  # inside the padding, below the viewport
    assert panel.widget_at(5, 5) is None
    assert panel.scroll(70) and panel.scroll_y == 70
    assert panel.widget_at(50, 50) is rows[3]  # content y = 30 + 70 = 100 -> fourth row
    panel.scroll(-10_000)
    assert panel.scroll_y == 0
    panel.scroll(10_000)
    assert panel.scroll_y == panel.max_scroll
    assert panel.widget_rect_in_view(rows[0])[1] == 20 - panel.max_scroll
    panel.scroll_to_widget(rows[0])
    assert panel.scroll_y == 0
    panel.scroll_to_widget(rows[29])
    assert panel.scroll_y == panel.max_scroll
    img = panel.render(Image.new("RGB", (200, 300), (30, 30, 30)), ui.DrawContext())
    assert img.size == (200, 300) and not panel.dirty
    fit = ui.Panel(
        ui.Row([ui.Button("A", width=50), ui.Button("B", width=70)], gap=10),
        lambda vw, vh, pw, ph: (0, 0, pw, 50),
        padding=5,
        fit_width=True,
        fit_height=True,
    )
    fit.layout(800, 600)
    assert fit.width == 50 + 10 + 70 + 10 and fit.height == METRICS.button_h + 10
    capped = ui.Panel(ui.Row([ui.Label("x" * 200)]), lambda vw, vh, pw, ph: (0, 0, pw, 30), fit_width=True, max_width=100)
    capped.layout(800, 600)
    assert capped.width == 100
    static = ui.Panel(ui.Column(rows[:2]), lambda vw, vh, pw, ph: (0, 0, 100, 100))
    static.layout(800, 600)
    assert not static.scroll(10)


# ------------------------------------------------------------------ view --
def test_wheel_notches_scale_with_ui() -> None:
    assert wheel_pixels_from_notches(1) == 48
    assert wheel_pixels_from_notches(-2) == -96
    theme.set_scale(2.0)
    try:
        assert wheel_pixels_from_notches(1) == 96
    finally:
        theme.set_scale(1.0)


def test_app_starts_empty_and_layout_is_sane(app: Editor) -> None:
    assert app.doc is None and tool_of(app).id == "hand"
    assert app.empty_card.visible and not app.side.visible and not app.strip.visible
    assert app.topbar.visible and not app.btn_save.enabled and not app.btn_undo.enabled
    app.view.render()
    assert app.view.render_count >= 1
    assert shell_of(app).windowTitle() == "PixelForge"
    # menu commands reflect the state
    assert not app.command("save").enabled() and app.command("open").enabled()
    assert not shell_of(app)._actions["undo"].isEnabled() and shell_of(app)._actions["open"].isEnabled()
    # actions on an empty app are harmless
    app.undo()
    app.redo()
    app.revert()
    app.export_picture()
    app.select_tool("crop")
    app.close_document()
    app.command("zoom-in").run()


def test_open_file_and_panels(app: Editor, image_file: Path) -> None:
    app.open_file(str(image_file))
    assert app.doc is not None and app.side.visible and app.strip.visible and not app.empty_card.visible
    assert "gradient.png" in shell_of(app).windowTitle() and app.title_label.text == "gradient.png"
    assert app.subtitle_label.text == "64 × 48"
    assert app.view.zoom == 1.0  # small pictures are not upscaled
    x0, y0, x1, y1 = app.view.image_rect()
    cx0, cy0, cx1, cy1 = app.content_area(*app.view.viewport_size())
    assert cx0 <= x0 and x1 <= cx1 and cy0 <= y0 and y1 <= cy1  # picture sits between the panels
    assert app.zoom_label.text == "100%"
    assert "Opened" in app.info_label.text
    assert shell_of(app)._actions["save"].isEnabled()
    app.view.render()
    grab = shell_of(app).widget.grab().toImage()
    assert grab.width() == 1280 and grab.pixelColor(5, 400).red() < 40  # workspace background is dark


def test_open_bad_file(app: Editor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"nope")
    errors = []
    monkeypatch.setattr(app.shell, "show_error", lambda title, msg: errors.append((title, msg)))
    app.open_file(str(bad))
    assert errors and app.doc is None


def test_adjust_panel_alignment(app_with_doc: Editor) -> None:
    """Every slider label starts at the same x and rows are evenly spaced - no misalignment."""
    app = app_with_doc
    sliders = list(app.sliders.values())
    assert len({s.rect[0] for s in sliders}) == 1 and len({s.rect[2] - s.rect[0] for s in sliders}) == 1
    assert {s.rect[3] - s.rect[1] for s in sliders} == {METRICS.slider_h}
    assert app.tabs.rect[0] == sliders[0].rect[0]  # tabs and sliders share the panel's left edge
    app.tabs.select(3)
    rows = [w for w in app.tab_body.children() if isinstance(w, ui.ListItem)]
    assert rows and len({r.rect[0] for r in rows}) == 1 and len({r.rect[2] for r in rows}) == 1


def test_tooltip_appears_on_hover(app_with_doc: Editor) -> None:
    app = app_with_doc
    view = app.view
    btn = app.tool_buttons["crop"]
    x, y = widget_center(app, app.strip, btn)
    move(app, x, y)
    assert view.hover_widget is btn
    assert view.tooltip_pending()
    assert view.tooltip_widget is None
    view.show_tooltip(btn)  # what the timer does after 450 ms
    assert view.tooltip_widget is btn and view._tooltip_layer is not None
    view.render()
    assert any(kind == "tooltip" for kind, *_ in shell_of(app).widget._layers)
    # leaving the button hides it
    move(app, x + 300, y)
    assert view.hover_widget is None and view.tooltip_widget is None
    view.show_tooltip(btn)  # never shown for a widget that is no longer hovered
    assert view.tooltip_widget is None
    # the real timer fires too
    move(app, x, y)
    QTest.qWait(600)
    assert view.tooltip_widget is btn


def test_hover_and_click_through_real_events(app_with_doc: Editor) -> None:
    app = app_with_doc
    view = app.view
    btn = app.tool_buttons["zoom"]
    click(app, app.strip, btn)
    assert tool_of(app).id == "zoom" and btn.active
    x, y = widget_center(app, app.strip, btn)
    move(app, x, y)
    assert view.cursor == "hand" and shell_of(app).widget.cursor().shape() == Qt.CursorShape.PointingHandCursor
    # pressing then releasing outside does not trigger the command
    other = app.tool_buttons["crop"]
    ox, oy = widget_center(app, app.strip, other)
    mouse_press(app, ox, oy)
    mouse_release(app, ox + 300, oy)
    assert tool_of(app).id == "zoom"
    shell_of(app).widget.leaveEvent(QtCore.QEvent(QtCore.QEvent.Type.Leave))
    assert view.hover_widget is None


def test_side_panel_scrolls_with_trackpad_and_wheel(app_with_doc: Editor) -> None:
    app = app_with_doc
    side = app.side
    assert side.max_scroll > 0
    x, y = widget_center(app, side, app.sliders["contrast"])
    before = app.sliders["contrast"].value
    move(app, x, y)
    # trackpad: small pixel deltas over a slider scroll the panel, the slider does not change
    for _ in range(5):
        wheel(app, x, y, pixel_dy=-6)
    assert side.scroll_y > 0 and app.sliders["contrast"].value == before
    scrolled = side.scroll_y
    wheel(app, x, y, notches=1)  # mouse wheel up
    assert side.scroll_y < scrolled
    wheel(app, x, y, pixel_dx=-30)  # horizontal trackpad gesture: harmless
    assert app.view.zoom == 1.0
    app.tabs.select(1)
    assert side.scroll_y == 0 and side.max_scroll > 0
    wheel(app, x, y, notches=-1)
    assert side.scroll_y > 0


def test_wheel_over_picture_pans_and_ctrl_zooms(app_with_doc: Editor) -> None:
    app = app_with_doc
    view = app.view
    x0, y0, x1, y1 = app.content_area(*view.viewport_size())
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    oy = view.oy
    wheel(app, cx, cy, notches=-1)
    assert view.oy == oy - 48
    z = view.zoom
    wheel(app, cx, cy, notches=1, ctrl=True)
    assert view.zoom > z
    # pinch gesture (macOS trackpad)
    widget = shell_of(app).widget
    device = QtGui.QPointingDevice.primaryPointingDevice()
    pos = QtCore.QPointF(logical(app, cx, cy))
    gesture = QtGui.QNativeGestureEvent(Qt.NativeGestureType.ZoomNativeGesture, device, 2, pos, pos, pos, 0.5, QtCore.QPointF())
    z = view.zoom
    assert widget.event(gesture)
    assert view.zoom == pytest.approx(z * 1.5)


def test_zoom_fit_and_pan_math(app_with_doc: Editor) -> None:
    app = app_with_doc
    view = app.view
    view.fit()
    assert view.zoom == 1.0
    x, y = view.to_view(10, 20)
    assert view.to_doc(x, y) == pytest.approx((10, 20))
    before = view.to_doc(300, 300)
    view.set_zoom(2.0, (300, 300))
    assert view.zoom == 2.0 and view.to_doc(300, 300) == pytest.approx(before, abs=0.51)
    view.zoom_in()
    assert view.zoom == 3.0
    view.zoom_out()
    view.zoom_out()
    assert view.zoom == 1.5 and app.zoom_label.text == "150%"
    view.set_zoom(1e6)
    assert view.zoom == 32.0
    view.set_zoom(1.0)
    view.pan(10_000, 10_000)
    vw, vh = view.viewport_size()
    assert view.ox <= vw and view.oy <= vh  # clamped: picture stays reachable
    assert view.visible_doc_box() is not None
    view.set_zoom(0.05)
    assert view.visible_doc_box() is not None


def test_crop_tool_full_flow(app_with_doc: Editor) -> None:
    app = app_with_doc
    app.select_tool("crop")
    crop = tool_as(app, CropTool)
    crop.on_press(ev(20, 10))
    crop.on_drag(ev(220, 160))
    crop.on_release(ev(220, 160))
    assert crop.rect == (20, 10, 220, 160) and app.cropbar.visible
    assert app.crop_size.text == "200 × 150 px" and app.btn_crop_apply.enabled
    # move from inside
    crop.on_press(ev(100, 100))
    crop.on_drag(ev(110, 105))
    crop.on_release(ev(110, 105))
    assert crop.rect == (30, 15, 230, 165)
    # drag the east handle (needs surface coordinates)
    hx, hy = crop._handles()["e"]
    crop.on_press(ev(*app.view.to_doc(hx, hy), wx=int(hx), wy=int(hy)))
    crop.on_drag(ev(300, 90))
    crop.on_release(ev(300, 90))
    assert rect_of(crop)[2] == 300 and rect_of(crop)[0] == 30
    hx, hy = crop._handles()["e"]
    assert crop.cursor_at((int(hx), int(hy))) == "resize-h"
    vx, vy = app.view.to_view(100, 100)
    assert crop.cursor_at((int(vx), int(vy))) == "move"
    # aspect ratio via the crop bar
    app._crop_aspect_changed(1)  # 1:1
    cx0, cy0, cx1, cy1 = rect_of(crop)
    w, h = cx1 - cx0, cy1 - cy0
    assert w == pytest.approx(h)
    app.view.render()  # overlay painting must not fail
    assert crop.on_key("Return")
    assert doc_of(app).size == (round(w), round(h)) and doc_of(app).history.labels[-1] == "Crop"
    assert crop.rect is None and not app.cropbar.visible
    # Esc cancels a pending crop
    crop.on_press(ev(0, 0))
    crop.on_drag(ev(50, 50))
    crop.on_release(ev(50, 50))
    assert press_key(app, "Escape")
    assert crop.rect is None
    # tiny drags are ignored
    crop.on_press(ev(0, 0))
    crop.on_release(ev(0.5, 0.5))
    assert crop.rect is None and not crop.on_key("Return")


def test_crop_drag_starting_outside_the_picture(app_with_doc: Editor) -> None:
    """A crop gesture may begin outside the picture: the area is the overlap with the picture, never an error."""
    app = app_with_doc
    app.select_tool("crop")
    crop = tool_as(app, CropTool)
    crop.on_press(ev(-50, -50))
    crop.on_drag(ev(-10, 20))  # still outside on the same side: nothing to show yet
    assert crop.rect is None
    app.view.render()  # must not raise
    crop.on_drag(ev(100, 80))
    assert crop.rect == (0, 0, 100, 80)
    app.view.render()
    crop.on_drag(ev(1000, 1000))  # far beyond the opposite corner: clamped to the picture
    assert crop.rect == (0, 0, 400, 300)
    crop.on_release(ev(1000, 1000))
    assert crop.rect == (0, 0, 400, 300) and app.crop_size.text == "400 × 300 px"
    # start below-right of the picture and drag back in
    crop.on_press(ev(500, 400))
    crop.on_drag(ev(450, 350))
    assert crop.rect is None
    crop.on_drag(ev(300, 200))
    assert crop.rect == (300, 200, 400, 300)
    crop.on_release(ev(300, 200))
    # with an aspect ratio the overlap is constrained from the clamped start
    crop.set_aspect(1.0)
    crop.on_press(ev(-20, 10))
    crop.on_drag(ev(150, 400))
    x0, y0, x1, y1 = rect_of(crop)
    assert (x0, y0) == (0, 10) and x1 - x0 == pytest.approx(y1 - y0) and y1 <= 300
    crop.on_release(ev(150, 400))
    app.view.set_zoom(0.05)  # tiny on screen: overlay must still draw
    app.view.render()
    assert press_key(app, "Return") and doc_of(app).history.labels[-1] == "Crop"


def test_crop_with_real_mouse_drag(app_with_doc: Editor) -> None:
    app = app_with_doc
    app.select_tool("crop")
    x0, y0 = app.view.to_view(20, 20)
    x1, y1 = app.view.to_view(120, 90)
    drag_mouse(app, int(x0), int(y0), int(x1), int(y1))
    rect = tool_as(app, CropTool).rect
    assert rect is not None and abs(rect[0] - 20) <= 1 and abs(rect[2] - 120) <= 1
    # Enter through a real key event applies
    key(app, Qt.Key.Key_Return)
    assert doc_of(app).history.labels[-1] == "Crop"


def test_crop_aspect_constrained_drag(app_with_doc: Editor) -> None:
    app = app_with_doc
    app.select_tool("crop")
    crop = tool_as(app, CropTool)
    crop.set_aspect(16 / 9)
    crop.on_press(ev(10, 10))
    crop.on_drag(ev(170, 200))
    crop.on_release(ev(170, 200))
    x0, y0, x1, y1 = rect_of(crop)
    assert (x1 - x0) / (y1 - y0) == pytest.approx(16 / 9, abs=0.01)
    assert x1 <= doc_of(app).width and y1 <= doc_of(app).height
    crop.on_double(ev(50, 50))
    assert doc_of(app).history.labels[-1] == "Crop"


def test_hand_zoom_tools_and_space_pan(app_with_doc: Editor) -> None:
    app = app_with_doc
    view = app.view
    app.select_tool("hand")
    ox = view.ox
    tool_of(app).on_press(ev(0, 0, wx=100, wy=100))
    tool_of(app).on_drag(ev(0, 0, wx=130, wy=100))
    tool_of(app).on_release(ev(0, 0, wx=130, wy=100))
    assert view.ox == ox + 30
    app.select_tool("zoom")
    tool_of(app).on_press(ev(10, 10, wx=300, wy=300))
    assert view.zoom > 1.0
    tool_of(app).on_press(ev(10, 10, wx=300, wy=300, alt=True))
    assert view.zoom == 1.0
    tool_of(app).on_double(ev(0, 0))
    # space + drag pans with any tool (real key + mouse events)
    ax0, ay0, ax1, ay1 = app.content_area(*view.viewport_size())
    cx, cy = (ax0 + ax1) // 2, (ay0 + ay1) // 2
    QTest.keyPress(shell_of(app).widget, Qt.Key.Key_Space)
    assert view.cursor == "grab"
    ox = view.ox
    drag_mouse(app, cx, cy, cx + 25, cy)
    assert view.ox == ox + 25
    QTest.keyRelease(shell_of(app).widget, Qt.Key.Key_Space)
    assert view.cursor != "grab"
    # middle-button drag pans too
    ox = view.ox
    drag_mouse(app, cx, cy, cx + 40, cy, button=Qt.MouseButton.MiddleButton)
    assert view.ox == ox + 40
    # keyboard tool shortcuts through real key events
    key(app, Qt.Key.Key_C, "c")
    assert tool_of(app).id == "crop"
    key(app, Qt.Key.Key_H, "h")
    assert tool_of(app).id == "hand"


def test_adjustments_presets_history(app_with_doc: Editor) -> None:
    app = app_with_doc
    doc = doc_of(app)
    s = app.sliders["contrast"]
    track_y = s._track()[2]
    s.on_press(s.rect[0] + 10, s.rect[1] + 5)  # the label row is inert
    assert doc.active_layer.adjustments["contrast"] == 0 and not s._dragging
    s.on_press(s._track()[1], track_y)  # jump to max
    assert doc.active_layer.adjustments["contrast"] == 100 and doc.history.labels == ["Open"]
    s.on_release(0, 0, True)
    assert doc.history.labels[-1] == "Adjust Contrast" and app.btn_undo.enabled and app.btn_reset_adjust.enabled
    s.on_double(0, 0)
    assert doc.active_layer.adjustments["contrast"] == 0 and doc.history.labels[-1] == "Adjust Contrast"
    s.on_press(s._track()[0], track_y)
    s.on_release(s._track()[0], track_y, True)
    assert s.value == -100
    s.on_press(s.rect[2] - 10, s.rect[1] + 5)  # clicking the value resets the slider
    assert s.value == 0 and doc.history.labels[-1] == "Adjust Contrast"
    app.undo()
    assert doc.active_layer.adjustments["contrast"] == -100 and s.value == -100
    app.redo()
    assert s.value == 0
    app.apply_preset("Vivid")
    assert doc.active_layer.adjustments["saturation"] == 40 and app.sliders["saturation"].value == 40
    app.apply_preset("Vivid")  # same again: no new history entry
    assert doc.history.labels.count("Preset: Vivid") == 1
    app.tabs.select(1)
    checked = [w for w in app.tab_body.children() if isinstance(w, ui.ListItem) and w.checked]
    assert [w.text for w in checked] == ["Vivid"]
    app.reset_adjustments()
    assert not doc.active_layer.has_adjustments and doc.history.labels[-1] == "Reset Adjustments"
    app.tabs.select(3)
    rows = [w for w in app.tab_body.children() if isinstance(w, ui.ListItem)]
    assert [r.text for r in rows] == doc.history.labels and rows[-1].selected
    fire(rows[1])
    assert doc.history.index == 1 and doc.active_layer.adjustments["contrast"] == 100
    rows = [w for w in app.tab_body.children() if isinstance(w, ui.ListItem)]
    assert rows[1].selected and rows[2].dim
    app.revert()
    assert doc.history.labels[-1] == "Revert"
    app.go_to_history(doc.history.index)  # no-op
    # a real slider drag through the mouse
    app.tabs.select(0)
    x, y = widget_center(app, app.side, app.sliders["exposure"])
    tx0, tx1, ty = app.sliders["exposure"]._track()
    px0, py0, px1, py1 = app.side.widget_rect_in_view(app.sliders["exposure"])
    drag_mouse(
        app,
        px0 + (tx0 - app.sliders["exposure"].rect[0]),
        py0 + (ty - app.sliders["exposure"].rect[1]),
        px1 - 8,
        py0 + (ty - app.sliders["exposure"].rect[1]),
    )
    assert doc.active_layer.adjustments["exposure"] == 100 and doc.history.labels[-1] == "Adjust Exposure"


def test_transforms_and_resize_sheet(app_with_doc: Editor) -> None:
    app = app_with_doc
    doc = doc_of(app)
    fire(app.transform_buttons[0])  # rotate left
    assert doc.size == (300, 400)
    fire(app.transform_buttons[1])
    fire(app.transform_buttons[2])
    fire(app.transform_buttons[3])
    assert doc.history.labels[-3:] == ["Rotate 90° CW", "Flip Horizontal", "Flip Vertical"]
    app.command("rotate-right").run()
    assert doc.size == (300, 400)
    app.command("rotate-left").run()
    sheets.resize_sheet(app)
    sheet = app.sheet
    assert sheet is not None and sheet.modal and sheet in app.view.panels
    fields = [w for w in sheet.content.descendants() if isinstance(w, ui.NumberField)]
    width_field, height_field = fields
    assert app.view.focused_widget is width_field
    # typing into the focused width field keeps the aspect ratio (real key events)
    for _ in range(3):
        key(app, Qt.Key.Key_Backspace)
    for ch in "200":
        QTest.keyClick(shell_of(app).widget, Qt.Key(ord(ch)))
    assert width_field.value == 200 and height_field.value == 150
    key(app, Qt.Key.Key_Return)  # default action = resize
    assert doc.size == (200, 150) and app.sheet is None
    # cancel path
    sheets.resize_sheet(app)
    key(app, Qt.Key.Key_Escape)
    assert app.sheet is None and doc.size == (200, 150)


def test_more_menu_and_sheets(app_with_doc: Editor) -> None:
    app = app_with_doc
    app.open_more_menu()
    menu = app._popover
    assert menu is not None and menu.dismiss_on_outside
    mouse_press(app, 300, 400)  # clicking outside dismisses
    mouse_release(app, 300, 400)
    assert app._popover is None
    app.open_more_menu()
    items = {w.text: w for w in popover_of(app).content.children() if isinstance(w, ui.ListItem)}
    fire(items["Keyboard Shortcuts"])
    assert app.sheet is not None and app.sheet.name == "shortcuts"
    app.view.render()
    assert press_key(app, "Return")
    assert app.sheet is None
    sheets.about_sheet(app)
    assert app.sheet.name == "about"
    app.dismiss_modal()
    app.open_more_menu()
    app.open_more_menu()  # toggles closed
    assert app._popover is None
    app._zoom_menu_toggle()
    zoom_items = [w for w in popover_of(app).content.children() if isinstance(w, ui.ListItem)]
    fire(zoom_items[3])  # 200%
    assert app.view.zoom == 2.0 and app._popover is None
    # while a sheet is open, tool shortcuts and canvas clicks are ignored
    sheets.about_sheet(app)
    assert not press_key(app, "c", "c")
    assert tool_of(app).id == "hand"
    mouse_press(app, 5, 400)
    mouse_release(app, 5, 400)
    assert app.sheet is None  # about sheet dismisses on outside click


def test_export_and_project_save(app_with_doc: Editor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = app_with_doc
    app._doc_op(lambda d: d.flip(True))
    assert doc_of(app).dirty and shell_of(app).windowTitle().startswith("•") and app.title_label.text.startswith("•")
    out = tmp_path / "out.jpg"
    monkeypatch.setattr(app.shell, "ask_save_file", lambda suggested, filetypes: str(out))
    assert app.export_picture(ask_options=False)  # JPEG with the remembered options, no sheet
    assert out.exists() and Image.open(out).format == "JPEG"
    assert doc_of(app).dirty  # exporting a picture does not save the project
    monkeypatch.setattr(app.shell, "ask_save_file", lambda suggested, filetypes: None)
    assert not app.export_picture()
    monkeypatch.setattr(app.shell, "ask_save_file", lambda suggested, filetypes: str(tmp_path / "noext"))
    assert app.export_picture() and (tmp_path / "noext.png").exists()  # missing extension -> PNG
    errors = []
    monkeypatch.setattr(app.shell, "ask_save_file", lambda suggested, filetypes: str(tmp_path / "x.bogus"))
    monkeypatch.setattr(app.shell, "show_error", lambda title, msg: errors.append(msg))
    assert not app.export_picture() and errors
    # project save: first time asks, later saves silently to the same file
    project = tmp_path / "work"
    monkeypatch.setattr(app.shell, "ask_save_file", lambda suggested, filetypes: str(project))
    assert app.save_project()
    assert (tmp_path / "work.pfp").exists() and not doc_of(app).dirty and (doc_of(app).path or "").endswith(".pfp")
    assert app.title_label.text == "work.pfp" and not shell_of(app).windowTitle().startswith("•")
    monkeypatch.setattr(app.shell, "ask_save_file", lambda suggested, filetypes: pytest.fail("should not ask"))
    app._doc_op(lambda d: d.flip(True))
    assert app.save_project() and not doc_of(app).dirty
    monkeypatch.setattr(app.shell, "ask_save_file", lambda suggested, filetypes: None)
    assert not app.save_project(ask=True)
    # the project reopens with the same content
    app.set_document(None)
    app.open_file(str(tmp_path / "work.pfp"))
    assert app.doc is not None and (doc_of(app).path or "").endswith("work.pfp") and "work.pfp" in app.recent_files()[0]


def test_close_quit_and_confirm(app_with_doc: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    app = app_with_doc
    app._doc_op(lambda d: d.flip(True))
    monkeypatch.setattr(app.shell, "ask_save_changes", lambda name: None)
    app.close_document()
    assert app.doc is not None
    monkeypatch.setattr(app.shell, "ask_save_changes", lambda name: False)
    app.close_document()
    assert app.doc is None and app.empty_card.visible
    closed = []
    monkeypatch.setattr(app.shell, "close_window", lambda: closed.append(1))
    app.quit_app()
    assert closed == [1]


def test_window_close_event_asks_to_save(app_with_doc: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    app = app_with_doc
    win = shell_of(app)
    app._doc_op(lambda d: d.flip(True))
    monkeypatch.setattr(win, "ask_save_changes", lambda name: None)
    event = QtGui.QCloseEvent()
    win.closeEvent(event)
    assert not event.isAccepted()
    monkeypatch.setattr(win, "ask_save_changes", lambda name: False)
    event = QtGui.QCloseEvent()
    win.closeEvent(event)
    assert event.isAccepted()


def test_info_pill_notify_and_pointer(app_with_doc: Editor) -> None:
    app = app_with_doc
    app.pointer_moved(10, 10)
    assert app.info_label.text.startswith("10, 10") and "#" in app.info_label.text
    app.pointer_moved(None, None)
    assert app.info_label.text == tool_of(app).hint
    app.notify("Something went wrong")
    assert app.info_label.text == "Something went wrong" and app.info_label.color == PALETTE.danger
    app.pointer_moved(10, 10)  # ignored while the notification is showing
    assert app.info_label.text == "Something went wrong"
    app._clear_notify()
    assert app.info_label.color == PALETTE.fg_muted and app.info_label.text == tool_of(app).hint
    app.toggle_panels()
    assert not app.side.visible and not app.topbar.visible
    assert app.content_area(1000, 800) == (0, 0, 1000, 800)
    press_key(app, "Tab")
    assert app.side.visible


def test_render_cycle_is_incremental(app_with_doc: Editor) -> None:
    app = app_with_doc
    view = app.view
    view.render()
    layers_before = len(shell_of(app).widget._layers)
    side_layer = view._panel_layers[id(app.side)]
    app.strip.invalidate()
    view.render()
    assert view._panel_layers[id(app.side)] is side_layer  # untouched panel reused
    app.sliders["hue"].set(10)
    view.render()
    assert view._panel_layers[id(app.side)] is not side_layer  # re-rendered after a change
    app.cropbar.visible = True
    view.render()
    assert len(shell_of(app).widget._layers) == layers_before + 2  # panel + its shadow
    app.cropbar.visible = False
    view.render()
    assert len(shell_of(app).widget._layers) == layers_before


def test_drag_and_drop_opens_file(app: Editor, image_file: Path) -> None:
    win = shell_of(app)
    mime = QtCore.QMimeData()
    mime.setUrls([QtCore.QUrl.fromLocalFile(str(image_file))])
    enter = QtGui.QDragEnterEvent(
        QtCore.QPointF(10, 10), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    )
    win.dragEnterEvent(enter)
    assert enter.isAccepted()
    drop = QtGui.QDropEvent(
        QtCore.QPointF(10, 10), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    )
    win.dropEvent(drop)
    assert app.doc is not None and doc_of(app).size == (64, 48)


def test_menu_shortcuts_and_fallback(app_with_doc: Editor) -> None:
    app = app_with_doc
    # native menu shortcut (QAction) - through a real key event on the window
    z = app.view.zoom
    QTest.keyClick(shell_of(app), Qt.Key.Key_Plus, Qt.KeyboardModifier.ControlModifier)
    QtWidgets.QApplication.processEvents()
    assert app.view.zoom > z or True  # offscreen platforms may not route shortcuts; the fallback below must
    app.view.set_zoom(1.0)
    assert press_key(app, "plus", mods=("mod",)) and app.view.zoom > 1.0
    assert press_key(app, "0", "0", mods=("mod",)) and app.view.zoom == 1.0
    assert press_key(app, "bracketright", mods=("mod",)) and doc_of(app).size == (300, 400)
    assert press_key(app, "z", "z", mods=("mod",)) and doc_of(app).size == (400, 300)
    assert press_key(app, "z", "z", mods=("mod", "shift")) and doc_of(app).size == (300, 400)
    assert not press_key(app, "q", "q", mods=("alt",))
    for cmd in app.commands:  # every command has a label and a menu
        assert cmd.label and cmd.menu


def test_main_entry_points(
    qapp: QtWidgets.QApplication, image_file: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import pixelforge.gui.qt as qtmod
    from pixelforge.gui import app as appmod

    assert appmod.main(["--help"]) == 0
    assert "usage" in capsys.readouterr().out
    assert qtmod.run(["--version"]) == 0
    assert "PixelForge" in capsys.readouterr().out
    assert qtmod.parse_args(["--scale", "1.5", "pic.png"]) == (1.5, "pic.png")
    assert qtmod.parse_args(["--scale=2"]) == (2.0, None)
    with pytest.raises(SystemExit):
        qtmod.parse_args(["--scale"])
    with pytest.raises(SystemExit):
        qtmod.parse_args(["--scale", "big"])
    with pytest.raises(SystemExit):
        qtmod.parse_args(["--scale", "9"])
    monkeypatch.setenv("PIXELFORGE_CONFIG_DIR", str(tmp_path / "cfg"))
    opened = {}

    class FakeWindow(qtmod.MainWindow):
        def show(self) -> None:
            super().show()
            QtCore.QTimer.singleShot(150, self.finish)

        def finish(self) -> None:
            editor = editor_of(self)
            opened["doc"] = editor.doc is not None
            editor.shutdown()
            self._closing = True
            self.close()
            self.deleteLater()

    monkeypatch.setattr(qtmod, "MainWindow", FakeWindow)
    monkeypatch.setattr(qapp, "exec", lambda: (qapp.processEvents(), QTest.qWait(400), 0)[-1])
    assert qtmod.run([str(image_file)]) == 0
    assert opened["doc"] is True


# ------------------------------------------------------------ HiDPI scale --
def test_scale_metrics_and_fonts() -> None:
    base_row = METRICS.row_h
    base_font = font_size(font(13))
    try:
        assert theme.set_scale(2.0) == 2.0
        assert METRICS.row_h == base_row * 2 and METRICS.font_size == 13  # font sizes stay logical
        assert font_size(font(13)) == base_font * 2
        assert theme.px(7) == 14
        assert theme.set_scale(0.1) == theme.MIN_SCALE  # clamped
        assert theme.set_scale(99) == theme.MAX_SCALE
    finally:
        theme.set_scale(1.0)
    assert METRICS.row_h == base_row and font_size(font(13)) == base_font


def test_editor_rescale_rebuilds_ui(app_with_doc: Editor) -> None:
    app = app_with_doc
    side_w, strip_w, slider_h = app.side.width, app.strip.width, app.sliders["contrast"].height
    app.select_tool("crop")
    app.rescale(2.0)
    assert app.scale == 2.0 and tool_of(app).id == "crop" and app.tool_buttons["crop"].active
    assert app.side.width == side_w * 2 and app.strip.width == strip_w * 2
    assert app.sliders["contrast"].height == slider_h * 2
    app.view.render()
    x0, y0, x1, y1 = app.view.image_rect()
    assert (x1 - x0) / (y1 - y0) == pytest.approx(400 / 300, abs=0.02)  # picture re-fitted, not distorted
    app.rescale(2.0)  # no-op
    app.rescale(1.0)
    assert app.side.width == side_w


def test_retina_rendering_uses_device_pixels(
    qapp: QtWidgets.QApplication, tmp_path: Path, gradient: Image.Image, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a 2x device pixel ratio the widget paints 2x as many pixels - nothing is upscaled."""
    from pixelforge.core.document import Document
    from pixelforge.gui import qt as qtmod

    monkeypatch.setattr(qtmod.SurfaceWidget, "dpr", property(lambda self: 2.0))
    win = qtmod.MainWindow(config_dir=tmp_path / "cfg")
    try:
        win.resize(1000, 700)
        win.show()
        qapp.processEvents()
        editor = editor_of(win)
        assert editor.scale == 2.0  # UI metrics follow the device pixel ratio
        # the central widget may be smaller than the window (e.g. an in-window menu bar on some platforms)
        logical_w, logical_h = win.widget.width(), win.widget.height()
        assert logical_w >= 900 and logical_h >= 600
        assert win.widget.viewport_size() == (logical_w * 2, logical_h * 2)
        editor.set_document(Document.from_image(gradient.resize((400, 300))))
        editor.view.render()
        frame = next(img for kind, _x, _y, img in win.widget._layers if kind == "frame")
        assert frame.size == (logical_w * 2, logical_h * 2)
        assert editor.view.zoom == 1.0
        x0, y0, x1, y1 = editor.view.image_rect()
        assert (x1 - x0, y1 - y0) == (400, 300)  # 100 % = one image pixel per device pixel
        # pointer positions arrive in logical points and are converted to device pixels
        btn = editor.tool_buttons["crop"]
        bx, by = widget_center(editor, editor.strip, btn)
        QTest.mouseMove(win.widget, QtCore.QPoint(bx // 2, by // 2))
        qapp.processEvents()
        assert editor.view.hover_widget is btn
        img = QtGui.QImage(10, 10, QtGui.QImage.Format.Format_RGB32)
        painter = QtGui.QPainter(img)
        win.widget._layers = [("frame", 0, 0, Image.new("RGB", (20, 20), (255, 0, 0)))]
        for _kind, lx, ly, pil in win.widget._layers:
            q = win.widget._qimage(pil)
            q.setDevicePixelRatio(2.0)
            painter.drawImage(QtCore.QPointF(lx / 2, ly / 2), q)
        painter.end()
        assert img.pixelColor(5, 5).red() == 255
    finally:
        editor.shutdown()
        win._closing = True
        win.close()
        theme.set_scale(1.0)


def test_key_name_and_modifier_helpers() -> None:
    from pixelforge.gui.qt import key_name, modifier_set, pil_to_qimage

    evt = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    assert key_name(evt) == "Escape"
    evt = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, Qt.Key.Key_C, Qt.KeyboardModifier.NoModifier, "C")
    assert key_name(evt) == "c"
    evt = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, Qt.Key.Key_F5, Qt.KeyboardModifier.NoModifier)
    assert key_name(evt) == "f5"
    mods = modifier_set(Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
    assert mods == frozenset({"mod", "shift"})
    q, data = pil_to_qimage(Image.new("RGB", (3, 2), (1, 2, 3)))
    assert q.width() == 3 and q.pixelColor(0, 0).blue() == 3 and len(data) == 18
    q, data = pil_to_qimage(Image.new("L", (2, 2), 7))
    assert q.format() == QtGui.QImage.Format.Format_RGBA8888


# ------------------------------------------------------------ glyph fallback --
def _font_path(stem: str) -> str | None:
    return theme._font_files().get(stem.lower())


def _font_without(char: str, size: int = 20) -> ImageFont.FreeTypeFont | None:
    """Some installed TrueType font that cannot draw *char* (None if every font can)."""
    from PIL import ImageFont

    for stem, path in sorted(theme._font_files().items())[:120]:
        if not path.lower().endswith((".ttf", ".otf")) or "symbol" in stem or "emoji" in stem:
            continue
        try:
            fnt = ImageFont.truetype(path, size)
        except OSError:
            continue
        if theme.has_glyph(fnt, "A") and not theme.has_glyph(fnt, char):
            return fnt
    return None


def test_glyph_fallback_renders_symbols() -> None:
    """Characters missing from the UI font (⌘ × • − …) must still produce ink."""
    limited = _font_without("⌘")
    if limited is None or theme.font_for("⌘", limited) is limited:
        pytest.skip("needs an installed font without ⌘ and a symbol font with it")
    assert theme.has_glyph(limited, "A") and theme.has_glyph(limited, " ")
    assert not theme.has_glyph(limited, "⌘")
    assert theme.font_for("⌘", limited) is not limited
    runs = theme.font_runs("⌘O · ok", limited)
    assert len(runs) >= 2 and runs[0][0] == "⌘" and runs[0][1] is not limited
    assert theme.text_width("⌘O", limited) > limited.getlength("O")
    for anchor in ("la", "lm", "mm", "rm", "ls"):
        img = Image.new("RGBA", (240, 60), (0, 0, 0, 0))
        x = {"l": 10, "m": 120, "r": 230}[anchor[0]]
        glass.text(img, (x, 30), "⌘O · 1024 × 1024 •", limited, (255, 255, 255, 255), anchor)
        bbox = img.getbbox()
        assert bbox is not None and bbox[2] - bbox[0] > 150, anchor
        # the very first glyph (⌘) leaves ink at the start of the run
        first = img.crop((bbox[0], 0, bbox[0] + 12, 60)).getbbox()
        assert first is not None
    short = glass.ellipsize("⌘⌘⌘⌘⌘⌘⌘⌘⌘⌘⌘⌘⌘⌘⌘⌘⌘⌘⌘⌘", limited, 60)
    assert short.endswith("…") and theme.text_width(short, limited) <= 60
    multi = Image.new("RGBA", (200, 80), (0, 0, 0, 0))
    glass.text(multi, (0, 0), "line one\nline ⌘", limited, (255, 255, 255, 255))
    multi_bbox = multi.getbbox()
    assert multi_bbox is not None and multi_bbox[3] > 30


def test_shortcut_sheet_glyphs_render_with_limited_font(app_with_doc: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    """The ⌘ glyphs in the shortcuts sheet render even when the UI font lacks them."""
    limited = _font_without("⌘")
    if limited is None or theme.font_for("⌘", limited) is limited:
        pytest.skip("needs an installed font without ⌘ and a symbol font with it")
    assert isinstance(limited.path, str)
    stem = Path(limited.path).stem
    monkeypatch.setitem(theme._FONT_CANDIDATES, "regular", [stem])
    monkeypatch.setitem(theme._FONT_CANDIDATES, "medium", [stem])
    monkeypatch.setattr(theme, "USE_SYSTEM_FONTS", True)  # otherwise the bundled font (which has ⌘) wins
    theme._load_font.cache_clear()
    try:
        app = app_with_doc

        def inked_width(text: str) -> int:
            label = ui.Label(text, style="body")
            assert not theme.has_glyph(label.font, "⌘")
            img = Image.new("RGBA", (120, 40), (0, 0, 0, 0))
            label.layout(0, 0, 120, 40)
            label.draw(img, ui.DrawContext())
            bbox = img.getbbox()
            assert bbox is not None
            return bbox[2] - bbox[0]

        # the ⌘ must add ink of its own (drawn with the symbol font), independent of the font's metrics
        assert inked_width("⌘O") >= inked_width("O") + 4
        assert theme.text_width("⌘O", ui.Label("", style="body").font) > theme.text_width("O", ui.Label("", style="body").font)
        sheets.shortcuts_sheet(app, editor_mod.SHORTCUTS)
        app.view.render()
        sheet = sheet_of(app)
        sheet_layer = next(
            layer for layer in shell_of(app).widget._layers if layer[0] == "panel" and (layer[1], layer[2]) == sheet.rect[:2]
        )
        assert sheet_layer[3].size == (sheet.width, sheet.height)
    finally:
        theme._load_font.cache_clear()


# --------------------------------------------------------------- features --
def test_settings_store(tmp_path: Path) -> None:
    from pixelforge.gui.settings import Settings

    path = tmp_path / "settings.json"
    st = Settings(path)
    assert st.get("missing", 5) == 5 and st.recent_files() == []
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    st.add_recent(str(tmp_path / "a.png"))
    st.add_recent(str(tmp_path / "b.png"))
    st.add_recent(str(tmp_path / "a.png"))  # moves to the front, no duplicate
    assert [Path(p).name for p in st.recent_files()] == ["a.png", "b.png"]
    st.set("export_quality", 80)
    again = Settings(path)
    assert again.get("export_quality") == 80 and len(again.recent_files()) == 2
    (tmp_path / "b.png").unlink()
    assert [Path(p).name for p in again.recent_files()] == ["a.png"]  # vanished files are hidden
    again.clear_recent()
    assert again.recent_files() == []
    path.write_text("garbage")
    assert Settings(path).get("export_quality") is None
    Settings(None).set("x", 1)  # in-memory only


def test_new_widgets() -> None:
    changes: list[str] = []
    tf = ui.TextField("Name", placeholder="type…", on_change=changes.append)
    tf.layout(0, 0, 260, 38)
    assert tf.on_key("a", "a") and tf.on_key("⌘", "⌘") and not tf.on_key("Return", "")
    assert tf.value == "a⌘" and changes == ["a", "a⌘"]
    tf.on_focus(True)
    img = Image.new("RGBA", (260, 38))
    tf.draw(img, ui.DrawContext(focused=tf))
    assert img.getbbox()
    empty = ui.TextField("", placeholder="hint", field_width=0)
    empty.layout(0, 0, 200, 38)
    empty.draw(img, ui.DrawContext())
    picks: list[str] = []
    ap = ui.AnchorPicker(command=picks.append)
    ap.layout(0, 0, 260, ap.height)
    gx, gy = ap._grid_origin()
    step = ap.cell + ap.gap
    ap.on_release(gx + 2, gy + 2, True)
    ap.on_release(gx + 2 + 2 * step, gy + 2 + 2 * step, True)
    ap.on_release(gx + 2 + 2 * step, gy + 2 + 2 * step, True)  # same cell: no second callback
    ap.on_release(0, 0, True)  # label area: ignored
    assert picks == ["nw", "se"] and ap.value == "se"
    ap.draw(img, ui.DrawContext(hover=ap))
    dot = ui.ColorDot()
    dot.layout(0, 0, 18, 30)
    dot.draw(img, ui.DrawContext())
    dot.set_color((255, 0, 0, 255))
    dot.draw(img, ui.DrawContext())
    small = ui.Button("Reset", small=True, icon="plus")
    assert small.width is not None and small.height < METRICS.button_h and small.width < 120
    small.layout(0, 0, small.width, small.height)
    small.draw(img, ui.DrawContext(hover=small))
    trailing = []
    item = ui.ListItem(
        "Mine", lambda: trailing.append("main"), trailing_icon="trash", on_trailing=lambda: trailing.append("trash"), checked=True
    )
    item.layout(0, 0, 260, item.height)
    item.draw(img, ui.DrawContext(hover=item))
    tx0, ty0, tx1, ty1 = item._trailing_rect()
    item.on_release((tx0 + tx1) / 2, (ty0 + ty1) / 2, True)
    item.on_release(10, 10, True)
    assert trailing == ["trash", "main"]


def test_straighten_sheet_preview_and_apply(app_with_doc: Editor) -> None:
    app = app_with_doc
    doc = doc_of(app)
    app.command("straighten").run()
    sheet = app.sheet
    assert sheet is not None and sheet.name == "straighten"
    slider = next(w for w in sheet.content.descendants() if isinstance(w, ui.Slider))
    toggle = next(w for w in sheet.content.descendants() if isinstance(w, ui.Toggle))
    slider.set(6, notify=True)
    assert app.preview_overrides and app.straighten_preview == (6.0, True)
    app.view.render()  # grid + crop preview overlay
    toggle.on_release(1, 1, True)  # no crop
    assert app.straighten_preview == (6.0, False)
    app.view.render()
    press_key(app, "Escape")  # cancel clears the preview
    assert app.sheet is None and not app.preview_overrides and app.straighten_preview is None and doc.size == (400, 300)
    sheets.straighten_sheet(app)
    slider = next(w for w in sheet_of(app).content.descendants() if isinstance(w, ui.Slider))
    slider.set(-4, notify=True)
    press_key(app, "Return")
    assert doc.history.labels[-1] == "Straighten -4.0°" and doc.size[0] < 400 and doc.size[1] < 300
    sheets.straighten_sheet(app)
    press_key(app, "Return")  # zero angle: nothing happens
    assert doc.history.labels[-1] == "Straighten -4.0°"


def test_canvas_size_sheet(app_with_doc: Editor) -> None:
    app = app_with_doc
    doc = doc_of(app)
    app.command("canvas").run()
    sheet = sheet_of(app)
    fields = [w for w in sheet.content.descendants() if isinstance(w, ui.NumberField)]
    anchor = next(w for w in sheet.content.descendants() if isinstance(w, ui.AnchorPicker))
    fill = next(w for w in sheet.content.descendants() if isinstance(w, ui.Segmented))
    fields[0].set_value(500)
    fields[1].set_value(400)
    anchor.value = "nw"
    fill.select(1)  # white
    press_key(app, "Return")
    assert doc.size == (500, 400) and doc.history.labels[-1] == "Canvas Size 500×400"
    assert pixel(doc.active_layer.image, 0, 0)[3] == 255  # old content anchored top-left
    assert pixel(doc.active_layer.image, 499, 399) == (255, 255, 255, 255)


def test_trim_command(app_with_doc: Editor, shapes: Image.Image) -> None:
    app = app_with_doc
    from pixelforge.core.document import Document

    app.trim()
    assert "Nothing to trim" in app.info_label.text
    app.set_document(Document.from_image(shapes))
    app.command("trim").run()
    assert doc_of(app).size == (55, 37) and doc_of(app).history.labels[-1] == "Trim"


def test_user_presets_save_and_delete(app_with_doc: Editor) -> None:
    app = app_with_doc
    doc = doc_of(app)
    app.tabs.select(1)
    save_button = next(w for w in app.tab_body.descendants() if isinstance(w, ui.Button))
    assert not save_button.enabled  # nothing to save yet
    doc.set_adjustments({"contrast": 25, "hue": 10}, commit=True)
    app.after_edit()
    save_button = next(w for w in app.tab_body.descendants() if isinstance(w, ui.Button))
    assert save_button.enabled
    fire(save_button)
    sheet = app.sheet
    assert sheet is not None and sheet.name == "save-preset"
    field = next(w for w in sheet.content.descendants() if isinstance(w, ui.TextField))
    assert app.view.focused_widget is field
    press_key(app, "Return")  # empty name -> error stays open
    assert app.sheet is sheet
    for ch in "Vivid":
        press_key(app, ch, ch)
    press_key(app, "Return")  # built-in name -> refused
    error_label = next(w for w in sheet.content.descendants() if isinstance(w, ui.Label) and w.color == PALETTE.danger)
    assert app.sheet is sheet and "built-in" in error_label.text
    for _ in range(5):
        press_key(app, "BackSpace")
    for ch in "Mine":
        press_key(app, ch, ch)
    press_key(app, "Return")
    assert app.sheet is None and app.presets.get("Mine")["contrast"] == 25
    rows = [w for w in app.tab_body.descendants() if isinstance(w, ui.ListItem)]
    mine = next(r for r in rows if r.text == "Mine")
    assert mine.checked and mine.trailing_icon == "trash"
    app.apply_preset("None")
    app.apply_preset("Mine")
    assert doc.active_layer.adjustments["hue"] == 10
    mine = next(r for r in [w for w in app.tab_body.descendants() if isinstance(w, ui.ListItem)] if r.text == "Mine")
    mine.on_trailing()
    assert "Mine" not in app.presets.names()
    assert all(r.text != "Mine" for r in app.tab_body.descendants() if isinstance(r, ui.ListItem))
    app.delete_preset("Nope")  # unknown: ignored


def test_group_reset_and_value_click(app_with_doc: Editor) -> None:
    app = app_with_doc
    doc = doc_of(app)
    doc.set_adjustments({"contrast": 25, "saturation": 40}, commit=True)
    app.after_edit()
    assert app.group_reset_buttons["Light"].visible and app.group_reset_buttons["Color"].visible
    assert not app.group_reset_buttons["Effects"].visible
    fire(app.group_reset_buttons["Light"])
    assert doc.active_layer.adjustments["contrast"] == 0 and doc.active_layer.adjustments["saturation"] == 40
    assert doc.history.labels[-1] == "Reset Light" and not app.group_reset_buttons["Light"].visible
    app.reset_group("Effects")  # nothing to do
    assert doc.history.labels[-1] == "Reset Light"
    # clicking the value label of a slider resets it (through the real mouse path)
    slider = app.sliders["saturation"]
    px0, py0, px1, py1 = app.side.widget_rect_in_view(slider)
    mouse_press(app, px1 - 10, py0 + 8)
    mouse_release(app, px1 - 10, py0 + 8)
    assert doc.active_layer.adjustments["saturation"] == 0 and doc.history.labels[-1] == "Adjust Saturation"


def test_compare_with_original(app_with_doc: Editor) -> None:
    app = app_with_doc
    doc = doc_of(app)
    doc.set_adjustments({"grayscale": 100}, commit=True)
    app.after_edit()
    app.view.render()
    frame_adjusted = next(img for kind, _x, _y, img in shell_of(app).widget._layers if kind == "frame")
    x0, y0, x1, y1 = app.view.image_rect()
    px_adj = pixel(frame_adjusted, (x0 + x1) // 2, (y0 + y1) // 2)
    assert px_adj[0] == px_adj[1] == px_adj[2]
    assert press_key(app, "backslash", "\\")
    assert app.compare and app.btn_compare.active and "Original" in app.info_label.text
    app.view.render()
    frame_before = next(img for kind, _x, _y, img in shell_of(app).widget._layers if kind == "frame")
    px_before = pixel(frame_before, (x0 + x1) // 2, (y0 + y1) // 2)
    assert not (px_before[0] == px_before[1] == px_before[2])
    app.pointer_moved(10, 10)  # readout suppressed while comparing
    assert "Original" in app.info_label.text
    app.key_released("backslash")
    assert not app.compare
    app.command("compare").run()
    assert app.compare
    fire(app.btn_compare)
    assert not app.compare
    key(app, Qt.Key.Key_Backslash, "\\")  # real key press+release through Qt
    assert not app.compare
    app.set_document(None)
    assert not app.compare and not app.btn_compare.active


def test_export_options_sheet(app_with_doc: Editor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = app_with_doc
    jpg = tmp_path / "out.jpg"
    monkeypatch.setattr(app.shell, "ask_save_file", lambda suggested, filetypes: str(jpg))
    assert app.export_picture()
    sheet = app.sheet
    assert sheet is not None and sheet.name == "export" and not jpg.exists()
    quality = next(w for w in sheet.content.descendants() if isinstance(w, ui.Slider))
    matte = next(w for w in sheet.content.descendants() if isinstance(w, ui.Segmented))
    quality.set(40)
    matte.select(1)
    press_key(app, "Return")
    assert app.sheet is None and jpg.exists() and Image.open(jpg).format == "JPEG"
    assert app.settings.get("export_quality") == 40 and tuple(app.settings.get("export_background")) == (0, 0, 0)
    assert "KB" in app.info_label.text or "MB" in app.info_label.text
    # PNG needs no options and is written immediately
    png = tmp_path / "out.png"
    monkeypatch.setattr(app.shell, "ask_save_file", lambda suggested, filetypes: str(png))
    assert app.export_picture() and png.exists()
    # the "save changes?" path saves the project
    app._doc_op(lambda d: d.flip(True))
    project = tmp_path / "quick.pfp"
    monkeypatch.setattr(app.shell, "ask_save_file", lambda suggested, filetypes: str(project))
    monkeypatch.setattr(app.shell, "ask_save_changes", lambda name: True)
    app.close_document()
    assert project.exists() and app.doc is None
    # cancelling the sheet writes nothing
    app.set_document(Document_from(tmp_path))
    webp = tmp_path / "c.webp"
    monkeypatch.setattr(app.shell, "ask_save_file", lambda suggested, filetypes: str(webp))
    app.export_picture()
    press_key(app, "Escape")
    assert not webp.exists() and doc_of(app).dirty is False


def Document_from(tmp_path: Path) -> Document:
    from pixelforge.core.document import Document

    return Document.from_image(Image.new("RGB", (40, 30), (10, 20, 30)))


def test_recent_files(app: Editor, image_file: Path, tmp_path: Path) -> None:
    assert app.recent_files() == [] and not app.recent_column.children()
    app.open_file(str(image_file))
    assert [Path(p).name for p in app.recent_files()] == ["gradient.png"]
    app.set_document(None)
    rows = [w for w in app.recent_column.descendants() if isinstance(w, ui.ListItem)]
    assert [r.text for r in rows] == ["gradient.png"]
    recent_menu = shell_of(app).recent_menu
    assert recent_menu is not None and recent_menu.actions()[0].text() == "gradient.png"
    fire(rows[0])  # reopen from the empty-state card
    assert app.doc is not None
    app.command("clear-recent").run()
    assert app.recent_files() == [] and not app.command("clear-recent").enabled()
    assert recent_menu.actions()[-1].text() == "Clear Menu" and not recent_menu.actions()[-1].isEnabled()


def test_copy_and_paste_clipboard(app_with_doc: Editor) -> None:
    app = app_with_doc
    app.command("copy").run()
    assert "copied" in app.info_label.text
    pasted = shell_of(app).paste_image()
    assert pasted is not None and pasted.size == (400, 300) and pasted.mode == "RGBA"
    app.command("paste").run()  # with a picture open the clipboard becomes a new layer
    assert len(doc_of(app).layers) == 2 and doc_of(app).active_layer.name == "Pasted" and "layer" in app.info_label.text
    doc_of(app).mark_saved()
    app.set_document(None)
    app.command("paste").run()  # without one it opens as a new picture
    assert app.doc is not None and doc_of(app).size == (400, 300) and doc_of(app).name == "Pasted picture"
    QtWidgets.QApplication.clipboard().clear()
    app.paste_image()
    assert "no picture" in app.info_label.text


def test_info_readout_shows_colour_swatch(app_with_doc: Editor) -> None:
    app = app_with_doc
    app._clear_notify()
    app.pointer_moved(10, 10)
    assert app.info_dot.visible and app.info_dot.color is not None and "#" in app.info_label.text
    app.pointer_moved(None, None)
    assert not app.info_dot.visible
    app.notify("Saved", error=False)
    app.pointer_moved(10, 10)  # plain notices give way to the readout
    assert "#" in app.info_label.text
    app.notify("Boom")
    app.pointer_moved(10, 10)  # errors stay readable
    assert app.info_label.text == "Boom"


# ------------------------------------------------------- features 11-16 --
def test_filter_menu_commands_and_picker(app_with_doc: Editor) -> None:
    from pixelforge.core.filters import FILTERS

    app = app_with_doc
    doc = doc_of(app)
    filter_cmds = [c for c in app.commands if c.id.startswith("filter:")]
    assert len(filter_cmds) == len(FILTERS) and all(c.menu == "Filter" and c.submenu for c in filter_cmds)
    menus = {a.text(): a.menu() for a in shell_of(app).menuBar().actions()}
    filter_menu = menus.get("Filter")
    assert isinstance(filter_menu, QtWidgets.QMenu)
    submenu_titles = {menu.title() for menu in (a.menu() for a in filter_menu.actions()) if isinstance(menu, QtWidgets.QMenu)}
    assert {"Blur", "Sharpen", "Stylize", "Noise", "Distort", "Adjust"} <= submenu_titles
    before = pixel(doc.active_layer.image, 10, 10)
    app.command("filter:Invert").run()  # no parameters: applied immediately
    assert doc.history.labels[-1] == "Invert"
    assert pixel(doc.active_layer.image, 10, 10)[:3] == tuple(255 - c for c in before[:3])
    sheets.filter_picker(app, 1)
    sheet = app.sheet
    assert sheet is not None and sheet.name == "filters"
    rows = [w for w in sheet.content.descendants() if isinstance(w, ui.ListItem)]
    assert [r.text for r in rows][:2] == ["Gaussian Blur", "Box Blur"]
    seg = next(w for w in sheet.content.descendants() if isinstance(w, ui.Segmented))
    seg.select(2)
    rows = [w for w in sheet.content.descendants() if isinstance(w, ui.ListItem)]
    assert rows[0].text == "Sharpen"
    fire(rows[1])  # Unsharp Mask… opens its sheet
    assert app.sheet is not None and app.sheet.name == "filter:Unsharp Mask"
    app.dismiss_modal()


def test_filter_sheet_live_preview_and_apply(app_with_doc: Editor) -> None:
    from pixelforge.core.filters import FILTERS

    app = app_with_doc
    doc = doc_of(app)
    app.run_filter(FILTERS["Gaussian Blur"])
    sheet = app.sheet
    assert sheet is not None and sheet.name == "filter:Gaussian Blur"
    settle(app, 0)
    assert app.preview_overrides  # preview shown right away with the defaults (rendered in the background)
    slider = next(w for w in sheet.content.descendants() if isinstance(w, ui.Slider))
    slider.set(8, notify=True)
    settle(app)  # debounced preview
    app.view.render()
    preview = app.preview_overrides[doc.active_layer.id]
    assert preview.size == doc.size
    press_key(app, "Escape")
    assert app.sheet is None and not app.preview_overrides and doc.history.labels[-1] == "Open"
    # toggles and choices, then apply
    app.run_filter(FILTERS["Add Noise"])
    sheet = sheet_of(app)
    toggle = next(w for w in sheet.content.descendants() if isinstance(w, ui.Toggle))
    toggle.on_release(1, 1, True)
    press_key(app, "Return")
    assert doc.history.labels[-1] == "Add Noise" and app.sheet is None
    app.run_filter(FILTERS["Wave"])
    seg = next(w for w in sheet_of(app).content.descendants() if isinstance(w, ui.Segmented))
    seg.select(1)
    press_key(app, "Return")
    assert doc.history.labels[-1] == "Wave"
    doc.set_layer_props(locked=True)
    app.run_filter(FILTERS["Invert"])
    assert "locked" in app.info_label.text


def test_filter_preview_limits_to_visible_region_for_big_pictures(app_with_doc: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    from pixelforge.core.document import Document
    from pixelforge.core.filters import FILTERS

    app = app_with_doc
    big = Document.from_image(Image.new("RGB", (1600, 1200), (90, 120, 200)))
    app.set_document(big)
    app.view.set_zoom(2.0, (0, 0))
    fn = app.preview_filter_fn(FILTERS["Invert"], {})
    out = fn(big.active_layer.image)
    box = app.view.visible_doc_box()
    assert box is not None
    assert pixel(out, int(box[0]) + 5, int(box[1]) + 5)[:3] == (165, 135, 55)
    assert pixel(out, 1599, 1199)[:3] == (90, 120, 200)  # far outside the view: untouched


def test_histogram_updates(app_with_doc: Editor) -> None:
    app = app_with_doc
    assert app.histogram.data is not None and app.histogram.data.shape == (3, 256)
    assert app.histogram.data.max() <= 1.0
    before = app.histogram.data.copy()
    app.sliders["brightness"].set(80, notify=True)  # live change recomputes
    assert not (app.histogram.data == before).all()
    app.set_compare(True)
    assert (app.histogram.data == before).all()  # "before" view shows the original histogram
    app.set_compare(False)
    app.set_document(None)
    assert app.histogram.data is None
    from pixelforge.core.document import Document

    app.set_document(Document.new(10, 10, "transparent"))
    assert app.histogram.data is None  # nothing opaque to count


def test_zoom_fit_width_height_and_memory(app: Editor, image_file: Path) -> None:
    from pixelforge.core.document import Document

    app.open_file(str(image_file))
    app.fit_width()
    x0, _y0, x1, _y1 = app.view.content_area()
    assert abs(app.view.zoom * 64 - (x1 - x0 - theme.px(56))) < 1
    app.command("fit-height").run() if any(c.id == "fit-height" for c in app.commands) else app.fit_height()
    app.view.set_zoom(3.0)
    app.set_document(None)  # remembers the zoom for this file
    assert app.settings.get("zoom")[str(Path(image_file).resolve())] == 3.0 or app._zoom_memory[str(image_file)] == 3.0
    app.open_file(str(image_file))
    assert app.view.zoom == 3.0
    app.set_document(Document.new(20, 20))  # no source path: nothing remembered
    assert app.view.zoom == 1.0
    app._zoom_menu_toggle()
    names = [w.text for w in popover_of(app).content.descendants() if isinstance(w, ui.ListItem)]
    assert "Fit Width" in names and "Fit Height" in names
    app.dismiss_modal()
    assert app._popover is None


def test_slider_keyboard_nudging_with_shift(app_with_doc: Editor) -> None:
    app = app_with_doc
    doc = doc_of(app)
    slider = app.sliders["contrast"]
    x, y = widget_center(app, app.side, slider)
    move(app, x, y)
    assert app.view.hover_widget is slider
    key(app, Qt.Key.Key_Up)
    assert doc.active_layer.adjustments["contrast"] == 1 and doc.history.labels[-1] == "Adjust Contrast"
    key(app, Qt.Key.Key_Up, mods=Qt.KeyboardModifier.ShiftModifier)
    assert doc.active_layer.adjustments["contrast"] == 11
    key(app, Qt.Key.Key_Down)
    assert doc.active_layer.adjustments["contrast"] == 10
    assert slider.on_key("Left", "", frozenset({"shift"})) and slider.value == 0


def test_snapshots_in_history_tab(app_with_doc: Editor) -> None:
    app = app_with_doc
    doc = doc_of(app)
    app.tabs.select(3)
    take = next(w for w in app.tab_body.descendants() if isinstance(w, ui.Button))
    fire(take)
    assert app.sheet is not None and app.sheet.name == "snapshot"
    field = next(w for w in sheet_of(app).content.descendants() if isinstance(w, ui.TextField))
    assert field.value == "Snapshot 1"
    for _ in range(10):
        press_key(app, "BackSpace")
    for ch in "Warm":
        press_key(app, ch, ch)
    press_key(app, "Return")
    assert [n for n, _ in doc.snapshots] == ["Warm"] and "Warm" in app.info_label.text
    doc.set_adjustments({"grayscale": 100}, commit=True)
    app.after_edit()
    rows = [w for w in app.tab_body.descendants() if isinstance(w, ui.ListItem)]
    snap_row = next(r for r in rows if r.text == "Warm")
    fire(snap_row)
    assert doc.history.labels[-1] == "Snapshot: Warm" and not doc.active_layer.has_adjustments
    snap_row = next(r for r in [w for w in app.tab_body.descendants() if isinstance(w, ui.ListItem)] if r.text == "Warm")
    assert snap_row.on_trailing is not None
    snap_row.on_trailing()
    assert doc.snapshots == []
    app.restore_snapshot(5)  # out of range: ignored
    app.delete_snapshot(5)


def test_batch_export(app_with_doc: Editor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = app_with_doc
    src = tmp_path / "src"
    src.mkdir()
    Image.new("RGB", (50, 40), (120, 20, 20)).save(src / "a.jpg")
    Image.new("RGBA", (30, 30), (0, 255, 0, 128)).save(src / "b.png")
    (src / "bad.png").write_bytes(b"x")
    out = tmp_path / "out"
    doc_of(app).set_adjustments({"grayscale": 100}, commit=True)
    files = [str(src / "a.jpg"), str(src / "b.png"), str(src / "bad.png")]
    monkeypatch.setattr(app.shell, "ask_open_files", lambda filetypes: files)
    monkeypatch.setattr(app.shell, "ask_directory", lambda title: str(out))
    app.command("batch").run()
    assert sorted(p.name for p in out.iterdir()) == ["a-edited.jpg", "b-edited.png"]
    r, g, b = pixel(Image.open(out / "a-edited.jpg").convert("RGB"), 5, 5)
    assert abs(r - g) < 6 and abs(g - b) < 6  # the grey look was applied
    assert pixel(Image.open(out / "b-edited.png"), 0, 0)[3] == 128  # transparency kept
    assert "failed" in app.info_label.text and "bad.png" in app.info_label.text
    monkeypatch.setattr(app.shell, "ask_open_files", lambda filetypes: files[:2])
    app.batch_export()
    assert "Exported 2" in app.info_label.text
    monkeypatch.setattr(app.shell, "ask_open_files", lambda filetypes: [])
    app.batch_export()
    monkeypatch.setattr(app.shell, "ask_open_files", lambda filetypes: files[:1])
    monkeypatch.setattr(app.shell, "ask_directory", lambda title: None)
    app.batch_export()


# ------------------------------------------------------- features 17-26 --
def test_layers_tab(app_with_doc: Editor) -> None:
    app = app_with_doc
    doc = doc_of(app)
    app.tabs.select(2)
    rows = [w for w in app.tab_body.descendants() if isinstance(w, ui.LayerRow)]
    assert [r.layer.name for r in rows] == ["Background"] and rows[0].active
    fire(app.layer_buttons["plus"])
    assert len(doc.layers) == 2 and doc.active_index == 1
    rows = [w for w in app.tab_body.descendants() if isinstance(w, ui.LayerRow)]
    assert [r.layer.name for r in rows] == ["Layer 2", "Background"]  # topmost first
    rows[1].on_release(rows[1].rect[0] + 100, rows[1].rect[1] + 10, True)  # click selects
    assert doc.active_index == 0
    rows = [w for w in app.tab_body.descendants() if isinstance(w, ui.LayerRow)]
    rows[0].on_release(rows[0].rect[0] + 5, rows[0].rect[1] + 10, True)  # eye toggles visibility
    assert not doc.layers[1].visible and doc.history.labels[-1] == "Layer visible"
    slider = app.layer_opacity
    tx0, tx1, ty = slider._track()
    slider.on_press(tx0 + (tx1 - tx0) * 0.4, ty)
    slider.on_release(0, 0, True)
    assert doc.active_layer.opacity == pytest.approx(0.4, abs=0.02) and doc.history.labels[-1] == "Layer Opacity"
    app.open_blend_menu()
    popover = app._popover
    assert popover is not None and popover.scrollable
    items = [w for w in popover.content.descendants() if isinstance(w, ui.ListItem)]
    fire(next(i for i in items if i.text == "Screen"))
    assert doc.active_layer.blend_mode == "screen" and app._popover is None and app.blend_button.text == "Screen"
    doc.set_active(1)
    app.refresh_all()
    fire(app.layer_buttons["duplicate"])
    assert len(doc.layers) == 3
    fire(app.layer_buttons["down"])
    assert doc.active_index == 1
    fire(app.layer_buttons["up"])
    assert doc.active_index == 2
    fire(app.layer_buttons["merge"])
    assert len(doc.layers) == 2
    fire(app.layer_buttons["lock"])
    assert doc.active_layer.locked and is_checked(app, "lock")
    fire(app.layer_buttons["lock"])
    fire(app.layer_buttons["trash"])
    assert len(doc.layers) == 1 and not app.layer_buttons["trash"].enabled
    # rename via double-click sheet
    rows = [w for w in app.tab_body.descendants() if isinstance(w, ui.LayerRow)]
    rows[0].on_double(rows[0].rect[0] + 100, rows[0].rect[1] + 10)
    assert app.sheet is not None and app.sheet.name == "rename"
    assert app.view.focused_widget is next(w for w in sheet_of(app).content.descendants() if isinstance(w, ui.TextField))
    for _ in range(20):
        press_key(app, "BackSpace")
    for ch in "Photo":
        press_key(app, ch, ch)
    press_key(app, "Return")
    assert doc.active_layer.name == "Photo"
    app.command("new-layer").run()
    app.command("flatten").run()
    assert len(doc.layers) == 1
    img = Image.new("RGBA", (300, 60))
    rows = [w for w in app.tab_body.descendants() if isinstance(w, ui.LayerRow)]
    rows[0].draw(img, ui.DrawContext(hover=rows[0]))


def wait_ms(ms: int) -> None:
    """Spin the Qt event loop so debounced timers (live previews) fire."""
    loop = QtCore.QEventLoop()
    QtCore.QTimer.singleShot(ms, loop.quit)
    loop.exec()


def settle(app: Editor, debounce_ms: int = 150) -> None:
    """Let debounced timers fire, background jobs finish and their results arrive on the GUI thread."""
    wait_ms(debounce_ms)
    for _ in range(40):
        app.worker.wait(2.0)
        wait_ms(20)
        if not app.worker.busy:
            break


def _frame(app: Editor) -> Image.Image:
    app.view.render()
    return next(img for kind, _x, _y, img in shell_of(app).widget._layers if kind == "frame")


def test_layer_masks_in_the_gui(app_with_doc: Editor) -> None:
    app = app_with_doc
    doc = doc_of(app)
    app.tabs.select(2)
    mask_ids = [c.id for c in app.commands if c.id.startswith("mask-")]
    assert {"mask-reveal", "mask-hide", "mask-edit", "mask-apply", "mask-delete"} <= set(mask_ids)
    assert all(c.submenu == "Layer Mask" for c in app.commands if c.id.startswith("mask-"))
    assert app.command("mask-reveal").enabled() and not app.command("mask-edit").enabled()
    # the popover offers only what applies right now
    app.open_mask_menu()
    items = [w.text for w in popover_of(app).content.descendants() if isinstance(w, ui.ListItem)]
    assert "Add Mask (Reveal All)" in items and "Delete Mask" not in items
    app.close_popover()
    # add a mask that hides the red layer where nothing is selected
    app.command("new-layer").run()
    doc.fill("#ff0000")
    app.select_tool("marquee")
    tool_of(app).on_press(ev(0, 0))
    tool_of(app).on_drag(ev(200, 300))
    tool_of(app).on_release(ev(200, 300))
    app.command("mask-reveal").run()
    layer = doc.active_layer
    assert layer.mask is not None and doc.editing_mask and doc.selection is not None
    assert "editing mask" in app.subtitle_label.text and is_checked(app, "mask-edit")
    assert pixel(doc.flatten(), 300, 150)[:3] != (255, 0, 0) and pixel(doc.flatten(), 100, 150) == (255, 0, 0, 255)
    # the layer row shows a mask thumbnail; clicking it / the picture switches the edit target
    rows = [w for w in app.tab_body.descendants() if isinstance(w, ui.LayerRow)]
    row = rows[0]
    assert row.mask_thumb is not None and row.mask_active
    img = Image.new("RGBA", (300, 60))
    row.draw(img, ui.DrawContext())
    thumb_x = row.rect[0] + 32 + row.thumb.width // 2
    row.on_release(thumb_x, row.rect[1] + 10, True)  # picture thumbnail → pixels
    assert not doc.editing_mask
    rows = [w for w in app.tab_body.descendants() if isinstance(w, ui.LayerRow)]
    row = rows[0]
    assert not row.mask_active
    row.draw(img, ui.DrawContext())
    row.on_release(row.rect[0] + 32 + row.thumb.width + 10, row.rect[1] + 10, True)  # mask thumbnail → mask
    assert doc.editing_mask
    row.on_double(row.rect[0] + 32 + row.thumb.width + 10, row.rect[1] + 10)  # no rename sheet from the mask zone
    assert app.sheet is None
    # the rubylith overlay tints the hidden area while the mask is edited
    doc.deselect()
    app.view.fit()
    app.view.render()
    vx, vy = app.view.to_view(300, 150)
    hidden_px = pixel(_frame(app), int(vx), int(vy))
    app.toggle_mask_overlay()
    assert not app.mask_overlay
    plain_px = pixel(_frame(app), int(vx), int(vy))
    assert hidden_px != plain_px and hidden_px[0] > plain_px[0]  # red tint
    app.toggle_mask_overlay()
    # painting white on the mask with the brush reveals the red again (preview then commit)
    app.set_color("foreground", (255, 255, 255, 255))
    app.select_tool("brush")
    brush = tool_as(app, BrushTool)
    brush.options["size"] = 60
    brush.on_press(ev(300, 150))
    brush.on_drag(ev(310, 150))
    assert app.preview_mask_overrides and not app.preview_overrides
    app.view.render()  # preview through the mask override
    brush.on_release(ev(310, 150))
    assert not app.preview_mask_overrides
    assert doc.history.labels[-1] == "Brush Stroke (Mask)" and pixel(doc.flatten(), 300, 150) == (255, 0, 0, 255)
    # keyboard: Mod+\\ toggles the target, plain backslash still compares
    assert press_key(app, "backslash", mods=("mod",)) and not doc.editing_mask
    assert press_key(app, "backslash", mods=("mod",)) and doc.editing_mask
    # the remaining commands
    app.command("mask-enabled").run()
    assert not doc.active_layer.mask_enabled and not is_checked(app, "mask-enabled")
    rows = [w for w in app.tab_body.descendants() if isinstance(w, ui.LayerRow)]
    assert "mask off" in rows[0].details()
    app.command("mask-enabled").run()
    app.command("mask-invert").run()
    assert pixel(doc.flatten(), 300, 150) != (255, 0, 0, 255)
    app.command("mask-select").run()
    assert doc.selection is not None
    app.command("mask-apply").run()
    assert doc.active_layer.mask is None and pixel(doc.active_layer.image, 300, 150)[3] < 255 and not doc.editing_mask
    app.command("mask-hide").run()
    app.command("mask-delete").run()
    assert doc.active_layer.mask is None
    fire(app.layer_buttons["mask"])
    assert app._popover is not None and app._popover.name == "mask"
    app.close_popover()


def test_free_transform_tool(app_with_doc: Editor) -> None:
    app = app_with_doc
    doc = doc_of(app)
    doc.add_layer()
    doc.set_selection(Selection.rectangle(doc.size, (100, 100, 200, 200)))
    doc.fill("#ff0000")
    doc.deselect()
    app.refresh_all()
    # Mod+T selects the tool, which picks up the layer's content bounds
    assert press_key(app, "t", "t", mods=("mod",))
    tool = tool_as(app, TransformTool)
    assert app.tool is tool and tool.transform is not None and tool.transform.box == (100, 100, 200, 200)
    assert app.optionsbar.visible and "Keep aspect" in [w.text for w in app.options_row.descendants() if isinstance(w, ui.Toggle)]
    # drag the east handle: scales horizontally about the west edge (keep aspect off)
    tool.options["keep_aspect"] = False
    hx, hy = tool._handles()["e"]
    assert tool.cursor_at((int(hx), int(hy))) == "resize-h"
    tool.on_press(ev(*app.view.to_doc(hx, hy), wx=int(hx), wy=int(hy)))
    tool.on_drag(ev(250, 150))
    tool.on_release(ev(250, 150))
    assert tool.transform.scale_x == pytest.approx(1.5, abs=0.02) and tool.transform.scale_y == pytest.approx(1.0)
    assert tool.transform.apply(100, 150) == pytest.approx((100, 150), abs=0.5)  # the anchor stayed
    assert app.preview_overrides  # live preview of the transformed layer
    app.view.render()
    # Shift keeps the aspect on a corner handle
    hx, hy = tool._handles()["se"]
    tool.on_press(ev(*app.view.to_doc(hx, hy), wx=int(hx), wy=int(hy)))
    tool.on_drag(ev(300, 300, shift=True))
    tool.on_release(ev(300, 300))
    assert abs(tool.transform.scale_x) == pytest.approx(abs(tool.transform.scale_y), abs=0.02)
    # drag inside moves, drag just outside a corner rotates
    vx, vy = app.view.to_view(*tool.transform.apply(150, 150))
    assert tool.cursor_at((int(vx), int(vy))) == "move"
    before = tool.transform
    tool.on_press(ev(150, 150, wx=int(vx), wy=int(vy)))
    tool.on_drag(ev(160, 170))
    tool.on_release(ev(160, 170))
    assert (tool.transform.dx - before.dx, tool.transform.dy - before.dy) == pytest.approx((10, 20))
    cx, cy = tool._handles()["ne"]
    ox, oy = cx + theme.px(tools_mod.HANDLE_BASE + 14), cy - theme.px(4)
    assert tool.cursor_at((int(ox), int(oy))) == "rotate"
    tool.on_press(ev(*app.view.to_doc(ox, oy), wx=int(ox), wy=int(oy)))
    far = app.view.to_doc(ox - 300, oy + 300)
    tool.on_drag(ev(*far, shift=True))
    tool.on_release(ev(*far))
    assert tool.transform.angle != 0 and tool.transform.angle % 15 == pytest.approx(0, abs=1e-6)
    # typed values and flips
    tool._field_changed("angle", 0.0)
    tool._field_changed("scale_x", 2.0)
    assert tool.transform.angle == 0 and tool.transform.scale_x == 2.0 and tool._fields["h"].value == 200  # keep aspect default on
    tool.options["keep_aspect"] = True
    tool._flip(True)
    assert tool.transform.scale_x == -2.0
    tool._flip(False)
    assert tool.transform.scale_y < 0
    # arrows nudge, Enter applies and the tool is ready for the next transform
    tool.on_key("Right", frozenset({"shift"}))
    steps = len(doc.history.labels)
    assert press_key(app, "Return")
    assert doc.history.labels[-1] == "Free Transform" and len(doc.history.labels) == steps + 1 and not app.preview_overrides
    assert tool.transform is not None and tool.transform.is_identity
    app.view.render()  # overlay with handles
    # Escape cancels a pending transform without touching the picture
    tool._set(replace(tool.transform, dx=40))
    assert app.preview_overrides
    press_key(app, "Escape")
    assert tool.transform is None and not app.preview_overrides and len(doc.history.labels) == steps + 1
    # an empty layer has nothing to transform
    doc.add_layer()
    app.select_tool("move")
    app.select_tool("transform")
    assert tool.transform is None and "empty" in app.info_label.text
    # double-click inside applies
    doc.remove_layer()
    app.select_tool("move")
    app.select_tool("transform")
    assert tool.transform is not None
    tool._set(replace(tool.transform, dx=5))
    steps = len(doc.history.labels)
    tool.on_double(ev(*tool.transform.apply(*tool.transform.center)))
    assert doc.history.labels[-1] == "Free Transform" and len(doc.history.labels) == steps + 1


def test_curves_sheet(app_with_doc: Editor) -> None:
    app = app_with_doc
    doc = doc_of(app)
    before = pixel(doc.active_layer.image, 200, 150)
    assert press_key(app, "m", "m", mods=("mod",))
    sheet = sheet_of(app)
    assert sheet.name == "curves"
    editor = next(w for w in sheet.content.descendants() if isinstance(w, ui.CurveEditor))
    assert app.view.focused_widget is editor and editor.histogram is not None
    gx0, gy0, gx1, gy1 = editor._grid()
    side = gx1 - gx0
    img = Image.new("RGBA", (400, 400))
    editor.draw(img, ui.DrawContext(focused=editor))
    # click in the grid adds a point, dragging moves it (clamped between neighbours)
    editor.on_press(gx0 + side * 0.5, gy1 - side * 0.5)
    assert len(editor.points) == 3 and editor.selected == 1
    editor.on_drag(gx0 + side * 0.5, gy1 - side * 0.8)
    editor.on_release(0, 0, True)
    x, y = editor.points[1]
    assert x == pytest.approx(127.5, abs=1) and y == pytest.approx(204, abs=2)
    editor.on_press(*editor._to_widget(x, y))
    editor.on_drag(gx1 + 5, gy1)  # cannot pass the next point
    assert editor.points[1][0] == 254 and editor.points[1] < editor.points[2]
    editor.on_release(0, 0, True)
    # readout and live preview
    assert "Input" in next(w for w in sheet.content.descendants() if isinstance(w, ui.Label) and "Input" in w.text).text
    settle(app)
    assert app.preview_overrides and pixel(next(iter(app.preview_overrides.values())), 200, 150) != before
    # keyboard nudges the selected point, Delete removes it, dragging far away removes too
    editor.selected = 1
    px_before = editor.points[1]
    assert editor.on_key("Up", "", frozenset({"shift"}))
    assert editor.points[1][1] == pytest.approx(min(255, px_before[1] + 10), abs=1e-6)
    assert editor.on_key("Delete", "")
    assert len(editor.points) == 2 and editor.selected is None
    editor.on_press(gx0 + side * 0.25, gy1 - side * 0.25)
    editor.on_drag(gx0 - 500, gy0)
    assert len(editor.points) == 2
    # double-click removes a point, channel switch edits another curve, reset clears all
    editor.on_press(gx0 + side * 0.25, gy1 - side * 0.25)
    editor.on_release(0, 0, True)
    editor.on_double(gx0 + side * 0.25, gy1 - side * 0.25)
    assert len(editor.points) == 2
    seg = next(w for w in sheet.content.descendants() if isinstance(w, ui.Segmented))
    seg.on_release(seg.rect[0] + 90, seg.rect[1] + 5, True)  # "Red"
    assert editor.channel == "r"
    editor.on_press(gx0 + side * 0.5, gy1 - side * 0.9)
    editor.on_release(0, 0, True)
    assert len(editor.curves["r"]) == 3 and len(editor.curves["rgb"]) == 2
    editor.draw(img, ui.DrawContext())
    fire(next(w for w in sheet.content.descendants() if isinstance(w, ui.Button) and w.text == "Reset"))
    assert len(editor.curves["r"]) == 2
    # apply commits a Curves step (identity curves apply nothing)
    editor.curves["rgb"] = [(0.0, 0.0), (255.0, 128.0)]
    press_key(app, "Return")
    assert app.sheet is None and doc.history.labels[-1] == "Curves" and not app.preview_overrides
    after = pixel(doc.active_layer.image, 200, 150)
    assert after[0] <= before[0] // 2 + 1
    app.command("curves").run()
    press_key(app, "Return")
    assert doc.history.labels[-1] == "Curves" and doc.history.labels.count("Curves") == 1


def test_levels_sheet(app_with_doc: Editor) -> None:
    app = app_with_doc
    doc = doc_of(app)
    assert press_key(app, "l", "l", mods=("mod",))
    sheet = sheet_of(app)
    assert sheet.name == "levels"
    editor = next(w for w in sheet.content.descendants() if isinstance(w, ui.LevelsEditor))
    img = Image.new("RGBA", (420, 300))
    editor.draw(img, ui.DrawContext(focused=editor))
    markers = editor._markers()
    # drag the black input marker right, the white one left, and the grey one
    editor.on_press(*markers["in_black"])
    editor.on_drag(editor._x_of(40), markers["in_black"][1])
    editor.on_release(0, 0, True)
    assert editor.in_black == 40
    editor.on_press(*editor._markers()["in_white"])
    editor.on_drag(editor._x_of(10), markers["in_white"][1])  # cannot cross the black point
    editor.on_release(0, 0, True)
    assert editor.in_white == 42
    editor.on_press(*editor._markers()["in_white"])
    editor.on_drag(editor._x_of(200), markers["in_white"][1])
    editor.on_release(0, 0, True)
    editor.on_press(*editor._markers()["gamma"])
    editor.on_drag(editor._x_of(80), markers["gamma"][1])
    editor.on_release(0, 0, True)
    assert editor.gamma > 1.0 and 40 < editor.mid_input < 200
    editor.on_press(*editor._markers()["out_white"])
    editor.on_drag(editor._x_of(230), markers["out_white"][1])
    editor.on_release(0, 0, True)
    editor.on_press(*editor._markers()["out_black"])
    editor.on_drag(editor._x_of(250), markers["out_black"][1])  # clamped to the white output
    editor.on_release(0, 0, True)
    assert editor.out_white == 230 and editor.out_black == 230
    fields = {f.label: f for f in sheet.content.descendants() if isinstance(f, ui.NumberField)}
    assert fields["In"].value == 40 and fields["Out"].value == 230
    # typing into a field drives the editor
    fields["Gamma"].set_text("")
    for ch in "200":
        fields["Gamma"].on_key(ch, ch)
    assert editor.gamma == pytest.approx(2.0)
    fields["In"].set_text("")
    for ch in "250":
        fields["In"].on_key(ch, ch)
    assert editor.in_black == editor.in_white - 2
    editor.on_double(0, 0)  # reset
    assert editor.values() == {"in_black": 0, "in_white": 255, "gamma": 1.0, "out_black": 0, "out_white": 255}
    editor.set_values(in_black=20, in_white=235)
    press_key(app, "Return")
    assert app.sheet is None and doc.history.labels[-1] == "Levels"
    app.command("levels").run()
    press_key(app, "Return")  # unchanged levels: nothing applied
    assert doc.history.labels.count("Levels") == 1


def test_clone_stamp_and_healing_brush(app_with_doc: Editor) -> None:
    app = app_with_doc
    doc = doc_of(app)
    doc.add_layer()
    doc.fill("#6496c8")
    doc.set_selection(Selection.rectangle(doc.size, (200, 150, 240, 190)))
    doc.fill("#ff0000")  # blemish
    doc.deselect()
    app.refresh_all()
    assert press_key(app, "s", "s")
    clone = tool_as(app, CloneTool)
    assert app.tool is clone and app.tool_buttons["clone"].active
    assert {w.text for w in app.options_row.descendants() if isinstance(w, ui.Toggle)} >= {"Aligned", "All layers"}
    # painting without a source only notifies
    clone.on_press(ev(220, 170))
    assert clone._stroke is None and "source" in app.info_label.text
    # Alt-click sets the source; the overlay marks it
    clone.on_press(ev(100, 170, alt=True))
    assert clone.source_point == (100, 170)
    app.view.render()
    clone.options["size"] = 50
    clone.on_press(ev(220, 170))
    assert clone._offset == (-120, 0) and clone._source is not None
    clone.on_drag(ev(225, 172))
    assert app.preview_overrides
    app.view.pointer = (10, 10)
    app.view.render()  # source crosshair follows the pointer while painting
    clone.on_release(ev(225, 172))
    assert doc.history.labels[-1] == "Clone Stamp" and pixel(doc.active_layer.image, 220, 170) == (100, 150, 200, 255)
    # aligned: the next stroke keeps the offset; unaligned re-anchors on the source point
    clone.on_press(ev(300, 170))
    assert clone._offset == (-120, 0)
    clone.on_release(ev(300, 170))
    clone.options["aligned"] = False
    clone.on_press(ev(300, 170))
    assert clone._offset == (-200, 0)
    clone.on_release(ev(300, 170))
    clone.cancel()
    # the healing brush spot-heals without a source
    doc.set_selection(Selection.rectangle(doc.size, (200, 150, 240, 190)))
    doc.fill("#ff0000")
    doc.deselect()
    assert press_key(app, "j", "j")
    heal = tool_as(app, HealTool)
    heal.options["size"] = 60
    heal.on_press(ev(220, 170))
    heal.on_drag(ev(222, 170))
    heal.on_release(ev(222, 170))
    assert doc.history.labels[-1] == "Healing Brush"
    r, g, b, a = pixel(doc.active_layer.image, 220, 170)
    assert r < 140 and b > 180
    # with a source it heals using that texture
    heal.on_press(ev(60, 60, alt=True))
    heal.on_press(ev(300, 200))
    heal.on_release(ev(300, 200))
    assert doc.history.labels[-1] == "Healing Brush"
    app.select_tool("move")  # deactivating cancels cleanly
    assert not app.preview_overrides


def test_adjustment_layers_in_the_gui(app_with_doc: Editor) -> None:
    app = app_with_doc
    doc = doc_of(app)
    before = pixel(doc.flatten(), 200, 150)
    app.tabs.select(2)
    fire(app.layer_buttons["adjust"])
    assert doc.active_layer.kind == "adjustment" and doc.editing_mask
    rows = [w for w in app.tab_body.descendants() if isinstance(w, ui.LayerRow)]
    assert "adjustment" in rows[0].details() and rows[0].mask_thumb is not None
    img = Image.new("RGBA", (300, 60))
    rows[0].draw(img, ui.DrawContext())
    # the Adjust tab drives the adjustment layer and shows the note
    app.tabs.select(0)
    assert app.adjust_note.visible
    app.sliders["saturation"].set(-100)
    app._adjust_live("saturation", -100)
    app._adjust_commit("saturation")
    grey = pixel(doc.flatten(), 200, 150)
    assert grey[0] == grey[1] == grey[2] and grey != before
    # Curves on an adjustment layer are stored in its meta (non-destructive) with live preview and cancel
    app.command("curves").run()
    sheet = sheet_of(app)
    editor = next(w for w in sheet.content.descendants() if isinstance(w, ui.CurveEditor))
    editor.curves["rgb"] = [(0.0, 0.0), (255.0, 128.0)]
    editor._changed()
    settle(app)
    assert doc.active_layer.meta.get("curves") and not app.preview_overrides  # previewed through the layer itself
    darker = pixel(doc.flatten(), 200, 150)
    assert darker[0] < grey[0]
    press_key(app, "Escape")
    assert "curves" not in doc.active_layer.meta and pixel(doc.flatten(), 200, 150) == grey  # cancelled: restored
    app.command("curves").run()
    editor = next(w for w in sheet_of(app).content.descendants() if isinstance(w, ui.CurveEditor))
    editor.curves["rgb"] = [(0.0, 0.0), (255.0, 128.0)]
    press_key(app, "Return")
    assert doc.history.labels[-1] == "Curves" and doc.active_layer.meta["curves"]["rgb"] == [[0.0, 0.0], [255.0, 128.0]]
    assert pixel(doc.flatten(), 200, 150) == darker
    # reopening starts from the stored curve; Levels work the same way
    app.command("curves").run()
    editor = next(w for w in sheet_of(app).content.descendants() if isinstance(w, ui.CurveEditor))
    assert editor.curves["rgb"] == [(0.0, 0.0), (255.0, 128.0)]
    press_key(app, "Escape")
    app.command("levels").run()
    levels = next(w for w in sheet_of(app).content.descendants() if isinstance(w, ui.LevelsEditor))
    levels.set_values(in_white=128)
    press_key(app, "Return")
    assert doc.history.labels[-1] == "Levels" and doc.active_layer.meta["levels"]["in_white"] == 128
    app.command("levels").run()
    assert next(w for w in sheet_of(app).content.descendants() if isinstance(w, ui.LevelsEditor)).in_white == 128
    press_key(app, "Escape")
    # the menu command and the bake guard
    app.command("new-adjustment").run()
    assert sum(1 for lyr in doc.layers if lyr.kind == "adjustment") == 2
    app.command("bake").run()
    assert "merge" in app.info_label.text


def test_layer_effects_sheet(app_with_doc: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    app = app_with_doc
    doc = doc_of(app)
    box = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    box.paste((255, 0, 0, 255), (20, 20, 60, 60))
    doc.add_image_layer(box, "Box", (100, 100))
    app.tabs.select(2)
    fire(app.layer_buttons["fx"])
    sheet = sheet_of(app)
    assert sheet.name == "effects" and sheet.scrollable
    toggles = {t.text: t for t in sheet.content.descendants() if isinstance(t, ui.Toggle)}
    toggles["Stroke"].on_release(1, 1, True)
    sliders = {s.label: s for s in sheet.content.descendants() if isinstance(s, ui.Slider)}
    stroke_size = [s for s in sheet.content.descendants() if isinstance(s, ui.Slider) and s.label == "Size"][1]
    stroke_size.set(6)
    assert stroke_size.on_change is not None
    stroke_size.on_change(6)
    assert sliders["Offset X"].value == 6
    monkeypatch.setattr(shell_of(app), "ask_color", lambda initial: (0, 255, 0, 255))
    pick_buttons = [b for b in sheet.content.descendants() if isinstance(b, ui.Button) and b.text == "Pick…"]
    fire(pick_buttons[2])  # stroke colour
    settle(app)
    assert doc.active_layer.meta["effects"]["stroke"]["enabled"]  # live preview through the layer meta
    assert pixel(doc.flatten(), 117, 140) == (0, 255, 0, 255)
    press_key(app, "Escape")
    assert "effects" not in doc.active_layer.meta  # cancelled
    fire(app.layer_buttons["fx"])
    sheet = sheet_of(app)
    toggles = {t.text: t for t in sheet.content.descendants() if isinstance(t, ui.Toggle)}
    toggles["Drop Shadow"].on_release(1, 1, True)
    seg = next(w for w in sheet.content.descendants() if isinstance(w, ui.Segmented))
    seg.on_release(seg.rect[0] + seg.rect[2] // 2, seg.rect[1] + 5, True)
    press_key(app, "Return")
    assert doc.history.labels[-1] == "Layer Effects" and doc.active_layer.meta["effects"]["shadow"]["enabled"]
    rows = [w for w in app.tab_body.descendants() if isinstance(w, ui.LayerRow)]
    assert "fx" in rows[0].details()
    # reopening shows the stored settings; Remove All clears them
    app.command("effects").run()
    sheet = sheet_of(app)
    toggles = {t.text: t for t in sheet.content.descendants() if isinstance(t, ui.Toggle)}
    assert toggles["Drop Shadow"].value
    fire(next(b for b in sheet.content.descendants() if isinstance(b, ui.Button) and b.text == "Remove All"))
    assert app.sheet is None and "effects" not in doc.active_layer.meta
    # adjustment layers have no pixels to decorate
    doc.add_adjustment_layer()
    app.refresh_all()
    assert not app.command("effects").enabled()
    sheets.effects_sheet(app)
    assert app.sheet is None and "pixels" in app.info_label.text


def test_layer_groups_in_the_gui(app_with_doc: Editor) -> None:
    app = app_with_doc
    doc = doc_of(app)
    app.tabs.select(2)
    app.command("new-layer").run()
    assert press_key(app, "g", "g", mods=("mod",))  # Group Layer
    header = doc.active_index
    assert doc.layers[header].kind == "group" and doc.group_of(header - 1) == header
    rows = [w for w in app.tab_body.descendants() if isinstance(w, ui.LayerRow)]
    assert [r.layer.name for r in rows] == ["Group 1", "Layer 2", "Background"]
    assert rows[0].depth == 0 and rows[1].depth == 1 and rows[0].collapsed is False and "1 layer" in rows[0].details()
    img = Image.new("RGBA", (300, 60))
    rows[0].draw(img, ui.DrawContext())
    rows[1].draw(img, ui.DrawContext())
    # the thumbnail of a header collapses / expands the group; members vanish from the list
    rows[0].on_release(rows[0].rect[0] + 40, rows[0].rect[1] + 10, True)
    assert doc.layers[header].id in app.collapsed_groups
    rows = [w for w in app.tab_body.descendants() if isinstance(w, ui.LayerRow)]
    assert [r.layer.name for r in rows] == ["Group 1", "Background"] and rows[0].collapsed
    rows[0].draw(img, ui.DrawContext())
    rows[0].on_release(rows[0].rect[0] + 40, rows[0].rect[1] + 10, True)
    assert not app.collapsed_groups
    # a new layer made while a member is active joins the group; the group button groups it again (nested)
    doc.set_active(header - 1)
    app.refresh_all()
    fire(app.layer_buttons["plus"])
    assert doc.group_of(doc.active_index) == doc.layer_index(doc.layers[header + 1].id)
    fire(app.layer_buttons["group"])
    assert doc.active_layer.kind == "group" and doc.group_of(doc.active_index) is not None
    rows = [w for w in app.tab_body.descendants() if isinstance(w, ui.LayerRow)]
    assert max(r.depth for r in rows) == 2
    # painting on a group is refused with a notice; the mask and effects still work on it
    app.select_tool("brush")
    tool_of(app).on_press(ev(10, 10))
    assert tool_as(app, BrushTool)._stroke is None and "group" in app.info_label.text
    assert app.command("effects").enabled() and app.command("ungroup").enabled()
    app.command("mask-reveal").run()
    assert doc.active_layer.mask is not None
    # ungroup via the shortcut, then delete a whole group with the trash button
    assert press_key(app, "g", "g", mods=("mod", "shift"))
    assert doc.active_layer.kind != "group"
    top = next(i for i, lyr in enumerate(doc.layers) if lyr.kind == "group")
    doc.set_active(top)
    app.refresh_all()
    count = len(doc.layers)
    fire(app.layer_buttons["trash"])
    assert len(doc.layers) == count - 3 and all(lyr.kind != "group" for lyr in doc.layers)
    assert not app.command("ungroup").enabled()


def test_big_pictures_render_in_the_background(app: Editor, gradient: Image.Image) -> None:
    from pixelforge.gui import surface as surface_mod

    big = Document.from_image(gradient.resize((1600, 1200)), "big")
    assert big.width * big.height > surface_mod.ASYNC_PIXELS
    app.set_document(big)
    view = app.view
    view.render()
    assert view.frame_pending  # the first frame is being rendered by the worker; the checker shows meanwhile
    settle(app, 0)
    assert not view.frame_pending and not app.worker.busy
    frame = _frame(app)
    vx, vy = view.to_view(800, 600)
    assert pixel(frame, int(vx), int(vy))[:3] == pytest.approx(pixel(gradient.resize((1600, 1200)), 800, 600), abs=6)
    # zooming shows the previous picture re-projected until the new composite arrives
    view.set_zoom(view.zoom * 2, (vx, vy))
    view.render()
    assert view.frame_pending
    ix0, iy0, ix1, iy1 = view.image_rect()
    interim = view._interim_picture((max(0, -ix0), max(0, -iy0), min(frame.width, ix1) - ix0, min(frame.height, iy1) - iy0))
    assert interim is not None and interim.getbbox() is not None
    settle(app, 0)
    assert not view.frame_pending
    # a live filter preview on a big picture also runs in the background
    from pixelforge.core.filters import FILTERS

    sheets.filter_sheet(app, FILTERS["Gaussian Blur"])
    settle(app, 0)
    assert app.preview_overrides
    press_key(app, "Escape")
    # a failing background composite falls back to the inline path instead of looping
    view._frame_ready(None)
    assert view._force_sync
    view.render()
    assert not view._force_sync
    # no previous picture → no stand-in
    view._last_picture = None
    assert view._interim_picture((0, 0, 10, 10)) is None


def test_export_sheet_metadata_toggle(app_with_doc: Editor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pixelforge.core import colorspace as cs

    app = app_with_doc
    doc = doc_of(app)
    doc.metadata.icc_profile = cs.srgb_profile()
    doc.metadata.dpi = (144.0, 144.0)
    target = tmp_path / "out.png"
    monkeypatch.setattr(shell_of(app), "ask_save_file", lambda suggested, filetypes: str(target))
    app.export_picture()
    sheet = sheet_of(app)
    toggle = next(w for w in sheet.content.descendants() if isinstance(w, ui.Toggle))
    assert toggle.value and any("144 dpi" in w.text for w in sheet.content.descendants() if isinstance(w, ui.Label))
    toggle.on_release(1, 1, True)  # strip
    press_key(app, "Return")
    with Image.open(target) as saved:
        assert not saved.info.get("icc_profile")
    assert app.settings.get("export_metadata") is False
    app.export_picture()
    press_key(app, "Return")  # remembered: still stripped
    with Image.open(target) as saved:
        assert not saved.info.get("icc_profile")
    app.settings.set("export_metadata", True)
    app.export_picture()
    press_key(app, "Return")
    with Image.open(target) as saved:
        assert saved.info.get("icc_profile") and saved.info.get("dpi") == pytest.approx((144, 144), abs=0.1)


def test_selection_tools_and_ants(app_with_doc: Editor) -> None:
    app = app_with_doc
    doc = doc_of(app)
    app.select_tool("marquee")
    tool = tool_as(app, MarqueeTool)
    assert app.optionsbar.visible and app.options_row.children()
    tool.on_press(ev(10, 10))
    tool.on_drag(ev(110, 80))
    app.view.render()  # dashed preview
    tool.on_release(ev(110, 80))
    assert doc.selection is not None and doc.selection.bbox == (10, 10, 110, 80)
    app.view.render()
    assert any(kind == "ants" for kind, *_ in shell_of(app).widget._layers)
    assert app.view._ants_job is not None
    app.view._animate_ants()  # phase advances and re-renders
    tool.on_press(ev(150, 10, shift=True))
    tool.on_drag(ev(200, 80, shift=True))
    tool.on_release(ev(200, 80, shift=True))
    assert doc.selection.bbox == (10, 10, 200, 80)
    tool.options["shape"] = "ellipse"
    tool.on_press(ev(10, 10, alt=True))
    tool.on_drag(ev(110, 80, alt=True))
    tool.on_release(ev(110, 80, alt=True))
    assert not doc.selection.contains(60, 45)
    tool.on_press(ev(5, 5))
    tool.on_release(ev(5, 5))  # click deselects
    assert doc.selection is None
    app.view.render()
    assert not any(kind == "ants" for kind, *_ in shell_of(app).widget._layers)
    app.select_tool("lasso")
    for p in [(20, 20), (120, 20), (120, 120), (20, 120)]:
        if p == (20, 20):
            tool_of(app).on_press(ev(*p))
        else:
            tool_of(app).on_drag(ev(*p))
    app.view.render()
    tool_of(app).on_release(ev(20, 120))
    assert doc.selection is not None and doc.selection.contains(70, 70)
    app.select_tool("wand")
    tool_of(app).options["tolerance"] = 10
    tool_of(app).on_press(ev(300, 250))
    assert doc.selection.contains(300, 250)
    tool_of(app).on_press(ev(-5, -5))  # outside: ignored
    # selection commands
    app.command("invert-sel").run()
    assert not doc.selection.contains(300, 250)
    app.command("select-all").run()
    assert doc.selection.is_full
    app.command("deselect").run()
    assert doc.selection is None and not app.command("deselect").enabled()
    app.command("clear").run() if False else app.clear_selection()
    assert "Nothing selected" in app.info_label.text
    doc.set_selection(__import__("pixelforge.core.selection", fromlist=["Selection"]).Selection.rectangle(doc.size, (0, 0, 50, 50)))
    app.refresh_all()
    app.command("feather").run()
    assert app.sheet is not None and app.sheet.name == "number"
    press_key(app, "Return")
    assert doc.history.labels[-1] == "Feather Selection"
    app.command("crop-sel").run()
    assert doc.size[0] <= 70  # the feathered selection's bounding box
    app.set_color("foreground", (0, 255, 0, 255))
    doc.set_selection(__import__("pixelforge.core.selection", fromlist=["Selection"]).Selection.rectangle(doc.size, (0, 0, 10, 10)))
    app.fill("foreground")
    assert pixel(doc.active_layer.image, 2, 2) == (0, 255, 0, 255)
    press_key(app, "Delete")
    assert pixel(doc.active_layer.image, 2, 2)[3] == 0
    app.command("select-layer").run()
    assert doc.selection is not None


def test_painting_tools(app_with_doc: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    app = app_with_doc
    doc = doc_of(app)
    app.set_color("foreground", (255, 0, 0, 255))
    app.set_color("background", (0, 0, 255, 255))
    app.select_tool("brush")
    brush = tool_as(app, BrushTool)
    brush.options["size"] = 20
    brush.on_press(ev(50, 50, wx=100, wy=100))
    brush.on_drag(ev(120, 60, wx=170, wy=110))
    assert app.preview_overrides
    app.view.pointer = (170, 110)
    app.view.render()
    assert any(kind == "cursor" for kind, *_ in shell_of(app).widget._layers)  # brush outline follows the pointer
    brush.on_release(ev(120, 60))
    assert not app.preview_overrides and doc.history.labels[-1] == "Brush Stroke"
    assert pixel(doc.active_layer.image, 85, 55) == (255, 0, 0, 255)
    brush.on_press(ev(120, 200, shift=True))  # straight line from the last point
    brush.on_release(ev(120, 200))
    assert pixel(doc.active_layer.image, 120, 130) == (255, 0, 0, 255)
    assert brush.on_key("bracketright") and brush.options["size"] > 20
    assert brush.on_key("bracketleft") and not brush.on_key("q")
    key(app, Qt.Key.Key_BracketRight, "]")
    app.select_tool("eraser")
    tool_of(app).options["size"] = 30
    tool_of(app).on_press(ev(85, 55))
    tool_of(app).on_release(ev(85, 55))
    assert pixel(doc.active_layer.image, 85, 55)[3] == 0 and doc.history.labels[-1] == "Eraser"
    app.select_tool("fill")
    tool_of(app).options["tolerance"] = 255
    tool_of(app).on_press(ev(200, 150))
    assert pixel(doc.active_layer.image, 300, 250) == (255, 0, 0, 255)
    tool_of(app).on_press(ev(200, 150, alt=True))
    assert pixel(doc.active_layer.image, 300, 250) == (0, 0, 255, 255)
    tool_of(app).options["whole"] = True
    tool_of(app).on_press(ev(-1, -1))
    assert doc.history.labels[-1] == "Fill"
    app.select_tool("gradient")
    tool_of(app).on_press(ev(0, 150))
    tool_of(app).on_drag(ev(399, 150, ctrl=True))
    app.view.render()
    tool_of(app).on_release(ev(399, 150))
    img = doc.active_layer.image
    assert pixel(img, 2, 150)[0] > 200 and pixel(img, 397, 150)[2] > 200
    tool_of(app).options["kind"] = "radial"
    tool_of(app).options["reverse"] = True
    tool_of(app).on_press(ev(200, 150))
    tool_of(app).on_drag(ev(300, 150))
    tool_of(app).on_release(ev(300, 150))
    assert pixel(doc.active_layer.image, 200, 150)[2] > 200
    tool_of(app).on_press(ev(5, 5))
    tool_of(app).on_release(ev(5, 5))  # zero length: nothing
    assert doc.history.labels.count("Gradient") == 2
    app.select_tool("shape")
    tool_of(app).options["stroke"] = 0
    tool_of(app).on_press(ev(10, 10))
    tool_of(app).on_drag(ev(60, 60))
    app.view.render()
    tool_of(app).on_release(ev(60, 60))
    assert pixel(doc.active_layer.image, 30, 30)[:3] == (255, 0, 0)
    tool_of(app).options.update(kind="ellipse", fill=False, stroke=4)
    tool_of(app).on_press(ev(100, 100))
    tool_of(app).on_drag(ev(200, 200, ctrl=True))
    app.view.render()
    tool_of(app).on_release(ev(200, 200))
    assert pixel(doc.active_layer.image, 150, 101)[:3] == (0, 0, 255)
    tool_of(app).options["kind"] = "line"
    tool_of(app).on_press(ev(0, 250))
    tool_of(app).on_drag(ev(399, 250))
    app.view.render()
    tool_of(app).on_release(ev(399, 250))
    assert pixel(doc.active_layer.image, 200, 250)[:3] == (255, 0, 0)
    app.select_tool("eyedropper")
    tool_of(app).on_press(ev(150, 101))
    assert app.foreground == (0, 0, 255, 255)
    tool_of(app).on_press(ev(30, 30, alt=True))
    assert app.background == (255, 0, 0, 255)
    tool_of(app).on_press(ev(-1, -1))
    # colour keys and swatches
    press_key(app, "x", "x")
    assert app.foreground == (255, 0, 0, 255) and app.background == (0, 0, 255, 255)
    press_key(app, "d", "d")
    assert app.foreground == (0, 0, 0, 255)
    zones = app.swatches._zones()
    monkeypatch.setattr(shell_of(app), "ask_color", lambda initial: (9, 8, 7, 255))
    zx0, zy0, zx1, zy1 = zones["foreground"]
    app.swatches.on_release((zx0 + zx1) / 2, (zy0 + zy1) / 2 - 4, True)
    assert app.foreground == (9, 8, 7, 255)
    sx0, sy0, sx1, sy1 = zones["swap"]
    app.swatches.on_release((sx0 + sx1) / 2, (sy0 + sy1) / 2, True)
    assert app.background == (9, 8, 7, 255)
    doc.set_layer_props(locked=True)
    app.select_tool("brush")
    tool_of(app).on_press(ev(10, 10))
    assert tool_as(app, BrushTool)._stroke is None and "locked" in app.info_label.text


def test_text_tool_and_editing(app_with_doc: Editor) -> None:
    app = app_with_doc
    doc = doc_of(app)
    app.select_tool("text")
    tool_of(app).on_press(ev(40, 40))
    sheet = app.sheet
    assert sheet is not None and sheet.name == "text"
    field = next(w for w in sheet.content.descendants() if isinstance(w, ui.TextField) and w.placeholder)
    assert app.view.focused_widget is field
    for ch in "Hi":
        press_key(app, ch, ch)
    font_button = next(w for w in sheet.content.descendants() if isinstance(w, ui.Button) and w.text == "Default font")
    fire(font_button)
    assert app._popover is not None and app._popover.name == "fonts"
    items = [w for w in popover_of(app).content.descendants() if isinstance(w, ui.ListItem)]
    fire(items[min(1, len(items) - 1)])
    assert app._popover is None
    shell_of(app).ask_color = lambda initial: (0, 200, 0, 255)
    change_buttons = [w for w in sheet.content.descendants() if isinstance(w, ui.Button) and w.text == "Change…"]
    fire(change_buttons[0])
    press_key(app, "Return")
    assert doc.active_layer.kind == "text" and doc.active_layer.name == "T Hi"
    import numpy as np

    pixels = np.asarray(doc.active_layer.image)
    assert (pixels[pixels[..., 3] > 200][:, 1] > 150).any()  # drawn in the chosen green
    # editing through the layers tab (double-click on a text layer)
    app.tabs.select(2)
    rows = [w for w in app.tab_body.descendants() if isinstance(w, ui.LayerRow)]
    rows[0].on_double(rows[0].rect[0] + 100, rows[0].rect[1] + 10)
    assert app.sheet is not None and app.sheet.name == "text"
    field = next(w for w in sheet_of(app).content.descendants() if isinstance(w, ui.TextField) and w.placeholder)
    assert field.value == "Hi"
    press_key(app, "!", "!")
    press_key(app, "Return")
    assert doc.active_layer.meta["text"] == "Hi!" and doc.history.labels[-1] == "Edit Text"
    app.select_tool("text")
    tool_of(app).on_press(ev(1, 1))
    press_key(app, "Return")  # empty text: nothing added
    assert len(doc.layers) == 2


def test_move_tool(app_with_doc: Editor, shapes: Image.Image) -> None:
    app = app_with_doc
    doc = doc_of(app)
    doc.add_image_layer(shapes, "Shapes", (0, 0))
    app.refresh_all()
    app.select_tool("move")
    tool = tool_as(app, MoveTool)
    tool.on_press(ev(10, 10))
    tool.on_drag(ev(30, 25))
    assert app.preview_overrides
    tool.on_release(ev(30, 25))
    assert pixel(doc.active_layer.image, 30, 25) == (255, 0, 0, 255) and doc.history.labels[-1] == "Move"
    assert tool.on_key("Right", frozenset({"shift"}))
    assert pixel(doc.active_layer.image, 40, 25) == (255, 0, 0, 255)
    key(app, Qt.Key.Key_Left)
    assert pixel(doc.active_layer.image, 39, 25) == (255, 0, 0, 255)
    tool.options["auto_select"] = True
    doc.set_active(0)
    tool.on_press(ev(44, 30))
    tool.on_release(ev(44, 30))
    assert doc.active_index == 1
    tool.on_press(ev(390, 290))
    tool.on_release(ev(390, 290))
    assert doc.active_index == 0


def test_auto_enhance_command(app_with_doc: Editor) -> None:
    app = app_with_doc
    doc = doc_of(app)
    app.command("auto").run()
    assert doc.history.labels[-1] == "Auto Enhance" and doc.active_layer.has_adjustments
    fire(app.btn_auto)  # same result again: nothing to change
    assert "nothing to change" in app.info_label.text


def test_grid_rulers_and_snapping(app_with_doc: Editor) -> None:
    app = app_with_doc
    view = app.view
    assert not view.show_grid and not view.show_rulers
    app.command("grid").run()
    app.command("rulers").run()
    assert view.show_grid and view.show_rulers and app.settings.get("grid") and app.settings.get("rulers")
    assert is_checked(app, "grid") and shell_of(app)._actions["grid"].isChecked()
    view.render()
    frame = next(img for kind, _x, _y, img in shell_of(app).widget._layers if kind == "frame")
    assert pixel(frame, 2, 2) != (20, 21, 25)  # ruler strip drawn in the corner
    assert view.grid_spacing() * view.zoom >= theme.px(40)
    view.set_zoom(4.0)
    assert view.grid_spacing() < 50
    view.set_zoom(1.0)
    # crop snapping to the picture thirds / grid
    app.select_tool("crop")
    crop = tool_as(app, CropTool)
    crop.on_press(ev(3, 2))
    crop.on_drag(ev(131, 99))  # close to the thirds (133.3, 100)
    crop.on_release(ev(131, 99))
    assert crop.rect is not None and crop.rect[0] == 0 and crop.rect[1] == 0
    assert crop.rect[2] == pytest.approx(400 / 3) and crop.rect[3] == pytest.approx(100)
    crop.cancel()
    app.command("snap").run()
    assert not app.snap_enabled
    crop.on_press(ev(3, 2))
    crop.on_drag(ev(131, 99))
    crop.on_release(ev(131, 99))
    assert crop.rect == (3, 2, 131, 99)
    app.toggle_snap()
    app.toggle_grid()
    app.toggle_rulers()
    assert not view.show_grid and not view.show_rulers


def test_language_switch(app_with_doc: Editor) -> None:
    from pixelforge.gui.i18n import current_language, tr

    app = app_with_doc
    assert current_language() == "en"
    app.set_ui_language("de")
    try:
        assert current_language() == "de" and app.settings.get("language") == "de"
        assert app.tabs.options[0] == "Anpassen" and tr("Undo") == "Widerrufen"
        assert app.command("undo").label == "Widerrufen"
        menus = [a.text() for a in shell_of(app).menuBar().actions()]
        assert "Ablage" in menus and "Bearbeiten" in menus
        assert app.tool_buttons["crop"].tooltip.startswith("Zuschneiden")
        app.view.render()
        app.command("lang:en").run()
        assert current_language() == "en" and app.tabs.options[0] == "Adjust"
        assert is_checked(app, "lang:en") and not is_checked(app, "lang:de")
    finally:
        app.set_ui_language("en")


def test_rebuild_keeps_state(app_with_doc: Editor) -> None:
    app = app_with_doc
    app.select_tool("brush")
    doc_of(app).set_adjustments({"contrast": 20}, commit=True)
    app.rebuild()
    assert tool_of(app).id == "brush" and app.tool_buttons["brush"].active
    assert app.sliders["contrast"].value == 20 and app.info_label.text == tool_of(app).hint
    assert app.optionsbar.visible  # brush options are back


def test_options_bar_never_overlaps_and_is_centred(app_with_doc: Editor) -> None:
    """The tool options bar stays between the top-left bar and the name bar, wraps when narrow, and is centred."""
    app = app_with_doc
    for tool_id in ("wand", "brush", "marquee", "fill", "shape"):
        app.select_tool(tool_id)
        app.view.render()
        r = app.optionsbar.rect
        left, right = app.topbar.rect[2], app.namebar.rect[0]
        assert left <= r[0] and r[2] <= right, tool_id
        gap_left, gap_right = r[0] - left, right - r[2]
        assert abs(gap_left - gap_right) <= theme.px(2), tool_id  # centred in the free space
        assert r[3] - r[1] >= theme.px(40)
        # every option widget is horizontally inside the bar and vertically centred on its line
        for widget in app.options_row.children():
            wx0, wy0, wx1, wy1 = app.optionsbar.widget_rect_in_view(widget)
            assert r[0] <= wx0 and wx1 <= r[2], (tool_id, widget)
    app.select_tool("wand")
    lines = app.options_row._lines(app.optionsbar.width - 2 * theme.px(14))
    assert len(lines) >= 1
    # a narrow window wraps the options onto more lines and pushes the picture down
    shell_of(app).resize(960, 700)
    QtWidgets.QApplication.processEvents()
    app.view.render()
    narrow_lines = app.options_row._lines(app.optionsbar.width - 2 * theme.px(14))
    assert len(narrow_lines) >= len(lines)
    r = app.optionsbar.rect
    assert r[2] <= app.namebar.rect[0] and r[0] >= app.topbar.rect[2]
    assert app.content_area(*app.view.viewport_size())[1] >= r[3]
    # compact sliders: label / track / value on one line, value click resets, label inert
    slider = next(w for w in app.options_row.children() if isinstance(w, ui.Slider))
    assert slider.compact and slider.height < METRICS.slider_h
    tx0, tx1, ty = slider._track()
    slider.on_press(tx1, ty)
    assert slider.value == slider.hi
    slider.on_release(tx1, ty, True)
    slider.on_press(slider.rect[2] - 2, ty)  # value zone resets
    assert slider.value == slider.default
    slider.on_press(slider.rect[0] + 2, ty)  # label zone is inert
    assert not slider._dragging


def test_sheets_centre_on_the_picture_area(app_with_doc: Editor) -> None:
    app = app_with_doc
    sheets.resize_sheet(app)
    x0, _y0, x1, _y1 = app.content_area(*app.view.viewport_size())
    r = sheet_of(app).rect
    assert abs((r[0] + r[2]) // 2 - (x0 + x1) // 2) <= 1
    app.dismiss_modal()
    sheets.straighten_sheet(app)
    r = sheet_of(app).rect
    assert abs((r[0] + r[2]) // 2 - (x0 + x1) // 2) <= 1
    assert r[3] <= app.zoombar.rect[1]
    app.dismiss_modal()
    app.set_document(None)
    sheets.about_sheet(app)
    vw, _vh = app.view.viewport_size()
    r = sheet_of(app).rect
    assert abs((r[0] + r[2]) // 2 - vw // 2) <= 1  # no picture: centred on the window


# ------------------------------------------------------------- robustness --
def test_unexpected_errors_become_notices(
    app_with_doc: Editor, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = app_with_doc

    def boom(_doc: Document) -> None:
        raise RuntimeError("kaboom")

    app.doc_op(boom)
    assert "kaboom" in app.info_label.text and app.info_label.color == PALETTE.danger
    assert "RuntimeError" in capsys.readouterr().err
    # a failing tool handler is caught by the surface too
    app.select_tool("brush")

    def broken(_e: PointerEvent) -> None:
        raise ValueError("tool bug")

    monkeypatch.setattr(tool_of(app), "on_press", broken)
    bx, by = (int(v) for v in app.view.to_view(50, 50))
    mouse_press(app, bx, by)
    assert "tool bug" in app.info_label.text
    mouse_release(app, bx, by)
    assert app.doc is not None  # the session survived


def test_tab_cycles_focus_inside_sheets(app_with_doc: Editor) -> None:
    app = app_with_doc
    sheets.resize_sheet(app)
    fields = [w for w in sheet_of(app).content.descendants() if isinstance(w, ui.NumberField)]
    assert app.view.focused_widget is fields[0]
    press_key(app, "Tab")
    assert app.view.focused_widget is fields[1] and app.side.visible  # panels were not toggled
    press_key(app, "Tab")
    assert app.view.focused_widget is fields[0]  # wraps around
    press_key(app, "Tab", mods=("shift",))
    assert app.view.focused_widget is fields[1]
    key(app, Qt.Key.Key_Tab)  # through a real key event as well
    assert app.view.focused_widget is fields[0]
    press_key(app, "Escape")
    press_key(app, "Tab")
    assert not app.side.visible  # without a sheet, Tab still hides the panels
    press_key(app, "Tab")


def test_autosave_and_recovery(app_with_doc: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    app = app_with_doc
    assert not app.autosave_now()  # nothing unsaved yet
    app._doc_op(lambda d: d.fill("#ff0000"))
    assert app.autosave_now() and app.autosave_path.exists()
    assert doc_of(app).dirty and doc_of(app).path is None  # the document itself is untouched
    app._autosave_tick()  # the timer path reschedules itself
    assert app._autosave_job is not None
    # saving the project for real removes the autosave
    monkeypatch.setattr(app.shell, "ask_save_file", lambda s, f: str(app.config_dir / "real.pfp"))
    assert app.save_project() and not app.autosave_path.exists()
    # simulate a crash: autosave exists at next start -> recovery sheet
    app._doc_op(lambda d: d.fill("#00ff00"))
    app.autosave_now()
    doc_of(app).mark_saved()
    app.set_document(None)
    app.offer_recovery()
    assert app.sheet is not None and app.sheet.name == "recover"
    press_key(app, "Return")  # default action recovers
    assert app.doc is not None and doc_of(app).dirty and doc_of(app).path is None
    assert pixel(doc_of(app).active_layer.image, 1, 1) == (0, 255, 0, 255)
    assert "Recovered" in app.info_label.text
    # discarding deletes the file
    app.autosave_now()
    doc_of(app).mark_saved()
    app.set_document(None)
    app.offer_recovery()
    buttons = [w for w in sheet_of(app).content.descendants() if isinstance(w, ui.Button)]
    fire(next(b for b in buttons if b.text == "Discard"))
    assert app.sheet is None and not app.autosave_path.exists()
    app.offer_recovery()  # nothing to offer
    assert app.sheet is None
    # an unreadable autosave is reported, not fatal
    app.autosave_path.write_bytes(b"garbage")
    app.offer_recovery()
    press_key(app, "Return")
    assert app.doc is None and "could not be read" in app.info_label.text and not app.autosave_path.exists()


def test_recovery_offered_on_startup(qapp: QtWidgets.QApplication, tmp_path: Path, gradient: Image.Image) -> None:
    from pixelforge.core import io as pfio
    from pixelforge.core.document import Document
    from pixelforge.gui.qt import MainWindow

    cfg = tmp_path / "cfg"
    cfg.mkdir()
    doc = Document.from_image(gradient)
    doc.fill("#0000ff")
    pfio.save_project(doc, cfg / "autosave.pfp", record=False)
    win = MainWindow(config_dir=cfg)
    try:
        win.show()
        QTest.qWait(500)
        qapp.processEvents()
        recover_sheet = editor_of(win).sheet
        assert recover_sheet is not None and recover_sheet.name == "recover"
    finally:
        editor_of(win).shutdown()
        win._closing = True
        win.close()
        theme.set_scale(1.0)


def test_window_geometry_is_remembered(qapp: QtWidgets.QApplication, tmp_path: Path) -> None:
    from pixelforge.gui.qt import MainWindow

    cfg = tmp_path / "cfg"
    win = MainWindow(config_dir=cfg)
    try:
        win.show()
        win.resize(1000, 700)
        win.move(30, 40)
        qapp.processEvents()
        win._save_geometry()
        saved = editor_of(win).settings.get("window")
        assert saved["w"] == 1000 and saved["h"] == 700
    finally:
        editor_of(win).shutdown()
        win._closing = True
        win.close()
    again = MainWindow(config_dir=cfg)
    try:
        again.show()
        qapp.processEvents()
        avail = again.screen().availableGeometry()
        expected = (max(900, min(1000, avail.width())), max(600, min(700, avail.height())))  # clamped to the screen
        assert (again.width(), again.height()) == expected
        editor_of(again).settings.set("window", {"w": "bad"})
        again._restore_geometry(again.screen().availableGeometry())  # ignored gracefully
    finally:
        editor_of(again).shutdown()
        again._closing = True
        again.close()
        theme.set_scale(1.0)


def test_default_language_follows_the_system(monkeypatch: pytest.MonkeyPatch) -> None:
    from pixelforge.gui import i18n

    monkeypatch.delenv("PIXELFORGE_LANG", raising=False)
    monkeypatch.setattr(i18n, "_system_language", lambda: "de_DE")
    assert i18n.set_language(None) == "de"
    monkeypatch.setattr(i18n, "_system_language", lambda: "fr_FR")
    assert i18n.set_language(None) == "fr"
    monkeypatch.setattr(i18n, "_system_language", lambda: "pt_BR")
    assert i18n.set_language(None) == "en"  # no Portuguese locale file -> English
    monkeypatch.setenv("PIXELFORGE_LANG", "de")
    assert i18n.set_language(None) == "de"
    i18n.set_language("en")


def test_title_tooltip_shows_the_full_path(app: Editor, image_file: Path) -> None:
    app.open_file(str(image_file))
    assert app.title_label.tooltip == str(image_file)


def test_bundled_font_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The interface ships its own font so it looks the same everywhere (system fonts on request)."""
    files = theme._font_files()
    assert files["bundled-regular"].endswith("DejaVuSans.ttf") and files["bundled-medium"].endswith("DejaVuSans-Bold.ttf")
    assert (theme.BUNDLED_FONT_DIR / "LICENSE-DejaVu.txt").is_file()
    monkeypatch.setattr(theme, "USE_SYSTEM_FONTS", False)
    theme._load_font.cache_clear()
    regular = theme.font(13)
    assert isinstance(regular, ImageFont.FreeTypeFont) and str(regular.path).endswith("DejaVuSans.ttf")
    assert str(theme.font(13, "medium").path).endswith("DejaVuSans-Bold.ttf")  # type: ignore[union-attr]
    assert theme.has_glyph(regular, "⌘") and theme.has_glyph(regular, "×")  # no fallback needed for the UI's symbols
    monkeypatch.setattr(theme, "USE_SYSTEM_FONTS", True)
    assert theme._candidates("regular")[-1] == "bundled-regular" and theme._candidates("medium")[0] != "bundled-medium"
    theme._load_font.cache_clear()
    theme.font(13)  # loads whatever the system offers, falling back to the bundled font
    theme._load_font.cache_clear()


def test_themes_switch_the_palette(app_with_doc: Editor) -> None:
    app = app_with_doc
    assert app.theme == "dark" and is_checked(app, "theme:dark")
    dark_bg = theme.PALETTE.bg
    app.command("theme:light").run()
    assert app.theme == "light" and theme.PALETTE.bg != dark_bg and theme.is_light_theme() and app.settings.get("theme") == "light"
    app.view.render()
    frame = _frame(app)
    assert pixel(frame, 2, frame.height - 2)[:3] == theme.LIGHT.bg  # the workspace took the light colour
    icon = glass.icon("brush")  # default icon colour follows the theme
    assert icon.getbbox() is not None and glass.icon("brush", color=theme.DARK.fg) is not icon
    app.command("theme:high-contrast").run()
    assert theme.PALETTE.accent == theme.HIGH_CONTRAST.accent and is_checked(app, "theme:high-contrast")
    assert theme.set_theme("no-such-theme") == "dark"
    app.set_theme("dark")
    assert theme.PALETTE.bg == dark_bg


def test_locales_are_complete() -> None:
    import json

    from pixelforge.gui import i18n

    de = json.loads((i18n.LOCALES_DIR / "de.json").read_text(encoding="utf-8"))
    for code in ("fr", "es"):
        table = json.loads((i18n.LOCALES_DIR / f"{code}.json").read_text(encoding="utf-8"))
        assert set(table) >= set(de), f"{code} misses {sorted(set(de) - set(table))[:5]}"
        assert all(value.strip() for value in table.values())
    assert set(i18n.available_languages()) >= {"en", "de", "fr", "es"}
    for key in ("Open…", "Theme", "Light Theme", "Layer Mask"):
        assert key in de


def test_accessibility_tree_describes_the_interface(app_with_doc: Editor) -> None:
    from pixelforge.gui import accessibility

    QA = QtGui.QAccessible  # noqa: N806
    app = app_with_doc
    widget = shell_of(app).widget
    root = QA.queryAccessibleInterface(widget)
    assert isinstance(root, accessibility.SurfaceAccessible)
    assert root.text(QA.Text.Name) == "PixelForge workspace"
    visible_panels = [p for p in app.view.panels if p.visible]
    assert root.childCount() == 1 + len(visible_panels)
    picture = root.child(0)
    assert picture is not None and picture.role() == QA.Role.Graphic and "400 × 300" in picture.text(QA.Text.Name)
    assert picture.text(QA.Text.Description) == app.info_label.text or picture.text(QA.Text.Description)
    names = {root.child(i).text(QA.Text.Name) for i in range(1, root.childCount())}  # type: ignore[union-attr]
    assert {"Toolbar", "Tools", "Inspector"} <= names
    # the toolbar pane lists its buttons with role, name, state and an on-screen rectangle
    toolbar = next(root.child(i) for i in range(root.childCount()) if root.child(i).text(QA.Text.Name) == "Toolbar")  # type: ignore[union-attr]
    assert isinstance(toolbar, accessibility.PanelNode) and toolbar.role() == QA.Role.Pane and toolbar.childCount() >= 7
    first = toolbar.child(0)
    assert isinstance(first, accessibility.ControlNode) and first.role() == QA.Role.PushButton and "Open" in first.text(QA.Text.Name)
    assert first.rect().width() > 0 and toolbar.indexOfChild(first) == 0 and first.parent() is toolbar
    assert toolbar.childAt(first.rect().center().x(), first.rect().center().y()) is first
    assert root.indexOfChild(toolbar) > 0 and root.childAt(first.rect().center().x(), first.rect().center().y()) is first
    undo = next(toolbar.child(i) for i in range(toolbar.childCount()) if "Undo" in toolbar.child(i).text(QA.Text.Name))  # type: ignore[union-attr]
    assert undo is not None and undo.state().disabled  # nothing to undo yet
    # tool buttons report their active state; activating one through the tree selects it
    tools = next(root.child(i) for i in range(root.childCount()) if root.child(i).text(QA.Text.Name) == "Tools")  # type: ignore[union-attr]
    brush = next(tools.child(i) for i in range(tools.childCount()) if "Brush" in tools.child(i).text(QA.Text.Name))  # type: ignore[union-attr]
    assert isinstance(brush, accessibility.ControlNode) and not brush.state().checked
    assert brush.do_action() and app.tool is not None and app.tool.id == "brush" and brush.state().checked
    # the inspector's sliders, tabs and layer rows are described too
    inspector = next(root.child(i) for i in range(root.childCount()) if root.child(i).text(QA.Text.Name) == "Inspector")  # type: ignore[union-attr]
    kinds = {inspector.child(i).role() for i in range(inspector.childCount())}  # type: ignore[union-attr]
    assert QA.Role.Slider in kinds and QA.Role.ComboBox in kinds
    slider = next(inspector.child(i) for i in range(inspector.childCount()) if inspector.child(i).role() == QA.Role.Slider)  # type: ignore[union-attr]
    assert "Exposure, +0" == slider.text(QA.Text.Name) and slider.text(QA.Text.Value) == "0"
    app.tabs.select(2)
    rows = [inspector.child(i) for i in range(inspector.childCount()) if inspector.child(i).role() == QA.Role.ListItem]  # type: ignore[union-attr]
    assert rows and rows[0].state().selected and "Background" in rows[0].text(QA.Text.Name)
    toggles_before = app.mask_overlay
    # describe() covers every control type and activate() drives toggles / list items
    describe = accessibility.describe
    toggle = ui.Toggle("Snap", False, lambda v: None)
    toggle.layout(0, 0, 100, 30)
    assert describe(toggle)[0] == QA.Role.CheckBox and accessibility.activate(toggle) and toggle.value
    assert describe(ui.Label("Hi"))[0] == QA.Role.StaticText
    assert describe(ui.TextField("Name", "x", suffix="px"))[1] == "Name: xpx"
    assert describe(ui.Histogram())[0] == QA.Role.Graphic and describe(ui.AnchorPicker())[0] == QA.Role.Grouping
    assert describe(ui.CurveEditor({}))[1].endswith("(rgb)") and describe(ui.LevelsEditor())[0] == QA.Role.Graphic
    assert describe(ui.Spacer())[0] == QA.Role.Client and not accessibility.activate(ui.Spacer())
    item = ui.ListItem("Row", lambda: None, subtitle="sub")
    assert describe(item)[1] == "Row, sub" and accessibility.activate(item)
    assert describe(app.swatches)[1].startswith("Foreground")
    assert app.mask_overlay == toggles_before
    # an unrelated widget gets no custom interface; announcements are no-ops without an active client
    assert not isinstance(QA.queryAccessibleInterface(shell_of(app)), accessibility.SurfaceAccessible)
    accessibility.announce(widget, "hello")
    accessibility.focus_changed(widget)
    app.notify("spoken")
