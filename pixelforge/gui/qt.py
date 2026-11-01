"""Qt (PySide6) shell: a HiDPI-aware window that paints the :class:`Surface` and provides native menus/dialogs.

Everything visible is still rendered by Pillow in *device* pixels; this layer
only blits the resulting images with the screen's device-pixel ratio, so the
interface and the picture are pixel-sharp on Retina / HiDPI displays.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

from .. import APP_NAME, __version__
from . import accessibility
from .editor import CONFIG_DIR, Command, Editor
from .i18n import tr
from .surface import Layer, Surface, wheel_pixels_from_notches

Qt = QtCore.Qt

CURSORS: dict[str, Qt.CursorShape] = {
    "arrow": Qt.CursorShape.ArrowCursor,
    "hand": Qt.CursorShape.PointingHandCursor,
    "grab": Qt.CursorShape.OpenHandCursor,
    "grabbing": Qt.CursorShape.ClosedHandCursor,
    "cross": Qt.CursorShape.CrossCursor,
    "move": Qt.CursorShape.SizeAllCursor,
    "resize-h": Qt.CursorShape.SizeHorCursor,
    "resize-v": Qt.CursorShape.SizeVerCursor,
    "resize-nwse": Qt.CursorShape.SizeFDiagCursor,
    "resize-nesw": Qt.CursorShape.SizeBDiagCursor,
    "resize-all": Qt.CursorShape.SizeAllCursor,
    "rotate": Qt.CursorShape.OpenHandCursor,
    "text": Qt.CursorShape.IBeamCursor,
    "zoom-in": Qt.CursorShape.CrossCursor,
}

KEY_NAMES: dict[int, str] = {
    Qt.Key.Key_Escape: "Escape",
    Qt.Key.Key_Return: "Return",
    Qt.Key.Key_Enter: "KP_Enter",
    Qt.Key.Key_Backspace: "BackSpace",
    Qt.Key.Key_Delete: "Delete",
    Qt.Key.Key_Tab: "Tab",
    Qt.Key.Key_Backtab: "Tab",
    Qt.Key.Key_Space: "space",
    Qt.Key.Key_Left: "Left",
    Qt.Key.Key_Right: "Right",
    Qt.Key.Key_Up: "Up",
    Qt.Key.Key_Down: "Down",
    Qt.Key.Key_Plus: "plus",
    Qt.Key.Key_Equal: "equal",
    Qt.Key.Key_Minus: "minus",
    Qt.Key.Key_BracketLeft: "bracketleft",
    Qt.Key.Key_BracketRight: "bracketright",
    Qt.Key.Key_Backslash: "backslash",
}


def key_name(event: QtGui.QKeyEvent) -> str:
    name = KEY_NAMES.get(event.key())
    if name:
        return name
    text = event.text()
    if text and text.isprintable():
        return text.lower()
    return QtGui.QKeySequence(event.key()).toString().lower()


def modifier_set(mods: Qt.KeyboardModifier) -> frozenset[str]:
    out = set()
    if mods & Qt.KeyboardModifier.ControlModifier:  # Command on macOS (Qt swaps Ctrl/Meta there)
        out.add("mod")
    if mods & Qt.KeyboardModifier.ShiftModifier:
        out.add("shift")
    if mods & Qt.KeyboardModifier.AltModifier:
        out.add("alt")
    return frozenset(out)


def pil_to_qimage(img: Image.Image) -> tuple[QtGui.QImage, bytes]:
    """Convert a Pillow image to a QImage; the returned bytes must stay alive with the image."""
    if img.mode == "RGB":
        data = img.tobytes()
        fmt = QtGui.QImage.Format.Format_RGB888
        stride = img.width * 3
    else:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        data = img.tobytes()
        fmt = QtGui.QImage.Format.Format_RGBA8888
        stride = img.width * 4
    return QtGui.QImage(data, img.width, img.height, stride, fmt), data


class _Dispatcher(QtCore.QObject):
    """Runs callables on the GUI thread; ``call`` may be invoked from any thread (queued signal)."""

    requested = QtCore.Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.requested.connect(self._run, Qt.ConnectionType.QueuedConnection)

    def call(self, fn: Callable[[], None]) -> None:
        self.requested.emit(fn)

    @QtCore.Slot(object)
    def _run(self, fn: Callable[[], None]) -> None:
        fn()


class SurfaceWidget(QtWidgets.QWidget):
    """Paints the surface's layers and feeds it pointer/keyboard events in device pixels."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.surface: Surface | None = None
        self.editor: Editor | None = None
        self._layers: list[Layer] = []
        self._qimages: dict[int, tuple[QtGui.QImage, bytes, Image.Image]] = {}
        self._timers: set[QtCore.QTimer] = set()
        self._dpr = 1.0
        self.setObjectName("pixelforge-surface")
        self.setAccessibleName("PixelForge workspace")
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)
        self.grabGesture(Qt.GestureType.PinchGesture)

    # ------------------------------------------------------------ backend --
    @property
    def dpr(self) -> float:
        return self.devicePixelRatioF()

    def viewport_size(self) -> tuple[int, int]:
        dpr = self.dpr
        return max(1, int(round(self.width() * dpr))), max(1, int(round(self.height() * dpr)))

    def present(self, layers: list[Layer]) -> None:
        self._layers = layers
        live = {id(img) for _k, _x, _y, img in layers}
        for key in [k for k in self._qimages if k not in live]:
            del self._qimages[key]
        self.update()

    def set_cursor(self, name: str) -> None:
        self.setCursor(CURSORS.get(name, Qt.CursorShape.ArrowCursor))

    def after(self, ms: int, fn: Callable[[], None]) -> object:
        timer = QtCore.QTimer(self)
        timer.setSingleShot(True)

        def fire() -> None:
            self._timers.discard(timer)
            timer.deleteLater()
            fn()

        timer.timeout.connect(fire)
        self._timers.add(timer)
        timer.start(max(0, ms))
        return timer

    def focus_changed(self) -> None:
        """The surface moved its keyboard focus to another control."""
        accessibility.focus_changed(self)

    def after_cancel(self, handle: object) -> None:
        timer = handle
        if isinstance(timer, QtCore.QTimer) and timer in self._timers:
            timer.stop()
            self._timers.discard(timer)
            timer.deleteLater()

    # ------------------------------------------------------------ painting --
    def _qimage(self, img: Image.Image) -> QtGui.QImage:
        entry = self._qimages.get(id(img))
        if entry is None or entry[2] is not img:
            qimg, data = pil_to_qimage(img)
            entry = (qimg, data, img)
            self._qimages[id(img)] = entry
        return entry[0]

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        dpr = self.dpr
        if not self._layers:
            painter.fillRect(self.rect(), QtGui.QColor(20, 21, 25))
        for _kind, x, y, img in self._layers:
            qimg = self._qimage(img)
            qimg.setDevicePixelRatio(dpr)
            painter.drawImage(QtCore.QPointF(x / dpr, y / dpr), qimg)
        painter.end()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._check_dpr()
        if self.surface is not None:
            self.surface.resized()

    def _check_dpr(self) -> None:
        dpr = self.dpr
        if abs(dpr - self._dpr) > 1e-3:
            self._dpr = dpr
            window = self.window()
            if isinstance(window, MainWindow) and self.editor is not None:
                self.editor.rescale(dpr * window.user_scale)

    # -------------------------------------------------------------- events --
    def _pos(self, event) -> tuple[int, int]:
        p = event.position()
        dpr = self.dpr
        return int(round(p.x() * dpr)), int(round(p.y() * dpr))

    def _mods(self, event) -> tuple[bool, bool, bool]:
        mods = event.modifiers()
        return (
            bool(mods & Qt.KeyboardModifier.ShiftModifier),
            bool(mods & Qt.KeyboardModifier.ControlModifier),
            bool(mods & Qt.KeyboardModifier.AltModifier),
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.surface is None:
            return
        self.setFocus()
        x, y = self._pos(event)
        button = 2 if event.button() == Qt.MouseButton.MiddleButton else 1
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self.surface.press(x, y, *self._mods(event), button=button)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.surface is None:
            return
        x, y = self._pos(event)
        if event.buttons() & (Qt.MouseButton.LeftButton | Qt.MouseButton.MiddleButton):
            self.surface.drag(x, y, *self._mods(event))
        else:
            self.surface.motion(x, y)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.surface is None:
            return
        x, y = self._pos(event)
        button = 2 if event.button() == Qt.MouseButton.MiddleButton else 1
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self.surface.release(x, y, *self._mods(event), button=button)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.surface is None or event.button() != Qt.MouseButton.LeftButton:
            return
        x, y = self._pos(event)
        self.surface.double(x, y, *self._mods(event))

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        if self.surface is not None:
            self.surface.leave()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if self.surface is None:
            return
        x, y = self._pos(event)
        pixels = event.pixelDelta()
        if not pixels.isNull():  # trackpad: precise pixel deltas
            dx, dy = pixels.x() * self.dpr, pixels.y() * self.dpr
        else:  # mouse wheel: notches of 15 degrees
            angle = event.angleDelta()
            dx, dy = wheel_pixels_from_notches(angle.x() / 120), wheel_pixels_from_notches(angle.y() / 120)
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        self.surface.wheel(x, y, dx, dy, ctrl)
        event.accept()

    def event(self, event: QtCore.QEvent) -> bool:
        if isinstance(event, QtGui.QNativeGestureEvent) and self.surface is not None:
            gesture = event  # macOS trackpad pinch
            if gesture.gestureType() == Qt.NativeGestureType.ZoomNativeGesture and self.surface.doc is not None:
                p = gesture.position()
                anchor = (p.x() * self.dpr, p.y() * self.dpr)
                self.surface.set_zoom(self.surface.zoom * (1.0 + gesture.value()), anchor)
                return True
        if isinstance(event, QtWidgets.QGestureEvent) and self.surface is not None:
            pinch = event.gesture(Qt.GestureType.PinchGesture)
            if isinstance(pinch, QtWidgets.QPinchGesture) and self.surface.doc is not None:
                center = pinch.centerPoint()
                anchor = (center.x() * self.dpr, center.y() * self.dpr)
                self.surface.set_zoom(self.surface.zoom * pinch.scaleFactor(), anchor)
                return True
        return super().event(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if self.editor is None:
            return
        name = key_name(event)
        if name == "space" and event.isAutoRepeat():
            return
        if self.editor.key_pressed(name, event.text(), modifier_set(event.modifiers())):
            event.accept()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        if self.editor is not None and not event.isAutoRepeat():
            self.editor.key_released(key_name(event))


class MainWindow(QtWidgets.QMainWindow):
    """The application window; implements the editor's :class:`~pixelforge.gui.editor.Shell` protocol."""

    def __init__(self, config_dir: Path | None = None, user_scale: float = 1.0) -> None:
        super().__init__()
        self.user_scale = user_scale
        self.setWindowTitle(APP_NAME)
        self.setAcceptDrops(True)
        self._closing = False
        self._actions: dict[str, QtGui.QAction] = {}
        self._dispatcher = _Dispatcher()
        accessibility.install()
        self.editor: Editor | None = None
        self.widget = SurfaceWidget(self)
        self.setCentralWidget(self.widget)
        self.surface = Surface(None, self.widget)
        self.widget.surface = self.surface
        screen = self.screen() or QtWidgets.QApplication.primaryScreen()
        dpr = screen.devicePixelRatio() if screen is not None else 1.0
        self.widget._dpr = dpr
        self.editor = Editor(self, config_dir or CONFIG_DIR, scale=dpr * user_scale)
        self.surface.app = self.editor
        self.widget.editor = self.editor
        if not self._actions:  # the editor may already have requested menus during construction
            self._build_menus()
        self.setMinimumSize(900, 600)
        if screen is not None:
            avail = screen.availableGeometry()
            w, h = min(1320, max(900, avail.width() - 80)), min(840, max(600, avail.height() - 120))
            self.resize(w, h)
            self.move(avail.x() + (avail.width() - w) // 2, avail.y() + max(0, (avail.height() - h) // 2 - 20))
            self._restore_geometry(avail)
        self.widget.setFocus()

    # --------------------------------------------------------------- menus --
    def menus_changed(self) -> None:
        """Rebuild the menu bar (after a language change)."""
        if self.editor is None:
            return
        for action in list(self._actions.values()):
            self.removeAction(action)
        self._actions.clear()
        self.menuBar().clear()
        self._build_menus()

    def _build_menus(self) -> None:
        bar = self.menuBar()
        bar.setNativeMenuBar(True)
        menus: dict[str, QtWidgets.QMenu] = {}
        submenus: dict[tuple[str, str], QtWidgets.QMenu] = {}
        self.recent_menu: QtWidgets.QMenu | None = None
        if self.editor is None:
            return
        for cmd in self.editor.commands:
            menu = menus.get(cmd.menu)
            if menu is None:
                menu = menus[cmd.menu] = bar.addMenu(cmd.menu)
            if cmd.id == "clear-recent":
                self.recent_menu = menu.addMenu(tr("Open Recent"))
                menu.addSeparator()
                continue  # "Clear" lives inside the Open Recent submenu
            if cmd.id == "quit":
                menu.addSeparator()
            if cmd.submenu:
                key = (cmd.menu, cmd.submenu)
                sub = submenus.get(key)
                if sub is None:
                    sub = submenus[key] = menu.addMenu(cmd.submenu)
                menu = sub
            action = QtGui.QAction(cmd.label, self)
            if cmd.checked is not None:
                action.setCheckable(True)
            if cmd.shortcut:
                action.setShortcut(QtGui.QKeySequence(cmd.shortcut.replace("Mod", "Ctrl")))
                action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            if cmd.role == "quit":
                action.setMenuRole(QtGui.QAction.MenuRole.QuitRole)
            elif cmd.role == "about":
                action.setMenuRole(QtGui.QAction.MenuRole.AboutRole)
            action.triggered.connect(lambda _checked=False, c=cmd: self._run_command(c))
            menu.addAction(action)
            self.addAction(action)  # shortcuts keep working even when the menu bar is hidden
            self._actions[cmd.id] = action
        if sys.platform != "darwin":
            bar.hide()  # the glass UI has its own controls; macOS gets the native menu bar
        self.commands_changed()

    def _run_command(self, cmd: Command) -> None:
        if cmd.enabled():
            cmd.run()

    def commands_changed(self) -> None:
        if self.editor is None or not self._actions:
            return
        for cmd in self.editor.commands:
            action = self._actions.get(cmd.id)
            if action is not None:
                action.setEnabled(cmd.enabled())
                if cmd.checked is not None:
                    action.setChecked(bool(cmd.checked()))
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self) -> None:
        if self.recent_menu is None or self.editor is None:
            return
        editor = self.editor
        self.recent_menu.clear()
        recent = editor.recent_files()
        for path in recent:
            action = self.recent_menu.addAction(Path(path).name)
            action.setToolTip(path)
            action.triggered.connect(lambda _checked=False, p=path: editor.open_file(p))
        if recent:
            self.recent_menu.addSeparator()
        clear = self.recent_menu.addAction(tr("Clear Menu"))
        clear.setEnabled(bool(recent))
        clear.triggered.connect(lambda _checked=False: editor.clear_recent())

    # --------------------------------------------------------------- shell --
    def set_title(self, title: str) -> None:
        self.setWindowTitle(title)

    @staticmethod
    def _filter(filetypes: list[tuple[str, str]]) -> str:
        return ";;".join(f"{name} ({patterns})" for name, patterns in filetypes)

    def ask_open_file(self, filetypes: list[tuple[str, str]]) -> str | None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open picture", "", self._filter(filetypes))
        return path or None

    def ask_save_file(self, suggested: str, filetypes: list[tuple[str, str]]) -> str | None:
        path, selected = QtWidgets.QFileDialog.getSaveFileName(self, "Save picture as", suggested, self._filter(filetypes))
        if not path:
            return None
        if not Path(path).suffix and selected:
            start = selected.find("*.")
            if start >= 0:
                ext = selected[start + 1 :].split()[0].rstrip(")")
                path += ext
        return path

    def ask_open_files(self, filetypes: list[tuple[str, str]]) -> list[str]:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Choose pictures", "", self._filter(filetypes))
        return list(paths)

    def ask_directory(self, title: str) -> str | None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, title, "")
        return path or None

    def ask_color(self, initial) -> tuple[int, int, int, int] | None:
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(*initial[:3]), self, tr("Choose a colour"))
        if not color.isValid():
            return None
        return (color.red(), color.green(), color.blue(), 255)

    def show_error(self, title: str, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, title, message)

    def ask_save_changes(self, name: str) -> bool | None:
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setText(f"Save changes to “{name}” before closing?")
        box.setInformativeText("Your edits will be lost if you don't save them.")
        box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Save)
        answer = box.exec()
        if answer == QtWidgets.QMessageBox.StandardButton.Save:
            return True
        if answer == QtWidgets.QMessageBox.StandardButton.Discard:
            return False
        return None

    def after(self, ms: int, fn: Callable[[], None]) -> object:
        return self.widget.after(ms, fn)

    def after_cancel(self, handle: object) -> None:
        self.widget.after_cancel(handle)

    def call_soon(self, fn: Callable[[], None]) -> None:
        """Thread-safe: run *fn* on the GUI thread as soon as the event loop gets to it."""
        self._dispatcher.call(fn)

    def announce(self, message: str) -> None:
        """Hand a status message to assistive technology."""
        accessibility.announce(self.widget, message)

    def focus_changed(self) -> None:
        self.widget.focus_changed()

    def close_window(self) -> None:
        self._closing = True
        self.close()

    def copy_image(self, image: Image.Image) -> None:
        qimg, _data = pil_to_qimage(image.convert("RGBA"))
        QtWidgets.QApplication.clipboard().setImage(qimg.copy())

    def paste_image(self) -> Image.Image | None:
        qimg = QtWidgets.QApplication.clipboard().image()
        if qimg.isNull():
            return None
        qimg = qimg.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
        width, height = qimg.width(), qimg.height()
        data = bytes(qimg.constBits())
        stride = qimg.bytesPerLine()
        rows = [data[y * stride : y * stride + width * 4] for y in range(height)]
        return Image.frombytes("RGBA", (width, height), b"".join(rows))

    # ------------------------------------------------------------ geometry --
    def _restore_geometry(self, avail: QtCore.QRect) -> None:
        saved = self.editor.settings.get("window") if self.editor is not None else None
        if not isinstance(saved, dict):
            return
        try:
            w, h = int(saved["w"]), int(saved["h"])
            x, y = int(saved["x"]), int(saved["y"])
        except (KeyError, TypeError, ValueError):
            return
        w, h = max(900, min(w, avail.width())), max(600, min(h, avail.height()))
        x = max(avail.x(), min(x, avail.x() + avail.width() - w))
        y = max(avail.y(), min(y, avail.y() + avail.height() - h))
        self.resize(w, h)
        self.move(x, y)
        if saved.get("maximized"):
            self.showMaximized()

    def _save_geometry(self) -> None:
        if self.editor is None:
            return
        geo = self.normalGeometry() if self.isMaximized() else self.geometry()
        self.editor.settings.set(
            "window", {"x": geo.x(), "y": geo.y(), "w": geo.width(), "h": geo.height(), "maximized": self.isMaximized()}
        )

    # -------------------------------------------------------------- events --
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        editor = self.editor
        if self._closing or editor is None or editor._confirm_discard():
            self._save_geometry()
            if editor is not None:
                editor.clear_autosave()
                editor.shutdown()
            event.accept()
        else:
            event.ignore()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls and self.editor is not None:
            self.editor.open_file(urls[0].toLocalFile())
            event.acceptProposedAction()

    def moveEvent(self, event: QtGui.QMoveEvent) -> None:
        super().moveEvent(event)
        self.widget._check_dpr()


def parse_args(argv: list[str]) -> tuple[float, str | None]:
    """Return (user_scale, path) from the command line."""
    scale = 1.0
    path: str | None = None
    args = list(argv)
    while args:
        arg = args.pop(0)
        if arg == "--scale":
            if not args:
                raise SystemExit("--scale needs a factor, e.g. --scale 1.25")
            value = args.pop(0)
        elif arg.startswith("--scale="):
            value = arg.split("=", 1)[1]
        else:
            path = arg
            continue
        try:
            scale = float(value)
        except ValueError:
            raise SystemExit(f"invalid scale factor: {value!r}") from None
        if not 0.5 <= scale <= 4:
            raise SystemExit("scale must be between 0.5 and 4")
    return scale, path


def run(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help"):
        print(
            f"{APP_NAME} {__version__}\nusage: python -m pixelforge [--scale FACTOR] [picture]\n\n"
            "  --scale FACTOR   extra UI scale on top of the display's pixel ratio (e.g. 1.25)\n\n"
            "Batch processing: python -m pixelforge.cli --help"
        )
        return 0
    if argv and argv[0] in ("-V", "--version"):
        print(f"{APP_NAME} {__version__}")
        return 0
    user_scale, path = parse_args(argv)
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    instance = QtWidgets.QApplication.instance()
    app = instance if isinstance(instance, QtWidgets.QApplication) else QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    window = MainWindow(user_scale=user_scale)
    window.show()
    if path:
        editor = window.editor
        assert editor is not None
        QtCore.QTimer.singleShot(50, lambda: editor.open_file(path))
    return app.exec()
