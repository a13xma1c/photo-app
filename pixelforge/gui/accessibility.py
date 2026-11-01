"""Accessibility: an accessible-object tree for the Pillow-rendered interface.

Everything PixelForge shows is painted onto one Qt widget, so assistive
technology would otherwise see a single blank canvas.  :func:`install`
registers a :class:`QAccessible` factory that describes the surface as a
tree - one *pane* per floating panel, with its buttons, sliders, toggles,
list rows and text fields as children that report a role, a name, their
state and their position on screen.  Screen readers and UI automation tools
can then read and activate the controls like native ones.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from PySide6 import QtCore, QtGui, QtWidgets

from .i18n import tr
from .ui import (
    AnchorPicker,
    Button,
    ColorSwatches,
    Column,
    CurveEditor,
    FlowRow,
    Histogram,
    IconButton,
    Label,
    LayerRow,
    LevelsEditor,
    ListItem,
    Panel,
    Row,
    Segmented,
    Separator,
    Slider,
    Spacer,
    TextField,
    Toggle,
    Widget,
)

if TYPE_CHECKING:
    from .surface import Surface

QA = QtGui.QAccessible
CONTAINERS = (Column, Row, FlowRow, Spacer, Separator)


def describe(widget: Widget) -> tuple[QtGui.QAccessible.Role, str, dict[str, bool], str]:
    """``(role, name, state flags, description)`` of a widget for assistive technology."""
    flags: dict[str, bool] = {}
    description = widget.tooltip or ""
    if isinstance(widget, Toggle):
        flags["checkable"], flags["checked"] = True, bool(widget.value)
        return QA.Role.CheckBox, widget.text, flags, description
    if isinstance(widget, Button):
        flags["checkable"] = False
        name = widget.text or widget.tooltip or (widget.icon or "")
        return QA.Role.PushButton, name, flags, description
    if isinstance(widget, IconButton):
        flags["checkable"], flags["checked"] = True, bool(getattr(widget, "active", False))
        return QA.Role.PushButton, widget.tooltip or widget.icon, flags, ""
    if isinstance(widget, Slider):
        value = widget.fmt.format(int(widget.value)) if widget.integer else f"{widget.value:.1f}"
        return QA.Role.Slider, f"{widget.label}, {value}", flags, description
    if isinstance(widget, Segmented):
        current = widget.options[widget.index] if 0 <= widget.index < len(widget.options) else ""
        return QA.Role.ComboBox, f"{current} ({' / '.join(widget.options)})", flags, description
    if isinstance(widget, LayerRow):
        flags["selectable"], flags["selected"] = True, bool(widget.active)
        details = widget.details()
        name = widget.layer.name + (f", {details}" if details else "") + ("" if widget.layer.visible else f", {tr('hidden')}")
        return QA.Role.ListItem, name, flags, ""
    if isinstance(widget, ListItem):
        flags["selectable"], flags["selected"] = True, bool(widget.selected or widget.checked)
        name = widget.text + (f", {widget.subtitle}" if widget.subtitle else "")
        return QA.Role.ListItem, name, flags, ""
    if isinstance(widget, TextField):
        flags["editable"] = True
        return QA.Role.EditableText, f"{widget.label}: {widget.text}{widget.suffix}", flags, widget.placeholder
    if isinstance(widget, Label):
        return QA.Role.StaticText, widget.text, flags, ""
    if isinstance(widget, ColorSwatches):
        fg, bg = widget.get_colors()
        return QA.Role.PushButton, tr("Foreground and background colours") + f": #{fg[0]:02x}{fg[1]:02x}{fg[2]:02x}", flags, ""
    if isinstance(widget, Histogram):
        return QA.Role.Graphic, tr("Histogram"), flags, ""
    if isinstance(widget, CurveEditor):
        return QA.Role.Graphic, tr("Curve editor") + f" ({widget.channel})", flags, ""
    if isinstance(widget, LevelsEditor):
        return QA.Role.Graphic, tr("Levels editor"), flags, ""
    if isinstance(widget, AnchorPicker):
        return QA.Role.Grouping, f"{widget.label}: {widget.value}", flags, ""
    return QA.Role.Client, type(widget).__name__, flags, description


def activate(widget: Widget) -> bool:
    """Perform the widget's default action (what a click would do); True when something happened."""
    command: Callable[[], None] | None = getattr(widget, "command", None)
    if isinstance(widget, Toggle):
        x0, y0, _x1, _y1 = widget.rect
        widget.on_release(x0 + 1, y0 + 1, True)
        return True
    if isinstance(widget, LayerRow):
        widget.on_select()
        return True
    if isinstance(widget, (Button, IconButton, ListItem)) and callable(command) and widget.enabled:
        command()
        return True
    return False


def leaf_widgets(panel: Panel) -> list[Widget]:
    """The controls of a panel in reading order (containers themselves are skipped)."""
    out: list[Widget] = []

    def walk(widget: Widget) -> None:
        if not widget.visible:
            return
        if isinstance(widget, CONTAINERS):
            for child in widget.children():
                walk(child)
        else:
            out.append(widget)

    walk(panel.content)
    return out


class _Node(QtGui.QAccessibleInterface):
    """Base for the virtual accessible objects (they have no QObject of their own)."""

    def __init__(self, owner: SurfaceAccessible, parent: QtGui.QAccessibleInterface | None) -> None:
        super().__init__()
        self._owner = owner
        self._parent = parent

    # geometry helpers --------------------------------------------------------
    def _global_rect(self, device_rect: tuple[int, int, int, int]) -> QtCore.QRect:
        widget = self._owner.widget
        dpr = getattr(widget, "dpr", 1.0) or 1.0
        x0, y0, x1, y1 = device_rect
        top_left = widget.mapToGlobal(QtCore.QPoint(int(x0 / dpr), int(y0 / dpr)))
        return QtCore.QRect(top_left.x(), top_left.y(), max(1, int((x1 - x0) / dpr)), max(1, int((y1 - y0) / dpr)))

    # QAccessibleInterface ----------------------------------------------------
    def isValid(self) -> bool:  # noqa: N802
        return True

    def object(self) -> QtCore.QObject | None:  # type: ignore[override]
        return None

    def window(self) -> QtGui.QWindow | None:  # type: ignore[override]
        handle = self._owner.widget.window().windowHandle()
        return handle

    def relations(self, match: Any = None) -> list:  # noqa: ARG002
        return []

    def parent(self) -> QtGui.QAccessibleInterface | None:  # type: ignore[override]
        return self._parent

    def childAt(self, x: int, y: int) -> QtGui.QAccessibleInterface | None:  # type: ignore[override]  # noqa: N802
        for i in range(self.childCount()):
            child = self.child(i)
            if child is not None and child.rect().contains(x, y):
                return child
        return None

    def child(self, index: int) -> QtGui.QAccessibleInterface | None:  # type: ignore[override]
        return None

    def childCount(self) -> int:  # noqa: N802
        return 0

    def indexOfChild(self, child: QtGui.QAccessibleInterface) -> int:  # noqa: N802
        for i in range(self.childCount()):
            if self.child(i) is child:
                return i
        return -1

    def setText(self, t: Any, text: str) -> None:  # noqa: N802
        pass

    def interface_cast(self, t: Any) -> Any:
        return None


class ControlNode(_Node):
    """One interface control (button, slider, row, …) inside a panel."""

    def __init__(self, owner: SurfaceAccessible, parent: PanelNode, panel: Panel, widget: Widget) -> None:
        super().__init__(owner, parent)
        self.panel = panel
        self.widget = widget

    def text(self, t: Any) -> str:
        role, name, _flags, description = describe(self.widget)
        if t == QA.Text.Name:
            return name
        if t == QA.Text.Description:
            return description
        if t == QA.Text.Value and isinstance(self.widget, (Slider, TextField)):
            return str(getattr(self.widget, "value", ""))
        return ""

    def role(self) -> QtGui.QAccessible.Role:
        return describe(self.widget)[0]

    def state(self) -> QtGui.QAccessible.State:
        _role, _name, flags, _desc = describe(self.widget)
        st = QA.State()
        st.focusable = True
        st.disabled = not self.widget.enabled
        st.invisible = not self.widget.visible
        st.focused = self._owner.surface.focused_widget is self.widget
        st.checkable = flags.get("checkable", False)
        st.checked = flags.get("checked", False)
        st.selectable = flags.get("selectable", False)
        st.selected = flags.get("selected", False)
        st.editable = flags.get("editable", False)
        return st

    def rect(self) -> QtCore.QRect:
        return self._global_rect(self.panel.widget_rect_in_view(self.widget))

    def do_action(self) -> bool:
        done = activate(self.widget)
        if done:
            self._owner.surface.schedule()
        return done


class PanelNode(_Node):
    """A floating glass panel (pane) holding its controls."""

    def __init__(self, owner: SurfaceAccessible, parent: QtGui.QAccessibleInterface, panel: Panel) -> None:
        super().__init__(owner, parent)
        self.panel = panel
        self._children: dict[int, ControlNode] = {}

    def _controls(self) -> list[ControlNode]:
        widgets = leaf_widgets(self.panel)
        live = {id(w) for w in widgets}
        for key in [k for k in self._children if k not in live]:
            del self._children[key]
        nodes = []
        for widget in widgets:
            node = self._children.get(id(widget))
            if node is None or node.widget is not widget:
                node = self._children[id(widget)] = ControlNode(self._owner, self, self.panel, widget)
            nodes.append(node)
        return nodes

    def text(self, t: Any) -> str:
        if t == QA.Text.Name:
            return tr(PANEL_NAMES.get(self.panel.name, self.panel.name or "Panel"))
        return ""

    def role(self) -> QtGui.QAccessible.Role:
        return QA.Role.Dialog if self.panel.modal else QA.Role.Pane

    def state(self) -> QtGui.QAccessible.State:
        st = QA.State()
        st.invisible = not self.panel.visible
        st.modal = self.panel.modal
        return st

    def rect(self) -> QtCore.QRect:
        return self._global_rect(self.panel.rect)

    def childCount(self) -> int:  # noqa: N802
        return len(self._controls())

    def child(self, index: int) -> QtGui.QAccessibleInterface | None:  # type: ignore[override]
        controls = self._controls()
        return controls[index] if 0 <= index < len(controls) else None


class PictureNode(_Node):
    """The picture itself (the area between the panels)."""

    def text(self, t: Any) -> str:
        app = self._owner.surface.app
        doc = getattr(app, "doc", None)
        if t == QA.Text.Name:
            if doc is None:
                return tr("No picture open")
            return f"{doc.name}, {doc.width} × {doc.height}"
        if t == QA.Text.Description and doc is not None:
            tool = getattr(app, "tool", None)
            return tr(tool.hint) if tool is not None else ""
        return ""

    def role(self) -> QtGui.QAccessible.Role:
        return QA.Role.Graphic

    def state(self) -> QtGui.QAccessible.State:
        st = QA.State()
        st.focusable = True
        return st

    def rect(self) -> QtCore.QRect:
        surface = self._owner.surface
        if surface.doc is None:
            w, h = surface.viewport_size()
            return self._global_rect((0, 0, w, h))
        return self._global_rect(surface.image_rect())


PANEL_NAMES = {
    "topbar": "Toolbar",
    "strip": "Tools",
    "side": "Inspector",
    "namebar": "Document",
    "zoombar": "Zoom",
    "infobar": "Status",
    "optionsbar": "Tool options",
    "cropbar": "Crop",
    "empty": "Welcome",
}


class SurfaceAccessible(QtWidgets.QAccessibleWidget):
    """Accessible root for the surface widget: the picture plus one node per visible panel."""

    def __init__(self, widget: QtWidgets.QWidget, surface: Surface) -> None:
        super().__init__(widget, QA.Role.Client, tr("PixelForge workspace"))
        self.widget: Any = widget
        self.surface = surface
        self._picture = PictureNode(self, self)
        self._panels: dict[int, PanelNode] = {}

    def _nodes(self) -> list[QtGui.QAccessibleInterface]:
        panels = [p for p in self.surface.panels if p.visible]
        live = {id(p) for p in panels}
        for key in [k for k in self._panels if k not in live]:
            del self._panels[key]
        nodes: list[QtGui.QAccessibleInterface] = [self._picture]
        for panel in panels:
            node = self._panels.get(id(panel))
            if node is None or node.panel is not panel:
                node = self._panels[id(panel)] = PanelNode(self, self, panel)
            nodes.append(node)
        return nodes

    def childCount(self) -> int:  # noqa: N802
        return len(self._nodes())

    def child(self, index: int) -> QtGui.QAccessibleInterface | None:  # type: ignore[override]
        nodes = self._nodes()
        return nodes[index] if 0 <= index < len(nodes) else None

    def indexOfChild(self, child: QtGui.QAccessibleInterface) -> int:  # noqa: N802
        for i, node in enumerate(self._nodes()):
            if node is child:
                return i
        return -1

    def childAt(self, x: int, y: int) -> QtGui.QAccessibleInterface | None:  # type: ignore[override]  # noqa: N802
        for node in reversed(self._nodes()[1:]):  # panels float above the picture
            if node.rect().contains(x, y):
                inner = node.childAt(x, y)
                return inner if inner is not None else node
        return self._picture

    def text(self, t: Any) -> str:
        if t == QA.Text.Name:
            return tr("PixelForge workspace")
        return super().text(t)


_installed = False


def install() -> None:
    """Register the accessibility factory (idempotent)."""
    global _installed
    if _installed:
        return

    def factory(classname: str, obj: QtCore.QObject) -> QtGui.QAccessibleInterface | None:
        surface = getattr(obj, "surface", None)
        if isinstance(obj, QtWidgets.QWidget) and surface is not None and obj.objectName() == "pixelforge-surface":
            return SurfaceAccessible(obj, surface)
        return None

    QA.installFactory(factory)
    _installed = True


def announce(widget: QtWidgets.QWidget, message: str) -> None:
    """Let assistive technology speak *message* (notifications, tool changes)."""
    if not QA.isActive():
        return
    try:
        event = QtGui.QAccessibleAnnouncementEvent(widget, message)
        QA.updateAccessibility(event)
    except (AttributeError, TypeError):  # older Qt without announcement events
        pass


def focus_changed(widget: QtWidgets.QWidget) -> None:
    if QA.isActive():
        QA.updateAccessibility(QtGui.QAccessibleEvent(widget, QA.Event.Focus))
