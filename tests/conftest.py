"""Shared fixtures.

GUI tests run against the real Qt widgets on Qt's *offscreen* platform, so no
display server is needed.  They are skipped when PySide6 is not installed.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pixelforge.core.document import Document  # noqa: E402
from pixelforge.core.layers import Layer  # noqa: E402

if TYPE_CHECKING:
    from PySide6 import QtWidgets

    from pixelforge.gui.editor import Editor
    from pixelforge.gui.qt import MainWindow


# ----------------------------------------------------------------- images --
def make_gradient(width: int = 64, height: int = 48, alpha: bool = False) -> Image.Image:
    """Deterministic colourful test image (horizontal red ramp, vertical green ramp, blue constant)."""
    arr = np.zeros((height, width, 4 if alpha else 3), np.uint8)
    arr[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    arr[..., 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    arr[..., 2] = 128
    if alpha:
        arr[..., 3] = 255
    return Image.fromarray(arr, "RGBA" if alpha else "RGB")


def make_shapes(width: int = 64, height: int = 48) -> Image.Image:
    """RGBA image with a transparent background, a red square and a blue circle."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle((4, 4, 24, 24), fill=(255, 0, 0, 255))
    d.ellipse((30, 10, 58, 40), fill=(0, 0, 255, 255))
    return img


@pytest.fixture
def gradient() -> Image.Image:
    return make_gradient()


@pytest.fixture
def shapes() -> Image.Image:
    return make_shapes()


@pytest.fixture
def doc(gradient: Image.Image) -> Document:
    return Document.from_image(gradient, "test")


@pytest.fixture
def two_layer_doc(gradient: Image.Image, shapes: Image.Image) -> Document:
    d = Document.from_image(gradient, "test")
    d.add_layer(Layer(shapes, "Shapes"))
    return d


@pytest.fixture
def image_file(tmp_path: Path, gradient: Image.Image) -> Path:
    path = tmp_path / "gradient.png"
    gradient.save(path)
    return path


def mean_rgb(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float64).reshape(-1, 3).mean(axis=0)


def pixel_diff(a: Image.Image, b: Image.Image) -> float:
    return float(np.abs(np.asarray(a.convert("RGBA"), dtype=np.float64) - np.asarray(b.convert("RGBA"), dtype=np.float64)).mean())


# ------------------------------------------------------------------- GUI --
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp() -> Iterator[QtWidgets.QApplication]:
    """A single offscreen QApplication for the whole session (skips when PySide6 is missing)."""
    widgets = pytest.importorskip("PySide6.QtWidgets")
    instance = widgets.QApplication.instance()
    app = instance if isinstance(instance, widgets.QApplication) else widgets.QApplication(["pixelforge-tests"])
    yield app


@pytest.fixture
def window(qapp: QtWidgets.QApplication, tmp_path: Path) -> Iterator[MainWindow]:
    """A MainWindow with an isolated config directory, at UI scale 1."""
    from pixelforge.gui.qt import MainWindow
    from pixelforge.gui.theme import set_scale, set_theme

    win = MainWindow(config_dir=tmp_path / "config")
    win.resize(1280, 800)
    win.show()
    qapp.processEvents()
    yield win
    assert win.editor is not None
    win.editor.shutdown()
    win._closing = True
    win.close()
    win.deleteLater()
    qapp.processEvents()
    set_scale(1.0)
    set_theme("dark")


@pytest.fixture
def app(window: MainWindow) -> Editor:
    """The editor behind the window (what most GUI tests talk to)."""
    assert window.editor is not None
    return window.editor


@pytest.fixture
def app_with_doc(app: Editor, gradient: Image.Image) -> Editor:
    from pixelforge.core.document import Document

    app.set_document(Document.from_image(gradient.resize((400, 300)), "gradient"))
    app.shell.surface.render()
    return app
