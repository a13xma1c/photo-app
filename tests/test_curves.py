"""Curves and Levels maths."""

import numpy as np
import pytest
from PIL import Image

from pixelforge.core import curves as cv
from pixelforge.core.document import Document
from tests.helpers import pixel


def test_identity_and_normalisation() -> None:
    assert cv.curve_lut(cv.IDENTITY_POINTS) == list(range(256))
    assert cv.is_identity(cv.default_curves()) and cv.is_identity(None)
    assert not cv.is_identity({"rgb": [(0, 0), (128, 100), (255, 255)]})
    assert cv.normalize_points([(300, -5), (10, 20), (10, 30)]) == [(10.0, 30.0), (255.0, 0.0)]
    assert cv.curve_lut([]) == list(range(256))  # no points: identity
    assert cv.curve_lut([(100, 40)]) == [40] * 256  # one point: constant
    assert not cv.is_identity_curve([(0, 0), (100, 100)])  # ends short of 255: not identity (flat beyond)


def test_curve_is_smooth_monotone_and_through_points() -> None:
    points = [(0, 0), (64, 30), (128, 160), (200, 220), (255, 255)]
    lut = cv.curve_lut(points)
    for x, y in points:
        assert lut[int(x)] == y
    assert all(lut[i] <= lut[i + 1] for i in range(255))  # no overshoot / reversal
    assert 30 < lut[96] < 160
    # flat beyond the end points, clamped to 0..255
    lut = cv.curve_lut([(50, 20), (200, 240)])
    assert lut[:50] == [20] * 50 and lut[200:] == [240] * 56
    inverted = cv.curve_lut([(0, 255), (255, 0)])
    assert inverted[0] == 255 and inverted[255] == 0 and all(inverted[i] >= inverted[i + 1] for i in range(255))
    values = cv.curve_values(points, np.array([0.0, 64.0, 255.0]))
    assert values.tolist() == pytest.approx([0, 30, 255])


def test_apply_curves_per_channel() -> None:
    img = Image.new("RGBA", (2, 1))
    img.putpixel((0, 0), (100, 100, 100, 200))
    img.putpixel((1, 0), (0, 255, 128, 255))
    curves = {"rgb": [(0, 0), (255, 255)], "r": [(0, 0), (255, 128)]}  # red halved, others untouched
    out = cv.apply_curves(img, curves)
    assert pixel(out, 0, 0) == (50, 100, 100, 200) and pixel(out, 1, 0) == (0, 255, 128, 255)
    assert cv.apply_curves(img, cv.default_curves()) is img
    # the master curve is applied before the channel curves
    master = {"rgb": [(0, 0), (255, 128)], "r": [(0, 0), (255, 128)]}
    assert pixel(cv.apply_curves(img, master), 1, 0)[1] == 128 and pixel(cv.apply_curves(img, master), 0, 0)[0] == 25
    rgb = cv.apply_curves(img.convert("RGB"), curves)
    assert rgb.mode == "RGB" and pixel(rgb, 0, 0) == (50, 100, 100)
    data = cv.serializable({"rgb": [(0, 0), (255, 255)]})
    assert data == {"rgb": [[0.0, 0.0], [255.0, 255.0]]}


def test_levels_points_and_histogram() -> None:
    pts = cv.levels_points(20, 235, 1.0, 0, 255)
    assert pts[0] == (20, 0) and pts[-1] == (235, 255) and pts[1] == (127.5, 127.5)
    brighter = cv.levels_points(0, 255, 2.0, 0, 255)[1]
    assert brighter[1] > 127.5
    assert cv.levels_points(0, 0, 1.0, 0, 255)[-1][0] == 1  # white clamps above black
    img = Image.new("RGBA", (4, 1), (0, 0, 0, 0))
    img.putpixel((0, 0), (255, 0, 10, 255))
    img.putpixel((1, 0), (255, 0, 10, 255))
    hist = cv.histogram(img)
    assert hist.shape == (3, 256) and hist[0, 255] == 1.0 and hist[1, 0] == 1.0 and hist[2, 10] == 1.0 and hist[0, 0] == 0
    assert cv.histogram(Image.new("RGBA", (2, 2), (0, 0, 0, 0))).max() == 0  # fully transparent


def test_document_apply_curves(gradient: Image.Image) -> None:
    doc = Document.from_image(gradient)
    before = pixel(doc.active_layer.image, 40, 10)
    doc.apply_curves({"rgb": [(0, 0), (255, 128)]})
    assert doc.history.labels[-1] == "Curves"
    after = pixel(doc.active_layer.image, 40, 10)
    assert all(a <= b // 2 + 1 for a, b in zip(after[:3], before[:3], strict=True)) and after[3] == 255
