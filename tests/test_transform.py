"""Free Transform: the affine maths, premultiplied resampling and the document operation."""

import math

import pytest
from PIL import Image

from pixelforge.core.document import Document, DocumentError
from pixelforge.core.layers import Layer
from pixelforge.core.painting import Transform, transform_image
from pixelforge.core.selection import Selection
from tests.helpers import gray, mask_of, pixel, sel


def square_doc() -> Document:
    doc = Document.new(100, 100, "transparent")
    doc.add_layer()
    doc.set_selection(Selection.rectangle(doc.size, (40, 40, 60, 60)))
    doc.fill("#ff0000")
    doc.deselect()
    return doc


def test_transform_maths() -> None:
    t = Transform((40.0, 40.0, 60.0, 60.0))
    assert t.is_identity and t.center == (50, 50) and t.apply(45, 47) == pytest.approx((45, 47))
    scaled = Transform((40.0, 40.0, 60.0, 60.0), scale_x=2, scale_y=0.5, dx=10)
    assert scaled.apply(40, 40) == pytest.approx((40, 45))  # (40-50)*2+50+10, (40-50)*0.5+50
    assert scaled.unapply(*scaled.apply(43, 57)) == pytest.approx((43, 57))
    rotated = Transform((40.0, 40.0, 60.0, 60.0), angle=90)
    assert rotated.apply(60, 50) == pytest.approx((50, 40))  # east point moves up (counter-clockwise on screen)
    assert rotated.bounds() == pytest.approx((40, 40, 60, 60))
    corners = Transform((0.0, 0.0, 10.0, 20.0), angle=45).corners()
    assert len(corners) == 4 and math.isclose(math.dist(corners[0], corners[2]), math.hypot(10, 20))
    with pytest.raises(ValueError):
        Transform((0.0, 0.0, 1.0, 1.0), scale_x=0.0).inverse()
    assert Transform((0.0, 0.0, 1.0, 1.0), pivot=(0.0, 0.0)).center == (0, 0)


def test_transform_image_keeps_edges_clean() -> None:
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    img.paste((255, 0, 0, 255), (40, 40, 60, 60))
    assert transform_image(img, Transform((40.0, 40.0, 60.0, 60.0))) is img  # identity: no work
    out = transform_image(img, Transform((40.0, 40.0, 60.0, 60.0), scale_x=2, scale_y=2))
    assert pixel(out, 50, 50) == (255, 0, 0, 255) and pixel(out, 35, 50)[:3] == (255, 0, 0)
    assert pixel(out, 5, 5)[3] == 0
    # no dark fringe: every partially transparent edge pixel is still red
    for x in range(100):
        r, g, b, a = pixel(out, x, 50)
        if 0 < a < 255:
            assert r > 200 and g < 40 and b < 40
    moved = transform_image(img, Transform((40.0, 40.0, 60.0, 60.0), dx=30, dy=-20))
    assert pixel(moved, 80, 30) == (255, 0, 0, 255) and pixel(moved, 50, 50)[3] == 0
    mask = Image.new("L", (100, 100), 0)
    mask.paste(255, (40, 40, 60, 60))
    flipped = transform_image(mask, Transform((40.0, 40.0, 60.0, 60.0), scale_x=-1, dx=25))
    assert flipped.mode == "L" and gray(flipped, 75, 50) == 255 and gray(flipped, 50, 50) == 0


def test_content_box_and_transform_active() -> None:
    doc = square_doc()
    assert doc.content_box() == (40, 40, 60, 60)
    doc.set_selection(Selection.rectangle(doc.size, (10, 10, 30, 30)))
    assert doc.content_box() == (10, 10, 30, 30)  # the selection wins
    doc.deselect()
    doc.transform_active(Transform((40.0, 40.0, 60.0, 60.0), scale_x=2, scale_y=2, angle=0, dx=10))
    assert doc.history.labels[-1] == "Free Transform"
    img = doc.active_layer.image
    assert pixel(img, 60, 50) == (255, 0, 0, 255) and pixel(img, 36, 50)[3] == 0  # doubled (30..70) and moved right by 10
    assert doc.content_box() == pytest.approx((40, 30, 80, 70), abs=3)
    # identity is a no-op, locked layers refuse
    steps = len(doc.history.labels)
    doc.transform_active(Transform((0.0, 0.0, 1.0, 1.0)))
    assert len(doc.history.labels) == steps
    doc.set_layer_props(locked=True)
    with pytest.raises(DocumentError):
        doc.transform_active(Transform((0.0, 0.0, 1.0, 1.0), dx=1))
    # the preview matches the committed result
    doc.set_layer_props(locked=False)
    t = Transform((40.0, 30.0, 80.0, 70.0), angle=45)
    preview = doc.preview_transform(t)
    doc.transform_active(t)
    assert preview.tobytes() == doc.active_layer.image.tobytes()
    assert Document.new(10, 10, "transparent").content_box() is None


def test_transform_selected_pixels_only_and_selection_follows() -> None:
    doc = square_doc()
    doc.set_selection(Selection.rectangle(doc.size, (40, 40, 50, 60)))  # left half of the square
    doc.transform_active(Transform((40.0, 40.0, 50.0, 60.0), dx=-30))
    img = doc.active_layer.image
    assert pixel(img, 15, 50) == (255, 0, 0, 255)  # moved part
    assert pixel(img, 55, 50) == (255, 0, 0, 255)  # unselected right half stayed
    assert pixel(img, 45, 50)[3] == 0  # source of the moved part is empty now
    assert sel(doc).contains(15, 50) and not sel(doc).contains(45, 50)
    # a selection moved completely off the canvas disappears
    doc.transform_active(Transform((10.0, 40.0, 20.0, 60.0), dx=-500))
    assert doc.selection is None


def test_transform_moves_mask_with_pixels_or_alone(gradient: Image.Image) -> None:
    doc = Document.from_image(gradient)
    doc.add_layer(Layer(Image.new("RGBA", gradient.size, (255, 0, 0, 255)), "Red"))
    doc.set_selection(Selection.rectangle(doc.size, (0, 0, 32, 48)))
    doc.add_mask(mode="selection")
    doc.deselect()
    doc.set_mask_editing(False)
    doc.transform_active(Transform((0.0, 0.0, 64.0, 48.0), dx=16))
    assert gray(mask_of(doc), 40, 10) == 255 and gray(mask_of(doc), 8, 10) == 0  # mask travelled with the pixels
    assert pixel(doc.active_layer.image, 8, 10)[3] == 0
    doc.set_mask_editing(True)
    assert doc.content_box() == (16, 0, 48, 48)  # the mask's white area
    doc.transform_active(Transform((16.0, 0.0, 48.0, 48.0), dx=-16))
    assert gray(mask_of(doc), 8, 10) == 255 and pixel(doc.active_layer.image, 8, 10)[3] == 0  # only the mask moved
    assert doc.preview_transform(Transform((0.0, 0.0, 32.0, 48.0), dx=5)).mode == "RGBA"
