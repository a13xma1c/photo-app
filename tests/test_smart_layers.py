"""Smart layers: 3D and text layers keep their filters, moves and transforms through re-rendering."""

import numpy as np
import pytest
from PIL import Image

from pixelforge.core import three_d as t3
from pixelforge.core.document import Document, DocumentError
from pixelforge.core.painting import Transform
from pixelforge.core.selection import Selection
from tests.helpers import pixel


def blur_amount(image: Image.Image) -> float:
    """Mean gradient of the alpha channel: a blurred object has soft edges."""
    alpha = np.asarray(image.getchannel("A"), np.float32)
    return float(np.abs(np.diff(alpha, axis=1)).max())


def test_filters_on_3d_layers_survive_moves() -> None:
    doc = Document.new(200, 150, "white")
    doc.add_3d_layer(t3.Scene(shape="cube", size=90, color=(200, 40, 40)))
    sharp = doc.active_layer.image
    doc.apply_filter("Gaussian Blur", radius=6)
    assert doc.smart_filters() == [{"kind": "filter", "name": "Gaussian Blur", "params": {"radius": 6}}]
    blurred = doc.active_layer.image
    assert blur_amount(blurred) < blur_amount(sharp) and doc.history.labels[-1] == "Gaussian Blur"
    # moving / rotating / transforming re-renders the object *and* re-applies the filter stack
    doc.offset_active(20, 0)
    moved = doc.active_layer.image
    assert doc.scene_of().position == (20.0, 0.0) and blur_amount(moved) < blur_amount(sharp)  # type: ignore[union-attr]
    assert moved.getbbox() != blurred.getbbox()
    doc.update_3d_layer(rotation=(10, 40, 0))
    assert blur_amount(doc.active_layer.image) < blur_amount(sharp)
    doc.apply_curves({"rgb": [(0, 0), (255, 128)]})
    assert len(doc.smart_filters()) == 2 and doc.smart_filters()[1]["kind"] == "curves"
    dark = pixel(doc.active_layer.image, 100, 75)
    doc.transform_active(Transform((50.0, 30.0, 150.0, 120.0), scale_x=1.5, scale_y=1.5, angle=10, dx=5))
    scene = doc.scene_of()
    assert scene is not None and scene.size == pytest.approx(90 * 1.5) and scene.rotation[2] == pytest.approx(10)
    assert scene.position == pytest.approx((25.0, 0.0))
    assert pixel(doc.active_layer.image, 100, 75)[0] <= dark[0] + 2 or pixel(doc.active_layer.image, 100, 75)[3] == 0
    # the stack can be cleared; rasterizing turns the layer into plain pixels where filters bake as usual
    doc.clear_smart_filters()
    assert not doc.smart_filters() and blur_amount(doc.active_layer.image) > blur_amount(blurred)
    doc.clear_smart_filters()  # nothing to do
    assert doc.history.labels[-1] == "Remove Smart Filters"
    doc.apply_filter("Invert")
    plain = doc.rasterize_layer()
    assert plain.kind == "pixel" and "3d" not in plain.meta and "filters" not in plain.meta and doc.scene_of() is None
    doc.apply_filter("Invert")
    assert not doc.smart_filters() and doc.history.labels[-1] == "Invert"
    with pytest.raises(DocumentError):
        doc.rasterize_layer()


def test_painting_on_smart_layers_is_refused_but_masks_work() -> None:
    doc = Document.new(100, 100, "white")
    doc.add_3d_layer(t3.Scene(shape="sphere", size=60))
    with pytest.raises(DocumentError, match="rasterize"):
        doc.fill("#000000")
    with pytest.raises(DocumentError):
        doc.bake_adjustments()
    doc.set_adjustments({"brightness": 40})
    assert doc.active_layer.has_adjustments  # live adjustments are fine
    doc.add_mask()
    doc.set_selection(Selection.rectangle(doc.size, (0, 0, 50, 100)))
    doc.fill("#000000")  # goes to the mask
    assert doc.history.labels[-1] == "Fill (Mask)" and pixel(doc.flatten(), 30, 50) == (255, 255, 255, 255)
    doc.deselect()
    doc.offset_active(10, 0)  # while editing the mask only the mask moves
    assert doc.scene_of().position == (0.0, 0.0)  # type: ignore[union-attr]
    doc.set_mask_editing(False)
    doc.offset_active(10, 0)
    assert doc.scene_of().position == (10.0, 0.0)  # type: ignore[union-attr]
    with pytest.raises(DocumentError):
        doc.add_smart_filter({"kind": "filter", "name": "Invert"}, "x", index=0)


def test_text_layers_are_smart_too() -> None:
    doc = Document.new(300, 120, "white")
    doc.add_text_layer("Hello", (20, 30), "#000000", None, 40)
    doc.apply_filter("Gaussian Blur", radius=3)
    assert doc.smart_filters() and blur_amount(doc.active_layer.image) < 255
    doc.offset_active(15, 5)
    assert doc.active_layer.meta["position"] == [35.0, 35.0] and doc.active_layer.kind == "text"
    assert blur_amount(doc.active_layer.image) < 255  # still blurred after the move
    doc.update_text_layer(text="Hi")
    assert doc.active_layer.name == "T Hi" and blur_amount(doc.active_layer.image) < 255
    doc.transform_active(Transform((35.0, 35.0, 135.0, 75.0), scale_x=2, scale_y=2, dx=10))
    assert doc.active_layer.meta["font_size"] == 80 and doc.active_layer.meta["position"][0] == pytest.approx(35 - 50 + 10)
    plain = doc.rasterize_layer()
    assert plain.kind == "pixel" and plain.name == "Hi" and "text" not in plain.meta
    with pytest.raises(DocumentError):
        doc.update_text_layer(text="x")
