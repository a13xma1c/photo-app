"""Adjustment layers: non-destructive sliders / curves / levels applied to everything below them."""

from pathlib import Path

import pytest
from PIL import Image

from pixelforge.core import io as pfio
from pixelforge.core.document import Document, DocumentError
from pixelforge.core.layers import Layer
from pixelforge.core.selection import Selection
from tests.helpers import pixel


def test_adjustment_layer_affects_layers_below_only(gradient: Image.Image) -> None:
    doc = Document.from_image(gradient)
    before = pixel(doc.flatten(), 40, 10)
    layer = doc.add_adjustment_layer()
    assert layer.kind == "adjustment" and layer.mask is not None and doc.editing_mask
    assert doc.history.labels[-1] == "New Adjustment Layer 'Adjustment 1'"
    assert pixel(doc.flatten(), 40, 10) == before  # neutral: no change
    doc.set_adjustments({"saturation": -100})
    grey = pixel(doc.flatten(), 40, 10)
    assert grey[0] == grey[1] == grey[2] and grey[3] == 255
    # a layer above the adjustment layer is untouched
    doc.add_layer(Layer(Image.new("RGBA", doc.size, (255, 0, 0, 255)), "Red"))
    assert pixel(doc.flatten(), 40, 10) == (255, 0, 0, 255)
    doc.set_layer_props(visible=False)
    # opacity fades the effect, the mask limits it, "compare" (neutral) skips it
    doc.set_active(1)
    doc.set_layer_props(opacity=0.5)
    half = pixel(doc.flatten(), 40, 10)
    assert grey[0] < half[0] < before[0] or grey[0] > half[0] > before[0]
    doc.set_layer_props(opacity=1.0)
    doc.set_selection(Selection.rectangle(doc.size, (0, 0, 32, 48)))
    doc.fill("#000000")  # paints the mask: the left half is hidden from the effect
    doc.deselect()
    assert pixel(doc.flatten(), 10, 10)[:3] == pixel(gradient, 10, 10) and pixel(doc.flatten(), 40, 10)[:3] != pixel(gradient, 40, 10)
    assert pixel(doc.composite(neutral=True), 40, 10)[:3] == pixel(gradient, 40, 10)
    # scaled / cropped renders agree with the full one
    small = doc.composite(0.5)
    assert pixel(small, 20, 5) == pytest.approx(pixel(doc.flatten(), 40, 10), abs=40)
    crop = doc.composite(1.0, (38, 8, 42, 12))
    assert pixel(crop, 2, 2) == pixel(doc.flatten(), 40, 10)


def test_adjustment_layer_curves_levels_and_blend(gradient: Image.Image) -> None:
    doc = Document.from_image(gradient)
    doc.add_adjustment_layer("Tones")
    before = pixel(doc.flatten(), 40, 10)
    doc.set_layer_meta(curves={"rgb": [[0, 0], [255, 128]]}, label="Curves")
    assert doc.history.labels[-1] == "Curves"
    darker = pixel(doc.flatten(), 40, 10)
    assert all(d <= b // 2 + 1 for d, b in zip(darker[:3], before[:3], strict=True))
    doc.set_layer_meta(levels={"in_black": 0, "in_white": 128, "gamma": 1.0, "out_black": 0, "out_white": 255})
    assert pixel(doc.flatten(), 40, 10) == pytest.approx(before, abs=3)  # the levels undo the curve
    doc.set_layer_meta(curves=None, levels=None)
    assert "curves" not in doc.active_layer.meta and pixel(doc.flatten(), 40, 10) == before
    doc.set_adjustments({"brightness": 50})
    doc.set_layer_props(blend_mode="multiply")
    assert pixel(doc.flatten(), 40, 10) != before
    # adjustment layers cannot be baked or lose their mask; merging bakes them into the layer below
    with pytest.raises(DocumentError):
        doc.bake_adjustments()
    with pytest.raises(DocumentError):
        doc.remove_mask()
    doc.set_mask_editing(False)
    assert doc.editing_mask  # still the mask
    expected = pixel(doc.flatten(), 40, 10)
    doc.merge_down()
    assert len(doc.layers) == 1 and doc.layers[0].kind == "background" and pixel(doc.flatten(), 40, 10) == expected


def test_adjustment_layer_alone_and_below_composite(gradient: Image.Image) -> None:
    doc = Document.new(20, 20, "transparent")
    doc.add_adjustment_layer()
    doc.set_adjustments({"brightness": 80})
    assert pixel(doc.flatten(), 5, 5)[3] == 0  # nothing below: still empty
    doc2 = Document.from_image(gradient)
    doc2.add_adjustment_layer()
    below = doc2.composite_below(1)
    assert below.size == gradient.size and pixel(below, 10, 10)[:3] == pixel(gradient, 10, 10)
    assert doc2.composite_below(0).getbbox() is None
    assert doc2.composite_below(1, 0.5).size == (32, 24)


def test_adjustment_layer_roundtrip(tmp_path: Path, gradient: Image.Image) -> None:
    doc = Document.from_image(gradient)
    doc.add_adjustment_layer("Look")
    doc.set_adjustments({"contrast": 30})
    doc.set_layer_meta(curves={"rgb": [[0, 0], [128, 100], [255, 255]]})
    expected = pixel(doc.flatten(), 40, 10)
    path = tmp_path / "adj.pfp"
    pfio.save_project(doc, path)
    loaded = pfio.load_project(path)
    layer = loaded.layers[1]
    assert layer.kind == "adjustment" and layer.adjustments["contrast"] == 30 and layer.meta["curves"]["rgb"][1] == [128, 100]
    assert pixel(loaded.flatten(), 40, 10) == expected
    assert loaded.duplicate_layer().kind == "adjustment"
