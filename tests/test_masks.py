"""Layer masks: rendering, operations, the mask edit target, transforms and persistence."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixelforge.core import io as pfio
from pixelforge.core.document import Document, DocumentError
from pixelforge.core.layers import Layer, mask_to_rgba, rgba_to_mask
from pixelforge.core.selection import Selection
from tests.helpers import gray, mask_of, pixel, sel


def half_mask(size: tuple[int, int]) -> Image.Image:
    """White on the left half, black on the right."""
    mask = Image.new("L", size, 0)
    mask.paste(255, (0, 0, size[0] // 2, size[1]))
    return mask


def red_over_gradient(gradient: Image.Image) -> Document:
    doc = Document.from_image(gradient, "masked")
    doc.add_layer(Layer(Image.new("RGBA", gradient.size, (255, 0, 0, 255)), "Red"))
    return doc


def half_masked(gradient: Image.Image) -> Document:
    """Red layer over the gradient with a committed mask that reveals only the left half."""
    doc = red_over_gradient(gradient)
    doc.set_selection(Selection.rectangle(doc.size, (0, 0, doc.width // 2, doc.height)))
    doc.add_mask(mode="selection")
    doc.deselect()
    return doc


# ------------------------------------------------------------------ layer --
def test_layer_mask_validation_and_helpers(shapes: Image.Image) -> None:
    with pytest.raises(ValueError):
        Layer(shapes, mask=Image.new("L", (3, 3)))
    rgb_mask = Image.new("RGB", shapes.size, (255, 255, 255))
    layer = Layer(shapes, mask=rgb_mask)
    assert layer.mask is not None and layer.mask.mode == "L"  # converted
    assert layer.effective_mask is layer.mask
    layer.mask_enabled = False
    assert layer.effective_mask is None and layer.masked_image() is layer.image
    # duplicate copies the mask, copy shares it
    layer.mask_enabled = True
    dup = layer.duplicate()
    assert dup.mask is not None and dup.mask is not layer.mask and layer.copy().mask is layer.mask
    # round trip between the grey picture painting tools see and the mask
    mask = half_mask(shapes.size)
    rgba = mask_to_rgba(mask)
    assert rgba.mode == "RGBA" and pixel(rgba, 0, 0) == (255, 255, 255, 255) and pixel(rgba, 60, 0) == (0, 0, 0, 255)
    assert rgba_to_mask(rgba).tobytes() == mask.tobytes()
    erased = rgba.copy()
    erased.putalpha(Image.new("L", rgba.size, 0))
    assert rgba_to_mask(erased).getextrema() == (0, 0)  # erased pixels hide


def test_manifest_records_mask_state(shapes: Image.Image) -> None:
    layer = Layer(shapes, mask=half_mask(shapes.size), mask_enabled=False)
    data = layer.to_manifest()
    assert data["has_mask"] and data["mask_enabled"] is False
    restored = Layer.from_manifest(data, shapes, half_mask(shapes.size))
    assert restored.mask is not None and not restored.mask_enabled
    assert Layer.from_manifest(data, shapes).mask is None


# -------------------------------------------------------------- rendering --
def test_mask_multiplies_alpha_in_composite(gradient: Image.Image) -> None:
    doc = red_over_gradient(gradient)
    doc.add_mask()  # reveal all: nothing changes yet
    assert pixel(doc.flatten(), 10, 10) == (255, 0, 0, 255) and pixel(doc.flatten(), 50, 10) == (255, 0, 0, 255)
    doc.layers[1] = doc.layers[1].copy(mask=half_mask(doc.size))
    doc.touch()
    flat = doc.flatten()
    assert pixel(flat, 10, 10) == (255, 0, 0, 255)  # left half: red shows
    assert pixel(flat, 50, 10)[:3] == pixel(gradient, 50, 10)  # right half: hidden, gradient shows through
    # disabled mask renders as if absent
    doc.set_mask_enabled(False)
    assert pixel(doc.flatten(), 50, 10) == (255, 0, 0, 255)
    doc.set_mask_enabled(True)
    assert pixel(doc.flatten(), 50, 10)[:3] == pixel(gradient, 50, 10)
    # zoomed and cropped views use the same mask region
    small = doc.composite(0.5)
    assert pixel(small, 5, 5) == (255, 0, 0, 255) and pixel(small, 25, 5)[:3] != (255, 0, 0)
    zoomed = doc.composite(2.0, (60, 0, 80, 20))  # doc x 30..40 → inside the hidden half
    assert pixel(zoomed, 5, 5)[:3] != (255, 0, 0)
    rendered_mask = doc.render_mask(0.5)
    assert rendered_mask is not None and rendered_mask.size == small.size
    assert doc.render_mask(1.0, (0, 0, 0, 0)) is None


def test_mask_overrides_preview_without_committing(gradient: Image.Image) -> None:
    doc = red_over_gradient(gradient)
    doc.add_mask()
    hidden = Image.new("L", doc.size, 0)
    flat = doc.composite(overrides=None, mask_overrides={doc.active_layer.id: hidden})
    assert pixel(flat, 10, 10)[:3] == pixel(gradient, 10, 10)
    assert pixel(doc.flatten(), 10, 10) == (255, 0, 0, 255)  # nothing committed
    assert doc.render_mask(1.0, override=hidden) is not None


# ------------------------------------------------------------- operations --
def test_add_mask_modes_and_errors(gradient: Image.Image) -> None:
    doc = red_over_gradient(gradient)
    doc.add_mask(mode="hide")
    assert doc.active_layer.mask is not None and doc.active_layer.mask.getextrema() == (0, 0)
    assert doc.editing_mask and doc.history.labels[-1] == "Add Layer Mask"
    with pytest.raises(DocumentError):
        doc.add_mask()
    doc.remove_mask()
    assert doc.active_layer.mask is None and not doc.editing_mask
    with pytest.raises(DocumentError):
        doc.remove_mask()
    with pytest.raises(DocumentError):
        doc.add_mask(mode="sideways")
    # from a selection: white inside, black outside; "hide" inverts
    doc.set_selection(Selection.rectangle(doc.size, (0, 0, 32, 48)))
    doc.add_mask(mode="selection")
    assert gray(mask_of(doc), 5, 5) == 255 and gray(mask_of(doc), 50, 5) == 0
    doc.remove_mask()
    doc.add_mask(mode="hide")
    assert gray(mask_of(doc), 5, 5) == 0 and gray(mask_of(doc), 50, 5) == 255
    # locked layers can't get masks
    doc.remove_mask()
    doc.set_layer_props(locked=True)
    with pytest.raises(DocumentError):
        doc.add_mask()


def test_invert_apply_and_select_mask(gradient: Image.Image) -> None:
    doc = half_masked(gradient)
    doc.invert_mask()
    assert pixel(doc.flatten(), 10, 10)[:3] == pixel(gradient, 10, 10) and pixel(doc.flatten(), 50, 10) == (255, 0, 0, 255)
    doc.select_mask()
    assert not sel(doc).contains(10, 10) and sel(doc).contains(50, 10)
    doc.remove_mask(apply=True)
    layer = doc.active_layer
    assert layer.mask is None and pixel(layer.image, 10, 10)[3] == 0 and pixel(layer.image, 50, 10) == (255, 0, 0, 255)
    for op in (doc.invert_mask, doc.select_mask):
        with pytest.raises(DocumentError):
            op()
    with pytest.raises(DocumentError):
        doc.set_mask_enabled(False)


def test_painting_on_the_mask(gradient: Image.Image) -> None:
    doc = red_over_gradient(gradient)
    doc.add_mask()
    assert doc.edit_image().mode == "RGBA" and pixel(doc.edit_image(), 0, 0) == (255, 255, 255, 255)
    # painting black on the mask hides the red; the pixels are untouched
    before = doc.active_layer.image
    doc.set_selection(Selection.rectangle(doc.size, (0, 0, 20, 20)))
    doc.fill("#000000")
    assert doc.history.labels[-1] == "Fill (Mask)"
    assert doc.active_layer.image is before
    assert gray(mask_of(doc), 5, 5) == 0 and gray(mask_of(doc), 40, 40) == 255
    assert pixel(doc.flatten(), 5, 5)[:3] == pixel(gradient, 5, 5) and pixel(doc.flatten(), 40, 40) == (255, 0, 0, 255)
    doc.deselect()
    # the eraser hides as well (erased = black), undo brings the mask back
    cov = np.zeros((doc.height, doc.width), np.float32)
    cov[30:40, 30:40] = 1.0
    doc.erase(cov)
    assert gray(mask_of(doc), 35, 35) == 0
    assert doc.undo() and gray(mask_of(doc), 35, 35) == 255
    # switching the target back edits pixels again
    doc.set_mask_editing(False)
    assert not doc.editing_mask and doc.edit_image() is doc.active_layer.image
    doc.fill("#00ff00")
    assert pixel(doc.active_layer.image, 40, 40) == (0, 255, 0, 255) and doc.history.labels[-1] == "Fill"
    # filters work on the mask too (e.g. blurring its edge)
    doc.set_mask_editing(True)
    doc.layers[1] = doc.layers[1].copy(mask=half_mask(doc.size))
    doc.apply_filter("Gaussian Blur", radius=4)
    edge = gray(mask_of(doc), 32, 10)
    assert 0 < edge < 255
    # the flag alone is not enough: a layer without a mask never reports mask editing
    doc.set_active(0)
    assert not doc.editing_mask
    doc.set_mask_editing(True)
    assert not doc.mask_editing


def test_move_keeps_mask_linked(gradient: Image.Image) -> None:
    doc = half_masked(gradient)
    doc.set_mask_editing(False)
    doc.offset_active(10, 0)
    mask = doc.active_layer.mask
    assert mask is not None and gray(mask, 5, 5) == 0 and gray(mask, 15, 5) == 255 and gray(mask, 41, 5) == 255
    assert doc.undo() and gray(mask_of(doc), 5, 5) == 255
    # while editing the mask only the mask moves
    doc.set_mask_editing(True)
    doc.offset_active(-10, 0)
    assert pixel(doc.active_layer.image, 60, 10) == (255, 0, 0, 255)  # pixels stayed
    assert gray(mask_of(doc), 25, 5) == 0


def test_transforms_carry_masks(gradient: Image.Image) -> None:
    doc = half_masked(gradient)
    doc.rotate90()
    mask = doc.active_layer.mask
    assert mask is not None and mask.size == doc.size == (48, 64)
    doc.flip()
    doc.resize(32, 24)
    assert doc.active_layer.mask is not None and doc.active_layer.mask.size == (32, 24)
    doc.canvas_size(40, 30)
    doc.crop((0, 0, 20, 20))
    doc.rotate(15)
    doc.straighten(3)
    doc.scale(2)
    for layer in doc.layers:
        assert layer.mask is None or layer.mask.size == doc.size
    assert doc.active_layer.mask is not None
    # merging bakes masks in
    doc.merge_down()
    assert len(doc.layers) == 1 and doc.layers[0].mask is None


def test_duplicate_and_undo_preserve_masks(gradient: Image.Image) -> None:
    doc = red_over_gradient(gradient)
    doc.add_mask()
    doc.duplicate_layer()
    assert doc.active_layer.mask is not None and doc.active_layer.mask is not doc.layers[1].mask
    doc.remove_mask()
    assert doc.active_layer.mask is None
    assert doc.undo() and doc.active_layer.mask is not None


def test_project_roundtrip_with_masks(tmp_path: Path, gradient: Image.Image) -> None:
    doc = red_over_gradient(gradient)
    doc.add_mask()
    doc.layers[1] = doc.layers[1].copy(mask=half_mask(doc.size), mask_enabled=False)
    path = tmp_path / "masked.pfp"
    pfio.save_project(doc, path)
    loaded = pfio.load_project(path)
    layer = loaded.layers[1]
    assert layer.mask is not None and not layer.mask_enabled and layer.mask.tobytes() == half_mask(doc.size).tobytes()
    assert loaded.layers[0].mask is None
    loaded.set_mask_enabled(True, 1)
    assert pixel(loaded.flatten(), 50, 10)[:3] == pixel(gradient, 50, 10)
