import numpy as np
import pytest
from PIL import Image

from pixelforge.core.adjustments import default_adjustments
from pixelforge.core.document import ANCHORS, Document, DocumentError
from pixelforge.core.layers import Layer
from pixelforge.core.painting import Stroke
from pixelforge.core.selection import Selection
from tests.conftest import pixel_diff
from tests.helpers import copied, pixel, sel


# ------------------------------------------------------------ construction --
def test_new_document_backgrounds() -> None:
    for bg, px in (
        ("white", (255, 255, 255, 255)),
        ("black", (0, 0, 0, 255)),
        ("transparent", (0, 0, 0, 0)),
        ("#ff0000", (255, 0, 0, 255)),
    ):
        doc = Document.new(10, 6, bg)
        assert doc.size == (10, 6)
        assert pixel(doc.active_layer.image, 0, 0) == px
        assert doc.history.labels == ["New"]
        assert not doc.dirty
    with pytest.raises(DocumentError):
        Document.new(0, 5)
    with pytest.raises(DocumentError):
        Document.new(50000, 5)


def test_from_image_and_basic_properties(gradient: Image.Image) -> None:
    doc = Document.from_image(gradient, "g")
    assert doc.size == gradient.size and doc.name == "g"
    assert len(doc.layers) == 1 and doc.active_layer.kind == "background"
    assert doc.active_layer.image.mode == "RGBA"
    assert doc.history.labels == ["Open"] and not doc.can_undo and not doc.can_redo
    assert doc.layer_index(doc.active_layer.id) == 0 and doc.layer_index(999999) is None
    assert doc.selection_mask() is None
    with pytest.raises(DocumentError):
        Document(10, 10, [Layer.blank((5, 5))])


def test_from_image_applies_exif_orientation(gradient: Image.Image) -> None:
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 270 CW
    gradient.info["exif"] = exif.tobytes()
    import io

    buf = io.BytesIO()
    gradient.save(buf, "JPEG", exif=exif.tobytes())
    buf.seek(0)
    doc = Document.from_image(Image.open(buf))
    assert doc.size == (gradient.height, gradient.width)


# --------------------------------------------------------------- rendering --
def test_composite_single_layer_equals_pixels(doc: Document, gradient: Image.Image) -> None:
    flat = doc.flatten()
    assert flat.mode == "RGBA" and pixel_diff(flat, gradient) == 0
    half = doc.composite(0.5)
    assert half.size == (32, 24)
    tiny = doc.thumbnail((16, 16))
    assert max(tiny.size) == 16


def test_composite_view_box_matches_crop_of_full(two_layer_doc: Document) -> None:
    doc = two_layer_doc
    doc.set_layer_props(blend_mode="multiply", opacity=0.7)
    full = doc.composite(1.0)
    region = doc.composite(1.0, (10, 5, 40, 30))
    assert region.size == (30, 25)
    assert pixel_diff(region, full.crop((10, 5, 40, 30))) == 0
    zoomed = doc.composite(2.0, (20, 10, 80, 60))
    assert zoomed.size == (60, 50)
    assert pixel_diff(zoomed, full.crop((10, 5, 40, 30)).resize((60, 50), Image.Resampling.NEAREST)) == 0
    # out-of-range boxes are clipped
    assert doc.composite(1.0, (-5, -5, 10, 10)).size == (10, 10)
    assert doc.composite(1.0, (60, 40, 70, 50)).size == (4, 8)
    assert doc.composite(1.0, (100, 100, 120, 120)).size == (1, 1)
    with pytest.raises(ValueError):
        doc.composite(0)


def test_composite_respects_visibility_opacity_and_overrides(two_layer_doc: Document, gradient: Image.Image, shapes: Image.Image) -> None:
    doc = two_layer_doc
    assert pixel(doc.flatten(), 10, 10) == (255, 0, 0, 255)
    doc.set_layer_props(visible=False)
    assert pixel(doc.flatten(), 10, 10) == pixel(gradient.convert("RGBA"), 10, 10)
    assert pixel(doc.composite(include_hidden=True), 10, 10) == (255, 0, 0, 255)
    doc.set_layer_props(visible=True, opacity=0.5)
    px = pixel(doc.flatten(), 10, 10)
    base = pixel(gradient, 10, 10)
    assert abs(px[0] - (255 + base[0]) / 2) <= 1
    doc.set_layer_props(opacity=1.0)
    override = Image.new("RGBA", doc.size, (0, 255, 0, 255))
    assert pixel(doc.composite(overrides={doc.active_layer.id: override}), 30, 5)[1] > 200
    # the cache must not return the overridden result afterwards
    assert pixel(doc.flatten(), 30, 5)[1] < 200


def test_blend_modes_in_composite(gradient: Image.Image) -> None:
    doc = Document.from_image(gradient)
    doc.add_layer(Layer(Image.new("RGBA", doc.size, (0, 0, 0, 255)), "Black"))
    doc.set_layer_props(blend_mode="multiply")
    assert pixel(doc.flatten(), 30, 20)[:3] == (0, 0, 0)
    doc.set_layer_props(blend_mode="screen")
    assert pixel(doc.flatten(), 30, 20)[:3] == pixel(gradient, 30, 20)
    doc.set_layer_props(blend_mode="difference")
    assert pixel(doc.flatten(), 30, 20)[:3] == pixel(gradient, 30, 20)


def test_render_cache_invalidation_on_live_edits(doc: Document) -> None:
    before = pixel(doc.flatten(), 60, 40)
    doc.set_adjustment("brightness", 80)  # live, uncommitted
    after = pixel(doc.flatten(), 60, 40)
    assert after != before
    doc.active_layer.adjustments["brightness"] = 0
    doc.touch()
    assert pixel(doc.flatten(), 60, 40) == before


def test_export_image_modes(shapes: Image.Image) -> None:
    doc = Document.from_image(shapes)
    rgba = doc.export_image("RGBA")
    assert rgba.mode == "RGBA" and pixel(rgba, 0, 0)[3] == 0
    rgb = doc.export_image("RGB")
    assert rgb.mode == "RGB" and pixel(rgb, 0, 0) == (255, 255, 255)
    black = doc.export_image("RGB", background=(0, 0, 0))
    assert pixel(black, 0, 0) == (0, 0, 0)


def test_pick_color(two_layer_doc: Document, gradient: Image.Image) -> None:
    doc = two_layer_doc
    assert doc.pick_color(10, 10) == (255, 0, 0, 255)
    assert doc.pick_color(10, 10, merged=False) == (255, 0, 0, 255)
    doc.set_active(0)
    assert doc.pick_color(10.7, 10.2, merged=False) == pixel(gradient.convert("RGBA"), 10, 10)
    assert doc.pick_color(-1, 0) is None and doc.pick_color(0, 1000) is None


# ------------------------------------------------------------------ layers --
def test_add_remove_duplicate_move_layers(doc: Document) -> None:
    layer = doc.add_layer()
    assert len(doc.layers) == 2 and doc.active_layer is layer and layer.name == "Layer 2"
    assert doc.history.labels[-1] == "New Layer 'Layer 2'"
    dup = doc.duplicate_layer()
    assert len(doc.layers) == 3 and doc.active_index == 2 and dup.name == "Layer 2 copy"
    doc.lower_layer()
    assert doc.active_index == 1 and doc.layers[1] is dup
    doc.raise_layer()
    assert doc.active_index == 2
    doc.move_layer(2, 0)
    assert doc.layers[0] is dup and doc.active_index == 0
    doc.move_layer(0, 0)  # no-op
    doc.remove_layer(0)
    assert len(doc.layers) == 2 and dup not in doc.layers
    doc.remove_layer()
    with pytest.raises(DocumentError):
        doc.remove_layer()
    with pytest.raises(DocumentError):
        doc.set_active(5)
    with pytest.raises(DocumentError):
        doc.add_layer(Layer.blank((3, 3)))
    named = doc.add_layer(name="Custom")
    assert named.name == "Custom"
    doc.add_layer(index=0, name="Bottom")
    assert doc.layers[0].name == "Bottom" and doc.active_index == 0


def test_add_image_layer_centres_or_positions(doc: Document) -> None:
    small = Image.new("RGBA", (10, 10), (0, 255, 0, 255))
    layer = doc.add_image_layer(small, "Small")
    assert pixel(layer.image, 32, 24) == (0, 255, 0, 255)
    assert pixel(layer.image, 0, 0)[3] == 0
    layer2 = doc.add_image_layer(small, "Pos", (0, 0))
    assert pixel(layer2.image, 0, 0) == (0, 255, 0, 255)
    assert doc.paste(small, None, "Pasted").name == "Pasted"


def test_layer_props_and_locking(doc: Document) -> None:
    doc.set_layer_props(name="Photo", opacity=0.5, blend_mode="overlay", locked=True, visible=False)
    layer = doc.active_layer
    assert (layer.name, layer.opacity, layer.blend_mode, layer.locked, layer.visible) == ("Photo", 0.5, "overlay", True, False)
    assert doc.history.labels[-1].startswith("Layer ")
    with pytest.raises(DocumentError):
        doc.set_layer_props(color="red")
    with pytest.raises(DocumentError):
        doc.fill("#fff")  # locked
    with pytest.raises(DocumentError):
        doc.bake_adjustments()
    doc.set_layer_props(locked=False, commit=False)
    doc.fill("#fff")
    assert pixel(doc.active_layer.image, 1, 1) == (255, 255, 255, 255)


def test_adjustments_live_commit_reset_bake(doc: Document, gradient: Image.Image) -> None:
    doc.set_adjustment("contrast", 40)
    assert doc.active_layer.adjustments["contrast"] == 40
    assert doc.history.labels == ["Open"]  # live edit not recorded
    doc.set_adjustment("contrast", 40, commit=True)
    assert doc.history.labels[-1] == "Adjust Contrast"
    doc.set_adjustments({"saturation": 30}, commit=True, label="Look")
    assert doc.active_layer.adjustments["contrast"] == 0 and doc.active_layer.adjustments["saturation"] == 30
    rendered = doc.flatten()
    doc.bake_adjustments()
    assert doc.active_layer.adjustments == default_adjustments()
    assert pixel_diff(doc.active_layer.image, rendered) == 0
    assert doc.history.labels[-1] == "Apply Adjustments"
    doc.bake_adjustments()  # nothing to do
    assert doc.history.labels[-1] == "Apply Adjustments"
    doc.set_adjustments({"hue": 90})
    doc.reset_adjustments()
    assert not doc.active_layer.has_adjustments and doc.history.labels[-1] == "Reset Adjustments"


def test_bake_respects_selection(doc: Document, gradient: Image.Image) -> None:
    doc.set_selection(Selection.rectangle(doc.size, (0, 0, 32, 48)))
    doc.set_adjustments({"grayscale": 100})
    doc.bake_adjustments()
    left, right = pixel(doc.active_layer.image, 10, 10), pixel(doc.active_layer.image, 50, 10)
    assert left[0] == left[1] == left[2]
    assert right[:3] == pixel(gradient, 50, 10)


def test_merge_down_and_flatten(two_layer_doc: Document, gradient: Image.Image) -> None:
    doc = two_layer_doc
    doc.set_layer_props(opacity=0.5, blend_mode="normal")
    doc.layers[0].adjustments["grayscale"] = 100
    expected = doc.flatten()
    doc.merge_down()
    assert len(doc.layers) == 1 and doc.active_index == 0
    assert pixel_diff(doc.active_layer.image, expected) <= 1
    assert doc.active_layer.adjustments == default_adjustments()
    assert doc.history.labels[-1] == "Merge 'Shapes' Down"
    with pytest.raises(DocumentError):
        doc.merge_down()
    doc.add_layer()
    doc.add_layer()
    doc.flatten_layers()
    assert len(doc.layers) == 1 and doc.active_layer.kind == "background"


def test_merge_visible(two_layer_doc: Document) -> None:
    doc = two_layer_doc
    doc.add_layer(name="Hidden")
    doc.set_layer_props(visible=False)
    flat = doc.flatten()
    doc.merge_visible()
    assert [layer.name for layer in doc.layers] == ["Merged", "Hidden"]
    assert pixel_diff(doc.active_layer.image, flat) == 0
    doc.merge_visible()  # only one visible layer: no-op
    assert len(doc.layers) == 2


# ------------------------------------------------------------- pixel edits --
def test_apply_to_active_with_selection(doc: Document, gradient: Image.Image) -> None:
    doc.set_selection(Selection.rectangle(doc.size, (0, 0, 32, 48)))
    doc.apply_filter("Invert")
    img = doc.active_layer.image
    assert pixel(img, 10, 10)[:3] == tuple(255 - c for c in pixel(gradient, 10, 10))
    assert pixel(img, 50, 10)[:3] == pixel(gradient, 50, 10)
    assert doc.history.labels[-1] == "Invert"
    with pytest.raises(DocumentError):
        doc.apply_to_active(lambda im: im.resize((3, 3)), "Bad")
    preview = doc.preview_active(lambda im: Image.new("RGBA", im.size, (0, 255, 0, 255)))
    assert pixel(preview, 10, 10) == (0, 255, 0, 255) and pixel(preview, 50, 10)[:3] == pixel(gradient, 50, 10)
    assert doc.history.labels[-1] == "Invert"  # preview does not commit


def test_fill_clear_paint_erase(doc: Document) -> None:
    doc.set_selection(Selection.rectangle(doc.size, (0, 0, 10, 10)))
    doc.fill("#00ff00")
    assert pixel(doc.active_layer.image, 5, 5) == (0, 255, 0, 255)
    assert pixel(doc.active_layer.image, 20, 20)[1] != 255
    doc.clear()
    assert pixel(doc.active_layer.image, 5, 5)[3] == 0
    doc.deselect()
    stroke = Stroke(doc.size, 8, 1.0)
    stroke.add_point(30, 30)
    doc.paint(stroke.coverage, "#0000ff", 1.0)
    assert pixel(doc.active_layer.image, 30, 30) == (0, 0, 255, 255)
    doc.erase(stroke.coverage)
    assert pixel(doc.active_layer.image, 30, 30)[3] == 0
    assert doc.history.labels[-3:] == ["Deselect", "Brush Stroke", "Eraser"]


def test_gradient_shape_and_text_layers(doc: Document) -> None:
    doc.gradient((0, 0), (63, 0), "#000000", "#ffffff", "linear")
    img = doc.active_layer.image
    assert pixel(img, 0, 10)[0] < 5 and pixel(img, 63, 10)[0] > 250
    doc.draw_shape("rectangle", (10, 10, 30, 30), fill_color="#ff0000")
    assert pixel(doc.active_layer.image, 20, 20)[:3] == (255, 0, 0)
    layer = doc.add_text_layer("Hi", (2, 2), "#00ff00", None, 20)
    assert layer.kind == "text" and layer.name == "T Hi" and len(doc.layers) == 2
    assert layer.image.getbbox() is not None
    updated = doc.update_text_layer(text="Bye", color=(0, 0, 255, 255))
    assert updated.name == "T Bye" and updated.meta["text"] == "Bye"
    pixels = np.asarray(updated.image)
    assert (pixels[pixels[..., 3] > 200][:, 2] > 200).any()
    doc.set_active(0)
    with pytest.raises(DocumentError):
        doc.update_text_layer(text="x")


def test_offset_active_moves_pixels_and_selection(shapes: Image.Image) -> None:
    doc = Document.from_image(shapes)
    doc.set_selection(Selection.rectangle(doc.size, (4, 4, 25, 25)))
    doc.offset_active(10, 5)
    assert pixel(doc.active_layer.image, 14, 9) == (255, 0, 0, 255)
    assert pixel(doc.active_layer.image, 4, 4)[3] == 0
    assert sel(doc).bbox == (14, 9, 35, 30)
    doc.offset_active(0, 0)  # no-op
    assert doc.history.labels[-1] == "Move"
    assert doc.history.labels.count("Move") == 1


def test_bucket_fill(shapes: Image.Image) -> None:
    doc = Document.from_image(shapes)
    doc.bucket_fill(10, 10, "#00ff00", tolerance=10)  # the red square
    assert pixel(doc.active_layer.image, 10, 10) == (0, 255, 0, 255)
    assert pixel(doc.active_layer.image, 44, 25) == (0, 0, 255, 255)
    doc.bucket_fill(0, 0, "#ffffff", tolerance=0)  # transparent background
    assert pixel(doc.active_layer.image, 0, 0) == (255, 255, 255, 255)
    assert pixel(doc.active_layer.image, 44, 25) == (0, 0, 255, 255)
    doc.bucket_fill(-5, -5, "#000")  # outside: ignored
    assert doc.history.labels[-1] == "Paint Bucket" and doc.history.labels.count("Paint Bucket") == 2
    doc.set_selection(Selection.rectangle(doc.size, (0, 0, 5, 5)))
    doc.bucket_fill(1, 1, "#ff00ff", tolerance=0)
    assert pixel(doc.active_layer.image, 1, 1)[:3] == (255, 0, 255)
    assert pixel(doc.active_layer.image, 20, 40)[:3] == (255, 255, 255)  # outside selection untouched


# -------------------------------------------------------------- transforms --
def test_rotate_and_flip(doc: Document, gradient: Image.Image) -> None:
    doc.rotate90(True)
    assert doc.size == (48, 64)
    assert pixel(doc.active_layer.image, 47, 0)[:3] == pixel(gradient, 0, 0)
    doc.rotate90(False)
    assert pixel_diff(doc.active_layer.image, gradient) == 0
    doc.rotate180()
    assert pixel(doc.active_layer.image, 63, 47)[:3] == pixel(gradient, 0, 0)
    doc.rotate180()
    doc.flip(True)
    assert pixel(doc.active_layer.image, 63, 0)[:3] == pixel(gradient, 0, 0)
    doc.flip(True)
    doc.flip(False)
    assert pixel(doc.active_layer.image, 0, 47)[:3] == pixel(gradient, 0, 0)
    assert doc.history.labels[-1] == "Flip Vertical"


def test_arbitrary_rotation_and_selection_reset(doc: Document) -> None:
    doc.select_all()
    doc.rotate(45)
    assert doc.size[0] > 64 and doc.size[1] > 48
    assert doc.selection is None
    assert pixel(doc.active_layer.image, 0, 0)[3] == 0  # transparent corner
    doc.rotate(0)  # no-op
    assert doc.history.labels[-1] == "Rotate 45°"
    doc2 = Document.new(20, 10, "white")
    doc2.rotate(90, expand=False, fill=(255, 0, 0, 255))
    assert doc2.size == (20, 10) and pixel(doc2.active_layer.image, 0, 0) == (255, 0, 0, 255)


def test_resize_scale_and_canvas_size(two_layer_doc: Document) -> None:
    doc = two_layer_doc
    doc.set_selection(Selection.rectangle(doc.size, (0, 0, 32, 24)))
    doc.resize(128, 96)
    assert doc.size == (128, 96) and all(layer.size == (128, 96) for layer in doc.layers)
    assert sel(doc).contains(63, 47) and not sel(doc).contains(66, 50)
    bbox = sel(doc).bbox
    assert bbox is not None and bbox[2] <= 65 and bbox[3] <= 49  # at most a partial pixel of bilinear softness
    doc.scale(0.5)
    assert doc.size == (64, 48)
    doc.resize(64, 48)  # no-op
    with pytest.raises(DocumentError):
        doc.resize(0, 10)
    doc.canvas_size(80, 60, "center")
    assert doc.size == (80, 60)
    assert pixel(doc.layers[0].image, 0, 0)[3] == 0  # new area transparent
    assert pixel(doc.layers[0].image, 8 + 5, 6 + 5)[3] == 255
    assert sel(doc).contains(9, 7) and sel(doc).contains(39, 29) and not sel(doc).contains(7, 5)
    doc.canvas_size(60, 40, "nw", fill=(255, 255, 255, 255))
    assert doc.size == (60, 40)
    with pytest.raises(DocumentError):
        doc.canvas_size(10, 10, "middle")


@pytest.mark.parametrize("anchor", ANCHORS)
def test_canvas_size_anchors(anchor: str) -> None:
    doc = Document.new(10, 10, "black")
    doc.canvas_size(30, 30, anchor)
    alpha = np.asarray(doc.active_layer.image)[..., 3]
    ys, xs = np.nonzero(alpha)
    expected_x = {"w": 0, "e": 20}.get(anchor[-1] if anchor != "center" else "", 10)
    expected_y = {"n": 0, "s": 20}.get(anchor[0] if anchor != "center" else "", 10)
    assert xs.min() == expected_x and ys.min() == expected_y


def test_crop_variants(doc: Document, gradient: Image.Image) -> None:
    doc.crop((10, 5, 40, 35))
    assert doc.size == (30, 30)
    assert pixel(doc.active_layer.image, 0, 0)[:3] == pixel(gradient, 10, 5)
    doc.crop((-10, -10, 1000, 1000))  # clipped to the image
    assert doc.size == (30, 30)
    with pytest.raises(DocumentError):
        doc.crop((5, 5, 5, 20))
    doc.set_selection(Selection.ellipse(doc.size, (5, 5, 25, 25)))
    doc.crop_to_selection()
    assert doc.size == (20, 20) and doc.selection is None
    assert doc.history.labels[-1] == "Crop to Selection"
    with pytest.raises(DocumentError):
        doc.crop_to_selection()


def test_trim(shapes: Image.Image) -> None:
    doc = Document.from_image(shapes)
    doc.trim()
    assert doc.size == (55, 37)  # bbox of the red square + blue circle
    assert doc.history.labels[-1] == "Trim"
    doc.trim()  # nothing more to trim
    assert doc.history.labels.count("Trim") == 1


# --------------------------------------------------------------- selection --
def test_selection_operations(doc: Document) -> None:
    doc.select_all()
    assert sel(doc).is_full and doc.history.labels[-1] == "Selection"
    doc.deselect()
    assert doc.selection is None and doc.history.labels[-1] == "Deselect"
    doc.deselect()  # no-op, no new history entry
    assert doc.history.labels.count("Deselect") == 1
    doc.set_selection(Selection.rectangle(doc.size, (0, 0, 10, 10)))
    doc.set_selection(Selection.rectangle(doc.size, (5, 5, 20, 20)), mode="add")
    assert sel(doc).bbox == (0, 0, 20, 20)
    doc.set_selection(Selection.rectangle(doc.size, (0, 0, 10, 10)), mode="subtract")
    assert not sel(doc).contains(2, 2) and sel(doc).contains(15, 15)
    doc.invert_selection()
    assert sel(doc).contains(2, 2) and not sel(doc).contains(15, 15)
    assert doc.history.labels[-1] == "Invert Selection"
    doc.set_selection(Selection.empty(doc.size))
    assert doc.selection is None
    with pytest.raises(DocumentError):
        doc.set_selection(Selection.all((3, 3)))
    doc.set_selection(Selection.rectangle(doc.size, (10, 10, 20, 20)))
    doc.modify_selection("grow", 2)
    assert sel(doc).bbox == (8, 8, 22, 22)
    doc.modify_selection("shrink", 2)
    assert sel(doc).bbox == (10, 10, 20, 20)
    doc.modify_selection("feather", 1.5)
    bbox = sel(doc).bbox
    assert bbox is not None and bbox[0] < 10
    doc.modify_selection("border", 1)
    with pytest.raises(DocumentError):
        doc.modify_selection("wobble", 1)
    doc.deselect()
    with pytest.raises(DocumentError):
        doc.modify_selection("grow", 1)
    doc.invert_selection()  # inverse of nothing selects all
    assert sel(doc).is_full


def test_magic_wand_and_layer_alpha_selection(two_layer_doc: Document) -> None:
    doc = two_layer_doc
    doc.magic_wand(10, 10, tolerance=5)
    assert sel(doc).contains(10, 10) and not sel(doc).contains(44, 25)
    doc.magic_wand(44, 25, tolerance=5, mode="add")
    assert sel(doc).contains(10, 10) and sel(doc).contains(44, 25)
    doc.magic_wand(-1, -1)  # ignored
    doc.select_layer_alpha()
    assert sel(doc).contains(10, 10) and not sel(doc).contains(0, 0)
    doc.magic_wand(0, 0, tolerance=0, merged=False)
    assert sel(doc).contains(0, 0) and not sel(doc).contains(10, 10)


def test_copy_selection_and_paste(two_layer_doc: Document) -> None:
    doc = two_layer_doc
    img, pos = copied(doc)
    assert img.size == doc.size and pos == (0, 0)
    doc.set_selection(Selection.rectangle(doc.size, (4, 4, 25, 25)))
    img, pos = copied(doc)
    assert img.size == (21, 21) and pos == (4, 4) and pixel(img, 0, 0) == (255, 0, 0, 255)
    merged, _ = copied(doc, merged=True)
    assert pixel(merged, 0, 0) == (255, 0, 0, 255)
    doc.set_selection(Selection.ellipse(doc.size, (30, 10, 58, 40)))
    piece, _ = copied(doc)
    assert pixel(piece, 0, 0)[3] == 0  # masked by the ellipse
    layer = doc.paste(piece, (0, 0))
    assert len(doc.layers) == 3 and layer.name == "Pasted"


# ----------------------------------------------------------------- history --
def test_undo_redo_restores_everything(doc: Document, gradient: Image.Image) -> None:
    doc.add_layer(name="L2")
    doc.fill("#ff0000")
    doc.set_selection(Selection.rectangle(doc.size, (0, 0, 5, 5)))
    doc.rotate90(True)
    assert doc.size == (48, 64)
    assert doc.undo() and doc.size == (64, 48) and doc.selection is not None
    assert doc.undo() and doc.selection is None
    assert doc.undo() and pixel(doc.active_layer.image, 1, 1)[3] == 0
    assert doc.undo() and len(doc.layers) == 1
    assert not doc.undo()
    assert doc.redo() and len(doc.layers) == 2
    assert doc.redo() and pixel(doc.active_layer.image, 1, 1) == (255, 0, 0, 255)
    doc.go_to_history(4)
    assert doc.size == (48, 64)
    doc.go_to_history(0)
    doc.fill("#000")  # new branch discards redo
    assert not doc.can_redo and len(doc.history) == 2


def test_dirty_flag_and_revert(doc: Document) -> None:
    assert not doc.dirty
    doc.fill("#000")
    assert doc.dirty
    doc.mark_saved()
    assert not doc.dirty
    doc.undo()
    assert doc.dirty
    doc.redo()
    assert not doc.dirty
    doc.fill("#fff")
    doc.revert()
    assert doc.history.labels[-1] == "Revert"
    assert pixel(doc.active_layer.image, 5, 5)[:3] != (255, 255, 255)
    fresh = Document.new(4, 4)
    fresh.revert()  # nothing to revert
    assert fresh.history.labels == ["New"]


def test_history_snapshots_are_isolated_from_live_edits(doc: Document) -> None:
    doc.fill("#ff0000")
    doc.active_layer.adjustments["contrast"] = 77  # live mutation without commit
    doc.undo()
    doc.redo()
    assert doc.active_layer.adjustments["contrast"] == 0  # snapshot was a copy


def test_render_cache_is_bounded(doc: Document) -> None:
    cache = doc._cache
    cache.max_bytes = 64 * 48 * 4 * 3  # room for ~3 full renders
    for i in range(10):
        doc.composite(1.0, (0, 0, 64, 48 - i))
    assert len(cache) <= cache.capacity
    assert cache._bytes <= cache.max_bytes
    doc.composite(1.0, (0, 0, 64, 48))
    doc.composite(1.0, (0, 0, 64, 48))  # same key: no growth
    before = len(cache)
    doc.composite(1.0, (0, 0, 64, 48))
    assert len(cache) == before
    doc.pick_color(3, 3)  # uncached one-off render does not pollute the cache
    assert len(cache) == before
    cache.clear()
    assert len(cache) == 0 and cache._bytes == 0


def test_inscribed_box_and_straighten(gradient: Image.Image) -> None:
    from pixelforge.core.document import inscribed_box

    assert inscribed_box(400, 300, 0) == (0, 0, 400, 300)
    left, top, right, bottom = inscribed_box(400, 300, 5)
    assert 0 < left < 30 and 0 < top < 30 and right - left < 400 and bottom - top < 300
    assert inscribed_box(300, 400, -10)[2] < 300  # portrait
    l2, t2, r2, b2 = inscribed_box(100, 100, 45)
    assert abs((r2 - l2) - 69) <= 2 and abs((b2 - t2) - 69) <= 2  # 100 / sqrt(2)
    doc = Document.from_image(Image.new("RGB", (400, 300), (200, 50, 50)))
    doc.straighten(5)
    assert doc.size == (right - left, bottom - top)
    assert doc.history.labels[-1] == "Straighten +5.0°"
    arr = np.asarray(doc.active_layer.image)
    assert (arr[..., 3] == 255).all()  # no empty corners survive the crop
    doc.straighten(0)  # no-op
    assert doc.history.labels.count("Straighten +5.0°") == 1
    expanded = Document.from_image(Image.new("RGB", (400, 300)))
    expanded.straighten(-8, crop=False)
    assert expanded.size[0] > 400 and expanded.size[1] > 300 and expanded.history.labels[-1] == "Straighten -8.0°"
    assert pixel(expanded.active_layer.image, 0, 0)[3] == 0


def test_composite_neutral_ignores_adjustments(doc: Document, gradient: Image.Image) -> None:
    doc.set_adjustments({"brightness": 80, "grayscale": 100}, commit=True)
    adjusted = doc.composite()
    before = doc.composite(neutral=True)
    assert pixel_diff(before, gradient) == 0
    assert pixel_diff(adjusted, gradient) > 0
    assert pixel_diff(doc.composite(neutral=True), before) == 0  # cached path agrees


def test_trim_reports_result(shapes: Image.Image) -> None:
    doc = Document.from_image(shapes)
    assert doc.trim() is True and doc.size == (55, 37)
    assert doc.trim() is False


def test_snapshots_survive_undo(doc: Document) -> None:
    doc.fill("#ff0000")
    index = doc.take_snapshot("  ")  # empty names get a default
    assert doc.snapshots[0][0] == "Snapshot 1" and index == 0
    doc.fill("#00ff00")
    doc.undo()
    doc.undo()
    assert pixel(doc.active_layer.image, 1, 1)[:3] != (255, 0, 0)
    doc.restore_snapshot(0)
    assert pixel(doc.active_layer.image, 1, 1) == (255, 0, 0, 255)
    assert doc.history.labels[-1] == "Snapshot: Snapshot 1" and len(doc.snapshots) == 1
    doc.rename_snapshot(0, "Red")
    doc.rename_snapshot(0, "")
    assert doc.snapshots[0][0] == "Red"
    doc.delete_snapshot(0)
    assert doc.snapshots == []


def test_history_depth_adapts_to_picture_size(monkeypatch: pytest.MonkeyPatch) -> None:
    from pixelforge.core import document as docmod

    small = Document.from_image(Image.new("RGB", (800, 600)))
    assert small.history_capacity() == docmod.HISTORY_MAX_STEPS
    huge = Document.from_image(Image.new("RGB", (6000, 4000)))
    assert docmod.HISTORY_MIN_STEPS <= huge.history_capacity() < docmod.HISTORY_MAX_STEPS
    assert small.dirty is False
    small.mark_dirty()
    assert small.dirty
    # the real limit is memory: states are dropped (oldest first) once the budget is exceeded
    monkeypatch.setattr(docmod, "HISTORY_BUDGET_BYTES", 40 * 1024 * 1024)
    monkeypatch.setattr(docmod, "HISTORY_MIN_STEPS", 2)
    doc = Document.from_image(Image.new("RGB", (2000, 1500)))  # 12 MB per full RGBA image
    colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00"]
    for i in range(8):
        doc.fill(colors[i % 4])  # every fill changes everything: no patch possible
    assert 2 <= len(doc.history) <= 4 and doc.history_bytes() <= 40 * 1024 * 1024 + 12 * 1024 * 1024
    assert doc.can_undo and doc.undo()


def test_history_keeps_small_edits_as_patches(gradient: Image.Image) -> None:
    from pixelforge.core import document as docmod

    doc = Document.from_image(gradient.resize((400, 300)))
    full = 400 * 300 * 4
    before = doc.history_bytes()
    for i in range(10):
        doc.set_selection(Selection.rectangle(doc.size, (i * 10, 0, i * 10 + 10, 10)))
        doc.fill("#ff0000")
    doc.deselect()
    # 10 fills of 10×10 px: the older states shrank to patches instead of 10 more full images
    assert doc.history_bytes() < before + 2 * full
    assert len(doc.history) == 22
    states = doc.history.states
    patched = [st.snapshot for st in states if st.snapshot.patches]
    assert patched and all(st.layers[0].image is docmod._PLACEHOLDER for st in patched)
    # undo walks the patch chain back to exact pixels
    for _ in range(21):
        assert doc.undo()
    assert pixel(doc.active_layer.image, 5, 5)[:3] == pixel(gradient.resize((400, 300)), 5, 5)
    for _ in range(21):
        assert doc.redo()
    assert pixel(doc.active_layer.image, 95, 5) == (255, 0, 0, 255)
    # coalesced states (replace_current) keep the chain intact
    doc.invert_selection()
    doc.set_selection(Selection.rectangle(doc.size, (0, 100, 50, 150)))
    doc.fill("#00ff00")
    doc.select_all()
    doc.invert_selection()  # replaces the current state
    assert doc.undo() and doc.undo() and pixel(doc.active_layer.image, 25, 125) == (0, 255, 0, 255)
    assert doc.undo() and pixel(doc.active_layer.image, 25, 125) != (0, 255, 0, 255)  # back before the fill
    # masks are patched as well; snapshots (bookmarks) stay full
    doc.deselect()
    doc.add_mask()
    doc.set_selection(Selection.rectangle(doc.size, (0, 0, 20, 20)))
    doc.fill("#000000")
    mask_patches = [p for st in doc.history.states for key, p in st.snapshot.patches.items() if key[1] and p.box is not None]
    assert mask_patches and mask_patches[0].box == (0, 0, 20, 20)
    doc.take_snapshot("bookmark")
    assert not doc.snapshots[0][1].patches
