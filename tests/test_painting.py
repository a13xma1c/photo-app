import numpy as np
import pytest
from PIL import Image

from pixelforge.core import painting
from pixelforge.core.painting import Stroke, brush_stamp
from pixelforge.gui.theme import font_size
from tests.helpers import gray, pixel

SIZE = (60, 40)


def blank(color: tuple[int, int, int, int] = (0, 0, 0, 0)) -> Image.Image:
    return Image.new("RGBA", SIZE, color)


# ------------------------------------------------------------------ brush --
def test_brush_stamp_shape_and_hardness() -> None:
    hard = brush_stamp(21, 1.0)
    soft = brush_stamp(21, 0.0)
    assert hard.shape == soft.shape == (21, 21)
    assert hard[10, 10] == 1.0 and soft[10, 10] == pytest.approx(1.0, abs=0.05)
    assert hard[0, 0] == 0 and soft[0, 0] == 0  # corners outside the disc
    assert hard.sum() > soft.sum()  # soft brush carries less coverage
    assert hard[10, 2] == 1.0  # well inside -> full
    assert 0 < soft[10, 3] < 1  # gradient region
    assert brush_stamp(1).shape == (1, 1)
    assert brush_stamp(0).shape == (1, 1)


def test_stroke_accumulates_along_path_with_dirty_bbox() -> None:
    stroke = Stroke(SIZE, diameter=6, hardness=1.0)
    stroke.add_point(10, 20)
    assert stroke.dirty is not None
    box = stroke.take_dirty()
    assert box == (7, 17, 13, 23)
    assert stroke.dirty is None
    stroke.add_point(40, 20)
    assert stroke.coverage[20, 25] > 0.9  # middle of the segment is covered (spacing)
    assert stroke.coverage[5, 25] == 0
    x0, y0, x1, y1 = stroke.take_dirty()
    assert x0 >= 7 and x1 <= 44 and y0 >= 17 and y1 <= 23
    # stamping the same point again does not add coverage beyond 1
    stroke.add_point(40, 20)
    assert stroke.coverage.max() <= 1.0
    mask = stroke.mask_image()
    assert mask.mode == "L" and mask.size == SIZE


def test_stroke_outside_canvas_is_clipped() -> None:
    stroke = Stroke(SIZE, diameter=10)
    stroke.add_points([(-20, -20), (-5, -5)])
    assert stroke.coverage.sum() == 0
    stroke.add_point(0, 0)
    assert stroke.coverage[0, 0] > 0


def test_paint_color_opacity_and_selection() -> None:
    stroke = Stroke(SIZE, diameter=8, hardness=1.0)
    stroke.add_point(30, 20)
    layer = blank()
    out = painting.paint(layer, stroke.coverage, "#ff0000")
    assert out is not layer and out.size == SIZE
    assert pixel(out, 30, 20) == (255, 0, 0, 255)
    assert pixel(out, 0, 0) == (0, 0, 0, 0)
    half = painting.paint(layer, stroke.coverage, (0, 0, 255, 255), opacity=0.5)
    assert pixel(half, 30, 20)[3] in (127, 128)
    # painting over an opaque layer mixes colours by coverage
    on_white = painting.paint(blank((255, 255, 255, 255)), stroke.coverage, "#000000", opacity=0.5)
    assert pixel(on_white, 30, 20)[0] in (127, 128)
    # selection restricts painting
    sel = Image.new("L", SIZE, 0)
    sel.paste(255, (0, 0, 30, 40))  # left half only
    out = painting.paint(layer, stroke.coverage, "#00ff00", selection=sel)
    assert pixel(out, 28, 20)[3] == 255 and pixel(out, 32, 20)[3] == 0
    # mask image input works too
    out2 = painting.paint(layer, stroke.mask_image(), "#ff0000")
    assert pixel(out2, 30, 20) == (255, 0, 0, 255)


def test_paint_with_empty_coverage_returns_copy() -> None:
    layer = blank((1, 2, 3, 4))
    out = painting.paint(layer, np.zeros((SIZE[1], SIZE[0]), np.float32), "#fff")
    assert out is not layer and out.tobytes() == layer.tobytes()
    with pytest.raises(ValueError):
        painting.paint(layer, np.zeros((3, 3), np.float32), "#fff")
    with pytest.raises(ValueError):
        painting.paint(layer, np.zeros((SIZE[1], SIZE[0]), np.float32), "#fff", selection=Image.new("L", (2, 2)))


def test_erase() -> None:
    layer = blank((10, 20, 30, 255))
    stroke = Stroke(SIZE, diameter=10, hardness=1.0)
    stroke.add_point(20, 20)
    out = painting.erase(layer, stroke.coverage)
    assert pixel(out, 20, 20)[3] == 0
    assert pixel(out, 50, 20)[3] == 255
    partial = painting.erase(layer, stroke.coverage, opacity=0.5)
    assert pixel(partial, 20, 20)[3] in (127, 128)
    sel = Image.new("L", SIZE, 0)
    restricted = painting.erase(layer, stroke.coverage, selection=sel)
    assert pixel(restricted, 20, 20)[3] == 255


# ------------------------------------------------------------------- fill --
def test_fill_and_clear() -> None:
    layer = blank((0, 0, 0, 255))
    assert pixel(painting.fill(layer, "#00ff00"), 5, 5) == (0, 255, 0, 255)
    mask = Image.new("L", SIZE, 0)
    mask.paste(255, (0, 0, 10, 10))
    out = painting.fill(layer, "#00ff00", mask)
    assert pixel(out, 5, 5) == (0, 255, 0, 255) and pixel(out, 20, 20) == (0, 0, 0, 255)
    half = pixel(painting.fill(layer, (255, 255, 255, 255), opacity=0.5), 5, 5)
    assert 120 <= half[0] <= 135
    cleared = painting.clear(layer, mask)
    assert pixel(cleared, 5, 5)[3] == 0 and pixel(cleared, 20, 20)[3] == 255
    assert pixel(painting.clear(layer), 20, 20)[3] == 0


# --------------------------------------------------------------- gradient --
@pytest.mark.parametrize("kind", painting.GRADIENT_TYPES)
def test_gradient_types(kind: str) -> None:
    grad = painting.gradient_image(SIZE, (0, 20), (59, 20), "#000000", "#ffffff", kind)
    assert grad.size == SIZE and grad.mode == "RGBA"
    left, right = pixel(grad, 0, 20)[0], pixel(grad, 59, 20)[0]
    if kind == "linear":
        assert left < 5 and right > 250
        assert 100 < pixel(grad, 30, 20)[0] < 155
    elif kind == "radial":
        assert left < 5 and right > 250
        assert pixel(grad, 0, 0)[0] > pixel(grad, 0, 20)[0]  # distance from the start grows
    else:  # reflected
        assert left < 5 and right > 250


def test_gradient_degenerate_and_apply_with_selection() -> None:
    grad = painting.gradient_image(SIZE, (5, 5), (5, 5), "#ff0000", "#0000ff")
    assert pixel(grad, 0, 0) == (0, 0, 255, 255)  # zero length -> end colour
    with pytest.raises(ValueError):
        painting.gradient_image(SIZE, (0, 0), (1, 1), "#000", "#fff", kind="spiral")
    layer = blank((0, 255, 0, 255))
    sel = Image.new("L", SIZE, 0)
    sel.paste(255, (0, 0, 30, 40))
    out = painting.apply_gradient(layer, (0, 0), (59, 0), "#000000", "#ffffff", selection=sel)
    assert pixel(out, 45, 20) == (0, 255, 0, 255)  # untouched outside selection
    assert pixel(out, 0, 20)[0] < 5  # gradient inside
    assert pixel(painting.apply_gradient(layer, (0, 0), (59, 0), "#000", "#fff"), 45, 20)[1] > 100


# ------------------------------------------------------------------- text --
def test_render_text_and_bbox() -> None:
    img = painting.render_text((200, 80), (10, 10), "Hello", "#ff0000", None, 32)
    assert img.size == (200, 80)
    bbox = img.getbbox()
    assert bbox is not None and bbox[0] >= 10 and bbox[1] >= 10
    arr = np.asarray(img)
    colors = {tuple(px) for px in arr[arr[..., 3] > 200][:, :3]}
    assert colors == {(255, 0, 0)}
    assert painting.render_text((50, 50), (0, 0), "", "#fff").getbbox() is None
    tb = painting.text_bbox("Hello", None, 32)
    assert tb[2] > tb[0] and tb[3] > tb[1]
    multi = painting.render_text((200, 120), (0, 0), "a\nb", "#fff", None, 24)
    single_bbox = painting.render_text((200, 120), (0, 0), "a", "#fff", None, 24).getbbox()
    multi_bbox = multi.getbbox()
    assert single_bbox is not None and multi_bbox is not None and multi_bbox[3] > single_bbox[3]
    outlined = painting.render_text((200, 80), (10, 10), "Hi", "#ff0000", None, 32, stroke_width=2, stroke_color="#0000ff")
    outlined_arr = np.asarray(outlined)
    assert (outlined_arr[outlined_arr[..., 3] > 200][:, 2] > 200).any()


def test_fonts() -> None:
    fonts = painting.system_fonts()
    assert isinstance(fonts, dict)
    font = painting.load_font(None, 20)
    assert font is not None
    assert painting.load_font("definitely-not-a-font", 20) is not None
    if fonts:
        name = next(iter(fonts))
        assert font_size(painting.load_font(name, 12)) == 12


# ----------------------------------------------------------------- shapes --
@pytest.mark.parametrize("kind", painting.SHAPE_TYPES)
def test_draw_shapes(kind: str) -> None:
    layer = blank()
    out = painting.draw_shape(layer, kind, (10, 10, 50, 30), fill_color="#ff0000", outline_color="#0000ff", width=2)
    assert out.size == SIZE and out.getbbox() is not None
    if kind != "line":
        assert pixel(out, 30, 20)[:3] == (255, 0, 0)  # filled centre
        assert pixel(out, 10, 20)[2] > 150  # blue outline on the edge
    else:
        assert pixel(out, 30, 20)[3] > 0
    assert pixel(out, 0, 0)[3] == 0


def test_draw_shape_options() -> None:
    layer = blank()
    with pytest.raises(ValueError):
        painting.draw_shape(layer, "star", (0, 0, 10, 10))
    # no fill, no outline -> default thin black outline
    out = painting.draw_shape(layer, "rectangle", (10, 10, 50, 30))
    assert pixel(out, 30, 20)[3] == 0 and pixel(out, 10, 20)[3] > 0
    # selection clips the shape
    sel = Image.new("L", SIZE, 0)
    sel.paste(255, (0, 0, 30, 40))
    out = painting.draw_shape(layer, "ellipse", (10, 10, 50, 30), fill_color="#fff", selection=sel)
    assert pixel(out, 20, 20)[3] > 0 and pixel(out, 40, 20)[3] == 0
    aliased = painting.draw_shape(layer, "rectangle", (10, 10, 50, 30), fill_color="#fff", antialias=False)
    assert set(np.unique(np.asarray(aliased)[..., 3])) <= {0, 255}


def test_offset_image() -> None:
    layer = blank()
    layer.paste((255, 0, 0, 255), (0, 0, 10, 10))
    moved = painting.offset_image(layer, 5, 3)
    assert pixel(moved, 5, 3) == (255, 0, 0, 255)
    assert pixel(moved, 0, 0)[3] == 0
    mask = Image.new("L", SIZE, 0)
    mask.paste(255, (0, 0, 10, 10))
    moved_mask = painting.offset_image(mask, -5, 0)
    assert moved_mask.mode == "L" and gray(moved_mask, 4, 0) == 255 and gray(moved_mask, 5, 0) == 0


# ------------------------------------------------------------ clone / heal --
def _blemished() -> tuple[Image.Image, np.ndarray]:
    """Checkerboard texture with a red blemish and the coverage of a brush over the blemish."""
    tex = Image.new("RGBA", (60, 40), (100, 150, 200, 255))
    for y in range(40):
        for x in range(60):
            if (x + y) % 2:
                tex.putpixel((x, y), (120, 170, 220, 255))
    tex.paste((255, 0, 0, 255), (30, 10, 40, 20))
    cov = np.zeros((40, 60), np.float32)
    cov[10:20, 30:40] = 1
    return tex, cov


def test_clone_stamp_copies_shifted_source() -> None:
    tex, cov = _blemished()
    out = painting.clone(tex, cov, tex, (-20, 0))
    assert pixel(out, 35, 15) == pixel(tex, 15, 15) and pixel(out, 36, 15) == pixel(tex, 16, 15)
    assert pixel(out, 5, 5) == pixel(tex, 5, 5)  # untouched outside the coverage
    half = painting.clone(tex, cov, tex, (-20, 0), opacity=0.5)
    assert 150 < pixel(half, 35, 15)[0] < 255
    # a selection limits the stroke; an empty coverage is a copy
    sel = Image.new("L", tex.size, 0)
    sel.paste(255, (0, 0, 35, 40))
    limited = painting.clone(tex, cov, tex, (-20, 0), selection=sel)
    assert pixel(limited, 32, 15) != (255, 0, 0, 255) and pixel(limited, 37, 15) == (255, 0, 0, 255)
    assert painting.clone(tex, np.zeros((40, 60), np.float32), tex).tobytes() == tex.tobytes()
    with pytest.raises(ValueError):
        painting.clone(tex, cov, None)
    with pytest.raises(ValueError):
        painting.clone(tex, cov, tex.resize((10, 10)))
    with pytest.raises(ValueError):
        painting.clone(tex, np.zeros((3, 3), np.float32), tex)


def test_healing_keeps_surrounding_colour_and_source_texture() -> None:
    tex, cov = _blemished()
    healed = painting.clone(tex, cov, tex, (-20, 0), heal=True)
    r, g, b, a = pixel(healed, 35, 15)
    assert r < 140 and g > 130 and b > 180 and a == 255  # the red is gone, the blue-grey colour is back
    assert pixel(healed, 35, 15) != pixel(healed, 36, 15)  # checker texture reproduced
    spot = painting.clone(tex, cov, None, heal=True)
    r, g, b, a = pixel(spot, 35, 15)
    assert r < 140 and g > 130 and b > 180 and a == 255
    assert abs(pixel(spot, 35, 15)[0] - pixel(spot, 36, 15)[0]) <= 3  # smooth fill, no texture
    # transparent layers: healing does not invent opaque pixels outside the stroke
    clear = Image.new("RGBA", (60, 40), (0, 0, 0, 0))
    clear.paste((200, 100, 50, 255), (25, 5, 45, 25))
    spot_clear = painting.clone(clear, cov, None, heal=True)
    assert pixel(spot_clear, 5, 5)[3] == 0 and pixel(spot_clear, 35, 15)[3] == 255
