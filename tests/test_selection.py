import numpy as np
import pytest
from PIL import Image

from pixelforge.core.selection import COMBINE_MODES, Selection, color_match_mask, flood_mask

SIZE = (40, 30)


def test_rectangle_and_coverage() -> None:
    sel = Selection.rectangle(SIZE, (10, 5, 20, 15))
    assert sel.bbox == (10, 5, 20, 15)
    assert sel.coverage() == pytest.approx(100 / 1200)
    assert sel.contains(15, 10) and not sel.contains(5, 5) and not sel.contains(-1, 0)
    assert not sel.is_empty and not sel.is_full
    # reversed corners are normalised
    assert Selection.rectangle(SIZE, (20, 15, 10, 5)).bbox == (10, 5, 20, 15)
    assert Selection.rectangle(SIZE, (10, 10, 10, 20)).is_empty


def test_all_and_empty() -> None:
    assert Selection.all(SIZE).is_full
    assert Selection.empty(SIZE).is_empty
    assert Selection.all(SIZE).invert().is_empty


def test_ellipse_is_antialiased_and_inside_box() -> None:
    sel = Selection.ellipse(SIZE, (5, 5, 35, 25))
    assert sel.bbox == (5, 5, 35, 25)
    arr = np.asarray(sel.mask)
    assert arr[15, 20] == 255  # centre
    assert arr[5, 5] == 0  # corner of box outside the ellipse
    assert len(np.unique(arr)) > 2  # partial coverage along the edge
    hard = Selection.ellipse(SIZE, (5, 5, 35, 25), antialias=False)
    assert set(np.unique(np.asarray(hard.mask))) <= {0, 255}
    assert Selection.ellipse(SIZE, (5, 5, 5, 5)).is_empty


def test_polygon() -> None:
    tri = Selection.polygon(SIZE, [(0, 0), (39, 0), (0, 29)])
    assert tri.contains(5, 5) and not tri.contains(38, 28)
    assert Selection.polygon(SIZE, [(0, 0), (1, 1)]).is_empty


@pytest.mark.parametrize("mode", COMBINE_MODES)
def test_combine_modes(mode: str) -> None:
    a = Selection.rectangle(SIZE, (0, 0, 20, 30))
    b = Selection.rectangle(SIZE, (10, 0, 30, 30))
    out = a.combine(b, mode)
    expected = {"replace": (10, 0, 30, 30), "add": (0, 0, 30, 30), "subtract": (0, 0, 10, 30), "intersect": (10, 0, 20, 30)}[mode]
    assert out.bbox == expected


def test_combine_errors() -> None:
    a = Selection.rectangle(SIZE, (0, 0, 5, 5))
    with pytest.raises(ValueError):
        a.combine(Selection.all((5, 5)))
    with pytest.raises(ValueError):
        a.combine(a, "xor")


def test_feather_grow_shrink_border() -> None:
    sel = Selection.rectangle(SIZE, (10, 10, 20, 20))
    feathered = sel.feather(2)
    arr = np.asarray(feathered.mask)
    assert 0 < arr[9, 15] < 255  # soft edge outside the hard rectangle
    assert sel.feather(0) is sel
    assert sel.grow(2).bbox == (8, 8, 22, 22)
    assert sel.shrink(2).bbox == (12, 12, 18, 18)
    border = sel.border(1)
    assert border.contains(9, 15) and border.contains(10, 15) and not border.contains(15, 15)
    assert sel.grow(0) is sel and sel.shrink(0) is sel


def test_transformed_and_as_array() -> None:
    sel = Selection.rectangle(SIZE, (0, 0, 10, 10))
    flipped = sel.transformed(lambda m: m.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    assert flipped.bbox == (30, 0, 40, 10)
    arr = sel.as_array()
    assert arr.shape == (30, 40, 1) and arr.max() == 1.0 and arr.dtype == np.float32


def test_from_alpha(shapes: Image.Image) -> None:
    sel = Selection.from_alpha(shapes)
    assert sel.contains(10, 10) and sel.contains(44, 25) and not sel.contains(0, 0)


def test_magic_wand_contiguous_vs_global(shapes: Image.Image) -> None:
    # two disconnected transparent regions? background is connected; test with the red square vs blue circle
    img = shapes.copy()
    img.paste((255, 0, 0, 255), (40, 42, 60, 47))  # second red patch, disconnected from the square
    contiguous = Selection.magic_wand(img, (10, 10), tolerance=10, contiguous=True)
    assert contiguous.contains(10, 10) and not contiguous.contains(45, 44)
    everywhere = Selection.magic_wand(img, (10, 10), tolerance=10, contiguous=False)
    assert everywhere.contains(10, 10) and everywhere.contains(45, 44)
    assert not everywhere.contains(44, 25)  # blue circle untouched


def test_magic_wand_tolerance(gradient: Image.Image) -> None:
    tight = Selection.magic_wand(gradient, (32, 24), tolerance=4)
    loose = Selection.magic_wand(gradient, (32, 24), tolerance=60)
    assert tight.coverage() < loose.coverage()
    assert Selection.magic_wand(gradient, (32, 24), tolerance=255).is_full


def test_flood_mask_low_level() -> None:
    arr = np.zeros((5, 6, 4), np.uint8)
    arr[2, :, 0] = 255  # a red line splits the image in two
    top = flood_mask(arr, (0, 0), tolerance=0)
    assert top[0:2].all() and not top[2:].any()
    match = color_match_mask(arr, (0, 2), tolerance=0)
    assert match[2].all() and not match[0].any()
    with pytest.raises(ValueError):
        color_match_mask(arr, (99, 99), 0)


def test_outline_and_marching_ants() -> None:
    sel = Selection.rectangle(SIZE, (10, 10, 20, 20))
    edge = sel.outline()
    arr = np.asarray(edge)
    assert arr[10, 15] == 255 and arr[15, 10] == 255  # border pixels
    assert arr[15, 15] == 0  # interior
    assert arr.sum() // 255 == 36  # 10x10 square perimeter
    half = sel.outline(0.5)
    assert half.size == (20, 15)
    double = sel.outline(2.0)
    assert double.size == (80, 60)
    ants = Selection.marching_ants(edge, phase=0)
    assert ants.mode == "RGBA" and ants.size == edge.size
    a = np.asarray(ants)
    assert (a[..., 3] > 0).sum() == 36
    colors = {tuple(px) for px in a[a[..., 3] > 0][:, :3]}
    assert colors == {(0, 0, 0), (255, 255, 255)}
    shifted = np.asarray(Selection.marching_ants(edge, phase=4))
    assert not np.array_equal(shifted, a)


def test_selection_requires_L_mode_conversion() -> None:
    sel = Selection(Image.new("RGB", SIZE, (255, 255, 255)))
    assert sel.mask.mode == "L" and sel.is_full
