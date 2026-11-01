"""Property-based tests (Hypothesis) for the algebraic parts of the engine.

Blend modes, selection set operations, curves, transforms and history patches
have laws that must hold for *any* input - exactly what random inputs probe
better than hand-picked pixels.
"""

import math

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from PIL import Image

from pixelforge.core import curves
from pixelforge.core.blend import BLEND_MODE_NAMES, blend, composite_over
from pixelforge.core.document import Document
from pixelforge.core.painting import Transform, transform_image
from pixelforge.core.selection import Selection

unit_pixels = hnp.arrays(np.float32, (3, 4, 3), elements=st.floats(0.0, 1.0, width=32))
masks = hnp.arrays(np.uint8, (12, 10), elements=st.integers(0, 255))


def as_mask(arr: np.ndarray) -> Selection:
    return Selection(Image.fromarray(arr, "L"))


def mask_array(selection: Selection) -> np.ndarray:
    return np.asarray(selection.mask)


# -------------------------------------------------------------- blend modes --
@given(unit_pixels, unit_pixels, st.sampled_from(BLEND_MODE_NAMES))
@settings(max_examples=150, deadline=None)
def test_blend_modes_stay_in_range(backdrop: np.ndarray, source: np.ndarray, mode: str) -> None:
    out = blend(mode, backdrop, source)
    assert out.shape == backdrop.shape and np.all(np.isfinite(out))
    assert float(out.min()) >= -1e-5 and float(out.max()) <= 1 + 1e-5


@given(unit_pixels, unit_pixels)
@settings(max_examples=60, deadline=None)
def test_blend_identities(backdrop: np.ndarray, source: np.ndarray) -> None:
    white, black = np.ones_like(source), np.zeros_like(source)
    assert np.allclose(blend("normal", backdrop, source), source)
    assert np.allclose(blend("multiply", backdrop, white), backdrop, atol=1e-6)
    assert np.allclose(blend("screen", backdrop, black), backdrop, atol=1e-6)
    assert np.allclose(blend("darken", backdrop, source), np.minimum(backdrop, source))
    assert np.allclose(blend("lighten", backdrop, source), np.maximum(backdrop, source))
    assert np.allclose(blend("difference", backdrop, source), blend("difference", source, backdrop))
    assert np.allclose(blend("multiply", backdrop, source), blend("multiply", source, backdrop))


@given(unit_pixels, unit_pixels, hnp.arrays(np.float32, (3, 4, 1), elements=st.floats(0.0, 1.0, width=32)))
@settings(max_examples=60, deadline=None)
def test_source_over_alpha_laws(dst_rgb: np.ndarray, src_rgb: np.ndarray, src_a: np.ndarray) -> None:
    dst_a = np.ones_like(src_a)
    out_rgb, out_a = composite_over(dst_rgb, dst_a, src_rgb, src_a, "normal")
    assert np.allclose(out_a, 1.0)  # over an opaque backdrop the result stays opaque
    expected = src_rgb * src_a + dst_rgb * (1 - src_a)
    assert np.allclose(out_rgb, expected, atol=1e-5)
    # a fully transparent source changes nothing; a fully opaque one replaces
    same_rgb, _ = composite_over(dst_rgb, dst_a, src_rgb, np.zeros_like(src_a), "normal")
    assert np.allclose(same_rgb, dst_rgb, atol=1e-6)
    replaced_rgb, _ = composite_over(dst_rgb, dst_a, src_rgb, np.ones_like(src_a), "normal")
    assert np.allclose(replaced_rgb, src_rgb, atol=1e-6)


# -------------------------------------------------------------- selections --
@given(masks, masks)
@settings(max_examples=100, deadline=None)
def test_selection_algebra(a_arr: np.ndarray, b_arr: np.ndarray) -> None:
    a, b = as_mask(a_arr), as_mask(b_arr)
    union, inter, diff = a.combine(b, "add"), a.combine(b, "intersect"), a.combine(b, "subtract")
    ua, ia, da = mask_array(union), mask_array(inter), mask_array(diff)
    assert np.all(ua >= a_arr) and np.all(ua >= b_arr)  # A ∪ B ⊇ A, B
    assert np.all(ia <= a_arr) and np.all(ia <= b_arr)  # A ∩ B ⊆ A, B
    assert np.all(da <= a_arr)  # A − B ⊆ A
    assert np.all(mask_array(diff.combine(b, "intersect")) <= np.minimum(255 - b_arr, b_arr))  # (A−B) ∩ B is (soft-)empty
    assert np.array_equal(mask_array(a.combine(b, "add")), mask_array(b.combine(a, "add")))  # commutative
    assert np.array_equal(mask_array(a.combine(b, "intersect")), mask_array(b.combine(a, "intersect")))
    assert np.array_equal(mask_array(a.invert().invert()), a_arr)  # involution
    assert np.array_equal(mask_array(a.combine(a, "add")), a_arr) and np.array_equal(mask_array(a.combine(a, "intersect")), a_arr)
    assert a.combine(b, "replace") is b
    if a.bbox is not None:
        x0, y0, x1, y1 = a.bbox
        assert 0 <= x0 < x1 <= 10 and 0 <= y0 < y1 <= 12
        assert not np.any(a_arr[:y0]) and not np.any(a_arr[y1:]) and not np.any(a_arr[:, :x0]) and not np.any(a_arr[:, x1:])
    else:
        assert a.is_empty


# ------------------------------------------------------------------ curves --
@given(
    st.lists(st.tuples(st.floats(0, 255), st.floats(0, 255)), min_size=2, max_size=8).map(
        lambda pts: sorted({round(x): y for x, y in pts}.items())
    )
)
@settings(max_examples=150, deadline=None)
def test_curve_lut_is_bounded_and_passes_through_points(points: list[tuple[float, float]]) -> None:
    lut = curves.curve_lut(points)
    assert len(lut) == 256 and all(0 <= v <= 255 for v in lut)
    for x, y in points:
        assert lut[int(x)] == int(math.floor(y + 0.5 + 1e-6))  # the table rounds half up


@given(st.lists(st.floats(0, 255), min_size=2, max_size=8))
@settings(max_examples=150, deadline=None)
def test_monotone_points_give_monotone_curve(values: list[float]) -> None:
    ys = sorted(values)
    xs = np.linspace(0, 255, len(ys))
    lut = curves.curve_lut(list(zip(xs.tolist(), ys, strict=True)))
    assert all(lut[i] <= lut[i + 1] for i in range(255))  # no overshoot: monotone data → monotone curve


# --------------------------------------------------------------- transform --
@given(
    st.floats(0.05, 8),
    st.floats(-8, -0.05) | st.floats(0.05, 8),
    st.floats(-360, 360),
    st.floats(-500, 500),
    st.floats(-500, 500),
    st.floats(0, 300),
    st.floats(0, 300),
)
@settings(max_examples=150, deadline=None)
def test_transform_inverse_roundtrip(sx: float, sy: float, angle: float, dx: float, dy: float, x: float, y: float) -> None:
    t = Transform((10.0, 20.0, 110.0, 220.0), sx, sy, angle, dx, dy)
    fx, fy = t.apply(x, y)
    bx, by = t.unapply(fx, fy)
    assert abs(bx - x) < 1e-6 * max(1, abs(x)) + 1e-4 and abs(by - y) < 1e-6 * max(1, abs(y)) + 1e-4
    cx, cy = t.apply(*t.center)
    assert abs(cx - (t.center[0] + dx)) < 1e-6 and abs(cy - (t.center[1] + dy)) < 1e-6  # the pivot only translates


@given(st.integers(-30, 30), st.integers(-30, 30))
@settings(max_examples=40, deadline=None)
def test_integer_translation_is_exact(dx: int, dy: int) -> None:
    img = Image.new("RGBA", (64, 48), (0, 0, 0, 0))
    img.paste((200, 30, 90, 255), (20, 15, 40, 30))
    moved = transform_image(img, Transform((20.0, 15.0, 40.0, 30.0), dx=dx, dy=dy))
    x, y = 30 + dx, 22 + dy
    if 0 <= x < 64 and 0 <= y < 48:
        assert moved.getpixel((x, y)) == (200, 30, 90, 255)
    if 0 <= 30 - dx < 64 and 0 <= 22 - dy < 48 and (abs(dx) > 20 or abs(dy) > 15):
        assert moved.getpixel((30 - dx, 22 - dy))[3] == 0  # type: ignore[index]


# ----------------------------------------------------------------- history --
@given(st.lists(st.tuples(st.integers(0, 30), st.integers(0, 20), st.integers(1, 12), st.integers(1, 12)), min_size=1, max_size=6))
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_history_patches_restore_exact_pixels(rects: list[tuple[int, int, int, int]]) -> None:
    doc = Document.new(40, 30, "#336699")
    states = [np.asarray(doc.active_layer.image).copy()]
    for i, (x, y, w, h) in enumerate(rects):
        doc.set_selection(Selection.rectangle(doc.size, (x, y, x + w, y + h)))
        doc.fill(f"#{(i * 40) % 256:02x}{(i * 90) % 256:02x}{(i * 130) % 256:02x}")
        states.append(np.asarray(doc.active_layer.image).copy())
    doc.deselect()
    # walk back through every state and forward again: the patches rebuild each image exactly
    for expected in reversed(states[:-1]):
        while not np.array_equal(np.asarray(doc.active_layer.image), expected):
            assert doc.undo()
    while doc.redo():
        pass
    assert np.array_equal(np.asarray(doc.active_layer.image), states[-1])
