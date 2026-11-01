import numpy as np
import pytest

from pixelforge.core.blend import (
    BLEND_MODE_NAMES,
    BLEND_MODES,
    blend,
    composite_over,
    composite_premultiplied,
    unpremultiply,
)


def arr(*rgb: float) -> np.ndarray:
    return np.array([[rgb]], np.float32)


@pytest.mark.parametrize("mode", BLEND_MODE_NAMES)
def test_every_mode_returns_values_in_range(mode: str) -> None:
    rng = np.random.default_rng(1)
    cb = rng.random((8, 8, 3), dtype=np.float32)
    cs = rng.random((8, 8, 3), dtype=np.float32)
    out = blend(mode, cb, cs)
    assert out.shape == cb.shape
    assert np.isfinite(out).all()
    assert out.min() >= 0 and out.max() <= 1


@pytest.mark.parametrize(
    "mode", ["multiply", "screen", "overlay", "darken", "lighten", "difference", "exclusion", "hard light", "soft light"]
)
def test_white_and_black_identities(mode: str) -> None:
    cb = np.array([[[0.3, 0.6, 0.9]]], np.float32)
    black, white = arr(0, 0, 0), arr(1, 1, 1)
    if mode == "multiply":
        assert np.allclose(blend(mode, cb, white), cb)
        assert np.allclose(blend(mode, cb, black), 0)
    if mode == "screen":
        assert np.allclose(blend(mode, cb, black), cb)
        assert np.allclose(blend(mode, cb, white), 1)
    if mode == "darken":
        assert np.allclose(blend(mode, cb, white), cb)
    if mode == "lighten":
        assert np.allclose(blend(mode, cb, black), cb)
    if mode == "difference":
        assert np.allclose(blend(mode, cb, black), cb)
        assert np.allclose(blend(mode, cb, white), 1 - cb)
    if mode == "exclusion":
        assert np.allclose(blend(mode, cb, black), cb)
    if mode in ("overlay", "hard light", "soft light"):
        # 50% grey source leaves the backdrop (nearly) unchanged
        assert np.allclose(blend(mode, cb, arr(0.5, 0.5, 0.5)), cb, atol=0.02)


def test_specific_values() -> None:
    assert blend("multiply", arr(0.5, 0.5, 0.5), arr(0.5, 0.5, 0.5))[0, 0, 0] == pytest.approx(0.25)
    assert blend("screen", arr(0.5, 0.5, 0.5), arr(0.5, 0.5, 0.5))[0, 0, 0] == pytest.approx(0.75)
    assert blend("linear dodge", arr(0.7, 0.7, 0.7), arr(0.7, 0.7, 0.7))[0, 0, 0] == pytest.approx(1.0)
    assert blend("linear burn", arr(0.2, 0.2, 0.2), arr(0.2, 0.2, 0.2))[0, 0, 0] == pytest.approx(0.0)
    assert blend("subtract", arr(0.8, 0.8, 0.8), arr(0.3, 0.3, 0.3))[0, 0, 0] == pytest.approx(0.5)
    assert blend("divide", arr(0.25, 0.25, 0.25), arr(0.5, 0.5, 0.5))[0, 0, 0] == pytest.approx(0.5)
    assert blend("color dodge", arr(0.5, 0.5, 0.5), arr(0.5, 0.5, 0.5))[0, 0, 0] == pytest.approx(1.0)
    assert blend("color burn", arr(0.5, 0.5, 0.5), arr(0.5, 0.5, 0.5))[0, 0, 0] == pytest.approx(0.0)
    assert blend("normal", arr(0.1, 0.2, 0.3), arr(0.9, 0.8, 0.7))[0, 0].tolist() == pytest.approx([0.9, 0.8, 0.7])


def test_non_separable_modes() -> None:
    red = arr(1.0, 0.0, 0.0)
    gray = arr(0.5, 0.5, 0.5)
    # luminosity: keep backdrop colour, take source luminance -> brighter/darker red
    out = blend("luminosity", red, gray)[0, 0]
    assert out[0] > out[1] and out[0] > out[2]
    # color: take source hue/sat, keep backdrop luminance -> grey backdrop becomes reddish with same lum
    out = blend("color", gray, red)[0, 0]
    assert out[0] > out[1]
    lum = 0.3 * out[0] + 0.59 * out[1] + 0.11 * out[2]
    assert lum == pytest.approx(0.5, abs=0.02)
    # saturation of a grey source removes colour
    out = blend("saturation", red, gray)[0, 0]
    assert abs(out[0] - out[1]) < 1e-5 and abs(out[1] - out[2]) < 1e-5
    # hue of grey backdrop stays grey (sat 0)
    out = blend("hue", gray, red)[0, 0]
    assert abs(out[0] - out[1]) < 1e-5


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        blend("nope", arr(0, 0, 0), arr(1, 1, 1))
    assert "normal" in BLEND_MODES


def test_composite_over_alpha_math() -> None:
    dst_rgb, dst_a = arr(0.2, 0.4, 0.6), np.array([[[1.0]]], np.float32)
    src_rgb = arr(1.0, 0.0, 0.0)
    # opaque source replaces
    rgb, a = composite_over(dst_rgb, dst_a, src_rgb, np.array([[[1.0]]], np.float32))
    assert np.allclose(rgb, src_rgb) and np.allclose(a, 1)
    # fully transparent source leaves backdrop
    rgb, a = composite_over(dst_rgb, dst_a, src_rgb, np.array([[[0.0]]], np.float32))
    assert np.allclose(rgb, dst_rgb) and np.allclose(a, 1)
    # half transparent: linear mix
    rgb, a = composite_over(dst_rgb, dst_a, src_rgb, np.array([[[0.5]]], np.float32))
    assert np.allclose(rgb, (src_rgb + dst_rgb) / 2)
    # over transparent backdrop: source colour, source alpha
    rgb, a = composite_over(dst_rgb, np.array([[[0.0]]], np.float32), src_rgb, np.array([[[0.5]]], np.float32), "multiply")
    assert np.allclose(rgb, src_rgb) and np.allclose(a, 0.5)


def test_blend_mode_with_partial_backdrop_alpha() -> None:
    # Backdrop 50% opaque white, multiply with black source: where backdrop is missing the source shows unblended.
    dst_rgb, dst_a = arr(1, 1, 1), np.array([[[0.5]]], np.float32)
    rgb, a = composite_over(dst_rgb, dst_a, arr(0, 0, 0), np.array([[[1.0]]], np.float32), "multiply")
    assert np.allclose(a, 1)
    assert np.allclose(rgb, 0)  # black * white = black, black over nothing = black


def test_premultiplied_path_matches_straight_path() -> None:
    rng = np.random.default_rng(3)
    dst_rgb = rng.random((4, 4, 3), dtype=np.float32)
    dst_a = rng.random((4, 4, 1), dtype=np.float32)
    src_rgb = rng.random((4, 4, 3), dtype=np.float32)
    src_a = rng.random((4, 4, 1), dtype=np.float32)
    for mode in ("normal", "screen", "soft light"):
        rgb1, a1 = composite_over(dst_rgb, dst_a, src_rgb, src_a, mode)
        p, a2 = composite_premultiplied(dst_rgb * dst_a, dst_a, src_rgb, src_a, mode)
        assert np.allclose(a1, a2, atol=1e-5)
        assert np.allclose(rgb1, unpremultiply(p, a2), atol=1e-4)


def test_composite_premultiplied_empty_backdrop() -> None:
    src_rgb, src_a = arr(0.2, 0.3, 0.4), np.array([[[0.5]]], np.float32)
    p, a = composite_premultiplied(None, None, src_rgb, src_a, "overlay")
    assert np.allclose(p, src_rgb * 0.5) and np.allclose(a, 0.5)
    assert np.allclose(unpremultiply(p, a), src_rgb)
    assert unpremultiply(np.zeros((1, 1, 3), np.float32), np.zeros((1, 1, 1), np.float32)).tolist() == [[[0, 0, 0]]]
