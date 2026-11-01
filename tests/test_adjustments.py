import numpy as np
import pytest
from PIL import Image

from pixelforge.core.adjustments import (
    GROUPS,
    SLIDERS,
    apply_adjustments,
    default_adjustments,
    is_neutral,
    normalize_adjustments,
)
from tests.conftest import make_gradient, mean_rgb, pixel_diff
from tests.helpers import pixel


def adj(**kw: int) -> dict[str, int]:
    return {**default_adjustments(), **kw}


def test_slider_specs_are_consistent() -> None:
    assert set(GROUPS) == {spec.group for spec in SLIDERS.values()}
    for key, spec in SLIDERS.items():
        assert spec.min <= spec.default <= spec.max, key
        assert spec.label


def test_default_and_normalize() -> None:
    defaults = default_adjustments()
    assert set(defaults) == set(SLIDERS)
    assert is_neutral(defaults)
    assert is_neutral(None)
    assert is_neutral({})
    normalized = normalize_adjustments({"contrast": 500, "unknown": 3, "blur": -4, "hue": 12.6})
    assert normalized["contrast"] == SLIDERS["contrast"].max
    assert normalized["blur"] == 0
    assert normalized["hue"] == 13
    assert "unknown" not in normalized
    assert not is_neutral(normalized)


def test_neutral_adjustments_keep_pixels(gradient: Image.Image) -> None:
    out = apply_adjustments(gradient, default_adjustments())
    assert out.mode == "RGB"
    assert pixel_diff(out, gradient) == 0
    rgba = gradient.convert("RGBA")
    out = apply_adjustments(rgba, None)
    assert out.mode == "RGBA"
    assert pixel_diff(out, rgba) == 0


def test_alpha_channel_is_preserved(shapes: Image.Image) -> None:
    out = apply_adjustments(shapes, adj(contrast=50, saturation=40, hue=90))
    assert out.mode == "RGBA"
    assert out.getchannel("A").tobytes() == shapes.getchannel("A").tobytes()


def test_palette_image_with_transparency(shapes: Image.Image) -> None:
    pal = shapes.convert("P")
    out = apply_adjustments(pal, adj(brightness=10))
    assert out.mode == "RGB"  # no transparency info -> RGB


@pytest.mark.parametrize("key", ["brightness", "exposure"])
def test_light_sliders_direction(gradient: Image.Image, key: str) -> None:
    brighter = mean_rgb(apply_adjustments(gradient, adj(**{key: 40})))
    darker = mean_rgb(apply_adjustments(gradient, adj(**{key: -40})))
    base = mean_rgb(gradient)
    assert (brighter > base).all()
    assert (darker < base).all()


def test_contrast_changes_spread(gradient: Image.Image) -> None:
    base = np.asarray(gradient, dtype=np.float64).std()
    assert np.asarray(apply_adjustments(gradient, adj(contrast=60)), dtype=np.float64).std() > base
    assert np.asarray(apply_adjustments(gradient, adj(contrast=-60)), dtype=np.float64).std() < base


def test_gamma_brightens_midtones() -> None:
    mid = Image.new("RGB", (4, 4), (128, 128, 128))
    assert pixel(apply_adjustments(mid, adj(gamma=60)), 0, 0)[0] > 128
    assert pixel(apply_adjustments(mid, adj(gamma=-60)), 0, 0)[0] < 128
    # black & white stay fixed under gamma
    assert pixel(apply_adjustments(Image.new("RGB", (1, 1), (0, 0, 0)), adj(gamma=60)), 0, 0) == (0, 0, 0)
    assert pixel(apply_adjustments(Image.new("RGB", (1, 1), (255, 255, 255)), adj(gamma=60)), 0, 0) == (255, 255, 255)


def test_highlights_and_shadows_target_their_ranges() -> None:
    dark = Image.new("RGB", (2, 2), (30, 30, 30))
    bright = Image.new("RGB", (2, 2), (225, 225, 225))
    # highlights recovery darkens bright pixels but barely touches dark ones
    assert pixel(apply_adjustments(bright, adj(highlights=-80)), 0, 0)[0] < 225
    assert pixel(apply_adjustments(dark, adj(highlights=-80)), 0, 0)[0] == 30
    # shadow lift brightens dark pixels but not bright ones
    assert pixel(apply_adjustments(dark, adj(shadows=80)), 0, 0)[0] > 30
    assert pixel(apply_adjustments(bright, adj(shadows=80)), 0, 0)[0] == 225


def test_saturation_and_grayscale() -> None:
    color = Image.new("RGB", (2, 2), (200, 100, 50))
    gray = pixel(apply_adjustments(color, adj(grayscale=100)), 0, 0)
    assert gray[0] == gray[1] == gray[2]
    desat = pixel(apply_adjustments(color, adj(saturation=-100)), 0, 0)
    assert max(desat) - min(desat) < 3
    more = pixel(apply_adjustments(color, adj(saturation=60)), 0, 0)
    assert max(more) - min(more) > 150
    half = pixel(apply_adjustments(color, adj(grayscale=50)), 0, 0)
    assert 0 < max(half) - min(half) < 150


def test_vibrance_boosts_muted_more_than_saturated() -> None:
    muted = Image.new("RGB", (1, 1), (140, 120, 110))
    vivid = Image.new("RGB", (1, 1), (255, 40, 20))

    def sat(px: tuple[int, ...]) -> float:
        return (max(px) - min(px)) / max(max(px), 1)

    m0, v0 = sat(pixel(muted, 0, 0)), sat(pixel(vivid, 0, 0))
    m1 = sat(pixel(apply_adjustments(muted, adj(vibrance=80)), 0, 0))
    v1 = sat(pixel(apply_adjustments(vivid, adj(vibrance=80)), 0, 0))
    assert m1 > m0
    assert (m1 - m0) > (v1 - v0)
    assert sat(pixel(apply_adjustments(vivid, adj(vibrance=-100)), 0, 0)) < v0


def test_temperature_and_tint() -> None:
    gray = Image.new("RGB", (1, 1), (128, 128, 128))
    warm = pixel(apply_adjustments(gray, adj(temperature=60)), 0, 0)
    cool = pixel(apply_adjustments(gray, adj(temperature=-60)), 0, 0)
    assert warm[0] > 128 > warm[2]
    assert cool[0] < 128 < cool[2]
    magenta = pixel(apply_adjustments(gray, adj(tint=60)), 0, 0)
    green = pixel(apply_adjustments(gray, adj(tint=-60)), 0, 0)
    assert magenta[1] < 128 < green[1]


def test_hue_rotation_180_is_roughly_inverse_hue() -> None:
    red = Image.new("RGB", (1, 1), (255, 0, 0))
    rotated = pixel(apply_adjustments(red, adj(hue=180)), 0, 0)
    assert rotated[0] < 20 and rotated[1] > 200 and rotated[2] > 200  # cyan
    back = pixel(apply_adjustments(apply_adjustments(red, adj(hue=120)), adj(hue=-120)), 0, 0)
    assert abs(back[0] - 255) <= 4 and back[1] <= 4 and back[2] <= 4


def test_sepia_tones() -> None:
    gray = Image.new("RGB", (1, 1), (128, 128, 128))
    r, g, b = pixel(apply_adjustments(gray, adj(sepia=100)), 0, 0)
    assert r > g > b


def test_blur_reduces_detail_and_is_scale_aware() -> None:
    img = make_gradient(200, 120)
    noisy = np.asarray(img).astype(np.int16)
    rng = np.random.default_rng(0)
    noisy += rng.integers(-40, 40, noisy.shape, dtype=np.int16)
    noisy_img = Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), "RGB")
    blurred = apply_adjustments(noisy_img, adj(blur=6))
    assert np.asarray(blurred, dtype=np.float64).std() < np.asarray(noisy_img, dtype=np.float64).std()
    # preview at half scale should approximate the downscaled full-resolution result
    full = apply_adjustments(noisy_img, adj(blur=6)).resize((100, 60), Image.Resampling.BILINEAR)
    half_src = noisy_img.resize((100, 60), Image.Resampling.BILINEAR)
    preview = apply_adjustments(half_src, adj(blur=6), doc_size=(200, 120), scale=0.5)
    assert pixel_diff(full, preview) < 6


def test_blur_on_rgba_keeps_edges_clean(shapes: Image.Image) -> None:
    out = apply_adjustments(shapes, adj(blur=20))
    assert out.mode == "RGBA"
    # premultiplied blur: semi-transparent edge pixels next to the red square stay reddish, not dark
    arr = np.asarray(out)
    edge = arr[14, 25]
    assert 0 < edge[3] < 255
    assert edge[0] > 200 and edge[1] < 30


def test_sharpen_increases_edge_contrast() -> None:
    img = Image.new("RGB", (40, 20), (100, 100, 100))
    img.paste((160, 160, 160), (20, 0, 40, 20))
    soft = img.filter(__import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(1.5))
    sharp = apply_adjustments(soft, adj(sharpen=80))
    row_soft = np.asarray(soft, dtype=np.float64)[10, :, 0]
    row_sharp = np.asarray(sharp, dtype=np.float64)[10, :, 0]
    assert np.abs(np.diff(row_sharp)).max() > np.abs(np.diff(row_soft)).max()


def test_vignette_darkens_corners_only() -> None:
    white = Image.new("RGB", (100, 80), (200, 200, 200))
    out = np.asarray(apply_adjustments(white, adj(vignette=80)))
    assert out[40, 50, 0] == 200  # centre untouched
    assert out[0, 0, 0] < 150  # corner darkened
    assert out[0, 0, 0] == out[79, 99, 0]  # symmetric


def test_vignette_region_rendering_matches_full() -> None:
    white = Image.new("RGB", (100, 80), (200, 200, 200))
    full = apply_adjustments(white, adj(vignette=80))
    region = apply_adjustments(white.crop((60, 40, 100, 80)), adj(vignette=80), doc_size=(100, 80), region=(60, 40, 100, 80))
    assert pixel_diff(full.crop((60, 40, 100, 80)), region) < 1
