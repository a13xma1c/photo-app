import numpy as np
import pytest
from PIL import Image

from pixelforge.core import filters
from pixelforge.core.filters import CATEGORIES, FILTERS, Param, apply_filter, filters_in
from tests.conftest import make_gradient, pixel_diff
from tests.helpers import pixel


@pytest.fixture
def rgba() -> Image.Image:
    img = make_gradient(48, 32, alpha=True)
    img.paste((0, 0, 0, 0), (0, 0, 8, 32))  # transparent strip on the left
    return img


def test_registry_is_consistent() -> None:
    assert FILTERS
    for spec in FILTERS.values():
        assert spec.category in CATEGORIES, spec.name
        for p in spec.params:
            assert p.kind in ("int", "float", "bool", "choice")
            if p.kind == "choice":
                assert p.choices
            elif p.kind != "bool":
                assert isinstance(p.default, (int, float)) and p.min <= p.default <= p.max, (spec.name, p.name)
    assert sum(len(filters_in(c)) for c in CATEGORIES) == len(FILTERS)


@pytest.mark.parametrize("name", sorted(FILTERS))
def test_every_filter_runs_with_defaults(name: str, rgba: Image.Image) -> None:
    out = apply_filter(rgba, name)
    assert out.size == rgba.size
    assert out.mode == "RGBA"
    if name not in (
        "Drop Shadow",
        "Outline",
        "Wave",
        "Swirl",
        "Motion Blur",
        "Gaussian Blur",
        "Box Blur",
        "Median",
        "Smooth",
        "Despeckle",
        "Pixelate",
        "Oil Paint",
    ):
        # colour-only filters keep the alpha channel untouched
        assert out.getchannel("A").tobytes() == rgba.getchannel("A").tobytes(), name


def test_param_coerce() -> None:
    p = Param("x", "X", 0, 10, 5, kind="int")
    assert p.coerce(3.7) == 4 and p.coerce(-5) == 0 and p.coerce("12") == 10
    f = Param("y", "Y", 0.0, 1.0, 0.5)
    assert f.coerce("0.25") == 0.25
    b = Param("z", "Z", 0, 1, 1, kind="bool")
    assert b.coerce("yes") is True and b.coerce("0") is False and b.coerce(0) is False
    c = Param("c", "C", 0, 0, 0, kind="choice", choices=("a", "b"))
    assert c.coerce("a") == "a"
    with pytest.raises(ValueError):
        c.coerce("q")


def test_unknown_filter_and_param(rgba: Image.Image) -> None:
    with pytest.raises(KeyError):
        apply_filter(rgba, "Nope")
    with pytest.raises(ValueError):
        apply_filter(rgba, "Gaussian Blur", sigma=3)
    spec = FILTERS["Gaussian Blur"]
    assert spec.defaults() == {"radius": 3}


def test_invert_and_desaturate(rgba: Image.Image) -> None:
    inv = apply_filter(rgba, "Invert")
    px, ipx = pixel(rgba, 40, 16), pixel(inv, 40, 16)
    assert ipx[:3] == tuple(255 - c for c in px[:3]) and ipx[3] == px[3]
    gray = pixel(apply_filter(rgba, "Desaturate"), 40, 16)
    assert gray[0] == gray[1] == gray[2]


def test_threshold_posterize_solarize(rgba: Image.Image) -> None:
    th = np.asarray(apply_filter(rgba, "Threshold", level=128))
    assert set(np.unique(th[..., :3])) <= {0, 255}
    post = np.asarray(apply_filter(rgba, "Posterize", bits=1))
    assert set(np.unique(post[..., :3])) <= {0, 128}
    sol = pixel(apply_filter(Image.new("RGBA", (2, 2), (200, 50, 200, 255)), "Solarize", threshold=128), 0, 0)
    assert sol[:3] == (55, 50, 55)


def test_levels() -> None:
    img = Image.new("RGBA", (3, 1))
    img.putdata([(0, 0, 0, 255), (128, 128, 128, 255), (255, 255, 255, 255)])
    out = apply_filter(img, "Levels", in_black=64, in_white=192)
    assert pixel(out, 0, 0)[0] == 0 and pixel(out, 2, 0)[0] == 255
    assert pixel(out, 1, 0)[0] in (127, 128)
    squeezed = apply_filter(img, "Levels", out_black=50, out_white=100)
    assert pixel(squeezed, 0, 0)[0] == 50 and pixel(squeezed, 2, 0)[0] == 100
    gamma = pixel(apply_filter(img, "Levels", gamma=2.0), 1, 0)[0]
    assert gamma > 128
    degenerate = apply_filter(img, "Levels", in_black=200, in_white=100)  # white clamped above black
    assert pixel(degenerate, 2, 0)[0] == 255


def test_color_balance_and_colorize() -> None:
    gray = Image.new("RGBA", (2, 2), (128, 128, 128, 255))
    out = pixel(apply_filter(gray, "Color Balance", red=30, green=-20, blue=0), 0, 0)
    assert out == (158, 108, 128, 255)
    col = pixel(apply_filter(gray, "Colorize", hue=0, saturation=100), 0, 0)
    assert col[0] > col[1] and col[0] > col[2]


def test_auto_contrast_and_equalize_expand_range() -> None:
    flat = make_gradient(32, 32, alpha=True)
    arr = np.asarray(flat).copy()
    arr[..., :3] = arr[..., :3] // 2 + 64  # compress to 64..191
    img = Image.fromarray(arr, "RGBA")
    ac = np.asarray(apply_filter(img, "Auto Contrast", cutoff=0))
    assert ac[..., 0].min() == 0 and ac[..., 0].max() == 255
    eq = np.asarray(apply_filter(img, "Equalize"))
    assert eq[..., 0].max() - eq[..., 0].min() > 200


def test_blurs_reduce_variance(rgba: Image.Image) -> None:
    base = np.asarray(rgba, dtype=np.float64)[..., :3].std()
    for name, kw in (
        ("Gaussian Blur", {"radius": 4}),
        ("Box Blur", {"radius": 3}),
        ("Motion Blur", {"length": 9, "angle": 0}),
        ("Motion Blur", {"length": 5, "angle": 90}),
        ("Smooth", {}),
        ("Median", {"size": 5}),
    ):
        out = np.asarray(apply_filter(rgba, name, **kw), dtype=np.float64)[..., :3]
        assert out.std() <= base, name
    # Motion blur along the horizontal smears a vertical line horizontally
    line = Image.new("RGBA", (41, 11), (0, 0, 0, 255))
    line.paste((255, 255, 255, 255), (20, 0, 21, 11))
    horiz = np.asarray(apply_filter(line, "Motion Blur", length=9, angle=0))
    assert horiz[5, 16, 0] > 0 and horiz[5, 20, 0] < 255
    vert = np.asarray(apply_filter(line, "Motion Blur", length=9, angle=90))
    assert vert[5, 16, 0] == 0


def test_gaussian_blur_premultiplied_edges() -> None:
    img = Image.new("RGBA", (40, 20), (0, 0, 0, 0))
    img.paste((255, 255, 255, 255), (20, 0, 40, 20))
    out = apply_filter(img, "Gaussian Blur", radius=3)
    px = pixel(out, 19, 10)
    assert 0 < px[3] < 255
    assert px[0] > 200  # colour stays white even where alpha fades (no dark fringe)


def test_sharpen_filters_increase_edges() -> None:
    img = Image.new("RGBA", (40, 20), (100, 100, 100, 255))
    img.paste((160, 160, 160, 255), (20, 0, 40, 20))
    soft = apply_filter(img, "Gaussian Blur", radius=1.5)
    for name in ("Sharpen", "Unsharp Mask", "Edge Enhance", "Detail"):
        out = np.asarray(apply_filter(soft, name), dtype=np.float64)[10, :, 0]
        assert np.abs(np.diff(out)).max() >= np.abs(np.diff(np.asarray(soft, dtype=np.float64)[10, :, 0])).max() - 1e-6, name


def test_stylize_filters(rgba: Image.Image) -> None:
    pix = apply_filter(rgba, "Pixelate", block=8)
    arr = np.asarray(pix)
    block = arr[8:16, 16:24]
    assert (block == block[0, 0]).all()  # uniform blocks
    edges = apply_filter(Image.new("RGBA", (20, 20), (128, 128, 128, 255)), "Find Edges")
    assert np.asarray(edges)[5:15, 5:15, :3].max() == 0  # flat image has no edges
    for name in ("Emboss", "Contour", "Oil Paint", "Vignette"):
        assert apply_filter(rgba, name).size == rgba.size
    vig = np.asarray(apply_filter(Image.new("RGBA", (60, 40), (200, 200, 200, 255)), "Vignette", strength=80, softness=50))
    assert vig[20, 30, 0] == 200 and vig[0, 0, 0] < 200


def test_drop_shadow_and_outline(shapes: Image.Image) -> None:
    shadow = apply_filter(shapes, "Drop Shadow", offset_x=6, offset_y=6, blur=0, opacity=100)
    assert pixel(shadow, 28, 28)[3] > 0  # shadow below-right of the red square
    assert pixel(shadow, 28, 28)[:3] == (0, 0, 0)
    assert pixel(shapes, 28, 28)[3] == 0
    outline = apply_filter(shapes, "Outline", width=2, color="white")
    assert pixel(outline, 2, 14) == (255, 255, 255, 255)
    assert pixel(outline, 10, 10) == (255, 0, 0, 255)  # interior intact


def test_noise_filters(rgba: Image.Image) -> None:
    a = apply_filter(rgba, "Add Noise", amount=30, seed=7)
    b = apply_filter(rgba, "Add Noise", amount=30, seed=7)
    c = apply_filter(rgba, "Add Noise", amount=30, seed=8)
    assert pixel_diff(a, b) == 0 and pixel_diff(a, c) > 0 and pixel_diff(a, rgba) > 0
    mono = np.asarray(apply_filter(Image.new("RGBA", (20, 20), (128, 128, 128, 255)), "Add Noise", amount=40, monochrome=True, seed=1))
    assert (mono[..., 0] == mono[..., 1]).all()
    color = np.asarray(apply_filter(Image.new("RGBA", (20, 20), (128, 128, 128, 255)), "Add Noise", amount=40, monochrome=False, seed=1))
    assert not (color[..., 0] == color[..., 1]).all()
    grain = apply_filter(rgba, "Film Grain", amount=40, size=2)
    assert pixel_diff(grain, rgba) > 0
    assert apply_filter(rgba, "Despeckle").size == rgba.size


def test_distort_filters(rgba: Image.Image) -> None:
    for name, kw in (
        ("Wave", {"amplitude": 5, "wavelength": 20, "direction": "horizontal"}),
        ("Wave", {"amplitude": 5, "wavelength": 20, "direction": "vertical"}),
        ("Swirl", {"angle": 180, "radius": 100}),
    ):
        out = apply_filter(rgba, name, **kw)
        assert out.size == rgba.size and pixel_diff(out, rgba) > 0
    assert pixel_diff(apply_filter(rgba, "Wave", amplitude=0), rgba) == 0
    assert pixel_diff(apply_filter(rgba, "Swirl", angle=0), rgba) == 0


def test_difference_helper(rgba: Image.Image) -> None:
    assert filters.difference(rgba, rgba) == 0
    assert filters.difference(rgba, apply_filter(rgba, "Invert")) > 0
