import pytest

from pixelforge.core.color import clamp, contrast_text_color, luminance, parse_color, to_hex


def test_clamp() -> None:
    assert clamp(-5) == 0
    assert clamp(300) == 255
    assert clamp(12.5) == 12.5
    assert clamp(5, 0, 3) == 3


@pytest.mark.parametrize(
    "value, expected",
    [
        ("#ff0000", (255, 0, 0, 255)),
        ("#F00", (255, 0, 0, 255)),
        ("00ff0080", (0, 255, 0, 128)),
        ((1, 2, 3), (1, 2, 3, 255)),
        ((1, 2, 3, 4), (1, 2, 3, 4)),
        ([10, 20, 30], (10, 20, 30, 255)),
    ],
)
def test_parse_color(value: str | tuple[int, ...] | list[int], expected: tuple[int, int, int, int]) -> None:
    assert parse_color(value) == expected


@pytest.mark.parametrize("bad", ["#12345", "zzz", (1, 2), (1, 2, 3, 4, 5)])
def test_parse_color_invalid(bad: str | tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        parse_color(bad)


def test_to_hex() -> None:
    assert to_hex((255, 0, 128)) == "#ff0080"
    assert to_hex((255, 0, 128, 7), alpha=True) == "#ff008007"
    assert to_hex("#abc") == "#aabbcc"


def test_luminance_and_contrast() -> None:
    assert luminance((255, 255, 255)) == pytest.approx(255)
    assert luminance((0, 0, 0)) == 0
    assert contrast_text_color("#ffffff") == "#000000"
    assert contrast_text_color((10, 10, 10)) == "#ffffff"
