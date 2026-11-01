"""Small typed helpers shared by the test-suite."""

from __future__ import annotations

from PIL import Image

from pixelforge.core.document import Document
from pixelforge.core.selection import Selection


def pixel(image: Image.Image, x: int, y: int) -> tuple[int, ...]:
    """``image.getpixel`` narrowed to a tuple of channels (what every test here expects)."""
    value = image.getpixel((x, y))
    if isinstance(value, tuple):
        return tuple(int(c) for c in value)
    assert value is not None
    return (int(value),)


def gray(image: Image.Image, x: int, y: int) -> int:
    """Single-channel pixel value of an ``L``/``1`` image."""
    return pixel(image, x, y)[0]


def sel(doc: Document) -> Selection:
    """The document's selection, asserted to exist."""
    assert doc.selection is not None
    return doc.selection


def copied(doc: Document, merged: bool = False) -> tuple[Image.Image, tuple[int, int]]:
    result = doc.copy_selection(merged=merged)
    assert result is not None
    return result


def mask_of(doc: Document) -> Image.Image:
    """The active layer's mask, asserted to exist."""
    mask = doc.active_layer.mask
    assert mask is not None
    return mask
