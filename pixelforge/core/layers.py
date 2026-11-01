"""Layer model.

A :class:`Layer` is a thin record around an RGBA :class:`PIL.Image.Image`
whose size always equals the document size.  Layer images are treated as
**immutable**: every edit produces a new image object and assigns it to
``layer.image``.  This makes undo snapshots cheap (they just keep references)
and lets the renderer cache per-image results keyed by ``id(image)``.

A layer may carry a **mask**: an ``L`` image of the same size whose value
multiplies the layer's alpha when it is rendered (white reveals, black hides).
Masks are immutable in the same way as layer images.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace

from PIL import Image, ImageChops

from .adjustments import default_adjustments, is_neutral, normalize_adjustments
from .blend import BLEND_MODE_NAMES

_ids = itertools.count(1)


def next_layer_id() -> int:
    return next(_ids)


def ensure_rgba(img: Image.Image) -> Image.Image:
    """Return *img* as RGBA (the same object if it already is)."""
    return img if img.mode == "RGBA" else img.convert("RGBA")


def ensure_mask(img: Image.Image) -> Image.Image:
    """Return *img* as a single-channel ``L`` mask (the same object if it already is)."""
    return img if img.mode == "L" else img.convert("L")


def mask_to_rgba(mask: Image.Image) -> Image.Image:
    """A mask shown as an opaque grey picture - the form in which painting tools edit it."""
    return Image.merge("RGBA", (mask, mask, mask, Image.new("L", mask.size, 255)))


def rgba_to_mask(image: Image.Image) -> Image.Image:
    """Inverse of :func:`mask_to_rgba`: luminance scaled by alpha (erased pixels hide)."""
    image = ensure_rgba(image)
    gray = image.convert("L")
    alpha = image.getchannel("A")
    if alpha.getextrema() == (255, 255):
        return gray
    return ImageChops.multiply(gray, alpha)


@dataclass
class Layer:
    image: Image.Image
    name: str = "Layer"
    visible: bool = True
    opacity: float = 1.0
    blend_mode: str = "normal"
    locked: bool = False
    adjustments: dict[str, int] = field(default_factory=default_adjustments)
    kind: str = "pixel"  # "pixel" | "text" | "background"
    meta: dict = field(default_factory=dict)
    mask: Image.Image | None = None  # "L" image, same size as the layer
    mask_enabled: bool = True
    id: int = field(default_factory=next_layer_id)

    def __post_init__(self) -> None:
        self.image = ensure_rgba(self.image)
        self.opacity = float(min(1.0, max(0.0, self.opacity)))
        if self.blend_mode not in BLEND_MODE_NAMES:
            raise ValueError(f"unknown blend mode {self.blend_mode!r}")
        self.adjustments = normalize_adjustments(self.adjustments)
        if self.mask is not None:
            self.mask = ensure_mask(self.mask)
            if self.mask.size != self.image.size:
                raise ValueError("mask size must match the layer size")

    # --------------------------------------------------------------------
    @property
    def size(self) -> tuple[int, int]:
        return self.image.size

    @property
    def has_adjustments(self) -> bool:
        return not is_neutral(self.adjustments)

    @property
    def effective_mask(self) -> Image.Image | None:
        """The mask that takes part in rendering (None when absent or disabled)."""
        return self.mask if self.mask is not None and self.mask_enabled else None

    def masked_image(self) -> Image.Image:
        """The layer pixels with the (enabled) mask baked into alpha."""
        mask = self.effective_mask
        if mask is None:
            return self.image
        out = self.image.copy()
        out.putalpha(ImageChops.multiply(self.image.getchannel("A"), mask))
        return out

    def copy(self, **changes) -> Layer:
        """Shallow copy (shares the image) with optional field overrides.

        The copy keeps the same ``id`` unless one is given, so GUI code can
        keep tracking a layer across undo/redo.
        """
        new = replace(self, **changes)
        new.adjustments = dict(new.adjustments)
        new.meta = dict(new.meta)
        return new

    def duplicate(self, name: str | None = None) -> Layer:
        """Independent copy with a new id (image and mask copied as well)."""
        mask = self.mask.copy() if self.mask is not None else None
        return self.copy(image=self.image.copy(), mask=mask, name=name or f"{self.name} copy", id=next_layer_id())

    @classmethod
    def blank(cls, size: tuple[int, int], name: str = "Layer", color: tuple = (0, 0, 0, 0), **kw) -> Layer:
        return cls(Image.new("RGBA", size, color), name=name, **kw)

    def to_manifest(self) -> dict:
        """JSON-serialisable description (without pixel data)."""
        return {
            "name": self.name,
            "visible": self.visible,
            "opacity": self.opacity,
            "blend_mode": self.blend_mode,
            "locked": self.locked,
            "adjustments": dict(self.adjustments),
            "kind": self.kind,
            "meta": dict(self.meta),
            "has_mask": self.mask is not None,
            "mask_enabled": self.mask_enabled,
            "id": self.id,
        }

    @classmethod
    def from_manifest(cls, data: dict, image: Image.Image, mask: Image.Image | None = None) -> Layer:
        return cls(
            image,
            name=data.get("name", "Layer"),
            visible=bool(data.get("visible", True)),
            opacity=float(data.get("opacity", 1.0)),
            blend_mode=data.get("blend_mode", "normal"),
            locked=bool(data.get("locked", False)),
            adjustments=data.get("adjustments") or {},
            kind=data.get("kind", "pixel"),
            meta=dict(data.get("meta") or {}),
            mask=mask,
            mask_enabled=bool(data.get("mask_enabled", True)),
        )
