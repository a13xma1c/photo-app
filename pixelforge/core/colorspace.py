"""Colour management: embedded ICC profiles and EXIF metadata.

PixelForge edits in sRGB (what screens assume).  Pictures that carry another
profile (Adobe RGB, Display P3, …) are converted to sRGB when opened so they
look right on screen, and exports embed an sRGB profile.  EXIF data travels
with the document and is written back on export - with the orientation tag
reset, because the pixels are already upright.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

from PIL import Image, ImageCms

ORIENTATION_TAG = 0x0112
_SRGB_PROFILE: bytes | None = None


def srgb_profile() -> bytes:
    """The sRGB ICC profile as bytes (built once with LittleCMS)."""
    global _SRGB_PROFILE
    if _SRGB_PROFILE is None:
        _SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    return _SRGB_PROFILE


def profile_description(icc: bytes | None) -> str:
    """Human-readable name of an ICC profile ("" when absent or unreadable)."""
    if not icc:
        return ""
    try:
        profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        return str(ImageCms.getProfileDescription(profile)).strip()
    except (OSError, ValueError, ImageCms.PyCMSError):
        return ""


def is_srgb(icc: bytes | None) -> bool:
    """True when *icc* is absent or describes (a flavour of) sRGB."""
    if not icc:
        return True
    description = profile_description(icc).lower()
    return "srgb" in description or "iec61966" in description


def to_srgb(image: Image.Image, icc: bytes | None) -> tuple[Image.Image, bool]:
    """Convert *image* from its embedded profile to sRGB.

    Returns ``(image, converted)``; the image is returned untouched when the
    profile is missing, already sRGB or cannot be applied.
    """
    if is_srgb(icc):
        return image, False
    assert icc is not None
    try:
        source = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        target = ImageCms.createProfile("sRGB")
        mode = "RGBA" if image.mode == "RGBA" else "RGB"
        rgb = image.convert("RGB")
        converted = ImageCms.profileToProfile(rgb, source, target, outputMode="RGB")
        assert converted is not None
        if mode == "RGBA":
            converted.putalpha(image.convert("RGBA").getchannel("A"))
        return converted, True
    except (OSError, ValueError, ImageCms.PyCMSError):
        return image, False


def upright_exif(exif: bytes | None) -> bytes | None:
    """EXIF bytes with the orientation reset to 1 (the document's pixels are already rotated)."""
    if not exif:
        return None
    try:
        data = Image.Exif()
        data.load(exif)
        if data.get(ORIENTATION_TAG, 1) != 1:
            data[ORIENTATION_TAG] = 1
            return data.tobytes()
        return exif
    except Exception:  # noqa: BLE001 - corrupt EXIF: better to drop it than to fail the export
        return None


@dataclass
class Metadata:
    """What a picture carries besides pixels: its ICC profile (sRGB after conversion), EXIF and resolution."""

    icc_profile: bytes | None = None
    exif: bytes | None = None
    dpi: tuple[float, float] | None = None
    source_profile: str = ""  # description of the profile the picture was opened with
    extra: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_image(cls, image: Image.Image) -> Metadata:
        info = image.info
        exif = info.get("exif")
        if not exif:
            try:
                parsed = image.getexif()
                exif = parsed.tobytes() if len(parsed) else None
            except Exception:  # noqa: BLE001
                exif = None
        dpi = info.get("dpi")
        return cls(
            icc_profile=info.get("icc_profile") or None,
            exif=upright_exif(exif if isinstance(exif, bytes) else None),
            dpi=(float(dpi[0]), float(dpi[1])) if dpi else None,
            source_profile=profile_description(info.get("icc_profile")),
        )

    def save_kwargs(self, fmt: str) -> dict[str, object]:
        """Keyword arguments for :meth:`PIL.Image.Image.save` that carry the metadata *fmt* supports."""
        kwargs: dict[str, object] = {}
        if fmt in ("JPEG", "PNG", "WEBP", "TIFF") and self.icc_profile:
            kwargs["icc_profile"] = self.icc_profile
        if fmt in ("JPEG", "PNG", "WEBP") and self.exif:
            kwargs["exif"] = self.exif
        if fmt in ("JPEG", "PNG", "TIFF", "BMP") and self.dpi:
            kwargs["dpi"] = self.dpi
        return kwargs

    def describe(self) -> str:
        parts = []
        if self.source_profile:
            parts.append(self.source_profile)
        if self.exif:
            parts.append("EXIF")
        if self.dpi:
            parts.append(f"{self.dpi[0]:g} dpi")
        return " · ".join(parts)
