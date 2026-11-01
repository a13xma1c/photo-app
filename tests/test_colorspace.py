"""Colour management: ICC profiles converted to sRGB on open, metadata carried to exports and project files."""

import io
from pathlib import Path

import pytest
from PIL import Image, ImageCms

from pixelforge.cli import main as cli_main
from pixelforge.core import colorspace as cs
from pixelforge.core import io as pfio
from pixelforge.core.document import Document
from tests.helpers import pixel


def test_profile_helpers() -> None:
    srgb = cs.srgb_profile()
    assert len(srgb) > 100 and cs.is_srgb(srgb) and cs.is_srgb(None) and "sRGB" in cs.profile_description(srgb)
    assert cs.profile_description(b"not a profile") == "" and not cs.is_srgb(b"garbage")
    img = Image.new("RGBA", (4, 4), (10, 20, 30, 128))
    same, converted = cs.to_srgb(img, srgb)
    assert same is img and not converted
    broken, converted = cs.to_srgb(img, b"garbage")
    assert broken is img and not converted


def other_rgb_profile() -> bytes:
    """An RGB profile that is *not* recognised as sRGB (the built-in sRGB with its description renamed)."""
    return cs.srgb_profile().replace("sRGB".encode("utf-16-be"), "wide".encode("utf-16-be"))


def test_to_srgb_converts_other_profiles() -> None:
    other = other_rgb_profile()
    assert not cs.is_srgb(other) and "wide" in cs.profile_description(other)
    img = Image.new("RGBA", (4, 4), (128, 128, 128, 200))
    out, converted = cs.to_srgb(img, other)
    assert converted and out is not img and out.mode == "RGBA" and pixel(out, 0, 0)[3] == 200
    assert pixel(out, 0, 0)[:3] == pytest.approx((128, 128, 128), abs=2)  # same primaries: colours survive the transform
    # a profile that cannot describe RGB data (LAB) is left alone
    lab_bytes = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()
    assert not cs.is_srgb(lab_bytes)
    same, converted = cs.to_srgb(img, lab_bytes)
    assert same is img and not converted


def test_exif_orientation_is_reset() -> None:
    exif = Image.Exif()
    exif[cs.ORIENTATION_TAG] = 6
    exif[0x010F] = "PixelForge Cam"
    upright = cs.upright_exif(exif.tobytes())
    assert upright is not None
    parsed = Image.Exif()
    parsed.load(upright)
    assert parsed[cs.ORIENTATION_TAG] == 1 and parsed[0x010F] == "PixelForge Cam"
    assert cs.upright_exif(None) is None and cs.upright_exif(b"\xff\xfe broken") is None
    plain = Image.Exif()
    plain[0x010F] = "x"
    assert cs.upright_exif(plain.tobytes()) == plain.tobytes()


def test_metadata_travels_from_open_to_export(tmp_path: Path) -> None:
    src = tmp_path / "photo.jpg"
    exif = Image.Exif()
    exif[0x010F] = "PixelForge Cam"
    exif[cs.ORIENTATION_TAG] = 8
    Image.new("RGB", (20, 10), (200, 60, 40)).save(src, "JPEG", icc_profile=cs.srgb_profile(), exif=exif.tobytes(), dpi=(300, 300))
    doc = pfio.open_document(src)
    assert doc.size == (10, 20)  # rotated upright
    meta = doc.metadata
    assert meta.icc_profile and cs.is_srgb(meta.icc_profile) and meta.dpi == (300.0, 300.0) and "sRGB" in meta.source_profile
    assert "EXIF" in meta.describe() and "300 dpi" in meta.describe()
    out = tmp_path / "out.jpg"
    pfio.save_image(doc, out)
    with Image.open(out) as saved:
        saved_exif = saved.getexif()
        assert saved.info.get("icc_profile") and saved_exif[0x010F] == "PixelForge Cam" and saved_exif.get(cs.ORIENTATION_TAG, 1) == 1
        assert saved.info.get("dpi") == (300, 300)
    stripped = tmp_path / "stripped.jpg"
    pfio.save_image(doc, stripped, metadata=False)
    with Image.open(stripped) as saved:
        assert not saved.info.get("icc_profile") and not saved.getexif()
    png = tmp_path / "out.png"
    pfio.save_image(doc, png)
    with Image.open(png) as saved:
        assert saved.info.get("icc_profile") and saved.getexif()[0x010F] == "PixelForge Cam"
    gif = tmp_path / "out.gif"
    pfio.save_image(doc, gif)  # formats without metadata support just skip it
    # project files keep it too
    project = tmp_path / "photo.pfp"
    pfio.save_project(doc, project)
    loaded = pfio.load_project(project)
    assert loaded.metadata.icc_profile == meta.icc_profile and loaded.metadata.exif == meta.exif and loaded.metadata.dpi == (300.0, 300.0)
    assert loaded.metadata.source_profile == meta.source_profile
    # opening without colour management keeps pixels and profile as they are
    raw = pfio.open_document(src, color_manage=False)
    assert raw.metadata.icc_profile == cs.srgb_profile()


def test_document_from_image_converts_to_srgb() -> None:
    other = other_rgb_profile()
    img = Image.new("RGB", (8, 8), (128, 128, 128))
    img.info["icc_profile"] = other
    doc = Document.from_image(img)
    assert cs.is_srgb(doc.metadata.icc_profile) and doc.metadata.source_profile.startswith("wide")
    untouched = Document.from_image(img, color_manage=False)
    assert untouched.metadata.icc_profile == other
    plain = Document.from_image(Image.new("RGB", (2, 2)))
    assert plain.metadata.icc_profile is None and plain.metadata.describe() == ""


def test_cli_metadata_flags(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    Image.new("RGB", (6, 6), (1, 2, 3)).save(src, "PNG", icc_profile=cs.srgb_profile(), dpi=(72, 72))
    out, err = io.StringIO(), io.StringIO()
    assert cli_main([str(src), "--info"], out, err) == 0 and "sRGB" in out.getvalue()
    dest = tmp_path / "out.png"
    assert cli_main([str(src), str(dest), "--strip-metadata", "--no-color-management"], io.StringIO(), err) == 0
    with Image.open(dest) as saved:
        assert not saved.info.get("icc_profile")
