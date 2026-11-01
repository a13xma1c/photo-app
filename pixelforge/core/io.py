"""Opening/saving images and PixelForge project files (``.pfp``).

A project file is a zip archive containing ``manifest.json`` (document size,
layer properties, selection flag) plus one PNG per layer (and one per layer
mask), so layers, masks, blend modes and non-destructive adjustments survive a
save/load round trip.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .. import __version__
from .colorspace import Metadata
from .document import Document, DocumentError
from .layers import Layer
from .selection import Selection

PROJECT_EXT = ".pfp"
PROJECT_FORMAT_VERSION = 1

IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".ico", ".ppm")
EXPORT_FORMATS: dict[str, tuple[str, bool]] = {
    # extension: (Pillow format, supports alpha)
    ".png": ("PNG", True),
    ".jpg": ("JPEG", False),
    ".jpeg": ("JPEG", False),
    ".webp": ("WEBP", True),
    ".bmp": ("BMP", False),
    ".tif": ("TIFF", True),
    ".tiff": ("TIFF", True),
    ".gif": ("GIF", False),
    ".ico": ("ICO", True),
}


class UnsupportedFormat(DocumentError):
    pass


def is_project_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() == PROJECT_EXT


def open_image(path: str | Path) -> Image.Image:
    """Load an image file as RGBA, honouring EXIF orientation (first frame for animations); metadata stays in ``.info``."""
    with Image.open(path) as im:
        im.load()
        if getattr(im, "n_frames", 1) > 1:
            im.seek(0)
        info = dict(im.info)
        img = ImageOps.exif_transpose(im) or im
        rgba = img.convert("RGBA")
        for key in ("icc_profile", "exif", "dpi"):
            if key in info and key not in rgba.info:
                rgba.info[key] = info[key]
        return rgba


def open_document(path: str | Path, color_manage: bool = True) -> Document:
    """Open either a project file or a plain image as a Document."""
    path = Path(path)
    if is_project_file(path):
        return load_project(path)
    img = open_image(path)
    doc = Document.from_image(img, name=path.stem, color_manage=color_manage)
    doc.source_path = str(path)
    return doc


def save_image(doc: Document, path: str | Path, quality: int = 92, background: tuple = (255, 255, 255), metadata: bool = True) -> None:
    """Export the flattened document in the format implied by the extension (with its ICC profile / EXIF unless *metadata* is off)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in EXPORT_FORMATS:
        raise UnsupportedFormat(f"unsupported export format '{ext or path.name}'")
    fmt, has_alpha = EXPORT_FORMATS[ext]
    img = doc.export_image("RGBA" if has_alpha else "RGB", background)
    # without metadata the profile must be suppressed explicitly: Pillow would otherwise take it from ``img.info``
    kwargs: dict = doc.metadata.save_kwargs(fmt) if metadata else {"icc_profile": None}
    if fmt == "JPEG":
        kwargs.update(quality=int(quality), optimize=True, subsampling=0 if quality >= 90 else 2)
    elif fmt == "WEBP":
        kwargs.update(quality=int(quality), method=4)
    elif fmt == "PNG":
        kwargs.update(optimize=True)
    elif fmt == "GIF":
        img = img.convert("P", palette=Image.Palette.ADAPTIVE)
    img.save(path, fmt, **kwargs)


def save_project(doc: Document, path: str | Path, record: bool = True) -> None:
    """Write a project file. With ``record=False`` the document's path/saved state is left untouched (autosave)."""
    path = Path(path)
    manifest: dict[str, Any] = {
        "app": "PixelForge",
        "app_version": __version__,
        "format_version": PROJECT_FORMAT_VERSION,
        "width": doc.width,
        "height": doc.height,
        "name": doc.name,
        "active_index": doc.active_index,
        "has_selection": doc.selection is not None,
        "dpi": list(doc.metadata.dpi) if doc.metadata.dpi else None,
        "source_profile": doc.metadata.source_profile,
        "layers": [],
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        if doc.metadata.icc_profile:
            zf.writestr("meta/profile.icc", doc.metadata.icc_profile)
        if doc.metadata.exif:
            zf.writestr("meta/exif.bin", doc.metadata.exif)
        for i, layer in enumerate(doc.layers):
            file_name = f"layers/{i:03d}.png"
            entry = layer.to_manifest()
            entry["file"] = file_name
            if layer.mask is not None:
                entry["mask_file"] = f"layers/{i:03d}.mask.png"
                zf.writestr(entry["mask_file"], _png_bytes(layer.mask))
            manifest["layers"].append(entry)
            zf.writestr(file_name, _png_bytes(layer.image))
        if doc.selection is not None:
            zf.writestr("selection.png", _png_bytes(doc.selection.mask))
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
    if record:
        doc.path = str(path)
        doc.name = path.stem
        doc.mark_saved()


def load_project(path: str | Path) -> Document:
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as zf:
            manifest = json.loads(zf.read("manifest.json"))
            if manifest.get("format_version", 0) > PROJECT_FORMAT_VERSION:
                raise UnsupportedFormat("project was saved by a newer version of PixelForge")
            size = (int(manifest["width"]), int(manifest["height"]))
            layers = []
            for entry in manifest["layers"]:
                img = Image.open(io.BytesIO(zf.read(entry["file"])))
                img.load()
                if img.size != size:
                    raise DocumentError("layer size in project does not match document size")
                mask = None
                if entry.get("mask_file"):
                    mask = Image.open(io.BytesIO(zf.read(entry["mask_file"])))
                    mask.load()
                    if mask.size != size:
                        raise DocumentError("mask size in project does not match document size")
                layers.append(Layer.from_manifest(entry, img, mask))
            selection = None
            if manifest.get("has_selection") and "selection.png" in zf.namelist():
                mask = Image.open(io.BytesIO(zf.read("selection.png")))
                mask.load()
                selection = Selection(mask)
            names = set(zf.namelist())
            metadata = Metadata(
                icc_profile=zf.read("meta/profile.icc") if "meta/profile.icc" in names else None,
                exif=zf.read("meta/exif.bin") if "meta/exif.bin" in names else None,
                dpi=(float(manifest["dpi"][0]), float(manifest["dpi"][1])) if manifest.get("dpi") else None,
                source_profile=str(manifest.get("source_profile") or ""),
            )
    except (zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise DocumentError(f"not a valid PixelForge project: {exc}") from exc
    # group membership refers to layer ids, which are re-issued on load: remap them
    id_map = {entry["id"]: layer.id for entry, layer in zip(manifest["layers"], layers, strict=True) if "id" in entry}
    for layer in layers:
        gid = layer.meta.get("group")
        if gid is not None:
            if gid in id_map:
                layer.meta["group"] = id_map[gid]
            else:
                layer.meta.pop("group")
    doc = Document(size[0], size[1], layers, name=manifest.get("name") or path.stem)
    doc.active_index = max(0, min(len(layers) - 1, int(manifest.get("active_index", len(layers) - 1))))
    doc.selection = selection
    doc.metadata = metadata
    doc.history.clear()
    doc.commit("Open Project")
    doc.path = str(path)
    doc.mark_saved()
    return doc


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", compress_level=6)
    return buf.getvalue()
