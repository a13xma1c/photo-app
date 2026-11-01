import io
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from pixelforge.cli import main as cli_main
from pixelforge.cli import parse_adjust, parse_box, parse_filter, parse_size
from pixelforge.core import io as pfio
from pixelforge.core.document import Document, DocumentError
from pixelforge.core.layers import Layer
from pixelforge.core.presets import PRESETS, PresetStore, preset_adjustments
from pixelforge.core.selection import Selection
from tests.conftest import pixel_diff
from tests.helpers import pixel


# --------------------------------------------------------------------- io --
def test_open_image_converts_and_orients(tmp_path: Path, gradient: Image.Image) -> None:
    exif = Image.Exif()
    exif[0x0112] = 8  # rotate 90 CW in EXIF terms -> transposed result
    path = tmp_path / "photo.jpg"
    gradient.save(path, "JPEG", exif=exif.tobytes(), quality=95)
    img = pfio.open_image(path)
    assert img.mode == "RGBA" and img.size == (gradient.height, gradient.width)
    # animated gif -> first frame
    frames = [Image.new("P", (8, 8), i) for i in range(3)]
    gif = tmp_path / "anim.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:])
    assert pfio.open_image(gif).size == (8, 8)


def test_open_document_dispatches(tmp_path: Path, image_file: Path) -> None:
    doc = pfio.open_document(image_file)
    assert isinstance(doc, Document) and doc.source_path == str(image_file) and doc.name == "gradient"
    assert not pfio.is_project_file(image_file)
    proj = tmp_path / "p.pfp"
    pfio.save_project(doc, proj)
    assert pfio.is_project_file(proj)
    assert isinstance(pfio.open_document(proj), Document)


@pytest.mark.parametrize("ext", sorted(pfio.EXPORT_FORMATS))
def test_save_image_all_formats(tmp_path: Path, shapes: Image.Image, ext: str) -> None:
    doc = Document.from_image(shapes)
    path = tmp_path / f"out{ext}"
    pfio.save_image(doc, path, quality=80)
    with Image.open(path) as im:
        fmt, has_alpha = pfio.EXPORT_FORMATS[ext]
        assert im.format == fmt
        if fmt == "ICO":
            assert max(im.size) <= 64  # icons are stored at standard sizes
            return
        assert im.size == shapes.size
        if has_alpha:
            assert pixel(im.convert("RGBA"), 0, 0)[3] == 0
        else:
            assert min(pixel(im.convert("RGB"), 0, 0)) >= 235  # matted on white (JPEG may add artefacts)


def test_save_image_unsupported_and_matte(tmp_path: Path, shapes: Image.Image) -> None:
    doc = Document.from_image(shapes)
    with pytest.raises(pfio.UnsupportedFormat):
        pfio.save_image(doc, tmp_path / "out.xyz")
    pfio.save_image(doc, tmp_path / "out.bmp", background=(0, 0, 0))
    assert pixel(Image.open(tmp_path / "out.bmp").convert("RGB"), 0, 0) == (0, 0, 0)


def test_project_roundtrip(tmp_path: Path, gradient: Image.Image, shapes: Image.Image) -> None:
    doc = Document.from_image(gradient, "art")
    doc.add_layer(Layer(shapes, "Shapes", opacity=0.6, blend_mode="screen", locked=True, adjustments={"contrast": 25}))
    doc.add_text_layer("Hello", (3, 3), "#ff0000", None, 14)
    doc.set_layer_props(visible=False)
    doc.set_active(1)
    doc.set_selection(Selection.rectangle(doc.size, (1, 2, 10, 12)))
    flat_before = doc.composite(include_hidden=True)
    path = tmp_path / "art.pfp"
    pfio.save_project(doc, path)
    assert doc.path == str(path) and not doc.dirty and doc.name == "art"

    loaded = pfio.load_project(path)
    assert loaded.size == doc.size and len(loaded.layers) == 3
    assert loaded.active_index == 1 and loaded.path == str(path) and not loaded.dirty
    assert loaded.history.labels == ["Open Project"]
    shp = loaded.layers[1]
    assert (shp.name, shp.opacity, shp.blend_mode, shp.locked) == ("Shapes", 0.6, "screen", True)
    assert shp.adjustments["contrast"] == 25
    txt = loaded.layers[2]
    assert txt.kind == "text" and txt.meta["text"] == "Hello" and not txt.visible
    assert loaded.selection is not None and loaded.selection.bbox == (1, 2, 10, 12)
    assert pixel_diff(loaded.composite(include_hidden=True), flat_before) == 0
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["format_version"] == pfio.PROJECT_FORMAT_VERSION
        assert len(manifest["layers"]) == 3


def test_load_project_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pfp"
    bad.write_bytes(b"not a zip")
    with pytest.raises(DocumentError):
        pfio.load_project(bad)
    with zipfile.ZipFile(tmp_path / "nomanifest.pfp", "w") as zf:
        zf.writestr("x.txt", "hi")
    with pytest.raises(DocumentError):
        pfio.load_project(tmp_path / "nomanifest.pfp")
    with zipfile.ZipFile(tmp_path / "future.pfp", "w") as zf:
        zf.writestr("manifest.json", json.dumps({"format_version": 99, "width": 1, "height": 1, "layers": []}))
    with pytest.raises(pfio.UnsupportedFormat):
        pfio.load_project(tmp_path / "future.pfp")
    # layer size mismatch
    buf = io.BytesIO()
    Image.new("RGBA", (2, 2)).save(buf, "PNG")
    with zipfile.ZipFile(tmp_path / "mismatch.pfp", "w") as zf:
        zf.writestr(
            "manifest.json", json.dumps({"format_version": 1, "width": 4, "height": 4, "layers": [{"file": "layers/000.png", "name": "L"}]})
        )
        zf.writestr("layers/000.png", buf.getvalue())
    with pytest.raises(DocumentError):
        pfio.load_project(tmp_path / "mismatch.pfp")


# ---------------------------------------------------------------- presets --
def test_builtin_presets() -> None:
    assert "None" in PRESETS and PRESETS["None"] == {}
    for name in PRESETS:
        adj = preset_adjustments(name)
        assert set(adj) >= set(PRESETS[name])
    assert preset_adjustments("Vivid")["saturation"] == 40


def test_preset_store_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "cfg" / "presets.json"
    store = PresetStore(path)
    assert store.names()[: len(PRESETS)] == list(PRESETS)
    store.save("Mine", {"contrast": 15, "bogus": 1, "blur": 99})
    assert "Mine" in store.names()
    assert store.get("Mine")["contrast"] == 15 and store.get("Mine")["blur"] == 20 and "bogus" not in store.get("Mine")
    assert store.get("Vivid") == preset_adjustments("Vivid")
    again = PresetStore(path)
    assert again.get("Mine")["contrast"] == 15
    with pytest.raises(ValueError):
        store.save("Vivid", {})
    with pytest.raises(ValueError):
        store.delete("Vivid")
    store.delete("Mine")
    assert "Mine" not in PresetStore(path).names()
    store.delete("Missing")  # silently ignored
    path.write_text("{not json")
    assert PresetStore(path).user == {}
    memory = PresetStore(None)
    memory.save("X", {"hue": 5})
    assert memory.get("X")["hue"] == 5


# -------------------------------------------------------------------- cli --
def test_cli_parsers() -> None:
    assert parse_filter("Gaussian Blur:radius=3, foo=bar") == ("Gaussian Blur", {"radius": "3", "foo": "bar"})
    assert parse_filter("Invert") == ("Invert", {})
    assert parse_adjust("Contrast=12.7") == ("contrast", 12)
    assert parse_size("800x600") == (800, 600) and parse_size("800x") == (800, None) and parse_size("x600") == (None, 600)
    assert parse_box("1,2,3,4") == (1, 2, 4, 6)
    import argparse

    for fn, bad in (
        (parse_filter, "Nope"),
        (parse_filter, "Invert:oops"),
        (parse_adjust, "x=1"),
        (parse_adjust, "contrast=a"),
        (parse_size, "800"),
        (parse_size, "axb"),
        (parse_box, "1,2"),
        (parse_box, "a,b,c,d"),
    ):
        with pytest.raises(argparse.ArgumentTypeError):
            fn(bad)


def test_cli_lists_and_usage() -> None:
    out = io.StringIO()
    assert cli_main(["--list-filters"], out=out) == 0
    assert "Gaussian Blur" in out.getvalue()
    out = io.StringIO()
    assert cli_main(["--list-presets"], out=out) == 0
    assert "Vivid" in out.getvalue()
    err = io.StringIO()
    assert cli_main([], out=io.StringIO(), err=err) == 2
    assert "input" in err.getvalue()
    err = io.StringIO()
    assert cli_main(["only.png"], out=io.StringIO(), err=err) == 2
    err = io.StringIO()
    assert cli_main(["a.png", "b.png", "out.png"], out=io.StringIO(), err=err) == 2
    assert "--out-dir" in err.getvalue()


def test_cli_single_file_pipeline(tmp_path: Path, image_file: Path) -> None:
    out_path = tmp_path / "out.jpg"
    out = io.StringIO()
    code = cli_main(
        [
            str(image_file),
            str(out_path),
            "--preset",
            "Vivid",
            "--adjust",
            "temperature=20",
            "--resize",
            "32x",
            "--filter",
            "Unsharp Mask:radius=1",
            "--rotate",
            "90",
            "--flip",
            "h",
            "--quality",
            "80",
        ],
        out=out,
    )
    assert code == 0 and out_path.exists()
    with Image.open(out_path) as im:
        assert im.size == (32, 43)  # 64x48 -> rotated to 48x64 -> width 32 keeps the aspect ratio
    assert "->" in out.getvalue()


def test_cli_batch_info_and_errors(tmp_path: Path, image_file: Path, gradient: Image.Image) -> None:
    second = tmp_path / "second.png"
    gradient.resize((20, 20)).save(second)
    out = io.StringIO()
    code = cli_main(
        [str(image_file), str(second), "--out-dir", str(tmp_path / "batch"), "--scale", "0.5", "--format", "webp", "--crop", "0,0,16,16"],
        out=out,
    )
    assert code == 0
    assert (tmp_path / "batch" / "gradient.webp").exists() and (tmp_path / "batch" / "second.webp").exists()
    with Image.open(tmp_path / "batch" / "gradient.webp") as im:
        assert im.size == (8, 8)
    out = io.StringIO()
    assert cli_main([str(image_file), "--info"], out=out) == 0
    assert "64x48" in out.getvalue()
    err = io.StringIO()
    assert cli_main([str(tmp_path / "missing.png"), str(tmp_path / "x.png")], out=io.StringIO(), err=err) == 1
    assert "cannot open" in err.getvalue()
    err = io.StringIO()
    assert cli_main([str(image_file), str(tmp_path / "x.png"), "--crop", "0,0,0,0"], out=io.StringIO(), err=err) == 1
    err = io.StringIO()
    assert cli_main([str(image_file), str(tmp_path / "x.bogus")], out=io.StringIO(), err=err) == 1
    assert "cannot save" in err.getvalue()
    # project output, arbitrary rotation, trim
    out = io.StringIO()
    assert cli_main([str(image_file), str(tmp_path / "p.pfp"), "--rotate", "30", "--trim"], out=out) == 0
    assert pfio.load_project(tmp_path / "p.pfp").size[0] > 64


def test_save_project_without_recording(tmp_path: Path, gradient: Image.Image) -> None:
    doc = Document.from_image(gradient, "pic")
    doc.fill("#00ff00")
    assert doc.dirty and doc.path is None
    pfio.save_project(doc, tmp_path / "auto.pfp", record=False)
    assert (tmp_path / "auto.pfp").exists() and doc.dirty and doc.path is None and doc.name == "pic"
    assert pixel(pfio.load_project(tmp_path / "auto.pfp").active_layer.image, 1, 1) == (0, 255, 0, 255)
