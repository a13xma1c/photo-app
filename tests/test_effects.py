"""Layer effects: normalisation, dilation / erosion and rendering through the document."""

from pathlib import Path

import numpy as np
from PIL import Image

from pixelforge.core import effects as fx
from pixelforge.core import io as pfio
from pixelforge.core.document import Document
from tests.helpers import pixel


def boxed_doc() -> Document:
    doc = Document.new(200, 200, "#404040")
    box = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    box.paste((255, 0, 0, 255), (20, 20, 60, 60))
    doc.add_image_layer(box, "Box", (50, 50))  # red square at 70..110
    return doc


def test_normalise_and_padding() -> None:
    assert fx.is_empty(None) and fx.is_empty(fx.default_effects()) and fx.padding(None) == 0
    norm = fx.normalize_effects({"shadow": {"enabled": 1, "dx": 900, "color": "#ff0000", "bogus": 1}, "nothing": {}})
    assert norm["shadow"]["enabled"] is True and norm["shadow"]["dx"] == 500 and norm["shadow"]["color"] == [255, 0, 0]
    assert "bogus" not in norm["shadow"] and "nothing" not in norm and norm["glow"]["enabled"] is False
    assert fx.normalize_effects({"stroke": {"position": "sideways", "opacity": 250}})["stroke"] == {
        **fx.DEFAULTS["stroke"],
        "position": "outside",
        "opacity": 100,
    }
    assert fx.padding({"shadow": {"enabled": True, "dx": 10, "dy": 0, "blur": 2}}) == 10 + 6 + 2
    assert fx.padding({"glow": {"enabled": True, "size": 10}}, scale=2.0) == 62


def test_dilate_and_erode() -> None:
    alpha = Image.new("L", (60, 60), 0)
    alpha.paste(255, (20, 20, 40, 40))
    grown = fx.dilate(alpha, 5)
    assert grown[30, 30] == 1.0 and grown[30, 16] > 0.9 and grown[30, 12] < 0.1  # ~5 px out
    shrunk = fx.erode(alpha, 5)
    assert shrunk[30, 30] == 1.0 and shrunk[30, 23] < 0.1 and shrunk[30, 28] > 0.9
    assert np.array_equal(fx.dilate(alpha, 0), np.asarray(alpha) / 255.0)


def test_effects_render_in_composite() -> None:
    doc = boxed_doc()
    plain = pixel(doc.flatten(), 120, 120)
    doc.set_layer_meta(effects={"shadow": {"enabled": True, "dx": 12, "dy": 12, "blur": 0, "opacity": 100}})
    assert doc.history.labels[-1] == "Layer Settings"
    assert pixel(doc.flatten(), 115, 115) == (0, 0, 0, 255) and pixel(doc.flatten(), 90, 90) == (255, 0, 0, 255)
    doc.set_layer_meta(effects={"stroke": {"enabled": True, "size": 4, "color": [0, 255, 0], "position": "outside"}})
    assert pixel(doc.flatten(), 68, 90) == (0, 255, 0, 255) and pixel(doc.flatten(), 62, 90) == plain
    doc.set_layer_meta(effects={"stroke": {"enabled": True, "size": 4, "color": [0, 255, 0], "position": "inside"}})
    assert pixel(doc.flatten(), 71, 90) == (0, 255, 0, 255) and pixel(doc.flatten(), 68, 90) == plain
    doc.set_layer_meta(effects={"stroke": {"enabled": True, "size": 4, "color": [0, 255, 0], "position": "center"}})
    assert pixel(doc.flatten(), 69, 90)[1] > 200 and pixel(doc.flatten(), 71, 90)[1] > 200
    doc.set_layer_meta(effects={"glow": {"enabled": True, "size": 8, "color": [255, 255, 0], "opacity": 100}})
    r, g, b, a = pixel(doc.flatten(), 66, 90)
    assert r > 150 and g > 150 and b < 120  # yellowish glow just outside the box
    # zoomed-out and partial renders include the effect too; "effects" shows in the layer key
    small = doc.composite(0.5)
    assert pixel(small, 33, 45)[0] > 100
    part = doc.composite(1.0, (60, 85, 70, 95))
    assert pixel(part, 6, 5)[0] > 150
    # removing the effects restores the plain look; empty effects are dropped from meta
    doc.set_layer_meta(effects=None)
    assert "effects" not in doc.active_layer.meta and pixel(doc.flatten(), 68, 90) == plain


def test_effects_with_mask_and_roundtrip(tmp_path: Path) -> None:
    doc = boxed_doc()
    doc.add_mask()
    doc.layers[1] = doc.layers[1].copy(mask=Image.new("L", doc.size, 0))  # everything hidden
    doc.set_layer_meta(effects={"stroke": {"enabled": True, "size": 4, "color": [0, 255, 0]}})
    assert pixel(doc.flatten(), 68, 90) == pixel(doc.flatten(), 10, 10)  # no pixels → no stroke
    doc.layers[1] = doc.layers[1].copy(mask=Image.new("L", doc.size, 255))
    doc.touch()
    assert pixel(doc.flatten(), 68, 90) == (0, 255, 0, 255)
    path = tmp_path / "fx.pfp"
    pfio.save_project(doc, path)
    loaded = pfio.load_project(path)
    assert loaded.layers[1].meta["effects"]["stroke"]["enabled"] and pixel(loaded.flatten(), 68, 90) == (0, 255, 0, 255)
