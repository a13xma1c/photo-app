"""Brush presets: tips, dynamics, the library and stroke previews."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixelforge.core import brushes as br
from pixelforge.core.painting import Stroke


def test_preset_normalisation_and_serialisation() -> None:
    wild = br.BrushPreset("X", "sideways", 9000, 150, 0, 725.0, 1, -5, 300, 300, 0, 99)
    n = wild.normalized()
    assert (n.tip, n.size, n.hardness, n.spacing, n.angle, n.roundness) == ("round", 500, 100, 1, 5.0, 5)
    assert (n.scatter, n.size_jitter, n.opacity_jitter, n.flow, n.count) == (0, 100, 100, 1, 16)
    data = n.to_dict()
    assert "builtin" not in data and br.BrushPreset.from_dict({**data, "bogus": 1}) == n
    assert "px" in n.description() and "flow" in n.description()
    assert br.BrushPreset("S", "spatter", scatter=10).description().startswith("spatter")


def test_tip_shapes() -> None:
    round_tip = br.BrushPreset("r", "round", 21, 100).stamp()
    square = br.BrushPreset("s", "square", 21, 100).stamp()
    assert round_tip.shape == square.shape == (21, 21)
    assert round_tip[10, 10] == 1 and round_tip[0, 0] == 0 and square[0, 0] == 1  # a square fills its corners
    soft = br.BrushPreset("soft", "round", 21, 0).stamp()
    assert 0 < soft[10, 2] < soft[10, 6] < 1
    chalk = br.BrushPreset("c", "chalk", 31, 80).stamp()
    assert 0 < chalk.mean() < round_tip.mean() and chalk.max() > 0.9
    spatter = br.BrushPreset("sp", "spatter", 41).stamp()
    assert 0 < spatter.mean() < 0.5 and spatter.max() == 1
    flat = br.BrushPreset("f", "round", 40, 100, roundness=25).stamp()
    assert flat[20, 5] == 1 and flat[5, 20] == 0  # squashed vertically
    rotated = br.BrushPreset("f", "round", 40, 100, roundness=25, angle=90).stamp()
    assert rotated[5, 20] > 0.5 and rotated[20, 5] < 0.5
    assert br.BrushPreset("one", "round", 1).stamp().shape == (1, 1)
    assert br.BrushPreset("sized", "round", 10).stamp(size=4).shape == (4, 4)


def test_custom_tip_from_image() -> None:
    img = Image.new("RGBA", (50, 30), (0, 0, 0, 0))
    img.paste((255, 255, 255, 255), (10, 5, 40, 25))  # a white rectangle
    texture = br.texture_from_image(img, size=32)
    preset = br.BrushPreset("custom", "custom", 32, texture=texture)
    tip = preset.tip_image()
    assert tip is not None and tip.mode == "L" and tip.size == (32, 32)
    stamp = preset.stamp()
    assert stamp[16, 16] > 0.9 and stamp[2, 16] < 0.2  # the rectangle is wider than tall
    assert br.BrushPreset("broken", "custom", 10, texture="!!notbase64").tip_image() is None
    assert br.BrushPreset("broken", "custom", 10, texture="!!notbase64").stamp().shape == (10, 10)  # falls back to round
    assert br.BrushPreset("plain").tip_image() is None


def test_stroke_dynamics() -> None:
    size = (200, 100)
    plain = Stroke(size, preset=br.BrushPreset("p", "round", 20, 100, 10), seed=1)
    plain.add_point(20, 50)
    plain.add_point(180, 50)
    assert plain.coverage[50, 100] == 1 and plain.coverage[50, 5] == 0 and float(np.max(plain.coverage)) == 1
    # flow < 100 builds up: a single dab is faint, overlapping dabs darken
    airbrush = Stroke(size, preset=br.BrushPreset("a", "round", 20, 100, 5, flow=20), seed=1)
    airbrush.add_point(100, 50)
    assert 0.19 < airbrush.coverage[50, 100] < 0.21
    airbrush.add_point(140, 50)
    assert airbrush.coverage[50, 120] > 0.5
    # scatter wanders off the path, opacity jitter varies, count multiplies the dabs
    scattered = Stroke(size, preset=br.BrushPreset("s", "round", 6, 100, 50, scatter=200, count=3), seed=3)
    for x in range(20, 180, 4):
        scattered.add_point(x, 50)
    rows = np.flatnonzero(scattered.coverage.any(axis=1))
    assert int(np.min(rows)) < 40 and int(np.max(rows)) > 60
    jitter = Stroke(size, preset=br.BrushPreset("j", "round", 10, 100, 100, opacity_jitter=80, size_jitter=80), seed=5)
    for x in range(20, 180, 2):
        jitter.add_point(x, 50)
    peaks = {round(float(jitter.coverage[50, x]), 2) for x in range(20, 180, 10)}
    assert len(peaks) > 3 and float(np.max(jitter.coverage)) <= 1
    # the same seed reproduces the same stroke; the legacy signature still works
    again = Stroke(size, preset=br.BrushPreset("s", "round", 6, 100, 50, scatter=200, count=3), seed=3)
    for x in range(20, 180, 4):
        again.add_point(x, 50)
    assert np.array_equal(again.coverage, scattered.coverage)
    legacy = Stroke(size, 8, 1.0)
    assert legacy.preset.size == 8 and legacy.preset.hardness == 100 and legacy.preset.spacing == 15


def test_library_and_previews(tmp_path: Path) -> None:
    lib = br.BrushLibrary(tmp_path / "brushes.json")
    assert lib.names()[: len(br.BUILTIN_PRESETS)] == [p.name for p in br.BUILTIN_PRESETS]
    assert lib.get("Airbrush").flow == 12 and lib.is_builtin("Airbrush")
    with pytest.raises(ValueError):
        lib.save(br.BrushPreset("Airbrush"))
    with pytest.raises(ValueError):
        lib.delete("Airbrush")
    saved = lib.save(br.BrushPreset("My Splat", "spatter", 33, scatter=40, builtin=True))
    assert not saved.builtin and lib.get("My Splat").size == 33 and "My Splat" in lib.names()
    reloaded = br.BrushLibrary(tmp_path / "brushes.json")
    assert reloaded.get("My Splat") == saved and reloaded.presets()[-1].name == "My Splat"
    reloaded.delete("My Splat")
    assert "My Splat" not in br.BrushLibrary(tmp_path / "brushes.json").names()
    (tmp_path / "brushes.json").write_text("not json")
    assert br.BrushLibrary(tmp_path / "brushes.json").user == {}
    for preset in br.BUILTIN_PRESETS:
        cov = br.stroke_preview(preset, (120, 36))
        assert cov.shape == (36, 120) and cov.max() > 0.3
