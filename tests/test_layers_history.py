import pytest
from PIL import Image

from pixelforge.core.history import History
from pixelforge.core.layers import Layer, ensure_rgba
from tests.helpers import pixel


# ------------------------------------------------------------------ Layer --
def test_layer_converts_to_rgba_and_validates(gradient: Image.Image) -> None:
    layer = Layer(gradient, "L")
    assert layer.image.mode == "RGBA"
    assert layer.size == gradient.size
    assert layer.opacity == 1.0
    assert not layer.has_adjustments
    with pytest.raises(ValueError):
        Layer(gradient, blend_mode="bogus")
    assert Layer(gradient, opacity=3.0).opacity == 1.0
    assert Layer(gradient, opacity=-1).opacity == 0.0


def test_layer_copy_shares_image_but_not_dicts(gradient: Image.Image) -> None:
    layer = Layer(gradient, "A", adjustments={"contrast": 10})
    clone = layer.copy(name="B")
    assert clone.image is layer.image
    assert clone.id == layer.id
    assert clone.name == "B" and layer.name == "A"
    clone.adjustments["contrast"] = 50
    assert layer.adjustments["contrast"] == 10
    assert clone.has_adjustments


def test_layer_duplicate_is_independent(gradient: Image.Image) -> None:
    layer = Layer(gradient, "A")
    dup = layer.duplicate()
    assert dup.id != layer.id
    assert dup.image is not layer.image
    assert dup.name == "A copy"
    assert dup.image.tobytes() == layer.image.tobytes()


def test_layer_manifest_roundtrip(gradient: Image.Image) -> None:
    layer = Layer(
        gradient,
        "Text",
        visible=False,
        opacity=0.4,
        blend_mode="screen",
        locked=True,
        adjustments={"hue": 20},
        kind="text",
        meta={"text": "hi"},
    )
    data = layer.to_manifest()
    restored = Layer.from_manifest(data, layer.image)
    for field in ("name", "visible", "opacity", "blend_mode", "locked", "kind", "meta"):
        assert getattr(restored, field) == getattr(layer, field)
    assert restored.adjustments["hue"] == 20


def test_blank_and_ensure_rgba() -> None:
    blank = Layer.blank((10, 5), "B", (1, 2, 3, 4))
    assert pixel(blank.image, 0, 0) == (1, 2, 3, 4)
    rgb = Image.new("RGB", (2, 2))
    assert ensure_rgba(rgb).mode == "RGBA"
    rgba = Image.new("RGBA", (2, 2))
    assert ensure_rgba(rgba) is rgba


# ---------------------------------------------------------------- History --
def test_history_push_undo_redo() -> None:
    h = History[int](capacity=10)
    assert h.current is None and not h.can_undo and not h.can_redo
    h.push("a", 1)
    h.push("b", 2)
    h.push("c", 3)
    assert h.labels == ["a", "b", "c"] and h.index == 2 and h.current == 3
    assert h.undo() == 2 and h.can_redo
    assert h.undo() == 1 and not h.can_undo
    assert h.undo() is None
    assert h.redo() == 2
    h.push("d", 4)  # discards redo branch
    assert h.labels == ["a", "b", "d"] and not h.can_redo
    assert h.redo() is None


def test_history_capacity_and_go_to() -> None:
    h = History[int](capacity=3)
    for i in range(5):
        h.push(str(i), i)
    assert h.labels == ["2", "3", "4"] and len(h) == 3 and h.index == 2
    assert h.go_to(0) == 2
    with pytest.raises(IndexError):
        h.go_to(7)
    h.replace_current("x", 99)
    assert h.current == 99 and h.labels[0] == "x"
    h.clear()
    assert len(h) == 0 and h.index == -1
    h.replace_current("first", 5)  # on empty history behaves like push
    assert h.labels == ["first"]
    with pytest.raises(ValueError):
        History(capacity=0)
    assert [s.label for s in h.states] == ["first"]
