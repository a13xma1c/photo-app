"""Layer groups: membership, rendering, reordering in and out of groups, merge / ungroup / delete and persistence."""

from pathlib import Path

import pytest
from PIL import Image

from pixelforge.core import io as pfio
from pixelforge.core.document import Document, DocumentError
from pixelforge.core.layers import Layer
from tests.helpers import pixel


def solid(size: tuple[int, int], color: tuple[int, int, int, int], name: str) -> Layer:
    return Layer(Image.new("RGBA", size, color), name)


def names(doc: Document) -> list[str]:
    return [layer.name for layer in doc.layers]


def grouped_doc() -> Document:
    """Background, then a group holding Red and Green, then Blue on top."""
    doc = Document.new(20, 20, "white")
    doc.add_layer(solid(doc.size, (255, 0, 0, 255), "Red"))
    doc.group_active("Pair")
    doc.set_active(1)  # Red
    doc.add_layer(solid(doc.size, (0, 255, 0, 128), "Green"))  # joins the group (active layer is a member)
    doc.set_active(doc.layer_index(next(lyr.id for lyr in doc.layers if lyr.kind == "group")) or 0)
    doc.add_layer(solid(doc.size, (0, 0, 255, 255), "Blue"))
    doc.set_layer_props(visible=False)
    return doc


def test_group_structure_and_rendering() -> None:
    doc = grouped_doc()
    assert names(doc) == ["Background", "Red", "Green", "Pair", "Blue"]
    header = 3
    assert doc.layers[header].kind == "group" and doc.group_members(header) == [1, 2]
    assert doc.group_of(1) == header and doc.group_of(2) == header and doc.group_of(4) is None
    assert doc._block(header) == (1, 4) and doc._block(1) == (1, 2)
    assert doc.effective_visible(1)
    # the group renders like its members would
    assert pixel(doc.flatten(), 5, 5) == (127, 128, 0, 255)  # green 50 % over red
    # hiding the header hides the members; group opacity and mask act on the whole group
    doc.set_layer_props(header, visible=False)
    assert not doc.effective_visible(1) and pixel(doc.flatten(), 5, 5) == (255, 255, 255, 255)
    doc.set_layer_props(header, visible=True, opacity=0.5)
    r, g, b, a = pixel(doc.flatten(), 5, 5)
    assert 180 < r < 200 and 180 < g < 200 and b > 120  # half-way to white
    doc.set_layer_props(header, opacity=1.0)
    doc.set_active(header)
    doc.add_mask(mode="hide")
    assert pixel(doc.flatten(), 5, 5) == (255, 255, 255, 255)
    doc.remove_mask()
    doc.set_layer_meta(header, effects={"stroke": {"enabled": True, "size": 2, "color": [0, 0, 0]}})
    assert pixel(doc.flatten(), 5, 5) == (127, 128, 0, 255)  # a stroke needs an edge; a full-canvas group has none inside
    # groups have no pixels of their own
    with pytest.raises(DocumentError):
        doc.fill("#000000")
    with pytest.raises(DocumentError):
        doc.merge_down(1)  # Red's neighbour below is outside the group
    assert doc.composite(include_hidden=True).getbbox() is not None


def test_raise_and_lower_cross_group_boundaries() -> None:
    doc = grouped_doc()
    blue = 4
    # lowering Blue (above the group) steps into the group as its top member
    doc.set_active(blue)
    doc.lower_layer()
    assert names(doc) == ["Background", "Red", "Green", "Blue", "Pair"] and doc.group_of(3) == 4 and doc.active_index == 3
    doc.lower_layer()
    assert names(doc) == ["Background", "Red", "Blue", "Green", "Pair"] and doc.group_of(2) == 4
    doc.lower_layer()
    assert names(doc) == ["Background", "Blue", "Red", "Green", "Pair"] and doc.group_of(1) == 4
    doc.lower_layer()  # bottom of the group: steps out below it, same position
    assert names(doc) == ["Background", "Blue", "Red", "Green", "Pair"] and doc.group_of(1) is None
    doc.lower_layer()  # plain swap with the background
    assert names(doc) == ["Blue", "Background", "Red", "Green", "Pair"] and doc.active_index == 0
    doc.lower_layer()  # nothing below
    assert doc.active_index == 0
    # raising: past the background, into the group from below, through it and out above the header
    doc.raise_layer()
    assert names(doc) == ["Background", "Blue", "Red", "Green", "Pair"] and doc.group_of(1) is None
    doc.raise_layer()
    assert names(doc) == ["Background", "Blue", "Red", "Green", "Pair"] and doc.group_of(1) == 4  # joined at the bottom
    doc.raise_layer()
    doc.raise_layer()
    assert names(doc) == ["Background", "Red", "Green", "Blue", "Pair"] and doc.group_of(3) == 4
    doc.raise_layer()
    assert names(doc) == ["Background", "Red", "Green", "Pair", "Blue"] and doc.group_of(4) is None and doc.active_index == 4
    doc.raise_layer()  # top already
    assert doc.active_index == 4
    # moving the header moves the whole block
    doc.set_active(3)
    doc.raise_layer()
    assert names(doc) == ["Background", "Blue", "Red", "Green", "Pair"] and doc.active_index == 4
    doc.lower_layer()
    assert names(doc) == ["Background", "Red", "Green", "Pair", "Blue"]
    doc.lower_layer()
    assert names(doc) == ["Red", "Green", "Pair", "Background", "Blue"] and doc.active_index == 2
    assert doc.undo() and names(doc) == ["Background", "Red", "Green", "Pair", "Blue"]


def test_group_delete_duplicate_merge_ungroup(tmp_path: Path) -> None:
    doc = grouped_doc()
    header = 3
    doc.set_active(header)
    doc.duplicate_layer()
    assert names(doc) == ["Background", "Red", "Green", "Pair", "Red", "Green", "Pair copy", "Blue"]
    assert doc.group_of(4) == 6 and doc.group_of(5) == 6 and doc.active_index == 6
    doc.remove_layer()
    assert names(doc) == ["Background", "Red", "Green", "Pair", "Blue"] and doc.history.labels[-1] == "Delete Group 'Pair copy'"
    # project round trip keeps the membership (ids are re-issued on load)
    path = tmp_path / "groups.pfp"
    pfio.save_project(doc, path)
    loaded = pfio.load_project(path)
    assert names(loaded) == names(doc) and loaded.group_of(1) == 3 and loaded.group_of(2) == 3 and loaded.group_of(4) is None
    assert pixel(loaded.flatten(), 5, 5) == pixel(doc.flatten(), 5, 5)
    # merge the group into one layer
    expected = pixel(doc.flatten(), 5, 5)
    doc.set_active(header)
    doc.merge_down()
    assert names(doc) == ["Background", "Pair", "Blue"] and doc.layers[1].kind == "pixel" and pixel(doc.flatten(), 5, 5) == expected
    assert doc.undo()
    # ungroup from the header or from a member
    doc.set_active(1)
    doc.ungroup()
    assert names(doc) == ["Background", "Red", "Green", "Blue"] and all(doc.group_of(i) is None for i in range(4))
    with pytest.raises(DocumentError):
        doc.ungroup(1)
    # nested groups: ungrouping the inner one hands its members to the outer one
    doc.set_active(1)
    doc.group_active("Inner")
    doc.group_active("Outer")
    assert names(doc) == ["Background", "Red", "Inner", "Outer", "Green", "Blue"]
    assert doc.group_of(1) == 2 and doc.group_of(2) == 3 and doc._block(3) == (1, 4)
    assert pixel(doc.flatten(), 5, 5) == (127, 128, 0, 255)  # nested groups render like the flat stack
    doc.set_layer_props(3, visible=False)
    assert not doc.effective_visible(1)
    doc.set_layer_props(3, visible=True)
    doc.ungroup(2)
    assert names(doc) == ["Background", "Red", "Outer", "Green", "Blue"] and doc.group_of(1) == 2
    doc.set_active(1)
    doc.remove_layer()
    doc.remove_layer()  # the now-empty group header
    assert names(doc) == ["Background", "Green", "Blue"]
    doc.set_active(0)
    with pytest.raises(DocumentError):
        Document.new(5, 5).remove_layer()


def test_merge_visible_with_groups() -> None:
    doc = grouped_doc()
    doc.merge_visible()
    assert [layer.name for layer in doc.layers] == ["Merged", "Blue"] and not doc.layers[1].visible
    assert pixel(doc.flatten(), 5, 5) == (127, 128, 0, 255)
