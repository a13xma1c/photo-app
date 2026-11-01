"""The :class:`Document`: a stack of layers, a selection and an undo history.

The document is the single model object the GUI talks to.  All edits go
through methods here which (a) never mutate layer images in place, (b) record
a named history state via :meth:`Document.commit`, and (c) invalidate the
render cache.

Rendering happens in :meth:`Document.composite`, which can render any
rectangular *view box* at any *scale* so the GUI only ever pays for the pixels
that are actually visible.
"""

from __future__ import annotations

import json
import math
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageOps

from . import painting
from .adjustments import apply_adjustments, default_adjustments, is_neutral, normalize_adjustments
from .blend import composite_premultiplied, unpremultiply
from .color import parse_color
from .colorspace import Metadata, srgb_profile, to_srgb
from .curves import apply_curves
from .effects import apply_effects
from .effects import is_empty as no_effects
from .effects import padding as effects_padding
from .filters import apply_filter
from .history import History
from .layers import Layer, ensure_rgba, mask_to_rgba, rgba_to_mask
from .selection import Selection
from .three_d import Scene, extrusion_scene
from .three_d import render as render_scene

Box = tuple[int, int, int, int]
ANCHORS: tuple[str, ...] = ("nw", "n", "ne", "w", "center", "e", "sw", "s", "se")
MAX_DIMENSION = 20000
SMART_KINDS: tuple[str, ...] = ("3d", "text")  # layers re-rendered from their meta: filters on them stay live
HISTORY_BUDGET_BYTES = 1_500_000_000  # undo states are dropped (oldest first) once they hold more than this
HISTORY_MIN_STEPS, HISTORY_MAX_STEPS = 8, 100
PATCH_MAX_FRACTION = 0.6  # an older image is kept as a patch when the changed area is at most this share of it


class DocumentError(Exception):
    """Raised for invalid edits (bad sizes, locked layers, ...)."""


_PLACEHOLDER = Image.new("RGBA", (1, 1), (0, 0, 0, 0))  # stands in for images stored as patches


@dataclass
class Patch:
    """An older image kept as just the pixels that differ from the same layer's image in a *newer* state."""

    newer: Snapshot
    layer_id: int
    box: Box | None  # None: identical to the newer image
    pixels: Image.Image | None
    mask: bool = False  # patch of the layer mask rather than its pixels


@dataclass(frozen=True)
class Snapshot:
    """One history state: layer records plus (optionally) patches replacing full images to save memory."""

    width: int
    height: int
    layers: tuple[Layer, ...]
    active_index: int
    selection: Selection | None
    patches: dict[tuple[int, bool], Patch] = field(default_factory=dict)  # (layer id, is_mask) → patch

    def _layer(self, layer_id: int) -> Layer | None:
        return next((layer for layer in self.layers if layer.id == layer_id), None)

    def image_of(self, layer_id: int, mask: bool = False) -> Image.Image | None:
        """The full image (or mask) of a layer in this state, rebuilding it from patches when needed."""
        layer = self._layer(layer_id)
        if layer is None:
            return None
        patch = self.patches.get((layer_id, mask))
        if patch is None:
            return layer.mask if mask else layer.image
        base = patch.newer.image_of(layer_id, mask)
        if base is None:
            return None
        if patch.box is None or patch.pixels is None:
            return base
        rebuilt = base.copy()
        rebuilt.paste(patch.pixels, patch.box[:2])
        return rebuilt

    def materialize(self) -> None:
        """Resolve every patch back into a full image (before the state it refers to goes away)."""
        for (layer_id, mask), _patch in list(self.patches.items()):
            image = self.image_of(layer_id, mask)
            layer = self._layer(layer_id)
            if layer is not None:
                if mask:
                    layer.mask = image
                else:
                    assert image is not None
                    layer.image = image
        self.patches.clear()

    def nbytes(self, seen: set[int]) -> int:
        """Memory held by images this state references that were not counted yet (tracked through *seen* ids)."""
        total = 0
        for layer in self.layers:
            for img in (layer.image, layer.mask):
                if img is not None and img is not _PLACEHOLDER and id(img) not in seen:
                    seen.add(id(img))
                    total += img.width * img.height * len(img.getbands())
        for patch in self.patches.values():
            if patch.pixels is not None and id(patch.pixels) not in seen:
                seen.add(id(patch.pixels))
                total += patch.pixels.width * patch.pixels.height * len(patch.pixels.getbands())
        return total


class _RenderCache:
    """Small LRU cache for rendered layers / composites, bounded by entry count and bytes (thread-safe)."""

    def __init__(self, capacity: int = 32, max_bytes: int = 256 * 1024 * 1024) -> None:
        self.capacity = capacity
        self.max_bytes = max_bytes
        self._data: OrderedDict = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    @staticmethod
    def _size(img: Image.Image) -> int:
        return img.width * img.height * len(img.getbands())

    def get(self, key):
        with self._lock:
            value = self._data.get(key)
            if value is not None:
                self._data.move_to_end(key)
            return value

    def put(self, key, value: Image.Image) -> None:
        with self._lock:
            if key in self._data:
                self._bytes -= self._size(self._data[key])
            self._data[key] = value
            self._data.move_to_end(key)
            self._bytes += self._size(value)
            while self._data and (len(self._data) > self.capacity or self._bytes > self.max_bytes):
                _, evicted = self._data.popitem(last=False)
                self._bytes -= self._size(evicted)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._bytes = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


def inscribed_box(width: int, height: int, angle: float) -> Box:
    """Largest upright rectangle fully inside a ``width × height`` picture rotated by *angle* degrees about its centre.

    Returned as an integer crop box in the (unexpanded) rotated canvas.
    """
    a = abs(math.radians(angle)) % math.pi
    if a > math.pi / 2:
        a = math.pi - a
    if a < 1e-9:
        return 0, 0, width, height
    sin_a, cos_a = math.sin(a), math.cos(a)
    long_side, short_side = max(width, height), min(width, height)
    if short_side <= 2 * sin_a * cos_a * long_side or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * short_side
        wr, hr = (x / sin_a, x / cos_a) if width >= height else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (width * cos_a - height * sin_a) / cos_2a, (height * cos_a - width * sin_a) / cos_2a
    wr, hr = max(1, int(math.floor(wr)) - 1), max(1, int(math.floor(hr)) - 1)
    left, top = (width - wr) // 2, (height - hr) // 2
    return left, top, left + wr, top + hr


def _diff_bbox(a: Image.Image, b: Image.Image) -> Box | None:
    """Bounding box of the pixels where *a* and *b* differ in any channel (None when identical)."""
    diff = np.asarray(ImageChops.difference(a, b))
    changed = diff.any(axis=2) if diff.ndim == 3 else diff.astype(bool)
    rows = np.flatnonzero(changed.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(changed.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _check_size(width: int, height: int) -> tuple[int, int]:
    width, height = int(width), int(height)
    if width < 1 or height < 1:
        raise DocumentError("image dimensions must be at least 1x1")
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise DocumentError(f"image dimensions may not exceed {MAX_DIMENSION}px")
    return width, height


class Document:
    """A multi-layer image with selection, history and rendering."""

    def __init__(
        self, width: int, height: int, layers: Iterable[Layer] | None = None, name: str = "Untitled", initial_label: str = "Open"
    ) -> None:
        self.width, self.height = _check_size(width, height)
        self.layers: list[Layer] = list(layers or [])
        for layer in self.layers:
            if layer.size != self.size:
                raise DocumentError("layer size must match document size")
        if not self.layers:
            self.layers.append(Layer.blank(self.size, "Background", (255, 255, 255, 255), kind="background"))
        self.active_index = len(self.layers) - 1
        self.selection: Selection | None = None
        self.mask_editing = False  # paint on the active layer's mask instead of its pixels
        self.name = name
        self.path: str | None = None  # project file path (.pfp)
        self.source_path: str | None = None  # image the document was opened from
        self.metadata = Metadata()  # ICC profile / EXIF / dpi carried from the opened picture to exports
        self.history: History[Snapshot] = History()
        self.snapshots: list[tuple[str, Snapshot]] = []  # named bookmarks, independent of undo
        self._cache = _RenderCache()
        self._saved_index = 0
        self.commit(initial_label)

    # ------------------------------------------------------------ factories --
    @classmethod
    def new(cls, width: int, height: int, background: str | tuple = "white", name: str = "Untitled") -> Document:
        if background == "transparent":
            color: tuple = (0, 0, 0, 0)
        elif background == "white":
            color = (255, 255, 255, 255)
        elif background == "black":
            color = (0, 0, 0, 255)
        else:
            color = parse_color(background)
        width, height = _check_size(width, height)
        layer = Layer.blank((width, height), "Background", color, kind="background")
        return cls(width, height, [layer], name=name, initial_label="New")

    @classmethod
    def from_image(cls, image: Image.Image, name: str = "Untitled", color_manage: bool = True) -> Document:
        """A document from a picture: upright (EXIF), converted to sRGB when it carries another profile, metadata kept."""
        metadata = Metadata.from_image(image)
        image = ImageOps.exif_transpose(image) or image
        rgba = ensure_rgba(image)
        if color_manage:
            rgba, converted = to_srgb(rgba, metadata.icc_profile)
            if converted or metadata.icc_profile:
                metadata.icc_profile = srgb_profile()  # we edit and export in sRGB
        _check_size(*rgba.size)
        doc = cls(rgba.width, rgba.height, [Layer(rgba, "Background", kind="background")], name=name)
        doc.metadata = metadata
        return doc

    # -------------------------------------------------------------- queries --
    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def active_layer(self) -> Layer:
        return self.layers[self.active_index]

    @property
    def dirty(self) -> bool:
        return self.history.index != self._saved_index

    @property
    def can_undo(self) -> bool:
        return self.history.can_undo

    @property
    def can_redo(self) -> bool:
        return self.history.can_redo

    def layer_index(self, layer_id: int) -> int | None:
        for i, layer in enumerate(self.layers):
            if layer.id == layer_id:
                return i
        return None

    def selection_mask(self) -> Image.Image | None:
        """The selection mask image or None when everything is editable."""
        return self.selection.mask if self.selection is not None else None

    @property
    def editing_mask(self) -> bool:
        """True while pixel edits go to the active layer's mask (always, for adjustment layers)."""
        layer = self.active_layer
        return layer.mask is not None and (self.mask_editing or layer.kind == "adjustment")

    def edit_image(self, index: int | None = None) -> Image.Image:
        """The RGBA picture that painting tools edit: the layer pixels, or its mask shown as grey."""
        index = self._check_index(index)
        layer = self.layers[index]
        if index == self.active_index and self.editing_mask:
            assert layer.mask is not None
            return mask_to_rgba(layer.mask)
        return layer.image

    def set_mask_editing(self, editing: bool) -> None:
        """Choose whether the active layer's pixels or its mask receive edits (not recorded in history)."""
        layer = self.active_layer
        self.mask_editing = layer.mask is not None and (bool(editing) or layer.kind == "adjustment")

    def mark_saved(self) -> None:
        self._saved_index = self.history.index

    # -------------------------------------------------------------- history --
    def _snapshot(self) -> Snapshot:
        return Snapshot(self.width, self.height, tuple(layer.copy() for layer in self.layers), self.active_index, self.selection)

    def _restore(self, snap: Snapshot) -> None:
        self.width, self.height = snap.width, snap.height
        self.layers = [layer.copy(image=snap.image_of(layer.id), mask=snap.image_of(layer.id, mask=True)) for layer in snap.layers]
        self.active_index = min(snap.active_index, len(self.layers) - 1)
        self.selection = snap.selection
        self._cache.clear()

    def history_capacity(self) -> int:
        """Upper bound on undo steps for the current size; the byte budget usually trims earlier for big pictures."""
        bytes_per_state = max(1, self.width * self.height * 4 * max(1, len(self.layers)))
        return max(HISTORY_MIN_STEPS, min(HISTORY_MAX_STEPS, 4 * HISTORY_BUDGET_BYTES // bytes_per_state))

    def history_bytes(self) -> int:
        """Approximate memory held by the undo history (distinct images and patches)."""
        seen: set[int] = set()
        return sum(state.snapshot.nbytes(seen) for state in self.history.states)

    def commit(self, label: str) -> None:
        """Record the current state as a named history entry."""
        self._record(label)
        self._cache.clear()

    def _record(self, label: str, replace: bool = False) -> None:
        """Push (or replace) the current state and shrink the previous one to a patch where possible."""
        new = self._snapshot()
        if replace and self.history.index >= 0:
            previous = self.history.at(self.history.index - 1)
            if previous is not None:
                previous.materialize()  # its patches point at the state being replaced
            self.history.replace_current(label, new)
        else:
            self.history.capacity = self.history_capacity()
            self.history.push(label, new)
            previous = self.history.at(self.history.index - 1)
        if previous is not None:
            self._compact(previous, new)
        self._trim_history()

    @staticmethod
    def _compact(older: Snapshot, newer: Snapshot) -> None:
        """Replace images in *older* by patches against *newer* wherever only part of a layer changed."""
        for layer in older.layers:
            new_layer = newer._layer(layer.id)
            if new_layer is None:
                continue
            for is_mask in (False, True):
                old_img = layer.mask if is_mask else layer.image
                new_img = new_layer.mask if is_mask else new_layer.image
                if old_img is None or new_img is None or old_img is _PLACEHOLDER or (layer.id, is_mask) in older.patches:
                    continue
                if old_img is new_img:
                    box: Box | None = None  # unchanged: just point at the newer state so the reference moves forward
                else:
                    if old_img.size != new_img.size or old_img.mode != new_img.mode:
                        continue
                    box = _diff_bbox(old_img, new_img)
                    if box is not None and (box[2] - box[0]) * (box[3] - box[1]) > PATCH_MAX_FRACTION * old_img.width * old_img.height:
                        continue
                older.patches[(layer.id, is_mask)] = Patch(newer, layer.id, box, old_img.crop(box) if box else None, is_mask)
                if is_mask:
                    layer.mask = _PLACEHOLDER.convert("L")
                else:
                    layer.image = _PLACEHOLDER

    def _trim_history(self) -> None:
        """Drop the oldest undo states while the history holds more than its memory budget."""
        while len(self.history) > HISTORY_MIN_STEPS and self.history.can_undo and self.history_bytes() > HISTORY_BUDGET_BYTES:
            self.history.drop_oldest()

    def mark_dirty(self) -> None:
        """Treat the current state as unsaved (used after recovering an autosave)."""
        self._saved_index = -1

    def undo(self) -> bool:
        snap = self.history.undo()
        if snap is None:
            return False
        self._restore(snap)
        return True

    def redo(self) -> bool:
        snap = self.history.redo()
        if snap is None:
            return False
        self._restore(snap)
        return True

    def go_to_history(self, index: int) -> None:
        snap = self.history.go_to(index)
        if snap is not None:
            self._restore(snap)

    def revert(self) -> None:
        """Jump back to the first history state (and record that as a new step)."""
        if self.history.can_undo:
            self.go_to_history(0)
            self.commit("Revert")

    def touch(self) -> None:
        """Invalidate render caches after an uncommitted (live preview) change."""
        self._cache.clear()

    # ------------------------------------------------------------ snapshots --
    def take_snapshot(self, name: str) -> int:
        """Bookmark the current state under *name*; returns its index. Snapshots survive undo/redo."""
        name = name.strip() or f"Snapshot {len(self.snapshots) + 1}"
        self.snapshots.append((name, self._snapshot()))
        return len(self.snapshots) - 1

    def restore_snapshot(self, index: int) -> None:
        """Make a snapshot the current state (recorded as a new history step)."""
        name, snap = self.snapshots[index]
        self._restore(snap)
        self.commit(f"Snapshot: {name}")

    def rename_snapshot(self, index: int, name: str) -> None:
        _old, snap = self.snapshots[index]
        self.snapshots[index] = (name.strip() or _old, snap)

    def delete_snapshot(self, index: int) -> None:
        del self.snapshots[index]

    # ------------------------------------------------------------ rendering --
    def _layer_key(self, layer: Layer):
        return (
            layer.id,
            id(layer.image),
            layer.visible,
            layer.opacity,
            layer.blend_mode,
            tuple(sorted(layer.adjustments.items())),
            id(layer.mask) if layer.mask is not None else None,
            layer.mask_enabled,
            json.dumps(layer.meta, sort_keys=True, default=str) if layer.kind == "adjustment" or "effects" in layer.meta else None,
        )

    def _region(self, image: Image.Image, scale: float, view_box: Box) -> Image.Image:
        """*image* (document sized) resampled to *view_box* of the scaled picture."""
        sx0, sy0, sx1, sy1 = view_box
        out_size = (sx1 - sx0, sy1 - sy0)
        if scale == 1.0:
            if (sx0, sy0) == (0, 0) and out_size == self.size:
                return image
            return image.crop(view_box)
        box = (sx0 / scale, sy0 / scale, min(sx1 / scale, float(self.width)), min(sy1 / scale, float(self.height)))
        resample = Image.Resampling.NEAREST if scale > 1 else Image.Resampling.BILINEAR
        return image.resize(out_size, resample, box=box, reducing_gap=2.0 if scale < 1 else None)

    def _render_layer(
        self,
        layer: Layer,
        image: Image.Image,
        scale: float,
        view_box: Box,
        neutral: bool = False,
        mask: Image.Image | None = None,
    ) -> Image.Image:
        """Layer pixels for *view_box* (scaled space) with adjustments and *mask* applied (adjustments skipped when *neutral*)."""
        sx0, sy0, sx1, sy1 = view_box
        out_size = (sx1 - sx0, sy1 - sy0)
        full = (sx0, sy0) == (0, 0) and out_size == (math.ceil(self.width * scale), math.ceil(self.height * scale))
        adj = default_adjustments() if neutral else layer.adjustments
        effects = layer.meta.get("effects")
        # Pad the region a little for spatial effects so blur/sharpen edges are
        # computed with real neighbours instead of the crop boundary.
        pad = 0
        if adj.get("blur") or adj.get("sharpen"):
            pad = int(min(64, math.ceil((adj.get("blur", 0) * max(self.size) * scale / 800) * 3) + 4))
        if not no_effects(effects):
            pad = max(pad, min(600, effects_padding(effects, scale)))
        px0, py0 = max(0, sx0 - pad), max(0, sy0 - pad)
        px1 = min(math.ceil(self.width * scale), sx1 + pad)
        py1 = min(math.ceil(self.height * scale), sy1 + pad)
        doc_box = (px0 / scale, py0 / scale, min(px1 / scale, float(self.width)), min(py1 / scale, float(self.height)))
        if scale == 1.0 and full:
            src = image
        elif scale == 1.0:
            src = image.crop((int(px0), int(py0), int(px1), int(py1)))
        else:
            # NEAREST when zooming in (crisp pixels, like Photoshop); a cheap
            # two-step bilinear reduction when zooming out.
            resample = Image.Resampling.NEAREST if scale > 1 else Image.Resampling.BILINEAR
            src = image.resize((px1 - px0, py1 - py0), resample, box=doc_box, reducing_gap=2.0 if scale < 1 else None)
        if not is_neutral(adj):
            src = apply_adjustments(src, adj, doc_size=self.size, region=doc_box, scale=scale)
        src = ensure_rgba(src)
        if mask is not None:
            mask_region = self._region(mask, scale, (px0, py0, px1, py1))
            src = src.copy()
            src.putalpha(ImageChops.multiply(src.getchannel("A"), mask_region))
        if not no_effects(effects):
            src = apply_effects(src, effects, scale)
        if pad:
            src = src.crop((sx0 - px0, sy0 - py0, sx0 - px0 + out_size[0], sy0 - py0 + out_size[1]))
        return src

    def composite(
        self,
        scale: float = 1.0,
        view_box: Box | None = None,
        overrides: dict[int, Image.Image] | None = None,
        include_hidden: bool = False,
        use_cache: bool = True,
        neutral: bool = False,
        mask_overrides: dict[int, Image.Image] | None = None,
    ) -> Image.Image:
        """Render the visible layers.

        ``scale`` is output pixels per document pixel, ``view_box`` an integer
        box in *scaled* pixel space (defaults to the whole document) and
        ``overrides`` maps layer ids to replacement images (used for live
        previews of strokes/filters without committing); ``mask_overrides``
        does the same for layer masks.  ``use_cache=False`` keeps one-off
        renders (e.g. colour picking) from evicting cached views.
        ``neutral=True`` renders the layers without their adjustments ("before" view).
        """
        if scale <= 0:
            raise ValueError("scale must be positive")
        full_w, full_h = math.ceil(self.width * scale), math.ceil(self.height * scale)
        if view_box is None:
            view_box = (0, 0, full_w, full_h)
        sx0, sy0, sx1, sy1 = view_box
        sx0, sy0 = max(0, int(sx0)), max(0, int(sy0))
        sx1, sy1 = min(full_w, int(sx1)), min(full_h, int(sy1))
        view_box = (sx0, sy0, sx1, sy1)
        if sx1 <= sx0 or sy1 <= sy0:
            return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        overrides = overrides or {}
        mask_overrides = mask_overrides or {}
        layers = [layer for i, layer in enumerate(self.layers) if include_hidden or self.effective_visible(i)]
        if not use_cache:
            return self._blend_layers(layers, scale, view_box, overrides, use_cache=False, neutral=neutral, mask_overrides=mask_overrides)
        key = self._composite_key(layers, scale, view_box, neutral, overrides, mask_overrides)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = self._blend_layers(layers, scale, view_box, overrides, neutral=neutral, mask_overrides=mask_overrides)
        self._cache.put(key, result)
        return result

    def _composite_key(
        self,
        layers: Sequence[Layer],
        scale: float,
        view_box: Box,
        neutral: bool,
        overrides: dict[int, Image.Image],
        mask_overrides: dict[int, Image.Image],
    ) -> tuple:
        return (
            "composite",
            scale,
            view_box,
            neutral,
            tuple(self._layer_key(layer) for layer in layers),
            tuple((k, id(v)) for k, v in overrides.items()),
            tuple((k, id(v)) for k, v in mask_overrides.items()),
        )

    def _clip_view_box(self, scale: float, view_box: Box | None) -> Box | None:
        full_w, full_h = math.ceil(self.width * scale), math.ceil(self.height * scale)
        if view_box is None:
            return (0, 0, full_w, full_h)
        sx0, sy0, sx1, sy1 = view_box
        box = (max(0, int(sx0)), max(0, int(sy0)), min(full_w, int(sx1)), min(full_h, int(sy1)))
        return box if box[2] > box[0] and box[3] > box[1] else None

    def cached_composite(
        self,
        scale: float = 1.0,
        view_box: Box | None = None,
        overrides: dict[int, Image.Image] | None = None,
        neutral: bool = False,
        mask_overrides: dict[int, Image.Image] | None = None,
    ) -> Image.Image | None:
        """The composite :meth:`composite` would return *if it is already cached* - None means it has to be rendered."""
        box = self._clip_view_box(scale, view_box)
        if box is None:
            return None
        layers = [layer for i, layer in enumerate(self.layers) if self.effective_visible(i)]
        return self._cache.get(self._composite_key(layers, scale, box, neutral, overrides or {}, mask_overrides or {}))

    def _blend_layers(
        self,
        layers: Sequence[Layer],
        scale: float,
        view_box: Box,
        overrides: dict[int, Image.Image],
        use_cache: bool = True,
        neutral: bool = False,
        mask_overrides: dict[int, Image.Image] | None = None,
    ) -> Image.Image:
        sx0, sy0, sx1, sy1 = view_box
        mask_overrides = mask_overrides or {}
        rendered_layers: list[tuple[Layer, Image.Image | None]] = []  # None = adjustment layer (applies to what is below)
        pending: dict[int, list[Layer]] = {}  # group id → members seen so far (they precede their header)
        header_ids = {layer.id for layer in layers if layer.kind == "group"}
        for layer in layers:
            gid = layer.meta.get("group")
            if gid is not None and gid in header_ids:
                # a member (or a nested header together with its own members) waits for its header
                pending.setdefault(gid, []).extend(pending.pop(layer.id, []) + [layer])
                continue
            mask = mask_overrides[layer.id] if layer.id in mask_overrides else layer.effective_mask
            if layer.kind == "group":
                members = pending.pop(layer.id, [])
                if not members:
                    continue
                group_image = self._render_group(layer, members, scale, view_box, overrides, mask_overrides, use_cache, neutral)
                if group_image.getchannel("A").getbbox() is not None:
                    rendered_layers.append((layer, group_image))
                continue
            if layer.kind == "adjustment":
                if not neutral and rendered_layers:
                    rendered_layers.append((layer, None))
                continue
            image = overrides.get(layer.id, layer.image)
            lkey = ("layer", self._layer_key(layer), id(image), id(mask) if mask is not None else None, scale, view_box, neutral)
            cached = self._cache.get(lkey) if use_cache else None
            if cached is None:
                cached = self._render_layer(layer, image, scale, view_box, neutral, mask)
                if use_cache:
                    self._cache.put(lkey, cached)
            if cached.getchannel("A").getbbox() is None:
                continue  # fully transparent: contributes nothing
            rendered_layers.append((layer, cached))
        if not rendered_layers:
            return Image.new("RGBA", (sx1 - sx0, sy1 - sy0), (0, 0, 0, 0))
        if len(rendered_layers) == 1 and rendered_layers[0][0].opacity >= 1.0 and rendered_layers[0][1] is not None:
            return rendered_layers[0][1]  # fast path: nothing to blend
        acc_p = acc_a = None
        for layer, rendered in rendered_layers:
            if rendered is None:
                if acc_p is not None and acc_a is not None:
                    adj_mask = mask_overrides[layer.id] if layer.id in mask_overrides else layer.effective_mask
                    acc_p = self._apply_adjustment_layer(layer, acc_p, acc_a, scale, view_box, adj_mask)
                continue
            arr = np.asarray(rendered).astype(np.float32)
            arr *= np.float32(1 / 255)
            src_rgb, src_a = arr[..., :3], arr[..., 3:4]
            if layer.opacity < 1.0:
                src_a = src_a * np.float32(layer.opacity)
            acc_p, acc_a = composite_premultiplied(acc_p, acc_a, src_rgb, src_a, layer.blend_mode)
        assert acc_p is not None and acc_a is not None
        out = np.empty((*acc_a.shape[:2], 4), np.float32)
        out[..., :3] = unpremultiply(acc_p, acc_a)
        out[..., 3:4] = acc_a
        np.clip(out, 0, 1, out=out)
        out *= 255
        out += 0.5
        return Image.fromarray(out.astype(np.uint8), "RGBA")

    def _render_group(
        self,
        header: Layer,
        members: Sequence[Layer],
        scale: float,
        view_box: Box,
        overrides: dict[int, Image.Image],
        mask_overrides: dict[int, Image.Image],
        use_cache: bool = True,
        neutral: bool = False,
    ) -> Image.Image:
        """Composite a group's members on their own, then apply the header's mask and effects (opacity/blend happen when blending)."""
        inner = self._blend_layers(members, scale, view_box, overrides, use_cache, neutral, mask_overrides)
        mask = mask_overrides[header.id] if header.id in mask_overrides else header.effective_mask
        if mask is not None:
            mask_region = self._region(mask, scale, view_box)
            inner = inner.copy()
            inner.putalpha(ImageChops.multiply(inner.getchannel("A"), mask_region))
        effects = header.meta.get("effects")
        if not no_effects(effects):
            inner = apply_effects(inner, effects, scale)
        return inner

    def _apply_adjustment_layer(
        self, layer: Layer, acc_p: np.ndarray, acc_a: np.ndarray, scale: float, view_box: Box, mask: Image.Image | None
    ) -> np.ndarray:
        """Run the adjustment layer's sliders / curves / levels over the premultiplied backdrop, through its mask and opacity."""
        sx0, sy0, sx1, sy1 = view_box
        straight = unpremultiply(acc_p, acc_a)
        rgba = np.concatenate([straight, acc_a], axis=2)
        img = Image.fromarray((np.clip(rgba, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA")
        doc_box = (sx0 / scale, sy0 / scale, min(sx1 / scale, float(self.width)), min(sy1 / scale, float(self.height)))
        if layer.has_adjustments:
            img = apply_adjustments(img, layer.adjustments, doc_size=self.size, region=doc_box, scale=scale)
        curves = layer.meta.get("curves")
        if curves:
            img = apply_curves(img, curves)
        levels = layer.meta.get("levels")
        if levels:
            img = apply_filter(img, "Levels", **levels)
        adjusted = np.asarray(ensure_rgba(img)).astype(np.float32)[..., :3] * np.float32(1 / 255)
        weight = np.full(acc_a.shape, np.float32(layer.opacity), np.float32)
        if mask is not None:
            mask_region = self._region(mask, scale, view_box)
            weight *= np.asarray(mask_region, dtype=np.float32)[..., None] * np.float32(1 / 255)
        if layer.blend_mode != "normal":
            blended_p, _ = composite_premultiplied(acc_p, acc_a, adjusted, acc_a * weight, layer.blend_mode)
            return blended_p
        return acc_p * (1 - weight) + adjusted * acc_a * weight

    def render_mask(self, scale: float = 1.0, view_box: Box | None = None, override: Image.Image | None = None) -> Image.Image | None:
        """The active layer's mask (or *override*) resampled like :meth:`composite` would - for mask overlays."""
        mask = override if override is not None else self.active_layer.mask
        if mask is None:
            return None
        full_w, full_h = math.ceil(self.width * scale), math.ceil(self.height * scale)
        if view_box is None:
            view_box = (0, 0, full_w, full_h)
        sx0, sy0, sx1, sy1 = view_box
        box = (max(0, int(sx0)), max(0, int(sy0)), min(full_w, int(sx1)), min(full_h, int(sy1)))
        if box[2] <= box[0] or box[3] <= box[1]:
            return None
        return self._region(mask, scale, box)

    def flatten(self) -> Image.Image:
        """Full-resolution RGBA composite of all visible layers."""
        return self.composite(1.0)

    def composite_below(self, index: int, scale: float = 1.0) -> Image.Image:
        """Composite of the visible layers underneath *index* (what an adjustment layer there works on)."""
        index = self._check_index(index)
        layers = [layer for layer in self.layers[:index] if layer.visible]
        full = (0, 0, math.ceil(self.width * scale), math.ceil(self.height * scale))
        if not layers:
            return Image.new("RGBA", (full[2], full[3]), (0, 0, 0, 0))
        return self._blend_layers(layers, scale, full, {}, use_cache=False)

    def export_image(self, mode: str = "RGBA", background: tuple = (255, 255, 255)) -> Image.Image:
        """Flattened image converted for export; RGB output is matted onto *background*."""
        flat = self.flatten()
        if mode == "RGBA":
            return flat
        matte = Image.new("RGBA", flat.size, (*parse_color(background)[:3], 255))
        return Image.alpha_composite(matte, flat).convert(mode)

    def thumbnail(self, size: tuple[int, int] = (128, 128)) -> Image.Image:
        scale = min(size[0] / self.width, size[1] / self.height, 1.0)
        return self.composite(scale)

    def pick_color(self, x: float, y: float, merged: bool = True) -> tuple[int, int, int, int] | None:
        """Colour under a document coordinate from the composite or the active layer.

        Only the single pixel that is needed gets rendered, so this is cheap
        even for huge documents with expensive adjustments.
        """
        ix, iy = int(math.floor(x)), int(math.floor(y))
        if not (0 <= ix < self.width and 0 <= iy < self.height):
            return None
        if not merged:
            return self.active_layer.image.getpixel((ix, iy))  # type: ignore[return-value]
        return self.composite(1.0, (ix, iy, ix + 1, iy + 1), use_cache=False).getpixel((0, 0))  # type: ignore[return-value]

    # ------------------------------------------------------- layer management --
    def _check_index(self, index: int | None) -> int:
        index = self.active_index if index is None else index
        if not 0 <= index < len(self.layers):
            raise DocumentError(f"no layer at index {index}")
        return index

    def _require_unlocked(self, index: int, pixels: bool = True) -> Layer:
        """The layer at *index* if it may be edited; with *pixels* it must also have pixels of its own (groups have none)."""
        layer = self.layers[index]
        if layer.locked:
            raise DocumentError(f"layer '{layer.name}' is locked")
        if pixels and layer.kind == "group":
            raise DocumentError(f"'{layer.name}' is a group - select a layer inside it to edit pixels")
        return layer

    # ---------------------------------------------------------------- groups --
    # A group is a header layer (kind "group") whose members sit directly below
    # it in the stack and carry ``meta["group"] = header.id``.  Groups may nest.
    def group_members(self, index: int) -> list[int]:
        """Indices of the direct members of the group whose header is at *index* (bottom → top)."""
        header = self.layers[index]
        if header.kind != "group":
            return []
        members = []
        for i in range(index - 1, -1, -1):
            layer = self.layers[i]
            if layer.meta.get("group") == header.id:
                members.append(i)
            elif self.group_of(i) is None or self._block_top(i) < index:
                # a nested member whose header is one of ours keeps scanning; anything else ends the group
                if not self._inside_group(i, header.id):
                    break
        return sorted(members)

    def _inside_group(self, index: int, header_id: int) -> bool:
        """True when the layer at *index* is (transitively) inside the group with *header_id*."""
        seen = 0
        gid = self.layers[index].meta.get("group")
        while gid is not None and seen < len(self.layers):
            if gid == header_id:
                return True
            header_index = self.layer_index(gid)
            if header_index is None:
                return False
            gid = self.layers[header_index].meta.get("group")
            seen += 1
        return False

    def group_of(self, index: int) -> int | None:
        """Index of the header of the group the layer at *index* belongs to (None for top-level layers)."""
        gid = self.layers[index].meta.get("group")
        return self.layer_index(gid) if gid is not None else None

    def _block(self, index: int) -> tuple[int, int]:
        """``(start, end)`` (end exclusive) of the layers that move together with *index*: a header and everything inside it."""
        if self.layers[index].kind != "group":
            return index, index + 1
        start = index
        for i in range(index - 1, -1, -1):
            if self._inside_group(i, self.layers[index].id):
                start = i
            else:
                break
        return start, index + 1

    def _block_top(self, index: int) -> int:
        return self._block(index)[1] - 1

    def effective_visible(self, index: int) -> bool:
        """Visible itself and inside no hidden group."""
        i: int | None = index
        seen = 0
        while i is not None and seen < len(self.layers):
            if not self.layers[i].visible:
                return False
            i = self.group_of(i)
            seen += 1
        return True

    def group_active(self, name: str | None = None) -> Layer:
        """Put the active layer (or group) into a new group; the group becomes active."""
        index = self.active_index
        start, end = self._block(index)
        inner = self.layers[index]
        header = Layer(
            Image.new("RGBA", self.size, (0, 0, 0, 0)),
            name or f"Group {sum(1 for lyr in self.layers if lyr.kind == 'group') + 1}",
            kind="group",
        )
        header.meta = {"group": inner.meta["group"]} if inner.meta.get("group") is not None else {}
        self.layers[index] = inner.copy(meta={**inner.meta, "group": header.id})
        self.layers.insert(end, header)
        self.active_index = end
        self.commit(f"Group '{inner.name}'")
        return header

    def ungroup(self, index: int | None = None) -> None:
        """Dissolve a group: its members take its place (and its parent group, if any)."""
        index = self._check_index(index)
        if self.layers[index].kind != "group":
            parent = self.group_of(index)
            if parent is None:
                raise DocumentError("the layer is not in a group")
            index = parent
        header = self.layers[index]
        outer = header.meta.get("group")
        for i in self.group_members(index):
            meta = dict(self.layers[i].meta)
            if outer is None:
                meta.pop("group", None)
            else:
                meta["group"] = outer
            self.layers[i] = self.layers[i].copy(meta=meta)
        del self.layers[index]
        self.active_index = min(max(0, index - 1), len(self.layers) - 1)
        self.commit(f"Ungroup '{header.name}'")

    def merge_group(self, index: int | None = None) -> Layer:
        """Replace a group by one pixel layer that looks exactly like it."""
        index = self._check_index(index)
        header = self.layers[index]
        if header.kind != "group":
            raise DocumentError("not a group")
        start, end = self._block(index)
        members = [layer for i, layer in enumerate(self.layers[start:index], start) if self.effective_visible(i)]
        image = self._render_group(header, members, 1.0, (0, 0, self.width, self.height), {}, {}, use_cache=False)
        merged = Layer(image, header.name, visible=header.visible, locked=header.locked)
        if header.meta.get("group") is not None:
            merged.meta["group"] = header.meta["group"]
        self.layers[start:end] = [merged]
        self.active_index = start
        self.commit(f"Merge Group '{header.name}'")
        return merged

    def set_active(self, index: int) -> None:
        self.active_index = self._check_index(index)

    def add_layer(self, layer: Layer | None = None, index: int | None = None, name: str | None = None, commit: bool = True) -> Layer:
        """Insert *layer* (or a new transparent one) above the active layer."""
        if layer is None:
            layer = Layer.blank(self.size, name or f"Layer {len(self.layers) + 1}")
        elif layer.size != self.size:
            raise DocumentError("layer size must match document size")
        if name:
            layer.name = name
        if index is None:
            index = self.active_index + 1
            active = self.layers[self.active_index] if self.layers else None
            if active is not None and active.kind != "group" and active.meta.get("group") is not None and "group" not in layer.meta:
                layer.meta["group"] = active.meta["group"]  # new layers join the group of the active layer
        else:
            index = max(0, min(len(self.layers), index))
        self.layers.insert(index, layer)
        self.active_index = index
        if commit:
            self.commit(f"New Layer '{layer.name}'")
        return layer

    def add_image_layer(self, image: Image.Image, name: str = "Image", position: tuple[int, int] | None = None) -> Layer:
        """Place an arbitrary image on a new layer (centred by default)."""
        rgba = ensure_rgba(image)
        canvas = Image.new("RGBA", self.size, (0, 0, 0, 0))
        if position is None:
            position = ((self.width - rgba.width) // 2, (self.height - rgba.height) // 2)
        canvas.paste(rgba, (int(position[0]), int(position[1])))
        return self.add_layer(Layer(canvas, name))

    def add_adjustment_layer(self, name: str | None = None, adjustments: dict[str, int] | None = None) -> Layer:
        """Insert an adjustment layer above the active layer: it has no pixels, only adjustments applied to what is below."""
        layer = Layer(
            Image.new("RGBA", self.size, (0, 0, 0, 0)),
            name or f"Adjustment {sum(1 for lyr in self.layers if lyr.kind == 'adjustment') + 1}",
            kind="adjustment",
            adjustments=adjustments or default_adjustments(),
            mask=Image.new("L", self.size, 255),
        )
        self.add_layer(layer, commit=False)
        self.mask_editing = True
        self.commit(f"New Adjustment Layer '{layer.name}'")
        return layer

    def set_layer_meta(self, index: int | None = None, commit: bool = True, label: str = "Layer Settings", **meta) -> Layer:
        """Update entries of a layer's ``meta`` (``None`` removes a key); used for adjustment-layer curves and levels."""
        index = self._check_index(index)
        layer = self.layers[index]
        new_meta = {**layer.meta, **{k: v for k, v in meta.items() if v is not None}}
        for key, value in meta.items():
            if value is None:
                new_meta.pop(key, None)
        self.layers[index] = layer.copy(meta=new_meta)
        self._cache.clear()
        if commit:
            self.commit(label)
        return self.layers[index]

    def remove_layer(self, index: int | None = None) -> None:
        """Delete a layer - or a whole group with everything inside it."""
        index = self._check_index(index)
        start, end = self._block(index)
        if end - start >= len(self.layers):
            raise DocumentError("cannot delete the only layer")
        name = self.layers[index].name
        del self.layers[start:end]
        self.active_index = min(start, len(self.layers) - 1)
        self.commit(f"Delete {'Group' if end - start > 1 else 'Layer'} '{name}'")

    def duplicate_layer(self, index: int | None = None) -> Layer:
        """Duplicate a layer (or a group with its members) right above the original."""
        index = self._check_index(index)
        start, end = self._block(index)
        originals = self.layers[start:end]
        copies = [layer.duplicate(name=layer.name if i != index else None) for i, layer in enumerate(originals, start)]
        id_map = {orig.id: dup.id for orig, dup in zip(originals, copies, strict=True)}
        for dup in copies:
            if dup.kind == "background":
                dup.kind = "pixel"
            gid = dup.meta.get("group")
            if gid in id_map:
                dup.meta["group"] = id_map[gid]
        self.layers[end:end] = copies
        self.active_index = end + (index - start)
        self.commit(f"Duplicate {'Group' if end - start > 1 else 'Layer'} '{self.layers[index].name}'")
        return self.layers[self.active_index]

    def move_layer(self, src: int, dst: int) -> None:
        """Move the layer at *src* so that it ends up at index *dst* (a group header takes its members along)."""
        src = self._check_index(src)
        dst = max(0, min(len(self.layers) - 1, dst))
        if src == dst:
            return
        start, end = self._block(src)
        block = self.layers[start:end]
        del self.layers[start:end]
        insert_at = max(0, min(len(self.layers), dst - (len(block) - 1)))
        self.layers[insert_at:insert_at] = block
        self.active_index = insert_at + len(block) - 1
        self.commit("Reorder Layers")

    def _set_group(self, index: int, header_id: int | None) -> None:
        layer = self.layers[index]
        meta = dict(layer.meta)
        if header_id is None:
            meta.pop("group", None)
        else:
            meta["group"] = header_id
        self.layers[index] = layer.copy(meta=meta)

    def raise_layer(self, index: int | None = None) -> None:
        """Move one step up - into the group right above, out of the current group at its top, or past the neighbour."""
        index = self._check_index(index)
        start, end = self._block(index)
        layer = self.layers[index]
        gid = layer.meta.get("group")
        if end >= len(self.layers):
            if gid is not None:  # top of the list and inside a group: impossible (the header is above) - nothing to do
                return
            return
        above = self.layers[end]
        if above.kind == "group" and above.id == gid:
            # at the top of our group: step out above the header
            self._set_group(index, above.meta.get("group"))
            self.move_layer(index, end)
            self._record("Move Out of Group", replace=True)
            return
        above_gid = above.meta.get("group")
        if above_gid is not None and above_gid != gid and above.kind != "group":
            # the layer above is the bottom member of a group: join it
            self._set_group(index, above_gid)
            self.commit("Move Into Group")
            return
        if above_gid is not None and above_gid != gid and above.kind == "group":
            self._set_group(index, above_gid)
            self.commit("Move Into Group")
            return
        _above_start, above_end = self._block(end)
        self.move_layer(index, above_end - 1)

    def lower_layer(self, index: int | None = None) -> None:
        """Move one step down - into the group right below, out of the current group at its bottom, or past the neighbour."""
        index = self._check_index(index)
        start, end = self._block(index)
        layer = self.layers[index]
        gid = layer.meta.get("group")
        below_index = start - 1
        if below_index < 0:
            if gid is not None:  # bottom of the list inside a group: step out
                header_index = self.group_of(index)
                outer = self.layers[header_index].meta.get("group") if header_index is not None else None
                self._set_group(index, outer)
                self.commit("Move Out of Group")
            return
        below = self.layers[below_index]
        if gid is not None and below.meta.get("group") != gid and not self._inside_group(below_index, gid):
            # bottom member of our group: step out below the group (same position, no longer a member)
            header_index = self.group_of(index)
            outer = self.layers[header_index].meta.get("group") if header_index is not None else None
            self._set_group(index, outer)
            self.commit("Move Out of Group")
            return
        size = end - start
        if below.kind == "group" and below.meta.get("group") == gid:
            # a group header right below: step into it as its top member
            self._set_group(index, below.id)
            self.move_layer(index, below_index + size - 1)
            self._record("Move Into Group", replace=True)
            return
        below_start, _below_end = self._block(below_index)
        self.move_layer(index, below_start + size - 1)

    def set_layer_props(self, index: int | None = None, commit: bool = True, **props) -> Layer:
        """Change name/visible/opacity/blend_mode/locked of a layer."""
        index = self._check_index(index)
        layer = self.layers[index]
        allowed = {"name", "visible", "opacity", "blend_mode", "locked"}
        unknown = set(props) - allowed
        if unknown:
            raise DocumentError(f"unknown layer properties: {sorted(unknown)}")
        new = layer.copy(**props)
        self.layers[index] = new
        self._cache.clear()
        if commit:
            what = ", ".join(sorted(props))
            self.commit(f"Layer {what}")
        return new

    def set_adjustments(self, adj: dict[str, int], index: int | None = None, commit: bool = False, label: str = "Adjust") -> None:
        """Set the non-destructive adjustments of a layer (live preview by default)."""
        index = self._check_index(index)
        self.layers[index].adjustments = normalize_adjustments(adj)
        self._cache.clear()
        if commit:
            self.commit(label)

    def set_adjustment(self, key: str, value: int, index: int | None = None, commit: bool = False) -> None:
        index = self._check_index(index)
        adj = dict(self.layers[index].adjustments)
        adj[key] = value
        self.set_adjustments(adj, index, commit, label=f"Adjust {key.title()}")

    def reset_adjustments(self, index: int | None = None) -> None:
        self.set_adjustments(default_adjustments(), index, commit=True, label="Reset Adjustments")

    def bake_adjustments(self, index: int | None = None) -> None:
        """Apply the layer's adjustments to its pixels and clear the sliders."""
        index = self._check_index(index)
        layer = self._require_unlocked(index)
        if layer.kind == "adjustment":
            raise DocumentError("an adjustment layer has no pixels - merge it down to bake it")
        if layer.kind in SMART_KINDS:
            raise DocumentError("adjustments stay live on 3D and text layers - rasterize the layer to bake them")
        if not layer.has_adjustments:
            return
        rendered = apply_adjustments(layer.image, layer.adjustments, doc_size=self.size)
        mask = self.selection_mask()
        if mask is not None:
            rendered = Image.composite(rendered, layer.image, mask)
        self.layers[index] = layer.copy(image=ensure_rgba(rendered), adjustments=default_adjustments())
        self.commit("Apply Adjustments")

    def merge_down(self, index: int | None = None) -> None:
        """Merge a layer into the one below it (blend mode, opacity and adjustments are baked in); a group merges into one layer."""
        index = self._check_index(index)
        if self.layers[index].kind == "group":
            self.merge_group(index)
            return
        if index == 0:
            raise DocumentError("the bottom layer has nothing to merge into")
        top, below = self.layers[index], self.layers[index - 1]
        if below.kind == "group" or top.meta.get("group") != below.meta.get("group"):
            raise DocumentError("merge down works between two layers of the same group")
        base = below.copy(opacity=1.0, blend_mode="normal", visible=True)
        merged = self._blend_layers([base, top.copy(visible=True)], 1.0, (0, 0, self.width, self.height), {})
        self.layers[index - 1 : index + 1] = [below.copy(image=merged, adjustments=default_adjustments(), mask=None, mask_enabled=True)]
        self.active_index = index - 1
        self.commit(f"Merge '{top.name}' Down")

    def flatten_layers(self) -> None:
        flat = self.flatten()
        name = self.layers[0].name if len(self.layers) == 1 else "Background"
        self.layers = [Layer(flat, name, kind="background")]
        self.active_index = 0
        self.commit("Flatten Image")

    def merge_visible(self) -> None:
        """Merge every visible layer into one; hidden layers stay (headers of emptied groups go)."""
        visible = [i for i in range(len(self.layers)) if self.effective_visible(i) and self.layers[i].kind != "group"]
        if len(visible) < 2:
            return
        flat = self.flatten()
        keep = [
            layer
            for i, layer in enumerate(self.layers)
            if not self.effective_visible(i)
            and not (
                layer.kind == "group"
                and not any(self._inside_group(j, layer.id) for j in range(len(self.layers)) if not self.effective_visible(j))
            )
        ]
        merged = Layer(flat, "Merged", kind="pixel")
        first = visible[0]
        before = [layer for layer in keep if self.layers.index(layer) < first]
        after = [layer for layer in keep if self.layers.index(layer) >= first]
        self.layers = before + [merged] + after
        self.active_index = len(before)
        self.commit("Merge Visible")

    # ------------------------------------------------------------ layer masks --
    def add_mask(self, index: int | None = None, mode: str = "reveal") -> Layer:
        """Give a layer a mask: ``reveal`` (all white), ``hide`` (all black) or ``selection`` (white inside the selection).

        With a selection present, ``reveal``/``hide`` mean *reveal/hide the selection*.
        The new mask becomes the edit target.
        """
        index = self._check_index(index)
        layer = self._require_unlocked(index, pixels=False)
        if layer.mask is not None:
            raise DocumentError("the layer already has a mask")
        if mode not in ("reveal", "hide", "selection"):
            raise DocumentError(f"unknown mask mode {mode!r}")
        if self.selection is not None and not self.selection.is_empty:
            mask = self.selection.mask.copy()
            if mode == "hide":
                mask = ImageChops.invert(mask)
        else:
            mask = Image.new("L", self.size, 0 if mode == "hide" else 255)
        self.layers[index] = layer.copy(mask=mask, mask_enabled=True)
        self.active_index = index
        self.mask_editing = True
        self.commit("Add Layer Mask")
        return self.layers[index]

    def remove_mask(self, index: int | None = None, apply: bool = False) -> Layer:
        """Delete a layer's mask; with *apply* its effect is baked into the layer's alpha first."""
        index = self._check_index(index)
        layer = self._require_unlocked(index, pixels=False)
        if layer.mask is None:
            raise DocumentError("the layer has no mask")
        if layer.kind == "adjustment":
            raise DocumentError("an adjustment layer always keeps its mask - disable or fill it instead")
        image = layer.masked_image() if apply else layer.image
        self.layers[index] = layer.copy(image=image, mask=None, mask_enabled=True)
        if index == self.active_index:
            self.mask_editing = False
        self.commit("Apply Layer Mask" if apply else "Delete Layer Mask")
        return self.layers[index]

    def set_mask_enabled(self, enabled: bool, index: int | None = None) -> None:
        index = self._check_index(index)
        layer = self.layers[index]
        if layer.mask is None:
            raise DocumentError("the layer has no mask")
        if layer.mask_enabled == bool(enabled):
            return
        self.layers[index] = layer.copy(mask_enabled=bool(enabled))
        self.commit("Enable Layer Mask" if enabled else "Disable Layer Mask")

    def invert_mask(self, index: int | None = None) -> None:
        index = self._check_index(index)
        layer = self._require_unlocked(index, pixels=False)
        if layer.mask is None:
            raise DocumentError("the layer has no mask")
        self.layers[index] = layer.copy(mask=ImageChops.invert(layer.mask))
        self.commit("Invert Layer Mask")

    def select_mask(self, index: int | None = None, mode: str = "replace") -> None:
        """Load a layer's mask as the selection."""
        index = self._check_index(index)
        mask = self.layers[index].mask
        if mask is None:
            raise DocumentError("the layer has no mask")
        self.set_selection(Selection(mask.copy()), mode)
        self._record("Load Mask as Selection", replace=True)

    # ------------------------------------------------------------ pixel edits --
    def is_smart(self, index: int | None = None) -> bool:
        """True for layers whose picture is rendered from their settings (3D objects, text)."""
        index = self._check_index(index)
        return self.layers[index].kind in SMART_KINDS

    def _require_pixels(self, index: int) -> Layer:
        """Like :meth:`_require_unlocked` but also refuses smart layers, whose pixels would be lost on the next render."""
        layer = self._require_unlocked(index, pixels=True)
        if layer.kind in SMART_KINDS:
            what = "3D object" if layer.kind == "3d" else "text layer"
            raise DocumentError(f"'{layer.name}' is a {what} - rasterize it (Layer ▸ Rasterize Layer) to paint on its pixels")
        return layer

    def apply_to_active(self, fn: Callable[[Image.Image], Image.Image], label: str, respect_selection: bool = True) -> None:
        """Replace the active layer's pixels - or its mask while :attr:`editing_mask` - with ``fn(image)`` (limited to the selection)."""
        layer = self._require_unlocked(self.active_index, pixels=False) if self.editing_mask else self._require_pixels(self.active_index)
        result = self.preview_active(fn, respect_selection)
        if self.editing_mask:
            self.layers[self.active_index] = layer.copy(mask=rgba_to_mask(result))
            self.commit(f"{label} (Mask)")
        else:
            self.layers[self.active_index] = layer.copy(image=result)
            self.commit(label)

    def preview_active(self, fn: Callable[[Image.Image], Image.Image], respect_selection: bool = True) -> Image.Image:
        """Compute what :meth:`apply_to_active` would produce, without committing (RGBA, also for masks)."""
        source = self.edit_image()
        result = ensure_rgba(fn(source))
        if result.size != source.size:
            raise DocumentError("edit changed the layer size")
        mask = self.selection_mask() if respect_selection else None
        if mask is not None:
            result = Image.composite(result, source, mask)
        return result

    def apply_filter(self, name: str, **params) -> None:
        """Run a filter on the active layer; on 3D / text layers it becomes a *smart filter* that survives re-rendering."""
        if self.is_smart() and not self.editing_mask:
            self.add_smart_filter({"kind": "filter", "name": name, "params": dict(params)}, name)
            return
        self.apply_to_active(lambda im: apply_filter(im, name, **params), name)

    def apply_curves(self, curves: Mapping[str, Sequence[Sequence[float]]], label: str = "Curves") -> None:
        """Remap tones of the active layer through per-channel curves (see :mod:`pixelforge.core.curves`)."""
        if self.is_smart() and not self.editing_mask:
            self.add_smart_filter({"kind": "curves", "curves": {k: [list(p) for p in v] for k, v in curves.items()}}, label)
            return
        self.apply_to_active(lambda im: apply_curves(im, curves), label)

    # ---------------------------------------------------------- smart layers --
    @staticmethod
    def apply_smart_filters(image: Image.Image, filters: Sequence[Mapping[str, Any]]) -> Image.Image:
        """Run the recorded filter stack over a freshly rendered picture."""
        for entry in filters:
            if entry.get("kind") == "curves":
                image = apply_curves(image, entry.get("curves") or {})
            else:
                image = apply_filter(image, str(entry.get("name")), **dict(entry.get("params") or {}))
        return ensure_rgba(image)

    def smart_filters(self, index: int | None = None) -> list[dict[str, Any]]:
        index = self._check_index(index)
        return [dict(f) for f in self.layers[index].meta.get("filters") or []]

    def render_smart(self, index: int, meta: Mapping[str, Any] | None = None) -> Image.Image:
        """The picture of a smart layer: its 3D scene or text rendered, then its filter stack applied."""
        layer = self.layers[index]
        meta = layer.meta if meta is None else meta
        if layer.kind == "3d":
            base = self.render_3d(Scene.from_dict(meta["3d"]))
        elif layer.kind == "text":
            base = painting.render_text(
                self.size,
                tuple(meta["position"]),
                meta["text"],
                tuple(meta["color"]),
                meta.get("font_name"),
                meta.get("font_size", 32),
                stroke_width=meta.get("stroke_width", 0),
                stroke_color=tuple(meta["stroke_color"]) if meta.get("stroke_color") else None,
            )
        else:
            raise DocumentError("not a smart layer")
        return self.apply_smart_filters(base, meta.get("filters") or [])

    def add_smart_filter(self, entry: Mapping[str, Any], label: str, index: int | None = None) -> None:
        index = self._check_index(index)
        layer = self._require_unlocked(index)
        if layer.kind not in SMART_KINDS:
            raise DocumentError("smart filters need a 3D or text layer")
        filters = [*(layer.meta.get("filters") or []), dict(entry)]
        meta = {**layer.meta, "filters": filters}
        self.layers[index] = layer.copy(image=self.render_smart(index, meta), meta=meta)
        self.commit(label)

    def clear_smart_filters(self, index: int | None = None) -> None:
        """Remove every smart filter from a 3D / text layer (the picture goes back to the plain render)."""
        index = self._check_index(index)
        layer = self._require_unlocked(index)
        if not layer.meta.get("filters"):
            return
        meta = {k: v for k, v in layer.meta.items() if k != "filters"}
        self.layers[index] = layer.copy(image=self.render_smart(index, meta), meta=meta)
        self.commit("Remove Smart Filters")

    def rasterize_layer(self, index: int | None = None) -> Layer:
        """Turn a 3D / text layer into an ordinary pixel layer (its settings are dropped, the pixels stay)."""
        index = self._check_index(index)
        layer = self._require_unlocked(index)
        if layer.kind not in SMART_KINDS:
            raise DocumentError("only 3D and text layers can be rasterized")
        keep = {k: v for k, v in layer.meta.items() if k in ("group", "effects")}
        name = layer.name[2:] if layer.kind == "text" and layer.name.startswith("T ") else layer.name
        self.layers[index] = layer.copy(kind="pixel", meta=keep, name=name)
        self.commit(f"Rasterize '{name}'")
        return self.layers[index]

    def fill(self, color, opacity: float = 1.0, label: str = "Fill") -> None:
        mask = self.selection_mask()
        self.apply_to_active(lambda im: painting.fill(im, color, mask, opacity), label, respect_selection=False)

    def clear(self) -> None:
        """Delete the selected pixels (whole layer when nothing is selected)."""
        mask = self.selection_mask()
        self.apply_to_active(lambda im: painting.clear(im, mask), "Clear", respect_selection=False)

    def paint(self, coverage, color, opacity: float = 1.0, label: str = "Brush Stroke") -> None:
        mask = self.selection_mask()
        self.apply_to_active(lambda im: painting.paint(im, coverage, color, opacity, mask), label, respect_selection=False)

    def clone(
        self,
        coverage,
        source: Image.Image | None,
        offset: tuple[int, int] = (0, 0),
        opacity: float = 1.0,
        heal: bool = False,
    ) -> None:
        """Clone-stamp (or heal) pixels from *source* shifted by *offset* onto the active layer."""
        mask = self.selection_mask()
        label = "Healing Brush" if heal else "Clone Stamp"
        self.apply_to_active(lambda im: painting.clone(im, coverage, source, offset, opacity, mask, heal), label, respect_selection=False)

    def erase(self, coverage, opacity: float = 1.0) -> None:
        mask = self.selection_mask()
        self.apply_to_active(lambda im: painting.erase(im, coverage, opacity, mask), "Eraser", respect_selection=False)

    def gradient(self, start, end, color_a, color_b, kind: str = "linear") -> None:
        mask = self.selection_mask()
        self.apply_to_active(
            lambda im: painting.apply_gradient(im, start, end, color_a, color_b, kind, mask), "Gradient", respect_selection=False
        )

    def draw_shape(self, kind: str, box, fill_color=None, outline_color=None, width: int = 0) -> None:
        mask = self.selection_mask()
        self.apply_to_active(
            lambda im: painting.draw_shape(im, kind, box, fill_color, outline_color, width, mask),
            f"Draw {kind.title()}",
            respect_selection=False,
        )

    def offset_active(self, dx: int, dy: int, label: str = "Move") -> None:
        """Move the active layer's pixels (mask linked) - or only its mask while :attr:`editing_mask`.

        3D and text layers move through their settings so they stay editable.
        """
        if dx == 0 and dy == 0:
            return
        layer = self.active_layer
        if layer.kind in SMART_KINDS and not self.editing_mask:
            self._require_unlocked(self.active_index)
            if layer.kind == "3d":
                scene = self.scene_of()
                assert scene is not None
                self.update_3d_layer(scene=replace(scene, position=(scene.position[0] + dx, scene.position[1] + dy)), label=label)
            else:
                x, y = layer.meta["position"]
                self.update_text_layer(position=[float(x + dx), float(y + dy)], label=label)
            if layer.mask is not None:
                moved = self.layers[self.active_index]
                self.layers[self.active_index] = moved.copy(mask=painting.offset_image(moved.mask, dx, dy))  # type: ignore[arg-type]
                self._record(label, replace=True)
            return
        self.apply_to_active(lambda im: painting.offset_image(im, dx, dy), label, respect_selection=False)
        layer = self.active_layer
        if not self.editing_mask and layer.mask is not None:
            self.layers[self.active_index] = layer.copy(mask=painting.offset_image(layer.mask, dx, dy))
            self._record(label, replace=True)
        if self.selection is not None:
            self.selection = self.selection.transformed(lambda m: painting.offset_image(m, dx, dy))
            self._record(label, replace=True)

    def add_text_layer(
        self, text: str, position, color, font_name: str | None = None, font_size: int = 32, stroke_width: int = 0, stroke_color=None
    ) -> Layer:
        image = painting.render_text(
            self.size, position, text, color, font_name, font_size, stroke_width=stroke_width, stroke_color=stroke_color
        )
        meta = {
            "text": text,
            "position": [float(position[0]), float(position[1])],
            "color": list(parse_color(color)),
            "font_name": font_name,
            "font_size": int(font_size),
            "stroke_width": int(stroke_width),
            "stroke_color": list(parse_color(stroke_color)) if stroke_color is not None else None,
        }
        label = text.splitlines()[0][:20] if text.strip() else "Text"
        layer = Layer(image, f"T {label}", kind="text", meta=meta)
        return self.add_layer(layer)

    def update_text_layer(self, index: int | None = None, label: str = "Edit Text", **changes) -> Layer:
        """Re-render a text layer with changed text/font/colour/position (its smart filters are re-applied)."""
        index = self._check_index(index)
        layer = self._require_unlocked(index)
        if layer.kind != "text":
            raise DocumentError("not a text layer")
        meta = {**layer.meta, **changes}
        image = self.render_smart(index, meta)
        title = meta["text"].splitlines()[0][:20] if meta["text"].strip() else "Text"
        self.layers[index] = layer.copy(image=image, meta=meta, name=f"T {title}")
        self.commit(label)
        return self.layers[index]

    # ----------------------------------------------------- free transform --
    def content_box(self, index: int | None = None) -> Box | None:
        """What Free Transform works on: the selection's bounds, else the layer's opaque bounds (None when empty)."""
        index = self._check_index(index)
        if self.selection is not None and not self.selection.is_empty:
            return self.selection.bbox
        layer = self.layers[index]
        source = layer.mask if index == self.active_index and self.editing_mask else layer.image.getchannel("A")
        assert source is not None
        return source.getbbox()

    def _transformed_piece(
        self, transform: painting.Transform, image: Image.Image, resample: Image.Resampling | None = None
    ) -> Image.Image:
        """*image* with *transform* applied to the selected part only (everything when nothing is selected)."""
        if self.selection is None or self.selection.is_empty:
            return painting.transform_image(image, transform, resample)
        sel = self.selection.mask
        if image.mode == "RGBA":
            piece = image.copy()
            piece.putalpha(ImageChops.multiply(image.getchannel("A"), sel))
            rest = image.copy()
            rest.putalpha(ImageChops.multiply(image.getchannel("A"), ImageChops.invert(sel)))
            moved = painting.transform_image(piece, transform, resample)
            return Image.alpha_composite(rest, moved)
        piece = ImageChops.multiply(image, sel)
        rest = ImageChops.multiply(image, ImageChops.invert(sel))
        moved = painting.transform_image(piece, transform, resample)
        return ImageChops.lighter(rest, moved)

    def preview_transform(self, transform: painting.Transform, index: int | None = None) -> Image.Image:
        """The active layer's edit image (pixels or mask as grey) as :meth:`transform_active` would leave it."""
        index = self._check_index(index)
        return self._transformed_piece(transform, self.edit_image(index))

    def transform_active(self, transform: painting.Transform, label: str = "Free Transform") -> None:
        """Scale / rotate / move the selected pixels of the active layer (its mask moves along; the selection follows)."""
        if transform.is_identity:
            return
        layer = self._require_unlocked(self.active_index, pixels=not self.editing_mask)
        if layer.kind in SMART_KINDS and not self.editing_mask:
            self._transform_smart(layer, transform, label)
            return
        if self.editing_mask:
            assert layer.mask is not None
            new_layer = layer.copy(mask=self._transformed_piece(transform, layer.mask))
        else:
            mask = self._transformed_piece(transform, layer.mask) if layer.mask is not None else None
            new_layer = layer.copy(image=self._transformed_piece(transform, layer.image), mask=mask)
        self.layers[self.active_index] = new_layer
        if self.selection is not None:
            moved = self.selection.transformed(lambda m: painting.transform_image(m, transform))
            self.selection = None if moved.is_empty else moved
        self.commit(label)

    def _transform_smart(self, layer: Layer, transform: painting.Transform, label: str) -> None:
        """Free Transform of a 3D / text layer changes its settings (size, position, rotation) instead of its pixels."""
        scale_factor = (abs(transform.scale_x) + abs(transform.scale_y)) / 2
        cx, cy = transform.center
        new_cx, new_cy = transform.apply(cx, cy)
        if layer.kind == "3d":
            scene = self.scene_of()
            assert scene is not None
            rx, ry, rz = scene.rotation
            self.update_3d_layer(
                scene=replace(
                    scene,
                    size=scene.size * scale_factor,
                    position=(scene.position[0] + new_cx - cx, scene.position[1] + new_cy - cy),
                    rotation=(rx, ry, rz + transform.angle),
                ),
                label=label,
            )
            return
        x, y = layer.meta["position"]
        # scale the text about the transform's pivot, then move it like the pivot moved
        nx = cx + (x - cx) * scale_factor + (new_cx - cx)
        ny = cy + (y - cy) * scale_factor + (new_cy - cy)
        size = max(4, int(round(layer.meta.get("font_size", 32) * scale_factor)))
        self.update_text_layer(position=[float(nx), float(ny)], font_size=size, label=label)

    # ------------------------------------------------------------ 3D layers --
    def render_3d(self, scene: Scene, scale: float = 1.0, supersample: int = 2) -> Image.Image:
        """Render *scene* for this document (at *scale* of its size; use a small scale for quick previews)."""
        return render_scene(scene, self.size, supersample=supersample, scale=scale)

    def add_3d_layer(self, scene: Scene | None = None, name: str | None = None) -> Layer:
        """Add a 3D object layer: its picture is rendered from the :class:`Scene` kept in ``meta["3d"]``."""
        scene = (scene or Scene()).normalized()
        image = self.render_3d(scene)
        layer = Layer(image, name or f"3D {scene.shape.title()}", kind="3d", meta={"3d": scene.to_dict()})
        return self.add_layer(layer, commit=True)

    def scene_of(self, index: int | None = None) -> Scene | None:
        index = self._check_index(index)
        layer = self.layers[index]
        if layer.kind != "3d" or "3d" not in layer.meta:
            return None
        return Scene.from_dict(layer.meta["3d"])

    def update_3d_layer(
        self,
        index: int | None = None,
        scene: Scene | None = None,
        label: str = "Edit 3D Object",
        image: Image.Image | None = None,
        **changes,
    ) -> Layer:
        """Re-render a 3D layer with a new *scene* (or field *changes*) and record it; *image* skips the render."""
        index = self._check_index(index)
        layer = self._require_unlocked(index)
        current = self.scene_of(index)
        if current is None:
            raise DocumentError("not a 3D layer")
        new_scene = scene or current
        if changes:
            new_scene = replace(new_scene, **changes)
        new_scene = new_scene.normalized()
        if image is None or image.size != self.size:
            image = self.render_3d(new_scene)
        filters = layer.meta.get("filters") or []
        if filters:
            image = self.apply_smart_filters(image, filters)
        self.layers[index] = layer.copy(image=image, meta={**layer.meta, "3d": new_scene.to_dict()})
        self.commit(label)
        return self.layers[index]

    def extrude_layer(self, index: int | None = None, depth: int = 30) -> Layer:
        """Turn the opaque pixels of a layer into a textured 3D extrusion on a new layer above it."""
        index = self._check_index(index)
        source = self.layers[index]
        if source.kind in ("group", "adjustment"):
            raise DocumentError("only layers with pixels can be extruded")
        image = source.masked_image()
        if self.selection is not None and not self.selection.is_empty:
            image = image.copy()
            image.putalpha(ImageChops.multiply(image.getchannel("A"), self.selection.mask))
        try:
            scene = extrusion_scene(image, depth)
        except ValueError as exc:
            raise DocumentError(str(exc)) from exc
        self.active_index = index
        return self.add_3d_layer(scene, name=f"3D {source.name}")

    # ---------------------------------------------------------- transforms --
    def _transform_all(
        self,
        fn: Callable[[Image.Image], Image.Image],
        label: str,
        keep_selection: bool = True,
        mask_fn: Callable[[Image.Image], Image.Image] | None = None,
    ) -> None:
        mask_op = mask_fn or fn
        new_layers = [
            layer.copy(image=ensure_rgba(fn(layer.image)), mask=mask_op(layer.mask) if layer.mask is not None else None)
            for layer in self.layers
        ]
        sizes = {layer.size for layer in new_layers}
        if any(layer.mask is not None and layer.mask.size != layer.size for layer in new_layers):
            raise DocumentError("transform produced inconsistent mask sizes")
        if len(sizes) != 1:
            raise DocumentError("transform produced inconsistent layer sizes")
        self.width, self.height = _check_size(*new_layers[0].size)
        self.layers = new_layers
        if self.selection is not None and keep_selection:
            self.selection = self.selection.transformed(mask_fn or fn)
            if self.selection.size != self.size:
                self.selection = None
        elif not keep_selection:
            self.selection = None
        self.commit(label)

    def rotate90(self, clockwise: bool = True) -> None:
        method = Image.Transpose.ROTATE_270 if clockwise else Image.Transpose.ROTATE_90
        self._transform_all(lambda im: im.transpose(method), "Rotate 90° CW" if clockwise else "Rotate 90° CCW")

    def rotate180(self) -> None:
        self._transform_all(lambda im: im.transpose(Image.Transpose.ROTATE_180), "Rotate 180°")

    def flip(self, horizontal: bool = True) -> None:
        method = Image.Transpose.FLIP_LEFT_RIGHT if horizontal else Image.Transpose.FLIP_TOP_BOTTOM
        self._transform_all(lambda im: im.transpose(method), "Flip Horizontal" if horizontal else "Flip Vertical")

    def rotate(self, angle: float, expand: bool = True, fill: tuple | None = None) -> None:
        """Rotate by an arbitrary angle (counter-clockwise, degrees)."""
        angle = float(angle) % 360
        if angle == 0:
            return
        color = (0, 0, 0, 0) if fill is None else parse_color(fill)
        self._transform_all(
            lambda im: im.rotate(angle, resample=Image.Resampling.BICUBIC, expand=expand, fillcolor=color if im.mode == "RGBA" else 0),
            f"Rotate {angle:g}°",
            keep_selection=False,
        )

    def straighten(self, angle: float, crop: bool = True, fill: tuple | None = None) -> None:
        """Rotate by a small *angle* (degrees, counter-clockwise) to level the horizon.

        With ``crop=True`` the result is cropped to the largest upright rectangle
        that contains only picture (no empty corners); otherwise the canvas grows
        to hold the whole rotated picture.
        """
        angle = float(angle)
        if abs(angle) < 1e-6:
            return
        if not crop:
            self.rotate(angle, expand=True, fill=fill)
            self._record(f"Straighten {angle:+.1f}°", replace=True)
            return
        box = inscribed_box(self.width, self.height, angle)
        color = (0, 0, 0, 0) if fill is None else parse_color(fill)

        def op(im: Image.Image) -> Image.Image:
            fillcolor = color if im.mode == "RGBA" else 0
            return im.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=fillcolor).crop(box)

        self._transform_all(op, f"Straighten {angle:+.1f}°", keep_selection=False)

    def resize(self, width: int, height: int, resample: Image.Resampling = Image.Resampling.LANCZOS) -> None:
        width, height = _check_size(width, height)
        if (width, height) == self.size:
            return
        self._transform_all(
            lambda im: im.resize((width, height), resample),
            f"Image Size {width}×{height}",
            # masks are resampled bilinearly: Lanczos ringing would leak selection outside its bounds
            mask_fn=lambda m: m.resize((width, height), Image.Resampling.BILINEAR),
        )

    def scale(self, factor: float) -> None:
        self.resize(max(1, round(self.width * factor)), max(1, round(self.height * factor)))

    def canvas_size(self, width: int, height: int, anchor: str = "center", fill: tuple | None = None) -> None:
        """Change the canvas size without scaling content; *anchor* says where the old content goes."""
        width, height = _check_size(width, height)
        if anchor not in ANCHORS:
            raise DocumentError(f"anchor must be one of {ANCHORS}")
        dx, dy = width - self.width, height - self.height
        horizontal = "" if anchor == "center" else anchor[-1]
        vertical = "" if anchor == "center" else anchor[0]
        col = 0 if horizontal == "w" else dx if horizontal == "e" else dx // 2
        row = 0 if vertical == "n" else dy if vertical == "s" else dy // 2
        color = (0, 0, 0, 0) if fill is None else parse_color(fill)

        def op(im: Image.Image) -> Image.Image:
            mode = im.mode
            out = Image.new(mode, (width, height), color if mode == "RGBA" else 0)
            out.paste(im, (col, row))
            return out

        self._transform_all(op, f"Canvas Size {width}×{height}")

    def crop(self, box: Sequence[float]) -> None:
        x0, y0, x1, y1 = box
        left, right = sorted((int(round(x0)), int(round(x1))))
        top, bottom = sorted((int(round(y0)), int(round(y1))))
        left, top = max(0, left), max(0, top)
        right, bottom = min(self.width, right), min(self.height, bottom)
        if right - left < 1 or bottom - top < 1:
            raise DocumentError("crop area is empty")
        self._transform_all(lambda im: im.crop((left, top, right, bottom)), "Crop")

    def crop_to_selection(self) -> None:
        if self.selection is None or self.selection.is_empty:
            raise DocumentError("nothing selected")
        bbox = self.selection.bbox
        self.crop(bbox)  # type: ignore[arg-type]
        self.selection = None
        self._record("Crop to Selection", replace=True)

    def trim(self) -> bool:
        """Crop away fully transparent borders; returns False when there was nothing to trim."""
        bbox = self.flatten().getchannel("A").getbbox()
        if bbox is None or bbox == (0, 0, self.width, self.height):
            return False
        self.crop(bbox)
        self._record("Trim", replace=True)
        return True

    # ------------------------------------------------------------ selection --
    def set_selection(self, selection: Selection | None, mode: str = "replace", commit: bool = True) -> None:
        if selection is not None and selection.size != self.size:
            raise DocumentError("selection size must match the document")
        if selection is None:
            self.selection = None
        elif self.selection is None or mode == "replace":
            self.selection = None if selection.is_empty else selection
        else:
            combined = self.selection.combine(selection, mode)
            self.selection = None if combined.is_empty else combined
        if commit:
            self.commit("Selection" if self.selection is not None else "Deselect")

    def select_all(self) -> None:
        self.set_selection(Selection.all(self.size))

    def deselect(self) -> None:
        if self.selection is not None:
            self.set_selection(None)

    def invert_selection(self) -> None:
        current = self.selection or Selection.empty(self.size)
        self.set_selection(current.invert())
        self._record("Invert Selection", replace=True)

    def modify_selection(self, op: str, amount: float) -> None:
        if self.selection is None:
            raise DocumentError("nothing selected")
        if op == "feather":
            new = self.selection.feather(amount)
        elif op == "grow":
            new = self.selection.grow(int(amount))
        elif op == "shrink":
            new = self.selection.shrink(int(amount))
        elif op == "border":
            new = self.selection.border(int(amount))
        else:
            raise DocumentError(f"unknown selection operation {op!r}")
        self.set_selection(new)
        self._record(f"{op.title()} Selection", replace=True)

    def magic_wand(
        self, x: float, y: float, tolerance: int = 32, contiguous: bool = True, merged: bool = True, mode: str = "replace"
    ) -> None:
        ix, iy = int(x), int(y)
        if not (0 <= ix < self.width and 0 <= iy < self.height):
            return
        source = self.flatten() if merged else self.active_layer.image
        self.set_selection(Selection.magic_wand(source, (ix, iy), tolerance, contiguous), mode)
        self._record("Magic Wand", replace=True)

    def select_layer_alpha(self, index: int | None = None, mode: str = "replace") -> None:
        index = self._check_index(index)
        self.set_selection(Selection.from_alpha(self.layers[index].image), mode)

    # ------------------------------------------------------------ clipboard --
    def copy_selection(self, merged: bool = False) -> tuple[Image.Image, tuple[int, int]] | None:
        """Return (image, (x, y)) of the selected pixels (whole layer if no selection)."""
        source = self.flatten() if merged else self.active_layer.image
        if self.selection is None:
            return source.copy(), (0, 0)
        bbox = self.selection.bbox
        if bbox is None:
            return None
        piece = source.crop(bbox)
        alpha = piece.getchannel("A")
        piece.putalpha(ImageChops.multiply(alpha, self.selection.mask.crop(bbox)))
        return piece, (bbox[0], bbox[1])

    def paste(self, image: Image.Image, position: tuple[int, int] | None = None, name: str = "Pasted") -> Layer:
        return self.add_image_layer(image, name, position)

    def bucket_fill(self, x: float, y: float, color, tolerance: int = 32, contiguous: bool = True, merged: bool = False) -> None:
        """Paint-bucket: fill the colour-similar region around (x, y)."""
        ix, iy = int(x), int(y)
        if not (0 <= ix < self.width and 0 <= iy < self.height):
            return
        source = self.flatten() if merged else self.active_layer.image
        region = Selection.magic_wand(source, (ix, iy), tolerance, contiguous)
        mask = region.mask
        if self.selection is not None:
            mask = ImageChops.multiply(mask, self.selection.mask)
        self.apply_to_active(lambda im: painting.fill(im, color, mask), "Paint Bucket", respect_selection=False)
