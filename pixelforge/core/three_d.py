"""3D objects: primitive meshes, extrusions of pictures, and a small software renderer.

A :class:`Scene` describes one object (shape, size, rotation, position, colour,
light) and is stored in a layer's ``meta["3d"]`` so the object stays editable.
:func:`render` turns it into an RGBA picture of the document's size with a
z-buffer rasterizer written in NumPy: perspective projection, smooth
(Blinn-Phong) shading with per-pixel normals, optional texture on the caps of
an extrusion (that is how a text layer becomes 3D text), supersampled
anti-aliasing and an optional wireframe.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .color import parse_color

SHAPES: tuple[str, ...] = ("cube", "sphere", "cylinder", "cone", "torus", "pyramid", "extrusion")
Array = np.ndarray


@dataclass(frozen=True)
class Scene:
    shape: str = "cube"
    size: float = 220.0  # nominal diameter in document pixels
    depth: int = 40  # extrusion depth in % of the size
    rotation: tuple[float, float, float] = (-20.0, 30.0, 0.0)  # degrees about x, y, z
    position: tuple[float, float] = (0.0, 0.0)  # centre in document pixels (0, 0 = document centre)
    color: tuple[int, int, int] = (82, 140, 255)
    light: tuple[float, float] = (-40.0, 35.0)  # azimuth, elevation in degrees
    ambient: int = 25  # %
    shininess: int = 40  # 1..128
    specular: int = 45  # %
    wireframe: bool = False
    segments: int = 32
    texture: str | None = None  # base64 PNG (RGBA) painted on the caps of an extrusion
    mask: str | None = None  # base64 PNG (L) - the outline that is extruded
    hollow: bool = False

    def normalized(self) -> Scene:
        return replace(
            self,
            shape=self.shape if self.shape in SHAPES else "cube",
            size=float(min(20000.0, max(4.0, self.size))),
            depth=int(min(400, max(1, self.depth))),
            rotation=tuple(float(r) % 360 for r in self.rotation),  # type: ignore[arg-type]
            position=(float(self.position[0]), float(self.position[1])),
            color=tuple(int(min(255, max(0, c))) for c in parse_color(self.color)[:3]),  # type: ignore[arg-type]
            light=(float(self.light[0]) % 360, float(min(89.0, max(-89.0, self.light[1])))),
            ambient=int(min(100, max(0, self.ambient))),
            shininess=int(min(128, max(1, self.shininess))),
            specular=int(min(100, max(0, self.specular))),
            segments=int(min(96, max(6, self.segments))),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["rotation"], data["position"], data["light"] = list(self.rotation), list(self.position), list(self.light)
        data["color"] = list(self.color)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scene:
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        for key in ("rotation", "position", "light", "color"):
            if key in known and isinstance(known[key], list):
                known[key] = tuple(known[key])
        return cls(**known).normalized()

    def mask_image(self) -> Image.Image | None:
        return _decode(self.mask, "L")

    def texture_image(self) -> Image.Image | None:
        return _decode(self.texture, "RGBA")


def _decode(data: str | None, mode: str) -> Image.Image | None:
    if not data:
        return None
    try:
        img = Image.open(io.BytesIO(base64.b64decode(data)))
        img.load()
        return img.convert(mode)
    except Exception:  # noqa: BLE001 - broken data simply means "no picture"
        return None


def encode_image(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ------------------------------------------------------------------- meshes --
@dataclass
class Mesh:
    """Triangles in object space (the object roughly fills the unit cube, -0.5 … 0.5)."""

    vertices: Array  # (N, 3) float32
    faces: Array  # (M, 3) int32
    uvs: Array | None = None  # (N, 2) texture coordinates in 0..1 for textured vertices (NaN elsewhere)
    normals: Array = field(default_factory=lambda: np.zeros((0, 3), np.float32))

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, np.float32)
        self.faces = np.asarray(self.faces, np.int32).reshape(-1, 3)
        if self.normals.shape != self.vertices.shape:
            self.normals = vertex_normals(self.vertices, self.faces)


def vertex_normals(vertices: Array, faces: Array) -> Array:
    """Area-weighted average of the adjacent face normals (shared vertices shade smoothly, duplicated ones stay flat)."""
    normals = np.zeros_like(vertices, dtype=np.float32)
    if len(faces):
        a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        face_n = np.cross(b - a, c - a)
        for i in range(3):
            np.add.at(normals, faces[:, i], face_n)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.maximum(length, 1e-9)).astype(np.float32)


def _ring(n: int, radius: float, y: float, phase: float = 0.0) -> Array:
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False) + phase
    return np.stack([np.cos(angles) * radius, np.full(n, y), np.sin(angles) * radius], axis=1).astype(np.float32)


def _quad(a: int, b: int, c: int, d: int) -> list[list[int]]:
    return [[a, b, c], [a, c, d]]


def cube_mesh() -> Mesh:
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            base = len(verts)
            u, v = [i for i in range(3) if i != axis]
            for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = [0.0, 0.0, 0.0]
                p[axis] = 0.5 * sign
                p[u], p[v] = 0.5 * su, 0.5 * sv
                verts.append(p)
            order = _quad(base, base + 1, base + 2, base + 3)
            if sign * (1 if axis != 1 else -1) > 0:
                order = [f[::-1] for f in order]
            faces += order
    return Mesh(np.array(verts), np.array(faces))


def sphere_mesh(segments: int) -> Mesh:
    rings, slices = max(4, segments // 2), segments
    verts = []
    for i in range(rings + 1):
        phi = math.pi * i / rings
        y = math.cos(phi) * 0.5
        r = math.sin(phi) * 0.5
        for j in range(slices):
            theta = 2 * math.pi * j / slices
            verts.append([math.cos(theta) * r, y, math.sin(theta) * r])
    faces = []
    for i in range(rings):
        for j in range(slices):
            a, b = i * slices + j, i * slices + (j + 1) % slices
            c, d = (i + 1) * slices + (j + 1) % slices, (i + 1) * slices + j
            faces += _quad(a, b, c, d)
    return Mesh(np.array(verts), np.array(faces))


def cylinder_mesh(segments: int, top_radius: float = 0.5) -> Mesh:
    """A cylinder (or a cone / frustum when *top_radius* differs) with separately shaded caps."""
    bottom, top = _ring(segments, 0.5, -0.5), _ring(segments, max(0.0, top_radius), 0.5)
    verts = [bottom, top]
    faces: list[list[int]] = []
    for j in range(segments):
        a, b = j, (j + 1) % segments
        faces += _quad(a, b, segments + b, segments + a)
    # caps with their own vertices (flat shading)
    base = 2 * segments
    verts += [bottom.copy(), np.array([[0.0, -0.5, 0.0]], np.float32)]
    for j in range(segments):
        faces.append([base + segments, base + (j + 1) % segments, base + j])
    if top_radius > 1e-6:
        base2 = base + segments + 1
        verts += [top.copy(), np.array([[0.0, 0.5, 0.0]], np.float32)]
        for j in range(segments):
            faces.append([base2 + segments, base2 + j, base2 + (j + 1) % segments])
    return Mesh(np.concatenate(verts), np.array(faces))


def torus_mesh(segments: int, tube: float = 0.18) -> Mesh:
    major = 0.5 - tube
    tube_segments = max(6, segments // 2)
    verts = []
    for i in range(segments):
        u = 2 * math.pi * i / segments
        for j in range(tube_segments):
            v = 2 * math.pi * j / tube_segments
            r = major + tube * math.cos(v)
            verts.append([math.cos(u) * r, tube * math.sin(v), math.sin(u) * r])
    faces = []
    for i in range(segments):
        for j in range(tube_segments):
            a = i * tube_segments + j
            b = ((i + 1) % segments) * tube_segments + j
            c = ((i + 1) % segments) * tube_segments + (j + 1) % tube_segments
            d = i * tube_segments + (j + 1) % tube_segments
            faces += _quad(a, d, c, b)
    return Mesh(np.array(verts), np.array(faces))


def pyramid_mesh() -> Mesh:
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    apex = [0.0, 0.5, 0.0]
    corners = [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5]]
    for i in range(4):
        base = len(verts)
        verts += [corners[i], corners[(i + 1) % 4], apex]
        faces.append([base, base + 2, base + 1])
    base = len(verts)
    verts += corners
    faces += _quad(base, base + 1, base + 2, base + 3)
    return Mesh(np.array(verts), np.array(faces))


def _runs(row: Array) -> list[tuple[int, int]]:
    """``(start, end)`` pairs of consecutive True values in a 1-D boolean array."""
    padded = np.concatenate([[False], row, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(edges[i]), int(edges[i + 1])) for i in range(0, len(edges), 2)]


def _rectangles(arr: Array) -> list[tuple[int, int, int, int]]:
    """Cover the True pixels of *arr* with rectangles: runs of a row merge with identical runs in the rows below."""
    h = arr.shape[0]
    open_rects: dict[tuple[int, int], int] = {}  # (x0, x1) → y0 of a run still growing downwards
    rects: list[tuple[int, int, int, int]] = []
    for y in range(h + 1):
        runs = set(_runs(arr[y])) if y < h else set()
        for key in list(open_rects):
            if key not in runs:
                rects.append((key[0], open_rects.pop(key), key[1], y))
        for key in runs:
            open_rects.setdefault(key, y)
    return rects


def extrusion_mesh(mask: Image.Image, depth: float) -> Mesh:
    """Extrude the opaque area of *mask* (mode L) by *depth* (fraction of the mask's larger side).

    The front and back caps are rectangles covering the opaque pixels and carry
    texture coordinates; the sides follow the pixel outline with one quad per
    straight boundary segment.  Everything is scaled so the mask's larger side
    spans 1.0, centred on the origin.
    """
    arr = np.asarray(mask.convert("L")) > 127
    h, w = arr.shape
    scale = 1.0 / max(w, h, 1)
    half = depth / 2
    verts: list[list[float]] = []
    uvs: list[list[float]] = []
    faces: list[list[int]] = []

    def vertex(px_: float, py_: float, z: float, textured: bool) -> int:
        verts.append([(px_ - w / 2) * scale, (h / 2 - py_) * scale, z])
        uvs.append([px_ / w, py_ / h] if textured else [math.nan, math.nan])
        return len(verts) - 1

    for x0, y0, x1, y1 in _rectangles(arr):  # caps: front (z = -half, towards the viewer) and back
        for z, flip in ((-half, False), (half, True)):
            a, b = vertex(x0, y0, z, True), vertex(x1, y0, z, True)
            c, d = vertex(x1, y1, z, True), vertex(x0, y1, z, True)
            quad = _quad(a, b, c, d)
            faces += [f[::-1] for f in quad] if flip else quad

    def side_vertex(px_: int, py_: int, z: float) -> int:
        return vertex(px_, py_, z, False)  # own vertices per segment → flat-shaded, crisp sides

    padded = np.pad(arr, 1)
    # boundary edges of each kind, merged into straight segments
    left = arr & ~padded[1:-1, :-2]
    right = arr & ~padded[1:-1, 2:]
    top = arr & ~padded[:-2, 1:-1]
    bottom = arr & ~padded[2:, 1:-1]
    for x in range(w):  # vertical segments (left / right edges) run along y
        for y0, y1 in _runs(left[:, x]):
            a, b = side_vertex(x, y1, -half), side_vertex(x, y0, -half)
            faces += _quad(a, b, side_vertex(x, y0, half), side_vertex(x, y1, half))
        for y0, y1 in _runs(right[:, x]):
            a, b = side_vertex(x + 1, y0, -half), side_vertex(x + 1, y1, -half)
            faces += _quad(a, b, side_vertex(x + 1, y1, half), side_vertex(x + 1, y0, half))
    for y in range(h):  # horizontal segments (top / bottom edges) run along x
        for x0, x1 in _runs(top[y]):
            a, b = side_vertex(x0, y, -half), side_vertex(x1, y, -half)
            faces += _quad(a, b, side_vertex(x1, y, half), side_vertex(x0, y, half))
        for x0, x1 in _runs(bottom[y]):
            a, b = side_vertex(x1, y + 1, -half), side_vertex(x0, y + 1, -half)
            faces += _quad(a, b, side_vertex(x0, y + 1, half), side_vertex(x1, y + 1, half))
    if not faces:
        return cube_mesh()
    return Mesh(np.array(verts, np.float32), np.array(faces), np.array(uvs, np.float32))


def make_mesh(scene: Scene) -> Mesh:
    shape = scene.shape
    if shape == "sphere":
        return sphere_mesh(scene.segments)
    if shape == "cylinder":
        return cylinder_mesh(scene.segments)
    if shape == "cone":
        return cylinder_mesh(scene.segments, 0.0)
    if shape == "torus":
        return torus_mesh(scene.segments)
    if shape == "pyramid":
        return pyramid_mesh()
    if shape == "extrusion":
        mask = scene.mask_image()
        if mask is not None:
            return extrusion_mesh(mask, scene.depth / 100)
    return cube_mesh()


# ----------------------------------------------------------------- renderer --
def rotation_matrix(rx: float, ry: float, rz: float) -> Array:
    ax, ay, az = (math.radians(v) for v in (rx, ry, rz))
    cx, sx, cy, sy, cz, sz = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (mz @ my @ mx).astype(np.float32)


def light_direction(azimuth: float, elevation: float) -> Array:
    """Unit vector *towards* the light in view space (x right, y up, z towards the viewer)."""
    a, e = math.radians(azimuth), math.radians(elevation)
    return np.array([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)], np.float32)


def project(vertices: Array, scene: Scene, size: tuple[int, int], scale: float) -> tuple[Array, Array, Array]:
    """Object → view space → screen. Returns screen xy (N, 2), view-space depth (N,), view-space positions (N, 3)."""
    rot = rotation_matrix(*scene.rotation)
    view = vertices @ rot.T * np.float32(scene.size)  # in document pixels, centred on the object
    focal = scene.size * 4.0
    # the viewer sits at +z; points with larger z are closer
    depth = view[:, 2]
    f = focal / np.maximum(focal - depth, focal * 0.2)
    cx = (size[0] / 2 + scene.position[0]) * scale
    cy = (size[1] / 2 + scene.position[1]) * scale
    sx = cx + view[:, 0] * f * scale
    sy = cy - view[:, 1] * f * scale
    return np.stack([sx, sy], axis=1).astype(np.float32), depth.astype(np.float32), view.astype(np.float32)


def rasterize(mesh: Mesh, screen: Array, depth: Array, normals: Array, size: tuple[int, int]) -> dict[str, Array]:
    """Z-buffer rasterization: per-pixel hit mask, interpolated normals and (where textured) uv coordinates."""
    w, h = size
    zbuf = np.full((h, w), -np.inf, np.float32)
    nbuf = np.zeros((h, w, 3), np.float32)
    has_uv = mesh.uvs is not None
    uvbuf = np.full((h, w, 2), np.nan, np.float32) if has_uv else np.zeros((0, 0, 2), np.float32)
    faces = mesh.faces
    tri = screen[faces]  # (M, 3, 2)
    area = (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1]) - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1])
    for m in range(len(faces)):
        a2 = area[m]
        if abs(a2) < 1e-6:
            continue
        pts = tri[m]
        x0, x1 = int(math.floor(pts[:, 0].min())), int(math.ceil(pts[:, 0].max()))
        y0, y1 = int(math.floor(pts[:, 1].min())), int(math.ceil(pts[:, 1].max()))
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1 + 1), min(h, y1 + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        px_ = (np.arange(x0, x1, dtype=np.float32) + 0.5)[None, :]
        py_ = (np.arange(y0, y1, dtype=np.float32) + 0.5)[:, None]
        (ax, ay), (bx, by), (cx, cy) = pts
        inv = 1.0 / a2
        w0 = ((bx - px_) * (cy - py_) - (cx - px_) * (by - py_)) * inv
        w1 = ((cx - px_) * (ay - py_) - (ax - px_) * (cy - py_)) * inv
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not inside.any():
            continue
        idx = faces[m]
        z = w0 * depth[idx[0]] + w1 * depth[idx[1]] + w2 * depth[idx[2]]
        region = zbuf[y0:y1, x0:x1]
        hit = inside & (z > region)
        if not hit.any():
            continue
        np.copyto(region, z, where=hit)
        rows, cols = np.nonzero(hit)
        w0h, w1h, w2h = w0[rows, cols], w1[rows, cols], w2[rows, cols]
        n = np.outer(w0h, normals[idx[0]]) + np.outer(w1h, normals[idx[1]]) + np.outer(w2h, normals[idx[2]])
        nbuf[rows + y0, cols + x0] = n
        if has_uv:
            assert mesh.uvs is not None
            uv = np.outer(w0h, mesh.uvs[idx[0]]) + np.outer(w1h, mesh.uvs[idx[1]]) + np.outer(w2h, mesh.uvs[idx[2]])
            uvbuf[rows + y0, cols + x0] = uv
    return {"z": zbuf, "normals": nbuf, "uv": uvbuf, "hit": np.isfinite(zbuf)}


def shade(buffers: dict[str, Array], scene: Scene, texture: Image.Image | None) -> Array:
    """Blinn-Phong shading of the rasterized buffers → float RGBA (H, W, 4) in 0..1."""
    hit = buffers["hit"]
    h, w = hit.shape
    out = np.zeros((h, w, 4), np.float32)
    flat = np.flatnonzero(hit)
    if not flat.size:
        return out
    n = buffers["normals"].reshape(-1, 3)[flat]
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-6)
    light = light_direction(*scene.light)
    view_dir = np.array([0.0, 0.0, 1.0], np.float32)
    n = np.where((n @ view_dir)[:, None] < 0, -n, n)  # back faces that show through shade like front faces
    diffuse = np.clip(n @ light, 0, 1)
    half_vec = light + view_dir
    half_vec /= np.linalg.norm(half_vec)
    spec = np.clip(n @ half_vec, 0, 1) ** float(scene.shininess) * (scene.specular / 100)
    base = np.tile(np.array(scene.color, np.float32) / 255, (len(n), 1))
    if texture is not None and buffers["uv"].size:
        uv = buffers["uv"].reshape(-1, 2)[flat]
        textured = np.isfinite(uv[:, 0])
        if textured.any():
            tex = np.asarray(texture.convert("RGBA"), np.float32) / 255
            th, tw = tex.shape[:2]
            tx = np.clip((uv[textured, 0] * tw).astype(int), 0, tw - 1)
            ty = np.clip((uv[textured, 1] * th).astype(int), 0, th - 1)
            sample = tex[ty, tx]
            alpha = sample[:, 3:4]
            base[textured] = sample[:, :3] * alpha + base[textured] * (1 - alpha)
    ambient = scene.ambient / 100
    rgb = base * (ambient + (1 - ambient) * diffuse[:, None]) + spec[:, None]
    flat_out = out.reshape(-1, 4)
    flat_out[flat, :3] = np.clip(rgb, 0, 1)
    flat_out[flat, 3] = 1.0
    return out


def render(scene: Scene, size: tuple[int, int], supersample: int = 2, scale: float = 1.0) -> Image.Image:
    """Render *scene* into an RGBA picture of *size* document pixels (drawn at *scale*, anti-aliased by *supersample*)."""
    scene = scene.normalized()
    mesh = make_mesh(scene)
    ss = max(1, int(supersample))
    out_w, out_h = max(1, int(round(size[0] * scale))), max(1, int(round(size[1] * scale)))
    buf_size = (out_w * ss, out_h * ss)
    screen, depth, view = project(mesh.vertices, scene, size, scale * ss)
    rot = rotation_matrix(*scene.rotation)
    normals = (mesh.normals @ rot.T).astype(np.float32)
    img = Image.new("RGBA", buf_size, (0, 0, 0, 0))
    # work only on the part of the buffer the object covers - cost follows the object, not the document
    rx0 = max(0, int(math.floor(screen[:, 0].min())) - 1)
    ry0 = max(0, int(math.floor(screen[:, 1].min())) - 1)
    rx1 = min(buf_size[0], int(math.ceil(screen[:, 0].max())) + 2)
    ry1 = min(buf_size[1], int(math.ceil(screen[:, 1].max())) + 2)
    if rx1 > rx0 and ry1 > ry0:
        local = screen - np.array([rx0, ry0], np.float32)
        buffers = rasterize(mesh, local, depth, normals, (rx1 - rx0, ry1 - ry0))
        rgba = shade(buffers, scene, scene.texture_image() if scene.shape == "extrusion" else None)
        img.paste(Image.fromarray(np.clip(rgba * 255 + 0.5, 0, 255).astype(np.uint8), "RGBA"), (rx0, ry0))
    if scene.wireframe:
        draw = ImageDraw.Draw(img)
        pts = screen
        for f in mesh.faces:
            poly = [(float(pts[i, 0]), float(pts[i, 1])) for i in f]
            draw.polygon(poly, outline=(255, 255, 255, 230), width=ss)
    if ss > 1:
        img = img.resize((out_w, out_h), Image.Resampling.LANCZOS)
    return img


def bounding_box(scene: Scene, size: tuple[int, int]) -> tuple[float, float, float, float]:
    """Screen-space bounding box of the object in document pixels (for hit-testing and handles)."""
    mesh = make_mesh(scene.normalized())
    screen, _depth, _view = project(mesh.vertices, scene.normalized(), size, 1.0)
    return float(screen[:, 0].min()), float(screen[:, 1].min()), float(screen[:, 0].max()), float(screen[:, 1].max())


def extrusion_scene(layer_image: Image.Image, depth: int = 30, color: tuple[int, int, int] | None = None) -> Scene:
    """A scene extruding the opaque pixels of *layer_image*, textured with the picture itself."""
    rgba = layer_image.convert("RGBA")
    alpha = rgba.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        raise ValueError("the layer has no opaque pixels to extrude")
    piece = rgba.crop(bbox)
    # keep the mesh manageable: at most ~220 px on the longer side for the outline
    longest = max(piece.size)
    factor = min(1.0, 220 / longest)
    mask = piece.getchannel("A").resize(
        (max(1, round(piece.width * factor)), max(1, round(piece.height * factor))), Image.Resampling.BILINEAR
    )
    texture = piece.copy()
    texture.thumbnail((512, 512))
    if color is None:
        arr = np.asarray(piece, np.float32)
        weights = arr[..., 3:4] / 255
        mean = (arr[..., :3] * weights).sum(axis=(0, 1)) / max(1.0, float(weights.sum()))
        color = (int(mean[0]), int(mean[1]), int(mean[2]))
    center_x = (bbox[0] + bbox[2]) / 2 - layer_image.width / 2
    center_y = (bbox[1] + bbox[3]) / 2 - layer_image.height / 2
    return Scene(
        shape="extrusion",
        size=float(longest),
        depth=depth,
        rotation=(-15.0, 25.0, 0.0),
        position=(center_x, center_y),
        color=color,
        mask=encode_image(mask),
        texture=encode_image(texture),
    ).normalized()
