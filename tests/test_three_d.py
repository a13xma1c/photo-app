"""3D objects: meshes, projection, the rasterizer / shader, extrusions and 3D layers in the document."""

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixelforge.core import io as pfio
from pixelforge.core import three_d as t3
from pixelforge.core.document import Document, DocumentError
from pixelforge.core.painting import render_text
from tests.helpers import pixel


def test_scene_normalisation_and_serialisation() -> None:
    wild = t3.Scene(shape="blob", size=1.0, depth=0, rotation=(400.0, -30.0, 720.0), color=(300, -5, 10), light=(370.0, 95.0), ambient=150)
    n = wild.normalized()
    assert n.shape == "cube" and n.size == 4.0 and n.depth == 1 and n.rotation == (40.0, 330.0, 0.0)
    assert n.color == (255, 0, 10) and n.light == (10.0, 89.0) and n.ambient == 100
    data = n.to_dict()
    assert isinstance(data["rotation"], list) and t3.Scene.from_dict({**data, "unknown": 1}) == n
    assert t3.Scene.from_dict({"shape": "sphere", "color": [1, 2, 3]}).color == (1, 2, 3)
    assert t3.Scene().mask_image() is None and t3.Scene(texture="!!").texture_image() is None
    encoded = t3.encode_image(Image.new("L", (4, 4), 200))
    assert t3.Scene(mask=encoded).mask_image().getpixel((1, 1)) == 200  # type: ignore[union-attr]


@pytest.mark.parametrize("shape", [s for s in t3.SHAPES if s != "extrusion"])
def test_primitive_meshes_are_sane(shape: str) -> None:
    mesh = t3.make_mesh(t3.Scene(shape=shape, segments=12))
    assert mesh.faces.min() >= 0 and mesh.faces.max() < len(mesh.vertices)
    assert np.all(np.abs(mesh.vertices) <= 0.5 + 1e-6)  # fits the unit cube
    assert mesh.normals.shape == mesh.vertices.shape
    lengths = np.linalg.norm(mesh.normals, axis=1)
    assert np.allclose(lengths[lengths > 0], 1.0, atol=1e-5)
    # the mesh is closed enough to render from the front: its picture is not empty
    img = t3.render(t3.Scene(shape=shape, size=60, segments=12), (100, 100), supersample=1)
    assert img.size == (100, 100) and img.getbbox() is not None


def test_normals_flat_versus_smooth() -> None:
    cube = t3.cube_mesh()
    # a cube face's four vertices share one normal (flat), pointing outwards along an axis
    n = cube.normals[cube.faces[0]]
    assert np.allclose(n, n[0]) and math.isclose(abs(n[0]).max(), 1.0, abs_tol=1e-6)
    sphere = t3.sphere_mesh(16)
    # on a sphere the smooth normal points away from the centre
    v = sphere.vertices[40]
    assert np.dot(sphere.normals[40], v / np.linalg.norm(v)) > 0.95


def test_rotation_and_light_vectors() -> None:
    assert np.allclose(t3.rotation_matrix(0, 0, 0), np.eye(3))
    r = t3.rotation_matrix(0, 90, 0)
    assert np.allclose(r @ np.array([1.0, 0, 0]), [0, 0, -1], atol=1e-6)
    light = t3.light_direction(0, 0)
    assert np.allclose(light, [0, 0, 1]) and math.isclose(float(np.linalg.norm(t3.light_direction(33, 21))), 1.0, abs_tol=1e-6)
    assert t3.light_direction(0, 90)[1] == pytest.approx(1.0)


def test_projection_perspective_and_bounding_box() -> None:
    scene = t3.Scene(shape="cube", size=100, rotation=(0, 0, 0), position=(10, -20))
    mesh = t3.cube_mesh()
    screen, depth, view = t3.project(mesh.vertices, scene, (400, 300), 1.0)
    # the object is centred on the document centre plus its position
    assert screen[:, 0].mean() == pytest.approx(210, abs=1) and screen[:, 1].mean() == pytest.approx(130, abs=1)
    near = screen[depth > 0]
    far = screen[depth < 0]
    assert np.ptp(near[:, 0]) > np.ptp(far[:, 0])  # closer faces look bigger
    box = t3.bounding_box(scene, (400, 300))
    assert box[0] < 210 < box[2] and box[1] < 130 < box[3] and box[2] - box[0] > 100


def test_render_shading_and_options() -> None:
    size = (120, 120)
    sphere = t3.render(t3.Scene(shape="sphere", size=80, color=(255, 0, 0), light=(0.0, 0.0), ambient=20, specular=0), size)
    centre = pixel(sphere, 60, 60)
    edge = pixel(sphere, 60, 24)
    assert centre[3] == 255 and centre[0] > edge[0] > 0 and centre[1] == 0 and pixel(sphere, 2, 2)[3] == 0  # lit from the front
    dark = t3.render(t3.Scene(shape="sphere", size=80, color=(255, 0, 0), light=(0.0, 0.0), ambient=0, specular=0), size)
    assert pixel(dark, 60, 24)[0] < edge[0]  # less ambient: darker rim
    shiny = t3.render(t3.Scene(shape="sphere", size=80, color=(255, 0, 0), light=(0.0, 0.0), ambient=20, specular=100), size)
    assert pixel(shiny, 60, 60)[1] > 50  # the highlight adds white
    wire = t3.render(t3.Scene(shape="cube", size=80, wireframe=True, color=(0, 0, 255)), size)
    assert any(pixel(wire, x, y)[0] > 100 for x in range(20, 100, 2) for y in range(20, 100, 2))  # white lines over blue
    half = t3.render(t3.Scene(shape="cube", size=80), size, supersample=1, scale=0.5)
    assert half.size == (60, 60) and half.getbbox() is not None
    assert t3.render(t3.Scene(shape="cube", size=80, position=(5000, 5000)), size).getbbox() is None  # off-canvas


def test_extrusion_mesh_and_scene() -> None:
    mask = Image.new("L", (20, 10), 0)
    mask.paste(255, (2, 2, 12, 8))  # a 10×6 block
    mesh = t3.extrusion_mesh(mask, 0.3)
    assert mesh.uvs is not None and len(mesh.faces) == 2 * 2 + 4 * 2  # two cap quads + four side quads
    assert np.nanmin(mesh.uvs) >= 0 and np.nanmax(mesh.uvs) <= 1
    assert np.isnan(mesh.uvs[-1]).all()  # side vertices carry no texture
    zs = np.unique(np.round(mesh.vertices[:, 2].astype(np.float64), 4))
    assert zs.tolist() == pytest.approx([-0.15, 0.15])
    # rectangles cover exactly the opaque pixels
    arr = np.asarray(mask) > 127
    rects = t3._rectangles(arr)
    covered = np.zeros_like(arr)
    for x0, y0, x1, y1 in rects:
        covered[y0:y1, x0:x1] = True
    assert np.array_equal(covered, arr) and t3._runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]
    assert isinstance(t3.extrusion_mesh(Image.new("L", (5, 5), 0), 0.3), t3.Mesh)  # empty mask: falls back to a cube
    text = render_text((300, 120), (10, 10), "Hi", "#40a0ff", None, 90)
    scene = t3.extrusion_scene(text, 25)
    assert scene.shape == "extrusion" and scene.mask and scene.texture and scene.color[2] > scene.color[0]
    img = t3.render(scene, (300, 120), supersample=1)
    assert img.getbbox() is not None
    rgb = np.asarray(img)[..., :3][np.asarray(img)[..., 3] > 0]
    assert rgb[:, 2].mean() > rgb[:, 0].mean()  # textured with the blue text
    with pytest.raises(ValueError):
        t3.extrusion_scene(Image.new("RGBA", (10, 10), (0, 0, 0, 0)))


def test_document_3d_layers(tmp_path: Path) -> None:
    doc = Document.new(200, 150, "white")
    layer = doc.add_3d_layer(t3.Scene(shape="sphere", size=100, color=(0, 120, 255)))
    assert layer.kind == "3d" and layer.name == "3D Sphere" and doc.scene_of() is not None
    assert pixel(doc.flatten(), 100, 75)[2] > 150 and doc.history.labels[-1] == "New Layer '3D Sphere'"
    doc.update_3d_layer(color=(255, 0, 0), label="Recolour")
    assert doc.scene_of().color == (255, 0, 0) and pixel(doc.flatten(), 100, 75)[0] > 150  # type: ignore[union-attr]
    assert doc.history.labels[-1] == "Recolour"
    doc.update_3d_layer(scene=t3.Scene(shape="cube", size=40), image=Image.new("RGBA", doc.size, (0, 255, 0, 255)))
    assert pixel(doc.active_layer.image, 5, 5) == (0, 255, 0, 255)  # a supplied picture skips the render
    doc.update_3d_layer(image=Image.new("RGBA", (3, 3)), shape="torus")  # wrong size: rendered instead
    assert doc.active_layer.image.size == doc.size and doc.scene_of().shape == "torus"  # type: ignore[union-attr]
    assert doc.undo() and pixel(doc.active_layer.image, 5, 5) == (0, 255, 0, 255)
    doc.set_active(0)
    assert doc.scene_of() is None
    with pytest.raises(DocumentError):
        doc.update_3d_layer(shape="cube")
    # extrusion of a pixel layer (textured), refused for layers without pixels
    doc.fill("#ff00ff")
    extruded = doc.extrude_layer(depth=20)
    assert extruded.kind == "3d" and doc.scene_of().shape == "extrusion" and extruded.name == "3D Background"  # type: ignore[union-attr]
    doc.add_adjustment_layer()
    with pytest.raises(DocumentError):
        doc.extrude_layer()
    doc.add_layer()
    with pytest.raises(DocumentError):
        doc.extrude_layer()
    # project files keep the scene
    path = tmp_path / "three.pfp"
    pfio.save_project(doc, path)
    loaded = pfio.load_project(path)
    scenes = [loaded.scene_of(i) for i in range(len(loaded.layers)) if loaded.layers[i].kind == "3d"]
    assert len(scenes) == 2 and scenes[0].shape == "extrusion" and scenes[0].texture  # type: ignore[union-attr]
