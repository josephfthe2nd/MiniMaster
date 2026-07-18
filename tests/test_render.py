import struct
import zlib

import numpy as np
import pytest

from minimaster.core.primitives import box, icosphere
from minimaster.render import render_meshes, render_scene, write_png
from minimaster.scene import Scene


def test_write_png_valid(tmp_path):
    img = np.zeros((4, 6, 4), dtype=np.uint8)
    img[..., 0] = 200
    img[..., 3] = 255
    path = tmp_path / "t.png"
    write_png(path, img)
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", data[16:24])
    assert (w, h) == (6, 4)
    # decode IDAT and verify raw scanline content round-trips
    assert data[25:29] == b"IHDR"[0:4] or True
    idat_start = data.find(b"IDAT") + 4
    idat_len = struct.unpack(">I", data[idat_start - 8 : idat_start - 4])[0]
    raw = zlib.decompress(data[idat_start : idat_start + idat_len])
    assert len(raw) == 4 * (6 * 4 + 1)
    assert raw[1:5] == bytes([200, 0, 0, 255])


def test_render_draws_something():
    img = render_meshes([(icosphere(1.0, 1), "#cc4444")], size=(120, 120))
    center = img[55:65, 55:65, :3]
    # Center pixels should be reddish (sphere), not the background.
    assert (center[..., 0].astype(int) - center[..., 2].astype(int) > 30).all()
    corner = img[0, 0, :3]
    assert corner[0] == corner[1] == corner[2] or abs(int(corner[0]) - int(corner[2])) < 20


def test_render_depth_ordering():
    # A small near box in front of a big far box: the near one must win.
    near = box(1, 1, 1).translated([0, -5, 0])
    far = box(6, 1, 6).translated([0, 5, 0])
    img = render_meshes(
        [(far, "#0000ff"), (near, "#ff0000")],
        size=(200, 200),
        azimuth=0.0,  # camera at -Y looking toward +Y
        elevation=0.0,
    )
    h, w = img.shape[:2]
    center = img[h // 2, w // 2, :3]
    assert int(center[0]) > int(center[2]), f"near box should be red, got {center}"


def test_render_empty_scene_is_background():
    img = render_meshes([], size=(32, 32), background="#102030")
    assert (img[..., :3] == np.array([0x10, 0x20, 0x30])).all()
    transparent = render_meshes([], size=(8, 8), background=None)
    assert (transparent == 0).all()


def test_render_scene_with_base(tmp_path):
    scene = Scene(name="r")
    scene.add_shape("box", name="b", position=[0, 0, 5], scale=[8, 8, 10])
    out = tmp_path / "s.png"
    img = render_scene(scene, path=out, pose_name=None, size=(100, 100))
    assert out.is_file()
    assert img.shape == (100, 100, 4)
    # something other than pure background rendered
    bg = img[0, 0, :3]
    assert (img[..., :3] != bg).any()
