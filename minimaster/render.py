"""Headless flat-shaded z-buffer renderer producing PNG previews.

No GUI or third-party imaging dependencies: rasterization is numpy, PNG
encoding is stdlib zlib. Used for CLI previews, docs galleries, and tests;
the interactive viewport has its own (faster, rougher) painter's algorithm.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

from .bases import build_base
from .core import math3d as m3
from .core.mesh import Mesh


def _hex_to_rgb(color: str) -> np.ndarray:
    c = color.lstrip("#")
    if len(c) != 6:
        raise ValueError(f"expected #rrggbb color, got {color!r}")
    return np.array([int(c[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float64) / 255.0


def write_png(path, rgba: np.ndarray) -> None:
    """Write an (h, w, 4) uint8 array as a PNG."""
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    h, w = rgba.shape[:2]

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))
    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
    Path(path).write_bytes(data)


def render_meshes(
    colored_meshes: list[tuple[Mesh, str]],
    size: tuple[int, int] = (800, 800),
    azimuth: float = 35.0,
    elevation: float = 22.0,
    background: str | None = "#f2efe9",
    fov: float = 28.0,
    light_dir=(-0.45, -0.6, 0.75),
) -> np.ndarray:
    """Render meshes (each with a #rrggbb color) to an (h, w, 4) uint8 image.

    The camera orbits the combined bounding box center at the given azimuth
    (degrees around +Z from +Y... 0 looks from -Y toward +Y) and elevation.
    """
    w, h = size
    img = np.zeros((h, w, 4), dtype=np.uint8)
    if background is not None:
        img[:, :, :3] = (_hex_to_rgb(background) * 255).astype(np.uint8)
        img[:, :, 3] = 255

    meshes = [(mesh, col) for mesh, col in colored_meshes if len(mesh.faces)]
    if not meshes:
        return img

    combined = Mesh.merge([mesh for mesh, _ in meshes])
    lo, hi = combined.bounds
    center = (lo + hi) / 2.0
    radius = float(np.linalg.norm(hi - lo)) / 2.0
    if radius < 1e-9:
        radius = 1.0

    az, el = np.radians(azimuth), np.radians(elevation)
    distance = radius / np.tan(np.radians(fov) / 2.0) * 1.15
    eye = center + distance * np.array(
        [np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)]
    )
    view = m3.look_at(eye, center)
    focal = (h / 2.0) / np.tan(np.radians(fov) / 2.0)

    light = np.asarray(light_dir, dtype=np.float64)
    light = light / np.linalg.norm(light)

    zbuf = np.full((h, w), np.inf)

    for mesh, color_hex in meshes:
        base_color = _hex_to_rgb(color_hex)
        cam = m3.transform_points(view, mesh.vertices)
        tris = cam[mesh.faces]  # (m, 3, 3)

        normals_world = mesh.face_normals()
        centroids = tris.mean(axis=1)
        normals_cam = normals_world @ view[:3, :3].T
        facing = np.einsum("ij,ij->i", normals_cam, centroids) < 0.0

        shade = 0.30 + 0.70 * np.clip(normals_world @ light, 0.0, None)
        colors = np.clip(base_color[None, :] * shade[:, None], 0.0, 1.0)

        depths = -tris[:, :, 2]
        valid = facing & (depths.min(axis=1) > 1e-6)

        xs = w / 2.0 + focal * tris[:, :, 0] / depths
        ys = h / 2.0 - focal * tris[:, :, 1] / depths

        for i in np.nonzero(valid)[0]:
            tx, ty, tz = xs[i], ys[i], depths[i]
            x0 = max(int(np.floor(tx.min())), 0)
            x1 = min(int(np.ceil(tx.max())) + 1, w)
            y0 = max(int(np.floor(ty.min())), 0)
            y1 = min(int(np.ceil(ty.max())) + 1, h)
            if x0 >= x1 or y0 >= y1:
                continue
            px, py = np.meshgrid(
                np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5
            )
            d = (tx[1] - tx[0]) * (ty[2] - ty[0]) - (tx[2] - tx[0]) * (ty[1] - ty[0])
            if abs(d) < 1e-12:
                continue
            l1 = ((px - tx[0]) * (ty[2] - ty[0]) - (py - ty[0]) * (tx[2] - tx[0])) / d
            l2 = ((py - ty[0]) * (tx[1] - tx[0]) - (px - tx[0]) * (ty[1] - ty[0])) / d
            l0 = 1.0 - l1 - l2
            inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
            if not inside.any():
                continue
            # Perspective-correct depth: interpolate 1/z linearly in screen space.
            inv_z = l0 / tz[0] + l1 / tz[1] + l2 / tz[2]
            depth = np.where(inv_z > 1e-12, 1.0 / np.maximum(inv_z, 1e-12), np.inf)
            zregion = zbuf[y0:y1, x0:x1]
            update = inside & (depth < zregion)
            if not update.any():
                continue
            zregion[update] = depth[update]
            rgb = (colors[i] * 255).astype(np.uint8)
            region = img[y0:y1, x0:x1]
            region[update] = np.array([*rgb, 255], dtype=np.uint8)
    return img


def render_scene(
    scene,
    path=None,
    pose_name: str | None = "__active__",
    with_base: bool = True,
    **kwargs,
) -> np.ndarray:
    """Render a scene's posed shapes (plus base) with their shape colors."""
    colored: list[tuple[Mesh, str]] = [
        (mesh, shape.color) for shape, mesh in scene.build_shape_meshes(pose_name)
    ]
    if with_base:
        base_mesh = build_base(scene.base)
        if base_mesh is not None:
            figure = Mesh.merge([m for m, _ in colored])
            if len(figure.faces):
                lo, hi = figure.bounds
                center = (lo + hi) / 2.0
                colored = [
                    (m.translated([-center[0], -center[1], -lo[2]]), c) for m, c in colored
                ]
            colored.append((base_mesh, "#6e6a63"))
    img = render_meshes(colored, **kwargs)
    if path is not None:
        write_png(path, img)
    return img
