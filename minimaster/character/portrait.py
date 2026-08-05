"""Portrait renderer: rasterise a character with the skin shading model.

Same z-buffered software rasteriser approach as :mod:`minimaster.render`, but
the per-vertex colour comes from :mod:`minimaster.character.shading` — linear
light transport, wrapped subsurface diffuse, Fresnel specular, cavity AO and a
three-point rig — and each shell carries its own :class:`Material`, so skin,
eyes and teeth shade differently in one pass.

    shells = [Shell(mesh, shading.SKIN), Shell(eye_mesh, shading.EYE)]
    img = render_portrait(shells, size=(600, 800), azimuth=25)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core import math3d as m3
from ..core.mesh import Mesh
from ..render import write_png  # noqa: F401  (re-exported for convenience)
from . import shading as sh


@dataclass
class Shell:
    mesh: Mesh
    material: sh.Material = field(default_factory=lambda: sh.GENERIC)
    ao: bool = True
    ao_strength: float = 1.0
    base_colors: np.ndarray | None = None  # per-vertex albedo override


# MakeHuman authors Y-up; MiniMaster is Z-up.
MH_TO_Z_UP = np.array([[1.0, 0, 0, 0], [0, 0, -1.0, 0], [0, 1.0, 0, 0],
                       [0, 0, 0, 1.0]])


def character_shells(base, verts=None, skin=None, eye_color="#5b7c8d",
                     levels: int = 0, to_z_up: bool = True,
                     with_eyes: bool = True, with_teeth: bool = False):
    """Build render shells for a morphed MakeHuman character.

    ``base`` is a :class:`~minimaster.character.mhbase.BaseMesh`, ``verts`` its
    morphed control vertices (or None for the neutral mesh).
    """
    from ..core.subdiv import quads_to_mesh, subdivide_quads
    from . import mhbase as mh

    xf = MH_TO_Z_UP if to_z_up else np.eye(4)

    def build(v, q):
        if levels:
            v, q = subdivide_quads(v, q, levels)
        return quads_to_mesh(v, q).transform(xf)

    body = build(*base.body_cage(verts))
    shells = [Shell(body, skin or sh.SKIN)]

    if with_eyes:
        # gaze is -Y in MakeHuman, which the Z-up transform sends to -Y as well
        forward = np.array([0.0, -1.0, 0.0])
        for name in ("helper-l-eye", "helper-r-eye"):
            if name not in base.groups:
                continue
            ev, eq = base.helper_group(name, verts)
            mesh = build(ev, eq)
            cols = sh.eye_vertex_colors(mesh.vertices, forward, iris=eye_color)
            shells.append(Shell(mesh, sh.EYE, ao=True, ao_strength=0.35,
                                base_colors=cols))
    if with_teeth:
        for name in ("helper-upper-teeth", "helper-lower-teeth"):
            if name in base.groups:
                shells.append(Shell(build(*base.helper_group(name, verts)),
                                    sh.TEETH))
    return shells


def _vertex_normals(mesh: Mesh) -> np.ndarray:
    """Area-weighted vertex normals."""
    v, f = mesh.vertices, mesh.faces
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    out = np.zeros_like(v)
    for i in range(3):
        np.add.at(out, f[:, i], fn)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, n, out=np.zeros_like(out), where=n > 1e-12)


def render_portrait(shells, size=(600, 800), azimuth=25.0, elevation=6.0,
                    background="#20242c", fov=26.0, rig=None,
                    exposure=1.15, fit=None, ao_strength=1.0):
    """Render shells to an (h, w, 4) uint8 image."""
    w, h = size
    rig = rig or sh.LightRig()
    img = np.zeros((h, w, 4), dtype=np.uint8)
    if background is not None:
        bg = sh.linear_to_srgb(sh.tonemap(sh.hex_to_linear(background), 1.0))
        img[:, :, :3] = (bg * 255).astype(np.uint8)
        img[:, :, 3] = 255

    shells = [s for s in shells if len(s.mesh.faces)]
    if not shells:
        return img

    combined = Mesh.merge([s.mesh for s in shells])
    lo, hi = combined.bounds
    center = (lo + hi) / 2.0
    radius = float(np.linalg.norm(hi - lo)) / 2.0 or 1.0
    if fit is not None:  # (center, radius) override, e.g. to frame the head
        center, radius = np.asarray(fit[0], dtype=np.float64), float(fit[1])

    az, el = np.radians(azimuth), np.radians(elevation)
    distance = radius / np.tan(np.radians(fov) / 2.0) * 1.05
    eye = center + distance * np.array(
        [np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)])
    view = m3.look_at(eye, center)
    focal = (h / 2.0) / np.tan(np.radians(fov) / 2.0)
    zbuf = np.full((h, w), np.inf)

    for shell in shells:
        mesh = shell.mesh
        vn = _vertex_normals(mesh)
        ao = None
        if shell.ao:
            ao = sh.cavity_ao(mesh.vertices, mesh.faces, vn,
                              strength=ao_strength * shell.ao_strength)
        cam = m3.transform_points(view, mesh.vertices)
        n_cam = vn @ view[:3, :3].T
        n_cam /= np.maximum(np.linalg.norm(n_cam, axis=1, keepdims=True), 1e-12)
        # view direction: from the surface toward the camera (origin in cam space)
        vd = -cam
        vd /= np.maximum(np.linalg.norm(vd, axis=1, keepdims=True), 1e-12)

        vcol = sh.shade_vertices(n_cam, vd, shell.material, rig, ao,
                                 base_colors=shell.base_colors)

        tris = cam[mesh.faces]
        depths = -tris[:, :, 2]
        fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        facing = np.einsum("ij,ij->i", fn, tris.mean(axis=1)) < 0.0
        valid = facing & (depths.min(axis=1) > 1e-6)
        xs = w / 2.0 + focal * tris[:, :, 0] / depths
        ys = h / 2.0 - focal * tris[:, :, 1] / depths
        cols = vcol[mesh.faces]  # (m, 3, 3)

        for i in np.nonzero(valid)[0]:
            tx, ty, tz = xs[i], ys[i], depths[i]
            x0 = max(int(np.floor(tx.min())), 0)
            x1 = min(int(np.ceil(tx.max())) + 1, w)
            y0 = max(int(np.floor(ty.min())), 0)
            y1 = min(int(np.ceil(ty.max())) + 1, h)
            if x0 >= x1 or y0 >= y1:
                continue
            px, py = np.meshgrid(np.arange(x0, x1) + 0.5,
                                 np.arange(y0, y1) + 0.5)
            d = ((tx[1] - tx[0]) * (ty[2] - ty[0])
                 - (tx[2] - tx[0]) * (ty[1] - ty[0]))
            if abs(d) < 1e-12:
                continue
            l1 = ((px - tx[0]) * (ty[2] - ty[0])
                  - (py - ty[0]) * (tx[2] - tx[0])) / d
            l2 = ((py - ty[0]) * (tx[1] - tx[0])
                  - (px - tx[0]) * (ty[1] - ty[0])) / d
            l0 = 1.0 - l1 - l2
            inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
            if not inside.any():
                continue
            inv_z = l0 / tz[0] + l1 / tz[1] + l2 / tz[2]
            depth = np.where(inv_z > 1e-12, 1.0 / np.maximum(inv_z, 1e-12), np.inf)
            region = zbuf[y0:y1, x0:x1]
            upd = inside & (depth < region)
            if not upd.any():
                continue
            region[upd] = depth[upd]
            c = (l0[:, :, None] * cols[i, 0]
                 + l1[:, :, None] * cols[i, 1]
                 + l2[:, :, None] * cols[i, 2])
            rgb = (sh.linear_to_srgb(sh.tonemap(c, exposure)) * 255).astype(np.uint8)
            tgt = img[y0:y1, x0:x1]
            tgt[..., :3][upd] = rgb[upd]
            tgt[..., 3][upd] = 255
    return img
