"""Portrait renderer: rasterise a character with the skin shading model.

Same z-buffered software rasteriser approach as :mod:`minimaster.render`, but
the per-vertex colour comes from :mod:`minimaster.character.shading` — linear
light transport, wrapped subsurface diffuse, Fresnel specular, cavity AO and a
three-point rig — and each shell carries its own :class:`Material`, so skin,
eyes and teeth shade differently in one pass.

    shells = [Shell(mesh, shading.SKIN), Shell(eye_mesh, shading.EYE)]
    img = render_portrait(shells, size=(600, 800), azimuth=25)

Triangles are rasterised by the vectorised visibility buffer in
:mod:`minimaster.character.raster` (no per-triangle Python loop; pixel-identical
to the loop it replaced, ~15x faster), and the view-independent shading terms
(vertex normals, cavity AO) are cached per shell by content, so re-rendering a
scene after editing one shell does not redo the others. ``check=`` lets a
worker thread abort a render; :func:`visibility_buffer` / :func:`pick` expose
the id buffer for picking.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.mesh import Mesh
from ..render import write_png  # noqa: F401  (re-exported for convenience)
from . import raster as R
from . import shading as sh
from .raster import CancelToken, Cancelled  # noqa: F401  (re-exported for workers)


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
    return sh.vertex_normals(mesh.vertices, mesh.faces)


def _background(size, background):
    w, h = size
    img = np.zeros((h, w, 4), dtype=np.uint8)
    if background is not None:
        bg = sh.linear_to_srgb(sh.tonemap(sh.hex_to_linear(background), 1.0))
        img[:, :, :3] = (bg * 255).astype(np.uint8)
        img[:, :, 3] = 255
    return img


def _camera(shells, size, azimuth, elevation, fov, fit) -> R.Camera:
    """The camera render_portrait, visibility_buffer and pick share."""
    lo = np.min([s.mesh.vertices.min(axis=0) for s in shells], axis=0)
    hi = np.max([s.mesh.vertices.max(axis=0) for s in shells], axis=0)
    return R.orbit_camera(lo, hi, size, azimuth, elevation, fov, fit)


def _shell_colors(shell: Shell, cam, view, rig, ao_strength, cache, ao=True):
    """Linear radiance of every vertex of ``shell`` (``cam``: its camera-space
    vertices). ``ao=False`` skips cavity AO (draft rendering)."""
    mesh = shell.mesh
    strength = ao_strength * shell.ao_strength
    use_ao = ao and shell.ao
    carried = getattr(mesh, "shading_normals", None)
    if carried is not None:
        # a mesh that carries its own shading (collar.CollarMesh: a root collar drawn with the skin's
        # normals and occlusion at its rim, so it shades as the skin it lies on)
        vn = carried
        term = getattr(mesh, "shading_cavity", None)
        occ = sh.ao_from_term(term, strength) if use_ao and term is not None else None
    elif cache is False:
        vn = sh.vertex_normals(mesh.vertices, mesh.faces)
        occ = (sh.cavity_ao(mesh.vertices, mesh.faces, vn, strength=strength)
               if use_ao else None)
    else:
        vn, term = (sh.DEFAULT_CACHE if cache is None else cache).terms(
            mesh.vertices, mesh.faces, ao=use_ao)
        occ = sh.ao_from_term(term, strength) if use_ao else None
    n_cam = vn @ view[:3, :3].T
    n_cam /= np.maximum(np.linalg.norm(n_cam, axis=1, keepdims=True), 1e-12)
    # view direction: from the surface toward the camera (origin in cam space)
    vd = -cam
    vd /= np.maximum(np.linalg.norm(vd, axis=1, keepdims=True), 1e-12)
    return sh.shade_vertices(n_cam, vd, shell.material, rig, occ,
                             base_colors=shell.base_colors)


def render_portrait(shells, size=(600, 800), azimuth=25.0, elevation=6.0,
                    background="#20242c", fov=26.0, rig=None,
                    exposure=1.15, fit=None, ao_strength=1.0,
                    check=None, cache=None):
    """Render shells to an (h, w, 4) uint8 image.

    ``check`` makes the render abortable: it is called (with no arguments)
    before each shell is shaded and before each chunk of triangles is
    rasterised, never per pixel. Whatever it raises propagates out of
    ``render_portrait`` -- raise :class:`Cancelled` (``portrait.Cancelled``) --
    and no image is returned. :class:`CancelToken` is a ready-made, thread-safe
    ``check`` (``token.cancel()`` from another thread). A cancelled render
    leaves nothing half-done behind: cache entries are inserted whole.

    ``cache`` is the :class:`~minimaster.character.shading.ShadingCache` that
    keeps the view-independent shading terms (default: the shared one;
    ``False`` computes them afresh every call).
    """
    w, h = size
    rig = rig or sh.LightRig()
    img = _background(size, background)

    shells = [s for s in shells if len(s.mesh.faces)]
    if not shells:
        return img

    camera = _camera(shells, size, azimuth, elevation, fov, fit)
    proj = R.Projected([s.mesh for s in shells], camera, check)
    colors = []
    for shell, cam in zip(shells, proj.cam):
        if check:
            check()
        colors.append(_shell_colors(shell, cam, camera.view, rig, ao_strength, cache))

    ids, _ = R.rasterize(proj.sx, proj.sy, proj.depth, proj.faces, (w, h),
                         np.nonzero(proj.valid)[0], check)
    if check:
        check()
    vis, rgb = R.shade_gouraud(ids, proj.sx, proj.sy, proj.faces,
                               np.concatenate(colors), exposure,
                               sh.tonemap, sh.linear_to_srgb)
    flat = img.reshape(h * w, 4)
    flat[vis, :3] = rgb
    flat[vis, 3] = 255
    return img


# --------------------------------------------------------------------------
# id buffer / picking


@dataclass
class VisBuffer:
    """What is under every pixel: ``shell`` and ``face`` (index within that
    shell's mesh) are ``-1`` where nothing is, ``depth`` is camera-space
    distance along the view axis (``inf`` where nothing is). Pixel ``(x, y)``
    is column ``x``, row ``y`` (origin top left, sampled at its centre)."""

    camera: R.Camera
    shell: np.ndarray
    face: np.ndarray
    depth: np.ndarray

    def pick(self, x, y):
        """``(shell index, face index)`` at pixel ``(x, y)``, or ``None``
        (background, or outside the image). ``x``, ``y`` may be fractional: the
        pixel containing them is used."""
        h, w = self.shell.shape
        px, py = int(np.floor(x)), int(np.floor(y))
        if not (0 <= px < w and 0 <= py < h) or self.shell[py, px] < 0:
            return None
        return int(self.shell[py, px]), int(self.face[py, px])

    def pick_many(self, xs, ys):
        """Vectorised :meth:`pick`: two int arrays, ``-1`` for a miss."""
        h, w = self.shell.shape
        px = np.floor(np.asarray(xs, dtype=np.float64)).astype(np.int64)
        py = np.floor(np.asarray(ys, dtype=np.float64)).astype(np.int64)
        ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        px, py = np.where(ok, px, 0), np.where(ok, py, 0)
        shell = np.where(ok, self.shell[py, px], -1)
        return shell, np.where(shell >= 0, self.face[py, px], -1)

    def world_point(self, x, y):
        """World position of the surface under pixel ``(x, y)`` (its centre), or
        ``None``: the inverse of the projection at the stored depth."""
        h, w = self.shell.shape
        px, py = int(np.floor(x)), int(np.floor(y))
        if not (0 <= px < w and 0 <= py < h) or self.shell[py, px] < 0:
            return None
        z = float(self.depth[py, px])
        f = self.camera.focal
        pc = np.array([(px + 0.5 - w / 2.0) * z / f, -(py + 0.5 - h / 2.0) * z / f, -z])
        view = self.camera.view
        return view[:3, :3].T @ (pc - view[:3, 3])


def visibility_buffer(shells, size=(600, 800), azimuth=25.0, elevation=6.0,
                      fov=26.0, fit=None, check=None, dtype=np.float64) -> VisBuffer:
    """Rasterise only the ids (no shading: ~4 ms for the body at 320x480).

    The camera is the one :func:`render_portrait` uses for the same arguments.
    ``shell`` indices refer to the ``shells`` list as given (empty shells keep
    their slot)."""
    w, h = size
    live = [i for i, s in enumerate(shells) if len(s.mesh.faces)]
    if not live:
        full = np.full((h, w), -1, dtype=np.int32)
        return VisBuffer(R.orbit_camera(np.zeros(3), np.ones(3), size, azimuth,
                                        elevation, fov, fit),
                         full, full.copy(), np.full((h, w), np.inf))
    chosen = [shells[i] for i in live]
    camera = _camera(chosen, size, azimuth, elevation, fov, fit)
    proj = R.Projected([s.mesh for s in chosen], camera, check)
    ids, z = R.rasterize(proj.sx, proj.sy, proj.depth, proj.faces, (w, h),
                         np.nonzero(proj.valid)[0], check, dtype=dtype)
    hit = ids >= 0
    shell, face = proj.locate(np.where(hit, ids, 0))
    shell = np.where(hit, np.asarray(live)[shell], -1).astype(np.int32)
    face = np.where(hit, face, -1).astype(np.int32)
    return VisBuffer(camera, shell, face, z)


def pick(shells, size, azimuth, elevation, fit, x, y, fov=26.0, check=None):
    """The ``(shell index, triangle index)`` visible at pixel ``(x, y)`` of
    ``render_portrait(shells, size, azimuth, elevation, fit=fit)``, or ``None``
    over background. Builds the id buffer for the call; keep a
    :func:`visibility_buffer` to pick many pixels of one view."""
    return visibility_buffer(shells, size, azimuth, elevation, fov, fit, check).pick(x, y)
