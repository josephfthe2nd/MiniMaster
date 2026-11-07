"""Vectorised visibility-buffer rasteriser (pure numpy, no tkinter).

:func:`portrait.render_portrait` used to loop over triangles in Python and call
``np.meshgrid`` for each one (95k calls for a feathered wing pair). This module
produces the *same pixels* without the loop:

1. **Setup.** Cull back faces and sliver triangles exactly like the old loop,
   project, and compute each survivor's integer pixel bounding box.
2. **Bucket.** Triangles are grouped by bounding-box size class
   (:data:`CLASSES`); each class is rasterised on a fixed ``B x B`` candidate grid
   per triangle, evaluated for a whole chunk (about :data:`MAX_CANDIDATES`
   candidates) at once. Triangles larger than the biggest class take a
   per-triangle path, because for them the per-call overhead is small next to
   the pixels they cover.
3. **Resolve.** Fragments are reduced to a *visibility buffer*: the nearest
   depth per pixel and the id of the triangle that owns it. The tie rule
   matches the old draw order: at an exactly equal depth the lowest triangle id
   wins, so the result does not depend on bucketing or chunking.
4. **Shade.** Gouraud colours are interpolated for the *visible* pixels only
   (:func:`shade_gouraud`), with the same expressions as the old loop.

The id buffer is also what :func:`portrait.pick` reads, and :func:`rasterize`
calls an optional ``check()`` between chunks so a worker thread can abort
(:class:`Cancelled`, :class:`CancelToken`).

Everything is float64 by default and reproduces the old loop bit for bit
(``tests/test_raster.py`` keeps the loop as the reference). ``dtype=np.float32``
is available for draft rendering; it is not pixel-exact.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from ..core import math3d as m3

# bounding-box edge classes; a triangle is rasterised on the smallest B x B grid
# that covers its box, triangles beyond the last class take the per-triangle path
CLASSES = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128])
MAX_CANDIDATES = 1_500_000      # candidate pixels evaluated per vectorised chunk
_BBOX_EPS = 1e-6                # guard band (pixels) so FP noise never loses a pixel
_BIG = np.iinfo(np.int32).max


# --------------------------------------------------------------------------
# cancellation


class Cancelled(Exception):
    """Raised by a ``check`` callback to abort a render in progress."""


class CancelToken:
    """A thread-safe flag to hand to a worker as its ``check`` callback.

        token = CancelToken()
        worker: render_portrait(shells, check=token)
        UI thread, on the next edit: token.cancel()

    Calling the token raises :class:`Cancelled` once :meth:`cancel` was called.
    """

    def __init__(self):
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def __call__(self) -> None:
        if self._event.is_set():
            raise Cancelled()


# --------------------------------------------------------------------------
# camera


@dataclass
class Camera:
    """The orbit camera of the portrait renderer."""

    view: np.ndarray        # 4x4 world -> camera (looks down -Z, +Y up)
    focal: float            # pixels
    width: int
    height: int

    def project(self, cam: np.ndarray):
        """Camera-space points -> (screen x, screen y, depth). Depth is clamped
        away from zero so far-behind vertices do not divide by zero; faces that
        reach the near plane are culled by :func:`front_faces` anyway."""
        depth = -cam[:, 2]
        safe = np.maximum(depth, 1e-6)
        sx = self.width / 2.0 + self.focal * cam[:, 0] / safe
        sy = self.height / 2.0 - self.focal * cam[:, 1] / safe
        return sx, sy, depth


def orbit_camera(lo, hi, size, azimuth=25.0, elevation=6.0, fov=26.0, fit=None) -> Camera:
    """Camera that frames the box ``lo..hi`` (or ``fit=(center, radius)``) from
    ``azimuth`` / ``elevation`` degrees. Z-up scenes, camera starts on -Y."""
    w, h = size
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
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
    return Camera(view, float(focal), int(w), int(h))


def front_faces(cam: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Mask of faces that face the camera and lie wholly in front of it.

    ``cam`` are camera-space vertices. This is the old loop's culling test,
    spelled identically so borderline triangles are decided identically.
    """
    tris = cam[faces]
    depths = -tris[:, :, 2]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    facing = np.einsum("ij,ij->i", fn, tris.mean(axis=1)) < 0.0
    return facing & (depths.min(axis=1) > 1e-6)


class Projected:
    """Shells put through one camera: per-shell camera-space vertices, and the
    merged screen-space vertices / faces / culling mask the rasteriser consumes."""

    def __init__(self, meshes, camera: Camera, check=None):
        self.camera = camera
        self.cam = []
        faces, valid = [], []
        voff = 0
        self.face_offsets = [0]
        for mesh in meshes:
            if check:
                check()
            c = m3.transform_points(camera.view, mesh.vertices)
            self.cam.append(c)
            faces.append(mesh.faces + voff)
            valid.append(front_faces(c, mesh.faces))
            voff += len(mesh.vertices)
            self.face_offsets.append(self.face_offsets[-1] + len(mesh.faces))
        self.faces = np.concatenate(faces) if faces else np.zeros((0, 3), dtype=np.int64)
        self.valid = np.concatenate(valid) if valid else np.zeros(0, dtype=bool)
        allcam = np.concatenate(self.cam) if self.cam else np.zeros((0, 3))
        self.sx, self.sy, self.depth = camera.project(allcam)

    def locate(self, face_ids: np.ndarray):
        """Merged face ids -> (shell index, face index within that shell)."""
        offs = np.asarray(self.face_offsets)
        shell = np.searchsorted(offs, face_ids, side="right") - 1
        return shell, face_ids - offs[shell]


# --------------------------------------------------------------------------
# rasterisation


def _edge_setup(sx, sy, depth, faces, face_ids, w, h, dtype):
    """Per-triangle edge-function inputs and integer pixel boxes.

    Pixel centres are at ``i + 0.5``; the box holds every centre that can lie
    inside the triangle (plus a 1e-6 px guard band: the candidates are then
    decided by the very same barycentric arithmetic as the old loop).
    """
    tri = faces[face_ids]
    f0, f1, f2 = tri[:, 0], tri[:, 1], tri[:, 2]
    ax, ay, bx, by, cx, cy = sx[f0], sy[f0], sx[f1], sy[f1], sx[f2], sy[f2]
    d = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
    x0 = np.maximum(np.ceil(np.minimum(np.minimum(ax, bx), cx) - 0.5 - _BBOX_EPS), 0)
    x1 = np.minimum(np.floor(np.maximum(np.maximum(ax, bx), cx) - 0.5 + _BBOX_EPS), w - 1)
    y0 = np.maximum(np.ceil(np.minimum(np.minimum(ay, by), cy) - 0.5 - _BBOX_EPS), 0)
    y1 = np.minimum(np.floor(np.maximum(np.maximum(ay, by), cy) - 0.5 + _BBOX_EPS), h - 1)
    ok = (np.abs(d) >= 1e-12) & (x0 <= x1) & (y0 <= y1)
    k = np.nonzero(ok)[0]

    def g(a):
        return a[k].astype(dtype, copy=False)

    return dict(
        id=face_ids[k].astype(np.int32), ax=g(ax), ay=g(ay), bx=g(bx), by=g(by),
        cx=g(cx), cy=g(cy), d=g(d),
        za=g(depth[f0]), zb=g(depth[f1]), zc=g(depth[f2]),
        x0=x0[k].astype(np.int32), y0=y0[k].astype(np.int32),
        bw=(x1[k] - x0[k] + 1).astype(np.int32), bh=(y1[k] - y0[k] + 1).astype(np.int32),
    )


def _inside(T, sel, ix, iy, dtype):
    """Barycentric test of the candidate pixel grids ``ix, iy`` (len(sel), P) for
    the triangles ``sel``: returns (rows, cols, l0, l1, l2) of the covered ones."""
    px = ix.astype(dtype) + dtype(0.5)
    py = iy.astype(dtype) + dtype(0.5)
    ax, ay = T["ax"][sel][:, None], T["ay"][sel][:, None]
    bx, by = T["bx"][sel][:, None], T["by"][sel][:, None]
    cx, cy = T["cx"][sel][:, None], T["cy"][sel][:, None]
    d = T["d"][sel][:, None]
    l1 = ((px - ax) * (cy - ay) - (py - ay) * (cx - ax)) / d
    l2 = ((py - ay) * (bx - ax) - (px - ax) * (by - ay)) / d
    l0 = dtype(1.0) - l1 - l2
    return l0, l1, l2


def _fragments(T, sel, B, w, h, dtype):
    """Covered pixels of the triangles ``sel`` on a ``B x B`` candidate grid each:
    (pixel index, depth, face id)."""
    ox = np.tile(np.arange(B, dtype=np.int32), B)
    oy = np.repeat(np.arange(B, dtype=np.int32), B)
    ix = T["x0"][sel][:, None] + ox[None, :]
    iy = T["y0"][sel][:, None] + oy[None, :]
    l0, l1, l2 = _inside(T, sel, ix, iy, dtype)
    inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0) & (ix < w) & (iy < h)
    ti, ci = np.nonzero(inside)
    l0, l1, l2 = l0[ti, ci], l1[ti, ci], l2[ti, ci]
    inv = l0 / T["za"][sel][ti] + l1 / T["zb"][sel][ti] + l2 / T["zc"][sel][ti]
    pix = iy[ti, ci].astype(np.int64) * w + ix[ti, ci]
    return pix, inv, T["id"][sel][ti]


def _depth_from_inverse(inv, dtype):
    """Perspective-correct depth, ``inf`` where the inverse depth is not positive
    enough (the old loop never writes those fragments)."""
    one = dtype(1.0)
    return np.where(inv > 1e-12, one / np.maximum(inv, 1e-12), np.inf).astype(dtype, copy=False)


def _resolve(zbuf, idbuf, pix, dep, tri):
    """Fold fragments into the visibility buffer: nearest depth, then lowest id.

    The order of the folds does not matter (a lexicographic minimum), which is
    what makes the buckets and chunks interchangeable.
    """
    finite = dep < np.inf
    if not finite.all():
        pix, dep, tri = pix[finite], dep[finite], tri[finite]
    if not len(pix):
        return
    # remember the depths this fold may improve: a whole-buffer copy is cheaper
    # than a gather once the chunk is big next to the image
    if len(pix) * 8 > zbuf.size:
        before = zbuf.copy()
        np.minimum.at(zbuf, pix, dep)
        win = dep == zbuf[pix]
        pw, tw = pix[win], tri[win]
        stale = zbuf[pw] < before[pw]
    else:
        before = zbuf[pix]
        np.minimum.at(zbuf, pix, dep)
        now = zbuf[pix]
        win = dep == now
        pw, tw = pix[win], tri[win]
        stale = now[win] < before[win]
    idbuf[pw[stale]] = _BIG          # a nearer depth: the previous owner is stale
    np.minimum.at(idbuf, pw, tw)


def _per_triangle(T, j, w, zbuf, idbuf, dtype):
    """The old loop's method for one (large) triangle."""
    x0, y0 = int(T["x0"][j]), int(T["y0"][j])
    gx, gy = np.meshgrid(np.arange(x0, x0 + T["bw"][j], dtype=np.int32),
                         np.arange(y0, y0 + T["bh"][j], dtype=np.int32))
    px = gx.astype(dtype) + dtype(0.5)
    py = gy.astype(dtype) + dtype(0.5)
    ax, ay, bx, by, cx, cy, d = (T[k][j] for k in ("ax", "ay", "bx", "by", "cx", "cy", "d"))
    l1 = ((px - ax) * (cy - ay) - (py - ay) * (cx - ax)) / d
    l2 = ((py - ay) * (bx - ax) - (px - ax) * (by - ay)) / d
    l0 = dtype(1.0) - l1 - l2
    ins = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
    if not ins.any():
        return
    l0, l1, l2 = l0[ins], l1[ins], l2[ins]
    inv = l0 / T["za"][j] + l1 / T["zb"][j] + l2 / T["zc"][j]
    pix = gy[ins].astype(np.int64) * w + gx[ins]
    _resolve(zbuf, idbuf, pix, _depth_from_inverse(inv, dtype),
             np.full(len(pix), T["id"][j], dtype=np.int32))


def rasterize(sx, sy, depth, faces, size, face_ids=None, check=None,
              dtype=np.float64, max_candidates: int = MAX_CANDIDATES):
    """Visibility buffer of projected triangles.

    ``sx, sy, depth`` are per-vertex screen coordinates and positive depths,
    ``faces`` index them. ``face_ids`` selects the triangles to draw (the ones
    that survived culling); default is all of them. Returns
    ``(ids, z)``: ``ids`` is an ``(h, w)`` int32 array holding the index into
    ``faces`` of the nearest triangle at each pixel centre (``-1`` where none),
    ``z`` the matching depth (``inf`` where none).

    ``check`` is called before every vectorised chunk and every few large
    triangles; whatever it raises propagates.
    """
    w, h = size
    zbuf = np.full(h * w, np.inf, dtype=dtype)
    idbuf = np.full(h * w, _BIG, dtype=np.int32)
    if face_ids is None:
        face_ids = np.arange(len(faces))
    if len(face_ids):
        T = _edge_setup(np.asarray(sx), np.asarray(sy), np.asarray(depth),
                        np.asarray(faces), np.asarray(face_ids), w, h, dtype)
        _fill(T, w, h, zbuf, idbuf, check, dtype, max_candidates)
    idbuf[idbuf == _BIG] = -1
    return idbuf.reshape(h, w), zbuf.reshape(h, w)


def _fill(T, w, h, zbuf, idbuf, check, dtype, max_candidates):
    if not len(T["id"]):
        return
    extent = np.maximum(T["bw"], T["bh"])
    cls = np.searchsorted(CLASSES, extent)          # smallest class >= extent
    order = np.argsort(cls, kind="stable")
    bounds = np.searchsorted(cls[order], np.arange(len(CLASSES) + 2))
    for c in range(len(CLASSES)):
        lo, hi = bounds[c], bounds[c + 1]
        B = int(CLASSES[c])
        step = max(1, max_candidates // (B * B))
        for s in range(lo, hi, step):
            if check:
                check()
            sel = order[s:min(hi, s + step)]
            pix, inv, tri = _fragments(T, sel, B, w, h, dtype)
            _resolve(zbuf, idbuf, pix, _depth_from_inverse(inv, dtype), tri)
    for n, j in enumerate(order[bounds[len(CLASSES)]:]):
        if check and n % 8 == 0:
            check()
        _per_triangle(T, j, w, zbuf, idbuf, dtype)


# --------------------------------------------------------------------------
# deferred shading


def shade_gouraud(ids, sx, sy, faces, vcol, exposure, tonemap, to_srgb):
    """Gouraud-shade the visible pixels of a visibility buffer.

    ``ids`` is the ``(h, w)`` id buffer, ``vcol`` the linear colour of every
    vertex. Barycentrics are recomputed in screen space for the covered pixels
    only (the old loop interpolated colour the same, affine way). Returns
    ``(flat pixel indices, uint8 rgb)``.
    """
    h, w = ids.shape
    flat = ids.reshape(-1)
    vis = np.nonzero(flat >= 0)[0]
    f = faces[flat[vis]]
    f0, f1, f2 = f[:, 0], f[:, 1], f[:, 2]
    ax, ay, bx, by, cx, cy = sx[f0], sy[f0], sx[f1], sy[f1], sx[f2], sy[f2]
    px = (vis % w) + 0.5
    py = (vis // w) + 0.5
    d = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
    l1 = ((px - ax) * (cy - ay) - (py - ay) * (cx - ax)) / d
    l2 = ((py - ay) * (bx - ax) - (px - ax) * (by - ay)) / d
    l0 = 1.0 - l1 - l2
    c = l0[:, None] * vcol[f0] + l1[:, None] * vcol[f1] + l2[:, None] * vcol[f2]
    return vis, (to_srgb(tonemap(c, exposure)) * 255).astype(np.uint8)
