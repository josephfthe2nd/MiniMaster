"""Draft tier of the interactive preview: proxies and a fast shader (no tkinter).

While the user orbits, drags a slider or poses an appendage, the preview cannot
afford the full meshes (a body is 26k triangles, a feathered wing pair 171k).
It draws *proxies* instead, then re-renders at full quality when the edits stop
(``portrait.render_portrait``, abortable with ``check=``). This module is the
tk-free part of that:

* :class:`ClusterProxy` -- a decimated copy of a mesh made by **vertex
  clustering**: vertices falling in the same grid cell merge into one. The
  clustering (which vertices merge, which triangles survive) is computed once
  from the *rest* mesh; after that the proxy of any deformed version of the mesh
  (a morphed body, a posed wing) is a ``bincount`` average of the deformed
  vertices -- no decimation per frame. Clustering is per *piece* (feather,
  membrane, bone ...) with a single ``np.unique`` over ``(piece, cell)`` keys,
  so the 542 pieces of a feathered wing cost the same as one.
* :class:`BodyProxy` -- the MakeHuman body as a ``ClusterProxy`` of about 4k
  triangles, updated per morph in about 0.3 ms.
* :func:`render_draft` -- flat or Gouraud shading of shells at draft quality
  (float32 rasteriser, per-face colour table), >= 60 fps at 320x480 for a body
  and two proxied wings.

Nothing here knows about wings: posing the appendage is the caller's business
(``PlacedWing.pose`` and friends); the proxy only turns the posed vertices into
a small mesh. For callers that can deform individual vertices cheaply (skinning),
:attr:`ClusterProxy.rep` names one source vertex per proxy vertex, so only those
need to be deformed.
"""

from __future__ import annotations

import numpy as np

from ..core.mesh import Mesh
from . import portrait as P
from . import raster as R
from . import shading as sh

_MAX_CELLS = 1024          # per axis: keeps the (piece, cell) key inside int64


def pieces_from_ranges(ranges, n_vertices: int) -> np.ndarray:
    """Per-vertex piece ids from ``(start, stop)`` vertex ranges, one per piece
    (e.g. ``rig.ranges.values()`` of a wing rig). Vertices in no range get -1."""
    piece = np.full(n_vertices, -1, dtype=np.int64)
    for i, (a, b) in enumerate(ranges):
        piece[a:b] = i
    return piece


def _faces_from_labels(labels, tris):
    """Map triangles through ``labels``; drop collapsed ones; drop duplicates
    (same triangle, same orientation) while keeping the opposite orientation,
    which is the other side of a thin slab."""
    f = labels[tris]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    if not len(f):
        return f
    rot = np.argmin(f, axis=1)                      # canonical cyclic rotation
    f = np.take_along_axis(f, (np.arange(3)[None, :] + rot[:, None]) % 3, axis=1)
    k = int(f.max()) + 1
    _, first = np.unique((f[:, 0] * k + f[:, 1]) * k + f[:, 2], return_index=True)
    return f[np.sort(first)]


def _cell_labels(verts, piece, cell):
    """Cluster id of every vertex: one id per occupied (piece, grid cell).

    ``cell`` is the cell edge, a scalar or one value per piece (indexed by the
    piece's rank among the occupied ones). Cells are anchored at each piece's
    own minimum corner.
    """
    verts = np.asarray(verts, dtype=np.float64)
    if piece is None:
        piece = np.zeros(len(verts), dtype=np.int64)
    order = np.argsort(piece, kind="stable")
    ps = piece[order]
    starts = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]])
    rank = np.empty(len(verts), dtype=np.int64)
    rank[order] = np.cumsum(np.r_[False, ps[1:] != ps[:-1]])
    lo = np.minimum.reduceat(verts[order], starts, axis=0)
    cell = np.broadcast_to(np.asarray(cell, dtype=np.float64), (len(starts),))
    q = np.floor((verts - lo[rank]) / cell[rank][:, None]).astype(np.int64)
    m = int(q.max()) + 2
    key = ((rank * m + q[:, 0]) * m + q[:, 1]) * m + q[:, 2]
    return np.unique(key, return_inverse=True)[1].reshape(-1)


def _target_labels(v, faces, target_tris, iters=14):
    ext = float((v.max(axis=0) - v.min(axis=0)).max())
    lo, hi = ext / 400.0, ext / 2.0
    labels = None
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        labels = _cell_labels(v, None, mid)
        if len(_faces_from_labels(labels, faces)) > target_tris:
            lo = mid
        else:
            hi = mid
    return labels


class ClusterProxy:
    """A decimated copy of a mesh by vertex clustering, with fixed topology.

        proxy = ClusterProxy.by_target(rest_vertices, faces, 4000)   # or by_cell / by_pieces
        small = proxy.mesh(deformed_vertices)        # per frame: one bincount average

    ``faces`` are the proxy's triangles (indices into proxy vertices); ``rest``
    its vertices for the rest mesh; ``rep[i]`` is the source vertex nearest
    proxy vertex ``i`` at rest. ``update(vertices)`` maps *any* deformation of
    the source vertices (same count, same order) to proxy vertices.
    """

    def __init__(self, rest_vertices, faces, labels):
        rest_vertices = np.asarray(rest_vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        kept = _faces_from_labels(labels, faces)
        used = np.unique(kept)                      # clusters some triangle still uses
        remap = np.full(int(labels.max()) + 1 if len(labels) else 0, -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        lab = remap[labels]
        self.faces = remap[kept]
        self.n_vertices = len(used)
        self.n_source = len(rest_vertices)
        sel = np.flatnonzero(lab >= 0)              # source vertices that feed a proxy vertex
        self._src = sel
        self._lab = lab[sel]
        self._count = np.bincount(self._lab, minlength=self.n_vertices).astype(np.float64)
        self.rest = self.update(rest_vertices)
        if len(sel):
            d = np.linalg.norm(rest_vertices[sel] - self.rest[self._lab], axis=1)
            order = np.lexsort((d, self._lab))      # nearest member first within a cluster
            first = np.r_[True, self._lab[order][1:] != self._lab[order][:-1]]
            self.rep = sel[order[first]]
        else:                                       # every triangle collapsed: nothing left
            self.rep = sel

    # -- construction -------------------------------------------------------

    @classmethod
    def by_cell(cls, rest_vertices, faces, cell, piece=None):
        """Cluster with a grid cell of edge ``cell`` (scalar, or one per piece)."""
        return cls(rest_vertices, faces, _cell_labels(rest_vertices, piece, cell))

    @classmethod
    def by_target(cls, rest_vertices, faces, target_tris: int, iters: int = 14):
        """Bisect the cell size until about ``target_tris`` triangles remain."""
        v = np.asarray(rest_vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        return cls(v, faces, _target_labels(v, faces, target_tris, iters))

    @classmethod
    def by_pieces(cls, rest_vertices, faces, piece, detail: float = 3.0,
                  min_cells: int = 3, max_cells: int = 14):
        """Cluster every piece on its own grid, all pieces in one pass.

        A piece with ``t`` triangles gets about ``clip(sqrt(t) / detail,
        min_cells, max_cells)`` cells along its longest side. Larger ``detail``
        is coarser. ``piece`` gives each vertex's piece id (see
        :func:`pieces_from_ranges`); triangles must not span pieces.
        """
        v = np.asarray(rest_vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        piece = np.asarray(piece, dtype=np.int64)
        ids, rank = np.unique(piece, return_inverse=True)
        rank = rank.reshape(-1)
        order = np.argsort(rank, kind="stable")
        starts = np.flatnonzero(np.r_[True, rank[order][1:] != rank[order][:-1]])
        lo = np.minimum.reduceat(v[order], starts, axis=0)
        hi = np.maximum.reduceat(v[order], starts, axis=0)
        tris = np.bincount(rank[faces[:, 0]], minlength=len(ids))
        cells = np.clip(np.sqrt(tris) / detail, min_cells, max_cells)
        cell = np.maximum((hi - lo).max(axis=1) / cells, 1e-12)
        cell = np.maximum(cell, (hi - lo).max(axis=1) / _MAX_CELLS)
        return cls(v, faces, _cell_labels(v, rank, cell))

    # -- per-frame ----------------------------------------------------------

    def _average(self, members) -> np.ndarray:
        """Mean of each cluster's member rows (``members`` is already gathered
        to ``self._src`` order)."""
        out = np.empty((self.n_vertices, 3))
        for a in range(3):
            out[:, a] = np.bincount(self._lab, weights=members[:, a], minlength=self.n_vertices)
        return out / self._count[:, None]

    def update(self, vertices) -> np.ndarray:
        """Proxy vertices for a deformation ``vertices`` of the source mesh: the
        mean of each cluster's members. Well under a millisecond for a body."""
        return self._average(np.asarray(vertices, dtype=np.float64)[self._src])

    def mesh(self, vertices=None) -> Mesh:
        """The proxy as a :class:`Mesh` (the rest shape if ``vertices`` is None)."""
        return Mesh(self.rest if vertices is None else self.update(vertices), self.faces)


class BodyProxy(ClusterProxy):
    """The MakeHuman body cage as a ~4k-triangle proxy (~15% of 26,756).

    Built once from the base mesh (about 30 ms, depends on nothing else). Per
    morph, ``mesh(verts)`` takes the same morphed control vertices
    ``portrait.character_shells`` takes and returns a Z-up :class:`Mesh`.
    """

    def __init__(self, base, target_tris: int = 4000):
        rest_cage, quads = base.body_cage(None)
        tris = np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
        super().__init__(rest_cage, tris, _target_labels(rest_cage, tris, target_tris))
        # the source vertices as indices into the FULL control-vertex array, so a
        # morph result is used directly (no body_cage() copy first)
        self._full_src = np.asarray(base.body_verts_idx)[self._src]

    def points(self, verts=None, to_z_up: bool = True) -> np.ndarray:
        """Proxy vertices for the morphed control vertices ``verts`` (the neutral
        body if None)."""
        pts = self.rest if verts is None else self._average(
            np.asarray(verts, dtype=np.float64)[self._full_src])
        return pts @ P.MH_TO_Z_UP[:3, :3].T if to_z_up else pts

    def mesh(self, verts=None, to_z_up: bool = True) -> Mesh:
        return Mesh(self.points(verts, to_z_up), self.faces)


# --------------------------------------------------------------------------
# draft shading


def _face_colors(shell, cam, rig, faces):
    """Linear colour of the faces ``faces`` (an index array) of one shell: its
    material lit at each face's centre with the face normal (``cam``: camera-space
    vertices; no AO)."""
    f = shell.mesh.faces[faces]
    t = cam[f]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    c = t.mean(axis=1)
    vd = -c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    base = None
    if shell.base_colors is not None:
        base = np.asarray(shell.base_colors, dtype=np.float64)[f].mean(axis=1)
    return sh.shade_vertices(n, vd, shell.material, rig, None, base_colors=base)


def render_draft(shells, size=(320, 480), azimuth=25.0, elevation=6.0, fov=26.0,
                 fit=None, shading="flat", background="#20242c", rig=None,
                 exposure=1.15, check=None, dtype=np.float32):
    """Draft-quality render of shells: the same camera, lights, materials and
    output as :func:`portrait.render_portrait` for the same arguments, minus
    cavity AO, with a float32 rasteriser (edges may differ from the final by a
    pixel).

    ``shading`` is ``"flat"`` (one colour per triangle, a table lookup per
    pixel: the fastest, and right for clustered thin parts whose averaged
    vertex normals shade badly), ``"smooth"`` (Gouraud from vertex normals) or
    one of those per shell, as a sequence. ``check`` is as in ``render_portrait``.
    """
    w, h = size
    rig = rig or sh.LightRig()
    img = P._background(size, background)
    live = [s for s in shells if len(s.mesh.faces)]
    modes = [shading] * len(shells) if isinstance(shading, str) else list(shading)
    if len(modes) != len(shells) or any(m not in ("flat", "smooth") for m in modes):
        raise ValueError("shading must be 'flat', 'smooth' or one of them per shell")
    modes = [m for s, m in zip(shells, modes) if len(s.mesh.faces)]
    if not live:
        return img

    camera = P._camera(live, size, azimuth, elevation, fov, fit)
    proj = R.Projected([s.mesh for s in live], camera, check)
    ids, _ = R.rasterize(proj.sx, proj.sy, proj.depth, proj.faces, (w, h),
                         np.nonzero(proj.valid)[0], check, dtype=dtype)
    flat_px = ids.reshape(-1)
    vis = np.nonzero(flat_px >= 0)[0]
    if not len(vis):
        return img
    shell_of_face = np.repeat(np.arange(len(live)), np.diff(proj.face_offsets))
    vshell = shell_of_face[flat_px[vis]]
    out = np.empty((len(vis), 3), dtype=np.uint8)

    smooth_shells = [i for i, m in enumerate(modes) if m == "smooth"]
    flat_shells = [i for i, m in enumerate(modes) if m == "flat"]
    if flat_shells:
        lut = np.zeros((len(proj.faces), 3), dtype=np.uint8)
        for i in flat_shells:
            if check:
                check()
            a, b = proj.face_offsets[i], proj.face_offsets[i + 1]
            drawn = np.flatnonzero(proj.valid[a:b])          # back faces are never looked up
            col = _face_colors(live[i], proj.cam[i], rig, drawn)
            lut[a + drawn] = (sh.linear_to_srgb(sh.tonemap(col, exposure)) * 255).astype(np.uint8)
        if smooth_shells:
            sel = np.isin(vshell, flat_shells)
            out[sel] = lut[flat_px[vis[sel]]]
        else:
            out = lut[flat_px[vis]]
    if smooth_shells:
        colors = []
        for i, (shell, cam) in enumerate(zip(live, proj.cam)):
            if check:
                check()
            colors.append(P._shell_colors(shell, cam, camera.view, rig, 0.0, False, ao=False)
                          if i in smooth_shells else np.zeros((len(cam), 3)))
        sel = np.isin(vshell, smooth_shells)
        idm = np.full(len(flat_px), -1, dtype=np.int32)
        idm[vis[sel]] = flat_px[vis[sel]]
        pix, rgb = R.shade_gouraud(idm.reshape(h, w), proj.sx, proj.sy, proj.faces,
                                   np.concatenate(colors), exposure,
                                   sh.tonemap, sh.linear_to_srgb)
        out[sel] = rgb
    flat = img.reshape(h * w, 4)
    flat[vis, :3] = out
    flat[vis, 3] = 255
    return img
