"""Catmull-Clark subdivision for closed quad control cages.

The morphable-human generator authors a coarse quad cage (a box-modeled body
whose vertex positions are functions of the character sliders) and smooths it
into an organic surface here. Catmull-Clark on a closed, consistently-wound
quad cage yields a closed, consistently-wound quad mesh — so watertightness is
preserved through every subdivision level, and the final triangulation passes
the same integrity gate as every other MiniMaster mesh.

Fully vectorized: one subdivision level is a handful of ``np.add.at``
scatter-adds over the cage's corner/edge incidence — no Python loops over
faces.
"""

from __future__ import annotations

import numpy as np

from .mesh import Mesh


def _cc_once(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One Catmull-Clark level for a closed quad mesh.

    New vertex layout: [moved original verts | face points | edge points].
    Each input quad becomes 4 quads, winding preserved.
    """
    V, F = len(verts), len(faces)
    fp = verts[faces].mean(axis=1)  # (F, 3) face points

    # undirected edges from the 4 directed edges of every face
    a = faces
    b = np.roll(faces, -1, axis=1)
    und = np.sort(np.stack([a.ravel(), b.ravel()], axis=1), axis=1)  # (4F, 2)
    uniq, inv = np.unique(und, axis=0, return_inverse=True)
    E = len(uniq)

    ecnt = np.zeros(E)
    np.add.at(ecnt, inv, 1.0)
    if not np.all(ecnt == 2.0):
        bad = int((ecnt != 2.0).sum())
        raise ValueError(
            f"cage is not a closed 2-manifold: {bad} edges not shared by "
            "exactly two quads")

    # edge points: (both endpoints + both adjacent face points) / 4
    face_idx = np.repeat(np.arange(F), 4)
    efp = np.zeros((E, 3))
    np.add.at(efp, inv, fp[face_idx])
    mid = (verts[uniq[:, 0]] + verts[uniq[:, 1]]) / 2.0
    epts = (verts[uniq[:, 0]] + verts[uniq[:, 1]] + efp) / 4.0

    # vertex points: (Q + 2R + (n-3) v) / n
    Qs = np.zeros((V, 3))
    Qc = np.zeros(V)
    np.add.at(Qs, faces.ravel(), np.repeat(fp, 4, axis=0))
    np.add.at(Qc, faces.ravel(), 1.0)
    Rs = np.zeros((V, 3))
    Rc = np.zeros(V)
    np.add.at(Rs, uniq[:, 0], mid)
    np.add.at(Rc, uniq[:, 0], 1.0)
    np.add.at(Rs, uniq[:, 1], mid)
    np.add.at(Rc, uniq[:, 1], 1.0)
    n = Rc  # valence (== Qc on a closed manifold)
    vpts = (Qs / Qc[:, None] + 2.0 * (Rs / Rc[:, None])
            + (n - 3.0)[:, None] * verts) / n[:, None]

    new_verts = np.vstack([vpts, fp, epts])
    foff, eoff = V, V + F
    edge_id = inv.reshape(F, 4)  # edge i runs corner i -> corner i+1
    fcenter = foff + np.arange(F)
    quads = []
    for i in range(4):
        quads.append(np.stack([
            faces[:, i],                     # moved original corner
            eoff + edge_id[:, i],            # edge point ahead of it
            fcenter,                         # face point
            eoff + edge_id[:, (i - 1) % 4],  # edge point behind it
        ], axis=1))
    return new_verts, np.concatenate(quads, axis=0).astype(np.int64)


def subdivide_quads(verts, faces, levels: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """``levels`` rounds of Catmull-Clark on a closed quad cage."""
    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 4)
    for _ in range(int(levels)):
        verts, faces = _cc_once(verts, faces)
    return verts, faces


def quads_to_mesh(verts, faces) -> Mesh:
    """Triangulate a quad mesh (uniform A-C diagonal, winding preserved)."""
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 4)
    tris = np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]], axis=0)
    return Mesh(np.asarray(verts, dtype=np.float64), tris)


def subdivided_mesh(verts, faces, levels: int = 2) -> Mesh:
    """Cage -> smoothed triangle Mesh in one call."""
    v, f = subdivide_quads(verts, faces, levels)
    return quads_to_mesh(v, f)


# --------------------------------------------------------------------------
# cage-authoring helpers


def cube_cage(size=1.0, center=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """A unit closed quad cube (outward wound) — the smallest valid cage."""
    s = np.broadcast_to(np.asarray(size, dtype=np.float64), (3,)) / 2.0
    c = np.asarray(center, dtype=np.float64)
    v = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=np.float64) * s + c
    f = np.array([
        [3, 2, 1, 0],  # bottom (-Z)
        [4, 5, 6, 7],  # top (+Z)
        [0, 1, 5, 4],  # front (-Y)
        [2, 3, 7, 6],  # back (+Y)
        [1, 2, 6, 5],  # right (+X)
        [3, 0, 4, 7],  # left (-X)
    ], dtype=np.int64)
    return v, f


def loft_rings(rings: list[np.ndarray], close_bottom: bool = True,
               close_top: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Stitch N-vertex rings (all the same N, all wound CCW seen from +axis)
    into a closed quad tube with polar quad-fan caps.

    The caps are quad fans to a pole vertex: for N even this is N/2 quads per
    cap (pairs of ring edges per quad); N must be even.
    """
    N = len(rings[0])
    if N % 2 != 0:
        raise ValueError("loft_rings needs an even ring vertex count")
    if any(len(r) != N for r in rings):
        raise ValueError("all rings must share one vertex count")
    R = len(rings)
    verts = [np.asarray(r, dtype=np.float64) for r in rings]
    flat = np.vstack(verts)
    faces = []
    for i in range(R - 1):
        bi, bn = i * N, (i + 1) * N
        for k in range(N):
            kn = (k + 1) % N
            # outward for CCW rings stacked bottom->top
            faces.append([bi + k, bi + kn, bn + kn, bn + k])
    extra = []
    if close_bottom:
        pole = len(flat) + len(extra)
        extra.append(flat[0:N].mean(axis=0))
        for k in range(0, N, 2):
            faces.append([pole, (k + 2) % N, k + 1, k])
    if close_top:
        pole = len(flat) + len(extra)
        extra.append(flat[(R - 1) * N:R * N].mean(axis=0))
        b = (R - 1) * N
        for k in range(0, N, 2):
            faces.append([pole, b + k, b + k + 1, b + (k + 2) % N])
    if extra:
        flat = np.vstack([flat, np.asarray(extra)])
    return flat, np.asarray(faces, dtype=np.int64)
