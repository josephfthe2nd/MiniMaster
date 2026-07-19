"""Ray-mesh intersection (vectorized Möller-Trumbore).

Used for snap-to-surface part placement: a click becomes a world ray, the ray
is intersected against a shape's triangles, and the nearest front hit gives
the attachment point and outward normal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import Mesh

_EPS = 1e-9


def screen_ray(
    x: float, y: float, width: int, height: int, view: np.ndarray, focal: float
) -> tuple[np.ndarray, np.ndarray]:
    """World-space (origin, unit direction) for a canvas pixel.

    Exact inverse of the viewport projection ``sx = w/2 + focal*cx/(-cz)``,
    ``sy = h/2 - focal*cy/(-cz)`` under the world-to-camera matrix ``view``.
    """
    rot = view[:3, :3]
    eye = -rot.T @ view[:3, 3]
    d_cam = np.array(
        [(x - width / 2.0) / focal, -(y - height / 2.0) / focal, -1.0]
    )
    d_world = rot.T @ d_cam
    return eye, d_world / np.linalg.norm(d_world)


@dataclass
class RayHit:
    t: float  # distance along the (unit) ray direction
    point: np.ndarray  # world hit position
    normal: np.ndarray  # unit outward normal of the hit triangle
    face: int  # triangle index


def ray_mesh_hit(origin, direction, mesh: Mesh) -> RayHit | None:
    """Nearest intersection of ray(origin, direction) with the mesh, or None.

    Backfaces count as hits too (a click on a silhouette edge can graze a
    backface first); the returned normal is always the triangle's own outward
    normal.
    """
    if not len(mesh.faces):
        return None
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    n = np.linalg.norm(direction)
    if n < _EPS:
        raise ValueError("ray direction must be non-zero")
    direction = direction / n

    tri = mesh.triangles()  # (m, 3, 3)
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    e1 = v1 - v0
    e2 = v2 - v0
    pvec = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    ok = np.abs(det) > _EPS
    inv_det = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tvec = origin - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    qvec = np.cross(tvec, e1)
    v = np.einsum("j,ij->i", direction, qvec) * inv_det
    t = np.einsum("ij,ij->i", e2, qvec) * inv_det
    hit = ok & (u >= -_EPS) & (v >= -_EPS) & (u + v <= 1.0 + _EPS) & (t > _EPS)
    if not hit.any():
        return None
    idx = int(np.argmin(np.where(hit, t, np.inf)))
    normal = np.cross(e1[idx], e2[idx])
    nl = np.linalg.norm(normal)
    if nl < _EPS:
        return None
    return RayHit(
        t=float(t[idx]),
        point=origin + direction * float(t[idx]),
        normal=normal / nl,
        face=idx,
    )


def raycast_meshes(named_meshes, origin, direction):
    """Nearest hit across ``[(key, Mesh), ...]``: returns (key, RayHit) or None.

    The nearest hit along an eye ray is the visually front surface, which is
    exactly what snap-to-surface placement wants.
    """
    best_key, best = None, None
    for key, mesh in named_meshes:
        hit = ray_mesh_hit(origin, direction, mesh)
        if hit is not None and (best is None or hit.t < best.t):
            best_key, best = key, hit
    if best is None:
        return None
    return best_key, best
