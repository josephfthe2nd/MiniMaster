"""Triangle mesh container with transforms, merging, and integrity checks.

A mesh here is always a triangle soup with shared vertices: ``vertices`` is
(n, 3) float64 and ``faces`` is (m, 3) int64 indices. Printable shells must be
watertight and consistently outward-oriented; ``integrity_report`` verifies
both. A scene exports as several overlapping watertight shells in one STL,
which every slicer unions at slice time.
"""

from __future__ import annotations

import numpy as np


class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        if len(self.faces) and self.faces.max() >= len(self.vertices):
            raise ValueError("face index out of range")

    def copy(self) -> "Mesh":
        return Mesh(self.vertices.copy(), self.faces.copy())

    # -- geometry ---------------------------------------------------------

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if not len(self.vertices):
            z = np.zeros(3)
            return z, z
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def transform(self, m: np.ndarray) -> "Mesh":
        """Return the mesh through a 4x4 affine. Winding is flipped when the
        transform is mirroring (negative determinant) so shells stay
        outward-oriented."""
        verts = self.vertices @ m[:3, :3].T + m[:3, 3]
        faces = self.faces
        if np.linalg.det(m[:3, :3]) < 0:
            faces = faces[:, [0, 2, 1]]
        return Mesh(verts, faces)

    def translated(self, v) -> "Mesh":
        return Mesh(self.vertices + np.asarray(v, dtype=np.float64), self.faces)

    def scaled(self, s) -> "Mesh":
        s = np.asarray(s, dtype=np.float64)
        faces = self.faces
        if s.ndim and np.prod(s) < 0 or s.ndim == 0 and s < 0:
            faces = faces[:, [0, 2, 1]]
        return Mesh(self.vertices * s, faces)

    @staticmethod
    def merge(meshes) -> "Mesh":
        meshes = [m for m in meshes if len(m.faces)]
        if not meshes:
            return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
        verts, faces, offset = [], [], 0
        for m in meshes:
            verts.append(m.vertices)
            faces.append(m.faces + offset)
            offset += len(m.vertices)
        return Mesh(np.vstack(verts), np.vstack(faces))

    def triangles(self) -> np.ndarray:
        """(m, 3, 3) corner positions."""
        return self.vertices[self.faces]

    def face_normals(self, normalized: bool = True) -> np.ndarray:
        t = self.triangles()
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        if normalized:
            lens = np.linalg.norm(n, axis=1, keepdims=True)
            n = np.divide(n, lens, out=np.zeros_like(n), where=lens > 1e-14)
        return n

    def area(self) -> float:
        return float(np.linalg.norm(self.face_normals(normalized=False), axis=1).sum() / 2.0)

    def volume(self) -> float:
        """Signed volume; positive for watertight outward-oriented shells."""
        t = self.triangles()
        return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)

    # -- integrity --------------------------------------------------------

    def degenerate_faces(self, eps: float = 1e-12) -> np.ndarray:
        """Indices of faces with repeated vertices or ~zero area."""
        f = self.faces
        repeated = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
        areas = np.linalg.norm(self.face_normals(normalized=False), axis=1) / 2.0
        return np.nonzero(repeated | (areas < eps))[0]

    def integrity_report(self) -> dict:
        """Watertightness / orientation report.

        A closed, consistently oriented surface (or disjoint union of such
        shells) has every directed edge appearing exactly once and every
        undirected edge appearing exactly twice.
        """
        f = self.faces
        if not len(f):
            return {
                "watertight": False,
                "boundary_edges": 0,
                "nonmanifold_edges": 0,
                "duplicate_directed_edges": 0,
                "degenerate_faces": 0,
                "volume": 0.0,
                "empty": True,
            }
        directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
        # Encode edge (a, b) as a single integer for fast uniqueness checks.
        nv = int(len(self.vertices))
        dir_keys = directed[:, 0].astype(np.int64) * nv + directed[:, 1]
        dup_directed = int(len(dir_keys) - len(np.unique(dir_keys)))
        und = np.sort(directed, axis=1)
        und_keys = und[:, 0].astype(np.int64) * nv + und[:, 1]
        _, counts = np.unique(und_keys, return_counts=True)
        boundary = int((counts == 1).sum())
        nonmanifold = int((counts > 2).sum())
        degenerate = int(len(self.degenerate_faces()))
        vol = self.volume()
        watertight = (
            boundary == 0 and nonmanifold == 0 and dup_directed == 0 and degenerate == 0
        )
        return {
            "watertight": watertight,
            "boundary_edges": boundary,
            "nonmanifold_edges": nonmanifold,
            "duplicate_directed_edges": dup_directed,
            "degenerate_faces": degenerate,
            "volume": vol,
            "empty": False,
        }

    @property
    def is_watertight(self) -> bool:
        return bool(self.integrity_report()["watertight"])

    def __repr__(self) -> str:
        return f"Mesh({len(self.vertices)} verts, {len(self.faces)} tris)"
