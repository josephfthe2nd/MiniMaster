"""Binary STL writing and reading.

Writing is the export path for printing; reading exists so tests (and users)
can round-trip and validate produced files. Binary STL only — ASCII STL is
rejected with a clear error.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .mesh import Mesh

_HEADER_SIZE = 80
_TRI_DTYPE = np.dtype(
    [
        ("normal", "<f4", (3,)),
        ("v0", "<f4", (3,)),
        ("v1", "<f4", (3,)),
        ("v2", "<f4", (3,)),
        ("attr", "<u2"),
    ]
)


def write_stl(mesh: Mesh, path, name: str = "minimaster") -> None:
    path = Path(path)
    tris = mesh.triangles().astype(np.float32)
    normals = mesh.face_normals(normalized=True).astype(np.float32)
    record = np.zeros(len(tris), dtype=_TRI_DTYPE)
    record["normal"] = normals
    record["v0"] = tris[:, 0]
    record["v1"] = tris[:, 1]
    record["v2"] = tris[:, 2]
    header = f"MiniMaster STL: {name}".encode("ascii", "replace")[:_HEADER_SIZE]
    header = header.ljust(_HEADER_SIZE, b" ")
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(struct.pack("<I", len(tris)))
        fh.write(record.tobytes())


def read_stl(path) -> Mesh:
    """Read a binary STL, welding exactly-coincident vertices."""
    path = Path(path)
    data = path.read_bytes()
    if len(data) < _HEADER_SIZE + 4:
        raise ValueError(f"{path}: too small to be a binary STL")
    (count,) = struct.unpack_from("<I", data, _HEADER_SIZE)
    expected = _HEADER_SIZE + 4 + count * _TRI_DTYPE.itemsize
    if len(data) != expected:
        if data.lstrip()[:5] == b"solid":
            raise ValueError(f"{path}: looks like an ASCII STL; only binary is supported")
        raise ValueError(
            f"{path}: corrupt binary STL (size {len(data)}, expected {expected} "
            f"for {count} triangles)"
        )
    record = np.frombuffer(data, dtype=_TRI_DTYPE, count=count, offset=_HEADER_SIZE + 4)
    corners = np.stack(
        [record["v0"], record["v1"], record["v2"]], axis=1
    ).reshape(-1, 3)
    verts, inverse = np.unique(corners, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    return Mesh(verts.astype(np.float64), faces)
