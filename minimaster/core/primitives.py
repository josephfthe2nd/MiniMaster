"""Watertight low-poly primitive builders.

Every builder returns a closed, outward-oriented :class:`Mesh` centered on the
origin (bounding-box centered), sized in scene units before the owning shape's
transform is applied. Low segment counts are the point — the faceted look is
the carved-token aesthetic.

The ``PRIMITIVES`` registry maps primitive kind names to (builder,
default_params); scene shapes are instantiated through :func:`build`.
"""

from __future__ import annotations

import numpy as np

from .mesh import Mesh


def _revolve(profile, segments: int) -> Mesh:
    """Revolve a polyline profile of (radius, z) pairs around +Z.

    The first and last profile points must have radius 0 (poles); interior
    points must have radius > 0. Produces a closed, outward-oriented surface.
    """
    segments = int(segments)
    if segments < 3:
        raise ValueError("segments must be >= 3")
    profile = [(float(r), float(z)) for r, z in profile]
    if len(profile) < 3:
        raise ValueError("profile needs at least 3 points")
    if profile[0][0] != 0.0 or profile[-1][0] != 0.0:
        raise ValueError("profile must start and end at radius 0")
    for r, _ in profile[1:-1]:
        if r <= 0.0:
            raise ValueError("interior profile points must have radius > 0")

    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    verts: list[np.ndarray] = []
    rows: list[np.ndarray] = []  # per profile row: vertex indices (1 or segments)
    for r, z in profile:
        start = sum(len(v) for v in verts)
        if r == 0.0:
            rows.append(np.array([start]))
            verts.append(np.array([[0.0, 0.0, z]]))
        else:
            rows.append(np.arange(start, start + segments, dtype=np.int64))
            ring = np.column_stack([r * cos_t, r * sin_t, np.full(segments, z)])
            verts.append(ring)
    vertices = np.vstack(verts)

    faces: list[tuple[int, int, int]] = []
    s1 = np.arange(segments)
    s2 = (s1 + 1) % segments
    for lower, upper in zip(rows[:-1], rows[1:]):
        if len(lower) == 1:  # bottom apex fan
            a = lower[0]
            for i, j in zip(s1, s2):
                faces.append((a, upper[j], upper[i]))
        elif len(upper) == 1:  # top apex fan
            a = upper[0]
            for i, j in zip(s1, s2):
                faces.append((a, lower[i], lower[j]))
        else:  # quad band
            for i, j in zip(s1, s2):
                faces.append((lower[i], lower[j], upper[j]))
                faces.append((lower[i], upper[j], upper[i]))
    return Mesh(vertices, np.array(faces, dtype=np.int64))


def _require_positive(**dims: float) -> None:
    """Negative or zero dimensions would build inverted or degenerate shells
    (inward-facing normals slice as cavities), so reject them up front."""
    for name, value in dims.items():
        if not value > 0.0:
            raise ValueError(f"{name} must be positive, got {value}")


def box(width: float = 1.0, depth: float = 1.0, height: float = 1.0) -> Mesh:
    """Axis-aligned box: width along X, depth along Y, height along Z."""
    _require_positive(width=width, depth=depth, height=height)
    x, y, z = width / 2.0, depth / 2.0, height / 2.0
    v = np.array(
        [
            [-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
            [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z],
        ]
    )
    f = np.array(
        [
            [0, 2, 1], [0, 3, 2],  # bottom (normal -Z)
            [4, 5, 6], [4, 6, 7],  # top (+Z)
            [0, 1, 5], [0, 5, 4],  # front (-Y)
            [2, 3, 7], [2, 7, 6],  # back (+Y)
            [1, 2, 6], [1, 6, 5],  # right (+X)
            [3, 0, 4], [3, 4, 7],  # left (-X)
        ],
        dtype=np.int64,
    )
    return Mesh(v, f)


def wedge(width: float = 1.0, depth: float = 1.0, height: float = 1.0) -> Mesh:
    """Right-triangular prism: rectangular base, vertical face at -X, sloped
    face falling toward +X, extruded along Y. Bounding-box centered."""
    _require_positive(width=width, depth=depth, height=height)
    x, y, z = width / 2.0, depth / 2.0, height / 2.0
    v = np.array(
        [
            [-x, -y, -z], [x, -y, -z], [-x, -y, z],  # front triangle (-Y)
            [-x, y, -z], [x, y, -z], [-x, y, z],     # back triangle (+Y)
        ]
    )
    f = np.array(
        [
            [0, 1, 2],            # front cap (-Y)
            [3, 5, 4],            # back cap (+Y)
            [0, 3, 4], [0, 4, 1],  # bottom (-Z)
            [0, 2, 5], [0, 5, 3],  # vertical face (-X)
            [1, 4, 5], [1, 5, 2],  # slope
        ],
        dtype=np.int64,
    )
    return Mesh(v, f)


def cylinder(
    radius: float = 0.5,
    height: float = 1.0,
    segments: int = 8,
    taper: float = 1.0,
) -> Mesh:
    """N-gon prism. ``taper`` scales the top radius: 1 = prism, 0 = cone,
    values between give a frustum. ``segments=4`` with taper gives pyramids."""
    _require_positive(radius=radius, height=height)
    if taper < 0.0:
        raise ValueError(f"taper must be >= 0, got {taper}")
    r_top = radius * float(taper)
    z = height / 2.0
    profile: list[tuple[float, float]] = [(0.0, -z), (radius, -z)]
    if r_top > 1e-9:
        profile.append((r_top, z))
    profile.append((0.0, z))
    return _revolve(profile, segments)


def capsule(radius: float = 0.5, height: float = 1.0, segments: int = 8) -> Mesh:
    """Capsule of total height ``height`` (clamped to >= 2*radius) with
    low-poly faceted caps."""
    _require_positive(radius=radius, height=height)
    height = max(float(height), 2.0 * radius)
    mid = height / 2.0 - radius
    profile: list[tuple[float, float]] = [(0.0, -mid - radius)]
    cap_angles = (40.0, 75.0)
    for a in cap_angles:
        t = np.radians(a)
        profile.append((radius * np.sin(t), -mid - radius * np.cos(t)))
    if mid > 1e-9:
        profile.append((radius, -mid))
        profile.append((radius, mid))
    else:  # degenerate mid-section: the capsule is a sphere
        profile.append((radius, 0.0))
    for a in reversed(cap_angles):
        t = np.radians(a)
        profile.append((radius * np.sin(t), mid + radius * np.cos(t)))
    profile.append((0.0, mid + radius))
    return _revolve(profile, segments)


_ICO_T = (1.0 + np.sqrt(5.0)) / 2.0
_ICO_VERTS = np.array(
    [
        [-1, _ICO_T, 0], [1, _ICO_T, 0], [-1, -_ICO_T, 0], [1, -_ICO_T, 0],
        [0, -1, _ICO_T], [0, 1, _ICO_T], [0, -1, -_ICO_T], [0, 1, -_ICO_T],
        [_ICO_T, 0, -1], [_ICO_T, 0, 1], [-_ICO_T, 0, -1], [-_ICO_T, 0, 1],
    ],
    dtype=np.float64,
)
_ICO_FACES = np.array(
    [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ],
    dtype=np.int64,
)


def icosphere(radius: float = 0.5, subdivisions: int = 1) -> Mesh:
    """Icosphere. ``subdivisions=0`` is a raw icosahedron (the most 'carved'
    ball); 1-2 get progressively rounder."""
    _require_positive(radius=radius)
    subdivisions = int(subdivisions)
    if not 0 <= subdivisions <= 3:
        raise ValueError("subdivisions must be in 0..3")
    verts = [v / np.linalg.norm(v) for v in _ICO_VERTS]
    faces = [tuple(f) for f in _ICO_FACES]
    for _ in range(subdivisions):
        midpoint: dict[tuple[int, int], int] = {}

        def mid(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            if key not in midpoint:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                midpoint[key] = len(verts) - 1
            return midpoint[key]

        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces
    return Mesh(np.array(verts) * float(radius), np.array(faces, dtype=np.int64))


def torus(
    radius: float = 0.5,
    thickness: float = 0.2,
    segments: int = 8,
    minor_segments: int = 6,
) -> Mesh:
    """Torus in the XY plane: ``radius`` to the tube center, tube diameter
    ``thickness``."""
    _require_positive(radius=radius, thickness=thickness)
    segments, minor_segments = int(segments), int(minor_segments)
    if segments < 3 or minor_segments < 3:
        raise ValueError("torus needs >= 3 segments on both axes")
    if thickness >= 2.0 * radius:
        raise ValueError(
            f"torus thickness ({thickness}) must be < 2*radius ({2 * radius}); "
            "a fatter tube would pass through its own axis"
        )
    r_minor = float(thickness) / 2.0
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi, minor_segments, endpoint=False)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    ring = radius + r_minor * np.cos(pp)
    verts = np.column_stack(
        [(ring * np.cos(tt)).ravel(), (ring * np.sin(tt)).ravel(), (r_minor * np.sin(pp)).ravel()]
    )
    faces = []
    for s in range(segments):
        s2 = (s + 1) % segments
        for t in range(minor_segments):
            t2 = (t + 1) % minor_segments
            a = s * minor_segments + t
            b = s2 * minor_segments + t
            c = s2 * minor_segments + t2
            d = s * minor_segments + t2
            faces += [(a, b, c), (a, c, d)]
    return Mesh(verts, np.array(faces, dtype=np.int64))


PRIMITIVES: dict[str, tuple] = {
    "box": (box, {"width": 1.0, "depth": 1.0, "height": 1.0}),
    "wedge": (wedge, {"width": 1.0, "depth": 1.0, "height": 1.0}),
    "cylinder": (cylinder, {"radius": 0.5, "height": 1.0, "segments": 8, "taper": 1.0}),
    "capsule": (capsule, {"radius": 0.5, "height": 1.0, "segments": 8}),
    "icosphere": (icosphere, {"radius": 0.5, "subdivisions": 1}),
    "torus": (torus, {"radius": 0.5, "thickness": 0.2, "segments": 8, "minor_segments": 6}),
}


def build(kind: str, params: dict | None = None) -> Mesh:
    """Instantiate a registered primitive, filling in default params."""
    if kind not in PRIMITIVES:
        raise KeyError(f"unknown primitive {kind!r}; known: {sorted(PRIMITIVES)}")
    builder, defaults = PRIMITIVES[kind]
    merged = dict(defaults)
    if params:
        unknown = set(params) - set(defaults)
        if unknown:
            raise KeyError(f"unknown params for {kind!r}: {sorted(unknown)}")
        merged.update(params)
    return builder(**merged)


def default_params(kind: str) -> dict:
    return dict(PRIMITIVES[kind][1])
