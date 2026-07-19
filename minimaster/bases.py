"""Token bases the mini stands on.

Bases are built with their *top* surface at z=0 (the figure stands on z=0), so
they occupy z in [-height, 0].
"""

from __future__ import annotations

from .core.mesh import Mesh
from .core.primitives import _revolve, box, icosphere

BASE_STYLES = ("round", "cobble", "square", "none")

# Deterministic cobble layout: (angle deg, radius fraction, size fraction,
# squash). Hand-placed so stones vary but never overhang the rim.
_COBBLES = (
    (10, 0.55, 0.34, 0.32), (65, 0.28, 0.30, 0.38), (105, 0.68, 0.28, 0.30),
    (150, 0.40, 0.36, 0.34), (200, 0.66, 0.30, 0.28), (245, 0.22, 0.28, 0.36),
    (285, 0.58, 0.34, 0.30), (330, 0.38, 0.26, 0.34), (0, 0.0, 0.38, 0.30),
)


def round_base(diameter: float = 25.0, height: float = 3.0, segments: int = 24) -> Mesh:
    r = diameter / 2.0
    profile = [
        (0.0, -height),
        (r, -height),
        (r, -height * 0.45),
        (r * 0.92, 0.0),  # beveled lip
        (0.0, 0.0),
    ]
    return _revolve(profile, segments)


def square_base(diameter: float = 25.0, height: float = 3.0) -> Mesh:
    return box(diameter, diameter, height).translated([0.0, 0.0, -height / 2.0])


def cobble_base(diameter: float = 25.0, height: float = 3.0,
                segments: int = 24) -> Mesh:
    """Round base with a carved cobblestone top: flattened boulders sunk into
    the surface (each an independent watertight shell)."""
    import numpy as np

    base = round_base(diameter, height, segments)
    r = diameter / 2.0
    stones = []
    for angle, rad_f, size_f, squash in _COBBLES:
        a = np.radians(angle)
        cx, cy = r * rad_f * np.cos(a), r * rad_f * np.sin(a)
        d = r * size_f
        stone = icosphere(0.5, 1).transform(
            # flattened boulder, mostly sunk below the top surface
            np.diag([d * 2.0, d * 1.7, d * squash * 2.0, 1.0])
        ).translated([cx, cy, d * squash * 0.55])
        stones.append(stone)
    return Mesh.merge([base, *stones])


def build_base(spec: dict | None) -> Mesh | None:
    """Build the base described by a scene's ``base`` dict (or None)."""
    if not spec:
        return None
    style = spec.get("style", "round")
    if style == "none":
        return None
    diameter = float(spec.get("diameter", 25.0))
    height = float(spec.get("height", 3.0))
    if diameter <= 0 or height <= 0:
        raise ValueError("base diameter and height must be positive")
    if style == "round":
        return round_base(diameter, height, int(spec.get("segments", 24)))
    if style == "cobble":
        return cobble_base(diameter, height, int(spec.get("segments", 24)))
    if style == "square":
        return square_base(diameter, height)
    raise ValueError(f"unknown base style {style!r}; known: {BASE_STYLES}")
