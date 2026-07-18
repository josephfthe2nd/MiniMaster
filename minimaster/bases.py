"""Token bases the mini stands on.

Bases are built with their *top* surface at z=0 (the figure stands on z=0), so
they occupy z in [-height, 0].
"""

from __future__ import annotations

from .core.mesh import Mesh
from .core.primitives import _revolve, box

BASE_STYLES = ("round", "square", "none")


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
    if style == "square":
        return square_base(diameter, height)
    raise ValueError(f"unknown base style {style!r}; known: {BASE_STYLES}")
