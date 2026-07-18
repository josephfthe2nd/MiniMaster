"""Assemble a posed scene into a print-ready mesh and write STL files.

This is the single pipeline shared by the GUI, the CLI, and tests:

    scene -> FK pose -> per-shape posed shells -> scale to target height
          -> stand on z=0, centered -> + base -> merged STL
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .bases import build_base
from .core.mesh import Mesh
from .core.stl import write_stl
from .scene import Scene

# Total figure heights (mm) for tabletop size categories, heroic-ish scale.
SIZE_PRESETS = {"tiny": 15.0, "small": 24.0, "medium": 32.0, "large": 45.0, "huge": 60.0}


class ExportError(RuntimeError):
    pass


def assemble(
    scene: Scene,
    pose_name: str | None = "__active__",
    height: float | None = None,
    size: str | None = None,
    with_base: bool = True,
    base_override: dict | None = None,
    check: bool = True,
) -> Mesh:
    """Build the final printable mesh for a scene.

    ``height`` (mm) or ``size`` (a SIZE_PRESETS key) uniformly scales the
    figure to that total height; with neither, scene units are used as mm
    directly. The figure is centered on XY and stands on z=0; the base (from
    the scene's base spec unless overridden) sits below z=0.
    """
    if height is not None and size is not None:
        raise ExportError("give either height or size, not both")
    if size is not None:
        if size not in SIZE_PRESETS:
            raise ExportError(f"unknown size {size!r}; known: {sorted(SIZE_PRESETS)}")
        height = SIZE_PRESETS[size]

    figure = scene.build_merged_mesh(pose_name)
    if not len(figure.faces):
        raise ExportError("scene has no shapes to export")

    lo, hi = figure.bounds
    extent_z = hi[2] - lo[2]
    if height is not None:
        if extent_z <= 1e-9:
            raise ExportError("figure has no height to scale")
        figure = figure.scaled(height / extent_z)
        lo, hi = figure.bounds
    center = (lo + hi) / 2.0
    figure = figure.translated([-center[0], -center[1], -lo[2]])

    parts = [figure]
    if with_base:
        base_mesh = build_base(base_override if base_override is not None else scene.base)
        if base_mesh is not None:
            parts.append(base_mesh)
    merged = Mesh.merge(parts)

    if check:
        report = merged.integrity_report()
        if not report["watertight"]:
            raise ExportError(f"assembled mesh is not watertight: {report}")
    return merged


def export_stl(scene: Scene, path, **kwargs) -> dict:
    """Assemble and write a binary STL. Returns a summary report."""
    mesh = assemble(scene, **kwargs)
    path = Path(path)
    write_stl(mesh, path, name=scene.name)
    lo, hi = mesh.bounds
    return {
        "path": str(path),
        "triangles": int(len(mesh.faces)),
        "size_mm": [float(v) for v in (hi - lo)],
        "volume_mm3": mesh.volume(),
        "watertight": bool(mesh.integrity_report()["watertight"]),
    }
