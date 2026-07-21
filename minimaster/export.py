"""Assemble a posed scene into a print-ready mesh and write STL files.

This is the single pipeline shared by the GUI, the CLI, and tests:

    scene -> FK pose -> per-shape posed shells -> scale to target height
          -> stand on z=0, centered -> + base -> merged STL
"""

from __future__ import annotations

from pathlib import Path

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

    shape_meshes = scene.build_shape_meshes(pose_name)
    if check:
        for shape, mesh in shape_meshes:
            rep = mesh.integrity_report()
            if not rep["watertight"] or not rep["outward"]:
                raise ExportError(
                    f"shape {shape.name!r} is not a printable shell "
                    f"(watertight={rep['watertight']}, outward={rep['outward']}, "
                    f"volume={rep['volume']:.3f})"
                )
    figure = Mesh.merge([mesh for _, mesh in shape_meshes])
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
        try:
            base_mesh = build_base(
                base_override if base_override is not None else scene.base
            )
        except ValueError as exc:
            raise ExportError(str(exc)) from exc
        if base_mesh is not None:
            parts.append(base_mesh)
    merged = Mesh.merge(parts)

    if check:
        report = merged.integrity_report()
        if not report["watertight"]:
            raise ExportError(f"assembled mesh is not watertight: {report}")
    return merged


def _place_on_base(figure: Mesh, scene: Scene, height, size, with_base,
                   base_override=None) -> Mesh:
    """Scale ``figure`` to the target height, stand it on z=0 centered, and
    drop the base beneath it — the shared tail of the assemble pipelines."""
    if size is not None:
        if size not in SIZE_PRESETS:
            raise ExportError(f"unknown size {size!r}; known: {sorted(SIZE_PRESETS)}")
        height = SIZE_PRESETS[size]
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
        try:
            base_mesh = build_base(
                base_override if base_override is not None else scene.base
            )
        except ValueError as exc:
            raise ExportError(str(exc)) from exc
        if base_mesh is not None:
            parts.append(base_mesh)
    return Mesh.merge(parts)


def assemble_body(
    scene: Scene,
    pose_name: str | None = "__active__",
    resolution: float = 0.6,
    blend: float = 0.6,
    height: float | None = None,
    size: str | None = None,
    with_base: bool = True,
    exclude=None,
    auto_watertight: bool = True,
) -> tuple[Mesh, dict]:
    """Bake the scene into ONE fused body solid, scaled and based for print.

    Unlike :func:`assemble` (a pile of shells the slicer unions), this returns a
    single continuous skin from the signed-distance field. The manifold dual-
    contouring extractor keeps the surface watertight even where limbs cross;
    ``auto_watertight`` is a belt-and-suspenders net that widens the blend to
    recover from any residual degenerate config. Returns ``(mesh, report)``
    where report carries the final blend and watertightness.
    """
    from .core import bodymesh

    if height is not None and size is not None:
        raise ExportError("give either height or size, not both")
    kw = {} if exclude is None else {"exclude": exclude}
    used_blend = blend
    figure = bodymesh.body_from_scene(scene, resolution, blend, pose_name, **kw)
    if auto_watertight and not figure.integrity_report()["watertight"]:
        for factor in (1.6, 2.4, 3.4):
            trial = bodymesh.body_from_scene(
                scene, resolution, blend * factor, pose_name, **kw)
            if trial.integrity_report()["watertight"]:
                figure, used_blend = trial, blend * factor
                break
    if not len(figure.faces):
        raise ExportError("scene baked to an empty body (no body shapes?)")
    merged = _place_on_base(figure, scene, height, size, with_base)
    rep = merged.integrity_report()
    return merged, {
        "triangles": int(len(merged.faces)),
        "watertight": bool(rep["watertight"]),
        "blend": used_blend,
        "resolution": resolution,
    }


def export_stl(scene: Scene, path, **kwargs) -> dict:
    """Assemble and write a binary STL. Returns a summary report."""
    check = kwargs.pop("check", True)
    mesh = assemble(scene, check=check, **kwargs)
    path = Path(path)
    write_stl(mesh, path, name=scene.name)
    lo, hi = mesh.bounds
    return {
        "path": str(path),
        "triangles": int(len(mesh.faces)),
        "size_mm": [float(v) for v in (hi - lo)],
        "volume_mm3": mesh.volume(),
        # assemble(check=True) raises on any integrity problem, so reaching
        # this point with check on means the mesh passed the gate.
        "watertight": check or bool(mesh.integrity_report()["watertight"]),
    }
