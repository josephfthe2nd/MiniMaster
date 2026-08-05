"""Turn a morphed character into a print-ready solid.

The realistic Character Lab path could previously only render pictures. This
is the other half: scale to a real miniature size, stand the figure on z=0,
attach a base, gate every shell, and hand back a mesh the STL writer accepts.

Two facts drive the design:

* **The character is several shells that overlap** (body, two eyeballs, teeth),
  not one solid. MiniMaster's export contract has always been "overlapping
  watertight shells, unioned by the slicer at slice time" — every slicer does
  this correctly, and it is why no boolean union is needed. Each shell is
  checked individually; the union is the slicer's job.
* **The mesh is authored in decimetres-ish MakeHuman units** where a default
  adult is ~17.5 units for 175 cm, i.e. 1 unit = 10 cm. Rather than trust that
  constant, the scale is derived from the *actual* height of the built figure,
  so it stays correct under every morph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bases import build_base
from ..core.mesh import Mesh

# Total figure height in mm for the standard tabletop categories.
SIZE_PRESETS = {"tiny": 15.0, "small": 24.0, "medium": 32.0,
                "large": 45.0, "huge": 60.0, "bust": 90.0, "display": 150.0}

# Below this, a feature is finer than a good resin printer can resolve.
RESIN_XY_MM = 0.035
RESIN_MIN_FEATURE_MM = 0.20


class PrintError(RuntimeError):
    pass


@dataclass
class PrintReport:
    height_mm: float
    scale: float
    triangles: int
    shells: int
    volume_mm3: float
    watertight: bool
    notes: list

    def as_text(self) -> str:
        lines = [
            f"height       {self.height_mm:.1f} mm",
            f"scale        x{self.scale:.4f}",
            f"shells       {self.shells}",
            f"triangles    {self.triangles}",
            f"resin volume {self.volume_mm3 / 1000.0:.2f} cm3",
            f"watertight   {self.watertight}",
        ]
        lines += [f"note         {n}" for n in self.notes]
        return "\n".join(lines)


def resolve_height(size=None, height=None) -> float:
    if size is not None and height is not None:
        raise PrintError("give either size or height, not both")
    if height is not None:
        if height <= 0:
            raise PrintError("height must be positive")
        return float(height)
    key = size or "medium"
    if key not in SIZE_PRESETS:
        raise PrintError(f"unknown size {key!r}; known: {sorted(SIZE_PRESETS)}")
    return SIZE_PRESETS[key]


def check_shells(shells) -> list:
    """Verify every shell independently. Returns per-shell reports; raises on
    the first shell that is not a printable solid."""
    reports = []
    for name, mesh in shells:
        if not len(mesh.faces):
            raise PrintError(f"shell {name!r} is empty")
        rep = mesh.integrity_report()
        if not rep["watertight"]:
            raise PrintError(
                f"shell {name!r} is not watertight "
                f"(boundary={rep['boundary_edges']}, "
                f"nonmanifold={rep['nonmanifold_edges']}, "
                f"degenerate={rep['degenerate_faces']})")
        if rep["volume"] <= 0:
            raise PrintError(f"shell {name!r} is inside-out (volume "
                             f"{rep['volume']:.4f})")
        reports.append((name, rep))
    return reports


def feature_notes(height_mm: float, figure_height_units: float) -> list:
    """Advisory notes about what survives at this scale."""
    notes = []
    mm_per_unit = height_mm / max(figure_height_units, 1e-9)
    # a human eye aperture is ~2.5 cm across => 0.25 MakeHuman units
    eye_mm = 0.25 * mm_per_unit
    lip_mm = 0.02 * mm_per_unit  # the lip crack
    if lip_mm < RESIN_XY_MM:
        notes.append(
            f"lip/eyelid creases are ~{lip_mm:.3f} mm here, below a resin "
            f"printer's {RESIN_XY_MM} mm pixel - they will not resolve")
    if eye_mm < RESIN_MIN_FEATURE_MM:
        notes.append(
            f"eye aperture ~{eye_mm:.2f} mm: facial detail will read as soft; "
            "consider a larger size (bust/display) for portrait work")
    if height_mm >= 70:
        notes.append("at this scale facial geometry is worth the detail; "
                     "consider subdividing before export")
    return notes


def assemble_character(shells, size=None, height=None, with_base=True,
                       base_spec=None, check=True, sink=0.0):
    """Scale, stand, base and merge a character into one printable mesh.

    ``shells`` is ``[(name, Mesh)]`` in scene units. Returns
    ``(mesh, PrintReport)``.
    """
    shells = [(n, m) for n, m in shells if len(m.faces)]
    if not shells:
        raise PrintError("character has no geometry to export")
    target = resolve_height(size, height)

    figure = Mesh.merge([m for _, m in shells])
    lo, hi = figure.bounds
    extent = float(hi[2] - lo[2])
    if extent <= 1e-9:
        raise PrintError("figure has no height to scale")
    scale = target / extent

    scaled = [(n, m.scaled(scale)) for n, m in shells]
    reports = check_shells(scaled) if check else []

    figure = Mesh.merge([m for _, m in scaled])
    lo, hi = figure.bounds
    center = (lo + hi) / 2.0
    # centre on XY, stand on z=0, then optionally sink into the base a touch
    offset = np.array([-center[0], -center[1], -lo[2] + sink])
    placed = [(n, m.translated(offset)) for n, m in scaled]

    parts = [m for _, m in placed]
    if with_base:
        spec = base_spec if base_spec is not None else {
            "style": "round", "diameter": max(target * 0.78, 12.0),
            "height": max(target * 0.09, 2.0)}
        base_mesh = build_base(spec)
        if base_mesh is not None:
            parts.append(base_mesh)

    merged = Mesh.merge(parts)
    total_volume = sum(r["volume"] for _, r in reports) if reports else \
        merged.volume()
    report = PrintReport(
        height_mm=target,
        scale=scale,
        triangles=int(len(merged.faces)),
        shells=len(parts),
        volume_mm3=float(abs(total_volume)),
        watertight=all(r["watertight"] for _, r in reports) if reports else
        bool(merged.integrity_report()["watertight"]),
        notes=feature_notes(target, extent),
    )
    return merged, report


def character_print_shells(base, verts=None, levels: int = 0,
                           with_eyes: bool = True, with_teeth: bool = True):
    """Named closed shells for printing, in Z-up scene units.

    Only *closed* islands are included — open assets (hair cards, clothes)
    would fail the per-shell gate and are handled separately.
    """
    from ..core.subdiv import quads_to_mesh, subdivide_quads
    from .portrait import MH_TO_Z_UP

    def build(v, q):
        if levels:
            v, q = subdivide_quads(v, q, levels)
        return quads_to_mesh(v, q).transform(MH_TO_Z_UP)

    out = [("body", build(*base.body_cage(verts)))]
    names = []
    if with_eyes:
        names += ["helper-l-eye", "helper-r-eye"]
    if with_teeth:
        names += ["helper-upper-teeth", "helper-lower-teeth"]
    for name in names:
        if name not in base.groups:
            continue
        mesh = build(*base.helper_group(name, verts))
        if mesh.integrity_report()["watertight"]:
            out.append((name.replace("helper-", ""), mesh))
    return out
