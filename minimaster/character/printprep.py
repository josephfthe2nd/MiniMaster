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

Appendages (wings, tails, horns) join through :func:`appendage_print_shells`, which
is the one place that moves them from the MakeHuman Y-up frame the appendage
objects live in to the Z-up frame of the print shells. ``assemble_character``'s
``reference=`` then makes the figure -- not the wing tips -- set the scale, the
centring and the base, so "medium / 32 mm" stays the height of the BODY.
"""

from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass, field

import numpy as np

from ..bases import build_base
from ..core.mesh import Mesh

# Total figure height in mm for the standard tabletop categories.
SIZE_PRESETS = {"tiny": 15.0, "small": 24.0, "medium": 32.0,
                "large": 45.0, "huge": 60.0, "bust": 90.0, "display": 150.0}

# Slack added to the farthest appendage when fit_base widens the base.
BASE_MARGIN_MM = 1.5

# Below this, a feature is finer than a good resin printer can resolve.
RESIN_XY_MM = 0.035
RESIN_MIN_FEATURE_MM = 0.20


# The shells that ARE the figure. ``reference=FIGURE_SHELLS`` in assemble_character.
FIGURE_SHELLS = ("body", "l-eye", "r-eye", "upper-teeth", "lower-teeth")

# Printed thickness the wing sheets are never allowed below when the floor is switched
# on (``min_thickness_mm``): the 0.3 mm the wing builders were tuned around.
DEFAULT_MIN_THICKNESS_MM = 0.30


class PrintError(RuntimeError):
    """A figure that cannot be printed. ``failures`` lists every ``(shell, reason)`` the
    gate found, not only the one the message leads with."""

    def __init__(self, message, failures=()):
        super().__init__(message)
        self.failures = list(failures)


@dataclass
class PrintReport:
    height_mm: float                  # the requested height: the FIGURE's (see ``reference``)
    scale: float                      # mm per scene unit
    triangles: int
    shells: int
    # Sum of the per-shell volumes. Overlapping shells (every feather on a wing, a
    # wing root inside the torso) are counted once PER SHELL, so this is an upper
    # bound on what the slicer prints (+18% on a bat wing, +82% on an angel wing),
    # not the resin volume. The base is not included.
    shell_volume_mm3: float
    watertight: bool
    notes: list
    # None unless ``reference=`` was given
    overall_height_mm: float | None = None    # figure + appendages, from z=0
    base_diameter_mm: float | None = None
    overhang_mm: float = 0.0          # how far the farthest shell sticks out past the base rim
    # the shells as they were placed (mm, the frame of the written STL), for post-assembly
    # checks; not part of the report's identity
    placed_shells: list = field(default_factory=list, repr=False, compare=False)

    @property
    def volume_mm3(self) -> float:
        """Deprecated alias of :attr:`shell_volume_mm3` (it is NOT the union volume)."""
        return self.shell_volume_mm3

    def as_text(self) -> str:
        lines = [f"height       {self.height_mm:.1f} mm"]
        if (self.overall_height_mm is not None
                and abs(self.overall_height_mm - self.height_mm) > 0.05):
            lines.append(f"overall      {self.overall_height_mm:.1f} mm with appendages")
        lines += [
            f"scale        x{self.scale:.4f}",
            f"shells       {self.shells}",
            f"triangles    {self.triangles}",
            f"shell volume {self.shell_volume_mm3 / 1000.0:.2f} cm3 "
            "(sum over shells, overlaps counted twice)",
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


def _analyse(shells):
    """``(failures, reports)``: one integrity report per shell, computed once."""
    failures, reports = [], []
    for name, mesh in shells:
        if not len(mesh.faces):
            failures.append((name, f"shell {name!r} is empty"))
            continue
        rep = mesh.integrity_report()
        reports.append((name, rep))
        if not rep["watertight"]:
            failures.append((name, f"shell {name!r} is not watertight "
                                   f"(boundary={rep['boundary_edges']}, "
                                   f"nonmanifold={rep['nonmanifold_edges']}, "
                                   f"degenerate={rep['degenerate_faces']})"))
        elif rep["volume"] <= 0:
            failures.append((name, f"shell {name!r} is inside-out (volume "
                                   f"{rep['volume']:.4f})"))
    return failures, reports


def shell_failures(shells) -> list:
    """``[(name, reason)]`` for EVERY shell that is not a printable solid (empty, open,
    non-manifold, degenerate or inside-out). The gate used to stop at the first."""
    return _analyse(shells)[0]


def check_shells(shells) -> list:
    """Verify every shell independently. Returns per-shell reports; raises a
    :class:`PrintError` naming ALL the shells that are not printable solids."""
    failures, reports = _analyse(list(shells))
    if failures:
        msgs = [m for _, m in failures]
        text = msgs[0] if len(msgs) == 1 else (
            f"{len(msgs)} shells are not printable solids: " + "; ".join(msgs))
        raise PrintError(text, failures)
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


def figure_scale(shells, size=None, height=None, reference=None) -> float:
    """Millimetres per scene unit that :func:`assemble_character` would use: the target
    height over the height of the ``reference`` shells (all shells when ``None``)."""
    target = resolve_height(size, height)
    shells = [(n, m) for n, m in shells if len(m.faces)]
    ref = _reference_meshes(shells, reference)
    lo, hi = Mesh.merge(ref).bounds
    extent = float(hi[2] - lo[2])
    if extent <= 1e-9:
        raise PrintError("figure has no height to scale")
    return target / extent


def _reference_meshes(shells, reference):
    if reference is None:
        return [m for _, m in shells]
    names = {reference} if isinstance(reference, str) else set(reference)
    ref = [m for n, m in shells if n in names]
    if not ref:
        raise PrintError(
            f"reference shells {sorted(names)} not found among "
            f"{[n for n, _ in shells][:8]}...")
    return ref


def assemble_character(shells, size=None, height=None, with_base=True,
                       base_spec=None, check=True, sink=0.0, reference=None,
                       fit_base=False):
    """Scale, stand, base and merge a character into one printable mesh.

    ``shells`` is ``[(name, Mesh)]`` in scene units. Returns
    ``(mesh, PrintReport)``.

    ``reference`` names the shells that ARE the figure (body, eyes, teeth:
    :data:`FIGURE_SHELLS`). Without it every shell counts, so a wing tip that stands
    above the head sets the height ("medium" bat-winged figures came out 26.9 mm tall in
    the body) and the bbox that centres the figure and sizes the base includes the span.
    With it the height, the standing plane, the XY centring and the default base come from
    the reference shells only: ``size="medium"`` means a 32 mm BODY, and the appendages
    ride along at the same scale. Shells outside the reference may then stick out past the
    base; the report says by how much (``overhang_mm``) and ``fit_base=True`` widens the
    base to carry them.
    """
    shells = [(n, m) for n, m in shells if len(m.faces)]
    if not shells:
        raise PrintError("character has no geometry to export")
    target = resolve_height(size, height)

    figure = Mesh.merge(_reference_meshes(shells, reference))
    lo, hi = figure.bounds
    extent = float(hi[2] - lo[2])
    if extent <= 1e-9:
        raise PrintError("figure has no height to scale")
    scale = target / extent

    scaled = [(n, m.scaled(scale)) for n, m in shells]
    reports = check_shells(scaled) if check else []

    figure = Mesh.merge(_reference_meshes(scaled, reference))
    lo, hi = figure.bounds
    center = (lo + hi) / 2.0
    # centre on XY, stand on z=0, then optionally sink into the base a touch
    offset = np.array([-center[0], -center[1], -lo[2] + sink])
    placed = [(n, m.translated(offset)) for n, m in scaled]

    notes = feature_notes(target, extent)
    parts = [m for _, m in placed]
    base_diameter = overhang = None
    if with_base:
        spec = base_spec if base_spec is not None else {
            "style": "round", "diameter": max(target * 0.78, 12.0),
            "height": max(target * 0.09, 2.0)}
        reach = 0.0
        if reference is not None:
            reach = max((float(np.hypot(m.vertices[:, 0], m.vertices[:, 1]).max())
                         for _, m in placed), default=0.0)
            if fit_base and spec.get("style", "round") != "none":
                want = 2.0 * (reach + BASE_MARGIN_MM)
                if want > float(spec.get("diameter", 25.0)):
                    notes.append(f"base widened from {float(spec.get('diameter', 25.0)):.1f} "
                                 f"to {want:.1f} mm to carry the appendages")
                    spec = {**spec, "diameter": want}
        base_mesh = build_base(spec)
        if base_mesh is not None:
            parts.append(base_mesh)
            base_diameter = float(spec.get("diameter", 25.0))
            if reference is not None:
                overhang = max(0.0, reach - base_diameter / 2.0)
                if overhang > 0.5:
                    notes.append(
                        f"appendages reach {reach:.1f} mm from the centre, {overhang:.1f} mm "
                        f"past the {base_diameter / 2.0:.1f} mm base rim: fit_base=True or a "
                        "wider base_spec carries them, or the overhang stays unsupported")

    merged = Mesh.merge(parts)
    shell_volume = sum(m.volume() for _, m in placed)
    overall = None
    if reference is not None:
        top = max(float(m.vertices[:, 2].max()) for _, m in placed)
        overall = top
        if top - target > 0.05:
            notes.append(f"{target:.1f} mm is the body; with appendages the figure stands "
                         f"{top:.1f} mm tall")
    report = PrintReport(
        height_mm=target,
        scale=scale,
        triangles=int(len(merged.faces)),
        shells=len(parts),
        shell_volume_mm3=float(abs(shell_volume)),
        watertight=all(r["watertight"] for _, r in reports) if reports else
        bool(merged.integrity_report()["watertight"]),
        notes=notes,
        overall_height_mm=overall,
        base_diameter_mm=base_diameter,
        overhang_mm=float(overhang or 0.0),
        placed_shells=placed,
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


# --------------------------------------------------------------------------
# appendages


@dataclass
class FloorAdjustment:
    """One thing :func:`apply_thickness_floor` changed, so a report can say so."""

    appendage: str          # the appendage's kind (and position in the list)
    param: str
    old: float
    new: float
    reason: str

    def as_text(self) -> str:
        return f"{self.appendage}: {self.param} {self.old:.3f} -> {self.new:.3f} ({self.reason})"


def default_wing_thickness(style: str) -> float:
    """The ``thickness`` a wing of this style is built with when none is given."""
    from . import wings
    fn = wings.feathered_pieces if style == "feathered" else wings.membrane_pieces
    return float(inspect.signature(fn).parameters["thickness"].default)


def apply_thickness_floor(appendages, mm_per_unit: float,
                          floor_mm: float = DEFAULT_MIN_THICKNESS_MM):
    """Thicken wing sheets that would print thinner than ``floor_mm``.

    A wing's ``thickness`` is in the wing's own units and ``Appendage.scale`` multiplies it,
    so the printed sheet is ``thickness * scale * mm_per_unit``. A half-size wing, or a
    tiny figure, prints below what a resin printer resolves while every shell stays
    watertight. The floor is ``thickness >= floor_mm / (mm_per_unit * scale)``; the wing is
    rebuilt with the larger value. Returns ``(appendages, [FloorAdjustment])`` -- the list is
    the input with the changed wings replaced by copies, so the caller's objects are not
    touched. Only wings carry a ``thickness``; tails and horns, and anything that cannot be
    copied with new params, pass through (their thin ends are the report's business).
    """
    if mm_per_unit <= 0 or floor_mm <= 0:
        raise ValueError("mm_per_unit and floor_mm must be positive")
    out, changes = [], []
    for i, app in enumerate(appendages):
        params = getattr(app, "params", None)
        if (getattr(app, "kind", None) != "wing" or not isinstance(params, dict)
                or not dataclasses.is_dataclass(app)):
            out.append(app)
            continue
        style = params.get("style", "membrane")
        have = float(params.get("thickness", default_wing_thickness(style)))
        scale = float(getattr(app, "scale", 1.0))
        want = floor_mm / (mm_per_unit * scale)
        if want <= have * (1.0 + 1e-9):
            out.append(app)
            continue
        out.append(dataclasses.replace(app, params={**params, "thickness": want}))
        changes.append(FloorAdjustment(
            f"wing[{i}]", "thickness", have, want,
            f"{floor_mm:g} mm floor at {mm_per_unit:.3f} mm/unit, scale {scale:g}"))
    return out, changes


def ready_made_shells(app):
    """``[(name, Mesh)]`` if ``app`` is a ready-made appendage -- one ``(name, Mesh)`` pair or a
    list of them (an appendage of several shells, named ``group/piece``) -- else ``None``."""
    if isinstance(app, tuple) and len(app) == 2 and isinstance(app[0], str):
        return [app]
    if isinstance(app, list) and app and all(
            isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str) for x in app):
        return list(app)
    return None


def _is_rigged(app) -> bool:
    return callable(getattr(app, "build_rigs", None)) and getattr(app, "kind", "wing") == "wing"


def appendage_print_shells(appendages, base=None, verts=None, split_pieces: bool = False, *,
                           min_thickness_mm=None, mm_per_unit=None, adjustments=None):
    """Named closed shells for appendages, in Z-up scene units: the frame of
    :func:`character_print_shells`, so the two lists concatenate.

    ``appendages`` are objects with ``build(base, verts) -> [(name, Mesh)]`` (and, for
    rigged wings, ``build_rigs``) such as :class:`~.appendages.Appendage`, or ready-made
    ``(name, Mesh)`` pairs (a list of pairs is one appendage of several shells). Appendage
    geometry is in the MakeHuman frame (Y up); this is the one place it is turned to Z up
    (:data:`portrait.MH_TO_Z_UP`, a proper rotation). Ready-made meshes are taken to be in
    that same MakeHuman frame.

    ``split_pieces`` returns a wing as one shell per rig piece (``wing_l/primary05``): the
    unit a printability report can name. The mirrored wing's winding comes from
    :meth:`PlacedWing.piece_meshes`, so it is right here too. Other kinds are one shell each.

    ``min_thickness_mm`` (with ``mm_per_unit``, see :func:`figure_scale`) applies
    :func:`apply_thickness_floor` first; the changes are appended to ``adjustments`` if
    that list is given. Off by default. Shell names must be unique.
    """
    from .portrait import MH_TO_Z_UP

    appendages = list(appendages)
    if min_thickness_mm is not None:
        if mm_per_unit is None:
            raise ValueError("min_thickness_mm needs mm_per_unit (figure_scale gives it)")
        appendages, changes = apply_thickness_floor(appendages, mm_per_unit, min_thickness_mm)
        if adjustments is not None:
            adjustments.extend(changes)
    out = []
    for app in appendages:
        ready = ready_made_shells(app)
        if ready is not None:
            out += [(n, m.transform(MH_TO_Z_UP)) for n, m in ready]
        elif split_pieces and _is_rigged(app):
            controls = app.pose_controls() if hasattr(app, "pose_controls") else {}
            preset = app.pose_preset() if hasattr(app, "pose_preset") else None
            for name, pw in app.build_rigs(base, verts):
                out += [(f"{name}/{piece}", m.transform(MH_TO_Z_UP))
                        for piece, m in pw.piece_meshes(preset, **controls)]
        else:
            out += [(n, m.transform(MH_TO_Z_UP)) for n, m in app.build(base, verts)]
    names = [n for n, _ in out]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"appendages produce shells with the same name {dup[:3]}: "
                         "give each appendage its own name (T1 ids)")
    return out
