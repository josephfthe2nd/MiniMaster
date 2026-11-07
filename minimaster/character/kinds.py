"""The kind registry: what an appendage *is*, behind the pure-data document.

A saved character holds :class:`~minimaster.character.appendage_spec.AppendageSpec`
records (a kind name, parameters, controls, a placement). This module is the heavy side
that gives those names a meaning. A :class:`Kind` declares

* its parameters as :class:`ParamSpec` rows (float | int | choice | bool, with range,
  default, step and a ``when=(other_key, allowed)`` condition for style-dependent ones),
  which is what a property grid is generated from;
* its pose controls and their *dynamic* ranges (``control_range(name, state)``);
* its placement presets (``shoulder_blades``, ``tail_base``, ``temples``, ...);
* ``instantiate(spec, ctx)``, which builds the appendage on a morphed body and returns
  named :class:`Placed` objects (not a body-free ``build(params, scale)``: a wing needs
  the body to find its root, a harvested limb needs it to be cut from).

``Appendage`` (appendages.py) stays the placement engine: every kind builds one of those
and asks it for the geometry. The kind ids are the ones the rest of the code already
uses (``wing``, ``tail``, ``horn`` and ``harvest``, the extra limb), so
``creature_anatomy`` keeps working.

Placement follows the body (design D4). A preset is resolved ONCE to cage quad ids on
the neutral body and the quad id is what the document stores: the quad exists on every
morph, so on a taller, smaller, heavier or younger body the appendage stays on the same
part of the body. The right-hand side uses the exact mirror image of the stored
left-hand quad.

Nothing in the document is ever dropped: an unknown kind, parameter, control or style is
skipped with a :class:`AppendageWarning` when something is built from it and stays in the
file.
"""

from __future__ import annotations

import copy
import difflib
import json
import math
import warnings
import weakref
from dataclasses import dataclass, field

import numpy as np

from ..core.mesh import Mesh
from . import appendages as A
from . import hornrig as HR
from . import junction as J
from . import tailrig as T
from .appendage_spec import SIDES, AppendageSpec, next_id, valid_id
from . import wingroot as WR  # noqa: E402  (after the others: the tail track also adds an import after ``junction``)
from . import wings as W  # noqa: E402


class AppendageWarning(UserWarning):
    """Something in an appendage record was skipped when building it."""


class SpecError(ValueError):
    """A record cannot be built (bad placement, unknown style, ...). The record is kept
    in the document; the appendage is skipped with a warning."""


# --------------------------------------------------------------------------
# parameters


@dataclass(frozen=True)
class ParamSpec:
    """One user-facing parameter of a kind.

    ``type`` is ``float``, ``int``, ``bool`` or ``choice``. ``when=(key, allowed)`` makes
    the parameter apply only while the parameter ``key`` has one of the ``allowed`` values
    (the wing's ``gull`` is a membrane parameter, ``primaries`` a feathered one), and the
    same ``key`` may be declared twice with different ``when`` where the default differs
    by style. ``essential`` parameters (a wing's style) cannot fall back to a default:
    an invalid value skips the appendage.
    """

    key: str
    label: str
    type: str = "float"
    default: object = 0.0
    lo: float | None = None
    hi: float | None = None
    step: float | None = None
    choices: tuple = ()
    when: tuple | None = None
    essential: bool = False
    help: str = ""

    def applies(self, values: dict) -> bool:
        if self.when is None:
            return True
        key, allowed = self.when
        return values.get(key) in tuple(allowed)

    def coerce(self, value):
        """``value`` as this parameter's type, clamped to ``[lo, hi]``. ``ValueError`` for
        anything that is not a valid value (wrong type, NaN, not among the choices)."""
        if self.type == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and value in (0, 1):
                return bool(value)
            if isinstance(value, str) and value.strip().lower() in ("true", "false", "1", "0",
                                                                       "yes", "no", "on", "off"):
                return value.strip().lower() in ("true", "1", "yes", "on")
            raise ValueError(f"{self.key}: {value!r} is not a boolean")
        if self.type == "choice":
            for c in self.choices:
                if value == c or (isinstance(value, str) and value.strip().lower() == str(c)):
                    return c
            raise ValueError(f"{self.key}: {value!r} is not one of {list(self.choices)}")
        if isinstance(value, bool) or value is None:
            raise ValueError(f"{self.key}: {value!r} is not a number")
        try:
            x = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{self.key}: {value!r} is not a number") from None
        if not math.isfinite(x):
            raise ValueError(f"{self.key}: {value!r} is not finite")
        if self.type == "int":
            x = int(round(x))
        if self.lo is not None:
            x = max(type(x)(self.lo), x)
        if self.hi is not None:
            x = min(type(x)(self.hi), x)
        return x

    def to_dict(self) -> dict:
        d = {"key": self.key, "label": self.label, "type": self.type, "default": self.default}
        for k in ("lo", "hi", "step"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        if self.choices:
            d["choices"] = list(self.choices)
        if self.when:
            d["when"] = [self.when[0], list(self.when[1])]
        return d


@dataclass(frozen=True)
class ControlSpec:
    """A pose control (not a build parameter): a number that moves an existing appendage."""

    key: str
    label: str
    default: float = 0.0


# --------------------------------------------------------------------------
# presets: where on the body things go
#
# A preset is resolved ONCE, on the neutral body (the default macro values: the body these
# constants were tuned on), to a cage quad; the quad id is what a document stores. They are
# written down here rather than searched for at build time so that placing a preset needs
# no morph library; tests/test_kinds.py recomputes every one from its point and fails if
# the base mesh ever changes under them.

PRESET_POINTS = {
    "shoulder_blades": A.WING_ANCHOR,        # the rearmost point of the scapula
    "waist": A.WING_ROOT,                    # where a bat wing's attachment line ends
    "tail_base": (0.02, 1.02, -0.69),        # the sacrococcygeal midline, 1 R above the gluteal apex
    "temples": (0.85, 8.55, 0.05),           # up and out on the skull
    "shoulder_socket": (2.0, 4.2, 0.15),     # the side of the chest, under the shoulder
    "hip_socket": (1.5, 0.4, 0.15),          # the outer hip
    "lower_arms": (1.239, 2.97, 0.884),      # the front of the flank at the waist, below the arm
    "upper_arms": (1.441, 4.364, 0.906),     # the upper chest, in front of the shoulder
    "extra_legs": (1.268, 3.063, 0.216),     # the flank at the waist (legs are too wide for the hip)
}
PRESET_FACES = {
    "shoulder_blades": 10296,
    "waist": 10593,
    "tail_base": 10562,           # the +x quad of the two around the midline edge 4215-4205
    "temples": 12116,
    "shoulder_socket": 7954,
    "hip_socket": 10647,
    "lower_arms": 10411,
    "upper_arms": 7974,
    "extra_legs": 10349,
}
# WING_ROOT minus the centroid of the "waist" quad on the neutral body: the end of a
# membrane wing's attachment line is "the waist quad plus a little way into the back"
WING_ROOT_OFFSET = (0.04479999999999995, -0.076125, 0.12734999999999996)

# How far a tail or horn is sunk into the body, along the surface normal. At offset 0 a
# tail is tangent to the skin (0.5% of its volume inside, a hairline weld) and a horn
# touches it at a point; the sink is measured: tail ~25% / horn ~65% of the volume embedded
# (tests/test_kinds.py), which is a weld a 32 mm print survives.
TAIL_SINK = 0.0          # a tail's sink is its bone root (tailrig.STEM_R x R below the skin along its axis), not an offset
LEGACY_TAIL_SINK = -0.6  # what the first tail preset stored in ``placement.offset`` (read as the new default)
LEGACY_TAIL_BASE_FACE = 3921    # the first tail_base quad: 0.36 R off the midline, on the side of a buttock
HORN_SINK = -0.55
HARVEST_SINK = -0.5      # along the limb axis: a harvested limb has no surface normal
TAIL_TIP = 0.09          # radius at the end of a tail (the first version's was 0.02: a point)
HORN_TIP = 0.08          # and of a horn (0.012)
HARVEST_TRIM = 0.6       # shaves the root of the copy (units of the neutral limb)
HARVEST_NECK = 0.5       # the cut end keeps this much of the limb's width (limbrig.taper_root)
# the skin around a root (junction.py): how wide the fillet is, in root radii, and how much the
# root flares, per kind. A generic default: the anatomy of each kind is a later track's.
# Facets around a tube: 16 (it was 10 / 8-12). The skin meets a tube along a seam, and a seam
# against a 10-gon is a zigzag between its facets that no displacement of the body can hide
# (24 sides: a clean curve), so the junction wants round roots.
TAIL_RING_VERTS = 16
HORN_RING_VERTS = 16
TAIL_FILLET, TAIL_FLARE, TAIL_FILLET_HEIGHT = 1.5, 0.35, 1.0
HORN_FILLET, HORN_FLARE, HORN_FILLET_HEIGHT = 1.0, 0.0, 0.6
WING_FILLET, WING_FILLET_HEIGHT = 2.0, 0.3        # in elbow radii of the arm (the mounds of the shoulder anatomy carry the bulk, wingroot.py)
# the scapular mass, as FRACTIONS of the trunk's radius (junction.Mound(scale="trunk"): the core
# resolves them from the body's own skeleton). In arm lengths (0.55 / 0.07 of a 3 unit arm) it was a
# pillow 1.7 units wide on any body: the folded wing's 'lobes' and a child's boxy shoulders
WING_MOUND_RADIUS, WING_MOUND_HEIGHT = 0.5, 0.055
HARVEST_FILLET, HARVEST_FILLET_HEIGHT = 1.0, 0.7
# the radius of the neutral body's own limb along its bones, as a fraction of the upper bone:
# (positions along the upper bone, radii there, positions along the forearm from the elbow,
# radii there), measured on the base mesh (the median distance of the surface to the bone).
LIMB_RADII = {
    "arm": ((0.0, 0.25, 0.5, 0.75, 1.0), (0.242, 0.217, 0.190, 0.180, 0.182),
            (0.25, 0.5), (0.162, 0.136)),
    "leg": ((0.0, 0.25, 0.5, 0.75, 1.0), (0.284, 0.270, 0.213, 0.144, 0.130),
            (0.25, 0.5), (0.133, 0.115)),
}


# --------------------------------------------------------------------------
# the context a kind is built in


class BuildContext:
    """Everything a kind needs to place something on one morphed body."""

    def __init__(self, base, verts=None, lib=None, report: list | None = None):
        self.base = base
        self.verts = base.verts if verts is None else verts
        self.lib = lib
        self.report = report if report is not None else []
        self._cage = None

    @property
    def cage(self):
        """``(vertices, quads)`` of the morphed body cage."""
        if self._cage is None:
            self._cage = self.base.body_cage(self.verts)
        return self._cage

    @property
    def mirror_map(self) -> np.ndarray:
        return A.mirror_face_map(self.base)

    def neutral_cage(self):
        """The cage of the neutral body (default macro values) that presets and v1
        world-point anchors are resolved against."""
        return neutral_cage(self.base, self.lib)

    def warn(self, message: str) -> None:
        self.report.append(message)
        warnings.warn(message, AppendageWarning, stacklevel=3)


_NEUTRAL: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def neutral_cage(base, lib=None):
    """``(vertices, quads)`` of the neutral body: the unmorphed base plus the default
    macro targets. Cached per base mesh."""
    cached = _NEUTRAL.get(base)
    if cached is None:
        from . import mhbase as mh
        lib = lib if lib is not None else mh.TargetLibrary()
        verts = lib.apply(base.verts, mh.macro_weights())
        cached = base.body_cage(verts)
        _NEUTRAL[base] = cached
    return cached


def nearest_face(cage, point) -> int:
    """The cage quad whose centroid is nearest ``point``."""
    v, q = cage
    return int(np.argmin(np.linalg.norm(v[q].mean(axis=1) - np.asarray(point, dtype=float),
                                        axis=1)))


# --------------------------------------------------------------------------
# what instantiate returns


@dataclass
class Placed:
    """One built, placed appendage (one side of a spec).

    ``mesh()`` is in the MakeHuman frame (Y up, the frame of the morphed body); the print
    and preview frame is Z up (``portrait.MH_TO_Z_UP``). The name derives from the spec id
    (``wing-1.l``) and so does the joint prefix a host armature attaches the wing's joints
    under, so two wing specs never collide."""

    name: str
    spec_id: str
    kind: "Kind"
    side: str                      # "" (midline), "l" or "r"
    anchor: np.ndarray             # where it joins the body: the bound quad's surface point
    matrix: np.ndarray | None      # the 4x4 placement (wings, tails, horns), else None
    params: dict                   # the cleaned, applicable parameters it was built from
    controls: dict                 # the effective pose controls of this side
    rig: object = None             # a PlacedWing for wings
    static: Mesh | None = None     # the mesh under the spec's own controls (loft kinds, limbs)
    limb: object = None            # a limbrig.LimbRig for a harvested arm or leg
    repose: object = None          # controls -> Mesh: a kind that re-evaluates its loft
    root_depth: float = 0.0        # how far the root is sunk below the skin (units): the junction
                                   # takes the centreline from there to a few fillet widths out
    model: object = None           # a kind's own anatomy (a tailrig.PlacedTail for a tail)
    _junctions: dict = field(default_factory=dict, repr=False, compare=False)   # controls -> spec

    def mesh(self, controls: dict | None = None) -> Mesh:
        """The mesh under ``controls`` (default: the spec's own for this side)."""
        if controls is None and self.static is not None:
            return self.static
        if self.limb is not None:
            return self.limb.pose(self.controls if controls is None else controls)
        if self.repose is not None:
            return self.repose(self.controls if controls is None else controls)
        if self.rig is None:
            return self.static
        c = dict(self.controls if controls is None else controls)
        preset = c.pop("preset", None)
        return self.rig.pose(preset, **c) if (c or preset) else self.rig.mesh()

    def layers(self, controls: dict | None = None) -> dict:
        """The anatomical layers of this appendage under ``controls`` (default: the spec's own)
        as named meshes in the MakeHuman frame: ``"bone"`` (the internal chain, hidden),
        ``"muscle"`` (the masses that shape the surface) and ``"skin"`` (the shell that is
        printed and rendered: ``mesh()``). A kind without an anatomy yet has the skin only."""
        return self.kind.layers(self, self.controls if controls is None else controls)

    def junction(self, controls: dict | None = None):
        """The :class:`~.junction.JunctionSpec` of this appendage's root (how the body's skin
        should mould around it), under ``controls`` (default: the spec's own), or ``None`` for
        an appendage without one (its kind has none, or the spec's ``fillet`` is 0). In the
        MakeHuman frame; the spec comes from the placed geometry, never from the body, so it
        is the same on every body the appendage is placed on."""
        ctl = self.controls if controls is None else controls
        key = json.dumps(ctl, sort_keys=True, default=str)
        if key not in self._junctions:
            if len(self._junctions) >= 24:               # a pose drag asks for a new one per frame
                self._junctions.pop(next(iter(self._junctions)))
            self._junctions[key] = self.kind.junction(self, ctl)
        return self._junctions[key]

    @property
    def joint_prefix(self) -> str:
        return self.name

    @property
    def control_names(self) -> tuple:
        """The pose controls this appendage answers to (a leg has ``knee`` where an arm has
        ``elbow``)."""
        if self.limb is not None:
            return tuple(self.limb.control_names)
        return tuple(c.key for c in self.kind.controls)

    def control_range(self, name: str, controls: dict | None = None):
        """``(lo, hi)`` this placed appendage allows ``name`` right now (a placed wing
        allows less than a free one, and less as it folds; a limb's swing depends on how
        far it is raised; a tail's curl on the curl it was built with), or ``None`` if it
        has no such control."""
        if name not in self.control_names:
            return None
        state = self.controls if controls is None else controls
        if self.limb is not None:
            return self.limb.control_range(name, state)
        if self.repose is not None:
            return self.kind.control_range(name, state, self.params)
        if self.rig is None:
            return None
        return self.rig.rig.control_range(name, float(state.get("fold", 0.0)))

    @property
    def material(self):
        return self.kind.material(self.params)


# --------------------------------------------------------------------------
# the skin around a root: the parameter every kind with a junction carries


def fillet_param(default: float = 1.0) -> "ParamSpec":
    return ParamSpec("fillet", "Skin fillet", "float", default, 0.0, 4.0, 0.05,
                     help="how far the skin flows into the root, in root radii "
                          "(junction.py); 0 = the skin is left as it is")


def flare_param(default: float = 0.0) -> "ParamSpec":
    return ParamSpec("flare", "Root flare", "float", default, 0.0, 2.0, 0.05,
                     help="broadens the first quarter of the length (1 = twice as wide at "
                          "the base) so the root is broad where it enters the skin")


# --------------------------------------------------------------------------
# kinds


def _split_controls(controls: dict) -> tuple[dict, dict]:
    """``(shared, per_side)`` from a record's controls: scalars and the ``"*"`` object are
    shared by both sides, ``"l"`` / ``"r"`` objects override one."""
    shared = {k: v for k, v in controls.items() if k not in ("*", "l", "r")
              and not isinstance(v, dict)}
    if isinstance(controls.get("*"), dict):
        shared.update(controls["*"])
    sides = {s: controls[s] for s in ("l", "r") if isinstance(controls.get(s), dict)}
    return shared, sides


def side_controls(controls: dict, letter: str) -> dict:
    """The controls that apply to one side (``"l"``, ``"r"`` or ``""`` for the midline)."""
    shared, sides = _split_controls(controls)
    return {**shared, **sides.get(letter, {})}


class Kind:
    """Base class: one registered kind of appendage."""

    id = ""
    label = ""
    params: tuple = ()
    controls: tuple = ()
    presets: dict = {}                 # name -> placement defaults
    default_side = "center"
    pose_presets = False               # does ``--pose ID:preset=NAME`` mean anything?
    placement_keys = ("anchor", "scale", "roll", "offset", "up_hint")   # what it honours
    anatomy_key = ""                   # creature_anatomy.CONSEQUENCES key (default: id)
    paired = False                     # is a symmetric pair its typical installation?
    fillet_default = 1.0               # the ``fillet`` parameter's default (width, in root radii)
    fillet_height = 0.6                # how high the skin climbs the root, over the fillet width
    junction_owner = None              # the part of the body the root belongs to (junction.JunctionSpec.owner)

    # -- parameters ---------------------------------------------------------------------

    def param(self, key: str, values: dict | None = None):
        """The ParamSpec for ``key`` that applies given ``values`` (or the first)."""
        for p in self.params:
            if p.key == key and (values is None or p.applies(values)):
                return p
        return None

    def defaults(self, **given) -> dict:
        """The default value of every applicable parameter, with ``given`` on top."""
        values: dict = {}
        for p in self.params:
            if p.key in given:
                values[p.key] = given[p.key]
            elif p.applies(values) and p.key not in values:
                values[p.key] = p.default
        return values

    def clean(self, params: dict, owner: str = ""):
        """``(values, notes)``: every applicable parameter, coerced and clamped, with
        defaults filled in. Unknown keys and invalid values become ``notes`` (the
        parameter is ignored / defaulted and stays in the record); an invalid ESSENTIAL
        parameter raises :class:`SpecError`."""
        notes: list[str] = []
        values: dict = {}
        known = {p.key for p in self.params}
        for p in self.params:
            if p.key in values or not p.applies(values):
                continue
            if p.key in params:
                try:
                    values[p.key] = p.coerce(params[p.key])
                    if p.type in ("float", "int") and values[p.key] != float(params[p.key]):
                        notes.append(f"{owner}{p.key} = {params[p.key]!r} is outside "
                                     f"[{p.lo}, {p.hi}]: using {values[p.key]!r}")
                    continue
                except ValueError as exc:
                    if p.essential:
                        raise SpecError(str(exc)) from None
                    notes.append(f"{owner}{exc}: using the default {p.default!r}")
            values[p.key] = p.default
        for key in params:
            if key not in known:
                near = difflib.get_close_matches(key, sorted(known), n=1)
                notes.append(f"{owner}unknown parameter {key!r} is kept in the file but "
                             f"ignored" + (f" (did you mean {near[0]!r}?)" if near else ""))
        return values, notes

    def draft_params(self, values: dict) -> dict:
        """Cheaper values for a quick interactive preview (orbit, drag): the same
        appendage without the fine detail. Applied by ``instantiate(..., draft=True)``."""
        return {}

    # -- controls -----------------------------------------------------------------------

    def control_range(self, name: str, state: dict | None = None, params: dict | None = None):
        """``(lo, hi)`` the control ``name`` takes, given the other controls ``state`` and
        the record's cleaned ``params`` (a free wing's range; a PLACED one's is narrower,
        see ``Placed.control_range``), or ``None`` if this kind has no such control."""
        return None

    def clean_controls(self, controls: dict, owner: str = ""):
        """``(controls, notes)``: the controls this kind understands, with unknown or
        invalid ones left out (and named in ``notes``); the record keeps them."""
        notes = []
        for key in controls:
            notes.append(f"{owner}unknown control {key!r} is kept in the file but ignored")
        return {}, notes

    # -- the skin around the root (junction.py) ------------------------------------------------

    def junction(self, placed, controls: dict | None = None):
        """The junction spec of one placed appendage of this kind (``None``: none). A kind
        with a root the skin should mould around overrides this."""
        return None

    def layers(self, placed, controls: dict | None = None) -> dict:
        """``{layer name: Mesh}`` of one placed appendage (see :meth:`Placed.layers`); the base
        class only has the skin."""
        return {"skin": placed.mesh(controls)}

    def layer_colors(self, placed, name: str, controls: dict | None = None):
        """Per-vertex colour weights of layer ``name`` for an ecorche view (``ecorche.py``): ``None``
        (one colour) or an array in 0..1 per vertex of that layer's mesh, 1 = the layer's own colour."""
        return None

    # -- anatomy ------------------------------------------------------------------------

    def load(self, params: dict) -> float:
        """How big this appendage is compared with a typical one of its kind (1.0): what
        ``creature_anatomy`` scales the body adaptation by."""
        return 1.0

    # -- look ---------------------------------------------------------------------------

    def material(self, params: dict):
        from . import shading as sh
        return sh.GENERIC

    # -- placement ----------------------------------------------------------------------

    def default_preset(self, values: dict) -> str:
        return next(iter(self.presets))

    def id_stem(self, params: dict) -> str:
        """What a new record's id is named after (``"<stem>-<n>"``): the kind, or for
        an extra limb the limb (``arm-1``, ``leg-1``)."""
        return self.id

    def resolve_placement(self, spec: AppendageSpec, values: dict, ctx: BuildContext) -> dict:
        """The record's placement with the preset's values filled in, and the anchor
        resolved to a cage quad: ``{face, weights, scale, roll, offset, up_hint, aim}``."""
        pl = spec.placement
        name = pl.get("preset") if isinstance(pl.get("preset"), str) else None
        preset = self.presets.get(name) or self.presets[self.default_preset(values)]
        out = {
            "scale": _num(pl.get("scale"), preset.get("scale", 1.0)),
            "roll": _num(pl.get("roll"), preset.get("roll", 0.0)),
            "offset": _num(pl.get("offset"), preset.get("offset", 0.0)),
            "up_hint": _vec(pl.get("up_hint"), preset.get("up_hint", (0.0, 1.0, 0.0))),
            "aim": pl.get("aim"),
            "weights": None,
            "midline": False,
        }
        anchor = pl.get("anchor")
        n_quads = len(ctx.cage[1])
        if isinstance(anchor, dict) and "face" in anchor:
            face = anchor["face"]
            if not isinstance(face, int) or isinstance(face, bool) or not 0 <= face < n_quads:
                raise SpecError(f"anchor face {face!r} is not a quad of this body "
                                f"(0..{n_quads - 1})")
            corners = anchor.get("corners")
            if corners is not None and [int(c) for c in ctx.cage[1][face]] != list(corners):
                raise SpecError(f"anchor face {face} is not the same quad here as when it "
                                "was saved (a different base mesh?)")
            out["face"] = face
            w = anchor.get("weights")
            if w is not None:
                if len(w) != len(ctx.cage[1][face]) or not all(
                        isinstance(x, (int, float)) and math.isfinite(x) for x in w):
                    raise SpecError("anchor weights must be one finite number per corner")
                out["weights"] = tuple(float(x) for x in w)
        elif isinstance(anchor, dict) and "point" in anchor:
            # a v1-style world point: converted to a body-relative anchor ONCE, on the
            # neutral body (its mirror image if it is on the right), instead of being
            # re-snapped to whichever quad is nearest on every body
            p = _vec(anchor["point"], None)
            if p is None:
                raise SpecError("anchor point must be three numbers")
            p = (abs(p[0]), p[1], p[2])
            out["face"] = nearest_face(ctx.neutral_cage(), p)
        else:
            out["face"] = PRESET_FACES[preset["face"]]
        return out

    # -- building -----------------------------------------------------------------------

    def sides(self, spec: AppendageSpec) -> str:
        if spec.side not in SIDES:
            raise SpecError(f"unknown side {spec.side!r}; use {', '.join(SIDES)}")
        return spec.side

    def instantiate(self, spec: AppendageSpec, ctx: BuildContext,
                    draft: bool = False) -> list:
        side = self.sides(spec)
        values, notes = self.clean(spec.params, f"appendage {spec.id!r}: ")
        for note in notes:
            ctx.warn(note)
        if spec.controls and not self.controls:
            # a kind without pose controls: say so once (a kind WITH controls reports per side)
            shared, per_side = _split_controls(spec.controls)
            merged = {**shared, **{k: v for d in per_side.values() for k, v in d.items()}}
            for note in self.clean_controls(merged, f"appendage {spec.id!r}: ")[1]:
                ctx.warn(note)
        build_values = dict(values)
        if draft:
            build_values.update(self.draft_params(values))
        pl = self.resolve_placement(spec, values, ctx)
        return self.build(spec, side, values, build_values, pl, ctx)

    def build(self, spec, side, values, build_values, pl, ctx) -> list:
        raise NotImplementedError

    # -- helpers shared by the kinds -------------------------------------------------------

    def _engine(self, side: str, pl: dict, **over) -> A.Appendage:
        kw = dict(scale=pl["scale"], up_hint=tuple(pl["up_hint"]), roll=pl["roll"],
                  offset=pl["offset"], anchor_face=pl["face"], anchor_weights=pl["weights"],
                  side=side, anchor_midline=bool(pl.get("midline", False)))
        kw.update(over)
        return A.Appendage(self.id, **kw)

    def _names(self, spec: AppendageSpec, idxs) -> list:
        """``[(idx, shell name, side letter)]`` for the sides built."""
        if spec.side == "center":
            return [(i, spec.id, "") for i in idxs]
        return [(i, f"{spec.id}.{'l' if i == 0 else 'r'}", "l" if i == 0 else "r")
                for i in idxs]

    def _surface(self, ctx: BuildContext, app: A.Appendage, idx: int) -> np.ndarray:
        body_v, body_q = ctx.cage
        hint = np.asarray(app.up_hint, dtype=np.float64)
        anchor = app._face_anchor(body_v, body_q, idx, hint, ctx.base)
        return anchor.resolve(body_v)[0]

    def describe(self) -> dict:
        return {"id": self.id, "label": self.label, "default_side": self.default_side,
                "placement_keys": list(self.placement_keys), "paired": self.paired,
                "presets": {k: v.get("label", k) for k, v in self.presets.items()},
                "params": [p.to_dict() for p in self.params],
                "controls": [c.key for c in self.controls]}


def _num(value, default):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return default


def _vec(value, default):
    try:
        v = tuple(float(x) for x in value)
        if len(v) == 3 and all(math.isfinite(x) for x in v):
            return v
    except (TypeError, ValueError):
        pass
    return default


# -- tails and horns ---------------------------------------------------------------------


def _tip(values: dict):
    """The builders' ``tip_radius``: the radius, or ``None`` (a point) for 0."""
    r = float(values["tip_radius"])
    return r if r > 0.0 else None


class _NumericControls:
    """``clean_controls`` for a kind whose pose controls are plain numbers: the ones it has,
    finite, are used; the rest are named in a note and stay in the record."""

    def clean_controls(self, controls, owner=""):
        if not self.controls:
            return {}, ([f"{owner}{self.id} has no pose controls; {sorted(controls)} kept but "
                         "ignored"] if controls else [])
        out, notes = {}, []
        names = [c.key for c in self.controls]
        for key, value in controls.items():
            if key not in names:
                near = difflib.get_close_matches(str(key), names, n=1)
                notes.append(f"{owner}{self.id} has no control {key!r}; kept in the file but "
                             f"ignored" + (f" (did you mean {near[0]!r}?)" if near
                                           else f" (it has: {', '.join(names)})"))
            elif (isinstance(value, (int, float)) and not isinstance(value, bool)
                  and math.isfinite(value)):
                out[key] = float(value)
            else:
                notes.append(f"{owner}control {key} = {value!r} is not a finite number; ignored")
        return out, notes


class _LoftKind(_NumericControls, Kind):
    """A tapered tube swept out of the body. Its pose controls (a tail's) re-evaluate the
    loft with other bend values and place it with the same frame: there is no joint rig,
    nothing consumes one, and a loft is 1-7 ms."""

    def builder_args(self, build_values: dict, controls: dict) -> dict:
        """The keyword arguments of the ``appendages`` builder for these parameters and
        pose controls."""
        raise NotImplementedError

    def build(self, spec, side, values, build_values, pl, ctx):
        app = self._engine(side, pl, params={})
        body_v, body_q = ctx.cage
        builder = A.BUILDERS[self.id]
        names = {i: (n, s) for i, n, s in self._names(spec, app._indices())}
        out = []
        for idx in app._indices():
            name, letter = names[idx]
            frame = app._anchor_frame(body_v, body_q, idx, app._point(idx), ctx.base)
            ctrl = {}
            if self.controls:               # a kind without controls was reported once already
                ctrl, notes = self.clean_controls(side_controls(spec.controls, letter),
                                                  f"appendage {spec.id!r}: ")
                for note in notes:
                    ctx.warn(note)

            def repose(controls, frame=frame):
                return builder(**self.builder_args(build_values, controls)).transform(frame)

            out.append(Placed(name=name, spec_id=spec.id, kind=self, side=letter,
                              anchor=self._surface(ctx, app, idx), matrix=frame,
                              params=dict(values), controls=ctrl, static=repose(ctrl),
                              repose=repose, root_depth=abs(float(min(pl["offset"], 0.0)))))
        return out

    def junction(self, placed, controls=None):
        """The skin flows into the first part of the tube: its rings from the sunk base to a
        few fillet widths beyond the skin, the radius of the inscribed circle (a facet's
        middle), so the seam with the skin is clean."""
        fillet = float(placed.params.get("fillet", self.fillet_default))
        if fillet <= 0.0:
            return None
        mesh = placed.mesh(controls)
        n = int(placed.params["ring_verts"])
        rings = int(placed.params["segments"]) + 1
        centres, radii = J.ring_tube(mesh.vertices, n, rings)
        base_r = float(placed.params["base_radius"])
        scale = float(np.cbrt(abs(np.linalg.det(placed.matrix[:3, :3]))))
        reach = 1.5 * placed.root_depth + 1.5 * fillet * base_r * scale
        k = J.rings_for_length(centres, reach)
        return J.JunctionSpec(placed.name, centres[:k], radii[:k], width=fillet,
                              base_radius=base_r * scale, height=self.fillet_height,
                              owner=self.junction_owner)


class TailKind(_LoftKind):
    """A tail: the continuation of the spine, as an anatomy (``tailrig.py``): a caudal vertebra
    chain, the muscle sleeve around it and the skin over both, rooted on the MIDLINE of the
    sacrum (the anchor is the cage quad over the sacrococcygeal level paired with its mirror
    image, so the tail leaves exactly on the sagittal plane and its axis is the sacral surface
    normal clamped to 5-18 degrees above horizontal), sunk into the body by its hidden bone root
    (the stem), and sized by the body (``length`` and ``base_radius`` are the values on the
    neutral body; a taller body has a proportionally longer, thicker tail).

    Pose controls re-evaluate the chain: ``curl`` and ``droop`` ADD to the parameters of the same
    name (same units, so 0 is the tail as built; the sum stays inside the parameter's range) and
    are both SAGITTAL (``droop`` bends the tail down by gravity after a stiff root, ``curl`` curls
    the distal part up); ``lift`` raises the whole tail about the sacrococcygeal joint and ``sway``
    flexes it sideways (toward +X), in degrees. ``tip_radius`` is the radius at the end: a rounded
    end 0.3 mm across by default."""

    id = "tail"
    label = "Tail"
    default_side = "center"
    paired = False
    fillet_default = TAIL_FILLET
    fillet_height = TAIL_FILLET_HEIGHT
    junction_owner = "trunk"
    pitch_clamp = (5.0, 18.0)          # the axis' elevation above horizontal, midline tails (degrees)
    controls = (ControlSpec("curl", "Curl (added to the shape)", 0.0),
                ControlSpec("droop", "Droop (added to the shape)", 0.0),
                ControlSpec("lift", "Lift (degrees, up)", 0.0),
                ControlSpec("sway", "Sway (degrees, toward +X)", 0.0))
    params = (
        ParamSpec("length", "Length", "float", 6.0, 1.0, 20.0, 0.1,
                  help="on the neutral body; scaled with the body's height"),
        ParamSpec("base_radius", "Base radius", "float", 0.45, 0.1, 2.0, 0.01,
                  help="on the neutral body; scaled with the body's height"),
        ParamSpec("segments", "Segments", "int", 12, 4, 40, 1),
        ParamSpec("ring_verts", "Ring vertices", "int", TAIL_RING_VERTS, 6, 24, 1),
        ParamSpec("curl", "Curl", "float", 0.55, -2.0, 2.0, 0.01,
                  help="curls the distal 45% of the tail up (sagittal)"),
        ParamSpec("droop", "Droop", "float", 0.9, -2.0, 2.0, 0.01,
                  help="how far gravity bends the tail down after its stiff root (sagittal)"),
        ParamSpec("tip_radius", "Tip radius", "float", TAIL_TIP, 0.0, 0.4, 0.005,
                  help="radius at the end of the taper, which is rounded; 0 = the finest tip "
                       "the skin allows (0.03 units)"),
        ParamSpec("flare", "Root flare", "float", TAIL_FLARE, 0.0, 2.0, 0.05,
                  help="the trumpet where the caudal muscles spread onto the sacrum: the radius at the "
                       "skin is 1 + flare times the base radius, easing out over about two radii "
                       "(0 = a straight tube)"),
        fillet_param(TAIL_FILLET),
        ParamSpec("vertebrae", "Caudal vertebrae", "int", 0, 0, 60, 1,
                  help="0 = by the length (round(48 L / H): 17 on the default tail)"),
        ParamSpec("crest", "Dorsal ridge", "float", 0.0, 0.0, 1.5, 0.05,
                  help="a dorsal ridge along the first 60% of the tail, in base radii (a "
                       "lizard's); 0 = a mammal's smooth back"),
    )
    presets = {"tail_base": {"label": "Tail base (sacrococcygeal midline)", "face": "tail_base",
                             "up_hint": (0.0, 1.0, 0.0), "offset": TAIL_SINK,
                             "side": "center"}}

    def load(self, params):
        # the pelvis pays for the tail's size: its length and its thickness (the base radius), over
        # their values on the neutral body (creature_anatomy.TailEffect)
        return (float(params.get("length", 6.0)) / 6.0) * (float(params.get("base_radius", 0.45)) / 0.45)

    def material(self, params):
        return material_named("skin")

    def resolve_placement(self, spec, values, ctx):
        """The placement of a tail: the stored preset's quad, but

        * the FIRST ``tail_base`` quad (3921, on the side of a buttock, which every document
          written by the first tails stores) is read as the new one on the midline, and the
          first preset's sink (-0.6 along the normal) as the new default (0): the sink is the
          tail's own bone root now. The document is not rewritten;
        * a ``center`` tail on a quad that has an edge on the midline is bound to that quad AND
          its mirror image (``anchor.midline = false`` in the record switches it off), so it is
          placed on the sagittal plane. A tail clicked onto the side of the body stays there."""
        pl = super().resolve_placement(spec, values, ctx)
        anchor = spec.placement.get("anchor")
        if pl["face"] == LEGACY_TAIL_BASE_FACE and pl["weights"] is None:
            pl["face"] = PRESET_FACES["tail_base"]
        if abs(pl["offset"] - LEGACY_TAIL_SINK) < 1e-9:
            pl["offset"] = TAIL_SINK
        want = anchor.get("midline") if isinstance(anchor, dict) else None
        if want is None:
            want = spec.side == "center" and A.is_midline_face(ctx.base, pl["face"])
        pl["midline"] = bool(want) and spec.side == "center"
        return pl

    def body_scale(self, ctx) -> float:
        """The body's height over the neutral body's: what ``length`` and ``base_radius`` scale by."""
        h = float(np.ptp(ctx.cage[0][:, 1]))
        h0 = float(np.ptp(ctx.neutral_cage()[0][:, 1]))
        return h / h0 if h0 > 0 else 1.0

    def tail_params(self, build_values) -> "T.TailParams":
        v = build_values
        return T.TailParams(
            length=float(v["length"]), base_radius=float(v["base_radius"]),
            tip_radius=float(v["tip_radius"]), curl=float(v["curl"]), droop=float(v["droop"]),
            flare=float(v.get("flare", TAIL_FLARE)), segments=int(v["segments"]),
            ring_verts=int(v["ring_verts"]), vertebrae=int(v.get("vertebrae", 0)),
            crest=float(v.get("crest", 0.0)))

    def build(self, spec, side, values, build_values, pl, ctx):
        app = self._engine(side, pl, params={})
        body_v, body_q = ctx.cage
        k_body = self.body_scale(ctx)
        tparams = self.tail_params(build_values)
        names = {i: (n, s) for i, n, s in self._names(spec, app._indices())}
        out = []
        for idx in app._indices():
            name, letter = names[idx]
            frame = app._anchor_frame(body_v, body_q, idx, app._point(idx), ctx.base)
            frame = self._root_frame(frame, pl, k_body, app, body_v, body_q, idx, ctx)
            ctrl = {}
            ctrl, notes = self.clean_controls(side_controls(spec.controls, letter),
                                              f"appendage {spec.id!r}: ")
            for note in notes:
                ctx.warn(note)

            def clamp(c, build_values=build_values):
                return {k: min(max(x, self.control_range(k, None, build_values)[0]),
                                self.control_range(k, None, build_values)[1])
                        for k, x in c.items()}

            pt = T.PlacedTail(tparams, frame, clamp)
            out.append(Placed(name=name, spec_id=spec.id, kind=self, side=letter,
                              anchor=self._surface(ctx, app, idx), matrix=frame,
                              params=dict(values), controls=ctrl, static=pt.skin(ctrl),
                              repose=pt.skin, model=pt,
                              root_depth=tparams.stem * pt.scale))
        return out

    def _root_frame(self, frame, pl, k_body, app, body_v, body_q, idx, ctx):
        """The tail's frame: the engine's, scaled by the body and, on the midline, the axis tilted
        to the sacral surface's elevation clamped to ``pitch_clamp`` degrees (the first part of a
        tail is the continuation of the sacrum's axis, not of whatever way the cage faces)."""
        m = np.array(frame, dtype=np.float64)
        scale = pl["scale"]
        if pl.get("midline"):
            z = m[:3, 2] / scale
            e = math.degrees(math.asin(max(-1.0, min(1.0, float(z[1])))))
            lo, hi = self.pitch_clamp
            d = math.radians(min(max(e, lo), hi) - e)
            if d:
                c, s_ = math.cos(d), math.sin(d)
                rot = np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])
                m[:3, :3] = m[:3, :3] @ rot
                surface = m[:3, 3] - z * pl["offset"]
                m[:3, 3] = surface + (m[:3, 2] / scale) * pl["offset"]
        m[:3, :3] *= k_body
        return m

    def junction(self, placed, controls=None):
        """The skin around the root: see :func:`tailrig.junction_spec` (the sacral rise, the
        cleft kept, the cheeks wrapped onto the tail's flanks)."""
        fillet = float(placed.params.get("fillet", self.fillet_default))
        if fillet <= 0.0 or placed.model is None:
            return None
        return T.junction_spec(placed.name, placed.model.model(controls), placed.model.frame,
                               fillet, self.fillet_height)

    def layers(self, placed, controls=None):
        return placed.model.layers(controls)

    def layer_colors(self, placed, name, controls=None):
        """The muscle sleeve is muscle belly on the proximal part and tendon beyond (1 -> 0)."""
        return placed.model.model(controls).muscle_belly() if name == "muscle" else None

    def builder_args(self, build_values, controls):
        """The arguments of the LEGACY tube builder (``appendages.make_tail``) for these parameters and
        controls: the registry tail is built by :class:`tailrig.TailModel` instead (``tail_params``), this
        stays for the API (a legacy ``Appendage('tail')`` and the print report's analytic law use that
        builder)."""
        c = {k: float(controls.get(k, 0.0)) for k in ("curl", "droop", "lift", "sway")}
        for k in c:
            lo, hi = self.control_range(k, None, build_values)
            c[k] = min(max(c[k], lo), hi)
        return dict(length=build_values["length"], base_radius=build_values["base_radius"],
                    segments=build_values["segments"], ring_verts=build_values["ring_verts"],
                    curl=build_values["curl"] + c["curl"], droop=build_values["droop"] + c["droop"],
                    tip_radius=_tip(build_values), lift=c["lift"], sway=c["sway"],
                    flare=float(build_values.get("flare", 0.0)))

    def control_range(self, name, state=None, params=None):
        params = params or {}
        if name in ("curl", "droop"):
            base = float(params.get(name, self.param(name).default))
            return (-2.0 - base, 2.0 - base)          # the sum stays inside the parameter's range
        if name == "lift":
            return (-60.0, 90.0)
        if name == "sway":
            return (-90.0, 90.0)
        return None


# the horn shapes (hornrig.SHAPES): the parameters per variant, as the first versions' table rows
# for the registry's own introspection (length, base_radius, curve, twist, turns, segments,
# ring_verts); the registry horn is built by hornrig.HornModel
HORN_SHAPES = {name: (row["length"], row["base_radius"], row["curve"], row["twist"], row["turns"],
                      row["segments"], HORN_RING_VERTS) for name, row in HR.SHAPES.items()}

# Where a horn sits on the head. The research (docs/anatomy_horn.md) found these quads on the
# neutral cage: the crown of the frontal bone just in front of the coronal suture (bovid and
# devil horns), the frontal eminence (short brow horns) and the frontotemporal crest above the
# temporal fossa (a ram's). The first preset's point, ``temples``, is on the PARIETAL bone behind
# the ear plane, 0.6 behind the bregma: records that still name it (or its quad) are read as the
# shape's own site, see ``HornKind.resolve_placement``.
HORN_PRESET_POINTS = {
    "horn_crown": (0.473, 8.283, 0.980),
    "horn_brow": (0.354, 8.215, 1.216),
    "horn_temple_crest": (0.616, 8.011, 0.809),
}
HORN_PRESET_FACES = {"horn_crown": 11861, "horn_brow": 7314, "horn_temple_crest": 11860}
PRESET_POINTS.update(HORN_PRESET_POINTS)
PRESET_FACES.update(HORN_PRESET_FACES)
LEGACY_HORN_FACE = PRESET_FACES["temples"]
HORN_LOAD_REF = 1.6 * 0.17 ** 2       # length x radius^2 of the default horn: load 1
HORN_LOAD_CAP = 2.5


def _horn_param_specs() -> tuple:
    rows = HR.SHAPES
    out = [ParamSpec("shape", "Shape", "choice", "curved", choices=tuple(rows),
                     help="curved | ram | spiral | antelope | ossicone: sets the defaults below")]
    for key, label, field, lo, hi, step, typ, help_ in (
            ("length", "Length", "length", 0.3, 12.0, 0.1, "float",
             "on the neutral head; scaled with the head (the eye distance)"),
            ("base_radius", "Base radius", "base_radius", 0.05, 0.6, 0.005, "float",
             "on the neutral head, scaled with it and never over 0.4 of the head's half-width"),
            ("segments", "Segments", "segments", 3, 60, 1, "int", ""),
            ("curve", "Curve", "curve", -4.0, 4.0, 0.01, "float",
             "the total turn of the axis in radians is curve x length / 2: an equiangular spiral"),
            ("twist", "Twist", "twist", -3.0, 3.0, 0.01, "float", "turns the bend plane"),
            ("turns", "Turns", "turns", -4.0, 4.0, 0.05, "float", "a corkscrew's turns"),
            ("tip_radius", "Tip radius", "tip", 0.0, 0.3, 0.005, "float",
             "radius at the end, which is rounded (at least 0.012)"),
            ("flare", "Root flare", "flare", 0.0, 1.5, 0.05, "float",
             "the keratin sheath's broad base: 1 + flare times the radius at the skin, easing out "
             "over about a third of a base radius"),
            ("rings", "Growth rings", "rings", 0.0, 0.15, 0.005, "float",
             "annulus amplitude in radii (a bovid sheath's transverse ridges); 0 = smooth"),
            ("tilt", "Tilt from the skull", "tilt", 0.0, 60.0, 1.0, "float",
             "degrees between the horn's axis and the skull's surface normal (15 to 45 reads as "
             "grown out of the bone)")):
        for shape, row in rows.items():
            out.append(ParamSpec(key, label, typ, row[field], lo, hi, step, help=help_,
                                 when=("shape", (shape,))))
    out.append(ParamSpec("ring_verts", "Ring vertices", "int", HORN_RING_VERTS, 5, 24, 1))
    out.append(ParamSpec("boss", "Skin boss", "float", 1.0, 0.0, 2.0, 0.05,
                         help="how high the scalp is tented around the root, in tents (the base "
                              "radius x 0.35, 0.5 for a ram); 0 = none"))
    out.append(fillet_param(HORN_FILLET))
    return tuple(out)


class HornPlacement:
    """What a placed horn carries beyond its mesh (kept on ``Placed.repose.horn``): the model
    and site it was built from, the skin plan, the skin it comes out of and what the junction needs.
    The layers under the skin (``bone``, ``muscle``) are built from it on demand, in the horn's local
    frame, and carried to each side."""

    def __init__(self, model, site, plan, keeps, cage, head, matrix, field, raw_field):
        self.model, self.site, self.plan, self.keeps = model, site, plan, keeps
        self.cage, self.head, self.matrix = cage, head, matrix
        self.field, self.raw_field = field, raw_field
        self._skin = None
        self._layers = None

    def skin(self):
        """The sheath in the horn's local frame (the same on both sides: the right is the mirror)."""
        if self._skin is None:
            self._skin = self.model.skin(self.field)
        return self._skin

    def layers_local(self):
        """``{name: Mesh}`` of the bone and the soft tissue in the horn's local frame."""
        if self._layers is None:
            sc = self.site.scale
            rho_max = self.plan.tent_radius / sc * 1.05
            bone = self.model.bone(self.field)
            plate = HR.skull_plate(self.model, self.raw_field, rho_max, HR.SCALP_THICK / sc,
                                   0.5 * HR.SCALP_THICK / sc)
            soft = HR.soft_tissue(self.model, self.field, self.raw_field, rho_max,
                                  HR.SKIN_THICK / sc, HR.SCALP_THICK / sc)
            self._layers = {"bone": Mesh.merge([bone, plate]), "muscle": soft}
        return self._layers


class HornKind(_LoftKind):
    """A horn: the frontal bone's cornual process, the scalp tented around it and the keratin
    sheath over both (``hornrig.py``), placed on the frontal bone (the crown, the brow or the
    frontotemporal crest) and sized by the head (``length`` and ``base_radius`` are the values on
    the neutral head: the eye distance sets the scale, and the radius is never more than 0.40 of
    the head's half-width, nor more than keeps the tent 0.075 off the midline). The axis is the skull's own surface normal tilted by ``tilt`` toward the
    shape's lean, and the bend plane comes from the head's axes: neither is a hint in the record.
    ``shape`` picks a variant (curved, ram, spiral, antelope, or an ossicone: skin-covered bone).
    The curve is an equiangular spiral (``curve`` x length / 2 radians in total). No pose controls
    (a horn does not move)."""

    id = "horn"
    label = "Horn"
    default_side = "both"
    paired = True
    fillet_default = HORN_FILLET
    fillet_height = HORN_FILLET_HEIGHT
    junction_owner = "head"
    placement_keys = ("anchor", "scale", "roll", "offset")
    params = _horn_param_specs()
    presets = {
        "horn_crown": {"label": "Crown (frontal bone, in front of the coronal suture)",
                       "face": "horn_crown", "offset": HORN_SINK, "side": "both"},
        "horn_brow": {"label": "Brow (the frontal eminence)", "face": "horn_brow",
                      "offset": HORN_SINK, "side": "both"},
        "horn_temple_crest": {"label": "Temple crest (frontotemporal, a ram's)",
                              "face": "horn_temple_crest", "offset": HORN_SINK, "side": "both"},
        # the first version's name: the parietal eminence behind the ear plane. Records that name it
        # build at the shape's own site (resolve_placement)
        "temples": {"label": "Temples (legacy: the shape's own site)",
                    "face": "temples", "offset": HORN_SINK, "side": "both"},
    }

    def default_preset(self, values):
        return HR.SHAPES.get(values.get("shape", "curved"), HR.SHAPES["curved"])["site"]

    def load(self, params):
        # the neck pays for the horn's mass and its lever arm: length x radius^2 over the default
        # horn's (hornrig: ``load``), capped (a record that says 12 x 0.6 does not need a neck of 40)
        return min(float(params.get("length", 1.6)) * float(params.get("base_radius", 0.17)) ** 2
                   / HORN_LOAD_REF, HORN_LOAD_CAP)

    def material(self, params):
        return material_named(HR.SHAPES.get(params.get("shape", "curved"), HR.SHAPES["curved"])
                              ["material"])

    def draft_params(self, values):
        # the growth rings make 3x the rings: a draft does without them
        return {"rings": 0.0}

    def resolve_placement(self, spec, values, ctx):
        """The record's placement, with two readings of the first version's records:

        * the first preset's quad (``temples``, 12116, on the parietal eminence behind the ear: the
          horns came out of the back of the skull) is read as the shape's own site on the frontal
          bone (``horn_crown``, the ram's ``horn_temple_crest``): the document is not rewritten;
        * the first preset's sink (-0.55 along the surface normal, a constant 2.7 radii) is read as
          "the anatomical root" (the sheath's foot follows the skin down to its bowl,
          ``hornrig.HornModel.foot_rings``): ``offset`` is a slide of the horn's origin along its axis
          now, 0 by default (any other value is that many units out, negative = deeper)."""
        pl = super().resolve_placement(spec, values, ctx)
        if pl["face"] == LEGACY_HORN_FACE and pl["weights"] is None:
            site = HR.SHAPES.get(values.get("shape", "curved"), HR.SHAPES["curved"])["site"]
            pl["face"] = PRESET_FACES[site]
        if abs(pl["offset"] - HORN_SINK) < 1e-9:
            pl["offset"] = 0.0
        return pl

    def horn_params(self, build_values) -> "HR.HornParams":
        v = build_values
        return HR.HornParams(
            shape=v.get("shape", "curved"), length=float(v["length"]),
            base_radius=float(v["base_radius"]), segments=int(v["segments"]),
            ring_verts=int(v["ring_verts"]), curve=float(v["curve"]), twist=float(v["twist"]),
            turns=float(v["turns"]), tip_radius=float(v["tip_radius"]),
            flare=float(v.get("flare", 0.15)), rings=float(v.get("rings", 0.0)),
            tilt=float(v["tilt"]))

    def build(self, spec, side, values, build_values, pl, ctx):
        app = self._engine(side, pl, params={})
        head = HR.head_frame(ctx.base, ctx.verts, ctx.cage)
        hp = self.horn_params(build_values)
        names = {i: (n, s) for i, n, s in self._names(spec, app._indices())}
        left = self._surface(ctx, app, 0)               # the stored quad's point (the left-hand one)
        site = HR.horn_site(ctx.cage, head, left, hp, scale=pl["scale"], roll=pl["roll"])
        boss = float(values.get("boss", 1.0))
        hp = hp.with_(base_radius=site.base_radius, boss=boss)
        model = HR.HornModel(hp)
        ml = site.matrix(float(pl["offset"]))
        inv = np.linalg.inv(ml)
        plan = HR.skin_plan(hp, site, float(values.get("fillet", self.fillet_default)),
                            self.fillet_height)
        plan = HR.SkinPlan(plan.tent_radius, plan.tent_height * max(boss, 0.0), plan.fillet,
                           plan.fillet_height)
        k = head.k
        keeps = []
        for ctr, rad in ((head.ear, 0.30), (head.eye, 0.30)):
            keeps.append(((inv @ np.append(ctr, 1.0))[:3], rad * k / site.scale))
        # the skin the sheath comes out of: the cage as the junction moulds it around this horn (the
        # foot of the sheath must lie on that surface, not on the unmoulded one)
        cage = ctx.cage
        jspec = (HR.junction_spec("horn", model, site, plan, keeps).transformed(ml)
                 if plan.fillet > 0 else None)
        if jspec is not None:
            cage = HR.moulded_cage(ctx.base, ctx.verts, jspec)
        n_local = inv[:3, :3] @ site.normal
        reach = model.R * (1.0 + hp.flare) * 2.4 + 0.2
        fallback = HR.PlaneField(n_local)
        field = HR.SkinField(cage, ml, reach=reach, fallback=fallback)
        raw_field = field if cage is ctx.cage else HR.SkinField(ctx.cage, ml, reach=reach,
                                                                fallback=fallback)
        flip = np.diag([-1.0, 1.0, 1.0, 1.0])
        out = []
        for idx in app._indices():
            name, letter = names[idx]
            m = ml if idx == 0 else flip @ ml
            anchor = left if idx == 0 else flip[:3, :3] @ left
            hpl = HornPlacement(model, site, plan, tuple(keeps), ctx.cage, head, m, field, raw_field)
            mesh = hpl.skin().transform(m)

            def repose(controls, mesh=mesh):
                return mesh

            repose.horn = hpl
            out.append(Placed(name=name, spec_id=spec.id, kind=self, side=letter,
                              anchor=anchor, matrix=m, params=dict(values), controls={},
                              static=mesh, repose=repose,
                              root_depth=float(model.p.sink * site.scale)))
        return out

    def junction(self, placed, controls=None):
        """The skin around the root: the tube of the sheath's first part with the tent as a mound
        along the horn's axis (``hornrig.junction_spec``), the ear and the eye kept."""
        fillet = float(placed.params.get("fillet", self.fillet_default))
        hpl = getattr(placed.repose, "horn", None)
        if fillet <= 0.0 or hpl is None:
            return None
        spec = HR.junction_spec(placed.name, hpl.model, hpl.site, hpl.plan, hpl.keeps)
        return spec.transformed(placed.matrix)

    def layers(self, placed, controls=None):
        hpl = getattr(placed.repose, "horn", None)
        if hpl is None:
            return {"skin": placed.mesh(controls)}
        m = placed.matrix
        loc = hpl.layers_local()
        return {"bone": loc["bone"].transform(m), "muscle": loc["muscle"].transform(m),
                "skin": placed.mesh(controls)}


# -- the wing ----------------------------------------------------------------------------

# the muscular upper arm of the shoulder anatomy (wingroot.py, wings.Sleeve), per style: it reaches
# 0.14 of the humerus' length in radius where the arm crosses the skin (0.135 / 0.149 / 0.137 on the
# bat / dragon / feathered wing). The feathered arm is already as thick as a human forearm at the
# shoulder (0.25-0.33), so it gains about half of what the bat's bone-thin rod does
WING_SLEEVES = {"membrane": W.Sleeve(gain=1.1), "dragon": W.Sleeve(gain=1.0),
                "feathered": W.Sleeve(gain=0.5)}

_MEMBRANE_STYLES = ("membrane", "dragon")
# the DRAGON style differs from the bat in these (wings.DRAGON); a test pins the defaults
# below to what the builders really use
_WING_POSE_LABELS = (
    ("fold", "Fold", 0.0), ("raise_", "Raise", 0.0), ("swing", "Swing", 0.0),
    ("twist", "Twist", 0.0), ("fan", "Fan", 1.0), ("curl", "Curl", 0.0),
    ("wrist_flex", "Wrist flex", 0.0), ("hand_twist", "Hand twist", 0.0))


def _wing_arm_radius(placed_wing) -> float:
    """The radius of the wing's arm bone in the wing's own units (the median distance of the
    ``arm`` piece's vertices to the shoulder-elbow axis; 0.2 if the style has no such piece)."""
    rig = placed_wing.rig
    piece = next((q for q in rig.pieces if q.name == "arm"), None)
    if piece is None:
        return 0.2
    j = rig.posed_joints(None)
    a, b = np.asarray(j["shoulder"]), np.asarray(j["elbow"])
    ab = b - a
    t = np.clip(((piece.mesh.vertices - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return float(np.median(np.linalg.norm(piece.mesh.vertices - (a + t[:, None] * ab), axis=1)))


class WingKind(Kind):
    id = "wing"
    label = "Wing"
    default_side = "both"
    paired = True
    controls = tuple(ControlSpec(*c) for c in _WING_POSE_LABELS)
    pose_presets = True
    params = (
        ParamSpec("style", "Style", "choice", "membrane", choices=("membrane", "dragon", "feathered"),
                  essential=True, help="bat, dragon or bird/angel wing"),
        ParamSpec("span", "Span", "float", 13.0, 3.0, 30.0, 0.5),
        # -- the membrane (bat) and dragon wing
        ParamSpec("thickness", "Thickness", "float", 0.22, 0.05, 0.8, 0.01,
                  when=("style", _MEMBRANE_STYLES)),
        ParamSpec("sweep", "Sweep", "float", 0.0, 0.0, 0.8, 0.01, when=("style", _MEMBRANE_STYLES)),
        ParamSpec("droop", "Droop", "float", 0.0, -0.4, 0.8, 0.01, when=("style", _MEMBRANE_STYLES)),
        ParamSpec("gull", "Gull (arm crook)", "float", 0.16, -0.4, 0.5, 0.01,
                  when=("style", _MEMBRANE_STYLES)),
        ParamSpec("camber", "Camber", "float", 0.06, 0.0, 0.2, 0.005, when=("style", ("membrane",))),
        ParamSpec("camber", "Camber", "float", 0.05, 0.0, 0.2, 0.005, when=("style", ("dragon",))),
        ParamSpec("root_chord", "Root chord", "float", 1.4, 0.3, 4.0, 0.1,
                  when=("style", _MEMBRANE_STYLES)),
        ParamSpec("billow", "Billow", "float", 0.06, 0.0, 0.3, 0.005, when=("style", _MEMBRANE_STYLES)),
        ParamSpec("propatagium", "Leading-edge membrane", "bool", True,
                  when=("style", _MEMBRANE_STYLES)),
        ParamSpec("rows", "Rows", "int", 22, 4, 48, 1, when=("style", _MEMBRANE_STYLES)),
        ParamSpec("spike", "Finger tip spikes", "float", 0.10, 0.0, 0.6, 0.01,
                  when=("style", ("membrane",))),
        ParamSpec("spike", "Finger tip spikes", "float", 0.30, 0.0, 0.6, 0.01,
                  when=("style", ("dragon",))),
        ParamSpec("bone_scale", "Bone thickness", "float", 1.0, 0.3, 2.0, 0.05,
                  when=("style", ("membrane",))),
        ParamSpec("bone_scale", "Bone thickness", "float", 1.35, 0.3, 2.0, 0.05,
                  when=("style", ("dragon",))),
        ParamSpec("spur", "Wrist spur", "float", 0.0, 0.0, 0.5, 0.01, when=("style", ("membrane",))),
        ParamSpec("spur", "Wrist spur", "float", 0.16, 0.0, 0.5, 0.01, when=("style", ("dragon",))),
        # -- the feathered wing
        ParamSpec("thickness", "Thickness", "float", 0.20, 0.05, 0.8, 0.01,
                  when=("style", ("feathered",))),
        ParamSpec("sweep", "Sweep", "float", 0.0, 0.0, 0.8, 0.01, when=("style", ("feathered",))),
        ParamSpec("droop", "Droop", "float", 0.10, -0.4, 0.8, 0.01, when=("style", ("feathered",))),
        ParamSpec("primaries", "Primaries", "int", 10, 3, 20, 1, when=("style", ("feathered",))),
        ParamSpec("secondaries", "Secondaries", "int", 14, 3, 24, 1, when=("style", ("feathered",))),
        ParamSpec("tertials", "Tertials", "int", 7, 1, 14, 1, when=("style", ("feathered",))),
        ParamSpec("scapulars", "Scapulars", "int", 7, 0, 14, 1, when=("style", ("feathered",))),
        ParamSpec("coverts", "Coverts", "bool", True, when=("style", ("feathered",))),
        ParamSpec("underwing", "Underwing coverts", "bool", True, when=("style", ("feathered",))),
        ParamSpec("bones", "Show the arm bones", "bool", True),
        fillet_param(WING_FILLET),
    )
    fillet_default = WING_FILLET
    fillet_height = WING_FILLET_HEIGHT
    presets = {"shoulder_blades": {
        "label": "Shoulder blades", "face": "shoulder_blades", "up_hint": A.WING_UP_HINT,
        "roll": A.WING_ROLL, "offset": A.WING_OFFSET, "side": "both"}}
    placement_keys = ("anchor", "scale", "roll", "offset", "up_hint", "aim")

    def load(self, params):
        from .wings import DEFAULT_SPAN
        return float(params.get("span", DEFAULT_SPAN)) / DEFAULT_SPAN

    def junction(self, placed, controls=None):
        """The skin around the root: the shoulder anatomy's relief (the scapular plateau, the
        latissimus fan, the trapezius slope and the deltoid-level cap, :mod:`wingroot`) and the arm
        sleeve as the tube the skin flows into. A wing without an arm piece (``bones`` off) has no
        junction: there is nothing for the skin to flow into."""
        fillet = float(placed.params.get("fillet", self.fillet_default))
        if fillet <= 0.0 or placed.model is None:
            return None
        ctl = placed.controls if controls is None else controls
        return placed.model.spec(ctl, placed.name)

    def layers(self, placed, controls=None):
        """bone (scapula plate, strut, humerus), muscle (the masses and the arm's belly) and skin
        (the wing's shell): :meth:`wingroot.WingRoot.layers`; the skin only without an arm piece."""
        if placed.model is None:
            return Kind.layers(self, placed, controls)
        return placed.model.layers(placed.mesh(controls), controls)

    def draft_params(self, values):
        if values.get("style") == "feathered":
            # the arm stays: it is the sleeve the body's skin flows into (wingroot.py), a few hundred
            # triangles, and the draft's skin must have the same junction as the final's
            return {"coverts": False, "underwing": False}
        return {}

    def material(self, params):
        return material_named(params.get("style", "membrane"))

    # -- controls --------------------------------------------------------------------------

    def control_range(self, name, state=None, params=None):
        from .wingrig import control_range
        if name not in {c.key for c in self.controls}:
            return None
        return control_range(name, float((state or {}).get("fold", 0.0)))

    def clean_controls(self, controls, owner=""):
        from .wingrig import PRESETS
        out, notes = {}, []
        for key, value in controls.items():
            if key == "preset":
                if value in PRESETS:
                    out[key] = value
                else:
                    near = difflib.get_close_matches(str(value), list(PRESETS), n=1)
                    notes.append(f"{owner}unknown pose preset {value!r} ignored"
                                 + (f" (did you mean {near[0]!r}?)" if near else ""))
            elif key in A.Appendage.POSE_KEYS:
                if (isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value)):
                    out[key] = float(value)
                else:
                    notes.append(f"{owner}control {key} = {value!r} is not a finite number; "
                                 "ignored")
            else:
                near = difflib.get_close_matches(str(key), list(A.Appendage.POSE_KEYS), n=1)
                notes.append(f"{owner}unknown control {key!r} is kept in the file but ignored"
                             + (f" (did you mean {near[0]!r}?)" if near else ""))
        return out, notes

    # -- placement -------------------------------------------------------------------------

    def _aim(self, style: str, aim, ctx: BuildContext) -> dict:
        """Where the attachment line ends, as Appendage keywords. A record may carry its
        own ``aim`` ({face, offset} or {point, relative}); otherwise the style's default:
        a membrane aims at the waist QUAD (so it follows the waist), a feathered wing
        only aims, from a point relative to its anchor."""
        if isinstance(aim, dict) and "face" in aim:
            n_quads = len(ctx.cage[1])
            face = aim["face"]
            if not isinstance(face, int) or isinstance(face, bool) or not 0 <= face < n_quads:
                raise SpecError(f"aim face {face!r} is not a quad of this body")
            off = _vec(aim.get("offset", (0.0, 0.0, 0.0)), None)
            if off is None:
                raise SpecError("aim offset must be three numbers")
            return {"root_face": face, "root_offset": off}
        if isinstance(aim, dict) and "point" in aim:
            p = _vec(aim["point"], None)
            if p is None:
                raise SpecError("aim point must be three numbers")
            return {"root_point": p, "root_relative": bool(aim.get("relative", False))}
        if style in A.WING_PLACEMENT:
            return dict(A.WING_PLACEMENT[style])
        return {"root_face": PRESET_FACES["waist"], "root_offset": WING_ROOT_OFFSET}

    def build(self, spec, side, values, build_values, pl, ctx):
        controls_raw = spec.controls
        shape = {k: v for k, v in build_values.items() if k != "fillet"}   # not a builder keyword
        sleeve = WING_SLEEVES[values["style"]]
        shape["sleeve"] = sleeve                     # the muscular upper arm of the shoulder anatomy
        app = self._engine(side, pl, params=shape,
                           **self._aim(values["style"], pl["aim"], ctx))
        names = {i: (n, s) for i, n, s in self._names(spec, app._indices())}
        out = []
        lm = surface = None
        for idx, pw in app.place_rigs(ctx.base, ctx.verts):
            if lm is None:                           # the body's girdle, once for the pair
                lm = WR.landmarks(ctx.base, ctx.verts)
                surface = WR.BodySurface(*ctx.cage)
            name, letter = names[idx]
            ctrl, notes = self.clean_controls(side_controls(controls_raw, letter),
                                              f"appendage {spec.id!r}: ")
            for note in notes:
                ctx.warn(note)
            arm = next((q for q in pw.rig.pieces if q.name == "arm"), None)
            model = None if arm is None else WR.WingRoot(
                letter or ("r" if idx else "l"), pw, pw.matrix, lm, surface,
                WR.arm_profile(arm.mesh.vertices, sleeve.sides, pw.matrix),
                float(values.get("fillet", self.fillet_default)), values["style"], WING_FILLET_HEIGHT)
            out.append(Placed(name=name, spec_id=spec.id, kind=self, side=letter,
                              anchor=self._surface(ctx, app, idx), matrix=pw.matrix,
                              params=dict(values), controls=ctrl, rig=pw, model=model))
        return out


# -- extra limbs -------------------------------------------------------------------------

LIMB_CHAINS = {"arm": ("l-shoulder", "l-elbow", "l-hand"),
               "leg": ("l-upper-leg", "l-knee", "l-ankle")}


class HarvestKind(_NumericControls, Kind):
    """An extra arm or leg cut out of the character's own body (``Appendage`` 'harvest').

    The copy is the character's own limb, so it matches their build and re-harvests on every
    body. It is bound to a body quad, sunk along its own axis so its cut end is buried in
    the torso, and posed by a :class:`~minimaster.character.limbrig.LimbRig` about the
    shoulder (hip) joint: ``raise_``, ``swing``, ``twist`` and ``elbow`` (``knee`` for a
    leg). A preset is a place on the body AND the pose the limb starts in (``lower_arms``
    hang below the original pair, ``upper_arms`` are raised above it).

    Honoured placement: ``anchor`` and ``offset`` (the sink along the limb axis) and
    ``scale`` (about the shoulder joint; a smaller second pair). NOT offered:
    ``roll`` (the ``twist`` control is the roll about the limb) and ``up_hint`` (the
    direction is the pose). Lengths (``radius``, ``trim``, the sink) are in units of the
    neutral body's limb and scale with the character's own limb (``LIMB_REFERENCE_LENGTH``).
    """

    id = "harvest"
    label = "Extra limb"
    default_side = "both"
    paired = True
    placement_keys = ("anchor", "scale", "offset")
    pose_presets = False
    controls = (ControlSpec("swing", "Swing (forward)", 0.0),
                ControlSpec("raise_", "Raise (away from the body)", 0.0),
                ControlSpec("twist", "Twist about the limb", 0.0),
                ControlSpec("elbow", "Elbow flex", 0.0),
                ControlSpec("knee", "Knee flex", 0.0))
    params = (
        ParamSpec("limb", "Limb", "choice", "arm", choices=("arm", "leg"), essential=True,
                  help="copy the character's arm or leg"),
        ParamSpec("radius", "Reach around the bones", "float", 1.15, 0.9, 2.5, 0.05,
                  when=("limb", ("arm",))),
        ParamSpec("radius", "Reach around the bones", "float", 1.5, 0.9, 2.5, 0.05,
                  when=("limb", ("leg",))),
        ParamSpec("trim", "Trim at the root", "float", HARVEST_TRIM, 0.0, 2.0, 0.05),
        ParamSpec("neck", "Root thickness", "float", HARVEST_NECK, 0.2, 1.0, 0.05,
                  help="how much of its width the cut end keeps (1 = the limb as cut): "
                       "a narrow root fits inside the torso"),
        fillet_param(HARVEST_FILLET),
    )
    fillet_default = HARVEST_FILLET
    presets = {
        "lower_arms": {"label": "Lower arms (a second pair below the original)",
                       "face": "lower_arms", "offset": -2.0, "side": "both",
                       "params": {"limb": "arm"},
                       "controls": {"raise_": 10.0, "swing": 10.0, "elbow": 10.0}},
        "upper_arms": {"label": "Upper arms (a second pair raised above the original)",
                       "face": "upper_arms", "offset": -1.8, "side": "both",
                       "params": {"limb": "arm"},
                       "controls": {"raise_": 65.0, "swing": -5.0, "elbow": -15.0}},
        "extra_legs": {"label": "Extra legs (a second pair splayed out from the waist)",
                       "face": "extra_legs", "offset": -2.6, "side": "both",
                       "params": {"limb": "leg"},
                       "controls": {"raise_": 45.0, "swing": -10.0, "knee": 20.0}},
        # the first version's presets: a straight copy at the original's own shoulder
        # (it overlaps the original arm); kept so older documents still build
        "shoulder_socket": {"label": "Shoulder socket (legacy: overlaps the original arm)",
                            "face": "shoulder_socket", "offset": HARVEST_SINK, "side": "both"},
        "hip_socket": {"label": "Hip socket (legacy)", "face": "hip_socket",
                       "offset": HARVEST_SINK, "side": "both"},
    }

    def default_preset(self, values):
        return "extra_legs" if values.get("limb") == "leg" else "lower_arms"

    def junction(self, placed, controls=None):
        """The limb (from the pivot inside the torso, through the elbow, a little into the
        forearm, in the pose it is in) as the tube the flank's skin flows into. The radii
        are the character's own limb's, as fractions of its upper bone (measured on the
        neutral body: ``LIMB_RADII``), narrowed like the cut end is (``neck``): never fatter
        than the mesh, so the skin that "is inside the tube" really is hidden by it."""
        fillet = float(placed.params.get("fillet", self.fillet_default))
        if fillet <= 0.0:
            return None
        ctl = dict(placed.controls if controls is None else controls)
        j = placed.limb.joints(ctl)
        sh, el, ha = (np.asarray(j[k]) for k in ("shoulder", "elbow", "hand"))
        upper = float(np.linalg.norm(el - sh))
        prof = LIMB_RADII[placed.params.get("limb", "arm")]
        s_up, r_up, s_lo, r_lo = prof
        pts = [sh + t * (el - sh) for t in s_up] + [el + t * (ha - el) for t in s_lo]
        radii = np.array(list(r_up) + list(r_lo)) * upper
        neck = float(placed.params.get("neck", HARVEST_NECK))
        along = np.array(list(s_up) + [1.0] * len(s_lo))
        t = np.clip(along * upper / max(A.NECK_LENGTH * upper, 1e-9), 0.0, 1.0)
        radii = radii * (neck + (1.0 - neck) * t * t * (3.0 - 2.0 * t)) * J.HUG
        base_r = float(radii[-len(s_lo) - 1]) / J.HUG
        # the neutral body's limb is only a model of the placed copy (a different body, a pose, a
        # root tapered to fit the torso): the skin may only count as hidden where the mesh IS
        radii = J.fit_radii_to_mesh(pts, radii, placed.mesh(controls).vertices)
        return J.JunctionSpec(placed.name, pts, radii, width=fillet, base_radius=base_r,
                              height=HARVEST_FILLET_HEIGHT, owner="trunk")

    def material(self, params):
        return material_named("skin")

    def control_range(self, name, state=None, params=None):
        """The range of ``name`` for a typical limb of the kind (an arm unless ``params``
        say leg); a placed limb knows its own (``Placed.control_range``)."""
        if name not in {c.key for c in self.controls}:
            return None
        from . import limbrig
        return limbrig.typical_geometry((params or {}).get("limb", "arm")).control_range(
            name, state)

    def build(self, spec, side, values, build_values, pl, ctx):
        if not pl["scale"] > 0.0:
            raise SpecError(f"scale = {pl['scale']:g} must be positive")
        chain = LIMB_CHAINS[values["limb"]]
        params = {k: build_values[k] for k in ("radius", "trim", "neck")}
        app = self._engine(side, pl, chain=chain, params=params)
        names = {i: (n, s) for i, n, s in self._names(spec, app._indices())}
        out = []
        try:
            rigs = app.place_limbs(ctx.base, ctx.verts)
        except ValueError as exc:          # e.g. "selected no faces": this body has no such limb
            raise SpecError(f"cannot cut the {values['limb']} out of this body ({exc})") from None
        for idx, rig in rigs:
            name, letter = names[idx]
            owner = f"appendage {spec.id!r}: "
            ctrl, notes = self.clean_controls(side_controls(spec.controls, letter), owner)
            for key in list(ctrl):
                if key not in rig.control_names:      # an elbow on a leg, a knee on an arm
                    notes.append(f"{owner}control {key!r} does not apply to {self._article(rig)}"
                                 f" (it has {', '.join(rig.control_names)}); ignored")
                    del ctrl[key]
            canon = {("elbow" if k == "knee" else k): val for k, val in ctrl.items()}
            for key in ("raise_", "twist", "elbow", "swing"):    # swing's range depends on raise_
                if key in canon:
                    lo, hi = rig.geometry.control_range(key, canon)
                    if not lo - 1e-9 <= canon[key] <= hi + 1e-9:
                        notes.append(f"{owner}control {key} = {canon[key]:g} is outside "
                                     f"[{lo:.0f}, {hi:.0f}] for this limb: using "
                                     f"{min(max(canon[key], lo), hi):.0f}")
            for note in notes:
                ctx.warn(note)
            out.append(Placed(name=name, spec_id=spec.id, kind=self, side=letter,
                              anchor=self._surface(ctx, app, idx), matrix=None,
                              params=dict(values), controls=ctrl, limb=rig,
                              static=rig.pose(ctrl)))
        return out

    @staticmethod
    def _article(rig) -> str:
        return "a leg" if rig.limb == "leg" else "an arm"

    def id_stem(self, params):
        return "leg" if params.get("limb") == "leg" else "arm"


# --------------------------------------------------------------------------
# materials (a kind's look; the per-vertex albedo of tools/wing_view.py stays there)


_MATERIALS: dict = {}


def material_named(name: str):
    """The :class:`~minimaster.character.shading.Material` a kind renders with (``skin``,
    ``bone``, or a wing style). Built on first use so importing this module stays cheap."""
    if not _MATERIALS:
        from . import shading as sh
        _MATERIALS.update(
            skin=sh.SKIN,
            bone=sh.Material(base_color="#d8cfb8", roughness=0.45, specular=0.5, f0=0.04,
                             wrap=0.15, scatter="#a89878", rim=0.15),
            membrane=sh.Material(base_color="#6f4f3f", roughness=0.74, specular=0.28,
                                 f0=0.035, wrap=0.40, scatter="#8c4a38", rim=0.24),
            dragon=sh.Material(base_color="#5a3d36", roughness=0.70, specular=0.30,
                               f0=0.035, wrap=0.40, scatter="#8a3b2c", rim=0.26),
            feathered=sh.Material(base_color="#cdc4b6", roughness=0.55, specular=0.42,
                                  f0=0.040, wrap=0.26, scatter="#9a8f80", rim=0.20))
    return _MATERIALS.get(name, _MATERIALS["skin"])


# --------------------------------------------------------------------------
# the registry

KINDS: dict[str, Kind] = {}


def register(kind: Kind) -> Kind:
    """Add a kind (or replace one of the same id)."""
    if not kind.id:
        raise ValueError("a kind needs an id")
    KINDS[kind.id] = kind
    return kind


for _k in (WingKind(), TailKind(), HornKind(), HarvestKind()):
    register(_k)
del _k


def kind_of(spec_or_id) -> Kind | None:
    """The registered kind of a spec (or a kind id), or ``None`` if there is none here."""
    return KINDS.get(getattr(spec_or_id, "kind", spec_or_id))


def instantiate_all(specs, base, verts=None, lib=None, draft: bool = False,
                    report: list | None = None) -> list:
    """Build every enabled spec on a morphed body: a list of :class:`Placed`.

    What cannot be built is skipped with an :class:`AppendageWarning` (and named in
    ``report``) and left in the document: an unknown kind, a duplicate id, an unknown
    style, an anchor that is not a quad of this body."""
    ctx = BuildContext(base, verts, lib, report)
    out: list = []
    seen: set = set()
    for spec in specs:
        if not spec.enabled:
            continue
        kind = KINDS.get(spec.kind)
        if kind is None:
            ctx.warn(f"appendage {spec.id!r}: unknown kind {spec.kind!r} is kept in the "
                     f"document but not built (known: {', '.join(sorted(KINDS))})")
            continue
        if spec.id in seen or not valid_id(spec.id):
            ctx.warn(f"appendage {spec.id!r}: the id is duplicated or not made of letters, "
                     "digits, '-' and '_'; skipped")
            continue
        seen.add(spec.id)
        try:
            out.extend(kind.instantiate(spec, ctx, draft))
        except SpecError as exc:
            ctx.warn(f"appendage {spec.id!r}: {exc}; skipped")
    return out


# --------------------------------------------------------------------------
# creating specs


def new_spec(kind_id: str, existing=(), *, side: str | None = None, params: dict | None = None,
             preset: str | None = None, scale=None, roll=None, offset=None, at=None,
             face: int | None = None, controls: dict | None = None, name: str = "",
             spec_id: str | None = None, base=None, lib=None) -> AppendageSpec:
    """A new record for ``kind_id`` placed by a preset (or ``at`` a body point, or on a
    cage quad ``face``). Strict, unlike building: an unknown kind, parameter, side or
    preset is a ``ValueError`` with a hint, because this is where a person typed it.

    The anchor is resolved here, once, to a cage quad id; ``base`` (and ``lib`` for ``at``)
    additionally let the record carry the quad's corners so a different base mesh is
    noticed later."""
    kind = KINDS.get(kind_id)
    if kind is None:
        near = difflib.get_close_matches(str(kind_id), sorted(KINDS), n=1)
        raise ValueError(f"unknown kind {kind_id!r}; the kinds are {', '.join(sorted(KINDS))}"
                         + (f" (did you mean {near[0]!r}?)" if near else ""))
    params = dict(params or {})
    known = {p.key for p in kind.params}
    for key in params:
        if key not in known:
            near = difflib.get_close_matches(key, sorted(known), n=1)
            raise ValueError(f"{kind_id} has no parameter {key!r}; it has "
                             f"{', '.join(sorted(known))}" + (f" (did you mean {near[0]!r}?)"
                                                              if near else ""))
    # a preset may imply parameters (the extra_legs preset is for limb=leg)
    for key, val in (kind.presets.get(preset) or {}).get("params", {}).items() if preset else ():
        if key in params and params[key] != val:
            raise ValueError(f"preset {preset!r} is for {key}={val!r}, not {key}={params[key]!r}")
        params.setdefault(key, val)
    values = kind.defaults(**{k: v for k, v in params.items()
                              if any(q.key == k and q.essential for q in kind.params)})
    for key, raw in list(params.items()):
        p = kind.param(key, values)
        if p is None:
            when = next(q.when for q in kind.params if q.key == key)
            raise ValueError(f"{key} only applies when {when[0]} is "
                             f"{' or '.join(str(w) for w in when[1])} "
                             f"(it is {values.get(when[0])!r})")
        coerced = p.coerce(raw)
        if p.type in ("float", "int") and coerced != float(raw):
            raise ValueError(f"{key} = {raw!r} is outside the range {p.lo:g}..{p.hi:g}")
        params[key] = coerced
    side = side or None
    if side is not None and side not in SIDES:
        raise ValueError(f"unknown side {side!r}; use {', '.join(SIDES)}")
    preset_name = preset or kind.default_preset(kind.defaults(**params))
    if preset_name not in kind.presets:
        raise ValueError(f"{kind_id} has no placement preset {preset_name!r}; it has "
                         f"{', '.join(kind.presets)}")
    p = kind.presets[preset_name]
    if at is not None:
        pt = (abs(float(at[0])), float(at[1]), float(at[2]))
        if base is None:
            raise ValueError("placing 'at' a point needs the base mesh")
        face = nearest_face(neutral_cage(base, lib), pt)
    elif face is None:
        face = PRESET_FACES[p["face"]]
    # a person typed this: a bool, a fraction or a quad that does not exist is an error here,
    # not a traceback (or, for a negative number, the LAST quad) later
    n_quads = 13378
    if base is not None:
        n_quads = len(base.body_cage(None)[1])
    if isinstance(face, bool) or not isinstance(face, (int, np.integer)) or not 0 <= int(face) < n_quads:
        raise ValueError(f"face = {face!r} is not a cage quad (an integer from 0 to {n_quads - 1})")
    anchor = {"face": int(face)}
    if base is not None:
        anchor["corners"] = [int(c) for c in base.body_cage(None)[1][face]]
    placement = {"preset": preset_name, "anchor": anchor}
    for key, val in (("scale", scale), ("roll", roll), ("offset", offset)):
        if val is not None:
            if key not in kind.placement_keys:
                raise ValueError(f"{kind_id} does not use {key!r} (it uses "
                                 f"{', '.join(kind.placement_keys)})")
            placement[key] = float(val)
    for key in ("roll", "offset", "scale"):
        if key in kind.placement_keys and key not in placement and key in p:
            placement[key] = p[key]
    if "up_hint" in kind.placement_keys:
        placement.setdefault("up_hint", list(p.get("up_hint", (0.0, 1.0, 0.0))))
    if spec_id is not None and not valid_id(spec_id):
        raise ValueError(f"{spec_id!r} is not a valid id (letters, digits, '-' and '_')")
    if spec_id is not None and any(e.id == spec_id for e in existing):
        raise ValueError(f"there already is an appendage called {spec_id!r}")
    # a preset is a pose as well as a place: the limb starts in the position its name says,
    # unless the caller says what the controls are (an empty dict means "none")
    start_controls = copy.deepcopy(p.get("controls", {})) if controls is None else dict(controls)
    return AppendageSpec(id=spec_id or next_id(kind.id_stem(params), existing), kind=kind_id,
                         side=side or p.get("side", kind.default_side), params=params,
                         controls=start_controls, placement=placement, name=name)


def _scalar(text: str):
    t = text.strip()
    low = t.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        return t


def parse_kv(text: str, what: str = "key=value") -> dict:
    """``"a=1,b=true,c=name"`` -> ``{"a": 1, "b": True, "c": "name"}``."""
    out: dict = {}
    for part in filter(None, (s.strip() for s in text.split(","))):
        if "=" not in part:
            raise ValueError(f"expected {what}, got {part!r}")
        key, _, val = part.partition("=")
        out[key.strip()] = _scalar(val)
    return out


# keys of a CLI --add that are placement / document fields rather than kind parameters
ADD_META_KEYS = ("side", "preset", "id", "name", "scale", "roll", "offset", "at", "face")


def parse_add(text: str):
    """``"wing:style=feathered,span=15,side=left"`` -> ``(kind, params, meta)``."""
    kind, _, rest = text.partition(":")
    kind = kind.strip()
    if not kind:
        raise ValueError(f"--add wants KIND[:key=value,...], got {text!r}")
    kv = parse_kv(rest, "key=value") if rest else {}
    meta = {k: kv.pop(k) for k in ADD_META_KEYS if k in kv}
    if "at" in meta:
        pts = str(meta["at"]).replace("/", " ").split()
        if len(pts) != 3:
            raise ValueError("at= wants three numbers, e.g. at=0.8/4.4/-0.7")
        meta["at"] = tuple(float(x) for x in pts)
    return kind, kv, meta


def add_from_text(text: str, existing=(), base=None, lib=None) -> AppendageSpec:
    """The record a ``--add KIND[:k=v,...]`` asks for."""
    kind, params, meta = parse_add(text)
    meta = dict(meta)
    if "id" in meta:
        meta["spec_id"] = str(meta.pop("id"))
    if "name" in meta:
        meta["name"] = str(meta["name"])
    for k in ("side", "preset"):
        if k in meta:
            meta[k] = str(meta[k])
    return new_spec(kind, existing, params=params, base=base, lib=lib, **meta)


def pose_from_text(text: str, specs) -> tuple:
    """``"wing-1.r:fold=1,raise_=20"`` -> ``(spec, side_key, {control: value})``.

    ``ID`` poses both sides, ``ID.l`` / ``ID.r`` one side only."""
    head, _, rest = text.partition(":")
    if not rest:
        raise ValueError(f"--pose wants ID[.l|.r]:control=value,..., got {text!r}")
    spec_id, _, side_key = head.strip().partition(".")
    spec = next((s for s in specs if s.id == spec_id), None)
    if spec is None:
        have = ", ".join(s.id for s in specs) or "none"
        raise ValueError(f"no appendage {spec_id!r} to pose (the character has: {have})")
    if side_key not in ("", "l", "r"):
        raise ValueError(f"pose side must be .l or .r, not .{side_key}")
    kind = KINDS.get(spec.kind)
    values = parse_kv(rest, "control=value")
    if kind is not None:
        names = {c.key for c in kind.controls} | ({"preset"} if kind.pose_presets else set())
        for key in values:
            if key not in names:
                near = difflib.get_close_matches(key, sorted(names), n=1)
                raise ValueError(f"{spec.kind} has no control {key!r}"
                                 + (f" (it has: {', '.join(sorted(names))})" if names
                                    else " (it has no controls)")
                                 + (f"; did you mean {near[0]!r}?" if near else ""))
    return spec, side_key, values


def apply_pose(spec: AppendageSpec, side_key: str, values: dict) -> None:
    """Record ``values`` as the spec's controls (shared, or for one side)."""
    key = side_key or "*"
    target = spec.controls.get(key)
    if not isinstance(target, dict):
        target = spec.controls[key] = {}
    target.update(values)


def format_kinds() -> str:
    """The text of ``--list-kinds``."""
    lines: list[str] = []
    for kind in KINDS.values():
        lines.append(f"{kind.id}  ({kind.label}; default side: {kind.default_side}; "
                     f"placement: {', '.join(kind.placement_keys)})")
        lines.append(f"  presets: {', '.join(kind.presets)}")
        for p in kind.params:
            if p.type == "choice":
                rng = "|".join(str(c) for c in p.choices)
            elif p.type == "bool":
                rng = "true|false"
            else:
                rng = f"{p.lo:g}..{p.hi:g}" if p.lo is not None and p.hi is not None else "any"
            when = f"  [{p.when[0]}: {'|'.join(p.when[1])}]" if p.when else ""
            lines.append(f"  {p.key:<14s}{rng:<26s} default {p.default!s:<8s}{when}")
        if kind.controls:
            lines.append("  pose controls: " + ", ".join(c.key for c in kind.controls)
                         + (", preset" if kind.pose_presets else "") + " (see --pose)")
        lines.append("")
    return "\n".join(lines).rstrip()
