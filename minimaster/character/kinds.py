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
import math
import warnings
import weakref
from dataclasses import dataclass

import numpy as np

from ..core.mesh import Mesh
from . import appendages as A
from .appendage_spec import SIDES, AppendageSpec, next_id, valid_id


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
    "tail_base": (0.0, 1.6, -1.15),          # below the sacrum
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
    "tail_base": 3921,
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
TAIL_SINK = -0.6
HORN_SINK = -0.55
HARVEST_SINK = -0.5      # along the limb axis: a harvested limb has no surface normal
TAIL_TIP = 0.09          # radius at the end of a tail (the first version's was 0.02: a point)
HORN_TIP = 0.08          # and of a horn (0.012)
HARVEST_TRIM = 0.6       # shaves the root of the copy (units of the neutral limb)
HARVEST_NECK = 0.5       # the cut end keeps this much of the limb's width (limbrig.taper_root)


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
                  side=side)
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
                              repose=repose))
        return out


class TailKind(_LoftKind):
    """A tail: a tapered tube with a rounded tip, bound to the quad below the sacrum.

    Pose controls re-evaluate the loft: ``curl`` and ``droop`` ADD to the parameters of the
    same name (same units, so 0 is the tail as built; the sum stays inside the parameter's
    range), ``lift`` (up) and ``sway`` (toward +X) bend the whole tail from its base, in
    degrees. ``tip_radius`` is the radius at the end: the first version's tail ended in a
    point 0.07 mm across at 32 mm, below what a printer resolves; the default is a rounded
    end 0.3 mm across."""

    id = "tail"
    label = "Tail"
    default_side = "center"
    paired = False
    controls = (ControlSpec("curl", "Curl (added to the shape)", 0.0),
                ControlSpec("droop", "Droop (added to the shape)", 0.0),
                ControlSpec("lift", "Lift (degrees, up)", 0.0),
                ControlSpec("sway", "Sway (degrees, toward +X)", 0.0))
    params = (
        ParamSpec("length", "Length", "float", 6.0, 1.0, 20.0, 0.1),
        ParamSpec("base_radius", "Base radius", "float", 0.45, 0.1, 2.0, 0.01),
        ParamSpec("segments", "Segments", "int", 12, 4, 40, 1),
        ParamSpec("ring_verts", "Ring vertices", "int", 10, 6, 24, 1),
        ParamSpec("curl", "Curl", "float", 0.55, -2.0, 2.0, 0.01),
        ParamSpec("droop", "Droop", "float", 0.9, -2.0, 2.0, 0.01),
        ParamSpec("tip_radius", "Tip radius", "float", TAIL_TIP, 0.0, 0.4, 0.005,
                  help="radius at the end of the taper, which is rounded; 0 = the first "
                       "version's point (0.07 mm across at 32 mm: below what a printer makes)"),
    )
    presets = {"tail_base": {"label": "Tail base (below the sacrum)", "face": "tail_base",
                             "up_hint": (0.0, 1.0, 0.0), "offset": TAIL_SINK,
                             "side": "center"}}

    def load(self, params):
        return float(params.get("length", 6.0)) / 6.0

    def material(self, params):
        return material_named("skin")

    def builder_args(self, build_values, controls):
        c = {k: float(controls.get(k, 0.0)) for k in ("curl", "droop", "lift", "sway")}
        # a tail outside its ranges facets into a zigzag that overlaps itself (watertight, so the
        # gate never sees it): the document and CLI path clamp as the panel does
        for k in c:
            lo, hi = self.control_range(k, None, build_values)
            c[k] = min(max(c[k], lo), hi)
        return dict(length=build_values["length"], base_radius=build_values["base_radius"],
                    segments=build_values["segments"], ring_verts=build_values["ring_verts"],
                    curl=build_values["curl"] + c["curl"], droop=build_values["droop"] + c["droop"],
                    tip_radius=_tip(build_values), lift=c["lift"], sway=c["sway"])

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


# the horn shapes: parameters per variant (the first row is the first version's horn).
# Every variant is make_horn with other numbers; ``turns`` makes it a helix.
HORN_SHAPES = {
    #            length base_r  curve  twist  turns  segments ring_verts
    "curved":   (2.2,   0.42,   1.1,   0.0,   0.0,   8,       8),     # short, curved: the default
    "ram":      (4.2,   0.55,   2.1,   0.6,   0.0,  16,      12),     # thick, curling back
    "spiral":   (4.5,   0.34,   1.2,   0.0,   3.0,  36,      12),     # a slender corkscrew
    "antelope": (5.2,   0.28,   0.7,   0.0,   1.0,  28,      12),     # long, slim, a lyre
}


def _horn_param_specs() -> tuple:
    out = [ParamSpec("shape", "Shape", "choice", "curved", choices=tuple(HORN_SHAPES),
                     help="curved | ram | spiral | antelope: sets the defaults below")]
    for key, label, col, lo, hi, step, typ in (
            ("length", "Length", 0, 0.3, 12.0, 0.1, "float"),
            ("base_radius", "Base radius", 1, 0.08, 1.5, 0.01, "float"),
            ("segments", "Segments", 5, 3, 40, 1, "int"),
            ("ring_verts", "Ring vertices", 6, 5, 24, 1, "int"),
            ("curve", "Curve", 2, -3.0, 3.0, 0.01, "float"),
            ("twist", "Twist", 3, -3.0, 3.0, 0.01, "float"),
            ("turns", "Turns", 4, -4.0, 4.0, 0.05, "float")):
        for shape, row in HORN_SHAPES.items():
            out.append(ParamSpec(key, label, typ, row[col], lo, hi, step, when=("shape", (shape,))))
    out.append(ParamSpec("tip_radius", "Tip radius", "float", HORN_TIP, 0.0, 0.3, 0.005,
                         help="radius at the end, which is rounded; 0 = the first version's "
                              "point (0.04 mm across at 32 mm)"))
    return tuple(out)


class HornKind(_LoftKind):
    """A horn: ``shape`` picks a variant (curved, ram, spiral, antelope), each the same
    tapered tube swept along a bend; ``tip_radius`` rounds the end to a printable size.
    No pose controls (a horn does not move)."""

    id = "horn"
    label = "Horn"
    default_side = "both"
    paired = True
    params = _horn_param_specs()
    presets = {"temples": {"label": "Temples (up and out on the skull)", "face": "temples",
                           "up_hint": (0.0, 1.0, 0.0), "offset": HORN_SINK, "side": "both"}}

    def load(self, params):
        return float(params.get("length", 2.2)) / 2.2

    def material(self, params):
        return material_named("bone")

    def builder_args(self, build_values, controls):
        return dict(length=build_values["length"], base_radius=build_values["base_radius"],
                    segments=build_values["segments"], ring_verts=build_values["ring_verts"],
                    curve=build_values["curve"], twist=build_values["twist"],
                    turns=build_values["turns"], tip_radius=_tip(build_values))


# -- the wing ----------------------------------------------------------------------------

_MEMBRANE_STYLES = ("membrane", "dragon")
# the DRAGON style differs from the bat in these (wings.DRAGON); a test pins the defaults
# below to what the builders really use
_WING_POSE_LABELS = (
    ("fold", "Fold", 0.0), ("raise_", "Raise", 0.0), ("swing", "Swing", 0.0),
    ("twist", "Twist", 0.0), ("fan", "Fan", 1.0), ("curl", "Curl", 0.0),
    ("wrist_flex", "Wrist flex", 0.0), ("hand_twist", "Hand twist", 0.0))


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
    )
    presets = {"shoulder_blades": {
        "label": "Shoulder blades", "face": "shoulder_blades", "up_hint": A.WING_UP_HINT,
        "roll": A.WING_ROLL, "offset": A.WING_OFFSET, "side": "both"}}
    placement_keys = ("anchor", "scale", "roll", "offset", "up_hint", "aim")

    def load(self, params):
        from .wings import DEFAULT_SPAN
        return float(params.get("span", DEFAULT_SPAN)) / DEFAULT_SPAN

    def draft_params(self, values):
        if values.get("style") == "feathered":
            return {"coverts": False, "underwing": False, "bones": False}
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
        app = self._engine(side, pl, params=dict(build_values),
                           **self._aim(values["style"], pl["aim"], ctx))
        names = {i: (n, s) for i, n, s in self._names(spec, app._indices())}
        out = []
        for idx, pw in app.place_rigs(ctx.base, ctx.verts):
            name, letter = names[idx]
            ctrl, notes = self.clean_controls(side_controls(controls_raw, letter),
                                              f"appendage {spec.id!r}: ")
            for note in notes:
                ctx.warn(note)
            out.append(Placed(name=name, spec_id=spec.id, kind=self, side=letter,
                              anchor=self._surface(ctx, app, idx), matrix=pw.matrix,
                              params=dict(values), controls=ctrl, rig=pw))
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
    )
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
