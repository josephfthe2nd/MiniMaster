"""A real rig for wings: joints, skin weights and poses.

``wings.py`` lays out a wing's forelimb and hangs a surface on it. This module
turns that into something you can *pose*: the skeleton becomes a
:class:`~minimaster.core.armature.Armature` of named joints, every vertex of
the wing is bound to those joints, and folding, raising or flapping the wing is
a rotation of joints, not a rebuild of the mesh.

Binding follows how each kind of piece actually moves:

* **rigid pieces** -- a feather, the thumb claw, a joint ball -- are welded to
  one bone with weight 1. A feather is stiff; smearing it across two bones
  shears the shingled plates apart.
* **chain pieces** -- a finger tube -- blend only over the bones of their own
  chain, by proximity, so a finger bends smoothly at its knuckles with no seam
  and is never pulled by the finger next door.
* **the membrane** blends over the arm and finger bones by distance in the
  wing's plane. A vertex hugging a finger follows that finger; one mid-way
  between two fingers follows both, which is exactly how a patagium stretches
  and slackens as the fan opens and closes. The root edge is bound to a bone
  that belongs to the *body*, so the sheet stays glued to the flank however the
  arm moves.

Skinning is plain linear blend (the same convention as
:func:`minimaster.core.bodymesh.pose_body_mesh`) so the topology never changes:
a shell that is watertight at rest is watertight in every pose.

Posing semantics are those of :class:`Armature`: the rotation stored at a joint
spins that joint's outgoing bones about the joint. The wing is authored in its
own frame (``+X`` span, ``+Y`` down, ``+Z`` trailing edge); poses are expressed
there, and :class:`PlacedWing` carries the placement transform that puts the
posed mesh on the body.

A host that has its own body armature can drive the wing instead of re-meshing it:
:meth:`PlacedWing.attach` adds the wing's joints to that armature,
:meth:`PlacedWing.body_pose` turns a wing-frame pose into rotations of those joints (their
axes are the body's, so each rotation is conjugated by the placement) and
:meth:`PlacedWing.skin_weights` exports the weights by bone name. The host's own skinning
then reproduces the finished mesh exactly for every piece but the membrane slab, whose
pleats and stretch limiter are a post-process on the vertices (:class:`SkinWeights`
lists the pieces concerned; ``docs/wings.md`` has the measured limit).

Input is validated: an unknown control name, a control that is not a finite number, an
unknown preset, a joint the wing does not have, are errors -- not a silent rest pose or a
NaN mesh.
"""

from __future__ import annotations

import difflib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace

import numpy as np

from ..core import math3d as m3
from ..core.armature import Armature
from ..core.mesh import Mesh


# --------------------------------------------------------------------------
# pieces: a mesh plus how it is bound


@dataclass
class Piece:
    """One closed solid of a wing and the rule that binds it to the rig."""

    name: str
    mesh: Mesh
    bone: str | None = None            # rigid to this bone (child-joint key)
    chain: tuple[str, ...] | None = None   # smooth over just these bones
    membrane: bool = False             # distance-weighted over the arm + fingers
    # Slack bookkeeping for a membrane slab (see WingRig._apply_pleats): the
    # slab's (rows, columns) grid shape and the column ranges of each panel
    # between two bone lines. Optional; without it the sheet is just skinned.
    pleat: dict | None = None


# --------------------------------------------------------------------------
# pose vocabulary
#
# Convention, fixed by core.math3d (R = Rz Ry Rx, right-handed):
#   rot_y(+t) takes +X toward -Z   -> swinging a bone BACK toward the trailing
#                                     edge (+Z) is a NEGATIVE ry
#   rot_z(+t) takes +X toward +Y   -> +Y is DOWN, so RAISING a bone is a
#                                     negative rz
#   rot_x(+t) takes +Z toward -Y   -> lifts the trailing edge (leading edge
#                                     dips): positive rx is "twist trailing-up"


def swing_back(deg: float) -> tuple[float, float, float]:
    """Rotate a joint so its bones swing back toward the trailing edge."""
    return (0.0, -float(deg), 0.0)


def raise_(deg: float) -> tuple[float, float, float]:
    """Rotate a joint so its bones lift toward the wing's upper surface."""
    return (0.0, 0.0, -float(deg))


def twist(deg: float) -> tuple[float, float, float]:
    """Rotate about the span axis; positive lifts the trailing edge."""
    return (float(deg), 0.0, 0.0)


def compose(*eulers) -> tuple[float, float, float]:
    """Sum small euler offsets applied to the same joint."""
    return tuple(float(sum(e[i] for e in eulers)) for i in range(3))


@dataclass
class PoseControls:
    """The few numbers an animator actually turns.

    ``fold``   0 spread .. 1 stowed. Closes the elbow and wrist and brings the
               fingers together, the way a real wing folds.
    ``raise_`` degrees the whole arm is lifted (flap), about the wing frame's chord axis Z
               on a free wing and about the BODY's fore-aft axis once the wing is placed
               on a body (``WingSkeleton.place_on_body``): + lifts the tip, on either side.
    ``swing``  degrees the arm is swung back (+) or forward (-): in the wing's plane on a
               free wing, about the body's vertical axis on a placed one.
    ``twist``  degrees about the span axis, trailing edge up (+).
    ``fan``    0 fingers together .. 1 rest spread (independent of ``fold``).
    ``curl``   degrees every finger bends at its knuckles, toward the underside
               (+) -- the downstroke's cupped hand.
    ``wrist_flex``  degrees the hand is cocked out of the wing plane at the wrist,
               tip up (+) or drooping (-). In a real flap the wrist leads and the
               hand trails: on the upstroke it hangs below the wrist.
    ``hand_twist``  degrees the hand rolls about its own axis (wash-out / pronation).
    """

    fold: float = 0.0
    raise_: float = 0.0
    swing: float = 0.0
    twist: float = 0.0
    fan: float = 1.0
    curl: float = 0.0
    wrist_flex: float = 0.0
    hand_twist: float = 0.0

    def __setattr__(self, name, value):
        # A misspelt control (``flod=1``) must not become a stray attribute that is
        # silently ignored, and NaN/inf must not reach ``np.clip`` (it passes NaN
        # through and the mesh comes out NaN). Every way of setting a control --
        # the constructor, ``flap(..., **kw)``, ``pose(**kw)``, plain assignment --
        # goes through here.
        _check_control_name(name)
        object.__setattr__(self, name, _finite_control(name, value))

    @classmethod
    def flap(cls, phase_deg: float, amp: float = 45.0, up_reach: float | None = None,
             down_reach: float | None = None, **kw) -> "PoseControls":
        """One instant of a flap cycle. ``phase_deg`` 0 = mid-stroke going up,
        90 = top of the upstroke, 270 = bottom of the downstroke. The shoulder
        leads; the wrist and the finger curl lag it (40 and 70 degrees of phase),
        so the hand trails the wrist -- hanging below it when the wing is raised --
        instead of the whole wing swinging like a plank.

        ``amp`` is the shoulder's swing either side of rest. ``up_reach`` / ``down_reach``
        (default ``amp``) size the two halves separately, for a wing whose range is lopsided
        (a placed wing rests ~50 degrees above horizontal, so it has far more room to drop than
        to rise; ``WingRig.flap`` / ``PlacedWing.flap`` fill them in from the wing's own range)
        -- asked for more than the range allows, the top of the stroke would sit clamped."""
        ph = np.radians(phase_deg)
        up = max(0.0, float(np.sin(ph)))                        # 0 on the downstroke
        reach = amp if (up_reach if up else down_reach) is None else (up_reach if up else down_reach)
        c = cls(raise_=reach * float(np.sin(ph)),
                wrist_flex=-0.6 * amp * float(np.sin(ph - np.radians(40.0))),
                curl=0.35 * amp * float(np.sin(ph - np.radians(70.0))) + 0.3 * amp * 0.35,
                # the upstroke is not the downstroke played backwards: the arm
                # half-folds and the fingers close to shed the air, then the arm
                # extends and the hand pronates for the power stroke
                fold=0.30 * max(0.0, float(np.sin(ph + np.radians(20.0)))),
                fan=1.0 - 0.30 * up,
                twist=-15.0 * float(np.sin(ph + np.radians(90.0))),
                swing=8.0 * float(np.cos(ph)))
        for k, v in kw.items():
            setattr(c, k, v)
        return c


def _control_names() -> tuple[str, ...]:
    return tuple(f.name for f in fields(PoseControls))


def _check_control_name(name: str) -> None:
    known = _control_names()
    if name not in known:
        near = difflib.get_close_matches(str(name), known, n=1)
        raise TypeError(f"unknown pose control {name!r}"
                        + (f" (did you mean {near[0]!r}?)" if near else "")
                        + f"; the controls are {', '.join(known)}")


def _finite_control(name: str, value):
    """``value`` if it is a finite real number, else a clear error: TypeError for
    something that is not a number at all, ValueError for NaN / infinity."""
    if value is None or isinstance(value, (str, bytes)):
        raise TypeError(f"pose control {name!r} must be a number, got {value!r}")
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"pose control {name!r} must be a number, got {value!r}") from None
    if not math.isfinite(x):
        raise ValueError(f"pose control {name!r} must be finite, got {value!r}")
    return value


# Anatomical ranges of the controls, from skeleton and flight photographs (and
# the literature where a photo cannot show it): a wing cannot be wrenched past
# what the joints allow, so the controls are clamped to these.
#
# These are what the JOINTS allow. A skeleton narrows them to what its skin takes (a membrane's
# stretch) and, once placed, what the body leaves (the spine, the hip): ``WingSkeleton.limits`` /
# ``placed_limits``, applied by ``clamp_controls`` / ``control_range``.
LIMITS = {
    "fold": (0.0, 1.0),
    "raise_": (-70.0, 100.0),         # shoulder elevation: hanging .. wrists-over-back
    "swing": (-30.0, 85.0),           # humerus protraction .. stowed along the spine
    "twist": (-45.0, 45.0),           # roll about the span axis: leading edge down .. up
    "fan": (0.0, 1.5),
    "curl": (-10.0, 150.0),           # knuckle flexion
    "wrist_flex": (-60.0, 45.0),      # hand drooping .. cocked up
    "hand_twist": (-30.0, 30.0),
}

# A control is an offset on top of what ``fold`` already does, so its range is
# what the joint has LEFT, not a constant: a wing stowed along the spine has used
# up the humerus's swing (fold 1 + swing 85 once asked for 195 degrees at the
# shoulder, and half the membrane ended up on the far side of the body), and its
# wrist is already over the back.
FOLD_LIMITS = {
    "swing": lambda f: 85.0 - 80.0 * f,       # LIMITS["swing"] hi at fold 0, 5 degrees left stowed
    "raise_": lambda f: 100.0 - 75.0 * f,     # LIMITS["raise_"] hi at fold 0, 25 degrees left stowed
}
# ... and the other side: a stowed wing already lies on the back, so there is nothing left to
# press it into (fold 1 with swing -30 put 38% of a bat inside the torso, raise_ -70 16% of an angel).
FOLD_FLOORS = {
    "swing": lambda f: -30.0 * (1.0 - f),
    "raise_": lambda f: -70.0 * (1.0 - f),
}

# Named poses over the controls. A flap cycle is up -> down -> up: the upstroke
# lifts the wrist and half-folds the arm with the leading edge turned edge-on;
# the downstroke sweeps through level with the arm extended, leading edge down.
PRESETS = {
    "glide": {},
    "half_fold": {"fold": 0.5},
    "folded": {"fold": 1.0},
    "upstroke": {"raise_": 25.0, "fold": 0.35, "twist": -8.0, "fan": 0.95, "wrist_flex": -15.0},
    "downstroke": {"raise_": -20.0, "twist": 5.0, "curl": 12.0, "wrist_flex": 20.0},
    "mid_downstroke": {"raise_": 5.0, "twist": 10.0, "wrist_flex": 8.0},
}


def _bound(b, fold: float) -> float:
    """A range bound: a number, or a function of ``fold``."""
    return float(b(fold)) if callable(b) else float(b)


def control_range(name: str, fold: float = 0.0, fan_range=None, limits=None) -> tuple[float, float]:
    """``(lo, hi)`` the control ``name`` may take at ``fold``: the anatomical ``LIMITS``, the
    shoulder ranges that shrink as the wing stows (``FOLD_LIMITS`` above, ``FOLD_FLOORS`` below),
    a skeleton's own ``fan_range`` and ``limits`` (``{control: (lo, hi)}``, a bound may be a
    function of ``fold``), which can only narrow."""
    f = float(np.clip(fold, 0.0, 1.0))
    lo, hi = LIMITS[name]
    if name == "fan" and fan_range is not None:
        lo, hi = fan_range
    if name in FOLD_LIMITS:
        hi = FOLD_LIMITS[name](f)
    if name in FOLD_FLOORS:
        lo = FOLD_FLOORS[name](f)
    own = (limits or {}).get(name)
    if own is not None:
        lo, hi = max(lo, _bound(own[0], f)), min(hi, _bound(own[1], f))
    return float(lo) + 0.0, float(max(lo, hi)) + 0.0          # (+ 0.0: no -0.0 from -30 * 0)


def clamp_controls(c: "PoseControls", fan_range=None, limits=None) -> "PoseControls":
    """A copy of the controls limited to what the joints can do (the caller's
    object is left alone). ``fan_range`` narrows ``fan`` for a skeleton whose
    skin cannot take the full range (a membrane tears long before a feather).
    ``limits`` is a skeleton's own ``{control: (lo, hi)}`` (bounds may be functions
    of ``fold``): it is intersected with the anatomical ranges, so it can only narrow."""
    out = replace(c)
    for name in LIMITS:
        lo, hi = control_range(name, c.fold, fan_range, limits)
        setattr(out, name, float(np.clip(getattr(c, name), lo, hi)))
    return out


def grid_normals(grid: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """Per-vertex unit normals of an (n+1, m+1, 3) surface grid.

    The cross product of the two grid tangents is only trustworthy while those
    tangents are well apart. A scalloped patagium leaves each fingertip along
    a line nearly parallel to the grid's own rays, so ``du x dv`` collapses
    there and its direction (even its sign) is noise: on the dragon wing 205
    vertices were worse than 20 degrees from degenerate and one column of the
    slab came out with its thickness lying *in* the sheet. Where the tangents
    are close to parallel the normal is therefore taken from ``ref`` instead
    (default: the sheet's mean normal, made perpendicular to ``du``) and the
    two are blended smoothly. Well-conditioned vertices are untouched.
    """
    du = np.gradient(grid, axis=0)
    dv = np.gradient(grid, axis=1)
    cr = np.cross(du, dv)
    ln = np.linalg.norm(cr, axis=2, keepdims=True)
    nrm = cr / np.maximum(ln, 1e-12)
    sin = ln[..., 0] / np.maximum(np.linalg.norm(du, axis=2) * np.linalg.norm(dv, axis=2), 1e-12)
    t = np.clip((sin - 0.10) / 0.30, 0.0, 1.0)
    w = t * t * (3.0 - 2.0 * t)
    if bool((w >= 1.0).all()):
        return nrm
    if ref is None:
        good = (w > 0.99)[..., None]
        mean = (nrm * good).sum(axis=(0, 1))
        if float(np.linalg.norm(mean)) < 1e-9:
            return nrm
        ref = np.broadcast_to(mean / np.linalg.norm(mean), nrm.shape)
    ref = np.asarray(ref, dtype=np.float64)
    tu = du / np.maximum(np.linalg.norm(du, axis=2, keepdims=True), 1e-12)
    ref = ref - np.einsum("ijk,ijk->ij", ref, tu)[..., None] * tu     # perpendicular to the rays
    ref = ref / np.maximum(np.linalg.norm(ref, axis=2, keepdims=True), 1e-12)
    flip = np.where(np.einsum("ijk,ijk->ij", nrm, ref) < 0.0, -1.0, 1.0)[..., None]
    # a conditioned vertex keeps its own sign; only an unreliable one is flipped to agree
    out = np.where((w >= 1.0)[..., None], nrm, w[..., None] * (nrm * flip) + (1.0 - w[..., None]) * ref)
    return out / np.maximum(np.linalg.norm(out, axis=2, keepdims=True), 1e-12)


def _slab_normals(grid: np.ndarray, hint: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Thickness directions for a posed sheet: each vertex's normal is the sum of the
    *area vectors* of the cells around it, plus ``weight`` x the panel's ``hint``.

    A cell's area vector is its oriented area times its normal, so it always follows
    the grid's own orientation -- a surface that doubles back keeps its thickness on the
    side it was built on and the solid stays valid (signed volume positive) -- and a
    collapsing cell, whose direction is noise, counts for nothing because it is small. Only
    where every cell around a vertex has collapsed does the hint (the panel's normal, see
    ``_frame``) decide, through the small ``weight``, which makes the result a continuous
    function of the positions: no sign is ever picked by comparing against a reference,
    which is what made the two faces of the slab swap places at a vertex (a one-thickness
    jump, and a bow-tie in the shell) when a stowed sheet was nearly edge-on."""
    p00, p01 = grid[:-1, :-1], grid[:-1, 1:]
    p10, p11 = grid[1:, :-1], grid[1:, 1:]
    area = 0.5 * np.cross(p11 - p00, p01 - p10)          # oriented area vector du x dv
    tot = np.zeros_like(grid)
    tot[:-1, :-1] += area
    tot[:-1, 1:] += area
    tot[1:, :-1] += area
    tot[1:, 1:] += area
    return _unit(tot + weight[..., None] * hint)


# How far a membrane panel can close before it has to buckle into a pleat.
ELASTIC = 0.90
# How far skin may stretch (relative to its rest length) before the limiter
# pulls it back. Compression is free -- that is what the pleats are for.
MAX_STRETCH = 1.15
# Gauss-Seidel sweeps of the limiter (each visits every edge once, in four colour classes).
STRETCH_ITERS = 24
# Skin takes up a little more closure, and buckles a little less, while the wing is only
# cupped (the hand not yet folded back): ELASTIC_CUPPED is knocked off ELASTIC and the pleat
# gain is cut by PLEAT_CUPPED, both fading out by the time the hand has folded PLEAT_FOLD_FULL
# of the way back onto the forearm.
ELASTIC_CUPPED = 0.08
PLEAT_CUPPED = 0.5
PLEAT_FOLD_FULL = 0.3
# A finger's column of the sheet is laid along the finger (``WingRig._gather_fingers``)
# once one of its knuckles is bent this many degrees (a rest finger's hooked tip is bent
# 47, a curl of 40 degrees 67), and along the straight line between its ends once it has
# folded up (165 degrees stowed). Below that the column keeps its skinned arc.
BEND_FROM, BEND_SPAN = 75.0, 30.0       # sheet column follows the bent finger
FOLD_FROM, FOLD_SPAN = 120.0, 45.0      # ... then the straight line between its ends
# Tallest a single pleat may stand off the sheet (soft limit, wing units).
PLEAT_CAP = 0.30
# A pleat grows in over this much further closure of its panel, past what the skin takes
# up elastically (ELASTIC): a smooth onset, not a switch.
PLEAT_RAMP = 0.15
# No pleat lobe stands taller than this fraction of its own base.
PLEAT_SLOPE = 4.0
# How much the panel's own normal counts toward a slab vertex's thickness direction, as a
# fraction of (the sheet's thickness)^2: the area of the cells around a vertex that is
# small enough for the hint to matter.
SLAB_HINT = 0.35

# --------------------------------------------------------------------------
# the rig


def _smooth01(x: float) -> float:
    t = min(max(float(x), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _smooth01_arr(x):
    t = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _frame(chord, ray, up_hint, orient=None, cupped=1.0):
    """Orthonormal (chord, normal, side) frames for rows of a panel.

    ``chord`` joins a row's two bounding fingers, ``ray`` runs along the panel (outward
    from the arm) and ``up_hint`` is the slab's up vector there. The panel's normal is
    found two ways, each weighted by how well defined it is (its length): ``chord x ray``,
    the oriented normal of the flat quad, and the up vector made perpendicular to the
    chord. At rest they agree. A stowed panel standing on edge between two rods stacked
    one above the other has a vertical chord, so the second is then noise (and reversed
    on the far side of vertical) while the first turns smoothly with the panel; in the
    sliver along digit III chord and ray are nearly parallel and it is the first that is
    noise. ``orient`` (+-1 per row) fixes the side of ``chord x ray``: chosen at rest from
    the up vector and carried into the pose unchanged, so the normal never flips.

    A curl turns a finger's tip about the chord, and the up vector turns with it while the
    ray (taken from the metacarpals) does not; so while the wing is only cupped (``cupped``
    1, see ``WingRig._hand_fold``) the up axis leads, and once it is folded (0) the quad's
    normal does."""
    e1 = _unit(chord)
    a = np.cross(e1, _unit(ray))
    u = _unit(up_hint)
    b = u - np.einsum("ik,ik->i", u, e1)[:, None] * e1
    if orient is None:
        orient = np.where(np.einsum("ik,ik->i", a, b) < 0.0, -1.0, 1.0)
        if float(np.linalg.norm(a, axis=1).mean()) < 0.7:        # the sliver along digit III
            orient = np.zeros(len(chord))
    if not orient.any():                                         # (0 marks it: up axis only)
        e2 = _unit(b + 1e-12 * u)
    else:
        e2 = _unit(a * orient[:, None] * (1.0 - 0.75 * cupped) + b * (0.25 + 0.75 * cupped) + 1e-12 * u)
    return e1, e2, np.cross(e1, e2), orient


def _plan_segment_distance(pts_xz, a_xz, b_xz):
    ab = b_xz - a_xz
    denom = float(ab @ ab)
    if denom < 1e-12:
        return np.linalg.norm(pts_xz - a_xz, axis=1)
    t = np.clip((pts_xz - a_xz) @ ab / denom, 0.0, 1.0)
    return np.linalg.norm(pts_xz - (a_xz + t[:, None] * ab), axis=1)


def _segment_distance(pts, a, b):
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return np.linalg.norm(pts - a, axis=1)
    t = np.clip((pts - a) @ ab / denom, 0.0, 1.0)
    return np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)


def _top_k_normalised(w: np.ndarray, k: int) -> np.ndarray:
    """Keep each row's ``k`` largest weights, renormalise to sum to one."""
    if w.shape[1] > k:
        cut = np.sort(w, axis=1)[:, -k][:, None]
        w = np.where(w >= cut, w, 0.0)
    tot = w.sum(axis=1, keepdims=True)
    return w / np.maximum(tot, 1e-30)


class WingRig:
    """A wing skeleton + the pieces bound to it.

    ``skeleton`` provides ``joints`` (name -> rest position), ``parent`` (name ->
    parent name or None) and ``membrane_bones`` (bone keys the sheet may blend
    over). ``pieces`` is the list of :class:`Piece` that make up the wing.
    """

    def __init__(self, skeleton, pieces: list[Piece], influences: int = 4,
                 membrane_power: float = 4.0, membrane_eps: float = 0.06):
        self.skeleton = skeleton
        self.pieces = list(pieces)
        self.armature = Armature()
        for name, pos in skeleton.joints.items():
            self.armature.add_joint(name, pos, skeleton.parent[name])
        self._bone_keys = [c for _, c in self.armature.bones()]
        self._bone_index = {k: i for i, k in enumerate(self._bone_keys)}

        meshes = [p.mesh for p in self.pieces]
        self.rest_mesh = Mesh.merge(meshes)
        self.ranges: dict[str, tuple[int, int]] = {}
        start = 0
        for p in self.pieces:
            self.ranges[p.name] = (start, start + len(p.mesh.vertices))
            start += len(p.mesh.vertices)

        # Per-vertex influences: (bone index, weight) arrays of width K.
        n = len(self.rest_mesh.vertices)
        self._idx = np.zeros((n, influences), dtype=np.int64)
        self._wt = np.zeros((n, influences), dtype=np.float64)
        self._wt[:, 0] = 1.0
        self._membrane_power = membrane_power
        self._membrane_eps = membrane_eps
        self._K = influences
        for p in self.pieces:
            a, b = self.ranges[p.name]
            self._bind(p, a, b)
        self._plan_bone_cache = None

    # -- binding ----------------------------------------------------------

    def _bone_segments(self, keys):
        out = []
        for k in keys:
            j = self.armature.joints[k]
            out.append((self.armature.joints[j.parent].position, j.position))
        return out

    def _bind(self, piece: Piece, a: int, b: int) -> None:
        V = self.rest_mesh.vertices[a:b]
        if piece.bone is not None:
            self._idx[a:b, 0] = self._bone_index[piece.bone]
            self._wt[a:b, :] = 0.0
            self._wt[a:b, 0] = 1.0
            return
        if piece.chain is not None:
            keys = list(piece.chain)
            segs = self._bone_segments(keys)
            d = np.stack([_segment_distance(V, s0, s1) for s0, s1 in segs], axis=1)
            w = 1.0 / (d + 0.08 * self._scale_hint()) ** 4
        elif piece.membrane:
            keys = list(self.skeleton.membrane_bones)
            segs = self._bone_segments(keys)
            xz = V[:, [0, 2]]
            d = np.stack([_plan_segment_distance(xz, s0[[0, 2]], s1[[0, 2]])
                          for s0, s1 in segs], axis=1)
            w = 1.0 / (d + self._membrane_eps * self._scale_hint()) ** self._membrane_power
        else:
            raise ValueError(f"piece {piece.name!r} has no binding rule")
        w = _top_k_normalised(w, min(self._K, w.shape[1]))
        order = np.argsort(-w, axis=1)[:, : self._K]
        top = np.take_along_axis(w, order, axis=1)
        top = top / np.maximum(top.sum(axis=1, keepdims=True), 1e-30)   # ties may have cut one
        gidx = np.array([self._bone_index[k] for k in keys])[order]
        self._idx[a:b, : order.shape[1]] = gidx
        self._wt[a:b, :] = 0.0
        self._wt[a:b, : order.shape[1]] = top

    def _scale_hint(self) -> float:
        return float(getattr(self.skeleton, "scale", 1.0))

    # -- skinning ---------------------------------------------------------

    def check_pose(self, pose) -> None:
        """Raise for a raw pose dict (joint -> euler degrees) the rig cannot take: a
        joint the wing does not have (``KeyError``, with a "did you mean": a misspelt
        joint used to be ignored and the wing left at rest) or a rotation that is not
        three finite numbers (``ValueError``)."""
        if not pose:
            return
        for name, e in pose.items():
            if name not in self.armature.joints:
                near = difflib.get_close_matches(str(name), list(self.armature.joints), n=1)
                raise KeyError(f"the wing has no joint {name!r}"
                               + (f" (did you mean {near[0]!r}?)" if near else ""))
            try:
                a = np.asarray(e, dtype=np.float64)
            except (TypeError, ValueError):
                a = None
            if a is None or a.shape != (3,) or not np.isfinite(a).all():
                raise ValueError(f"the rotation of joint {name!r} must be three finite "
                                 f"euler angles in degrees, got {e!r}")

    def skin_matrices(self, pose=None) -> dict[str, np.ndarray]:
        self.check_pose(pose)
        return self.armature.skin_matrices(pose)

    def pose_mesh(self, pose=None) -> Mesh:
        """The wing under ``pose`` (joint name -> euler degrees), wing frame."""
        pose = pose or {}
        if not pose:
            return self.rest_mesh.copy()
        skins = self.skin_matrices(pose)
        R = np.stack([skins[k][:3, :3] for k in self._bone_keys])      # (B,3,3)
        t = np.stack([skins[k][:3, 3] for k in self._bone_keys])       # (B,3)
        V = self.rest_mesh.vertices
        out = np.zeros_like(V)
        for slot in range(self._K):
            w = self._wt[:, slot]
            act = w > 0.0
            if not act.any():
                continue
            bi = self._idx[act, slot]
            Ra = R[bi]
            moved = np.einsum("vij,vj->vi", Ra, V[act]) + t[bi]
            out[act] += w[act, None] * moved
        out = self._apply_pleats(out, pose)
        return Mesh(out, self.rest_mesh.faces)

    def _apply_pleats(self, posed: np.ndarray, pose=None, gain: float = 0.8) -> np.ndarray:
        """Re-evaluate each membrane panel between its two posed bone lines.

        Skinning moves the vertices that sit on a finger with that finger, but
        everything between two fingers is left wherever blended transforms drop
        it: the blend of two bones' rotations applied to the chord between them
        shortens, twists and stretches it (``s(1-s)(R_A - R_B)(b - a)``), and when
        the fingers close the sheet simply crumples. A real patagium is a
        continuous, inextensible skin stretched between bones: for every row of a
        panel, the posed points on its two bounding fingers define a chord, and the
        interior is that chord plus the panel's own rest-pose curvature (the
        scallop sag and billow, scaled by how far the gap has closed). That
        re-evaluation is applied **at every pose**, so there is no threshold at
        which the interior snaps from the skinned sheet to the chord-based one;
        it equals the rest mesh at rest and varies smoothly with the controls.
        What the sheet cannot lose -- the slack between the rest gap ``L`` and the
        posed gap ``g`` -- buckles into a pleat: a tent of slant ``L/2`` over a
        base ``g``, height sqrt(L^2 - g^2)/2, standing along the panel's own normal
        (``_frame``: ``chord x ray``, and the finger vertices' up axis, which
        skinning keeps stable). Skin is elastic, so a pleat only grows once the
        gap has closed by a few percent, and it comes in over ``PLEAT_RAMP`` of
        further closure and never stands taller than ``PLEAT_SLOPE`` times its
        base.
        Neighbouring panels pleat in opposite directions, the way a closing fan
        zig-zags.
        """
        specs = [(p, self.ranges[p.name]) for p in self.pieces if p.pleat]
        if not specs:
            return posed
        # A wing that is cupped (curled, flapping) but not folded keeps its skin taut: the
        # fingers lean in together, the sheet between them stretches a little more and
        # buckles a little less, and it never gathers into the pleats of a closing wing.
        cupped = 1.0 - _smooth01(self._hand_fold(pose) / PLEAT_FOLD_FULL) if pose else 1.0
        elastic = ELASTIC - ELASTIC_CUPPED * cupped
        gain = gain * (1.0 - PLEAT_CUPPED * cupped)
        for piece, (a, b) in specs:
            n1, m1 = piece.pleat["shape"]
            gains = piece.pleat.get("gains", [1.0] * len(piece.pleat["sectors"]))
            lobes = piece.pleat.get("lobes", [1] * len(piece.pleat["sectors"]))
            nv = n1 * m1
            rest = self.rest_mesh.vertices[a:b]
            pos = posed[a:b]
            r_mid = 0.5 * (rest[:nv] + rest[nv:2 * nv]).reshape(n1, m1, 3)
            r_half = 0.5 * np.linalg.norm(rest[:nv] - rest[nv:2 * nv], axis=1).reshape(n1, m1)
            r_up = _unit((rest[:nv] - rest[nv:2 * nv]).reshape(n1, m1, 3))
            top = pos[:nv].reshape(n1, m1, 3)
            bot = pos[nv:2 * nv].reshape(n1, m1, 3)
            skinned_mid = 0.5 * (top + bot)
            up = _unit(top - bot)                           # stable at the finger columns
            mid = skinned_mid
            joints = self.armature.posed_positions(pose) if piece.pleat.get("fingers") else None
            if piece.pleat.get("fingers") and pose:
                mid = self._gather_fingers(mid, r_mid, piece.pleat, pose)
            new_mid = mid.copy()
            frames = []
            limit_v = np.full((n1, m1 - 1), MAX_STRETCH)
            for k, (c0, c1) in enumerate(piece.pleat["sectors"]):
                g_rest = np.linalg.norm(r_mid[:, c1] - r_mid[:, c0], axis=1)
                chord_r = r_mid[:, c1] - r_mid[:, c0]
                chord_p = mid[:, c1] - mid[:, c0]
                g_pose = np.linalg.norm(chord_p, axis=1)
                ratio = g_pose / np.maximum(g_rest, 1e-12)
                # where each interior column sits across the panel at rest
                seg = np.linalg.norm(np.diff(r_mid[:, c0:c1 + 1], axis=1), axis=2)
                s_ij = np.concatenate([np.zeros((n1, 1)), np.cumsum(seg, axis=1)], axis=1)
                s_ij = s_ij / np.maximum(s_ij[:, -1:], 1e-12)
                # the panel's ray, from the arm out to the free edge: the mean direction of its
                # two bounding fingers' metacarpals (the first bone of a finger is only ever
                # turned by the fan, never folded back on itself, so it keeps pointing outward;
                # a column's local tangent, or its end-to-end vector, doubles back and passes
                # through zero in the middle of the concertina). The last panel's far edge is
                # the root line: its end-to-end vector, which is a straight run on the body.
                ray_r = self._panel_ray(k, piece.pleat, r_mid, self.skeleton.joints)
                ray_p = self._panel_ray(k, piece.pleat, mid, joints)
                ray_r = np.broadcast_to(ray_r, chord_r.shape)
                ray_p = np.broadcast_to(ray_p, chord_p.shape)
                e1r, e2r, e3r, side = _frame(chord_r, ray_r, r_up[:, c0] + r_up[:, c1])
                e1p, e2p, e3p, _ = _frame(chord_p, ray_p, up[:, c0] + up[:, c1], side, cupped)
                ruled_r = r_mid[:, c0][:, None, :] + chord_r[:, None, :] * s_ij[..., None]
                d_rest = r_mid[:, c0:c1 + 1] - ruled_r                    # curvature off the chord
                # carry the offset into the posed frame: components along the chord, the
                # wing-up axis and their cross product (an orthonormal frame at rest and
                # posed, so the rest pose is reproduced exactly)
                c1_ = np.einsum("ijk,ik->ij", d_rest, e1r)
                c2_ = np.einsum("ijk,ik->ij", d_rest, e2r)
                c3_ = np.einsum("ijk,ik->ij", d_rest, e3r)
                flat = np.minimum(ratio, 1.0)[:, None]     # the bulge flattens as the gap closes
                d_pose = (c1_[..., None] * e1p[:, None, :] + c2_[..., None] * e2p[:, None, :]
                          + c3_[..., None] * e3p[:, None, :]) * flat[..., None]
                ruled_p = mid[:, c0][:, None, :] + chord_p[:, None, :] * s_ij[..., None]
                # skin is elastic: it takes up a few percent of closure by itself and only
                # buckles beyond that, coming in smoothly over PLEAT_RAMP more of closure
                slack = np.maximum((elastic * g_rest) ** 2 - g_pose ** 2, 0.0)
                fade = _smooth01_arr((elastic - ratio) / PLEAT_RAMP)
                h = (gain * gains[k] * 0.5 * np.sqrt(slack) * fade
                     * (1.0 if k % 2 == 0 else -1.0))
                # k fine pleats instead of one bladder: each is a tent of slant L/2k
                # over a base g/k, so k lobes of height h/k
                kl = float(lobes[k])
                # a stowed wing is a bundle, not a tent: skin that cannot be taken
                # up by fine pleats is bunched toward the fingers instead of
                # standing up a few bone-thicknesses off the plane
                h = PLEAT_CAP * kl * np.tanh(h / (PLEAT_CAP * kl))
                # ... and no lobe stands taller than ``PLEAT_SLOPE`` of its own base (the gap
                # over the lobes): over a gap that is closing to nothing a tent is not a
                # spike but a crushed fold, and the pleat dies away with the gap (it has to:
                # where two fingers pass each other the chord reverses, and a pleat of fixed
                # height would flip from one side of the sheet to the other)
                lobe = h / kl
                room = PLEAT_SLOPE * g_pose / kl
                h = kl * lobe * room / (np.abs(lobe) + room + 1e-12)
                pleat = (h / kl)[:, None, None] * np.sin(kl * np.pi * s_ij)[..., None] * e2p[:, None, :]
                new_mid[:, c0 + 1:c1] = (ruled_p + d_pose + pleat)[:, 1:-1]
                frames.append((c0, c1, e2p))
                # a row whose two fingers are held farther apart than the skin can reach is
                # stretched evenly across, by the amount its ends dictate
                arc = np.maximum(seg.sum(axis=1), 1e-12)
                limit_v[:, c0:c1] = np.maximum(MAX_STRETCH, 1.02 * g_pose / arc)[:, None]
            pinned = np.zeros((n1, m1), dtype=bool)
            sectors = piece.pleat["sectors"]
            pinned[:, [c0 for c0, _ in sectors] + [sectors[-1][1]]] = True
            pinned |= (self._wt[a:a + nv, 0] >= 0.97).reshape(n1, m1)
            new_mid = self._limit_stretch(new_mid, r_mid, pinned, limit_v=limit_v)
            if np.allclose(new_mid, skinned_mid, atol=1e-12, rtol=0.0):
                continue
            # rebuild the two faces of the slab about the new mid-surface. The thickness
            # axis is the surface normal where that is well defined and the panel's own
            # normal where it is not (see ``_slab_normals``): not the skinned up vector,
            # which is a blend of differently rotated bones and passes through zero where
            # they are turned half way round from one another.
            # (On a finger's own column the skinned up vector stands in: two panels meeting
            # at a crease of a closed fan face opposite ways, and their normals cancel.)
            hint = up.copy()
            for c0, c1, e2p in frames:
                hint[:, c0 + 1:c1] = e2p[:, None, :]
            nrm = _slab_normals(new_mid, hint, SLAB_HINT * (2.0 * r_half) ** 2)
            # (A surface that doubles back is still wound consistently and the solid stays
            # valid; the normals are not re-oriented toward a skinned "up", which turned the
            # slab inside out wherever a panel folded over itself.)
            pos[:nv] = (new_mid + nrm * r_half[..., None]).reshape(-1, 3)
            pos[nv:2 * nv] = (new_mid - nrm * r_half[..., None]).reshape(-1, 3)
        return posed

    def _panel_ray(self, k, pleat, mid, joints) -> np.ndarray:
        """Unit vector along panel ``k`` (arm toward free edge), from the joints ``joints``
        (rest or posed): the mean direction of the metacarpals of its two bounding
        fingers; for the last panel the far side is the root line of ``mid``."""
        fingers = pleat["fingers"]
        c0, c1 = pleat["sectors"][k]

        def mc(name):
            n = joints["wrist"]
            return _unit(joints[self.skeleton.chains[name][1]] - n)

        v0 = mc(fingers[k])
        v1 = mc(fingers[k + 1]) if k + 1 < len(fingers) else _unit(mid[-1, c1] - mid[0, c1])
        return _unit(v0 + v1)

    def _hand_fold(self, pose) -> float:
        """How far the hand has folded back on the forearm, 0 (rest) .. 1 (stowed), read off
        the posed joints: the angle between the forearm and the hand's own axis (wrist to the
        knuckle of digit III), against what it is at rest."""
        def angle(P):
            fore, hand = P["wrist"] - P["elbow"], P["d3_mcp"] - P["wrist"]
            c = fore @ hand / (np.linalg.norm(fore) * np.linalg.norm(hand))
            return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
        rest = angle(self.skeleton.joints)
        return float(np.clip((angle(self.armature.posed_positions(pose)) - rest)
                             / max(170.0 - rest, 1e-6), 0.0, 1.0))

    def _gather_fingers(self, mid, r_mid, pleat, pose):
        """Lay the sheet's finger columns along their (bent) fingers, and where a finger
        has folded right up, along the straight line between its two ends.

        Skinning drops each row of a column wherever the blend of nearby bones puts it:
        near a knuckle that cuts the corner, and once a finger is a concertina (metacarpal
        out, first phalanx back, second out again) the column runs out, back and out
        and the panel between two such columns doubles over itself into a crumple. The
        skin does not do that. Past a bend a column sits on the finger, each row at the
        distance along the bone path it had at rest; past a hairpin the skin spans the
        finger's two ends and the column is the straight ray from the posed wrist to the
        posed fingertip (the closed fan), each row at the fraction of the way it was at
        rest. How far it goes is read off the finger itself, from its sharpest knuckle: a
        gentle curl (a knuckle turned 40 degrees or so on top of the finger's own hooked
        tip) keeps the skinned arc and a taut sheet.
        """
        J = self.armature.posed_positions(pose)
        w0 = J["wrist"]
        rest = self.skeleton.joints
        mid = mid.copy()
        last = 0.0
        for (c0, _), name in zip(pleat["sectors"], pleat["fingers"]):
            names = ["wrist"] + list(self.skeleton.chains[name][1:])
            path = np.array([J[j] for j in names])
            bones = np.diff(path, axis=0)
            turn = 0.0                                  # the sharpest knuckle, degrees
            for b0, b1 in zip(bones[:-1], bones[1:]):
                n0, n1 = float(np.linalg.norm(b0)), float(np.linalg.norm(b1))
                if min(n0, n1) > 1e-6:
                    turn = max(turn, float(np.degrees(np.arccos(np.clip(b0 @ b1 / (n0 * n1), -1, 1)))))
            bent = _smooth01((turn - BEND_FROM) / BEND_SPAN)
            last = folded = _smooth01((turn - FOLD_FROM) / FOLD_SPAN)
            if bent <= 0.0:
                continue
            # where each row sits along the finger at rest, as a distance along its bones (rest
            # data only: worked out once)
            rpath = np.array([rest[j] for j in names])
            seg = np.linalg.norm(np.diff(rpath, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            cache = self.__dict__.setdefault("_arc_cache", {})
            key = (id(pleat), name)
            if key not in cache:
                arc = np.empty(len(r_mid))
                for i, p in enumerate(r_mid[:, c0]):
                    d = [_plan_segment_distance(p[None, [0, 2]], a[[0, 2]], b[[0, 2]])[0]
                         for a, b in zip(rpath[:-1], rpath[1:])]
                    k = int(np.argmin(d))
                    ab = rpath[k + 1] - rpath[k]
                    u = float(np.clip((p - rpath[k]) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
                    arc[i] = cum[k] + u * seg[k]
                cache[key] = arc
            arc = cache[key].copy()
            arc = np.maximum.accumulate(arc)
            kk = np.clip(np.searchsorted(cum, arc, side="right") - 1, 0, len(seg) - 1)
            u = ((arc - cum[kk]) / np.maximum(seg[kk], 1e-12))[:, None]
            on_path = path[kk] + (path[kk + 1] - path[kk]) * u
            reach = np.linalg.norm(r_mid[:, c0] - rest["wrist"], axis=1)
            frac = (reach / max(reach[-1], 1e-12))[:, None]
            chord = w0 + (path[-1] - path[0]) * frac
            ideal = (1.0 - folded) * on_path + folded * chord
            mid[:, c0] = (1.0 - bent) * mid[:, c0] + bent * ideal
        # the column that closes the last panel (the sheet behind the last finger, from
        # the arm out to the body) is straightened with that finger: its ends are
        # already where the arm and the body put them, only the run between crumples
        c1 = pleat["sectors"][-1][1]
        if last > 0.0:
            reach = np.linalg.norm(r_mid[:, c1] - r_mid[0, c1], axis=1)
            frac = (reach / max(reach[-1], 1e-12))[:, None]
            line = mid[0, c1] + (mid[-1, c1] - mid[0, c1]) * frac
            mid[:, c1] = (1.0 - last) * mid[:, c1] + last * line
        return mid

    @staticmethod
    def _limit_stretch(mid, rest_mid, pinned, iters: int = STRETCH_ITERS, limit_v=None):
        """Pull any grid edge longer than MAX_STRETCH x its rest length back, by
        Gauss-Seidel projection: each edge in turn is shortened to the limit,
        moving both ends equally (``pinned`` vertices -- the ones riding a bone --
        stay put, so a free end takes all of it). Skin does not stretch, but
        linear-blend skinning across a hairpin collapses some rows and stretches
        their neighbours.

        Edges are visited in four colour classes (even and odd rows, even and odd
        columns), so every class is a set of disjoint edges and the sweep is exact
        Gauss-Seidel yet vectorised. A projection is a continuous function of the
        positions (nothing here counts or thresholds the active edges), so the
        limiter cannot make the sheet pop as a pose changes. Where a free chain
        lies between two pinned ends that are farther apart than the chain can
        reach, no solution exists; the sweeps then settle on an even strain along
        the chain instead of piling it on one edge."""
        mid = mid.copy()
        rest_u = np.linalg.norm(np.diff(rest_mid, axis=0), axis=2)      # between rows
        rest_v = np.linalg.norm(np.diff(rest_mid, axis=1), axis=2)      # between columns
        free = (~pinned).astype(np.float64)
        n1, m1 = mid.shape[:2]
        cap_v = np.full(rest_v.shape, MAX_STRETCH) if limit_v is None else np.asarray(limit_v)
        for _ in range(iters):
            total = 0.0
            for axis, rest in ((0, rest_u), (1, rest_v)):
                n = mid.shape[axis]
                for start in (0, 1):
                    lo = slice(start, n - 1, 2)
                    hi = slice(start + 1, n, 2)
                    k = min(len(range(*lo.indices(n))), len(range(*hi.indices(n))))
                    if k <= 0:
                        continue
                    lo = slice(start, start + 2 * k - 1, 2)
                    hi = slice(start + 1, start + 2 * k, 2)
                    if axis == 0:
                        pa, pb, fa, fb, rr = mid[lo], mid[hi], free[lo], free[hi], rest[lo]
                    else:
                        pa, pb = mid[:, lo], mid[:, hi]
                        fa, fb, rr = free[:, lo], free[:, hi], rest[:, lo]
                    d = pb - pa
                    length = np.linalg.norm(d, axis=2)
                    over = length - (MAX_STRETCH if axis == 0 else cap_v[:, lo]) * rr
                    tot = fa + fb
                    act = (over > 1e-9) & (tot > 0.0)               # an edge between two pinned ends cannot give
                    if not act.any():
                        continue
                    total += float(over[act].sum())
                    share = np.where(tot > 0.0, 1.0 / np.maximum(tot, 1e-12), 0.0)
                    corr = (np.where(act, over, 0.0) / np.maximum(length, 1e-12))[..., None] * d
                    if axis == 0:
                        mid[lo] = pa + (fa * share)[..., None] * corr
                        mid[hi] = pb - (fb * share)[..., None] * corr
                    else:
                        mid[:, lo] = pa + (fa * share)[..., None] * corr
                        mid[:, hi] = pb - (fb * share)[..., None] * corr
            if total <= 1e-6:                    # nothing is over the limit any more
                break
        return mid

    def pose_piece(self, name: str, pose=None) -> Mesh:
        a, b = self.ranges[name]
        full = self.pose_mesh(pose)
        piece = next(p for p in self.pieces if p.name == name)
        return Mesh(full.vertices[a:b], piece.mesh.faces)

    # -- controls ---------------------------------------------------------

    def pose_from_controls(self, controls: PoseControls | None = None, **kw) -> dict:
        """Joint rotations (wing frame) for the animator-level controls.

        ``controls`` is a :class:`PoseControls`, the name of a preset in
        :data:`PRESETS`, or ``None``; keyword controls are applied on top. An unknown
        control or preset name is an error (``TypeError`` / ``KeyError``, with a
        "did you mean"), and so is a control that is not a finite number
        (``ValueError``): none of them is silently ignored."""
        if isinstance(controls, str):                       # a named preset
            if controls not in PRESETS:
                near = difflib.get_close_matches(controls, list(PRESETS), n=1)
                raise KeyError(f"unknown pose preset {controls!r}"
                               + (f" (did you mean {near[0]!r}?)" if near else "")
                               + f"; the presets are {', '.join(PRESETS)}")
            controls = PoseControls(**PRESETS[controls])
        elif controls is not None and not isinstance(controls, PoseControls):
            raise TypeError("controls must be a PoseControls, a preset name or None, "
                            f"not {type(controls).__name__}; pass keyword controls as "
                            "pose(fold=0.5)")
        c = replace(controls) if controls is not None else PoseControls()
        for k, v in kw.items():
            setattr(c, k, v)                                # validated by PoseControls
        limits = getattr(self.skeleton, "control_limits", None)
        return self.skeleton.controls_to_pose(
            clamp_controls(c, getattr(self.skeleton, "fan_range", None),
                           limits() if limits is not None else None))

    def pose(self, controls: PoseControls | None = None, **kw) -> Mesh:
        """Convenience: ``rig.pose(fold=1.0)`` or ``rig.pose(raise_=40)``."""
        return self.pose_mesh(self.pose_from_controls(controls, **kw))

    def control_range(self, name: str, fold: float = 0.0) -> tuple[float, float]:
        """``(lo, hi)`` this wing allows the control ``name`` at ``fold`` (what
        ``pose_from_controls`` clamps to; a placed wing's differs, see ``placed``)."""
        limits = getattr(self.skeleton, "control_limits", None)
        return control_range(name, fold, getattr(self.skeleton, "fan_range", None),
                             limits() if limits is not None else None)

    def flap(self, phase_deg: float, amp: float = 45.0, **kw) -> PoseControls:
        """One instant of a flap cycle sized to THIS wing: ``PoseControls.flap`` with the two
        halves of the stroke reaching ``amp`` degrees or what the wing's own range leaves at
        that instant's fold, whichever is less (a placed wing rises far less than it drops)."""
        fold = PoseControls.flap(phase_deg, amp).fold
        lo, hi = self.control_range("raise_", fold)
        kw.setdefault("up_reach", min(amp, hi))
        kw.setdefault("down_reach", min(amp, -lo))
        return PoseControls.flap(phase_deg, amp, **kw)

    # -- queries used by tests and the body rig ---------------------------

    def posed_joints(self, pose=None) -> dict[str, np.ndarray]:
        self.check_pose(pose)
        return self.armature.posed_positions(pose)

    @property
    def bone_keys(self) -> list[str]:
        return list(self._bone_keys)

    def weights(self):
        """(bone index array (n, K), weight array (n, K))."""
        return self._idx.copy(), self._wt.copy()

    def skin_weights(self) -> "SkinWeights":
        """The skin weights as a host needs them: bone *names* (the child-joint keys of
        :attr:`armature`), per-vertex influences, the piece ranges, and which pieces
        a host's own skinning cannot finish (see :class:`SkinWeights`)."""
        return SkinWeights(
            bones=tuple(self._bone_keys), indices=self._idx.copy(), weights=self._wt.copy(),
            pieces=dict(self.ranges),
            post_processed=tuple(p.name for p in self.pieces if p.pleat))

    def dominant_bone(self, name: str) -> str:
        """The bone carrying the most weight in a piece (for tests)."""
        a, b = self.ranges[name]
        tot = np.zeros(len(self._bone_keys))
        for s in range(self._K):
            np.add.at(tot, self._idx[a:b, s], self._wt[a:b, s])
        return self._bone_keys[int(np.argmax(tot))]

    # -- placement --------------------------------------------------------

    def placed(self, matrix: np.ndarray) -> "PlacedWing":
        """This rig on a body. The rig now BELONGS to that placement: its shoulder controls are
        re-aimed at the body's axes (``WingSkeleton.place_on_body``), so ``raise_`` lifts and
        ``swing`` sweeps in body space. Place a second wing with a second rig (a mirrored pair is
        two rigs); ``PlacedWing(rig, matrix)`` alone leaves the rig's frame the world."""
        matrix = np.asarray(matrix, dtype=np.float64)
        place = getattr(self.skeleton, "place_on_body", None)
        if place is not None:
            place(matrix)
        return PlacedWing(self, matrix)


def euler_from_matrix(R) -> tuple[float, float, float]:
    """XYZ euler degrees ``(rx, ry, rz)`` of a 3x3 rotation, ``R = Rz Ry Rx`` (the
    convention of :func:`minimaster.core.math3d.euler_rotation`).

    ``ry`` is confined to [-90, 90], so a rotation that sweeps through ``|ry| = 90``
    (a joint swung straight back, or straight up) comes out with ``rx`` and ``rz``
    jumping by 180 degrees across it while ``ry`` turns round: the *matrix* is
    continuous and the triple is not. Euler triples therefore must not be keyed or
    interpolated; use :func:`pose_matrices` (or ``PlacedWing.body_pose(form="matrix")``)
    for that, or interpolate the controls and rebuild the pose.
    """
    R = np.asarray(R, dtype=np.float64)
    c = math.hypot(R[0, 0], R[1, 0])                       # |cos ry|, accurate near the poles
    ry = math.atan2(-R[2, 0], c)
    if c > 1e-9:
        rx = math.atan2(R[2, 1], R[2, 2])
        rz = math.atan2(R[1, 0], R[0, 0])
    else:                                                  # gimbal lock: only rx -/+ rz is defined
        rx = math.atan2(-R[1, 2], R[1, 1])
        rz = 0.0
    return tuple(float(math.degrees(v)) for v in (rx, ry, rz))


def pose_matrices(pose) -> dict[str, np.ndarray]:
    """The 3x3 rotation of every joint of a pose dict (joint -> euler degrees).

    Unlike the triples these are continuous as the controls change, so they can be
    blended (slerp / nlerp of their quaternions) by a host that wants in-betweens."""
    return {k: m3.euler_rotation(*e) for k, e in (pose or {}).items()}


@dataclass(frozen=True, eq=False)
class SkinWeights:
    """A wing's skin weights by bone *name*, for a host that skins the mesh itself.

    ``bones`` are the keys of the rig's armature (a bone is keyed by its child joint,
    as in :class:`Armature`; :meth:`PlacedWing.skin_weights` gives them with the
    attached joints' prefix, e.g. ``"wing_l:d3_tip"``). ``indices[v, k]`` indexes
    ``bones`` and ``weights[v, k]`` is the weight of vertex ``v``'s k-th influence:
    rows sum to 1, at most ``K`` (4) are non-zero, unused slots have weight 0.
    ``pieces`` maps each piece (membrane, a digit, one feather, ...) to its vertex range
    in the merged mesh.

    ``post_processed`` names the pieces whose finished shape is *not* plain linear-blend
    skinning: the membrane slab is re-evaluated between its fingers after skinning
    (``WingRig._apply_pleats``: pleats, the finger-column gather, the stretch limiter),
    which an external skinning engine does not run. Every other piece, the un-pleated
    propatagium included, is exactly the weighted sum of its bones' skin matrices.
    """

    bones: tuple
    indices: np.ndarray
    weights: np.ndarray
    pieces: dict
    post_processed: tuple = ()

    def prefixed(self, prefix: str) -> "SkinWeights":
        """The same weights with every bone name as ``f"{prefix}:{bone}"``."""
        return SkinWeights(tuple(f"{prefix}:{b}" for b in self.bones), self.indices,
                           self.weights, self.pieces, self.post_processed)

    def vertex_groups(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """``{bone name: (vertex indices, weights)}`` for every bone that carries any
        weight (the zero-length fan hinges ``d2_cmc .. d5_cmc`` and the like carry none)."""
        out = {}
        for b, name in enumerate(self.bones):
            v, k = np.nonzero((self.indices == b) & (self.weights > 0.0))
            if not len(v):
                continue
            w = self.weights[v, k]
            if len(np.unique(v)) != len(v):                 # one bone in two slots of a vertex
                v, inv = np.unique(v, return_inverse=True)
                w = np.bincount(inv, weights=w)
            out[name] = (v, w)
        return out

    def per_vertex(self) -> list[dict[str, float]]:
        """``[{bone name: weight}]``, one dict per vertex (non-zero influences only)."""
        out = []
        for idx, wt in zip(self.indices, self.weights):
            d: dict[str, float] = {}
            for i, w in zip(idx, wt):
                if w > 0.0:
                    d[self.bones[i]] = d.get(self.bones[i], 0.0) + float(w)
            out.append(d)
        return out


@dataclass
class PlacedWing:
    """A wing rig on a body: the wing-frame rig plus its 4x4 placement.

    The placement of a mirrored (right) wing has a negative determinant, which
    :meth:`Mesh.transform` already handles by flipping the winding, so posing in
    the wing frame and then placing keeps a mirrored pair an exact mirror image
    under any (symmetric) pose.

    Two ways to use it. :meth:`pose` / :meth:`mesh` give the *finished* placed mesh
    (membrane pleats included). To drive the wing from a host's own armature instead,
    :meth:`attach` adds its joints to a body :class:`Armature`, :meth:`body_pose` turns a
    pose into rotations for those joints and :meth:`skin_weights` exports the weights; the
    host then skins the placed rest mesh. See ``docs/wings.md`` for what that path can
    and cannot do.
    """

    rig: WingRig
    matrix: np.ndarray

    def mesh(self, pose=None) -> Mesh:
        return self.rig.pose_mesh(pose).transform(self.matrix)

    def pose(self, controls: PoseControls | None = None, **kw) -> Mesh:
        return self.rig.pose_mesh(self.rig.pose_from_controls(controls, **kw)).transform(
            self.matrix)

    def flap(self, phase_deg: float, amp: float = 45.0, **kw) -> PoseControls:
        """``WingRig.flap`` for this wing: a flap cycle sized to its placed range."""
        return self.rig.flap(phase_deg, amp, **kw)

    def piece_meshes(self, pose=None, **controls) -> list[tuple[str, Mesh]]:
        """``[(piece name, Mesh)]``: every closed solid of the wing, placed in BODY space
        (the frame :meth:`pose` returns) under ``pose``.

        ``pose`` is what :meth:`pose` takes (a :class:`PoseControls`, a preset name or
        ``None``, plus keyword ``controls``) or a raw pose dict as :meth:`mesh` takes. Each
        piece is a watertight, outward-oriented shell of its own, so a printability check can
        name the offending feather. The winding of a mirrored (right) wing is whatever
        :meth:`Mesh.transform` gave the whole wing -- the pieces are sliced out of that mesh,
        so the flip lives in exactly one place and a caller never has to know about it.
        """
        if isinstance(pose, Mapping):
            if controls:
                raise TypeError("keyword controls cannot be combined with a raw pose dict")
            placed = self.mesh(pose)
        else:
            placed = self.pose(pose, **controls) if (pose is not None or controls) else self.mesh()
        out = []
        v0 = f0 = 0
        for piece in self.rig.pieces:
            nv, nf = len(piece.mesh.vertices), len(piece.mesh.faces)
            out.append((piece.name, Mesh(placed.vertices[v0:v0 + nv],
                                         placed.faces[f0:f0 + nf] - v0)))
            v0 += nv
            f0 += nf
        assert v0 == len(placed.vertices) and f0 == len(placed.faces), \
            "a pose must keep every piece's vertex and face count"
        return out

    def joint_positions(self, pose=None) -> dict[str, np.ndarray]:
        """Joint positions in BODY space under ``pose``."""
        M = self.matrix
        return {k: M[:3, :3] @ v + M[:3, 3]
                for k, v in self.rig.posed_joints(pose).items()}

    def skin_matrices_body(self, pose=None) -> dict[str, np.ndarray]:
        """Skin matrices re-expressed in body space (``M S M^-1``), so a host
        armature can apply them to its own vertices."""
        M = self.matrix
        Mi = np.linalg.inv(M)
        return {k: M @ s @ Mi for k, s in self.rig.skin_matrices(pose).items()}

    # -- driving the wing from a host armature -------------------------------

    @property
    def default_prefix(self) -> str:
        """``"wing_l"`` or ``"wing_r"``: the wing is authored as a left wing and the
        right one is its mirror, i.e. the placement with a negative determinant (that is
        how :meth:`Appendage.build_rigs` places the pair). A single wing placed some
        other way should be given an explicit ``prefix``."""
        return "wing_r" if float(np.linalg.det(self.matrix[:3, :3])) < 0.0 else "wing_l"

    def joint_names(self, prefix: str | None = None) -> list[str]:
        """The names :meth:`attach` gives the wing's joints (``prefix:joint``)."""
        prefix = self.default_prefix if prefix is None else prefix
        return [f"{prefix}:{n}" for n in self.rig.skeleton.joints]

    def skin_weights(self, prefix: str | None = None) -> SkinWeights:
        """:meth:`WingRig.skin_weights` with the bone names of the attached joints."""
        prefix = self.default_prefix if prefix is None else prefix
        return self.rig.skin_weights().prefixed(prefix)

    def attach(self, armature: Armature, parent: str | None = "chest",
               prefix: str | None = None) -> list[str]:
        """Add this wing's joints to a body armature, in body space.

        The wing's root joint is parented to ``parent`` (the chest, normally) so
        the whole wing follows the torso. Returns the new joint names
        (``prefix:joint``). ``prefix`` defaults to :attr:`default_prefix`, so a mirrored
        pair can be attached one after the other; a name that is already taken is a
        ``ValueError`` and nothing is added. Drive the joints with :meth:`body_pose`
        (a wing-frame pose applied to them as it is would be wrong: their axes are the
        body's) and skin with :meth:`skin_weights`.
        """
        prefix = self.default_prefix if prefix is None else prefix
        if parent is not None and parent not in armature.joints:
            raise ValueError(f"parent joint {parent!r} does not exist")
        names = self.joint_names(prefix)
        taken = [n for n in names if n in armature.joints]
        if taken:
            raise ValueError(f"joint {taken[0]!r} already exists: attach each wing with its "
                             f"own prefix (this one would be {prefix!r}; the default is "
                             f"{self.default_prefix!r})")
        new = {}
        for name, pos in self.rig.skeleton.joints.items():
            par = self.rig.skeleton.parent[name]
            p = self.matrix[:3, :3] @ pos + self.matrix[:3, 3]
            new[name] = f"{prefix}:{name}"
            armature.add_joint(new[name], p, parent if par is None else new[par])
        return list(new.values())

    def _placement_rotation(self) -> np.ndarray:
        """The orthogonal part of the placement (rotation, possibly mirrored)."""
        L = self.matrix[:3, :3]
        g = L.T @ L
        s2 = float(np.trace(g)) / 3.0
        if s2 <= 0.0 or not np.allclose(g, s2 * np.eye(3), rtol=1e-9, atol=1e-9 * s2):
            raise ValueError("the placement is not a rotation (or mirror) times a uniform "
                             "scale, so a body armature cannot reproduce a wing-frame pose")
        return L / math.sqrt(s2)

    def body_pose(self, pose=None, prefix: str | None = None, form: str = "euler",
                  **controls) -> dict[str, object]:
        """A wing pose as rotations of the joints :meth:`attach` added, ready for the host
        armature (``body.skin_matrices(pw.body_pose(...))``).

        ``pose`` is a raw pose dict (joint -> euler degrees, wing frame, as
        :meth:`WingRig.pose_from_controls` returns), or what that method takes -- a
        :class:`PoseControls`, a preset name, ``None`` -- plus keyword ``controls``.
        The attached joints are placed in body space but a joint's rotation is
        about the *armature's* axes, not the wing's, so each wing-frame rotation ``R`` is
        conjugated by the placement ``Q`` (its rotation/mirror part, scale cancels):
        ``Q R Q^T``. That is exact for a mirrored wing too (the conjugate of a rotation is a
        rotation), and it makes the host's skin matrices equal :meth:`skin_matrices_body`.

        ``form="euler"`` (the default) gives ``(rx, ry, rz)`` degrees, the form
        :class:`Armature` poses take; ``form="matrix"`` gives the 3x3 rotations.
        Prefer the matrices for anything that blends: a triple jumps by 180 degrees in two of
        its angles when the true rotation passes ``|ry| = 90`` (see
        :func:`euler_from_matrix`). Keys are ``prefix:joint``; a joint missing from the pose
        is at rest.
        """
        if form not in ("euler", "matrix"):
            raise ValueError(f"form must be 'euler' or 'matrix', not {form!r}")
        if isinstance(pose, Mapping):
            if controls:
                raise TypeError("keyword controls cannot be combined with a raw pose dict")
            self.rig.check_pose(pose)
        else:
            pose = self.rig.pose_from_controls(pose, **controls)
        Q = self._placement_rotation()
        prefix = self.default_prefix if prefix is None else prefix
        out = {}
        for joint, e in pose.items():
            Rb = Q @ m3.euler_rotation(*e) @ Q.T
            out[f"{prefix}:{joint}"] = Rb if form == "matrix" else euler_from_matrix(Rb)
        return out
