"""A posable harvested limb: shoulder swing / raise / twist and elbow (knee) flex.

``kinds.HarvestKind`` copies the character's own arm or leg (``appendages.harvest_limb``),
moves it to where the preset puts it and hands it to a :class:`LimbRig`, which owns the
three joint markers the copy was cut around (shoulder, elbow, hand; hip, knee, ankle) and
poses it with the same machinery the studio's primitive figures use: a
:class:`~minimaster.core.armature.Armature`, :func:`~minimaster.core.bodymesh.skin_weights`
and :func:`~minimaster.core.bodymesh.pose_body_mesh`.

* **The weights are computed here, from the geometry the rig is built on** -- the cut
  region, which differs from body to body (1,276 faces on a tall body, 2,004 on the neutral
  one) -- so every re-harvest (a body slider moved, a parameter changed) gets weights that
  fit it. They cost 0.25 ms.
* **The cut end is pinned.** The vertices of the root cap, and a short run of the arm
  behind it, are bound 100% to a bone that never moves; the arm blends into the moving
  bones over a fraction of the upper-arm length. The weld that holds the limb to the torso
  therefore does not move in any pose (a rigid copy rotated about the shoulder would swing
  its cut face out of the torso).
* **The cut end is narrowed** (:func:`taper_root`): the harvested cut is as wide as the limb
  (a thigh's is wider than the torso it has to be buried in), so the first part of the limb
  draws in toward its axis, which is invisible inside the torso.
* **The pivot is the shoulder joint**, which sits inside the torso once the copy is placed;
  every rotation is about it, and the elbow's hinge is a fixed axis
  (``LimbGeometry.from_joints``), so the bones keep their lengths in every pose (the tests
  measure it).

Controls (degrees, all zero at the harvested pose; the right limb is the exact mirror of the
left one, so the same numbers pose a pair symmetrically):

``raise_``  lift the limb away from the body (abduction); negative lowers it
``swing``   swing it forward (positive) or back, in the plane through the limb and the
            forward axis, so it means the same thing hanging down and held out
``twist``   rotate about the limb's own axis (shoulder rotation)
``elbow``   flex the forearm (arm) or shank (leg: ``knee``) further; negative straightens

:func:`control_range` gives the anatomical range of each; it depends on the other controls
(a limb held out can swing less than one hanging).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core import bodymesh as bm
from ..core import math3d as m3
from ..core.armature import Armature
from ..core.mesh import Mesh

FORWARD = np.array([0.0, 0.0, 1.0])          # MakeHuman: the face looks along +Z
MIRROR = np.diag([-1.0, 1.0, 1.0])
CONTROLS = ("swing", "raise_", "twist", "elbow")
# ``knee`` is the leg's name for ``elbow``: one control, two names; a rig answers to the one
# that fits its limb (``LimbRig.control_names``)
ALIASES = {"knee": "elbow"}

# the pinned run is a fraction of the upper bone; the blend into the moving bones another
PIN_FRACTION = 0.10
BLEND_FRACTION = 0.55

# anatomical limits, degrees: (lo, hi) for raise_ (before the rest angle is subtracted),
# swing at its widest, twist, and the largest flexion of the hinge joint
_LIMITS = {
    "arm": dict(adduct=0.0, abduct=180.0, swing=(-60.0, 180.0), twist=(-75.0, 75.0), flex=145.0),
    "leg": dict(adduct=15.0, abduct=70.0, swing=(-20.0, 120.0), twist=(-45.0, 45.0), flex=135.0),
}


def rodrigues(axis, degrees: float) -> np.ndarray:
    """3x3 rotation of ``degrees`` about ``axis`` (right-handed)."""
    a = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3)
    a = a / n
    t = math.radians(degrees)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(t) * k + (1.0 - math.cos(t)) * (k @ k)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _smoothstep(x) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def taper_root(mesh: Mesh, shoulder, elbow, root_loop, root_pole=None, fraction: float = 0.5,
               length: float = 1.0) -> Mesh:
    """``mesh`` with the cut end drawn in toward the limb's axis (a short neck).

    The harvested end of a limb is a cut through the deltoid or the thigh: as wide as the
    limb and ragged. The limb is welded by burying that end in the torso, and a leg's cut is
    wider than the torso it would have to fit into. So the first ``length`` of the limb
    narrows smoothly to ``fraction`` of its radius at the cut: every vertex moves toward the
    axis (shoulder -> elbow) by a factor that rises from ``fraction`` at the cut to 1, which
    is a continuous map, so the shell stays closed and outward. Inside the torso the neck is
    invisible; ``fraction = 1`` leaves the mesh as it was.
    """
    if fraction >= 1.0 or length <= 0.0:
        return Mesh(mesh.vertices.copy(), mesh.faces)
    s = np.asarray(shoulder, dtype=np.float64)
    axis = _unit(np.asarray(elbow, dtype=np.float64) - s)
    v = mesh.vertices
    along = (v - s) @ axis
    rim = np.asarray(root_loop, dtype=np.int64)
    cut = float(along[rim].mean()) if len(rim) else float(along.min())
    on_axis = s + along[:, None] * axis
    f = fraction + (1.0 - fraction) * _smoothstep((along - cut) / length)
    out = on_axis + (v - on_axis) * f[:, None]
    if root_pole is not None:               # the cap's pole stays on the axis, at the cut
        out[root_pole] = on_axis[root_pole] + (v[root_pole] - on_axis[root_pole]) * fraction
    return Mesh(out, mesh.faces)


@dataclass(frozen=True)
class LimbGeometry:
    """The constants a limb's controls are defined by, in the LEFT-hand frame (the right
    limb is mirrored into it): the rest direction of the upper bone, the hinge axis, the
    rest flexion and the abduction angle."""

    limb: str
    upper: np.ndarray            # unit shoulder -> elbow
    hinge: np.ndarray            # unit axis the forearm flexes about (positive = more bent)
    rest_flex: float             # degrees the forearm is already bent at the harvested pose
    rest_abduction: float        # degrees the upper bone is out from straight down
    limits: tuple = ()           # ((key, value), ...) overrides of ``_LIMITS`` for this limb: a
                                 # second girdle is held to the range it can move in without
                                 # tenting the trunk or hitting the original limb

    @classmethod
    def from_joints(cls, shoulder, elbow, hand, limb: str = "arm", mirrored: bool = False,
                    limits: dict | None = None):
        s, e, h = (np.asarray(p, dtype=np.float64) for p in (shoulder, elbow, hand))
        if mirrored:
            s, e, h = (MIRROR @ p for p in (s, e, h))
        upper = _unit(e - s)
        fore = _unit(h - e)
        # the hinge is fixed by anatomy, not read off the (nearly straight) harvested leg:
        # an elbow flexes the forearm forward, a knee flexes the shank back
        bend_toward = FORWARD if limb == "arm" else -FORWARD
        hinge = _unit(np.cross(upper, bend_toward))
        in_plane = _unit(fore - float(fore @ hinge) * hinge)
        rest_flex = math.degrees(math.atan2(float(hinge @ np.cross(upper, in_plane)),
                                            float(upper @ in_plane)))
        rest_abduction = math.degrees(math.atan2(float(upper[0]), float(-upper[1])))
        return cls(limb, upper, hinge, rest_flex, rest_abduction,
                   tuple(sorted((limits or {}).items())))

    @property
    def lim(self) -> dict:
        """The anatomical limits: the defaults of the limb with this geometry's overrides."""
        out = dict(_LIMITS[self.limb])
        out.update(dict(self.limits))
        return out

    # -- ranges -----------------------------------------------------------------------

    def control_range(self, name: str, state: dict | None = None):
        """``(lo, hi)`` degrees the control ``name`` takes given the others in ``state``,
        or ``None`` for a name this limb does not have."""
        name = ALIASES.get(name, name)
        if name not in CONTROLS:
            return None
        lim = self.lim
        state = state or {}
        if name == "raise_":
            return (-(self.rest_abduction + lim["adduct"]), lim["abduct"] - self.rest_abduction)
        if name == "twist":
            return lim["twist"]
        if name == "elbow":
            return (min(0.0, -self.rest_flex), lim["flex"] - self.rest_flex)
        # swing: narrower as the limb is held out sideways (it would cross the body or the
        # shoulder); widest hanging and held straight up
        beta = math.radians(self.rest_abduction + float(state.get("raise_", 0.0)))
        s = abs(math.sin(beta))
        lo, hi = lim["swing"]
        if self.limb == "arm":
            return (lo + 15.0 * s, hi - 50.0 * s)
        return (lo, hi - 40.0 * s)

    def clamp(self, controls: dict) -> dict:
        """``controls`` (canonical names) with every value inside its range."""
        out = {k: float(controls.get(k, 0.0)) for k in CONTROLS}
        for name in ("raise_", "twist", "elbow", "swing"):       # swing last: depends on raise_
            lo, hi = self.control_range(name, out)
            out[name] = min(max(out[name], lo), hi)
        return out

    # -- the rotations ----------------------------------------------------------------

    def shoulder_rotation(self, controls: dict) -> np.ndarray:
        """3x3 about the shoulder, in the left-hand frame: twist about the harvested bone
        axis, then raise (about the front-back axis), then swing about the axis through the
        raised bone and the forward direction."""
        c = {k: float(controls.get(k, 0.0)) for k in CONTROLS}
        r_twist = rodrigues(self.upper, c["twist"])
        r_raise = m3.rot_z(c["raise_"])
        raised = r_raise @ self.upper
        r_swing = rodrigues(np.cross(raised, FORWARD), c["swing"])
        return r_swing @ r_raise @ r_twist

    def elbow_rotation(self, controls: dict) -> np.ndarray:
        return rodrigues(self.hinge, float(controls.get("elbow", 0.0)))


# the rest geometry of the neutral body's arm and leg, for ranges before a limb is placed:
# the upper bone's direction, the hinge axis, the harvested bend and abduction (degrees)
_TYPICAL = {
    "arm": ((0.67, -0.74, 0.0), (-0.74, -0.67, 0.0), 42.0, 42.0),
    "leg": ((0.10, -1.00, 0.08), (-1.00, -0.10, 0.0), 6.0, 5.7),
}


def typical_geometry(limb: str = "arm") -> LimbGeometry:
    """The :class:`LimbGeometry` of a typical (neutral body) arm or leg."""
    limb = "leg" if limb == "leg" else "arm"
    upper, hinge, flex, abd = _TYPICAL[limb]
    return LimbGeometry(limb, _unit(upper), _unit(hinge), flex, abd)


class LimbRig:
    """A harvested limb, placed, with its skeleton and weights; ``pose`` re-meshes it.

    ``mesh`` is the rest mesh IN PLACE (what the document's placement made of the copy),
    ``shoulder`` / ``elbow`` / ``hand`` the joint positions in the same frame (the markers
    of the side the copy was cut from), ``root_loop`` / ``root_pole`` the vertices of the cut
    that is welded into the torso, ``mirrored`` for a right-hand limb.
    """

    def __init__(self, mesh: Mesh, shoulder, elbow, hand, *, limb: str = "arm",
                 mirrored: bool = False, root_loop=None, root_pole=None,
                 limits: dict | None = None, pin_fraction: float = PIN_FRACTION,
                 blend_fraction: float = BLEND_FRACTION, girdle=None):
        self.mesh = mesh
        self.pin_fraction = float(pin_fraction)
        self.blend_fraction = float(blend_fraction)
        self.girdle = girdle                       # a limbgirdle.Girdle (left frame) or None
        self.limb = limb
        self.mirrored = bool(mirrored)
        self.joints_rest = {"shoulder": np.asarray(shoulder, dtype=np.float64),
                            "elbow": np.asarray(elbow, dtype=np.float64),
                            "hand": np.asarray(hand, dtype=np.float64)}
        self.geometry = LimbGeometry.from_joints(shoulder, elbow, hand, limb, mirrored, limits)
        self.root_vertices = np.unique(np.concatenate([
            np.asarray(root_loop if root_loop is not None else [], dtype=np.int64),
            np.asarray([] if root_pole is None else [root_pole], dtype=np.int64)]))
        self.armature = self._armature()
        self.bone_names, self.weights = self._weights()

    # -- construction -------------------------------------------------------------------

    def _armature(self) -> Armature:
        j = self.joints_rest
        arm = Armature()
        arm.add_joint("root", j["shoulder"])                    # never rotated: the pinned bone
        arm.add_joint("shoulder", j["shoulder"], "root")        # the pivot of every control
        arm.add_joint("elbow", j["elbow"], "shoulder")
        arm.add_joint("hand", j["hand"], "elbow")
        return arm

    def _weights(self):
        """Distance-weighted skinning to the two moving bones, with the cut end pinned.

        The pinned weight is 1 up to the farthest rim vertex along the upper bone plus a
        short run, then falls off smoothly over a fraction of the bone: a blend that long
        keeps the first part of the limb (inside the torso once placed) in its rest pose
        and lets the visible limb bend like one."""
        j = self.joints_rest
        v = self.mesh.vertices
        moving = Armature()
        moving.add_joint("shoulder", j["shoulder"])
        moving.add_joint("elbow", j["elbow"], "shoulder")
        moving.add_joint("hand", j["hand"], "elbow")
        names, w = bm.skin_weights(v, moving, max_bones=2)      # keys: elbow (upper), hand
        upper_len = float(np.linalg.norm(j["elbow"] - j["shoulder"]))
        axis = _unit(j["elbow"] - j["shoulder"])
        s = (v - j["shoulder"]) @ axis
        if len(self.root_vertices):
            s_rim = float(s[self.root_vertices].max())
        else:
            s_rim = float(s.min())
        start = s_rim + self.pin_fraction * upper_len
        pin = 1.0 - _smoothstep((s - start) / (self.blend_fraction * upper_len))   # 1 at the rim
        full = np.hstack([w * (1.0 - pin)[:, None], pin[:, None]])
        return names + ["shoulder"], full                       # key "shoulder": root -> shoulder

    # -- posing -------------------------------------------------------------------------

    @property
    def control_names(self) -> tuple:
        return ("swing", "raise_", "twist", "knee" if self.limb == "leg" else "elbow")

    def control_range(self, name: str, state: dict | None = None):
        if name not in self.control_names:
            return None
        return self.geometry.control_range(name, self._canonical(state or {}))

    def _canonical(self, controls: dict) -> dict:
        """``controls`` under the canonical names (``knee`` -> ``elbow``); the flex control
        of the OTHER limb (``knee`` on an arm, ``elbow`` on a leg) is dropped."""
        other = "elbow" if self.limb == "leg" else "knee"
        return {ALIASES.get(k, k): val for k, val in controls.items() if k != other}

    def _pose(self, controls: dict) -> dict:
        c = self.geometry.clamp(self._canonical(controls))
        rs = self.geometry.shoulder_rotation(c)
        re = self.geometry.elbow_rotation(c)
        if self.mirrored:                    # the right limb is the exact mirror of the left
            rs, re = MIRROR @ rs @ MIRROR, MIRROR @ re @ MIRROR
        return {"shoulder": m3.matrix_to_euler_xyz(rs), "elbow": m3.matrix_to_euler_xyz(re)}

    def pose(self, controls: dict | None = None) -> Mesh:
        """The limb under ``controls`` (a dict of the names above; missing = 0, out of range
        = clamped). Faces are the rest mesh's, so it stays a closed, outward shell."""
        canon = self._canonical(controls or {})
        if not any(float(canon.get(k, 0.0)) for k in CONTROLS):
            return Mesh(self.mesh.vertices.copy(), self.mesh.faces)
        return bm.pose_body_mesh(self.mesh, self.bone_names, self.weights, self.armature,
                                 self._pose(controls))

    def joints(self, controls: dict | None = None) -> dict:
        """Posed ``{"shoulder": p, "elbow": p, "hand": p}``."""
        pos = self.armature.posed_positions(self._pose(controls or {}))
        return {k: pos[k] for k in ("shoulder", "elbow", "hand")}
