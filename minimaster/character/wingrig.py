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
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

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
    ``raise_`` degrees the whole arm is lifted from the rest plane (flap).
    ``swing``  degrees the arm is swung back (+) or forward (-).
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

    @classmethod
    def flap(cls, phase_deg: float, amp: float = 45.0, **kw) -> "PoseControls":
        """One instant of a flap cycle. ``phase_deg`` 0 = mid-stroke going up,
        90 = top of the upstroke, 270 = bottom of the downstroke. The shoulder
        leads; the wrist and the finger curl lag it (40 and 70 degrees of phase),
        so the hand trails the wrist -- hanging below it when the wing is raised --
        instead of the whole wing swinging like a plank."""
        ph = np.radians(phase_deg)
        up = max(0.0, float(np.sin(ph)))                        # 0 on the downstroke
        c = cls(raise_=amp * float(np.sin(ph)),
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


# Anatomical ranges of the controls, from skeleton and flight photographs (and
# the literature where a photo cannot show it): a wing cannot be wrenched past
# what the joints allow, so the controls are clamped to these.
LIMITS = {
    "fold": (0.0, 1.0),
    "raise_": (-70.0, 100.0),         # shoulder elevation: hanging .. wrists-over-back
    "swing": (-30.0, 85.0),           # humerus protraction .. stowed along the spine
    "twist": (-45.0, 45.0),           # humerus roll: supination .. pronation
    "fan": (0.0, 1.5),
    "curl": (-10.0, 150.0),           # knuckle flexion
    "wrist_flex": (-60.0, 45.0),      # hand drooping .. cocked up
    "hand_twist": (-30.0, 30.0),
}

# Named poses over the controls. A flap cycle is up -> down -> up: the upstroke
# lifts the wrist and half-folds the arm with the leading edge turned edge-on;
# the downstroke sweeps through level with the arm extended, leading edge down.
PRESETS = {
    "glide": {},
    "half_fold": {"fold": 0.5},
    "folded": {"fold": 1.0},
    "upstroke": {"raise_": 55.0, "fold": 0.35, "twist": -15.0, "fan": 0.7, "wrist_flex": -30.0},
    "downstroke": {"raise_": -40.0, "twist": 15.0, "curl": 12.0, "wrist_flex": 20.0},
    "mid_downstroke": {"raise_": 5.0, "twist": 10.0, "wrist_flex": 8.0},
}


def clamp_controls(c: "PoseControls", fan_range=None) -> "PoseControls":
    """A copy of the controls limited to what the joints can do (the caller's
    object is left alone). ``fan_range`` narrows ``fan`` for a skeleton whose
    skin cannot take the full range (a membrane tears long before a feather)."""
    out = replace(c)
    for name, (lo, hi) in LIMITS.items():
        if name == "fan" and fan_range is not None:
            lo, hi = fan_range
        setattr(out, name, float(np.clip(getattr(c, name), lo, hi)))
    return out


# How far a membrane panel can close before it has to buckle into a pleat.
ELASTIC = 0.90
# How far skin may stretch (relative to its rest length) before the limiter
# pulls it back. Compression is free -- that is what the pleats are for.
MAX_STRETCH = 1.15

# --------------------------------------------------------------------------
# the rig


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

    def skin_matrices(self, pose=None) -> dict[str, np.ndarray]:
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
        out = self._apply_pleats(out)
        return Mesh(out, self.rest_mesh.faces)

    def _apply_pleats(self, posed: np.ndarray, gain: float = 0.8) -> np.ndarray:
        """Re-evaluate each membrane panel between its two posed bone lines.

        Skinning moves the vertices that sit on a finger with that finger, but
        everything between two fingers is left wherever blended transforms drop
        it, and when the fingers close the sheet simply crumples. A real
        patagium is a continuous, inextensible skin stretched between bones:
        for every row of a panel, the posed points on its two bounding fingers
        define a chord, and the interior is that chord plus the panel's own
        rest-pose curvature (the scallop sag and billow, scaled by how far the
        gap has closed). What the sheet cannot lose -- the slack between the
        rest gap ``L`` and the posed gap ``g`` -- buckles into a pleat: a tent of
        slant ``L/2`` over a base ``g``, height sqrt(L^2 - g^2)/2, standing along
        the bones' own up direction (taken from the finger vertices, which
        skinning keeps stable). Neighbouring panels pleat in opposite
        directions, the way a closing fan zig-zags.
        """
        specs = [(p, self.ranges[p.name]) for p in self.pieces if p.pleat]
        if not specs:
            return posed
        for piece, (a, b) in specs:
            n1, m1 = piece.pleat["shape"]
            gains = piece.pleat.get("gains", [1.0] * len(piece.pleat["sectors"]))
            lobes = piece.pleat.get("lobes", [1] * len(piece.pleat["sectors"]))
            nv = n1 * m1
            rest = self.rest_mesh.vertices[a:b]
            pos = posed[a:b]
            r_mid = 0.5 * (rest[:nv] + rest[nv:2 * nv]).reshape(n1, m1, 3)
            r_half = 0.5 * np.linalg.norm(rest[:nv] - rest[nv:2 * nv], axis=1).reshape(n1, m1)
            top = pos[:nv].reshape(n1, m1, 3)
            bot = pos[nv:2 * nv].reshape(n1, m1, 3)
            mid = 0.5 * (top + bot)
            up = top - bot                                  # stable at the finger columns
            up /= np.maximum(np.linalg.norm(up, axis=2, keepdims=True), 1e-12)
            new_mid = mid.copy()
            touched = False
            for k, (c0, c1) in enumerate(piece.pleat["sectors"]):
                g_rest = np.linalg.norm(r_mid[:, c1] - r_mid[:, c0], axis=1)
                g_pose = np.linalg.norm(mid[:, c1] - mid[:, c0], axis=1)
                # skin is elastic: it takes up a few percent of closure by itself,
                # and only buckles beyond that (a gentle curl must not inflate)
                g_eff = ELASTIC * g_rest
                slack = np.maximum(g_eff ** 2 - g_pose ** 2, 0.0)
                fade = np.clip(slack / (0.02 * g_rest ** 2 + 1e-30), 0.0, 1.0)
                if not fade.any():
                    continue
                touched = True
                # where each interior column sits across the panel at rest
                seg = np.linalg.norm(np.diff(r_mid[:, c0:c1 + 1], axis=1), axis=2)
                s_ij = np.concatenate([np.zeros((n1, 1)), np.cumsum(seg, axis=1)], axis=1)
                s_ij = s_ij / np.maximum(s_ij[:, -1:], 1e-12)
                chord_r = r_mid[:, c1] - r_mid[:, c0]
                chord_p = mid[:, c1] - mid[:, c0]
                e1r = chord_r / np.maximum(np.linalg.norm(chord_r, axis=1, keepdims=True), 1e-12)
                e1p = chord_p / np.maximum(np.linalg.norm(chord_p, axis=1, keepdims=True), 1e-12)
                n_axis = up[:, c0] + up[:, c1]
                n_axis /= np.maximum(np.linalg.norm(n_axis, axis=1, keepdims=True), 1e-12)
                # rest panel's up axis, to express the rest curvature in a frame
                n_rest = np.cross(e1r, np.array([0.0, 1.0, 0.0]))        # in-plane normal guess
                ratio = (g_pose / np.maximum(g_rest, 1e-12))[:, None, None]
                ruled_r = r_mid[:, c0][:, None, :] + chord_r[:, None, :] * s_ij[..., None]
                d_rest = r_mid[:, c0:c1 + 1] - ruled_r                    # curvature off the chord
                # carry the offset into the posed frame: components along the
                # chord, the wing-up axis and their cross product
                up_r = np.array([0.0, -1.0, 0.0])
                e3r = np.cross(e1r, up_r)
                e3r /= np.maximum(np.linalg.norm(e3r, axis=1, keepdims=True), 1e-12)
                e3p = np.cross(e1p, n_axis)
                e3p /= np.maximum(np.linalg.norm(e3p, axis=1, keepdims=True), 1e-12)
                c1_ = np.einsum("ijk,ik->ij", d_rest, e1r)
                c2_ = np.einsum("ijk,k->ij", d_rest, up_r)
                c3_ = np.einsum("ijk,ik->ij", d_rest, e3r)
                d_pose = (c1_[..., None] * e1p[:, None, :] + c2_[..., None] * n_axis[:, None, :]
                          + c3_[..., None] * e3p[:, None, :]) * ratio
                ruled_p = mid[:, c0][:, None, :] + chord_p[:, None, :] * s_ij[..., None]
                h = (gain * gains[k] * 0.5 * np.sqrt(slack) * fade
                     * (1.0 if k % 2 == 0 else -1.0))
                # k fine pleats instead of one bladder: each is a tent of slant L/2k
                # over a base g/k, so k lobes of height h/k
                kl = float(lobes[k])
                pleat = (h / kl)[:, None, None] * np.sin(kl * np.pi * s_ij)[..., None] * n_axis[:, None, :]
                fresh = ruled_p + d_pose + pleat
                w = fade[:, None, None]                                   # blend in with closure
                new_mid[:, c0:c1 + 1] = (1.0 - w) * mid[:, c0:c1 + 1] + w * fresh
            pinned = (self._wt[a:a + nv, 0] >= 0.97).reshape(n1, m1)
            limited = self._limit_stretch(new_mid, r_mid, pinned)
            if not touched and np.allclose(limited, new_mid):
                continue
            new_mid = limited
            # rebuild the two faces of the slab about the new mid-surface
            du = np.gradient(new_mid, axis=0)
            dv = np.gradient(new_mid, axis=1)
            nrm = np.cross(du, dv)
            nrm /= np.maximum(np.linalg.norm(nrm, axis=2, keepdims=True), 1e-12)
            # Keep the grid's own orientation. Flipping cells toward the skinned
            # "up" vector looked safe but turned the slab inside out wherever a
            # panel folded over itself (signed volume fell from +0.75 to -0.03 as
            # the wing stowed): a surface that doubles back is still wound
            # consistently, and the solid stays valid.
            pos[:nv] = (new_mid + nrm * r_half[..., None]).reshape(-1, 3)
            pos[nv:2 * nv] = (new_mid - nrm * r_half[..., None]).reshape(-1, 3)
        return posed

    @staticmethod
    def _limit_stretch(mid, rest_mid, pinned, iters: int = 16):
        """Jacobi passes that pull any grid edge longer than MAX_STRETCH x its
        rest length back, moving both ends equally (``pinned`` vertices -- those
        riding a bone -- stay put). Linear-blend skinning across a hairpin
        collapses some rows and stretches their neighbours; skin does not."""
        rest_u = np.linalg.norm(np.diff(rest_mid, axis=0), axis=2)      # between rows
        rest_v = np.linalg.norm(np.diff(rest_mid, axis=1), axis=2)      # between columns
        free = (~pinned)[..., None]
        for _ in range(iters):
            disp = np.zeros_like(mid)
            cnt = np.zeros(mid.shape[:2])
            for d_axis, rest in ((0, rest_u), (1, rest_v)):
                a = mid[:-1] if d_axis == 0 else mid[:, :-1]
                b = mid[1:] if d_axis == 0 else mid[:, 1:]
                d = b - a
                length = np.linalg.norm(d, axis=2)
                over = length - MAX_STRETCH * rest
                mask = over > 1e-9
                corr = (np.where(mask, over / np.maximum(length, 1e-12), 0.0))[..., None] * d * 0.5
                if d_axis == 0:
                    disp[:-1] += corr; disp[1:] -= corr
                    cnt[:-1] += mask; cnt[1:] += mask
                else:
                    disp[:, :-1] += corr; disp[:, 1:] -= corr
                    cnt[:, :-1] += mask; cnt[:, 1:] += mask
            if not cnt.any():
                break
            mid = mid + np.where(free, disp / np.maximum(cnt, 1.0)[..., None], 0.0)
        return mid

    def pose_piece(self, name: str, pose=None) -> Mesh:
        a, b = self.ranges[name]
        full = self.pose_mesh(pose)
        piece = next(p for p in self.pieces if p.name == name)
        return Mesh(full.vertices[a:b], piece.mesh.faces)

    # -- controls ---------------------------------------------------------

    def pose_from_controls(self, controls: PoseControls | None = None, **kw) -> dict:
        """Joint rotations (wing frame) for the animator-level controls."""
        if isinstance(controls, str):                       # a named preset
            controls = PoseControls(**PRESETS[controls])
        c = replace(controls) if controls is not None else PoseControls()
        for k, v in kw.items():
            setattr(c, k, v)
        return self.skeleton.controls_to_pose(
            clamp_controls(c, getattr(self.skeleton, "fan_range", None)))

    def pose(self, controls: PoseControls | None = None, **kw) -> Mesh:
        """Convenience: ``rig.pose(fold=1.0)`` or ``rig.pose(raise_=40)``."""
        return self.pose_mesh(self.pose_from_controls(controls, **kw))

    # -- queries used by tests and the body rig ---------------------------

    def posed_joints(self, pose=None) -> dict[str, np.ndarray]:
        return self.armature.posed_positions(pose)

    @property
    def bone_keys(self) -> list[str]:
        return list(self._bone_keys)

    def weights(self):
        """(bone index array (n, K), weight array (n, K))."""
        return self._idx.copy(), self._wt.copy()

    def dominant_bone(self, name: str) -> str:
        """The bone carrying the most weight in a piece (for tests)."""
        a, b = self.ranges[name]
        tot = np.zeros(len(self._bone_keys))
        for s in range(self._K):
            np.add.at(tot, self._idx[a:b, s], self._wt[a:b, s])
        return self._bone_keys[int(np.argmax(tot))]

    # -- placement --------------------------------------------------------

    def placed(self, matrix: np.ndarray) -> "PlacedWing":
        return PlacedWing(self, np.asarray(matrix, dtype=np.float64))


@dataclass
class PlacedWing:
    """A wing rig on a body: the wing-frame rig plus its 4x4 placement.

    The placement of a mirrored (right) wing has a negative determinant, which
    :meth:`Mesh.transform` already handles by flipping the winding, so posing in
    the wing frame and then placing keeps a mirrored pair an exact mirror image
    under any (symmetric) pose.
    """

    rig: WingRig
    matrix: np.ndarray

    def mesh(self, pose=None) -> Mesh:
        return self.rig.pose_mesh(pose).transform(self.matrix)

    def pose(self, controls: PoseControls | None = None, **kw) -> Mesh:
        return self.rig.pose_mesh(self.rig.pose_from_controls(controls, **kw)).transform(
            self.matrix)

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

    def attach(self, armature: Armature, parent: str | None = "chest",
               prefix: str = "wing") -> list[str]:
        """Add this wing's joints to a body armature, in body space.

        The wing's root joint is parented to ``parent`` (the chest, normally) so
        the whole wing follows the torso. Returns the new joint names.
        """
        names = {}
        for name, pos in self.rig.skeleton.joints.items():
            par = self.rig.skeleton.parent[name]
            new = f"{prefix}:{name}"
            p = self.matrix[:3, :3] @ pos + self.matrix[:3, 3]
            armature.add_joint(new, p, parent if par is None else names[par])
            names[name] = new
        return list(names.values())
