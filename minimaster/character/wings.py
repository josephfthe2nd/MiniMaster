"""Wings built the way wings are actually built -- and rigged the same way.

A wing is not a slab. In both bats and birds it is a **modified forelimb**:
humerus, then radius/ulna, then a wrist, and then an enormously elongated hand
that carries the flight surface. Everything a viewer reads as "wing" -- the
elbow crook, the finger struts fanning through a membrane, the scalloped
trailing edge, the layered feathers -- comes from that skeleton. So this module
builds the skeleton first, hangs the surface on it, and hands both to
:mod:`minimaster.character.wingrig`, which binds every vertex to the joints so
the wing can be posed (folded, raised, flapped) by rotating joints instead of
rebuilding a mesh.

Two styles share one arm:

* **membrane** (bat / dragon). The patagium is stretched over digits II-V. Real
  bats divide it into named regions: the *propatagium* in front of the arm, the
  *dactylopatagium* between the fingers, and the *plagiopatagium* from the arm
  back to the body flank. Digit I (the thumb) stays free and clawed at the
  wrist -- the single most recognisable bat/dragon detail.
* **feathered** (bird / angel). Primaries grow from the hand, secondaries from
  the forearm, and shingled rows of coverts lie over their bases.

Output contract: every piece is its own closed solid, merged into one mesh.
Overlapping closed shells are exactly what the rest of MiniMaster expects --
the slicer unions them -- so bones can sit *inside* the membrane and read as
ridges without any boolean work.

Authored frame (the wing is built at the origin, then :func:`appendages.place`
puts it on the body)::

    +X  spanwise, shoulder -> wingtip
    +Y  down (the wing's underside); camber bulges toward -Y
    +Z  chordwise, leading edge -> trailing edge
"""

from __future__ import annotations

import numpy as np

from ..core import math3d as m3
from ..core.mesh import Mesh
from .wingrig import (Piece, PoseControls, WingRig, compose, raise_,
                      swing_back, twist)

DEFAULT_SPAN = 13.0         # wing units; one MakeHuman unit is a decimetre
CAMBER = 0.06               # of chord, at rest; a gliding bat reaches 0.12 (armwing 0.118-0.131)
CAMBER_PEAK = 0.72          # exponent placing max camber near 30% chord


def _dir(deg: float) -> np.ndarray:
    """Unit vector in the wing plane, measured from +X toward +Z."""
    a = np.radians(deg)
    return np.array([np.cos(a), 0.0, np.sin(a)])


# --------------------------------------------------------------------------
# mesh primitives (self-contained so this module never imports appendages)


def _orient(mesh: Mesh) -> Mesh:
    """Flip the winding if the solid came out inside-out."""
    if mesh.volume() < 0:
        return Mesh(mesh.vertices, mesh.faces[:, [0, 2, 1]])
    return mesh


def _quads_to_tris(quads) -> np.ndarray:
    q = np.asarray(quads, dtype=np.int64)
    return np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])


def _tube(points, radii, sides: int = 8) -> Mesh:
    """A closed tapered tube through ``points``, capped with fans at both ends."""
    pts = np.asarray(points, dtype=np.float64)
    rad = np.asarray(radii, dtype=np.float64)
    n = len(pts)
    tangents = np.zeros_like(pts)
    tangents[1:-1] = pts[2:] - pts[:-2]
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(norms, 1e-12)

    # one reference frame carried along the tube; bones are short, so a single
    # seed direction is enough to stay twist-free
    ref = np.array([0.0, 1.0, 0.0])
    if abs(float(tangents[0] @ ref)) > 0.95:
        ref = np.array([0.0, 0.0, 1.0])
    rings = []
    th = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    for i in range(n):
        t = tangents[i]
        u = ref - float(ref @ t) * t
        lu = np.linalg.norm(u)
        u = u / lu if lu > 1e-9 else np.array([0.0, 0.0, 1.0])
        w = np.cross(t, u)
        ref = u                       # carry the frame forward
        rings.append(pts[i] + rad[i] * (np.cos(th)[:, None] * u[None, :]
                                        + np.sin(th)[:, None] * w[None, :]))
    verts = np.vstack(rings)
    quads = []
    for i in range(n - 1):
        b0, b1 = i * sides, (i + 1) * sides
        for k in range(sides):
            kn = (k + 1) % sides
            quads.append([b0 + k, b0 + kn, b1 + kn, b1 + k])
    tris = list(_quads_to_tris(quads))

    # fan caps: one centre vertex per end
    c0 = len(verts)
    verts = np.vstack([verts, pts[0][None, :], pts[-1][None, :]])
    c1 = c0 + 1
    last = (n - 1) * sides
    for k in range(sides):
        kn = (k + 1) % sides
        tris.append([c0, kn, k])
        tris.append([c1, last + k, last + kn])
    return _orient(Mesh(verts, np.asarray(tris, dtype=np.int64)))


def _grid_normals(grid: np.ndarray) -> np.ndarray:
    """Per-vertex normals of an (n+1, m+1, 3) surface grid."""
    du = np.gradient(grid, axis=0)
    dv = np.gradient(grid, axis=1)
    nrm = np.cross(du, dv)
    ln = np.linalg.norm(nrm, axis=2, keepdims=True)
    return nrm / np.maximum(ln, 1e-12)


def _slab_from_grid(grid: np.ndarray, thickness: np.ndarray) -> Mesh:
    """Thicken a surface grid into a closed solid.

    ``grid`` is (n+1, m+1, 3) mid-surface points, ``thickness`` an (n+1, m+1)
    field. Emits a top sheet, a bottom sheet, and a rim stitching their shared
    perimeter, wound so every edge is used exactly twice.
    """
    n1, m1, _ = grid.shape
    n, m = n1 - 1, m1 - 1
    nrm = _grid_normals(grid)
    half = (thickness / 2.0)[:, :, None]
    top = (grid + nrm * half).reshape(-1, 3)
    bot = (grid - nrm * half).reshape(-1, 3)
    verts = np.vstack([top, bot])
    nv = n1 * m1

    def T(i, j):
        return i * m1 + j

    def B(i, j):
        return nv + i * m1 + j

    quads = []
    for i in range(n):
        for j in range(m):
            quads.append([T(i, j), T(i, j + 1), T(i + 1, j + 1), T(i + 1, j)])
            quads.append([B(i, j), B(i + 1, j), B(i + 1, j + 1), B(i, j + 1)])

    # Perimeter in the same rotational sense the top sheet is wound in, so each
    # rim quad can re-use its boundary edge backwards.
    per = ([(0, j) for j in range(m1)]
           + [(i, m) for i in range(1, n1)]
           + [(n, j) for j in range(m - 1, -1, -1)]
           + [(i, 0) for i in range(n - 1, 0, -1)])
    for k in range(len(per)):
        a = per[k]
        b = per[(k + 1) % len(per)]
        quads.append([T(*b), T(*a), B(*a), B(*b)])

    return _orient(Mesh(verts, _quads_to_tris(quads)))


def _resample(polyline, count: int) -> np.ndarray:
    """Even arc-length resampling of a polyline to ``count`` points."""
    p = np.asarray(polyline, dtype=np.float64)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        return np.repeat(p[:1], count, axis=0)
    want = np.linspace(0.0, total, count)
    out = np.empty((count, 3))
    for k in range(3):
        out[:, k] = np.interp(want, s, p[:, k])
    return out


def _lift(flat_xz, flat_y, query_xz, sigma):
    """Gaussian-weighted height lookup, to sit bones on the membrane surface.

    The membrane is a parametric sweep, so a bone drawn in the flat plan has no
    closed-form (u, v). Weighting the surrounding grid heights is smooth, needs
    no solver, and is exact wherever a bone lies on a grid line.
    """
    d2 = ((query_xz[:, None, :] - flat_xz[None, :, :]) ** 2).sum(axis=2)
    w = np.exp(-d2 / (2.0 * sigma * sigma))
    tot = w.sum(axis=1)
    out = (w @ flat_y) / np.maximum(tot, 1e-30)
    # far outside the membrane the weights underflow; fall back to the nearest
    lost = tot < 1e-12
    if lost.any():
        out[lost] = flat_y[np.argmin(d2[lost], axis=1)]
    return out


def _ball(center, radius, n_lat: int = 6, n_lon: int = 10) -> Mesh:
    """A closed UV sphere, used for joint knobs."""
    c = np.asarray(center, dtype=np.float64)
    verts = [c + np.array([0.0, 0.0, radius]), c + np.array([0.0, 0.0, -radius])]
    for i in range(1, n_lat):
        th = np.pi * i / n_lat
        for j in range(n_lon):
            ph = 2.0 * np.pi * j / n_lon
            verts.append(c + radius * np.array([np.sin(th) * np.cos(ph),
                                                np.sin(th) * np.sin(ph), np.cos(th)]))
    tris = []
    ring = lambda i, j: 2 + (i - 1) * n_lon + (j % n_lon)        # noqa: E731
    for j in range(n_lon):
        tris.append([0, ring(1, j), ring(1, j + 1)])
        tris.append([1, ring(n_lat - 1, j + 1), ring(n_lat - 1, j)])
    for i in range(1, n_lat - 1):
        for j in range(n_lon):
            a, b = ring(i, j), ring(i, j + 1)
            c2, d = ring(i + 1, j), ring(i + 1, j + 1)
            tris.append([a, c2, d])
            tris.append([a, d, b])
    return _orient(Mesh(np.asarray(verts), np.asarray(tris, dtype=np.int64)))


def _polyline(points, per_segment: int = 5) -> np.ndarray:
    """Densify a polyline so a smooth tube can bend along it."""
    pts = [np.asarray(p, dtype=np.float64) for p in points]
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        for k in range(1, per_segment + 1):
            out.append(a + (b - a) * (k / per_segment))
    return np.asarray(out)


def _euler_from_matrix(R) -> tuple[float, float, float]:
    """Euler XYZ degrees (R = Rz Ry Rx) of a 3x3 rotation."""
    ry = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    if abs(np.cos(ry)) > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:                                  # gimbal lock; not reached by wing poses
        rx = np.arctan2(-R[1, 2], R[1, 1])
        rz = 0.0
    return tuple(float(np.degrees(v)) for v in (rx, ry, rz))


def _rotation_about(axis, deg) -> np.ndarray:
    """Rodrigues rotation of ``deg`` degrees about a unit ``axis``."""
    u = np.asarray(axis, dtype=np.float64)
    u = u / np.linalg.norm(u)
    a = np.radians(deg)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


# --------------------------------------------------------------------------
# the shared arm
#
# Bone lengths are quoted RELATIVE TO FOREARM (radius) length -- the standard
# measure in chiropteran morphometrics. Angles are in the wing plane, measured
# from +X (span) toward +Z (trailing edge). Digits are chains: a metacarpal and
# one or two phalanges, each with its own knuckle joint, so a finger can bend.

# Measured on Pteropus (flying fox) skeletons and flight photographs. The arm is
# a Z: the humerus sweeps aft and the forearm sweeps FORWARD, so the wrist is
# the most forward point of the planform. Digits are a metacarpal plus one or
# two phalanges with a small bend at the first knuckle and a strong aft hook at
# the last -- the falcate wingtip. The fan is measured off the forearm axis on a
# flying-fox skeleton (II/III/IV/V = 32/56/87/127) and the roost silhouette
# (the leading edge turns ~60 deg at the wrist and V runs straight aft); digit
# II is held close to III, as in the second skeleton, so it lies inside the
# leading edge instead of standing out as a second horn.
BAT_SPEC = {
    "humerus": (0.72, 25.0),          # (length, absolute angle)
    "forearm": (1.00, -30.0),
    "thumb": (0.28, -50.0),           # bone length, angle off the forearm axis
    "claw": 0.12,                     # claw length beyond the thumb bone
    # name: metacarpal, phalanges, fan angle of the metacarpal off the forearm
    # axis, and the bend (degrees, + = aft) added at each knuckle
    "digits": {
        "II": {"mc": 0.60, "p": (0.13,), "fan": 36.0, "bends": (6.0,)},
        "III": {"mc": 0.71, "p": (0.57, 0.70), "fan": 47.0, "bends": (3.0, 46.0)},
        "IV": {"mc": 0.70, "p": (0.43, 0.33), "fan": 76.0, "bends": (8.0, 30.0)},
        "V": {"mc": 0.73, "p": (0.36, 0.30), "fan": 106.0, "bends": (6.0, 24.0)},
    },
}
# Folding closes the Z of the arm: the humerus tucks back along the BODY (the
# shoulder entry is an offset from lying exactly along the line from shoulder to
# root, so the stowed wing hangs along the back wherever the placement points the
# wing frame), the
# forearm swings FORWARD onto it (negative = toward the leading edge; it starts
# forward of the humerus, so folding it the other way would open the elbow), and
# the hand folds back along the forearm.
BAT_SPEC["fold"] = (-10.0, -100.0, 135.0)   # shoulder offset, elbow, wrist swing at fold=1
# Stowed, the fingers hairpin at their knuckles (MCP +100..150, IP +120..160 in
# the photographs), which is what cuts a folded wing to ~0.15 of its spread.
BAT_SPEC["knuckle_fold"] = (95.0, 85.0)     # MCP, IP swing at fold=1

# A bird's hand is short (carpometacarpus + one finger) and its ulna long; the
# flight surface is feathers rooted along ulna and hand, not skin between fingers.
BIRD_SPEC = {
    "humerus": (0.80, 20.0),
    "forearm": (1.00, -6.0),
    "thumb": (0.18, -35.0),                      # the alula's digit
    # carpometacarpus + one phalanx, 0.58 FA in all; the primaries root along it
    "digits": {"III": {"mc": 0.36, "p": (0.22,), "fan": 15.0, "bends": (3.0,)}},
    "fold": (-10.0, -140.0, 140.0),
    "curl_gain": 0.35,         # the hand flexes as a unit: a curl must not split the fan
}
_NUM = {"II": 2, "III": 3, "IV": 4, "V": 5}


class WingSkeleton:
    """Joint positions of a wing's forelimb in the flat wing plane.

    ``joints`` maps a joint name to its rest position and ``parent`` gives the
    hierarchy, which is exactly what an :class:`~minimaster.core.armature.Armature`
    needs. ``root`` is the end of the wing's attachment line on the body; it is
    the only joint with no parent and, bound to the torso, it never moves with
    the arm. Each digit has a base joint ``dN_cmc`` coincident with the wrist so
    the fan can open and close one finger at a time.

    ``span`` always describes the wing **spread**.
    """

    def __init__(self, span=13.0, sweep=0.0, spec=None, root_chord=1.7, scale=None,
                 tip_pad=0.0, root=None):
        spec = BAT_SPEC if spec is None else spec
        self.fold_angles = spec.get("fold", (0.0, 150.0, 150.0))
        self.knuckle_fold = spec.get("knuckle_fold", (0.0, 0.0))
        self.curl_gain = spec.get("curl_gain", 1.0)
        self.feathers: list[dict] = []
        J: dict[str, np.ndarray] = {}
        P: dict[str, str | None] = {}

        def add(name, pos, parent):
            J[name] = np.asarray(pos, dtype=np.float64)
            P[name] = parent

        add("root", (0.0, 0.0, root_chord), None)
        add("shoulder", (0.0, 0.0, 0.0), "root")
        hum_len, hum_ang = spec["humerus"]
        fore_len, fore_ang = spec["forearm"]
        a_f = fore_ang + sweep
        add("elbow", J["shoulder"] + hum_len * _dir(hum_ang), "shoulder")
        add("wrist", J["elbow"] + fore_len * _dir(a_f), "elbow")
        th_len, th_off = spec["thumb"]
        add("thumb_cmc", J["wrist"], "wrist")
        add("thumb_tip", J["wrist"] + th_len * _dir(a_f + th_off), "thumb_cmc")

        self.fan: dict[str, float] = {}
        self.chains: dict[str, list[str]] = {}         # digit -> joint names
        self.out_angle: dict[str, float] = {}          # joint -> its bone's plan angle
        self.membrane_bones: list[str] = ["shoulder", "elbow", "wrist"]
        for name, d in spec["digits"].items():
            n = _NUM[name]
            cmc = f"d{n}_cmc"
            add(cmc, J["wrist"], "wrist")
            self.fan[name] = d["fan"]
            ang = a_f + d["fan"]
            mcp = f"d{n}_mcp"
            add(mcp, J[cmc] + d["mc"] * _dir(ang), cmc)
            self.out_angle[cmc] = ang
            chain, prev = [cmc, mcp], mcp
            self.membrane_bones.append(mcp)
            last = len(d["p"]) - 1
            for i, (ln, bend) in enumerate(zip(d["p"], d["bends"])):
                self.out_angle[prev] = ang + bend
                ang += bend
                jn = f"d{n}_tip" if i == last else f"d{n}_ip"
                add(jn, J[prev] + ln * _dir(ang), prev)
                chain.append(jn)
                self.membrane_bones.append(jn)
                prev = jn
            self.chains[name] = chain

        # ``span`` describes the wing SPREAD; scale so the reach matches it
        pts = [v for k, v in J.items() if k != "root"]
        # needle fingertips poke past the last joint; ``span`` counts them
        for chain in self.chains.values():
            d = J[chain[-1]] - J[chain[-2]]
            pts.append(J[chain[-1]] + d / max(float(np.linalg.norm(d)), 1e-12) * tip_pad
                       * SPIKE_SCALE.get(next(k for k, v in self.chains.items() if v is chain), 1.0))
        pts = np.array(pts)
        reach = float(pts[:, 0].max() - pts[:, 0].min())
        self.scale = span / max(reach, 1e-9) if scale is None else float(scale)
        self.joints = {k: v * self.scale for k, v in J.items()}
        if root is not None:
            # where the attachment line ends on the body, in wing units (plan x, z)
            self.joints["root"] = np.array([float(root[0]), 0.0, float(root[1])])
        self.parent = P
        self.span = span
        self.fold = 0.0
        self._hum_angle = hum_ang
        self._root_for_angle = root is not None

    @property
    def body_angle(self) -> float:
        """Plan angle of the body line (shoulder -> root) in the wing frame. A
        wing stows along it: that is down the back, not the frame's +Z, which on
        a real placement points down-and-outward."""
        d = self.joints["root"] - self.joints["shoulder"]
        return float(np.degrees(np.arctan2(d[2], d[0])))

    # -- convenience views used by the surface builders -------------------

    @property
    def shoulder(self): return self.joints["shoulder"]
    @property
    def elbow(self): return self.joints["elbow"]
    @property
    def wrist(self): return self.joints["wrist"]
    @property
    def root(self): return self.joints["root"]
    @property
    def thumb(self): return self.joints["thumb_tip"]

    @property
    def digits(self) -> dict[str, np.ndarray]:
        """Digit tips by name (II..V)."""
        return {n: self.joints[c[-1]] for n, c in self.chains.items()}

    @property
    def tip(self) -> np.ndarray:
        return self.digits["III"]

    def chain_points(self, digit: str) -> list[np.ndarray]:
        """Wrist, knuckles and tip of a digit, in order."""
        return [self.joints["wrist"]] + [self.joints[j] for j in self.chains[digit][1:]]

    def lift_joints(self, height_at) -> None:
        """Move every joint to the surface height under it, so the rig's hinge
        axes pass through the bones as they are actually modelled."""
        for v in self.joints.values():
            v[1] = height_at(v[None, [0, 2]])[0]

    def add_joint(self, name: str, pos, parent: str) -> None:
        """Extend the hierarchy (feather roots and tips hang off arm joints)."""
        self.joints[name] = np.asarray(pos, dtype=np.float64)
        self.parent[name] = parent

    def add_feather(self, row: str, root: str, tip: str, parent: str,
                    yaw: float, target: float) -> None:
        """Register a rigged feather: ``root``/``tip`` joints, the arm joint it
        grows from, its rest direction and the direction it folds toward (both
        plan angles from +X toward +Z)."""
        self.feathers.append({"row": row, "root": root, "tip": tip, "parent": parent,
                              "yaw": yaw, "target": target})

    # -- rig controls -----------------------------------------------------

    def _plan_swing(self, pose, joint: str) -> float:
        """Degrees the plan direction of a joint's outgoing bones has turned
        (toward the trailing edge) from the swings of it and its ancestors."""
        total, j = 0.0, joint
        while j is not None:
            total += -pose.get(j, (0.0, 0.0, 0.0))[1]
            j = self.parent[j]
        return total

    def controls_to_pose(self, c: PoseControls) -> dict[str, tuple[float, float, float]]:
        """Joint rotations (wing frame) for the animator-level controls."""
        f = float(np.clip(c.fold, 0.0, 1.0))
        s_off, e_fold, w_fold = self.fold_angles
        # swing the humerus from its rest direction onto the body line
        s_fold = self.body_angle - self._hum_angle + s_off
        pose: dict[str, tuple[float, float, float]] = {}
        pose["shoulder"] = compose(raise_(c.raise_), swing_back(c.swing + s_fold * f),
                                   twist(c.twist))
        pose["elbow"] = swing_back(e_fold * f)
        pose["wrist"] = swing_back(w_fold * f)
        pose["thumb_cmc"] = swing_back(-25.0 * f)
        # the fan closes toward the forearm axis as the wing folds
        s = float(np.clip(c.fan, 0.0, 1.5)) * (1.0 - 0.92 * f)
        for name, phi in self.fan.items():
            n = _NUM[name]
            pose[f"d{n}_cmc"] = swing_back(phi * (s - 1.0))
        if c.curl:
            for name, chain in self.chains.items():
                for jn in chain[1:-1]:                    # mcp and ip knuckles
                    a = self.out_angle[jn]
                    axis = np.array([-np.sin(np.radians(a)), 0.0, np.cos(np.radians(a))])
                    pose[jn] = _euler_from_matrix(_rotation_about(axis, c.curl * self.curl_gain))
        if f and any(self.knuckle_fold):
            # Stowing hairpins each finger at its knuckles (in the wing plane) --
            # but only once the wing is mostly stowed: half-folded, the hand is
            # folded and the fingers closed yet still straight.
            k = float(np.clip((f - 0.45) / 0.55, 0.0, 1.0))
            k = k * k * (3.0 - 2.0 * k)
            for name, chain in self.chains.items():
                for jn, amount in zip(chain[1:-1], self.knuckle_fold):
                    pose[jn] = compose(pose.get(jn, (0.0, 0.0, 0.0)), swing_back(amount * k))
        # Feathers fold like a closing fan: each turns about its own root toward
        # the folded direction, allowing for whatever its parent bone has done.
        for fe in self.feathers:
            now = fe["yaw"] + self._plan_swing(pose, fe["parent"])
            want = fe["yaw"] + (self.body_angle + fe["target"] - fe["yaw"]) * f
            delta = want - now
            if abs(delta) > 1e-9:
                pose[fe["root"]] = swing_back(delta)
        return pose


# --------------------------------------------------------------------------
# membrane (bat / dragon) wing


def _scallop_arc(a, b, toward, depth_frac, k):
    """Points strictly between ``a`` and ``b`` on a concave arc bowed toward
    ``toward`` (the wrist). Depth is proportional to the gap, so every scallop
    stays in scale; the endpoints stay sharp because they are finger tips."""
    d = b - a
    perp = np.array([d[2], 0.0, -d[0]])
    ln = float(np.linalg.norm(perp))
    t = np.linspace(0.0, 1.0, k + 2)[1:-1]
    if ln < 1e-9:
        return [a + d * s for s in t]
    perp = perp / ln
    if float(perp @ (toward - (a + b) / 2.0)) < 0:
        perp = -perp
    depth = depth_frac * float(np.linalg.norm(d))
    return [a + d * s + perp * depth * np.sin(np.pi * s) for s in t]


# Scallop depth is perpendicular sag over the tip-to-tip chord, and it differs
# gap by gap on a real wing (measured on flying-fox photographs): the outer
# panel is taut, the IV-V gap is the deep one, and the edge from V back to the
# body is one long shallow arc. Dragon art exaggerates every gap.
SCALLOP_BAT = {"III-IV": 0.03, "IV-V": 0.26, "V-root": 0.09}
SCALLOP_DRAGON = {"III-IV": 0.40, "IV-V": 0.50, "V-root": 0.30}


def membrane_pieces(span=13.0, thickness=0.16, sweep=0.0, droop=0.0, gull=0.16,
                    camber=CAMBER, scallop=None, root_chord=1.4,
                    billow=0.04, propatagium=True, bones=True, rows=22,
                    spike=0.10, root=None, spec=None):
    """A patagium stretched over a knuckled forelimb, as a skeleton + pieces.

    The sheet is a *fan*: every grid line runs from the arm out to the free
    edge, and the lines that carry the digits run exactly down the bones, so a
    panel's border and its bone coincide and the membrane can be bound to the
    fingers it is stretched between.
    """
    sk = WingSkeleton(span=span, sweep=sweep, spec=spec, root_chord=root_chord,
                      tip_pad=spike if bones else 0.0, root=root)
    D, W, E, S, R = sk.digits, sk.wrist, sk.elbow, sk.shoulder, sk.root
    J_ = sk.joints

    # -- outer boundary, ordered by angle about the wrist ---------------
    if scallop is None:
        depth = SCALLOP_BAT
    elif isinstance(scallop, dict):
        depth = {**SCALLOP_BAT, **scallop}
    else:                                   # one number for every gap
        depth = {"III-IV": scallop, "IV-V": scallop, "V-root": SCALLOP_BAT["V-root"]}
    tail = S + np.array([0.0, 0.0, 0.14 * float(R[2])])        # on the root line
    segments = [                                                # (a, b, interior points, kind)
        # the membrane's leading edge follows digit III's own knuckles out to
        # the tip, so the hooked bone never sticks out in front of the sheet
        (D["II"], D["III"], [J_["d3_mcp"], J_["d3_ip"]], 6, "dist"),
        (D["III"], D["IV"], _scallop_arc(D["III"], D["IV"], W, depth["III-IV"], 7), 8, "dist"),
        (D["IV"], D["V"], _scallop_arc(D["IV"], D["V"], W, depth["IV-V"], 9), 10, "dist"),
        (D["V"], R, _scallop_arc(D["V"], R, W, depth["V-root"], 9), 10, "te"),
        (R, tail, [], 7, "root"),
    ]
    B, kind, sector, sloc = [], [], [], []
    for si, (a, b, mid, nseg, k) in enumerate(segments):
        pts = [a] + list(mid) + [b]
        poly = _resample(np.array(pts), nseg + 1)
        last = si == len(segments) - 1
        take = poly if last else poly[:-1]
        for j, p in enumerate(take):
            B.append(p)
            kind.append(k)
            sector.append(si)
            sloc.append(j / nseg)
    B = np.array(B)
    nu = len(B)
    kind = np.array(kind)

    # -- inner boundary: the wrist for the fingers, then up the arm ------
    arm = [W, E, S]
    arm_len = np.cumsum([0.0] + [np.linalg.norm(arm[i + 1] - arm[i]) for i in range(2)])

    def arm_at(rho):
        d = rho * arm_len[-1]
        i = int(np.clip(np.searchsorted(arm_len, d, side="right") - 1, 0, 1))
        f = (d - arm_len[i]) / max(arm_len[i + 1] - arm_len[i], 1e-12)
        return arm[i] + (arm[i + 1] - arm[i]) * f

    n_back = int((kind == "te").sum() + (kind == "root").sum())
    sigma = np.zeros(nu)
    sigma[kind == "te"] = np.linspace(0.0, 1.0, n_back + 1)[:(kind == "te").sum()]
    sigma[kind == "root"] = np.linspace(0.0, 1.0, n_back + 1)[(kind == "te").sum():][
        :(kind == "root").sum()]
    I = np.array([W if kind[c] == "dist" else arm_at(float(sigma[c]) ** 1.6)
                  for c in range(nu)])
    I[-1] = S
    t0 = np.where(kind == "dist", 0.04, 0.04 * (1.0 - sigma))
    t0[-1] = 0.0

    # -- the sheet -------------------------------------------------------
    n = rows
    tau = np.linspace(0.0, 1.0, n + 1)
    tt = t0[None, :] + (1.0 - t0)[None, :] * tau[:, None]          # (n+1, nu)
    grid = I[None, :, :] + tt[:, :, None] * (B - I)[None, :, :]

    # camber: arch along each ray, strongest in the plagiopatagium where the
    # rays are chordwise; billow: panels bulge between the fingers
    ray = np.linalg.norm(B - I, axis=1)
    w_cam = np.where(kind == "dist", 0.25, 1.0)
    arch = np.sin(np.pi * np.clip(tt, 0, 1) ** CAMBER_PEAK)
    grid[:, :, 1] -= camber * ray[None, :] * w_cam[None, :] * arch
    # sector gap lengths for the billow amplitude
    gaps = {}
    for si, (a, b, _, _, _) in enumerate(segments):
        gaps[si] = float(np.linalg.norm(b - a))
    amp = np.array([billow * gaps[sector[c]] if kind[c] in ("dist", "te") else 0.0
                    for c in range(nu)])
    prof = np.array([np.sin(np.pi * sloc[c]) for c in range(nu)])
    grid[:, :, 1] -= (amp * prof)[None, :] * tt
    # the whole wing arcs downward toward the tip
    grid[:, :, 1] += droop * span * (np.clip(grid[:, :, 0], 0.0, None) / span) ** 2
    # gull wing: the hand panel rises a few degrees off the arm, so the wrist
    # is the low point of the wing (measured +5..+11 deg on flying foxes)
    grid[:, :, 1] -= gull * np.clip(grid[:, :, 0] - float(W[0]), 0.0, None)

    thick = np.maximum(thickness * (1.0 - 0.4 * tt), thickness * 0.3)
    # column range of each panel between two bone lines (the root line, last,
    # is body and never pleats)
    starts = [0]
    for (_, _, _, nseg, _) in segments:
        starts.append(starts[-1] + nseg)
    panels = [(starts[i], starts[i + 1]) for i in range(4)]
    parts: list[Piece] = [Piece("membrane", _slab_from_grid(grid, thick), membrane=True,
                                pleat={"shape": grid.shape[:2], "sectors": panels,
                                       # the long inboard panel hangs as a drape,
                                       # not a tent as deep as the fingers' panels
                                       "gains": [1.0, 1.0, 1.0, 0.45]})]

    # -- propatagium: skin fills the elbow notch in front of the arm -----
    # With the arm a Z (humerus aft, forearm forward) the elbow sits behind the
    # straight line from shoulder to wrist; the propatagium is the triangle
    # between that straight leading edge and the arm.
    if propatagium:
        u = np.linspace(0.0, 1.0, 19)
        arm_pts = _resample(np.array([S, E, W]), 19)
        edge = S[None, :] + (W - S)[None, :] * u[:, None]
        # keep a sliver at the ends so the slab never collapses to a line
        edge = edge + np.array([0.0, 0.0, -0.02 * sk.scale])[None, :] * (
            1.0 - 4.0 * u * (1.0 - u))[:, None]
        pgrid = np.zeros((19, 4, 3))
        for j, f in enumerate(np.linspace(0.0, 1.0, 4)):
            pgrid[:, j, :] = arm_pts + (edge - arm_pts) * f
        pgrid[:, :, 1] += droop * span * (np.clip(pgrid[:, :, 0], 0, None) / span) ** 2
        pgrid[:, :, 1] -= gull * np.clip(pgrid[:, :, 0] - float(W[0]), 0.0, None)
        pthick = np.full(pgrid.shape[:2], thickness * 0.9)
        parts.append(Piece("propatagium", _slab_from_grid(pgrid, pthick), membrane=True))

    # -- lift the skeleton onto the surface, then lay the bones on it ----
    flat_xz = grid[:, :, [0, 2]].reshape(-1, 2)
    flat_y = grid[:, :, 1].reshape(-1)
    sigma_l = 0.045 * span
    sk.lift_joints(lambda q: _lift(flat_xz, flat_y, q, sigma_l))
    # the thumb hangs ventrally: tilted about 30 degrees off the wing plane
    sk.joints["thumb_tip"][1] += 0.5 * float(np.linalg.norm(sk.thumb - sk.wrist))
    if bones:
        parts += _membrane_bones(sk, thickness, spike, (spec or BAT_SPEC).get("claw", 0.12))
    return sk, parts


# a real fingertip is a bare membrane cusp with the bone ending at the edge; the
# longest digit keeps a short point, the others barely poke past
SPIKE_SCALE = {"II": 0.0, "III": 1.0, "IV": 0.45, "V": 0.45}


def _membrane_bones(sk: WingSkeleton, thickness: float, spike: float,
                    CLAW: float = 0.12) -> list[Piece]:
    """Arm and finger bones: tubes through the (lifted) joints, so they read as
    ridges under the skin, plus joint knobs, needle fingertips and the clawed
    thumb.

    Radii are measured on a flying-fox skeleton as fractions of the forearm
    (FA): forearm ~0.053 FA across, digit bases ~0.034 FA tapering to about half.
    A printed bone cannot be that fine, so there is a floor tied to the
    membrane's own thickness -- a bone must stay thicker than the sheet it
    holds up.
    """
    out: list[Piece] = []
    J = sk.joints
    fa = sk.scale                                  # forearm length in wing units
    floor = 0.60 * thickness

    def R(frac):
        return max(frac * fa, floor)

    up = np.array([0.0, -0.06 * thickness, 0.0])   # ridges stand proud on both faces

    def tube(points, r0, r1, per, sides=8):
        pts = _polyline([p + up for p in points], per)
        t = np.linspace(0.0, 1.0, len(pts))
        return _tube(pts, r0 + (r1 - r0) * t, sides)

    # arm: shoulder -> elbow -> wrist as one smooth tube over the two arm bones
    out.append(Piece("arm", tube([J["shoulder"], J["elbow"], J["wrist"]],
                                 R(0.050), R(0.034), 6, 9), chain=("elbow", "wrist")))
    out.append(Piece("wrist_knob", _ball(J["wrist"] + up, 1.25 * R(0.034)), bone="wrist"))
    out.append(Piece("elbow_knob", _ball(J["elbow"] + up, 1.15 * R(0.042)), bone="elbow"))

    for name, chain in sk.chains.items():
        pts = [J["wrist"]] + [J[j] for j in chain[1:]]
        keys = tuple(chain[1:])                    # one bone per segment
        # the fingertip is a needle that pokes a little past the membrane edge
        d = pts[-1] - pts[-2]
        d = d / max(float(np.linalg.norm(d)), 1e-12)
        if SPIKE_SCALE[name] > 0.0 and spike > 0.0:
            pts = pts + [pts[-1] + d * spike * SPIKE_SCALE[name] * fa]
        base = R(0.030) if name != "II" else R(0.024)
        pp = _polyline([p + up for p in pts], 6)
        t = np.linspace(0.0, 1.0, len(pp))
        # taper to about half by the last joint, then to a needle over the spike
        has_spike = SPIKE_SCALE[name] > 0.0 and spike > 0.0
        n_main = 6 * (len(pts) - 1 - (1 if has_spike else 0))
        r = np.linspace(base, 0.5 * base, n_main + 1)
        if has_spike:
            r = np.concatenate([r, np.linspace(0.5 * base, 0.25 * floor, 6 + 1)[1:]])
        out.append(Piece(f"digit_{name}", _tube(pp, np.maximum(r, 0.25 * floor), 7), chain=keys))

    # digit I: free, clawed, and the detail that says "bat" at a glance. It
    # hangs ventrally and ends in a hooked claw.
    w, th = J["wrist"], J["thumb_tip"]
    out.append(Piece("thumb", tube([w, th], R(0.028), R(0.020), 4, 7), bone="thumb_tip"))
    fwd = (th - w) / max(float(np.linalg.norm(th - w)), 1e-9)
    claw_len = CLAW * fa
    mid = th + fwd * 0.55 * claw_len + np.array([0.0, 0.25 * claw_len, 0.0])
    tip = th + fwd * 0.85 * claw_len + np.array([0.0, 0.90 * claw_len, 0.0])
    out.append(Piece("claw", _tube(np.array([th + up, mid + up, tip]),
                                   np.array([R(0.022), 0.6 * R(0.022), 0.15 * R(0.022)]), 7),
                     bone="thumb_tip"))
    return out


def membrane_rig(**kw) -> WingRig:
    """A bat/dragon wing with its skeleton, skin weights and poses."""
    sk, pieces = membrane_pieces(**kw)
    return WingRig(sk, pieces)


def membrane_wing(fold=0.0, **kw) -> Mesh:
    """The membrane wing as one mesh (spread, or folded by posing the rig)."""
    rig = membrane_rig(**kw)
    return rig.pose(fold=fold) if fold else rig.rest_mesh


# --------------------------------------------------------------------------
# feathered (bird / angel) wing
#
# Everything here is from measurements of eagle, galah and flamingo wings,
# quoted relative to the forearm (FA). Primaries are numbered ornithologically:
# P1 is the innermost, at the wrist; P10 the outermost. The longest, P7/P8, is
# about 1.65 FA; P10 is only ~0.7 of it, which is what makes a soaring wing's
# tip a rounded point instead of a blade.

# length as a fraction of the longest primary, P1..P10
PRIMARY_RATIO = (0.60, 0.68, 0.76, 0.83, 0.90, 0.97, 1.00, 1.00, 0.92, 0.70)
LONGEST_PRIMARY = 1.65                    # FA
# vane width / length: the broad inner primaries down to the narrow outer ones
PRIMARY_WIDTH = (0.20, 0.19, 0.18, 0.16, 0.15, 0.14, 0.13, 0.13, 0.11, 0.105)
SECONDARY_RATIO = (0.62, 0.65, 0.67, 0.69, 0.70, 0.70, 0.70, 0.70, 0.68, 0.66,
                   0.64, 0.62, 0.61, 0.60)
TERTIAL_RATIO = (0.60, 0.62, 0.58, 0.55)
FA_P = LONGEST_PRIMARY


def _bump(t, centre, sigma):
    return np.exp(-((t - centre) / sigma) ** 2)


def _blade(length, width, thickness, curl=0.10, cup=0.16, sweep=0.08,
           lead_frac=0.30, tip=0.74, tip_pow=1.8, emarg=0.0, emarg_in=0.0,
           calamus=0.10, tip_floor=0.07, n=11, m=8) -> Mesh:
    """One feather: a thin curved blade with a raised shaft down its middle.

    A bare quill (the calamus) runs the first tenth of the length, then the
    vane widens to its maximum by ~40% and holds nearly parallel-sided until the
    tip, where it closes in an ogive. ``lead_frac`` is the share of the vane in
    front of the shaft: the narrow leading vane against the broad trailing one
    is what makes a flight feather aerodynamic, and it shows in silhouette.
    ``tip_floor`` is the width (fraction of ``width``) the tip closes down to.
    ``emarg`` notches the leading vane near the tip (and ``emarg_in`` the
    trailing one a little earlier): the slotted look of soaring primaries that
    lets the outer feathers separate into fingers. Because the vane stays broad
    to the tip, neighbours overlap along their length instead of touching at a
    point.
    """
    t = np.linspace(0.0, 1.0, n + 1)
    rise = np.clip((t - calamus) / (0.42 - calamus), 0.0, 1.0) ** 0.8
    ogive = np.clip(1.0 - (np.clip(t - tip, 0.0, None) / (1.0 - tip)) ** tip_pow,
                    0.0, 1.0) ** 0.5
    # the quill end stays thin, but the tip closes onto a rounded cap of width
    # ``tip_floor`` rather than a point: a real feather ends in a curve, and a
    # point read as a saw along the trailing edge
    floor = np.where(t > 0.5, tip_floor, 0.07)
    W = width * np.maximum(rise * ogive, floor)
    wl = lead_frac * W * (1.0 - emarg * _bump(t, 0.62, 0.08))
    wt = (1.0 - lead_frac) * W * (1.0 - emarg_in * _bump(t, 0.56, 0.06))
    v = np.linspace(-1.0, 1.0, m + 1)
    z = np.where(v[None, :] < 0, v[None, :] * wl[:, None], v[None, :] * wt[:, None])
    grid = np.zeros((n + 1, m + 1, 3))
    grid[:, :, 0] = (length * t)[:, None]
    grid[:, :, 2] = z + (sweep * length * t ** 2)[:, None]
    # the shaft droops along its length and the vanes cup away from it
    grid[:, :, 1] = ((curl * length * t ** 2)[:, None]
                     + cup * W[:, None] * np.abs(v)[None, :] ** 1.5)
    # raised along the shaft, thin toward the vane edges and the tip
    th = thickness * (1.0 - 0.68 * np.abs(v))[None, :] * (1.0 - 0.45 * t)[:, None]
    return _slab_from_grid(grid, np.maximum(th, thickness * 0.22))


def _place_feather(mesh: Mesh, root, yaw_deg, roll_deg=0.0) -> Mesh:
    """Rotate a feather about its root: yaw in the wing plane, then roll."""
    a = np.radians(yaw_deg)
    ry = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
    b = np.radians(roll_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    m = np.eye(4)
    m[:3, :3] = ry @ rx
    m[:3, 3] = root
    return mesh.transform(m)


def _along(points, fractions):
    """Points at the given fractions of a polyline's length."""
    p = np.asarray(points, dtype=np.float64)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    out = []
    for f in np.atleast_1d(fractions):
        d = float(f) * s[-1]
        i = int(np.clip(np.searchsorted(s, d, side="right") - 1, 0, len(seg) - 1))
        u = (d - s[i]) / max(seg[i], 1e-12)
        out.append(p[i] + (p[i + 1] - p[i]) * u)
    return np.array(out)


def feathered_pieces(span=13.0, thickness=0.20, sweep=0.0, droop=0.10,
                     primaries=10, secondaries=14, tertials=7, scapulars=10,
                     coverts=True, underwing=True, bones=True, root=None, spec=None):
    """A bird wing as skeleton + pieces: every flight feather (and the scapulars
    and alula) has its own root and tip joint, so the fan opens and closes and
    the whole plumage folds together; each covert is bound to the feather it
    lies over."""
    spec = BIRD_SPEC if spec is None else spec
    # span is the full tip-to-shoulder reach INCLUDING the feathers, so measure
    # the plumage at unit scale and rescale to land on the request.
    unit = _feather_layout(WingSkeleton(span=1.0, sweep=sweep, spec=spec, scale=1.0),
                           primaries, secondaries, tertials, scapulars)
    reach = max(f["tip"][0] for f in unit)
    sk = WingSkeleton(span=span, sweep=sweep, spec=spec, scale=span / max(reach, 1e-9),
                      root=root)
    s = sk.scale
    layout = _feather_layout(sk, primaries, secondaries, tertials, scapulars)

    def height(x):
        """The wing arcs downward toward the tip."""
        return droop * span * (np.maximum(x, 0.0) / max(span, 1e-9)) ** 2

    # Shingling comes from ROLL (each feather's leading vane tilts up and its
    # trailing vane down, so it rides over its outboard neighbour), which keeps
    # every root on the arm. The vertical step only settles depth order.
    step = 0.05 * thickness
    parts: list[Piece] = []
    bone_of: dict[tuple, str] = {}
    for fe in layout:
        root_pt = fe["root"].copy()
        root_pt[1] = height(root_pt[0]) - fe["layer"] * step
        tip = root_pt + _dir(fe["yaw"]) * fe["length"]
        tip[1] = height(tip[0]) - fe["layer"] * step
        rj, tj = f"{fe['row']}{fe['i']:02d}_r", f"{fe['row']}{fe['i']:02d}_t"
        sk.add_joint(rj, root_pt, fe["parent"])
        sk.add_joint(tj, tip, rj)
        sk.add_feather(fe["row"], rj, tj, fe["parent"], fe["yaw"], fe["fold_to"])
        blade = _blade(fe["length"], fe["width"], thickness,
                       curl=0.16 if fe["row"] == "primary" else 0.08,
                       lead_frac=fe["lead"], tip=fe["tip_at"], tip_pow=fe["tip_pow"],
                       emarg=fe["emarg"], emarg_in=fe["emarg_in"], tip_floor=fe["tip_floor"],
                       n=16)
        parts.append(Piece(f"{fe['row']}{fe['i']:02d}",
                           _place_feather(blade, root_pt, fe["yaw"], fe["roll"]), bone=tj))
        bone_of[(fe["row"], fe["i"])] = tj
    if coverts:
        parts += _coverts(sk, layout, bone_of, thickness, height, step, underwing)
    if bones:
        parts += _feather_bones(sk, height, s)
    return sk, parts


def _feather_layout(sk: WingSkeleton, n_pri: int, n_sec: int, n_ter: int, n_sca: int = 10):
    """Where each flight feather roots, which way it points and how big it is.

    Everything is in the wing plane (plan angles from +X toward +Z), relative to
    the forearm, then scaled by ``sk.scale``. Primaries root along the hand with
    about 65% overlap and fan from P1 (+70 deg) to P10 (+8 deg), which continues
    the hand axis; secondaries root along the ulna's trailing side at +78..+100
    deg, so the flight-feather row is one continuous scalloped edge with no step
    where primaries meet secondaries. Tertials and scapulars on the humerus fair
    the trailing edge into the back; the alula sits on the thumb. All of them are
    rigged feathers, so a fold re-aims every one.
    """
    s = sk.scale
    J = sk.joints
    out = []
    # layer = how far above the wing's base plane a feather sits. Dorsal rule:
    # each feather lies OVER its outboard neighbour, and the stack runs
    # tertials > inner secondaries > outer secondaries > P1 > ... > P10.
    layer = 0

    def common(**kw):
        d = dict(tip_at=0.74, tip_pow=1.8, emarg=0.0, emarg_in=0.0, roll=-3.0,
                 tip_floor=0.25, fold_to=4.0)
        d.update(kw)
        return d

    hand = [J["wrist"], J["d3_mcp"], J["d3_tip"]]
    fr = np.linspace(0.10, 1.0, n_pri)
    roots = _along(hand, fr) + np.array([0.0, 0.0, 0.03 * s])     # calamus on the trailing side
    for i in range(n_pri - 1, -1, -1):                            # outermost first = lowest
        k = int(round(i * 9 / max(n_pri - 1, 1)))
        L = PRIMARY_RATIO[k] * FA_P * s
        outer = k >= 5
        parent = "wrist" if fr[i] <= 0.5 else "d3_mcp"
        out.append(common(row="primary", i=i + 1, root=roots[i], parent=parent,
                          yaw=70.0 - 62.0 * i / max(n_pri - 1, 1),
                          length=L, width=PRIMARY_WIDTH[k] * L,
                          lead=0.27 if outer else 0.37,
                          emarg=0.7 if k >= 4 else 0.0, emarg_in=0.3 if k >= 6 else 0.0,
                          tip_at=0.78, tip_pow=2.0, tip_floor=0.12, fold_to=2.0,
                          roll=-6.0 - 8.0 * (1.0 - i / max(n_pri - 1, 1)), layer=layer))
        layer += 1
    fr = np.linspace(0.0, 1.0, n_sec)
    roots = _along([J["wrist"], J["elbow"]], fr) + np.array([0.0, 0.0, 0.06 * s])
    for i in range(n_sec):
        k = int(round(i * (len(SECONDARY_RATIO) - 1) / max(n_sec - 1, 1)))
        L = SECONDARY_RATIO[k] * FA_P * s
        out.append(common(row="secondary", i=i + 1, root=roots[i], parent="elbow",
                          yaw=78.0 + 22.0 * fr[i], length=L, width=0.23 * L,
                          lead=0.40, tip_at=0.82, tip_pow=3.2, tip_floor=0.30, fold_to=4.0,
                          roll=-9.0 - 3.0 * fr[i], layer=layer))
        layer += 1
    # tertials: seven, overlapping ~70%, shortening smoothly so the trailing edge
    # curves up into the back instead of ending in a few flaps
    fr = np.linspace(0.05, 0.92, n_ter)
    roots = _along([J["elbow"], J["shoulder"]], fr) + np.array([0.0, 0.0, 0.08 * s])
    ratio = np.linspace(0.62, 0.48, n_ter)
    for i in range(n_ter):
        L = ratio[i] * FA_P * s
        out.append(common(row="tertial", i=i + 1, root=roots[i], parent="shoulder",
                          yaw=100.0 + 18.0 * i / max(n_ter - 1, 1), length=L, width=0.27 * L,
                          lead=0.40, tip_at=0.78, tip_pow=2.2, tip_floor=0.30, fold_to=6.0,
                          roll=-10.0, layer=layer))
        layer += 1
    # scapulars: soft feathers over the humerus root that fair the wing into the back
    for i, f in enumerate(np.linspace(0.05, 0.60, n_sca)):
        root = J["shoulder"] + (J["elbow"] - J["shoulder"]) * f + np.array([0.0, 0.0, 0.12 * s])
        L = 0.74 * s
        out.append(common(row="scapular", i=i + 1, root=root, parent="shoulder",
                          yaw=100.0 + 30.0 * i / max(n_sca - 1, 1), length=L, width=0.25 * L,
                          lead=0.40, tip_at=0.74, tip_pow=2.2, tip_floor=0.28, fold_to=8.0,
                          roll=-9.0, layer=layer))
        layer += 1
    # alula: three small stiff feathers on the thumb
    for i, ln in enumerate((0.30, 0.41, 0.50)):
        root = J["wrist"] + (J["thumb_tip"] - J["wrist"]) * (0.35 + 0.3 * i)
        out.append(common(row="alula", i=i + 1, root=root, parent="thumb_cmc",
                          yaw=22.0 - 5.0 * i, length=ln * s, width=0.22 * ln * s,
                          lead=0.30, tip_at=0.70, tip_pow=2.0, tip_floor=0.2, fold_to=15.0,
                          roll=-6.0, layer=layer))
        layer += 1
    for fe in out:
        fe["tip"] = fe["root"] + _dir(fe["yaw"]) * fe["length"]
    return out


def _jit(i, amount=0.08):
    """Deterministic +-amount length jitter, so a tier's lower edge is not a ruled line."""
    return 1.0 + amount * float(np.sin(7.3 * i + 1.1))


def _coverts(sk, layout, bone_of, thickness, height, step, underwing=True) -> list[Piece]:
    """The shingles over the flight feathers' bases. Each covert is welded to the
    bone of the feather it lies beside, so it fans with it; the leading-edge
    rows ride on the nearest secondary or tertial.

    Tiers, from the trailing edge forward (each lies over the one behind it):
    primary coverts, greater, median, then lesser rows. A tier is STAGGERED
    against the one behind it -- each covert straddles two flight feathers, so
    the rows brick rather than stack into columns -- and ends in a rounded tip.
    ``underwing`` adds the ventral covert field, which is what an angel seen from
    the front shows.
    """
    out: list[Piece] = []
    s = sk.scale
    J = sk.joints
    by = {(f["row"], f["i"]): f for f in layout}
    n_sec = sum(1 for f in layout if f["row"] == "secondary")
    n_pri = sum(1 for f in layout if f["row"] == "primary")
    sec_pitch = (J["elbow"] - J["wrist"]) / max(n_sec - 1, 1)
    pri_pitch = (J["d3_tip"] - J["wrist"]) * 0.9 / max(n_pri - 1, 1)

    def covert(name, fe, length, width_over_len, fwd, tier, bone, pitch, stagger, yaw_off,
               ventral=False, tip_floor=0.30):
        root = fe["root"] + pitch * stagger + np.array([0.0, 0.0, -fwd * s])
        if ventral:
            root[1] = height(root[0]) + tier * 0.6 * thickness
        else:
            root[1] = height(root[0]) - (fe["layer"] + tier * 0.7 * thickness / step) * step
        length = length * _jit(fe["i"] + 3 * tier)
        blade = _blade(length, width_over_len * length, thickness * 0.9, curl=0.05,
                       lead_frac=0.33, tip=0.70, tip_pow=2.6, calamus=0.04,
                       tip_floor=tip_floor, n=10, m=6)
        roll = (-fe["roll"] + 3.0) if ventral else (fe["roll"] - 3.0)
        out.append(Piece(name, _place_feather(blade, root, fe["yaw"] + yaw_off, roll), bone=bone))

    for (row, i), fe in by.items():
        b = bone_of.get((row, i))
        if b is None:
            continue
        for ventral in ((False, True) if underwing else (False,)):
            tag = "under_" if ventral else ""
            if row == "primary":
                k = i - 1
                frac = max(0.30, 0.42 - 0.03 * k)
                covert(f"{tag}primary_covert{i:02d}", fe, frac * FA_P * s, 0.34, 0.02, 1, b,
                       pri_pitch, 0.5, 2.0, ventral)
            elif row == "secondary":
                covert(f"{tag}greater{i:02d}", fe, 0.375 * FA_P * s, 0.55, 0.05, 1, b,
                       sec_pitch, 0.5, 3.0, ventral)
                if not ventral:
                    covert(f"median{i:02d}", fe, 0.22 * FA_P * s, 0.70, 0.06, 2, b,
                           sec_pitch, 0.25, 6.0, ventral, tip_floor=0.35)
    # lesser / marginal coverts: scale-like rows along the arm. Their roots sit
    # just BEHIND the bone line and their tips point straight back, so the arm is
    # buried under rounded overlapping scales and no quill ends poke forward.
    arm = [J["shoulder"], J["elbow"], J["wrist"]]
    host = [f for f in layout if f["row"] in ("secondary", "tertial")]
    for r, (length, tier) in enumerate(((0.14, 3), (0.12, 4), (0.10, 5))):
        n = 18 - 2 * r
        pts = _along(arm, np.linspace(0.02, 0.98, n) + (0.5 / n) * (r % 2))
        for j, p in enumerate(pts):
            root = p + np.array([0.0, 0.0, 0.02 * s - 0.05 * s * r])
            root[1] = height(root[0]) - tier * 0.7 * thickness
            hf = min(host, key=lambda f: float(np.linalg.norm(f["root"] - root)))
            blade = _blade(length * FA_P * s * _jit(j + 5 * r), 0.7 * length * FA_P * s,
                           thickness * 0.8, curl=0.03, lead_frac=0.4, tip=0.70, tip_pow=2.4,
                           calamus=0.0, tip_floor=0.35, n=8, m=5)
            out.append(Piece(f"lesser{r}_{j:02d}", _place_feather(blade, root, 90.0, -12.0),
                             bone=bone_of[(hf["row"], hf["i"])]))
    return out


def _feather_bones(sk, height, s) -> list[Piece]:
    """The arm under the plumage. Buried by the lesser coverts, but it gives the
    wing a rounded leading edge and the rig something to hold."""
    J = sk.joints
    out = []

    def tube(points, r0, r1, per=5):
        pts = _polyline([np.array([p[0], height(p[0]) - 0.02 * s, p[2]]) for p in points], per)
        t = np.linspace(0.0, 1.0, len(pts))
        return _tube(pts, r0 + (r1 - r0) * t, 9)

    out.append(Piece("arm", tube([J["shoulder"], J["elbow"], J["wrist"]], 0.052 * s,
                                 0.038 * s, 6), chain=("elbow", "wrist")))
    out.append(Piece("hand", tube([J["wrist"], J["d3_mcp"], J["d3_tip"]], 0.030 * s,
                                  0.012 * s, 5), chain=("d3_mcp", "d3_tip")))
    out.append(Piece("wrist_knob", _ball(J["wrist"] + np.array([0, -0.02 * s, 0]), 0.045 * s),
                     bone="wrist"))
    return out


def feathered_rig(**kw) -> WingRig:
    """A bird/angel wing with its skeleton, skin weights and poses."""
    sk, pieces = feathered_pieces(**kw)
    return WingRig(sk, pieces)


def feathered_wing(fold=0.0, **kw) -> Mesh:
    """The feathered wing as one mesh (spread, or folded by posing the rig)."""
    rig = feathered_rig(**kw)
    return rig.pose(fold=fold) if fold else rig.rest_mesh


# --------------------------------------------------------------------------

STYLES = {"membrane": membrane_wing, "feathered": feathered_wing}
RIGS = {"membrane": membrane_rig, "feathered": feathered_rig}


def make_wing(style: str = "membrane", **kw) -> Mesh:
    """Build a wing. ``style`` is ``"membrane"`` (bat/dragon) or ``"feathered"``."""
    builder = STYLES.get(style)
    if builder is None:
        raise KeyError(f"unknown wing style {style!r}; have {sorted(STYLES)}")
    return builder(**kw)


def make_wing_rig(style: str = "membrane", **kw) -> WingRig:
    """Build a wing together with its skeleton, skin weights and poses."""
    builder = RIGS.get(style)
    if builder is None:
        if style in STYLES:
            raise NotImplementedError(f"no rig for style {style!r} yet")
        raise KeyError(f"unknown wing style {style!r}; have {sorted(STYLES)}")
    return builder(**kw)
