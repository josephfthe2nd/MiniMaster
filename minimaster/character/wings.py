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
from .wingrig import (Piece, PoseControls, WingRig, compose, grid_normals,
                      raise_, swing_back, twist)

DEFAULT_SPAN = 13.0         # wing units; one MakeHuman unit is a decimetre
CAMBER = 0.06               # of chord, at rest; a gliding bat reaches 0.12 (armwing 0.118-0.131)
CAMBER_PEAK = 0.72          # exponent placing max camber near 30% chord
# The hand panels are stretched tighter than the arm panel, so they carry a
# fraction of its camber: 0.42 x 6% = 2.5% of the ray (1-2% was invisible).
HAND_CAMBER = 0.42
BILLOW = 0.06               # panel bulge between two bones, as a fraction of the gap between their tips
PLAGIO_BILLOW = 0.62        # the loose arm panel pillows less per unit gap than a taut finger panel
WASHOUT = 5.0               # degrees: the outer wing is twisted leading-edge-down at the tip


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


def _slab_from_grid(grid: np.ndarray, thickness: np.ndarray) -> Mesh:
    """Thicken a surface grid into a closed solid.

    ``grid`` is (n+1, m+1, 3) mid-surface points, ``thickness`` an (n+1, m+1)
    field. Emits a top sheet, a bottom sheet, and a rim stitching their shared
    perimeter, wound so every edge is used exactly twice.
    """
    n1, m1, _ = grid.shape
    n, m = n1 - 1, m1 - 1
    nrm = grid_normals(grid)
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


def _surface_height(flat_xz, flat_y, query_xz, k: int = 4):
    """Height of a sampled surface under plan points: inverse-square weighting of
    the ``k`` nearest grid vertices. Exact where a point sits on a vertex (the
    bone columns of the membrane do), and local, so a bone follows the skin
    instead of floating on a smoothed copy of it (a 0.6-unit Gaussian let digit IV
    break through the underside as a slit)."""
    q = np.atleast_2d(query_xz)
    d2 = ((q[:, None, :] - flat_xz[None, :, :]) ** 2).sum(axis=2)
    idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
    dk = np.take_along_axis(d2, idx, axis=1)
    w = 1.0 / (dk + 1e-9)
    return (w * flat_y[idx]).sum(axis=1) / w.sum(axis=1)


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


def _then(first, second) -> tuple[float, float, float]:
    """Euler form of "``first``, then ``second``" (either may be ``None``): the
    matrix product, not the sum of the angles."""
    if first is None:
        return tuple(second)
    return _euler_from_matrix(m3.euler_rotation(*second) @ m3.euler_rotation(*first))


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
    "thumb": (0.22, -50.0),           # bone length, angle off the forearm axis
    "claw": 0.14,                     # claw length beyond the thumb bone: a blunt peg-like hook
    # name: metacarpal, phalanges, fan angle of the metacarpal off the forearm
    # axis, and the bend (degrees, + = aft) added at each knuckle
    "digits": {
        "II": {"mc": 0.50, "p": (0.13,), "fan": 46.0, "bends": (8.0,)},
        "III": {"mc": 0.71, "p": (0.57, 0.70), "fan": 50.0, "bends": (14.0, 34.0)},
        "IV": {"mc": 0.70, "p": (0.44, 0.46), "fan": 86.0, "bends": (8.0, 16.0)},
        "V": {"mc": 0.73, "p": (0.40, 0.34), "fan": 106.0, "bends": (6.0, 24.0)},
    },
}
# Folding closes the Z of the arm: the humerus tucks back along the BODY (the
# shoulder entry is an offset from lying exactly along the line from shoulder to
# root, so the stowed wing hangs along the back wherever the placement points the
# wing frame), the
# forearm swings FORWARD onto it (negative = toward the leading edge; it starts
# forward of the humerus, so folding it the other way would open the elbow), and
# the hand folds back along the forearm.
BAT_SPEC["fold"] = (-10.0, -110.0, 172.0)   # shoulder offset, elbow, wrist swing at fold=1
# Stowed, the fingers fold at their knuckles like a concertina: each phalanx lies
# back along the one before (the full 180 deg less the rest bend: at 0.90 every knuckle
# was left ~17 deg open and the Z of one finger crossed the Z of the next), so metacarpal,
# P1 and P2 end up one rod along the forearm -- which is what cuts a folded wing to ~0.15
# of its spread. (The photographs show MCP +100..150 and IP +120..160.)
BAT_SPEC["knuckle_fold"] = 1.0              # fraction of a full concertina at each knuckle
BAT_SPEC["fold_lift"] = 12.0                # degrees a stowed wing is lifted off its (buried) plane
BAT_SPEC["stow_fan"] = {"II": -12.0, "III": -9.0, "IV": -6.0, "V": -3.0}
BAT_SPEC["stow_tilt"] = {"II": -8.0, "III": 8.0, "IV": -8.0, "V": 8.0}
BAT_SPEC["fan_range"] = (0.55, 1.2)         # the sheet cannot take a fan closed much past this

# A bird's hand is short (carpometacarpus + one finger) and its ulna long; the
# flight surface is feathers rooted along ulna and hand, not skin between fingers.
BIRD_SPEC = {
    "humerus": (0.80, 20.0),
    "forearm": (1.00, -6.0),
    "thumb": (0.18, -35.0),                      # the alula's digit
    # carpometacarpus + one phalanx, 0.58 FA in all; the primaries root along it
    "digits": {"III": {"mc": 0.36, "p": (0.22,), "fan": 15.0, "bends": (3.0,)}},
    "fold": (-10.0, -140.0, 172.0),
    "fold_lift": 20.0,         # a feathered wing is deep: it needs to stand off the back further
    "curl_gain": 0.35,         # the hand flexes as a unit: a curl must not split the fan
    # A wing placed on a body (WingSkeleton.stow_against) stows FLAT on the back rather than
    # as a plank leaning off the plane it was mounted in:
    "stow_flat": True,
    "stow_lift": 0.0,          # the humerus's own lift once the plane lies on the back
    "stow_toe_in": 8.0,        # the wing's axis turns toward the spine: a pair stows in a V
    "stow_lean": 6.0,          # ... and its lower end stands off the back, its top sinks in (the weld)
    "stow_pitch": 8.0,         # flight feathers lift their tips off the buttocks (camber)
    "stow_taper": 8.0,         # edge feathers turn toward the middle: broad covert mass, narrow tips
    "stow_follow": 0.3,        # feathers follow their bone until this much of the fold is done
    "curl_cup": 0.35,          # a hand curl cups the primaries: outermost droop 0.35 x curl
    "raise_arch": 0.30,        # a raised wing arches: the wrist hangs 0.3 x raise below the line
}
_NUM = {"II": 2, "III": 3, "IV": 4, "V": 5}


# The fingers stay straight while the wing closes (the hand folds back, the fan
# collapses, the membrane pleats between straight fingers) and only fold up at their
# knuckles over the last stretch: a finger folding in the wing plane passes through
# an L, a U and a spiral on its way to the Z, and a membrane cannot follow that.
KNUCKLE_FOLD_START = 0.75


def knuckle_phase(f: float) -> float:
    """How far the knuckle concertina has gone (0 straight .. 1 stowed) at fold ``f``."""
    k = float(np.clip((f - KNUCKLE_FOLD_START) / (1.0 - KNUCKLE_FOLD_START), 0.0, 1.0))
    return k * k * (3.0 - 2.0 * k)


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
        self.knuckle_fold = float(spec.get("knuckle_fold", 0.0))
        self.bends = {n: list(d["bends"]) for n, d in spec["digits"].items()}
        self.curl_gain = spec.get("curl_gain", 1.0)
        self.fold_lift = float(spec.get("fold_lift", 0.0))
        # where each metacarpal points when stowed (degrees off the forearm axis, same
        # convention as the digit's rest ``fan``); a digit not listed closes to 8% of its
        # rest angle. Deliberate, so a stowed hand is a row of parallel rods with daylight
        # between them and not four rods that happen to coincide.
        self.stow_fan = {k: float(v) for k, v in spec.get("stow_fan", {}).items()}
        # ... and how far each is tilted out of the wing plane then (degrees, + toward
        # the underside): alternate fingers sit above and below the plane, so a closed
        # hand is a zig-zag stack, not a flat row, and fits in half the width
        self.stow_tilt = {k: float(v) for k, v in spec.get("stow_tilt", {}).items()}
        # Stowing against a body (see stow_against): lift of the humerus once the plane lies
        # flat on the back, and how far the wing's axis turns toward the spine
        self.stow_lift = float(spec.get("stow_lift", self.fold_lift))
        self.stow_toe_in = float(spec.get("stow_toe_in", 0.0))
        self.stow_lean = float(spec.get("stow_lean", 0.0))
        self.stow_pitch = float(spec.get("stow_pitch", 0.0))
        self.stow_taper = float(spec.get("stow_taper", 0.0))
        self.stow_follow = float(spec.get("stow_follow", 0.0))
        self.stow_flat = bool(spec.get("stow_flat", False))
        self.curl_cup = float(spec.get("curl_cup", 0.0))
        self.raise_arch = float(spec.get("raise_arch", 0.0))
        self.stow_axes: tuple | None = None
        # skin tears long before a feather does, so a membrane's fan is clamped
        self.fan_range = tuple(spec.get("fan_range", (0.0, 1.5)))
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

    def stow_against(self, out, down, side) -> None:
        """Tell the wing where the back it is stowed against is, as unit vectors in the
        WING frame: ``out`` the back's outward normal, ``down`` along the spine, ``side``
        away from the spine. A stowed wing then pivots about its root until its plane lies
        flat on that back (the wing is mounted at an angle so the SPREAD wing sweeps behind
        the shoulder; folded, that same lean would be a plank standing off the back) and
        its axis turns ``stow_toe_in`` degrees toward the spine, so a pair stows in a V."""
        self.stow_axes = tuple(np.asarray(v, dtype=np.float64) / np.linalg.norm(v)
                               for v in (out, down, side))

    @property
    def stows_flat(self) -> bool:
        """True once the wing knows its back (``stow_against``) and its spec stows flat."""
        return self.stow_flat and self.stow_axes is not None

    def _stow_rotation(self, f: float) -> np.ndarray | None:
        """Rotation (about the root, wing frame) that carries the spread wing a fraction
        ``f`` of the way onto its stowed attitude, or ``None`` without a back."""
        if not self.stows_flat or f <= 0.0:
            return None
        out, down, side = self.stow_axes
        t = np.radians(self.stow_toe_in)
        axis = down - float(down @ out) * out
        axis /= np.linalg.norm(axis)
        lateral = side - float(side @ out) * out
        lateral /= np.linalg.norm(lateral)
        b0 = np.cos(t) * axis - np.sin(t) * lateral           # down the back, toed in
        lean = np.radians(self.stow_lean)
        # the plane leans: its lower end stands off the back (the spine arches out at the
        # shoulder blades and the buttocks, and a flat plank has to clear both)
        b_t = np.cos(lean) * b0 + np.sin(lean) * out          # the stowed wing's axis
        out = np.cos(lean) * out - np.sin(lean) * b0
        b = self.joints["root"] - self.joints["shoulder"]
        b = b / np.linalg.norm(b)
        u = np.array([0.0, -1.0, 0.0])                        # the wing's upper side
        F0 = np.stack([b, u, np.cross(b, u)], axis=1)
        F1 = np.stack([b_t, out, np.cross(b_t, out)], axis=1)
        Q = F1 @ F0.T
        ang = np.arccos(np.clip((np.trace(Q) - 1.0) / 2.0, -1.0, 1.0))
        if ang < 1e-9:
            return None
        k = np.array([Q[2, 1] - Q[1, 2], Q[0, 2] - Q[2, 0], Q[1, 0] - Q[0, 1]])
        if np.linalg.norm(k) < 1e-9:                          # 180 degrees: pick the axis from Q + I
            k = np.sqrt(np.maximum((np.diag(Q) + 1.0) / 2.0, 0.0))
        return _rotation_about(k, np.degrees(ang) * f)

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

    def _heading_after(self, pose, joint: str, heading_deg: float) -> float:
        """Plan heading (degrees from +X toward +Z) that a direction at rest
        ``heading_deg`` has once the rotations of ``joint`` and all its ancestors
        are applied. Read from the accumulated rotation matrix, not from euler
        components: a 172-degree wrist swing has no unique euler form."""
        chain, j = [], joint
        while j is not None:
            chain.append(j)
            j = self.parent[j]
        R = np.eye(3)
        for name in reversed(chain):                       # root first
            R = R @ m3.euler_rotation(*pose.get(name, (0.0, 0.0, 0.0)))
        a = np.radians(heading_deg)
        v = R @ np.array([np.cos(a), 0.0, np.sin(a)])
        return float(np.degrees(np.arctan2(v[2], v[0])))

    def _stow_lateral(self, pose) -> dict[str, float]:
        """Where each flight feather's root sits across the stowed wing: -1 .. +1 about the
        middle of the bundle, along the in-plane perpendicular of the body line. Feathers at
        the edges are turned toward the middle, so the stowed wing tapers from the broad
        covert mass to a narrow tip instead of hanging as parallel columns."""
        R: dict[str, np.ndarray] = {}
        P: dict[str, np.ndarray] = {}
        rest = self.joints
        pos: dict[str, float] = {}
        ba = np.radians(self.body_angle)
        perp = np.array([-np.sin(ba), 0.0, np.cos(ba)])
        want = [fe for fe in self.feathers if fe["row"] != "alula"]
        parents = {fe["parent"] for fe in want}

        def frame(j):
            if j in R:
                return
            par = self.parent[j]
            if par is None:
                R[j], P[j] = np.eye(3), rest[j].copy()
                return
            frame(par)
            R[j] = R[par] @ m3.euler_rotation(*pose.get(j, (0.0, 0.0, 0.0)))
            P[j] = P[par] + R[par] @ (rest[j] - rest[par])

        for j in parents:
            frame(j)
        for fe in want:
            par = fe["parent"]
            p = P[par] + R[par] @ (rest[fe["root"]] - rest[par])
            pos[fe["root"]] = float(p @ perp)
        vals = np.array(list(pos.values()))
        mid, half = float((vals.max() + vals.min()) / 2.0), max(float(np.ptp(vals)) / 2.0, 1e-9)
        return {k: (v - mid) / half for k, v in pos.items()}

    def controls_to_pose(self, c: PoseControls) -> dict[str, tuple[float, float, float]]:
        """Joint rotations (wing frame) for the animator-level controls."""
        f = float(np.clip(c.fold, 0.0, 1.0))
        s_off, e_fold, w_fold = self.fold_angles
        # swing the humerus from its rest direction onto the body line
        s_fold = self.body_angle - self._hum_angle + s_off
        pose: dict[str, tuple[float, float, float]] = {}
        Q = self._stow_rotation(f)
        if Q is not None:
            pose["root"] = _euler_from_matrix(Q)
        # The shoulder: lift about the axis perpendicular to the arm AS IT IS NOW
        # (so a stowed wing can be raised too -- about the frame's Z it could not),
        # i.e. swing about Y after rotating about Z, then roll about the span axis.
        swing_total = c.swing + s_fold * f
        # A stowed wing is also lifted off the plane it was welded in: that plane
        # is sunk into the back so a print is joined, but a wing lying along it
        # would be buried in the torso rather than resting on it.
        lift = c.raise_ + (self.stow_lift if self.stows_flat else self.fold_lift) * f
        pose["shoulder"] = _euler_from_matrix(
            m3.rot_y(-swing_total) @ m3.rot_z(-lift) @ m3.rot_x(c.twist))
        planar: dict[str, tuple[float, float, float]] = {
            "shoulder": swing_back(swing_total), "elbow": swing_back(e_fold * f),
            "wrist": swing_back(w_fold * f)}
        pose["elbow"] = swing_back(e_fold * f)
        # The wrist hinges in the wing plane (the fold) and cocks out of it
        # (wrist_flex, hand_twist) about the HAND's own axes.
        a_h = np.radians(self.out_angle[self.chains["III"][0]])
        hand_axis = np.array([np.cos(a_h), 0.0, np.sin(a_h)])
        perp = np.array([-np.sin(a_h), 0.0, np.cos(a_h)])
        # a raised wing arches: the hand lags the wrist and hangs below it (and a lowered
        # wing's hand rides up), so raise/lower is not a rigid swing of the whole limb
        flex = c.wrist_flex - self.raise_arch * c.raise_
        pose["wrist"] = _euler_from_matrix(
            m3.rot_y(-w_fold * f) @ _rotation_about(perp, -flex)
            @ _rotation_about(hand_axis, c.hand_twist))
        # the thumb tucks along the forearm instead of standing out sideways
        pose["thumb_cmc"] = swing_back(70.0 * f)
        planar["thumb_cmc"] = pose["thumb_cmc"]
        # the fan closes toward the forearm axis as the wing folds
        fan = float(np.clip(c.fan, 0.0, 1.5))
        for name, phi in self.fan.items():
            n = _NUM[name]
            psi = self.stow_fan.get(name, 0.08 * phi)         # the angle it closes to
            ang = fan * ((1.0 - f) * phi + f * psi)
            pose[f"d{n}_cmc"] = swing_back(ang - phi)
            tilt = self.stow_tilt.get(name, 0.0) * f
            if tilt:
                a = np.radians(self.out_angle[f"d{n}_cmc"] - phi + ang)    # its plan angle now
                axis = np.array([-np.sin(a), 0.0, np.cos(a)])
                pose[f"d{n}_cmc"] = _euler_from_matrix(
                    _rotation_about(axis, tilt) @ m3.euler_rotation(*pose[f"d{n}_cmc"]))
            planar[f"d{n}_cmc"] = pose[f"d{n}_cmc"]
        if c.curl:
            for name, chain in self.chains.items():
                for jn in chain[1:-1]:                    # mcp and ip knuckles
                    a = self.out_angle[jn]
                    axis = np.array([-np.sin(np.radians(a)), 0.0, np.cos(np.radians(a))])
                    pose[jn] = _euler_from_matrix(_rotation_about(axis, c.curl * self.curl_gain))
        if f and self.knuckle_fold:
            # Stowing folds each finger like a concertina at its knuckles (in the
            # wing plane) -- but only once the wing is mostly stowed: half-folded,
            # the hand is folded and the fingers closed yet still straight.
            k = knuckle_phase(f)
            for name, chain in self.chains.items():
                for jn, bend in zip(chain[1:-1], self.bends[name]):
                    amount = (180.0 - bend) * self.knuckle_fold
                    # rotations do not add: a curl (about the digit's own flexion
                    # axis) is applied first and the in-plane concertina swings the
                    # flexed finger, so the two are multiplied as matrices
                    pose[jn] = _then(pose.get(jn), swing_back(amount * k))
                    planar[jn] = _then(planar.get(jn), swing_back(amount * k))
        # Feathers fold like a closing fan: each turns about its own root toward
        # the folded direction, allowing for whatever its parent bone has done.
        # ``fan`` spreads the plumage about its own axis, independently of fold:
        # the primaries about the hand axis, the secondaries about the normal.
        hand_dir = float(np.degrees(a_h))
        ss = lambda x: (lambda u: u * u * (3.0 - 2.0 * u))(float(np.clip(x, 0.0, 1.0)))  # noqa: E731
        follow = self.stow_follow
        # Feathers follow the bone they grow from until late in the fold, so a half-folded wing
        # fans out of its arm the way a real one does instead of every feather swinging to the
        # stowed heading at once. The primaries close their fan toward the hand on the way.
        w2 = ss((f - follow) / (1.0 - follow)) if follow > 0.0 else f
        close = ss(f / 0.8) if follow > 0.0 else 0.0
        # what the FOLD alone does to each bone (a swing, the fan joint or a curl must not
        # drag the plumage: feather headings are absolute in the plane)
        fold_only = {k: v for k, v in planar.items()
                     if k != "shoulder" and (k == "thumb_cmc" or not k.endswith("_cmc"))}
        fold_only["shoulder"] = swing_back(s_fold * f)
        for name, phi in self.fan.items():
            fold_only[f"d{_NUM[name]}_cmc"] = swing_back(phi * (-0.92 * fan * f))
        lat = self._stow_lateral(pose) if (self.stow_taper and f) else {}
        n_pri = sum(1 for fe in self.feathers if fe["row"] == "primary")
        for fe in self.feathers:
            y0 = fe["yaw"]
            if fe["row"] == "primary":
                y_eff = hand_dir + (y0 - hand_dir) * fan
                y_eff += (hand_dir - y_eff) * close
            elif fe["row"] == "secondary":
                y_eff = 90.0 + (y0 - 90.0) * float(np.sqrt(fan))
            else:
                y_eff = y0
            # headings come from the IN-PLANE rotations only: lifting, twisting or
            # cocking the wing tilts its plane but must not read as a change of
            # where the feathers point within it (raise 90 once made P1-P10 span
            # 123 deg instead of 62)
            now = self._heading_after(planar, fe["parent"], y0)
            stowed = self.body_angle + fe["target"] - lat.get(fe["root"], 0.0) * self.stow_taper
            if follow > 0.0:
                carried = self._heading_after(fold_only, fe["parent"], y_eff)
                want = carried + ((stowed - carried + 180.0) % 360.0 - 180.0) * w2
            else:
                want = y_eff + (stowed - y_eff) * f
            # the shortest way round, so a feather never spins the long way
            delta = (want - now + 180.0) % 360.0 - 180.0
            flight = fe["row"] in ("primary", "secondary")
            # A long flight feather droops toward the body (its camber), so stowed flat on
            # the back its tip would dig into the buttocks: it lifts about its own root. A
            # curl cups the hand the other way: the outer primaries droop most.
            pitch = self.stow_pitch * f if (flight and self.stows_flat) else 0.0
            if fe["row"] == "primary" and c.curl and self.curl_cup:
                rank = int(fe["root"][len("primary"):len("primary") + 2]) - 1      # P1 (0) .. P10
                pitch -= c.curl * self.curl_cup * rank / max(n_pri - 1, 1)
            if pitch:
                a0 = np.radians(y0)
                perp = np.array([-np.sin(a0), 0.0, np.cos(a0)])
                pose[fe["root"]] = _euler_from_matrix(
                    m3.rot_y(-delta) @ _rotation_about(perp, -pitch))
            elif abs(delta) > 1e-9:
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
SCALLOP_BAT = {"III-IV": 0.12, "IV-V": 0.28, "V-root": 0.10}
SCALLOP_DRAGON = {"III-IV": 0.40, "IV-V": 0.50, "V-root": 0.18}


def membrane_pieces(span=13.0, thickness=0.22, sweep=0.0, droop=0.0, gull=0.16,
                    camber=CAMBER, scallop=None, root_chord=1.4,
                    billow=BILLOW, propatagium=True, bones=True, rows=22,
                    spike=0.10, bone_scale=1.0, root=None, spec=None):
    """A patagium stretched over a knuckled forelimb, as a skeleton + pieces.

    The sheet is a *fan*: every grid line runs from the arm out to the free
    edge, and the lines that carry the digits run exactly down the bones, so a
    panel's border and its bone coincide and the membrane can be bound to the
    fingers it is stretched between.
    """
    spec = BAT_SPEC if spec is None else spec
    if bone_scale != 1.0 and spec.get("stow_fan"):
        # heavier ribs need more daylight between them when the hand closes
        stow = spec["stow_fan"]
        ref = stow[max(stow, key=lambda k: stow[k])]
        spec = {**spec, "stow_fan": {k: ref + (v - ref) * bone_scale for k, v in stow.items()},
                "stow_tilt": {k: v * bone_scale for k, v in spec.get("stow_tilt", {}).items()}}
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
        (D["III"], D["IV"], _scallop_arc(D["III"], D["IV"], W, depth["III-IV"], 11), 12, "dist"),
        (D["IV"], D["V"], _scallop_arc(D["IV"], D["V"], W, depth["IV-V"], 11), 12, "dist"),
        (D["V"], R, _scallop_arc(D["V"], R, W, depth["V-root"], 13), 14, "te"),
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
    n_back = int((kind == "te").sum() + (kind == "root").sum())
    sigma = np.zeros(nu)
    sigma[kind == "te"] = np.linspace(0.0, 1.0, n_back + 1)[:(kind == "te").sum()]
    sigma[kind == "root"] = np.linspace(0.0, 1.0, n_back + 1)[(kind == "te").sum():][
        :(kind == "root").sum()]
    # The inner edge of the back panels runs up the arm: the trailing edge's rays
    # leave the FOREARM (wrist -> elbow) and the root line's rays leave the HUMERUS
    # (elbow -> shoulder), meeting at the elbow, where the ray to the root corner
    # is the seam. (One parameter along the whole arm put the root corner's ray on
    # the forearm, so the rays crossed in front of the humerus: the arm-to-body
    # fan folded over itself and the sheet creased over the elbow.)
    te_cols = np.flatnonzero(kind == "te")
    root_cols = np.flatnonzero(kind == "root")
    I = np.array([W] * nu)
    fore = E - W

    def side_of(c):                       # which side of the forearm's line edge point c is on
        return np.sign(fore[2] * (B[c][0] - W[0]) - fore[0] * (B[c][2] - W[2]))

    # An edge point that lies forward of the forearm's line is hidden behind the elbow:
    # no point of the forearm sees it without the ray cutting across the humerus, so
    # its ray comes from the elbow itself. The rays of the visible points are spread
    # over the whole forearm, so the cell next to the hidden ones is not a huge wedge.
    visible = [c for c in te_cols if side_of(c) == side_of(te_cols[0])]
    for c in te_cols:
        I[c] = E
    for k, c in enumerate(visible):
        I[c] = W + (E - W) * (k / max(len(visible), 1)) ** 1.6
    for k, c in enumerate(root_cols):
        I[c] = E + (S - E) * (k / max(len(root_cols) - 1, 1))
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
    # (the weight differs between the hand panels and the arm panel, but it must
    # change gradually: a step at digit V left a 0.13-unit ledge down the finger,
    # more than half the membrane's thickness)
    w_cam = np.where(kind == "dist", HAND_CAMBER, 1.0)
    pad = 4
    ext = np.concatenate([np.full(pad, w_cam[0]), w_cam, np.full(pad, w_cam[-1])])
    w_cam = np.convolve(ext, np.ones(2 * pad + 1) / (2 * pad + 1), mode="valid")
    arch = np.sin(np.pi * np.clip(tt, 0, 1) ** CAMBER_PEAK)
    grid[:, :, 1] -= camber * ray[None, :] * w_cam[None, :] * arch
    # sector gap lengths for the billow amplitude
    gaps = {}
    for si, (a, b, _, _, _) in enumerate(segments):
        gaps[si] = float(np.linalg.norm(b - a))
    # Sector 0 is the sliver along digit III (II hugs it): its "gap" is a leading
    # edge, not a panel, and a bulge there would be a lump on the bone. The finger
    # panels pillow by ``billow`` of the gap; the loose arm panel by less per unit.
    amp = np.array([0.0 if sector[c] == 0 else
                    billow * gaps[sector[c]] * (PLAGIO_BILLOW if kind[c] == "te" else 1.0)
                    if kind[c] in ("dist", "te") else 0.0
                    for c in range(nu)])
    prof = np.array([np.sin(np.pi * sloc[c]) for c in range(nu)])
    grid[:, :, 1] -= (amp * prof)[None, :] * tt
    # the whole wing arcs downward toward the tip
    grid[:, :, 1] += droop * span * (np.clip(grid[:, :, 0], 0.0, None) / span) ** 2
    # gull wing: the hand panel rises a few degrees off the arm, so the wrist
    # is the low point of the wing (measured +5..+11 deg on flying foxes)
    grid[:, :, 1] -= gull * np.clip(grid[:, :, 0] - float(W[0]), 0.0, None)
    # washout: the outer wing is twisted leading-edge-down, so toward the tip the
    # trailing edge stands a little higher than the digit-III leading edge it
    # trails (the hand is not a rigid plane)
    lead = np.array([W] + [J_[j] for j in sk.chains["III"][1:]])
    span_out = max(float(lead[-1, 0] - W[0]), 1e-9)
    along = np.clip((grid[:, :, 0] - float(W[0])) / span_out, 0.0, 1.0)
    behind = np.clip(grid[:, :, 2] - np.interp(grid[:, :, 0], lead[:, 0], lead[:, 2]), 0.0, None)
    grid[:, :, 1] -= np.tan(np.radians(WASHOUT)) * behind * along ** 1.5

    thick = np.maximum(thickness * (1.0 - 0.25 * tt), thickness * 0.7)   # never below ~0.3 mm printed
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
                                       "gains": [1.0, 1.0, 1.0, 0.8],
                                       "lobes": [1, 2, 2, 3],
                                       # the digit each panel's first column runs down
                                       "fingers": ["II", "III", "IV", "V"]})]

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
    height_at = lambda q: _surface_height(flat_xz, flat_y, q)        # noqa: E731
    sk.lift_joints(height_at)
    # the thumb hangs ventrally: tilted about 30 degrees off the wing plane
    sk.joints["thumb_tip"][1] += 0.5 * float(np.linalg.norm(sk.thumb - sk.wrist))
    if bones:
        parts += _membrane_bones(sk, thickness, spike, spec.get("claw", 0.12),
                                 height_at, bone_scale)
    return sk, parts


CLAW_RADIUS = 0.028                # of FA: the thumb claw's base radius, below the print floor on purpose
RIDGE_FLOOR = 0.55                 # of the membrane thickness: finger ridges taper to here, no thinner
# The humerus is not a constant rod driven into the back (that read as a bolt). It
# keeps the elbow's radius over its outer half and then tapers toward the shoulder
# joint (which is sunk in the back) to this fraction of it, so where it disappears
# into the torso it is a spindle growing out of the scapula, not a cut-off pipe.
HUMERUS_ROOT_TAPER = 0.5
HUMERUS_TAPER_SPAN = 0.85          # fraction of the humerus, from the shoulder, over which it tapers

# a real fingertip is a bare membrane cusp with the bone ending at the edge; the
# longest digit keeps a short point, the others barely poke past
SPIKE_SCALE = {"II": 0.0, "III": 1.0, "IV": 0.45, "V": 0.45}


def _membrane_bones(sk: WingSkeleton, thickness: float, spike: float,
                    CLAW: float = 0.12, height_at=None, bone_scale: float = 1.0) -> list[Piece]:
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
    floor = 0.72 * thickness * bone_scale
    ridge_floor = RIDGE_FLOOR * thickness * bone_scale        # a ridge must stay proud of the sheet

    def R(frac):
        return max(frac * fa * bone_scale, floor)

    up = np.zeros(3)                               # centred on the sheet: proud on both faces

    def on_skin(points):
        """Densify a polyline and drop each sample onto the sheet: the bone
        follows the skin's own height (camber, billow) along its whole length."""
        pp = _polyline(points, 6)
        if height_at is not None:
            pp[:, 1] = height_at(pp[:, [0, 2]])
        return pp

    def tube(points, r0, r1, per, sides=8, skin=False):
        pts = on_skin(points) if skin else _polyline([p + up for p in points], per)
        t = np.linspace(0.0, 1.0, len(pts))
        return _tube(pts, r0 + (r1 - r0) * t, sides)

    # arm: shoulder -> elbow -> wrist as one smooth tube over the two arm bones,
    # the humerus tapering into the shoulder (see HUMERUS_ROOT_TAPER)
    pp = on_skin([J["shoulder"], J["elbow"], J["wrist"]])
    ts = np.linspace(0.0, 1.0, len(pp))
    mid_i = len(pp) // 2                                 # the elbow sample
    r_arm = np.empty(len(pp))
    r_elbow = R(0.042)
    u_h = np.clip(ts[:mid_i + 1] / ts[mid_i] / HUMERUS_TAPER_SPAN, 0.0, 1.0)   # 0 shoulder .. 1
    r_arm[:mid_i + 1] = np.maximum(
        r_elbow * (HUMERUS_ROOT_TAPER + (1.0 - HUMERUS_ROOT_TAPER) * u_h * u_h * (3.0 - 2.0 * u_h)),
        floor)                                           # ... but never thinner than a printable bone
    u_f = (ts[mid_i:] - ts[mid_i]) / (1.0 - ts[mid_i])
    r_arm[mid_i:] = r_elbow + (R(0.034) - r_elbow) * u_f
    out.append(Piece("arm", _tube(pp, r_arm, 9), chain=("elbow", "wrist")))
    out.append(Piece("wrist_knob", _ball(J["wrist"] + up, 1.25 * R(0.034), 8, 14), bone="wrist"))
    out.append(Piece("elbow_knob", _ball(J["elbow"] + up, 1.3 * R(0.042), 8, 14), bone="elbow"))

    for name, chain in sk.chains.items():
        pts = [J["wrist"]] + [J[j] for j in chain[1:]]
        keys = tuple(chain[1:])                    # one bone per segment
        # the fingertip is a needle that pokes a little past the membrane edge
        d = pts[-1] - pts[-2]
        d = d / max(float(np.linalg.norm(d)), 1e-12)
        if SPIKE_SCALE[name] > 0.0 and spike > 0.0:
            pts = pts + [pts[-1] + d * spike * SPIKE_SCALE[name] * fa]
        base = R(0.030) if name != "II" else R(0.024)
        pp = on_skin(pts)
        t = np.linspace(0.0, 1.0, len(pp))
        # taper to about half by the last joint, then to a needle over the spike
        has_spike = SPIKE_SCALE[name] > 0.0 and spike > 0.0
        n_main = 6 * (len(pts) - 1 - (1 if has_spike else 0))
        # taper toward the last joint, but never below the ridge floor: a bone slimmer
        # than the sheet's own half-thickness stops reading as a ridge and the digit
        # turns into a dark slit along the membrane
        r = np.linspace(base, max(0.5 * base, ridge_floor), n_main + 1)
        if has_spike:
            r = np.concatenate([r, np.linspace(0.5 * base, 0.25 * floor, 6 + 1)[1:]])
        out.append(Piece(f"digit_{name}", _tube(pp, np.maximum(r, 0.25 * floor), 7), chain=keys))

    # digit I: free, clawed, and the detail that says "bat" at a glance. It
    # hangs ventrally and ends in a hooked claw.
    w, th = J["wrist"], J["thumb_tip"]
    out.append(Piece("thumb", tube([w, th], R(0.028), R(0.020), 4, 7), bone="thumb_tip"))
    fwd = (th - w) / max(float(np.linalg.norm(th - w)), 1e-9)
    claw_len = CLAW * fa
    # The claw is a blunt peg, not a needle: it is NOT held to the bone floor (its
    # radius comes straight from the skeleton), tapers only to ~60% and ends flat,
    # so it prints as a stub rather than a hair. The hook curls forward and up (into
    # the wing plane), so it reads from above.
    r_c = CLAW_RADIUS * fa * bone_scale
    mid = th + fwd * 0.55 * claw_len + np.array([0.0, -0.14 * claw_len, 0.0])
    tip = th + fwd * 0.92 * claw_len + np.array([0.0, -0.55 * claw_len, 0.0])
    out.append(Piece("claw", _tube(_polyline(np.array([th + up, mid + up, tip]), 3),
                                   np.linspace(r_c, 0.62 * r_c, 7), 8),
                     bone="thumb_tip"))
    return out


def membrane_rig(**kw) -> WingRig:
    """A bat wing with its skeleton, skin weights and poses."""
    sk, pieces = membrane_pieces(**kw)
    return WingRig(sk, pieces)


# Dragon art exaggerates what a bat wing already is: pinched, deep scallops in
# every gap, spiny hooked tips, a flatter sheet over heavier ribs.
DRAGON = dict(scallop=SCALLOP_DRAGON, spike=0.22, camber=0.05, bone_scale=1.35)


def dragon_rig(**kw) -> WingRig:
    """The same skeleton and rig as the bat, in the exaggerated dragon style."""
    return membrane_rig(**{**DRAGON, **kw})


def membrane_wing(fold=0.0, **kw) -> Mesh:
    """The membrane wing as one mesh (spread, or folded by posing the rig)."""
    rig = membrane_rig(**kw)
    return rig.pose(fold=fold) if fold else rig.rest_mesh


def dragon_wing(fold=0.0, **kw) -> Mesh:
    rig = dragon_rig(**kw)
    return rig.pose(fold=fold) if fold else rig.rest_mesh


# --------------------------------------------------------------------------
# feathered (bird / angel) wing
#
# Everything here is from measurements of eagle, galah and flamingo wings,
# quoted relative to the forearm (FA). Primaries are numbered ornithologically:
# P1 is the innermost, at the wrist; P10 the outermost. The longest, P7/P8, is
# about 1.65 FA; P10 is only ~0.7 of it, which is what makes a soaring wing's
# tip a rounded point instead of a blade.

# length as a fraction of the longest primary, P1..P10. The row steps up smoothly
# to the longest pair and comes back down in two even steps, so the tip is a
# rounded point (P10 a little shorter than P9) and not a blade with a stub beside it.
PRIMARY_RATIO = (0.60, 0.68, 0.76, 0.83, 0.90, 0.96, 1.00, 0.99, 0.92, 0.80)
LONGEST_PRIMARY = 1.65                    # FA
# vane width / length: the broad inner primaries down to the narrow outer ones
PRIMARY_WIDTH = (0.20, 0.19, 0.18, 0.165, 0.15, 0.14, 0.13, 0.12, 0.11, 0.105)
# the slotted outer primaries: how deeply the leading / trailing vane is cut back
# behind the notch, P1..P10 (the inner four are plain, rounded paddles)
PRIMARY_EMARG = (0.0, 0.0, 0.0, 0.25, 0.45, 0.60, 0.72, 0.80, 0.85, 0.85)
PRIMARY_EMARG_IN = (0.0, 0.0, 0.0, 0.0, 0.12, 0.22, 0.32, 0.42, 0.46, 0.42)
SECONDARY_RATIO = (0.62, 0.65, 0.67, 0.69, 0.70, 0.70, 0.70, 0.70, 0.68, 0.66,
                   0.64, 0.62, 0.61, 0.60)
TERTIAL_RATIO = (0.60, 0.62, 0.58, 0.55)
# The covert tiers are fractions of the length of the flight feather they lie over:
# each tier is roughly 60% of the one behind it, so the steps read as a geometric
# progression (greater ~50% of a secondary, median ~30%, lesser ~15%)
GREATER_RATIO = 0.50
MEDIAN_RATIO = 0.30
LESSER_RATIO = (0.17, 0.13)
MARGINAL_RATIO = 0.15
# the alula: three small stiff feathers, as a fraction of the longest primary
ALULA_RATIO = (0.32, 0.29, 0.26)
NOSE_LIFT = 0.07                          # how far above the wing plane the arm's axis runs (x scale)
NOSE_RADIUS = (0.085, 0.066)              # arm radius at the shoulder and at the wrist (x scale)
ALULA_OFF_EDGE = 26.0                     # degrees the alula stands off the leading edge
FA_P = LONGEST_PRIMARY


def _bump(t, centre, sigma):
    return np.exp(-((t - centre) / sigma) ** 2)


def _step(x, a, b):
    """Smoothstep from 0 at ``a`` to 1 at ``b``."""
    u = np.clip((x - a) / (b - a), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _blade(length, width, thickness, curl=0.10, cup=0.16, sweep=0.08,
           lead_frac=0.30, tip=0.74, tip_pow=1.8, emarg=0.0, emarg_in=0.0,
           calamus=0.10, tip_floor=0.07, root_floor=0.07, n=11, m=8,
           notch=0.55, rise_at=0.42, ridge=1.35, vane=0.90, ridge_dir=-1.0, butt=0.0) -> Mesh:
    """One feather: a flat vane with a raised shaft down its middle.

    A bare quill (the calamus) runs the first tenth of the length, then the
    vane widens to its maximum by ``rise_at`` and holds parallel-sided until the
    tip, where it closes in an ogive onto a rounded point. ``lead_frac`` is the
    share of the vane in front of the shaft: the narrow leading vane against
    the broad trailing one is what makes a flight feather aerodynamic, and it
    shows in silhouette. ``tip_floor`` is the width (fraction of ``width``) the
    tip closes down to and ``root_floor`` the width of the quill end (a covert's
    quill must not shrink to a spearhead that pokes out of the feather in front
    of it); ``butt`` rounds the quill end over that fraction of the length.
    Nothing goes below half a ``thickness`` wide in absolute terms, so a tip or a
    quill never thins to a sliver narrower than the sheet is thick.

    ``emarg`` cuts the leading vane down sharply at ``notch`` (a fraction of the
    length) and keeps it narrow to the tip, and ``emarg_in`` does the same to
    the trailing vane a little nearer the root: the slotted look of soaring
    primaries, where the outer feathers separate into fingers.

    The vane is a flat sheet ``vane`` x ``thickness`` thick (so it never goes
    thinner than a printer resolves) with a shaft ridge ``ridge`` x ``thickness``
    standing off one face only -- up on a dorsal feather (``ridge_dir`` -1),
    down on a ventral one (+1) -- thinning toward the tip.
    """
    # Sample the length unevenly: uniform along the body, but dense toward the
    # very tip, where the outline curves hardest, and (on a notched feather) at
    # the notch, whose step must read as a corner and not a ramp. A tip closed in
    # 2-3 straight segments reads as a chamfer, and a row of them as a saw.
    n_tip = max(6, n // 2)
    t = np.concatenate([np.linspace(0.0, tip, max(n - n_tip, 4), endpoint=False),
                        tip + (1.0 - tip) * np.sin(np.linspace(0.0, np.pi / 2.0, n_tip + 1))])
    if emarg or emarg_in:
        t = np.union1d(t, np.clip([notch - 0.12, notch - 0.075, notch - 0.05, notch - 0.03,
                                   notch - 0.005, notch + 0.04], 0.05, tip))
        # union1d only drops EXACT duplicates: a notch sample 1e-16 from a grid row
        # left a sliver quad band and ~20 flipped-normal creases per feather
        t = t[np.concatenate([[True], np.diff(t) > 3e-3])]
    n = len(t) - 1
    rise = np.clip((t - calamus) / (rise_at - calamus), 0.0, 1.0) ** 0.8
    ogive = np.clip(1.0 - (np.clip(t - tip, 0.0, None) / (1.0 - tip)) ** tip_pow,
                    0.0, 1.0) ** 0.5
    floor = np.where(t > 0.5, tip_floor, root_floor)
    W = width * np.maximum(rise * ogive, floor)
    if butt > 0.0:
        # a rounded butt: the quill end of a covert closes in a quarter-ellipse
        # instead of a squared-off stub that reads as a tooth on the row's edge
        W = W * np.sqrt(1.0 - (1.0 - np.clip(t / butt, 0.0, 1.0)) ** 2)
    W = np.maximum(W, 0.5 * thickness)         # never a sliver narrower than the sheet is thick

    wl = lead_frac * W * (1.0 - emarg * _step(t, notch - 0.05, notch + 0.04))
    wt = (1.0 - lead_frac) * W * (1.0 - emarg_in * _step(t, notch - 0.12, notch - 0.03))
    # columns bunch toward the shaft, so the ridge has a column of its own on
    # either side of it
    u = np.linspace(-1.0, 1.0, m + 1)
    v = np.sign(u) * np.abs(u) ** 1.4
    z = np.where(v[None, :] < 0, v[None, :] * wl[:, None], v[None, :] * wt[:, None])
    av = np.abs(v)
    # a sheet, thinning slightly to its edge, plus a ridge along the shaft that
    # fades to half toward the tip
    flat = vane * (1.0 - 0.12 * av ** 2)[None, :] * np.ones((n + 1, 1))
    extra = (ridge - vane) * np.exp(-(av / 0.20) ** 2)[None, :] * (1.0 - 0.5 * t)[:, None]
    # the very end of the feather is bevelled to 60% of the sheet, so the tip closes
    # in a rounded end instead of a flat wall (which reads as a spike on every tip)
    bevel = 1.0 - 0.4 * _step(t, 0.965, 1.0)
    th = thickness * (flat + extra) * bevel[:, None]
    grid = np.zeros((n + 1, m + 1, 3))
    grid[:, :, 0] = (length * t)[:, None]
    grid[:, :, 2] = z + (sweep * length * t ** 2)[:, None]
    # the shaft droops along its length and the vanes cup away from it, each vane
    # curling down in proportion to its own width (so a notched, narrow vane keeps
    # a gentle slope instead of a cliff)
    grid[:, :, 1] = ((curl * length * t ** 2)[:, None]
                     + cup * np.abs(z) * av[None, :] ** 0.5)
    slab = _slab_from_grid(grid, th)
    # The ridge stands off one face only, so the other stays a clean sheet. It is a
    # plain vertical offset of both faces applied AFTER the slab is built: folding it
    # into the grid would tilt the grid's normals sideways wherever a narrow vane
    # puts adjacent columns a hair apart, and the sheet would twist.
    nv = (n + 1) * (m + 1)
    verts = slab.vertices.copy()
    shift = (ridge_dir * 0.5 * thickness * extra).ravel()
    verts[:nv, 1] += shift
    verts[nv:, 1] += shift
    return Mesh(verts, slab.faces)


def _place_feather(mesh: Mesh, root, yaw_deg, roll_deg=0.0, pitch_deg=0.0) -> Mesh:
    """Rotate a feather about its root: pitch (tip up is negative, about the
    feather's cross axis), then roll about its own axis, then yaw in the wing plane."""
    a = np.radians(yaw_deg)
    ry = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
    b = np.radians(roll_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    c = np.radians(pitch_deg)
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    m = np.eye(4)
    m[:3, :3] = ry @ rx @ rz
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
                     primaries=10, secondaries=14, tertials=7, scapulars=7,
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
        blade = _blade(fe["length"], fe["width"], thickness, curl=fe["curl"], cup=fe["cup"],
                       lead_frac=fe["lead"], tip=fe["tip_at"], tip_pow=fe["tip_pow"],
                       emarg=fe["emarg"], emarg_in=fe["emarg_in"], tip_floor=fe["tip_floor"],
                       root_floor=fe.get("root_floor", 0.07), notch=fe["notch"],
                       rise_at=fe.get("rise_at", 0.42), n=16)
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
    the forearm, then scaled by ``sk.scale``. Primaries root along the hand and
    fan from P1 (+70 deg) to P10 (+8 deg), which continues the hand axis: packed
    ~4 deg apart on the inner feathers and ~10 deg apart on the outer ones, so
    the inner wing is a closed sheet and the outer primaries are separate fingers
    with visible slots between them. Secondaries root along the ulna's trailing
    side at +78..+100 deg, so the flight-feather row is one continuous scalloped
    edge with no step where primaries meet secondaries. Tertials and scapulars on
    the humerus fair the trailing edge into the back; the alula stands on the
    thumb at the leading edge. All of them are rigged feathers, so a fold re-aims
    every one.
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
                 tip_floor=0.25, fold_to=4.0, notch=0.60, cup=0.12, curl=0.08)
        d.update(kw)
        return d

    hand = [J["wrist"], J["d3_mcp"], J["d3_tip"]]
    hv = J["d3_tip"] - J["wrist"]
    hand_yaw = float(np.degrees(np.arctan2(hv[2], hv[0])))
    fr = np.linspace(0.10, 1.0, n_pri)
    roots = _along(hand, fr) + np.array([0.0, 0.0, 0.03 * s])     # calamus on the trailing side
    # fan angle of each primary: the gaps widen steadily toward the tip
    gap = 0.7 + 0.9 * np.arange(max(n_pri - 1, 0)) / max(n_pri - 2, 1)
    fan = np.concatenate([[0.0], np.cumsum(gap)]) / max(gap.sum(), 1e-9)
    for i in range(n_pri - 1, -1, -1):                            # outermost first = lowest
        k = int(round(i * 9 / max(n_pri - 1, 1)))
        L = PRIMARY_RATIO[k] * FA_P * s * _jit(i, 0.025)
        outer = k >= 5
        parent = "wrist" if fr[i] <= 0.5 else "d3_mcp"
        out.append(common(row="primary", i=i + 1, root=roots[i], parent=parent,
                          yaw=70.0 - 62.0 * float(fan[i])
                          + (0.8 * float(np.sin(5.1 * i)) if k >= 4 else 0.0),
                          length=L, width=PRIMARY_WIDTH[k] * L,
                          lead=0.27 if outer else 0.37,
                          emarg=PRIMARY_EMARG[k], emarg_in=PRIMARY_EMARG_IN[k],
                          notch=0.60 if k >= 6 else 0.64 + 0.01 * float(np.sin(2.3 * i)),
                          tip_at=0.70 if k >= 6 else 0.80, tip_pow=1.6 if k >= 6 else 2.0,
                          tip_floor=0.07 if k >= 6 else 0.12, fold_to=2.0, cup=0.08, curl=0.16,
                          roll=-6.0 - 8.0 * (1.0 - i / max(n_pri - 1, 1)), layer=layer))
        layer += 1
    fr = np.linspace(0.0, 1.0, n_sec)
    roots = _along([J["wrist"], J["elbow"]], fr) + np.array([0.0, 0.0, 0.06 * s])
    for i in range(n_sec):
        k = int(round(i * (len(SECONDARY_RATIO) - 1) / max(n_sec - 1, 1)))
        L = SECONDARY_RATIO[k] * FA_P * s * _jit(i + 20, 0.025)
        out.append(common(row="secondary", i=i + 1, root=roots[i], parent="elbow",
                          yaw=78.0 + 22.0 * fr[i] + 1.5 * float(np.sin(3.7 * i)),
                          length=L, width=0.23 * L, rise_at=0.34,
                          lead=0.40, tip_at=0.80, tip_pow=2.8, tip_floor=0.20, fold_to=4.0,
                          roll=-9.0 - 3.0 * fr[i], layer=layer))
        layer += 1
    # tertials: seven, overlapping ~70%, shortening smoothly so the trailing edge
    # curves up into the back instead of ending in a few flaps
    fr = np.linspace(0.05, 0.92, n_ter)
    roots = _along([J["elbow"], J["shoulder"]], fr) + np.array([0.0, 0.0, 0.08 * s])
    ratio = np.linspace(0.52, 0.38, n_ter)
    for i in range(n_ter):
        L = ratio[i] * FA_P * s
        out.append(common(row="tertial", i=i + 1, root=roots[i], parent="shoulder",
                          yaw=92.0 + 18.0 * i / max(n_ter - 1, 1), length=L, width=0.26 * L,
                          lead=0.40, tip_at=0.76, tip_pow=2.2, tip_floor=0.18, fold_to=6.0,
                          rise_at=0.34, roll=-10.0, layer=layer))
        layer += 1
    # scapulars: soft, lanceolate feathers over the humerus root that fair the wing
    # into the back. A near-parallel rank trailing along the back (as on the
    # flamingo and galah), each longer than the one outboard of it, tapering to a
    # rounded point: fat blunt ones read as a cluster of pods at the shoulder.
    for i, f in enumerate(np.linspace(0.05, 0.60, n_sca)):
        root = J["shoulder"] + (J["elbow"] - J["shoulder"]) * f + np.array([0.0, 0.0, 0.12 * s])
        L = 0.70 * s * (1.0 - 0.20 * i / max(n_sca - 1, 1))
        out.append(common(row="scapular", i=i + 1, root=root, parent="shoulder",
                          yaw=98.0 + 14.0 * i / max(n_sca - 1, 1), length=L, width=0.26 * L,
                          lead=0.40, tip_at=0.62, tip_pow=1.7, tip_floor=0.12, fold_to=8.0,
                          rise_at=0.30, roll=-9.0, layer=layer))
        layer += 1
    # alula: three small stiff feathers (25-35% of the longest primary) on the
    # thumb at the leading edge of the wrist, longest and lowest first. They stand
    # ALULA_OFF_EDGE degrees off the leading edge (the hand's own axis), the way a
    # bird's alula is cocked up and out, and ride on top of the primary bases.
    hd = np.array([np.cos(np.radians(hand_yaw)), 0.0, np.sin(np.radians(hand_yaw))])
    for i, (ratio, f, off) in enumerate(zip(ALULA_RATIO, (0.36, 0.22, 0.08), (1.0, 0.7, 0.4))):
        ln = ratio * FA_P * s
        root = J["wrist"] + hd * f * float(np.linalg.norm(hv)) + np.array([0.0, 0.0, -0.02 * s])
        out.append(common(row="alula", i=i + 1, root=root, parent="thumb_cmc",
                          yaw=hand_yaw - ALULA_OFF_EDGE * off, length=ln, width=0.20 * ln,
                          lead=0.30, tip_at=0.64, tip_pow=1.9, tip_floor=0.12, fold_to=15.0,
                          rise_at=0.30, cup=0.04, curl=0.03, roll=-12.0, layer=layer, root_floor=0.30))
        layer += 8                    # a clear step, so the three never z-fight where they overlap
    for fe in out:
        fe["tip"] = fe["root"] + _dir(fe["yaw"]) * fe["length"]
    return out


def _jit(i, amount=0.08):
    """Deterministic +-amount length jitter, so a tier's lower edge is not a ruled line."""
    return 1.0 + amount * float(np.sin(7.3 * i + 1.1))


def _covert_blade(length, wol, thickness, ventral=False, n=10, m=6, tip_floor=0.0,
                  lead_frac=0.42, curl=0.04) -> Mesh:
    """A covert: a broad vane with a rounded (U-shaped) end that carries full width
    from close to the quill, since the quill end is buried under the row in front.
    A lozenge -- narrow at both ends -- is what made a row of coverts read as grains
    of rice. ``wol`` is width over length."""
    tip = float(np.clip(1.0 - 0.62 * wol, 0.45, 0.88))
    return _blade(length, wol * length, thickness, curl=curl, cup=0.0, sweep=0.06,
                  lead_frac=lead_frac, tip=tip, tip_pow=2.0, calamus=0.0, rise_at=0.22,
                  root_floor=0.55, tip_floor=tip_floor, n=n, m=m, ridge=1.15, vane=0.90,
                  ridge_dir=1.0 if ventral else -1.0, butt=0.14)


def _coverts(sk, layout, bone_of, thickness, height, step, underwing=True) -> list[Piece]:
    """The shingles over the flight feathers' bases. Each covert is welded to the
    bone of the feather it lies beside, so it fans with it; the leading-edge
    rows ride on the nearest secondary or tertial.

    Tiers, from the trailing edge forward (each lies over the one behind it):
    primary coverts, greater, median, then lesser rows, then the marginals on the
    nose of the arm. Their lengths step down geometrically as a share of the
    flight feather they cover (greater 50%, median 30%, lesser 17% / 13%,
    marginal 10%), so each tier is a visible step and not a repeat of the one
    behind. A tier is STAGGERED against the one behind it -- each covert
    straddles two flight feathers, so the rows brick rather than stack into
    columns -- and ends in a rounded tip. ``underwing`` adds the ventral covert
    field, which is what an angel seen from the front shows.
    """
    out: list[Piece] = []
    s = sk.scale
    J = sk.joints
    by = {(f["row"], f["i"]): f for f in layout}
    n_sec = sum(1 for f in layout if f["row"] == "secondary")
    n_pri = sum(1 for f in layout if f["row"] == "primary")
    sec_pitch = (J["elbow"] - J["wrist"]) / max(n_sec - 1, 1)
    pri_pitch = (J["d3_tip"] - J["wrist"]) * 0.9 / max(n_pri - 1, 1)

    def covert(name, fe, length, wol, fwd, tier, bone, pitch, stagger, yaw_off,
               ventral=False, tip_floor=0.0):
        root = fe["root"] + pitch * stagger + np.array([0.0, 0.0, -fwd * s])
        if ventral:
            root[1] = height(root[0]) + tier * 0.6 * thickness
        else:
            root[1] = height(root[0]) - (fe["layer"] + tier * 0.7 * thickness / step) * step
        length = length * _jit(fe["i"] + 3 * tier, 0.03)
        blade = _covert_blade(length, wol, thickness, ventral, tip_floor=tip_floor)
        roll = (-fe["roll"] + 3.0) if ventral else (fe["roll"] - 3.0)
        out.append(Piece(name, _place_feather(blade, root, fe["yaw"] + yaw_off, roll), bone=bone))

    for (row, i), fe in by.items():
        b = bone_of.get((row, i))
        if b is None:
            continue
        for ventral in ((False, True) if underwing else (False,)):
            tag = "under_" if ventral else ""
            if row == "primary":
                # outer primaries carry shorter coverts (46% of P1 down to 32% of P10)
                frac = 0.46 - 0.14 * (i - 1) / max(n_pri - 1, 1)
                covert(f"{tag}primary_covert{i:02d}", fe, frac * fe["length"], 0.36, 0.02, 1, b,
                       pri_pitch, 0.5, 2.0, ventral)
            elif row == "secondary":
                covert(f"{tag}greater{i:02d}", fe, GREATER_RATIO * fe["length"], 0.40, 0.05, 1, b,
                       sec_pitch, 0.5, 3.0, ventral)
                covert(f"{tag}median{i:02d}", fe, MEDIAN_RATIO * fe["length"], 0.52, 0.06, 2, b,
                       sec_pitch, 0.25, 6.0, ventral)
    # the tertials get their own greater and median rows, so the covert field
    # runs on past the elbow to the back instead of stopping at it
    ter_pitch = (J["shoulder"] - J["elbow"]) * 0.87 / max(
        sum(1 for f in layout if f["row"] == "tertial") - 1, 1)
    for (row, i), fe in by.items():
        b = bone_of.get((row, i))
        if row != "tertial" or b is None:
            continue
        for ventral in ((False, True) if underwing else (False,)):
            tag = "under_" if ventral else ""
            covert(f"{tag}ter_greater{i:02d}", fe, GREATER_RATIO * fe["length"], 0.42, 0.05, 1, b,
                   ter_pitch, 0.5, 3.0, ventral)
            covert(f"{tag}ter_median{i:02d}", fe, MEDIAN_RATIO * fe["length"], 0.55, 0.06, 2, b,
                   ter_pitch, 0.25, 6.0, ventral)
    # lesser / marginal coverts: scale-like rows along the arm. They are sized as a
    # share of the secondaries they sit in front of. The lesser rows root just
    # BEHIND the bone line and point straight back; the marginals root AHEAD of it
    # and sweep back over it with rounded ends, so the leading edge is feathered
    # right to the front, not a bare rod with quill ends poking out. Each row lies
    # a little higher than the one behind it, so the nose of the arm is thick and
    # the rows shingle toward it.
    arm = [J["shoulder"], J["elbow"], J["wrist"]]
    host = [f for f in layout if f["row"] in ("secondary", "tertial")]
    secs = [f["length"] for f in layout if f["row"] == "secondary"]
    ref = float(np.mean(secs)) if secs else 0.66 * FA_P * s

    def arm_row(tag, r, ratio, n, tier, ahead, wol, roll, ventral, stagger, nose=False,
                pitch=0.0, curl=0.04):
        pts = _along(arm, np.linspace(0.02, 0.98, n) + (stagger / n))
        for j, p in enumerate(pts):
            root = p + np.array([0.0, 0.0, -ahead * s])
            hf = min(host, key=lambda f: float(np.linalg.norm(f["root"] - root)))
            if nose:
                # a nose feather roots on the front of the arm's own tube, on its
                # axis, and wraps over (or under) it
                root[1] = height(root[0]) - NOSE_LIFT * s
                root[2] = p[2] - (NOSE_RADIUS[0] + (NOSE_RADIUS[1] - NOSE_RADIUS[0])
                                  * min(1.0, j / max(n - 1, 1))) * s
            elif ventral:
                root[1] = height(root[0]) + tier * 0.6 * thickness
            else:
                # above the median coverts of the feather it lies in front of, each
                # row a little higher than the one behind it
                root[1] = height(root[0]) - (hf["layer"] + 2 * 0.7 * thickness / step
                                             + tier * 0.35 * thickness / step) * step
            blade = _covert_blade(ratio * ref * _jit(j + 5 * r, 0.04), wol, thickness, ventral,
                                  n=10, m=4, lead_frac=0.45, curl=curl)
            out.append(Piece(f"{tag}{j:02d}", _place_feather(blade, root, 90.0, roll, pitch),
                             bone=bone_of[(hf["row"], hf["i"])]))

    for r, ratio in enumerate(LESSER_RATIO):
        arm_row(f"lesser{r}_", r, ratio, 22 + 4 * r, 1.0 + r, -0.02 * (1 - r), 0.75, -12.0,
                False, 0.5 * (r % 2))
    arm_row("marginal0_", 0, MARGINAL_RATIO, 36, 3.0, 0.0, 1.0, -8.0, False, 0.0,
            nose=True, pitch=-58.0, curl=0.30)
    if underwing:
        arm_row("under_lesser0_", 0, 0.15, 22, 1.2, 0.0, 0.75, 12.0, True, 0.0)
        arm_row("under_marginal0_", 0, 0.12, 28, 1.0, 0.0, 0.95, 8.0, True, 0.5,
                nose=True, pitch=52.0, curl=-0.25)
    return out


def _feather_bones(sk, height, s) -> list[Piece]:
    """The arm under the plumage. Buried by the lesser coverts, but it gives the
    wing a rounded leading edge and the rig something to hold."""
    J = sk.joints
    out = []

    def tube(points, r0, r1, per=5, lift=0.02):
        pts = _polyline([np.array([p[0], height(p[0]) - lift * s, p[2]]) for p in points], per)
        t = np.linspace(0.0, 1.0, len(pts))
        return _tube(pts, r0 + (r1 - r0) * t, 9)

    # the arm is the core of the leading edge: thick enough that the marginal rows
    # wrap a solid nose instead of a bare rod, and raised to the middle of the stack
    out.append(Piece("arm", tube([J["shoulder"], J["elbow"], J["wrist"]], NOSE_RADIUS[0] * s,
                                 NOSE_RADIUS[1] * s, 6, lift=NOSE_LIFT), chain=("elbow", "wrist")))
    out.append(Piece("hand", tube([J["wrist"], J["d3_mcp"], J["d3_tip"]], 0.040 * s,
                                  0.012 * s, 5, lift=0.04), chain=("d3_mcp", "d3_tip")))
    out.append(Piece("wrist_knob", _ball(J["wrist"] + np.array([0, -0.02 * s, 0]), 0.030 * s),
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

STYLES = {"membrane": membrane_wing, "dragon": dragon_wing, "feathered": feathered_wing}
RIGS = {"membrane": membrane_rig, "dragon": dragon_rig, "feathered": feathered_rig}


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
