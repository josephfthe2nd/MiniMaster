"""The horn as an anatomy: a bony core on the frontal bone, the soft tissue that tents the scalp
around it, and the keratin sheath over it.

A horn is not a cone pushed into the scalp. Every cranial protrusion that real animals (and the
few humans) carry is the same three-layer arrangement, and this module builds the three layers
from ONE set of numbers, so the surface that is printed and rendered is derived from the layers
below it, never a second, unrelated shape:

* ``bone``   the cornual process of the frontal bone: a cone about 0.8 R at the skin, tapering over
             the first 0.62 of the length (the last 0.4 of a bovid horn is solid keratin) and
             continuous with the skull below the skin (the pedestal);
* ``muscle`` the soft tissue between the skull and the skin that the pedestal lifts into a tent
             (scalp, galea, periosteum and the horn's germinal cuff): a lens on the head, thickest
             at the horn and falling to the ordinary scalp thickness at ``rho_m`` (= ``tent_radius``);
             its upper surface is the moulded skin LESS the skin thickness, its lower one the skull's
             outer table;
* ``skin``   the keratin sheath (the core is derived from it, never nearer the surface than the
             keratin wall), the shell that is printed and rendered (the scalp around it is the body itself, moulded by the junction spec this
             module also produces so that it rises to the numbers of the muscle layer).

THE LOCAL FRAME (the one :class:`~minimaster.character.kinds.HornKind` places on the head) is
``z`` = the axis of the horn's first part (out of the skull), ``x`` = the direction the horn bends
toward, ``y = z cross x``; ``s`` runs along the axis from the skin point of the root (negative inside
the head: the hidden stem). The units are those of the horn parameters (``length``,
``base_radius``, at the neutral head); the kind scales the mesh to the head (the eye distance over
the neutral one).

THE NUMBERS (``docs/anatomy_horn.md`` says where each one comes from and which are guesses):

* the sheath radius law is ``r(s) = tip + (R - tip) (1 - s/L)^p`` with ``p`` 1.05 (a cone, not a thorn)
  times a periople-like root flare ``1 + F exp(-s / 0.35 R)`` and growth rings (annuli) of
  amplitude ``rings`` x r, pitch 0.8 R proximally and 0.55 of that at the tip;
* the section is an ellipse whose major axis lies in the bend plane (minor : major 0.85 for the
  scimitar, 0.80 for the ram with a 3-lobe keel, 0.9 for the antelope, 1 for the corkscrew);
* the centreline has the curvature of an equiangular spiral, ``kappa(s) = k / (S0 - s)``: it is
  non-zero at the root and rises toward the tip, for a total turn ``Psi = curve L / 2`` (the
  first versions' ``curve`` keeps its meaning); a lyre changes sign; a corkscrew adds a helical
  term; ``twist`` turns the bend plane;
* the root has a FOOT instead of a flat cap: rings that follow the moulded skin the sheath comes out of (lowered
  to it along the axis), a flare along the skin, a groove and a bead at the skin line, and a rounded bowl
  under the scalp (``HornModel.foot_rings``); the bone is the same foot and bowl inset, so it is inside the sheath
  everywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from ..core.mesh import Mesh
from . import appendages as A
from . import junction as J

# --------------------------------------------------------------------------
# constants (every one is documented in docs/anatomy_horn.md)

IOD_REF = 0.582          # the inter-ocular distance of the neutral head: horn sizes are at this head
TIP_FLOOR = 0.012        # the sheath's tip radius is at least this (a point is not printable)
TAPER_P = 1.05           # the exponent of the sheath's taper (1.25 made a thorn)
FLARE_LEN = 0.35         # decay length of the root flare, in root radii
SINK_FLOOR = 0.3         # the stem reaches this many base radii under the skin even for a horn that stands square
SINK_TILT = 0.9          # ... plus this x the root's radius x tan(tilt): the skin falls away on the far side
FLARE_MAX = 0.6          # the root flare is limited to this (a flare of 1.5 is a trumpet 2.5 radii wide)
CAP_DEPTH = 0.5          # the buried end is a rounded cap this many root radii deep
CORE_BASE = 0.82         # the cornual process' radius at the skin, in base radii
CORE_MIN_WALL = 0.12     # the keratin wall is at least this thick (base radii) over the core
CORE_PEDESTAL = 1.12     # the process widens into the skull to this many base radii
RING_PITCH = 0.8         # annulus pitch at the root, in base radii
RING_PITCH_TIP = 0.55    # ... falling to this fraction of it at the tip
RING_FIRST = 0.9         # the first annulus is this many base radii above the skin
RING_PER_PERIOD = 6      # rings per annulus period (the loft must resolve the ridge)
TIP_CAP = (32.0, 62.0, 82.0)    # the rounded end: rings on a quarter sphere (degrees)
# the sheath's foot (see ``HornModel.skin``): rings at these heights above the LOCAL skin (in root
# radii), the first one buried; a groove and a bead (the keratin's crisp rim) sit on them
FOOT_HEIGHTS = (-0.25, 0.0, 0.1, 0.22, 0.36, 0.52, 0.75, 1.05, 1.4)
FOOT_BLEND = 2.5         # the rings go from following the skin (low) to the planar first shaft ring (top)
GROOVE_H, GROOVE_W, GROOVE_D = 0.10, 0.09, 0.05     # the groove at the skin line: height, width, depth (root radii)
BEAD_H, BEAD_W, BEAD_E = 0.34, 0.14, 0.07           # the bead above it: height, width, swell (root radii)
BOWL_RINGS = (22.0, 45.0, 66.0, 82.0)    # the buried bowl under the collar (degrees on a quarter ellipse)
BEND_LIMIT = 0.8         # a ring's radius times the path curvature may not exceed this
HELIX_RHO = 2.0          # a corkscrew's circle has this many local radii
HUG = 0.97               # the skin hugs this fraction of the (inscribed) sheath radius
HALFWIDTH_CAP = 0.40     # the base radius is at most this share of the head's half-width at that height

# the skin around the root (junction_spec): in base radii
TENT_RADIUS = 2.2        # rho_m: the tent's radius
TENT_FLOOR = 0.36        # ... at least three mean cage edges (the cage is 0.125 an edge)
TENT_MARGIN = 0.075      # the tent stays this far from the midline
TENT_HEIGHT = 0.35       # H of the tent (base radii); a ram's is TENT_HEIGHT_RAM
TENT_HEIGHT_RAM = 0.5
TENT_MIN = 0.10          # ... and at least this (units): the cage resolves nothing smaller
SKIN_THICK = 0.06        # scalp skin thickness, units (0.6 cm; 3-8 mm)
SCALP_THICK = 0.065      # galea + periosteum + skin over the outer table, units (6.7 mm)
PLATE_GAP = 0.004        # the skull plate lies this far under the soft tissue (two coincident surfaces z-fight in a section view)


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


# --------------------------------------------------------------------------
# the shapes

# One row per ``shape``: the defaults of the parameters and what the anatomy of that kind of horn
# says about its orientation (head frame: x out, y up, z forward; left side) and its section.
SHAPES = {
    "curved": dict(length=1.6, base_radius=0.17, curve=1.31, twist=0.0, turns=0.0, segments=28,
                   tip=0.06, profile="spiral", q=1.8, aspect=0.85, lobe=0.0, rings=0.05,
                   flare=0.15, tilt=27.0, lean=(-0.31, 0.80, -0.52), bend=(0.0, 0.0, -1.0),
                   core=0.62, material="bone", site="horn_crown"),
    "ram": dict(length=3.0, base_radius=0.28, curve=3.2, twist=0.0, turns=0.0, segments=36,
                tip=0.07, profile="spiral", q=1.35, aspect=0.80, lobe=0.06, rings=0.06,
                flare=0.15, tilt=41.0, lean=(0.14, 0.49, -0.86), bend=(0.45, -0.65, -0.55),
                core=0.6, material="bone", site="horn_temple_crest"),
    "spiral": dict(length=4.0, base_radius=0.13, curve=0.35, twist=0.0, turns=2.25, segments=48,
                   tip=0.04, profile="spiral", q=2.0, aspect=1.0, lobe=0.0, rings=0.03,
                   flare=0.15, tilt=40.0, lean=(-0.42, 0.83, -0.37), bend=(0.0, 0.0, -1.0),
                   core=0.62, material="bone", site="horn_crown"),
    "antelope": dict(length=4.0, base_radius=0.13, curve=0.2, twist=0.0, turns=0.0, segments=40,
                     tip=0.04, profile="lyre", q=2.0, aspect=0.90, lobe=0.0, rings=0.04,
                     flare=0.15, tilt=45.0, lean=(-0.31, 0.80, -0.52), bend=(0.7, 0.0, -0.7),
                     core=0.62, material="bone", site="horn_crown"),
    # a giraffe's ossicone: bone that stays under the skin and hair, so the whole surface is skin
    # material; it tapers by only a quarter, ends in a dome and meets the head in a concave fillet
    "ossicone": dict(length=0.45, base_radius=0.15, curve=0.2, twist=0.0, turns=0.0, segments=16,
                     tip=0.105, profile="spiral", q=2.0, aspect=1.0, lobe=0.0, rings=0.0,
                     flare=0.45, tilt=20.0, lean=(-0.31, 0.80, -0.52), bend=(0.0, 0.0, -1.0),
                     core=1.0, material="skin", site="horn_crown"),
}
SHAPE_NAMES = tuple(SHAPES)


@dataclass(frozen=True)
class HornParams:
    """What the anatomy is built from (units of the horn parameters, at the neutral head)."""

    shape: str = "curved"
    length: float = 1.6
    base_radius: float = 0.17
    segments: int = 28
    ring_verts: int = 16
    curve: float = 1.31
    twist: float = 0.0
    turns: float = 0.0
    tip_radius: float = 0.06
    flare: float = 0.15
    rings: float = 0.05          # growth-ring amplitude, in local radii
    tilt: float = 27.0           # degrees between the axis and the skull's surface normal
    boss: float = 1.0            # the collar's size and the tent's, in 1 = the default (0: none)

    def with_(self, **kw) -> "HornParams":
        return replace(self, **kw)

    @property
    def row(self) -> dict:
        return SHAPES.get(self.shape, SHAPES["curved"])

    @property
    def root_radius(self) -> float:
        """The radius of the sheath at the skin (flare included)."""
        return self.base_radius * (1.0 + self.flare_eff)

    @property
    def flare_eff(self) -> float:
        return min(max(float(self.flare), 0.0), FLARE_MAX)

    @property
    def sink(self) -> float:
        """How deep the full-radius stem reaches below the skin point along the axis: far enough that
        the skin, which falls away by ``radius x tan(tilt)`` on the side the horn leans away from
        (and by the skull's curvature), never shows the stem's end. A rounded cap of ``CAP_DEPTH`` root
        radii follows it."""
        Ra = self.root_radius
        return SINK_FLOOR * self.base_radius + SINK_TILT * Ra * math.tan(math.radians(min(self.tilt, 70.0)))

    @property
    def cap_depth(self) -> float:
        return CAP_DEPTH * self.root_radius

    @property
    def tip(self) -> float:
        return max(float(self.tip_radius), TIP_FLOOR)

    @property
    def tent_height(self) -> float:
        return TENT_HEIGHT_RAM if self.shape == "ram" else TENT_HEIGHT


# --------------------------------------------------------------------------
# the head: landmarks that follow every morph


@dataclass(frozen=True)
class Head:
    """Landmarks of one morphed head (MakeHuman frame, the left half)."""

    eye: np.ndarray            # the left eye joint
    top: np.ndarray            # the crown marker (head-2)
    root: np.ndarray           # the skull base (head)
    ear: np.ndarray            # the centre of the left ear
    iod: float                 # inter-ocular distance
    left: np.ndarray           # the head axes, unit: toward the character's left,
    up: np.ndarray             # up the head,
    fwd: np.ndarray            # and forward (left x up = fwd)

    @property
    def k(self) -> float:
        """The head's size over the neutral head's: what horn lengths and radii scale by."""
        return self.iod / IOD_REF

    @property
    def axes(self) -> np.ndarray:
        return np.stack([self.left, self.up, self.fwd], axis=1)


def head_frame(base, verts, cage) -> Head:
    """The head landmarks of the morphed body ``verts`` (``cage`` = its body cage): the eye,
    crown and skull-base joint markers (they move with every morph) and the ear, found as the
    widest part of the head below the eye line."""
    eye = A.joint_position(base, "l-eye", verts)
    top = A.joint_position(base, "head-2", verts)
    root = A.joint_position(base, "head", verts)
    iod = 2.0 * abs(float(eye[0]))
    k = iod / IOD_REF
    cv = cage[0]
    head = cv[cv[:, 1] > root[1] - 0.3 * k]
    band = ((head[:, 1] > root[1] - 0.15 * k) & (head[:, 1] < eye[1] + 0.25 * k)
            & (head[:, 2] < root[2] + 0.5 * k) & (head[:, 0] > 0))
    hb = head[band]
    xm = float(hb[:, 0].max())
    ear = hb[hb[:, 0] > xm - 0.08 * k].mean(axis=0)
    up = top - root
    up[0] = 0.0
    up /= max(float(np.linalg.norm(up)), 1e-12)
    fwd = np.array([0.0, 0.0, 1.0])
    fwd = fwd - float(fwd @ up) * up
    fwd /= np.linalg.norm(fwd)
    left = np.cross(up, fwd)
    return Head(np.asarray(eye, dtype=np.float64), np.asarray(top, dtype=np.float64),
                np.asarray(root, dtype=np.float64), np.asarray(ear, dtype=np.float64), iod,
                left, up, fwd)


def local_halfwidth(cage_v, point, radius: float = 0.35) -> float:
    """The head's half-width at the height of ``point``: the largest |x| of the cage vertices in a
    slab 0.08 thick around it (within ``radius`` front and back of it in z: not the ears)."""
    sl = (np.abs(cage_v[:, 1] - point[1]) < 0.08) & (np.abs(cage_v[:, 2] - point[2]) < radius * 2.0)
    return float(np.abs(cage_v[sl, 0]).max()) if sl.any() else 0.61


def smooth_normal(cage_v, quads, point, radius: float) -> np.ndarray:
    """The skull's surface normal at ``point``, smoothed over ``radius`` (Gaussian weights of the
    area-weighted vertex normals): a quad's own normal is a few degrees off on this coarse cage."""
    nrm = J.vertex_normals(cage_v, quads)
    d = np.linalg.norm(cage_v - point, axis=1)
    m = d < 2.0 * radius
    w = np.exp(-(d[m] / radius) ** 2)
    n = (nrm[m] * w[:, None]).sum(axis=0)
    ln = float(np.linalg.norm(n))
    return n / ln if ln > 1e-12 else np.array([0.0, 1.0, 0.0])


def _rotate(v, axis, angle):
    """``v`` turned about the unit ``axis`` by ``angle`` radians (Rodrigues)."""
    c, s = math.cos(angle), math.sin(angle)
    ax, ay, az = float(axis[0]), float(axis[1]), float(axis[2])
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    d = (ax * vx + ay * vy + az * vz) * (1.0 - c)
    return np.array([vx * c + (ay * vz - az * vy) * s + ax * d,
                     vy * c + (az * vx - ax * vz) * s + ay * d,
                     vz * c + (ax * vy - ay * vx) * s + az * d])


@dataclass(frozen=True)
class Site:
    """Where and how one (left-hand) horn leaves the head."""

    point: np.ndarray          # the skin point on the axis (on the cage surface)
    normal: np.ndarray         # the smoothed skull normal there
    axis: np.ndarray           # the horn's initial direction (unit): the normal tilted by ``tilt``
    bend: np.ndarray           # the unit direction the horn bends toward (perpendicular to the axis)
    halfwidth: float           # the head's half-width at that height
    scale: float               # local units -> world units: the head's size over the neutral head's
    base_radius: float         # the base radius in LOCAL units after the limits
    limited: str = ""          # what limited the radius ("" = nothing)

    @property
    def radius(self) -> float:
        """The base radius in world units."""
        return self.base_radius * self.scale

    def matrix(self, sink_offset: float = 0.0) -> np.ndarray:
        """The 4x4 that puts the local horn (``z`` = axis, ``x`` = bend, ``y = z x x``) on the head."""
        m = np.eye(4)
        z, x = self.axis, self.bend
        y = np.cross(z, x)
        m[:3, :3] = np.stack([x, y, z], axis=1) * self.scale
        m[:3, 3] = self.point + z * sink_offset
        return m


def horn_site(cage, head: Head, point, params: HornParams, *, scale: float = 1.0, roll: float = 0.0,
              tilt: float | None = None) -> Site:
    """The site of a horn at the skin ``point`` of the left half (``x`` > 0; a point on the right is
    treated as its mirror image): the radius from the head, the axis from the skull's surface normal.

    * the size follows the head: ``scale x (iod / IOD_REF)`` times the parameters, with the base
      radius limited to 0.35 of the head's half-width at that height and so that the tent keeps
      ``TENT_MARGIN`` off the midline;
    * the axis is the smoothed surface normal tilted by ``tilt`` degrees (default: the shape's, 15 to
      45 for a horn that reads as grown out of the skull) toward the shape's lean direction,
      expressed in the head's own axes;
    * the bend plane contains the axis and the shape's bend direction (also in head axes), turned about
      the axis by ``roll`` degrees."""
    cv, cq = cage
    point = np.asarray(point, dtype=np.float64)
    side = 1.0 if point[0] >= 0 else -1.0
    row = params.row
    k = head.k * scale
    R = params.base_radius * k
    n = smooth_normal(cv, cq, point, max(1.2 * R, 0.1))
    H = head.axes * np.array([side, 1.0, 1.0])      # head axes of this side (a point on the right mirrors)
    lean = H @ np.asarray(row["lean"], dtype=np.float64)
    bend = H @ np.asarray(row["bend"], dtype=np.float64)
    t = lean - float(lean @ n) * n
    if float(np.linalg.norm(t)) < 1e-6:
        t = head.up - float(head.up @ n) * n
    t /= np.linalg.norm(t)
    tau = math.radians(min(max(float(params.tilt if tilt is None else tilt), 0.0), 60.0))
    axis = n * math.cos(tau) + t * math.sin(tau)
    axis /= np.linalg.norm(axis)
    w = bend - float(bend @ axis) * axis
    if float(np.linalg.norm(w)) < 1e-6:
        w = np.cross(axis, head.up)
    w /= np.linalg.norm(w)
    if roll:
        w = _rotate(w, axis, math.radians(roll))
    hw = local_halfwidth(cv, point)
    limit, why = R, ""
    cap = HALFWIDTH_CAP * hw
    if limit > cap:
        limit, why = cap, "head width"
    margin = (abs(float(point[0])) - TENT_MARGIN) / TENT_RADIUS
    if limit > margin > 0.0:
        limit, why = margin, "midline"
    return Site(point=point, normal=n, axis=axis, bend=w, halfwidth=hw, scale=k,
                base_radius=limit / k, limited=why)



# --------------------------------------------------------------------------
# the skin the horn comes out of, in the horn's own frame


class PlaneField:
    """The skin as the plane through the horn's origin with the skull normal ``normal`` (in the
    horn's LOCAL frame; ``z`` = the axis): ``height(xy)`` is the skin's ``z`` at ``(x, y)``. What a
    horn is built against when it is not on a body (tests, the numbers of the docs)."""

    def __init__(self, normal):
        n = np.asarray(normal, dtype=np.float64)
        self.n = n / np.linalg.norm(n)

    def height(self, xy):
        xy = np.asarray(xy, dtype=np.float64)
        return -(self.n[0] * xy[:, 0] + self.n[1] * xy[:, 1]) / self.n[2]


class SkinField:
    """The skin as a height field over the horn's local ``xy`` plane, sampled by casting rays along
    the horn's axis at the body's triangles (``cage`` = ``(vertices, quads)`` in the world frame,
    ``matrix`` = local -> world): ``height(xy)`` is the ``z`` of the surface nearest ``z = 0``."""

    def __init__(self, cage, matrix, reach: float, fallback: PlaneField):
        v, q = cage
        inv = np.linalg.inv(matrix)
        loc = v @ inv[:3, :3].T + inv[:3, 3]
        near = ((np.abs(loc[q][:, :, 0]).min(axis=1) < reach) & (np.abs(loc[q][:, :, 1]).min(axis=1) < reach)
                & (np.abs(loc[q][:, :, 2]).min(axis=1) < max(reach, 0.8)))
        quads = q[near]
        tri = np.concatenate([loc[quads][:, [0, 1, 2]], loc[quads][:, [0, 2, 3]]])
        self.tri = tri
        self.fallback = fallback
        self.reach = reach

    def height(self, xy):
        xy = np.asarray(xy, dtype=np.float64)
        out = self.fallback.height(xy)
        if not len(self.tri) or not len(xy):
            return out
        a, b, c = self.tri[:, 0], self.tri[:, 1], self.tri[:, 2]
        e1, e2 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
        det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        ok = np.abs(det) > 1e-14
        det = np.where(ok, det, 1.0)
        za, zb, zc = a[:, 2], b[:, 2], c[:, 2]
        for lo in range(0, len(xy), 96):                      # all the points of a chunk against all the triangles
            p = xy[lo:lo + 96]
            d = p[:, None, :] - a[None, :, :2]
            u = (d[:, :, 0] * e2[None, :, 1] - d[:, :, 1] * e2[None, :, 0]) / det[None, :]
            w = (e1[None, :, 0] * d[:, :, 1] - e1[None, :, 1] * d[:, :, 0]) / det[None, :]
            hit = ok[None, :] & (u >= -1e-9) & (w >= -1e-9) & (u + w <= 1.0 + 1e-9)
            z = za[None, :] + u * (zb - za)[None, :] + w * (zc - za)[None, :]
            nearest = np.where(hit, np.abs(z), np.inf)
            j = np.argmin(nearest, axis=1)
            got = np.isfinite(nearest[np.arange(len(p)), j])
            out[lo:lo + 96][got] = z[np.arange(len(p)), j][got]
        return out


# --------------------------------------------------------------------------
# the model


def _ring(centre, n_vec, b_vec, a, b, ring_verts, lobe=0.0, keel=0.0):
    th = np.linspace(0.0, 2.0 * np.pi, ring_verts, endpoint=False)
    rad = np.ones_like(th)
    if lobe:
        rad = 1.0 + lobe * np.cos(3.0 * (th - keel))
    return (centre[None, :] + (a * rad * np.cos(th))[:, None] * n_vec[None, :]
            + (b * rad * np.sin(th))[:, None] * b_vec[None, :])


class HornModel:
    """The horn in its local frame: centreline, frames and the three layers' radii.

    Everything derives from one :class:`HornParams`. The meshes are built on demand
    (:meth:`skin`, :meth:`bone`), the skin is the one that is printed."""

    def __init__(self, params: HornParams):
        self.p = params
        self.R = float(params.base_radius)
        self.L = float(params.length)
        self._stations = self._make_stations()
        self._path = None

    # -- the radius law -----------------------------------------------------------------

    def radius(self, s, flare: bool = True):
        """The sheath's mean radius at ``s`` (units), including the root flare (``flare=False``: without
        it; the skin's own foot flares along the local skin, :meth:`skin`) and the growth rings;
        the buried stem holds its root value."""
        p, R, L = self.p, self.R, self.L
        s = np.asarray(s, dtype=np.float64)
        t = np.clip(s / L, 0.0, 1.0)
        tip = min(p.tip, 0.95 * R)
        base = tip + (R - tip) * (1.0 - t) ** TAPER_P
        sv = np.maximum(s, 0.0)
        fl = 1.0 + (p.flare_eff * np.exp(-sv / (FLARE_LEN * R)) if flare else 0.0)
        return base * fl * (1.0 + self.ring_wave(s))

    def ring_wave(self, s):
        """The growth rings' relative radius change at ``s``: ridges of pitch 0.8 R (0.55 of that at
        the tip), none for the first RING_FIRST R above the skin and none on the tip."""
        p, R, L = self.p, self.R, self.L
        amp = float(p.rings)
        s = np.asarray(s, dtype=np.float64)
        if amp <= 0.0:
            return np.zeros_like(s)
        pitch0 = RING_PITCH * R
        c = 1.0 - RING_PITCH_TIP
        # phase(s) = integral of ds / pitch(s), pitch(s) = pitch0 (1 - c s / L)
        sc = np.clip(s, 0.0, L)
        phase = -(L / (c * pitch0)) * np.log(np.maximum(1.0 - c * sc / L, 1e-9))
        phase0 = -(L / (c * pitch0)) * math.log(max(1.0 - c * RING_FIRST * R / L, 1e-9))
        x = phase - phase0
        ridge = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.mod(x, 1.0) ** 0.75))
        env = smoothstep((s + 0.3 * R) / (0.9 * R)) * \
            (1.0 - smoothstep((s - 0.84 * L) / (0.12 * L)))
        return amp * ridge * env

    # -- the centreline -------------------------------------------------------------------

    def _make_stations(self) -> np.ndarray:
        """The stations (s) the sheath's rings are at: the buried stem, the root's dense rings and
        the shaft, fine enough to resolve the growth rings."""
        p, R, L = self.p, self.R, self.L
        n = max(int(p.segments), 3)
        if p.rings > 0.0:
            n = max(n, int(math.ceil(L / (RING_PITCH * R * (1.0 - 0.6 * (1.0 - RING_PITCH_TIP))
                                           / RING_PER_PERIOD))))
        shaft = np.linspace(0.0, L, n + 1)
        dense = R * np.array([0.04, 0.1, 0.18, 0.28, 0.4, 0.55, 0.75, 1.0, 1.3])
        dense = dense[dense < 0.5 * L]
        st = np.unique(np.concatenate([shaft, dense]))
        # drop stations that crowd another by less than a tenth of the dense spacing
        keep = [0]
        for i in range(1, len(st)):
            if st[i] - st[keep[-1]] > 0.02 * R:
                keep.append(i)
        return st[keep]

    def _curvature(self, s):
        """``(kN, kB)``: the curvature along the bend direction N and the third axis B at ``s``."""
        p, R, L = self.p, self.R, self.L
        row = p.row
        psi = float(p.curve) * L / 2.0
        psi_b = float(p.twist) * L / 2.0
        s = np.asarray(s, dtype=np.float64)
        if row["profile"] == "lyre":
            w1 = 0.55 * L
            w2 = L - w1
            f = np.where(s < w1, (math.pi / (2.0 * w1)) * np.sin(np.pi * s / w1),
                         -0.7 * (math.pi / (2.0 * w2)) * np.sin(np.pi * np.clip(s - w1, 0, w2) / w2))
            kN, kB = psi * f, psi_b * f
        else:
            q = float(row["q"])
            S0 = q * L
            norm = math.log(q / (q - 1.0))
            f = 1.0 / (norm * (S0 - np.clip(s, 0.0, L)))
            kN, kB = psi * f, psi_b * f
        return kN, kB

    def helix_offset(self, s, N, B):
        """The corkscrew's displacement from the axis curve at ``s`` (``(n, 3)``): a circle of radius
        ``HELIX_RHO`` x the local radius in the axis' bend frame ``N``, ``B``, turning ``turns`` times over the
        length and growing in from the skin (a horn starts on its axis)."""
        p, L = self.p, self.L
        if not p.turns:
            return np.zeros((len(s), 3))
        sp = np.clip(s, 0.0, L)
        rho = HELIX_RHO * np.minimum(self.radius(sp, flare=False), 1.0 * self.R * 1.5) * smoothstep(sp / (0.2 * L))
        phi = 2.0 * np.pi * float(p.turns) * sp / L + math.radians(float(p.row.get("phase", 0.0)))
        return (rho * np.cos(phi))[:, None] * N + (rho * np.sin(phi))[:, None] * B

    def path(self):
        """``(s, centre, T, N, B)`` on a fine grid from the buried end to the tip: straight below the
        skin, then the axis curve bent by the curvature law (an equiangular spiral, a lyre), plus the
        corkscrew's helical offset; the frames are carried along by parallel transport."""
        if self._path is not None:
            return self._path
        L = self.L
        sink = self.p.sink
        s = np.unique(np.concatenate([np.linspace(0.0, L, 481), self._stations,
                                      np.linspace(-(sink + self.p.cap_depth) * 1.05, 0.0, 12)]))
        kN, kB = self._curvature(np.where(s > 0, s, 0.0))
        kN = np.where(s > 0, kN, 0.0)
        kB = np.where(s > 0, kB, 0.0)
        # a ring's radius times the curvature may not exceed BEND_LIMIT (the loft would fold on
        # its own inner side): extreme parameter combinations lose some turn instead
        r = self.radius(np.clip(s, 0.0, L), flare=False)
        mag = np.hypot(kN, kB)
        cap = BEND_LIMIT / np.maximum(r, 1e-9)
        sc = np.where(mag > cap, cap / np.maximum(mag, 1e-12), 1.0)
        kN, kB = kN * sc, kB * sc
        T = np.array([0.0, 0.0, 1.0])
        N = np.array([1.0, 0.0, 0.0])
        B = np.array([0.0, 1.0, 0.0])
        i0 = int(np.searchsorted(s, 0.0))          # s[i0] == 0
        n = len(s)
        pos = np.zeros((n, 3))
        Ts = np.zeros((n, 3))
        Ns = np.zeros((n, 3))
        Bs = np.zeros((n, 3))
        pos[i0], Ts[i0], Ns[i0], Bs[i0] = 0.0, T, N, B
        for i in range(i0 + 1, n):
            ds = s[i] - s[i - 1]
            km = 0.5 * (kN[i] + kN[i - 1]), 0.5 * (kB[i] + kB[i - 1])
            e = km[0] * N + km[1] * B
            ang = float(np.hypot(*km)) * ds
            pos[i] = pos[i - 1] + T * ds
            if ang > 1e-12:
                axis = np.array([T[1] * e[2] - T[2] * e[1], T[2] * e[0] - T[0] * e[2],
                                 T[0] * e[1] - T[1] * e[0]])
                axis /= math.sqrt(float(axis @ axis))
                Tm = _rotate(T, axis, 0.5 * ang)               # the midpoint heading for the step
                pos[i] = pos[i - 1] + Tm * ds
                T, N, B = _rotate(T, axis, ang), _rotate(N, axis, ang), _rotate(B, axis, ang)
            Ts[i], Ns[i], Bs[i] = T, N, B
        for i in range(i0 - 1, -1, -1):
            ds = s[i + 1] - s[i]
            pos[i] = pos[i + 1] - np.array([0.0, 0.0, 1.0]) * ds
            Ts[i], Ns[i], Bs[i] = Ts[i0], Ns[i0], Bs[i0]
        if self.p.turns:
            pos = pos + self.helix_offset(s, Ns, Bs)
            # the tangent of the new path and frames carried along it (minimal rotations)
            d = np.gradient(pos, s, axis=0)
            Ts = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
            N = Ns[i0]
            for i in range(n):
                if i == 0:
                    N = Ns[0] - float(Ns[0] @ Ts[0]) * Ts[0]
                else:
                    ax = np.cross(Ts[i - 1], Ts[i])
                    sn = float(np.linalg.norm(ax))
                    if sn > 1e-12:
                        N = _rotate(N, ax / sn, math.atan2(sn, float(Ts[i - 1] @ Ts[i])))
                N = N - float(N @ Ts[i]) * Ts[i]
                N = N / np.linalg.norm(N)
                Ns[i] = N
                Bs[i] = np.cross(Ts[i], N)
        self._path = (s, pos, Ts, Ns, Bs)
        return self._path

    def frames_at(self, st):
        """``(centre, T, N, B)`` interpolated on the path at the stations ``st``."""
        s, pos, T, N, B = self.path()
        out = [np.stack([np.interp(st, s, a[:, c]) for c in range(3)], axis=1)
               for a in (pos, T, N, B)]
        c, t, n, b = out
        # re-orthonormalise
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        n = n - (n * t).sum(1, keepdims=True) * t
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        b = np.cross(t, n)
        return c, t, n, b

    # -- sections -----------------------------------------------------------------------------

    def semi_axes(self, s, flare: bool = True):
        """``(a, b)``: the major (along the bend direction) and minor semi-axes at ``s``."""
        r = self.radius(s, flare)
        asp = float(self.p.row["aspect"])
        return r, r * asp

    # -- the meshes -----------------------------------------------------------------------------

    def _cap_rings(self, centre, direction, nvec, bvec, a, b, ring_verts, lobe, keel, height=None):
        """Rings of a rounded end on a quarter ellipsoid, from the section outward (``direction``);
        ``height`` is how far it reaches (default: the major radius: a dome)."""
        h = a if height is None else height
        out = []
        for deg in TIP_CAP:
            phi = math.radians(deg)
            out.append(_ring(centre + direction * h * math.sin(phi), nvec, bvec,
                             a * math.cos(phi), b * math.cos(phi), ring_verts, lobe, keel))
        return out

    def foot_rings(self, field=None):
        """The rings of the sheath's lower part in the horn's local frame: ``(bowl, foot, shaft_s)``. The
        buried bowl (rings from its bottom up), the foot (rings that follow the skin it comes out of, with
        a groove at the skin line and a bead above it: the keratin's crisp rim) and the stations of the
        shaft above the foot's last (planar) ring. ``field`` is the skin (a :class:`SkinField`; default:
        the plane of a skull normal at ``tilt``)."""
        p = self.p
        nv = int(p.ring_verts)
        R = self.R
        lobe = float(p.row["lobe"])
        keel = math.pi                                 # the keel on the outside of the bend
        th = np.linspace(0.0, 2.0 * np.pi, nv, endpoint=False)
        ct, st_ = np.cos(th), np.sin(th)
        if field is None:
            tau = math.radians(p.tilt)
            field = PlaneField((-math.sin(tau), 0.0, math.cos(tau)))
        Ra = float(self.radius(0.0))                   # the root radius (flare included)
        a0, b0 = (float(x) for x in self.semi_axes(0.0))
        z_skin = field.height(np.stack([a0 * ct, b0 * st_], axis=1))
        boss = min(max(float(p.boss), 0.0), 2.0)
        # a horn shorter than its own foot (a stub on a head that tilts it): the foot is compressed
        # so that the first shaft ring is at most 0.85 of the length up
        heights = np.array(FOOT_HEIGHTS)
        room = 0.85 * self.L - float(z_skin.max())
        if room < heights[-1] * Ra:
            f = max(room / (heights[-1] * Ra), 0.05)
            heights = np.where(heights > 0, heights * f, heights)
        s_top = max(float(z_skin.max()) + float(heights[-1]) * Ra, 0.0)
        st = self._stations
        shaft_s = st[st > s_top + 0.04 * R]      # (the foot's last ring is the planar ring at s_top)
        if not len(shaft_s):
            shaft_s = np.array([min(s_top + 0.5 * (self.L - s_top), self.L)])
        # the foot: K rings; ring i is at z = (1 - t_i) (skin + h_i Ra) + t_i s_top
        K = len(FOOT_HEIGHTS)
        foot = []
        for i, hh in enumerate(heights):
            hh = float(hh)
            t = (i / (K - 1)) ** FOOT_BLEND
            keep = 1.0 - t
            # the groove, the bead and the flare are functions of the NOMINAL height above the skin
            # (they fade as the ring leaves it); a ring that flares is wider, so it is lowered to the skin
            # at ITS OWN place, in two passes
            groove = GROOVE_D * math.exp(-((hh - GROOVE_H) / GROOVE_W) ** 2)
            bead = BEAD_E * math.exp(-((hh - BEAD_H) / BEAD_W) ** 2)
            fl = p.flare_eff * math.exp(-max(hh, 0.0) / FLARE_LEN)
            m = 1.0 + keep * (boss * (bead - groove) + fl)
            rad = 1.0 + (0.0 if lobe == 0.0 else lobe * np.cos(3.0 * (th - keel)))
            z = (1.0 - t) * (z_skin + hh * Ra) + t * s_top
            for _ in range(2):
                c, tt, nn, bb = self.frames_at(np.maximum(z, 0.0))
                a, b = self.semi_axes(np.maximum(z, 0.0), flare=False)
                v = c + (a * m * rad * ct)[:, None] * nn + (b * m * rad * st_)[:, None] * bb
                # the skin under this ring's own xy, in the horn's frame (the ring is nearly flat in
                # the xy plane: its centre is on the axis below the skin)
                z = (1.0 - t) * (field.height(v[:, :2]) + hh * Ra) + t * s_top
            c, tt, nn, bb = self.frames_at(np.maximum(z, 0.0))
            a, b = self.semi_axes(np.maximum(z, 0.0), flare=False)
            v = c + (a * m * rad * ct)[:, None] * nn + (b * m * rad * st_)[:, None] * bb
            v[:, 2] = np.where(z > 0.0, v[:, 2], z)       # below the skin point the stem is straight
            foot.append(v)
        # the buried bowl under the first ring: a quarter ellipse down to its centre
        depth = 0.85 * max(p.sink + p.cap_depth, 0.8 * R)
        c0 = foot[0].mean(axis=0)
        bowl = []
        for deg in BOWL_RINGS[::-1]:
            phi = math.radians(deg)
            ring = c0 + (foot[0] - c0) * math.cos(phi)
            bowl.append(ring - np.array([0.0, 0.0, 1.0]) * depth * math.sin(phi))
        return bowl, foot, shaft_s

    def skin(self, field=None) -> Mesh:
        """The keratin sheath, closed: from the bottom of the buried bowl up the foot, the shaft, to a
        rounded tip (see :meth:`foot_rings`)."""
        p = self.p
        nv = int(p.ring_verts)
        lobe = float(p.row["lobe"])
        keel = math.pi
        bowl, foot, shaft_s = self.foot_rings(field)
        rings = list(bowl) + list(foot)
        c, t, n, b = self.frames_at(shaft_s)
        a, bb = self.semi_axes(shaft_s, flare=False)
        for i in range(len(shaft_s)):
            rings.append(_ring(c[i], n[i], b[i], a[i], bb[i], nv, lobe, keel))
        rings += self._cap_rings(c[-1], t[-1], n[-1], b[-1], a[-1], bb[-1], nv, lobe, keel)
        return A._loft(rings)

    # -- the bone ---------------------------------------------------------------------------------

    def core_radius(self, s):
        """The cornual process' mean radius at ``s``: 0.82 R at the skin, a cone to the end of the
        core (``core`` x L), never closer than ``CORE_MIN_WALL`` R to the sheath's surface; below the
        skin it widens into the skull (the pedestal)."""
        p, R, L = self.p, self.R, self.L
        s = np.asarray(s, dtype=np.float64)
        Lc = max(float(p.row["core"]) * L, 1e-9)
        u = np.clip(s / Lc, 0.0, 1.0)
        cone = CORE_BASE * R * (1.0 - u) ** 1.0
        ped = CORE_BASE * R + (CORE_PEDESTAL - CORE_BASE) * R * smoothstep(-s / (0.9 * p.sink + 1e-9))
        core = np.where(s > 0, cone, ped)
        cap = self.radius(s) - CORE_MIN_WALL * R * np.where(s > 0, 1.0, 1.0)
        return np.minimum(core, np.maximum(cap, 0.02 * R)) * (s <= Lc)

    def bone(self, field=None) -> Mesh:
        """The cornual process: a closed solid, the sheath's own foot and bowl inset so that it is inside
        the sheath everywhere (the pedestal in the skull follows the skin the sheath comes out of), then a
        cone of ``core_radius`` up the shaft to the end of the core (``core`` x L)."""
        p = self.p
        nv = int(p.ring_verts)
        Lc = float(p.row["core"]) * self.L
        bowl, foot, shaft_s = self.foot_rings(field)
        rings = []
        for ring in list(bowl[1:]) + list(foot):          # (the bone's bottom is shallower than the sheath's)
            c = ring.mean(axis=0)
            r = float(np.linalg.norm(ring - c, axis=1).mean())
            # the cornual process is CORE_BASE x R wide at the skin and widens into the skull (the
            # pedestal) by at most CORE_PEDESTAL / CORE_BASE; never nearer the surface than the keratin wall
            f = min(CORE_PEDESTAL * self.R / max(r, 1e-12), 1.0 - CORE_MIN_WALL * self.R / max(r, 1e-12), 0.9)
            q = ring.copy()                                   # inset in the xy plane
            q[:, :2] = c[:2] + (ring[:, :2] - c[:2]) * f
            if len(rings) < len(bowl) - 1:                    # the bone's bowl is shallower than the sheath's
                q[:, 2] = ring[:, 2] + 0.25 * (foot[0][:, 2] - ring[:, 2])
            rings.append(q)
        asp = float(p.row["aspect"])
        up = shaft_s[shaft_s < Lc * 0.995]
        if len(up):
            c, t, n, b = self.frames_at(up)
            rr = self.core_radius(up)
            for i in range(len(up)):
                rings.append(_ring(c[i], n[i], b[i], rr[i], rr[i] * asp, nv))
            tail = self._cap_rings(c[-1], t[-1], n[-1], b[-1], rr[-1], rr[-1] * asp, nv, 0.0, 0.0)
        else:
            top = rings[-1]
            c = top.mean(axis=0)
            tail = [c + (top - c) * math.cos(math.radians(d)) + np.array([0.0, 0.0, 1.0]) * 0.05 * self.R
                    * math.sin(math.radians(d)) for d in TIP_CAP]
        return A._loft(rings + tail)

    # -- numbers -----------------------------------------------------------------------------------

    def centreline(self, n: int = 60):
        """``(s, points)``: the centreline at ``n`` stations from the skin to the tip."""
        s = np.linspace(0.0, self.L, n)
        c, *_ = self.frames_at(s)
        return s, c


# --------------------------------------------------------------------------
# the skin around the root


@dataclass(frozen=True)
class SkinPlan:
    """The numbers of the skin around one horn root, in WORLD units (the soft tissue layer
    and the junction spec both come from them)."""

    tent_radius: float       # rho_m: where the tent ends
    tent_height: float       # H: the tent's height over the skull at the horn
    fillet: float            # the tube fillet's width, in base radii (0: none)
    fillet_height: float     # ... and its height over its width


def skin_plan(params: HornParams, site: Site, fillet: float, fillet_height: float) -> SkinPlan:
    """The tent of one horn: radius ``TENT_RADIUS`` base radii (at least ``TENT_FLOOR``: the cage is
    0.125 an edge, and a tent under three edges across cannot be resolved), kept ``TENT_MARGIN`` off
    the midline; height ``tent_height`` base radii (at least ``TENT_MIN``)."""
    R = site.radius
    x0 = abs(float(site.point[0]))
    rho = min(max(TENT_RADIUS * R, TENT_FLOOR * site.scale), max(x0 - TENT_MARGIN, 0.5 * TENT_FLOOR))
    H = max(params.tent_height * R, TENT_MIN * site.scale)
    return SkinPlan(rho, H, float(fillet), float(fillet_height))


def moulded_cage(base, verts, spec):
    """``(vertices, quads)`` of the body cage with the skin moulded around ``spec`` alone: what
    :func:`~.junction.apply_junctions` gives for that one spec (the same displacement and the same
    validity pass), without the pipeline's bookkeeping. The horn's collar and foot are built against
    this surface, so they meet the skin the body will really have."""
    cage_v, cage_q = base.body_cage(verts)
    if spec is None or not spec.active:
        return cage_v, cage_q
    normals = J.vertex_normals(cage_v, cage_q)
    idx, d = J.junction_displacement(spec, cage_v, normals)
    total = np.zeros_like(cage_v)
    if len(idx):
        total[idx] += d
        total, _, _ = J.limit_displacement(cage_v, cage_q, total)
    return cage_v + total, cage_q


def junction_spec(name: str, model: "HornModel", site: Site, plan: SkinPlan, keeps_world=()):
    """The :class:`~.junction.JunctionSpec` of one LEFT horn in the horn's LOCAL frame (the kind
    carries it to each side with ``transformed``): the sheath's first part as the tube the scalp
    flows into, and the pedestal's tent as a mound that lifts the skin along the horn's own axis
    (not the skin normal: that is what made a plateau and a ledge across the forehead). ``keeps_world``
    are ``(centre, radius)`` in the horn's local frame: the ear, the eye, the other horn."""
    p = model.p
    R = model.R
    sc = site.scale
    nv = int(p.ring_verts)
    reach = 1.5 * p.sink + 1.5 * max(plan.fillet, 0.5) * R
    s = np.unique(np.concatenate([np.linspace(-p.sink * 0.9, 0.0, 4),
                                  np.linspace(0.0, reach, 8)[1:]]))
    c, t, n, b = model.frames_at(s)
    a, bb = model.semi_axes(s)
    radii = bb * math.cos(math.pi / nv) * HUG
    mounds = ()
    if plan.tent_height > 0 and plan.tent_radius > 0:
        mounds = (J.Mound((0.0, 0.0, 0.0), plan.tent_radius / sc, plan.tent_height / sc,
                          direction=(0.0, 0.0, 1.0)),)
    keep = tuple(J.Keep(tuple(ctr), rad, 0.6) for ctr, rad in keeps_world)
    tube_w = plan.fillet
    H_f = plan.fillet_height * tube_w * R
    move = 1.2 * (plan.tent_height / sc + H_f)
    return J.JunctionSpec(name, tuple(map(tuple, c)), tuple(radii),
                          width=tube_w if tube_w > 0 else 1.0, base_radius=R,
                          height=plan.fillet_height, sharpness=2.0, strength=1.0 if tube_w > 0 else 1.0,
                          mounds=mounds, keep=keep, max_move=move)


# --------------------------------------------------------------------------
# the soft tissue and the skull plate under the tent


def _annulus_solid(rho, theta, z_top, z_bot, n_theta):
    """A closed solid between two surfaces over an annulus of the horn's local ``xy`` plane: ``rho``
    (the radii, inner first), ``z_top`` / ``z_bot`` as ``(len(rho), n_theta)`` arrays. The top and
    bottom surfaces are joined by an inner and an outer wall: a torus."""
    n_r = len(rho)
    ct, st = np.cos(theta), np.sin(theta)
    xy = np.stack([np.outer(rho, ct), np.outer(rho, st)], axis=2)         # (n_r, n_theta, 2)
    top = np.concatenate([xy, z_top[:, :, None]], axis=2).reshape(-1, 3)
    bot = np.concatenate([xy, z_bot[:, :, None]], axis=2).reshape(-1, 3)
    verts = np.concatenate([top, bot])
    nb = n_r * n_theta
    quads = []
    for i in range(n_r - 1):
        for j in range(n_theta):
            jn = (j + 1) % n_theta
            a, b = i * n_theta + j, i * n_theta + jn
            c, d = (i + 1) * n_theta + jn, (i + 1) * n_theta + j
            quads.append([a, b, c, d])                       # top
            quads.append([nb + a, nb + d, nb + c, nb + b])   # bottom, reversed
    for j in range(n_theta):                                 # the inner and the outer walls
        jn = (j + 1) % n_theta
        a, b = j, jn
        quads.append([nb + a, nb + b, b, a])
        a, b = (n_r - 1) * n_theta + j, (n_r - 1) * n_theta + jn
        quads.append([a, b, nb + b, nb + a])
    tris = []
    for q in quads:
        tris.append([q[0], q[1], q[2]])
        tris.append([q[0], q[2], q[3]])
    mesh = Mesh(verts, np.asarray(tris, dtype=np.int64))
    if mesh.volume() < 0:
        mesh = Mesh(mesh.vertices, mesh.faces[:, [0, 2, 1]])
    return mesh


def soft_tissue(model: "HornModel", moulded, raw, rho_max: float, skin_thick: float, scalp_thick: float,
                n_rho: int = 7, n_theta: int = 32) -> Mesh:
    """The ``muscle`` layer in the horn's local frame: the soft tissue the scalp is tented over (galea,
    periosteum, the horn's germinal cuff), a lens around the cornual process from its surface out to
    ``rho_max`` (the tent's radius): its upper surface is the MOULDED skin less the skin's thickness, its
    lower one the skull's outer table, the RAW skin less the whole scalp's thickness. So it is thickest
    at the horn (the tent's height more) and ``scalp - skin`` thick where the skin is not lifted."""
    p = model.p
    rho_in = max(float(model.core_radius(0.0)), 0.5 * model.R)
    rho = np.linspace(rho_in, max(rho_max, rho_in * 1.5), n_rho)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    z_top = np.empty((n_rho, n_theta))
    z_bot = np.empty((n_rho, n_theta))
    for i, r in enumerate(rho):
        xy = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
        z_top[i] = moulded.height(xy) - skin_thick
        z_bot[i] = raw.height(xy) - scalp_thick
    z_top = np.maximum(z_top, z_bot + 0.2 * (scalp_thick - skin_thick))
    return _annulus_solid(rho, theta, z_top, z_bot, n_theta)


def skull_plate(model: "HornModel", raw, rho_max: float, scalp_thick: float, bone_thick: float,
                n_rho: int = 7, n_theta: int = 32) -> Mesh:
    """The skull's outer table around the horn, a plate ``bone_thick`` thick (the cornual process grows
    out of it): the lower boundary of the soft tissue."""
    rho_in = max(float(model.core_radius(0.0)), 0.5 * model.R)
    rho = np.linspace(rho_in, max(rho_max, rho_in * 1.5), n_rho)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    z_top = np.empty((n_rho, n_theta))
    for i, r in enumerate(rho):
        z_top[i] = raw.height(np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)) - scalp_thick - PLATE_GAP
    return _annulus_solid(rho, theta, z_top, z_top - bone_thick, n_theta)
