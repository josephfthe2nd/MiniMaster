"""The wing's root: a shoulder girdle on the back, a muscular upper arm, and the skin over both.

A wing is a forelimb. The body does not hand it over through a hole in the skin: there is a
girdle (a scapula plate on the ribs, a strut to the sternum), a mass of muscle that gathers over
the girdle and wraps the proximal arm (trapezius, the scapular/rhomboid plateau, the latissimus
fan, the deltoid-level cap of the humeral head, the arm's own belly), and ONE skin sheet over all
of it that flows tangentially into the arm sleeve. This module is that anatomy for a
:class:`~.kinds.WingKind` wing, built from the body it sits on (``docs/anatomy_wing.md`` has the
references and the measured numbers):

* :func:`landmarks`: where the girdle is, read off the live body (the shoulder, neck and
  spine-1 markers; the back surface is ray-cast), so it follows every slider and every body
  (child, heavy, tall, short, female);
* :class:`Mass`: one named muscle mass as ONE set of numbers (where it sits, which way it runs,
  how far it reaches, how high it stands over the plain skin, how deep it goes). The mass is
  the SOURCE of both the skin relief (:meth:`Mass.mound`, the junction's mound, displaced
  backwards: ``RELIEF_DIRECTION``) and the muscle mesh of the ecorche view
  (:meth:`WingRoot.muscles`: a pad whose top surface is the relief minus the skin's
  thickness), so the skin IS the envelope of the muscle plus its thickness, not a second
  unrelated shape;
* :class:`WingRoot`: the model of one placed wing: the masses, the bone chain, the arm sleeve
  (the rig's own upper-arm rings, ``wings.Sleeve``), the junction spec (:meth:`WingRoot.spec`)
  and the layers (:meth:`WingRoot.layers`: ``bone``, ``muscle``, ``skin``).

Everything here is authored for the LEFT wing in the MakeHuman frame (Y up, +X the character's
left, +Z the front, the back is -Z); the right wing is its x-mirror (``mirrored()``), which
keeps a pair exactly symmetric on a symmetric body.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from ..core.mesh import Mesh
from . import appendages as A
from . import junction as J

# --------------------------------------------------------------------------
# numbers (units: one MakeHuman unit = 10 cm of the character)

SHX_REF = 1.72            # the left shoulder joint's x on the default-macro body: the body size s = shx / this
SKIN_T = 0.06             # the skin and the fat over the muscle (0.05-0.1 over the scapular spine, wing/H08)
_EPS = 1e-12
# the masses of a back stand OUT of the back: the relief displaces the skin backwards, not along its own
# normal (which swings to the side at the flank and would shut the slot between the arm and the trunk,
# the posterior axillary fold, or swell the flank into a breast); None: along the skin normal
RELIEF_DIRECTION = (0.0, 0.0, -1.0)


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > _EPS else v


def _smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


# --------------------------------------------------------------------------
# the body: a surface to ray-cast


class BodySurface:
    """The (undeformed, morphed) body cage as a surface to ask questions of: where does a ray
    leave the skin, what is the normal there, which skin point lies over a given (x, y)."""

    def __init__(self, cage_v, cage_q):
        self.v = np.asarray(cage_v, dtype=np.float64)
        self.q = np.asarray(cage_q)
        self._vn = None                # the vertex normals, on first use (a ray to the skin needs none)
        self.memo: dict = {}          # what a model derives from THIS surface alone (the pair shares it)
        corners = self.v[self.q]
        self._centroid = corners.mean(axis=1)
        self._radius = np.linalg.norm(corners - self._centroid[:, None, :], axis=2).max(axis=1)

    @property
    def vn(self) -> np.ndarray:
        if self._vn is None:
            self._vn = J.vertex_normals(self.v, self.q)
        return self._vn

    def hit(self, origin, direction, near: float = 4.0):
        """The first hit of the ray ``origin + t * direction`` (0 < t <= ``near``) with the skin:
        ``(t, point, normal)`` (the normal interpolated from the vertex normals) or ``None``."""
        h = self.hit_quad(origin, direction, near)
        return None if h is None else h[:3]

    def hit_quad(self, origin, direction, near: float = 4.0):
        """:meth:`hit` and the cage quad it is on: ``(t, point, normal, quad, weights)`` with
        ``weights`` over the quad's four corners (``point`` is their weighted sum), or ``None``."""
        o = np.asarray(origin, dtype=np.float64)
        d = _unit(direction)
        rel = self._centroid - o
        along = rel @ d
        perp2 = (rel * rel).sum(axis=1) - along * along
        # the quads whose bounding sphere the ray's segment passes through: a few dozen of 13,378
        cand = np.nonzero((perp2 <= self._radius ** 2) & (along >= -self._radius)
                          & (along <= near + self._radius))[0]
        if not len(cand):
            return None
        best = (np.inf, None)
        for corners in ((0, 1, 2), (0, 2, 3)):
            ids = self.q[cand][:, corners]
            tri = self.v[ids]
            e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
            h = np.cross(d, e2)
            a = (e1 * h).sum(axis=1)
            ok = np.abs(a) > 1e-12
            f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
            s = o - tri[:, 0]
            u = f * (s * h).sum(axis=1)
            qv = np.cross(s, e1)
            w = f * (d * qv).sum(axis=1)
            t = f * (e2 * qv).sum(axis=1)
            m = ok & (u >= -1e-9) & (w >= -1e-9) & (u + w <= 1.0 + 1e-9) & (t > 1e-9) & (t <= near)
            if m.any():
                i = int(np.argmin(np.where(m, t, np.inf)))
                if t[i] < best[0]:
                    n = ((1.0 - u[i] - w[i]) * self.vn[ids[i, 0]] + u[i] * self.vn[ids[i, 1]]
                         + w[i] * self.vn[ids[i, 2]])
                    wq = np.zeros(4)
                    wq[list(corners)] = (1.0 - u[i] - w[i], u[i], w[i])
                    best = (float(t[i]), (o + d * t[i], _unit(n), int(cand[i]), tuple(float(x) for x in wq)))
        if best[1] is None:
            return None
        return (best[0],) + best[1]

    def first_hit_distances(self, origin, directions, near: float = 4.0) -> np.ndarray:
        """The distance to the first skin along each of ``directions`` from ``origin`` (``0`` where a ray
        meets none within ``near``): :meth:`hit` for many rays from ONE point, the quads near it sorted out once."""
        o = np.asarray(origin, dtype=np.float64)
        cand = np.nonzero(np.linalg.norm(self._centroid - o, axis=1) <= near + self._radius)[0]
        out = np.zeros(len(directions))
        if not len(cand):
            return out
        tris = [self.v[self.q[cand][:, c]] for c in ((0, 1, 2), (0, 2, 3))]
        e1 = [t[:, 1] - t[:, 0] for t in tris]
        e2 = [t[:, 2] - t[:, 0] for t in tris]
        rel = [o - t[:, 0] for t in tris]
        for k, d in enumerate(directions):
            d = _unit(d)
            best = np.inf
            for tri, a1, a2, s in zip(tris, e1, e2, rel):
                h = np.cross(d, a2)
                a = (a1 * h).sum(axis=1)
                ok = np.abs(a) > 1e-12
                f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
                u = f * (s * h).sum(axis=1)
                qv = np.cross(s, a1)
                w = f * (d * qv).sum(axis=1)
                t = f * (a2 * qv).sum(axis=1)
                m = ok & (u >= -1e-9) & (w >= -1e-9) & (u + w <= 1.0 + 1e-9) & (t > 1e-9) & (t <= near)
                if m.any():
                    best = min(best, float(t[m].min()))
            out[k] = 0.0 if not np.isfinite(best) else best
        return out

    def normal_at(self, point, radius: float = 0.45) -> np.ndarray:
        """The skin normal near ``point``: the distance-weighted mean of the vertex normals within
        ``radius`` (the nearest vertex's if there are none)."""
        d = np.linalg.norm(self.v - np.asarray(point, dtype=np.float64), axis=1)
        sel = np.nonzero(d < radius)[0]
        if not len(sel):
            return self.vn[int(np.argmin(d))]
        w = 1.0 / (d[sel] + 0.05)
        return _unit((self.vn[sel] * w[:, None]).sum(axis=0))

    def project(self, points, direction, back: float = 3.0, near: float = 1.2):
        """The skin points under ``points`` along ``direction`` (rays start ``back`` behind):
        ``(points, normals)`` of the same length (a point whose ray misses keeps its place and
        the direction as its normal)."""
        d = _unit(direction)
        P = np.asarray(points, dtype=np.float64)
        out, nrm = P.copy(), np.tile(-d, (len(P), 1))
        for i, p in enumerate(P):
            h = self.hit(p - d * back, d, near=back + near)
            if h is not None:
                out[i], nrm[i] = h[1], h[2]
        return out, nrm


# --------------------------------------------------------------------------
# landmarks


@dataclass(frozen=True)
class Landmarks:
    """The body's own girdle landmarks (left side; everything follows the live body)."""

    shx: float              # the left shoulder joint's x
    lt: float               # neck - spine-1 (the upper-trunk height)
    spine1: tuple           # joint-spine-1 (about T7, the inferior scapular angle level)
    neck: tuple             # joint-neck (C7)
    shoulder: tuple         # joint-l-shoulder (the human glenohumeral joint)
    elbow: tuple = (0.0, 0.0, 0.0)      # joint-l-elbow

    @property
    def s(self) -> float:
        """The body size every length below scales by (1 = the default-macro body)."""
        return self.shx / SHX_REF


def landmarks(base, verts) -> Landmarks:
    """The landmarks of the body ``verts`` (the morphed control vertices)."""
    sh = A.joint_position(base, "l-shoulder", verts)
    s1 = A.joint_position(base, "spine-1", verts)
    nk = A.joint_position(base, "neck", verts)
    el = A.joint_position(base, "l-elbow", verts)
    return Landmarks(shx=float(sh[0]), lt=float(nk[1] - s1[1]), spine1=tuple(s1), neck=tuple(nk),
                     shoulder=tuple(sh), elbow=tuple(el))


def socket_anchor(lm: Landmarks, surface: BodySurface, fx: float = A.WING_SOCKET_X,
                  fy: float = A.WING_SOCKET_Y):
    """Where the skin of the back lies over the wing's shoulder joint (version 2 of the placement,
    ``appendages.WING_SOCKET_*``; the left side): the point of the back at ``fx`` of the human
    shoulder joint's x and ``fy`` of the upper trunk's height above spine-1, seen from behind, as
    the cage quad it is on and the weights of that quad's corners: ``(quad, weights)``, so the
    placement engine binds it like any quad anchor (and mirrors it for the right wing). The joint
    itself lies ``WING_SOCKET_DEPTH`` x s under it, on the ribs under the shoulder blade. ``None``
    if no ray from behind meets the body there."""
    h = surface.hit_quad((fx * lm.shx, lm.spine1[1] + fy * lm.lt, -6.0), (0.0, 0.0, 1.0), near=9.0)
    return None if h is None else (h[3], h[4])


# --------------------------------------------------------------------------
# one muscle mass


@dataclass(frozen=True)
class Mass:
    """One named muscle mass of the wing's girdle, as ONE set of numbers.

    ``center`` is a point ON the skin, ``axis`` the unit direction the belly runs along (in the
    skin's tangent plane), ``half_long`` / ``half_wide`` the reach of the footprint along and
    across it, ``height`` how far the mass stands over the plain skin at its crest and ``pad``
    how deep it goes under it (the muscle mesh's thickness, not the relief). Everything else is
    derived: the skin relief (:meth:`mound`) and the muscle mesh (:meth:`WingRoot.muscles`)."""

    name: str
    label: str
    center: tuple
    axis: tuple
    half_long: float
    half_wide: float
    height: float
    pad: float

    def mound(self) -> J.Mound:
        """The skin relief of this mass: a smooth bulge along the skin normal over an elliptical
        footprint of ``half_long`` x ``half_wide`` and ``height`` at the crest."""
        return J.Mound(self.center, self.half_wide, self.height, axis=self.axis,
                       stretch=self.half_long / self.half_wide, direction=RELIEF_DIRECTION)

    def weight(self, points) -> np.ndarray:
        """The mound's influence (1 at the crest, 0 outside the footprint) at ``points``."""
        P = np.asarray(points, dtype=np.float64) - np.asarray(self.center)
        a = _unit(self.axis)
        par = P @ a
        perp = np.linalg.norm(P - par[:, None] * a, axis=1)
        d = np.sqrt((par / self.half_long) ** 2 + (perp / self.half_wide) ** 2)
        return J._bump(d)

    def mirrored(self) -> "Mass":
        return replace(self, center=(-self.center[0],) + tuple(self.center[1:]),
                       axis=(-self.axis[0],) + tuple(self.axis[1:]))


# --------------------------------------------------------------------------
# the masses on a body


# where each permanent mass sits, in the body's own units: (x as a share of the shoulder
# half-width, y as a share of the upper-trunk height above spine-1, axis (x, y) in the
# back's plane, reach along / across as shares of Lt / shx, height and depth in units of s).
# Tuned by eye on the neutral body and checked on the others (docs/anatomy_wing.md).
PERMANENT_MASSES = (
    # name, label, x/shx, (y - spine1)/Lt, axis (x, y), long/Lt, wide/shx, height/s, pad/s
    ("scapular", "infraspinatus / teres plateau on the shoulder blade, under the wing's root",
     0.66, 0.28, (0.40, 0.92), 0.46, 0.34, 0.14, 0.30),
    ("rhomboid", "rhomboid / middle trapezius from the blade's medial border toward the spine",
     0.30, 0.40, (0.90, -0.44), 0.24, 0.17, 0.06, 0.18),
    ("latissimus", "latissimus / teres fan sweeping to the flank",
     0.66, -0.55, (0.17, 0.98), 0.80, 0.32, 0.10, 0.25),
    ("trapezius", "trapezius slope from the neck to the wing's root",
     0.34, 0.74, (0.866, -0.5), 0.30, 0.15, 0.07, 0.20),
    # the wing's elevators over the supraspinous fossa: the rounded shoulder mound behind and above the root
    # (TAKEAWAYS wing checklist 2, wing/H24): along the scapular spine, between the root and the shoulder's top
    ("supraspinatus", "supraspinatus / the wing's elevators above the scapular spine, behind and above the root",
     0.66, 0.80, (0.90, 0.42), 0.30, 0.18, 0.15, 0.25),
)
# the shoulder boss: the muscle over the humeral head and the buried humerus (the wing's own deltoid and
# the origins of its elevators on the blade: the "rounded shoulder mound behind and above the root" of
# wing/H24). It runs from the skin over the joint to where the arm crosses the skin and a little beyond, so
# it follows the arm. (name, label, reach beyond the half distance joint -> crossing in sleeve radii, wide in
# sleeve radii, height in s, pad in s)
CAP = ("deltoid", "shoulder boss over the humeral head and the buried humerus", 1.0, 2.2, 0.30, 0.35)
BOSS_STEEP, BOSS_SHALLOW = 85.0, 40.0      # degrees between the arm and the plain skin: no boss / the whole boss
BOSS_FLAT, BOSS_FLAT_SHARE = 15.0, 0.5     # ... and less again for an arm lying along the back (cap_mass)
# the boss' crest moved this many sleeve radii from the run, half up the back and half the way the arm leaves (its
# acute side): the arm grows out of the flank of a shoulder mound behind and above it (wing/H24), and the skin
# under the leaning arm rises to meet it (the visible join's minimum at 1.3 root radii: 20 -> 29 degrees)
CAP_LIFT = 0.8
CAP_LIFT_DIR = 0.5                         # None: up the back; w: blend w of the arm's own direction (its acute side)
MIN_HEIGHT = 0.03                          # a mass lower than this (units) is left out


def _tangent(v, n):
    t = np.asarray(v, dtype=np.float64) - float(np.dot(v, n)) * np.asarray(n)
    return _unit(t)


def permanent_masses(lm: Landmarks, surf: BodySurface) -> list:
    """The masses that are body anatomy (they do not move when the wing does): the scapular
    plateau, the latissimus fan and the trapezius slope, on the back of THIS body."""
    out = []
    s1y = lm.spine1[1]
    for name, label, fx, fy, ax, lon, wide, h, pad in PERMANENT_MASSES:
        x = fx * lm.shx
        y = s1y + fy * lm.lt
        hit = surf.hit((x, y, -6.0), (0.0, 0.0, 1.0), near=9.0)
        if hit is None:
            continue
        _, c, n = hit
        axis = _tangent((ax[0], ax[1], 0.0), n)
        out.append(Mass(name, label, tuple(c), tuple(axis), lon * lm.lt, wide * lm.shx,
                        h * lm.s, pad * lm.s))
    return out


def cap_mass(lm: Landmarks, surf: BodySurface, shoulder, elbow, sleeve_r: float | None = None):
    """The shoulder boss: a ridge from the skin over the joint to where the arm (shoulder ->
    elbow) crosses the skin, so the arm emerges from the lateral flank of a hill whose crest is
    over the shoulder blade (the skin there tilts toward the arm: the exit is steep) and not
    from the flat of the back. ``None`` if the arm never crosses the skin."""
    S, E = np.asarray(shoulder, dtype=np.float64), np.asarray(elbow, dtype=np.float64)
    d = _unit(E - S)
    hit = surf.hit(S, d, near=float(np.linalg.norm(E - S)) + 2.0)
    if hit is None:
        return None
    _, X, n = hit
    over = surf.hit(S, (0.0, 0.0, -1.0), near=3.0)            # the skin over the joint
    A0 = over[1] if over is not None else X - d
    name, label, reach, wide, h, pad = CAP
    s = lm.s
    r = 0.4 * s if sleeve_r is None else float(sleeve_r)          # the arm's radius where it crosses the skin
    # the boss is what tilts the skin so that the arm leaves it steeply: an arm that already
    # leaves the plain skin at BOSS_STEEP degrees or more (the posed arm, say, sticking out of the
    # flank) needs none, one that leaves it at BOSS_SHALLOW or less (the spread wing: 45 degrees) gets all
    exit_deg = math.degrees(math.asin(min(abs(float(np.dot(d, n))), 1.0)))
    need = float(np.clip((BOSS_STEEP - exit_deg) / (BOSS_STEEP - BOSS_SHALLOW), 0.0, 1.0))
    # ... and an arm lying along the back (the folded wing) cannot be made to leave steeply: a whole boss beside it is
    # a lobe on the back (round 1's 0.48), so below BOSS_SHALLOW it eases to BOSS_FLAT_SHARE of it at BOSS_FLAT
    flat = float(np.clip((exit_deg - BOSS_FLAT) / (BOSS_SHALLOW - BOSS_FLAT), 0.0, 1.0))
    need *= BOSS_FLAT_SHARE + (1.0 - BOSS_FLAT_SHARE) * flat
    if need * h * s < MIN_HEIGHT:
        return None
    h = h * need
    run = X - A0
    half = 0.5 * float(np.linalg.norm(run))
    mid = A0 + 0.5 * run
    if CAP_LIFT:            # the crest behind and ABOVE the root (wing/H24): up the back, in the skin's plane
        mid = mid + CAP_LIFT * r * _tangent((0.0, 1.0, 0.0) if CAP_LIFT_DIR is None else
                                            CAP_LIFT_DIR * d + (1.0 - CAP_LIFT_DIR) * np.array([0.0, 1.0, 0.0]), n)
    c_hit = surf.hit(mid + n * 3.0, -n, near=6.0)
    c = c_hit[1] if c_hit is not None else mid
    nc = c_hit[2] if c_hit is not None else n
    along = _tangent(run, nc)
    return Mass(name, label, tuple(c), tuple(along), half + reach * r, wide * r, h * s, pad * s)


def envelope(mass: Mass, others) -> Mass | None:
    """``mass`` standing ``mass.height`` over the skin AS THE OTHERS LEAVE IT, not over the plain skin:
    its height less what the ``others`` already raise at its crest (the junction ADDS every mound, so a
    boss over the scapular plateau stood 0.48 units high where each was 0.19 and 0.34: the round-1
    folded wing's lobe). ``None`` if nothing is left of it (``MIN_HEIGHT``)."""
    c = np.asarray(mass.center, dtype=np.float64)[None]
    below = sum(float(m.height * m.weight(c)[0]) for m in others)
    h = mass.height - below
    if h < MIN_HEIGHT:
        return None
    return replace(mass, height=h)


def _relief(mass: Mass | None):
    """``points -> heights``: how far the junction raises the skin there for ``mass`` (``None``: no relief)."""
    if mass is None:
        return None
    return lambda points, m=mass: m.height * m.weight(points)


# --------------------------------------------------------------------------
# meshes of the layers

BONE_SHARE = 0.30         # the humerus' radius as a share of the sleeve's (a human humerus is 0.2-0.3 of the upper arm)
HEAD_SHARE = 0.45         # the humeral head's radius as a share of the muscle sleeve's at the joint (human: 2.2 of 4.5 cm)
GLENOID = (0.85, 0.30, 0.45)   # the glenoid's rim (a ring on the blade the head sits in): its radius, its thickness and
                               # how far under the head's centre, in head radii
# The shoulder blade of THIS body, which the wing's joint sits on (version 3 of the placement): the outline in the
# body's own units (x / shx, (y - spine-1) / Lt) from wing/H03 (Gray 204) and H01 (Albinus): the superior angle, the
# superior border to the neck of the glenoid, the lateral border down to the inferior angle (at about spine-1's level),
# the medial border (0.37 / 0.53 of the biacromial width apart at the spine's root / the inferior angle: 0.40 / 0.53
# of shx here), back up to the root of the spine. Height 0.85 Lt = 0.44 of the biacromial width (H01, H03: 0.42-0.47).
SCAPULA_BLADE = ((0.41, 0.88), (0.62, 0.86), (0.82, 0.80), (0.93, 0.68), (0.88, 0.52), (0.70, 0.24),
                 (0.53, -0.02), (0.47, 0.12), (0.40, 0.56))
# the spine of the scapula: from the medial border up across the blade to the acromion (rising 22.7-24.9 degrees,
# H03), its crest just under the skin at the acromion (the bony landmark of the shoulder top)
SCAPULA_SPINE = ((0.40, 0.57), (0.60, 0.66), (0.78, 0.74), (0.95, 0.81))
SCAPULA_DEPTH = 0.34      # s: the blade under the PLAIN skin (skin, fat and the 2-3 cm infraspinatus over it, wing/H08)
SCAPULA_THICK = 0.05      # s
SPINE_RADIUS = (0.05, 0.065)      # s: the spine's ridge at its root and at the acromion
ACROMION_DEPTH = 0.08     # s: the acromion's crest (the bone's surface) under the skin
CLAVICLE_RADIUS = (0.065, 0.05)   # s: at the acromion and at the sternum
CLAVICLE_DEPTH = 0.08     # s: the clavicle's surface under the skin of the shoulder top and of the chest


def _orient(mesh: Mesh) -> Mesh:
    return Mesh(mesh.vertices, mesh.faces[:, [0, 2, 1]]) if mesh.volume() < 0 else mesh


def _tube_from_rings(rings) -> Mesh:
    """A closed tube through ``rings`` (``(n, sides, 3)``), capped with a fan at each end."""
    n, sides, _ = rings.shape
    verts = np.vstack([rings.reshape(-1, 3), rings[0].mean(axis=0)[None], rings[-1].mean(axis=0)[None]])
    faces = []
    for i in range(n - 1):
        for k in range(sides):
            a, b = i * sides + k, i * sides + (k + 1) % sides
            c, d = (i + 1) * sides + (k + 1) % sides, (i + 1) * sides + k
            faces += [[a, b, c], [a, c, d]]
    c0, c1 = n * sides, n * sides + 1
    last = (n - 1) * sides
    for k in range(sides):
        kn = (k + 1) % sides
        faces += [[c0, kn, k], [c1, last + k, last + kn]]
    return _orient(Mesh(verts, np.asarray(faces, dtype=np.int64)))


def _ball(center, radius, n_lat: int = 7, n_lon: int = 14) -> Mesh:
    from .wings import _ball as ball
    return ball(center, radius, n_lat, n_lon)


def _lens(top_pole, top_rings, bottom_pole, bottom_rings, wall: bool) -> Mesh:
    """A closed solid between two nested stacks of rings: ``top_rings`` (``(n, sides, 3)``,
    inside out) over ``bottom_rings``, each ending in a pole. ``wall`` joins the outer rings with a
    band (a plate); without it the outer rings are the same points (a pad that ends in an edge,
    which then share their vertices)."""
    n, sides, _ = top_rings.shape
    if wall:
        verts = np.vstack([top_pole[None], top_rings.reshape(-1, 3), bottom_pole[None],
                           bottom_rings.reshape(-1, 3)])
        tb = lambda i, k: 1 + i * sides + k % sides                              # noqa: E731
        bb = lambda i, k: 2 + n * sides + i * sides + k % sides                  # noqa: E731
        bp = 1 + n * sides
        rim_b = bb
    else:
        # the outer ring is shared: bottom rings stop one short of it
        verts = np.vstack([top_pole[None], top_rings.reshape(-1, 3), bottom_pole[None],
                           bottom_rings[:-1].reshape(-1, 3)])
        tb = lambda i, k: 1 + i * sides + k % sides                              # noqa: E731
        bp = 1 + n * sides
        rim_b = None

        def bb(i, k):
            return tb(n - 1, k) if i == n - 1 else bp + 1 + i * sides + k % sides
    f = []
    for k in range(sides):
        # the fans wind as the rings do (round 3 wound them the other way: every pad's pole was a hole the bone
        # showed through in the ecorche, the round-3 review's "holes")
        f.append([0, tb(0, k), tb(0, k + 1)])                       # the top fan
        f.append([bp, bb(0, k + 1), bb(0, k)])                      # the bottom fan
    for i in range(n - 1):
        for k in range(sides):
            a, b, c, d = tb(i, k), tb(i, k + 1), tb(i + 1, k + 1), tb(i + 1, k)
            f += [[a, c, b], [a, d, c]]
            a, b, c, d = bb(i, k), bb(i, k + 1), bb(i + 1, k + 1), bb(i + 1, k)
            f += [[a, b, c], [a, c, d]]
    if wall:
        for k in range(sides):
            a, b = tb(n - 1, k), tb(n - 1, k + 1)
            c, d = rim_b(n - 1, k + 1), rim_b(n - 1, k)
            f += [[a, d, c], [a, c, b]]
    return _orient(Mesh(verts, np.asarray(f, dtype=np.int64)))


# --------------------------------------------------------------------------
# the muscles of the ecorche: sheets laid under the skin the junction moulds

# The muscles of the wing's girdle as a dissection shows them (wing/H06 Sobotta, H05 Gray 409; H07 for the deep
# layer). Each is an OUTLINE on the back in the body's own units (x / shx, (y - spine-1) / Lt; read off H06 with
# its shoulder joints, C7 and the inferior scapular angles as the landmarks, scratchpad junction/S7_wing_out/
# h06_grid.png), the LAYER it lies in (0 outermost: the wing's own deltoid over the root, then the trapezius kite,
# the latissimus, the infraspinatus / teres on the blade, the rhomboid under the trapezius) and its own thickness
# at the belly in s. The name is the MASS (PERMANENT_MASSES, CAP) whose relief the junction adds over it: that
# relief is muscle, it thickens the sheet. The top of every sheet lies the skin's thickness under the moulded
# skin (``MouldedSkin``) and, where an outer sheet covers it, under that sheet's own thickness: the skin is the
# envelope of the muscle plus its thickness (tests/test_wing_root.py casts rays in from the pipeline's shaped,
# subdivided skin of six bodies and measures it).
SHEETS = (
    # name, label, layer, thickness at the belly (s), outline ((x / shx, (y - spine-1) / Lt), ...)
    ("trapezius", "trapezius: the kite from the nape and the thoracic spine to the acromion and the scapular spine",
     1, 0.05, ((0.005, 1.18), (0.16, 1.15), (0.28, 1.06), (0.50, 0.94), (0.78, 0.86), (0.97, 0.82), (0.88, 0.73),
               (0.64, 0.63), (0.44, 0.53), (0.31, 0.22), (0.16, -0.24), (0.005, -0.56))),
    ("latissimus", "latissimus dorsi: the sheet from the lower thoracic spine over the inferior angle to the axilla",
     2, 0.05, ((0.01, -0.12), (0.30, 0.02), (0.60, 0.04), (0.86, 0.24), (0.95, 0.04), (0.88, -0.45),
               (0.76, -0.95), (0.54, -1.10), (0.30, -0.96), (0.01, -0.62))),
    ("scapular", "infraspinatus / teres on the blade, below the scapular spine (the wing's root leaves through it)",
     3, 0.12, ((0.43, 0.53), (0.63, 0.65), (0.87, 0.75), (0.98, 0.62), (0.99, 0.35), (0.86, 0.12),
               (0.63, -0.02), (0.50, -0.03), (0.44, 0.20))),
    ("supraspinatus", "supraspinatus: the supraspinous fossa above the scapular spine, under the trapezius",
     3, 0.10, ((0.41, 0.87), (0.62, 0.87), (0.86, 0.86), (0.97, 0.82), (0.78, 0.74), (0.60, 0.66), (0.42, 0.60))),
    ("rhomboid", "rhomboids: from the spines of C7-T5 to the blade's medial border, under the trapezius",
     4, 0.08, ((0.01, 0.96), (0.18, 0.87), (0.45, 0.56), (0.51, 0.05), (0.40, -0.01), (0.01, 0.34))),
)
CAP_SHEET = ("deltoid", "the wing's deltoid: the cap over its humeral head that the arm leaves through", 0, 0.10)
SHEET_RINGS, SHEET_SIDES = 7, 40
EDGE_DIP = 0.025           # units: the top of a sheet rounds off this much further under the skin at its edge (bellies read)
EDGE_FADE = 0.25           # the share of the radius over which a sheet's thickness (and its relief) closes to its edge
MUSCLE_LEVEL = 1           # the subdivision of the moulded skin the sheets are laid under (the picture's is 1 or 2)


def _polygon_centroid(poly) -> np.ndarray:
    P = np.asarray(poly, dtype=np.float64)
    a, b = P, np.roll(P, -1, axis=0)
    cr = a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]
    A = 0.5 * cr.sum()
    if abs(A) < _EPS:
        return P.mean(axis=0)
    return np.array([((a[:, 0] + b[:, 0]) * cr).sum(), ((a[:, 1] + b[:, 1]) * cr).sum()]) / (6.0 * A)


def _ray_polygon(c, dirs, poly) -> np.ndarray:
    """How far a 2D ray from ``c`` along each of ``dirs`` (unit, (m, 2)) runs to the polygon ``poly`` ((k, 2),
    closed): the FIRST crossing (``inf`` where a ray meets none)."""
    c = np.asarray(c, dtype=np.float64)
    d = np.asarray(dirs, dtype=np.float64).reshape(-1, 2)
    a = np.asarray(poly, dtype=np.float64)
    e = np.roll(a, -1, axis=0) - a
    r = a - c
    den = -d[:, None, 0] * e[None, :, 1] + d[:, None, 1] * e[None, :, 0]
    safe = np.where(np.abs(den) > 1e-12, den, 1.0)
    t = (-r[None, :, 0] * e[None, :, 1] + r[None, :, 1] * e[None, :, 0]) / safe
    u = (d[:, None, 0] * r[None, :, 1] - d[:, None, 1] * r[None, :, 0]) / safe
    ok = (np.abs(den) > 1e-12) & (t > 1e-9) & (u >= -1e-9) & (u <= 1.0 + 1e-9)
    return np.where(ok, t, np.inf).min(axis=1)


class MouldedSkin:
    """The body's skin as the junction moulds it, subdivided (``MUSCLE_LEVEL``), as the surface the muscle
    sheets are laid under: a :class:`~.collar.SkinPatch` around the back, in the LEFT frame (the right wing's
    side mirrored onto it). :meth:`from_behind` finds where a ray from behind (+Z) first meets it."""

    SEED_CELL = 0.05

    def __init__(self, vertices, faces, centre, radius, mirror: bool = False):
        from . import collar as CO
        V = np.asarray(vertices, dtype=np.float64)
        F = np.asarray(faces, dtype=np.int64)
        if mirror:
            V, F = V * MIRROR, F[:, ::-1]
        self.patch = CO.SkinPatch(V, F, np.asarray(centre, dtype=np.float64), float(radius))
        P = self.patch
        back = np.nonzero(P.N[:, 2] < -0.05)[0]
        xy = P.V[back, :2]
        # a cell holds a vertex wherever the back is (1.5 edge lengths; the level-1 body: about 0.09)
        edge = float(np.median(np.linalg.norm(P.V[P.F] - P.V[np.roll(P.F, 1, axis=1)], axis=2)))   # (any winding)
        self.cell = max(self.SEED_CELL, 1.5 * edge)
        self._o = xy.min(axis=0) - self.cell
        ij = np.floor((xy - self._o) / self.cell).astype(np.int64) + 1
        self._dims = ij.max(axis=0) + 2
        grid = np.full(tuple(self._dims), -1, dtype=np.int64)
        order = np.argsort(-P.V[back, 2])          # the smallest z (the backmost: the first skin seen from behind) is written last
        grid[ij[order, 0], ij[order, 1]] = back[order]
        self._grid = grid

    def from_behind(self, xy, iterations: int = 3):
        """``(points, normals, found)``: where a ray from behind (+Z) at each ``xy`` first meets the skin (a seed
        on the backmost back-facing vertex near it, then the tangent plane solved for the ray's z and the closest
        point, ``iterations`` times). ``found`` is False beyond the back (the skin there faces up or sideways)."""
        Q = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        P = self.patch
        ij = np.floor((Q - self._o) / self.cell).astype(np.int64) + 1
        best = np.full(len(Q), -1, dtype=np.int64)
        bd = np.full(len(Q), np.inf)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ii, jj = ij[:, 0] + di, ij[:, 1] + dj
                ok = (ii >= 0) & (ii < self._dims[0]) & (jj >= 0) & (jj < self._dims[1])
                g = np.where(ok, self._grid[np.clip(ii, 0, self._dims[0] - 1), np.clip(jj, 0, self._dims[1] - 1)], -1)
                d = np.where(g >= 0, ((P.V[np.maximum(g, 0), :2] - Q) ** 2).sum(axis=1), np.inf)
                better = d < bd
                best[better], bd[better] = g[better], d[better]
        found = best >= 0
        guess = np.maximum(best, 0)
        pts = np.column_stack([Q, P.V[guess, 2]])
        for _ in range(iterations):
            pt, n, _, guess = P.closest(pts, guess)
            nz = np.minimum(n[:, 2], -1e-3)
            z = pt[:, 2] - ((Q[:, 0] - pt[:, 0]) * n[:, 0] + (Q[:, 1] - pt[:, 1]) * n[:, 1]) / nz
            pts = np.column_stack([Q, z])
        pt, n, _, _ = P.closest(pts, guess)
        found &= (bd < (2.0 * self.cell) ** 2) & (n[:, 2] < -0.02)
        found &= np.linalg.norm(pt[:, :2] - Q, axis=1) < 0.04
        return pt, n, found

    def closest(self, points):
        return self.patch.closest(points)


@dataclass(frozen=True)
class Sheet:
    """One muscle of the ecorche (left frame): its outline on the back (2D, units), the layer it lies in, its own
    thickness at the belly (units) and the relief of its mass (``relief(points) -> heights``, or ``None``)."""

    name: str
    label: str
    layer: int
    thickness: float
    outline: tuple
    relief: object = None

    @property
    def centre(self) -> np.ndarray:
        return _polygon_centroid(self.outline)

    def radius_along(self, dirs) -> np.ndarray:
        return _ray_polygon(self.centre, dirs, self.outline)

    def rho(self, xy) -> np.ndarray:
        """How far out each 2D point is: 0 at the centre, 1 on the outline (``inf`` off its rays)."""
        q = np.asarray(xy, dtype=np.float64).reshape(-1, 2) - self.centre
        r = np.linalg.norm(q, axis=1)
        d = q / np.maximum(r, _EPS)[:, None]
        return r / np.maximum(self.radius_along(d), _EPS)

    def thick(self, rho, points) -> np.ndarray:
        """The sheet's thickness at shares ``rho`` of its radius and skin ``points``: its own belly (a lens) plus
        the relief its mass raises there, closing to nothing at the edge (0 outside)."""
        rho = np.asarray(rho, dtype=np.float64)
        inside = rho < 1.0
        rr = np.where(inside, rho, 1.0)
        own = self.thickness * np.sqrt(np.maximum(1.0 - rr * rr, 0.0))
        rel = np.zeros_like(rr) if self.relief is None else np.asarray(self.relief(points), dtype=np.float64)
        return np.where(inside, own + rel * _smooth((1.0 - rr) / EDGE_FADE), 0.0)


def _sheet_grid(sh: Sheet, skin: MouldedSkin, rings: int, sides: int):
    """The sample points of a sheet on the skin: ``(S, N, Q, rho)``: skin points, normals, their 2D ``xy`` and
    the share of the radius, the pole first then ``rings`` x ``sides`` (ring-major), or ``None`` if the sheet's
    centre is not on the back. Each spoke reaches the outline or, past the back (the skin turning up over the
    shoulder or round the flank), the last point a ray from behind meets on it."""
    th = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    dirs = np.column_stack([np.cos(th), np.sin(th)])
    rho = np.arange(1, rings + 1) / rings
    c = sh.centre
    R = sh.radius_along(dirs)
    if not np.all(np.isfinite(R)):
        return None
    pole, pole_n, pole_ok = skin.from_behind(c[None])
    if not pole_ok[0]:
        return None
    Q = c[None, None, :] + rho[:, None, None] * (R[None, :, None] * dirs[None])
    _, _, ok = skin.from_behind(Q.reshape(-1, 2))
    ok = ok.reshape(rings, sides)
    # the share of each spoke on the back: the last ring found, then bisected toward the first one missed (the
    # edge follows the body's outline, not the rings: a ring-wise cut drew a sawtooth on the flank)
    lo, hi = np.ones(sides), np.ones(sides)
    cut = ~ok.all(axis=0)
    for j in np.nonzero(cut)[0]:
        b = int(np.argmin(ok[:, j]))
        lo[j], hi[j] = (rho[b - 1] if b > 0 else 0.0), rho[b]
    for _ in range(5 if cut.any() else 0):
        mid = 0.5 * (lo + hi)
        _, _, okm = skin.from_behind(c[None] + (mid * R)[cut, None] * dirs[cut])
        lo[cut] = np.where(okm, mid[cut], lo[cut])
        hi[cut] = np.where(okm, hi[cut], mid[cut])
    reach = np.maximum(lo, 0.5 / rings)
    Q = c[None, None, :] + rho[:, None, None] * ((R * reach)[None, :, None] * dirs[None])
    S, N, ok = skin.from_behind(Q.reshape(-1, 2))
    S, N, ok = S.reshape(rings, sides, 3), N.reshape(rings, sides, 3), ok.reshape(rings, sides)
    for i in range(rings):                         # a straggler: the ring inside it (or the pole)
        miss = ~ok[i]
        if miss.any():
            S[i, miss] = S[i - 1, miss] if i else pole[0]
            N[i, miss] = N[i - 1, miss] if i else pole_n[0]
    r_all = np.concatenate([[0.0], np.repeat(rho, sides)])       # a shortened spoke closes at its own rim
    return (np.vstack([pole, S.reshape(-1, 3)]), np.vstack([pole_n, N.reshape(-1, 3)]),
            np.vstack([c[None], Q.reshape(-1, 2)]), r_all)


def _grid_lens(top, bottom, rings: int, sides: int, wall: bool) -> Mesh:
    return _lens(top[0], top[1:].reshape(rings, sides, 3), bottom[0], bottom[1:].reshape(rings, sides, 3), wall=wall)


def sheet_meshes(sheets, skin: MouldedSkin, s: float, rings: int = SHEET_RINGS, sides: int = SHEET_SIDES) -> dict:
    """``{name: Mesh}`` of the muscle ``sheets`` (left frame) laid under ``skin``: each a closed lens whose top is
    the skin's thickness (``SKIN_T``) under the skin, rounding ``EDGE_DIP`` x ``s`` further down at its edge, and,
    where outer sheets (a lower ``layer``) cover it, under their thickness too; whose bottom is its own thickness
    (:meth:`Sheet.thick`) under its top. A sheet whose outline runs past the back (over the shoulder top, round
    the flank) stops where rays from behind stop meeting the back."""
    out = {}
    ordered = sorted(sheets, key=lambda sh: sh.layer)
    for k, sh in enumerate(ordered):
        grid = _sheet_grid(sh, skin, rings, sides)
        if grid is None:
            continue
        S, N, Q, r_all = grid
        cover = np.zeros(len(S))
        for up in ordered[:k]:
            if up.layer < sh.layer:
                cover += up.thick(up.rho(Q), S)
        depth = SKIN_T + EDGE_DIP * _smooth((r_all - 0.55) / 0.45) + cover
        top = S - N * depth[:, None]
        bottom = top - N * sh.thick(r_all, S)[:, None]
        out[sh.name] = _grid_lens(top, bottom, rings, sides, wall=False)
    return out


# --------------------------------------------------------------------------
# the bones of the girdle


def _xy(lm: Landmarks, pts) -> tuple:
    """Points given as (x / shx, (y - spine-1) / Lt) as (x, y) of the left frame."""
    return tuple((fx * lm.shx, lm.spine1[1] + fy * lm.lt) for fx, fy in pts)


def scapula_blade(lm: Landmarks, skin: MouldedSkin, relief=None, rings: int = 5, sides: int = 36,
                  depth: float | None = None) -> Mesh | None:
    """The shoulder blade (left frame): ``SCAPULA_BLADE`` laid ``depth`` (default ``SCAPULA_DEPTH`` x s) under the
    PLAIN skin (under a moulded ``skin`` by that plus the ``relief`` the masses raise there: muscle, not bone),
    ``SCAPULA_THICK`` thick. ``None`` off the back."""
    sh = Sheet("scapula", "the shoulder blade", 0, 0.0, _xy(lm, SCAPULA_BLADE))
    grid = _sheet_grid(sh, skin, rings, sides)
    if grid is None:
        return None
    S, N, _, _ = grid
    rel = np.zeros(len(S)) if relief is None else np.asarray(relief(S), dtype=np.float64)
    top = S - N * ((SCAPULA_DEPTH * lm.s if depth is None else depth) + rel)[:, None]
    return _grid_lens(top, top - N * SCAPULA_THICK * lm.s, rings, sides, wall=True)


def acromion(lm: Landmarks, surface: BodySurface) -> np.ndarray | None:
    """The acromion (left frame): ``ACROMION_DEPTH`` x s under the skin at the top of the shoulder, over the back of the
    human shoulder joint (seen from above)."""
    x, z = 0.90 * lm.shx, lm.shoulder[2] - 0.22 * lm.s
    top = surface.hit((x, lm.neck[1] + 3.0, z), (0.0, -1.0, 0.0), near=8.0)
    if top is None:
        return None
    return top[1] - top[2] * (ACROMION_DEPTH + SPINE_RADIUS[1]) * lm.s


def scapula_spine(lm: Landmarks, skin: MouldedSkin, surface: BodySurface, relief=None, n: int = 12,
                  depth: float | None = None) -> tuple:
    """``(Mesh, acromion)``: the spine of the scapula (left frame), a ridge from the medial border along
    ``SCAPULA_SPINE`` rising from the blade toward the skin, to the acromion at the top of the shoulder
    (:func:`acromion`); ``(None, None)`` off the back."""
    poly = np.asarray(_xy(lm, SCAPULA_SPINE))
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)]) / max(float(seg.sum()), _EPS)
    u = np.linspace(0.0, 1.0, n)
    xy = np.column_stack([np.interp(u, t, poly[:, 0]), np.interp(u, t, poly[:, 1])])
    S, N, ok = skin.from_behind(xy)
    top = acromion(lm, surface)
    if not ok[0] or top is None:
        return None, None
    last = int(np.argmin(ok)) - 1 if not ok.all() else n - 1      # the back turns up into the shoulder top there
    S, N, u = S[:last + 1], N[:last + 1], u[:last + 1]
    rel = np.zeros(len(S)) if relief is None else np.asarray(relief(S), dtype=np.float64)
    root = SCAPULA_DEPTH * lm.s if depth is None else depth
    depth = root * (1.0 - _smooth(u)) + ACROMION_DEPTH * lm.s * _smooth(u) + rel * (1.0 - u)
    path = np.vstack([S - N * depth[:, None], top[None]])
    for _ in range(2):
        path[1:-1] = 0.25 * path[:-2] + 0.5 * path[1:-1] + 0.25 * path[2:]
    w = np.linspace(0.0, 1.0, len(path))
    r = (SPINE_RADIUS[0] + (SPINE_RADIUS[1] - SPINE_RADIUS[0]) * w) * lm.s
    return _path_tube(path, r), top


def clavicle(lm: Landmarks, surface: BodySurface, acromion) -> Mesh | None:
    """The strut of the girdle (left frame): the clavicle from the acromion over the top of the rib cage to the
    sternum's manubrium beside the midline, ``CLAVICLE_DEPTH`` x s under the skin all the way (a bird's coracoid and
    furcula brace its shoulder against the sternum, wing/A02, A07; on a winged human the clavicle is that strut: it
    runs OVER the thorax, not through it as round 3's did)."""
    if acromion is None:
        return None
    A0 = np.asarray(acromion, dtype=np.float64)
    front = surface.hit((0.12 * lm.shx, lm.neck[1] - 0.08 * lm.lt, 6.0), (0.0, 0.0, -1.0), near=9.0)
    if front is None:
        return None
    B = front[1] - front[2] * (CLAVICLE_DEPTH + CLAVICLE_RADIUS[1]) * lm.s
    t = np.linspace(0.0, 1.0, 16)
    path = [A0]
    for ti in t[1:-1]:
        # the skin over the chord, seen from up at the shoulder and from the front at the sternum, then sunk under it
        p = (1.0 - ti) * A0 + ti * B
        out = _unit(np.array([0.0, 1.0 - 0.7 * ti, 0.25 + 0.9 * ti]))
        hit = surface.hit(p + out * 3.0, -out, near=6.0)
        rad = CLAVICLE_RADIUS[0] + (CLAVICLE_RADIUS[1] - CLAVICLE_RADIUS[0]) * ti
        path.append(p if hit is None else hit[1] - hit[2] * (CLAVICLE_DEPTH + rad) * lm.s)
    path.append(B)
    path = np.asarray(path)
    for _ in range(3):                                     # an S-curved bar, not a polyline
        path[1:-1] = 0.25 * path[:-2] + 0.5 * path[1:-1] + 0.25 * path[2:]
    r = (CLAVICLE_RADIUS[0] + (CLAVICLE_RADIUS[1] - CLAVICLE_RADIUS[0]) * t) * lm.s
    return _path_tube(path, r)


def _torus(centre, normal, R: float, r: float, n: int = 24, m: int = 8) -> Mesh:
    """A closed ring round ``centre`` in the plane of ``normal``: major radius ``R``, tube radius ``r``."""
    nrm = _unit(normal)
    a = _unit(np.cross(nrm, (1.0, 0.0, 0.0)) if abs(nrm[0]) < 0.9 else np.cross(nrm, (0.0, 1.0, 0.0)))
    b = np.cross(nrm, a)
    u = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    v = np.linspace(0.0, 2.0 * math.pi, m, endpoint=False)
    radial = np.cos(u)[:, None] * a[None] + np.sin(u)[:, None] * b[None]
    pts = (np.asarray(centre)[None, None] + R * radial[:, None]
           + r * (np.cos(v)[None, :, None] * radial[:, None] + np.sin(v)[None, :, None] * nrm[None, None]))
    verts = pts.reshape(-1, 3)
    faces = []
    for i in range(n):
        for j in range(m):
            p00, p10 = i * m + j, ((i + 1) % n) * m + j
            p01, p11 = i * m + (j + 1) % m, ((i + 1) % n) * m + (j + 1) % m
            faces += [[p00, p10, p11], [p00, p11, p01]]
    return _orient(Mesh(verts, np.asarray(faces, dtype=np.int64)))


def _path_tube(path, radii, sides: int = 10) -> Mesh:
    """A closed tube of varying radius along a polyline."""
    path = np.asarray(path, dtype=np.float64)
    tang = np.gradient(path, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), _EPS)
    ref = np.array([0.0, 1.0, 0.0])
    if abs(float(tang[0] @ ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    th = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    rings = []
    for p, tg, r in zip(path, tang, radii):
        a = ref - float(ref @ tg) * tg
        a /= max(float(np.linalg.norm(a)), _EPS)
        b = np.cross(tg, a)
        ref = a
        rings.append(p + r * (np.cos(th)[:, None] * a[None] + np.sin(th)[:, None] * b[None]))
    return _tube_from_rings(np.stack(rings))


# the human arm hangs beside the trunk and its skin must not be moulded by the wing's root: spheres along
# the upper arm from this share of its length (below the deltoid, where the wing's own arm leaves the
# back) to the elbow, as wide as the arm is on THIS body (the median distance from the bone to the first
# skin it meets, measured around the bone) plus a margin
ARM_KEEP_STATIONS = (0.35, 0.55, 0.78)      # shares of the upper bone, from the shoulder to the elbow
ARM_KEEP_MARGIN = 0.15                      # in s
ARM_KEEP_FADE = 0.6
ARM_PROBES = 12


def arm_keep(lm: Landmarks, surface: BodySurface) -> tuple:
    """:class:`~.junction.Keep` spheres around the human upper arm (left frame), so the junction
    does not mould the arm's skin or deepen the armpit."""
    S, E = np.asarray(lm.shoulder), np.asarray(lm.elbow)
    axis = _unit(E - S)
    u = _unit(np.cross(axis, (0.0, 0.0, 1.0)))
    v = np.cross(axis, u)
    out = []
    for t in ARM_KEEP_STATIONS:
        c = S + (E - S) * t
        dirs = [math.cos(2.0 * math.pi * k / ARM_PROBES) * u + math.sin(2.0 * math.pi * k / ARM_PROBES) * v
                for k in range(ARM_PROBES)]
        radii = surface.first_hit_distances(c, dirs, near=2.5)
        out.append(J.Keep(tuple(c), float(np.median(radii)) + ARM_KEEP_MARGIN * lm.s, ARM_KEEP_FADE))
    return tuple(out)


# the wing's root belongs to the trunk (the guard of the S1b core only moulds that part's skin); cores
# without the ``owner`` field have no such guard
_OWNER = {"owner": "trunk"} if "owner" in J.JunctionSpec.__dataclass_fields__ else {}


# --------------------------------------------------------------------------
# the model of one placed wing

# the skin meets the arm sleeve along a fillet this high over its width (junction.py)
FILLET_HEIGHT = 0.3
JOIN_GUARD = False                 # see WingRoot.spec
# degrees: an arm leaving the plain skin more shallowly than COLLAR_MIN_EXIT gets no root collar, one leaving at
# COLLAR_FULL_EXIT or more the whole collar, between them its strength ramps (``collar_strength``): a pose or a
# slider that moves the exit by a degree changes the collar by a little, it does not switch it on or off (round 3:
# a hard cut at 43 with the heavy dragon collared at 44.7 and the heavy feathered folded wing not at 40.4)
COLLAR_MIN_EXIT = 40.0
COLLAR_FULL_EXIT = 45.0
FILLET_PULL = 1.0                 # the share of the in-plane redistribution of the fillet (junction.py)
# where along the humerus the junction tube is sampled (the feathered arm's from 0.15 of the way out: with the joint on
# the blade its rounded buried end is narrower than its tube, and a tube from the joint itself called skin over the
# heavy man's and the child's shoulders hidden that the arm does not cover)
TUBE_STATIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
TUBE_START = {"feathered": 0.15}
MIRROR = np.array([-1.0, 1.0, 1.0])


@dataclass(frozen=True)
class ArmProfile:
    """The rest upper arm of a wing as the rig built it: the ring centres of the humerus (body
    frame, root first), the radius of the skin surface at each (the circumscribed circle) and its
    inscribed circle, and the distance along it from the shoulder joint (0..1)."""

    centers: np.ndarray        # (n, 3)
    radius: np.ndarray         # (n,) the mean distance of a ring vertex to its centre
    inscribed: np.ndarray      # (n,) the smallest, times cos(pi / sides): the circle inside the facets
    t: np.ndarray              # (n,) 0 at the joint .. 1 at the elbow
    sides: int = 14

    def at(self, t) -> tuple:
        """``(centre, skin radius, inscribed radius)`` at ``t`` along the humerus."""
        c = np.stack([np.interp(t, self.t, self.centers[:, k]) for k in range(3)], axis=-1)
        return c, np.interp(t, self.t, self.radius), np.interp(t, self.t, self.inscribed)


def arm_profile(piece_vertices, sides: int, matrix) -> ArmProfile:
    """The :class:`ArmProfile` of an arm piece built as ``wings._tube`` builds it (ring-major,
    ``sides`` vertices a ring, two cap poles at the end), placed by ``matrix`` (the wing's 4x4)."""
    M = np.asarray(matrix, dtype=np.float64)
    V = np.asarray(piece_vertices, dtype=np.float64)[:-2]
    rings = (V @ M[:3, :3].T + M[:3, 3]).reshape(-1, sides, 3)
    n_ring = len(rings)
    elbow = n_ring // 2                          # the elbow is the middle sample of the arm polyline
    c = rings.mean(axis=1)[:elbow + 1]
    dist = np.linalg.norm(rings[:elbow + 1] - c[:, None], axis=2)
    seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    t = t / max(float(t[-1]), _EPS)
    # the inscribed circle of the polygon (the skin meets the sleeve just inside every facet, as ring_tube does)
    return ArmProfile(c, dist.mean(axis=1), dist.min(axis=1) * math.cos(math.pi / sides), t, int(sides))


def virtual_arm_profile(placed_rig, matrix, sleeve=None, rings: int = 9) -> ArmProfile:
    """The :class:`ArmProfile` of a wing built WITHOUT its arm piece (a feathered wing with ``bones``
    off: the arm is not drawn, the coverts cover it): the rest humerus of the rig (its shoulder and
    elbow joints, placed by ``matrix``) with the radius the feathered arm would have there
    (``wings.NOSE_RADIUS``, scaled by the skeleton, plus the ``sleeve``'s swell). The anatomy (the
    masses, the bones, the muscle) is the same as with the arm; the skin has nothing to flow into."""
    from . import wings as W
    sk = placed_rig.rig.skeleton
    j = placed_rig.joint_positions(None)
    S, E = np.asarray(j["shoulder"], dtype=np.float64), np.asarray(j["elbow"], dtype=np.float64)
    t = np.linspace(0.0, 1.0, rings)
    k = float(np.linalg.norm(np.asarray(matrix, dtype=np.float64)[:3, 0]))      # the placement's scale
    l_h = float(np.linalg.norm(np.asarray(sk.joints["elbow"]) - np.asarray(sk.joints["shoulder"])))
    r = (W.NOSE_RADIUS[0] + 0.5 * t * (W.NOSE_RADIUS[1] - W.NOSE_RADIUS[0])) * sk.scale
    if sleeve is not None:
        r = r + sleeve.extra(t, l_h)
    r = r * k
    return ArmProfile(S[None, :] + t[:, None] * (E - S)[None, :], r, r, t, 0)


def visible_join(plain: BodySurface, skin: BodySurface, S, E, r: float, k: float = 1.3, n: int = 24) -> tuple:
    """``(min, mean)`` of the angle (degrees) between the arm's axis (``S`` -> ``E``) and the tangent plane of the
    skin ``skin`` on a ring ``k`` root radii (``r``) round where the axis crosses the ``plain`` skin: the join one
    SEES (the angle where the axis itself meets the skin is hidden inside the arm). 90: the skin is square to the
    arm; the minimum is the side the arm leans over (negative: the skin folds over the arm). ``None`` if the axis
    does not cross the skin."""
    S, E = np.asarray(S, dtype=np.float64), np.asarray(E, dtype=np.float64)
    d = _unit(E - S)
    h = plain.hit(S, d, near=float(np.linalg.norm(E - S)) + 2.0)
    if h is None:
        return None
    X, nrm = h[1], h[2]
    u = _unit(np.cross(nrm, d))
    v = np.cross(nrm, u)
    out = []
    for th in np.linspace(0.0, 2.0 * math.pi, n, endpoint=False):
        q = X + k * r * (math.cos(th) * u + math.sin(th) * v)
        hh = skin.hit(q + 1.5 * nrm, -nrm, near=3.0)
        if hh is not None:
            out.append(math.degrees(math.asin(max(-1.0, min(1.0, float(d @ hh[2]))))))
    if not out:
        return None
    return float(np.min(out)), float(np.mean(out))


class WingRoot:
    """The shoulder anatomy of one placed wing (``Placed.model``): see the module docstring.

    Built once per placed wing from the body's landmarks and surface, the placed rig (for the
    posed joints) and the rest arm; everything pose-dependent (the deltoid cap, the junction
    tube, the layers) is derived on demand from the controls."""

    def __init__(self, letter: str, rig, matrix, lm: Landmarks, surface: BodySurface,
                 profile: ArmProfile, fillet: float, style: str = "membrane",
                 fillet_height: float = FILLET_HEIGHT, tube: bool = True, body=None):
        self.letter = letter
        self.rig = rig                      # a wingrig.PlacedWing
        self.matrix = np.asarray(matrix, dtype=np.float64)
        self.lm = lm
        self.surface = surface
        self.profile = profile
        self.fillet = float(fillet)
        self.fillet_height = float(fillet_height)
        self.style = style
        # False: the wing has no arm piece (``virtual_arm_profile``): the junction is the masses only (no
        # tube the skin could flow into: there is no arm skin), there is no collar, the layers have no sleeve
        self.tube = bool(tube)
        self._left_masses = None
        self._left_keep = None
        self._exit_r = None
        # ``(base, verts, controls -> JunctionSpec)``: the body this wing is placed on and its own junction, so the
        # muscles are laid under the skin as the junction moulds it (``moulded``); None: no body (a synthetic test)
        self.body = body
        self._moulded: dict = {}

    def _keep(self) -> tuple:
        if self._left_keep is None:
            self._left_keep = self.surface.memo.get("keep")
            if self._left_keep is None:
                self._left_keep = self.surface.memo["keep"] = arm_keep(self.lm, self.surface)
        return self._left_keep

    # -- the arm, in the left frame ----------------------------------------------------

    def _to_left(self, p):
        return np.asarray(p, dtype=np.float64) * (MIRROR if self.letter == "r" else 1.0)

    def _pose(self, controls):
        ctl = dict(controls or {})
        preset = ctl.pop("preset", None)
        return self.rig.rig.pose_from_controls(preset, **ctl) if (ctl or preset) else None

    def _carried(self, points, controls=None) -> np.ndarray:
        """Rest points of the humerus (the placed frame) carried to the pose ``controls``."""
        pose = self._pose(controls)
        if pose is None:
            return np.asarray(points, dtype=np.float64)
        M = self.rig.skin_matrices_body(pose)["elbow"]
        return np.asarray(points, dtype=np.float64) @ M[:3, :3].T + M[:3, 3]

    def joints(self, controls=None) -> tuple:
        """``(shoulder, elbow)`` of the posed upper arm in the LEFT frame."""
        j = self.rig.joint_positions(self._pose(controls))
        return self._to_left(j["shoulder"]), self._to_left(j["elbow"])

    # -- the masses --------------------------------------------------------------------

    def permanent(self) -> list:
        if self._left_masses is None:
            self._left_masses = self.surface.memo.get("masses")
            if self._left_masses is None:
                self._left_masses = self.surface.memo["masses"] = permanent_masses(self.lm, self.surface)
        return self._left_masses

    def masses(self, controls=None) -> list:
        """All the masses (left frame) of the wing under ``controls``: the permanent ones and the
        cap, which follows the posed arm."""
        S, E = self.joints(controls)
        out = list(self.permanent())
        cap = cap_mass(self.lm, self.surface, S, E, self.exit_radius())
        if cap is not None:
            cap = envelope(cap, out)
        if cap is not None:
            out.append(cap)
        return out

    def exit_radius(self) -> float:
        """The radius of the rest arm's skin where its axis crosses the plain skin (the sleeve at the root;
        the elbow's radius if the rest arm never crosses it)."""
        if self._exit_r is None:
            c = self._to_left(self.profile.centers)
            L = float(np.linalg.norm(c[-1] - c[0]))
            hit = self.surface.hit(c[0], c[-1] - c[0], near=L + 1.0)
            tt = 1.0 if hit is None else min(hit[0] / max(L, _EPS), 1.0)
            self._exit_r = float(np.interp(tt, self.profile.t, self.profile.radius))
        return self._exit_r

    def collar_up(self, controls=None):
        if not self.tube:
            return None
        if not self.collar_fits(controls):
            return None
        return self._collar_up(controls)

    def collar_fits(self, controls=None) -> bool:
        """Whether the root collar can be built for the arm in pose ``controls``: an arm with its own piece
        that leaves the PLAIN skin more steeply than ``COLLAR_MIN_EXIT`` degrees. An arm lying along the back (the
        folded wing: 12-35 degrees in the third placement) leans over the skin on most of its ring: collar.py cuts
        the width to its floor there and the hair-thin shell that is left crossed itself 300-1,200 times, with rim
        normals 175 degrees off and a leak on the heavy body (measured on six bodies before the cut-off)."""
        return self.collar_strength(controls) > 0.0

    def collar_strength(self, controls=None) -> float:
        """How much of the root collar the arm in pose ``controls`` gets (0..1): nothing at an exit of
        ``COLLAR_MIN_EXIT`` degrees or less, all of it at ``COLLAR_FULL_EXIT`` or more, a smoothstep between."""
        if not self.tube:
            return 0.0
        ang = self.exit_angle(controls)
        if ang is None:
            return 0.0
        return float(_smooth((ang - COLLAR_MIN_EXIT) / (COLLAR_FULL_EXIT - COLLAR_MIN_EXIT)))

    def exit_angle(self, controls=None) -> float | None:
        """The angle (degrees, 90 = square) at which the posed arm's axis leaves the PLAIN skin, or ``None``
        if it does not cross it."""
        S, E = self.joints(controls)
        d = _unit(E - S)
        hit = self.surface.hit(S, d, near=float(np.linalg.norm(E - S)) + 2.0)
        if hit is None:
            return None
        return math.degrees(math.asin(min(abs(float(d @ hit[2])), 1.0)))

    def _collar_up(self, controls=None):
        """The open side of the root collar, in the PLACED frame (``collar.CollarProfile.up``): the
        direction on the back the posed arm comes from (minus its direction, projected into the skin's
        tangent plane where it crosses the skin): the obtuse side of the join, where the back's skin runs
        into the arm. ``None`` if the arm does not cross the skin."""
        S, E = self.joints(controls)
        d = _unit(E - S)
        hit = self.surface.hit(S, d, near=float(np.linalg.norm(E - S)) + 2.0)
        if hit is None:
            return None
        n = hit[2]
        up = -(d - float(d @ n) * n)
        if float(np.linalg.norm(up)) < 1e-6:          # the arm leaves square: any side is open
            up = np.array([0.0, -1.0, 0.0])
        return self._to_left(_unit(up))

    # -- the junction ------------------------------------------------------------------

    def spec(self, controls=None, name: str = "wing"):
        """The :class:`~.junction.JunctionSpec` of this wing's root under ``controls``."""
        ts = np.asarray(TUBE_STATIONS, dtype=np.float64)
        ts[0] = max(ts[0], TUBE_START.get(self.style, 0.0))
        pts, _, r_in = self.profile.at(ts)
        # the rest arm's ring centres, carried by the pose (the humerus' skin follows the bone named after
        # its far joint, "elbow"; the sleeve is rigid over it): the tube lies on the arm's own axis, which
        # is NOT the shoulder -> elbow line of the joints (0.2-0.25 wing units off it on a feathered wing,
        # whose arm is lifted into the middle of the feather stack: a tube on the joint line lay half
        # outside the arm and the core's conform step cut it away)
        line = self._to_left(self._carried(pts, controls))
        ref = float(r_in[-1])
        if not self.tube:
            line, r_in = (), ()
        spec = J.JunctionSpec(name, line, tuple(np.asarray(r_in) * J.HUG), width=self.fillet,
                              base_radius=ref if self.tube else None,
                              height=self.fillet_height, pull=FILLET_PULL,
                              mounds=tuple(m.mound() for m in self.masses(controls)),
                              keep=self._keep(), join_guard=JOIN_GUARD, **_OWNER)
        return spec.mirrored() if self.letter == "r" else spec

    # -- the layers ---------------------------------------------------------------------

    def _sleeve_rings(self, skin: Mesh) -> np.ndarray:
        """The rings of the humerus of the placed (posed) skin mesh ``skin``: ``(n, sides, 3)``, in
        the placed frame (the wing's own arm piece is a ring-major tube; see :func:`arm_profile`)."""
        a, b = self.rig.rig.ranges["arm"]
        sides = self.profile.sides
        rings = skin.vertices[a:b - 2].reshape(-1, sides, 3)
        return rings[:len(self.profile.t)]

    def arm_mesh(self, skin: Mesh) -> Mesh:
        """The arm piece of the placed (posed) skin mesh ``skin``: the wall the root collar meets."""
        a, b = self.rig.rig.ranges["arm"]
        arm = next(q for q in self.rig.rig.pieces if q.name == "arm")
        return Mesh(skin.vertices[a:b], arm.mesh.faces)

    def _virtual_rings(self, controls=None, sides: int = 16) -> np.ndarray:
        """Rings of the arm a wing without its arm piece would have (:func:`virtual_arm_profile`),
        along the POSED humerus, in the placed frame."""
        S, E = self.joints(controls)
        t, r = self.profile.t, self.profile.radius
        axis = _unit(E - S)
        u = _unit(np.cross(axis, (0.0, 0.0, 1.0)) if abs(axis[2]) < 0.9 else np.cross(axis, (1.0, 0.0, 0.0)))
        v = np.cross(axis, u)
        th = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
        ring = np.cos(th)[:, None] * u[None] + np.sin(th)[:, None] * v[None]
        rings = (S[None, None, :] + t[:, None, None] * (E - S)[None, None, :]
                 + r[:, None, None] * ring[None, :, :])
        return self._to_left(rings)

    def sleeve_meshes(self, skin: Mesh, controls=None) -> dict:
        """``{"muscle": Mesh, "bone": Mesh}`` of the arm in the placed frame: the skin's own rings
        scaled about their centres to the muscle (the skin minus its thickness) and to the bone
        (a share of the sleeve), so the three nest exactly and pose with the skin. A wing without its
        arm piece has the rings of :func:`virtual_arm_profile` (the arm under its coverts)."""
        rings = self._sleeve_rings(skin) if self.tube else self._virtual_rings(controls)
        c = rings.mean(axis=1, keepdims=True)
        r = np.linalg.norm(rings - c, axis=2).mean(axis=1)
        r_mus = np.maximum(r - SKIN_T, 0.55 * r)
        r_bone = BONE_SHARE * r
        out = {"muscle": _tube_from_rings(c + (rings - c) * (r_mus / r)[:, None, None])}
        # the shaft runs from inside the head (on the joint) and stops a ring short of the elbow, inside the cap
        bone = c[:-1] + (rings[:-1] - c[:-1]) * (r_bone[:-1] / r[:-1])[:, None, None]
        step = float(np.linalg.norm(c[1, 0] - c[0, 0]))
        w = min(0.7 * HEAD_SHARE * float(r_mus[0]) / max(step, _EPS), 0.9)
        bone[0] = (1.0 - w) * bone[0] + w * bone[1]
        out["shaft"] = _tube_from_rings(bone)
        out["bone"] = Mesh.merge([out["shaft"], self._head(rings, r_mus, controls)])
        return out

    def head_radius(self) -> float:
        """The humeral head's radius (``HEAD_SHARE`` of the muscle sleeve's at the joint)."""
        r = float(self.profile.radius[0])
        return HEAD_SHARE * max(r - SKIN_T, 0.55 * r)

    def _head(self, rings, r_mus, controls=None) -> Mesh:
        """The humeral head: a ball ON the joint (the pivot the rig turns the arm about), inside the buried end of
        the sleeve."""
        S, _ = self.joints(controls)
        R = HEAD_SHARE * float(r_mus[0])
        if self.letter == "r":          # built on the left and mirrored: the pair is an exact mirror pair
            return _ball(S, R).transform(np.diag([-1.0, 1.0, 1.0, 1.0]))
        return _ball(S, R)

    def glenoid(self) -> Mesh | None:
        """The socket the head sits in (left frame): a rim on the blade round the rest joint, under the head's centre
        (the joint does not move when the arm turns about it)."""
        S, _ = self.joints(None)
        over = self.surface.hit(S, (0.0, 0.0, -1.0), near=4.0)
        n = np.array([0.0, 0.0, -1.0]) if over is None else over[2]          # outward
        R = self.head_radius()
        k, t, down = GLENOID
        return _torus(S - n * down * R, n, k * R, t * R)

    # -- the muscles -------------------------------------------------------------------

    def moulded(self, controls=None, shaped=None) -> MouldedSkin | None:
        """The skin the muscles lie under (left frame): ``shaped`` (the body's vertices after every junction, as a
        view meshes them) or, without it, the body moulded by this wing's own junction under ``controls``,
        subdivided ``MUSCLE_LEVEL`` times. ``None`` without a body."""
        if self.body is None:
            return None
        import json
        from . import collar as CO
        base, verts, junction = self.body
        key = (json.dumps(controls or {}, sort_keys=True, default=str),
               None if shaped is None else CO.verts_digest(shaped))
        if key in self._moulded:
            return self._moulded[key]
        if shaped is None:
            spec = junction(controls) if junction is not None else None
            shaped = verts if spec is None else J.apply_junctions(base, verts, [spec])
        V, F = CO.shaped_skin(base, shaped, MUSCLE_LEVEL)
        lm = self.lm
        centre = np.array([0.5 * lm.shx, lm.spine1[1] + 0.15 * lm.lt, lm.spine1[2]])
        skin = MouldedSkin(V, F, centre, 1.3 * math.hypot(0.75 * lm.shx, 1.45 * lm.lt), mirror=self.letter == "r")
        if len(self._moulded) >= 2:
            self._moulded.pop(next(iter(self._moulded)))
        self._moulded[key] = skin
        return skin

    def sheets(self, controls=None) -> list:
        """The muscle :class:`Sheet` s of this wing (left frame) under ``controls``: ``SHEETS`` on this body, each
        carrying the relief of the mass of its name, and the wing's deltoid over the root (``CAP_SHEET``)."""
        lm = self.lm
        masses = {m.name: m for m in self.masses(controls)}
        out = []
        for name, label, layer, thick, outline in SHEETS:
            poly = tuple((fx * lm.shx, lm.spine1[1] + fy * lm.lt) for fx, fy in outline)
            out.append(Sheet(name, label, layer, thick * lm.s, poly, _relief(masses.get(name))))
        cap = self._cap_outline(controls, masses.get(CAP[0]))
        if cap is not None:
            name, label, layer, thick = CAP_SHEET
            out.append(Sheet(name, label, layer, thick * lm.s, cap, _relief(masses.get(CAP[0]))))
        return out

    def _cap_outline(self, controls, cap: Mass | None, n: int = 24):
        """The wing's deltoid on the back (2D, left frame): the shoulder boss' footprint (``cap_mass``), or, where
        the arm leaves steeply and has none, a round cap 1.7 root radii about where the arm crosses the skin."""
        r = self.exit_radius()
        if cap is not None:
            c, a, lo, wi = np.asarray(cap.center), _unit(cap.axis), cap.half_long, cap.half_wide
        else:
            S, E = self.joints(controls)
            d = _unit(E - S)
            hit = self.surface.hit(S, d, near=float(np.linalg.norm(E - S)) + 2.0)
            if hit is None:
                return None
            c, a, lo, wi = hit[1], _tangent(d, hit[2]), 1.7 * r, 1.7 * r
        n0 = self.surface.normal_at(c)
        a = _tangent(a, n0)
        b = np.cross(n0, a)
        t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        ring = c[None] + lo * np.cos(t)[:, None] * a[None] + wi * np.sin(t)[:, None] * b[None]
        return tuple(map(tuple, ring[:, :2]))

    def muscles(self, skin: Mesh, controls=None, shaped=None, moulded: MouldedSkin | None = None) -> dict:
        """``{name: Mesh}`` of every muscle in the placed frame: the sheets (:func:`sheet_meshes`, laid under the
        moulded skin: ``moulded``, or :meth:`moulded` of ``shaped`` / of this wing's own junction) and the arm's
        own belly (the sleeve)."""
        out = {}
        ms = moulded if moulded is not None else self.moulded(controls, shaped)
        if ms is not None:
            for name, mesh in sheet_meshes(self.sheets(controls), ms, self.lm.s).items():
                out[name] = mesh.transform(np.diag([-1.0, 1.0, 1.0, 1.0])) if self.letter == "r" else mesh
        out["sleeve"] = self.sleeve_meshes(skin, controls)["muscle"]
        return out

    def bones(self, skin: Mesh, controls=None, shaped=None, moulded: MouldedSkin | None = None) -> dict:
        """``{name: Mesh}`` of the bones in the placed frame: the shoulder blade with its spine and acromion
        (``scapula``), the clavicle that braces it against the sternum, the humerus with its head (the wing's joint
        sits on the blade in version 3 of the placement, under the root of the spine)."""
        left = {}
        # the bones keep their depth under the PLAIN skin (the relief over them is muscle): they are laid under
        # the body as it is without any junction (the fillet climbing the arm would cut the blade round the root)
        ms = moulded if moulded is not None else (self.moulded(controls, self.body[1]) if self.body else None)
        if ms is not None:
            blade = scapula_blade(self.lm, ms)
            spine, acro = scapula_spine(self.lm, ms, self.surface)
            parts = [m for m in (blade, spine, self.glenoid() if self.tube else None) if m is not None]
            if parts:
                left["scapula"] = Mesh.merge(parts)
            left["clavicle"] = clavicle(self.lm, self.surface, acro)
        out = {}
        for k, mesh in left.items():
            if mesh is None:
                continue
            out[k] = mesh.transform(np.diag([-1.0, 1.0, 1.0, 1.0])) if self.letter == "r" else mesh
        out["humerus"] = self.sleeve_meshes(skin, controls)["bone"]
        return out

    def layers(self, skin: Mesh, controls=None, shaped=None) -> dict:
        """``{"bone", "muscle", "skin"}`` (merged meshes, the placed frame): see the module docstring. ``shaped``:
        the body as a view meshes it (every appendage's junction), else this wing's own junction moulds it."""
        return {"bone": Mesh.merge(list(self.bones(skin, controls, shaped).values())),
                "muscle": Mesh.merge(list(self.muscles(skin, controls, shaped).values())),
                "skin": skin}
