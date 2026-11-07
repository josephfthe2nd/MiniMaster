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
        self.vn = J.vertex_normals(self.v, self.q)
        self.memo: dict = {}          # what a model derives from THIS surface alone (the pair shares it)
        corners = self.v[self.q]
        self._centroid = corners.mean(axis=1)
        self._radius = np.linalg.norm(corners - self._centroid[:, None, :], axis=2).max(axis=1)

    def hit(self, origin, direction, near: float = 4.0):
        """The first hit of the ray ``origin + t * direction`` (0 < t <= ``near``) with the skin:
        ``(t, point, normal)`` (the normal interpolated from the vertex normals) or ``None``."""
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
                    best = (float(t[i]), (o + d * t[i], _unit(n)))
        if best[1] is None:
            return None
        return best[0], best[1][0], best[1][1]

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
    ("scapular", "infraspinatus / rhomboid plateau over the shoulder blade",
     0.52, 0.12, (-0.15, 0.99), 0.85, 0.40, 0.16, 0.30),
    ("latissimus", "latissimus / teres fan sweeping to the flank",
     0.66, -0.55, (0.17, 0.98), 0.80, 0.32, 0.13, 0.25),
    ("trapezius", "trapezius slope from the neck to the shoulder",
     0.26, 0.60, (0.866, -0.5), 0.62, 0.34, 0.11, 0.20),
)
# the shoulder boss: the muscle that lies over the buried humerus and the humeral head (the
# posterior deltoid / infraspinatus belly, the "scapular spine" ridge). It runs from the skin over
# the joint to where the arm crosses the skin, so it follows the arm. (name, label, reach beyond the
# half distance joint -> crossing in s, wide in s, height in s, pad in s)
CAP = ("deltoid", "shoulder boss over the humeral head and the buried humerus", 0.6, 0.85, 0.34, 0.35)
BOSS_STEEP, BOSS_SHALLOW = 75.0, 40.0      # degrees between the arm and the plain skin: no boss / the whole boss
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


def cap_mass(lm: Landmarks, surf: BodySurface, shoulder, elbow):
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
    # the boss is what tilts the skin so that the arm leaves it steeply: an arm that already
    # leaves the plain skin at BOSS_STEEP degrees or more (the posed arm, say, sticking out of the
    # flank) needs none, one that leaves it at BOSS_SHALLOW or less (the spread wing: 45 degrees) gets all
    exit_deg = math.degrees(math.asin(min(abs(float(np.dot(d, n))), 1.0)))
    need = float(np.clip((BOSS_STEEP - exit_deg) / (BOSS_STEEP - BOSS_SHALLOW), 0.0, 1.0))
    if need * h * s < MIN_HEIGHT:
        return None
    h = h * need
    run = X - A0
    half = 0.5 * float(np.linalg.norm(run))
    mid = A0 + 0.5 * run
    c_hit = surf.hit(mid + n * 3.0, -n, near=6.0)
    c = c_hit[1] if c_hit is not None else mid
    nc = c_hit[2] if c_hit is not None else n
    along = _tangent(run, nc)
    return Mass(name, label, tuple(c), tuple(along), half + reach * s, wide * s, h * s, pad * s)


# --------------------------------------------------------------------------
# meshes of the layers

BONE_SHARE = 0.30         # the humerus' radius as a share of the sleeve's (a human humerus is 0.2-0.3 of the upper arm)
HEAD_SHARE = 0.75         # the humeral head's radius as a share of the muscle sleeve's at the joint
PAD_RINGS, PAD_SIDES = 8, 28
SCAPULA_DEPTH, SCAPULA_THICK = 0.60, 0.10     # under the skin, in s: the plate lies deep to the muscle pads
SCAPULA_OUTLINE = (       # (along, across) in shares of (length, width): inferior angle, superior angle, glenoid angle
    (-0.50, -0.06), (0.50, -0.36), (0.12, 0.50))


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
        f.append([0, tb(0, k + 1), tb(0, k)])                       # the top fan
        f.append([bp, bb(0, k), bb(0, k + 1)])                      # the bottom fan
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


def pad_mesh(mass: "Mass", surface: BodySurface, n_ring: int = PAD_RINGS, sides: int = PAD_SIDES) -> Mesh:
    """The muscle mesh of a mass (left frame): a lens over the skin of the plain body whose TOP is
    the skin relief of the mass (:meth:`Mass.mound`) minus the skin's thickness and whose bottom
    reaches ``pad`` under the skin at the crest and meets the top at the edge of the footprint."""
    c0 = np.asarray(mass.center)
    n0 = surface.normal_at(c0)
    u = _tangent(mass.axis, n0)
    v = np.cross(n0, u)
    th = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    rho = np.arange(1, n_ring + 1) / n_ring
    flat = (c0[None, None, :] + rho[:, None, None] * (
        mass.half_long * np.cos(th)[None, :, None] * u[None, None, :]
        + mass.half_wide * np.sin(th)[None, :, None] * v[None, None, :]))
    skin, nrm = surface.project(flat.reshape(-1, 3), -n0)
    mound = mass.mound()
    relief = J._mound_displacement(mound, skin, nrm).reshape(n_ring, sides, 3)   # the skin's own relief
    w = mass.weight(skin).reshape(n_ring, sides)
    relief[-1] = 0.0                                              # the edge of the footprint
    w[-1] = 0.0
    P, N = skin.reshape(n_ring, sides, 3), nrm.reshape(n_ring, sides, 3)
    top = P + relief - N * SKIN_T
    bottom = P - N * (SKIN_T + mass.pad * np.sqrt(w))[..., None]
    pole_p, pole_n = surface.project(c0[None], -n0)
    pole_r = J._mound_displacement(mound, pole_p, pole_n)[0]
    tp = pole_p[0] + pole_r - pole_n[0] * SKIN_T
    bp = pole_p[0] - pole_n[0] * (SKIN_T + mass.pad)
    return _lens(tp, top, bp, bottom, wall=False)


def _plate_outline(n_per_edge: int = 6) -> np.ndarray:
    """The outline of a scapula in (along, across) shares: a rounded triangle."""
    pts = np.asarray(SCAPULA_OUTLINE, dtype=np.float64)
    for _ in range(2):                                            # Chaikin corner cutting, twice
        nxt = []
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            nxt += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        pts = np.asarray(nxt)
    return pts


def scapula_plate(lm: Landmarks, surface: BodySurface, masses) -> Mesh | None:
    """The wing's scapula (left frame): a thin curved plate lying on the back under the muscle
    pads, about 1.7 s long and 0.9 s wide, long axis along the scapular plateau, its glenoid
    angle toward the arm. ``None`` if the plateau is not on this body."""
    plateau = next((m for m in masses if m.name == "scapular"), None)
    if plateau is None:
        return None
    c0 = np.asarray(plateau.center)
    n0 = surface.normal_at(c0)
    u = _tangent(plateau.axis, n0)
    v = np.cross(n0, u)
    L, Wd = 1.7 * lm.s, 0.9 * lm.s
    outline = _plate_outline()
    pts = (c0[None, None, :] + np.asarray([0.33, 0.66, 1.0])[:, None, None] * (
        (outline[:, 0] * L)[None, :, None] * u[None, None, :]
        + (outline[:, 1] * Wd)[None, :, None] * v[None, None, :]))
    skin, nrm = surface.project(pts.reshape(-1, 3), -n0)
    P = skin.reshape(3, -1, 3)
    d0, d1 = SCAPULA_DEPTH * lm.s, (SCAPULA_DEPTH + SCAPULA_THICK) * lm.s
    # the plate is a flat bone that keeps its size: it is the projected outline moved straight in along the
    # normal at its centre (offsetting along each point's own normal would shrink it on the curved back)
    top, bottom = P - n0 * d0, P - n0 * d1
    pp, _ = surface.project(c0[None], -n0)
    pole_t, pole_b = pp[0] - n0 * d0, pp[0] - n0 * d1
    return _lens(pole_t, top, pole_b, bottom, wall=True)


def furcula(lm: Landmarks, surface: BodySurface, shoulder) -> Mesh | None:
    """The strut from the humeral head to the sternum (left frame): a bone of the girdle that lies
    inside the chest, from the joint forward and toward the midline, ending just under the front
    of the sternum."""
    S = np.asarray(shoulder, dtype=np.float64)
    target_xy = (0.14 * lm.shx, S[1] + 0.18 * lm.lt)
    front = surface.hit((target_xy[0], target_xy[1], 6.0), (0.0, 0.0, -1.0), near=9.0)
    if front is None:
        return None
    end = front[1] - front[2] * 0.45 * lm.s
    mid = 0.5 * (S + end) + np.array([0.0, 0.0, -0.1 * lm.s])
    t = np.linspace(0.0, 1.0, 12)[:, None]
    path = (1 - t) ** 2 * S[None] + 2 * (1 - t) * t * mid[None] + t ** 2 * end[None]
    r = (0.12 - 0.04 * t[:, 0]) * lm.s
    return _path_tube(path, r)


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
FILLET_PULL = 1.0                 # the share of the in-plane redistribution of the fillet (junction.py)
TUBE_STATIONS = (0.0, 0.25, 0.5, 0.75, 1.0)       # where along the humerus the junction tube is sampled
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


class WingRoot:
    """The shoulder anatomy of one placed wing (``Placed.model``): see the module docstring.

    Built once per placed wing from the body's landmarks and surface, the placed rig (for the
    posed joints) and the rest arm; everything pose-dependent (the deltoid cap, the junction
    tube, the layers) is derived on demand from the controls."""

    def __init__(self, letter: str, rig, matrix, lm: Landmarks, surface: BodySurface,
                 profile: ArmProfile, fillet: float, style: str = "membrane",
                 fillet_height: float = FILLET_HEIGHT):
        self.letter = letter
        self.rig = rig                      # a wingrig.PlacedWing
        self.matrix = np.asarray(matrix, dtype=np.float64)
        self.lm = lm
        self.surface = surface
        self.profile = profile
        self.fillet = float(fillet)
        self.fillet_height = float(fillet_height)
        self.style = style
        self._left_masses = None
        self._left_keep = None

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
        cap = cap_mass(self.lm, self.surface, S, E)
        if cap is not None:
            out.append(cap)
        return out

    # -- the junction ------------------------------------------------------------------

    def spec(self, controls=None, name: str = "wing"):
        """The :class:`~.junction.JunctionSpec` of this wing's root under ``controls``."""
        S, E = self.joints(controls)
        ts = np.asarray(TUBE_STATIONS)
        pts, _, r_in = self.profile.at(ts)
        # the rest arm's rings, carried by the pose: the tube is the straight shoulder -> elbow
        # line with the rest radii (the sleeve is rigid over the humerus)
        line = S[None, :] + (E - S)[None, :] * ts[:, None]
        ref = float(r_in[-1])
        spec = J.JunctionSpec(name, line, r_in * J.HUG, width=self.fillet, base_radius=ref,
                              height=self.fillet_height, pull=FILLET_PULL,
                              mounds=tuple(m.mound() for m in self.masses(controls)),
                              keep=self._keep(), **_OWNER)
        return spec.mirrored() if self.letter == "r" else spec

    # -- the layers ---------------------------------------------------------------------

    def _sleeve_rings(self, skin: Mesh) -> np.ndarray:
        """The rings of the humerus of the placed (posed) skin mesh ``skin``: ``(n, sides, 3)``, in
        the placed frame (the wing's own arm piece is a ring-major tube; see :func:`arm_profile`)."""
        a, b = self.rig.rig.ranges["arm"]
        sides = self.profile.sides
        rings = skin.vertices[a:b - 2].reshape(-1, sides, 3)
        return rings[:len(self.profile.t)]

    def sleeve_meshes(self, skin: Mesh) -> dict:
        """``{"muscle": Mesh, "bone": Mesh}`` of the arm in the placed frame: the skin's own rings
        scaled about their centres to the muscle (the skin minus its thickness) and to the bone
        (a share of the sleeve), so the three nest exactly and pose with the skin."""
        rings = self._sleeve_rings(skin)
        c = rings.mean(axis=1, keepdims=True)
        r = np.linalg.norm(rings - c, axis=2).mean(axis=1)
        r_mus = np.maximum(r - SKIN_T, 0.55 * r)
        r_bone = BONE_SHARE * r
        out = {}
        for key, rr, lo, hi in (("muscle", r_mus, 0, len(rings)), ("bone", r_bone, 1, len(rings) - 1)):
            # the bone stops a ring short of each end of the sleeve, inside its caps
            ring = c[lo:hi] + (rings[lo:hi] - c[lo:hi]) * (rr[lo:hi] / r[lo:hi])[:, None, None]
            out[key] = _tube_from_rings(ring)
        out["bone"] = Mesh.merge([out["bone"], self._head(rings, r_mus)])
        return out

    def _head(self, rings, r_mus) -> Mesh:
        """The humeral head: a ball inside the buried end of the sleeve (its cap)."""
        c0 = rings[0].mean(axis=0)
        axis = _unit(rings[1].mean(axis=0) - c0)
        R = HEAD_SHARE * float(r_mus[0])
        centre = c0 + axis * 1.2 * R
        if self.letter == "r":          # built on the left and mirrored: the pair is an exact mirror pair
            return _ball(centre * MIRROR, R).transform(np.diag([-1.0, 1.0, 1.0, 1.0]))
        return _ball(centre, R)

    def muscles(self, skin: Mesh, controls=None) -> dict:
        """``{name: Mesh}`` of every muscle mass in the placed frame: the pads of the masses (their
        top is the skin relief minus the skin's thickness) and the arm's own belly."""
        out = {}
        for m in self.masses(controls):
            mesh = pad_mesh(m, self.surface)
            if self.letter == "r":
                mesh = mesh.transform(np.diag([-1.0, 1.0, 1.0, 1.0]))
            out[m.name] = mesh
        out["sleeve"] = self.sleeve_meshes(skin)["muscle"]
        return out

    def bones(self, skin: Mesh, controls=None) -> dict:
        """``{name: Mesh}`` of the bones in the placed frame: the wing's scapula plate on the back, the
        strut to the sternum, the humerus with its head."""
        left = {}
        S, _ = self.joints(controls)
        left["scapula"] = scapula_plate(self.lm, self.surface, self.permanent())
        left["furcula"] = furcula(self.lm, self.surface, S)
        out = {}
        for k, mesh in left.items():
            if mesh is None:
                continue
            out[k] = mesh.transform(np.diag([-1.0, 1.0, 1.0, 1.0])) if self.letter == "r" else mesh
        out["humerus"] = self.sleeve_meshes(skin)["bone"]
        return out

    def layers(self, skin: Mesh, controls=None) -> dict:
        """``{"bone", "muscle", "skin"}`` (merged meshes, the placed frame): see the module docstring."""
        return {"bone": Mesh.merge(list(self.bones(skin, controls).values())),
                "muscle": Mesh.merge(list(self.muscles(skin, controls).values())),
                "skin": skin}
