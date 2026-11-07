"""The girdle of an extra limb: where a second shoulder (or hip) sits on the body, and the
flesh that carries the limb.

THE PROBLEM. A harvested limb used to be the character's own arm cut at a ring and sunk into
the torso: the shoulder joint ended up 2 units inside the chest (the sternum for the "lower
arms", the base of the neck for the "upper arms", the upper chest for the "extra legs"), what
showed was a cylinder cut, and the skin did nothing about it. Real extra limbs (the
polymelia cases of the reference library, the Hindu multi-armed bronzes) root on a GIRDLE: a
bony outrigger carries the joint outside the ribs, a thick muscle sleeve (the deltoid dome,
the pectoral and latissimus fans) spreads over the trunk, and the skin is one continuous
envelope over all of it.

WHAT THIS MODULE IS. Pure functions in the MakeHuman frame (Y up, +X the character's left):

* :func:`measure` finds the girdle on THIS body from the joint markers and one horizontal
  slice of the torso, so it follows every morph (the tall, the child and the heavy body get
  their girdle in the same relative place): the pivot ``P`` (the second glenohumeral centre),
  the skin point and outward normal under it, and the limb's own scale ``R`` (the equivalent
  radius of its upper-arm section) and ``Lu`` (its length). Everything else is in ``R`` and
  ``Lu`` so it scales with the body and with the limb.
* :func:`cut_limb` takes the limb out of the body cage with an exact PLANE cut across the
  upper bone (the old proximity cut ended in a ragged rim of cage quads), and
  :func:`dome_root` closes that cut with a flared proximal limb that ends in the shoulder
  dome: no cut ring, no flat cap.
* :class:`Anatomy` holds the three layers derived from the same parameters (bone, muscle,
  skin) and the skin junction (the skirt and the folds; no groove: below the cage's resolution).

See docs/anatomy_limb.md for the references and the measured numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.mesh import Mesh

_EPS = 1e-12

# -- the recipe (all lengths in R, the equivalent radius of the limb's upper-arm section at
#    0.25 of its length, or in Lu, the length of the upper bone) ------------------------------

ARM_DROP = 0.85            # lower pair: the pivot is 0.85 Lu below the primary joint
LEG_RISE = 1.5             # hip pair: 1.5 R_leg above the hip joint (the iliac crest)
PIVOT_OUT = 0.4            # the pivot stands this many R outside the skin
DOME_RADIUS = 1.1          # the dome (glenohumeral head under the deltoid), in R
CUT_T = 0.30               # the plane the limb is cut at, in Lu from the joint
R_AT = 0.4                 # R is measured on the section at this t (isolated on every body) ...
R_RATIO = {"arm": 0.906, "leg": 0.953}      # ... divided by this: r(0.4 Lu) / r(0.25 Lu) on the neutral body


# ==============================================================================================
# slicing a closed triangle mesh


def quad_triangles(quads) -> np.ndarray:
    q = np.asarray(quads)
    return np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])


def slice_loops(verts, tris, point, normal) -> list:
    """The closed polylines where the plane (``point``, ``normal``) cuts a closed triangle
    mesh: a list of ``(k, 3)`` arrays (the first point is not repeated). Loops are linked by
    the mesh edges they cross, so the result is exact (no float hashing)."""
    v = np.asarray(verts, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    n = n / max(float(np.linalg.norm(n)), _EPS)
    s = (v - np.asarray(point, dtype=np.float64)) @ n
    pos = s > 0.0
    t = np.asarray(tris)
    ps = pos[t]
    mixed = ps.any(axis=1) & ~ps.all(axis=1)
    tt = t[mixed]
    if not len(tt):
        return []
    segs = []
    pts = {}
    for tri in tt:
        keys = []
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            if pos[a] != pos[b]:
                key = (int(a), int(b)) if a < b else (int(b), int(a))
                if key not in pts:
                    sa, sb = s[key[0]], s[key[1]]
                    f = sa / (sa - sb)
                    pts[key] = v[key[0]] + f * (v[key[1]] - v[key[0]])
                keys.append(key)
        if len(keys) == 2:
            segs.append(tuple(keys))
    nxt: dict = {}
    for a, b in segs:
        nxt.setdefault(a, []).append(b)
        nxt.setdefault(b, []).append(a)
    used = set()
    loops = []
    for a, b in segs:
        if (a, b) in used or (b, a) in used:
            continue
        loop = [a]
        prev, cur = a, b
        used.add((a, b))
        while cur != a:
            loop.append(cur)
            cands = [c for c in nxt[cur] if c != prev and (cur, c) not in used and (c, cur) not in used]
            if not cands:
                break
            prev, cur = cur, cands[0]
            used.add((prev, cur))
        loops.append(np.array([pts[k] for k in loop]))
    return loops


def polygon_area_centroid(xy):
    """Signed area and centroid of a closed polygon given as ``(k, 2)``."""
    x, y = xy[:, 0], xy[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    a = 0.5 * cr.sum()
    if abs(a) < _EPS:
        return 0.0, xy.mean(axis=0)
    cx = ((x + x1) * cr).sum() / (6.0 * a)
    cy = ((y + y1) * cr).sum() / (6.0 * a)
    return float(a), np.array([cx, cy])


def _point_in_polygon(p, xy) -> bool:
    x, y = p
    px, py = xy[:, 0], xy[:, 1]
    qx, qy = np.roll(px, -1), np.roll(py, -1)
    cond = (py > y) != (qy > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = (qx - px) * (y - py) / (qy - py) + px
    return bool((cond & (x < xi)).sum() % 2)


def torso_loop(verts, tris, y: float) -> np.ndarray:
    """The loop of the horizontal slice at height ``y`` that contains the midline (the trunk,
    not an arm or a leg): ``(k, 3)``."""
    loops = slice_loops(verts, tris, (0.0, y, 0.0), (0.0, 1.0, 0.0))
    best, best_a = None, 0.0
    for lp in loops:
        xy = lp[:, [0, 2]]
        a, c = polygon_area_centroid(xy)
        if _point_in_polygon(c, xy) and abs(a) > best_a and xy[:, 0].min() < 0.0 < xy[:, 0].max():
            best, best_a = lp, abs(a)
    if best is None:
        raise ValueError(f"no torso section at y = {y:.3f}")
    return best


def ray_to_loop(loop, origin_xz, direction_xz):
    """The farthest crossing of the ray ``origin + t d`` (in the xz plane) with the closed
    polygon ``loop`` (``(k, 3)``): ``(point3, outward normal3 (horizontal, unit))``. The normal
    is the polygon's own at the crossing, so it follows the skin and not the ray."""
    xy = loop[:, [0, 2]]
    o = np.asarray(origin_xz, dtype=np.float64)
    d = np.asarray(direction_xz, dtype=np.float64)
    d = d / max(float(np.linalg.norm(d)), _EPS)
    a = xy
    b = np.roll(xy, -1, axis=0)
    e = b - a
    den = d[0] * (-e[:, 1]) - d[1] * (-e[:, 0])
    w = a - o
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w[:, 0] * (-e[:, 1]) - w[:, 1] * (-e[:, 0])) / den
        u = (d[0] * w[:, 1] - d[1] * w[:, 0]) / den
    ok = np.isfinite(t) & np.isfinite(u) & (t > 0) & (u >= 0) & (u <= 1)
    if not ok.any():
        raise ValueError("the ray does not leave the torso section")
    i = int(np.nonzero(ok)[0][np.argmax(t[ok])])
    p2 = a[i] + u[i] * e[i]
    tangent = e[i] / max(float(np.linalg.norm(e[i])), _EPS)
    nrm = np.array([tangent[1], -tangent[0]])
    if nrm @ d < 0:
        nrm = -nrm
    # a near-tangent crossing (a sharp corner of the section): use the ray
    if nrm @ d < 0.6:
        nrm = d
    point = np.array([p2[0], float(loop[0, 1]), p2[1]])
    return point, np.array([nrm[0], 0.0, nrm[1]])


def section_radius(verts, tris, point, axis) -> float:
    """Equivalent radius (sqrt(area / pi)) of the section of the mesh by the plane through
    ``point`` perpendicular to ``axis``: the loop whose centroid is nearest ``point``."""
    ax = np.asarray(axis, dtype=np.float64)
    ax = ax / max(float(np.linalg.norm(ax)), _EPS)
    helper = np.array([0.0, 1.0, 0.0]) if abs(ax[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(ax, helper)
    e1 /= max(float(np.linalg.norm(e1)), _EPS)
    e2 = np.cross(ax, e1)
    best, best_d = None, np.inf
    for lp in slice_loops(verts, tris, point, ax):
        if len(lp) < 8:
            continue
        rel = lp - np.asarray(point)
        xy = np.stack([rel @ e1, rel @ e2], axis=1)
        a, c = polygon_area_centroid(xy)
        d = float(np.linalg.norm(c))
        if d < best_d:
            best, best_d = abs(a), d
    if best is None:
        raise ValueError("the limb has no section there")
    return math.sqrt(best / math.pi)


# ==============================================================================================
# the sites: where a second girdle sits, from the body's own landmarks


@dataclass(frozen=True)
class Site:
    """A recipe for a place on the trunk: the limb that is copied, the height of the pivot
    (``drop`` Lu below the primary joint for an arm pair, ``rise`` R above the hip joint for a
    leg pair), the azimuth ``phi`` around the section's centroid (0 = the lateral midline,
    negative = toward the back), the scale of the second limb and its rest pose (degrees:
    ``raise_``, ``swing``, ``elbow`` of the rig)."""

    limb: str
    drop: float | None = None
    rise: float | None = None
    phi: float = 0.0
    scale: float = 1.0
    pose: tuple = (0.0, 0.0, 0.0)
    limits: tuple = ()           # overrides of limbrig._LIMITS: the range of motion this girdle allows
    out: float = PIVOT_OUT       # how far the pivot stands outside the skin, in R
    dome: float = DOME_RADIUS    # the radius of the dome, in R
    lean: float = 0.0            # how far the dome's far end leans into the trunk (dome radii): a leg's axis runs
                                 # along the wall, so its dome has to turn inward to close inside the pelvis
    fan: tuple | None = None     # a pair that shares the primary shoulder: the pivot's offset from the
                                 # joint marker in R (medial, up, posterior) instead of a flank rule


SITES = {
    "lower_arms": Site("arm", drop=ARM_DROP, phi=0.0, scale=0.9, pose=(-7.0, 8.0, 20.0),
                       limits=(("abduct", 58.0), ("adduct", -28.0), ("swing", (-25.0, 55.0))), lean=0.4),
    # the upper pair fans out of the primary shoulder (the multi-armed bronzes: one broad shoulder
    # dome per side carries all its arms, the raised pair leaving its posterior face): the pivot is
    # on the scapular spine / trapezius ridge, 1.35 R behind and above the primary joint, a third of
    # a radius under the back skin; GUESS (iconography, not anatomy: docs/anatomy_limb.md)
    "upper_arms": Site("arm", fan=(-0.55, 0.5, -1.05), scale=0.85, pose=(68.0, -30.0, -10.0),
                       limits=(("abduct", 160.0), ("adduct", -60.0), ("swing", (-60.0, 45.0))),
                       out=0.0, dome=1.07),
    "extra_legs": Site("leg", rise=LEG_RISE, phi=-30.0, scale=0.7, pose=(49.0, 5.0, 25.0),
                       limits=(("abduct", 75.0), ("adduct", -38.0), ("swing", (-20.0, 50.0))),
                       out=0.1, dome=0.97, lean=0.7),
}

CHAINS = {"arm": ("l-shoulder", "l-elbow", "l-hand"), "leg": ("l-upper-leg", "l-knee", "l-ankle")}
REFERENCE_LENGTH = {"arm": 2.4423469356466336, "leg": 4.0260352512777695}


@dataclass(frozen=True)
class Girdle:
    """A girdle measured on one body (LEFT-hand frame; the right one is its mirror image)."""

    limb: str
    joints: tuple              # (joint, elbow, hand) of the donor limb
    Lu: float                  # the upper bone's length
    R: float                   # equivalent radius of the upper section at 0.25 Lu
    y: float                   # height of the pivot
    skin: np.ndarray           # the point of the skin under the pivot
    normal: np.ndarray         # the outward normal there (horizontal)
    pivot: np.ndarray          # the second joint centre: skin + PIVOT_OUT * R * normal (scaled)
    centroid: np.ndarray       # the centroid of the trunk section at y
    scale: float = 1.0
    section: tuple = ()        # (half width, half depth) of the trunk section at y
    loop: np.ndarray | None = None      # the trunk section at y ((k, 3) polyline, left frame)
    fan: bool = False                   # the girdle is the primary shoulder's (a fan pair)


def _reflect(p):
    return np.asarray(p, dtype=np.float64) * np.array([-1.0, 1.0, 1.0])


def limb_scale(base, verts, limb: str, side: str = "l") -> float:
    """The length of this body's upper bone over the neutral body's."""
    from . import appendages as A
    chain = [A.mirror_joint_name(n) if side == "r" else n for n in CHAINS[limb]]
    s, e = (A.joint_position(base, n, verts) for n in chain[:2])
    return float(np.linalg.norm(e - s)) / REFERENCE_LENGTH[limb]


def measure(base, verts, site: Site, side: str = "l", scale: float | None = None,
            out_extra: float = 0.0, point=None, phi: float | None = None) -> Girdle:
    """Find the girdle of ``site`` on the body ``verts`` (the UNDEFORMED morphed vertices).

    Left frame always: for ``side="r"`` the right markers are used and the cage is reflected, so
    a right girdle is computed from the right body, not copied from the left one. ``scale`` is
    the second limb's scale (default: the site's), ``out_extra`` moves the pivot further out of
    the skin (in R), ``phi`` overrides the site's azimuth. ``point`` (left frame) is a place the
    person chose instead of the site's rule: the girdle is built at its height, toward it."""
    from . import appendages as A
    chain = [A.mirror_joint_name(n) if side == "r" else n for n in CHAINS[site.limb]]
    pts = [A.joint_position(base, n, verts) for n in chain]
    body_v, body_q = base.body_cage(verts)
    if side == "r":
        pts = [_reflect(p) for p in pts]
        body_v = _reflect(body_v)
    tris = quad_triangles(body_q)
    j, e, h = pts
    Lu = float(np.linalg.norm(e - j))
    axis = (e - j) / Lu
    R = section_radius(body_v, tris, j + axis * (R_AT * Lu), axis) / R_RATIO[site.limb]
    s = site.scale if scale is None else float(scale)
    if site.fan is not None and point is None:
        pv = j + R * np.asarray(site.fan)
        loop = torso_loop(body_v, tris, float(pv[1]))
        _, c = polygon_area_centroid(loop[:, [0, 2]])
        skin, n = ray_to_loop(loop, (pv[0], pv[2]), (0.0, -1.0))       # the back skin behind the pivot
        pivot = pv + out_extra * R * s * n
        half = (float((loop[:, 0].max() - loop[:, 0].min()) / 2), float((loop[:, 2].max() - loop[:, 2].min()) / 2))
        return Girdle(site.limb, (j, e, h), Lu, R, float(pv[1]), skin, n, pivot, np.array([c[0], pv[1], c[1]]),
                      s, half, loop, True)
    if point is not None:
        y = float(point[1])
    elif site.drop is not None:
        y = float(j[1] - site.drop * Lu)
    else:
        y = float(j[1] + site.rise * R)
    loop = torso_loop(body_v, tris, y)
    _, c = polygon_area_centroid(loop[:, [0, 2]])
    if point is not None:
        direction = (float(point[0] - c[0]), float(point[2] - c[1]))
    else:
        ang = math.radians(site.phi if phi is None else phi)
        direction = (math.cos(ang), math.sin(ang))
    skin, n = ray_to_loop(loop, c, direction)
    pivot = skin + (site.out + out_extra) * R * s * n
    half = (float((loop[:, 0].max() - loop[:, 0].min()) / 2), float((loop[:, 2].max() - loop[:, 2].min()) / 2))
    return Girdle(site.limb, (j, e, h), Lu, R, y, skin, n, pivot, np.array([c[0], y, c[1]]), s, half, loop)


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# ==============================================================================================
# cutting the limb out of the body: the quads beyond a plane across the upper bone


@dataclass
class LimbCut:
    """The limb cut out of a body, in the donor frame: an open triangle mesh whose root ring
    (the boundary of the quads beyond the plane across the upper bone, ordered) is ``ring``, and
    whose far end (the hand) is still open."""

    verts: np.ndarray
    tris: np.ndarray
    ring: np.ndarray            # ordered vertex indices of the root ring
    plane_t: float              # where it is cut, in Lu from the joint


SPREAD = 0.6            # how far the cut ring's vertices are moved toward even spacing around the bone


def _spread_ring(P, axis, amount: float):
    """The planar ring ``P`` (ordered, ``(k, 3)``) with its vertices slid along the ring toward an even
    spacing in angle (by ``amount``): the cage has tiny quads where its edge loops converge (the front
    of the armpit), and the faces there are slivers. The ring's shape is kept (every new vertex is on
    the ring's own polygon), only where the vertices are on it changes."""
    u = np.asarray(axis, dtype=np.float64)
    u = u / max(float(np.linalg.norm(u)), _EPS)
    c = P.mean(axis=0)
    e1, e2 = _perp(u)
    q = P - c
    xy = np.stack([q @ e1, q @ e2], axis=1)
    th = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
    rho = np.linalg.norm(xy, axis=1)
    n = len(P)
    direction = 1.0 if th[-1] > th[0] else -1.0
    th_u = th[0] + direction * 2.0 * math.pi * np.arange(n) / n
    new = (1.0 - amount) * th + amount * th_u
    # the radius at the new angles: periodic linear interpolation of the ring's own (angle, radius)
    order = np.argsort(th)
    t_s, r_s = th[order], rho[order]
    period = 2.0 * math.pi
    t_ext = np.concatenate([t_s - period, t_s, t_s + period])
    r_ext = np.concatenate([r_s, r_s, r_s])
    rho_new = np.interp(new, t_ext, r_ext)
    plane = (q @ u)
    return c + np.outer(plane, u) + rho_new[:, None] * (np.cos(new)[:, None] * e1 + np.sin(new)[:, None] * e2)


RING_DIHEDRAL_MAX = 40.0     # degrees: the crease the seam between the cage's limb and the dome may have at worst


def ring_dihedral(V, T, ring) -> float:
    """The largest angle (degrees) between the two faces on an edge of the root ring: how visible
    the seam between the body's own limb and the dome is (a smooth one is a few degrees)."""
    V = np.asarray(V)
    T = np.asarray(T)
    ring_set = set(int(x) for x in ring)
    fv = _face_vectors(V, T)
    fn = fv / np.maximum(np.linalg.norm(fv, axis=1, keepdims=True), _EPS)
    first: dict = {}
    worst = 0.0
    for fi, t in enumerate(T):
        for k in range(3):
            a, b = int(t[k]), int(t[(k + 1) % 3])
            if a in ring_set and b in ring_set:
                key = (a, b) if a < b else (b, a)
                if key in first:
                    cosv = float(np.clip(fn[first[key]] @ fn[fi], -1.0, 1.0))
                    worst = max(worst, math.degrees(math.acos(cosv)))
                else:
                    first[key] = fi
    return worst


def _face_vectors(v, tris):
    """Twice the area vector of every triangle."""
    p = v[tris]
    return np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])


def cut_limb(base, body_v, body_q, joints, Lu: float, R: float, t0: float = CUT_T,
             reach: float = 1.4, hand_reach: float = 2.4) -> LimbCut:
    """The part of the body cage that is the limb beyond the plane ``t0 Lu`` from the joint,
    perpendicular to the upper bone.

    A quad joins when all four corners are within ``reach R`` of the bone chain (``hand_reach``
    beyond the elbow, for the hand and its fingers) and beyond the plane; the largest connected
    piece is kept (a strip of flank skin that is within reach of the arm is joined to it only
    through the armpit, which is on the near side of the plane). Its root ring is the boundary
    of those quads, slid onto one plane across the bone."""
    from .appendages import _split_pinch_vertices, boundary_loops, largest_component
    j, e, h = (np.asarray(p, dtype=np.float64) for p in joints)
    axis = (e - j) / Lu
    dist = np.full(len(body_v), np.inf)
    for a, b in ((j, e), (e, h)):
        ab = b - a
        t = np.clip(((body_v - a) @ ab) / max(float(ab @ ab), _EPS), 0.0, 1.0)
        dist = np.minimum(dist, np.linalg.norm(body_v - (a + t[:, None] * ab), axis=1))
    along = (body_v - j) @ axis
    lim = np.where(along < 0.9 * Lu, reach * R, hand_reach * R)
    keep = (dist <= lim) & (along >= t0 * Lu)
    sel = keep[body_q].all(axis=1)
    if not sel.any():
        raise ValueError("the limb selects no faces")
    quads = largest_component(body_q[sel])
    # a quad with three corners on the root ring would be an "ear" lying in the plane of the ring
    # once the ring is flattened: shave those off (the ring then runs through the next quads)
    for _ in range(40):
        used = np.unique(quads)
        remap = -np.ones(len(body_v), dtype=np.int64)
        remap[used] = np.arange(len(used))
        local = remap[quads]
        loops0 = boundary_loops(local)
        if not loops0:
            raise ValueError("the limb has no root ring")
        ring0 = loops0[int(np.argmin([float(((body_v[used[lp]] - j) @ axis).mean()) for lp in loops0]))]
        bad = np.isin(local, ring0).sum(axis=1) >= 3
        if not bad.any():
            break
        quads = largest_component(quads[~bad])
    verts = body_v[used]
    local, extra = _split_pinch_vertices(local)
    if extra:
        verts = np.vstack([verts, verts[np.asarray(extra, dtype=np.int64)]])
    loops = boundary_loops(local)
    if not loops:
        raise ValueError("the limb has no root ring")
    mean_along = [float(((verts[lp] - j) @ axis).mean()) for lp in loops]
    ring = np.asarray(loops[int(np.argmin(mean_along))], dtype=np.int64)
    if len(set(ring.tolist())) != len(ring):
        raise ValueError("the root ring of the limb revisits a vertex")
    if len(loops) > 1 and sorted(mean_along)[1] - sorted(mean_along)[0] < 0.3 * Lu:
        raise ValueError(f"the cut at {t0:.2f} Lu is not one clean ring")
    # the boundary of a set of quads is ragged along the bone (up to a quad): slide each ring vertex
    # along the bone onto the plane of the ring's mean. The cage is smooth there (the surface moves by
    # the taper's slope times the slide, about 1% of the radius) and the ring is planar, evenly spread
    # (the cage's own vertices) and free of the near-duplicate points a plane cut makes
    al = (verts[ring] - j) @ axis
    verts = verts.copy()
    tris = quad_triangles(local)
    flat = verts.copy()
    flat[ring] = verts[ring] + np.outer(al.mean() - al, axis)
    # spread the ring's vertices as far as no face of the cut turns over or collapses
    old = _face_vectors(verts, tris)
    for amount in (SPREAD, 0.4, 0.25, 0.1, 0.0):
        trial = flat.copy()
        trial[ring] = _spread_ring(flat[ring], axis, amount)
        new = _face_vectors(trial, tris)
        dots = (old * new).sum(axis=1) / np.maximum(np.linalg.norm(old, axis=1) * np.linalg.norm(new, axis=1), _EPS)
        if (dots > 0.5).all() and (np.linalg.norm(new, axis=1) > 0.1 * np.linalg.norm(old, axis=1)).all():
            break
    verts = trial
    return LimbCut(verts, tris, ring, t0)


def dome_root(cut: LimbCut, joint, axis, R: float, k_flare: int = 5, k_dome: int = 8,
              dome: float = DOME_RADIUS, slope: float | None = None, lean=None):
    """Close the root ring of ``cut`` with a flared proximal limb that ends in a hemisphere
    (the shoulder dome) centred on ``joint``: ``(verts, tris, info)`` of the open limb with its
    root ring continued by rings that swell by about 15% toward the joint's plane and then close
    over it (a quarter circle of radius ``dome R``), ending at a single pole. The ring's
    own shape (its ellipse, longer front to back) is kept, so the dome is an ellipsoid of the
    arm's own section. ``lean`` (a vector, units of the dome radius) shears the closing hemisphere toward
    it (the trunk's inward direction): a limb whose axis runs along the wall (a leg: vertical) would
    otherwise close into the air above its girdle. ``joint`` must lie on the ring's own axis (the line through the ring's
    centre along ``axis``); ``slope`` is the limb's own taper at the ring (relative radius per
    unit of the ring's distance to the joint) so the flare continues it."""
    from .appendages import boundary_loops
    j = np.asarray(joint, dtype=np.float64)
    u = np.asarray(axis, dtype=np.float64)
    u = u / max(float(np.linalg.norm(u)), _EPS)
    verts = [row for row in cut.verts]
    tris = [tuple(t) for t in cut.tris]
    loop = list(cut.ring)
    for lp in boundary_loops(cut.tris):                # the ring in the direction the faces want
        if set(lp) == set(loop):
            loop = list(lp)
            break
    P = cut.verts[loop]
    c0 = P.mean(axis=0)
    q = P - c0
    off = q @ u                                       # the ring is ragged along the bone: flattened below
    q = q - off[:, None] * u
    e1, e2 = _perp(u)
    xy = np.stack([q @ e1, q @ e2], axis=1)
    area, _ = polygon_area_centroid(xy)
    r_ring = math.sqrt(abs(area) / math.pi)
    r_dome = max(dome * R, 1.02 * r_ring)
    m_p = r_dome / r_ring
    d_total = float((c0 - j) @ u)                    # the ring's distance from the joint plane
    # the cage is a polygon: the rings of the dome are the low-pass shape of the ring (a Fourier
    # series of its radius, up to the fourth harmonic) at even angles, blended in as the rings go
    n = len(loop)
    th = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
    direction = 1.0 if th[-1] > th[0] else -1.0        # the loop runs either way around the axis
    rho = np.linalg.norm(xy, axis=1)
    cols = [np.ones(n)]
    for h in range(1, 5):
        cols += [np.cos(h * th), np.sin(h * th)]
    coef, *_ = np.linalg.lstsq(np.stack(cols, axis=1), rho, rcond=None)
    th_u = th[0] + direction * 2.0 * math.pi * np.arange(n) / n

    def smooth_radius(angles):
        out = np.full(len(angles), coef[0])
        for h in range(1, 5):
            out += coef[2 * h - 1] * np.cos(h * angles) + coef[2 * h] * np.sin(h * angles)
        return out

    # the flare: radius ratio m(f), f = 0 at the ring, 1 at the joint's plane; m(1) = m_p and
    # m'(1) = 0 (the dome starts as a sphere's equator), m'(0) = the slope the limb itself has there
    a1 = 2.0 * (m_p - 1.0) if slope is None else min(max(float(slope), 0.0), 3.0 * (m_p - 1.0))
    b1 = 3.0 * (m_p - 1.0) - 2.0 * a1
    c1 = a1 - 2.0 * (m_p - 1.0)
    # the first ring is farther from the cage ring than the cage ring is ragged along the bone, so no
    # triangle of the first band folds back over the cage's own faces; the rest are evenly spaced
    f1 = min(0.5, (float(np.abs(off).max()) + 0.1 * R) / max(d_total, _EPS))
    fs = [f1 + (1.0 - f1) * k / (k_flare - 1) for k in range(k_flare)]
    ring_rows = []
    for k, f in enumerate(fs):
        m = 1.0 + a1 * f + b1 * f * f + c1 * f ** 3
        ring_rows.append((c0 + (j - c0) * f, m, float(_smooth(f)), 1.0))
    shear = np.zeros(3)
    if lean is not None:
        lv = np.asarray(lean, dtype=np.float64)
        shear = lv - (lv @ u) * u
    for k in range(1, k_dome + 1):
        phi = 0.5 * math.pi * k / (k_dome + 1)
        ring_rows.append((j - u * (r_dome * math.sin(phi)) + shear * r_dome * math.sin(phi) ** 2,
                          m_p * math.cos(phi), 1.0, 1.0))
    prev = loop
    equator = None
    for ri, (center, m, w, g) in enumerate(ring_rows):
        base_i = len(verts)
        if ri == k_flare - 1:
            equator = np.arange(base_i, base_i + n)          # the ring at the joint's plane: the dome starts beyond it
        ang = th * (1.0 - w) + th_u * w
        rr = smooth_radius(ang) + (rho - smooth_radius(th)) * (1.0 - w)       # the ring's own ripple dies out
        for i in range(n):
            verts.append(center + m * rr[i] * (math.cos(ang[i]) * e1 + math.sin(ang[i]) * e2)
                         + u * (off[i] * (1.0 - g)))
        cur = list(range(base_i, base_i + n))
        for i in range(n):
            a, b = prev[i], prev[(i + 1) % n]
            a2, b2 = cur[i], cur[(i + 1) % n]
            tris.append((b, a, a2))
            tris.append((b, a2, b2))
        prev = cur
    pole = len(verts)
    verts.append(j - u * r_dome + shear * r_dome)
    for i in range(n):
        a, b = prev[i], prev[(i + 1) % n]
        tris.append((pole, b, a))
    info = dict(pole=pole, ring=np.asarray(loop), equator=equator, r_ring=r_ring, r_dome=r_dome, d_total=d_total,
                new=np.arange(len(cut.verts), len(verts)))
    return np.array(verts), np.array(tris, dtype=np.int64), info


def _perp(u):
    helper = np.array([0.0, 1.0, 0.0]) if abs(u[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(u, helper)
    e1 /= max(float(np.linalg.norm(e1)), _EPS)
    return e1, np.cross(u, e1)


# ==============================================================================================
# the radius law of the limb and the junction (the flesh around the root)

# the radius of the upper limb along its length, relative to R (r at 0.25 Lu): measured on the
# neutral body's section (the research round: 1.07 at the joint -- the dome --, 1.0 at 0.2 Lu,
# then 0.93 ... 0.76, 0.78 at the elbow), t in Lu from the joint
RADIUS_T = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
RADIUS_F = (1.07, 1.04, 1.0, 0.93, 0.87, 0.84, 0.82, 0.80, 0.78, 0.76, 0.78)
FOREARM_T = (0.25, 0.5)
FOREARM_F = (0.66, 0.56)       # r / R of the forearm 0.25 / 0.5 of the way from the elbow (the neutral body's)


def radius_law(t):
    """r / R of the upper limb at ``t`` Lu from the joint."""
    return np.interp(t, RADIUS_T, RADIUS_F)


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > _EPS else v


# ==============================================================================================
# the deep mass (hidden): what holds the dome to the trunk


def ellipsoid(center, axes, radii, rings: int = 10, segments: int = 16) -> Mesh:
    """A closed, outward ellipsoid with ``radii`` along the (unit) ``axes`` (3 rows)."""
    c = np.asarray(center, dtype=np.float64)
    ax = np.asarray(axes, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    verts = [c + ax[2] * r[2]]
    for i in range(1, rings):
        th = math.pi * i / rings
        for k in range(segments):
            ph = 2.0 * math.pi * k / segments
            verts.append(c + r[0] * math.sin(th) * math.cos(ph) * ax[0]
                         + r[1] * math.sin(th) * math.sin(ph) * ax[1] + r[2] * math.cos(th) * ax[2])
    verts.append(c - ax[2] * r[2])
    faces = []
    for k in range(segments):
        faces.append((0, 1 + (k + 1) % segments, 1 + k))
    for i in range(rings - 2):
        a0, a1 = 1 + i * segments, 1 + (i + 1) * segments
        for k in range(segments):
            k1 = (k + 1) % segments
            faces.append((a0 + k, a0 + k1, a1 + k1))
            faces.append((a0 + k, a1 + k1, a1 + k))
    last = 1 + (rings - 2) * segments
    bot = len(verts) - 1
    for k in range(segments):
        faces.append((bot, last + k, last + (k + 1) % segments))
    mesh = Mesh(np.array(verts), np.array(faces, dtype=np.int64))
    if mesh.volume() < 0:                      # a left-handed set of axes turns it inside out
        mesh = Mesh(mesh.vertices, mesh.faces[:, ::-1])
    return mesh


def flank_frame(normal):
    """``(n, up, ant)``: the outward normal, up and the anterior tangent at a flank point of
    the left side (anterior is +z for a lateral normal)."""
    n = _unit(normal)
    up = np.array([0.0, 1.0, 0.0])
    ant = _unit(np.cross(n, up))
    if ant[2] < 0:
        ant = -ant
    up = _unit(np.cross(ant, n))
    if up[1] < 0:
        up = -up
    return n, up, ant


PLUG_FRACTION = 0.5          # the plug's centre is this far from the skin toward the trunk's centre
PLUG_AXES = (1.0, 1.2, 1.0)  # semi-axes in R of the limb: along the normal, up, along the skin


def plug_mesh(girdle: Girdle) -> Mesh:
    """The deep root mass of the girdle (serratus / latissimus / scapular analogue), hidden in the
    trunk under the dome: an ellipsoid halfway from the skin to the centre of the trunk section,
    never closer than 0.8 of its own radius to the skin. Part of the printed shell so that the
    weld to the body is a volume, not a tangent contact."""
    n, up, ant = flank_frame(girdle.normal)
    Rs = girdle.R * girdle.scale
    c = girdle.skin + (girdle.centroid - girdle.skin) * PLUG_FRACTION
    c[1] = girdle.y
    xz = girdle.loop[:, [0, 2]]
    dmin = float(np.min(np.linalg.norm(xz - c[[0, 2]], axis=1)))
    a, b, t = PLUG_AXES
    k = min(1.0, 0.8 * dmin / (max(a, t) * Rs))
    return ellipsoid(c, np.array([n, up, ant]), (a * Rs * k, b * Rs, t * Rs * k))


# ==============================================================================================
# the limb of one side, placed on its girdle


@dataclass
class SideBuild:
    """One limb built on its girdle (the world frame of its side)."""

    mesh: Mesh
    joints: tuple              # (pivot, elbow, hand) at rest, in the world frame of the side
    girdle: Girdle             # LEFT frame
    info: dict
    side: str
    site: Site
    plug: Mesh                 # the hidden deep mass, in the world frame of the side (also part of ``mesh``)


def build_side(base, verts, site: Site, side: str = "l", scale: float | None = None,
               out_extra: float = 0.0, point=None, dome: float | None = None) -> SideBuild:
    """The limb of ``site`` on one side of the body ``verts``: cut out of the body across the
    upper bone, closed with the flared proximal limb and the dome, moved onto the girdle's pivot
    (scaled about it) and joined to the deep mass. Built in the left frame and reflected back for
    the right side."""
    from .appendages import cap_boundaries
    dome = site.dome if dome is None else dome
    g = measure(base, verts, site, side, scale, out_extra, point)
    body_v, body_q = base.body_cage(verts)
    if side == "r":
        body_v, body_q = _reflect(body_v), body_q[:, ::-1]       # a reflection turns the quads inside out
    j, e, h = g.joints
    u = (e - j) / g.Lu
    best = None
    last = None
    for t0 in (CUT_T, 0.35, 0.40, 0.45, 0.25):
        try:
            cut = cut_limb(base, body_v, body_q, g.joints, g.Lu, g.R, t0)
        except ValueError as exc:
            last = exc
            continue
        # the limb's own axis through the ring is parallel to the bone but not on it (the arm's section
        # is centred a little in front of the joint marker): the dome is centred on that axis so the
        # limb does not bend where the dome starts, and that point is the second joint
        c_ring = cut.verts[cut.ring].mean(axis=0)
        jn = c_ring - ((c_ring - j) @ u) * u
        slope = None
        try:                               # the limb's own taper just beyond the cut
            tris_b = quad_triangles(body_q)
            r0 = section_radius(body_v, tris_b, j + u * (cut.plane_t * g.Lu), e - j)
            r1 = section_radius(body_v, tris_b, j + u * ((cut.plane_t + 0.1) * g.Lu), e - j)
            slope = (r0 - r1) / r0 * (cut.plane_t / 0.1)
        except ValueError:
            pass
        V, T, info = dome_root(cut, jn, e - j, g.R, dome=dome, slope=slope, lean=-g.normal * site.lean)
        worst = ring_dihedral(V, T, info["ring"])
        if best is None or worst < best[0]:
            best = (worst, cut, jn, V, T, info)
        if worst < RING_DIHEDRAL_MAX:
            break
    if best is None:
        raise ValueError(f"cannot cut the {site.limb} out of this body ({last})")
    _, cut, jn, V, T, info = best
    V, T = cap_boundaries(V, T)
    sc = g.scale
    V = g.pivot + sc * (V - jn)
    limb = Mesh(V, T)
    plug = plug_mesh(g)
    mesh = Mesh.merge([limb, plug])
    joints = tuple(g.pivot + sc * (np.asarray(p) - jn) for p in (jn, e, h))
    info = dict(info, plane_t=cut.plane_t, limb_faces=len(T), n_limb_verts=len(V), joint=jn,
                cut_points=cut.verts if side == "l" else _reflect(cut.verts))
    if side == "r":
        mesh = Mesh(_reflect(mesh.vertices), mesh.faces[:, ::-1])
        plug = Mesh(_reflect(plug.vertices), plug.faces[:, ::-1])
        joints = tuple(_reflect(p) for p in joints)
    return SideBuild(mesh, joints, g, info, side, site, plug)


GIRDLE_PIN, GIRDLE_BLEND = 0.0, 0.45      # the rig of a girdle limb: the root (dome, flare) is pinned, the arm turns over 0.45 Lu


def _mirror_build(b: SideBuild) -> SideBuild:
    """The right-hand twin of a left-hand build (vertex for vertex)."""
    mesh = Mesh(_reflect(b.mesh.vertices), b.mesh.faces[:, ::-1])
    plug = Mesh(_reflect(b.plug.vertices), b.plug.faces[:, ::-1])
    return SideBuild(mesh, tuple(_reflect(p) for p in b.joints), b.girdle, b.info, "r", b.site, plug)


def place_limbs(app, base, verts=None) -> list:
    """``[(idx, LimbRig)]`` for the sides of ``app`` (an :class:`~.appendages.Appendage` with a
    ``girdle`` site): the character's own arm or leg, cut across the upper bone, closed with the
    shoulder dome and put on the girdle that :func:`measure` finds on this body. The right limb
    is the exact reflection of the left one on a symmetric body (and the pair's right limb when
    it is built alone), and is built from the right limb on an asymmetric one."""
    from . import appendages as A
    from .limbrig import LimbRig
    v = base.verts if verts is None else verts
    site = SITES[app.girdle]
    body_v, body_q = base.body_cage(v)
    scale = app.scale                      # the record's own scale (the preset's 0.9 / 0.7 is stored in it)
    left = build_side(base, v, site, "l", scale, app.girdle_out, _point_of(app, base, body_v, body_q, 0),
                      app.girdle_dome)
    out = []
    right = None
    for idx in app._indices():
        if idx == 0:
            b = left
        else:
            if right is None:
                try:
                    own = build_side(base, v, site, "r", scale, app.girdle_out,
                                     _point_of(app, base, body_v, body_q, 1), app.girdle_dome)
                except ValueError:
                    own = None
                if own is not None and not A._is_mirror_image(left.info["cut_points"], own.info["cut_points"]):
                    right = own
                else:
                    right = _mirror_build(left)
            b = right
        pivot, el, ha = b.joints
        rig = LimbRig(b.mesh, pivot, el, ha, limb=site.limb, mirrored=idx == 1, root_loop=b.info["equator"],
                      root_pole=b.info["pole"], limits=dict(site.limits), pin_fraction=GIRDLE_PIN,
                      blend_fraction=GIRDLE_BLEND, girdle=b)
        out.append((idx, rig))
    return out


def _point_of(app, base, body_v, body_q, idx):
    """The point a user-anchored girdle is built at (the left-frame position of the bound quad),
    or ``None`` when the girdle is the preset's own (found by its landmark rule)."""
    if not app.girdle_free:
        return None
    anchor = app._face_anchor(body_v, body_q, idx, np.asarray(app.up_hint, dtype=np.float64), base)
    p, _ = anchor.resolve(body_v)
    return _reflect(p) if idx == 1 else p


# ==============================================================================================
# meshes of the anatomy: tubes, capsules


def tube_mesh(points, radii, segments: int = 14, caps: tuple = (True, True), cap_rings: int = 4,
              ellipse=None) -> Mesh:
    """A closed tube along the polyline ``points`` with ``radii`` at each point, rounded at the
    ends (hemispheres of the end radii) when ``caps`` say so. ``ellipse`` (a tuple ``(vector,
    ratio)``) stretches every ring by ``ratio`` along ``vector`` (projected into the ring's plane)."""
    P = np.asarray(points, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    n = len(P)
    tang = np.zeros_like(P)
    tang[1:-1] = P[2:] - P[:-2]
    tang[0] = P[1] - P[0]
    tang[-1] = P[-1] - P[-2]
    tang = tang / np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), _EPS)
    # parallel-transport frames
    e1, _ = _perp(tang[0])
    frames = []
    cur = e1
    for i in range(n):
        cur = cur - (cur @ tang[i]) * tang[i]
        cur = cur / max(float(np.linalg.norm(cur)), _EPS)
        frames.append((cur, np.cross(tang[i], cur)))
    th = 2.0 * math.pi * np.arange(segments) / segments
    rings = []
    for i in range(n):
        a, b = frames[i]
        sx = sy = 1.0
        pts = P[i] + r[i] * (np.cos(th)[:, None] * a + np.sin(th)[:, None] * b)
        if ellipse is not None:
            vec, ratio = ellipse
            w = np.asarray(vec, dtype=np.float64)
            w = w - (w @ tang[i]) * tang[i]
            ln = float(np.linalg.norm(w))
            if ln > 1e-9:
                w = w / ln
                rel = pts - P[i]
                pts = P[i] + rel + (ratio - 1.0) * (rel @ w)[:, None] * w
        rings.append(pts)
    rows = list(rings)
    start_cap, end_cap = [], []
    if caps[0]:
        for k in range(1, cap_rings + 1):
            phi = 0.5 * math.pi * k / (cap_rings + 1)
            a, b = frames[0]
            start_cap.append(P[0] - tang[0] * (r[0] * math.sin(phi)) + math.cos(phi) * (rings[0] - P[0]))
    if caps[1]:
        for k in range(1, cap_rings + 1):
            phi = 0.5 * math.pi * k / (cap_rings + 1)
            end_cap.append(P[-1] + tang[-1] * (r[-1] * math.sin(phi)) + math.cos(phi) * (rings[-1] - P[-1]))
    rows = start_cap[::-1] + rows + end_cap
    verts = [row for ring in rows for row in ring]
    faces = []
    m = len(rows)
    for i in range(m - 1):
        for k in range(segments):
            k1 = (k + 1) % segments
            a, b = i * segments + k, i * segments + k1
            c, d = (i + 1) * segments + k1, (i + 1) * segments + k
            faces.append((a, b, c))
            faces.append((a, c, d))
    # poles / flat caps
    first = np.array(rows[0])
    last = np.array(rows[-1])
    ps = len(verts)
    verts.append(first.mean(axis=0) - (tang[0] * r[0] * 0.0 if caps[0] else 0.0))
    for k in range(segments):
        k1 = (k + 1) % segments
        faces.append((ps, k1, k))
    pe = len(verts)
    verts.append(last.mean(axis=0))
    base_i = (m - 1) * segments
    for k in range(segments):
        k1 = (k + 1) % segments
        faces.append((pe, base_i + k, base_i + k1))
    mesh = Mesh(np.array(verts), np.array(faces, dtype=np.int64))
    if mesh.volume() < 0:
        mesh = Mesh(mesh.vertices, mesh.faces[:, ::-1])
    return mesh


# ==============================================================================================
# the model of one placed limb: junction (the flesh around the root) and layers


# the muscle masses of the girdle that live in the BODY's skin (the cage carries features of
# 0.8 R and more), as Mounds, in R of the second limb: heights are over the original skin.
MASSES = {
    "arm": dict(skirt=(2.8, 0.25, 0.35), anterior=(1.0, 0.20), posterior=(1.5, 0.30)),
    "leg": dict(skirt=(2.6, 0.25, 0.3), anterior=(1.0, 0.18), posterior=(1.2, 0.25)),
}
SKIN_THICKNESS = 0.12          # of R: skin and fat over muscle (6-10 mm on a lean male)
KEEP_REACH = 1.45              # the original limb is protected to this many R around its bones
KEEP_FROM_FAN = 0.5            # a fan pair shares the primary shoulder: the arm is protected from half way down
KEEP_FROM_LEG = 0.35           # a leg is protected from this fraction of its thigh down (the girdle sits at the hip)


class LimbModel:
    """What a placed girdle limb knows about its anatomy: the build (left frame girdle, joints,
    the deep mass), the rig that poses it, and the junction / layers derived from both."""

    def __init__(self, rig, build: SideBuild):
        self.rig = rig
        self.build = build
        self.girdle = build.girdle
        self.right = build.side == "r"

    # -- frames --------------------------------------------------------------------------

    def _left(self, p):
        return _reflect(p) if self.right else np.asarray(p, dtype=np.float64)

    def _world(self, p):
        return _reflect(p) if self.right else np.asarray(p, dtype=np.float64)

    def posed_joints(self, controls=None) -> tuple:
        """The posed ``(pivot, elbow, hand)`` in the LEFT frame."""
        j = self.rig.joints(controls or {})
        return tuple(self._left(j[k]) for k in ("shoulder", "elbow", "hand"))

    @property
    def R(self) -> float:
        """The limb's own radius unit (the donor's R times the scale)."""
        return self.girdle.R * self.girdle.scale

    # -- the junction --------------------------------------------------------------------

    def mounds(self, mass: float = 1.0) -> tuple:
        """The muscle masses of the girdle as skin mounds (left frame)."""
        from . import junction as J
        g = self.girdle
        cfg = MASSES[g.limb]
        S = g.skin
        n, up, ant = flank_frame(g.normal)
        post = -ant
        R = self.R
        m = float(mass)
        if g.fan:                                    # the trapezius ridge and the scapular plateau under the shared dome
            medial = _unit(np.array([-1.0, 0.0, 0.0]) - (np.array([-1.0, 0.0, 0.0]) @ n) * n)
            return (J.Mound(tuple(S), 2.4 * R, 0.30 * R * m),
                    J.Mound(tuple(S + up * 0.4 * R), 1.0 * R, 0.25 * R * m,
                            end=tuple(S + up * 1.4 * R + medial * 1.6 * R)))
        rs, hs, off = cfg["skirt"]
        ra, ha = cfg["anterior"]
        rp, hp = cfg["posterior"]
        ridge_end = S + ant * (0.9 + 2.4 * math.cos(math.radians(20))) * R + up * 2.4 * math.sin(math.radians(20)) * R
        out = [
            J.Mound(tuple(S + post * off * R), rs * R, hs * R * m),
            J.Mound(tuple(S + ant * 0.9 * R), ra * R, ha * R * m, end=tuple(ridge_end)),
            J.Mound(tuple(S + post * 0.8 * R + up * 0.2 * R), rp * R, hp * R * m,
                    end=tuple(S + post * (2.4 if g.limb == "arm" else 1.5) * R + up * 0.5 * R)),
        ]
        return tuple(out)

    def keeps(self) -> tuple:
        """Keep spheres along the original limb: its skin is not this appendage's to mould."""
        from . import junction as J
        a, b, c = (np.asarray(p, dtype=np.float64) for p in self.girdle.joints)
        rk = KEEP_REACH * self.girdle.R
        out = []
        if self.girdle.limb == "leg":      # a second hip girdle sits just above the first hip: protect the thigh below it
            a = a + KEEP_FROM_LEG * (b - a)
        elif self.girdle.fan:              # a fan pair shares the shoulder: protect the arm from below its cap
            a = a + KEEP_FROM_FAN * (b - a)
        for p0, p1 in ((a, b), (b, c)):
            n = max(int(math.ceil(float(np.linalg.norm(p1 - p0)) / (1.1 * self.girdle.R))), 1)
            for i in range(n + 1):
                out.append(J.Keep(tuple(p0 + (p1 - p0) * i / n), rk, 0.3))
        return tuple(out)

    def junction(self, controls=None, fillet: float = 1.0, height: float = 0.35, mass: float = 1.0,
                 hug: float = 0.97):
        """The :class:`~.junction.JunctionSpec` of the root in the WORLD frame of its side."""
        from . import junction as J
        pv, el, ha = self.posed_joints(controls)
        R = self.R
        g = self.girdle
        ts = np.array([0.0, 0.15, 0.3, 0.5, 0.75, 1.0])
        pts = [pv + t * (el - pv) for t in ts]
        radii = [float(radius_law(t)) * R * hug for t in ts]
        spec = J.JunctionSpec(f"girdle", pts, radii, width=fillet,
                              base_radius=R * float(radius_law(0.25)), height=height,
                              mounds=self.mounds(mass), keep=self.keeps())
        return spec.mirrored() if self.right else spec

    # -- layers --------------------------------------------------------------------------

    def _skin_points(self, skin: Mesh) -> np.ndarray:
        """The limb's own skin vertices (not the deep mass) in the left frame."""
        v = skin.vertices[: self.build.info["n_limb_verts"]]
        return _reflect(v) if self.right else v

    def _profile(self, pts, chain, tangents, half: float):
        """``(centres, radii)`` of the skin around the bone at every station: the centroid of the
        skin vertices in a slab (``half`` thick) across the bone and the smallest distance of the
        slab's vertices to that centroid (the inscribed radius of the section)."""
        centres = np.array(chain, dtype=np.float64)
        out = np.full(len(chain), np.nan)
        for i, (c, tau) in enumerate(zip(chain, tangents)):
            rel = pts - c
            al = rel @ tau
            sel = np.abs(al) < half
            if sel.sum() < 6:
                continue
            perp = rel[sel] - al[sel][:, None] * tau
            cen = perp.mean(axis=0)
            centres[i] = c + cen
            out[i] = float(np.linalg.norm(perp - cen, axis=1).min())
        good = ~np.isnan(out)
        if good.any() and not good.all():
            out = np.interp(np.arange(len(out)), np.nonzero(good)[0], out[good])
        return centres, out

    def layers(self, skin: Mesh, controls=None) -> dict:
        """``{"bone", "muscle", "skin"}`` in the MakeHuman frame of this side.

        All three come from one set of numbers: the joints of the posed limb, the limb unit ``R``,
        the girdle (the trunk section the masses sit on) and the skin thickness. The muscle masses
        are sized from the skin's own inscribed radius at each station along the bone, less the
        skin thickness (``SKIN_THICKNESS`` R), so the skin is their envelope and they never come
        through it; the bone is sized from ``R`` and sits inside the muscle."""
        g = self.girdle
        R = self.R
        t_skin = SKIN_THICKNESS * R
        pv, el, ha = self.posed_joints(controls)
        pts = self._skin_points(skin)
        n, up, ant = flank_frame(g.normal)
        u = _unit(el - pv)
        L_up = float(np.linalg.norm(el - pv))
        L_fo = float(np.linalg.norm(ha - el))
        fwd = np.array([0.0, 0.0, 1.0])

        def perp_to(vec, ref):
            w = np.asarray(ref, dtype=np.float64)
            w = w - (w @ vec) * vec
            if np.linalg.norm(w) < 1e-6:
                w = np.array([0.0, 1.0, 0.0]) - vec[1] * vec
            return _unit(w)

        a_up = perp_to(u, fwd)                       # anterior of the upper arm
        b_up = _unit(np.cross(u, a_up))
        lat = perp_to(u, n)                          # the side away from the trunk
        f = _unit(ha - el)
        a_fo = perp_to(f, fwd)
        b_fo = _unit(np.cross(f, a_fo))
        ts = np.linspace(0.1, 1.0, 10)
        chain_up = np.array([pv + t * (el - pv) for t in ts])
        cen_up, prof_up = self._profile(pts, chain_up, [u] * len(ts), 0.05 * L_up)
        prof_up = np.maximum(prof_up - t_skin, 0.25 * R)
        tf = np.linspace(0.0, 0.85, 8)
        chain_fo = np.array([el + t * (ha - el) for t in tf])
        cen_fo, prof_fo = self._profile(pts, chain_fo, [f] * len(tf), 0.06 * L_fo)
        prof_fo = np.maximum(prof_fo - t_skin, 0.2 * R)

        def r_up(t):
            return float(np.interp(t, ts, prof_up))

        def c_up(t):
            return np.array([np.interp(t, ts, cen_up[:, k]) for k in range(3)])

        def r_fo(t):
            return float(np.interp(t, tf, prof_fo))

        def c_fo(t):
            return np.array([np.interp(t, tf, cen_fo[:, k]) for k in range(3)])

        def ell(center, axis_a, axis_b, axis_t, ra, rb, rt):
            return ellipsoid(center, np.array([axis_a, axis_b, axis_t]), (ra, rb, rt))

        # -- muscle: the sleeve of the limb in bellies, centred on the limb's own section ---------
        dome_sel = ((pts - pv) @ u) < 0.15 * R
        dome_r = float(np.linalg.norm(pts[dome_sel] - pv, axis=1).min()) if dome_sel.any() else DOME_RADIUS * R
        muscle = [ellipsoid(pv, np.eye(3), (dome_r - t_skin,) * 3, rings=10, segments=16)]
        self._parts = {}
        MARGIN = 0.85                                 # of the inscribed radius: a bend of the skin shears its sections

        def belly(track, radius, t0, t1, off_dir, off, ra, rb, spindle=0.6, k=7, up=None):
            """A spindle along the limb's own centre line (so it follows a bend of the skin): the
            section is an ellipse ``ra`` x ``rb`` of the local inscribed radius, shifted ``off`` toward
            ``off_dir``."""
            ts_ = np.linspace(t0, t1, k)
            pts_ = [track(t) + off_dir * (off * radius(t) * MARGIN) for t in ts_]
            prof = np.sin(np.pi * (np.arange(k) + 0.5) / k) ** spindle
            rad = np.array([rb * radius(t) * MARGIN for t in ts_]) * prof
            return tube_mesh(pts_, rad, segments=12, caps=(True, True), cap_rings=3,
                             ellipse=(off_dir, ra / rb))

        t_d = 0.24                                    # deltoid: the cap's tail down the lateral side to its V
        muscle.append(belly(c_up, r_up, 0.05, 0.62, lat, 0.18, 0.78, 0.7))
        muscle.append(belly(c_up, r_up, 0.18, 0.88, a_up, 0.28, 0.72, 0.62))       # biceps
        muscle.append(belly(c_up, r_up, 0.12, 0.82, -a_up, 0.28, 0.72, 0.66))      # triceps
        muscle.append(belly(c_fo, r_fo, 0.02, 0.62, a_fo, 0.22, 0.7, 0.62))        # the flexors
        muscle.append(belly(c_fo, r_fo, 0.02, 0.6, -a_fo, 0.22, 0.68, 0.6))        # the extensors
        # the masses on the trunk (under the mounds of the junction: heights less the skin), as
        # tubes along the trunk section so they follow its curve
        cfg = MASSES[g.limb]
        strips = () if g.fan else (("anterior", (0.9, 2.2), 0.55), ("posterior", (-0.8, -2.8), 0.9))
        for key, (a0, a1), up_ext in strips:
            h_mound = cfg[key][1] * R - t_skin
            if h_mound <= 0.0:
                continue
            semi_n = 0.42 * R
            arcs = np.linspace(a0, a1, 7) * R
            walk = [loop_walk(g.loop, g.skin, d) for d in arcs]
            centers = [sk + nrm * (h_mound - semi_n - 0.2 * R) for sk, nrm in walk]
            taper = np.array([0.45, 0.8, 1.0, 1.0, 1.0, 0.8, 0.45])
            muscle.append(tube_mesh(centers, semi_n * taper, segments=12, caps=(True, True), cap_rings=3,
                                    ellipse=(np.array([0.0, 1.0, 0.0]), up_ext * R / semi_n)))
        # -- bone -------------------------------------------------------------------------
        bone = [ellipsoid(pv, np.eye(3), (0.5 * R,) * 3, rings=8, segments=14)]
        hs = np.array([0.0, 0.2, 0.5, 0.8, 0.95, 1.0])
        hr = np.array([0.20, 0.17, 0.15, 0.16, 0.2, 0.26]) * R
        bone.append(tube_mesh([pv + t * (el - pv) for t in hs], hr, segments=10))
        wr = el + 0.8 * (ha - el)
        for sgn, r0 in ((1.0, 0.12), (-1.0, 0.11)):
            off = b_fo * (0.14 * R * sgn)
            bone.append(tube_mesh([el + off, 0.5 * (el + wr) + off, wr + off * 0.7],
                                  [r0 * R, 0.9 * r0 * R, 0.7 * r0 * R], segments=8))
        if g.limb == "leg":                           # the ankle: a small block (the foot beyond it is wider than any bone)
            bone.append(ell(wr + (ha - wr) * 0.4, b_fo, a_fo, f, 0.16 * R, 0.16 * R, 0.5 * np.linalg.norm(ha - wr)))
        else:                                         # the hand: the carpals and metacarpals, a flat block
            bone.append(ell(wr + (ha - wr) * 0.4, b_fo, a_fo, f, 0.3 * R, 0.1 * R, 0.5 * np.linalg.norm(ha - wr)))
        if g.fan or g.limb == "leg":                 # the scapular spine under the shared dome / the iliac strut, toward the spine
            medial = _unit(np.array([-1.0, 0.0, 0.0]) - (np.array([-1.0, 0.0, 0.0]) @ n) * n)
            if g.limb == "leg":                       # toward the sacrum: the middle of the pelvic section
                medial = _unit((g.centroid - g.pivot) * np.array([1.0, 0.0, 1.0]))
            tip = g.pivot + medial * 2.2 * R - n * 0.8 * R
            mid = 0.5 * (g.pivot + tip) - n * 0.2 * R
            bone.append(tube_mesh([g.pivot - n * 0.25 * R, mid, tip], [0.15 * R, 0.14 * R, 0.12 * R], segments=10))
        else:
            dirc = _unit(0.80 * n - 0.55 * ant + 0.17 * up)       # medial-anterior: from the costal arch to the joint
            start = g.pivot - dirc * 3.0 * R
            clav = tube_mesh([start, 0.5 * (start + g.pivot), g.pivot], [0.2 * R, 0.17 * R, 0.2 * R], segments=10)
            bone.append(clav)
            walk = [loop_walk(g.loop, g.skin, d * R) for d in np.linspace(-1.8, -3.4, 6)]
            thin = 0.16 * R                               # the scapula: a plate on the back of the ribs
            bone.append(tube_mesh([sk - nrm * (0.5 * R + thin) for sk, nrm in walk],
                                  thin * np.array([0.5, 0.9, 1, 1, 0.9, 0.5]), segments=10, caps=(True, True),
                                  cap_rings=3, ellipse=(np.array([0.0, 1.0, 0.0]), 1.5 * R / thin)))
        self._parts = {"muscle": list(muscle), "bone": list(bone)}       # the pieces, for the checks
        out = {"bone": bone, "muscle": muscle}
        res = {}
        for k, meshes in out.items():
            m = Mesh.merge(meshes)
            res[k] = Mesh(_reflect(m.vertices), m.faces[:, ::-1]) if self.right else m
        res["skin"] = skin
        return res


def loop_walk(loop, point, distance: float):
    """The point of the trunk section ``loop`` (``(k, 3)``, an xz polygon) a signed ``distance``
    along it from the point nearest ``point`` (positive = anterior), and the outward horizontal
    normal there."""
    pts = np.asarray(loop, dtype=np.float64)
    k = len(pts)
    i0 = int(np.argmin(np.linalg.norm(pts[:, [0, 2]] - np.asarray(point)[[0, 2]], axis=1)))
    nxt = pts[(i0 + 1) % k] - pts[i0]
    step = 1 if nxt[2] >= 0 else -1                 # the direction of increasing z at a lateral point
    rem = abs(distance)
    sgn = step if distance >= 0 else -step
    cur = pts[i0].copy()
    i = i0
    for _ in range(k):
        j = (i + sgn) % k
        seg = pts[j] - cur
        ln = float(np.linalg.norm(seg[[0, 2]]))
        if ln >= rem:
            cur = cur + seg * (rem / max(ln, _EPS))
            rem = 0.0
            tangent = seg[[0, 2]] / max(ln, _EPS)
            break
        rem -= ln
        cur = pts[j].copy()
        i = j
        tangent = seg[[0, 2]] / max(ln, _EPS)
    nrm = np.array([tangent[1], 0.0, -tangent[0]])
    c_xz = pts[:, [0, 2]].mean(axis=0)
    if (nrm[[0, 2]] @ (cur[[0, 2]] - c_xz)) < 0:
        nrm = -nrm
    return cur, nrm
