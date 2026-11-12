"""The root collar: the seam between the shaped skin and an appendage, built on the appendage's own shell.

WHY.  The skin junction (:mod:`.junction`) moves the vertices of the 13,378-quad body cage, which is
what gives the skin MASS around a root.  It cannot give it an exact tangent seam: a cage quad is
about 0.28 units across and a root 0.4-0.6, so the skin meets the root 53-100 degrees from tangent
and a crisp ring or flange survives in every close-up (the round-1 critics: "a flange", "a faint ring
band", "the seam survives on every body").  The collar is a transition that does not depend on the
cage at all: a thin closed shell whose OUTER rim lies on the shaped skin, tangent to it, that rises
along the appendage's wall to join the wall tangentially.  It is part of the APPENDAGE's own
geometry (resolution independent: 64 x 22 samples whatever the body is subdivided to) and the
slicer unions it with the body and the appendage like every other overlapping shell.

WHAT IT IS.  For every angle ``theta`` around the appendage's axis:

* ``X(theta)`` is where the wall's generator (a line along the appendage's surface) pierces the
  skin, found on the REAL appendage mesh (a ray from the axis), not on an idealised tube;
* the skin is walked outward from ``X`` over the subdivided body (a step along the tangent
  direction, a closest-point projection back onto the surface: a geodesic-ish radial distance
  that follows the curved sacrum) to the rim ``R = P(W(theta))``;
* the visible profile is ``top(rho) = P(rho) + h(rho) D(rho)``: the skin point ``P`` plus a height
  ``h`` along a direction ``D`` that is the skin normal at the rim and the wall's generator at the
  wall.  ``h = H (1 - sqrt(1 - x^k))`` with ``x = 1 - rho / W`` is vertical at the wall (tangent to
  the appendage) and flat at the rim with the second derivative vanishing for ``k >= 3`` (tangent to
  the skin, G1 by construction because ``h >= 0`` and ``h' = 0`` there);
* past the rim the sheet dips under the skin (``sink``) and closes under the body; past the wall it
  curves INTO the appendage and closes inside it.  Both closing parts are hidden, so every vertex
  whose normal is ever seen has only top-sheet neighbours (the renderer's area weighted vertex
  normals are not polluted by a side wall: that is the ring this module exists to remove);
* ``W(theta)`` is not symmetric: the open side (the side facing ``up``: the back, the flank) has
  the full width, the folded side (the underside of a tail, against the thigh) a small one: the
  references have the dorsal contour run into the tail tangent-continuously and the ventral side a
  crease (docs/collar.md), and a collar that filled the natal cleft would be the blister the
  reviewers named.  Where the wall leans over the skin (a lifted tail) the width falls to the floor.

API: ``Kind.collar_profile(placed)`` (``kinds.py``; ``None``: no collar), :func:`collar_pieces`,
:func:`build_collar` (the geometry alone, any skin and any wall mesh), :func:`join_report` (the
numbers).  Everything is in the MakeHuman frame (Y up) like ``Placed.mesh()``.
"""

from __future__ import annotations

import hashlib
import math
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass, replace

import numpy as np

from ..core.mesh import Mesh

_EPS = 1e-12

# the sampling of the shell (the task asks for >= 48 x >= 12; the visible part is 3 + RINGS rings)
SEGMENTS = 72
RINGS = 14
SKIRT_U = (0.04, 0.1, 0.2, 0.35, 0.6, 1.0)      # the dive under the skin past the rim, in skirt widths (graded: a gentle start)
SKIRT_WIDTH = 0.9                               # the skirt is this many widths (of the visible reach) wide
ARC_ANGLES = (2.5, 7.0, 16.0, 32.0, 55.0, 90.0)
LEAN_FROM_DEG, LEAN_FULL_DEG = 62.0, 100.0      # where the wall leans over the skin (angle to the outward skin direction) the collar fades out
FIT_TOL = 0.02                                  # the exit point must be this close (units) to the skin to carry a collar
RIM_TOL_DEG, RIM_ITERATIONS = 15.0, 1                   # a rim whose sheet is more than this off the skin's tangent plane gets narrower (and the sheet is built again)
RIM_SMOOTHING = 10                              # rounds of 1-2-1 smoothing of the width around the ring
SINK_FLOOR = 0.35                               # the sink of the thinnest collar, as a share of the full sink
MIN_WIDTH = 0.006                               # the collar never gets thinner than this (units) so the shell has no zero-area faces
EXTRA_NEAR_WALL = 0.5                           # one extra station between the wall and the first one (in steps)
VISIBLE_WIDTH = 0.04                            # a collar thinner than this (units) is not counted in the report's angles
SKIN_TOL = 0.002                                # the visible sheet may sink this far under the skin (the facets of the subdivided surface)
WALL_PAD = 0.002                                # the join ring stands this far (units) outside the faceted wall, so the collar always wins the depth test there
CREASE_DROP_DEG = 28.0                          # ... or whose chord leaves the tangent plane of the one before by more than this (a groove narrower than a step)
CREASE_STEP_DEG = 18.0                          # a crease: consecutive stations of the march whose skin normals differ by more than this           # degrees: the sheet curves into the appendage
SKIN_MATERIALS = ("skin", "appendage")
MIN_LEVEL = 1          # the body must be meshed at level 1 or 2: level 0 is the 0.28-unit cage, a coarse polyhedron the collar would stand off in flaps


# --------------------------------------------------------------------------
# the profile


@dataclass(frozen=True)
class CollarProfile:
    """How a kind wants its collar, in units of the ROOT RADIUS (the radius of the tube where
    it leaves the skin) so the same profile fits every body and every tail.

    ``width``        reach along the skin on the OPEN side (the side facing ``up``), in root radii.
    ``ventral``      the same on the opposite (folded) side, in root radii.
    ``height``       how far the sheet climbs the wall, in root radii.
    ``sink``         how far the underside is sunk under the skin, in root radii (the weld).
    ``falloff``      exponent ``k`` of the profile ``h = H (1 - sqrt(1 - x^k))``: 2 is a quarter
                     ellipse (C1 at the rim), 3 is curvature-continuous there, larger is flatter
                     near the rim and steeper at the wall.
    ``lift``         a hair (units) by which the rim stands above the skin (0: a positive lift shows as a hairline).
    ``ramp``         the slope (tangent) at which the sheet leaves the skin at the rim: 0.035 is 2.0 degrees, enough that the
                     sheet is clear of the skin's facets 0.05 units in, so the rim and the skin do not z-fight along
                     a wide coplanar band (which shows as a ragged edge where their shading differs).
    ``up``           MakeHuman-frame direction of the open side (``None``: a uniform collar).
    ``material``     ``"skin"`` (the body's) or ``"appendage"`` (the appendage's own material).
    ``strength``     scales width and height; 0 switches the collar off.
    """

    width: float = 1.6
    ventral: float = 0.0
    height: float = 0.5
    sink: float = 0.2
    falloff: float = 2.5
    lift: float = 0.0
    ramp: float = 0.035
    open_deg: float = 60.0
    fold_deg: float = 115.0
    up: tuple | None = (0.0, 1.0, 0.0)
    material: str = "skin"
    strength: float = 1.0
    segments: int = SEGMENTS
    rings: int = RINGS

    def __post_init__(self):
        for name in ("width", "ventral", "height", "sink", "falloff", "lift", "ramp", "strength", "open_deg", "fold_deg"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    and v >= 0):
                raise ValueError(f"{name} must be a finite number >= 0, got {v!r}")
            object.__setattr__(self, name, float(v))
        if self.falloff < 1.5:
            raise ValueError("falloff must be >= 1.5")
        if not 0.0 <= self.open_deg < self.fold_deg <= 180.0:
            raise ValueError("0 <= open_deg < fold_deg <= 180")
        if self.material not in SKIN_MATERIALS:
            raise ValueError(f"material must be one of {SKIN_MATERIALS}, got {self.material!r}")
        if self.up is not None:
            up = tuple(float(x) for x in self.up)
            if len(up) != 3 or not all(math.isfinite(x) for x in up) or not any(up):
                raise ValueError(f"up must be three finite numbers, not all zero, got {self.up!r}")
            object.__setattr__(self, "up", up)
        if int(self.segments) < 16 or int(self.rings) < 4:
            raise ValueError("segments >= 16 and rings >= 4")
        object.__setattr__(self, "segments", int(self.segments))
        object.__setattr__(self, "rings", int(self.rings))

    @property
    def active(self) -> bool:
        return self.strength > 0 and self.width > 0 and self.height > 0

    def with_(self, **changes) -> "CollarProfile":
        return replace(self, **changes)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("width", "ventral", "height", "sink", "falloff", "lift", "ramp",
                                           "open_deg", "fold_deg", "material", "strength", "segments", "rings")}
        d["up"] = None if self.up is None else list(self.up)
        return d

    def digest(self) -> str:
        import json
        return hashlib.blake2b(json.dumps(self.to_dict(), sort_keys=True).encode(),
                               digest_size=8).hexdigest()


# --------------------------------------------------------------------------
# small vector helpers


def _unit(v, eps=_EPS):
    v = np.asarray(v, dtype=np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def _smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _cross(a, b):
    return np.cross(a, b)


# --------------------------------------------------------------------------
# closest point on triangles


def closest_on_triangles(p, a, b, c):
    """Ericson's closest point of ``p`` (n, 3) on the triangles ``a b c`` (n, 3 each):
    ``(point, barycentric (n, 3))``. Vectorised, exact (all seven regions)."""
    ab, ac, ap = b - a, c - a, p - a
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)
    bp = p - b
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)
    cp = p - c
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    n = len(p)
    u = np.zeros(n)
    v = np.zeros(n)
    w = np.zeros(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        den = va + vb + vc
        iv, iw = vb / den, vc / den                       # interior
        t_ab = d1 / (d1 - d3)
        t_ac = d2 / (d2 - d6)
        t_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    # decide the region (Ericson's order)
    done = np.zeros(n, dtype=bool)

    def put(mask, bu, bv, bw):
        m = mask & ~done
        u[m], v[m], w[m] = bu[m] if np.ndim(bu) else bu, bv[m] if np.ndim(bv) else bv, \
            bw[m] if np.ndim(bw) else bw
        done[m] = True

    one, zero = np.ones(n), np.zeros(n)
    put((d1 <= 0) & (d2 <= 0), one, zero, zero)                                     # vertex a
    put((d3 >= 0) & (d4 <= d3), zero, one, zero)                                    # vertex b
    put((vc <= 0) & (d1 >= 0) & (d3 <= 0), 1.0 - t_ab, t_ab, zero)                 # edge ab
    put((d6 >= 0) & (d5 <= d6), zero, zero, one)                                    # vertex c
    put((vb <= 0) & (d2 >= 0) & (d6 <= 0), 1.0 - t_ac, zero, t_ac)                 # edge ac
    put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), zero, 1.0 - t_bc, t_bc)    # edge bc
    put(np.ones(n, dtype=bool), 1.0 - iv - iw, iv, iw)                              # interior
    bad = ~np.isfinite(u) | ~np.isfinite(v) | ~np.isfinite(w)
    if bad.any():                                           # a degenerate triangle: its vertex a
        u[bad], v[bad], w[bad] = 1.0, 0.0, 0.0
    bary = np.stack([u, v, w], axis=1)
    pt = a * u[:, None] + b * v[:, None] + c * w[:, None]
    return pt, bary


# --------------------------------------------------------------------------
# the skin: a local patch of the body mesh, closest-point queries, vertex normals


class SkinPatch:
    """The part of a triangle mesh (the subdivided body) around a point, with smooth vertex
    normals (the SAME area weighted normals the renderer shades with) and closest-point queries.

    ``closest(points)`` returns the closest point on the surface, its interpolated vertex normal
    and the signed distance (positive outside, along the normal): a nearest vertex by a uniform
    grid, then the triangles around it (so the cost is linear in the number of queries, not in the
    size of the patch)."""

    def __init__(self, vertices, faces, centre, radius):
        V = np.asarray(vertices, dtype=np.float64)
        F = np.asarray(faces, dtype=np.int64)
        centre = np.asarray(centre, dtype=np.float64)
        d2 = ((V - centre) ** 2).sum(axis=1)
        inside = d2 < radius * radius
        keep = inside[F].any(axis=1)
        F = F[keep]
        used = np.unique(F)
        remap = np.full(len(V), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        self.V = V[used]
        self.F = remap[F]
        self.n_vertices = len(self.V)
        if not len(self.F):
            raise ValueError("no skin near the collar")
        fn = _cross(self.V[self.F[:, 1]] - self.V[self.F[:, 0]],
                    self.V[self.F[:, 2]] - self.V[self.F[:, 0]])
        acc = np.zeros_like(self.V)
        flat = self.F.ravel()
        for c in range(3):
            acc[:, c] = np.bincount(flat, weights=np.repeat(fn[:, c], 3), minlength=len(self.V))
        self.N = _unit(acc)
        self._grid()
        self._incident()

    # -- acceleration structures ---------------------------------------------------------------

    def _grid(self):
        e = np.linalg.norm(self.V[self.F[:, 1]] - self.V[self.F[:, 0]], axis=1)
        self.cell = max(2.2 * float(np.median(e)), 1e-4)
        self.origin = self.V.min(axis=0) - 1e-9
        # the fine grid holds every vertex; the coarse ones (for a point far from the skin: a wall point
        # inside the body, 0.5 units under it) a thin sample, and a greedy walk finishes the job
        self._levels = [self._make_grid(self.cell, 1), self._make_grid(self.cell * 6.0, 9),
                        self._make_grid(self.cell * 36.0, 81)]

    def _make_grid(self, cell, stride):
        sub = np.arange(0, len(self.V), stride)
        idx = np.floor((self.V[sub] - self.origin) / cell).astype(np.int64)
        dims = idx.max(axis=0) + 3
        key = (idx[:, 0] + 1) * dims[1] * dims[2] + (idx[:, 1] + 1) * dims[2] + idx[:, 2] + 1
        order = np.argsort(key, kind="stable")
        return cell, dims, key[order], sub[order]

    def _block_search(self, P, level):
        """Nearest vertex among the 2x2x2 cells around each point (the block that holds the point's
        cell and the neighbour on the side of the point it is nearest to) at one grid level (-1: none)."""
        cell, dims, keys, order = level
        n = len(P)
        f = (P - self.origin) / cell
        idx = np.floor(f).astype(np.int64)
        side = np.where(f - idx >= 0.5, 1, -1)                      # the neighbour cell the point leans toward
        qs, ds, vs = [], [], []
        dy, dz = dims[1], dims[2]
        for ox in (0, 1):
            for oy in (0, 1):
                for oz in (0, 1):
                    ix = idx[:, 0] + 1 + ox * side[:, 0]
                    iy = idx[:, 1] + 1 + oy * side[:, 1]
                    iz = idx[:, 2] + 1 + oz * side[:, 2]
                    ok = ((ix >= 0) & (ix < dims[0]) & (iy >= 0) & (iy < dy)
                          & (iz >= 0) & (iz < dz))
                    key = ix * dy * dz + iy * dz + iz
                    lo = np.searchsorted(keys, key, side="left")
                    hi = np.searchsorted(keys, key, side="right")
                    cnt = np.where(ok, hi - lo, 0)
                    tot = int(cnt.sum())
                    if not tot:
                        continue
                    q = np.repeat(np.arange(n), cnt)
                    start = np.cumsum(cnt) - cnt
                    vi = order[np.arange(tot) - np.repeat(start, cnt) + np.repeat(lo, cnt)]
                    qs.append(q)
                    vs.append(vi)
                    ds.append(((self.V[vi] - P[q]) ** 2).sum(axis=1))
        best = np.full(n, -1, dtype=np.int64)
        if qs:
            q, d, vi = np.concatenate(qs), np.concatenate(ds), np.concatenate(vs)
            o = np.lexsort((d, q))
            q, vi = q[o], vi[o]
            first = np.concatenate([[True], q[1:] != q[:-1]])
            best[q[first]] = vi[first]
        return best

    def _incident(self):
        nv = self.n_vertices
        flat = self.F.ravel()
        tri = np.repeat(np.arange(len(self.F)), 3)
        o = np.argsort(flat, kind="stable")
        sf, st = flat[o], tri[o]
        start = np.searchsorted(sf, np.arange(nv), side="left")
        end = np.searchsorted(sf, np.arange(nv), side="right")
        cnt = end - start
        kmax = int(cnt.max())
        table = np.full((nv, kmax), -1, dtype=np.int64)
        col = np.arange(len(sf)) - start[sf]
        table[sf, col] = st
        self._table = table

    def nearest_vertex(self, points, guess=None, steps: int = 12):
        """Index of the patch vertex nearest to each point.

        Without ``guess``: a 3x3x3 block of grid cells (brute force for a point that found
        nothing). With ``guess`` (the answer for a nearby point, e.g. the previous ring of a
        march): a greedy walk over the vertex rings of the mesh, a handful of table lookups."""
        P = np.asarray(points, dtype=np.float64)
        n = len(P)
        if guess is not None:
            g = np.asarray(guess, dtype=np.int64).copy()
            for _ in range(steps):
                tri = self._table[g]                                    # (n, K)
                cv = self.F[np.where(tri >= 0, tri, 0)].reshape(n, -1)  # (n, 3K)
                valid = np.repeat(tri >= 0, 3, axis=1)
                d = ((self.V[cv] - P[:, None, :]) ** 2).sum(axis=2)
                d = np.where(valid, d, np.inf)
                cur = ((self.V[g] - P) ** 2).sum(axis=1)
                j = d.argmin(axis=1)
                better = d[np.arange(n), j] < cur - 1e-18
                if not better.any():
                    break
                g[better] = cv[np.arange(n), j][better]
            return g
        best = np.full(n, -1, dtype=np.int64)
        todo = np.arange(n)
        for k, level in enumerate(self._levels):
            if not len(todo):
                break
            found = self._block_search(P[todo], level)
            if k and (found >= 0).any():                           # a sampled grid: walk to the real nearest
                hit = found >= 0
                found[hit] = self.nearest_vertex(P[todo][hit], guess=found[hit], steps=80)
            best[todo] = found
            todo = todo[found < 0]
        for s0 in range(0, len(todo), 256):
            m = todo[s0:s0 + 256]
            d = ((P[m][:, None, :] - self.V[None, :, :]) ** 2).sum(axis=2)
            best[m] = d.argmin(axis=1)
        return best

    def closest(self, points, guess=None):
        """``(point, normal, signed distance, nearest vertex)`` of the surface nearest to each
        point (positive outside); ``guess`` warm-starts the vertex search (see above)."""
        P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        v = self.nearest_vertex(P, guess)
        cand = self._table[v]                                   # (n, K) triangle ids or -1
        ok = cand >= 0
        tri = np.where(ok, cand, 0)
        n, K = cand.shape
        a = self.V[self.F[tri, 0]].reshape(-1, 3)
        b = self.V[self.F[tri, 1]].reshape(-1, 3)
        c = self.V[self.F[tri, 2]].reshape(-1, 3)
        pp = np.repeat(P, K, axis=0)
        pt, bary = closest_on_triangles(pp, a, b, c)
        d2 = ((pt - pp) ** 2).sum(axis=1).reshape(n, K)
        d2 = np.where(ok, d2, np.inf)
        j = d2.argmin(axis=1)
        rows = np.arange(n)
        sel = rows * K + j
        point = pt[sel]
        bw = bary[sel]
        corner = self.F[tri[rows, j]]
        normal = _unit((self.N[corner] * bw[:, :, None]).sum(axis=1))
        signed = ((P - point) * normal).sum(axis=1)
        return point, normal, signed, v


def skin_from_cage(base, shaped_verts, levels: int):
    """``(vertices, triangles)`` of the body the character is meshed with: the cage of
    ``shaped_verts`` Catmull-Clark subdivided ``levels`` times (``portrait.character_shells`` /
    ``printprep.character_print_shells`` build exactly this), in the MakeHuman frame."""
    from ..core.subdiv import quads_to_mesh, subdivide_quads
    cv, cq = base.body_cage(shaped_verts)
    v, q = subdivide_quads(cv, cq, max(int(levels), 0))
    m = quads_to_mesh(v, q)
    return m.vertices, m.faces


# --------------------------------------------------------------------------
# rays against the appendage's wall


def ray_mesh_first_hit(origins, dirs, tris, eps=1e-10):
    """Distance along each ray (``origins``, unit ``dirs``: (n, 3)) to the FIRST triangle of
    ``tris`` (m, 3, 3) it hits in front of the origin, ``inf`` if none. Moller-Trumbore over all
    pairs (the wall near a root is a few hundred triangles)."""
    n, m = len(origins), len(tris)
    if not n or not m:
        return np.full(n, np.inf)
    a = tris[:, 0]
    e1 = tris[:, 1] - a
    e2 = tris[:, 2] - a
    d = dirs[:, None, :]
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
    hx = dy * e2[None, :, 2] - dz * e2[None, :, 1]
    hy = dz * e2[None, :, 0] - dx * e2[None, :, 2]
    hz = dx * e2[None, :, 1] - dy * e2[None, :, 0]
    det = e1[None, :, 0] * hx + e1[None, :, 1] * hy + e1[None, :, 2] * hz
    ok = np.abs(det) > eps
    inv = 1.0 / np.where(ok, det, 1.0)
    sx = origins[:, None, 0] - a[None, :, 0]
    sy = origins[:, None, 1] - a[None, :, 1]
    sz = origins[:, None, 2] - a[None, :, 2]
    u = inv * (sx * hx + sy * hy + sz * hz)
    qx = sy * e1[None, :, 2] - sz * e1[None, :, 1]
    qy = sz * e1[None, :, 0] - sx * e1[None, :, 2]
    qz = sx * e1[None, :, 1] - sy * e1[None, :, 0]
    v = inv * (dx * qx + dy * qy + dz * qz)
    t = inv * (e2[None, :, 0] * qx + e2[None, :, 1] * qy + e2[None, :, 2] * qz)
    hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9) & (t > 1e-9)
    t = np.where(hit, t, np.inf)
    return t.min(axis=1)


# --------------------------------------------------------------------------
# the axis of the appendage: arclength, tangents, a transported frame


class Axis:
    """A polyline centreline with a rotation-minimising frame (``u`` along the projected
    ``up``), evaluated at arclength."""

    def __init__(self, points, radii, up):
        P = np.asarray(points, dtype=np.float64)
        self.P = P
        self.R = np.asarray(radii, dtype=np.float64)
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        T = np.zeros_like(P)
        T[1:-1] = _unit(P[2:] - P[:-2])
        T[0] = _unit(P[1] - P[0])
        T[-1] = _unit(P[-1] - P[-2])
        self.T = T
        ref = np.asarray(up if up is not None else (0.0, 1.0, 0.0), dtype=np.float64)
        ref = ref / max(np.linalg.norm(ref), _EPS)
        # the transported frame starts where ``ref`` is best conditioned
        k = int(np.argmin(np.abs(T @ ref)))
        u = np.zeros_like(P)
        u0 = ref - (ref @ T[k]) * T[k]
        if np.linalg.norm(u0) < 0.2:                      # the axis runs along ``ref``: use lateral
            alt = np.array([1.0, 0.0, 0.0])
            u0 = alt - (alt @ T[k]) * T[k]
        u[k] = u0 / np.linalg.norm(u0)
        for i in range(k + 1, len(P)):
            w = u[i - 1] - (u[i - 1] @ T[i]) * T[i]
            u[i] = w / max(np.linalg.norm(w), _EPS)
        for i in range(k - 1, -1, -1):
            w = u[i + 1] - (u[i + 1] @ T[i]) * T[i]
            u[i] = w / max(np.linalg.norm(w), _EPS)
        self.U = u
        self.V = _cross(T, u)

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def at(self, s):
        """``(point, tangent, u, v, radius)`` at arclength(s) ``s`` (clipped)."""
        s = np.clip(np.asarray(s, dtype=np.float64), 0.0, self.s[-1])
        i = np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(self.s) - 2)
        t = ((s - self.s[i]) / np.maximum(self.s[i + 1] - self.s[i], _EPS))[..., None]
        pt = self.P[i] + t * (self.P[i + 1] - self.P[i])
        rad = self.R[i] + t[..., 0] * (self.R[i + 1] - self.R[i])
        T = _unit(self.T[i] * (1 - t) + self.T[i + 1] * t)
        u = self.U[i] * (1 - t) + self.U[i + 1] * t
        u = _unit(u - (u * T).sum(-1, keepdims=True) * T)
        v = _cross(T, u)
        return pt, T, u, v, rad


# --------------------------------------------------------------------------
# the wall of the appendage, sampled along generators


class Wall:
    """The appendage's own surface seen from its axis: ``point(theta, s)`` is where the ray from
    the axis at arclength ``s`` in direction ``theta`` (angle from the frame's ``u``) first hits
    the real mesh (so an elliptical, flared, faceted wall is followed exactly)."""

    def __init__(self, axis: Axis, mesh: Mesh):
        self.axis = axis
        self.tris = mesh.triangles()
        cen = self.tris.mean(axis=1)
        # the axial position of each triangle: the arclength of the nearest axis vertex pair
        d = np.linalg.norm(cen[:, None, :] - axis.P[None, :, :], axis=2)
        j = d.argmin(axis=1)
        self.tri_s = axis.s[j]
        self.edge = float(np.max(np.linalg.norm(self.tris[:, 1] - self.tris[:, 0], axis=1)))

    def radius(self, theta, s):
        """``(distance, direction, axis point)`` for angles ``theta`` (n,) at arclengths ``s`` (n,)."""
        c, T, u, v, rad = self.axis.at(s)
        e = np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
        dist = np.full(len(theta), np.inf)
        # group the rays by station so each group only meets the triangles of its slab
        order = np.argsort(s)
        so = s[order]
        i = 0
        n = len(s)
        slab = max(0.35, 2.5 * self.edge)
        while i < n:
            j = int(np.searchsorted(so, so[i] + 0.3, side="right"))
            j = max(j, i + 1)
            idx = order[i:j]
            lo, hi = float(so[i]), float(so[j - 1])
            near = (self.tri_s > lo - slab) & (self.tri_s < hi + slab)
            if near.any():
                dist[idx] = ray_mesh_first_hit(c[idx], e[idx], self.tris[near])
            i = j
        bad = ~np.isfinite(dist)
        dist = np.where(bad, rad * 1.0, dist)
        return dist, e, c

    def point(self, theta, s):
        dist, e, c = self.radius(theta, s)
        return c + dist[:, None] * e

    def facet_correction(self, theta, s_ref, keep: int = 6):
        """``delta(theta)``: what a low pass over the ring at the station ``s_ref`` (harmonics ``0 ..
        keep`` of a double-density sampling) removes from the wall's radius.  The 16-gon of a loft's
        rings is a polygon whose facet normals differ from the smooth shading normals by up to 11
        degrees, and a collar that follows the facets shows them as a scalloped seam; the ellipse and
        the lobing of the wall (harmonics 2 and 4) are kept.  It is measured at ONE station and applied at
        the stations of the exit curve (which differ by a radius or more between the back and the
        underside: a low pass over radii taken at different stations rings at the steps)."""
        n = len(theta)
        th2 = np.arange(2 * n) * (math.pi / n)
        dist, _, _ = self.radius(th2, np.full(2 * n, float(s_ref)))
        f = np.fft.rfft(dist)
        f[keep + 1:] = 0.0
        low = np.fft.irfft(f, len(dist))
        return (low - dist)[::2]

    def smooth_points(self, theta, s, delta):
        """The wall points at ``(theta, s)`` with the facet correction ``delta`` applied radially."""
        dist, e, c = self.radius(theta, s)
        return c + (dist + delta + WALL_PAD)[:, None] * e


# --------------------------------------------------------------------------
# the collar


def _profile_h(x, H, k):
    """``H (1 - sqrt(1 - x^k))`` for ``x`` in 0..1 (1 at the wall, 0 at the rim)."""
    x = np.clip(x, 0.0, 1.0)
    return H * (1.0 - np.sqrt(np.maximum(1.0 - x ** k, 0.0)))


def _profile_stations(rings: int, ratio: float, k: float):
    """Fractions ``u = rho / W`` of the visible profile from the wall (0) to the rim (1) for the
    profile ``h = H (1 - sqrt(1 - (1-u)^k))`` with ``H / W = ratio``, spaced half equally in ARC
    LENGTH and half equally in TURNING ANGLE of the tangent.  Equal arc length alone leaves the
    first chord next to the wall 18 degrees off the wall (the curvature of the profile is
    concentrated there) and the renderer's area weighted vertex normal at the join averages that
    chord in; equal turning alone makes a few very long chords on the nearly flat part near the rim,
    which cut the curved skin under them."""
    u = np.linspace(0.0, 1.0, 6001) ** 2
    h = _profile_h(1.0 - u, ratio, k)
    du, dh = np.diff(u), np.diff(h)
    seg = np.hypot(du, dh)
    ang = np.arctan2(du, -dh)                                  # 0 along the wall, pi/2 along the skin
    ang = np.maximum.accumulate(np.clip(ang, 0.0, math.pi / 2))
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    turn = np.concatenate([[0.0], ang])
    measure = 0.5 * arc / arc[-1] + 0.5 * turn / max(turn[-1], _EPS)
    # two extra, very short chords next to the wall: the vertex normal AT the join is the average of
    # the faces on both sides, and the first chord must be nearly parallel to the wall
    want = np.sort(np.concatenate([np.linspace(0.0, 1.0, rings + 1),
                                   [EXTRA_NEAR_WALL / rings]]))
    return np.interp(want, measure, u)


def _ring_smooth(v, rounds: int = 2):
    """Circular 1-2-1 smoothing of a per-angle array."""
    v = np.asarray(v, dtype=np.float64)
    for _ in range(rounds):
        v = 0.25 * np.roll(v, 1) + 0.5 * v + 0.25 * np.roll(v, -1)
    return v


def _march(skin: SkinPatch, start, direction, steps, vertex=None):
    """Walk over the skin from ``start`` (n, 3) along ``direction`` (n, 3, a tangent): for every
    column of ``steps`` (n, m) take that much of a step and project back onto the surface.
    Returns ``(points (m, n, 3), normals (m, n, 3), directions (m, n, 3))`` of the m new stations."""
    P = np.asarray(start, dtype=np.float64)
    g = _unit(direction)
    pts, nrm, dirs = [], [], []
    for i in range(steps.shape[1]):
        q = P + steps[:, i:i + 1] * g
        P, N, _, vertex = skin.closest(q, guess=vertex)
        g = _unit(g - (g * N).sum(-1, keepdims=True) * N)
        pts.append(P)
        nrm.append(N)
        dirs.append(g)
    return np.array(pts), np.array(nrm), np.array(dirs)


def build_collar(skin: SkinPatch, points, radii, wall: Mesh, profile: CollarProfile,
                 diag: dict | None = None):
    """The collar shell (a closed :class:`Mesh`) of an appendage whose centreline is
    ``points`` / ``radii`` (root first, the first points inside the body) and whose real surface
    is ``wall``, on the ``skin`` (a :class:`SkinPatch` of the body as it is meshed), or ``None``
    when the appendage does not come out of the skin (nothing to blend). ``diag`` collects the
    intermediate numbers (the exit curve, the widths, the rings) for the tests and the report."""
    if not profile.active:
        return None
    axis = Axis(points, radii, profile.up)
    n = profile.segments
    theta = np.arange(n) * (2.0 * math.pi / n)

    away = np.degrees(np.abs((theta + math.pi) % (2.0 * math.pi) - math.pi))      # angle from the open side

    # -- where the axis comes out of the skin --------------------------------------------------
    ns = int(np.clip(axis.length / 0.03, 24, 240))
    s_ax = np.linspace(0.0, axis.length, ns)
    pa, _, _, _, ra = axis.at(s_ax)
    _, _, sd_ax, _ = skin.closest(pa)
    inside = np.nonzero(sd_ax < 0.0)[0]
    if not len(inside):
        return None
    after = np.nonzero((sd_ax >= 0.0) & (np.arange(ns) > inside[0]))[0]
    if not len(after):
        return None
    s_e = float(s_ax[after[0]])
    r_root = float(ra[after[0]])

    # -- the exit curve: where each generator of the wall pierces the skin --------------------------
    wallf = Wall(axis, wall)
    K = 8
    s_lo = max(0.0, s_e - 2.0 * r_root)
    s_hi = min(axis.length, s_e + 3.0 * r_root)
    st = np.linspace(s_lo, s_hi, K)
    n_scan = max(n // 3, 16)                  # a coarse scan around the ring, the secant steps below do the rest
    th_scan = np.arange(n_scan) * (2.0 * math.pi / n_scan)
    wp = wallf.point(np.repeat(th_scan, K), np.tile(st, n_scan))
    _, _, sdw, _ = skin.closest(wp)
    sdw = sdw.reshape(n_scan, K)
    outside = sdw >= 0.0
    s_scan = np.full(n_scan, np.nan)
    for j in range(n_scan):
        for k in range(1, K):
            if outside[j, k] and not outside[j, k - 1]:
                a_, b_ = sdw[j, k - 1], sdw[j, k]
                f = a_ / (a_ - b_) if a_ != b_ else 0.5
                s_scan[j] = st[k - 1] + f * (st[k] - st[k - 1])
                break
    ok = np.isfinite(s_scan)
    if not ok.any():
        return None
    if not ok.all():                         # no crossing in range: borrow the nearest crossing
        idx = np.arange(n_scan)
        for j in np.nonzero(~ok)[0]:
            d = np.minimum((idx[ok] - j) % n_scan, (j - idx[ok]) % n_scan)
            s_scan[j] = s_scan[ok][d.argmin()]
    s_x = np.interp(theta, np.concatenate([th_scan, [2.0 * math.pi]]),
                    np.concatenate([s_scan, s_scan[:1]]))
    for _ in range(3):                       # a few secant steps on the real wall
        wp = wallf.point(theta, s_x)
        _, _, sdx, _ = skin.closest(wp)
        s_x = np.clip(s_x - 0.8 * sdx, 0.0, axis.length)
    delta = wallf.facet_correction(theta, s_e)
    X = wallf.smooth_points(theta, s_x, delta)
    Xs, Nx, sdx, v_start = skin.closest(X)
    _, Tx, ux, vx, _ = axis.at(s_x)
    e_x = np.cos(theta)[:, None] * ux + np.sin(theta)[:, None] * vx            # outward, in the section

    # -- width and height per angle ------------------------------------------------------------------
    r0 = r_root * profile.strength
    ratio = profile.ventral / max(profile.width, _EPS)
    w_theta = 1.0 - (1.0 - ratio) * _smoothstep((away - profile.open_deg) / (profile.fold_deg - profile.open_deg))
    g0 = _unit(e_x - (e_x * Nx).sum(-1, keepdims=True) * Nx)                      # outward along the skin
    u_vis = _profile_stations(profile.rings, 0.35 * profile.height / max(profile.width, _EPS), profile.falloff)
    skirt_u = np.asarray(SKIRT_U)
    wide = profile.width * r0

    sink0 = profile.sink * r0

    def sink_of(W):
        # a thin collar (the fold side: the skin is a crease there, two walls close together) is sunk
        # little: a pad 0.1 deep under a V-shaped cleft leaves the body through the other wall
        return sink0 * np.maximum(SINK_FLOOR, (W / wide) ** 0.7)

    def run(W):
        eps = np.maximum(SKIRT_WIDTH * W, 1.5 * sink_of(W))      # the dive is gentle even where the collar is thin
        steps = np.concatenate([
            W[:, None] * np.diff(u_vis)[None, :],
            eps[:, None] * (np.diff(np.concatenate([[0.0], skirt_u])))[None, :]], axis=1)
        return _march(skin, Xs, g0, steps, v_start)

    # The first march, at the width the side asks for, says where the surface goes: (1) its direction
    # at the rim against the wall's generator is the lean of the wall over the skin there (where it
    # leans over the skin, the underside of a lifted tail, there is no room for a fillet), (2) a crease
    # (the natal cleft, the flank: consecutive normals turning more than ``CREASE_STEP_DEG``) must not
    # be marched over.  The width is cut to that, smoothed around the ring (no steps in the rim) and
    # the march is redone.
    # where the exit curve could not be fitted to the skin (a closed fold of flesh next to the root: the
    # skin is 0.02 or more off the wall there, the signed distance cannot be trusted) there is no collar
    fit_ok = np.abs(sdx) <= FIT_TOL
    fit_ok = fit_ok & np.roll(fit_ok, 1) & np.roll(fit_ok, -1)
    W = np.maximum(wide * w_theta * np.where(fit_ok, 1.0, 0.0), MIN_WIDTH)
    nrings = len(u_vis) - 1
    u_an = u_vis[::2] if nrings % 2 == 0 else u_vis                              # the analysis marches every second station
    steps_an = W[:, None] * np.diff(u_an)[None, :]
    P_a, N_a, G_a = _march(skin, Xs, g0, steps_an, v_start)
    na = len(u_an) - 1
    beta = np.degrees(np.arccos(np.clip((Tx * G_a[na - 1]).sum(-1), -1.0, 1.0)))   # wall vs outward skin
    lean = _smoothstep((beta - LEAN_FROM_DEG) / (LEAN_FULL_DEG - LEAN_FROM_DEG))
    step = np.degrees(np.arccos(np.clip((N_a[1:na] * N_a[: na - 1]).sum(-1), -1.0, 1.0)))
    chord = P_a[1:na] - P_a[: na - 1]                                             # a step that climbs or drops steeply
    drop = np.abs((chord * N_a[: na - 1]).sum(-1)) / np.maximum(np.linalg.norm(chord, axis=-1), _EPS)
    bad = (step > CREASE_STEP_DEG) | (drop > math.sin(math.radians(CREASE_DROP_DEG)))   # (stations - 1, n)
    first = np.where(bad.any(axis=0), bad.argmax(axis=0), na)
    reach = u_an[np.minimum(first + 1, na)]                                       # fraction of the width before the crease
    W = np.maximum(W * lean * np.minimum(1.0, reach / 0.85), MIN_WIDTH)
    W = np.where(fit_ok, _ring_smooth(W, RIM_SMOOTHING), MIN_WIDTH)             # no collar where the fit failed, not even a smoothed one

    def construct(W):
        H = np.maximum(profile.height * r0 * (W / wide) ** 0.7, 0.3 * MIN_WIDTH)
        P_m, N_m, G_m = run(W)
        # next to the wall the marched points jitter by a millimetre-ish (the facets of the subdivided
        # skin) which is as much as the distance between stations there: the stations next to the wall
        # lie on the straight chord from the wall to the rim, blended into the marched points
        rho = W[None, :] * u_vis[1:, None]                                            # (rings, n)
        chord = Xs[None] + (rho / W[None, :])[..., None] * (P_m[nrings - 1] - Xs)[None]
        blend = _smoothstep(rho / (0.1 * W[None, :] + 0.03))[..., None]
        P_m = P_m.copy()
        P_m[:nrings] = chord + blend * (P_m[:nrings] - chord)
        # the exit point on the wall and the nearest skin point differ by the residual of the secant steps
        # (a few thousandths): the sheet starts exactly ON the wall and blends into the skin by the rim
        shift = X - Xs
        fade = ((1.0 - u_vis[1:]) ** 3)[:, None, None]
        P_m[:nrings] += fade * shift[None]
        # the same for the skirt past the rim, where the skin may be a crease (the apex of the cleft: the
        # normals flip within a step and the top and bottom of the skirt would cross): close to the rim the
        # skirt is the straight continuation, blended into the marched points
        eps = np.maximum(SKIRT_WIDTH * W, 1.5 * sink_of(W))
        rho_s = eps[None, :] * skirt_u[:, None]                                       # (skirt, n) distance past the rim
        straight = P_m[nrings - 1][None] + rho_s[..., None] * G_m[nrings - 1][None]
        blend_s = _smoothstep(rho_s / 0.06)[..., None]
        P_m[nrings:] = straight + blend_s * (P_m[nrings:] - straight)
        P = np.concatenate([X[None], P_m])                                            # stations 0 .. rings + skirt
        N = np.concatenate([Nx[None], N_m])
        n_vis = len(u_vis)
        sink = sink_of(W)
        # the wall point H up the generator: the profile ends there, tangent to the wall
        E = wallf.smooth_points(theta, np.clip(s_x + H, 0.0, axis.length), delta)
        Hv = np.linalg.norm(E - X, axis=1)
        D_end = _unit(E - X)

        # the underside is the skin pushed down by the sink; where the collar is a hair the skin may be a crease
        # (the apex of the cleft: the normals flip within a step) and an offset along them folds, so the
        # underside there is the sheet translated along the normal at the rim
        thin = 1.0 - _smoothstep(W / (0.1 * wide))
        Nb = _unit((1.0 - thin)[None, :, None] * N + thin[None, :, None] * N[n_vis - 1][None])

        # -- the sheet ----------------------------------------------------------------------------------------
        x = 1.0 - u_vis                                                               # 1 at the wall, 0 at the rim
        top = []
        for i in range(n_vis):
            xi = x[i]
            h = profile.lift * (1.0 - xi * xi) + _profile_h(xi, Hv, profile.falloff)
            if profile.ramp > 0.0:                    # leaves the skin at a small angle: clear of its facets
                L = np.clip(0.45 * W, 0.05, 0.5)
                t = np.clip(xi * W / L, 0.0, 1.0)
                h = h + profile.ramp * L * t * (1.0 - _smoothstep(t))
            w = _smoothstep(xi)
            D = _unit((1.0 - w) * N[i] + w * D_end)
            top.append(P[i] + h[:, None] * D)
        top[0] = E.copy()
        for k, sg in enumerate(skirt_u):
            i = n_vis + k
            h = profile.lift - (profile.lift + sink) * _smoothstep(sg)
            top.append(P[i] + h[:, None] * Nb[i])
        top = np.array(top)                                                           # (stations, n, 3)
        bottom = P - sink[None, :, None] * Nb                                         # same stations, under the skin
        # the wall tangent beyond E, then curving into the appendage
        e_in = -_unit(e_x - (e_x * D_end).sum(-1, keepdims=True) * D_end)
        rc = 0.3 * r0
        arcs = np.array([E + rc * (math.sin(math.radians(a)) * D_end
                                   + (1.0 - math.cos(math.radians(a))) * e_in * 1.0)
                         for a in ARC_ANGLES])
        if diag is not None:
            diag.update(s_exit=s_e, r_root=r_root, X=X, s_x=s_x, sd_exit=sdx, W=W, H=Hv, beta=beta, lean=lean,
                        E=E, P=P, N=N, top=top, bottom=bottom, arcs=arcs, theta=theta, axis=axis)
        return W, Hv, E, P, N, top, bottom, arcs, n_vis

    # the rim must continue the skin: where the sheet near the rim is not tangent to the surface (a fold in
    # the flesh, a fat crease the march cannot see through) the width shrinks there, smoothed, and the sheet
    # is built again (twice at most)
    for it in range(RIM_ITERATIONS + 1):
        W, Hv, E, P, N, top, bottom, arcs, n_vis = construct(W)
        j, jn = np.arange(n), (np.arange(n) + 1) % n
        fr = np.cross(top[n_vis - 1][jn] - top[n_vis - 1][j], top[n_vis - 2][j] - top[n_vis - 1][j])
        fr = _unit(fr)
        mismatch = np.degrees(np.arccos(np.clip(np.abs((fr * N[n_vis - 1]).sum(-1)), 0.0, 1.0)))
        mismatch = np.where(W > 0.04, np.maximum(mismatch, np.roll(mismatch, 1)), 0.0)    # a hair-thin collar cannot be seen
        if it == RIM_ITERATIONS or not (mismatch > RIM_TOL_DEG).any():
            break
        keep = np.clip(1.0 - (mismatch - RIM_TOL_DEG) / 20.0, 0.25, 1.0)
        W = np.where(fit_ok, np.maximum(_ring_smooth(W * _ring_smooth(keep, 2), 2), MIN_WIDTH), MIN_WIDTH)
    return assemble_shell(top, bottom, arcs)


def assemble_shell(top, bottom, arcs):
    """Close the rings into one watertight triangle mesh, outward oriented.

    ``top`` (S, n, 3): the visible sheet from the wall (station 0) outward over the rim to the
    end of the skirt (station S-1); ``bottom`` (S, n, 3): the same stations pushed under the
    skin (only 0 .. S-2 are used: the last equals ``top[-1]``); ``arcs`` (A, n, 3): the sheet
    carried on into the appendage past the wall. Fans close both ends at one centre vertex."""
    S, n, _ = top.shape
    A = len(arcs)
    rings = []                                  # from the inside of the appendage, over the top, to the underside
    c_top = arcs[-1].mean(axis=0)
    c_bot = bottom[0].mean(axis=0)
    for k in range(A - 1, -1, -1):
        rings.append(arcs[k])
    for i in range(S):
        rings.append(top[i])
    for i in range(S - 2, -1, -1):
        rings.append(bottom[i])
    mid = 0.5 * (bottom[0] + c_bot[None])
    rings.append(mid)
    R = np.array(rings)                         # (rings, n, 3)
    nr = len(R)
    verts = np.concatenate([R.reshape(-1, 3), c_top[None], c_bot[None]])
    ct, cb = nr * n, nr * n + 1
    j = np.arange(n)
    jn = (j + 1) % n
    faces = []
    for r in range(nr - 1):
        a = r * n + j
        b = r * n + jn
        c = (r + 1) * n + jn
        d = (r + 1) * n + j
        faces.append(np.stack([a, b, c], axis=1))
        faces.append(np.stack([a, c, d], axis=1))
    faces.append(np.stack([np.full(n, ct), jn, j], axis=1))
    last = (nr - 1) * n + j
    faces.append(np.stack([np.full(n, cb), last, (nr - 1) * n + jn], axis=1))
    F = np.concatenate(faces)
    m = Mesh(verts, F)
    if m.volume() < 0:
        m = Mesh(verts, F[:, [0, 2, 1]])
    return m


# --------------------------------------------------------------------------
# the pipeline seam

_SKIN_CACHE: "OrderedDict" = OrderedDict()           # (vertices digest, levels) -> (vertices, triangles)
_SKIN_LOCK = threading.Lock()
SKIN_CACHE_ENTRIES = 2


def verts_digest(verts) -> str:
    return hashlib.blake2b(np.ascontiguousarray(verts, dtype=np.float64).tobytes(),
                           digest_size=8).hexdigest()


def shaped_skin(base, shaped_verts, levels: int):
    """``(vertices, triangles)`` of the shaped body subdivided ``levels`` times (the surface the
    collar sits on), cached by content: one subdivision serves every appendage of the body."""
    key = (verts_digest(shaped_verts), int(levels))
    with _SKIN_LOCK:
        if key in _SKIN_CACHE:
            _SKIN_CACHE.move_to_end(key)
            return _SKIN_CACHE[key]
    skin = skin_from_cage(base, shaped_verts, levels)
    with _SKIN_LOCK:
        _SKIN_CACHE[key] = skin
        while len(_SKIN_CACHE) > SKIN_CACHE_ENTRIES:
            _SKIN_CACHE.popitem(last=False)
    return skin


def skin_from_shell(mesh):
    """``(vertices, triangles)`` in the MakeHuman frame from the body mesh a pipeline has already
    built in the Z-up frame (``portrait.character_shells`` / ``printprep.character_print_shells``):
    the same surface, without subdividing the cage a second time."""
    from .portrait import MH_TO_Z_UP
    R, t = MH_TO_Z_UP[:3, :3], MH_TO_Z_UP[:3, 3]
    return (mesh.vertices - t) @ R, mesh.faces


def collar_material(placed, profile: CollarProfile):
    """The material the collar renders with: the body's skin or the appendage's own."""
    if profile.material == "appendage":
        return placed.material
    from . import shading as sh
    return sh.SKIN


def collar_pieces(base, shaped_verts, placed, controls=None, *, levels: int = 2, skin=None,
                  profile: CollarProfile | None = None, diag: dict | None = None) -> list:
    """``[(name, Mesh, material)]``: the collar of one placed appendage (``[]``: none), in the
    MakeHuman frame like ``Placed.mesh()``.

    ``shaped_verts`` are the body's vertices AFTER :func:`~.junction.apply_junctions` (the skin
    the collar sits on); ``levels`` is the subdivision the body is meshed with (the collar is
    fitted to THAT surface: a collar fitted to level 2 stands off a level-1 body by the
    Catmull-Clark shrink); ``skin`` is that body mesh as ``(vertices, triangles)`` in the same
    frame if the caller has it already (see :func:`skin_from_shell`) or a :class:`SkinPatch`;
    ``controls`` the pose the appendage is drawn in (default: its own).  ``profile`` overrides
    ``placed.kind.collar_profile``.  At ``levels`` 0 (the cage itself) there is no collar (``MIN_LEVEL``)."""
    kind = placed.kind
    prof = profile if profile is not None else kind.collar_profile(placed)
    if prof is None or not prof.active or int(levels) < MIN_LEVEL:
        return []
    ctl = placed.controls if controls is None else controls
    axis = kind.collar_axis(placed, ctl)
    if axis is None:
        return []
    points, radii = axis
    wall = kind.collar_wall(placed, ctl)
    if isinstance(skin, SkinPatch):
        patch = skin
    else:
        if skin is None:
            skin = shaped_skin(base, shaped_verts, levels)
        r_max = float(np.max(radii))
        reach = (prof.width + 2.4) * r_max + 0.5
        centre = np.asarray(placed.anchor, dtype=np.float64)
        patch = SkinPatch(skin[0], skin[1], centre, reach)
    mesh = build_collar(patch, points, radii, wall, prof, diag)
    if mesh is None:
        return []
    return [(f"{placed.name}/collar", mesh, collar_material(placed, prof))]


def collar_shells(base, shaped_verts, placed, *, levels: int = 2, skin=None, controls=None,
                  enabled: bool = True) -> list:
    """The collars of every appendage of ``placed`` that has one, as ``[(placed, name, Mesh,
    material)]`` (the three places that mesh the body use this: the picture, the STL and the
    lab).  ``controls`` is ``placed -> controls`` (the pose each appendage is drawn in; default
    its own).  A collar that cannot be built is skipped with an :class:`~.kinds.AppendageWarning`
    (the picture and the print still come out); ``enabled=False`` (a document's
    ``anatomy["collar"] = false`` or the junction off) returns ``[]``."""
    out = []
    if not enabled:
        return out
    for p in placed or ():
        if p.kind.collar_profile(p) is None:
            continue
        try:
            for name, mesh, mat in collar_pieces(
                    base, shaped_verts, p, None if controls is None else controls(p),
                    levels=levels, skin=skin):
                out.append((p, name, mesh, mat))
        except (ValueError, KeyError, IndexError, FloatingPointError, np.linalg.LinAlgError) as exc:
            from .kinds import AppendageWarning
            warnings.warn(f"appendage {p.name!r}: no root collar ({exc})", AppendageWarning,
                          stacklevel=2)
    return out


# --------------------------------------------------------------------------
# measuring a collar (tests, tools/collar_report.py, docs/collar.md)


def count_crossings(vertices, faces, other_vertices=None, other_faces=None) -> int:
    """How many pairs of triangles cross each other: within one mesh, or (with ``other_*``) between
    two. Triangles that share a vertex are not counted (neighbours), nor are touching or coplanar ones
    (the open-segment test of ``junction_checks.pairs_cross``)."""
    from . import junction_checks as JC
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    if other_vertices is None:
        allv, allf = V, F
        rows = np.arange(len(F))
        others = rows
    else:
        W = np.asarray(other_vertices, dtype=np.float64)
        G = np.asarray(other_faces, dtype=np.int64)
        allv = np.concatenate([V, W])
        allf = np.concatenate([F, G + len(V)])
        rows = np.arange(len(F))
        others = np.arange(len(F), len(allf))
    P = allv[allf]
    pairs = JC.overlapping_pairs(P.min(axis=1), P.max(axis=1), rows, others)
    if not len(pairs):
        return 0
    if other_vertices is None:
        pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    share = (allf[pairs[:, 0]][:, :, None] == allf[pairs[:, 1]][:, None, :]).any(axis=(1, 2))
    pairs = pairs[~share]
    if not len(pairs):
        return 0
    return int(JC.pairs_cross(allv, allf, pairs).sum())


def _angles(a, b):
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1.0, 1.0)))


def _stats(x, prefix) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {f"{prefix}_p50": float(np.percentile(x, 50)), f"{prefix}_p95": float(np.percentile(x, 95)),
            f"{prefix}_max": float(x.max())}


def _shaded_normals(points, mesh: Mesh, vertex_normals) -> np.ndarray:
    """The shading normal of ``mesh`` (barycentric over its vertex normals, as the renderer interpolates) at the
    point of its surface nearest to each of ``points``."""
    tris = mesh.triangles()
    cen = tris.mean(axis=1)
    out = np.zeros((len(points), 3))
    for i, q in enumerate(points):
        cand = np.argsort(((cen - q) ** 2).sum(-1))[:6]
        pt, bary = closest_on_triangles(np.repeat(q[None], len(cand), 0), tris[cand, 0], tris[cand, 1],
                                        tris[cand, 2])
        j = int(((pt - q) ** 2).sum(-1).argmin())
        v = (vertex_normals[mesh.faces[cand[j]]] * bary[j][:, None]).sum(axis=0)
        out[i] = v / max(np.linalg.norm(v), _EPS)
    return out


def join_report(base, shaped_verts, placed, controls=None, *, levels: int = 2, skin=None,
                profile: CollarProfile | None = None) -> dict:
    """The numbers of one collar (``{}`` if the appendage has none):

    * ``rim_*``: the angle between the RENDERER's vertex normal of the collar at its rim (the normals the
      shading uses: area weighted over the collar's own faces) and the skin's smooth normal there, degrees
      (0 = the surface continues the skin without a visible edge);
    * ``wall_*``: the same where the collar meets the appendage, against the appendage's vertex normal
      field (the shaded normal of the appendage's shell at that point);
    * ``before_*``: for comparison, the angle between the skin and the wall where they meet WITHOUT the
      collar (what the junction alone leaves: 90 = the skin runs into the wall, 0 = tangent);
    * ``above_skin_min``: the lowest the visible sheet gets over the skin (units; >= 0 means it never
      sinks below it), ``crease_p99``: the 99th percentile of the dihedral between neighbouring faces of
      the visible sheet (the creases the collar itself adds);
    * ``self_crossings`` (the shell with itself), ``below_skin_vertices`` (vertices of the visible sheet
      more than ``SKIN_TOL`` under the skin), ``watertight`` / ``outward``, ``faces``, ``width_*`` / ``height_*`` (units).
    """
    from . import shading as sh
    diag: dict = {}
    t0 = __import__("time").perf_counter()
    pieces = collar_pieces(base, shaped_verts, placed, controls, levels=levels, skin=skin,
                           profile=profile, diag=diag)
    seconds = __import__("time").perf_counter() - t0
    if not pieces:
        return {}
    mesh = pieces[0][1]
    prof = profile if profile is not None else placed.kind.collar_profile(placed)
    n = prof.segments
    ctl = placed.controls if controls is None else controls
    wall = placed.kind.collar_wall(placed, ctl)      # the surface the collar meets (a wing: its arm piece)
    top, arcs, N = diag["top"], diag["arcs"], diag["N"]
    A = len(arcs)
    n_vis = N.shape[0] - len(SKIRT_U)                     # stations of the visible sheet
    vn = sh.vertex_normals(mesh.vertices, mesh.faces)
    rim = vn[(A + n_vis - 1) * n:(A + n_vis) * n]
    # a collar thinner than VISIBLE_WIDTH (0.08 mm at medium) cannot be seen: the fold side of a tail; the
    # angles, the creases and the height over the skin are measured where it can
    vis = diag["W"] >= VISIBLE_WIDTH
    vis = vis & np.roll(vis, 1) & np.roll(vis, -1)
    rep = {"seconds": seconds, "faces": int(len(mesh.faces)), "watertight": bool(mesh.is_watertight),
           "outward": bool(mesh.volume() > 0), "visible_share": float(vis.mean())}
    rep.update(_stats(_angles(rim, N[n_vis - 1])[vis], "rim"))
    # the appendage's shaded normal at E: closest point on the wall, barycentric over its vertex normals
    vw = sh.vertex_normals(wall.vertices, wall.faces)
    bn = _shaded_normals(diag["E"], wall, vw)
    rep.update(_stats(_angles(vn[A * n:(A + 1) * n], bn)[vis], "wall"))
    # the angle skin / wall without the collar: the skin normal at the exit against the wall normal there
    wall_n = _shaded_normals(diag["X"], wall, vw)
    sk_n = diag["N"][0]
    rep.update(_stats(np.degrees(np.arccos(np.clip(np.abs((wall_n * sk_n).sum(-1)), 0.0, 1.0)))[vis],
                      "before"))
    # visible sheet vs the skin
    patch = skin if isinstance(skin, SkinPatch) else None
    if patch is None:
        sk = skin if skin is not None else shaped_skin(base, shaped_verts, levels)
        r_max = float(np.max(placed.kind.collar_axis(placed, ctl)[1]))
        patch = SkinPatch(sk[0], sk[1], np.asarray(placed.anchor, dtype=np.float64),
                          (prof.width + 2.4) * r_max + 0.5)
    sheet = top[1:n_vis][:, vis].reshape(-1, 3)
    _, _, sd, _ = patch.closest(sheet)
    rep["above_skin_min"] = float(sd.min())
    # creases of the visible sheet: dihedral between the face normals of neighbouring quads of the rings
    T = top[:n_vis]
    j, jn = np.arange(n), (np.arange(n) + 1) % n
    fn = np.cross(T[1:, jn] - T[1:, j], T[:-1, j] - T[1:, j])               # (n_vis - 1, n, 3)
    fn /= np.maximum(np.linalg.norm(fn, axis=-1, keepdims=True), _EPS)
    vf = vis & np.roll(vis, -1)
    d_rad = _angles(fn[1:], fn[:-1])[:, vf].ravel()
    d_cir = _angles(fn, np.roll(fn, -1, axis=1))[:, vf].ravel()
    rep["crease_p99"] = float(np.percentile(np.concatenate([d_rad, d_cir]), 99))
    rep["self_crossings"] = count_crossings(mesh.vertices, mesh.faces)
    rep["below_skin_vertices"] = int((sd < -SKIN_TOL).sum())               # visible sheet vertices under the skin
    rep["width_min"], rep["width_max"] = float(diag["W"].min()), float(diag["W"].max())
    rep["height_min"], rep["height_max"] = float(diag["H"].min()), float(diag["H"].max())
    return rep
