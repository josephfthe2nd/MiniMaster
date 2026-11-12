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
SKIRT_U = (0.3, 0.65, 1.0)      # the skirt that closes the sheet under the skin past the dive, in skirt lengths
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
ALPHA_WALL_DEG = 20.0   # the sheet meets the appendage's wall at this angle and goes in: a clean contact line (round 4)
GAMMA_SKIN_DEG = 5.0    # ... and goes under the skin at this angle (CollarProfile.ramp = its tangent)
CUT_WIDTH = 0.02        # units: a column narrower than this (or CUT_REL root radii) has no collar: the ring opens
CUT_REL = 0.08
DIVE_SHARE = 0.5        # past the crossing the sheet dives to this share of the sink, then the skirt closes it
DIVE_U = (0.4, 0.75, 1.0)
SKIRT_LEN = 2.0         # the skirt that closes the sheet under the skin is this many sinks long
MIN_SECTOR = 6          # fewer columns than this: no collar
MIN_RUN = 10            # a run of collar columns shorter than this is dropped (it would be all taper)
GUARD_ITERATIONS = 3    # cut the columns whose triangles cross (or whose sheet folds), at most this many times
GUARD_CHECK = True
TAPER_COLS = 4.0        # a run of collar columns tapers to nothing over this many columns at each end
SINK_MAX_REL = 0.12     # root radii: the underside need only be under the skin for the slicer's union, not deep
SINK_MIN_REL = 0.01     # ... and never less than this
ARC_RADIUS = 0.3        # the turn into the appendage, in its radius at the root
TURN_MAX_DEG = 40.0     # the skin under a collar may turn this far from the exit's normal; the reach ends there
CONVERGE_MIN = 0.35      # neighbouring columns may come this close (of their spacing at the exit) before the reach ends
TOP_SMOOTHING = 4       # rounds of 1-2-1 smoothing of the top edge's height around the ring
RAY_GROUP = 0.04        # rays within this much arclength of each other are cast against one slab of the wall
RAY_SLAB_PAD = 0.03     # ... the triangles spanning it, padded by this much
EXIT_STEPS = 6          # secant steps fitting the exit curve to the skin
BOTTOM_STEP = 3         # the underside of the shell keeps every third station
POLY_GAP_SPREAD = 0.4     # corners of a tilted ring seen from the axis are unevenly spaced: this much spread is a polygon
FOLD_DEG = 60.0         # a sheet whose normal where it meets the skin is this far off the skin's is folded
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
    ramp: float = 0.0875
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
# a shell that carries its own shading


class CollarMesh(Mesh):
    """A collar shell that carries the normals and the cavity term it is SHADED with
    (``shading_normals`` (n, 3), ``shading_cavity`` (n,), or ``None``: the renderer computes its own).

    Why: the renderer shades every shell with area weighted vertex normals and a cavity AO computed on
    that shell alone, so a separate shell lying on the skin is drawn with its own normals and its own
    occlusion: a tone patch the size of its footprint, and where the two surfaces are within the depth
    buffer's precision of each other (the rim, tangent to the skin) a serrated z-fighting outline (the
    round-3 "sticker").  The collar therefore carries the SKIN's shading near its rim and the WALL's near
    its top edge, and its own between (``shading_transfer``): whichever surface wins the depth test at a
    seam, the pixel is the same.  This is the normal-transfer used to hide the seams of separately
    modelled pieces of a character; the geometry (and so the print) is unchanged by it.

    Every transform keeps the attributes (normals turn with the inverse transpose); ``Mesh.merge``
    and the print path drop them (a print does not shade)."""

    def __init__(self, vertices, faces, shading_normals=None, shading_cavity=None, profile_x=None):
        super().__init__(vertices, faces)
        self.shading_normals = None if shading_normals is None else np.asarray(shading_normals, dtype=np.float64)
        self.shading_cavity = None if shading_cavity is None else np.asarray(shading_cavity, dtype=np.float64)
        self.profile_x = None if profile_x is None else np.asarray(profile_x, dtype=np.float64)

    def _like(self, verts, faces, linear=None):
        n = self.shading_normals
        if n is not None and linear is not None:
            n = _unit(n @ np.linalg.inv(linear))                 # (M^-T n) as rows
        return CollarMesh(verts, faces, n, self.shading_cavity, self.profile_x)

    def copy(self) -> "CollarMesh":
        return CollarMesh(self.vertices.copy(), self.faces.copy(),
                          None if self.shading_normals is None else self.shading_normals.copy(),
                          None if self.shading_cavity is None else self.shading_cavity.copy(),
                          None if self.profile_x is None else self.profile_x.copy())

    def transform(self, m: np.ndarray) -> "CollarMesh":
        out = Mesh.transform(self, m)
        return self._like(out.vertices, out.faces, np.asarray(m, dtype=np.float64)[:3, :3])

    def translated(self, v) -> "CollarMesh":
        out = Mesh.translated(self, v)
        return self._like(out.vertices, out.faces)

    def scaled(self, s) -> "CollarMesh":
        out = Mesh.scaled(self, s)
        s = np.asarray(s, dtype=np.float64)
        return self._like(out.vertices, out.faces, np.diag(np.broadcast_to(s, (3,))) if s.ndim else
                          np.eye(3) * float(s))


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

    def closest_bary(self, points, guess=None):
        """``(point, corner vertex ids (n, 3), barycentric weights (n, 3), nearest vertex)`` of the
        surface nearest to each point: what :meth:`closest` and :meth:`sample` interpolate with."""
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
        return pt[sel], self.F[tri[rows, j]], bary[sel], v

    def closest(self, points, guess=None):
        """``(point, normal, signed distance, nearest vertex)`` of the surface nearest to each
        point (positive outside); ``guess`` warm-starts the vertex search (see above)."""
        P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        point, corner, bw, v = self.closest_bary(P, guess)
        normal = _unit((self.N[corner] * bw[:, :, None]).sum(axis=1))
        signed = ((P - point) * normal).sum(axis=1)
        return point, normal, signed, v

    @property
    def cavity(self) -> np.ndarray:
        """The renderer's cavity term (:func:`shading.cavity_term`) of every patch vertex: the same
        numbers the picture's ambient occlusion of the whole mesh has, away from the patch's border
        (the term looks two rings of neighbours out)."""
        if getattr(self, "_cavity", None) is None:
            from . import shading as sh
            self._cavity = sh.cavity_term(self.V, self.F, self.N)
        return self._cavity

    def sample(self, points, guess=None):
        """``(normal, cavity term, signed distance)`` of the surface at the point nearest to each of
        ``points``: the renderer's smooth vertex normal and its cavity term, interpolated as the
        picture interpolates them across a triangle."""
        P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        point, corner, bw, _ = self.closest_bary(P, guess)
        normal = _unit((self.N[corner] * bw[:, :, None]).sum(axis=1))
        term = (self.cavity[corner] * bw).sum(axis=1)
        return normal, term, ((P - point) * normal).sum(axis=1)


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
        # ... and its axial extent: its corners projected onto the nearest segment of the axis (a ray cast from
        # the axis at s, square to it, can only hit a triangle that spans s: the narrow slab of rays below)
        V = mesh.vertices
        a, b = axis.P[:-1], axis.P[1:]
        ab = b - a
        L2 = np.maximum((ab * ab).sum(-1), _EPS)
        t = np.clip(((V[:, None, :] - a[None]) * ab[None]).sum(-1) / L2[None], 0.0, 1.0)     # (nv, segs)
        foot = a[None] + t[..., None] * ab[None]
        k = ((V[:, None, :] - foot) ** 2).sum(-1).argmin(axis=1)
        vs = axis.s[k] + t[np.arange(len(V)), k] * np.sqrt(L2[k])
        fs = vs[mesh.faces]
        self.tri_lo, self.tri_hi = fs.min(axis=1), fs.max(axis=1)

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
        pad = RAY_SLAB_PAD
        while i < n:
            j = int(np.searchsorted(so, so[i] + RAY_GROUP, side="right"))
            j = max(j, i + 1)
            idx = order[i:j]
            lo, hi = float(so[i]), float(so[j - 1])
            near = (self.tri_hi >= lo - pad) & (self.tri_lo <= hi + pad)
            if near.any():
                dist[idx] = ray_mesh_first_hit(c[idx], e[idx], self.tris[near])
            i = j
        miss = ~np.isfinite(dist)
        if miss.any():                              # a bent axis: try the old, wide slab before giving up
            slab = max(0.35, 2.5 * self.edge)
            for q in np.nonzero(miss)[0]:
                near = (self.tri_s > s[q] - slab) & (self.tri_s < s[q] + slab)
                if near.any():
                    dist[q] = ray_mesh_first_hit(c[q:q + 1], e[q:q + 1], self.tris[near])[0]
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
        theta = np.asarray(theta, dtype=np.float64)
        n = len(theta)
        m = 2 * max(n, 72)
        th2 = np.arange(m) * (2.0 * math.pi / m)
        dist, _, _ = self.radius(th2, np.full(m, float(s_ref)))
        f = np.fft.rfft(dist)
        f[keep + 1:] = 0.0
        low = np.fft.irfft(f, m)
        at, _, _ = self.radius(theta, np.full(n, float(s_ref)))
        low_at = np.interp(theta % (2.0 * math.pi), np.concatenate([th2, [2.0 * math.pi]]),
                           np.concatenate([low, low[:1]]))
        return low_at - at

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

    # -- where the axis comes out of the skin --------------------------------------------------
    ns = int(np.clip(axis.length / 0.03, 24, 240))
    s_ax = np.linspace(0.0, axis.length, ns)
    pa, _, _, _, ra = axis.at(s_ax)
    _, _, sd_ax, _ = skin.closest(pa)
    inside = np.nonzero(sd_ax < 0.0)[0]
    if not len(inside):
        if diag is not None:
            diag["dropped"] = "the appendage's axis never enters the skin"
        return None
    after = np.nonzero((sd_ax >= 0.0) & (np.arange(ns) > inside[0]))[0]
    if not len(after):
        if diag is not None:
            diag["dropped"] = "the appendage's axis never comes out of the skin"
        return None
    s_e = float(s_ax[after[0]])
    r_root = float(ra[after[0]])

    # -- the columns: on the corners of the wall's polygon ----------------------------------------------
    # The collar's top edge goes INTO the wall at ALPHA_WALL_DEG; between two columns its edge is a straight
    # chord, and where a corner of the appendage's polygon (a 24-gon horn, a 32-sided wing sleeve) fell between
    # two columns the chord dipped inside the corner and the contact line dipped with it, by the corner's depth
    # over tan(alpha): a scallop per facet.  So the number of columns is a multiple of the polygon's sides and
    # the first column sits on a corner: every chord then lies ON a facet.
    wallf = Wall(axis, wall)
    theta, sides = _polygon_columns(wallf, s_e + 0.5 * profile.height * r_root * profile.strength, n)
    n = len(theta)
    theta0 = float(theta[0])
    away = np.degrees(np.abs((theta + math.pi) % (2.0 * math.pi) - math.pi))      # angle from the open side

    # -- the exit curve: where each generator of the wall pierces the skin --------------------------
    K = 8
    s_lo = max(0.0, s_e - 2.0 * r_root)
    s_hi = min(axis.length, s_e + 3.0 * r_root)
    st = np.linspace(s_lo, s_hi, K)
    n_scan = max(n // 3, 16)                  # a coarse scan around the ring, the secant steps below do the rest
    th_scan = np.arange(n_scan) * (2.0 * math.pi / n_scan)
    wp = wallf.point(np.repeat(th_scan + theta0, K), np.tile(st, n_scan))
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
        if diag is not None:
            diag["dropped"] = "the appendage's wall does not cross the skin near its exit"
        return None
    if not ok.all():                         # no crossing in range: borrow the nearest crossing
        idx = np.arange(n_scan)
        for j in np.nonzero(~ok)[0]:
            d = np.minimum((idx[ok] - j) % n_scan, (j - idx[ok]) % n_scan)
            s_scan[j] = s_scan[ok][d.argmin()]
    s_x = np.interp((theta - theta0) % (2.0 * math.pi), np.concatenate([th_scan, [2.0 * math.pi]]),
                    np.concatenate([s_scan, s_scan[:1]]))
    # secant steps on the real wall: the signed distance of the wall point to the skin goes to zero (the step of
    # 0.8 sd that assumed the wall square to the skin crawled where it leans, and a wide tail's exit never fitted)
    s_prev, sd_prev = None, None
    for _ in range(EXIT_STEPS):
        wp = wallf.point(theta, s_x)
        _, _, sdx, _ = skin.closest(wp)
        if (np.abs(sdx) < 0.2 * FIT_TOL).all():
            break
        if s_prev is None:
            s_new = s_x - 0.8 * sdx
        else:
            slope = (sdx - sd_prev) / np.where(np.abs(s_x - s_prev) > 1e-9, s_x - s_prev, 1e-9)
            ok_ = (slope > 0.05) & np.isfinite(slope)
            s_new = np.where(ok_, s_x - sdx / np.where(ok_, slope, 1.0), s_x - 0.8 * sdx)
            s_new = np.clip(s_new, s_x - 0.5 * r_root, s_x + 0.5 * r_root)
        s_prev, sd_prev = s_x, sdx
        s_x = np.clip(s_new, 0.0, axis.length)
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
    tan_g0 = math.tan(math.radians(GAMMA_SKIN_DEG)) if profile.ramp <= 0.0 else float(profile.ramp)
    v_vis = _visible_stations(profile.rings, profile.height / max(profile.width, _EPS), profile.falloff,
                              math.tan(math.radians(ALPHA_WALL_DEG)), tan_g0)
    skirt_u = np.asarray(SKIRT_U)
    wide = profile.width * r0
    sink0 = profile.sink * r0
    tan_g = math.tan(math.radians(GAMMA_SKIN_DEG)) if profile.ramp <= 0.0 else float(profile.ramp)
    tan_a = math.tan(math.radians(ALPHA_WALL_DEG))
    cut = max(CUT_WIDTH, CUT_REL * r0)                    # a collar narrower than this is no collar: the sector ends

    def sink_of(W):
        # a thin collar (the fold side: the skin is a crease there, two walls close together) is sunk
        # little: a pad 0.1 deep under a V-shaped cleft leaves the body through the other wall
        return sink0 * np.maximum(SINK_FLOOR, (W / wide) ** 0.7)

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
    nrings = len(v_vis) - 1
    u_an = v_vis[::2] if nrings % 2 == 0 else v_vis                              # the analysis marches every second station
    steps_an = W[:, None] * np.diff(u_an)[None, :]
    P_a, N_a, G_a = _march(skin, Xs, g0, steps_an, v_start)
    na = len(u_an) - 1
    lean_at = None                                                                # set below, once the reach is known
    step = np.degrees(np.arccos(np.clip((N_a[1:na] * N_a[: na - 1]).sum(-1), -1.0, 1.0)))
    chord = P_a[1:na] - P_a[: na - 1]                                             # a step that climbs or drops steeply
    drop = np.abs((chord * N_a[: na - 1]).sum(-1)) / np.maximum(np.linalg.norm(chord, axis=-1), _EPS)
    bad = (step > CREASE_STEP_DEG) | (drop > math.sin(math.radians(CREASE_DROP_DEG)))   # (stations - 1, n)
    first = np.where(bad.any(axis=0), bad.argmax(axis=0), na)
    reach = u_an[np.minimum(first + 1, na)]                                       # fraction of the width before the crease
    # the march itself must go outward: a step that turns back toward the wall (a fold of the surface the
    # closest point jumped across) ends the reach there too
    back = ((P_a[1:na] - P_a[: na - 1]) * g0[None]).sum(-1) < 0.0
    first_b = np.where(back.any(axis=0), back.argmax(axis=0), na)
    reach = np.minimum(reach, u_an[np.minimum(first_b + 1, na)])
    # ... and so does a skin that turns away by more than TURN_MAX_DEG over the reach (a collar wrapped round
    # the convex edge of the sacrum or a buttock folded its rim back over itself: a lowered tail on the heavy body).
    # (A limit on a march the skin holds back -- less than 0.6 of the way asked -- was tried and removed: switched
    # off over 158 builds (six bodies x 18 cases at level 1, three at level 2) it added one pre-guard crossing in all,
    # removed two folds and kept more collar: 0.14 of a ring summed over the builds.)
    turn = np.degrees(np.arccos(np.clip((N_a[:na] * Nx[None]).sum(-1), -1.0, 1.0))) > TURN_MAX_DEG
    first_t = np.where(turn.any(axis=0), turn.argmax(axis=0), na)
    reach = np.minimum(reach, u_an[np.minimum(first_t, na)])
    # ... and so does a column whose path over the skin runs into its neighbour's: marching outward from a concave
    # stretch of the exit curve, neighbouring paths converge and cross (a caustic) and the rim folds back over
    # itself there.  The reach of both ends where their spacing, measured along the exit curve's own direction,
    # has shrunk below CONVERGE_MIN of what it is at the exit
    jn = (np.arange(n) + 1) % n
    c0 = Xs[jn] - Xs
    l0 = np.linalg.norm(c0, axis=-1)
    c0 = c0 / np.maximum(l0, _EPS)[:, None]
    sep = ((P_a[:na, jn] - P_a[:na]) * c0[None]).sum(-1) / np.maximum(l0, _EPS)[None]
    conv = sep < CONVERGE_MIN
    first_c = np.where(conv.any(axis=0), conv.argmax(axis=0), na)
    first_c = np.minimum(first_c, np.roll(first_c, 1))                            # both columns of the pair
    reach = np.minimum(reach, u_an[np.maximum(np.minimum(first_c, na) - 1, 0)])
    # the lean of the wall over the skin: the wall's generator against the outward direction of the skin a third of
    # the way out, or at the last station before a crease or a fold (measured at the far end of the march, a path
    # that had run on over the spine's furrow said the back's own wall leaned over it, and cut the dorsal collar)
    last = np.minimum.reduce([first, first_b, first_t, first_c])
    k_lean = np.clip(np.minimum(na // 3, last - 1), 0, na - 1)
    G_l = G_a[k_lean, np.arange(n)]
    beta = np.degrees(np.arccos(np.clip((Tx * G_l).sum(-1), -1.0, 1.0)))         # wall vs outward skin
    lean = _smoothstep((beta - LEAN_FROM_DEG) / (LEAN_FULL_DEG - LEAN_FROM_DEG))
    if diag is not None:
        diag["limits"] = dict(fit_ok=fit_ok.copy(), lean=lean.copy(), reach=reach.copy(), first=first.copy(),
                              first_b=first_b.copy(), first_t=first_t.copy(), first_c=first_c.copy(), W0=W.copy())
    W = np.maximum(W * lean * np.minimum(1.0, reach / 0.85), MIN_WIDTH)
    W = np.where(fit_ok, _ring_smooth(W, RIM_SMOOTHING), MIN_WIDTH)             # no collar where the fit failed, not even a smoothed one

    def ends_taper(keep):
        """1 in the middle of a run of kept columns, falling to 0 over ``TAPER_COLS`` toward its ends (a ring: 1)."""
        if keep.all():
            return np.ones(n)
        d = np.full(n, float(n))
        idx = np.arange(n)
        for j in np.nonzero(~keep)[0]:
            d = np.minimum(d, np.minimum((idx - j) % n, (j - idx) % n))
        return _smoothstep(d / TAPER_COLS)

    def construct(W, keep, sink_scale):
        """The rings of the columns ``keep`` at the reach ``W`` (n,): ``(top, bottom, arcs, x_top, info)``."""
        # a run of columns ends in a taper: the reach, the height, the sink and the turn into the appendage all
        # fall toward its two end columns, so the loops that close it (its caps) are small and hidden
        taper = ends_taper(keep)
        Wk = cut + np.maximum(W - cut, 0.0) * taper
        # the height fades to nothing where the width falls to the cut: the visible sheet ends on the skin
        fade = _smoothstep((Wk - cut) / (1.5 * cut))
        H = np.maximum(profile.height * r0 * (Wk / wide) ** 0.7 * fade, 1e-4)
        # the height (and with it the visible reach) is smoothed round the ring: the fade near the cut turned a
        # dip of the width three columns wide into a notch, and the sheet creased 60 degrees across it
        H = np.where(keep, np.maximum(_ring_smooth(np.where(keep, H, 1e-4), 3), 1e-4), H)
        q = H / Wk
        x_e, hne, u_c, g_c, u_x = _column_profile(q, profile.falloff, tan_a, tan_g)
        sink = np.minimum(sink_of(Wk), SINK_MAX_REL * r_root) * sink_scale * (0.25 + 0.75 * taper)
        sink = np.maximum(sink, SINK_MIN_REL * r_root)          # never nil: the underside would lie on the sheet
        dive = DIVE_SHARE * sink                                                  # how deep the sheet goes past the crossing
        # the dive leaves the crossing at the crossing slope and bends down to ``dive`` within ``Ld`` (a short
        # hidden sheet: a long one over the curved skin of the back folded over itself)
        Ld = np.clip(3.0 * dive, 0.02, 0.25)
        u_d = u_x + Ld / Wk
        eps = np.maximum(SKIRT_LEN * sink, 0.03)
        u_vis = v_vis[None, :] * u_x[:, None]                                     # (n, stations): the visible part
        u_dv = u_x[:, None] + (u_d - u_x)[:, None] * np.asarray(DIVE_U)[None, :]  # the dive under the skin
        steps = np.concatenate([Wk[:, None] * np.diff(u_vis, axis=1),
                                Wk[:, None] * np.diff(np.concatenate([u_x[:, None], u_dv], axis=1), axis=1),
                                eps[:, None] * np.diff(np.concatenate([[0.0], skirt_u]))[None, :]], axis=1)
        P_m, N_m, G_m = _march(skin, Xs, g0, steps, v_start)
        nv = nrings                                                               # marched visible stations
        # the march can stall (the closest point held back by a fold of the skin for a few steps, then let go):
        # the visible stations are put back where they belong along the path it walked, by its arc length
        P_m, N_m = _respace(Xs, P_m, N_m, np.cumsum(steps[:, :nv], axis=1).T, nv)
        nd = len(DIVE_U)
        # next to the wall the marched points jitter by a millimetre-ish (the facets of the subdivided
        # skin) which is as much as the distance between stations there: the stations next to the wall
        # lie on the straight chord from the wall to the crossing, blended into the marched points
        rho = (Wk[:, None] * u_vis[:, 1:]).T                                      # (nv, n)
        Wv = (Wk * u_x)[None, :]
        chord_p = Xs[None] + (rho / np.maximum(Wv, _EPS))[..., None] * (P_m[nv - 1] - Xs)[None]
        blend = _smoothstep(rho / (0.1 * Wv + 0.03))[..., None]
        P_m = P_m.copy()
        P_m[:nv] = chord_p + blend * (P_m[:nv] - chord_p)
        # the exit point on the wall and the nearest skin point differ by the residual of the secant steps
        shift = X - Xs
        P_m[:nv] += ((1.0 - v_vis[1:]) ** 3)[:, None, None] * shift[None]
        # past the crossing the sheet is hidden: the dive and the skirt are the straight continuation of the skin's
        # tangent there, offset along the normal there (a strip marched on over a curved, creased skin turned
        # sideways column by column and its neighbours crossed it); they are short, so a convex skin's rise over
        # the tangent (L^2 / 2R) stays well inside the dive
        rho_s = np.cumsum(steps[:, nv:], axis=1).T                                # (nd + skirt, n) distance past it
        P_m[nv:] = P_m[nv - 1][None] + rho_s[..., None] * G_m[nv - 1][None]
        N_m = N_m.copy()
        N_m[nv:] = N_m[nv - 1][None]
        P = np.concatenate([X[None], P_m])                                        # stations 0 .. visible, dive, skirt
        N = np.concatenate([Nx[None], N_m])
        n_vis = nv + 1
        # the wall point H up the generator, ON the real (faceted) wall: the sheet arrives there at ALPHA and goes in
        # the axial height of the top edge is smoothed around the ring: the exit curve under it carries the skin's
        # facets (a few thousandths, column to column) and the contact line on the wall zigzagged with them
        s_top = _ring_smooth(s_x + H, TOP_SMOOTHING)
        E = wallf.point(theta, np.clip(s_top, 0.0, axis.length))
        Hv = np.linalg.norm(E - X, axis=1)
        # the wall's generator: from the exit up the (smoothed) wall far enough that the facets cannot turn it (a
        # tapered column's E is a hair from X, and E - X pointed wherever the facet put it)
        Hg = np.maximum(H, 0.25 * r_root)
        gen = _unit(wallf.smooth_points(theta, np.clip(s_x + Hg, 0.0, axis.length), delta) - X)
        # ... but where the collar is tall enough, the profile ends exactly on E (its last chord then arrives at the
        # designed angle; aimed along the smoothed generator it arrived anywhere from 3 to 55 degrees)
        trust = _smoothstep(Hv / (0.25 * r_root))[:, None]
        D_end = _unit(trust * _unit(E - X) + (1.0 - trust) * gen)
        thin = 1.0 - _smoothstep(W / (0.1 * wide))
        Nb = _unit((1.0 - thin)[None, :, None] * N + thin[None, :, None] * N[n_vis - 1][None])
        # the underside is the skin offset inward along its normals; where neighbouring normals converge faster
        # than the points are apart (a convex bend of the skin under a narrow collar: a slim shoulder) the offset
        # points swap places and the underside crosses itself.  The sink stays under that focal distance.
        jn = (np.arange(n) + 1) % n
        def focal(Pa, Pb, Na, Nb_):
            dv = Pb - Pa
            d = np.linalg.norm(dv, axis=-1)
            conv = ((Nb_ - Na) * dv).sum(-1) / np.maximum(d, _EPS)          # how fast the offsets approach
            return np.where(conv > 1e-9, 0.5 * d / np.maximum(conv, 1e-9), np.inf)
        lim_c = focal(P[:, :], P[:, jn], Nb[:, :], Nb[:, jn]).min(axis=0)            # between neighbour columns
        lim_r = focal(P[:-1], P[1:], Nb[:-1], Nb[1:]).min(axis=0)                  # along a column
        lim = np.minimum(np.minimum(lim_c, np.roll(lim_c, 1)), lim_r)
        sink = np.minimum(sink, np.maximum(lim, SINK_MIN_REL * r_root))
        # ... smoothed round the ring and never above that (a sink that jumped from one column to the next twisted the
        # underside and the skirt between them into each other); eroded by both neighbours first, so a mirrored
        # pair of roots gets mirrored sinks (a one-sided erosion made the ring order matter)
        eroded = np.minimum.reduce([sink, np.roll(sink, 1), np.roll(sink, -1)])
        sink = np.minimum(sink, np.maximum(_ring_smooth(eroded, 2), SINK_MIN_REL * r_root))
        sink = np.minimum.reduce([sink, np.roll(sink, 1) * 1.5, np.roll(sink, -1) * 1.5])
        dive = np.minimum(dive, 0.5 * sink)
        # -- the sheet: the profile, the dive, the skirt -----------------------------------------------------------
        top = np.empty_like(P)
        top[0] = E
        for i in range(1, n_vis):
            u = u_vis[:, i]
            f = _column_height(u, x_e, hne, u_c, g_c, q, tan_g, profile.falloff)
            w = _smoothstep(1.0 - v_vis[i])
            D = _unit((1.0 - w) * N[i] + w * D_end)
            top[i] = P[i] + (Hv * f)[:, None] * D
        for k in range(nd):
            i = n_vis + k
            d_ = Ld * DIVE_U[k]
            h = -tan_g * d_ - np.maximum(dive - tan_g * Ld, 0.0) * DIVE_U[k] ** 2
            top[i] = P[i] + np.maximum(h, -dive)[:, None] * Nb[i]
        for k, sg in enumerate(skirt_u):
            i = n_vis + nd + k
            # the skirt ends HALF way down to the underside, which closes it with an edge of its own (a skirt that met
            # the underside in a knife edge made slivers that crossed their neighbours' where two columns differed)
            top[i] = P[i] - (dive + 0.5 * (sink - dive) * _smoothstep(sg))[:, None] * Nb[i]
        bottom = P - sink[None, :, None] * Nb                                     # same stations, under the skin
        # into the appendage: on along the arrival direction (ALPHA into the wall), curving to the axis
        e_in = -_unit(e_x - (e_x * D_end).sum(-1, keepdims=True) * D_end)
        a_in = np.arctan(tan_a * hne / np.maximum(x_e, _EPS))                    # the profile's angle to the wall at E
        t0 = _unit(np.cos(a_in)[:, None] * D_end + np.sin(a_in)[:, None] * e_in)
        n_in = _unit(e_in - (e_in * t0).sum(-1, keepdims=True) * t0)
        # the turn into the appendage is a fraction of its own radius there, not of the collar's size (a doubled
        # collar turned through the axis and met the other side's), and small at the ends of a run
        r_wall = np.linalg.norm(E - axis.at(np.clip(s_x + H, 0.0, axis.length))[0], axis=1)
        rc = (ARC_RADIUS * np.minimum(r_root, r_wall) * (0.3 + 0.7 * taper))[:, None]
        arcs = np.array([E + rc * (math.sin(math.radians(a)) * t0 + (1.0 - math.cos(math.radians(a))) * n_in)
                         for a in ARC_ANGLES])
        x_top = np.concatenate([1.0 - v_vis, -0.02 * np.arange(1, nd + 1), -0.1 - 0.1 * np.arange(1, len(skirt_u) + 1)])
        info = dict(H=Hv, E=E, P=P, N=N, n_vis=n_vis, u_x=u_x, sink=sink, x_e=x_e, a_in=np.degrees(a_in), Wk=Wk)
        return top, bottom, arcs, x_top, info

    keep = _drop_short_runs((W >= cut) & fit_ok, MIN_RUN)
    sink_scale = np.ones(n)
    shell = None
    log = []
    for it in range(GUARD_ITERATIONS + 1):
        keep = _drop_short_runs(keep, MIN_RUN)
        if keep.sum() < MIN_SECTOR:
            if diag is not None:
                diag["dropped"] = (f"the collar is narrower than {cut:.3f} units all round "
                                   "(the wall leans over the skin or the skin is a crease everywhere)"
                                   if it == 0 else "the collar crossed itself and was cut away")
                diag["guard_log"] = log
            return None
        top, bottom, arcs, x_top, info = construct(W, keep, sink_scale)
        # the sheet must go INTO the skin where it meets it: a column whose sheet there faces away from the skin's
        # normal by far more than the crossing angle is folded back over itself (a lowered tail's flank)
        nvi = info["n_vis"]
        fr = _unit(np.cross(top[nvi - 1][(np.arange(n) + 1) % n] - top[nvi - 1],
                            top[nvi - 2] - top[nvi - 1]))
        mism = np.degrees(np.arccos(np.clip((fr * info["N"][nvi - 1]).sum(-1), -1.0, 1.0)))
        pair = keep & np.roll(keep, -1) & (info["Wk"] * info["u_x"] > 2.0 * cut)   # (j, j+1) both there, not a tapered end
        fp = pair & (mism > FOLD_DEG)
        fold = fp | np.roll(fp, 1)
        shell = assemble_shell(top, bottom, arcs, x_top, keep=keep)
        crossing = crossing_pairs(shell.vertices, shell.faces) if GUARD_CHECK else np.zeros((0, 2), dtype=np.int64)
        badcol = fold.copy()
        if len(crossing):
            badcol[np.unique(shell.column_of[shell.faces[crossing.ravel(), 0]])] = True
            badcol &= keep
        log.append(dict(crossings=int(len(crossing)), folds=int(fold.sum()), bad=int(badcol.sum()),
                        kept=int(keep.sum())))
        if not badcol.any():
            break
        if it == GUARD_ITERATIONS:
            shell = None
            break
        # no collar in those columns: the runs on either side taper to the gap (narrowing them in place sheared the
        # columns against their neighbours and crossed more than it uncrossed)
        keep = keep & ~(badcol | np.roll(badcol, 1) | np.roll(badcol, -1))
        shell = None
    if shell is None:
        if diag is not None:
            diag["dropped"] = "the collar crossed itself and was cut away"
            diag["guard_log"] = log
        return None
    if diag is not None:
        diag.update(s_exit=s_e, r_root=r_root, X=X, s_x=s_x, sd_exit=sdx, W=np.where(keep, info["Wk"], 0.0), beta=beta,
                    lean=lean, theta=theta, axis=axis, top=top, bottom=bottom, arcs=arcs, keep=keep,
                    guard_iterations=it, guard_log=log, cut=cut, sides=sides, theta0=theta0, **info)
        diag["vid_top"] = shell.vid_top
        diag["vid_arcs"] = shell.vid_arcs
    return shell


def _visible_stations(rings: int, q: float, k: float, tan_a: float, tan_g: float):
    """Fractions ``v`` (0 at the wall, 1 where the sheet meets the skin) of the visible profile of a column of
    height over reach ``q`` (see :func:`_column_profile`), spaced half equally in ARC LENGTH and half equally in
    TURNING ANGLE: the curvature is concentrated near the wall, and equal arc length alone cuts it into a few
    long chords, equal turning alone into tiny ones the skin's facets swamp."""
    x_e, hne, u_c, g_c, u_x = (float(np.ravel(a)[0]) for a in _column_profile(np.array([q]), k, tan_a, tan_g))
    v = np.linspace(0.0, 1.0, 4001)
    u = v * u_x
    h = q * _column_height(u, np.array(x_e), np.array(hne), np.array(u_c), np.array(g_c), np.array(q), tan_g, k)
    du, dh = np.diff(u), np.diff(h)
    seg = np.hypot(du, dh)
    ang = np.unwrap(np.arctan2(-dh, du))
    turn = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(np.concatenate([[ang[0]], ang]))))])
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    measure = 0.5 * arc / arc[-1] + 0.5 * turn / max(turn[-1], _EPS)
    return np.interp(np.linspace(0.0, 1.0, rings + 1), measure, v)


def _polygon_columns(wallf: "Wall", s_ref: float, n: int, probe: int = 720):
    """``(theta (columns,), sides)``: the column angles of a collar on the wall's polygon.  The wall's radius
    around the axis at ``s_ref`` (``probe`` rays) has a kink at every corner of a polygon (the second difference
    of the radius is a sharp negative spike there, a smooth section's curvature is a fraction of it); with the
    corners found, every facet gets the same number of columns (the smallest that makes at least ``n`` in all,
    at most twice ``n``), the first ON its corner.  A wall with no clear corners (a smooth or irregular loft)
    keeps ``n`` uniform columns from 0 (sides 0)."""
    uniform = np.arange(n) * (2.0 * math.pi / n)
    th = np.arange(probe) * (2.0 * math.pi / probe)
    r, _, _ = wallf.radius(th, np.full(probe, float(np.clip(s_ref, 0.0, wallf.axis.length))))
    d2 = np.roll(r, -1) - 2.0 * r + np.roll(r, 1)
    spike = -d2
    thr = max(6.0 * float(np.median(np.abs(d2))), 2e-4 * float(r.mean()))
    peak = (spike > thr) & (spike >= np.roll(spike, 1)) & (spike > np.roll(spike, -1))
    cand = np.nonzero(peak)[0]
    if len(cand) < 3:
        return uniform, 0
    # the corners are the strongest kinks; a ray that crosses the diagonal of a twisted quad of the wall kinks
    # too, more weakly: take the most kinks that are evenly spaced round the ring
    cand = cand[np.argsort(-spike[cand])]
    cand = cand[spike[cand] > 0.05 * spike[cand[0]]]
    idx = None
    for k in range(min(len(cand), 2 * n), 2, -1):
        t = np.sort(cand[:k])
        gaps = np.diff(np.concatenate([t, [t[0] + probe]])).astype(np.float64)
        if gaps.std() <= POLY_GAP_SPREAD * gaps.mean():
            idx = t
            break
    if idx is None:                                           # not a polygon: corners all over the place
        return uniform, 0
    k = len(idx)
    # sub-sample corner angles: the vertex of a parabola through the spike and its neighbours
    a_, b_, c_ = spike[(idx - 1) % probe], spike[idx], spike[(idx + 1) % probe]
    den = a_ - 2.0 * b_ + c_
    off = np.where(np.abs(den) > 1e-15, 0.5 * (a_ - c_) / np.where(np.abs(den) > 1e-15, den, 1.0), 0.0)
    corners = (idx + np.clip(off, -0.5, 0.5)) * (2.0 * math.pi / probe)
    per = int(math.ceil(n / k))
    if per * k > 2 * n:
        return uniform, 0
    nxt = np.concatenate([corners[1:], [corners[0] + 2.0 * math.pi]])
    f = np.arange(per) / per
    theta = (corners[:, None] + (nxt - corners)[:, None] * f[None, :]).ravel()
    return theta, k


def _respace(start, P, N, want, nv):
    """The first ``nv`` stations of marched paths ``P`` / ``N`` ((m, n, 3), from ``start`` (n, 3)) moved along each
    path's polyline to the arc lengths ``want`` ((nv, n), increasing): a stalled march bunches its stations and
    then jumps, and the strip between two columns, one bunched and one not, crossed itself."""
    pts = np.concatenate([start[None], P[:nv]])                                   # (nv + 1, n, 3)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=-1)                           # (nv, n)
    acc = np.concatenate([np.zeros((1, pts.shape[1])), np.cumsum(seg, axis=0)])  # (nv + 1, n)
    total = acc[-1]
    stall = (seg.max(axis=0) > 3.0 * np.maximum(np.median(seg, axis=0), 1e-12)) | (seg.min(axis=0) < 0.2 * np.maximum(np.median(seg, axis=0), 1e-12))
    if not stall.any():
        return P, N
    P = P.copy()
    N = N.copy()
    nrm = np.concatenate([N[:1], N[:nv]])
    for j in np.nonzero(stall & (total > 1e-9))[0]:
        t = want[:, j] / max(float(want[-1, j]), 1e-12) * total[j]               # the same share of the walked path
        for c in range(3):
            P[:nv, j, c] = np.interp(t, acc[:, j], pts[:, j, c])
            N[:nv, j, c] = np.interp(t, acc[:, j], nrm[:, j, c])
    N[:nv] = _unit(N[:nv])
    return P, N


def _drop_short_runs(keep, min_run: int):
    """``keep`` without its runs of fewer than ``min_run`` columns (a run tapers over ``TAPER_COLS`` at each end:
    a shorter one is all taper, its two caps a column apart, and its loops crossed each other)."""
    keep = np.asarray(keep, dtype=bool).copy()
    n = len(keep)
    if keep.all() or not keep.any():
        return keep
    start = [j for j in range(n) if keep[j] and not keep[j - 1]]
    for s0 in start:
        run = [s0]
        while keep[(run[-1] + 1) % n] and len(run) < n:
            run.append((run[-1] + 1) % n)
        if len(run) < min_run:
            keep[run] = False
    return keep


def _hn(x, k):
    x = np.clip(x, 0.0, 1.0)
    return 1.0 - np.sqrt(np.maximum(1.0 - x ** k, 0.0))


def _hn_d(x, k):
    x = np.clip(x, 0.0, 1.0)
    return k * x ** (k - 1.0) / (2.0 * np.sqrt(np.maximum(1.0 - x ** k, 1e-300)))


def _bisect_increasing(fun, target, lo, hi, iters: int = 60):
    """Vectorised bisection for ``fun(x) = target`` with ``fun`` increasing on ``[lo, hi]``."""
    lo = np.broadcast_to(np.asarray(lo, dtype=np.float64), np.shape(target)).copy()
    hi = np.broadcast_to(np.asarray(hi, dtype=np.float64), np.shape(target)).copy()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        up = fun(mid) < target
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    return 0.5 * (lo + hi)


def _column_profile(q, k, tan_a, tan_g):
    """The profile of a column whose height over reach is ``q``: ``(x_e, hn(x_e), u_c, g_c, u_x)``.

    The quarter-ellipse-like profile ``hn(x) = 1 - sqrt(1 - x^k)`` (x: 1 at the wall, 0 at the rim) is cut at
    BOTH ends so the collar crosses the two surfaces it joins at a definite angle instead of lying tangent on
    them (a tangent join leaves the contact line wherever the facets of the two meshes happen to cross, a
    serrated or scalloped edge, and a band where the two z-fight):

    * at the wall it ends at ``x_e`` where it is ``ALPHA_WALL_DEG`` off the wall, rescaled so it still reaches
      the full height there (``g(u) = hn(x_e (1 - u)) / hn(x_e)``, u: 0 at the wall, 1 at the old rim);
    * toward the skin it follows ``g`` while it is steeper than ``tan_g`` and then goes on straight at that
      slope, through the skin at ``u_x`` (the visible reach) and under it: the near-flat part that added less
      than a hair of height (and was the wide tinted 'patch' of the round-3 critics) is under the skin.

    The shading of both edges is the skin's and the wall's (:func:`shading_transfer`), so neither angle is seen."""
    q = np.maximum(np.asarray(q, dtype=np.float64), 1e-6)
    x_e = _bisect_increasing(lambda x: _hn_d(x, k), 1.0 / (q * tan_a), 1e-9, 1.0 - 1e-12)
    hne = np.maximum(_hn(x_e, k), 1e-12)
    t = tan_g * hne / (q * np.maximum(x_e, 1e-12))
    flat = t >= _hn_d(x_e, k)                       # the whole profile is flatter than the crossing slope
    x_c = np.where(flat, x_e, _bisect_increasing(lambda x: _hn_d(x, k), np.minimum(t, _hn_d(x_e, k)), 0.0, x_e))
    u_c = np.where(flat, 0.0, 1.0 - x_c / np.maximum(x_e, 1e-12))
    g_c = _hn(x_c, k) / hne
    u_x = u_c + q * g_c / tan_g
    return x_e, hne, u_c, g_c, u_x


def _column_height(u, x_e, hne, u_c, g_c, q, tan_g, k):
    """``h / H`` of the columns at ``u`` (see :func:`_column_profile`): ``g(u)`` up to ``u_c``, a straight line
    at the crossing slope after it (negative past ``u_x``: under the skin)."""
    u = np.asarray(u, dtype=np.float64)
    g = _hn(x_e * (1.0 - np.minimum(u, u_c)), k) / hne
    return np.where(u <= u_c, g, g_c - (tan_g / np.maximum(q, 1e-6)) * (u - u_c))


def _ear_clip(poly):
    """Triangles ``(m, 3)`` (indices into ``poly``, (k, 2)) of a simple polygon, in its own orientation."""
    pts = np.asarray(poly, dtype=np.float64)
    k = len(pts)
    area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    sign = 1.0 if area >= 0 else -1.0
    idx = list(range(k))
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 4 * k * k:
        guard += 1
        m = len(idx)
        cut_one = False
        for t in range(m):
            i0, i1, i2 = idx[t - 1], idx[t], idx[(t + 1) % m]
            a, b, c = pts[i0], pts[i1], pts[i2]
            cr = sign * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
            if cr <= 1e-14:
                continue
            ok = True
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                p = pts[j]
                d1 = sign * ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))
                d2 = sign * ((c[0] - b[0]) * (p[1] - b[1]) - (c[1] - b[1]) * (p[0] - b[0]))
                d3 = sign * ((a[0] - c[0]) * (p[1] - c[1]) - (a[1] - c[1]) * (p[0] - c[0]))
                if d1 >= 0 and d2 >= 0 and d3 >= 0:
                    ok = False
                    break
            if ok:
                tris.append((i0, i1, i2))
                idx.pop(t)
                cut_one = True
                break
        if not cut_one:                              # not simple (a degenerate loop): fan the rest
            for t in range(1, len(idx) - 1):
                tris.append((idx[0], idx[t], idx[t + 1]))
            idx = []
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    return np.array(tris, dtype=np.int64).reshape(-1, 3)


def assemble_shell(top, bottom, arcs, x_top=None, keep=None):
    """Close the rings into one watertight triangle mesh, outward oriented (a :class:`CollarMesh`).

    ``top`` (S, n, 3): the sheet from the wall (station 0) outward over the skin to the end of the skirt
    (station S-1); ``bottom`` (S, n, 3): the same stations pushed under the skin (every ``BOTTOM_STEP``-th and the last three are
    used; the skirt's end and the underside's are joined by an edge); ``arcs`` (A, n, 3): the sheet carried on into the appendage past the wall.
    Every column is a closed loop (arcs, top, bottom, back into the appendage) swept around the axis; ``keep``
    (n,) says which columns there are: all of them close into a ring (a torus), a run of them into a sector
    closed by its two end loops (ear-clipped caps), several runs into several closed pieces of one mesh.
    ``x_top`` (S,): where each station of ``top`` lies on the profile (1 at the wall, 0 at the crossing of
    the skin, negative past it): every vertex gets one in ``profile_x`` (the arcs above 1, the hidden underside
    -1), what :func:`shading_transfer` blends the skin's, the collar's and the wall's shading by. The mesh also
    carries ``column_of`` (the column of every vertex) and ``vid_top`` / ``vid_arcs`` (the vertex of every
    station and column, -1 where there is no column)."""
    S, n, _ = top.shape
    A = len(arcs)
    keep = np.ones(n, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    if x_top is None:
        x_top = np.linspace(1.0, -1.0, S)
    x_top = np.asarray(x_top, dtype=np.float64)
    # the underside is hidden and nearly flat: every BOTTOM_STEP-th station of it (and its two ends) is enough, and
    # its rows were 40 % of the shell's faces (the crossing check's cost)
    B = sorted(set(range(0, S, BOTTOM_STEP)) | {S - 3, S - 2, S - 1})    # fine where it meets the skirt's end
    loop = np.concatenate([arcs[::-1], top, bottom[B][::-1]])               # (L, n, 3): one closed loop per column
    xl = np.concatenate([1.0 + 0.1 * np.arange(A, 0, -1), x_top, np.full(len(B), -1.0)])
    L = len(loop)
    cols = np.nonzero(keep)[0]
    vid = np.full((L, n), -1, dtype=np.int64)
    vid[:, cols] = np.arange(L * len(cols)).reshape(len(cols), L).T
    verts = loop[:, cols].transpose(1, 0, 2).reshape(-1, 3)
    px = np.tile(xl, len(cols))
    colof = np.repeat(cols, L)
    faces = []
    ring = bool(keep.all())
    li, ln = np.arange(L), (np.arange(L) + 1) % L
    # the runs of kept columns (a ring: one run that wraps)
    runs = []
    if ring:
        runs = [list(range(n)) + [0]]
    else:
        start = [j for j in range(n) if keep[j] and not keep[j - 1]]
        for s in start:
            r = [s]
            while keep[(r[-1] + 1) % n] and len(r) < n:
                r.append((r[-1] + 1) % n)
            runs.append(r)
    for r in runs:
        for j, jn in zip(r[:-1], r[1:]):
            a, b = vid[li, j], vid[ln, j]
            c, d = vid[ln, jn], vid[li, jn]
            faces.append(np.stack([a, b, c], axis=1))
            faces.append(np.stack([a, c, d], axis=1))
        if not ring:
            for j, rev in ((r[0], True), (r[-1], False)):
                # the end loop, flattened in its own plane (its two principal directions), ear-clipped
                pts = loop[:, j]
                c0 = pts.mean(axis=0)
                _, _, vt = np.linalg.svd(pts - c0, full_matrices=False)
                p2 = (pts - c0) @ vt[:2].T
                tri = _ear_clip(p2)
                ids = vid[:, j][tri]
                faces.append(ids[:, ::-1] if rev else ids)
    F = np.concatenate(faces) if faces else np.zeros((0, 3), dtype=np.int64)
    m = CollarMesh(verts, F, profile_x=px)
    if len(F) and m.volume() < 0:
        m = CollarMesh(verts, F[:, [0, 2, 1]], profile_x=px)
    m.column_of = colof
    m.vid_top = vid[A:A + S]
    m.vid_arcs = vid[A - 1::-1] if A else vid[:0]
    return m


# --------------------------------------------------------------------------
# the shading the collar is drawn with


SHADE_SKIN_TO = 0.55    # the skin's shading holds from the rim (x = 0) to this far up the profile (x: 1 at the wall) ...
SHADE_WALL_FROM = 0.45   # ... and the wall's from here to the wall; the collar's own normals between
SHADE_AO_WALL_FROM = 0.9   # the occlusion is the skin's up to here, the wall's at the top edge


def shade_weights(x):
    """``(skin, own, wall)`` weights (each (n,), summing to 1) of a collar vertex at profile position ``x``."""
    x = np.asarray(x, dtype=np.float64)
    w_skin = 1.0 - _smoothstep(x / SHADE_SKIN_TO)
    w_wall = _smoothstep((x - SHADE_WALL_FROM) / (1.0 - SHADE_WALL_FROM))
    over = w_skin + w_wall
    scale = np.where(over > 1.0, 1.0 / np.maximum(over, _EPS), 1.0)        # the two ranges overlap a little
    w_skin, w_wall = w_skin * scale, w_wall * scale
    return w_skin, np.clip(1.0 - w_skin - w_wall, 0.0, 1.0), w_wall


def outward(mesh: Mesh) -> Mesh:
    """``mesh`` with its triangles wound outward (a closed mesh of negative volume turned over): the wall a kind
    hands over may be a mirrored piece wound inside out (the right wing's arm), and its normals, which the collar
    is shaded with and measured against, would point in."""
    if mesh is not None and len(mesh.faces) and mesh.volume() < 0.0:
        return Mesh(mesh.vertices, mesh.faces[:, ::-1])
    return mesh


def shading_transfer(mesh: "CollarMesh", skin: SkinPatch, wall: Mesh | None = None):
    """Set ``mesh.shading_normals`` / ``shading_cavity`` (see :class:`CollarMesh`): near the rim the
    normal and the cavity term of the SKIN under each vertex (``skin``: the body as it is meshed, the
    renderer's area weighted normals and its cavity term), near the top edge those of the WALL (the
    appendage's mesh), the collar's own between, blended by :func:`shade_weights` on ``mesh.profile_x``.
    Hidden vertices (the underside, the parts inside the appendage) keep the nearest blend."""
    from . import shading as sh
    wall = outward(wall)
    V = mesh.vertices
    x = mesh.profile_x if mesh.profile_x is not None else np.zeros(len(V))
    w_skin, w_own, w_wall = shade_weights(x)
    # the occlusion is the skin's under the whole visible sheet and the wall's at its top edge: the collar's own
    # cavity term (computed on the thin shell alone, and scaled by its station spacing) drew a dark ring half way
    # up the fillet
    a_wall = _smoothstep((x - SHADE_AO_WALL_FROM) / (1.0 - SHADE_AO_WALL_FROM))
    n_own = sh.vertex_normals(V, mesh.faces)
    N = w_own[:, None] * n_own
    T = np.zeros(len(V))
    k = (w_skin > 0.0) | (a_wall < 1.0)
    if k.any():
        n_s, t_s, _ = skin.sample(V[k])
        N[k] += w_skin[k, None] * n_s
        T[k] += (1.0 - a_wall[k]) * t_s
    k = (w_wall > 0.0) | (a_wall > 0.0)
    if k.any():
        if wall is not None and len(wall.faces):
            cen = V[k].mean(axis=0)
            reach = float(np.linalg.norm(V[k] - cen, axis=1).max()) + 0.5
            wp = SkinPatch(wall.vertices, wall.faces, cen, reach)
            n_w, t_w, _ = wp.sample(V[k])
        else:
            n_w, t_w = n_own[k], np.zeros(int(k.sum()))
        N[k] += w_wall[k, None] * n_w
        T[k] += a_wall[k] * t_w
    mesh.shading_normals = _unit(N)
    mesh.shading_cavity = np.clip(T, 0.0, 1.0)
    return mesh


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


def wants_collar(specs, enabled: bool = True) -> bool:
    """Does any enabled appendage of ``specs`` (records, or built ``Placed``) get a root collar?  A built one
    asks its kind (``collar_profile``); a record asks whether its kind has a collar at all and its ``collar``
    parameter (the kind's default when absent) is above 0, which costs nothing (no appendage is built)."""
    if not enabled:
        return False
    from . import kinds as K
    for sp in specs or ():
        if hasattr(sp, "kind") and hasattr(sp.kind, "collar_profile") and hasattr(sp, "anchor"):
            prof = sp.kind.collar_profile(sp)                     # a Placed
            if prof is not None and prof.active:
                return True
            continue
        if not getattr(sp, "enabled", True):
            continue
        kind = K.kind_of(sp)
        if kind is None or type(kind).collar_profile is K.Kind.collar_profile:
            continue
        params = dict(getattr(sp, "params", {}) or {})
        spec = kind.param("collar", params) or kind.param("collar")
        default = spec.default if spec is not None else 1.0
        try:
            if float(params.get("collar", default)) > 0.0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def default_levels(specs, enabled: bool = True) -> int:
    """The subdivision level a character is meshed at when none is asked for: ``MIN_LEVEL`` (1) when an
    appendage gets a root collar (the collar needs a subdivided body, and without it the default picture and
    print showed the bare pipe in a flange the collar exists to remove), else 0 (the cage, as before)."""
    return MIN_LEVEL if wants_collar(specs, enabled) else 0


def collar_material(placed, profile: CollarProfile):
    """The material the collar renders with: the body's skin or the appendage's own."""
    if profile.material == "appendage":
        return placed.material
    from . import shading as sh
    return sh.SKIN


def collar_pieces(base, shaped_verts, placed, controls=None, *, levels: int = 2, skin=None,
                  profile: CollarProfile | None = None, diag: dict | None = None, shade: bool = True) -> list:
    """``[(name, Mesh, material)]``: the collar of one placed appendage (``[]``: none), in the
    MakeHuman frame like ``Placed.mesh()``.

    ``shaped_verts`` are the body's vertices AFTER :func:`~.junction.apply_junctions` (the skin
    the collar sits on); ``levels`` is the subdivision the body is meshed with (the collar is
    fitted to THAT surface: a collar fitted to level 2 stands off a level-1 body by the
    Catmull-Clark shrink); ``skin`` is that body mesh as ``(vertices, triangles)`` in the same
    frame if the caller has it already (see :func:`skin_from_shell`) or a :class:`SkinPatch`;
    ``controls`` the pose the appendage is drawn in (default: its own).  ``profile`` overrides
    ``placed.kind.collar_profile``.  At ``levels`` 0 (the cage itself) there is no collar (``MIN_LEVEL``).
    The mesh is a :class:`CollarMesh` that carries the skin's and the wall's shading at its two edges
    (:func:`shading_transfer`; ``shade=False``: none, the renderer computes its own)."""
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
    if shade:
        shading_transfer(mesh, patch, wall)
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
        prof = p.kind.collar_profile(p)
        if prof is None:
            continue
        diag: dict = {}
        try:
            got = collar_pieces(base, shaped_verts, p, None if controls is None else controls(p),
                                levels=levels, skin=skin, diag=diag)
            for name, mesh, mat in got:
                out.append((p, name, mesh, mat))
            if not got and diag.get("dropped"):
                # a collar the kind asked for and the construction could not build is said, not swallowed
                from .kinds import AppendageWarning
                warnings.warn(f"appendage {p.name!r}: no root collar ({diag['dropped']})", AppendageWarning,
                              stacklevel=2)
        except (ValueError, KeyError, IndexError, FloatingPointError, np.linalg.LinAlgError) as exc:
            from .kinds import AppendageWarning
            warnings.warn(f"appendage {p.name!r}: no root collar ({exc})", AppendageWarning,
                          stacklevel=2)
    return out


# --------------------------------------------------------------------------
# measuring a collar (tests, tools/collar_report.py, docs/collar.md)


def crossing_pairs(vertices, faces) -> np.ndarray:
    """``(k, 2)`` the pairs of triangles of ONE mesh that cross each other (as :func:`count_crossings`),
    with a broad phase sized for a collar: a uniform grid of cells 1.5 median triangles wide (the
    generic grid of ``junction_checks.overlapping_pairs`` takes the 90th percentile, and the long
    triangles of a collar's fans and skirt made it pair 250,000 boxes: 0.9 s), the few long triangles
    tested against every box directly. About 40 ms for a 7,000-face collar."""
    from . import junction_checks as JC
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    if not len(F):
        return np.zeros((0, 2), dtype=np.int64)
    P = V[F]
    lo, hi = P.min(axis=1), P.max(axis=1)
    ext = (hi - lo).max(axis=1)
    cell = max(1.5 * float(np.median(ext)), 1e-6)
    big = ext > 4.0 * cell
    small = np.nonzero(~big)[0]
    out = []
    if len(small):
        origin = lo.min(axis=0) - 1e-6
        t, k, _ = JC._cells(lo[small], hi[small], cell, origin, max_span=6)
        o = np.argsort(k, kind="stable")
        ks, ts = k[o], small[t[o]]
        left = np.searchsorted(ks, k, side="left")
        right = np.searchsorted(ks, k, side="right")
        cnt = right - left
        tot = int(cnt.sum())
        if tot:
            ii = np.repeat(small[t], cnt)
            start = np.cumsum(cnt) - cnt
            jj = ts[np.arange(tot) - np.repeat(start, cnt) + np.repeat(left, cnt)]
            kk = np.repeat(k, cnt)
            keep = ii < jj
            ii, jj, kk = ii[keep], jj[keep], kk[keep]
            ok = np.all(lo[ii] <= hi[jj], axis=1) & np.all(lo[jj] <= hi[ii], axis=1)
            ii, jj, kk = ii[ok], jj[ok], kk[ok]
            corner = np.floor((np.maximum(lo[ii], lo[jj]) - origin) / cell).astype(np.int64)
            first = kk == JC._key(corner[:, 0], corner[:, 1], corner[:, 2])
            out.append(np.stack([ii[first], jj[first]], axis=1))
    bigs = np.nonzero(big)[0]
    for s0 in range(0, len(bigs), 64):
        bb = bigs[s0:s0 + 64]
        ok = (np.all(lo[None, :, :] <= hi[bb][:, None, :], axis=2) & np.all(lo[bb][:, None, :] <= hi[None, :, :], axis=2))
        ok[np.arange(len(bb)), bb] = False
        bi, js = np.nonzero(ok)
        bi = bb[bi]
        keep = ~big[js] | (js > bi)                       # a big-big pair once, big-small always
        bi, js = bi[keep], js[keep]
        out.append(np.stack([np.minimum(js, bi), np.maximum(js, bi)], axis=1))
    if not out:
        return np.zeros((0, 2), dtype=np.int64)
    allp = np.concatenate(out)
    nf = np.int64(len(F))
    key = np.unique(allp[:, 0].astype(np.int64) * nf + allp[:, 1])
    pairs = np.stack([key // nf, key % nf], axis=1)
    share = (F[pairs[:, 0]][:, :, None] == F[pairs[:, 1]][:, None, :]).any(axis=(1, 2))
    pairs = pairs[~share]
    if not len(pairs):
        return pairs
    return pairs[JC.pairs_cross(V, F, pairs)]


def count_crossings(vertices, faces, other_vertices=None, other_faces=None) -> int:
    """How many pairs of triangles cross each other: within one mesh, or (with ``other_*``) between
    two. Triangles that share a vertex are not counted (neighbours), nor are touching or coplanar ones
    (the open-segment test of ``junction_checks.pairs_cross``)."""
    from . import junction_checks as JC
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    if other_vertices is None:
        return int(len(crossing_pairs(V, F)))
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


def _resample_columns(cols, sub_t: int = 4, sub_s: int = 6):
    """``cols`` (m, n, 3) stations x columns (closed around) -> (m', n * sub_t, 3) by linear interpolation:
    the surface the triangles between them draw, sampled finely."""
    m, n, _ = cols.shape
    s = np.linspace(0, m - 1, (m - 1) * sub_s + 1)
    i = np.clip(np.floor(s).astype(int), 0, m - 2)
    f = (s - i)[:, None, None]
    c = cols[i] * (1 - f) + cols[i + 1] * f
    t = np.arange(n * sub_t) / sub_t
    j = np.floor(t).astype(int) % n
    g = (t - np.floor(t))[None, :, None]
    return c[:, j] * (1 - g) + c[:, (j + 1) % n] * g


def _wave(h, ok, half: int = 12) -> float:
    """p95 - p5 of what a quadratic fit over ``2 half + 1`` samples (three columns either side, at four samples
    a column) leaves of the circular signal ``h``, over the samples whose whole window is ``ok`` and finite: the
    scallops and teeth of a contact line (a facet is three columns), not its slow rise and fall around the
    ring, nor the taper at the end of a run of columns."""
    h = np.asarray(h, dtype=np.float64)
    m = len(h)
    good = np.isfinite(h) & ok
    if good.sum() < 2 * half + 2:
        return float("nan")
    x = np.where(good, h, 0.0)
    w = good.astype(np.float64)
    k = np.arange(-half, half + 1)
    idx = (np.arange(m)[:, None] + k[None, :]) % m
    full = w[idx].all(axis=1)
    # a quadratic least-squares fit over the window (Savitzky-Golay): the curve's own bend is not a wobble
    A = np.stack([np.ones_like(k), k, k * k], axis=1).astype(np.float64)
    coef = np.linalg.pinv(A)[0]                                   # the fit's value at the window's centre
    avg = x[idx] @ coef
    r = (h - avg)[full]
    return float(np.percentile(r, 95) - np.percentile(r, 5)) if len(r) > 8 else float("nan")


def contact_lines(diag: dict, wall: Mesh, skin: SkinPatch, sub_t: int = 4) -> dict:
    """Where the collar's SURFACE (its triangles, sampled finely) enters the wall and where it enters the
    skin, column by column, and how wavy those two lines are: what the eye sees as the edge of the collar
    (a scalloped or serrated contact line is a facet of one mesh crossing a tangent sheet of the other).

    * ``wall_wave``: p95 - p5 of the local wobble (:func:`_wave`: what a quadratic fit over three columns either side
      leaves) of the position, along the axis, at which the surface enters the wall, in root radii (the samples between
      two kept columns, the tapered ends of a run included); ``wall_contact_missing``: the share
      of the kept columns whose surface never enters the wall (a gap between the collar and the appendage);
    * ``rim_wave``: the same for the distance from the wall at which the surface enters the skin."""
    wall = outward(wall)
    top, arcs = diag["top"], diag["arcs"]
    n_vis = int(diag["n_vis"])
    r = float(diag["r_root"])
    keep = np.asarray(diag.get("keep", np.ones(top.shape[1], dtype=bool)), dtype=bool)
    _, Tx, _, _, _ = diag["axis"].at(diag["s_x"])
    X = diag["X"]
    keepd = np.repeat(keep & np.roll(keep, -1), sub_t)          # samples between two kept columns
    out = {}
    cols = np.concatenate([top[:n_vis][::-1], arcs], axis=0)   # from the crossing of the skin up to E and on in
    D = _resample_columns(cols, sub_t)
    m2, n2, _ = D.shape
    wp = SkinPatch(wall.vertices, wall.faces, X[keep].mean(axis=0), 8.0 * r + 1.0)
    _, _, sd, _ = wp.closest(D.reshape(-1, 3))
    sd = sd.reshape(m2, n2)
    Xd = _resample_columns(X[None].repeat(2, 0), sub_t)[0]
    Td = _resample_columns(Tx[None].repeat(2, 0), sub_t)[0]
    inside = sd < 0.0
    k = np.where(inside.any(axis=0), inside.argmax(axis=0), -1)
    hw = np.full(n2, np.nan)
    j = np.nonzero(k > 0)[0]
    a_, b_ = sd[k[j] - 1, j], sd[k[j], j]
    f = np.where(a_ != b_, a_ / np.where(a_ != b_, a_ - b_, 1.0), 0.5)[:, None]
    C = D[k[j] - 1, j] * (1 - f) + D[k[j], j] * f
    s_xd = _resample_columns(np.asarray(diag["s_x"], dtype=np.float64)[None, :, None].repeat(2, 0), sub_t)[0, :, 0]
    hw[j] = s_xd[j] + ((C - Xd[j]) * Td[j]).sum(-1)      # where on the wall, along the axis (not over the jittery exit)
    out["wall_contact_missing"] = float(np.mean(~np.isfinite(hw[keepd]))) if keepd.any() else 0.0
    out["wall_wave"] = _wave(hw, keepd) / r
    D = _resample_columns(top, sub_t)
    m2, n2, _ = D.shape
    _, _, sd, _ = skin.closest(D.reshape(-1, 3))
    sd = sd.reshape(m2, n2)
    start = max(0, (n_vis - 4) * 6)                              # from a few stations inside the crossing on
    inside = sd[start:] < -1e-4
    k = np.where(inside.any(axis=0), inside.argmax(axis=0) + start, -1)
    rho = np.full(n2, np.nan)
    j = np.nonzero(k > 0)[0]
    rho[j] = np.linalg.norm(D[k[j], j] - Xd[j], axis=1)
    out["rim_wave"] = _wave(rho, keepd) / r
    return out


def join_report(base, shaped_verts, placed, controls=None, *, levels: int = 2, skin=None,
                profile: CollarProfile | None = None) -> dict:
    """The numbers of one collar (``{}`` if the appendage has none). Angles in degrees, lengths in units
    unless named otherwise. "Visible" columns are the kept ones at least ``VISIBLE_WIDTH`` wide.

    * ``rim_*``: the angle between the normal the RENDERER shades the collar with at the line where it meets
      the skin (``CollarMesh.shading_normals``: the skin's own there, :func:`shading_transfer`) and the skin's
      smooth normal: 0 = no edge can be seen in the shading;
    * ``wall_*``: the same where the collar meets the appendage, against the appendage's shaded normal;
    * ``geo_rim_*`` / ``geo_wall_*``: the same with the collar's OWN area weighted vertex normals (its geometry,
      what a print has): the angle at which the surface crosses the skin (``GAMMA_SKIN_DEG``) and the wall
      (``ALPHA_WALL_DEG``, less where the profile is flatter); ``fold_columns``: kept columns whose own rim
      normal is more than 90 degrees from the skin's (a sheet folded back over itself);
    * ``lip_max``: the highest the line where the sheet meets the skin stands over the skin (a step a print shows);
    * ``before_*``: the angle between the skin and the wall where they meet WITHOUT the collar (90 = the skin
      runs into the wall, 0 = tangent: what the junction alone leaves);
    * ``above_skin_min`` / ``below_skin_vertices``: over the visible sheet (between the wall and the crossing);
      ``crease_p99``: the 99th percentile of the dihedral between neighbouring faces of the visible sheet;
    * ``self_crossings`` (the shell with itself, exact), ``watertight`` / ``outward``, ``faces``, ``width_*`` /
      ``height_*``, ``kept_share`` (the share of the ring that has a collar), ``guard_iterations``;
    * ``wall_wave`` / ``rim_wave`` / ``wall_contact_missing``: :func:`contact_lines` (root radii).
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
    ctl = placed.controls if controls is None else controls
    wall = outward(placed.kind.collar_wall(placed, ctl))   # the surface the collar meets (a wing: its arm piece)
    top, N = diag["top"], diag["N"]
    n_vis = int(diag["n_vis"])
    keep = np.asarray(diag["keep"], dtype=bool)
    vt = diag["vid_top"]
    vn_geo = sh.vertex_normals(mesh.vertices, mesh.faces)
    vn_shd = mesh.shading_normals if getattr(mesh, "shading_normals", None) is not None else vn_geo
    vis = keep & (diag["W"] * diag["u_x"] >= VISIBLE_WIDTH)                    # the reach above the skin
    vis = vis & np.roll(vis, 1) & np.roll(vis, -1)
    rep = {"seconds": seconds, "faces": int(len(mesh.faces)), "watertight": bool(mesh.is_watertight),
           "outward": bool(mesh.volume() > 0), "visible_share": float(vis.mean()), "kept_share": float(keep.mean()),
           "guard_iterations": int(diag.get("guard_iterations", 0))}
    if not vis.any():
        vis = keep.copy()
    rim_ids = vt[n_vis - 1][vis]
    skin_n = N[n_vis - 1][vis]
    rep.update(_stats(_angles(vn_shd[rim_ids], skin_n), "rim"))
    rep.update(_stats(_angles(vn_geo[rim_ids], skin_n), "geo_rim"))
    rep["fold_columns"] = int((_angles(vn_geo[vt[n_vis - 1][keep]], N[n_vis - 1][keep]) > 90.0).sum())
    vw = sh.vertex_normals(wall.vertices, wall.faces)
    bn = _shaded_normals(diag["E"][vis], wall, vw)
    e_ids = vt[0][vis]
    rep.update(_stats(_angles(vn_shd[e_ids], bn), "wall"))
    rep.update(_stats(_angles(vn_geo[e_ids], bn), "geo_wall"))
    wall_n = _shaded_normals(diag["X"][vis], wall, vw)
    sk_n = diag["N"][0][vis]
    rep.update(_stats(np.degrees(np.arccos(np.clip(np.abs((wall_n * sk_n).sum(-1)), 0.0, 1.0))), "before"))
    patch = skin if isinstance(skin, SkinPatch) else None
    if patch is None:
        sk = skin if skin is not None else shaped_skin(base, shaped_verts, levels)
        r_max = float(np.max(placed.kind.collar_axis(placed, ctl)[1]))
        patch = SkinPatch(sk[0], sk[1], np.asarray(placed.anchor, dtype=np.float64),
                          (prof.width + 2.4) * r_max + 0.5)
    sheet = top[1:n_vis - 1][:, vis].reshape(-1, 3)
    _, _, sd, _ = patch.closest(sheet)
    rep["above_skin_min"] = float(sd.min()) if len(sd) else 0.0
    rep["below_skin_vertices"] = int((sd < -SKIN_TOL).sum())
    _, _, sdr, _ = patch.closest(top[n_vis - 1][keep])
    rep["lip_max"] = float(np.abs(sdr).max())
    T = top[:n_vis]
    n = T.shape[1]
    j, jn = np.arange(n), (np.arange(n) + 1) % n
    fn = np.cross(T[1:, jn] - T[1:, j], T[:-1, j] - T[1:, j])
    fn /= np.maximum(np.linalg.norm(fn, axis=-1, keepdims=True), _EPS)
    vf = vis & np.roll(vis, -1)
    d_rad = _angles(fn[1:], fn[:-1])[:, vf].ravel()
    d_cir = _angles(fn, np.roll(fn, -1, axis=1))[:, vf & np.roll(vf, -1)].ravel()
    both = np.concatenate([d_rad, d_cir])
    rep["crease_p99"] = float(np.percentile(both, 99)) if len(both) else 0.0
    rep["self_crossings"] = int(len(crossing_pairs(mesh.vertices, mesh.faces)))
    rep.update(contact_lines(diag, wall, patch))
    W = diag["W"][keep]
    rep["width_min"], rep["width_max"] = float(W.min()), float(W.max())
    rep["height_min"], rep["height_max"] = float(diag["H"][keep].min()), float(diag["H"][keep].max())
    return rep
