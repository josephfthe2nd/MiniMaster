"""The weld: the body's own skin moulded onto each appendage's root (round 6).

WHY.  The skin junction (:mod:`.junction`) moves the 13,378-quad body CAGE: a cage quad is ~0.28 units and a
root 0.4-0.6, so the cage cannot make a tangent fillet (the skin meets a root 55-80 degrees from tangent).
The root collar (:mod:`.collar`) laid a second, thin shell over the skin: as geometry its outline stays on
the skin (a print, an STL viewer), and it was absent at level 0, in the lab draft and wherever its guard
dropped it.  The weld moves the SKIN ITSELF.  The subdivided body (levels 1-2) is derived (morph targets
address the cage only), so its vertices may be moved freely at mesh time: around every root the skin's
vertices are moved onto a fillet that leaves the appendage's wall tangentially and lands on the original
skin.  One surface climbs the root; there is no second shell and no rim on the skin.

THE FIELDS.  ``a`` is the height over the ORIGINAL skin, ``b`` the distance outside the root's WALL (both
signed distances along the smooth normal, :class:`Surface`).  The welded skin is the zero set of the smooth
union ``smin(a, b / region; k)``: tangent to the wall at height ``k``, landing on the skin ``region * k``
from the wall.  Its profile is ``(a, b) = (k (1 - h)^2, region k h^2)`` for ``h`` in 0..1 (``h`` is also the
blend's weight there).  ``k`` varies over the skin (:func:`_reach`): the kind's own shape
(:class:`WeldProfile`: long on the open side, a crease on the other), less where the wall leans over the
skin (``acute``), at most the concave radius of the skin (``valley``: the cleft under a tail, an armpit),
fading with the distance from the root and away from an earlier root's weld.

THE SETS (region-local).  ``cap``: the vertices under the root (``b < 0``), holes filled; ``top``: the skin's
last ring outside the wall, thinned so that no triangle has all three corners on it; ``rim``: the region's
outer ring (fixed); ``sleeve``: the rest.  ``cap1``: the cap's first ring.

THE CONSTRUCTION.  Every sleeve vertex gets a share ``u`` of the way from the top ring to the rim (graph
distances along the mesh: monotone, never two rings on one spot) and goes to the profile point ``h(u)`` (even
spacing by arc length), first by an analytic placement, then by Newton steps onto the zero set; its ``h`` is
restored along the surface (the spacing that makes the skin ARRIVE tangentially is kept, not relaxed away)
and it is evened out along its ring only.  The top ring is pinned ON the wall at height ``k`` (a clean contact
polyline); ``cap1`` continues up the wall just under it; the rest of the cap is a harmonic membrane spanning
``cap1`` (inside the convex hull of a ring on the wall: inside the root, without folds).  A guard finds folded
faces (a face whose normal turns away from the blend's own gradient), shrinks the reach there and solves
again; if the skin still crosses itself the root is left unwelded and a warning says so: torn skin never ships.

API: :class:`WeldProfile`, :func:`weld_skin` (one root, any skin and wall), :func:`weld_targets` /
:func:`weld_mesh` / :func:`weld_body` (the pipelines), :func:`arrival_report` (the measures).
Everything is in the MakeHuman frame (Y up) like ``Placed.mesh()`` unless a function says otherwise.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass

import numpy as np

from ..core.mesh import Mesh
from . import collar as CO

MIN_LEVEL = 1          # the default subdivision of a figure with a root that wants a weld (level 0 is the cage)
FOLD_COS = 0.26        # a visible face whose normal is more than ~75 degrees from the blend's gradient is folded
CLEAR_SHARE = 0.6     # a probe at height s up the skin's normal whose own distance to the skin is under this share of s ...
CLEAR_K = 0.45        # ... has another part of the skin near it: the reach there is at most this many times s
CLIMB_MAX = 1.0       # the last ring climbs the wall at most this many reaches (where the wall runs along the skin, the
                      # height it can reach there is its reach: a crease, not a sliver dragged along the root)
PROJECT_ROUNDS = 3    # Newton steps of the sleeve onto the zero set ...
RESTORE_ROUNDS = 2    # ... and steps along it back to its own profile parameter
U_SMOOTH = 4          # rounds of smoothing of the share of the way to the rim
WALL_MARGIN = 0.8     # units of wall past the clip that are searched (true normals at the clip)
CAP1_ON = 0.7         # the cap's first ring: this many edges on past the last ring, the way the skin arrives ...
CAP1_BACK = 0.1       # ... and this share of the way back to where it was, under the wall (distinct points)
MIN_AREA = 1e-10      # a welded triangle smaller than this is degenerate (Mesh.degenerate_faces: 1e-12): refused
LAND_MAX = 1.5        # a fillet lands at most this many times its nominal distance (``region`` reaches) out
GUARD_SHRINK = 0.45    # the reach where a fold was found is cut by this share per guard round
_H = np.linspace(0.0, 1.0, 513)


@dataclass(frozen=True)
class WeldProfile:
    """How the skin is welded onto one root (lengths in the root's radii unless named in units).

    ``reach``: the fillet's height up the wall, ``k``, in root radii (at most ``reach_max`` units);
    ``region``: it lands on the skin this many ``k`` from the wall; ``strength``: the record's ``collar``
    (0..2) scales ``k``; ``up`` / ``open`` / ``ventral``: the kind's own shape, the factor of ``k`` on the side
    ``up`` points to (the back above a tail, the spine beside a wing) and on the opposite side (the crease under
    a tail), blended smoothly round the root (``up=None``: the same all round); ``acute``: the least share of
    ``k`` where the wall leans over the skin (a fold, not a web); ``valley``: ``k`` is at most this many concave
    radii of the skin; ``wall_clip``: the wall within the root radius + this many ``k`` of the root is the one
    the skin is welded to; ``inset``: units the cap's first ring lies under the wall; ``grade``: >1 packs the
    rings toward the wall; ``guard``: re-solves with a smaller reach where a face folded."""

    reach: float = 0.9
    reach_max: float = 0.75
    region: float = 2.0
    strength: float = 1.0
    up: tuple | None = None
    open: float = 1.0
    ventral: float = 1.0
    acute: float = 0.35
    valley: float = 0.8
    wall_clip: float = 1.7
    inset: float = 0.02
    grade: float = 1.5
    relax: int = 2
    guard: int = 3

    def __post_init__(self):
        for key in ("reach", "reach_max", "region", "strength", "open", "ventral", "acute", "valley",
                    "wall_clip", "inset", "grade"):
            v = getattr(self, key)
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0.0:
                raise ValueError(f"WeldProfile.{key} must be a finite number >= 0, got {v!r}")
        if self.region <= 0.2 or self.grade <= 0.0:
            raise ValueError("WeldProfile.region must exceed 0.2 and grade 0")
        if self.up is not None:
            u = np.asarray(self.up, dtype=np.float64)
            if u.shape != (3,) or not np.all(np.isfinite(u)) or np.linalg.norm(u) < 1e-9:
                raise ValueError(f"WeldProfile.up must be a non-zero 3-vector or None, got {self.up!r}")
            object.__setattr__(self, "up", tuple(float(x) for x in u / np.linalg.norm(u)))

    @property
    def active(self) -> bool:
        return self.strength > 0.0 and self.reach > 0.0 and self.reach_max > 0.0

    def k(self, radius: float) -> float:
        """The fillet's reach ``k`` (units) on a root of ``radius``."""
        return min(self.reach * float(radius), self.reach_max) * self.strength

    def with_(self, **changes) -> "WeldProfile":
        d = asdict(self)
        d.update(changes)
        return WeldProfile(**d)

    def digest(self) -> str:
        return hashlib.blake2b(repr(sorted(asdict(self).items())).encode(), digest_size=8).hexdigest()

    def shape(self, points, centre) -> np.ndarray:
        """The kind's factor of the reach at ``points`` around a root at ``centre`` (1 all round without ``up``)."""
        P = np.asarray(points, dtype=np.float64)
        if self.up is None or (self.open == 1.0 and self.ventral == 1.0):
            return np.ones(len(P))
        d = CO._unit(P - np.asarray(centre, dtype=np.float64))
        c = d @ np.asarray(self.up)
        t = np.clip(0.5 + 0.5 * c, 0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)
        return self.ventral + (self.open - self.ventral) * t


@dataclass
class WeldTarget:
    """One root to weld: its wall (outward, MakeHuman frame), where it is, how thick, how."""

    name: str
    wall: Mesh
    centre: np.ndarray
    radius: float
    profile: WeldProfile
    axis: np.ndarray | None = None       # the root's centreline, from inside the body outward

    def digest(self) -> str:
        """What decides the weld of this root (for caches)."""
        h = hashlib.blake2b(digest_size=10)
        for arr in (self.wall.vertices, self.wall.faces, self.centre, [self.radius],
                    self.axis if self.axis is not None else [0.0]):
            h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        h.update(self.profile.digest().encode())
        return h.hexdigest()


# --------------------------------------------------------------------------
# the mesh's connectivity, once per topology


class Topology:
    """Vertex neighbours and vertex -> triangle incidence of one triangle list (CSR), cached by content: every
    body at one subdivision level has the same triangles, so the 0.1-0.4 s this costs is paid once."""

    _CACHE: "OrderedDict" = OrderedDict()
    _LOCK = threading.Lock()

    def __init__(self, faces, n_vertices: int):
        F = np.asarray(faces, dtype=np.int64)
        self.F = F
        self.n = int(n_vertices)
        e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        e = np.concatenate([e, e[:, ::-1]])
        key = np.unique(e[:, 0] * self.n + e[:, 1])
        a, b = key // self.n, key % self.n
        self.nb_ptr = np.concatenate([[0], np.cumsum(np.bincount(a, minlength=self.n))])
        self.nb = b
        flat = F.ravel()
        o = np.argsort(flat, kind="stable")
        self.vf_ptr = np.concatenate([[0], np.cumsum(np.bincount(flat, minlength=self.n))])
        self.vf = (o // 3).astype(np.int64)

    @classmethod
    def of(cls, faces, n_vertices: int) -> "Topology":
        F = np.ascontiguousarray(faces, dtype=np.int64)
        key = (int(n_vertices), F.shape, hashlib.blake2b(F.tobytes(), digest_size=12).hexdigest())
        with cls._LOCK:
            if key in cls._CACHE:
                cls._CACHE.move_to_end(key)
                return cls._CACHE[key]
        topo = cls(F, n_vertices)
        with cls._LOCK:
            cls._CACHE[key] = topo
            while len(cls._CACHE) > 4:
                cls._CACHE.popitem(last=False)
        return topo

    def table(self, ids) -> np.ndarray:
        """Padded neighbour table of the vertices ``ids``: ``(len(ids), K)`` vertex ids, ``-1`` = none."""
        ids = np.asarray(ids, dtype=np.int64)
        s, e = self.nb_ptr[ids], self.nb_ptr[ids + 1]
        cnt = e - s
        K = int(cnt.max()) if len(cnt) else 1
        tab = np.full((len(ids), max(K, 1)), -1, dtype=np.int64)
        row = np.repeat(np.arange(len(ids)), cnt)
        col = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        tab[row, col] = self.nb[np.repeat(s, cnt) + col]
        return tab

    def faces_of(self, ids) -> np.ndarray:
        """The triangles (indices) with a corner in ``ids``."""
        ids = np.asarray(ids, dtype=np.int64)
        s, e = self.vf_ptr[ids], self.vf_ptr[ids + 1]
        cnt = e - s
        col = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        return np.unique(self.vf[np.repeat(s, cnt) + col])


# --------------------------------------------------------------------------
# a surface with exact closest points


class Surface(CO.SkinPatch):
    """:class:`collar.SkinPatch` whose closest point is the closest TRIANGLE's, not only among the triangles of
    the nearest vertex: the best triangle walks to a vertex-adjacent one while that is closer.  On the tail's
    tube (triangles up to 0.7 units long) the nearest vertex's triangles missed the closest one for 6.5 % of the
    points within 0.6 units of the root, by up to 0.2 units (round 6, defect 9); the weld's inset is 0.02."""

    def __init__(self, vertices, faces, centre, radius, walk: int = 2):
        super().__init__(vertices, faces, centre, radius)
        self.walk = int(walk)
        V = np.asarray(vertices, dtype=np.float64)
        F = np.asarray(faces, dtype=np.int64)
        d2 = ((V - np.asarray(centre, dtype=np.float64)) ** 2).sum(axis=1)
        self.used = np.unique(F[(d2 < radius * radius)[F].any(axis=1)])     # patch vertex i is mesh vertex used[i]
        if self.walk:
            K = self._table.shape[1]
            nb = self._table[self.F].reshape(len(self.F), 3 * K)       # triangles sharing a corner
            nb = np.sort(nb, axis=1)
            dup = np.concatenate([np.zeros((len(nb), 1), bool), nb[:, 1:] == nb[:, :-1]], axis=1)
            nb[dup] = -1
            self._tri_nb = nb

    def vertex_normals_of(self, V, ids):
        """``(normal, patch vertex)`` of the mesh vertices ``ids`` (ON this surface): its own smooth normal, no
        search."""
        j = np.searchsorted(self.used, ids)
        j = np.clip(j, 0, len(self.used) - 1)
        ok = self.used[j] == ids
        if not ok.all():
            _, n, _, v = self.closest(np.asarray(V)[ids[~ok]])
            out_n = self.N[j].copy()
            out_v = j.copy()
            out_n[~ok], out_v[~ok] = n, v
            return out_n, out_v
        return self.N[j], j

    def closest_bary(self, points, guess=None):
        P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        v = self.nearest_vertex(P, guess)
        n = len(P)
        rows = np.arange(n)
        best_t, best_d = self._eval(P, self._table[v])
        for _ in range(self.walk):
            t, d = self._eval(P, self._tri_nb[best_t])
            better = d < best_d - 1e-18
            if not better.any():
                break
            best_t = np.where(better, t, best_t)
            best_d = np.where(better, d, best_d)
        corner = self.F[best_t]
        pt, bary = CO.closest_on_triangles(P, self.V[corner[:, 0]], self.V[corner[:, 1]], self.V[corner[:, 2]])
        del rows
        return pt, corner, bary, v

    def _eval(self, P, cand):
        ok = cand >= 0
        tri = np.where(ok, cand, 0)
        n, K = cand.shape
        a = self.V[self.F[tri, 0]].reshape(-1, 3)
        b = self.V[self.F[tri, 1]].reshape(-1, 3)
        c = self.V[self.F[tri, 2]].reshape(-1, 3)
        pp = np.repeat(P, K, axis=0)
        pt, _ = CO.closest_on_triangles(pp, a, b, c)
        d2 = np.where(ok, ((pt - pp) ** 2).sum(axis=1).reshape(n, K), np.inf)
        j = d2.argmin(axis=1)
        return tri[np.arange(n), j], d2[np.arange(n), j]


# --------------------------------------------------------------------------
# small helpers


def smin(a, b, k):
    """The polynomial smooth minimum of ``a`` and ``b`` over ``k`` and the weight of ``a`` in it."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h), h


def _arc(aspect: float) -> np.ndarray:
    """Arc length along the unit profile ``((1-h)^2, aspect h^2)`` at ``_H``."""
    return np.concatenate([[0.0], np.cumsum(np.hypot(np.diff((1.0 - _H) ** 2), aspect * np.diff(_H ** 2)))])


_ASPECTS = np.linspace(0.5, 8.0, 31)
_U = np.linspace(0.0, 1.0, 257)
_HU = np.array([np.interp(_U * a[-1], a, _H) for a in (_arc(x) for x in _ASPECTS)])     # h at an arc share u


def h_of_u(u, aspect):
    """The profile parameter ``h`` a share ``u`` of the way along the profile's arc (even spacing), for a profile
    of ``aspect`` (landing distance over height), bilinear in a table."""
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    x = np.clip((np.asarray(aspect, dtype=np.float64) - _ASPECTS[0]) / (_ASPECTS[1] - _ASPECTS[0]), 0.0,
                len(_ASPECTS) - 1.000001)
    i = np.floor(x).astype(np.int64)
    f = x - i
    y = u * (len(_U) - 1)
    j = np.minimum(np.floor(y).astype(np.int64), len(_U) - 2)
    g = y - j
    lo = _HU[i, j] * (1 - g) + _HU[i, j + 1] * g
    hi = _HU[i + 1, j] * (1 - g) + _HU[i + 1, j + 1] * g
    return lo * (1 - f) + hi * f


def _graph_distance(tab, has, length, seed, passable, iters: int = 200) -> np.ndarray:
    """Shortest path lengths (edges weighted by ``length``) from the region-local vertices ``seed`` through the
    ``passable`` ones (inf where not reached): Bellman-Ford rounds over the padded neighbour table."""
    n = len(seed)
    d = np.where(seed, 0.0, np.inf)
    safe = np.where(has, tab, 0)
    for _ in range(iters):
        cand = np.where(has, d[safe] + length, np.inf).min(axis=1)
        new = np.where(seed, 0.0, np.where(passable, np.minimum(d, cand), np.inf))
        if np.array_equal(new, d):
            break
        d = new
    del n
    return d


def _components(tab, has, mask) -> np.ndarray:
    """Labels of the connected components of the region-local vertices ``mask`` (-1 elsewhere)."""
    lab = np.where(mask, np.arange(len(mask)), -1)
    safe = np.where(has, tab, 0)
    for _ in range(len(mask)):
        nl = np.where(has & mask[safe], lab[safe], -1).max(axis=1)
        new = np.where(mask, np.maximum(lab, nl), -1)
        if np.array_equal(new, lab):
            break
        lab = new
    return lab


# --------------------------------------------------------------------------
# one root


def weld_skin(V, F, wall, centre, radius, profile: WeldProfile | None = None, *, axis=None,
              topo: Topology | None = None, locked=None, others=(), diag: dict | None = None) -> np.ndarray:
    """A copy of the skin's vertices ``V`` (triangles ``F``) welded onto ``wall`` (a closed Mesh, wound outward)
    around ``centre``; ``radius`` is the root's radius.  Vertices outside the weld's region are returned
    untouched (bit for bit), and so are the ``locked`` ones (a bool mask: an earlier root's weld).  ``axis`` is the
    root's centreline (points from inside the body outward; default: the wall's long axis near the root): the
    skin's last ring climbs the wall along it.  ``others``: the walls of the figure's other roots (Meshes): skin
    inside one of them is that root's plug and is not moved out of it.  ``diag`` receives the sets, the reach, the
    guard's rounds and the timings."""
    t0 = time.perf_counter()
    pr = profile or WeldProfile()
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    out = V.copy()
    centre = np.asarray(centre, dtype=np.float64)
    k0 = pr.k(radius)
    info = {"applied": False, "k0": k0, "guard": 0}
    if diag is not None:
        diag.update(info)
    if not pr.active or k0 < 1e-4 or wall is None or not len(wall.faces):
        return out
    topo = topo if topo is not None else Topology.of(F, len(V))
    spread = max(pr.open, pr.ventral, 1.0)
    rad = float(radius) + (1.25 * pr.region * spread + 0.5) * k0 + 0.3
    try:
        # the root's own centre is where its axis comes out of the skin (a wing's anchor is its joint, inside the
        # back, up to a unit from where the arm leaves it): the kind's shape and the fade are measured from there
        # the wall near the root, and a margin past it so that its normals are true wherever a counted wall point
        # can be (a patch cut at the clip had skewed normals on its ragged edge: the sign of b flipped there)
        clip_r = float(radius) + pr.wall_clip * k0 * spread
        wp = Surface(wall.vertices, wall.faces, centre, clip_r + WALL_MARGIN)
        ax_pts = _axis_points(axis, wp, centre, float(radius))
        solid = None
        if wall.is_watertight:
            from .junction_checks import shells_of
            solid = shells_of(wall)                  # inside ANY of its shells (a limb's skin and its deep mass overlap)
        ball = np.nonzero(((V - centre) ** 2).sum(axis=1) < (rad + 1.5) ** 2)[0]
        skin = Surface(V, F[topo.faces_of(ball)], centre, rad + 1.5, walk=0)
        centre = _exit_point(ax_pts, skin, centre)
        near = np.nonzero(((V - centre) ** 2).sum(axis=1) < rad * rad)[0]
        if len(near) < 8:
            return out
    except ValueError:
        return out
    # the skin within reach of the wall only: a nearest wall VERTEX farther than the reach plus the wall's longest
    # edge cannot have a wall point within the reach (the closest-triangle search is the expensive part)
    kmax = pr.region * k0 * spread                  # a vertex is in the region only if b < region * k
    # first by the distance to the root's centreline (the wall lies within ``r_wall`` of it near the root), then by
    # the nearest wall vertex
    wnear = wp.V[np.linalg.norm(wp.V - centre, axis=1) < clip_r + 0.5]
    r_wall = float(_axis_distance(ax_pts, wnear).max()) if len(wnear) else float(radius) * 2.0
    near = near[_axis_distance(ax_pts, V[near]) < r_wall + kmax + 0.05]
    if len(near) < 8:
        return out
    wv = wp.nearest_vertex(V[near])
    emax = float(np.linalg.norm(wp.V[wp.F[:, 1]] - wp.V[wp.F[:, 0]], axis=1).max()) if len(wp.F) else 0.0
    keep_near = np.linalg.norm(V[near] - wp.V[wv], axis=1) < kmax + emax + 0.05
    near, wv = near[keep_near], wv[keep_near]
    if len(near) < 8:
        return out
    wq, nb0, b0, gb = wp.closest(V[near], wv)
    na0, ga = skin.vertex_normals_of(V, near)
    kv = _reach(pr, V[near], na0, nb0, centre, float(radius), k0, spread)
    # only the wall near the root counts: skin whose closest wall point is farther out (a drooping tail beside the
    # thigh, an arm lying along the back) is not welded to it
    kv = np.where(np.linalg.norm(wq - centre, axis=1) < clip_r, kv, 0.0)
    if locked is not None and np.any(locked):
        kv, keep = _away_from(V, near, b0, kv, locked, pr)
    else:
        keep = np.ones(len(near), bool)
    for ow in others or ():
        if ow is None or not len(ow.faces):
            continue
        try:
            op = Surface(ow.vertices, ow.faces, centre, rad + 1.0, walk=1)
        except ValueError:
            continue
        _, _, bo, _ = op.closest(V[near])
        keep &= bo > 2.0 * pr.inset                 # under (or on) another root's wall: its plug, not this sleeve
    # the faces whose crossings decide (the region's and, around it, every earlier root's weld)
    ball_faces = topo.faces_of(near[np.asarray(locked, dtype=bool)[near]]) if locked is not None and np.any(locked) \
        else np.zeros(0, dtype=np.int64)
    t_setup = time.perf_counter() - t0
    best = None
    rounds = 0
    for rounds in range(pr.guard + 1):
        res = _solve(V, F, topo, near, keep, b0, nb0, na0, ga, gb, kv, skin, wp, pr, k0, ax_pts, solid)
        if res is None:
            break
        region, P = res["region"], res["P"]
        # the ground truth: the welded skin around the root must not cross itself where it did not before
        fi = np.union1d(topo.faces_of(region), ball_faces)
        trial = V.copy()
        trial[region] = P
        pairs = CO.crossing_pairs(trial, F[fi])
        # the junction may leave a few crossings of its own (a thick tail's cheeks): the weld must not add any
        old_pairs = CO.crossing_pairs(V, F[fi]) if len(pairs) else np.zeros((0, 2), dtype=np.int64)
        new_pairs = pairs
        if len(old_pairs) and len(pairs):
            key = lambda pp: {tuple(k) for k in np.sort(F[fi][pp].reshape(-1, 6), axis=1)}
            old_keys = key(old_pairs)
            keys = np.sort(F[fi][pairs].reshape(-1, 6), axis=1)
            new_pairs = pairs[np.array([tuple(k) not in old_keys for k in keys], dtype=bool)]
        res["crossings"] = max(0, int(len(pairs)) - int(len(old_pairs)))
        T = trial[F[fi]]
        area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
        res["degenerate"] = int((area < MIN_AREA).sum())
        res["crossings"] += res["degenerate"]
        res["crossings_before"] = int(len(old_pairs))
        bad_v = np.unique(F[fi][new_pairs].ravel()) if len(new_pairs) else np.zeros(0, dtype=np.int64)
        res["bad"] = bad_v
        score = (res["crossings"], len(res["flipped"]), len(res["folded"]))
        if best is None or score < best["score"]:
            best = res
            best["score"] = score
            best["trial"] = trial
        if not res["crossings"] and not len(res["flipped"]):
            break
        if rounds == pr.guard:
            break
        # shrink the reach around the crossing and the turned-over faces (and only there), then solve again
        w = np.zeros(len(near))
        w[res["region_pos"][res["flipped"]]] = 1.0
        pos = np.full(topo.n, -1, dtype=np.int64)
        pos[near] = np.arange(len(near))
        bp = pos[bad_v]
        w[bp[bp >= 0]] = 1.0
        w = _smooth_on(topo, near, w, 3) * 4.0
        kv = np.maximum(kv * (1.0 - GUARD_SHRINK * np.clip(w, 0.0, 1.0)), 1e-4)
    if best is None:
        if diag is not None:
            diag.update(info, seconds=time.perf_counter() - t0)
        return out
    region = best["region"]
    applied = best["crossings"] == 0
    if applied:
        out = best["trial"]
    info.update(applied=applied, guard=rounds, crossings=best["crossings"], folded=int(len(best["folded"])),
                crossings_before=best.get("crossings_before", 0),
                centre=centre,
                region=int(len(region)), seconds=time.perf_counter() - t0, setup=t_setup,
                kv_min=float(best["kv"].min()), kv_med=float(np.median(best["kv"])), edge=best["edge"])
    if diag is not None:
        diag.update(info)
        diag["ids"] = best["ids"]
        diag["kv"] = best["kv"]
    return out


def _axis_points(axis, wp, centre, radius) -> np.ndarray:
    """The root's centreline as points (inside the body first); without one, the long axis of the wall near the
    root, pointing away from the body (from ``centre`` toward the wall's mass)."""
    if axis is not None:
        A = np.asarray(axis, dtype=np.float64).reshape(-1, 3)
        if len(A) >= 2 and np.linalg.norm(A[-1] - A[0]) > 1e-6:
            return A
    W = wp.V[np.linalg.norm(wp.V - centre, axis=1) < 3.0 * radius + 0.5]
    if len(W) < 3:
        W = wp.V
    m = W.mean(axis=0)
    _, _, vt = np.linalg.svd(W - m, full_matrices=False)
    d = vt[0] if (vt[0] @ (m - centre)) >= 0 else -vt[0]
    return np.array([centre - d * radius, centre + d * 3.0 * radius])


def _exit_point(A, skin, fallback) -> np.ndarray:
    """Where the centreline ``A`` (inside the body first) comes out of the ``skin``; ``fallback`` if it does not."""
    L = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(A, axis=0), axis=1))])
    if L[-1] <= 0.0:
        return np.asarray(fallback, dtype=np.float64)
    s = np.linspace(0.0, L[-1], 96)
    pts = np.stack([np.interp(s, L, A[:, k]) for k in range(3)], axis=1)
    _, _, sd, _ = skin.closest(pts)
    inside = np.nonzero(sd < 0.0)[0]
    if not len(inside):
        return np.asarray(fallback, dtype=np.float64)
    after = np.nonzero((sd >= 0.0) & (np.arange(len(s)) > inside[0]))[0]
    if not len(after):
        return np.asarray(fallback, dtype=np.float64)
    j = after[0]
    f = sd[j - 1] / (sd[j - 1] - sd[j]) if sd[j - 1] != sd[j] else 0.5
    return pts[j - 1] + f * (pts[j] - pts[j - 1])


def _axis_distance(A, P) -> np.ndarray:
    """Distance of each of ``P`` to the polyline ``A``."""
    P = np.asarray(P, dtype=np.float64)
    out = np.full(len(P), np.inf)
    for i in range(len(A) - 1):
        a, b = A[i], A[i + 1]
        ab = b - a
        t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-18), 0.0, 1.0)
        out = np.minimum(out, np.linalg.norm(P - (a + t[:, None] * ab), axis=1))
    return out


def _axis_dir(A, P) -> np.ndarray:
    """The direction of the centreline ``A`` at the segment nearest to each of ``P``."""
    seg = A[1:] - A[:-1]
    L2 = np.maximum((seg * seg).sum(axis=1), 1e-18)
    rel = P[:, None, :] - A[None, :-1, :]
    t = np.clip((rel * seg[None]).sum(axis=2) / L2[None], 0.0, 1.0)
    d2 = ((rel - t[:, :, None] * seg[None]) ** 2).sum(axis=2)
    return CO._unit(seg[d2.argmin(axis=1)])


def _reach(pr, P, na, nb, centre, radius, k0, spread):
    """The fillet's reach ``k`` at the skin points ``P`` (normals ``na``; the wall's at their closest point:
    ``nb``): the kind's shape, less where the wall leans over the skin, fading away from the root."""
    c = (na * nb).sum(axis=1)
    kv = k0 * np.clip(1.0 + c, pr.acute, 1.0) * np.clip(pr.shape(P, centre), 0.05, 4.0)
    d = np.linalg.norm(P - centre, axis=1)
    d0 = radius + 0.5 * pr.region * k0 * spread
    d1 = radius + 1.25 * pr.region * k0 * spread
    x = np.clip((d - d0) / max(d1 - d0, 1e-9), 0.0, 1.0)
    return kv * (1.0 - x * x * (3.0 - 2.0 * x))


def _away_from(V, near, b0, kv, locked, pr):
    """An earlier root's weld is not touched: its vertices are left out and the reach ends before them."""
    L = np.nonzero(np.asarray(locked, dtype=bool))[0]
    P = V[near]
    lo, hi = P.min(axis=0) - 2.0, P.max(axis=0) + 2.0
    L = L[np.all((V[L] >= lo) & (V[L] <= hi), axis=1)]
    keep = ~np.asarray(locked, dtype=bool)[near]
    if not len(L):
        return kv, keep
    dmin = np.full(len(P), np.inf)
    Q = V[L]
    for s in range(0, len(P), 512):
        d = np.sqrt(((P[s:s + 512, None, :] - Q[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        dmin[s:s + 512] = d
    cap = 0.8 * (np.maximum(b0, 0.0) + dmin) / pr.region
    return np.minimum(kv, cap), keep


def _smooth_on(topo, ids, w, rounds):
    tab = topo.table(ids)
    loc = np.full(topo.n, -1, dtype=np.int64)
    loc[ids] = np.arange(len(ids))
    has = tab >= 0
    lt = np.where(has, loc[np.where(has, tab, 0)], -1)
    inn = lt >= 0
    safe = np.where(inn, lt, 0)
    cnt = np.maximum(inn.sum(axis=1), 1)
    for _ in range(rounds):
        w = 0.5 * w + 0.5 * (np.where(inn, w[safe], 0.0)).sum(axis=1) / cnt
    return w


def _solve(V, F, topo, near, keep, b0n, nb0n, na0n, gan, gbn, kvn, skin, wp, pr, k0, ax_pts, solid=None):
    """One construction with the reach ``kvn`` (per vertex of ``near``): ``None`` when there is nothing to weld,
    else the region, the new positions, the sets and the folded faces."""
    inr = keep & (kvn > 1e-6) & (b0n < pr.region * kvn)          # (skin under far wall: not this root's plug)
    region_pos = np.nonzero(inr)[0]
    region = near[region_pos]
    if len(region) < 8:
        return None
    b0, na0, ga, gb, kv = (x[region_pos] for x in (b0n, na0n, gan, gbn, kvn))
    n = topo.n
    loc = np.full(n, -1, dtype=np.int64)
    loc[region] = np.arange(len(region))
    tab = topo.table(region)
    has = tab >= 0
    lt = np.where(has, loc[np.where(has, tab, 0)], -1)
    inn = lt >= 0
    lsafe = np.where(inn, lt, 0)
    tsafe = np.where(has, tab, 0)
    cnt = np.maximum(has.sum(axis=1), 1)
    P0 = V[region]
    Vout = V[tsafe]
    elen = np.where(has, np.linalg.norm(V[tsafe] - P0[:, None, :], axis=2), np.inf)
    edge = float(np.median(elen[has]))
    P = P0.copy()

    def lap(X, mask=None):
        pts = np.where(inn[:, :, None], X[lsafe], Vout)
        m = (pts * has[:, :, None]).sum(axis=1) / cnt[:, None]
        return m if mask is None else m[mask]

    def smooth_scalar(x, rounds):
        for _ in range(rounds):
            xn = np.where(inn, x[lsafe], x[:, None])
            x = 0.5 * x + 0.5 * (xn * has).sum(axis=1) / cnt
        return x

    # -- the reach: no deeper than the skin's concave radius (a valley), no higher than the room over the skin
    conc = np.maximum(((lap(P0) - P0) * na0).sum(axis=1), 0.0)
    rho = edge * edge / np.maximum(4.0 * conc, 1e-9)
    room = np.full(len(region), np.inf)
    for s_ in (0.2, 0.4, 0.7, 1.0):
        hgt = s_ * kv
        _, _, sd, _ = skin.closest(P0 + hgt[:, None] * na0, ga)
        short = (sd < CLEAR_SHARE * hgt) & ~np.isfinite(room)
        room[short] = hgt[short]
    kv = np.minimum(kv, np.where(np.isfinite(room), CLEAR_K * room, kv))
    kv = np.minimum(kv, pr.valley * smooth_scalar(rho, 2))
    for _ in range(6):                      # smoothed without ever raising a value: a cut stays cut (the cleft)
        kv = np.minimum(kv, smooth_scalar(kv, 1))
    kv = np.maximum(kv, 1e-4)

    # -- the sets
    outer = (has & ~inn).any(axis=1)                       # touches a vertex outside the region
    cap = b0 < 0.0
    lab = _components(lt, inn, ~cap)                        # skin outside the wall enclosed by the cap: the plug's
    reach_out = np.zeros(len(region) + 1, bool)
    reach_out[lab[outer & ~cap] + 1] = True
    cap |= ~cap & ~reach_out[lab + 1]
    lf = loc[F[topo.faces_of(region)]]
    lrf = lf[(lf >= 0).all(axis=1)]                          # the triangles wholly in the region
    for _ in range(12):
        top = ~cap & (inn & cap[lsafe]).any(axis=1)
        all_top = top[lrf].all(axis=1)
        if not all_top.any():
            break
        tri = lrf[all_top]                                 # three corners on the last ring: laid flat on the wall
        cap[np.unique(tri[np.arange(len(tri)), np.argmin(b0[tri], axis=1)])] = True
    top = ~cap & (inn & cap[lsafe]).any(axis=1)
    rim = outer & ~cap & ~top
    sleeve = ~cap & ~top & ~rim
    cap1 = cap & (inn & top[lsafe]).any(axis=1)
    capin = cap & ~cap1
    if not top.any() or not sleeve.any():
        return None

    # -- the fields and the moves
    _, na_i, _, _ = skin.closest(P0, ga)
    _, nb_i, _, _ = wp.closest(P0, gb)
    t_i = nb_i - (nb_i * na_i).sum(axis=1)[:, None] * na_i
    tl = np.linalg.norm(t_i, axis=1)
    t_i = t_i / np.maximum(tl, 1e-9)[:, None]
    g_a, g_b = ga.copy(), gb.copy()
    Rv = np.full(len(region), pr.region)
    h_t = np.zeros(len(region))

    def fields(mask):
        idx = np.nonzero(mask)[0]
        _, na, a, va = skin.closest(P[idx], g_a[idx])
        g_a[idx] = va
        _, nb, b, vb = wp.closest(P[idx], g_b[idx])
        g_b[idx] = vb
        return a, na, b, nb

    def project(mask, rounds):
        for _ in range(rounds):
            a, na, b, nb = fields(mask)
            Rm = Rv[mask]
            f, hh = smin(a, b / Rm, kv[mask])
            g = na * hh[:, None] + nb * ((1.0 - hh) / Rm)[:, None]
            P[mask] -= (f / np.maximum((g * g).sum(axis=1), 0.05))[:, None] * g

    def restore(mask, rounds):
        """Along the surface, across the rings, until the blend's weight is the vertex's own ``h``."""
        for _ in range(rounds):
            a, na, b, nb = fields(mask)
            Rm = Rv[mask]
            f, hh = smin(a, b / Rm, kv[mask])
            g = CO._unit(na * hh[:, None] + nb * ((1.0 - hh) / Rm)[:, None])
            grad = nb / Rm[:, None] - na
            grad = grad - (grad * g).sum(axis=1)[:, None] * g
            gl = np.maximum((grad * grad).sum(axis=1), 1e-6)
            step = np.clip((h_t[mask] - hh) * 2.0 * kv[mask] / gl, -0.5 * edge, 0.5 * edge)
            free = (hh > 0.0) & (hh < 1.0)
            P[mask] += np.where(free, step, 0.0)[:, None] * grad
            P[mask] -= (f / np.maximum((g * g).sum(axis=1), 0.05))[:, None] * g

    def climb(mask, height, depth, rounds=6, start=None):
        """Onto the wall ``depth`` under it, then up the wall's GENERATOR (the root's axis laid on the wall) until
        the height over the original skin is ``height`` or it has climbed ``CLIMB_MAX`` of it: a vertex keeps its
        place round the root.  Returns the height reached."""
        idx = np.nonzero(mask)[0]
        if start is not None:
            P[idx] = start
        moved = np.zeros(len(idx))
        cap_climb = CLIMB_MAX * np.maximum(height, edge)
        for _ in range(rounds):
            a, na, b, nb = fields(mask)
            P[idx] -= (b + depth)[:, None] * nb
            gd = _axis_dir(ax_pts, P[idx])
            gd = CO._unit(gd - (gd * nb).sum(axis=1)[:, None] * nb)
            rate = (na * gd).sum(axis=1)
            step = np.clip((height - a) / np.maximum(rate, 0.25), -2.0 * edge, 2.0 * edge)
            step = np.clip(moved + step, -cap_climb, cap_climb) - moved
            moved += step
            P[idx] += step[:, None] * gd
        a, na, b, nb = fields(mask)
        P[idx] -= (b + depth)[:, None] * nb
        return a

    def along_ring(mask, factor=0.5, among=None):
        """Even the vertices out along their own ring (the direction in which neither field changes); ``among``:
        average over these neighbours only (the last ring: its own two)."""
        if among is None:
            L = lap(P, mask)
        else:
            w = (inn & among[lsafe]).astype(np.float64)[mask]
            L = (P[lsafe[mask]] * w[:, :, None]).sum(axis=1) / np.maximum(w.sum(axis=1), 1.0)[:, None]
            L = np.where((w.sum(axis=1) > 0)[:, None], L, P[mask])
        a, na, b, nb = fields(mask)
        c = np.cross(na, nb)
        cl = np.linalg.norm(c, axis=1)
        c = c / np.maximum(cl, 1e-9)[:, None]
        dd = ((L - P[mask]) * c).sum(axis=1)
        P[mask] += (factor * np.where(cl > 0.2, dd, 0.0))[:, None] * c

    # -- the last ring ON the wall at height k; where it cannot climb that high its reach is what it reached, and so
    # is the reach of the skin next to it (blending back to its own over the fillet's width)
    reached = climb(top, kv[top], 0.0, start=P0[top] - b0[top][:, None] * nb_i[top])
    kt_all = np.zeros(len(region))
    kt_all[top] = np.clip(np.minimum(kv[top], reached), 1e-4, None)
    ring = (inn & top[lsafe]) & top[:, None]                  # the last ring's own neighbours
    rc = np.maximum(ring.sum(axis=1), 1)
    for _ in range(max(8, int(top.sum()) // 3)):                 # smoothed along the ring without ever rising (as
        # many rounds as the ring is long: a level-2 ring has twice the vertices, and a zigzag of reached heights
        # left a ragged contact line on the wall)
        nbm = np.where(ring, kt_all[lsafe], 0.0).sum(axis=1) / rc
        kt_all = np.where(top & (ring.sum(axis=1) > 0), np.minimum(kt_all, 0.5 * kt_all + 0.5 * nbm), kt_all)
    if np.any(kt_all[top] < kv[top] - 1e-6):                   # lowered: climb to the new height
        climb(top, kt_all[top], 0.0, rounds=3)
    kt = kt_all[top]
    ti = np.nonzero(top)[0]
    dmin = np.full(len(region), np.inf)
    src = np.zeros(len(region), dtype=np.int64)
    for s0 in range(0, len(region), 512):
        d2 = ((P0[s0:s0 + 512, None, :] - P0[ti][None, :, :]) ** 2).sum(axis=2)
        j = d2.argmin(axis=1)
        src[s0:s0 + 512] = ti[j]
        dmin[s0:s0 + 512] = np.sqrt(d2[np.arange(len(j)), j])
    w = np.clip(dmin / np.maximum(pr.region * kv, 1e-6), 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)
    kv = np.where(cap | top, kv, np.minimum(kv, kt_all[src] + (kv - kt_all[src]) * w))
    kv[top] = kt
    for _ in range(4):
        kv = np.where(top | cap, kv, np.minimum(kv, smooth_scalar(kv, 1)))

    # -- the share of the way from the last ring to the rim (graph distances: monotone, no two rings on one spot) and
    # the landing: where the vertex's own path meets the rim, at most LAND_MAX nominal fillets out
    d_top = _graph_distance(lt, inn, elen, top, ~cap)
    d_rim = _graph_distance(lt, inn, elen, rim | (outer & ~cap), ~cap)
    d_rim = np.where(np.isfinite(d_rim), d_rim, np.maximum(pr.region * kv - b0, edge))
    land = smooth_scalar(np.maximum(b0, 0.0) + d_rim, 2)
    land = np.where(rim, np.maximum(b0, 1e-4), np.maximum(land, 1e-4))
    land = np.minimum(land, LAND_MAX * pr.region * kv)
    kv = np.minimum(kv, land / 1.2)
    Rv[:] = np.clip(land / kv, 1.2, _ASPECTS[-1])
    d_land = np.minimum(d_rim, np.maximum(land - np.maximum(b0, 0.0), 0.0))
    fin = np.isfinite(d_top) & np.isfinite(d_land)
    u = np.ones(len(region))
    u[fin] = d_top[fin] / np.maximum(d_top[fin] + d_land[fin], 1e-9)
    u = np.where(d_land <= 0.0, 1.0, np.clip(u, 0.0, 1.0))
    # the rings of a graph distance zigzag on a triangulated quad mesh (a diagonal is a short way to the last
    # ring): the share is smoothed over the sleeve, the last ring held at 0 and the rim at 1
    held = top | rim | cap | (d_land <= 0.0)
    for _ in range(U_SMOOTH):
        un = np.where(inn & ~cap[lsafe], u[lsafe], np.where(inn, 0.0, 1.0))
        cn = np.maximum((has & ~(inn & cap[lsafe])).sum(axis=1), 1)
        u = np.where(held, u, 0.5 * u + 0.5 * (un * (has & ~(inn & cap[lsafe]))).sum(axis=1) / cn)
    u = np.clip(u, 0.0, 1.0) ** pr.grade
    h_t[:] = h_of_u(u, Rv)

    # -- the sleeve: analytic placement on its profile, onto the zero set, its h restored, evened along its ring
    m = sleeve
    a_t = kv * (1.0 - h_t) ** 2
    b_t = Rv * kv * h_t * h_t
    slide = (b_t - b0) / np.maximum(tl, 0.5)
    P[m] = P0[m] + slide[m][:, None] * t_i[m] + a_t[m][:, None] * na_i[m]
    project(sleeve, PROJECT_ROUNDS)
    restore(sleeve, RESTORE_ROUNDS)
    for _ in range(pr.relax):
        along_ring(top, among=top)
        climb(top, kv[top], 0.0, rounds=2)
        along_ring(sleeve)
        if RESTORE_ROUNDS:
            restore(sleeve, 1)
        else:
            project(sleeve, 1)

    # -- cap1 continues up the wall just under it; the rest of the cap is a membrane spanning cap1
    if cap1.any():
        # just under its neighbours on the last ring (which are on the wall), deeper the deeper it was: the strip
        # between them turns into the wall at once and is hidden in it (climbing it up the wall as well folded it on
        # a dome, where the wall runs along the skin)
        tops = np.where(inn & top[lsafe], 1.0, 0.0)
        st = (P[lsafe] * tops[:, :, None]).sum(axis=1) / np.maximum(tops.sum(axis=1), 1.0)[:, None]
        # ... carried on a little the way the skin arrives (from the sleeve over the last ring): the strip from the
        # last ring to it continues the skin's own direction under the wall, so the last ring's vertex normals are
        # not turned by a strip folding straight back into the root (a dark serrated line along the contact)
        sls = np.where(inn & sleeve[lsafe], 1.0, 0.0)
        sm = (P[lsafe] * sls[:, :, None]).sum(axis=1) / np.maximum(sls.sum(axis=1), 1.0)[:, None]
        arr = np.where((sls.sum(axis=1) > 0)[:, None], CO._unit(P - sm), 0.0)
        arr = np.where(top[:, None], arr, 0.0)
        ad = (arr[lsafe] * tops[:, :, None]).sum(axis=1)
        ad = CO._unit(ad)
        P[cap1] = st[cap1] + CAP1_ON * edge * ad[cap1]
        a, na, b, nb = fields(cap1)
        P[cap1] -= (b + 2.0 * pr.inset)[:, None] * nb
        # a share of the way back to where it was (inside the wall already): two of them under the same two
        # neighbours on the ring were laid on one spot (zero-area triangles: the printed body was not watertight)
        P[cap1] = (1.0 - CAP1_BACK) * P[cap1] + CAP1_BACK * P0[cap1]
    if capin.any():
        _membrane(P, capin, lt, inn, has, Vout, cnt)
    # the whole plug inside the wall (a membrane spanning a ring on a wall that is not convex -- a dome with its arm
    # beside the flank -- bulged out of it on the heavy body)
    for _ in range(3):
        if not cap.any():
            break
        _, nb, b, _ = wp.closest(P[cap])
        # (deep in the root the sign of b is not to be trusted; at the faceted wall the smooth normal's sign is off by
        # a hair: under the wall by less than ``inset`` counts as on it)
        bad = (b > -pr.inset) & (b < 3.0 * edge)
        if not bad.any():
            break
        idx = np.nonzero(cap)[0][bad]
        P[idx] -= (b[bad] + 2.0 * pr.inset)[:, None] * nb[bad]
    if cap.any() and solid is not None:
        # and by ray parity against the closed wall: a point a hair outside a facet reads as under the wall along
        # the interpolated normal; it goes under the FACET it is nearest to
        cidx = np.nonzero(cap)[0]
        cidx = cidx[np.linalg.norm(P[cidx] - P0[cidx], axis=1) > 0.0]
        if len(cidx):
            out = ~solid.inside(P[cidx])
            if out.any():
                o = cidx[out]
                q, corner, _, _ = wp.closest_bary(P[o])
                T = wp.V[corner]
                fnrm = CO._unit(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]))
                P[o] = q - 2.0 * pr.inset * fnrm

    # -- folds: a visible face with a moved corner whose normal turned away from the blend's gradient
    mov = sleeve | top
    vf = lrf[(~cap[lrf]).all(axis=1) & mov[lrf].any(axis=1)]
    a, na, b, nb = fields(mov)
    _, hh = smin(a, b / Rv[mov], kv[mov])
    g = na0.copy()                                  # the rim did not move: the skin's own normal
    g[mov] = CO._unit(na * hh[:, None] + nb * ((1.0 - hh) / Rv[mov])[:, None])
    T = P[vf]
    fn = CO._unit(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]))
    gm = CO._unit(g[vf].sum(axis=1))
    dots = (fn * gm).sum(axis=1)
    folded = np.unique(vf[dots < FOLD_COS])
    flipped = np.unique(vf[dots < 0.0])
    ids = dict(region=region, sleeve=region[sleeve], top=region[top], cap=region[cap], cap1=region[cap1],
               fixed=region[rim])
    return {"region": region, "region_pos": region_pos, "P": P, "folded": folded, "flipped": flipped, "ids": ids,
            "kv": kv,
            "edge": edge}


def _membrane(P, mask, lt, inn, has, Vout, cnt, iters: int = 400, tol: float = 1e-7):
    """The vertices ``mask`` as the uniform-weight harmonic membrane spanning their neighbours (conjugate
    gradients on the graph Laplacian; every other vertex, in or out of the region, is held where it is)."""
    idx = np.nonzero(mask)[0]
    m = len(idx)
    pos = np.full(len(P), -1, dtype=np.int64)
    pos[idx] = np.arange(m)
    T = lt[idx]
    H = has[idx]
    In = inn[idx]
    free = In & (pos[np.where(In, T, 0)] >= 0)
    fpos = np.where(free, pos[np.where(In, T, 0)], 0)
    deg = H.sum(axis=1).astype(np.float64)
    # the right-hand side: the held neighbours' positions
    held = np.where((In & ~free)[:, :, None], P[np.where(In, T, 0)], 0.0)
    held = held + np.where((H & ~In)[:, :, None], Vout[idx], 0.0)
    rhs = held.sum(axis=1)

    def A(x):
        return deg[:, None] * x - np.where(free[:, :, None], x[fpos], 0.0).sum(axis=1)

    x = P[idx].copy()
    r = rhs - A(x)
    p = r.copy()
    rs = (r * r).sum(axis=0)
    for _ in range(iters):
        Ap = A(p)
        alpha = rs / np.maximum((p * Ap).sum(axis=0), 1e-30)
        x += alpha * p
        r -= alpha * Ap
        rn = (r * r).sum(axis=0)
        if float(rn.max()) < tol * tol:
            break
        p = r + (rn / np.maximum(rs, 1e-30)) * p
        rs = rn
    P[idx] = x


# --------------------------------------------------------------------------
# the pipelines


def weld_targets(placed, controls=None, enabled: bool = True) -> list:
    """``[WeldTarget]`` of every appendage of ``placed`` whose kind welds its root (``Kind.weld_target``), in
    the pose ``controls(p)`` (default: its own)."""
    if not enabled:
        return []
    out = []
    for p in placed or ():
        fn = getattr(p.kind, "weld_target", None)
        if fn is None:
            continue
        try:
            t = fn(p, None if controls is None else controls(p))
        except (ValueError, KeyError, IndexError, TypeError, np.linalg.LinAlgError) as exc:
            from .kinds import AppendageWarning
            warnings.warn(f"appendage {p.name!r}: no root weld ({exc})", AppendageWarning, stacklevel=2)
            continue
        if t is not None:
            out.append(t)
    return out


def weld_vertices(V, F, targets, *, diags: list | None = None) -> np.ndarray:
    """``V`` (MakeHuman frame) welded onto every target in turn; a later root leaves an earlier one's weld
    alone.  A root whose weld would make the skin cross itself is left as it was, with a warning."""
    V = np.asarray(V, dtype=np.float64)
    if not targets:
        return V
    topo = Topology.of(F, len(V))
    locked = np.zeros(len(V), bool)
    out = V
    for t in targets:
        d: dict = {}
        try:
            others = [o.wall for o in targets if o is not t]
            new = weld_skin(out, F, t.wall, t.centre, t.radius, t.profile, axis=t.axis, topo=topo, locked=locked,
                            others=others, diag=d)
        except (ValueError, IndexError, FloatingPointError, np.linalg.LinAlgError) as exc:
            new, d = out, {"applied": False, "error": str(exc)}
        d["name"] = t.name
        if diags is not None:
            diags.append(d)
        if d.get("applied"):
            moved = np.any(new != out, axis=1)
            locked |= moved
            if "ids" in d:
                locked[d["ids"]["region"]] = True
            out = new
        elif d.get("crossings", 0) or d.get("error"):
            from .kinds import AppendageWarning
            warnings.warn(f"appendage {t.name!r}: the skin is not welded onto the root here (the weld would "
                          "fold the skin); it meets the root as the junction left it", AppendageWarning, stacklevel=2)
    return out


def weld_mesh(mesh: Mesh, targets, *, z_up: bool = True, diags: list | None = None) -> Mesh:
    """The body ``mesh`` (Z up when ``z_up``, the frame the pipelines hold it in) welded onto ``targets``: a NEW
    mesh with the same triangles whose moved rows are the only ones that differ (the input is not touched; with
    no target it is returned as it is)."""
    if not targets:
        return mesh
    if z_up:
        V, F = CO.skin_from_shell(mesh)
    else:
        V, F = mesh.vertices, mesh.faces
    W = weld_vertices(V, F, targets, diags=diags)
    moved = np.nonzero(np.any(W != V, axis=1))[0]
    if not len(moved):
        return mesh
    verts = np.array(mesh.vertices, dtype=np.float64, copy=True)
    if z_up:
        from .portrait import MH_TO_Z_UP
        R, t = MH_TO_Z_UP[:3, :3], MH_TO_Z_UP[:3, 3]
        verts[moved] = W[moved] @ R.T + t
    else:
        verts[moved] = W[moved]
    return Mesh(verts, mesh.faces)


def weld_body(base, shaped_verts, placed, mesh_or_VF=None, controls=None, levels: int = 1, *,
              enabled: bool = True, diags: list | None = None):
    """The body welded onto every root of ``placed``: ``mesh_or_VF`` is the body as the pipeline meshed it (a
    Z-up :class:`Mesh`, returned as a Mesh) or ``(vertices, triangles)`` in the MakeHuman frame (returned as
    vertices); ``None`` meshes ``shaped_verts`` at ``levels`` (MakeHuman frame).  ``controls``: ``placed ->
    controls`` (the pose each appendage is drawn in)."""
    targets = weld_targets(placed, controls, enabled)
    if mesh_or_VF is None:
        V, F = CO.skin_from_cage(base, shaped_verts, levels)
        return weld_vertices(V, F, targets, diags=diags)
    if isinstance(mesh_or_VF, Mesh):
        return weld_mesh(mesh_or_VF, targets, diags=diags)
    V, F = mesh_or_VF
    return weld_vertices(V, F, targets, diags=diags)


def wants_weld(specs, enabled: bool = True) -> bool:
    """Does any enabled appendage of ``specs`` (records, or built ``Placed``) weld its root?  Costs nothing for a
    record (no appendage is built): its kind welds and its ``collar`` parameter (the kind's default when absent)
    is above 0."""
    if not enabled:
        return False
    from . import kinds as K
    for sp in specs or ():
        if hasattr(sp, "kind") and hasattr(sp, "anchor"):
            fn = getattr(sp.kind, "weld_profile", None)
            if fn is not None and fn(sp) is not None:
                return True
            continue
        if not getattr(sp, "enabled", True):
            continue
        kind = K.kind_of(sp)
        if kind is None or not kind.welds(dict(getattr(sp, "params", {}) or {}), sp):
            continue
        return True
    return False


def seam(character) -> str:
    """How the skin meets the appendages' roots in ``character``: ``"weld"`` (the default: this module),
    ``"collar"`` (the round-4 overlay shell, :mod:`.collar`: a document that says ``anatomy["collar"] =
    "shell"``, the command line's ``--collar-shell``), or ``"none"`` (``anatomy["collar"] = false``, the
    command line's ``--no-collar``, or the junction off: the bare root).  Never both."""
    if not getattr(character, "junction_enabled", True):
        return "none"
    v = getattr(character, "anatomy", {}).get("collar", True)
    if isinstance(v, str):
        return "collar" if v.strip().lower() == "shell" else ("none" if v.strip().lower() in ("", "false", "off", "none")
                                                              else "weld")
    return "weld" if bool(v) else "none"


def default_levels(character) -> int:
    """The subdivision a character (a :class:`~.sliders.Character`) is meshed at when none is asked for:
    ``MIN_LEVEL`` (1) when a root of its RECORDS is welded (or, with the old collar asked for, has a collar),
    else 0 (the cage).  The ONE decision the command line, the lab's final render and its export all use (they
    disagreed: the command line asked the built appendages, the lab the records; the ossicone came out at
    different levels).  Read from the records: nothing is built."""
    how = seam(character)
    specs = list(getattr(character, "appendages", ()) or ())
    if how == "weld":
        return MIN_LEVEL if wants_weld(specs) else 0
    if how == "collar":
        return CO.default_levels(specs, True)
    return 0


def welded_shell_mesh(mesh: Mesh, placed, character=None, *, controls=None, diags: list | None = None) -> Mesh:
    """The body ``mesh`` a pipeline built (Z up) welded onto the roots of ``placed`` when ``character``'s seam
    is the weld (``character=None``: always); the mesh itself otherwise."""
    if character is not None and seam(character) != "weld":
        return mesh
    return weld_mesh(mesh, weld_targets(placed, controls), diags=diags)


# --------------------------------------------------------------------------
# the measures (tests, docs/weld.md, the session's tables)


def arrival_report(V, F, wall, centre, radius, *, V0=None, tol: float = 2e-3, reach: float | None = None,
                   up=None) -> dict:
    """How the skin ``V`` (triangles ``F``) meets the root ``wall`` around ``centre``, from the geometry alone (it
    reads the bare junction and a welded skin alike).

    * ``angles`` / ``p50`` / ``p90`` / ``max``: degrees between the normal of every triangle that reaches the wall
      from outside (a corner on or under the wall, ``tol``, and a corner outside it) and the wall's normal where it
      reaches it: 0 = the skin arrives tangentially, 90 = it runs into the wall (a pipe in a plate);
      ``open_*`` / ``crease_*``: the same split by the side: with ``up`` (a kind's open side), the half of the
      ring ``up`` points to and the other half; without, the crease where the wall faces the skin (the ORIGINAL
      skin's normal there, ``V0``, against the wall's, below -0.2);
    * ``crossings``: pairs of triangles around the root that cross (``crossings_before``: of ``V0``);
    * ``plug_outside``: vertices that were under the wall in ``V0`` and are outside it now (more than ``tol``);
    * ``moved`` / ``moved_far``: how many vertices moved and the farthest from ``centre``."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    centre = np.asarray(centre, dtype=np.float64)
    rad = float(reach) if reach is not None else float(radius) + 2.5
    near = ((V - centre) ** 2).sum(axis=1) < rad * rad
    ids = np.nonzero(near)[0]
    fi = np.nonzero(near[F].all(axis=1))[0]
    wp = Surface(wall.vertices, wall.faces, centre, rad + 1.0)
    rep: dict = {}
    _, _, sd, _ = wp.closest(V[ids])
    s = np.full(len(V), np.inf)
    s[ids] = sd
    f = F[fi]
    cont = f[(s[f] <= tol).any(axis=1) & (s[f] > tol).any(axis=1)]
    T = V[cont]
    cr = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    area = 0.5 * np.linalg.norm(cr, axis=1)
    fn = CO._unit(cr)
    w = (s[cont] <= tol).astype(np.float64)
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1.0)
    pt = (T * w[:, :, None]).sum(axis=1)
    if len(pt):
        _, n, _, _ = wp.closest(pt)
        ang = np.degrees(np.arccos(np.clip((fn * n).sum(axis=1), -1.0, 1.0)))
        keep = area > 0.02 * float(np.median(area))
    else:
        n = np.zeros((0, 3))
        ang = np.zeros(0)
        keep = np.zeros(0, bool)
    crease = np.zeros(len(pt), bool)
    if up is not None and len(pt):
        crease = (CO._unit(pt - centre) @ CO._unit(np.asarray(up, dtype=np.float64))) < 0.0
    elif V0 is not None and len(pt):
        sk = Surface(np.asarray(V0, dtype=np.float64), F, centre, rad + 1.0, walk=0)
        _, na, _, _ = sk.closest(pt)
        crease = (na * n).sum(axis=1) < -0.2
    rep["angles"] = ang[keep]
    for tag, m in (("", keep), ("open_", keep & ~crease), ("crease_", keep & crease)):
        a = ang[m]
        rep[tag + "n"] = int(m.sum())
        for q, name in ((50, "p50"), (90, "p90"), (100, "max")):
            rep[tag + name] = float(np.percentile(a, q)) if len(a) else float("nan")
    rep["crossings"] = int(len(CO.crossing_pairs(V, f)))
    if V0 is not None:
        V0 = np.asarray(V0, dtype=np.float64)
        rep["crossings_before"] = int(len(CO.crossing_pairs(V0, f)))
        # the plug: inside any of the closed wall's shells by ray parity (deep in a root the sign of a closest-point
        # distance flips at the medial axis); the signed distance where the wall is not closed
        if wall.is_watertight:
            from .junction_checks import points_in_mesh
            _, _, s0, _ = wp.closest(V0[ids])
            was_in = points_in_mesh(V0[ids], wall.vertices, wall.faces) & (s0 < -tol)
            rep["plug_outside"] = int((was_in & ~points_in_mesh(V[ids], wall.vertices, wall.faces)).sum())
        else:
            _, _, s0, _ = wp.closest(V0[ids])
            rep["plug_outside"] = int(((s0 < -tol) & (sd > tol)).sum())
        moved = np.linalg.norm(V - V0, axis=1) > 1e-12
        rep["moved"] = int(moved.sum())
        rep["moved_far"] = float(np.linalg.norm(V[moved] - centre, axis=1).max()) if moved.any() else 0.0
    return rep
