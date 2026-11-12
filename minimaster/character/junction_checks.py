"""Geometry checks on a displaced body cage: triangle self-intersection, stretch.

The junction moves body cage vertices; this module says whether the result is still a
sane skin. Everything is pure numpy on ``(n, 3)`` vertices and ``(m, 4)`` quads (a cage).

* :func:`self_intersection_pairs` finds the pairs of triangles that cross each other
  (every quad is split ``(0,1,2) (0,2,3)``), optionally only those with a triangle in a
  vertex ``region``; a uniform grid keeps the cost proportional to the region, not the body.
* :func:`stretch_report` measures what a displacement did to the cage: per-quad area ratio,
  per-edge length ratio and the Lipschitz ratio of the displacement (how differently two
  neighbouring vertices moved, per edge length).
* :func:`ring_distance`, :func:`vertex_ring`, :func:`share_a_quad`: how far a vertex is from a set,
  in rings of quads, and whether two vertices are neighbours.

Used by ``junction.limit_displacement`` (to keep a displacement valid) and by
``junction.junction_quality`` / the tests (to measure that it is).
"""

from __future__ import annotations

import threading

import numpy as np

_EPS = 1e-12
_LOCK = threading.Lock()


def _cross(a, b) -> np.ndarray:
    out = np.empty_like(a)
    out[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    out[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    out[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return out


def triangulate(quads) -> np.ndarray:
    """The ``(2m, 3)`` triangles of ``m`` quads: triangle ``t`` and ``t + m`` are quad ``t``."""
    q = np.asarray(quads)
    return np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])


def segment_hits_triangle(p0, p1, a, b, c, eps: float = 1e-9) -> np.ndarray:
    """Moller-Trumbore, vectorised over rows: does the OPEN segment ``p0 -> p1`` pierce
    the triangle ``a b c`` (strictly inside both)?"""
    e1, e2 = b - a, c - a
    d = p1 - p0
    h = _cross(d, e2)
    det = (e1 * h).sum(axis=1)
    ok = np.abs(det) > 1e-14
    inv = 1.0 / np.where(ok, det, 1.0)
    s = p0 - a
    u = inv * (s * h).sum(axis=1)
    qv = _cross(s, e1)
    v = inv * (d * qv).sum(axis=1)
    t = inv * (e2 * qv).sum(axis=1)
    return ok & (u > eps) & (v > eps) & (u + v < 1.0 - eps) & (t > eps) & (t < 1.0 - eps)


# --------------------------------------------------------------------------
# broad phase: a uniform grid over triangle bounding boxes


def _cells(lo, hi, cell, origin, max_span=6):
    """Grid keys of every cell the boxes ``lo..hi`` overlap: ``(tri_index, key, clamped)``
    (a box that spans more than ``max_span`` cells along an axis is clamped to its first
    cells and ``clamped`` is True: the pair test afterwards still checks the real boxes, but
    the pairs are then de-duplicated the slow way)."""
    i0 = np.floor((lo - origin) / cell).astype(np.int64)
    i1 = np.floor((hi - origin) / cell).astype(np.int64)
    full = i1 - i0 + 1
    span = np.minimum(full, max_span)
    n = span[:, 0] * span[:, 1] * span[:, 2]
    tri = np.repeat(np.arange(len(lo)), n)
    start = np.cumsum(n) - n
    k = np.arange(int(n.sum())) - np.repeat(start, n)
    sx = span[tri, 0]
    sy = span[tri, 1]
    cx = i0[tri, 0] + k % sx
    cy = i0[tri, 1] + (k // sx) % sy
    cz = i0[tri, 2] + k // (sx * sy)
    return tri, _key(cx, cy, cz), bool((full > max_span).any())


def _key(cx, cy, cz):
    return (cx + (1 << 20)) * (1 << 42) + (cy + (1 << 20)) * (1 << 21) + (cz + (1 << 20))


def overlapping_pairs(lo, hi, rows, others=None, grow: float = 0.0):
    """Index pairs ``(i, j)`` with ``i`` in ``rows``, ``j`` in ``others`` (default: all
    boxes), ``i != j``, whose boxes ``lo..hi`` (grown by ``grow``) overlap. A pair of two
    ``rows`` boxes appears once per order; the caller de-duplicates if it matters."""
    rows = np.asarray(rows)
    if not len(rows):
        return np.zeros((0, 2), dtype=np.int64)
    ext = (hi - lo).max(axis=1)
    cell = max(float(np.percentile(ext[rows], 90)), 1e-6) * 1.5 + 2.0 * grow
    origin = lo.min(axis=0) - 1e-6
    others = np.arange(len(lo)) if others is None else np.asarray(others)
    ti, ki, ci = _cells(lo[rows] - grow, hi[rows] + grow, cell, origin)
    tj, kj, cj = _cells(lo[others] - grow, hi[others] + grow, cell, origin)
    order = np.argsort(kj, kind="stable")
    kj_s, tj_s = kj[order], others[tj[order]]
    left = np.searchsorted(kj_s, ki, side="left")
    right = np.searchsorted(kj_s, ki, side="right")
    cnt = right - left
    tot = int(cnt.sum())
    if not tot:
        return np.zeros((0, 2), dtype=np.int64)
    ii = np.repeat(rows[ti], cnt)
    start = np.cumsum(cnt) - cnt
    jj = tj_s[np.arange(tot) - np.repeat(start, cnt) + np.repeat(left, cnt)]
    kk = np.repeat(ki, cnt)
    keep = ii != jj
    ii, jj, kk = ii[keep], jj[keep], kk[keep]
    ok = (np.all(lo[ii] - grow <= hi[jj] + grow, axis=1)
          & np.all(lo[jj] - grow <= hi[ii] + grow, axis=1))
    ii, jj, kk = ii[ok], jj[ok], kk[ok]
    if ci or cj:                                   # a clamped box: de-duplicate the slow way
        n = len(lo)
        uniq = np.unique(ii.astype(np.int64) * n + jj)
        return np.stack([uniq // n, uniq % n], axis=1)
    # a pair shares several cells; keep it only in the cell of the corner of its box overlap
    corner = np.floor((np.maximum(lo[ii], lo[jj]) - grow - origin) / cell).astype(np.int64)
    first = kk == _key(corner[:, 0], corner[:, 1], corner[:, 2])
    return np.stack([ii[first], jj[first]], axis=1)


# --------------------------------------------------------------------------
# self-intersection


def _region_mask(region, n):
    reg = np.asarray(region)
    if reg.dtype != bool:
        mask = np.zeros(n, dtype=bool)
        mask[reg] = True
        reg = mask
    return reg


class IntersectionTester:
    """Self-intersection of a cage whose vertices MOVE within a known margin: the
    triangles near the region are selected once (a box around the region grown by ``pad``),
    and :meth:`pairs` then tests only those for any vertex positions within ``pad`` of the
    original ones. ``region``: a vertex mask or indices; only triangles with a vertex in it
    are tested (against all the selected triangles)."""

    def __init__(self, verts, quads, region, pad: float = 0.0):
        v = np.asarray(verts, dtype=np.float64)
        t_all = triangulate(quads)
        reg = _region_mask(region, len(v))
        self.empty = not reg.any()
        self.t = t_all[:0]
        if self.empty:
            return
        rv = v[reg]
        lo, hi = rv.min(axis=0), rv.max(axis=0)
        e = np.linalg.norm(v[t_all[:, 0]] - v[t_all[:, 1]], axis=1)
        margin = float(np.percentile(e, 99.5)) + 2.0 * pad
        near_v = np.all((v >= lo - margin) & (v <= hi + margin), axis=1)
        sel = near_v[t_all].any(axis=1)
        self.t = t_all[sel]
        self.rows = np.nonzero(reg[self.t].any(axis=1))[0]
        self.all_idx = np.arange(len(self.t))
        self.index = np.nonzero(sel)[0]                    # selected triangle -> triangle of the cage

    def pairs(self, verts) -> np.ndarray:
        """``(k, 2)`` pairs of crossing triangles (indices into the cage's triangle list)."""
        if self.empty or not len(self.rows):
            return np.zeros((0, 2), dtype=np.int64)
        v = np.asarray(verts, dtype=np.float64)
        t = self.t
        P = v[t]
        lo, hi = P.min(axis=1), P.max(axis=1)
        pairs = overlapping_pairs(lo, hi, self.rows, self.all_idx)
        if not len(pairs):
            return pairs
        share = (t[pairs[:, 0]][:, :, None] == t[pairs[:, 1]][:, None, :]).any(axis=(1, 2))
        pairs = pairs[~share]
        if not len(pairs):
            return pairs
        A, B = P[pairs[:, 0]], P[pairs[:, 1]]
        hit = np.zeros(len(pairs), dtype=bool)
        for k in range(3):
            hit |= segment_hits_triangle(A[:, k], A[:, (k + 1) % 3], B[:, 0], B[:, 1], B[:, 2])
            hit |= segment_hits_triangle(B[:, k], B[:, (k + 1) % 3], A[:, 0], A[:, 1], A[:, 2])
        out = np.sort(self.index[pairs[hit]], axis=1)
        return np.unique(out, axis=0) if len(out) else out.reshape(0, 2)


def self_intersection_pairs(verts, quads, region=None, tris=None) -> np.ndarray:
    """The pairs of triangles of the cage that cross each other: ``(k, 2)`` triangle indices
    ``(a, b)`` with ``a < b`` (triangle ``t`` and ``t + len(quads)`` are quad ``t``).

    Two triangles that share a vertex never count (neighbours touch by construction).
    ``region``: a boolean mask over the vertices or the indices of vertices; only triangles
    with a vertex in it are tested (against ALL triangles near them); ``None``: everything
    (slow on a whole body)."""
    v = np.asarray(verts, dtype=np.float64)
    if region is None:
        region = np.ones(len(v), dtype=bool)
    return IntersectionTester(v, quads, region).pairs(v)


def ring_distance(quads, mask, max_rings: int = 3) -> np.ndarray:
    """How many cage rings each vertex is from the vertices of ``mask`` (0 on them, 1 for the
    quads' other corners, ...; ``max_rings + 1`` beyond)."""
    q = np.asarray(quads)
    dist = np.full(len(np.asarray(mask)), max_rings + 1, dtype=np.int64)
    cur = np.array(mask, dtype=bool, copy=True)
    dist[cur] = 0
    for r in range(1, max_rings + 1):
        nxt = vertex_ring(q, cur)
        dist[nxt & ~cur] = r
        cur = nxt
    return dist


def pairs_cross(verts, tris, pairs) -> np.ndarray:
    """For each ``(a, b)`` of ``pairs`` (triangle indices into ``tris``): do the two triangles
    cross, with the vertices ``verts``?"""
    v = np.asarray(verts, dtype=np.float64)
    pairs = np.asarray(pairs)
    A, B = v[tris[pairs[:, 0]]], v[tris[pairs[:, 1]]]
    hit = np.zeros(len(pairs), dtype=bool)
    for k in range(3):
        hit |= segment_hits_triangle(A[:, k], A[:, (k + 1) % 3], B[:, 0], B[:, 1], B[:, 2])
        hit |= segment_hits_triangle(B[:, k], B[:, (k + 1) % 3], A[:, 0], A[:, 1], A[:, 2])
    return hit


_NEIGHBOUR_CACHE: dict = {}


def neighbour_keys(quads) -> np.ndarray:
    """Sorted keys ``a * n + b`` of every ordered pair of vertices that share a quad of the cage
    (cached per cage array); ``n`` is ``quads.max() + 1``."""
    q = np.asarray(quads)
    key = (q.ctypes.data, q.shape, int(q.sum()), int(q[::97].sum()))
    with _LOCK:
        hit = _NEIGHBOUR_CACHE.get(key)
    if hit is not None and hit[0] is q:
        return hit[1]
    n = int(q.max()) + 1
    keys = np.unique(np.concatenate([q[:, a].astype(np.int64) * n + q[:, b]
                                     for a in range(4) for b in range(4) if a != b]))
    if len(q) > 2000:
        with _LOCK:
            if len(_NEIGHBOUR_CACHE) > 8:
                _NEIGHBOUR_CACHE.clear()
            _NEIGHBOUR_CACHE[key] = (q, keys)
    return keys


def share_a_quad(quads, a, b) -> np.ndarray:
    """For vertex index arrays ``a`` and ``b``: does a quad of the cage hold both?"""
    q = np.asarray(quads)
    keys = neighbour_keys(q)
    n = int(q.max()) + 1
    k = np.asarray(a, dtype=np.int64) * n + np.asarray(b)
    i = np.minimum(np.searchsorted(keys, k), len(keys) - 1)
    return keys[i] == k


def vertex_ring(quads, mask) -> np.ndarray:
    """``mask`` grown by one ring: every vertex of any quad that has a vertex in ``mask``."""
    q = np.asarray(quads)
    out = np.array(mask, dtype=bool, copy=True)
    out[q[np.asarray(mask)[q].any(axis=1)].ravel()] = True
    return out


# --------------------------------------------------------------------------
# stretch


_EDGE_CACHE: dict = {}


def cage_edges(quads) -> np.ndarray:
    """The unique undirected edges ``(e, 2)`` of a quad cage (sorted; cached per cage array:
    the body's topology never changes)."""
    q = np.asarray(quads)
    key = (q.ctypes.data, q.shape, int(q.sum()), int(q[::97].sum()))
    with _LOCK:
        hit = _EDGE_CACHE.get(key)
    if hit is not None and hit[0] is q:
        return hit[1]
    a = q.ravel()
    b = np.roll(q, -1, axis=1).ravel()
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    n = int(q.max()) + 1
    k = np.unique(lo.astype(np.int64) * n + hi)
    out = np.stack([k // n, k % n], axis=1)
    if len(q) > 2000:                                       # only the big, long-lived cages
        with _LOCK:
            if len(_EDGE_CACHE) > 8:
                _EDGE_CACHE.clear()
            _EDGE_CACHE[key] = (q, out)
    return out


def stretch_report(old, new, quads, edges=None) -> dict:
    """What the move ``old -> new`` did to the cage, over the quads / edges it touched:

    * ``area_growth_max`` / ``area_ratio_min``: new / old quad area (the area vector's length);
    * ``edge_growth_max`` / ``edge_ratio_min``: new / old edge length;
    * ``lipschitz_max`` / ``lipschitz_p99``: ``|d_a - d_b| / |old edge|`` over the edges with
      a moved end (how differently two neighbours moved, per edge length: 1 means the
      edge may grow to twice its length);
    * ``moved`` / ``max_move``.
    """
    old = np.asarray(old, dtype=np.float64)
    new = np.asarray(new, dtype=np.float64)
    q = np.asarray(quads)
    d = new - old
    mag = np.linalg.norm(d, axis=1)
    moved = mag > 0
    out = {"moved": int(moved.sum()), "max_move": float(mag.max()) if len(mag) else 0.0,
           "area_growth_max": 1.0, "area_ratio_min": 1.0, "edge_growth_max": 1.0,
           "edge_ratio_min": 1.0, "lipschitz_max": 0.0, "lipschitz_p99": 0.0}
    if not moved.any():
        return out
    qi = np.nonzero(moved[q].any(axis=1))[0]
    a0 = np.linalg.norm(_cross(old[q[qi, 2]] - old[q[qi, 0]], old[q[qi, 3]] - old[q[qi, 1]]), axis=1)
    a1 = np.linalg.norm(_cross(new[q[qi, 2]] - new[q[qi, 0]], new[q[qi, 3]] - new[q[qi, 1]]), axis=1)
    r = a1 / np.maximum(a0, _EPS)
    out["area_growth_max"], out["area_ratio_min"] = float(r.max()), float(r.min())
    e = cage_edges(q) if edges is None else edges
    e = e[moved[e].any(axis=1)]
    l0 = np.linalg.norm(old[e[:, 0]] - old[e[:, 1]], axis=1)
    l1 = np.linalg.norm(new[e[:, 0]] - new[e[:, 1]], axis=1)
    er = l1 / np.maximum(l0, _EPS)
    out["edge_growth_max"], out["edge_ratio_min"] = float(er.max()), float(er.min())
    lip = np.linalg.norm(d[e[:, 0]] - d[e[:, 1]], axis=1) / np.maximum(l0, _EPS)
    out["lipschitz_max"], out["lipschitz_p99"] = float(lip.max()), float(np.percentile(lip, 99))
    return out


# --------------------------------------------------------------------------
# a tube against the closed shell(s) it stands for


def _rotation(a: float, b: float, c: float) -> np.ndarray:
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1.0]])
    ry = np.array([[cb, 0, sb], [0, 1.0, 0], [-sb, 0, cb]])
    rx = np.array([[1.0, 0, 0], [0, cc, -sc], [0, sc, cc]])
    return rz @ ry @ rx


# three rays in directions no mesh is aligned with (a loft's rings are planar and axis-aligned
# meshes are common: a ray along an axis grazes edges); the majority vote is the answer
_RAY_ROTATIONS = (_rotation(0.6180339, 1.3247179, 2.2360679), _rotation(2.7182818, 0.5772156, 1.4142135),
                  _rotation(1.6180339, 2.5029078, 0.3010299))


def face_components(faces, n_vertices: int | None = None) -> np.ndarray:
    """The connected shell (by shared vertices) each triangle belongs to: ``(m,)`` labels. An
    appendage is several closed shells that overlap each other (a wing: the arm, the membrane, the
    bones), so \"inside the mesh\" is inside any one of them, never the parity of all."""
    F = np.asarray(faces)
    if not len(F):
        return np.zeros(0, dtype=np.int64)
    n = int(F.max()) + 1 if n_vertices is None else int(n_vertices)
    lab = np.arange(n)
    a = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    b = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    while True:
        m = np.minimum(lab[a], lab[b])
        new = lab.copy()
        np.minimum.at(new, a, m)
        np.minimum.at(new, b, m)
        new = new[new]                                    # pointer jumping: log(diameter) rounds
        if np.array_equal(new, lab):
            break
        lab = new
    return lab[F[:, 0]]


def _inside_one_shell(P0, V0, F):
    votes = np.zeros(len(P0), dtype=np.int64)
    for R in _RAY_ROTATIONS:
        P = P0 @ R.T
        T = (V0 @ R.T)[F]                                       # (m, 3, 3)
        lo, hi = T.min(axis=1), T.max(axis=1)
        a, b, c = T[:, 0], T[:, 1], T[:, 2]
        e1, e2 = b - a, c - a
        det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]         # twice the signed xy area
        ok = np.abs(det) > 1e-14
        parity = np.zeros(len(P), dtype=np.int64)
        for s in range(0, len(P), 48):
            Ps = P[s:s + 48]
            cand = ((Ps[:, None, 0] >= lo[None, :, 0]) & (Ps[:, None, 0] <= hi[None, :, 0])
                    & (Ps[:, None, 1] >= lo[None, :, 1]) & (Ps[:, None, 1] <= hi[None, :, 1])
                    & (Ps[:, None, 2] < hi[None, :, 2]) & ok[None, :])
            i, j = np.nonzero(cand)
            if not len(i):
                continue
            rel = Ps[i, :2] - a[j, :2]
            u = (rel[:, 0] * e2[j, 1] - rel[:, 1] * e2[j, 0]) / det[j]
            v = (e1[j, 0] * rel[:, 1] - e1[j, 1] * rel[:, 0]) / det[j]
            z = a[j, 2] + u * e1[j, 2] + v * e2[j, 2]
            hit = (u >= 0) & (v >= 0) & (u + v < 1.0) & (z > Ps[i, 2])
            parity += np.bincount(s + i[hit], minlength=len(P))
        votes += (parity % 2).astype(np.int64)
    return votes >= 2


class Shells:
    """The closed shells of a triangle mesh (an appendage is several that overlap: a wing has the
    arm, the membrane, the finger bones, ...), with their boxes: \"inside the mesh\" is inside ANY
    shell, never the parity of all of them. Built once per mesh (:func:`face_components` and a
    box per shell), then asked :meth:`inside` / :meth:`clearance` for as many points as needed; a
    shell whose box does not hold a point is not looked at for it, so the cost follows the points'
    neighbourhood and not the mesh."""

    def __init__(self, vertices, faces, components=None):
        self.V = np.asarray(vertices, dtype=np.float64)
        F = np.asarray(faces)
        lab = face_components(F, len(self.V)) if components is None else np.asarray(components)
        order = np.argsort(lab, kind="stable")
        ls = lab[order]
        self.starts = np.nonzero(np.concatenate([[True], ls[1:] != ls[:-1]]))[0] if len(ls) else np.zeros(0, int)
        self.stops = np.concatenate([self.starts[1:], [len(ls)]]) if len(ls) else np.zeros(0, int)
        self.F = F[order]
        if len(self.F):
            T = self.V[self.F]
            self.lo = np.minimum.reduceat(T.min(axis=1), self.starts)
            self.hi = np.maximum.reduceat(T.max(axis=1), self.starts)
        else:
            self.lo = self.hi = np.zeros((0, 3))

    def __len__(self):
        return len(self.starts)

    def faces(self, k):
        return self.F[self.starts[k]:self.stops[k]]

    def near(self, P, pad: float = 0.0):
        """The shells whose box (grown by ``pad``) holds at least one of the points ``P``."""
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        if not len(P) or not len(self):
            return []
        hit = ((P[:, None, :] >= self.lo[None] - pad) & (P[:, None, :] <= self.hi[None] + pad)).all(axis=2)
        return [int(k) for k in np.nonzero(hit.any(axis=0))[0]]

    def inside(self, P) -> np.ndarray:
        """Which of the points lie inside ANY shell (ray parity along three oblique directions, the
        majority wins; a point exactly on a surface is either)."""
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        out = np.zeros(len(P), dtype=bool)
        for k in self.near(P):
            sel = np.nonzero(~out & np.all((P >= self.lo[k]) & (P <= self.hi[k]), axis=1))[0]
            if len(sel):
                out[sel] = _inside_one_shell(P[sel], self.V, self.faces(k))
        return out

    def clearance(self, P, reach: float = 1.0) -> np.ndarray:
        """For each point, the radius of the largest ball around it that lies inside one shell
        (``0`` for a point inside none), up to ``reach``: the distance to that shell's surface."""
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        out = np.zeros(len(P))
        for k in self.near(P, reach):
            sel = np.nonzero(np.all((P >= self.lo[k]) & (P <= self.hi[k]), axis=1))[0]
            if not len(sel):
                continue
            ins = _inside_one_shell(P[sel], self.V, self.faces(k))
            if ins.any():
                d = distance_to_mesh(P[sel[ins]], self.V, self.faces(k), reach)
                out[sel[ins]] = np.maximum(out[sel[ins]], d)
        return out

    def section_radius(self, P, T, reach: float = 1.0) -> np.ndarray:
        """For each point ``P[k]`` and unit direction ``T[k]``, the radius of the largest circle
        around the point in the plane perpendicular to ``T[k]`` that lies inside one shell (``0`` for
        a point inside none), up to ``reach``: the distance from the point to the curve in which the
        plane cuts that shell. Unlike :meth:`clearance` it does not mind a cap of the shell that
        lies within a radius along the direction: a tube's section needs room across it, and its
        end ball may stand a little out of the shell, where nothing but the shell's own end is."""
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        T = np.asarray(T, dtype=np.float64).reshape(-1, 3)
        out = np.zeros(len(P))
        for k in self.near(P, reach):
            F = self.faces(k)
            sel = np.nonzero(np.all((P >= self.lo[k]) & (P <= self.hi[k]), axis=1))[0]
            if not len(sel):
                continue
            ins = _inside_one_shell(P[sel], self.V, F)
            tri = self.V[F]
            for m in sel[ins]:
                p, t = P[m], T[m]
                near = np.all((tri.max(axis=1) >= p - reach) & (tri.min(axis=1) <= p + reach), axis=1)
                tt = tri[near]
                sd = (tt - p) @ t                                   # signed distance of each corner
                pos = sd > 0.0                                      # (a corner on the plane counts as below)
                cross = pos.any(axis=1) & ~pos.all(axis=1)
                tt, sd, pos = tt[cross], sd[cross], pos[cross]
                if not len(tt):
                    continue
                # the plane cuts two of the triangle's three edges: the segment between those points
                pts = []
                for i, j in ((0, 1), (1, 2), (2, 0)):
                    hit = pos[:, i] != pos[:, j]
                    u = sd[:, i] / np.where(hit, sd[:, i] - sd[:, j], 1.0)
                    pts.append((hit, tt[:, i] + u[:, None] * (tt[:, j] - tt[:, i])))
                (h0, p0), (h1, p1), (h2, p2) = pts
                A = np.where(h0[:, None], p0, p1)
                B = np.where((h0 & h1)[:, None], p1, p2)
                ab = B - A
                w = np.clip(((p - A) * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), _EPS), 0.0, 1.0)
                d = np.linalg.norm(p - (A + w[:, None] * ab), axis=1)
                out[m] = max(out[m], min(float(d.min()), reach))
        return out


_SHELLS_CACHE: list = []


def shells_of(mesh) -> "Shells":
    """The :class:`Shells` of a :class:`~minimaster.core.mesh.Mesh`, kept for the last few meshes
    (an appendage's static mesh is asked about again and again; the key is the identity of its
    arrays, which are held so that an id cannot be reused while the entry lives)."""
    v, f = mesh.vertices, mesh.faces
    with _LOCK:
        for entry in _SHELLS_CACHE:
            if entry[0] is v and entry[1] is f:
                return entry[2]
    sh = Shells(v, f)
    with _LOCK:
        _SHELLS_CACHE.append((v, f, sh))
        del _SHELLS_CACHE[:-4]
    return sh



def points_in_mesh(points, vertices, faces, components=None) -> np.ndarray:
    """Which of ``points`` lie inside the closed triangle mesh (inside any of its shells:
    :class:`Shells`)."""
    return Shells(vertices, faces, components).inside(points)


def _closest_on_triangles(p, a, b, c) -> np.ndarray:
    """Distance from each point ``p[k]`` to its own triangle ``(a[k], b[k], c[k])`` (Ericson,
    Real-Time Collision Detection 5.1.5), vectorised over rows."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    out = np.empty_like(p)
    done = np.zeros(len(p), dtype=bool)

    def put(mask, q):
        m = mask & ~done
        out[m] = q[m]
        done[m] = True

    put((d1 <= 0) & (d2 <= 0), a)
    put((d3 >= 0) & (d4 <= d3), b)
    vc = d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        v = d1 / (d1 - d3)
        put((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + v[:, None] * ab)
        put((d6 >= 0) & (d5 <= d6), c)
        vb = d5 * d2 - d1 * d6
        w = d2 / (d2 - d6)
        put((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + w[:, None] * ac)
        va = d3 * d6 - d5 * d4
        w2 = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + w2[:, None] * (c - b))
        den = va + vb + vc
        vv, ww = vb / den, vc / den
        put(np.ones(len(p), dtype=bool), a + ab * vv[:, None] + ac * ww[:, None])
    return np.linalg.norm(p - out, axis=1)


def distance_to_mesh(points, vertices, faces, reach: float = 1.0) -> np.ndarray:
    """The distance of each of ``points`` to the surface of the triangle mesh, up to ``reach``
    (``reach`` where nothing is nearer): only triangles whose box, grown by ``reach``, holds the
    point are measured, so the cost follows the points' neighbourhoods and not the mesh."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    V = np.asarray(vertices, dtype=np.float64)
    T = V[np.asarray(faces)]
    lo, hi = T.min(axis=1) - reach, T.max(axis=1) + reach
    out = np.full(len(P), float(reach))
    for s in range(0, len(P), 64):
        Ps = P[s:s + 64]
        cand = np.all((Ps[:, None, :] >= lo[None]) & (Ps[:, None, :] <= hi[None]), axis=2)
        i, j = np.nonzero(cand)
        if not len(i):
            continue
        d = _closest_on_triangles(Ps[i], T[j, 0], T[j, 1], T[j, 2])
        best = np.full(len(Ps), float(reach))
        np.minimum.at(best, i, d)
        out[s:s + 64] = best
    return out


def tube_stations(points, radii, per_segment: int = 2):
    """The stations along a tube's centreline where it is looked at: every point of the
    polyline and ``per_segment - 1`` between each pair: ``(centres, radii, tangents, owners)``,
    ``owners`` a ``(stations, points)`` boolean matrix saying which polyline points a station
    depends on (a point's own station: that point; a station between two: both)."""
    pts = np.asarray(points, dtype=np.float64)
    rad = np.asarray(radii, dtype=np.float64)
    n = len(pts)
    tang = np.empty_like(pts)
    tang[1:-1] = pts[2:] - pts[:-2]
    tang[0], tang[-1] = pts[1] - pts[0], pts[-1] - pts[-2]
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), _EPS)
    k = np.concatenate([(np.arange(n - 1)[:, None] + (np.arange(per_segment) / per_segment)[None, :]).ravel(),
                        [n - 1.0]])
    i0 = np.minimum(np.floor(k + 1e-9).astype(int), n - 2)
    f = (k - i0)[:, None]
    centres = pts[i0] + f * (pts[i0 + 1] - pts[i0])
    radius = rad[i0] + f[:, 0] * (rad[i0 + 1] - rad[i0])
    tangent = tang[i0] + f * (tang[i0 + 1] - tang[i0])
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), _EPS)
    owners = np.zeros((len(k), n), dtype=bool)
    rows = np.arange(len(k))
    owners[rows, i0] = f[:, 0] < 1.0 - 1e-9
    owners[rows, i0 + 1] = f[:, 0] > 1e-9
    owners[-1, :] = False
    owners[-1, n - 1] = True
    return centres, radius, tangent, owners


def _ring_basis(tangent):
    ref = np.where(np.abs(tangent[:, :1]) < 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
    u = np.cross(tangent, ref)
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), _EPS)
    return u, np.cross(tangent, u)


def tube_ring_samples(centres, radius, tangent, sides: int = 12) -> np.ndarray:
    """``(stations, sides, 3)``: points on the tube's surface, a ring perpendicular to the
    centreline at every station."""
    u, w = _ring_basis(tangent)
    ang = (np.arange(sides) + 0.5) * (2.0 * np.pi / sides)
    return (centres[:, None, :] + radius[:, None, None]
            * (np.cos(ang)[None, :, None] * u[:, None, :] + np.sin(ang)[None, :, None] * w[:, None, :]))


def tube_in_mesh(points, radii, vertices=None, faces=None, sides: int = 12, per_segment: int = 2,
                 shells=None) -> dict:
    """How much of a tube (a polyline with a radius at each point, :class:`~.junction.JunctionSpec`)
    lies inside the closed mesh it stands for: the skin that "is hidden in the tube" must really be
    hidden by the appendage, and the slicer's union only hides what is inside it.

    A ring of ``sides`` points on the tube's surface is tested at every station
    (:func:`tube_stations`). Returns ``inside`` (the fraction of the ring samples at each station that
    are inside the mesh, ``(stations,)``), ``centre`` (is the station's centre itself inside),
    ``owners`` (see :func:`tube_stations`), ``fraction`` (the mean of ``inside``), ``worst`` (the
    smallest) and ``max_outside`` (how far the ring sample that is farthest out of the mesh is from
    its surface: 0 for a tube inside it)."""
    sh = Shells(vertices, faces) if shells is None else shells
    centres, radius, tangent, owners = tube_stations(points, radii, per_segment)
    ring = tube_ring_samples(centres, radius, tangent, sides)
    flat = np.concatenate([ring.reshape(-1, 3), centres])
    ins = sh.inside(flat)
    k = len(centres)
    frac = ins[:k * sides].reshape(k, sides).mean(axis=1)
    out_pts = flat[:k * sides][~ins[:k * sides]]
    depth = float(distance_to_mesh(out_pts, sh.V, sh.F, reach=float(np.max(radii))).max()) if len(out_pts) else 0.0
    return {"inside": frac, "centre": ins[k * sides:], "owners": owners,
            "fraction": float(frac.mean()), "worst": float(frac.min()), "max_outside": depth}


FIT_HUG = 0.98             # the fitted radius is this fraction of the room the mesh leaves
FIT_SLACK = 1.06           # a tube is left alone where it is no fatter than this times the room
FIT_MIN = 0.2              # an end point with less than this fraction of its radius to spare is no tube
FIT_FLOOR = 0.02           # ... and a point in the middle keeps at least this fraction (a wire, not a hole)


def fit_tube_to_mesh(points, radii, vertices=None, faces=None, shells=None):
    """``(points, radii)`` of the tube cut down to what really lies inside the closed mesh it
    stands for. The radius at a station is at most the room across the mesh there
    (:meth:`Shells.section_radius`: the circle in the plane perpendicular to the centreline that
    lies inside the shell, times ``FIT_HUG``); a point whose centre is not inside the mesh, or that
    has less than ``FIT_MIN`` of its radius to spare, is DROPPED when it is at an end of the
    polyline (the sunk end of a cut limb that has no mesh: a tube cannot be hidden by a shell that
    is not there) and given ``FIT_FLOOR`` of its radius when it is in the middle. A tube that already
    fits (within ``FIT_SLACK``) comes back exactly as it was."""
    pts = np.asarray(points, dtype=np.float64)
    rad = np.asarray(radii, dtype=np.float64)
    n = len(pts)
    sh = Shells(vertices, faces) if shells is None else shells
    reach = 1.5 * float(rad.max())
    tang = np.empty_like(pts)
    tang[1:-1] = pts[2:] - pts[:-2]
    tang[0], tang[-1] = pts[1] - pts[0], pts[-1] - pts[-2]
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), _EPS)
    mid = 0.5 * (pts[1:] + pts[:-1])
    mt = pts[1:] - pts[:-1]
    mt /= np.maximum(np.linalg.norm(mt, axis=1, keepdims=True), _EPS)
    room = sh.section_radius(np.concatenate([pts, mid]), np.concatenate([tang, mt]), reach)
    at, between = room[:n], room[n:]
    allow = np.minimum(rad, FIT_HUG * at)
    # the ends first: a point whose centre is outside the shell or that has no room is not part of the
    # tube, and the segment to it (half of it is outside) must not squeeze the points that are
    keep = np.ones(n, dtype=bool)
    lo, hi = 0, n - 1
    while lo < hi and (at[lo] <= 0 or allow[lo] < FIT_MIN * rad[lo]):
        keep[lo] = False
        lo += 1
    while hi > lo and (at[hi] <= 0 or allow[hi] < FIT_MIN * rad[hi]):
        keep[hi] = False
        hi -= 1
    # then the ball half-way between two points (the mean of their radii) must fit too
    scale = np.ones(n)
    if hi > lo:
        r_mid = 0.5 * (allow[lo + 1:hi + 1] + allow[lo:hi])
        squeeze = np.minimum(1.0, FIT_HUG * between[lo:hi] / np.maximum(r_mid, 1e-12))
        np.minimum(scale[lo + 1:hi + 1], squeeze, out=scale[lo + 1:hi + 1])
        np.minimum(scale[lo:hi], squeeze, out=scale[lo:hi])
    allow = allow * scale
    if keep.all() and (rad <= FIT_SLACK * np.maximum(allow, 0.0)).all() and (at > 0).all():
        return pts, rad
    out = np.maximum(allow, FIT_FLOOR * rad)
    return pts[keep], np.minimum(rad, out)[keep]


# --------------------------------------------------------------------------
# a patch of the cage, subdivided on its own


def subdivide_patch(verts, quads) -> tuple:
    """One Catmull-Clark level of an OPEN patch of a cage (``core.subdiv`` wants a closed one):
    ``(vertices, quads, exact)``, laid out like ``core.subdiv`` (the moved originals, the face points,
    the edge points). A vertex or edge of the patch's rim does not have its whole neighbourhood, so
    the vertex stays where it is and the edge point is the midpoint: ``exact`` is a bool mask of the
    new vertices that are what the subdivision of the whole body gives (the rest is the rim; look
    only at what lies a ring or more inside). ``verts`` may be a LIST of vertex arrays (the same
    patch moved in different ways): the topology is worked out once and ``vertices`` is then a
    list too."""
    many = isinstance(verts, (list, tuple))
    vs = [np.asarray(v, dtype=np.float64) for v in (verts if many else [verts])]
    F = np.asarray(quads)
    V, nf = len(vs[0]), len(F)
    a, b = F, np.roll(F, -1, axis=1)
    und = np.sort(np.stack([a.ravel(), b.ravel()], axis=1), axis=1)
    key = und[:, 0].astype(np.int64) * V + und[:, 1]
    ukey, first, inv = np.unique(key, return_index=True, return_inverse=True)
    uniq = und[first]
    inv = inv.ravel()
    E = len(uniq)
    ecnt = np.bincount(inv, minlength=E)
    interior_e = ecnt == 2
    flat = F.ravel()
    Qc = np.maximum(np.bincount(flat, minlength=V).astype(float), 1.0)
    Rc = (np.bincount(uniq[:, 0], minlength=V) + np.bincount(uniq[:, 1], minlength=V)).astype(float)
    rim_v = np.zeros(V, dtype=bool)
    rim_v[uniq[~interior_e].ravel()] = True
    n = np.maximum(Rc, 1.0)
    face_of = np.repeat(np.arange(nf), 4)
    outs = []
    for verts in vs:
        fp = verts[F].mean(axis=1)
        efp = np.stack([np.bincount(inv, weights=fp[face_of, c], minlength=E) for c in range(3)], axis=1)
        mid = (verts[uniq[:, 0]] + verts[uniq[:, 1]]) / 2.0
        epts = np.where(interior_e[:, None], (verts[uniq[:, 0]] + verts[uniq[:, 1]] + efp) / 4.0, mid)
        Qs = np.stack([np.bincount(flat, weights=np.repeat(fp[:, c], 4), minlength=V) for c in range(3)], axis=1)
        Rs = np.stack([np.bincount(uniq[:, 0], weights=mid[:, c], minlength=V)
                       + np.bincount(uniq[:, 1], weights=mid[:, c], minlength=V) for c in range(3)], axis=1)
        vpts = (Qs / Qc[:, None] + 2.0 * (Rs / n[:, None]) + (n - 3.0)[:, None] * verts) / n[:, None]
        outs.append(np.vstack([np.where(rim_v[:, None], verts, vpts), fp, epts]))
    eoff, foff = V + nf, V
    edge_id = inv.reshape(nf, 4)
    fcenter = foff + np.arange(nf)
    out = np.concatenate([np.stack([F[:, i], eoff + edge_id[:, i], fcenter, eoff + edge_id[:, (i - 1) % 4]], axis=1)
                          for i in range(4)]).astype(np.int64)
    exact = np.ones(len(outs[0]), dtype=bool)
    exact[:V] = ~rim_v
    exact[eoff:] = interior_e
    return (outs if many else outs[0]), out, exact


# --------------------------------------------------------------------------
# what a seam looks like in the picture (round 4: the collar's outline and its contact line)


def _box_blur(img: np.ndarray, r: int) -> np.ndarray:
    """A (2r+1)-wide box blur of a 2-D array with edge replication, by cumulative sums."""
    out = img.astype(np.float64)
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r + 1, r)
        p = np.pad(out, pad, mode="edge")
        c = np.cumsum(p, axis=axis)
        hi = np.take(c, np.arange(2 * r + 1, c.shape[axis]), axis=axis)
        lo = np.take(c, np.arange(0, c.shape[axis] - 2 * r - 1), axis=axis)
        out = (hi - lo) / (2 * r + 1)
    return out


def high_pass(rgb: np.ndarray, r: int = 2) -> np.ndarray:
    """Luminance minus a near-Gaussian blur of it (three box blurs of radius ``r``): edges and fine tone
    steps, the measure the round-3 critics read the collar's outline with."""
    g = np.asarray(rgb, dtype=np.float64)[..., :3].mean(axis=2)
    b = g
    for _ in range(3):
        b = _box_blur(b, r)
    return g - b


def seam_report(shells, collar_ids, body_id: int = 0, wall_ids=(), *, size=(320, 320), azimuth=180.0,
                elevation=10.0, fit=None) -> dict:
    """How the seams of the root collars read in one picture (``shells``: portrait shells; ``collar_ids``: the
    indices of the collar shells, ``body_id`` the body's, ``wall_ids`` the appendages').

    * ``outline_hp``: the median high-pass magnitude (:func:`high_pass`) on the pixels where the collar's
      footprint meets the body's skin -- its outline on the skin: a sticker's outline is a tone step and a
      serrated z-fight there; ``skin_hp``: the same on the skin 4 to 10 pixels away from every seam and
      silhouette (the skin's own detail); ``outline_ratio`` the first over the second (about 1: the outline
      cannot be told from the skin around it);
    * ``wall_jag``: of the image rows and columns that cross the line where the collar meets an appendage, the
      share that cross it more than once within 3 pixels (a sawtooth or z-fighting contact line);
    * ``outline_px`` / ``wall_px``: how many pixels those lines are long in this view."""
    from . import portrait as po
    img = po.render_portrait(shells, size=size, azimuth=azimuth, elevation=elevation, fit=fit)
    vb = po.visibility_buffer(shells, size, azimuth, elevation, fit=fit)
    sid = vb.shell
    col = np.isin(sid, np.asarray(list(collar_ids), dtype=np.int64))
    body = sid == body_id
    wall = np.isin(sid, np.asarray(list(wall_ids), dtype=np.int64)) if len(wall_ids) else np.zeros_like(body)
    bg = sid < 0

    def nb(mask):
        out = np.zeros_like(mask)
        out[1:] |= mask[:-1]
        out[:-1] |= mask[1:]
        out[:, 1:] |= mask[:, :-1]
        out[:, :-1] |= mask[:, 1:]
        return out

    def grow(mask, k):
        for _ in range(k):
            mask = mask | nb(mask)
        return mask

    hp = np.abs(high_pass(img))
    depth = vb.depth
    # where the footprint meets the skin ON the surface (the two pixels at about the same depth), not where the
    # body passes in front of the collar or the collar's own silhouette stands over skin behind it
    z_tol = 0.004 * float(np.nanmedian(np.where(np.isfinite(depth), depth, np.nan)))
    outline = np.zeros_like(col)
    for sh_ in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        o_c = np.roll(col, sh_, axis=(0, 1))
        o_b = np.roll(body, sh_, axis=(0, 1))
        o_z = np.roll(depth, sh_, axis=(0, 1))
        with np.errstate(invalid="ignore"):
            same = np.abs(o_z - depth) < z_tol
        outline |= (col & o_b & same) | (body & o_c & same)
    outline &= ~grow(bg | wall, 3)                          # the silhouettes and the wall's line are not the outline
    seams = outline | (col & nb(wall)) | (wall & nb(col)) | (bg & nb(~bg))
    near = grow(seams, 10) & ~grow(seams, 4) & (body | col) & ~grow(bg, 6)
    # medians: a handful of pixels where a run of the collar ends in the cleft beside the root carry the cleft's
    # own crease; the outline is the long line round the footprint
    rep = {"outline_px": int(outline.sum()), "outline_hp": float(np.median(hp[outline])) if outline.any() else 0.0,
           "skin_hp": float(np.median(hp[near])) if near.any() else 0.0}
    rep["outline_ratio"] = rep["outline_hp"] / max(rep["skin_hp"], 1e-9)
    # the contact line with the appendage: runs of collar / wall along rows and columns
    jag = cross = 0
    for a in (sid, sid.T):
        c_ = np.isin(a, np.asarray(list(collar_ids), dtype=np.int64))
        w_ = np.isin(a, np.asarray(list(wall_ids), dtype=np.int64)) if len(wall_ids) else np.zeros_like(c_)
        state = np.where(c_, 1, np.where(w_, 2, 0))
        for row in state:
            idx = np.nonzero((row[1:] != row[:-1]) & (row[1:] > 0) & (row[:-1] > 0))[0]
            if not len(idx):
                continue
            cross += len(idx)
            if len(idx) > 1:
                jag += int(np.sum(np.diff(idx) <= 3)) + 0
    rep["wall_px"] = int(cross)
    rep["wall_jag"] = float(jag) / max(cross, 1)
    return rep
