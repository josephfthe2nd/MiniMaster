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
