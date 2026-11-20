"""A hair system: strands as data, the operations on them, and the meshes made of them.

A STRAND is a curve that grows from a root: a hair, a barb of a feather, a whisker, a blade of grass.  All
the strands of one system have the same number of points, so a system is one array (``Strands.points``,
``(N, K, 3)``) and every operation is one vectorised step over all of them.  Nothing here knows what the
strands grow on: roots and headings come from a particle system (``core/particles.py``) or from anywhere
else, and the body, the wing or the feather that uses them lives in ``minimaster/character``.

Data and meshing are separate.  The operations make and change strands:

``grow``        strands leave their roots along a heading and bend as they go; ``steer`` is the hook for
                forces and collision (a callback that turns the heading at every step)
``trim``        a haircut: every strand is cut where it leaves a region (a signed margin function), and
                resampled along what is left
``between``     children: strands interpolated between consecutive guides
``clump``       strands of a group drawn towards the group's middle strand
``roughen``     repeatable noise on length and direction

and the meshers turn them into something drawn or printed:

``tube_mesh``   one closed tube per strand (hair, whiskers, fur): honest, and expensive
``sheet_mesh``  the strands ZIPPED: consecutive strands are the rows of one surface, thickened into a closed
                slab.  This is what a feather's vane is (barbs hooked to their neighbours), and it costs two
                triangles a segment where a tube costs twelve.  Alternate rows raised a little make every
                strand a ridge; the far face is a coarse sheet of its own (every second row, three columns:
                a sixth of the near face's triangles), so the closed slab takes about two thirds of what one
                with two fine faces would.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .mesh import Mesh

__all__ = ["Strands", "grow", "cut_length", "trim", "between", "clump", "roughen", "arclength", "sample", "hash01",
           "tube_mesh", "sheet_mesh", "sheet_meshes"]


@dataclass
class Strands:
    """``N`` strands of ``K`` points each (root first).  ``attrs`` holds anything else per strand, as arrays
    of length ``N`` (the length a trim kept, the group a strand clumps with, the guide it came from)."""

    points: np.ndarray
    attrs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float64)
        if self.points.ndim != 3 or self.points.shape[2] != 3:
            raise ValueError(f"strands are (N, K, 3) points, not {self.points.shape}")

    def __len__(self) -> int:
        return len(self.points)

    @property
    def steps(self) -> int:
        return self.points.shape[1]

    def lengths(self) -> np.ndarray:
        return arclength(self.points)[:, -1]

    def take(self, index) -> "Strands":
        return Strands(self.points[index], {k: np.asarray(v)[index] for k, v in self.attrs.items()})


def hash01(*key) -> np.ndarray:
    """Repeatable numbers in 0..1 from integers (arrays broadcast): the same on every run and machine, with
    no random state to carry about.  What every "random" variation here is drawn from."""
    x = np.zeros(np.broadcast(*[np.asarray(k) for k in key]).shape, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in key:
            x = x ^ (np.asarray(k).astype(np.int64).astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
                     + (x << np.uint64(6)) + (x >> np.uint64(2)))
            x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)


def _turn(v, axis, angle):
    """``v`` (..., 3) turned about ``axis`` (..., 3), unit, by ``angle`` (...) radians (Rodrigues)."""
    v = np.asarray(v, dtype=np.float64)
    axis = np.broadcast_to(np.asarray(axis, dtype=np.float64), v.shape)
    angle = np.asarray(angle, dtype=np.float64)
    c, s = np.cos(angle)[..., None], np.sin(angle)[..., None]
    return v * c + np.cross(axis, v) * s + axis * (axis * v).sum(axis=-1, keepdims=True) * (1.0 - c)


def grow(roots, heading, length, steps: int = 8, bend=0.0, bend_axis=(0.0, 1.0, 0.0), steer=None,
         attrs: dict | None = None) -> Strands:
    """Strands of ``steps`` segments that leave ``roots`` (N, 3) along ``heading`` (N, 3) and are ``length``
    long (a number, or one per strand).

    ``bend`` (radians, a number or one per strand) is how far a strand has turned about ``bend_axis`` by its
    end, evenly along its length: a barb sweeping towards the feather's tip, a hair curling over.
    ``steer(points, heading, travelled) -> heading`` is called before every step with where the strands are,
    which way they point and how far they have come: gravity, wind, a comb and collision are all a ``steer``
    (push ``points`` out of a body in place and return the heading slid along its skin)."""
    P = np.array(roots, dtype=np.float64)
    n = len(P)
    d = _unit(np.broadcast_to(np.asarray(heading, dtype=np.float64), (n, 3)).copy())
    L = np.broadcast_to(np.asarray(length, dtype=np.float64), (n,))
    b = np.broadcast_to(np.asarray(bend, dtype=np.float64), (n,)) / max(int(steps), 1)
    axis = _unit(np.broadcast_to(np.asarray(bend_axis, dtype=np.float64), (n, 3)))
    pts = np.empty((n, int(steps) + 1, 3))
    pts[:, 0] = P
    ds = L / max(int(steps), 1)
    if steer is None:
        # nothing steers them: every strand is an arc, and all its chords are known at once
        turned = (np.arange(int(steps)) + 0.5)[None, :] * b[:, None]
        chords = _turn(d[:, None, :], axis[:, None, :], turned) * ds[:, None, None]
        pts[:, 1:] = P[:, None, :] + np.cumsum(chords, axis=1)
        return Strands(pts, dict(attrs or {}))
    d = _turn(d, axis, 0.5 * b)                                   # (the chord of the first arc)
    for k in range(1, int(steps) + 1):
        d = _unit(np.asarray(steer(P, d, ds * (k - 1)), dtype=np.float64))
        P = P + d * ds[:, None]
        pts[:, k] = P
        d = _turn(d, axis, b)
    return Strands(pts, dict(attrs or {}))


def arclength(points) -> np.ndarray:
    """``(N, K)``: how far along its strand each point is."""
    seg = np.linalg.norm(np.diff(np.asarray(points, dtype=np.float64), axis=1), axis=2)
    return np.concatenate([np.zeros((len(seg), 1)), np.cumsum(seg, axis=1)], axis=1)


def sample(points, at) -> np.ndarray:
    """Points at the lengths ``at`` (N, C) along the strands ``points`` (N, K, 3): ``(N, C, 3)``."""
    P = np.asarray(points, dtype=np.float64)
    s = arclength(P)
    at = np.clip(np.asarray(at, dtype=np.float64), 0.0, s[:, -1:])
    k = np.clip((at[:, :, None] >= s[:, None, 1:]).sum(axis=2), 0, P.shape[1] - 2)       # the segment each is in
    rows = np.arange(len(P))[:, None]
    s0, s1 = s[rows, k], s[rows, k + 1]
    f = ((at - s0) / np.maximum(s1 - s0, 1e-300))[:, :, None]
    return P[rows, k] * (1.0 - f) + P[rows, k + 1] * f


def cut_length(strands: Strands, margin) -> np.ndarray:
    """How long each strand is where it first leaves a region: ``margin(points) -> (M,)`` is positive inside
    the region the strands may fill and negative outside (it is given all the points at once, strand after
    strand).  A strand that never leaves is all its own length."""
    P = strands.points
    n, K, _ = P.shape
    m = np.asarray(margin(P.reshape(-1, 3)), dtype=np.float64).reshape(n, K)
    s = arclength(P)
    out = m < 0.0
    first = np.where(out.any(axis=1), out.argmax(axis=1), K)       # the first point outside (K: none)
    rows = np.arange(n)
    j = np.clip(first, 1, K - 1)
    m0, m1 = m[rows, j - 1], m[rows, j]
    f = np.clip(m0 / np.maximum(m0 - m1, 1e-300), 0.0, 1.0)        # where between them the margin is 0
    return np.where(first >= K, s[:, -1], np.where(first == 0, 0.0, s[rows, j - 1] + f * (s[rows, j] - s[rows, j - 1])))


def trim(strands: Strands, margin, at=(0.0, 0.5, 1.0), scale=1.0) -> Strands:
    """A haircut: every strand is cut where it first leaves the region ``margin`` describes
    (:func:`cut_length`), and resampled at the fractions ``at`` ((C,), or (N, C): each strand its own) of
    what is left, times ``scale`` (a number or one per strand: a strand cut shorter still than the region
    allows).  The length kept is ``attrs["kept"]``."""
    kept = cut_length(strands, margin) * np.broadcast_to(np.asarray(scale, dtype=np.float64), (len(strands),))
    at = np.asarray(at, dtype=np.float64)
    at = at[None, :] if at.ndim == 1 else at
    return Strands(sample(strands.points, kept[:, None] * at), {**strands.attrs, "kept": kept})


def between(guides: Strands, per_gap: int) -> Strands:
    """Children: ``per_gap`` strands interpolated in every gap between consecutive guides (the guides stay,
    in order).  For guides that leave a line (barbs along a shaft) or a row of roots (a fringe); a field of
    hair interpolates between the guides nearest each child instead, which is a weighted sum of ``take``s."""
    P = guides.points
    if per_gap <= 0 or len(P) < 2:
        return guides
    f = (np.arange(per_gap + 1) / (per_gap + 1.0))[None, :, None, None]
    mid = (P[:-1, None] * (1.0 - f) + P[1:, None] * f).reshape(-1, P.shape[1], 3)
    pts = np.concatenate([mid, P[-1:]])
    attrs = {"guide": np.concatenate([np.repeat(np.arange(len(P) - 1), per_gap + 1), [len(P) - 1]])}
    return Strands(pts, attrs)


def clump(strands: Strands, group, amount=0.5, power: float = 1.5) -> Strands:
    """The strands of each ``group`` (an integer per strand) drawn towards the group's mean strand: not at
    all at the root, ``amount`` (0..1) of the way at the tip.  What makes hair locks and what closes the
    barbs of a feather into the pieces a split vane hangs in."""
    P = strands.points
    g = np.asarray(group)
    ids, inv = np.unique(g, return_inverse=True)
    mean = np.zeros((len(ids),) + P.shape[1:])
    np.add.at(mean, inv, P)
    mean /= np.bincount(inv, minlength=len(ids))[:, None, None]
    t = (np.linspace(0.0, 1.0, P.shape[1]) ** power)[None, :, None]
    a = np.broadcast_to(np.asarray(amount, dtype=np.float64), (len(P),))[:, None, None]
    return Strands(P + (mean[inv] - P) * a * t, {**strands.attrs, "group": g})


def roughen(strands: Strands, amount: float, seed: int = 0, waves: float = 2.0) -> Strands:
    """Repeatable unevenness: every strand is displaced by up to ``amount`` (a length) across itself, in a
    wave of its own phase that grows from nothing at the root."""
    P = strands.points
    n, K, _ = P.shape
    i = np.arange(n)
    tan = _unit(np.gradient(P, axis=1))
    ref = np.where(np.abs(tan[..., 1:2]) < 0.9, [[[0.0, 1.0, 0.0]]], [[[1.0, 0.0, 0.0]]])
    a = _unit(np.cross(tan, ref))
    b = np.cross(tan, a)
    t = np.linspace(0.0, 1.0, K)[None, :]
    ph = 2.0 * np.pi * (waves * t + hash01(i, seed, 1)[:, None])
    r = amount * t * (0.5 + 0.5 * hash01(i, seed, 2)[:, None])
    ang = 2.0 * np.pi * hash01(i, seed, 3)[:, None]
    off = (np.sin(ph) * np.cos(ang))[..., None] * a + (np.sin(ph) * np.sin(ang))[..., None] * b
    return Strands(P + r[..., None] * off, dict(strands.attrs))


# --------------------------------------------------------------------------
# meshes


def tube_mesh(strands: Strands, radius, sides: int = 6, taper: float = 0.3) -> Mesh:
    """One closed tube per strand, all in one mesh: ``radius`` at the root (a number or one per strand),
    ``taper`` of it at the tip, a fan over each end.  ``2 * sides`` triangles a segment: for hair that is
    seen strand by strand; for anything that lies together, zip it (:func:`sheet_mesh`)."""
    P = strands.points
    n, K, _ = P.shape
    r0 = np.broadcast_to(np.asarray(radius, dtype=np.float64), (n,))
    r = r0[:, None] * (1.0 - (1.0 - taper) * np.linspace(0.0, 1.0, K)[None, :])
    tan = _unit(np.gradient(P, axis=1))
    ref = np.where(np.abs(tan[..., 1:2]) < 0.9, [[[0.0, 1.0, 0.0]]], [[[1.0, 0.0, 0.0]]])
    a = _unit(np.cross(tan, ref))
    b = np.cross(tan, a)
    ang = 2.0 * np.pi * np.arange(sides) / sides
    ring = (np.cos(ang)[None, None, :, None] * a[:, :, None, :] + np.sin(ang)[None, None, :, None] * b[:, :, None, :])
    V = (P[:, :, None, :] + r[:, :, None, None] * ring).reshape(n, K * sides, 3)
    V = np.concatenate([V, P[:, :1], P[:, -1:]], axis=1)                       # ... and the two end centres
    j = np.arange(sides)
    jn = (j + 1) % sides
    F = []
    for k in range(K - 1):
        p, q = k * sides, (k + 1) * sides
        F += [np.stack([p + j, p + jn, q + jn], axis=1), np.stack([p + j, q + jn, q + j], axis=1)]
    root, tip = K * sides, K * sides + 1
    last = (K - 1) * sides
    F += [np.stack([np.full(sides, root), jn, j], axis=1), np.stack([np.full(sides, tip), last + j, last + jn], axis=1)]
    F = np.concatenate(F)
    per = K * sides + 2
    faces = (F[None] + (per * np.arange(n))[:, None, None]).reshape(-1, 3)
    m = Mesh(V.reshape(-1, 3), faces)
    return m if m.volume() >= 0 else Mesh(m.vertices, faces[:, ::-1].copy())


def _grid_normals(grid: np.ndarray) -> np.ndarray:
    """Unit normals of a grid of points ``(..., R, C, 3)`` (rows then columns)."""
    du = np.gradient(grid, axis=-3)
    dv = np.gradient(grid, axis=-2)
    return _unit(np.cross(dv, du))


def sheet_mesh(grid, fine, coarse, tip=None, tip_fine: float = 0.0, tip_coarse: float = 0.0,
               normals=None, keel: int | None = None) -> Mesh:
    """Strands zipped into one closed slab.

    ``grid`` (R, C, 3) is the surface the strands lie in: every row a strand (or two that meet: the barbs of
    a feather's two vanes, joined at the shaft, column ``keel``), consecutive rows consecutive strands.
    ``R`` is odd.  The FINE face stands ``fine`` (R, C) off the grid along its normals and has every row and
    column: raise the odd rows and every strand is a ridge.  The COARSE face lies ``coarse`` (R, C) off the
    other way and is made of the even rows and three columns only (the two edges and the keel): flat between
    them, a sixth of the fine face's triangles at seven columns.  The edges are stitched row for row, so the
    slab is closed and every edge is used twice.

    ``tip`` (a point) closes the last row to a point with a fan on each face instead of a wall (a feather
    ends in a point that its last barbs run into); ``tip_fine`` / ``tip_coarse`` are the two faces' offsets
    there.  ``normals`` (R, C, 3) replaces the grid's own."""
    return sheet_meshes(np.asarray(grid)[None], np.asarray(fine)[None], np.asarray(coarse)[None],
                        None if tip is None else np.asarray(tip)[None], tip_fine, tip_coarse,
                        None if normals is None else np.asarray(normals)[None], keel)[0]


def sheet_meshes(grids, fine, coarse, tips=None, tip_fine=0.0, tip_coarse=0.0, normals=None,
                 keel: int | None = None, both: bool = False) -> list:
    """:func:`sheet_mesh` for ``F`` sheets of one size at once (``grids`` (F, R, C, 3), ``fine`` and
    ``coarse`` (F, R, C), ``tips`` (F, 3), the tip offsets a number or (F,)): one pass of arithmetic over all
    of them, and one set of triangles shared by all.  A list of ``F`` meshes.

    ``both``: the far face is as fine as the near one (every row and column, ``coarse`` its offsets), for a
    sheet that is seen from both sides: ridges on both, for two thirds more triangles."""
    G = np.asarray(grids, dtype=np.float64)
    F_, R, C, _ = G.shape
    if R < 3 or R % 2 == 0 or C < 3:
        raise ValueError(f"a sheet needs an odd number of rows (3 or more) and 3 or more columns, not {R} x {C}")
    k = C // 2 if keel is None else int(keel)
    N = _grid_normals(G) if normals is None else np.asarray(normals, dtype=np.float64)
    top = G + N * np.asarray(fine, dtype=np.float64)[..., None]
    bot = G - N * np.asarray(coarse, dtype=np.float64)[..., None]
    if not both:
        bot = bot[:, ::2][:, :, [0, k, C - 1]]                                                       # (F, Re, 3, 3)
    V = [top.reshape(F_, -1, 3), bot.reshape(F_, -1, 3)]
    if tips is not None:
        tp = np.asarray(tips, dtype=np.float64)
        n_tip = _unit(N[:, R - 1].mean(axis=1))
        tf = np.broadcast_to(np.asarray(tip_fine, dtype=np.float64), (F_,))[:, None]
        tc = np.broadcast_to(np.asarray(tip_coarse, dtype=np.float64), (F_,))[:, None]
        V += [(tp + n_tip * tf)[:, None], (tp - n_tip * tc)[:, None]]
    V = np.concatenate(V, axis=1)
    faces = _sheet_faces(R, C, k, tips is not None, bool(both))
    # which way round: the triangles are wound for normals that point out of the fine face; flip a sheet
    # whose volume comes out negative (its normals were given the other way)
    a, b, c = V[:, faces[:, 0]], V[:, faces[:, 1]], V[:, faces[:, 2]]
    vol = (a * np.cross(b, c)).sum(axis=2).sum(axis=1)
    flipped = faces[:, ::-1]
    return [Mesh(V[i], faces if vol[i] >= 0 else flipped) for i in range(F_)]


def _slab_faces(R: int, C: int, tipped: bool) -> np.ndarray:
    """The triangles of a sheet with TWO fine faces: the near one (vertices ``r * C + c``), the far one
    (``R * C + r * C + c``), the two long edges and the root end as walls, and a wall or (``tipped``: two
    more vertices, last) a fan across the other end."""
    nt = R * C

    def T(r, c):
        return np.asarray(r) * C + np.asarray(c)

    def B(r, c):
        return nt + np.asarray(r) * C + np.asarray(c)

    F = []
    r, c = np.meshgrid(np.arange(R - 1), np.arange(C - 1), indexing="ij")
    r, c = r.ravel(), c.ravel()
    F += [np.stack([T(r, c), T(r, c + 1), T(r + 1, c + 1)], axis=1), np.stack([T(r, c), T(r + 1, c + 1), T(r + 1, c)], axis=1)]
    F += [np.stack([B(r, c), B(r + 1, c + 1), B(r, c + 1)], axis=1), np.stack([B(r, c), B(r + 1, c), B(r + 1, c + 1)], axis=1)]
    rr = np.arange(R - 1)
    for col, flip in ((0, False), (C - 1, True)):                    # the long edges
        a = [np.stack([T(rr, col), T(rr + 1, col), B(rr + 1, col)], axis=1),
             np.stack([T(rr, col), B(rr + 1, col), B(rr, col)], axis=1)]
        F += [x[:, ::-1] if flip else x for x in a]
    cc = np.arange(C - 1)

    def end(row, flip):
        a = [np.stack([T(row, cc + 1), T(row, cc), B(row, cc)], axis=1),
             np.stack([T(row, cc + 1), B(row, cc), B(row, cc + 1)], axis=1)]
        return [x[:, ::-1] if flip else x for x in a]

    F += end(0, False)
    if not tipped:
        F += end(R - 1, True)
    else:
        it, ib = 2 * nt, 2 * nt + 1
        F.append(np.stack([T(R - 1, cc), T(R - 1, cc + 1), np.full(C - 1, it)], axis=1))
        F.append(np.stack([B(R - 1, cc + 1), B(R - 1, cc), np.full(C - 1, ib)], axis=1))
        F.append(np.array([[T(R - 1, 0), it, ib], [T(R - 1, 0), ib, B(R - 1, 0)],
                           [it, T(R - 1, C - 1), ib], [T(R - 1, C - 1), B(R - 1, C - 1), ib]]))
    faces = np.concatenate(F).astype(np.int64)
    faces.setflags(write=False)
    return faces


@lru_cache(maxsize=256)
def _sheet_faces(R: int, C: int, k: int, tipped: bool, both: bool = False) -> np.ndarray:
    """The triangles of :func:`sheet_mesh` for a grid of ``R`` x ``C`` with its keel at column ``k``: the
    fine face (vertices ``r * C + c``), the coarse face (after them: even row ``e``, column 0 / 1 / 2 at
    ``R * C + 3 e + c``), the two long edges, the wall across the root end, and a wall or (``tipped``: the
    two extra vertices come last) a fan across the other.  The same for every feather of a size, so kept.
    ``both``: two fine faces (:func:`_slab_faces`)."""
    if both:
        return _slab_faces(R, C, tipped)
    Re = (R + 1) // 2
    nt = R * C

    def T(r, c):
        return np.asarray(r) * C + np.asarray(c)

    def B(e, c):
        return nt + np.asarray(e) * 3 + np.asarray(c)

    F = []
    r, c = np.meshgrid(np.arange(R - 1), np.arange(C - 1), indexing="ij")
    r, c = r.ravel(), c.ravel()
    F += [np.stack([T(r, c), T(r, c + 1), T(r + 1, c + 1)], axis=1), np.stack([T(r, c), T(r + 1, c + 1), T(r + 1, c)], axis=1)]
    e, c = np.meshgrid(np.arange(Re - 1), np.arange(2), indexing="ij")
    e, c = e.ravel(), c.ravel()
    F += [np.stack([B(e, c), B(e + 1, c + 1), B(e, c + 1)], axis=1), np.stack([B(e, c), B(e + 1, c), B(e + 1, c + 1)], axis=1)]
    # the two long edges: three fine rows over one coarse gap
    e = np.arange(Re - 1)
    for col, bc, flip in ((0, 0, False), (C - 1, 2, True)):
        a = [np.stack([T(2 * e, col), T(2 * e + 1, col), B(e, bc)], axis=1),
             np.stack([T(2 * e + 1, col), T(2 * e + 2, col), B(e + 1, bc)], axis=1),
             np.stack([T(2 * e + 1, col), B(e + 1, bc), B(e, bc)], axis=1)]
        F += [x[:, ::-1] if flip else x for x in a]

    def end(row, e_row, flip):
        """The wall across one end: the fine row's columns fanned onto the coarse row's three."""
        out = []
        for c0, c1, b0, b1 in ((0, k, 0, 1), (k, C - 1, 1, 2)):
            cc = np.arange(c0, c1)
            out.append(np.stack([T(row, cc + 1), T(row, cc), np.full(len(cc), B(e_row, b0))], axis=1))
            out.append(np.array([[T(row, c1), B(e_row, b0), B(e_row, b1)]]))
        return [x[:, ::-1] if flip else x for x in out]

    F += end(0, 0, False)
    if not tipped:
        F += end(R - 1, Re - 1, True)
    else:
        it, ib = nt + Re * 3, nt + Re * 3 + 1
        cc = np.arange(C - 1)
        F.append(np.stack([T(R - 1, cc), T(R - 1, cc + 1), np.full(C - 1, it)], axis=1))
        F.append(np.array([[B(Re - 1, 1), B(Re - 1, 0), ib], [B(Re - 1, 2), B(Re - 1, 1), ib]]))
        F.append(np.array([[T(R - 1, 0), it, ib], [T(R - 1, 0), ib, B(Re - 1, 0)],
                           [it, T(R - 1, C - 1), ib], [T(R - 1, C - 1), B(Re - 1, 2), ib]]))
    faces = np.concatenate(F).astype(np.int64)
    faces.setflags(write=False)
    return faces
