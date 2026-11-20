"""Strands: hair grown from points on the skin, meshed as locks a printer can make.

A single hair cannot be printed (at 32 mm a figure's head is 4 mm tall and the thinnest wall a resin printer
holds is about 0.2 mm), and it cannot be seen either: sculpted hair is a mass of LOCKS, each a tapered strip
with a crest, lying over one another.  So a strand here is the centreline of one lock:

``grow``        steps every strand at once from its root: it leaves the skin at ``lift``, turns towards the
                ground at ``droop`` radians per unit of length, and where it meets the body it is pushed out
                to its clearance and slides along the skin (:class:`~.appendages.SkinProbe` knows where the
                body is).  A wave or a curl is laid over the path afterwards.
``lock_meshes`` turns each centreline into a closed shell: a strip with a crest along its top and a flat
                underside that lies ON the skin (the frame follows the skin's normal), its root sunk into
                the body, its tip rounded off no thinner than ``tip_floor``.
``cap_mesh``    the undercoat: the region's own skin thickened outwards, thinning to nothing at its edge (a
                hairline, not a bathing cap), so no bare scalp shows between the locks and every lock has
                something to fuse to.

Every shell is closed and overlaps the body (the root and the underside are inside it), which is what the
print pipeline asks of an appendage's pieces; the slicer unions them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.mesh import Mesh

__all__ = ["Growth", "LockShape", "grow", "lock_meshes", "free_of_skin", "cap_mesh", "clean_patch", "DOWN",
           "UNDER"]

DOWN = np.array([0.0, -1.0, 0.0])            # the MakeHuman frame is Y up
NEAR_SKIN = 0.35                             # a lock nearer the skin than this lies flat on it (units)
UNDER = 0.31                                 # the share of a lock's thickness that is below its centreline


@dataclass(frozen=True)
class Growth:
    """How the strands of one head of hair grow (lengths in body units)."""

    segments: int = 12
    lift: float = 20.0          # degrees between the hair and the skin where it leaves it
    droop: float = 2.5          # how fast it turns towards the ground (radians per unit of length)
    clearance: float = 0.02     # height of a lock's centreline over the skin it lies on (its underside is
                                # ``UNDER`` of its thickness below that: keep it under the skin)
    rise: float = 0.0           # clearance gained per unit of length travelled: the hair's volume
    wave: float = 0.0           # sideways wave: amplitude (units)
    wave_length: float = 1.2    # length of one wave (units)
    curl: float = 0.0           # ringlets: radius of the helix (units)
    curl_pitch: float = 0.8     # length of one turn (units)
    settle: float = 0.25        # the share of the length over which wave and curl fade in from the root


@dataclass(frozen=True)
class LockShape:
    """The cross-section of a lock and how it changes along the lock."""

    width: float = 0.30         # across the lock, at its widest (units)
    thickness: float = 0.17     # crest to underside, at its thickest
    sides: int = 12             # vertices round the section (12: the grooved section; fewer: a plain oval)
    root: float = 0.55          # the width at the root over the widest
    tip: float = 0.20           # the width at the tip over the widest
    taper: float = 2.2          # how late the taper comes (1: a straight wedge)
    tip_floor: float = 0.14     # no part of a lock is narrower or thinner than this (0.26 mm at 32 mm)
    sink: float = 0.06          # how deep the root starts under the skin


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def grow(roots, normals, flow, lengths, growth: Growth = Growth(), probe=None, phase=None, bias=None):
    """Centrelines of ``N`` strands: ``(points, ups, height)``: points and ups ``(N, segments + 1, 3)``,
    ``height`` ``(N, segments + 1)`` each point's height over the skin (see :func:`free_of_skin`).

    ``roots``, ``normals`` and ``flow`` (the direction the hair is combed, along the skin) are per strand;
    ``lengths`` one number each.  ``probe`` (``distance(points, normals=True)``) is the body the hair lies
    on and never enters; without one the strands hang in free air.  ``ups`` is the direction a lock's crest
    faces at each point: the skin's normal where the lock lies on the skin.  ``phase`` (N numbers in 0..1)
    is where each strand's wave and curl begin, ``bias`` (N, 3) a constant pull added to gravity (wind, a
    parting, a ponytail's tie)."""
    P0 = np.asarray(roots, dtype=np.float64)
    N0 = _unit(np.asarray(normals, dtype=np.float64))
    T0 = _unit(np.asarray(flow, dtype=np.float64))
    L = np.broadcast_to(np.asarray(lengths, dtype=np.float64), (len(P0),)).copy()
    S = int(growth.segments)
    n = len(P0)
    pts = np.empty((n, S + 1, 3))
    ups = np.empty((n, S + 1, 3))
    height = np.zeros((n, S + 1))                 # each point's height over the skin (inf with no body to lie on)
    lift = math.radians(growth.lift)
    d = _unit(math.cos(lift) * T0 + math.sin(lift) * N0)
    pull = DOWN[None, :] if bias is None else DOWN[None, :] + np.asarray(bias, dtype=np.float64)
    p = P0.copy()
    up = N0.copy()
    pts[:, 0], ups[:, 0] = p, up
    ds = L / S
    for k in range(1, S + 1):
        # turn towards the pull: only the part of it across the hair turns the hair
        across = pull - (pull * d).sum(axis=1, keepdims=True) * d
        d = _unit(d + (growth.droop * ds)[:, None] * across)
        p = p + d * ds[:, None]
        if probe is not None:
            dist, nrm = probe.distance(p, normals=True)
            clear = growth.clearance + growth.rise * ds * k
            low = dist < clear
            if low.any():
                p[low] += ((clear - dist)[low])[:, None] * nrm[low]
                into = np.minimum((d * nrm).sum(axis=1), 0.0)
                d[low] = _unit(d[low] - into[low, None] * nrm[low])
            near = dist < clear + NEAR_SKIN
            up = np.where(near[:, None], nrm, up)
            height[:, k] = np.maximum(dist, clear)
        else:
            height[:, k] = np.inf
        up = _unit(up - (up * d).sum(axis=1, keepdims=True) * d)
        pts[:, k], ups[:, k] = p, up
    if growth.wave > 0.0 or growth.curl > 0.0:
        pts = _styled(pts, ups, L, growth, np.zeros(n) if phase is None else np.asarray(phase, dtype=np.float64))
        if probe is not None:
            flat = pts.reshape(-1, 3)
            dist, nrm = probe.distance(flat, normals=True)
            clear = np.repeat(growth.clearance + growth.rise * np.outer(ds, np.arange(S + 1)), 1).reshape(-1)
            low = dist < clear
            flat[low] += ((clear - dist)[low])[:, None] * nrm[low]
            pts = flat.reshape(n, S + 1, 3)
            pts[:, 0] = P0
            height = np.maximum(dist, clear).reshape(n, S + 1)
            height[:, 0] = 0.0
    return pts, ups, height


def free_of_skin(height, thickness, margin: float = 0.01, blend: float = 0.05) -> np.ndarray:
    """0..1 per point of a strand: 0 where a lock of ``thickness`` at that ``height`` has its underside in
    the skin (it prints as a ridge on the body), 1 where it is clear of it (it prints as a strand of its
    own, and has to be thick enough alone).  What :func:`lock_meshes` takes as ``free``."""
    under = UNDER * np.broadcast_to(np.asarray(thickness, dtype=np.float64), (len(height),))[:, None]
    return np.clip((np.asarray(height) - under + margin) / blend, 0.0, 1.0)


def _styled(pts, ups, lengths, growth: Growth, phase):
    """The paths with the wave and the curl laid over them (both fade in from the root)."""
    n, m, _ = pts.shape
    t = np.linspace(0.0, 1.0, m)[None, :]
    s = t * lengths[:, None]
    tan = _unit(np.gradient(pts, axis=1))
    side = _unit(np.cross(tan, ups))
    ramp = np.clip(t / max(growth.settle, 1e-6), 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    out = pts.copy()
    if growth.wave > 0.0:
        a = 2.0 * math.pi * (s / growth.wave_length + phase[:, None])
        out = out + (growth.wave * ramp * np.sin(a))[:, :, None] * side
    if growth.curl > 0.0:
        a = 2.0 * math.pi * (s / growth.curl_pitch + phase[:, None])
        out = out + (growth.curl * ramp)[:, :, None] * (np.cos(a)[:, :, None] * side + np.sin(a)[:, :, None] * ups)
    return out


# --------------------------------------------------------------------------
# locks


# a lock's section, across (0 the middle, 1 the edge) and up, in units of its thickness: the underside is
# flat at -UNDER (it lies in the skin), the crest at 1 - UNDER; a lower ridge either side of the crest with a
# groove between, so one lock reads as several strands lying together; the edge is a wall, not a knife
_TOP = ((1.0, 0.10), (0.84, 0.36), (0.62, 0.50), (0.40, 0.40), (0.0, 1.0 - UNDER))
# ... and where it hangs clear of the body: an oval through the same twelve points (its underside's corners
# drawn in), as thick as the lock from every side, because nothing backs it there
_PLAIN = ((1.0, 0.19), (0.84, 0.46), (0.62, 0.58), (0.40, 0.65), (0.0, 1.0 - UNDER))
_PLAIN_UNDER = 0.60            # the half-width of the oval's underside over the section's


def _section(shape: LockShape, plain: bool = False) -> np.ndarray:
    """The unit cross-section: ``(M, 2)`` points (across in -1..1, up in units of the thickness), running
    from the right edge over the top to the left edge and back underneath.  ``sides`` 12 (or more) is the
    grooved section above; fewer is an oval pinched to a crest (``sides`` vertices), for a draft.  ``plain``:
    the oval a lock is where it hangs free of the body (nothing behind it: it has to be thick enough to
    print from every side)."""
    m = int(shape.sides)
    if m >= 12:
        top = np.array(_PLAIN if plain else _TOP)
        right = top[:-1]                                   # edge ... inner ridge
        left = top[-2::-1] * np.array([-1.0, 1.0])         # mirrored, inner ridge ... edge
        k = _PLAIN_UNDER if plain else 1.0
        under = np.array([(-k, -UNDER), (0.0, -UNDER), (k, -UNDER)])
        return np.concatenate([right, top[-1:], left, under])
    a = 2.0 * math.pi * np.arange(m) / m
    x, y = np.cos(a), np.sin(a)
    y = np.where(y > 0, y * (1.0 - UNDER), y * UNDER)
    return np.stack([x, y], axis=1)


def _profiles(shape: LockShape, m: int, width, thickness):
    """Half-width and thickness at each of ``m`` stations, per lock: ``(N, m)`` each."""
    t = np.linspace(0.0, 1.0, m)
    grow_in = np.clip(t / 0.22, 0.0, 1.0)
    grow_in = grow_in * grow_in * (3.0 - 2.0 * grow_in)
    w = (shape.root + (1.0 - shape.root) * grow_in) * (1.0 - (1.0 - shape.tip) * t ** shape.taper)
    h = (0.6 + 0.4 * grow_in) * (1.0 - (1.0 - max(shape.tip, 0.6)) * t ** shape.taper)
    a = np.maximum(0.5 * np.asarray(width, dtype=np.float64)[:, None] * w[None, :], 0.5 * shape.tip_floor)
    b = np.maximum(np.asarray(thickness, dtype=np.float64)[:, None] * h[None, :], shape.tip_floor)
    return a, b


def lock_meshes(points, ups, shape: LockShape = LockShape(), width=None, thickness=None, root_normals=None,
                free=None, sink=None):
    """One closed shell per strand: a list of :class:`~minimaster.core.mesh.Mesh`.

    ``width`` and ``thickness`` (per strand, default the shape's own) let the locks of one head differ.  The
    first station is sunk ``sink`` (per strand; default ``shape.sink``) along ``root_normals`` (default the
    first ``ups``) into the body: that is the weld.  ``free`` (N, stations, 0..1) says where a lock hangs
    clear of the body: there its section is the plain one at full thickness (see :func:`_section`), the
    grooved one where it lies on the skin."""
    P = np.asarray(points, dtype=np.float64)
    U = np.asarray(ups, dtype=np.float64)
    n, m, _ = P.shape
    width = np.full(n, shape.width) if width is None else np.broadcast_to(np.asarray(width, float), (n,))
    thickness = (np.full(n, shape.thickness) if thickness is None
                 else np.broadcast_to(np.asarray(thickness, float), (n,)))
    P = P.copy()
    rn = _unit(U[:, 0] if root_normals is None else np.asarray(root_normals, dtype=np.float64))
    depth = np.full(n, shape.sink) if sink is None else np.broadcast_to(np.asarray(sink, float), (n,))
    P[:, 0] = P[:, 0] - depth[:, None] * rn
    tan = _unit(np.gradient(P, axis=1))
    up = _unit(U - (U * tan).sum(axis=2, keepdims=True) * tan)
    side = _unit(np.cross(tan, up))
    a, b = _profiles(shape, m, width, thickness)
    sec = _section(shape)                                             # (M, 2)
    M = len(sec)
    sx = np.broadcast_to(sec[None, None, :, 0], (n, m, M))
    sy = np.broadcast_to(sec[None, None, :, 1], (n, m, M))
    if free is not None and M >= 12:
        f = np.clip(np.asarray(free, dtype=np.float64), 0.0, 1.0)[:, :, None]
        plain = _section(shape, plain=True)
        sx = (1.0 - f) * sx + f * plain[None, None, :, 0]
        sy = (1.0 - f) * sy + f * plain[None, None, :, 1]
        # (and where it hangs free it does not thin to a film towards its tip: the lock's own thickness)
        b = np.maximum(b, f[:, :, 0] * thickness[:, None])
    rings = (P[:, :, None, :]
             + (a[:, :, None] * sx)[..., None] * side[:, :, None, :]
             + (b[:, :, None] * sy)[..., None] * up[:, :, None, :])
    root_c = P[:, 0]
    tip_c = P[:, -1] + tan[:, -1] * np.minimum(a[:, -1], 0.5 * b[:, -1])[:, None]
    verts = np.concatenate([rings.reshape(n, m * M, 3), root_c[:, None, :], tip_c[:, None, :]], axis=1)
    faces = _tube_faces(m, M)
    return [Mesh(verts[i], faces) for i in range(n)]


def _tube_faces(m: int, M: int) -> np.ndarray:
    """Triangles of a tube of ``m`` rings of ``M`` vertices with a fan over each end (the two extra
    vertices: ``m * M`` the root's centre, ``m * M + 1`` the tip's), facing outwards for the rings
    :func:`lock_meshes` builds."""
    j = np.arange(M)
    jn = (j + 1) % M
    out = []
    for k in range(m - 1):
        a, b = k * M, (k + 1) * M
        out.append(np.stack([a + j, a + jn, b + jn], axis=1))
        out.append(np.stack([a + j, b + jn, b + j], axis=1))
    root, tip = m * M, m * M + 1
    out.append(np.stack([np.full(M, root), jn, j], axis=1))
    last = (m - 1) * M
    out.append(np.stack([np.full(M, tip), last + j, last + jn], axis=1))
    # (side = tangent x up, so side x up = -tangent: the section runs clockwise seen from the tip, and the
    # triangles above are turned over)
    return np.concatenate(out)[:, ::-1].astype(np.int64)


# --------------------------------------------------------------------------
# the undercoat


def clean_patch(quads, mask) -> np.ndarray:
    """``mask`` (a boolean per quad) grown until the patch it marks has a simple edge: every vertex on its
    boundary has exactly two boundary edges (a patch pinched at a vertex cannot be thickened into a closed
    shell).  Quads round a pinch are added, never removed."""
    Q = np.asarray(quads)
    mask = np.asarray(mask, dtype=bool).copy()
    for _ in range(8):
        sel = Q[mask]
        a, b = sel.ravel(), np.roll(sel, -1, axis=1).ravel()
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        key = lo * (int(Q.max()) + 1) + hi
        uniq, count = np.unique(key, return_counts=True)
        edge = uniq[count == 1]
        ends = np.concatenate([edge // (int(Q.max()) + 1), edge % (int(Q.max()) + 1)])
        v, c = np.unique(ends, return_counts=True)
        pinch = v[c != 2]
        if not len(pinch):
            break
        mask |= np.isin(Q, pinch).any(axis=1)
    return mask


def cap_mesh(verts, quads, mask, thickness: float, sink: float = 0.05, edge: float = 0.2, rings: int = 3,
             levels: int = 1) -> Mesh | None:
    """The skin of the quads in ``mask`` thickened into a closed shell: ``thickness`` outwards in the middle,
    ``edge`` of that at the patch's own edge (over ``rings`` rows of quads), ``sink`` inwards everywhere.

    The patch is taken from the cage subdivided ``levels`` times (the skin that is drawn and printed, which
    lies a little inside the cage on a head), so the shell follows the skin and not the cage's facets."""
    from ..core.subdiv import subdivide_quads
    V = np.asarray(verts, dtype=np.float64)
    Q = np.asarray(quads)
    mask = clean_patch(Q, mask)
    if not mask.any():
        return None
    for _ in range(int(levels)):
        F = len(Q)
        V, Q = subdivide_quads(V, Q, 1)
        mask = np.tile(mask, 4)                 # (the four children of quad f are k * F + f)
        assert len(Q) == 4 * F
    from .scatter import vertex_normals
    nrm = vertex_normals(V, Q)
    sel = Q[mask]
    ids, local = np.unique(sel, return_inverse=True)
    local = local.reshape(sel.shape)
    k = len(ids)
    # how many rows in from the edge each vertex is
    a, b = local.ravel(), np.roll(local, -1, axis=1).ravel()
    key = np.minimum(a, b) * k + np.maximum(a, b)
    uniq, first, count = np.unique(key, return_index=True, return_counts=True)
    on_edge = np.zeros(k, dtype=bool)
    be = first[count == 1]
    on_edge[a[be]] = True
    on_edge[b[be]] = True
    depth = np.where(on_edge, 0.0, np.inf)
    span = int(rings) * (2 ** int(levels))
    for step in range(1, span + 1):
        d = depth[local]                                    # (F, 4)
        reach = np.minimum(d.min(axis=1, keepdims=True) + 1.0, d)
        np.minimum.at(depth, local.ravel(), reach.ravel())
    w = np.clip(depth / span, 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)
    out_off = thickness * (edge + (1.0 - edge) * w)
    P, N = V[ids], nrm[ids]
    outer = P + out_off[:, None] * N
    inner = P - sink * N
    tri = [local[:, [0, 1, 2]], local[:, [0, 2, 3]],
           k + local[:, [2, 1, 0]], k + local[:, [3, 2, 0]]]
    ea, eb = a[be], b[be]                                   # boundary edges, in their quad's own direction
    tri += [np.stack([eb, ea, k + ea], axis=1), np.stack([eb, k + ea, k + eb], axis=1)]
    return Mesh(np.concatenate([outer, inner]), np.concatenate(tri).astype(np.int64))
