"""Extra arms, wings, tails and horns for the realistic character.

THE CONSTRAINT. The 1,280 morph targets address the base mesh by *vertex
index*, so the body's 19,158 vertices cannot be renumbered: inserting or
deleting a vertex would silently corrupt every morph. That rules out cutting a
hole in the torso and stitching a wing into it.

THE APPROACH. An appendage is its own **closed watertight shell** that overlaps
the body — exactly the contract the STL export already relies on, where the
slicer unions overlapping solids at slice time. Nothing about the base mesh
changes, every morph keeps working, and the per-shell gate still applies.

The part that makes it feel attached rather than glued on is the
:class:`SurfaceAnchor`: instead of a fixed position, an appendage is bound to a
*triangle of the body* by barycentric coordinates plus a local frame. Resolving
that anchor against the current morphed vertices gives a position and
orientation that **follow the body** — make the character broader, taller or
fatter and the wing roots travel with the shoulder blades, because they are
defined relative to the surface rather than to the world.

Two ways to get the appendage geometry:

* **Harvested** — copy a region of the body itself, selected by proximity to a
  bone chain read from the joint markers. An extra arm harvested this way is
  the character's *own* arm, so it already matches their proportions, build and
  musculature, and it re-harvests correctly after any slider change.
* **Procedural** — tails, wings and horns swept from rings, since the body has
  no such parts to copy.
"""

from __future__ import annotations

import dataclasses
import weakref
from dataclasses import dataclass, field

import numpy as np

from ..core.mesh import Mesh
from .wingrig import PoseControls


# --------------------------------------------------------------------------
# closing an open region


def boundary_loops(faces) -> list[list[int]]:
    """Ordered vertex loops around the open boundary of a quad/tri mesh.

    Loops are traced by consuming DIRECTED boundary edges, not by a
    vertex-to-successor map. A cut through a real body can produce a *pinch
    vertex* that belongs to two loops at once (boundary degree 4); a
    vertex-keyed walk silently drops one of them and leaves the mesh open,
    whereas consuming each directed edge exactly once splits the figure-eight
    into the two loops it actually is.
    """
    faces = np.asarray(faces)
    n = faces.shape[1]
    count: dict[tuple[int, int], int] = {}
    for f in faces:
        for i in range(n):
            a, b = int(f[i]), int(f[(i + 1) % n])
            key = (a, b) if a < b else (b, a)
            count[key] = count.get(key, 0) + 1

    out_edges: dict[int, list[int]] = {}
    for f in faces:
        for i in range(n):
            a, b = int(f[i]), int(f[(i + 1) % n])
            key = (a, b) if a < b else (b, a)
            if count[key] == 1:
                out_edges.setdefault(a, []).append(b)

    loops: list[list[int]] = []
    for start in sorted(out_edges):
        while out_edges.get(start):
            loop = [start]
            cur = out_edges[start].pop()
            while cur != start:
                loop.append(cur)
                nxt = out_edges.get(cur)
                if not nxt:          # open chain (should not happen on a
                    break            # manifold cut) - abandon it
                cur = nxt.pop()
            if len(loop) >= 3 and cur == start:
                loops.append(loop)
    return loops


def cap_boundaries(verts, faces):
    """Fan-fill every boundary loop so the region becomes a closed solid.

    Returns (verts, tri_faces). The cap winding is taken from the directed
    boundary edges, so it matches the surrounding surface.
    """
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    tris = (np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]])
            if faces.shape[1] == 4 else faces.copy())
    loops = boundary_loops(faces)
    if not loops:
        return verts, tris
    extra_v = []
    extra_f = []
    for loop in loops:
        pole = len(verts) + len(extra_v)
        extra_v.append(verts[loop].mean(axis=0))
        m = len(loop)
        for i in range(m):
            a, b = loop[i], loop[(i + 1) % m]
            # boundary runs a->b on the open side, so the cap closes b->a
            extra_f.append([pole, b, a])
    verts = np.vstack([verts, np.asarray(extra_v)])
    tris = np.vstack([tris, np.asarray(extra_f, dtype=np.int64)])
    return verts, tris


# --------------------------------------------------------------------------
# left and right


def mirror_joint_name(name: str) -> str:
    """The joint marker on the other side of the body: ``l-`` <-> ``r-``.

    Only the LEADING side prefix (after an optional ``joint-``) is swapped. A chain of
    replace() calls cannot do this: ``"l-finger-2-1"`` contains ``"r-"`` inside
    ``finger-`` and came out as ``"r-fingel-2-1"`` (21 of the 55 left-hand markers were
    mangled, so a mirrored leg or finger harvest raised KeyError). Names on the midline
    (``pelvis``, ``spine-1``, ``head``) are returned unchanged.
    """
    head, rest = ("joint-", name[6:]) if name.startswith("joint-") else ("", name)
    if rest.startswith("l-"):
        return head + "r-" + rest[2:]
    if rest.startswith("r-"):
        return head + "l-" + rest[2:]
    return name


_MIRROR_TOL = 1e-9
_MIRROR_MAPS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _mirror_index_map(points: np.ndarray) -> np.ndarray:
    """For each point, the index of the point at its x-mirror position.

    Exact-match lookup on a 1e-6 grid first (the mesh is symmetric to ~1e-15, so nearly
    every point matches by hash), then brute force only for the few that straddle a grid
    boundary. Raises if some point has no mirror partner within ``_MIRROR_TOL``.
    """
    pts = np.asarray(points, dtype=np.float64)
    mir = pts * np.array([-1.0, 1.0, 1.0])
    quant = 1e-6
    keys = lambda a: [tuple(k) for k in np.round(a / quant).astype(np.int64)]  # noqa: E731
    table = {k: i for i, k in enumerate(keys(pts))}
    out = np.array([table.get(k, -1) for k in keys(mir)], dtype=np.int64)
    todo = np.nonzero(out < 0)[0]
    for i in todo:
        out[i] = int(np.argmin(np.linalg.norm(pts - mir[i], axis=1)))
    err = np.abs(pts[out] - mir).max(axis=1) if len(pts) else np.zeros(0)
    if len(err) and (err.max() > _MIRROR_TOL or not np.array_equal(out[out], np.arange(len(out)))):
        raise ValueError(
            f"the body mesh is not mirror-symmetric (worst partner is {err.max():.2e} away); "
            "a mirrored appendage needs a symmetric base")
    return out


def mirror_face_map(base) -> np.ndarray:
    """``map[f]`` is the cage quad that is the x-mirror image of quad ``f`` (an
    involution; a quad on the midline may be its own mirror).

    The cage topology never changes with the morphs and the unmorphed base is symmetric
    (max centroid error ~1e-15), so this is computed ONCE per base mesh and cached. It
    is what lets a side be stored once: the left-hand face is persisted and the right
    wing is placed on ``map[face]``, an exact mirror of the left, whatever the body is.
    """
    cached = _MIRROR_MAPS.get(base)
    if cached is None:
        bv, bq = base.body_cage(None)
        cached = _mirror_index_map(bv[bq].mean(axis=1))
        cached.setflags(write=False)
        _MIRROR_MAPS[base] = cached
    return cached


_MIRROR_VERTS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def mirror_vertex_map(base) -> np.ndarray:
    """``map[i]`` is the cage vertex at the x-mirror position of cage vertex ``i``."""
    cached = _MIRROR_VERTS.get(base)
    if cached is None:
        bv, _ = base.body_cage(None)
        cached = _mirror_index_map(bv)
        cached.setflags(write=False)
        _MIRROR_VERTS[base] = cached
    return cached


def mirror_anchor_weights(base, body_q, face: int, weights) -> tuple:
    """The weights of an anchor on ``face`` expressed on its mirror face.

    A quad and its mirror image are stored with their corners in a different order, so
    non-uniform weights must follow the corner VERTICES, not the corner positions.
    """
    mf = int(mirror_face_map(base)[face])
    vmap = mirror_vertex_map(base)
    corners = [int(c) for c in body_q[mf]]
    out = [0.0] * len(corners)
    for c, w in zip(body_q[face], weights):
        out[corners.index(int(vmap[int(c)]))] = float(w)
    return tuple(out)


# --------------------------------------------------------------------------
# anchoring to the body surface


@dataclass
class SurfaceAnchor:
    """A frame bound to a body triangle, so it follows every morph."""

    corners: tuple            # every corner of the bound face, not just three
    weights: tuple            # barycentric weights over those corners
    up_hint: tuple[float, float, float] = (0.0, 0.0, 1.0)
    offset: float = 0.0          # push out along the surface normal

    @classmethod
    def from_point(cls, verts, faces, point, up_hint=(0.0, 0.0, 1.0)):
        """Bind to the face whose centroid is nearest ``point``.

        The WHOLE face is kept. Binding to its first three corners instead
        makes a mirrored pair asymmetric: the body cage is quads, and a quad
        and its mirror image are stored with different corner orders, so
        ``face[:3]`` picks a different sub-triangle on each side and the two
        frames end up a degree or two apart -- which a long wing turns into a
        visible tilt.
        """
        f = np.asarray(faces)
        cent = np.asarray(verts)[f].mean(axis=1)
        i = int(np.argmin(np.linalg.norm(cent - np.asarray(point), axis=1)))
        k = f.shape[1]
        return cls(corners=tuple(int(x) for x in f[i]),
                   weights=(1.0 / k,) * k, up_hint=up_hint)

    def resolve(self, verts):
        """(origin, basis) for the current mesh. Basis columns are
        (tangent, bitangent, normal); the normal points out of the body."""
        v = np.asarray(verts, dtype=np.float64)
        p = v[list(self.corners)]
        w = np.asarray(self.weights, dtype=np.float64)
        origin = (w[:, None] * p).sum(axis=0)
        # Newell's normal: uses every corner and depends only on the cyclic
        # order, so mirror-image faces give exactly mirror-image normals.
        nxt = np.roll(p, -1, axis=0)
        n = np.cross(p, nxt).sum(axis=0)
        ln = np.linalg.norm(n)
        n = n / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0])
        up = np.asarray(self.up_hint, dtype=np.float64)
        t = up - np.dot(up, n) * n
        lt = np.linalg.norm(t)
        if lt < 1e-9:                       # up parallel to the normal
            t = np.cross(n, [1.0, 0.0, 0.0])
            lt = np.linalg.norm(t)
            if lt < 1e-9:
                t = np.cross(n, [0.0, 1.0, 0.0])
                lt = np.linalg.norm(t)
        t = t / lt
        bt = np.cross(n, t)
        return origin + n * self.offset, np.stack([t, bt, n], axis=1)


def anchor_matrix(anchor: SurfaceAnchor, verts, scale=1.0, rotation=None) -> np.ndarray:
    """The 4x4 that puts an appendage (authored +Z = outward, origin at its
    base) onto the body at ``anchor``, resolved against the morphed ``verts``."""
    origin, basis = anchor.resolve(verts)
    m = np.eye(4)
    rot = basis if rotation is None else basis @ rotation
    m[:3, :3] = rot * float(scale)
    m[:3, 3] = origin
    return m


def place(mesh: Mesh, anchor: SurfaceAnchor, verts, scale=1.0,
          rotation=None) -> Mesh:
    """Put an appendage (authored +Z = outward, origin at its base) onto the
    body at ``anchor``, resolved against the current morphed ``verts``."""
    return mesh.transform(anchor_matrix(anchor, verts, scale, rotation))


# --------------------------------------------------------------------------
# harvesting a limb from the body itself


def joint_position(base, name, verts=None) -> np.ndarray:
    """Centroid of a ``joint-*`` marker cube — the joint's live position."""
    key = name if name.startswith("joint-") else f"joint-{name}"
    v, _ = base.helper_group(key, verts)
    return v.mean(axis=0)


def largest_component(faces):
    """Keep only the biggest edge-connected island of faces.

    A proximity harvest can also catch a stray patch elsewhere on the body
    (the radius clips the armpit or the ribs), and two disjoint pieces sharing
    a pinch vertex cap into a non-manifold edge. A limb is one piece, so the
    stray islands are dropped.
    """
    faces = np.asarray(faces)
    n = faces.shape[1]
    vert_faces: dict[int, list[int]] = {}
    for fi, f in enumerate(faces):
        for v in f:
            vert_faces.setdefault(int(v), []).append(fi)
    seen = np.zeros(len(faces), dtype=bool)
    best: list[int] = []
    for start in range(len(faces)):
        if seen[start]:
            continue
        stack, comp = [start], []
        seen[start] = True
        while stack:
            fi = stack.pop()
            comp.append(fi)
            for v in faces[fi]:
                for fj in vert_faces[int(v)]:
                    if not seen[fj]:
                        seen[fj] = True
                        stack.append(fj)
        if len(comp) > len(best):
            best = comp
    return faces[np.sort(np.asarray(best, dtype=np.int64))]


def _split_pinch_vertices(faces):
    """Duplicate any vertex that appears in two separate boundary loops.

    Such a vertex belongs to two rims at once; capping both would make the
    edges around it non-manifold. Giving each loop its own copy separates them.
    Returns (faces, extra_sources) where extra_sources[i] is the original index
    a newly appended vertex was copied from.
    """
    faces = np.asarray(faces).copy()
    loops = boundary_loops(faces)
    counts: dict[int, int] = {}
    for loop in loops:
        for v in set(loop):
            counts[v] = counts.get(v, 0) + 1
    pinched = {v for v, c in counts.items() if c > 1}
    if not pinched:
        return faces, []
    extra: list[int] = []
    next_id = int(faces.max()) + 1
    for loop in loops[1:]:                     # first loop keeps the original
        for v in set(loop) & pinched:
            dup = next_id + len(extra)
            extra.append(v)
            # rewrite only the faces that touch BOTH v and this loop
            in_loop = set(loop)
            for fi, f in enumerate(faces):
                if v in f and len(in_loop & set(int(x) for x in f)) >= 2:
                    faces[fi] = np.where(f == v, dup, f)
    return faces, extra


def _segment_distance(pts, a, b):
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return np.linalg.norm(pts - a, axis=1)
    t = np.clip((pts - a) @ ab / denom, 0.0, 1.0)
    return np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)


# Upper-arm and thigh length (shoulder -> elbow, hip -> knee) of the NEUTRAL body, in scene
# units: what the lengths of a harvest (radius, trim, sink) are given in. A test recomputes
# them from the neutral body, so a changed base mesh cannot drift unnoticed.
LIMB_REFERENCE_LENGTH = {"arm": 2.4423469356466336, "leg": 4.0260352512777695}


# how much of the upper bone the narrowed root of a harvested limb takes (see limbrig.taper_root)
NECK_LENGTH = 0.45


def _is_mirror_image(a, b, tol: float = 1e-9) -> bool:
    """Is ``b`` the x-mirror image of ``a`` as a set of points (in any vertex order)?"""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    m = a * np.array([-1.0, 1.0, 1.0])
    # order both by their coordinates rounded to 1e-6 (a mirrored point rounds the same, so
    # the two sorted lists pair off point for point) and compare; the few points that
    # straddle a rounding edge are looked up by distance instead
    ia, ib = np.lexsort(np.round(m.T[::-1] / 1e-6)), np.lexsort(np.round(b.T[::-1] / 1e-6))
    off = np.abs(m[ia] - b[ib]).max(axis=1)
    return all(np.linalg.norm(b - m[ia[k]], axis=1).min() <= tol for k in np.nonzero(off > tol)[0])


@dataclass
class HarvestInfo:
    """What :func:`harvest_limb` knows about the cut it made.

    ``root_loop`` are the vertex indices (into the returned mesh) of the rim of the cut at
    the chain's first joint, ``root_pole`` the vertex that closes it, ``root_center`` their
    mean. A rig pins exactly these so the end that is welded into the torso never moves."""

    root_loop: np.ndarray
    root_pole: int
    root_center: np.ndarray


def harvest_region(base, chain, verts=None, radius=1.15, trim=0.0):
    """Cut a closed shell out of the body around a bone chain.

    ``chain`` is a list of joint-marker names, e.g.
    ``["l-shoulder", "l-elbow", "l-hand"]``. A body quad joins the harvest when
    all four of its corners lie within ``radius`` of the chain and beyond
    ``trim`` along it (``trim`` shaves the root so the copy starts below the
    shoulder). The cut is fan-capped, so the result is a printable solid.
    """
    return harvest_limb(base, chain, verts, radius=radius, trim=trim)[0]


def _harvest_cut(base, chain, v, radius, trim):
    body_v, body_q = base.body_cage(v)
    pts = [joint_position(base, n, v) for n in chain]

    dist = np.full(len(body_v), np.inf)
    for a, b in zip(pts[:-1], pts[1:]):
        dist = np.minimum(dist, _segment_distance(body_v, a, b))
    keep_v = dist <= radius
    if trim > 0.0:
        axis = pts[-1] - pts[0]
        n = np.linalg.norm(axis)
        if n > 1e-9:
            axis = axis / n
            along = (body_v - pts[0]) @ axis
            keep_v &= along >= trim
    sel = keep_v[body_q].all(axis=1)
    if not sel.any():
        raise ValueError(f"harvest of {chain} selected no faces "
                         f"(radius {radius} too small?)")
    quads = largest_component(body_q[sel])
    used = np.unique(quads)
    remap = -np.ones(len(body_v), dtype=np.int64)
    remap[used] = np.arange(len(used))
    local = remap[quads]
    verts = body_v[used]
    local, extra = _split_pinch_vertices(local)
    if extra:
        verts = np.vstack([verts, verts[np.asarray(extra, dtype=np.int64)]])
    hv, hf = cap_boundaries(verts, local)
    # the root cut is the boundary loop nearest the chain's first joint; cap_boundaries
    # appended one pole per loop, in the order boundary_loops lists them
    loops = boundary_loops(local)
    root = int(np.argmin([np.linalg.norm(verts[lp].mean(axis=0) - pts[0]) for lp in loops]))
    loop = np.asarray(loops[root], dtype=np.int64)
    info = HarvestInfo(root_loop=loop, root_pole=len(verts) + root,
                       root_center=verts[loop].mean(axis=0))
    return Mesh(hv, hf), info


# A cut whose boundary loop revisits a vertex caps into a non-manifold edge (the fan pole joins
# that vertex twice). It is a property of WHERE the cylinder meets the body's quads, so a body
# that fails at one radius closes at a slightly different one: 6 of 500 random bodies failed
# the default extra-leg cut, 0.2% of them the first version's.
_CUT_RETRY = (1.0, 0.97, 1.03, 0.94, 1.06, 0.91, 1.09)


def harvest_limb(base, chain, verts=None, radius=1.15, trim=0.0):
    """:func:`harvest_region` plus a :class:`HarvestInfo` about its root cut:
    ``(Mesh, HarvestInfo)``. The mesh is exactly what ``harvest_region`` returns -- unless that
    cut does not close into a manifold solid on this body, in which case the radius (and the
    trim with it) is nudged by a few percent until it does."""
    v = base.verts if verts is None else verts
    first = None
    for k in _CUT_RETRY:
        try:
            mesh, info = _harvest_cut(base, chain, v, radius * k, trim * k)
        except ValueError:
            if first is None:
                raise
            continue
        if first is None:
            first = (mesh, info)
        if mesh.integrity_report()["watertight"]:
            return mesh, info
    return first


# --------------------------------------------------------------------------
# procedural appendages (authored with the base at the origin, +Z outward)


def _ring(center, r_t, r_b, basis, n=10):
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return (center[None, :] + r_t * np.cos(th)[:, None] * basis[:, 0][None, :]
            + r_b * np.sin(th)[:, None] * basis[:, 1][None, :])


def _loft(rings, close=True) -> Mesh:
    n = len(rings[0])
    verts = np.vstack(rings)
    faces = []
    for i in range(len(rings) - 1):
        b0, b1 = i * n, (i + 1) * n
        for k in range(n):
            kn = (k + 1) % n
            faces.append([b0 + k, b0 + kn, b1 + kn, b1 + k])
    quads = np.asarray(faces, dtype=np.int64)
    v, tris = cap_boundaries(verts, quads) if close else (verts, quads)
    mesh = Mesh(v, tris)
    if mesh.volume() < 0:
        mesh = Mesh(mesh.vertices, mesh.faces[:, [0, 2, 1]])
    return mesh


def _transport(basis, old_dir, new_dir):
    """``basis`` carried along from direction ``old_dir`` to ``new_dir`` by the smallest
    rotation between them (a parallel-transport frame: the ring does not spin about the
    path). ``_frame`` picks its reference axis from the direction and switches it at a
    threshold, which twists the rings by 90-160 degrees in one segment of a bent tube."""
    axis = np.cross(old_dir, new_dir)
    s = float(np.linalg.norm(axis))
    c = float(old_dir @ new_dir)
    if s < 1e-12:
        return basis
    k = axis / s
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return (np.eye(3) + s * kx + (1.0 - c) * (kx @ kx)) @ basis


# the rounded end of a blunt tube: rings on a quarter sphere of the tip radius (degrees from
# the last ring's plane), then the loft's fan cap closes the last, small one
_TIP_CAP = (32.0, 62.0, 82.0)


def _round_tip(rings, pos, direction, basis, radius, minor, ring_verts) -> None:
    """Append the rings of a rounded end to ``rings``: ``pos`` is the centre of the last
    ring, ``radius`` its (major) radius, ``minor`` the ratio of the minor axis."""
    for deg in _TIP_CAP:
        phi = np.radians(deg)
        rings.append(_ring(pos + direction * radius * np.sin(phi),
                           radius * np.cos(phi), radius * minor * np.cos(phi), basis, ring_verts))


def root_flare(t: float, flare: float, flare_length: float) -> float:
    """The radius multiplier of a flared tube root at the fraction ``t`` of its length:
    ``1 + flare`` at the base, falling to 1 at ``flare_length`` (a smooth ``(1 - t/L)^2``,
    C1 where it ends), so the base is broad where it enters the skin."""
    if not flare or t >= flare_length:
        return 1.0
    return 1.0 + float(flare) * (1.0 - t / flare_length) ** 2


def make_tail(length=6.0, base_radius=0.45, segments=12, ring_verts=10,
              curl=0.55, droop=0.9, *, tip_radius=None, lift=0.0, sway=0.0,
              flare=0.0, flare_length=0.25) -> Mesh:
    """A tapered, curling tail. +Z is outward from the attachment point.

    ``curl`` bends the path about the local Y axis and ``droop`` about the local X axis,
    both growing linearly toward the tip (the first version's names; on the default tail
    a positive ``curl`` raises the end and a positive ``droop`` lowers it).

    ``tip_radius`` is the radius at the end of the taper. ``None`` is the first version's law
    (``base_radius * (1 - t) ** 0.85 + 0.02``: a point, 0.07 mm across at 32 mm); a number
    gives ``tip_radius + (base_radius - tip_radius) * (1 - t) ** 0.85`` closed by a rounded
    cap, which is a tip a printer can make. (A number also carries the rings along the path
    by parallel transport; ``None`` keeps the first version's ring frames, which twist by
    up to 90-160 degrees in one segment of a bent tail, bit for bit.)

    ``lift`` (up) and ``sway`` (toward +X) are degrees of total bend, spread evenly along
    the length, in the vertical and the lateral plane: raise or swing the whole tail from
    its base. Zero (the default) is bit-for-bit the first version.

    ``flare`` broadens the root: the radius is multiplied by ``1 + flare`` at the base,
    easing back to 1 over the first ``flare_length`` of the length (see :func:`root_flare`);
    0 (the default) leaves every vertex exactly where it was.
    """
    basis = np.eye(3)
    rings = []
    pos = np.zeros(3)
    direction = np.array([0.0, 0.0, 1.0])
    step = length / segments
    n = segments + 1                            # rings, each one a rotation
    last_pos = last_dir = last_basis = prev_dir = None
    for i in range(segments + 1):
        t = i / segments
        if tip_radius is None:
            r = base_radius * (1.0 - t) ** 0.85 + 0.02
        else:
            r = tip_radius + (base_radius - tip_radius) * (1.0 - t) ** 0.85
        if flare:
            r *= root_flare(t, flare, flare_length)
        # bend progressively: droop away from the body and curl sideways
        ang = np.array([droop * t * 0.9, curl * t * 0.7, 0.0]) * step
        if lift or sway:
            ang = ang + np.array([-np.radians(sway) / n, np.radians(lift) / n, 0.0])
        rot = _euler(ang)
        direction = rot @ direction
        if tip_radius is None or last_basis is None:
            b = _frame(direction)
        else:               # the rounded tail carries its rings along (no twist band)
            b = _transport(last_basis, last_dir, direction)
        rings.append(_ring(pos, r, r * 0.92, b, ring_verts))
        last_pos, last_dir, last_basis, last_r = pos, direction, b, r
        pos = pos + direction * step
    if tip_radius is not None:
        _round_tip(rings, last_pos, last_dir, last_basis, last_r, 0.92, ring_verts)
    return _loft(rings)


def make_horn(length=2.2, base_radius=0.42, segments=8, ring_verts=8,
              curve=1.1, twist=0.0, *, tip_radius=None, turns=0.0,
              flare=0.0, flare_length=0.25) -> Mesh:
    """A tapered horn. ``curve`` bends it progressively (a spiral when large) and ``twist``
    bends it the other way. ``turns`` (default 0: the bend grows toward the tip) makes it a
    helix instead: the bend is constant along the horn and its plane turns about the horn's
    own axis that many times from base to tip (a corkscrew, a lyre at less than one turn).
    ``tip_radius`` is the radius at the end (``None``: the first version's point,
    ``base_radius * (1 - t) ** 1.25 + 0.012``; a number: ``tip_radius + (base_radius -
    tip_radius) * (1 - t) ** 1.25`` closed by a rounded cap, with the rings carried along the
    path by parallel transport; ``None`` keeps the first version's ring frames).

    ``flare`` broadens the root like a tail's (:func:`root_flare`); 0 changes nothing."""
    basis = np.eye(3)
    rings = []
    pos = np.zeros(3)
    direction = np.array([0.0, 0.0, 1.0])
    step = length / segments
    last_pos = last_dir = last_basis = None
    for i in range(segments + 1):
        t = i / segments
        if tip_radius is None:
            r = base_radius * (1.0 - t) ** 1.25 + 0.012
        else:
            r = tip_radius + (base_radius - tip_radius) * (1.0 - t) ** 1.25
        if flare:
            r *= root_flare(t, flare, flare_length)
        if turns:
            # a helix: the bend is the same all along, and its plane turns with the length
            phi = 2.0 * np.pi * turns * t
            a0, a1 = curve * step * np.cos(phi) - twist * step * np.sin(phi), \
                curve * step * np.sin(phi) + twist * step * np.cos(phi)
        else:
            a0, a1 = curve * t * step, twist * t * step
        rot = _euler(np.array([a0, a1, 0.0]))
        direction = rot @ direction
        if tip_radius is None or last_basis is None:
            b = _frame(direction)
        else:
            b = _transport(last_basis, last_dir, direction)
        rings.append(_ring(pos, r, r, b, ring_verts))
        last_pos, last_dir, last_basis, last_r = pos, direction, b, r
        pos = pos + direction * step
    if tip_radius is not None:
        _round_tip(rings, last_pos, last_dir, last_basis, last_r, 1.0, ring_verts)
    return _loft(rings)


# Wings are their own module: a wing is a modified forelimb, not a shape, and
# building one properly (skeleton, patagium or feathers) is a job in itself.
from .wings import make_wing  # noqa: E402,F401  (re-exported for BUILDERS)


def _euler(a):
    cx, sx = np.cos(a[0]), np.sin(a[0])
    cy, sy = np.cos(a[1]), np.sin(a[1])
    cz, sz = np.cos(a[2]), np.sin(a[2])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _frame(direction):
    d = np.asarray(direction, dtype=np.float64)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(d @ up)) > 0.98:
        up = np.array([1.0, 0.0, 0.0])
    t = np.cross(up, d)
    t /= max(float(np.linalg.norm(t)), 1e-12)
    b = np.cross(d, t)
    return np.stack([t, b, d], axis=1)


# --------------------------------------------------------------------------
# the appendage spec that a character document can carry


BUILDERS = {"tail": make_tail, "horn": make_horn, "wing": make_wing}

# Placement of a wing on the shoulder blades of the MakeHuman base. These are
# not arbitrary taste: the anchor is the rearmost point of the scapula, the
# hint aims the span up and out, and the roll turns the wing's chord from
# "straight out of the back" (which leaves the root chord hanging several
# units behind the figure, touching only at a point) to "down the flank",
# which is where a bat's plagiopatagium actually runs. See back_wings().
WING_ANCHOR = (0.84, 4.39, -0.73)
WING_UP_HINT = (1.0, 1.4, -0.45)
WING_ROLL = -95.0
# and sunk into the back, because a wing that only grazes the skin is joined
# along a hairline: at offset 0 just 2% of the root lies inside the torso,
# against ~40% here, which is a weld a 32 mm print can actually survive.
WING_OFFSET = -0.9
# Where a wing's attachment line ends on the body: the lower back at the waist,
# well off the midline (so a stowed pair lies side by side on the back instead of
# piling onto the spine; z stays inside the back surface, -0.80 at that x). A
# bat's plagiopatagium runs from the shoulder down the flank
# to the hip, so the root edge is aimed at this body point (converted to the
# wing's own frame) instead of at a fixed local direction. The wing's local +Z
# points down-and-OUTWARD on the back, so without this the root hangs in the air
# beside the figure.
WING_ROOT = (1.20, 0.40, -0.60)


# Per-style placement overrides. A feathered wing has no membrane to glue to the waist, so
# its root only aims the stowed wing, and it is the pivot the wing stows about. A folded wing
# is a plank that has to clear the shoulder blades and the buttocks, so the pivot sits
# OUTSIDE the back surface; and it is given RELATIVE to the anchor (down the back from the
# scapula, 0.27 behind its surface), so the wing stows the same distance off a slim back and
# a heavy one and its orientation stays independent of the body's sliders.
WING_PLACEMENT = {"feathered": {"root_point": (0.60, -3.50, -0.27), "root_relative": True}}


def back_wings(style: str = "membrane", scale: float = 1.0,
               mirror: bool = True, **params) -> "Appendage":
    """A mirrored pair of wings sitting on the shoulder blades.

    Wraps the placement constants so callers do not have to rediscover the
    orientation, which is the difference between wings that are attached and
    wings that float behind the back.
    """
    place = dict(root_point=WING_ROOT)
    place.update(WING_PLACEMENT.get(style, {}))
    return Appendage("wing", anchor_point=WING_ANCHOR, mirror=mirror,
                     up_hint=WING_UP_HINT, roll=WING_ROLL, offset=WING_OFFSET,
                     scale=scale, params=dict(style=style, **params), **place)


@dataclass
class Appendage:
    """A declarative appendage on a character."""

    kind: str                                   # tail | horn | wing | harvest
    anchor_point: tuple = (0.0, 0.0, 0.0)       # where on the body (scene units)
    scale: float = 1.0
    params: dict = field(default_factory=dict)
    chain: tuple = ()                           # for kind="harvest"
    mirror: bool = False                        # also place a mirrored copy
    # Which way the appendage's local +X aims once it is on the body. The
    # anchor's tangent is this hint projected into the surface, so it decides
    # the roll: a wing spans along its local X, and without a lateral hint a
    # back-mounted wing sweeps off in whatever direction the triangle happens
    # to face. Default is world up, which suits horns and tails.
    up_hint: tuple = (0.0, 1.0, 0.0)
    # Degrees of roll about the appendage's own +X axis, applied before
    # placement. ``up_hint`` aims +X but leaves the appendage free to spin
    # around it, and for anything wider than a spike that spin is the whole
    # difference between attached and floating: a wing's local +Z would
    # otherwise follow the surface normal, so its chord would stick straight
    # out from the back instead of running down the flank the way a real
    # plagiopatagium does.
    roll: float = 0.0
    # How far to sink the appendage along the surface normal. Negative buries
    # the root INSIDE the body. Overlapping shells are what the slicer unions,
    # so an appendage that merely touches the skin is welded along a hairline
    # and can print detached; a broad root needs real interpenetration.
    offset: float = 0.0
    # Body-space point where the appendage's attachment line ends (wings only).
    # Mirrored with the anchor; converted into the wing's own frame at build.
    root_point: tuple | None = None
    # ``root_point`` as an offset from the anchor's surface point instead of a point in body
    # space (so it follows the body's morphs). Mirrored like the point.
    root_relative: bool = False
    # ---- topological placement (additive; ``None`` keeps the legacy behaviour exactly) ----
    # ``anchor_point`` is a point in WORLD space that is re-snapped to whichever body face is
    # nearest on every build, so on a taller or smaller body it lands on a different part of
    # the body (a wing shoulder 7.7 units below the shoulder blade). ``anchor_face`` is the
    # cage quad itself: the binding is topological, the quad exists on every morph of the
    # body and carries the appendage with it. It is the LEFT-hand quad; the right side uses
    # its exact mirror image (``mirror_face_map``). ``anchor_weights`` are barycentric
    # weights over the quad's corners (default: its centre).
    anchor_face: int | None = None
    anchor_weights: tuple | None = None
    # ``root_point`` bound to the body the same way: the end of the wing's attachment line is
    # the centroid of cage quad ``root_face`` (mirrored for the right side) plus
    # ``root_offset`` (mirrored in x), so the waist target moves with the waist.
    root_face: int | None = None
    root_offset: tuple = (0.0, 0.0, 0.0)
    # which sides to build: ``None`` = legacy (``mirror`` decides: one or a pair), or one of
    # ``center`` / ``left`` / ``right`` / ``both``. ``right`` is the exact mirror of the
    # left-hand definition, so a lone right wing equals the right wing of a pair.
    side: str | None = None

    # pose controls that may ride along in ``params`` for a wing; everything
    # else in ``params`` shapes the wing itself
    # derived from PoseControls so a new control can never be left out (the list
    # was hand-written, and wrist_flex / hand_twist crashed the builder)
    POSE_KEYS = tuple(f.name for f in dataclasses.fields(PoseControls))
    # a named pose (wingrig.PRESETS) may ride along too; explicit controls
    # override what the preset sets
    PRESET_KEY = "preset"

    def _has_root(self) -> bool:
        return self.root_point is not None or self.root_face is not None

    def _face_for(self, face, idx, base) -> int:
        """The cage quad for side ``idx``: the stored (left-hand) quad, or its mirror."""
        if idx == 0:
            return int(face)
        if base is None:
            raise ValueError("placing the right-hand side of a face-anchored appendage "
                             "needs the base mesh (for its mirror-face map)")
        return int(mirror_face_map(base)[int(face)])

    def _face_anchor(self, body_v, body_q, idx, hint, base) -> SurfaceAnchor:
        f = self._face_for(self.anchor_face, idx, base)
        corners = tuple(int(c) for c in body_q[f])
        k = len(corners)
        if self.anchor_weights is None:
            weights = (1.0 / k,) * k
        elif idx == 0:
            weights = tuple(float(w) for w in self.anchor_weights)
        else:
            weights = mirror_anchor_weights(base, body_q, int(self.anchor_face),
                                            self.anchor_weights)
        return SurfaceAnchor(corners=corners, weights=weights, up_hint=hint)

    def _anchor_frame(self, body_v, body_q, idx, pt, base=None):
        """The 4x4 placement of the appendage for side ``idx`` (0 left, 1 right)."""
        hint = np.asarray(self.up_hint, dtype=np.float64)
        if idx == 1:                    # mirrored side needs a mirrored hint
            hint = hint * np.array([-1.0, 1.0, 1.0])
        if self.anchor_face is not None:
            anchor = self._face_anchor(body_v, body_q, idx, hint, base)
        else:
            anchor = SurfaceAnchor.from_point(body_v, body_q, pt, hint)
        anchor.offset = self.offset
        if self._has_root():
            return self._rooted_frame(anchor, body_v, idx, body_q, base)
        roll = self.roll
        rot = _euler(np.radians([roll, 0.0, 0.0]))
        if idx == 1:
            # The mirrored anchor already carries a mirrored tangent, and
            # reflecting the tangent flips the handedness of the frame's
            # bitangent (n' x t' = -M(n x t)). Negating local X on top of that
            # cancels out into a 180-degree roll, not a mirror; negating local
            # Y is what leaves a true mirror. The roll rides inside that
            # reflection, so the pair stays a mirror image at any roll angle.
            rot = np.diag([1.0, -1.0, 1.0]) @ rot
        return anchor_matrix(anchor, body_v, self.scale, rot)

    def _rooted_frame(self, anchor, body_v, idx, body_q=None, base=None) -> np.ndarray:
        """Placement of an appendage that has a ROOT POINT, built from body-space
        vectors rather than from the anchor quad's frame.

        The anchor's tangent frame comes from ONE quad of the morphed body, so a
        shoulder or pectoral slider re-aims it: the same constants gave a span
        axis 19.5 degrees different on the plain body and on the anatomy-adjusted
        one, and the wing's rake was an accident of the sliders. Here the span
        axis is the hint itself, the chord points at the root (so the root is
        in-plane on the back), and only the ORIGIN follows the body, sunk a fixed
        distance into the back. The right wing is the exact mirror of the left.
        """
        resolved, basis = anchor.resolve(body_v)
        pt = resolved - basis[:, 2] * anchor.offset  # the surface point, before any sink
        back = np.array([0.0, 0.0, -1.0])            # MakeHuman: the back faces -Z
        # sink a fixed distance straight into the back, not along one quad's normal
        origin = pt + back * self.offset
        x = np.asarray(self.up_hint, dtype=np.float64)
        rp = self._root_world(pt, idx, body_v, body_q, base)
        mirrored = idx == 1
        if mirrored:
            x = x * np.array([-1.0, 1.0, 1.0])
        x = x / np.linalg.norm(x)
        z = rp - origin
        z = z - float(z @ x) * x                    # the part perpendicular to the span axis
        z = z / max(float(np.linalg.norm(z)), 1e-12)
        y = np.cross(z, x)                          # right-handed (x, y, z) for the left wing
        if mirrored:
            # the exact mirror of the left frame: (x', y', z') = D (x, y, z) with
            # D = diag(-1,1,1), which flips handedness -- so y' = -(z' x x')
            y = -y
        m = np.eye(4)
        m[:3, :3] = np.stack([x, y, z], axis=1) * float(self.scale)
        m[:3, 3] = origin
        return m

    def _root_world(self, surface_pt, idx, body_v=None, body_q=None, base=None) -> np.ndarray:
        """The root target in body space for side ``idx``; ``surface_pt`` is the anchor's
        surface point on that side (before the sink)."""
        if self.root_face is not None:
            f = self._face_for(self.root_face, idx, base)
            off = np.asarray(self.root_offset, dtype=np.float64)
            if idx == 1:
                off = off * np.array([-1.0, 1.0, 1.0])
            return np.asarray(body_v, dtype=np.float64)[body_q[f]].mean(axis=0) + off
        rp = np.asarray(self.root_point, dtype=np.float64)
        if idx == 1:
            rp = rp * np.array([-1.0, 1.0, 1.0])
        return np.asarray(surface_pt, dtype=np.float64) + rp if self.root_relative else rp

    def _roll_toward_root(self, anchor, body_v, idx) -> float:
        """Roll (degrees about the span axis) that puts the attachment line in
        the wing's own plane.

        A wing's plane contains its span axis (aimed up and out) and its chord.
        With a fixed roll the plane slopes forward as it descends the back, so a
        root target projected into it lands on the belly instead of the waist.
        Rolling until the chord points at the target makes the plane contain the
        span axis AND the line to the root: the root is then exactly in-plane
        and on the back, for any body shape.
        """
        origin, basis = anchor.resolve(body_v)
        rp = np.asarray(self.root_point, dtype=np.float64)
        if idx == 1:
            rp = rp * np.array([-1.0, 1.0, 1.0])
        t, bt, n = basis[:, 0], basis[:, 1], basis[:, 2]
        r = rp - origin
        r = r - float(r @ t) * t                    # the part perpendicular to the span axis
        ln = float(np.linalg.norm(r))
        if ln < 1e-9:
            return self.roll
        zb, zn = float(r @ bt) / ln, float(r @ n) / ln
        # local +Z = -sin(roll) bt + cos(roll) n; a mirrored side negates the bt term
        return float(np.degrees(np.arctan2(-zb if idx == 0 else zb, zn)))

    def _points(self):
        points = [np.asarray(self.anchor_point, dtype=np.float64)]
        if self.mirror:
            points.append(points[0] * np.array([-1.0, 1.0, 1.0]))
        return points

    def _indices(self) -> list[int]:
        """The sides to build: 0 = the stored (left-hand) one, 1 = its mirror image."""
        if self.side is None:
            return [0, 1] if self.mirror else [0]
        try:
            return {"center": [0], "left": [0], "right": [1], "both": [0, 1]}[self.side]
        except KeyError:
            raise ValueError(f"unknown side {self.side!r}; use center, left, right or both"
                             ) from None

    def _tag(self, idx: int) -> str:
        """Suffix of the shell name: none for a lone midline appendage, else ``_l``/``_r``."""
        lone = (not self.mirror) if self.side is None else self.side == "center"
        return "" if lone else ("_l" if idx == 0 else "_r")

    def _point(self, idx: int):
        pt = np.asarray(self.anchor_point, dtype=np.float64)
        return pt * np.array([-1.0, 1.0, 1.0]) if idx == 1 else pt

    def place_rigs(self, base, verts=None):
        """``[(idx, PlacedWing)]`` for the sides to build (``idx`` 0 = left, 1 = right).

        The engine behind :meth:`build_rigs`, which only adds the legacy names."""
        if self.kind != "wing":
            raise ValueError(f"only wings are rigged, not {self.kind!r}")
        from .wings import make_wing_rig
        v = base.verts if verts is None else verts
        body_v, body_q = base.body_cage(v)
        shape = {k: x for k, x in self.params.items()
                 if k not in self.POSE_KEYS and k != self.PRESET_KEY}
        style = shape.pop("style", "membrane")
        out = []
        for idx in self._indices():
            M = self._anchor_frame(body_v, body_q, idx, self._point(idx), base)
            kw = dict(shape)
            if self._has_root() and "root" not in kw:
                surface = M[:3, 3] + np.array([0.0, 0.0, self.offset])      # before the sink
                rp = self._root_world(surface, idx, body_v, body_q, base)
                local = np.linalg.inv(M) @ np.append(rp, 1.0)      # into wing units
                kw["root"] = (float(local[0]), float(local[2]))
            rig = make_wing_rig(style, **kw)
            # where the back is, in the wing's own frame: a stowed wing lies against it
            Rm = M[:3, :3] / float(self.scale)
            axes = Rm.T @ np.array([[0.0, 0.0, -1.0], [0.0, -1.0, 0.0],
                                    [-1.0 if idx == 1 else 1.0, 0.0, 0.0]]).T
            rig.skeleton.stow_against(axes[:, 0], axes[:, 1], axes[:, 2])
            out.append((idx, rig.placed(M)))
        return out

    def build_rigs(self, base, verts=None):
        """``[(name, PlacedWing)]`` -- wings as posable rigs on the current body.

        Only wings have a rig. Each carries its skeleton, skin weights and the
        placement that puts it on the morphed body, so the pair can be posed
        (fold, raise, flap) and re-meshed without rebuilding anything.
        """
        return [(f"wing{self._tag(idx)}", pw) for idx, pw in self.place_rigs(base, verts)]

    def pose_controls(self) -> dict:
        """The pose controls riding in ``params`` (e.g. ``fold``)."""
        return {k: x for k, x in self.params.items() if k in self.POSE_KEYS}

    def pose_preset(self):
        """The named pose riding in ``params`` (e.g. ``"folded"``), or ``None``."""
        return self.params.get(self.PRESET_KEY)

    def place_meshes(self, base, verts=None):
        """``[(idx, Mesh)]`` for a tail, horn or harvested limb on the sides to build."""
        v = base.verts if verts is None else verts
        body_v, body_q = base.body_cage(v)
        out = []
        for idx in self._indices():
            pt = self._point(idx)
            if self.kind == "harvest":
                chain = list(self.chain)
                if idx == 1:                    # mirror swaps the side prefix
                    chain = [mirror_joint_name(c) for c in chain]
                mesh = harvest_region(base, chain, v, **self.params)
                src = joint_position(base, chain[0], v)
                shift = pt - src
                if self.anchor_face is not None:
                    # body-relative: the copy's shoulder goes to the quad it is bound to
                    # (so it follows the morphs), then SINKS along the limb axis by
                    # ``offset`` -- the harvest has no surface normal to sink along
                    hint = np.asarray(self.up_hint, dtype=np.float64)
                    anchor = self._face_anchor(body_v, body_q, idx, hint, base)
                    origin, _ = anchor.resolve(body_v)
                    axis = joint_position(base, chain[-1], v) - src
                    ln = float(np.linalg.norm(axis))
                    shift = origin - src + (self.offset * axis / ln if ln > 1e-9 else 0.0)
                mesh = mesh.translated(shift)
            else:
                builder = BUILDERS.get(self.kind)
                if builder is None:
                    raise KeyError(f"unknown appendage kind {self.kind!r}; "
                                   f"have {sorted(BUILDERS) + ['harvest']}")
                mesh = builder(**self.params).transform(
                    self._anchor_frame(body_v, body_q, idx, pt, base))
            out.append((idx, mesh))
        return out

    def place_limbs(self, base, verts=None):
        """``[(idx, LimbRig)]`` for a harvested limb on the sides to build: the character's
        own arm or leg, copied, bound to the body quad ``anchor_face`` and sunk, with a rig
        (:mod:`limbrig`) to pose it.

        The copy's shoulder joint goes to the quad's surface point plus ``offset`` along the
        limb's own axis (negative = back into the torso); ``scale`` scales it about that
        joint. ``params`` are ``radius`` and ``trim`` of :func:`harvest_limb` and ``neck``
        (:func:`limbrig.taper_root`: how much of its width the cut end keeps; 1 = as cut).
        Everything that is a LENGTH (radius, trim, the sink) is given for the neutral body and
        scaled by the ratio of this body's upper arm (leg) to the neutral one, so the same
        record cuts the same limb and welds it as deeply on a child, a tall body or a heavy one.
        The right side is cut from the right limb with the left/right markers swapped and
        carries the mirrored controls; on a symmetric body it is the left one reflected,
        vertex for vertex, so a pair is an exact mirror pair (and a lone right limb is the
        pair's right limb).
        """
        if self.kind != "harvest" or self.anchor_face is None:
            raise ValueError("place_limbs needs a harvest bound to a body quad (anchor_face)")
        from .limbrig import LimbRig
        v = base.verts if verts is None else verts
        body_v, body_q = base.body_cage(v)
        limb = "leg" if "leg" in str(self.chain[0]) else "arm"
        left_chain = list(self.chain)
        right_chain = [mirror_joint_name(c) for c in left_chain]

        def cut(chain):
            s_joint, e_joint, h_joint = (joint_position(base, n, v) for n in chain)
            ls = float(np.linalg.norm(e_joint - s_joint)) / LIMB_REFERENCE_LENGTH[limb]
            kw = dict(self.params)
            kw["radius"] = float(kw.get("radius", 1.15)) * ls
            kw["trim"] = float(kw.get("trim", 0.0)) * ls
            neck = float(kw.pop("neck", 1.0))
            mesh, info = harvest_limb(base, chain, v, **kw)
            if neck < 1.0:
                from .limbrig import taper_root
                mesh = taper_root(mesh, s_joint, e_joint, info.root_loop, info.root_pole,
                                  fraction=neck,
                                  length=NECK_LENGTH * float(np.linalg.norm(e_joint - s_joint)))
            return mesh, info, (s_joint, e_joint, h_joint), ls

        left = cut(left_chain)
        out = []
        for idx in self._indices():
            if idx == 0:
                mesh, info, (s_joint, e_joint, h_joint), ls = left
            else:
                mesh, info, (s_joint, e_joint, h_joint), ls = cut(right_chain)
                lm = left[0]
                if _is_mirror_image(lm.vertices, mesh.vertices):
                    # a symmetric body: the right limb IS the left one reflected, with the
                    # same vertex numbering and the winding flipped (cut from the right arm it
                    # has the same vertices but its quads may split along the other diagonal)
                    mesh = Mesh(lm.vertices * np.array([-1.0, 1.0, 1.0]), lm.faces[:, ::-1])
                    info = left[1]
            anchor = self._face_anchor(body_v, body_q, idx,
                                       np.asarray(self.up_hint, dtype=np.float64), base)
            target, _ = anchor.resolve(body_v)
            axis = h_joint - s_joint
            axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
            # the sink scales with the limb: scaled about the shoulder joint, a bigger limb's
            # cut end would otherwise leave the torso (weld lost beyond scale ~1.5)
            pivot = target + self.offset * ls * self.scale * axis
            place = lambda p: pivot + self.scale * (np.asarray(p, dtype=np.float64) - s_joint)  # noqa: E731
            placed = Mesh(place(mesh.vertices), mesh.faces)
            out.append((idx, LimbRig(placed, pivot, place(e_joint), place(h_joint), limb=limb,
                                     mirrored=idx == 1, root_loop=info.root_loop,
                                     root_pole=info.root_pole)))
        return out

    def build(self, base, verts=None):
        """Return ``[(name, Mesh)]`` placed on the current morphed body."""
        v = base.verts if verts is None else verts
        if self.kind == "wing":
            controls, preset = self.pose_controls(), self.pose_preset()
            return [(name, pw.pose(preset, **controls) if (controls or preset) else pw.mesh())
                    for name, pw in self.build_rigs(base, v)]
        return [(f"{self.kind}{self._tag(idx)}", mesh)
                for idx, mesh in self.place_meshes(base, v)]
