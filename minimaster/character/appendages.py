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

from dataclasses import dataclass, field

import numpy as np

from ..core.mesh import Mesh


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


def place(mesh: Mesh, anchor: SurfaceAnchor, verts, scale=1.0,
          rotation=None) -> Mesh:
    """Put an appendage (authored +Z = outward, origin at its base) onto the
    body at ``anchor``, resolved against the current morphed ``verts``."""
    origin, basis = anchor.resolve(verts)
    m = np.eye(4)
    rot = basis if rotation is None else basis @ rotation
    m[:3, :3] = rot * float(scale)
    m[:3, 3] = origin
    return mesh.transform(m)


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


def harvest_region(base, chain, verts=None, radius=1.15, trim=0.0):
    """Cut a closed shell out of the body around a bone chain.

    ``chain`` is a list of joint-marker names, e.g.
    ``["l-shoulder", "l-elbow", "l-hand"]``. A body quad joins the harvest when
    all four of its corners lie within ``radius`` of the chain and beyond
    ``trim`` along it (``trim`` shaves the root so the copy starts below the
    shoulder). The cut is fan-capped, so the result is a printable solid.
    """
    v = base.verts if verts is None else verts
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
    return Mesh(hv, hf)


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


def make_tail(length=6.0, base_radius=0.45, segments=12, ring_verts=10,
              curl=0.55, droop=0.9) -> Mesh:
    """A tapered, curling tail. +Z is outward from the attachment point."""
    basis = np.eye(3)
    rings = []
    pos = np.zeros(3)
    direction = np.array([0.0, 0.0, 1.0])
    step = length / segments
    for i in range(segments + 1):
        t = i / segments
        r = base_radius * (1.0 - t) ** 0.85 + 0.02
        # bend progressively: droop away from the body and curl sideways
        ang = np.array([droop * t * 0.9, curl * t * 0.7, 0.0]) * step
        rot = _euler(ang)
        direction = rot @ direction
        b = _frame(direction)
        rings.append(_ring(pos, r, r * 0.92, b, ring_verts))
        pos = pos + direction * step
    return _loft(rings)


def make_horn(length=2.2, base_radius=0.42, segments=8, ring_verts=8,
              curve=1.1, twist=0.0) -> Mesh:
    basis = np.eye(3)
    rings = []
    pos = np.zeros(3)
    direction = np.array([0.0, 0.0, 1.0])
    step = length / segments
    for i in range(segments + 1):
        t = i / segments
        r = base_radius * (1.0 - t) ** 1.25 + 0.012
        rot = _euler(np.array([curve * t, twist * t, 0.0]) * step)
        direction = rot @ direction
        rings.append(_ring(pos, r, r, _frame(direction), ring_verts))
        pos = pos + direction * step
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
WING_UP_HINT = (1.0, 1.5, 0.2)
WING_ROLL = -95.0
# and sunk into the back, because a wing that only grazes the skin is joined
# along a hairline: at offset 0 just 2% of the root lies inside the torso,
# against ~40% here, which is a weld a 32 mm print can actually survive.
WING_OFFSET = -0.9


def back_wings(style: str = "membrane", scale: float = 1.0,
               mirror: bool = True, **params) -> "Appendage":
    """A mirrored pair of wings sitting on the shoulder blades.

    Wraps the placement constants so callers do not have to rediscover the
    orientation, which is the difference between wings that are attached and
    wings that float behind the back.
    """
    return Appendage("wing", anchor_point=WING_ANCHOR, mirror=mirror,
                     up_hint=WING_UP_HINT, roll=WING_ROLL, offset=WING_OFFSET,
                     scale=scale, params=dict(style=style, **params))


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

    def build(self, base, verts=None):
        """Return ``[(name, Mesh)]`` placed on the current morphed body."""
        v = base.verts if verts is None else verts
        body_v, body_q = base.body_cage(v)
        out = []
        points = [np.asarray(self.anchor_point, dtype=np.float64)]
        if self.mirror:
            points.append(points[0] * np.array([-1.0, 1.0, 1.0]))
        for idx, pt in enumerate(points):
            if self.kind == "harvest":
                chain = list(self.chain)
                if idx == 1:                    # mirror swaps the side prefix
                    chain = [c.replace("l-", "@").replace("r-", "l-")
                              .replace("@", "r-") for c in chain]
                mesh = harvest_region(base, chain, v, **self.params)
                src = joint_position(base, chain[0], v)
                mesh = mesh.translated(pt - src)
            else:
                builder = BUILDERS.get(self.kind)
                if builder is None:
                    raise KeyError(f"unknown appendage kind {self.kind!r}; "
                                   f"have {sorted(BUILDERS) + ['harvest']}")
                proto = builder(**self.params)
                hint = np.asarray(self.up_hint, dtype=np.float64)
                if idx == 1:            # mirrored side needs a mirrored hint
                    hint = hint * np.array([-1.0, 1.0, 1.0])
                anchor = SurfaceAnchor.from_point(body_v, body_q, pt, hint)
                anchor.offset = self.offset
                rot = _euler(np.radians([self.roll, 0.0, 0.0]))
                if idx == 1:
                    # The mirrored anchor already carries a mirrored tangent,
                    # and reflecting the tangent flips the handedness of the
                    # frame's bitangent (n' x t' = -M(n x t)). Negating local X
                    # on top of that cancels out into a 180-degree roll, not a
                    # mirror; negating local Y is what leaves a true mirror.
                    # The roll rides inside that reflection, so the pair stays
                    # a mirror image at any roll angle.
                    rot = np.diag([1.0, -1.0, 1.0]) @ rot
                mesh = place(proto, anchor, body_v, self.scale, rot)
            side = "" if len(points) == 1 else ("_l" if idx == 0 else "_r")
            out.append((f"{self.kind}{side}", mesh))
        return out
