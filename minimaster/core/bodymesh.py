"""Dynamic body mesh: blend primitive shapes into one skinned solid.

Instead of exporting a figure as a pile of independent watertight shells (one
per primitive, unioned by the slicer), this module treats each shape as a
*field source* — a signed-distance function — and fuses them with a smooth
minimum into a single continuous surface. That surface is extracted as one
watertight mesh with a manifold dual-contouring pass (a Surface Nets variant
that stays a 2-manifold even where two sheets cross one voxel), then bound to
the armature with **smooth skin weights** so posing deforms the body across a
joint (an elbow creases) instead of rigidly transforming separate shells.

The pipeline is pure numpy:

    scene shapes ──▶ signed-distance field on a voxel grid (smooth-union blend)
                 ──▶ manifold dual contouring  ──▶ one watertight Mesh
                 ──▶ per-vertex bone weights  ──▶ linear-blend skinning

Resolution is a knob: a coarse grid keeps the chunky/carved low-poly look; a
fine grid melts toward a smooth organic body. Extraction runs in a couple of
seconds, so it's a "bake" step, not a live-drag operation — the primitive
preview stays the fast authoring view.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import math3d as m3
from . import primitives
from .mesh import Mesh

# --------------------------------------------------------------------------
# signed-distance functions (evaluated in each shape's local/canonical frame)
#
# Each SDF takes local points (n, 3) and returns signed distance (negative
# inside) in *local* units. The zero level-set is exact under any affine shape
# transform — a world point is inside the shape iff the canonical SDF of its
# inverse-transformed position is negative — so the fused surface is faithful;
# only off-surface magnitude is mildly distorted by non-uniform scale, which
# just varies the blend width slightly.


def _sdf_sphere(p, radius):
    return np.linalg.norm(p, axis=1) - radius


def _sdf_capsule(p, radius, height):
    h = max(float(height), 2.0 * radius)
    a = h / 2.0 - radius  # half-length of the core segment along Z
    z = np.clip(p[:, 2], -a, a)
    d = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2 + (p[:, 2] - z) ** 2)
    return d - radius


def _sdf_cylinder(p, radius, height):
    dxy = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - radius
    dz = np.abs(p[:, 2]) - height / 2.0
    outside = np.sqrt(np.maximum(dxy, 0.0) ** 2 + np.maximum(dz, 0.0) ** 2)
    inside = np.minimum(np.maximum(dxy, dz), 0.0)
    return outside + inside


def _sdf_box(p, half):
    q = np.abs(p) - np.asarray(half, dtype=np.float64)
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(q.max(axis=1), 0.0)
    return outside + inside


def _canonical_sdf(kind: str, params: dict):
    """Return (sdf_fn(local_pts) -> dist, canonical_radius) for a shape kind.

    ``params`` is merged with the primitive's defaults, so a shape that leaves
    radius/height unset (the common case — the TRS scale does the sizing) still
    resolves to the unit primitive the mesh builder would produce.
    """
    p = primitives.default_params(kind)
    p.update(params or {})
    if kind == "icosphere":
        r = float(p["radius"])
        return (lambda pts: _sdf_sphere(pts, r)), r
    if kind == "capsule":
        r, h = float(p["radius"]), float(p["height"])
        return (lambda pts: _sdf_capsule(pts, r, h)), r
    if kind == "cylinder":
        # taper isn't representable as a plain cylinder SDF; use the mean
        # radius so a cone-ish limb still blends at roughly the right girth.
        r = float(p["radius"]) * (1.0 + float(p.get("taper", 1.0))) / 2.0
        r = max(r, float(p["radius"]) * 0.15)
        h = float(p["height"])
        return (lambda pts: _sdf_cylinder(pts, r, h)), r
    if kind == "box":
        half = np.array([p["width"], p["depth"], p["height"]], dtype=np.float64) / 2.0
        return (lambda pts: _sdf_box(pts, half)), float(half.min())
    if kind == "wedge":
        # a wedge is a cut box; approximate as its bounding box for blending
        half = np.array([p["width"], p["depth"], p["height"]], dtype=np.float64) / 2.0
        return (lambda pts: _sdf_box(pts, half)), float(half.min())
    if kind == "torus":
        # rare in bodies (skeleton ribs); approximate as a flat box
        r, t = float(p["radius"]), float(p["thickness"])
        half = np.array([r + t / 2, r + t / 2, t / 2], dtype=np.float64)
        return (lambda pts: _sdf_box(pts, half)), float(t / 2)
    raise KeyError(f"no SDF for primitive {kind!r}")


def _smin(a, b, k):
    """Polynomial smooth-minimum (Inigo Quilez): blends two SDFs over width k."""
    if k <= 0.0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


# --------------------------------------------------------------------------
# field element: one shape reduced to a world-space SDF


@dataclass
class FieldSource:
    inv: np.ndarray  # 4x4 world->local
    sdf: object  # local SDF fn
    world_scale: float  # local-units -> world-units factor
    reach: float  # world-space radius of influence (for bounds/culling)
    center: np.ndarray  # world-space center


def _shape_source(shape, world: np.ndarray | None = None) -> FieldSource:
    mat = shape.local_matrix()
    if world is not None:  # bake a posed figure: fold the bone's skin matrix in
        mat = world @ mat
    inv = np.linalg.inv(mat)
    sdf, canon_r = _canonical_sdf(shape.kind, shape.params)
    scale = np.abs(shape.scale)
    world_scale = float(scale.min()) if scale.min() > 1e-9 else 1.0
    # a generous world-space reach so the grid bounds enclose the whole shape
    reach = float(canon_r * scale.max() * 2.0 + np.linalg.norm(mat[:3, :3] @ [0, 0, 1]))
    return FieldSource(inv, sdf, world_scale, reach, mat[:3, 3].copy())


def _eval_field(sources, pts, blend):
    """Fused signed distance (world units) at world points ``pts`` (n, 3)."""
    out = None
    for s in sources:
        local = pts @ s.inv[:3, :3].T + s.inv[:3, 3]
        d = s.sdf(local) * s.world_scale
        out = d if out is None else _smin(out, d, blend)
    return out


# --------------------------------------------------------------------------
# Manifold Dual Contouring: like Surface Nets, but a cell whose surface has two
# disconnected sheets (two arms crossing one voxel) emits one vertex PER sheet,
# so the result stays a 2-manifold even where naive Surface Nets pinches.

# cube corners (unit-cell offsets), the 12 edges as corner pairs, and each of
# the 6 faces as its 4 boundary edges
_CORNER = np.array(
    [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
     (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)], dtype=np.float64)
_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3),
          (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
_FACE_EDGES = [[0, 5, 1, 4], [2, 7, 3, 6], [0, 9, 2, 8],
               [1, 11, 3, 10], [4, 10, 6, 8], [5, 11, 7, 9]]
# per axis: the 4 surrounding cells (offset from the low lattice point) in CCW
# order for a +axis normal, and which cube-edge the grid edge is in each cell
_AXIS_QUAD = {
    0: ([(0, -1, -1), (0, 0, -1), (0, 0, 0), (0, -1, 0)], [3, 2, 0, 1]),
    1: ([(-1, 0, -1), (-1, 0, 0), (0, 0, 0), (0, 0, -1)], [7, 5, 4, 6]),
    2: ([(-1, -1, 0), (0, -1, 0), (0, 0, 0), (-1, 0, 0)], [11, 10, 8, 9]),
}


def _uf_find(parent, x):
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


def _uf_union(parent, a, b):
    parent[_uf_find(parent, a)] = _uf_find(parent, b)


def dual_contour(field: np.ndarray, spacing, origin) -> Mesh:
    """Manifold dual contouring of ``field == 0`` (inside < 0).

    Each cell's active edges are grouped into surface components (marching-
    squares connectivity per face, diagonals split), one vertex per component;
    a quad joins the four cells around every sign-changing grid edge, routed to
    each cell's component containing that edge. Guarantees a watertight,
    outward 2-manifold even where two sheets pass through one cell.
    """
    F = np.asarray(field, dtype=np.float64)
    F = np.where(F == 0.0, 1e-9, F)
    spacing = np.broadcast_to(np.asarray(spacing, dtype=np.float64), (3,))
    origin = np.asarray(origin, dtype=np.float64)
    NX, NY, NZ = F.shape

    offs = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
            (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    C = [F[dx:dx + NX - 1, dy:dy + NY - 1, dz:dz + NZ - 1] for dx, dy, dz in offs]
    inside = [c < 0.0 for c in C]
    s = np.sum(inside, axis=0)
    active = (s > 0) & (s < 8)
    if not active.any():
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))

    ci, cj, ck = (a.astype(np.int64) for a in np.nonzero(active))
    Csign = np.stack([inside[c][active] for c in range(8)], axis=1)  # (n, 8)
    Cval = np.stack([C[c][active] for c in range(8)], axis=1)        # (n, 8)
    # field gradient = surface normal, for sharp-feature (QEF) vertex placement
    grad = np.stack(np.gradient(F, spacing[0], spacing[1], spacing[2]), axis=-1)
    Gval = np.stack([grad[dx:dx + NX - 1, dy:dy + NY - 1, dz:dz + NZ - 1][active]
                     for dx, dy, dz in offs], axis=1)  # (n, 8, 3)
    cell_pos = np.full(active.shape, -1, dtype=np.int64)
    cell_pos[active] = np.arange(len(ci))

    verts: list[np.ndarray] = []
    e2v: dict[tuple[int, int], int] = {}  # (active-cell index, cube-edge) -> vertex
    for p in range(len(ci)):
        sgn, val = Csign[p], Cval[p]
        act = [e for e in range(12) if sgn[_EDGES[e][0]] != sgn[_EDGES[e][1]]]
        act_set = set(act)
        parent = {e: e for e in act}
        for fe in _FACE_EDGES:
            fa = [e for e in fe if e in act_set]
            if len(fa) == 2:
                _uf_union(parent, fa[0], fa[1])
            elif len(fa) == 4:  # face saddle: keep the two diagonals apart
                by_inside: dict[int, list[int]] = {}
                for e in fa:
                    a, b = _EDGES[e]
                    ins = a if sgn[a] else b
                    by_inside.setdefault(ins, []).append(e)
                for grp in by_inside.values():
                    for e in grp[1:]:
                        _uf_union(parent, grp[0], e)
        comps: dict[int, list[int]] = {}
        for e in act:
            comps.setdefault(_uf_find(parent, e), []).append(e)
        base = np.array([ci[p], cj[p], ck[p]], dtype=np.float64)
        cell_lo = origin + base * spacing
        gval = Gval[p]
        for es in comps.values():
            pts, qpts, normals = [], [], []
            for e in es:
                a, b = _EDGES[e]
                va, vb = val[a], val[b]
                d = va - vb
                t = min(max(va / d if d != 0.0 else 0.5, 0.0), 1.0)
                loc = _CORNER[a] + t * (_CORNER[b] - _CORNER[a])
                world = origin + (base + loc) * spacing
                pts.append(world)
                g = gval[a] + t * (gval[b] - gval[a])
                gl = np.linalg.norm(g)
                if gl > 1e-9:
                    qpts.append(world)
                    normals.append(g / gl)
            centroid = np.mean(pts, axis=0)
            # QEF: place the vertex where the crossings' tangent planes meet
            # (sharp edges/corners survive), biased to the centroid where the
            # planes are parallel (flat regions stay put)
            if normals:
                A = np.array(normals)
                rhs = np.einsum("ij,ij->i", A, np.array(qpts)) - A @ centroid
                delta, *_ = np.linalg.lstsq(A, rhs, rcond=0.08)
                X = centroid + delta
            else:
                X = centroid
            # keep the vertex inside its own cell: disjoint boxes can't coincide,
            # which preserves the watertight 2-manifold
            X = np.clip(X, cell_lo + 0.05 * spacing, cell_lo + 0.95 * spacing)
            vidx = len(verts)
            verts.append(X)
            for e in es:
                e2v[(p, e)] = vidx

    faces: list[tuple[int, int, int]] = []
    for axis, (cell_offs, cube_edges) in _AXIS_QUAD.items():
        lo_slices = [slice(0, NX - 1), slice(1, NY - 1), slice(1, NZ - 1)]
        lo_slices[axis] = slice(0, [NX, NY, NZ][axis] - 1)
        # inner cells only on the two non-axis dims (so all 4 neighbours exist)
        for d in range(3):
            if d != axis:
                lo_slices[d] = slice(1, [NX, NY, NZ][d] - 1)
        hi = [slice(sl.start, sl.stop) for sl in lo_slices]
        hi[axis] = slice(lo_slices[axis].start + 1, lo_slices[axis].stop + 1)
        Fa = F[lo_slices[0], lo_slices[1], lo_slices[2]]
        Fb = F[hi[0], hi[1], hi[2]]
        idx = np.nonzero((Fa < 0.0) != (Fb < 0.0))
        low = [idx[0], idx[1], idx[2]]
        # map local nonzero indices back to grid lattice coords
        low = [low[d] + lo_slices[d].start for d in range(3)]
        for a in range(len(low[0])):
            i, j, k = int(low[0][a]), int(low[1][a]), int(low[2][a])
            vs = []
            for (dx, dy, dz), e in zip(cell_offs, cube_edges):
                vs.append(e2v[(int(cell_pos[i + dx, j + dy, k + dz]), e)])
            if F[i, j, k] >= 0.0:  # low endpoint outside -> reverse for outward
                vs = vs[::-1]
            faces.append((vs[0], vs[1], vs[2]))
            faces.append((vs[0], vs[2], vs[3]))

    if not verts or not faces:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    mesh = Mesh(np.array(verts), np.array(faces, dtype=np.int64))
    if mesh.volume() < 0.0:
        mesh = Mesh(mesh.vertices, mesh.faces[:, [0, 2, 1]])
    return mesh


# --------------------------------------------------------------------------
# scene -> body mesh


# shapes whose names carry these substrings are treated as gear/detail, not
# body — kept out of the fused body by default (they can layer on top instead)
ACCESSORY_KEYWORDS = (
    "sword", "shield", "axe", "club", "dagger", "cape", "belt", "collar",
    "eye", "hair", "beard", "tusk", "ear", "mouth", "brow", "nose",
    "rib", "spine_column",
)


def _mesh_from_sources(sources, resolution: float, blend: float,
                       pad: float = 3.0) -> Mesh:
    sources = list(sources)
    if not sources:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for s in sources:
        lo = np.minimum(lo, s.center - s.reach)
        hi = np.maximum(hi, s.center + s.reach)
    margin = pad * resolution + blend
    lo -= margin
    hi += margin

    dims = np.maximum(np.ceil((hi - lo) / resolution).astype(int) + 1, 2)
    xs = lo[0] + np.arange(dims[0]) * resolution
    ys = lo[1] + np.arange(dims[1]) * resolution
    zs = lo[2] + np.arange(dims[2]) * resolution
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    field = _eval_field(sources, pts, blend).reshape(dims)
    # seal the border so the surface can't run off the grid edge
    field[0, :, :] = field[-1, :, :] = 1e3
    field[:, 0, :] = field[:, -1, :] = 1e3
    field[:, :, 0] = field[:, :, -1] = 1e3

    return dual_contour(field, resolution, lo)


def build_field_mesh(shapes, resolution: float = 0.7, blend: float = 0.0,
                     pad: float = 3.0) -> Mesh:
    """Fuse ``shapes`` (at their rest transforms) into one watertight mesh at
    the given voxel ``resolution`` (world units per cell; smaller = smoother).
    ``blend`` is the smooth-union width in world units (0 = a hard union)."""
    return _mesh_from_sources([_shape_source(s) for s in shapes],
                              resolution, blend, pad)


def _region_of(bone, armature) -> str:
    """Which anatomical part a bone belongs to: the head, one limb, or the
    axial 'core' (torso+pelvis). Limbs are keyed by their root joint so each
    arm/leg is its own region."""
    if not bone or bone not in armature.joints:
        return "core"
    for name in [bone, *armature.ancestors(bone)]:
        if name == "head_top":
            return "head"
        if name.startswith(("shoulder2_", "shoulder_", "hip_")):
            return name
    return "core"


def body_regions(scene, resolution: float = 0.5, blend: float = 0.8,
                 pose_name: str | None = None, exclude=ACCESSORY_KEYWORDS):
    """Fuse the body into ONE clean solid per anatomical region — head,
    torso/pelvis ('core'), and each arm and leg — rather than one whole-body
    blob. Limbs stay distinct (they don't melt into the torso) while their own
    muscles fuse smoothly *within* the part; regions overlap at the joints, so
    the union prints as one connected figure. Each part also carries its own
    (modal) color. Returns a list of ``(region, mesh, color)``.

    Falls back to a single 'core' fuse when the scene has no humanoid armature.
    """
    pose = None if pose_name in (None, "rest") else scene.resolve_pose(pose_name)
    skins = scene.armature.skin_matrices(pose) if pose else {}
    groups: dict[str, list] = {}
    for s in scene.shapes:
        if exclude and any(k in s.name for k in exclude):
            continue
        groups.setdefault(_region_of(s.bone, scene.armature), []).append(s)
    out = []
    for region, shapes in groups.items():
        mesh = _mesh_from_sources(
            [_shape_source(s, skins.get(s.bone)) for s in shapes], resolution, blend)
        if not len(mesh.faces):
            continue
        counts: dict[str, int] = {}
        for s in shapes:  # the region wears its most common shape color
            counts[s.color] = counts.get(s.color, 0) + 1
        out.append((region, mesh, max(counts, key=counts.get)))
    return out


def body_from_scene(scene, resolution: float = 0.5, blend: float = 0.8,
                    pose_name: str | None = None, exclude=ACCESSORY_KEYWORDS) -> Mesh:
    """One watertight body mesh: the union of the per-region solids (see
    :func:`body_regions`). Regions overlap at the joints, so the merged shells
    print as a single connected figure — the same overlapping-shells contract
    the shape export already relies on, but each shell is a cleanly fused part.
    """
    parts = body_regions(scene, resolution, blend, pose_name, exclude)
    return Mesh.merge([m for _, m, _ in parts])


# --------------------------------------------------------------------------
# smooth skinning: bind body-mesh vertices to the armature by distance


def skin_weights(vertices: np.ndarray, armature, max_bones: int = 4,
                 power: float = 4.0, eps: float = 0.35):
    """Per-vertex linear-blend weights over the nearest ``max_bones`` bones.

    Weight falls off as 1 / (distance-to-bone + eps)**power, keeping the
    closest few bones and renormalizing. Returns (bone_names, weights) where
    weights is (n_verts, n_kept_bones) aligned to bone_names.
    """
    bones = armature.bones()  # (parent, child) pairs; child name is the key
    if not bones:
        return [], np.zeros((len(vertices), 0))
    names = [child for _, child in bones]
    A = np.array([armature.joints[p].position for p, _ in bones])  # (b, 3)
    B = np.array([armature.joints[c].position for _, c in bones])  # (b, 3)

    V = np.asarray(vertices, dtype=np.float64)
    AB = B - A  # (b, 3)
    denom = np.einsum("bj,bj->b", AB, AB)
    denom = np.where(denom < 1e-12, 1.0, denom)
    # projection parameter t of each vertex onto each bone segment, clamped
    t = np.einsum("vbj,bj->vb", V[:, None, :] - A[None, :, :], AB) / denom
    t = np.clip(t, 0.0, 1.0)
    closest = A[None, :, :] + t[:, :, None] * AB[None, :, :]  # (v, b, 3)
    dist = np.linalg.norm(V[:, None, :] - closest, axis=2)  # (v, b)

    w = 1.0 / (dist + eps) ** power
    keep = min(max_bones, w.shape[1])
    # zero out all but the top-`keep` bones per vertex
    cut = np.sort(w, axis=1)[:, -keep][:, None]
    w = np.where(w >= cut, w, 0.0)
    w /= w.sum(axis=1, keepdims=True)
    return names, w


def pose_body_mesh(mesh: Mesh, bone_names, weights, armature, pose) -> Mesh:
    """Deform a rest-pose body mesh by linear-blend skinning under ``pose``."""
    if not bone_names:
        return mesh.copy()
    skins = armature.skin_matrices(pose)
    V = mesh.vertices
    out = np.zeros_like(V)
    for bi, name in enumerate(bone_names):
        wcol = weights[:, bi]
        active = wcol > 0.0
        if not active.any():
            continue
        M = skins[name]
        moved = V[active] @ M[:3, :3].T + M[:3, 3]
        out[active] += wcol[active, None] * moved
    return Mesh(out, mesh.faces)
