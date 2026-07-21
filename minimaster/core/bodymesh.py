"""Dynamic body mesh: blend primitive shapes into one skinned solid.

Instead of exporting a figure as a pile of independent watertight shells (one
per primitive, unioned by the slicer), this module treats each shape as a
*field source* — a signed-distance function — and fuses them with a smooth
minimum into a single continuous surface. That surface is extracted as one
watertight mesh with a vectorized Surface Nets pass, then bound to the armature
with **smooth skin weights** so posing deforms the body continuously across a
joint (an elbow creases) instead of rigidly transforming separate shells.

The pipeline is pure numpy:

    scene shapes ──▶ signed-distance field on a voxel grid (smooth-union blend)
                 ──▶ Surface Nets  ──▶ one watertight Mesh
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
# Surface Nets (vectorized): field grid -> watertight quad-derived triangle mesh


def surface_nets(field: np.ndarray, spacing, origin) -> Mesh:
    """Extract the ``field == 0`` isosurface (inside < 0) as a watertight Mesh.

    Naive Surface Nets: one vertex per grid cell that straddles the surface,
    placed at the average of its edge crossings; a quad joins the four cells
    around every sign-changing grid edge. The result is a closed 2-manifold,
    consistently outward-oriented.
    """
    F = np.asarray(field, dtype=np.float64)
    F = np.where(F == 0.0, 1e-9, F)  # avoid ambiguous exact-zero corners
    spacing = np.broadcast_to(np.asarray(spacing, dtype=np.float64), (3,))
    origin = np.asarray(origin, dtype=np.float64)
    NX, NY, NZ = F.shape

    offs = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
            (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    C = np.stack([F[dx:dx + NX - 1, dy:dy + NY - 1, dz:dz + NZ - 1]
                  for dx, dy, dz in offs])  # (8, cx, cy, cz)
    inside = C < 0.0
    s = inside.sum(0)
    active = (s > 0) & (s < 8)

    edges = [(0, 1), (2, 3), (4, 5), (6, 7),
             (0, 2), (1, 3), (4, 6), (5, 7),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    off_arr = np.array(offs, dtype=np.float64)
    accum = np.zeros((3,) + active.shape)
    count = np.zeros(active.shape)
    for a, b in edges:
        Fa, Fb = C[a], C[b]
        sc = (Fa < 0.0) != (Fb < 0.0)
        denom = Fa - Fb
        t = np.where(sc, Fa / np.where(denom == 0.0, 1.0, denom), 0.0)
        oa, ob = off_arr[a], off_arr[b]
        for ax in range(3):
            accum[ax] += np.where(sc, oa[ax] + t * (ob[ax] - oa[ax]), 0.0)
        count += sc
    cnt = np.where(count > 0.0, count, 1.0)
    # clamp strictly inside the cell so each cell's vertex lives in its own
    # disjoint box: no two vertices can coincide, which kills zero-area slivers
    vlocal = np.clip(accum / cnt, 0.08, 0.92)  # (3, cx, cy, cz)

    ci, cj, ck = np.nonzero(active)
    voff = np.stack([vlocal[0][active], vlocal[1][active], vlocal[2][active]], axis=1)
    verts = origin + (np.stack([ci, cj, ck], axis=1) + voff) * spacing
    vid = np.full(active.shape, -1, dtype=np.int64)
    vid[active] = np.arange(int(active.sum()))

    quads: list[np.ndarray] = []

    def emit(ijk_low, cells, F_low):
        """cells: list of 4 (i,j,k) index-arrays in CCW order for +axis normal."""
        v = [vid[c] for c in cells]  # each (n,)
        quad = np.stack(v, axis=1)  # (n, 4) as A,B,C,D
        flip = F_low < 0.0  # inside at low endpoint -> normal already +axis
        # keep A,B,C,D when low endpoint is inside, else reverse the loop
        ordered = np.where(flip[:, None], quad, quad[:, ::-1])
        quads.append(ordered)

    # x-edges: low lattice (i,j,k) -> (i+1,j,k); cells vary in (j,k)
    Fa = F[0:NX - 1, 1:NY - 1, 1:NZ - 1]
    Fb = F[1:NX, 1:NY - 1, 1:NZ - 1]
    ii, jj, kk = np.nonzero((Fa < 0.0) != (Fb < 0.0))
    i, j, k = ii, jj + 1, kk + 1
    emit((i, j, k), [(i, j - 1, k - 1), (i, j, k - 1), (i, j, k), (i, j - 1, k)],
         F[i, j, k])

    # y-edges: low (i,j,k) -> (i,j+1,k); cells vary in (k,i)  [(e1,e2)=(z,x)]
    Fa = F[1:NX - 1, 0:NY - 1, 1:NZ - 1]
    Fb = F[1:NX - 1, 1:NY, 1:NZ - 1]
    ii, jj, kk = np.nonzero((Fa < 0.0) != (Fb < 0.0))
    i, j, k = ii + 1, jj, kk + 1
    emit((i, j, k), [(i - 1, j, k - 1), (i - 1, j, k), (i, j, k), (i, j, k - 1)],
         F[i, j, k])

    # z-edges: low (i,j,k) -> (i,j,k+1); cells vary in (i,j)  [(e1,e2)=(x,y)]
    Fa = F[1:NX - 1, 1:NY - 1, 0:NZ - 1]
    Fb = F[1:NX - 1, 1:NY - 1, 1:NZ]
    ii, jj, kk = np.nonzero((Fa < 0.0) != (Fb < 0.0))
    i, j, k = ii + 1, jj + 1, kk
    emit((i, j, k), [(i - 1, j - 1, k), (i, j - 1, k), (i, j, k), (i - 1, j, k)],
         F[i, j, k])

    if not quads or not len(verts):
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    Q = np.vstack(quads)  # (m, 4)
    tris = np.vstack([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])
    return Mesh(verts, tris)


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

    mesh = surface_nets(field, resolution, lo)
    if mesh.volume() < 0.0:  # normalize to outward orientation
        mesh = Mesh(mesh.vertices, mesh.faces[:, [0, 2, 1]])
    return mesh


def build_field_mesh(shapes, resolution: float = 0.7, blend: float = 0.0,
                     pad: float = 3.0) -> Mesh:
    """Fuse ``shapes`` (at their rest transforms) into one watertight mesh at
    the given voxel ``resolution`` (world units per cell; smaller = smoother).
    ``blend`` is the smooth-union width in world units (0 = a hard union)."""
    return _mesh_from_sources([_shape_source(s) for s in shapes],
                              resolution, blend, pad)


def _scene_sources(scene, pose, exclude):
    """FieldSources for a scene's body shapes, folding in each shape's posed
    bone transform when ``pose`` is given (bakes the figure as posed)."""
    skins = scene.armature.skin_matrices(pose) if pose else {}
    out = []
    for s in scene.shapes:
        if exclude and any(k in s.name for k in exclude):
            continue
        out.append(_shape_source(s, skins.get(s.bone)))
    return out


def body_from_scene(scene, resolution: float = 0.6, blend: float = 0.6,
                    pose_name: str | None = None, exclude=ACCESSORY_KEYWORDS) -> Mesh:
    """Bake a scene's body shapes into one watertight solid, posed if asked.

    This is the export path: it fuses the *posed* primitive stack directly, so
    there's no skinning to distort — limbs that a pose spreads apart fuse
    cleanly. ``exclude`` drops gear/detail shapes by name substring; pass an
    empty tuple to fuse absolutely everything.
    """
    pose = None if pose_name in (None, "rest") else scene.resolve_pose(pose_name)
    return _mesh_from_sources(_scene_sources(scene, pose, exclude), resolution, blend)


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
