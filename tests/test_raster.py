"""The vectorised visibility-buffer rasteriser, the shading caches and picking.

The golden arrangement: the per-triangle Python loop that ``render_portrait``
used to be is kept here, verbatim, as ``_reference_render_portrait`` (together
with the ``np.add.at`` versions of the normals and cavity AO it used). The new
renderer must reproduce its output *pixel for pixel* in float64 -- RGB and alpha
-- on a golden set (``test_golden_*``); ``_reference_ids`` is the same loop cut
down to the visibility buffer, used to attack the rasteriser directly with
adversarial triangle soups (exact ties, vertices on pixel centres, huge and
off-screen triangles). The reference is slow (milliseconds per triangle), so
the asset scenes here are small; ``MM_GOLDEN_FULL=1`` runs the full 320x480
bat and angel scenes (several minutes).
"""

import os
import threading
import time

import numpy as np
import pytest

from minimaster.character import appendages as A
from minimaster.character import mhbase as mh
from minimaster.character import portrait as P
from minimaster.character import raster as R
from minimaster.character import shading as sh
from minimaster.core import math3d as m3
from minimaster.core import primitives
from minimaster.core import raycast as rc
from minimaster.core.mesh import Mesh
from minimaster.core.subdiv import cube_cage, quads_to_mesh

needs_assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")


# ==========================================================================
# the reference: the code this module replaced, kept verbatim (np.add.at and
# the per-triangle loop). Do not "improve" it.


def _ref_vertex_normals(mesh):
    v, f = mesh.vertices, mesh.faces
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    out = np.zeros_like(v)
    for i in range(3):
        np.add.at(out, f[:, i], fn)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, n, out=np.zeros_like(out), where=n > 1e-12)


def _ref_neighbours(faces):
    a = faces
    src = np.concatenate([a[:, i] for i in range(a.shape[1])])
    dst = np.concatenate([a[:, (i + 1) % a.shape[1]] for i in range(a.shape[1])])
    return np.concatenate([src, dst]), np.concatenate([dst, src])


def _ref_smooth(values, faces, iterations=2):
    v = np.asarray(values, dtype=np.float64).copy()
    src, dst = _ref_neighbours(faces)
    for _ in range(int(iterations)):
        acc = np.zeros(len(v))
        cnt = np.zeros(len(v))
        np.add.at(acc, src, v[dst])
        np.add.at(cnt, src, 1.0)
        cnt = np.maximum(cnt, 1.0)
        v = 0.5 * v + 0.5 * (acc / cnt)
    return v


def _ref_cavity_ao(verts, faces, normals, strength=1.0, radius_scale=1.0, smooth=2):
    n = len(verts)
    src, dst = _ref_neighbours(faces)
    acc = np.zeros((n, 3))
    cnt = np.zeros(n)
    np.add.at(acc, src, verts[dst])
    np.add.at(cnt, src, 1.0)
    cnt = np.maximum(cnt, 1.0)
    mean_nb = acc / cnt[:, None]
    delta = mean_nb - verts
    edge = np.zeros(n)
    np.add.at(edge, src, np.linalg.norm(verts[dst] - verts[src], axis=1))
    edge = np.maximum(edge / cnt, 1e-9) * radius_scale
    proj = np.einsum("ij,ij->i", delta, normals) / edge
    proj = _ref_smooth(proj, faces, smooth)
    occ = 1.0 / (1.0 + np.exp(-3.0 * proj))
    ao = 1.0 - strength * np.clip((occ - 0.5) * 2.0, 0.0, 1.0)
    return np.clip(ao, 0.0, 1.0)


def _reference_render_portrait(shells, size=(600, 800), azimuth=25.0, elevation=6.0,
                               background="#20242c", fov=26.0, rig=None,
                               exposure=1.15, fit=None, ao_strength=1.0):
    w, h = size
    rig = rig or sh.LightRig()
    img = np.zeros((h, w, 4), dtype=np.uint8)
    if background is not None:
        bg = sh.linear_to_srgb(sh.tonemap(sh.hex_to_linear(background), 1.0))
        img[:, :, :3] = (bg * 255).astype(np.uint8)
        img[:, :, 3] = 255

    shells = [s for s in shells if len(s.mesh.faces)]
    if not shells:
        return img

    combined = Mesh.merge([s.mesh for s in shells])
    lo, hi = combined.bounds
    center = (lo + hi) / 2.0
    radius = float(np.linalg.norm(hi - lo)) / 2.0 or 1.0
    if fit is not None:  # (center, radius) override, e.g. to frame the head
        center, radius = np.asarray(fit[0], dtype=np.float64), float(fit[1])

    az, el = np.radians(azimuth), np.radians(elevation)
    distance = radius / np.tan(np.radians(fov) / 2.0) * 1.05
    eye = center + distance * np.array(
        [np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)])
    view = m3.look_at(eye, center)
    focal = (h / 2.0) / np.tan(np.radians(fov) / 2.0)
    zbuf = np.full((h, w), np.inf)

    for shell in shells:
        mesh = shell.mesh
        vn = _ref_vertex_normals(mesh)
        ao = None
        if shell.ao:
            ao = _ref_cavity_ao(mesh.vertices, mesh.faces, vn,
                                strength=ao_strength * shell.ao_strength)
        cam = m3.transform_points(view, mesh.vertices)
        n_cam = vn @ view[:3, :3].T
        n_cam /= np.maximum(np.linalg.norm(n_cam, axis=1, keepdims=True), 1e-12)
        vd = -cam
        vd /= np.maximum(np.linalg.norm(vd, axis=1, keepdims=True), 1e-12)

        vcol = sh.shade_vertices(n_cam, vd, shell.material, rig, ao,
                                 base_colors=shell.base_colors)

        tris = cam[mesh.faces]
        depths = -tris[:, :, 2]
        fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        facing = np.einsum("ij,ij->i", fn, tris.mean(axis=1)) < 0.0
        valid = facing & (depths.min(axis=1) > 1e-6)
        xs = w / 2.0 + focal * tris[:, :, 0] / depths
        ys = h / 2.0 - focal * tris[:, :, 1] / depths
        cols = vcol[mesh.faces]  # (m, 3, 3)

        for i in np.nonzero(valid)[0]:
            tx, ty, tz = xs[i], ys[i], depths[i]
            x0 = max(int(np.floor(tx.min())), 0)
            x1 = min(int(np.ceil(tx.max())) + 1, w)
            y0 = max(int(np.floor(ty.min())), 0)
            y1 = min(int(np.ceil(ty.max())) + 1, h)
            if x0 >= x1 or y0 >= y1:
                continue
            px, py = np.meshgrid(np.arange(x0, x1) + 0.5,
                                 np.arange(y0, y1) + 0.5)
            d = ((tx[1] - tx[0]) * (ty[2] - ty[0])
                 - (tx[2] - tx[0]) * (ty[1] - ty[0]))
            if abs(d) < 1e-12:
                continue
            l1 = ((px - tx[0]) * (ty[2] - ty[0])
                  - (py - ty[0]) * (tx[2] - tx[0])) / d
            l2 = ((py - ty[0]) * (tx[1] - tx[0])
                  - (px - tx[0]) * (ty[1] - ty[0])) / d
            l0 = 1.0 - l1 - l2
            inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
            if not inside.any():
                continue
            inv_z = l0 / tz[0] + l1 / tz[1] + l2 / tz[2]
            depth = np.where(inv_z > 1e-12, 1.0 / np.maximum(inv_z, 1e-12), np.inf)
            region = zbuf[y0:y1, x0:x1]
            upd = inside & (depth < region)
            if not upd.any():
                continue
            region[upd] = depth[upd]
            c = (l0[:, :, None] * cols[i, 0]
                 + l1[:, :, None] * cols[i, 1]
                 + l2[:, :, None] * cols[i, 2])
            rgb = (sh.linear_to_srgb(sh.tonemap(c, exposure)) * 255).astype(np.uint8)
            tgt = img[y0:y1, x0:x1]
            tgt[..., :3][upd] = rgb[upd]
            tgt[..., 3][upd] = 255
    return img


def _reference_ids(sx, sy, depth, faces, size, face_ids=None):
    """The loop's inner part only: (id buffer, depth buffer) by strict ``<`` in
    ascending triangle order, as the old renderer drew."""
    w, h = size
    zbuf = np.full((h, w), np.inf)
    ids = np.full((h, w), -1, dtype=np.int32)
    xs, ys, zs = sx[faces], sy[faces], depth[faces]
    for i in (range(len(faces)) if face_ids is None else face_ids):
        tx, ty, tz = xs[i], ys[i], zs[i]
        x0 = max(int(np.floor(tx.min())), 0)
        x1 = min(int(np.ceil(tx.max())) + 1, w)
        y0 = max(int(np.floor(ty.min())), 0)
        y1 = min(int(np.ceil(ty.max())) + 1, h)
        if x0 >= x1 or y0 >= y1:
            continue
        px, py = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        d = (tx[1] - tx[0]) * (ty[2] - ty[0]) - (tx[2] - tx[0]) * (ty[1] - ty[0])
        if abs(d) < 1e-12:
            continue
        l1 = ((px - tx[0]) * (ty[2] - ty[0]) - (py - ty[0]) * (tx[2] - tx[0])) / d
        l2 = ((py - ty[0]) * (tx[1] - tx[0]) - (px - tx[0]) * (ty[1] - ty[0])) / d
        l0 = 1.0 - l1 - l2
        inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        if not inside.any():
            continue
        inv_z = l0 / tz[0] + l1 / tz[1] + l2 / tz[2]
        dep = np.where(inv_z > 1e-12, 1.0 / np.maximum(inv_z, 1e-12), np.inf)
        region = zbuf[y0:y1, x0:x1]
        upd = inside & (dep < region)
        if not upd.any():
            continue
        region[upd] = dep[upd]
        ids[y0:y1, x0:x1][upd] = i
    return ids, zbuf


def _assert_same_image(new, ref, what=""):
    assert new.shape == ref.shape and new.dtype == ref.dtype
    diff = (new != ref).any(axis=2)
    assert not diff.any(), (
        f"{what}: {int(diff.sum())} of {diff.size} pixels differ "
        f"(max channel diff {int(np.abs(new.astype(int) - ref.astype(int)).max())})")


# ==========================================================================
# scenes


def _uv_sphere(n_lon, n_lat, radius=1.0):
    """A closed UV sphere with 2 * n_lon * (n_lat - 1) triangles."""
    lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    lat = np.linspace(0, np.pi, n_lat + 1)[1:-1]
    ring = np.array([[np.cos(a) * np.sin(b), np.sin(a) * np.sin(b), np.cos(b)]
                     for b in lat for a in lon])
    verts = np.vstack([[0, 0, 1.0], ring, [0, 0, -1.0]]) * radius
    faces = []
    for j in range(n_lon):
        k = (j + 1) % n_lon
        faces.append([0, 1 + k, 1 + j])
        faces.append([len(verts) - 1, 1 + (n_lat - 2) * n_lon + j, 1 + (n_lat - 2) * n_lon + k])
    for i in range(n_lat - 2):
        for j in range(n_lon):
            k = (j + 1) % n_lon
            a, b = 1 + i * n_lon + j, 1 + i * n_lon + k
            c, d = a + n_lon, b + n_lon
            faces += [[a, b, d], [a, d, c]]
    return Mesh(verts, np.array(faces))


def _sphere(subdiv=2, center=(0, 0, 0), radius=1.0, material=None, **kw):
    """subdiv 0..3: icosphere (20..1280 tris); larger: a UV sphere of ~20 * 4^(subdiv-3)... tris."""
    if subdiv <= 3:
        mesh = primitives.icosphere(radius=radius, subdivisions=subdiv)
    else:
        n = 32 * 2 ** (subdiv - 3)
        mesh = _uv_sphere(n, n // 2, radius)
    return P.Shell(mesh.translated(center), material or sh.SKIN, **kw)


def _giant_quad_shell(z=-1.2, half=6.0):
    """A ground plane of two big triangles: far larger than any size class."""
    v = np.array([[-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]])
    return P.Shell(Mesh(v, [[0, 1, 2], [0, 2, 3]]), sh.GENERIC, ao=False)


def _synthetic_scenes():
    cube = P.Shell(quads_to_mesh(*cube_cage(1.3)).translated((0.9, -0.2, 0.1)), sh.TEETH)
    return {
        "sphere": [_sphere(3)],
        "overlap": [_sphere(2), _sphere(2, (0.7, 0.2, 0.1), 0.8, sh.EYE), cube],
        "ground": [_sphere(2), _giant_quad_shell()],
        # the very same geometry in two shells: every pixel is an exact depth
        # tie, and the lowest triangle id (first shell) must win, as it drew first
        "coincident": [_sphere(2, material=sh.SKIN), _sphere(2, material=sh.EYE)],
    }


SYNTH_VIEWS = [(25.0, 6.0, None, (97, 131)), (200.0, -20.0, None, (150, 90)),
               (90.0, 10.0, ((0.3, 0.0, 0.6), 0.25), (128, 128))]


@pytest.mark.parametrize("scene", sorted(_synthetic_scenes()))
@pytest.mark.parametrize("az,el,fit,size", SYNTH_VIEWS)
def test_synthetic_scenes_are_pixel_identical_to_the_old_loop(scene, az, el, fit, size):
    shells = _synthetic_scenes()[scene]
    new = P.render_portrait(shells, size=size, azimuth=az, elevation=el, fit=fit)
    ref = _reference_render_portrait(shells, size=size, azimuth=az, elevation=el, fit=fit)
    _assert_same_image(new, ref, scene)
    assert (new[:, :, 3] == ref[:, :, 3]).all()


def test_coincident_shells_resolve_ties_to_the_first_shell():
    shells = _synthetic_scenes()["coincident"]
    first = P.render_portrait(shells[:1], size=(80, 100))
    both = P.render_portrait(shells, size=(80, 100))
    _assert_same_image(both, first, "second coincident shell must never win a tie")
    # ... and in the other order the other material shows through
    swapped = P.render_portrait(shells[::-1], size=(80, 100))
    assert (swapped != first).any()


def test_transparent_background_keeps_alpha_zero_outside():
    shells = [_sphere(2)]
    new = P.render_portrait(shells, size=(60, 80), background=None)
    ref = _reference_render_portrait(shells, size=(60, 80), background=None)
    _assert_same_image(new, ref)
    assert new[0, 0, 3] == 0 and new[40, 30, 3] == 255


def test_empty_and_degenerate_input():
    empty = Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    img = P.render_portrait([P.Shell(empty)], size=(16, 12))
    assert img.shape == (12, 16, 4)
    # a collapsed (zero-area) triangle draws nothing and breaks nothing
    flat = Mesh(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0.0]]), [[0, 1, 2]])
    ref = _reference_render_portrait([P.Shell(flat)], size=(20, 20))
    _assert_same_image(P.render_portrait([P.Shell(flat)], size=(20, 20)), ref)


# -- the rasteriser against the loop, on adversarial soups ---------------------


def _soup(rng, n, size, spread, snap=None, depth_levels=None):
    w, h = size
    nv = 3 * n
    sx = rng.uniform(-spread, w + spread, nv)
    sy = rng.uniform(-spread, h + spread, nv)
    if snap:
        sx, sy = np.round(sx / snap) * snap, np.round(sy / snap) * snap
    if depth_levels:
        depth = rng.choice(depth_levels, nv).astype(np.float64)
    else:
        depth = rng.uniform(0.5, 30.0, nv)
    return sx, sy, depth, np.arange(nv).reshape(n, 3)


@pytest.mark.parametrize("case", [
    dict(n=400, spread=5, snap=None),          # plain small triangles
    dict(n=300, spread=40, snap=None),         # some off-screen, some big
    dict(n=300, spread=3, snap=0.5),           # vertices and edges on pixel centres
    dict(n=300, spread=3, snap=0.5, depth_levels=[2.0, 4.0]),   # exact depth ties
    dict(n=60, spread=300, snap=None),         # huge: past the largest class
    dict(n=500, spread=2, snap=1.0, depth_levels=[3.0]),        # all coplanar, all ties
])
@pytest.mark.parametrize("size", [(37, 29), (101, 64)])
def test_rasterizer_matches_the_loop_on_triangle_soups(case, size):
    rng = np.random.default_rng(1234 + case["n"] + size[0])
    sx, sy, depth, faces = _soup(rng, size=size, **case)
    ref_ids, ref_z = _reference_ids(sx, sy, depth, faces, size)
    ids, z = R.rasterize(sx, sy, depth, faces, size)
    assert np.array_equal(ids, ref_ids)
    assert np.array_equal(z, ref_z)


def test_fragments_at_absurd_depth_are_dropped_like_the_loop():
    """The loop never writes a fragment whose inverse depth is <= 1e-12 (depth >= 1e12)."""
    size = (30, 30)
    sx = np.array([2.0, 25.0, 12.0, 2.0, 25.0, 12.0])
    sy = np.array([2.0, 3.0, 26.0, 2.0, 3.0, 26.0])
    depth = np.array([1e13, 1e13, 1e13, 5.0, 5.0, 5.0])      # the first triangle is far beyond
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    ids, z = R.rasterize(sx, sy, depth, faces, size)
    ref_ids, ref_z = _reference_ids(sx, sy, depth, faces, size)
    assert np.array_equal(ids, ref_ids) and np.array_equal(z, ref_z)
    only_far = R.rasterize(sx, sy, depth, faces, size, face_ids=np.array([0]))[0]
    assert (only_far == -1).all()


def test_result_does_not_depend_on_chunking():
    rng = np.random.default_rng(7)
    size = (90, 70)
    sx, sy, depth, faces = _soup(rng, 700, size, 15, snap=0.5, depth_levels=[2.0, 3.0, 5.0])
    whole, zw = R.rasterize(sx, sy, depth, faces, size)
    for cap in (1, 50, 2000):
        ids, z = R.rasterize(sx, sy, depth, faces, size, max_candidates=cap)
        assert np.array_equal(ids, whole) and np.array_equal(z, zw), cap


def test_triangles_larger_than_every_class_take_the_per_triangle_path():
    size = (400, 300)
    # one triangle covering most of the frame, one small one in front of it
    sx = np.array([-20.0, 430.0, 200.0, 150.0, 190.0, 170.0])
    sy = np.array([-10.0, -5.0, 330.0, 100.0, 100.0, 140.0])
    depth = np.array([5.0, 5.0, 5.0, 2.0, 2.0, 2.0])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    assert max(sx.max() - sx.min(), sy.max() - sy.min()) > R.CLASSES[-1]
    ids, z = R.rasterize(sx, sy, depth, faces, size)
    ref_ids, ref_z = _reference_ids(sx, sy, depth, faces, size)
    assert np.array_equal(ids, ref_ids) and np.array_equal(z, ref_z)
    assert (ids == 0).sum() > 50_000 and (ids == 1).sum() > 100     # both really drawn


def test_dropping_the_large_triangle_path_would_be_caught(monkeypatch):
    """Guard for the guard: without the per-triangle path the test above fails."""
    size = (400, 300)
    sx = np.array([-20.0, 430.0, 200.0])
    sy = np.array([-10.0, -5.0, 330.0])
    depth = np.array([5.0, 5.0, 5.0])
    faces = np.array([[0, 1, 2]])
    monkeypatch.setattr(R, "_per_triangle", lambda *a, **k: None)
    ids, _ = R.rasterize(sx, sy, depth, faces, size)
    assert (ids >= 0).sum() == 0          # the path is what draws these


def test_float32_draft_mode_is_close_but_not_required_to_be_exact():
    rng = np.random.default_rng(3)
    size = (120, 90)
    sx, sy, depth, faces = _soup(rng, 400, size, 8)
    a, _ = R.rasterize(sx, sy, depth, faces, size)
    b, _ = R.rasterize(sx, sy, depth, faces, size, dtype=np.float32)
    assert (a != b).mean() < 0.02


# -- golden set on real scenes ------------------------------------------------


@pytest.fixture(scope="module")
def world():
    if not mh.available():
        pytest.skip("CC0 MakeHuman assets not fetched")
    return mh.load()


def _rng_colors(n, seed):
    c = np.random.default_rng(seed).uniform(0.05, 0.9, (n, 3))
    return sh.srgb_to_linear(c)


_WING_MAT = sh.Material(base_color="#6f4f3f", roughness=0.74, specular=0.28,
                        f0=0.035, wrap=0.40, scatter="#8c4a38", rim=0.24)


def _scene(world, style, controls, body, wing_kw=None):
    """Body (levels=0, optional) plus the wing pair of ``style`` posed by ``controls``."""
    base, lib = world
    shells = list(P.character_shells(base, None, levels=0, with_eyes=True)) if body else []
    app = A.back_wings(style)
    app.params.update(wing_kw or {})
    for _, pw in app.build_rigs(base, None):
        mesh = pw.pose(**controls).transform(P.MH_TO_Z_UP)
        shells.append(P.Shell(mesh, _WING_MAT, base_colors=_rng_colors(len(mesh.vertices), 5)))
    return shells


# the angel wing at draft detail (15k triangles each, not 85k) keeps the slow
# reference loop affordable; MM_GOLDEN_FULL=1 runs the full-detail wing
_DRAFT = dict(coverts=False, underwing=False, bones=False,
              primaries=5, secondaries=7, tertials=4, scapulars=4)
GOLDEN_SCENES = {
    "bat spread": ("membrane", {}, True, None),
    "bat fold": ("membrane", {"fold": 0.7}, False, None),
    "angel fold": ("feathered", {"fold": 1.0}, False, _DRAFT),
    "angel raise": ("feathered", {"raise_": 35.0}, False, _DRAFT),
}
GOLDEN_VIEWS = [   # 4 azimuths, 2 zoomed fits, 3 sizes
    ("az0", 0.0, 0.0, None, (200, 300)),
    ("az90", 90.0, 10.0, None, (240, 180)),
    ("az200", 200.0, 25.0, None, (160, 160)),
    ("az315", 315.0, -15.0, None, (200, 300)),
    ("zoom a", 20.0, 5.0, "zoom", (240, 180)),
    ("zoom b", 160.0, 5.0, "zoom", (160, 160)),
]
_golden_cache = {}


def _golden_shells(world, name):
    if name not in _golden_cache:
        style, controls, body, kw = GOLDEN_SCENES[name]
        _golden_cache[name] = _scene(world, style, controls, body, kw)
    return _golden_cache[name]


def _zoom_fit(shells):
    """Frame a patch of a wing, a tenth of the figure's height across: the
    triangles become tens of pixels wide, past most of the size classes."""
    lo = np.min([s.mesh.vertices.min(axis=0) for s in shells], axis=0)
    hi = np.max([s.mesh.vertices.max(axis=0) for s in shells], axis=0)
    v = shells[-1].mesh.vertices
    centre = v[np.argmin(np.linalg.norm(v - np.median(v, axis=0), axis=1))]   # a point ON the wing
    return centre, 0.10 * float(hi[2] - lo[2])


@needs_assets
@pytest.mark.parametrize("view", GOLDEN_VIEWS, ids=[v[0] for v in GOLDEN_VIEWS])
@pytest.mark.parametrize("name", sorted(GOLDEN_SCENES))
def test_golden_scenes_are_pixel_identical_to_the_old_loop(world, name, view):
    _, az, el, fit, size = view
    shells = _golden_shells(world, name)
    fit = _zoom_fit(shells) if fit == "zoom" else None
    ref = _reference_render_portrait(shells, size=size, azimuth=az, elevation=el, fit=fit)
    new = P.render_portrait(shells, size=size, azimuth=az, elevation=el, fit=fit)
    _assert_same_image(new, ref, f"{name} {view[0]}")
    assert (new[:, :, 3] == ref[:, :, 3]).all()
    assert len(np.unique(new.reshape(-1, 4), axis=0)) > 30        # a real picture,
    assert (new[:, :, :3] != new[0, 0, :3]).any(axis=2).mean() > 0.02   # not an empty frame


@needs_assets
@pytest.mark.parametrize("radius", [0.012, 0.002])
def test_extreme_zoom_on_the_body_matches_the_old_loop(world, monkeypatch, radius):
    """Zoom until single quads are hundreds of pixels wide: the largest size
    classes and, at the extreme, the per-triangle path (which an early spike of
    this rasteriser silently skipped) must reproduce the loop."""
    base, _ = world
    shells = P.character_shells(base, None, levels=0, with_eyes=False)
    lo, hi = shells[0].mesh.bounds
    h = hi[2] - lo[2]
    fit = (np.array([(lo[0] + hi[0]) / 2, lo[1], lo[2] + 0.72 * h]), radius * h)
    used = []
    real = R._per_triangle
    monkeypatch.setattr(R, "_per_triangle", lambda *a, **k: used.append(1) or real(*a, **k))
    ref = _reference_render_portrait(shells, size=(320, 480), azimuth=0, elevation=0, fit=fit)
    new = P.render_portrait(shells, size=(320, 480), azimuth=0, elevation=0, fit=fit)
    _assert_same_image(new, ref, f"zoom {radius}")
    assert (len(used) > 0) == (radius <= 0.002)   # 4 quads wider than 128 px at 0.002


FULL_SIZES = [(320, 480), (400, 300), (500, 500)]
_full_cache = {}


@needs_assets
@pytest.mark.skipif(not os.environ.get("MM_GOLDEN_FULL"),
                    reason="minutes of reference loop; set MM_GOLDEN_FULL=1")
@pytest.mark.parametrize("view", GOLDEN_VIEWS, ids=[v[0] for v in GOLDEN_VIEWS])
@pytest.mark.parametrize("name", sorted(GOLDEN_SCENES))
def test_golden_full_scenes(world, name, view):
    """The same 24 renders on the real scenes: body + two full-detail wings
    (43k / 171k triangles), at 320x480, 400x300 and 500x500."""
    style, controls, _, _ = GOLDEN_SCENES[name]
    if name not in _full_cache:
        _full_cache[name] = _scene(world, style, controls, True, None)
    shells = _full_cache[name]
    _, az, el, fit, _ = view
    size = FULL_SIZES[GOLDEN_VIEWS.index(view) % 3]
    fit = _zoom_fit(shells) if fit == "zoom" else None
    ref = _reference_render_portrait(shells, size=size, azimuth=az, elevation=el, fit=fit)
    new = P.render_portrait(shells, size=size, azimuth=az, elevation=el, fit=fit)
    _assert_same_image(new, ref, f"{name} {view[0]} {size}")
    assert (new[:, :, 3] == ref[:, :, 3]).all()


# -- speed --------------------------------------------------------------------


def test_the_new_renderer_is_much_faster_than_the_loop():
    shells = [_sphere(4), _sphere(3, (0.5, 0.3, 0.2), 0.7)]          # ~5k + 1.3k triangles
    t0 = time.perf_counter()
    ref = _reference_render_portrait(shells, size=(160, 200))
    t_ref = time.perf_counter() - t0
    best = min(_timed(lambda: P.render_portrait(shells, size=(160, 200), cache=False))
               for _ in range(3))
    _assert_same_image(P.render_portrait(shells, size=(160, 200)), ref)
    assert t_ref / best > 4.0, f"loop {t_ref:.2f}s vs new {best:.3f}s"


def _timed(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


# ==========================================================================
# normals / AO: bincount instead of np.add.at, identical numbers


def _meshes_for_numerics():
    rng = np.random.default_rng(11)
    cube = quads_to_mesh(*cube_cage(2.0))
    # a lumpy random mesh with a vertex used by many faces and an unused vertex
    verts = rng.normal(size=(40, 3))
    faces = rng.integers(0, 39, size=(120, 3))
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
                  & (faces[:, 0] != faces[:, 2])]
    out = {"sphere": primitives.icosphere(1.0, 3), "cube": cube,
           "lumpy": Mesh(verts, faces),
           "empty": Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))}
    return out


@pytest.mark.parametrize("name", sorted(_meshes_for_numerics()))
def test_normals_and_ao_are_bit_identical_to_the_add_at_versions(name):
    mesh = _meshes_for_numerics()[name]
    vn_ref = _ref_vertex_normals(mesh)
    vn = sh.vertex_normals(mesh.vertices, mesh.faces)
    assert np.array_equal(vn, vn_ref)
    for strength, rs, smooth in ((1.0, 1.0, 2), (0.35, 1.7, 0), (1.4, 0.5, 4)):
        ref = _ref_cavity_ao(mesh.vertices, mesh.faces, vn_ref, strength, rs, smooth)
        new = sh.cavity_ao(mesh.vertices, mesh.faces, vn, strength, rs, smooth)
        assert np.array_equal(new, ref), (strength, rs, smooth)
    noise = np.random.default_rng(2).normal(size=len(mesh.vertices))
    assert np.array_equal(sh.smooth_vertex_scalar(noise, mesh.faces, 3),
                          _ref_smooth(noise, mesh.faces, 3))


@needs_assets
def test_normals_and_ao_are_bit_identical_on_the_body(world):
    base, _ = world
    body = P.character_shells(base, None, levels=0, with_eyes=False)[0].mesh
    vn_ref = _ref_vertex_normals(body)
    assert np.array_equal(sh.vertex_normals(body.vertices, body.faces), vn_ref)
    assert np.array_equal(sh.cavity_ao(body.vertices, body.faces, vn_ref),
                          _ref_cavity_ao(body.vertices, body.faces, vn_ref))


class _NoAddAt:
    """``numpy`` as seen by a module, except that ``np.add.at`` raises. (Patching
    ``np.add.at`` itself is impossible on old numpy, where ufunc attributes are
    read-only.)"""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        attr = getattr(self._real, name)
        if name != "add":
            return attr

        class Add:
            __call__ = staticmethod(attr)

            def at(self, *a, **k):
                raise AssertionError("np.add.at is slow on numpy < 1.25; use bincount")

            def __getattr__(self, n):
                return getattr(attr, n)
        return Add()


def test_shading_and_rendering_never_call_np_add_at(monkeypatch):
    """np.add.at is ~4x slower before numpy 1.25, our floor is 1.24."""
    for mod in (sh, P):
        monkeypatch.setattr(mod, "np", _NoAddAt(np))
    mesh = primitives.icosphere(1.0, 2)
    vn = sh.vertex_normals(mesh.vertices, mesh.faces)
    sh.cavity_ao(mesh.vertices, mesh.faces, vn)
    sh.smooth_vertex_scalar(np.zeros(len(mesh.vertices)), mesh.faces)
    P.render_portrait([P.Shell(mesh)], size=(40, 40), cache=False)
    with pytest.raises(AssertionError):            # the guard itself works
        sh.np.add.at(np.zeros(3), [0], 1.0)


def test_portrait_vertex_normals_is_the_same_function():
    mesh = primitives.icosphere(1.0, 2)
    assert np.array_equal(P._vertex_normals(mesh), _ref_vertex_normals(mesh))


# ==========================================================================
# the shading cache


def test_cache_reuses_terms_for_an_unchanged_shell():
    cache = sh.ShadingCache()
    body, wing_a = _sphere(3), _sphere(2, (2, 0, 0))
    wing_b = _sphere(2, (2, 0.4, 0))                  # the "edited" appendage
    P.render_portrait([body, wing_a], size=(40, 50), cache=cache)
    assert (cache.normal_builds, cache.cavity_builds) == (2, 2)
    P.render_portrait([body, wing_b], size=(40, 50), cache=cache)
    # the body hit; only the edited shell was computed again
    assert (cache.normal_builds, cache.cavity_builds) == (3, 3)
    assert cache.hits >= 1
    # a different camera on the same shells recomputes nothing at all
    P.render_portrait([body, wing_b], size=(40, 50), azimuth=140, cache=cache)
    assert (cache.normal_builds, cache.cavity_builds) == (3, 3)


def test_cache_output_is_identical_with_and_without():
    shells = [_sphere(3), _sphere(2, (1.5, 0, 0), 0.5, sh.EYE, ao_strength=0.35)]
    cache = sh.ShadingCache()
    a = P.render_portrait(shells, size=(70, 90), cache=False)
    b = P.render_portrait(shells, size=(70, 90), cache=cache)
    c = P.render_portrait(shells, size=(70, 90), cache=cache)          # served from cache
    _assert_same_image(b, a)
    _assert_same_image(c, a)
    assert cache.hits >= 2


def test_cache_is_keyed_by_content_not_identity():
    """Editing a mesh in place must not be served the old terms."""
    cache = sh.ShadingCache()
    shell = _sphere(2)
    before = P.render_portrait([shell], size=(50, 50), cache=cache)
    shell.mesh.vertices[:, 2] *= 0.5                       # squash it, same object
    after = P.render_portrait([shell], size=(50, 50), cache=cache)
    ref = _reference_render_portrait([shell], size=(50, 50))
    assert (before != after).any()
    _assert_same_image(after, ref)
    assert cache.normal_builds == 2


def test_cache_strength_is_applied_after_the_cached_term():
    cache = sh.ShadingCache()
    mesh = primitives.icosphere(1.0, 2)
    vn, term = cache.terms(mesh.vertices, mesh.faces)
    for strength in (0.0, 0.35, 1.0, 1.4):
        assert np.array_equal(sh.ao_from_term(term, strength),
                              _ref_cavity_ao(mesh.vertices, mesh.faces, vn, strength))
    cache.terms(mesh.vertices, mesh.faces)
    assert cache.cavity_builds == 1


def test_cache_is_bounded():
    cache = sh.ShadingCache(max_entries=3)
    for i in range(8):
        m = _sphere(1, (i, 0, 0)).mesh
        cache.terms(m.vertices, m.faces)
    assert len(cache) == 3
    tiny = sh.ShadingCache(max_bytes=1)            # smaller than any entry: keeps only the latest
    for i in range(4):
        m = _sphere(1, (i, 0, 0)).mesh
        tiny.terms(m.vertices, m.faces)
    assert len(tiny) == 1
    cache.clear()
    assert len(cache) == 0 and cache.nbytes == 0


def test_cache_can_be_shared_between_threads():
    shells = [_sphere(3), _sphere(2, (1.5, 0, 0), 0.5)]
    cache = sh.ShadingCache()
    ref = P.render_portrait(shells, size=(60, 70), cache=False)
    results, errors = [], []

    def work():
        try:
            results.append(P.render_portrait(shells, size=(60, 70), cache=cache))
        except Exception as exc:        # pragma: no cover - failure path
            errors.append(exc)
    threads = [threading.Thread(target=work) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    for img in results:
        _assert_same_image(img, ref)


# ==========================================================================
# cancellation


def _scene_for_cancel():
    return [_sphere(3), _sphere(2, (1.5, 0, 0), 0.5), _sphere(2, (-1.5, 0, 0), 0.5)]


def test_check_is_called_between_shells_and_between_raster_chunks():
    shells = _scene_for_cancel()
    calls = []
    P.render_portrait(shells, size=(60, 80), check=lambda: calls.append(1), cache=False)
    n_shells = len(shells)
    assert len(calls) >= n_shells + 1          # per shell, and at least once in the raster
    # a tiny chunk cap would multiply the raster checkpoints; here we just show
    # the count is a few dozen, never per pixel or per triangle
    assert len(calls) < 200

    calls2 = []
    sx, sy, depth, faces = _soup(np.random.default_rng(0), 300, (50, 50), 5)
    R.rasterize(sx, sy, depth, faces, (50, 50), check=lambda: calls2.append(1), max_candidates=64)
    assert len(calls2) > 10                    # chunked work checks between chunks


def test_check_can_abort_at_any_checkpoint():
    shells = _scene_for_cancel()
    total = []
    P.render_portrait(shells, size=(60, 80), check=lambda: total.append(1), cache=False)
    for stop_at in (1, 2, len(total) // 2, len(total)):
        count = [0]

        def check():
            count[0] += 1
            if count[0] >= stop_at:
                raise P.Cancelled()
        with pytest.raises(P.Cancelled):
            P.render_portrait(shells, size=(60, 80), check=check, cache=False)
        assert count[0] == stop_at


def test_cancelled_render_leaves_the_cache_consistent():
    shells = _scene_for_cancel()
    cache = sh.ShadingCache()
    count = [0]

    def check():
        count[0] += 1
        if count[0] == 3:
            raise P.Cancelled()
    with pytest.raises(P.Cancelled):
        P.render_portrait(shells, size=(60, 80), check=check, cache=cache)
    ref = _reference_render_portrait(shells, size=(60, 80))
    _assert_same_image(P.render_portrait(shells, size=(60, 80), cache=cache), ref)


def test_cancel_token():
    token = P.CancelToken()
    token()                                    # not cancelled: a no-op
    assert not token.cancelled
    img = P.render_portrait(_scene_for_cancel(), size=(40, 40), check=token)
    assert img.shape == (40, 40, 4)
    token.cancel()
    assert token.cancelled
    with pytest.raises(P.Cancelled):
        P.render_portrait(_scene_for_cancel(), size=(40, 40), check=token)


def test_cancel_from_another_thread_stops_a_render_in_progress():
    shells = [_sphere(6)]                          # ~20k triangles
    token = P.CancelToken()
    reached, resume = threading.Event(), threading.Event()
    calls, out = [0], {}

    def check():
        calls[0] += 1
        if calls[0] == 3:                          # mid-render: let the UI thread cancel
            reached.set()
            resume.wait(30)
        token()

    def work():
        try:
            P.render_portrait(shells, size=(300, 400), check=check, cache=False)
            out["done"] = True
        except P.Cancelled:
            out["cancelled"] = calls[0]
    t = threading.Thread(target=work)
    t.start()
    assert reached.wait(30)
    token.cancel()
    resume.set()
    t.join(30)
    assert out == {"cancelled": 3}                 # stopped at the very next checkpoint


# ==========================================================================
# the id buffer and picking


def test_pick_returns_the_shell_and_triangle_under_a_pixel():
    a, b = _sphere(3), _sphere(3, (3.0, 0, 0), 0.8)
    empty = P.Shell(Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)))
    shells = [empty, a, b]                         # an empty shell keeps its slot
    size = (200, 160)
    vb = P.visibility_buffer(shells, size, 0.0, 0.0)
    assert vb.pick(-1, 5) is None and vb.pick(5, 5) is None and vb.pick(500, 5) is None
    inside = [(x, y) for x in range(0, 200, 5) for y in range(0, 160, 5) if vb.pick(x, y)]
    assert inside
    x0, y0 = inside[len(inside) // 2]
    seen = set()
    for x in range(0, 200, 10):
        for y in range(0, 160, 10):
            hit = vb.pick(x, y)
            if hit:
                assert hit[0] in (1, 2) and 0 <= hit[1] < len(shells[hit[0]].mesh.faces)
                seen.add(hit[0])
    assert seen == {1, 2}
    # the module-level helper builds the same buffer
    assert P.pick(shells, size, 0.0, 0.0, None, x0, y0) == vb.pick(x0, y0)
    assert vb.pick(x0 + 0.7, y0 + 0.2) == vb.pick(x0, y0)           # fractional -> its pixel
    shell, face = vb.pick_many([x0, -3, 1e6], [y0, 4, 5])
    assert list(shell) == [vb.pick(x0, y0)[0], -1, -1] and face[0] == vb.pick(x0, y0)[1]


def test_id_buffer_agrees_with_the_rendered_image():
    """Where the id buffer says 'nothing', the image is background, and vice versa."""
    shells = _synthetic_scenes()["overlap"]
    size = (90, 110)
    img = P.render_portrait(shells, size=size, azimuth=60, elevation=10, background=None)
    vb = P.visibility_buffer(shells, size, azimuth=60, elevation=10)
    assert np.array_equal(vb.shell >= 0, img[:, :, 3] == 255)
    assert np.isfinite(vb.depth[vb.shell >= 0]).all() and np.isinf(vb.depth[vb.shell < 0]).all()


@needs_assets
def test_picking_agrees_with_ray_mesh_hit_to_within_one_face(world):
    base, _ = world
    shells = P.character_shells(base, None, levels=0, with_eyes=False)
    body = shells[0].mesh
    size, az, el = (200, 300), 35.0, 8.0
    vb = P.visibility_buffer(shells, size, az, el)
    camera = vb.camera
    inner = vb.shell >= 0
    # keep to pixels well inside the silhouette: the id buffer culls back faces
    # and ray_mesh_hit does not, so grazing hits can legitimately differ
    for _ in range(3):
        inner = inner & np.roll(inner, 3, 0) & np.roll(inner, -3, 0) \
            & np.roll(inner, 3, 1) & np.roll(inner, -3, 1)
    ys, xs = np.nonzero(inner)
    take = np.linspace(0, len(xs) - 1, 150).astype(int)
    exact = near = 0
    for x, y in zip(xs[take], ys[take]):
        # the rasteriser samples pixel centres; screen_ray takes plain coordinates
        origin, direction = rc.screen_ray(x + 0.5, y + 0.5, size[0], size[1],
                                          camera.view, camera.focal)
        hit = rc.ray_mesh_hit(origin, direction, body)
        shell, face = vb.pick(x, y)
        assert shell == 0
        if hit.face == face:
            exact += 1
            continue
        shared = set(body.faces[hit.face]) & set(body.faces[face])
        assert shared, f"pixel ({x},{y}): ray hit face {hit.face}, id buffer {face}"
        near += 1
    assert exact >= 0.9 * len(take), (exact, near)
    # and the depth buffer unprojects to the ray's hit point
    x, y = int(xs[take[0]]), int(ys[take[0]])
    origin, direction = rc.screen_ray(x + 0.5, y + 0.5, size[0], size[1],
                                      camera.view, camera.focal)
    hit = rc.ray_mesh_hit(origin, direction, body)
    assert np.linalg.norm(vb.world_point(x, y) - hit.point) < 0.05
    assert vb.world_point(0, 0) is None


@needs_assets
def test_pick_maps_to_the_right_shell_among_body_eyes_and_wings(world):
    base, _ = world
    shells = _scene(world, "membrane", {}, True)
    vb = P.visibility_buffer(shells, (200, 300), 180.0, 10.0)        # from behind: the wings
    hits = {vb.pick(x, y)[0] for x in range(0, 200, 4) for y in range(0, 300, 4) if vb.pick(x, y)}
    assert len(shells) - 1 in hits and len(shells) - 2 in hits       # both wings
    assert 0 in hits                                                 # the body too
    for x in range(0, 200, 7):
        for y in range(0, 300, 7):
            hit = vb.pick(x, y)
            if hit:
                assert 0 <= hit[1] < len(shells[hit[0]].mesh.faces)
