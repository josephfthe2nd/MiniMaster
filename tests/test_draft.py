"""Draft tier building blocks: cluster proxies, the body proxy, the draft shader."""

import time

import numpy as np
import pytest

from minimaster.character import appendages as A
from minimaster.character import draft as D
from minimaster.character import mhbase as mh
from minimaster.character import portrait as P
from minimaster.character import shading as sh
from minimaster.core import primitives
from minimaster.core.mesh import Mesh

needs_assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")


# -- ClusterProxy on synthetic meshes -------------------------------------------


def _grid_sheet(n=20, size=10.0):
    """A flat n x n vertex sheet of 2(n-1)^2 triangles in the XY plane."""
    xs = np.linspace(0, size, n)
    v = np.array([[x, y, 0.0] for y in xs for x in xs])
    f = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            f += [[a, a + 1, a + n + 1], [a, a + n + 1, a + n]]
    return v, np.array(f)


def test_by_cell_clusters_vertices_and_keeps_topology_valid():
    v, f = _grid_sheet(20)
    proxy = D.ClusterProxy.by_cell(v, f, cell=2.5)
    assert 0 < len(proxy.faces) < len(f) / 4
    assert proxy.faces.min() >= 0 and proxy.faces.max() == proxy.n_vertices - 1
    # no collapsed triangle survives
    fa = proxy.faces
    assert ((fa[:, 0] != fa[:, 1]) & (fa[:, 1] != fa[:, 2]) & (fa[:, 0] != fa[:, 2])).all()
    # the flat sheet stays flat and inside its own bounds
    assert np.allclose(proxy.rest[:, 2], 0.0)
    assert proxy.rest.min() >= -1e-9 and proxy.rest.max() <= 10.0 + 1e-9


def test_update_is_the_cluster_mean_and_is_linear():
    v, f = _grid_sheet(16)
    proxy = D.ClusterProxy.by_cell(v, f, cell=3.0)
    rng = np.random.default_rng(0)
    deformed = v + rng.normal(size=v.shape) * 0.3
    got = proxy.update(deformed)
    # recompute the mean per cluster the slow way
    want = np.zeros_like(got)
    for k in range(proxy.n_vertices):
        members = deformed[proxy._src[proxy._lab == k]]
        want[k] = members.mean(axis=0)
    assert np.allclose(got, want, atol=1e-12)
    assert np.allclose(proxy.update(v), proxy.rest)
    # linear: translating the source translates the proxy
    assert np.allclose(proxy.update(v + [1, 2, 3]), proxy.rest + [1, 2, 3])
    assert isinstance(proxy.mesh(deformed), Mesh)
    assert np.array_equal(proxy.mesh().faces, proxy.faces)


def test_rep_is_a_member_nearest_each_proxy_vertex():
    v, f = _grid_sheet(16)
    proxy = D.ClusterProxy.by_cell(v, f, cell=3.0)
    assert len(proxy.rep) == proxy.n_vertices
    for k in range(proxy.n_vertices):
        members = proxy._src[proxy._lab == k]
        assert proxy.rep[k] in members
        d = np.linalg.norm(v[members] - proxy.rest[k], axis=1)
        assert np.isclose(np.linalg.norm(v[proxy.rep[k]] - proxy.rest[k]), d.min())


def test_a_cell_larger_than_the_mesh_leaves_an_empty_proxy_that_still_works():
    v, f = _grid_sheet(8)
    proxy = D.ClusterProxy.by_cell(v, f, cell=100.0)
    assert len(proxy.faces) == 0 and proxy.n_vertices == 0 and len(proxy.rep) == 0
    assert proxy.update(v).shape == (0, 3)
    assert len(proxy.mesh().faces) == 0
    assert D.render_draft([P.Shell(proxy.mesh())], size=(20, 20)).shape == (20, 20, 4)


def test_by_target_lands_near_the_requested_triangle_count():
    mesh = primitives.icosphere(1.0, 3)                      # 1280 triangles
    for target in (100, 300, 600):
        proxy = D.ClusterProxy.by_target(mesh.vertices, mesh.faces, target)
        assert 0.6 * target <= len(proxy.faces) <= 1.1 * target, (target, len(proxy.faces))


def test_clusters_never_mix_pieces_and_labelling_is_one_unique(monkeypatch):
    """Two overlapping spheres as two pieces: one np.unique over (piece, cell)
    keys, and no cluster holds vertices of both."""
    a = primitives.icosphere(1.0, 2)
    b = primitives.icosphere(1.0, 2)                         # exactly on top of a
    v = np.vstack([a.vertices, b.vertices])
    f = np.vstack([a.faces, b.faces + len(a.vertices)])
    piece = np.repeat([0, 1], len(a.vertices))
    calls = []
    real = D._cell_labels
    monkeypatch.setattr(D, "_cell_labels", lambda *x, **k: calls.append(1) or real(*x, **k))
    proxy = D.ClusterProxy.by_pieces(v, f, piece)
    assert len(calls) == 1
    for k in range(proxy.n_vertices):
        assert len(set(piece[proxy._src[proxy._lab == k]])) == 1
    # no proxy triangle straddles the two pieces
    owner = piece[proxy.rep][proxy.faces]
    assert (owner[:, 0] == owner[:, 1]).all() and (owner[:, 1] == owner[:, 2]).all()
    assert {0, 1} == set(owner[:, 0])


def test_by_pieces_decimates_big_pieces_more_than_small_ones():
    big = _grid_sheet(40)
    small = _grid_sheet(6, 1.0)
    v = np.vstack([big[0], small[0] + [50, 0, 0]])
    f = np.vstack([big[1], small[1] + len(big[0])])
    piece = np.repeat([0, 1], [len(big[0]), len(small[0])])
    proxy = D.ClusterProxy.by_pieces(v, f, piece)
    owner = piece[proxy.rep][proxy.faces][:, 0]
    n_big, n_small = (owner == 0).sum(), (owner == 1).sum()
    assert n_big < (len(big[1])) / 3          # the 3042-triangle sheet shrank a lot
    assert n_small >= 8                        # the tiny piece keeps a useful shape
    # a piece id of -1 / non-contiguous ids are fine
    proxy2 = D.ClusterProxy.by_pieces(v, f, piece * 7 + 3)
    assert len(proxy2.faces) == len(proxy.faces)


def test_pieces_from_ranges():
    p = D.pieces_from_ranges([(0, 3), (3, 5), (6, 8)], 9)
    assert list(p) == [0, 0, 0, 1, 1, -1, 2, 2, -1]


def test_proxy_of_a_closed_thin_slab_keeps_both_sides():
    """A thin slab has two sides; the proxy must not drop either (back-face
    culling then picks the visible one)."""
    v, f = _grid_sheet(12)
    top = v.copy()
    top[:, 2] = 0.05
    verts = np.vstack([v, top])
    faces = np.vstack([f, f[:, [0, 2, 1]] + len(v)])
    proxy = D.ClusterProxy.by_cell(verts, faces, cell=3.0)
    m = proxy.mesh()
    n = m.face_normals()
    assert (n[:, 2] > 0.5).any() and (n[:, 2] < -0.5).any()


# -- the body proxy -------------------------------------------------------------


@pytest.fixture(scope="module")
def world():
    if not mh.available():
        pytest.skip("CC0 MakeHuman assets not fetched")
    base, lib = mh.load()
    return base, lib, D.BodyProxy(base, 4000)


def _silhouette(mesh, fit, az, el, size=(320, 480)):
    return P.visibility_buffer([P.Shell(mesh)], size, az, el, fit=fit).shell >= 0


def _iou(full, proxy, views, fit=None):
    lo, hi = full.bounds
    fit = fit or ((lo + hi) / 2.0, float(np.linalg.norm(hi - lo)) / 2.0)
    out = []
    for az, el in views:
        a, b = _silhouette(full, fit, az, el), _silhouette(proxy, fit, az, el)
        out.append((a & b).sum() / (a | b).sum())
    return out


VIEWS = [(0, 0), (90, 0), (180, 10), (35, 20), (270, 5)]


@needs_assets
def test_body_proxy_size_and_silhouette(world):
    base, lib, proxy = world
    assert 3500 <= len(proxy.faces) <= 4500            # ~4k of 26,756 triangles
    for verts in (None, lib.apply(base.verts, mh.macro_weights(gender=1.0, height=0.9, weight=0.8)),
                  lib.apply(base.verts, mh.macro_weights(gender=0.0, age=0.18))):   # man, tall, heavy; child
        full = P.character_shells(base, verts, levels=0, with_eyes=False)[0].mesh
        ious = _iou(full, proxy.mesh(verts), VIEWS)
        assert min(ious) >= 0.95, np.round(ious, 3)


@needs_assets
def test_body_proxy_update_is_a_bincount_average_in_well_under_a_millisecond(world):
    base, lib, proxy = world
    verts = lib.apply(base.verts, mh.macro_weights(gender=1.0, muscle=0.9))
    best = min(_timed(lambda: proxy.points(verts)) for _ in range(40))
    assert best < 1e-3, f"{best * 1e3:.2f} ms"
    # exactly the cluster mean of the morphed cage vertices, then Z-up
    cage, _ = base.body_cage(verts)
    mean = proxy.update(cage)
    assert np.allclose(proxy.points(verts, to_z_up=False), mean)
    assert np.allclose(proxy.points(verts), mean @ P.MH_TO_Z_UP[:3, :3].T)
    # the proxy follows the morph: a taller body gives a taller proxy
    tall = lib.apply(base.verts, mh.macro_weights(height=1.0))
    short = lib.apply(base.verts, mh.macro_weights(height=0.0))
    ht = lambda m: np.ptp(proxy.points(m)[:, 2])
    assert ht(tall) > ht(short) * 1.05


def _timed(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


@needs_assets
def test_body_proxy_is_built_once_from_the_base_mesh_only(world):
    base, lib, proxy = world
    again = D.BodyProxy(base, 4000)
    assert np.array_equal(again.faces, proxy.faces) and np.allclose(again.rest, proxy.rest)
    coarse = D.BodyProxy(base, 2000)
    assert len(coarse.faces) < len(proxy.faces)


# -- appendage proxies (the bat wing rig) ------------------------------------------


def _bat(world):
    base, lib, _ = world
    app = A.back_wings("membrane")
    return app.build_rigs(base, None)


@needs_assets
def test_wing_cluster_proxy_is_small_follows_the_pose_and_keeps_its_shape(world):
    pieces_wing = _bat(world)[0][1]
    rig = pieces_wing.rig
    n = len(rig.rest_mesh.vertices)
    piece = D.pieces_from_ranges([rig.ranges[p.name] for p in rig.pieces], n)
    assert (piece >= 0).all()
    proxy = D.ClusterProxy.by_pieces(rig.rest_mesh.vertices, rig.rest_mesh.faces, piece)
    assert len(proxy.faces) * 4 < len(rig.rest_mesh.faces)
    # silhouette overlap with the full posed wing (measured 0.80-0.83 spread, 0.59-0.70
    # at fold 0.8: the draft has no membrane pleats); a draft, not a likeness
    for controls, floor in (({}, 0.75), ({"fold": 0.8}, 0.5)):
        posed = pieces_wing.pose(**controls).transform(P.MH_TO_Z_UP)
        small = Mesh(proxy.update(pieces_wing.pose(**controls).vertices), proxy.faces).transform(P.MH_TO_Z_UP)
        # the proxy lies within the posed wing's bounding box (it is an average)
        lo, hi = posed.bounds
        assert (small.vertices >= lo - 1e-9).all() and (small.vertices <= hi + 1e-9).all()
        ious = _iou(posed, small, VIEWS)
        assert min(ious) >= floor, np.round(ious, 3)
    # a folded wing is a different (smaller) proxy: it follows the pose
    spread = Mesh(proxy.update(pieces_wing.pose().vertices), proxy.faces)
    folded = Mesh(proxy.update(pieces_wing.pose(fold=1.0).vertices), proxy.faces)
    assert np.ptp(folded.vertices, axis=0).max() < np.ptp(spread.vertices, axis=0).max()
    # rep lets a caller deform only the proxy's vertices
    assert np.allclose(proxy.rest, proxy.update(rig.rest_mesh.vertices))
    assert len(proxy.rep) == proxy.n_vertices


# -- render_draft ---------------------------------------------------------------


def _scene():
    ball = P.Shell(primitives.icosphere(1.0, 3), sh.SKIN)
    other = P.Shell(primitives.icosphere(0.5, 2).translated((1.6, 0.2, 0.0)), sh.EYE)
    return [ball, other]


def test_draft_matches_the_final_render_in_silhouette_and_roughly_in_colour():
    shells = _scene()
    final = P.render_portrait(shells, size=(96, 120), background=None)
    for mode in ("flat", "smooth", ["flat", "smooth"]):
        draft = D.render_draft(shells, size=(96, 120), shading=mode, background=None)
        assert draft.shape == final.shape and draft.dtype == np.uint8
        a, b = final[:, :, 3] == 255, draft[:, :, 3] == 255
        assert (a & b).sum() / (a | b).sum() > 0.97
        both = a & b
        diff = np.abs(draft[both, :3].astype(int) - final[both, :3].astype(int))
        assert diff.mean() < 14, (mode, diff.mean())
    # same background as the final renderer
    d = D.render_draft(shells, size=(96, 120))
    assert (d[0, 0] == P.render_portrait(shells, size=(96, 120))[0, 0]).all()


def test_draft_float64_is_close_to_float32():
    shells = _scene()
    a = D.render_draft(shells, size=(80, 100), dtype=np.float64)
    b = D.render_draft(shells, size=(80, 100), dtype=np.float32)
    assert (a != b).any(axis=2).mean() < 0.01


def test_draft_per_shell_modes_and_validation():
    shells = _scene()
    empty = P.Shell(Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)))
    img = D.render_draft(shells + [empty], size=(60, 70), shading=["flat", "smooth", "flat"])
    assert img.shape == (70, 60, 4)
    with pytest.raises(ValueError):
        D.render_draft(shells, shading="phong")
    with pytest.raises(ValueError):
        D.render_draft(shells, shading=["flat"])
    assert D.render_draft([empty], size=(10, 10)).shape == (10, 10, 4)


def test_draft_flat_uses_per_vertex_albedo_and_materials():
    mesh = primitives.icosphere(1.0, 2)
    red = np.tile(sh.hex_to_linear("#d02020"), (len(mesh.vertices), 1))
    plain = D.render_draft([P.Shell(mesh, sh.GENERIC, ao=False)], size=(50, 50))
    painted = D.render_draft([P.Shell(mesh, sh.GENERIC, ao=False, base_colors=red)], size=(50, 50))
    m = painted[:, :, 3] == 255
    assert painted[25, 25, 0] > painted[25, 25, 2] * 1.5
    assert (plain[25, 25] != painted[25, 25]).any() and m.any()


def test_draft_check_is_called_and_can_cancel():
    shells = _scene()
    calls = []
    D.render_draft(shells, size=(60, 70), check=lambda: calls.append(1))
    assert len(calls) >= 3
    token = P.CancelToken()
    token.cancel()
    with pytest.raises(P.Cancelled):
        D.render_draft(shells, size=(60, 70), check=token)


@needs_assets
def test_draft_frame_rate_for_body_and_two_proxied_wings(world):
    """Orbit frames at 320x480 for body + two proxied wings: well past interactive.

    The design target is >= 60 fps (measured about 100 fps on the reference
    machine, see docs/renderer.md); the assertion is loose so a loaded CI
    machine does not flake, but is far below what the full renderer reaches.
    """
    base, lib, proxy = world
    wings = _bat(world)
    shells = [P.Shell(proxy.mesh(None), sh.SKIN)]
    for _, pw in wings:
        rig = pw.rig
        cp = D.ClusterProxy.by_pieces(
            rig.rest_mesh.vertices, rig.rest_mesh.faces,
            D.pieces_from_ranges([rig.ranges[p.name] for p in rig.pieces],
                                 len(rig.rest_mesh.vertices)))
        shells.append(P.Shell(Mesh(cp.update(pw.pose().vertices), cp.faces).transform(P.MH_TO_Z_UP),
                              sh.GENERIC))
    total = sum(len(s.mesh.faces) for s in shells)
    assert total < 9000
    times = []
    for k in range(25):
        t = time.perf_counter()
        D.render_draft(shells, size=(320, 480), azimuth=10 * k, shading=["smooth", "flat", "flat"])
        times.append(time.perf_counter() - t)
    best = min(times)
    assert best < 1 / 30, f"{best * 1e3:.1f} ms per frame"
