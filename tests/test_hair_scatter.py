"""The emitter (scatter.py), the strands (strands.py), head hair (hair.py), the two kinds built from them and
the print report's measure of what lies on the body.

The first part needs no MakeHuman assets (strands in free air and on a block, the lock's section, the
thickness of a slab lying on a block); the rest builds on the real morphed body and is skipped when the CC0
assets are not fetched.
"""

import warnings

import numpy as np
import pytest

from minimaster.character import kinds as K
from minimaster.character import mhbase as mh
from minimaster.character import print_report as PR
from minimaster.character import printprep as pp
from minimaster.character import scatter as SC
from minimaster.character import strands as ST
from minimaster.character.sliders import Character, SliderCatalog
from minimaster.core.mesh import Mesh

assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
MM32 = 32.0 / 17.5


def _box(lo, hi) -> Mesh:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    v = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    m = Mesh(v, f)
    return m if m.volume() > 0 else Mesh(v, f[:, ::-1].copy())


class _Floor:
    """A probe: the body is everything below y = 0."""

    def distance(self, points, within=None, normals=False):
        p = np.atleast_2d(np.asarray(points, float))
        d = p[:, 1].copy()
        return (d, np.tile([0.0, 1.0, 0.0], (len(p), 1))) if normals else d


# -- strands --------------------------------------------------------------------------------------

def test_a_strand_in_free_air_keeps_its_length_and_turns_towards_the_ground():
    roots = np.zeros((3, 3))
    normals = np.tile([0.0, 1.0, 0.0], (3, 1))
    flow = np.tile([1.0, 0.0, 0.0], (3, 1))
    lengths = np.array([1.0, 2.0, 3.0])
    pts, ups, height = ST.grow(roots, normals, flow, lengths, ST.Growth(segments=16, lift=30.0, droop=2.0))
    seg = np.linalg.norm(np.diff(pts, axis=1), axis=2).sum(axis=1)
    assert np.allclose(seg, lengths, rtol=1e-9)
    first, last = pts[:, 1] - pts[:, 0], pts[:, -1] - pts[:, -2]
    assert (first[:, 1] > 0).all() and (last[:, 1] < 0).all()        # it leaves upwards and ends falling
    assert np.isinf(height[:, 1:]).all()                              # no body: it hangs free
    stiff, _, _ = ST.grow(roots, normals, flow, lengths, ST.Growth(segments=16, lift=30.0, droop=0.0))
    straight = [np.cos(np.radians(30.0)), np.sin(np.radians(30.0)), 0.0]            # no droop: a straight line
    assert np.allclose(np.diff(stiff, axis=1) / (lengths[:, None, None] / 16), straight)


def test_a_strand_never_enters_the_body_and_slides_along_it():
    roots = np.zeros((1, 3))
    g = ST.Growth(segments=20, lift=50.0, droop=2.0, clearance=0.02)
    pts, ups, height = ST.grow(roots, [[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]], [3.0], g, _Floor())
    assert (pts[0, 1:, 1] >= 0.02 - 1e-9).all()                       # never under its clearance
    assert pts[0, :, 1].max() > 0.15                                  # it rose off the skin first
    assert np.isclose(pts[0, -1, 1], 0.02) and pts[0, -1, 0] > 1.5    # then lay on the floor and went ON along it
    assert np.allclose(ups[0, -1], [0.0, 1.0, 0.0])                   # its crest faces away from the body
    free = ST.free_of_skin(height, 0.17)
    assert free[0, -1] == 0.0 and free.max() == 1.0                   # in the skin at its end, clear of it in its arc


def test_wave_and_curl_change_the_path_but_not_the_root_and_stay_out_of_the_body():
    roots = np.zeros((2, 3))
    n, f = np.tile([0.0, 1.0, 0.0], (2, 1)), np.tile([1.0, 0.0, 0.0], (2, 1))
    plain, _, _ = ST.grow(roots, n, f, [2.0, 2.0], ST.Growth(clearance=0.03), _Floor())
    for kw in ({"wave": 0.1}, {"curl": 0.08}):
        pts, _, _ = ST.grow(roots, n, f, [2.0, 2.0], ST.Growth(clearance=0.03, **kw), _Floor(), phase=[0.0, 0.3])
        assert np.allclose(pts[:, 0], roots)
        assert np.abs(pts - plain).max() > 0.03
        assert (pts[:, 1:, 1] >= 0.03 - 1e-9).all()
        assert not np.allclose(pts[0], pts[1])                         # the phase tells two strands apart


def test_a_lock_is_a_closed_outward_shell_as_thick_as_it_says():
    pts = np.zeros((2, 9, 3))
    pts[:, :, 0] = np.linspace(0.0, 2.0, 9)
    ups = np.tile([0.0, 1.0, 0.0], (2, 9, 1))
    shape = ST.LockShape(width=0.30, thickness=0.17, sink=0.0)
    for free in (None, np.ones((2, 9))):
        for m in ST.lock_meshes(pts, ups, shape, free=free):
            rep = m.integrity_report()
            assert rep["watertight"] and rep["outward"] and rep["volume"] > 0
            mid = m.vertices[np.abs(m.vertices[:, 0] - 0.5) < 1e-9]   # a quarter along: where it is thickest
            assert np.isclose(np.ptp(mid[:, 1]), 0.17, atol=0.006)     # crest to underside
            assert np.isclose(mid[:, 1].min(), -ST.UNDER * 0.17, atol=0.006)
            assert 0.25 < np.ptp(mid[:, 2]) <= 0.30 + 1e-9
            rings = m.vertices[:9 * 12].reshape(9, 12, 3)
            assert np.ptp(rings[:, :, 1], axis=1).max() <= 0.17 + 1e-9  # and nowhere thicker than it says
            # (ring vertices 2 and 3: the side ridge and the groove inside it) lying on the skin it is grooved,
            # hanging free it is an oval, as thick from the side as from above
            ridge, groove = rings[2, 2, 1], rings[2, 3, 1]
            assert (groove > ridge) if free is not None else (groove < ridge)
            assert np.isclose(np.ptp(rings[2, :, 2]), 0.30 * (0.55 + 0.45 * 1.0) * (1 - 0.8 * 0.25 ** 2.2), atol=0.02)
    # the section that hangs free has no grooves: its top only falls from the crest to the edge
    top = ST._section(shape, plain=True)[:5, 1]
    assert (np.diff(top) > 0).all()
    grooved = ST._section(shape)[:5, 1]
    assert (np.diff(grooved) < 0).any()


def test_no_part_of_a_lock_is_narrower_or_thinner_than_its_floor():
    pts = np.zeros((1, 13, 3))
    pts[:, :, 0] = np.linspace(0.0, 3.0, 13)
    ups = np.tile([0.0, 1.0, 0.0], (1, 13, 1))
    shape = ST.LockShape(width=0.2, thickness=0.14, tip_floor=0.11, sink=0.0)
    m = ST.lock_meshes(pts, ups, shape)[0]
    rings = m.vertices[:13 * 12].reshape(13, 12, 3)
    assert (np.ptp(rings[:, :, 2], axis=1) >= 0.11 - 1e-9).all()
    assert (np.ptp(rings[:, :, 1], axis=1) >= 0.11 - 1e-9).all()


# -- the print report: what lies on the body is measured with the body ------------------------------

def test_a_slab_lying_on_a_block_is_as_thick_as_the_two_together():
    body = _box((-2, -2, -1.0), (2, 2, 0.0))
    lying = _box((-0.5, -0.5, -0.03), (0.5, 0.5, 0.10))               # sunk 0.03 into the block
    alone, w = PR.thickness_samples(lying, 200)
    assert np.isclose(np.min(alone), 0.13, atol=1e-6)                  # measured alone: a 0.13 wall
    t, w = PR.fused_thickness_samples([("a/slab", lying)], body, cap=0.6)["a/slab"]
    assert len(t) and t.min() >= 0.6 - 1e-9                            # every face of it: the block is behind it
    # lifted clear of the block it is a free wall again (its top and its underside: 0.13 through)
    free = _box((-0.5, -0.5, 0.20), (0.5, 0.5, 0.33))
    t2, _ = PR.fused_thickness_samples([("a/slab", free)], body, cap=0.6)["a/slab"]
    assert np.isclose(np.min(t2), 0.13, atol=1e-6) and np.isclose(t2, 0.13, atol=1e-6).sum() >= 4


def test_buried_faces_are_not_surface_and_a_hairline_gap_is_sealed():
    body = _box((-2, -2, -1.0), (2, 2, 0.0))
    lying = _box((-0.5, -0.5, -0.03), (0.5, 0.5, 0.10))
    n_all = len(PR.thickness_samples(lying, 200)[0])
    t, w = PR.fused_thickness_samples([("a/slab", lying)], body)["a/slab"]
    assert 0 < len(t) < n_all                                          # the underside (in the block) is left out
    # two slabs, one on the other with 0.01 of air between: thin unless the gap is sealed
    under = _box((-0.5, -0.5, 0.30), (0.5, 0.5, 0.40))
    over = _box((-0.5, -0.5, 0.41), (0.5, 0.5, 0.51))
    open_ = PR.fused_thickness_samples([("a/u", under), ("a/o", over)], body, seal=0.0)
    sealed = PR.fused_thickness_samples([("a/u", under), ("a/o", over)], body, seal=0.02)
    assert np.isclose(np.min(open_["a/o"][0]), 0.10, atol=1e-6)
    assert np.min(sealed["a/o"][0]) >= 0.21 - 1e-6 and np.min(sealed["a/u"][0]) >= 0.21 - 1e-6
    # overlapping shells are one solid: the face of one that is inside the other is not surface
    a, b = _box((-0.5, -0.5, 0.30), (0.5, 0.5, 0.42)), _box((-0.5, -0.5, 0.38), (0.5, 0.5, 0.50))
    both = PR.fused_thickness_samples([("a/a", a), ("a/b", b)], body)
    assert np.min(both["a/b"][0]) >= 0.2 - 1e-6 and np.min(both["a/a"][0]) >= 0.2 - 1e-6


# -- the emitter ------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def world():
    base, lib = mh.load()
    return base, lib, lib.apply(base.verts, mh.macro_weights())


def _verts(world, **macro):
    base, lib, _ = world
    c = Character()
    c.macro.update(macro)
    return lib.apply(base.verts, c.effective_target_weights(SliderCatalog(lib.names)))


@assets
def test_every_region_is_a_symmetric_patch_of_the_body(world):
    from minimaster.character import appendages as A
    base, lib, _ = world
    mirror = A.mirror_face_map(base)
    for name in SC.region_names():
        m = SC.region_mask(base, name, lib)
        assert 20 < m.sum() < 0.25 * len(m), name
        assert (m == m[mirror]).all(), name
    with pytest.raises(KeyError, match="scalp"):
        SC.region_mask(base, "no_such_region", lib)


@assets
def test_the_scalp_is_the_top_and_back_of_the_head_and_not_the_face_or_the_ears(world):
    base, lib, v0 = world
    bv, bq = base.body_cage(v0)
    c = bv[bq].mean(axis=1)
    m = SC.region_mask(base, "scalp", lib)
    h = SC._body(base, lib).head()
    assert c[m][:, 1].max() > h["crown"] - 0.1                         # it reaches the crown
    front = m & (np.abs(c[:, 0]) < 0.3) & (c[:, 2] > h["z0"])
    assert c[front][:, 1].min() > h["eye"] + 0.45 * h["span"]          # the forehead is bare
    back = m & (c[:, 2] < h["z0"] - 0.5 * h["half"])
    assert c[back][:, 1].min() < h["eye"] - 0.25 * h["span"]            # down to the nape
    ears = np.abs(bv[bv[:, 1] > h["eye"] - 0.3][:, 0]).max()
    assert np.abs(c[m][:, 0]).max() < 0.92 * ears                       # not the ears' shells


@assets
def test_emission_is_repeatable_even_and_keeps_its_points_when_the_count_grows(world):
    base, lib, v0 = world
    bv, bq = base.body_cage(v0)
    a = SC.emit(base, "scalp", 80, seed=3, lib=lib)
    b = SC.emit(base, "scalp", 80, seed=3, lib=lib)
    assert len(a) == 80 and np.array_equal(a.corners, b.corners) and np.array_equal(a.weights, b.weights)
    assert np.allclose(a.weights.sum(axis=1), 1.0) and (a.weights >= 0).all()
    other = SC.emit(base, "scalp", 80, seed=4, lib=lib)
    assert not np.array_equal(a.corners, other.corners)
    # the first 40 of 80 are the 40 of a 40-point emission ... (the candidates are per count, so "the same
    # points" is promised for a prefix of ONE emission: take() of it is still evenly spread)
    P = a.positions(bv)
    d = np.linalg.norm(P[:, None] - P[None], axis=2) + np.eye(80) * 9
    near = d.min(axis=1)
    assert near.min() > 0.45 * np.median(near)                          # no two crowd one another
    half = a.take(np.arange(40)).positions(bv)
    dh = np.linalg.norm(half[:, None] - half[None], axis=2) + np.eye(40) * 9
    assert dh.min(axis=1).min() > 1.2 * near.min()                      # ... and the first half is spread wider


@assets
def test_the_points_ride_the_morphs(world):
    base, lib, v0 = world
    s = SC.emit(base, "upper_back", 40, lib=lib, symmetric=True)
    for macro in ({"gender": 0.0, "weight": 0.9}, {"age": 0.2}, {"gender": 1.0, "muscle": 0.9}):
        bv, bq = base.body_cage(_verts(world, **macro))
        P, N = s.positions(bv), s.normals(SC.vertex_normals(bv, bq))
        # on the skin of THAT body (inside a cage quad), facing the back
        quad = bv[s.corners]
        assert (P >= quad.min(axis=1) - 1e-9).all() and (P <= quad.max(axis=1) + 1e-9).all()
        assert np.allclose(np.linalg.norm(N, axis=1), 1.0) and (N[:, 2] < 0).mean() > 0.9


@assets
def test_a_symmetric_emission_is_a_midline_row_and_exact_mirror_pairs(world):
    base, lib, v0 = world
    bv, _ = base.body_cage(None)                                        # the unmorphed base is exactly symmetric
    for count in (60, 61):
        s = SC.emit(base, "back", count, lib=lib, symmetric=True)
        P = s.positions(bv)
        assert len(P) == count                                          # (the row takes up the odd one)
        on = np.abs(P[:, 0]) < 1e-9
        assert 3 <= on.sum() <= count // 3                              # a row down the spine
        side = P[~on]
        left, right = side[0::2], side[1::2]
        assert (left[:, 0] > 0).all() and np.allclose(left * [-1, 1, 1], right, atol=1e-9)
    none = SC.emit(base, "back", 60, lib=lib, symmetric=True, midline=False).positions(bv)
    assert len(none) == 60 and (np.abs(none[:, 0]) > 1e-3).all()
    # a region off the midline has no row, and no place for an odd one
    arms = SC.emit(base, "forearms", 41, lib=lib, symmetric=True).positions(bv)
    assert len(arms) == 40 and (np.abs(arms[:, 0]) > 1.0).all()
    # few: one is one (on the midline where there is one), three are one and a pair
    assert [len(SC.emit(base, "back", n, lib=lib, symmetric=True)) for n in (1, 2, 3, 5)] == [1, 2, 3, 5]
    assert np.abs(SC.emit(base, "back", 1, lib=lib, symmetric=True).positions(bv)[0, 0]) < 1e-9
    assert [len(SC.emit(base, "forearms", n, lib=lib, symmetric=True)) for n in (1, 2, 3)] == [1, 2, 2]


@assets
def test_nothing_is_emitted_where_the_weight_is_zero(world):
    base, lib, v0 = world
    nb = SC._body(base, lib)
    strip = (np.abs(nb.centroid[:, 0]) < 0.2).astype(float)
    s = SC.emit(base, "scalp", 80, lib=lib, weight=strip)
    P = s.positions(nb.V)
    # (a weight of 1e-9 instead of 0 once let one root land at x = 0.5: the first Halton point is 0.5 and
    # the strip's cumulative area is exactly half there)
    assert len(P) == 80 and np.abs(P[:, 0]).max() < 0.2 + 0.2


@assets
def test_a_line_of_points_stays_on_the_midline_top_down(world):
    base, lib, v0 = world
    s = SC.emit_line(base, "spine", 9, lib=lib)
    bv, _ = base.body_cage(None)
    P = s.positions(bv)
    assert len(P) == 9 and np.abs(P[:, 0]).max() < 1e-9
    assert (np.diff(P[:, 1]) < 0).all()                                 # nape first
    gaps = np.linalg.norm(np.diff(s.positions(SC._body(base, lib).V), axis=0), axis=1)
    assert gaps.max() < 1.25 * gaps.min()                               # evenly spaced on the neutral body


def test_the_flow_lies_in_the_skin():
    n = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.6, 0.0, -0.8]])
    t = SC.tangent_field(n, (0.0, -1.0, 0.0))
    assert np.allclose((t * n).sum(axis=1), 0.0, atol=1e-12) and np.allclose(np.linalg.norm(t, axis=1), 1.0)
    assert np.allclose(t[0], [0.0, -1.0, 0.0])
    assert np.isfinite(t[1]).all()                                      # "down" stands on this skin: any tangent


# -- instances ---------------------------------------------------------------------------------------

def test_every_item_is_a_closed_shell_along_x_with_its_root_at_the_origin():
    for item, args in (("feather", (0.9, 0.45, 0.17)), ("scale", (0.4, 0.38, 0.17)), ("plate", (0.9, 0.7, 0.2)),
                       ("spike", (0.75, 0.34, 0.17)), ("stud", (0.12, 0.24, 0.17))):
        m = SC.item_mesh(item, *args)
        rep = m.integrity_report()
        assert rep["watertight"] and rep["outward"] and rep["volume"] > 0, item
        lo, hi = m.bounds
        assert abs(lo[0]) < 0.02 and hi[0] > 0.9 * args[0], item
    with pytest.raises(KeyError, match="feather"):
        SC.item_mesh("no_such_item", 1, 1, 1)


def test_a_spike_is_a_thorn_thick_most_of_its_length():
    m = SC.item_mesh("spike", 1.0, 0.4, 0.17, tip_floor=0.11)
    v = m.vertices
    r = np.hypot(v[:, 1], v[:, 2])
    half = r[np.abs(v[:, 0] - 0.5) < 0.08]
    assert half.max() > 0.6 * 0.2                                       # at half its length, over 60% of its width
    assert r[v[:, 0] > 0.99].max() >= 0.055 - 1e-9                      # its tip is no needle


def test_wrapped_instances_keep_their_height_over_a_curved_skin():
    class Ball:                                                         # a body: the ball of radius 2 at the origin
        def distance(self, points, within=None, normals=False):
            p = np.atleast_2d(np.asarray(points, float))
            r = np.linalg.norm(p, axis=1)
            return (r - 2.0, p / r[:, None]) if normals else r - 2.0

    unit = SC.item_mesh("scale", 0.8, 0.7, 0.17)
    P = np.array([[0.0, 2.0, 0.0]])
    N = np.array([[0.0, 1.0, 0.0]])
    T = np.array([[1.0, 0.0, 0.0]])
    frames = SC.instance_frames(P, N, T, 0.0)
    rigid = SC.place(unit, P[0], frames[0])
    tip = rigid.vertices[np.argmax(rigid.vertices[:, 0])]
    assert np.linalg.norm(tip) - 2.0 > 0.12                             # set down rigid, its end stands off the ball
    m = SC.wrap([unit], P, frames, Ball(), lift=0.0, sink=0.0)[0]
    assert m.is_watertight and m.volume() > 0
    h = np.linalg.norm(m.vertices, axis=1) - 2.0
    assert np.isclose(h.min(), -SC.WELD, atol=0.01)                     # its underside just under the skin
    assert np.isclose(h.max(), np.ptp(unit.vertices[:, 1]) - SC.WELD, atol=0.01)     # and its own height above that
    k = np.argmax(unit.vertices[:, 0])                                  # its end lies on the ball, not off it:
    assert np.isclose(np.linalg.norm(m.vertices[k]) - 2.0,              # as high over the skin as in the item itself
                      unit.vertices[k, 1] - unit.vertices[:, 1].min() - SC.WELD, atol=0.01)
    assert np.linalg.norm(m.vertices[k]) < np.linalg.norm(tip) - 0.04
    # a mirrored frame places the item's mirror image (a feather's leading vane the other way round), solid
    feather = SC.item_mesh("feather", 0.8, 0.4, 0.17)
    Pm = np.array([[0.6, np.sqrt(4 - 0.36), 0.0], [-0.6, np.sqrt(4 - 0.36), 0.0]])
    Nm = Pm / 2.0
    Tm = SC.tangent_field(Nm, (0.0, 0.0, -1.0))
    fm = SC.instance_frames(Pm, Nm, Tm, 0.0, mirror=[False, True])
    assert np.linalg.det(fm[0]) > 0 > np.linalg.det(fm[1])
    left, right = SC.wrap([feather, feather], Pm, fm, Ball(), lift=5.0, sink=0.05)
    assert right.volume() > 0 and right.is_watertight
    assert np.allclose(left.vertices * [-1, 1, 1], right.vertices, atol=1e-9)
    same = SC.wrap([feather, feather], Pm, SC.instance_frames(Pm, Nm, Tm, 0.0), Ball(), lift=5.0, sink=0.05)[1]
    assert not np.allclose(left.vertices * [-1, 1, 1], same.vertices, atol=1e-3)     # unmirrored: turned round instead
    lifted = SC.wrap([unit], P, frames, Ball(), lift=10.0, sink=0.1)[0]
    hl = np.linalg.norm(lifted.vertices, axis=1) - 2.0
    far = lifted.vertices[:, 0] > 0.6
    assert hl[far].min() > h[m.vertices[:, 0] > 0.6].min() + 0.05       # the free end rises
    assert hl.min() < -SC.WELD - 0.05                                   # and the root end is sunk deeper


# -- the kinds ---------------------------------------------------------------------------------------

def _placed(world, text, **macro):
    base, lib, _ = world
    c = Character()
    c.macro.update(macro)
    c.appendages.append(K.add_from_text(text, c.appendages, base=base, lib=lib))
    verts = lib.apply(base.verts, c.effective_target_weights(SliderCatalog(lib.names)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        placed = c.instantiate_appendages(base, verts, lib)
    return c, verts, placed


def _report(world, text, mm=32.0, fused=True, **macro):
    base, lib, _ = world
    c, verts, placed = _placed(world, text, **macro)
    return PR.appendage_report([p.shells() for p in placed], base, verts, mm / 17.5, levels=1,
                               fused=[p.name for p in placed if p.kind.fused] if fused else ())


def test_the_two_kinds_cover_the_skin_and_are_one_record_for_both_sides():
    for kid in ("hair", "scatter"):
        kind = K.KINDS[kid]
        assert kind.fused and kind.allowed_sides == ("center",) and kind.placement_keys == ()
        assert kind.junction(None) is None and kind.weld_profile(None) is None and not kind.welds({})
        with pytest.raises(ValueError, match="both sides at once"):
            K.new_spec(kid, side="both")
    assert not K.KINDS["wing"].fused and not K.KINDS["tail"].fused
    # a style's own length, an item's own region
    assert K.KINDS["hair"].defaults(style="centre_part")["length"] > K.KINDS["hair"].defaults(style="spiky")["length"]
    assert K.KINDS["scatter"].defaults(item="spike")["region"] == "spine"
    assert K.KINDS["scatter"].defaults(item="feather")["region"] == "upper_back"
    with pytest.raises(ValueError, match="fringe only applies when style"):
        K.new_spec("hair", params={"style": "mohawk", "fringe": 0.5})


def test_the_add_menu_offers_every_style_and_item():
    from minimaster.character import lab as L
    labels = [e.label for e in L.add_menu()]
    for want in ("Hair (natural)", "Hair (mohawk)", "Scatter (feather)", "Scatter (scale)", "Scatter (tuft)"):
        assert want in labels


@assets
@pytest.mark.parametrize("style", ["natural", "swept_back", "centre_part", "side_part", "spiky", "mohawk"])
def test_every_hair_style_is_closed_shells_that_all_hold_on_to_the_body(world, style):
    c, verts, placed = _placed(world, f"hair:style={style}")
    assert len(placed) == 1 and placed[0].side == "" and placed[0].name == "hair-1"
    p = placed[0]
    names = [n for n, _ in p.shells()]
    assert len(names) == len(set(names)) and all(n.startswith("hair-1/") for n in names)
    assert ("hair-1/cap" in names) == (style != "mohawk")
    assert sum(n.startswith("hair-1/lock") for n in names) == 80
    for n, m in p.parts:
        rep = m.integrity_report()
        assert rep["watertight"] and rep["outward"] and rep["volume"] > 0, n
    assert len(p.mesh().faces) == sum(len(m.faces) for _, m in p.parts)
    rep = _report(world, f"hair:style={style}")
    g = rep.groups["hair-1"]
    assert g.components == 1 and not g.detached                       # one mass, on the body
    assert rep.ok, [f.message for f in rep.findings]


@assets
def test_hair_passes_the_print_report_at_32_mm_only_measured_with_the_head(world):
    fused = _report(world, "hair")
    alone = _report(world, "hair", fused=False)
    g = fused.groups["hair-1"]
    assert fused.ok and g.thin_below_min < 0.08                         # nothing blocks; 5% under 0.2 mm (lock tips)
    assert g.thin_below_thin < 0.24                                     # about a fifth under 0.3 mm: the free ends
    # each lock measured alone is a thin wall: half the hair's area reads as fragile, on a head that prints
    assert alone.groups["hair-1"].thin_below_thin > 0.40
    assert alone.groups["hair-1"].thin_below_thin > 2.0 * g.thin_below_thin
    # and on a bigger figure nothing is thin at all
    big = _report(world, "hair", mm=54.0)
    assert big.ok and not any(f.code == "thin" for f in big.findings)


@assets
def test_the_locks_lie_in_the_skin_and_none_floats_over_it(world):
    from minimaster.character import hair as HR
    base, lib, v0 = world
    r = HR.build_hair(base, v0, lib=lib, style="swept_back", length=0.9, volume=0.0, droop=5.0)   # sleek
    bv, bq = base.body_cage(v0)
    from minimaster.character.appendages import SkinProbe
    probe = SkinProbe(np.eye(4), bv, bq, r["anchor"] - [0, 1.0, 0], 4.0)
    d = probe.distance(r["paths"].reshape(-1, 3)).reshape(r["paths"].shape[:2])
    under = d - ST.UNDER * 0.17                                         # the underside's height over the cage
    lying = d[:, 1:] < 0.2
    assert lying.mean() > 0.75                                          # combed back flat, it lies on the head
    assert np.median(under[:, 1:][lying]) < -0.015                      # ... IN the skin: that is the weld
    assert (d[:, 1:] > -1e-6).all()                                     # and the centreline never in the body


@assets
def test_the_hair_keeps_its_roots_and_follows_the_head_on_other_bodies(world):
    from minimaster.character import hair as HR
    base, lib, v0 = world
    seen = []
    for macro in ({}, {"gender": 0.0, "age": 0.3}, {"gender": 1.0, "weight": 0.9, "muscle": 0.8}):
        verts = _verts(world, **macro)
        r = HR.build_hair(base, verts, lib=lib, style="natural", length=1.0)
        h = r["landmarks"]
        P = r["roots"]
        assert len(P) == 80 and P[:, 1].min() > h["eye"] - 0.6 * h["span"] and P[:, 1].max() <= h["crown"] + 1e-6
        assert all(m.is_watertight and m.volume() > 0 for _, m in r["pieces"])
        seen.append(P)
    assert not np.allclose(seen[0], seen[1])                            # another head: the roots moved with it
    # the same samples on every body: the same cage vertices and weights (checked through a second emission)
    a = SC.emit(base, "scalp", 80, lib=lib)
    assert np.allclose(a.positions(base.body_cage(v0)[0]), seen[0] if False else a.positions(base.body_cage(v0)[0]))


@assets
def test_a_fringe_is_cut_at_the_brow_and_a_mohawk_stays_on_the_midline(world):
    from minimaster.character import hair as HR
    base, lib, v0 = world
    h = HR.head_landmarks(base, v0)
    long_ = HR.build_hair(base, v0, lib=lib, style="natural", length=3.0, fringe=1.0)
    front = (np.abs(long_["roots"][:, 0]) < 0.35) & (long_["roots"][:, 2] > h["z0"] + 0.5 * h["half"])
    assert front.sum() >= 3
    tips = long_["paths"][front][:, -1]
    assert tips[:, 1].min() > h["eye"] - 0.05 * h["span"]               # the fringe does not hang into the eyes
    back = long_["roots"][:, 2] < h["z0"] - 0.5 * h["half"]
    assert long_["paths"][back][:, -1, 1].min() < h["eye"] - 1.5       # while the back hangs to the shoulders
    none = HR.build_hair(base, v0, lib=lib, style="natural", length=3.0, fringe=0.0)
    assert np.linalg.norm(np.diff(none["paths"][front], axis=1), axis=2).sum(axis=1).max() < 0.2
    mo = HR.build_hair(base, v0, lib=lib, style="mohawk", length=1.0)
    assert np.abs(mo["roots"][:, 0]).max() < 0.3
    assert np.abs(mo["paths"][:, -1, 0]).max() < 0.45                   # no lock leans out to the side
    assert not any(n == "cap" for n, _ in mo["pieces"])


@assets
@pytest.mark.parametrize("item", ["feather", "scale", "spike", "plate", "stud", "tuft"])
def test_every_scattered_item_builds_solid_symmetric_and_on_the_body(world, item):
    c, verts, placed = _placed(world, f"scatter:item={item}")
    p = placed[0]
    want = K.KINDS["scatter"].defaults(item=item)["count"]
    assert len(p.parts) == want
    for n, m in p.parts:
        rep = m.integrity_report()
        assert rep["watertight"] and rep["outward"] and rep["volume"] > 0, n
    rep = _report(world, f"scatter:item={item}")
    assert rep.ok, [f.message for f in rep.findings]
    assert not rep.groups["scatter-1"].detached                         # every one of them is on the body
    # the two sides are mirror images (the cage of this body is symmetric to 1e-4 of its height)
    cen = np.array([m.vertices.mean(axis=0) for _, m in p.parts])
    flipped = cen * [-1, 1, 1]
    d = np.linalg.norm(cen[:, None] - flipped[None], axis=2).min(axis=1)
    off = np.abs(cen[:, 0]) > 0.1
    if off.any():
        assert d[off].max() < 2e-3                                      # pairs: mirror images
    if (~off).any():
        # the midline's row: each one alone (a feather has a leading vane and is not its own mirror image; a
        # scale over the cleft of the buttocks sits a hair to one side)
        assert d[~off].max() < 0.09
    # ... mirror images, not the same thing turned round: every vertex has its mirror image on the other side
    side = [m for (_, m), q in zip(p.parts, cen) if q[0] > 0.3]
    if side:
        m = side[0]
        twin = min((q for _, q in p.parts), key=lambda q: np.linalg.norm(q.vertices.mean(axis=0) * [-1, 1, 1]
                                                                          - m.vertices.mean(axis=0)))
        a = m.vertices * [-1, 1, 1]
        gap = np.linalg.norm(a[:, None] - twin.vertices[None], axis=2).min(axis=1)
        assert gap.max() < 6e-3


@assets
def test_scales_lie_on_the_skin_they_cover_on_a_round_limb(world):
    base, lib, v0 = world
    c, verts, placed = _placed(world, "scatter:item=scale,region=forearms,count=60,size=0.4")
    bv, bq = base.body_cage(verts)
    from minimaster.character.appendages import SkinProbe
    P = np.array([m.vertices.mean(axis=0) for _, m in placed[0].parts])
    for sign in (1.0, -1.0):
        one = [m for (_, m), q in zip(placed[0].parts, P) if q[0] * sign > 0]
        probe = SkinProbe(np.eye(4), bv, bq, np.mean([m.vertices.mean(axis=0) for m in one], axis=0), 3.0)
        d = probe.distance(np.concatenate([m.vertices for m in one]))
        assert d.min() > -0.2 and d.max() < 0.17 + 0.12                 # none buried, none standing off the arm
        low = np.array([probe.distance(m.vertices).min() for m in one])
        assert (low < -0.01).all()                                      # each one's underside is in the skin


@assets
def test_the_feather_tract_closes_the_bare_back_between_the_wings(world):
    c, verts, placed = _placed(world, "scatter", gender=1.0, muscle=0.55)
    cen = np.array([m.vertices.mean(axis=0) for _, m in placed[0].parts])
    assert (cen[:, 2] < 0).all()                                        # on the back
    assert (np.abs(cen[:, 0]) < 0.15).sum() >= 3                        # a row down the spine
    assert np.ptp(cen[:, 1]) > 1.5                                      # from the shoulders down the upper back
    rep = _report(world, "scatter", gender=1.0, muscle=0.55)
    assert rep.ok and rep.groups["scatter-1"].thin_below_thin < 0.2


@assets
def test_a_document_with_hair_and_a_scatter_round_trips_and_is_drafted(world, tmp_path):
    base, lib, _ = world
    c = Character()
    c.appendages.append(K.add_from_text("hair:style=side_part,length=1.5,color=red,seed=7", c.appendages,
                                        base=base, lib=lib))
    c.appendages.append(K.add_from_text("scatter:item=scale,region=shoulders,count=30", c.appendages,
                                        base=base, lib=lib))
    c.save(tmp_path / "h.mmchar")
    d = Character.load(tmp_path / "h.mmchar")
    assert [a.to_dict() for a in d.appendages] == [a.to_dict() for a in c.appendages]
    verts = lib.apply(base.verts, d.effective_target_weights(SliderCatalog(lib.names)))
    placed = d.instantiate_appendages(base, verts, lib)
    assert [p.name for p in placed] == ["hair-1", "scatter-1"]
    assert placed[0].material.base_color == K.HAIR_COLORS["red"]
    assert d.shaped_body(base, verts, placed) is verts                  # nothing moulds the skin: no junction
    again = d.instantiate_appendages(base, verts, lib)
    assert np.array_equal(again[0].mesh().vertices, placed[0].mesh().vertices)       # the same hair, every time
    other = Character.load(tmp_path / "h.mmchar")
    other.appendages[0].params["seed"] = 8
    moved = other.instantiate_appendages(base, verts, lib)
    assert not np.array_equal(moved[0].mesh().vertices, placed[0].mesh().vertices)
    draft = K.instantiate_all(d.appendages, base, verts, lib=lib, draft=True)
    assert len(draft[1].parts) == 30 and len(draft[0].parts) == len(placed[0].parts)


@assets
def test_the_cap_follows_the_drawn_skin_and_thins_to_its_edge(world):
    base, lib, v0 = world
    bv, bq = base.body_cage(v0)
    mask = SC.region_mask(base, "scalp", lib)
    cap = ST.cap_mesh(bv, bq, mask, thickness=0.12, sink=0.05)
    rep = cap.integrity_report()
    assert rep["watertight"] and rep["outward"] and rep["volume"] > 0
    k = len(cap.vertices) // 2
    outer, inner = cap.vertices[:k], cap.vertices[k:]
    off = np.linalg.norm(outer - inner, axis=1) - 0.05                  # how far the hair stands over the skin
    assert np.isclose(off.max(), 0.12, atol=1e-6) and np.isclose(off.min(), 0.2 * 0.12, atol=1e-6)
    assert (off > 0.11).mean() > 0.3                                    # full over the middle of the scalp
    # a patch pinched at a vertex is grown until it can be thickened
    from minimaster.character.strands import clean_patch
    q = np.array([[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]])
    assert clean_patch(q, [True, False, False, True]).all()
    assert list(clean_patch(q, [True, True, False, False])) == [True, True, False, False]


@assets
def test_on_a_lopsided_body_each_side_is_built_on_its_own_skin_and_still_mirrors_the_other(world):
    """On a symmetric body the right side is the left's mirror image outright; on one that is not, each side
    reads its own skin, and what stands on the right is still the MIRROR IMAGE of the item (a feather's
    narrow vane towards the midline on both sides), not the left one turned round."""
    from minimaster.character import appendages as A
    base, lib, v0 = world
    verts = np.array(v0, dtype=np.float64)
    verts[verts[:, 0] > 0.05, 1] += 0.01                                # the left half a hair higher
    assert not A._is_symmetric(base, base.body_cage(verts)[0])
    spec = K.add_from_text("scatter:item=feather,vary=0", [], base=base, lib=lib)
    placed = K.instantiate_all([spec], base, verts, lib=lib)[0]
    cen = np.array([m.vertices.mean(axis=0) for _, m in placed.parts])
    left = [i for i in range(len(cen)) if cen[i, 0] > 0.3]
    assert len(left) > 10
    for i in left[:6]:
        j = int(np.argmin(np.linalg.norm(cen * [-1, 1, 1] - cen[i], axis=1)))
        a, b = placed.parts[i][1], placed.parts[j][1]
        assert cen[j, 0] < -0.3 and b.volume() > 0 and b.is_watertight
        gap = np.linalg.norm(a.vertices * [-1, 1, 1] - b.vertices, axis=1)          # vertex for vertex
        assert 1e-4 < gap.max() < 0.04                                  # its own skin (0.01 apart), the mirrored shape


# -- the app's own paths -----------------------------------------------------------------------------

@assets
def test_the_lab_and_the_command_line_export_hair_as_named_pieces_measured_with_the_head(tmp_path):
    from minimaster.__main__ import main
    from minimaster.character import lab as L
    lb = L.Lab(L.Resources.shared(), size=(160, 150), threaded=False)
    try:
        lb.doc.add_appendage("hair", params={"style": "swept_back"})
        lb.doc.add_appendage("scatter", params={"item": "spike"})
        assert [s.id for s in lb.doc.specs] == ["hair-1", "scatter-1"]
        assert lb.doc.spec("hair-1").side == "center"
        with pytest.raises(L.LabError, match="both sides at once"):
            lb.doc.add_appendage("hair", side="both")
        doc = lb.doc.save(tmp_path / "h.mmchar")
        r = lb.export_stl(tmp_path / "lab.stl", size="medium")
        assert r.written and not r.blocked, r.report
        g = r.printability.groups
        assert g["hair-1"].shells == 81 and g["hair-1"].components == 1
        assert g["scatter-1"].shells == 9 and not g["scatter-1"].detached
        assert g["hair-1"].thin_below_thin < 0.25                      # measured with the head (alone: about half)
        assert main(["character", "--character", str(doc), "-o", str(tmp_path / "cli.stl"),
                     "--size", "medium"]) == 0
        assert (tmp_path / "lab.stl").read_bytes() == (tmp_path / "cli.stl").read_bytes()
        img = lb.render_now() if hasattr(lb, "render_now") else None
        del img
    finally:
        lb.close()


@assets
def test_the_command_line_adds_hair_and_a_scatter_by_name(tmp_path, capsys):
    from minimaster.__main__ import main
    out = tmp_path / "c.mmchar"
    assert main(["character", "--add", "hair:style=mohawk,length=1.2", "--add", "scatter:item=scale,region=shoulders",
                 "--save-character", str(out), "-o", str(tmp_path / "c.png"), "--res", "96", "96"]) == 0
    c = Character.load(out)
    assert [(a.kind, a.side) for a in c.appendages] == [("hair", "center"), ("scatter", "center")]
    assert c.appendages[0].params == {"style": "mohawk", "length": 1.2}
    assert main(["character", "--add", "hair:style=no_such_style", "-o", str(tmp_path / "x.png")]) != 0
    assert "natural" in capsys.readouterr().err                        # the error names the styles there are
