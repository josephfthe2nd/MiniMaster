"""Skin-conforming junctions (``minimaster.character.junction``).

The body cannot get new vertices (1,280 morph targets address it by vertex index) but its vertices
can be moved after the morphs, so the skin flows into an appendage root instead of a closed shell
simply overlapping it. The first half of this file needs no assets: a synthetic closed quad
cage (a sphere) is the body, so the mechanism is pinned down exactly (the fillet law, the caps,
the checks, the symmetry, the determinism). The second half builds real appendages on the real
morphed body and is skipped when the CC0 assets are not fetched.
"""

import dataclasses
import hashlib
import time

import numpy as np
import pytest

from minimaster.character import junction as J
from minimaster.character import mhbase as mh
from minimaster.core.subdiv import cage_is_closed, cube_cage, subdivide_quads

assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")


# -- a synthetic body ----------------------------------------------------------------------


class FakeBase:
    """The part of ``BaseMesh`` the junction touches: ``body_cage`` and ``body_verts_idx``.
    The body is the first ``n`` rows; ``extra`` rows after it (eyes, teeth, joint markers in the
    real mesh) must never move."""

    def __init__(self, verts, quads, extra=0):
        self.body_verts_idx = np.arange(len(verts))
        self.body_quads = quads
        self.n = len(verts)
        self.verts = np.vstack([verts, np.full((extra, 3), 7.0)])

    def body_cage(self, verts=None):
        v = self.verts if verts is None else verts
        return v[self.body_verts_idx], self.body_quads


def sphere(levels=4, radius=10.0):
    v, q = cube_cage(2.0)
    v, q = subdivide_quads(v, q, levels)
    v = v / np.linalg.norm(v, axis=1, keepdims=True) * radius        # a closed quad sphere
    return v, q


@pytest.fixture(scope="module")
def body():
    v, q = sphere()
    return FakeBase(v, q, extra=6)


def tube_up(x=0.0, radius=0.6, depth=1.0, length=6.0, radius_end=None, **kw):
    """A straight tube leaving the sphere's +Z pole at (x, 0), root ``depth`` inside the skin."""
    z0 = np.sqrt(100.0 - x * x) - depth
    pts = [(x, 0.0, z0 + t) for t in np.linspace(0.0, length, 4)]
    rad = [radius + (0 if radius_end is None else (radius_end - radius) * t / length)
           for t in np.linspace(0.0, length, 4)]
    return J.JunctionSpec("tube", pts, rad, **kw)


def displacement(base, specs, **kw):
    out = J.apply_junctions(base, base.verts, specs, **kw)
    return out - base.verts


# -- the description -----------------------------------------------------------------------


def test_a_spec_is_validated_and_hashable_data():
    spec = tube_up(width=1.5, height=0.8, sharpness=3.0)
    assert spec.active and spec.fillet_width == pytest.approx(1.5 * 0.6)
    assert hash(spec) is not None and spec == dataclasses.replace(spec)
    for bad in (dict(width=-1.0), dict(sharpness=0.5), dict(pull=2.0), dict(strength=float("nan"))):
        with pytest.raises(ValueError):
            tube_up(**bad)
    with pytest.raises(ValueError):
        J.JunctionSpec("x", [(0, 0, 0)], [0.5])               # a centreline needs two points
    with pytest.raises(ValueError):
        J.JunctionSpec("x", [(0, 0, 0), (0, 0, 1)], [0.5, -0.1])
    assert not tube_up(strength=0.0).active and not tube_up(width=0.0).active
    # numpy scalars are numbers (a width read out of an array), booleans are not
    assert tube_up(width=np.float32(1.5), height=np.float64(0.5)).fillet_width == pytest.approx(0.9)
    assert hash(tube_up(width=np.int64(2))) == hash(tube_up(width=2.0))
    with pytest.raises(ValueError):
        tube_up(width=True)
    assert J.JunctionSpec("m", (), (), mounds=(J.Mound((0, 0, 10), 2.0, 0.3),)).active


def test_a_spec_round_trips_and_transforms():
    spec = tube_up(width=1.2, mounds=(J.Mound((0, 0, 10), 2.0, 0.3, end=(1, 0, 10)),
                                      J.Mound((0, 1, 10), 1.0, -0.1, axis=(1, 0, 0), stretch=2.0)),
                   keep=(J.Keep((3, 0, 9), 0.5),))
    assert J.JunctionSpec.from_dict(spec.to_dict()) == spec
    assert J.JunctionSpec.from_dict(spec.to_dict()).digest() == spec.digest()
    assert tube_up(width=1.3).digest() != tube_up(width=1.2).digest()
    mirrored = spec.mirrored()
    assert mirrored.mirrored() == spec
    assert mirrored.mounds[0].center == (-0.0, 0.0, 10.0) and mirrored.mounds[0].end == (-1.0, 0.0, 10.0)
    assert mirrored.mounds[1].axis == (-1.0, 0.0, 0.0) and mirrored.keep[0].center == (-3.0, 0.0, 9.0)
    assert np.allclose(np.asarray(mirrored.points) * [-1, 1, 1], np.asarray(spec.points))
    moved = spec.transformed(np.diag([2.0, 2.0, 2.0, 1.0]))
    assert moved.radii == pytest.approx(tuple(2 * r for r in spec.radii))
    assert moved.fillet_width == pytest.approx(2 * spec.fillet_width)
    assert moved.mounds[0].height == pytest.approx(0.6)
    # the digest of a LIST does not depend on its order (the junction is a sum)
    a, b = tube_up(x=2.0), tube_up(x=-2.0)
    assert J.specs_digest([a, b]) == J.specs_digest([b, a])


def test_the_fillet_law_is_c1_at_both_ends():
    u = np.linspace(0.0, 1.0, 1001)
    f = J.fillet_remap(u)
    assert f[0] == 0.0 and f[-1] == 1.0
    assert np.all(np.diff(f) > 0)                              # monotone: the skin never folds back
    h = 1e-6
    assert (J.fillet_remap(1.0) - J.fillet_remap(1.0 - h)) / h == pytest.approx(1.0, abs=1e-4)
    assert (J.fillet_remap(h) - J.fillet_remap(0.0)) / h == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize("k", [2.0, 3.0])
def test_the_profile_is_tangent_to_the_wall_and_to_the_skin(k):
    W, H = 0.8, 0.5
    rho = np.linspace(0.0, W, 2001)
    h = J.fillet_height(rho, W, H, k)
    assert h[0] == pytest.approx(H) and h[-1] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(h) <= 1e-12)                          # it falls monotonically to the skin
    e = 1e-5
    # horizontal at the skin: d h / d rho -> 0 at rho = W
    slope_skin = (J.fillet_height(W, W, H, k) - J.fillet_height(W - e, W, H, k)) / e
    assert slope_skin == pytest.approx(0.0, abs=1e-3)
    # vertical at the wall: the climb per unit of distance is huge at rho -> 0
    slope_wall = (J.fillet_height(e, W, H, k) - J.fillet_height(0.0, W, H, k)) / e
    assert slope_wall < -50
    if k == 3.0:     # curvature-continuous at W: the second derivative vanishes there
        e2 = 1e-3
        second = (J.fillet_height(W - 2 * e2, W, H, k) - 2 * J.fillet_height(W - e2, W, H, k)
                  + J.fillet_height(W, W, H, k)) / e2 ** 2
        assert abs(second) < 5e-2 * H / W ** 2


# -- the mechanism on a synthetic body -------------------------------------------------------


def test_nothing_moves_without_an_active_spec_and_the_input_is_returned_as_is(body):
    v = body.verts
    assert J.apply_junctions(body, v, []) is v
    assert J.apply_junctions(body, v, None) is v
    assert J.apply_junctions(body, v, [tube_up(strength=0.0)]) is v
    far = tube_up(x=0.0).transformed(np.diag([1.0, 1.0, 1.0, 1.0]))
    far = far.transformed(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -40.0], [0, 0, 0, 1.0]]))
    assert J.apply_junctions(body, v, [far]) is v               # nothing within reach: untouched
    info: dict = {}
    assert J.apply_junctions(body, v, [], info) is v and info["moved"] == 0


def test_the_input_is_never_modified_and_the_shape_is_kept(body):
    v = body.verts.copy()
    out = J.apply_junctions(body, v, [tube_up(width=1.5)])
    assert np.array_equal(v, body.verts)                        # not mutated
    assert out is not v and out.shape == v.shape and out.dtype == v.dtype
    assert not np.array_equal(out, v)
    # only body cage vertices move: the rows after them (eyes, teeth, joint markers) do not
    assert np.array_equal(out[body.n:], v[body.n:])
    # and the topology is the body's own, untouched (quads are the base's)
    assert body.body_cage(out)[1] is body.body_quads


def test_a_vertex_beyond_the_fillet_width_does_not_move_and_the_law_holds_inside_it(body):
    spec = tube_up(width=1.5)
    W = spec.fillet_width
    v = body.verts
    d = displacement(body, [spec])
    dist, _ = J.tube_distance(v, np.asarray(spec.points), np.asarray(spec.radii))
    assert np.all(d[dist >= W] == 0.0)                           # exactly unchanged at >= W
    assert np.any((dist > 0) & (dist < W) & (np.linalg.norm(d, axis=1) > 1e-6))
    # the remaining distance of every moved outside vertex is W f(delta / W), within the skin's
    # curvature (the radius is 10: the climb is along the wall, so the distance is kept to O(t^2))
    out = v + d
    new_dist, _ = J.tube_distance(out, np.asarray(spec.points), np.asarray(spec.radii))
    sel = (dist > 0.05 * W) & (dist < W) & (np.linalg.norm(d, axis=1) > 0)
    want = W * J.fillet_remap(dist[sel] / W)
    assert np.abs(new_dist[sel] - want).max() < 0.06 * W
    # tangent to the skin where it ends: the displacement dies like (W - delta)^2 or faster
    # (the default height is 0.6 of the width: at 0.8 W it is below 0.05 W, at 0.9 W below 0.02 W)
    for lo, hi, cap in ((0.8, 1.0, 0.05), (0.9, 1.0, 0.02)):
        near = (dist > lo * W) & (dist < hi * W)
        assert near.any() and np.linalg.norm(d[near], axis=1).max() < cap * W, (lo, hi)


@pytest.fixture(scope="module")
def fine():
    v, q = sphere(levels=6)
    return FakeBase(v, q)


def test_the_skin_climbs_the_tube_and_ends_tangent_to_it(fine):
    body = fine
    spec = tube_up(width=2.0, height=1.0, sharpness=2.0)
    W = spec.fillet_width
    v = body.verts
    out = v + displacement(body, [spec])
    dist, c = J.tube_distance(v, np.asarray(spec.points), np.asarray(spec.radii))
    close = (dist > 0) & (dist < 0.3 * W)
    assert close.sum() > 10
    # the vertices nearest the wall have risen a good part of the way up it (outward, +Z here)
    rise = (out - v)[close][:, 2]
    assert rise.min() > 0.25 * W
    # and the skin there is tangent to the wall: its normal is close to the wall's normal,
    # where the unmoulded skin is perpendicular to it
    from minimaster.character.junction import vertex_normals
    q = body.body_quads
    def angle(pos, normals):
        wall = pos[close] - c[close]
        wall /= np.linalg.norm(wall, axis=1, keepdims=True)
        return np.degrees(np.arccos(np.clip((wall * normals[close]).sum(axis=1), -1, 1)))

    angle_old = angle(v, vertex_normals(v, q))
    angle_new = angle(out, vertex_normals(out, q))
    assert angle_old.mean() > 80 and angle_new.mean() < 55


def test_a_tube_buried_under_the_skin_leaves_it_alone(fine):
    """Only the EXPOSED part of a tube is moulded around: the skin over a tube lying a radius
    and a bit under it stays as it is, and the same tube with its axis in the skin does move it."""
    def lying(z):
        pts = [(-1.2, 0.0, z), (0.0, 0.0, z), (1.2, 0.0, z)]
        return J.JunctionSpec("lying", pts, [0.6] * 3, width=1.5, base_radius=0.6, height=1.0)

    for z in (8.8, 9.0):
        assert J.apply_junctions(fine, fine.verts, [lying(z)]) is fine.verts, z
    out = J.apply_junctions(fine, fine.verts, [lying(9.97)])
    assert np.abs(out - fine.verts).max() > 0.3
    # and the part of a tube that is buried does not lift the skin along the way out
    spec = tube_up(depth=1.0, width=1.5)
    assert J.apply_junctions(fine, fine.verts, [spec]) is not fine.verts


@pytest.mark.parametrize("pts,rad", [
    ([(0, 0, 9.0), (0, 0, 10.2), (0.8, 0, 11.4), (2.2, 0, 12.2)], [0.9, 0.75, 0.45, 0.2]),
    ([(0, 0, 9.4), (0, 0, 10.4), (1.2, 0, 10.8)], [0.9, 0.9, 0.3])])
def test_vertices_inside_the_tube_stay_inside_it(fine, pts, rad):
    """A bent or tapered tube would leave the skin that was inside it OUTSIDE once it climbed
    (28 of 29 vertices: the guard brings them back to just under the wall)."""
    spec = J.JunctionSpec("bent", pts, rad, width=2.0, base_radius=max(rad), height=1.0)
    v = fine.verts
    out = J.apply_junctions(fine, v, [spec])
    d0, _ = J.tube_distance(v, np.asarray(pts, float), np.asarray(rad))
    d1, _ = J.tube_distance(out, np.asarray(pts, float), np.asarray(rad))
    inside = d0 <= 0
    assert inside.sum() > 20 and np.all(d1[inside] <= 1e-9)


def test_the_displacement_is_capped_and_quads_never_flip_or_collapse(body):
    for width in (1.5, 3.0, 5.0):
        spec = tube_up(width=width, radius=0.9, height=1.0)
        info: dict = {}
        out = J.apply_junctions(body, body.verts, [spec], info)
        mag = np.linalg.norm(out - body.verts, axis=1)
        assert mag.max() <= spec.move_limit() + 1e-12
        q = body.body_quads
        old = J._quad_vectors(body.verts, q)
        new = J._quad_vectors(out, q)
        ol, nl = np.linalg.norm(old, axis=1), np.linalg.norm(new, axis=1)
        dot = (old * new).sum(axis=1) / (ol * nl)
        assert dot.min() >= J.MIN_NORMAL_DOT - 1e-9, (width, info)       # no quad flipped
        assert (nl / ol).min() >= J.MIN_AREA_RATIO - 1e-9, (width, info)  # none collapsed
    assert info["moved"] > 0


def test_apply_backs_off_a_needle_that_would_flip_quads(fine):
    spec = J.JunctionSpec("needle", (), (), mounds=(J.Mound((0, 0, 10.0), 0.4, 3.0),))
    q = fine.body_quads
    idx, d = J.junction_displacement(spec, fine.verts, J.vertex_normals(fine.verts, q))
    raw = np.zeros_like(fine.verts)
    raw[idx] = d
    old = J._quad_vectors(fine.verts, q)

    def worst(v):
        new = J._quad_vectors(v, q)
        return ((old * new).sum(axis=1) / (np.linalg.norm(old, axis=1) * np.linalg.norm(new, axis=1))).min()

    assert worst(fine.verts + raw) < J.MIN_NORMAL_DOT                  # the test is not vacuous
    info: dict = {}
    out = J.apply_junctions(fine, fine.verts, [spec], info)
    assert worst(out) >= J.MIN_NORMAL_DOT - 1e-9 and info["limited"] > 0 and info["rounds"] >= 1


def test_each_spec_is_capped_by_its_own_limit_not_by_the_largest(fine):
    small = tube_up(x=0.0, width=1.5, max_move=0.1)
    big = tube_up(x=6.0, width=1.5, max_move=5.0)
    out = J.apply_junctions(fine, fine.verts, [small, big])
    near_small = np.linalg.norm(fine.verts - np.array([0.0, 0.0, 10.0]), axis=1) < 2.0
    assert np.linalg.norm((out - fine.verts)[near_small], axis=1).max() <= 0.1 + 1e-12
    near_big = np.linalg.norm(fine.verts - np.array([6.0, 0.0, 8.0]), axis=1) < 2.0
    assert np.linalg.norm((out - fine.verts)[near_big], axis=1).max() > 0.3


def test_overlapping_specs_do_not_depend_on_their_order(fine):
    a, b = tube_up(x=0.0, width=1.5), tube_up(x=0.5, width=1.5)
    ab = J.apply_junctions(fine, fine.verts, [a, b])
    ba = J.apply_junctions(fine, fine.verts, [b, a])
    assert np.abs(ab - ba).max() < 1e-14
    both = np.linalg.norm(ab - fine.verts, axis=1)
    only_a = np.linalg.norm(J.apply_junctions(fine, fine.verts, [a]) - fine.verts, axis=1)
    assert (both > only_a + 1e-6).any()                                   # the second one added to it


def test_a_displacement_that_would_flip_a_quad_is_backed_off():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    q = np.array([[0, 1, 2, 3]])
    push = np.zeros((4, 3))
    push[2] = (-1.6, -1.6, 0.0)                                  # drives a corner through the quad
    out, factor, rounds = J.limit_displacement(v, q, push)
    assert rounds > 0 and factor[2] < 1.0 and np.all(out[[0, 1, 3]] == 0.0)
    new = J._quad_vectors(v + out, q)[0]
    old = J._quad_vectors(v, q)[0]
    assert new @ old > 0 and np.linalg.norm(new) >= J.MIN_AREA_RATIO * np.linalg.norm(old) - 1e-9
    # a displacement that does not hurt is left exactly alone
    ok = np.zeros((4, 3))
    ok[2] = (0.1, 0.1, 0.0)
    same, f2, r2 = J.limit_displacement(v, q, ok)
    assert r2 == 0 and np.array_equal(same, ok) and np.all(f2 == 1.0)


def test_a_mirrored_pair_gives_a_mirror_symmetric_body_and_the_order_does_not_matter(fine):
    body = fine
    left = tube_up(x=3.0, width=1.5)
    right = left.mirrored()
    out = J.apply_junctions(body, body.verts, [left, right])
    assert np.abs(out - body.verts).max() > 0.05 and (np.abs(out - body.verts).max(axis=1) > 0).sum() > 40
    # the sphere is symmetric to rounding; so must the result be (the junction adds nothing)
    n = body.n
    mirror = _mirror_map(body.verts[:n])
    flipped = out[:n][mirror] * [-1, 1, 1]
    assert np.abs(out[:n] - flipped).max() < 1e-12
    again = J.apply_junctions(body, body.verts, [right, left])
    assert np.abs(out - again).max() < 1e-14                     # a sum: no order dependence
    # a lone right spec is exactly the pair's right half
    only_right = J.apply_junctions(body, body.verts, [right])
    sel = np.arange(len(out)) < n
    sel &= body.verts[:, 0] < -1.0
    assert np.abs((only_right - out)[sel]).max() < 1e-12


def _mirror_map(v):
    key = lambda a: [tuple(k) for k in np.round(a / 1e-6).astype(np.int64)]    # noqa: E731
    table = {k: i for i, k in enumerate(key(v))}
    m = np.array([table.get(k, -1) for k in key(v * [-1, 1, 1])])
    for i in np.nonzero(m < 0)[0]:
        m[i] = int(np.argmin(np.linalg.norm(v - v[i] * [-1, 1, 1], axis=1)))
    return m


def test_it_is_deterministic_and_idempotent_as_a_function_of_the_inputs(body):
    specs = [tube_up(x=2.0, width=1.5), tube_up(x=-2.0, width=1.5)]
    a = J.apply_junctions(body, body.verts, specs)
    b = J.apply_junctions(body, body.verts, list(specs))
    assert a.tobytes() == b.tobytes()                            # same inputs, same bytes
    h = hashlib.sha1(a.tobytes()).hexdigest()
    for _ in range(3):
        assert hashlib.sha1(J.apply_junctions(body, body.verts, specs).tobytes()).hexdigest() == h
    # a function of (verts, specs) only: shaping a copy gives the same
    assert J.apply_junctions(body, body.verts.copy(), specs).tobytes() == a.tobytes()


def test_a_mound_bulges_along_the_normal_a_hollow_dents_it_and_keep_holds_the_skin(fine):
    body = fine
    mound = J.Mound((0, 0, 10.0), radius=2.0, height=0.4)
    spec = J.JunctionSpec("m", (), (), mounds=(mound,))
    d = displacement(body, [spec])
    v = body.verts
    r = np.linalg.norm(v - np.array([0, 0, 10.0]), axis=1)
    assert np.all(d[r >= 2.0] == 0.0)
    top = np.argmin(r)
    assert d[top] @ (v[top] / np.linalg.norm(v[top])) == pytest.approx(0.4 * (1 - 3 * (r[top] / 2) ** 2
                                                                         + 2 * (r[top] / 2) ** 3), rel=0.05)
    assert np.linalg.norm(d, axis=1).max() <= 0.4 + 1e-9
    hollow = J.JunctionSpec("h", (), (), mounds=(dataclasses.replace(mound, height=-0.4),))
    assert np.allclose(displacement(body, [hollow]), -d, atol=1e-12)
    held = J.JunctionSpec("k", (), (), mounds=(mound,), keep=(J.Keep((0.0, 0.0, 10.0), 1.0),))
    dk = displacement(body, [held])
    inside = r < 1.0
    assert np.all(dk[inside] == 0.0) and np.linalg.norm(dk[(r > 1.6) & (r < 2.0)], axis=1).max() > 0
    ridge = J.JunctionSpec("r", (), (), mounds=(J.Mound((-3.0, 0, 9.5), 1.0, 0.2, end=(3.0, 0, 9.5)),))
    dr = displacement(body, [ridge])
    on_line = np.abs(v[:, 1]) < 0.15
    along = dr[on_line & (np.abs(v[:, 0]) < 3) & (v[:, 2] > 9)]
    assert len(along) and np.linalg.norm(along, axis=1).min() > 0.09       # a ridge along the segment, not a bump


def test_two_overlapping_specs_are_capped_on_the_sum(body):
    a = tube_up(x=0.0, width=1.5)
    b = tube_up(x=0.4, width=1.5)
    out = J.apply_junctions(body, body.verts, [a, b])
    mag = np.linalg.norm(out - body.verts, axis=1)
    assert mag.max() <= max(a.move_limit(), b.move_limit()) + 1e-12


def test_it_is_fast_on_a_body_of_this_size(body):
    specs = [tube_up(x=3.0, width=1.5), tube_up(x=-3.0, width=1.5), tube_up(x=0.0, width=1.5)]
    t = []
    for _ in range(5):
        t0 = time.perf_counter()
        J.apply_junctions(body, body.verts, specs)
        t.append(time.perf_counter() - t0)
    assert min(t) < 0.05, t


def test_cage_is_closed_tells_a_closed_cage_from_an_open_one():
    v, q = cube_cage(1.0)
    assert cage_is_closed(q) and cage_is_closed(subdivide_quads(v, q, 2)[1])
    assert not cage_is_closed(q[:-1])
    assert not cage_is_closed(np.zeros((0, 4), dtype=int))


# -- tube specs from geometry ------------------------------------------------------------------


def test_ring_tube_reads_the_rings_of_a_loft_and_hugs_inside_the_facets():
    from minimaster.character import appendages as A
    m = A.make_tail(length=4.0, base_radius=0.5, segments=8, ring_verts=12, curl=0.0, droop=0.0,
                    tip_radius=0.1)
    c, r = J.ring_tube(m.vertices, 12, 9)
    assert c.shape == (9, 3) and np.allclose(c[:, :2], 0.0, atol=1e-9)
    assert np.all(np.diff(c[:, 2]) > 0)
    # inside the inscribed circle of the polygon times the hug, strictly decreasing along the taper
    assert r[0] == pytest.approx(0.5 * 0.92 * np.cos(np.pi / 12) * J.HUG, rel=1e-6)
    assert np.all(np.diff(r) < 0)
    assert J.rings_for_length(c, 1.1) == 4 and J.rings_for_length(c, 99.0) == 9
    assert J.rings_for_length(c, 0.0) == 3                      # never fewer than three points


def test_flare_widens_the_root_only_and_zero_changes_nothing():
    from minimaster.character import appendages as A
    plain = A.make_tail(length=4.0, segments=8, ring_verts=12)
    zero = A.make_tail(length=4.0, segments=8, ring_verts=12, flare=0.0)
    assert np.array_equal(plain.vertices, zero.vertices) and np.array_equal(plain.faces, zero.faces)
    wide = A.make_tail(length=4.0, segments=8, ring_verts=12, flare=1.0, flare_length=0.25)
    ring = lambda m, k: np.linalg.norm(m.vertices[12 * k:12 * k + 12]                   # noqa: E731
                                       - m.vertices[12 * k:12 * k + 12].mean(axis=0), axis=1).mean()
    assert ring(wide, 0) == pytest.approx(2.0 * ring(plain, 0), rel=0.02)    # 1 + flare at the base
    assert ring(wide, 1) == pytest.approx(ring(plain, 1) * A.root_flare(1 / 8, 1.0, 0.25), rel=0.02)
    assert ring(wide, 3) == pytest.approx(ring(plain, 3), rel=1e-9)        # t = 3/8 > 0.25: untouched
    assert A.root_flare(0.0, 1.0, 0.25) == 2.0 and A.root_flare(0.25, 1.0, 0.25) == 1.0
    assert A.root_flare(0.1, 0.0, 0.25) == 1.0
    assert wide.integrity_report()["watertight"] and wide.volume() > plain.volume()
    h0 = A.make_horn(length=2.0, ring_verts=8)
    assert np.array_equal(h0.vertices, A.make_horn(length=2.0, ring_verts=8, flare=0.0).vertices)
    assert A.make_horn(length=2.0, ring_verts=8, flare=0.5).volume() > h0.volume()


# -- on the real body ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def world():
    base, lib = mh.load()
    verts = lib.apply(base.verts, mh.macro_weights(gender=1.0, muscle=0.55))
    return base, lib, verts


def _placed(world, *texts, **macro):
    from minimaster.character import kinds
    from minimaster.character.sliders import Character
    base, lib, verts = world
    if macro:
        verts = lib.apply(base.verts, mh.macro_weights(**macro))
    c = Character()
    for t in texts:
        c.appendages.append(kinds.add_from_text(t, c.appendages, base=base, lib=lib))
    return c, verts, c.instantiate_appendages(base, verts, lib)


@assets
def test_each_kind_supplies_a_generic_junction_built_from_its_own_geometry(world):
    c, v, placed = _placed(world, "tail", "horn", "wing", "harvest:preset=upper_arms")
    specs = {p.name: p.junction() for p in placed}
    assert all(s is not None and s.active for s in specs.values())
    tail = specs["tail-1"]
    assert len(tail.points) >= 3 and np.all(np.diff(tail.radii) < 0)       # it tapers
    # read off the tail's own rings: the first ring is where the loft starts (the frame's origin,
    # sunk into the body) and its radius is the builder's law at t = 0 -- the minor axis of the
    # ellipse (0.92 of the base radius) times the inscribed-circle factor of 16 facets and the hug
    tail_placed = next(p for p in placed if p.name == "tail-1")
    assert np.allclose(tail.points[0], tail_placed.matrix[:3, 3], atol=1e-9)
    assert tail.radii[0] == pytest.approx(0.45 * 0.92 * np.cos(np.pi / 16) * J.HUG, rel=1e-6)
    assert tail.base_radius == pytest.approx(0.45) and tail.width == pytest.approx(1.5)
    # a tube is described from the sunk base outward: the first point is inside the body, the
    # last is clear of the skin
    base = world[0]
    cage_v, cage_q = base.body_cage(world[2])
    from minimaster.character.junction import vertex_normals
    nrm = vertex_normals(cage_v, cage_q)
    first, last = np.asarray(tail.points[0]), np.asarray(tail.points[-1])
    i, k = (int(np.argmin(np.linalg.norm(cage_v - p, axis=1))) for p in (first, last))
    assert (first - cage_v[i]) @ nrm[i] < -0.2 and (last - cage_v[k]) @ nrm[k] > 0.3
    # the right-hand twin is the exact mirror image of the left
    l, r = specs["horn-1.l"], specs["horn-1.r"]
    assert np.allclose(np.asarray(l.points) * [-1, 1, 1], np.asarray(r.points), atol=1e-9)
    assert l.radii == pytest.approx(r.radii)
    assert np.allclose(np.asarray(l.mirrored().points), np.asarray(r.points), atol=1e-9)
    assert specs["wing-1.l"].mounds and not specs["tail-1"].mounds          # the scapular mass
    arm = specs["arm-1.l"]
    assert 0.1 < min(arm.radii) and max(arm.radii) < 0.5 * 2.6              # a limb, not the torso
    # the spec comes from the geometry, not from the body: a taller body moves it, not reshapes it
    c2, v2, placed2 = _placed(world, "tail", height=1.0)
    s2 = placed2[0].junction()
    assert s2.radii == pytest.approx(tail.radii) and not np.allclose(s2.points, tail.points)


@assets
def test_a_junction_of_zero_fillet_is_none_and_the_parameter_is_in_range(world):
    from minimaster.character import kinds
    for kind in ("tail", "horn", "wing", "harvest"):
        p = kinds.KINDS[kind].param("fillet")
        assert p is not None and p.lo == 0.0 and p.default > 0
    c, v, placed = _placed(world, "tail:fillet=0", "horn:fillet=0,id=horn-9")
    assert [p.junction() for p in placed] == [None, None, None]
    from minimaster.character.junction import shaped_body
    assert shaped_body(world[0], v, placed) is v


@assets
def test_the_junction_leaves_topology_and_numbering_alone_and_moves_only_body_vertices(world):
    base, lib, verts = world
    c, v, placed = _placed(world, "tail", "horn", "wing", "harvest:preset=upper_arms")
    out = c.shaped_body(base, v, placed)
    assert out.shape == v.shape and out is not v
    moved = np.nonzero(np.abs(out - v).max(axis=1) > 0)[0]
    assert len(moved) > 200 and np.isin(moved, base.body_verts_idx).all()
    # the same quads, the same vertex count: morph targets still address the same vertices
    assert base.body_cage(out)[1].shape == (13378, 4) and len(out) == 19158
    # the joint markers, the eyes and the teeth did not move
    for name in ("joint-l-shoulder", "helper-l-eye", "helper-upper-teeth"):
        idx = np.unique(base.quads[base.groups[name]])
        assert np.array_equal(out[idx], v[idx])


@assets
def test_without_junction_specs_everything_is_bit_identical(world, tmp_path):
    from minimaster.__main__ import main
    from minimaster.character import printprep
    base, lib, verts = world
    c, v, placed = _placed(world)                                         # a body-only character
    assert c.shaped_body(base, v, placed) is v
    assert c.junction_enabled and "junction" not in c.anatomy
    # a v1 document is written exactly as before: nothing new in it
    assert "junction" not in c.to_dict() and c.to_dict()["version"] == 1
    # the STL of a body-only character is the one the unshaped code path makes
    stl = tmp_path / "body.stl"
    assert main(["character", "--gender", "1", "--muscle", "0.55", "-o", str(stl),
                 "--size", "medium"]) == 0
    from minimaster.core.stl import write_stl
    mesh, _ = printprep.assemble_character(printprep.character_print_shells(base, v), size="medium")
    ref = tmp_path / "ref.stl"
    write_stl(mesh, ref, name="character")
    assert stl.read_bytes() == ref.read_bytes()
    # and the switch off gives the unshaped body for a character WITH appendages too
    c2, v2, placed2 = _placed(world, "tail", "horn")
    c2.anatomy["junction"] = False
    assert not c2.junction_enabled and c2.shaped_body(base, v2, placed2) is v2


@assets
def test_the_junction_switch_is_saved_only_when_it_is_off(world, tmp_path):
    from minimaster.character.sliders import Character
    c, v, placed = _placed(world, "tail")
    assert "junction" not in c.to_dict()["anatomy"]
    c.anatomy["junction"] = False
    path = tmp_path / "c.mmchar"
    c.save(path)
    assert Character.load(path).anatomy["junction"] is False and not Character.load(path).junction_enabled
    assert Character.load(path).to_dict() == c.to_dict()


@assets
def test_the_skin_moulds_around_each_kind_without_a_flip_or_a_collapse(world):
    base, lib, verts = world
    for texts, macro in ((("tail",), {}), (("horn",), {}), (("wing:style=membrane",), {}),
                         (("wing:style=feathered",), {}), (("harvest:preset=upper_arms",), {}),
                         (("harvest:preset=lower_arms",), {}), (("harvest:preset=extra_legs",), {}),
                         (("tail", "horn", "wing", "harvest:preset=lower_arms"), {"age": 0.1875}),
                         (("tail", "horn", "wing", "harvest:preset=lower_arms"),
                          {"weight": 1.0, "gender": 1.0, "muscle": 1.0})):
        c, v, placed = _placed(world, *texts, **macro)
        info: dict = {}
        out = c.shaped_body(base, v, placed, info)
        assert info["moved"] > 0, texts
        cv0, q = base.body_cage(v)
        cv1, _ = base.body_cage(out)
        old, new = J._quad_vectors(cv0, q), J._quad_vectors(cv1, q)
        ol, nl = np.linalg.norm(old, axis=1), np.linalg.norm(new, axis=1)
        assert ((old * new).sum(axis=1) / (ol * nl)).min() >= J.MIN_NORMAL_DOT - 1e-9, (texts, macro)
        assert (nl / ol).min() >= J.MIN_AREA_RATIO - 1e-9, (texts, macro)
        for spec in J.junction_specs(placed):
            assert np.linalg.norm(cv1 - cv0, axis=1).max() <= max(
                s.move_limit() for s in J.junction_specs(placed)) + 1e-9


@assets
def test_a_pair_of_horns_gives_a_mirror_symmetric_skin_on_the_real_body(world):
    base, lib, verts = world
    c, v, placed = _placed(world, "horn", "wing")
    out = c.shaped_body(base, v, placed)
    from minimaster.character.appendages import mirror_vertex_map
    vm = mirror_vertex_map(base)
    cv, _ = base.body_cage(out)
    assert np.abs(cv[vm] * [-1, 1, 1] - cv).max() < 1e-12


@assets
def test_one_junction_on_a_fixed_body_is_a_pure_function(world):
    base, lib, verts = world
    c, v, placed = _placed(world, "tail", "horn")
    a = c.shaped_body(base, v, placed)
    b = c.shaped_body(base, v.copy(), placed)
    assert a.tobytes() == b.tobytes()
    # never fed back: shaping the shaped body would be a second junction on the first
    twice = c.shaped_body(base, a, placed)
    assert twice.tobytes() != a.tobytes()


@assets
def test_a_pair_and_a_tail_take_well_under_fifty_milliseconds(world):
    base, lib, verts = world
    c, v, placed = _placed(world, "tail", "horn")                         # tail + horn pair
    specs = J.junction_specs(placed)
    assert len(specs) == 3
    t = []
    for _ in range(7):
        t0 = time.perf_counter()
        J.apply_junctions(base, v, specs)
        t.append(time.perf_counter() - t0)
    assert min(t) < 0.05, t


@assets
@pytest.mark.parametrize("text", ["tail", "horn", "wing", "harvest:preset=upper_arms",
                                  "harvest:preset=lower_arms"])
def test_the_skin_meets_the_root_at_a_smaller_angle_and_without_new_creases(world, text):
    base, lib, verts = world
    c, v, placed = _placed(world, text)
    spec = placed[0].junction()
    out = c.shaped_body(base, v, placed)
    m = J.measure_junction(base, v, out, spec, levels=2)
    assert m["flipped"] == 0 and m["min_area_ratio"] >= J.MIN_AREA_RATIO - 1e-9
    # the skin ran into the root at 56-92 degrees from tangent; now it meets it at 15-30 less
    assert m["join_angle_deg_after"] < m["join_angle_deg_before"] - 15, m
    # no crease the body did not already have: the 99th percentile of what the junction adds to
    # the crease of an edge, on the twice-subdivided skin, and the 95th percentile of the zone
    assert m["dihedral_increase_p99_deg"] < 14.0, m
    assert m["p95_dihedral_deg_after"] < max(14.0, m["p95_dihedral_deg_before"] + 6.0), m


@assets
def test_the_subdivided_shaped_body_is_smooth_where_the_cage_is_coarse(world):
    base, lib, verts = world
    c, v, placed = _placed(world, "horn")
    out = c.shaped_body(base, v, placed)
    spec = placed[0].junction()
    p95 = []
    for levels in (0, 1, 2):
        m = J.measure_junction(base, v, out, spec, levels=levels)
        p95.append(m["p95_dihedral_deg_after"])
    assert p95[2] < p95[1] < p95[0], p95                       # each Catmull-Clark level rounds it
    assert p95[2] < 12.0


# -- the pipeline ----------------------------------------------------------------------------


@assets
def test_the_cli_png_and_stl_use_the_shaped_body(world, tmp_path, monkeypatch):
    """The picture, the STL and the report all see the shaped skin; the appendages were placed on
    the unshaped one. Counting the calls proves the CLI asks for it; the STL bytes prove the
    shaped body is what is printed."""
    from minimaster.__main__ import main
    from minimaster.core.stl import read_stl
    a, b = tmp_path / "a.stl", tmp_path / "b.stl"
    common = ["character", "--gender", "1", "--muscle", "0.55", "--add", "tail", "--add", "horn",
              "--size", "medium"]
    assert main(common + ["-o", str(a)]) == 0
    assert main(common + ["--no-junction", "-o", str(b)]) == 0
    assert a.read_bytes() != b.read_bytes()
    ma, mb = read_stl(a), read_stl(b)
    assert len(ma.faces) == len(mb.faces)                      # the same topology, other positions
    # (the figure is re-centred on its body, so compare the bodies from the shared pipeline)
    from minimaster.character import printprep
    base, lib, verts = world
    c, v, placed = _placed(world, "tail", "horn")
    shaped = c.shaped_body(base, v, placed)
    body_a = printprep.character_print_shells(base, shaped)[0][1]
    body_b = printprep.character_print_shells(base, v)[0][1]
    moved = np.abs(body_a.vertices - body_b.vertices).max(axis=1)
    assert 50 < (moved > 1e-3).sum() < len(moved) / 4
    # the PNG path
    pa, pb = tmp_path / "a.png", tmp_path / "b.png"
    assert main(common[:-2] + ["--azimuth", "200", "--res", "200", "300", "-o", str(pa)]) == 0
    assert main(common[:-2] + ["--azimuth", "200", "--res", "200", "300", "--no-junction",
                               "-o", str(pb)]) == 0
    assert pa.read_bytes() != pb.read_bytes()
    from minimaster.character import junction
    calls = []
    real = junction.apply_junctions
    monkeypatch.setattr(junction, "apply_junctions",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    assert main(common + ["-o", str(tmp_path / "c.stl")]) == 0
    assert len(calls) == 1
    calls.clear()
    assert main(common + ["--no-junction", "-o", str(tmp_path / "d.stl")]) == 0
    assert calls == []


@assets
def test_cli_saves_and_restores_the_junction_switch(tmp_path):
    from minimaster.__main__ import main
    from minimaster.character.sliders import Character
    mm = tmp_path / "c.mmchar"
    assert main(["character", "--add", "tail", "--no-junction", "--save-character", str(mm),
                 "-o", str(tmp_path / "x.png"), "--res", "100", "150"]) == 0
    assert Character.load(mm).anatomy.get("junction") is False
    assert main(["character", "--character", str(mm), "--junction", "--save-character", str(mm),
                 "-o", str(tmp_path / "x.png"), "--res", "100", "150"]) == 0
    assert "junction" not in Character.load(mm).anatomy


@assets
def test_a_tail_that_only_touches_the_skin_is_welded_by_the_junction(tmp_path):
    """With offset 0 a tail is tangent to the skin: a hairline weld and a blocked export
    (``--no-junction``: tests/test_character_cli.py). The skin that flows up around its root
    encloses part of it, so the weld is real."""
    from minimaster.__main__ import main
    out = tmp_path / "o.stl"
    assert main(["character", "--add", "tail:offset=0", "-o", str(out), "--size", "medium"]) == 0


@assets
def test_the_figure_stays_one_connected_component_and_the_report_has_no_new_block(world):
    from minimaster.character import lab as L
    res = L.Resources.shared()
    for adds in (("tail",), ("horn",), ("wing",), ("wing:style=feathered",),
                 ("harvest:preset=upper_arms",), ("harvest:preset=lower_arms",),
                 ("tail", "horn", "wing")):
        for junction in (False, True):
            doc = L.LabDocument(res)
            doc.set_anatomy(junction=junction)
            for t in adds:
                kind, _, rest = t.partition(":")
                params = dict(kv.split("=") for kv in rest.split(",")) if rest else {}
                preset = params.pop("preset", None)
                doc.add_appendage(kind, preset=preset, params=params)
            r = doc.engine.export(doc.character, None, size="medium", force=True)
            rep = r.printability
            assert rep is not None and [f for f in rep.blocking] == [], (adds, junction, rep.as_text())
            assert all(g.components == 1 for g in rep.groups.values()), (adds, junction)


@assets
def test_the_lab_engine_draws_prints_and_picks_the_shaped_skin(tmp_path):
    from minimaster.character import lab as L
    from minimaster.character import portrait
    res = L.Resources.shared()
    doc = L.LabDocument(res)
    doc.add_appendage("tail")
    eng = doc.engine
    body = eng.body_state(doc.character)
    # the scene's body shell is the shaped skin; the unshaped one is what the appendage was placed on
    scene = eng.final_scene(doc.character)
    plain = portrait.character_shells(res.base, body.verts, eye_color="#5b7c8d")[0].mesh
    assert scene.shells[0].mesh.vertices.shape == plain.vertices.shape
    assert np.abs(scene.shells[0].mesh.vertices - plain.vertices).max() > 0.05
    # the placement did not chase the displacement: the anchor is the same with the junction off
    a1 = eng.anchor_points(doc.character.appendages[0], body)
    doc2 = L.LabDocument(res)
    doc2.add_appendage("tail")
    doc2.set_anatomy(junction=False)
    scene2 = doc2.engine.final_scene(doc2.character)
    assert np.abs(scene2.shells[0].mesh.vertices - plain.vertices).max() == 0.0
    a2 = doc2.engine.anchor_points(doc2.character.appendages[0], doc2.engine.body_state(doc2.character))
    assert all(np.array_equal(p, q) for (_, p), (_, q) in zip(a1, a2))
    # the draft body is the shaped skin too (cluster-averaged)
    draft = eng.draft_scene(doc.character)
    assert draft.body is not None and draft.body.key != body.key
    assert not np.array_equal(draft.shells[0].mesh.vertices,
                              res.body_proxy.mesh(body.verts).vertices)
    # the cache: the second request does no work, a pose drag does not reshape the skin
    n = eng.stats["junction"]
    eng.final_scene(doc.character)
    eng.draft_scene(doc.character)
    assert eng.stats["junction"] == n


@assets
def test_the_draft_of_a_drag_does_not_reshape_the_skin_on_every_frame():
    """A pose drag keeps the cached skin of the rest pose (the final render, which follows the pose,
    reshapes it once); a body-slider or shape drag draws the unshaped skin (the final has the
    fillet). Reshaping 5 specs is 10 ms a frame and the 4k-triangle proxy averages a fillet away."""
    from minimaster.character import lab as L
    res = L.Resources.shared()
    doc = L.LabDocument(res)
    doc.add_appendage("wing")
    doc.add_appendage("tail")
    eng = doc.engine
    shaped = eng.draft_scene(doc.character)
    body = eng.body_state(doc.character)
    assert shaped.body.key != body.key                       # the exact draft has the fillet
    doc.set_control("wing-1", "fold", 0.1)
    eng.draft_scene(doc.character, "pose", "wing-1")
    n = eng.stats["junction"]        # (the pose draft poses one wing and mirrors it: one new skin)
    for k in range(1, 5):
        doc.set_control("wing-1", "fold", 0.2 * k)
        eng.draft_scene(doc.character, "pose", "wing-1")
    assert eng.stats["junction"] == n                        # then the rest-pose skin, cached
    doc.set_control("wing-1", "fold", 0.8)
    eng.final_scene(doc.character)
    assert eng.stats["junction"] == n + 1                    # the final follows the pose, once
    doc.set_param("tail-1", "length", 7.0)
    drag = eng.draft_scene(doc.character, "shape", "tail-1")
    assert drag.body.key == eng.body_state(doc.character).key and eng.stats["junction"] == n + 1
    doc.set_macro("height", 0.6)
    drag = eng.draft_scene(doc.character, "body")
    assert drag.body.key == eng.body_state(doc.character).key and eng.stats["junction"] == n + 1


@assets
def test_the_lab_pick_sees_the_skin_the_user_sees():
    """Click-to-place reads the id buffer of the body as it is DRAWN: the shaped skin. (The quad
    ids are topological, so a click still stores a quad of the undeformed body.)"""
    from minimaster.character import lab as L
    from minimaster.character.portrait import MH_TO_Z_UP
    res = L.Resources.shared()
    doc = L.LabDocument(res)
    doc.add_appendage("tail")
    eng = doc.engine
    body = eng.body_state(doc.character)
    scene = eng.draft_scene(doc.character)
    cv0, q = res.base.body_cage(body.verts)
    cv1, _ = res.base.body_cage(scene.body.verts)
    cent0, cent1 = cv0[q].mean(axis=1), cv1[q].mean(axis=1)
    move = np.linalg.norm(cent1 - cent0, axis=1)
    order = np.argsort(-move)[:80]
    view = L.ViewState(azimuth=200, elevation=10, center=tuple(MH_TO_Z_UP[:3, :3] @ cent1[order[0]]),
                       radius=1.5)
    size = (400, 400)
    sx, sy, _ = view.project(cent1[order] @ MH_TO_Z_UP[:3, :3].T, size)
    found = None
    for k, quad in enumerate(order):
        pick = eng.pick(doc.character, view, size, sx[k], sy[k], body_only=True)
        if pick is not None and pick.kind == "body" and pick.quad == quad:
            found = (quad, pick, sx[k], sy[k])
            break
    assert found is not None, "no displaced quad could be picked"
    quad, pick, x, y = found
    assert np.dot(pick.weights, cv1[q[pick.quad]]) == pytest.approx(pick.point, abs=1e-6)
    assert move[quad] > 0.05
    # with the junction off the same pixel shows another surface
    doc2 = L.LabDocument(res)
    doc2.add_appendage("tail")
    doc2.set_anatomy(junction=False)
    other = doc2.engine.pick(doc2.character, view, size, x, y, body_only=True)
    assert other is None or np.linalg.norm(other.point - pick.point) > 0.02 or other.quad != quad


@assets
def test_the_lab_export_equals_the_cli_stl_with_the_junction(tmp_path):
    from minimaster.__main__ import main
    from minimaster.character import lab as L
    res = L.Resources.shared()
    doc = L.LabDocument(res)
    doc.add_appendage("tail", params={"length": 7})
    doc.add_appendage("horn", side="left")
    doc.add_appendage("harvest", preset="upper_arms")
    mm = doc.save(tmp_path / "c.mmchar")
    lab_stl, cli_stl = tmp_path / "lab.stl", tmp_path / "cli.stl"
    doc.engine.export(doc.character, lab_stl, size="small", force=True)
    assert main(["character", "--character", str(mm), "-o", str(cli_stl), "--size", "small",
                 "--force"]) == 0
    assert lab_stl.read_bytes() == cli_stl.read_bytes()
    doc.set_anatomy(junction=False)
    mm2 = doc.save(tmp_path / "c2.mmchar")
    doc.engine.export(doc.character, tmp_path / "lab2.stl", size="small", force=True)
    assert main(["character", "--character", str(mm2), "-o", str(tmp_path / "cli2.stl"),
                 "--size", "small", "--force"]) == 0
    assert (tmp_path / "lab2.stl").read_bytes() == (tmp_path / "cli2.stl").read_bytes()
    assert (tmp_path / "lab2.stl").read_bytes() != lab_stl.read_bytes()


@assets
def test_subdivided_prints_work_and_the_open_teeth_are_left_out(world, tmp_path):
    """``minimaster character --subdivide 1 -o x.stl`` died with "cage is not a closed
    2-manifold: 38 edges" even without appendages: the two teeth islands are open (38 boundary
    edges) and Catmull-Clark of an open cage raises. At level 0 they were dropped by the
    watertight check after building; now they are left out before subdividing."""
    from minimaster.__main__ import main
    from minimaster.character import printprep
    base, lib, verts = world
    for levels in (0, 1, 2):
        shells = printprep.character_print_shells(base, verts, levels=levels)
        assert [n for n, _ in shells] == ["body", "l-eye", "r-eye"], levels
        assert all(m.integrity_report()["watertight"] for _, m in shells)
    assert len(printprep.character_print_shells(base, verts, levels=1)[0][1].faces) == 4 * 2 * 13378
    out = tmp_path / "sub.stl"
    assert main(["character", "--subdivide", "1", "-o", str(out), "--size", "small"]) == 0
    assert out.stat().st_size > 1_000_000
    out2 = tmp_path / "sub2.stl"
    assert main(["character", "--subdivide", "1", "--add", "tail", "--add", "horn", "-o", str(out2),
                 "--size", "medium"]) == 0
    assert out2.stat().st_size > 1_000_000
