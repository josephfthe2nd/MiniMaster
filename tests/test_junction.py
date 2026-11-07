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


# the S1b guards (smoothing, stretch / Lipschitz / gap limits, crossing check) off: the LAW of a
# spec alone, which the first two tests below pin
LAW_ONLY = dict(max_lipschitz=None, max_area_growth=None, max_edge_growth=None,
                intersections=False, smooth=0, tight_gap=None, min_area_ratio=0.0, min_normal_dot=-1.0)


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
    d = displacement(body, [spec], **LAW_ONLY)       # the law itself: the guards come next
    dist, _ = J.tube_distance(v, np.asarray(spec.points), np.asarray(spec.radii))
    assert np.all(d[dist >= W] == 0.0)                           # exactly unchanged at >= W
    # the guarded result also leaves everything beyond W alone, and moves less than the law
    dg = displacement(body, [spec])
    assert np.all(dg[dist >= W] == 0.0) and np.linalg.norm(dg, axis=1).max() <= np.linalg.norm(
        d, axis=1).max() + 1e-12
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
    out = v + displacement(body, [spec], **LAW_ONLY)             # the law itself (guards: below)
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
    # with the guards on (no edge stretches past 2x, neighbours differ by <= 1.25 edge lengths)
    # the skin still climbs and is still more tangent to the wall than it was: a vertical
    # wall is as steep as a 0.15 edge can follow a 1.2 rise, so less than the law
    guarded = v + displacement(body, [spec])
    assert (guarded - v)[close][:, 2].min() > 0.08 * W
    assert angle(guarded, vertex_normals(guarded, q)).mean() < angle_old.mean() - 15


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
    # S1: 8 ms. S1b adds the crossing test, the smoothing and the facing-gap limits on three specs:
    # ~70-100 ms cold (the lab caches the result by body and specs, and a drag draws the unshaped skin)
    assert min(t) < 0.15, t


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
    from minimaster.character import kinds as kinds_mod
    c, v, placed = _placed(world, "tail", "horn", "wing", "harvest:preset=upper_arms")
    specs = {p.name: p.junction() for p in placed}
    assert all(s is not None and s.active for s in specs.values())
    tail = specs["tail-1"]
    assert len(tail.points) >= 3 and np.all(np.diff(tail.radii) <= 1e-12)   # it tapers (the stem is constant)
    assert tail.radii[-1] < tail.radii[0]
    # read off the tail's own anatomy (tailrig.py): the chain starts at the buried end of the skin shell
    # (the hidden stem, 1.75 R below the skin point, on the tail's own axis) and its radius is the skin's
    # at the root: the flared semi-axis (1 + F = 1.35 R) times the inscribed-circle factor of 16 facets and
    # the hug
    tail_placed = next(p for p in placed if p.name == "tail-1")
    stem = np.array([0.0, 0.0, -1.75 * 0.45, 1.0])      # in the tail's local frame (the matrix scales it)
    assert np.allclose(tail.points[0], (tail_placed.matrix @ stem)[:3], atol=1e-9)
    k0 = tail_placed.model.scale              # the body's height over the neutral body's: the tail follows it
    assert tail.radii[0] == pytest.approx(0.45 * 1.35 * np.cos(np.pi / 16) * J.HUG * k0, rel=1e-6)
    assert tail.base_radius == pytest.approx(0.45 * k0) and tail.width == pytest.approx(kinds_mod.TAIL_FILLET)
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
    assert specs["wing-1.l"].mounds and specs["tail-1"].mounds and specs["tail-1"].keep   # scapular mass; sacral rise, cleft
    arm = specs["arm-1.l"]
    assert 0.1 < min(arm.radii) and max(arm.radii) < 0.5 * 2.6              # a limb, not the torso
    # the spec comes from the geometry, not from the body: a taller body moves it, and a tail is sized by the
    # body (``TailKind.body_scale``: the height over the neutral body's), so it scales it, not reshapes it
    c2, v2, placed2 = _placed(world, "tail", height=1.0)
    s2 = placed2[0].junction()
    k = placed2[0].model.scale / k0
    assert k > 1.3 and s2.radii == pytest.approx(np.asarray(tail.radii) * k)
    assert not np.allclose(s2.points, tail.points)


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
def test_a_pair_and_a_tail_take_well_under_150_milliseconds(world):
    base, lib, verts = world
    c, v, placed = _placed(world, "tail", "horn")                         # tail + horn pair
    specs = J.junction_specs(placed)
    assert len(specs) == 3
    t = []
    for _ in range(7):
        t0 = time.perf_counter()
        J.apply_junctions(base, v, specs)
        t.append(time.perf_counter() - t0)
    # S1: 8 ms. S1b adds the crossing test, the smoothing and the facing-gap limits on three specs:
    # ~70-100 ms cold (the lab caches the result by body and specs, and a drag draws the unshaped skin)
    assert min(t) < 0.15, t


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
    # the skin ran into the root at 56-92 degrees from tangent; now it meets it at 15-30 less. The
    # tail's S1 figure (27 degrees) was bought with 2.5-3.4x stretched edges and 12 new crossings
    # of the cleft's two cheeks; inside the 2x cap and without crossings the cage gives ~9 there
    # (the tail sits in the cleft; its anatomy is another track's).
    need = {"tail": 5.0}.get(text, 15.0)
    assert m["join_angle_deg_after"] < m["join_angle_deg_before"] - need, m
    # no crease the body did not already have: the 99th percentile of what the junction adds to
    # the crease of an edge, on the twice-subdivided skin (S1: 14), and the 95th percentile of the zone
    assert m["dihedral_increase_p99_deg"] < 10.0, m
    # ... and the WORST edge (S1: it could rise 60-95 degrees where the body already has a crease:
    # the cleft under the tail, the armpit under the wing arm)
    assert m["dihedral_increase_max_deg"] < {"tail": 50.0}.get(text, 30.0), m   # the cleft: 43 here
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
    """A tail slid out of its stem (offset +1.0) is tangent to the skin: a hairline weld and a blocked
    export (``--no-junction``: tests/test_character_cli.py). The skin that flows up around its root
    encloses part of it, so the weld is real."""
    from minimaster.__main__ import main
    out = tmp_path / "o.stl"
    assert main(["character", "--add", "tail:offset=1.0", "-o", str(out), "--size", "medium"]) == 0


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


# ==========================================================================================
# S1b: the core made robust (validity guards, ownership, body-relative sizes, quality helper)
# ==========================================================================================

from minimaster.character import bodyframe as BF  # noqa: E402
from minimaster.character import junction_checks as JC  # noqa: E402


def unit_quad():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    return v, np.array([[0, 1, 2, 3]])


def test_a_quad_that_would_grow_past_the_cap_is_backed_off():
    v, q = unit_quad()
    push = np.zeros((4, 3))
    push[2] = (1.5, 1.5, 0.0)                                  # the far corner: the area becomes 3.25x
    off, _, r0 = J.limit_displacement(v, q, push)              # S1: no growth guard
    assert r0 == 0 and np.array_equal(off, push)
    out, factor, rounds = J.limit_displacement(v, q, push, max_area_growth=2.0)
    new = np.linalg.norm(J._quad_vectors(v + out, q)[0])
    assert rounds > 0 and factor[2] < 1.0
    assert new <= 2.0 * np.linalg.norm(J._quad_vectors(v, q)[0]) + 1e-9
    assert new > 1.7 * np.linalg.norm(J._quad_vectors(v, q)[0])          # ... as little as it takes


def test_an_edge_that_would_stretch_past_the_cap_is_backed_off():
    v, q = unit_quad()
    push = np.zeros((4, 3))
    push[1] = push[2] = (2.0, 0.0, 0.0)           # one edge becomes 3x, the quad 3x
    # only the edge guard, so it is the one that is tested
    out, factor, _ = J.limit_displacement(v, q, push, max_edge_growth=2.0)
    new = v + out
    assert np.linalg.norm(new[1] - new[0]) <= 2.0 * 1.0 + 1e-9 and factor[1] < 1.0
    free, _, r = J.limit_displacement(v, q, push)
    assert r == 0 and np.linalg.norm((v + free)[1] - (v + free)[0]) == pytest.approx(3.0)


def test_neighbours_may_not_move_more_than_the_lipschitz_bound_apart():
    v = np.array([[0, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0], [0, 0.1, 0]], dtype=float)
    q = np.array([[0, 1, 2, 3]])
    push = np.zeros((4, 3))
    push[1] = push[2] = (0.0, 0.0, 0.3)               # 3 edge lengths apart from the others
    out, factor, _ = J.limit_displacement(v, q, push, max_lipschitz=1.25)
    d = np.linalg.norm(out[1] - out[0]) / 0.1
    assert d <= 1.25 + 1e-6 and factor[1] < 1.0 and d > 1.0        # as much as it allows
    free, _, r = J.limit_displacement(v, q, push)
    assert r == 0 and np.linalg.norm(free[1] - free[0]) / 0.1 == pytest.approx(3.0)


def two_cubes(gap, levels=2):
    """Two rounded closed cages facing each other across ``gap`` (a cleft): ``(base, n)``, the
    first ``n`` vertices are cube A (its +X face looks at cube B's -X face)."""
    v, q = cube_cage(2.0)
    v, q = subdivide_quads(v, q, levels)
    v2 = v + np.array([v[:, 0].max() - v[:, 0].min() + gap, 0.0, 0.0])
    return FakeBase(np.vstack([v, v2]), np.vstack([q, q + len(v)])), len(v)


def test_the_skin_is_not_pushed_through_the_opposite_skin():
    # two plain boxes (6 big quads each): the +X face of A pushed 0.6 into B, which stands 0.3 away
    # and a little off to the side, so the edges of A cross the interior of B's -X face
    v, q = cube_cage(2.0)
    v2 = v + np.array([2.3, 0.3, 0.3])
    base2 = FakeBase(np.vstack([v, v2]), np.vstack([q, q + 8]))
    v = base2.verts
    face = np.nonzero(v[:8, 0] > 0.5)[0]
    push = np.zeros_like(v)
    push[face] = (0.6, 0.0, 0.0)
    q = base2.body_quads
    moved = np.nonzero(np.linalg.norm(push, axis=1) > 0)[0]
    assert len(JC.self_intersection_pairs(v + push, q, moved)) > 0        # the test is not vacuous
    assert len(JC.self_intersection_pairs(v, q, moved)) == 0
    # the cage guard
    out, factor, _ = J.limit_displacement(v, q, push, intersections=True)
    assert len(JC.self_intersection_pairs(v + out, q, moved)) == 0
    assert 0 < factor[face].mean() < 1.0                          # backed off, not wiped out
    # a pair that is hidden inside the appendage's own tube is allowed (the slicer's union)
    everywhere = lambda pts: np.ones(len(pts), dtype=bool)  # noqa: E731
    same, f2, _ = J.limit_displacement(v, q, push, intersections=True, benign=everywhere)
    assert np.array_equal(same, push) and np.all(f2 == 1.0)


def test_skin_next_to_skin_it_already_touches_is_left_alone_and_fades_back_in():
    base2, n = two_cubes(0.02, levels=3)                          # a closed cleft: 0.02 apart
    v = base2.verts
    # a mound that wants to lift cube A's +X face by 0.1 along X (toward the other cube)
    xa = v[:n, 0].max()
    spec = J.JunctionSpec("m", (), (), mounds=(J.Mound((xa, 0.0, 0.0), 0.8, 0.1,
                                                       direction=(1.0, 0.0, 0.0)),))
    free = J.apply_junctions(base2, v, [spec], tight_gap=None, intersections=False) - v
    kept = J.apply_junctions(base2, v, [spec], intersections=False) - v
    mag_free, mag_kept = np.linalg.norm(free, axis=1), np.linalg.norm(kept, axis=1)
    assert mag_free.max() > 0.05                                  # not vacuous: it moves the skin
    facing = (v[:n, 0] > xa - 0.05) & (np.abs(v[:n, 1]) < 0.15) & (np.abs(v[:n, 2]) < 0.15)
    assert mag_free[:n][facing].min() > 0.05 and mag_kept[:n][facing].max() < 1e-12   # touching: held
    assert mag_kept.max() > 0.0              # and it fades back in: the skin further on still moves


def test_the_smoothing_spreads_a_step_and_keeps_what_is_pinned_and_what_never_moved():
    v, q = sphere(levels=4)
    d = np.zeros_like(v)
    near = np.nonzero(np.linalg.norm(v - np.array([0, 0, 10.0]), axis=1) < 1.5)[0]
    core = near[np.linalg.norm(v[near] - np.array([0, 0, 10.0]), axis=1) < 0.4]
    d[near] = (0, 0, 0.2)
    d[core] = (0, 0, 1.0)                                          # a spike on a plateau
    pinned = np.zeros(len(v), dtype=bool)
    pinned[core] = True
    out = J.diffuse_displacement(v, q, d, pinned, 15)
    assert np.array_equal(out[core], d[core])                      # pinned: as given
    rim = np.setdiff1d(np.arange(len(v)), near)
    assert np.all(out[rim] == 0.0)                                 # never moved: still not
    ref = JC.stretch_report(v, v + d, q)["lipschitz_max"]
    assert JC.stretch_report(v, v + out, q)["lipschitz_max"] < ref  # a smoother field
    assert np.array_equal(J.diffuse_displacement(v, q, d, pinned, 0), d)


def test_a_tube_lying_in_the_skin_does_not_move_the_skin_straight_above_it():
    """``sps``: over a tube the skin normal IS the wall's normal and nothing climbs (S1's
    mutant that dropped the factor lifted a ridge over the whole length of a lying tube)."""
    pts = [(-1.5, 0.0, 9.97), (0.0, 0.0, 9.97), (1.5, 0.0, 9.97)]
    spec = J.JunctionSpec("lying", pts, [0.6] * 3, width=1.5, base_radius=0.6, height=1.0)
    v, q = sphere(levels=6)
    idx, d = J.junction_displacement(spec, v, J.vertex_normals(v, q))
    full = np.zeros_like(v)
    full[idx] = d
    above = (np.abs(v[:, 1]) < 0.05) & (np.abs(v[:, 0]) < 0.3) & (v[:, 2] > 9.9)
    beside = (np.abs(v[:, 1]) > 0.7) & (np.abs(v[:, 1]) < 1.0) & (np.abs(v[:, 0]) < 1.0)
    assert above.sum() >= 3 and np.linalg.norm(full[above], axis=1).max() < 0.04    # without sps: ~0.9
    assert np.linalg.norm(full[beside], axis=1).max() > 0.1


class _Frame:
    """A stand-in for a BodyFrame: ``size`` of a part."""
    def size(self, scale, near=None):
        return {"head": 1.0, "trunk": 2.0, "arm": 0.5, "leg": 0.7, "local": 3.0}[scale]


def test_a_relative_mound_is_resolved_from_the_body_and_stays_absolute_without_one():
    m = J.Mound((0, 0, 10.0), 0.5, 0.1, scale="trunk")
    spec = J.JunctionSpec("w", (), (), mounds=(m,))
    assert J.resolve_spec(spec, None) is spec                     # no skeleton (a synthetic body): as written
    r = J.resolve_spec(spec, _Frame()).mounds[0]
    assert (r.radius, r.height, r.scale) == (1.0, 0.2, None)
    assert J.resolve_spec(J.JunctionSpec("a", (), (), mounds=(J.Mound((0, 0, 10.0), 0.5, 0.1),)),
                          _Frame()).mounds[0].radius == 0.5       # an absolute mound is left alone
    with pytest.raises(ValueError, match="scale"):
        J.Mound((0, 0, 0), 1.0, 0.1, scale="elbow")
    # serialised only when set (the digest of a spec without it is the one it always had), and
    # a relative size is a ratio: a similarity transform moves the centre, not the fractions
    assert "scale" not in J.Mound((0, 0, 0), 1.0, 0.1).to_dict()
    assert J.Mound.from_dict(m.to_dict()) == m and J.JunctionSpec.from_dict(spec.to_dict()) == spec
    big = np.diag([2.0, 2.0, 2.0, 1.0])
    t = m.transformed(big)
    assert (t.radius, t.height, t.scale) == (0.5, 0.1, "trunk")
    plain = J.JunctionSpec("p", [(0, 0, 0), (0, 0, 1)], [0.1, 0.1])
    assert "owner" not in plain.to_dict() and plain.digest() == J.JunctionSpec.from_dict(
        plain.to_dict()).digest()
    owned = plain.with_(owner="trunk")
    assert owned.to_dict()["owner"] == "trunk" and J.JunctionSpec.from_dict(owned.to_dict()) == owned
    assert owned.mirrored().owner == "trunk" and owned.transformed(big).owner == "trunk"
    assert owned.digest() != plain.digest()
    with pytest.raises(ValueError, match="owner"):
        plain.with_(owner="")


def test_the_tube_is_fitted_to_the_mesh_it_stands_for_and_never_fatter():
    """A tube thicker than the real mesh leaves the skin that 'is inside it' sticking out of a
    thin root: the radii are capped by the radius of the mesh around the centreline."""
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    zs = np.linspace(0.0, 4.0, 9)
    verts = np.array([(0.3 * np.cos(a), 0.3 * np.sin(a), z) for z in zs for a in ang])
    pts = [(0, 0, 0.5), (0, 0, 1.5), (0, 0, 2.5)]
    fat = J.fit_radii_to_mesh(pts, [0.5, 0.5, 0.5], verts)
    assert all(r <= 0.5 for r in fat)
    assert fat == pytest.approx([0.3 * J.HUG] * 3, rel=0.02)            # the mesh's, hugged
    thin = J.fit_radii_to_mesh(pts, [0.1, 0.1, 0.1], verts)
    assert thin == pytest.approx([0.1] * 3)                             # never FATTER than given
    off = J.fit_radii_to_mesh([(5, 5, 5), (5, 5, 6)], [0.4, 0.2], verts)  # no mesh anywhere near
    assert off == pytest.approx([0.4, 0.2])


def test_facing_pairs_finds_the_two_sides_of_a_cleft_and_not_neighbours():
    base2, n = two_cubes(0.1)
    v, q = base2.verts, base2.body_quads
    nrm = J.vertex_normals(v, q)
    a = np.nonzero(v[:n, 0] > v[:n, 0].max() - 0.01)[0]
    vi, ui, dist = J.facing_pairs(v, q, nrm, a, 0.2)
    assert len(vi) > 0 and np.all(vi < n) and np.all(ui >= n) and dist.min() == pytest.approx(0.1, abs=1e-6)
    none = J.facing_pairs(v, q, nrm, a, 0.05)[0]
    assert len(none) == 0                                          # out of reach
    assert not JC.share_a_quad(q, vi, ui).any()


def test_ring_distance_counts_the_rings_around_a_mask():
    v, q = sphere(levels=3)
    mask = np.zeros(len(v), dtype=bool)
    mask[0] = True
    rd = JC.ring_distance(q, mask, 3)
    assert rd[0] == 0 and (rd == 1).sum() >= 3 and rd.max() == 4 and (rd == 2).sum() > (rd == 1).sum()


# -- on the real body: the junction is a VALID skin --------------------------------------------

KINDS6 = ("tail", "horn", "wing:style=membrane", "wing:style=feathered", "harvest:preset=upper_arms",
          "harvest:preset=lower_arms", "harvest:preset=extra_legs")
BODIES5 = {"neutral": {"gender": 1.0, "muscle": 0.55}, "tall": {"gender": 1.0, "muscle": 0.55, "height": 1.0},
           "short": {"gender": 1.0, "muscle": 0.55, "height": 0.0}, "child": {"age": 0.1875},
           "heavy": {"weight": 1.0, "gender": 1.0, "muscle": 1.0}}


def _shaped(world, texts, macro=None, poses=(), **kw):
    """``(character, verts, placed, shaped verts, info)`` for appendage texts on a body."""
    from minimaster.character import kinds
    from minimaster.character.sliders import Character
    base, lib, verts = world
    if macro:
        verts = lib.apply(base.verts, mh.macro_weights(**macro))
    c = Character()
    for t in ([texts] if isinstance(texts, str) else texts):
        c.appendages.append(kinds.add_from_text(t, c.appendages, base=base, lib=lib))
    for p in poses:
        kinds.apply_pose(*kinds.pose_from_text(p, c.appendages))
    placed = c.instantiate_appendages(base, verts, lib)
    info: dict = {}
    out = J.apply_junctions(base, verts, J.junction_specs(placed), info, **kw)
    return c, verts, placed, out, info


@assets
@pytest.mark.parametrize("body", sorted(BODIES5))
@pytest.mark.parametrize("text", KINDS6)
def test_the_junction_is_a_valid_skin_on_every_kind_and_body(world, text, body):
    """The reviewer's count: S1 stretched cage quads up to 2.5-6.5x (edges 2-5x, neighbours
    3-4 edge lengths apart) and made 12-52 new pairs of crossing triangles in the cleft and the
    armpit (30-64 on the subdivided print body). Every bound of ``quality_violations`` holds
    for every kind on five bodies, on the cage and on the body subdivided once."""
    base, lib, verts = world
    c, v, placed, out, info = _shaped(world, text, BODIES5[body])
    q = J.junction_quality(base, v, out, J.junction_specs(placed), levels=(1,))
    assert J.quality_violations(q) == [], (text, body, q)
    assert q["moved"] == 0 or q["max_move"] > 0.02
    assert q["max_move"] <= max(s.move_limit() for s in J.junction_specs(placed)) + 1e-9


@assets
@pytest.mark.parametrize("body", ["neutral", "child", "heavy"])
def test_the_tail_and_the_wing_do_not_cross_the_skin_on_the_twice_subdivided_body(world, body):
    base, lib, verts = world
    for text in ("tail", "wing"):
        c, v, placed, out, info = _shaped(world, text, BODIES5[body])
        q = J.junction_quality(base, v, out, J.junction_specs(placed), levels=(1, 2))
        assert J.quality_violations(q) == [] and q["L2_selfx_new_visible"] == 0, (text, body)


@assets
def test_quality_violations_names_what_is_broken():
    good = {"flipped": 0, "area_growth_max": 1.5, "area_ratio_min": 0.5, "edge_growth_max": 1.4,
            "lipschitz_max": 0.9, "selfx_new_visible": 0, "L1_selfx_new_visible": 0}
    assert J.quality_violations(good) == []
    for key, bad in (("flipped", 2), ("area_growth_max", 3.0), ("area_ratio_min", 0.05),
                     ("edge_growth_max", 2.5), ("lipschitz_max", 2.0), ("L1_selfx_new_visible", 4)):
        msgs = J.quality_violations({**good, key: bad})
        assert len(msgs) == 1, (key, msgs)
    # and the helper measures a real stretch: S1's raw displacement on the real tail is flagged
    world = _real_world()
    base, lib, verts = world
    c, v, placed, _, _ = _shaped(world, "tail", BODIES5["child"])
    raw = J.apply_junctions(base, v, J.junction_specs(placed), **LAW_ONLY)
    q = J.junction_quality(base, v, raw, J.junction_specs(placed), levels=(1,))
    assert J.quality_violations(q), q            # the unguarded law stretches and crosses the cleft


def _real_world():
    base, lib = mh.load()
    return base, lib, lib.apply(base.verts, mh.macro_weights(gender=1.0, muscle=0.55))


@assets
def test_a_wing_pose_does_not_puff_the_back_into_lobes_and_a_child_is_not_a_neutral_body(world):
    """S1's body-independent scapular mound (1.74 units wide, 0.22 high) made two pillows on the
    back beside a FOLDED wing and boxy shoulders on a child. It is now a fraction of the body's
    own trunk: nothing on the back moves by more than a few millimetres, folded or not."""
    base, lib, verts = world
    results = {}
    for label, macro, poses in (("neutral", BODIES5["neutral"], ()), ("child", BODIES5["child"], ()),
                                ("folded", BODIES5["neutral"], ("wing-1:preset=folded",)),
                                ("swung", BODIES5["neutral"], ("wing-1:swing=30",))):
        c, v, placed, out, info = _shaped(world, "wing", macro, poses)
        q = J.junction_quality(base, v, out, J.junction_specs(placed), levels=(1,))
        assert J.quality_violations(q) == [], label
        assert q["max_move"] < 0.35 and q["moved"] > 0, (label, q["max_move"])
        # beyond the root's fillet only the mound moves the skin (S1: up to 0.22 on the whole back)
        cv0, cq = base.body_cage(v)
        cv1, _ = base.body_cage(out)
        far = np.ones(len(cv0), dtype=bool)
        for sp in J.junction_specs(placed):
            far &= J.tube_distance(cv0, np.asarray(sp.points), np.asarray(sp.radii))[0] > sp.fillet_width
        assert np.linalg.norm(cv1 - cv0, axis=1)[far].max() < 0.12, label
        results[label] = (q["max_move"], J.junction_specs(placed)[0], BF.frame_or_none(
            base, lib.apply(base.verts, mh.macro_weights(**macro))))
    for label in ("neutral", "folded"):
        # the mound the spec carries is relative; resolved on the body it is a scapula, not a back
        m = J.resolve_spec(results[label][1], results[label][2]).mounds[0]
        assert m.radius < 0.9 and m.height < 0.12, (label, m)
    rn = J.resolve_spec(results["neutral"][1], results["neutral"][2]).mounds[0]
    rc = J.resolve_spec(results["child"][1], results["child"][2]).mounds[0]
    assert rc.radius < 0.8 * rn.radius and rc.height < 0.8 * rn.height     # the child's is smaller
    assert results["child"][0] < results["neutral"][0] + 0.02


@assets
def test_a_swung_arm_does_not_make_spikes_on_the_chest(world):
    """The default lower arms swung 50 degrees across the front: the S1 core tented the chest
    with two conical teats (the spec's tube was fatter than the arm's own mesh and the buried
    root tented the skin). The tube now follows the mesh and only the exposed part moulds."""
    base, lib, verts = world
    c, v, placed, out, info = _shaped(world, "harvest:preset=lower_arms", BODIES5["neutral"],
                                      ("arm-1:swing=50",))
    specs = J.junction_specs(placed)
    q = J.junction_quality(base, v, out, specs, levels=(1,))
    assert J.quality_violations(q) == [], q
    # the cone is a stretch signature: the unguarded law (S1's) grows a quad 3x and makes
    # neighbours move 2 edge lengths apart there
    raw = J.apply_junctions(base, v, specs, **LAW_ONLY)
    bad = J.quality_violations(J.junction_quality(base, v, raw, specs, levels=()))
    assert any("quad grew" in m for m in bad) and any("edge lengths apart" in m for m in bad), bad
    cv0, cq = base.body_cage(v)
    cv1, _ = base.body_cage(out)
    nrm = J.vertex_normals(cv0, cq)
    chest = (cv0[:, 2] > 0.8) & (cv0[:, 1] > 3.0) & (cv0[:, 1] < 5.0) & (np.abs(cv0[:, 0]) < 2.0)
    out_of_skin = ((cv1 - cv0) * nrm).sum(axis=1)[chest]
    assert out_of_skin.max() < 0.35 and info["moved"] < 400, (out_of_skin.max(), info)
    # the tube follows the mesh: no radius is fatter than the arm it stands for
    for p in placed:
        spec = p.junction()
        measured = J.fit_radii_to_mesh(spec.points, [10.0] * len(spec.radii), p.mesh().vertices)
        assert np.all(np.asarray(spec.radii) <= 1.15 * np.asarray(measured)), (spec.radii, measured)


@assets
@pytest.mark.parametrize("text", ["horn:flare=2", "horn:length=8,base_radius=1.0", "horn:flare=1.5,base_radius=0.8"])
def test_an_oversized_horn_root_does_not_swell_the_head(world, text):
    """S1: ``flare=2`` moved 2,803 vertices (1,424 backed off by the limiter) and swelled the
    temples; a root fatter than the head it grows from has no skin to flow into, so it moulds
    nothing; a big but sane one moulds the head locally and boundedly."""
    base, lib, verts = world
    c, v, placed, out, info = _shaped(world, text, BODIES5["neutral"])
    q = J.junction_quality(base, v, out, J.junction_specs(placed), levels=(1,))
    assert J.quality_violations(q) == []
    head_part = q.get("by_part", {}).get("head", (0, 0.0))
    assert q["moved"] <= 700 and q["max_move"] <= 0.3, (text, q["moved"], q["max_move"])
    assert head_part[0] == q["moved"]                  # only the head's own skin
    if text != "horn:flare=1.5,base_radius=0.8":
        assert q["moved"] == 0                         # thicker than the head: nothing to mould


@assets
@pytest.mark.parametrize("text,away", [("harvest:preset=extra_legs", ("arm_l", "arm_r", "head")),
                                       ("harvest:preset=upper_arms", ("arm_l", "arm_r", "head")),
                                       ("harvest:preset=lower_arms", ("arm_l", "arm_r", "head", "leg_l", "leg_r"))])
def test_an_extra_limb_does_not_mould_the_skin_of_the_original_arm_or_the_head(world, text, away):
    """S1: the extra legs dragged the ORIGINAL arm's armpit into a web (1,094 vertices moved) and
    the extra upper arms raised a ridge along its trapezius: foreign skin is left alone."""
    base, lib, verts = world
    c, v, placed, out, info = _shaped(world, text, BODIES5["neutral"])
    q = J.junction_quality(base, v, out, J.junction_specs(placed), levels=())
    assert q["moved"] > 50
    parts = q["by_part"]
    assert parts["trunk"][0] > 0.8 * q["moved"]
    for name in away:
        assert parts.get(name, (0, 0.0))[1] <= 0.05, (name, parts)           # a hair at the rim, no more


@assets
def test_the_owner_of_each_kind_and_the_skeleton_that_tells_parts_apart(world):
    base, lib, verts = world
    for text, owner in (("tail", "trunk"), ("horn", "head"), ("wing", "trunk"),
                        ("harvest:preset=extra_legs", "trunk")):
        c, v, placed, out, info = _shaped(world, text)
        assert {s.owner for s in J.junction_specs(placed)} == {owner}, text
    frame = BF.frame_or_none(base, verts)
    assert frame is not None and BF.frame_or_none(base, verts) is frame          # cached
    assert {p: frame.part_size(p) for p in frame.parts}["head"] > 0.5
    cage, _ = base.body_cage(verts)
    scores = frame.part_scores(cage)
    owners = np.array(frame.parts)[scores.argmin(axis=1)]
    # the map is anatomical: the head is the top of the figure, the legs the bottom, arms are lateral
    assert cage[owners == "head"][:, 1].min() > 5.5 and cage[owners == "leg_l"][:, 1].max() < 1.0
    assert np.all(np.abs(cage[owners == "arm_l"][:, 0]) > 1.5) and np.all(cage[owners == "arm_l"][:, 0] > 0)
    assert np.all(cage[owners == "arm_r"][:, 0] < 0) and np.all(cage[owners == "leg_r"][:, 0] < 0)
    # a size is a body dimension: a child's trunk is smaller than an adult's
    child = BF.frame_or_none(base, lib.apply(base.verts, mh.macro_weights(age=0.1875)))
    assert child.size("trunk") < 0.8 * frame.size("trunk") and frame.size("local", (0, 3.0, 0)) == frame.size("trunk")
    w = frame.owner_weights(np.array([[0.0, 3.0, 0.0], [4.0, 3.0, 0.5]]), frame.parts.index("trunk"))
    assert w[0] > 0.99 and w[1] < 0.01             # the spine's skin is the trunk's, a hand's is not
    assert frame.resolve_owner("arm", [3.0, 3.0, 0.0]) == frame.parts.index("arm_l")
    with pytest.raises(ValueError):
        frame.resolve_owner("tail", [0, 0, 0])


@assets
def test_the_harvest_tube_tracks_the_mesh_it_stands_for_and_the_tail_reaches_past_its_fillet(world):
    base, lib, verts = world
    c, v, placed, out, info = _shaped(world, "harvest:preset=lower_arms", BODIES5["neutral"])
    p = placed[0]
    spec = p.junction()
    # the elbow's radius the fillet is sized by is the mesh's own, not a model of a neutral arm
    mesh = p.mesh().vertices
    el = np.asarray(spec.points[len(spec.points) - 3])
    t = np.asarray(spec.points[-2]) - np.asarray(spec.points[-4])
    t = t / np.linalg.norm(t)
    rel = mesh - el
    slab = np.abs(rel @ t) < 0.15
    perp = np.linalg.norm(rel - (rel @ t)[:, None] * t, axis=1)[slab]
    assert slab.sum() > 10 and 0.7 * np.median(perp) < spec.base_radius < 1.3 * np.median(perp)
    # the tail's tube runs from the sunk base to 1.5 sink + 1.5 fillet widths (not three rings)
    ct, vt, pt, ot, it = _shaped(world, "tail", BODIES5["neutral"])
    tail = pt[0]
    ts = tail.junction()
    reach = 1.5 * tail.root_depth + 1.5 * ts.fillet_width
    length = float(np.linalg.norm(np.diff(np.asarray(ts.points), axis=0), axis=1).sum())
    assert length >= 0.8 * reach, (length, reach)


@assets
def test_the_junction_of_an_appendage_the_core_cannot_place_is_skipped_not_fatal(world):
    base, lib, verts = world
    c, v, placed, out, info = _shaped(world, "tail")
    assert out is not v and info["moved"] > 0
    out2 = J.apply_junctions(base, v, [J.junction_specs(placed)[0].with_(strength=0.0)])
    assert out2 is v                                           # nothing active: the input itself


@assets
def test_the_lab_export_measures_the_root_weld_against_the_body_that_prints(tmp_path, monkeypatch):
    """The report assumes the body is the one that is printed: the skin moulded around the tail.
    (Before S1b nothing pinned the lab path; the CLI's was pinned.)"""
    from minimaster.character import lab as L
    from minimaster.character import print_report
    res = L.Resources.shared()
    doc = L.LabDocument(res)
    doc.add_appendage("tail")
    seen = {}
    real = print_report.appendage_report

    def spy(*a, **kw):
        seen["body_verts"], seen["verts"] = kw.get("body_verts"), a[2]
        return real(*a, **kw)

    monkeypatch.setattr(print_report, "appendage_report", spy)
    doc.engine.export(doc.character, tmp_path / "o.stl", size="small", force=True)
    body = doc.engine.body_state(doc.character)
    assert seen["verts"] is body.verts and seen["body_verts"] is not None
    assert np.abs(seen["body_verts"] - body.verts).max() > 0.05      # the shaped skin, not the placed-on one


@assets
def test_the_draft_of_a_pose_drag_shapes_both_sides_of_a_mirrored_pair():
    """A pose drag builds only the left wing and mirrors it: the skin must have the right-hand
    fillet too (the half-built path adds ``spec.mirrored()``)."""
    from minimaster.character import lab as L
    from minimaster.character.appendages import mirror_vertex_map
    res = L.Resources.shared()
    doc = L.LabDocument(res)
    doc.add_appendage("wing")
    eng = doc.engine
    draft = eng.draft_scene(doc.character, "pose", "wing-1")
    body = eng.body_state(doc.character)
    cv0, cq = res.base.body_cage(body.verts)
    cv1, _ = res.base.body_cage(draft.body.verts)
    d = cv1 - cv0
    vm = mirror_vertex_map(res.base)
    moved = np.linalg.norm(d, axis=1) > 0.01
    assert (moved & (cv0[:, 0] > 0)).sum() > 50 and (moved & (cv0[:, 0] < 0)).sum() > 50
    assert np.abs(d[vm] * [-1, 1, 1] - d).max() < 1e-9


def test_the_skin_never_climbs_higher_than_the_tube_stands_above_it():
    """A root that grazes the skin from inside (axis 0.3 under it, a radius of 0.6: it stands 0.3
    above the skin) is a hair high, not a spike of the fillet's height (1.5): the swung arms of S1
    tented the chest with cones where a buried root passed under it."""
    pts = [(-1.5, 0.0, 9.7), (0.0, 0.0, 9.7), (1.5, 0.0, 9.7)]
    spec = J.JunctionSpec("graze", pts, [0.6] * 3, width=2.5, base_radius=0.6, height=1.0)
    v, q = sphere(levels=6)
    idx, d = J.junction_displacement(spec, v, J.vertex_normals(v, q))
    assert len(idx) > 20
    assert np.linalg.norm(d, axis=1).max() < 0.4                 # without `stand`: ~1.1
    # the same tube standing a full radius out of the skin moulds it by the fillet's height
    tall = J.JunctionSpec("tall", [(0.0, 0.0, 9.4), (0.0, 0.0, 10.6), (0.0, 0.0, 11.8)], [0.6] * 3,
                          width=2.5, base_radius=0.6, height=1.0)
    idx, d = J.junction_displacement(tall, v, J.vertex_normals(v, q))
    assert np.linalg.norm(d, axis=1).max() > 1.0


@assets
def test_the_default_call_smooths_the_displacement_over_the_cage(world):
    """``smooth`` is on by default: the raw displacement is defined through skin normals that jump
    in a crease, and the smoothed field has the smaller neighbour differences (S1: 3-4 edge lengths)."""
    import inspect
    assert inspect.signature(J.apply_junctions).parameters["smooth"].default == J.DIFFUSE_ITERATIONS > 0
    base, lib, verts = world
    c, v, placed, _, _ = _shaped(world, "wing", BODIES5["neutral"])
    specs = J.junction_specs(placed)
    rough = J.apply_junctions(base, v, specs, **{**LAW_ONLY, "smooth": 0})
    smooth = J.apply_junctions(base, v, specs, **{**LAW_ONLY, "smooth": J.DIFFUSE_ITERATIONS})
    default = J.apply_junctions(base, v, specs, max_lipschitz=None, max_area_growth=None,
                                max_edge_growth=None, intersections=False, tight_gap=None)
    cv0, cq = base.body_cage(v)
    lip = [JC.stretch_report(cv0, base.body_cage(o)[0], cq)["lipschitz_max"] for o in (rough, smooth)]
    assert lip[1] < 0.8 * lip[0], lip                          # smoother
    assert np.array_equal(default, smooth)                      # and that is what the default does


@assets
def test_the_fillet_is_no_wider_than_the_part_of_the_body_it_lies_on(world):
    """``WIDTH_CAP``: a root 0.7 thick on a head 0.86 across would raise a fillet as wide as the
    root over the temple; it is capped at 0.6 radii of the head."""
    base, lib, verts = world
    c, v, placed, out, info = _shaped(world, "horn:base_radius=0.7", BODIES5["neutral"])
    frame = BF.frame_or_none(base, v)
    cap = J.WIDTH_CAP * frame.part_size("head")
    cv0, _ = base.body_cage(v)
    assert info["moved"] > 20
    for sp in J.junction_specs(placed):
        assert sp.fillet_width > cap + 0.05                      # the spec asks for more than the cap
        alone = J.apply_junctions(base, v, [sp])                 # this horn on its own
        moved = np.linalg.norm(base.body_cage(alone)[0] - cv0, axis=1) > 0
        reach = J.tube_distance(cv0, np.asarray(sp.points), np.asarray(sp.radii))[0]
        assert moved.sum() > 10 and reach[moved].max() <= cap + 1e-6, (reach[moved].max(), cap)
