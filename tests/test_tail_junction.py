"""How the skin of the sacrum moulds around a tail root (``tailrig.junction_spec``).

The generic junction of the first version tented the skin around a pipe and folded the two walls of the
gluteal cleft through each other (12 new self-intersecting pairs, 9 of them outside the tail). The tail's own
spec is the anatomy of the root: the skin tube is the tail's flared skin, a low sacral rise runs up the back
(the erector spinae funnel into the dorsal tail), the medial cheeks are drawn onto the tail's flanks, and the
cleft below the root is KEPT (beads of Keep spheres down the midline): the tail leaves a crease above the
perineum, the cleft is not filled.
"""

import numpy as np
import pytest

from minimaster.character import junction as J
from minimaster.character import kinds as K
from minimaster.character import mhbase as mh
from minimaster.character import tailrig as T
from minimaster.character.sliders import Character, SliderCatalog

assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")

BODIES = {
    "neutral": dict(gender=1.0, muscle=0.55), "tall": dict(height=1.0, gender=1.0), "short": dict(height=0.0),
    "child": dict(age=0.1875), "heavy": dict(weight=1.0, gender=1.0, muscle=1.0), "female": dict(gender=0.0),
}
CLEFT_LOW = (4241, 4250, 4259)        # the cleft and the perineum, below the apex (4223): the junction leaves them


@pytest.fixture(scope="module")
def world():
    base, lib = mh.load()
    return base, lib, SliderCatalog(lib.names)


def _build(world, name, text="tail", controls=None):
    base, lib, cat = world
    c = Character()
    c.macro.update(BODIES[name])
    c.appendages.append(K.add_from_text(text, c.appendages, base=base, lib=lib))
    if controls:
        c.appendages[0].controls = dict(controls)
    verts = lib.apply(base.verts, c.effective_target_weights(cat))
    placed = c.instantiate_appendages(base, verts, lib)
    shaped = c.shaped_body(base, verts, placed)
    return c, verts, placed, shaped


def _R(p):
    return p.params["base_radius"] * p.model.scale


# -- the spec ---------------------------------------------------------------------------------------------


@assets
def test_the_spec_is_the_tails_root_anatomy(world):
    c, v, (p,), _ = _build(world, "neutral")
    spec = p.junction()
    R = _R(p)
    assert spec.active and spec.width == pytest.approx(K.TAIL_FILLET) and spec.height == pytest.approx(K.TAIL_FILLET_HEIGHT)
    assert spec.base_radius == pytest.approx(R)
    # the tube: the skin's flared section from the buried end to 2 R beyond the skin point
    pts = np.asarray(spec.points)
    reach = np.linalg.norm(pts[-1] - p.anchor)
    assert 1.5 * R <= reach <= 2.1 * R and spec.radii[0] == pytest.approx(R * 1.35 * np.cos(np.pi / 16) * J.HUG)
    # the sacral rise is a ridge up the back (a mound with an end), the cheeks two mirrored mounds drawn inward
    ridge, left, right = spec.mounds
    assert ridge.end is not None and np.linalg.norm(np.asarray(ridge.end) - np.asarray(ridge.center)) == pytest.approx(
        (T.MOUND_UP[1] - T.MOUND_UP[0]) * R)
    assert ridge.height == pytest.approx(T.MOUND_HEIGHT * R) and ridge.radius == pytest.approx(T.MOUND_RADIUS * R)
    assert left.mirrored().center == pytest.approx(right.center, abs=1e-4)      # (the base mesh itself is 5e-5 asymmetric)
    assert np.dot(left.direction, np.asarray(left.center) - np.asarray(p.anchor)) < 0      # toward the axis
    assert np.dot(right.direction, np.asarray(right.center) - np.asarray(p.anchor)) < 0
    # the cleft beads run down the midline, below the root
    assert len(spec.keep) == len(T.KEEP_BEADS) >= 3
    for k in spec.keep:
        assert abs(k.center[0]) < 1e-4 and k.center[1] < p.anchor[1]
    ys = [k.center[1] for k in spec.keep]
    assert ys == sorted(ys, reverse=True)
    # the whole spec is symmetric about the sagittal plane: it is its own mirror image
    m = spec.mirrored()
    assert np.allclose(m.points, spec.points, atol=1e-4)
    assert np.allclose(sorted(k.center for k in m.keep), sorted(k.center for k in spec.keep), atol=1e-4)
    assert np.allclose(sorted(mo.center for mo in m.mounds), sorted(mo.center for mo in spec.mounds), atol=1e-4)


@assets
def test_a_pose_moves_the_free_tail_in_the_spec_not_the_root(world):
    _, _, (rest,), _ = _build(world, "neutral")
    _, _, (up,), _ = _build(world, "neutral", controls={"lift": 60.0})
    a, b = rest.junction(), up.junction()
    assert np.allclose(a.points[:5], b.points[:5], atol=1e-9)            # the buried stem and the skin point
    assert not np.allclose(a.points[-1], b.points[-1], atol=1e-3)
    assert a.mounds == b.mounds and a.keep == b.keep                      # the body's masses stay


# -- what it does to the skin ------------------------------------------------------------------------------


@assets
@pytest.mark.parametrize("name", list(BODIES))
def test_the_displacement_is_local_bounded_and_symmetric_and_the_cleft_stays(world, name):
    base, lib, cat = world
    c, v, (p,), shaped = _build(world, name)
    cv0, cq = base.body_cage(v)
    cv1, _ = base.body_cage(shaped)
    d = cv1 - cv0
    mag = np.linalg.norm(d, axis=1)
    R = _R(p)
    spec = p.junction()
    assert mag.max() <= spec.move_limit() + 1e-12 and mag.max() <= 2.0 * R           # ... a bounded fillet (the collar's height is H = W = 1.5 R)
    moved = np.nonzero(mag > 0)[0]
    assert 40 < len(moved) < 260                                                        # a root's worth of vertices
    # zero outside the root's neighbourhood: within 3.5 R sideways and from -3.5 R to +5 R along the back
    rel = cv0[moved] - p.anchor
    assert np.abs(rel[:, 0]).max() <= 3.5 * R + 0.3 and rel[:, 1].min() >= -3.5 * R and rel[:, 1].max() <= 5.0 * R
    # the cleft and the perineum below the apex are untouched, and the midline vertices stay on the plane
    for i in CLEFT_LOW:
        assert mag[i] <= 0.02 * R, (name, i, mag[i] / R)
    on_plane = np.nonzero(cv0[:, 0] == 0.0)[0]
    assert (np.abs(cv1[on_plane, 0]) < 1e-3 * R).all()      # (the spec is as symmetric as the base mesh: 5e-5)
    # mirror symmetry of the displacement (a symmetric spec on a symmetric body)
    vmap = np.array(__import__("minimaster.character.appendages", fromlist=["x"]).mirror_vertex_map(base))
    nb = len(cv0)
    mirrored = d[vmap[:nb]] * np.array([-1.0, 1.0, 1.0])
    assert np.abs(mirrored - d).max() < 0.01 * mag.max() + 1e-3         # within 1% (the base itself is 5e-5 asymmetric)
    # no quad flips or collapses (the validity pass) and the fillet really lifts the skin onto the root
    info = J.measure_junction(base, v, shaped, spec, levels=1)
    assert info["flipped"] == 0 and info["min_area_ratio"] >= J.MIN_AREA_RATIO
    assert info["join_angle_deg_after"] < info["join_angle_deg_before"] - 10.0       # (heavy: -15 degrees; the others -20 to -40)


def _seg_tri(P0, P1, A, B, C):
    e1, e2, D = B - A, C - A, P1 - P0
    h = np.cross(D, e2)
    a = (e1 * h).sum(1)
    ok = np.abs(a) > 1e-14
    f = np.where(ok, 1.0 / np.where(ok, a, 1), 0)
    s = P0 - A
    u = f * (s * h).sum(1)
    qv = np.cross(s, e1)
    v = f * (D * qv).sum(1)
    t = f * (e2 * qv).sum(1)
    e = 1e-9
    return ok & (u > e) & (v > e) & (u + v < 1 - e) & (t > e) & (t < 1 - e)


def self_intersecting_pairs(verts, quads, region):
    """Pairs of non-adjacent triangles of the cage that cross, among those with a corner in ``region``."""
    tri = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    P = verts[tri]
    lo, hi = P.min(1), P.max(1)
    pairs = set()
    for i in np.nonzero(region[tri].any(1))[0]:
        ov = np.nonzero(np.all(lo <= hi[i] + 1e-9, 1) & np.all(hi >= lo[i] - 1e-9, 1))[0]
        ov = ov[ov != i]
        ov = ov[~(tri[ov][:, :, None] == tri[i][None, None, :]).any((1, 2))]
        if not len(ov):
            continue
        A, B, C = P[i]
        for X, Y in ((A, B), (B, C), (C, A)):
            h = _seg_tri(np.broadcast_to(X, (len(ov), 3)), np.broadcast_to(Y, (len(ov), 3)), P[ov, 0], P[ov, 1], P[ov, 2])
            pairs.update((min(i, j), max(i, j)) for j in ov[h])
        for k in range(3):
            h = _seg_tri(P[ov, k], P[ov, (k + 1) % 3], *(np.broadcast_to(q, (len(ov), 3)) for q in (A, B, C)))
            pairs.update((min(i, j), max(i, j)) for j in ov[h])
    return pairs, tri


def new_pairs_outside_the_tail(base, v, shaped, p):
    """``(below, above)``: cage triangles that cross each other after the junction and did not before, with
    both outside the tail's own skin (a pair inside the tail is hidden by the union and harmless), counted
    below the root's centre (the cleft side, local x < 0) and above it (the dorsal side)."""
    cv0, cq = base.body_cage(v)
    cv1, _ = base.body_cage(shaped)
    R = _R(p)
    region = np.linalg.norm(cv1 - p.anchor, axis=1) < 4.0 * R
    before, _ = self_intersecting_pairs(cv0, cq, region)
    after, tri = self_intersecting_pairs(cv1, cq, region)
    new = after - before
    if not new:
        return 0, 0
    from test_tail_anatomy import inside as _inside
    cent = np.array([[cv1[tri[i]].mean(axis=0) for i in pr] for pr in new])          # (pairs, 2, 3)
    out = ~_inside(cent.reshape(-1, 3), p.mesh()).reshape(-1, 2)
    up = np.asarray(p.matrix[:3, 0]) / p.model.scale
    x = ((cent - p.anchor) @ up)                                                      # along the back
    bad = out.all(axis=1)
    return int((bad & (x.max(axis=1) < 0)).sum()), int((bad & (x.max(axis=1) >= 0)).sum())


@assets
@pytest.mark.parametrize("name", ["neutral", "child", "heavy", "short"])
def test_the_gluteal_cleft_is_not_folded_through_itself(world, name):
    """The first generic junction added 12 self-intersecting pairs in the cleft (9 outside the tail) on the
    neutral body, 30-64 after subdivision. The cleft beads keep the two walls apart: none below the root's
    centre outside the tail; above it (the dorsal furrow beside the tail's wall) at most a few tiny ones on the
    most extreme body (the heavy one has 3)."""
    base, _, _ = world
    c, v, (p,), shaped = _build(world, name)
    below, above = new_pairs_outside_the_tail(base, v, shaped, p)
    assert below == 0 and above <= 4, (name, below, above)


@assets
def test_without_the_cleft_beads_the_walls_do_cross(world, monkeypatch):
    """Fail-first for the test above: with the keep spheres shrunk to nothing the child's cleft is folded."""
    base, _, _ = world
    monkeypatch.setattr(T, "KEEP_RADIUS", 1e-3)
    c, v, (p,), shaped = _build(world, "child")
    assert new_pairs_outside_the_tail(base, v, shaped, p)[0] > 0


@assets
def test_the_sacral_rise_is_a_mound_up_the_back_and_not_down_the_cleft(world):
    base, _, _ = world
    c, v, (p,), _ = _build(world, "neutral")
    spec = p.junction()
    only = spec.with_(points=(), radii=(), keep=(), mounds=(spec.mounds[0],))
    out = J.apply_junctions(base, v, [only])
    cv0, _q = base.body_cage(v)
    cv1, _ = base.body_cage(out)
    d = cv1 - cv0
    R = _R(p)
    up = np.asarray(p.matrix[:3, 0]) / p.model.scale                      # the tail frame's up
    along = (cv0 - p.anchor) @ up
    nrm = J.vertex_normals(cv0, base.body_cage(v)[1])
    lifted = np.nonzero((d * nrm).sum(1) > 0.05 * R)[0]
    assert len(lifted) >= 4
    assert (along[lifted] > -1.5 * R).all() and (along[lifted] < 4.2 * R).all()     # the ridge: from the root up the back
    assert np.linalg.norm(d, axis=1).max() == pytest.approx(T.MOUND_HEIGHT * R, rel=0.15)
