"""Where a tail is rooted: the MIDLINE of the sacrum, on the sagittal plane.

The first tail was bound to one cage quad (3921) on the side of a buttock: its origin was 0.36 R off the
midline and the normal of that one quad is yawed ~10 degrees off the sagittal plane, so the whole tail came
out of the side of the left buttock like a pipe. A midline anchor binds the quad AND its mirror image
(``appendages.face_anchor(midline=True)``): origin and normal lie exactly on the plane x = 0 on a symmetric
body. These tests pin that, the landmark rule the preset follows (vertex ids verified on the real mesh over
several morphs), the migration of documents written by the first tail, and what must NOT change (a tail
clicked onto the side of the body, a pair of tails, every other kind's anchor).
"""

import math

import numpy as np
import pytest

from minimaster.character import appendages as A
from minimaster.character import kinds as K
from minimaster.character import mhbase as mh
from minimaster.character.sliders import Character

assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")

BODIES = {
    "neutral": {}, "male": dict(gender=1.0, muscle=0.55), "tall": dict(height=1.0, gender=1.0),
    "short": dict(height=0.0), "child": dict(age=0.1875), "heavy": dict(weight=1.0, gender=1.0, muscle=1.0),
    "female": dict(gender=0.0), "old": dict(age=1.0),
}
# the research's landmarks (vertex ids of the cage == global ids): the midline chain and the apex
APEX, EDGE_A, EDGE_B, CLEFT_TOP = 4223, 4215, 4205, 4387


@pytest.fixture(scope="module")
def world():
    base, lib = mh.load()
    return base, lib


def _verts(world, **macro):
    base, lib = world
    return lib.apply(base.verts, mh.macro_weights(**macro))


def _height(cage_v):
    return float(np.ptp(cage_v[:, 1]))


# -- the landmark rule -----------------------------------------------------------------------------


@assets
@pytest.mark.parametrize("name", list(BODIES))
def test_the_midline_landmarks_hold_on_every_morph(world, name):
    """Vertex ids 4223 (apex), 4215-4205 (the edge the tail roots on) and 4387 (the cleft top) are exactly
    on x = 0 on every morph, and the root point P (the middle of that edge) sits where the research put it,
    in units of R = 0.0265 H (H the height of the morphed cage): the apex 0.7-1.15 R below it, the cleft
    top 0.8-1.4 R above it, at 0.50-0.585 of the body's height."""
    base, _ = world
    cv, _q = base.body_cage(_verts(world, **BODIES[name]))
    H = _height(cv)
    R = 0.0265 * H
    for i in (APEX, EDGE_A, EDGE_B, CLEFT_TOP):
        assert cv[i][0] == 0.0, (name, i)
    P = 0.5 * (cv[EDGE_A] + cv[EDGE_B])
    arc = np.linalg.norm(cv[EDGE_A] - cv[APEX]) + 0.5 * np.linalg.norm(cv[EDGE_B] - cv[EDGE_A])
    assert 0.7 <= arc / R <= 1.15, arc / R
    assert 0.8 <= np.linalg.norm(cv[CLEFT_TOP] - P) / R <= 1.4
    assert 0.50 <= (P[1] - cv[:, 1].min()) / H <= 0.585
    assert cv[CLEFT_TOP][1] > P[1] > cv[APEX][1]


@assets
def test_the_preset_is_the_plus_x_quad_of_the_midline_edge_and_a_midline_quad(world):
    base, lib = world
    f = K.PRESET_FACES["tail_base"]
    bv, bq = base.body_cage(None)
    assert sorted(int(c) for c in bq[f] if c in (EDGE_A, EDGE_B)) == sorted([EDGE_A, EDGE_B])
    assert A.is_midline_face(base, f)
    mf = int(A.mirror_face_map(base)[f])
    assert mf != f and A.is_midline_face(base, mf)
    assert bv[bq[f]].mean(axis=0)[0] > 0 > bv[bq[mf]].mean(axis=0)[0]     # the stored quad is the +x one
    assert K.nearest_face(K.neutral_cage(base, lib), K.PRESET_POINTS["tail_base"]) == f
    # the side quads are not midline quads
    for side in (3921, 7954, 10296, 10647):
        assert not A.is_midline_face(base, side)


# -- the anchor itself -----------------------------------------------------------------------------


@assets
@pytest.mark.parametrize("name", ["neutral", "tall", "child", "heavy", "female"])
def test_a_midline_anchor_is_exactly_on_the_sagittal_plane_and_a_single_quad_is_not(world, name):
    base, _ = world
    v = _verts(world, **BODIES[name])
    cv, cq = base.body_cage(v)
    f = K.PRESET_FACES["tail_base"]
    hint = np.array([0.0, 1.0, 0.0])
    mid = A.face_anchor(base, cq, f, None, hint, midline=True)
    origin, basis = mid.resolve(cv)
    assert origin[0] == 0.0 and basis[0, 2] == 0.0 and basis[0, 0] == 0.0       # origin, normal, up-tangent
    # the lateral axis is +x, the normal points out of the back
    assert basis[0, 1] == pytest.approx(1.0) and basis[2, 2] < 0
    # ... the first preset's single quad (3921) was off the plane in both its origin and its normal
    # (this is the bug: the whole tail came out of the side of one buttock)
    R = 0.0265 * _height(cv)
    o1, b1 = A.face_anchor(base, cq, K.LEGACY_TAIL_BASE_FACE, None, hint).resolve(cv)
    assert abs(o1[0]) > 0.2 * R and math.degrees(math.asin(abs(b1[0, 2]))) > 5.0
    # and even the new preset's quad alone has a normal that is yawed off the plane
    _o2, b2 = A.face_anchor(base, cq, f, None, hint, midline=False).resolve(cv)
    assert math.degrees(math.asin(abs(b2[0, 2]))) > 5.0


@assets
def test_weights_slide_the_midline_anchor_along_the_spine_and_it_stays_on_the_plane(world):
    base, _ = world
    cv, cq = base.body_cage(_verts(world))
    f = K.PRESET_FACES["tail_base"]
    corners = [int(c) for c in cq[f]]
    hint = np.array([0.0, 1.0, 0.0])
    ys = []
    for w_up in (0.0, 0.5, 1.0):
        # weights toward the edge's upper vertex (the cleft top side) or the lower (the apex side)
        w = [0.0] * 4
        w[corners.index(EDGE_A)], w[corners.index(EDGE_B)] = (1.0 - w_up), w_up
        o, _b = A.face_anchor(base, cq, f, w, hint, midline=True).resolve(cv)
        assert o[0] == 0.0
        ys.append(o[1])
    assert ys[0] < ys[1] < ys[2] or ys[0] > ys[1] > ys[2]
    # weights over a quad that is not a midline quad still give a point on the plane (both quads are used)
    side = A.face_anchor(base, cq, 3921, [0.4, 0.3, 0.2, 0.1], hint, midline=True).resolve(cv)[0]
    assert abs(side[0]) < 1e-12


@assets
def test_an_asymmetric_body_puts_the_anchor_between_the_two_halves(world):
    """Not forced to x = 0 arithmetically: the origin is the mean of the two mirrored points, so a body
    that has been shifted sideways carries it along."""
    base, _ = world
    v = _verts(world)
    v[:, 0] += 0.25
    cv, cq = base.body_cage(v)
    o, _ = A.face_anchor(base, cq, K.PRESET_FACES["tail_base"], None, np.array([0.0, 1.0, 0.0]),
                         midline=True).resolve(cv)
    assert o[0] == pytest.approx(0.25, abs=1e-9)


# -- the kind ----------------------------------------------------------------------------------------


@assets
@pytest.mark.parametrize("name", list(BODIES))
def test_a_default_tail_leaves_the_midline_along_the_sacral_axis(world, name):
    base, lib = world
    v = _verts(world, **BODIES[name])
    p = K.instantiate_all([K.new_spec("tail")], base, v)[0]
    assert abs(p.anchor[0]) < 1e-12                          # the root point: x = 0, exactly
    M = p.matrix
    axis = M[:3, 2] / np.linalg.norm(M[:3, 2])
    assert abs(axis[0]) < 1e-12                              # the first part of the tail is in the sagittal plane
    elevation = math.degrees(math.asin(axis[1]))
    assert 5.0 - 1e-9 <= elevation <= 18.0 + 1e-9, elevation  # the sacral surface's elevation, clamped
    assert axis[2] < 0                                       # out of the back
    # the whole shell is symmetric about x = 0 at rest (a mirrored pair of halves)
    m = p.mesh()
    assert abs(m.vertices[:, 0].mean()) < 1e-9


@assets
def test_the_migration_reads_the_first_tail_base_as_the_new_one_and_keeps_the_document(world):
    base, lib = world
    v = _verts(world)
    bv, bq = base.body_cage(None)
    legacy = K.new_spec("tail")
    legacy.placement = {"preset": "tail_base",
                        "anchor": {"face": K.LEGACY_TAIL_BASE_FACE,
                                   "corners": [int(c) for c in bq[K.LEGACY_TAIL_BASE_FACE]]},
                        "offset": K.LEGACY_TAIL_SINK, "up_hint": [0.0, 1.0, 0.0]}
    before = legacy.to_dict()
    new = K.new_spec("tail")
    a, b = K.instantiate_all([legacy], base, v)[0], K.instantiate_all([new], base, v)[0]
    assert np.array_equal(a.anchor, b.anchor) and np.allclose(a.matrix, b.matrix, atol=1e-12)
    assert np.array_equal(a.mesh().vertices, b.mesh().vertices)
    assert legacy.to_dict() == before                        # nothing is written back
    # a v1 world-point record (the first preset's point) is read the same way
    old_point = K.new_spec("tail")
    old_point.placement = {"anchor": {"point": [0.0, 1.6, -1.15]}, "offset": K.LEGACY_TAIL_SINK}
    c = K.instantiate_all([old_point], base, v)[0]
    assert np.array_equal(c.anchor, b.anchor)


@assets
def test_a_tail_placed_on_the_side_stays_there_and_midline_false_switches_it_off(world):
    base, _ = world
    v = _verts(world)
    side = K.new_spec("tail", face=7954)
    s = K.instantiate_all([side], base, v)[0]
    cv, cq = base.body_cage(v)
    assert s.anchor[0] == pytest.approx(cv[cq[7954]].mean(axis=0)[0], abs=1e-12) and abs(s.anchor[0]) > 0.5
    off = K.new_spec("tail")
    off.placement["anchor"]["midline"] = False
    o = K.instantiate_all([off], base, v)[0]
    cent = cv[cq[K.PRESET_FACES["tail_base"]]].mean(axis=0)
    assert np.allclose(o.anchor, cent, atol=1e-12) and o.anchor[0] > 0.01     # one quad: off the plane
    pair = K.new_spec("tail", side="both")
    ps = K.instantiate_all([pair], base, v)
    assert [p.name for p in ps] == ["tail-1.l", "tail-1.r"]
    assert ps[0].anchor[0] > 0 > ps[1].anchor[0]             # a pair is not on the midline


@assets
def test_the_sink_is_the_bone_root_not_an_offset(world):
    """The first preset sank the tail 0.6 along the (tilted) normal and ended with the cut ring exposed; the
    stem is 1.75 R along the tail's own axis, so the first ring is inside the body on every body."""
    base, _ = world
    from minimaster.core.subdiv import quads_to_mesh
    assert K.TAIL_SINK == 0.0
    for name in ("neutral", "tall", "short", "child", "heavy", "female"):
        v = _verts(world, **BODIES[name])
        p = K.instantiate_all([K.new_spec("tail")], base, v)[0]
        body = quads_to_mesh(*base.body_cage(v))
        n_ring = p.params["ring_verts"]
        ring = p.mesh().vertices[:n_ring]                    # the buried end of the loft
        inside = _parity_inside(ring, body.vertices[body.faces])
        assert inside.all(), name
        R = p.params["base_radius"] * p.model.scale
        assert p.root_depth == pytest.approx(1.75 * R)
        # ... and the root is well below the skin: the nearest body vertex is not within 0.75 R of the ring
        # centre (the body's own surface is 2 cage quads away)
        assert np.linalg.norm(body.vertices - ring.mean(axis=0), axis=1).min() > 0.5 * R


def _parity_inside(pts, tris):
    """Even-odd ray test along +X for the points ``pts`` against triangles ``tris`` (m, 3, 3)."""
    out = np.zeros(len(pts), dtype=bool)
    pts = np.asarray(pts) + np.array([0.0, 1.234e-6, 0.8765e-6])      # off the planes the rings lie in
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    v0, v1 = c[:, 1:] - a[:, 1:], b[:, 1:] - a[:, 1:]
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    ok = np.abs(den) > 1e-14
    safe = np.where(ok, den, 1.0)
    for k, p in enumerate(pts):
        v2 = p[1:] - a[:, 1:]
        d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
        u = np.where(ok, (d11 * d20 - d01 * d21) / safe, -1.0)
        w = np.where(ok, (d00 * d21 - d01 * d20) / safe, -1.0)
        hit = ok & (u >= 0) & (w >= 0) & (u + w <= 1)
        bx = a[:, 0] + u * (c[:, 0] - a[:, 0]) + w * (b[:, 0] - a[:, 0])
        out[k] = ((hit & (bx > p[0])).sum() % 2) == 1
    return out


# -- the tail follows the body ----------------------------------------------------------------------


@assets
def test_the_tail_is_sized_by_the_body_so_a_taller_body_has_a_longer_thicker_tail(world):
    """``length`` and ``base_radius`` are the values on the neutral body; the model is scaled by the body's
    height over the neutral body's (R stays 0.0265 H, the research's landmark rule, on every body)."""
    base, lib = world
    h0 = _height(K.neutral_cage(base, lib)[0])
    spec = K.new_spec("tail")
    seen = []
    for name in ("neutral", "tall", "short", "child"):
        v = _verts(world, **BODIES[name])
        p = K.instantiate_all([spec], base, v)[0]
        H = _height(base.body_cage(v)[0])
        k = p.model.scale
        assert k == pytest.approx(H / h0, rel=1e-9)
        seen.append((H, k * p.params["base_radius"] / H))
    # R / H is the same on every body (the default radius 0.45 is 0.0268 of the neutral body's height)
    assert max(r for _, r in seen) - min(r for _, r in seen) < 1e-9
    assert seen[0][1] == pytest.approx(0.0268, abs=0.0005)
    # the document's own scale still multiplies on top
    big = K.new_spec("tail", scale=1.5)
    v = _verts(world)
    assert K.instantiate_all([big], base, v)[0].model.scale == pytest.approx(
        1.5 * K.instantiate_all([spec], base, v)[0].model.scale)


@assets
def test_other_kinds_do_not_use_the_midline_anchor(world):
    base, _ = world
    v = _verts(world)
    for text in ("horn", "wing", "harvest:preset=upper_arms"):
        c = Character()
        from minimaster.character import kinds
        c.appendages.append(kinds.add_from_text(text, c.appendages, base=base))
        for p in c.instantiate_appendages(base, v):
            assert abs(p.anchor[0]) > 0.05, (text, p.name)
