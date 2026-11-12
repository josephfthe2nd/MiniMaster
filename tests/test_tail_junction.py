"""How the skin of the sacrum moulds around a tail root (``tailrig.junction_spec``).

The generic junction of the first version tented the skin around a pipe and folded the two walls of the
gluteal cleft through each other (12 new self-intersecting pairs, 9 of them outside the tail). The tail's own
spec is the anatomy of the root: the skin tube is the tail's flared skin, the muscle masses at the root
(``tailrig.ROOT_MASSES``: the sacral wedge up the back, the oblique gluteal bands that flank the root) are its mounds and
at the same time the bulk of the body-side muscles of the ``muscle`` layer (round 4: tests/test_tail_layers_in_skin.py), and
the cleft below the root is KEPT (beads of Keep spheres down the midline): the tail leaves a crease above the
perineum, the cleft is not filled. Every size is in ``R_j = min(R, R_MAX)`` so that a thick tail cannot shred the skin.
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
    c.macro.update(BODIES[name] if isinstance(name, str) else name)
    c.appendages.append(K.add_from_text(text, c.appendages, base=base, lib=lib))
    if controls:
        c.appendages[0].controls = dict(controls)
    verts = lib.apply(base.verts, c.effective_target_weights(cat))
    placed = c.instantiate_appendages(base, verts, lib)
    shaped = c.shaped_body(base, verts, placed)
    return c, verts, placed, shaped


def _R(p):
    """The base radius of the built tail (body units)."""
    return p.params["base_radius"] * p.model.scale


def _Rj(p):
    """The radius the junction's sizes are in: the base radius capped at ``R_MAX`` (scaled with the body)."""
    return min(p.params["base_radius"], 0.62) * p.model.scale          # (0.62: a literal, not the constant under test)


# -- the spec ---------------------------------------------------------------------------------------------


@assets
def test_the_spec_is_the_tails_root_anatomy(world):
    c, v, (p,), _ = _build(world, "neutral")
    spec = p.junction()
    R = _R(p)
    assert spec.active and spec.width == pytest.approx(K.TAIL_FILLET) and spec.height == pytest.approx(K.TAIL_FILLET_HEIGHT)
    assert spec.base_radius == pytest.approx(R)                      # (the default tail is under the cap)
    # it carries its own cleft Keep and is the trunk's: neither the generic cleft hold nor the part guess is wanted
    assert spec.owner == "trunk" and spec.tight_gap == 0.0 and spec.smooth is not None
    # the tube: the skin's flared section from the buried end to one fillet width + 1 R beyond the skin point
    pts = np.asarray(spec.points)
    reach = np.linalg.norm(pts[-1] - p.anchor)
    assert 2.3 * R <= reach <= 2.7 * R and spec.radii[0] == pytest.approx(R * 1.35 * np.cos(np.pi / K.TAIL_RING_VERTS) * J.HUG)
    assert np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() >= 4.0 * R          # stem + reach: the fillet's whole length
    # the masses: the sacral wedge is one elongated mound up the back, the two gluteal bands flank the root
    sacral, left, right = spec.mounds
    up = np.asarray(p.matrix[:3, 0]) / p.model.scale                  # the tail frame's up (the back)
    assert sacral.axis is not None and abs(np.dot(sacral.axis, up)) > 0.999 and sacral.stretch == pytest.approx(3.5 / 1.6)
    assert sacral.height == pytest.approx(0.60 * R) and sacral.radius == pytest.approx(1.6 * R)     # literals: the wedge is 0.6 R thick
    # ... with its crest AT the top of the root (1 R up the back), not up the back (the convex hump of the first try)
    assert np.dot(np.asarray(sacral.center) - p.anchor, up) == pytest.approx(1.0 * R)
    assert left.height == pytest.approx(0.25 * R) and left.radius == pytest.approx(0.6 * R)
    assert left.mirrored().center == pytest.approx(right.center, abs=1e-4)      # (the base mesh itself is 5e-5 asymmetric)
    lat = np.asarray(p.matrix[:3, 1]) / p.model.scale                  # the frame's lateral axis
    assert abs(np.dot(np.asarray(left.center) - p.anchor, lat)) == pytest.approx(1.55 * R, rel=0.02)  # medial (round 3: 2.2 R)
    # ... an oblique band: its long axis runs down the back and out (infero-laterally), 2 : 1
    assert np.dot(left.axis, up) < -0.5 and abs(np.dot(left.axis, lat)) > 0.5 and left.stretch == pytest.approx(1.1 / 0.6)

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
def test_the_spec_and_the_muscles_are_built_from_the_same_masses(world):
    """One description: each gluteal mound of the spec is the footprint and the bulk of the gluteal band of the muscle layer
    (the band lies inside the mass's ellipse and spans most of its length), the sacral wedge's mound is the bulk of the
    dorsalis near the root, and the mounds' sizes are the masses' numbers."""
    c, v, (p,), shaped = _build(world, "neutral")
    spec = p.junction()
    R = _R(p)
    model = p.model.model()
    for mass, mound in zip(T.ROOT_MASSES, spec.mounds):
        assert mound.radius == pytest.approx(mass.half[1] * R) and mound.height == pytest.approx(mass.height * R)
        assert mound.stretch == pytest.approx(mass.half[0] / mass.half[1])
    pieces = dict(model.body_muscles(p.model.root_skin()))
    for mass, name in ((T.ROOT_MASSES[1], "gluteus_maximus_l"), (T.ROOT_MASSES[2], "gluteus_maximus_r")):
        loc = pieces[name].vertices / p.params["base_radius"]
        ax = np.array(mass.axis[:2])
        rel = loc[:, :2] - np.array(mass.center)
        along, across = rel @ ax, rel @ np.array([-ax[1], ax[0]])
        u = np.hypot(along / mass.half[0], across / mass.half[1])
        assert np.percentile(u, 98) <= 1.05, (name, np.percentile(u, 98))         # inside the footprint (the clip moves a hair)
        assert np.ptp(along) >= 1.6 * mass.half[0]                                # ... and as long as most of it



@assets
def test_a_pose_moves_the_free_tail_in_the_spec_not_the_root(world):
    _, _, (rest,), _ = _build(world, "neutral")
    _, _, (up,), _ = _build(world, "neutral", controls={"lift": 60.0})
    a, b = rest.junction(), up.junction()
    R = _R(rest)
    assert np.linalg.norm(np.asarray(a.points[0]) - b.points[0]) < 0.25 * R          # the buried end stays buried (the bend starts 1.5 R below the skin)
    assert np.linalg.norm(np.asarray(a.points[-1]) - b.points[-1]) > 1.0 * R         # the free end is where the pose puts it
    assert a.keep == b.keep and a.mounds[1:] == b.mounds[1:]              # the cleft and the gluteal masses stay
    assert b.mounds[0].height < 0.7 * a.mounds[0].height                  # ... the sacral wedge is taken back as the tail lifts


@assets
def test_a_lifted_tail_takes_the_sacral_wedge_back_and_raises_no_sack(world):
    """At lift 90 the tail lies up the back across the wedge: with the rest-pose wedge the skin behind the root
    was a blister (the reviewers' 'sack or goiter'). The wedge fades out between 20 and 80 degrees; the skin
    below the root is not raised more than at rest."""
    base, _, _ = world
    rises = {}
    for lift in (0.0, 60.0, 90.0):
        c, v, (p,), shaped = _build(world, "neutral", controls={"lift": lift})
        R = p.params["base_radius"]
        a = T.SkinPatch.from_cage(base.body_cage(v)[0], base.body_cage(v)[1], p.model.frame, R)
        b = T.SkinPatch.from_cage(base.body_cage(shaped)[0], base.body_cage(shaped)[1], p.model.frame, R)
        dz = b.z - a.z
        crest = dz[np.searchsorted(a.xs, 3.0), len(a.ys) // 2]          # over the wedge, 3 R up the back (beyond the fillet)
        below = dz[a.xs < -0.5].max()                                   # (round 4: half a radius below the centre; the root
        # itself rises with the lift now: its stem pivots at the sacrocaudal joint)
        rises[lift] = (crest, below)
    # (round 4, measured: 0.24, 0.06, 0.0 R; with the wedge kept at full height in every pose: 0.23, 0.2+)
    assert rises[0.0][0] > 0.15 and rises[60.0][0] < 0.5 * rises[0.0][0] and rises[90.0][0] < 0.4 * rises[0.0][0]
    assert rises[90.0][1] <= rises[0.0][1] + 0.02                       # nothing new below the root
    spec0, spec90 = _build(world, "neutral")[2][0].junction(), _build(world, "neutral", controls={"lift": 90.0})[2][0].junction()
    assert len(spec90.mounds) == len(spec0.mounds) - 1 == 2             # the wedge is gone (the gluteal pair stays)
    assert [m.height for m in spec90.mounds] == pytest.approx([m.height for m in spec0.mounds[1:]])


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
    R = _Rj(p)
    spec = p.junction()
    assert mag.max() <= spec.move_limit() + 1e-12 and mag.max() <= 1.4 * R + 1e-12      # a bounded fillet
    moved = np.nonzero(mag > 0)[0]
    assert 105 < len(moved) < 320                                                       # a root's worth of vertices (round 4: 116 - 147)
    # zero outside the root's neighbourhood: within 4 R sideways, from 4 R below to 7 R above along the back
    rel = cv0[moved] - p.anchor
    assert np.abs(rel[:, 0]).max() <= 4.0 * R and rel[:, 1].min() >= -4.0 * R and rel[:, 1].max() <= 7.0 * R
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
    # no quad flips or collapses (the validity pass) and the fillet really lifts the skin onto the root: the skin
    # meets the tail 15-26 degrees closer to tangent at level 1 (round 2b: neutral 73 -> 58, child 75 -> 51); the heavy
    # body's closed cleft holds the root: 66 -> 57 (round 2: 67 -> 75, its tube then dropped by the join guard)
    info = J.measure_junction(base, v, shaped, spec, levels=1)
    assert info["flipped"] == 0 and info["min_area_ratio"] >= J.MIN_AREA_RATIO
    gain = 6.0 if name == "heavy" else 12.0
    assert info["join_angle_deg_after"] < info["join_angle_deg_before"] - gain, info


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
@pytest.mark.parametrize("name", ["neutral", "child", "short", "heavy"])
def test_the_cleft_beads_keep_the_cleft_out_of_the_junction(world, name):
    """Fail-first for the cleft: the keep spheres hold the skin of the cleft below the root. Without them (the keep removed
    from the spec) 30 - 70 more cage vertices move (in round 4, with the tube's displacement no longer diffused over the
    cage, the cleft is not dragged far: what the beads now hold is WHICH skin moves), all of them in the cleft below
    the root's centre and near the midline, and the guard that backs off the vertices of quads that would fold or cross
    has to act on 2.5 - 3.6 times as many vertices. With them the cleft and the perineum below the apex do not move at
    all. (Round 2b's version measured the drag of the cleft below the apex, 0.004 R; it cannot see the round-4 junction.)"""
    base, _, _ = world
    c, v, (p,), shaped = _build(world, name)
    spec = p.junction()
    info_with, info_without = {}, {}
    out1 = J.apply_junctions(base, v, [spec], info_with)
    out0 = J.apply_junctions(base, v, [spec.with_(keep=())], info_without)
    assert info_without["limited"] >= 2 * info_with["limited"] + 10, (info_with, info_without)
    cv0 = base.body_cage(v)[0]
    m1 = np.linalg.norm(base.body_cage(out1)[0] - cv0, axis=1) > 0
    m0 = np.linalg.norm(base.body_cage(out0)[0] - cv0, axis=1) > 0
    extra = np.nonzero(m0 & ~m1)[0]
    assert len(extra) >= 25, len(extra)
    up = np.asarray(p.matrix[:3, 0]) / p.model.scale
    R = _R(p)
    rel = cv0[extra] - p.anchor
    assert (rel @ up < 0.2 * R).mean() > 0.9 and np.abs(rel[:, 0]).max() < 2.0 * R      # the cleft, below the root, near the midline
    low = list(CLEFT_LOW)
    assert np.linalg.norm(base.body_cage(out1)[0][low] - cv0[low], axis=1).max() == 0.0



def _only(spec, *mounds):
    """``spec`` reduced to the given mounds (no tube, no cleft): what those masses alone do to the skin."""
    return spec.with_(points=(), radii=(), keep=(), mounds=tuple(mounds))


@assets
def test_the_sacral_wedge_is_a_mound_up_the_back_and_not_down_the_cleft(world):
    """The wedge alone (no tube; the spec's own cleft beads kept): it raises the skin from the root up the back, its
    crest is at the root (the lumbar line runs down it onto the tail) and the cleft below is held."""
    base, _, _ = world
    c, v, (p,), _ = _build(world, "neutral")
    spec = p.junction()
    out = J.apply_junctions(base, v, [spec.with_(points=(), radii=(), mounds=(spec.mounds[0],))])
    cv0, cq = base.body_cage(v)
    cv1, _ = base.body_cage(out)
    d = cv1 - cv0
    R = _R(p)
    up = np.asarray(p.matrix[:3, 0]) / p.model.scale                      # the tail frame's up
    along = (cv0 - p.anchor) @ up
    nrm = J.vertex_normals(cv0, cq)
    rise = (d * nrm).sum(1)
    lifted = np.nonzero(rise > 0.05 * R)[0]
    assert len(lifted) >= 8
    assert (along[lifted] > -0.75 * R).all() and (along[lifted] < 4.6 * R).all()     # the wedge: from the root up the back
    assert -0.5 * R < along[np.argmax(rise)] < 2.0 * R                              # its crest is at the top of the root
    # the wedge is as thick as the mass says (literal 0.6 R: a wedge 3 times lower, or none, fails)
    assert np.linalg.norm(d, axis=1).max() == pytest.approx(0.60 * R, rel=0.15)


@assets
def test_the_gluteal_bands_lift_the_skin_beside_the_root_along_their_oblique_footprint(world):
    """The critics' round-1 mutant 'cheek mounds off' changed 0.2 % of the pixels; round 3's round gluteal masses were 2.2 R_j
    out and their lenses stood out of the hip. The gluteal masses are oblique bands (coccygeal gluteus maximus with the
    caudofemoralis) beside the root: alone they lift the skin by their height (0.25 R); the rise is longer along the band
    (infero-lateral) than across it; nothing moves further than GLUTEAL_REACH R_j from the midline."""
    base, _, _ = world
    c, v, (p,), shaped = _build(world, "neutral")
    spec = p.junction()
    R = _R(p)
    cv0, cq = base.body_cage(v)
    gl, gr = spec.mounds[1], spec.mounds[2]
    both = J.apply_junctions(base, v, [_only(spec, gl, gr)])
    d = np.linalg.norm(base.body_cage(both)[0] - cv0, axis=1)
    assert d.max() == pytest.approx(0.25 * R, rel=0.2)
    moved = d > 0.02 * R
    lat = np.asarray(p.matrix[:3, 1]) / p.model.scale
    assert np.abs((cv0[moved] - p.anchor) @ lat).max() <= T.GLUTEAL_REACH * R
    rel = cv0[moved] - np.asarray(gl.center)
    left = (cv0[moved] - p.anchor) @ lat * np.sign(np.dot(np.asarray(gl.center) - p.anchor, lat)) > 0
    ax = np.asarray(gl.axis)
    across = np.cross(ax, np.asarray(p.matrix[:3, 2]) / p.model.scale)
    assert np.ptp(rel[left] @ ax) > 1.5 * np.ptp(rel[left] @ across)            # a band, not a ball
    # without them the skin beside the root is lower than with them (the rest of the spec is the same)
    without = J.apply_junctions(base, v, [spec.with_(mounds=spec.mounds[:1])])
    with_ = J.apply_junctions(base, v, [spec])
    z0 = np.linalg.norm(base.body_cage(without)[0] - base.body_cage(with_)[0], axis=1)
    assert z0.max() > 0.12 * R


# (round 4) the envelope tests that were here measured the skin with the same circular height field that built the round-3
# lenses and passed while 11 % of the tall body's gluteal lens stood out of the skin: they are replaced by the true signed
# distance in tests/test_tail_layers_in_skin.py (all muscle and bone vertices, parity against the closed body, levels 1
# and 2, six bodies, base radii 0.5 - 0.85, random bodies) and its two-sided companion (the muscle fills the bulge).


# -- thick tails -----------------------------------------------------------------------------------------------


def _quality(base, v, p):
    specs = J.junction_specs([p])
    info = {}
    out = J.apply_junctions(base, v, specs, info)
    return out, info, J.junction_quality(base, v, out, specs, levels=(1,))


@assets
@pytest.mark.parametrize("name", ["neutral", "child"])
@pytest.mark.parametrize("text", ["tail:base_radius=0.7", "tail:base_radius=0.85", "tail:base_radius=1.2",
                                  "tail:base_radius=1.5,length=14", "tail:base_radius=2.0,length=14",
                                  "tail:base_radius=1.0,flare=1.5"])
def test_a_thick_tails_cage_displacement_is_bounded_and_untorn(world, name, text):
    """The UI allows a base radius up to 2.0 (4 times the default). The first version's sizes grew with R: at 0.85 the
    lumbar skin tore in zipper-like creases (673 vertices moved, 125 backed off) and at 1.5 the junction moved
    3,167 vertices by 2.4 units into a belt at the waist. Every size is now in R_j = min(R, R_MAX): the number of
    vertices moved, the largest move and the reach are bounded in R_j whatever the radius, nothing flips and nothing new
    crosses itself, at the cage and once subdivided. (Round 3 named this test "does not shred the skin"; what it measures
    is the CAGE displacement of the junction: the shreds the critic saw at 0.7 - 1.2 are in the collar, whose own report
    counts its self-crossings: tools/collar_report.py, the core track's.)"""
    base, _, _ = world
    c, v, (p,), shaped = _build(world, name, text)
    out, info, q = _quality(base, v, p)
    Rj = _Rj(p)
    cv0 = base.body_cage(v)[0]
    mag = np.linalg.norm(base.body_cage(out)[0] - cv0, axis=1)
    moved = np.nonzero(mag > 0)[0]
    rel = cv0[moved] - p.anchor
    assert len(moved) <= 720 and info["limited"] <= 80, (len(moved), info["limited"])      # (measured: at most 689 and 64; the first version: 3,167 and 125)
    assert mag.max() <= 1.4 * Rj + 1e-12
    assert rel[:, 1].max() <= 7.0 * Rj and rel[:, 1].min() >= -4.5 * Rj and np.abs(rel[:, 0]).max() <= 4.5 * Rj    # no belt at the waist
    assert q["flipped"] == 0 and J.quality_violations(q) == [] and q["selfx_new_visible"] == 0 and q.get("L1_selfx_new_visible", 0) == 0


@assets
def test_the_junctions_sizes_stop_growing_at_r_max():
    """The cap itself: a model with a base radius of 0.62 (neutral-body units) and one of 2.0 give the same mounds, fillet
    reach and keep spheres; below the cap they scale with R."""
    def spec(R):
        m = T.TailModel(T.TailParams(base_radius=R, length=8.0))
        return T.junction_spec("t", m, np.eye(4), 1.5, 0.5)
    a, b, c = spec(0.62), spec(0.9), spec(2.0)
    for x in (b, c):
        assert [m.height for m in x.mounds] == pytest.approx([m.height for m in a.mounds])
        assert [m.radius for m in x.mounds] == pytest.approx([m.radius for m in a.mounds])
        assert [k.radius for k in x.keep] == pytest.approx([k.radius for k in a.keep])
        assert x.base_radius == pytest.approx(a.base_radius) and x.max_move == pytest.approx(a.max_move)
        assert x.width == a.width
    d = spec(0.31)
    assert d.mounds[0].height == pytest.approx(0.5 * a.mounds[0].height) and d.base_radius == pytest.approx(0.31)
    assert a.base_radius == pytest.approx(0.62) and a.max_move == pytest.approx(1.4 * 0.62)
    # the tube keeps the tail's own radius (a fat tail is a fat tube; only what the skin does around it is capped)
    assert c.radii[0] > 3.0 * a.radii[0] * 0.95


# -- the poses ---------------------------------------------------------------------------------------------


@assets
@pytest.mark.parametrize("lift", [60.0, 90.0])
@pytest.mark.parametrize("name", ["neutral", "child", "heavy"])
def test_a_lifted_tail_leaves_a_valid_skin(world, name, lift):
    base, _, _ = world
    c, v, (p,), shaped = _build(world, name, controls={"lift": lift})
    out, info, q = _quality(base, v, p)
    assert J.quality_violations(q) == [] and q["selfx_new_visible"] == 0 and q.get("L1_selfx_new_visible", 0) == 0
    assert q["max_move"] <= 1.4 * _Rj(p) + 1e-12


# -- round 2b: the collar, the guard, the cost ---------------------------------------------------------------------


@assets
def test_the_tails_collar_is_tuned_with_the_wedge_and_capped_for_a_thick_tail(world):
    """The tail's collar (``collar.CollarProfile``) is wider and taller on the back than the generic one (1.8 / 0.7 root
    radii, curvature-continuous at its rim: falloff 3), the wedge carries the rest of the ramp; a tail thicker than
    ``tailrig.R_MAX`` gets the collar of a tail of that radius (the junction is capped the same way) and one wider than
    the sacrum (base radius > 1.2) none at all (it cost 1-2 s a render and the tail hides it)."""
    import dataclasses
    from minimaster.character import collar as CO
    _, _, (p,), _ = _build(world, "neutral")
    prof = p.kind.collar_profile(p)
    assert (prof.width, prof.height, prof.falloff, prof.ventral) == (1.8, 0.7, 3.0, 0.0) and prof.strength == 1.0
    assert prof.up == (0.0, 1.0, 0.0)
    generic = CO.CollarProfile()
    assert prof.width > generic.width and prof.height > generic.height
    thick = dataclasses.replace(p, params=dict(p.params, base_radius=0.93))
    assert p.kind.collar_profile(thick).strength == pytest.approx(T.R_MAX / 0.93)
    assert p.kind.collar_profile(dataclasses.replace(p, params=dict(p.params, base_radius=1.5))) is None
    assert p.kind.collar_profile(dataclasses.replace(p, params=dict(p.params, collar=0.0))) is None
    assert K.TAIL_RING_VERTS == 24 and p.params["ring_verts"] == 24     # the collar's wall join showed the 16-gon's facets


@assets
@pytest.mark.parametrize("name,controls", [("heavy", None), ("heavy", {"lift": 60.0}), ("neutral", {"lift": 60.0})])
def test_the_masses_are_never_dropped_by_the_join_guard(world, name, controls):
    """The masses are the muscle layer: the core's join guard, which may apply a spec tube-only or mounds-only when the
    cage's join gets worse, would leave the lens of the layer poking through the skin (the heavy body lifted 60: the
    guard chose tube-only and 0.21 R of the gluteal lens showed). The tail's spec is exempt (``join_guard=False``; the
    collar makes the join, not the cage) and its mounds are applied."""
    base, _, _ = world
    c, v, (p,), shaped = _build(world, name, controls=controls)
    spec = p.junction(controls)
    assert spec.join_guard is False
    info = {}
    J.apply_junctions(base, v, [spec], info)
    assert info["tube_only"] == [] and info["mounds_only"] == [] and info["dropped"] == []
    # (round 2b also asserted that on the heavy body lifted 60 the guard WOULD have applied the spec tube-only: with the
    # round-4 masses and lift it no longer does, so that case cannot show the exemption; what is left is the flag and that
    # the masses are applied whole)


@assets
def test_the_body_side_layers_sample_the_skin_only_when_asked(world):
    """Moulding and subdividing the body's skin around the root (``RootSkin``, 0.2 - 0.4 s) is the layers' only need; a tail
    is built on every edit, so it happens on the first ``layers()`` call, not in ``instantiate`` or ``junction``."""
    _, _, (p,), _ = _build(world, "neutral")
    assert callable(p.model._patch) and not p.model._skins
    p.mesh({"lift": 30.0})
    p.junction()
    assert callable(p.model._patch) and not p.model._skins              # skin, pose and junction: nothing sampled
    p.layers()
    assert len(p.model._skins) == 1 and isinstance(next(iter(p.model._skins.values())), T.RootSkin)   # the layers' own skin


def _dorsal_bulge(world, name, levels=1):
    """The side silhouette of the body and the collar (``|x| < 1.5 R``) from the top of the tail's root up 5 R: its
    largest bulge out beyond the straight chord from where it meets the tail to its top, in R (> 0: a hump the
    lumbar line runs over before it reaches the tail; a concave ramp onto the tail is <= 0)."""
    from minimaster.character import collar as CO
    from minimaster.character import portrait as po
    base, _, _ = world
    c, v, (p,), shaped = _build(world, name)
    body = po.character_shells(base, shaped, levels=levels, with_eyes=False)[0].mesh
    meshes = [body.transform(np.linalg.inv(po.MH_TO_Z_UP))] + [
        m for _, _, m, _ in CO.collar_shells(base, shaped, [p], levels=levels, skin=CO.skin_from_shell(body))]
    R = _R(p)
    tv = p.mesh().vertices
    near = np.linalg.norm(tv - p.anchor, axis=1) < 3.0 * R
    levels_y = tv[near, 1].max() + np.linspace(0.02, 5.0, 100) * R
    z = np.full(len(levels_y), np.inf)
    for m in meshes:
        V, F = m.vertices, m.faces
        E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        E = E[(np.abs(V[E, 0]) < 1.5 * R).all(axis=1)]
        a, b = V[E[:, 0]], V[E[:, 1]]
        lo, hi = np.minimum(a[:, 1], b[:, 1]), np.maximum(a[:, 1], b[:, 1])
        for i, L in enumerate(levels_y):
            k = (lo <= L) & (hi > L)
            if k.any():
                f = (L - a[k, 1]) / (b[k, 1] - a[k, 1])
                z[i] = min(z[i], (a[k, 2] + f * (b[k, 2] - a[k, 2])).min())
    ok = np.isfinite(z)
    y, z = levels_y[ok], z[ok]
    chord = z[0] + (z[-1] - z[0]) * (y - y[0]) / (y[-1] - y[0])
    return float(((chord - z) / R).max())                          # (the back of the body is -z)


@assets
def test_the_lumbar_line_runs_down_a_ramp_onto_the_tail_without_a_hump(world):
    """The side view the user was shown: the back ran over a bulge (the wedge's crest 2.4 R up the back, then the cage
    fillet and the collar) before the tail left from under it. With the wedge's crest at the root the dorsal silhouette
    from the root up 5 R is a ramp: no bulge over its chord (measured at level 1: 0.00 R on the neutral and child
    bodies, 0.03 R on the heavy one, whose fat fold above the sacrum is its own; the round-2 state: 0.23 R there)."""
    # (round 4: the neutral and child assertions could not fail, the round-2 code scored 0.00 there too; only the heavy body
    # discriminates here: tests/test_tail_layers_in_skin.py::test_the_back_rises_onto_the_tail_with_its_crest_at_the_root
    # sees the hump on every body)
    assert _dorsal_bulge(world, "heavy") < 0.1
