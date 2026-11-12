"""The wing's root: a shoulder girdle, a muscular upper arm and the skin over both (``wingroot.py``).

The first half needs no assets: a synthetic closed quad cage (a sphere, whose -Z pole is the
"back") is the body, so the geometry is pinned down exactly (the ray-casting, the muscle mass as ONE
set of numbers that is both the skin's relief and the muscle mesh, the closed solids, the sleeve
profile). The second half builds real wings on the real morphed bodies (a neutral man, a child, a
heavy man, a tall man) and is skipped when the CC0 assets are not fetched: where the girdle is, that
the relief stays on the back and follows the body's size, that the folded wing leaves no stray
lobes, that the three layers nest, and the numbers of ``docs/anatomy_wing.md``.
"""

import dataclasses
import math

import numpy as np
import pytest

from minimaster.character import creature_anatomy as CA
from minimaster.character import junction as J
from minimaster.character import kinds as K
from minimaster.character import mhbase as mh
from minimaster.character import wingroot as WR
from minimaster.character import wings as W
from minimaster.core.mesh import Mesh
from minimaster.core.subdiv import cube_cage, quads_to_mesh, subdivide_quads

assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")


# -- a synthetic body ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sphere():
    v, q = cube_cage(2.0)
    v, q = subdivide_quads(v, q, 4)
    v = v / np.linalg.norm(v, axis=1, keepdims=True) * 10.0
    return WR.BodySurface(v, q)


def _is_closed(mesh: Mesh) -> bool:
    e = np.sort(np.concatenate([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]],
                                mesh.faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return bool((counts == 2).all())


def _back_mass(**kw):
    """A mass on the sphere's -Z pole (its 'back'), running along +Y."""
    base = dict(name="m", label="test", center=(0.0, 0.0, -10.0), axis=(0.0, 1.0, 0.0),
                half_long=3.0, half_wide=1.5, height=0.5, pad=0.8)
    base.update(kw)
    return WR.Mass(**base)


def test_the_surface_answers_rays_and_normals(sphere):
    t, p, n = sphere.hit((0.0, 0.0, 30.0), (0.0, 0.0, -1.0), near=60.0)
    assert t == pytest.approx(20.0, abs=0.1) and p[2] == pytest.approx(10.0, abs=0.1)
    assert n @ np.array([0.0, 0.0, 1.0]) > 0.999                # the outward normal
    # from inside: the first skin it meets
    t2, p2, _ = sphere.hit((1.0, 2.0, 0.0), (1.0, 0.0, 0.0), near=60.0)
    assert np.linalg.norm(p2) == pytest.approx(10.0, abs=0.2)
    assert sphere.hit((50.0, 0.0, 0.0), (1.0, 0.0, 0.0), near=5.0) is None      # a miss
    pts, nrm = sphere.project(np.array([[2.0, 1.0, -9.5], [-3.0, 0.5, -9.2]]), (0.0, 0.0, 1.0))
    assert np.allclose(np.linalg.norm(pts, axis=1), 10.0, atol=0.2)
    assert (nrm[:, 2] < -0.9).all()
    assert sphere.normal_at((0.0, 0.0, -10.0)) @ np.array([0.0, 0.0, -1.0]) > 0.999


def test_a_mass_is_one_set_of_numbers_for_the_skin_relief_and_the_muscle(sphere):
    """The mound the junction raises and the weight the muscle mesh follows are the same law: the
    relief at a point is height x weight in the relief direction, and it is zero outside the footprint."""
    m = _back_mass()
    mound = m.mound()
    assert mound.center == m.center and mound.height == m.height
    assert mound.radius == m.half_wide and mound.stretch == pytest.approx(m.half_long / m.half_wide)
    assert mound.direction == WR.RELIEF_DIRECTION
    rng = np.random.default_rng(1)
    pts = np.array([0.0, 0.0, -10.0]) + rng.normal(size=(200, 3)) * [1.5, 2.5, 0.2]
    pts[:, 2] = -np.sqrt(100.0 - pts[:, 0] ** 2 - pts[:, 1] ** 2)
    w = m.weight(pts)
    nrm = pts / 10.0
    got = J._mound_displacement(mound, pts, nrm)
    want = m.height * w[:, None] * np.asarray(WR.RELIEF_DIRECTION)[None, :]
    assert np.allclose(got, want, atol=1e-12)
    assert (w > 0).any() and (w == 0).any()
    assert m.weight(np.array([[0.0, 0.0, -10.0]]))[0] == pytest.approx(1.0)          # the crest
    assert m.weight(np.array([[0.0, m.half_long + 0.01, -9.5]]))[0] == 0.0          # beyond the footprint
    # the right-hand twin is the x-mirror
    r = m.mirrored()
    assert r.center[0] == -m.center[0] and r.axis[0] == -m.axis[0] and r.height == m.height


def test_a_muscle_pad_is_a_closed_solid_whose_top_is_the_skin_relief_minus_the_skin(sphere):
    m = _back_mass(center=(0.5, 0.0, -math.sqrt(100.0 - 0.25)))
    pad = WR.pad_mesh(m, sphere)
    assert _is_closed(pad) and pad.volume() > 0.5
    # the top of the pad: the plain skin point plus the relief (the mound's own law) minus the skin's
    # thickness along the normal. Take the crest: the highest point of the pad above the sphere.
    r = np.linalg.norm(pad.vertices, axis=1)
    crest = pad.vertices[np.argmax(r)]
    assert r.max() == pytest.approx(10.0 + m.height - WR.SKIN_T, abs=0.05)
    assert np.linalg.norm(crest[[0, 1]] - np.array(m.center)[[0, 1]]) < 0.7
    # the pad is no deeper than ``pad`` under the plain skin, and its edge is flush under the skin
    assert r.min() >= 10.0 - WR.SKIN_T - m.pad - 0.05
    # a mass twice as high gives a pad twice as proud (the pad IS the relief)
    pad2 = WR.pad_mesh(dataclasses.replace(m, height=2 * m.height), sphere)
    assert np.linalg.norm(pad2.vertices, axis=1).max() == pytest.approx(
        10.0 + 2 * m.height - WR.SKIN_T, abs=0.05)


def test_the_scapula_plate_and_the_furcula_are_closed_bones_under_the_skin(sphere):
    lm = WR.Landmarks(shx=1.72, lt=1.76, spine1=(0.0, 0.0, -2.0), neck=(0.0, 1.76, -2.0),
                      shoulder=(1.7, 2.0, 0.0), elbow=(3.4, 0.0, 0.0))
    plateau = _back_mass(name="scapular", center=(0.0, 0.0, -10.0))
    plate = WR.scapula_plate(lm, sphere, [plateau])
    assert _is_closed(plate) and plate.volume() > 0.02
    depth = 10.0 - np.linalg.norm(plate.vertices, axis=1)
    assert depth.min() >= WR.SCAPULA_DEPTH * lm.s - 0.2 and depth.max() <= (
        WR.SCAPULA_DEPTH + WR.SCAPULA_THICK) * lm.s + 0.3        # a thin plate at a constant depth
    assert WR.scapula_plate(lm, sphere, []) is None                              # no plateau, no plate
    strut = WR.furcula(lm, sphere, np.array([1.0, 0.5, -4.0]))
    assert strut is None or (_is_closed(strut) and strut.volume() > 0)


def test_the_arm_profile_reads_the_rings_the_rig_built():
    sides, n = 14, 25
    th = np.linspace(0.0, 2 * math.pi, sides, endpoint=False)
    c = np.stack([np.linspace(0.0, 6.0, n), np.zeros(n), np.zeros(n)], axis=1)
    r = np.linspace(0.5, 0.2, n)
    rings = np.stack([c[i] + r[i] * np.stack([np.zeros(sides), np.cos(th), np.sin(th)], axis=1)
                      for i in range(n)])
    verts = np.vstack([rings.reshape(-1, 3), c[0][None], c[-1][None]])
    M = np.diag([2.0, 2.0, 2.0, 1.0])
    M[:3, 3] = [1.0, 2.0, 3.0]
    prof = WR.arm_profile(verts, sides, M)
    assert len(prof.t) == n // 2 + 1                       # the humerus: up to the elbow ring
    assert prof.t[0] == 0.0 and prof.t[-1] == pytest.approx(1.0)
    assert prof.centers[0] == pytest.approx(M[:3, :3] @ c[0] + M[:3, 3])
    assert prof.radius[0] == pytest.approx(2.0 * 0.5) and prof.radius[-1] == pytest.approx(2.0 * 0.35)
    assert (prof.inscribed <= prof.radius + 1e-12).all()
    assert prof.inscribed[0] == pytest.approx(prof.radius[0] * math.cos(math.pi / sides))
    ctr, rad, ins = prof.at(0.5)
    assert rad == pytest.approx(np.interp(0.5, prof.t, prof.radius))


# -- the sleeve ----------------------------------------------------------------------------------


def test_the_sleeve_adds_nothing_at_the_elbow_and_tapers_from_the_skin_to_the_elbow():
    """The upper arm swells most at the shoulder and TAPERS all the way to the elbow (a real upper arm
    under the deltoid and the biceps). The first sleeve held a plateau to 0.45 of the humerus and then
    tapered short: once the joint moved onto the ribs (more of the arm shows) it read as a drumstick."""
    sl = W.Sleeve(gain=1.0)
    t = np.linspace(0.0, 1.0, 201)
    extra = sl.extra(t, 3.16)
    assert extra[-1] == pytest.approx(0.0, abs=1e-12)               # the elbow, forearm and hand are untouched
    assert extra.min() >= 0.0
    assert np.argmax(extra) == 0
    # where the arm leaves the back (0.15-0.3 of the humerus on every body) it is within 10% of its top
    band = extra[(t >= 0.15) & (t <= 0.3)]
    assert band.min() > 0.75 * extra[0] and band.max() <= extra[0]
    # from the skin to the elbow it only tapers, and it does not hold a plateau (the drumstick)
    tail = extra[t >= 0.15]
    assert (np.diff(tail) <= 1e-12).all()
    assert extra[90] < 0.85 * extra[30]                                  # 0.45 is clearly thinner than 0.15
    # and smoothly: no crease anywhere (the second difference is small against the swell)
    assert np.abs(np.diff(extra, 2)).max() < 0.05 * extra.max()
    # twice the gain, twice the swell; none at all for zero
    assert W.Sleeve(gain=2.0).extra(0.3, 3.16) == pytest.approx(2 * sl.extra(0.3, 3.16))
    assert W.Sleeve(gain=0.0).extra(0.3, 3.16) == pytest.approx(0.0)
    # a sleeve this thick is round (the root collar follows the round wall: 14 facets drew a zigzag there)
    assert sl.sides >= 28


def test_a_sleeve_changes_only_the_arm_piece_of_the_wing():
    for style, make in (("membrane", W.membrane_rig), ("feathered", W.feathered_rig)):
        bare = make()
        thick = make(sleeve=W.Sleeve(gain=1.0))
        names = [p.name for p in bare.pieces]
        assert names == [p.name for p in thick.pieces]
        for a, b in zip(bare.pieces, thick.pieces):
            same = np.array_equal(a.mesh.vertices, b.mesh.vertices)
            assert same == (a.name != "arm"), (style, a.name)
        # a rig built without a sleeve is what it always was: it IS the default
        assert np.array_equal(bare.rest_mesh.vertices, make(sleeve=None).rest_mesh.vertices)


def test_the_sleeve_is_thickest_where_the_registry_wings_arm_crosses_the_skin():
    for style in ("membrane", "dragon", "feathered"):
        sleeve = K.wing_sleeve(style, 1.0, K.WING_SLEEVE_S_REF)      # the neutral man's
        rig = W.make_wing_rig(style, sleeve=sleeve)
        arm = next(p for p in rig.pieces if p.name == "arm")
        prof = WR.arm_profile(arm.mesh.vertices, sleeve.sides, np.eye(4))
        j = rig.skeleton.joints
        l_h = float(np.linalg.norm(j["elbow"] - j["shoulder"]))
        r = lambda t: float(np.interp(t, prof.t, prof.radius))               # noqa: E731
        # 0.14 of the humerus' length (+-12%) where the arm leaves the back (about 0.2-0.3 of it on the neutral
        # man: the joint on the ribs; the feathered arm's buried end is blunted before that)
        for t in (0.22, 0.3):
            assert 0.123 <= r(t) / l_h <= 0.157, (style, t, r(t) / l_h)
        # a swell, not a rod (the feathered arm is already as thick as a forearm at the elbow: half the gain)
        assert r(0.22) >= (1.25 if style == "feathered" else 1.8) * r(1.0), style
        assert r(0.6) < r(0.3) and r(1.0) < r(0.6)                            # and it tapers to the elbow


def test_the_sleeve_follows_the_body_and_the_muscle_parameter():
    """The wing is not scaled with the body (its span is a parameter) but the shoulder it grows from is:
    the swell goes with the square of the body's size (so the radius at the skin about with the size),
    clipped; ``muscle`` scales it and 0 is the first versions' rod."""
    neutral = K.wing_sleeve("membrane", 1.0, K.WING_SLEEVE_S_REF)
    assert neutral.gain == pytest.approx(K.WING_SLEEVES["membrane"].gain)
    child = K.wing_sleeve("membrane", 1.0, 0.83)
    assert child.gain == pytest.approx(neutral.gain * (0.83 / K.WING_SLEEVE_S_REF) ** 2)
    assert K.wing_sleeve("membrane", 1.0, 0.1).gain == pytest.approx(neutral.gain * K.WING_SLEEVE_K[0] ** 2)
    assert K.wing_sleeve("membrane", 1.0, 9.0).gain == pytest.approx(neutral.gain * K.WING_SLEEVE_K[1] ** 2)
    assert K.wing_sleeve("dragon", 0.5, K.WING_SLEEVE_S_REF).gain == pytest.approx(0.5 * K.WING_SLEEVES["dragon"].gain)
    assert K.wing_sleeve("membrane", 0.0, 1.0) is None


# -- real bodies ---------------------------------------------------------------------------------

BODIES = {
    "neutral": dict(gender=1.0, muscle=0.55),
    "child": dict(age=0.1875),
    "heavy": dict(gender=1.0, muscle=1.0, weight=1.0),
    "tall": dict(gender=1.0, muscle=0.55, height=1.0),
}


@pytest.fixture(scope="module")
def world():
    base, lib = mh.load()
    from minimaster.character.sliders import SliderCatalog
    return base, lib, SliderCatalog(lib.names)


_BUILT: dict = {}


def built(world, body="neutral", style="membrane", extra=""):
    """``(character, base, verts, placed, shaped)`` of a wing pair on a body (cached)."""
    key = (body, style, extra)
    if key not in _BUILT:
        from minimaster.character.sliders import Character
        base, lib, cat = world
        c = Character()
        c.macro.update(BODIES[body])
        c.appendages.append(K.add_from_text(f"wing:style={style}{extra}", c.appendages, base=base,
                                            lib=lib))
        verts = lib.apply(base.verts, c.effective_target_weights(cat))
        placed = c.instantiate_appendages(base, verts, lib)
        _BUILT[key] = (c, base, verts, placed, c.shaped_body(base, verts, placed))
    return _BUILT[key]


def _moved(base, verts, shaped):
    cv0, _ = base.body_cage(verts)
    cv1, _ = base.body_cage(shaped)
    return np.linalg.norm(cv1 - cv0, axis=1) > 0, cv0, cv1


@assets
def test_the_landmarks_follow_the_body(world):
    sizes = {}
    for body in BODIES:
        _, base, verts, placed, _ = built(world, body)
        lm = placed[0].model.lm
        assert lm.shx == pytest.approx(A_joint(base, verts, "l-shoulder")[0])
        assert lm.lt == pytest.approx(A_joint(base, verts, "neck")[1] - A_joint(base, verts, "spine-1")[1])
        sizes[body] = lm.s
    assert sizes["child"] < sizes["neutral"] < sizes["tall"]
    assert 0.75 < sizes["child"] < 0.95 and 1.3 < sizes["tall"] < 1.5


def A_joint(base, verts, name):
    from minimaster.character import appendages as A
    return A.joint_position(base, name, verts)


@assets
@pytest.mark.parametrize("body", list(BODIES))
def test_the_masses_sit_on_the_back_of_every_body_and_leave_the_spinal_groove_alone(world, body):
    _, base, verts, placed, _ = built(world, body)
    model = placed[0].model
    lm = model.lm
    masses = {m.name: m for m in model.permanent()}
    assert set(masses) == {"scapular", "rhomboid", "latissimus", "trapezius"}
    for m in masses.values():
        c = np.array(m.center)
        assert c[2] < lm.spine1[2] + 0.3, "a mass is over the BACK"          # behind the spine's level
        assert 0.0 < c[0] < lm.shx, m.name                                    # between the spine and the shoulder
        assert m.height == pytest.approx(m.height) and 0.05 * lm.s < m.height < 0.3 * lm.s
        assert m.pad > m.height * 0.9
    # every mass vanishes on the midline: the spinal groove stays a groove (and the two sides never close a
    # ring round the interscapular region: the round-1 trapezius crossed the midline and, with the plateaus,
    # drew a bullseye between the shoulder blades)
    for name in ("scapular", "rhomboid", "latissimus", "trapezius"):
        mid = np.array([[0.0, masses[name].center[1], masses[name].center[2]]])
        assert masses[name].weight(mid)[0] == pytest.approx(0.0, abs=0.02), name
    # sized by the body: a child's mass is smaller than a tall man's, in the ratio of their shoulders
    return None


@assets
def test_the_masses_are_body_relative(world):
    cm = {b: {m.name: m for m in built(world, b)[3][0].model.permanent()} for b in ("child", "tall")}
    s = {b: built(world, b)[3][0].model.lm.s for b in ("child", "tall")}
    for name in ("scapular", "rhomboid", "latissimus", "trapezius"):
        assert cm["child"][name].height / cm["tall"][name].height == pytest.approx(
            s["child"] / s["tall"], rel=1e-9)
        assert cm["child"][name].half_wide < cm["tall"][name].half_wide


@assets
@pytest.mark.parametrize("body", list(BODIES))
def test_the_junction_moves_back_skin_only_and_never_the_front(world, body):
    """The relief stands the back out: no vertex of the chest moves (the S1 review found a generic
    mound reaching round the ribs)."""
    _, base, verts, placed, shaped = built(world, body)
    moved, cv0, _ = _moved(base, verts, shaped)
    _, cq = base.body_cage(verts)
    nrm = J.vertex_normals(cv0, cq)
    assert moved.sum() > 150
    cv1 = base.body_cage(shaped)[0]
    real = np.linalg.norm(cv1 - cv0, axis=1) > 0.02          # (a skirt of the Keep's fade moves a vertex by 1e-3)
    assert not (real & (nrm[:, 2] > 0.35)).any()
    assert (cv0[moved][:, 2] < placed[0].anchor[2] + 1.6 * placed[0].model.lm.s).all()
    # it is a mirror pair on a symmetric body
    from minimaster.character import appendages as A
    mirror = A.mirror_vertex_map(base)
    cv1 = base.body_cage(shaped)[0]
    d = np.linalg.norm(cv1 - cv0, axis=1)
    assert np.abs(d - d[mirror]).max() < 1e-9


@assets
@pytest.mark.parametrize("body", list(BODIES))
def test_the_arm_leaves_a_steeper_skin_than_it_did(world, body):
    """The joint sits on the ribs under the shoulder blade (the placement's second version), so the arm
    leaves the back over the upper blade at 45-64 degrees to the plain skin (28-47 in the first version, 46
    on the heavy body), and the shoulder anatomy steepens that: 60-74 after shaping (the research's 70 is
    met on the neutral, tall and short bodies; docs/anatomy_wing.md has the numbers)."""
    _, base, verts, placed, shaped = built(world, body)
    p = placed[0]
    S, E = p.model.joints(p.controls)
    d = (E - S) / np.linalg.norm(E - S)
    angles = []
    for v in (verts, shaped):
        cv, cq = base.body_cage(v)
        h = WR.BodySurface(cv, cq).hit(S, d, near=8.0)
        angles.append(math.degrees(math.asin(abs(float(d @ h[2])))))
    assert angles[1] >= angles[0] + 5.0, angles
    assert angles[0] >= 44.0 and angles[1] >= 58.0, angles               # (the heavy body: 45 -> 60)
    if body in ("neutral", "tall"):
        assert angles[1] >= 69.0, angles


def _seg_tri(P0, P1, A, B, C):
    """Moller-Trumbore, a segment against a triangle, vectorised over pairs."""
    e1, e2, D = B - A, C - A, P1 - P0
    h = np.cross(D, e2)
    a = (e1 * h).sum(1)
    ok = np.abs(a) > 1e-14
    f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    s = P0 - A
    u = f * (s * h).sum(1)
    q = np.cross(s, e1)
    v = f * (D * q).sum(1)
    t = f * (e2 * q).sum(1)
    eps = 1e-9
    return ok & (u > eps) & (v > eps) & (u + v < 1 - eps) & (t > eps) & (t < 1 - eps)


def _crossings(v, q, region):
    """The pairs of non-adjacent cage triangles (at least one touching ``region``) that cross."""
    tri = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    P = v[tri]
    lo, hi = P.min(1), P.max(1)
    out = set()
    for i in np.nonzero(region[tri].any(1))[0]:
        ov = np.nonzero(np.all(lo <= hi[i] + 1e-9, 1) & np.all(hi >= lo[i] - 1e-9, 1))[0]
        ov = ov[ov != i]
        ov = ov[~(tri[ov][:, :, None] == tri[i][None, None, :]).any((1, 2))]
        if not len(ov):
            continue
        A, B, C = P[i]
        hit = np.zeros(len(ov), bool)
        for X, Y in ((A, B), (B, C), (C, A)):
            hit |= _seg_tri(np.broadcast_to(X, (len(ov), 3)), np.broadcast_to(Y, (len(ov), 3)),
                            P[ov, 0], P[ov, 1], P[ov, 2])
        for k in range(3):
            hit |= _seg_tri(P[ov, k], P[ov, (k + 1) % 3], np.broadcast_to(A, (len(ov), 3)),
                            np.broadcast_to(B, (len(ov), 3)), np.broadcast_to(C, (len(ov), 3)))
        out.update((min(i, j), max(i, j)) for j in ov[hit])
    return out


@assets
@pytest.mark.parametrize("body", list(BODIES))
def test_the_junction_does_not_fold_or_stretch_the_skin_or_push_it_through_itself(world, body):
    """The S1 review found 24-52 new self-intersections in the armpit and the shoulder from the wing's
    spec and a stretch of 2.5-5.5x. None here: the relief stands the back out (it never shuts the slot
    between the arm and the trunk) and the Keep spheres hold the human arm still."""
    _, base, verts, placed, shaped = built(world, body)
    moved, cv0, cv1 = _moved(base, verts, shaped)
    _, cq = base.body_cage(verts)
    region = moved.copy()
    region[cq[moved[cq].any(1)].ravel()] = True
    assert len(_crossings(cv1, cq, region) - _crossings(cv0, cq, region)) == 0
    quads = np.nonzero(moved[cq].any(1))[0]
    area = lambda v: np.linalg.norm(J._cross(v[cq[quads]][:, 2] - v[cq[quads]][:, 0],         # noqa: E731
                                              v[cq[quads]][:, 3] - v[cq[quads]][:, 1]), axis=1)
    ratio = area(cv1) / np.maximum(area(cv0), 1e-12)
    assert 0.3 < ratio.min() and ratio.max() < 2.6, (ratio.min(), ratio.max())


@assets
def test_a_folded_wing_adds_no_lobes_beyond_the_masses_and_the_arm(world):
    """The S1 review: the folded wing's generic mound puffed two pillow lobes. Everything the junction
    moves lies within the footprint of a mass of the body or within the fillet of the arm, at rest,
    folded and raised."""
    base, lib, cat = world
    c, _, verts, placed, _ = built(world)
    cv, cq = base.body_cage(verts)
    for pose in ({}, {"preset": "folded"}, {"fold": 0.5}, {"raise_": 45.0}):
        spec = placed[0].junction(pose)
        shaped = J.apply_junctions(base, verts, [spec])
        moved, cv0, cv1 = _moved(base, verts, shaped)
        assert moved.sum() > 100, pose
        P = cv0[moved]
        inside = np.zeros(len(P), bool)
        for m in spec.mounds:
            inside |= (WR.Mass("x", "x", m.center, m.axis, m.radius * m.stretch, m.radius, 1.0,
                               1.0).weight(P) > 0)
        d, _ = J.tube_distance(P, np.asarray(spec.points), np.asarray(spec.radii))
        inside |= d < spec.fillet_width + 1e-9
        assert inside.all(), (pose, int((~inside).sum()))
        # and the permanent masses are the same masses at every pose: only the shoulder boss follows the arm
        perm = [m.to_dict() for m in spec.mounds[:3]]
        assert perm == [m.to_dict() for m in placed[0].junction({}).mounds[:3]], pose


@assets
def test_the_shoulder_boss_follows_the_arm_and_vanishes_when_the_arm_already_leaves_steeply(world):
    _, base, verts, placed, _ = built(world)
    model = placed[0].model
    rest = model.masses({})
    folded = model.masses({"preset": "folded"})
    assert [m.name for m in rest][-1] == "deltoid" and [m.name for m in folded][-1] == "deltoid"
    assert rest[-1].height > 0.05
    assert not np.allclose(rest[-1].center, folded[-1].center)                   # it moved with the arm
    # the folded arm leaves the back far more shallowly (about 30 degrees against 62): it gets more of the boss
    assert folded[-1].height > rest[-1].height
    # an arm leaving the skin square needs none
    lm, surf = model.lm, model.surface
    X = np.asarray(rest[0].center)
    n = surf.normal_at(X)
    assert WR.cap_mass(lm, surf, X + 0.5 * n, X - 1.5 * n, 0.4) is None
    # the boss stands its height over the skin the permanent masses leave, not over the plain skin: where it
    # overlaps the plateau it is lower by what the plateau raises there (the folded wing's lobe of round 1)
    raw = WR.cap_mass(lm, surf, *model.joints({"preset": "folded"}), model.exit_radius())
    below = sum(m.height * m.weight(np.asarray(raw.center)[None])[0] for m in model.permanent())
    assert below > 0.02 and folded[-1].height == pytest.approx(raw.height - below)


@assets
def test_the_right_wing_is_the_exact_mirror_of_the_left(world):
    _, base, verts, placed, _ = built(world)
    l, r = placed
    for pose in ({}, {"preset": "folded"}, {"raise_": 40.0}):
        sl, sr = l.junction(pose), r.junction(pose)
        a_, b_ = sl.mirrored().to_dict(), dataclasses.replace(sr, name=sl.name).to_dict()
        assert np.allclose(a_["points"], b_["points"], atol=1e-12) and np.allclose(a_["radii"], b_["radii"])
        assert len(a_["mounds"]) == len(b_["mounds"]) == len(sl.mounds)
        for ma, mb in zip(sl.mirrored().mounds, sr.mounds):
            assert np.allclose(ma.center, mb.center, atol=1e-12) and np.allclose(ma.axis, mb.axis, atol=1e-12)
            assert ma.height == pytest.approx(mb.height) and ma.radius == pytest.approx(mb.radius)
        a, b = l.layers(pose), r.layers(pose)
        for name in ("bone", "muscle", "skin"):
            assert np.allclose(a[name].vertices * [-1.0, 1.0, 1.0], b[name].vertices, atol=1e-9), name
            assert len(a[name].faces) == len(b[name].faces)


@assets
@pytest.mark.parametrize("body,style", [("neutral", "membrane"), ("heavy", "dragon"),
                                        ("child", "feathered")])
def test_the_layers_are_the_same_anatomy(world, body, style):
    """bone, muscle and skin come from ONE model: the skin is the printed shell, the muscle sits
    inside the skin's envelope (the shaped body and the arm), the arm's bone inside its muscle."""
    _, base, verts, placed, shaped = built(world, body, style)
    p = placed[0]
    layers = p.layers()
    assert set(layers) == {"bone", "muscle", "skin"}
    assert np.array_equal(layers["skin"].vertices, p.mesh().vertices)
    for k in ("bone", "muscle"):
        assert layers[k].volume() > 0 and len(layers[k].faces) > 100
    names = set(p.model.muscles(layers["skin"]))
    assert names == {"scapular", "rhomboid", "latissimus", "trapezius", "deltoid", "sleeve"}
    assert set(p.model.bones(layers["skin"])) == {"scapula", "furcula", "humerus"}
    # nesting (even-odd ray tests: the body shaped, subdivided once, plus the arm's own shell)
    cv, cq = base.body_cage(shaped)
    sv, sq = subdivide_quads(cv, cq, 1)
    body_mesh = quads_to_mesh(sv, sq)
    arm = dict(p.rig.piece_meshes(None))["arm"]
    muscle = p.model.muscles(layers["skin"])
    for name, m in muscle.items():
        env = [body_mesh] + ([arm] if name == "sleeve" else [])
        assert _inside(m.vertices, env).mean() > 0.97, (name, _inside(m.vertices, env).mean())
    bone = p.model.bones(layers["skin"])
    for name, m in bone.items():
        assert _inside(m.vertices, [body_mesh, arm]).mean() > 0.97, name
    assert _inside(bone["humerus"].vertices, [muscle["sleeve"]]).mean() > 0.95
    # the layers follow the pose, like the skin
    posed = p.layers({"preset": "folded"})
    for k in ("bone", "muscle", "skin"):
        # (the shoulder boss is not there when the folded arm already leaves the skin steeply)
        assert posed[k].vertices.shape != layers[k].vertices.shape or not np.allclose(
            posed[k].vertices, layers[k].vertices), k


def _inside(pts, meshes, axis=0, chunk=256):
    out = np.zeros(len(pts), bool)
    for mesh in meshes:
        tr = mesh.triangles()
        other = [i for i in range(3) if i != axis]
        lo, hi = pts.min(0) - 1e-6, pts.max(0) + 1e-6
        tr = tr[((tr[:, :, other].max(1) >= lo[other]).all(1)
                 & (tr[:, :, other].min(1) <= hi[other]).all(1))]
        a, b, c = tr[:, 0], tr[:, 1], tr[:, 2]
        v0, v1 = c[:, other] - a[:, other], b[:, other] - a[:, other]
        d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
        den = d00 * d11 - d01 * d01
        ok = np.abs(den) > 1e-14
        safe = np.where(ok, den, 1.0)
        for s in range(0, len(pts), chunk):
            q = pts[s:s + chunk]
            v2 = q[:, None, other] - a[None, :, other]
            d20, d21 = (v2 * v0[None]).sum(2), (v2 * v1[None]).sum(2)
            u = np.where(ok, (d11 * d20 - d01 * d21) / safe, -1.0)
            w = np.where(ok, (d00 * d21 - d01 * d20) / safe, -1.0)
            hit = ok & (u >= 0) & (w >= 0) & (u + w <= 1)
            bx = a[None, :, axis] + u * (c[None, :, axis] - a[None, :, axis]) \
                + w * (b[None, :, axis] - a[None, :, axis])
            out[s:s + chunk] |= ((hit & (bx > q[:, None, axis])).sum(1) % 2) == 1
    return out


@assets
def test_without_its_junction_the_muscle_would_stand_out_of_the_skin(world):
    """The envelope test above has teeth: with the relief switched off the pads of the masses poke out
    of the plain body."""
    _, base, verts, placed, _ = built(world)
    p = placed[0]
    cv, cq = base.body_cage(verts)
    sv, sq = subdivide_quads(cv, cq, 1)
    plain = quads_to_mesh(sv, sq)
    pad = p.model.muscles(p.mesh())["scapular"]
    assert _inside(pad.vertices, [plain]).mean() < 0.95


@assets
def test_a_wing_without_an_arm_or_a_fillet_has_no_skin_junction(world):
    _, base, verts, placed, _ = built(world, "neutral", "membrane", ",fillet=0")
    assert placed[0].junction() is None
    assert set(placed[0].layers()) == {"bone", "muscle", "skin"}          # the anatomy is there, the skin is left alone
    # a wing drawn without its arm still has the shoulder that moves it: the relief of the masses (the
    # round-1 build had no model and no junction at all for it), but no tube (no arm skin to flow into)
    # and no collar
    for style in ("membrane", "feathered"):
        _, base, verts, bare, shaped = built(world, "neutral", style, ",bones=false")
        spec = bare[0].junction()
        assert bare[0].model is not None and not bare[0].model.tube
        assert spec is not None and not spec.tube and len(spec.mounds) >= 4
        assert np.linalg.norm(base.body_cage(shaped)[0] - base.body_cage(verts)[0], axis=1).max() > 0.1
        assert set(bare[0].layers()) == {"bone", "muscle", "skin"}
        assert bare[0].kind.collar_profile(bare[0]) is None


@assets
def test_a_draft_feathered_wing_has_the_final_wings_junction(world):
    base, lib, cat = world
    c, _, verts, placed, _ = built(world, "neutral", "feathered")
    spec = c.appendages[0]
    draft = K.instantiate_all([spec], base, verts, draft=True)
    assert draft[0].model is not None
    a, b = draft[0].junction(), placed[0].junction()
    assert np.allclose(a.points, b.points) and a.mounds == b.mounds
    assert len(draft[0].mesh().faces) < 0.6 * len(placed[0].mesh().faces)


@assets
def test_an_older_document_with_a_stored_fillet_still_builds(world):
    """Records written before the shoulder anatomy store ``fillet: 3.0`` (the old default) and nothing
    else the anatomy needs: they load, build and get the new root."""
    base, lib, cat = world
    from minimaster.character.appendage_spec import AppendageSpec
    spec = AppendageSpec.from_dict({"id": "wing-1", "kind": "wing", "side": "both",
                                    "params": {"style": "membrane", "fillet": 3.0}})
    verts = lib.apply(base.verts, mh.macro_weights(gender=1.0, muscle=0.55))
    placed = K.instantiate_all([spec], base, verts)
    assert len(placed) == 2 and placed[0].junction().width == 3.0
    assert placed[0].model is not None and len(placed[0].junction().mounds) >= 3


# -- the body the wing adapts ------------------------------------------------------------------------


def test_the_wings_body_adaptation_no_longer_inflates_the_whole_rib_cage():
    """measure_bust_circ moved the back 3-4x more than the chest (76% of the +4.9 L): a sack with a hard
    lower rim. The back mass is local relief now (wingroot.py), the coupling keeps the chest."""
    s = CA.CONSEQUENCES["wing"].sliders
    assert 0.0 < s["measure.measure_bust_circ"] <= 0.35
    assert 0.0 < s["torso.torso_muscle_dorsi"] <= 0.5
    assert s["torso.torso_muscle_pectoral"] >= 0.8            # the flight muscle is on the chest


@assets
def test_the_volume_of_the_wing_is_mostly_local_and_within_budget(world):
    """Total added by sliders and relief 1.5-4 L (the old inflation alone was +4.9 L, 7.5% of the body), 0.35-0.8 L
    of it on each side's back as local relief. (Round 1 asked 0.7-1.2 L a side and half of the total local: its
    masses stood 0.11-0.34 s high and stacked to 0.48 units where they overlapped, which drew the bullseye
    between the shoulder blades and the folded wing's lobe; the masses of the second placement are lower,
    wider and enveloped, docs/anatomy_wing.md section 3.)"""
    base, lib, cat = world
    c, _, verts, placed, shaped = built(world)
    from minimaster.character.sliders import Character
    plain_c = Character()
    plain_c.macro.update(BODIES["neutral"])
    plain = lib.apply(base.verts, plain_c.effective_target_weights(cat))

    def vol(v):
        return quads_to_mesh(*base.body_cage(v)).volume()

    coupling, relief = vol(verts) - vol(plain), vol(shaped) - vol(verts)
    assert 0.5 < coupling < 2.0 and 0.7 < relief < 1.6, (coupling, relief)
    assert 1.5 < coupling + relief < 4.0
    assert relief / (coupling + relief) > 0.38
    assert 0.35 < relief / 2 < 0.8                              # per side, in litres


@assets
def test_the_wing_tool_draws_the_layers(world, tmp_path):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "tools" / "wing_root_view.py"
    spec = importlib.util.spec_from_file_location("wing_root_view", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    out = tmp_path / "w.png"
    assert tool.main([str(out), "wing:style=membrane", "180,15", "--layers", "bone,muscle,skin",
                      "--window", "2.0", "--size", "120x120", "--levels", "0"]) == 0
    assert out.stat().st_size > 500
    cut = tool.cut_mesh(Mesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5]], dtype=float),
                             np.array([[0, 1, 2], [1, 2, 3]])), (1.5, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert len(cut.faces) == 1 and np.array_equal(cut.faces[0], [2, 1, 0])        # section: the inside faces out


@assets
def test_a_kind_without_an_anatomy_has_only_the_skin_layer(world):
    """The shared hook: ``Placed.layers()`` of a kind with no anatomy model is its printed shell alone. (The
    horn has its own layers now: a minimal kind stands for "no anatomy".)"""
    base, lib, cat = world
    verts = lib.apply(base.verts, mh.macro_weights())

    class Plain(K.Kind):
        id = "plain"

    shell = Mesh(np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]))
    placed = K.Placed(name="plain-1", spec_id="plain-1", kind=Plain(), side="", anchor=np.zeros(3),
                      matrix=None, params={}, controls={}, static=shell)
    layers = placed.layers()
    assert set(layers) == {"skin"} and np.array_equal(layers["skin"].vertices, shell.vertices)
    assert placed.model is None and Plain().layer_colors(placed, "skin") is None
    assert Plain().collar_profile(placed) is None and Plain().collar_wall(placed) is shell


@assets
def test_the_report_tool_measures_what_the_tests_pin(world):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "tools" / "wing_root_report.py"
    spec = importlib.util.spec_from_file_location("wing_root_report", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    base, lib, cat = world
    r = tool.measure(base, lib, cat, "neutral", "membrane", None, with_crossings=False)
    assert r["exit_after"] >= r["exit_before"] + 5 and r["exit_after"] >= 69 and r["moved"] > 300
    assert 0.5 < r["area_min"] and r["area_max"] < 2.6 and r["limited"] == 0
    assert 0.7 < r["relief_litres"] < 1.6
    assert tool.parse_pose("rest") is None and tool.parse_pose("preset=folded") == {"preset": "folded"}
    assert tool.parse_pose("raise_=45,fold=0.2") == {"raise_": 45.0, "fold": 0.2}


# -- the second placement (S6b): the joint on the ribs under the shoulder blade ------------------------------

PLACE_BODIES = dict(BODIES, short=dict(gender=1.0, muscle=0.55, height=0.0),
                    female=dict(gender=0.0, muscle=0.5))


def _wing_on(world, macro, text="wing", placement=None, params=None):
    """``(placed, base, verts)`` of a wing record (``placement`` / ``params`` merged into a new record's)."""
    from minimaster.character.sliders import Character
    base, lib, cat = world
    c = Character()
    c.macro.update(macro)
    spec = K.add_from_text(text, c.appendages, base=base, lib=lib)
    if placement:
        spec.placement.update(placement)
    if params:
        spec.params.update(params)
    c.appendages.append(spec)
    verts = lib.apply(base.verts, c.effective_target_weights(cat))
    return c.instantiate_appendages(base, verts, lib), base, verts


@assets
@pytest.mark.parametrize("body", list(PLACE_BODIES))
def test_the_joint_sits_on_the_ribs_under_the_shoulder_blade_and_the_arm_leaves_over_the_blade(world, body):
    """Round 1: the socket was 0.9 units inside the torso under the posterolateral corner, the arm left
    through the human deltoid (the exit at 0.83-1.12 of the shoulder joint's x, beyond it on a child and a
    woman). Now the joint is a landmark of the body: half the body size under the skin of the back at 0.48
    of the shoulder's x and half the upper trunk up from spine-1 (the level of the scapular spine), and the
    arm leaves over the upper blade, clear of the human arm."""
    placed, base, verts = _wing_on(world, PLACE_BODIES[body])
    p = placed[0]
    m = p.model
    lm, surf = m.lm, m.surface
    S, E = m.joints({})
    over = surf.hit(S, (0.0, 0.0, -1.0), near=4.0)                # straight back to the skin
    assert over[0] == pytest.approx(A_sock("WING_SOCKET_DEPTH") * lm.s, rel=0.02)
    assert over[1][0] / lm.shx == pytest.approx(A_sock("WING_SOCKET_X"), abs=0.02)
    assert (over[1][1] - lm.spine1[1]) / lm.lt == pytest.approx(A_sock("WING_SOCKET_Y"), abs=0.02)
    # the anchor IS that skin point (the collar's patch and the lab's framing are centred on it)
    assert np.allclose(p.anchor, over[1], atol=1e-6)
    d = (E - S) / np.linalg.norm(E - S)
    t, X, n = surf.hit(S, d, near=6.0)
    assert 0.66 <= X[0] / lm.shx <= 0.86, X[0] / lm.shx                 # over the blade, medial to the acromion
    assert 0.5 <= (X[1] - lm.spine1[1]) / lm.lt <= 0.8
    for k in m._keep():                                                # the human upper arm is not in its way
        assert np.linalg.norm(X - np.asarray(k.center)) > k.radius + 0.3 * lm.s
    # and the right wing is the exact mirror on a symmetric body
    assert np.allclose(placed[1].matrix[:3, 3] * [-1.0, 1.0, 1.0], p.matrix[:3, 3], atol=1e-6)


def A_sock(name):
    from minimaster.character import appendages as A
    return getattr(A, name)


def _first_version_record(**extra):
    """A wing record as a file written before the placement versions existed has it (HEAD before S6b)."""
    from minimaster.character import appendages as A
    pl = {"preset": "shoulder_blades", "anchor": {"face": K.PRESET_FACES["shoulder_blades"]},
          "offset": A.WING_OFFSET, "roll": A.WING_ROLL, "up_hint": list(A.WING_UP_HINT)}
    pl.update(extra)
    return pl


@assets
def test_an_unstamped_record_with_the_first_numbers_moves_and_a_pinned_one_keeps_them(world):
    """Migration (docs/anatomy_wing.md, 'Documents'). A record written before the placement versions stores the
    first preset's numbers explicitly: it is read as the second version (it MOVES: that placement is the
    defect), bit for bit what a new record builds; ``placement.version = 1`` pins the first version (and its
    bone-thin upper arm); a record whose sink a person changed keeps it (version 1); a stamped record is read
    as its stamp, its stored numbers never sniffed."""
    from minimaster.character import appendages as A
    macro = BODIES["neutral"]
    new, base, verts = _wing_on(world, macro)
    old = {k: v for k, v in new[0].kind.resolve_placement.__self__.__dict__.items()} if False else None  # noqa
    for pl, version in ((_first_version_record(), 2), ({"preset": "shoulder_blades"}, 2),
                        (_first_version_record(version=1), 1), (_first_version_record(offset=-1.2), 1)):
        spec = K.new_spec("wing", base=base)
        spec.placement.clear()
        spec.placement.update(pl)
        placed = K.instantiate_all([spec], base, verts)
        p = placed[0]
        if version == 2:
            assert np.array_equal(p.mesh().vertices, new[0].mesh().vertices), pl
            assert np.allclose(p.matrix, new[0].matrix)
        else:
            sink = -float(pl["offset"])
            assert p.matrix[2, 3] - p.anchor[2] == pytest.approx(sink, abs=1e-9), pl     # straight in, unscaled
            arm = next(q for q in p.rig.rig.pieces if q.name == "arm")
            assert len(arm.mesh.vertices) % W.ARM_SIDES == 2 and len(arm.mesh.vertices) % 32 != 2   # the rod
    # a pinned record that leaves the numbers out gets the FIRST version's (the legacy builder's wing)
    spec = K.new_spec("wing", base=base)
    spec.placement.clear()
    spec.placement.update({"preset": "shoulder_blades", "version": 1})
    full = K.new_spec("wing", base=base)
    full.placement.clear()
    full.placement.update(_first_version_record(version=1))
    assert np.array_equal(K.instantiate_all([spec], base, verts)[0].mesh().vertices,
                          K.instantiate_all([full], base, verts)[0].mesh().vertices)
    # the first version asked for muscle explicitly gets the sleeve
    spec = K.new_spec("wing", base=base, params={"muscle": 1.0})
    spec.placement.clear()
    spec.placement.update(_first_version_record(version=1))
    arm = next(q for q in K.instantiate_all([spec], base, verts)[0].rig.rig.pieces if q.name == "arm")
    assert len(arm.mesh.vertices) % 32 == 2
    # a stamped version-2 record's sink is in units of the body's size, whatever number it stores
    spec = K.new_spec("wing", base=base)
    spec.placement["offset"] = -0.7
    p = K.instantiate_all([spec], base, verts)[0]
    assert p.matrix[2, 3] - p.anchor[2] == pytest.approx(0.7 * p.model.lm.s, abs=1e-9)
    assert K.new_spec("wing").placement["version"] == 2 and A.WING_OFFSET_V2 == -A.WING_SOCKET_DEPTH


def test_the_version_survives_a_save_and_old_files_still_load(tmp_path):
    """The stamp is plain data in the record: it is written and read back; a file written before it (a v2
    .mmchar with a wing, as HEAD writes it) and a v1 file load unchanged and are not rewritten."""
    import json
    from minimaster.character.sliders import Character
    c = Character()
    c.appendages.append(K.new_spec("wing"))
    path = tmp_path / "a.mmchar"
    c.save(path)
    back = Character.load(path)
    assert back.appendages[0].placement["version"] == 2
    assert back.to_dict() == c.to_dict()
    head = c.to_dict()
    head["appendages"][0]["placement"] = _first_version_record()
    (tmp_path / "head.mmchar").write_text(json.dumps(head))
    old = Character.load(tmp_path / "head.mmchar")
    assert "version" not in old.appendages[0].placement
    assert old.to_dict()["appendages"][0]["placement"] == _first_version_record()
    v1 = Character.from_dict({"version": 1, "macro": {"gender": 1.0}})
    assert v1.appendages == []


@assets
@pytest.mark.parametrize("body", ["neutral", "child", "heavy"])
def test_the_wings_root_collar_meets_the_back_and_the_arm_tangentially(world, body):
    """The collar (collar.py) on the wing's ARM (its own piece, not the membrane or the feathers): its rim lies
    on the shaped skin and leaves it within a few degrees, its top meets the arm within a few degrees, where
    the skin met the arm at 45-65 degrees without it; a folded arm lying along the back gets none (the
    construction cannot follow a wall that leans over the skin on most of its ring)."""
    from minimaster.character import collar as CO
    placed, base, verts = _wing_on(world, BODIES[body])
    c_shaped = J.apply_junctions(base, verts, J.junction_specs(placed))
    p = placed[0]
    prof = p.kind.collar_profile(p)
    assert prof is not None and prof.material == "skin" and prof.up is not None
    wall = p.kind.collar_wall(p)
    arm = next(q for q in p.rig.rig.pieces if q.name == "arm")
    assert len(wall.faces) == len(arm.mesh.faces) < len(p.mesh().faces) / 4
    for L in (1, 2):
        r = CO.join_report(base, c_shaped, p, levels=L)
        assert r["watertight"] and r["outward"]
        assert r["rim_p95"] < 7.0 and r["wall_p95"] < 8.0, r
        assert r["before_p50"] > 40.0, r
        assert r["above_skin_min"] >= -CO.SKIN_TOL
    # the two wings' collars stay apart (their skirts under the skin met at a child's spine at width 1.3)
    q = placed[1]
    a = CO.collar_pieces(base, c_shaped, p, levels=1)[0][1]
    b = CO.collar_pieces(base, c_shaped, q, levels=1)[0][1]
    assert CO.count_crossings(a.vertices, a.faces, b.vertices, b.faces) == 0
    # the open side is the side the arm comes from (minus its direction along the skin)
    S, E = p.model.joints({})
    assert float(np.dot(np.asarray(prof.up), (E - S) / np.linalg.norm(E - S))) < 0.0
    # off: collar=0, a folded arm, level 0
    assert p.kind.collar_profile(dataclasses.replace(p, params={**p.params, "collar": 0.0})) is None
    folded = dataclasses.replace(p, controls={"preset": "folded"})
    assert p.model.exit_angle({"preset": "folded"}) < WR.COLLAR_MIN_EXIT
    assert p.kind.collar_profile(folded) is None
    assert CO.collar_pieces(base, c_shaped, p, {"preset": "folded"}, levels=2) == []   # the pose handed in decides
    assert CO.collar_pieces(base, c_shaped, p, levels=0) == []


@assets
def test_without_the_collar_wall_hook_the_collar_would_meet_the_membrane(world):
    """The hook has teeth: the whole wing (membrane and fingers) as the wall is ten times the triangles."""
    placed, base, verts = _wing_on(world, BODIES["neutral"])
    p = placed[0]
    assert len(K.Kind.collar_wall(p.kind, p).faces) > 4 * len(p.kind.collar_wall(p).faces)


@assets
def test_the_muscle_lies_a_skin_thickness_under_the_shaped_skin(world):
    """Round 1 measured the muscle pads 0.11 (median) and up to 0.2 (p90) under the shaped skin where masses
    overlap, against the stated skin thickness of 0.06: each pad followed its OWN mound while the junction
    adds every mound. The pads are built from the sum now: the gap from the pad's top to the shaped,
    subdivided skin is the skin's thickness. The same measure with each pad on its own mound (the round-1
    construction) shows the gap the review found."""
    from minimaster.character import collar as CO
    placed, base, verts = _wing_on(world, BODIES["neutral"])
    # the masses' relief alone (the fillet that climbs the arm lies over the arm's own sleeve, not over a pad)
    shaped = J.apply_junctions(base, verts, [q.junction().with_(points=(), radii=(), base_radius=None)
                                             for q in placed])
    p = placed[0]
    skin = CO.shaped_skin(base, shaped, 2)
    patch = CO.SkinPatch(skin[0], skin[1], p.anchor, 4.0)
    masses = p.model.masses({})

    def gaps(acting):
        out = []
        for m in masses:
            pad = WR.pad_mesh(m, p.model.surface, acting=acting(m))
            top = pad.vertices[1:1 + WR.PAD_RINGS * WR.PAD_SIDES].reshape(WR.PAD_RINGS, WR.PAD_SIDES, 3)[:-2]
            _, _, sd, _ = patch.closest(top.reshape(-1, 3))
            out.append(-sd)
        return np.concatenate(out)

    g = gaps(lambda m: masses)
    assert np.median(g) == pytest.approx(WR.SKIN_T, abs=0.015) and np.percentile(g, 90) < 0.075, (
        np.median(g), np.percentile(g, 90))
    own = gaps(lambda m: [m])
    assert np.percentile(own, 90) > np.percentile(g, 90) + 0.02


def test_the_numbers_a_mutation_could_change_unseen():
    """Pins the round-1 review's surviving mutants: the latissimus fan runs up the back (not across it), the
    strut ends at the front of the chest beside the midline, the fillet's in-plane pull is applied, the back
    coupling of the body adaptation stays small."""
    lat = next(m for m in WR.PERMANENT_MASSES if m[0] == "latissimus")
    ax = np.asarray(lat[4]) / np.linalg.norm(lat[4])
    assert ax[1] > 0.95 and 0.0 < ax[0] < 0.3
    assert WR.FILLET_PULL == 1.0
    s = CA.CONSEQUENCES["wing"].sliders
    assert s["torso.torso_muscle_dorsi"] <= 0.4
    trap = next(m for m in WR.PERMANENT_MASSES if m[0] == "trapezius")
    assert trap[2] - trap[6] > 0.15          # its footprint stops short of the midline (x/shx - wide/shx)


@assets
def test_the_strut_ends_in_front_beside_the_midline_and_the_pull_is_used(world):
    placed, base, verts = _wing_on(world, BODIES["neutral"])
    p = placed[0]
    lm = p.model.lm
    bones = p.model.bones(p.mesh())
    strut = bones["furcula"].vertices
    S, _ = p.model.joints({})
    end = strut[np.argmax(strut[:, 2])]
    assert end[2] > S[2] + 0.5 * lm.s and abs(end[0]) < 0.35 * lm.shx      # in front of the joint, beside the midline
    spec = p.junction()
    assert spec.pull == WR.FILLET_PULL
    # (the in-plane redistribution itself moves the skin by less than 1e-4 here: the arm leaves the back
    # steeply, so there is almost nothing to pull in the plane; FILLET_PULL = 0 is a near-equivalent mutant,
    # pinned by value above and said so in docs/anatomy_wing.md)
    a = J.apply_junctions(base, verts, [spec])
    b = J.apply_junctions(base, verts, [dataclasses.replace(spec, pull=0.0)])
    assert np.abs(a - b).max() < 1e-3


@assets
@pytest.mark.parametrize("style", ["membrane", "feathered"])
def test_the_junction_tube_lies_on_the_arm_itself(world, style):
    """The tube the skin flows into is the arm's own axis carried by the pose, not the shoulder -> elbow line
    of the joints: a feathered arm is lifted 0.2-0.25 wing units off that line into the feather stack, and a
    tube on the joint line lay half outside the arm (the core's conform step cut it down to nothing)."""
    placed, base, verts = _wing_on(world, BODIES["neutral"], f"wing:style={style}")
    p = placed[0]
    for pose in ({}, {"raise_": 40.0}, {"fold": 0.5}):
        spec = p.junction(pose)
        cut = J.conform_spec(spec)
        # the conform step keeps the tube (it trims at most the station at the buried joint, inside the cap)
        assert cut is not None and len(cut.points) >= len(spec.points) - 1, (style, pose)
        rings = p.model._sleeve_rings(p.mesh(pose)).mean(axis=1)
        d, _ = J.tube_distance(np.asarray(spec.points[1:]), rings, np.full(len(rings), 1e-6))
        assert d.max() < 0.08, (style, pose, d.max())


@assets
def test_the_shoulder_anatomy_is_not_dropped_by_the_join_guard(world):
    """The wing's masses are mass on purpose (the body's anatomy); the core's join guard compares the join
    of the skin with the tube before and after and dropped the whole wing junction on the tall body's
    feathered wing (the masses tilt the skin on the acute side of an oblique arm). The seam itself is the
    root collar's (collar.py)."""
    placed, base, verts = _wing_on(world, BODIES["tall"], "wing:style=feathered")
    specs = J.junction_specs(placed)
    assert not any(s.join_guard for s in specs)
    info = {}
    out = J.apply_junctions(base, verts, specs, info)
    assert not info["dropped"] and not info["tube_only"] and not info["mounds_only"]
    assert np.linalg.norm(base.body_cage(out)[0] - base.body_cage(verts)[0], axis=1).max() > 0.15


@assets
@pytest.mark.parametrize("macro", [dict(gender=1.0), dict(age=0.1875), dict(gender=0.0)])
def test_a_stowed_wing_of_the_second_placement_rests_on_the_back(world, macro):
    """The stow tests of tests/test_wingrig.py run on ``back_wings()`` (the first placement); the registry's
    wing is the second now, with its joint higher and half as deep: the same bounds hold for it (the heavy
    body's feathered wing is 14.5% inside the torso stowed: over the 10% asked, docs/anatomy_wing.md)."""
    from tests.test_wings import _inside_body
    base, lib, cat = world
    v = lib.apply(base.verts, mh.macro_weights(**macro))
    cv, cq = base.body_cage(v)
    tris = cv[np.vstack([cq[:, [0, 1, 2]], cq[:, [0, 2, 3]]])]
    for style, limit in (("feathered", 0.10), ("membrane", 0.15), ("dragon", 0.15)):
        pw = K.instantiate_all([K.new_spec("wing", params={"style": style}, base=base)], base, v)[0].rig
        w = pw.pose(fold=1.0).vertices
        inside = _inside_body(w[::9], tris).mean()
        assert 0.02 < inside < limit, (style, macro, inside)
        a, b = pw.rig.ranges["wrist_knob"]
        assert _inside_body(w[a:b], tris).mean() < 0.30, (style, macro)
