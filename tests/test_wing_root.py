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


def test_the_sleeve_adds_nothing_at_the_elbow_and_peaks_where_the_arm_crosses_the_skin():
    sl = W.Sleeve(gain=1.0)
    t = np.linspace(0.0, 1.0, 201)
    extra = sl.extra(t, 3.16)
    assert extra[-1] == pytest.approx(0.0, abs=1e-12)               # the elbow, forearm and hand are untouched
    assert extra.min() >= 0.0
    belly = extra[(t >= 0.2) & (t <= 0.45)]
    assert belly.max() - belly.min() < 0.02 * 3.16                  # a plateau, so the swell is at the skin on any body
    assert np.argmax(extra) / 200 <= 0.2 and extra[100] > 0.9 * belly.max()
    # from the plateau to the elbow the sleeve only tapers
    assert (np.diff(extra[t >= 0.45]) <= 1e-12).all()
    # and smoothly: no crease anywhere (the second difference is small against the swell)
    assert np.abs(np.diff(extra, 2)).max() < 0.05 * belly.max()
    # twice the gain, twice the swell; none at all for zero
    assert W.Sleeve(gain=2.0).extra(0.3, 3.16) == pytest.approx(2 * sl.extra(0.3, 3.16))
    assert W.Sleeve(gain=0.0).extra(0.3, 3.16) == pytest.approx(0.0)


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
        rig = W.make_wing_rig(style, sleeve=K.WING_SLEEVES[style])
        arm = next(p for p in rig.pieces if p.name == "arm")
        prof = WR.arm_profile(arm.mesh.vertices, K.WING_SLEEVES[style].sides, np.eye(4))
        j = rig.skeleton.joints
        l_h = float(np.linalg.norm(j["elbow"] - j["shoulder"]))
        r = lambda t: float(np.interp(t, prof.t, prof.radius))               # noqa: E731
        # 0.14 of the humerus' length (+-10%) where the arm leaves the back (0.3-0.45 of it)
        for t in (0.3, 0.36, 0.45):
            assert 0.126 <= r(t) / l_h <= 0.154, (style, t, r(t) / l_h)
        assert r(0.36) >= 1.4 * r(1.0)                                        # a swell, not a rod
        assert r(0.75) < r(0.45) and r(1.0) < r(0.75)                         # and it tapers to the elbow


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
    assert set(masses) == {"scapular", "latissimus", "trapezius"}
    for m in masses.values():
        c = np.array(m.center)
        assert c[2] < lm.spine1[2] + 0.3, "a mass is over the BACK"          # behind the spine's level
        assert 0.0 < c[0] < lm.shx, m.name                                    # between the spine and the shoulder
        assert m.height == pytest.approx(m.height) and 0.05 * lm.s < m.height < 0.3 * lm.s
        assert m.pad > m.height * 0.9
    # the plateau and the fan vanish on the midline: the spinal groove stays a groove
    for name in ("scapular", "latissimus"):
        mid = np.array([[0.0, masses[name].center[1], masses[name].center[2]]])
        assert masses[name].weight(mid)[0] == pytest.approx(0.0, abs=0.02), name
    # sized by the body: a child's mass is smaller than a tall man's, in the ratio of their shoulders
    return None


@assets
def test_the_masses_are_body_relative(world):
    cm = {b: {m.name: m for m in built(world, b)[3][0].model.permanent()} for b in ("child", "tall")}
    s = {b: built(world, b)[3][0].model.lm.s for b in ("child", "tall")}
    for name in ("scapular", "latissimus", "trapezius"):
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
    """The arm's axis crosses the shaped skin more steeply than the plain skin: it grows from a hill,
    not from the flat of the back (the plain exit is 28-47 degrees; see docs/anatomy_wing.md for the
    numbers and why they stop short of 70)."""
    _, base, verts, placed, shaped = built(world, body)
    p = placed[0]
    S, E = p.model.joints(p.controls)
    d = (E - S) / np.linalg.norm(E - S)
    angles = []
    for v in (verts, shaped):
        cv, cq = base.body_cage(v)
        h = WR.BodySurface(cv, cq).hit(S, d, near=8.0)
        angles.append(math.degrees(math.asin(abs(float(d @ h[2])))))
    assert angles[1] >= angles[0] + 8.0, angles
    assert angles[1] >= 40.0, angles                          # (46 on the heavy body, 57-62 on the others)


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
    assert [m.name for m in rest][-1] == "deltoid" and rest[-1].height > 0.2
    folded = model.masses({"preset": "folded"})
    assert [m.name for m in folded][-1] == "deltoid"
    assert not np.allclose(rest[-1].center, folded[-1].center)                   # it moved with the arm
    # fold = 0.5 sticks the arm out of the flank at a steep angle: no boss needed
    assert "deltoid" not in [m.name for m in model.masses({"fold": 0.5})]
    assert len(model.masses({"fold": 0.5})) == 3


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
    assert names == {"scapular", "latissimus", "trapezius", "deltoid", "sleeve"}
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
    _, base, verts, bare, _ = built(world, "neutral", "membrane", ",bones=false")
    assert bare[0].model is None and bare[0].junction() is None
    assert set(bare[0].layers()) == {"skin"}


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
    """Total added by sliders and relief 2.2-5 L (the old inflation alone was +4.9 L, 7.5% of the body),
    at least half of it local relief, 0.7-1.2 L of it on each side's back."""
    base, lib, cat = world
    c, _, verts, placed, shaped = built(world)
    from minimaster.character.sliders import Character
    plain_c = Character()
    plain_c.macro.update(BODIES["neutral"])
    plain = lib.apply(base.verts, plain_c.effective_target_weights(cat))

    def vol(v):
        return quads_to_mesh(*base.body_cage(v)).volume()

    coupling, relief = vol(verts) - vol(plain), vol(shaped) - vol(verts)
    assert 0.5 < coupling < 2.5 and 1.4 < relief < 2.4, (coupling, relief)
    assert 2.2 < coupling + relief < 5.0
    assert relief / (coupling + relief) > 0.5
    assert 0.7 < relief / 2 < 1.2                               # per side, in litres (the research's 0.7-1.2)


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
    """The shared hook: ``Placed.layers()`` of a horn (no anatomy yet) is its printed shell alone."""
    base, lib, cat = world
    from minimaster.character.sliders import Character
    c = Character()
    c.appendages.append(K.add_from_text("horn", c.appendages, base=base, lib=lib))
    placed = c.instantiate_appendages(base, lib.apply(base.verts, mh.macro_weights()), lib)
    layers = placed[0].layers()
    assert set(layers) == {"skin"} and np.array_equal(layers["skin"].vertices, placed[0].mesh().vertices)
    assert placed[0].model is None and K.KINDS["horn"].layer_colors(placed[0], "skin") is None


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
    assert r["exit_after"] >= r["exit_before"] + 8 and r["moved"] > 300
    assert 0.5 < r["area_min"] and r["area_max"] < 2.6 and r["limited"] == 0
    assert 1.2 < r["relief_litres"] < 2.2
    assert tool.parse_pose("rest") is None and tool.parse_pose("preset=folded") == {"preset": "folded"}
    assert tool.parse_pose("raise_=45,fold=0.2") == {"raise_": 45.0, "fold": 0.2}
