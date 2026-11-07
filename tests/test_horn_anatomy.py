"""The horn as an anatomy (``hornrig.py``, ``kinds.HornKind``): where it sits, how it leaves the skull, how big it
is for the head, the equiangular curve and the cone, the keratin sheath with its growth rings and its foot, the
layers (bone, soft tissue, skin) and the skin that is moulded around the root.

The first half needs no assets (the model in its own frame); the second builds horns on the real morphed
body and is skipped when the CC0 assets are not fetched. Every number checked here is one the research
(``docs/anatomy_horn.md``) gives, and each test is written so that the mistake it guards against fails it.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from minimaster.character import creature_anatomy as ca
from minimaster.character import ecorche
from minimaster.character import hornrig as HR
from minimaster.character import junction as J
from minimaster.character import kinds as K
from minimaster.character import mhbase as mh
from minimaster.character import print_report as pr
from minimaster.character import printprep as pp
from minimaster.character.appendage_spec import AppendageSpec
from minimaster.core.mesh import Mesh

assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")

BODIES = {
    "neutral": {},
    "tall": dict(height=1.0),
    "short": dict(height=0.0),
    "child": dict(age=0.1875),
    "old": dict(age=1.0),
    "female": dict(gender=0.0),
}
SHAPES = HR.SHAPE_NAMES


# -- helpers ---------------------------------------------------------------------------------------


def parity_inside(pts, mesh, chunk=256):
    """Even-odd ray test along +X of ``pts`` against ``mesh`` (triangles cropped to the points' box)."""
    pts = np.asarray(pts, dtype=np.float64)
    out = np.zeros(len(pts), dtype=bool)
    if not len(pts):
        return out
    tris = mesh.vertices[mesh.faces]
    lo, hi = pts[:, 1:].min(0) - 1e-9, pts[:, 1:].max(0) + 1e-9
    keep = ((tris[:, :, 1:].max(1) >= lo).all(1) & (tris[:, :, 1:].min(1) <= hi).all(1)
            & (tris[:, :, 0].max(1) >= pts[:, 0].min() - 1e-9))
    a, b, c = (tris[keep][:, i] for i in range(3))
    v0, v1 = c[:, 1:] - a[:, 1:], b[:, 1:] - a[:, 1:]
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    ok = np.abs(den) > 1e-14
    safe = np.where(ok, den, 1.0)
    for s in range(0, len(pts), chunk):
        p = pts[s:s + chunk]
        v2 = p[:, None, 1:] - a[None, :, 1:]
        d20, d21 = (v2 * v0[None]).sum(2), (v2 * v1[None]).sum(2)
        u = np.where(ok, (d11 * d20 - d01 * d21) / safe, -1.0)
        w = np.where(ok, (d00 * d21 - d01 * d20) / safe, -1.0)
        hit = ok & (u >= 0) & (w >= 0) & (u + w <= 1)
        bx = a[None, :, 0] + u * (c[None, :, 0] - a[None, :, 0]) + w * (b[None, :, 0] - a[None, :, 0])
        out[s:s + chunk] = ((hit & (bx > p[:, None, 0])).sum(1) % 2) == 1
    return out


def quads_mesh(v, q):
    tris = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    return Mesh(v, tris)


@pytest.fixture(scope="module")
def world():
    base, lib = mh.load()
    return base, lib


@pytest.fixture(scope="module")
def bodies(world):
    base, lib = world
    return {name: lib.apply(base.verts, mh.macro_weights(**kw)) for name, kw in BODIES.items()}


def place(world, v, text="horn", **kw):
    base, _ = world
    spec = K.add_from_text(text, [], base=base) if isinstance(text, str) else text
    return K.instantiate_all([spec], base, v, **kw)


def left_of(world, v, shape="curved", **params):
    base, _ = world
    spec = K.new_spec("horn", params={"shape": shape, **params}, side="left", base=base)
    return K.instantiate_all([spec], base, v)[0]


def angle(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))))))


# ==============================================================================================
# the model in its own frame (no assets)


def model(**kw):
    shape = kw.pop("shape", "curved")
    row = HR.SHAPES[shape]
    base = dict(shape=shape, length=row["length"], base_radius=row["base_radius"], segments=row["segments"],
                curve=row["curve"], twist=0.0, turns=0.0, tip_radius=row["tip"], flare=row["flare"],
                rings=row["rings"], tilt=row["tilt"])
    base.update(kw)
    return HR.HornModel(HR.HornParams(**base))


def test_the_registry_shapes_are_the_researchs_numbers():
    h = K.KINDS["horn"]
    assert h.param("shape").choices == SHAPES == ("curved", "ram", "spiral", "antelope", "ossicone")
    d = h.defaults()
    assert (d["length"], d["base_radius"], d["tip_radius"]) == (1.6, 0.17, 0.06)
    r = h.defaults(shape="ram")
    assert (r["length"], r["base_radius"]) == (3.0, 0.28)
    assert HR.SHAPES["ossicone"]["material"] == "skin" and HR.SHAPES["curved"]["material"] == "bone"
    assert h.material({"shape": "ossicone"}).base_color != h.material({"shape": "curved"}).base_color
    # the orientation rule of the research: 15 to 45 degrees from the skull's surface normal
    for shape, row in HR.SHAPES.items():
        assert 15.0 <= row["tilt"] <= 45.0, shape


def test_the_taper_is_a_cone_not_a_thorn():
    m = model(rings=0.0, flare=0.0)
    R, L, tip = m.R, m.L, m.p.tip
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        want = tip + (R - tip) * (1.0 - t) ** HR.TAPER_P
        assert float(m.radius(t * L)) == pytest.approx(want, rel=1e-9)
    # the old 1.25 exponent is a thorn: its mid-shaft is thinner than the cone's by more than 5%
    thorn = tip + (R - tip) * 0.5 ** 1.25
    assert float(m.radius(0.5 * L)) > 1.05 * thorn
    assert HR.TAPER_P == pytest.approx(1.05)
    r = m.radius(np.linspace(0, L, 200))
    assert (np.diff(r) < 0).all()


def test_the_root_flare_decays_over_a_third_of_a_radius_and_the_stem_holds_it():
    m = model(rings=0.0, flare=0.15)
    R = m.R
    r0 = float(m.radius(0.0))
    assert r0 == pytest.approx(1.15 * float(m.radius(0.0, flare=False)))
    # relaxed to the cone by about a radius up the shaft
    assert float(m.radius(1.5 * R)) == pytest.approx(float(m.radius(1.5 * R, flare=False)), rel=0.01)
    assert float(m.radius(-0.5 * R)) == pytest.approx(r0)                # held below the skin point
    # the flare is limited: a flare of 1.5 is a trumpet 2.5 radii wide
    assert HR.HornParams(flare=1.5).flare_eff == HR.FLARE_MAX < 1.5


def test_the_growth_rings_are_ridges_of_the_researchs_pitch_and_none_where_they_should_not_be():
    amp = 0.05
    m = model(shape="ram", rings=amp)
    R, L = m.R, m.L
    s = np.linspace(0.0, L, 4000)
    w = m.ring_wave(s)
    assert w.max() <= amp * 1.0001 and w.min() >= -1e-12
    assert w.max() > 0.9 * amp                                           # they exist
    assert np.abs(w[s < 0.1 * R]).max() < 0.1 * amp                      # none right at the skin point
    assert np.abs(w[s > 0.98 * L]).max() < 1e-9                          # none on the tip
    peaks = np.nonzero((w[1:-1] > w[:-2]) & (w[1:-1] >= w[2:]) & (w[1:-1] > 0.5 * amp))[0]
    pitch = np.diff(s[peaks + 1])
    assert len(peaks) >= 6
    assert pitch[0] == pytest.approx(HR.RING_PITCH * R, rel=0.25)         # 0.8 R at the root
    assert pitch[-1] < 0.8 * pitch[0]                                    # tighter distally
    assert not model(rings=0.0).ring_wave(s).any()
    # the loft resolves them: at least RING_PER_PERIOD - 1 stations per period
    st = m._stations
    assert (np.diff(st[st > 2 * R]) < pitch.min() / (HR.RING_PER_PERIOD - 1.5)).all()


def test_the_curve_is_an_equiangular_spiral_that_keeps_the_saved_curve_meaning():
    m = model(rings=0.0, flare=0.0, shape="curved")
    L, q = m.L, HR.SHAPES["curved"]["q"]
    psi = m.p.curve * L / 2.0                                            # the first versions' total turn
    s, pos, T, N, B = m.path()
    sel = s >= 0
    s, T = s[sel], T[sel]
    # the total turn
    turn = math.acos(float(np.clip(T[0] @ T[-1], -1, 1)))
    assert turn == pytest.approx(psi, rel=0.01)
    # the curvature k / (S0 - s), non-zero at the root and rising to the tip
    k = psi / math.log(q / (q - 1.0))
    kap = np.linalg.norm(np.gradient(T, s, axis=0), axis=1)
    for u in (0.1, 0.3, 0.5, 0.7, 0.9):
        i = int(np.searchsorted(s, u * L))
        assert kap[i] == pytest.approx(k / (q * L - s[i]), rel=0.05)
    assert kap[2] > 0.3 * kap[len(kap) // 2]                              # not the old 0 at the root
    assert (np.diff(kap[5:-5]) > -1e-6).all()
    # a straight horn is straight
    assert np.allclose(model(curve=0.0).path()[2][:, :2], 0.0, atol=1e-12)


def test_a_lyre_changes_sign_and_a_corkscrew_winds_around_a_straight_axis():
    lyre = model(shape="antelope")
    s, pos, T, N, B = lyre.path()
    sel = s >= 0
    ang = np.degrees(np.arctan2(T[sel][:, 0], T[sel][:, 2]))              # the bend is in the x-z plane
    assert ang.max() > 15 and ang[-1] < ang.max() - 5                      # out, then back in
    cork = model(shape="spiral", turns=2.25)
    plain = model(shape="spiral", turns=0.0)
    off_plane = lambda c: np.linalg.svd(c - c.mean(axis=0))[1][2]                    # noqa: E731
    c, c0 = cork.centreline(200)[1], plain.centreline(200)[1]
    assert off_plane(c0) < 1e-9                                  # the planar spiral stays in its plane
    assert off_plane(c) > 0.25 * HR.HELIX_RHO * cork.R           # the corkscrew winds out of it
    # ... around the SAME axis: the pair of curves agree on where the horn is going (the helix is a
    # displacement of a planar axis, not a curvature that would turn the whole horn)
    assert np.linalg.norm(c[-1] - c0[-1]) < 1.2 * HR.HELIX_RHO * cork.R
    sk = cork.skin()
    assert sk.integrity_report()["watertight"]


def test_the_curvature_never_folds_the_loft_on_itself():
    for shape in SHAPES:
        m = model(shape=shape, curve=4.0, length=12.0)
        s, pos, T, N, B = m.path()
        kap = np.linalg.norm(np.gradient(T, s, axis=0), axis=1)
        r = m.radius(np.clip(s, 0.0, m.L), flare=False)
        assert (kap * r < HR.BEND_LIMIT * 1.25).all(), shape


@pytest.mark.parametrize("shape", SHAPES)
def test_the_sheath_is_closed_outward_and_ends_in_a_rounded_tip(shape):
    m = model(shape=shape)
    sk = m.skin()
    rep = sk.integrity_report()
    assert rep["watertight"] and rep["outward"] and rep["volume"] > 0 and rep["degenerate_faces"] == 0, rep
    # a rounded end of at least the tip radius (never a point: 0.2 mm is what a printer resolves)
    assert m.p.tip >= 0.012
    s, pos, T, N, B = m.path()
    end = pos[-1]
    ring = sk.vertices[np.linalg.norm(sk.vertices - end, axis=1) < 1.05 * float(m.radius(m.L))]
    assert len(ring) >= 8
    # the section is an ellipse whose minor axis is `aspect` of the major
    a, b = m.semi_axes(0.5 * m.L)
    assert b / a == pytest.approx(HR.SHAPES[shape]["aspect"])


def test_the_foot_follows_the_skin_it_comes_out_of_not_a_plane():
    """The old root was a flat fan cap pushed into the scalp (a 'faceted cut'): the foot is a stack of rings that
    follow the skin they come out of, so on a tilted skin the foot is a tilted collar and on a flat one it is level."""
    m = model()
    tilt = 0.5
    tilted = HR.PlaneField((-math.sin(tilt), 0.0, math.cos(tilt)))
    flat = HR.PlaneField((0.0, 0.0, 1.0))
    bowl_t, foot_t, _ = m.foot_rings(tilted)
    bowl_f, foot_f, _ = m.foot_rings(flat)
    ra = float(m.radius(0.0))
    # the ring at the skin line (h = 0, the second ring): its heights spread by about 2 ra tan(tilt) on a tilted
    # skin and are level on a flat one
    spread_t = np.ptp(foot_t[1][:, 2])
    spread_f = np.ptp(foot_f[1][:, 2])
    assert spread_f < 2e-3 * ra                      # (level, to the bend of the axis above the skin)
    assert spread_t == pytest.approx(2.0 * ra * math.tan(tilt) * (1 + m.p.flare_eff * 0.0), rel=0.2)
    # and every ring vertex of the foot sits at its nominal height over the skin below it
    for ring, h in zip(foot_t[:2], HR.FOOT_HEIGHTS[:2]):
        above = ring[:, 2] - tilted.height(ring[:, :2])
        assert np.allclose(above, h * ra, atol=0.05 * ra)


def test_the_foot_is_buried_below_the_skin_and_the_groove_and_bead_are_at_the_skin_line():
    m = model(flare=0.15)
    field = HR.PlaneField((-math.sin(0.47), 0.0, math.cos(0.47)))
    sk = m.skin(field)
    v = sk.vertices
    h = v[:, 2] - field.height(v[:, :2])
    ra = float(m.radius(0.0))
    rho = np.hypot(v[:, 0], v[:, 1])
    # nothing sticks out sideways below the skin: below it the sheath is at most the flare's width
    assert rho[h < -0.3 * ra].max() <= (1.0 + m.p.flare_eff) * ra * 1.001 + 1e-9
    # the skin line: the sheath is a bit narrower at the groove (h = GROOVE_H) than at the bead above (BEAD_H)
    near_wall = (rho > 0.5 * ra) & (np.abs(v[:, 1]) < 1e-3 * ra + 0.02) & (v[:, 0] < 0)
    assert near_wall.sum() > 0


# ==============================================================================================
# on the real body


@assets
@pytest.mark.parametrize("body", list(BODIES))
@pytest.mark.parametrize("preset", ["horn_crown", "horn_brow", "horn_temple_crest"])
def test_every_preset_sits_on_the_frontal_bone_on_every_body(world, bodies, body, preset):
    """The research's forbidden zones, as measured on the morphed head: |x| in [0.41, 0.72] of the head's
    half-width, the footprint above eye + 0.45 (top - eye), base centre 0.40 head-widths in front of the
    ear, and never over the ear."""
    base, _ = world
    v = bodies[body]
    cage = base.body_cage(v)
    head = HR.head_frame(base, v, cage)
    spec = K.new_spec("horn", preset=preset, side="left", base=base)
    p = K.instantiate_all([spec], base, v)[0]
    a = p.anchor
    hw0 = 0.861 * head.k                                   # the head's half-width with the ears, on the neutral body 0.861
    # (the research: 0.549 / 0.411 / 0.716 of it on the neutral body; a child's head is relatively wider)
    assert 0.40 <= a[0] / hw0 <= 0.82, (a[0] / hw0)
    # (the research's 0.45 and 0.40 are for the footprint's lowest point and the centre of the base; the
    # centre of the crest quad is at 0.47 / 0.84 on the neutral body and a little under on others)
    assert a[1] >= head.eye[1] + 0.40 * (head.top[1] - head.eye[1]) - 1e-9
    assert a[2] >= head.ear[2] + 0.35 * head.k - 1e-9
    assert np.linalg.norm(a - head.ear) > 0.5 * head.k
    assert a[0] - HR.TENT_MARGIN > 0
    # no part of the skin plan reaches the midline
    hpl = p.repose.horn
    assert a[0] - hpl.plan.tent_radius >= HR.TENT_MARGIN - 1e-9


@assets
def test_the_first_versions_temples_quad_is_the_parietal_eminence_and_old_records_are_read_as_the_new_site(world, bodies):
    base, _ = world
    v = bodies["neutral"]
    cage = base.body_cage(v)
    head = HR.head_frame(base, v, cage)
    old = cage[0][cage[1][K.LEGACY_HORN_FACE]].mean(axis=0)
    assert old[2] < head.ear[2] + 0.40 * head.k                 # behind the frontal bone (0.6 behind the bregma)
    rec = {"id": "horn-1", "kind": "horn", "side": "both",
           "placement": {"preset": "temples", "anchor": {"face": K.LEGACY_HORN_FACE},
                         "offset": K.HORN_SINK, "up_hint": [0.0, 1.0, 0.0]}}
    got = K.instantiate_all([AppendageSpec.from_dict(rec)], base, v)
    new = K.instantiate_all([K.new_spec("horn", side="both", base=base)], base, v)
    assert [p.name for p in got] == ["horn-1.l", "horn-1.r"]
    for a, b in zip(got, new):
        assert np.array_equal(a.mesh().vertices, b.mesh().vertices)
        assert a.anchor[2] > head.ear[2] + 0.40 * head.k
    # a ram record is read as the ram's own site
    rec["params"] = {"shape": "ram"}
    ram = K.instantiate_all([AppendageSpec.from_dict(rec)], base, v)[0]
    assert np.allclose(ram.anchor, cage[0][cage[1][K.PRESET_FACES["horn_temple_crest"]]].mean(axis=0))
    # the legacy parameters still mean what they said (an explicit base radius is the radius on the neutral head)
    rec["params"] = {"base_radius": 0.15, "length": 2.0, "curve": 1.0}
    big = K.instantiate_all([AppendageSpec.from_dict(rec)], base, v)[0]
    assert float(big.repose.horn.model.R) == pytest.approx(0.15)
    assert float(big.repose.horn.model.L) == pytest.approx(2.0)
    # an explicit offset is a slide along the axis, the sentinel is the anatomical default (0)
    rec["params"] = {}
    rec["placement"]["offset"] = 0.0
    zero = K.instantiate_all([AppendageSpec.from_dict(rec)], base, v)[0]
    assert np.array_equal(zero.mesh().vertices, new[0].mesh().vertices)


@assets
@pytest.mark.parametrize("body", list(BODIES))
@pytest.mark.parametrize("shape", SHAPES)
def test_the_axis_is_the_skull_normal_tilted_by_the_shapes_angle(world, bodies, body, shape):
    base, _ = world
    v = bodies[body]
    cage = base.body_cage(v)
    p = left_of(world, v, shape)
    hpl = p.repose.horn
    axis = p.matrix[:3, 2] / np.linalg.norm(p.matrix[:3, 2])
    n = hpl.site.normal
    a = angle(axis, n)
    assert a == pytest.approx(HR.SHAPES[shape]["tilt"], abs=1e-6)
    assert 15.0 <= a <= 45.0
    # the normal is the skull's own (smoothed), not the quad's: within a few degrees of another smoothing
    assert angle(n, HR.smooth_normal(cage[0], cage[1], hpl.site.point, 0.2)) < 9.0
    # the bend plane contains the axis and is perpendicular to it
    bend = p.matrix[:3, 0] / np.linalg.norm(p.matrix[:3, 0])
    assert abs(float(axis @ bend)) < 1e-9


@assets
def test_tilt_zero_is_the_surface_normal_and_a_flat_horn_cannot_skim_the_skull(world, bodies):
    base, _ = world
    v = bodies["neutral"]
    cage = base.body_cage(v)
    for tilt in (0.0, 30.0, 60.0):
        p = left_of(world, v, "curved", tilt=tilt)
        hpl = p.repose.horn
        a = angle(p.matrix[:3, 2], hpl.site.normal)
        assert a == pytest.approx(tilt, abs=1e-6)
    # the lean direction is in the head's own axes: a child's and a tall body's horns lean the same way
    leans = []
    for body in ("tall", "child", "neutral"):
        pp_ = left_of(world, bodies[body], "curved")
        ax = pp_.matrix[:3, 2] / np.linalg.norm(pp_.matrix[:3, 2])
        leans.append(ax)
    for ax in leans:
        assert angle(ax, leans[2]) < 12.0


@assets
@pytest.mark.parametrize("shape", SHAPES)
def test_size_follows_the_head_and_is_limited_by_it(world, bodies, shape):
    base, _ = world
    ratios = {}
    for body, v in bodies.items():
        cage = base.body_cage(v)
        head = HR.head_frame(base, v, cage)
        p = left_of(world, v, shape)
        hpl = p.repose.horn
        R = hpl.site.radius
        hw = hpl.site.halfwidth
        assert R <= HR.HALFWIDTH_CAP * hw + 1e-9
        assert 0.10 <= R / hw <= HR.HALFWIDTH_CAP + 1e-9, (body, R / hw)
        sc = np.cbrt(abs(np.linalg.det(p.matrix[:3, :3])))
        assert sc == pytest.approx(head.k)                                   # lengths scale with the head
        ratios[body] = R / head.k
    # radius per head size is the same on every body unless the head limited it
    vals = np.array(list(ratios.values()))
    assert vals.max() / vals.min() < 1.25
    # and a tall body (a bigger head) has a bigger horn in units
    tall, short = (left_of(world, bodies[b], shape).repose.horn.site.radius for b in ("tall", "short"))
    assert tall > 1.3 * short


@assets
def test_a_huge_base_radius_is_limited_to_the_head_and_keeps_the_tent_off_the_midline(world, bodies):
    base, _ = world
    v = bodies["neutral"]
    p = left_of(world, v, "curved", base_radius=0.6)
    hpl = p.repose.horn
    assert hpl.site.radius <= HR.HALFWIDTH_CAP * hpl.site.halfwidth + 1e-9
    assert hpl.site.radius < 0.6
    assert p.anchor[0] - hpl.plan.tent_radius >= HR.TENT_MARGIN - 1e-9
    # at the brow, near the midline, the margin is what limits it
    q = left_of(world, v, "curved", base_radius=0.35)
    spec = K.new_spec("horn", preset="horn_brow", params={"base_radius": 0.35}, side="left", base=base)
    q = K.instantiate_all([spec], base, v)[0]
    assert q.anchor[0] - q.repose.horn.plan.tent_radius >= HR.TENT_MARGIN - 1e-9
    assert q.repose.horn.site.limited == "midline"


@assets
@pytest.mark.parametrize("shape", SHAPES)
def test_a_pair_is_an_exact_mirror_and_so_are_its_specs(world, bodies, shape):
    base, _ = world
    spec = K.new_spec("horn", params={"shape": shape}, side="both", base=base)
    l, r = K.instantiate_all([spec], base, bodies["neutral"])
    assert (l.name, r.name) == ("horn-1.l", "horn-1.r")
    flip = np.array([-1.0, 1.0, 1.0])
    assert np.allclose(r.mesh().vertices, l.mesh().vertices * flip, atol=1e-12)
    assert np.allclose(r.anchor, l.anchor * flip, atol=1e-12)
    assert r.mesh().integrity_report()["outward"]
    sl, sr = l.junction(), r.junction()
    assert np.allclose(np.asarray(sl.points) * flip, np.asarray(sr.points), atol=1e-9)
    assert np.allclose(sl.radii, sr.radii)
    assert np.allclose(np.asarray(sl.mounds[0].center) * flip, sr.mounds[0].center, atol=1e-9)
    assert np.allclose(np.asarray(sl.mounds[0].direction) * flip, sr.mounds[0].direction, atol=1e-9)
    assert np.allclose(np.asarray(sl.mirrored().points), np.asarray(sr.points), atol=1e-9)
    # a lone right horn is the pair's right one
    one = K.instantiate_all([K.new_spec("horn", params={"shape": shape}, side="right", base=base)], base,
                            bodies["neutral"])[0]
    assert np.allclose(one.mesh().vertices, r.mesh().vertices, atol=1e-12)


@assets
def test_the_horns_clear_the_ear_and_the_head_outside_their_foot(world, bodies):
    base, _ = world
    for shape in SHAPES:
        for body in ("neutral", "tall", "child"):
            v = bodies[body]
            p = left_of(world, v, shape)
            hpl = p.repose.horn
            m = p.mesh().vertices
            inv = np.linalg.inv(p.matrix)
            loc = m @ inv[:3, :3].T + inv[:3, 3]
            far = m[loc[:, 2] > 2.6 * hpl.model.R + 0.3]
            if not len(far):
                continue
            cv = base.body_cage(v)[0]
            d = np.linalg.norm(far[:, None, :] - cv[None, ::4, :], axis=2).min() / hpl.site.scale
            assert d >= 0.08, (shape, body, d)                                   # the research's clearance
            assert np.linalg.norm(far - hpl.head.ear, axis=1).min() / hpl.site.scale >= 0.15


@assets
def test_a_ram_curls_back_down_past_the_ear_and_forward_not_over_the_skull(world, bodies):
    base, _ = world
    v = bodies["neutral"]
    p = left_of(world, v, "ram")
    hpl = p.repose.horn
    s, c = hpl.model.centreline(80)
    w = c @ p.matrix[:3, :3].T + p.matrix[:3, 3]
    assert w[:, 2].min() < p.anchor[2] - 0.4 * hpl.site.scale                  # back past the base
    assert w[-1][1] < p.anchor[1] - 0.2 * hpl.site.scale                       # the tip is below the base
    assert w[:, 0].min() >= p.anchor[0] - 0.05                                 # never inward over the vault
    assert w[:, 0].max() > p.anchor[0] + 0.3 * hpl.site.scale                  # it curls OUT, not hugging the head


@assets
def test_the_two_horns_of_a_pair_never_meet_over_the_head(world, bodies):
    """The first corkscrew leaned inward and its tips crossed the midline (and each other): every vertex of a
    horn stays on its own side of the sagittal plane, with the tent's margin at the root."""
    for body in ("neutral", "child", "tall"):
        for shape in SHAPES:
            p = left_of(world, bodies[body], shape)
            x = p.mesh().vertices[:, 0]
            assert x.min() > 0.5 * HR.TENT_MARGIN, (body, shape, x.min())
            assert x[np.argmax(p.mesh().vertices[:, 1])] > 0.5 * p.anchor[0]


# -- the sheath meets the skin ---------------------------------------------------------------------


def shaped_body(world, v, placed):
    base, _ = world
    specs = J.junction_specs(placed)
    out = J.apply_junctions(base, v, specs)
    cv, cq = base.body_cage(out)
    return out, quads_mesh(cv, cq)


@assets
@pytest.mark.parametrize("shape", SHAPES)
def test_the_root_is_buried_in_the_moulded_skin_and_no_triangle_lies_in_it(world, bodies, shape):
    """Acceptance 1 of the research: no flat cap visible (no horn triangle near the skin is coplanar with it),
    every vertex well below the skin is inside the body."""
    base, _ = world
    v = bodies["neutral"]
    p = left_of(world, v, shape)
    hpl = p.repose.horn
    out, body = shaped_body(world, v, [p])
    m = p.mesh()
    inv = np.linalg.inv(p.matrix)
    loc = (m.vertices @ inv[:3, :3].T + inv[:3, 3])
    h = loc[:, 2] - hpl.field.height(loc[:, :2])                              # height over the MOULDED skin
    ra = hpl.model.p.root_radius
    deep = (h < -0.3 * ra) & (np.hypot(loc[:, 0], loc[:, 1]) < 1.6 * ra)         # at the root (a ram's tip passes the neck)
    assert deep.sum() > 30
    assert parity_inside(m.vertices[deep], body).all()
    # the triangles that come out of the skin are not parallel to it
    tri = loc[m.faces]
    cen = tri.mean(axis=1)
    hc = cen[:, 2] - hpl.field.height(cen[:, :2])
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    sn = np.linalg.inv(p.matrix)[:3, :3] @ hpl.site.normal
    sn /= np.linalg.norm(sn)
    cosang = np.abs(n @ sn)
    sel = (np.abs(hc) < 0.25 * ra) & (np.hypot(cen[:, 0], cen[:, 1]) < 1.6 * ra)
    assert sel.sum() > 10
    # no triangle at the skin line is within 5 degrees of the skin's own plane (the old flat cap was at 0)
    assert cosang[sel].max() < math.cos(math.radians(5.0)), cosang[sel].max()


@assets
def test_the_built_sheath_touches_the_real_moulded_skin_at_its_skin_line(world, bodies):
    """The foot ring at the skin line (the second ring above the bowl) of the BUILT mesh lies on the skin the
    body really has after the junction (not on a plane through the anchor): on every body, within a hair."""
    base, _ = world
    for body, v in bodies.items():
        p = left_of(world, v, "curved")
        hpl = p.repose.horn
        out, _ = shaped_body(world, v, [p])
        real = HR.SkinField(base.body_cage(out), p.matrix, 0.9, hpl.field.fallback)
        nv = int(p.params["ring_verts"])
        m = p.mesh()
        inv = np.linalg.inv(p.matrix)
        loc = m.vertices @ inv[:3, :3].T + inv[:3, 3]
        k = len(HR.BOWL_RINGS) + 1                                         # the bowl's rings, then the buried ring
        ring = loc[k * nv:(k + 1) * nv]
        ra = hpl.model.p.root_radius
        assert np.abs(ring[:, 2] - real.height(ring[:, :2])).max() < 0.15 * ra, body
        # and not a plane: the skin's height over the ring varies by more than the plane's, on every body
        plane = np.abs(ring[:, 2] - hpl.field.fallback.height(ring[:, :2])).max()
        assert plane > 0.05 * ra, (body, plane)


@assets
def test_the_skins_foot_reaches_the_moulded_surface_on_every_body(world, bodies):
    """The foot is built against the moulded cage: at the sheath's wall the first foot ring is under the skin
    and the sheath emerges from it (the ring at the skin line is within a hair of the skin)."""
    for body, v in bodies.items():
        p = left_of(world, v, "curved")
        hpl = p.repose.horn
        out, shaped = shaped_body(world, v, [p])
        # the moulded surface the foot was built against is the one the body really has: sample both
        base, _ = world
        f_real = HR.SkinField(base.body_cage(out), p.matrix, 0.9, hpl.field.fallback)
        th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        ra = hpl.model.p.root_radius
        xy = np.stack([1.3 * ra * np.cos(th), 1.3 * ra * np.sin(th)], axis=1)
        assert np.abs(f_real.height(xy) - hpl.field.height(xy)).max() < 1e-9, body


@assets
@pytest.mark.parametrize("shape", SHAPES)
def test_the_skin_is_moulded_locally_without_flips_or_ear_and_midline_drag(world, bodies, shape):
    base, _ = world
    v = bodies["neutral"]
    p = left_of(world, v, shape)
    hpl = p.repose.horn
    info = {}
    spec = p.junction()
    out = J.apply_junctions(base, v, [spec], info)
    cv, cq = base.body_cage(v)
    cs = base.body_cage(out)[0]
    d = np.linalg.norm(cs - cv, axis=1)
    moved = d > 1e-12
    assert info["limited"] == 0 and info["moved"] == int(moved.sum()) and 10 < moved.sum() < 150
    # local: nothing beyond the tent (plus a root radius) from the anchor moves
    far = np.linalg.norm(cv - p.anchor, axis=1) > hpl.plan.tent_radius + hpl.site.radius
    assert (d[far] < 1e-12).all()
    # the ear, the eye and the midline stay where they were
    assert (d[np.linalg.norm(cv - hpl.head.ear, axis=1) < 0.3 * hpl.head.k] < 1e-12).all()
    assert (d[np.abs(cv[:, 0]) < HR.TENT_MARGIN] < 1e-12).all()
    # the tent is real (its height outside the sheath's wall) and bounded by the spec's own cap
    inv = np.linalg.inv(p.matrix)
    loc = cv @ inv[:3, :3].T + inv[:3, 3]
    ra = hpl.model.p.root_radius * hpl.site.scale
    outside = moved & (np.hypot(loc[:, 0], loc[:, 1]) * hpl.site.scale > 1.2 * ra)
    assert d[outside].max() > 0.15 * hpl.plan.tent_height
    assert d.max() <= spec.move_limit() + 1e-9
    # no quad flips or collapses
    m = J.measure_junction(base, v, out, spec, levels=0)
    assert m["flipped"] == 0 and m["min_area_ratio"] > 0.5


@assets
def test_extreme_parameters_do_not_swell_the_head_or_drag_the_ears(world, bodies):
    """The reviewer of S1: horn:flare=2 moved 2803 vertices and the validity pass backed off 1424; length 8 with
    base radius 1.0 swelled the temples and dragged the ears into points."""
    base, _ = world
    v = bodies["neutral"]
    cv, cq = base.body_cage(v)
    head = HR.head_frame(base, v, base.body_cage(v))
    for text in ("horn:flare=1.5,side=left", "horn:length=8,base_radius=0.6,side=left",
                 "horn:shape=ram,flare=1.5,base_radius=0.6,side=left",
                 "horn:shape=ossicone,flare=1.5,base_radius=0.6,length=3,side=left"):
        spec = K.add_from_text(text, [], base=base)
        p = K.instantiate_all([spec], base, v)[0]
        info = {}
        out = J.apply_junctions(base, v, [p.junction()], info)
        cs = base.body_cage(out)[0]
        d = np.linalg.norm(cs - cv, axis=1)
        assert info["moved"] < 300 and info["limited"] < 10, (text, info)
        assert (d[cv[:, 1] < head.eye[1]] < 0.01).all(), text                    # the face and the cheek stay (a ram's tent's edge reaches the temple by < 0.1 cm)
        assert (d[np.linalg.norm(cv - head.ear, axis=1) < 0.3] < 1e-12).all(), text


@assets
def test_the_tent_is_what_the_mound_says_and_comes_from_the_same_numbers_as_the_layers(world, bodies):
    base, _ = world
    v = bodies["neutral"]
    p = left_of(world, v, "curved")
    hpl = p.repose.horn
    spec = p.junction()
    sc = hpl.site.scale
    assert spec.mounds[0].radius == pytest.approx(hpl.plan.tent_radius)
    assert spec.mounds[0].height == pytest.approx(hpl.plan.tent_height)
    assert np.allclose(spec.mounds[0].direction, p.matrix[:3, 2] / sc, atol=1e-9)       # along the axis, not the skin normal
    assert spec.fillet_width == pytest.approx(1.0 * hpl.site.radius)
    assert len(spec.keep) == 2
    # fillet 0 = no junction; boss 0 = the tube only
    assert left_of(world, v, "curved", fillet=0.0).junction() is None
    bare = left_of(world, v, "curved", boss=0.0).junction()
    assert not bare.mounds or bare.mounds[0].height < 1e-9


# -- the layers ----------------------------------------------------------------------------------------


@assets
@pytest.mark.parametrize("shape", SHAPES)
def test_layers_are_bone_muscle_and_skin_from_the_same_parameters(world, bodies, shape):
    base, _ = world
    v = bodies["neutral"]
    p = left_of(world, v, shape)
    L = p.layers()
    assert set(L) == {"bone", "muscle", "skin"}
    assert L["skin"] is p.mesh() or np.array_equal(L["skin"].vertices, p.mesh().vertices)
    for name, mesh in L.items():
        rep = mesh.integrity_report()
        assert rep["watertight"] and rep["outward"] and rep["volume"] > 0, (name, rep)
    again = p.layers()
    for name in L:
        assert np.array_equal(L[name].vertices, again[name].vertices)
    # the layers come from the same parameters: a longer horn has a longer core, the soft tissue (the skin's
    # tent) does not depend on the length
    longer = left_of(world, v, shape, length=HR.SHAPES[shape]["length"] * 1.5)
    ext = lambda m: np.linalg.norm(m.bounds[1] - m.bounds[0])                       # noqa: E731
    assert ext(longer.layers()["bone"]) > 1.08 * ext(L["bone"])                    # (the extent includes the skull plate)
    assert np.allclose(longer.layers()["muscle"].vertices, L["muscle"].vertices, atol=2e-3)     # (the path is sampled over L)


@assets
def test_the_bone_is_a_cone_inside_the_sheath_and_continuous_into_the_skull(world, bodies):
    base, _ = world
    p = left_of(world, bodies["neutral"], "curved")
    hpl = p.repose.horn
    m = hpl.model
    # 0.82 R at the skin, a cone to 0.62 of the length, never nearer the surface than the keratin wall
    assert float(m.core_radius(0.0)) == pytest.approx(HR.CORE_BASE * m.R)
    core_mesh = m.bone(hpl.field).transform(p.matrix)                       # the process at the skin: 0.82..1.12 R wide
    inv = np.linalg.inv(p.matrix)
    loc = core_mesh.vertices @ inv[:3, :3].T + inv[:3, 3]
    at_skin = np.abs(loc[:, 2] - hpl.field.height(loc[:, :2])) < 0.04
    assert at_skin.sum() > 10
    assert HR.CORE_BASE * m.R * 0.9 < np.hypot(loc[at_skin, 0], loc[at_skin, 1]).max() < 1.2 * m.R * (1 + m.p.flare_eff)
    Lc = HR.SHAPES["curved"]["core"] * m.L
    assert float(m.core_radius(0.99 * Lc)) < 0.05 * m.R
    assert float(m.core_radius(1.01 * Lc)) == 0.0
    s = np.linspace(0.0, Lc, 100)
    assert (m.radius(s) - m.core_radius(s) >= HR.CORE_MIN_WALL * m.R - 1e-12).all()
    assert float(m.core_radius(-0.5 * m.R)) > float(m.core_radius(0.0))        # the pedestal widens into the skull
    # inside the sheath: every vertex of the core is inside the skin shell
    assert parity_inside(core_mesh.vertices, p.mesh()).all()


@assets
def test_the_soft_tissue_is_a_lens_under_the_moulded_skin_and_over_the_skull(world, bodies):
    """Its upper surface is the skin the body REALLY has (the junction's moulded cage) less the skin's thickness,
    its lower one the skull's outer table: the layers are derived from the numbers the body is moulded with."""
    base, _ = world
    v = bodies["neutral"]
    p = left_of(world, v, "curved")
    hpl = p.repose.horn
    out, body = shaped_body(world, v, [p])
    sc = hpl.site.scale
    mus = hpl.layers_local()["muscle"]
    inv = np.linalg.inv(p.matrix)
    f_body = HR.SkinField(base.body_cage(out), p.matrix, 1.2, hpl.field.fallback)
    f_raw = HR.SkinField(base.body_cage(v), p.matrix, 1.2, hpl.field.fallback)
    pts = mus.vertices
    top = pts[: len(pts) // 2]
    bot = pts[len(pts) // 2:]
    assert np.abs(top[:, 2] - (f_body.height(top[:, :2]) - HR.SKIN_THICK / sc)).max() < 1e-6
    assert np.abs(bot[:, 2] - (f_raw.height(bot[:, :2]) - HR.SCALP_THICK / sc)).max() < 1e-6
    # thicker at the horn (the tent) than where the skin is not lifted
    thick = top[:, 2] - bot[:, 2]
    rho = np.hypot(top[:, 0], top[:, 1])
    assert thick[rho < 1.5 * hpl.model.R].min() > thick[rho > 0.95 * rho.max()].min()
    # the muscle's whole upper surface is below the skin (parity): inside the shaped body
    world_pts = top[rho > 1.4 * hpl.model.p.root_radius] @ p.matrix[:3, :3].T + p.matrix[:3, 3]
    assert len(world_pts) > 20 and parity_inside(world_pts, body).all()


@assets
def test_ecorche_draws_the_three_layers_and_cuts_them(world, bodies):
    base, _ = world
    v = bodies["neutral"]
    spec = K.new_spec("horn", side="left", base=base)
    placed = K.instantiate_all([spec], base, v)
    body_only = ecorche.layer_shells(base, v, placed, ("skin",), body=True)
    every = ecorche.layer_shells(base, v, placed, ecorche.LAYERS, body=False)
    assert len(every) == 3
    assert len(ecorche.layer_shells(base, v, placed, ("bone",), body=False)) == 1
    cut = ecorche.layer_shells(base, v, placed, ecorche.LAYERS, cut=True, body=False)
    assert len(cut[1].mesh.faces) < len(every[1].mesh.faces)                 # the soft tissue is cut open
    assert len(cut[0].mesh.faces) == len(every[0].mesh.faces)                # the innermost layer stays whole
    assert len(body_only) > 1
    with pytest.raises(ValueError, match="unknown layer"):
        ecorche.parse_layers("bone,fat")


def _load_tool(name):
    path = Path(__file__).resolve().parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@assets
def test_the_horn_view_tool_sections_along_the_axis_and_renders(world, bodies, tmp_path):
    tool = _load_tool("horn_view")
    base, lib = world
    v = bodies["neutral"]
    placed = K.instantiate_all([K.new_spec("horn", side="left", base=base)], base, v)
    plane, view = tool.section_plane(placed)
    p = placed[0]
    axis = p.matrix[:3, 2] / np.linalg.norm(p.matrix[:3, 2])
    bend = p.matrix[:3, 0] / np.linalg.norm(p.matrix[:3, 0])
    assert abs(float(plane[1] @ axis)) < 1e-9 and abs(float(plane[1] @ bend)) < 1e-9     # contains both
    out = tmp_path / "s.png"
    assert tool.main([str(out), "horn:side=left", "--section", "--r", "0.7", "--size", "120x120"]) == 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    out2 = tmp_path / "t.png"
    assert tool.main([str(out2), "horn", "0,10", "--bodies", "child", "--size", "100x100"]) == 0


# -- anatomy of the body, print and documents ----------------------------------------------------------


def test_the_neck_pays_for_the_horns_mass_and_lever_arm_not_the_forehead_sliders():
    eff = ca.CONSEQUENCES["horn"]
    assert "forehead.forehead_nubian" not in eff.sliders and "forehead.forehead_temple" not in eff.sliders
    assert eff.sliders["neck.neck_scale_horiz"] == pytest.approx(0.15)
    assert eff.sliders["neck.neck_scale_depth"] == pytest.approx(0.10)
    horn = K.KINDS["horn"]
    d = horn.defaults()
    assert horn.load(d) == pytest.approx(1.0)                                         # the default pair is the unit
    assert horn.load({**d, "length": 3.2}) == pytest.approx(2.0)                      # the mass: length x radius^2
    assert horn.load({**d, "base_radius": 0.17 * 1.2}) == pytest.approx(1.44)
    assert horn.load({**d, "base_radius": 0.34}) == pytest.approx(K.HORN_LOAD_CAP)    # capped at 2.5
    assert horn.load({**horn.defaults(shape="ram")}) == pytest.approx(K.HORN_LOAD_CAP)
    assert horn.load({"length": 12.0, "base_radius": 0.6}) == pytest.approx(K.HORN_LOAD_CAP)
    # the document path: a record's load is the kind's, times its scale, with the pair as the typical case
    assert ca.appendage_load(K.new_spec("horn")) == pytest.approx(1.0)
    one = K.new_spec("horn", side="left")
    assert ca.appendage_load(one) == pytest.approx(1.0 / ca.PAIR_FACTOR)


@assets
@pytest.mark.parametrize("body", list(BODIES))
def test_the_default_horns_print_clean_with_a_real_weld_and_a_printable_tip(world, bodies, body):
    base, _ = world
    v = bodies[body]
    placed = K.instantiate_all([K.new_spec("horn", side="both", base=base)], base, v)
    shaped, _ = shaped_body(world, v, placed)
    shells = pp.character_print_shells(base, shaped)
    app = [[(p.name, p.mesh())] for p in placed]
    mm = pp.figure_scale(shells + pp.appendage_print_shells(app), size="medium", reference=pp.FIGURE_SHELLS)
    rep = pr.appendage_report(app, base, v, mm, body_verts=shaped)
    assert rep.ok and not rep.blocking, rep.as_text()
    for g in rep.groups.values():
        assert g.root_embed >= 0.2 and g.detached == [], (g.name, g.root_embed)
    tip_mm = 2.0 * HR.HornModel(placed[0].repose.horn.model.p).p.tip * placed[0].repose.horn.site.scale * mm
    assert tip_mm >= 0.2


@assets
def test_building_a_pair_is_fast_and_the_document_round_trips(world, bodies):
    import time

    base, _ = world
    spec = K.new_spec("horn", side="both", base=base)
    t0 = time.perf_counter()
    K.instantiate_all([spec], base, bodies["neutral"])
    assert time.perf_counter() - t0 < 1.0
    again = AppendageSpec.from_dict(spec.to_dict())
    a = K.instantiate_all([spec], base, bodies["neutral"])
    b = K.instantiate_all([again], base, bodies["neutral"])
    for x, y in zip(a, b):
        assert np.array_equal(x.mesh().vertices, y.mesh().vertices)
    # draft: the same horn without the growth rings
    d = K.instantiate_all([spec], base, bodies["neutral"], draft=True)
    assert len(d[0].mesh().faces) < len(a[0].mesh().faces)
    assert d[0].params == a[0].params
