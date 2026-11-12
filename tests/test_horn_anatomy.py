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


def random_face(world, seed, amount=0.35):
    """The morphed body of ``--randomize-face SEED`` (the CLI's own path: ``sliders.randomize`` over the Face group)."""
    from minimaster.character.sliders import Character, SliderCatalog, randomize
    base, lib = world
    cat = SliderCatalog(lib.names)
    c = Character()
    c.sliders.update(randomize(cat, seed=seed, amount=amount))
    return lib.apply(base.verts, c.effective_target_weights(cat))


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
                curve=row["curve"], twist=0.0, turns=row["turns"], tip_radius=row["tip"], flare=row["flare"],
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


def _kr(m):
    """The largest curvature x radius of the FINAL centreline (corkscrew included), measured on a uniform
    resampling of it (the path's own grid has the stations in it)."""
    s, pos, T, N, B = m.path()
    su = np.linspace(0.0, m.L, 801)
    pu = np.stack([np.interp(su, s, pos[:, c]) for c in range(3)], axis=1)
    d = np.gradient(pu, su, axis=0)
    t = d / np.linalg.norm(d, axis=1, keepdims=True)
    kap = np.linalg.norm(np.gradient(t, su, axis=0), axis=1)
    return float((kap * m.taper(su))[2:-2].max())


GRID = [dict()] + [dict(turns=t) for t in (4.0, -4.0)] + [dict(length=12.0), dict(curve=4.0, length=12.0),
                                                           dict(curve=-4.0), dict(turns=4.0, length=0.3)]


@pytest.mark.parametrize("shape", SHAPES)
def test_the_curvature_never_folds_the_loft_and_no_sheath_crosses_itself(shape):
    """The round-1 bend limit was applied BEFORE the corkscrew's helix was added (the default spiral bent at
    2.06 radii per radius and crossed itself 126 times per horn; turns 4 on a curved horn 170-470 times, a ram of
    length 12 coiled into itself 2,352 times) and the test of it zeroed ``turns``. Now: the final path, helix
    included, bends at most 0.66 / radius (a fixed number: raising ``BEND_LIMIT`` to 0.85 fails it) and no triangle of the
    sheath crosses another, for the shape's own turns and for +-4 turns, length 12, curve +-4 and a stub."""
    from minimaster.character import collar as CO
    for kw in GRID:
        m = model(shape=shape, **kw)
        # (round 4: 0.66, a fixed number under the 0.7 limit; the worst of the grid is 0.607 with the keel's lobes and the
        # rings taken off the radius it is measured on. Round 3's 0.8 let BEND_LIMIT 0.85 through, the critic's M4: 0.741)
        assert _kr(m) < 0.66, (shape, kw, _kr(m))
        sk = m.skin()
        assert CO.count_crossings(sk.vertices, sk.faces) == 0, (shape, kw)


def test_a_corkscrew_coils_no_tighter_than_its_thickness_allows():
    """A helix of pitch P bends at most 2 pi / 2P (where its radius equals P / 2 pi, which a corkscrew passes through
    as it grows out of the root): ``turns`` is capped so a coil is at least ``PITCH_MIN`` base radii long, and the
    planar spiral turns at most ``PSI_MAX`` (a lyre ``LYRE_PSI_MAX``)."""
    m = model(shape="curved", turns=4.0)
    assert m.turns_eff == pytest.approx(m.L / (HR.PITCH_MIN * m.R)) and m.turns_eff < 4.0
    # ... and so the corkscrew stays a corkscrew: without the cap the bend limit takes the whole helix in (0.15 local
    # radii off the axis at 70% of the length instead of 0.6)
    s, pos, *_ = m.path()
    s0, pos0, *_ = model(shape="curved", turns=0.0).path()
    u = 0.7 * m.L
    amp = np.linalg.norm(pos[np.searchsorted(s, u)] - pos0[np.searchsorted(s0, u)]) / float(m.taper(u))
    assert amp > 0.4
    assert model(shape="spiral").turns_eff == HR.SHAPES["spiral"]["turns"]          # the default is not capped
    assert model(shape="spiral", turns=-4.0).turns_eff < 0
    assert abs(model(shape="ram", length=12.0).psi) == pytest.approx(HR.PSI_MAX)
    assert abs(model(shape="antelope", curve=4.0).psi) == pytest.approx(HR.LYRE_PSI_MAX)
    assert model(shape="ram").psi == pytest.approx(HR.SHAPES["ram"]["curve"] * 3.0 / 2.0)     # the default keeps its 275 degrees


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


def test_the_foot_is_buried_below_the_skin_has_no_step_and_its_rings_start_above_the_collar():
    """The round-1 groove and bead (0.05 and 0.07 root radii) were invisible at any head-scale view and their test
    could not fail; they are gone: the skin's collar (``collar.py``, ``HornKind.collar_profile``) is the rim now, and the
    sheath under it is smooth. Below the skin the sheath is no wider than its root; above it the foot's rings never
    widen going up (no bead, no step) and the growth rings start ``RING_START`` base radii up (under the collar's top
    edge a ridge made the join ragged)."""
    m = model(flare=0.0)
    field = HR.PlaneField((-math.sin(0.47), 0.0, math.cos(0.47)))
    bowl, foot, shaft_s = m.foot_rings(field)
    ra = float(m.radius(0.0))
    size = [float(np.linalg.norm(r - r.mean(axis=0), axis=1).mean()) for r in foot]
    assert all(b <= a * (1.0 + 1e-6) for a, b in zip(size[1:], size[2:])), size
    v = m.skin(field).vertices
    h = v[:, 2] - field.height(v[:, :2])
    assert np.hypot(v[:, 0], v[:, 1])[h < -0.3 * ra].max() <= ra * 1.001
    s = np.linspace(0.0, m.L, 4000)
    w = m.ring_wave(s)
    assert np.abs(w[s < HR.RING_START * m.R]).max() == 0.0 and w.max() > 0.0
    # the stem under the foot reaches the junction's tube (so the tube the skin is hidden in lies in the sheath)
    assert bowl[-1][:, 2].max() <= -HR.TUBE_DEPTH * m.p.sink + 1e-12


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
    # the normal is the skull's own (smoothed), not the quad's: within a few degrees of another smoothing (12: the
    # smoothing radius is 1.2 base radii, and the round-2 skull scale makes a child's horn 8% bigger than round 1's)
    assert angle(n, HR.smooth_normal(cage[0], cage[1], hpl.site.point, 0.2)) < 12.0
    # it is tilted TOWARD the shape's lean (in the head's own axes), not anywhere on the cone of that angle: a lean
    # turned inward over the vault, or a sign slip, fails here
    head = hpl.head
    lean = head.axes @ np.asarray(HR.SHAPES[shape]["lean"], dtype=np.float64)
    t_want = lean - float(lean @ n) * n
    t_want /= np.linalg.norm(t_want)
    t_got = axis - math.cos(math.radians(a)) * n
    t_got /= np.linalg.norm(t_got)
    assert float(t_got @ t_want) > 0.9999
    # the bend plane contains the axis and is perpendicular to it
    bend = p.matrix[:3, 0] / np.linalg.norm(p.matrix[:3, 0])
    assert abs(float(axis @ bend)) < 1e-9


@assets
def test_tilt_zero_is_the_surface_normal_and_a_flat_horn_cannot_skim_the_skull(world, bodies):
    base, _ = world
    v = bodies["neutral"]
    cage = base.body_cage(v)
    for tilt in (0.0, 30.0, HR.TILT_MAX):            # (round 4: the range ends at 45, it was 60)
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
    """On a symmetric head the right horn, built on ITS OWN site (round 2: the round-1 pair mirrored the left horn,
    and a random face's right horn floated or sank), is the mirror image of the left one to rounding (1e-9: the two
    are computed separately, no longer one from the other)."""
    base, _ = world
    spec = K.new_spec("horn", params={"shape": shape}, side="both", base=base)
    l, r = K.instantiate_all([spec], base, bodies["neutral"])
    assert (l.name, r.name) == ("horn-1.l", "horn-1.r")
    flip = np.array([-1.0, 1.0, 1.0])
    assert np.allclose(r.mesh().vertices, l.mesh().vertices * flip, atol=1e-9)
    assert np.allclose(r.anchor, l.anchor * flip, atol=1e-12)
    assert np.linalg.det(r.matrix[:3, :3]) < 0 < np.linalg.det(l.matrix[:3, :3])     # the mirror IMAGE (a keel of the other hand)
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
    assert np.allclose(one.mesh().vertices, r.mesh().vertices, atol=1e-9)


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
            hpl = p.repose.horn
            inv = np.linalg.inv(p.matrix)
            loc = p.mesh().vertices @ inv[:3, :3].T + inv[:3, 3]
            out = loc[:, 2] > hpl.field.height(loc[:, :2])            # (the buried stem and bowl are inside the head)
            x = p.mesh().vertices[out, 0]
            assert x.min() > 0.5 * HR.TENT_MARGIN, (body, shape, x.min())
            assert x[np.argmax(p.mesh().vertices[out, 1])] > 0.5 * p.anchor[0]


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
        bowl, foot, _ = hpl.model.foot_rings(hpl.field)
        k = len(bowl) + 1                                                  # the bowl's rings and the stem, the buried ring
        ring = loc[k * nv:(k + 1) * nv]
        assert np.allclose(ring, foot[1])
        ra = hpl.model.p.root_radius
        assert np.abs(ring[:, 2] - real.height(ring[:, :2])).max() < 0.15 * ra, body
        # and not a plane: the skin's height over the ring varies by more than the plane's, on every body
        plane = np.abs(ring[:, 2] - hpl.field.fallback.height(ring[:, :2])).max()
        assert plane > 0.05 * ra, (body, plane)


@assets
def test_the_skins_foot_reaches_the_moulded_surface_on_every_body(world, bodies):
    """The foot is built against the skin the FIGURE has: ``hornrig.moulded_cage`` is ``junction.moulded_cage`` (the
    whole pipeline: the head's silhouette bound, the join guard, the validity pass) over the specs of the pair, which
    are the specs ``Placed.junction`` hands the figure. Round 1 re-implemented the raw law and missed the core's
    guards: its field was 0.03 off the real skin. Now the two are the same to rounding, for a lone horn and for each
    horn of a pair (the figure with both horns), on every body and every shape."""
    base, _ = world
    th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    for body, v in bodies.items():
        for text in ("horn:side=left", "horn", "horn:shape=ram", "horn:shape=ossicone"):
            if body != "neutral" and text != "horn":
                continue
            placed = place(world, v, text)
            out, _ = shaped_body(world, v, placed)          # the figure: every horn's spec through apply_junctions
            for p in placed:
                hpl = p.repose.horn
                drawn = base.body_cage(out)
                if hpl.model.p.fillet_foot:
                    # (round 4: the ossicone's fillet meets the skin at a few degrees and is fitted to the skin as it is
                    # DRAWN, the figure's cage subdivided once)
                    from minimaster.core.subdiv import subdivide_quads
                    drawn = subdivide_quads(drawn[0], drawn[1], 1)
                f_real = HR.SkinField(drawn, p.matrix, 0.9, hpl.field.fallback)
                ra = hpl.model.p.root_radius
                for k in (1.1, 1.3, 1.8):
                    xy = np.stack([k * ra * np.cos(th), k * ra * np.sin(th)], axis=1)
                    assert np.abs(f_real.height(xy) - hpl.field.height(xy)).max() < 1e-9, (body, text, p.name)
                # and the spec the foot was built against is the one the figure gets
                assert p.junction() is hpl.spec or p.junction().digest() == hpl.spec.digest()


@assets
def test_the_junctions_tube_lies_inside_the_built_sheath(world, bodies):
    """The horn's spec has ``conform=False`` (the core cannot cut its tube to a mesh that does not exist yet when the
    foot is built against the moulded skin), so the tube must lie in the sheath by construction
    (``HornModel.inscribed``: the minor semi-axis less the keel's lobes and the polygon, the flare measured from the
    low side of the skin, and a stem that reaches the tube's deepest station): every sample of it is inside the built
    mesh, on every shape, both sides, the bodies and random faces."""
    from minimaster.character import junction_checks as JC
    base, lib = world
    cases = [(bodies[b], s) for b in ("neutral", "child", "tall") for s in SHAPES]
    for seed in (0, 9):
        cases.append((random_face(world, seed), "curved"))
    for v, shape in cases:
        for p in place(world, v, f"horn:shape={shape}"):
            spec = p.junction()
            assert spec.conform is False and spec.owner == "head"
            m = p.mesh()
            res = JC.tube_in_mesh(spec.points, spec.radii, m.vertices, m.faces)
            assert res["worst"] == 1.0 and res["max_outside"] == 0.0, (shape, p.name, res["worst"])


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
    # (round 2: the core's 8% head silhouette bound cut the ram's tent into a step and backed 2 of its vertices off;
    # the ram's spec sets its own bound, SHAPES["ram"]["silhouette"]: 0 again; the other shapes keep the core's)
    assert info["limited"] == 0 and info["moved"] == int(moved.sum()) and 10 < moved.sum() < 150
    assert spec.silhouette == HR.SHAPES[shape].get("silhouette", HR.HEAD_SILHOUETTE) and spec.owner == "head"
    assert not spec.conform
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
    # (the path is sampled over L; a corkscrew's coil starts within the junction tube's reach and its pitch is L / turns,
    # so its tube, hence the moulded skin over the lens, moves by a few thousandths)
    # (round 4: the lens is pressed under the DRAWN skin, ``tuck_under_skin``, whose fillet next to the root follows the
    # tube: on a corkscrew one rim vertex of 448 that sits at the threshold moves 0.04; the rest as before)
    tol = 2e-3 if not HR.SHAPES[shape]["turns"] else 2e-2
    d = np.linalg.norm(longer.layers()["muscle"].vertices - L["muscle"].vertices, axis=1)
    assert np.percentile(d, 99) < tol and d.max() < 0.05, (shape, np.percentile(d, 99), d.max())


def bone_rings(p):
    """The cornual process of a placed horn as rings ``(n, ring_verts, 3)`` in its local frame, from the skull's outer
    table up to the start of its tip (the bowl under the table and the tip's cap left out), as ``HornPlacement`` builds it."""
    hpl = p.repose.horn
    m = hpl.model
    sc = hpl.site.scale
    table = HR.SCALP_THICK / sc + HR.PLATE_GAP
    mesh = m.bone(hpl.field, plate=lambda xy: hpl.raw_field.height(xy) - table, skin_thick=HR.SKIN_THICK / sc)
    nv = int(m.p.ring_verts)
    cap = len(m.p.row.get("cap", HR.TIP_CAP)) if m.p.row["material"] == "skin" else len(HR.TIP_CAP)
    n = (len(mesh.vertices) // nv)
    rings = mesh.vertices[: n * nv].reshape(n, nv, 3)
    return rings[len(HR.BOWL_RINGS):n - cap], mesh


@assets
@pytest.mark.parametrize("shape", ["curved", "ram", "antelope", "spiral"])
def test_the_bone_is_a_cone_that_flares_into_the_skull_with_no_step(world, bodies, shape):
    """A05 (goat skull, scale bar): the cornual process is a linear cone, 0.75 / 0.50 / 0.25 of its base at 25 / 50 / 75 %
    of its length, that grows out of the frontal bone with no step. Round 3's was a pedestal 1.12 R wide under the skin
    that stepped down to the cone at the foot's top (the critic's 'plumbing reducer', ref_vs_render.png top row).
    Measured on the built bone: along every generatrix from the skull's outer table up, the section never widens going
    up and turns by at most the flare's arc step (10 degrees) plus the path's own curl (round 3: a flat shoulder, 90 degrees
    in one ring); it starts on the
    table (a concave flare tangent to it, 1.4 base radii wide) and the cone's widths are A05's."""
    base, _ = world
    for body in ("neutral", "child"):
        p = left_of(world, bodies[body], shape)
        hpl = p.repose.horn
        m = hpl.model
        rings, _ = bone_rings(p)
        rings = rings[1:]                                  # (from the table up: the first ring is tucked into it)
        centres = rings.mean(axis=1)
        widths = np.linalg.norm(rings - centres[:, None, :], axis=2).mean(axis=1)
        assert (np.diff(widths) <= 1e-6 * m.R).all(), (shape, body, np.round(widths / m.R, 3))
        worst = max(float(np.max(np.degrees(np.arccos(np.clip(
            (np.diff(rings[:, j], axis=0)[:-1] / np.linalg.norm(np.diff(rings[:, j], axis=0)[:-1], axis=1)[:, None]
             * np.diff(rings[:, j], axis=0)[1:] / np.linalg.norm(np.diff(rings[:, j], axis=0)[1:], axis=1)[:, None]).sum(1),
            -1.0, 1.0))))) for j in range(0, rings.shape[1], 3))
        assert worst < 20.0, (shape, body, worst)               # (the ram curls from its root: 16 degrees a ring there)
        # the flare: 1.4 base radii wide on the table (CORE_BASE + CORE_FLARE), the cone 0.82 at the skin point
        # (a fixed number: the cone alone is 0.82 R at the skin point; the flare makes it 1.42 R on the table, more on a
        # tilted one)
        assert widths[0] > 1.15 * m.R, (shape, body, widths[0] / m.R)
        Lc = HR.SHAPES[shape]["core"] * m.L
        for u, want in ((0.25, 0.75), (0.5, 0.5), (0.75, 0.25)):
            assert float(m.core_radius(u * Lc)) / float(m.core_radius(0.0)) == pytest.approx(want, abs=0.06), (shape, u)
        assert float(m.core_radius(1.01 * Lc)) == 0.0
        s = np.linspace(0.0, Lc, 100)
        assert (m.radius(s) - m.core_radius(s) >= HR.CORE_MIN_WALL * m.R - 1e-12).all()
        # inside the sheath above the skin: every bone vertex over the moulded skin is in the sheath's shell
        bone = p.layers()["bone"]
        inv = np.linalg.inv(p.matrix)
        loc = bone.vertices @ inv[:3, :3].T + inv[:3, 3]
        over = loc[:, 2] > hpl.field.height(loc[:, :2]) + 0.02
        assert over.sum() > 50 and parity_inside(bone.vertices[over], p.mesh()).all(), (shape, body)


@assets
def test_the_ossicones_bone_is_a_column_with_a_dome_not_a_spike(world, bodies):
    """A29 / A30 (live giraffes: a skin-covered column ending in a dome) and A16 (the skull: knobbed bone tips): the bone
    fills the ossicone a skin thickness under its surface, up to a dome under the skin's dome. Round 3's was a fat cone
    with a thin spike on top (the critic's ref_vs_render.png, layers_oss.png): 0.1 of its base wide at 90 % of its height."""
    p = left_of(world, bodies["neutral"], "ossicone")
    hpl = p.repose.horn
    m = hpl.model
    rings, mesh = bone_rings(p)
    centres = rings.mean(axis=1)
    widths = np.linalg.norm(rings - centres[:, None, :], axis=2).mean(axis=1)
    col = widths[np.argmax(centres[:, 2] > 0.3 * m.L):]
    assert col[-1] > 0.6 * col[0], np.round(col / m.R, 3)                       # a column, not a cone to a point
    t = m.bone_inset(HR.SKIN_THICK / hpl.site.scale)
    assert 0.2 * m.R <= t <= 0.35 * m.R
    top, skin_top = float(mesh.vertices[:, 2].max()), float(hpl.skin().vertices[:, 2].max())
    assert skin_top - 2.5 * t < top < skin_top - 0.5 * t, (top, skin_top, t)   # its dome about a skin thickness under the skin's
    from minimaster.character import junction_checks as JC
    sk = hpl.skin()
    assert JC.Shells(sk.vertices, sk.faces).inside(mesh.vertices[mesh.vertices[:, 2] > 0.1 * m.L]).all()


@assets
@pytest.mark.parametrize("shape", SHAPES)
def test_the_layers_are_under_the_drawn_skin_by_true_distance(world, bodies, shape):
    """Every vertex of the soft tissue and of the bone (the cornual process, its flare and the skull's outer table) is
    either in the sheath (the bone above the skin) or at least most of ``LAYER_DEPTH`` under the skin the figure is
    DRAWN with, by the true closest-point distance to the shaped body subdivided once and twice, on three bodies. Round 3
    tested a proxy (the field the lens is built from, a 3 % / 0.2 R allowance) and the ram's lens and plate stood up to
    0.1 base radii out of the temple; its own lens-under-the-moulded-skin and top-over-bottom checks were true by
    construction and are gone. The lens is thickest at the horn (the tent: the skin is lifted there, the skull is not)."""
    from minimaster.character import collar as CO
    from minimaster.character import junction_checks as JC
    base, _ = world
    for body in ("neutral", "child", "tall"):
        v = bodies[body]
        placed = place(world, v, f"horn:shape={shape}")
        out, _ = shaped_body(world, v, placed)
        for lv, depth in ((1, 0.9 * HR.LAYER_DEPTH), (2, 0.5 * HR.LAYER_DEPTH)):
            skin = CO.shaped_skin(base, out, lv)
            for p in placed:
                hpl = p.repose.horn
                patch = CO.SkinPatch(skin[0], skin[1], p.anchor, 8.0 * hpl.site.radius + 0.6)
                sheath = JC.Shells(p.mesh().vertices, p.mesh().faces)
                for name in ("muscle", "bone"):
                    P = p.layers()[name].vertices
                    _, _, sd, _ = patch.closest(P)
                    ok = sd <= -depth
                    if name == "bone":
                        ok |= sheath.inside(P)
                    assert ok.all(), (shape, body, lv, p.name, name, int((~ok).sum()), float(sd[~ok].max()))
        p = placed[0]
        hpl = p.repose.horn
        mus = hpl.layers_local()["muscle"].vertices[: 2 * 7 * 32]          # the horn's own lens (a ram's has its temporalis too)
        # ... and it stops short of the eye (the junction's keep there): the ram's reached 0.49-0.53 of the keep's radius
        # from the eye's centre, into the eyelid's fold (now 0.79-0.84)
        eye_c, eye_r = hpl.keeps[1]
        assert (np.linalg.norm(mus - np.asarray(eye_c), axis=1) / eye_r).min() > 0.7, (shape, body)
        top, bot = mus[: len(mus) // 2], mus[len(mus) // 2:]
        thick = top[:, 2] - bot[:, 2]
        rho = np.hypot(top[:, 0], top[:, 1])
        assert thick[rho < 1.5 * hpl.model.R].mean() > 1.5 * thick[rho > 0.95 * rho.max()].mean()


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


# -- round 2: a random face, the skull's size, the extremes, the print ------------------------------------------


@assets
@pytest.mark.parametrize("seed", [0, 9, 4, 11])
def test_each_horn_of_a_pair_sits_on_its_own_skin_on_a_random_face(world, seed):
    """``--randomize-face`` makes the head asymmetric: round 1 built the right horn as the mirror of the left one and on
    seeds 0 and 9 it sank 1.9 base radii into the scalp or floated 1.7 above it (the critic's measure: the unmoulded
    skin's height over the horn's origin along its axis, in root radii). Each horn is now built on its own quad, its
    own skull normal and its own skin: both origins are within 0.3 root radii of the skin, and the right horn is still
    the mirror IMAGE of a left one (the same model, a mirroring frame)."""
    base, _ = world
    v = random_face(world, seed)
    l, r = place(world, v, "horn")
    cage = base.body_cage(v)
    for p in (l, r):
        hpl = p.repose.horn
        f = HR.SkinField(cage, p.matrix, 0.9, hpl.field.fallback)
        z0 = float(f.height(np.zeros((1, 2)))[0])
        assert abs(z0) < 0.3 * hpl.model.p.root_radius, (seed, p.name, z0 / hpl.model.p.root_radius)
        # the foot's skin-line ring is on that side's moulded skin (to the two passes that lower it: 0.1 root radii)
        bowl, foot, _ = hpl.model.foot_rings(hpl.field)
        assert np.abs(foot[1][:, 2] - hpl.field.height(foot[1][:, :2])).max() < 0.1 * hpl.model.p.root_radius
    assert l.repose.horn.model is r.repose.horn.model
    assert np.linalg.det(r.matrix[:3, :3]) < 0
    # and the head IS asymmetric there: the right anchor is not the mirror of the left one
    assert np.linalg.norm(r.anchor * np.array([-1.0, 1.0, 1.0]) - l.anchor) > 1e-4


@assets
def test_the_horn_is_sized_by_the_skull_not_the_eyes(world, bodies):
    """Round 1 scaled the horn by the eye distance: ``eyes.eye_trans.in_out`` -1..1 changed it 0.89..1.17 on the same
    skull and a random face 0.54..1.28 where the skull was 1.28 wide (horns 0.43-0.92 x what the skull implies, which
    then printed fragile). The scale is now the skull's half-width over the neutral one (``skull_halfwidth``): moving
    the eyes changes nothing, a wider skull (``head.head_scale_horiz``) a proportionally bigger horn, and on random
    faces the base radius stays the same share of the skull unless the head limits it (and then the whole horn keeps
    its proportions)."""
    from minimaster.character.sliders import Character, SliderCatalog
    base, lib = world
    cat = SliderCatalog(lib.names)

    def body(**sl):
        c = Character()
        c.sliders.update(sl)
        return lib.apply(base.verts, c.effective_target_weights(cat))

    neutral = bodies["neutral"]
    k0 = HR.head_frame(base, neutral, base.body_cage(neutral)).k
    assert k0 == pytest.approx(1.0, abs=0.01)
    ks = []
    for io in (-1.0, 1.0):
        v = body(**{"eyes.eye_trans.in_out": io})
        h = HR.head_frame(base, v, base.body_cage(v))
        assert h.iod / HR.IOD_REF == pytest.approx(1.0 + 0.16 * io, abs=0.06)            # the eyes did move ...
        ks.append(h.k)
    assert ks[0] == pytest.approx(k0, abs=1e-9) and ks[1] == pytest.approx(k0, abs=1e-9)   # ... the horn's scale did not
    wide = body(**{"head.head_scale_horiz": 1.0})
    hw = HR.head_frame(base, wide, base.body_cage(wide))
    assert hw.k == pytest.approx(k0 * hw.skull / HR.SKULL_REF / k0, rel=1e-9) and hw.k > 1.15 * k0
    shares = []
    for seed in (0, 9, 4, 11, 7, 3):
        p = place(world, random_face(world, seed), "horn:side=left")[0]
        hpl = p.repose.horn
        assert hpl.site.scale == pytest.approx(hpl.head.skull / HR.SKULL_REF)
        if not hpl.site.limited:
            shares.append(hpl.site.radius / hpl.head.skull)
        # a limited horn keeps its proportions: length / radius as asked
        assert hpl.model.L / hpl.model.R == pytest.approx(1.6 / 0.17, rel=1e-6)
    assert len(shares) >= 5 and max(shares) / min(shares) < 1.05
    # Round 3 asserted here that seeds 0 and 9 are limited by the midline ("a narrow brow"). They are not: the face
    # moves the whole head 0.16 sideways (the joints, the eyes and every slab of the skull together), the skull was
    # measured as its largest |x| from WORLD x = 0 (22 % too wide) and the midline margin from world x = 0 too (the
    # left horn is 0.33 from x = 0 but 0.49 from the head's own midline). The scale is the skull's own half-width (half
    # the slab's width) and nothing limits the default horn there.
    from minimaster.character import appendages as A
    for seed in (0, 9):
        v = random_face(world, seed)
        p = place(world, v, "horn:side=left")[0]
        hpl = p.repose.horn
        mid = float(A.joint_position(base, "head", v)[0])
        assert mid < -0.1                                                        # the head IS moved sideways
        cv = base.body_cage(v)[0]
        h = float(hpl.head.top[1] - hpl.head.root[1])
        xs = cv[np.abs(cv[:, 1] - (hpl.head.root[1] + HR.SKULL_AT * h)) < HR.SKULL_SLAB * h, 0]
        half = 0.5 * float(xs.max() - xs.min())
        assert np.abs(xs).max() > 1.15 * half                                    # (what round 3 took for the width)
        assert hpl.site.scale == pytest.approx(half / HR.SKULL_REF, rel=1e-9)
        assert abs(p.anchor[0] - mid) > 0.4 and hpl.site.limited == ""
        assert hpl.site.radius == pytest.approx(0.17 * hpl.site.scale, rel=1e-9)


@assets
def test_a_head_moved_sideways_moves_its_horns_and_changes_nothing_else(world, bodies):
    """``--randomize-face`` moves the whole head sideways (seeds 0 and 9 by -0.16, seed 57 at 0.6 by +0.17: the
    skull-base and crown joints, the eyes and every slab of the skull together). Round 3 measured the skull's width,
    the head's width at the site and the distance from the midline (the radius limit, the tent's and the fillet's
    margins) from WORLD x = 0, so a sideways move changed the horn's size and what limited it. A rigid sideways shift
    of the body now translates both horns, to rounding, and changes no size, no limit and no shape (a limited one
    included: the brow site with a big radius is limited by the midline on both)."""
    base, _ = world
    v = bodies["neutral"]
    d = np.array([0.16, 0.0, 0.0])
    for text in ("horn", "horn:shape=ram", "horn:preset=horn_brow,base_radius=0.3"):
        a, b = place(world, v, text), place(world, v + d, text)
        for p, q in zip(a, b):
            hp, hq = p.repose.horn, q.repose.horn
            assert hq.head.mid == pytest.approx(hp.head.mid + 0.16, abs=1e-12)
            assert hq.head.k == pytest.approx(hp.head.k, abs=1e-12), text
            assert hq.site.radius == pytest.approx(hp.site.radius, abs=1e-12) and hq.site.limited == hp.site.limited
            assert hq.plan.tent_radius == pytest.approx(hp.plan.tent_radius, abs=1e-12)
            assert hq.plan.fillet == pytest.approx(hp.plan.fillet, abs=1e-12)
            assert np.allclose(q.mesh().vertices - d, p.mesh().vertices, atol=1e-9), (text, p.name)
    assert place(world, v, "horn:preset=horn_brow,base_radius=0.3")[0].repose.horn.site.limited == "midline"


@assets
def test_a_limited_horn_says_so_in_the_report(world, bodies):
    """A saved ``base_radius`` the head cannot carry (the first version's 0.42) was limited silently to about a third
    of its size; the build report says it now (a note, not a warning: every parameter at its limit must still build
    with warnings as errors), and the default horn says nothing."""
    base, _ = world
    report = []
    K.instantiate_all([K.new_spec("horn", params={"base_radius": 0.42}, base=base)], base, bodies["neutral"],
                      report=report)
    assert len(report) == 1 and "built at" in report[0] and "horn-1" in report[0]
    report = []
    first = K.new_spec("horn", base=base)
    K.instantiate_all([first, K.new_spec("horn", [first], params={"shape": "ram"}, base=base)], base,
                      bodies["neutral"], report=report)
    assert report == []                         # (the default ram is limited to 0.99 by the head's width: no note)


@assets
@pytest.mark.parametrize("shape", SHAPES)
def test_extremes_build_clean_skin_and_sheaths_that_do_not_cross(world, bodies, shape):
    """The critic's grid, on the body: fillet and turns were not in the round-1 extremes (a ram at fillet 3 moved 403
    vertices and backed 10 off; turns +-4 crossed itself 170-470 times; a ram of length 12, 2,209). Every extreme of
    every shape builds a pair whose sheaths cross neither themselves nor each other, a valid junction (no quality
    violation, fewer than 300 vertices moved and 25 backed off for the pair: the head's own skin only).
    Round 4 (the round-3 critic: the grid had no tilt, ring_verts or twist, and all three broke at UI values): the
    tilt at its new maximum (45; 60 split the cuff and laid a keratin tongue on the scalp), the fewest facets (24) and
    twist +-3 (an antelope at twist 3 crossed itself 166 times, a curved pair met over the head with 140 crossing faces,
    a curved horn of length 12 at twist 3 crossed itself 3,884 times and its partner 3,384 times)."""
    from minimaster.character import collar as CO
    base, _ = world
    v = bodies["neutral"]
    for e in ("fillet=4", "flare=1.5", "base_radius=0.6", "length=12", "turns=4", "turns=-4", "boss=2", "collar=2",
              "base_radius=0.6,flare=1.5,fillet=4", f"tilt={HR.TILT_MAX:g}", f"ring_verts={HR.RING_VERTS_MIN}",
              "twist=3", "twist=-3", "twist=3,length=12", f"tilt={HR.TILT_MAX:g},twist=-3,ring_verts={HR.RING_VERTS_MIN}"):
        placed = place(world, v, f"horn:shape={shape},{e}")
        info = {}
        out = J.apply_junctions(base, v, J.junction_specs(placed), info)
        q = J.junction_quality(base, v, out, J.junction_specs(placed), levels=())
        assert J.quality_violations(q) == [], (shape, e)
        assert info["moved"] < 300 and info["limited"] < 25, (shape, e, info["moved"], info["limited"])
        a, b = (p.mesh() for p in placed)
        assert CO.count_crossings(a.vertices, a.faces) == 0, (shape, e)
        assert CO.count_crossings(a.vertices, a.faces, b.vertices, b.faces) == 0, (shape, e)
        assert placed[0].repose.horn.plan.fillet <= HR.FILLET_MAX
        # and each sheath stays on its own side of the head's midline (the pair never meets over the head)
        for p in placed:
            hpl = p.repose.horn
            side = -1.0 if hpl.site.right else 1.0
            inv = np.linalg.inv(p.matrix)
            loc = p.mesh().vertices @ inv[:3, :3].T + inv[:3, 3]
            up = p.mesh().vertices[loc[:, 2] > hpl.field.height(loc[:, :2])]
            assert ((up[:, 0] - hpl.head.mid) * side).min() > 0.5 * HR.TENT_MARGIN, (shape, e, p.name)


@assets
def test_the_ui_ranges_stop_where_the_horn_breaks_and_old_records_beyond_them_still_build(world, bodies):
    """Round 3: the tilt slider went to 60 (the cuff split into two lobes with a skin-coloured crack across the horn,
    and the low side of the sheath lay along the scalp as a keratin tongue) and ring_verts down to 5 (a hexagonal
    sheath tore the cuff into teeth). The ranges stop at 45 and 24; typing a value past them is an error, and a saved
    record that has one builds at the limit with a warning that says so (the record keeps its value)."""
    base, _ = world
    horn = K.KINDS["horn"]
    for shape in SHAPES:
        assert horn.param("tilt", {"shape": shape}).hi == HR.TILT_MAX == 45.0
        assert horn.param("ring_verts", {"shape": shape}).lo == HR.RING_VERTS_MIN == 24
    with pytest.raises(ValueError, match="outside the range"):
        K.add_from_text("horn:tilt=60", [], base=base)
    with pytest.raises(ValueError, match="outside the range"):
        K.add_from_text("horn:ring_verts=6", [], base=base)
    rec = {"id": "horn-1", "kind": "horn", "side": "left", "params": {"tilt": 60, "ring_verts": 6}}
    spec = AppendageSpec.from_dict(rec)
    with pytest.warns(K.AppendageWarning) as got:
        p = K.instantiate_all([spec], base, bodies["neutral"])[0]
    text = " ".join(str(w.message) for w in got)
    assert "tilt = 60" in text and "ring_verts = 6" in text
    assert p.params["tilt"] == 45.0 and p.params["ring_verts"] == 24
    assert angle(p.matrix[:3, 2], p.repose.horn.site.normal) == pytest.approx(45.0, abs=1e-6)
    assert spec.to_dict()["params"] == {"tilt": 60, "ring_verts": 6}            # the document is not rewritten


@assets
@pytest.mark.parametrize("shape", SHAPES)
def test_the_default_of_every_shape_prints_at_small_and_medium(world, bodies, shape):
    """Round 1 tested the print of the curved default at "medium" only; the spiral and the antelope were fragile there
    (28% and 33% of their area under 0.3 mm). Every shape's default pair now prints at "medium" with no blocking
    finding and no thin-area warning, and at "small" with no blocking finding (the slender ones warn there: the
    report says so), with a real weld and a tip the printer resolves (measured on the mesh: the last ring)."""
    base, _ = world
    v = bodies["neutral"]
    placed = place(world, v, f"horn:shape={shape}")
    shaped, _ = shaped_body(world, v, placed)
    shells = pp.character_print_shells(base, shaped)
    app = [[(p.name, p.mesh())] for p in placed]
    for size in ("small", "medium"):
        mm = pp.figure_scale(shells + pp.appendage_print_shells(app), size=size, reference=pp.FIGURE_SHELLS)
        rep = pr.appendage_report(app, base, v, mm, body_verts=shaped)
        assert not rep.blocking, (shape, size, rep.as_text())
        if size == "medium":
            assert not [f for f in rep.findings if f.code == "thin"], (shape, rep.as_text())
        for g in rep.groups.values():
            assert 0.15 <= g.root_embed <= 0.75, (shape, g.root_embed)
        hpl = placed[0].repose.horn
        s, pos, T, N, B = hpl.model.path()
        tip_ring = hpl.model.skin().vertices
        far = np.linalg.norm(tip_ring - pos[-1], axis=1) < 1.2 * float(hpl.model.radius(hpl.model.L))
        tip_mm = 2.0 * np.linalg.norm(tip_ring[far] - tip_ring[far].mean(axis=0), axis=1).max() * hpl.site.scale * mm
        assert tip_mm >= 0.2, (shape, size, tip_mm)


# -- round 4: the facets the cuff meets, the temporalis under a ram, a document of round 3 --------------------------------


FACET_DEPTH = 0.004      # units: 24 facets are 0.0022-0.0036 deep (the ram's 32: 0.0020-0.0030), 16 facets 0.005-0.008


@assets
def test_the_sheath_is_round_enough_where_its_cuff_meets_it(world, bodies):
    """Round 3's mutation M7 (the default 24 facets back to 16) survived every geometric test; only a literal check of
    the default caught it. Measured on the BUILT sheath, where the skin's cuff meets it (the rings up to 1.2 base radii
    over the skin): the depth of the polygon's facets (their sagitta: a quarter of a corner's distance from the chord
    of its two neighbours), in world units, is under ``FACET_DEPTH`` on three bodies. The cuff meets the sheath nearly
    tangentially, so a facet's depth becomes a scallop of the contact line about 20 times as long (the critic's 0.16 R
    from 0.0086 R) and a coarse polygon's corners stand out through the cuff as teeth. 24 sides keep the curved, spiral
    and antelope horns under it, the ram needs its 32 (24 would be 0.0048 on the tall body); a 16-gon fails every shape."""
    from minimaster.character import collar as CO
    for shape in ("curved", "ram", "spiral", "antelope"):
        for body in ("neutral", "tall", "child"):
            p = left_of(world, bodies[body], shape)
            hpl = p.repose.horn
            m = hpl.model
            nv = int(m.p.ring_verts)
            sk = hpl.skin().vertices
            rings = sk[: (len(sk) // nv) * nv].reshape(-1, nv, 3)
            c = rings.mean(axis=1)
            h = c[:, 2] - hpl.field.height(c[:, :2])
            sel = (h > 0.0) & (h < 1.2 * m.R)
            assert sel.sum() >= 3, (shape, body)
            # the sagitta of a facet: a quarter of a corner's distance from the chord of its two neighbours
            r = rings[sel]
            a, b = np.roll(r, 1, axis=1), np.roll(r, -1, axis=1)
            chord = b - a
            off = (r - a) - (((r - a) * chord).sum(axis=2) / (chord * chord).sum(axis=2))[..., None] * chord
            depth = 0.25 * np.linalg.norm(off, axis=2).max() * hpl.site.scale
            assert depth < FACET_DEPTH, (shape, body, nv, depth)


@assets
def test_a_ram_sits_on_a_temporalis_the_skin_swells_over(world, bodies):
    """A horn on the temple crest sits on the temporalis (H05: the fan fills the temporal fossa under the superior
    temporal line; H14: a strong one bulges the temple), round 3's critic: 'no temporal swelling under the root' (the
    swell tried in round 2 was taken out). The ram's spec has a second mound down and back from the root, along the skin's
    normal: on the body it lifts the skin over the fossa outward (by 0.006 units or more where the muscle is, against the
    same spec without it; the ram's head bound, 0.12, holds it to about half the 0.032 asked), it does not touch the eye
    or the ear, and the muscle layer has the temporalis there (under the drawn skin: the layers' containment test). A
    horn at the crown has none."""
    base, _ = world
    for body in ("neutral", "child"):
        v = bodies[body]
        placed = place(world, v, "horn:shape=ram")
        specs = J.junction_specs(placed)
        assert all(len(s.mounds) == 2 for s in specs)
        bare = [s.with_(mounds=s.mounds[:1]) for s in specs]
        a, b = J.apply_junctions(base, v, specs), J.apply_junctions(base, v, bare)
        cv = base.body_cage(v)[0]
        lift = base.body_cage(a)[0] - base.body_cage(b)[0]
        for p in placed:
            hpl = p.repose.horn
            ctr, n, fwd = hpl.temporalis
            near = np.linalg.norm(cv - ctr, axis=1) < 0.12 * hpl.head.k
            assert near.sum() >= 3 and float((lift[near] @ n).mean()) > 0.006, (body, p.name, float((lift[near] @ n).mean()))
            mus = p.layers()["muscle"].vertices
            assert np.linalg.norm(mus - ctr, axis=1).min() < 0.05
            # down and back from the root, in front of the ear, above the eye's level
            assert float((ctr - hpl.site.point) @ hpl.head.up) < 0.0 and float((ctr - hpl.site.point) @ hpl.head.fwd) < 0.0
        eye_ear = [np.linalg.norm(cv - c, axis=1) < 0.2 for c in (placed[0].repose.horn.head.eye,
                                                                   placed[0].repose.horn.head.ear)]
        for zone in eye_ear:
            assert np.abs(lift[zone]).max() < 1e-3, body
    curved = place(world, bodies["neutral"], "horn")[0]
    assert curved.repose.horn.temporalis is None and len(curved.junction().mounds) == 1


@assets
def test_a_document_saved_by_round_3_still_loads_builds_and_saves_unchanged(world):
    """The critic's own .mmchar written by round 3 (a random face; a ram, and a spiral on the brow with ring_verts 16):
    it loads, both pairs build (the spiral's 16 facets at the new minimum 24, with a warning that says so; the document
    keeps its 16), and it saves byte for byte as it was. (How it renders now: docs/anatomy_horn.md, round 4.)"""
    import warnings

    from minimaster.character.sliders import Character, SliderCatalog
    base, lib = world
    path = Path(__file__).parent / "fixtures" / "horns_round3.mmchar"
    c = Character.load(path)
    assert [a.id for a in c.appendages] == ["horn-1", "horn-2"]
    v = lib.apply(base.verts, c.effective_target_weights(SliderCatalog(lib.names)))
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        placed = c.instantiate_appendages(base, v, lib)
    text = [str(w.message) for w in got if issubclass(w.category, K.AppendageWarning)]
    assert len(text) == 1 and "ring_verts = 16" in text[0] and "horn-2" in text[0], text
    assert [p.name for p in placed] == ["horn-1.l", "horn-1.r", "horn-2.l", "horn-2.r"]
    assert placed[2].params["ring_verts"] == 24 and placed[0].params["ring_verts"] == 32
    for p in placed:
        rep = p.mesh().integrity_report()
        assert rep["watertight"] and rep["outward"], p.name
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "again.mmchar"
        c.save(out)
        assert out.read_text() == path.read_text()
