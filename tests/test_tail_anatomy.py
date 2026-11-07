"""The tail's anatomy: a caudal vertebra chain, the muscle sleeve around it, the skin over both.

``tailrig.TailModel`` builds the three layers from ONE parameter set. The first half of this file needs no
assets (the model alone, in its local frame: x up, y lateral, z along the axis, s = 0 at the skin point of
the root, negative inside the body); the second half places tails on the real morphed bodies.
"""

import math

import numpy as np
import pytest

from minimaster.character import kinds as K
from minimaster.character import mhbase as mh
from minimaster.character import tailrig as T

assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")

R0 = 0.45          # the default base radius
CONTROLS = ("curl", "droop", "lift", "sway")


def model(**kw) -> T.TailModel:
    return T.TailModel(T.TailParams(**kw))


def inside(points, mesh, chunk=256):
    """Even-odd ray test along +X of ``points`` against the closed ``mesh``."""
    tris = mesh.vertices[mesh.faces]
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    v0, v1 = c[:, 1:] - a[:, 1:], b[:, 1:] - a[:, 1:]
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    ok = np.abs(den) > 1e-14
    safe = np.where(ok, den, 1.0)
    out = np.zeros(len(points), dtype=bool)
    points = np.asarray(points) + np.array([0.0, 1.234e-6, 0.8765e-6])     # off the planes the rings lie in
    for s in range(0, len(points), chunk):
        p = points[s:s + chunk]
        v2 = p[:, None, 1:] - a[None, :, 1:]
        d20, d21 = (v2 * v0[None]).sum(2), (v2 * v1[None]).sum(2)
        u = np.where(ok, (d11 * d20 - d01 * d21) / safe, -1.0)
        w = np.where(ok, (d00 * d21 - d01 * d20) / safe, -1.0)
        hit = ok & (u >= 0) & (w >= 0) & (u + w <= 1)
        bx = a[None, :, 0] + u * (c[None, :, 0] - a[None, :, 0]) + w * (b[None, :, 0] - a[None, :, 0])
        out[s:s + chunk] = ((hit & (bx > p[:, None, 0])).sum(1) % 2) == 1
    return out


# -- the layers ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("kw", [{}, dict(length=9.0, base_radius=0.7, droop=1.6, curl=1.2),
                                dict(length=3.0, base_radius=0.3, ring_verts=8, segments=6),
                                dict(lift=60.0, sway=50.0), dict(tip_radius=0.0)])
def test_the_layers_are_closed_solids(kw):
    m = model(**kw)
    for mesh in (m.skin(), m.muscle(), m.bone()):
        assert mesh.is_watertight
        assert mesh.volume() > 0
        assert not len(mesh.degenerate_faces(1e-14))


@pytest.mark.parametrize("kw", [{}, dict(droop=1.8, curl=-1.0), dict(lift=70.0), dict(sway=-80.0, droop=-1.0)])
def test_every_layer_is_inside_the_next_one(kw):
    """Bone inside muscle inside skin, point by point (the vertices of the inner layer lie inside the closed
    outer one): the skin is the envelope, not a second shape that happens to be nearby."""
    m = model(**kw)
    skin, muscle, bone = m.skin(), m.muscle(), m.bone()
    assert inside(muscle.vertices, skin).all()
    assert inside(bone.vertices, muscle).all()
    assert muscle.volume() < skin.volume() and bone.volume() < 0.25 * muscle.volume()


def test_the_skin_is_the_muscle_surface_plus_the_skin_thickness():
    """At every station and every angle the skin ring is further from the axis than the muscle ring by
    about the skin thickness (the lobing is damped on the skin, so it is a band, not an equality). Change
    the thickness of one layer and this fails."""
    m = model()
    s = m.s[m.s <= m.L]
    sk, mu = m.section(s, "skin"), m.section(s, "muscle")
    c = m.centre(s)[:, None, :]
    gap = np.linalg.norm(sk - c, axis=2) - np.linalg.norm(mu - c, axis=2)
    t = m.skin_thickness(s)[:, None]
    assert (gap > 0.4 * t).all() and (gap < 1.6 * t).all()
    # the thickness is the research's: 0.08 R at the root rising to 0.12 of the local radius distally
    assert m.skin_thickness(0.0) == pytest.approx(0.08 * m.law(0.0), rel=1e-9)
    assert m.skin_thickness(m.L) == pytest.approx(max(0.02, 0.12 * m.law(m.L)), rel=1e-9)


def test_the_muscle_bellies_are_on_the_proximal_part_and_tendons_follow():
    m = model()
    t = np.array([0.0, 0.05, 0.12, 0.3, 0.55, 0.8, 1.0])
    g = m.belly(t * m.L)
    assert g[0] == pytest.approx(1.0) and g[2] == pytest.approx(1.0)          # full up to 12% of the length
    assert g[4] == pytest.approx(0.0, abs=1e-9) and g[5] == 0 and g[6] == 0   # none past 55%
    assert (np.diff(g) <= 1e-12).all()


# -- the numbers ---------------------------------------------------------------------------------------------


def test_the_radius_law_the_flare_and_the_aspect():
    m = model()
    R, F = m.R, m.p.flare
    plain = lambda s: m.tip + (R - m.tip) * (1.0 - s / m.L) ** 0.85          # the registry tail's law   # noqa: E731
    assert m.law(0.0) == pytest.approx(R * (1.0 + F))                          # the skin: 1.35 R
    assert m.law(-m.stem) == pytest.approx(R * (1.0 + F))                      # held inside the body
    assert m.law(0.9 * R) == pytest.approx(plain(0.9 * R) * (1.0 + F * math.exp(-1.0)), rel=1e-9)
    for s in (2.0 * R, 0.25 * m.L, 0.5 * m.L, 0.75 * m.L):                    # back to the law by 2 R
        assert m.law(s) == pytest.approx(plain(s), rel=0.05 if s == 2.0 * R else 0.02 if s < 0.3 * m.L else 0.0075)
    assert m.law(m.L) == pytest.approx(m.tip + 0.0)                          # the tip
    # the section is an ellipse measured in the ring frame: wider than tall at the root, taller beyond
    asp = m.aspect(np.array([-1.0, 0.0, R, 2.0 * R, 0.3 * m.L, m.L]))
    assert asp[0] == asp[1] == pytest.approx(1.08) and asp[3] == pytest.approx(1.0)     # (literals: 1.08 at the skin, 1.0 at 2 R ...)
    assert asp[4] == asp[5] == pytest.approx(0.92) and asp[2] < asp[1]                     # ... 0.92 beyond 0.3 L
    ring = m.section(m.s, "skin")[8]                      # a ring of the built skin: its measured aspect
    fr = m.fr[8]
    x, y = (ring - m.centre(m.s[8])) @ fr[:, 0], (ring - m.centre(m.s[8])) @ fr[:, 1]
    assert np.ptp(y) / np.ptp(x) == pytest.approx(m.aspect(m.s[8]), rel=0.04)


def test_the_bone_and_the_skin_taper_like_the_reference_animals():
    """Measured on the reference plates (docs/anatomy_tail.md): the cat's caudal vertebrae are 0.98, 0.59, 0.50,
    0.32, 0.19 of their root width at 25, 50, 75, 95 and 100 % of the length (cm_Anatomical_technology), and a
    soft tail's width is 0.79 / 0.47 / 0.34 (kangaroo outline, cm_PSM_V08_D430) to 0.87 / 0.61 / 0.40 (rhinoceros
    iguana, ov_964703a3) of its base at 25 / 50 / 75 %, up to 0.9 / 0.8 at 50 / 75 % (spider monkey)."""
    m = model()
    t = np.array([0.0, 0.25, 0.5, 0.75, 0.95, 1.0])
    b = m.bone_radius(t * m.L)
    cat = np.array([1.0, 0.98, 0.59, 0.50, 0.32, 0.19])
    assert np.abs(b / b[0] - cat).max() < 0.17
    plain = lambda x: m.tip + (m.R - m.tip) * (1.0 - x) ** 0.85          # noqa: E731
    w = plain(t[:4]) / plain(0.0)
    assert 0.79 - 0.1 <= w[1] <= 0.87 + 0.1 and 0.47 <= w[2] <= 0.9 and 0.34 <= w[3] <= 0.8


def test_the_vertebra_chain_follows_the_research():
    m = model()
    n = m.p.n_vertebrae()
    assert n == round(48 * m.L / T.H_REF) == 17
    assert model(length=2.0).p.n_vertebrae() == 6 and model(length=20.0).p.n_vertebrae() == 30   # 6..30
    assert model(vertebrae=22).p.n_vertebrae() == 22
    e = m.vertebra_edges()
    lens = np.diff(e)
    assert len(lens) == n and (lens > 0).all()
    assert lens[0] / lens.max() == pytest.approx(0.55 * 1.0, abs=0.12)         # Cd1 is the shortest of the proximal ones
    assert abs(np.argmax(lens) / (n - 1) - 0.29) < 0.15                        # longest near 29% of the chain
    assert lens[-1] < 0.5 * lens.max() and (np.diff(lens[np.argmax(lens):]) < 1e-9).all()   # then tapering
    assert e[0] < -0.2 * m.R and e[0] > -m.stem                                 # Cd1 starts inside the body ...
    assert e[-1] < m.L                                                          # ... and the chain ends in the tip
    # a bone mesh that really has the pieces: processes on the first six, chevrons from Cd4
    assert m.bone().volume() > 0 and len(m.bone().faces) > 40 * n


def test_the_first_part_of_the_tail_is_stiff_then_it_droops_in_the_sagittal_plane():
    m = model()
    s_stiff = 2.5 * m.R                                          # the research's stiff proximal part (a literal: not the constant under test)
    t_stiff = m.tangent(np.linspace(-m.stem, s_stiff, 20))
    assert np.degrees(np.arccos(np.clip(t_stiff @ np.array([0, 0, 1.0]), -1, 1))).max() < 3.0
    pts = m.centre(np.linspace(-m.stem, m.L, 200))
    assert np.abs(pts[:, 1]).max() < 1e-12                        # no lateral deflection by default
    assert pts[-1, 0] < -1.0                                      # it ends below the axis: it droops (x is up)
    # the controls bend in their own planes
    assert abs(model(sway=40.0).centre(m.L)[1]) > 1.0 and model(sway=40.0).centre(m.L)[1] > 0
    assert model(lift=40.0).centre(m.L)[0] > m.centre(m.L)[0] + 1.0
    assert model(droop=1.8).centre(m.L)[0] < m.centre(m.L)[0] - 0.5
    assert model(curl=1.8).centre(m.L)[0] > m.centre(m.L)[0] + 0.3            # the distal curl is up
    # lift rotates the tail about the sacrococcygeal joint: the part inside the body does not move
    assert np.allclose(model(lift=60.0).centre(-m.stem), m.centre(-m.stem), atol=1e-12)
    assert np.allclose(model(lift=60.0).centre(0.0), m.centre(0.0), atol=1e-12)


@pytest.mark.parametrize("kw", [dict(droop=2.0, curl=2.0), dict(droop=-2.0, curl=-2.0), dict(lift=90.0, droop=2.0),
                                dict(lift=-60.0, curl=2.0), dict(sway=90.0, droop=2.0, curl=2.0, lift=90.0),
                                dict(length=1.0, droop=2.0, base_radius=0.5)])
def test_no_pose_folds_the_shell_on_its_inner_side(kw):
    """The curvature is limited (radius x curvature <= 0.8): consecutive rings never cross, whatever the
    controls (the old builder faceted into a zigzag outside its ranges)."""
    m = model(**kw)
    ring = m.section(m.s, "skin")
    cen, tg = m.centre(m.s), m.tangent(m.s)
    for i in range(len(m.s) - 1):
        assert ((ring[i + 1] - cen[i]) @ tg[i] > 0).all(), (kw, i)           # the next ring is ahead of this one
        assert ((ring[i] - cen[i + 1]) @ tg[i + 1] < 0).all(), (kw, i)       # and this one behind the next
    assert m.skin().is_watertight and m.skin().volume() > 0


@pytest.mark.parametrize("kw", [dict(lift=170.0), dict(lift=170.0, droop=2.0, curl=2.0),
                                dict(sway=170.0, length=3.0, base_radius=1.0), dict(droop=8.0, length=3.0, base_radius=0.8)])
def test_the_curvature_limit_really_limits(kw, monkeypatch):
    """Radius x curvature never exceeds 0.8 (``BEND_LIMIT``), at any station, for controls far outside the
    kind's ranges (inside them the limit is a safety net that never has to act: the worst is 0.72 at lift 90);
    and these controls DO exceed it without the limit, so the limiter is what keeps the shell from folding."""
    def worst(m):
        t, sd = m._tang, m._sd
        kappa = np.linalg.norm(np.diff(t, axis=0), axis=1) / np.diff(sd)
        r = m.law(sd[1:]) * 0.92
        return float((kappa * r).max())

    assert worst(model(**kw)) <= 0.8 * 1.02
    monkeypatch.setattr(T, "BEND_LIMIT", 1e6)
    assert worst(model(**kw)) > 0.8 * 1.1


def test_the_default_tail_bends_gently():
    m = model()
    assert m.min_bend_radius(upto=0.4 * m.L) >= 3.0 * m.R                    # the research: >= 3 R over the first 40%
    assert m.min_bend_radius() >= 1.5 * m.law(m.L * 0.95)


def test_the_rings_are_dense_at_the_root_and_the_tip_is_rounded():
    m = model()
    ds = np.diff(m.s)
    root = m.s[:-1] < 4.0 * m.R
    assert ds[root].max() <= 0.45 * m.R * (1 + 1e-9)                         # a ring every 0.4 R at the root
    assert len(m.s) >= 25
    assert m.skin().vertices[:, :].shape[1] == 3
    cap = m._cap_rings("skin", m.p.ring_verts)
    r = [np.linalg.norm(c - c.mean(axis=0), axis=1).mean() for c in cap]
    assert r[0] > r[1] > r[2] > 0                                            # a quarter sphere closes the tip


def test_a_tip_radius_of_zero_is_the_finest_tip_the_skin_allows():
    m = model(tip_radius=0.0)
    assert m.tip == pytest.approx(0.007) or m.tip >= 0.007
    assert m.skin().is_watertight


def test_the_crest_adds_a_dorsal_ridge_only_when_asked():
    smooth, ridge = model(), model(crest=0.4)
    s = smooth.s[10]
    a, b = smooth.section(np.array([s]), "skin"), ridge.section(np.array([s]), "skin")
    assert np.allclose(a, model().section(np.array([s]), "skin"))
    d = np.linalg.norm(b - a, axis=2)[0]
    assert d[0] > 0.1 * ridge.R and d[len(d) // 2] < 1e-12                    # dorsal vertex up, ventral untouched
    assert ridge.skin().is_watertight


# -- on the body ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def world():
    base, lib = mh.load()
    return base, lib, lib.apply(base.verts, mh.macro_weights())


@assets
def test_a_placed_tail_exposes_its_three_layers_in_the_makehuman_frame(world):
    base, _, v0 = world
    p = K.instantiate_all([K.new_spec("tail")], base, v0)[0]
    layers = p.layers()
    assert set(layers) == {"bone", "muscle", "skin"}
    assert np.array_equal(layers["skin"].vertices, p.mesh().vertices)         # the skin IS the printed shell
    cv, _q = base.body_cage(v0)
    for name, mesh in layers.items():
        assert mesh.is_watertight
        c = mesh.vertices.mean(axis=0)
        assert abs(c[0]) < 0.02                                               # on the sagittal plane
        assert np.linalg.norm(c - p.anchor) < 4.0                             # near the root, in the body's frame
    # the bone chain starts inside the body, below the skin, and is inside the muscle which is inside the skin
    assert inside(layers["bone"].vertices, layers["muscle"]).all()
    assert inside(layers["muscle"].vertices, layers["skin"]).all()
    # the layers follow the pose controls (the same model drives all three)
    posed = p.layers({"droop": 0.8, "lift": 20.0})
    assert not np.allclose(posed["skin"].vertices, layers["skin"].vertices)
    assert np.allclose(posed["skin"].vertices, p.mesh({"droop": 0.8, "lift": 20.0}).vertices)
    # the bone moves with the pose beyond the joint; Cd1 (the first spool, 7 rings of 8) is in the body and stays
    assert np.allclose(posed["bone"].vertices[:56], layers["bone"].vertices[:56], atol=1e-9)
    assert not np.allclose(posed["bone"].vertices[-100:], layers["bone"].vertices[-100:], atol=1e-3)
    # a kind without an anatomy has the skin only
    h = K.instantiate_all([K.new_spec("horn")], base, v0)[0]
    assert set(h.layers()) == {"skin"}


@assets
def test_pose_controls_clamp_and_re_evaluate_the_chain(world):
    base, _, v0 = world
    p = K.instantiate_all([K.new_spec("tail")], base, v0)[0]
    rest = p.mesh({})
    for ctl in ({"lift": 40.0}, {"sway": 40.0}, {"droop": 0.5}, {"curl": -0.5}):
        assert not np.allclose(p.mesh(ctl).vertices, rest.vertices), ctl
    wild = p.mesh({"lift": 1e9, "curl": 99.0, "sway": -400.0})
    edge = p.mesh({"lift": 90.0, "curl": 2.0 - 0.55, "sway": -90.0})
    assert np.allclose(wild.vertices, edge.vertices, atol=1e-9) and wild.is_watertight


@assets
def test_the_tail_is_a_pose_of_a_chain_not_a_rubber_tube(world):
    """Lifting keeps the buried part where it is (the sacrum does not move) and rotates the free tail about
    the sacrococcygeal joint; the first ring is identical in every pose."""
    base, _, v0 = world
    p = K.instantiate_all([K.new_spec("tail")], base, v0)[0]
    n = p.params["ring_verts"]
    a, b = p.mesh({}).vertices[:n], p.mesh({"lift": 70.0, "sway": 30.0}).vertices[:n]
    assert np.allclose(a, b, atol=1e-12)


@assets
@pytest.mark.parametrize("macro", [{}, {"height": 1.0}, {"height": 0.0}, {"age": 0.1875},
                                   {"weight": 1.0, "gender": 1.0, "muscle": 1.0}, {"gender": 0.0}])
def test_the_default_tail_clears_the_body_beyond_its_root(world, macro):
    """Beyond 3 R from the skin point the tail surface keeps off the body (the buttocks, the thighs): the
    stiff start and the +10..15 degree axis clear the cheeks."""
    base, lib, _ = world
    v = lib.apply(base.verts, mh.macro_weights(**macro))
    p = K.instantiate_all([K.new_spec("tail")], base, v)[0]
    from minimaster.core.subdiv import subdivide_quads
    cv, cq = base.body_cage(v)
    sv, _sq = subdivide_quads(cv, cq, 1)
    R = p.params["base_radius"] * p.model.scale
    m = p.mesh()
    far = m.vertices[np.linalg.norm(m.vertices - p.anchor, axis=1) > 3.5 * R]
    # nearest body vertex to each far tail vertex (the body is dense enough at level 1)
    dmin = np.array([np.linalg.norm(sv - q, axis=1).min() for q in far[::5]])
    assert dmin.min() > 0.08 * R


@assets
def test_the_tail_prints_it_is_welded_and_its_tip_is_above_the_floor():
    """The registry tail on the real figure through the CLI path: a clean printability report (no blocking
    finding), the root piece 30-60% buried, the tip not thinner than the report's floor at 32 mm."""
    from minimaster.__main__ import main
    import io
    import contextlib
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "t.stl"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert main(["character", "--add", "tail", "-o", str(out), "--size", "medium"]) == 0
        text = buf.getvalue()
        assert "printability  OK" in text and "0 blocking" in text
        line = next(l for l in text.splitlines() if "tail-1" in l and "buried" in l)
        buried = float(line.split("% buried")[0].split()[-1])
        assert 25.0 <= buried <= 65.0, line
