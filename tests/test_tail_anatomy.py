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
    skin, muscle, bone = m.skin(), m.muscle(), m.bone_chain()          # (the sacrum wedge lies beside the sleeve, under the body's skin)
    assert inside(muscle.vertices, skin).all()
    assert inside(bone.vertices, muscle).all()
    assert muscle.volume() < skin.volume() and bone.volume() < 0.25 * muscle.volume()


def test_the_skin_is_the_envelope_of_the_muscle_plus_the_skin_thickness():
    """The muscle is the primary shape (lobed where the bellies are) and the skin its envelope: at every vertex of
    every ring the skin lies at least one skin thickness off the muscle (it never cuts into a belly), exactly one thickness
    on the crest of a belly, and further (fat and fascia) in the grooves between the bellies. Change the
    thickness of one layer alone, or the lobing of the skin, and this fails."""
    m = model()
    s = m.s[m.s <= m.L]
    sk, mu = m.section(s, "skin"), m.section(s, "muscle")
    gap = np.linalg.norm(sk - mu, axis=2)                          # vertex k of the skin ring to vertex k of the muscle ring
    t = m.skin_thickness(s)[:, None]
    assert (gap >= t - 1e-12).all()
    th = 2.0 * np.pi * np.arange(m.p.ring_verts) / m.p.ring_verts
    crest = m.lobes(th) >= -1e-9                                   # the bellies and their shoulders: the skin is the muscle + t
    assert np.allclose(gap[:, crest], np.broadcast_to(t, gap.shape)[:, crest], atol=1e-9)
    belly = (m.belly(s) > 0.99)
    groove = m.lobes(th) < -0.1                                    # the grooves: 0, 90, 180, 270 degrees
    assert crest.sum() == 12 and groove.sum() == m.p.ring_verts - 12           # (24 around: 12 and 12; 16 around: 12 and 4)
    assert (gap[np.ix_(belly, groove)] > 1.05 * np.broadcast_to(t, gap.shape)[np.ix_(belly, groove)]).all()   # fat in the grooves
    assert np.allclose(gap[m.belly(s) == 0.0][:, groove], np.broadcast_to(t, gap.shape)[m.belly(s) == 0.0][:, groove], atol=1e-9)
    # the thickness is the research's: 0.08 R at the root rising to 0.12 of the local radius distally
    assert m.skin_thickness(0.0) == pytest.approx(0.08 * m.law(0.0), rel=1e-9)
    assert m.skin_thickness(m.L) == pytest.approx(max(0.02, 0.12 * m.law(m.L)), rel=1e-9)


def test_the_muscle_has_four_bellies_where_the_muscle_is_and_a_round_tendon_beyond():
    """A section of the muscle on the proximal part shows two dorsal and two ventral bellies with a groove between them
    (the dorsal median groove, the lateral bands): the belly's crest is at least 10 % further from the axis than the
    groove; beyond 55 % of the length (tendon, fascia, fat) the section is a plain ellipse."""
    m = model()
    th = 2.0 * np.pi * np.arange(16) / 16
    for frac, lobed in ((0.05, True), (0.12, True), (0.3, True), (0.6, False), (0.9, False)):
        s = np.array([frac * m.L])
        ring = m.section(s, "muscle", 16)[0]
        c, fr = m.centre(s)[0], m.frames(s)[0]
        x, y = (ring - c) @ fr[:, 0], (ring - c) @ fr[:, 1]
        r = np.hypot(x, y)
        crest, groove = r[[2, 6, 10, 14]].mean(), r[[0, 4, 8, 12]].mean()      # 45, 135, 225, 315 degrees / 0, 90, 180, 270
        if lobed and frac <= 0.12:
            assert crest > 1.10 * groove, (frac, crest / groove)
        elif lobed:
            assert crest > 1.02 * groove
        else:
            assert crest == pytest.approx(groove, rel=0.06)                      # (the ellipse itself is 8 % off a circle at the root)


def test_the_muscle_bellies_are_on_the_proximal_part_and_tendons_follow():
    m = model()
    t = np.array([0.0, 0.05, 0.12, 0.3, 0.55, 0.8, 1.0])
    g = m.belly(t * m.L)
    assert g[0] == pytest.approx(1.0) and g[2] == pytest.approx(1.0)          # full up to 12% of the length
    assert g[4] == pytest.approx(0.0, abs=1e-9) and g[5] == 0 and g[6] == 0   # none past 55%
    assert (np.diff(g) <= 1e-12).all()


# -- the body-side masses and the sacrum ---------------------------------------------------------------------------


def _ramp_patch(R, slope=0.3):
    """A body skin that leans: z = slope * x over the grid (a tilted plane), in R."""
    xs = np.arange(T.SkinPatch.X_RANGE[0], T.SkinPatch.X_RANGE[1] + 1e-9, T.SkinPatch.STEP)
    ys = np.arange(T.SkinPatch.Y_RANGE[0], T.SkinPatch.Y_RANGE[1] + 1e-9, T.SkinPatch.STEP)
    return T.SkinPatch(xs, ys, slope * xs[:, None] + 0.0 * ys[None, :], R)


def test_a_skin_patch_samples_the_cage_height_over_the_local_grid_and_fills_holes():
    """The patch is what the mass lenses and the sacrum lie on: the body's own skin as a height field in the tail's frame.
    A tilted plane of quads in a frame scaled by 2 and moved comes back exactly; a hole is filled from its neighbours."""
    n = 41
    gx, gy = np.meshgrid(np.linspace(-12, 12, n), np.linspace(-12, 12, n), indexing="ij")
    verts = np.stack([gx.ravel(), gy.ravel(), 0.3 * gx.ravel() + 0.1 * gy.ravel()], axis=1)
    idx = np.arange(n * n).reshape(n, n)
    quads = np.stack([idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()], axis=1)
    frame = np.eye(4)
    frame[:3, :3] *= 2.0
    frame[:3, 3] = [1.0, -2.0, 3.0]
    body = verts @ frame[:3, :3].T + frame[:3, 3]                       # the plane in the body's frame
    R = 0.5
    patch = T.SkinPatch.from_cage(body, quads, frame, R)
    x, y = np.array([0.0, 2.0, 5.5]), np.array([0.0, -1.5, 3.0])
    assert patch.height(x, y) == pytest.approx(0.3 * x + 0.1 * y, abs=1e-9)          # (x, y, z in base radii: the slope is the plane's)
    # a missing band of triangles (4 R wide, x = 4..8 R: the middle of the grid) leaves a hole the neighbours fill from
    # both sides: finite, and within the slope over half the hole (a hole AT the grid's edge is filled from one side only,
    # flat: off by the slope times its whole width)
    c = verts[quads].mean(axis=1)[:, 0]
    keep = (c < 2.0) | (c > 4.0)
    holed = T.SkinPatch.from_cage(body, quads[keep], frame, R)
    assert np.isfinite(holed.z).all() and holed.height(x, y) == pytest.approx(patch.height(x, y), abs=0.3 * 2.0 + 0.05)
    # an empty cage is the flat plane
    assert (T.SkinPatch.from_cage(body, quads[:0], frame, R).z == 0.0).all()
    assert (T.SkinPatch.flat(R).z == 0.0).all()


def test_the_body_side_muscles_are_closed_solids_with_their_own_shapes():
    """The muscle layer's pieces outside the caudal sleeve (round 4: the round-3 lenses were a height field over the skin, a
    blade above the root and two balls beside it that stood out of the hip): on a flat skin, each is a closed solid with
    the shape of its muscle. The sacrocaudalis dorsalis runs ON THE SACRUM's dorsal face, beside the median crest, into
    the tail; the ventralis on its ventral face; the gluteal band (coccygeal gluteus maximus with the caudofemoralis)
    runs from beside the root infero-laterally and stays within ``GLUTEAL_REACH`` R_j of the midline; left and right are
    mirror images."""
    m = T.TailModel(T.TailParams())
    R = m.R
    pieces = dict(m.body_muscles())
    assert set(pieces) == {f"{n}_{s}" for n in ("sacrocaudalis_dorsalis", "gluteus_maximus", "sacrocaudalis_ventralis")
                           for s in "lr"}
    for name, mesh in pieces.items():
        assert mesh.is_watertight and mesh.volume() > 0 and not len(mesh.degenerate_faces(1e-14)), name
        mirror = pieces[name[:-1] + ("r" if name.endswith("l") else "l")]
        assert np.allclose(np.sort(mesh.vertices[:, 1]), np.sort(-mirror.vertices[:, 1]), atol=1e-9), name
    c, tang, nd, w, h = m.sacral_axis(40)
    # the dorsalis lies on the dorsal face of the sacrum (its vertices: dorsal of the bone's mid-surface, beside the crest)
    dl = pieces["sacrocaudalis_dorsalis_l"].vertices
    body = dl[dl[:, 2] < m.sacral_apex()[2]]                                  # the part along the sacrum (not in the tail)
    j = np.argmin(np.linalg.norm(body[:, None, :] - c[None], axis=2), axis=1)
    assert len(body) > 50 and (((body - c[j]) * nd[j]).sum(axis=1) > 0.0).mean() > 0.95
    assert (body[:, 1] > 0.0).all() and np.abs(body[:, 1]).max() < w.max() + 0.2 * R
    # the ventralis on its ventral face
    vl = pieces["sacrocaudalis_ventralis_l"].vertices
    body = vl[vl[:, 2] < m.sacral_apex()[2]]
    j = np.argmin(np.linalg.norm(body[:, None, :] - c[None], axis=2), axis=1)
    assert len(body) > 30 and (((body - c[j]) * nd[j]).sum(axis=1) < 0.0).mean() > 0.95
    # the gluteal band: medial, oblique (its long axis points down and out), inside the reach
    g = pieces["gluteus_maximus_l"].vertices
    k = m.Rj
    assert np.abs(g[:, 1]).max() <= T.GLUTEAL_REACH * k + 1e-9 and g[:, 1].min() > 0.0
    xy = g[:, :2] - g[:, :2].mean(axis=0)
    axis = np.linalg.svd(xy, full_matrices=False)[2][0]
    axis = axis if axis[1] > 0 else -axis
    assert axis[0] < -0.4 and axis[1] > 0.4                                      # infero-lateral: down (-x) and out (+y)
    ext = xy @ axis
    assert np.ptp(ext) > 1.6 * np.ptp(xy @ np.array([-axis[1], axis[0]]))     # a band, longer than it is wide (not a ball)
    # on the body side (z < 0: not in the tail) nothing is above the skin plane less one skin thickness (the flat skin)
    t = float(m.skin_thickness(0.0))
    for name, mesh in pieces.items():
        v = mesh.vertices
        assert v[v[:, 2] <= 0.0][:, 2].max() <= -t + 1e-6, name


def test_a_lifted_tail_takes_the_wedge_back_in_the_muscle_as_in_the_mound():
    """The sacral wedge fades as the tail lifts (it is in the tail's own way): the dorsalis' bulk near the root and the
    spec's mound use the one weight; the gluteal band stays."""
    w = lambda lift: T.lift_weight(T.ROOT_MASSES[0].fade, lift)                  # noqa: E731
    assert w(0.0) == 1.0 and w(20.0) == pytest.approx(1.0) and w(80.0) == pytest.approx(0.0) and w(90.0) == 0.0
    assert 0.4 < w(50.0) < 0.6 and w(-40.0) == 1.0                                # (lowering the tail is not lifting it)
    assert T.lift_weight(T.ROOT_MASSES[1].fade, 90.0) == 1.0                      # the gluteal masses stay
    rest, up = T.TailModel(T.TailParams()), T.TailModel(T.TailParams(lift=80.0))
    flat = T.RootSkin.flat(0.45)

    assert up.dorsalis(flat, 1.0).volume() < 0.8 * rest.dorsalis(flat, 1.0).volume()   # less bulk near the root
    assert up.gluteal(flat, 1.0).volume() == pytest.approx(rest.gluteal(flat, 1.0).volume(), rel=0.1)   # (the wedge's edge
    # reaches the band's medial end: the rest of it is the gluteal mass, which stays)
    assert [mo.height for mo in T.junction_spec("t", up, np.eye(4), 1.5, 0.5).mounds[:2]] == pytest.approx(
        [T.ROOT_MASSES[1].height * 0.45] * 2)                                    # the wedge's mound is gone from the spec too


def test_the_muscle_layer_is_the_sleeve_and_the_body_side_muscles_and_its_colour_weights_follow():
    m = model()
    mesh, belly = m.muscle_whole()
    pieces = [p for _, p in m.body_muscles()]
    n_sleeve = len(m.muscle().vertices)
    assert len(belly) == len(mesh.vertices) == n_sleeve + sum(len(p.vertices) for p in pieces)
    assert np.array_equal(belly[:n_sleeve], m.muscle_belly()) and (belly[n_sleeve:] == 1.0).all()
    assert belly[0] == 1.0 and belly[n_sleeve - 1] == 0.0                         # root pole: belly; tip pole: tendon
    assert np.array_equal(m.muscle_belly_whole(), belly)                          # (the colours need no geometry)
    assert mesh.volume() > m.muscle().volume() + 0.9 * sum(p.volume() for p in pieces) - 1e-9


def _axis(points):
    """The unit principal axis of a point set."""
    P = np.asarray(points) - np.mean(points, axis=0)
    return np.linalg.svd(P, full_matrices=False)[2][0]


def sacrocaudal_angles(m, n_sacrum_rings=11, ring=16):
    """``(sacrocaudal, chord, intercaudal)`` measured on the MESHES: the angle (degrees) between the axis of the sacrum's
    distal fifth (its last three ring centroids) and the centrum of Cd1, the angle of the sacrum's chord (base -> apex) below the
    tail's axis, and the angles between consecutive centra of the first seven caudals."""
    from minimaster.character import junction_checks as JC
    sac = m.sacrum().vertices[:n_sacrum_rings * ring].reshape(n_sacrum_rings, ring, 3).mean(axis=1)   # base ... apex
    d_sac = sac[-1] - sac[-3]
    d_sac /= np.linalg.norm(d_sac)
    chain = m.bone_chain()
    comp = JC.face_components(chain.faces, len(chain.vertices))
    pieces = [np.unique(chain.faces[comp == lab]) for lab in np.unique(comp)]
    big = max(len(v) for v in pieces)
    centra = sorted((v for v in pieces if len(v) == big),
                    key=lambda v: float(np.linalg.norm(chain.vertices[v].mean(axis=0) - sac[-1])))
    axes = []
    for v in centra[:7]:
        a = _axis(chain.vertices[v])
        axes.append(a if a @ (chain.vertices[v].mean(axis=0) - sac[-1]) > 0 else -a)

    def ang(a, b):
        return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))
    chord = sac[-1] - sac[0]
    chord_deg = float(np.degrees(np.arctan2(-chord[0], chord[2])))
    return ang(d_sac, axes[0]), chord_deg, [ang(axes[i], axes[i + 1]) for i in range(len(axes) - 1)]


@pytest.mark.parametrize("lift", [0.0, 60.0, 90.0, -60.0])
def test_the_sacrum_is_a_tilted_curved_wedge_that_cd1_continues(lift):
    """Round 3 built the sacrum as a vertical slab up the back under the skin (its axis 88 degrees below horizontal) with the
    first caudal leaving its lower end at 104 degrees: a branch, not the continuation of the spine. The references say the
    caudal column continues the sacral curve (the cat plate within ~5 degrees, Gray97's coccyx continues the sacrum;
    TAKEAWAYS section 3: 0 - 20 degrees). Measured on the meshes: the sacrum is a curved wedge whose base lies UP the back and
    DEEP (anterior-superior) and whose apex carries Cd1 on the tail's axis; Cd1's centrum continues the axis of the
    sacrum's distal fifth within 20 degrees at rest; when the tail lifts, the sacrocaudal joint takes a share bounded by
    its range and no joint of the first caudals bends more than 30 degrees (the sacrum itself does not move)."""
    m = model(lift=lift)
    sac = m.sacrum()
    assert sac.is_watertight and sac.volume() > 0
    sc, chord, inter = sacrocaudal_angles(m)
    if lift == 0.0:
        assert 2.0 <= sc <= 20.0, sc
    else:
        assert sc <= T.SACRUM["apex_deg"] + max(abs(x) for x in T.LIFT_JOINT) + 3.0, (lift, sc)
    assert 30.0 <= chord <= 65.0, chord                       # the chord leans: neither a vertical slab (90) nor the tail's line
    assert max(inter) <= 30.0, (lift, inter)                  # (the raised lemur: ~100 degrees over the first four joints)
    v = sac.vertices[:11 * 16].reshape(11, 16, 3).mean(axis=1)
    base, apex = v[0], v[-1]
    Rs = m.Rs
    assert base[0] > apex[0] + 1.0 * Rs and base[2] < apex[2] - 1.0 * Rs        # base up the back and deep
    assert np.allclose(model().sacrum().vertices, sac.vertices)                  # the pose does not move it
    assert np.linalg.norm(apex - np.array([0.0, 0.0, m.joint_s()])) < 0.25 * m.R   # the apex is at the sacrocaudal joint


def test_a_mass_is_the_mound_it_makes_in_the_spec_and_mirrors():
    a = T.ROOT_MASSES[1]
    assert a.mirrored().center == (a.center[0], -a.center[1]) and a.mirrored().name == "gluteal_r"
    assert a.mirrored().angle == -a.angle and a.mirrored() == T.ROOT_MASSES[2]
    band = T.mass_mound(a, 0.5)
    assert band.axis == pytest.approx(a.axis) and band.stretch == pytest.approx(a.half[0] / a.half[1])
    assert band.radius == pytest.approx(a.half[1] * 0.5) and band.height == pytest.approx(a.height * 0.5)
    assert a.axis[0] < -0.5 and a.axis[1] > 0.5                                 # the band runs down and out
    wedge = T.mass_mound(T.ROOT_MASSES[0], 0.5, weight=0.5)
    assert wedge.stretch == pytest.approx(T.ROOT_MASSES[0].half[0] / T.ROOT_MASSES[0].half[1]) and wedge.axis == (1.0, 0.0, 0.0)
    assert wedge.height == pytest.approx(0.5 * T.ROOT_MASSES[0].height * 0.5)
    assert T.bump(0.0) == 1.0 and T.bump(1.0) == 0.0 and T.bump(0.5) == pytest.approx(0.5)      # the mound's own bump
    # the mass weight is the junction's bump of the ellipse along the mass's OWN axis: half way out along and across
    m = model()
    p = np.array(a.center) + 0.5 * a.half[0] * np.array(a.axis[:2])
    q = np.array(a.center) + 0.5 * a.half[1] * np.array([-a.axis[1], a.axis[0]])
    z = a.rise * m.Rj / m.R                                                       # (on the band's own plane: its centre's height)
    assert float(m.mass_weight(a, p[0], p[1], z)) == pytest.approx(float(T.bump(0.5)), rel=1e-6)
    assert float(m.mass_weight(a, q[0], q[1], z)) == pytest.approx(float(T.bump(0.5)), rel=1e-6)
    # every mass stays within GLUTEAL_REACH of the midline (its ellipse's lateral extent)
    for mass in T.ROOT_MASSES:
        ax, ay, _ = mass.axis
        reach = abs(mass.center[1]) + np.hypot(mass.half[0] * ay, mass.half[1] * ax)
        assert reach <= T.GLUTEAL_REACH + 1e-9, mass.name


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
    # lift (round 4): a share at the sacrocaudal joint (Cd1's rostral end, inside the body: the stem pivots there and the
    # tail leaves the sacrum already raised), bounded by the joint's range, the rest by the first free caudals OUTSIDE the
    # body, so a raised tail stands off the back before it rises (round 2b bent it from 1.5 R inside the body: at lift 90
    # 7 % of the tail lay inside the back, the "bolster"). What lies before the joint does not move.
    up = model(lift=60.0)
    sj = up.joint_s()
    assert sj == pytest.approx(-m.R) and -m.stem < sj < 0.0
    assert np.allclose(up.tangent(-m.stem), m.tangent(-m.stem), atol=1e-12)
    assert np.allclose(up.centre(sj - 0.05 * m.R), m.centre(sj - 0.05 * m.R), atol=1e-6)       # the joint and the stem before it stay
    assert np.linalg.norm(up.centre(0.0) - m.centre(0.0)) > 0.05 * m.R                        # the skin point rises with the stem
    pitch = lambda mm, s: np.degrees(np.arcsin(mm.tangent(s)[0]))              # noqa: E731   (x is up)
    joint, free = up.lift_split()
    assert joint == pytest.approx(T.LIFT_SHARE * 60.0) and joint + free == pytest.approx(60.0)
    assert 0.5 * joint < pitch(up, 0.0) < joint                               # most of the joint's share is taken by the skin point ...
    assert pitch(up, sj + T.LIFT_JOINT_LEN * m.R) == pytest.approx(joint, abs=3.0)   # ... all of it by Cd2-3; the rest beyond
    assert pitch(model(lift=60.0, droop=0.0, curl=0.0), (T.LIFT_FROM + T.LIFT_LEN + 0.5) * m.R) == pytest.approx(60.0, abs=5.0)   # (the curvature limit trims the joint)
    for lift in (90.0, -60.0):                                                 # the joint's own range bounds its share
        j, f = model(lift=lift).lift_split()
        assert T.LIFT_JOINT[0] <= j <= T.LIFT_JOINT[1] and j + f == pytest.approx(lift)
    # a lifted tail's elbow is a bend of the chain, not a bulge: the outer wall of the bend stays within 1.35 R of the axis
    assert up.skin().is_watertight


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


@pytest.mark.parametrize("kw", [dict(droop=8.0, curl=-6.0, length=2.5, base_radius=0.7), dict(curl=8.0, length=3.0, base_radius=0.8),
                                dict(sway=170.0, length=3.0, base_radius=1.0), dict(droop=8.0, length=3.0, base_radius=0.8)])
def test_the_curvature_limit_really_limits(kw, monkeypatch):
    """Radius x curvature never exceeds 0.8 (``BEND_LIMIT``), at any station, for controls far outside the
    kind's ranges (inside them the limit is a safety net; since round 4 a lift, shared between the sacrocaudal joint and
    3.5 R of free caudals and stopped at the vertical, never reaches it: 0.51 at lift 170);
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


class _Plain(K.Kind):
    """A kind with no anatomy (the base class's layers): a synthetic one, because every registered kind has one now."""

    id = "plain"


def _plain_placed():
    from minimaster.core.mesh import Mesh
    tri = Mesh(np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 1, 2]]))
    return K.Placed(name="plain-1", spec_id="plain-1", kind=_Plain(), side="", anchor=np.zeros(3), matrix=None,
                    params={}, controls={}, static=tri)


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
    # the caudal chain is inside the muscle sleeve which is inside the skin; the sacrum wedge and the body-side lenses of the
    # muscle layer lie in the body (under the skin the body had), not in the tail
    m = p.model.model()
    n_sleeve, n_sacrum = len(m.muscle().vertices), len(m.sacrum().vertices)
    muscle, bone = layers["muscle"], layers["bone"]
    sleeve = m.muscle().transform(p.model.frame)                                # (the whole muscle layer overlaps itself at the root: parity needs one closed piece)
    assert inside(bone.vertices[:-n_sacrum], sleeve).all()
    assert inside(muscle.vertices[:n_sleeve], layers["skin"]).all()
    from test_tail_anchor import _parity_inside
    from minimaster.core.subdiv import quads_to_mesh
    body = quads_to_mesh(*base.body_cage(v0))
    assert _parity_inside(bone.vertices[-n_sacrum:], body.vertices[body.faces]).all()      # the sacrum is in the body
    assert len(muscle.vertices) == n_sleeve + sum(len(pc.vertices) for _, pc in m.body_muscles())
    # the layers follow the pose controls (the same model drives all three)
    ctl = {"droop": 0.8, "lift": 20.0}
    posed = p.layers(ctl)
    assert not np.allclose(posed["skin"].vertices, layers["skin"].vertices)
    assert np.allclose(posed["skin"].vertices, p.mesh(ctl).vertices)
    # ... the muscle too: it is the posed MODEL's muscle (a layer that ignored the pose, the critics' mutant, fails here)
    pm = p.model.model(ctl)
    assert np.allclose(posed["muscle"].vertices, pm.muscle_whole(p.model.root_skin(ctl))[0].transform(p.model.frame).vertices, atol=1e-12)
    assert not np.allclose(posed["muscle"].vertices[:n_sleeve], layers["muscle"].vertices[:n_sleeve], atol=1e-3)
    # the bone moves with the pose beyond the joint; below the bend (1.5 R under the skin) the first spool is where it was
    assert not np.allclose(posed["bone"].vertices[-n_sacrum - 100:-n_sacrum], layers["bone"].vertices[-n_sacrum - 100:-n_sacrum], atol=1e-3)
    # a kind without an anatomy has the skin only (the base class's layers; the horn has bone and muscle now)
    plain = _plain_placed()
    assert set(plain.layers()) == {"skin"} and plain.layers()["skin"] is plain.static
    h = K.instantiate_all([K.new_spec("horn")], base, v0)[0]
    assert set(h.layers()) == {"bone", "muscle", "skin"}


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
    """Lifting rotates the free tail about the sacrococcygeal joint, which is inside the body; the sacrum does not
    move, the skin point stays, and the buried end of the tail stays buried (it moves by less than a fifth of a base
    radius: the bend starts 1.5 R below the skin point)."""
    base, _, v0 = world
    p = K.instantiate_all([K.new_spec("tail")], base, v0)[0]
    n = p.params["ring_verts"]
    R = p.params["base_radius"] * p.model.scale
    a, b = p.mesh({}).vertices[:n], p.mesh({"lift": 70.0, "sway": 30.0}).vertices[:n]
    assert np.abs(a - b).max() < 0.2 * R
    from test_tail_anchor import _parity_inside
    from minimaster.core.subdiv import quads_to_mesh
    body = quads_to_mesh(*base.body_cage(v0))
    assert _parity_inside(b, body.vertices[body.faces]).all()                  # still inside the body
    assert np.allclose(p.layers()["bone"].vertices[-len(p.model.model().sacrum().vertices):],
                       p.layers({"lift": 70.0})["bone"].vertices[-len(p.model.model().sacrum().vertices):], atol=1e-12)   # the sacrum stays


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
