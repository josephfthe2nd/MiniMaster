"""The tail's bone against the BODY it is placed in (round 6).

The round-5 critic measured the round-4 sacrum on the meshes: its principal axis 28 - 40 degrees above the horizontal (a
standing human's is about 55 - 60, Gray 111), its anterior end at 73 % of the body's depth from the back (through the
pelvic cavity at the height of the pubis), and its curve reversed (concave dorsally, a hockey stick turning UP at its
anterior end): a quadruped's horizontal sacrum driven through a human pelvis. The test that guarded it bounded the chord
only against the TAIL's axis (30 - 65 degrees) and the depth only from below (the base deeper than the apex), so it could
not see any of that. These tests bound the sacrum against the body: its tilt from the body's horizontal, its anterior
extent as a share of the body's own depth at that height, the sign of its curve, and that it lies in the body; and the
caudal column against the sacrum it continues.
"""

import math

import numpy as np
import pytest

from minimaster.character import junction_checks as JC
from minimaster.character import mhbase as mh
from minimaster.character import tailrig as T
from minimaster.character.sliders import SliderCatalog
from minimaster.core.subdiv import quads_to_mesh, subdivide_quads

from test_tail_layers_in_skin import BODIES, _build, inside_closed

assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")

ALL_BODIES = dict(BODIES, slim_f=dict(gender=0.0, weight=0.0))


@pytest.fixture(scope="module")
def world():
    base, lib = mh.load()
    return base, lib, SliderCatalog(lib.names)


def _midline_depth(body, y, band=0.06):
    """``(back z, front z)`` of the body's midline (the plane x = 0) at the height ``y`` (MakeHuman frame: Y up, +Z the
    front): the section of the triangle mesh ``body`` with x = 0, its points within ``band`` of that height."""
    V, F = body.vertices, body.faces
    d = V[:, 0]
    pts = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = F[:, i], F[:, j]
        cross = (d[a] < 0.0) != (d[b] < 0.0)
        t = d[a[cross]] / (d[a[cross]] - d[b[cross]])
        pts.append(V[a[cross]] + t[:, None] * (V[b[cross]] - V[a[cross]]))
    P = np.vstack(pts)
    for b in (band, 2 * band, 4 * band):                 # (widened where the section is sparse at that height)
        near = P[np.abs(P[:, 1] - y) < b]
        if len(near) >= 2 and np.ptp(near[:, 2]) > 0.5:
            break
    return float(near[:, 2].min()), float(near[:, 2].max())


def sacrum_in_body(base, v, p):
    """The sacrum of the placed tail ``p`` measured in the body (the morphed verts ``v`` BEFORE the junction: the body's own
    depth, not a skin the junction drew up the tail): ``dict(tilt, rise, advance, anterior, bow_upper, bow_dorsal,
    bow_ventral, inside)`` with the chord (base -> apex ring centroids) ``tilt`` degrees below the body's horizontal, the
    base ``rise`` / ``advance`` (in sacral lengths) above / in front of the apex, the anterior-most vertex at ``anterior`` of
    the body's depth from the back at its height, the dorsal sagitta of the upper sacrum (base to 60 % of the way to the apex)
    over its own chord and of the whole sacrum's centreline (+: dorsal, in sacral lengths), and the share of its vertices
    inside the body."""
    m = p.model.model(p.controls)
    F = p.model.frame
    n, rings = 16, 11
    mesh = m.sacrum(n, rings).transform(F)
    cen = mesh.vertices[:rings * n].reshape(rings, n, 3).mean(axis=1)          # base ... apex
    base_c, apex_c = cen[0], cen[-1]
    chord = apex_c - base_c
    L = float(np.linalg.norm(chord))
    tilt = math.degrees(math.atan2(-chord[1], -chord[2]))                     # down and back (-Y, -Z)

    def sagitta(pts):
        c = pts[-1] - pts[0]
        dorsal = np.array([0.0, c[2], -c[1]])                                   # perpendicular in the sagittal plane ...
        dorsal /= np.linalg.norm(dorsal)
        if dorsal[2] > 0:                                                       # ... pointing back (-Z)
            dorsal = -dorsal
        return (pts - pts[0]) @ dorsal / L

    upper = sagitta(cen[:7])
    whole = sagitta(cen)
    body = quads_to_mesh(*subdivide_quads(*base.body_cage(v), 1))
    V = mesh.vertices
    k = int(np.argmax(V[:, 2]))
    zb, zf = _midline_depth(body, V[k, 1])
    return dict(tilt=tilt, rise=(base_c[1] - apex_c[1]) / L, advance=(base_c[2] - apex_c[2]) / L,
                anterior=(V[k, 2] - zb) / (zf - zb), bow_upper=float(upper.max()), bow_dorsal=float(whole.max()),
                bow_ventral=float(-whole.min()), inside=float(inside_closed(V, body).mean()))


@assets
@pytest.mark.parametrize("name", sorted(ALL_BODIES))
def test_the_sacrum_sits_in_the_pelvis_as_a_human_sacrum(world, name):
    """On every body: the chord leans 50 - 70 degrees below the body's horizontal (Gray 111, the standing column: 60 - 62;
    measured 59), the base lies above and in front of the apex, the anterior end (the promontory) at most half the body's
    depth from the back (measured 36 - 44 %; round 4: 73 - 77 %), the upper sacrum bows DORSALLY of its chord by 3 % of the
    length or more (concave anteriorly; measured 5.6 %) and the whole bows more dorsally than ventrally (the lowest segments
    turn back into the tail by less), and every vertex of it is inside the body."""
    base, _, _ = world
    T.TailModel  # (the module under test)
    c, v, (p,), shaped = _build_body(world, name)
    s = sacrum_in_body(base, v, p)
    assert 50.0 <= s["tilt"] <= 70.0, (name, s)
    assert s["rise"] >= 0.6 and s["advance"] >= 0.25, (name, s)
    assert s["anterior"] <= 0.5, (name, s)
    assert s["bow_upper"] >= 0.03 and s["bow_dorsal"] > s["bow_ventral"], (name, s)
    assert s["inside"] == 1.0, (name, s)


def _build_body(world, name, text="tail", controls=None):
    base, lib, cat = world
    saved = BODIES.get(name)
    BODIES[name] = ALL_BODIES[name]
    try:
        return _build(world, name, text, controls=controls)
    finally:
        if saved is None:
            BODIES.pop(name, None)


ROUND4_SACRUM = dict(length=2.4, w_base=1.30, w_apex=0.42, thick_base=0.50, thick_apex=0.22, upper=(0.0, 1.0),
                     apex=(1.04, -0.22))


@assets
def test_the_round_4_quadruped_sacrum_fails_the_body_bounds(world, monkeypatch):
    """Fail-first: the round-4 sacrum (its axis turning from 6 degrees below the TAIL's axis at the apex to 80 at the base,
    the turn growing toward the base, the apex on the tail's axis 1.06 R deep; the neutral body's axis is 11.7 degrees above
    horizontal) leans 33 degrees, reaches 70 % or more of the body's depth and bows the wrong way."""
    base, _, _ = world
    monkeypatch.setattr(T, "SACRUM", dict(T.SACRUM, **ROUND4_SACRUM))
    monkeypatch.setattr(T, "sacral_angle", lambda u: 6.0 + 74.0 * np.asarray(u) ** 0.85 - 11.7)
    c, v, (p,), shaped = _build_body(world, "neutral")
    s = sacrum_in_body(base, v, p)
    assert s["tilt"] < 50.0 and s["anterior"] > 0.6 and s["bow_dorsal"] < s["bow_ventral"], s


def column_angles(m):
    """``(sacrocaudal, intercaudal)``, degrees, on the MESHES: between the axis of the sacrum's distal fifth (its last three
    ring centroids) and Cd1's centrum (its principal axis), and between consecutive centra of the first seven caudals."""
    n, rings = 16, 11
    sac = m.sacrum(n, rings).vertices[:rings * n].reshape(rings, n, 3).mean(axis=1)
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
        P = chain.vertices[v] - chain.vertices[v].mean(axis=0)
        a = np.linalg.svd(P, full_matrices=False)[2][0]
        axes.append(a if a @ (chain.vertices[v].mean(axis=0) - sac[-1]) > 0 else -a)

    def ang(a, b):
        return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))
    return ang(d_sac, axes[0]), [ang(axes[i], axes[i + 1]) for i in range(len(axes) - 1)]


@pytest.mark.parametrize("elevation", [5.0, 11.7, 18.0])
@pytest.mark.parametrize("lift", [0.0, 60.0, 90.0, -60.0])
def test_cd1_continues_the_sacral_apex_and_the_column_bends_into_the_tail(elevation, lift):
    """The caudal column grows FROM the sacrum: Cd1's centrum continues the axis of the sacrum's distal fifth within 20
    degrees (TAKEAWAYS section 3: 0 - 20; measured 15 - 19) at rest and lifted, on the root elevations the bodies have
    (5 - 18 degrees); the turn from the down-pointing sacrum into the tail is spread over the first caudal joints (no joint
    of the first seven bends more than 30 degrees at rest (measured 19 - 27), 55 lifted 60 - 90 (measured up to 51 on the steepest
    root): the raised lemur bends ~100 degrees at its sacrocaudal joints,
    tail/cm_Lemur_catta_skelet_1782); the column has joined the tail's own axis past the root; the pose does not move the
    sacrum."""
    m = T.TailModel(T.TailParams(lift=lift, elevation=elevation))
    sc, inter = column_angles(m)
    assert sc <= 20.0, (elevation, lift, sc)
    assert max(inter) <= (30.0 if lift <= 0.0 else 55.0), (elevation, lift, inter)
    s_join = m.chain_join()
    far = np.linspace(s_join, m.L, 30)
    assert np.allclose(m.bone_centre(far), m.centre(far), atol=1e-6)            # from the join, the column IS the tail's axis
    rest = T.TailModel(T.TailParams(elevation=elevation))
    assert np.allclose(m.sacrum().vertices, rest.sacrum().vertices)             # the pose does not move the sacrum
    assert np.allclose(m.bone_centre(m.bone_path()[0][0]), rest.bone_centre(rest.bone_path()[0][0]))   # nor Cd1's root


def test_a_column_that_leaves_the_apex_along_the_tail_kinks_and_fails(monkeypatch):
    """Fail-first: Cd1 leaving the sacral apex along the tail's own axis (the round-4 arrangement, where the sacrum was laid
    along the tail instead) makes a kink of 45 degrees or more at the sacrocaudal joint of a human sacrum."""
    monkeypatch.setattr(T, "CD1_TURN", float(T.sacral_angle(0.0)) + 11.7)
    sc, inter = column_angles(T.TailModel(T.TailParams(elevation=11.7)))
    assert sc > 40.0, sc
