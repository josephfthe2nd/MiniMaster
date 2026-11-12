"""The tail's layers against the skin the figure is MESHED with (round 4).

Round 3's critic measured the gluteal lenses of the muscle layer standing out of the moulded level-2 skin by up to 0.71 R
on the tall body (11 % of their vertices) and by up to 1.18 R at a base radius of 0.62, while the test that claimed "the
skin is the envelope of the muscle" passed: it measured the skin with the same circular height field that built the
lens, dropped the lens' rim and ran at the default radius only. These tests measure what the defect IS:

* a TRUE signed distance: inside / outside by ray parity against the closed body (three directions, the majority wins)
  and the closed tail, the distance by the closest point of the body's surface; over ALL vertices of the muscle and the
  bone layers (the caudal sleeve and chain are inside the tail's own shell, which the slicer unions with the body); on
  the body subdivided once and twice; on six bodies, random full bodies and base radii 0.45 - 0.85;
* that the muscle is not buried to pass: where a root mass raises the skin, the muscle lies within a few skin thicknesses
  of it and above the skin the body had (the bulge is muscle, not fat);
* that a lifted tail stands off the back, and that the lumbar line's rise over the body has its crest at the root.
"""

import numpy as np
import pytest

from minimaster.character import collar as CO
from minimaster.character import junction as J
from minimaster.character import junction_checks as JC
from minimaster.character import kinds as K
from minimaster.character import mhbase as mh
from minimaster.character import print_report as PR
from minimaster.character import tailrig as T
from minimaster.character.sliders import Character, SliderCatalog, randomize
from minimaster.core.mesh import Mesh
from minimaster.core.subdiv import quads_to_mesh, subdivide_quads

assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")

BODIES = {
    "neutral": dict(gender=1.0, muscle=0.55), "tall": dict(height=1.0, gender=1.0), "short": dict(height=0.0),
    "child": dict(age=0.1875), "heavy": dict(weight=1.0, gender=1.0, muscle=1.0), "female": dict(gender=0.0),
}


@pytest.fixture(scope="module")
def world():
    base, lib = mh.load()
    return base, lib, SliderCatalog(lib.names)


def _build(world, name, text="tail", controls=None, seed=None, junction=True):
    base, lib, cat = world
    c = Character()
    c.macro.update(BODIES.get(name, {}))
    if not junction:
        c.anatomy["junction"] = False
    if seed is not None:
        c.sliders.update(randomize(cat, seed=seed, groups=()))
    c.appendages.append(K.add_from_text(text, c.appendages, base=base, lib=lib))
    if controls:
        c.appendages[0].controls = dict(controls)
    verts = lib.apply(base.verts, c.effective_target_weights(cat))
    placed = c.instantiate_appendages(base, verts, lib)
    return c, verts, placed, c.shaped_body(base, verts, placed)


def _rotations():
    out = [np.eye(3)]
    for ax, ang in (((0.3, 1.0, 0.2), 0.9), ((1.0, -0.4, 0.7), 2.1)):
        a = np.asarray(ax) / np.linalg.norm(ax)
        k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        out.append(np.eye(3) + np.sin(ang) * k + (1 - np.cos(ang)) * k @ k)
    return out


def inside_closed(points, mesh):
    """Ray parity of ``points`` against the closed ``mesh`` along three directions, the majority wins (a ray that grazes a
    shared edge or runs in the plane of a ring cannot flip the answer: a single +X ray did, in this module's first draft)."""
    votes = sum(PR.Solid(Mesh(mesh.vertices @ r.T, mesh.faces)).contains(np.asarray(points) @ r.T).astype(int)
                for r in _rotations())
    return votes >= 2


def exposure(p, shaped, base, levels=(1, 2), layers=None):
    """``{(level, layer): (n vertices, n exposed, largest distance outside in R)}``: a vertex of the muscle or bone layer is
    EXPOSED if it lies outside the body (subdivided ``level`` times) and outside the tail's own closed skin."""
    L = layers if layers is not None else p.kind.layers_on(p, shaped)
    R = p.params["base_radius"] * p.model.scale
    tail = JC.Shells(L["skin"].vertices, L["skin"].faces)
    cv, cq = base.body_cage(shaped)
    out = {}
    for lev in levels:
        body = quads_to_mesh(*subdivide_quads(cv, cq, lev))
        patch = CO.SkinPatch(body.vertices, body.faces, p.anchor, 12.0 * R)
        for name in ("muscle", "bone"):
            P = L[name].vertices
            free = ~tail.inside(P)                                   # what the tail's shell hides is hidden
            exposed = np.zeros(len(P), dtype=bool)
            if free.any():
                exposed[free] = ~inside_closed(P[free], body)
            dist = np.abs(patch.closest(P[exposed])[2]) / R if exposed.any() else np.zeros(0)
            out[(lev, name)] = (len(P), int(exposed.sum()), float(dist.max()) if len(dist) else 0.0)
    return out


CASES = ([(n, "tail", (0, 1, 2), None, True) for n in BODIES]
         + [(n, f"tail:base_radius={r}", (2,), None, True) for n in BODIES for r in (0.5, 0.62, 0.85)]
         + [("neutral", "tail", (1,), s, True) for s in (3, 11)]
         + [("neutral", "tail", (1,), None, False), ("tall", "tail@lift=60", (1,), None, True),
            ("heavy", "tail@lift=-60", (1,), None, True)])


@assets
@pytest.mark.parametrize("name,text,levels,seed,junction", CASES)
def test_no_muscle_or_bone_of_the_tail_lies_outside_the_skin_it_is_meshed_with(world, name, text, levels, seed, junction):
    """Every vertex of the muscle and the bone layers (``Placed.layers()`` as the ecorche draws them, laid under the shaped
    body) lies inside the body or inside the tail's own skin: none is outside both, on the cage itself (the default level
    of the CLI and the lab, which ``--layers`` draws) and on the body subdivided once and twice, on six bodies, base radii
    0.5, 0.62 and 0.85 on all six, two random full bodies, a document with the junction switched
    off, a lifted and a lowered tail. (Round 3: the tall body's gluteal lens 11 % outside by up to 0.71 R, 27 % by up to
    1.18 R at 0.62. Fail-first: these muscles laid without the skin's ceiling,
    ``test_without_the_ceiling_the_muscle_pierces_the_skin``.)"""
    base, _, _ = world
    controls = None
    if "@" in text:
        text, cs = text.split("@")
        controls = {k: float(x) for k, x in (kv.split("=") for kv in cs.split(","))}
    c, v, (p,), shaped = _build(world, name, text, controls=controls, seed=seed, junction=junction)
    for (lev, layer), (n, exposed, worst) in exposure(p, shaped, base, levels).items():
        assert n > 300
        assert exposed == 0, (name, text, seed, lev, layer, exposed, worst)


@assets
def test_without_the_ceiling_the_muscle_pierces_the_skin(world, monkeypatch):
    """Fail-first for the test above: the same muscles laid on the skin the masses alone would make, WITHOUT the clip under
    every level of the shaped skin (``RootSkin.clip`` a no-op), stand out of the skin of a document whose junction is
    switched off (its skin did not rise over the masses): the ceiling, not luck, keeps them in."""
    base, _, _ = world
    monkeypatch.setattr(T.RootSkin, "clip", lambda self, points, normals, t, each=False, rounds=3: np.array(points, dtype=float).reshape(-1, 3))
    c, v, (p,), shaped = _build(world, "neutral", junction=False)
    assert exposure(p, shaped, base, (1,))[(1, "muscle")][1] > 0


def _band_top(p, shaped):
    """The top grids of the two gluteal bands (local frame, units), the rise their masses add there (``w``), their signed
    distances to the moulded skin (twice subdivided) and to the skin the body had, and the skin thickness; the vertices the
    tail's own shell hides are left out."""
    m = p.model.model(p.controls)
    skin = p.model.root_skin(p.controls, shaped)
    nu, nv = T.SHEET_GRID
    pieces = dict(m.body_muscles(skin))
    tops, ws = [], []
    for i, name in ((1, "gluteus_maximus_l"), (2, "gluteus_maximus_r")):
        top = pieces[name].vertices[:nu * nv]                                # the lens' top grid comes first
        tops.append(top)
        ws.append(m.mass_weight(T.ROOT_MASSES[i], *(top / m.R).T) * T.ROOT_MASSES[i].height * m.Rj)
    top, w = np.vstack(tops), np.concatenate(ws)
    keep = ~m.inside_skin(top)
    return (w[keep], skin.moulded[2].closest(top[keep])[2], skin.plain.closest(top[keep])[2],
            float(m.skin_thickness(0.0)), m.Rj)


def _two_sided(p, shaped):
    w, d_moulded, d_plain, t, Rj = _band_top(p, shaped)
    sel = w > 0.15 * Rj
    return (sel.sum(), float(d_moulded[sel].max() / t), float(np.mean(d_moulded[sel] > -3.0 * t)),
            float(np.mean(d_plain[sel] > 0.5 * w[sel] - 0.5 * t)))


@assets
@pytest.mark.parametrize("name", ["neutral", "tall", "child", "female"])
def test_the_muscle_fills_the_bulge_the_mass_raises_and_is_not_buried(world, name):
    """Two-sided, so that neither a muscle that pierces the skin nor one buried deep to pass survives: over the gluteal
    band's top where its mass raises the skin by more than 0.15 R_j, (1) the top lies at least one skin thickness under the
    moulded skin (the ceiling), (2) within three thicknesses of it over three quarters of that area (not buried), and
    (3) above the skin the body had by at least half the rise less half a thickness over three quarters of it: the
    bulge the junction raises is this muscle, not fat over a muscle that stayed where it was. (The heavy body's gluteal
    mass adds less than 0.15 R_j: its buttock's crown stands 0.6 R proud, twice the mass's own height of centre.)"""
    c, v, (p,), shaped = _build(world, name)
    n, top_max, near, raised = _two_sided(p, shaped)
    assert n >= 20
    assert top_max <= -0.98                                                       # (1) under the ceiling
    assert near >= 0.75                                                           # (2) not buried
    assert raised >= 0.75                                                         # (3) the bulge is muscle


@assets
def test_a_skin_that_did_not_rise_leaves_the_muscle_without_its_bulge(world):
    """Fail-first for (3): with the junction switched off (the document's ``anatomy.junction``) the skin does not rise over
    the masses; the ceiling keeps the band under it (1), and so it is NOT above the skin the body had by half the rise:
    (3) sees the difference between a raised skin with muscle under it and an unraised one."""
    c, v, (p,), shaped = _build(world, "neutral", junction=False)
    n, top_max, near, raised = _two_sided(p, shaped)
    assert top_max <= -0.98 and raised < 0.25


@assets
@pytest.mark.parametrize("name", ["neutral", "heavy", "child"])
@pytest.mark.parametrize("lift", [60.0, 90.0])
def test_a_lifted_tail_stands_off_the_back(world, name, lift):
    """Round 2b bent a lifted tail from 1.5 R inside the body: at lift 60 - 90 it lay along the sacrum and the lumbar skin
    as a bolster (the critic), and at lift 90 7 % of the neutral body's tail and 44 % of the heavy body's lay INSIDE the
    back (measured: -0.18 R and -1.36 R). Now a share of the lift is taken at the sacrocaudal joint and the rest by the
    first free caudals outside the body: beyond 2.5 R from its skin point no vertex of the tail is inside the body, and
    the tail keeps at least 0.3 R off the skin."""
    base, _, _ = world
    c, v, (p,), shaped = _build(world, name, controls={"lift": lift})
    m = p.model.model(p.controls)
    R = p.params["base_radius"] * p.model.scale
    skin = p.mesh()
    n = m.p.ring_verts
    s = np.concatenate([np.repeat(m.s, n), np.full(len(skin.vertices) - len(m.s) * n, -1.0)])   # the cap rings / poles: not looked at
    far = skin.vertices[s > 2.5 * m.R]
    body = quads_to_mesh(*subdivide_quads(*base.body_cage(shaped), 1))
    assert not inside_closed(far, body).any()
    patch = CO.SkinPatch(body.vertices, body.faces, p.anchor, 14.0 * R)
    gap = np.abs(patch.closest(far)[2]) / R
    assert gap.min() >= 0.3, (name, lift, gap.min())


@assets
@pytest.mark.parametrize("lift", [-60.0, -30.0])
def test_a_lowered_tail_leaves_a_valid_skin(world, lift):
    """The negative lift (the tail pressed down over the cleft): round 3 tore a diagonal edge up the buttock at -60 (in the
    collar). The cage junction itself: no flipped or collapsed quad, no new visible self-crossing at the cage or once
    subdivided, the cleft below the apex untouched."""
    base, _, _ = world
    for name in ("neutral", "heavy", "child"):
        c, v, (p,), shaped = _build(world, name, controls={"lift": lift})
        specs = J.junction_specs([p])
        q = J.junction_quality(base, v, shaped, specs, levels=(1,))
        assert J.quality_violations(q) == [] and q["selfx_new_visible"] == 0 and q.get("L1_selfx_new_visible", 0) == 0, name
        cv0, cv1 = base.body_cage(v)[0], base.body_cage(shaped)[0]
        assert np.linalg.norm(cv1[[4241, 4250, 4259]] - cv0[[4241, 4250, 4259]], axis=1).max() == 0.0


def midline_rise(p, shaped, xs):
    """The rise of the moulded skin over the skin the body had along the back's midline, at ``xs`` (R_j up the back from
    the skin point, local frame), twice subdivided: the plain skin's points and their signed distance to the moulded one."""
    m = p.model.model(p.controls)
    skin = p.model.root_skin(p.controls, shaped)
    P, _ = skin.surface(np.stack([np.asarray(xs) * m.Rj, np.zeros(len(xs))], axis=1))
    return -skin.moulded[2].closest(P)[2] / m.Rj


@assets
@pytest.mark.parametrize("name", ["neutral", "child", "heavy", "female"])
def test_the_back_rises_onto_the_tail_with_its_crest_at_the_root(world, name):
    """Round 3's ramp test could not fail on the neutral and child bodies (the round-2 code scored 0.00 there too) and it
    passed with the sacral wedge's centre moved back 2.4 R up the back (the critic's mutant M1). What the user saw was the
    lumbar line running over a hump before the tail: measured here directly as the rise of the moulded skin over the skin
    the body had, along the midline above the root (twice subdivided): it is largest at the root's top, falls up the back
    without a second hump, halves within 3 R_j of the skin point and is under a quarter of its root value 3.5 R_j up."""
    c, v, (p,), shaped = _build(world, name)
    xs = np.linspace(1.5, 5.5, 17)
    r = midline_rise(p, shaped, xs)
    assert crest_profile_ok(xs, r), np.round(r, 2)


def crest_profile_ok(xs, r):
    """The rise is largest at the root's top, falls up the back with no second hump, halves within 3 R_j of the skin
    point and is under a quarter of its root value 3.5 R_j up (measured: halved by 2.5 - 2.75, 0.07 - 0.14 at 3.5; the
    wedge centred 2.4 R up the back: halved only by 3.75, 0.41 - 0.57 at 3.5)."""
    half = xs[np.argmax(r < 0.5 * r[0])] if (r < 0.5 * r[0]).any() else np.inf
    at35 = float(np.interp(3.5, xs, r))
    return bool(r[0] >= 0.25 and np.argmax(r) <= 2 and (np.diff(r) <= 0.02).all() and half <= 3.0 and at35 < 0.25 * r[0])


@assets
def test_a_wedge_centred_up_the_back_makes_a_hump_and_fails_the_crest(world, monkeypatch):
    """Fail-first for the crest (the critic's M1): the sacral wedge's centre moved from 1.0 to 2.4 R up the back."""
    monkeypatch.setattr(T, "ROOT_MASSES", (T.ROOT_MASSES[0].__class__(**{**T.ROOT_MASSES[0].__dict__, "center": (2.4, 0.0)}),)
                        + T.ROOT_MASSES[1:])
    xs = np.linspace(1.5, 5.5, 17)
    for name in ("neutral", "child"):
        c, v, (p,), shaped = _build(world, name)
        assert not crest_profile_ok(xs, midline_rise(p, shaped, xs)), name
