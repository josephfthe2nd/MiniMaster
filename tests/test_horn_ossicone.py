"""The skin-covered ossicone as ONE surface (round 4, ``hornrig.HornModel._fillet_foot``, ``hornrig.shade_foot``).

The round-3 critic found the ossicone (skin over bone, horn/A29 and A30: hair and skin continuous, no seam line) drawn
with a pale scalloped lip round its base where the skin's cuff (``collar.py``, a second shell) met it, a hard tone band
at 55 % of its height (its foot rings bunched at 1.625 / 1.708 / 1.738 base radii, so the profile's slope jumped from
-0.30 to -0.09 under the first shaft ring) and two tone rings in the scalp. Now the ossicone has no cuff: its own foot is
the concave fillet of A29 (0.5 base radii), the profile is C1 from the skin to the dome, the fillet meets the skin the
figure is DRAWN with (the moulded cage subdivided once) at a few degrees and dives under it, and it carries the skin's
shading where it meets it (the renderer draws each shell with its own normals). Numbers in docs/anatomy_horn.md.
"""

import math

import numpy as np
import pytest

from minimaster.character import collar as CO
from minimaster.character import hornrig as HR
from minimaster.character import junction as J
from minimaster.character import kinds as K
from minimaster.character import mhbase as mh
from minimaster.character import portrait as po
from minimaster.character import shading as sh
from minimaster.character.appendage_spec import AppendageSpec
from minimaster.core.mesh import Mesh

assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")

BODIES = {"neutral": {}, "child": dict(age=0.1875), "tall": dict(height=1.0), "heavy": dict(weight=1.0, muscle=1.0)}


@pytest.fixture(scope="module")
def world():
    return mh.load()


@pytest.fixture(scope="module")
def bodies(world):
    base, lib = world
    return {name: lib.apply(base.verts, mh.macro_weights(**kw)) for name, kw in BODIES.items()}


def figure(world, v, text="horn:shape=ossicone"):
    base, _ = world
    placed = K.instantiate_all([K.add_from_text(text, [], base=base)], base, v)
    shaped = J.apply_junctions(base, v, J.junction_specs(placed))
    return placed, shaped


def generatrices(p):
    """``(rings, field)``: the built sheath's rings in the horn's local frame (bowl, foot and shaft, without the dome)
    as an ``(n_rings, ring_verts, 3)`` array, and the skin it was built on."""
    hpl = p.repose.horn
    m = hpl.model
    nv = int(m.p.ring_verts)
    bowl, foot, shaft_s = m.foot_rings(hpl.field)
    n = len(bowl) + len(foot) + len(shaft_s)
    v = hpl.skin().vertices[: n * nv].reshape(n, nv, 3)
    return v, hpl.field


def turning(points):
    """Degrees between consecutive segments of a polyline ``(n, 3)``."""
    seg = np.diff(points, axis=0)
    seg = seg / np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-15)
    return np.degrees(np.arccos(np.clip((seg[:-1] * seg[1:]).sum(axis=1), -1.0, 1.0)))


def test_the_ossicone_is_skin_over_bone_with_its_own_fillet_and_no_cuff_parameter():
    row = HR.SHAPES["ossicone"]
    assert row["material"] == "skin" and row["foot"] == "fillet" and row["collar"] is None
    # A29: the valley fillet is 0.5 base radii; the column tapers to about 0.8 of its base before the dome
    p = HR.HornParams(shape="ossicone", length=row["length"], base_radius=row["base_radius"], tip_radius=row["tip"],
                      flare=row["flare"], tilt=row["tilt"], ring_verts=row["ring_verts"], segments=row["segments"])
    assert p.fillet_foot and p.fillet_rho == pytest.approx(0.5 * p.base_radius) and p.flare_eff == 0.0
    m = HR.HornModel(p)
    for u, want in ((0.25, 0.94), (0.5, 0.88), (0.75, 0.82)):                     # A29 (of the column's base)
        assert float(m.taper(u * m.L)) / m.R == pytest.approx(want, abs=0.03), u
    # H / W of the column 1.83 (A29 / the giraffe silhouette: 1.5-1.75 of a base that includes some of the fillet)
    assert 1.5 <= m.L / (2.0 * m.R) <= 2.0
    horn = K.KINDS["horn"]
    assert "collar" not in horn.defaults(shape="ossicone") and "collar" in horn.defaults(shape="curved")
    # every keratin shape keeps its skin cuff
    for shape, r in HR.SHAPES.items():
        if r["material"] == "bone":
            assert r["collar"] is not None and r.get("foot") is None, shape


@assets
@pytest.mark.parametrize("levels", [1, 2])
def test_no_collar_piece_is_built_for_an_ossicone(world, bodies, levels):
    """Round 3 drew the ossicone's cuff as a second shell whose top edge was a pale scalloped lip. No cuff now, on
    either horn, at either level the body is meshed with, and an old record that set ``collar`` keeps it in the file
    but builds without one."""
    base, _ = world
    placed, shaped = figure(world, bodies["neutral"])
    assert len(placed) == 2
    for p in placed:
        assert p.kind.collar_profile(p) is None and p.material.base_color == K.material_named("skin").base_color
        assert CO.collar_pieces(base, shaped, p, levels=levels) == []
    assert CO.collar_shells(base, shaped, placed, levels=levels) == []
    rec = AppendageSpec.from_dict({"id": "horn-1", "kind": "horn", "side": "left",
                                   "params": {"shape": "ossicone", "collar": 1.5}})
    q = K.instantiate_all([rec], base, bodies["neutral"])[0]
    assert q.kind.collar_profile(q) is None and rec.to_dict()["params"]["collar"] == 1.5


@assets
@pytest.mark.parametrize("body", list(BODIES))
def test_the_ossicones_profile_turns_smoothly_from_the_skin_to_the_dome(world, bodies, body):
    """Measured on the BUILT sheath, along every generatrix (one vertex of each ring, every other one round), from
    where the fillet leaves the skin it was built on (a twentieth of the fillet's radius up: below that the rings lie on
    the drawn skin and turn with its facets, 1-2.5 degrees) to the start of the dome: consecutive segments turn by at
    most the fillet's arc step (5 degrees) plus one, and over the column above the fillet by under 2 degrees (round
    3: 11.6 degrees where the bunched foot rings met the first shaft ring, the 'sock line' at 55 % of the height)."""
    p = figure(world, bodies[body])[0][0]
    rings, field = generatrices(p)
    m = p.repose.horn.model
    rho = m.p.fillet_rho
    worst, worst_col = 0.0, 0.0
    for j in range(0, rings.shape[1], 2):
        g = rings[:, j]
        h = g[:, 2] - field.height(g[:, :2])
        k0 = int(np.argmax(h > 0.05 * rho))
        tt = turning(g[k0:])
        worst = max(worst, float(tt.max()))
        col = np.nonzero(h[k0:] > 1.2 * rho)[0]                          # the column (above the fillet's top)
        worst_col = max(worst_col, float(tt[col[0]:].max()))
    assert worst < 6.0, (body, worst)                   # (a fixed number: the arc's 5-degree step plus one)
    assert worst_col < 2.0, (body, worst_col)


@assets
@pytest.mark.parametrize("levels", [1, 2])
def test_the_ossicone_meets_the_drawn_skin_at_a_few_degrees_and_its_rim_is_under_it(world, bodies, levels):
    """Against the skin the figure is DRAWN with (the shaped body subdivided ``levels`` times, true closest-point
    distances), along every generatrix of the built sheath: the segment that crosses the skin does so within the fillet's
    crossing angle (6 degrees) plus 8 of the skin's tangent plane (measured: median 4.4-4.8, at most 11.8; round 3: 47 median,
    69 at most, and the bare sheath met the skin at 40 degrees on
    average, ``join_report``'s before_p50, and the cuff's top edge met the sheath at its own angle), and every vertex of
    the deepest ring is under the skin (no lip, no flange standing off it: a fillet tangent to the CAGE stood up to 0.2
    base radii off the drawn skin), on four bodies and both horns."""
    base, _ = world
    gamma = HR.SHAPES["ossicone"]["gamma"]
    for body in BODIES:
        placed, shaped = figure(world, bodies[body])
        skin = CO.shaped_skin(base, shaped, levels)
        for p in placed:
            hpl = p.repose.horn
            R = hpl.site.radius
            patch = CO.SkinPatch(skin[0], skin[1], p.anchor, 6.0 * R + 0.4)
            rings, _ = generatrices(p)
            world_rings = rings @ p.matrix[:3, :3].T + p.matrix[:3, 3]
            nr, nv = rings.shape[:2]
            _, n, sd, _ = patch.closest(world_rings.reshape(-1, 3))
            sd, n = sd.reshape(nr, nv), n.reshape(nr, nv, 3)
            crossing = []
            for j in range(nv):
                i = int(np.nonzero((sd[:-1, j] < 0.0) & (sd[1:, j] >= 0.0))[0][-1])       # the last way up out of it
                seg = world_rings[i + 1, j] - world_rings[i, j]
                crossing.append(math.degrees(math.asin(abs(float(seg @ n[i, j])) / float(np.linalg.norm(seg)))))
            assert max(crossing) < gamma + 8.0, (body, levels, p.name, max(crossing))
            bowl, foot, _ = hpl.model.foot_rings(hpl.field)
            rim = foot[0] @ p.matrix[:3, :3].T + p.matrix[:3, 3]
            _, _, sd_rim, _ = patch.closest(rim)
            assert sd_rim.max() < -0.2 * HR.SHAPES["ossicone"]["dive"][0] * R, (body, levels, p.name, sd_rim.max())


@assets
def test_the_ossicone_carries_the_skins_shading_where_it_meets_the_skin(world, bodies):
    """The renderer shades every shell with its own normals: where the foot dives under the skin at a few degrees the
    two shadings interleaved into a ragged ring. The ossicone's mesh carries the drawn skin's normals (the renderer's own
    area weighted ones) at and under the skin, its own well up the fillet, through the pipeline's own transform
    (``Shell(mesh.transform(MH_TO_Z_UP), material)``); a keratin horn carries none (its seam is the cuff's)."""
    base, _ = world
    placed, shaped = figure(world, bodies["neutral"])
    skin = CO.shaped_skin(base, shaped, 1)
    for p in placed:
        m = p.mesh()
        assert isinstance(m, HR.ShadedMesh) and m.shading_normals.shape == m.vertices.shape
        hpl = p.repose.horn
        R, rho = hpl.site.radius, hpl.model.p.fillet_rho * hpl.site.scale
        patch = CO.SkinPatch(skin[0], skin[1], p.anchor, 6.0 * R + 0.4)
        _, n, sd, _ = patch.closest(m.vertices)
        # (at and just under the skin: a vertex deeper in the stem and the bowl has no one nearest skin point)
        low = (sd < HR.SHADE_BAND[0] * rho) & (sd > -0.3 * rho) & (np.linalg.norm(m.vertices - p.anchor, axis=1) < 3.0 * R)
        ang = np.degrees(np.arccos(np.clip((m.shading_normals[low] * n[low]).sum(axis=1), -1.0, 1.0)))
        # (the closest point is found from the nearest vertex's triangles, on a patch of another size here: a
        # neighbouring triangle's interpolated normal differs by a degree or two at a few vertices; the foot's own normals
        # there are 6-14 degrees off the skin's)
        assert low.sum() > 50 and np.percentile(ang, 90) < 0.5 and ang.max() < 3.0, (p.name, ang.max())
        own = sh.vertex_normals(m.vertices, m.faces)
        rim = low & (sd > -0.05 * rho)
        assert np.median(np.degrees(np.arccos(np.clip((own[rim] * n[rim]).sum(axis=1), -1.0, 1.0)))) > 4.0
        high = sd > HR.SHADE_BAND[1] * rho * 1.01
        assert np.allclose(m.shading_normals[high], own[high], atol=1e-9)
        # ... and the attributes survive the pipeline's transform (normals by the inverse transpose; the right horn's
        # frame mirrors)
        mz = m.transform(po.MH_TO_Z_UP)
        assert isinstance(mz, HR.ShadedMesh)
        assert np.allclose(mz.shading_normals, m.shading_normals @ po.MH_TO_Z_UP[:3, :3].T, atol=1e-9)
        assert np.allclose(mz.shading_cavity, m.shading_cavity)
        # the print path merges plain meshes: the same geometry, no shading
        merged = Mesh.merge([m])
        assert type(merged) is Mesh and np.array_equal(merged.vertices, m.vertices)
    curved = K.instantiate_all([K.add_from_text("horn", [], base=base)], base, bodies["neutral"])[0]
    assert getattr(curved.mesh(), "shading_normals", None) is None


@assets
def test_the_ossicone_prints_clean_and_builds_quickly(world, bodies):
    """One closed outward shell per horn with a real weld, built in under a second for the pair (the fillet foot is
    fitted to the cage subdivided once: about 50 ms more than a keratin horn's)."""
    import time

    from minimaster.character import print_report as pr
    from minimaster.character import printprep as pp
    base, _ = world
    v = bodies["neutral"]
    spec = K.add_from_text("horn:shape=ossicone", [], base=base)
    t0 = time.perf_counter()
    placed = K.instantiate_all([spec], base, v)
    assert time.perf_counter() - t0 < 1.0
    for p in placed:
        rep = p.mesh().integrity_report()
        assert rep["watertight"] and rep["outward"] and rep["degenerate_faces"] == 0, rep
    shaped = J.apply_junctions(base, v, J.junction_specs(placed))
    shells = pp.character_print_shells(base, shaped)
    app = [[(p.name, p.mesh())] for p in placed]
    for size in ("small", "medium"):
        mm = pp.figure_scale(shells + pp.appendage_print_shells(app), size=size, reference=pp.FIGURE_SHELLS)
        rep = pr.appendage_report(app, base, v, mm, body_verts=shaped)
        assert not rep.blocking, rep.as_text()
        for g in rep.groups.values():
            assert 0.15 <= g.root_embed <= 0.85 and g.detached == [], (size, g.root_embed)
