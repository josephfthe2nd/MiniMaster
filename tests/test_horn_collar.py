"""The horn's root collar (``collar.py`` enabled by ``HornKind.collar_profile``): the skin's own cuff around the root.

Round 1 left the keratin sheath meeting the scalp along a ring at 40-58 degrees (the join angle of the bare junction,
measured below as ``before``) with a contact line as ragged as the 0.28-unit cage. The collar is a skin-material shell
on the horn's own geometry whose rim lies on the subdivided skin, tangent to it, and whose top meets the sheath's wall
tangentially: a narrow low rim of skin tucked around a keratin sheath (horn/A06, A22: bovids have no fillet), a wide
concave fillet into an ossicone (horn/A29: 0.5 base radii). Numbers in docs/anatomy_horn.md."""

import time

import numpy as np
import pytest

from minimaster.character import collar as CO
from minimaster.character import hornrig as HR
from minimaster.character import junction as J
from minimaster.character import kinds as K
from minimaster.character import mhbase as mh
from minimaster.character.appendage_spec import AppendageSpec

assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
SHAPES = HR.SHAPE_NAMES
# (round 4: the keratin shapes; the skin-covered ossicone has no cuff, its own foot is the fillet: test_horn_ossicone.py)
KERATIN = tuple(s for s in SHAPES if HR.SHAPES[s]["collar"] is not None)


@pytest.fixture(scope="module")
def world():
    return mh.load()


@pytest.fixture(scope="module")
def neutral(world):
    base, lib = world
    return lib.apply(base.verts, mh.macro_weights())


def figure(world, v, text):
    base, _ = world
    placed = K.instantiate_all([K.add_from_text(text, [], base=base)], base, v)
    shaped = J.apply_junctions(base, v, J.junction_specs(placed))
    return placed, shaped


def test_every_keratin_shape_asks_for_a_skin_cuff_and_the_ossicone_for_none():
    """A keratin sheath gets a narrow low rim of skin tucked round it (no fillet: A06, A22), the ram a wider one (its root
    is on the temple crest); the ossicone (skin over bone, A29) none: round 3's cuff drew a pale scalloped lip round a
    surface that is skin all over, and its own foot is the fillet now (``test_horn_ossicone.py``)."""
    assert KERATIN == ("curved", "ram", "spiral", "antelope")
    for shape in KERATIN:
        w, h = HR.SHAPES[shape]["collar"]
        assert 0.3 <= w <= 1.0 and 0.2 <= h <= 0.8
    assert HR.SHAPES["ram"]["collar"][0] > 1.5 * HR.SHAPES["curved"]["collar"][0]
    assert HR.SHAPES["ossicone"]["collar"] is None
    assert K.KINDS["horn"].defaults()["collar"] == 1.0
    assert K.KINDS["horn"].defaults()["ring_verts"] == 24          # (a 16-gon's facets poked through the cuff)


@assets
@pytest.mark.parametrize("shape", KERATIN)
@pytest.mark.parametrize("levels", [1, 2])
def test_the_collar_is_a_closed_cuff_tangent_to_the_skin_and_the_sheath_on_both_sides(world, neutral, shape, levels):
    """On both horns of the pair (the right one has a mirroring frame: the collar was never run on one before), at the
    two levels the body is meshed with: closed and outward, never under the skin, no crossing of its own, all round the
    root (no fold side on a horn), the rim tangent to the skin and the top tangent to the sheath (p95 under 12 degrees;
    the bare junction met the sheath at 40-58 degrees on average)."""
    base, _ = world
    placed, shaped = figure(world, neutral, f"horn:shape={shape}")
    for p in placed:
        prof = p.kind.collar_profile(p)
        assert prof is not None and prof.up is None and prof.ventral == prof.width and prof.material == "skin"
        rep = CO.join_report(base, shaped, p, levels=levels)
        assert rep["watertight"] and rep["outward"], (shape, p.name)
        # (the collar's own crossings are hidden in the union; collar.py allows 2% of its faces, a horn's has none on
        # this body and up to 8 of 7,000 on others: docs/anatomy_horn.md)
        assert rep["below_skin_vertices"] == 0 and rep["self_crossings"] <= 0.002 * rep["faces"], (shape, p.name, rep)
        assert rep["visible_share"] >= 0.9
        assert rep["rim_p95"] < 12.0 and rep["wall_p95"] < 12.0, (shape, p.name, levels, rep)
        assert rep["before_p50"] > 30.0                    # (what it replaces: a crisp ring)


@assets
def test_no_collar_at_level_0_with_collar_0_or_without_the_junction(world, neutral):
    base, _ = world
    placed, shaped = figure(world, neutral, "horn")
    assert CO.collar_pieces(base, shaped, placed[0], levels=0) == []
    off, shaped_off = figure(world, neutral, "horn:collar=0")
    assert off[0].kind.collar_profile(off[0]) is None
    assert CO.collar_shells(base, shaped_off, off, levels=2) == []
    nofillet, sh2 = figure(world, neutral, "horn:fillet=0")
    assert CO.collar_shells(base, sh2, nofillet, levels=2) == []           # no junction tube: nothing to grow around
    # strength scales it
    big, sh3 = figure(world, neutral, "horn:collar=2,side=left")
    w1 = CO.join_report(base, shaped, placed[0], levels=1)["width_max"]
    w2 = CO.join_report(base, sh3, big[0], levels=1)["width_max"]
    assert w2 > 1.6 * w1


@assets
def test_an_old_record_gets_the_collar_and_round_trips(world, neutral):
    """A saved horn has no ``collar`` parameter: it builds with the default 1.0 (it renders with a cuff at levels 1-2,
    disclosed in docs/anatomy_horn.md); the parameter is not written into a record that never set it."""
    base, _ = world
    rec = {"id": "horn-1", "kind": "horn", "side": "both", "params": {"shape": "curved"}}
    spec = AppendageSpec.from_dict(rec)
    assert "collar" not in spec.to_dict().get("params", {})
    placed = K.instantiate_all([spec], base, neutral)
    assert placed[0].params["collar"] == 1.0 and placed[0].kind.collar_profile(placed[0]) is not None


@assets
def test_the_collar_axis_runs_the_whole_sheath_and_a_pair_is_quick_enough(world, neutral):
    """The collar finds the wall by rays from its axis; the junction tube alone ends half a unit up a 3.4-unit antelope,
    and every triangle beyond it was tested by every ray of the root (0.55 s a collar). The axis runs to the tip now."""
    base, _ = world
    placed, shaped = figure(world, neutral, "horn:shape=antelope")
    p = placed[0]
    pts, rad = p.kind.collar_axis(p)
    spec = p.junction()
    assert np.allclose(pts[: len(spec.points)], spec.points) and len(pts) > len(spec.points) + 10
    s, pos, *_ = p.repose.horn.model.path()
    tip = pos[-1] @ p.matrix[:3, :3].T + p.matrix[:3, 3]
    assert np.linalg.norm(pts[-1] - tip) < p.repose.horn.model.R * p.repose.horn.site.scale
    skin = CO.shaped_skin(base, shaped, 2)
    CO.collar_shells(base, shaped, placed, levels=2, skin=skin)                  # warm
    t0 = time.perf_counter()
    CO.collar_shells(base, shaped, placed, levels=2, skin=skin)
    assert time.perf_counter() - t0 < 2.0                                          # (under load; ~0.3 s a pair quiet)


@assets
@pytest.mark.parametrize("seed", [0, 9])
def test_the_collar_holds_on_a_random_face(world, seed):
    """On an asymmetric head each horn's collar is fitted to its own skin."""
    from minimaster.character.sliders import Character, SliderCatalog, randomize
    base, lib = world
    cat = SliderCatalog(lib.names)
    c = Character()
    c.sliders.update(randomize(cat, seed=seed))
    v = lib.apply(base.verts, c.effective_target_weights(cat))
    placed, shaped = figure(world, v, "horn")
    for p in placed:
        rep = CO.join_report(base, shaped, p, levels=2)
        assert rep["watertight"] and rep["outward"] and rep["below_skin_vertices"] == 0
        assert rep["rim_p95"] < 12.0 and rep["wall_p95"] < 12.0, (seed, p.name, rep)


@assets
def test_the_print_counts_the_horns_collar_in_its_group_but_not_as_its_weld_or_thickness(world, neutral):
    """Through the pipeline's own pieces (``collar_shells`` on the body meshed at level 1, as the CLI and the lab export
    hand them to the report): each horn's group has two shells, the collar is part of its volume but not its weld and
    not a thickness subject, and nothing is detached."""
    from minimaster.character import print_report as PR
    from minimaster.character import printprep as pp
    base, _ = world
    placed, shaped = figure(world, neutral, "horn")
    skin = CO.shaped_skin(base, shaped, 1)
    pieces = CO.collar_shells(base, shaped, placed, levels=1, skin=skin)
    assert sorted(n for _, n, _, _ in pieces) == ["horn-1.l/collar", "horn-1.r/collar"]
    with_c = [[(p.name, p.mesh())] + [(n, m) for q, n, m, _ in pieces if q is p] for p in placed]
    bare = [[(p.name, p.mesh())] for p in placed]
    figure_shells = pp.character_print_shells(base, shaped, levels=1)
    r1 = PR.appendage_report(with_c, base, shaped, 1.9, levels=1, figure_shells=figure_shells, body_verts=shaped)
    r0 = PR.appendage_report(bare, base, shaped, 1.9, levels=1, figure_shells=figure_shells, body_verts=shaped)
    assert not r1.blocking
    for name in ("horn-1.l", "horn-1.r"):
        g1, g0 = r1.groups[name], r0.groups[name]
        assert g1.shells == 2 and g0.shells == 1 and g1.detached == []
        assert g1.root_embed == pytest.approx(g0.root_embed, abs=0.03)
        assert g1.thin_p5_mm == pytest.approx(g0.thin_p5_mm, rel=1e-6)
        assert g1.volume_mm3 > g0.volume_mm3
