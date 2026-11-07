"""Appendages on the print path: the frame seam, the figure-referenced size contract,
the piece-level wing meshes and the thickness floor.

The watertight gate is topological, so a wing in the wrong frame, a figure sized by its wing
tips and a wing thinner than a printer resolves all pass it. These tests measure the
placement and the size instead.
"""

import numpy as np
import pytest

from minimaster.character import appendages as A
from minimaster.character import mhbase as mh
from minimaster.character import print_report as pr
from minimaster.character import printprep as pp
from minimaster.character.portrait import MH_TO_Z_UP
from minimaster.core import primitives
from minimaster.core.mesh import Mesh

needs_assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")

TAIL = A.Appendage("tail", anchor_point=(0.0, 1.6, -1.15))
HORN = A.Appendage("horn", anchor_point=(0.85, 8.55, 0.05), mirror=True)
EPS_MM = 0.1            # an appendage root further than this from the body is not attached


def _ball(r=1.0, at=(0.0, 0.0, 0.0)):
    return primitives.icosphere(radius=r, subdivisions=2).translated(at)


@pytest.fixture(scope="module")
def figure():
    base, lib = mh.load()
    v = lib.apply(base.verts, mh.macro_weights(gender=1.0, muscle=0.55))
    return base, v, pp.character_print_shells(base, v)


# -- the reference= size contract ------------------------------------------


def test_reference_makes_the_figure_not_the_wing_tip_set_the_height():
    body = ("body", _ball(2.0))
    wing = ("wing", _ball(1.0, (4.0, 0.0, 4.0)))          # sticks out and above the head
    _, plain = pp.assemble_character([body, wing], height=30.0, with_base=False)
    assert plain.height_mm == 30.0
    mesh, rep = pp.assemble_character([body, wing], height=30.0, with_base=False,
                                      reference=["body"])
    placed = dict(rep.placed_shells)
    lo, hi = placed["body"].bounds
    assert hi[2] - lo[2] == pytest.approx(30.0, rel=1e-9)         # the BODY is 30 mm
    assert lo[2] == pytest.approx(0.0, abs=1e-9)                  # and stands on z=0
    assert (lo[0] + hi[0]) == pytest.approx(0.0, abs=1e-9)        # centred on XY by the body ...
    assert mesh.bounds[1][0] > 5.0                                # ... not by the wing
    assert rep.overall_height_mm > 30.0 + 1.0                     # the wing stands above it
    assert any("with appendages" in n for n in rep.notes)
    # without reference the same shells are scaled by the wing tip: the body shrinks
    plain_placed = dict(pp.assemble_character([body, wing], height=30.0, with_base=False)[1]
                        .placed_shells)
    assert np.ptp(plain_placed["body"].vertices[:, 2]) < 27.0


def test_reference_none_changes_nothing_about_the_default_path():
    shells = [("a", _ball(2.0)), ("b", _ball(1.0, (0.5, 0, 1.0)))]
    m1, r1 = pp.assemble_character(shells, height=30.0)
    m2, r2 = pp.assemble_character(shells, height=30.0, reference=None, fit_base=True)
    assert np.array_equal(m1.vertices, m2.vertices) and np.array_equal(m1.faces, m2.faces)
    assert r1.notes == r2.notes and r1.overhang_mm == 0.0 and r1.overall_height_mm is None


def test_unknown_reference_names_are_an_error():
    with pytest.raises(pp.PrintError, match="reference"):
        pp.assemble_character([("a", _ball())], height=30.0, reference=["body"])


def test_figure_scale_matches_what_assemble_uses():
    shells = [("body", _ball(2.0)), ("wing", _ball(1.0, (4, 0, 4)))]
    s = pp.figure_scale(shells, height=30.0, reference=["body"])
    assert s == pytest.approx(30.0 / 4.0)
    _, rep = pp.assemble_character(shells, height=30.0, reference=["body"])
    assert rep.scale == pytest.approx(s)


def test_wings_that_overhang_the_base_are_reported_and_fit_base_carries_them():
    shells = [("body", _ball(2.0)), ("wing", _ball(1.0, (9.0, 0.0, 2.0)))]
    _, rep = pp.assemble_character(shells, height=30.0, reference=["body"])
    assert rep.overhang_mm > 10.0
    assert any("past the" in n and "base rim" in n for n in rep.notes)
    _, fit = pp.assemble_character(shells, height=30.0, reference=["body"], fit_base=True)
    assert fit.overhang_mm == 0.0
    assert fit.base_diameter_mm > rep.base_diameter_mm
    assert any("widened" in n for n in fit.notes)
    # fit_base never narrows a base that is already wide enough
    _, wide = pp.assemble_character(shells, height=30.0, reference=["body"], fit_base=True,
                                    base_spec={"style": "round", "diameter": 200.0,
                                               "height": 3.0})
    assert wide.base_diameter_mm == 200.0


def test_shell_volume_is_named_for_what_it_is():
    shells = [("a", _ball(1.0)), ("b", _ball(1.0, (0.2, 0, 0)))]       # nearly the same ball twice
    _, rep = pp.assemble_character(shells, height=30.0, with_base=False)
    one = _ball(1.0).scaled(rep.scale).volume()
    assert rep.shell_volume_mm3 == pytest.approx(2 * one, rel=1e-9)    # overlap counted twice
    assert rep.volume_mm3 == rep.shell_volume_mm3                      # the old name still reads
    assert "overlaps counted twice" in rep.as_text()


def test_the_gate_reports_every_failing_shell_not_the_first():
    full = _ball(1.0)
    holed = Mesh(full.vertices, full.faces[1:])
    flipped = Mesh(full.vertices, full.faces[:, [0, 2, 1]])
    shells = [("ok", _ball(1.0)), ("holed", holed), ("flipped", flipped)]
    assert [n for n, _ in pp.shell_failures(shells)] == ["holed", "flipped"]
    with pytest.raises(pp.PrintError) as err:
        pp.assemble_character(shells, height=30.0)
    assert [n for n, _ in err.value.failures] == ["holed", "flipped"]
    assert "holed" in str(err.value) and "flipped" in str(err.value)
    assert "not watertight" in str(err.value) and "inside-out" in str(err.value)


# -- PlacedWing.piece_meshes -------------------------------------------------


@needs_assets
@pytest.mark.parametrize("style", ["membrane", "dragon", "feathered"])
@pytest.mark.parametrize("preset", [None, "folded", "half_fold"])
def test_piece_meshes_of_both_sides_are_closed_outward_solids(figure, style, preset):
    base, v, _ = figure
    app = A.back_wings(style)
    for name, pw in app.build_rigs(base, v):
        pieces = pw.piece_meshes(preset)
        assert [n for n, _ in pieces] == [p.name for p in pw.rig.pieces]
        for n, m in pieces:
            rep = m.integrity_report()
            assert rep["watertight"] and rep["volume"] > 0, (name, n, rep)
        whole = pw.pose(preset) if preset else pw.mesh()
        assert sum(len(m.faces) for _, m in pieces) == len(whole.faces)
        assert sum(len(m.vertices) for _, m in pieces) == len(whole.vertices)


@needs_assets
def test_a_mirrored_wing_needs_the_flip_that_piece_meshes_applies(figure):
    """The premise: slicing the piece's own (unflipped) faces out of the placed right wing
    gives negative volumes, because the placement mirrors. piece_meshes slices the placed
    faces instead, so the right wing's pieces are the exact mirror of the left's."""
    base, v, _ = figure
    (_, left), (_, right) = A.back_wings("membrane").build_rigs(base, v)
    assert np.linalg.det(right.matrix[:3, :3]) < 0 < np.linalg.det(left.matrix[:3, :3])
    placed = right.mesh()
    a = 0
    naive = []
    for p in right.rig.pieces:
        b = a + len(p.mesh.vertices)
        naive.append(Mesh(placed.vertices[a:b], p.mesh.faces).volume())
        a = b
    assert all(x < 0 for x in naive)                                   # every piece inside-out
    lp, rp = left.piece_meshes(), right.piece_meshes()
    for (_, ml), (_, mr) in zip(lp, rp):
        assert mr.volume() == pytest.approx(ml.volume(), rel=1e-9)


@needs_assets
def test_piece_meshes_follow_the_pose_forms_of_pose(figure):
    base, v, _ = figure
    pw = dict(A.back_wings("membrane").build_rigs(base, v))["wing_l"]

    def stacked(pieces):
        return np.vstack([m.vertices for _, m in pieces])

    assert np.allclose(stacked(pw.piece_meshes()), pw.mesh().vertices)
    assert np.allclose(stacked(pw.piece_meshes("folded")), pw.pose("folded").vertices)
    assert np.allclose(stacked(pw.piece_meshes(fold=0.5)), pw.pose(fold=0.5).vertices)
    raw = pw.rig.pose_from_controls("half_fold")
    assert np.allclose(stacked(pw.piece_meshes(raw)), pw.mesh(raw).vertices)
    with pytest.raises(TypeError):
        pw.piece_meshes(raw, fold=0.5)
    assert not np.allclose(stacked(pw.piece_meshes("folded")), stacked(pw.piece_meshes()))


# -- appendage_print_shells --------------------------------------------------


@needs_assets
def test_appendage_shells_are_z_up_named_and_pass_the_gate(figure):
    base, v, body = figure
    whole = pp.appendage_print_shells([A.back_wings("membrane"), TAIL, HORN], base, v)
    assert [n for n, _ in whole] == ["wing_l", "wing_r", "tail", "horn_l", "horn_r"]
    assert pp.shell_failures(whole) == []
    feet = dict(body)["body"].bounds[0][2]
    for n, m in whole:
        assert m.bounds[0][2] > feet + 1.0, n                # above the feet: Z is up
    wing, tail = dict(whole)["wing_l"], dict(whole)["tail"]
    assert wing.vertices[:, 2].mean() > tail.vertices[:, 2].mean()   # shoulders above the pelvis
    cage = dict(body)["body"]
    assert wing.vertices[:, 1].mean() > cage.vertices[:, 1].mean()   # on the BACK (+y here)
    # the converted mesh IS the MakeHuman one turned by MH_TO_Z_UP, nothing else
    mh_wing = dict(A.back_wings("membrane").build(base, v))["wing_l"]
    assert np.allclose(wing.vertices, mh_wing.transform(MH_TO_Z_UP).vertices)


@needs_assets
@pytest.mark.parametrize("style", ["membrane", "feathered"])
def test_split_pieces_are_the_same_wing_one_shell_per_rig_piece(figure, style):
    base, v, _ = figure
    app = A.back_wings(style)
    whole = pp.appendage_print_shells([app], base, v)
    split = pp.appendage_print_shells([app], base, v, split_pieces=True)
    assert len(split) > 20 and all("/" in n for n, _ in split)
    assert {n.split("/")[0] for n, _ in split} == {"wing_l", "wing_r"}
    assert "wing_l/arm" in dict(split) and "wing_r/arm" in dict(split)
    assert pp.shell_failures(split) == []
    assert sum(len(m.faces) for _, m in split) == sum(len(m.faces) for _, m in whole)
    # the pieces of one side stack back into exactly that side's wing
    for side, mesh in whole:
        mine = [m for n, m in split if n.startswith(side + "/")]
        assert np.allclose(np.vstack([m.vertices for m in mine]), mesh.vertices)


@needs_assets
def test_a_posed_wing_prints_in_its_pose_pieces_and_whole(figure):
    base, v, _ = figure
    folded = A.back_wings("membrane", preset="folded")
    split = pp.appendage_print_shells([folded], base, v, split_pieces=True)
    whole = pp.appendage_print_shells([folded], base, v)
    rest = pp.appendage_print_shells([A.back_wings("membrane")], base, v)
    assert pp.shell_failures(split) == []
    assert np.allclose(np.vstack([m.vertices for n, m in split if n.startswith("wing_l/")]),
                       dict(whole)["wing_l"].vertices)
    assert not np.allclose(dict(whole)["wing_l"].vertices, dict(rest)["wing_l"].vertices)


@needs_assets
def test_duplicate_shell_names_are_refused(figure):
    base, v, _ = figure
    with pytest.raises(ValueError, match="same name"):
        pp.appendage_print_shells([A.back_wings("membrane"), A.back_wings("feathered")], base, v)


def test_ready_made_named_meshes_are_taken_in_the_makehuman_frame():
    blob = _ball(0.5, (1.0, 8.0, -2.0))
    (name, m), = pp.appendage_print_shells([("blob", blob)])
    assert name == "blob"
    assert np.allclose(m.vertices, blob.transform(MH_TO_Z_UP).vertices)
    assert m.vertices[:, 2].mean() == pytest.approx(8.0, abs=1e-6)       # MakeHuman y is height
    assert m.integrity_report()["volume"] > 0


# -- the frame is checked after assembly --------------------------------------


@needs_assets
@pytest.mark.parametrize("which", ["bat", "dragon", "angel", "tail", "horns"])
def test_every_appendage_root_sits_on_the_body_in_the_stl_frame(figure, which):
    base, v, body = figure
    apps = {"bat": [A.back_wings("membrane")], "dragon": [A.back_wings("dragon")],
            "angel": [A.back_wings("feathered")], "tail": [TAIL], "horns": [HORN]}[which]
    for split in (False, True):
        shells = body + pp.appendage_print_shells(apps, base, v, split_pieces=split)
        _, rep = pp.assemble_character(shells, size="medium", reference=pp.FIGURE_SHELLS)
        gaps = pr.root_gaps(rep.placed_shells)
        assert gaps and max(gaps.values()) <= EPS_MM, gaps
        assert pr.detached_appendages(rep.placed_shells, EPS_MM) == {}


@needs_assets
@pytest.mark.parametrize("which", ["bat", "dragon", "angel", "tail", "horns"])
def test_an_unconverted_makehuman_appendage_fails_the_frame_check(figure, which):
    """What the three old 'exports through the print gate' tests did: add Y-up appendages to
    Z-up shells. The gate passes (every shell is a closed solid); the roots are millimetres
    from the body, which is what this check sees."""
    base, v, body = figure
    apps = {"bat": [A.back_wings("membrane")], "dragon": [A.back_wings("dragon")],
            "angel": [A.back_wings("feathered")], "tail": [TAIL], "horns": [HORN]}[which]
    wrong = body + [s for a in apps for s in a.build(base, v)]            # NOT converted
    _, rep = pp.assemble_character(wrong, size="medium", reference=pp.FIGURE_SHELLS)
    assert rep.watertight                                                  # the gate is happy
    far = pr.detached_appendages(rep.placed_shells, EPS_MM)
    assert far and min(far.values()) > 10 * EPS_MM, far


# -- the thickness floor ------------------------------------------------------


def test_the_floor_is_scale_aware_and_leaves_what_is_thick_enough():
    tiny = 15.0 / 17.48                  # a 17.5-unit figure printed 15 mm tall: 0.86 mm per unit
    apps = [A.back_wings("membrane"),                                   # thickness 0.22 by default
            A.back_wings("feathered", scale=0.5),                       # 0.20, at half size
            A.back_wings("membrane", thickness=2.0),                    # plenty
            TAIL]
    out, changes = pp.apply_thickness_floor(apps, tiny, 0.30)
    assert [c.appendage for c in changes] == ["wing[0]", "wing[1]"]
    want = 0.30 / tiny
    assert out[0].params["thickness"] == pytest.approx(want)
    assert out[1].params["thickness"] == pytest.approx(want / 0.5)       # half-size wing: doubled
    assert out[2] is apps[2] and out[3] is apps[3]
    assert "thickness" not in apps[0].params                              # the input is not touched
    assert changes[0].old == pytest.approx(0.22) and changes[1].old == pytest.approx(0.20)
    assert "0.3 mm floor" in changes[0].as_text()
    same, none = pp.apply_thickness_floor(apps, 5.0, 0.30)               # large print: nothing to do
    assert none == [] and all(a is b for a, b in zip(same, apps))
    with pytest.raises(ValueError):
        pp.apply_thickness_floor(apps, 0.0, 0.3)


@needs_assets
def test_the_floor_option_thickens_the_wing_and_is_off_by_default(figure):
    base, v, _ = figure
    app = A.back_wings("membrane")
    mm = 15.0 / 17.48
    off = pp.appendage_print_shells([app], base, v)
    log = []
    on = pp.appendage_print_shells([app], base, v, min_thickness_mm=0.30, mm_per_unit=mm,
                                   adjustments=log)
    assert [c.param for c in log] == ["thickness"]
    assert dict(on)["wing_l"].volume() > 1.2 * dict(off)["wing_l"].volume()
    assert pp.shell_failures(on) == []
    again = pp.appendage_print_shells([app], base, v)
    assert np.array_equal(dict(again)["wing_l"].vertices, dict(off)["wing_l"].vertices)
    with pytest.raises(ValueError, match="mm_per_unit"):
        pp.appendage_print_shells([app], base, v, min_thickness_mm=0.30)
