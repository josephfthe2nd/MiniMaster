"""The character print pipeline: scale, stand, base, gate."""

import numpy as np
import pytest

from minimaster.character import mhbase as mh
from minimaster.character import printprep as pp
from minimaster.core import primitives
from minimaster.core.mesh import Mesh


def _ball(r=1.0, at=(0.0, 0.0, 0.0)):
    return primitives.icosphere(radius=r, subdivisions=2).translated(at)


def test_resolve_height_presets_and_custom():
    assert pp.resolve_height(size="medium") == 32.0
    assert pp.resolve_height(height=41.5) == 41.5
    assert pp.resolve_height() == 32.0  # default
    with pytest.raises(pp.PrintError):
        pp.resolve_height(size="medium", height=32.0)
    with pytest.raises(pp.PrintError):
        pp.resolve_height(size="enormous")
    with pytest.raises(pp.PrintError):
        pp.resolve_height(height=0.0)


def test_scales_to_exact_height_and_stands_on_zero():
    shells = [("a", _ball(2.0))]
    mesh, rep = pp.assemble_character(shells, height=30.0, with_base=False)
    lo, hi = mesh.bounds
    assert hi[2] - lo[2] == pytest.approx(30.0, rel=1e-9)
    assert lo[2] == pytest.approx(0.0, abs=1e-9)
    assert (lo[0] + hi[0]) == pytest.approx(0.0, abs=1e-9)  # centred on XY
    assert rep.height_mm == 30.0
    assert rep.scale == pytest.approx(30.0 / 4.0)


def test_base_is_added_below_the_figure():
    shells = [("a", _ball(1.0))]
    with_base, _ = pp.assemble_character(shells, height=32.0, with_base=True)
    without, _ = pp.assemble_character(shells, height=32.0, with_base=False)
    assert len(with_base.faces) > len(without.faces)
    assert with_base.bounds[0][2] < without.bounds[0][2]  # base sits under z=0


def test_gate_rejects_open_shells():
    # a sphere with a face punched out: has real height, but is not a solid
    full = _ball(1.0)
    holed = Mesh(full.vertices, full.faces[1:])
    with pytest.raises(pp.PrintError, match="not watertight"):
        pp.assemble_character([("bad", holed)], height=30.0)


def test_gate_rejects_zero_height():
    flat = Mesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                np.array([[0, 1, 2]], dtype=np.int64))
    with pytest.raises(pp.PrintError, match="no height"):
        pp.assemble_character([("flat", flat)], height=30.0)


def test_gate_rejects_inverted_shells():
    m = _ball(1.0)
    flipped = Mesh(m.vertices, m.faces[:, [0, 2, 1]])
    with pytest.raises(pp.PrintError, match="inside-out"):
        pp.assemble_character([("flipped", flipped)], height=30.0)


def test_gate_rejects_empty():
    with pytest.raises(pp.PrintError, match="no geometry"):
        pp.assemble_character([], height=30.0)


def test_overlapping_shells_are_allowed():
    """The export contract is overlapping watertight shells; the slicer unions
    them. Two intersecting balls must pass."""
    shells = [("a", _ball(1.0)), ("b", _ball(1.0, (0.5, 0, 0)))]
    mesh, rep = pp.assemble_character(shells, height=30.0, with_base=False)
    assert rep.watertight
    assert rep.shells == 2


def test_report_text_and_notes():
    shells = [("a", _ball(1.0))]
    _, small = pp.assemble_character(shells, size="tiny", with_base=False)
    _, big = pp.assemble_character(shells, size="display", with_base=False)
    assert "watertight" in small.as_text()
    assert small.volume_mm3 < big.volume_mm3
    assert any("facial geometry" in n for n in big.notes)


@pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
def test_real_character_exports_watertight():
    base, lib = mh.load()
    v = lib.apply(base.verts, mh.macro_weights(gender=1.0, muscle=0.7))
    shells = pp.character_print_shells(base, v)
    names = [n for n, _ in shells]
    assert "body" in names and "l-eye" in names and "r-eye" in names
    mesh, rep = pp.assemble_character(shells, size="medium")
    assert rep.watertight
    assert rep.height_mm == 32.0
    lo, hi = mesh.bounds
    assert hi[2] - lo[2] > 32.0        # base adds height under the figure
    assert rep.volume_mm3 > 0


@pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
def test_print_shells_only_includes_closed_islands():
    base = mh.BaseMesh()
    for name, mesh in pp.character_print_shells(base, with_teeth=True):
        rep = mesh.integrity_report()
        assert rep["watertight"], name
        assert rep["volume"] > 0, name


@pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
def test_morphs_do_not_break_printability():
    base, lib = mh.load()
    for kw in ({"gender": 0.0}, {"gender": 1.0, "weight": 1.0},
               {"age": 0.1875}, {"muscle": 1.0, "height": 1.0}):
        v = lib.apply(base.verts, mh.macro_weights(**kw))
        _, rep = pp.assemble_character(
            pp.character_print_shells(base, v), size="medium")
        assert rep.watertight, kw
