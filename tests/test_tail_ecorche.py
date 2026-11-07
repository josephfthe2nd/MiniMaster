"""The ecorche view: the anatomical layers drawn, so they can be shown and checked.

``Placed.layers()`` gives named meshes (bone, muscle, skin); ``ecorche.layer_shells`` turns them into render
shells (optionally cut along the sagittal plane, with the muscle sleeve coloured belly -> tendon) and the CLI
(``--layers bone,muscle,skin --cut``) and ``tools/junction_view.py --layers`` use it.
"""

import numpy as np
import pytest

from minimaster.character import ecorche as E
from minimaster.character import kinds as K
from minimaster.character import mhbase as mh
from minimaster.character import portrait as po
from minimaster.core.mesh import Mesh

assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")


def test_layer_names_parse_in_anatomical_order_and_a_typo_is_named():
    assert E.parse_layers("skin,bone") == ("bone", "skin")
    assert E.parse_layers(["muscle"]) == ("muscle",)
    assert E.parse_layers("Bone, MUSCLE ,skin") == E.LAYERS
    with pytest.raises(ValueError, match="unknown layer 'flesh'.*bone, muscle, skin"):
        E.parse_layers("bone,flesh")


def test_cut_mesh_drops_the_positive_side_and_turns_the_rest_inside_out():
    tris = np.array([[0, 1, 2], [3, 4, 5]])
    v = np.array([[1.0, 0, 0], [1, 1, 0], [1, 0, 1], [-1.0, 0, 0], [-1, 1, 0], [-1, 0, 1]])
    cut = E.cut_mesh(Mesh(v, tris), (0, 0, 0), (1, 0, 0))
    assert len(cut.faces) == 1 and set(cut.faces[0]) == {3, 4, 5}        # the x < 0 face stays
    assert list(cut.faces[0]) == [5, 4, 3]                                # reversed: its inside faces the viewer
    far = E.cut_mesh(Mesh(v, tris), (5, 0, 0), (1, 0, 0))                 # a plane beyond both: nothing is cut
    assert len(far.faces) == 2


@pytest.fixture(scope="module")
def world():
    base, lib = mh.load()
    return base, lib, lib.apply(base.verts, mh.macro_weights())


@assets
def test_layer_shells_draw_each_layer_with_its_own_material_and_colour_the_sleeve(world):
    base, _, v0 = world
    p = K.instantiate_all([K.new_spec("tail")], base, v0)[0]
    shells = E.layer_shells(base, v0, [p], ("bone", "muscle", "skin"), body=False)
    assert [s.material for s in shells] == [E.BONE, E.MUSCLE, p.material]
    assert shells[0].base_colors is None and shells[2].base_colors is None
    cols = shells[1].base_colors                                          # belly -> tendon on the sleeve
    assert cols is not None and len(cols) == len(shells[1].mesh.vertices)
    muscle_red, tendon = (np.asarray(c) for c in (po.sh.hex_to_linear(E.MUSCLE.base_color),
                                                  po.sh.hex_to_linear(E.TENDON)))
    assert np.allclose(cols[0], muscle_red) and np.allclose(cols[-1], tendon)   # root pole: belly; tip pole: tendon
    # the model's belly fraction is the same one that colours it
    assert np.allclose(p.model.model().muscle_belly()[:5], 1.0)
    # layers that a kind does not have are left out; a horn has the skin only
    h = K.instantiate_all([K.new_spec("horn")], base, v0)
    assert len(E.layer_shells(base, v0, h, ("bone", "muscle", "skin"), body=False)) == 2   # a pair: two skins


@assets
def test_a_cut_view_removes_half_of_the_body_and_the_outer_layers_but_keeps_the_innermost(world):
    base, _, v0 = world
    p = K.instantiate_all([K.new_spec("tail")], base, v0)[0]
    whole = E.layer_shells(base, v0, [p], ("bone", "muscle", "skin"))
    cut = E.layer_shells(base, v0, [p], ("bone", "muscle", "skin"), cut=True)
    n = lambda shells: [len(s.mesh.faces) for s in shells]                  # noqa: E731
    wn, cn = n(whole), n(cut)
    assert cn[0] < 0.7 * wn[0]                                             # the body: about half
    assert cn[-3] == wn[-3]                                                 # the bone is whole ...
    assert cn[-2] < 0.7 * wn[-2] and cn[-1] < 0.7 * wn[-1]                 # ... the muscle and the skin are cut
    # what stays is on the x <= anchor side (Z-up frame has the same x)
    for s in cut[-2:]:
        used = np.unique(s.mesh.faces)
        assert s.mesh.vertices[used][:, 0].min() < p.anchor[0] - 0.3
        assert (s.mesh.vertices[s.mesh.faces].mean(axis=1)[:, 0] <= p.anchor[0] + 1e-9).all()


@assets
def test_the_cli_draws_the_layers_and_names_an_unknown_one(tmp_path, capsys):
    from minimaster.__main__ import main
    png = tmp_path / "layers.png"
    assert main(["character", "--add", "tail", "--layers", "bone,muscle,skin", "--cut", "--azimuth", "90",
                 "--elevation", "3", "-o", str(png)]) == 0
    assert png.stat().st_size > 1000
    plain = tmp_path / "plain.png"
    assert main(["character", "--add", "tail", "--azimuth", "90", "--elevation", "3", "-o", str(plain)]) == 0
    assert png.read_bytes() != plain.read_bytes()                           # a different picture
    assert main(["character", "--add", "tail", "--layers", "bone,flesh", "-o", str(tmp_path / "x.png")]) == 2
    assert "unknown layer 'flesh'" in capsys.readouterr().err


@assets
def test_the_junction_view_tool_takes_layers(tmp_path):
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "tools" / "junction_view.py"
    spec = importlib.util.spec_from_file_location("junction_view_tool", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    out = tmp_path / "jv.png"
    assert tool.main([str(out), "tail", "tail", "90,3", "--layers", "bone,muscle", "--cut", "--size", "120x120"]) == 0
    assert out.stat().st_size > 500
