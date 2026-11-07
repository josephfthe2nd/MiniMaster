"""The kind registry, placement that follows the body, and everything built from a record.

The first half needs no MakeHuman assets (parameters, controls, parsing); the second builds
appendages on the real morphed body and is skipped when the CC0 assets are not fetched.
"""

import inspect
import math
import warnings

import numpy as np
import pytest

from minimaster.character import appendages as A
from minimaster.character import kinds as K
from minimaster.character import mhbase as mh
from minimaster.character import wings as W
from minimaster.character.appendage_spec import AppendageSpec
from minimaster.character.sliders import Character
from minimaster.core.armature import Armature
from minimaster.core.subdiv import quads_to_mesh

assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")


# -- parameters ---------------------------------------------------------------------

def test_the_four_kinds_keep_their_ids():
    assert set(K.KINDS) == {"wing", "tail", "horn", "harvest"}
    # creature_anatomy keys its consequences by these ids; renaming one would disable them
    from minimaster.character import creature_anatomy as ca
    assert set(K.KINDS) == set(ca.CONSEQUENCES)
    assert {k for k, kind in K.KINDS.items() if kind.paired} == set(ca.PAIRED_KINDS)


def test_every_parameter_is_a_scalar_with_sane_limits():
    structured = {"scallop", "spec", "root", "claw", "root_taper"}
    for kind in K.KINDS.values():
        for p in kind.params:
            assert p.type in ("float", "int", "bool", "choice"), (kind.id, p.key)
            assert p.key not in structured, (kind.id, p.key)
            if p.type in ("float", "int"):
                assert p.lo < p.hi and p.lo <= p.default <= p.hi, (kind.id, p.key)
                assert p.step and p.step > 0, (kind.id, p.key)
            if p.type == "choice":
                assert p.default in p.choices, (kind.id, p.key)
            if p.when is not None:
                controller = kind.param(p.when[0])
                assert controller is not None and controller.type == "choice", (kind.id, p.key)
                assert set(p.when[1]) <= set(controller.choices), (kind.id, p.key)


def test_tertials_cannot_be_zero():
    """tertials=0 crashed the feathered builder (a broadcast error), so the range starts at 1."""
    p = K.KINDS["wing"].param("tertials")
    assert p.lo >= 1
    values, notes = K.KINDS["wing"].clean({"style": "feathered", "tertials": 0})
    assert values["tertials"] == 1 and any("outside" in n for n in notes)


def _builder_defaults(fn):
    return {n: p.default for n, p in inspect.signature(fn).parameters.items()}


@pytest.mark.parametrize("style", ["membrane", "dragon", "feathered"])
def test_wing_defaults_are_what_the_builders_really_use(style):
    """A parameter default that drifted from the builder's would make a freshly added wing
    differ from ``back_wings()``."""
    kind = K.KINDS["wing"]
    values = kind.defaults(style=style)
    real = _builder_defaults(W.feathered_pieces if style == "feathered" else W.membrane_pieces)
    if style == "dragon":
        real.update({k: v for k, v in W.DRAGON.items() if k in real})
    for key, value in values.items():
        if key in ("style", "fillet"):      # the style is not a builder argument; the fillet is
            continue                        # how the BODY'S SKIN meets the wing (junction.py)
        assert key in real, f"{key} is not a {style} builder argument"
        assert value == pytest.approx(real[key]), (style, key)


@pytest.mark.parametrize("kind_id,fn", [("tail", A.make_tail), ("horn", A.make_horn)])
def test_loft_defaults_are_what_the_builders_really_use(kind_id, fn):
    kind = K.KINDS[kind_id]
    real = _builder_defaults(fn)
    # the registry adds the horn's ``shape``; the builders' own ``tip_radius`` stays None =
    # the first version's point (a legacy Appendage is untouched) while the registry's
    # default is rounded; the tail's ``lift`` / ``sway`` are pose controls, not parameters
    # (``fillet`` is how the body's skin meets the root, junction.py; ``flare_length`` stays the
    # builders' own; ``ring_verts`` and ``flare`` have registry defaults that are rounder and
    # broader than the builders': 16 facets, so the skin's seam with the tube is a clean curve)
    # (the registry tail is the anatomical one, tailrig.py: it adds the caudal vertebra count and a
    # dorsal ridge to the legacy builder's parameters, and its ``flare`` is the root trumpet, 0.35)
    # (the registry horn is the anatomical one, hornrig.py: it adds the growth rings, the tilt from the
    # skull and the skin boss to the legacy builder's parameters, and its numbers are the research's
    # (a 1.6 long horn of base radius 0.17 on the neutral head, radius and length scale with the head),
    # not the first version's 2.2 / 0.42: the legacy builder keeps its own defaults, test_limbs.py)
    registry_only = {"tail": {"fillet", "vertebrae", "crest"}, "horn": {"shape", "fillet", "rings", "tilt", "boss"}}[kind_id]
    builder_only = {"tail": {"lift", "sway", "flare_length"}, "horn": {"flare_length"}}[kind_id]
    assert set(kind.defaults()) - registry_only == set(real) - builder_only
    anatomical = {"horn": {"length", "base_radius", "segments", "curve", "twist", "turns"}}.get(kind_id, set())
    for key, value in kind.defaults().items():
        if key in real and key not in {"tip_radius", "ring_verts", "flare"} | anatomical:
            assert value == pytest.approx(real[key]), key
    assert kind.defaults()["ring_verts"] == 16 and real["ring_verts"] in (8, 10)
    assert real["tip_radius"] is None and kind.defaults()["tip_radius"] > 0     # blunt by default


def test_when_selects_the_parameters_of_a_style():
    kind = K.KINDS["wing"]
    bat, bird = kind.defaults(style="membrane"), kind.defaults(style="feathered")
    assert "gull" in bat and "primaries" not in bat
    assert "primaries" in bird and "gull" not in bird
    assert kind.defaults(style="dragon")["spike"] == 0.30 and bat["spike"] == 0.10
    assert kind.param("spike", {"style": "dragon"}).default == 0.30
    # a parameter that does not apply is neither coerced nor complained about
    values, notes = kind.clean({"style": "feathered", "gull": "not a number"})
    assert "gull" not in values and notes == []


def test_param_coercion():
    f = K.ParamSpec("x", "X", "float", 1.0, 0.0, 2.0, 0.1)
    assert f.coerce("1.5") == 1.5 and f.coerce(5) == 2.0 and f.coerce(-1) == 0.0
    for bad in (float("nan"), float("inf"), "abc", None, True):
        with pytest.raises(ValueError):
            f.coerce(bad)
    i = K.ParamSpec("n", "N", "int", 3, 1, 9, 1)
    assert i.coerce(2.6) == 3 and isinstance(i.coerce(2.6), int) and i.coerce(100) == 9
    b = K.ParamSpec("b", "B", "bool", True)
    assert b.coerce("false") is False and b.coerce(1) is True and b.coerce(True) is True
    with pytest.raises(ValueError):
        b.coerce("maybe")
    c = K.ParamSpec("c", "C", "choice", "a", choices=("a", "b"))
    assert c.coerce("B") == "b"
    with pytest.raises(ValueError, match="one of"):
        c.coerce("z")


def test_clean_keeps_unknown_keys_out_of_the_build_and_says_so():
    values, notes = K.KINDS["tail"].clean({"length": 8.0, "wobble": 1, "curl": "x"}, "tail-1: ")
    assert values["length"] == 8.0 and "wobble" not in values
    assert values["curl"] == 0.55                      # invalid value -> the default
    assert any("wobble" in n and "kept in the file" in n for n in notes)
    assert any("curl" in n and "default" in n for n in notes)
    near, _ = K.KINDS["tail"].clean({"lenght": 3})
    assert near["length"] == 6.0
    _, hint = K.KINDS["tail"].clean({"lenght": 3})
    assert any("did you mean 'length'" in n for n in hint)


def test_an_unknown_style_is_fatal_for_that_record_only():
    with pytest.raises(K.SpecError, match="gryphon"):
        K.KINDS["wing"].clean({"style": "gryphon"})
    with pytest.raises(K.SpecError, match="limb"):
        K.KINDS["harvest"].clean({"limb": "tentacle"})


def test_wing_controls_are_sanitised_before_the_rig_sees_them():
    kind = K.KINDS["wing"]
    ctrl, notes = kind.clean_controls(
        {"fold": 0.5, "raise_": "high", "levitate": 1, "preset": "hover", "swing": float("nan"),
         "twist": True}, "w: ")
    assert ctrl == {"fold": 0.5}
    joined = " ".join(notes)
    for word in ("raise_", "levitate", "hover", "swing", "twist"):
        assert word in joined
    assert kind.clean_controls({"preset": "folded", "fan": 0.5})[0] == {"preset": "folded", "fan": 0.5}
    # kinds without controls keep the record and say so
    assert K.KINDS["tail"].clean_controls({"fold": 1})[0] == {}


def test_controls_can_differ_per_side():
    both = {"fold": 0.2, "*": {"raise_": 5}, "l": {"fold": 0.9}, "r": {"preset": "folded"}}
    assert K.side_controls(both, "l") == {"fold": 0.9, "raise_": 5}
    assert K.side_controls(both, "r") == {"fold": 0.2, "raise_": 5, "preset": "folded"}
    assert K.side_controls(both, "") == {"fold": 0.2, "raise_": 5}
    assert K.side_controls({}, "l") == {}


def test_free_control_ranges_depend_on_the_other_controls():
    kind = K.KINDS["wing"]
    lo0, hi0 = kind.control_range("swing", {"fold": 0.0})
    lo1, hi1 = kind.control_range("swing", {"fold": 1.0})
    assert (lo0, hi0) == (-30.0, 85.0) and hi1 < hi0 and lo1 > lo0
    assert kind.control_range("nonsense") is None
    assert K.KINDS["tail"].control_range("fold") is None


# -- creating records ----------------------------------------------------------------------

def test_new_spec_places_by_preset_and_picks_ids():
    a = K.new_spec("wing")
    assert (a.id, a.side, a.params) == ("wing-1", "both", {})
    assert a.placement["anchor"] == {"face": K.PRESET_FACES["shoulder_blades"]}
    assert a.placement["preset"] == "shoulder_blades"
    assert a.placement["roll"] == A.WING_ROLL and a.placement["offset"] == A.WING_OFFSET
    b = K.new_spec("wing", [a], side="left", params={"style": "feathered", "span": 15})
    assert b.id == "wing-2" and b.side == "left" and b.params == {"style": "feathered", "span": 15.0}
    t = K.new_spec("tail", [a, b])
    assert t.id == "tail-1" and t.side == "center" and t.placement["offset"] == K.TAIL_SINK
    assert K.new_spec("horn").placement["offset"] == K.HORN_SINK
    # the extra limb's preset follows its limb
    leg = K.new_spec("harvest", params={"limb": "leg"})
    assert leg.placement["preset"] == "extra_legs"
    assert K.new_spec("harvest").placement["preset"] == "lower_arms"
    # ... and a preset implies its limb
    assert K.new_spec("harvest", preset="extra_legs").params == {"limb": "leg"}
    assert K.new_spec("harvest", preset="upper_arms").params == {"limb": "arm"}
    with pytest.raises(ValueError, match="is for limb='leg'"):
        K.new_spec("harvest", preset="extra_legs", params={"limb": "arm"})
    assert K.new_spec("tail", spec_id="my-tail").id == "my-tail"


def test_new_spec_is_strict_because_a_person_typed_it():
    with pytest.raises(ValueError, match="did you mean 'wing'"):
        K.new_spec("wng")
    with pytest.raises(ValueError, match="did you mean 'length'"):
        K.new_spec("tail", params={"lenght": 3})
    with pytest.raises(ValueError, match="only applies when style is"):
        K.new_spec("wing", params={"style": "feathered", "gull": 0.2})
    with pytest.raises(ValueError, match="outside the range"):
        K.new_spec("wing", params={"span": 99})
    with pytest.raises(ValueError, match="one of"):
        K.new_spec("wing", params={"style": "gryphon"})
    with pytest.raises(ValueError, match="side"):
        K.new_spec("wing", side="up")
    with pytest.raises(ValueError, match="preset"):
        K.new_spec("wing", preset="elbow")
    with pytest.raises(ValueError, match="does not use 'roll'"):       # a harvest has no roll ...
        K.new_spec("harvest", roll=30.0)
    assert K.new_spec("harvest", scale=0.8).placement["scale"] == 0.8  # ... but honours scale
    with pytest.raises(ValueError, match="already"):
        K.new_spec("tail", [K.new_spec("tail")], spec_id="tail-1")
    with pytest.raises(ValueError, match="not a valid id"):
        K.new_spec("tail", spec_id="a.b")


def test_the_command_line_grammar():
    assert K.parse_add("tail") == ("tail", {}, {})
    kind, params, meta = K.parse_add("wing:style=feathered,span=15,side=left,at=0.8/4.4/-0.7")
    assert kind == "wing" and params == {"style": "feathered", "span": 15}
    assert meta == {"side": "left", "at": (0.8, 4.4, -0.7)}
    assert K.parse_kv("a=1,b=2.5,c=true,d=False,e=name") == {
        "a": 1, "b": 2.5, "c": True, "d": False, "e": "name"}
    with pytest.raises(ValueError, match="key=value"):
        K.parse_kv("oops")
    with pytest.raises(ValueError, match="KIND"):
        K.parse_add(":a=1")
    with pytest.raises(ValueError, match="three numbers"):
        K.parse_add("tail:at=1/2")
    spec = K.add_from_text("wing:style=dragon,id=bat,name=Big,scale=1.5")
    assert spec.id == "bat" and spec.name == "Big" and spec.placement["scale"] == 1.5


def test_pose_text_goes_to_shared_or_one_side():
    wing = K.new_spec("wing")
    spec, side, vals = K.pose_from_text("wing-1:fold=0.6,preset=folded", [wing])
    K.apply_pose(spec, side, vals)
    spec, side, vals = K.pose_from_text("wing-1.r:raise_=20", [wing])
    K.apply_pose(spec, side, vals)
    assert wing.controls == {"*": {"fold": 0.6, "preset": "folded"}, "r": {"raise_": 20}}
    assert K.side_controls(wing.controls, "r") == {"fold": 0.6, "preset": "folded", "raise_": 20}
    with pytest.raises(ValueError, match="no appendage 'wing-9'"):
        K.pose_from_text("wing-9:fold=1", [wing])
    with pytest.raises(ValueError, match="did you mean 'fold'"):
        K.pose_from_text("wing-1:flod=1", [wing])
    with pytest.raises(ValueError, match="no controls"):
        K.pose_from_text("horn-1:fold=1", [K.new_spec("horn")])
    with pytest.raises(ValueError, match="has no control 'fold'"):
        K.pose_from_text("tail-1:fold=1", [K.new_spec("tail")])
    with pytest.raises(ValueError, match=r"\.l or \.r"):
        K.pose_from_text("wing-1.x:fold=1", [wing])


def test_the_kind_listing_names_every_kind_and_parameter():
    text = K.format_kinds()
    for kind in K.KINDS.values():
        assert text.count(f"\n{kind.id}  (") + text.startswith(f"{kind.id}  (") == 1
        for p in kind.params:
            assert p.key in text
    assert "style: membrane|dragon" in text and "pose controls" in text


# -- building: unknown things are skipped with a warning, never deleted ------------------------

@pytest.fixture(scope="module")
def world():
    base, lib = mh.load()
    return base, lib, lib.apply(base.verts, mh.macro_weights())


def _body(world, **macro):
    base, lib, _ = world
    return lib.apply(base.verts, mh.macro_weights(**macro))


@assets
def test_unknown_kinds_params_styles_and_controls_are_skipped_with_a_warning(world):
    base, _, v0 = world
    doc = Character.from_dict({"appendages": [
        {"id": "gizzard-1", "kind": "gizzard", "params": {"x": 1}, "future": True},
        {"id": "wing-1", "kind": "wing", "params": {"style": "gryphon"}},
        {"id": "tail-1", "kind": "tail", "params": {"length": 7.0, "wobble": 3},
         "controls": {"fold": 1}},
        {"id": "horn-1", "kind": "horn", "controls": {"fold": 1}},
        {"id": "wing-2", "kind": "wing", "side": "left", "params": {"style": "membrane"},
         "controls": {"levitate": 1, "fold": 0.5, "preset": "hover"}},
    ]})
    before = doc.to_dict()
    with pytest.warns(K.AppendageWarning) as caught:
        placed = doc.instantiate_appendages(base, v0)
    text = " | ".join(str(w.message) for w in caught)
    assert [p.name for p in placed] == ["tail-1", "horn-1", "wing-2.l"]
    for word in ("gizzard", "gryphon", "wobble", "has no pose controls", "levitate", "hover",
                 "tail has no control 'fold'"):
        assert word in text, word
    assert doc.to_dict() == before                 # the document is untouched


@assets
def test_disabled_and_duplicate_records_are_not_built(world):
    base, _, v0 = world
    off = K.new_spec("tail")
    off.enabled = False
    a, b = K.new_spec("horn"), K.new_spec("horn")
    b.id = a.id                                    # a hand-edited duplicate
    with pytest.warns(K.AppendageWarning, match="duplicated"):
        placed = K.instantiate_all([off, a, b], base, v0)
    assert [p.name for p in placed] == ["horn-1.l", "horn-1.r"]


@assets
def test_a_harvest_that_selects_nothing_is_skipped_not_fatal(world):
    base, _, v0 = world
    spec = K.new_spec("harvest", side="left")
    spec.params["radius"] = 0.9
    spec.placement["anchor"] = {"face": 7954}
    away = v0.copy()
    away[base.body_verts_idx] += [100.0, 0.0, 0.0]      # the body is nowhere near its own joints
    with pytest.warns(K.AppendageWarning, match="cannot cut the arm"):
        placed = K.instantiate_all([spec, K.new_spec("horn")], base, away)
    assert [p.name for p in placed] == ["horn-1.l", "horn-1.r"]    # the horn after it is built


@assets
def test_anchor_weights_place_off_centre_and_mirror_exactly(world):
    base, _, v0 = world
    plain = K.new_spec("horn")
    off = K.new_spec("horn")
    off.placement["anchor"]["weights"] = [0.7, 0.1, 0.15, 0.05]
    a, b = K.instantiate_all([plain], base, v0), K.instantiate_all([off], base, v0)
    assert not np.allclose(a[0].anchor, b[0].anchor)
    assert np.allclose(b[1].anchor, b[0].anchor * [-1.0, 1.0, 1.0], atol=1e-12)
    assert np.allclose(b[1].mesh().vertices, b[0].mesh().vertices * [-1.0, 1.0, 1.0], atol=1e-9)
    bad = K.new_spec("horn")
    bad.placement["anchor"]["weights"] = [1.0, 2.0]
    with pytest.warns(K.AppendageWarning, match="one finite number per corner"):
        assert K.instantiate_all([bad], base, v0) == []


@assets
def test_an_anchor_that_is_not_a_quad_of_this_body_is_skipped(world):
    base, _, v0 = world
    far = K.new_spec("tail")
    far.placement["anchor"] = {"face": 999999}
    moved = K.new_spec("horn", base=base)
    moved.placement["anchor"]["corners"][0] += 1     # saved against a different base mesh
    with pytest.warns(K.AppendageWarning) as caught:
        placed = K.instantiate_all([far, moved], base, v0)
    assert placed == []
    text = " ".join(str(w.message) for w in caught)
    assert "999999" in text and "different base mesh" in text


@assets
def test_a_v1_world_point_becomes_a_body_relative_anchor(world):
    """A record that only has a world POINT is converted once, on the neutral body, to the
    quad nearest it; its mirror image (x < 0) reads as the left-hand definition."""
    base, _, v0 = world
    preset = K.instantiate_all([K.new_spec("horn")], base, v0)
    by_point = AppendageSpec(id="horn-1", kind="horn", side="both",
                             placement={"anchor": {"point": [0.85, 8.55, 0.05]}})
    mirrored = AppendageSpec(id="horn-1", kind="horn", side="both",
                             placement={"anchor": {"point": [-0.85, 8.55, 0.05]}})
    for spec in (by_point, mirrored):
        got = K.instantiate_all([spec], base, v0)
        for a, b in zip(preset, got):
            assert np.array_equal(a.mesh().vertices, b.mesh().vertices)
    # and on another body the point does NOT re-snap to a different quad
    tall = _body(world, height=1.0)
    a = K.instantiate_all([K.new_spec("horn")], base, tall)
    b = K.instantiate_all([by_point], base, tall)
    assert np.array_equal(a[0].mesh().vertices, b[0].mesh().vertices)


@assets
def test_presets_are_the_nearest_quads_to_their_points_on_the_neutral_body(world):
    """The preset quad ids are written down in kinds.py so that placing one needs no
    morph library. This recomputes them: it fails if the base mesh ever changes under them."""
    base, _, v0 = world
    cage = base.body_cage(v0)
    assert set(K.PRESET_FACES) == set(K.PRESET_POINTS)
    for name, point in K.PRESET_POINTS.items():
        assert K.nearest_face(cage, point) == K.PRESET_FACES[name], name
    assert K.neutral_cage(base)[1] is not None
    n = K.nearest_face(K.neutral_cage(base), K.PRESET_POINTS["waist"])
    centroid = cage[0][cage[1][n]].mean(axis=0)
    assert np.allclose(centroid + K.WING_ROOT_OFFSET, A.WING_ROOT, atol=1e-12)


# -- the legacy engine is untouched ------------------------------------------------------------

def test_the_new_appendage_fields_default_to_the_legacy_behaviour():
    a = A.Appendage("wing", mirror=True)
    assert (a.anchor_face, a.anchor_weights, a.root_face, a.side) == (None, None, None, None)
    assert a._indices() == [0, 1] and A.Appendage("tail")._indices() == [0]
    assert [a._tag(i) for i in (0, 1)] == ["_l", "_r"] and A.Appendage("tail")._tag(0) == ""
    sided = {s: A.Appendage("horn", side=s) for s in ("center", "left", "right", "both")}
    assert [s._indices() for s in sided.values()] == [[0], [0], [1], [0, 1]]
    assert sided["center"]._tag(0) == "" and sided["right"]._tag(1) == "_r"
    with pytest.raises(ValueError, match="unknown side"):
        A.Appendage("horn", side="up")._indices()
    # back_wings keeps its absolute world-point root: the body-relative root is a kind preset
    app = A.back_wings("membrane")
    assert app.root_point == A.WING_ROOT and not app.root_relative and app.root_face is None


# -- the left-right machinery --------------------------------------------------------------

@assets
def test_the_mirror_face_map_is_exact_and_an_involution(world):
    base, _, _ = world
    m = A.mirror_face_map(base)
    bv, bq = base.body_cage(None)
    c = bv[bq].mean(axis=1)
    assert len(m) == len(bq) == 13378
    assert np.array_equal(m[m], np.arange(len(m)))
    assert np.abs(c[m] - c * np.array([-1.0, 1.0, 1.0])).max() < 1e-12
    assert A.mirror_face_map(base) is m                 # computed once per base, cached
    vm = A.mirror_vertex_map(base)
    assert np.array_equal(vm[vm], np.arange(len(vm)))
    assert np.abs(bv[vm] - bv * np.array([-1.0, 1.0, 1.0])).max() < 1e-12
    with pytest.raises(ValueError, match="not mirror-symmetric"):
        A._mirror_index_map(np.array([[1.0, 0, 0], [3.0, 0, 0]]))


@assets
def test_mirrored_anchor_weights_follow_the_corner_vertices(world):
    """A quad and its mirror image store their corners in a different order, so a
    non-uniform anchor must be permuted, not copied."""
    base, _, v0 = world
    bv, bq = base.body_cage(v0)
    f = 10296
    w = (0.7, 0.1, 0.15, 0.05)
    app = A.Appendage("tail", anchor_face=f, anchor_weights=w, side="both")
    hint = np.array([0.0, 1.0, 0.0])
    left = app._face_anchor(bv, bq, 0, hint, base).resolve(bv)[0]
    right = app._face_anchor(bv, bq, 1, hint * [-1, 1, 1], base).resolve(bv)[0]
    assert np.allclose(right, left * [-1.0, 1.0, 1.0], atol=1e-12)
    assert not np.array_equal(bq[A.mirror_face_map(base)[f]], bq[f][::-1])    # order does differ
    # uniform weights are permutation-free, and a missing base is a clear error
    plain = A.Appendage("tail", anchor_face=f)
    with pytest.raises(ValueError, match="mirror-face map"):
        plain._anchor_frame(bv, bq, 1, np.zeros(3))


@assets
def test_the_side_prefix_swap_is_right_for_every_joint_marker(world):
    """Regression: the old chained replace() turned 'l-finger-2-1' into 'r-fingel-2-1' and
    'l-upper-leg' into 'r-uppel-leg' (21 of the 55 left-hand markers)."""
    base, _, _ = world
    markers = [g[len("joint-"):] for g in base.groups if g.startswith("joint-l-")]
    assert len(markers) == 55

    def old(c):                                             # the code this replaced
        return c.replace("l-", "@").replace("r-", "l-").replace("@", "r-")

    assert sum(1 for m in markers if f"joint-{old(m)}" not in base.groups) == 21
    for m in markers:
        r = A.mirror_joint_name(m)
        assert r == "r-" + m[2:] and f"joint-{r}" in base.groups, m
        assert A.mirror_joint_name(r) == m
        assert A.mirror_joint_name("joint-" + m) == "joint-" + r
    for mid in ("pelvis", "spine-1", "head", "neck", "joint-ground"):
        assert A.mirror_joint_name(mid) == mid


@assets
def test_a_mirrored_leg_harvest_builds(world):
    """Regression: KeyError \"no group 'joint-r-uppel-leg'\" for the right-hand leg."""
    base, _, v0 = world
    app = A.Appendage("harvest", anchor_point=(1.7, 0.4, 0.1), mirror=True,
                      chain=("l-upper-leg", "l-knee", "l-ankle"),
                      params=dict(radius=1.5, trim=0.6))
    (nl, left), (nr, right) = app.build(base, v0)
    assert (nl, nr) == ("harvest_l", "harvest_r")
    assert left.integrity_report()["watertight"] and right.integrity_report()["watertight"]
    assert abs(left.volume() - right.volume()) < 0.2 * left.volume()


# -- placement follows the body ------------------------------------------------------------------

def _wing_shoulder(pw):
    return pw.matrix[:3, :3] @ pw.rig.skeleton.joints["shoulder"] + pw.matrix[:3, 3]


BODIES = {
    "neutral": {}, "tall": dict(height=1.0), "short": dict(height=0.0),
    "height .25": dict(height=0.25), "height .75": dict(height=0.75),
    "child": dict(age=0.1875), "old": dict(age=1.0), "baby-ish": dict(age=0.1),
    "heavy": dict(weight=1.0, gender=1.0, muscle=1.0), "slim": dict(weight=0.0, gender=0.0),
    "weight 1": dict(weight=1.0), "weight 0": dict(weight=0.0)}


@assets
@pytest.mark.parametrize("style", ["membrane", "feathered"])
def test_wing_shoulders_stay_on_the_scapula_across_bodies(world, style):
    """Acceptance (bug 1): the wing shoulder keeps its place relative to the scapula marker
    across macro height 0..1, age, weight and the tall / short / child / heavy bodies."""
    base, _, v0 = world
    spec = K.new_spec("wing", params={"style": style}, base=base)

    def offset(v):
        pw = K.instantiate_all([spec], base, v)[0]
        return _wing_shoulder(pw.rig) - A.joint_position(base, "l-scapula", v), pw

    ref, _ = offset(v0)
    worst = 0.0
    for name, kw in BODIES.items():
        d, _ = offset(_body(world, **kw))
        assert np.linalg.norm(d) < 2.0, (name, d)                      # sits ON the shoulder blade
        worst = max(worst, abs(d[1] - ref[1]))
    assert worst < 0.6, f"the wing shoulder moves {worst:.2f} units against the scapula"


@assets
def test_the_old_world_point_anchor_fails_the_same_tolerance(world):
    """The control for the test above: the legacy placement (a world point re-snapped on
    every build) puts the shoulder 6+ units off the scapula on a tall body, so the
    tolerance is one the bug could not pass."""
    base, _, v0 = world
    scap = lambda v: A.joint_position(base, "l-scapula", v)            # noqa: E731

    def legacy_dy(v):
        pw = A.back_wings("membrane").build_rigs(base, v)[0][1]
        return (_wing_shoulder(pw) - scap(v))[1]

    ref = legacy_dy(v0)
    assert abs(legacy_dy(_body(world, height=1.0)) - ref) > 3.0
    assert abs(legacy_dy(_body(world, height=0.0)) - ref) > 2.0


@assets
def test_the_membrane_root_follows_the_waist(world):
    """The second world-fixed constant: WING_ROOT. With only the origin on a quad the
    shoulder-to-root distance in wing units, measured against the torso, varied 0.3 .. 1.4
    across bodies (1.3 .. 9.9 units); aimed at a waist QUAD it is the same fraction of the
    torso on every body (0.72 .. 0.92)."""
    base, _, v0 = world
    spec = K.new_spec("wing", params={"style": "membrane"}, base=base)
    app = A.back_wings("membrane")
    app.anchor_face = K.PRESET_FACES["shoulder_blades"]          # the half-fix: origin on a quad,
    rel, hybrid = [], []                                          # root still a world point
    for name, kw in BODIES.items():
        v = _body(world, **kw)
        torso = np.linalg.norm(A.joint_position(base, "l-scapula", v)
                               - A.joint_position(base, "pelvis", v))
        sk = K.instantiate_all([spec], base, v)[0].rig.rig.skeleton
        rel.append(np.linalg.norm(sk.joints["root"] - sk.joints["shoulder"]) / torso)
        sk0 = app.build_rigs(base, v)[0][1].rig.skeleton
        hybrid.append(np.linalg.norm(sk0.joints["root"] - sk0.joints["shoulder"]) / torso)
    assert 0.65 < min(rel) and max(rel) < 1.0 and max(rel) / min(rel) < 1.4, rel
    assert max(hybrid) / min(hybrid) > 3.0, hybrid                # what a face id alone leaves


@assets
def test_the_membrane_root_is_exactly_where_it_always_was_on_the_neutral_body(world):
    base, _, v0 = world
    for style in ("membrane", "dragon", "feathered"):
        new = K.instantiate_all([K.new_spec("wing", params={"style": style}, base=base)],
                                base, v0)
        # the registry wing has the shoulder anatomy's muscular upper arm (wings.Sleeve, S3); with the same
        # sleeve the legacy builder produces the registry's wing bit for bit: the ROOT is where it always was
        old = A.back_wings(style, sleeve=K.WING_SLEEVES[style]).build(base, v0)
        for p, (_, mesh) in zip(new, old):
            assert np.array_equal(p.mesh().vertices, mesh.vertices), style
            assert np.array_equal(p.mesh().faces, mesh.faces), style


@assets
def test_a_single_right_wing_is_the_exact_mirror_of_the_pairs_right_wing(world):
    base, _, v0 = world
    for style in ("membrane", "feathered"):
        for v in (v0, _body(world, height=1.0)):
            pair = K.instantiate_all([K.new_spec("wing", params={"style": style})], base, v)
            right = K.instantiate_all(
                [K.new_spec("wing", side="right", params={"style": style})], base, v)
            left = K.instantiate_all(
                [K.new_spec("wing", side="left", params={"style": style})], base, v)
            assert [p.name for p in right] == ["wing-1.r"] and [p.name for p in left] == ["wing-1.l"]
            assert np.array_equal(right[0].mesh().vertices, pair[1].mesh().vertices)
            assert np.array_equal(right[0].mesh().faces, pair[1].mesh().faces)
            assert np.array_equal(left[0].mesh().vertices, pair[0].mesh().vertices)
    # and the pair really is a mirror image
    pair = K.instantiate_all([K.new_spec("wing")], base, v0)
    mirrored = pair[0].mesh().vertices * np.array([-1.0, 1.0, 1.0])
    assert np.allclose(mirrored, pair[1].mesh().vertices, atol=1e-9)


@assets
def test_two_wing_specs_coexist(world):
    """Bug 4: two wing specs used to collide on shell names (wing_l twice) and on joint
    prefixes ('joint wing_l:root already exists')."""
    base, _, v0 = world
    a = K.new_spec("wing", params={"style": "membrane"})
    b = K.new_spec("wing", [a], side="right", params={"style": "membrane", "span": 9.0},
                   controls={"fold": 1.0})
    placed = K.instantiate_all([a, b], base, v0)
    names = [p.name for p in placed]
    assert names == ["wing-1.l", "wing-1.r", "wing-2.r"] and len(set(names)) == 3
    body = Armature()
    body.add_joint("chest", (0.0, 5.0, 0.0))
    for p in placed:
        p.rig.attach(body, "chest", prefix=p.joint_prefix)
    assert "wing-1.l:root" in body.joints and "wing-2.r:root" in body.joints
    # their own shapes and poses
    assert placed[2].mesh().bounds[1][0] - placed[2].mesh().bounds[0][0] < \
        placed[1].mesh().bounds[1][0] - placed[1].mesh().bounds[0][0]


@assets
def test_controls_are_per_side(world):
    base, _, v0 = world
    spec = K.new_spec("wing", controls={"l": {"fold": 1.0}})
    left, right = K.instantiate_all([spec], base, v0)
    span = lambda m: m.bounds[1][0] - m.bounds[0][0]            # noqa: E731
    assert span(left.mesh()) < 0.5 * span(right.mesh())         # the left one is stowed
    assert left.controls == {"fold": 1.0} and right.controls == {}
    assert np.array_equal(right.mesh().vertices,
                          K.instantiate_all([K.new_spec("wing")], base, v0)[1].mesh().vertices)
    both = K.new_spec("wing", controls={"*": {"preset": "folded"}, "r": {"preset": "glide"}})
    l2, r2 = K.instantiate_all([both], base, v0)
    assert span(l2.mesh()) < 0.5 * span(r2.mesh())
    assert not np.array_equal(left.mesh().vertices, left.mesh({"fold": 0.0}).vertices)
    # a placed wing allows less than a free one, and less as it folds
    free = K.KINDS["wing"].control_range("raise_", {"fold": 0.0})
    placed_range = right.control_range("raise_")
    assert placed_range[0] > free[0] and placed_range[1] < free[1]
    assert left.control_range("swing")[1] < right.control_range("swing")[1]
    assert right.control_range("nonsense") is None


@assets
def test_building_does_not_touch_the_body_and_is_deterministic(world):
    base, _, v0 = world
    keep = v0.copy()
    specs = [K.new_spec("wing"), K.new_spec("tail"), K.new_spec("horn")]
    a = K.instantiate_all(specs, base, v0)
    b = K.instantiate_all(specs, base, v0)
    assert np.array_equal(v0, keep)
    for p, q in zip(a, b):
        assert np.array_equal(p.mesh().vertices, q.mesh().vertices)


@assets
def test_draft_values_make_a_cheaper_feathered_wing(world):
    base, _, v0 = world
    spec = K.new_spec("wing", side="left", params={"style": "feathered"})
    full = K.instantiate_all([spec], base, v0)[0].mesh()
    draft = K.instantiate_all([spec], base, v0, draft=True)[0].mesh()
    assert len(draft.faces) < 0.6 * len(full.faces)
    assert K.KINDS["wing"].draft_params({"style": "membrane"}) == {}
    assert spec.params == {"style": "feathered"}                # a draft never alters the record


# -- tails and horns -----------------------------------------------------------------------------

def _parity_inside(pts, tris, chunk=512):
    """Even-odd ray test along +X; triangles cropped to the points' (y, z) box."""
    out = np.zeros(len(pts), dtype=bool)
    if not len(pts):
        return out
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


def _embedded(appendage_mesh, body_mesh, n=60000, seed=0):
    """Share of the appendage's VOLUME inside the body (Monte Carlo, fixed seed): the
    printing reviewer's method."""
    lo, hi = appendage_mesh.bounds
    pts = lo + np.random.default_rng(seed).random((n, 3)) * (hi - lo)
    pts = pts[_parity_inside(pts, appendage_mesh.vertices[appendage_mesh.faces])]
    assert len(pts) > 1500
    return float(_parity_inside(pts, body_mesh.vertices[body_mesh.faces]).mean())


@assets
def test_tails_and_horns_are_sunk_into_the_body(world):
    """Bug 7: at offset 0 a tail is tangent to the skin (3.8% of its volume inside on the
    reviewer's body, 0.5% here) and a horn touches it at a point -- a hairline weld. The
    presets sink them: horn 58-70% of the volume embedded; the tail's sink is its hidden bone
    ROOT now (``tailrig.STEM_R`` = 1.75 base radii along its own axis), so the default tail is
    36% embedded and ``offset`` slides it along that axis (+0.9 units: only touching)."""
    base, _, v0 = world
    body = quads_to_mesh(*base.body_cage(v0))

    def embedded(kind, params=None, n=60000, **placement):
        spec = K.new_spec(kind, params=params)
        spec.placement.update(placement)
        return [_embedded(p.mesh(), body, n=n) for p in K.instantiate_all([spec], base, v0)]

    # the default tail has a rounded tip and ends in the same stem: 36% embedded; the windows reject the
    # neighbouring slides (+0.3: 24%, +0.6: 11%, +0.9: tangent)
    tail, = embedded("tail")
    assert 0.33 <= tail <= 0.39, tail
    assert 0.20 <= embedded("tail", offset=0.3)[0] <= 0.27
    assert 0.08 <= embedded("tail", offset=0.6)[0] <= 0.14
    assert embedded("tail", offset=0.9)[0] < 0.02
    # (and deeper: the sink grows)
    assert embedded("tail", offset=-0.3)[0] > tail
    # The registry horn is the anatomical one (hornrig.py): the first version's horn was a 0.42 radius
    # cone sunk 0.55 along the surface normal (65% of its volume inside; at offset 0 it only touched
    # the skin). The new one is 0.17 across (the head's size sets it), 1.6 long, and its root is the
    # sheath's foot, which follows the skin it comes out of down to a rounded bowl: 15.7% of its
    # volume is inside the (unmoulded) body at the default, 18.5% when it is sunk 0.1 deeper along its
    # axis (the junction's moulded skin encloses more: the printability report's weld is 32%). A horn can
    # no longer only touch the skin: its foot always reaches it, so a slid-out horn stands on a taller foot
    # (a pedicle) and is as buried as before (13.5% at +0.1).
    horns = embedded("horn", n=30000)
    assert all(0.11 <= h <= 0.20 for h in horns), horns
    deeper = embedded("horn", n=30000, offset=-0.1)
    assert all(d > h + 0.01 for d, h in zip(deeper, horns)), (deeper, horns)


@assets
def test_tail_and_horn_presets_equal_the_legacy_placement_plus_the_sink(world):
    base, _, v0 = world
    # The HORN preset is no longer the legacy engine's placement plus the sink (that premise, one
    # quad on the parietal eminence behind the ear plane and a 0.42 cone sunk 0.55 along its normal, is
    # what made the horns come out of the back of the skull as pipes: hornrig.py, test_horn_anatomy.py).
    # What stays: the legacy engine still builds a horn on the old preset's point, the preset (and
    # the quad) are still known, and a record that names them is read as the shape's own site on the
    # frontal bone.
    horns = K.instantiate_all([K.new_spec("horn", params={"tip_radius": 0.0, "ring_verts": 8,
                                                          "flare": 0.0})], base, v0)
    olds = A.Appendage("horn", anchor_point=K.PRESET_POINTS["temples"], mirror=True,
                       offset=K.HORN_SINK).build(base, v0)
    assert [p.name for p in horns] == ["horn-1.l", "horn-1.r"] and len(olds) == 2
    assert K.PRESET_FACES["temples"] == K.LEGACY_HORN_FACE != K.PRESET_FACES["horn_crown"]
    cage = base.body_cage(v0)
    for p in horns:                                 # the new horns are on the frontal bone, in front of the bregma
        assert abs(p.anchor[0]) == pytest.approx(0.473, abs=0.01) and p.anchor[2] > 0.9
        assert K.nearest_face(cage, np.abs(p.anchor)) == K.PRESET_FACES["horn_crown"]
    # The TAIL preset is not: its premise (one quad on the side of a buttock, sunk 0.6 along that quad's
    # normal) is what made the tail come out of the side of a buttock (tailrig.py, test_tail_anchor.py). What
    # stays: the legacy engine still builds a tail on the preset's own point, and that point still snaps
    # to the preset's quad (the registry binds that quad AND its mirror image: the midline).
    tail = K.instantiate_all([K.new_spec("tail")], base, v0)[0]
    old = A.Appendage("tail", anchor_point=K.PRESET_POINTS["tail_base"], offset=K.TAIL_SINK)
    frame = old._anchor_frame(*base.body_cage(v0), 0, np.asarray(old.anchor_point), base)
    cent = base.body_cage(v0)[0][base.body_cage(v0)[1][K.PRESET_FACES["tail_base"]]].mean(axis=0)
    assert np.allclose(frame[:3, 3], cent, atol=1e-12)
    assert np.linalg.norm(tail.anchor - cent) < 0.1 and tail.anchor[0] == 0.0


@assets
def test_tails_and_horns_follow_the_body(world):
    """Bug 1 for the loft kinds: on a tall body the world-point tail snapped to y = 1.35
    while the pelvis marker had moved to y = 5.18."""
    base, _, v0 = world
    tall = _body(world, height=1.0)
    spec = K.new_spec("tail")
    for v in (v0, tall):
        p = K.instantiate_all([spec], base, v)[0]
        pelvis = A.joint_position(base, "pelvis", v)
        assert np.linalg.norm(p.anchor - pelvis) < 3.0
    legacy = A.Appendage("tail", anchor_point=K.PRESET_POINTS["tail_base"])
    root = lambda m: m.vertices[np.argmin(np.linalg.norm(                      # noqa: E731
        m.vertices - A.joint_position(base, "pelvis", tall), axis=1))]
    assert np.linalg.norm(root(legacy.build(base, tall)[0][1])
                          - A.joint_position(base, "pelvis", tall)) > 2.0


# -- extra limbs ------------------------------------------------------------------------------------

@assets
@pytest.mark.parametrize("limb", ["arm", "leg"])
def test_a_harvested_limb_is_placed_on_its_quad_and_sunk_along_its_axis(world, limb):
    base, _, v0 = world
    chain = K.LIMB_CHAINS[limb]
    # the harvested pose (controls={}: no preset pose) and the limb as cut (neck 1: untapered)
    spec = K.new_spec("harvest", params={"limb": limb, "neck": 1.0}, side="left", base=base,
                      controls={})

    def build(v, offset):
        spec.placement["offset"] = offset
        return K.instantiate_all([spec], base, v)[0]

    flush, sunk = build(v0, 0.0), build(v0, -0.5)
    axis = A.joint_position(base, chain[-1], v0) - A.joint_position(base, chain[0], v0)
    axis /= np.linalg.norm(axis)
    # the sink is exactly |offset| along the limb axis, and moves nothing else
    shift = sunk.mesh().vertices - flush.mesh().vertices
    assert np.allclose(shift, -0.5 * axis, atol=1e-12)
    assert flush.mesh().integrity_report()["watertight"]
    assert flush.mesh().integrity_report()["outward"]
    # the copy is the character's own limb moved so its shoulder joint sits on the quad it
    # is bound to (legacy: on a world point that does not move with the body)
    d = K.KINDS["harvest"].defaults(limb=limb)
    raw = A.harvest_region(base, list(chain), v0, radius=d["radius"], trim=d["trim"])
    src = A.joint_position(base, chain[0], v0)
    assert np.allclose(flush.mesh().vertices - raw.vertices, flush.anchor - src, atol=1e-9)


@assets
def test_a_harvested_arm_follows_the_body(world):
    base, _, v0 = world
    tall = _body(world, height=1.0)
    spec = K.new_spec("harvest", side="left", base=base)
    for v in (v0, tall):
        p = K.instantiate_all([spec], base, v)[0]
        assert np.linalg.norm(p.mesh().vertices - p.anchor, axis=1).min() < 0.8
    old = A.Appendage("harvest", anchor_point=K.PRESET_POINTS["shoulder_socket"],
                      chain=K.LIMB_CHAINS["arm"], params=dict(radius=1.15, trim=0.6))
    m = old.build(base, tall)[0][1]
    face_origin = K.instantiate_all([spec], base, tall)[0].anchor
    assert np.linalg.norm(m.vertices - face_origin, axis=1).min() > 3.0     # legacy: left behind


@assets
def test_a_harvested_limb_scales_about_its_shoulder_and_has_no_roll_or_up_hint(world):
    """scale is honoured (about the shoulder joint, which the sink and the pose pivot on);
    roll and up_hint have no meaning for a limb (the twist control is its roll, the pose is
    its direction) and stay dead, and are not offered."""
    base, _, v0 = world
    plain = K.new_spec("harvest", side="left", controls={})
    odd = K.new_spec("harvest", side="left", controls={})
    odd.placement.update(roll=77.0, up_hint=[0.3, 0.3, 0.9])
    big = K.new_spec("harvest", side="left", controls={})
    big.placement["scale"] = 1.5
    a, o, b = (K.instantiate_all([sp], base, v0)[0] for sp in (plain, odd, big))
    assert np.array_equal(a.mesh().vertices, o.mesh().vertices)         # dead fields stay dead
    pivot = a.limb.joints_rest["shoulder"]
    surface = a.anchor                                                   # where the quad is
    pivot_b = b.limb.joints_rest["shoulder"]
    # the sink scales with the limb (it used not to: the weld was lost beyond scale ~1.5)
    assert np.allclose(pivot_b - surface, 1.5 * (pivot - surface), atol=1e-9)
    assert np.allclose(b.mesh().vertices - pivot_b, 1.5 * (a.mesh().vertices - pivot), atol=1e-9)
    assert b.mesh().volume() == pytest.approx(1.5 ** 3 * a.mesh().volume(), rel=1e-9)
    assert K.KINDS["harvest"].placement_keys == ("anchor", "scale", "offset")


# -- every parameter at its limits ----------------------------------------------------------------

def _extreme_cases():
    for kind in K.KINDS.values():
        selectors = {"wing": ("style", ["membrane", "dragon", "feathered"]),
                     "harvest": ("limb", ["arm", "leg"])}.get(kind.id, (None, [None]))
        for choice in selectors[1]:
            given = {selectors[0]: choice} if choice else {}
            values = kind.defaults(**given)
            for p in kind.params:
                if p.type in ("float", "int") and p.applies(values):
                    for which in ("lo", "hi"):
                        yield pytest.param(kind.id, given, p.key, getattr(p, which),
                                           id=f"{kind.id}-{choice or 'x'}-{p.key}-{which}")


@assets
@pytest.mark.parametrize("kind_id,given,key,value", list(_extreme_cases()))
def test_every_parameter_at_its_limits_builds_a_watertight_solid(world, kind_id, given, key,
                                                                 value):
    base, _, v0 = world
    spec = K.new_spec(kind_id, params={**given, key: value}, side="left")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        placed = K.instantiate_all([spec], base, v0)
    assert len(placed) == 1
    mesh = placed[0].mesh()
    rep = mesh.integrity_report()
    assert rep["watertight"] and rep["outward"] and rep["volume"] > 0, rep
    assert np.isfinite(mesh.vertices).all()


@assets
def test_every_kind_builds_on_every_body(world):
    base, lib, _ = world
    specs = [K.new_spec("wing", params={"style": "membrane"}), K.new_spec("tail"),
             K.new_spec("horn"), K.new_spec("harvest"),
             K.new_spec("harvest", [], params={"limb": "leg"}, spec_id="leg-1")]
    for name, kw in BODIES.items():
        v = lib.apply(base.verts, mh.macro_weights(**kw))
        for p in K.instantiate_all(specs, base, v):
            rep = p.mesh().integrity_report()
            assert rep["watertight"] and rep["outward"], (name, p.name, rep)
            assert all(math.isfinite(x) for x in p.anchor), (name, p.name)


@pytest.mark.parametrize("bad", [-1, 999999, 1.9, True])
def test_a_face_that_is_not_a_cage_quad_is_an_error_not_a_traceback(bad):
    with pytest.raises(ValueError, match="not a cage quad"):
        K.new_spec("tail", face=bad)


@assets
def test_a_tail_control_outside_its_range_is_clamped_on_the_document_path(world):
    """lift=1e9 / curl=5 used to build a zigzag that overlaps itself (still watertight); the
    panel clamped, the document and CLI path did not."""
    base, _, v0 = world
    wild = K.new_spec("tail", controls={"lift": 1e9, "curl": 5.0, "sway": -400.0})
    edge = K.new_spec("tail", controls={"lift": 90.0, "curl": 2.0 - K.KINDS["tail"].param("curl").default,
                                        "sway": -90.0})
    a = K.instantiate_all([wild], base, v0)[0].mesh()
    b = K.instantiate_all([edge], base, v0)[0].mesh()
    assert np.allclose(a.vertices, b.vertices)
