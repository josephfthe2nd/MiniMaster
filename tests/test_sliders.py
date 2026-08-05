"""The derived slider catalog, the .mmchar document, and randomization.

Most of this needs no assets: build_catalog() takes a list of names, so the
grammar is tested against synthetic names as well as the real library.
"""

import numpy as np
import pytest

from minimaster.character import mhbase as mh
from minimaster.character.sliders import (Character, SliderCatalog,
                                          build_catalog, randomize)


# -- the grammar, on synthetic names ---------------------------------------

def test_antonym_pair_becomes_one_bipolar_slider():
    cat = SliderCatalog(["nose/nose-curve-concave", "nose/nose-curve-convex"])
    assert len(cat) == 1
    s = cat.sliders[0]
    assert s.kind == "bipolar" and s.lo == -1.0 and s.hi == 1.0
    assert s.weights(1.0) == {"nose/nose-curve-convex": 1.0}
    assert s.weights(-0.5) == {"nose/nose-curve-concave": 0.5}
    assert s.weights(0.0) == {}


def test_left_right_fold_into_one_sided_slider():
    cat = SliderCatalog(["cheek/l-cheek-bones-incr", "cheek/r-cheek-bones-incr",
                         "cheek/l-cheek-bones-decr", "cheek/r-cheek-bones-decr"])
    assert len(cat) == 1
    s = cat.sliders[0]
    assert s.sided
    w = s.weights(1.0)
    assert w == {"cheek/l-cheek-bones-incr": 1.0, "cheek/r-cheek-bones-incr": 1.0}


def test_asymmetry_biases_the_two_sides():
    cat = SliderCatalog(["cheek/l-cheek-bones-incr", "cheek/r-cheek-bones-incr",
                         "cheek/l-cheek-bones-decr", "cheek/r-cheek-bones-decr"])
    s = cat.sliders[0]
    w = s.weights(1.0, asymmetry=0.5)
    assert w["cheek/l-cheek-bones-incr"] > w["cheek/r-cheek-bones-incr"]
    # and it never exceeds full strength
    assert max(w.values()) <= 1.0
    assert s.weights(1.0, asymmetry=0.0) == s.weights(1.0)


def test_exclusive_set_becomes_a_choice_covering_both_sides():
    names = [f"ears/{side}-ear-shape-{shape}"
             for side in ("l", "r")
             for shape in ("pointed", "round", "square", "triangle")]
    cat = SliderCatalog(names)
    s = cat.sliders[0]
    assert s.kind == "choice" and len(s.options) == 4
    w = s.weights(0)
    assert len(w) == 2                      # both ears, not just the left
    assert set(w.values()) == {1.0}
    assert set(s.targets()) == set(names)   # nothing dropped


def test_lone_target_becomes_unipolar():
    cat = SliderCatalog(["stomach/stomach-pregnant"])
    s = cat.sliders[0]
    assert s.kind == "unipolar" and s.lo == 0.0
    assert s.weights(-1.0) == {}            # clamped, never negative
    assert s.weights(0.75) == {"stomach/stomach-pregnant": 0.75}


def test_translation_group_splits_per_axis():
    names = [f"armslegs/l-foot-trans-{t}"
             for t in ("in", "out", "up", "down", "forward", "backward")]
    cat = SliderCatalog(names)
    assert len(cat) == 3                    # one slider per axis
    assert all(s.kind == "bipolar" for s in cat.sliders)
    covered = {t for s in cat.sliders for t in s.targets()}
    assert covered == set(names)


def test_expressions_are_excluded_from_the_shape_catalog():
    cat = SliderCatalog(["expression/units/african/eye-left-closure",
                         "nose/nose-curve-convex", "nose/nose-curve-concave"])
    assert len(cat) == 1
    assert all("expression" not in s.key for s in cat.sliders)


def test_macro_targets_are_excluded():
    cat = SliderCatalog(["macrodetails/caucasian-male-young",
                         "nose/nose-curve-convex", "nose/nose-curve-concave"])
    assert len(cat) == 1


def test_catalog_lookup_and_search():
    cat = SliderCatalog(["nose/nose-curve-convex", "nose/nose-curve-concave"])
    key = cat.sliders[0].key
    assert key in cat
    assert cat[key] is cat.sliders[0]
    assert cat.get("nope") is None
    assert cat.find("curve")
    assert cat.find("zzz") == []


def test_catalog_weights_merges_and_ignores_unknown():
    cat = SliderCatalog(["nose/nose-curve-convex", "nose/nose-curve-concave"])
    key = cat.sliders[0].key
    w = cat.weights({key: 0.5, "not.a.slider": 1.0})
    assert w == {"nose/nose-curve-convex": 0.5}


# -- the character document -------------------------------------------------

def test_character_roundtrip(tmp_path):
    c = Character(name="hero")
    c.macro["gender"] = 1.0
    c.sliders["nose.nose_curve"] = 0.8
    c.asymmetry["nose.nose_curve"] = 0.2
    p = tmp_path / "hero.mmchar"
    c.save(p)
    back = Character.load(p)
    assert back.name == "hero"
    assert back.macro["gender"] == 1.0
    assert back.sliders["nose.nose_curve"] == 0.8
    assert back.asymmetry["nose.nose_curve"] == 0.2


def test_character_tolerates_unknown_sliders():
    """A character saved against a different asset set must still load."""
    cat = SliderCatalog(["nose/nose-curve-convex", "nose/nose-curve-concave"])
    c = Character()
    c.sliders["some.removed.slider"] = 1.0
    w = c.target_weights(cat)   # must not raise
    assert all(not k.startswith("some.") for k in w)


def test_randomize_is_seeded_bounded_and_face_only():
    cat = SliderCatalog(["nose/nose-curve-convex", "nose/nose-curve-concave",
                         "hip/hip-scale-horiz-incr", "hip/hip-scale-horiz-decr"])
    a = randomize(cat, seed=3)
    b = randomize(cat, seed=3)
    assert a == b
    assert randomize(cat, seed=4) != a or not a
    for key, val in a.items():
        s = cat[key]
        assert s.group == "Face"
        assert s.lo <= val <= s.hi


# -- against the real 1,280-target library ---------------------------------

@pytest.fixture(scope="module")
def lib():
    return mh.TargetLibrary()


@pytest.fixture(scope="module")
def cat(lib):
    return SliderCatalog(lib.names)


@pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
class TestRealLibrary:
    def test_every_shape_target_is_reachable(self, cat, lib):
        hit, total = cat.coverage(lib.names)
        assert hit == total, f"{total - hit} targets unreachable"
        assert total > 800

    def test_catalog_size_and_groups(self, cat):
        assert 300 < len(cat) < 500
        groups = cat.groups()
        assert {"Face", "Body", "Asymmetry"} <= set(groups)
        assert {"Nose", "Eyes", "Mouth", "Cheeks", "Ears"} <= set(groups["Face"])

    def test_every_slider_references_real_targets(self, cat, lib):
        for s in cat.sliders:
            for t in s.targets():
                assert t in lib, f"{s.key} -> missing {t}"

    def test_the_named_controls_exist(self, cat):
        for want in ("nose_curve", "cheek_bones", "chin_prognathism"):
            assert cat.find(want), want

    def test_sliders_actually_move_the_mesh(self, cat, lib):
        base = mh.BaseMesh()
        import random
        for s in random.Random(11).sample(cat.sliders, 25):
            w = s.weights(s.hi)
            assert w, s.key
            v = lib.apply(base.verts, w)
            assert not np.allclose(v, base.verts), s.key

    def test_symmetric_slider_moves_both_sides(self, cat, lib):
        base = mh.BaseMesh()
        s = cat["cheek.cheek_bones"]
        v = lib.apply(base.verts, s.weights(1.0))
        moved = base.verts[np.linalg.norm(v - base.verts, axis=1) > 1e-6]
        assert moved[:, 0].min() < 0 < moved[:, 0].max()

    def test_character_stays_printable(self, cat, lib):
        from minimaster.character import printprep as pp
        base = mh.BaseMesh()
        c = Character()
        c.macro["gender"] = 1.0
        c.sliders.update(randomize(cat, seed=5, amount=0.6))
        v = lib.apply(base.verts, c.target_weights(cat))
        _, rep = pp.assemble_character(
            pp.character_print_shells(base, v), size="medium")
        assert rep.watertight
