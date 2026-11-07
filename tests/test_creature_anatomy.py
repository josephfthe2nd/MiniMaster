"""Appendages reshaping the body that carries them.

The point of this module is that a wing is not a sticker: adding one has to
move the host's chest, back and shoulders, because in real flying animals the
limb and the trunk that powers it are one system. These tests check that the
mapping is anatomically directed (chest up, waist down), that it scales with
the appendage rather than saturating, and that the reshaped body still passes
the print gate.
"""

import numpy as np
import pytest

from minimaster.character import appendages as A
from minimaster.character import creature_anatomy as ca
from minimaster.character import mhbase as mh
from minimaster.character.sliders import Character, SliderCatalog
from minimaster.core.mesh import Mesh


# -- the load model (no assets needed) -------------------------------------

def test_a_pair_of_wings_is_the_typical_installation():
    """Coefficients are written for the normal configuration of each kind,
    so a wing PAIR scores 1.0 and a lone wing scores less."""
    pair = A.Appendage("wing", mirror=True)
    single = A.Appendage("wing", mirror=False)
    assert ca.appendage_load(pair) == pytest.approx(1.0)
    assert ca.appendage_load(single) < ca.appendage_load(pair)


def test_a_second_tail_costs_more_than_one():
    """Tails are normally solitary, so mirroring one is the unusual case."""
    assert ca.appendage_load(A.Appendage("tail")) == pytest.approx(1.0)
    assert ca.appendage_load(A.Appendage("tail", mirror=True)) > 1.0


def test_load_tracks_appendage_size():
    """Load is measured against the builder's own default span, so the two
    cannot drift apart when the wing is retuned."""
    from minimaster.character.wings import DEFAULT_SPAN
    small = A.Appendage("wing", mirror=True, params=dict(span=DEFAULT_SPAN / 2))
    big = A.Appendage("wing", mirror=True, params=dict(span=DEFAULT_SPAN * 2))
    assert ca.appendage_load(small) == pytest.approx(0.5)
    assert ca.appendage_load(big) == pytest.approx(2.0)


def test_scale_multiplies_the_load():
    a = A.Appendage("horn", scale=1.0)
    b = A.Appendage("horn", scale=3.0)
    assert ca.appendage_load(b) == pytest.approx(3 * ca.appendage_load(a))


def test_bigger_wings_demand_more_than_smaller_ones():
    """The whole point of the PAIRED_KINDS correction: a default pair must not
    already sit at the cap, or every wing above default looks identical."""
    small, _, _ = ca.consequences(
        [A.Appendage("wing", mirror=True, params=dict(span=6.0))])
    big, _, _ = ca.consequences(
        [A.Appendage("wing", mirror=True, params=dict(span=14.0))])
    key = "torso.torso_muscle_pectoral"
    assert small[key] < big[key] <= 1.0


def test_wings_build_the_chest_and_trim_the_abdomen():
    s, macro, _ = ca.consequences([A.Appendage("wing", mirror=True)])
    assert s["torso.torso_muscle_pectoral"] > 0
    assert s["torso.torso_muscle_dorsi"] > 0
    assert s["measure.measure_bust_circ"] > 0
    assert s["measure.measure_shoulder_dist"] > 0
    # flyers are light: the abdomen and legs pay for the chest
    assert s["measure.measure_waist_circ"] < 0
    assert s["measure.measure_thigh_circ"] < 0
    assert macro["weight"] < 0


def test_tails_broaden_the_pelvis_not_the_chest():
    s, _, _ = ca.consequences([A.Appendage("tail")])
    assert s["hip.hip_scale_horiz"] > 0
    assert s["buttocks.buttocks_volume"] > 0
    assert "torso.torso_muscle_pectoral" not in s


def test_horns_load_the_neck():
    s, _, _ = ca.consequences([A.Appendage("horn", mirror=True)])
    assert s["neck.neck_scale_horiz"] > 0
    assert s["forehead.forehead_nubian"] > 0


def test_stacked_appendages_accumulate_but_stay_capped():
    apps = [A.Appendage("wing", mirror=True),
            A.Appendage("harvest", mirror=True)]
    s, _, notes = ca.consequences(apps)
    assert len(notes) == 2
    # both contribute to the shoulders
    solo, _, _ = ca.consequences([A.Appendage("wing", mirror=True)])
    assert s["measure.measure_shoulder_dist"] > solo["measure.measure_shoulder_dist"]
    assert all(abs(v) <= 1.0 for v in s.values())


def test_cap_is_respected():
    s, _, _ = ca.consequences(
        [A.Appendage("wing", mirror=True, scale=10.0)], cap=0.4)
    assert all(abs(v) <= 0.4 for v in s.values())


def test_unknown_kinds_are_ignored_not_fatal():
    s, m, notes = ca.consequences([A.Appendage("gizzard")])
    assert s == {} and m == {} and notes == []


def test_zero_sized_appendage_costs_nothing():
    s, _, _ = ca.consequences([A.Appendage("wing", mirror=True, scale=0.0)])
    assert s == {}


# -- folding into a Character ----------------------------------------------

def test_apply_adds_to_user_values_instead_of_replacing_them():
    c = Character()
    c.sliders["torso.torso_muscle_pectoral"] = 0.2
    ca.apply_to(c, [A.Appendage("wing", mirror=True, scale=0.4)])
    assert c.sliders["torso.torso_muscle_pectoral"] == pytest.approx(
        0.2 + 0.85 * 0.4)


def test_apply_never_pushes_a_slider_out_of_range():
    c = Character()
    c.sliders["torso.torso_muscle_pectoral"] = 0.9
    ca.apply_to(c, [A.Appendage("wing", mirror=True, scale=4.0)])
    assert c.sliders["torso.torso_muscle_pectoral"] == 1.0


def test_apply_keeps_macro_in_the_unit_range():
    c = Character()
    c.macro["weight"] = 0.02
    ca.apply_to(c, [A.Appendage("wing", mirror=True, scale=5.0)])
    assert c.macro["weight"] == 0.0


def test_a_catalog_filters_out_sliders_the_asset_set_lacks():
    c = Character()
    tiny = SliderCatalog(["nose/nose-curve-convex", "nose/nose-curve-concave"])
    notes = ca.apply_to(c, [A.Appendage("wing", mirror=True)], catalog=tiny)
    assert c.sliders == {}          # nothing in this catalog matched
    assert notes                    # but the reasoning is still reported


# -- the flight arithmetic --------------------------------------------------

def test_a_human_sized_flyer_is_told_the_truth():
    # 65 litres ~ 66 kg, 2 m span
    rep = ca.flight_report(65.0, 20.0)
    assert rep.body_mass_kg == pytest.approx(65.0 * 1.01, rel=1e-6)
    assert rep.wingspan_m == pytest.approx(2.0)
    assert not rep.can_power_fly and not rep.can_soar
    assert "fantasy" in rep.verdict
    assert any("soaring" in n for n in rep.notes)


def test_a_bat_sized_body_can_actually_fly():
    rep = ca.flight_report(1.0, 4.0)      # ~1 kg, 0.4 m span
    assert rep.can_power_fly and rep.can_soar
    assert "plausible" in rep.verdict


def test_soaring_only_band():
    rep = ca.flight_report(20.0, 40.0)    # ~20 kg, 4 m span
    assert rep.can_soar and not rep.can_power_fly
    assert "soaring" in rep.verdict


def test_required_muscle_is_the_avian_fraction():
    rep = ca.flight_report(65.0, 20.0)
    assert rep.required_muscle_kg == pytest.approx(
        rep.body_mass_kg * ca.BIRD_PECTORALIS_FRACTION)


def test_wing_loading_flags_the_impossible_case():
    human = ca.flight_report(65.0, 20.0)
    bird = ca.flight_report(0.5, 12.0)
    assert human.wing_loading_kg_m2 > 20      # far outside the flying band
    assert bird.wing_loading_kg_m2 < 20


def test_report_text_mentions_the_numbers():
    text = ca.flight_report(65.0, 20.0).as_text()
    for want in ("body mass", "wingspan", "flight muscle", "wing loading",
                 "verdict"):
        assert want in text


def test_wingspan_is_measured_from_geometry_not_the_parameter():
    """Wings are swept and drooped, so the built span is not the span param."""
    wing = A.make_wing(span=9.0)
    measured = ca.wingspan_of([wing])
    assert measured > 0
    assert ca.wingspan_of([]) == 0.0
    assert ca.wingspan_of([A.make_wing(span=18.0)]) > measured


# -- against the real base mesh --------------------------------------------

@pytest.fixture(scope="module")
def assets():
    base, lib = mh.load()
    return base, lib, SliderCatalog(lib.names)


@pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
class TestOnRealBody:
    def test_every_referenced_slider_exists(self, assets):
        _, _, cat = assets
        missing = sorted({k for eff in ca.CONSEQUENCES.values()
                          for k in eff.sliders if k not in cat})
        assert not missing, f"anatomy maps to sliders that do not exist: {missing}"

    def test_every_referenced_slider_is_bipolar(self, assets):
        """Several coefficients are negative (a flyer's waist narrows), which
        is only meaningful on a slider that goes below zero."""
        _, _, cat = assets
        for eff in ca.CONSEQUENCES.values():
            for key, val in eff.sliders.items():
                if val < 0:
                    assert cat[key].lo < 0, key

    def test_every_referenced_macro_exists(self):
        macro = set(Character().macro)
        for eff in ca.CONSEQUENCES.values():
            assert set(eff.macro) <= macro

    def _cage(self, assets, character):
        base, lib, cat = assets
        v = lib.apply(base.verts, character.target_weights(cat))
        return base.body_cage(v)

    def _volume(self, assets, character):
        cv, cq = self._cage(assets, character)
        tris = np.vstack([cq[:, [0, 1, 2]], cq[:, [0, 2, 3]]])
        return Mesh(cv, tris).volume()

    def test_wings_actually_deepen_the_chest(self, assets):
        """The visible payoff: the upper trunk gains depth, the belly does not."""
        _, _, cat = assets
        plain = Character()
        plain.macro["gender"] = 1.0
        winged = Character()
        winged.macro["gender"] = 1.0
        ca.apply_to(winged, [A.Appendage("wing", mirror=True,
                                         params=dict(span=14.0))], cat)
        a = self._cage(assets, plain)[0]
        b = self._cage(assets, winged)[0]

        def depth(v, lo, hi):
            band = v[(v[:, 1] > lo) & (v[:, 1] < hi)]
            return float(np.ptp(band[:, 2]))

        assert depth(b, 3.5, 5.0) > depth(a, 3.5, 5.0) * 1.05   # chest out
        assert depth(b, 1.0, 2.0) <= depth(a, 1.0, 2.0)         # belly in

    def test_a_winged_body_is_heavier_with_muscle(self, assets):
        plain = Character()
        plain.macro["gender"] = 1.0
        winged = Character()
        winged.macro["gender"] = 1.0
        ca.apply_to(winged, [A.Appendage("wing", mirror=True)])
        assert self._volume(assets, winged) > self._volume(assets, plain)

    def test_the_reshaped_body_still_prints(self, assets):
        from minimaster.character import printprep as pp
        base, lib, cat = assets
        c = Character(name="seraph")
        c.macro["gender"] = 1.0
        c.macro["muscle"] = 0.7
        apps = [A.Appendage("wing", anchor_point=(1.5, 3.0, -0.9), mirror=True,
                            params=dict(span=12.0)),
                A.Appendage("horn", anchor_point=(0.85, 8.55, 0.05),
                            mirror=True)]
        ca.apply_to(c, apps, cat)
        v = lib.apply(base.verts, c.target_weights(cat))
        from minimaster.character import print_report as pr
        body = pp.character_print_shells(base, v)
        shells = body + pp.appendage_print_shells(apps, base, v)
        _, rep = pp.assemble_character(shells, size="medium", reference=pp.FIGURE_SHELLS)
        assert rep.watertight
        assert rep.shells >= 8
        # the reshaped body carries the wings and horns: each root is on it in the STL frame
        # (the old version added Y-up appendages to Z-up shells and never looked)
        gaps = pr.root_gaps(rep.placed_shells)
        assert set(gaps) == {"wing_l", "wing_r", "horn_l", "horn_r"}
        assert max(gaps.values()) <= 0.1
        wrong = [s for a in apps for s in a.build(base, v)]
        far = pp.assemble_character(body + wrong, size="medium", reference=pp.FIGURE_SHELLS)[1]
        assert min(pr.root_gaps(far.placed_shells).values()) > 1.0

    def test_flight_report_uses_the_characters_own_mass(self, assets):
        base, lib, cat = assets
        big = Character()
        big.macro["gender"] = 1.0
        big.macro["weight"] = 1.0
        small = Character()
        small.macro["gender"] = 0.0
        small.macro["weight"] = 0.0
        heavy = ca.flight_report(self._volume(assets, big), 20.0)
        light = ca.flight_report(self._volume(assets, small), 20.0)
        assert heavy.body_mass_kg > light.body_mass_kg
        assert heavy.required_muscle_kg > light.required_muscle_kg
        # a human-scale body lands in a plausible mass range, not milligrams
        assert 30.0 < light.body_mass_kg < 150.0


# -- the derived layer: the body is never baked (design D6, bug 3) -----------------------------

import copy                                                       # noqa: E402

from minimaster.character import kinds as K                       # noqa: E402
from minimaster.character.appendage_spec import AppendageSpec     # noqa: E402

PECTORAL = "torso.torso_muscle_pectoral"


def _winged(**kw):
    c = Character(name="winged", **kw)
    c.appendages += [K.new_spec("wing"), K.new_spec("tail"), K.new_spec("horn")]
    return c


def test_apply_to_bakes_the_adaptation_and_a_second_call_adds_it_again():
    """The behaviour the derived layer replaces (kept for the older tools): pectoral
    0.85 after one apply, 1.00 after a second, with no inverse."""
    c = Character()
    apps = [A.Appendage("wing", mirror=True)]
    ca.apply_to(c, apps)
    assert c.sliders[PECTORAL] == pytest.approx(0.85)
    ca.apply_to(c, apps)
    assert c.sliders[PECTORAL] == 1.0


def test_the_effective_body_is_recomputed_not_accumulated():
    c = _winged()
    first = c.effective()
    for _ in range(3):
        again = c.effective()
    assert again == first
    assert first[0][PECTORAL] == pytest.approx(0.85)
    assert c.sliders == {} and c.macro == Character().macro        # nothing written back


def test_the_effective_values_are_what_apply_to_would_have_made():
    user = Character()
    user.sliders[PECTORAL] = 0.2
    legacy = copy.deepcopy(user)
    ca.apply_to(legacy, [A.Appendage("wing", mirror=True), A.Appendage("tail"),
                         A.Appendage("horn", mirror=True)])
    mine = copy.deepcopy(user)
    mine.appendages += _winged().appendages
    sliders, macro, notes = mine.effective()
    assert sliders == legacy.sliders and macro == legacy.macro
    assert len(notes) == 3
    assert mine.sliders == {PECTORAL: 0.2}


def test_removing_an_appendage_restores_the_body_exactly():
    c = _winged()
    plain = Character()
    assert c.effective()[0] != plain.effective()[0]
    for spec_id in ("wing-1", "tail-1", "horn-1"):
        assert c.remove_appendage(spec_id)
    assert c.effective() == (plain.sliders, plain.macro, [])
    # switching one off is the same as removing it
    d = _winged()
    for spec in d.appendages:
        spec.enabled = False
    assert d.effective() == (plain.sliders, plain.macro, [])


def test_a_switched_off_record_costs_nothing_while_the_others_still_do():
    c = _winged()
    only_wing = Character()
    only_wing.appendages.append(K.new_spec("wing"))
    c.spec("tail-1").enabled = False
    c.spec("horn-1").enabled = False
    assert c.effective() == only_wing.effective()
    apps = c.appendages
    assert ca.consequences(apps) == ca.consequences([c.spec("wing-1")])
    assert [n.split(" ")[0] for n in ca.anatomy_delta(apps)[1]] == ["wing"]


def test_the_anatomy_switch_and_cap_are_the_only_stored_state():
    c = _winged()
    c.anatomy["enabled"] = False
    assert c.effective()[0] == {} and c.effective()[2] == []
    c.anatomy.update(enabled=True, cap=0.3)
    sliders, macro, _ = c.effective()
    assert sliders and all(abs(v) <= 0.3 + 1e-12 for v in sliders.values())
    again = Character.from_dict(c.to_dict())
    assert again.anatomy == {"enabled": True, "cap": 0.3} and again.effective() == c.effective()
    assert "sliders" not in c.to_dict()["anatomy"]


def test_save_and_load_leave_the_effective_body_unchanged(tmp_path):
    c = _winged()
    c.sliders["nose.nose_curve"] = 0.4
    before = c.effective()
    c.save(tmp_path / "c.mmchar")
    after = Character.load(tmp_path / "c.mmchar")
    assert after.sliders == {"nose.nose_curve": 0.4}               # no anatomy in the saved sliders
    assert after.effective() == before
    after.save(tmp_path / "d.mmchar")
    assert Character.load(tmp_path / "d.mmchar").effective() == before


def test_a_record_the_registry_knows_sets_its_own_load():
    wing = K.new_spec("wing")
    big = K.new_spec("wing", params={"span": 26.0})
    assert ca.appendage_load(wing) == pytest.approx(1.0)
    assert ca.appendage_load(big) == pytest.approx(2.0)
    wing.side = "left"
    assert ca.appendage_load(wing) == pytest.approx(1.0 / ca.PAIR_FACTOR)   # a lone wing is lighter
    tail = K.new_spec("tail", params={"length": 12.0})
    assert ca.appendage_load(tail) == pytest.approx(2.0)
    tail.side = "both"
    assert ca.appendage_load(tail) == pytest.approx(2.0 * ca.PAIR_FACTOR)
    scaled = K.new_spec("horn", scale=3.0)
    assert ca.appendage_load(scaled) == pytest.approx(3.0)


def test_a_harvested_limb_costs_the_same_whatever_its_dead_scale_says():
    """A harvest ignores scale (the old class even multiplied the anatomy by it): the
    kind owns that decision."""
    a = K.new_spec("harvest")
    b = K.new_spec("harvest")
    b.placement["scale"] = 5.0
    assert ca.appendage_load(a) == ca.appendage_load(b) == pytest.approx(1.0)
    assert ca.appendage_load(A.Appendage("harvest", mirror=True, scale=5.0)) == pytest.approx(5.0)


def test_records_that_cannot_be_built_cost_nothing():
    unknown = AppendageSpec(id="g-1", kind="gizzard")
    bad_style = AppendageSpec(id="w-1", kind="wing", side="both", params={"style": "gryphon"})
    off = K.new_spec("wing")
    off.enabled = False
    assert ca.appendage_load(bad_style) == 0.0
    assert ca.anatomy_delta([unknown, bad_style, off]) == ({"sliders": {}, "macro": {}}, [])


def test_the_delta_is_returned_without_touching_anything():
    c = Character()
    c.sliders[PECTORAL] = 0.5
    apps = _winged().appendages
    snapshot = copy.deepcopy((c.sliders, c.macro, apps))
    delta, notes = ca.anatomy_delta(apps)
    assert delta["sliders"][PECTORAL] == pytest.approx(0.85) and delta["macro"]["weight"] < 0
    assert len(notes) == 3
    sliders, macro, _ = ca.effective_values(c.sliders, c.macro, apps)
    assert sliders[PECTORAL] == 1.0                               # clamped on the SUM
    assert (c.sliders, c.macro, apps) == snapshot


def test_kind_ids_of_the_old_tables_still_work_for_records():
    """The kind ids 'harvest' / PAIRED_KINDS keep their meaning for records: an extra limb
    pair adapts the shoulders like the legacy class always did."""
    legacy, _, _ = ca.consequences([A.Appendage("harvest", mirror=True)])
    record, _, _ = ca.consequences([K.new_spec("harvest")])
    assert record == legacy


@pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
class TestDerivedBodyOnRealMesh:
    def _verts(self, assets, c):
        base, lib, cat = assets
        return lib.apply(base.verts, c.effective_target_weights(cat))

    def test_save_load_build_leaves_the_body_unchanged(self, assets, tmp_path):
        c = _winged()
        c.macro.update(gender=1.0, muscle=0.6)
        v0 = self._verts(assets, c)
        c.save(tmp_path / "c.mmchar")
        c2 = Character.load(tmp_path / "c.mmchar")
        assert np.array_equal(self._verts(assets, c2), v0)
        c2.save(tmp_path / "d.mmchar")
        assert np.array_equal(self._verts(assets, Character.load(tmp_path / "d.mmchar")), v0)

    def test_removing_the_appendages_gives_the_plain_body_back(self, assets):
        _, _, cat = assets
        c = _winged()
        plain = Character()
        v_winged = self._verts(assets, c)
        assert not np.allclose(v_winged, self._verts(assets, plain))     # it did reshape the body
        c.appendages.clear()
        assert np.array_equal(self._verts(assets, c), self._verts(assets, plain))
        # the user's own body is unaffected all along
        assert _winged().target_weights(cat) == plain.target_weights(cat)

    def test_two_applies_are_not_additive(self, assets):
        base, lib, cat = assets
        c = _winged()
        w1 = c.effective_target_weights(cat)
        c.effective_target_weights(cat)
        assert c.effective_target_weights(cat) == w1

    def test_the_adapted_body_is_what_the_old_bake_produced(self, assets):
        base, lib, cat = assets
        derived = _winged()
        baked = Character()
        ca.apply_to(baked, [A.Appendage("wing", mirror=True), A.Appendage("tail"),
                            A.Appendage("horn", mirror=True)], cat)
        assert np.array_equal(self._verts(assets, derived),
                              lib.apply(base.verts, baked.target_weights(cat)))
