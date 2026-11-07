"""What a tail asks of the pelvis (``creature_anatomy.TailEffect``).

The flat coefficients of the first version (hip_scale_horiz +0.45, hip_scale_depth +0.35, buttocks_volume +0.55,
dorsi +0.30, hips_circ +0.35) were applied even to the default tail, whose base radius is 0.21 of the hip span:
a size the pelvis already carries. The coupling follows the tail's SIZE (length and base radius): nearly nothing for
the default tail, more for a bigger one, capped.
"""

import pytest

from minimaster.character import appendages as A
from minimaster.character import creature_anatomy as ca
from minimaster.character import kinds as K

BUT = "buttocks.buttocks_volume"


def tail_sliders(**params):
    s, _m, _n = ca.consequences([K.new_spec("tail", params=params)])
    return s


def test_the_default_tail_asks_for_almost_nothing_of_the_pelvis():
    s = tail_sliders()
    assert 0 < s[BUT] <= 0.15 + 1e-12                                # the research: <= 0.15 for s <= 1
    assert all(0 < v <= 0.15 + 1e-12 for v in s.values())
    assert set(s) == {"hip.hip_scale_horiz", "hip.hip_scale_depth", BUT, "torso.torso_muscle_dorsi",
                      "measure.measure_hips_circ"}                   # the pelvis, not the chest
    flat = ca.CONSEQUENCES["tail"].sliders
    assert flat[BUT] == pytest.approx(0.15) and "torso.torso_muscle_pectoral" not in flat


def test_a_bigger_tail_asks_more_in_proportion_to_its_size_and_is_capped():
    a, b, c = tail_sliders(), tail_sliders(length=12.0), tail_sliders(length=12.0, base_radius=0.9)
    assert b[BUT] == pytest.approx(0.15 + 0.35 * 1.0)                # load 2: base + slope
    assert b["hip.hip_scale_depth"] == pytest.approx(0.05 + 0.15 * 1.0)
    assert c[BUT] == pytest.approx(0.6)                              # load 4: capped
    assert c["hip.hip_scale_depth"] == pytest.approx(0.3)
    assert all(b[k] > a[k] for k in a)
    # below load 1 it scales down linearly, down to nothing
    half = tail_sliders(length=3.0)
    assert half[BUT] == pytest.approx(0.075)


def test_the_load_is_the_length_and_the_thickness_over_the_neutral_tail():
    kind = K.KINDS["tail"]
    assert kind.load({}) == pytest.approx(1.0)
    assert kind.load({"length": 12.0}) == pytest.approx(2.0) and kind.load({"base_radius": 0.9}) == pytest.approx(2.0)
    assert ca.appendage_load(K.new_spec("tail", params={"length": 12.0, "base_radius": 0.9})) == pytest.approx(4.0)
    assert ca.appendage_load(A.Appendage("tail")) == pytest.approx(1.0)          # a legacy record: by its length


def test_a_pair_of_tails_costs_more_than_one_and_a_switched_off_one_nothing():
    one = tail_sliders()
    s, _, _ = ca.consequences([K.new_spec("tail", side="both")])
    assert s[BUT] > one[BUT]
    off = K.new_spec("tail")
    off.enabled = False
    assert ca.consequences([off])[0] == {}
