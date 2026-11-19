"""The tail's parameter ranges must not make monsters (round 6).

The round-5 critic, at the UI maxima: ``base_radius`` 1.2 - 2.0 is a cone or balloon that swallows the buttocks, the cleft and
the thighs from behind; ``flare`` 2.0 is a flat plate (a flange) at the root; ``crest`` 0.5 - 1.5 stands at full height where
the tail meets the back, a V tooth with a sawtooth side contour. Each is measured here as what the defect IS, at the
parameter's maximum, with a fail-first for each fix.
"""

import math

import numpy as np
import pytest

from minimaster.character import kinds as K
from minimaster.character import mhbase as mh
from minimaster.character import tailrig as T
from minimaster.character.sliders import SliderCatalog

from test_tail_layers_in_skin import BODIES, _build

assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")


@pytest.fixture(scope="module")
def world():
    base, lib = mh.load()
    return base, lib, SliderCatalog(lib.names)


# -- the dorsal ridge ---------------------------------------------------------------------------------------------------


def ridge_at_root(m):
    """How far (in R) the skin's dorsal-most vertex of each ring between the skin point and half a base radius out stands
    above the dorsal-most vertex of the same section built without the ridge: the ridge where the tail meets the back."""
    s = m.s[(m.s >= 0.0) & (m.s <= 0.5 * m.R)]
    flat = T.TailModel(m.p.with_(crest=0.0))
    up = m.frames(s)[:, :, 0]
    top = (m.section(s, "skin") - m.centre(s)[:, None, :]) @ np.array([1.0, 0.0, 0.0])
    top0 = (flat.section(s, "skin") - flat.centre(s)[:, None, :]) @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(up[:, 0], 1.0)                                           # (at rest the ring's up is +x)
    return float((top.max(axis=1) - top0.max(axis=1)).max() / m.R)


@pytest.mark.parametrize("crest", [0.5, 1.0, 1.5])
def test_the_dorsal_ridge_rises_out_of_the_root_instead_of_standing_at_the_back(crest):
    """At the UI maximum (1.5 base radii) the ridge stood at full height AT the skin point, where it cut through the moulded
    back as a V tooth (the critic). Now it rises out of the root over ``CREST_RISE`` R: within half a base radius of the
    skin point it stands at most 0.12 of its full height (measured 0.10), none inside the body, and it reaches its full
    height further out (a ridge, not a parameter that does nothing)."""
    m = T.TailModel(T.TailParams(crest=crest))
    assert ridge_at_root(m) <= 0.12 * crest + 1e-9, (crest, ridge_at_root(m))
    assert float(m.crest_height(-0.5 * m.R)) == 0.0 and float(m.crest_height(0.0)) == 0.0
    s = np.linspace(0.0, 0.6 * m.L, 200)
    assert m.crest_height(s).max() >= 0.6 * crest * m.R                         # (measured 0.70: it then falls toward 0.6 L)


def test_the_round_5_ridge_stands_at_full_height_on_the_back_and_fails(monkeypatch):
    """Fail-first: the round-5 ridge (full height from the skin point inward, falling only toward 0.6 L)."""
    monkeypatch.setattr(T.TailModel, "crest_height",
                        lambda self, s: self.p.crest * self.R * (1.0 - np.clip(np.asarray(s) / (0.6 * self.L), 0.0, 1.0)))
    assert ridge_at_root(T.TailModel(T.TailParams(crest=1.5))) > 0.8 * 1.5


# -- the root flare ------------------------------------------------------------------------------------------------------


def root_wall_slope(m):
    """The steepest the skin's radius falls along the axis over the root (radius per length): a trumpet's wall, or at
    2 and more a plate standing across the axis."""
    s = np.linspace(0.0, 3.0 * m.R, 400)
    r = m.law(s)
    return float((-np.diff(r) / np.diff(s)).max())


@pytest.mark.parametrize("flare", [0.0, 0.35, 1.0, 2.0])
@pytest.mark.parametrize("radius", [0.2, 0.45, 0.9])
def test_a_flared_root_is_a_trumpet_not_a_plate_and_no_wider_than_a_pelvis_carries(flare, radius):
    """At flare 2 the round-5 root eased out over 0.9 R: its wall fell 2.2 radii per length (66 degrees: a flange, the
    critic's plate in --flat). Now the trumpet lengthens with the flare so that its own wall falls at most ``FLARE_SLOPE``
    (33 degrees; with the tail's own taper on top the skin falls at most 0.8, measured 0.75), and the radius at the skin is at
    most what a pelvis carries (``ROOT_CAP`` = 0.9 on the neutral body; the flare saturates). The default tail is unchanged:
    1.35 R at the skin, easing over 0.9 R."""
    m = T.TailModel(T.TailParams(flare=flare, base_radius=radius))
    assert root_wall_slope(m) <= T.FLARE_SLOPE + 0.15, (flare, radius, root_wall_slope(m))
    assert float(m.law(0.0)) <= max(T.ROOT_CAP, radius) + 1e-9
    d = T.TailModel(T.TailParams())
    assert float(d.law(0.0)) == pytest.approx(0.45 * 1.35) and d.flare_length() == pytest.approx(0.9 * 0.45)


def test_the_round_5_flare_is_a_plate_and_fails(monkeypatch):
    """Fail-first: the round-5 flare (decay 0.9 R whatever the flare, no cap) falls more than 2 radii per length at 2."""
    monkeypatch.setattr(T.TailModel, "flare_length", lambda self: T.FLARE_LAMBDA * self.R)
    monkeypatch.setattr(T.TailModel, "flare", property(lambda self: float(self.p.flare)))
    m = T.TailModel(T.TailParams(flare=2.0))
    assert root_wall_slope(m) > 2.0 and float(m.law(0.0)) > T.ROOT_CAP


# -- the base radius -----------------------------------------------------------------------------------------------------


def root_reach(world, name, params):
    """``(the tail's root half width, the hip joints' half distance, the build notes)``: the largest lateral reach of the
    tail's skin within half a base radius of its skin point, against the MakeHuman hip joints (joint-l/r-upper-leg) of the
    same body: a root that reaches past them covers the buttocks. ``params`` go into the record as a saved document has them
    (``Kind.clean`` clamps them, with a note)."""
    from minimaster.character.sliders import Character
    base, lib, cat = world
    c = Character()
    c.macro.update(BODIES[name])
    spec = K.new_spec("tail")
    spec.params.update(params)
    c.appendages.append(spec)
    v = lib.apply(base.verts, c.effective_target_weights(cat))
    report: list = []
    (p,) = K.instantiate_all(c.appendages, base, v, lib, report=report)
    m = p.model.model(p.controls)
    skin = p.mesh()
    n = m.p.ring_verts
    s = np.concatenate([np.repeat(m.s, n), np.full(len(skin.vertices) - len(m.s) * n, -1.0)])
    root = skin.vertices[(s >= 0.0) & (s <= 0.5 * m.R)]
    hips = [np.asarray(v)[np.unique(base.quads[base.groups[g]])].mean(axis=0)[0]
            for g in ("joint-l-upper-leg", "joint-r-upper-leg")]
    return float(np.abs(root[:, 0] - p.anchor[0]).max()), 0.5 * abs(hips[0] - hips[1]), report


@assets
@pytest.mark.parametrize("name", ["neutral", "heavy", "child", "female"])
def test_the_thickest_tail_leaves_the_buttocks_beside_it(world, name):
    """At the UI maximum (base radius 0.9, flare 2) and for a document saved with 2.0 (the old maximum; clamped to 0.9 with
    a note) the root reaches at most 0.95 of the way to the hip joints (measured 0.84 - 0.88); round 5's 1.2 reached 1.56 of
    it and its 2.0 2.6: the cone over both buttocks."""
    for params in (dict(base_radius=0.9, flare=2.0), dict(base_radius=2.0)):
        reach, hip, report = root_reach(world, name, params)
        assert reach <= 0.95 * hip, (name, params, reach, hip)
        if params["base_radius"] > K.TAIL_MAX_RADIUS:
            assert any("base_radius" in str(r) for r in report), report


def test_a_saved_base_radius_above_the_range_is_built_at_the_maximum_with_a_note():
    kind = K.kind_of("tail")
    values, notes = kind.clean({"base_radius": 2.0})
    assert values["base_radius"] == K.TAIL_MAX_RADIUS == T.ROOT_CAP and any("base_radius" in n for n in notes)
    values, notes = kind.clean({"base_radius": 0.62})
    assert values["base_radius"] == 0.62 and not notes


@assets
def test_without_the_cap_a_thick_root_covers_the_buttocks(world, monkeypatch):
    """Fail-first: the round-5 range (2.0) and an uncapped flare: the root reaches past the hip joints."""
    monkeypatch.setattr(K, "TAIL_MAX_RADIUS", 2.0)
    spec = [p for p in K.TailKind.params if p.key == "base_radius"][0]
    monkeypatch.setattr(K.TailKind, "params", tuple(
        p if p.key != "base_radius" else type(p)(**{**p.__dict__, "hi": 2.0}) for p in K.TailKind.params))
    monkeypatch.setattr(T.TailModel, "flare", property(lambda self: float(self.p.flare)))
    reach, hip, _ = root_reach(world, "neutral", dict(base_radius=1.2))
    assert spec.hi == 0.9 and reach > 1.2 * hip, (reach, hip)
