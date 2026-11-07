"""The printability report: what the watertight gate cannot see.

Every shell of a winged, tailed figure can be a closed solid while the print is still
wrong: a tail that only touches the skin, three feathers hanging in the air, a sheet thinner
than the printer resolves. These tests pin the measurements and the verdicts, on analytic
shapes first and then on the real figure.
"""

import dataclasses
import json
import time

import numpy as np
import pytest

from minimaster.character import appendages as A
from minimaster.character import mhbase as mh
from minimaster.character import print_report as pr
from minimaster.character import printprep as pp
from minimaster.character import wings as W
from minimaster.core import primitives
from minimaster.core.mesh import Mesh

needs_assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")

TAIL = A.Appendage("tail", anchor_point=(0.0, 1.6, -1.15))
HORN = A.Appendage("horn", anchor_point=(0.85, 8.55, 0.05), mirror=True)
REF_MM = 32.0 / 17.48            # mm per unit of a 17.5-unit figure printed "medium"


def _ball(r=1.0, at=(0.0, 0.0, 0.0), sub=2):
    return primitives.icosphere(radius=r, subdivisions=sub).translated(at)


def _mh(zup_point):
    """A Z-up point as the MakeHuman point appendage_print_shells will turn back into it."""
    x, y, z = zup_point
    return (x, z, -y)


# -- geometry ---------------------------------------------------------------


def test_solid_contains_agrees_with_the_analytic_sphere():
    ball = _ball(1.0, sub=3)
    solid = pr.Solid(ball)
    P = np.random.default_rng(0).uniform(-1.5, 1.5, (4000, 3))
    r = np.linalg.norm(P, axis=1)
    inside = solid.contains(P)
    assert inside[r < 0.93].all() and not inside[r > 1.001].any()
    assert not pr.Solid(ball).contains(np.zeros((0, 3))).any()


def test_a_cropped_query_still_counts_the_crossings_beyond_the_points():
    """The triangle crop keeps everything at x >= the smallest x asked about, so a point on
    the far side of a thin shell from a second shell still sees an even count."""
    slab = primitives.box(10.0, 10.0, 0.5)                       # closed, 0.5 thick
    s = pr.Solid(slab)
    pts = np.array([[-6.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.3], [0.0, 0.0, 0.6],
                    [5.5, 0.0, 0.0]])
    assert s.contains(pts).tolist() == [False, True, False, False, False]


def test_overlap_components_chain_and_isolate():
    a, b, c = _ball(1.0), _ball(1.0, (1.5, 0, 0)), _ball(1.0, (3.0, 0, 0))
    far = _ball(0.5, (9.0, 0, 0))
    solids = [pr.Solid(m) for m in (a, b, c, far)]
    labels, found = pr.overlap_components(solids)
    assert labels[0] == labels[1] == labels[2] != labels[3]      # a-b, b-c overlap; far does not
    assert (0, 1) in found and (1, 2) in found and (0, 2) not in found
    # known links are re-used without testing them again
    labels2, _ = pr.overlap_components(solids, links={(0, 1), (1, 2)})
    assert labels2 == labels
    # a ball inside another one, with no surface crossing, still overlaps
    inner = pr.Solid(_ball(0.3))
    assert pr.shells_overlap(solids[0], inner) and pr.shells_overlap(inner, solids[0])


def test_embedded_volume_share_is_the_lens_volume():
    body = pr.Solid(_ball(1.0, sub=3))
    other = _ball(1.0, (1.0, 0.0, 0.0), sub=3)
    lens = np.pi * (4 + 1.0) * (2 - 1.0) ** 2 / 12.0            # two unit spheres, centres 1 apart
    want = lens / (4.0 / 3.0 * np.pi)                            # 0.3125 of the second one
    assert pr.embedded_volume_share(other, body) == pytest.approx(want, abs=0.03)
    assert pr.embedded_volume_share(_ball(0.3), body) > 0.97
    assert pr.embedded_volume_share(_ball(0.5, (3.0, 0, 0)), body) == 0.0
    assert pr.embedded_volume_share(_ball(0.5, (1.4, 0, 0)), body) < 0.05    # grazing


def test_thickness_is_the_distance_through_the_solid_not_a_rim_artefact():
    slab = primitives.box(10.0, 10.0, 0.5)
    t, w = pr.thickness_samples(slab, rays=300)
    assert w.sum() == pytest.approx(slab.area(), rel=1e-9)
    top_bottom = np.isclose(t, 0.5)
    assert w[top_bottom].sum() / w.sum() == pytest.approx(200.0 / 220.0, abs=0.02)
    assert t.min() > 0.49                                        # never the 0.000 of a vertex normal
    assert pr._share_below(t, w, 0.6) == pytest.approx(200.0 / 220.0, abs=0.02)
    assert pr._share_below(t, w, 0.4) == 0.0
    assert pr._percentile(t, w, 0.05) == pytest.approx(0.5)


def test_thickness_sampling_is_area_proportional_and_deterministic():
    m = _ball(1.0, sub=3)
    t1, w1 = pr.thickness_samples(m, rays=100)
    t2, w2 = pr.thickness_samples(m, rays=100)
    assert np.array_equal(t1, t2) and np.array_equal(w1, w2)
    assert len(t1) == 100 and w1.sum() == pytest.approx(m.area(), rel=1e-9)
    assert np.allclose(t1, 2.0, atol=0.05)                       # a unit ball is 2 across


def test_root_gap_measures_the_distance_to_the_surface_not_to_a_vertex():
    body = _ball(2.0, sub=3)
    shells = [("body", body), ("arm", _ball(0.5, (3.5, 0, 0), sub=2)),        # 1.0 clear
              ("horn", _ball(0.5, (1.8, 0, 0), sub=2)),                       # buried
              ("tail", _ball(0.5, (0.0, 0.0, 3.0), sub=2))]                   # 0.5 clear
    gaps = pr.root_gaps(shells)
    assert gaps["horn"] == 0.0
    assert gaps["arm"] == pytest.approx(1.0, abs=0.06)
    assert gaps["tail"] == pytest.approx(0.5, abs=0.06)
    assert set(pr.detached_appendages(shells, 0.1)) == {"arm", "tail"}
    with pytest.raises(KeyError):
        pr.root_gaps([("arm", body)])


@needs_assets
def test_two_hundred_rays_estimate_the_bat_membrane_like_the_full_probe(figure):
    """Evenly spaced quantiles alias with the grid order of a slab (every 24th face of the
    membrane is a rim column) and read 1.7% of its area under 0.2 mm against the full probe's
    0.44%. The low-discrepancy sample has no period: 0.41%."""
    base, v, _, _ = figure
    (_, pw), _ = A.back_wings("membrane").build_rigs(base, v)
    membrane = dict(pw.piece_meshes())["membrane"]
    assert len(membrane.faces) > 4000
    t_full, w_full = pr.thickness_samples(membrane, rays=len(membrane.faces))
    t, w = pr.thickness_samples(membrane, rays=200)
    for limit in (0.2 / REF_MM, 0.3 / REF_MM):
        assert pr._share_below(t, w, limit) == pytest.approx(
            pr._share_below(t_full, w_full, limit), abs=0.006)
    assert pr._percentile(t, w, 0.05) * REF_MM == pytest.approx(
        pr._percentile(t_full, w_full, 0.05) * REF_MM, abs=0.02)


# -- the alula (design bug 8) -----------------------------------------------


def _components(rig_pieces_by_name):
    names = list(rig_pieces_by_name)
    solids = [pr.Solid(rig_pieces_by_name[n]) for n in names]
    labels, _ = pr.overlap_components(solids)
    comps = {}
    for n, lab in zip(names, labels):
        comps.setdefault(lab, []).append(n)
    return sorted(comps.values(), key=len)


@pytest.mark.parametrize("kw", [{}, {"span": 8.0}, {"thickness": 0.12}, {"span": 18.0, "thickness": 0.3},
                                {"primaries": 8, "secondaries": 10}])
def test_the_spread_feathered_wing_is_one_overlap_connected_component(kw):
    """The three alula feathers used to float 0.03-0.06 units clear of the hand and every
    covert, an island the watertight gate cannot see and a slicer prints as a speck."""
    rig = W.feathered_rig(**kw)
    comps = _components({p.name: p.mesh for p in rig.pieces})
    assert len(comps) == 1, [c for c in comps if len(c) < 10]


@pytest.mark.parametrize("preset", ["glide", "half_fold", "folded", "upstroke", "downstroke",
                                    "mid_downstroke"])
def test_the_feathered_wing_stays_one_component_in_every_preset_pose(preset):
    rig = W.feathered_rig()
    posed = rig.pose_mesh(rig.pose_from_controls(preset))
    ranges = rig.ranges
    pieces = {}
    for p in rig.pieces:
        a, b = ranges[p.name]
        pieces[p.name] = Mesh(posed.vertices[a:b], p.mesh.faces)
    assert len(_components(pieces)) == 1


def test_the_alula_mount_is_a_closed_solid_that_ends_inside_the_hand():
    rig = W.feathered_rig()
    by = {p.name: p for p in rig.pieces}
    mount = by["alula_mount"].mesh
    rep = mount.integrity_report()
    assert rep["watertight"] and rep["volume"] > 0
    assert by["alula_mount"].bone == "thumb_cmc"                  # rigid with the alula's own joint
    assert pr.shells_overlap(pr.Solid(mount), pr.Solid(by["hand"].mesh))
    for i in (1, 2, 3):
        assert pr.shells_overlap(pr.Solid(mount), pr.Solid(by[f"alula0{i}"].mesh)), i
    # and the draft preview (no bones) does not grow a floating rod
    _, parts = W.feathered_pieces(bones=False)
    assert "alula_mount" not in [p.name for p in parts]


# -- the report on the real figure --------------------------------------------


@pytest.fixture(scope="module")
def figure():
    base, lib = mh.load()
    v = lib.apply(base.verts, mh.macro_weights(gender=1.0, muscle=0.55))
    shells = pp.character_print_shells(base, v)
    return base, v, shells, dict(shells)["body"]


def _codes(rep, level=None):
    return sorted((f.code, f.where) for f in rep.findings
                  if f.level != "info" and (level is None or f.level == level))


@needs_assets
def test_a_tangent_tail_blocks_and_a_bat_wing_pair_passes(figure):
    base, v, _, body = figure
    rep = pr.appendage_report([A.back_wings("membrane"), TAIL], base, v, REF_MM)
    assert not rep.ok
    assert _codes(rep, "block") == [("root-weld", "tail")]
    tail = rep.groups["tail"]
    assert tail.root_embed < 0.05                                 # the measured 2-4%
    for side in ("wing_l", "wing_r"):
        g = rep.groups[side]
        assert g.root_embed >= 0.08 and g.components == 1 and g.shells == 25
        assert g.thin_below_min < 0.05 and g.detached == []
    assert ("tube-tip", "tail") in _codes(rep, "warn")
    text = rep.as_text()
    assert "BLOCKED" in text and "assumes the slicer unions overlapping shells" in text
    json.dumps(rep.as_dict())                                     # a GUI can serialise it


@needs_assets
@pytest.mark.parametrize("style", ["membrane", "feathered"])
@pytest.mark.parametrize("preset", [None, "folded"])
def test_a_mirrored_pair_gets_the_same_report_on_both_sides(figure, style, preset):
    """The right wing is the exact mirror of the left, so the report must not judge the two by
    different pieces or numbers. (An early exit at the warn threshold did: Monte Carlo noise of
    a point decided whether the folded angel's right wing was measured on its arm, 8%, or on
    its hand, 71%.)"""
    base, v, _, _ = figure
    app = A.back_wings(style, **({"preset": preset} if preset else {}))
    rep = pr.appendage_report([app], base, v, REF_MM)
    left, right = rep.groups["wing_l"], rep.groups["wing_r"]
    assert set(left.root_pieces) == set(right.root_pieces) and len(left.root_pieces) >= 2
    assert max(left.root_pieces, key=left.root_pieces.get) == \
        max(right.root_pieces, key=right.root_pieces.get)
    assert left.root_embed == pytest.approx(right.root_embed, abs=0.025)
    for piece, share in left.root_pieces.items():
        assert share == pytest.approx(right.root_pieces[piece], abs=0.025), piece
    assert left.thin_below_min == pytest.approx(right.thin_below_min, abs=1e-9)
    assert [(f.level, f.code) for f in rep.findings if f.where == "wing_l"] == \
        [(f.level, f.code) for f in rep.findings if f.where == "wing_r"]


@needs_assets
def test_a_tangent_horn_blocks_and_the_sunk_ones_pass(figure):
    base, v, _, _ = figure
    rep = pr.appendage_report([HORN], base, v, REF_MM)
    assert _codes(rep, "block") == [("root-weld", "horn_l"), ("root-weld", "horn_r")]
    assert rep.groups["horn_l"].root_embed < 0.01                 # the measured 0.1-0.2%
    for app, off in ((dataclasses.replace(HORN, offset=-0.3), 0.30),
                     (dataclasses.replace(TAIL, offset=-0.45), 0.15)):
        ok = pr.appendage_report([app], base, v, REF_MM)
        assert [f for f in ok.findings if f.code == "root-weld"] == [], app.kind
        assert ok.groups[next(iter(ok.groups))].root_embed > off


@needs_assets
def test_the_default_angel_is_one_welded_component_with_no_blocking_finding(figure):
    base, v, _, _ = figure
    rep = pr.appendage_report([A.back_wings("feathered")], base, v, REF_MM)
    assert rep.ok, rep.as_text()
    assert [f for f in rep.findings if f.code == "detached"] == []
    assert all(g.components == 1 for g in rep.groups.values())
    assert rep.groups["wing_l"].shells == 275


@needs_assets
def test_a_part_floating_clear_of_the_body_blocks_and_a_speck_only_warns(figure):
    base, v, _, body = figure
    chest = body.vertices[(body.vertices[:, 2] > 3) & (body.vertices[:, 2] < 5)].mean(axis=0)
    inside = _ball(1.0, _mh(chest))
    floating = _ball(1.0, _mh(chest + np.array([0.0, 0.0, 30.0])))
    speck = _ball(0.1, _mh(chest + np.array([0.0, 0.0, 30.0])))
    rep = pr.appendage_report([("island", floating)], base, v, REF_MM)
    assert ("detached", "island") in _codes(rep, "block")
    assert ("root-weld", "island") in _codes(rep, "block")
    # a speck that is part of a welded appendage is a warning: 0.03 mm3, 0.1% of its group
    both = [[("thing/arm", inside), ("thing/speck", speck)]]       # ONE appendage, two shells
    rep = pr.appendage_report(both, base, v, REF_MM)
    assert _codes(rep, "block") == []
    assert _codes(rep, "warn") == [("detached", "thing")]
    assert rep.groups["thing"].components == 2
    assert rep.groups["thing"].detached[0][0] == ["thing/speck"]
    # ... but the same part at a size that makes it a real island blocks
    big = pr.appendage_report(both, base, v, 40.0 * REF_MM)
    assert ("detached", "thing") in _codes(big, "block")


@needs_assets
def test_thin_wings_block_small_ones_warn_and_the_size_change_costs_nothing(figure, monkeypatch):
    base, v, _, _ = figure
    cache = {}
    app = A.back_wings("membrane")
    calls = []
    real = pr.thickness_samples
    monkeypatch.setattr(pr, "thickness_samples", lambda *a, **k: calls.append(1) or real(*a, **k))
    sizes = {"tiny": 15.0, "small": 24.0, "medium": 32.0, "large": 45.0}
    verdict = {}
    for name, mm in sizes.items():
        rep = pr.appendage_report([app], base, v, mm / 17.48, cache=cache)
        verdict[name] = (rep.groups["wing_l"].thin_below_min,
                         [(f.code, f.level) for f in rep.findings if f.code == "thin"])
        if name == "tiny":
            n_first = len(calls)
    assert n_first > 0 and len(calls) == n_first, "re-measured the thickness for a new size"
    assert verdict["tiny"][0] > 0.5 and ("thin", "block") in verdict["tiny"][1]
    assert verdict["small"][0] < 0.05 and ("thin", "warn") in verdict["small"][1]
    assert verdict["medium"][1] == [] and verdict["large"][1] == []


@needs_assets
def test_the_thickness_floor_lifts_the_block_and_the_report_names_the_change(figure):
    base, v, _, _ = figure
    tiny = 15.0 / 17.48
    app = A.back_wings("membrane")
    bare = pr.appendage_report([app], base, v, tiny)
    assert ("thin", "wing_l") in _codes(bare, "block")
    lifted = pr.appendage_report([app], base, v, tiny, min_thickness_mm=0.30)
    assert "thin" not in [f.code for f in lifted.blocking]
    assert lifted.groups["wing_l"].thin_below_min < 0.1 < bare.groups["wing_l"].thin_below_min
    info = [f for f in lifted.findings if f.level == "info"]
    assert len(info) == 1 and "thickness 0.220 ->" in info[0].message


@needs_assets
def test_a_folded_membrane_warns_in_the_posed_mesh_and_the_rest_one_stays_clean(figure):
    base, v, _, _ = figure
    rest = pr.appendage_report([A.back_wings("membrane")], base, v, REF_MM)
    folded = pr.appendage_report([A.back_wings("membrane", preset="folded")], base, v, REF_MM)
    assert [f for f in rest.findings if f.code == "thin-posed"] == []
    assert rest.groups["wing_l"].posed_below_min is None
    assert ("thin-posed", "wing_l") in _codes(folded, "warn")
    assert folded.groups["wing_l"].posed_below_min > 0.10
    assert folded.ok                                              # a warning, not a block
    # the feathers are rigid: only the smooth-skinned arm and hand are probed again, and
    # they stay thick, so a folded angel has nothing to warn about
    angel = pr.appendage_report([A.back_wings("feathered", preset="folded")], base, v, REF_MM)
    assert angel.groups["wing_l"].posed_below_min < 0.05
    assert [f for f in angel.findings if f.code == "thin-posed"] == []


@needs_assets
def test_every_failing_shell_is_reported_as_its_own_blocking_finding(figure):
    base, v, shells, _ = figure
    full = _ball(1.0)
    holed = Mesh(full.vertices, full.faces[1:])
    flipped = Mesh(full.vertices, full.faces[:, [0, 2, 1]])
    figure_shells = list(shells) + [("bad-a", holed), ("bad-b", flipped)]
    rep = pr.appendage_report([], base, v, REF_MM, figure_shells=figure_shells)
    assert _codes(rep, "block") == [("shell", "bad-a"), ("shell", "bad-b")]
    # and inside an appendage: an open piece of a wing is named, not just counted
    chest = (0.0, 3.0, 0.0)
    rep = pr.appendage_report([[("thing/arm", _ball(1.0, chest)),
                                ("thing/oops", holed.translated(chest))]], base, v, REF_MM)
    assert ("shell", "thing/oops") in _codes(rep, "block")


@needs_assets
def test_appendages_that_share_volume_warn_and_names_stay_unique(figure):
    base, v, _, _ = figure
    shells = dict(pp.appendage_print_shells([TAIL], base, v))
    tail = shells["tail"]
    mid = tail.vertices.mean(axis=0)
    rep = pr.appendage_report([TAIL, ("blob", _ball(0.5, _mh(mid)))], base, v, REF_MM)
    assert ("overlap", "tail x blob") in _codes(rep, "warn")
    twice = pr.appendage_report([TAIL, TAIL], base, v, REF_MM)
    assert list(twice.groups) == ["tail", "tail#2"]


@needs_assets
def test_a_part_below_the_feet_and_the_base_overhang_are_warned_about(figure):
    base, v, shells, body = figure
    feet = body.vertices[:, 2].min()
    low = _ball(0.5, _mh((0.0, 0.0, feet - 1.5)))
    rep = pr.appendage_report([("low/arm", low)], base, v, REF_MM)
    assert ("below-feet", "low") in _codes(rep, "warn")
    app = A.back_wings("membrane")
    assembled = pp.assemble_character(shells + pp.appendage_print_shells([app], base, v),
                                      size="medium", reference=pp.FIGURE_SHELLS)[1]
    rep = pr.appendage_report([app], base, v, assembled.scale, assembly=assembled)
    assert ("overhang", "base") in _codes(rep, "warn")


# -- tails and horns: the analytic profile is the builder's --------------------


def _ring_diameter(mesh, ring, ring_verts):
    p = mesh.vertices[ring * ring_verts:(ring + 1) * ring_verts]
    h = ring_verts // 2
    return float(min(np.linalg.norm(p[i] - p[i + h]) for i in range(h)))


@pytest.mark.parametrize("kind,builder,rv,nseg", [("tail", A.make_tail, 10, 12),
                                                  ("horn", A.make_horn, 8, 8)])
def test_the_tube_profile_matches_every_ring_of_the_mesh_the_builder_makes(kind, builder, rv, nseg):
    """If a builder changes its radius law this fails, instead of the report quietly
    describing a tail that no longer exists. Ring k of the loft is station k / segments."""
    mesh = builder()
    t = np.linspace(0.0, 1.0, 2001)
    d = pr.tube_diameters(kind, {}, scale=1.0)
    for k in range(nseg + 1):
        want = float(np.interp(k / nseg, t, d))
        assert _ring_diameter(mesh, k, rv) == pytest.approx(want, rel=0.04), (kind, k)
    for scale in (0.5, 2.0):
        assert pr.tube_diameters(kind, {}, scale=scale)[0] == pytest.approx(scale * d[0])
    assert pr.tube_diameters(kind, {"base_radius": 0.9}, scale=1.0)[0] > 1.7 * d[0]
    assert (np.diff(d) <= 1e-12).all()                           # tapers monotonically
    assert pr.tube_diameters("wing", {}) is None


@pytest.mark.parametrize("kind,builder,rv", [("tail", A.make_tail, 10), ("horn", A.make_horn, 8)])
def test_the_tube_profile_follows_the_params_the_appendage_is_built_with(kind, builder, rv):
    params = {"base_radius": 0.8}
    mesh = builder(**params).scaled(0.5)               # Appendage.scale = 0.5
    d = pr.tube_diameters(kind, params, scale=0.5)
    assert d[0] == pytest.approx(_ring_diameter(mesh, 0, rv), rel=0.03)
    assert d[0] > 1.5 * pr.tube_diameters(kind, {}, scale=0.5)[0]


@needs_assets
def test_a_tail_that_is_mostly_a_thread_blocks_at_a_size_that_cannot_hold_it(figure):
    base, v, _, _ = figure
    sunk = dataclasses.replace(TAIL, offset=-0.6)
    medium = pr.appendage_report([sunk], base, v, REF_MM)
    assert medium.ok and ("tube-tip", "tail") in _codes(medium, "warn")
    assert medium.groups["tail"].tube["tip_mm"] == pytest.approx(0.073, abs=0.01)
    tiny = pr.appendage_report([dataclasses.replace(HORN, offset=-0.3)], base, v, 0.4 * REF_MM)
    assert ("tube-thin", "horn_l") in _codes(tiny, "block")       # a third of the horn < 0.2 mm
    assert tiny.groups["horn_l"].tube["below_min"] > 0.25


# -- cost -------------------------------------------------------------------


@needs_assets
def test_the_report_is_interactive_for_a_bat_and_a_tail_and_free_when_cached(figure):
    base, v, _, _ = figure
    apps = [A.back_wings("membrane"), TAIL]
    cache = {}
    t0 = time.perf_counter()
    first = pr.appendage_report(apps, base, v, REF_MM, cache=cache)
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    again = pr.appendage_report(apps, base, v, REF_MM * 0.9, cache=cache)
    warm = time.perf_counter() - t0
    assert cold < 3.0, f"cold report took {cold:.2f} s (measured ~0.3 s)"
    assert warm < 0.5 and warm < cold, (cold, warm)
    # the cached report at another size is the report a fresh run gives at that size ...
    fresh = pr.appendage_report(apps, base, v, REF_MM * 0.9)
    assert [(f.level, f.code, f.where) for f in again.findings] == \
        [(f.level, f.code, f.where) for f in fresh.findings]
    assert again.groups["wing_l"].thin_below_thin == pytest.approx(
        fresh.groups["wing_l"].thin_below_thin)
    # ... and a size that really changes the verdict changes it
    assert ("thin", "wing_l") in _codes(again, "warn") and ("thin", "wing_l") not in _codes(first)
    # a different body is a different measurement
    v2 = v.copy()
    v2[:, 0] *= 1.05
    other = pr.appendage_report(apps, base, v2, REF_MM, cache=cache)
    assert other.groups["tail"].root_embed != first.groups["tail"].root_embed
