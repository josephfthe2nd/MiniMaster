"""The weld (``minimaster.character.weld``, round 6): the body's own subdivided skin moulded onto every root.

The user: the appendages "seem placed onto the model", "slapped over skin"; the skin does not "mold or conform to
connect around" the root.  Two earlier ways failed: moving the cage (a quad is 0.28 units, a root 0.4-0.6) and an
overlay collar shell (a patch outline in the geometry, absent at level 0 and wherever its guard dropped it).  These
tests measure what the defect IS, from the geometry alone (``weld.arrival_report``: they read the bare junction of
e7e16b8 and a welded skin alike):

* the ANGLE at which the skin's last strip arrives on the root (0 = tangent, 90 = a pipe in a plate): 55-76 degrees
  bare on every kind, a fraction of that welded;
* that the skin never crosses itself (exact triangle tests) and stays one closed surface: the weld moves vertices,
  never the triangles, and a root whose weld would fold the skin is left as the junction made it, with a warning;
* that the plug under a root stays inside it, the last ring lies on the wall, and nothing outside the weld moves (a
  figure without appendages is byte-identical; a drooping tail is not webbed to the thighs);
* the pipelines (the command line, the lab's final render and export, the print report) carry the welded body and
  one function decides the default level for all of them; documents keep their meaning (``collar`` = 0 is the bare
  root, ``anatomy["collar"] = false`` no seam, ``"shell"`` the round-4 collar).
"""

import json
import subprocess
import sys
import time
import warnings

import numpy as np
import pytest

from minimaster.character import collar as CO
from minimaster.character import kinds as K
from minimaster.character import mhbase as mh
from minimaster.character import portrait as po
from minimaster.character import weld as W
from minimaster.character.sliders import Character, SliderCatalog
from minimaster.core.mesh import Mesh
from minimaster.core.subdiv import cube_cage, quads_to_mesh, subdivide_quads

assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")

BODIES6 = {"neutral": dict(gender=1.0, muscle=0.55), "heavy": dict(weight=1.0, gender=1.0, muscle=1.0),
           "child": dict(age=0.1875), "slim_f": dict(gender=0.0, weight=0.0), "tall": dict(height=1.0, gender=1.0),
           "female": dict(gender=0.0)}


# -- a synthetic body: a sphere, a closed tube leaving its +Z pole ------------------------------------------------------


def _sphere(levels=6, radius=6.0):
    v, q = cube_cage(2.0)
    v, q = subdivide_quads(v, q, levels)
    v = v / np.linalg.norm(v, axis=1, keepdims=True) * radius
    m = quads_to_mesh(v, q)
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


def _tube(x=0.0, r=0.6, z0=4.5, z1=9.0, ring=24, rings=20):
    zs = np.linspace(z0, z1, rings)
    ang = np.arange(ring) * 2 * np.pi / ring
    pts = np.array([[x + r * np.cos(a), r * np.sin(a), z] for z in zs for a in ang])
    faces = []
    for i in range(rings - 1):
        for j in range(ring):
            a, b = i * ring + j, i * ring + (j + 1) % ring
            faces += [[a, b, b + ring], [a, b + ring, a + ring]]
    pts = np.vstack([pts, [x, 0.0, z0], [x, 0.0, z1]])
    n0, n1 = len(pts) - 2, len(pts) - 1
    last = (rings - 1) * ring
    for j in range(ring):
        faces.append([n0, (j + 1) % ring, j])
        faces.append([n1, last + j, last + (j + 1) % ring])
    return CO.outward(Mesh(pts, np.array(faces)))


@pytest.fixture(scope="module")
def sphere():
    return _sphere()


def _weld_tube(sphere, x=0.0, profile=None, diag=None, **kw):
    V, F = sphere
    wall = _tube(x=x)
    z = np.sqrt(36.0 - x * x)
    centre = np.array([x, 0.0, z])
    axis = np.array([[x, 0.0, z - 1.0], [x, 0.0, z + 3.0]])
    out = W.weld_skin(V, F, wall, centre, 0.6, profile or W.WeldProfile(), axis=axis, diag=diag, **kw)
    return out, wall, centre


def test_a_profile_is_validated_hashable_data():
    p = W.WeldProfile()
    assert p.active and p.digest() == W.WeldProfile().digest() != p.with_(reach=0.5).digest()
    assert not p.with_(strength=0.0).active and p.k(0.5) == pytest.approx(0.45)
    assert p.k(5.0) == pytest.approx(p.reach_max)                          # capped in units
    for bad in (dict(reach=-1.0), dict(region=0.1), dict(up=(0, 0, 0)), dict(inset=float("nan"))):
        with pytest.raises(ValueError):
            W.WeldProfile(**bad)
    up = W.WeldProfile(up=(0.0, 2.0, 0.0), open=1.0, ventral=0.2)
    s = up.shape(np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]), np.zeros(3))
    assert s[0] == pytest.approx(1.0) and s[1] == pytest.approx(0.2) and 0.2 < s[2] < 1.0


def test_the_default_kind_does_not_weld():
    assert K.Kind().weld_profile(None) is None and not K.Kind().welds({})


def test_a_tube_on_a_sphere_is_welded_tangentially_into_one_closed_skin(sphere):
    """What the weld replaces: a tube leaving a sphere square (the bare junction: ~90 degrees); welded, the skin's last
    strip arrives on the tube nearly tangentially, nothing crosses, the plug stays inside, the last ring is ON the
    wall, and the triangles are the sphere's own (one closed surface)."""
    V, F = sphere
    d: dict = {}
    out, wall, centre = _weld_tube(sphere, diag=d)
    assert d["applied"] and d["crossings"] == 0
    before = W.arrival_report(V, F, wall, centre, 0.6, V0=V)
    after = W.arrival_report(out, F, wall, centre, 0.6, V0=V)
    assert before["p50"] > 70.0
    assert after["p50"] < 15.0 and after["p90"] < 30.0, after          # measured 11.1 / 25.8 (edge 0.15; the first
    #                                                                       strip outside the wall, round 7's measure)
    assert after["crossings"] == 0 and after["plug_outside"] == 0
    wp = W.Surface(wall.vertices, wall.faces, centre, 4.0)
    _, _, b_top, _ = wp.closest(out[d["ids"]["top"]])
    lift = W.TOP_LIFT * d["edge"]                                            # the last ring: a hair off the wall
    assert 0.5 * lift < b_top.min() and b_top.max() < 1.5 * lift, (b_top.min(), b_top.max(), lift)
    _, _, b_cap, _ = wp.closest(out[d["ids"]["cap"]])
    assert b_cap.max() < 0.0                                                 # the plug is inside the tube
    _, _, b_sl, _ = wp.closest(out[d["ids"]["sleeve"]])
    assert b_sl.min() > -2e-3                                                # the sleeve is outside it


def test_nothing_outside_the_weld_moves_and_no_target_is_the_mesh_itself(sphere):
    V, F = sphere
    d: dict = {}
    out, wall, centre = _weld_tube(sphere, diag=d)
    moved = np.nonzero(np.any(out != V, axis=1))[0]
    assert len(moved) and np.isin(moved, d["ids"]["region"]).all()
    assert np.all(out[np.setdiff1d(np.arange(len(V)), d["ids"]["region"])] == V[np.setdiff1d(np.arange(len(V)),
                                                                                         d["ids"]["region"])])
    assert np.linalg.norm(V[moved] - centre, axis=1).max() < 0.6 + 3.0 * W.WeldProfile().k(0.6) + 0.3
    mesh = Mesh(V, F)
    assert W.weld_mesh(mesh, []) is mesh and W.weld_vertices(V, F, []) is V
    assert W.weld_mesh(mesh, [], z_up=False) is mesh


def test_an_earlier_root_s_weld_is_left_alone(sphere):
    """Two roots side by side: the second weld ends before the first one's region and never moves it."""
    V, F = sphere
    t1 = W.WeldTarget("a", _tube(x=-1.2), np.array([-1.2, 0.0, np.sqrt(36 - 1.44)]), 0.6, W.WeldProfile(),
                      np.array([[-1.2, 0, 4.8], [-1.2, 0, 9.0]]))
    t2 = W.WeldTarget("b", _tube(x=1.2), np.array([1.2, 0.0, np.sqrt(36 - 1.44)]), 0.6, W.WeldProfile(),
                      np.array([[1.2, 0, 4.8], [1.2, 0, 9.0]]))
    diags: list = []
    both = W.weld_vertices(V, F, [t1, t2], diags=diags)
    only1 = W.weld_vertices(V, F, [t1])
    r1 = diags[0]["ids"]["region"]
    assert diags[0]["applied"] and diags[1]["applied"]
    assert np.all(both[r1] == only1[r1])
    assert len(CO.crossing_pairs(both, F[W.Topology.of(F, len(V)).faces_of(np.concatenate([r1, diags[1]["ids"]["region"]]))])) == 0


def test_a_weld_that_would_cross_the_skin_is_left_out_with_a_warning(sphere, monkeypatch):
    """The guard's last resort: torn skin never ships."""
    V, F = sphere
    real = CO.crossing_pairs
    calls = {"n": 0}

    def fake(verts, faces):
        calls["n"] += 1
        if verts is V or np.array_equal(verts, V):
            return np.zeros((0, 2), dtype=np.int64)
        return np.array([[0, 1]], dtype=np.int64)              # every welded trial "crosses"

    monkeypatch.setattr(W.CO, "crossing_pairs", fake)
    t = W.WeldTarget("a", _tube(), np.array([0.0, 0.0, 6.0]), 0.6, W.WeldProfile(), np.array([[0, 0, 5.0], [0, 0, 9.0]]))
    diags: list = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = W.weld_vertices(V, F, [t], diags=diags)
    assert np.array_equal(out, V) and not diags[0]["applied"] and calls["n"] >= 2
    assert any("not welded onto the root" in str(w.message) for w in caught)
    monkeypatch.setattr(W.CO, "crossing_pairs", real)


# -- the real bodies --------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def world():
    base, lib = mh.load()
    return base, lib, SliderCatalog(lib.names)


_FIG: dict = {}


def _figure(world, text, body="neutral", poses=(), levels=1):
    """``(character, placed, V0, F, targets, V1, diags)``: the shaped body meshed at ``levels`` (MakeHuman frame),
    the weld targets and the welded skin, as the pipelines make them."""
    key = (text, body, tuple(poses), levels)
    if key in _FIG:
        return _FIG[key]
    base, lib, cat = world
    c = Character()
    c.macro.update(BODIES6[body])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in text.split(";"):
            c.appendages.append(K.add_from_text(t, c.appendages, base=base, lib=lib))
        for t in poses:
            K.apply_pose(*K.pose_from_text(t, c.appendages))
        verts = lib.apply(base.verts, c.effective_target_weights(cat))
        placed = c.instantiate_appendages(base, verts, lib)
        shaped = c.shaped_body(base, verts, placed)
        shell = po.character_shells(base, shaped, levels=levels, with_eyes=False)[0]
        V0, F = CO.skin_from_shell(shell.mesh)
        V0 = np.array(V0)
        targets = W.weld_targets(placed)
        diags: list = []
        V1 = W.weld_vertices(V0, F, targets, diags=diags)
    _FIG[key] = (c, placed, V0, F, targets, V1, diags)
    if len(_FIG) > 24:
        _FIG.pop(next(iter(_FIG)))
    return _FIG[key]


def _reports(fig):
    """Per welded root: ``(diag, before, after)`` arrival reports around the root's own centre, split by its open side."""
    c, placed, V0, F, targets, V1, diags = fig
    out = []
    for t, d in zip(targets, diags):
        centre = d.get("centre", t.centre)
        before = W.arrival_report(V0, F, t.wall, centre, t.radius, V0=V0, up=t.profile.up)
        after = W.arrival_report(V1, F, t.wall, centre, t.radius, V0=V0, up=t.profile.up)
        out.append((t, d, before, after))
    return out


# the arrival angle on the side each kind opens to (its p50), level 1, neutral body: what the round-6 weld reaches
# (measured: tail 22, wing 16, dragon 16, feathered 23, horn 30, ram 31, lower arms 10, extra legs 8 / 26; bare 47-76)
L1_OPEN_P50 = {"tail": 30.0, "wing": 25.0, "wing:style=dragon": 25.0, "wing:style=feathered": 32.0, "horn": 40.0,
               "horn:shape=ram": 40.0, "harvest:preset=lower_arms": 16.0, "harvest:preset=extra_legs": 36.0}
# (ram: 34 against a bare 43 -- the cuff is one level-1 edge wide; docs/weld.md)


@assets
@pytest.mark.parametrize("text", sorted(L1_OPEN_P50))
def test_every_kind_s_root_is_welded_at_level_one(world, text):
    fig = _figure(world, text)
    reps = _reports(fig)
    assert reps, text
    for t, d, before, after in reps:
        assert d["applied"], (text, t.name, d)
        assert after["open_p50"] <= L1_OPEN_P50[text], (text, t.name, after["open_p50"], before["open_p50"])
        assert after["open_p50"] < 0.8 * before["open_p50"]                       # the bare junction: 34-76
        assert after["crossings"] <= after["crossings_before"] and after["plug_outside"] == 0


# tail and lower arms on six bodies (level 1): the open side's p50 (measured 7-19 tail, 5-8 arms; the first pass of
# round 6: 19-35 and 7-10)
SIX_BOUND = {"tail": 22.0, "harvest:preset=lower_arms": 14.0}


@assets
@pytest.mark.parametrize("body", sorted(BODIES6))
@pytest.mark.parametrize("text", sorted(SIX_BOUND))
def test_the_tail_and_the_lower_arms_are_welded_on_six_bodies(world, text, body):
    for t, d, before, after in _reports(_figure(world, text, body)):
        assert d["applied"], (text, body, t.name, d)
        assert after["open_p50"] <= SIX_BOUND[text], (text, body, t.name, after["open_p50"])
        assert after["crossings"] <= after["crossings_before"] and after["plug_outside"] == 0


@assets
@pytest.mark.parametrize("text,bound", [("tail", 16.0), ("harvest:preset=lower_arms", 10.0)])     # measured 11.8, 4.7
def test_level_two_arrives_flatter_than_level_one(world, text, bound):
    for t, d, before, after in _reports(_figure(world, text, levels=2)):
        assert d["applied"], (text, t.name, d)
        assert after["open_p50"] <= bound, (text, t.name, after["open_p50"])
        assert after["crossings"] <= after["crossings_before"]


HARD = [("wing", ()), ("wing:style=dragon", ()), ("tail", ("tail-1:lift=40",)), ("tail", ("tail-1:lift=-40",)), ("tail", ("tail-1:lift=60",)),
        ("tail", ("tail-1:lift=-60",)), ("tail:base_radius=0.85", ()), ("tail:base_radius=0.9,length=14", ()),
        ("wing", ("wing-1:preset=half_fold",)),
        ("wing", ("wing-1:preset=folded",)), ("harvest:preset=upper_arms", ()), ("harvest:preset=extra_legs", ()),
        ("tail;harvest:preset=extra_legs", ()), ("wing:style=feathered;harvest:preset=upper_arms", ())]


@assets
@pytest.mark.parametrize("text,poses", HARD)
@pytest.mark.parametrize("body", ["neutral", "heavy"])
def test_the_skin_never_crosses_itself_and_stays_closed_on_the_hard_cases(world, text, poses, body):
    """Lifted and drooping tails, thick tails, half-folded and folded wings, the fan pair, extra legs and the
    combinations: whatever the weld does (or leaves out), the skin around every root crosses itself no more than the
    junction left it, the plug stays inside, and the triangles are the body's own."""
    c, placed, V0, F, targets, V1, diags = _figure(world, text, body, poses)
    assert V1.shape == V0.shape and np.isfinite(V1).all()
    assert Mesh(V1, F).integrity_report()["watertight"]                   # one closed skin, no degenerate triangle
    for t, d in zip(targets, diags):
        after = W.arrival_report(V1, F, t.wall, d.get("centre", t.centre), t.radius, V0=V0)
        assert after["crossings"] <= after["crossings_before"], (text, poses, body, t.name, after)
        assert after["plug_outside"] == 0, (text, poses, body, t.name)
        if not d.get("applied"):
            assert np.array_equal(V1[d.get("ids", {}).get("region", np.zeros(0, int))],
                                  V0[d.get("ids", {}).get("region", np.zeros(0, int))]) or len(diags) > 1


@assets
def test_the_plug_is_inside_the_root_and_the_last_ring_on_it(world):
    c, placed, V0, F, targets, V1, diags = _figure(world, "tail")
    t, d = targets[0], diags[0]
    wp = W.Surface(t.wall.vertices, t.wall.faces, t.centre, 5.0)
    _, _, b_top, _ = wp.closest(V1[d["ids"]["top"]])
    _, _, b_cap, _ = wp.closest(V1[d["ids"]["cap"]])
    lift = W.TOP_LIFT * d["edge"]                                         # the last ring: a hair off the wall, outside
    assert 0.5 * lift < b_top.min() and b_top.max() < 1.5 * lift and b_cap.max() < 0.0
    moved = np.nonzero(np.any(V1 != V0, axis=1))[0]
    assert np.isin(moved, d["ids"]["region"]).all()                       # nothing outside the region moves


@assets
def test_a_figure_without_appendages_is_byte_identical(world):
    base, lib, cat = world
    c = Character()
    verts = lib.apply(base.verts, c.effective_target_weights(cat))
    shell = po.character_shells(base, verts, levels=1, with_eyes=False)[0]
    assert W.welded_shell_mesh(shell.mesh, []) is shell.mesh
    assert W.welded_shell_mesh(shell.mesh, [], c) is shell.mesh


@assets
def test_a_drooping_tail_is_not_webbed_to_the_thighs(world):
    """The prototype webbed a drooping tail to the thighs 1.8 units down; the wall that counts is the root's."""
    c, placed, V0, F, targets, V1, diags = _figure(world, "tail", poses=("tail-1:lift=-60",))
    t, d = targets[0], diags[0]
    centre = d.get("centre", t.centre)
    moved = np.any(V1 != V0, axis=1)
    far = np.linalg.norm(V0 - centre, axis=1) > 1.6
    assert not np.any(moved & far)
    assert moved.sum() > 50                                               # and it IS welded at its root
    # no moved skin is welded to the tail past its root: the wall point it was nearest to is near the root
    pr = t.profile
    clip = t.radius + pr.wall_clip * pr.k(t.radius) * max(pr.open, pr.ventral, 1.0)
    wp = W.Surface(t.wall.vertices, t.wall.faces, centre, 6.0)
    q, _, _, _ = wp.closest(V0[moved & ~np.isin(np.arange(len(V0)), d["ids"]["cap"])])
    assert np.linalg.norm(q - centre, axis=1).max() < clip + 0.05


@assets
def test_an_arm_hanging_along_the_flank_is_not_webbed_to_it(world):
    c, placed, V0, F, targets, V1, diags = _figure(world, "harvest:preset=lower_arms")
    moved = np.any(V1 != V0, axis=1)
    for t, d in zip(targets, diags):
        centre = d.get("centre", t.centre)
        near = np.linalg.norm(V0 - centre, axis=1) < 1.5
        assert moved[near].sum() > 50
    far = np.ones(len(V0), bool)
    for t, d in zip(targets, diags):
        far &= np.linalg.norm(V0 - d.get("centre", t.centre), axis=1) > 1.6
    assert not np.any(moved & far)                                       # the forearm along the flank: no web


@assets
def test_a_root_is_welded_within_a_generous_budget(world):
    """Measured on this box (4 cores, shared): tail 0.4-1.3 s at level 1, 3-6 s at level 2 (the budget the plan set,
    0.15 / 0.5 s, is NOT met: docs/weld.md); a generous bound so a regression of an order of magnitude shows."""
    c, placed, V0, F, targets, V1, diags = _figure(world, "tail")
    W.weld_vertices(V0, F, targets)
    t0 = time.perf_counter()
    W.weld_vertices(V0, F, targets)
    assert time.perf_counter() - t0 < 6.0


# -- documents, the switch, the level -----------------------------------------------------------------------------------


@assets
def test_collar_zero_is_the_bare_root_and_the_seam_switch_keeps_its_meaning(world):
    fig = _figure(world, "tail:collar=0")
    assert not fig[4] and np.array_equal(fig[5], fig[2])                 # no target, not a vertex moved
    c = Character()
    c.appendages.append(K.new_spec("tail"))
    assert W.seam(c) == "weld" and "collar" not in json.dumps(c.to_dict())
    c.anatomy["collar"] = False
    assert W.seam(c) == "none"
    c.anatomy["collar"] = "shell"
    assert W.seam(c) == "collar" and c.collar_enabled
    back = Character.from_dict(json.loads(json.dumps(c.to_dict())))
    assert W.seam(back) == "collar"
    c.anatomy.pop("collar")
    c.anatomy["junction"] = False
    assert W.seam(c) == "none"


@assets
def test_old_documents_load_and_weld(world):
    """The round-3 horn fixture (a 16-sided spiral) and the wing fixtures pinned to earlier placements still build;
    their roots are welded or left as they were, and the skin never crosses itself."""
    import pathlib
    base, lib, cat = world
    root = pathlib.Path(__file__).parent / "fixtures"
    files = [root / "horns_round3.mmchar", root / "wing_c808d60" / "wing_untouched.mmchar"]
    for f in files:
        c = Character.load(str(f))
        assert W.seam(c) == "weld"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            verts = lib.apply(base.verts, c.effective_target_weights(cat))
            placed = c.instantiate_appendages(base, verts, lib)
            shaped = c.shaped_body(base, verts, placed)
            shell = po.character_shells(base, shaped, levels=1, with_eyes=False)[0]
            V0, F = CO.skin_from_shell(shell.mesh)
            targets = W.weld_targets(placed)
            diags: list = []
            V1 = W.weld_vertices(np.array(V0), F, targets, diags=diags)
        assert targets and any(d.get("applied") for d in diags), f
        for t, d in zip(targets, diags):
            r = W.arrival_report(V1, F, t.wall, d.get("centre", t.centre), t.radius, V0=np.array(V0))
            assert r["crossings"] <= r["crossings_before"], (f, t.name)


def test_one_function_decides_the_default_level_from_the_records():
    """B-I3: the command line asked the BUILT appendages (``collar.default_levels(placed)``), the lab the records; an
    ossicone came out at different levels.  Now ``weld.default_levels(character)`` decides for both, from the
    records, building nothing."""
    def doc(*texts, **anatomy):
        c = Character()
        for t in texts:
            c.appendages.append(K.add_from_text(t, c.appendages))
        c.anatomy.update(anatomy)
        return c

    assert W.default_levels(doc()) == 0
    assert W.default_levels(doc("tail")) == 1
    assert W.default_levels(doc("tail:collar=0")) == 0
    assert W.default_levels(doc("horn:shape=ossicone")) == 0              # no weld on a skin-covered horn
    assert W.default_levels(doc("horn")) == 1
    assert W.default_levels(doc("harvest:preset=lower_arms")) == 1
    assert W.default_levels(doc("tail", collar=False)) == 0
    assert W.default_levels(doc("tail", junction=False)) == 0
    assert W.default_levels(doc("tail", collar="shell")) == CO.default_levels(doc("tail").appendages)
    old = doc("tail")                                                    # an old document's thick tail: built at the
    old.appendages[0].params["base_radius"] = 2.0                         # range's end (0.9), so it is welded like any
    assert W.default_levels(old) == 1


@assets
def test_the_command_line_and_the_lab_take_the_same_level(tmp_path):
    from minimaster.character import lab as L
    res = L.Resources.shared()
    for text, want in (("tail", 1), ("horn:shape=ossicone", 0)):
        done = subprocess.run([sys.executable, "-m", "minimaster", "character", "--add", text, "-o",
                               str(tmp_path / "x.stl")], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        assert ("subdivided once" in done.stdout) == (want == 1), (text, done.stdout)
        lb = L.Lab(res, size=(120, 120), threaded=False)
        try:
            lb.doc.add_appendage(text.split(":")[0], params=dict(
                kv.split("=") for kv in text.split(":")[1].split(",")) if ":" in text else None)
            assert lb.final_levels() == want, text
        finally:
            lb.close()


@assets
def test_the_panel_s_export_path_takes_the_level_policy_and_welds():
    """B-I4: ``Lab.export_async`` (what the Character tab's Export button runs) meshes the body at the policy's
    level and carries the welded skin."""
    from minimaster.character import lab as L
    res = L.Resources.shared()
    lb = L.Lab(res, size=(120, 120), threaded=False)
    try:
        lb.doc.add_appendage("tail")
        task = lb.export_async(None, size="medium", force=True)
        assert task.wait(300.0) and task.error is None, task.error
        tri_weld = task.value.assembly.triangles
        assert lb.engine.stats["weld"] >= 1
        lb.doc.set_anatomy(collar=False)                                # no seam: level 0, the cage
        task = lb.export_async(None, size="medium", force=True)
        assert task.wait(300.0) and task.error is None
        assert task.value.assembly.triangles < tri_weld / 2
    finally:
        lb.close()


# -- the pipelines ------------------------------------------------------------------------------------------------------


@assets
def test_the_lab_draws_the_welded_skin_in_the_final_render_and_caches_it():
    from minimaster.character import lab as L
    res = L.Resources.shared()
    d = L.LabDocument(res)
    spec = d.add_appendage("tail")
    eng = d.engine
    draft = eng.draft_scene(d.character)
    assert any("welded onto the roots in the final render only" in n for n in draft.notes)
    final = eng.final_scene(d.character, levels=1)
    assert [o for o in final.owners if o[0] == "tail-1"] == [("tail-1", "")]     # no collar shell
    assert eng.stats["weld"] == 1
    plain = eng.body_shells(eng.shaped_state(d.character, eng.body_state(d.character),
                                             [(spec, eng.built(spec, eng.body_state(d.character), False), False, False)]),
                            "#5b7c8d", 1)[0].mesh
    welded = final.shells[0].mesh
    assert welded is not plain and np.array_equal(welded.faces, plain.faces)
    assert np.any(welded.vertices != plain.vertices)
    eng.final_scene(d.character, levels=1)
    assert eng.stats["weld"] == 1                                         # cached
    d.set_control(spec.id, "lift", 40.0)
    eng.final_scene(d.character, levels=1)
    assert eng.stats["weld"] == 2                                         # another pose, another weld
    d.set_anatomy(collar=False)
    final = eng.final_scene(d.character, levels=1)
    assert eng.stats["weld"] == 2 and np.array_equal(final.shells[0].mesh.vertices, eng.body_shells(
        eng.shaped_state(d.character, eng.body_state(d.character),
                         [(spec, eng.built(spec, eng.body_state(d.character), False), False, False)]),
        "#5b7c8d", 1)[0].mesh.vertices)


@assets
def test_the_command_line_prints_and_draws_the_welded_skin(tmp_path):
    outs = {}
    for tag, extra in (("weld", []), ("bare", ["--no-collar"])):
        stl = tmp_path / f"{tag}.stl"
        done = subprocess.run([sys.executable, "-m", "minimaster", "character", "--add", "tail", "--subdivide", "1",
                               *extra, "-o", str(stl)], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        assert "tail-1           1 shell" in done.stdout and "printability  OK" in done.stdout
        png = tmp_path / f"{tag}.png"
        done = subprocess.run([sys.executable, "-m", "minimaster", "character", "--add", "tail", "--subdivide", "1",
                               "--azimuth", "150", "--elevation", "20", "--res", "200", "260", *extra, "-o", str(png)],
                              capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        outs[tag] = (stl.read_bytes(), png.read_bytes())
    assert outs["weld"][0] != outs["bare"][0] and outs["weld"][1] != outs["bare"][1]
    assert len(outs["weld"][0]) == len(outs["bare"][0])                   # the same triangles, moved


@assets
def test_the_layers_stay_inside_the_welded_skin(world):
    """The weld only moves skin outward near the root (onto the fillet, over the original skin): the tail's bone and
    muscle stay inside the body or the tail as round 4 measured them against the junction's skin."""
    from test_tail_layers_in_skin import inside_closed
    c, placed, V0, F, targets, V1, diags = _figure(world, "tail")
    p = placed[0]
    from minimaster.character import printprep as pp
    body = Mesh(V1, F)
    tail = p.mesh()
    layers = p.kind.layers_on(p, c.shaped_body(world[0], world[1].apply(world[0].verts, c.effective_target_weights(world[2])), placed))
    for name in ("bone", "muscle"):
        pts = layers[name].vertices
        out = ~inside_closed(pts, body) & ~inside_closed(pts, tail)
        assert out.mean() < 0.01, (name, out.mean())
    del pp


# --------------------------------------------------------------------------
# round 6, second pass: the sleeve follows the last ring as a membrane, cuts to the reach fade at a bounded slope, the
# guard takes a patch out instead of the whole root.  Each fails on the first pass (f90a61f) or with the named switch.


def _back_reach(fig):
    """The share of the nominal reach that the last ring keeps on the root's open side (median)."""
    c, placed, V0, F, targets, V1, diags = fig
    t, d = targets[0], diags[0]
    top = d["ids"]["top"]
    kv = dict(zip(d["ids"]["region"].tolist(), d["kv"].tolist()))
    side = CO._unit(V0[top] - np.asarray(d.get("centre", t.centre))) @ np.asarray(t.profile.up)
    return float(np.median([kv[int(v)] for v in top[side > 0.6]])) / d["k0"]


@assets
@pytest.mark.parametrize("body", sorted(BODIES6))
def test_the_tail_keeps_its_reach_on_the_back(world, body):
    """The skin climbs the tail's back by (nearly) the whole reach on every body, with at most one guard round, and the
    weld runs round (nearly) the whole ring.  The first pass cut the back's reach to 0.4 of it for two sliver triangles
    (``PLACEMENT = "profile"`` brings that back); the heavy body lost it to a probe from the floor of the spinal furrow
    (the open-side gate: ``openn``), measured 0.61 there and 0.99-1.0 on the others."""
    fig = _figure(world, "tail", body)
    t, d, before, after = _reports(fig)[0]
    assert d["applied"] and d["guard"] <= 1, (body, d["guard"])
    assert d["kept"] > 0.8, (body, d["kept"])                                # measured 0.85-0.94
    # (the last ring may climb an oblique wall by two reaches, ``CLIMB_MAX``: with one it reached 0.75 of its height)
    assert _back_reach(fig) > (0.55 if body == "heavy" else 0.9), (body, _back_reach(fig))
    assert after["open_p50"] <= 22.0, (body, after["open_p50"])            # measured 7.4-18.8


@assets
def test_the_tail_is_welded_at_level_two_without_a_guard_round(world):
    """Level 2: the first ring of the plug moves in proportion to the local reach (``CAP1_FULL``); pushed on by a fixed
    step it crossed its mirror image in the cleft under the root and the whole weld was refused."""
    fig = _figure(world, "tail", levels=2)
    t, d, before, after = _reports(fig)[0]
    assert d["applied"] and d["guard"] == 0 and d["kept"] > 0.8, d
    assert _back_reach(fig) > 0.9                                            # measured 1.0
    assert after["crossings"] <= after["crossings_before"] and after["plug_outside"] == 0


def _fold_near(monkeypatch, V, point, radius):
    """Make the exact crossing test report a fold wherever a welded triangle has moved within ``radius`` of ``point``
    (a synthetic fold that shrinking the reach cannot clear: only taking that patch out of the weld does)."""
    real = CO.crossing_pairs
    point = np.asarray(point, dtype=np.float64)

    def fake(verts, faces):
        if np.array_equal(verts, V):
            return np.zeros((0, 2), dtype=np.int64)
        moved = np.any(verts[faces] != V[faces], axis=2).any(axis=1)
        bad = np.nonzero(moved & (np.linalg.norm(V[faces].mean(axis=1) - point, axis=1) < radius))[0]
        return np.stack([bad[:-1], bad[1:]], axis=1) if len(bad) > 1 else np.zeros((0, 2), dtype=np.int64)

    monkeypatch.setattr(W.CO, "crossing_pairs", fake)
    return real


def test_a_patch_that_would_fold_is_left_out_and_the_rest_of_the_root_stays_welded(sphere, monkeypatch):
    """The guard's hard rounds (``GUARD_HARD``): a fold beside a tube that shrinking the reach does not clear takes
    that patch out of the weld; the rest of the ring stays welded at its own reach, and the weld says how much of the
    ring it kept.  The first pass gave the whole root up (``GUARD_HARD = 0`` brings that back)."""
    V, F = sphere
    spot = np.array([0.8, 0.0, 5.95])
    real = _fold_near(monkeypatch, V, spot, 0.3)
    d: dict = {}        # (the mechanism, apart from the policy: on this small ring -- 24 vertices -- one patch is
    out, wall, centre = _weld_tube(sphere, profile=W.WeldProfile(kept_min=0.5), diag=d)   # 0.29, under the shipped 0.75)
    monkeypatch.setattr(W.CO, "crossing_pairs", real)
    assert d["applied"] and d["guard"] > W.WeldProfile().guard, d              # the hard rounds ran
    assert 0.6 <= d["kept"] < 0.98, d["kept"]                                  # most of the ring, not all of it (0.71)
    moved = np.any(out != V, axis=1)
    assert moved[V[:, 0] < -0.3].sum() > 50                                    # the far side is welded ...
    assert not moved[np.linalg.norm(V - spot, axis=1) < 0.25].any()            # ... the patch is not
    assert len(real(out, F[W.Topology.of(F, len(V)).faces_of(d["ids"]["region"])])) == 0


def test_a_root_that_could_only_be_welded_part_of_the_way_round_is_left_bare_with_a_warning(sphere, monkeypatch):
    """Under ``KEPT_MIN`` of the ring the weld steps aside: close up, a weld that stops part of the way round reads
    no better than the bare crease and leaves a ragged fringe where it stops (the fan arms on the slim body kept half
    of their ring, the extra legs on the heavy one 0.5-0.7)."""
    V, F = sphere
    real = _fold_near(monkeypatch, V, np.array([0.8, 0.0, 5.95]), 1.0)             # half the root "folds"
    t = W.WeldTarget("a", _tube(), np.array([0.0, 0.0, 6.0]), 0.6, W.WeldProfile(), np.array([[0, 0, 5.0], [0, 0, 9.0]]))
    diags: list = []
    with pytest.warns(K.AppendageWarning, match="of its ring could be welded"):
        out = W.weld_vertices(V, F, [t], diags=diags)
    monkeypatch.setattr(W.CO, "crossing_pairs", real)
    d = diags[0]
    assert not d["applied"] and d["refused"] and 0.0 < d["kept"] < W.KEPT_MIN, d
    assert np.array_equal(out, V)


@assets
def test_the_fan_arms_on_the_slim_body_are_left_bare_and_the_lab_says_so(world):
    """A real root the weld cannot take most of the way round (kept 0.5): bare, with the warning, and the lab's
    final render carries the note.  (When the fan pair's girdle is rebuilt this case should start to weld: move it to
    the welded lists then.)"""
    c, placed, V0, F, targets, V1, diags = _figure(world, "harvest:preset=upper_arms", "slim_f")
    for t, d in zip(targets, diags):
        assert not d["applied"] and d["refused"] and d["kept"] < W.KEPT_MIN, d
    assert np.array_equal(V1, V0)
    from minimaster.character import lab as L
    doc = L.LabDocument(L.Resources.shared())
    doc.character.macro.update(BODIES6["slim_f"])
    doc.add_appendage("harvest", preset="upper_arms")
    final = doc.engine.final_scene(doc.character, levels=1)
    assert any("could not be welded onto the root" in n for n in final.notes), final.notes


@assets
@pytest.mark.parametrize("body", ["neutral", "child"])
def test_a_mirrored_pair_is_welded_alike_on_both_sides(world, body):
    """Each side of a pair keeps to its own half of the body (``half``): without it the side welded first took the
    skin between a pair of legs, and its mirror image got what was left (neutral: 7 against 21 degrees and 1.0
    against 0.86 of the ring; child: welded on the left, bare on the right)."""
    reps = _reports(_figure(world, "harvest:preset=extra_legs", body))
    (tl, dl, bl, al), (tr, dr, br_, ar) = reps
    assert dl["applied"] and dr["applied"], (dl["kept"], dr["kept"])
    assert abs(dl["kept"] - dr["kept"]) < 0.06, (dl["kept"], dr["kept"])
    assert abs(al["open_p50"] - ar["open_p50"]) < 4.0, (al["open_p50"], ar["open_p50"])
    c, placed, V0, F, targets, V1, diags = _figure(world, "harvest:preset=extra_legs", body)
    for t, d in zip(targets, diags):                                  # nothing welded across the midline
        moved = d["ids"]["region"][np.any(V1[d["ids"]["region"]] != V0[d["ids"]["region"]], axis=1)]
        assert np.all(V0[moved, 0] * np.sign(t.centre[0]) > 0.0)


def test_a_cut_to_the_reach_fades_at_a_bounded_slope_and_never_crosses_the_wall(sphere):
    """``_envelope_on``: a vertex whose reach is cut to 0 lowers its neighbours' by at most the slope times the
    distance, leaves the far side alone, and with ``among`` does not act through vertices outside the set."""
    V, F = sphere
    topo = W.Topology.of(F, len(V))
    ids = np.nonzero(V[:, 2] > 5.0)[0]
    x = np.full(len(ids), 1.0)
    seed = int(np.argmax(V[ids, 2]))
    x[seed] = 0.0
    y = W._envelope_on(topo, V, ids, x, 0.5)
    dist = np.linalg.norm(V[ids] - V[ids[seed]], axis=1)
    assert y[seed] == 0.0 and np.all(y <= x + 1e-12)
    assert np.all(y >= np.minimum(1.0, 0.5 * dist) - 1e-9)                   # never below slope * straight distance
    assert np.all(y[dist > 2.4] == 1.0)                                      # the far side keeps its reach
    assert y[(dist > 0) & (dist < 0.5)].max() < 0.5                          # ... and the near one fades
    among = np.ones(len(ids), bool)
    among[seed] = False
    assert np.all(W._envelope_on(topo, V, ids, x, 0.5, among=among)[among] == 1.0)


# --------------------------------------------------------------------------
# round 7: a clean line where the welded skin meets a root of another colour


def test_the_wall_does_not_come_through_the_skin_s_last_strip(sphere):
    """The skin arrives nearly flat, and a faceted wall bulges between the last ring's vertices: with the ring ON the
    wall (``TOP_LIFT = 0``) the wall came through the last strip in a row of teeth, a sawtooth line wherever the root
    is another colour than the skin.  A hair off the wall, every point of the ring's own edges and of the strip
    behind it is outside the wall."""
    V, F = sphere
    d: dict = {}
    out, wall, centre = _weld_tube(sphere, diag=d)
    top = np.zeros(len(V), bool)
    top[d["ids"]["top"]] = True
    cap = np.zeros(len(V), bool)
    cap[d["ids"]["cap"]] = True
    strip = F[top[F].any(axis=1) & ~cap[F].any(axis=1)]
    T = out[strip]
    mids = np.concatenate([(T[:, i] + T[:, (i + 1) % 3]) / 2.0 for i in range(3)] + [T.mean(axis=1)])
    wp = W.Surface(wall.vertices, wall.faces, centre, 4.0)
    _, _, b, _ = wp.closest(mids)
    assert len(strip) > 40 and b.min() > 0.0, b.min()                         # measured +0.004 (ON the wall: -0.004)


@assets
def test_a_horn_s_base_is_drawn_in_the_skin_s_colour_past_the_welded_skin(world):
    """The sheath carries its own albedo: the figure's skin up to where the welded skin ends on it and a little
    more, keratin above (``kinds.HORN_TINT_*``), so the line that shows is a ring of the sheath's own mesh.  A
    one-colour view (another material) and a print ignore it; ``collar=0`` (no weld) and the skin-covered
    ossicone have none."""
    c, placed, V0, F, targets, V1, diags = _figure(world, "horn:shape=ram")
    skin = np.asarray(K.material_named("skin").linear_color())
    for p, t, d in zip(placed, targets, diags):
        mesh = p.mesh()
        cols = mesh.colors_for(p.material)
        assert cols is not None and cols.shape == (len(mesh.vertices), 3)
        assert mesh.colors_for(K.material_named("skin")) is None
        is_skin = np.linalg.norm(cols - skin, axis=1) < 1e-9
        is_horn = np.linalg.norm(cols - np.asarray(p.material.linear_color()), axis=1) < 1e-9
        assert is_skin.sum() > 50 and is_horn.sum() > 2 * is_skin.sum()        # the base only (measured 612 of 3746)
        # every vertex of the sheath beside the welded skin's last ring is skin-coloured
        ring = V1[d["ids"]["top"]]
        near = np.linalg.norm(np.asarray(mesh.vertices)[:, None, :] - ring[None, :, :], axis=2).min(axis=1) < 0.05
        assert near.sum() > 20 and is_skin[near].all()
    base, lib, cat = world
    for text in ("horn:collar=0", "horn:shape=ossicone"):
        ch = Character()
        ch.appendages.append(K.add_from_text(text, ch.appendages, base=base, lib=lib))
        verts = lib.apply(base.verts, ch.effective_target_weights(cat))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pl = ch.instantiate_appendages(base, verts, lib)
        assert getattr(pl[0].mesh(), "colors_for", None) is None, text
