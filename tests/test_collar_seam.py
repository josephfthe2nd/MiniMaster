"""Round 4 of the root collar (``collar.py``): what the round-3 critics found, measured the way they saw it.

* the STICKER: the collar was drawn as a separate shell with its own normals and occlusion, so its footprint was a
  tone patch with a serrated, z-fighting outline (``seam_report``'s ``outline_ratio``: the high-pass of the picture
  on the outline against the skin around it; round 3 measured 2-8 on the tail, the wing and the horns);
* the SAWTOOTH / SCALLOPED contact line on the wall: a sheet tangent to a faceted wall (``contact_lines``'s
  ``wall_wave``, in root radii; round 3: 0.03-0.07 on the horns) and z-fighting along the arm (``wall_jag``);
* FOLDS and SELF-CROSSINGS: hundreds of crossing triangles on the heavy and child tails, the feathered wing, the
  lowered tail (a rim folded back 178 degrees), the horn at full tilt; none is allowed now, on six bodies;
* the DEFAULT LEVEL: the CLI and the Character tab meshed the body at level 0, where there is no collar, so every
  default picture and print still showed the pipe in a flange;
* the side a tail may have, the print report's blindness to a crossing shell, a collar dropped in silence.

docs/collar.md has the numbers before and after.
"""

import json
import subprocess
import sys
import warnings

import numpy as np
import pytest

from minimaster.character import collar as CO
from minimaster.character import junction_checks as JC
from minimaster.character import kinds as K
from minimaster.character import mhbase as mh
from minimaster.character import portrait as po
from minimaster.character import print_report as PR
from minimaster.character import shading as sh
from minimaster.character.sliders import Character, SliderCatalog
from minimaster.core.mesh import Mesh

assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")

BODIES6 = {"neutral": {"gender": 1.0, "muscle": 0.55}, "tall": {"gender": 1.0, "muscle": 0.55, "height": 1.0},
           "short": {"gender": 1.0, "muscle": 0.55, "height": 0.0}, "child": {"age": 0.1875},
           "heavy": {"weight": 1.0, "gender": 1.0, "muscle": 1.0}, "slim_f": {"gender": 0.0, "weight": 0.0}}

# every kind that has a collar, at rest and at the poses and parameters round 3 broke it at
CASES = [("tail", {}), ("tail", {"lift": -60.0}), ("tail", {"lift": 60.0}), ("tail", {"lift": 90.0}),
         ("tail:base_radius=0.85", {}), ("tail:base_radius=1.2", {}), ("tail:collar=2", {}),
         ("wing:style=membrane", {}), ("wing:style=membrane", {"swing": -45.0}), ("wing:style=membrane", {"raise_": 60.0}),
         ("wing:style=feathered", {}), ("wing:style=feathered", {"swing": -45.0}),
         ("horn", {}), ("horn:shape=ram", {}), ("horn:tilt=45", {}),
         ("horn:ring_verts=24", {}), ("horn:base_radius=0.6,collar=2", {})]
# (the horn track's round 4: the ossicone has no cuff, its own foot is the fillet; tilt ends at 45 and ring_verts starts
# at 24, hornrig.TILT_MAX / RING_VERTS_MIN: typing 60 or 6 is an error now)


@pytest.fixture(scope="module")
def world():
    base, lib = mh.load()
    return base, lib, SliderCatalog(lib.names)


def _shaped(world, text, body="neutral", controls=None):
    base, lib, cat = world
    c = Character()
    c.macro.update(BODIES6[body])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp = K.add_from_text(text, c.appendages, base=base, lib=lib)
        if controls:
            sp.controls.update(controls)
        c.appendages.append(sp)
        verts = lib.apply(base.verts, c.effective_target_weights(cat))
        placed = c.instantiate_appendages(base, verts, lib)
        shaped = c.shaped_body(base, verts, placed)
    return c, placed, shaped


def _tail_profile(p):
    """The tail's collar at any radius (the tail kind asks for none above ``kinds.TAIL_COLLAR_MAX_R``)."""
    from minimaster.character import tailrig as T
    R = float(p.params.get("base_radius", 0.45))
    k = float(p.params.get("collar", 1.0)) * min(1.0, T.R_MAX / max(R, 1e-9))
    return CO.CollarProfile(strength=k, up=(0.0, 1.0, 0.0), **K.TAIL_COLLAR)


def _scene(world, text, body, levels, controls=None):
    """Body, appendages and collars as the CLI's picture builds them: ``(shells, collar ids, wall ids, placed)``."""
    base = world[0]
    c, placed, shaped = _shaped(world, text, body, controls)
    shells = po.character_shells(base, shaped, levels=levels, with_eyes=False)
    shells += [po.Shell(p.mesh().transform(po.MH_TO_Z_UP), p.material) for p in placed]
    walls = list(range(1, len(shells)))
    cs = CO.collar_shells(base, shaped, placed, levels=levels, skin=CO.skin_from_shell(shells[0].mesh))
    first = len(shells)
    shells += [po.Shell(m.transform(po.MH_TO_Z_UP), mat) for _, _, m, mat in cs]
    return shells, list(range(first, len(shells))), walls, placed


def _fit(placed, name, r):
    p = [q for q in placed if q.name == name][0]
    return po.MH_TO_Z_UP[:3, :3] @ p.anchor + po.MH_TO_Z_UP[:3, 3], r


# -- the sticker: the collar's outline in the picture ---------------------------------------------------------------------


@assets
@pytest.mark.parametrize("text,body,name,r,views,bound", [
    ("tail", "neutral", "tail-1", 1.6, [(180, 10), (150, 25), (90, 5)], 1.6),
    ("tail", "heavy", "tail-1", 1.8, [(180, 10), (150, 25)], 1.6),
    ("wing", "neutral", "wing-1.l", 1.0, [(180, 10), (215, 15), (145, 15)], 1.6),
    ("horn:shape=ram", "neutral", "horn-1.l", 0.45, [(10, 10), (50, 10)], 1.6),
    # the curved horn's collar is a steep little fillet (0.45 root radii wide, 0.3 high: the horn's profile): its own
    # concave shading reads as a soft ring there (drawn with the skin's normals all over it measures 0.9-1.3)
    ("horn", "neutral", "horn-1.l", 0.35, [(10, 10), (60, 15)], 3.5),
])
def test_the_collars_outline_cannot_be_told_from_the_skin_around_it(world, text, body, name, r, views, bound):
    """The round-3 'sticker': at level 1 the collar's footprint was a lighter shield with a serrated outline (tail,
    wing) or a stair-stepped ring (horn). Measured on the picture: the high-pass on the line where the footprint
    meets the skin against the skin around it. Round 3 (HEAD 835af09, the same views, levels 1-2, neutral / heavy /
    child): tail 2.0-5.7, wing 3.0-8.3, ram 1.7-4.7, curved horn 5.0-7.2. The collar now carries the skin's normals
    and occlusion where it meets the skin."""
    shells, collars, walls, placed = _scene(world, text, body, 1)
    assert collars
    for az, el in views:
        rep = JC.seam_report(shells, collars, 0, walls, size=(320, 320), azimuth=az, elevation=el,
                             fit=_fit(placed, name, r))
        assert rep["outline_px"] > 50, (az, el, rep)
        assert rep["outline_ratio"] < bound, (text, body, az, el, rep)
        assert rep["wall_jag"] < 0.08, (text, body, az, el, rep)


@assets
def test_the_shading_rides_on_the_mesh_through_the_pipelines_own_transform(world):
    """The critics' tools (and the CLI, the lab) build ``Shell(mesh.transform(MH_TO_Z_UP), material)`` themselves:
    the shading the collar carries must survive that and be what the renderer uses."""
    base = world[0]
    c, placed, shaped = _shaped(world, "tail")
    skin = CO.shaped_skin(base, shaped, 1)
    (name, m, mat), = CO.collar_pieces(base, shaped, placed[0], levels=1, skin=skin)
    assert isinstance(m, CO.CollarMesh) and m.shading_normals is not None and m.shading_cavity is not None
    mz = m.transform(po.MH_TO_Z_UP)
    R = po.MH_TO_Z_UP[:3, :3]
    assert isinstance(mz, CO.CollarMesh)
    assert np.allclose(mz.shading_normals, m.shading_normals @ R.T, atol=1e-9)
    # the renderer shades with the carried normals, not its own
    cam = np.zeros((len(mz.vertices), 3)) + np.array([0.0, 0.0, -10.0])
    view = np.eye(4)
    rig = sh.LightRig()
    carried = po._shell_colors(po.Shell(mz, mat), cam, view, rig, 1.0, False)
    plain = po._shell_colors(po.Shell(Mesh(mz.vertices, mz.faces), mat), cam, view, rig, 1.0, False)
    assert np.abs(carried - plain).max() > 0.01
    own = sh.vertex_normals(mz.vertices, mz.faces)
    expect = sh.shade_vertices(mz.shading_normals, -cam / np.linalg.norm(cam, axis=1, keepdims=True), mat, rig,
                               sh.ao_from_term(mz.shading_cavity, 1.0))
    assert np.allclose(carried, expect) and not np.allclose(own, mz.shading_normals)


@assets
def test_both_wings_collars_are_shaded_with_outward_normals_at_the_arm(world):
    """The right wing's arm piece (the wall ``WingKind.collar_wall`` hands over) is wound inside out; the collar's
    top edge, shaded with the wall's normals, was drawn with them pointing into the arm (and HEAD's ``wall_p95``
    read 179.9 degrees on that side). The wall is turned outward first."""
    base = world[0]
    c, placed, shaped = _shaped(world, "wing")
    skin = CO.shaped_skin(base, shaped, 2)
    assert len(placed) == 2
    for p in placed:
        d = {}
        (_, m, _), = CO.collar_pieces(base, shaped, p, levels=2, skin=skin, diag=d)
        keep = d["keep"]
        ids = d["vid_top"][0][keep]                                  # the top edge, on the arm
        axis_pt = d["axis"].at(np.clip(d["s_x"][keep], 0, d["axis"].length))[0]
        radial = CO._unit(d["E"][keep] - axis_pt)
        assert np.median((m.shading_normals[ids] * radial).sum(-1)) > 0.8, p.name
        rep = CO.join_report(base, shaped, p, levels=2, skin=skin)
        assert rep["wall_p95"] < 2.0 and rep["geo_wall_p95"] < 45.0, (p.name, rep)


# -- the contact line on the wall ------------------------------------------------------------------------------------------


@assets
@pytest.mark.parametrize("text", ["horn", "horn:shape=ram", "tail", "wing"])
@pytest.mark.parametrize("levels", [1, 2])
def test_the_contact_line_on_the_wall_is_not_wavy(world, text, levels):
    """Where the collar's surface enters the appendage, as a line along the axis: what a quadratic fit over three
    columns either side leaves of it (scallops, teeth), p95 - p5, in root radii. Round 3 measured 0.036-0.05 on
    the curved horn and up to 0.07 on the ram (a sheet tangent to a 24- or 32-gon within 2-3 degrees)."""
    base = world[0]
    for body in ("neutral", "child"):
        c, placed, shaped = _shaped(world, text, body)
        rep = CO.join_report(base, shaped, placed[0], levels=levels, skin=CO.shaped_skin(base, shaped, levels))
        assert rep["wall_wave"] < 0.03 and rep["wall_contact_missing"] < 0.02, (text, body, levels, rep)


# -- no fold, no crossing --------------------------------------------------------------------------------------------------


def _fold_columns(m, d):
    """Kept columns whose own vertex normal where the sheet meets the skin is more than 90 degrees off the skin's."""
    own = sh.vertex_normals(m.vertices, m.faces)
    ids = d["vid_top"][d["n_vis"] - 1][d["keep"]]
    return int(((own[ids] * d["N"][d["n_vis"] - 1][d["keep"]]).sum(-1) < 0.0).sum())


@assets
@pytest.mark.parametrize("body", sorted(BODIES6))
def test_no_collar_crosses_itself_or_folds_on_any_body_kind_pose_or_extreme(world, body):
    """Round 3: 99-288 crossings on the feathered wing at rest, 265/376 on the child's and the heavy body's tail,
    3550 on a lowered tail (its rim folded back 178 degrees), 369 on a horn at full tilt; the test covered three
    bodies and the membrane wing. Now: zero, every kind with a collar, at rest and at its extremes, six bodies."""
    base = world[0]
    built = 0
    for text, ctl in CASES:
        c, placed, shaped = _shaped(world, text, body, ctl)
        skin = CO.shaped_skin(base, shaped, 1)
        for p in placed[:1]:
            prof = p.kind.collar_profile(p) or (_tail_profile(p) if text.startswith("tail") else None)
            if prof is None:
                continue
            d = {}
            pieces = CO.collar_pieces(base, shaped, p, levels=1, skin=skin, profile=prof, diag=d)
            if not pieces:
                assert d.get("dropped"), (text, ctl, body)              # never in silence
                continue
            m = pieces[0][1]
            built += 1
            assert m.is_watertight and m.volume() > 0, (text, ctl, body)
            assert CO.count_crossings(m.vertices, m.faces) == 0, (text, ctl, body, d.get("guard_log"))
            assert _fold_columns(m, d) == 0, (text, ctl, body)
    assert built >= len(CASES) - 4


@assets
@pytest.mark.parametrize("body", ["neutral", "heavy", "child"])
def test_no_collar_crosses_itself_at_level_two(world, body):
    base = world[0]
    for text, ctl in [("tail", {}), ("tail", {"lift": -60.0}), ("wing:style=feathered", {}), ("wing", {}),
                      ("horn:tilt=45", {}), ("horn:shape=ram", {})]:
        c, placed, shaped = _shaped(world, text, body, ctl)
        pieces = CO.collar_pieces(base, shaped, placed[0], levels=2, skin=CO.shaped_skin(base, shaped, 2))
        for _, m, _ in pieces:
            assert CO.count_crossings(m.vertices, m.faces) == 0, (text, ctl, body)


@assets
def test_a_lowered_tail_on_the_heavy_body_keeps_its_rim_on_the_skin(world):
    """Round 3: lift -60 folded the rim back over itself (rim normal 178 degrees from the skin's) on the heavy body."""
    base = world[0]
    c, placed, shaped = _shaped(world, "tail", "heavy", {"lift": -60.0})
    rep = CO.join_report(base, shaped, placed[0], levels=2, skin=CO.shaped_skin(base, shaped, 2))
    assert rep["fold_columns"] == 0 and rep["geo_rim_max"] < 45.0 and rep["self_crossings"] == 0, rep
    assert rep["lip_max"] < 0.01 and rep["kept_share"] > 0.25, rep


@assets
def test_the_reach_stops_before_the_skin_turns_away_or_the_paths_converge(world):
    """The guard is a net, not the design: the collar stays whole because its reach ends before the skin under it turns
    more than TURN_MAX_DEG from the exit's normal (a lowered tail's collar wrapped round the sacrum) and before
    neighbouring paths over the skin converge (CONVERGE_MIN: a caustic, the rim folds). Without the first, the FIRST
    construction of the lowered tail on the tall body crosses itself 57-117 times (2-5 with it); without the second it
    folds 6 columns; either way the guard then cuts gaps (kept 0.40-0.43 of the ring instead of 0.47-0.57)."""
    base = world[0]
    for text, body, ctl, lv in (("tail", "tall", {"lift": -60.0}, 1), ("tail", "tall", {"lift": -60.0}, 2),
                                ("tail:base_radius=1.2", "neutral", {}, 1)):
        c, placed, shaped = _shaped(world, text, body, ctl)
        p = placed[0]
        prof = p.kind.collar_profile(p) or _tail_profile(p)
        d = {}
        assert CO.collar_pieces(base, shaped, p, levels=lv, skin=CO.shaped_skin(base, shaped, lv), profile=prof, diag=d)
        first = d["guard_log"][0]                                       # before the guard cut anything
        assert first["folds"] == 0, (text, body, lv, d["guard_log"])
        if text == "tail":
            assert first["crossings"] < 20 and np.mean(d["keep"]) > 0.45, (text, body, lv, d["guard_log"])


@assets
def test_the_horn_at_full_tilt_has_no_crack_and_no_teeth(world):
    """Round 3: tilt=60 split the collar into two lobes with a skin-coloured crack across the horn; ring_verts=6 tore it
    into teeth at the hexagon's corners."""
    base = world[0]
    for text in ("horn:tilt=45", "horn:ring_verts=24"):        # (the ranges' new ends: hornrig.TILT_MAX, RING_VERTS_MIN)
        c, placed, shaped = _shaped(world, text, "neutral")
        for p in placed:
            rep = CO.join_report(base, shaped, p, levels=2, skin=CO.shaped_skin(base, shaped, 2))
            # (a hair of the line may miss at the two tapered ends of a run of columns, where the sheet is nil)
            assert rep["wall_contact_missing"] < 0.02 and rep["wall_wave"] < 0.03, (text, p.name, rep)
            assert rep["self_crossings"] == 0 and rep["fold_columns"] == 0, (text, p.name, rep)


@assets
def test_a_collar_that_cannot_be_built_says_so(world, monkeypatch):
    """Round 3: a tail of base_radius 2 on the heavy body got no collar and no word. A kind that asks for a collar the
    construction cannot fit now gets an AppendageWarning with the reason."""
    base = world[0]
    c, placed, shaped = _shaped(world, "tail:base_radius=2.0", "heavy")
    monkeypatch.setattr(K.TailKind, "collar_profile", lambda self, p: _tail_profile(p))
    d = {}
    pieces = CO.collar_pieces(base, shaped, placed[0], levels=1, skin=CO.shaped_skin(base, shaped, 1), diag=d)
    if pieces:
        pytest.skip("this body's 2.0 tail got a collar: nothing dropped to report")
    assert d["dropped"]
    with pytest.warns(K.AppendageWarning, match="no root collar"):
        assert CO.collar_shells(base, shaped, placed, levels=1, skin=CO.shaped_skin(base, shaped, 1)) == []


# -- the default level --------------------------------------------------------------------------------------------------


def test_a_collar_is_wanted_by_the_records_that_will_have_one():
    from minimaster.character.appendage_spec import AppendageSpec
    tail = AppendageSpec(id="tail-1", kind="tail")
    assert CO.wants_collar([tail]) and CO.default_levels([tail]) == CO.MIN_LEVEL == 1
    assert not CO.wants_collar([AppendageSpec(id="tail-1", kind="tail", params={"collar": 0.0})])
    assert not CO.wants_collar([AppendageSpec(id="tail-1", kind="tail", enabled=False)])
    assert not CO.wants_collar([tail], enabled=False) and CO.default_levels([tail], enabled=False) == 0
    # a girdle limb: the collar is ON by default since the limb track's round 4 (kinds.HarvestKind, GIRDLE_COLLAR)
    assert CO.wants_collar([AppendageSpec(id="arm-1", kind="harvest")])
    assert not CO.wants_collar([AppendageSpec(id="arm-1", kind="harvest", params={"collar": 0.0})])
    assert CO.wants_collar([AppendageSpec(id="horn-1", kind="horn")])
    assert CO.wants_collar([AppendageSpec(id="x-1", kind="no-such-kind"), tail])
    assert CO.default_levels([]) == 0


def _cli(tmp_path, *args):
    return subprocess.run([sys.executable, "-m", "minimaster", "character", *args], capture_output=True, text=True)


@assets
def test_the_cli_meshes_a_character_with_a_collar_at_level_one_by_default(tmp_path):
    """Round 3: ``--subdivide`` defaulted to 0, where there is no collar: the default picture and STL of every
    appendage showed the pipe in a flange. Now the default is 1 when a root collar is asked for; ``--subdivide 0``
    is honoured and says what it costs; a character without appendages stays at 0.  (Round 6: the collar is asked
    for with ``--collar-shell``; the default seam is the weld, tests/test_weld.py.)"""
    done = _cli(tmp_path, "--add", "tail", "--collar-shell", "-o", str(tmp_path / "a.stl"))
    assert done.returncode == 0, done.stderr
    assert "subdivided once" in done.stdout and "tail-1           2 shells" in done.stdout
    zero = _cli(tmp_path, "--add", "tail", "--collar-shell", "--subdivide", "0", "-o", str(tmp_path / "b.stl"))
    assert zero.returncode == 0 and "needs a subdivided body" in zero.stdout and "tail-1           1 shell" in zero.stdout
    assert (tmp_path / "a.stl").stat().st_size > 2 * (tmp_path / "b.stl").stat().st_size
    off = _cli(tmp_path, "--add", "tail", "--no-collar", "-o", str(tmp_path / "c.stl"))
    assert "subdivided once" not in off.stdout and (tmp_path / "c.stl").stat().st_size == (tmp_path / "b.stl").stat().st_size
    plain = _cli(tmp_path, "-o", str(tmp_path / "d.png"), "--res", "60", "80")
    assert plain.returncode == 0 and "subdivided" not in plain.stdout


@assets
def test_the_lab_draws_and_exports_a_character_with_a_collar_at_level_one(tmp_path):
    """The Character tab's final preview and export (``Lab(levels=None)``, what character_panel.py builds) mesh
    the body at level 1 when a root collar is asked for, else 0; an explicit level is honoured; the draft is the
    proxy either way and says the collar is the final render's."""
    from minimaster.character import lab as L
    res = L.Resources.shared()
    lb = L.Lab(res, size=(200, 200), threaded=False)
    try:
        assert lb.final_levels() == 0
        lb.doc.add_appendage("tail")
        lb.doc.set_anatomy(collar="shell")                   # round 6: the collar is the document's choice
        assert lb.final_levels() == 1
        scene = lb.engine.final_scene(lb.doc.character, levels=lb.final_levels())
        assert any(isinstance(s.mesh, CO.CollarMesh) for s in scene.shells)
        assert not any("needs a subdivided body" in n for n in scene.notes)
        lb.doc.set_anatomy(collar=False)
        assert lb.final_levels() == 0
        lb.doc.set_anatomy(collar="shell")
        r = lb.export_stl(None, size="medium", force=True)
        assert r.printability.groups["tail-1"].shells == 2            # the collar is in the exported group
        fixed = L.Lab(res, size=(200, 200), threaded=False, levels=0)
        fixed.doc.add_appendage("tail")
        assert fixed.final_levels() == 0
        fixed.close()
    finally:
        lb.close()


# -- the side, the print report ----------------------------------------------------------------------------------------------


@assets
def test_a_tail_cannot_be_moved_to_a_side_in_the_lab():
    from minimaster.character import lab as L
    doc = L.LabDocument(L.Resources.shared())
    doc.add_appendage("tail")
    with pytest.raises(L.LabError, match="no 'left' side"):
        doc.set_side("tail-1", "left")
    assert doc.spec("tail-1").side == "center"
    doc.add_appendage("wing")
    assert doc.set_side("wing-1", "left") and doc.spec("wing-1").side == "left"


@assets
def test_the_print_report_flags_a_collar_that_crosses_itself(world):
    """Every shell is checked alone and the slicer is trusted to union them; a shell that crosses ITSELF is a risk no
    union fixes (round 3: hundreds of crossings, never flagged)."""
    base = world[0]
    c, placed, shaped = _shaped(world, "tail")
    p = placed[0]
    (name, m, _), = CO.collar_pieces(base, shaped, p, levels=1, skin=CO.shaped_skin(base, shaped, 1))
    figure = [("body", po.character_shells(base, shaped, levels=1, with_eyes=False)[0].mesh)]
    clean = PR.appendage_report([[(p.name, p.mesh()), (name, m)]], base, shaped, 1.9, levels=1,
                                figure_shells=figure, body_verts=shaped)
    assert not any(f.code == "self-crossing" for f in clean.findings)
    # push one vertex of the visible sheet through the shell to the other side: a crossing, still closed
    bad = Mesh(m.vertices.copy(), m.faces)
    d = {}
    CO.collar_pieces(base, shaped, p, levels=1, skin=CO.shaped_skin(base, shaped, 1), diag=d)
    j = int(np.nonzero(d["keep"])[0][len(np.nonzero(d["keep"])[0]) // 2])
    v = d["vid_top"][d["n_vis"] // 2][j]
    bad.vertices[v] -= 3.0 * d["sink"][j] * d["N"][d["n_vis"] // 2][j] + 0.05 * d["N"][d["n_vis"] // 2][j]
    assert CO.count_crossings(bad.vertices, bad.faces) > 0 and bad.is_watertight
    rep = PR.appendage_report([[(p.name, p.mesh()), (name, bad)]], base, shaped, 1.9, levels=1,
                              figure_shells=figure, body_verts=shaped)
    flagged = [f for f in rep.findings if f.code == "self-crossing"]
    assert flagged and flagged[0].level == "warn" and "collar" in flagged[0].where


def test_the_fast_crossing_counter_agrees_with_the_generic_broad_phase():
    rng = np.random.default_rng(11)
    for _ in range(6):
        V = rng.uniform(-1, 1, (60, 3))
        F = rng.integers(0, 60, (80, 3))
        F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
        P = V[F]
        pairs = JC.overlapping_pairs(P.min(1), P.max(1), np.arange(len(F)))
        pairs = pairs[pairs[:, 0] < pairs[:, 1]]
        share = (F[pairs[:, 0]][:, :, None] == F[pairs[:, 1]][:, None, :]).any(axis=(1, 2))
        slow = int(JC.pairs_cross(V, F, pairs[~share]).sum())
        assert CO.count_crossings(V, F) == slow
