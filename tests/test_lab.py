"""The tk-free core of the Character tab: ``minimaster.character.lab``.

Everything the tab does is driven here WITHOUT tkinter: the document edits and their
validation, undo / redo with drag coalescing, files, the caches, the draft and final
scenes, picking, the two-tier scheduler (with a fake clock and a worker that can be
cancelled) and the STL export with the printability report.

Pictures are kept small and the wings are bats: the suite is already long.
"""

import json
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from minimaster.character import appendages as A
from minimaster.character import kinds
from minimaster.character import lab as L
from minimaster.character import mhbase as mh
from minimaster.character.portrait import MH_TO_Z_UP
from minimaster.character.sliders import Character

needs_assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")


@pytest.fixture(scope="module")
def res():
    return L.Resources.shared()


@pytest.fixture
def doc(res):
    return L.LabDocument(res)


@pytest.fixture
def lab(res):
    lb = L.Lab(res, size=(320, 300), threaded=False)
    yield lb
    lb.close()


def _json(d):
    return json.dumps(d.character.to_dict(), sort_keys=True)


# -- the module ------------------------------------------------------------------------------


def test_the_core_never_imports_tkinter():
    code = ("import sys; import minimaster.character.lab as lab; "
            "lab.Lab; assert 'tkinter' not in sys.modules, 'tkinter was imported'")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


# -- add / remove / duplicate / rename / enable ---------------------------------------------


@needs_assets
class TestStructure:
    def test_add_writes_the_record_the_cli_writes(self, doc, res):
        spec = doc.add_appendage("wing", params={"style": "feathered", "span": 15})
        cli = kinds.add_from_text("wing:style=feathered,span=15", [], base=res.base, lib=res.lib)
        assert spec.to_dict() == cli.to_dict()
        assert doc.specs == [spec] and doc.selected == "wing-1"

    def test_add_by_preset_and_side(self, doc):
        horn = doc.add_appendage("horn", preset="temples", side="left")
        assert horn.side == "left" and horn.placement["preset"] == "temples"
        leg = doc.add_appendage("harvest", params={"limb": "leg"})
        assert leg.placement["preset"] == "extra_legs"

    @pytest.mark.parametrize("kw, word", [
        (dict(kind="dragonfly"), "unknown kind"),
        (dict(kind="wing", preset="knees"), "no placement preset"),
        (dict(kind="wing", side="top"), "unknown side"),
        (dict(kind="wing", params={"span": 99}), "outside the range"),
        (dict(kind="wing", params={"primaries": 12}), "only applies when style"),
        (dict(kind="tail", face=-1), "not a cage quad"),
    ])
    def test_add_refuses_what_new_spec_refuses(self, doc, kw, word):
        kind = kw.pop("kind")
        before = _json(doc)
        with pytest.raises(L.LabError, match=word):
            doc.add_appendage(kind, **kw)
        assert _json(doc) == before and not doc.can_undo

    def test_ids_are_stable_and_never_renumbered(self, doc):
        a = doc.add_appendage("wing")
        b = doc.add_appendage("wing")
        assert (a.id, b.id) == ("wing-1", "wing-2")
        doc.remove_appendage("wing-1")
        assert [s.id for s in doc.specs] == ["wing-2"]
        assert doc.add_appendage("tail").id == "tail-1"

    def test_remove_selects_a_neighbour_and_unknown_ids_are_refused(self, doc):
        for k in ("wing", "tail", "horn"):
            doc.add_appendage(k)
        doc.select("tail-1")
        doc.remove_appendage("tail-1")
        assert doc.selected == "horn-1"
        with pytest.raises(L.LabError, match="no appendage 'tail-1'"):
            doc.remove_appendage("tail-1")

    def test_duplicate_is_an_independent_copy_with_its_own_id(self, doc):
        w = doc.add_appendage("wing", params={"span": 12})
        doc.set_control("wing-1", "fold", 0.4)
        dup = doc.duplicate_appendage("wing-1")
        assert dup.id == "wing-2" and dup.name == "wing-1 copy" and doc.selected == "wing-2"
        assert dup.params == w.params and dup.controls == w.controls
        assert dup.placement == w.placement and dup.side == w.side
        dup.params["span"] = 20
        dup.controls["*"]["fold"] = 1.0
        dup.placement["anchor"]["face"] = 1
        assert w.params["span"] == 12 and w.controls["*"]["fold"] == 0.4
        assert w.placement["anchor"]["face"] == 10296

    def test_rename_and_enable(self, doc):
        doc.add_appendage("tail")
        assert doc.rename_appendage("tail-1", "  my tail ") and doc.spec("tail-1").name == "my tail"
        assert doc.spec("tail-1").label() == "my tail"
        assert doc.set_enabled("tail-1", False) and not doc.spec("tail-1").enabled
        assert doc.set_enabled("tail-1", False) is False          # no change: no undo entry
        events = []
        doc.listeners.append(events.append)
        doc.rename_appendage("tail-1", "renamed")
        assert [e.scope for e in events] == ["meta"]              # a rename needs no new picture


# -- parameters ------------------------------------------------------------------------------


@needs_assets
class TestParams:
    def test_set_param_stores_the_coerced_value(self, doc):
        doc.add_appendage("wing")
        assert doc.set_param("wing-1", "span", 15) == 15.0
        assert doc.spec("wing-1").params["span"] == 15.0
        assert doc.set_param("wing-1", "propatagium", "false") is False
        assert doc.set_param("wing-1", "style", "dragon") == "dragon"

    @pytest.mark.parametrize("key, value, word", [
        ("span", 99, r"outside the range 3\.\.30"),
        ("span", 1, r"outside the range 3\.\.30"),
        ("span", "wide", "not a number"),
        ("span", float("nan"), "not finite"),
        ("style", "pigeon", "not one of"),
        ("rows", True, "not a number"),
        ("primaries", 12, "only applies when style is feathered"),
        ("wingspan", 12, "no parameter 'wingspan'"),
    ])
    def test_invalid_values_are_refused_and_change_nothing(self, doc, key, value, word):
        doc.add_appendage("wing")
        before, undo = _json(doc), doc.can_undo
        with pytest.raises(L.LabError, match=word):
            doc.set_param("wing-1", key, value)
        assert _json(doc) == before
        assert doc.undo_stack._undo == doc.undo_stack._undo[:1] and doc.can_undo == undo

    def test_clamp_keeps_the_value_in_range(self, doc):
        doc.add_appendage("wing")
        assert doc.set_param("wing-1", "span", 99, clamp=True) == 30.0
        assert doc.set_param("wing-1", "span", -4, clamp=True) == 3.0

    def test_when_follows_the_current_style(self, doc):
        doc.add_appendage("wing", params={"style": "feathered"})
        assert doc.set_param("wing-1", "primaries", 12) == 12
        names = {p.key for p, _ in doc.param_specs("wing-1")}
        assert "primaries" in names and "billow" not in names
        with pytest.raises(L.LabError, match="only applies when style is membrane or dragon"):
            doc.set_param("wing-1", "billow", 0.1)
        doc.set_param("wing-1", "style", "membrane")              # parameters of the old style go
        assert "primaries" not in doc.spec("wing-1").params
        assert doc.set_param("wing-1", "billow", 0.1) == 0.1
        assert {p.key for p, _ in doc.param_specs("wing-1")} >= {"billow", "span", "style"}

    def test_param_specs_carry_the_ranges_and_current_values(self, doc):
        doc.add_appendage("tail", params={"length": 8})
        rows = {p.key: (p, v) for p, v in doc.param_specs("tail-1")}
        assert rows["length"][1] == 8 and (rows["length"][0].lo, rows["length"][0].hi) == (1.0, 20.0)
        assert rows["curl"][1] == 0.55                            # a default shows as itself

    def test_unknown_kinds_are_kept_but_cannot_be_edited(self, res):
        ch = Character.from_dict({"format": "minimaster-character", "version": 2,
                                  "appendages": [{"id": "gill-1", "kind": "gill", "side": "both"}]})
        d = L.LabDocument(res, ch)
        with pytest.raises(L.LabError, match="does not know"):
            d.set_param("gill-1", "span", 3)
        assert d.character.to_dict()["appendages"][0]["kind"] == "gill"


# -- pose controls ---------------------------------------------------------------------------


@needs_assets
class TestControls:
    def test_range_is_the_placed_wings_not_the_free_wings(self, doc):
        doc.add_appendage("wing")
        placed = doc.engine.placed(doc.character, "wing-1")
        free = kinds.KINDS["wing"].control_range("raise_", {})
        live = doc.control_range("wing-1", "raise_")
        assert live == placed[0].control_range("raise_", {}) and live != free

    def test_range_shrinks_as_the_wing_folds(self, doc):
        doc.add_appendage("wing")
        unfolded = doc.control_range("wing-1", "raise_")
        doc.set_control("wing-1", "fold", 0.8)
        folded = doc.control_range("wing-1", "raise_")
        assert folded != unfolded and folded[1] - folded[0] < unfolded[1] - unfolded[0]

    def test_value_outside_the_live_range_is_refused_or_clamped(self, doc):
        doc.add_appendage("wing")
        lo, hi = doc.control_range("wing-1", "raise_")
        with pytest.raises(L.LabError, match="outside the range"):
            doc.set_control("wing-1", "raise_", hi + 5)
        assert "controls" not in doc.spec("wing-1").to_dict() or not doc.spec("wing-1").controls
        assert doc.set_control("wing-1", "raise_", hi + 5, clamp=True) == pytest.approx(hi)
        assert doc.set_control("wing-1", "raise_", lo - 5, clamp=True) == pytest.approx(lo)
        assert doc.set_control("wing-1", "raise_", 0.5 * (lo + hi)) == pytest.approx(0.5 * (lo + hi))

    def test_per_side_and_shared_controls_land_where_the_cli_puts_them(self, doc):
        doc.add_appendage("wing")
        doc.set_control("wing-1", "fold", 0.5)
        doc.set_control("wing-1", "raise_", 10, side="r")
        assert doc.spec("wing-1").controls == {"*": {"fold": 0.5}, "r": {"raise_": 10.0}}
        spec, key, vals = kinds.pose_from_text("wing-1.r:raise_=10", doc.specs)
        cli = kinds.add_from_text("wing", [], base=doc.resources.base, lib=doc.resources.lib)
        kinds.apply_pose(cli, "", {"fold": 0.5})
        kinds.apply_pose(cli, key, vals)
        assert cli.controls == doc.spec("wing-1").controls

    def test_pose_preset_is_validated_and_clearable(self, doc):
        doc.add_appendage("wing")
        assert doc.set_control("wing-1", "preset", "folded") == "folded"
        with pytest.raises(L.LabError, match="unknown pose preset"):
            doc.set_control("wing-1", "preset", "hover")
        doc.set_control("wing-1", "preset", None)
        assert doc.spec("wing-1").controls == {}

    @pytest.mark.parametrize("args, word", [
        (("horn-1", "fold", 0.5), "no pose controls"),
        (("wing-1", "flap", 0.5), "no control 'flap'"),
        (("wing-1", "fold", "x"), "not a finite number"),
        (("wing-1", "fold", True), "not a finite number"),
        (("wing-1", "fold", 0.5, "x"), "side must be"),
    ])
    def test_refusals(self, doc, args, word):
        doc.add_appendage("wing")
        doc.add_appendage("horn")
        with pytest.raises(L.LabError, match=word):
            doc.set_control(*args)

    def test_a_control_edit_is_a_pose_change_of_one_appendage(self, doc):
        doc.add_appendage("wing")
        events = []
        doc.listeners.append(events.append)
        doc.set_control("wing-1", "fold", 0.3)
        assert [(e.scope, e.spec_id) for e in events] == [("pose", "wing-1")]


# -- placement -------------------------------------------------------------------------------


@needs_assets
class TestPlacement:
    def test_numeric_fields_are_validated_against_the_kind(self, doc):
        doc.add_appendage("wing")
        doc.add_appendage("harvest")
        assert doc.set_placement("wing-1", "scale", 1.5) == 1.5
        assert doc.spec("wing-1").placement["scale"] == 1.5
        with pytest.raises(L.LabError, match="outside the range"):
            doc.set_placement("wing-1", "scale", 9)
        assert doc.set_placement("wing-1", "roll", 999, clamp=True) == 180.0
        with pytest.raises(L.LabError, match="does not use 'roll'"):
            doc.set_placement("arm-1", "roll", 10.0)           # a harvested limb ignores it
        with pytest.raises(L.LabError, match="not one of"):
            doc.set_placement("wing-1", "up_hint", 1)

    def test_side(self, doc):
        doc.add_appendage("wing")
        assert doc.set_side("wing-1", "right") and doc.spec("wing-1").side == "right"
        with pytest.raises(L.LabError, match="unknown side"):
            doc.set_side("wing-1", "up")

    def test_preset_resets_the_anchor_and_keeps_scale_and_side(self, doc):
        doc.add_appendage("harvest", params={"limb": "arm"})
        doc.set_anchor("arm-1", 5)
        doc.set_side("arm-1", "left")
        doc.set_placement_preset("arm-1", "hip_socket")
        pl = doc.spec("arm-1").placement
        assert pl["preset"] == "hip_socket" and pl["anchor"]["face"] == kinds.PRESET_FACES["hip_socket"]
        assert doc.spec("arm-1").side == "left"
        with pytest.raises(L.LabError, match="no placement preset"):
            doc.set_placement_preset("arm-1", "temples")

    def test_set_anchor_writes_the_quad_corners_like_new_spec(self, doc, res):
        doc.add_appendage("tail")
        assert doc.set_anchor("tail-1", 3000) == 3000
        anchor = doc.spec("tail-1").placement["anchor"]
        assert anchor == {"face": 3000, "corners": [int(c) for c in res.base.body_cage(None)[1][3000]]}
        doc.set_anchor("tail-1", 3000, [1, 1, 0, 0])
        assert doc.spec("tail-1").placement["anchor"]["weights"] == [0.5, 0.5, 0.0, 0.0]
        for bad, word in ((10 ** 6, "not a cage quad"), (True, "not a cage quad"),
                          ((3000, [1, 2]), "one non-negative number per quad corner"),
                          ((3000, [0, 0, 0, 0]), "must not all be zero")):
            face, w = bad if isinstance(bad, tuple) else (bad, None)
            with pytest.raises(L.LabError, match=word):
                doc.set_anchor("tail-1", face, w)

    def test_the_stored_quad_is_the_left_hand_one(self, doc, res):
        doc.add_appendage("wing")
        cage_v, cage_q = res.base.body_cage(None)
        mf = A.mirror_face_map(res.base)
        face = 10296                                           # the shoulder blade, x > 0
        assert cage_v[cage_q[face]].mean(axis=0)[0] > 0
        right = int(mf[face])
        w = [0.4, 0.3, 0.2, 0.1]
        stored = doc.set_anchor("wing-1", right, w, mirror=True)
        assert stored == face                                  # the mirror of the right quad
        a = doc.spec("wing-1").placement["anchor"]
        assert a["corners"] == [int(c) for c in cage_q[face]]
        back = A.mirror_anchor_weights(res.base, cage_q, face, a["weights"])
        assert back == pytest.approx(tuple(w), abs=1e-6)       # weights followed the vertices

    def test_place_at_a_pick_on_the_right_flank_stores_the_mirrored_quad(self, doc, res):
        doc.add_appendage("tail")
        cage_v, cage_q = res.base.body_cage(None)
        mf = A.mirror_face_map(res.base)
        q = int(mf[7954])                                      # right shoulder socket
        assert cage_v[cage_q[q]].mean(axis=0)[0] < 0
        pick = L.Pick("body", quad=q, point=cage_v[cage_q[q]].mean(axis=0),
                      weights=(0.25, 0.25, 0.25, 0.25))
        assert doc.place_at_pick("tail-1", pick) == 7954       # normalised to the left hand
        assert doc.spec("tail-1").side == "center"             # a midline tail stays a tail
        with pytest.raises(L.LabError, match="click the body"):
            doc.place_at_pick("tail-1", L.Pick("appendage", spec_id="x"))

    def test_a_single_horn_follows_the_side_that_was_clicked(self, doc, res):
        doc.add_appendage("horn", side="left")
        cage_v, cage_q = res.base.body_cage(None)
        q = int(A.mirror_face_map(res.base)[12116])            # right temple
        pick = L.Pick("body", quad=q, point=cage_v[cage_q[q]].mean(axis=0))
        doc.place_at_pick("horn-1", pick)
        assert doc.spec("horn-1").side == "right"
        assert doc.spec("horn-1").placement["anchor"]["face"] == 12116

    def test_one_click_is_one_undo_step_even_when_it_changes_the_side_too(self, doc, res):
        doc.add_appendage("horn", side="left")
        cage_v, cage_q = res.base.body_cage(None)
        q = int(A.mirror_face_map(res.base)[7954])             # the right shoulder: a new anchor too
        before = _json(doc)
        steps = len(doc.undo_stack._undo)
        doc.place_at_pick("horn-1", L.Pick("body", quad=q, point=cage_v[cage_q[q]].mean(axis=0)))
        assert doc.spec("horn-1").side == "right" and _json(doc) != before
        assert doc.spec("horn-1").placement["anchor"]["face"] == 7954       # (stored left-handed)
        assert len(doc.undo_stack._undo) == steps + 1
        doc.undo()
        assert _json(doc) == before                            # the anchor AND the side came back

    def test_a_gesture_context_joins_an_open_gesture(self, doc):
        doc.add_appendage("tail")
        steps = len(doc.undo_stack._undo)
        doc.begin_gesture()
        with doc.gesture():
            doc.set_param("tail-1", "length", 7)
        doc.set_param("tail-1", "curl", 1.0)                   # still inside the outer drag
        doc.end_gesture()
        assert len(doc.undo_stack._undo) == steps + 1

    def test_a_placed_right_wing_really_lands_where_it_was_placed(self, doc, res):
        """The point of the mirroring rule: the built wing's anchor is on the clicked quad."""
        doc.add_appendage("tail", side="right")
        cage_v, cage_q = res.base.body_cage(None)
        q = int(A.mirror_face_map(res.base)[10647])             # the right hip
        centre = cage_v[cage_q[q]].mean(axis=0)
        doc.place_at_pick("tail-1", L.Pick("body", quad=q, point=centre))
        placed = doc.engine.placed(doc.character, "tail-1")
        assert [p.side for p in placed] == ["r"]
        assert np.linalg.norm(placed[0].anchor - centre) < 0.6   # the quad is ~0.3 across


# -- the body --------------------------------------------------------------------------------


@needs_assets
class TestBody:
    def test_macro_and_sliders_are_validated(self, doc):
        assert doc.set_macro("height", 0.7) == 0.7 and doc.character.macro["height"] == 0.7
        with pytest.raises(L.LabError, match="unknown macro slider"):
            doc.set_macro("tallness", 0.5)
        with pytest.raises(L.LabError, match="outside the range 0..1"):
            doc.set_macro("age", 1.5)
        assert doc.set_macro("age", 1.5, clamp=True) == 1.0
        assert doc.set_slider("torso.torso_vshape", 0.4) == 0.4
        assert doc.character.sliders == {"torso.torso_vshape": 0.4}
        doc.set_slider("torso.torso_vshape", 0.0)
        assert doc.character.sliders == {}                    # zero is the absence of the slider
        with pytest.raises(L.LabError, match="unknown slider"):
            doc.set_slider("torso.nope", 0.1)
        with pytest.raises(L.LabError, match="outside the range -1..1"):
            doc.set_slider("torso.torso_vshape", 2)

    def test_body_edits_say_body(self, doc):
        events = []
        doc.listeners.append(events.append)
        doc.set_macro("weight", 0.6)
        doc.set_slider("torso.torso_vshape", 0.2)
        doc.set_anatomy(enabled=False)
        assert {e.scope for e in events} == {"body"}

    def test_the_adaptation_toggle_changes_the_body_and_lists_notes(self, doc):
        doc.add_appendage("wing")
        on = doc.engine.body_state(doc.character)
        assert doc.anatomy_notes() and "wing" in doc.anatomy_notes()[0]
        doc.set_anatomy(enabled=False)
        off = doc.engine.body_state(doc.character)
        assert off.key != on.key and doc.anatomy_notes() == []
        doc.set_anatomy(enabled=True)
        again = doc.engine.body_state(doc.character)
        assert again.key == on.key and again.verts is on.verts   # removing / restoring is exact
        assert np.array_equal(again.verts, on.verts)

    def test_reset_body_keeps_the_appendages(self, doc):
        doc.add_appendage("tail")
        doc.set_macro("height", 0.9)
        doc.set_slider("torso.torso_vshape", 0.4)
        assert doc.reset_body()
        assert doc.character.macro == Character().macro and doc.character.sliders == {}
        assert [s.id for s in doc.specs] == ["tail-1"]

    def test_the_engines_weights_equal_the_documents(self, doc, res):
        doc.add_appendage("wing")
        doc.set_macro("gender", 0.8)
        doc.set_slider("torso.torso_vshape", 0.4)
        expect = res.lib.apply(res.base.verts, doc.character.effective_target_weights(res.catalog))
        assert np.array_equal(doc.engine.body_state(doc.character).verts, expect)


# -- undo / redo -----------------------------------------------------------------------------


@needs_assets
class TestUndo:
    def test_add_undo_redo(self, doc):
        doc.add_appendage("wing")
        doc.add_appendage("tail")
        assert doc.undo() and [s.id for s in doc.specs] == ["wing-1"]
        assert doc.undo() and doc.specs == []
        assert not doc.undo()
        assert doc.redo() and doc.redo() and [s.id for s in doc.specs] == ["wing-1", "tail-1"]
        assert not doc.redo()

    def test_undo_clears_a_selection_that_no_longer_exists(self, doc):
        doc.add_appendage("tail")
        assert doc.selected == "tail-1"
        doc.undo()
        assert doc.selected is None
        doc.redo()
        doc.select("tail-1")
        assert doc.selected == "tail-1"

    def test_a_new_edit_clears_the_redo_stack(self, doc):
        doc.add_appendage("wing")
        doc.undo()
        doc.add_appendage("tail")
        assert not doc.can_redo

    def test_ten_edits_of_one_control_are_one_undo_step(self, doc):
        doc.add_appendage("tail")
        steps = len(doc.undo_stack._undo)
        for i in range(10):
            doc.set_param("tail-1", "length", 6.0 + 0.5 * (i + 1))
        assert len(doc.undo_stack._undo) == steps + 1
        doc.undo()
        assert doc.spec("tail-1").params.get("length", 6.0) == 6.0     # back to before the drag
        assert doc.redo() and doc.spec("tail-1").params["length"] == 11.0

    def test_different_controls_are_different_steps(self, doc):
        doc.add_appendage("tail")
        steps = len(doc.undo_stack._undo)
        doc.set_param("tail-1", "length", 8)
        doc.set_param("tail-1", "curl", 1.0)
        doc.set_param("tail-1", "length", 9)
        assert len(doc.undo_stack._undo) == steps + 3

    def test_a_pause_ends_the_drag(self, res):
        now = [100.0]
        d = L.LabDocument(res, clock=lambda: now[0])
        d.add_appendage("tail")
        steps = len(d.undo_stack._undo)
        d.set_param("tail-1", "length", 7)
        now[0] += 0.3
        d.set_param("tail-1", "length", 8)                      # within the window: same step
        assert len(d.undo_stack._undo) == steps + 1
        now[0] += 5.0
        d.set_param("tail-1", "length", 9)                      # long after: a new step
        assert len(d.undo_stack._undo) == steps + 2

    def test_gestures_make_one_step_however_many_edits(self, doc):
        doc.add_appendage("wing")
        steps = len(doc.undo_stack._undo)
        doc.begin_gesture()
        for v in (0.1, 0.2, 0.3):
            doc.set_control("wing-1", "fold", v)
        for v in (10, 12):                                      # even a DIFFERENT control
            doc.set_control("wing-1", "swing", v)
        doc.end_gesture()
        assert len(doc.undo_stack._undo) == steps + 1
        doc.undo()
        assert doc.spec("wing-1").controls == {}
        doc.begin_gesture()
        doc.set_control("wing-1", "fold", 0.5)
        doc.end_gesture()
        doc.begin_gesture()
        doc.set_control("wing-1", "fold", 0.6)
        doc.end_gesture()
        assert len(doc.undo_stack._undo) == steps + 2           # two gestures, two steps

    def test_a_gesture_that_changed_nothing_leaves_no_step(self, doc):
        doc.add_appendage("wing")
        steps = len(doc.undo_stack._undo)
        doc.begin_gesture()
        doc.end_gesture()
        assert len(doc.undo_stack._undo) == steps

    def test_undo_during_a_gesture_closes_it_first(self, doc):
        doc.add_appendage("wing")
        doc.begin_gesture()
        doc.set_control("wing-1", "fold", 0.5)
        assert doc.undo()
        assert doc.spec("wing-1").controls == {}

    def test_a_failed_edit_leaves_no_step_and_no_change(self, doc):
        doc.add_appendage("wing")
        steps, before = len(doc.undo_stack._undo), _json(doc)
        with pytest.raises(L.LabError):
            doc.set_param("wing-1", "span", 500)
        assert len(doc.undo_stack._undo) == steps and _json(doc) == before

    def test_a_noop_is_not_an_edit(self, doc):
        doc.add_appendage("tail")
        steps = len(doc.undo_stack._undo)
        events = []
        doc.listeners.append(events.append)
        doc.set_enabled("tail-1", True)
        doc.set_side("tail-1", "center")
        assert len(doc.undo_stack._undo) == steps and events == []

    def test_every_edit_kind_round_trips_through_undo(self, doc):
        doc.add_appendage("wing")
        doc.add_appendage("harvest")
        edits = [
            lambda: doc.set_param("wing-1", "span", 17),
            lambda: doc.set_control("wing-1", "fold", 0.5),
            lambda: doc.set_placement("wing-1", "scale", 1.3),
            lambda: doc.set_side("wing-1", "left"),
            lambda: doc.set_placement_preset("arm-1", "hip_socket"),
            lambda: doc.set_anchor("wing-1", 123),
            lambda: doc.set_macro("height", 0.8),
            lambda: doc.set_slider("torso.torso_vshape", 0.3),
            lambda: doc.set_anatomy(cap=0.5),
            lambda: doc.rename_appendage("wing-1", "big"),
            lambda: doc.set_enabled("wing-1", False),
            lambda: doc.duplicate_appendage("wing-1"),
            lambda: doc.remove_appendage("arm-1"),
        ]
        snaps = [_json(doc)]
        for edit in edits:
            doc.break_coalescing()
            edit()
            snaps.append(_json(doc))
        assert len(set(snaps)) == len(snaps)
        for want in reversed(snaps[:-1]):
            assert doc.undo() and _json(doc) == want
        for want in snaps[1:]:
            assert doc.redo() and _json(doc) == want


# -- files -----------------------------------------------------------------------------------


@needs_assets
class TestFiles:
    def test_save_load_round_trip(self, doc, res, tmp_path):
        doc.add_appendage("wing", params={"style": "dragon"})
        doc.add_appendage("tail")
        doc.set_control("wing-1", "fold", 0.5)
        doc.set_macro("height", 0.7)
        doc.set_anatomy(cap=0.8)
        path = doc.save(tmp_path / "c")                         # no extension: .mmchar
        assert path.suffix == ".mmchar" and path.exists() and not doc.dirty
        fresh = L.LabDocument(res)
        fresh.load(path)
        assert _json(fresh) == _json(doc) and fresh.path == path and not fresh.can_undo
        # the body is unchanged by saving and loading (the adaptation is derived)
        a = doc.engine.body_state(doc.character)
        b = fresh.engine.body_state(fresh.character)
        assert a.key == b.key

    def test_dirty_follows_edits_and_saves(self, doc, tmp_path):
        assert not doc.dirty
        doc.add_appendage("tail")
        assert doc.dirty
        doc.save(tmp_path / "a.mmchar")
        assert not doc.dirty
        doc.undo()
        assert doc.dirty

    def test_save_without_a_path_is_an_error_then_saves_in_place(self, doc, tmp_path):
        with pytest.raises(L.LabError, match="no file yet"):
            doc.save()
        doc.save(tmp_path / "a.mmchar")
        doc.add_appendage("tail")
        assert doc.save() == tmp_path / "a.mmchar"
        assert len(json.loads((tmp_path / "a.mmchar").read_text())["appendages"]) == 1

    def test_unknown_kinds_survive_a_load_edit_save(self, doc, res, tmp_path):
        src = tmp_path / "future.mmchar"
        src.write_text(json.dumps({
            "format": "minimaster-character", "version": 2, "name": "x", "future_key": [1, 2],
            "appendages": [{"id": "gill-1", "kind": "gill", "side": "both", "params": {"n": 3}},
                           {"id": "tail-1", "kind": "tail", "side": "center"}]}))
        doc.load(src)
        doc.set_param("tail-1", "length", 7)
        out = doc.save(tmp_path / "out.mmchar")
        d = json.loads(out.read_text())
        assert d["future_key"] == [1, 2]
        assert d["appendages"][0] == {"id": "gill-1", "kind": "gill", "side": "both",
                                      "params": {"n": 3}, "controls": {}, "placement": {},
                                      "enabled": True}

    @pytest.mark.parametrize("content", ["not json", "[1, 2]", '{"appendages": 4}'])
    def test_bad_files_are_lab_errors_not_tracebacks(self, doc, tmp_path, content):
        (tmp_path / "bad.mmchar").write_text(content)
        doc.add_appendage("tail")
        with pytest.raises(L.LabError, match="cannot read"):
            doc.load(tmp_path / "bad.mmchar")
        assert [s.id for s in doc.specs] == ["tail-1"]          # the open document is untouched
        with pytest.raises(L.LabError, match="no such file"):
            doc.load(tmp_path / "missing.mmchar")

    def test_load_and_new_reset_history_and_selection(self, doc, tmp_path):
        doc.add_appendage("tail")
        doc.save(tmp_path / "a.mmchar")
        doc.add_appendage("wing")
        doc.load(tmp_path / "a.mmchar")
        assert not doc.can_undo and doc.selected is None and [s.id for s in doc.specs] == ["tail-1"]
        doc.new()
        assert doc.specs == [] and doc.path is None and not doc.dirty


# -- the engine: caches ----------------------------------------------------------------------


@needs_assets
class TestEngineCaches:
    def test_a_pose_edit_rebuilds_nothing_and_leaves_the_other_appendage_alone(self, res):
        d = L.LabDocument(res)
        d.add_appendage("wing")
        d.add_appendage("tail")
        eng = d.engine
        eng.draft_scene(d.character)
        tail_before = eng.placed(d.character, "tail-1")
        wing_before = eng.placed(d.character, "wing-1")
        counts = dict(eng.stats)
        d.set_control("wing-1", "fold", 0.4)
        eng.draft_scene(d.character)
        assert eng.stats["morph"] == counts["morph"]
        assert eng.stats["build_draft"] == counts["build_draft"]       # nothing rebuilt
        assert eng.stats["pose"] > counts["pose"]                      # the wing was re-posed
        assert eng.placed(d.character, "tail-1") is tail_before        # the SAME objects
        assert eng.placed(d.character, "wing-1") is wing_before

    def test_a_parameter_edit_rebuilds_only_that_appendage_and_the_body_if_it_adapts(self, res):
        d = L.LabDocument(res)
        d.add_appendage("tail")
        d.add_appendage("horn")
        eng = d.engine
        d.set_anatomy(enabled=False)           # so the edit cannot change the body
        eng.draft_scene(d.character)
        horn = eng.placed(d.character, "horn-1")
        n = eng.stats["build_draft"]
        d.set_param("tail-1", "length", 9.0)
        eng.draft_scene(d.character)
        assert eng.stats["build_draft"] == n + 1                       # the tail only
        assert eng.placed(d.character, "horn-1") is horn

    def test_orbiting_costs_no_builds_and_no_poses(self, res):
        d = L.LabDocument(res)
        d.add_appendage("wing")
        eng = d.engine
        view = L.ViewState()
        eng.render_draft(d.character, view, (200, 200), scale=2)
        counts = dict(eng.stats)
        for az in range(0, 90, 15):
            view.azimuth = az
            eng.render_draft(d.character, view, (200, 200), scale=2)
        assert dict(eng.stats) == counts

    def test_the_morph_is_cached_by_the_effective_weights(self, res):
        d = L.LabDocument(res)
        eng = d.engine
        a = eng.body_state(d.character)
        d.set_macro("height", 0.6)
        b = eng.body_state(d.character)
        d.set_macro("height", 0.5)
        c = eng.body_state(d.character)
        assert a.key != b.key and c.key == a.key and c.verts is a.verts
        assert eng.stats["morph"] == 2

    def test_a_disabled_or_unbuildable_spec_costs_nothing_and_says_why(self, res):
        ch = Character.from_dict({"format": "minimaster-character", "version": 2, "appendages": [
            {"id": "gill-1", "kind": "gill", "side": "both"},
            {"id": "wing-1", "kind": "wing", "side": "both", "params": {"style": "pigeon"}},
            {"id": "tail-1", "kind": "tail", "side": "center", "enabled": False}]})
        d = L.LabDocument(res, ch)
        scene = d.engine.draft_scene(d.character)
        assert [o[0] for o in scene.owners] == [None]               # only the body
        assert any("unknown kind 'gill'" in n for n in scene.notes)
        assert any("wing-1" in n and "skipped" in n for n in scene.notes)

    def test_a_duplicate_id_in_a_loaded_file_is_skipped_and_named(self, res):
        ch = Character.from_dict({"format": "minimaster-character", "version": 2, "appendages": [
            {"id": "tail-1", "kind": "tail", "side": "center"},
            {"id": "tail-1", "kind": "tail", "side": "center"}]})
        d = L.LabDocument(res, ch)
        for scene in (d.engine.draft_scene(d.character), d.engine.final_scene(d.character)):
            assert [o[0] for o in scene.owners if o[0]] == ["tail-1"]
            assert any("duplicated" in n for n in scene.notes)

    def test_draft_scene_is_a_proxy_and_the_owner_table_matches_the_shells(self, res):
        d = L.LabDocument(res)
        d.add_appendage("wing")
        d.add_appendage("tail")
        eng = d.engine
        draft = eng.draft_scene(d.character)
        final = eng.final_scene(d.character)
        assert [o for o in draft.owners] == [(None, ""), ("wing-1", "l"), ("wing-1", "r"),
                                             ("tail-1", "")]
        assert final.owners[-3:] == draft.owners[1:] and final.owners[0] == (None, "")
        assert len(draft.shells) == len(draft.owners)
        n_draft = sum(len(s.mesh.faces) for s in draft.shells)
        n_final = sum(len(s.mesh.faces) for s in final.shells)
        assert n_draft < 0.3 * n_final and len(draft.shells[0].mesh.faces) < 6000
        assert d.engine.stats["final_scene"] == 1


@needs_assets
class TestReplace:
    """A body drag does not rebuild the wings: it moves the last draft by how far the anchor moved."""

    def test_approx_draft_moves_wings_with_the_body_and_builds_nothing(self, res):
        d = L.LabDocument(res)
        d.add_appendage("wing")
        eng = d.engine
        first = eng.draft_scene(d.character)
        n = eng.stats["build_draft"]
        d.set_macro("height", 0.8)
        second = eng.draft_scene(d.character, mode="body")
        assert eng.stats["build_draft"] == n and eng.stats["replace"] == 1
        # the wing followed the shoulder up by about the body's growth
        up = lambda s: s.shells[1].mesh.vertices[:, 2].mean()      # noqa: E731
        assert up(second) - up(first) > 0.3
        grow = second.shells[0].mesh.vertices[:, 2].max() - first.shells[0].mesh.vertices[:, 2].max()
        assert up(second) - up(first) == pytest.approx(grow, abs=0.8 * grow)

    def test_the_shift_is_the_displacement_of_the_anchor(self, res):
        d = L.LabDocument(res)
        d.add_appendage("tail")
        eng = d.engine
        eng.draft_scene(d.character)
        old_state, built = eng._last_draft[L.spec_key(d.spec("tail-1"))]
        d.set_macro("height", 0.8)
        new_state = eng.body_state(d.character)
        shift = eng._anchor_shift(d.spec("tail-1"), old_state, new_state, built)
        exact = eng.built(d.spec("tail-1"), new_state, True)
        want = MH_TO_Z_UP[:3, :3] @ (exact.placed[0].anchor - built.placed[0].anchor)
        assert shift[0] == pytest.approx(want, abs=1e-9)

    def test_without_approx_the_exact_build_is_used(self, res):
        d = L.LabDocument(res)
        d.add_appendage("wing")
        eng = d.engine
        eng.draft_scene(d.character)
        n = eng.stats["build_draft"]
        d.set_macro("height", 0.8)
        eng.draft_scene(d.character)
        assert eng.stats["build_draft"] == n + 1 and eng.stats["replace"] == 0


@needs_assets
class TestHalfBuild:
    """A parameter drag of a paired appendage builds ONE side and mirrors it in the draft."""

    def _drag_scene(self, d, edited="wing-1"):
        return d.engine.draft_scene(d.character, mode="shape", edited=edited)

    def test_the_draft_builds_the_left_wing_only_and_the_scene_has_both_sides(self, res):
        d = L.LabDocument(res)
        d.add_appendage("wing")
        eng = d.engine
        d.set_param("wing-1", "span", 15.0)
        scene = self._drag_scene(d)
        assert eng.stats["build_half"] == 1 and eng.stats["build_draft"] == 0
        assert scene.owners == [(None, ""), ("wing-1", "l"), ("wing-1", "r")]
        left, right = scene.shells[1].mesh, scene.shells[2].mesh
        assert len(left.faces) == len(right.faces)
        assert np.array_equal(right.vertices, left.vertices * [-1, 1, 1])
        assert np.array_equal(right.faces, left.faces[:, ::-1])         # winding flipped with it

    def test_the_mirrored_side_is_where_the_exact_right_wing_is(self, res):
        d = L.LabDocument(res)
        d.add_appendage("wing", params={"span": 15.0})
        half = self._drag_scene(d)
        exact = d.engine.draft_scene(d.character)
        for i in (1, 2):
            a = half.shells[i].mesh.vertices
            b = exact.shells[i].mesh.vertices
            assert np.abs(a.min(axis=0) - b.min(axis=0)).max() < 0.35
            assert np.abs(a.max(axis=0) - b.max(axis=0)).max() < 0.35

    def test_an_exact_draft_in_the_cache_is_preferred(self, res):
        d = L.LabDocument(res)
        d.add_appendage("wing")
        d.engine.draft_scene(d.character)
        n = dict(d.engine.stats)
        self._drag_scene(d)                           # same spec again: nothing to build
        assert d.engine.stats["build_half"] == 0 and d.engine.stats["build_draft"] == n["build_draft"]

    def test_only_the_edited_paired_appendage_takes_the_shortcut(self, res):
        d = L.LabDocument(res)
        d.add_appendage("wing")
        d.add_appendage("tail")
        d.set_param("tail-1", "length", 8.0)
        self._drag_scene(d, "tail-1")                 # a tail is not paired: built exactly
        assert d.engine.stats["build_half"] == 0
        d.set_param("wing-1", "span", 14.0)
        d.set_param("tail-1", "length", 9.0)
        self._drag_scene(d, "wing-1")
        assert d.engine.stats["build_half"] == 1      # the wing, not the tail

    def test_per_side_poses_and_single_sides_are_built_exactly(self, res):
        d = L.LabDocument(res)
        d.add_appendage("wing")
        d.set_control("wing-1", "swing", 10, side="r")
        d.set_param("wing-1", "span", 14.0)
        n = d.engine.stats["build_draft"]
        scene = self._drag_scene(d)
        assert d.engine.stats["build_half"] == 0 and d.engine.stats["build_draft"] == n + 1
        assert [o[1] for o in scene.owners[1:]] == ["l", "r"]
        d2 = L.LabDocument(res)
        d2.add_appendage("wing", side="left")
        d2.set_param("wing-1", "span", 14.0)
        assert [o[1] for o in self._drag_scene(d2).owners[1:]] == ["l"]

    def test_the_lab_uses_it_for_parameter_and_placement_drags_only(self, res):
        lb = L.Lab(res, size=(240, 240), threaded=False)
        lb.scheduler.debounce = 1e9
        lb.doc.add_appendage("wing")
        eng = lb.engine
        builds = eng.stats["build_draft"]
        lb.doc.set_param("wing-1", "span", 14.0)
        lb.doc.set_placement("wing-1", "scale", 1.2)
        assert eng.stats["build_half"] == 2 and eng.stats["build_draft"] == builds
        lb.doc.set_side("wing-1", "left")             # a single edit: exact
        assert eng.stats["build_draft"] == builds + 1
        lb.close()

    def test_a_pose_drag_poses_one_side_and_mirrors_it(self, res):
        d = L.LabDocument(res)
        d.add_appendage("wing")
        eng = d.engine
        eng.draft_scene(d.character)
        d.set_control("wing-1", "fold", 0.4)
        n = eng.stats["pose"]
        scene = eng.draft_scene(d.character, mode="pose", edited="wing-1")
        assert eng.stats["pose"] == n + 1                        # one wing posed, not two
        left, right = scene.shells[1].mesh, scene.shells[2].mesh
        assert np.array_equal(right.vertices, left.vertices * [-1, 1, 1])
        assert scene.owners == [(None, ""), ("wing-1", "l"), ("wing-1", "r")]
        exact = eng.draft_scene(d.character)                     # an orbit afterwards: both posed
        assert eng.stats["pose"] == n + 2
        assert np.abs(exact.shells[2].mesh.vertices.max(axis=0)
                      - right.vertices.max(axis=0)).max() < 0.35

    def test_a_pose_drag_with_per_side_poses_poses_both(self, res):
        d = L.LabDocument(res)
        d.add_appendage("wing")
        d.set_control("wing-1", "swing", 9.0, side="r")
        eng = d.engine
        eng.draft_scene(d.character)
        n = eng.stats["pose"]
        d.set_control("wing-1", "fold", 0.4)
        eng.draft_scene(d.character, mode="pose", edited="wing-1")
        assert eng.stats["pose"] == n + 2

    def test_the_final_never_takes_the_shortcut(self, res):
        d = L.LabDocument(res)
        d.add_appendage("wing")
        d.set_param("wing-1", "span", 14.0)
        self._drag_scene(d)
        scene = d.engine.final_scene(d.character)
        assert d.engine.stats["build"] == 1 and d.engine.built(
            d.spec("wing-1"), d.engine.body_state(d.character), False).mirrored is False
        assert [o for o in scene.owners[-2:]] == [("wing-1", "l"), ("wing-1", "r")]


# -- the view --------------------------------------------------------------------------------


class TestViewState:
    def test_orbit_uses_the_viewport_constants(self):
        v = L.ViewState(azimuth=10, elevation=0)
        v.orbit(20, 10)
        assert (v.azimuth, v.elevation) == (20.0, 4.0)
        v.orbit(-100, 0)
        assert v.azimuth == pytest.approx(330.0)                 # wraps
        v.orbit(0, 1000)
        assert v.elevation == 89.0
        v.orbit(0, -1000)
        assert v.elevation == -89.0

    def test_zoom_scales_the_radius_within_limits(self):
        v = L.ViewState(radius=10)
        v.zoom(0.9)
        assert v.radius == pytest.approx(9.0)
        v.zoom(1e-9)
        assert v.radius == v.min_radius
        v.zoom(1e9)
        assert v.radius == v.max_radius

    def test_pan_moves_the_picture_with_the_pointer(self):
        v = L.ViewState(azimuth=40, elevation=15, center=(1.0, 2.0, 3.0), radius=9.0)
        size = (400, 500)
        # the point the camera looks at, and one on the same depth plane: both follow the
        # pointer pixel for pixel (a point nearer or farther moves a little more or less)
        right = v.camera(size).view[0, :3]
        points = np.array([v.center, np.asarray(v.center) + 2.0 * right])
        x0, y0, _ = v.project(points, size)
        v.pan(30, -20, size)
        x1, y1, _ = v.project(points, size)
        assert (x1 - x0) == pytest.approx([30, 30], abs=1e-6)
        assert (y1 - y0) == pytest.approx([-20, -20], abs=1e-6)
        assert v.center != (1.0, 2.0, 3.0)

    def test_the_camera_is_the_portrait_cameras(self):
        from minimaster.character import raster as R
        v = L.ViewState(azimuth=33, elevation=12, center=(0, 0, 1), radius=7, fov=26)
        cam = v.camera((300, 200))
        ref = R.orbit_camera(np.zeros(3), np.ones(3), (300, 200), 33, 12, 26,
                             (np.array([0, 0, 1.0]), 7.0))
        assert np.array_equal(cam.view, ref.view) and cam.focal == ref.focal

    def test_frame_fits_a_box(self):
        v = L.ViewState()
        v.frame([-1, -2, 0], [1, 2, 10])
        assert v.center == (0.0, 0.0, 5.0)
        assert v.radius == pytest.approx(np.linalg.norm([2, 4, 10]) / 2)


class TestFrame:
    def test_ppm_header_and_size_at_full_and_half_scale(self):
        img = np.zeros((5, 7, 4), np.uint8)
        img[:, :, 0] = np.arange(7)
        f = L.Frame(1, "final", img, (7, 5))
        data = f.ppm()
        assert data.startswith(b"P6 7 5 255\n") and len(data) == len(b"P6 7 5 255\n") + 7 * 5 * 3
        half = L.Frame(1, "draft", img[:3, :4], (7, 5), scale=2)
        d2 = half.ppm()
        head = b"P6 7 5 255\n"
        assert d2.startswith(head) and len(d2) == len(head) + 7 * 5 * 3       # cropped to display
        px = np.frombuffer(d2[len(head):], np.uint8).reshape(5, 7, 3)
        assert np.array_equal(px[:, 0, 0], px[:, 1, 0])                        # pixels doubled


# -- picking ---------------------------------------------------------------------------------


@needs_assets
class TestPicking:
    def _front_quads(self, res, d, view, size, n=6):
        """A few body quads (centroids) well inside the front of the silhouette."""
        verts = d.engine.body_state(d.character).verts
        cage_v, cage_q = res.base.body_cage(verts)
        cent = cage_v[cage_q].mean(axis=1)
        zc = cent @ MH_TO_Z_UP[:3, :3].T
        px, py, _ = view.project(zc, size)
        normal = np.cross(cage_v[cage_q[:, 2]] - cage_v[cage_q[:, 0]],
                          cage_v[cage_q[:, 3]] - cage_v[cage_q[:, 1]])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        facing = normal[:, 2] > 0.8                                          # faces the camera at az 0
        torso = (cent[:, 1] > -2) & (cent[:, 1] < 4) & (np.abs(cent[:, 0]) < 1.5)
        ids = np.flatnonzero(facing & torso)
        return ids[:: max(1, len(ids) // n)][:n], px, py, cent

    def test_a_pick_returns_the_quad_the_pixel_shows(self, res):
        d = L.LabDocument(res)
        view = L.ViewState(azimuth=0, elevation=0, center=(0, 0, 0), radius=10)
        size = (400, 500)
        ids, px, py, cent = self._front_quads(res, d, view, size)
        assert len(ids) >= 4
        hits = 0
        for q in ids:
            pick = d.engine.pick(d.character, view, size, px[q], py[q])
            assert pick.kind == "body"
            hits += pick.quad == q
            # the surface point is on the body next to that quad, and the weights rebuild it
            assert np.linalg.norm(pick.point - cent[q]) < 0.7
            cage_v, cage_q = res.base.body_cage(d.engine.body_state(d.character).verts)
            corners = cage_v[cage_q[pick.quad]]
            assert sum(pick.weights) == pytest.approx(1.0) and min(pick.weights) >= 0
            assert np.dot(pick.weights, corners) == pytest.approx(pick.point, abs=1e-6)
        assert hits >= len(ids) - 1                      # a neighbouring quad at most, rarely

    def test_the_pick_agrees_with_a_ray_cast_on_the_cage(self, res):
        from minimaster.core import raycast as rc
        from minimaster.core.mesh import Mesh
        d = L.LabDocument(res)
        view = L.ViewState(azimuth=25, elevation=6, center=(0, 0, 0), radius=10)
        size = (400, 500)
        verts = d.engine.body_state(d.character).verts
        cage_v, cage_q = res.base.body_cage(verts)
        tris = np.concatenate([cage_q[:, [0, 1, 2]], cage_q[:, [0, 2, 3]]])
        mesh = Mesh(cage_v @ MH_TO_Z_UP[:3, :3].T, tris)
        cam = view.camera(size)
        grid = [(x, y) for y in range(60, 440, 40) for x in range(60, 340, 40)]
        hits = [(x, y) for x, y in grid
                if d.engine.pick(d.character, view, size, x, y) is not None]
        assert len(hits) >= 8                                 # the silhouette is well covered
        for x, y in hits[:: max(1, len(hits) // 10)]:
            pick = d.engine.pick(d.character, view, size, x, y)
            assert pick is not None and pick.kind == "body"
            origin, direction = rc.screen_ray(x + 0.5, y + 0.5, size[0], size[1], cam.view, cam.focal)
            hit = rc.ray_mesh_hit(origin, direction, mesh)
            assert hit is not None
            assert int(hit.face) % res.n_quads == pick.quad or \
                np.linalg.norm(hit.point - MH_TO_Z_UP[:3, :3] @ pick.point) < 0.3

    def test_background_is_a_miss_and_an_appendage_is_an_appendage(self, res):
        d = L.LabDocument(res)
        d.add_appendage("wing")
        view = L.ViewState(azimuth=180, elevation=6, center=(0, 0, 0), radius=16)
        size = (400, 400)
        assert d.engine.pick(d.character, view, size, 2, 2) is None
        scene = d.engine.draft_scene(d.character)
        wing = scene.shells[1].mesh
        centre = wing.vertices[wing.faces].mean(axis=1)
        px, py, _ = view.project(centre, size)
        found = set()
        for i in range(0, len(centre), max(1, len(centre) // 40)):
            pick = d.engine.pick(d.character, view, size, px[i], py[i])
            if pick is not None and pick.kind == "appendage":
                found.add((pick.spec_id, pick.side))
        assert ("wing-1", "l") in found

    def test_body_only_looks_through_the_wings(self, res):
        d = L.LabDocument(res)
        d.add_appendage("wing")
        view = L.ViewState(azimuth=180, elevation=6, center=(0, 0, 0), radius=16)
        size = (400, 400)
        wing = d.engine.draft_scene(d.character).shells[1].mesh
        c = wing.vertices[wing.faces].mean(axis=1).mean(axis=0, keepdims=True)
        px, py, _ = view.project(c, size)
        assert d.engine.pick(d.character, view, size, px[0], py[0]).kind == "appendage"
        through = d.engine.pick(d.character, view, size, px[0], py[0], body_only=True)
        assert through is None or through.kind == "body"


# -- the lab: click, place, reveal -----------------------------------------------------------


@needs_assets
class TestLab:
    def test_adding_to_the_back_swings_the_view_round_and_frames_it(self, lab):
        lab.frame_content()
        assert lab.view.azimuth == 25.0
        r0 = lab.view.radius
        lab.doc.add_appendage("wing")
        assert lab.view.azimuth == L.VIEW_PRESETS["rear-quarter"][0]
        assert lab.view.radius > r0                              # the wings are in the frame
        lab.view.azimuth = 90
        lab.doc.add_appendage("tail")
        assert lab.view.azimuth == 90                            # a side view is left alone
        r1 = lab.view.radius
        lab.doc.add_appendage("horn")
        assert lab.view.radius == r1                             # never zooms IN by itself

    def test_every_change_draws_a_draft_and_arms_the_final(self, lab):
        frames = []
        lab.on_frame = frames.append
        lab.doc.add_appendage("tail")
        assert [f.tier for f in frames] == ["draft"] and lab.scheduler._pending is not None
        lab.doc.set_param("tail-1", "length", 7)
        assert [f.tier for f in frames] == ["draft", "draft"]
        lab.scheduler.wait_idle()
        assert [f.tier for f in frames] == ["draft", "draft", "final"]
        assert frames[-1].gen == lab.scheduler.gen

    def test_scopes_pick_the_cheap_path(self, lab):
        events = []
        lab.doc.listeners.append(events.append)
        lab.doc.add_appendage("wing")
        lab.scheduler.wait_idle()
        n = lab.engine.stats["build_draft"]
        lab.doc.set_macro("height", 0.7)                         # body: re-placed, not rebuilt
        assert lab.engine.stats["build_draft"] == n and lab.engine.stats["replace"] == 1
        lab.scheduler.wait_idle()
        lab.orbit(10, 0)                                         # the exact draft, built ONCE
        assert lab.engine.stats["build_draft"] == n + 1
        lab.orbit(10, 0)
        lab.zoom(0.9)
        lab.pan(5, 5)
        assert lab.engine.stats["build_draft"] == n + 1

    def test_click_selects_an_appendage_or_nothing(self, lab):
        lab.doc.add_appendage("wing")
        lab.doc.select(None)
        lab.view = L.ViewState(azimuth=180, elevation=6, center=(0, 0, 0), radius=16)
        scene = lab.engine.draft_scene(lab.doc.character)
        wing = scene.shells[1].mesh
        centre = wing.vertices[wing.faces].mean(axis=1)
        px, py, _ = lab.view.project(centre, lab.size)
        picked = None
        for i in range(0, len(centre), 25):
            msg = lab.click(px[i], py[i])
            if lab.doc.selected == "wing-1":
                picked = msg
                break
        assert picked == "selected wing-1"
        lab.click(2, 2)
        assert lab.doc.selected is None

    def test_placing_by_clicking_the_body_moves_the_anchor(self, lab):
        lab.doc.add_appendage("tail")
        lab.view = L.ViewState(azimuth=0, elevation=0, center=(0, 0, 0), radius=10)
        before = lab.doc.spec("tail-1").placement["anchor"]["face"]
        lab.start_placing("tail-1")
        msg = lab.click(2, 2)                                    # background: still armed
        assert "missed" in msg and lab.placing == "tail-1"
        v = lab.engine.body_state(lab.doc.character).verts
        cage_v, cage_q = lab.resources.base.body_cage(v)
        cent = cage_v[cage_q].mean(axis=1)
        chest = int(np.argmin(np.linalg.norm(cent - [0.8, 2.0, 1.2], axis=1)))   # front left chest
        assert cent[chest][0] > 0
        x, y, _ = lab.view.project(cent[chest] @ MH_TO_Z_UP[:3, :3].T, lab.size)
        picked = lab.pick(x[0], y[0], body_only=True)
        assert picked.kind == "body" and np.linalg.norm(cent[picked.quad] - cent[chest]) < 0.7
        msg = lab.click(x[0], y[0])
        assert lab.placing is None and f"placed on body quad {picked.quad}" in msg
        placement = lab.doc.spec("tail-1").placement["anchor"]
        assert placement["face"] == picked.quad != before       # a left-hand click is stored as is
        assert sum(placement["weights"]) == pytest.approx(1.0, abs=1e-5)
        lab.doc.undo()                                           # one undo step
        assert lab.doc.spec("tail-1").placement["anchor"]["face"] == before

    def test_load_frames_the_content_and_cancels_placing(self, lab, tmp_path):
        lab.doc.add_appendage("tail")
        lab.start_placing("tail-1")
        lab.doc.new()
        assert lab.placing is None

    def test_placing_is_cancelled_when_the_armed_appendage_is_deselected_or_removed(self, lab):
        lab.doc.add_appendage("tail")
        lab.doc.add_appendage("horn")
        lab.start_placing("horn-1")
        lab.doc.select("tail-1")                      # something else selected: no longer armed
        assert lab.placing is None
        lab.start_placing("tail-1")
        lab.doc.select("tail-1")                      # the same one: still armed
        assert lab.placing == "tail-1"
        lab.doc.remove_appendage("tail-1")
        assert lab.placing is None

    def test_the_overlay_builds_nothing_and_matches_the_built_anchors(self, lab):
        for kind in ("wing", "tail", "horn", "harvest"):
            lab.doc.add_appendage(kind)
        # in the middle of a parameter drag only the half-built draft exists: drawing the
        # markers must not trigger the exact build (that is what made drags 3x slower)
        lab.doc.set_param("wing-1", "span", 14.0)
        assert lab.engine.stats["build_half"] == 1
        stats = dict(lab.engine.stats)
        marks = lab.overlay_points()
        lab.floor_z()
        assert dict(lab.engine.stats) == stats                    # drawing the markers is free
        got = {name: p for name, p, _ in marks}
        for spec in lab.doc.specs:
            for placed in lab.engine.placed(lab.doc.character, spec.id):
                assert got[placed.name] == pytest.approx(
                    MH_TO_Z_UP[:3, :3] @ placed.anchor, abs=1e-9), placed.name
        assert sorted(got) == sorted(p.name for s in lab.doc.specs
                                     for p in lab.engine.placed(lab.doc.character, s.id))

    def test_the_floor_is_the_sole_of_the_body(self, lab):
        verts = lab.engine.body_state(lab.doc.character).verts
        cage = lab.resources.base.body_cage(verts)[0]
        assert lab.floor_z() == pytest.approx(cage[:, 1].min())
        lab.doc.set_macro("height", 0.9)
        assert lab.floor_z() != pytest.approx(cage[:, 1].min())

    def test_the_overlay_marks_every_appendage_and_the_selected_one(self, lab):
        lab.doc.add_appendage("wing")
        lab.doc.add_appendage("tail")
        lab.doc.select("tail-1")
        marks = lab.overlay_points()
        assert [(n, sel) for n, _, sel in marks] == [("wing-1.l", False), ("wing-1.r", False),
                                                     ("tail-1", True)]
        floor = lab.floor_z()
        assert all(p[2] > floor for _, p, _ in marks)


# -- the scheduler ---------------------------------------------------------------------------


class TestCache:
    def test_lru_by_entries_and_most_recently_used_survives(self):
        c = L.Cache(2)
        c.put("a", 1)
        c.put("b", 2)
        assert c.get("a") == 1                       # a is now the most recent
        c.put("c", 3)
        assert c.get("b") is None and c.get("a") == 1 and c.get("c") == 3

    def test_bounded_by_bytes_with_sizes_that_grow_after_insertion(self):
        class Blob:
            def __init__(self):
                self.n = 10
        c = L.Cache(100, max_bytes=100, sizeof=lambda b: b.n)
        blobs = [Blob() for _ in range(5)]
        for i, b in enumerate(blobs):
            c.put(i, b)
        assert len(c) == 5
        blobs[4].n = 60                              # the newest grows (a Built caching poses)
        c.put("x", Blob())
        assert len(c) < 7 and c.peek("x") is not None and c.peek(0) is None  # oldest evicted first
        big = Blob()
        big.n = 10_000
        c.put("big", big)
        assert c.peek("big") is big and len(c) == 1  # one oversized entry is still kept

    def test_get_or_build_builds_once(self):
        c = L.Cache(4)
        calls = []
        assert c.get_or_build("k", lambda: calls.append(1) or 7) == 7
        assert c.get_or_build("k", lambda: calls.append(1) or 8) == 7 and calls == [1]


@needs_assets
def test_a_built_appendage_accounts_for_its_poses(res):
    d = L.LabDocument(res)
    d.add_appendage("wing")
    spec, body = d.spec("wing-1"), d.engine.body_state(d.character)
    b = d.engine.built(spec, body, False)
    empty = b.nbytes()
    for i in range(10):
        b.posed(0, {"fold": 0.05 * i})
    assert len(b._poses) == 6                         # the per-build pose cache is bounded
    assert b.nbytes() > empty
    small = L.Engine(res)
    small._built = L.Cache(48, 1, lambda x: x.nbytes())       # a 1-byte budget keeps the newest only
    spec2 = d.spec("wing-1")
    small.built(spec2, body, False)
    d.set_param("wing-1", "span", 14.0)
    small.built(d.spec("wing-1"), body, False)
    assert len(small._built) == 1


class _Fake:
    """A scheduler under test with fake draft / final functions and a manual clock."""

    def __init__(self, threaded=False, final=None):
        self.now = 0.0
        self.drafts, self.shown, self.final_gens = [], [], []
        self.final_threads = set()
        self._final = final
        self.s = L.PreviewScheduler(self.draft, self.final, self.shown.append,
                                    clock=lambda: self.now, threaded=threaded)

    def draft(self, gen, payload):
        self.drafts.append(gen)
        return L.Frame(gen, "draft", np.zeros((2, 2, 4), np.uint8), (2, 2))

    def final(self, gen, payload, check):
        self.final_gens.append(gen)
        self.final_threads.add(threading.current_thread())
        if self._final is not None:
            self._final(gen, payload, check)
        return L.Frame(gen, "final", np.zeros((2, 2, 4), np.uint8), (2, 2))


class TestScheduler:
    def test_the_draft_is_immediate_and_the_final_waits_for_quiet(self):
        f = _Fake()
        gen = f.s.request("a")
        assert f.drafts == [gen] and [x.tier for x in f.shown] == ["draft"]
        f.now = 0.149
        f.s.tick()
        assert f.final_gens == [] and not f.s.final_due()
        f.now = 0.151
        assert f.s.final_due()
        f.s.tick()
        assert f.final_gens == [gen] and [x.tier for x in f.shown] == ["draft", "final"]
        assert f.s.idle

    def test_an_edit_inside_the_debounce_restarts_it(self):
        f = _Fake()
        f.s.request("a")
        f.now = 0.10
        f.s.request("b")                      # restarts the quiet time
        f.now = 0.20
        f.s.tick()
        assert f.final_gens == []             # 0.10 s of quiet only
        f.now = 0.26
        f.s.tick()
        assert f.final_gens == [2]            # only the latest payload is ever finalised

    def test_only_the_final_of_the_current_generation_is_shown(self):
        f = _Fake()
        f.s.request("a")
        f.s._results.put(("frame", 1, L.Frame(1, "final", np.zeros((2, 2, 4), np.uint8), (2, 2))))
        f.s.request("b")                      # gen 2 now; the queued gen-1 frame is stale
        f.s.tick()
        assert f.s.stale_dropped == 1 and [x.tier for x in f.shown] == ["draft", "draft"]

    def test_no_final_when_not_asked(self):
        f = _Fake()
        f.s.request("a", final=False)
        f.now = 5.0
        f.s.tick()
        assert f.final_gens == [] and f.s.idle

    def test_errors_in_the_final_are_reported_not_swallowed(self):
        def boom(gen, payload, check):
            raise RuntimeError("kaput")

        f = _Fake(final=boom)
        seen = []
        f.s.on_error = seen.append
        f.s.request("a")
        f.now = 1.0
        f.s.tick()
        assert [str(e) for e in seen] == ["kaput"] and f.s.errors and f.s.idle

    def test_threaded_final_runs_on_the_worker_and_arrives_through_the_queue(self):
        f = _Fake(threaded=True)
        f.s.debounce = 0.0
        main = threading.current_thread()
        f.s.request("a")
        assert f.s.wait_idle(5)
        assert f.final_threads and main not in f.final_threads
        assert [x.tier for x in f.shown] == ["draft", "final"]
        f.s.close()

    def test_a_new_request_cancels_the_final_in_flight(self):
        started, state = threading.Event(), {}

        def slow(gen, payload, check):
            started.set()
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 5.0:
                check()                              # the worker's checkpoints
                time.sleep(0.005)
            state["ran_to_the_end"] = True

        f = _Fake(threaded=True, final=slow)
        f.s.debounce = 0.0
        f.s.request("a")
        f.s.tick()
        assert started.wait(5)
        t0 = time.perf_counter()
        f.s.request("b")                             # cancels job 1
        deadline = time.perf_counter() + 2
        while f.s._worker.busy and time.perf_counter() < deadline:
            if f.s._worker.last_cancel_latency is not None:
                break
            time.sleep(0.001)
        lat = f.s._worker.last_cancel_latency
        assert lat is not None and lat < 0.1, lat   # cancel latency: one checkpoint
        assert time.perf_counter() - t0 < 0.5 and "ran_to_the_end" not in state
        f.s.close()

    def test_a_cancelled_job_never_delivers(self):
        release = threading.Event()

        def gated(gen, payload, check):
            release.wait(5)
            check()

        f = _Fake(threaded=True, final=gated)
        f.s.debounce = 0.0
        f.s.request("a")
        f.s.tick()
        time.sleep(0.05)
        f.s.request("b", final=False)                # cancels; gen is now 2
        release.set()
        time.sleep(0.1)
        f.s.tick()
        assert [x.tier for x in f.shown] == ["draft", "draft"]
        f.s.close()

    def test_a_result_from_before_the_last_edit_is_dropped_even_if_it_finished(self):
        gate = threading.Event()

        def uncancellable(gen, payload, check):
            gate.wait(5)                             # finishes without noticing the cancel

        f = _Fake(threaded=True, final=uncancellable)
        f.s.debounce = 0.0
        f.s.request("a")
        f.s.tick()
        time.sleep(0.05)
        f.s.request("b", final=False)
        gate.set()
        deadline = time.time() + 2
        while f.s.stale_dropped == 0 and time.time() < deadline:
            f.s.tick()
            time.sleep(0.005)
        assert f.s.stale_dropped == 1 and "final" not in [x.tier for x in f.shown]
        f.s.close()

    def test_the_payload_is_what_the_worker_sees(self):
        seen = []
        f = _Fake(threaded=True, final=lambda g, p, c: seen.append(p))
        f.s.debounce = 0.0
        payload = {"json": "{}"}
        f.s.request(payload)
        f.s.wait_idle(5)
        assert seen == [payload]
        f.s.close()


class TestBackgroundTask:
    def test_result_and_error_are_polled(self):
        t = L.BackgroundTask(lambda: 41 + 1)
        assert t.wait(5) and t.value == 42 and t.error is None
        e = L.BackgroundTask(lambda: 1 / 0)
        assert e.wait(5) and isinstance(e.error, ZeroDivisionError) and e.value is None

    def test_poll_before_done_is_false(self):
        gate = threading.Event()
        t = L.BackgroundTask(lambda: gate.wait(5))
        assert t.poll() is False
        gate.set()
        assert t.wait(5)


# -- the real worker -------------------------------------------------------------------------


@needs_assets
class TestThreadedLab:
    def test_a_final_frame_arrives_and_equals_a_direct_render(self, res):
        from minimaster.character import portrait
        lb = L.Lab(res, size=(240, 300), threaded=True, debounce=0.02)
        frames = []
        lb.on_frame = frames.append
        try:
            lb.doc.add_appendage("tail")
            lb.doc.add_appendage("horn")
            assert lb.scheduler.wait_idle(60)
            finals = [f for f in frames if f.tier == "final"]
            assert len(finals) == 1 and finals[0].gen == lb.scheduler.gen
            scene = lb.engine.final_scene(lb.doc.character)
            want = portrait.render_portrait(scene.shells, size=(240, 300), azimuth=lb.view.azimuth,
                                            elevation=lb.view.elevation, fit=lb.view.fit,
                                            background=L.BACKGROUND)
            assert np.array_equal(finals[0].image, want)          # bytes out == the renderer's
            assert finals[0].ppm().startswith(b"P6 240 300 255\n")
        finally:
            lb.close()

    def test_a_burst_of_edits_yields_one_final(self, res):
        lb = L.Lab(res, size=(200, 200), threaded=True, debounce=0.05)
        frames = []
        lb.on_frame = frames.append
        try:
            lb.doc.add_appendage("tail")
            for i in range(8):
                lb.doc.set_param("tail-1", "length", 6 + 0.5 * i)
                lb.tick()
            assert lb.scheduler.wait_idle(60)
            assert [f.tier for f in frames].count("final") == 1
            assert [f.tier for f in frames].count("draft") == 9
        finally:
            lb.close()


# -- export ----------------------------------------------------------------------------------


@needs_assets
class TestExport:
    def test_check_without_a_path_writes_nothing_and_reports(self, lab, tmp_path):
        lab.doc.add_appendage("tail")
        lab.doc.add_appendage("horn")
        r = lab.export_stl(None, size="medium")
        assert r.path is None and not r.written and not r.blocked
        assert "printability" in r.report_text and "tail-1" in r.report_text
        assert r.blocking == [] and r.warnings and "height" in r.assembly_text

    def test_the_stl_is_the_one_the_cli_writes(self, lab, tmp_path):
        from minimaster.__main__ import main
        lab.doc.add_appendage("tail", params={"length": 7})
        lab.doc.add_appendage("horn", side="left")
        lab.doc.set_macro("height", 0.6)
        mm = lab.doc.save(tmp_path / "c.mmchar")
        r = lab.export_stl(tmp_path / "lab.stl", size="small")
        assert r.written and (tmp_path / "lab.stl").stat().st_size > 1000
        assert main(["character", "--character", str(mm), "-o", str(tmp_path / "cli.stl"),
                     "--size", "small"]) == 0
        assert (tmp_path / "lab.stl").read_bytes() == (tmp_path / "cli.stl").read_bytes()

    def test_a_bat_wing_pair_is_exported_through_the_piece_path(self, lab, tmp_path):
        from minimaster.__main__ import main
        lab.doc.add_appendage("wing")
        lab.doc.set_control("wing-1", "fold", 0.4)
        lab.doc.set_control("wing-1", "swing", 12, side="r")
        mm = lab.doc.save(tmp_path / "w.mmchar")
        r = lab.export_stl(tmp_path / "lab.stl", size="medium", force=True)
        assert r.written and any("wing-1.l" in g for g in r.printability.groups)
        assert main(["character", "--character", str(mm), "-o", str(tmp_path / "cli.stl"),
                     "--size", "medium", "--force"]) == 0
        assert (tmp_path / "lab.stl").read_bytes() == (tmp_path / "cli.stl").read_bytes()

    def test_a_blocking_report_stops_the_export_until_forced(self, lab, tmp_path):
        lab.doc.add_appendage("tail")
        lab.doc.set_placement("tail-1", "offset", 3.0)             # floats off the body
        out = tmp_path / "x.stl"
        with pytest.raises(L.ExportBlocked) as err:
            lab.export_stl(out, size="medium")
        assert not out.exists() and err.value.result.blocked and err.value.result.blocking
        assert "not connected to the body" in err.value.result.report_text
        assert "blocks this export" in str(err.value)
        forced = lab.export_stl(out, size="medium", force=True)
        assert out.exists() and forced.written and forced.blocked

    def test_a_body_only_character_exports_without_a_report(self, lab, tmp_path):
        r = lab.export_stl(tmp_path / "b.stl", size="tiny")
        assert r.written and r.printability is None and r.report_text == ""

    def test_the_export_uses_the_figure_for_the_size(self, lab, tmp_path):
        from minimaster.core.stl import read_stl
        lab.doc.add_appendage("tail")
        lab.export_stl(tmp_path / "t.stl", size="medium")
        mesh = read_stl(tmp_path / "t.stl")
        lo, hi = mesh.bounds
        assert 30.0 < hi[2] - lo[2] < 40.0                         # 32 mm body + base

    def test_disabled_appendages_are_not_printed(self, lab, tmp_path):
        lab.doc.add_appendage("tail")
        lab.doc.set_enabled("tail-1", False)
        r = lab.export_stl(None, size="medium")
        assert r.printability is None


# -- the whole sequence, headless ------------------------------------------------------------


@needs_assets
def test_the_acceptance_sequence_headless(res, tmp_path):
    """add wing, add tail, tune, undo x2, redo, save, reload, export."""
    lb = L.Lab(res, size=(240, 240), threaded=False)
    d = lb.doc
    d.add_appendage("wing")
    d.add_appendage("tail")
    d.set_param("wing-1", "span", 12)
    d.set_control("wing-1", "fold", 0.3)
    d.set_param("tail-1", "length", 8)
    full = _json(d)
    assert d.undo()                                       # the tail's length
    after_one = _json(d)
    assert d.undo()                                       # the fold
    assert d.spec("tail-1").params == {} and not d.spec("wing-1").controls
    assert d.spec("wing-1").params["span"] == 12
    assert d.redo() and _json(d) == after_one             # the fold is back, the length is not
    assert d.spec("wing-1").controls == {"*": {"fold": 0.3}} and d.spec("tail-1").params == {}
    assert d.redo() and _json(d) == full                  # everything is back
    assert d.spec("tail-1").params["length"] == 8.0
    d.undo()
    path = d.save(tmp_path / "seq.mmchar")
    d2 = L.LabDocument(res)
    d2.load(path)
    assert _json(d2) == _json(d) and _json(d2) == after_one
    res1 = lb.export_stl(tmp_path / "a.stl", size="medium", force=True)
    lb2 = L.Lab(res, character=d2.character, size=(240, 240), threaded=False)
    res2 = lb2.export_stl(tmp_path / "b.stl", size="medium", force=True)
    assert res1.written and res2.written
    assert (tmp_path / "a.stl").read_bytes() == (tmp_path / "b.stl").read_bytes()
    lb.close()
    lb2.close()


@needs_assets
def test_the_re_place_shortcut_remembers_only_a_few_drafts(res):
    """Engine._last_draft kept every distinct shape edit's Built object long after the
    byte-bounded cache had dropped it: ~4 MB per feathered-wing edit (1.8 GB after 400 span
    edits). It serves the re-place of the spec being dragged, so a handful is enough."""
    d = L.LabDocument(res)
    d.add_appendage("tail")
    eng = d.engine
    d.set_anatomy(enabled=False)
    for k in range(40):
        d.set_param("tail-1", "length", 4.0 + 0.1 * k)
        eng.draft_scene(d.character)
    assert eng.stats["build_draft"] >= 40                      # every edit really was a new build
    assert len(eng._last_draft) <= L.Engine.LAST_DRAFT_KEEP


@needs_assets
def test_pose_presets_are_names_for_the_wing_and_nothing_for_a_kind_without_them():
    from minimaster.character.wingrig import PRESETS
    assert L.pose_presets(kinds.KINDS["wing"]) == tuple(PRESETS)
    for k in ("tail", "horn", "harvest"):
        assert L.pose_presets(kinds.KINDS[k]) == ()
