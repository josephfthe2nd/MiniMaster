"""The Character tab and its image view under a real Tk (skipped without tkinter or a display).

Run them headless with a virtual display:  ``xvfb-run -a python -m pytest tests/test_character_panel.py``.
The document logic they drive is tested without any display in ``test_lab.py``; here the point
is that the widgets reach it: the tab swaps the left pane, the property grid is generated from the
kind, sliders edit one undo step per drag, a click in the picture selects or places, the export
shows the printability report, and a sequence of operations through the panel gives the same
document and the same STL as the same sequence through the headless core.
"""

import json
import time

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")

from minimaster.character import lab as L  # noqa: E402
from minimaster.character import mhbase as mh  # noqa: E402

needs_assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")


def _display():
    try:
        root = tk.Tk()
    except tk.TclError as exc:                      # no DISPLAY: importorskip is not enough
        pytest.skip(f"no usable display: {exc}")
    return root


@pytest.fixture
def root():
    r = _display()
    r.geometry("1000x700")
    yield r
    r.destroy()


@pytest.fixture
def app():
    from minimaster.app import MiniMasterApp
    from minimaster.templates import load_template
    try:
        a = MiniMasterApp(load_template("human_fighter"))
    except tk.TclError as exc:
        pytest.skip(f"no usable display: {exc}")
    a.geometry("1260x830")
    a.update()
    yield a
    a.destroy()


@pytest.fixture
def panel(root):
    """A bare panel (no studio window) with its view in the same root."""
    from minimaster.character_panel import CharacterPanel
    p = CharacterPanel(root, view_parent=root, resources=L.Resources.shared())
    p.pack(side="right", fill="y")
    v = p.view
    v.pack(side="left", fill="both", expand=True)
    root.update()
    p.activate()
    pump(root, until=lambda: v.frames_shown > 0)
    yield p
    p.destroy()


def pump(widget, seconds=0.0, until=None, timeout=60.0):
    end = time.monotonic() + (timeout if until else seconds)
    while time.monotonic() < end:
        widget.update()
        if until is not None and until():
            return True
        time.sleep(0.003)
    return until is None


def settled(widget, p):
    """Wait until the final render of the last edit is on screen."""
    assert pump(widget, until=lambda: p.lab.scheduler.idle and p.view.frame is not None
                and p.view.frame.tier == "final")


def key(widget, sequence):
    """Send a key event to ``widget`` (key events go to the focus window)."""
    widget.focus_force()
    widget.update()
    widget.event_generate(sequence)
    widget.update()


def drag(widget, row, value):
    row.scale.event_generate("<ButtonPress-1>", x=1, y=1)
    row.scale.set(value)
    widget.update()
    row.scale.event_generate("<ButtonRelease-1>", x=1, y=1)
    widget.update()


# -- the image view (no lab needed) ---------------------------------------------------------


class TestImageView:
    def _view(self, root):
        from minimaster.character.imageview import ImageView
        v = ImageView(root, width=300, height=200)
        v.pack()
        root.update()
        return v

    def test_it_shows_a_frame_through_a_photoimage(self, root):
        v = self._view(root)
        img = np.zeros((200, 300, 4), np.uint8)
        img[:, :, 0] = 200
        v.show_frame(L.Frame(1, "final", img, (300, 200)))
        root.update()
        assert v.frames_shown == 1 and v.frame.tier == "final"
        assert (v._photo.width(), v._photo.height()) == (300, 200)
        assert v._photo.get(10, 10) == (200, 0, 0)
        half = L.Frame(2, "draft", img[:100, :150], (300, 200), scale=2)
        v.show_frame(half)
        assert (v._photo.width(), v._photo.height()) == (300, 200)      # upscaled to the pane
        assert "draft" in [v.itemcget(i, "text") for i in v.find_withtag("badge")]

    def test_gestures_call_back_with_the_viewport_conventions(self, root):
        v = self._view(root)
        got = {"orbit": [], "pan": [], "zoom": [], "click": []}
        v.on_orbit = lambda dx, dy: got["orbit"].append((dx, dy))
        v.on_pan = lambda dx, dy: got["pan"].append((dx, dy))
        v.on_zoom = got["zoom"].append
        v.on_click = lambda x, y: got["click"].append((x, y))
        v.event_generate("<ButtonPress-1>", x=50, y=50)
        v.event_generate("<B1-Motion>", x=60, y=55)
        v.event_generate("<B1-Motion>", x=70, y=55)
        v.event_generate("<ButtonRelease-1>", x=70, y=55)
        assert got["orbit"] == [(10, 5), (10, 0)] and got["click"] == []     # a drag is no click
        v.event_generate("<ButtonPress-3>", x=50, y=50)
        v.event_generate("<B3-Motion>", x=45, y=62)
        v.event_generate("<ButtonRelease-3>", x=45, y=62)
        assert got["pan"] == [(-5, 12)]
        v.event_generate("<ButtonPress-2>", x=10, y=10)
        v.event_generate("<B2-Motion>", x=12, y=10)
        assert got["pan"][-1] == (2, 0)                                       # middle pans too
        v.event_generate("<Button-4>", x=5, y=5)
        v.event_generate("<Button-5>", x=5, y=5)
        v.event_generate("<MouseWheel>", x=5, y=5, delta=120)
        v.event_generate("<MouseWheel>", x=5, y=5, delta=-120)
        assert got["zoom"] == [0.9, 1.1, 0.9, 1.1]
        v.event_generate("<ButtonPress-1>", x=80, y=90)                      # 3 px or less: a click
        v.event_generate("<B1-Motion>", x=82, y=91)
        v.event_generate("<ButtonRelease-1>", x=82, y=91)
        assert got["click"] == [(82, 91)] and len(got["orbit"]) == 2

    def test_a_resize_is_reported_once_it_settles(self, root):
        v = self._view(root)
        sizes = []
        v.on_resize = lambda w, h: sizes.append((w, h))
        v.configure(width=420, height=260)
        pump(root, 0.25)
        assert sizes and sizes[-1] == (420, 260) and len(sizes) <= 2

    def test_the_overlay_draws_grid_marks_and_a_label_for_the_selection(self, root):
        v = self._view(root)
        view = L.ViewState(azimuth=20, elevation=15, center=(0, 0, 0), radius=10)
        marks = [("wing-1.l", np.array([1.0, 0, 2.0]), True), ("wing-1.r", np.array([-1.0, 0, 2.0]), True),
                 ("tail-1", np.array([0.0, 1, 0.0]), False)]
        v.overlay_provider = lambda: (view, marks, -8.0)
        img = np.zeros((200, 300, 4), np.uint8)
        img[:, :] = (32, 36, 44, 255)
        v.show_frame(L.Frame(1, "final", img, (300, 200)))
        tags = [v.gettags(i) for i in v.find_withtag("overlay")]
        assert sum("grid" in t for t in tags) > 20
        assert sum(any(x.startswith("mark:") for x in t) for t in tags) >= 4
        labels = [v.itemcget(i, "text") for i in v.find_withtag("mark") if v.type(i) == "text"]
        assert labels == ["wing-1"]                        # one label for the pair, none unselected
        v.show_grid = False
        v.refresh_overlay()
        assert not any("grid" in v.gettags(i) for i in v.find_withtag("overlay"))

    def test_the_grid_is_cut_where_the_figure_is(self):
        from minimaster.character.imageview import _visible_runs
        img = np.zeros((50, 100, 4), np.uint8)
        img[:, :] = (32, 36, 44, 255)
        img[:, 40:60] = (200, 150, 120, 255)               # the figure stands in the middle
        runs = _visible_runs(0.0, 25.0, 99.0, 25.0, img, 1)
        assert len(runs) == 2
        (x0, _, x1, _), (x2, _, x3, _) = runs
        assert x0 == 0 and 36 <= x1 <= 40 and 60 <= x2 <= 64 and x3 == pytest.approx(99.0)
        assert _visible_runs(0.0, 25.0, 99.0, 25.0, None, 1) == [(0.0, 25.0, 99.0, 25.0)]
        # a draft at half size is looked up at half the coordinates
        small = img[::2, ::2]
        assert len(_visible_runs(0.0, 25.0, 99.0, 25.0, small, 2)) == 2
        # nothing is drawn off the picture
        assert _visible_runs(-40.0, 25.0, -5.0, 25.0, img, 1) == []


# -- tk objects that become garbage on a worker thread --------------------------------------


class TestVariablesAreSafeGarbage:
    """``tk.Variable.__del__`` calls into Tcl; the cyclic collector may run it on the preview worker."""

    @staticmethod
    def _orphan(cls, root):
        v = cls(root, value="x")
        v.cycle = v                          # only the cyclic collector can free it
        name = str(v)
        del v
        return name

    def test_a_variable_finalised_on_another_thread_is_deferred_to_the_ui_thread(self, root):
        import gc
        import threading

        from minimaster.character_panel import StringVar, drain_deferred, _UI_THREAD
        _UI_THREAD[0] = threading.get_ident()
        gc.collect()
        name = self._orphan(StringVar, root)
        errors = []
        t = threading.Thread(target=lambda: (gc.collect(), None), name="not-the-ui")
        t0 = time.perf_counter()
        t.start()
        t.join(5)
        assert time.perf_counter() - t0 < 0.5                      # no one second wait per object
        assert root.tk.getboolean(root.tk.call("info", "exists", name)), errors   # not touched yet
        assert drain_deferred() == 1
        assert not root.tk.getboolean(root.tk.call("info", "exists", name))       # the UI thread did it
        assert drain_deferred() == 0

    def test_the_plain_tk_variable_is_what_would_have_gone_wrong(self, root):
        """The control: with ``tk.StringVar`` the finaliser really does run on the other thread."""
        import gc
        import threading
        gc.collect()
        seen = []
        orig = tk.Variable.__del__

        def spy(self):
            seen.append(threading.current_thread().name)
            # do not call into Tcl from here: that is exactly the hazard
        tk.Variable.__del__ = spy
        try:
            self._orphan(tk.StringVar, root)
            t = threading.Thread(target=gc.collect, name="not-the-ui")
            t.start()
            t.join(5)
        finally:
            tk.Variable.__del__ = orig
        assert "not-the-ui" in seen


# -- the tab inside the studio ---------------------------------------------------------------


@needs_assets
class TestStudio:
    def test_the_tab_is_added_last_and_costs_nothing_until_opened(self, app):
        tabs = app.notebook.tabs()
        assert len(tabs) == 6 and tabs[-1] == str(app.character_panel)
        assert app.notebook.tab(tabs[-1], "text") == "Character"
        assert app.character_panel.lab is None                  # nothing loaded at start-up
        assert app.viewport.winfo_manager() == "pack"
        assert app.mode() == "model"

    def test_selecting_the_tab_swaps_the_left_pane_and_back(self, app):
        panel = app.character_panel
        app.notebook.select(5)
        app.update()
        assert app.mode() == "character" and panel.lab is not None
        assert not app.viewport.winfo_manager() and panel.view.winfo_manager() == "pack"
        # the two panes sit left of the notebook column, in the same place
        assert panel.view.winfo_x() < app._right.winfo_x()
        app.notebook.select(0)
        app.update()
        assert app.viewport.winfo_manager() == "pack" and not panel.view.winfo_manager()
        assert panel._poll_job is None                           # the poll loop stops with the tab
        app.notebook.select(5)
        app.update()
        assert panel._poll_job is not None

    def test_the_studio_keeps_working_after_a_visit(self, app):
        app.notebook.select(5)
        app.update()
        app.notebook.select(0)
        app.update()
        n = len(app.scene.shapes)
        app.add_shape("box")
        assert len(app.scene.shapes) == n + 1
        app.undo()
        assert len(app.scene.shapes) == n
        assert len(app.viewport.find_all()) > 200

    def test_undo_and_redo_follow_the_tab(self, app):
        panel = app.character_panel
        n = len(app.scene.shapes)
        app.add_shape("box")                                   # a studio edit
        app.notebook.select(5)
        app.update()
        panel.add(kind="tail")
        assert [s.id for s in panel.lab.doc.specs] == ["tail-1"]
        app.event_generate("<Control-z>")                      # the key goes to the character
        app.update()
        assert panel.lab.doc.specs == [] and len(app.scene.shapes) == n + 1
        app.event_generate("<Control-y>")
        app.update()
        assert [s.id for s in panel.lab.doc.specs] == ["tail-1"]
        app.notebook.select(0)
        app.update()
        app.undo()                                             # now the studio's
        assert len(app.scene.shapes) == n and [s.id for s in panel.lab.doc.specs] == ["tail-1"]

    def test_save_and_open_follow_the_tab(self, app, tmp_path):
        panel = app.character_panel
        app.notebook.select(5)
        app.update()
        panel.add(kind="tail")
        panel.save(tmp_path / "c.mmchar")
        app.path = None
        assert (tmp_path / "c.mmchar").exists()
        app.save()                                             # File > Save in this tab: no studio dialog
        assert not panel.lab.doc.dirty

    def test_without_the_assets_the_tab_says_so_and_the_app_boots(self, app, monkeypatch):
        monkeypatch.setattr(L.Resources, "available", classmethod(lambda cls: False))
        panel = app.character_panel
        app.notebook.select(5)
        app.update()
        assert panel.lab is None and panel.status_text().startswith("Character lab unavailable")
        texts = [w.cget("text") for w in panel.winfo_children() if w.winfo_class() == "TLabel"]
        assert any("assets are not installed" in t for t in texts)
        panel.undo()                                           # every command is a quiet no-op
        assert panel.redo() is False
        app.notebook.select(0)
        app.update()
        assert app.viewport.winfo_manager() == "pack"

    def test_the_status_bar_follows_the_character(self, app):
        panel = app.character_panel
        app.notebook.select(5)
        app.update()
        assert "character:" in app.status.cget("text")
        panel.add(kind="tail")
        assert "1 appendages" in app.status.cget("text") and "*" in app.status.cget("text")

    def test_escape_cancels_placing(self, app):
        panel = app.character_panel
        app.notebook.select(5)
        app.update()
        panel.add(kind="tail")
        panel.place_btn.invoke()
        assert panel.lab.placing == "tail-1" and panel.view.cget("cursor") == "crosshair"
        app.event_generate("<Escape>")
        app.update()
        assert panel.lab.placing is None and not panel._placement_var.get()


# -- the panel -------------------------------------------------------------------------------


@needs_assets
class TestPanel:
    def test_the_add_menu_comes_from_the_registry(self, panel):
        labels = [e.label for e in panel._menu_entries]
        assert labels == [e.label for e in L.add_menu()] and "Wing (feathered)" in labels
        n = panel.add_menu.index("end") + 1
        assert n == len(labels)
        spec = panel.add(panel.entry("dragon"))
        assert spec.params == {"style": "dragon"} and spec.id == "wing-1"
        assert list(panel.tree.get_children()) == ["wing-1"]
        panel.add_menu.invoke(labels.index("Tail"))            # the real menu command
        assert list(panel.tree.get_children()) == ["wing-1", "tail-1"]
        assert panel.tree.selection() == ("tail-1",)

    def test_remove_duplicate_rename_enable(self, panel):
        panel.add(kind="horn")
        panel.add(kind="tail")
        panel.tree.selection_set("horn-1")
        panel.update()
        dup = panel.duplicate()
        assert dup.id == "horn-2" and panel.lab.doc.selected == "horn-2"
        panel.name_var.set("left horn")
        key(panel.name_entry, "<Return>")
        assert panel.lab.doc.spec("horn-2").name == "left horn"
        assert panel.tree.item("horn-2", "text") == "left horn"
        assert panel.tree.item("horn-2", "values")[2] == "\u2713"
        panel.enabled_check.invoke()                                  # the "on" check box
        assert panel.lab.doc.spec("horn-2").enabled is False
        assert panel.tree.item("horn-2", "values")[2] == ""
        panel.enabled_check.invoke()
        assert panel.lab.doc.spec("horn-2").enabled is True
        panel.remove("horn-2")
        assert list(panel.tree.get_children()) == ["horn-1", "tail-1"]
        panel.tree.selection_set("tail-1")
        key(panel.tree, "<Delete>")
        assert list(panel.tree.get_children()) == ["horn-1"]

    def test_the_property_grid_is_generated_from_the_kind(self, panel):
        doc = panel.lab.doc
        panel.add(panel.entry("feathered"))
        keys = {k[1] for k in panel.prop_rows if isinstance(k, tuple) and k[0] == "param"}
        assert keys == {p.key for p, _ in doc.param_specs("wing-1")}
        assert {"primaries", "tertials", "coverts"} <= keys and "billow" not in keys
        assert set(panel.control_rows) == {"fold", "raise_", "swing", "twist", "fan", "curl",
                                           "wrist_flex", "hand_twist"}
        assert {k[1] for k in panel.prop_rows if isinstance(k, tuple) and k[0] == "placement"} \
            == {"scale", "roll", "offset"}
        # a horn has no pose controls and a different grid; a tail (since T3) bends with four
        panel.add(kind="horn")
        assert panel.control_rows == {}
        panel.add(kind="tail")
        assert set(panel.control_rows) == {"curl", "droop", "lift", "sway"}
        assert {k[1] for k in panel.prop_rows if isinstance(k, tuple) and k[0] == "param"} \
            == {p.key for p, _ in doc.param_specs(doc.selected)}

    def test_a_param_that_gates_others_rebuilds_the_grid(self, panel):
        panel.add(panel.entry("membrane"))
        assert ("param", "billow") in panel.prop_rows and ("param", "primaries") not in panel.prop_rows
        style = panel.prop_rows[("param", "style")]
        style.set("feathered")
        panel._set_param("wing-1", "style", "feathered", True)
        panel.update()
        assert panel.lab.doc.spec("wing-1").params["style"] == "feathered"
        assert ("param", "primaries") in panel.prop_rows and ("param", "billow") not in panel.prop_rows

    def test_dragging_a_slider_edits_the_document_in_one_undo_step(self, panel):
        doc = panel.lab.doc
        now = [1000.0]

        def slow_clock():                       # every edit "seconds" apart: only the gesture can merge
            now[0] += 5.0
            return now[0]

        doc.clock = slow_clock
        panel.add(kind="tail")
        steps = len(doc.undo_stack._undo)
        row = panel.prop_rows[("param", "length")]
        row.scale.event_generate("<ButtonPress-1>", x=1, y=1)
        for v in (7.0, 8.0, 9.0, 10.0):
            row.scale.set(v)
            panel.update()
        row.scale.event_generate("<ButtonRelease-1>", x=1, y=1)
        assert doc.spec("tail-1").params["length"] == 10.0
        assert len(doc.undo_stack._undo) == steps + 1
        assert panel.undo_btn.instate(["!disabled"])
        panel.undo()
        assert doc.spec("tail-1").params.get("length", 6.0) == 6.0
        # the widgets follow the undo (the grid is rebuilt from the document)
        assert panel.prop_rows[("param", "length")].value == pytest.approx(6.0)

    def test_a_typed_value_is_clamped_and_a_bad_one_is_refused(self, panel):
        doc = panel.lab.doc
        panel.add(kind="tail")
        row = panel.prop_rows[("param", "length")]
        row.text.set("99")
        key(row.entry, "<Return>")
        assert doc.spec("tail-1").params["length"] == 20.0 and row.text.get() == "20"
        before = doc.snapshot()
        row.text.set("long")
        key(row.entry, "<Return>")
        assert doc.snapshot() == before and row.text.get() == "20"

    def test_a_refused_edit_is_a_status_line_not_a_traceback(self, panel):
        panel.add(kind="tail")
        panel._set_param("tail-1", "wingspan", 3, False)
        assert "no parameter 'wingspan'" in panel.status.cget("text")
        assert str(panel.status.cget("foreground")) != ""
        assert not panel.ok

    def test_pose_ranges_are_the_live_ranges_of_the_placed_wing(self, panel):
        panel.add(panel.entry("membrane"))
        doc = panel.lab.doc
        raise_row = panel.control_rows["raise_"]
        lo, hi = doc.control_range("wing-1", "raise_")
        assert (float(raise_row.scale.cget("from")), float(raise_row.scale.cget("to"))) == (lo, hi)
        drag(panel, panel.control_rows["fold"], 0.8)
        lo2, hi2 = doc.control_range("wing-1", "raise_")
        assert (lo2, hi2) != (lo, hi)
        assert (float(raise_row.scale.cget("from")), float(raise_row.scale.cget("to"))) == (lo2, hi2)
        panel.control_rows["raise_"].text.set("500")                 # beyond the live range
        key(panel.control_rows["raise_"].entry, "<Return>")
        assert doc.spec("wing-1").controls["*"]["raise_"] == pytest.approx(hi2)

    def test_pose_side_writes_one_side(self, panel):
        panel.add(panel.entry("membrane"))
        panel.pose_side.set("r")
        panel._set_control("wing-1", "swing", 12.0)
        assert panel.lab.doc.spec("wing-1").controls == {"r": {"swing": 12.0}}
        panel.pose_side.set("both")
        panel._set_control("wing-1", "fold", 0.4)
        assert panel.lab.doc.spec("wing-1").controls["*"] == {"fold": 0.4}

    def test_a_pose_preset_from_the_menu(self, panel):
        panel.add(panel.entry("membrane"))
        panel.pose_var.set("folded")
        panel._set_control("wing-1", "preset", "folded")
        assert panel.lab.doc.spec("wing-1").controls["*"]["preset"] == "folded"
        panel._set_control("wing-1", "preset", None)
        assert panel.lab.doc.spec("wing-1").controls == {}

    def test_body_sliders_edit_the_body_and_follow_undo_and_reset(self, panel):
        doc = panel.lab.doc
        row = panel.body_rows[("macro", "height")]
        drag(panel, row, 0.8)
        assert doc.character.macro["height"] == pytest.approx(0.8)
        shape = panel.body_rows[("slider", "torso.torso_vshape")]
        drag(panel, shape, 0.5)
        assert doc.character.sliders["torso.torso_vshape"] == pytest.approx(0.5)
        panel.undo()
        assert "torso.torso_vshape" not in doc.character.sliders and shape.value == 0.0
        panel.reset_body()
        assert doc.character.macro["height"] == 0.5 and row.value == pytest.approx(0.5)

    def test_the_adaptation_switch_and_its_notes(self, panel):
        doc = panel.lab.doc
        panel.add(panel.entry("feathered"))
        pump(panel, until=lambda: "wing" in panel.notes_label.cget("text"))
        panel.adapt_var.set(False)
        panel._adapt_toggled()
        assert doc.character.anatomy["enabled"] is False
        pump(panel, until=lambda: "no adaptation" in panel.notes_label.cget("text"))
        panel.undo()
        assert panel.adapt_var.get() is True

    def test_the_view_buttons_and_the_poll_deliver_frames(self, panel):
        panel.add(kind="tail")
        settled(panel, panel)
        assert panel.view.frame.tier == "final" and panel.view.frame.size == panel.lab.size
        az = panel.lab.view.azimuth
        panel.lab.set_view("left")
        assert panel.lab.view.azimuth == 90.0 != az
        settled(panel, panel)

    def test_placing_by_clicking_the_body_and_escape(self, panel):
        doc = panel.lab.doc
        panel.add(kind="tail")
        panel.lab.set_view("back")
        settled(panel, panel)
        before = doc.spec("tail-1").placement["anchor"]["face"]
        panel.place_btn.invoke()
        assert panel.lab.placing == "tail-1" and "click the body" in panel.view.banner
        w, h = panel.view.size
        panel.view.event_generate("<ButtonPress-1>", x=w // 2, y=int(h * 0.47))
        panel.view.event_generate("<ButtonRelease-1>", x=w // 2, y=int(h * 0.47))
        panel.update()
        assert panel.lab.placing is None and not panel._placement_var.get()
        assert panel.view.banner == "" and panel.view.cget("cursor") == ""
        assert doc.spec("tail-1").placement["anchor"]["face"] != before
        assert "placed on body quad" in panel.status.cget("text")
        panel.place_btn.invoke()
        panel.escape()
        assert panel.lab.placing is None

    def test_adding_something_else_disarms_placing(self, panel):
        panel.add(kind="tail")
        panel.place_btn.invoke()
        assert panel.lab.placing == "tail-1"
        panel.add(kind="horn")                                   # the new one is selected
        assert panel.lab.placing is None and not panel._placement_var.get()
        assert panel.view.cget("cursor") == ""

    def test_clicking_an_appendage_selects_it_and_the_grid_follows(self, panel):
        doc = panel.lab.doc
        panel.add(panel.entry("membrane"))
        panel.add(kind="tail")
        panel.lab.set_view("back")
        settled(panel, panel)
        scene = panel.lab.engine.draft_scene(doc.character)
        wing = scene.shells[1].mesh
        centre = wing.vertices[wing.faces].mean(axis=1)
        px, py, _ = panel.lab.view.project(centre, panel.lab.size)
        doc.select("tail-1")
        panel.update()
        assert panel.tree.selection() == ("tail-1",)
        found = False
        for i in range(0, len(centre), 20):
            panel.view.event_generate("<ButtonPress-1>", x=int(px[i]), y=int(py[i]))
            panel.view.event_generate("<ButtonRelease-1>", x=int(px[i]), y=int(py[i]))
            panel.update()
            if doc.selected == "wing-1":
                found = True
                break
        assert found and panel.tree.selection() == ("wing-1",)
        assert ("param", "span") in panel.prop_rows                    # the grid is the wing's
        panel.view.event_generate("<ButtonPress-1>", x=3, y=3)
        panel.view.event_generate("<ButtonRelease-1>", x=3, y=3)       # background: deselect
        panel.update()
        assert doc.selected is None and panel.tree.selection() == ()

    def test_dragging_in_the_picture_orbits_and_the_markers_follow(self, panel):
        panel.add(panel.entry("membrane"))
        settled(panel, panel)
        marks_before = [panel.view.coords(i) for i in panel.view.find_withtag("mark")]
        az = panel.lab.view.azimuth
        panel.view.event_generate("<ButtonPress-1>", x=200, y=200)
        panel.view.event_generate("<B1-Motion>", x=240, y=200)
        panel.view.event_generate("<ButtonRelease-1>", x=240, y=200)
        panel.update()
        assert panel.lab.view.azimuth == pytest.approx((az + 20) % 360)
        assert [panel.view.coords(i) for i in panel.view.find_withtag("mark")] != marks_before
        assert panel.view.frame.tier in ("draft", "final")
        settled(panel, panel)

    def test_the_picture_is_refined_after_the_quiet_time(self, panel):
        panel.add(kind="tail")
        assert panel.view.frame.tier == "draft"                # the picture of the edit, at once
        settled(panel, panel)
        assert panel.view.frame.tier == "final" and panel.lab.scheduler.finals >= 1


# -- save / load / export through the panel -------------------------------------------------


@needs_assets
class TestFilesAndExport:
    def test_save_new_open_round_trip(self, panel, tmp_path):
        doc = panel.lab.doc
        panel.add(panel.entry("membrane"))
        panel.add(kind="tail")
        doc.set_control("wing-1", "fold", 0.3)
        snap = doc.snapshot()
        assert panel.save(tmp_path / "a.mmchar") == tmp_path / "a.mmchar"
        doc.new()
        panel.update()
        assert not panel.tree.get_children()
        assert panel.load(tmp_path / "a.mmchar")
        assert doc.snapshot() == snap and list(panel.tree.get_children()) == ["wing-1", "tail-1"]
        assert not doc.dirty
        assert not panel.load(tmp_path / "nope.mmchar")
        assert "no such file" in panel.status.cget("text")
        assert doc.snapshot() == snap

    def test_check_printability_shows_the_report(self, panel):
        panel.add(kind="tail")
        panel.add(kind="horn")
        panel.check_print()
        assert panel.wait_task()
        text = panel.report.get("1.0", "end")
        assert "printability" in text and "tail-1" in text and "height" in text
        assert panel.report.tag_ranges("ok") and not panel.report.tag_ranges("block")
        assert str(panel.force_btn.cget("state")) == "disabled"

    def test_the_report_is_scrolled_into_view(self, panel, tmp_path):
        panel.add(kind="tail")
        panel.update()
        assert panel.scroll.canvas.yview()[0] == 0.0
        panel.check_print()
        assert panel.wait_task()
        panel.update()
        panel.update()                                           # (the scroll is scheduled when idle)
        first, last = panel.scroll.canvas.yview()
        assert first > 0.0                                       # it moved
        canvas = panel.scroll.canvas
        top = panel.report.winfo_rooty() - canvas.winfo_rooty()
        bottom = top + panel.report.winfo_height()
        assert 0 <= top and bottom <= canvas.winfo_height()       # the whole report is on screen

    def test_export_writes_the_stl_and_shows_warnings_amber(self, panel, tmp_path):
        panel.add(kind="tail")
        panel.add(kind="horn")
        panel.export(tmp_path / "x.stl")
        assert panel.wait_task()
        assert (tmp_path / "x.stl").stat().st_size > 1000
        assert "wrote x.stl" in panel.status.cget("text")
        assert panel.report.tag_ranges("warn")                  # the horns' intersection
        assert not panel.report.tag_ranges("block")

    def test_a_blocking_report_stops_the_export_until_exported_anyway(self, panel, tmp_path):
        panel.add(kind="tail")
        panel.lab.doc.set_placement("tail-1", "offset", 3.0)
        out = tmp_path / "blocked.stl"
        panel.export(out)
        assert panel.wait_task()
        assert not out.exists()
        text = panel.report.get("1.0", "end")
        assert "BLOCK" in text and "not connected to the body" in text
        assert panel.report.tag_ranges("block")                 # in red
        assert str(panel.force_btn.cget("state")) == "normal"
        assert "blocked" in panel.status.cget("text")
        panel.force_btn.invoke()
        assert panel.wait_task()
        assert out.exists() and str(panel.force_btn.cget("state")) == "disabled"

    def test_edits_made_while_exporting_do_not_leak_into_the_file(self, panel, tmp_path):
        panel.add(kind="tail")
        panel.export(tmp_path / "frozen.stl")
        panel.lab.doc.set_param("tail-1", "length", 15.0)         # the task has its own copy
        assert panel.wait_task()
        other = tmp_path / "later.stl"
        panel.export(other)
        assert panel.wait_task()
        assert (tmp_path / "frozen.stl").read_bytes() != other.read_bytes()


# -- the same operations, three ways ---------------------------------------------------------


@needs_assets
def test_a_sequence_through_the_panel_equals_the_headless_core(panel, tmp_path):
    """add a wing, add a tail, tune, undo x2, redo, save, reload, export: the document and the
    STL are the same whether the operations come from widgets or from the core."""
    # -- through the panel
    panel.add(panel.entry("membrane"))
    panel.add(kind="tail")
    panel.lab.doc.select("wing-1")
    panel.update()
    drag(panel, panel.prop_rows[("param", "span")], 12.0)
    drag(panel, panel.control_rows["fold"], 0.3)
    panel.lab.doc.select("tail-1")
    panel.update()
    drag(panel, panel.prop_rows[("param", "length")], 8.0)
    panel.undo()
    panel.undo()
    panel.redo()
    path = tmp_path / "panel.mmchar"
    panel.save(path)
    panel.lab.doc.new()
    assert panel.load(path)
    panel.export(tmp_path / "panel.stl", force=True)
    assert panel.wait_task()
    # -- the same through the headless core (a fresh document, no widgets)
    res = L.Resources.shared()
    core = L.LabDocument(res)
    core.add_entry(next(e for e in L.add_menu() if "membrane" in e.label))
    core.add_appendage("tail")
    core.set_param("wing-1", "span", 12.0)
    core.set_control("wing-1", "fold", 0.3)
    core.set_param("tail-1", "length", 8.0)
    core.break_coalescing()
    core.undo()
    core.undo()
    core.redo()
    core_path = core.save(tmp_path / "core.mmchar")
    reloaded = L.LabDocument(res)
    reloaded.load(core_path)
    lab = L.Lab(res, character=reloaded.character, threaded=False)
    lab.export_stl(tmp_path / "core.stl", size="medium", force=True)
    lab.close()
    a = json.loads((tmp_path / "panel.mmchar").read_text())
    b = json.loads((tmp_path / "core.mmchar").read_text())
    assert a == b
    assert (tmp_path / "panel.stl").read_bytes() == (tmp_path / "core.stl").read_bytes()
    # the last edit (the tail's length) was undone by the first undo; the fold was not
    assert a["appendages"][0]["params"]["span"] == 12.0
    assert a["appendages"][0]["controls"] == {"*": {"fold": 0.3}}
    assert "length" not in a["appendages"][1]["params"]


# -- life cycle ------------------------------------------------------------------------------


@needs_assets
class TestLifeCycle:
    def test_destroying_the_panel_stops_the_poll_and_the_worker(self, root):
        import threading
        from minimaster.character_panel import CharacterPanel
        p = CharacterPanel(root, view_parent=root, resources=L.Resources.shared())
        p.pack()
        p.view.pack()
        root.update()
        p.activate()
        p.add(kind="tail")
        pump(root, until=lambda: p.lab.scheduler.finals >= 1)
        assert any(t.name == "lab-final" for t in threading.enumerate())
        p.destroy()
        assert not any(t.name == "lab-final" and t.is_alive() for t in threading.enumerate())

    def test_a_tab_switch_during_a_render_is_harmless(self, app):
        panel = app.character_panel
        app.notebook.select(5)
        app.update()
        panel.add(panel.entry("feathered"))
        app.notebook.select(0)                                  # the worker is still rendering
        pump(app, 0.5)
        app.notebook.select(5)
        settled(app, panel)
        assert panel.view.frame.tier == "final"

    def test_closing_the_window_with_a_render_in_flight(self, app):
        panel = app.character_panel
        app.notebook.select(5)
        app.update()
        panel.add(panel.entry("feathered"))
        pump(app, 0.2)                                          # the final is probably running
        # the app fixture destroys the window; the worker is a daemon and is told to stop
        assert panel.lab.scheduler._worker is not None
