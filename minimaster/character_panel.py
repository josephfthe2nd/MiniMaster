"""The "Character" tab: add, remove and tune appendages on a realistic character (needs tkinter).

The panel is thin. Every decision (what an edit means, what it is allowed to be, undo, the
preview tiers, the export and its printability report) lives in the tk-free
:mod:`minimaster.character.lab`; this module turns widgets into calls on a
:class:`~minimaster.character.lab.Lab` and shows what comes back:

* **Body** -- the macro sliders (gender, age, muscle, weight, height, ethnicity) and a few
  common shape sliders;
* **Appendages** -- the list, Add (a menu generated from the kind registry), Remove, Duplicate;
* **the selected appendage** -- a property grid GENERATED from its kind: name, enabled, side,
  placement (preset, scale / roll / offset, "place by clicking the body"), its ``ParamSpec``
  parameters (the ones whose ``when=`` holds), and its pose controls with the live range of the
  PLACED appendage;
* **Body adaptation** -- the switch, its cap and what it did to the body;
* **File / Print** -- Save / Load ``.mmchar``, Export STL with the printability report
  (blocking findings in red stop the export until "Export anyway", warnings in amber).

The picture lives in an :class:`~minimaster.character.imageview.ImageView`. How it shares the
window with the studio's primitive Viewport: the app swaps the LEFT pane while this tab is
selected (``MiniMasterApp._swap_stage``), so no studio widget moves and no studio test notices.

The base mesh and the morph targets load when the tab is first shown, not at start-up; without
the CC0 assets the tab says so and the rest of the app is unaffected.
"""

from __future__ import annotations

import math
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

RED = "#b03a2e"
AMBER = "#a76400"
GREEN = "#1e7a3c"
DIM = "#6b7280"

# a few common shape sliders next to the macros (those that exist in the catalog are shown)
COMMON_SLIDERS = (
    "torso.torso_vshape", "torso.torso_scale_horiz", "torso.torso_muscle_pectoral",
    "torso.torso_muscle_dorsi", "measure.measure_shoulder_dist", "neck.neck_scale_vert",
    "head.head_scale_horiz", "hip.hip_scale_horiz", "armslegs.hand_scale", "armslegs.foot_scale",
)
MACRO_LABELS = (("gender", "Gender (f - m)"), ("age", "Age"), ("muscle", "Muscle"),
                ("weight", "Weight"), ("height", "Height"), ("african", "African"),
                ("asian", "Asian"), ("caucasian", "Caucasian"))
POLL_MS = 15


# -- tk variables that are safe to be garbage ----------------------------------------------------
#
# ``tk.Variable.__del__`` calls into Tcl. The cyclic collector runs in whichever thread happens to
# allocate enough, and the preview worker allocates a lot; a variable orphaned by a rebuilt
# property grid would then be finalised ON THE WORKER, which either raises "main thread is not in
# main loop" after waiting a second per object or stalls the worker until the UI loop services
# the call. These variables defer their finaliser to the UI thread instead.

_UI_THREAD = [None]
_DEFERRED: list = []


class _SafeVar:
    def __del__(self):
        if _UI_THREAD[0] is None or threading.get_ident() == _UI_THREAD[0]:
            super().__del__()
        else:
            _DEFERRED.append(self)       # finalised once; the UI thread finishes the job


class StringVar(_SafeVar, tk.StringVar):
    pass


class BooleanVar(_SafeVar, tk.BooleanVar):
    pass


class DoubleVar(_SafeVar, tk.DoubleVar):
    pass


def drain_deferred() -> int:
    """Finish the finalisers that other threads deferred (call on the UI thread)."""
    n = 0
    while _DEFERRED:
        tk.Variable.__del__(_DEFERRED.pop())
        n += 1
    return n


def _fmt(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.3g}"


class SliderRow:
    """label | slider | typed value, for one number. ``on_change(value)`` returns the value the
    document stored (a typed value is clamped), which the row then shows; ``on_press`` /
    ``on_release`` bracket a drag (one undo step)."""

    def __init__(self, parent, label, lo, hi, value, *, step=None, integer=False,
                 on_change=None, on_press=None, on_release=None, width=17, echo=None):
        self.frame = ttk.Frame(parent)
        self.echo = echo                        # () -> the stored value, for a setter that returns none
        self.integer = integer
        self.step = 1.0 if integer else step
        self.on_change = on_change
        self._syncing = True
        self._last = None
        self.label = ttk.Label(self.frame, text=label, width=width, anchor="w")
        self.label.grid(row=0, column=0, sticky="w")
        self.var = DoubleVar(value=value)
        self.scale = ttk.Scale(self.frame, from_=lo, to=hi, variable=self.var,
                               command=self._moved)
        self.scale.grid(row=0, column=1, sticky="ew", padx=3)
        self.text = StringVar(value=_fmt(value))
        self.entry = ttk.Entry(self.frame, textvariable=self.text, width=7)
        self.entry.grid(row=0, column=2)
        self.frame.columnconfigure(1, weight=1)
        self.entry.bind("<Return>", self._typed)
        self.entry.bind("<FocusOut>", self._typed)
        if on_press:
            self.scale.bind("<ButtonPress-1>", lambda e: on_press(), add="+")
        if on_release:
            self.scale.bind("<ButtonRelease-1>", lambda e: on_release(), add="+")
        self._syncing = False
        self._last = self._snap(value)

    def _snap(self, v: float):
        if self.integer:
            return int(round(v))
        if self.step:
            return round(round(v / self.step) * self.step, 6)
        return float(v)

    def _moved(self, raw) -> None:
        if self._syncing:
            return
        v = self._snap(float(raw))
        if v == self._last:
            return
        self._apply(v)

    def _typed(self, _e=None) -> None:
        if self._syncing:
            return
        try:
            v = float(self.text.get())
            if not math.isfinite(v):
                raise ValueError("not finite")
            v = int(round(v)) if self.integer else v
        except (ValueError, OverflowError):
            self.text.set(_fmt(self._last))       # nan / inf / 1e999 / text: keep what is stored
            return
        self._apply(v)

    def _apply(self, v) -> None:
        self._last = v
        self.text.set(_fmt(v))
        stored = self.on_change(v) if self.on_change else v
        if stored is not None and stored != v:
            self.set(stored)
        elif stored is None and self.echo is not None:
            shown = self.echo()                   # a row whose setter returns nothing: ask
            if shown is not None and shown != v:
                self.set(shown)

    def set(self, value, lo=None, hi=None) -> None:
        """Show ``value`` (and a new range) without calling ``on_change``."""
        self._syncing = True
        try:
            if lo is not None and hi is not None:
                self.scale.configure(from_=lo, to=hi)
            self.var.set(float(value))
            self.text.set(_fmt(value))
            self._last = self._snap(float(value))
        finally:
            self._syncing = False

    @property
    def value(self) -> float:
        return float(self.var.get())

    def grid(self, **kw):
        self.frame.grid(**kw)


class ScrolledFrame(ttk.Frame):
    """A vertically scrollable frame (the wheel scrolls while the pointer is over it)."""

    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.bar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.bar.pack(side="right", fill="y")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win,
                                                                            width=e.width))
        for w in (self.canvas, self.inner):
            w.bind("<Enter>", self._enter, add="+")
            w.bind("<Leave>", self._leave, add="+")

    def reveal(self, widget) -> None:
        """Scroll so that ``widget`` (inside the frame) is in view, its bottom at the bottom."""
        self.update_idletasks()
        total = self.inner.winfo_height()
        view = self.canvas.winfo_height()
        if total <= view:
            return
        bottom = widget.winfo_rooty() - self.inner.winfo_rooty() + widget.winfo_height()
        top = min(max(bottom - view + 6, 0), total - view)
        self.canvas.yview_moveto(top / total)

    def _enter(self, _e):
        self.canvas.bind_all("<MouseWheel>", self._wheel)
        self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-2, "units"))
        self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(2, "units"))

    def _leave(self, _e):
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.unbind_all(seq)

    def _wheel(self, e):
        self.canvas.yview_scroll(-1 if e.delta > 0 else 1, "units")


class CharacterPanel(ttk.Frame):
    """The Character tab. ``app`` is the studio window (or ``None`` in tests); ``view_parent`` is
    the widget the image view is created in (default: the app)."""

    def __init__(self, master, app=None, view_parent=None, resources=None):
        super().__init__(master)
        _UI_THREAD[0] = threading.get_ident()
        self.app = app
        self._view_parent = view_parent
        self._resources = resources
        self.lab = None
        self._view = None
        self._built = False
        self._active = False
        self._poll_job = None
        self._syncing = False
        self._busy = False                 # a widget is editing: do not echo the change back
        self._task = None                  # the export / check running on a thread
        self._task_label = ""
        self._last_result = None
        self._message = ""
        self._message_error = False
        self.ok = True                     # did the last edit go through?
        self._notes_dirty = False
        self._notes_text = None
        self._button_state = None
        self._pushed_status = ""
        self.body_rows: dict = {}
        self.prop_rows: dict = {}
        self.control_rows: dict = {}
        self._menu_entries: list = []
        self._placement_var = BooleanVar(value=False)
        self.pose_side = StringVar(value="both")
        ttk.Label(self, text="Loading the character lab ...").pack(pady=20)

    # -- life cycle --------------------------------------------------------------------------

    @property
    def view(self):
        """The image view (created on first use; the app packs it in the left pane)."""
        if self._view is None:
            from .character.imageview import ImageView
            self._view = ImageView(self._view_parent or self.winfo_toplevel())
        return self._view

    @property
    def existing_view(self):
        """The image view if it was ever created, else ``None`` (never creates one)."""
        return self._view

    def activate(self):
        """The tab was selected: build on first use, start the poll loop, draw. Returns the view."""
        view = self.view
        if not self._built:
            self._build()
        if self.lab is not None:
            self._active = True
            if self._poll_job is None:
                self._poll()
            view.update_idletasks()
            w, h = view.size
            if w > 40 and h > 40:
                self.lab.size = (w, h)
            self.lab.refresh()
        return view

    def deactivate(self) -> None:
        self._active = False
        if self._poll_job is not None:
            try:
                self.after_cancel(self._poll_job)
            except tk.TclError:
                pass
            self._poll_job = None

    def destroy(self) -> None:
        self.deactivate()
        if self.lab is not None:
            self.lab.on_frame = self.lab.on_change = self.lab.on_error = None
            self.lab.close()
        super().destroy()
        drain_deferred()

    def sync(self) -> None:
        """Studio panels implement ``sync``; this one follows the character document instead."""

    # -- building ----------------------------------------------------------------------------

    def _build(self) -> None:
        for child in self.winfo_children():
            child.destroy()
        self._built = True
        from .character import lab as L
        if not L.Resources.available():
            ttk.Label(self, wraplength=300, justify="left", padding=8, text=(
                "The CC0 MakeHuman assets are not installed, so the realistic Character Lab "
                "is not available.\n\nRun:  python tools/fetch_makehuman_assets.py")).pack()
            self.view.banner = "MakeHuman assets not installed"
            self.view.refresh_overlay()
            return
        self.L = L
        # the final preview and the export mesh the body at level 1 when a root collar needs it (Lab.final_levels);
        # the draft is the proxy either way
        self.lab = L.Lab(self._resources, size=(640, 560), levels=None)
        self.lab.on_change = self._on_change
        self.lab.on_error = lambda exc: self._say(f"preview failed: {exc}", error=True)
        self.view.bind_lab(self.lab)
        self.view.on_click = self._clicked
        self.view.on_cancel = self.cancel_placing
        self.lab.frame_content(refresh=False)

        self.scroll = ScrolledFrame(self)
        self.scroll.pack(fill="both", expand=True)
        root = self.scroll.inner
        self._build_top(root)
        self._build_appendages(root)
        self.props = ttk.LabelFrame(root, text="Selected appendage", padding=4)
        self.props.pack(fill="x", padx=4, pady=3)
        self._build_body(root)
        self._build_adaptation(root)
        self._build_file_print(root)
        self.status = ttk.Label(self, text="", anchor="w", wraplength=320, padding=(6, 3))
        self.status.pack(side="bottom", fill="x")
        self._rebuild_props()
        self._refresh_list()
        self._sync_body()
        self._sync_buttons()

    def _build_top(self, root) -> None:
        bar = ttk.Frame(root)
        bar.pack(fill="x", padx=4, pady=(4, 0))
        self.undo_btn = ttk.Button(bar, text="Undo", width=4, command=self.undo)
        self.undo_btn.pack(side="left")
        self.redo_btn = ttk.Button(bar, text="Redo", width=4, command=self.redo)
        self.redo_btn.pack(side="left", padx=2)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=4)
        for name, label in (("front", "Front"), ("back", "Back"), ("left", "Left")):
            ttk.Button(bar, text=label, width=4,
                       command=lambda n=name: self.lab.set_view(n)).pack(side="left", padx=1)
        ttk.Button(bar, text="Frame", width=5,
                   command=lambda: self.lab.frame_content()).pack(side="left", padx=1)

    def _build_body(self, root) -> None:
        lf = ttk.LabelFrame(root, text="Body", padding=4)
        lf.pack(fill="x", padx=4, pady=3)
        doc, cat = self.lab.doc, self.lab.resources.catalog
        for key, label in MACRO_LABELS:
            row = SliderRow(lf, label, 0.0, 1.0, doc.character.macro.get(key, 0.5), step=0.01,
                            on_change=lambda v, k=key: self._edit(doc.set_macro, k, v, clamp=True),
                            on_press=doc.begin_gesture, on_release=doc.end_gesture)
            row.grid(row=len(self.body_rows), column=0, sticky="ew")
            self.body_rows[("macro", key)] = row
        lf.columnconfigure(0, weight=1)
        self.shape_open = BooleanVar(value=False)
        ttk.Checkbutton(lf, text="more shape sliders", variable=self.shape_open,
                        style="Toolbutton", command=self._toggle_shape).grid(
                            row=len(self.body_rows), column=0, sticky="w", pady=(4, 0))
        sub = ttk.LabelFrame(lf, text="Shape", padding=2)
        self.shape_frame = sub
        sub.grid(row=len(self.body_rows) + 1, column=0, sticky="ew", pady=(2, 0))
        sub.grid_remove()
        sub.columnconfigure(0, weight=1)
        n = 0
        for key in COMMON_SLIDERS:
            if key not in cat:
                continue
            sl = cat[key]
            row = SliderRow(sub, sl.label[:20], sl.lo, sl.hi, doc.character.sliders.get(key, 0.0),
                            step=0.01,
                            on_change=lambda v, k=key: self._edit(doc.set_slider, k, v, clamp=True),
                            on_press=doc.begin_gesture, on_release=doc.end_gesture)
            row.grid(row=n, column=0, sticky="ew")
            self.body_rows[("slider", key)] = row
            n += 1
        ttk.Button(lf, text="Reset body", command=self.reset_body).grid(
            row=len(self.body_rows) + 2, column=0, sticky="e", pady=(4, 0))

    def _toggle_shape(self) -> None:
        if self.shape_open.get():
            self.shape_frame.grid()
        else:
            self.shape_frame.grid_remove()

    def _build_appendages(self, root) -> None:
        lf = ttk.LabelFrame(root, text="Appendages", padding=4)
        lf.pack(fill="x", padx=4, pady=3)
        self.tree = ttk.Treeview(lf, columns=("kind", "side", "on"), height=5,
                                 selectmode="browse")
        self.tree.heading("#0", text="name")
        self.tree.heading("kind", text="kind")
        self.tree.heading("side", text="side")
        self.tree.heading("on", text="on")
        self.tree.column("#0", width=110, stretch=True)
        self.tree.column("kind", width=60, stretch=False)
        self.tree.column("side", width=55, stretch=False)
        self.tree.column("on", width=30, stretch=False, anchor="center")
        self.tree.pack(fill="x")
        self.tree.bind("<<TreeviewSelect>>", self._tree_selected)
        self.tree.bind("<Delete>", lambda e: self.remove())
        bar = ttk.Frame(lf)
        bar.pack(fill="x", pady=(4, 0))
        self.add_button = ttk.Menubutton(bar, text="Add  ▾", direction="below")
        menu = tk.Menu(self.add_button, tearoff=0)
        self.add_button["menu"] = menu
        self._menu_entries = self.L.add_menu()
        for entry in self._menu_entries:
            menu.add_command(label=entry.label, command=lambda e=entry: self.add(e))
        self.add_menu = menu
        self.add_button.pack(side="left")
        ttk.Button(bar, text="Remove", command=self.remove).pack(side="left", padx=3)
        ttk.Button(bar, text="Duplicate", command=self.duplicate).pack(side="left")

    def _build_adaptation(self, root) -> None:
        lf = ttk.LabelFrame(root, text="Body adaptation", padding=4)
        lf.pack(fill="x", padx=4, pady=3)
        self.adapt_var = BooleanVar(value=True)
        ttk.Checkbutton(lf, text="Adapt the body to the appendages",
                        variable=self.adapt_var, command=self._adapt_toggled).pack(anchor="w")
        self.junction_var = BooleanVar(value=True)
        ttk.Checkbutton(lf, text="Skin flows into the appendages (fillet)",
                        variable=self.junction_var,
                        command=self._junction_toggled).pack(anchor="w")
        self.cap_row = SliderRow(lf, "Cap", 0.0, 2.0, 1.0, step=0.05,
                                 on_change=lambda v: self._edit(self.lab.doc.set_anatomy,
                                                                cap=v, clamp=True),
                                 on_press=self.lab.doc.begin_gesture,
                                 on_release=self.lab.doc.end_gesture, width=6,
                                 echo=lambda: float(
                                     self.lab.doc.character.anatomy.get("cap", 1.0)))
        self.cap_row.frame.pack(fill="x")
        self.notes_label = ttk.Label(lf, text="", wraplength=300, justify="left", foreground=DIM)
        self.notes_label.pack(anchor="w", fill="x")

    def _build_file_print(self, root) -> None:
        lf = ttk.LabelFrame(root, text="File", padding=4)
        lf.pack(fill="x", padx=4, pady=3)
        for text, cmd in (("New", self.new_dialog), ("Open", self.load_dialog),
                          ("Save", self.save), ("Save as", self.save_dialog)):
            ttk.Button(lf, text=text, width=7, command=cmd).pack(side="left", padx=1)
        pf = ttk.LabelFrame(root, text="Print", padding=4)
        pf.pack(fill="x", padx=4, pady=3)
        from .character.printprep import SIZE_PRESETS
        row = ttk.Frame(pf)
        row.pack(fill="x")
        ttk.Label(row, text="size").pack(side="left")
        self.size_var = StringVar(value="medium")
        ttk.Combobox(row, textvariable=self.size_var, values=sorted(SIZE_PRESETS, key=SIZE_PRESETS.get),
                     state="readonly", width=8).pack(side="left", padx=4)
        self.base_var = BooleanVar(value=True)
        ttk.Checkbutton(row, text="base", variable=self.base_var).pack(side="left")
        self.fit_var = BooleanVar(value=False)
        ttk.Checkbutton(row, text="widen base", variable=self.fit_var).pack(side="left", padx=4)
        row2 = ttk.Frame(pf)
        row2.pack(fill="x", pady=(4, 0))
        ttk.Button(row2, text="Check printability", command=self.check_print).pack(side="left")
        ttk.Button(row2, text="Export STL ...", command=self.export_dialog).pack(side="left", padx=3)
        row3 = ttk.Frame(pf)
        row3.pack(fill="x", pady=(2, 0))
        self.force_btn = ttk.Button(row3, text="Export anyway (the report blocks it)",
                                    command=self.export_forced, state="disabled")
        self.force_btn.pack(side="left")
        self.report = tk.Text(pf, height=9, width=38, wrap="word", state="disabled",
                              font=("TkFixedFont", 8), borderwidth=1, relief="sunken")
        self.report.pack(fill="x", pady=(4, 0))
        for tag, color in (("block", RED), ("warn", AMBER), ("ok", GREEN), ("dim", DIM)):
            self.report.tag_configure(tag, foreground=color)
        self.report.tag_configure("head", font=("TkFixedFont", 8, "bold"))

    # -- the selected appendage: a grid generated from its kind --------------------------------

    def _rebuild_props(self) -> None:
        for child in self.props.winfo_children():
            child.destroy()
        self.prop_rows.clear()
        self.control_rows.clear()
        doc = self.lab.doc
        spec = doc.character.spec(doc.selected) if doc.selected else None
        if spec is None:
            ttk.Label(self.props, text="Add an appendage, or click one in the picture.",
                      foreground=DIM, wraplength=300).pack(anchor="w")
            return
        from .character import kinds
        kind = kinds.kind_of(spec)
        if kind is None:
            ttk.Label(self.props, wraplength=300, foreground=AMBER, text=(
                f"{spec.id} is a {spec.kind!r}, a kind this program does not know. It stays in "
                "the file untouched but cannot be edited here.")).pack(anchor="w")
            return
        self._prop_header(spec, kind)
        self._prop_placement(spec, kind)
        self._prop_params(spec, kind)
        self._prop_controls(spec, kind)

    def _prop_header(self, spec, kind) -> None:
        doc = self.lab.doc
        f = ttk.Frame(self.props)
        f.pack(fill="x")
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text=f"{kind.label} ({spec.id})", font=("TkDefaultFont", 9, "bold")).grid(
            row=0, column=0, sticky="w")
        self.name_var = StringVar(value=spec.name)
        e = ttk.Entry(f, textvariable=self.name_var, width=12)
        e.grid(row=0, column=1, sticky="ew", padx=4)
        self.name_entry = e
        for seq in ("<Return>", "<FocusOut>"):
            e.bind(seq, lambda ev, sid=spec.id: self._rename(sid))
        self.enabled_var = BooleanVar(value=spec.enabled)
        self.enabled_check = ttk.Checkbutton(
            f, text="on", variable=self.enabled_var,
            command=lambda: self._edit(doc.set_enabled, spec.id, self.enabled_var.get()))
        self.enabled_check.grid(row=0, column=2)
        ttk.Label(f, text="side").grid(row=1, column=0, sticky="w")
        self.side_var = StringVar(value=spec.side)
        sides = tuple(s for s in self.L.SIDES if s in getattr(kind, "allowed_sides", self.L.SIDES))
        cb = ttk.Combobox(f, textvariable=self.side_var, values=sides or self.L.SIDES, state="readonly",
                          width=8)
        cb.grid(row=1, column=1, sticky="w", padx=4, pady=2)
        cb.bind("<<ComboboxSelected>>",
                lambda ev: self._edit(doc.set_side, spec.id, self.side_var.get()))
        self.prop_rows["side"] = cb

    def _prop_placement(self, spec, kind) -> None:
        doc = self.lab.doc
        lf = ttk.LabelFrame(self.props, text="Placement", padding=3)
        lf.pack(fill="x", pady=(4, 0))
        lf.columnconfigure(0, weight=1)
        if len(kind.presets) > 1:
            self.preset_var = StringVar(value=spec.placement.get("preset", ""))
            cb = ttk.Combobox(lf, textvariable=self.preset_var, values=list(kind.presets),
                              state="readonly")
            cb.grid(row=0, column=0, sticky="ew")
            cb.bind("<<ComboboxSelected>>", lambda ev: self._edit(
                doc.set_placement_preset, spec.id, self.preset_var.get()))
            self.prop_rows["preset"] = cb
        else:
            ttk.Label(lf, foreground=DIM, text="preset: " + next(
                (v.get("label", k) for k, v in kind.presets.items()), "")).grid(
                    row=0, column=0, sticky="w")
        self.place_btn = tk.Checkbutton(
            lf, text="Place by clicking the body", variable=self._placement_var,
            indicatoron=False, command=self._placing_toggled, selectcolor="#ffe08a",
            padx=6, pady=2)
        self.place_btn.grid(row=1, column=0, sticky="ew", pady=2)
        row = 2
        for key in ("scale", "roll", "offset"):
            if key not in kind.placement_keys:
                continue
            lo, hi = self.L.PLACEMENT_RANGES[key]
            default = {"scale": 1.0, "roll": 0.0, "offset": 0.0}[key]
            r = SliderRow(lf, key, lo, hi, spec.placement.get(key, default), step=0.01,
                          on_change=lambda v, k=key: self._edit(doc.set_placement, spec.id, k, v,
                                                                clamp=True),
                          on_press=doc.begin_gesture, on_release=doc.end_gesture, width=8)
            r.grid(row=row, column=0, sticky="ew")
            self.prop_rows[("placement", key)] = r
            row += 1

    def _prop_params(self, spec, kind) -> None:
        doc = self.lab.doc
        rows = doc.param_specs(spec.id)
        if not rows:
            return
        lf = ttk.LabelFrame(self.props, text="Shape", padding=3)
        lf.pack(fill="x", pady=(4, 0))
        lf.columnconfigure(0, weight=1)
        gates = {p.when[0] for p in kind.params if p.when}
        for n, (p, value) in enumerate(rows):
            rebuild = p.key in gates
            if p.type == "bool":
                var = BooleanVar(value=bool(value))
                w = ttk.Checkbutton(lf, text=p.label, variable=var,
                                    command=lambda p=p, var=var, rb=rebuild: self._set_param(
                                        spec.id, p.key, var.get(), rb))
                w.grid(row=n, column=0, sticky="w")
                self.prop_rows[("param", p.key)] = var
            elif p.type == "choice":
                f = ttk.Frame(lf)
                f.grid(row=n, column=0, sticky="ew")
                ttk.Label(f, text=p.label, width=17, anchor="w").pack(side="left")
                var = StringVar(value=str(value))
                cb = ttk.Combobox(f, textvariable=var, values=list(p.choices), state="readonly",
                                  width=12)
                cb.pack(side="left")
                cb.bind("<<ComboboxSelected>>", lambda ev, p=p, var=var, rb=rebuild:
                        self._set_param(spec.id, p.key, var.get(), rb))
                self.prop_rows[("param", p.key)] = var
            else:
                integer = p.type == "int"
                step = 1 if integer else (p.step or (p.hi - p.lo) / 100.0)
                r = SliderRow(lf, p.label[:20], p.lo, p.hi, value, step=step, integer=integer,
                              on_change=lambda v, p=p, rb=rebuild: self._set_param(
                                  spec.id, p.key, v, rb),
                              on_press=doc.begin_gesture, on_release=doc.end_gesture)
                r.grid(row=n, column=0, sticky="ew")
                self.prop_rows[("param", p.key)] = r

    def _prop_controls(self, spec, kind) -> None:
        if not kind.controls:
            return
        from .character import kinds
        doc = self.lab.doc
        lf = ttk.LabelFrame(self.props, text="Pose", padding=3)
        lf.pack(fill="x", pady=(4, 0))
        lf.columnconfigure(0, weight=1)
        top = ttk.Frame(lf)
        top.grid(row=0, column=0, sticky="ew")
        presets = self.L.pose_presets(kind)
        if presets:
            ttk.Label(top, text="pose").pack(side="left")
            self.pose_var = StringVar(
                value=str(kinds.side_controls(spec.controls, "").get("preset", "")))
            cb = ttk.Combobox(top, textvariable=self.pose_var, values=("",) + tuple(presets),
                              state="readonly", width=12)
            cb.pack(side="left", padx=4)
            cb.bind("<<ComboboxSelected>>", lambda ev: self._set_control(
                spec.id, "preset", self.pose_var.get() or None))
        if spec.side == "both":
            ttk.Label(top, text="side").pack(side="left", padx=(6, 0))
            cb = ttk.Combobox(top, textvariable=self.pose_side, values=("both", "l", "r"),
                              state="readonly", width=5)
            cb.pack(side="left", padx=4)
            cb.bind("<<ComboboxSelected>>", lambda ev: self._sync_controls(spec.id))
        else:
            self.pose_side.set("both")
        letter = self._pose_letter()
        for n, c in enumerate(kind.controls, start=1):
            rng = doc.control_range(spec.id, c.key) or (-1.0, 1.0)
            state = kinds.side_controls(spec.controls, letter)
            r = SliderRow(lf, c.label, rng[0], rng[1], state.get(c.key, c.default), step=0.01,
                          on_change=lambda v, k=c.key: self._set_control(spec.id, k, v),
                          on_press=doc.begin_gesture, on_release=doc.end_gesture, width=12)
            r.grid(row=n, column=0, sticky="ew")
            self.control_rows[c.key] = r

    def _pose_letter(self) -> str | None:
        side = self.pose_side.get()
        return side if side in ("l", "r") else None

    def _sync_controls(self, spec_id: str) -> None:
        """Show the controls of the chosen side with the live range of the placed appendage."""
        from .character import kinds
        doc = self.lab.doc
        spec = doc.character.spec(spec_id)
        if spec is None:
            return
        letter = self._pose_letter()
        state = kinds.side_controls(spec.controls, letter or "")
        kind = kinds.kind_of(spec)
        for c in kind.controls:
            row = self.control_rows.get(c.key)
            if row is None:
                continue
            rng = doc.control_range(spec_id, c.key, letter) or (-1.0, 1.0)
            row.set(state.get(c.key, c.default), rng[0], rng[1])

    # -- edits (widget -> lab.doc), all through _edit so errors become status text -----------------

    def _say(self, text: str, error: bool = False) -> None:
        # the status line is a line: the full text of a long refusal is in the report below
        text = text if len(text) <= 140 else text[:137] + "..."
        self._message, self._message_error = text, error
        if self._built and self.lab is not None:
            self.status.configure(text=text, foreground=RED if error else "")
        if self.app is not None and self._active:
            self._push_status()

    def status_text(self) -> str:
        if self.lab is None:
            return "Character lab unavailable (MakeHuman assets missing)"
        doc = self.lab.doc
        bits = [f"character: {doc.name}{'*' if doc.dirty else ''}",
                f"{len(doc.specs)} appendages"]
        if self.lab.placing:
            bits.append(f"placing {self.lab.placing}: click the body (Esc cancels)")
        elif self._message:
            bits.append(self._message)
        else:
            bits.append("drag=orbit  mid/right-drag=pan  wheel=zoom  click=select")
        return "   |   ".join(bits)

    def _edit(self, fn, *args, **kw):
        """Run a document edit; a refused edit becomes a red status line, never a traceback."""
        self._busy = True
        self.ok = False
        try:
            result = fn(*args, **kw)
        except self.L.LabError as exc:
            self._say(str(exc), error=True)
            return None
        finally:
            self._busy = False
        self.ok = True
        self._say("")
        return result

    def _set_param(self, spec_id, key, value, rebuild_grid):
        stored = self._edit(self.lab.doc.set_param, spec_id, key, value, clamp=True)
        if rebuild_grid:
            self.after_idle(self._rebuild_props)         # never destroy a widget in its own callback
        return stored

    def _set_control(self, spec_id, name, value):
        stored = self._edit(self.lab.doc.set_control, spec_id, name, value,
                            side=self._pose_letter(), clamp=True)
        if name != "preset":
            self._sync_other_ranges(spec_id, name)
        return stored

    def _sync_other_ranges(self, spec_id, changed) -> None:
        """A control's range depends on the others (a folded wing rises less): refresh them."""
        doc = self.lab.doc
        letter = self._pose_letter()
        for key, row in self.control_rows.items():
            if key == changed:
                continue
            rng = doc.control_range(spec_id, key, letter)
            if rng is not None:
                row.scale.configure(from_=rng[0], to=rng[1])

    def _rename(self, spec_id) -> None:
        if self.lab.doc.character.spec(spec_id) is not None:
            self._edit(self.lab.doc.rename_appendage, spec_id, self.name_var.get())

    # -- commands (buttons, menus, keys; tests call these directly) ------------------------------

    def add(self, entry=None, kind: str | None = None, **kw):
        """Add an appendage: an Add-menu entry, or a kind id (``panel.add(kind="tail")``)."""
        if entry is not None:
            spec = self._edit(self.lab.doc.add_entry, entry, **kw)
        else:
            spec = self._edit(self.lab.doc.add_appendage, kind, **kw)
        if spec is not None:
            self._say(f"added {spec.id}")
        return spec

    def entry(self, label_part: str):
        """The Add-menu entry whose label contains ``label_part`` (case-insensitive)."""
        for e in self._menu_entries:
            if label_part.lower() in e.label.lower():
                return e
        raise KeyError(label_part)

    def remove(self, spec_id: str | None = None) -> None:
        spec_id = spec_id or self.lab.doc.selected
        if spec_id is not None:
            self._edit(self.lab.doc.remove_appendage, spec_id)

    def duplicate(self, spec_id: str | None = None):
        spec_id = spec_id or self.lab.doc.selected
        return None if spec_id is None else self._edit(self.lab.doc.duplicate_appendage, spec_id)

    def undo(self) -> bool:
        if self.lab is None:
            return False
        done = self.lab.doc.undo()
        self._say("undo" if done else "nothing to undo")
        return done

    def redo(self) -> bool:
        if self.lab is None:
            return False
        done = self.lab.doc.redo()
        self._say("redo" if done else "nothing to redo")
        return done

    def reset_body(self) -> None:
        self._edit(self.lab.doc.reset_body)
        self._sync_body()

    def escape(self) -> None:
        self.cancel_placing()

    def _adapt_toggled(self) -> None:
        self._edit(self.lab.doc.set_anatomy, enabled=self.adapt_var.get())

    def _junction_toggled(self) -> None:
        self._edit(self.lab.doc.set_anatomy, junction=self.junction_var.get())

    def _tree_selected(self, _e=None) -> None:
        if self._syncing:
            return
        sel = self.tree.selection()
        self.lab.doc.select(sel[0] if sel else None)

    # -- placing and clicking ------------------------------------------------------------------

    def _placing_toggled(self) -> None:
        if self._placement_var.get() and self.lab.doc.selected:
            self.lab.start_placing(self.lab.doc.selected)
            self.view.configure(cursor="crosshair")
            self.view.banner = "click the body to place " + self.lab.doc.selected
            self._say("click the body to place the appendage (Esc cancels)")
        else:
            self.cancel_placing()
        self.view.refresh_overlay()

    def cancel_placing(self) -> None:
        if self.lab is None:
            return
        self.lab.cancel_placing()
        self._placement_var.set(False)
        self.view.configure(cursor="")
        self.view.banner = ""
        self.view.refresh_overlay()

    def _clicked(self, x, y) -> None:
        was_placing = self.lab.placing is not None
        self._busy = True
        try:
            message = self.lab.click(x, y)
        except self.L.LabError as exc:
            message = str(exc)
            self._say(message, error=True)
            message = None
        finally:
            self._busy = False
        if message is not None:
            self._say(message)
        if was_placing and self.lab.placing is None:
            self.cancel_placing()

    # -- files ---------------------------------------------------------------------------------

    def _confirm_discard(self) -> bool:
        if not self.lab.doc.dirty:
            return True
        return messagebox.askokcancel("Character", "Discard the unsaved changes?", parent=self)

    def new_dialog(self) -> None:
        if self._confirm_discard():
            self.lab.doc.new()
            self._say("new character")

    def load_dialog(self) -> None:
        if not self._confirm_discard():
            return
        path = filedialog.askopenfilename(parent=self, filetypes=[("Character", "*.mmchar"),
                                                                  ("All", "*.*")])
        if path:
            self.load(path)

    def load(self, path) -> bool:
        self._edit(self.lab.doc.load, path)
        if not self.ok:
            return False
        self.lab.frame_content()
        self._say(f"opened {Path(path).name}")
        return True

    def save(self, path=None):
        if path is None and self.lab.doc.path is None:
            return self.save_dialog()
        target = self._edit(self.lab.doc.save, path)
        if target is not None:
            self._say(f"saved {target}")
        return target

    def save_dialog(self):
        path = filedialog.asksaveasfilename(parent=self, defaultextension=".mmchar",
                                            filetypes=[("Character", "*.mmchar")])
        return self.save(path) if path else None

    # -- print ---------------------------------------------------------------------------------

    def _export_options(self) -> dict:
        return dict(size=self.size_var.get(), with_base=self.base_var.get(),
                    fit_base=self.fit_var.get())

    def check_print(self) -> None:
        """Measure printability (an idle-time job on a thread; the report appears below)."""
        self._start_task(self.lab.export_async(None, **self._export_options()), "check")

    def export_dialog(self) -> None:
        path = filedialog.asksaveasfilename(parent=self, defaultextension=".stl",
                                            filetypes=[("STL", "*.stl")])
        if path:
            self.export(path)

    def export(self, path, force: bool = False) -> None:
        """Write the STL on a thread; blocking findings stop it unless ``force``."""
        self._export_path = path
        self._start_task(self.lab.export_async(path, force=force, **self._export_options()),
                         "export")

    def export_forced(self) -> None:
        if getattr(self, "_export_path", None):
            self.export(self._export_path, force=True)

    def _start_task(self, task, label: str) -> None:
        if self._task is not None and not self._task.done:
            self._say("still working on the last request ...")
            return
        self._task, self._task_label = task, label
        self._say("measuring printability ..." if label == "check" else "exporting ...")

    def wait_task(self, timeout: float = 120.0) -> bool:
        """Block until the running export / check finished and its result is shown (tests)."""
        import time
        end = time.monotonic() + timeout
        while self._task is not None and time.monotonic() < end:
            self._task._thread.join(0.01)
            self._poll_task()
        return self._task is None

    def _poll_task(self) -> None:
        task = self._task
        if task is None or not task.poll():
            return
        self._task = None
        label = self._task_label
        if task.error is not None:
            err = task.error
            result = getattr(err, "result", None)
            if result is not None:                    # the report blocks the export
                self._show_report(result)
                self.force_btn.configure(state="normal")
                self._say("blocked: " + str(err), error=True)
            else:
                self._say(str(err), error=True)
                self._write_report([(str(err), "block")])
                self.after_idle(lambda: self.scroll.reveal(self.report))
            return
        result = task.value
        self._show_report(result)
        self.force_btn.configure(state="disabled")
        if label == "export":
            self._say(f"wrote {Path(result.path).name}"
                      + (" (forced, see the report)" if result.blocked else ""),
                      error=result.blocked)
        else:
            self._say("printable" if not result.blocked else "blocked: see the report",
                      error=result.blocked)

    def _show_report(self, result) -> None:
        self._last_result = result
        lines = []
        for line in result.assembly_text.splitlines():
            lines.append((line, "dim"))
        for line in result.report_text.splitlines():
            tag = ("block" if line.startswith("BLOCK") else "warn" if line.startswith("WARN")
                   else "dim" if line.startswith(("note", "INFO")) else None)
            if line.startswith("printability"):
                tag = "ok" if " OK " in line else "block"
            lines.append((line, tag))
        if not result.report_text:
            lines.append(("no appendages: nothing to check beyond the body gate", "ok"))
        self._write_report(lines)
        self.after_idle(lambda: self.scroll.reveal(self.report))   # once the status line has its size

    def _write_report(self, lines) -> None:
        t = self.report
        t.configure(state="normal")
        t.delete("1.0", "end")
        for text, tag in lines:
            t.insert("end", text + "\n", (tag,) if tag else ())
        t.configure(state="disabled")

    # -- keeping the widgets in step with the document -----------------------------------------

    def _on_change(self, change) -> None:
        """The document changed (an edit anywhere, undo, load, a click in the picture)."""
        if not self._built:
            return
        scope, what = change.scope, change.what
        structural = what in ("add", "remove", "duplicate", "side", "preset")
        if scope == "selection":
            self._refresh_list()
            self._rebuild_props()
            self.view.refresh_overlay()
        elif scope == "all":
            self._refresh_list()
            self._rebuild_props()
            self._sync_body()
        elif structural or what in ("enabled", "rename"):
            self._refresh_list()
            if structural:
                self._rebuild_props()
        elif scope == "body" and not self._busy:
            self._sync_body()
        if self.lab.placing is not None and self.lab.placing != self.lab.doc.selected:
            self.cancel_placing()           # armed for something that is no longer selected
        self._sync_buttons()
        self._notes_dirty = True            # refreshed when the preview has settled (see _poll)
        if self.app is not None and self._active:
            self._push_status()

    def _refresh_list(self) -> None:
        doc = self.lab.doc
        self._syncing = True
        try:
            want = [s.id for s in doc.specs]
            have = list(self.tree.get_children())
            if want != have:
                self.tree.delete(*have)
                for s in doc.specs:
                    self.tree.insert("", "end", iid=s.id, text=s.label())
            for s in doc.specs:
                self.tree.item(s.id, text=s.label(), values=(s.kind, s.side,
                                                             "✓" if s.enabled else ""))
            if doc.selected and self.tree.exists(doc.selected):
                self.tree.selection_set(doc.selected)
            else:
                self.tree.selection_remove(*self.tree.selection())
        finally:
            self._syncing = False

    def _sync_body(self) -> None:
        doc = self.lab.doc
        for (kind, key), row in self.body_rows.items():
            if kind == "macro":
                row.set(doc.character.macro.get(key, 0.5))
            else:
                row.set(doc.character.sliders.get(key, 0.0))
        self.adapt_var.set(bool(doc.character.anatomy.get("enabled", True)))
        self.junction_var.set(doc.character.junction_enabled)
        self.cap_row.set(float(doc.character.anatomy.get("cap", 1.0)))

    def _sync_buttons(self) -> None:
        doc = self.lab.doc
        state = (doc.can_undo, doc.can_redo)
        if state != self._button_state:
            self._button_state = state
            self.undo_btn.configure(state="normal" if state[0] else "disabled")
            self.redo_btn.configure(state="normal" if state[1] else "disabled")

    def _push_status(self) -> None:
        text = self.status_text()
        if text != self._pushed_status:
            self._pushed_status = text
            self.app.set_status(text)

    def _update_notes(self) -> None:
        try:
            notes = self.lab.doc.anatomy_notes()
        except Exception:                                  # noqa: BLE001 - cosmetic
            notes = []
        text = "\n".join(notes) if notes else "(no adaptation)"
        frame = self.lab.frame
        if frame is not None and frame.notes:
            text += "\n" + "\n".join(frame.notes)
        if text != self._notes_text:
            self._notes_text = text
            self.notes_label.configure(text=text)

    # -- the poll loop (the only place the tab talks to the scheduler) -----------------------------

    def _poll(self) -> None:
        self._poll_job = None
        if not self._active or self.lab is None:
            return
        self.lab.tick()
        drain_deferred()
        if self._task is not None:
            self._poll_task()
        if self._notes_dirty and self.lab.scheduler.idle:     # not while a drag is in flight
            self._notes_dirty = False
            self._update_notes()
        self._poll_job = self.after(POLL_MS, self._poll)
