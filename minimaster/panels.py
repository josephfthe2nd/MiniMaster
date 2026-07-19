"""Side panels: Model (shapes), Rig (joints/bindings), Pose, Export.

Each panel talks to the app controller (select/mutate/refresh) and implements
``sync()`` to pull widget state from the scene. A ``_syncing`` guard prevents
widget callbacks from firing while sync writes to them.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import colorchooser, messagebox, ttk

import numpy as np

from . import parts as partlib
from .core import math3d as m3
from .core import primitives
from .export import SIZE_PRESETS

_INT_PARAMS = {"segments", "minor_segments", "subdivisions"}
NO_BONE = "(none)"


def _grid_entries(parent, rows, width=7):
    """rows: list of (label, [StringVar, ...]). Returns list of Entry widgets."""
    entries = []
    for r, (label, cells) in enumerate(rows):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=(0, 4))
        for c, var in enumerate(cells):
            e = ttk.Entry(parent, textvariable=var, width=width)
            e.grid(row=r, column=c + 1, padx=1, pady=1)
            entries.append(e)
    return entries


class ModelPanel(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=6)
        self.app = app
        self._syncing = False

        add = ttk.LabelFrame(self, text="Add shape", padding=4)
        add.pack(fill="x")
        for i, kind in enumerate(sorted(primitives.PRIMITIVES)):
            ttk.Button(
                add, text=kind, width=9,
                command=lambda k=kind: self.app.add_shape(k),
            ).grid(row=i // 3, column=i % 3, padx=2, pady=2)

        lf = ttk.LabelFrame(self, text="Shapes", padding=4)
        lf.pack(fill="both", expand=True, pady=(6, 0))
        self.listbox = tk.Listbox(lf, height=8, exportselection=False)
        self.listbox.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(lf, command=self.listbox.yview)
        sb.pack(side="right", fill="y")
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)

        btns = ttk.Frame(self)
        btns.pack(fill="x", pady=4)
        ttk.Button(btns, text="Duplicate", command=self.app.duplicate_selected).pack(
            side="left", padx=1)
        ttk.Button(btns, text="Mirror X", command=self.app.mirror_selected).pack(
            side="left", padx=1)
        ttk.Button(btns, text="Delete", command=self.app.delete_selected).pack(
            side="left", padx=1)

        tf = ttk.LabelFrame(self, text="Transform (mm / deg)", padding=4)
        tf.pack(fill="x")
        self.tvars = {
            key: [tk.StringVar() for _ in range(3)]
            for key in ("position", "rotation", "scale")
        }
        entries = _grid_entries(
            tf,
            [("pos", self.tvars["position"]), ("rot", self.tvars["rotation"]),
             ("scale", self.tvars["scale"])],
        )
        for e in entries:
            e.bind("<Return>", self._apply_transform)
            e.bind("<FocusOut>", self._apply_transform)

        self.params_frame = ttk.LabelFrame(self, text="Parameters", padding=4)
        self.params_frame.pack(fill="x", pady=(6, 0))
        self.param_vars: dict[str, tk.StringVar] = {}

        bind = ttk.LabelFrame(self, text="Binding", padding=4)
        bind.pack(fill="x", pady=(6, 0))
        ttk.Label(bind, text="bone").grid(row=0, column=0, sticky="w")
        self.bone_var = tk.StringVar()
        self.bone_combo = ttk.Combobox(
            bind, textvariable=self.bone_var, state="readonly", width=14)
        self.bone_combo.grid(row=0, column=1, padx=4)
        self.bone_combo.bind("<<ComboboxSelected>>", self._apply_bone)
        ttk.Button(bind, text="Auto-bind all", command=self.app.auto_bind_all).grid(
            row=0, column=2, padx=2)

        cf = ttk.Frame(self)
        cf.pack(fill="x", pady=(6, 0))
        ttk.Label(cf, text="color").pack(side="left")
        self.color_btn = tk.Button(cf, text="  ", width=3, command=self._pick_color)
        self.color_btn.pack(side="left", padx=4)

    # -- callbacks --------------------------------------------------------

    def _on_list_select(self, _e):
        if self._syncing:
            return
        sel = self.listbox.curselection()
        self.app.select_shape(self.listbox.get(sel[0]) if sel else None)

    def _apply_transform(self, _e=None):
        if self._syncing:
            return
        shape = self.app.selected_shape_obj()
        if shape is None:
            return
        try:
            new = {
                key: [float(v.get()) for v in cells]
                for key, cells in self.tvars.items()
            }
        except ValueError:
            return
        if (
            np.allclose(new["position"], shape.position)
            and np.allclose(new["rotation"], shape.rotation)
            and np.allclose(new["scale"], shape.scale)
        ):
            return

        def apply(scene):
            s = scene.get_shape(shape.name)
            s.position = np.array(new["position"])
            s.rotation = np.array(new["rotation"])
            s.scale = np.array(new["scale"])

        self.app.mutate(apply)

    def _apply_params(self, _e=None):
        if self._syncing:
            return
        shape = self.app.selected_shape_obj()
        if shape is None:
            return
        try:
            new_params = {}
            for key, var in self.param_vars.items():
                val = float(var.get())
                new_params[key] = int(val) if key in _INT_PARAMS else val
        except ValueError:
            return
        # The entries show defaults merged in; only mutate on a real change.
        shown = {**primitives.default_params(shape.kind), **shape.params}
        if new_params == shown:
            return

        def apply(scene):
            s = scene.get_shape(shape.name)
            try:
                primitives.build(s.kind, new_params)  # validate before applying
            except (ValueError, KeyError) as exc:
                raise ValueError(str(exc)) from exc
            s.params = new_params

        self.app.mutate(apply)

    def _apply_bone(self, _e=None):
        if self._syncing:
            return
        shape = self.app.selected_shape_obj()
        if shape is None:
            return
        choice = self.bone_var.get()
        bone = None if choice == NO_BONE else choice
        if bone == shape.bone:
            return
        self.app.mutate(lambda scene: setattr(scene.get_shape(shape.name), "bone", bone))

    def _pick_color(self):
        shape = self.app.selected_shape_obj()
        if shape is None:
            return
        rgb, hexcolor = colorchooser.askcolor(color=shape.color, parent=self)
        if hexcolor:
            self.app.mutate(
                lambda scene: setattr(scene.get_shape(shape.name), "color", hexcolor))

    # -- sync -------------------------------------------------------------

    def sync(self):
        self._syncing = True
        try:
            scene = self.app.scene
            names = [s.name for s in scene.shapes]
            current = list(self.listbox.get(0, "end"))
            if current != names:
                self.listbox.delete(0, "end")
                for n in names:
                    self.listbox.insert("end", n)
            self.listbox.selection_clear(0, "end")
            shape = self.app.selected_shape_obj()
            if shape is not None and shape.name in names:
                idx = names.index(shape.name)
                self.listbox.selection_set(idx)
                self.listbox.see(idx)

            bones = [NO_BONE] + scene.armature.bone_names()
            self.bone_combo.configure(values=bones)

            for w in self.params_frame.winfo_children():
                w.destroy()
            self.param_vars = {}

            if shape is None:
                for cells in self.tvars.values():
                    for v in cells:
                        v.set("")
                self.bone_var.set("")
                self.color_btn.configure(bg="#d9d9d9")
                return

            for key, cells in self.tvars.items():
                vals = getattr(shape, key)
                for v, val in zip(cells, vals):
                    v.set(f"{val:.2f}".rstrip("0").rstrip("."))
            self.bone_var.set(shape.bone if shape.bone else NO_BONE)
            self.color_btn.configure(bg=shape.color)

            defaults = primitives.default_params(shape.kind)
            merged = {**defaults, **shape.params}
            for r, (key, val) in enumerate(merged.items()):
                ttk.Label(self.params_frame, text=key).grid(row=r, column=0, sticky="w")
                var = tk.StringVar(value=str(val))
                self.param_vars[key] = var
                e = ttk.Entry(self.params_frame, textvariable=var, width=8)
                e.grid(row=r, column=1, padx=4, pady=1)
                e.bind("<Return>", self._apply_params)
                e.bind("<FocusOut>", self._apply_params)
        finally:
            self._syncing = False


class RigPanel(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=6)
        self.app = app
        self._syncing = False

        tf = ttk.LabelFrame(self, text="Joints", padding=4)
        tf.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(tf, show="tree", height=12, selectmode="browse")
        self.tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(tf, command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        pf = ttk.LabelFrame(self, text="Joint position", padding=4)
        pf.pack(fill="x", pady=(6, 0))
        self.pos_vars = [tk.StringVar() for _ in range(3)]
        for e in _grid_entries(pf, [("pos", self.pos_vars)]):
            e.bind("<Return>", self._apply_position)
            e.bind("<FocusOut>", self._apply_position)

        af = ttk.LabelFrame(self, text="Add / edit", padding=4)
        af.pack(fill="x", pady=(6, 0))
        self.name_var = tk.StringVar()
        ttk.Entry(af, textvariable=self.name_var, width=14).grid(row=0, column=0, padx=2)
        ttk.Button(af, text="Add child", command=self._add_child).grid(row=0, column=1)
        ttk.Button(af, text="Rename", command=self._rename).grid(row=0, column=2)
        ttk.Button(af, text="Delete joint", command=self._delete).grid(
            row=1, column=1, pady=2)
        ttk.Button(af, text="Auto-bind shapes", command=self.app.auto_bind_all).grid(
            row=1, column=2, pady=2)

        hint = ("A joint's rotation moves everything below it.\n"
                "Shapes bind to the bone above each child joint.")
        ttk.Label(self, text=hint, foreground="#777777", justify="left").pack(
            fill="x", pady=(6, 0))

    def _on_tree_select(self, _e):
        if self._syncing:
            return
        sel = self.tree.selection()
        name = sel[0] if sel else None
        if name == self.app.selected_joint:
            return  # programmatic selection_set echoes back asynchronously
        self.app.select_joint(name)

    def _apply_position(self, _e=None):
        if self._syncing:
            return
        name = self.app.selected_joint
        if name is None or name not in self.app.scene.armature.joints:
            return
        try:
            pos = [float(v.get()) for v in self.pos_vars]
        except ValueError:
            return
        if np.allclose(pos, self.app.scene.armature.joints[name].position):
            return
        self.app.mutate(lambda scene: scene.armature.move_joint(name, pos))

    def _add_child(self):
        parent = self.app.selected_joint
        name = self.name_var.get().strip()
        if not name:
            messagebox.showinfo("Add joint", "Type a name for the new joint first.",
                                parent=self)
            return
        if name in self.app.scene.armature.joints:
            messagebox.showerror("Add joint", f"Joint {name!r} already exists.",
                                 parent=self)
            return

        def apply(scene):
            if parent and parent in scene.armature.joints:
                pos = scene.armature.joints[parent].position + [0, 0, -4.0]
                scene.armature.add_joint(name, pos, parent)
            else:
                scene.armature.add_joint(name, [0, 0, 0], None)

        self.app.mutate(apply)
        self.app.select_joint(name)

    def _rename(self):
        old = self.app.selected_joint
        new = self.name_var.get().strip()
        if not old or not new or old == new:
            return
        if new in self.app.scene.armature.joints:
            messagebox.showerror("Rename joint", f"Joint {new!r} already exists.",
                                 parent=self)
            return
        self.app.mutate(lambda scene: scene.rename_joint(old, new))
        self.app.select_joint(new)

    def _delete(self):
        name = self.app.selected_joint
        if not name:
            return
        self.app.select_joint(None)
        self.app.mutate(lambda scene: scene.remove_joint(name))

    def sync(self):
        self._syncing = True
        try:
            arm = self.app.scene.armature

            have = []

            def collect(node=""):
                for child in self.tree.get_children(node):
                    have.append(child)
                    collect(child)

            collect()
            if [j.name for j in arm._ordered()] != have:
                self.tree.delete(*self.tree.get_children(""))
                for j in arm._ordered():
                    parent = j.parent if j.parent is not None else ""
                    self.tree.insert(parent, "end", iid=j.name, text=j.name, open=True)

            sel = self.app.selected_joint
            current = self.tree.selection()
            if sel and sel in arm.joints:
                if current != (sel,):
                    self.tree.selection_set(sel)
                self.tree.see(sel)
                for v, val in zip(self.pos_vars, arm.joints[sel].position):
                    v.set(f"{val:.2f}".rstrip("0").rstrip("."))
                self.name_var.set(sel)
            else:
                if current:
                    self.tree.selection_remove(*current)
                for v in self.pos_vars:
                    v.set("")
        finally:
            self._syncing = False


class PosePanel(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=6)
        self.app = app
        self._syncing = False
        self._stroke_open = False

        pf = ttk.LabelFrame(self, text="Poses", padding=4)
        pf.pack(fill="x")
        self.pose_var = tk.StringVar()
        self.pose_combo = ttk.Combobox(
            pf, textvariable=self.pose_var, state="readonly", width=16)
        self.pose_combo.grid(row=0, column=0, padx=2, columnspan=2)
        self.pose_combo.bind("<<ComboboxSelected>>", self._activate_pose)
        ttk.Button(pf, text="Rest", command=lambda: self._set_active(None)).grid(
            row=0, column=2, padx=2)
        self.new_var = tk.StringVar()
        ttk.Entry(pf, textvariable=self.new_var, width=12).grid(row=1, column=0, pady=2)
        ttk.Button(pf, text="New pose", command=self._new_pose).grid(row=1, column=1)
        ttk.Button(pf, text="Delete", command=self._delete_pose).grid(row=1, column=2)

        jf = ttk.LabelFrame(self, text="Joint rotation (deg)", padding=4)
        jf.pack(fill="x", pady=(6, 0))
        self.joint_label = ttk.Label(jf, text="click a joint in the viewport")
        self.joint_label.pack(anchor="w")
        self.scales = []
        self.scale_vars = []
        for axis in ("X", "Y", "Z"):
            row = ttk.Frame(jf)
            row.pack(fill="x")
            ttk.Label(row, text=axis, width=2).pack(side="left")
            var = tk.DoubleVar()
            scale = ttk.Scale(
                row, from_=-180, to=180, variable=var,
                command=lambda _v, a=len(self.scales): self._slider_moved(a))
            scale.pack(side="left", fill="x", expand=True, padx=4)
            scale.bind("<ButtonRelease-1>", lambda e: self._end_slider_stroke())
            val = ttk.Label(row, width=5)
            val.pack(side="left")
            self.scales.append((scale, val))
            self.scale_vars.append(var)
        ttk.Button(jf, text="Zero joint", command=self._zero_joint).pack(
            anchor="e", pady=(4, 0))

        hint = ("Pick a pose (or create one), click a joint,\n"
                "drag the sliders. Rotations move the subtree.")
        ttk.Label(self, text=hint, foreground="#777777", justify="left").pack(
            fill="x", pady=(6, 0))

    def _set_active(self, name):
        if name == self.app.scene.active_pose:
            return
        self.app.mutate(lambda scene: setattr(scene, "active_pose", name))

    def _activate_pose(self, _e=None):
        if not self._syncing:
            self._set_active(self.pose_var.get())

    RESERVED_POSE_NAMES = {"(active)", "(rest)", "rest", "__active__"}

    def _new_pose(self):
        name = self.new_var.get().strip() or "pose"
        if name in self.RESERVED_POSE_NAMES:
            messagebox.showerror(
                "New pose", f"{name!r} is reserved; pick another name.", parent=self)
            return
        scene = self.app.scene
        base = name
        i = 1
        while name in scene.poses:
            name = f"{base}.{i}"
            i += 1
        start = dict(scene.resolve_pose()) if scene.active_pose else {}

        def apply(s):
            s.poses[name] = dict(start)
            s.active_pose = name

        self.app.mutate(apply)

    def _delete_pose(self):
        name = self.app.scene.active_pose
        if not name:
            return

        def apply(scene):
            scene.poses.pop(name, None)
            scene.active_pose = None

        self.app.mutate(apply)

    def _end_slider_stroke(self):
        self._stroke_open = False

    def _slider_moved(self, _axis_idx):
        if self._syncing:
            return
        scene = self.app.scene
        joint = self.app.selected_joint
        if not scene.active_pose or not joint:
            return
        angles = tuple(round(v.get(), 1) for v in self.scale_vars)
        pose = scene.poses[scene.active_pose]
        if angles == tuple(pose.get(joint, (0.0, 0.0, 0.0))):
            return
        # One undo snapshot per stroke (mouse drag or run of key presses),
        # pushed before the first actual change — this also catches keyboard
        # edits, which never see a ButtonPress.
        if not getattr(self, "_stroke_open", False):
            self.app.push_undo_snapshot()
            self._stroke_open = True
        if not any(angles):
            pose.pop(joint, None)
        else:
            pose[joint] = angles
        for (scale, label), val in zip(self.scales, angles):
            label.configure(text=f"{val:.0f}")
        self.app.light_refresh()

    def _zero_joint(self):
        scene = self.app.scene
        joint = self.app.selected_joint
        if not scene.active_pose or not joint:
            return
        self.app.mutate(lambda s: s.poses[s.active_pose].pop(joint, None))

    def sync(self):
        self._syncing = True
        self._stroke_open = False
        try:
            scene = self.app.scene
            self.pose_combo.configure(values=list(scene.poses))
            self.pose_var.set(scene.active_pose or "")
            joint = self.app.selected_joint
            if joint and joint in scene.armature.joints:
                self.joint_label.configure(text=f"joint: {joint}")
                pose = scene.poses.get(scene.active_pose, {}) if scene.active_pose else {}
                angles = pose.get(joint, (0.0, 0.0, 0.0))
                for var, (pair, val), a in zip(self.scale_vars, self.scales, angles):
                    var.set(float(a))
                for (scale, label), a in zip(self.scales, angles):
                    label.configure(text=f"{a:.0f}")
                state = "normal" if scene.active_pose else "disabled"
                for scale, _ in self.scales:
                    scale.configure(state=state)
            else:
                self.joint_label.configure(
                    text="click a joint in the viewport")
                for var in self.scale_vars:
                    var.set(0.0)
                for scale, label in self.scales:
                    scale.configure(state="disabled")
                    label.configure(text="")
        finally:
            self._syncing = False


class PartsPanel(ttk.Frame):
    """Spore-style part palette: pick a part, click the model to place it,
    then adjust/mirror/delete the placed instance as a unit."""

    def __init__(self, master, app):
        super().__init__(master, padding=6)
        self.app = app
        self._syncing = False
        self._infos: dict[str, partlib.PartInfo] = {}
        self._thumbs: dict[str, tk.PhotoImage] = {}
        self._listing_key = None

        lf = ttk.LabelFrame(self, text="Part library", padding=4)
        lf.pack(fill="both", expand=True)
        top = ttk.Frame(lf)
        top.pack(fill="x")
        ttk.Label(top, text="category").pack(side="left")
        self.cat_var = tk.StringVar(value="(all)")
        self.cat_combo = ttk.Combobox(top, textvariable=self.cat_var,
                                      state="readonly", width=10)
        self.cat_combo.pack(side="left", padx=4)
        self.cat_combo.bind("<<ComboboxSelected>>", lambda e: self.sync())

        style = ttk.Style()
        style.configure("Parts.Treeview", rowheight=38)
        self.tree = ttk.Treeview(lf, show="tree", height=8, selectmode="browse",
                                 style="Parts.Treeview")
        self.tree.pack(side="left", fill="both", expand=True, pady=(4, 0))
        sb = ttk.Scrollbar(lf, command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=sb.set)

        ttk.Button(self, text="Place on model", command=self._arm).pack(
            fill="x", pady=(6, 2))
        ttk.Label(self, text="then click the mini in the viewport",
                  foreground="#777777").pack(fill="x")

        inst = ttk.LabelFrame(self, text="Selected part instance", padding=4)
        inst.pack(fill="x", pady=(8, 0))
        self.inst_label = ttk.Label(inst, text="(click a placed part)")
        self.inst_label.grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(inst, text="spin °").grid(row=1, column=0, sticky="w")
        self.spin_var = tk.StringVar(value="15")
        ttk.Entry(inst, textvariable=self.spin_var, width=6).grid(row=1, column=1)
        ttk.Button(inst, text="Spin", command=self._spin).grid(row=1, column=2,
                                                               padx=2)
        ttk.Label(inst, text="size ×").grid(row=2, column=0, sticky="w")
        self.scale_var = tk.StringVar(value="1.25")
        ttk.Entry(inst, textvariable=self.scale_var, width=6).grid(row=2, column=1)
        ttk.Button(inst, text="Scale", command=self._scale).grid(row=2, column=2,
                                                                 padx=2)
        btns = ttk.Frame(inst)
        btns.grid(row=3, column=0, columnspan=3, pady=(4, 0), sticky="w")
        ttk.Button(btns, text="Mirror", command=self.app.mirror_selected).pack(
            side="left", padx=1)
        ttk.Button(btns, text="Ungroup", command=self._ungroup).pack(
            side="left", padx=1)
        ttk.Button(btns, text="Delete", command=self.app.delete_selected).pack(
            side="left", padx=1)

        ttk.Button(self, text="Save selection as part…",
                   command=self._save_as_part).pack(fill="x", pady=(8, 0))

    # -- library ----------------------------------------------------------

    def _selected_info(self) -> partlib.PartInfo | None:
        sel = self.tree.selection()
        return self._infos.get(sel[0]) if sel else None

    def _arm(self):
        info = self._selected_info()
        if info is None:
            messagebox.showinfo("Place part", "Pick a part from the library first.",
                                parent=self)
            return
        self.app.arm_placement(info)

    # -- instance ops -----------------------------------------------------

    def _selected_group(self) -> str | None:
        shape = self.app.selected_shape_obj()
        return shape.group if shape is not None else None

    def _spin(self):
        group = self._selected_group()
        if not group:
            return
        try:
            degrees = float(self.spin_var.get())
        except ValueError:
            return
        axis = self.app.scene.group_outward_axis(group)
        self.app.mutate(
            lambda s: s.transform_group(group, rotation=m3.axis_angle(axis, degrees))
        )

    def _scale(self):
        group = self._selected_group()
        if not group:
            return
        try:
            factor = float(self.scale_var.get())
        except ValueError:
            return
        if factor <= 0:
            messagebox.showerror("Scale part", "Scale factor must be positive.",
                                 parent=self)
            return
        self.app.mutate(lambda s: s.transform_group(group, scale=factor))

    def _ungroup(self):
        group = self._selected_group()
        if not group:
            return
        self.app.mutate(lambda s: s.ungroup(group))

    def _save_as_part(self):
        SavePartDialog(self.app)

    # -- sync -------------------------------------------------------------

    def sync(self):
        self._syncing = True
        try:
            infos = partlib.list_parts()
            categories = ["(all)"] + sorted({i.category for i in infos})
            self.cat_combo.configure(values=categories)
            if self.cat_var.get() not in categories:
                self.cat_var.set("(all)")
            wanted = [
                i for i in infos
                if self.cat_var.get() in ("(all)", i.category)
            ]
            key = tuple((str(i.path), i.name, i.category) for i in wanted)
            if key != self._listing_key:
                self._listing_key = key
                self.tree.delete(*self.tree.get_children(""))
                self._infos = {}
                self._thumbs = {}
                for info in wanted:
                    iid = str(info.path)
                    self._infos[iid] = info
                    label = f"{info.name}  ({info.category})"
                    if info.user:
                        label += "  [user]"
                    kwargs = {}
                    if info.thumbnail is not None:
                        try:
                            img = tk.PhotoImage(file=str(info.thumbnail))
                            factor = max(1, img.width() // 34)
                            img = img.subsample(factor, factor)
                            self._thumbs[iid] = img
                            kwargs["image"] = img
                        except tk.TclError:
                            pass
                    self.tree.insert("", "end", iid=iid, text=label, **kwargs)

            shape = self.app.selected_shape_obj()
            group = shape.group if shape is not None else None
            if group and group in self.app.scene.groups:
                meta = self.app.scene.groups[group]
                joints = [j for j in meta.get("joints", [])
                          if j in self.app.scene.armature.joints]
                self.inst_label.configure(
                    text=(f"{group}  —  {meta.get('part', '?')}: "
                          f"{len(self.app.scene.group_members(group))} shapes, "
                          f"{len(joints)} joints")
                )
            else:
                self.inst_label.configure(text="(click a placed part)")
        finally:
            self._syncing = False


class SavePartDialog(tk.Toplevel):
    """Capture selected shapes (and optionally a joint subtree) as a reusable
    user-library part."""

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title("Save as part")
        self.transient(app)
        self.resizable(False, False)

        frame = ttk.Frame(self, padding=8)
        frame.pack(fill="both", expand=True)

        scene = app.scene
        selected = app.selected_shape_obj()
        default_name = "part"
        preselect: set[str] = set()
        default_root = "(none)"
        if selected is not None:
            if selected.group and selected.group in scene.groups:
                meta = scene.groups[selected.group]
                default_name = meta.get("part", "part")
                preselect = {s.name for s in scene.group_members(selected.group)}
                group_joints = set(meta.get("joints", []))
                roots = [
                    j for j in meta.get("joints", [])
                    if j in scene.armature.joints
                    and scene.armature.joints[j].parent not in group_joints
                ]
                if roots:
                    default_root = roots[0]
            else:
                default_name = selected.name
                preselect = {selected.name}

        ttk.Label(frame, text="name").grid(row=0, column=0, sticky="w")
        self.name_var = tk.StringVar(value=default_name)
        ttk.Entry(frame, textvariable=self.name_var, width=18).grid(
            row=0, column=1, sticky="w", pady=2)
        ttk.Label(frame, text="category").grid(row=1, column=0, sticky="w")
        self.cat_var = tk.StringVar(value="custom")
        ttk.Entry(frame, textvariable=self.cat_var, width=18).grid(
            row=1, column=1, sticky="w", pady=2)

        ttk.Label(frame, text="shapes to include").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.listbox = tk.Listbox(frame, selectmode="extended", height=8,
                                  exportselection=False, width=28)
        self.listbox.grid(row=3, column=0, columnspan=2, sticky="we")
        names = [s.name for s in scene.shapes]
        for i, n in enumerate(names):
            self.listbox.insert("end", n)
            if n in preselect:
                self.listbox.selection_set(i)

        ttk.Label(frame, text="posable root joint").grid(
            row=4, column=0, sticky="w", pady=(6, 0))
        self.root_var = tk.StringVar(value=default_root)
        joints = ["(none)"] + list(scene.armature.joints)
        ttk.Combobox(frame, textvariable=self.root_var, values=joints,
                     state="readonly", width=16).grid(row=4, column=1, sticky="w")

        row = ttk.Frame(frame)
        row.grid(row=5, column=0, columnspan=2, pady=(8, 0), sticky="e")
        ttk.Button(row, text="Cancel", command=self.destroy).pack(side="right",
                                                                  padx=2)
        ttk.Button(row, text="Save", command=self._save).pack(side="right")

    def _save(self):
        names = [self.listbox.get(i) for i in self.listbox.curselection()]
        root = self.root_var.get()
        try:
            part = partlib.part_from_selection(
                self.app.scene,
                names,
                name=self.name_var.get().strip() or "part",
                category=self.cat_var.get().strip() or "custom",
                joint_root=None if root == "(none)" else root,
            )
            path = partlib.save_user_part(part)
        except ValueError as exc:
            messagebox.showerror("Save as part", str(exc), parent=self)
            return
        self.destroy()
        self.app.set_status(f"saved part to {path}")
        self.app.refresh()  # the palette picks up the new part


class ExportPanel(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=6)
        self.app = app
        self._syncing = False

        sf = ttk.LabelFrame(self, text="Size", padding=4)
        sf.pack(fill="x")
        self.size_var = tk.StringVar(value="medium")
        sizes = sorted(SIZE_PRESETS, key=SIZE_PRESETS.get) + ["custom", "scene units"]
        for i, s in enumerate(sizes):
            label = s if s in ("custom", "scene units") else f"{s} ({SIZE_PRESETS[s]:g}mm)"
            ttk.Radiobutton(sf, text=label, value=s, variable=self.size_var).grid(
                row=i // 2, column=i % 2, sticky="w")
        hf = ttk.Frame(sf)
        hf.grid(row=(len(sizes) + 1) // 2, column=0, columnspan=2, sticky="w")
        ttk.Label(hf, text="custom height (mm)").pack(side="left")
        self.height_var = tk.StringVar(value="32")
        ttk.Entry(hf, textvariable=self.height_var, width=6).pack(side="left", padx=4)

        bf = ttk.LabelFrame(self, text="Base", padding=4)
        bf.pack(fill="x", pady=(6, 0))
        self.base_style = tk.StringVar()
        ttk.Label(bf, text="style").grid(row=0, column=0, sticky="w")
        combo = ttk.Combobox(bf, textvariable=self.base_style, state="readonly",
                             values=["round", "cobble", "square", "none"], width=8)
        combo.grid(row=0, column=1, padx=4)
        combo.bind("<<ComboboxSelected>>", self._apply_base)
        ttk.Label(bf, text="diameter").grid(row=1, column=0, sticky="w")
        self.base_diam = tk.StringVar()
        e = ttk.Entry(bf, textvariable=self.base_diam, width=8)
        e.grid(row=1, column=1, padx=4, pady=2)
        e.bind("<Return>", self._apply_base)
        e.bind("<FocusOut>", self._apply_base)

        pf = ttk.LabelFrame(self, text="Pose", padding=4)
        pf.pack(fill="x", pady=(6, 0))
        self.pose_var = tk.StringVar(value="(active)")
        self.pose_combo = ttk.Combobox(pf, textvariable=self.pose_var,
                                       state="readonly", width=16)
        self.pose_combo.pack(anchor="w")

        ttk.Button(self, text="Export STL…", command=self._export).pack(
            fill="x", pady=(10, 2))
        ttk.Button(self, text="Save PNG preview…", command=self._save_png).pack(fill="x")
        ttk.Button(self, text="Check printability", command=self._check).pack(
            fill="x", pady=(2, 0))
        self.stats = ttk.Label(self, text="", justify="left", foreground="#555555")
        self.stats.pack(fill="x", pady=(8, 0))

    def export_options(self) -> dict:
        """Collect export options; raises ValueError for bad numeric input."""
        size = self.size_var.get()
        opts: dict = {"pose_name": None, "with_base": True}
        pose = self.pose_var.get()
        if pose == "(active)":
            opts["pose_name"] = "__active__"
        elif pose == "(rest)":
            opts["pose_name"] = None
        else:
            opts["pose_name"] = pose
        if size == "custom":
            text = self.height_var.get().strip() or "32"
            try:
                opts["height"] = float(text)
            except ValueError:
                raise ValueError(f"custom height must be a number, got {text!r}")
        elif size != "scene units":
            opts["size"] = size
        if self.base_style.get() == "none":
            opts["with_base"] = False
        return opts

    def _options_or_error(self) -> dict | None:
        try:
            return self.export_options()
        except ValueError as exc:
            messagebox.showerror("Export", str(exc), parent=self)
            return None

    def _apply_base(self, _e=None):
        if self._syncing:
            return
        scene = self.app.scene
        style = self.base_style.get() or scene.base.get("style", "round")
        try:
            diameter = float(self.base_diam.get() or scene.base.get("diameter", 25.0))
        except ValueError:
            return
        if style != "none" and diameter <= 0:
            messagebox.showerror(
                "Base", "Base diameter must be positive.", parent=self)
            self.sync()  # restore the shown value
            return
        new = {**scene.base, "style": style, "diameter": diameter}
        if new == scene.base:
            return
        self.app.mutate(lambda s: setattr(s, "base", new))

    def _export(self):
        opts = self._options_or_error()
        if opts is not None:
            self.app.export_stl_dialog(opts)

    def _save_png(self):
        opts = self._options_or_error()
        if opts is not None:
            self.app.save_png_dialog(opts)

    def _check(self):
        from .export import ExportError, assemble

        try:
            mesh = assemble(self.app.scene, **self.export_options())
        except (ExportError, ValueError, KeyError) as exc:
            self.stats.configure(text=f"error: {exc}", foreground="#aa2222")
            return
        rep = mesh.integrity_report()
        lo, hi = mesh.bounds
        d = hi - lo
        ok = "watertight" if rep["watertight"] else "NOT WATERTIGHT"
        self.stats.configure(
            foreground="#227722" if rep["watertight"] else "#aa2222",
            text=(f"{len(mesh.faces)} triangles — {ok}\n"
                  f"{d[0]:.1f} × {d[1]:.1f} × {d[2]:.1f} mm\n"
                  f"{mesh.volume() / 1000:.1f} cm³ of resin"),
        )

    def sync(self):
        self._syncing = True
        try:
            scene = self.app.scene
            self.base_style.set(scene.base.get("style", "round"))
            self.base_diam.set(f"{scene.base.get('diameter', 25.0):g}")
            poses = ["(active)", "(rest)"] + list(scene.poses)
            self.pose_combo.configure(values=poses)
            if self.pose_var.get() not in poses:
                self.pose_var.set("(active)")
        finally:
            self._syncing = False
