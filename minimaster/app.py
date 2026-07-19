"""MiniMaster studio: tkinter main window wiring viewport, panels, and scene.

Tabs are modes: Model and Rig edit the rest scene, Pose and Export show the
posed figure. All mutations flow through ``mutate`` (undo snapshot + refresh).
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import __version__
from .core.raycast import raycast_meshes
from .export import ExportError, export_stl
from .panels import ExportPanel, ModelPanel, PartsPanel, PosePanel, RigPanel
from .parts import load_part, place_part
from .render import render_scene
from .scene import Scene, UndoStack
from .templates import list_templates, load_template
from .viewport import RenderItem, Viewport

NEW_SHAPE_SCALE = 6.0  # mm; unit primitives arrive at a useful mini-part size


class MiniMasterApp(tk.Tk):
    def __init__(self, scene: Scene | None = None):
        super().__init__()
        self.title("MiniMaster")
        self.geometry("1200x780")
        self.minsize(900, 560)

        self.scene = scene or Scene(name="untitled")
        self.path: Path | None = None
        self.undo_stack = UndoStack()
        self.selected_shape: str | None = None
        self.selected_joint: str | None = None
        self._grab_snapshot: str | None = None
        self._placement = None  # Part armed for snap-to-surface placement

        self.status = ttk.Label(self, text="", anchor="w", padding=(6, 2))
        self.status.pack(side="bottom", fill="x")

        self.viewport = Viewport(
            self,
            on_select_shape=self._viewport_select_shape,
            on_select_joint=self._viewport_select_joint,
            on_grab_move=self._grab_move,
            on_grab_end=self._grab_end,
        )
        self.viewport.pack(side="left", fill="both", expand=True)

        right = ttk.Frame(self, width=360)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True)
        self.model_panel = ModelPanel(self.notebook, self)
        self.parts_panel = PartsPanel(self.notebook, self)
        self.rig_panel = RigPanel(self.notebook, self)
        self.pose_panel = PosePanel(self.notebook, self)
        self.export_panel = ExportPanel(self.notebook, self)
        self._mode_by_tab: dict[str, str] = {}
        for panel, mode_name, label in (
            (self.model_panel, "model", "Model"),
            (self.parts_panel, "parts", "Parts"),
            (self.rig_panel, "rig", "Rig"),
            (self.pose_panel, "pose", "Pose"),
            (self.export_panel, "export", "Export"),
        ):
            self.notebook.add(panel, text=label)
            self._mode_by_tab[str(panel)] = mode_name
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self.viewport.pick_override = self._placement_click

        self._build_menu()
        self._bind_keys()
        self.refresh()
        self.after(50, self.viewport.frame_content)

    # -- mode -------------------------------------------------------------

    def mode(self) -> str:
        try:
            return self._mode_by_tab.get(self.notebook.select(), "model")
        except tk.TclError:
            return "model"

    # -- selection --------------------------------------------------------

    def selected_shape_obj(self):
        if self.selected_shape is None:
            return None
        try:
            return self.scene.get_shape(self.selected_shape)
        except KeyError:
            self.selected_shape = None
            return None

    def select_shape(self, name: str | None):
        self.selected_shape = name
        self.refresh()

    def select_joint(self, name: str | None):
        self.selected_joint = name
        self.refresh()

    def _on_tab_changed(self, _e=None):
        if self.mode() != "parts":
            self.cancel_placement()
        self.refresh()

    def _viewport_select_shape(self, name):
        if self.mode() in ("model", "parts"):
            self.select_shape(name)

    def _viewport_select_joint(self, name):
        if self.mode() in ("rig", "pose"):
            self.select_joint(name)

    # -- mutation / undo --------------------------------------------------

    def push_undo_snapshot(self):
        self.undo_stack.push(self.scene.to_json())

    def _abort_grab(self):
        """Cancel any in-progress grab (restoring the pre-grab scene) before
        another mutation or history jump — otherwise _grab_end would later
        restore/push a snapshot from a different timeline."""
        if self.viewport.grabbing:
            self.viewport.cancel_grab()

    def mutate(self, fn):
        self._abort_grab()
        snapshot = self.scene.to_json()
        try:
            fn(self.scene)
        except Exception as exc:
            self.scene = Scene.from_json(snapshot)
            messagebox.showerror("MiniMaster", str(exc), parent=self)
            self.refresh()
            return
        self.undo_stack.push(snapshot)
        self.refresh()

    def undo(self, _e=None):
        self._abort_grab()
        prev = self.undo_stack.undo(self.scene.to_json())
        if prev is not None:
            self.scene = Scene.from_json(prev)
            self.refresh()

    def redo(self, _e=None):
        self._abort_grab()
        nxt = self.undo_stack.redo(self.scene.to_json())
        if nxt is not None:
            self.scene = Scene.from_json(nxt)
            self.refresh()

    # -- shape ops --------------------------------------------------------

    def add_shape(self, kind: str):
        scale = [NEW_SHAPE_SCALE] * 3
        pos = [0.0, 0.0, NEW_SHAPE_SCALE / 2.0]

        def apply(scene):
            shape = scene.add_shape(kind, position=pos, scale=scale)
            self.selected_shape = shape.name

        self.mutate(apply)

    def duplicate_selected(self, _e=None):
        if self.selected_shape is None:
            return

        def apply(scene):
            dup = scene.duplicate_shape(self.selected_shape)
            self.selected_shape = dup.name

        self.mutate(apply)

    def mirror_selected(self, _e=None):
        if self.selected_shape is None:
            return
        shape = self.selected_shape_obj()
        if shape is None:
            return
        if self.mode() == "parts" and shape.group:
            group = shape.group

            def apply(scene):
                new = scene.mirror_group(group)
                members = scene.group_members(new)
                if members:
                    self.selected_shape = members[0].name

        else:

            def apply(scene):
                dup = scene.mirror_shape(self.selected_shape)
                self.selected_shape = dup.name

        self.mutate(apply)

    def delete_selected(self, _e=None):
        mode = self.mode()
        if mode not in ("model", "parts") or self.selected_shape is None:
            return
        shape = self.selected_shape_obj()
        if shape is None:
            return
        self.selected_shape = None
        if mode == "parts" and shape.group:
            group = shape.group
            self.mutate(lambda scene: scene.remove_group(group))
        else:
            name = shape.name
            self.mutate(lambda scene: scene.remove_shape(name))

    def auto_bind_all(self):
        if not self.scene.armature.bones():
            messagebox.showinfo(
                "Auto-bind", "Add joints in the Rig tab first.", parent=self)
            return
        self.mutate(lambda scene: scene.auto_bind())

    # -- grab (move in view plane) ---------------------------------------

    def start_grab(self, _e=None):
        if self.mode() not in ("model", "parts") or self.selected_shape is None:
            return
        self.cancel_placement()
        if self.viewport.start_grab():
            self._grab_snapshot = self.scene.to_json()
            self.set_status("grab: move mouse, click/Enter to confirm, Esc to cancel")

    def _grab_move(self, name, delta):
        try:
            shape = self.scene.get_shape(name)
        except KeyError:
            return
        if self.mode() == "parts" and shape.group in self.scene.groups:
            # move the whole part instance, grafted joints included
            self.scene.translate_group(shape.group, delta)
        else:
            shape.position = shape.position + delta
        self.light_refresh()

    def _grab_end(self, name, committed):
        if self._grab_snapshot is None:
            return
        if committed:
            self.undo_stack.push(self._grab_snapshot)
        else:
            self.scene = Scene.from_json(self._grab_snapshot)
        self._grab_snapshot = None
        self.refresh()

    # -- part placement ---------------------------------------------------

    def arm_placement(self, info):
        """Enter snap-to-surface placement: the next viewport click stamps the
        part."""
        self._abort_grab()
        try:
            part = load_part(info)
            part.validate()
        except Exception as exc:
            messagebox.showerror("Place part", str(exc), parent=self)
            return
        self._placement = part
        self.viewport.configure(cursor="crosshair")
        self.set_status(
            f"click the model to place {part.name!r} — Esc cancels")

    def cancel_placement(self):
        if self._placement is not None:
            self._placement = None
            self.viewport.configure(cursor="")
            self._update_status()

    def _placement_click(self, x, y) -> bool:
        """Viewport pick override: returns True when the click was consumed by
        placement mode."""
        if self._placement is None or self.mode() != "parts":
            return False
        if min(self.viewport.winfo_width(), self.viewport.winfo_height()) < 10:
            return True
        origin, direction = self.viewport.screen_ray(x, y)
        named = [
            (shape.name, mesh)
            for shape, mesh in self.scene.build_shape_meshes(None)
        ]
        hit = raycast_meshes(named, origin, direction)
        if hit is None:
            self.set_status("missed — click on the model (Esc cancels)")
            return True
        shape_name, ray_hit = hit
        part = self._placement

        def apply(scene):
            group = place_part(
                scene, part,
                point=ray_hit.point, normal=ray_hit.normal,
                attach_shape=shape_name,
            )
            members = scene.group_members(group)
            if members:
                self.selected_shape = members[0].name

        self.cancel_placement()
        self.mutate(apply)
        return True

    def _escape(self, _e=None):
        if self._placement is not None:
            self.cancel_placement()
        elif self.viewport.grabbing:
            self.viewport.cancel_grab()
        else:
            self.select_shape(None)

    def _return_key(self, _e=None):
        if self.viewport.grabbing:
            self.viewport._confirm_grab()

    # -- refresh ----------------------------------------------------------

    def _viewport_content(self):
        """(items, joints, bones, show_joints) for the current mode, healing a
        dangling active pose along the way."""
        mode = self.mode()
        posed = mode in ("pose", "export")
        pose_name = "__active__" if posed else None
        try:
            shape_meshes = self.scene.build_shape_meshes(pose_name)
        except KeyError:
            self.scene.active_pose = None
            shape_meshes = self.scene.build_shape_meshes(None)
        items = [
            RenderItem(shape.name, mesh.vertices, mesh.faces, shape.color)
            for shape, mesh in shape_meshes
        ]
        joints: dict = {}
        bones: list = []
        show_joints = mode in ("rig", "pose")
        if show_joints:
            pose = self.scene.resolve_pose(pose_name) if posed else {}
            joints = self.scene.armature.posed_positions(pose)
            bones = self.scene.armature.bones()
        return items, joints, bones, show_joints

    def refresh(self):
        mode = self.mode()
        items, joints, bones, show_joints = self._viewport_content()
        group_shapes: set[str] = set()
        if mode == "parts":
            shape = self.selected_shape_obj()
            if shape is not None and shape.group:
                group_shapes = {s.name for s in self.scene.group_members(shape.group)}
        self.viewport.set_selection(
            shape=self.selected_shape if mode in ("model", "parts") else None,
            joint=self.selected_joint if show_joints else None,
            shapes=group_shapes,
            redraw=False,
        )
        self.viewport.set_content(items, joints, bones, show_joints)
        for panel in (self.model_panel, self.parts_panel, self.rig_panel,
                      self.pose_panel, self.export_panel):
            panel.sync()
        self._update_status()

    def light_refresh(self):
        """Viewport-only refresh during continuous edits (sliders, grabs)."""
        self.viewport.set_content(*self._viewport_content())

    def set_status(self, text: str):
        self.status.configure(text=text)

    def _update_status(self):
        mode = self.mode()
        bits = [f"{self.scene.name}"]
        if self.path:
            bits.append(self.path.name)
        bits.append(f"{len(self.scene.shapes)} shapes / "
                    f"{len(self.scene.armature)} joints")
        if self._placement is not None:
            bits.append(f"placing {self._placement.name!r}: click the model  "
                        "(Esc cancels)")
        elif mode == "parts" and self.selected_shape:
            shape = self.selected_shape_obj()
            if shape is not None and shape.group:
                bits.append(f"part: {shape.group}  "
                            "(g=grab  m=mirror  x=delete)")
            else:
                bits.append(f"selected: {self.selected_shape}")
        elif mode == "model" and self.selected_shape:
            bits.append(f"selected: {self.selected_shape}  "
                        "(g=grab  Ctrl+D=dup  m=mirror  x=delete)")
        elif mode in ("rig", "pose") and self.selected_joint:
            bits.append(f"joint: {self.selected_joint}")
        else:
            bits.append("drag=orbit  mid/right-drag=pan  wheel=zoom  click=select")
        if self.scene.active_pose and mode in ("pose", "export"):
            bits.append(f"pose: {self.scene.active_pose}")
        self.set_status("   |   ".join(bits))

    # -- files ------------------------------------------------------------

    def _build_menu(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="New (blank)", command=self.new_blank)
        template_menu = tk.Menu(filemenu, tearoff=0)
        for name in list_templates():
            template_menu.add_command(
                label=name, command=lambda n=name: self.new_from_template(n))
        filemenu.add_cascade(label="New from template", menu=template_menu)
        filemenu.add_separator()
        filemenu.add_command(label="Open…", command=self.open_dialog,
                             accelerator="Ctrl+O")
        filemenu.add_command(label="Save", command=self.save, accelerator="Ctrl+S")
        filemenu.add_command(label="Save As…", command=self.save_as)
        filemenu.add_separator()
        filemenu.add_command(label="Export STL…",
                             command=lambda: self.export_panel._export())
        filemenu.add_separator()
        filemenu.add_command(label="Quit", command=self.destroy)
        menubar.add_cascade(label="File", menu=filemenu)

        editmenu = tk.Menu(menubar, tearoff=0)
        editmenu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        editmenu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")
        editmenu.add_separator()
        editmenu.add_command(label="Duplicate shape", command=self.duplicate_selected,
                             accelerator="Ctrl+D")
        editmenu.add_command(label="Mirror shape", command=self.mirror_selected,
                             accelerator="m")
        editmenu.add_command(label="Delete shape", command=self.delete_selected,
                             accelerator="x")
        menubar.add_cascade(label="Edit", menu=editmenu)

        viewmenu = tk.Menu(menubar, tearoff=0)
        viewmenu.add_command(label="Frame content",
                             command=self.viewport.frame_content, accelerator="f")
        menubar.add_cascade(label="View", menu=viewmenu)

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self._about)
        menubar.add_cascade(label="Help", menu=helpmenu)
        self.configure(menu=menubar)

    def _bind_keys(self):
        self.bind("<Control-z>", self.undo)
        self.bind("<Control-y>", self.redo)
        self.bind("<Control-d>", self.duplicate_selected)
        self.bind("<Control-o>", lambda e: self.open_dialog())
        self.bind("<Control-s>", lambda e: self.save())
        self.bind("<Escape>", self._escape)
        self.bind("<Return>", self._return_key)
        for key, fn in (("g", self.start_grab), ("m", self.mirror_selected),
                        ("x", self.delete_selected),
                        ("f", lambda e: self.viewport.frame_content())):
            self.bind(key, self._only_in_viewport(fn))
        self.bind("<Delete>", self._only_in_viewport(self.delete_selected))

    def _only_in_viewport(self, fn):
        def handler(e):
            if isinstance(self.focus_get(), (tk.Entry, ttk.Entry, tk.Text)):
                return  # don't steal keys from text fields
            fn(e)

        return handler

    def _confirm_discard(self) -> bool:
        return messagebox.askokcancel(
            "MiniMaster", "Discard the current scene?", parent=self)

    def new_blank(self):
        if not self._confirm_discard():
            return
        self.cancel_placement()
        self.scene = Scene(name="untitled")
        self.path = None
        self.undo_stack = UndoStack()
        self.selected_shape = self.selected_joint = None
        self.refresh()
        self.viewport.frame_content()

    def new_from_template(self, name: str):
        if not self._confirm_discard():
            return
        self.cancel_placement()
        self.scene = load_template(name)
        self.path = None
        self.undo_stack = UndoStack()
        self.selected_shape = self.selected_joint = None
        self.refresh()
        self.viewport.frame_content()

    def open_dialog(self):
        path = filedialog.askopenfilename(
            parent=self, filetypes=[("MiniMaster project", "*.mmp"), ("All", "*.*")])
        if not path:
            return
        try:
            scene = Scene.load(path)
        except Exception as exc:
            messagebox.showerror("Open", f"Could not open {path}:\n{exc}", parent=self)
            return
        self.cancel_placement()
        self.scene = scene
        self.path = Path(path)
        self.undo_stack = UndoStack()
        self.selected_shape = self.selected_joint = None
        self.refresh()
        self.viewport.frame_content()

    def save(self):
        if self.path is None:
            self.save_as()
            return
        self.scene.save(self.path)
        self.set_status(f"saved {self.path}")

    def save_as(self):
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension=".mmp",
            filetypes=[("MiniMaster project", "*.mmp")])
        if not path:
            return
        self.path = Path(path)
        self.scene.save(self.path)
        self.set_status(f"saved {self.path}")

    def export_stl_dialog(self, options: dict):
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension=".stl", filetypes=[("STL", "*.stl")])
        if not path:
            return
        try:
            report = export_stl(self.scene, path, **options)
        except (ExportError, KeyError) as exc:
            messagebox.showerror("Export", str(exc), parent=self)
            return
        d = report["size_mm"]
        messagebox.showinfo(
            "Export",
            f"Wrote {path}\n{report['triangles']} triangles, "
            f"{d[0]:.1f} × {d[1]:.1f} × {d[2]:.1f} mm",
            parent=self)

    def save_png_dialog(self, options: dict):
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension=".png", filetypes=[("PNG", "*.png")])
        if not path:
            return
        pose = options.get("pose_name", "__active__")
        try:
            render_scene(self.scene, path=path, pose_name=pose,
                         with_base=options.get("with_base", True),
                         azimuth=self.viewport.azimuth,
                         elevation=self.viewport.elevation)
        except (ValueError, KeyError) as exc:
            messagebox.showerror("Save PNG", str(exc), parent=self)
            return
        self.set_status(f"wrote {path}")

    def _about(self):
        messagebox.showinfo(
            "About MiniMaster",
            f"MiniMaster {__version__}\n\n"
            "Design, rig, pose, and export 3D-printable miniatures.\n"
            "Shapes → bones → poses → STL.",
            parent=self)


def run_app(scene: Scene | None = None):
    app = MiniMasterApp(scene)
    app.mainloop()
