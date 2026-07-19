"""Interactive 3D viewport on a tk Canvas.

Painter's-algorithm flat shading: triangles are depth-sorted and drawn as
canvas polygons (tk fills them in C, which is plenty fast for carved low-poly
scenes). Each polygon carries its shape name as a canvas tag, so picking is a
canvas hit test. Joints render as markers on top in rig/pose modes.

Controls: left-drag orbits, middle/right-drag pans, wheel zooms, left-click
selects (shapes or joints), `g` grabs the selected shape (move in view plane,
click/Return confirms, Esc cancels).
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass

import numpy as np

from .core import math3d as m3
from .core import raycast as rc

BG = "#1e1d1b"
GRID = "#33312e"
GRID_AXIS = "#4a4741"
BONE = "#e8b34b"
BONE_SELECTED = "#ffe08a"
JOINT = "#f0f0f0"
JOINT_SELECTED = "#ffcc33"
OUTLINE_SELECTED = "#ffcc33"


def shade_hex(color: str, factor: float) -> str:
    c = color.lstrip("#")
    rgb = [int(c[i : i + 2], 16) for i in (0, 2, 4)]
    rgb = [max(0, min(255, int(v * factor))) for v in rgb]
    return "#%02x%02x%02x" % tuple(rgb)


_SHADE_LEVELS = 32
_SHADE_LUT: dict[str, list[str]] = {}


def _shade_lut(color: str) -> list[str]:
    """Precomputed Lambert shades per color, so redraw never formats hex
    strings per triangle."""
    lut = _SHADE_LUT.get(color)
    if lut is None:
        lut = [
            shade_hex(color, 0.35 + 0.65 * i / (_SHADE_LEVELS - 1))
            for i in range(_SHADE_LEVELS)
        ]
        _SHADE_LUT[color] = lut
    return lut


@dataclass
class RenderItem:
    name: str | None  # shape name for picking, None = not pickable
    vertices: np.ndarray
    faces: np.ndarray
    color: str


class Viewport(tk.Canvas):
    def __init__(self, master, on_select_shape=None, on_select_joint=None,
                 on_grab_move=None, on_grab_end=None, **kwargs):
        super().__init__(master, bg=BG, highlightthickness=0, **kwargs)
        self.on_select_shape = on_select_shape
        self.on_select_joint = on_select_joint
        self.on_grab_move = on_grab_move  # (name, world_delta) during grab
        self.on_grab_end = on_grab_end  # (name, committed: bool)

        self.azimuth = 335.0
        self.elevation = 18.0
        self.distance = 90.0
        self.target = np.array([0.0, 0.0, 12.0])
        self.fov = 30.0

        self.items: list[RenderItem] = []
        self.joints: dict[str, np.ndarray] = {}
        self.bones: list[tuple[str, str]] = []
        self.show_joints = False
        self.selected_shape: str | None = None
        self.selected_shapes: set[str] = set()  # extra outlines (part groups)
        self.selected_joint: str | None = None
        # When set, gets first crack at clicks: pick_override(x, y) -> bool
        # (True = consumed). Used by part placement mode.
        self.pick_override = None

        self._light = np.array([-0.45, -0.6, 0.75])
        self._light = self._light / np.linalg.norm(self._light)
        self._drag: tuple[int, int] | None = None
        self._press_pos = (0, 0)
        self._drag_button = 0
        self._drag_moved = False
        self._grab: dict | None = None

        self.bind("<ButtonPress-1>", self._press1)
        self.bind("<B1-Motion>", self._motion1)
        self.bind("<ButtonRelease-1>", self._release1)
        for btn in (2, 3):
            self.bind(f"<ButtonPress-{btn}>", self._press_pan)
            self.bind(f"<B{btn}-Motion>", self._motion_pan)
        self.bind("<MouseWheel>", self._wheel)  # Windows/mac
        self.bind("<Button-4>", lambda e: self._zoom(0.9))
        self.bind("<Button-5>", lambda e: self._zoom(1.1))
        self.bind("<Motion>", self._motion_free)
        self.bind("<Configure>", lambda e: self.redraw())

    # -- public API -------------------------------------------------------

    def set_content(self, items, joints=None, bones=None, show_joints=False):
        self.items = items
        self.joints = joints or {}
        self.bones = bones or []
        self.show_joints = show_joints
        self.redraw()

    def set_selection(self, shape: str | None = None, joint: str | None = None,
                      shapes: set[str] | None = None, redraw: bool = True):
        self.selected_shape = shape
        self.selected_shapes = shapes or set()
        self.selected_joint = joint
        if redraw:
            self.redraw()

    def screen_ray(self, x: float, y: float):
        """World-space (origin, direction) for a canvas pixel."""
        view, focal = self._camera()
        return rc.screen_ray(
            x, y, max(self.winfo_width(), 1), max(self.winfo_height(), 1),
            view, focal,
        )

    def frame_content(self):
        """Center and fit the current content."""
        pts = [it.vertices for it in self.items if len(it.vertices)]
        if self.joints:
            pts.append(np.array(list(self.joints.values())))
        if not pts:
            return
        all_pts = np.vstack(pts)
        lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
        self.target = (lo + hi) / 2.0
        radius = max(float(np.linalg.norm(hi - lo)) / 2.0, 5.0)
        self.distance = radius / np.tan(np.radians(self.fov) / 2.0) * 1.3
        self.redraw()

    def start_grab(self):
        if self.selected_shape is None or self._grab is not None:
            return False
        self._grab = {"name": self.selected_shape, "last": None, "total": np.zeros(3)}
        self.configure(cursor="fleur")
        return True

    def cancel_grab(self):
        if self._grab is None:
            return
        grab = self._grab
        self._grab = None
        self.configure(cursor="")
        if self.on_grab_end:
            self.on_grab_end(grab["name"], False)

    def _confirm_grab(self):
        grab = self._grab
        self._grab = None
        self.configure(cursor="")
        if self.on_grab_end:
            self.on_grab_end(grab["name"], True)

    @property
    def grabbing(self) -> bool:
        return self._grab is not None

    # -- camera -----------------------------------------------------------

    def _eye(self) -> np.ndarray:
        az, el = np.radians(self.azimuth), np.radians(self.elevation)
        return self.target + self.distance * np.array(
            [np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)]
        )

    def _camera(self):
        view = m3.look_at(self._eye(), self.target)
        h = max(self.winfo_height(), 1)
        focal = (h / 2.0) / np.tan(np.radians(self.fov) / 2.0)
        return view, focal

    def _camera_axes(self):
        view, _ = self._camera()
        rot = view[:3, :3]
        return rot[0], rot[1]  # camera right/up in world space

    # -- input ------------------------------------------------------------

    def _press1(self, e):
        if self._grab is not None:
            self._confirm_grab()
            return
        self._drag = (e.x, e.y)
        self._press_pos = (e.x, e.y)
        self._drag_button = 1
        self._drag_moved = False

    def _motion1(self, e):
        if self._grab is not None:
            self._motion_free(e)
            return
        if self._drag is None:
            return
        dx, dy = e.x - self._drag[0], e.y - self._drag[1]
        # Click-vs-drag is judged from the press origin, not per event, so a
        # slow orbit can't masquerade as a click on release.
        if abs(e.x - self._press_pos[0]) + abs(e.y - self._press_pos[1]) > 3:
            self._drag_moved = True
        self.azimuth = (self.azimuth + dx * 0.5) % 360.0
        self.elevation = float(np.clip(self.elevation + dy * 0.4, -89.0, 89.0))
        self._drag = (e.x, e.y)
        self.redraw()

    def _release1(self, e):
        if self._drag is not None and not self._drag_moved:
            self._pick(e.x, e.y)
        self._drag = None

    def _press_pan(self, e):
        self._drag = (e.x, e.y)

    def _motion_pan(self, e):
        if self._drag is None:
            return
        dx, dy = e.x - self._drag[0], e.y - self._drag[1]
        right, up = self._camera_axes()
        _, focal = self._camera()
        scale = self.distance / focal
        self.target = self.target - right * dx * scale + up * dy * scale
        self._drag = (e.x, e.y)
        self.redraw()

    def _wheel(self, e):
        self._zoom(0.9 if e.delta > 0 else 1.1)

    def _zoom(self, factor: float):
        self.distance = float(np.clip(self.distance * factor, 2.0, 2000.0))
        self.redraw()

    def _motion_free(self, e):
        if self._grab is None:
            return
        if self._grab["last"] is None:
            self._grab["last"] = (e.x, e.y)
            return
        dx = e.x - self._grab["last"][0]
        dy = e.y - self._grab["last"][1]
        self._grab["last"] = (e.x, e.y)
        right, up = self._camera_axes()
        _, focal = self._camera()
        scale = self.distance / focal
        delta = right * dx * scale - up * dy * scale
        self._grab["total"] = self._grab["total"] + delta
        if self.on_grab_move:
            self.on_grab_move(self._grab["name"], delta)

    def _pick(self, x, y):
        if self.pick_override is not None and self.pick_override(x, y):
            return
        # Joints first (drawn on top).
        if self.show_joints:
            hits = self.find_overlapping(x - 4, y - 4, x + 4, y + 4)
            for item in reversed(hits):
                for tag in self.gettags(item):
                    if tag.startswith("joint:"):
                        if self.on_select_joint:
                            self.on_select_joint(tag[6:])
                        return
        hits = self.find_overlapping(x, y, x, y)
        for item in reversed(hits):  # topmost polygon first
            for tag in self.gettags(item):
                if tag.startswith("shape:"):
                    if self.on_select_shape:
                        self.on_select_shape(tag[6:])
                    return
        if self.on_select_shape:
            self.on_select_shape(None)

    # -- drawing ----------------------------------------------------------

    def _project(self, pts: np.ndarray, view, focal):
        cam = m3.transform_points(view, pts)
        depth = -cam[:, 2]
        w = max(self.winfo_width(), 1)
        h = max(self.winfo_height(), 1)
        safe = np.maximum(depth, 1e-6)
        sx = w / 2.0 + focal * cam[:, 0] / safe
        sy = h / 2.0 - focal * cam[:, 1] / safe
        return sx, sy, depth

    def redraw(self):
        self.delete("all")
        view, focal = self._camera()
        self._draw_grid(view, focal)

        polys = []  # (depth, screen coords, fill, outline, shape name)
        eye = self._eye()
        # Camera-relative key light: the model stays readable from any orbit.
        light = view[:3, :3].T @ np.array([-0.35, 0.45, 0.82])
        light = light / np.linalg.norm(light)
        for it in self.items:
            if not len(it.faces):
                continue
            sx, sy, depth = self._project(it.vertices, view, focal)
            keep_v = depth > 1e-6
            tris = it.faces
            t = it.vertices[tris]
            n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            lens = np.linalg.norm(n, axis=1, keepdims=True)
            n = np.divide(n, lens, out=np.zeros_like(n), where=lens > 1e-14)
            facing = np.einsum("ij,ij->i", n, t.mean(axis=1) - eye) < 0
            levels = (
                np.clip(n @ light, 0.0, 1.0) * (_SHADE_LEVELS - 1)
            ).astype(int)
            lut = _shade_lut(it.color)
            td = depth[tris].mean(axis=1)
            selected = it.name is not None and (
                it.name == self.selected_shape or it.name in self.selected_shapes
            )
            for i in np.nonzero(facing)[0]:
                f = tris[i]
                if not keep_v[f].all():
                    continue
                coords = (sx[f[0]], sy[f[0]], sx[f[1]], sy[f[1]], sx[f[2]], sy[f[2]])
                fill = lut[levels[i]]
                outline = OUTLINE_SELECTED if selected else fill
                polys.append((float(td[i]), coords, fill, outline, it.name))

        polys.sort(key=lambda p: -p[0])  # far to near
        for _, coords, fill, outline, name in polys:
            tags = ("shape:" + name,) if name else ()
            self.create_polygon(*coords, fill=fill, outline=outline, width=1, tags=tags)

        if self.show_joints and self.joints:
            self._draw_armature(view, focal)

    def _draw_grid(self, view, focal, extent=30.0, step=5.0):
        n = int(extent / step)
        for i in range(-n, n + 1):
            a = np.array([[i * step, -extent, 0], [i * step, extent, 0]])
            b = np.array([[-extent, i * step, 0], [extent, i * step, 0]])
            for seg in (a, b):
                sx, sy, depth = self._project(seg, view, focal)
                if (depth <= 1e-6).any():
                    continue
                color = GRID_AXIS if i == 0 else GRID
                self.create_line(sx[0], sy[0], sx[1], sy[1], fill=color)

    def _draw_armature(self, view, focal):
        pos = self.joints
        for parent, child in self.bones:
            a, b = pos.get(parent), pos.get(child)
            if a is None or b is None:
                continue
            pts = np.vstack([a, b])
            sx, sy, depth = self._project(pts, view, focal)
            if (depth <= 1e-6).any():
                continue
            selected = child == self.selected_joint
            self.create_line(
                sx[0], sy[0], sx[1], sy[1],
                fill=BONE_SELECTED if selected else BONE, width=3 if selected else 2,
            )
        for name, p in pos.items():
            sx, sy, depth = self._project(p.reshape(1, 3), view, focal)
            if depth[0] <= 1e-6:
                continue
            r = 4 if name == self.selected_joint else 3
            color = JOINT_SELECTED if name == self.selected_joint else JOINT
            self.create_oval(
                sx[0] - r, sy[0] - r, sx[0] + r, sy[0] + r,
                fill=color, outline="#222222", tags=("joint:" + name,),
            )
