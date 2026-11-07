"""A tk Canvas that shows a rendered numpy image and takes the camera input (needs tkinter).

The Character tab does not draw triangles on a canvas like the studio's
:class:`~minimaster.viewport.Viewport` (that painter manages 3 fps on a feathered wing); it shows
the picture :mod:`minimaster.character.lab` renders -- a draft proxy while the user is dragging,
the full render when they stop -- through a ``tk.PhotoImage``, and keeps everything that is not
the picture as ordinary canvas items on top of it: the ground grid, an anchor marker for every
appendage (the selected one highlighted), a small badge saying which tier is on screen.

The input conventions are the Viewport's: left drag orbits, middle / right drag pans, the wheel
zooms (Button-4/5 on X11), a left press and release that moves 3 px or less is a CLICK (the
panel turns it into a pick: select an appendage, or place one on the body). The widget does not
know what the gestures mean: it calls back with pixel deltas, and the lab owns the camera.

    view = ImageView(parent)
    view.bind_lab(lab)          # frames in, camera gestures / clicks / resizes out

Everything runs on the UI thread. Frames arrive as :class:`~minimaster.character.lab.Frame`
objects whose ``ppm()`` bytes were prepared elsewhere (on the render worker for the final tier),
so showing one is a single ``PhotoImage.configure(data=...)`` (about 15 ms at 900x800).
"""

from __future__ import annotations

import tkinter as tk

import numpy as np

BG = "#20242c"               # the renderer's background: an unrendered pane looks the same
GRID = "#2c313b"
GRID_AXIS = "#3b4250"
MARK = "#8da2b8"
MARK_SELECTED = "#ffcc33"
TEXT = "#8f9aa8"
CLICK_SLOP = 3               # pixels: more movement than this is a drag, not a click
GRID_STEP = 2.0              # world units (1 unit = 10 cm at human scale)
GRID_EXTENT = 12.0


def _visible_runs(x0, y0, x1, y1, img, scale, step: float = 4.0):
    """The parts of the segment (x0, y0)-(x1, y1) that lie over background pixels of ``img`` (an
    (h, w, 4) uint8 render at 1/``scale`` of the display size, background colour in its corner), as
    ``[(xa, ya, xb, yb)]``. With no image the whole segment."""
    if img is None:
        return [(x0, y0, x1, y1)]
    n = max(int(np.hypot(x1 - x0, y1 - y0) / step), 1) + 1
    t = np.linspace(0.0, 1.0, n)
    xs, ys = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
    h, w = img.shape[:2]
    ix, iy = (xs // scale).astype(int), (ys // scale).astype(int)
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    ix, iy = np.clip(ix, 0, w - 1), np.clip(iy, 0, h - 1)
    # the background is whatever the renderer painted in the corner (tone mapping makes it
    # differ a little from the hex colour); anything else under the line is the figure
    bg = np.all(img[iy, ix, :3] == img[0, 0, :3], axis=1)
    vis = bg & inside
    edges = np.flatnonzero(np.diff(np.r_[0, vis.astype(np.int8), 0]))
    return [(xs[a], ys[a], xs[b - 1], ys[b - 1]) for a, b in zip(edges[::2], edges[1::2]) if b - 1 > a]


class ImageView(tk.Canvas):
    def __init__(self, master, **kwargs):
        kwargs.setdefault("width", 640)
        kwargs.setdefault("height", 560)
        super().__init__(master, bg=BG, highlightthickness=0, **kwargs)
        # gesture callbacks (pixel deltas / a zoom factor / a pixel): set by bind_lab or the host
        self.on_orbit = None
        self.on_pan = None
        self.on_zoom = None
        self.on_click = None
        self.on_resize = None
        self.on_cancel = None
        # what to draw over the picture: () -> (view state, marks, floor z) or None
        self.overlay_provider = None
        self.show_grid = True
        self.banner = ""                 # a line of text in the corner (e.g. "click the body")

        self.frame = None                # the last Frame shown
        self.frames_shown = 0
        self._photo: tk.PhotoImage | None = None
        self._item = None
        self._shown_size = (0, 0)
        self._drag = None
        self._press = (0, 0)
        self._moved = False
        self._button = 0
        self._resize_job = None

        self.bind("<ButtonPress-1>", self._press1)
        self.bind("<B1-Motion>", self._motion1)
        self.bind("<ButtonRelease-1>", self._release1)
        for button in (2, 3):
            self.bind(f"<ButtonPress-{button}>", self._press_pan)
            self.bind(f"<B{button}-Motion>", self._motion_pan)
        self.bind("<MouseWheel>", self._wheel)                      # Windows / macOS
        self.bind("<Button-4>", lambda e: self._zoom(0.9))         # X11
        self.bind("<Button-5>", lambda e: self._zoom(1.1))
        self.bind("<Configure>", self._on_configure)
        self.bind("<Escape>", lambda e: self.on_cancel and self.on_cancel())

    # -- wiring ------------------------------------------------------------------------------

    def bind_lab(self, lab) -> None:
        """Frames from ``lab`` come in; gestures go to it. The panel adds the click rule."""
        lab.on_frame = self.show_frame
        self.on_orbit = lab.orbit
        self.on_pan = lab.pan
        self.on_zoom = lab.zoom
        self.on_resize = lab.set_size
        self.overlay_provider = lambda: (lab.view, lab.overlay_points(), lab.floor_z())

    @property
    def size(self) -> tuple:
        return (max(self.winfo_width(), 1), max(self.winfo_height(), 1))

    # -- the picture -------------------------------------------------------------------------

    def show_frame(self, frame) -> None:
        """Put a lab Frame on screen (the picture) and redraw the items over it."""
        data = frame.ppm()
        w, h = frame.size
        if self._photo is None:
            self._photo = tk.PhotoImage(width=w, height=h)
            self._item = self.create_image(0, 0, anchor="nw", image=self._photo)
        self._photo.configure(data=data)
        self._shown_size = (w, h)
        self.frame = frame
        self.frames_shown += 1
        self._draw_overlay()

    def refresh_overlay(self) -> None:
        """Redraw only the canvas items (selection changed, banner text changed)."""
        self._draw_overlay()

    def _draw_overlay(self) -> None:
        self.delete("overlay")
        if self._item is not None:
            self.tag_lower(self._item)
        provider = self.overlay_provider
        data = provider() if provider is not None else None
        w, h = self._shown_size
        if data is not None and w > 1:
            view, marks, floor = data
            if self.show_grid:
                self._grid(view, floor, (w, h))
            self._marks(view, marks, (w, h))
        tier = self.frame.tier if self.frame is not None else ""
        text = "  ".join(t for t in (tier, self.banner) if t)
        if text:
            self.create_text(8, 8, anchor="nw", text=text, fill=TEXT, tags=("overlay", "badge"),
                             font=("TkDefaultFont", 9))
        self.tag_raise("overlay")

    def _grid(self, view, floor: float, size) -> None:
        """The ground grid. It is a canvas overlay, so it would be drawn over the figure's legs;
        each line is therefore cut wherever the picture under it is not background."""
        n = int(GRID_EXTENT / GRID_STEP)
        ticks = [i * GRID_STEP for i in range(-n, n + 1)]
        lines = []
        for t in ticks:
            lines.append(((t, -GRID_EXTENT, floor), (t, GRID_EXTENT, floor), t == 0))
            lines.append(((-GRID_EXTENT, t, floor), (GRID_EXTENT, t, floor), t == 0))
        pts = np.array([p for a, b, _ in lines for p in (a, b)], dtype=np.float64)
        sx, sy, depth = view.project(pts, size)
        img, scale = (self.frame.image, self.frame.scale) if self.frame is not None else (None, 1)
        for i, (_, _, axis) in enumerate(lines):
            a, b = 2 * i, 2 * i + 1
            if depth[a] <= 1e-6 or depth[b] <= 1e-6:
                continue
            color = GRID_AXIS if axis else GRID
            for x0, y0, x1, y1 in _visible_runs(sx[a], sy[a], sx[b], sy[b], img, scale):
                self.create_line(x0, y0, x1, y1, fill=color, tags=("overlay", "grid"))

    def _marks(self, view, marks, size) -> None:
        if not marks:
            return
        pts = np.array([p for _, p, _ in marks], dtype=np.float64)
        sx, sy, depth = view.project(pts, size)
        named = set()
        for (label, _, selected), x, y, d in zip(marks, sx, sy, depth):
            if d <= 1e-6:
                continue
            r = 6 if selected else 4
            color = MARK_SELECTED if selected else MARK
            self.create_oval(x - r, y - r, x + r, y + r, outline=color, width=2 if selected else 1,
                             tags=("overlay", "mark", f"mark:{label}"))
            name = label.split(".")[0]                  # wing-1.l and wing-1.r: one label
            if selected and name not in named:
                named.add(name)
                self.create_text(x + r + 3, y - r, anchor="sw", text=name, fill=color,
                                 tags=("overlay", "mark", f"mark:{label}"),
                                 font=("TkDefaultFont", 9))

    # -- input -------------------------------------------------------------------------------

    def _press1(self, e) -> None:
        self.focus_set()
        self._drag = (e.x, e.y)
        self._press = (e.x, e.y)
        self._button = 1
        self._moved = False

    def _motion1(self, e) -> None:
        if self._drag is None or self._button != 1:
            return
        if abs(e.x - self._press[0]) + abs(e.y - self._press[1]) > CLICK_SLOP:
            self._moved = True
        if self._moved and self.on_orbit:
            self.on_orbit(e.x - self._drag[0], e.y - self._drag[1])
        if self._moved:
            self._drag = (e.x, e.y)

    def _release1(self, e) -> None:
        if self._drag is not None and not self._moved and self.on_click:
            self.on_click(e.x, e.y)
        self._drag = None
        self._button = 0

    def _press_pan(self, e) -> None:
        self._drag = (e.x, e.y)
        self._button = e.num

    def _motion_pan(self, e) -> None:
        if self._drag is None or self._button not in (2, 3):
            return
        dx, dy = e.x - self._drag[0], e.y - self._drag[1]
        self._drag = (e.x, e.y)
        if self.on_pan and (dx or dy):
            self.on_pan(dx, dy)

    def _wheel(self, e) -> None:
        self._zoom(0.9 if e.delta > 0 else 1.1)

    def _zoom(self, factor: float) -> None:
        if self.on_zoom:
            self.on_zoom(factor)

    def _on_configure(self, e) -> None:
        """The pane was resized: tell the lab (once the resizing settles, so dragging a window
        edge does not queue a render per pixel)."""
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(40, self._resized)

    def _resized(self) -> None:
        self._resize_job = None
        if self.on_resize:
            w, h = self.size
            self.on_resize(w, h)

    def destroy(self) -> None:
        if self._resize_job is not None:
            try:
                self.after_cancel(self._resize_job)
            except tk.TclError:
                pass
        self.on_orbit = self.on_pan = self.on_zoom = self.on_click = self.on_resize = None
        self.on_cancel = self.overlay_provider = None
        super().destroy()
        self._photo = None                 # the Tcl image goes now, on this thread
