"""The Character Lab core behind the interactive "Character" tab (no tkinter).

Everything the tab does is decided here, so it can be driven -- and tested -- without a
display; ``character_panel.py`` and ``imageview.py`` only translate widgets into calls on
:class:`Lab`. Design context: ``docs/appendages_system_design.md`` (D1, D7 steps 3-4, D9)
and ``docs/character_tab.md``.

====================  ============================================================================
:class:`LabDocument`  the :class:`~minimaster.character.sliders.Character` plus every edit that is
                      allowed on it: add / remove / duplicate / rename / enable an appendage, set a
                      parameter, control, placement field or side (validated against the kind's
                      ``ParamSpec`` ranges, ``when=`` conditions and the PLACED appendage's
                      ``control_range``), move an anchor to a picked body quad, body sliders, the
                      anatomy switch; snapshot undo / redo (JSON strings in a
                      :class:`~minimaster.scene.UndoStack`) with drag coalescing; load / save
                      ``.mmchar``. Every edit announces a :class:`Change` saying what must be redone.
:class:`Engine`       document -> pictures. Caches by content (effective body weights -> morphed
                      vertices; spec + body -> built appendage; pose -> mesh), so a drag on one
                      appendage rebuilds only that appendage. Draft scenes (``BodyProxy`` +
                      ``ClusterProxy`` of the posed appendages), final scenes, picking, and the
                      STL export with the printability report.
:class:`PreviewScheduler`
                      the two tiers: a DRAFT frame on the caller's thread at once, a FINAL frame on
                      a cancellable worker after ~150 ms of quiet (latest-wins slot, generation
                      counter, ``check()`` between shells and buckets, frozen snapshot in, bytes
                      out; the worker never touches tkinter).
:class:`Lab`          the controller the panel holds: document + view + engine + scheduler.
====================  ============================================================================

The frames: the camera is ``fit=(center, radius)`` of a :class:`ViewState` (orbit changes the
angles, pan moves the center, the wheel scales the radius), identical for the draft, the final,
the id buffer and the overlays, so what is drawn, picked and annotated always agree.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import queue
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..scene import UndoStack
from . import raster as R
from .appendage_spec import SIDES, AppendageSpec, next_id
from .sliders import Character

# the body proxy triangle budget of the draft tier and the draft's resolution rule
DRAFT_BODY_TRIANGLES = 4000
DRAFT_FULL_PIXELS = 420_000           # larger panes draw the draft at half resolution
DEBOUNCE_SECONDS = 0.15               # quiet time before the final render starts
BUILT_CACHE_BYTES = 192 << 20         # the built appendages (rigs, poses, proxies) kept for reuse
COALESCE_SECONDS = 0.6                # edits of one control closer together are one undo step
BACKGROUND = "#20242c"                # render_portrait's default background

# numeric placement fields and the range the tab lets them take
PLACEMENT_RANGES = {"scale": (0.25, 3.0), "roll": (-180.0, 180.0), "offset": (-3.0, 3.0)}
MACRO_KEYS = ("gender", "age", "muscle", "weight", "height", "african", "asian", "caucasian")

# camera conventions of viewport.py: 0.5 deg per pixel of azimuth, 0.4 of elevation
ORBIT_AZIMUTH = 0.5
ORBIT_ELEVATION = 0.4
VIEW_PRESETS = {"front": (0.0, 6.0), "back": (180.0, 6.0), "left": (90.0, 6.0),
                "right": (270.0, 6.0), "three-quarter": (25.0, 6.0), "rear-quarter": (155.0, 10.0)}


class LabError(ValueError):
    """An edit the document refuses (unknown id, value out of range, parameter that does not
    apply ...). The message is meant to be shown to the person. The document is unchanged."""


# --------------------------------------------------------------------------
# what the Add menu offers


@dataclass(frozen=True)
class AddEntry:
    """One line of the Add menu: a kind, the parameters that make this variant (a wing's style,
    a harvested limb) and optionally a placement preset other than the kind's default."""

    label: str
    kind: str
    params: tuple = ()             # ((key, value), ...): hashable
    preset: str | None = None

    @property
    def param_dict(self) -> dict:
        return dict(self.params)


def add_menu() -> list:
    """The entries of the Add menu, generated from the registry: one per kind, split by the kind's
    essential choice parameter (a wing's style, a harvested limb's limb) and one more for every
    placement preset the default placement does not already reach."""
    from . import kinds
    out = []
    for kind in kinds.KINDS.values():
        reached = set()
        choices = [p for p in kind.params if p.essential and p.type == "choice"]
        variants = [{choices[0].key: c} for c in choices[0].choices] if choices else [{}]
        for params in variants:
            reached.add(kind.default_preset(kind.defaults(**params)))
            tag = f" ({next(iter(params.values()))})" if params else ""
            out.append(AddEntry(f"{kind.label}{tag}", kind.id, tuple(params.items())))
        for name, preset in kind.presets.items():
            if name not in reached:
                out.append(AddEntry(f"{kind.label} @ {preset.get('label', name)}", kind.id,
                                    (), name))
    return out


def pose_presets(kind) -> tuple:
    """The named poses a kind with pose controls offers (a wing's ``folded``, ``upstroke`` ...)."""
    flag = getattr(kind, "pose_presets", None)
    if isinstance(flag, (tuple, list)):          # a kind that names its own
        return tuple(flag)
    # kinds.Kind.pose_presets is a flag (does --pose ID:preset=NAME mean anything?): the wing's
    # names are wingrig.PRESETS. (A tuple was assumed here before T3 made it a bool, and the
    # tab raised TypeError for a tail; only a run with tkinter could see it.)
    if flag is True and getattr(kind, "id", "") == "wing":
        from .wingrig import PRESETS
        return tuple(PRESETS)
    return ()


# --------------------------------------------------------------------------
# resources


class Resources:
    """The MakeHuman base mesh, the morph-target library, the slider catalog and the draft
    body proxy. Loaded once (``Resources.shared()``) and shared by every lab in the process."""

    _shared: "Resources | None" = None
    _lock = threading.Lock()

    def __init__(self, base=None, lib=None):
        from . import mhbase
        if base is None or lib is None:
            base, lib = mhbase.load()
        from .sliders import SliderCatalog
        self.base, self.lib = base, lib
        self.catalog = SliderCatalog(lib.names)
        self._body_proxy = None
        self._proxy_lock = threading.Lock()

    @classmethod
    def available(cls) -> bool:
        from . import mhbase
        return mhbase.available()

    @classmethod
    def shared(cls) -> "Resources":
        with cls._lock:
            if cls._shared is None:
                cls._shared = cls()
            return cls._shared

    @property
    def body_proxy(self):
        with self._proxy_lock:
            if self._body_proxy is None:
                from .draft import BodyProxy
                self._body_proxy = BodyProxy(self.base, DRAFT_BODY_TRIANGLES)
            return self._body_proxy

    @property
    def n_quads(self) -> int:
        return len(self.base.body_quads)


# --------------------------------------------------------------------------
# small tools


class Cache:
    """A thread-safe LRU keyed by content, bounded by entries and (with ``sizeof``) by bytes.
    ``get_or_build`` computes OUTSIDE the lock, so a main-thread draft is never blocked by a
    worker building the same thing (at worst both build it once)."""

    def __init__(self, limit: int, max_bytes: int | None = None, sizeof=None):
        self.limit = limit
        self.max_bytes = max_bytes
        self.sizeof = sizeof
        self._items: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                self.hits += 1
                return self._items[key]
            self.misses += 1
            return None

    def peek(self, key):
        with self._lock:
            return self._items.get(key)

    def put(self, key, value):
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.limit:
                self._items.popitem(last=False)
            if self.max_bytes is not None and self.sizeof is not None:
                # sizes grow after insertion (a Built caches its poses): re-measure on every put
                while len(self._items) > 1 and \
                        sum(self.sizeof(v) for v in self._items.values()) > self.max_bytes:
                    self._items.popitem(last=False)
        return value

    def get_or_build(self, key, build):
        value = self.get(key)
        if value is None:
            value = self.put(key, build())
        return value

    def __len__(self):
        return len(self._items)

    def clear(self):
        with self._lock:
            self._items.clear()


def _digest(obj) -> str:
    return hashlib.blake2b(json.dumps(obj, sort_keys=True, default=str).encode(),
                           digest_size=12).hexdigest()


def spec_key(spec: AppendageSpec) -> str:
    """What decides how an appendage is BUILT: everything but its pose controls, its display
    name and its enabled flag (a pose drag must not rebuild it)."""
    d = spec.to_dict()
    for k in ("controls", "name", "enabled"):
        d.pop(k, None)
    return _digest(d)


def controls_key(controls: dict) -> str:
    return _digest(controls)


@dataclass(frozen=True)
class Change:
    """What an edit changed, so the preview redoes only what it must.

    ``scope``: ``view`` camera only; ``pose`` one appendage's controls; ``shape`` one
    appendage's build (parameters, placement, side, enabled, add, remove); ``body`` the body
    sliders (appendages are re-placed on the new body); ``all`` everything (load, new, undo,
    redo); ``selection`` / ``meta`` need no new picture."""

    scope: str
    spec_id: str | None = None
    what: str = ""

    @property
    def renders(self) -> bool:
        return self.scope not in ("selection", "meta")


@dataclass
class ViewState:
    """The orbit camera of the tab. ``center``/``radius`` are the ``fit`` of the portrait
    renderer: orbit changes the angles, pan moves the center, zoom scales the radius."""

    azimuth: float = 25.0
    elevation: float = 6.0
    center: tuple = (0.0, 0.0, 9.0)
    radius: float = 12.0
    fov: float = 26.0
    min_radius: float = 0.5
    max_radius: float = 400.0

    def copy(self) -> "ViewState":
        return copy.copy(self)

    @property
    def fit(self):
        return (np.asarray(self.center, dtype=np.float64), float(self.radius))

    def camera(self, size) -> R.Camera:
        return R.orbit_camera(np.zeros(3), np.ones(3), size, self.azimuth, self.elevation,
                              self.fov, self.fit)

    def orbit(self, dx: float, dy: float) -> None:
        """Drag by (dx, dy) pixels: viewport.py's rule (0.5 deg / px azimuth, 0.4 elevation)."""
        self.azimuth = (self.azimuth + dx * ORBIT_AZIMUTH) % 360.0
        self.elevation = float(np.clip(self.elevation + dy * ORBIT_ELEVATION, -89.0, 89.0))

    def pan(self, dx: float, dy: float, size) -> None:
        """Middle / right drag by (dx, dy) pixels: the scene follows the pointer."""
        rot = self.camera(size).view[:3, :3]
        scale = self.radius * 1.05 * 2.0 / max(float(size[1]), 1.0)     # world units per pixel
        c = np.asarray(self.center, dtype=np.float64) - rot[0] * dx * scale + rot[1] * dy * scale
        self.center = tuple(float(x) for x in c)

    def zoom(self, factor: float) -> None:
        self.radius = float(np.clip(self.radius * factor, self.min_radius, self.max_radius))

    def frame(self, lo, hi) -> None:
        lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
        self.center = tuple(float(x) for x in (lo + hi) / 2.0)
        self.radius = max(float(np.linalg.norm(hi - lo)) / 2.0, 1.0)

    def project(self, points, size):
        """Screen ``(x, y, depth)`` of world points, with the camera of that image size."""
        cam = self.camera(size)
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        homo = np.c_[pts, np.ones(len(pts))]
        return cam.project((homo @ cam.view.T)[:, :3])


@dataclass
class Frame:
    """One picture for the display: ``image`` is (h, w, 4) uint8 at RENDER size; the display
    shows it upscaled ``scale`` times (the draft of a large pane is drawn at half size)."""

    gen: int
    tier: str                      # "draft" | "final"
    image: np.ndarray
    size: tuple                    # display size (w, h)
    scale: int = 1
    seconds: float = 0.0
    notes: tuple = ()
    _ppm: bytes | None = field(default=None, repr=False, compare=False)

    def ppm(self) -> bytes:
        """The picture as binary PPM at display size, ready for ``PhotoImage(data=...)``."""
        if self._ppm is None:
            w, h = self.size
            img = self.image[:, :, :3]
            if self.scale > 1:
                img = np.repeat(np.repeat(img, self.scale, axis=0), self.scale, axis=1)
            img = np.ascontiguousarray(img[:h, :w])
            self._ppm = b"P6 %d %d 255\n" % (img.shape[1], img.shape[0]) + img.tobytes()
        return self._ppm


@dataclass
class Pick:
    """What is under a pixel: the body (``quad`` of the cage, the surface ``point`` in the
    MakeHuman frame and the barycentric ``weights`` over the quad's corners) or an
    appendage (``spec_id``, ``side``)."""

    kind: str                      # "body" | "appendage"
    quad: int | None = None
    point: np.ndarray | None = None
    weights: tuple | None = None
    spec_id: str | None = None
    side: str = ""


@dataclass
class Scene:
    shells: list                   # portrait.Shell, Z-up
    owners: list                   # (spec id or None for the body, side letter) per shell
    lo: np.ndarray
    hi: np.ndarray
    notes: list = field(default_factory=list)


@dataclass
class BodyState:
    key: str
    verts: np.ndarray              # the morphed control vertices (MakeHuman frame)
    notes: list                    # what the anatomy layer adapted


@dataclass
class ExportResult:
    path: Path | None
    written: bool
    blocked: bool
    report_text: str
    assembly_text: str
    printability: object = None    # print_report.PrintabilityReport | None
    assembly: object = None        # printprep.PrintReport | None
    seconds: float = 0.0

    @property
    def blocking(self) -> list:
        return list(self.printability.blocking) if self.printability is not None else []

    @property
    def warnings(self) -> list:
        return list(self.printability.warnings) if self.printability is not None else []

    def as_text(self) -> str:
        return "\n".join(t for t in (self.assembly_text, self.report_text) if t)


class ExportBlocked(LabError):
    """The printability report blocks the export; nothing was written (``force=True`` writes)."""

    def __init__(self, result: ExportResult):
        super().__init__("the printability report blocks this export: "
                         + "; ".join(f.message for f in result.blocking[:3]))
        self.result = result


# --------------------------------------------------------------------------
# the document


class LabDocument:
    """A :class:`Character` and the edits the tab allows on it, with undo and files.

    Every edit validates first and changes nothing on failure (:class:`LabError`), pushes one
    snapshot on the undo stack (consecutive edits of the same control within
    :data:`COALESCE_SECONDS`, or inside ``begin_gesture`` / ``end_gesture``, are ONE step),
    marks the document dirty and tells the listeners what changed (:class:`Change`)."""

    def __init__(self, resources: Resources | None = None, character: Character | None = None,
                 clock=time.monotonic, engine: "Engine | None" = None):
        self._resources = resources
        self.character = character if character is not None else Character()
        self.clock = clock
        self.path: Path | None = None
        self.selected: str | None = None
        self.listeners: list = []
        self.undo_stack = UndoStack()
        self._engine = engine
        self._saved = self.snapshot()
        self._last_key = None
        self._last_time = -1e9
        self._gesture_before: str | None = None

    # -- plumbing ----------------------------------------------------------------------------

    @property
    def resources(self) -> Resources:
        if self._resources is None:
            self._resources = Resources.shared()
        return self._resources

    @property
    def engine(self) -> "Engine":
        if self._engine is None:
            self._engine = Engine(self.resources)
        return self._engine

    def snapshot(self) -> str:
        return json.dumps(self.character.to_dict(), sort_keys=True)

    @property
    def dirty(self) -> bool:
        return self.snapshot() != self._saved

    @property
    def name(self) -> str:
        return self.character.name

    @property
    def specs(self) -> list:
        return self.character.appendages

    def spec(self, spec_id: str) -> AppendageSpec:
        found = self.character.spec(spec_id)
        if found is None:
            have = ", ".join(a.id for a in self.specs) or "none"
            raise LabError(f"no appendage {spec_id!r} (the character has: {have})")
        return found

    def _emit(self, change: Change) -> None:
        for fn in list(self.listeners):
            fn(change)

    def _edit(self, change: Change, mutate, coalesce=None, after=None) -> bool:
        """Run ``mutate()`` as one undoable edit. ``False`` (and no undo entry, no event) if it
        changed nothing; the document is restored if it raises. ``after()`` runs just before the
        listeners hear about it (selection changes that belong to the edit)."""
        before = self.snapshot()
        try:
            mutate()
        except Exception:
            self.character = Character.from_dict(json.loads(before))
            raise
        if self.snapshot() == before:
            return False
        now = self.clock()
        if self._gesture_before is not None:
            pass                                   # the gesture pushes ONE entry when it ends
        elif (coalesce is not None and coalesce == self._last_key
              and now - self._last_time <= COALESCE_SECONDS):
            pass                                   # same control, same drag: one undo step
        else:
            self.undo_stack.push(before)
        self._last_key = coalesce
        self._last_time = now
        if after is not None:
            after()
        self._emit(change)
        return True

    def begin_gesture(self) -> None:
        """Start a drag: every edit until :meth:`end_gesture` is ONE undo step."""
        if self._gesture_before is None:
            self._gesture_before = self.snapshot()
            self._last_key = None

    def end_gesture(self) -> None:
        before, self._gesture_before = self._gesture_before, None
        if before is not None and self.snapshot() != before:
            self.undo_stack.push(before)
        self._last_key = None

    def break_coalescing(self) -> None:
        self._last_key = None

    @contextlib.contextmanager
    def gesture(self):
        """``with doc.gesture():`` -- every edit inside is ONE undo step. Joins a gesture that is
        already open (a slider drag) instead of ending it."""
        outer = self._gesture_before is not None
        self.begin_gesture()
        try:
            yield
        finally:
            if not outer:
                self.end_gesture()

    # -- undo / redo -------------------------------------------------------------------------

    @property
    def can_undo(self) -> bool:
        return self.undo_stack.can_undo

    @property
    def can_redo(self) -> bool:
        return self.undo_stack.can_redo

    def _restore(self, snapshot: str, what: str) -> None:
        self.character = Character.from_dict(json.loads(snapshot))
        if self.selected is not None and self.character.spec(self.selected) is None:
            self.selected = None
        self._last_key = None
        self._emit(Change("all", None, what))

    def undo(self) -> bool:
        self.end_gesture()
        prev = self.undo_stack.undo(self.snapshot())
        if prev is None:
            return False
        self._restore(prev, "undo")
        return True

    def redo(self) -> bool:
        self.end_gesture()
        nxt = self.undo_stack.redo(self.snapshot())
        if nxt is None:
            return False
        self._restore(nxt, "redo")
        return True

    # -- files -------------------------------------------------------------------------------

    def new(self) -> None:
        self.character = Character()
        self.path = None
        self.selected = None
        self.undo_stack = UndoStack()
        self._saved = self.snapshot()
        self._last_key = None
        self._gesture_before = None
        self._emit(Change("all", None, "new"))

    def load(self, path) -> None:
        try:
            raw = json.loads(Path(path).read_text())
            if not isinstance(raw, dict):
                raise ValueError("the file does not hold a character object")
            character = Character.from_dict(raw)
        except FileNotFoundError:
            raise LabError(f"no such file: {path}") from None
        except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
            raise LabError(f"cannot read {path} as a character: {exc}") from None
        self.character = character
        self.path = Path(path)
        self.selected = None
        self.undo_stack = UndoStack()
        self._saved = self.snapshot()
        self._last_key = None
        self._gesture_before = None
        self._emit(Change("all", None, "load"))

    def save(self, path=None) -> Path:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise LabError("this character has no file yet: give a path to save it")
        if target.suffix == "":
            target = target.with_suffix(".mmchar")
        self.character.save(target)
        self.path = target
        self._saved = self.snapshot()
        self._emit(Change("meta", None, "save"))
        return target

    # -- selection and naming ----------------------------------------------------------------

    def select(self, spec_id: str | None) -> None:
        if spec_id is not None:
            self.spec(spec_id)
        if spec_id != self.selected:
            self.selected = spec_id
            self._emit(Change("selection", spec_id, "select"))

    def rename_character(self, name: str) -> bool:
        name = (name or "").strip() or "unnamed"
        return self._edit(Change("meta", None, "name"),
                          lambda: setattr(self.character, "name", name), ("name",))

    # -- the body ----------------------------------------------------------------------------

    def set_macro(self, key: str, value, clamp: bool = False) -> float:
        """A body macro slider (gender, age, muscle, weight, height, african, asian, caucasian)
        in 0..1."""
        if key not in MACRO_KEYS:
            raise LabError(f"unknown macro slider {key!r}; the sliders are {', '.join(MACRO_KEYS)}")
        value = _number(key, value, 0.0, 1.0, clamp)
        self._edit(Change("body", None, f"macro:{key}"),
                   lambda: self.character.macro.__setitem__(key, value), ("macro", key))
        return value

    def set_slider(self, key: str, value, clamp: bool = False) -> float:
        """A catalog slider (-1..1 or 0..1 by kind; 0 removes it from the document)."""
        catalog = self.resources.catalog
        if key not in catalog:
            near = [s.key for s in catalog.find(key.split(".")[-1][:10])][:3]
            raise LabError(f"unknown slider {key!r}" + (f"; did you mean {near}?" if near else ""))
        sl = catalog[key]
        hi = float(len(sl.options) - 1) if sl.kind == "choice" else sl.hi
        value = _number(key, value, sl.lo, hi, clamp)

        def apply():
            if abs(value) < 1e-9:
                self.character.sliders.pop(key, None)
            else:
                self.character.sliders[key] = value

        self._edit(Change("body", None, f"slider:{key}"), apply, ("slider", key))
        return value

    def reset_body(self) -> bool:
        """Back to the default body: macros, shape sliders and asymmetry (appendages stay)."""
        default = Character()

        def apply():
            self.character.macro = dict(default.macro)
            self.character.sliders = {}
            self.character.asymmetry = {}

        return self._edit(Change("body", None, "reset"), apply)

    def set_anatomy(self, enabled: bool | None = None, cap: float | None = None,
                    clamp: bool = False) -> None:
        """The body-adaptation switch and its cap (the layer itself is derived, never saved)."""
        if cap is not None:
            cap = _number("anatomy cap", cap, 0.0, 2.0, clamp)

        def apply():
            if enabled is not None:
                self.character.anatomy["enabled"] = bool(enabled)
            if cap is not None:
                self.character.anatomy["cap"] = cap

        self._edit(Change("body", None, "anatomy"), apply, ("anatomy",))

    def anatomy_notes(self) -> list:
        """What the enabled appendages ask of the body (empty if the switch is off)."""
        return list(self.engine.body_state(self.character).notes)

    # -- appendages: structure ---------------------------------------------------------------

    def add_appendage(self, kind: str, *, params: dict | None = None, preset: str | None = None,
                      side: str | None = None, face: int | None = None, scale=None, roll=None,
                      offset=None, name: str = "", select: bool = True) -> AppendageSpec:
        """A new appendage of ``kind`` (a preset places it; ``params`` override defaults). The
        record is the same one ``minimaster character --add`` writes."""
        from . import kinds
        try:
            spec = kinds.new_spec(kind, self.specs, side=side, params=params, preset=preset,
                                  scale=scale, roll=roll, offset=offset, face=face, name=name,
                                  base=self.resources.base, lib=self.resources.lib)
        except ValueError as exc:
            raise LabError(str(exc)) from None

        def apply():
            self.character.appendages.append(spec)

        def choose():
            if select:
                self.selected = spec.id

        self._edit(Change("shape", spec.id, "add"), apply, after=choose)
        return spec

    def add_entry(self, entry: AddEntry, **kw) -> AppendageSpec:
        """Add what an :class:`AddEntry` of the Add menu describes."""
        return self.add_appendage(entry.kind, params=entry.param_dict, preset=entry.preset, **kw)

    def remove_appendage(self, spec_id: str) -> None:
        self.spec(spec_id)
        index = next(i for i, a in enumerate(self.specs) if a.id == spec_id)

        def apply():
            self.character.remove_appendage(spec_id)

        def choose():
            if self.selected == spec_id:
                rest = self.specs
                self.selected = rest[min(index, len(rest) - 1)].id if rest else None

        self._edit(Change("shape", spec_id, "remove"), apply, after=choose)

    def duplicate_appendage(self, spec_id: str, select: bool = True) -> AppendageSpec:
        """A copy with its own id (``wing-2``), name and nothing else changed: it sits exactly
        where the original does until it is moved."""
        src = self.spec(spec_id)
        dup = src.copy()
        dup.id = next_id(src.kind, self.specs)
        dup.name = f"{src.label()} copy"

        def apply():
            self.character.appendages.append(dup)

        def choose():
            if select:
                self.selected = dup.id

        self._edit(Change("shape", dup.id, "duplicate"), apply, after=choose)
        return dup

    def rename_appendage(self, spec_id: str, name: str) -> bool:
        spec = self.spec(spec_id)
        name = (name or "").strip()
        return self._edit(Change("meta", spec_id, "rename"),
                          lambda: setattr(spec, "name", name), ("rename", spec_id))

    def set_enabled(self, spec_id: str, enabled: bool) -> bool:
        spec = self.spec(spec_id)
        return self._edit(Change("shape", spec_id, "enabled"),
                          lambda: setattr(spec, "enabled", bool(enabled)))

    # -- appendages: parameters --------------------------------------------------------------

    def _kind(self, spec: AppendageSpec):
        from . import kinds
        kind = kinds.kind_of(spec)
        if kind is None:
            raise LabError(f"{spec.id!r} is a {spec.kind!r}, a kind this program does not know: "
                           "it is kept in the file but cannot be edited here")
        return kind

    def param_specs(self, spec_id: str) -> list:
        """``[(ParamSpec, current value)]`` of the parameters that apply to this appendage right
        now (``when=`` conditions evaluated): what a property grid shows."""
        spec = self.spec(spec_id)
        kind = self._kind(spec)
        values, _ = kind.clean(spec.params)
        return [(p, values[p.key]) for p in kind.params
                if p.applies(values) and kind.param(p.key, values) is p]

    def set_param(self, spec_id: str, key: str, value, clamp: bool = False):
        """Set a build parameter. Refused: a key the kind does not have, one whose ``when=``
        does not hold (the wing's ``primaries`` while the style is a bat), a value of the
        wrong type or outside its range (``clamp=True`` clamps instead). Returns the stored
        value. Parameters that stop applying when ``style`` changes are dropped from the record."""
        spec = self.spec(spec_id)
        kind = self._kind(spec)
        known = {p.key for p in kind.params}
        if key not in known:
            raise LabError(f"{spec.kind} has no parameter {key!r}; it has {', '.join(sorted(known))}")
        values, _ = kind.clean(spec.params)
        p = kind.param(key, values)
        if p is None:
            when = next(q.when for q in kind.params if q.key == key)
            raise LabError(f"{key} only applies when {when[0]} is "
                           f"{' or '.join(str(w) for w in when[1])} (it is {values.get(when[0])!r})")
        try:
            coerced = p.coerce(value)
        except ValueError as exc:
            raise LabError(str(exc)) from None
        if p.type in ("float", "int") and not clamp:
            x = float(value)
            if (p.lo is not None and x < p.lo - 1e-9) or (p.hi is not None and x > p.hi + 1e-9):
                raise LabError(f"{key} = {value!r} is outside the range {p.lo:g}..{p.hi:g}")

        def apply():
            spec.params[key] = coerced
            # parameters of the old style (or any other `when` branch) that no longer apply
            now, _ = kind.clean(spec.params)
            applicable = {q.key for q in kind.params if q.applies(now)}
            for stale in [k for k in spec.params if k in known and k not in applicable]:
                del spec.params[stale]

        self._edit(Change("shape", spec_id, f"param:{key}"), apply, ("param", spec_id, key))
        return coerced

    # -- appendages: pose controls -----------------------------------------------------------

    def control_range(self, spec_id: str, name: str, side: str | None = None):
        """``(lo, hi)`` the PLACED appendage allows the control ``name`` right now (a wing placed
        on a body allows less than a free one, and less as it folds), or ``None``."""
        spec = self.spec(spec_id)
        kind = self._kind(spec)
        if name not in {c.key for c in kind.controls}:
            return None
        placed = self.engine.placed(self.character, spec_id)
        if side:
            placed = [p for p in placed if p.side == side]
        from . import kinds
        lo = hi = None
        for p in placed:
            r = p.control_range(name, kinds.side_controls(spec.controls, p.side))
            if r is not None:
                lo = r[0] if lo is None else max(lo, r[0])
                hi = r[1] if hi is None else min(hi, r[1])
        if lo is None:                   # not built (disabled / not buildable): the free range
            return kind.control_range(name, kinds.side_controls(spec.controls, ""))
        return (lo, hi)

    def set_control(self, spec_id: str, name: str, value, side: str | None = None,
                    clamp: bool = False):
        """Pose an appendage (``fold``, ``raise_`` ... or ``preset`` with a pose name). The
        value must lie inside the PLACED appendage's range at its current fold (``clamp=True``
        clamps instead). ``side`` ``"l"`` / ``"r"`` poses one side only. Returns the stored value."""
        spec = self.spec(spec_id)
        kind = self._kind(spec)
        names = {c.key for c in kind.controls}
        if not names:
            raise LabError(f"{spec.kind} has no pose controls")
        if side not in (None, "", "l", "r"):
            raise LabError(f"side must be 'l' or 'r', not {side!r}")
        if name == "preset":
            stored = None if value in (None, "", "none") else value
            if stored is not None:
                cleaned, notes = kind.clean_controls({"preset": stored})
                if "preset" not in cleaned:
                    raise LabError(notes[0] if notes else f"unknown pose {value!r}")
        else:
            if name not in names:
                raise LabError(f"{spec.kind} has no control {name!r}; it has {', '.join(sorted(names))}")
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value):
                raise LabError(f"{name} = {value!r} is not a finite number")
            lo, hi = self.control_range(spec_id, name, side)
            stored = float(value)
            if not lo - 1e-9 <= stored <= hi + 1e-9:
                if not clamp:
                    raise LabError(f"{name} = {value!r} is outside the range {lo:g}..{hi:g} this "
                                   f"{spec.kind} allows at its current fold")
                stored = min(max(stored, lo), hi)

        def apply():
            key = side or "*"
            target = spec.controls.get(key)
            if not isinstance(target, dict):
                target = spec.controls[key] = {}
            if stored is None:
                target.pop(name, None)
                if not target:
                    spec.controls.pop(key, None)
            else:
                target[name] = stored

        self._edit(Change("pose", spec_id, f"control:{name}"), apply,
                   ("control", spec_id, name, side))
        return stored

    # -- appendages: placement ---------------------------------------------------------------

    def set_placement(self, spec_id: str, key: str, value, clamp: bool = False) -> float:
        """A numeric placement field: ``scale``, ``roll`` (degrees) or ``offset`` (the sink
        into the body along the normal; negative buries it). Only the fields the kind honours."""
        spec = self.spec(spec_id)
        kind = self._kind(spec)
        if key not in PLACEMENT_RANGES:
            raise LabError(f"placement field {key!r} is not one of {', '.join(PLACEMENT_RANGES)}")
        if key not in kind.placement_keys:
            raise LabError(f"{spec.kind} does not use {key!r} (it uses "
                           f"{', '.join(kind.placement_keys)})")
        lo, hi = PLACEMENT_RANGES[key]
        value = _number(key, value, lo, hi, clamp)
        self._edit(Change("shape", spec_id, f"placement:{key}"),
                   lambda: spec.placement.__setitem__(key, value), ("placement", spec_id, key))
        return value

    def set_side(self, spec_id: str, side: str) -> bool:
        """``center`` | ``left`` | ``right`` | ``both``. The left-hand definition is stored once;
        the right is its exact mirror."""
        spec = self.spec(spec_id)
        if side not in SIDES:
            raise LabError(f"unknown side {side!r}; use {', '.join(SIDES)}")
        return self._edit(Change("shape", spec_id, "side"), lambda: setattr(spec, "side", side))

    def set_placement_preset(self, spec_id: str, preset: str) -> bool:
        """Re-place on one of the kind's presets: the anchor goes to the preset's quad and the
        roll / offset / up-hint to its values; the scale and the side are kept."""
        from . import kinds
        spec = self.spec(spec_id)
        kind = self._kind(spec)
        if preset not in kind.presets:
            raise LabError(f"{spec.kind} has no placement preset {preset!r}; it has "
                           f"{', '.join(kind.presets)}")
        p = kind.presets[preset]
        corners = [int(c) for c in self.resources.base.body_cage(None)[1][kinds.PRESET_FACES[p["face"]]]]

        def apply():
            pl = spec.placement
            pl["preset"] = preset
            pl["anchor"] = {"face": kinds.PRESET_FACES[p["face"]], "corners": corners}
            for k in ("roll", "offset"):
                if k in kind.placement_keys and k in p:
                    pl[k] = p[k]
            if "up_hint" in kind.placement_keys:
                pl["up_hint"] = list(p.get("up_hint", (0.0, 1.0, 0.0)))

        return self._edit(Change("shape", spec_id, "preset"), apply)

    def set_anchor(self, spec_id: str, face: int, weights=None, *, mirror: bool = False) -> int:
        """Bind the appendage to cage quad ``face``. The stored quad is always the LEFT-hand one
        (design D4): with ``mirror=True`` the quad given is the right-hand one, so its mirror
        image (and mirrored weights) is stored. Returns the stored quad."""
        from . import appendages as A
        spec = self.spec(spec_id)
        self._kind(spec)
        base = self.resources.base
        if isinstance(face, bool) or not isinstance(face, (int, np.integer)) \
                or not 0 <= int(face) < self.resources.n_quads:
            raise LabError(f"face = {face!r} is not a cage quad (0..{self.resources.n_quads - 1})")
        face = int(face)
        quads = base.body_cage(None)[1]
        if weights is not None:
            weights = tuple(float(w) for w in weights)
            if len(weights) != len(quads[face]) or not all(math.isfinite(w) and w >= -1e-9
                                                           for w in weights):
                raise LabError("weights must be one non-negative number per quad corner")
            total = sum(weights)
            if total <= 1e-9:
                raise LabError("weights must not all be zero")
            weights = tuple(w / total for w in weights)
        if mirror:
            if weights is not None:
                weights = A.mirror_anchor_weights(base, quads, face, weights)
            face = int(A.mirror_face_map(base)[face])
        anchor = {"face": face, "corners": [int(c) for c in quads[face]]}
        if weights is not None:
            anchor["weights"] = [round(w, 6) for w in weights]
        self._edit(Change("shape", spec_id, "anchor"),
                   lambda: spec.placement.__setitem__("anchor", anchor), ("anchor", spec_id))
        return face

    def place_at_pick(self, spec_id: str, pick: Pick) -> int:
        """Move the appendage's anchor to where a body :class:`Pick` landed. A ``both`` / ``center``
        appendage keeps its side; a ``left`` / ``right`` one goes to the side that was clicked
        (the quad is normalised to the left-hand definition either way)."""
        if pick is None or pick.kind != "body" or pick.quad is None:
            raise LabError("click the body to place the appendage")
        spec = self.spec(spec_id)
        right_hand = bool(pick.point is not None and pick.point[0] < 0.0)
        with self.gesture():                      # the anchor and the side: one click, one step
            face = self.set_anchor(spec_id, pick.quad, pick.weights, mirror=right_hand)
            if spec.side in ("left", "right"):
                self.set_side(spec_id, "right" if right_hand else "left")
        return face


def _number(key, value, lo, hi, clamp):
    if isinstance(value, bool) or value is None:
        raise LabError(f"{key} = {value!r} is not a number")
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise LabError(f"{key} = {value!r} is not a number") from None
    if not math.isfinite(x):
        raise LabError(f"{key} = {value!r} is not finite")
    if not lo <= x <= hi:
        if not clamp:
            raise LabError(f"{key} = {value!r} is outside the range {lo:g}..{hi:g}")
        x = min(max(x, lo), hi)
    return x


# --------------------------------------------------------------------------
# building appendages for the preview


class _QuietContext:
    """A ``BuildContext`` whose ``warn`` records instead of calling ``warnings.warn`` (which is
    process-global state and not for a worker thread to touch)."""

    @staticmethod
    def make(base, verts, lib, report):
        from . import kinds

        class Ctx(kinds.BuildContext):
            def warn(self, message):
                self.report.append(message)

        return Ctx(base, verts, lib, report)


def _rest_bytes(placed) -> int:
    """Bytes of a placed appendage's own geometry (a static mesh, or a rig's rest pieces and
    skin weights, estimated as three times the pieces)."""
    if getattr(placed, "static", None) is not None:
        return placed.static.vertices.nbytes + placed.static.faces.nbytes
    try:
        pieces = placed.rig.rig.pieces
        return 3 * sum(q.mesh.vertices.nbytes + q.mesh.faces.nbytes for q in pieces)
    except AttributeError:
        return 0


class Built:
    """One appendage spec built on one body: a :class:`~kinds.Placed` per side, with its own
    caches of posed meshes and cluster proxies. Immutable inputs, mutable caches."""

    def __init__(self, spec_id: str, placed: list, notes: list, z_up: np.ndarray,
                 mirrored: bool = False):
        self.spec_id = spec_id
        self.placed = placed
        self.notes = notes
        self.mirrored = mirrored       # only the left side was built; the right is its mirror image
        self._z = z_up
        self._poses: OrderedDict = OrderedDict()
        self._proxies: dict = {}
        self._lock = threading.Lock()
        # the rest geometry: a rig keeps its pieces and skin weights (about three meshes' worth)
        self._rest = sum(_rest_bytes(p) for p in placed)

    def nbytes(self) -> int:
        """Bytes this build holds (rest geometry estimate + cached poses + proxies)."""
        with self._lock:
            n = self._rest + sum(m.vertices.nbytes + m.faces.nbytes for m in self._poses.values())
            n += sum(p.rest.nbytes + p.faces.nbytes for p in self._proxies.values())
        return n

    def posed(self, i: int, controls: dict, stats: Counter | None = None):
        """The Z-up mesh of side ``i`` under ``controls`` (cached by controls)."""
        key = (i, controls_key(controls))
        with self._lock:
            hit = self._poses.get(key)
            if hit is not None:
                self._poses.move_to_end(key)
                return hit
        if stats is not None:
            stats["pose"] += 1
        mesh = self.placed[i].mesh(controls).transform(self._z)
        with self._lock:
            self._poses[key] = mesh
            while len(self._poses) > 6:
                self._poses.popitem(last=False)
        return mesh

    def proxy(self, i: int):
        """A :class:`~draft.ClusterProxy` of side ``i`` (wings only; static kinds are small)."""
        placed = self.placed[i]
        if placed.rig is None:
            return None
        with self._lock:
            proxy = self._proxies.get(i)
        if proxy is None:
            from . import draft as D
            rest = placed.mesh({}).transform(self._z)
            pieces = D.pieces_from_ranges(placed.rig.rig.ranges.values(), len(rest.vertices))
            proxy = D.ClusterProxy.by_pieces(rest.vertices, rest.faces, pieces, detail=5.0,
                                             min_cells=2, max_cells=10)
            with self._lock:
                self._proxies[i] = proxy
        return proxy


# --------------------------------------------------------------------------
# the engine: document -> pictures


class Engine:
    """Document -> morphed body -> built appendages -> posed meshes -> shells -> pictures.

    Each stage is cached by CONTENT, so an edit redoes only the stages whose inputs changed: a
    pose drag re-poses one appendage (the body, the other appendages and every proxy are
    reused), a parameter drag rebuilds one appendage, orbiting rebuilds nothing. ``stats``
    counts the work actually done (``morph``, ``build``, ``pose``, ``final_scene`` ...).
    Safe to call from the UI thread and the final-render worker at once."""

    LAST_DRAFT_KEEP = 8                     # how many drafts the re-place shortcut remembers

    def __init__(self, resources: Resources):
        from . import shading as sh
        self.resources = resources
        self.stats: Counter = Counter()
        self.shading = sh.ShadingCache()
        self.report_cache: dict = {}
        self._bodies = Cache(8)                 # body key -> BodyState
        self._states = Cache(4)                 # inputs digest -> BodyState (skips the anatomy)
        # (spec key, body key, mode) -> Built; an angel's full pair is ~25 MB, so bound the bytes
        self._built = Cache(48, BUILT_CACHE_BYTES, lambda b: b.nbytes())
        self._shells = Cache(4)                 # body key -> full-detail body shells
        self._last_draft: dict = {}             # spec key -> (BodyState, Built): for the re-place
        self._lock = threading.Lock()

    # -- the body ----------------------------------------------------------------------------

    def body_state(self, character: Character) -> BodyState:
        """The morphed body of the character as built: the user's sliders plus what the enabled
        appendages ask of it (the derived anatomy layer), cached by the effective weights."""
        res = self.resources
        inputs = _digest({
            "macro": character.macro, "sliders": character.sliders,
            "asymmetry": character.asymmetry, "anatomy": character.anatomy,
            "appendages": [dict(spec.to_dict(), controls=None, name=None)
                           for spec in character.appendages]})
        state = self._states.get(inputs)
        if state is not None:
            return state
        from .mhbase import macro_weights
        sliders, macro, notes = character.effective(res.catalog)
        weights = dict(macro_weights(**macro))
        for name, w in res.catalog.weights(sliders, character.asymmetry).items():
            weights[name] = weights.get(name, 0.0) + w
        key = _digest(sorted((n, round(float(w), 9)) for n, w in weights.items() if w))

        def morph():
            self.stats["morph"] += 1
            return BodyState(key, res.lib.apply(res.base.verts, weights), list(notes))

        body = self._bodies.get_or_build(key, morph)
        state = BodyState(body.key, body.verts, list(notes))
        return self._states.put(inputs, state)

    # -- appendages --------------------------------------------------------------------------

    @staticmethod
    def buildable(character: Character, notes: list | None = None):
        """The enabled specs that can be built, in order (a duplicate or invalid id is skipped,
        as ``kinds.instantiate_all`` skips it, and named in ``notes``)."""
        from .appendage_spec import valid_id
        seen: set = set()
        out = []
        for spec in character.appendages:
            if not spec.enabled:
                continue
            if spec.id in seen or not valid_id(spec.id):
                if notes is not None:
                    notes.append(f"appendage {spec.id!r}: the id is duplicated or not made of "
                                 "letters, digits, '-' and '_'; skipped")
                continue
            seen.add(spec.id)
            out.append(spec)
        return out

    def built(self, spec: AppendageSpec, body: BodyState, draft: bool, half: bool = False) -> Built:
        """The spec built on the body: ``draft`` with the kind's ``draft_params``, ``half`` the
        left side only (the draft of a parameter drag mirrors it; see :meth:`half_ok`)."""
        mode = "half" if half else ("draft" if draft else "full")
        key = (spec_key(spec), body.key, mode)

        def build():
            self.stats[{"half": "build_half", "draft": "build_draft", "full": "build"}[mode]] += 1
            return self._build(spec, body, draft or half, half)

        built = self._built.get_or_build(key, build)
        if mode == "draft":
            self._last_draft[spec_key(spec)] = (body, built)
            # it only serves the re-place of the spec being dragged: keep the last few, or every
            # distinct shape edit pins a Built the byte-bounded cache has long dropped (~4 MB
            # per edit for a feathered wing, 1.8 GB after 400 span edits)
            while len(self._last_draft) > self.LAST_DRAFT_KEEP:
                self._last_draft.pop(next(iter(self._last_draft)))
        return built

    @staticmethod
    def half_ok(spec: AppendageSpec) -> bool:
        """May the draft of this spec be built as its left side plus a mirror image? A paired kind
        on both sides whose two sides are posed alike (the right side is the exact mirror of the
        left on a symmetric body; the final render never takes this shortcut)."""
        from . import kinds
        kind = kinds.kind_of(spec)
        return (kind is not None and kind.paired and spec.side == "both"
                and not any(k in spec.controls for k in ("l", "r")))

    def _build(self, spec: AppendageSpec, body: BodyState, draft: bool,
               half: bool = False) -> Built:
        from . import kinds
        from .portrait import MH_TO_Z_UP
        if half:
            spec = spec.copy()
            spec.side = "left"
        report: list = []
        kind = kinds.KINDS.get(spec.kind)
        placed: list = []
        if kind is None:
            report.append(f"appendage {spec.id!r}: unknown kind {spec.kind!r} is kept in the "
                          "document but not built")
        else:
            ctx = _QuietContext.make(self.resources.base, body.verts, self.resources.lib, report)
            try:
                placed = kind.instantiate(spec, ctx, draft)
            except kinds.SpecError as exc:
                report.append(f"appendage {spec.id!r}: {exc}; skipped")
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                report.append(f"appendage {spec.id!r}: cannot be built ({exc}); skipped")
        return Built(spec.id, placed, report, MH_TO_Z_UP, mirrored=half)

    def placed(self, character: Character, spec_id: str, draft: bool = True) -> list:
        """The :class:`~kinds.Placed` objects of one spec on the current body (cached)."""
        spec = character.spec(spec_id)
        if spec is None or not spec.enabled:
            return []
        return self.built(spec, self.body_state(character), draft).placed

    @staticmethod
    def side_controls(spec: AppendageSpec, placed) -> dict:
        """The cleaned controls of one side of an appendage (what ``Placed.mesh`` takes)."""
        from . import kinds
        kind = kinds.kind_of(spec)
        if kind is None or not kind.controls:
            return {}
        cleaned, _ = kind.clean_controls(kinds.side_controls(spec.controls, placed.side))
        return cleaned

    # -- scenes ------------------------------------------------------------------------------

    def draft_scene(self, character: Character, mode: str = "exact",
                    edited: str | None = None) -> Scene:
        """Body proxy (smooth) + cluster proxies of the posed appendages (flat).

        ``mode`` says how much shortcut a drag may take (the final render never does):
        ``"exact"`` builds what is missing; ``"body"`` (a body-slider drag) does not rebuild an
        appendage whose body changed but moves its last draft by how far its anchor moved;
        ``"shape"`` (a parameter drag of the spec ``edited``) builds only the left side of a
        paired appendage and mirrors it (:meth:`half_ok`); ``"pose"`` (a pose drag of ``edited``)
        poses only its left side and mirrors that."""
        from ..core.mesh import Mesh
        from . import shading as sh
        from .portrait import Shell
        body = self.body_state(character)
        proxy = self.resources.body_proxy
        shells = [Shell(proxy.mesh(body.verts), sh.SKIN)]
        owners = [(None, "")]
        notes: list = []
        flip = np.array([-1.0, 1.0, 1.0])
        for spec in self.buildable(character, notes):
            shift = None
            built = self._built.peek((spec_key(spec), body.key, "draft"))
            if built is None and mode == "body":
                prev = self._last_draft.get(spec_key(spec))
                if prev is not None:
                    shift = self._anchor_shift(spec, prev[0], body, prev[1])
                    if shift is not None:
                        built = prev[1]
                        self.stats["replace"] += 1
            if built is None and mode == "shape" and spec.id == edited and self.half_ok(spec):
                built = self.built(spec, body, True, half=True)
            if built is None:
                built = self.built(spec, body, True)
            notes += built.notes
            # the right side as the left's mirror image (x -> -x): built that way, or (in a pose
            # drag) posed that way -- both sides of a paired wing move alike
            mirror = built.mirrored or (mode in ("pose", "shape") and spec.id == edited
                                        and self.half_ok(spec) and len(built.placed) == 2
                                        and built.placed[0].side == "l")
            for i, placed in enumerate(built.placed[:1] if mirror else built.placed):
                mesh = built.posed(i, self.side_controls(spec, placed), self.stats)
                verts = mesh.vertices if shift is None else mesh.vertices + shift[i]
                proxy_i = built.proxy(i)
                if proxy_i is not None:
                    mesh = proxy_i.mesh(verts)
                elif shift is not None:
                    mesh = Mesh(verts, mesh.faces)
                shells.append(Shell(mesh, placed.material))
                owners.append((spec.id, placed.side))
                if mirror:
                    shells.append(Shell(Mesh(mesh.vertices * flip, mesh.faces[:, ::-1]),
                                        placed.material))
                    owners.append((spec.id, "r"))
        return _scene(shells, owners, notes)

    @staticmethod
    def side_letters(spec: AppendageSpec) -> list:
        """The sides a spec builds, as ``Placed.side`` letters (``""`` midline, ``l``, ``r``)."""
        return {"center": [""], "left": ["l"], "right": ["r"], "both": ["l", "r"]}.get(spec.side, [])

    def anchor_points(self, spec: AppendageSpec, body: BodyState) -> list:
        """``[(side letter, point)]``: where each side of the spec joins the body (the MakeHuman
        frame), read off the cage quad it is bound to -- no appendage is built, so the overlay
        markers cost nothing during a drag. ``[]`` if the anchor cannot be resolved."""
        from . import appendages as A
        from . import kinds
        kind = kinds.kind_of(spec)
        if kind is None:
            return []
        try:
            values, _ = kind.clean(spec.params)
            ctx = kinds.BuildContext(self.resources.base, body.verts, self.resources.lib, [])
            pl = kind.resolve_placement(spec, values, ctx)
            cage_v, cage_q = ctx.cage
            base = self.resources.base
            out = []
            for letter in self.side_letters(spec):
                face, weights = pl["face"], pl["weights"]
                if letter == "r":
                    if weights is not None:
                        weights = A.mirror_anchor_weights(base, cage_q, face, weights)
                    face = int(A.mirror_face_map(base)[face])
                corners = cage_q[face]
                w = np.full(len(corners), 1.0 / len(corners)) if weights is None \
                    else np.asarray(weights, dtype=np.float64)
                out.append((letter, (cage_v[corners] * w[:, None]).sum(axis=0)))
            return out
        except (kinds.SpecError, KeyError, IndexError, ValueError):
            return []

    def _anchor_shift(self, spec: AppendageSpec, old: BodyState, new: BodyState, built: Built):
        """Per-side Z-up displacement of the spec's anchor between two bodies (or ``None`` if
        the anchor cannot be resolved)."""
        from .portrait import MH_TO_Z_UP
        now = dict(self.anchor_points(spec, new))
        if not now or any(p.side not in now for p in built.placed):
            return None
        return [MH_TO_Z_UP[:3, :3] @ (now[p.side] - np.asarray(p.anchor, dtype=np.float64))
                for p in built.placed]

    def floor_z(self, character: Character) -> float:
        """The z of the soles in the Z-up frame (the grid stands there)."""
        body = self.body_state(character)
        return float(self.resources.base.body_cage(body.verts)[0][:, 1].min())

    def body_shells(self, body: BodyState, eye_color: str, levels: int = 0):
        from .portrait import character_shells

        def build():
            self.stats["body_shells"] += 1
            return character_shells(self.resources.base, body.verts, eye_color=eye_color,
                                    levels=levels)

        return self._shells.get_or_build((body.key, eye_color, levels), build)

    def final_scene(self, character: Character, check=None, levels: int = 0) -> Scene:
        """Full-detail body (eyes included) + the full appendages, posed."""
        from .portrait import Shell
        body = self.body_state(character)
        eye = str(character.materials.get("eye_color", "#5b7c8d"))
        shells = list(self.body_shells(body, eye, levels))
        owners = [(None, "")] * len(shells)
        notes: list = []
        for spec in self.buildable(character, notes):
            if check:
                check()
            built = self.built(spec, body, False)
            notes += built.notes
            for i, placed in enumerate(built.placed):
                if check:
                    check()
                mesh = built.posed(i, self.side_controls(spec, placed), self.stats)
                shells.append(Shell(mesh, placed.material))
                owners.append((spec.id, placed.side))
        self.stats["final_scene"] += 1
        return _scene(shells, owners, notes)

    # -- pictures ----------------------------------------------------------------------------

    def render_draft(self, character: Character, view: ViewState, size, *, gen: int = 0,
                     scale: int = 1, mode: str = "exact", edited: str | None = None) -> Frame:
        from .draft import render_draft as draw_draft
        t0 = time.perf_counter()
        scene = self.draft_scene(character, mode, edited)
        w, h = size
        rs = (max(1, -(-w // scale)), max(1, -(-h // scale)))
        modes = ["smooth"] + ["flat"] * (len(scene.shells) - 1)
        img = draw_draft(scene.shells, size=rs, azimuth=view.azimuth,
                           elevation=view.elevation, fov=view.fov, fit=view.fit,
                           shading=modes, background=BACKGROUND)
        return Frame(gen, "draft", img, (w, h), scale, time.perf_counter() - t0,
                     tuple(scene.notes))

    def render_final(self, character: Character, view: ViewState, size, *, gen: int = 0,
                     check=None, levels: int = 0) -> Frame:
        from .portrait import render_portrait
        t0 = time.perf_counter()
        scene = self.final_scene(character, check, levels)
        img = render_portrait(scene.shells, size=tuple(size), azimuth=view.azimuth,
                              elevation=view.elevation, fov=view.fov, fit=view.fit,
                              background=BACKGROUND, check=check, cache=self.shading)
        return Frame(gen, "final", img, tuple(size), 1, time.perf_counter() - t0,
                     tuple(scene.notes))

    def frame_bounds(self, character: Character):
        """``(lo, hi)`` of the draft scene: what the camera frames."""
        s = self.draft_scene(character)
        return s.lo, s.hi

    # -- picking -----------------------------------------------------------------------------

    def pick(self, character: Character, view: ViewState, size, x: float, y: float,
             body_only: bool = False):
        """What is at pixel (x, y) of the display: the body quad (with the surface point and its
        barycentric weights) or the appendage. ``body_only`` ignores appendages (placing an
        anchor through a wing). The id buffer is built from the FULL body (so the quad id is
        exact: ``quad = triangle % n_quads``) and the appendages' proxies."""
        from . import portrait
        from .portrait import MH_TO_Z_UP
        res = self.resources
        body = self.body_state(character)
        body_shell = self._shells.get_or_build(
            (body.key, "pick", 0),
            lambda: portrait.character_shells(res.base, body.verts, with_eyes=False)[0])
        shells, owners = [body_shell], [(None, "")]
        if not body_only:
            scene = self.draft_scene(character)
            shells += scene.shells[1:]
            owners += scene.owners[1:]
        vb = portrait.visibility_buffer(shells, tuple(size), view.azimuth, view.elevation,
                                        view.fov, view.fit)
        hit = vb.pick(x, y)
        if hit is None:
            return None
        shell_i, tri = hit
        spec_id, side = owners[shell_i]
        if spec_id is not None:
            return Pick("appendage", spec_id=spec_id, side=side)
        n = res.n_quads
        quad = int(tri % n)
        z_point = vb.world_point(x, y)
        point = None if z_point is None else MH_TO_Z_UP[:3, :3].T @ z_point
        weights = None
        if point is not None:
            cage_v, cage_q = res.base.body_cage(body.verts)
            weights = _quad_weights(cage_v[cage_q[quad]], point, second=tri >= n)
        return Pick("body", quad=quad, point=point, weights=weights)

    # -- export ------------------------------------------------------------------------------

    def export(self, character: Character, path=None, *, size: str | None = "medium",
               height: float | None = None, with_base: bool = True, levels: int = 0,
               fit_base: bool = False, force: bool = False, check=None) -> ExportResult:
        """The STL of the character with its appendages, the same assembly the
        ``minimaster character -o x.stl`` command performs, plus the printability report.

        Blocking findings stop the export (:class:`ExportBlocked`, nothing written) unless
        ``force``. ``path=None`` only measures (the panel's "check printability")."""
        from . import print_report, printprep
        from .portrait import MH_TO_Z_UP  # noqa: F401
        t0 = time.perf_counter()
        res = self.resources
        body = self.body_state(character)
        verts = body.verts
        shells = printprep.character_print_shells(res.base, verts, levels=levels)
        figure = list(shells)
        app_shells = []
        for spec in self.buildable(character):
            built = self.built(spec, body, False)
            for placed in built.placed:
                if check:
                    check()
                ctl = dict(self.side_controls(spec, placed))
                if placed.rig is not None:
                    preset = ctl.pop("preset", None)
                    app_shells.append([(f"{placed.name}/{n}", m)
                                       for n, m in placed.rig.piece_meshes(preset, **ctl)])
                else:
                    app_shells.append([(placed.name, placed.mesh())])
        if app_shells:
            shells = shells + printprep.appendage_print_shells(app_shells)
        try:
            mesh, rep = printprep.assemble_character(
                shells, size=size, height=height, with_base=with_base,
                reference=printprep.FIGURE_SHELLS if app_shells else None,
                fit_base=bool(app_shells) and fit_base)
        except printprep.PrintError as exc:
            raise LabError(f"cannot print this character: {exc}") from None
        prep = None
        report_text = ""
        blocked = False
        if app_shells:
            mm_per_unit = printprep.figure_scale(shells, size=size, height=height,
                                                 reference=printprep.FIGURE_SHELLS)
            prep = print_report.appendage_report(
                app_shells, res.base, verts, mm_per_unit, levels=levels, figure_shells=figure,
                assembly=rep, cache=self.report_cache)
            report_text = prep.as_text()
            blocked = bool(prep.blocking)
        result = ExportResult(Path(path) if path else None, False, blocked, report_text,
                              rep.as_text(), prep, rep)
        if path is not None:
            if blocked and not force:
                result.seconds = time.perf_counter() - t0
                raise ExportBlocked(result)
            from ..core.stl import write_stl
            write_stl(mesh, path, name="character")
            result.written = True
        result.seconds = time.perf_counter() - t0
        return result


def _scene(shells, owners, notes) -> Scene:
    live = [s.mesh.vertices for s in shells if len(s.mesh.faces)]
    lo = np.min([v.min(axis=0) for v in live], axis=0)
    hi = np.max([v.max(axis=0) for v in live], axis=0)
    return Scene(shells, owners, lo, hi, notes)


def _quad_weights(corners: np.ndarray, point: np.ndarray, second: bool) -> tuple:
    """Barycentric weights over the four corners of a cage quad for a point on one of its two
    triangles (``second``: the triangle (0, 2, 3), else (0, 1, 2)); non-negative, summing to 1."""
    idx = (0, 2, 3) if second else (0, 1, 2)
    a, b, c = (corners[i] for i in idx)
    v0, v1, v2 = b - a, c - a, point - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    den = d00 * d11 - d01 * d01
    if abs(den) < 1e-18:
        return (0.25, 0.25, 0.25, 0.25)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    u = 1.0 - v - w
    tri = np.clip([u, v, w], 0.0, None)
    tri = tri / tri.sum() if tri.sum() > 1e-12 else np.array([1 / 3] * 3)
    out = [0.0] * 4
    for i, wt in zip(idx, tri):
        out[i] = float(wt)
    return tuple(out)


# --------------------------------------------------------------------------
# the scheduler: draft now, final when quiet


class _FinalWorker:
    """One daemon thread running final jobs from a latest-wins slot. ``submit`` replaces the
    slot and cancels the job in flight; the job polls its :class:`CancelToken`."""

    def __init__(self, run, results: queue.Queue):
        self._run = run
        self._results = results
        self._cond = threading.Condition()
        self._slot = None
        self._token = None
        self._running = False
        self._closed = False
        self._t_cancel: float | None = None
        self._thread: threading.Thread | None = None
        self.last_cancel_latency: float | None = None

    @property
    def busy(self) -> bool:
        with self._cond:
            return self._running or self._slot is not None

    def submit(self, gen, payload) -> None:
        with self._cond:
            self._slot = (gen, payload)
            self._cancel_locked()
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, name="lab-final", daemon=True)
                self._thread.start()
            self._cond.notify()

    def _cancel_locked(self) -> None:
        if self._running and self._token is not None and not self._token.cancelled:
            self._t_cancel = time.perf_counter()
            self._token.cancel()

    def cancel(self) -> None:
        with self._cond:
            self._slot = None
            self._cancel_locked()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._slot = None
            self._cancel_locked()
            self._cond.notify()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._slot is None and not self._closed:
                    self._cond.wait()
                if self._closed:
                    return
                gen, payload = self._slot
                self._slot = None
                self._token = R.CancelToken()
                self._running = True
                token = self._token
            try:
                frame = self._run(gen, payload, token)
                self._results.put(("frame", gen, frame))
            except R.Cancelled:
                if self._t_cancel is not None:
                    self.last_cancel_latency = time.perf_counter() - self._t_cancel
                    self._t_cancel = None
            except Exception as exc:                    # noqa: BLE001 - reported, never silent
                self._results.put(("error", gen, exc))
            finally:
                with self._cond:
                    self._running = False
                    self._t_cancel = None


class PreviewScheduler:
    """Two tiers of preview.

    ``request(payload)`` draws the DRAFT on the calling (UI) thread at once, cancels any final
    render in flight, and arms the final after ``debounce`` seconds of quiet. The host calls
    ``tick()`` every ~15 ms (tk ``after(15)``): it starts the final on the worker when due and
    delivers finished frames of the CURRENT generation (older ones are dropped). The worker
    gets only the frozen ``payload`` and returns a :class:`Frame` (bytes); it never touches
    tkinter. ``threaded=False`` runs the final inline on ``tick`` (deterministic tests)."""

    def __init__(self, draft_fn, final_fn, deliver, *, clock=time.monotonic,
                 debounce: float = DEBOUNCE_SECONDS, threaded: bool = True, on_error=None):
        self.draft_fn, self.final_fn, self.deliver = draft_fn, final_fn, deliver
        self.clock, self.debounce, self.threaded = clock, debounce, threaded
        self.on_error = on_error
        self.gen = 0
        self._pending = None
        self._due = 0.0
        self._results: queue.Queue = queue.Queue()
        self._worker = _FinalWorker(self._run_final, self._results) if threaded else None
        self.stale_dropped = 0
        self.errors: list = []
        self.drafts = 0
        self.finals = 0

    def _run_final(self, gen, payload, token):
        frame = self.final_fn(gen, payload, token)
        if frame is not None:
            frame.ppm()                     # bytes out: the main thread only hands them to tk
        return frame

    @property
    def last_cancel_latency(self):
        return self._worker.last_cancel_latency if self._worker else None

    def request(self, payload, *, final: bool = True) -> int:
        """A change happened: cancel the final in flight, draw a draft now, arm the final."""
        self.gen += 1
        gen = self.gen
        self._pending = None
        if self._worker is not None:
            self._worker.cancel()
        frame = self.draft_fn(gen, payload)
        if frame is not None:
            self.drafts += 1
            self.deliver(frame)
        if final:
            self._pending = (gen, payload)
            self._due = self.clock() + self.debounce
        return gen

    def final_due(self, now: float | None = None) -> bool:
        return self._pending is not None and (self.clock() if now is None else now) >= self._due

    @property
    def idle(self) -> bool:
        return self._pending is None and self._results.empty() and \
            not (self._worker is not None and self._worker.busy)

    def tick(self, now: float | None = None) -> None:
        """Deliver finished frames; start the final if the quiet time has passed."""
        self._drain()
        if self.final_due(now):
            gen, payload = self._pending
            self._pending = None
            if self._worker is not None:
                self._worker.submit(gen, payload)
            else:
                try:
                    frame = self._run_final(gen, payload, R.CancelToken())
                    self._results.put(("frame", gen, frame))
                except R.Cancelled:
                    pass
                except Exception as exc:                # noqa: BLE001
                    self._results.put(("error", gen, exc))
                self._drain()

    def _drain(self) -> None:
        while True:
            try:
                kind, gen, item = self._results.get_nowait()
            except queue.Empty:
                return
            if gen != self.gen:
                self.stale_dropped += 1
            elif kind == "frame":
                if item is not None:
                    self.finals += 1
                    self.deliver(item)
            else:
                self.errors.append(item)
                if self.on_error is not None:
                    self.on_error(item)

    def wait_idle(self, timeout: float = 30.0, step: float = 0.005) -> bool:
        """Tick until nothing is pending or running (scripts and tests; the UI uses ``after``)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.tick()
            if self.idle:
                return True
            time.sleep(step)
        return False

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()


class BackgroundTask:
    """A function run on a thread whose result the UI polls (export, printability check)."""

    def __init__(self, fn):
        self._result: queue.Queue = queue.Queue()
        self.done = False
        self.value = None
        self.error: BaseException | None = None

        def run():
            try:
                self._result.put((True, fn()))
            except BaseException as exc:                # noqa: BLE001 - handed to the poller
                self._result.put((False, exc))

        self._thread = threading.Thread(target=run, name="lab-task", daemon=True)
        self._thread.start()

    def poll(self) -> bool:
        """``True`` once finished (then ``value`` or ``error`` is set)."""
        if not self.done:
            try:
                ok, item = self._result.get_nowait()
            except queue.Empty:
                return False
            self.done = True
            if ok:
                self.value = item
            else:
                self.error = item
        return True

    def wait(self, timeout: float = 60.0) -> bool:
        self._thread.join(timeout)
        return self.poll()


# --------------------------------------------------------------------------
# the controller


class Lab:
    """Document + view + engine + scheduler: what the Character tab holds.

        lab = Lab()                       # shares the loaded MakeHuman assets
        lab.on_frame = show_frame         # called on the UI thread with each Frame
        lab.doc.add_appendage("wing", params={"style": "feathered"})
        ...                               # every edit draws a draft and arms the final
        lab.tick()                        # every ~15 ms from the UI loop

    ``size`` is the display size in pixels; panes larger than :data:`DRAFT_FULL_PIXELS`
    draw their drafts at half size (the Frame upscales)."""

    def __init__(self, resources: Resources | None = None, character: Character | None = None,
                 *, size=(900, 800), threaded: bool = True, clock=time.monotonic,
                 debounce: float = DEBOUNCE_SECONDS, levels: int = 0):
        self.resources = resources or Resources.shared()
        self.engine = Engine(self.resources)
        self.doc = LabDocument(self.resources, character, clock=clock, engine=self.engine)
        self.view = ViewState()
        self.size = (int(size[0]), int(size[1]))
        self.levels = levels
        self.frame: Frame | None = None
        self.on_frame = None
        self.on_change = None
        self.on_error = None
        self.placing: str | None = None
        self.auto_reveal = True
        self.draft_full_pixels = DRAFT_FULL_PIXELS
        self.scheduler = PreviewScheduler(self._draft, self._final, self._deliver, clock=clock,
                                          debounce=debounce, threaded=threaded,
                                          on_error=self._error)
        self.doc.listeners.append(self._doc_changed)

    # -- plumbing ----------------------------------------------------------------------------

    def close(self) -> None:
        self.scheduler.close()

    def _error(self, exc) -> None:
        if self.on_error is not None:
            self.on_error(exc)

    def _deliver(self, frame: Frame) -> None:
        self.frame = frame
        if self.on_frame is not None:
            self.on_frame(frame)

    @property
    def draft_scale(self) -> int:
        w, h = self.size
        return 2 if w * h > self.draft_full_pixels else 1

    def _draft(self, gen, payload) -> Frame:
        character, view, size, scale, mode, edited = payload["live"]
        return self.engine.render_draft(character, view, size, gen=gen, scale=scale, mode=mode,
                                        edited=edited)

    def _final(self, gen, payload, token) -> Frame:
        character = Character.from_dict(json.loads(payload["json"]))
        return self.engine.render_final(character, payload["view"], payload["size"], gen=gen,
                                        check=token, levels=payload["levels"])

    def refresh(self, change: Change | None = None) -> int:
        """Draw a draft now and arm the final. The change decides the shortcut the draft may take:
        a body-slider drag re-places the appendages instead of rebuilding them, a parameter or
        placement drag builds the edited paired appendage one side only, a pose drag poses one."""
        mode, edited = "exact", None
        if change is not None:
            if change.scope == "body":
                mode = "body"
            elif change.scope == "shape" and change.what.startswith(("param:", "placement:")):
                mode, edited = "shape", change.spec_id
            elif change.scope == "pose":
                mode, edited = "pose", change.spec_id
        view = self.view.copy()
        payload = {
            "live": (self.doc.character, view, self.size, self.draft_scale, mode, edited),
            "json": self.doc.snapshot(), "view": view, "size": self.size,
            "levels": self.levels,
        }
        return self.scheduler.request(payload)

    def tick(self) -> None:
        self.scheduler.tick()

    def _doc_changed(self, change: Change) -> None:
        if self.placing is not None and (
                self.doc.character.spec(self.placing) is None
                or (change.scope == "selection" and change.spec_id != self.placing)):
            self.placing = None                     # armed for something gone or deselected
        if change.scope == "all" and change.what in ("load", "new"):
            self.frame_content(refresh=False)
        if change.renders:
            if change.what == "add" and self.auto_reveal:
                self._reveal(change.spec_id)
            self.refresh(change)
        if self.on_change is not None:
            self.on_change(change)

    def _reveal(self, spec_id: str) -> None:
        """After adding something, make sure it can be seen: swing the camera round to the back
        if it sits on the back of a body seen from the front, and zoom out if it sticks out of
        the frame (the camera never zooms IN by itself)."""
        placed = self.engine.placed(self.doc.character, spec_id)
        if not placed:
            return
        on_back = float(placed[0].anchor[2]) < 0.0           # MakeHuman: the back faces -Z
        front = math.cos(math.radians(self.view.azimuth)) > 0.3
        if on_back and front:
            self.view.azimuth = VIEW_PRESETS["rear-quarter"][0]
            self.view.elevation = VIEW_PRESETS["rear-quarter"][1]
        lo, hi = self.engine.frame_bounds(self.doc.character)
        if float(np.linalg.norm(hi - lo)) / 2.0 > self.view.radius:
            self.view.frame(lo, hi)

    # -- the view ----------------------------------------------------------------------------

    def set_size(self, w: int, h: int) -> None:
        w, h = max(int(w), 16), max(int(h), 16)
        if (w, h) != self.size:
            self.size = (w, h)
            self.refresh(Change("view"))

    def orbit(self, dx: float, dy: float) -> None:
        self.view.orbit(dx, dy)
        self.refresh(Change("view"))

    def pan(self, dx: float, dy: float) -> None:
        self.view.pan(dx, dy, self.size)
        self.refresh(Change("view"))

    def zoom(self, factor: float) -> None:
        self.view.zoom(factor)
        self.refresh(Change("view"))

    def set_view(self, name: str) -> None:
        if name not in VIEW_PRESETS:
            raise LabError(f"unknown view {name!r}; use {', '.join(VIEW_PRESETS)}")
        self.view.azimuth, self.view.elevation = VIEW_PRESETS[name]
        self.refresh(Change("view"))

    def frame_content(self, refresh: bool = True) -> None:
        lo, hi = self.engine.frame_bounds(self.doc.character)
        self.view.frame(lo, hi)
        if refresh:
            self.refresh(Change("view"))

    # -- picking and placing -----------------------------------------------------------------

    def pick(self, x: float, y: float, body_only: bool = False):
        return self.engine.pick(self.doc.character, self.view, self.size, x, y, body_only)

    def start_placing(self, spec_id: str) -> None:
        self.doc.spec(spec_id)
        self.placing = spec_id

    def cancel_placing(self) -> None:
        self.placing = None

    def click(self, x: float, y: float) -> str:
        """A click on the image: places the armed appendage on the body quad under the pointer,
        else selects the appendage under it (or nothing). Returns a message for the status bar."""
        if self.placing is not None:
            pick = self.pick(x, y, body_only=True)
            if pick is None:
                return "missed the body: click on the body (Esc cancels)"
            spec_id, self.placing = self.placing, None
            self.doc.place_at_pick(spec_id, pick)
            return f"{spec_id} placed on body quad {pick.quad}"
        pick = self.pick(x, y)
        if pick is not None and pick.kind == "appendage":
            self.doc.select(pick.spec_id)
            return f"selected {pick.spec_id}"
        self.doc.select(None)
        return "nothing selected" if pick is None else f"body quad {pick.quad}"

    # -- overlays (world points the view draws as canvas items) -------------------------------

    def overlay_points(self) -> list:
        """``[(label, world point Z-up, selected)]`` anchor markers of the appendages. Read off
        the body quads the appendages are bound to: nothing is built to draw them."""
        from .portrait import MH_TO_Z_UP
        body = self.engine.body_state(self.doc.character)
        out = []
        for spec in self.engine.buildable(self.doc.character):
            for letter, point in self.engine.anchor_points(spec, body):
                out.append((f"{spec.id}.{letter}" if letter else spec.id,
                            MH_TO_Z_UP[:3, :3] @ point, spec.id == self.doc.selected))
        return out

    def floor_z(self) -> float:
        return self.engine.floor_z(self.doc.character)

    # -- export ------------------------------------------------------------------------------

    def export_stl(self, path, **kw) -> ExportResult:
        return self.engine.export(self.doc.character, path, levels=kw.pop("levels", 0), **kw)

    def export_async(self, path=None, **kw) -> BackgroundTask:
        """:meth:`export_stl` on a thread (an angel's report takes seconds), on a frozen copy of
        the document so edits made meanwhile do not leak into the file. ``path=None`` only checks
        printability. Poll the returned :class:`BackgroundTask`."""
        frozen = Character.from_dict(json.loads(self.doc.snapshot()))
        levels = kw.pop("levels", 0)
        return BackgroundTask(lambda: self.engine.export(frozen, path, levels=levels, **kw))
