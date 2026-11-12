"""Skin-conforming junctions: the body's skin flows into an appendage root.

THE PROBLEM. An appendage is its own closed shell that overlaps the body (the slicer unions
overlapping shells), so where a tail, horn, wing arm or extra limb enters the skin there is a
hard intersection curve and nothing else: the skin does not rise to meet it, no flesh gathers
around it, it reads as "slapped over skin". The body mesh cannot get new vertices (1,280
morph targets address it BY VERTEX INDEX) but its vertices can be MOVED after the morphs are
applied. That is what this module does.

THE MECHANISM. A :class:`JunctionSpec` describes the root of an appendage for the body:

* a **centreline** with radii (the first part of the appendage, root first, starting
  inside the body): the tube the skin should flow into;
* a **fillet** of width ``W = width * base_radius``: every body vertex outside the tube by
  less than ``W`` is moved so that its remaining distance is ``rho = W f(delta / W)`` with
  ``f(u) = 2u^2 - u^3`` (it is pulled toward the tube in the skin plane, tangent to the
  tube at the join and unchanged at ``W``), and it climbs the tube wall to the height of a
  concave profile ``h(rho) = H (1 - sqrt(1 - (1 - rho/W)^k))`` that is tangent to the wall
  at ``rho = 0`` and to the skin at ``rho = W`` (``k = 2``: a quarter-ellipse, a circle for
  ``H = W``; ``k = 3``: curvature-continuous at ``W``). The pull only redistributes the
  vertices ALONG the profile (the cage is 2-4 quads wide at this scale); every vertex ends
  on the profile. Vertices inside the tube climb with it (they are hidden) and are kept
  inside it;
* optional **mounds** (a smooth outward bulge along the skin normal: scapular mass, sacral
  rise, deltoid cap; a negative height is a hollow; a mound with an ``end`` is a ridge or
  a groove) and **keep** spheres (what must stay sharp: lips, eyelids, a hand).

THE GUARDS (S1b). The law above is the SHAPE the skin wants; the cage (0.28 units a quad, 0.08
in the cleft) cannot always follow it, and the body has other skin than the root's. In order:
the **owner** (``JunctionSpec.owner``, :mod:`.bodyframe`): only the skin of the part of the body
the root belongs to moves, the fillet is capped by that part's size and a root thicker than the part
moulds nothing; the **exposed** tube only (a section buried under the skin never tents it, and
the skin never climbs higher than the tube stands); **relative sizes** (``Mound.scale``); the tube
**follows the real mesh** (:func:`fit_radii_to_mesh`); the displacement is **smoothed** over the
cage (:func:`diffuse_displacement`), **faded** where two sheets of skin already touch (a closed
cleft); and a **validity pass** (:func:`limit_displacement`) backs vertices off until no cage quad
flips, collapses or grows past 2x, no edge grows past 2x, no two neighbours move more than 1.25
edge lengths apart, and the skin has not newly crossed itself (:mod:`.junction_checks`) outside the
tube. :func:`junction_quality` reports all of it.

:func:`apply_junctions` is a pure, deterministic function of ``(verts, specs)``: it
returns new vertices of the same shape (topology and numbering are untouched, so every
morph target still addresses the same vertices), moving only body cage vertices, and
returns ``verts`` ITSELF when there is nothing to do (so a character without appendages is
bit-identical to what it was). Every displacement is capped, and quads that would flip or
collapse are backed off (:func:`apply_junctions` reports what it did in ``info``).

THE FRAMES. Placement (``SurfaceAnchor``, the mirror-face map, ``kinds.instantiate_all``)
must keep using the UNDEFORMED morphed vertices, otherwise an anchor chases its own
displacement. The junction is applied AFTER the morph weights and the placement, BEFORE the
body is meshed, rendered or printed: ``verts -> instantiate -> shaped_body -> everything
else``. See :func:`shaped_body`.

Everything here is in the MakeHuman frame (Y up), the frame of the morphed body.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from dataclasses import dataclass, field, replace

import numpy as np

from . import junction_checks as JC
from .bodyframe import frame_or_none

# --------------------------------------------------------------------------
# defaults

DEFAULT_WIDTH = 1.0           # fillet width, in units of the base radius
DEFAULT_HEIGHT = 0.6          # fillet height / width (1 = a quarter circle)
DEFAULT_SHARPNESS = 3.0       # k of the profile: 2 = quarter ellipse (C1 at W), 3 = curvature-continuous
DEFAULT_LEAD = 0.25           # how far (in fillet widths) the skin starts to rise before the tube
MAX_MOVE_FACTOR = 1.6         # a vertex moves at most this many fillet widths
MIN_AREA_RATIO = 0.12         # a quad may not shrink below this fraction of its own area
MIN_NORMAL_DOT = 0.2          # ... nor turn its normal past this (cos of ~78 degrees)
LIMIT_ITERATIONS = 80
NORMAL_SMOOTHING = 2          # S1b: rounds of averaging of the skin normals the displacement is built on
TIGHT_GAP = 0.04              # S1b: skin within two rings of facing skin closer than this is not moved (fading back in over 2 more)
INTERSECTION_MIN_MOVE = 0.01   # S1b: vertices that move by less than this do not start a crossing
DIFFUSE_ITERATIONS = 20       # S1b: smoothing rounds of the raw displacement (diffuse_displacement)
LIMIT_BACKOFF = 0.9
MAX_AREA_GROWTH = 2.0         # S1b: a cage quad may not grow past this multiple of its own area ...
MAX_EDGE_GROWTH = 2.0         # ... nor an edge past this multiple of its length ...
MAX_LIPSCHITZ = 1.25          # ... and two neighbours' displacements differ by at most this many edge lengths

# S6: how far a spec may push the skin of the part it belongs to OUT of that part, as a fraction of
# the skin point's own distance from the part's bone (at least half the part's radius): a root on a
# head is a thin cuff on bone, not a reshaping of the skull (S1b's default horn widened the head 41 %
# 15 % below the crown, a ram's horn 22-29 % at the temple). Parts that are not listed are not bound.
PART_SILHOUETTE = {"head": 0.08}
JOIN_WORSE = 2.0              # S6: a spec whose tube join gets worse by more than this many degrees is not applied (the mean over a patch is good to about a degree)
JOIN_BAND = 0.25              # ... measured on the skin within this many fillet widths of the wall
JOIN_REACH = 0.4              # ... on a patch of the cage reaching this far (units) beyond that band, and two rings

_EPS = 1e-12
MOUND_SCALES = ("local", "head", "trunk", "arm", "leg")


# --------------------------------------------------------------------------
# the description


def _vec(p, name="point") -> tuple:
    t = tuple(float(x) for x in p)
    if len(t) != 3 or not all(math.isfinite(x) for x in t):
        raise ValueError(f"{name} must be three finite numbers, got {p!r}")
    return t


def _mirror_point(p) -> tuple:
    return (-p[0], p[1], p[2])


@dataclass(frozen=True)
class Mound:
    """A smooth bulge of the skin along its own normal (``height`` < 0: a hollow).

    The influence is ``w(u) = 1 - 3u^2 + 2u^3`` of ``u = distance / radius`` (C1 at both
    ends, zero beyond ``radius``). ``end`` turns the point into a ridge (or groove) along the
    segment ``center -> end``; ``axis`` with ``stretch`` > 1 elongates a point mound along
    that axis. ``direction`` replaces the skin normal by a fixed direction.

    ``scale`` makes ``radius`` and ``height`` FRACTIONS of a body dimension that
    :func:`apply_junctions` resolves from the body's own skeleton (:mod:`.bodyframe`):
    ``"head"``, ``"trunk"``, ``"arm"``, ``"leg"`` (the radius of that part: a head 1.0, a
    trunk 1.2, an arm 0.5 units) or ``"local"`` (the part ``center`` lies in). A scapular mound
    of ``radius=0.55, height=0.07, scale="trunk"`` is a third of a back on any body, where
    0.55 units is a pillow on a child. Without a skeleton (a synthetic body) the numbers stay
    absolute."""

    center: tuple
    radius: float
    height: float
    end: tuple | None = None
    axis: tuple | None = None
    stretch: float = 1.0
    direction: tuple | None = None
    scale: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "center", _vec(self.center, "mound center"))
        if self.end is not None:
            object.__setattr__(self, "end", _vec(self.end, "mound end"))
        if self.axis is not None:
            object.__setattr__(self, "axis", _vec(self.axis, "mound axis"))
        if self.direction is not None:
            object.__setattr__(self, "direction", _vec(self.direction, "mound direction"))
        if not (math.isfinite(self.radius) and self.radius > 0):
            raise ValueError("mound radius must be positive")
        if not math.isfinite(self.height):
            raise ValueError("mound height must be finite")
        if not (math.isfinite(self.stretch) and self.stretch > 0):
            raise ValueError("mound stretch must be positive")
        if self.scale is not None and self.scale not in MOUND_SCALES:
            raise ValueError(f"mound scale must be one of {MOUND_SCALES} or None, got {self.scale!r}")

    def mirrored(self) -> "Mound":
        m = _mirror_point
        return Mound(m(self.center), self.radius, self.height,
                     None if self.end is None else m(self.end),
                     None if self.axis is None else m(self.axis), self.stretch,
                     None if self.direction is None else m(self.direction), self.scale)

    def transformed(self, matrix) -> "Mound":
        M = np.asarray(matrix, dtype=np.float64)
        s = _uniform_scale(M)

        def pt(p):
            return None if p is None else tuple(M[:3, :3] @ np.asarray(p) + M[:3, 3])

        def vec(p):
            return None if p is None else tuple(M[:3, :3] @ np.asarray(p) / s)

        k = 1.0 if self.scale else s          # a relative size is not a length: it is not scaled
        return Mound(pt(self.center), self.radius * k, self.height * k, pt(self.end),
                     vec(self.axis), self.stretch, vec(self.direction), self.scale)

    def to_dict(self) -> dict:
        d = {"center": list(self.center), "radius": self.radius, "height": self.height}
        for k in ("end", "axis", "direction"):
            if getattr(self, k) is not None:
                d[k] = list(getattr(self, k))
        if self.stretch != 1.0:
            d["stretch"] = self.stretch
        if self.scale is not None:
            d["scale"] = self.scale
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Mound":
        return cls(d["center"], d["radius"], d["height"], d.get("end"), d.get("axis"),
                   d.get("stretch", 1.0), d.get("direction"), d.get("scale"))


@dataclass(frozen=True)
class Keep:
    """A sphere where the junction must leave the skin as it is (what must stay sharp: the
    lip line, the eyelids, a hand): the displacement of the spec is zero inside ``radius``
    and grows back to full strength over ``fade * radius`` beyond it (smoothly)."""

    center: tuple
    radius: float
    fade: float = 0.5

    def __post_init__(self):
        object.__setattr__(self, "center", _vec(self.center, "keep center"))
        if not (math.isfinite(self.radius) and self.radius > 0):
            raise ValueError("keep radius must be positive")
        if not (math.isfinite(self.fade) and self.fade > 0):
            raise ValueError("keep fade must be positive")

    def mirrored(self) -> "Keep":
        return Keep(_mirror_point(self.center), self.radius, self.fade)

    def transformed(self, matrix) -> "Keep":
        M = np.asarray(matrix, dtype=np.float64)
        s = _uniform_scale(M)
        return Keep(tuple(M[:3, :3] @ np.asarray(self.center) + M[:3, 3]), self.radius * s,
                    self.fade)

    def to_dict(self) -> dict:
        return {"center": list(self.center), "radius": self.radius, "fade": self.fade}

    @classmethod
    def from_dict(cls, d: dict) -> "Keep":
        return cls(d["center"], d["radius"], d.get("fade", 0.5))


def _uniform_scale(M) -> float:
    s = abs(float(np.linalg.det(M[:3, :3]))) ** (1.0 / 3.0)
    return s if s > _EPS else 1.0


class ShellRef:
    """The mesh an appendage's tube stands for, as the junction asks about it: a lazy handle (a
    function that builds the mesh, so a spec that is never applied never poses a wing) with the
    similarity that carries the mesh to the frame of the spec (a transformed spec has the
    transformed mesh). It is not part of a spec's value: not compared, not serialised, not in
    its digest. The mirror image of a handle is its TWIN: the tube of the right-hand spec is cut
    down by mirroring what the left-hand one was cut down to, so a mirrored pair stays exactly
    mirror-symmetric (a point just inside a mesh on one side must not be just outside on the other)."""

    def __init__(self, mesh_fn, matrix=None, twin=None):
        self._fn = mesh_fn
        self.matrix = np.eye(4) if matrix is None else np.asarray(matrix, dtype=np.float64)
        self._twin = twin
        self._shells = None
        self._fits: dict = {}

    def mirrored(self) -> "ShellRef":
        if self._twin is not None:
            return self._twin
        return ShellRef(self._fn, np.diag([-1.0, 1.0, 1.0, 1.0]) @ self.matrix, twin=self)

    def transformed(self, matrix) -> "ShellRef":
        return ShellRef(self._fn, np.asarray(matrix, dtype=np.float64) @ self.matrix)

    def shells(self):
        if self._shells is None:
            mesh = self._fn()
            if np.allclose(self.matrix, np.eye(4)):
                self._shells = JC.shells_of(mesh)
            else:
                M = self.matrix
                self._shells = JC.Shells(mesh.vertices @ M[:3, :3].T + M[:3, 3], mesh.faces)
        return self._shells

    def fit(self, spec: "JunctionSpec"):
        """``(points, radii)`` of ``spec``'s tube cut down to what lies inside the mesh."""
        key = spec.digest()
        if key not in self._fits:
            if self._twin is not None:
                pts, rad = self._twin.fit(spec.mirrored())
                pts = np.asarray(pts) * np.array([-1.0, 1.0, 1.0])
            else:
                pts, rad = JC.fit_tube_to_mesh(spec.points, spec.radii, shells=self.shells())
            self._fits[key] = (pts, rad)
        return self._fits[key]


@dataclass(frozen=True)
class JunctionSpec:
    """The root of one appendage, as the body's skin sees it.

    ``points`` / ``radii``: the centreline polyline and the radius of the surface the skin
    should hug at each point, root FIRST (the first points are inside the body, the
    appendage's sunk end); give only the first part of the appendage (to a few fillet
    widths beyond where it leaves the skin), or the skin would also flow into a tail that
    curls back toward the buttock. Hand over the radius of the surface the skin should
    meet (a polygonal tube: the inscribed circle, so the skin lands inside the facets).

    ``width`` is the fillet width in units of ``base_radius`` (``None``: the largest of
    ``radii``); ``height`` the fillet height over its width; ``sharpness`` the exponent
    ``k`` of the profile; ``pull`` how much of the in-plane redistribution is applied
    (0 = none: the skin only climbs); ``strength`` scales the whole spec (0 switches it off).
    ``lead``: the skin only moulds around the EXPOSED part of the tube, and begins to rise
    ``lead`` fillet widths before the tube's axis reaches the skin (a tube buried deeper than
    that leaves the skin above it alone). ``mounds`` and ``keep`` as described in the module docstring; ``max_move`` caps how
    far one vertex may move (default ``MAX_MOVE_FACTOR`` fillet widths, and at least the
    tallest mound). ``owner``: the part of the body this root belongs to (``"head"``,
    ``"trunk"``, ``"arm"``, ``"leg"`` or a part of :mod:`.bodyframe`); the spec only moulds the
    skin of that part (and the fillet is capped by its size). ``None``: the part the tube comes
    out of the skin in, which is right unless the appendage is rooted where another part's skin
    is (an extra limb that starts at the original shoulder).

    ``ownership=False`` switches the skeleton off for this spec alone (what ``frame=False`` does for
    a whole call): the spec moulds the skin of every part, its fillet is not capped by the size of a
    part and a root thicker than the part is not left out; use it only for a root that straddles
    parts on purpose. ``silhouette`` bounds how far the spec may push the skin of the part it
    belongs to OUT of that part (a fraction of the vertex's own distance from the part's axis, at
    least half the part's radius): ``None`` is the part's default (``PART_SILHOUETTE``: 0.08 for a
    head, no bound for the others), ``0`` switches the bound off, any other number is the bound.

    ``shell`` (a :class:`ShellRef`, or ``None``): the mesh of the appendage this tube is the first
    part of. :func:`apply_junctions` cuts a tube that does not lie inside it down to what does
    (:func:`conform_spec`); the spec itself keeps the tube the kind built. Not part of the value of
    the spec (not compared, serialised or hashed). ``conform=False`` leaves the tube as the kind
    built it (a kind that has made sure it lies in its mesh, or whose tube deliberately stands for more
    than the shell). ``join_guard=False`` exempts the spec from :func:`apply_junctions`' ``drop_worse``
    (a spec whose mounds are mass on purpose, a skirt or a fold, which the join angle may not judge).

    ``tight_gap`` / ``smooth`` override :func:`apply_junctions`' own for this spec alone
    (``None``: the call's; ``tight_gap=0``: this spec does not leave skin that nearly touches
    skin where it is, because the spec holds the cleft itself with a ``keep``; ``smooth=0``: the
    tube displacement is not diffused over the cage). The spec that knows its own anatomy may
    use them, the generic ones do not.
    """

    name: str
    points: tuple
    radii: tuple
    width: float = DEFAULT_WIDTH
    base_radius: float | None = None
    height: float = DEFAULT_HEIGHT
    sharpness: float = DEFAULT_SHARPNESS
    pull: float = 1.0
    strength: float = 1.0
    mounds: tuple = ()
    keep: tuple = ()
    max_move: float | None = None
    lead: float = DEFAULT_LEAD
    owner: str | None = None
    tight_gap: float | None = None
    smooth: int | None = None
    ownership: bool = True
    silhouette: float | None = None
    conform: bool = True
    join_guard: bool = True
    shell: object = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        pts = tuple(_vec(p) for p in self.points)
        rad = tuple(float(r) for r in self.radii)
        object.__setattr__(self, "points", pts)
        object.__setattr__(self, "radii", rad)
        object.__setattr__(self, "mounds", tuple(self.mounds))
        object.__setattr__(self, "keep", tuple(self.keep))
        if pts:
            if len(pts) != len(rad):
                raise ValueError("points and radii must have the same length")
            if len(pts) < 2:
                raise ValueError("a junction centreline needs at least two points")
            if not all(math.isfinite(r) and r > 0 for r in rad):
                raise ValueError("radii must be positive and finite")
        for name in ("width", "height", "sharpness", "pull", "strength", "lead"):
            v = getattr(self, name)
            if not (isinstance(v, numbers.Real) and not isinstance(v, bool)
                    and math.isfinite(v) and v >= 0):
                raise ValueError(f"{name} must be a finite number >= 0, got {v!r}")
            object.__setattr__(self, name, float(v))
        if self.owner is not None and not (isinstance(self.owner, str) and self.owner):
            raise ValueError(f"owner must be a part name or None, got {self.owner!r}")
        if self.tight_gap is not None and not (
                isinstance(self.tight_gap, numbers.Real) and not isinstance(self.tight_gap, bool)
                and math.isfinite(self.tight_gap) and self.tight_gap >= 0):
            raise ValueError(f"tight_gap must be None or a finite number >= 0, got {self.tight_gap!r}")
        if self.smooth is not None and not (
                isinstance(self.smooth, numbers.Integral) and not isinstance(self.smooth, bool)
                and self.smooth >= 0):
            raise ValueError(f"smooth must be None or an integer >= 0, got {self.smooth!r}")
        for flag in ("ownership", "conform", "join_guard"):
            if not isinstance(getattr(self, flag), (bool, np.bool_)):
                raise ValueError(f"{flag} must be a bool, got {getattr(self, flag)!r}")
            object.__setattr__(self, flag, bool(getattr(self, flag)))
        if self.silhouette is not None and not (
                isinstance(self.silhouette, numbers.Real) and not isinstance(self.silhouette, bool)
                and math.isfinite(self.silhouette) and self.silhouette >= 0):
            raise ValueError(f"silhouette must be None or a finite number >= 0, got {self.silhouette!r}")
        if self.sharpness < 1.0:
            raise ValueError("sharpness must be >= 1")
        if self.pull > 1.0:
            raise ValueError("pull must be within 0..1")

    # -- derived -------------------------------------------------------------------------

    @property
    def tube(self) -> bool:
        return bool(self.points)

    @property
    def reference_radius(self) -> float:
        return float(self.base_radius) if self.base_radius is not None else (
            max(self.radii) if self.radii else 0.0)

    @property
    def fillet_width(self) -> float:
        """The absolute width ``W`` of the fillet (0 when there is no tube or it is off)."""
        return self.width * self.reference_radius if self.tube else 0.0

    @property
    def active(self) -> bool:
        return self.strength > 0 and (
            (self.tube and self.fillet_width > 0) or bool(self.mounds))

    def move_limit(self) -> float:
        if self.max_move is not None:
            return float(self.max_move)
        tallest = max((abs(m.height) for m in self.mounds), default=0.0)
        return max(MAX_MOVE_FACTOR * max(self.fillet_width * max(self.height, 1.0), 0.0),
                   1.25 * tallest)

    def with_(self, **changes) -> "JunctionSpec":
        return replace(self, **changes)

    # -- frames --------------------------------------------------------------------------

    def mirrored(self) -> "JunctionSpec":
        """The x-mirror image of this spec (the right-hand twin of a left-hand spec)."""
        return replace(self, points=tuple(_mirror_point(p) for p in self.points),
                       mounds=tuple(m.mirrored() for m in self.mounds),
                       keep=tuple(k.mirrored() for k in self.keep),
                       shell=None if self.shell is None else self.shell.mirrored())

    def transformed(self, matrix) -> "JunctionSpec":
        """This spec carried by a 4x4 similarity (rotation x uniform scale + translation)."""
        M = np.asarray(matrix, dtype=np.float64)
        s = _uniform_scale(M)
        pts = tuple(tuple(M[:3, :3] @ np.asarray(p) + M[:3, 3]) for p in self.points)
        return replace(
            self, points=pts, radii=tuple(r * s for r in self.radii),
            base_radius=None if self.base_radius is None else self.base_radius * s,
            mounds=tuple(m.transformed(M) for m in self.mounds),
            keep=tuple(k.transformed(M) for k in self.keep),
            max_move=None if self.max_move is None else self.max_move * s,
            shell=None if self.shell is None else self.shell.transformed(M))

    # -- serialisation ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = {"name": self.name, "points": [list(p) for p in self.points],
             "radii": list(self.radii), "width": self.width, "height": self.height,
             "sharpness": self.sharpness, "pull": self.pull, "strength": self.strength}
        if self.base_radius is not None:
            d["base_radius"] = self.base_radius
        if self.mounds:
            d["mounds"] = [m.to_dict() for m in self.mounds]
        if self.keep:
            d["keep"] = [k.to_dict() for k in self.keep]
        if self.max_move is not None:
            d["max_move"] = self.max_move
        if self.lead != DEFAULT_LEAD:
            d["lead"] = self.lead
        if self.owner is not None:
            d["owner"] = self.owner
        if self.tight_gap is not None:
            d["tight_gap"] = self.tight_gap
        if self.smooth is not None:
            d["smooth"] = self.smooth
        if not self.ownership:
            d["ownership"] = False
        if not self.conform:
            d["conform"] = False
        if not self.join_guard:
            d["join_guard"] = False
        if self.silhouette is not None:
            d["silhouette"] = self.silhouette
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "JunctionSpec":
        return cls(d["name"], d.get("points", ()), d.get("radii", ()),
                   d.get("width", DEFAULT_WIDTH), d.get("base_radius"),
                   d.get("height", DEFAULT_HEIGHT), d.get("sharpness", DEFAULT_SHARPNESS),
                   d.get("pull", 1.0), d.get("strength", 1.0),
                   tuple(Mound.from_dict(m) for m in d.get("mounds", ())),
                   tuple(Keep.from_dict(k) for k in d.get("keep", ())), d.get("max_move"),
                   d.get("lead", DEFAULT_LEAD), d.get("owner"), d.get("tight_gap"), d.get("smooth"),
                   d.get("ownership", True), d.get("silhouette"), d.get("conform", True),
                   d.get("join_guard", True))

    def digest(self) -> str:
        """A content hash (a cache key): equal specs have equal digests."""
        return hashlib.blake2b(json.dumps(self.to_dict(), sort_keys=True).encode(),
                               digest_size=8).hexdigest()


def specs_digest(specs) -> str:
    """The digest of a list of specs (order matters not: the junction is a sum)."""
    return hashlib.blake2b("|".join(sorted(s.digest() for s in specs if s is not None)).encode(),
                           digest_size=8).hexdigest()


# --------------------------------------------------------------------------
# geometry helpers


def _cross(a, b) -> np.ndarray:
    """``np.cross`` for ``(n, 3)`` arrays, three times faster (``np.cross`` is slow on small rows)."""
    out = np.empty_like(a)
    out[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    out[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    out[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return out


def vertex_normals(verts, quads) -> np.ndarray:
    """Area-weighted outward vertex normals of a quad mesh (unit; zero for an isolated vertex)."""
    verts = np.asarray(verts, dtype=np.float64)
    quads = np.asarray(quads)
    p = verts[quads]
    fn = _cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])         # twice the area, as a vector
    n = len(verts)
    flat = quads.ravel()
    vn = np.stack([np.bincount(flat, weights=np.repeat(fn[:, c], 4), minlength=n)
                   for c in range(3)], axis=1)
    ln = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(ln, _EPS)


def smooth_normals(verts, quads, normals, rounds: int = 2) -> np.ndarray:
    """``normals`` averaged over the cage edges ``rounds`` times (unit again): the skin normal
    of a vertex in a crease (the gluteal cleft, an armpit) flips between neighbours that are
    0.08 apart, and any displacement built on it is as noisy; averaged, it is the normal of the
    surface at the scale of the fillet."""
    n = np.asarray(normals, dtype=np.float64)
    e = JC.cage_edges(quads)
    a = np.concatenate([e[:, 0], e[:, 1]])
    b = np.concatenate([e[:, 1], e[:, 0]])
    cnt = np.maximum(np.bincount(a, minlength=len(n)), 1)[:, None]
    for _ in range(rounds):
        acc = np.stack([np.bincount(a, weights=n[b, c], minlength=len(n)) for c in range(3)], axis=1)
        n = n + acc / cnt
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), _EPS)
    return n


def tube_distance(points, centre, radii, radius: bool = False):
    """Signed distance of ``points`` to the round-cone chain ``centre`` / ``radii`` and the
    closest centreline point: ``(distance, closest)`` (``(distance, closest, radius there)``
    with ``radius=True``). Negative inside the tube.

    Per segment the distance is ``|p - c(t)| - r(t)`` at the closest centreline point
    ``c(t)`` (exact for a constant radius, a close bound for a gentle taper); the chain's
    distance is the minimum over segments."""
    pts = np.asarray(points, dtype=np.float64)
    best = np.full(len(pts), np.inf)
    closest = np.zeros_like(pts)
    rad_at = np.zeros(len(pts))
    for a, b, ra, rb in zip(centre[:-1], centre[1:], radii[:-1], radii[1:]):
        ab = b - a
        den = float(ab @ ab)
        t = np.clip(((pts - a) @ ab) / max(den, _EPS), 0.0, 1.0)
        c = a + t[:, None] * ab
        r = ra + t * (rb - ra)
        d = _norm(pts - c) - r
        better = d < best
        best = np.where(better, d, best)
        closest = np.where(better[:, None], c, closest)
        rad_at = np.where(better, r, rad_at)
    return (best, closest, rad_at) if radius else (best, closest)


def fillet_remap(u):
    """The in-plane remap of the fillet: ``f(u) = 2u^2 - u^3`` (``f(0)=0``, ``f(1)=1``,
    ``f'(1)=1``: the skin is tangent to the tube at the join and unchanged at ``W``)."""
    u = np.asarray(u, dtype=np.float64)
    return 2.0 * u * u - u ** 3


def fillet_height(rho, W: float, H: float, k: float):
    """The fillet profile: the height a skin vertex at remaining distance ``rho`` (0..W)
    from the wall has climbed, ``H (1 - sqrt(1 - x^k))`` with ``x = 1 - rho/W``. Vertical at
    the wall (``rho = 0``: tangent to the tube), horizontal at ``rho = W`` (tangent to the
    skin); its second derivative vanishes at ``W`` for ``k >= 3``."""
    x = np.clip(1.0 - np.asarray(rho, dtype=np.float64) / W, 0.0, 1.0)
    return H * (1.0 - np.sqrt(np.maximum(1.0 - x ** k, 0.0)))


def _norm(v) -> np.ndarray:
    """Row lengths of an ``(n, 3)`` array, without ``np.linalg.norm``'s per-call overhead (which is
    most of its cost on the small arrays of a junction)."""
    return np.sqrt((v * v).sum(axis=1))


def _unit(v):
    return v / np.maximum(_norm(v), _EPS)[:, None]


def _smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _bump(u):
    """1 at 0, 0 at 1, C1 at both ends."""
    u = np.clip(u, 0.0, 1.0)
    return 1.0 - u * u * (3.0 - 2.0 * u)


# --------------------------------------------------------------------------
# the displacement of one spec


def _segment_distance(pts, a, b):
    ab = b - a
    t = np.clip(((pts - a) @ ab) / max(float(ab @ ab), _EPS), 0.0, 1.0)
    return np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)


def _candidates(spec: JunctionSpec, cols: np.ndarray) -> np.ndarray:
    """Indices of the cage vertices that can possibly be affected by ``spec`` (a bounding
    box test on the contiguous ``(3, n)`` columns of the cage: cheap, and it keeps the cost
    proportional to the junction, not the body)."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    W = spec.fillet_width
    if spec.tube and W > 0:
        pts = np.asarray(spec.points)
        r = max(spec.radii)
        lo = np.minimum(lo, pts.min(axis=0) - r - W)
        hi = np.maximum(hi, pts.max(axis=0) + r + W)
    for m in spec.mounds:
        extent = np.array([m.radius * max(m.stretch, 1.0)] * 3)
        pts = [m.center] + ([m.end] if m.end is not None else [])
        lo = np.minimum(lo, np.min(pts, axis=0) - extent)
        hi = np.maximum(hi, np.max(pts, axis=0) + extent)
    inside = (cols[0] >= lo[0]) & (cols[0] <= hi[0])
    for c in (1, 2):
        inside &= (cols[c] >= lo[c]) & (cols[c] <= hi[c])
    return np.nonzero(inside)[0]


def tube_closest(points, centre, radii):
    """Like :func:`tube_distance` but with the position ALONG the centreline too:
    ``(distance, closest, radius there, arclength of the closest point from the root)``."""
    pts = np.asarray(points, dtype=np.float64)
    best = np.full(len(pts), np.inf)
    closest = np.zeros_like(pts)
    rad_at = np.zeros(len(pts))
    arc = np.zeros(len(pts))
    along = 0.0
    for a, b, ra, rb in zip(centre[:-1], centre[1:], radii[:-1], radii[1:]):
        ab = b - a
        den = float(ab @ ab)
        t = np.clip(((pts - a) @ ab) / max(den, _EPS), 0.0, 1.0)
        c = a + t[:, None] * ab
        r = ra + t * (rb - ra)
        dd = _norm(pts - c) - r
        better = dd < best
        best = np.where(better, dd, best)
        closest = np.where(better[:, None], c, closest)
        rad_at = np.where(better, r, rad_at)
        arc = np.where(better, along + t * math.sqrt(den), arc)
        along += math.sqrt(den)
    return best, closest, rad_at, arc


def tube_emergence(spec: JunctionSpec, P, N):
    """Where the centreline of ``spec`` comes OUT of the skin, read off the skin vertices
    ``P`` (normals ``N``) around it: ``(arclength, point, radius there)`` of the first sample
    of the centreline whose axis is no more than half a radius below the skin, or ``None`` when
    the whole tube is buried (then there is nothing to mould the skin around). The centreline
    is sampled every ~0.05 fillet widths; the skin height at a sample is the distance to the
    nearest skin vertex along that vertex's normal (a sample with no skin vertex within two
    fillet widths and radii counts as deeply buried)."""
    pts = np.asarray(spec.points)
    rad = np.asarray(spec.radii)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    W = spec.fillet_width
    n = int(np.clip(cum[-1] / max(0.05 * W, 0.02), 8, 200))
    s = np.linspace(0.0, cum[-1], n)
    idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    t = (s - cum[idx]) / np.maximum(seg[idx], _EPS)
    samples = pts[idx] + t[:, None] * (pts[idx + 1] - pts[idx])
    radius = rad[idx] + t * (rad[idx + 1] - rad[idx])
    if not len(P):
        return None
    d2 = ((samples[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)
    j = d2.argmin(axis=1)
    h = ((samples - P[j]) * N[j]).sum(axis=1)
    h[np.sqrt(d2[np.arange(n), j]) > 2.0 * (W + radius)] = -np.inf
    up = np.nonzero(h >= -0.5 * radius)[0]
    if not len(up):
        return None
    i = int(up[0])
    return float(s[i]), samples[i], float(radius[i])


def _tube_displacement(spec, P, N, inside_margin, W=None, weight=None):
    """The fillet displacement of the points ``P`` (with skin normals ``N``): ``(m, 3)``.

    ``W``: the fillet width in effect (capped by the body); ``weight``: a per-point 0..1 factor
    (the part of the body the root belongs to)."""
    W = spec.fillet_width if W is None else W
    H = spec.height * W
    k = spec.sharpness
    centre = np.asarray(spec.points)
    radii = np.asarray(spec.radii)
    D, C, Rc, S = tube_closest(P, centre, radii)
    out = np.zeros_like(P)
    act = D < W
    if weight is not None:
        act &= weight > 0.0
    if not act.any():
        return out
    Pa, Na, Da, Ca, rad_c, Sa = P[act], N[act], D[act], C[act], Rc[act], S[act]
    nu = Pa - Ca
    nu /= np.maximum(np.linalg.norm(nu, axis=1, keepdims=True), _EPS)
    cps = (nu * Na).sum(axis=1)                                  # cos of the angle skin-normal / wall-normal
    sps = np.sqrt(np.maximum(1.0 - cps * cps, 0.0))              # 0 above the tube, 1 beside it
    away = nu - cps[:, None] * Na                                # in the skin plane, away from the tube
    away /= np.maximum(np.linalg.norm(away, axis=1, keepdims=True), 1e-9)
    climb = Na - cps[:, None] * nu                               # up the wall: the skin normal's tangent part
    climb /= np.maximum(np.linalg.norm(climb, axis=1, keepdims=True), 1e-9)
    # only the EXPOSED tube is moulded around: the weight falls to zero as the closest point of
    # the centreline sinks below the skin (a tube buried under the scalp leaves it alone) ...
    height_c = ((Ca - Pa) * Na).sum(axis=1)
    tol = np.minimum(np.maximum(rad_c, 0.0), W) + spec.lead * W
    expose = _smoothstep(1.0 + height_c / np.maximum(tol, _EPS))
    # ... and the skin never climbs higher than the tube stands above it: the wall above this
    # vertex is the tube's radius and its axis' height over the skin plus what is left of the
    # tube beyond the closest point along the way out (a root that grazes the skin from inside on
    # its way out is a hair high, not a spike of the fillet's height)
    seg_dir = np.zeros_like(centre)
    seg_dir[:-1] = np.diff(centre, axis=0)
    seg_len = np.linalg.norm(seg_dir, axis=1)
    tang = _unit(seg_dir[np.minimum(np.searchsorted(np.cumsum(seg_len[:-1]), Sa, side="right"),
                                    len(centre) - 2)])
    total_len = float(seg_len[:-1].sum())
    wall = height_c + np.maximum(rad_c, 0.0) + np.maximum(total_len - Sa, 0.0) * np.maximum(
        (tang * Na).sum(axis=1), 0.0)
    stand = _smoothstep(wall / max(0.8 * H, _EPS))
    outside = Da > 0.0
    u = np.clip(Da / W, 0.0, 1.0)
    rho = np.where(outside, W * fillet_remap(u), Da)                 # the remaining distance
    lift = np.where(outside, fillet_height(np.maximum(rho, 0.0), W, H, k), H)
    pull = np.where(outside, (Da - rho) * spec.pull, 0.0)
    wgt = sps * expose * stand
    if weight is not None:
        wgt = wgt * weight[act]
    out[act] = (-pull * wgt)[:, None] * away + (lift * wgt)[:, None] * climb
    # an inside vertex is hidden only while it STAYS inside: a tapered or bent tube can leave
    # it outside after it climbs, so bring it back to just under the wall
    moved = P[act] + out[act]
    Dn, Cn = tube_distance(moved, centre, radii)
    stray = (~outside) & (Dn > -inside_margin)
    if stray.any():
        back = moved[stray] - Cn[stray]
        back /= np.maximum(np.linalg.norm(back, axis=1, keepdims=True), _EPS)
        fix = (Dn[stray] + inside_margin)[:, None] * back
        o = out[act]
        o[stray] -= fix
        out[act] = o
    return out


def limit_silhouette(frame, P, d, owner: int, cap: float, protect=None):
    """``d`` with the outward growth of the part ``owner`` bounded: a skin point may move away from
    the bone of its part by at most ``cap`` of its distance from the bone (at least ``cap`` half
    the part's radius: a point on the axis has no distance), the whole displacement of a point
    that would move more is scaled down (its direction is kept) by a soft clip: growth ``g`` becomes
    ``g a / (g^4 + a^4)^(1/4)`` with ``a`` the bound, so it never exceeds it and is hardly changed below it. ``protect`` (0..1 per point): how much a point is left alone
    (the ones hidden in the appendage's own tube: 1 inside it, falling to 0 at its wall, so the bound does
    not step where the skin comes out of the tube)."""
    c, dist = frame.bone_closest(P, owner)
    u = (P - c) / np.maximum(dist, _EPS)[:, None]
    growth = np.maximum((d * u).sum(axis=1), 0.0)
    allowed = cap * np.maximum(dist, 0.5 * frame.part_size(owner))
    # a SOFT clip (growth g becomes g a / (g^4 + a^4)^(1/4): the same as g well below the bound, 0.84 a
    # at it, a from above): a hard cut would leave a crease in the displacement where it starts to bind,
    # and the validity pass would back the neighbours off
    scale = allowed / np.maximum((growth ** 4 + allowed ** 4) ** 0.25, _EPS)
    if protect is not None:                          # 1: left alone (hidden in the tube), 0: bound; in between: both
        w = np.asarray(protect, dtype=np.float64)
        scale = w + (1.0 - w) * scale
    return d * np.where(growth > 0.0, scale, 1.0)[:, None]


def join_angle(cage_v, quads, spec: JunctionSpec, moved_v, patch=None):
    """``(before, after, patch)``: how steeply the skin meets the wall of the tube of ``spec``,
    before (``cage_v``) and after (``moved_v``) a displacement, in degrees: over the skin vertices
    within ``JOIN_BAND`` fillet widths outside the wall of the cage subdivided ONCE (what is rendered
    and printed; the cage is a quad of 0.28 units where a root is 0.4), the mean angle between the
    skin normal and the wall's outward normal there: 90 for skin that simply runs into the tube, 0
    for skin tangent to it (:func:`measure_junction` has the same number for a whole body). The
    subdivision is that of the patch of the cage around the wall (``patch``: its quad indices, found
    when ``None`` and handed back), 2 rings out: a tenth of the cost of the whole body, the same
    numbers. ``None`` for a side where no skin vertex lies in the band."""
    centre, radii = np.asarray(spec.points), np.asarray(spec.radii)
    W = spec.fillet_width
    if patch is None:
        cols = np.ascontiguousarray(cage_v.T)
        cand = _candidates(spec, cols)
        D, _ = tube_distance(cage_v[cand], centre, radii)
        reg = np.zeros(len(cage_v), dtype=bool)
        reg[cand[D < JOIN_BAND * W + JOIN_REACH]] = True
        reg = JC.vertex_ring(quads, JC.vertex_ring(quads, reg))
        patch = np.nonzero(reg[quads].all(axis=1))[0]
    pq = quads[patch]
    used = np.unique(pq)
    remap = np.full(len(cage_v), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    (sv0, sv1), sq, exact = JC.subdivide_patch([cage_v[used], moved_v[used]], remap[pq])
    out = []
    for sv in (sv0, sv1):
        d, c = tube_distance(sv, centre, radii)
        band = (d > 0) & (d < JOIN_BAND * W) & exact
        if band.sum() < 3:
            out.append(None)
            continue
        nrm = vertex_normals(sv, sq)
        nu = sv[band] - c[band]
        nu /= np.maximum(np.linalg.norm(nu, axis=1, keepdims=True), _EPS)
        out.append(float(np.degrees(np.arccos(np.clip((nu * nrm[band]).sum(axis=1), -1.0, 1.0))).mean()))
    return out[0], out[1], patch


def _mound_displacement(m: Mound, P, N):
    c = np.asarray(m.center)
    if m.end is not None:
        d = _segment_distance(P, c, np.asarray(m.end))
    elif m.axis is not None and m.stretch != 1.0:
        a = np.asarray(m.axis)
        a = a / max(float(np.linalg.norm(a)), _EPS)
        rel = P - c
        par = rel @ a
        perp = np.linalg.norm(rel - par[:, None] * a, axis=1)
        d = np.sqrt(perp ** 2 + (par / m.stretch) ** 2)
    else:
        d = np.linalg.norm(P - c, axis=1)
    w = _bump(d / m.radius)
    direction = N if m.direction is None else np.broadcast_to(
        np.asarray(m.direction) / max(float(np.linalg.norm(m.direction)), _EPS), N.shape)
    return (m.height * w)[:, None] * direction


def _keep_weight(spec: JunctionSpec, P) -> np.ndarray:
    w = np.ones(len(P))
    for k in spec.keep:
        d = np.linalg.norm(P - np.asarray(k.center), axis=1)
        w = np.minimum(w, _smoothstep((d - k.radius) / max(k.fade * k.radius, _EPS)))
    return w


# the fillet is capped by the size of the body part it lies on (a head is not a 1.2-unit wide
# plateau); a root thicker than the part is no root the skin can flow into at all
WIDTH_CAP = 0.6              # W <= this many radii of the part the root belongs to
FIT_MIN = 0.02               # a root this close to "too thick" is left out altogether (no 1,000 vertices moved by a hair)
FIT_FULL, FIT_NONE = 0.7, 1.1     # root radius / part radius: moulded in full up to the first, not at all from the second


def resolve_spec(spec: JunctionSpec, frame=None) -> JunctionSpec:
    """``spec`` with every relative mound (``Mound.scale``) turned into absolute lengths on
    the body ``frame`` describes (without a frame the numbers stay as they are)."""
    if frame is None or not any(m.scale for m in spec.mounds):
        return spec
    out = []
    for m in spec.mounds:
        if m.scale:
            unit = frame.size(m.scale, near=m.center)
            m = replace(m, radius=m.radius * unit, height=m.height * unit, scale=None)
        out.append(m)
    return replace(spec, mounds=tuple(out))


def junction_displacement(spec: JunctionSpec, cage_v, normals, candidates=None, frame=None,
                          quads=None, smooth=0):
    """``(indices, displacement)``: how ``spec`` moves the cage vertices it touches, before
    the global limits. Pure. ``frame``: the body's :class:`~.bodyframe.BodyFrame` (optional):
    with it the spec only moulds the skin of the part of the body its root lies in (the
    original arm's armpit is left alone by an extra arm), the fillet is capped by that part's
    size and mounds with a ``scale`` are resolved. ``quads`` (the cage) with ``smooth`` > 0:
    the tube's displacement is smoothed over the cage that many rounds
    (:func:`diffuse_displacement`; mounds, being smooth already, are not)."""
    spec = resolve_spec(spec, frame)
    cand = _candidates(spec, np.ascontiguousarray(cage_v.T)) if candidates is None else candidates
    if not len(cand) or not spec.active:
        return cand[:0], np.zeros((0, 3))
    if not spec.ownership:                    # this spec straddles parts on purpose: no skeleton for it
        frame = None
    P, N = cage_v[cand], normals[cand]
    d = np.zeros_like(P)
    owner_w = None
    owner = None
    W = spec.fillet_width
    fit = 1.0
    exit_info = None
    if spec.tube and W > 0:
        exit_info = tube_emergence(spec, P, N)
        if frame is not None:
            at = exit_info[1] if exit_info is not None else np.asarray(spec.points)[0]
            near = np.linalg.norm(P - at, axis=1)
            contact = P[near < 1.5 * (exit_info[2] if exit_info is not None else max(spec.radii)) + W]
            owner = frame.resolve_owner(spec.owner, contact if len(contact) else at[None, :])
            owner_w = frame.owner_weights(P, owner)
            unit = frame.part_size(owner)
            W = min(W, WIDTH_CAP * unit)
            if exit_info is not None:
                ratio = exit_info[2] / max(unit, _EPS)
                fit = float(_smoothstep((FIT_NONE - ratio) / (FIT_NONE - FIT_FULL)))
        if exit_info is not None and W > 0 and fit > FIT_MIN:
            dt = fit * _tube_displacement(spec, P, N, inside_margin=0.04 * min(spec.radii),
                                          W=W, weight=owner_w)
            if smooth > 0 and quads is not None and (dt != 0).any():
                full = np.zeros_like(cage_v)
                full[cand] = dt
                hidden = np.zeros(len(cage_v), dtype=bool)
                hidden[cand] = tube_distance(P, np.asarray(spec.points), np.asarray(spec.radii))[0] < 0.0
                dt = diffuse_displacement(cage_v, quads, full, hidden, smooth)[cand]
            d += dt
    for m in spec.mounds:
        w = None
        if frame is not None:
            if owner is None:
                owner = frame.resolve_owner(spec.owner, np.asarray(m.center)[None, :])
                owner_w = frame.owner_weights(P, owner)
            w = owner_w
        dm = _mound_displacement(m, P, N)
        d += dm if w is None else dm * w[:, None]
    d *= (spec.strength * _keep_weight(spec, P))[:, None]
    if frame is not None and owner is not None:
        cap = spec.silhouette if spec.silhouette is not None else PART_SILHOUETTE.get(frame.parts[owner], 0.0)
        if cap > 0 and (d != 0).any():
            hidden = None
            if spec.tube:                       # inside the tube: hidden, left alone; the bound comes in at its wall
                depth = -tube_distance(P, np.asarray(spec.points), np.asarray(spec.radii))[0]
                hidden = _smoothstep(depth / max(0.5 * spec.reference_radius, _EPS))
            d = limit_silhouette(frame, P, d, owner, cap, hidden)
    limit = spec.move_limit()
    mag = np.linalg.norm(d, axis=1)
    over = mag > limit
    if over.any():
        d[over] *= (limit / mag[over])[:, None]
    return cand, d


# --------------------------------------------------------------------------
# validity: no flipped, collapsed or stretched quad, no new self-intersection


def _quad_vectors(v, q):
    p = v[q]
    return _cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])            # twice the area vector


def _touching(quads, vertex_mask) -> np.ndarray:
    return np.nonzero(vertex_mask[quads].any(axis=1))[0]


def facing_pairs(cage_v, quads, normals, moved, reach: float, min_dot: float = -0.2):
    """Pairs ``(v, u, distance)`` of cage vertices that FACE each other across a gap: ``v`` is
    one of ``moved`` (indices), ``u`` any vertex within ``reach`` whose normal points back at
    it (``n_v . n_u < min_dot``) and which is not a neighbour of ``v`` (sharing a quad). That is
    the two cheeks of a cleft, the arm and the flank at an armpit: where a skin that moves
    toward the opposite skin must keep its gap."""
    moved = np.asarray(moved)
    if not len(moved):
        z = np.zeros(0, dtype=np.int64)
        return z, z, np.zeros(0)
    lo, hi = cage_v[moved].min(axis=0) - reach, cage_v[moved].max(axis=0) + reach
    near = np.nonzero(np.all((cage_v >= lo) & (cage_v <= hi), axis=1))[0]
    P = cage_v[near]
    where = np.full(len(cage_v), -1, dtype=np.int64)
    where[near] = np.arange(len(near))
    pr = JC.overlapping_pairs(P, P, where[moved], None, grow=0.5 * reach)
    if not len(pr):
        z = np.zeros(0, dtype=np.int64)
        return z, z, np.zeros(0)
    v, u = near[pr[:, 0]], near[pr[:, 1]]
    dist = np.linalg.norm(cage_v[v] - cage_v[u], axis=1)
    keep = (dist < reach) & ((normals[v] * normals[u]).sum(axis=1) < min_dot)
    v, u, dist = v[keep], u[keep], dist[keep]
    if len(v):                                   # not neighbours: no quad holds both
        far = ~JC.share_a_quad(quads, v, u)
        v, u, dist = v[far], u[far], dist[far]
    return v, u, dist


def limit_displacement(cage_v, quads, disp, min_area_ratio=MIN_AREA_RATIO,
                       min_normal_dot=MIN_NORMAL_DOT, iterations=LIMIT_ITERATIONS, *,
                       max_area_growth=None, max_edge_growth=None, max_lipschitz=None,
                       intersections=False, benign=None):
    """Back off the displacement of the vertices of any quad that would flip (its normal
    turns past ``min_normal_dot``) or collapse (its area falls below ``min_area_ratio`` of
    its own); ``(disp, factor, rounds)`` with ``factor`` the per-vertex scale applied.

    The S1b guards are opt-in here (``apply_junctions`` turns them on): ``max_area_growth``
    (a quad's area may not grow past this multiple of its own), ``max_edge_growth`` (nor an
    edge's length), ``max_lipschitz`` (two neighbours' displacements may differ by at most
    this many of their edge's lengths: 1 lets an edge double), and ``intersections`` (no pair
    of triangles may NEW-ly cross each other: the skin is not allowed to pierce the opposite
    skin of a cleft or an armpit; a pair whose two triangles are both ``benign(points)`` - for
    instance inside the appendage's own tube, where the slicer's union hides them - is
    allowed)."""
    moved = np.linalg.norm(disp, axis=1) > 0
    factor = np.ones(len(disp))
    if not moved.any():
        return disp, factor, 0
    qi = _touching(quads, moved)
    tq = quads[qi]
    old = _quad_vectors(cage_v, tq)
    old_len = np.linalg.norm(old, axis=1)
    edges = None
    if max_edge_growth is not None or max_lipschitz is not None:
        edges = JC.cage_edges(tq)
        edges = edges[moved[edges].any(axis=1)]
        e_old = np.maximum(np.linalg.norm(cage_v[edges[:, 0]] - cage_v[edges[:, 1]], axis=1), _EPS)

    def violators(f):
        """The vertices that must back off for the local limits (quad flip / collapse /
        growth, edge growth, Lipschitz) with the scale ``f``: a boolean mask."""
        d = disp * f[:, None]
        new_v = cage_v + d
        new = _quad_vectors(new_v, tq)
        new_len = np.linalg.norm(new, axis=1)
        dot = (new * old).sum(axis=1) / np.maximum(new_len * old_len, _EPS)
        bad = (dot < min_normal_dot) | (new_len < min_area_ratio * old_len)
        if max_area_growth is not None:
            bad |= new_len > max_area_growth * old_len
        hit = np.zeros(len(disp), dtype=bool)
        hit[tq[bad].ravel()] = True
        if edges is not None:
            if max_edge_growth is not None:
                e_new = np.linalg.norm(new_v[edges[:, 0]] - new_v[edges[:, 1]], axis=1)
                hit[edges[e_new > max_edge_growth * e_old].ravel()] = True
            if max_lipschitz is not None:
                lip = np.linalg.norm(d[edges[:, 0]] - d[edges[:, 1]], axis=1) / e_old
                hit[edges[lip > max_lipschitz].ravel()] = True
        return hit & moved & (f > 0)

    tester = None
    if intersections:
        tris = JC.triangulate(quads)
        mag0 = np.linalg.norm(disp, axis=1)
        # a pair of triangles can only cross anew if one of them moves: the vertices that
        # move by more than a hair are the ones to look at (the fringe moves too little to
        # close a gap that was not already closed)
        tester = JC.IntersectionTester(cage_v, quads, mag0 > INTERSECTION_MIN_MOVE,
                                       float(mag0.max()))

    def crossings(f):
        """The vertices of the triangles that NEWLY cross each other with the scale ``f``."""
        new_v = cage_v + disp * f[:, None]
        pairs = tester.pairs(new_v)
        if len(pairs):                                # crossing already before the move: not ours
            pairs = pairs[~JC.pairs_cross(cage_v, tris, pairs)]
        if len(pairs) and benign is not None:
            cen = new_v[tris[pairs]].mean(axis=2)                           # (k, 2, 3)
            pairs = pairs[~benign(cen.reshape(-1, 3)).reshape(-1, 2).all(axis=1)]
        hit = np.zeros(len(disp), dtype=bool)
        if len(pairs):
            hit[tris[pairs].ravel()] = True
        return hit & moved & (f > 0)

    rounds = 0
    for _outer in range(8 if intersections else 1):
        for _ in range(iterations):                    # the local limits: cheap, run to the end
            hit = violators(factor)
            if not hit.any():
                break
            rounds += 1
            factor[hit] *= LIMIT_BACKOFF
            factor[factor < 1e-3] = 0.0
        if tester is None:
            break
        hit = crossings(factor)
        if not hit.any():
            break
        rounds += 1
        # a crossing is a gap closed: back the whole neighbourhood off, not one vertex
        near = JC.vertex_ring(quads, hit)
        factor[near & moved & (factor > 0)] *= 0.5
        factor[factor < 1e-3] = 0.0
    else:
        factor[crossings(factor)] = 0.0                # out of rounds: leave those vertices alone
    return disp * factor[:, None], factor, rounds


def diffuse_displacement(cage_v, quads, disp, pinned=None, iterations=DIFFUSE_ITERATIONS):
    """Smooth the displacement over the cage: the part of a harmonic extension of the skin's
    contact with the tube.

    The raw displacement of a spec is a smooth function of position, but it is defined through
    skin normals, which jump across a crease (the gluteal cleft, an armpit) where the cage is
    also much denser (edges of 0.08 next to 0.3): two neighbours then move by very different
    amounts per edge length (3-4x). The vertices hidden inside the tube (``pinned``) keep the
    displacement they were given, a vertex that did not move stays where it is (the rim of the
    zone), and every other moved vertex takes, ``iterations`` times, the average of its
    neighbours' displacement weighted by 1 / edge length (Jacobi: order independent). That is
    the fillet's own shape on a cage of any local density: the rise spreads over as many rings
    as the cage has between the wall and the rim instead of being chopped to what the densest
    edge allows. Returns the smoothed displacement (a copy)."""
    d = np.array(disp, dtype=np.float64, copy=True)
    moved = (d != 0).any(axis=1)
    if not moved.any() or iterations <= 0:
        return d
    e = JC.cage_edges(quads[_touching(quads, moved)])
    used = np.unique(e)                                   # work on the patch, not the whole body
    loc = np.searchsorted(used, e)
    a = np.concatenate([loc[:, 0], loc[:, 1]])
    b = np.concatenate([loc[:, 1], loc[:, 0]])
    pos = cage_v[used]
    w = 1.0 / np.maximum(_norm(pos[a] - pos[b]), 1e-6)
    n = len(used)
    free = moved[used] if pinned is None else (moved & ~np.asarray(pinned))[used]
    wsum = np.maximum(np.bincount(a, weights=w, minlength=n), _EPS)
    dl = d[used]
    for _ in range(iterations):
        num = np.stack([np.bincount(a, weights=w * dl[b, c], minlength=n) for c in range(3)], axis=1)
        dl[free] = num[free] / wsum[free, None]
    d[used] = dl
    return d


# --------------------------------------------------------------------------
# the public function


def apply_junctions(base, verts, specs, info: dict | None = None, *,
                    min_area_ratio=MIN_AREA_RATIO, min_normal_dot=MIN_NORMAL_DOT,
                    max_area_growth=MAX_AREA_GROWTH, max_edge_growth=MAX_EDGE_GROWTH,
                    max_lipschitz=MAX_LIPSCHITZ, intersections=True, smooth=DIFFUSE_ITERATIONS,
                    tight_gap=TIGHT_GAP, frame=None, drop_worse=JOIN_WORSE):
    """The body vertices ``verts`` (19,158 x 3, the MORPHED control vertices) with the skin
    moulded around every :class:`JunctionSpec` of ``specs``.

    Pure and deterministic; topology and vertex numbering are untouched (only body cage
    vertices move; eyes, teeth and joint markers are not); returns ``verts`` itself when
    there is no active spec. Each spec is computed from the INPUT body and the displacements
    are summed (so the result does not depend on the order of the specs, and a mirrored pair
    gives a mirror-symmetric result on a symmetric body), then capped and checked for
    flipped or collapsed quads. Never feed the result back in as ``verts`` (it would
    be a second junction on the first); ``info`` (a dict, optional) receives ``moved``,
    ``max_move``, ``limited`` (vertices backed off), ``rounds``, ``specs``, ``dropped`` (the names
    of the specs that were left out because they made the join worse, see ``drop_worse``),
    ``tube_only`` / ``mounds_only`` (the specs that were applied in that form only for that reason),
    ``conformed`` (the specs whose tube was cut down to its mesh) and ``join`` (``{name: (degrees
    before, after)}`` of the specs with a tube, as computed first).

    S1b keywords (all backward compatible: a call that passes none of them gets the guarded
    behaviour; pass ``None`` / 0 to switch one off): ``max_area_growth`` / ``max_edge_growth``
    (a cage quad's area / an edge's length may not grow past this multiple of its own, 2),
    ``max_lipschitz`` (two neighbours may not move more than this many edge lengths apart,
    1.25), ``intersections`` (no pair of triangles may newly cross each other outside the
    appendage's own tube), ``smooth`` (rounds of :func:`diffuse_displacement` of each spec's
    tube displacement, 20), ``tight_gap`` (skin within two rings of skin it nearly touches,
    closer than this many units, 0.04, is not moved), ``frame`` (the body's
    :class:`~.bodyframe.BodyFrame`; ``None``: built from ``base``, cached; ``False``: none, so
    no owner protection and no relative sizes). ``min_area_ratio`` / ``min_normal_dot`` as
    before.

    S6: ``drop_worse`` (degrees, 2; ``None`` switches it off): a junction that makes the join with its
    root WORSE than no junction at all (:func:`join_angle`, on the cage, over the skin next to
    the wall, with the displacement as it comes out of the validity pass) is not applied: the guards
    can leave a displacement that only crumples the skin (smoothing and the cleft hold next to a
    closed cleft did: the heavy figure's tail joined at 94 degrees before and 102 after). The
    spec is tried with its tube only (``info["tube_only"]``: its mounds, which are mass and move the
    skin by the wall too, are left out), then with its mounds only (``info["mounds_only"]``), and a
    spec that is worse in both forms is left out whole (``info["dropped"]``); a spec with no mounds is
    left out whole at once. A spec's own check is made on its own displacement as it comes out of the
    validity pass; the sum of the other specs is not re-checked."""
    specs = [s for s in (specs or ()) if s is not None and s.active]
    if info is not None:
        info.update(moved=0, max_move=0.0, limited=0, rounds=0, specs=len(specs), dropped=[],
                    mounds_only=[], tube_only=[], conformed=[], join={})
    if not specs:
        return verts
    if frame is False:
        frame = None
    elif frame is None:
        frame = frame_or_none(base, verts)
    specs = [resolve_spec(s, frame) for s in specs]
    conformed = [conform_spec(s) for s in specs]
    if info is not None:
        info["conformed"] = [s.name for s, c in zip(specs, conformed) if c is not s]
    specs = [c for c in conformed if c is not None and c.active]
    if not specs:
        return verts
    cage_v, cage_q = base.body_cage(verts)
    normals = smooth_normals(cage_v, cage_q, vertex_normals(cage_v, cage_q), NORMAL_SMOOTHING)
    cols = np.ascontiguousarray(cage_v.T)
    cands = [_candidates(spec, cols) for spec in specs]
    gaps = [tight_gap if s.tight_gap is None else s.tight_gap for s in specs]
    tight_ws = {}
    for gap in sorted({g for g in gaps if g and g > 0}):
        # where two sheets of skin touch (the closed cleft: 0.02 apart) the subdivided surface
        # crosses itself at the slightest move of the cage around it, though the cage does not:
        # the skin within two rings of such a place is left where it is (and fades back in)
        region = np.zeros(len(cage_v), dtype=bool)
        for cand, g in zip(cands, gaps):
            if g == gap:
                region[cand] = True
        rows = np.nonzero(region)[0]
        tv, tu, _td = facing_pairs(cage_v, cage_q, normals, rows, gap)
        if len(tv):
            tight = np.zeros(len(cage_v), dtype=bool)
            tight[tv] = True
            tight[tu] = True
            tight_ws[gap] = _smoothstep((JC.ring_distance(cage_q, tight, 4) - 2.0) / 2.0)

    def weighted(k, spec, idx, d):
        gap = gaps[k]
        tight_w = tight_ws.get(gap) if gap else None
        return d if tight_w is None else d * tight_w[idx][:, None]

    parts = []                               # (spec, indices, displacement, the spec whose join is checked)
    for k, (spec, cand) in enumerate(zip(specs, cands)):
        idx, d = junction_displacement(spec, cage_v, normals, cand, frame, quads=cage_q,
                                       smooth=smooth if spec.smooth is None else spec.smooth)
        if len(idx):
            parts.append((spec, idx, weighted(k, spec, idx, d), spec, k))

    def assemble(chosen):
        total = np.zeros_like(cage_v)
        limit = 0.0
        for spec, idx, d, _join, _k in chosen:
            total[idx] += d
            limit = max(limit, spec.move_limit())
        mag = np.linalg.norm(total, axis=1)
        if limit > 0:
            over = mag > limit
            if over.any():                               # two overlapping specs: one cap on the sum
                total[over] *= (limit / mag[over])[:, None]
        return total

    def validity(total, chosen):
        tubes = [(np.asarray(p[0].points), np.asarray(p[0].radii)) for p in chosen if p[0].tube]

        def hidden(points):                  # inside an appendage's own tube: the slicer's union hides it
            d = np.full(len(points), np.inf)
            for c, r in tubes:
                d = np.minimum(d, tube_distance(points, c, r)[0])
            return d < 0.0

        return limit_displacement(
            cage_v, cage_q, total, min_area_ratio, min_normal_dot,
            max_area_growth=max_area_growth, max_edge_growth=max_edge_growth,
            max_lipschitz=max_lipschitz, intersections=intersections, benign=hidden)

    def worse(chosen, factor):
        """The positions in ``chosen`` of the specs with a tube whose join got worse, and every
        such spec's ``(before, after)``."""
        bad, seen = [], {}
        for n, (_spec, idx, d, join_spec, _k) in enumerate(chosen):
            if not (join_spec.tube and join_spec.fillet_width > 0 and join_spec.join_guard):
                continue
            eff = np.zeros_like(cage_v)
            eff[idx] = d * factor[idx][:, None]
            before, after, _ = join_angle(cage_v, cage_q, join_spec, cage_v + eff)
            if before is None or after is None:
                continue
            seen[n] = (before, after)
            if after > before + drop_worse:
                bad.append(n)
        return bad, seen

    total = assemble(parts)
    if not (np.linalg.norm(total, axis=1) > 0).any():
        return verts
    total, factor, rounds = validity(total, parts)
    joins, dropped, mounds_only, tube_only = {}, [], [], []
    if drop_worse is not None:
        bad, seen = worse(parts, factor)
        joins = {parts[n][3].name: v for n, v in seen.items()}
        if bad:
            # The join is the seam of the TUBE with the skin, but a spec's mounds (a sacral rise, a pec
            # fold) are mass that moves the skin by the wall too: look at which of the two is at fault.
            # First the spec with its tube only (the seam is what the guard is about), then with its
            # mounds only; one that is worse in both forms is left out whole.
            final = [p for n, p in enumerate(parts) if n not in bad]
            pending = [parts[n] for n in bad]
            for form in ("tube", "mounds"):
                trial = []
                for spec, _i, _d, join_spec, k in pending:
                    if not spec.mounds:
                        continue                                  # no other form of it to try
                    sub = spec.with_(mounds=()) if form == "tube" else spec.with_(points=(), radii=(), base_radius=None)
                    if not sub.active:
                        continue
                    i2, d2 = junction_displacement(sub, cage_v, normals, cands[k], frame, quads=cage_q,
                                                   smooth=smooth if spec.smooth is None else spec.smooth)
                    if len(i2):
                        trial.append((sub, i2, weighted(k, spec, i2, d2), join_spec, k))
                if not trial:
                    continue
                every = final + trial
                t_total = assemble(every)
                if not (np.linalg.norm(t_total, axis=1) > 0).any():
                    continue
                t_total, t_factor, _r = validity(t_total, every)
                bad_t, _ = worse(every, t_factor)
                ok = [p for n, p in enumerate(trial) if len(final) + n not in bad_t]
                final += ok
                took = {p[4] for p in ok}
                (tube_only if form == "tube" else mounds_only).extend(p[3].name for p in ok)
                pending = [p for p in pending if p[4] not in took]
            dropped = [p[3].name for p in pending]
            parts = final
            total = assemble(parts)
            if not (np.linalg.norm(total, axis=1) > 0).any():
                if info is not None:
                    info.update(dropped=dropped, mounds_only=mounds_only, tube_only=tube_only, join=joins)
                return verts
            total, factor, rounds = validity(total, parts)
    out = np.array(verts, dtype=np.float64, copy=True)
    out[base.body_verts_idx] = cage_v + total
    if info is not None:
        mag = np.linalg.norm(total, axis=1)
        info.update(moved=int((mag > 0).sum()), max_move=float(mag.max()),
                    limited=int((factor < 1.0).sum()), rounds=int(rounds), dropped=dropped,
                    mounds_only=mounds_only, tube_only=tube_only, join=joins)
    return out


# --------------------------------------------------------------------------
# building a spec from the geometry of an appendage root

HUG = 0.97        # the skin hugs this fraction of the (inscribed) tube radius: just inside the wall


def ring_tube(vertices, ring_verts: int, rings: int, hug: float = HUG):
    """``(centres, radii)`` of the first ``rings`` rings of a loft (``appendages._loft``
    stacks its rings ring-major: ring ``i`` is rows ``i * ring_verts ...``).

    The radius is the SMALLEST distance of a ring vertex to its centre times the cosine of
    half the facet angle (the inscribed circle of the polygon) times ``hug``: the skin
    meets the tube just inside every facet, so the seam between the two meshes is a clean
    curve and not a zigzag between the facets."""
    ring = np.asarray(vertices, dtype=np.float64)[:rings * ring_verts].reshape(
        rings, ring_verts, 3)
    c = ring.mean(axis=1)
    r = np.linalg.norm(ring - c[:, None], axis=2).min(axis=1)
    return c, r * math.cos(math.pi / ring_verts) * hug


def fit_radii_to_mesh(points, radii, vertices, hug: float = HUG, percentile: float = 20.0,
                      reach: float = 1.5):
    """``radii`` capped by the radius of the REAL appendage mesh ``vertices`` around the
    centreline ``points`` (the skin that "is inside the tube" must really be hidden by the
    appendage: a spec that is fatter than the mesh leaves the skin that climbed with it sticking
    out of a thin root, a limb tapered to fit the torso in particular).

    At each point the mesh vertices within half a segment along the axis are looked at; their
    distance to the axis, at the ``percentile`` (an elliptical or lumpy section: its inscribed
    radius, not its mean), times ``hug`` is the radius there. Where the mesh has no vertices at all
    (the sunk end of a cut limb, a centreline that runs on past the shell) there is no tube: the
    radius there is a wire (``FIT_FLOOR`` of the given one), not a radius extrapolated from the
    nearest point that has a mesh, which stood for skin that nothing covers (S1b: all 68 skin vertices
    the extra upper arms' phantom tube "hid" were outside the arm). Never larger than the given
    radius. Returns a tuple. (:func:`~.junction_checks.fit_tube_to_mesh`, which the junction
    applies through ``JunctionSpec.shell``, uses the shell itself.)"""
    pts = np.asarray(points, dtype=np.float64)
    rad = np.asarray(radii, dtype=np.float64)
    V = np.asarray(vertices, dtype=np.float64)
    n = len(pts)
    got = np.full(n, np.nan)
    for i in range(n):
        a, b = pts[max(i - 1, 0)], pts[min(i + 1, n - 1)]
        t = b - a
        ln = float(np.linalg.norm(t))
        if ln < _EPS:
            continue
        t /= ln
        half = 0.5 * max(float(np.linalg.norm(pts[i] - a)), float(np.linalg.norm(b - pts[i])))
        rel = V - pts[i]
        along = rel @ t
        perp = np.linalg.norm(rel - along[:, None] * t, axis=1)
        sel = (np.abs(along) <= half) & (perp < reach * max(rad[i], _EPS) + 0.5)
        if sel.sum() >= 6:
            got[i] = float(np.percentile(perp[sel], percentile)) * hug
    got = np.where(np.isfinite(got), got, JC.FIT_FLOOR * rad)      # no mesh here: no tube, a wire
    return tuple(float(r) for r in np.minimum(rad, got))


def conform_spec(spec: JunctionSpec) -> JunctionSpec | None:
    """``spec`` with its tube cut down to what lies inside the mesh it stands for (``spec.shell``,
    :func:`~.junction_checks.fit_tube_to_mesh`); the spec itself when it has no shell or its tube
    already fits. The skin that counts as HIDDEN in the tube must really be hidden by the
    appendage: a centreline point outside the mesh (the sunk end of a cut limb, a limb posed away
    from the model the spec was measured on) or a radius fatter than the shell tents skin that
    nothing covers (S1b: 68 of 68 skin vertices of the extra upper arms that the spec called hidden
    were outside the arm). A tube that is cut keeps its fillet width (``base_radius`` is pinned).
    ``None`` for a spec with nothing left of its tube and no mounds."""
    if not spec.tube or spec.shell is None or not spec.conform:
        return spec
    try:
        pts, rad = spec.shell.fit(spec)
    except (ValueError, IndexError, TypeError):          # the mesh cannot be built: leave the tube
        return spec
    if len(pts) == len(spec.points) and np.array_equal(rad, np.asarray(spec.radii)):
        return spec
    base = spec.reference_radius
    if len(pts) < 2:
        return spec.with_(points=(), radii=(), base_radius=None) if spec.mounds else None
    return spec.with_(points=tuple(map(tuple, pts)), radii=tuple(float(r) for r in rad), base_radius=base)


def rings_for_length(centres, length: float, minimum: int = 3) -> int:
    """How many leading rings reach ``length`` along the centreline (at least ``minimum``)."""
    seg = np.linalg.norm(np.diff(np.asarray(centres), axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    k = int(np.searchsorted(cum, length)) + 1
    return int(min(max(k, minimum), len(cum)))


# --------------------------------------------------------------------------
# the pipeline seam


def junction_specs(placed) -> list:
    """The junction specs of built appendages (:class:`~.kinds.Placed`): one per appendage
    that has one (``Placed.junction()``), in order. An appendage whose junction cannot be
    worked out is skipped with a warning (:class:`~.kinds.AppendageWarning`): the skin is
    left as it is there, the picture and the print still come out."""
    out = []
    for p in placed or ():
        j = getattr(p, "junction", None)
        if not callable(j):
            continue
        try:
            spec = j()
        except (ValueError, KeyError, IndexError, TypeError, np.linalg.LinAlgError) as exc:
            import warnings

            from .kinds import AppendageWarning
            warnings.warn(f"appendage {getattr(p, 'name', '?')!r}: no skin junction ({exc})",
                          AppendageWarning, stacklevel=2)
            continue
        if spec is not None:
            out.append(spec)
    return out


def moulded_cage(base, verts, specs, **kw):
    """``(vertices, quads)`` of the body cage with the skin moulded around ``specs`` exactly as
    :func:`apply_junctions` gives it (the same displacement, the same guards and the same validity
    pass; ``kw`` are its keywords). For a part that is built against the skin the body will really
    have (a horn's collar and foot), not against a law that the pipeline does not apply."""
    cage_v, cage_q = base.body_cage(verts)
    specs = [specs] if isinstance(specs, JunctionSpec) else list(specs or ())
    out = apply_junctions(base, verts, specs, **kw)
    return (cage_v if out is verts else base.body_cage(out)[0]), cage_q


def shaped_body(base, verts, placed, enabled: bool = True, info: dict | None = None):
    """The body the character is MESHED, RENDERED and PRINTED with: the morphed (and
    anatomy-adapted) ``verts`` with the skin moulded around every appendage in ``placed``.

    ``verts`` must be the UNDEFORMED morphed vertices the appendages were placed on (they
    stay the frame of all placement); the result is only for the body mesh. ``enabled=False``
    (a document's ``anatomy["junction"] = False``) and an empty ``placed`` return ``verts``
    itself."""
    if not enabled or not placed:
        return verts
    return apply_junctions(base, verts, junction_specs(placed), info)


# --------------------------------------------------------------------------
# measuring a junction (tests, tools/junction_report.py)


def _quad_dihedrals(v, q) -> np.ndarray:
    """The angle (degrees) between the normals of every pair of quads sharing an edge."""
    n = _quad_vectors(v, q)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), _EPS)
    a = q
    b = np.roll(q, -1, axis=1)
    key = np.sort(np.stack([a.ravel(), b.ravel()], axis=1), axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    ks = key[order]
    same = np.all(ks[1:] == ks[:-1], axis=1)
    i0 = order[:-1][same] // 4
    i1 = order[1:][same] // 4
    cosv = np.clip((n[i0] * n[i1]).sum(axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cosv)), i0, i1


def measure_junction(base, verts, shaped, spec: JunctionSpec, levels: int = 1) -> dict:
    """Numbers that say how smooth a junction is, on the body subdivided ``levels`` times
    (``verts``: the body before, ``shaped``: after).

    * ``min_area_ratio`` / ``flipped``: per CAGE quad, new area / old area (smallest over the
      quads the junction touches) and how many turned their normal past 78 degrees;
    * ``join_angle_deg_before`` / ``_after``: the angle between the skin and the tube wall
      where they meet (the mean, over the skin vertices within a quarter of the fillet width
      of the wall, of the angle between the skin normal and the wall normal: 90 for a skin
      that simply runs into the tube, 0 for one that is tangent to it);
    * ``p95_dihedral_deg_*`` / ``max_dihedral_deg_*``: the crease between neighbouring quads
      of the SUBDIVIDED skin in the fillet zone (outside the tube), before and after. The
      body's own creases (the gluteal cleft, an armpit) are in both;
    * ``dihedral_increase_p99_deg`` / ``_max_deg``: how much the junction INCREASED the crease
      of the same edge (99th percentile / worst): what it added, without the body's own;
    * ``curvature_step_*``: how much the skin's curvature jumps across the fillet boundary
      (``rho = W``), from the umbrella operator on the subdivided body: the difference of the
      median curvature just inside and just outside over the median magnitude inside; the body's
      own curvature varies too, so read the "before" next to the "after".
    """
    from ..core.subdiv import subdivide_quads
    spec = conform_spec(spec) or spec           # the tube the junction really used
    cage_v0, cage_q = base.body_cage(verts)
    cage_v1, _ = base.body_cage(shaped)
    W = spec.fillet_width
    centre = np.asarray(spec.points)
    radii = np.asarray(spec.radii)
    out: dict = {"width": W}
    qi = _touching(cage_q, np.linalg.norm(cage_v1 - cage_v0, axis=1) > 0)
    if len(qi):
        old, new = _quad_vectors(cage_v0, cage_q[qi]), _quad_vectors(cage_v1, cage_q[qi])
        ol, nl = np.linalg.norm(old, axis=1), np.linalg.norm(new, axis=1)
        out["min_area_ratio"] = float((nl / np.maximum(ol, _EPS)).min())
        out["flipped"] = int((((new * old).sum(axis=1) / np.maximum(ol * nl, _EPS)) < 0.2).sum())
    else:
        out["min_area_ratio"], out["flipped"] = 1.0, 0
    sub = []
    for tag, v in (("before", cage_v0), ("after", cage_v1)):
        sv, sq = subdivide_quads(v, cage_q, levels) if levels else (v, cage_q)
        d, c = tube_distance(sv, centre, radii)
        nrm = vertex_normals(sv, sq)
        band = (d > 0) & (d < 0.25 * W)
        if band.any():
            nu = sv[band] - c[band]
            nu /= np.maximum(np.linalg.norm(nu, axis=1, keepdims=True), _EPS)
            out[f"join_angle_deg_{tag}"] = float(np.degrees(np.arccos(np.clip(
                (nu * nrm[band]).sum(axis=1), -1.0, 1.0))).mean())
        dih, i0, i1 = _quad_dihedrals(sv, sq)
        sub.append((dih, i0, i1, d, sq))
        out[f"_kappa_{tag}"] = _curvature_profile(sv, sq, nrm, d, W)
    dih_b = sub[0][0]
    dih_a, i0, i1, d, sq = sub[1]
    dq = d[sq].min(axis=1)
    zone = (dq > -0.05) & (dq < 1.5 * W)
    sel = zone[i0] & zone[i1]
    if sel.any():
        out["max_dihedral_deg_before"] = float(dih_b[sel].max())
        out["max_dihedral_deg_after"] = float(dih_a[sel].max())
        out["p95_dihedral_deg_before"] = float(np.percentile(dih_b[sel], 95))
        out["p95_dihedral_deg_after"] = float(np.percentile(dih_a[sel], 95))
        inc = (dih_a - dih_b)[sel]
        out["dihedral_increase_p99_deg"] = float(np.percentile(inc, 99))
        out["dihedral_increase_max_deg"] = float(inc.max())
    for tag in ("before", "after"):
        out[f"curvature_step_{tag}"] = _curvature_step(out.pop(f"_kappa_{tag}"), W)
    return out


def junction_quality(base, verts, shaped, specs=None, levels=(), frame=None,
                     intersections: bool = True) -> dict:
    """What ``shaped`` (the body after :func:`apply_junctions`) did to the skin of ``verts``,
    as numbers a reviewer or a test can bound. Everything is measured on the cage over the
    quads and edges the junction touched, and (``levels``) on the Catmull-Clark subdivided
    body, which is what is rendered and printed:

    * ``moved``, ``max_move`` and ``flipped`` (cage quads whose normal turned past the limit);
    * stretch (:func:`~.junction_checks.stretch_report`): ``area_growth_max`` /
      ``area_ratio_min`` (new / old area of a quad), ``edge_growth_max`` / ``edge_ratio_min``,
      ``lipschitz_max`` / ``lipschitz_p99`` (how differently two neighbours moved, per edge
      length: 1 lets the edge double);
    * crossings (``intersections``): ``selfx_before`` / ``selfx_after`` the pairs of triangles
      of the cage that cross each other, ``selfx_new`` the pairs that cross only after, and
      ``selfx_new_visible`` those not hidden inside an appendage's own tube (``specs``): the skin
      piercing the opposite skin of a cleft or an armpit. With ``levels`` the same per level
      (``L1_selfx_new`` ...) on the subdivided body;
    * ``by_part`` (a skeleton ``frame``, built from ``base`` when ``None``): for each part of the
      body (:mod:`.bodyframe`) ``(vertices moved, largest move)``, to see that a root only
      moulds the skin of the part it belongs to.

    :func:`quality_violations` turns it into a list of broken bounds."""
    from ..core.subdiv import subdivide_quads
    specs = [c for c in (conform_spec(sp) for sp in (specs or ()) if sp is not None) if c is not None]
    cage_v0, cage_q = base.body_cage(verts)
    cage_v1, _ = base.body_cage(shaped)
    out = JC.stretch_report(cage_v0, cage_v1, cage_q)
    moved = np.linalg.norm(cage_v1 - cage_v0, axis=1) > 0
    qi = _touching(cage_q, moved)
    if len(qi):
        old, new = _quad_vectors(cage_v0, cage_q[qi]), _quad_vectors(cage_v1, cage_q[qi])
        ol, nl = np.linalg.norm(old, axis=1), np.linalg.norm(new, axis=1)
        out["flipped"] = int((((new * old).sum(axis=1) / np.maximum(ol * nl, _EPS))
                              < MIN_NORMAL_DOT).sum())
    else:
        out["flipped"] = 0
    tubes = [(np.asarray(sp.points), np.asarray(sp.radii)) for sp in specs if sp.tube]

    def hidden(points):
        d = np.full(len(points), np.inf)
        for c, r in tubes:
            d = np.minimum(d, tube_distance(points, c, r)[0])
        return d < 0.0

    def crossings(v0, v1, q, tag):
        mv = np.linalg.norm(v1 - v0, axis=1) > 0
        region = JC.vertex_ring(q, mv)
        tris = JC.triangulate(q)
        tester = JC.IntersectionTester(v1, q, region, float(np.linalg.norm(v1 - v0, axis=1).max()))
        p1 = tester.pairs(v1)
        p0 = tester.pairs(v0)
        old_set = set(map(tuple, p0.tolist()))
        new = np.array([p for p in map(tuple, p1.tolist()) if p not in old_set],
                       dtype=np.int64).reshape(-1, 2)
        vis = new
        if len(new) and tubes:
            cen = v1[tris[new]].mean(axis=2)
            vis = new[~hidden(cen.reshape(-1, 3)).reshape(-1, 2).all(axis=1)]
        out[f"{tag}selfx_before"], out[f"{tag}selfx_after"] = len(p0), len(p1)
        out[f"{tag}selfx_new"], out[f"{tag}selfx_new_visible"] = len(new), len(vis)

    if intersections and moved.any():
        crossings(cage_v0, cage_v1, cage_q, "")
        for lv in levels:
            sv0, sq = subdivide_quads(cage_v0, cage_q, lv)
            sv1, _ = subdivide_quads(cage_v1, cage_q, lv)
            crossings(sv0, sv1, sq, f"L{lv}_")
    if frame is None:
        frame = frame_or_none(base, verts)
    if frame is not None:
        mag = np.linalg.norm(cage_v1 - cage_v0, axis=1)
        part = frame.part_scores(cage_v0[moved]).argmin(axis=1)
        out["by_part"] = {}
        for i, name in enumerate(frame.parts):
            sel = part == i
            if sel.any():
                out["by_part"][name] = (int(sel.sum()), float(mag[moved][sel].max()))
    return out


def quality_violations(q: dict, area_growth=MAX_AREA_GROWTH, edge_growth=MAX_EDGE_GROWTH,
                       lipschitz=MAX_LIPSCHITZ, tolerance: float = 1e-6) -> list:
    """The bounds of :func:`junction_quality` ``q`` that are broken, as readable strings
    (empty: a valid junction): a flipped quad, area / edge growth or neighbour displacement
    difference past the guards of :func:`apply_junctions`, a smaller area than
    ``MIN_AREA_RATIO``, and any NEW crossing of the skin with itself outside the appendage's
    own tube (cage and every subdivided level measured)."""
    bad = []
    if q.get("flipped", 0):
        bad.append(f"{q['flipped']} flipped cage quads")
    if q.get("area_growth_max", 1.0) > area_growth + tolerance:
        bad.append(f"a quad grew {q['area_growth_max']:.2f}x (limit {area_growth})")
    if q.get("area_ratio_min", 1.0) < MIN_AREA_RATIO - tolerance:
        bad.append(f"a quad shrank to {q['area_ratio_min']:.2f} of its area")
    if q.get("edge_growth_max", 1.0) > edge_growth + tolerance:
        bad.append(f"an edge grew {q['edge_growth_max']:.2f}x (limit {edge_growth})")
    if q.get("lipschitz_max", 0.0) > lipschitz + tolerance:
        bad.append(f"neighbours moved {q['lipschitz_max']:.2f} edge lengths apart (limit {lipschitz})")
    for key, v in q.items():
        if key.endswith("selfx_new_visible") and v:
            bad.append(f"{v} new self-intersections outside the tube ({key})")
    return bad


def _curvature_profile(v, q, nrm, d, W):
    """Per-vertex signed mean curvature (umbrella operator . normal / edge^2) with the
    distance ``d`` to the tube, for vertices in 0 < d < 2 W."""
    n = len(v)
    a = np.concatenate([q[:, 0], q[:, 1], q[:, 2], q[:, 3]])
    b = np.concatenate([q[:, 1], q[:, 2], q[:, 3], q[:, 0]])
    s = np.zeros_like(v)
    cnt = np.zeros(n)
    elen = np.zeros(n)
    np.add.at(s, a, v[b])
    np.add.at(s, b, v[a])
    np.add.at(cnt, a, 1.0)
    np.add.at(cnt, b, 1.0)
    np.add.at(elen, a, np.linalg.norm(v[a] - v[b], axis=1))
    np.add.at(elen, b, np.linalg.norm(v[a] - v[b], axis=1))
    mean = s / np.maximum(cnt, 1)[:, None]
    ell = elen / np.maximum(cnt, 1)
    kappa = ((mean - v) * nrm).sum(axis=1) * 2.0 / np.maximum(ell * ell, _EPS)
    sel = (d > 0) & (d < 2.0 * W)
    return d[sel], kappa[sel]


def _curvature_step(profile, W) -> float:
    d, k = profile
    if not len(d) or W <= 0:
        return 0.0
    u = d / W
    inner = k[(u > 0.55) & (u <= 1.0)]
    outer = k[(u > 1.0) & (u < 1.45)]
    ref = k[(u > 0.1) & (u <= 1.0)]
    if len(inner) < 3 or len(outer) < 3 or not len(ref):
        return 0.0
    scale = max(float(np.median(np.abs(ref))), 1e-9)
    return float(abs(np.median(inner) - np.median(outer)) / scale)
