"""Skin-conforming junctions: the body's skin flows into an appendage root.

THE PROBLEM. An appendage is its own closed shell that overlaps the body (the slicer
unions overlapping shells), so where a tail, horn, wing arm or extra limb enters the skin
there is a hard intersection curve and nothing else: the skin does not rise to meet it, no
flesh gathers around it, it reads as "slapped over skin". The body mesh cannot get new
vertices (1,280 morph targets address it BY VERTEX INDEX) but its vertices can be MOVED
after the morphs are applied. That is what this module does.

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
from dataclasses import dataclass, replace

import numpy as np

# --------------------------------------------------------------------------
# defaults

DEFAULT_WIDTH = 1.0           # fillet width, in units of the base radius
DEFAULT_HEIGHT = 0.6          # fillet height / width (1 = a quarter circle)
DEFAULT_SHARPNESS = 3.0       # k of the profile: 2 = quarter ellipse (C1 at W), 3 = curvature-continuous
DEFAULT_LEAD = 0.25           # how far (in fillet widths) the skin starts to rise before the tube
MAX_MOVE_FACTOR = 1.6         # a vertex moves at most this many fillet widths
MIN_AREA_RATIO = 0.12         # a quad may not shrink below this fraction of its own area
MIN_NORMAL_DOT = 0.2          # ... nor turn its normal past this (cos of ~78 degrees)
LIMIT_ITERATIONS = 12
LIMIT_BACKOFF = 0.6

_EPS = 1e-12


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
    that axis. ``direction`` replaces the skin normal by a fixed direction."""

    center: tuple
    radius: float
    height: float
    end: tuple | None = None
    axis: tuple | None = None
    stretch: float = 1.0
    direction: tuple | None = None

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

    def mirrored(self) -> "Mound":
        m = _mirror_point
        return Mound(m(self.center), self.radius, self.height,
                     None if self.end is None else m(self.end),
                     None if self.axis is None else m(self.axis), self.stretch,
                     None if self.direction is None else m(self.direction))

    def transformed(self, matrix) -> "Mound":
        M = np.asarray(matrix, dtype=np.float64)
        s = _uniform_scale(M)

        def pt(p):
            return None if p is None else tuple(M[:3, :3] @ np.asarray(p) + M[:3, 3])

        def vec(p):
            return None if p is None else tuple(M[:3, :3] @ np.asarray(p) / s)

        return Mound(pt(self.center), self.radius * s, self.height * s, pt(self.end),
                     vec(self.axis), self.stretch, vec(self.direction))

    def to_dict(self) -> dict:
        d = {"center": list(self.center), "radius": self.radius, "height": self.height}
        for k in ("end", "axis", "direction"):
            if getattr(self, k) is not None:
                d[k] = list(getattr(self, k))
        if self.stretch != 1.0:
            d["stretch"] = self.stretch
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Mound":
        return cls(d["center"], d["radius"], d["height"], d.get("end"), d.get("axis"),
                   d.get("stretch", 1.0), d.get("direction"))


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
    tallest mound).
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
                       keep=tuple(k.mirrored() for k in self.keep))

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
            max_move=None if self.max_move is None else self.max_move * s)

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
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "JunctionSpec":
        return cls(d["name"], d.get("points", ()), d.get("radii", ()),
                   d.get("width", DEFAULT_WIDTH), d.get("base_radius"),
                   d.get("height", DEFAULT_HEIGHT), d.get("sharpness", DEFAULT_SHARPNESS),
                   d.get("pull", 1.0), d.get("strength", 1.0),
                   tuple(Mound.from_dict(m) for m in d.get("mounds", ())),
                   tuple(Keep.from_dict(k) for k in d.get("keep", ())), d.get("max_move"),
                   d.get("lead", DEFAULT_LEAD))

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
        d = np.linalg.norm(pts - c, axis=1) - r
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


def _tube_displacement(spec, P, N, inside_margin):
    """The fillet displacement of the points ``P`` (with skin normals ``N``): ``(m, 3)``."""
    W = spec.fillet_width
    H = spec.height * W
    k = spec.sharpness
    centre = np.asarray(spec.points)
    radii = np.asarray(spec.radii)
    D, C, Rc = tube_distance(P, centre, radii, radius=True)
    out = np.zeros_like(P)
    act = D < W
    if not act.any():
        return out
    Pa, Na, Da, Ca, rad_c = P[act], N[act], D[act], C[act], Rc[act]
    nu = Pa - Ca
    nu /= np.maximum(np.linalg.norm(nu, axis=1, keepdims=True), _EPS)
    cps = (nu * Na).sum(axis=1)                                  # cos of the angle skin-normal / wall-normal
    sps = np.sqrt(np.maximum(1.0 - cps * cps, 0.0))              # 0 above the tube, 1 beside it
    away = nu - cps[:, None] * Na                                # in the skin plane, away from the tube
    away /= np.maximum(np.linalg.norm(away, axis=1, keepdims=True), 1e-9)
    climb = Na - cps[:, None] * nu                               # up the wall: the skin normal's tangent part
    climb /= np.maximum(np.linalg.norm(climb, axis=1, keepdims=True), 1e-9)
    # only the EXPOSED tube is moulded around: the weight falls to zero as the closest point of
    # the centreline sinks below the skin (a tube buried under the scalp leaves it alone)
    height_c = ((Ca - Pa) * Na).sum(axis=1)
    reach = np.maximum(rad_c, 0.0) + spec.lead * W
    expose = _smoothstep(1.0 + height_c / np.maximum(reach, _EPS))
    outside = Da > 0.0
    u = np.clip(Da / W, 0.0, 1.0)
    rho = np.where(outside, W * fillet_remap(u), Da)             # the remaining distance
    lift = np.where(outside, fillet_height(np.maximum(rho, 0.0), W, H, k), H)
    pull = np.where(outside, (Da - rho) * spec.pull, 0.0)
    wgt = sps * expose
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


def junction_displacement(spec: JunctionSpec, cage_v, normals, candidates=None):
    """``(indices, displacement)``: how ``spec`` moves the cage vertices it touches, before
    the global limits. Pure."""
    cand = _candidates(spec, np.ascontiguousarray(cage_v.T)) if candidates is None else candidates
    if not len(cand) or not spec.active:
        return cand[:0], np.zeros((0, 3))
    P, N = cage_v[cand], normals[cand]
    d = np.zeros_like(P)
    if spec.tube and spec.fillet_width > 0:
        d += _tube_displacement(spec, P, N, inside_margin=0.04 * min(spec.radii))
    for m in spec.mounds:
        d += _mound_displacement(m, P, N)
    d *= (spec.strength * _keep_weight(spec, P))[:, None]
    limit = spec.move_limit()
    mag = np.linalg.norm(d, axis=1)
    over = mag > limit
    if over.any():
        d[over] *= (limit / mag[over])[:, None]
    return cand, d


# --------------------------------------------------------------------------
# validity: no flipped or collapsed quad


def _quad_vectors(v, q):
    p = v[q]
    return _cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])            # twice the area vector


def _touching(quads, vertex_mask) -> np.ndarray:
    return np.nonzero(vertex_mask[quads].any(axis=1))[0]


def limit_displacement(cage_v, quads, disp, min_area_ratio=MIN_AREA_RATIO,
                       min_normal_dot=MIN_NORMAL_DOT, iterations=LIMIT_ITERATIONS):
    """Back off the displacement of the vertices of any quad that would flip (its normal
    turns past ``min_normal_dot``) or collapse (its area falls below ``min_area_ratio`` of
    its own); ``(disp, factor, rounds)`` with ``factor`` the per-vertex scale applied."""
    moved = np.linalg.norm(disp, axis=1) > 0
    factor = np.ones(len(disp))
    if not moved.any():
        return disp, factor, 0
    qi = _touching(quads, moved)
    old = _quad_vectors(cage_v, quads[qi])
    old_len = np.linalg.norm(old, axis=1)
    rounds = 0
    for rounds in range(1, iterations + 1):
        new = _quad_vectors(cage_v + disp * factor[:, None], quads[qi])
        new_len = np.linalg.norm(new, axis=1)
        dot = (new * old).sum(axis=1) / np.maximum(new_len * old_len, _EPS)
        bad = (dot < min_normal_dot) | (new_len < min_area_ratio * old_len)
        if not bad.any():
            return disp * factor[:, None], factor, rounds - 1
        corners = np.unique(quads[qi[bad]])
        factor[corners] *= LIMIT_BACKOFF
    return disp * factor[:, None], factor, rounds


# --------------------------------------------------------------------------
# the public function


def apply_junctions(base, verts, specs, info: dict | None = None, *,
                    min_area_ratio=MIN_AREA_RATIO, min_normal_dot=MIN_NORMAL_DOT):
    """The body vertices ``verts`` (19,158 x 3, the MORPHED control vertices) with the skin
    moulded around every :class:`JunctionSpec` of ``specs``.

    Pure and deterministic; topology and vertex numbering are untouched (only body cage
    vertices move; eyes, teeth and joint markers are not); returns ``verts`` itself when
    there is no active spec. Each spec is computed from the INPUT body and the displacements
    are summed (so the result does not depend on the order of the specs, and a mirrored pair
    gives a mirror-symmetric result on a symmetric body), then capped and checked for
    flipped or collapsed quads. Never feed the result back in as ``verts`` (it would
    be a second junction on the first); ``info`` (a dict, optional) receives ``moved``,
    ``max_move``, ``limited`` (vertices backed off), ``rounds`` and ``specs``."""
    specs = [s for s in (specs or ()) if s is not None and s.active]
    if info is not None:
        info.update(moved=0, max_move=0.0, limited=0, rounds=0, specs=len(specs))
    if not specs:
        return verts
    cage_v, cage_q = base.body_cage(verts)
    normals = vertex_normals(cage_v, cage_q)
    total = np.zeros_like(cage_v)
    limit = 0.0
    cols = np.ascontiguousarray(cage_v.T)
    for spec in specs:
        idx, d = junction_displacement(spec, cage_v, normals, _candidates(spec, cols))
        if len(idx):
            total[idx] += d
            limit = max(limit, spec.move_limit())
    mag = np.linalg.norm(total, axis=1)
    if limit > 0:
        over = mag > limit
        if over.any():                                   # two overlapping specs: one cap on the sum
            total[over] *= (limit / mag[over])[:, None]
    if not (np.linalg.norm(total, axis=1) > 0).any():
        return verts
    total, factor, rounds = limit_displacement(cage_v, cage_q, total, min_area_ratio,
                                               min_normal_dot)
    out = np.array(verts, dtype=np.float64, copy=True)
    out[base.body_verts_idx] = cage_v + total
    if info is not None:
        mag = np.linalg.norm(total, axis=1)
        info.update(moved=int((mag > 0).sum()), max_move=float(mag.max()),
                    limited=int((factor < 1.0).sum()), rounds=int(rounds))
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
