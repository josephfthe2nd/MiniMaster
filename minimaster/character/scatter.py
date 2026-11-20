"""Points on the body's skin that ride every morph: the emitter of the hair and scatter kinds.

A *region* is a set of cage quads (the scalp, the upper back, the forearms ...) worked out once, on the
neutral body, from its joints; the cage's topology never changes, so the same quads are the scalp on every
body.  The emitter spreads points over a region on that neutral body and stores each as four cage vertices
and four weights (:class:`Samples`): on a morphed body the same weights give the same place on the skin, with
the skin's normal there.  Nothing here is random: the candidates are a Halton sequence and the points are
chosen farthest-first, so

* a seed gives the same points on every run and every machine,
* the first ``n`` points of a longer emission are themselves evenly spread (raising the count ADDS points,
  the ones already there stay where they are), and
* a symmetric emission is sampled on the left half and mirrored through the cage's own mirror map, vertex
  for vertex, so the two sides are exact mirror images on a symmetric body.

``strands.py`` grows hair from these points and ``kinds.ScatterKind`` stands instances (feathers, scales,
spikes) on them.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass

import numpy as np

__all__ = ["Samples", "REGIONS", "region_mask", "region_names", "emit", "emit_line", "vertex_normals",
           "tangent_field", "head_frame"]


# --------------------------------------------------------------------------
# samples


@dataclass(frozen=True)
class Samples:
    """``N`` points on the cage: ``corners`` (N, 4) cage vertex ids and ``weights`` (N, 4) that sum to 1."""

    corners: np.ndarray
    weights: np.ndarray

    def __len__(self) -> int:
        return len(self.corners)

    def positions(self, verts) -> np.ndarray:
        """Where the points are on a body with cage vertices ``verts``."""
        V = np.asarray(verts, dtype=np.float64)
        return (V[self.corners] * self.weights[:, :, None]).sum(axis=1)

    def normals(self, vertex_normals_) -> np.ndarray:
        """The skin's outward normal at each point, from the cage's vertex normals (:func:`vertex_normals`)."""
        n = (np.asarray(vertex_normals_)[self.corners] * self.weights[:, :, None]).sum(axis=1)
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

    def mirrored(self, vertex_mirror) -> "Samples":
        """The x-mirror images of the points (``vertex_mirror``: ``appendages.mirror_vertex_map``)."""
        return Samples(np.asarray(vertex_mirror)[self.corners], self.weights.copy())

    def take(self, index) -> "Samples":
        return Samples(self.corners[index], self.weights[index])

    @staticmethod
    def concat(parts) -> "Samples":
        parts = [p for p in parts if len(p)]
        if not parts:
            return Samples(np.zeros((0, 4), dtype=np.int64), np.zeros((0, 4)))
        return Samples(np.concatenate([p.corners for p in parts]), np.concatenate([p.weights for p in parts]))


def vertex_normals(verts, quads) -> np.ndarray:
    """Area-weighted outward normals of the cage's vertices."""
    V = np.asarray(verts, dtype=np.float64)
    Q = np.asarray(quads)
    n = np.cross(V[Q[:, 2]] - V[Q[:, 0]], V[Q[:, 3]] - V[Q[:, 1]])        # twice the quad's area vector
    out = np.zeros_like(V)
    for k in range(4):
        np.add.at(out, Q[:, k], n)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def tangent_field(normals, direction) -> np.ndarray:
    """``direction`` (one vector, or one per point) laid into the skin: its part along the surface, unit
    length.  Where it stands straight on the skin (nothing left along the surface) any tangent is returned."""
    n = np.asarray(normals, dtype=np.float64)
    d = np.broadcast_to(np.asarray(direction, dtype=np.float64), n.shape)
    t = d - (d * n).sum(axis=1, keepdims=True) * n
    length = np.linalg.norm(t, axis=1, keepdims=True)
    bad = length[:, 0] < 1e-6
    if bad.any():
        # any tangent: the normal crossed with the axis it is least along
        axis = np.eye(3)[np.argmin(np.abs(n[bad]), axis=1)]
        t = t.copy()
        t[bad] = np.cross(n[bad], axis)
        length = np.linalg.norm(t, axis=1, keepdims=True)
    return t / np.maximum(length, 1e-12)


# --------------------------------------------------------------------------
# the neutral body and its landmarks


class _Body:
    """The neutral cage with what the regions are worked out from: quad centroids and normals, and the joints."""

    def __init__(self, base, lib=None):
        from . import kinds as K
        from . import mhbase as mh
        self.base = base
        self.V, self.Q = K.neutral_cage(base, lib)
        self.V = np.asarray(self.V, dtype=np.float64)
        self.Q = np.asarray(self.Q)
        self.centroid = self.V[self.Q].mean(axis=1)
        n = np.cross(self.V[self.Q[:, 2]] - self.V[self.Q[:, 0]], self.V[self.Q[:, 3]] - self.V[self.Q[:, 1]])
        self.area = 0.5 * np.linalg.norm(n, axis=1)
        self.normal = n / np.maximum(2.0 * self.area[:, None], 1e-12)
        lib = lib if lib is not None else mh.TargetLibrary()
        full = lib.apply(base.verts, mh.macro_weights())
        quads = np.asarray(base.quads)
        self._full, self._quads = full, quads

    def joint(self, name: str) -> np.ndarray:
        return self._full[np.unique(self._quads[self.base.groups["joint-" + name]])].mean(axis=0)

    def head(self):
        return head_frame(self.V, self.joint("head"), self.joint("head-2"), self.joint("l-eye"))


def head_frame(verts, head_joint, crown_joint, eye_joint) -> dict:
    """The head's own measure: ``eye`` and ``crown`` heights, ``z0`` (the vertical axis the azimuth turns
    about: the middle of the skull front to back) and ``half`` (half its depth)."""
    V = np.asarray(verts, dtype=np.float64)
    eye_y = float(eye_joint[1])
    span = float(crown_joint[1]) - float(head_joint[1])
    band = V[(V[:, 1] > eye_y + 0.15 * span) & (V[:, 1] < eye_y + 0.55 * span) & (np.abs(V[:, 0]) < 0.6 * span)]
    crown_y = float(V[np.abs(V[:, 0]) < 0.6 * span][:, 1].max())
    z_lo, z_hi = float(band[:, 2].min()), float(band[:, 2].max())
    return {"eye": eye_y, "crown": crown_y, "z0": 0.5 * (z_lo + z_hi), "half": 0.5 * (z_hi - z_lo),
            "span": crown_y - eye_y}


# the hairline: height above the eyes (0) towards the crown (1) against the azimuth from straight ahead
# (degrees: 0 the middle of the forehead, 90 over the ear, 180 the nape).  Forehead, the corner at the temple,
# the sideburn in front of the ear, over and round the ear, down to the nape.
HAIRLINE = ((0.0, 0.66), (28.0, 0.66), (46.0, 0.60), (58.0, 0.34), (66.0, 0.02), (74.0, -0.02), (80.0, 0.26),
            (108.0, 0.26), (118.0, 0.02), (132.0, -0.30), (150.0, -0.44), (180.0, -0.46))
BEARD_LINE = ((0.0, -0.46), (20.0, -0.50), (40.0, -0.52), (60.0, -0.40), (70.0, -0.10))


def _head_coords(body: _Body):
    h = body.head()
    c = body.centroid
    az = np.degrees(np.arctan2(np.abs(c[:, 0]), c[:, 2] - h["z0"]))
    t = (c[:, 1] - h["eye"]) / h["span"]
    return h, az, t


def _scalp(body: _Body) -> np.ndarray:
    h, az, t = _head_coords(body)
    line = np.interp(az, [a for a, _ in HAIRLINE], [v for _, v in HAIRLINE])
    c, n = body.centroid, body.normal
    # the skull only: within the head's own width (the ear's shell stands out past it and faces forward or
    # back), and skin that faces away from the skull's middle
    centre = np.array([0.0, h["eye"] + 0.25 * h["span"], h["z0"]])
    out = c - centre
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return (t > line) & ((n * out).sum(axis=1) > 0.35) & (t < 1.2)


def _torso_limits(body: _Body):
    sh = body.joint("l-shoulder")
    neck = body.joint("neck")
    pelvis = body.joint("pelvis")
    return sh, neck, pelvis


def _back(body: _Body, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """The back of the trunk between the pelvis (0) and the base of the neck (1), ``lo..hi`` of it."""
    sh, neck, pelvis = _torso_limits(body)
    c, n = body.centroid, body.normal
    y0 = pelvis[1] + 0.35
    y1 = neck[1] - 0.45
    t = (c[:, 1] - y0) / (y1 - y0)
    return (t > lo) & (t < hi) & (n[:, 2] < -0.35) & (np.abs(c[:, 0]) < 0.86 * sh[0]) & (c[:, 2] < sh[2] + 0.1)


def _chest(body: _Body) -> np.ndarray:
    sh, neck, pelvis = _torso_limits(body)
    c, n = body.centroid, body.normal
    y0 = pelvis[1] + 0.35 + 0.55 * (neck[1] - 0.45 - pelvis[1] - 0.35)
    return (c[:, 1] > y0) & (c[:, 1] < neck[1] - 0.55) & (n[:, 2] > 0.35) & (np.abs(c[:, 0]) < 0.80 * sh[0])


def _shoulders(body: _Body) -> np.ndarray:
    sh = body.joint("l-shoulder")
    c, n = body.centroid, body.normal
    p = c.copy()
    p[:, 0] = np.abs(p[:, 0])
    nn = n.copy()
    nn[:, 0] *= np.sign(c[:, 0] + 1e-12)
    d = np.linalg.norm(p - (sh + np.array([0.05, 0.1, 0.0])), axis=1)
    up_out = (nn * np.array([0.6, 0.8, 0.0])).sum(axis=1)
    return (d < 0.95) & (up_out > 0.25)


def _segment(body: _Body, a: str, b: str, lo: float, hi: float, radius: float) -> np.ndarray:
    """The skin round the limb segment from joint ``a`` to joint ``b`` (the left one; both sides are taken),
    between ``lo`` and ``hi`` of its length."""
    A, B = body.joint(a), body.joint(b)
    c = body.centroid.copy()
    c[:, 0] = np.abs(c[:, 0])
    ab = B - A
    s = ((c - A) @ ab) / float(ab @ ab)
    d = np.linalg.norm(c - (A + np.clip(s, 0.0, 1.0)[:, None] * ab), axis=1)
    return (s > lo) & (s < hi) & (d < radius)


def _spine(body: _Body) -> np.ndarray:
    """The strip either side of the back's midline, nape to the base of the spine (what :func:`emit_line` walks)."""
    sh, neck, pelvis = _torso_limits(body)
    c, n = body.centroid, body.normal
    return ((np.abs(c[:, 0]) < 0.16) & (n[:, 2] < -0.3) & (c[:, 1] > pelvis[1] + 0.25)
            & (c[:, 1] < neck[1] + 0.15))


def _beard(body: _Body) -> np.ndarray:
    h, az, t = _head_coords(body)
    c, n = body.centroid, body.normal
    upper = np.interp(az, [a for a, _ in BEARD_LINE], [v for _, v in BEARD_LINE])
    # below the line of the mouth's corners round to the jaw's angle, down to the fold under the chin
    return (t < upper) & (t > -0.80) & (az < 72.0) & (c[:, 2] > h["z0"] - 0.15 * h["half"]) & (
        (n[:, 2] > -0.2) | (n[:, 1] < -0.5))


REGIONS = {
    "scalp": _scalp,
    "beard": _beard,
    "back": lambda b: _back(b, 0.0, 1.0),
    "upper_back": lambda b: _back(b, 0.52, 1.02),
    "lower_back": lambda b: _back(b, 0.0, 0.5),
    "chest": _chest,
    "shoulders": _shoulders,
    "forearms": lambda b: _segment(b, "l-elbow", "l-hand", 0.10, 0.88, 0.62),
    "upper_arms": lambda b: _segment(b, "l-shoulder", "l-elbow", 0.30, 0.92, 0.75),
    "shins": lambda b: _segment(b, "l-knee", "l-ankle", 0.10, 0.90, 0.85),
    "thighs": lambda b: _segment(b, "l-upper-leg", "l-knee", 0.30, 0.92, 1.25),
    "spine": _spine,
}

_BODIES: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_MASKS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _body(base, lib=None) -> _Body:
    b = _BODIES.get(base)
    if b is None:
        b = _BODIES[base] = _Body(base, lib)
    return b


def region_names() -> tuple:
    return tuple(REGIONS)


def region_mask(base, name: str, lib=None) -> np.ndarray:
    """Which cage quads are region ``name`` (a boolean per quad; the same on every body).  Symmetric: a quad
    and its mirror image are both in or both out."""
    if name not in REGIONS:
        raise KeyError(f"unknown region {name!r}; the regions are {', '.join(REGIONS)}")
    per = _MASKS.setdefault(base, {})
    if name not in per:
        from . import appendages as A
        m = np.asarray(REGIONS[name](_body(base, lib)), dtype=bool)
        m = m & m[A.mirror_face_map(base)]
        m.setflags(write=False)
        per[name] = m
    return per[name]


# --------------------------------------------------------------------------
# the emitter


def _halton(n: int, bases=(2, 3, 5), skip: int = 0) -> np.ndarray:
    out = np.empty((n, len(bases)))
    idx = np.arange(1 + skip, n + 1 + skip)
    for k, b in enumerate(bases):
        r, f, i = np.zeros(n), 1.0 / b, idx.copy()
        while i.max() > 0:
            r += f * (i % b)
            i //= b
            f /= b
        out[:, k] = r
    return out


def _bilinear(u, v) -> np.ndarray:
    return np.stack([(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v], axis=1)


def _candidates(body: _Body, faces: np.ndarray, k: int, seed: int, weight=None):
    """``k`` area-proportional candidate points on ``faces``: ``(face, weights, position)``."""
    area = body.area[faces] * (1.0 if weight is None else np.asarray(weight, dtype=np.float64)[faces])
    cdf = np.cumsum(area)
    h = _halton(k, skip=97 * int(seed))
    pick = np.minimum(np.searchsorted(cdf, h[:, 0] * cdf[-1], side="right"), len(faces) - 1)
    f = faces[pick]
    w = _bilinear(h[:, 1], h[:, 2])
    pos = (body.V[body.Q[f]] * w[:, :, None]).sum(axis=1)
    return f, w, pos


def _farthest_first(pos: np.ndarray, count: int, start: np.ndarray, first: int) -> np.ndarray:
    """Indices of ``count`` of ``pos`` chosen farthest-first; ``start`` is each candidate's distance to what
    is already there (its own mirror image, for a symmetric emission), ``first`` the one to begin with."""
    d = start.copy()
    out = np.empty(min(count, len(pos)), dtype=np.int64)
    k = int(first)
    for i in range(len(out)):
        out[i] = k
        d = np.minimum(d, np.linalg.norm(pos - pos[k], axis=1))
        d[k] = -1.0
        k = int(np.argmax(d))
    return out


def _midline(body: _Body, mask: np.ndarray):
    """The cage's midline through the quads of ``mask``: ``(vertex ids, cumulative length)`` from the top
    down, or ``None`` if the region does not straddle the midline."""
    Q = body.Q[mask]
    on = np.abs(body.V[:, 0]) < 1e-6
    edges = set()
    for k in range(4):
        a, b = Q[:, k], Q[:, (k + 1) % 4]
        both = on[a] & on[b]
        for i, j in zip(a[both], b[both]):
            edges.add((int(min(i, j)), int(max(i, j))))
    if not edges:
        return None
    ids = np.asarray(sorted({i for e in edges for i in e}, key=lambda i: -body.V[i, 1]))      # top first
    seg = np.linalg.norm(np.diff(body.V[ids], axis=0), axis=1)
    return ids, np.concatenate([[0.0], np.cumsum(seg)])


def _on_line(ids, s, at) -> Samples:
    """Points at the lengths ``at`` along a midline (``_midline``): each on one of its edges."""
    k = np.clip(np.searchsorted(s, at, side="right") - 1, 0, len(ids) - 2)
    t = (at - s[k]) / np.maximum(s[k + 1] - s[k], 1e-12)
    corners = np.stack([ids[k], ids[k + 1], ids[k], ids[k + 1]], axis=1)
    weights = np.stack([1.0 - t, t, np.zeros_like(t), np.zeros_like(t)], axis=1)
    return Samples(corners, weights)


def emit(base, region: str, count: int, seed: int = 0, lib=None, symmetric: bool = False,
         weight=None, oversample: int = 24, midline: bool = True) -> Samples:
    """``count`` points spread evenly over ``region`` (see the module's notes on what "evenly" keeps).

    ``symmetric``: the points are mirror images of one another: a row down the midline first (where the
    region straddles it and ``midline`` is on; each of those is its own mirror image), then pairs, left
    and right in turn.  (An odd ``count`` on a region that does not reach the midline is one fewer: there
    is nowhere for the odd one to go.)  ``weight`` (one number per cage quad) thins or thickens parts of the region (the
    points gather where it is high, and a quad whose weight is 0 gets none)."""
    from . import appendages as A
    body = _body(base, lib)
    mask = region_mask(base, region, lib)
    if weight is not None:
        mask = mask & (np.asarray(weight, dtype=np.float64) > 0.0)       # nothing grows where the weight is 0
    count = int(count)
    if count <= 0 or not mask.any():
        return Samples(np.zeros((0, 4), dtype=np.int64), np.zeros((0, 4)))
    if symmetric:
        row = Samples(np.zeros((0, 4), dtype=np.int64), np.zeros((0, 4)))
        line = _midline(body, mask) if midline else None
        if line is not None:
            ids, s = line
            spacing = float(np.sqrt(body.area[mask].sum() / count))       # the field's own spacing
            k = min(int(round(s[-1] / max(spacing, 1e-9))), count // 3) if count >= 6 else 0
            if (count - k) % 2:                              # (what is left over the row goes in pairs)
                k += 1 if k + 1 <= max(count // 3, 1) else -1
            if k >= 1:
                row = _on_line(ids, s, (np.arange(k) + 0.5) / k * s[-1])
        half = (count - len(row)) // 2
        if half == 0 and not len(row):                       # one point, nowhere to mirror it: just the one
            return emit(base, region, count, seed, lib, False, weight, oversample)
        if half == 0:
            return row
        faces = np.nonzero(mask & (body.centroid[:, 0] > -1e-9))[0]
        f, w, pos = _candidates(body, faces, max(oversample * half, 600), seed, weight)
        keep = pos[:, 0] > 1e-6                              # the left half of a quad on the midline
        f, w, pos = f[keep], w[keep], pos[keep]
        # a point's own mirror image is 2x away, and the midline's row is already there
        start = 2.0 * pos[:, 0]
        if len(row):
            rp = row.positions(body.V)
            start = np.minimum(start, np.linalg.norm(pos[:, None, :] - rp[None, :, :], axis=2).min(axis=1))
        order = _farthest_first(pos, half, start, int(np.argmax(start)))
        left = Samples(body.Q[f[order]], w[order])
        right = left.mirrored(A.mirror_vertex_map(base))
        corners = np.stack([left.corners, right.corners], axis=1).reshape(-1, 4)
        weights = np.stack([left.weights, right.weights], axis=1).reshape(-1, 4)
        return Samples.concat([row, Samples(corners, weights)])
    faces = np.nonzero(mask)[0]
    f, w, pos = _candidates(body, faces, max(oversample * count, 600), seed, weight)
    order = _farthest_first(pos, count, np.full(len(pos), np.inf), 0)
    return Samples(body.Q[f[order]], w[order])


def emit_line(base, region: str, count: int, lib=None, start: float = 0.0, stop: float = 1.0) -> Samples:
    """``count`` points along the midline of a region that straddles it (the spine), evenly spaced by length
    on the neutral body, from the top down (``start..stop`` of the line).  Each sits on a midline edge of the
    cage, so it stays exactly on the midline of every symmetric body."""
    body = _body(base, lib)
    line = _midline(body, region_mask(base, region, lib))
    if line is None or count <= 0:
        return Samples(np.zeros((0, 4), dtype=np.int64), np.zeros((0, 4)))
    ids, s = line
    return _on_line(ids, s, (start + (stop - start) * (np.arange(count) + 0.5) / count) * s[-1])


# --------------------------------------------------------------------------
# which way things lie on a region

# the comb direction of a region: a limb segment's own axis (joint -> joint, the left one; mirrored on the
# right), or a fixed direction (x is "away from the midline").  Everything else lies DOWN the body.
REGION_FLOW = {
    "forearms": ("l-elbow", "l-hand"),
    "upper_arms": ("l-shoulder", "l-elbow"),
    "shins": ("l-knee", "l-ankle"),
    "thighs": ("l-upper-leg", "l-knee"),
    "shoulders": (0.75, -0.65, -0.1),
    "scalp": (0.0, -0.35, -1.0),
    "beard": (0.0, -1.0, 0.25),
    "chest": (0.0, -1.0, 0.0),
    # (down, and a little backwards: on the top of the shoulders "down" has nothing along the skin)
    "back": (0.0, -1.0, -0.45),
    "upper_back": (0.0, -1.0, -0.45),
    "lower_back": (0.0, -1.0, -0.45),
    "spine": (0.0, -1.0, -0.45),
}


def region_flow(base, verts, region: str, points, normals) -> np.ndarray:
    """The direction things lie in at ``points`` of ``region`` on the body ``verts`` (all the base mesh's
    vertices, morphed): unit vectors along the skin, mirror images of one another on the two sides."""
    P = np.asarray(points, dtype=np.float64)
    rule = REGION_FLOW.get(region, (0.0, -1.0, 0.0))
    if isinstance(rule[0], str):
        V = np.asarray(verts, dtype=np.float64)
        quads = np.asarray(base.quads)
        a, b = (V[np.unique(quads[base.groups["joint-" + n]])].mean(axis=0) for n in rule)
        d = b - a
    else:
        d = np.asarray(rule, dtype=np.float64)
    d = np.tile(d, (len(P), 1))
    d[:, 0] *= np.where(P[:, 0] < 0.0, -1.0, 1.0)
    return tangent_field(normals, d)


# --------------------------------------------------------------------------
# instances: what stands on the points

ITEMS = ("feather", "scale", "spike", "plate", "stud", "tuft")


def _lathe(profile, sides: int = 8) -> "Mesh":
    """A closed shell turned about +X from ``profile``, ``(k, 2)`` stations (x, radius) with radius > 0; a
    fan closes each end (at the first station's x and a little past the last)."""
    from ..core.mesh import Mesh
    prof = np.asarray(profile, dtype=np.float64)
    k = len(prof)
    a = 2.0 * np.pi * np.arange(sides) / sides
    ring = np.stack([np.zeros(sides), np.cos(a), np.sin(a)], axis=1)
    V = np.concatenate([np.array([x, 0.0, 0.0]) + r * ring for x, r in prof]
                       + [np.array([[prof[0, 0], 0.0, 0.0]]),
                          np.array([[prof[-1, 0] + 0.6 * prof[-1, 1], 0.0, 0.0]])])
    j = np.arange(sides)
    jn = (j + 1) % sides
    F = []
    for i in range(k - 1):
        p, q = i * sides, (i + 1) * sides
        F += [np.stack([p + j, q + jn, p + jn], axis=1), np.stack([p + j, q + j, q + jn], axis=1)]
    root, tip = k * sides, k * sides + 1
    last = (k - 1) * sides
    F += [np.stack([np.full(sides, root), j, jn], axis=1), np.stack([np.full(sides, tip), last + jn, last + j], axis=1)]
    m = Mesh(V, np.concatenate(F).astype(np.int64))
    return m if m.volume() > 0 else Mesh(V, m.faces[:, ::-1].copy())


def item_mesh(item: str, length: float, width: float, thickness: float, tip_floor: float = 0.11,
              symmetric: bool = False):
    """One instance in its own frame: its root at the origin, +X along its length (the way it points), +Y
    its upper face (away from the skin once it is laid down), +Z across it.  ``width`` and ``thickness`` are
    lengths (across, and through).  A closed shell."""
    from . import wings as W
    length, width, thickness = float(length), float(width), float(thickness)
    if item == "feather":
        # (a feather has a narrow leading vane and a broad trailing one; ``symmetric``: the one on the body's
        # midline, which has no side to lead with)
        return W._covert_blade(length, width / length, thickness, ventral=True, tip_floor=tip_floor, curl=-0.03,
                               **({"lead_frac": 0.5} if symmetric else {}))
    if item == "scale":
        # a shield: broad from the root, a rounded free end, a low keel; it lies nearly flat
        return W._blade(length, width, thickness, curl=-0.04, cup=0.10, sweep=0.0, lead_frac=0.5, tip=0.22,
                        tip_pow=2.2, calamus=0.0, rise_at=0.18, root_floor=0.72, tip_floor=tip_floor, n=8, m=6,
                        ridge=1.2, vane=0.95, ridge_dir=1.0, butt=0.12, vee=0.10)
    if item == "plate":
        # a dorsal plate: a kite standing on its root, thick at the foot
        return W._blade(length, width, thickness, curl=0.0, cup=0.0, sweep=0.0, lead_frac=0.5, tip=0.10,
                        tip_pow=1.25, calamus=0.0, rise_at=0.30, root_floor=0.55, tip_floor=tip_floor, n=9, m=6,
                        ridge=1.0, vane=1.0, ridge_dir=1.0, butt=0.0, vee=0.0, edge=0.8)
    if item == "spike":
        # (a thorn, not a needle: it keeps its width most of the way, so most of it is thick enough to print)
        t = np.linspace(0.0, 1.0, 8)
        r = np.maximum(0.5 * width * (1.0 - t) ** 0.6, 0.5 * tip_floor)
        return _lathe(np.stack([t * length, r], axis=1), sides=8)
    if item == "stud":
        # a dome: ``length`` is its height, ``width`` its diameter
        t = np.linspace(0.0, 0.92, 6)
        r = 0.5 * width * np.sqrt(np.maximum(1.0 - t * t, 0.0))
        return _lathe(np.stack([t * length, np.maximum(r, 0.5 * tip_floor)], axis=1), sides=10)
    raise KeyError(f"unknown item {item!r}; the items are {', '.join(ITEMS)}")


def instance_frames(points, normals, flow, lift, yaw=None, roll=None, mirror=None) -> np.ndarray:
    """``(N, 3, 3)`` rotations (columns: the instance's +X, +Y, +Z in the body's frame) of instances laid
    along ``flow`` and raised ``lift`` degrees off the skin; ``yaw`` turns each about the skin's normal and
    ``roll`` about its own length (degrees, per instance).  ``mirror`` (a boolean per instance) makes a
    frame left-handed: what is placed with it is the mirror image of the item (``place`` and ``wrap`` turn
    its triangles over)."""
    N = np.asarray(normals, dtype=np.float64)
    T = np.asarray(flow, dtype=np.float64)
    n = len(N)
    if yaw is not None:
        a = np.radians(np.asarray(yaw, dtype=np.float64))[:, None]
        T = np.cos(a) * T + np.sin(a) * np.cross(N, T)
    a = np.radians(np.broadcast_to(np.asarray(lift, dtype=np.float64), (n,)))[:, None]
    X = np.cos(a) * T + np.sin(a) * N
    Y = -np.sin(a) * T + np.cos(a) * N
    Z = np.cross(X, Y)
    if mirror is not None:
        # the instances of the right side are the MIRROR IMAGES of the left's (their +Z the other way: a
        # feather's leading vane stays towards the midline on both sides), not the same thing turned round
        Z = np.where(np.asarray(mirror, dtype=bool)[:, None], -Z, Z)
    if roll is not None:
        r = np.radians(np.asarray(roll, dtype=np.float64))[:, None]
        Y, Z = np.cos(r) * Y + np.sin(r) * Z, -np.sin(r) * Y + np.cos(r) * Z
    return np.stack([X, Y, Z], axis=2)


def place(mesh, origin, frame, scale: float = 1.0):
    """``mesh`` (an item, in its own frame) set at ``origin`` with the rotation ``frame`` (3, 3)."""
    from ..core.mesh import Mesh
    V = (np.asarray(mesh.vertices) * float(scale)) @ np.asarray(frame).T + np.asarray(origin)
    F = np.asarray(mesh.faces)
    return Mesh(V, F if np.linalg.det(frame) > 0 else F[:, ::-1].copy())


WELD = 0.04          # how deep the underside of something that lies on the skin is under it (units)


def wrap(meshes, points, frames, probe, lift, sink, grid=(5, 3)):
    """Flat instances (``meshes``: one per point, each in the item's own frame) laid ON the skin: a list of
    meshes in the body's frame.

    A rigid feather set down along the skin's tangent at its root is a shelf where the skin falls away (over
    the top of a shoulder) and is half buried where the skin rises under it.  Here each instance is bent over
    the skin it covers instead: the body ``probe`` is read at a small ``grid`` of points under it, and every
    vertex keeps its own height above the skin THERE: the instance's underside ``WELD`` under the skin, its
    root end ``sink`` deeper, and ``lift`` degrees' worth of its distance from the root on top (the free end
    stands off the skin a little, as a shingle's does).
    ``frames`` are ``instance_frames(...)`` at lift 0."""
    from ..core.mesh import Mesh
    P = np.asarray(points, dtype=np.float64)
    R = np.asarray(frames, dtype=np.float64)
    n = len(P)
    gx, gz = int(grid[0]), int(grid[1])
    rise = np.tan(np.radians(np.broadcast_to(np.asarray(lift, dtype=np.float64), (n,))))
    sink = np.broadcast_to(np.asarray(sink, dtype=np.float64), (n,))
    lo = np.array([[m.vertices[:, 0].min(), m.vertices[:, 2].min()] for m in meshes])
    hi = np.array([[m.vertices[:, 0].max(), m.vertices[:, 2].max()] for m in meshes])
    u = np.linspace(0.0, 1.0, gx)
    w = np.linspace(0.0, 1.0, gz)
    nx = lo[:, None, 0:1] + (hi - lo)[:, None, 0:1] * u[None, :, None]          # (n, gx, 1)
    nz = lo[:, None, 1:2] + (hi - lo)[:, None, 1:2] * w[None, :, None]          # (n, gz, 1)
    nodes = (P[:, None, None, :] + nx[:, :, None, :] * R[:, None, None, :, 0]
             + nz[:, None, :, :] * R[:, None, None, :, 2])                       # (n, gx, gz, 3)
    d, nrm = probe.distance(nodes.reshape(-1, 3), normals=True)
    move = (-d[:, None] * nrm).reshape(n, gx, gz, 3)                             # node -> the skin under it
    nrm = nrm.reshape(n, gx, gz, 3)
    out = []
    for i, m in enumerate(meshes):
        V = np.asarray(m.vertices, dtype=np.float64)
        a = np.clip((V[:, 0] - lo[i, 0]) / max(hi[i, 0] - lo[i, 0], 1e-12), 0.0, 1.0) * (gx - 1)
        b = np.clip((V[:, 2] - lo[i, 1]) / max(hi[i, 1] - lo[i, 1], 1e-12), 0.0, 1.0) * (gz - 1)
        a0 = np.minimum(a.astype(int), gx - 2)
        b0 = np.minimum(b.astype(int), gz - 2)
        fa, fb = (a - a0)[:, None], (b - b0)[:, None]

        def lerp(field):
            return ((1 - fa) * (1 - fb) * field[i, a0, b0] + fa * (1 - fb) * field[i, a0 + 1, b0]
                    + (1 - fa) * fb * field[i, a0, b0 + 1] + fa * fb * field[i, a0 + 1, b0 + 1])

        up = lerp(nrm)
        up /= np.maximum(np.linalg.norm(up, axis=1, keepdims=True), 1e-12)
        # its underside WELD under the skin all along (that is what fuses it to the body), its root end
        # dipped ``sink`` deeper (the row above covers it), its free end rising at ``lift``
        x = np.maximum(V[:, 0], 0.0)
        dip = np.clip(1.0 - x / max(0.45 * (hi[i, 0] - lo[i, 0]), 1e-12), 0.0, 1.0)
        height = (V[:, 1] - V[:, 1].min()) - WELD - sink[i] * dip + x * rise[i]
        W = P[i] + V[:, 0:1] * R[i, :, 0] + V[:, 2:3] * R[i, :, 2] + lerp(move) + height[:, None] * up
        F = np.asarray(m.faces)
        out.append(Mesh(W, F if np.linalg.det(R[i]) > 0 else F[:, ::-1].copy()))      # (a mirrored frame)
    return out
