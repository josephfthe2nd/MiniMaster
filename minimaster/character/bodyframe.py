"""The skeleton of the morphed body as the junction core sees it: which part of the body a
skin vertex belongs to, and how big that part is.

A junction moves the skin around an appendage root. Two things must not depend on how the
spec was written: it must never mould the skin of ANOTHER part of the body (an extra arm's
root must not drag the original arm's armpit; a horn must not swell the neck), and a spec
written in absolute units must not be a scapular mound the size of a child's whole back. Both
come from the same landmarks, the ``joint-*`` marker cubes of the MakeHuman base mesh (the
bones: pelvis, spine, neck, head, shoulder, elbow, hand, hip, knee, ankle), which move with
every morph target, so they are right for every body (a child, a heavy figure).

* :class:`BodyFrame` ``(base, verts)``: the bone segments grouped into **parts** (``head``,
  ``trunk``, ``arm_l`` / ``arm_r``, ``leg_l`` / ``leg_r``), the radius of each segment (how
  far its skin lies from its bone, measured on this body), :meth:`BodyFrame.part_scores` (the
  distance of points to each part in radii: the smallest is the part a skin point belongs
  to), :meth:`BodyFrame.owner_weights` (1 on the owner's skin, 0 on another part's, a smooth
  blend between), and :meth:`BodyFrame.size` (the radius of a part, the unit of a
  :class:`~.junction.Mound` with a ``scale``).
* :func:`frame_or_none` builds one when the base mesh has the markers (the synthetic bodies
  of the tests do not: every consumer treats ``None`` as "no skeleton: no protection, absolute
  sizes").

Everything is in the MakeHuman frame (Y up).
"""

from __future__ import annotations

import threading

import numpy as np

_EPS = 1e-12
_LOCK = threading.Lock()

# the bone chains, by part. A chain is a list of joint markers; a segment joins neighbours.
PARTS = {
    "head": (("head", "head-2"), ("head", "jaw")),
    "trunk": (("pelvis", "spine-4"), ("spine-4", "spine-3"), ("spine-3", "spine-2"),
              ("spine-2", "spine-1"), ("spine-1", "neck"), ("neck", "head"),
              ("pelvis", "l-upper-leg"), ("pelvis", "r-upper-leg"),
              ("l-shoulder", "r-shoulder")),
    "arm_l": (("l-shoulder", "l-elbow"), ("l-elbow", "l-hand"), ("l-hand", "l-finger-3-4")),
    "arm_r": (("r-shoulder", "r-elbow"), ("r-elbow", "r-hand"), ("r-hand", "r-finger-3-4")),
    "leg_l": (("l-upper-leg", "l-knee"), ("l-knee", "l-ankle"), ("l-ankle", "l-foot-2")),
    "leg_r": (("r-upper-leg", "r-knee"), ("r-knee", "r-ankle"), ("r-ankle", "r-foot-2")),
}
# how thick each segment is, as a fraction of the body's height (a prior: the measured radius is
# kept within 0.6-1.5 of it): the same order as PARTS
RADIUS_PRIOR = {
    "head": (0.062, 0.04),
    "trunk": (0.075, 0.075, 0.075, 0.075, 0.04, 0.035, 0.05, 0.05, 0.04),
    "arm_l": (0.027, 0.019, 0.012), "arm_r": (0.027, 0.019, 0.012),
    "leg_l": (0.04, 0.028, 0.02), "leg_r": (0.04, 0.028, 0.02),
}
# the parts a "scale" name stands for in a Mound / spec (the radius of the part's thickest
# segments is the unit)
SCALE_PARTS = {"head": ("head",), "trunk": ("trunk",), "arm": ("arm_l", "arm_r"),
               "leg": ("leg_l", "leg_r")}
# the SIZE of a part (the unit of a relative size, the cap of a fillet) is not read off the fitted
# segment radii (they settle on whichever vertices a round of the fit assigns to a segment, and the
# trunk's radius of one body varied 1.14..1.73 with the appendage whose anatomy coupling reshaped it):
# it is the MEDIAN, over the part's skin, of the distance to the nearest bone in the radii the
# bones are expected to have (``RADIUS_PRIOR``), times the body's height times this calibration,
# which makes the neutral male body's sizes what the relative sizes were tuned on (head 0.864, trunk
# 1.56, arm 0.494, leg 0.731 units). A median does not care what a few hundred vertices of one
# patch of the back do.
SIZE_FRACTION = {"head": 0.06697, "trunk": 0.08351, "arm": 0.01985, "leg": 0.04941}
SIZE_FAMILY = {"head": "head", "trunk": "trunk", "arm_l": "arm", "arm_r": "arm", "leg_l": "leg", "leg_r": "leg"}


def _joint(base, verts, name):
    from .appendages import joint_position
    return np.asarray(joint_position(base, name, verts), dtype=np.float64)


class BodyFrame:
    """The bone segments of one morphed body, their radii and the parts they form."""

    def __init__(self, base, verts, sample: int = 6):
        names = sorted({n for segs in PARTS.values() for s in segs for n in s})
        joints = {n: _joint(base, verts, n) for n in names}
        self.joints = joints
        self.parts = list(PARTS)
        seg_a, seg_b, seg_part, seg_names = [], [], [], []
        for pi, part in enumerate(self.parts):
            for a, b in PARTS[part]:
                seg_names.append((a, b))
                seg_a.append(joints[a])
                seg_b.append(joints[b])
                seg_part.append(pi)
        self.seg_a = np.asarray(seg_a)
        self.seg_b = np.asarray(seg_b)
        self.seg_part = np.asarray(seg_part)
        cage_v = base.body_cage(verts)[0]
        pts = cage_v[:: max(int(sample), 1)]
        d = self._seg_distances(pts)
        height = float(np.ptp(cage_v[:, 1]))
        index = {sn: i for i, sn in enumerate(seg_names)}

        def mir(n):
            return ("r-" + n[2:]) if n.startswith("l-") else ("l-" + n[2:]) if n.startswith("r-") else n

        mirror = np.array([index.get((mir(a), mir(b)), i) for i, (a, b) in enumerate(seg_names)])
        prior = np.array([height * f for p in self.parts for f in RADIUS_PRIOR[p]])
        rad = prior.copy()
        for _ in range(3):                  # radii <-> assignment: a few rounds settle it
            near = (d / rad[None, :]).argmin(axis=1)
            new = rad.copy()
            for s in range(len(rad)):
                sel = d[near == s, s]
                if len(sel) >= 8:
                    new[s] = float(np.percentile(sel, 75))
            rad = np.clip(new, 0.6 * prior, 1.5 * prior)
            rad = 0.5 * (rad + rad[mirror])      # a body is as thick on one side as on the other
        self.seg_radius = rad
        self._set_sizes(cage_v[::3], height)

    def _set_sizes(self, cage_v, height: float):
        """``self.sizes``: the size (radius) of each part, from the median of its skin's distance to
        the bones in their expected radii; a left part and its right twin share one."""
        d = self._seg_distances(cage_v)
        prior = np.array([height * f for p in self.parts for f in RADIUS_PRIOR[p]])
        near = (d / prior[None, :]).argmin(axis=1)
        ratio = d[np.arange(len(d)), near] / prior[near]
        owner = self.seg_part[near]
        self.sizes = {}
        for fam, frac in SIZE_FRACTION.items():
            ids = [i for i, p in enumerate(self.parts) if SIZE_FAMILY[p] == fam]
            sel = np.isin(owner, ids)
            k = float(np.median(ratio[sel])) if sel.sum() >= 8 else 1.0
            for i in ids:
                self.sizes[self.parts[i]] = frac * height * k

    # -- distances -------------------------------------------------------------------------

    def _seg_distances(self, pts) -> np.ndarray:
        """Distance of each point to each bone segment: ``(m, n_segments)``."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        a, ab = self.seg_a, self.seg_b - self.seg_a
        den = np.maximum((ab * ab).sum(axis=1), _EPS)
        out = np.empty((len(pts), len(a)))
        for s in range(0, len(pts), 4096):                      # a (chunk, segments, 3) block at a time
            rel = pts[s:s + 4096, None, :] - a[None, :, :]
            t = np.clip((rel * ab[None]).sum(axis=2) / den[None], 0.0, 1.0)
            off = rel - t[:, :, None] * ab[None]
            out[s:s + 4096] = np.sqrt((off * off).sum(axis=2))
        return out

    def part_scores(self, pts) -> np.ndarray:
        """``(m, n_parts)``: the distance of each point to each part's nearest bone, in that
        bone's own radius (0 on the bone, 1 about at its skin)."""
        d = self._seg_distances(pts) / self.seg_radius[None, :]
        out = np.full((len(d), len(self.parts)), np.inf)
        for pi in range(len(self.parts)):
            cols = np.nonzero(self.seg_part == pi)[0]
            out[:, pi] = d[:, cols].min(axis=1)
        return out

    def owner(self, pts) -> int:
        """The index of the part most of ``pts`` belong to (the root of an appendage: its
        contact with the skin)."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        if not len(pts):
            return 0
        votes = np.bincount(self.part_scores(pts).argmin(axis=1), minlength=len(self.parts))
        return int(votes.argmax())

    def resolve_owner(self, owner, at) -> int:
        """The part index for an ``owner`` hint of a spec: a part name (``head``, ``trunk``,
        ``arm_l`` ...), a family (``arm`` / ``leg``: the side nearest the point ``at``), or
        ``None`` (automatic: the part ``at`` and the skin around it belong to)."""
        at = np.asarray(at, dtype=np.float64).reshape(-1, 3)
        if owner is None:
            return self.owner(at)
        if owner in self.parts:
            return self.parts.index(owner)
        if owner in ("arm", "leg"):
            sc = self.part_scores(at.mean(axis=0, keepdims=True))[0]
            sides = [i for i, p in enumerate(self.parts) if p.startswith(owner + "_")]
            return int(min(sides, key=lambda i: sc[i]))
        raise ValueError(f"unknown owner {owner!r}: a part ({self.parts}), 'arm', 'leg' or None")

    def owner_weights(self, pts, owner: int, margin: float = 0.35) -> np.ndarray:
        """How much of ``owner``'s skin each point is: 1 where the owner's bone is nearer (in
        radii) than any other part's by ``margin`` or more, 0 where another part's is nearer by
        ``margin``, a smooth blend between (the armpit crease, the shoulder, the hip)."""
        sc = self.part_scores(pts)
        mine = sc[:, owner]
        others = np.delete(sc, owner, axis=1).min(axis=1)
        u = np.clip((others - mine) / margin * 0.5 + 0.5, 0.0, 1.0)
        return u * u * (3.0 - 2.0 * u)

    def bone_closest(self, pts, part):
        """``(closest points, distances)`` of ``pts`` to the nearest bone segment of ``part``
        (a name or an index): the axis a skin point of that part stands out from."""
        pi = part if isinstance(part, int) else self.parts.index(part)
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        sel = self.seg_part == pi
        a = self.seg_a[sel]
        ab = self.seg_b[sel] - a
        rel = pts[:, None, :] - a[None, :, :]
        t = np.clip((rel * ab[None]).sum(axis=2) / np.maximum((ab * ab).sum(axis=1), _EPS)[None], 0.0, 1.0)
        off = rel - t[:, :, None] * ab[None]
        d = np.sqrt((off * off).sum(axis=2))
        k = d.argmin(axis=1)
        rows = np.arange(len(pts))
        return a[k] + t[rows, k][:, None] * ab[k], d[rows, k]

    # -- sizes -----------------------------------------------------------------------------

    def size(self, scale: str, near=None) -> float:
        """The radius of a part, the unit of a relative size: ``scale`` is ``head``,
        ``trunk``, ``arm``, ``leg`` or ``local`` (the part nearest the point ``near``)."""
        if scale == "local":
            if near is None:
                raise ValueError("a 'local' size needs a point")
            return self.part_size(self.parts[self.owner(np.asarray(near, dtype=float)[None, :])])
        if scale not in SCALE_PARTS:
            raise ValueError(f"unknown scale {scale!r}: one of {sorted(SCALE_PARTS)} or 'local'")
        return float(np.mean([self.part_size(p) for p in SCALE_PARTS[scale]]))

    def part_size(self, part) -> float:
        """The radius of a part: the unit of a relative size and the cap of a fillet. Read off the
        median distance of its skin to its bones (``SIZE_FRACTION``), not off the fitted segment
        radii (which move 30-60 % with the anatomy coupling of whichever appendage is on the body)."""
        name = part if isinstance(part, str) else self.parts[int(part)]
        return float(self.sizes[name])


_FRAMES: dict = {}


def frame_or_none(base, verts):
    """A :class:`BodyFrame`, or ``None`` when ``base`` has no skeleton markers (a synthetic
    body). The last few frames are kept (a body is shaped again and again while an appendage is
    edited): the key is the base mesh and a hash of the morphed vertices."""
    if not hasattr(base, "groups") or "joint-pelvis" not in getattr(base, "groups", {}):
        return None
    v = np.ascontiguousarray(verts)
    key = (id(base), v.shape, hash(v.tobytes()))
    with _LOCK:                              # the lab shapes the skin from worker threads too
        frame = _FRAMES.get(key)
    if frame is None:
        try:
            frame = BodyFrame(base, verts)
        except (KeyError, ValueError, IndexError):
            return None
        with _LOCK:
            while len(_FRAMES) >= 6:
                _FRAMES.pop(next(iter(_FRAMES)), None)
            _FRAMES[key] = frame
    return frame
