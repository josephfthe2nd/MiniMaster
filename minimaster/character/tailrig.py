"""The tail as an anatomy: a caudal vertebra chain, the muscle sleeve around it, the skin over both.

A tail is not a tube pushed through the skin. It is the continuation of the spine: the sacrum
ends in an apex, the caudal vertebrae Cd1..Cdn follow it along the same axis, the sacro- and
ischio-caudal muscles arise on the sacrum and ilium and act on the distal vertebrae through long
tendons (so the bellies sit on the proximal third and the rest is tendon, fascia and skin), and
the skin is a thin envelope over all of it. This module builds those three layers from ONE set of
numbers, so the skin that is printed and rendered is the envelope of the muscle and the bone plus a
skin thickness, never a second, unrelated shape:

* ``bone``   the vertebra chain (centra, transverse processes, neural spines on the first four,
             chevron bones ventrally from Cd3-4) plus the sacral apex it grows from;
* ``muscle`` the sleeve: its outer surface is the skin's inward by the skin thickness, lobed where
             the muscle bellies are (dorsal pair with a median groove, ventral pair, lateral bands);
* ``skin``   the shell: the muscle surface plus ``skin_thickness`` (low-passed in the lobes).

THE LOCAL FRAME (the one :func:`~minimaster.character.kinds.TailKind` places on the body) is
``x`` = up in the sagittal plane, ``y`` = lateral, ``z`` = the axis of the first part of the tail
(out of the body); ``s`` runs along the axis, 0 at the skin point of the root, negative inside the
body (the hidden stem: the sacral apex, Cd1 and the first rings of the sleeve), positive outward.
The units are those of the tail parameters (``length``, ``base_radius``); the kind scales the
mesh to the body.

THE NUMBERS (``docs/anatomy_tail.md`` says where each one comes from, and which are guesses):

* the radius law is the registry tail's: ``r(t) = tip + (R - tip) (1 - t)^0.85`` over the length,
  times a root flare ``1 + F exp(-s / 0.9 R)`` held at ``1 + F`` inside the body;
* the section is an ellipse, wider than tall at the root (lateral : vertical 1.2 at the skin,
  1.0 at 2 R) and slightly taller than wide beyond (0.92), measured in the transported ring frame;
* the path is stiff for the first ``2.5 R`` (the sacrococcygeal joint and Cd1-3 are muscle bound),
  then ``droop`` bends it down in the sagittal plane by a smooth power law, ``curl`` curls the
  distal 45 % up, ``lift`` rotates it about the sacrococcygeal joint, ``sway`` flexes it
  sideways; the curvature is limited so the shell never folds on its own inner side;
* the caudal vertebra count is ``round(48 L / H_ref)`` (6..30), their lengths rise from Cd1 (0.55 of
  the longest) to a maximum at 29 % of the chain and fall to 0.35 at the tip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from ..core.mesh import Mesh
from . import appendages as A
from . import junction as J

# --------------------------------------------------------------------------
# constants (every one is documented in docs/anatomy_tail.md)

H_REF = 16.76            # the height of the neutral body: the tail parameters are at this size
STEM_R = 1.75            # the hidden stem (bone root) below the skin point, in base radii
FLARE_LAMBDA = 0.9       # decay of the root flare along the axis, in base radii
ROOT_ASPECT = 1.08       # lateral : vertical semi-axis at the skin point
MID_ASPECT = 0.92        # ... beyond 0.3 L (slightly taller than wide)
STIFF_R = 2.5            # the stiff proximal part, in base radii
DROOP_DEG = 85.0         # total sagittal droop per unit of the ``droop`` parameter
CURL_DEG = 45.0          # distal up-curl per unit of ``curl``
DROOP_POW = 1.2          # the droop grows with this power of the distance past the stiff root
CURL_FROM = 0.55         # the curl starts at this fraction of the length
BEND_LIMIT = 0.8         # a ring's radius times the path curvature may not exceed this
SKIN_ROOT, SKIN_TIP, SKIN_FLOOR = 0.08, 0.12, 0.02    # skin thickness: x radius, root to tip; floor in units
BELLY_END = 0.55         # the muscle bellies end here (tendons beyond), as a fraction of the length
BELLY_FULL = 0.12        # ... and are at full size before here
LOBE = 0.06              # relative depth of the muscle lobing
DAMP = 0.4               # how much of the lobing the skin keeps
# centrum radius / local skin radius at t = 0, .25, 1: calibrated on the cat skeleton plate
# (cm_Anatomical_technology: the bone is 0.98, 0.59, 0.50, 0.32, 0.19 of its root width at 25, 50, 75,
# 95 % and 100 % of the length), which is ~0.30-0.37 of the local radius all along (the research's
# guess of 0.40 / 0.55 distally made the bone 1.4-1.9 x too thick there)
BONE_FRAC = ((0.0, 0.25, 1.0), (0.30, 0.33, 0.33))
VERTEBRA_LENGTHS = ((0.0, 0.29, 1.0), (0.55, 1.0, 0.35))   # relative length at the chain fraction
TIP_CAP = (32.0, 62.0, 82.0)   # the rounded tip: rings on a quarter sphere (degrees)


# the skin around the root (junction_spec): positions in base radii R in the local frame
# (x up the back, y lateral, z out of the body); tuned on the real mesh, see docs/anatomy_tail.md
MOUND_UP = (0.2, 2.6)         # the sacral rise: a ridge up the back from here to there (x, in R)
MOUND_RADIUS = 1.5            # its half width (R)
MOUND_HEIGHT = 0.30           # its height (R)
CHEEK_Y, CHEEK_X = 1.9, -0.6  # the cheek wrap mounds: lateral and vertical position (R)
CHEEK_RADIUS, CHEEK_HEIGHT = 1.4, 0.18
KEEP_BEADS = ((-1.1, -0.3), (-1.6, -0.8), (-1.9, -1.4), (-2.1, -2.0))   # the cleft below the root, as spheres on the midline (x, z in R)
KEEP_RADIUS, KEEP_FADE = 0.9, 0.6     # ... each this wide (R), and growing back to full over this much of it


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


# --------------------------------------------------------------------------
# the parameters


@dataclass(frozen=True)
class TailParams:
    """What the anatomy is built from (units of the tail parameters, at the reference body)."""

    length: float = 6.0
    base_radius: float = 0.45
    tip_radius: float = 0.09
    curl: float = 0.55
    droop: float = 0.9
    lift: float = 0.0            # degrees, about the sacrococcygeal joint (+ up)
    sway: float = 0.0            # degrees of lateral flexion over the length (+ toward +y)
    flare: float = 0.35          # F of the root flare: the radius at the skin is 1 + F times R
    segments: int = 12           # rings along the free tail (the root has its own, denser rings)
    ring_verts: int = 16
    vertebrae: int = 0           # 0: round(48 L / H_REF)
    crest: float = 0.0           # a dorsal ridge (reptile), in base radii

    def with_(self, **kw) -> "TailParams":
        return replace(self, **kw)

    @property
    def stem(self) -> float:
        return STEM_R * self.base_radius

    @property
    def tip(self) -> float:
        return max(float(self.tip_radius), 0.35 * SKIN_FLOOR)

    def n_vertebrae(self) -> int:
        if self.vertebrae and self.vertebrae > 0:
            return int(min(max(self.vertebrae, 3), 60))
        return int(min(max(round(48.0 * self.length / H_REF), 6), 30))


# --------------------------------------------------------------------------
# the model


class TailModel:
    """The tail in its local frame: path, ring frames and the three layers' radii.

    Everything derives from one :class:`TailParams`. The meshes are built on demand
    (:meth:`skin`, :meth:`muscle`, :meth:`bone`), the skin is the one that is printed."""

    def __init__(self, params: TailParams):
        self.p = params
        p = params
        self.R, self.L = float(p.base_radius), float(p.length)
        self.tip = p.tip
        self.stem = p.stem
        self._path()
        self._stations()

    # -- the radius laws -------------------------------------------------------------------

    def law(self, s):
        """The vertical semi-axis of the SKIN at ``s`` (before the lobing): the registry tail's
        radius law times the root flare, held at its full value inside the body."""
        s = np.asarray(s, dtype=np.float64)
        t = np.clip(s / self.L, 0.0, 1.0)
        r = self.tip + (self.R - self.tip) * (1.0 - t) ** 0.85
        return r * (1.0 + self.p.flare * np.exp(-np.maximum(s, 0.0) / (FLARE_LAMBDA * self.R)))

    def aspect(self, s):
        """Lateral : vertical semi-axis ratio of the section at ``s``."""
        s = np.asarray(s, dtype=np.float64)
        knots_s = [0.0, self.R, 2.0 * self.R, max(0.3 * self.L, 3.0 * self.R)]
        return np.interp(s, knots_s, [ROOT_ASPECT, 1.0 + 0.6 * (ROOT_ASPECT - 1.0), 1.0, MID_ASPECT])

    def skin_thickness(self, s):
        s = np.asarray(s, dtype=np.float64)
        t = np.clip(s / self.L, 0.0, 1.0)
        return np.maximum(SKIN_FLOOR, (SKIN_ROOT + (SKIN_TIP - SKIN_ROOT) * t) * self.law(s))

    def belly(self, s):
        """How much of the sleeve is muscle belly at ``s`` (1 on the proximal part, 0 where only
        tendon, fascia and fat are left): a cosine fall from ``BELLY_FULL`` to ``BELLY_END``."""
        t = np.clip(np.asarray(s, dtype=np.float64) / self.L, 0.0, 1.0)
        u = np.clip((t - BELLY_FULL) / (BELLY_END - BELLY_FULL), 0.0, 1.0)
        return 0.5 + 0.5 * np.cos(np.pi * u)

    def bone_radius(self, s):
        """Radius of the vertebral centrum at ``s``: 0.30 of the local skin radius at the root, 0.33
        beyond a quarter of the length (``BONE_FRAC``)."""
        s = np.asarray(s, dtype=np.float64)
        t = np.clip(s / self.L, 0.0, 1.0)
        local = self.tip + (self.R - self.tip) * (1.0 - t) ** 0.85
        return np.interp(t, *BONE_FRAC) * local

    def lobes(self, theta):
        """The muscle lobing around the section (angle from the dorsal midline): a median groove
        dorsally, two dorsal and two ventral bellies, lateral bands; mean ~0."""
        th = np.asarray(theta, dtype=np.float64)
        return np.sin(2.0 * th) ** 2 - 0.5

    # -- the path --------------------------------------------------------------------------

    def _path(self):
        p = self.p
        R, L = self.R, self.L
        n = 900
        sd = np.unique(np.concatenate([np.linspace(-self.stem, L, n), [0.0]]))   # s = 0 is a grid point
        ds = np.diff(sd, prepend=sd[0] - (sd[1] - sd[0]))
        s = np.maximum(sd, 0.0)
        stiff = STIFF_R * R
        u = np.clip((s - stiff) / max(L - stiff, 1e-9), 0.0, 1.0)
        pitch = np.radians(p.lift) * smoothstep(s / (3.0 * R))
        pitch = pitch - np.radians(DROOP_DEG * p.droop) * u ** DROOP_POW
        v = np.clip((s - CURL_FROM * L) / ((1.0 - CURL_FROM) * L), 0.0, 1.0)
        pitch = pitch + np.radians(CURL_DEG * p.curl) * v * v
        yaw = np.radians(p.sway) * u
        # the curvature limit: a ring of radius r bent by more than r * BEND_LIMIT per unit of
        # length folds the shell on its own inner side
        r = np.maximum(self.law(sd) * 0.92, 1e-6)
        dp, dw = np.diff(pitch), np.diff(yaw)
        mag = np.hypot(dp, dw)
        lim = BEND_LIMIT * ds[1:] / r[1:]
        k = np.where(mag > lim, lim / np.maximum(mag, 1e-12), 1.0)
        pitch = np.concatenate([[pitch[0]], pitch[0] + np.cumsum(dp * k)])
        yaw = np.concatenate([[yaw[0]], yaw[0] + np.cumsum(dw * k)])
        tang = np.stack([np.sin(pitch), np.cos(pitch) * np.sin(yaw),
                         np.cos(pitch) * np.cos(yaw)], axis=1)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True)
        pos = np.zeros_like(tang)
        pos[1:] = np.cumsum(0.5 * (tang[1:] + tang[:-1]) * ds[1:, None], axis=0)
        pos -= np.array([np.interp(0.0, sd, pos[:, c]) for c in range(3)])
        self._sd, self._pos, self._tang, self._pitch, self._yaw = sd, pos, tang, pitch, yaw

    def centre(self, s):
        s = np.asarray(s, dtype=np.float64)
        return np.stack([np.interp(s, self._sd, self._pos[:, c]) for c in range(3)], axis=-1)

    def tangent(self, s):
        s = np.asarray(s, dtype=np.float64)
        t = np.stack([np.interp(s, self._sd, self._tang[:, c]) for c in range(3)], axis=-1)
        return t / np.linalg.norm(t, axis=-1, keepdims=True)

    def frames(self, s):
        """Ring frames at the (ascending) stations ``s``: ``(n, 3, 3)`` with columns (up,
        lateral, tangent), carried along the path by parallel transport (the rings do not spin
        about it)."""
        s = np.asarray(s, dtype=np.float64)
        tang = self.tangent(s)
        out = np.empty((len(s), 3, 3))
        b = np.eye(3)
        out[0] = A._transport(b, np.array([0.0, 0.0, 1.0]), tang[0])
        for i in range(1, len(s)):
            out[i] = A._transport(out[i - 1], tang[i - 1], tang[i])
        return out

    # -- the stations ------------------------------------------------------------------------

    def _stations(self):
        R, L = self.R, self.L
        # a ring every 0.4 R from the buried end to 4 R, then ``segments`` evenly to the tip
        root_end = min(4.0 * R, 0.5 * L)
        k = max(int(math.ceil((root_end + self.stem) / (0.4 * R))), 2)
        s_root = np.linspace(-self.stem, root_end, k + 1)
        n_free = max(int(self.p.segments), 2)
        s_free = np.linspace(root_end, L, n_free + 1)[1:]
        self.s = np.concatenate([s_root, s_free])
        self.fr = self.frames(self.s)

    # -- the section ---------------------------------------------------------------------------

    def section(self, s, layer: str = "skin", n: int | None = None, caps: bool = True):
        """The ring of ``layer`` ('skin' or 'muscle') at each station ``s``: ``(len(s), n, 3)``
        in the local frame. ``skin`` is the muscle surface moved out by the skin thickness
        (with the lobing damped); ``muscle`` is the lobed sleeve."""
        n = self.p.ring_verts if n is None else n
        s = np.asarray(s, dtype=np.float64)
        th = 2.0 * np.pi * np.arange(n) / n
        b = self.law(s)[:, None]                      # vertical semi-axis of the skin
        a = b * self.aspect(s)[:, None]
        t = self.skin_thickness(s)[:, None]
        g = self.belly(s)[:, None]
        lob = LOBE * g * self.lobes(th)[None, :]
        if layer == "muscle":
            ca, cb, scale = a - t, b - t, 1.0 + lob
        else:
            ca, cb, scale = a, b, 1.0 + DAMP * lob
        fr = self.frames(s) if len(s) != len(self.s) or not np.array_equal(s, self.s) else self.fr
        up, lat = fr[:, :, 0], fr[:, :, 1]
        crest = 0.0
        if self.p.crest > 0.0 and layer == "skin":
            # a dorsal ridge: a narrow raised line on the dorsal midline, from the root to 0.6 L
            ramp = (1.0 - np.clip(s / (0.6 * self.L), 0.0, 1.0))[:, None]
            crest = (self.p.crest * self.R * ramp * np.exp(-(np.angle(np.exp(1j * th)) / 0.3) ** 2)[None, :])
        x = (cb * scale + crest) * np.cos(th)[None, :]
        y = ca * scale * np.sin(th)[None, :]
        pts = (self.centre(s)[:, None, :] + x[:, :, None] * up[:, None, :]
               + y[:, :, None] * lat[:, None, :])
        return pts

    def _cap_rings(self, layer: str, n: int):
        """The rounded end: rings on a quarter sphere after the last station."""
        s_end = np.array([self.L])
        b = float(self.law(s_end)[0])
        a = b * float(self.aspect(s_end)[0])
        t = float(self.skin_thickness(s_end)[0])
        if layer == "muscle":
            a, b = a - t, b - t
        c = self.centre(s_end)[0]
        tg = self.tangent(s_end)[0]
        fr = self.fr[-1]
        th = 2.0 * np.pi * np.arange(n) / n
        rings = []
        for deg in TIP_CAP:
            ph = np.radians(deg)
            ring = (c + tg * b * math.sin(ph) + (b * math.cos(ph) * np.cos(th))[:, None] * fr[:, 0]
                    + (a * math.cos(ph) * np.sin(th))[:, None] * fr[:, 1])
            rings.append(ring)
        return rings

    # -- the layers ------------------------------------------------------------------------------

    def _loft(self, layer: str) -> Mesh:
        n = self.p.ring_verts
        s = self.s
        if layer == "muscle":
            # the buried end of the sleeve stops a skin thickness and a half short of the skin's own
            # (the two caps would be coplanar), so the layers nest strictly
            s = s.copy()
            s[0] += 1.5 * float(self.skin_thickness(s[0]))
        ring = self.section(s, layer, n)
        rings = [ring[i] for i in range(len(s))] + self._cap_rings(layer, n)
        return A._loft(rings)

    def muscle_belly(self) -> np.ndarray:
        """Per vertex of :meth:`muscle`: how much of the sleeve there is muscle belly (1) rather than
        tendon, fascia and fat (0): ring ``i`` has the belly fraction of its station, the rounded tip none,
        the buried cap's pole the root's. Lets a view colour the sleeve (``ecorche``)."""
        n = self.p.ring_verts
        g = np.concatenate([np.repeat(self.belly(self.s), n), np.zeros(len(TIP_CAP) * n)])
        return np.concatenate([g, [float(self.belly(self.s[0])), 0.0]])      # the two cap poles

    def skin(self) -> Mesh:
        """The shell that is printed and rendered (closed; the buried end is inside the body)."""
        return self._loft("skin")

    def muscle(self) -> Mesh:
        """The muscle sleeve (closed): the skin's inner surface, lobed where the bellies are."""
        return self._loft("muscle")

    # bone ---------------------------------------------------------------------------------------

    def vertebra_edges(self) -> np.ndarray:
        """The ``n + 1`` arc-length boundaries of the ``n`` caudal vertebrae (Cd1 first). Cd1
        starts one base radius below the skin point (the research: "Cd1 is ~1.0 R deep"); the chain
        ends just before the tip."""
        n = self.p.n_vertebrae()
        frac = (np.arange(n) + 0.5) / n
        w = np.interp(frac, *VERTEBRA_LENGTHS)
        start = -1.0 * self.R                       # Cd1's rostral end: the sacral apex is before it
        end = self.L - 0.5 * self.tip
        edges = start + (end - start) * np.concatenate([[0.0], np.cumsum(w / w.sum())])
        return edges

    def bone(self) -> Mesh:
        """The vertebra chain and the sacral apex it grows from, as one mesh of closed pieces."""
        pieces: list[Mesh] = []
        n = self.p.n_vertebrae()
        edges = self.vertebra_edges()
        # sample frames for every piece in one pass
        prof = np.array([0.0, 0.07, 0.2, 0.5, 0.8, 0.93, 1.0])
        mult = np.array([0.45, 0.92, 0.84, 0.78, 0.84, 0.92, 0.45])
        samples = []
        for i in range(n):
            s0, s1 = edges[i], edges[i + 1]
            gap = 0.04 * (s1 - s0)
            samples.append(np.concatenate([s0 + gap + (s1 - s0 - 2 * gap) * prof, [0.5 * (s0 + s1)]]))
        # the sacral apex: a wedge before Cd1
        a0 = max(edges[0] - 0.55 * self.R, -(self.stem - 0.1 * self.R))
        apex_s = np.linspace(a0, edges[0] - 0.05 * (edges[1] - edges[0]), 4)
        allS = np.concatenate([np.concatenate(samples), apex_s])
        order = np.argsort(allS)
        fr_sorted = self.frames(allS[order])
        fr = np.empty_like(fr_sorted)
        fr[order] = fr_sorted
        C = self.centre(allS)
        k = 0
        nv = 8
        for i in range(n):
            ss = samples[i]
            m = len(ss)
            F, Cc = fr[k:k + m], C[k:k + m]
            k += m
            rb = float(self.bone_radius(ss[-1]))
            rings = []
            for j in range(len(prof)):
                r = rb * mult[j]
                rings.append(_ellipse(Cc[j], r * 1.12, r, F[j], nv))
            pieces.append(A._loft(rings))
            mid_c, mid_f = Cc[-1], F[-1]
            # transverse processes on the first six (long on Cd1-3), a spine on the first four
            if i < 6:
                w = (0.65 * self.R) * (1.0 - 0.13 * i)
                for sign in (1.0, -1.0):
                    pieces.append(_spar(mid_c + sign * mid_f[:, 1] * 0.7 * rb,
                                        mid_c + sign * mid_f[:, 1] * (0.7 * rb + w),
                                        0.34 * rb, 0.12 * rb, mid_f[:, 0], mid_f[:, 2]))
            if i < 4:
                h = 0.25 * self.R * (1.0 - i / 4.0)
                pieces.append(_spar(mid_c + mid_f[:, 0] * 0.7 * rb,
                                    mid_c + mid_f[:, 0] * (0.7 * rb + h),
                                    0.30 * rb, 0.10 * rb, mid_f[:, 1], mid_f[:, 2]))
            # chevrons from Cd4 (index 3) to 0.55 of the chain, tapering to nothing
            last_chev = int(0.55 * n)
            if 3 <= i <= last_chev:
                ext = 0.9 * rb * (1.0 - (i - 3) / max(last_chev - 3 + 1, 1))
                pieces.append(_spar(mid_c - mid_f[:, 0] * 0.7 * rb,
                                    mid_c - mid_f[:, 0] * (0.7 * rb + ext),
                                    0.22 * rb, 0.08 * rb, mid_f[:, 1], mid_f[:, 2]))
        ca, fa = C[k:], fr[k:]
        rb0 = float(self.bone_radius(0.0))
        rings = [_ellipse(ca[j], rb0 * f * 1.2, rb0 * f, fa[j], nv)
                 for j, f in enumerate((0.55, 0.95, 1.25, 1.1))]
        pieces.append(A._loft(rings))
        return Mesh.merge(pieces)

    # -- numbers -------------------------------------------------------------------------------------

    def junction_chain(self, reach: float):
        """``(centres, radii)`` of the skin tube from the buried end to ``reach`` beyond the skin
        point: the radius is the INSCRIBED circle of the (elliptical, polygonal) section."""
        keep = self.s <= max(reach, self.s[2])
        s = self.s[keep]
        b = self.law(s)
        a = b * self.aspect(s)
        r = np.minimum(a, b) * math.cos(math.pi / self.p.ring_verts)
        return self.centre(s), r

    def min_bend_radius(self, upto: float | None = None) -> float:
        """The smallest radius of curvature of the centreline (units), over the first ``upto``
        of the free length (default all)."""
        sd = self._sd
        t = self._tang
        dth = np.linalg.norm(np.diff(t, axis=0), axis=1)
        ds = np.diff(sd)
        kappa = dth / np.maximum(ds, 1e-12)
        sel = np.ones(len(kappa), dtype=bool) if upto is None else sd[1:] <= upto
        k = float(kappa[sel].max()) if sel.any() else 0.0
        return float("inf") if k <= 1e-12 else 1.0 / k


def _ellipse(c, a, b, basis, n):
    """A ring about ``c``: semi-axis ``b`` along the first column of ``basis``, ``a`` along the second."""
    th = 2.0 * np.pi * np.arange(n) / n
    return c[None, :] + (b * np.cos(th))[:, None] * basis[:, 0][None, :] + (
        a * np.sin(th))[:, None] * basis[:, 1][None, :]


def _spar(c0, c1, r0, r1, perp_a, perp_b, n=6):
    """A small tapered closed tube from ``c0`` to ``c1`` (radii ``r0`` -> ``r1``): a transverse
    process, a neural spine or a chevron. ``perp_a`` / ``perp_b`` span the plane of its rings."""
    basis = np.stack([np.asarray(perp_a), np.asarray(perp_b), np.zeros(3)], axis=1)
    rings = []
    for f, r in ((0.0, r0), (0.5, 0.5 * (r0 + r1) * 1.0), (0.9, r1), (1.0, 0.55 * r1)):
        rings.append(_ellipse(c0 + (c1 - c0) * f, r, r, basis, n))
    return A._loft(rings)


# --------------------------------------------------------------------------
# a tail on a body


class PlacedTail:
    """A tail model with its placement: the local -> body frame (a 4x4 with a uniform scale) and
    the parameters it was built from. Models are cached per set of pose controls (a pose drag
    asks for a new one per frame; the junction and the layers ask for the same one again)."""

    def __init__(self, params: TailParams, frame, clamp=None):
        self.params = params
        self.frame = np.asarray(frame, dtype=np.float64)
        self.clamp = clamp                 # controls -> controls inside the kind's ranges
        self._models: dict = {}

    @property
    def scale(self) -> float:
        return float(abs(np.linalg.det(self.frame[:3, :3])) ** (1.0 / 3.0))

    def model(self, controls: dict | None = None) -> TailModel:
        c = {k: float(v) for k, v in (controls or {}).items()
             if k in ("curl", "droop", "lift", "sway")}
        if self.clamp is not None:
            c = self.clamp(c)
        key = tuple(sorted(c.items()))
        m = self._models.get(key)
        if m is None:
            if len(self._models) >= 24:
                self._models.pop(next(iter(self._models)))
            p = self.params
            m = self._models[key] = TailModel(p.with_(
                curl=p.curl + c.get("curl", 0.0), droop=p.droop + c.get("droop", 0.0),
                lift=c.get("lift", 0.0), sway=c.get("sway", 0.0)))
        return m

    def skin(self, controls=None) -> Mesh:
        return self.model(controls).skin().transform(self.frame)

    def layers(self, controls=None) -> dict:
        """``{"bone", "muscle", "skin"}`` meshes in the body's frame, from one model."""
        m = self.model(controls)
        return {"bone": m.bone().transform(self.frame), "muscle": m.muscle().transform(self.frame),
                "skin": m.skin().transform(self.frame)}


def junction_spec(name: str, model: TailModel, frame, fillet: float, height: float):
    """The :class:`~.junction.JunctionSpec` of a placed tail: how the body's skin moulds around
    its root. Built in the tail's local frame from the model (so it is the same on every body)
    and carried to the body by ``frame``:

    * the CENTRELINE is the skin tube from the buried end to 2 R beyond the skin point, with the
      inscribed radius of the (flared, elliptical) section: the skin flows into the tail's own
      trumpet;
    * a MOUND, the sacral rise: a ridge up the back (the erector spinae funnel into the dorsal
      tail), low and wide;
    * two MOUNDS that move the medial cheeks toward the axis (the gluteal masses wrap the root);
    * a KEEP sphere in the cleft below: the natal cleft and the perineum stay as they are.
    """
    R = model.R
    centres, radii = model.junction_chain(2.0 * R)
    spec = J.JunctionSpec(
        name, centres, radii * J.HUG, width=fillet, base_radius=R, height=height,
        mounds=(
            J.Mound((MOUND_UP[0] * R, 0.0, 0.0), MOUND_RADIUS * R, MOUND_HEIGHT * R,
                    end=(MOUND_UP[1] * R, 0.0, 0.0)),
            J.Mound((CHEEK_X * R, CHEEK_Y * R, 0.0), CHEEK_RADIUS * R, CHEEK_HEIGHT * R,
                    direction=(0.0, -1.0, 0.0)),
            J.Mound((CHEEK_X * R, -CHEEK_Y * R, 0.0), CHEEK_RADIUS * R, CHEEK_HEIGHT * R,
                    direction=(0.0, 1.0, 0.0)),
        ),
        keep=tuple(J.Keep((x * R, 0.0, z * R), KEEP_RADIUS * R, fade=KEEP_FADE) for x, z in KEEP_BEADS))
    return spec.transformed(frame)
