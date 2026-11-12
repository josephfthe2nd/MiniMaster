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
LIFT_FROM, LIFT_LEN = -1.5, 4.0   # the lift ramp (in R): the bend starts INSIDE the body, 1.5 R below the skin point, so the elbow is partly hidden
BEND_LIMIT = 0.8         # a ring's radius times the path curvature may not exceed this
SKIN_ROOT, SKIN_TIP, SKIN_FLOOR = 0.08, 0.12, 0.02    # skin thickness: x radius, root to tip; floor in units
BELLY_END = 0.55         # the muscle bellies end here (tendons beyond), as a fraction of the length
BELLY_FULL = 0.12        # ... and are at full size before here
LOBE = 0.12              # relative depth of the muscle lobing (peak to peak, dorsal and ventral pairs of bellies)
GROOVE_FILL = 0.0        # the skin fills the grooves between the bellies up to this lobe level (-0.5: not at all, +0.5: round)
# centrum radius / local skin radius at t = 0, .25, 1: calibrated on the cat skeleton plate
# (cm_Anatomical_technology: the bone is 0.98, 0.59, 0.50, 0.32, 0.19 of its root width at 25, 50, 75,
# 95 % and 100 % of the length), which is ~0.30-0.37 of the local radius all along (the research's
# guess of 0.40 / 0.55 distally made the bone 1.4-1.9 x too thick there)
BONE_FRAC = ((0.0, 0.25, 1.0), (0.30, 0.33, 0.33))
VERTEBRA_LENGTHS = ((0.0, 0.29, 1.0), (0.55, 1.0, 0.35))   # relative length at the chain fraction
TIP_CAP = (32.0, 62.0, 82.0)   # the rounded tip: rings on a quarter sphere (degrees)
# the sacrum the chain grows from (the last sacral vertebrae fused into a wedge that runs up the back under the
# skin, wide above, narrowing to the apex that carries Cd1): length, half widths top / apex, half thickness,
# depth of its centre below the skin at the apex / at the top (all in R)
SACRUM = dict(length=3.2, w_top=1.6, w_apex=0.8, thick=0.42, depth_apex=1.25, depth_top=0.85)


# the skin around the root (junction_spec): positions in base radii R in the local frame
# (x up the back, y lateral, z out of the body); see docs/anatomy_tail.md
R_MAX = 0.62                  # the junction's sizes (mounds, fillet, keeps) stop growing here: the base radius of a neutral-body tail
KEEP_BEADS = ((-1.1, -0.3), (-1.6, -0.8), (-1.9, -1.4), (-2.1, -2.0))   # the cleft below the root, as spheres on the midline (x, z in R)
KEEP_RADIUS, KEEP_FADE = 0.9, 0.6     # ... each this wide (R), and growing back to full over this much of it
REACH_R = 1.0                 # the skin tube reaches one fillet width + this many R past the skin point
MAX_MOVE_R = 1.4              # one cage vertex moves at most this many R (the default tail's largest move is 1.1 R)
SMOOTH = 5                    # rounds of smoothing of the tube displacement over the cage (the core's default is 20: it costs 4-7 degrees of the fillet's gain here)


@dataclass(frozen=True)
class RootMass:
    """A mass of muscle that lies in the body at the root of the tail, as the junction's mound and as a lens of the
    ``muscle`` layer: ONE description, so that the skin that is moulded over it is the envelope of the layer.

    ``center`` / ``half``: the footprint on the skin plane (x up the back, y lateral, in R): an ellipse, the mound's
    bump reaches 0 at its rim; ``height``: the thickness of tissue the mass adds ABOVE the skin the body already had
    (R); ``depth``: how far its underside reaches below that skin (R); ``fade``: the lift (degrees) over which a
    raised tail takes the mass back (``None``: never); ``muscle``: the anatomical name."""

    name: str
    center: tuple
    half: tuple
    height: float
    depth: float
    fade: tuple | None = None
    muscle: str = ""
    along_axis: bool = False          # the mound pushes along the root's axis (a midline mass over a furrow), not the normals

    def mirrored(self) -> "RootMass":
        return replace(self, name=self.name.replace("_l", "_r"), center=(self.center[0], -self.center[1]))


ROOT_MASSES = (
    # the dorsal caudal muscles (sacrocaudalis dorsalis, the caudal end of the multifidus) arise on the dorsal sacrum
    # and run onto the tail: a wedge on the midline whose crest is AT the top of the root and that thins out up the
    # sacrum. Centred on the root, only its falling (concave) flank shows above the tail, so the lumbar line runs down
    # a ramp onto the tail's top line; a mound centred up the back (2.4 R, the first try) made a convex hump with the
    # tail leaving from under it (the critics' "ledge", measured: tests/test_tail_junction.py)
    RootMass("sacral", (1.0, 0.0), (3.5, 1.6), 0.60, 1.0, fade=(20.0, 80.0), muscle="sacrocaudalis dorsalis / multifidus",
             along_axis=True),
    # the gluteal masses that flank the root (gluteus maximus caudal fibres insert on the sacrum and coccyx)
    RootMass("gluteal_l", (-0.3, 2.2), (1.6, 1.6), 0.25, 0.9, muscle="gluteus maximus (caudal fibres)"),
    RootMass("gluteal_r", (-0.3, -2.2), (1.6, 1.6), 0.25, 0.9, muscle="gluteus maximus (caudal fibres)"),
)


def bump(u):
    """The mound's influence: ``1 - 3u^2 + 2u^3`` of ``u`` = distance / radius (0 from 1 on)."""
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    return 1.0 - 3.0 * u * u + 2.0 * u ** 3


def keep_weight(x, y, z, k: float = 1.0):
    """The cleft beads' hold at local points (in R; ``k`` = R_j / R): 0 inside a bead, back to 1 over ``KEEP_FADE`` of
    its radius beyond it (:func:`junction._keep_weight` on the beads of :func:`junction_spec`)."""
    P = np.stack(np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (x, y, z))), axis=-1)
    w = np.ones(P.shape[:-1])
    for bx, bz in KEEP_BEADS:
        r = KEEP_RADIUS * k
        d = np.linalg.norm(P - np.array([bx * k, 0.0, bz * k]), axis=-1)
        w = np.minimum(w, smoothstep((d - r) / (KEEP_FADE * r)))
    return w


def lift_weight(fade, lift: float) -> float:
    """1 at rest, 0 once the tail is lifted past ``fade[1]`` degrees (a smooth fall from ``fade[0]``)."""
    if fade is None:
        return 1.0
    return float(1.0 - smoothstep((abs(lift) - fade[0]) / max(fade[1] - fade[0], 1e-9))) if lift > 0 else 1.0


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
    ring_verts: int = 24
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

    def __init__(self, params: TailParams, patch: "SkinPatch | None" = None):
        self.p = params
        p = params
        self.R, self.L = float(p.base_radius), float(p.length)
        self._patch = patch               # a SkinPatch, a function that makes one (sampled on first use) or None (flat)
        self.tip = p.tip
        self.stem = p.stem
        self._path()
        self._stations()

    @property
    def patch(self) -> "SkinPatch":
        """The body's skin over the root (only the body-side layers need it: sampled on first use)."""
        if callable(self._patch):
            self._patch = self._patch()
        if self._patch is None:
            self._patch = SkinPatch.flat(self.R)
        return self._patch

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
        pitch = np.radians(p.lift) * smoothstep((sd - LIFT_FROM * R) / (LIFT_LEN * R))
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
        in the local frame.

        The MUSCLE is the primary shape: the section the radius law leaves inside the skin's
        thickness, lobed where the bellies are (``LOBE``). The SKIN is its envelope plus the
        thickness: it follows the crest of every belly and fills the grooves between them (up
        to ``GROOVE_FILL``), so it never lies closer than the thickness to the muscle at any
        vertex and lies exactly one thickness off it on the crests."""
        n = self.p.ring_verts if n is None else n
        s = np.asarray(s, dtype=np.float64)
        th = 2.0 * np.pi * np.arange(n) / n
        b = self.law(s)[:, None]                      # vertical semi-axis of the skin
        a = b * self.aspect(s)[:, None]
        t = self.skin_thickness(s)[:, None]
        g = self.belly(s)[:, None]
        lob = LOBE * g * self.lobes(th)[None, :]
        ca, cb = a - t, b - t                         # the muscle's mean semi-axes
        if layer == "muscle":
            fa, fb = ca * (1.0 + lob), cb * (1.0 + lob)
        else:
            env = LOBE * g * np.maximum(self.lobes(th)[None, :], GROOVE_FILL)
            fa, fb = ca * (1.0 + env) + t, cb * (1.0 + env) + t
        fr = self.frames(s) if len(s) != len(self.s) or not np.array_equal(s, self.s) else self.fr
        up, lat = fr[:, :, 0], fr[:, :, 1]
        crest = 0.0
        if self.p.crest > 0.0 and layer == "skin":
            # a dorsal ridge: a narrow raised line on the dorsal midline, from the root to 0.6 L
            ramp = (1.0 - np.clip(s / (0.6 * self.L), 0.0, 1.0))[:, None]
            crest = (self.p.crest * self.R * ramp * np.exp(-(np.angle(np.exp(1j * th)) / 0.3) ** 2)[None, :])
        x = (fb + crest) * np.cos(th)[None, :]
        y = fa * np.sin(th)[None, :]
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

    def mass_weight(self, mass: "RootMass", x, y, z):
        """How much of ``mass`` the junction puts on the skin point ``(x, y, z)`` (local frame, in R): the mound's bump
        of the SAME distance :func:`junction._mound_displacement` uses (3-D, from the mound's centre on the skin plane,
        stretched along the back), times the cleft beads' hold (:func:`keep_weight`) and the lift fade. The lens of
        the muscle layer and the mound of the spec are one description through this."""
        k = min(self.R, R_MAX) / self.R                      # the junction's sizes are in R_j = min(R, R_MAX)
        cx, cy = mass.center[0] * k, mass.center[1] * k
        hx, hy = mass.half[0] * k, mass.half[1] * k
        x, y, z = (np.asarray(a, dtype=np.float64) for a in (x, y, z))
        u = np.sqrt(((x - cx) / hx) ** 2 + ((y - cy) ** 2 + z ** 2) / hy ** 2)
        return bump(u) * keep_weight(x, y, z, k) * lift_weight(mass.fade, self.p.lift)

    def lens(self, mass: "RootMass", n_ang: int = 24, n_rad: int = 6) -> Mesh:
        """The body-side mass ``mass`` as a closed lens on the body's own skin (:class:`SkinPatch`), in the local
        frame. Its top is the skin the body had plus what the mass adds (``height`` times :meth:`mass_weight`: the
        mound's bump, held in the cleft, faded as the tail lifts) minus the skin thickness: the junction raises the
        skin by that, so the skin is the lens plus one thickness where nothing else moulds it (next to the tail the
        fillet adds to it). Its underside reaches ``depth`` below. Both meet on the rim."""
        R = self.R
        k = min(R, R_MAX) / R
        h, dep = mass.height * k, mass.depth * k
        t = float(self.skin_thickness(0.0)) / R
        th = 2.0 * np.pi * np.arange(n_ang) / n_ang
        top_rows, bot_rows = [], []
        for j in range(1, n_rad + 1):
            r = j / n_rad
            x = (mass.center[0] + mass.half[0] * r * np.cos(th)) * k
            y = (mass.center[1] + mass.half[1] * r * np.sin(th)) * k
            raw = self.patch.height(x, y)
            top = raw + h * self.mass_weight(mass, x, y, raw) - t
            bot = raw - t - dep * float(bump(r)) if j < n_rad else top
            top_rows.append(np.stack([x, y, top], axis=1) * R)
            bot_rows.append(np.stack([x, y, bot], axis=1) * R)
        cx, cy = mass.center[0] * k, mass.center[1] * k
        raw0 = float(self.patch.height(cx, cy))
        c_top = np.array([cx, cy, raw0 + h * float(self.mass_weight(mass, cx, cy, raw0)) - t]) * R
        c_bot = np.array([cx, cy, raw0 - t - dep]) * R
        v_top = np.vstack([c_top[None]] + top_rows)                       # 1 + n_rad * n_ang
        v_bot = np.vstack([c_bot[None]] + bot_rows[:-1])                  # the rim ring is shared with the top
        nt = len(v_top)
        verts = np.vstack([v_top, v_bot])

        def top_id(j, k):                    # ring j = 1..n_rad
            return 1 + (j - 1) * n_ang + (k % n_ang)

        def bot_id(j, k):
            return top_id(j, k) if j == n_rad else nt + 1 + (j - 1) * n_ang + (k % n_ang)
        faces = []
        for k in range(n_ang):
            faces.append([0, top_id(1, k), top_id(1, k + 1)])                 # counter-clockwise seen from +z: faces out
            faces.append([nt, bot_id(1, k + 1), bot_id(1, k)])
        for j in range(1, n_rad):
            for k in range(n_ang):
                a, b, c, d = top_id(j, k), top_id(j, k + 1), top_id(j + 1, k + 1), top_id(j + 1, k)
                faces += [[a, c, b], [a, d, c]]
                a, b, c, d = bot_id(j, k), bot_id(j, k + 1), bot_id(j + 1, k + 1), bot_id(j + 1, k)
                faces += [[a, b, c], [a, c, d]]
        mesh = Mesh(verts, np.asarray(faces, dtype=np.int64))
        if mesh.volume() < 0:
            mesh = Mesh(mesh.vertices, mesh.faces[:, [0, 2, 1]])
        return mesh

    def lenses(self) -> list:
        """The body-side masses of the muscle layer, one closed lens each (:class:`RootMass`)."""
        return [self.lens(m) for m in ROOT_MASSES]

    def muscle_whole(self):
        """``(mesh, belly)``: the muscle layer as the caudal sleeve plus the body-side lenses (one mesh of closed
        pieces) and the per-vertex belly weight that colours it (the lenses are all muscle: 1)."""
        sleeve = self.muscle()
        parts = [sleeve] + self.lenses()
        belly = np.concatenate([self.muscle_belly()] + [np.ones(len(m.vertices)) for m in parts[1:]])
        return Mesh.merge(parts), belly

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
        """The whole bone layer: the caudal chain, the sacral apex it grows from and the sacral wedge above (one
        mesh of closed pieces)."""
        return Mesh.merge([self.bone_chain(), self.sacrum()])

    def bone_chain(self) -> Mesh:
        """The vertebra chain and the sacral apex it grows from, as one mesh of closed pieces: what the
        caudal muscle sleeve encloses (the sacrum wedge is beside it, under the body's skin)."""
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

    def sacrum(self, n: int = 12, rings: int = 9) -> Mesh:
        """The fused last sacral vertebrae as a wedge: from the apex (where the first caudal vertebra articulates)
        up the back under the skin, following the body's own skin (:class:`SkinPatch`) at a depth, widening
        upward. Local frame; a closed solid."""
        R, S = self.R, SACRUM
        u = np.linspace(0.0, 1.0, rings)
        x = -0.35 + (S["length"] + 0.35) * u
        depth = S["depth_apex"] + (S["depth_top"] - S["depth_apex"]) * u
        zc = self.patch.height(x, np.zeros_like(x)) - depth
        wy = S["w_apex"] + (S["w_top"] - S["w_apex"]) * smoothstep(u)
        wy = wy * (0.55 + 0.45 * np.sin(np.pi * 0.5 * np.minimum(u * 6.0, 1.0)))        # the blunt apex
        th = 2.0 * np.pi * np.arange(n) / n
        out = []
        for i in range(rings):
            ring = np.stack([np.full(n, x[i]), wy[i] * np.cos(th), zc[i] + S["thick"] * np.sin(th)], axis=1) * R
            out.append(ring)
        return A._loft(out)

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
# the skin the body already has where the tail roots (the mass lenses lie on it)


class SkinPatch:
    """The body's own skin over the root, as a height field in the tail's local frame: ``z(x, y)`` (all in base radii R,
    z out of the body, the skin point at 0) on a regular grid of ``x`` (up the back) and ``y`` (lateral),
    sampled ONCE when the tail is placed from the cage. The mass lenses of the muscle layer lie on it; the
    junction never reads it (a spec is the same on every body)."""

    STEP = 0.5            # grid step, in base radii
    X_RANGE = (-2.0, 9.0)
    Y_RANGE = (-4.0, 4.0)

    def __init__(self, xs, ys, z, R: float):
        self.xs, self.ys, self.z, self.R = np.asarray(xs), np.asarray(ys), np.asarray(z), float(R)

    @classmethod
    def flat(cls, R: float) -> "SkinPatch":
        """The plane z = 0 (a tail placed without a body)."""
        xs = np.arange(cls.X_RANGE[0], cls.X_RANGE[1] + 1e-9, cls.STEP)
        ys = np.arange(cls.Y_RANGE[0], cls.Y_RANGE[1] + 1e-9, cls.STEP)
        return cls(xs, ys, np.zeros((len(xs), len(ys))), R)

    @classmethod
    def from_cage(cls, verts, quads, frame, R: float) -> "SkinPatch":
        """Sample the cage's outermost surface over the grid. ``frame`` is the local -> body matrix, ``R`` the base
        radius in local units. A cell the cage does not cover (the closed cleft) takes its nearest covered
        neighbour's height."""
        xs = np.arange(cls.X_RANGE[0], cls.X_RANGE[1] + 1e-9, cls.STEP)
        ys = np.arange(cls.Y_RANGE[0], cls.Y_RANGE[1] + 1e-9, cls.STEP)
        inv = np.linalg.inv(np.asarray(frame, dtype=np.float64))
        loc = (np.asarray(verts) @ inv[:3, :3].T + inv[:3, 3]) / R
        tri = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
        c = loc[tri].mean(axis=1)
        near = ((c[:, 0] > cls.X_RANGE[0] - 1.5) & (c[:, 0] < cls.X_RANGE[1] + 1.5)
                & (np.abs(c[:, 1]) < cls.Y_RANGE[1] + 1.5) & (np.abs(c[:, 2]) < 5.0))
        tri = tri[near]
        z = np.full((len(xs), len(ys)), -np.inf)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        for t in tri:
            a, b, d = loc[t[0]], loc[t[1]], loc[t[2]]
            x0, x1 = min(a[0], b[0], d[0]), max(a[0], b[0], d[0])
            y0, y1 = min(a[1], b[1], d[1]), max(a[1], b[1], d[1])
            i0, i1 = np.searchsorted(xs, x0 - 1e-9), np.searchsorted(xs, x1 + 1e-9)
            j0, j1 = np.searchsorted(ys, y0 - 1e-9), np.searchsorted(ys, y1 + 1e-9)
            if i0 >= i1 or j0 >= j1:
                continue
            px, py = gx[i0:i1, j0:j1], gy[i0:i1, j0:j1]
            den = (b[1] - d[1]) * (a[0] - d[0]) + (d[0] - b[0]) * (a[1] - d[1])
            if abs(den) < 1e-12:
                continue
            l1 = ((b[1] - d[1]) * (px - d[0]) + (d[0] - b[0]) * (py - d[1])) / den
            l2 = ((d[1] - a[1]) * (px - d[0]) + (a[0] - d[0]) * (py - d[1])) / den
            l3 = 1.0 - l1 - l2
            hit = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
            if hit.any():
                zz = np.where(hit, l1 * a[2] + l2 * b[2] + l3 * d[2], -np.inf)
                z[i0:i1, j0:j1] = np.maximum(z[i0:i1, j0:j1], zz)
        bad = ~np.isfinite(z)
        if bad.all():
            return cls.flat(R)
        for _ in range(12):                          # fill what no triangle covered from the neighbours
            if not bad.any():
                break
            ok = ~bad
            pad = np.pad(np.where(ok, z, 0.0), 1)
            cnt = np.pad(ok.astype(float), 1)
            tot = pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
            num = cnt[:-2, 1:-1] + cnt[2:, 1:-1] + cnt[1:-1, :-2] + cnt[1:-1, 2:]
            fill = bad & (num > 0)
            z[fill] = tot[fill] / num[fill]
            bad = ~np.isfinite(z)
        z[~np.isfinite(z)] = 0.0
        return cls(xs, ys, z, R)

    def height(self, x, y):
        """Bilinear height at the local ``x``, ``y`` (in R); clamped to the grid."""
        x = np.clip(np.asarray(x, dtype=np.float64), self.xs[0], self.xs[-1])
        y = np.clip(np.asarray(y, dtype=np.float64), self.ys[0], self.ys[-1])
        fx = (x - self.xs[0]) / self.STEP
        fy = (y - self.ys[0]) / self.STEP
        i = np.clip(np.floor(fx).astype(int), 0, len(self.xs) - 2)
        j = np.clip(np.floor(fy).astype(int), 0, len(self.ys) - 2)
        u, v = fx - i, fy - j
        z = self.z
        return ((1 - u) * (1 - v) * z[i, j] + u * (1 - v) * z[i + 1, j]
                + (1 - u) * v * z[i, j + 1] + u * v * z[i + 1, j + 1])


# --------------------------------------------------------------------------
# a tail on a body


class PlacedTail:
    """A tail model with its placement: the local -> body frame (a 4x4 with a uniform scale) and
    the parameters it was built from. Models are cached per set of pose controls (a pose drag
    asks for a new one per frame; the junction and the layers ask for the same one again).
    ``patch``: the body's own skin over the root (:class:`SkinPatch`), which the mass lenses of the muscle layer
    lie on (a flat plane without one)."""

    def __init__(self, params: TailParams, frame, clamp=None, patch=None):
        self.params = params
        self.frame = np.asarray(frame, dtype=np.float64)
        self.clamp = clamp                 # controls -> controls inside the kind's ranges
        self._patch = patch                # SkinPatch, a function that makes one (lazy: only the layers need it) or None
        self._models: dict = {}

    def _patch_once(self) -> "SkinPatch":
        if callable(self._patch):
            self._patch = self._patch()
        if self._patch is None:
            self._patch = SkinPatch.flat(self.params.base_radius)
        return self._patch

    @property
    def patch(self) -> "SkinPatch":
        return self._patch_once()

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
                lift=c.get("lift", 0.0), sway=c.get("sway", 0.0)), self._patch_once)
        return m

    def skin(self, controls=None) -> Mesh:
        return self.model(controls).skin().transform(self.frame)

    def muscle_with_belly(self, controls=None):
        """``(mesh in the body's frame, per-vertex belly weight)`` of the whole muscle layer: the caudal
        sleeve and the body-side mass lenses (the lenses are all muscle: weight 1)."""
        m = self.model(controls)
        mesh, belly = m.muscle_whole()
        return mesh.transform(self.frame), belly

    def layers(self, controls=None) -> dict:
        """``{"bone", "muscle", "skin"}`` meshes in the body's frame, from one model."""
        m = self.model(controls)
        return {"bone": m.bone().transform(self.frame), "muscle": self.muscle_with_belly(controls)[0],
                "skin": m.skin().transform(self.frame)}


def junction_spec(name: str, model: TailModel, frame, fillet: float, height: float,
                  owner: str | None = "trunk", r_max: float = R_MAX):
    """The :class:`~.junction.JunctionSpec` of a placed tail: how the body's skin moulds around
    its root. Built in the tail's local frame from the model (so it is the same on every body)
    and carried to the body by ``frame``:

    * the CENTRELINE is the skin tube from the buried end to one fillet width + 1 R beyond the skin
      point, with the inscribed radius of the (flared, elliptical) section: the skin flows into the
      tail's own trumpet;
    * one MOUND for every :class:`RootMass` (the sacral wedge up the back, the two gluteal masses
      that flank the root), sized by the SAME numbers as the lenses of the muscle layer, and the
      sacral wedge taken back as the tail is lifted (it lies in the tail's own way then);
    * a row of KEEP spheres down the cleft below: the natal cleft and the perineum stay as they are.

    Every size is in ``R_j = min(R, r_max)``: a thicker tail than the body can carry moulds the
    skin as a tail of ``r_max`` would (the tube itself keeps its radius), so the junction's reach
    and displacement never grow with the base radius (the shredded belt of the first version).
    """
    R = model.R
    Rj = min(R, float(r_max))
    centres, radii = model.junction_chain((fillet + REACH_R) * Rj)
    mounds = []
    for m in ROOT_MASSES:
        w = lift_weight(m.fade, model.p.lift)
        if w <= 1e-3:
            continue
        mounds.append(mass_mound(m, Rj, w))
    spec = J.JunctionSpec(
        name, centres, radii * J.HUG, width=fillet, base_radius=Rj, height=height, owner=owner,
        tight_gap=0.0, smooth=SMOOTH, max_move=MAX_MOVE_R * Rj, mounds=tuple(mounds), join_guard=False,
        keep=tuple(J.Keep((x * Rj, 0.0, z * Rj), KEEP_RADIUS * Rj, fade=KEEP_FADE) for x, z in KEEP_BEADS))
    return spec.transformed(frame)


def mass_mound(m: RootMass, R: float, weight: float = 1.0):
    """The :class:`~.junction.Mound` of a root mass (local frame, ``R`` the base radius in local units): the bump the
    muscle adds to the skin the body already has, an ellipse of the mass's footprint, pushed OUT of the body along the
    root's own axis (local z), not along each vertex's normal: the lens of the muscle layer is a height over the skin
    plane, and a mound along the normals leaves a furrow as deep as it was (the heavy body's sacral groove between the
    erector masses: 0.1 R of a 0.6 R wedge got through) while the lens over it shows through the skin."""
    rx, ry = m.half
    kw = dict(direction=(0.0, 0.0, 1.0)) if m.along_axis else {}
    if abs(rx - ry) >= 1e-9:
        kw.update(axis=(1.0, 0.0, 0.0), stretch=rx / ry)
    return J.Mound((m.center[0] * R, m.center[1] * R, 0.0), ry * R, m.height * R * weight, **kw)
