"""The horn as an anatomy: a bony core on the frontal bone, the soft tissue that tents the scalp
around it, and the keratin sheath over it.

A horn is not a cone pushed into the scalp. Every cranial protrusion that real animals (and the
few humans) carry is the same three-layer arrangement, and this module builds the three layers
from ONE set of numbers, so the surface that is printed and rendered is derived from the layers
below it, never a second, unrelated shape:

* ``bone``   the cornual process of the frontal bone: a cone about 0.8 R at the skin, tapering over
             the first 0.62 of the length (the last 0.4 of a bovid horn is solid keratin) and
             continuous with the skull below the skin (the pedestal);
* ``muscle`` the soft tissue between the skull and the skin that the pedestal lifts into a tent
             (scalp, galea, periosteum and the horn's germinal cuff): a lens on the head, thickest
             at the horn and falling to the ordinary scalp thickness at ``rho_m`` (= ``tent_radius``);
             its upper surface is the moulded skin LESS the skin thickness, its lower one the skull's
             outer table;
* ``skin``   the keratin sheath (the core is derived from it, never nearer the surface than the
             keratin wall), the shell that is printed and rendered (the scalp around it is the body itself, moulded by the junction spec this
             module also produces so that it rises to the numbers of the muscle layer).

THE LOCAL FRAME (the one :class:`~minimaster.character.kinds.HornKind` places on the head) is
``z`` = the axis of the horn's first part (out of the skull), ``x`` = the direction the horn bends
toward, ``y = z cross x``; ``s`` runs along the axis from the skin point of the root (negative inside
the head: the hidden stem). The units are those of the horn parameters (``length``,
``base_radius``, at the neutral head); the kind scales the mesh to the head (the eye distance over
the neutral one).

THE NUMBERS (``docs/anatomy_horn.md`` says where each one comes from and which are guesses):

* the sheath radius law is ``r(s) = tip + (R - tip) (1 - s/L)^p`` with ``p`` 1.05 (a cone, not a thorn)
  times a periople-like root flare ``1 + F exp(-s / 0.35 R)`` and growth rings (annuli) of
  amplitude ``rings`` x r, pitch 0.8 R proximally and 0.55 of that at the tip;
* the section is an ellipse whose major axis lies in the bend plane (minor : major 0.85 for the
  scimitar, 0.80 for the ram with a 3-lobe keel, 0.9 for the antelope, 1 for the corkscrew);
* the centreline has the curvature of an equiangular spiral, ``kappa(s) = k / (S0 - s)``: it is
  non-zero at the root and rises toward the tip, for a total turn ``Psi = curve L / 2`` (the
  first versions' ``curve`` keeps its meaning); a lyre changes sign; a corkscrew adds a helical
  term; ``twist`` turns the bend plane;
* the root has a FOOT instead of a flat cap: rings that follow the moulded skin the sheath comes out of (lowered
  to it along the axis), a flare along the skin, a groove and a bead at the skin line, and a rounded bowl
  under the scalp (``HornModel.foot_rings``); the bone is the same foot and bowl inset, so it is inside the sheath
  everywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from ..core.mesh import Mesh
from . import appendages as A
from . import junction as J

# --------------------------------------------------------------------------
# constants (every one is documented in docs/anatomy_horn.md)

IOD_REF = 0.582          # the inter-ocular distance of the neutral head (no longer the scale: see SKULL_REF)
SKULL_REF = 0.6933       # the skull's half-width of the neutral head (macro defaults) at SKULL_AT: horn sizes are at this head
SKULL_AT = 0.6           # ... measured in a slab at this fraction of the way skull base -> crown (above the ears and the eyes)
SKULL_SLAB = 0.04        # ... of this half-thickness (a share of that height)
TIP_FLOOR = 0.012        # the sheath's tip radius is at least this (a point is not printable)
TAPER_P = 1.05           # the exponent of the sheath's taper (1.25 made a thorn)
FLARE_LEN = 0.35         # decay length of the root flare, in root radii
SINK_FLOOR = 0.3         # the stem reaches this many base radii under the skin even for a horn that stands square
SINK_TILT = 0.9          # ... plus this x the root's radius x tan(tilt): the skin falls away on the far side
FLARE_MAX = 0.6          # the root flare is limited to this (a flare of 1.5 is a trumpet 2.5 radii wide)
CAP_DEPTH = 0.5          # the buried end is a rounded cap this many root radii deep
CORE_BASE = 0.82         # the cornual process' radius at the skin, in base radii
CORE_MIN_WALL = 0.12     # the keratin wall is at least this thick (base radii) over the core
CORE_FLARE = 0.6         # the cornual process flares into the skull's outer table along a concave quarter circle of this
                         # many base radii (A05: the goat's core grows out of the frontal bone with no step; round 3's
                         # pedestal, 1.12 R wide under the skin, stepped to the cone at the foot's top like a pipe reducer)
RING_PITCH = 0.8         # annulus pitch at the root, in base radii
RING_PITCH_TIP = 0.55    # ... falling to this fraction of it at the tip
RING_START = 0.25        # no ring below this many base radii over the skin point (the root collar's top edge lies there)
RING_FIRST = 0.9         # the first annulus is this many base radii above the skin
RING_PER_PERIOD = 6      # rings per annulus period (the loft must resolve the ridge)
TIP_CAP = (32.0, 62.0, 82.0)    # the rounded end: rings on a quarter sphere (degrees)
# the sheath's foot (see ``HornModel.skin``): rings at these heights above the LOCAL skin (in root
# radii), the first one buried; a groove and a bead (the keratin's crisp rim) sit on them
# (round 2: denser near the skin, where the root collar's top edge meets the sheath, and shorter: the foot's tilted
# part above the collar showed as a crease)
FOOT_HEIGHTS = (-0.25, 0.0, 0.08, 0.16, 0.24, 0.34, 0.46, 0.62, 0.8)
BOWL_RINGS = (22.0, 45.0, 66.0, 82.0)    # the buried bowl under the collar (degrees on a quarter ellipse)
FOOT_GAP = 0.02          # consecutive foot rings are at least this many root radii apart along the axis
FOOT_MIN = 0.3           # a compressed foot keeps at least this share of its height
FOOT_TURN = 15.0         # degrees: the path turns at most this much over the foot (it is compressed otherwise)
TUBE_DEPTH = 0.9         # the junction's tube starts this many sinks under the skin point
BOWL_DEPTH = 0.5         # the bowl under the stem is this many root radii deep
STEM_DROP = 0.1          # the stem's level ring is this many root radii under the lowest point of the first foot ring
BEND_LIMIT = 0.7         # a ring's (largest) radius times the path curvature may not exceed this (0.8 on the mean
                         # radius let a ram's keel and rings cross on the inner side of a tight curl)
HELIX_RHO = 1.5          # a corkscrew's circle has this many local radii (before the bend limit)
HELIX_RAMP = 0.45        # the corkscrew grows out of the axis over this share of the length
HELIX_START = 1.2        # ... starting this many base radii up the shaft
HELIX_BEND = 0.8         # a corkscrew's path bends at most this share of the bend limit (the twisted ridged sheath of a
                         # ram with 4 turns crossed itself at 0.95)
HELIX_ITER = 6           # rounds of taking the helix in where the corkscrew bends too hard
PITCH_MIN = 5.0          # a corkscrew's coil is at least this many base radii long: a helix of pitch P bends at most 1 / (2 P / 2 pi), and below 4 r that is more than the bend limit for SOME helix radius, which the corkscrew passes through as it grows out of the root (and closer coils of the tube meet)
LYRE_PSI_MAX = 0.5 * math.pi  # a lyre turns out at most this much (radians), and back 0.7 of it: more made a loop
PSI_MAX = 1.75 * math.pi # the planar spiral turns at most this much (radians): past it the log spiral meets its first coil
HUG = 0.97               # the skin hugs this fraction of the (inscribed) sheath radius
HALFWIDTH_CAP = 0.40     # the base radius is at most this share of the head's half-width at that height
TILT_MAX = 45.0          # degrees: the most a horn leans off the skull's normal (round 4: the slider went to 60, where the
                         # low side of the sheath lay along the scalp as a keratin tongue and the skin's cuff split in two;
                         # the research's rule for a horn that reads as grown out of the bone is 15 to 45)
RING_VERTS_MIN = 24      # the sheath's polygon has at least this many sides (round 4: the slider went down to 5; a hexagonal
                         # sheath tore the skin's cuff into teeth at its corners, and a 16-gon's facets poked through it)
RING_VERTS_MAX = 48

# the skin around the root (junction_spec): in base radii
TENT_RADIUS = 2.2        # rho_m: the tent's radius
TENT_FLOOR = 0.36        # ... at least three mean cage edges (the cage is 0.125 an edge)
TENT_MARGIN = 0.075      # the tent stays this far from the midline
TENT_HEIGHT = 0.35       # H of the tent (base radii); a ram's is TENT_HEIGHT_RAM
TENT_HEIGHT_RAM = 0.4      # (round 2: 0.5; with the ram's own head bound, 0.12, a taller tent was cut into a step the validity pass backed off)
MIDLINE_RADII = 1.6      # the base radius is at most the distance from the midline less TENT_MARGIN over this
                         # (2.2, the tent's own radius, made the horns of a narrow-browed random face 0.54 of their size)
MIN_FILLET = 0.3         # ... and the fillet never narrower than this many base radii (by the midline)
FILLET_MAX = 1.5         # the junction's fillet is at most this many base radii wide on a head
TENT_MIN = 0.10          # ... and at least this (units): the cage resolves nothing smaller
COLLAR_SINK = 0.2        # the root collar's underside is sunk this many root radii under the skin (collar.py)
HEAD_SILHOUETTE = None   # the junction's bound on how far a horn's spec may push the head's skin out: None = the core's
                         # head default (8 % of the distance from the head's bone); the ram sets its own (SHAPES), whose
                         # root on the temple crest the 8 % cut into a step the validity pass backed off
SKIN_THICK = 0.06        # scalp skin thickness, units (0.6 cm; 3-8 mm)
SCALP_THICK = 0.065      # galea + periosteum + skin over the outer table, units (6.7 mm)
PLATE_GAP = 0.004        # the skull plate lies this far under the soft tissue (two coincident surfaces z-fight in a section view)


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


# the fillet foot of a skin-covered shape (the ossicone, round 4): see ``HornModel._fillet_foot``
FILLET_RHO = (0.2, 1.0)  # the fillet's radius is ``flare`` base radii, within these (A29: 0.5)
FILLET_STEP = 5.0        # degrees of the fillet's arc between two rings (the profile turns at most this much a ring)
FILLET_BLEND = 1.2       # base radii of the column above the fillet over which its rings go from following the skin
                         # to the planar rings of the shaft


def fillet_reach(rho: float, R: float, row: dict) -> float:
    """How far out of the column a fillet foot reaches under the skin (its deepest ring), local units."""
    g = math.radians(float(row.get("gamma", 12.0)))
    d1, d2 = (float(x) * R for x in row.get("dive", (0.06, 0.12)))
    f_g, h_g = rho * (1.0 - math.sin(g)), rho * (1.0 - math.cos(g))
    return f_g + (h_g + d1) / math.tan(g) + (d2 - d1) / math.tan(math.radians(40.0))


def fillet_rim(rho: float, row: dict):
    """``(f, h)`` of the point where a fillet foot of radius ``rho`` meets the skin it was built on: how far out of
    the column (``f``) and how high (``h`` = 0: on the skin), along the straight line that continues the arc where it
    is ``row["gamma"]`` degrees off the skin."""
    g = math.radians(float(row.get("gamma", 12.0)))
    f_g, h_g = rho * (1.0 - math.sin(g)), rho * (1.0 - math.cos(g))
    return f_g + h_g / math.tan(g), 0.0


# --------------------------------------------------------------------------
# the shapes

# One row per ``shape``: the defaults of the parameters and what the anatomy of that kind of horn
# says about its orientation (head frame: x out, y up, z forward; left side) and its section.
SHAPES = {
    "curved": dict(length=1.6, base_radius=0.17, curve=1.31, twist=0.0, turns=0.0, segments=28,
                   tip=0.06, profile="spiral", q=1.8, aspect=0.85, lobe=0.0, rings=0.05,
                   flare=0.0, tilt=27.0, lean=(-0.31, 0.80, -0.52), bend=(0.0, 0.0, -1.0),
                   core=0.62, material="bone", site="horn_crown", collar=(0.45, 0.3)),
    "ram": dict(length=3.0, base_radius=0.28, curve=3.2, twist=0.0, turns=0.0, segments=36,
                tip=0.07, profile="spiral", q=1.35, aspect=0.80, lobe=0.06, rings=0.06,
                flare=0.0, tilt=41.0, lean=(0.14, 0.49, -0.86), bend=(0.45, -0.65, -0.55),
                core=0.6, material="bone", site="horn_temple_crest", collar=(0.9, 0.3), ring_verts=32,
                silhouette=0.12, temporalis=True),
    # a markhor / kudu corkscrew (A01, A26): thick at the root, an open helix of 1.5 turns twice the local radius
    # wide (the first one, 4.0 x 0.13 with 2.25 turns, crossed itself and printed fragile)
    "spiral": dict(length=2.4, base_radius=0.17, curve=0.35, twist=0.0, turns=1.5, segments=48,
                   tip=0.055, profile="spiral", q=2.0, aspect=1.0, lobe=0.0, rings=0.03,
                   flare=0.0, tilt=30.0, lean=(-0.42, 0.83, -0.37), bend=(0.0, 0.0, -1.0),
                   core=0.62, material="bone", site="horn_crown", collar=(0.45, 0.3)),
    # (round 2: 3.4 x 0.16, tip 0.05; the first one, 4.0 x 0.13, printed fragile at "medium": 31% of its area under 0.3 mm)
    "antelope": dict(length=3.4, base_radius=0.16, curve=0.2, twist=0.0, turns=0.0, segments=40,
                     tip=0.055, profile="lyre", q=2.0, aspect=0.90, lobe=0.0, rings=0.04,
                     flare=0.0, tilt=40.0, lean=(-0.31, 0.80, -0.52), bend=(0.7, 0.0, -0.7),
                     core=0.62, material="bone", site="horn_crown", collar=(0.45, 0.3)),
    # a giraffe's ossicone: bone that stays under the skin and hair, so the whole surface is skin
    # material; it tapers by only a quarter, ends in a dome and meets the head in a concave fillet.
    # Round 4: ONE surface. The skin's cuff (collar.py) was a second shell whose top edge drew a pale scalloped lip round
    # the base and whose foot rings bunched into a tone band at 55 % of the height; skin over bone has no seam (A29, A30).
    # The fillet is the sheath's own foot (``foot="fillet"``): a concave quarter circle of ``flare`` base radii (A29:
    # 0.5) from the column down to the skin, which it meets at ``gamma`` degrees and then dives under; no collar.
    # Length 0.55 (0.45 read stubby beside A29 / the giraffe silhouette's H/W 1.5-1.75), tip 0.75 R (A29's taper
    # 0.94 / 0.88 / 0.82 of the column at 25 / 50 / 75 %), a finer dome, 40 facets (skin: no keratin facets to match)
    "ossicone": dict(length=0.55, base_radius=0.15, curve=0.2, twist=0.0, turns=0.0, segments=20,
                     tip=0.1125, profile="spiral", q=2.0, aspect=1.0, lobe=0.0, rings=0.0,
                     flare=0.5, tilt=20.0, lean=(-0.31, 0.80, -0.52), bend=(0.0, 0.0, -1.0),
                     core=1.0, material="skin", site="horn_crown", collar=None, ring_verts=40,
                     foot="fillet", gamma=6.0, dive=(0.06, 0.12), cap=(12.0, 24.0, 36.0, 48.0, 60.0, 71.0, 81.0, 88.0),
                     tube=(0.5, 0.4), tent=(2.6, 0.2, 0.03)),
}
SHAPE_NAMES = tuple(SHAPES)


@dataclass(frozen=True)
class HornParams:
    """What the anatomy is built from (units of the horn parameters, at the neutral head)."""

    shape: str = "curved"
    length: float = 1.6
    base_radius: float = 0.17
    segments: int = 28
    ring_verts: int = 24
    curve: float = 1.31
    twist: float = 0.0
    turns: float = 0.0
    tip_radius: float = 0.06
    flare: float = 0.15
    rings: float = 0.05          # growth-ring amplitude, in local radii
    tilt: float = 27.0           # degrees between the axis and the skull's surface normal
    boss: float = 1.0            # the collar's size and the tent's, in 1 = the default (0: none)

    def with_(self, **kw) -> "HornParams":
        return replace(self, **kw)

    @property
    def row(self) -> dict:
        return SHAPES.get(self.shape, SHAPES["curved"])

    @property
    def root_radius(self) -> float:
        """The radius of the sheath at the skin (flare included; a fillet foot's where it meets the skin)."""
        if self.fillet_foot:
            return self.base_radius + fillet_rim(self.fillet_rho, self.row)[0]
        return self.base_radius * (1.0 + self.flare_eff)

    @property
    def fillet_foot(self) -> bool:
        """Is the foot a concave fillet of the sheath's own surface (the skin-covered ossicone) rather than a flare
        under a skin cuff?"""
        return self.row.get("foot") == "fillet"

    @property
    def footprint(self) -> float:
        """How far the sheath reaches from its axis at the skin, in base radii (a fillet foot's buried rim)."""
        if self.fillet_foot:
            return 1.0 + fillet_reach(self.fillet_rho, self.base_radius, self.row) / max(self.base_radius, 1e-12)
        return 1.0 + self.flare_eff

    @property
    def fillet_rho(self) -> float:
        """A fillet foot's radius (local units): ``flare`` base radii, within ``FILLET_RHO``."""
        lo, hi = FILLET_RHO
        return min(max(float(self.flare), lo), hi) * self.base_radius

    @property
    def flare_eff(self) -> float:
        if self.fillet_foot:
            return 0.0                    # (the fillet is the foot's own: no exponential flare on the column)
        return min(max(float(self.flare), 0.0), FLARE_MAX)

    @property
    def sink(self) -> float:
        """How deep the full-radius stem reaches below the skin point along the axis: far enough that
        the skin, which falls away by ``radius x tan(tilt)`` on the side the horn leans away from
        (and by the skull's curvature), never shows the stem's end. A rounded cap of ``CAP_DEPTH`` root
        radii follows it."""
        Ra = self.root_radius
        return SINK_FLOOR * self.base_radius + SINK_TILT * Ra * math.tan(math.radians(min(self.tilt, 70.0)))

    @property
    def cap_depth(self) -> float:
        return CAP_DEPTH * self.root_radius

    @property
    def tip(self) -> float:
        return max(float(self.tip_radius), TIP_FLOOR)

    @property
    def tent_height(self) -> float:
        tent = self.row.get("tent")
        if tent is not None:
            return float(tent[1])
        return TENT_HEIGHT_RAM if self.shape == "ram" else TENT_HEIGHT


# --------------------------------------------------------------------------
# the head: landmarks that follow every morph


@dataclass(frozen=True)
class Head:
    """Landmarks of one morphed head (MakeHuman frame, the left half)."""

    eye: np.ndarray            # the left eye joint
    top: np.ndarray            # the crown marker (head-2)
    root: np.ndarray           # the skull base (head)
    ear: np.ndarray            # the centre of the left ear
    iod: float                 # inter-ocular distance
    left: np.ndarray           # the head axes, unit: toward the character's left,
    up: np.ndarray             # up the head,
    fwd: np.ndarray            # and forward (left x up = fwd)
    skull: float = SKULL_REF   # the skull's half-width (``skull_halfwidth``)
    mid: float = 0.0           # the x of the head's own midline (round 4: the skull-base joint's; a random face
                               # moves the whole head sideways, by 0.16 on the CLI's default seeds 0 and 9)

    @property
    def k(self) -> float:
        """The head's size over the neutral head's: what horn lengths and radii scale by. It is the
        SKULL's half-width over the neutral one (the horn grows from the skull): the round-1 scale, the
        eye distance, follows the eyes, which a face slider moves on a skull of the same width
        (``eyes.eye_trans.in_out`` -1..1 changed it 0.89..1.17, a random face 0.54..1.28 where the skull
        was 1.28 wide: horns 0.43-0.92 x the size the skull asks for)."""
        return self.skull / SKULL_REF

    @property
    def axes(self) -> np.ndarray:
        return np.stack([self.left, self.up, self.fwd], axis=1)


def head_frame(base, verts, cage) -> Head:
    """The head landmarks of the morphed body ``verts`` (``cage`` = its body cage): the eye,
    crown and skull-base joint markers (they move with every morph) and the ear, found as the
    widest part of the head below the eye line."""
    eye = A.joint_position(base, "l-eye", verts)
    top = A.joint_position(base, "head-2", verts)
    root = A.joint_position(base, "head", verts)
    # the head's own midline: a random face translates the whole head sideways (the skull-base and crown joints,
    # the eyes and every slab of the skull move together: measured on --randomize-face seeds 0, 9, 21, 57), so
    # widths and the distance from the midline are measured from it, not from world x = 0
    mid = 0.5 * float(root[0] + top[0])
    iod = abs(float(eye[0]) - float(A.joint_position(base, "r-eye", verts)[0]))
    cv = cage[0]
    skull = skull_halfwidth(cv, root, top)
    k = skull / SKULL_REF
    head = cv[cv[:, 1] > root[1] - 0.3 * k]
    band = ((head[:, 1] > root[1] - 0.15 * k) & (head[:, 1] < eye[1] + 0.25 * k)
            & (head[:, 2] < root[2] + 0.5 * k) & (head[:, 0] > mid))
    hb = head[band]
    xm = float(hb[:, 0].max())
    ear = hb[hb[:, 0] > xm - 0.08 * k].mean(axis=0)
    up = top - root
    up[0] = 0.0
    up /= max(float(np.linalg.norm(up)), 1e-12)
    fwd = np.array([0.0, 0.0, 1.0])
    fwd = fwd - float(fwd @ up) * up
    fwd /= np.linalg.norm(fwd)
    left = np.cross(up, fwd)
    return Head(np.asarray(eye, dtype=np.float64), np.asarray(top, dtype=np.float64),
                np.asarray(root, dtype=np.float64), np.asarray(ear, dtype=np.float64), iod,
                left, up, fwd, skull, mid)


def skull_halfwidth(cage_v, root, top) -> float:
    """The skull's half-width: half the width of the cage in a slab ``SKULL_AT`` of the way from the skull
    base joint to the crown joint (the parietal eminences; above the ears and the eyes, which a face slider
    moves on the same skull), ``SKULL_SLAB`` of that height thick. Half the WIDTH, not the largest ``|x|``:
    a random face moves the head sideways (seed 0 by 0.16: the largest ``|x|`` read 0.881 where the skull is
    0.719 wide each side, a horn 22 % too big)."""
    h = float(top[1] - root[1])
    y = float(root[1]) + SKULL_AT * h
    sl = np.abs(cage_v[:, 1] - y) < SKULL_SLAB * abs(h)
    return 0.5 * float(np.ptp(cage_v[sl, 0])) if sl.any() else SKULL_REF


def local_halfwidth(cage_v, point, radius: float = 0.35, mid: float = 0.0) -> float:
    """The head's half-width at the height of ``point``: the largest ``|x - mid|`` of the cage vertices in a
    slab 0.08 thick around it (within ``radius`` front and back of it in z: not the ears), ``mid`` the x of the
    head's own midline (``Head.mid``)."""
    sl = (np.abs(cage_v[:, 1] - point[1]) < 0.08) & (np.abs(cage_v[:, 2] - point[2]) < radius * 2.0)
    return float(np.abs(cage_v[sl, 0] - mid).max()) if sl.any() else 0.61


def smooth_normal(cage_v, quads, point, radius: float) -> np.ndarray:
    """The skull's surface normal at ``point``, smoothed over ``radius`` (Gaussian weights of the
    area-weighted vertex normals): a quad's own normal is a few degrees off on this coarse cage."""
    nrm = J.vertex_normals(cage_v, quads)
    d = np.linalg.norm(cage_v - point, axis=1)
    m = d < 2.0 * radius
    w = np.exp(-(d[m] / radius) ** 2)
    n = (nrm[m] * w[:, None]).sum(axis=0)
    ln = float(np.linalg.norm(n))
    return n / ln if ln > 1e-12 else np.array([0.0, 1.0, 0.0])


def _rotate(v, axis, angle):
    """``v`` turned about the unit ``axis`` by ``angle`` radians (Rodrigues)."""
    c, s = math.cos(angle), math.sin(angle)
    ax, ay, az = float(axis[0]), float(axis[1]), float(axis[2])
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    d = (ax * vx + ay * vy + az * vz) * (1.0 - c)
    return np.array([vx * c + (ay * vz - az * vy) * s + ax * d,
                     vy * c + (az * vx - ax * vz) * s + ay * d,
                     vz * c + (ax * vy - ay * vx) * s + az * d])


@dataclass(frozen=True)
class Site:
    """Where and how one (left-hand) horn leaves the head."""

    point: np.ndarray          # the skin point on the axis (on the cage surface)
    normal: np.ndarray         # the smoothed skull normal there
    axis: np.ndarray           # the horn's initial direction (unit): the normal tilted by ``tilt``
    bend: np.ndarray           # the unit direction the horn bends toward (perpendicular to the axis)
    halfwidth: float           # the head's half-width at that height
    scale: float               # local units -> world units: the head's size over the neutral head's
    base_radius: float         # the base radius in LOCAL units after the limits
    limited: str = ""          # what limited the radius ("" = nothing)
    mid: float = 0.0           # the x of the head's own midline (``Head.mid``)
    side: int = 0              # +1 the left horn, -1 the right one (0: by the point's side of ``mid``)

    @property
    def radius(self) -> float:
        """The base radius in world units."""
        return self.base_radius * self.scale

    @property
    def right(self) -> bool:
        if self.side:
            return self.side < 0
        return float(self.point[0]) < self.mid

    @property
    def offset(self) -> float:
        """How far the skin point is from the head's own midline (world units)."""
        return abs(float(self.point[0]) - self.mid)

    def matrix(self, sink_offset: float = 0.0) -> np.ndarray:
        """The 4x4 that puts the local horn (``z`` = axis, ``x`` = bend, ``y = z x x``) on the head. On
        the right side ``y = -(z x x)``: a mirroring frame, so the right horn is the mirror image of the
        left one (a corkscrew and a keel of the other hand) while it stands on ITS OWN site (on a
        symmetric head this is exactly ``diag(-1, 1, 1)`` of the left horn's matrix)."""
        m = np.eye(4)
        z, x = self.axis, self.bend
        y = np.cross(z, x) * (-1.0 if self.right else 1.0)
        m[:3, :3] = np.stack([x, y, z], axis=1) * self.scale
        m[:3, 3] = self.point + z * sink_offset
        return m


def horn_site(cage, head: Head, point, params: HornParams, *, scale: float = 1.0, roll: float = 0.0,
              tilt: float | None = None, side: int = 0) -> Site:
    """The site of a horn at the skin ``point`` of the left half (``x`` > the head's midline ``head.mid``; a
    point on the right, or ``side`` -1, is treated as its mirror image): the radius from the head, the axis from
    the skull's surface normal.

    * the size follows the head: ``scale x head.k`` (the skull's half-width over the neutral one) times the
      parameters, with the base radius limited to ``HALFWIDTH_CAP`` of the head's half-width at that height and
      so that the tent keeps ``TENT_MARGIN`` off the head's OWN midline (round 4: it was world x = 0, and a random
      face that moved the head 0.16 sideways was limited by a midline that was not there);
    * the axis is the smoothed surface normal tilted by ``tilt`` degrees (default: the shape's, 15 to
      45 for a horn that reads as grown out of the skull) toward the shape's lean direction,
      expressed in the head's own axes;
    * the bend plane contains the axis and the shape's bend direction (also in head axes), turned about
      the axis by ``roll`` degrees."""
    cv, cq = cage
    point = np.asarray(point, dtype=np.float64)
    mid = float(head.mid)
    side = int(side) if side else (1 if point[0] >= mid else -1)
    row = params.row
    k = head.k * scale
    R = params.base_radius * k
    n = smooth_normal(cv, cq, point, max(1.2 * R, 0.1))
    H = head.axes * np.array([side, 1.0, 1.0])      # head axes of this side (a point on the right mirrors)
    lean = H @ np.asarray(row["lean"], dtype=np.float64)
    bend = H @ np.asarray(row["bend"], dtype=np.float64)
    t = lean - float(lean @ n) * n
    if float(np.linalg.norm(t)) < 1e-6:
        t = head.up - float(head.up @ n) * n
    t /= np.linalg.norm(t)
    tau = math.radians(min(max(float(params.tilt if tilt is None else tilt), 0.0), TILT_MAX))
    axis = n * math.cos(tau) + t * math.sin(tau)
    axis /= np.linalg.norm(axis)
    w = bend - float(bend @ axis) * axis
    if float(np.linalg.norm(w)) < 1e-6:
        w = np.cross(axis, head.up)
    w /= np.linalg.norm(w)
    if roll:
        w = _rotate(w, axis, math.radians(roll))
    hw = local_halfwidth(cv, point, mid=mid)
    limit, why = R, ""
    cap = HALFWIDTH_CAP * hw
    if limit > cap:
        limit, why = cap, "head width"
    # (a fillet foot reaches further out under the skin than the column: the ossicone's, 2.1 base radii)
    margin = (abs(float(point[0]) - mid) - TENT_MARGIN) / max(MIDLINE_RADII, params.footprint + 0.1)
    if limit > margin > 0.0:
        limit, why = margin, "midline"
    return Site(point=point, normal=n, axis=axis, bend=w, halfwidth=hw, scale=k,
                base_radius=limit / k, limited=why, mid=mid, side=side)



TWIST_MARGIN = 0.5       # a twisted sheath keeps this many TENT_MARGINs off the head's midline (the pair never meets)


def midline_room(model: "HornModel", matrices, mids) -> float:
    """The least distance (world units) between a horn's sheath above its skin point and the head's midline, over
    the sites ``matrices`` (local -> world) with ``mids`` = ``(midline x, +1 left / -1 right)``: the centreline's
    distance from the midline less the sheath's largest radius there (the keel's lobes and the rings included)."""
    s, pos, *_ = model.path()
    up = s >= 0.0
    r = model.radius(s[up]) * (1.0 + float(model.p.row["lobe"]))
    room = math.inf
    for m, (mid, side) in zip(matrices, mids):
        m = np.asarray(m, dtype=np.float64)
        sc = float(abs(np.linalg.det(m[:3, :3])) ** (1.0 / 3.0))
        x = pos[up] @ m[0, :3] + m[0, 3]
        room = min(room, float(((x - mid) * side - r * sc).min()))
    return room


def twist_room(params: HornParams, matrices, mids, iterations: int = 10) -> float:
    """The share of its twist (``HornModel.twist_scale``) a horn of ``params`` keeps so that, on the sites
    ``matrices`` / ``mids`` (:func:`midline_room`), no part of its sheath comes nearer the head's midline than
    ``TWIST_MARGIN x TENT_MARGIN``: 1 when the full twist does (or when the untwisted horn does not either: then
    the twist is not what brings it there), else the largest share that does, by bisection."""
    need = TWIST_MARGIN * TENT_MARGIN
    if midline_room(HornModel(params), matrices, mids) >= need:
        return 1.0
    if midline_room(HornModel(params, twist_scale=0.0), matrices, mids) < need:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(iterations):
        f = 0.5 * (lo + hi)
        if midline_room(HornModel(params, twist_scale=f), matrices, mids) >= need:
            lo = f
        else:
            hi = f
    return lo


# --------------------------------------------------------------------------
# a skin-covered foot shaded as the skin it comes out of


class ShadedMesh(Mesh):
    """A shell that carries the normals and the cavity term it is SHADED with (``shading_normals`` (n, 3),
    ``shading_cavity`` (n,)): the protocol of ``collar.CollarMesh`` (the renderer, ``portrait._shell_colors``, uses
    them where it knows them; a renderer that does not draws the mesh with its own). The skin-covered ossicone's foot
    carries the SKIN's normals and occlusion where it meets the skin, blending to its own up the fillet (``shade_foot``):
    the renderer shades every shell with its own area weighted normals and its own cavity term, so the line where the
    foot dives under the skin at a few degrees was drawn as a ragged ring of two shadings (round 4, before this). The
    geometry, hence the print, is not changed by it; every transform keeps the attributes (normals by the inverse
    transpose), ``Mesh.merge`` and the print path drop them."""

    def __init__(self, vertices, faces, shading_normals=None, shading_cavity=None):
        super().__init__(vertices, faces)
        self.shading_normals = None if shading_normals is None else np.asarray(shading_normals, dtype=np.float64)
        self.shading_cavity = None if shading_cavity is None else np.asarray(shading_cavity, dtype=np.float64)

    def _like(self, out: Mesh, linear=None) -> "ShadedMesh":
        n = self.shading_normals
        if n is not None and linear is not None:
            n = n @ np.linalg.inv(np.asarray(linear, dtype=np.float64))
            n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        return ShadedMesh(out.vertices, out.faces, n, self.shading_cavity)

    def copy(self) -> "ShadedMesh":
        return self._like(Mesh.copy(self))

    def transform(self, m) -> "ShadedMesh":
        return self._like(Mesh.transform(self, m), np.asarray(m, dtype=np.float64)[:3, :3])

    def translated(self, v) -> "ShadedMesh":
        return self._like(Mesh.translated(self, v))

    def scaled(self, s) -> "ShadedMesh":
        s = np.asarray(s, dtype=np.float64)
        return self._like(Mesh.scaled(self, s), np.diag(np.broadcast_to(s, (3,))) if s.ndim else np.eye(3) * float(s))


SHADE_BAND = (0.05, 0.3) # the foot is shaded as the skin up to the first share of the fillet's radius over the skin,
                         # blending to its own shading by the second (the fillet is then about 45 degrees up its arc)


def shade_foot(mesh: Mesh, skin, centre, radius: float, rho: float) -> ShadedMesh:
    """``mesh`` (world frame) as a :class:`ShadedMesh` that carries the skin's shading where it meets the skin: the
    skin (``(vertices, triangles)`` of the drawn body near ``centre``, within ``radius``) gives the normal and the cavity
    term under each vertex (``collar.SkinPatch``: the renderer's own area weighted normals); the weight is 1 at and
    under the skin and falls to 0 between ``SHADE_BAND[0]`` and ``SHADE_BAND[1]`` fillet radii ``rho`` over it."""
    from . import collar as CO
    from . import shading as sh
    patch = CO.SkinPatch(skin[0], skin[1], centre, radius)
    _, n_skin, sd, nearest = patch.closest(mesh.vertices)
    own = sh.vertex_normals(mesh.vertices, mesh.faces)
    c_own = sh.cavity_term(mesh.vertices, mesh.faces, own)
    c_skin = sh.cavity_term(patch.V, patch.F, patch.N)[nearest]
    u = (sd - SHADE_BAND[0] * rho) / max((SHADE_BAND[1] - SHADE_BAND[0]) * rho, 1e-9)
    w = 1.0 - smoothstep(u)
    n = w[:, None] * n_skin + (1.0 - w)[:, None] * own
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return ShadedMesh(mesh.vertices, mesh.faces, n, w * c_skin + (1.0 - w) * c_own)


# --------------------------------------------------------------------------
# the skin the horn comes out of, in the horn's own frame


class PlaneField:
    """The skin as the plane through the horn's origin with the skull normal ``normal`` (in the
    horn's LOCAL frame; ``z`` = the axis): ``height(xy)`` is the skin's ``z`` at ``(x, y)``. What a
    horn is built against when it is not on a body (tests, the numbers of the docs)."""

    def __init__(self, normal):
        n = np.asarray(normal, dtype=np.float64)
        self.n = n / np.linalg.norm(n)

    def height(self, xy, hits: bool = False):
        xy = np.asarray(xy, dtype=np.float64)
        z = -(self.n[0] * xy[:, 0] + self.n[1] * xy[:, 1]) / self.n[2]
        return (z, np.ones(len(xy), dtype=bool)) if hits else z


class SkinField:
    """The skin as a height field over the horn's local ``xy`` plane, sampled by casting rays along
    the horn's axis at the body's triangles (``cage`` = ``(vertices, quads)`` in the world frame,
    ``matrix`` = local -> world): ``height(xy)`` is the ``z`` of the surface nearest ``z = 0``."""

    def __init__(self, cage, matrix, reach: float, fallback: PlaneField):
        v, q = cage
        inv = np.linalg.inv(matrix)
        loc = v @ inv[:3, :3].T + inv[:3, 3]
        near = ((np.abs(loc[q][:, :, 0]).min(axis=1) < reach) & (np.abs(loc[q][:, :, 1]).min(axis=1) < reach)
                & (np.abs(loc[q][:, :, 2]).min(axis=1) < max(reach, 0.8)))
        quads = q[near]
        tri = np.concatenate([loc[quads][:, [0, 1, 2]], loc[quads][:, [0, 2, 3]]])
        self.tri = tri
        self.fallback = fallback
        self.reach = reach

    def _grid(self):
        """A uniform grid over the triangles' ``xy`` boxes (round 4: the ossicone's foot is fitted to the cage
        subdivided once, four times the triangles, and testing every point against every triangle took 2 s a pair):
        ``(origin, cell, ny, keys, tris)``, the triangle ids sorted by the key of each cell they overlap."""
        if getattr(self, "_g", None) is not None:
            return self._g
        tri = self.tri
        lo, hi = tri[:, :, :2].min(axis=1), tri[:, :, :2].max(axis=1)
        cell = max(1.5 * float(np.median((hi - lo).max(axis=1))), 1e-4)
        origin = lo.min(axis=0) - 1e-9
        i0 = np.floor((lo - origin) / cell).astype(np.int64)
        i1 = np.floor((hi - origin) / cell).astype(np.int64)
        ny = int(i1[:, 1].max()) + 2
        span = i1 - i0 + 1
        cnt = span[:, 0] * span[:, 1]
        t = np.repeat(np.arange(len(tri)), cnt)
        k = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        sy = np.repeat(span[:, 1], cnt)
        ix = np.repeat(i0[:, 0], cnt) + k // sy
        iy = np.repeat(i0[:, 1], cnt) + k % sy
        key = ix * ny + iy
        order = np.argsort(key, kind="stable")
        self._g = (origin, cell, ny, key[order], t[order])
        return self._g

    def height(self, xy, hits: bool = False):
        """The skin's ``z`` over each point of ``xy``; ``hits=True`` also returns which points the rays found
        the skin under (the others are the fallback plane's: a ray past the head's silhouette). Of the triangles a
        ray hits, the one nearest ``z = 0``."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        out = self.fallback.height(xy)
        found = np.zeros(len(xy), dtype=bool)
        if not len(self.tri) or not len(xy):
            return (out, found) if hits else out
        origin, cell, ny, keys, tris = self._grid()
        ij = np.floor((xy - origin) / cell).astype(np.int64)
        pk = ij[:, 0] * ny + ij[:, 1]
        start = np.searchsorted(keys, pk, side="left")
        end = np.searchsorted(keys, pk, side="right")
        n = end - start
        if not n.any():
            return (out, found) if hits else out
        kmax = int(n.max())
        idx = start[:, None] + np.arange(kmax)[None, :]
        valid = np.arange(kmax)[None, :] < n[:, None]
        cand = tris[np.where(valid, idx, 0)]
        a, b, c = self.tri[cand, 0], self.tri[cand, 1], self.tri[cand, 2]           # (n, k, 3)
        e1, e2 = b[..., :2] - a[..., :2], c[..., :2] - a[..., :2]
        det = e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0]
        ok = valid & (np.abs(det) > 1e-14)
        det = np.where(ok, det, 1.0)
        d = xy[:, None, :] - a[..., :2]
        u = (d[..., 0] * e2[..., 1] - d[..., 1] * e2[..., 0]) / det
        w = (e1[..., 0] * d[..., 1] - e1[..., 1] * d[..., 0]) / det
        hit = ok & (u >= -1e-9) & (w >= -1e-9) & (u + w <= 1.0 + 1e-9)
        z = a[..., 2] + u * (b[..., 2] - a[..., 2]) + w * (c[..., 2] - a[..., 2])
        nearest = np.where(hit, np.abs(z), np.inf)
        j = np.argmin(nearest, axis=1)
        rows = np.arange(len(xy))
        got = np.isfinite(nearest[rows, j])
        out[got] = z[rows, j][got]
        found[:] = got
        return (out, found) if hits else out


# --------------------------------------------------------------------------
# the model


def _ring(centre, n_vec, b_vec, a, b, ring_verts, lobe=0.0, keel=0.0):
    th = np.linspace(0.0, 2.0 * np.pi, ring_verts, endpoint=False)
    rad = np.ones_like(th)
    if lobe:
        rad = 1.0 + lobe * np.cos(3.0 * (th - keel))
    return (centre[None, :] + (a * rad * np.cos(th))[:, None] * n_vec[None, :]
            + (b * rad * np.sin(th))[:, None] * b_vec[None, :])


class HornModel:
    """The horn in its local frame: centreline, frames and the three layers' radii.

    Everything derives from one :class:`HornParams`. The meshes are built on demand
    (:meth:`skin`, :meth:`bone`), the skin is the one that is printed."""

    def __init__(self, params: HornParams, twist_scale: float = 1.0):
        self.p = params
        self.twist_scale = float(min(max(twist_scale, 0.0), 1.0))   # see ``psi_b``
        self.R = float(params.base_radius)
        self.L = float(params.length)
        self._stations = self._make_stations()
        self._path = None

    # -- the radius law -----------------------------------------------------------------

    def radius(self, s, flare: bool = True):
        """The sheath's mean radius at ``s`` (units), including the root flare (``flare=False``: without
        it; the skin's own foot flares along the local skin, :meth:`skin`) and the growth rings;
        the buried stem holds its root value."""
        p, R, L = self.p, self.R, self.L
        s = np.asarray(s, dtype=np.float64)
        t = np.clip(s / L, 0.0, 1.0)
        tip = min(p.tip, 0.95 * R)
        base = tip + (R - tip) * (1.0 - t) ** TAPER_P
        sv = np.maximum(s, 0.0)
        fl = 1.0 + (p.flare_eff * np.exp(-sv / (FLARE_LEN * R)) if flare else 0.0)
        return base * fl * (1.0 + self.ring_wave(s))

    def ring_wave(self, s):
        """The growth rings' relative radius change at ``s``: ridges of pitch 0.8 R (0.55 of that at
        the tip), none for the first RING_FIRST R above the skin and none on the tip."""
        p, R, L = self.p, self.R, self.L
        amp = float(p.rings)
        s = np.asarray(s, dtype=np.float64)
        if amp <= 0.0:
            return np.zeros_like(s)
        pitch0 = RING_PITCH * R
        c = 1.0 - RING_PITCH_TIP
        # phase(s) = integral of ds / pitch(s), pitch(s) = pitch0 (1 - c s / L)
        sc = np.clip(s, 0.0, L)
        phase = -(L / (c * pitch0)) * np.log(np.maximum(1.0 - c * sc / L, 1e-9))
        phase0 = -(L / (c * pitch0)) * math.log(max(1.0 - c * RING_FIRST * R / L, 1e-9))
        x = phase - phase0
        ridge = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.mod(x, 1.0) ** 0.75))
        env = smoothstep((s - RING_START * R) / (0.8 * R)) * \
            (1.0 - smoothstep((s - 0.84 * L) / (0.12 * L)))
        return amp * ridge * env


    # -- the centreline -------------------------------------------------------------------

    def _make_stations(self) -> np.ndarray:
        """The stations (s) the sheath's rings are at: the buried stem, the root's dense rings and
        the shaft, fine enough to resolve the growth rings."""
        p, R, L = self.p, self.R, self.L
        n = max(int(p.segments), 3)
        if p.rings > 0.0:
            n = max(n, int(math.ceil(L / (RING_PITCH * R * (1.0 - 0.6 * (1.0 - RING_PITCH_TIP))
                                           / RING_PER_PERIOD))))
        shaft = np.linspace(0.0, L, n + 1)
        dense = R * np.array([0.04, 0.1, 0.18, 0.28, 0.4, 0.55, 0.75, 1.0, 1.3])
        dense = dense[dense < 0.5 * L]
        st = np.unique(np.concatenate([shaft, dense]))
        # drop stations that crowd another by less than a tenth of the dense spacing
        keep = [0]
        for i in range(1, len(st)):
            if st[i] - st[keep[-1]] > 0.02 * R:
                keep.append(i)
        return st[keep]

    def _curvature(self, s):
        """``(kN, kB)``: the curvature along the bend direction N and the third axis B at ``s``."""
        p, R, L = self.p, self.R, self.L
        row = p.row
        psi = self.psi
        psi_b = self.psi_b
        s = np.asarray(s, dtype=np.float64)
        if row["profile"] == "lyre":
            w1 = 0.55 * L
            w2 = L - w1
            f = np.where(s < w1, (math.pi / (2.0 * w1)) * np.sin(np.pi * s / w1),
                         -0.7 * (math.pi / (2.0 * w2)) * np.sin(np.pi * np.clip(s - w1, 0, w2) / w2))
            kN, kB = psi * f, psi_b * f
        else:
            q = float(row["q"])
            S0 = q * L
            norm = math.log(q / (q - 1.0))
            f = 1.0 / (norm * (S0 - np.clip(s, 0.0, L)))
            kN, kB = psi * f, psi_b * f
        return kN, kB

    def taper(self, s):
        """The sheath's smooth radius at ``s`` (the cone, no flare and no growth rings): what the path's
        bend limits are measured against (the rings ride on it; a helix built on the rippled radius rippled
        the centreline at the rings' pitch)."""
        p, R, L = self.p, self.R, self.L
        t = np.clip(np.asarray(s, dtype=np.float64) / L, 0.0, 1.0)
        tip = min(p.tip, 0.95 * R)
        return tip + (R - tip) * (1.0 - t) ** TAPER_P

    @property
    def turns_eff(self) -> float:
        """The corkscrew's turns, at most so many that a coil is ``PITCH_MIN`` base radii long (closer coils
        of a tube that thick run into each other: the first spiral's 2.25 turns at a radius twice its own
        crossed itself 126 times per horn)."""
        t = float(self.p.turns)
        cap = self.L / (PITCH_MIN * self.R)
        return math.copysign(min(abs(t), cap), t)

    @property
    def psi(self) -> float:
        """The planar spiral's total turn ``curve x L / 2`` (radians), at most ``PSI_MAX``: a log spiral that
        turns on past that runs into its own first coil (a ram of length 12, 19 radians, crossed itself
        2,352 times)."""
        psi = float(self.p.curve) * self.L / 2.0
        cap = LYRE_PSI_MAX if self.p.row["profile"] == "lyre" else PSI_MAX
        return math.copysign(min(abs(psi), cap), psi)

    @property
    def psi_b(self) -> float:
        """The out-of-plane turn ``twist x L / 2`` (radians): the bend plane turns by ``atan(psi_b / psi)`` and the
        total turn is ``hypot(psi, psi_b)``, which is held to the same cap as ``psi`` alone (round 4: it was not
        capped, and an antelope at twist 3 turned 5.1 radians sideways and crossed itself 166 times)."""
        psi_b = float(self.p.twist) * self.L / 2.0
        cap = LYRE_PSI_MAX if self.p.row["profile"] == "lyre" else PSI_MAX
        room = math.sqrt(max(cap * cap - self.psi * self.psi, 0.0))
        return math.copysign(min(abs(psi_b), room), psi_b) * self.twist_scale if psi_b else 0.0

    def helix_offset(self, s, N, B, rho_scale=None):
        """The corkscrew's displacement from the axis curve at ``s`` (``(n, 3)``): a circle of radius
        ``HELIX_RHO`` x the local (smooth) radius in the axis' bend frame ``N``, ``B``, turning ``turns_eff``
        times over the length and growing in from the skin (a horn starts on its axis); ``rho_scale`` (per
        ``s``) is what the bend limit leaves of it (:meth:`path`)."""
        p, L = self.p, self.L
        turns = self.turns_eff
        if not turns:
            return np.zeros((len(s), 3))
        sp = np.clip(s, 0.0, L)
        s0 = min(HELIX_START * self.R, 0.3 * L)                  # (above the foot: its rings follow the skin)
        u = np.clip((sp - s0) / (HELIX_RAMP * L), 0.0, 1.0)
        ramp = u * u * u * (u * (6.0 * u - 15.0) + 10.0)       # C2 at the root: a quadratic start bent the path there
        rho = HELIX_RHO * np.minimum(self.taper(sp), 1.5 * self.R) * ramp
        if rho_scale is not None:
            rho = rho * rho_scale
        phi = 2.0 * np.pi * turns * sp / L + math.radians(float(p.row.get("phase", 0.0)))
        return (rho * np.cos(phi))[:, None] * N + (rho * np.sin(phi))[:, None] * B

    def path(self):
        """``(s, centre, T, N, B)`` on a fine grid from the buried end to the tip: straight below the
        skin, then the axis curve bent by the curvature law (an equiangular spiral, a lyre), plus the
        corkscrew's helical offset; the frames are carried along by parallel transport. The curvature
        of the FINAL path times the smooth radius never exceeds ``BEND_LIMIT`` (the loft would fold on its
        inner side): the planar law is clamped, and so is the helix's radius where the sum still bends too
        hard."""
        if self._path is not None:
            return self._path
        L = self.L
        sink = self.p.sink
        s = np.unique(np.concatenate([np.linspace(0.0, L, 481), self._stations,
                                      np.linspace(-(sink + self.p.cap_depth) * 1.05, 0.0, 12)]))
        # no two samples closer than a fifth of the grid's step (a station a hair from a grid point made the
        # differentiated tangent of a corkscrew jump there, and the ring frames carried along it with it)
        keep = np.concatenate([[True], np.diff(s) > 0.2 * L / 480.0])
        keep |= np.isclose(s, 0.0)
        s = s[keep]
        kN, kB = self._curvature(np.where(s > 0, s, 0.0))
        kN = np.where(s > 0, kN, 0.0)
        kB = np.where(s > 0, kB, 0.0)
        # the radius the limit is measured on: the smooth taper with the keel's lobes and the growth rings on it
        r = self.taper(np.clip(s, 0.0, L)) * (1.0 + float(self.p.row["lobe"])) * (1.0 + max(float(self.p.rings), 0.0))
        mag = np.hypot(kN, kB)
        cap = BEND_LIMIT / np.maximum(r, 1e-9)
        sc = np.where(mag > cap, cap / np.maximum(mag, 1e-12), 1.0)
        kN, kB = kN * sc, kB * sc
        T = np.array([0.0, 0.0, 1.0])
        N = np.array([1.0, 0.0, 0.0])
        B = np.array([0.0, 1.0, 0.0])
        i0 = int(np.searchsorted(s, 0.0))          # s[i0] == 0
        n = len(s)
        pos = np.zeros((n, 3))
        Ts = np.zeros((n, 3))
        Ns = np.zeros((n, 3))
        Bs = np.zeros((n, 3))
        pos[i0], Ts[i0], Ns[i0], Bs[i0] = 0.0, T, N, B
        for i in range(i0 + 1, n):
            ds = s[i] - s[i - 1]
            km = 0.5 * (kN[i] + kN[i - 1]), 0.5 * (kB[i] + kB[i - 1])
            e = km[0] * N + km[1] * B
            ang = float(np.hypot(*km)) * ds
            pos[i] = pos[i - 1] + T * ds
            if ang > 1e-12:
                axis = np.array([T[1] * e[2] - T[2] * e[1], T[2] * e[0] - T[0] * e[2],
                                 T[0] * e[1] - T[1] * e[0]])
                axis /= math.sqrt(float(axis @ axis))
                Tm = _rotate(T, axis, 0.5 * ang)               # the midpoint heading for the step
                pos[i] = pos[i - 1] + Tm * ds
                T, N, B = _rotate(T, axis, ang), _rotate(N, axis, ang), _rotate(B, axis, ang)
            Ts[i], Ns[i], Bs[i] = T, N, B
        for i in range(i0 - 1, -1, -1):
            ds = s[i + 1] - s[i]
            pos[i] = pos[i + 1] - np.array([0.0, 0.0, 1.0]) * ds
            Ts[i], Ns[i], Bs[i] = Ts[i0], Ns[i0], Bs[i0]
        if self.turns_eff:
            # the bend limit of the corkscrew is worked out on a uniform grid (the path's own grid has the
            # sheath's stations in it, some a hair apart, where a per-sample scale would itself kink the path)
            su = np.linspace(0.0, L, 601)
            ip = lambda a: np.stack([np.interp(su, s, a[:, c]) for c in range(3)], axis=1)   # noqa: E731
            ax_u, n_u, b_u = ip(pos), ip(Ns), ip(Bs)
            r_u = np.interp(su, s, r)
            scale_u = np.ones(len(su))
            for _ in range(HELIX_ITER):
                pu = ax_u + self.helix_offset(su, n_u, b_u, scale_u)
                d = np.gradient(pu, su, axis=0)
                Tt = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
                over = (np.linalg.norm(np.gradient(Tt, su, axis=0), axis=1) * r_u / (HELIX_BEND * BEND_LIMIT))[2:-2]
                if over.max() <= 1.0:          # (the ends: one-sided differences)
                    break
                # take the helix in where the sum bends too hard, and a little around it (a step in its
                # radius would itself bend the path)
                # (the whole helix: a helix taken in locally passes through the radii at which it bends hardest,
                # 1 / 2c at a radius equal to its pitch over 2 pi, and kinks there)
                scale_u = scale_u * min(1.0 / float(over.max()), 0.9)
            pos = pos + self.helix_offset(s, Ns, Bs, np.interp(np.clip(s, 0.0, L), su, scale_u))
            # the tangent of the new path and frames carried along it (minimal rotations)
            d = np.gradient(pos, s, axis=0)
            Ts = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
            N = Ns[i0]
            for i in range(n):
                if i == 0:
                    N = Ns[0] - float(Ns[0] @ Ts[0]) * Ts[0]
                else:
                    ax = np.cross(Ts[i - 1], Ts[i])
                    sn = float(np.linalg.norm(ax))
                    if sn > 1e-12:
                        N = _rotate(N, ax / sn, math.atan2(sn, float(Ts[i - 1] @ Ts[i])))
                N = N - float(N @ Ts[i]) * Ts[i]
                N = N / np.linalg.norm(N)
                Ns[i] = N
                Bs[i] = np.cross(Ts[i], N)
        self._path = (s, pos, Ts, Ns, Bs)
        return self._path

    def frames_at(self, st):
        """``(centre, T, N, B)`` interpolated on the path at the stations ``st``."""
        s, pos, T, N, B = self.path()
        out = [np.stack([np.interp(st, s, a[:, c]) for c in range(3)], axis=1)
               for a in (pos, T, N, B)]
        c, t, n, b = out
        # re-orthonormalise
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        n = n - (n * t).sum(1, keepdims=True) * t
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        b = np.cross(t, n)
        return c, t, n, b

    # -- sections -----------------------------------------------------------------------------

    def semi_axes(self, s, flare: bool = True):
        """``(a, b)``: the major (along the bend direction) and minor semi-axes at ``s``."""
        r = self.radius(s, flare)
        asp = float(self.p.row["aspect"])
        return r, r * asp

    def inscribed(self, s):
        """The radius of the largest circle about the axis that lies inside the sheath's mesh at ``s``:
        the minor semi-axis without the growth rings (the troughs between them), less the keel's lobes,
        the groove at the skin line and the polygon of the rings. The junction's tube is this (times
        ``HUG``), so the skin it calls hidden is hidden by the sheath."""
        p, R, L = self.p, self.R, self.L
        s = np.asarray(s, dtype=np.float64)
        t = np.clip(s / L, 0.0, 1.0)
        tip = min(p.tip, 0.95 * R)
        base = tip + (R - tip) * (1.0 - t) ** TAPER_P
        # the foot's flare decays with the height over the LOCAL skin, which on the low side of a tilted
        # root is up to Ra tan(tilt) under the skin point, and over a compressed foot (a stub) faster
        Ra = R * (1.0 + p.flare_eff)
        drop = Ra * math.tan(math.radians(min(p.tilt, 70.0)))
        room = 0.85 * L - drop
        f = 1.0 if room >= FOOT_HEIGHTS[-1] * Ra else max(room / (FOOT_HEIGHTS[-1] * Ra), 0.05)
        fl = 1.0 + p.flare_eff * np.exp(-np.maximum(s + drop, 0.0) / (FLARE_LEN * R * f))
        lobe = float(p.row["lobe"])
        return (base * fl * float(p.row["aspect"]) * (1.0 - lobe)
                * math.cos(math.pi / int(p.ring_verts)))

    # -- the meshes -----------------------------------------------------------------------------

    def _cap_rings(self, centre, direction, nvec, bvec, a, b, ring_verts, lobe, keel, height=None, degrees=TIP_CAP):
        """Rings of a rounded end on a quarter ellipsoid, from the section outward (``direction``);
        ``height`` is how far it reaches (default: the major radius: a dome); ``degrees`` where the rings are on the
        quarter (an ossicone's dome is in plain sight: more of them)."""
        h = a if height is None else height
        out = []
        for deg in degrees:
            phi = math.radians(deg)
            out.append(_ring(centre + direction * h * math.sin(phi), nvec, bvec,
                             a * math.cos(phi), b * math.cos(phi), ring_verts, lobe, keel))
        return out

    def foot_rings(self, field=None):
        """The rings of the sheath's lower part in the horn's local frame: ``(bowl, foot, shaft_s)``. The
        buried bowl (rings from its bottom up), the foot (rings that follow the skin it comes out of, with
        a groove at the skin line and a bead above it: the keratin's crisp rim) and the stations of the
        shaft above the foot's last (planar) ring. ``field`` is the skin (a :class:`SkinField`; default:
        the plane of a skull normal at ``tilt``)."""
        p = self.p
        nv = int(p.ring_verts)
        R = self.R
        lobe = float(p.row["lobe"])
        keel = math.pi                                 # the keel on the outside of the bend
        th = np.linspace(0.0, 2.0 * np.pi, nv, endpoint=False)
        ct, st_ = np.cos(th), np.sin(th)
        if field is None:
            tau = math.radians(p.tilt)
            field = PlaneField((-math.sin(tau), 0.0, math.cos(tau)))
        if p.fillet_foot:
            return self._fillet_foot(field)
        Ra = float(self.radius(0.0))                   # the root radius (flare included)
        a0, b0 = (float(x) for x in self.semi_axes(0.0))
        z_skin = field.height(np.stack([a0 * ct, b0 * st_], axis=1))
        boss = min(max(float(p.boss), 0.0), 2.0)
        # a horn shorter than its own foot (a stub on a head that tilts it): the foot is compressed
        # so that the first shaft ring is at most 0.85 of the length up
        heights = np.array(FOOT_HEIGHTS)
        room = 0.85 * self.L - float(z_skin.max())
        # ... and so that the path turns at most FOOT_TURN over the foot (its rings take their frames from the
        # path at their own height: a ram that curls from its root, the short one a capped radius builds, crossed
        # its own foot)
        sp, _, Tp, _, _ = self.path()
        cos_t = Tp[sp >= 0.0] @ np.array([0.0, 0.0, 1.0])
        turned = np.nonzero(cos_t < math.cos(math.radians(FOOT_TURN)))[0]
        if len(turned):
            room = min(room, float(sp[sp >= 0.0][turned[0]]) - float(z_skin.max()))
        if room < heights[-1] * Ra:
            # (never flatter than FOOT_MIN of it: rings squeezed onto the tilted skin coincide on its high side)
            # (... unless that would put the foot's top past the tip: a stub shorter than its foot)
            top_room = (0.92 * self.L - float(z_skin.max())) / (heights[-1] * Ra)
            f = max(room / (heights[-1] * Ra), min(FOOT_MIN, top_room), 0.05)
            heights = np.where(heights > 0, heights * f, heights)
        s_top = max(float(z_skin.max()) + float(heights[-1]) * Ra, 0.0)
        st = self._stations
        shaft_s = st[st > s_top + 0.04 * R]      # (the foot's last ring is the planar ring at s_top)
        if not len(shaft_s):
            shaft_s = np.array([min(s_top + 0.5 * (self.L - s_top), self.L)])
        # the foot: K rings; ring i is at z = (1 - t_i) (skin + h_i Ra) + t_i s_top
        K = len(FOOT_HEIGHTS)
        foot = []
        for i, hh in enumerate(heights):
            hh = float(hh)
            t = 0.0 if i == 0 else float(smoothstep((i - 1) / (K - 2)))     # (the buried ring and the skin line: 0)
            keep = 1.0 - t
            # the flare is a function of the NOMINAL height above the skin (it fades as the ring leaves it);
            # a ring that flares is wider, so it is lowered to the skin at ITS OWN place, in two passes. The
            # rings go from following the skin to the planar first shaft ring along a smoothstep (a power law
            # left a tilted ring under a flat one: a crease across the foot)
            fl = p.flare_eff * math.exp(-max(hh, 0.0) / FLARE_LEN)
            m = 1.0 + keep * fl
            rad = 1.0 + (0.0 if lobe == 0.0 else lobe * np.cos(3.0 * (th - keel)))
            z = (1.0 - t) * (z_skin + hh * Ra) + t * s_top
            lo_z = None if not foot else z_prev + FOOT_GAP * Ra       # each ring above the one below it ...
            hi_z = s_top - (K - 1 - i) * FOOT_GAP * Ra                # ... and below the planar top ring
            for _ in range(2):
                c, tt, nn, bb = self.frames_at(np.maximum(z, 0.0))
                a, b = self.semi_axes(np.maximum(z, 0.0), flare=False)
                v = c + (a * m * rad * ct)[:, None] * nn + (b * m * rad * st_)[:, None] * bb
                # the skin under this ring's own xy, in the horn's frame (the ring is nearly flat in
                # the xy plane: its centre is on the axis below the skin)
                z = (1.0 - t) * (field.height(v[:, :2]) + hh * Ra) + t * s_top
                # (the skin under a ring higher up the foot can be higher than under the root ring that set the
                # top: the rings of a squeezed foot, a short horn the head limited, crossed there)
                z = np.minimum(z, hi_z) if i < K - 1 else z
                if lo_z is not None:
                    z = np.maximum(z, lo_z)
            z_prev = z
            c, tt, nn, bb = self.frames_at(np.maximum(z, 0.0))
            a, b = self.semi_axes(np.maximum(z, 0.0), flare=False)
            v = c + (a * m * rad * ct)[:, None] * nn + (b * m * rad * st_)[:, None] * bb
            v[:, 2] = np.where(z > 0.0, v[:, 2], z)       # below the skin point the stem is straight
            foot.append(v)
        # the buried stem: the first ring carried straight down (full width) to a level ring a hair under
        # its LOWEST point (on a tilted skin the first ring is tilted; a bowl hung from its middle turned
        # inward right under the skin on the high side, and the junction's tube, the sheath's first part,
        # stood outside it); then the bowl, a quarter ellipse down to its centre
        # (and at least as deep as the junction's tube reaches, ``junction_spec``: the tube is inside the sheath
        # whatever the skin under the foot does). It is the root section itself, centred on the axis: the first
        # ring's vertices on the high side of a tilted skin take their frames from the bent path above the skin
        # point, and a stem copied from that ring was off-centre (the tube stood 0.002 out of it)
        rad0 = 1.0 + (0.0 if lobe == 0.0 else lobe * np.cos(3.0 * (th - keel)))
        z_stem = min(float(foot[0][:, 2].min()), -TUBE_DEPTH * p.sink) - STEM_DROP * Ra
        stem = np.stack([a0 * rad0 * ct, b0 * rad0 * st_, np.full(nv, z_stem)], axis=1)
        # (a closing cap, not a second stem: 0.85 x (sink + cap) sent a flared ram's bowl through the midline of the head
        # into the other horn's)
        depth = BOWL_DEPTH * Ra
        c0 = stem.mean(axis=0)
        bowl = []
        for deg in BOWL_RINGS[::-1]:
            phi = math.radians(deg)
            ring = c0 + (stem - c0) * math.cos(phi)
            bowl.append(ring - np.array([0.0, 0.0, 1.0]) * depth * math.sin(phi))
        bowl.append(stem)
        return bowl, foot, shaft_s

    def fillet_generatrix(self, room: float | None = None):
        """``(h, f, t)`` of a fillet foot (``HornParams.fillet_foot``: the skin-covered ossicone), from its buried end
        up: the nominal height over the LOCAL skin, what the fillet adds to the column's radius there, and how far the
        ring has gone from following the skin (0) to the planar ring of the shaft (1).

        The fillet is a quarter circle of radius ``rho`` (``HornParams.fillet_rho``; A29: 0.5 base radii), tangent
        to the column at ``h = rho`` and, continued, to the skin. It is cut where it is ``gamma`` degrees off the
        skin and goes on straight from there, through the skin and ``dive[0]`` base radii under it (a sheet that
        ends tangent to the skin it lies on stands off it wherever the two surfaces' facets disagree: the lip the
        round-3 cuff drew), then more steeply to ``dive[1]``. The arc is sampled every ``FILLET_STEP`` degrees, so
        the profile turns that much a ring at most; above it the column's rings take ``FILLET_BLEND`` base radii to
        become planar, spaced like the shaft's. The radius is the column's own taper at each vertex's height plus
        the fillet: the profile is C1 everywhere (round 3's foot rings bunched under the first shaft ring and its
        slope jumped from -0.30 to -0.09 there: the tone band at 55 % of the height).

        ``room`` is how high the foot may reach over the skin point (a stub: the head limits a big radius and the length
        keeps the proportions): the fillet and the blend are scaled down together to fit under it."""
        p, R = self.p, self.R
        row = p.row
        rho = p.fillet_rho
        blend = FILLET_BLEND * R
        if room is not None and rho + blend > room:
            f = max(room / (rho + blend), 0.15)
            rho, blend = rho * f, blend * f
        g = math.radians(float(row.get("gamma", 12.0)))
        d1, d2 = (float(x) * R for x in row.get("dive", (0.06, 0.12)))
        phis = np.radians(np.append(np.arange(math.degrees(g), 90.0 - 1e-6, FILLET_STEP), 90.0))
        arc_h, arc_f = rho * (1.0 - np.cos(phis)), rho * (1.0 - np.sin(phis))
        f_g, h_g = float(arc_f[0]), float(arc_h[0])
        line_h = np.array([-d1, -0.6 * d1, -0.25 * d1, 0.3 * h_g])
        line_f = f_g + (h_g - line_h) / math.tan(g)
        deep_f = float(line_f[0]) + (d2 - d1) / math.tan(math.radians(40.0))
        top = rho + blend
        step = min(self.L / max(int(p.segments), 3), 0.25 * blend)
        col_h = np.linspace(rho, top, max(int(math.ceil((top - rho) / step)), 4) + 1)[1:]
        h = np.concatenate([[-d2], line_h, arc_h, col_h])
        f = np.concatenate([[deep_f], line_f, arc_f, np.zeros(len(col_h))])
        t = smoothstep((h - rho) / (top - rho))
        return h, f, t

    def _fillet_foot(self, field):
        """``(bowl, foot, shaft_s)`` of a fillet foot (:meth:`fillet_generatrix`): every ring lowered onto the skin
        under its OWN vertices (three passes: a wide ring on a tilted, curved scalp), the column's rings blended into
        planar ones standing on the skin's highest point round the column, the buried stem and bowl as for a keratin
        foot."""
        p, R = self.p, self.R
        nv = int(p.ring_verts)
        th = np.linspace(0.0, 2.0 * np.pi, nv, endpoint=False)
        ct, st_ = np.cos(th), np.sin(th)
        a0, b0 = (float(x) for x in self.semi_axes(0.0, flare=False))
        z_col = field.height(np.stack([a0 * ct, b0 * st_], axis=1))
        Z0 = float(z_col.max())
        # (a stub's foot is scaled to end under 0.85 of its length: a big radius the head limited, length kept in proportion)
        h_all, f_all, t_all = self.fillet_generatrix(room=0.85 * self.L - max(Z0, 0.0))
        foot = []
        for h, f, t in zip(h_all, f_all, t_all):
            z = (1.0 - t) * (z_col + h) + t * (Z0 + h)
            for _ in range(3):
                zc = np.maximum(z, 0.0)
                c, tt, nn, bb = self.frames_at(zc)
                a, b = self.semi_axes(zc, flare=False)
                v = c + ((a + f) * ct)[:, None] * nn + ((b + f) * st_)[:, None] * bb
                z = (1.0 - t) * (field.height(v[:, :2]) + h) + t * (Z0 + h)
            zc = np.maximum(z, 0.0)
            c, tt, nn, bb = self.frames_at(zc)
            a, b = self.semi_axes(zc, flare=False)
            v = c + ((a + f) * ct)[:, None] * nn + ((b + f) * st_)[:, None] * bb
            v[:, 2] = np.where(z > 0.0, v[:, 2], z)
            foot.append(v)
        s_top = Z0 + float(h_all[-1])
        st = self._stations
        shaft_s = st[st > s_top + 0.04 * R]
        if not len(shaft_s):
            shaft_s = np.array([min(s_top + 0.5 * (self.L - s_top), self.L)])
        # the buried stem (level, under the deepest ring, at the COLUMN's width: the dive turns in under the skin, a stem
        # as wide as the dive put three quarters of the horn's volume in the head) and the bowl that closes it
        Ra = p.root_radius
        z_stem = min(float(foot[0][:, 2].min()), -TUBE_DEPTH * p.sink) - STEM_DROP * Ra
        stem = np.stack([a0 * ct, b0 * st_, np.full(nv, z_stem)], axis=1)
        c0 = stem.mean(axis=0)
        bowl = []
        for deg in BOWL_RINGS[::-1]:
            phi = math.radians(deg)
            ring = c0 + (stem - c0) * math.cos(phi)
            bowl.append(ring - np.array([0.0, 0.0, 1.0]) * BOWL_DEPTH * Ra * math.sin(phi))
        bowl.append(stem)
        return bowl, foot, shaft_s

    def skin(self, field=None) -> Mesh:
        """The keratin sheath, closed: from the bottom of the buried bowl up the foot, the shaft, to a
        rounded tip (see :meth:`foot_rings`)."""
        p = self.p
        nv = int(p.ring_verts)
        lobe = float(p.row["lobe"])
        keel = math.pi
        bowl, foot, shaft_s = self.foot_rings(field)
        rings = list(bowl) + list(foot)
        c, t, n, b = self.frames_at(shaft_s)
        a, bb = self.semi_axes(shaft_s, flare=False)
        for i in range(len(shaft_s)):
            rings.append(_ring(c[i], n[i], b[i], a[i], bb[i], nv, lobe, keel))
        rings += self._cap_rings(c[-1], t[-1], n[-1], b[-1], a[-1], bb[-1], nv, lobe, keel,
                                 degrees=p.row.get("cap", TIP_CAP))
        return A._loft(rings)

    # -- the bone ---------------------------------------------------------------------------------

    def core_radius(self, s, skin_thick: float | None = None):
        """The cornual process' mean radius at ``s`` (local units).

        A keratin horn (A05, goat, scale bar): a LINEAR cone, ``CORE_BASE`` x R at the skin point, 0.75 / 0.5 / 0.25
        of that at 25 / 50 / 75 % of the core (``core`` x L), continued straight into the skull below the skin point,
        never nearer the sheath's surface than ``CORE_MIN_WALL`` R. (Round 3 widened it below the skin into a
        pedestal 1.12 R wide that stepped to the cone at the foot's top: the critic's "plumbing reducer"; the cone now
        flares into the skull along a concave fillet, :meth:`bone`.)

        A skin-covered horn (the ossicone, A29 / A30 / A16): the bone FILLS it, the column a skin thickness
        (``skin_thick``, local units; default 0.35 R) under the surface, to a dome under the skin's dome."""
        p, R, L = self.p, self.R, self.L
        s = np.asarray(s, dtype=np.float64)
        if p.row["material"] == "skin":
            t = self.bone_inset(skin_thick)
            return np.maximum(self.taper(np.clip(s, 0.0, L)) - t, 0.05 * R) * (s <= L)
        Lc = max(float(p.row["core"]) * L, 1e-9)
        cone = CORE_BASE * R * (1.0 - np.minimum(s / Lc, 1.0))
        cap = self.radius(np.maximum(s, 0.0)) - CORE_MIN_WALL * R
        return np.minimum(cone, np.maximum(cap, 0.02 * R)) * (s <= Lc)

    def bone_inset(self, skin_thick: float | None = None) -> float:
        """How far under a skin-covered horn's surface its bone is (local units): the skin's thickness, at most
        ``0.35`` base radii (a slender ossicone keeps a bone column)."""
        t = 0.35 * self.R if skin_thick is None else float(skin_thick)
        return min(t, 0.35 * self.R)

    def bone(self, field=None, plate=None, skin_thick: float | None = None) -> Mesh:
        """The cornual process as one closed solid (local frame): a column of :meth:`core_radius` that flares into
        the skull's outer table along a concave quarter circle of ``CORE_FLARE`` base radii (tangent to the table,
        A05: the core grows out of the frontal bone with no step), up the shaft to the end of the core: a blunt
        point on a keratin horn, a dome under the skin's dome on an ossicone.

        ``plate`` is the outer table's height over the horn's local ``xy`` (a callable ``xy -> z``: the unmoulded skin
        less the scalp, ``HornPlacement.layers_local``); without it, ``field`` (the skin) less ``SCALP_THICK``. The
        rings near the table follow it (each vertex at its own height over the table) and become planar over
        ``FILLET_BLEND`` base radii above the fillet, as the ossicone's skin does."""
        p, R = self.p, self.R
        nv = int(p.ring_verts)
        asp = float(p.row["aspect"])
        th = np.linspace(0.0, 2.0 * np.pi, nv, endpoint=False)
        ct, st_ = np.cos(th), np.sin(th)
        if field is None:
            tau = math.radians(p.tilt)
            field = PlaneField((-math.sin(tau), 0.0, math.cos(tau)))
        if plate is None:
            plate = lambda xy: field.height(xy) - SCALP_THICK          # noqa: E731
        skin = p.row["material"] == "skin"
        end = self.L if skin else float(p.row["core"]) * self.L
        if skin:
            end -= self.bone_inset(skin_thick)
        rho = CORE_FLARE * R
        # the generatrix: h over the table (the fillet's arc, then the column, blending to planar rings)
        phis = np.radians(np.arange(0.0, 90.0 + 1e-6, 10.0))
        arc_h, arc_f = rho * (1.0 - np.cos(phis)), rho * (1.0 - np.sin(phis))
        top = rho + FILLET_BLEND * R
        col_h = np.linspace(rho, top, 6)[1:]
        h_all = np.concatenate([[-0.5 * rho], arc_h, col_h])
        f_all = np.concatenate([[rho + 0.5 * rho / math.tan(math.radians(40.0))], arc_f, np.zeros(len(col_h))])
        t_all = smoothstep((h_all - rho) / (top - rho))
        r0 = float(self.core_radius(0.0, skin_thick))
        z_col = plate(np.stack([r0 * ct, r0 * asp * st_], axis=1))
        Z0 = float(z_col.max())
        rings = []
        for h, f, t in zip(h_all, f_all, t_all):
            z = (1.0 - t) * (z_col + h) + t * (Z0 + h)
            for _ in range(3):
                rc = self.core_radius(np.minimum(z, end * 0.999), skin_thick)
                xy = np.stack([(rc + f) * ct, (rc * asp + f) * st_], axis=1)
                z = (1.0 - t) * (plate(xy) + h) + t * (Z0 + h)
            zc = np.minimum(z, end * 0.999)
            rc = self.core_radius(zc, skin_thick)
            c, tt, nn, bb = self.frames_at(np.maximum(zc, 0.0))
            v = c + ((rc + f) * ct)[:, None] * nn + ((rc * asp + f) * st_)[:, None] * bb
            v[:, 2] = np.where(z > 0.0, v[:, 2], z)
            rings.append(v)
        z_top = Z0 + float(h_all[-1])
        st = self._stations
        up = st[(st > z_top + 0.04 * R) & (st < end * 0.995)]
        if len(up):
            c, t, n, b = self.frames_at(up)
            rr = self.core_radius(up, skin_thick)
            for i in range(len(up)):
                rings.append(_ring(c[i], n[i], b[i], rr[i], rr[i] * asp, nv))
            tail = self._cap_rings(c[-1], t[-1], n[-1], b[-1], rr[-1], rr[-1] * asp, nv, 0.0, 0.0,
                                   degrees=p.row.get("cap", TIP_CAP) if skin else TIP_CAP)
        else:
            top_ring = rings[-1]
            c = top_ring.mean(axis=0)
            tail = [c + (top_ring - c) * math.cos(math.radians(d)) + np.array([0.0, 0.0, 1.0]) * 0.3 * R
                    * math.sin(math.radians(d)) for d in TIP_CAP]
        # the bottom: a shallow bowl under the deepest ring (inside the table)
        low = rings[0]
        c0 = low.mean(axis=0)
        bowl = []
        for deg in BOWL_RINGS[::-1]:
            phi = math.radians(deg)
            ring = c0 + (low - c0) * math.cos(phi)
            ring[:, 2] = low[:, 2] - 0.25 * rho * math.sin(phi)
            bowl.append(ring)
        return A._loft(bowl + rings + tail)

    # -- numbers -----------------------------------------------------------------------------------

    def centreline(self, n: int = 60):
        """``(s, points)``: the centreline at ``n`` stations from the skin to the tip."""
        s = np.linspace(0.0, self.L, n)
        c, *_ = self.frames_at(s)
        return s, c


# --------------------------------------------------------------------------
# the skin around the root


@dataclass(frozen=True)
class SkinPlan:
    """The numbers of the skin around one horn root, in WORLD units (the soft tissue layer
    and the junction spec both come from them)."""

    tent_radius: float       # rho_m: where the tent ends
    tent_height: float       # H: the tent's height over the skull at the horn
    fillet: float            # the tube fillet's width, in base radii (0: none)
    fillet_height: float     # ... and its height over its width


def skin_plan(params: HornParams, site: Site, fillet: float, fillet_height: float) -> SkinPlan:
    """The tent of one horn: radius ``TENT_RADIUS`` base radii (at least ``TENT_FLOOR``: the cage is
    0.125 an edge, and a tent under three edges across cannot be resolved), kept ``TENT_MARGIN`` off
    the midline; height ``tent_height`` base radii (at least ``TENT_MIN``)."""
    R = site.radius
    x0 = site.offset                   # from the head's own midline (round 4: it was world x = 0)
    tent = params.row.get("tent") or (TENT_RADIUS, params.tent_height, TENT_MIN)
    rho = min(max(float(tent[0]) * R, TENT_FLOOR * site.scale), max(x0 - TENT_MARGIN, 0.5 * TENT_FLOOR))
    H = max(params.tent_height * R, float(tent[2]) * site.scale)
    # the scalp's fillet is at most FILLET_MAX base radii wide (the parameter goes to 4: a ram's 4 moved 400
    # vertices of the temple and the validity pass backed tens of them off), and it stays TENT_MARGIN off the
    # midline like the tent (a horn close to the midline, a random face's, has a narrow fillet on that side)
    room = (x0 - TENT_MARGIN) / max(R, 1e-9) - (1.0 + params.flare_eff)
    width = min(float(fillet), FILLET_MAX, max(room, MIN_FILLET))
    # a shape whose own foot is the fillet (the ossicone) asks the skin for a narrower, lower one of its own under it
    # (``row["tube"]``: width and height factors): the scalp only has to meet the foot's fillet, not to be it
    tube = params.row.get("tube")
    if tube is not None and width > 0.0:
        width, fillet_height = width * float(tube[0]), float(fillet_height) * float(tube[1])
    return SkinPlan(rho, H, width, float(fillet_height))


def moulded_cage(base, verts, specs):
    """``(vertices, quads)`` of the body cage with the skin moulded around ``specs`` (one spec or the
    specs of a pair, as :class:`~.kinds.Placed.junction` gives them to the figure): exactly what
    :func:`~.junction.apply_junctions` gives the body (``junction.moulded_cage``: the same displacement,
    the head's silhouette bound, the join guard and the validity pass). The horn's foot and its soft
    tissue are built against this surface, so they meet the skin the body will really have (the
    round-1 version re-implemented the raw law without the guards and missed the core's head bound)."""
    return J.moulded_cage(base, verts, specs)


# the temporalis under a horn on the temple crest (the ram): H05 (Gray 382: the fan fills the temporal fossa under the
# superior temporal line, its bulk behind the orbit) and H14 (a hypertrophied one bulges the temple from the brow up to
# the temporal line). In head units (x ``Head.k``): its centre this far down the head and back from the horn's root,
# its radius across the fan, how much longer it is front to back, and the swell it lifts the skin by
TEMPORALIS = dict(down=0.24, back=0.10, radius=0.30, stretch=1.45, height=0.032)


def temporalis_site(cage, head: Head, site: Site):
    """``(centre, normal, axis)`` (world) of the temporalis under a horn on the temple crest: ``TEMPORALIS`` down and
    back from the root in the head's own axes, on the skull's surface (the unmoulded cage), its long axis forward."""
    from . import collar as CO
    k = head.k
    T = TEMPORALIS
    c0 = np.asarray(site.point, dtype=np.float64) - T["down"] * k * head.up - T["back"] * k * head.fwd
    v, q = cage
    q = np.asarray(q)
    patch = CO.SkinPatch(v, np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]), c0, 0.8 * k + 0.2)
    pt, n, _, _ = patch.closest(c0[None, :])
    fwd = head.fwd - float(head.fwd @ n[0]) * n[0]
    return pt[0], n[0], fwd / max(float(np.linalg.norm(fwd)), 1e-12)


def temporalis_lens(moulded, raw, frame, k: float, skin_thick: float, scalp_thick: float,
                    n_rho: int = 6, n_theta: int = 32) -> Mesh:
    """The temporalis as a lens in ``frame`` (world, 4x4: ``z`` the skull's normal at its centre, ``x`` forward): over an
    ellipse ``TEMPORALIS`` wide, between the skull's outer table (the unmoulded skin, ``raw``, less the scalp) and the
    moulded skin less the skin (``moulded``): thickest where the junction's swell lifts the skin. Both fields are
    :class:`SkinField` in that frame. Returned in the world frame."""
    T = TEMPORALIS
    r = T["radius"] * k
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    u = np.linspace(0.06, 1.0, n_rho)[:, None]
    ex = r * T["stretch"] * np.cos(theta)[None, :] * u
    ey = r * np.sin(theta)[None, :] * u
    xy = np.stack([ex, ey], axis=2).reshape(-1, 2)
    z_top = (moulded.height(xy) - skin_thick).reshape(n_rho, n_theta)
    z_bot = (raw.height(xy) - scalp_thick).reshape(n_rho, n_theta)
    z_top = np.maximum(z_top, z_bot + 0.2 * (scalp_thick - skin_thick))
    mesh = _annulus_solid(np.hypot(ex, ey), theta, z_top, z_bot, n_theta)
    # (the annulus helper lays its rings out on circles: put them back on the ellipse)
    pts = mesh.vertices.copy()
    nb = n_rho * n_theta
    for half in (0, nb):
        pts[half:half + nb, 0] = ex.reshape(-1)
        pts[half:half + nb, 1] = ey.reshape(-1)
    m = np.asarray(frame, dtype=np.float64)
    return Mesh(pts @ m[:3, :3].T + m[:3, 3], mesh.faces)


def junction_spec(name: str, model: "HornModel", site: Site, plan: SkinPlan, keeps_world=(),
                  silhouette: float | None = HEAD_SILHOUETTE, temporalis=None):
    """The :class:`~.junction.JunctionSpec` of one LEFT horn in the horn's LOCAL frame (the kind
    carries it to each side with ``transformed``): the sheath's first part as the tube the scalp
    flows into, and the pedestal's tent as a mound that lifts the skin along the horn's own axis
    (not the skin normal: that is what made a plateau and a ledge across the forehead). ``keeps_world``
    are ``(centre, radius)`` in the horn's local frame: the ear, the eye, the other horn."""
    p = model.p
    R = model.R
    sc = site.scale
    nv = int(p.ring_verts)
    reach = 1.5 * p.sink + 1.5 * max(plan.fillet, 0.5) * R
    s = np.unique(np.concatenate([np.linspace(-p.sink * TUBE_DEPTH, 0.0, 4),
                                  np.linspace(0.0, reach, 8)[1:]]))
    c, t, n, b = model.frames_at(s)
    radii = model.inscribed(s) * HUG
    mounds = ()
    if plan.tent_height > 0 and plan.tent_radius > 0:
        mounds = (J.Mound((0.0, 0.0, 0.0), plan.tent_radius / sc, plan.tent_height / sc,
                          direction=(0.0, 0.0, 1.0)),)
    if temporalis is not None:
        # the skin's swell over the temporalis (H14), along the skin's own normal (``temporalis``: centre and long axis in
        # the horn's local frame, radius and height in local units)
        ctr, axis_l, rad, height = temporalis
        mounds = mounds + (J.Mound(tuple(ctr), rad, height, axis=tuple(axis_l), stretch=TEMPORALIS["stretch"]),)
    keep = tuple(J.Keep(tuple(ctr), rad, 0.6) for ctr, rad in keeps_world)
    tube_w = plan.fillet
    H_f = plan.fillet_height * tube_w * R
    move = 1.2 * (max(plan.tent_height / sc, temporalis[3] if temporalis is not None else 0.0) + H_f)
    # owner: the head (a horn moulds no other part); conform=False: the tube is the sheath's own first
    # part, inside it by construction (the core would cut it to the finished mesh, which does not exist
    # yet when the foot is built against the moulded skin: tests check the tube lies in the built sheath)
    return J.JunctionSpec(name, tuple(map(tuple, c)), tuple(radii),
                          width=tube_w if tube_w > 0 else 1.0, base_radius=R,
                          height=plan.fillet_height, sharpness=2.0, strength=1.0 if tube_w > 0 else 1.0,
                          mounds=mounds, keep=keep, max_move=move, owner="head", conform=False,
                          silhouette=silhouette)


def collar_axis(model: "HornModel", matrix, spec):
    """``(points, radii)`` in the world frame: the junction tube of ``spec`` (root first) carried on along the
    sheath's centreline to its tip, every half base radius, with the sheath's inscribed radius: the axis the root
    collar (``collar.py``) shoots its rays from."""
    pts = np.asarray(spec.points, dtype=np.float64)
    rad = np.asarray(spec.radii, dtype=np.float64)
    inv = np.linalg.inv(matrix)
    s_end = float((pts[-1] @ inv[:3, :3].T + inv[:3, 3])[2])
    step = 0.5 * model.R
    s = np.arange(s_end + step, model.L - 0.5 * step, step)
    if not len(s):
        return pts, rad
    c, *_ = model.frames_at(s)
    sc = float(abs(np.linalg.det(np.asarray(matrix)[:3, :3])) ** (1.0 / 3.0))
    world = c @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3]
    r = np.maximum(model.inscribed(s) * HUG * sc, 1e-4)
    return np.concatenate([pts, world]), np.concatenate([rad, r])


LAYER_DEPTH = 0.02       # units: the soft tissue keeps at least this far under the DRAWN skin, the bone outside the sheath
BONE_DEPTH = 0.04        # ... this far (under the soft tissue: a lens pressed in by the skin uncovered the plate below it)


def tuck_under_skin(layers: dict, skin, matrix, sheath: Mesh, centre, radius: float, inside=None) -> dict:
    """``layers`` (``{name: Mesh}`` in the horn's LOCAL frame) with every vertex that is not ``LAYER_DEPTH`` under the
    drawn skin (nor, for the bone, inside the sheath) moved in along the skin's normal to that depth. ``skin`` is the drawn skin near
    the horn as ``(vertices, quads)`` (the moulded cage subdivided once: the figure's default level; level 2 lies within
    0.006 of it). The lens and the plate are height fields along the horn's axis: where the skull curves away from the
    axis (the ram's temple) their rims stood out of the head (round 3: 4-13 vertices, up to 0.1 base radii)."""
    from . import collar as CO
    from . import junction_checks as JC
    v, q = skin
    m = np.asarray(matrix, dtype=np.float64)
    inv = np.linalg.inv(m)
    tris = np.concatenate([np.asarray(q)[:, [0, 1, 2]], np.asarray(q)[:, [0, 2, 3]]])
    patch = CO.SkinPatch(v, tris, centre, radius)
    sw = sheath.transform(m)
    shells = JC.Shells(sw.vertices, sw.faces)
    out = {}
    for name, mesh in layers.items():
        P = mesh.vertices @ m[:3, :3].T + m[:3, 3]
        # (the bone's core stands in the sheath above the skin; the soft tissue is under the skin everywhere)
        keep = shells.inside(P) if name == "bone" else np.zeros(len(P), dtype=bool)
        depth = BONE_DEPTH if name == "bone" else LAYER_DEPTH
        for _ in range(3):                    # (a moved vertex can find another nearest point: settle it)
            _, n, sd, _ = patch.closest(P)
            move = (sd > -depth) & ~keep
            if not move.any():
                break
            P = P.copy()
            P[move] -= (sd[move] + depth)[:, None] * n[move]
        _, _, sd, _ = patch.closest(P)
        left = np.nonzero((sd > -depth) & ~keep)[0]
        if len(left) and inside is not None:
            # (a vertex between two sheets of skin, the eyelid's fold, finds the other sheet's normal and does not settle:
            # slide it toward a point deep in the head until it is under the skin by the depth)
            c_in = np.asarray(inside, dtype=np.float64)
            for t in np.linspace(0.05, 1.0, 20):
                q = P[left] + t * (c_in - P[left])
                _, _, sq, _ = patch.closest(q)
                ok = sq <= -depth
                P[left[ok]] = q[ok]
                left = left[~ok]
                if not len(left):
                    break
        out[name] = Mesh(P @ inv[:3, :3].T + inv[:3, 3], mesh.faces)
    return out


# --------------------------------------------------------------------------
# the soft tissue and the skull plate under the tent


def _annulus_solid(rho, theta, z_top, z_bot, n_theta):
    """A closed solid between two surfaces over an annulus of the horn's local ``xy`` plane: ``rho``
    (the radii, inner first: ``(n_r,)`` or ``(n_r, n_theta)`` for an outline that depends on the angle),
    ``z_top`` / ``z_bot`` as ``(n_r, n_theta)`` arrays. The top and bottom surfaces are joined by an inner
    and an outer wall: a torus."""
    rho = np.asarray(rho, dtype=np.float64)
    if rho.ndim == 1:
        rho = np.repeat(rho[:, None], n_theta, axis=1)
    n_r = len(rho)
    ct, st = np.cos(theta), np.sin(theta)
    xy = np.stack([rho * ct[None, :], rho * st[None, :]], axis=2)         # (n_r, n_theta, 2)
    top = np.concatenate([xy, z_top[:, :, None]], axis=2).reshape(-1, 3)
    bot = np.concatenate([xy, z_bot[:, :, None]], axis=2).reshape(-1, 3)
    verts = np.concatenate([top, bot])
    nb = n_r * n_theta
    quads = []
    for i in range(n_r - 1):
        for j in range(n_theta):
            jn = (j + 1) % n_theta
            a, b = i * n_theta + j, i * n_theta + jn
            c, d = (i + 1) * n_theta + jn, (i + 1) * n_theta + j
            quads.append([a, b, c, d])                       # top
            quads.append([nb + a, nb + d, nb + c, nb + b])   # bottom, reversed
    for j in range(n_theta):                                 # the inner and the outer walls
        jn = (j + 1) % n_theta
        a, b = j, jn
        quads.append([nb + a, nb + b, b, a])
        a, b = (n_r - 1) * n_theta + j, (n_r - 1) * n_theta + jn
        quads.append([a, b, nb + b, nb + a])
    tris = []
    for q in quads:
        tris.append([q[0], q[1], q[2]])
        tris.append([q[0], q[2], q[3]])
    mesh = Mesh(verts, np.asarray(tris, dtype=np.int64))
    if mesh.volume() < 0:
        mesh = Mesh(mesh.vertices, mesh.faces[:, [0, 2, 1]])
    return mesh


def _outline(model: "HornModel", raw, rho_max: float, n_rho: int, n_theta: int, avoid=()):
    """``(rho, theta)`` of the annulus the layers under the skin are built over: from the cornual process
    out to ``rho_max``, but per angle no farther than the rays along the horn's axis still find the skull
    (on a strongly curved temple, the ram's, the far side of the annulus is past the head's silhouette and the
    fallback plane there stood 0.4 radii out of the head)."""
    rho_in = max(float(model.core_radius(0.0)), 0.5 * model.R)
    rho_out = max(rho_max, rho_in * 1.5)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    probe = np.linspace(rho_in, rho_out, 25)
    xy = np.stack([np.outer(probe, np.cos(theta)), np.outer(probe, np.sin(theta))], axis=2).reshape(-1, 2)
    z, hit = raw.height(xy, hits=True)
    # (round 4: nor where the skin under it is the eye's or the ear's, ``avoid`` = the junction's keeps in the horn's
    # frame: the ram's lens reached the brow, into the eyelid's fold, and stood out of it)
    pts = np.concatenate([xy, z[:, None]], axis=1)
    for ctr, rad in avoid:
        hit &= np.linalg.norm(pts - np.asarray(ctr, dtype=np.float64), axis=1) > float(rad)
    hit = hit.reshape(len(probe), n_theta)
    # the first miss along each ray from the inside ends the annulus there (at least a quarter of the way out)
    miss = ~hit
    first = np.where(miss.any(axis=0), miss.argmax(axis=0), len(probe))
    reach = probe[np.clip(first - 1, 0, len(probe) - 1)]
    reach = np.where(first == len(probe), rho_out, np.maximum(reach, rho_in + 0.25 * (rho_out - rho_in)))
    # smooth around the ring (no notch in the outline)
    for _ in range(3):
        reach = np.minimum(reach, 0.25 * np.roll(reach, 1) + 0.5 * reach + 0.25 * np.roll(reach, -1) + 1e-12)
    u = np.linspace(0.0, 1.0, n_rho)[:, None]
    return rho_in + u * (reach[None, :] - rho_in), theta


def soft_tissue(model: "HornModel", moulded, raw, rho_max: float, skin_thick: float, scalp_thick: float,
                n_rho: int = 7, n_theta: int = 32, avoid=()) -> Mesh:
    """The ``muscle`` layer in the horn's local frame: the soft tissue the scalp is tented over (galea,
    periosteum, the horn's germinal cuff), a lens around the cornual process from its surface out to
    ``rho_max`` (the tent's radius; less where the skull curves away, ``_outline``): its upper surface is the
    MOULDED skin less the skin's thickness, its lower one the skull's outer table, the RAW skin less the whole
    scalp's thickness. So it is thickest at the horn (the tent's height more) and ``scalp - skin`` thick where
    the skin is not lifted."""
    rho, theta = _outline(model, raw, rho_max, n_rho, n_theta, avoid)
    xy = np.stack([rho * np.cos(theta)[None, :], rho * np.sin(theta)[None, :]], axis=2).reshape(-1, 2)
    z_top = (moulded.height(xy) - skin_thick).reshape(n_rho, n_theta)
    z_bot = (raw.height(xy) - scalp_thick).reshape(n_rho, n_theta)
    z_top = np.maximum(z_top, z_bot + 0.2 * (scalp_thick - skin_thick))
    return _annulus_solid(rho, theta, z_top, z_bot, n_theta)


def skull_plate(model: "HornModel", raw, rho_max: float, scalp_thick: float, bone_thick: float,
                n_rho: int = 7, n_theta: int = 32, avoid=()) -> Mesh:
    """The skull's outer table around the horn, a plate ``bone_thick`` thick (the cornual process grows
    out of it): the lower boundary of the soft tissue, over the same outline (``_outline``)."""
    rho, theta = _outline(model, raw, rho_max, n_rho, n_theta, avoid)
    xy = np.stack([rho * np.cos(theta)[None, :], rho * np.sin(theta)[None, :]], axis=2).reshape(-1, 2)
    z_top = (raw.height(xy) - scalp_thick - PLATE_GAP).reshape(n_rho, n_theta)
    return _annulus_solid(rho, theta, z_top, z_top - bone_thick, n_theta)
