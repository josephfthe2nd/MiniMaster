"""The tail as an anatomy: a caudal vertebra chain, the muscle sleeve around it, the skin over both.

A tail is not a tube pushed through the skin. It is the continuation of the spine: the sacrum
ends in an apex, the caudal vertebrae Cd1..Cdn follow it along the same axis, the sacro- and
ischio-caudal muscles arise on the sacrum and ilium and act on the distal vertebrae through long
tendons (so the bellies sit on the proximal third and the rest is tendon, fascia and skin), and
the skin is a thin envelope over all of it. This module builds those three layers from ONE set of
numbers, so the skin that is printed and rendered is the envelope of the muscle and the bone plus a
skin thickness, never a second, unrelated shape:

* ``bone``   the vertebra chain (centra, transverse processes, neural spines on the first four,
             chevron bones ventrally from Cd3-4) and the sacrum it continues: a HUMAN sacrum set in the body's
             sagittal plane (round 6: tilted ~57 degrees, base anterior-superior, concave anteriorly, seated under the
             body's own back skin), whose apex segments turn back into the tail; the caudal column leaves the apex
             along it and bends into the tail's own axis over the first caudals (``bone_path``);
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
  the longest) to a maximum at 29 % of the chain and fall to 0.35 at the tip;
* the radius at the skin point is at most ``ROOT_CAP`` (the flare saturates: what a pelvis carries), the flare eases out
  no steeper than ``FLARE_SLOPE``, a dorsal ridge rises out of the root over ``CREST_RISE`` R (round 6).
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
FLARE_LAMBDA = 0.9       # decay of the root flare along the axis, in base radii ...
FLARE_SLOPE = 0.65       # ... or longer: the trumpet's wall leaves the skin at most this steep (radius per length; round 6)
ROOT_CAP = 0.9           # the radius a pelvis carries at the root (units of the tail parameters, neutral body; round 6): the
                         # flare saturates there (a base radius of 1.2 with its flare covered both buttocks, the cleft and
                         # the thighs from behind, the round-5 critic); the kind's base_radius stops there too
CREST_RISE, CREST_WIDTH = 2.5, 0.42   # the dorsal ridge rises out of the root over this many R; its half width (radians)
ROOT_ASPECT = 1.08       # lateral : vertical semi-axis at the skin point
MID_ASPECT = 0.92        # ... beyond 0.3 L (slightly taller than wide)
STIFF_R = 2.5            # the stiff proximal part, in base radii
DROOP_DEG = 85.0         # total sagittal droop per unit of the ``droop`` parameter
CURL_DEG = 45.0          # distal up-curl per unit of ``curl``
DROOP_POW = 1.2          # the droop grows with this power of the distance past the stiff root
CURL_FROM = 0.55         # the curl starts at this fraction of the length
# the lift (round 4): a share of it at the sacrocaudal joint (the stem pivots there: the tail leaves the sacrum already
# raised), bounded by the joint's range (``LIFT_JOINT``: flexion, extension, degrees), the rest by the first free caudals
# OUTSIDE the body (from ``LIFT_FROM`` R beyond the skin point, over ``LIFT_LEN`` R), so a raised tail first stands off
# the back and then rises (round 2b bent it from 1.5 R inside the body: at lift 60-90 it lay on the back as a bolster,
# 7 % of it inside the body at lift 90)
LIFT_SHARE, LIFT_JOINT = 0.35, (-15.0, 25.0)
LIFT_JOINT_LEN = 1.6             # the joint's share is spread over this many R from the joint (the sacrocaudal joint and the
                                 # first two caudal joints: 0.6 R folded the collar of a tail lifted 40 - 70 degrees, 110 - 200 crossings)
LIFT_FROM, LIFT_LEN = 0.4, 3.5
LIFT_MAX_ELEVATION = 90.0        # degrees above horizontal: a lifted tail stops at the vertical
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
# the sacrum the chain grows from (round 6): a HUMAN sacrum, set in the BODY's sagittal plane (its horizontal is the body's,
# not the tail's axis: the model's ``elevation`` converts), base anterior-superior and apex posterior-inferior, CONCAVE
# ANTERIORLY: walking from the base to the apex its axis (base -> apex) steepens from ``base_deg`` to ``apex_deg`` below the
# body's horizontal (pointing back), the turn growing toward the apex (``bend_pow``), so the chord leans ~57 degrees (Gray 111,
# tail/cm_Gray_111_Vertebral_column_coloured: the chord S1 -> apex 60 - 62 degrees in the standing column, the pelvic face
# concave; Gray97 for the wedge: the body and alae deep at the base, a narrow strip at the apex). The apex lies ``apex``
# (forward, up; R_s) from the root's skin point: just above the top of the natal cleft, under the skin of the sacral
# triangle. Round 4 laid the sacrum along the tail's line (a quadruped's, 33 degrees, reaching 73 % of the body's depth with
# its curve reversed): the round-5 critic. In R_s units (a body's sacrum, not the tail's: the base radius clipped to
# ``SACRUM_R``): length along its axis, half widths base / apex, half thicknesses base / apex.
SACRUM = dict(length=2.3, w_base=1.25, w_apex=0.36, thick_base=0.62, thick_apex=0.17,
              base_deg=42.0, peak_deg=74.0, peak_at=0.4, apex_deg=45.0, apex=(0.55, 0.80), upper=(0.15, 0.85))
SACRUM_DORSAL = 0.5             # the section's dorsal half thickness over its ventral one (the bulk, S1's body and the alae, is
                                 # in front; behind are the laminae and the median crest under 1 - 2 cm of tissue)
SACRUM_CLEAR = 0.25              # the sacrum is seated at least this many R_s under the body's own skin (the cage's), along the
                                 # body's front: ``PlacedTail.sacrum_shift`` (a shallow lower back: round 6's first sacrum put 7 - 18 %
                                 # of it outside the female, short and slim bodies)
SACRUM_SHIFT_MAX = 0.6           # ... by at most this many R_s (the bodies need 0.05 - 0.4; a tail placed off the midline, whose
                                 # frame is not the body's sagittal plane, asked for 2.0 units and threw its sacrum out of the hip)
SACRUM_R = (0.35, 0.50)          # R_s = the base radius clipped to this (units of the tail parameters, neutral body): a
                                 # thicker tail's sacrum and joint would reach the front of the pelvis (it is 4.2-4.5 R deep there)
SACRUM_JOINT = 0.06              # the sacrocaudal joint's gap before Cd1, in R_s
# the caudal column (round 6): Cd1 leaves the sacral apex turned ``CD1_TURN`` degrees from the apex's own axis toward the
# tail (it continues the sacrum: TAKEAWAYS section 3, 0 - 20), then the first caudals bend back into the tail's own axis,
# which they join ``CHAIN_JOIN`` x min(R, R_s) past the skin point (``TailModel.chain_join``; the path between turns its
# direction from the one to the other by a smoothstep, ``_blend_path``): the bend of a raised lemur's tail at its sacrocaudal
# joints (tail/cm_Lemur_catta_skelet_1782: ~100 degrees over four joints), here ~60 degrees over Cd1 - Cd3 at rest, because
# a human sacrum points down and the tail leaves pointing back.
CD1_TURN = 2.0
CHAIN_JOIN = 2.5
SLEEVE_UNDER = 0.3               # ... and lies at least this many R_s under the body's skin inside the body
SLEEVE_ORIGIN = 0.6              # the muscle sleeve's buried end narrows to this share of its section at the sacral apex


# the skin around the root (junction_spec): positions in base radii R in the local frame
# (x up the back, y lateral, z out of the body); see docs/anatomy_tail.md
R_MAX = 0.62                  # the junction's sizes (mounds, fillet, keeps) stop growing here: the base radius of a neutral-body tail
KEEP_BEADS = ((-1.1, -0.3), (-1.6, -0.8), (-1.9, -1.4), (-2.1, -2.0))   # the cleft below the root, as spheres on the midline (x, z in R)
KEEP_RADIUS, KEEP_FADE = 0.9, 0.6     # ... each this wide (R), and growing back to full over this much of it
REACH_R = 1.0                 # the skin tube reaches one fillet width + this many R past the skin point
MAX_MOVE_R = 1.4              # one cage vertex moves at most this many R (the default tail's largest move is 1.1 R)
SMOOTH = 5                    # rounds of smoothing of the tube displacement over the cage (the core's default is 20: it costs 4-7 degrees of the fillet's gain here)
# an axial mound inside the root (``(height, radius)`` in R_j: the hidden skin climbs further up the tail's wall than the
# fillet's top, so the subdivided skin does not meet the wall as a ledge) was tried in round 4 and is OFF: over six bodies
# it moved the junction-only join angle by -10 to +5 degrees (noise around 40 - 50) and, with the smoothing off, folded
# the heavy body's skin when lifted (docs/anatomy_tail.md section 10). ``None``: off.
CLIMB = None
# how the tail's skin adds to a neighbour's where their masses overlap over the sacrum (``JunctionSpec.stack``, round 6):
# "max": the tail raises the skin only by what the neighbour (extra legs' gluteal masses) has not already; summed, the two
# passed the move cap over the sacrum and the validity pass sank the midline into a pit with crumpled edges (the round-5
# integration critic, B-I1)
STACK = "max"


@dataclass(frozen=True)
class RootMass:
    """A mass of muscle that lies in the body at the root of the tail, as the junction's mound and as the bulk of a
    body-side muscle of the ``muscle`` layer (the dorsalis for the sacral wedge, the gluteal band for a gluteal mass): ONE
    description, so that the skin that is moulded over it has that muscle under it.

    ``center`` / ``half``: the footprint on the skin plane (x up the back, y lateral, in R): an ellipse (``half`` along
    its own axis, which is ``angle`` degrees from +x toward +y, and across it), the mound's bump reaches 0 at its rim;
    ``height``: the thickness of tissue the mass adds ABOVE the skin the body already had
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
    angle: float = 0.0                # the footprint's long axis, degrees from +x (up the back) toward +y
    rise: float = 0.0                 # the centre's height over the skin point's plane (R): a mass on the buttock's crown

    def mirrored(self) -> "RootMass":
        return replace(self, name=self.name.replace("_l", "_r"), center=(self.center[0], -self.center[1]),
                       angle=-self.angle)

    @property
    def axis(self) -> tuple:
        a = math.radians(self.angle)
        return (math.cos(a), math.sin(a), 0.0)


ROOT_MASSES = (
    # the dorsal caudal muscles (sacrocaudalis dorsalis, the caudal end of the multifidus) arise on the dorsal sacrum
    # and run onto the tail: a wedge on the midline whose crest is AT the top of the root and that thins out up the
    # sacrum. Centred on the root, only its falling (concave) flank shows above the tail, so the lumbar line runs down
    # a ramp onto the tail's top line; a mound centred up the back (2.4 R, the first try) made a convex hump with the
    # tail leaving from under it (the critics' "ledge", measured: tests/test_tail_junction.py)
    RootMass("sacral", (1.0, 0.0), (3.5, 1.6), 0.60, 1.0, fade=(20.0, 80.0), muscle="sacrocaudalis dorsalis / multifidus",
             along_axis=True),
    # the gluteal masses that flank the root: the coccygeal fibres of gluteus maximus with the caudofemoralis, a band from the
    # first caudals beside the root infero-laterally toward the femur (the cat's caudofemoralis; tail/wc_j42auhe2 for the
    # fibre direction). Since round 4 an oblique band within GLUTEAL_REACH of the midline (the round-3 footprint, a round
    # mass 2.2 R_j out, reached 3.8 R_j, past the body's silhouette, and its lens stood out of the hip)
    RootMass("gluteal_l", (-0.62, 1.55), (1.1, 0.6), 0.25, 0.9, muscle="gluteus maximus (coccygeal fibres) / caudofemoralis",
             angle=138.0, rise=0.3),
    RootMass("gluteal_r", (-0.62, -1.55), (1.1, 0.6), 0.25, 0.9, muscle="gluteus maximus (coccygeal fibres) / caudofemoralis",
             angle=-138.0, rise=0.3),
)
GLUTEAL_REACH = 2.5           # no body-side mass of the tail reaches further than this from the midline (R_j)

# the body-side muscles of the muscle layer (round 4; in R_j, the tail's local frame: x up the back, y lateral): their
# own shapes, each laid UNDER the skin the body is meshed with (RootSkin): the skin offset by one skin thickness is a
# ceiling, never the shape. References: tail/cm_Kangaroo_tail_anatomical_model (slender bellies along the vertebrae
# into long tendons), tail/cm_Superficial_muscles_of_a_cat (the muscles of the tail at the root, the gluteal rump
# beside it), tail/wc_j42auhe2 and wc_kpq8pswh (gluteus maximus: a broad quadrilateral sheet of parallel fibres
# fanning infero-laterally from the sacrum and the iliac crest).
# * sacrocaudalis dorsalis: a strand each side of the median sacral crest ON THE BONE, from the upper sacrum (``upto``
#   of its length from the apex) down its dorsal face (``y`` of its half width out) to the apex and into the tail's
#   dorsal belly; half width and thickness up the sacrum / at the apex (R_j), plus the sacral mound's rise near the root;
#   from ``fill[0]`` to ``fill[1]`` of its length its section turns to stand toward the skin and fills the room between the
#   bone and the skin (the mound IS this muscle's bulk);
DORSALIS = dict(upto=0.95, cover=0.45, medial=0.05, thick=(0.25, 0.6), rise=0.9, fill=0.85)
# * the gluteus maximus (round 4 drew only a band of its coccygeal fibres on the gluteal mass's footprint, inside
#   GLUTEAL_REACH; round 6:) the broad quadrilateral SHEET of the references (tail/wc_j42auhe2, wc_kpq8pswh; the round-5
#   critic: the band was a leaf 1 R_j long), (x up the back, y lateral; R_j, the +y side): its ORIGIN along the lateral border
#   of the sacrum from the iliac crest's level (``origin_top``) down past the root to the coccyx (``origin_bottom``), its
#   INSERTION toward the femur, the iliotibial tract over the greater trochanter (``insertion_top``) down to the gluteal
#   tuberosity (``insertion_bottom``): its fibres run infero-laterally (~40 degrees below the horizontal). Thick in its middle
#   (``thick`` R_j, plus the gluteal mound's rise where the mound is), thinning to every border.
GLUTEUS = dict(origin_top=(3.4, 1.15), origin_bottom=(-1.3, 0.65), insertion_top=(1.4, 3.7), insertion_bottom=(-2.0, 3.1),
               thick=0.5, rise=1.0, bundles=7, ridge=0.05)
# * sacrocaudalis ventralis: a strand each side, deep in the pelvis on the ventral face of the sacrum, into the tail's
#   ventral belly (only a cut shows it): offset from the sacrum's ventral face, half width, half thickness (R_s).
VENTRALIS = dict(offset=0.18, half_w=0.22, half_t=0.14)
MUSCLE_RING = 12              # vertices around a strand
DORSALIS_RINGS, VENTRALIS_RINGS, INTO_TAIL = 14, 10, (0.15, 0.55, 0.95, 1.35)   # body rings; rings in the tail (s, in R)
SHEET_GRID = (19, 27)          # the gluteal sheet's grid: along the fibres (origin -> insertion) x along the origin (round 6: denser, the sheet is broad)
ROOT_SKIN_REACH = 7.0         # the root skin patch: the cage within this many R_j of the root (the muscles lie within 4.5)


def sacral_angle(u):
    """The sacrum's axis (base -> apex) in degrees below the body's horizontal, pointing back, at ``u`` of its length from
    the APEX (0) to the base (1): ``base_deg`` at the base, steepening to ``peak_deg`` at ``peak_at`` (the upper sacrum is
    concave anteriorly), then turning back to ``apex_deg`` at the apex (the lowest segments extend into the tail)."""
    S = SACRUM
    u = np.asarray(u, dtype=np.float64)
    m = S["peak_at"]
    low = S["apex_deg"] + (S["peak_deg"] - S["apex_deg"]) * smoothstep(u / m)
    high = S["peak_deg"] + (S["base_deg"] - S["peak_deg"]) * smoothstep((u - m) / (1.0 - m))
    return np.where(u < m, low, high)


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
    elevation: float = 0.0       # the root axis' elevation above horizontal on the body (degrees): a lift stops at the vertical

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

    def __init__(self, params: TailParams, patch: "SkinPatch | None" = None, sacrum_shift=0.0):
        self.p = params
        p = params
        self.R, self.L = float(p.base_radius), float(p.length)
        self._patch = patch               # a SkinPatch, a function that makes one (sampled on first use) or None (flat)
        self._shift = sacrum_shift        # how far the body's sacrum is moved toward the front to lie under its skin (units),
                                          # or a function that measures it (PlacedTail.sacrum_shift: on first use)
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
        return r * (1.0 + self.flare * np.exp(-np.maximum(s, 0.0) / self.flare_length()))

    def aspect(self, s):
        """Lateral : vertical semi-axis ratio of the section at ``s``."""
        s = np.asarray(s, dtype=np.float64)
        knots_s = [0.0, self.R, 2.0 * self.R, max(0.3 * self.L, 3.0 * self.R)]
        return np.interp(s, knots_s, [ROOT_ASPECT, 1.0 + 0.6 * (ROOT_ASPECT - 1.0), 1.0, MID_ASPECT])

    def crest_height(self, s):
        """The dorsal ridge's height over the section at ``s`` (``crest`` base radii at most): it rises out of the root over
        ``CREST_RISE`` base radii past the skin point (round 6: it stood at full height AT the skin point, where it cut
        through the moulded back as a V tooth with a sawtooth side contour, the round-5 critic) and falls to nothing at
        0.6 L; none inside the body."""
        s = np.asarray(s, dtype=np.float64)
        rise = smoothstep(s / (CREST_RISE * self.R))
        fall = 1.0 - np.clip(s / (0.6 * self.L), 0.0, 1.0)
        return self.p.crest * self.R * rise * fall

    def flare_length(self) -> float:
        """The root flare's decay length along the axis: ``FLARE_LAMBDA`` base radii, longer for a big flare so that the
        trumpet's wall leaves the skin at most ``FLARE_SLOPE`` (radius per length; round 6: a flare of 2 eased out over 0.9 R
        stood up as a plate, the critic's flange)."""
        return self.R * max(FLARE_LAMBDA, self.flare / FLARE_SLOPE)

    @property
    def flare(self) -> float:
        """The root flare that is built: the parameter's, but no more than takes the root to ``ROOT_CAP`` (a tail already
        that thick has none)."""
        return float(min(max(float(self.p.flare), 0.0), max(ROOT_CAP / self.R - 1.0, 0.0)))

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
        joint, free = self.lift_split()
        s_j = self.joint_s()
        pitch = (np.radians(joint) * smoothstep((sd - s_j) / (LIFT_JOINT_LEN * R))
                 + np.radians(free) * smoothstep((sd - LIFT_FROM * R) / (LIFT_LEN * R)))
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
        # anchored at the sacrocaudal joint: what the lift turns about it moves, the stem before it does not (at rest the
        # stem is straight and the skin point is at the origin either way)
        pos -= np.array([np.interp(self.joint_s(), sd, pos[:, c]) for c in range(3)])
        pos += np.array([0.0, 0.0, self.joint_s()])
        self._sd, self._pos, self._tang, self._pitch, self._yaw = sd, pos, tang, pitch, yaw

    def lift_split(self):
        """``(at the sacrocaudal joint, in the free tail)``: how the lift (degrees) is shared. A tail is not lifted past the
        vertical (``LIFT_MAX_ELEVATION`` above horizontal, with its root axis' own ``elevation``): beyond it, it would lean
        into the back (the heavy body's lumbar bulge took 4 % of a tail lifted 90 degrees)."""
        lift = min(float(self.p.lift), LIFT_MAX_ELEVATION - float(self.p.elevation))
        joint = float(np.clip(LIFT_SHARE * lift, *LIFT_JOINT))
        return joint, lift - joint

    def joint_s(self) -> float:
        """Where the sacrocaudal joint is along the axis: the rostral end of Cd1, one base radius below the skin point (at
        most ``SACRUM_R[1]``: a thick tail's joint stays in the pelvis)."""
        return -min(self.R, SACRUM_R[1])

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
            crest = self.crest_height(s)[:, None] * np.exp(-(np.angle(np.exp(1j * th)) / CREST_WIDTH) ** 2)[None, :]
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
        if layer == "muscle":
            ring = self._sleeve_origin(s, ring)
        rings = [ring[i] for i in range(len(s))] + self._cap_rings(layer, n)
        return A._loft(rings)

    def _sleeve_origin(self, s, ring):
        """The buried part of the muscle sleeve (round 6): the caudal muscles wrap the first caudals and arise on the
        sacrum, so inside the body the sleeve follows the caudal column (its rings turn to the column's frames and centre on
        it) and narrows to ``SLEEVE_ORIGIN`` of its section at the sacral apex (their tendinous origins), instead of running
        on along the tail's axis as a straight pipe to a flat end in front of the sacrum (the round-5 critic). The buried
        stations are spread from the skin point back to the apex; the free tail is unchanged. Same rings, same vertex
        count."""
        buried = np.nonzero(s < 0.0)[0]
        if not len(buried):
            return ring
        s_lo = float(self.bone_path()[0][0])
        s_new = s_lo * s[buried] / s[buried[0]]                         # [s0, 0) -> [s_lo, 0)
        out = ring.copy()
        n = ring.shape[1]
        th = 2.0 * np.pi * np.arange(n) / n
        fr_skin = self.frames(s_new)
        fr_bone = self.bone_frames(s_new)
        c_skin, c_bone = self.centre(s_new), self.bone_centre(s_new)
        b = smoothstep(s_new / s_lo)                                    # 0 at the skin point, 1 at the origin
        sec = self.section(s_new, "muscle", n)
        for k, i in enumerate(buried):
            loc = sec[k] - c_skin[k]
            x, y = loc @ fr_skin[k][:, 0], loc @ fr_skin[k][:, 1]      # the ring in its own frame
            up = (1.0 - b[k]) * fr_skin[k][:, 0] + b[k] * fr_bone[k][:, 0]
            lat = (1.0 - b[k]) * fr_skin[k][:, 1] + b[k] * fr_bone[k][:, 1]
            up /= np.linalg.norm(up)
            lat -= (lat @ up) * up
            lat /= np.linalg.norm(lat)
            scale = 1.0 - (1.0 - SLEEVE_ORIGIN) * b[k]
            c = (1.0 - b[k]) * c_skin[k] + b[k] * c_bone[k]
            out[i] = c[None, :] + scale * (x[:, None] * up[None, :] + y[:, None] * lat[None, :])
        # inside the body the sleeve stays under the body's own skin (its height field over the root, sampled from the cage
        # before any junction: what a document with the junction off has), less a skin thickness and a margin for the
        # subdivided skin lying inside the cage on the convex back; the free tail is not touched
        P = out[buried].reshape(-1, 3)
        cap = self.patch.height(P[:, 0] / self.R, P[:, 1] / self.R) * self.R - SLEEVE_UNDER * self.Rs
        w = (1.0 - smoothstep(-np.repeat(s_new, n) / (0.5 * self.R)))      # no clamp at the skin point, full from 0.5 R in
        z = np.where((P[:, 2] > cap) & (P[:, 0] > 0.0), P[:, 2] + (1.0 - w) * (cap - P[:, 2]), P[:, 2])   # (up the back)
        P[:, 2] = z
        out[buried] = P.reshape(len(buried), n, 3)
        return out

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
        stretched along the mass's own axis), times the cleft beads' hold (:func:`keep_weight`) and the lift fade. The
        body-side muscles and the mounds of the spec are one description through this."""
        k = min(self.R, R_MAX) / self.R                      # the junction's sizes are in R_j = min(R, R_MAX)
        cx, cy = mass.center[0] * k, mass.center[1] * k
        hx, hy = mass.half[0] * k, mass.half[1] * k
        x, y, z = (np.asarray(a, dtype=np.float64) for a in (x, y, z))
        ax, ay, _ = mass.axis
        along = (x - cx) * ax + (y - cy) * ay
        across = -(x - cx) * ay + (y - cy) * ax
        u = np.sqrt((along / hx) ** 2 + (across ** 2 + (z - mass.rise * k) ** 2) / hy ** 2)
        return bump(u) * keep_weight(x, y, z, k) * lift_weight(mass.fade, self.p.lift)

    # -- the body-side muscles (round 4) ---------------------------------------------------------------------------

    @property
    def Rj(self) -> float:
        """The radius the junction's and the body-side muscles' sizes are in: ``min(R, R_MAX)``."""
        return min(self.R, R_MAX)

    def mass_rise(self, P, N):
        """The tissue the root masses add on the body's skin at the points ``P`` (local frame, units) with the skin
        normals ``N``: the sum over :data:`ROOT_MASSES` of ``height`` x :meth:`mass_weight` along the mound's own
        direction (the root's axis for the sacral wedge, the normal for the others), as a vector per point. The
        junction raises the skin by exactly these mounds (``junction_spec``); the muscles are their bulk."""
        P = np.asarray(P, dtype=np.float64)
        N = np.asarray(N, dtype=np.float64)
        out = np.zeros_like(P)
        loc = P / self.R
        k = self.Rj / self.R
        for m in ROOT_MASSES:
            w = self.mass_weight(m, loc[:, 0], loc[:, 1], loc[:, 2]) * m.height * k * self.R
            d = np.broadcast_to(np.array([0.0, 0.0, 1.0]), P.shape) if m.along_axis else N
            out += w[:, None] * d
        return out

    def _designed(self, skin: "RootSkin", xy):
        """``(top, normal, rise)``: for the local footprint points ``xy`` (units), the skin the masses alone would make
        (the body's skin before the junction, plus :meth:`mass_rise`) one skin thickness down, the skin's normal there
        and the rise along it."""
        P, N = skin.surface(xy)
        rise = self.mass_rise(P, N)
        t = float(self.skin_thickness(0.0))
        return P + rise - t * N, N, (rise * N).sum(axis=1)

    def _strand(self, skin, centres, normals, laterals, half_w, half_t, tail_rings):
        """A strand of muscle as a closed loft: elliptical rings (``half_w`` across, ``half_t`` along the skin normal)
        about ``centres`` (oriented by ``normals`` / ``laterals``), each ring pushed down along its normal until it lies
        one skin thickness under every level of the moulded skin (:meth:`RootSkin.clip`), then the rings ``tail_rings``
        (already in the tail's muscle sleeve)."""
        n = MUSCLE_RING
        th = 2.0 * np.pi * np.arange(n) / n
        rings = []
        t = float(self.skin_thickness(0.0))
        for c, nn, ll, w, h in zip(centres, normals, laterals, half_w, half_t):
            ring = c[None, :] + (w * np.cos(th))[:, None] * ll[None, :] + (h * np.sin(th))[:, None] * nn[None, :]
            rings.append(skin.clip(ring, nn, t, rounds=8))           # (along the ring's own normal, not the skin's: more rounds)
        return A._loft(rings + list(tail_rings))

    def _into_sleeve(self, side: float, y_frac: float, x_frac: float, half_w: float, half_t: float):
        """Rings of a strand inside the tail's own muscle sleeve (hidden in it): at ``INTO_TAIL`` stations along the
        axis, centred ``x_frac`` of the skin's vertical semi-axis up (down if negative) and ``y_frac`` of the lateral
        one out, the size a fraction of the vertical semi-axis."""
        n = MUSCLE_RING
        th = 2.0 * np.pi * np.arange(n) / n
        s = np.array(INTO_TAIL) * self.R
        b = self.law(s)
        a = b * self.aspect(s)
        fr = self.frames(s)
        cen = self.centre(s)
        out = []
        for i in range(len(s)):
            up, lat = fr[i][:, 0], side * fr[i][:, 1]
            c = cen[i] + x_frac * b[i] * up + y_frac * a[i] * lat
            out.append(c[None, :] + (half_w * b[i] * np.cos(th))[:, None] * lat[None, :]
                       + (half_t * b[i] * np.sin(th))[:, None] * up[None, :])
        return out

    def dorsalis(self, skin: "RootSkin", side: float) -> Mesh:
        """Sacrocaudalis dorsalis of one side (``side`` +1: +y), with the caudal multifidus: a flat band LYING ON the dorsal
        face of the sacrum beside the median crest, covering its side of the face (``cover`` of the sacrum's half width),
        from the upper sacrum down to the apex and on into the tail's dorsal belly (the kangaroo model: the bellies lie on
        the vertebrae and become long tendons). Its thickness grows toward the apex (``thick``, R_j) and fills at most
        ``fill`` of the room between the bone and the skin above it: the sacral wedge's rise is this muscle's bulk near the
        root. Round 6: round 4's strand was narrow and turned to stand toward the skin; on the steep human sacrum that stood
        up as two blades (and two knobs at the root). Every ring lies under every level of the moulded skin."""
        D = DORSALIS
        k = self.Rj
        n = DORSALIS_RINGS
        c, tang, nd, w_s, h_s = self.sacral_axis(n, upto=D["upto"])
        c, tang, nd, w_s, h_s = (a[::-1] for a in (c, tang, nd, w_s, h_s))      # from the upper sacrum down to the apex
        u = np.linspace(0.0, 1.0, n)                                          # 0 up the sacrum, 1 at the apex
        lat = np.broadcast_to(np.array([0.0, side, 0.0]), c.shape)
        w = D["cover"] * w_s                                                  # half width across the face
        off = w + D["medial"] * k                                             # beside the crest, never across the midline
        face = c + nd * (SACRUM_DORSAL * h_s)[:, None] + lat * off[:, None]   # on the dorsal face
        top, N, rise = self._designed(skin, face[:, :2])                      # the skin over it, as the masses make it
        gap = np.maximum(((top - face) * nd).sum(axis=1), 0.0)               # the room along the face's normal
        t = float(self.skin_thickness(0.0))
        want = (D["thick"][0] + (D["thick"][1] - D["thick"][0]) * u) * k + D["rise"] * np.maximum(rise, 0.0) * u
        h = 0.5 * np.clip(np.minimum(want, D["fill"] * (gap - t)), 0.06 * k, None)
        centres = face + nd * h[:, None]
        tail = self._into_sleeve(side, 0.40, 0.55, 0.26, 0.22)
        return self._strand(skin, centres, nd, lat, w, h, tail)

    def gluteal(self, skin: "RootSkin", side: float) -> Mesh:
        """Gluteus maximus of one side (with its coccygeal fibres, round 6): a broad sheet of parallel fibres from the
        lateral border of the sacrum and the coccyx beside the root, infero-laterally toward the femur (``GLUTEUS``): the
        rump's mass beside and in front of the root (TAKEAWAYS section 3; tail/wc_j42auhe2). Where the junction's gluteal mass
        raises the skin beside the root, its bulk is this sheet's. A lens of grid ``SHEET_GRID`` (along the fibres x along
        the origin), thick in its middle, thin at its borders, with ``bundles`` fibre bundles a ``ridge`` proud; laid under
        the skin. Round 4 drew only the medial leaf of it (1 R_j long, inside ``GLUTEAL_REACH``)."""
        G = GLUTEUS
        k = self.Rj
        nu, nv = SHEET_GRID
        a = np.linspace(0.0, 1.0, nu)[:, None, None]          # along the fibres: 0 at the origin, 1 at the insertion
        b = np.linspace(0.0, 1.0, nv)[None, :, None]          # along the origin: 0 at its top, 1 at the coccyx
        ot, ob = np.array(G["origin_top"]), np.array(G["origin_bottom"])
        it, ib = np.array(G["insertion_top"]), np.array(G["insertion_bottom"])
        xy = ((1 - a) * ((1 - b) * ot + b * ob) + a * ((1 - b) * it + b * ib))
        xy = xy * np.array([1.0, side])
        flat = (xy * k).reshape(-1, 2)
        top, N, rise = self._designed(skin, flat)
        u = 2.0 * a[..., 0] - 1.0
        v = 2.0 * b[..., 0] - 1.0
        prof = (np.clip(1.0 - v ** 2, 0.0, 1.0) ** 0.5 * np.clip(1.0 - u ** 2, 0.0, 1.0) ** 0.6
                * (1.0 + G["ridge"] * np.cos(np.pi * G["bundles"] * v)))       # thick in the middle, thin at every border
        thick = (G["thick"] * k + G["rise"] * np.maximum(rise.reshape(nu, nv), 0.0)) * prof
        t = float(self.skin_thickness(0.0))
        # the ceiling: how far each point must go down to lie a skin thickness under every level of the moulded skin,
        # spread over its neighbours (a grown, then averaged excess: the sheet goes down as a smooth surface, not point by
        # point into a jagged border), then the exact clip
        over = np.maximum(skin.depth(top) + t, 0.0).reshape(nu, nv)
        for _ in range(3):
            pad = np.pad(over, 1, mode="edge")
            grown = np.max([pad[i:i + nu, j:j + nv] for i in range(3) for j in range(3)], axis=0)
            over = np.maximum(over, 0.5 * (grown + over))
        for _ in range(4):
            pad = np.pad(over, 1, mode="edge")
            over = np.maximum(over, (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:] + 4 * over) / 8.0)
        top = top - over.reshape(-1)[:, None] * N
        top = skin.clip(top, N, t, each=True).reshape(nu, nv, 3)
        bottom = top - thick[..., None] * N.reshape(nu, nv, 3)
        return _lens_grid(top, bottom)

    def ventralis(self, side: float) -> Mesh:
        """Sacrocaudalis ventralis of one side: deep, on the ventral face of the lower sacrum, into the tail's ventral
        belly. Built from the bone alone (it is never near the skin)."""
        V = VENTRALIS
        Rs = self.Rs
        c, d, nd, w, th = self.sacral_axis(VENTRALIS_RINGS + 2)
        sel = slice(0, VENTRALIS_RINGS)                       # from the apex up the lower part of the sacrum
        cen = c[sel] - nd[sel] * (th[sel] + (V["offset"] + V["half_t"]) * Rs)[:, None]
        lat = np.array([0.0, 1.0, 0.0])
        off = side * 0.45 * w[sel]
        cen = cen + off[:, None] * lat[None, :]
        n = MUSCLE_RING
        ang = 2.0 * np.pi * np.arange(n) / n
        rings = []
        for i in range(VENTRALIS_RINGS - 1, -1, -1):          # from the sacrum toward the apex, then into the tail
            g = 0.5 + 0.5 * (1.0 - i / max(VENTRALIS_RINGS - 1, 1))
            rings.append(cen[i][None, :] + (V["half_w"] * Rs * g * np.cos(ang))[:, None] * (side * lat)[None, :]
                         + (V["half_t"] * Rs * g * np.sin(ang))[:, None] * nd[i][None, :])
        return A._loft(rings + self._into_sleeve(side, 0.40, -0.55, 0.24, 0.20))

    def body_muscles(self, skin: "RootSkin | None" = None) -> list:
        """``[(name, Mesh)]``: the muscle layer's pieces outside the caudal sleeve, in the local frame: the two
        sacrocaudalis dorsalis strands, the two gluteal sheets and the two sacrocaudalis ventralis strands, laid under
        ``skin`` (:class:`RootSkin`; a flat plane when ``None``). Vertex counts do not depend on the skin."""
        skin = skin if skin is not None else RootSkin.flat(self.R)
        out = []
        for side, tag in ((1.0, "l"), (-1.0, "r")):
            out.append((f"sacrocaudalis_dorsalis_{tag}", self.dorsalis(skin, side)))
        for side, tag in ((1.0, "l"), (-1.0, "r")):
            out.append((f"gluteus_maximus_{tag}", self.gluteal(skin, side)))
        for side, tag in ((1.0, "l"), (-1.0, "r")):
            out.append((f"sacrocaudalis_ventralis_{tag}", self.ventralis(side)))
        return out

    def muscle_whole(self, skin: "RootSkin | None" = None):
        """``(mesh, belly)``: the muscle layer as the caudal sleeve plus the body-side muscles (one mesh of closed
        pieces) and the per-vertex belly weight that colours it (the body-side muscles are all belly: 1)."""
        sleeve = self.muscle()
        parts = [sleeve] + [m for _, m in self.body_muscles(skin)]
        belly = np.concatenate([self.muscle_belly()] + [np.ones(len(m.vertices)) for m in parts[1:]])
        return Mesh.merge(parts), belly

    def muscle_belly_whole(self) -> np.ndarray:
        """The belly weights of :meth:`muscle_whole` without building the body-side geometry (its vertex counts are fixed)."""
        n_body = 2 * (DORSALIS_RINGS + len(INTO_TAIL)) * MUSCLE_RING + 2 * 2      # strands: rings + 2 poles each
        n_body += 2 * (VENTRALIS_RINGS + len(INTO_TAIL)) * MUSCLE_RING + 2 * 2
        nu, nv = SHEET_GRID
        n_body += 2 * (nu * nv + (nu - 2) * (nv - 2))
        return np.concatenate([self.muscle_belly(), np.ones(n_body)])


    def skin(self) -> Mesh:
        """The shell that is printed and rendered (closed; the buried end is inside the body)."""
        return self._loft("skin")

    def muscle(self) -> Mesh:
        """The muscle sleeve (closed): the skin's inner surface, lobed where the bellies are."""
        return self._loft("muscle")

    # bone ---------------------------------------------------------------------------------------

    def vertebra_edges(self) -> np.ndarray:
        """The ``n + 1`` boundaries of the ``n`` caudal vertebrae (Cd1 first), in the arc length of the caudal column's own
        path (:meth:`bone_path`: equal to the skin path's ``s`` from where the column has joined the tail's axis, so the
        column's start is negative: inside the body). Cd1 starts at the sacral apex (the sacrocaudal joint); the chain ends
        just before the tip."""
        n = self.p.n_vertebrae()
        frac = (np.arange(n) + 0.5) / n
        w = np.interp(frac, *VERTEBRA_LENGTHS)
        start = float(self.bone_path()[0][0])       # Cd1's rostral end, at the sacral apex
        end = self.L - 0.5 * self.tip
        edges = start + (end - start) * np.concatenate([[0.0], np.cumsum(w / w.sum())])
        return edges

    # the caudal column's own path (round 6) ------------------------------------------------------------------

    def body_vector(self, fwd, up) -> np.ndarray:
        """A vector of the body's sagittal plane (``fwd``: toward the front, ``up``: the body's up) in the local frame: the
        local axis ``z`` leaves the body pointing back, ``elevation`` degrees above the body's horizontal."""
        e = math.radians(float(self.p.elevation))
        return np.array([up * math.cos(e) + fwd * math.sin(e), 0.0, up * math.sin(e) - fwd * math.cos(e)])

    def bone_path(self):
        """``(s, centres, tangents)`` of the caudal column's centreline, from Cd1's rostral end at the sacral apex to the tip:
        Cd1 leaves the apex ``CD1_TURN`` degrees off the sacrum's apical axis, toward the tail; a cubic (Hermite) carries the
        first caudals from there into the tail's own axis, which they join ``CHAIN_JOIN`` x min(R, R_s) past the skin point
        with its tangent; from there the column IS the skin's path. ``s`` is the arc length, offset so that it equals the
        skin path's ``s`` past the join (negative at the apex). The sacrum does not move with the pose, the join does: a
        lifted tail bends at the sacrocaudal and first caudal joints."""
        bp = getattr(self, "_bp", None)
        if bp is not None:
            return bp
        Rs = self.Rs
        th = math.radians(float(sacral_angle(0.0)) + float(self.p.elevation) - CD1_TURN)
        t0 = np.array([-math.sin(th), 0.0, math.cos(th)])
        p0 = self.sacral_apex() + SACRUM_JOINT * Rs * t0
        s_join = self.chain_join()
        p1, t1 = self.centre(s_join), self.tangent(s_join)
        H, dH = _blend_path(p0, t0, p1, t1)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(H, axis=0), axis=1))])
        sel = self._sd > s_join + 1e-9
        s = np.concatenate([s_join - arc[-1] + arc, self._sd[sel]])
        self._bp = (s, np.vstack([H, self._pos[sel]]), np.vstack([dH, self._tang[sel]]))
        self._bp_basis = np.stack([np.array([math.cos(th), 0.0, math.sin(th)]), np.array([0.0, 1.0, 0.0]), t0], axis=1)
        return self._bp

    def chain_join(self) -> float:
        """Where the caudal column has joined the tail's own axis (``s``): ``CHAIN_JOIN`` x min(R, R_s) past the skin point,
        shrunk by (R / R_s)^1.5 for a tail thinner than the body's sacrum (its column must be inside its narrow skin by the
        skin point: the sacrum's apex is the body's, 0.8 R_s up, 3 R of a 0.1 tail)."""
        Rs = self.Rs
        return CHAIN_JOIN * min(self.R, Rs) * min(1.0, self.R / Rs) ** 1.5

    def bone_centre(self, s):
        sb, pos, _ = self.bone_path()
        s = np.asarray(s, dtype=np.float64)
        return np.stack([np.interp(s, sb, pos[:, c]) for c in range(3)], axis=-1)

    def bone_tangent(self, s):
        sb, _, tg = self.bone_path()
        s = np.asarray(s, dtype=np.float64)
        t = np.stack([np.interp(s, sb, tg[:, c]) for c in range(3)], axis=-1)
        return t / np.linalg.norm(t, axis=-1, keepdims=True)

    def bone_frames(self, s):
        """Frames (up = dorsal, lateral, tangent) along the caudal column at the ascending ``s``: the apex's frame (its
        dorsal normal up) carried by parallel transport."""
        s = np.asarray(s, dtype=np.float64)
        self.bone_path()
        tang = self.bone_tangent(s)
        out = np.empty((len(s), 3, 3))
        b = self._bp_basis
        out[0] = A._transport(b, b[:, 2], tang[0])
        for i in range(1, len(s)):
            out[i] = A._transport(out[i - 1], tang[i - 1], tang[i])
        return out

    def bone(self) -> Mesh:
        """The whole bone layer: the caudal chain and the sacrum it grows from (one
        mesh of closed pieces)."""
        return Mesh.merge([self.bone_chain(), self.sacrum()])

    def bone_chain(self) -> Mesh:
        """The vertebra chain from Cd1 (its rostral end at the sacrocaudal joint), as one mesh of closed pieces: what the
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
        allS = np.concatenate(samples)
        order = np.argsort(allS)
        fr_sorted = self.bone_frames(allS[order])
        fr = np.empty_like(fr_sorted)
        fr[order] = fr_sorted
        C = self.bone_centre(allS)
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
        return Mesh.merge(pieces)

    @property
    def Rs(self) -> float:
        """The sacrum's size unit: the base radius clipped to ``SACRUM_R`` (a body's sacrum, not the tail's)."""
        return float(np.clip(self.R, *SACRUM_R))

    def sacral_apex(self) -> np.ndarray:
        """The sacrum's apex (local frame): ``SACRUM["apex"]`` (forward, up; R_s) from the root's skin point in the body's
        sagittal plane, just above the top of the natal cleft and under the skin of the sacral triangle (round 4: on the
        tail's axis 1 R deep). The sacrum belongs to the pelvis: no pose moves it."""
        fwd, up = SACRUM["apex"]
        return self.body_vector(fwd * self.Rs + self.sacrum_shift, up * self.Rs)

    @property
    def sacrum_shift(self) -> float:
        if callable(self._shift):
            self._shift = float(self._shift())
        return float(self._shift)

    def sacral_axis(self, n: int = 12, upto: float = 1.0):
        """``(centres, tangents, dorsal normals, half widths, half thicknesses)`` of the sacrum at ``n`` stations from its
        apex over the first ``upto`` of its length (local frame, the rest pose). The tangent is the base -> apex
        direction: ``apex_deg`` below the BODY's horizontal (pointing back) at the apex, ``base_deg`` at the base: it
        steepens toward the apex (concave anteriorly). In the local frame an angle below the body's horizontal is that
        plus the tail axis' ``elevation`` below the tail's axis."""
        S, Rs = SACRUM, self.Rs
        L = S["length"] * Rs
        e = float(self.p.elevation)
        fine = np.linspace(0.0, L, 401)
        theta = np.radians(e + sacral_angle(fine / L))
        d = np.stack([-np.sin(theta), np.zeros_like(theta), np.cos(theta)], axis=1)
        steps = 0.5 * (d[1:] + d[:-1]) * np.diff(fine)[:, None]
        P = self.sacral_apex()[None, :] - np.vstack([np.zeros((1, 3)), np.cumsum(steps, axis=0)])
        sig = np.linspace(0.0, upto * L, n)
        c = np.stack([np.interp(sig, fine, P[:, i]) for i in range(3)], axis=1)
        th_ = np.radians(e + sacral_angle(sig / L))
        tang = np.stack([-np.sin(th_), np.zeros_like(th_), np.cos(th_)], axis=1)
        nd = np.stack([np.cos(th_), np.zeros_like(th_), np.sin(th_)], axis=1)
        # the upper sacrum (S1 - S2: the body, the alae and the auricular surfaces) is wide and deep, the lower half a narrow
        # strip (Gray97): both grow from the apex over ``upper`` of the length
        g = smoothstep((sig / L - S["upper"][0]) / (S["upper"][1] - S["upper"][0]))
        w = (S["w_apex"] + (S["w_base"] - S["w_apex"]) * g) * Rs
        h = (S["thick_apex"] + (S["thick_base"] - S["thick_apex"]) * g) * Rs
        return c, tang, nd, w, h

    def sacrum(self, n: int = 16, rings: int = 11) -> Mesh:
        """The fused sacral vertebrae as a tilted, curved wedge (:data:`SACRUM`): narrow at the apex that carries Cd1,
        wide (the alae) and thick at the base, which lies up the back and deep (anterior-superior); concave anteriorly. A
        flattened section with a dorsal median crest. Local frame, the rest pose; a closed solid."""
        c, _tang, nd, w, h = self.sacral_axis(rings)
        lat = np.array([0.0, 1.0, 0.0])
        ph = 2.0 * np.pi * np.arange(n) / n
        out = []
        cs, sn = np.cos(ph), np.sin(ph)
        sq_c = np.sign(cs) * np.abs(cs) ** 0.6                  # a flattened plate (a superellipse), not a tube
        sq_s = np.sign(sn) * np.abs(sn) ** 0.8
        crest = 1.0 + 0.30 * np.exp(-(np.angle(np.exp(1j * (ph - 0.5 * np.pi))) / 0.45) ** 2)   # the median sacral crest
        side = np.where(sq_s > 0.0, SACRUM_DORSAL, 1.0)        # thin behind, deep in front
        for i in range(rings):
            ring = (c[i][None, :] + (w[i] * sq_c)[:, None] * lat[None, :]
                    + (h[i] * sq_s * side * crest)[:, None] * nd[i][None, :])
            out.append(ring)
        return A._loft(out[::-1])

    def sacrum_dorsum(self, n: int = 13) -> np.ndarray:
        """Points of the sacrum's dorsal surface (local frame): the top of the median crest and the dorsal edges of the
        alae, at ``n`` stations: what must stay under the body's skin."""
        c, _t, nd, w, h = self.sacral_axis(n)
        lat = np.array([0.0, 1.0, 0.0])
        top = c + (1.30 * SACRUM_DORSAL * h)[:, None] * nd
        out = [top]
        for sgn in (1.0, -1.0):
            out.append(c + sgn * (0.8 * w)[:, None] * lat + (0.8 * SACRUM_DORSAL * h)[:, None] * nd)
        return np.vstack(out)

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

    def inside_skin(self, points, margin: float = 0.0) -> np.ndarray:
        """Which of the local ``points`` lie inside the tail's own skin (conservatively: within the inscribed circle of the
        section at the nearest point of the centreline, shrunk by ``margin`` of it), between the buried end and the tip:
        what the tail's shell hides in the union with the body."""
        P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        sd, pos = self._sd, self._pos
        sub = np.arange(0, len(sd), 4)
        best = np.empty(len(P), dtype=np.int64)
        for k in range(0, len(P), 512):
            d2 = ((P[k:k + 512, None, :] - pos[None, sub, :]) ** 2).sum(axis=2)
            best[k:k + 512] = sub[d2.argmin(axis=1)]
        cand = np.clip(best[:, None] + np.arange(-4, 5)[None, :], 0, len(sd) - 1)       # refine on the dense samples
        d = np.linalg.norm(P[:, None, :] - pos[cand], axis=2)
        j = cand[np.arange(len(P)), d.argmin(axis=1)]
        dist = d.min(axis=1)
        s = sd[j]
        r = self.law(s) * np.minimum(1.0, self.aspect(s)) * math.cos(math.pi / self.p.ring_verts)
        return (dist < r * (1.0 - margin)) & (s > sd[0] + 1e-9) & (s < self.L)

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


def _blend_path(p0, t0, p1, t1, n: int = 161):
    """``(points, unit tangents)`` of a path from ``p0`` leaving along ``t0`` to ``p1`` arriving along ``t1``: its direction
    turns from ``t0`` to ``t1`` by a smoothstep of the arc length (no turn at either end, the most in the middle), its length
    the one that lands it closest to ``p1``, and what it misses by is taken up by a smoothstep shift (zero slope at both
    ends, so both tangents hold). The caudal column's bend from the sacral apex into the tail (:meth:`TailModel.bone_path`)."""
    t0 = np.asarray(t0, dtype=np.float64) / np.linalg.norm(t0)
    t1 = np.asarray(t1, dtype=np.float64) / np.linalg.norm(t1)
    om = math.acos(float(np.clip(t0 @ t1, -1.0, 1.0)))
    u = np.linspace(0.0, 1.0, n)
    g = smoothstep(u)
    if om < 1e-6:
        dirs = np.broadcast_to(t0, (n, 3)).copy()
    else:
        dirs = (np.sin((1.0 - g) * om)[:, None] * t0 + np.sin(g * om)[:, None] * t1) / math.sin(om)
    steps = 0.5 * (dirs[1:] + dirs[:-1]) * (1.0 / (n - 1))
    unit = np.vstack([np.zeros((1, 3)), np.cumsum(steps, axis=0)])        # the path of length 1
    d = np.asarray(p1, dtype=np.float64) - np.asarray(p0, dtype=np.float64)
    end = unit[-1]
    length = max(float(d @ end) / max(float(end @ end), 1e-12), 0.25 * float(np.linalg.norm(d)))   # least squares
    pts = np.asarray(p0, dtype=np.float64) + length * unit
    pts = pts + smoothstep(u)[:, None] * (np.asarray(p1) - pts[-1])[None, :]
    tg = np.gradient(pts, axis=0)
    tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-12)
    tg[0], tg[-1] = t0, t1
    return pts, tg


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
# the skin the body already has where the tail roots (a height field: the first guess of a footprint point on it)


class SkinPatch:
    """The body's own skin over the root, as a height field in the tail's local frame: ``z(x, y)`` (all in base radii R,
    z out of the body, the skin point at 0) on a regular grid of ``x`` (up the back) and ``y`` (lateral),
    sampled ONCE when the tail is placed from the cage. ``RootSkin`` seeds the body-side muscles' footprints on it; the
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


def _lens_grid(top, bottom) -> Mesh:
    """A closed lens from two ``(nu, nv, 3)`` grids that share their perimeter (the ``top`` grid's): the top faces out
    along the grid's orientation, the bottom in the opposite sense; the perimeter is a sharp edge."""
    nu, nv = top.shape[:2]
    idx_top = np.arange(nu * nv).reshape(nu, nv)
    idx_bot = idx_top.copy()
    inner = np.zeros((nu, nv), dtype=bool)
    inner[1:-1, 1:-1] = True
    idx_bot[inner] = nu * nv + np.arange(int(inner.sum()))
    verts = np.vstack([top.reshape(-1, 3), bottom[inner]])
    faces = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            # split along the diagonal through an inner corner (a corner quad with one inner vertex would otherwise
            # have a triangle of three rim vertices on both sides: the same face twice)
            bd = not (inner[i, j] or inner[i + 1, j + 1])
            for idx, flip in ((idx_top, False), (idx_bot, True)):
                a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
                tri = [[a, b, d], [b, c, d]] if bd else [[a, b, c], [a, c, d]]
                faces += [t[::-1] for t in tri] if flip else tri
    mesh = Mesh(verts, np.asarray(faces, dtype=np.int64))
    if mesh.volume() < 0:
        mesh = Mesh(mesh.vertices, mesh.faces[:, [0, 2, 1]])
    return mesh


class RootSkin:
    """The body's skin around the root of the tail AS IT IS MESHED, in the tail's local frame (units of the tail
    parameters): the cage within ``ROOT_SKIN_REACH`` R_j of the root BEFORE the junction, subdivided twice (where the
    body-side muscles are laid out: :meth:`surface`), and AFTER it at subdivision levels 0, 1 and 2 (the muscle layer
    lies at least one skin thickness under every one of them: :meth:`clip`). Built once per pose
    (``PlacedTail.root_skin``) from the cage the pipeline really moulds (``junction.moulded_cage``), or from the
    shaped body a view hands over (``ecorche.layer_shells``)."""

    def __init__(self, plain, moulded, seeds, hidden=None):
        self.plain = plain                 # collar.SkinPatch: before the junction, level 2
        self.moulded = list(moulded)       # [collar.SkinPatch]: after it, levels 0, 1, 2 (the VISIBLE skin)
        self.seeds = seeds                 # SkinPatch (height field, in R): the first guess of a footprint point
        self.hidden = hidden               # local points -> bool: inside the tail's own shell (the union hides them)

    @staticmethod
    def _patch(V, q, hidden=None):
        """A closest-point patch of the quads ``q``; with ``hidden``, without the triangles the tail's shell hides (the
        skin the junction drew up inside the root is folded on itself there: its normals would call a point under the
        visible skin outside, and it is no ceiling of anything)."""
        from . import collar as CO
        tri = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
        if hidden is not None:
            keep = ~hidden(V[tri].mean(axis=1))
            if keep.sum() >= 8:
                tri = tri[keep]
        return CO.SkinPatch(V, tri, np.zeros(3), np.inf)

    @classmethod
    def flat(cls, R: float) -> "RootSkin":
        """A flat skin (the plane z = 0), for a tail without a body."""
        xs = np.linspace(-5.0, 10.0, 31) * R
        ys = np.linspace(-6.0, 6.0, 25) * R
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        V = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
        idx = np.arange(gx.size).reshape(gx.shape)
        q = np.stack([idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()], axis=1)
        pt = cls._patch(V, q)
        return cls(pt, [pt, pt, pt], SkinPatch.flat(R))

    @classmethod
    def from_cages(cls, cage_v, cage_q, moulded_v, frame, R: float, reach: float, hidden=None) -> "RootSkin":
        """From the morphed cage before (``cage_v``) and after (``moulded_v``) the junction, ``frame`` the tail's
        local -> body matrix, ``R`` the base radius, ``reach`` the patch's radius (local units) and ``hidden`` (local
        points -> bool) what the tail's own shell hides."""
        from . import junction_checks as JC
        inv = np.linalg.inv(np.asarray(frame, dtype=np.float64))
        U = np.asarray(cage_v) @ inv[:3, :3].T + inv[:3, 3]
        M = np.asarray(moulded_v) @ inv[:3, :3].T + inv[:3, 3]
        near = np.linalg.norm(U, axis=1) < reach
        near = JC.vertex_ring(cage_q, JC.vertex_ring(cage_q, near))         # two rings more: the subdivision's rim
        quads = np.asarray(cage_q)[near[np.asarray(cage_q)].all(axis=1)]
        used = np.unique(quads)
        remap = np.full(len(U), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        q0 = remap[quads]
        (U1, M1), q1, _ = JC.subdivide_patch([U[used], M[used]], q0)
        (U2, M2), q2, _ = JC.subdivide_patch([U1, M1], q1)
        seeds = SkinPatch.from_cage(cage_v, cage_q, frame, R)
        return cls(cls._patch(U2, q2), [cls._patch(M[used], q0, hidden), cls._patch(M1, q1, hidden),
                                        cls._patch(M2, q2, hidden)], seeds, hidden)

    def surface(self, xy):
        """``(points, normals)`` on the skin before the junction under the local footprint points ``xy`` (units): the
        height field's guess, then the closest point of the twice-subdivided skin."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        R = self.seeds.R
        z = self.seeds.height(xy[:, 0] / R, xy[:, 1] / R) * R
        P, N, _, _ = self.plain.closest(np.stack([xy[:, 0], xy[:, 1], z], axis=1))
        return P, N

    def depth(self, points):
        """The signed distance of ``points`` to the moulded skin, the LARGEST over levels 0, 1 and 2 (positive: outside
        at least one of them)."""
        P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return np.max([sk.closest(P)[2] for sk in self.moulded], axis=0)

    def clip(self, points, normals, t: float, each: bool = False, rounds: int = 3):
        """``points`` moved down along ``normals`` until every one lies at least ``t`` under every level of the moulded
        skin: each point by its own excess (``each``), or all of them by the largest (a strand's ring keeps its shape)."""
        P = np.array(points, dtype=np.float64).reshape(-1, 3)
        N = np.broadcast_to(np.asarray(normals, dtype=np.float64).reshape(-1, 3), P.shape)
        for _ in range(rounds):
            over = self.depth(P) + t
            if self.hidden is not None:
                over = np.where(self.hidden(P), -1.0, over)              # hidden in the tail: no ceiling there
            if each:
                if not (over > 1e-9).any():
                    break
                P = P - np.maximum(over, 0.0)[:, None] * N
            else:
                m = float(over.max())
                if m <= 1e-9:
                    break
                P = P - m * N
        return P


# --------------------------------------------------------------------------
# a tail on a body


class PlacedTail:
    """A tail model with its placement: the local -> body frame (a 4x4 with a uniform scale) and
    the parameters it was built from. Models are cached per set of pose controls (a pose drag
    asks for a new one per frame; the junction and the layers ask for the same one again).
    ``patch``: the body's own skin over the root (:class:`SkinPatch`), which the body-side muscles of the muscle layer
    lie on (a flat plane without one)."""

    def __init__(self, params: TailParams, frame, clamp=None, patch=None, body=None):
        self.params = params
        self.frame = np.asarray(frame, dtype=np.float64)
        self.clamp = clamp                 # controls -> controls inside the kind's ranges
        self._patch = patch                # SkinPatch, a function that makes one (lazy: only the layers need it) or None
        self.body = body                   # (base, verts, controls -> junction spec): the body the layers are laid in, or None
        self._models: dict = {}
        self._skins: dict = {}

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
                lift=c.get("lift", 0.0), sway=c.get("sway", 0.0)), self._patch_once, self.sacrum_shift)
        return m

    def sacrum_shift(self) -> float:
        """How far (local units, toward the body's front) the body's sacrum must move so that its dorsal surface
        (:meth:`TailModel.sacrum_dorsum`) lies at least ``SACRUM_CLEAR`` R_s under the body's own skin (the morphed cage, before
        any junction: a document with the junction off has that skin). Measured once per placed tail, on the first use of
        the bone or the body-side muscles; 0 without a body."""
        if getattr(self, "_sacrum_shift", None) is None:
            self._sacrum_shift = 0.0
            if self.body is not None:
                from . import collar as CO
                base, verts, _spec = self.body
                cv, cq = base.body_cage(verts)
                m = TailModel(self.params)
                F = self.frame
                k = self.scale
                tri = np.concatenate([cq[:, [0, 1, 2]], cq[:, [0, 2, 3]]])
                centre = F[:3, :3] @ m.sacral_apex() + F[:3, 3]
                near = np.linalg.norm(cv[tri].mean(axis=1) - centre, axis=1) < 4.0 * m.Rs * k + 1.5
                skin = CO.SkinPatch(cv, tri[near], centre, np.inf)
                fwd = m.body_vector(1.0, 0.0)
                shift = 0.0
                for _ in range(3):                   # the skin is not flat: settle in a few steps
                    P = (m.sacrum_dorsum() + shift * fwd) @ F[:3, :3].T + F[:3, 3]
                    over = float((skin.closest(P)[2] / k + SACRUM_CLEAR * m.Rs).max())
                    if over <= 1e-4:
                        break
                    shift += over
                self._sacrum_shift = min(shift, SACRUM_SHIFT_MAX * m.Rs)
        return self._sacrum_shift

    def skin(self, controls=None) -> Mesh:
        return self.model(controls).skin().transform(self.frame)

    def root_skin(self, controls=None, shaped=None) -> "RootSkin":
        """The body's skin around the root as it is meshed under ``controls`` (:class:`RootSkin`): moulded by this tail's
        own junction (``body``: ``(base, verts, spec_fn)`` handed over by the kind; the junction of the pipeline when the
        tail is the only appendage there), or ``shaped`` (the shaped body verts a view meshes: the skin of every
        appendage's junction). A flat plane without a body. Cached per pose and skin."""
        m = self.model(controls)
        if self.body is None and shaped is None:
            return RootSkin.flat(m.R)
        key = ((m.p.lift, m.p.sway, m.p.curl, m.p.droop), None if shaped is None else _digest(shaped))
        sk = self._skins.get(key)
        if sk is None:
            base, verts, spec_fn = self.body
            cage_v, cage_q = base.body_cage(verts)
            if shaped is None:
                from . import junction as Jn
                moulded = Jn.moulded_cage(base, verts, [spec_fn(controls)])[0]
            else:
                moulded = base.body_cage(shaped)[0]
            reach = ROOT_SKIN_REACH * m.Rj
            sk = RootSkin.from_cages(cage_v, cage_q, moulded, self.frame, m.R, reach, hidden=m.inside_skin)
            if len(self._skins) >= 6:
                self._skins.pop(next(iter(self._skins)))
            self._skins[key] = sk
        return sk

    def muscle_with_belly(self, controls=None, shaped=None):
        """``(mesh in the body's frame, per-vertex belly weight)`` of the whole muscle layer: the caudal
        sleeve and the body-side muscles laid under the body's skin (all muscle: weight 1)."""
        m = self.model(controls)
        mesh, belly = m.muscle_whole(self.root_skin(controls, shaped))
        return mesh.transform(self.frame), belly

    def layers(self, controls=None, shaped=None) -> dict:
        """``{"bone", "muscle", "skin"}`` meshes in the body's frame, from one model (``shaped``: see :meth:`root_skin`)."""
        m = self.model(controls)
        return {"bone": m.bone().transform(self.frame), "muscle": self.muscle_with_belly(controls, shaped)[0],
                "skin": m.skin().transform(self.frame)}


def _digest(verts) -> str:
    import hashlib
    return hashlib.blake2b(np.ascontiguousarray(verts, dtype=np.float64).tobytes(), digest_size=8).hexdigest()


def junction_spec(name: str, model: TailModel, frame, fillet: float, height: float,
                  owner: str | None = "trunk", r_max: float = R_MAX):
    """The :class:`~.junction.JunctionSpec` of a placed tail: how the body's skin moulds around
    its root. Built in the tail's local frame from the model (so it is the same on every body)
    and carried to the body by ``frame``:

    * the CENTRELINE is the skin tube from the buried end to one fillet width + 1 R beyond the skin
      point, with the inscribed radius of the (flared, elliptical) section: the skin flows into the
      tail's own trumpet;
    * one MOUND for every :class:`RootMass` (the sacral wedge up the back, the two gluteal masses
      that flank the root), sized by the SAME numbers as the body-side muscles of the muscle layer, and the
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
    if CLIMB is not None:
        mounds.append(J.Mound((0.0, 0.0, 0.0), CLIMB[1] * Rj, CLIMB[0] * Rj, direction=(0.0, 0.0, 1.0)))
    spec = J.JunctionSpec(
        name, centres, radii * J.HUG, width=fillet, base_radius=Rj, height=height, owner=owner,
        tight_gap=0.0, smooth=SMOOTH, max_move=MAX_MOVE_R * Rj, mounds=tuple(mounds), join_guard=False, stack=STACK,
        keep=tuple(J.Keep((x * Rj, 0.0, z * Rj), KEEP_RADIUS * Rj, fade=KEEP_FADE) for x, z in KEEP_BEADS))
    return spec.transformed(frame)


def mass_mound(m: RootMass, R: float, weight: float = 1.0):
    """The :class:`~.junction.Mound` of a root mass (local frame, ``R`` the base radius in local units): the bump the
    muscle adds to the skin the body already has, an ellipse of the mass's footprint, pushed OUT of the body along the
    root's own axis (local z), not along each vertex's normal: a mound along the normals leaves a furrow as deep as it was
    (the heavy body's sacral groove between the erector masses: 0.1 R of a 0.6 R wedge got through). The gluteal masses push
    along the normals, their ellipse along their own oblique axis, centred ``rise`` R over the skin point's plane."""
    rx, ry = m.half
    kw = dict(direction=(0.0, 0.0, 1.0)) if m.along_axis else {}
    if abs(rx - ry) >= 1e-9:
        kw.update(axis=m.axis, stretch=rx / ry)
    return J.Mound((m.center[0] * R, m.center[1] * R, m.rise * R), ry * R, m.height * R * weight, **kw)
