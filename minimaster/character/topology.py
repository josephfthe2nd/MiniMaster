"""Character Lab cage topology — constant index math, built once at import.

The morphable human is two closed quad cages (body + head) plus two icosphere
eyeballs. Topology NEVER changes: the 38 sliders drive vertex positions only.
This module owns every index: the torso grid, the crotch quad, the armhole
rims, limb stitching, orphan compaction, the head loft, the named facial
feature groups, and the skin-weight table. Geometry lives in generator.py.

Body layout (raw ids, before compaction):
  torso ring r (0..8) col k (0..15): id = 16*r + k   (ids 0..143)
  NECK_POLE = 144
  left leg ring j (0..7: L0..L5, F0, F1) vert m (0..7): 145 + 8j + m, pole 209
  right leg: 210 + 8j + m, pole 274
  left arm ring j (0..7: S0..S7): 275 + 8j + m, pole 339
  right arm: 340 + 8j + m, pole 404
Raw V = 405; the two armhole patch centers id(6,0)=96 and id(6,8)=104 become
unreferenced and are compacted away (a valence-0 vertex would NaN the
subdivider) -> final V = 403, F = 401, Euler characteristic 2.
"""

from __future__ import annotations

import numpy as np

from ..core.subdiv import loft_rings

N_COLS = 16
RAW_V = 405
NECK_POLE = 144
LLEG0, RLEG0, LARM0, RARM0 = 145, 210, 275, 340
LLEG_POLE, RLEG_POLE, LARM_POLE, RARM_POLE = 209, 274, 339, 404


def tid(r: int, k: int) -> int:
    return 16 * r + (k % 16)


def lleg(j, m):
    return LLEG0 + 8 * j + (m % 8)


def rleg(j, m):
    return RLEG0 + 8 * j + (m % 8)


def larm(j, m):
    return LARM0 + 8 * j + (m % 8)


def rarm(j, m):
    return RARM0 + 8 * j + (m % 8)


# band quads removed to open the armholes: (r, k) spans rings r->r+1, cols k->k+1
HOLE_QUADS = {(5, 15), (5, 0), (6, 15), (6, 0),   # left
              (5, 7), (5, 8), (6, 7), (6, 8)}     # right

# leg top loops: traversal order = the order the bridge quads consume them
A_LEFT = [tid(0, c) for c in (3, 2, 1, 0, 15, 14, 13, 12)]
A_RIGHT = [tid(0, c) for c in (11, 10, 9, 8, 7, 6, 5, 4)]
# armhole rims (derived by directed-edge cancellation in the design spec)
B_LEFT = [tid(5, 15), tid(5, 0), tid(5, 1), tid(6, 1),
          tid(7, 1), tid(7, 0), tid(7, 15), tid(6, 15)]
B_RIGHT = [tid(5, 7), tid(5, 8), tid(5, 9), tid(6, 9),
           tid(7, 9), tid(7, 8), tid(7, 7), tid(6, 7)]


def _bridge(rim, ring_fn):
    """Rim loop -> first limb ring: [Rim_m, Rim_m+1, Inner_m+1, Inner_m]."""
    return [[rim[m], rim[(m + 1) % 8], ring_fn(0, (m + 1) % 8), ring_fn(0, m)]
            for m in range(8)]


def _limb(ring_fn, nrings, pole):
    """Ring bands marching away from the body + a quad-fan tip cap."""
    qs = []
    for j in range(nrings - 1):
        for m in range(8):
            mn = (m + 1) % 8
            qs.append([ring_fn(j, m), ring_fn(j, mn),
                       ring_fn(j + 1, mn), ring_fn(j + 1, m)])
    last = nrings - 1
    for m in range(0, 8, 2):
        qs.append([pole, ring_fn(last, m), ring_fn(last, m + 1),
                   ring_fn(last, (m + 2) % 8)])
    return qs


def build_body_topology() -> tuple[np.ndarray, np.ndarray]:
    """Return (quads (401, 4) in COMPACT ids, keep (403,) raw ids kept).

    Geometry builders author a raw (405, 3) vertex array; the caller applies
    ``verts[keep]`` to align with the compact quads.
    """
    quads: list[list[int]] = []
    for r in range(8):
        for k in range(16):
            if (r, k) in HOLE_QUADS:
                continue
            quads.append([tid(r, k), tid(r, k + 1), tid(r + 1, k + 1), tid(r + 1, k)])
    for k in range(0, 16, 2):  # neck cap fan
        quads.append([NECK_POLE, tid(8, k), tid(8, k + 1), tid(8, (k + 2) % 16)])
    quads.append([tid(0, 12), tid(0, 11), tid(0, 4), tid(0, 3)])  # crotch

    quads += _bridge(A_LEFT, lleg) + _limb(lleg, 8, LLEG_POLE)
    quads += _bridge(A_RIGHT, rleg) + _limb(rleg, 8, RLEG_POLE)
    quads += _bridge(B_LEFT, larm) + _limb(larm, 8, LARM_POLE)
    quads += _bridge(B_RIGHT, rarm) + _limb(rarm, 8, RARM_POLE)

    q = np.asarray(quads, dtype=np.int64)
    keep = np.unique(q)
    remap = -np.ones(RAW_V, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    return remap[q], keep


BODY_QUADS, BODY_KEEP = build_body_topology()

# --------------------------------------------------------------------------
# head topology: literally loft_rings on 11 rings x 16 cols (indices only)

_dummy = [np.stack([np.cos(np.linspace(0, 2 * np.pi, 16, endpoint=False)),
                    np.sin(np.linspace(0, 2 * np.pi, 16, endpoint=False)),
                    np.full(16, float(r))], axis=1) for r in range(11)]
_hv, HEAD_QUADS = loft_rings(_dummy, close_bottom=True, close_top=True)
HEAD_V = len(_hv)  # 178: 11*16 rings + bottom pole 176 + top pole 177
HEAD_BOTTOM_POLE, HEAD_TOP_POLE = 176, 177
del _dummy, _hv


def hid(r: int, c: int) -> int:
    return 16 * r + (c % 16)


def _mirror_col(c: int) -> int:
    return (16 - c) % 16


def _mirrored(entries):
    """[(r, c, w)] -> adds the mirrored-column entries (c0/c8 not doubled)."""
    out = list(entries)
    for r, c, w in entries:
        mc = _mirror_col(c)
        if mc != c:
            out.append((r, mc, w))
    return out


def _group(entries) -> tuple[np.ndarray, np.ndarray]:
    idx = np.array([hid(r, c) for r, c, _ in entries], dtype=np.int64)
    w = np.array([w for _, _, w in entries], dtype=np.float64)
    return idx, w


# named facial vertex groups: (head vert ids, per-vert weight)
GROUPS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "brow": _group(_mirrored([(8, 1, 1.0), (8, 2, 1.0), (8, 3, 1.0),
                              (8, 0, 0.5), (8, 4, 0.5)])),
    "socket_l": _group([(7, 2, 1.0), (7, 3, 0.7), (7, 1, 0.5),
                        (6, 2, 0.35), (8, 2, 0.35)]),
    "socket_r": _group([(7, 14, 1.0), (7, 13, 0.7), (7, 15, 0.5),
                        (6, 14, 0.35), (8, 14, 0.35)]),
    "bridge": _group([(7, 0, 1.0), (6, 0, 0.6), (8, 0, 0.6)]),
    "nose_tip": _group([(5, 0, 1.0)]),
    "wings": _group([(5, 1, 1.0), (5, 15, 1.0), (6, 1, 0.4), (6, 15, 0.4)]),
    "cheekbone": _group([(6, 3, 1.0), (6, 13, 1.0), (6, 4, 0.6), (6, 12, 0.6),
                         (7, 3, 0.3), (7, 13, 0.3)]),
    "cheek_flesh": _group(_mirrored([(4, 3, 1.0), (5, 3, 1.0), (4, 2, 0.6),
                                     (5, 2, 0.6), (4, 4, 0.6), (5, 4, 0.6)])),
    "jaw": _group(_mirrored([(1, 3, 1.0), (1, 4, 1.0), (1, 5, 1.0),
                             (1, 2, 0.5), (2, 4, 0.4)])),
    "chin": _group([(1, 0, 1.0), (1, 1, 0.7), (1, 15, 0.7), (0, 0, 0.3)]),
    "lip_up": _group([(4, 0, 1.0), (4, 1, 0.8), (4, 15, 0.8)]),
    "lip_low": _group([(2, 0, 1.0), (2, 1, 0.8), (2, 15, 0.8)]),
    "mouth_crn": _group([(3, 2, 1.0), (3, 14, 1.0), (3, 1, 0.4), (3, 15, 0.4),
                         (2, 2, 0.3), (4, 2, 0.3), (2, 14, 0.3), (4, 14, 0.3)]),
    "ear": _group([(6, 5, 1.0), (7, 5, 1.0), (6, 11, 1.0), (7, 11, 1.0)]),
}

FACE_FRONT_COLS = (0, 1, 2, 3, 4, 12, 13, 14, 15)
SKULL_BACK_COLS = (6, 7, 8, 9, 10)

# neutral feature offsets baked before sliders (Hh units); the default face.
# Deltas are authored ~2x their target read: Catmull-Clark averages a lone
# feature column against its 13-degree neighbours, so subtle offsets vanish.
NEUTRAL_OFFSETS: list[tuple[int, int, tuple[float, float, float]]] = [
    (6, 0, (0.0, -0.130, 0.0)),    # nose dorsum
    (5, 0, (0.0, -0.210, 0.015)),  # nose tip
    (5, 1, (0.010, -0.100, 0.0)), (5, 15, (-0.010, -0.100, 0.0)),   # wings
    (7, 0, (0.0, -0.070, 0.0)),    # mid-bridge
    (8, 0, (0.0, -0.040, 0.0)),    # glabella
    (7, 2, (0.0, 0.055, 0.0)), (7, 14, (0.0, 0.055, 0.0)),   # sockets recessed
    (7, 3, (0.0, 0.042, 0.0)), (7, 13, (0.0, 0.042, 0.0)),
    (7, 1, (0.0, 0.028, 0.0)), (7, 15, (0.0, 0.028, 0.0)),   # inner canthus
    (8, 1, (0.0, -0.048, 0.0)), (8, 2, (0.0, -0.048, 0.0)),
    (8, 3, (0.0, -0.048, 0.0)),                               # brow ridge
    (8, 15, (0.0, -0.048, 0.0)), (8, 14, (0.0, -0.048, 0.0)),
    (8, 13, (0.0, -0.048, 0.0)),
    (4, 0, (0.0, -0.055, 0.0)),                               # upper lip
    (4, 1, (0.0, -0.042, 0.0)), (4, 15, (0.0, -0.042, 0.0)),
    (2, 0, (0.0, -0.058, 0.0)),                               # lower lip
    (2, 1, (0.0, -0.045, 0.0)), (2, 15, (0.0, -0.045, 0.0)),
    (3, 0, (0.0, -0.020, 0.0)),                               # mouth line
    (3, 1, (0.0, -0.020, 0.0)), (3, 15, (0.0, -0.020, 0.0)),  # (recessed)
    (3, 2, (0.0, -0.010, 0.0)), (3, 14, (0.0, -0.010, 0.0)),  # corners
    (1, 0, (0.0, -0.070, 0.015)),                             # chin ball
    (1, 1, (0.0, -0.050, 0.0)), (1, 15, (0.0, -0.050, 0.0)),
    (6, 3, (0.022, -0.028, 0.0)), (6, 13, (-0.022, -0.028, 0.0)),  # malar break
    (6, 5, (0.055, 0.020, 0.0)), (7, 5, (0.055, 0.020, 0.0)),      # ears
    (6, 11, (-0.055, 0.020, 0.0)), (7, 11, (-0.055, 0.020, 0.0)),
]


# --------------------------------------------------------------------------
# skin-weight table: raw body vert -> {bone_key: weight}; bone key = child
# joint name (binding to bone J applies J's PARENT joint rotation)

BONES = ["spine", "chest", "neck", "head_top",
         "elbow_l", "wrist_l", "hand_l", "elbow_r", "wrist_r", "hand_r",
         "knee_l", "ankle_l", "toe_l", "knee_r", "ankle_r", "toe_r"]
_BCOL = {b: i for i, b in enumerate(BONES)}


def build_weights() -> np.ndarray:
    """(403, len(BONES)) compact skin-weight matrix, rows sum to 1."""
    W = np.zeros((RAW_V, len(BONES)))

    def setw(raw_id, **bw):
        W[raw_id] = 0.0
        for bone, w in bw.items():
            W[raw_id, _BCOL[bone]] = w

    rim_l, rim_r = set(B_LEFT), set(B_RIGHT)
    for r in range(9):
        for k in range(16):
            i = tid(r, k)
            if r in (0, 1):
                setw(i, spine=1.0)
            elif r == 2:
                setw(i, spine=0.5, chest=0.5)
            elif r in (3, 4):
                setw(i, chest=1.0)
            elif r in (5, 6, 7):
                if i in rim_l:
                    setw(i, neck=0.7, elbow_l=0.3)
                elif i in rim_r:
                    setw(i, neck=0.7, elbow_r=0.3)
                else:
                    setw(i, neck=1.0)
            else:  # r == 8
                setw(i, neck=0.5, head_top=0.5)
    setw(NECK_POLE, head_top=1.0)

    for side, ring_fn, pole in (("l", larm, LARM_POLE), ("r", rarm, RARM_POLE)):
        el, wr, hd = f"elbow_{side}", f"wrist_{side}", f"hand_{side}"
        for m in range(8):
            setw(ring_fn(0, m), **{"neck": 0.4, el: 0.6})
            setw(ring_fn(1, m), **{el: 1.0})
            setw(ring_fn(2, m), **{el: 1.0})
            setw(ring_fn(3, m), **{el: 0.5, wr: 0.5})
            setw(ring_fn(4, m), **{wr: 1.0})
            setw(ring_fn(5, m), **{wr: 0.5, hd: 0.5})
            setw(ring_fn(6, m), **{hd: 1.0})
            setw(ring_fn(7, m), **{hd: 1.0})
        setw(pole, **{hd: 1.0})

    for side, ring_fn, pole in (("l", lleg, LLEG_POLE), ("r", rleg, RLEG_POLE)):
        kn, an, to = f"knee_{side}", f"ankle_{side}", f"toe_{side}"
        for m in range(8):
            setw(ring_fn(0, m), **{kn: 0.7, "spine": 0.3})
            setw(ring_fn(1, m), **{kn: 1.0})
            setw(ring_fn(2, m), **{kn: 0.5, an: 0.5})
            setw(ring_fn(3, m), **{an: 1.0})
            setw(ring_fn(4, m), **{an: 1.0})
            setw(ring_fn(5, m), **{an: 0.5, to: 0.5})
            setw(ring_fn(6, m), **{to: 1.0})
            setw(ring_fn(7, m), **{to: 1.0})
        setw(pole, **{to: 1.0})

    Wc = W[BODY_KEEP]
    rs = Wc.sum(axis=1, keepdims=True)
    if not np.allclose(rs, 1.0):
        raise AssertionError("skin-weight rows must sum to 1")
    return Wc


BODY_WEIGHTS = build_weights()
