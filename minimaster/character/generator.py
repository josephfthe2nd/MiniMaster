"""Character Lab generator: 38 sliders -> cage vertices -> subdivided body.

Pure and deterministic: identical params + pose give a byte-identical mesh.
Pipeline (fixed order, see docs/character_lab_spec.md):

    resolve drives -> joint layout -> ring tables -> place cage verts
    -> additive feature atoms -> LBS skin (cage) -> Catmull-Clark -> shells

Everything is authored in fractions of H (total height) — the ``height``
slider is a final uniform mm scale, so it is orthogonal to every proportion.
The head is authored in Hh units (head heights) about a chin-base origin.
"""

from __future__ import annotations

import numpy as np

from ..core import primitives
from ..core.armature import Armature
from ..core.mesh import Mesh
from ..core.subdiv import quads_to_mesh, subdivide_quads
from . import topology as T

# --------------------------------------------------------------------------
# parameter registry: name -> (group, lo, hi, default)

PARAMS: dict[str, tuple[str, float, float, float]] = {
    "gender": ("identity", 0.0, 1.0, 0.5),
    "height": ("identity", 20.0, 45.0, 30.0),
    "weight": ("composite", -1.0, 1.0, 0.0),
    "muscle": ("composite", -1.0, 1.0, 0.0),
    "shoulders": ("body", 0.8, 1.3, 1.0),
    "hips": ("body", 0.8, 1.3, 1.0),
    "bust": ("body", 0.0, 1.0, 0.15),
    "waist": ("body", 0.75, 1.3, 1.0),
    "neck_len": ("body", 0.7, 1.4, 1.0),
    "neck_thick": ("body", 0.75, 1.35, 1.0),
    "arm_len": ("body", 0.8, 1.25, 1.0),
    "leg_len": ("body", 0.8, 1.25, 1.0),
    "torso_len": ("body", 0.85, 1.2, 1.0),
    "hand_size": ("body", 0.7, 1.4, 1.0),
    "foot_size": ("body", 0.7, 1.4, 1.0),
    "head_size": ("body", 0.85, 1.25, 1.0),
    "brow_height": ("face", -1.0, 1.0, 0.0),
    "brow_depth": ("face", -1.0, 1.0, 0.0),
    "eye_size": ("face", -1.0, 1.0, 0.0),
    "eye_spacing": ("face", -1.0, 1.0, 0.0),
    "eye_height": ("face", -1.0, 1.0, 0.0),
    "eye_depth": ("face", -1.0, 1.0, 0.0),
    "nose_bridge_height": ("face", -1.0, 1.0, 0.0),
    "nose_bridge_depth": ("face", -1.0, 1.0, 0.0),
    "nose_length": ("face", -1.0, 1.0, 0.0),
    "nose_width": ("face", -1.0, 1.0, 0.0),
    "nose_tip": ("face", -1.0, 1.0, 0.0),
    "cheek_height": ("face", -1.0, 1.0, 0.0),
    "cheek_depth": ("face", -1.0, 1.0, 0.0),
    "cheek_full": ("face", -1.0, 1.0, 0.0),
    "jaw_width": ("face", -1.0, 1.0, 0.0),
    "chin_length": ("face", -1.0, 1.0, 0.0),
    "chin_width": ("face", -1.0, 1.0, 0.0),
    "mouth_width": ("face", -1.0, 1.0, 0.0),
    "lip_full": ("face", -1.0, 1.0, 0.0),
    "ear_size": ("face", -1.0, 1.0, 0.0),
    "face_width": ("face", -1.0, 1.0, 0.0),
    "face_length": ("face", -1.0, 1.0, 0.0),
}


def default_params() -> dict[str, float]:
    return {k: v[3] for k, v in PARAMS.items()}


def _headroom_clip(x, lo, hi):
    """Hard clip with 25% range headroom — composites may push past the
    slider range a little, then saturate (Fallout-style, monotone)."""
    span = hi - lo
    return float(np.clip(x, lo - 0.25 * span, hi + 0.25 * span))


def resolve(params: dict[str, float]) -> dict[str, float]:
    """User sliders -> effective drives (composite couplings applied)."""
    p = default_params()
    unknown = set(params) - set(p)
    if unknown:
        raise KeyError(f"unknown character params: {sorted(unknown)}")
    p.update(params)
    for k, (_, lo, hi, _) in PARAMS.items():
        p[k] = float(np.clip(p[k], lo, hi))

    g = p["gender"]
    gm = 2.0 * g - 1.0
    w, m = p["weight"], p["muscle"]
    d = dict(p)
    d["gm"] = gm
    d["sh_eff"] = _headroom_clip(p["shoulders"] * (1 - 0.14 * gm), *PARAMS["shoulders"][1:3])
    d["hips_eff"] = _headroom_clip(p["hips"] * (1 + 0.13 * gm), *PARAMS["hips"][1:3])
    d["waist_eff"] = _headroom_clip(p["waist"] * (1 - 0.10 * gm), *PARAMS["waist"][1:3])
    d["neck_thick_eff"] = _headroom_clip(p["neck_thick"] * (1 - 0.08 * gm),
                                         *PARAMS["neck_thick"][1:3])
    d["muscle_eff"] = _headroom_clip(m - 0.25 * gm, -1.0, 1.0)
    d["bust_drive"] = float(np.clip(p["bust"] + 0.8 * max(gm, 0.0), 0.0, 1.25))
    d["glute_drive"] = float(np.clip(0.5 * d["muscle_eff"] + 0.5 * w
                                     + 0.25 * max(gm, 0.0), -1.25, 1.25))
    # gender-coupled face drives
    face = {"jaw_width": -0.45, "brow_depth": -0.50, "chin_width": -0.30,
            "lip_full": 0.35, "cheek_height": 0.20, "face_width": -0.15}
    for name, k in face.items():
        d[name + "_eff"] = _headroom_clip(p[name] + k * gm, -1.0, 1.0)
    d["cheek_full_eff"] = _headroom_clip(p["cheek_full"] + 0.25 * gm + 0.45 * w,
                                         -1.0, 1.0)
    return d


# --------------------------------------------------------------------------
# joints

CHIN_BASE = 0.80  # neck-top / chin z at neutral, in H


def _chin_z(d):
    return (CHIN_BASE - 0.485) * 1.0 + 0.485 + 0.04 * (d["neck_len"] - 1.0) \
        if False else CHIN_BASE + 0.04 * (d["neck_len"] - 1.0)


def _torso_z(d, z):
    """torso_len scales every torso/neck z about the pelvis floor 0.485."""
    return 0.485 + (z - 0.485) * d["torso_len"]


HEAD_UNIT = 0.138  # head height as a fraction of H (~1/7.3, heroic-adult canon)


def joint_layout(d) -> Armature:
    """The canon humanoid armature, positions driven by the sliders."""
    Hh = HEAD_UNIT * d["head_size"]
    chin = _torso_z(d, _chin_z(d))
    arm = Armature()
    arm.add_joint("pelvis", [0, 0, _torso_z(d, 0.515)])
    arm.add_joint("spine", [0, 0, _torso_z(d, 0.62)], parent="pelvis")
    arm.add_joint("chest", [0, 0, _torso_z(d, 0.72)], parent="spine")
    arm.add_joint("neck", [0, 0, _torso_z(d, 0.80)], parent="chest")
    arm.add_joint("head_top", [0, 0.004, chin + Hh], parent="neck")
    sh_x = 0.094 * d["sh_eff"]
    sh_z = _torso_z(d, 0.78)
    al = d["arm_len"]
    for s, sx in (("l", 1.0), ("r", -1.0)):
        arm.add_joint(f"shoulder_{s}", [sx * sh_x, 0, sh_z], parent="chest")
        arm.add_joint(f"elbow_{s}", [sx * (sh_x + 0.006), 0, sh_z - 0.16 * al],
                      parent=f"shoulder_{s}")
        arm.add_joint(f"wrist_{s}", [sx * (sh_x + 0.012), 0, sh_z - 0.28 * al],
                      parent=f"elbow_{s}")
        arm.add_joint(f"hand_{s}", [sx * (sh_x + 0.012), 0, sh_z - 0.325 * al],
                      parent=f"wrist_{s}")
    hip_x = 0.045 * (1 + 0.10 * (d["hips_eff"] - 1.0))
    ll = d["leg_len"]

    def lz(z):  # leg z scales about the ankle
        return 0.055 + (z - 0.055) * ll

    for s, sx in (("l", 1.0), ("r", -1.0)):
        arm.add_joint(f"hip_{s}", [sx * hip_x, 0, lz(0.50)], parent="pelvis")
        arm.add_joint(f"knee_{s}", [sx * 0.043, 0, lz(0.26)], parent=f"hip_{s}")
        arm.add_joint(f"ankle_{s}", [sx * 0.040, 0, 0.055], parent=f"knee_{s}")
        arm.add_joint(f"toe_{s}", [sx * 0.040, -0.085 * d["foot_size"], 0.02],
                      parent=f"ankle_{s}")
    return arm


# --------------------------------------------------------------------------
# body cage geometry (raw 405 verts; compacted by the caller)

_PHI = np.radians((np.arange(16) + 0.5) * 22.5)  # col azimuths; sin>0 = back
_FRONT = np.sin(_PHI) < 0.0  # cols 8..15


def _torso_ring(v, r, hw, hd, z, col_scale=None, col_dy=None, col_dz=None):
    x = hw * np.cos(_PHI)
    y = hd * np.sin(_PHI)
    if col_scale is not None:
        x = x * col_scale
        y = y * col_scale
    if col_dy is not None:
        y = y + col_dy
    zz = np.full(16, z)
    if col_dz is not None:
        zz = zz + col_dz
    v[16 * r:16 * r + 16] = np.stack([x, y, zz], axis=1)


def body_verts_raw(d, arm: Armature) -> np.ndarray:
    """Raw (405, 3) body cage vertex positions in H fractions."""
    v = np.zeros((T.RAW_V, 3))
    gm, w = d["gm"], d["weight"]
    m = d["muscle_eff"]

    # ---- torso rings R0..R8
    ws = 1.0 + 0.30 * w
    wf = np.where(_FRONT, ws, 1.0 + 0.18 * w)  # front full, back 0.6x
    belly = np.where(_FRONT, -0.020 * max(w, 0.0), 0.0)
    lats = 1.0 + 0.10 * m

    _torso_ring(v, 0, 0.098, 0.064, 0.485)
    glute = np.zeros(16)
    gd = 0.010 + 0.014 * d["glute_drive"]
    glute[[3, 4]] = gd
    glute[[2, 5]] = 0.5 * gd
    _torso_ring(v, 1, 0.104 * d["hips_eff"] * (1 + 0.18 * w), 0.072,
                _torso_z(d, 0.530), col_dy=glute)
    _torso_ring(v, 2, 0.078 * d["waist_eff"] * (1 - 0.04 * max(m, 0.0)),
                0.056 * d["waist_eff"],
                _torso_z(d, 0.620 + 0.015 * max(gm, 0.0)),
                col_scale=wf, col_dy=belly)
    _torso_ring(v, 3, 0.090 * lats, 0.064, _torso_z(d, 0.660),
                col_scale=1.0 + (wf - 1.0) * 0.8, col_dy=belly * 0.6)
    pec = np.zeros(16)
    pec[[10, 11, 12, 13]] = -0.012 * m
    bust_dy = np.zeros(16)
    bust_dz = np.zeros(16)
    bd = d["bust_drive"]
    bust_dy[[11, 12]] = -0.032 * bd
    bust_dz[[11, 12]] = -0.006 * bd
    bust_dy[[10, 13]] = -0.016 * bd
    bust_dz[[10, 13]] = -0.003 * bd
    _torso_ring(v, 4, 0.100 * lats * (1 + 0.3 * (d["sh_eff"] - 1.0)) * (1 + 0.10 * w),
                0.068, _torso_z(d, 0.720), col_dy=pec + bust_dy, col_dz=bust_dz)
    _torso_ring(v, 5, 0.110 * d["sh_eff"], 0.066, _torso_z(d, 0.762))
    _torso_ring(v, 6, 0.118 * d["sh_eff"], 0.060, _torso_z(d, 0.790))
    # R7 much narrower than R6 so the R6->R8 span subdivides into a trapezius
    # SLOPE, not a coat-hanger shelf
    trap_dy = np.where(~_FRONT, 0.008 * m, 0.0)
    _torso_ring(v, 7, 0.078 * d["sh_eff"], 0.052,
                _torso_z(d, 0.818 + 0.012 * m), col_dy=trap_dy)
    neck_r = 0.042 * d["neck_thick_eff"] * (1 + 0.12 * w)
    _torso_ring(v, 8, neck_r, neck_r, _torso_z(d, CHIN_BASE + 0.045 * (d["neck_len"] - 1.0)))
    v[T.NECK_POLE] = [0.0, 0.004, _torso_z(d, CHIN_BASE + 0.07 * (d["neck_len"] - 1.0))]

    # ---- legs
    ll = d["leg_len"]
    fs = d["foot_size"]

    def lz(z):
        return 0.055 + (z - 0.055) * ll

    quad_m = (1 + 0.22 * m) * (1 + 0.20 * w)
    leg_rings = [  # (cz in leg-scaled z, cx, cy, r1, r2)
        (lz(0.455), 0.052, 0.004, 0.052 * quad_m, 0.058 * quad_m),   # L0
        (lz(0.370), 0.049, 0.0, 0.046 * quad_m, 0.050 * quad_m),     # L1
        (lz(0.260), 0.043, 0.0, 0.033, 0.035),                       # L2 knee
        (lz(0.205), 0.042, 0.006, 0.033 * (1 + 0.20 * m) * (1 + 0.10 * w),
         0.042 * (1 + 0.20 * m) * (1 + 0.10 * w)),                   # L3 calf
        (lz(0.120), 0.041, 0.0, 0.024, 0.028),                       # L4
        (0.055, 0.040, 0.0, 0.020 * (1 + 0.03 * w), 0.023 * (1 + 0.03 * w)),  # L5
    ]
    delta = np.radians(-22.5 + 45.0 * np.arange(8))
    tilt = np.radians(50.0)
    for ring_fn, pole, out in ((T.lleg, T.LLEG_POLE, 1.0), (T.rleg, T.RLEG_POLE, -1.0)):
        for j, (cz, cx, cy, r1, r2) in enumerate(leg_rings):
            px = out * cx + r1 * np.sin(delta) * out
            py = cy + r2 * np.cos(delta)
            for mm in range(8):
                v[ring_fn(j, mm)] = [px[mm], py[mm], cz]
        # calf belly sits high and back: push L3 back verts rearward
        for mm in range(8):
            i = ring_fn(3, mm)
            if v[i][1] > 0.006:
                v[i][1] += 0.012 * m
        # F0 instep: tilted ring, heel bias on the two rearmost verts
        c = np.array([out * 0.040, -0.030 * fs, 0.036])
        vy = np.array([0.0, np.cos(tilt), np.sin(tilt)])
        for mm in range(8):
            p = c + 0.028 * fs * np.sin(delta[mm]) * np.array([out, 0, 0]) \
                + 0.028 * fs * np.cos(delta[mm]) * vy
            v[ring_fn(6, mm)] = p
        backs = np.argsort([v[ring_fn(6, mm)][1] for mm in range(8)])[-2:]
        for mm in backs:
            v[ring_fn(6, int(mm))][1] += 0.022
        # F1 ball of foot: vertical ring
        c = np.array([out * 0.040, -0.112 * fs, 0.028])
        for mm in range(8):
            p = c + 0.038 * fs * np.sin(delta[mm]) * np.array([out, 0, 0]) \
                + 0.026 * fs * np.cos(delta[mm]) * np.array([0.0, 0.0, 1.0])
            v[ring_fn(7, mm)] = p
        v[pole] = [out * 0.044, -0.155 * fs, 0.022]

    # ---- arms
    sh_x = 0.094 * d["sh_eff"]
    sh_z = _torso_z(d, 0.78)
    al = d["arm_len"]
    hs = d["hand_size"]
    g = d["gender"]
    delt_r = 0.036 * (1 + 0.30 * m) * (1.05 + (0.92 - 1.05) * g)
    arm_rings = [  # (t (unit-ish), c offset from (sh_x,0,sh_z-ish), r1, r2, zdrop)
        ((0.97, 0.0, 0.24), 0.004, 0.026, 0.027, 0.006),              # S0
        ((0.60, 0.0, -0.80), 0.022, delt_r, delt_r, 0.026),           # S1
        ((0.0, 0.0, -1.0), 0.028, 0.028 * (1 + 0.12 * m) * (1 + 0.15 * w),
         0.031 * (1 + 0.12 * m) * (1 + 0.15 * w), 0.082),             # S2 z 0.700
        ((0.0, 0.0, -1.0), 0.031, 0.023, 0.023, 0.162),               # S3 elbow
        ((0.0, 0.0, -1.0), 0.034, 0.026 * (1 + 0.18 * m) * (1 + 0.08 * w),
         0.026 * (1 + 0.18 * m) * (1 + 0.08 * w), 0.197),             # S4
        ((0.0, 0.0, -1.0), 0.038, 0.016 * (1 + 0.03 * w),
         0.019 * (1 + 0.03 * w), 0.282),                              # S5 wrist
        ((0.0, 0.0, -1.0), 0.040, 0.014 * hs, 0.030 * hs, 0.310),     # S6 palm
        ((0.0, 0.0, -1.0), 0.040, 0.012 * hs, 0.026 * hs, 0.334),     # S7 knuckle
    ]
    aa = np.radians(45.0 * (np.arange(8) - 1.0))
    for ring_fn, pole, sx in ((T.larm, T.LARM_POLE, 1.0), (T.rarm, T.RARM_POLE, -1.0)):
        e2 = np.array([0.0, sx, 0.0])
        for j, (t, cdx, r1, r2, zdrop) in enumerate(arm_rings):
            tv = np.array([sx * t[0], t[1], t[2]])
            tv /= np.linalg.norm(tv)
            e1 = np.cross(e2, tv)
            e1 /= np.linalg.norm(e1)
            cz = sh_z + 0.002 - (zdrop * al if j >= 2 else zdrop)
            c = np.array([sx * (sh_x + cdx), 0.0, cz])
            for mm in range(8):
                v[ring_fn(j, mm)] = c + r1 * np.cos(aa[mm]) * e1 + r2 * np.sin(aa[mm]) * e2
        # biceps front / triceps back on S2 (+S3 back), geometric predicates
        for j, front_d, back_d in ((2, -0.012 * m, 0.010 * m), (3, 0.0, 0.010 * m)):
            ys = np.array([v[ring_fn(j, mm)][1] for mm in range(8)])
            cy = ys.mean()
            for mm in range(8):
                i = ring_fn(j, mm)
                if v[i][1] < cy - 1e-6 and front_d:
                    v[i][1] += front_d
                elif v[i][1] > cy + 1e-6 and back_d:
                    v[i][1] += back_d
        # thumb bump: the two front-most palm verts push inward + forward
        ys = np.array([v[ring_fn(6, mm)][1] for mm in range(8)])
        for mm in np.argsort(ys)[:2]:
            v[ring_fn(6, int(mm))] += [-sx * 0.006 * hs, -0.010 * hs, 0.0]
        v[pole] = [sx * (sh_x + 0.040), 0.0, sh_z + 0.002 - (0.334 + 0.020 * hs) * al]

    return v


# --------------------------------------------------------------------------
# head geometry (local Hh units about the chin-base origin)

_PSI = np.radians(np.array([0.0, 13.0, 28.0, 45.0, 70.0, 100.0, 135.0, 165.0,
                            180.0, -165.0, -135.0, -100.0, -70.0, -45.0, -28.0,
                            -13.0]))
_HFRONT = np.abs(np.degrees(_PSI)) < 90.0

# ring table rows r0..r10: (z, a half-width, f front depth, b back depth)
_HEAD_RINGS = np.array([
    (0.03, 0.16, 0.14, 0.18),
    (0.11, 0.30, 0.32, 0.35),
    (0.20, 0.27, 0.36, 0.38),
    (0.25, 0.26, 0.35, 0.39),
    (0.30, 0.26, 0.37, 0.39),
    (0.37, 0.28, 0.36, 0.40),
    (0.46, 0.32, 0.37, 0.41),
    (0.53, 0.33, 0.35, 0.41),
    (0.63, 0.33, 0.37, 0.40),
    (0.78, 0.29, 0.33, 0.37),
    (0.93, 0.19, 0.20, 0.25),
])


def head_verts_local(d) -> np.ndarray:
    """(178, 3) head shell in local Hh units (before Hh scale + placement)."""
    fw = d["face_width_eff"]
    fl = d["face_length"]
    v = np.zeros((T.HEAD_V, 3))
    for r in range(11):
        z, a, f, b = _HEAD_RINGS[r]
        if 1 <= r <= 9:
            z = 0.53 + (z - 0.53) * (1 + 0.10 * fl)
        else:
            z = z + (z - 0.53) * 0.10 * fl * 0.05
        wfac = np.where(_HFRONT, 1 + 0.10 * fw, 1 + 0.05 * fw) if 1 <= r <= 8 else 1.0
        x = a * np.sin(_PSI) * wfac
        y = np.where(_HFRONT, -f * np.cos(_PSI), b * np.abs(np.cos(_PSI)))
        v[16 * r:16 * r + 16] = np.stack([x, y, np.full(16, z)], axis=1)
    v[16 * 0:16 * 0 + 16, 1] += 0.05  # r0 tucks to the throat
    # cap poles are OURS to place (we reuse loft topology, not its geometry):
    # a domed crown and a tucked throat point — never leave them at the origin
    v[T.HEAD_BOTTOM_POLE] = [0.0, 0.06, -0.01]
    v[T.HEAD_TOP_POLE] = [0.0, -0.01, 1.03]

    for r, c, off in T.NEUTRAL_OFFSETS:
        v[T.hid(r, c)] += off

    # ---- facial slider atoms (all deltas in Hh)
    def add(group, dx=0.0, dy=0.0, dz=0.0, s=1.0):
        idx, wt = T.GROUPS[group]
        v[idx] += np.outer(wt * s, [dx, dy, dz])

    add("brow", dz=0.035 * d["brow_height"])
    add("brow", dy=-0.038 * d["brow_depth_eff"])
    es = d["eye_size"]
    for grp, ex in (("socket_l", 0.155), ("socket_r", -0.155)):
        idx, wt = T.GROUPS[grp]
        center = np.array([ex, -0.30, 0.53])
        v[idx] = center + (v[idx] - center) * (1 + 0.20 * es * wt)[:, None]
        v[idx] += np.outer(wt, [np.sign(ex) * 0.040 * d["eye_spacing"],
                                0.028 * d["eye_depth"],
                                0.030 * d["eye_height"]])
    v[T.GROUPS["bridge"][0][0]] += [0, -0.030 * d["nose_bridge_height"], 0]  # (7,c0)
    v[T.hid(8, 0)] += [0, -0.015 * d["nose_bridge_height"], 0]
    v[T.hid(6, 0)] += [0, -0.035 * d["nose_bridge_depth"], 0]
    v[T.hid(7, 0)] += [0, -0.010 * d["nose_bridge_depth"], 0]
    add("nose_tip", dz=-0.045 * d["nose_length"])
    add("wings", dz=-0.030 * d["nose_length"])
    wid, wwt = T.GROUPS["wings"]
    v[wid, 0] *= (1 + 0.35 * d["nose_width"] * wwt)
    v[T.hid(5, 0)] += [0, -0.020 * d["nose_tip"], 0.012 * d["nose_tip"]]
    add("cheekbone", dz=0.032 * d["cheek_height_eff"])
    cid, cwt = T.GROUPS["cheekbone"]
    malar = np.stack([np.sign(v[cid, 0]) * 0.55, np.full(len(cid), -0.75),
                      np.full(len(cid), 0.35)], axis=1)
    malar /= np.linalg.norm(malar, axis=1, keepdims=True)
    v[cid] += 0.040 * d["cheek_depth"] * cwt[:, None] * malar
    fid, fwt = T.GROUPS["cheek_flesh"]
    radial = v[fid].copy()
    radial[:, 2] = 0.0
    nrm = np.linalg.norm(radial, axis=1, keepdims=True)
    nrm[nrm < 1e-9] = 1.0
    v[fid] += 0.035 * d["cheek_full_eff"] * fwt[:, None] * radial / nrm
    jid, jwt = T.GROUPS["jaw"]
    v[jid, 0] += np.sign(v[jid, 0]) * 0.038 * d["jaw_width_eff"] * jwt
    add("chin", dz=-0.040 * d["chin_length"],
        dy=-0.010 * max(d["chin_length"], 0.0))
    for c in (1, 15):
        i = T.hid(1, c)
        v[i, 0] += np.sign(v[i, 0]) * 0.028 * d["chin_width_eff"]
    mid, mwt = T.GROUPS["mouth_crn"]
    v[mid, 0] += np.sign(v[mid, 0]) * 0.032 * d["mouth_width"] * mwt
    lf = d["lip_full_eff"]
    add("lip_up", dy=-0.022 * lf, dz=0.006 * lf)
    add("lip_low", dy=-0.026 * lf, dz=-0.006 * lf)
    eid, ewt = T.GROUPS["ear"]
    v[eid, 0] += np.sign(v[eid, 0]) * 0.035 * d["ear_size"] * ewt
    for r, zsgn in ((6, -1.0), (7, 1.0)):  # rows spread apart
        for c in (5, 11):
            v[T.hid(r, c), 2] += zsgn * 0.012 * d["ear_size"]
    return v


def head_world(d, local: np.ndarray) -> np.ndarray:
    Hh = HEAD_UNIT * d["head_size"]
    origin = np.array([0.0, 0.004, _torso_z(d, _chin_z(d))])
    return origin + local * Hh


def eyeball_meshes(d) -> list[Mesh]:
    Hh = HEAD_UNIT * d["head_size"]
    origin = np.array([0.0, 0.004, _torso_z(d, _chin_z(d))])
    r = 0.056 * Hh * (1 + 0.30 * d["eye_size"])
    out = []
    for sx in (1.0, -1.0):
        c = origin + Hh * np.array([sx * (0.155 + 0.040 * d["eye_spacing"]),
                                    -0.285 + 0.024 * d["eye_depth"],
                                    0.53 + 0.030 * d["eye_height"]])
        out.append(primitives.icosphere(radius=r, subdivisions=1).translated(c))
    return out


# --------------------------------------------------------------------------
# poses (the standard humanoid set, shared with the templates)

POSES: dict[str, dict[str, tuple[float, float, float]]] = {
    "idle": {"shoulder_l": (5, -6, 0), "shoulder_r": (5, 6, 0),
             "elbow_l": (-18, 0, 0), "elbow_r": (-18, 0, 0),
             "spine": (2, 0, 0), "neck": (-2, 0, 0)},
    "walk": {"shoulder_l": (16, 0, 0), "shoulder_r": (-18, 0, 0),
             "elbow_l": (-10, 0, 0), "elbow_r": (-28, 0, 0),
             "hip_l": (-18, 0, 0), "knee_l": (6, 0, 0),
             "hip_r": (14, 0, 0), "knee_r": (24, 0, 0),
             "ankle_r": (-12, 0, 0), "spine": (4, 0, 0)},
    "attack": {"shoulder_r": (-130, 10, 0), "elbow_r": (-35, 0, 0),
               "wrist_r": (-20, 0, 0), "shoulder_l": (-30, -25, 0),
               "elbow_l": (-30, 0, 0), "hip_l": (-16, 0, 0),
               "knee_l": (8, 0, 0), "hip_r": (12, 0, 0), "knee_r": (16, 0, 0),
               "spine": (6, 0, -14), "neck": (-6, 0, 8)},
    "guard": {"shoulder_l": (-42, -18, 0), "elbow_l": (-64, 20, 0),
              "shoulder_r": (24, 12, 0), "elbow_r": (-70, 0, 0),
              "hip_l": (-10, 0, 0), "hip_r": (8, 0, 0), "knee_l": (6, 0, 0),
              "knee_r": (12, 0, 0), "spine": (7, 0, 6)},
}


# --------------------------------------------------------------------------
# assembly

SKIN_COLOR = "#c99e7c"
EYE_COLOR = "#2e2b28"


def _lbs(verts, W, arm: Armature, pose) -> np.ndarray:
    skins = arm.skin_matrices(pose)
    B = np.stack([skins[b] for b in T.BONES])  # (nb, 4, 4)
    hom = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
    per_bone = np.einsum("bij,vj->bvi", B[:, :3, :], hom)  # (nb, v, 3)
    return np.einsum("vb,bvi->vi", W, per_bone)


def character_shells(params: dict[str, float] | None = None,
                     pose_name: str | None = None,
                     levels: int = 2) -> list[tuple[str, Mesh, str]]:
    """Build the character as ``[(name, mesh, color)]`` shells.

    ``levels`` is the Catmull-Clark depth (1 = live preview, 2 = beauty/print).
    """
    d = resolve(params or {})
    arm = joint_layout(d)
    body_raw = body_verts_raw(d, arm)
    body_v = body_raw[T.BODY_KEEP]
    head_v = head_world(d, head_verts_local(d))
    eyes = eyeball_meshes(d)

    # height = TRUE total height: normalize by the SUBDIVIDED rest extent
    # (Catmull-Clark pulls the surface inside its cage, so cage extent lies)
    # — measured at rest so posing (crouch, reach) never rescales the figure
    rb, _ = subdivide_quads(body_v, T.BODY_QUADS, levels)
    rh, _ = subdivide_quads(head_v, T.HEAD_QUADS, levels)
    zmin = min(rb[:, 2].min(), rh[:, 2].min())
    zmax = max(rb[:, 2].max(), rh[:, 2].max())
    norm = 1.0 / max(zmax - zmin, 1e-9)
    shift = -zmin

    if pose_name is not None:
        if pose_name not in POSES:
            raise KeyError(f"unknown pose {pose_name!r}; known: {sorted(POSES)}")
        pose = POSES[pose_name]
        body_v = _lbs(body_v, T.BODY_WEIGHTS, arm, pose)
        M = arm.skin_matrices(pose)["head_top"]
        head_v = head_v @ M[:3, :3].T + M[:3, 3]
        eyes = [e.transform(M) for e in eyes]

    mm = d["height"] * norm
    off = np.array([0.0, 0.0, shift])
    bv, bq = subdivide_quads((body_v + off) * mm, T.BODY_QUADS, levels)
    hv, hq = subdivide_quads((head_v + off) * mm, T.HEAD_QUADS, levels)
    shells = [("body", quads_to_mesh(bv, bq), SKIN_COLOR),
              ("head", quads_to_mesh(hv, hq), SKIN_COLOR),
              ("eye_l", eyes[0].translated(off).scaled(mm), EYE_COLOR),
              ("eye_r", eyes[1].translated(off).scaled(mm), EYE_COLOR)]
    return shells


def character_mesh(params=None, pose_name=None, levels: int = 2,
                   check: bool = True) -> Mesh:
    """One merged printable mesh (per-shell integrity-gated when ``check``)."""
    shells = character_shells(params, pose_name, levels)
    if check:
        for name, mesh, _ in shells:
            rep = mesh.integrity_report()
            if not (rep["watertight"] and rep["volume"] > 0):
                raise ValueError(f"character shell {name!r} failed the gate: {rep}")
    return Mesh.merge([mesh for _, mesh, _ in shells])
