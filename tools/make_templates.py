"""Generate the starter template scenes shipped in minimaster/templates/.

Run from the repo root:  python tools/make_templates.py

Templates are ordinary .mmp project files built around a shared parameterized
humanoid: joint layout comes from proportion multipliers, limb shapes are
derived from the joint positions (so every variant stays consistently rigged),
and species flavor comes from feature add-ons (beard, ears, tusks, skull,
ribs) plus gear. Characters face -Y (toward the default preview camera).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minimaster.scene import Scene  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "minimaster" / "templates"


# --------------------------------------------------------------------------
# geometry helpers


def direction_euler(d) -> list[float]:
    """Euler XYZ (degrees, rz=0) rotating the +Z shape axis onto direction d.

    Solves R = Ry(ry) @ Rx(rx) with R @ (0,0,-1) = d̂  (limb axes point from
    parent joint down toward the child joint).
    """
    d = np.asarray(d, dtype=np.float64)
    d = d / np.linalg.norm(d)
    rx = np.degrees(np.arcsin(np.clip(d[1], -1.0, 1.0)))
    if abs(d[1]) > 0.9999:  # straight along Y: ry is free
        return [rx, 0.0, 0.0]
    ry = np.degrees(np.arctan2(-d[0], -d[2]))
    return [float(rx), float(ry), 0.0]


def limb(
    scene: Scene,
    name: str,
    a,
    b,
    thickness: float,
    bone: str,
    color: str,
    taper: float = 1.0,
    overlap: float = 1.25,
    depth: float | None = None,
    segments: int = 6,
    kind: str = "cylinder",
):
    """Faceted limb segment spanning joint position a -> b."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = b - a
    length = float(np.linalg.norm(d))
    params = {"segments": segments, "taper": taper} if kind == "cylinder" else {}
    scene.add_shape(
        kind,
        name=name,
        params=params,
        position=(a + b) / 2.0,
        rotation=direction_euler(d),
        scale=[thickness, depth if depth is not None else thickness, length * overlap],
        bone=bone,
        color=color,
    )


def ball(scene: Scene, name: str, pos, diameter: float, bone: str, color: str,
         subdiv: int = 0):
    scene.add_shape(
        "icosphere",
        name=name,
        params={"subdivisions": subdiv},
        position=pos,
        scale=[diameter] * 3,
        bone=bone,
        color=color,
    )


# --------------------------------------------------------------------------
# the parameterized humanoid


def build_humanoid(
    name: str,
    height: float = 30.0,
    head: float = 1.0,
    legs: float = 1.0,
    arms: float = 1.0,
    bulk: float = 1.0,
    shoulders: float = 1.0,
    skin: str = "#b08d57",
    torso_color: str = "#8d6f4a",
    limb_color: str = "#a5824f",
    boot_color: str = "#6b543a",
    slim_torso: bool = False,
    nose: float = 1.0,
    eye_scale: float = 1.0,
    eye_color: str = "#2e2b28",
    eye_depth: float = 0.44,
    brow: bool = False,
    mouth: bool = True,
    hair: str | None = None,
    belt: bool = True,
    feminine: bool = False,
) -> Scene:
    """Rig + carved body for a humanoid. Returns the scene plus joint layout
    stashed on ``scene._layout`` for feature/gear builders."""
    H = height
    scene = Scene(name=name)
    arm_side = 1.0  # left = +X (character faces -Y)

    # ---- joint layout
    ankle_z = 0.07 * H
    hip_z = ankle_z + 0.42 * H * legs
    knee_z = ankle_z + (hip_z - ankle_z) * 0.52
    pelvis_z = hip_z + 0.03 * H
    spine_z = pelvis_z + 0.115 * H
    chest_z = spine_z + 0.115 * H
    neck_z = chest_z + 0.075 * H
    head_top_z = neck_z + 0.16 * H * head

    # femoral valgus: legs converge from the hips down to closer knees/ankles
    hip_x = 0.052 * H * np.sqrt(bulk)
    knee_x = hip_x * 0.82
    ankle_x = hip_x * 0.74
    shoulder_x = 0.094 * H * shoulders
    shoulder_z = neck_z - 0.02 * H
    elbow_x = shoulder_x + 0.006 * H
    elbow_z = shoulder_z - 0.14 * H * arms
    wrist_x = elbow_x + 0.012 * H
    wrist_z = elbow_z - 0.125 * H * arms
    hand_z = wrist_z - 0.055 * H

    arm = scene.armature
    arm.add_joint("pelvis", [0, 0, pelvis_z])
    arm.add_joint("spine", [0, 0, spine_z], parent="pelvis")
    arm.add_joint("chest", [0, 0, chest_z], parent="spine")
    arm.add_joint("neck", [0, 0, neck_z], parent="chest")
    arm.add_joint("head_top", [0, 0, head_top_z], parent="neck")
    for side, sx in (("l", +1.0), ("r", -1.0)):
        arm.add_joint(f"shoulder_{side}", [sx * shoulder_x, 0, shoulder_z], parent="chest")
        arm.add_joint(f"elbow_{side}", [sx * elbow_x, 0, elbow_z], parent=f"shoulder_{side}")
        arm.add_joint(f"wrist_{side}", [sx * wrist_x, 0, wrist_z], parent=f"elbow_{side}")
        arm.add_joint(f"hand_{side}", [sx * wrist_x, 0, hand_z], parent=f"wrist_{side}")
        arm.add_joint(f"hip_{side}", [sx * hip_x, 0, hip_z], parent="pelvis")
        arm.add_joint(f"knee_{side}", [sx * knee_x, 0, knee_z], parent=f"hip_{side}")
        arm.add_joint(f"ankle_{side}", [sx * ankle_x, 0, ankle_z], parent=f"knee_{side}")
        arm.add_joint(
            f"toe_{side}", [sx * ankle_x, -0.085 * H, 0.02 * H], parent=f"ankle_{side}"
        )

    J = {n: j.position for n, j in arm.joints.items()}

    # ---- torso: capsule construction for a smooth organic silhouette with
    # real depth (the old box/cylinder stack was plank-flat from the side)
    torso_w = 0.185 * H * shoulders * np.sqrt(bulk)
    torso_d = (0.10 if slim_torso else 0.118) * H * np.sqrt(bulk)
    hips_w = hip_x * 2 + 0.055 * H * np.sqrt(bulk)
    if feminine:
        hips_w *= 1.16  # wider hips, narrower shoulders (caller lowers `shoulders`)
    scene.add_shape(
        "capsule",
        name="hips",
        params={"segments": 8},
        position=[0, 0, (pelvis_z + hip_z) / 2],
        scale=[hips_w, torso_d, 0.17 * H],
        bone="spine",
        color=torso_color,
    )
    scene.add_shape(
        "capsule",
        name="glutes",
        params={"segments": 6},
        position=[0, torso_d * 0.36, hip_z + 0.008 * H],
        scale=[hips_w * 0.76, torso_d * 0.66, 0.105 * H],
        bone="spine",
        color=torso_color,
    )
    scene.add_shape(
        "capsule",
        name="belly",
        params={"segments": 8},
        position=[0, 0, (spine_z + pelvis_z) / 2 + 0.012 * H],
        scale=[torso_w * 0.76, torso_d * 0.94, 0.21 * H],
        bone="chest",
        color=torso_color,
    )
    scene.add_shape(
        "capsule",
        name="chest",
        params={"segments": 8},
        position=[0, -torso_d * 0.1, (chest_z + spine_z) / 2 + 0.03 * H],
        scale=[torso_w * 1.12, torso_d * 1.18, 0.26 * H],  # projects forward
        bone="chest",
        color=torso_color,
    )
    # chest: flat male pectoral plates, or a fuller/higher female bust
    for _sx in (+1.0, -1.0):
        if feminine:
            scene.add_shape(
                "capsule", name=f"pec_{'l' if _sx > 0 else 'r'}",
                params={"segments": 6},
                position=[_sx * torso_w * 0.26, -torso_d * 0.72,
                          chest_z - 0.005 * H],
                scale=[torso_w * 0.52, torso_d * 0.62, 0.12 * H],
                bone="chest", color=torso_color,
            )
        else:
            scene.add_shape(
                "capsule", name=f"pec_{'l' if _sx > 0 else 'r'}",
                params={"segments": 6},
                position=[_sx * torso_w * 0.3, -torso_d * 0.66,
                          chest_z + 0.02 * H],
                scale=[torso_w * 0.64, torso_d * 0.42, 0.075 * H],
                bone="chest", color=torso_color,
            )
    if belt:
        scene.add_shape(
            "cylinder",
            name="belt",
            params={"segments": 8},
            position=[0, 0, pelvis_z + 0.015 * H],
            scale=[hips_w * 1.02, torso_d * 1.06, 0.035 * H],
            bone="spine",
            color="#4f3d2a",
        )
    scene.add_shape(
        "cylinder",
        name="neck",
        params={"segments": 8, "taper": 0.82},
        position=[0, 0.004 * H, (neck_z + chest_z) / 2 + 0.015 * H],
        scale=[0.068 * H, 0.068 * H, 0.1 * H],
        bone="head_top",
        color=skin,
    )

    # ---- head
    head_span = head_top_z - neck_z
    head_c = neck_z + head_span * 0.58
    head_d = head_span * 1.02
    scene.add_shape(
        "icosphere",
        name="head",
        params={"subdivisions": 1},
        position=[0, 0.01 * H, head_c],  # skull mass sits slightly back
        scale=[head_d * 0.9, head_d * 1.08, head_d],
        bone="head_top",
        color=skin,
    )
    scene.add_shape(
        "capsule",
        name="jaw",
        params={"segments": 6},
        position=[0, -head_d * 0.12, head_c - head_d * 0.36],
        rotation=[80.0, 0.0, 0.0],  # subtle chin, OSRS heads are simple ovoids
        scale=[head_d * 0.46, head_d * 0.34, head_d * 0.5],
        bone="head_top",
        color=skin,
    )
    scene.add_shape(
        "box",
        name="nose",
        position=[0, 0.01 * H - head_d * 0.5 - 0.012 * H * nose, head_c - head_d * 0.08],
        scale=[0.045 * H, 0.055 * H * nose, 0.055 * H],
        rotation=[-12.0, 0.0, 0.0],
        bone="head_top",
        color=skin,
    )
    eye_d = head_d * 0.24 * eye_scale
    for side, sx in (("l", +1.0), ("r", -1.0)):
        scene.add_shape(
            "icosphere",
            name=f"eye_{side}",
            params={"subdivisions": 0},
            position=[sx * head_d * 0.17, 0.01 * H - head_d * eye_depth,
                      head_c + head_d * 0.04],
            scale=[eye_d, eye_d, eye_d],
            bone="head_top",
            color=eye_color,
        )
    if hair is not None:
        scene.add_shape(
            "icosphere",
            name="hair",
            params={"subdivisions": 1},
            position=[0, 0.014 * H + head_d * 0.06, head_c + head_d * 0.14],
            scale=[head_d * 0.94, head_d * 1.06, head_d * 0.78],
            bone="head_top",
            color=hair,
        )
    if mouth:
        scene.add_shape(
            "box",
            name="mouth",
            position=[0, 0.01 * H - head_d * 0.47, head_c - head_d * 0.3],
            scale=[head_d * 0.3, head_d * 0.1, head_d * 0.055],
            bone="head_top",
            color="#4a3a2c",
        )
    if brow:
        scene.add_shape(
            "box",
            name="brow",
            position=[0, 0.01 * H - head_d * 0.4, head_c + head_d * 0.23],
            rotation=[-24.0, 0.0, 0.0],
            scale=[head_d * 0.58, head_d * 0.22, head_d * 0.13],
            bone="head_top",
            color=skin,
        )

    # ---- arms + legs: 8-segment limbs with muscle taper — deltoid into
    # tapering upper arm, forearm narrowing to the wrist, thigh clearly
    # thicker than the calf, calf bulge high on the shin, thin ankle
    thigh_t = 0.098 * H * bulk
    shin_t = 0.078 * H * bulk
    uarm_t = 0.076 * H * bulk
    farm_t = 0.062 * H * bulk
    for side in ("l", "r"):
        sh, el, wr, hd = (J[f"{k}_{side}"] for k in ("shoulder", "elbow", "wrist", "hand"))
        hp, kn, an, to = (J[f"{k}_{side}"] for k in ("hip", "knee", "ankle", "toe"))
        sxs = +1.0 if side == "l" else -1.0
        # OSRS shoulders read as a simple cap, not a bulky ball
        ball(scene, f"shoulder_{side}", sh + [sxs * 0.002 * H, 0, -0.004 * H],
             uarm_t * 1.08, f"elbow_{side}", limb_color, subdiv=1)
        limb(scene, f"upper_arm_{side}", sh, el, uarm_t, f"elbow_{side}",
             limb_color, taper=0.72, segments=8)
        # biceps belly on the front, upper third of the upper arm
        scene.add_shape(
            "capsule", name=f"biceps_{side}", params={"segments": 6},
            position=[sh[0] + (el[0] - sh[0]) * 0.36, sh[1] - uarm_t * 0.5,
                      sh[2] + (el[2] - sh[2]) * 0.36],
            scale=[uarm_t * 0.8, uarm_t * 0.8, abs(el[2] - sh[2]) * 0.55],
            bone=f"elbow_{side}", color=limb_color,
        )
        ball(scene, f"elbow_pad_{side}", el, farm_t * 1.08, f"wrist_{side}",
             limb_color)
        # forearm flexor mass high near the elbow, tapering to a thin wrist
        scene.add_shape(
            "capsule", name=f"forearm_flex_{side}", params={"segments": 6},
            position=[el[0] + (wr[0] - el[0]) * 0.28, el[1] - farm_t * 0.15,
                      el[2] + (wr[2] - el[2]) * 0.28],
            scale=[farm_t * 1.15, farm_t * 1.15, abs(wr[2] - el[2]) * 0.5],
            bone=f"wrist_{side}", color=limb_color,
        )
        limb(scene, f"forearm_{side}", el, wr, farm_t * 0.92, f"wrist_{side}",
             limb_color, taper=0.5, segments=8)
        scene.add_shape(
            "box",
            name=f"hand_{side}",
            position=(wr + hd) / 2.0 + [0, 0, 0.006 * H],
            rotation=[0.0, 0.0, sxs * 8.0],  # palms turn slightly inward
            scale=[0.046 * H, 0.06 * H, 0.068 * H],
            bone=f"hand_{side}",
            color=skin,
        )
        scene.add_shape(
            "box",
            name=f"thumb_{side}",
            position=(wr + hd) / 2.0 + [-sxs * 0.026 * H, -0.018 * H, 0.012 * H],
            rotation=[0.0, sxs * 18.0, sxs * 24.0],
            scale=[0.024 * H, 0.03 * H, 0.042 * H],
            bone=f"hand_{side}",
            color=skin,
        )
        limb(scene, f"thigh_{side}", hp, kn, thigh_t, f"knee_{side}", limb_color,
             taper=0.78, overlap=1.46, segments=8)
        ball(scene, f"knee_pad_{side}", kn, shin_t * 1.16, f"ankle_{side}",
             limb_color)
        scene.add_shape(
            "capsule",
            name=f"calf_{side}",
            params={"segments": 6},
            position=[kn[0], kn[1] + shin_t * 0.32,
                      kn[2] - (kn[2] - an[2]) * 0.26],  # bulge high on the shin
            scale=[shin_t * 1.04, shin_t * 1.22, (kn[2] - an[2]) * 0.5],
            bone=f"ankle_{side}",
            color=boot_color,
        )
        limb(scene, f"shin_{side}", kn, an, shin_t * 0.9, f"ankle_{side}",
             boot_color, taper=0.52, overlap=1.28, segments=8)
        foot_len = abs(to[1] - an[1]) + 0.04 * H
        scene.add_shape(
            "box",
            name=f"foot_{side}",
            position=[(an[0] + to[0]) / 2 + sxs * 0.006 * H,
                      (an[1] + to[1]) / 2 - 0.004 * H, 0.028 * H],
            rotation=[0.0, 0.0, sxs * 9.0],  # relaxed outward splay
            scale=[0.066 * H, foot_len * 0.85, 0.048 * H],
            bone=f"toe_{side}",
            color=boot_color,
        )
        scene.add_shape(
            "box",
            name=f"heel_{side}",
            position=[an[0], an[1] + 0.024 * H, 0.03 * H],
            scale=[0.058 * H, 0.038 * H, 0.05 * H],
            bone=f"toe_{side}",
            color=boot_color,
        )
        scene.add_shape(
            "wedge",
            name=f"toe_{side}",
            position=[to[0] + sxs * 0.01 * H, to[1] - 0.018 * H, 0.022 * H],
            rotation=[0.0, 0.0, -90.0 + sxs * 9.0],  # follows the foot splay
            scale=[0.045 * H, 0.06 * H, 0.038 * H],
            bone=f"toe_{side}",
            color=boot_color,
        )

    scene._layout = {  # stash for feature/gear builders
        "H": H, "J": J, "head_c": head_c, "head_d": head_d,
        "torso_w": torso_w, "torso_d": torso_d, "skin": skin,
    }
    scene.base = {"style": "round", "diameter": 25.0, "height": 3.0}
    return scene


# --------------------------------------------------------------------------
# features


def add_beard(scene: Scene, color: str = "#8a5a33"):
    L = scene._layout
    H, head_c, head_d = L["H"], L["head_c"], L["head_d"]
    scene.add_shape(
        "cylinder",
        name="beard",
        params={"segments": 4, "taper": 0.3},
        position=[0, -head_d * 0.48, head_c - head_d * 0.78],
        rotation=[172.0, 0.0, 45.0],  # fat end up, hanging slightly forward
        scale=[head_d * 0.72, head_d * 0.6, head_d * 1.05],
        bone="head_top",
        color=color,
    )


def add_ears(scene: Scene, length: float = 1.0, tilt: float = 25.0):
    L = scene._layout
    head_c, head_d, skin = L["head_c"], L["head_d"], L["skin"]
    for side, sx in (("l", +1.0), ("r", -1.0)):
        scene.add_shape(
            "cylinder",
            name=f"ear_{side}",
            params={"segments": 3, "taper": 0.08},
            position=[sx * head_d * 0.52, head_d * 0.05, head_c + head_d * 0.08],
            rotation=[0.0, sx * (90.0 + tilt), 0.0],
            scale=[head_d * 0.28, head_d * 0.28, head_d * 0.55 * length],
            bone="head_top",
            color=skin,
        )


def add_tusks(scene: Scene, color: str = "#e8ddc4"):
    L = scene._layout
    H, head_c, head_d = L["H"], L["head_c"], L["head_d"]
    for side, sx in (("l", +1.0), ("r", -1.0)):
        scene.add_shape(
            "cylinder",
            name=f"tusk_{side}",
            params={"segments": 4, "taper": 0.3},
            position=[sx * head_d * 0.24, -head_d * 0.52, head_c - head_d * 0.3],
            rotation=[16.0, 0.0, 0.0],
            scale=[0.035 * H, 0.035 * H, 0.075 * H],
            bone="head_top",
            color=color,
        )


def add_cape(scene: Scene, color: str = "#8a2f2f", trim: str | None = None):
    """Iconic OSRS cape: a wide slab hanging off the upper back, bound to the
    chest so it sways with the torso. Slight backward tilt + a collar."""
    L = scene._layout
    H, J = L["H"], L["J"]
    chest_z = J["chest"][2]
    hip_z = J["hip_l"][2]
    top, bottom = chest_z + 0.06 * H, hip_z - 0.2 * H
    back_y = L["torso_d"] * 1.15
    scene.add_shape(
        "box",
        name="cape",
        position=[0, back_y, (top + bottom) / 2],
        rotation=[7.0, 0.0, 0.0],
        scale=[L["torso_w"] * 1.7, 0.03 * H, top - bottom],
        bone="chest",
        color=color,
    )
    scene.add_shape(
        "box",
        name="cape_collar",
        position=[0, back_y * 0.86, chest_z + 0.09 * H],
        scale=[L["torso_w"] * 1.5, 0.05 * H, 0.05 * H],
        bone="chest",
        color=trim or color,
    )


def add_ribs(scene: Scene, color: str):
    """Skeleton flavor: replace solid torso with slats (call after removing
    belly/chest shapes)."""
    L = scene._layout
    H, J = L["H"], L["J"]
    spine_z, chest_z = J["spine"][2], J["chest"][2]
    scene.add_shape(
        "cylinder",
        name="spine_column",
        params={"segments": 6},
        position=[0, 0, (spine_z + chest_z) / 2],
        scale=[0.045 * H, 0.045 * H, chest_z - spine_z + 0.1 * H],
        bone="chest",
        color=color,
    )
    w = L["torso_w"]
    for i in range(3):
        z = spine_z + (i + 0.55) * (chest_z - spine_z) / 2.7
        scene.add_shape(
            "torus",
            name=f"rib_{i}",
            params={"segments": 10, "minor_segments": 4},
            position=[0, 0, z],
            scale=[w * (1.12 - 0.14 * i), L["torso_d"] * 1.7, 0.042 * H],
            bone="chest",
            color=color,
        )


# --------------------------------------------------------------------------
# gear


def add_sword(scene: Scene, side: str = "r", color: str = "#b9c2c9",
              grip_color: str = "#6b543a"):
    L = scene._layout
    H, J = L["H"], L["J"]
    hand = (J[f"wrist_{side}"] + J[f"hand_{side}"]) / 2.0
    bone = f"hand_{side}"
    blade_l = 0.42 * H
    scene.add_shape(
        "box",
        name=f"sword_blade_{side}",
        position=hand + [0, 0, blade_l / 2 + 0.05 * H],
        scale=[0.045 * H, 0.014 * H, blade_l],
        bone=bone,
        color=color,
    )
    scene.add_shape(
        "cylinder",
        name=f"sword_tip_{side}",
        params={"segments": 4, "taper": 0.0},
        position=hand + [0, 0, blade_l + 0.075 * H],
        scale=[0.045 * H, 0.014 * H, 0.05 * H],
        bone=bone,
        color=color,
    )
    scene.add_shape(
        "box",
        name=f"sword_guard_{side}",
        position=hand + [0, 0, 0.045 * H],
        scale=[0.11 * H, 0.03 * H, 0.022 * H],
        bone=bone,
        color=grip_color,
    )
    scene.add_shape(
        "cylinder",
        name=f"sword_grip_{side}",
        params={"segments": 6},
        position=hand + [0, 0, -0.045 * H],
        scale=[0.03 * H, 0.03 * H, 0.09 * H],
        bone=bone,
        color=grip_color,
    )
    ball(scene, f"sword_pommel_{side}", hand + [0, 0, -0.1 * H], 0.045 * H, bone,
         color)


def add_shield(scene: Scene, side: str = "l", color: str = "#7d6a4d",
               boss_color: str = "#b9c2c9"):
    L = scene._layout
    H, J = L["H"], L["J"]
    mid = (J[f"elbow_{side}"] + J[f"wrist_{side}"]) / 2.0
    sx = +1.0 if side == "l" else -1.0
    pos = mid + [sx * 0.05 * H, 0, 0]
    bone = f"wrist_{side}"
    scene.add_shape(
        "cylinder",
        name=f"shield_{side}",
        params={"segments": 12},
        position=pos,
        rotation=[0.0, 90.0, 0.0],
        scale=[0.34 * H, 0.34 * H, 0.028 * H],
        bone=bone,
        color=color,
    )
    scene.add_shape(
        "cylinder",
        name=f"shield_boss_{side}",
        params={"segments": 8, "taper": 0.55},
        position=pos + [sx * 0.02 * H, 0, 0],
        rotation=[0.0, sx * 90.0, 0.0],
        scale=[0.1 * H, 0.1 * H, 0.035 * H],
        bone=bone,
        color=boss_color,
    )


def add_axe(scene: Scene, side: str = "r", haft_color: str = "#6b543a",
            head_color: str = "#b9c2c9"):
    L = scene._layout
    H, J = L["H"], L["J"]
    hand = (J[f"wrist_{side}"] + J[f"hand_{side}"]) / 2.0
    bone = f"hand_{side}"
    haft_l = 0.5 * H
    scene.add_shape(
        "cylinder",
        name=f"axe_haft_{side}",
        params={"segments": 6},
        position=hand + [0, 0, haft_l * 0.28],
        scale=[0.035 * H, 0.035 * H, haft_l],
        bone=bone,
        color=haft_color,
    )
    # Blade extends forward (-Y): wedge rotated so its slope tapers to a
    # front-bottom cutting edge, its vertical back face against the haft.
    scene.add_shape(
        "wedge",
        name=f"axe_head_{side}",
        position=hand + [0, -0.075 * H, haft_l * 0.40],
        rotation=[0.0, 0.0, -90.0],
        scale=[0.16 * H, 0.04 * H, 0.14 * H],
        bone=bone,
        color=head_color,
    )


def add_club(scene: Scene, side: str = "r", color: str = "#7b5c39"):
    L = scene._layout
    H, J = L["H"], L["J"]
    hand = (J[f"wrist_{side}"] + J[f"hand_{side}"]) / 2.0
    bone = f"hand_{side}"
    sx = +1.0 if side == "l" else -1.0
    tilt = [-14.0, sx * 10.0, 0.0]  # lean forward and away from the body
    tip = np.array(
        [np.sin(np.radians(sx * 10.0)), np.sin(np.radians(14.0)), 1.0]
    )  # approximate club axis after the tilt
    scene.add_shape(
        "cylinder",
        name=f"club_{side}",
        params={"segments": 6, "taper": 0.45},
        position=hand + tip * 0.17 * H,
        rotation=[180.0 + tilt[0], -tilt[1], 0.0],  # fat end up, tilted
        scale=[0.1 * H, 0.1 * H, 0.46 * H],
        bone=bone,
        color=color,
    )
    ball(scene, f"club_knob_{side}", hand + tip * 0.38 * H, 0.12 * H, bone, color)


def add_dagger(scene: Scene, side: str = "r", color: str = "#b9c2c9",
               grip_color: str = "#5d4a33"):
    L = scene._layout
    H, J = L["H"], L["J"]
    hand = (J[f"wrist_{side}"] + J[f"hand_{side}"]) / 2.0
    bone = f"hand_{side}"
    scene.add_shape(
        "cylinder",
        name=f"dagger_blade_{side}",
        params={"segments": 4, "taper": 0.08},
        position=hand + [0, -0.045 * H, 0.13 * H],
        rotation=[-22.0, 0.0, 0.0],  # angled forward, ready to stab
        scale=[0.05 * H, 0.02 * H, 0.22 * H],
        bone=bone,
        color=color,
    )
    scene.add_shape(
        "box",
        name=f"dagger_guard_{side}",
        position=hand + [0, -0.012 * H, 0.04 * H],
        rotation=[-22.0, 0.0, 0.0],
        scale=[0.09 * H, 0.028 * H, 0.02 * H],
        bone=bone,
        color=grip_color,
    )


# --------------------------------------------------------------------------
# poses


def humanoid_poses(hunch: float = 0.0) -> dict[str, dict]:
    """Standard pose set. ``hunch`` adds a permanent forward stoop (degrees)."""

    def merged(pose: dict) -> dict:
        if hunch:
            pose = dict(pose)
            pose["spine"] = tuple(
                np.add(pose.get("spine", (0, 0, 0)), (hunch, 0.0, 0.0))
            )
            pose.setdefault("neck", (-hunch * 0.6, 0.0, 0.0))
        return {k: tuple(float(x) for x in v) for k, v in pose.items()}

    poses = {
        "idle": {
            "shoulder_l": (5, -6, 0),
            "shoulder_r": (5, 6, 0),
            "elbow_l": (-18, 0, 0),
            "elbow_r": (-18, 0, 0),
            "spine": (2, 0, 0),
            "neck": (-2, 0, 0),
        },
        "walk": {
            "shoulder_l": (16, 0, 0),
            "shoulder_r": (-18, 0, 0),
            "elbow_l": (-10, 0, 0),
            "elbow_r": (-28, 0, 0),
            "hip_l": (-18, 0, 0),
            "knee_l": (6, 0, 0),
            "hip_r": (14, 0, 0),
            "knee_r": (24, 0, 0),
            "ankle_r": (-12, 0, 0),
            "spine": (4, 0, 0),
        },
        "attack": {
            "shoulder_r": (-130, 10, 0),
            "elbow_r": (-35, 0, 0),
            "wrist_r": (-20, 0, 0),
            "shoulder_l": (-30, -25, 0),
            "elbow_l": (-30, 0, 0),
            "hip_l": (-16, 0, 0),
            "knee_l": (8, 0, 0),
            "hip_r": (12, 0, 0),
            "knee_r": (16, 0, 0),
            "spine": (6, 0, -14),
            "neck": (-6, 0, 8),
        },
        "guard": {
            "shoulder_l": (-42, -18, 0),
            "elbow_l": (-64, 20, 0),
            "shoulder_r": (24, 12, 0),
            "elbow_r": (-70, 0, 0),
            "hip_l": (-10, 0, 0),
            "hip_r": (8, 0, 0),
            "knee_l": (6, 0, 0),
            "knee_r": (12, 0, 0),
            "spine": (7, 0, 6),
        },
        "charge": {  # running lunge, weapon arm trailing for the swing
            "spine": (14, 0, -10),
            "neck": (-10, 0, 6),
            "hip_l": (-38, 0, 0),
            "knee_l": (20, 0, 0),
            "ankle_l": (-10, 0, 0),
            "hip_r": (26, 0, 0),
            "knee_r": (42, 0, 0),
            "ankle_r": (-20, 0, 0),
            "shoulder_r": (38, 8, 0),
            "elbow_r": (-30, 0, 0),
            "shoulder_l": (-46, -12, 0),
            "elbow_l": (-40, 0, 0),
        },
        "rage": {  # both arms up and out, roaring at the sky
            "spine": (-8, 0, 0),
            "neck": (-14, 0, 0),
            "shoulder_l": (-125, -35, 0),
            "elbow_l": (-25, 0, 0),
            "wrist_l": (-15, 0, 0),
            "shoulder_r": (-125, 35, 0),
            "elbow_r": (-25, 0, 0),
            "wrist_r": (-15, 0, 0),
            "hip_l": (-8, 0, 0),
            "hip_r": (8, 0, 0),
            "knee_l": (4, 0, 0),
            "knee_r": (8, 0, 0),
        },
    }
    return {name: merged(pose) for name, pose in poses.items()}


# --------------------------------------------------------------------------
# templates


def make_human_fighter() -> Scene:
    scene = build_humanoid(
        "Human Fighter", height=30.0, head=1.05,
        skin="#b78a5c", torso_color="#5f4a33", limb_color="#6f5640",
        boot_color="#4a3826", hair="#4f3b28",
    )
    add_cape(scene, color="#8f2f2c", trim="#c9a24a")  # classic OSRS cape
    add_sword(scene, "r")
    add_shield(scene, "l")
    scene.poses = humanoid_poses()
    scene.active_pose = "idle"
    return scene


def make_dwarf() -> Scene:
    scene = build_humanoid(
        "Dwarf Warrior", height=24.0,
        head=1.2, legs=0.72, arms=0.9, bulk=1.3, shoulders=1.25,
        skin="#b58a5f", torso_color="#5a4531", limb_color="#6b5238",
        boot_color="#3f3020", hair="#8a5a28",
    )
    add_beard(scene, "#9a6432")
    scene.base["style"] = "cobble"
    add_axe(scene, "r")
    add_shield(scene, "l")
    scene.poses = humanoid_poses()
    scene.active_pose = "idle"
    scene.base["diameter"] = 25.0
    return scene


def make_goblin() -> Scene:
    scene = build_humanoid(
        "Goblin", height=20.0,
        head=1.35, legs=0.9, arms=1.1, bulk=0.78, shoulders=0.9,
        skin="#6f8a3e", torso_color="#4a5230", limb_color="#5d6b38",
        boot_color="#3a3322", slim_torso=True, nose=1.8,
        eye_scale=1.55, eye_color="#d8b23a",  # big amber goblin eyes
    )
    add_ears(scene, length=1.5, tilt=35.0)
    add_dagger(scene, "r")
    scene.poses = humanoid_poses(hunch=14.0)
    scene.active_pose = "idle"
    return scene


def make_orc() -> Scene:
    scene = build_humanoid(
        "Orc Brute", height=34.0,
        head=0.92, legs=0.88, arms=1.28, bulk=1.5, shoulders=1.5,
        skin="#5f7d3c", torso_color="#454f2e", limb_color="#546b38",
        boot_color="#332d1f", eye_scale=0.85, eye_color="#c23a2a", brow=True,
    )
    add_tusks(scene)
    add_ears(scene, length=0.9, tilt=15.0)
    add_club(scene, "r")
    scene.base["style"] = "cobble"
    scene.poses = humanoid_poses(hunch=12.0)
    scene.active_pose = "idle"
    scene.base["diameter"] = 32.0
    return scene


def make_skeleton() -> Scene:
    bone_col = "#cfc6ad"
    scene = build_humanoid(
        "Skeleton", height=30.0,
        head=1.0, bulk=0.55, shoulders=0.92,
        skin=bone_col, torso_color=bone_col, limb_color=bone_col,
        boot_color=bone_col, slim_torso=True,
        eye_scale=1.4, eye_color="#181614", eye_depth=0.34,  # sunken sockets
        mouth=False, belt=False,
    )
    # swap the solid torso for a ribcage; strip all muscle bellies — bone has
    # none. The joint balls (shoulder/elbow/knee pads) stay as epiphysis knobs.
    scene.remove_shape("belly")
    scene.remove_shape("chest")
    scene.remove_shape("glutes")
    scene.remove_shape("jaw")
    for side in ("l", "r"):
        scene.remove_shape(f"calf_{side}")
        scene.remove_shape(f"biceps_{side}")
        scene.remove_shape(f"forearm_flex_{side}")
    scene.remove_shape("pec_l")
    scene.remove_shape("pec_r")
    add_ribs(scene, bone_col)
    scene.remove_shape("nose")  # skulls read better without one
    add_sword(scene, "r", color="#a9a290", grip_color="#5d4a33")
    scene.poses = humanoid_poses()
    scene.active_pose = "idle"
    return scene


TEMPLATES = {
    "human_fighter": make_human_fighter,
    "dwarf": make_dwarf,
    "goblin": make_goblin,
    "orc": make_orc,
    "skeleton": make_skeleton,
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, factory in TEMPLATES.items():
        scene = factory()
        path = OUT_DIR / f"{name}.mmp"
        scene.save(path)
        merged = scene.build_merged_mesh()
        rep = merged.integrity_report()
        status = "ok" if rep["watertight"] else f"NOT WATERTIGHT {rep}"
        print(f"{path.name}: {len(scene.shapes)} shapes, {len(scene.armature)} joints, "
              f"{len(scene.poses)} poses, {len(merged.faces)} tris [{status}]")


if __name__ == "__main__":
    main()
