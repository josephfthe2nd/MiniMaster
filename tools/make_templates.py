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

    hip_x = 0.058 * H * np.sqrt(bulk)
    knee_x = hip_x * 1.02
    ankle_x = hip_x * 1.06
    shoulder_x = 0.10 * H * shoulders
    shoulder_z = neck_z - 0.012 * H
    elbow_x = shoulder_x + 0.012 * H
    elbow_z = shoulder_z - 0.14 * H * arms
    wrist_x = elbow_x + 0.02 * H
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

    # ---- torso
    torso_w = 0.19 * H * shoulders * np.sqrt(bulk)
    torso_d = (0.085 if slim_torso else 0.105) * H * np.sqrt(bulk)
    scene.add_shape(
        "box",
        name="hips",
        position=[0, 0, (pelvis_z + hip_z) / 2 - 0.01 * H],
        scale=[hip_x * 2 + 0.06 * H * np.sqrt(bulk), torso_d * 0.92, 0.10 * H],
        bone="spine",
        color=torso_color,
    )
    scene.add_shape(
        "box",
        name="belly",
        position=[0, 0, (spine_z + pelvis_z) / 2 + 0.01 * H],
        scale=[torso_w * 0.82, torso_d * 0.95, 0.13 * H],
        bone="chest",
        color=torso_color,
    )
    scene.add_shape(
        "cylinder",
        name="chest",
        params={"segments": 6, "taper": 0.78},
        position=[0, 0, (chest_z + spine_z) / 2 + 0.028 * H],
        rotation=[180.0, 0.0, 0.0],  # wider at shoulders, tapering to waist
        scale=[torso_w * 1.18, torso_d * 1.35, 0.185 * H],
        bone="chest",
        color=torso_color,
    )
    scene.add_shape(
        "cylinder",
        name="neck",
        params={"segments": 6},
        position=[0, 0, (neck_z + chest_z) / 2 + 0.015 * H],
        scale=[0.075 * H, 0.075 * H, 0.09 * H],
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
        position=[0, 0, head_c],
        scale=[head_d * 0.92, head_d * 0.98, head_d],
        bone="head_top",
        color=skin,
    )
    scene.add_shape(
        "box",
        name="nose",
        position=[0, -head_d * 0.52 - 0.012 * H * nose, head_c - head_d * 0.08],
        scale=[0.045 * H, 0.055 * H * nose, 0.055 * H],
        rotation=[-12.0, 0.0, 0.0],
        bone="head_top",
        color=skin,
    )

    # ---- arms + legs
    thigh_t = 0.095 * H * bulk
    shin_t = 0.082 * H * bulk
    uarm_t = 0.075 * H * bulk
    farm_t = 0.064 * H * bulk
    for side in ("l", "r"):
        sh, el, wr, hd = (J[f"{k}_{side}"] for k in ("shoulder", "elbow", "wrist", "hand"))
        hp, kn, an, to = (J[f"{k}_{side}"] for k in ("hip", "knee", "ankle", "toe"))
        ball(scene, f"shoulder_{side}", sh + [0, 0, 0.008 * H], uarm_t * 1.5,
             f"elbow_{side}", torso_color)
        limb(scene, f"upper_arm_{side}", sh, el, uarm_t, f"elbow_{side}", limb_color)
        ball(scene, f"elbow_pad_{side}", el, farm_t * 1.35, f"wrist_{side}", limb_color)
        limb(scene, f"forearm_{side}", el, wr, farm_t, f"wrist_{side}", limb_color,
             taper=0.8)
        scene.add_shape(
            "box",
            name=f"hand_{side}",
            position=(wr + hd) / 2.0 + [0, 0, -0.002 * H],
            scale=[0.055 * H, 0.07 * H, 0.075 * H],
            bone=f"hand_{side}",
            color=skin,
        )
        limb(scene, f"thigh_{side}", hp, kn, thigh_t, f"knee_{side}", limb_color,
             overlap=1.35)
        ball(scene, f"knee_pad_{side}", kn, shin_t * 1.3, f"ankle_{side}", limb_color)
        limb(scene, f"shin_{side}", kn, an, shin_t, f"ankle_{side}", boot_color,
             taper=0.78, overlap=1.3)
        foot_len = abs(to[1] - an[1]) + 0.05 * H
        scene.add_shape(
            "box",
            name=f"foot_{side}",
            position=[(an[0] + to[0]) / 2, (an[1] + to[1]) / 2 - 0.008 * H, 0.032 * H],
            scale=[0.075 * H, foot_len, 0.055 * H],
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
        z = spine_z + (i + 0.75) * (chest_z - spine_z) / 3.2
        scene.add_shape(
            "torus",
            name=f"rib_{i}",
            params={"segments": 8, "minor_segments": 4},
            position=[0, 0, z],
            scale=[w * (1.0 - 0.1 * i), L["torso_d"] * 1.35, 0.075 * H],
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
            "shoulder_l": (0, -4, 0),
            "shoulder_r": (0, 4, 0),
            "elbow_l": (-8, 0, 0),
            "elbow_r": (-8, 0, 0),
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
    }
    return {name: merged(pose) for name, pose in poses.items()}


# --------------------------------------------------------------------------
# templates


def make_human_fighter() -> Scene:
    scene = build_humanoid(
        "Human Fighter", height=30.0,
        skin="#c9a172", torso_color="#8d6f4a", limb_color="#a5824f",
    )
    add_sword(scene, "r")
    add_shield(scene, "l")
    scene.poses = humanoid_poses()
    scene.active_pose = "idle"
    return scene


def make_dwarf() -> Scene:
    scene = build_humanoid(
        "Dwarf Warrior", height=24.0,
        head=1.15, legs=0.72, arms=0.9, bulk=1.3, shoulders=1.25,
        skin="#c99b6f", torso_color="#7a5c3d", limb_color="#8e6d45",
    )
    add_beard(scene, "#9a6432")
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
        skin="#8aa050", torso_color="#5f6b3c", limb_color="#76894a",
        slim_torso=True, nose=1.8,
    )
    add_ears(scene, length=1.5, tilt=35.0)
    add_dagger(scene, "r")
    scene.poses = humanoid_poses(hunch=14.0)
    scene.active_pose = "idle"
    return scene


def make_orc() -> Scene:
    scene = build_humanoid(
        "Orc Brute", height=34.0,
        head=0.95, legs=0.95, arms=1.1, bulk=1.35, shoulders=1.3,
        skin="#7e9150", torso_color="#5c6b3f", limb_color="#6f8446",
    )
    add_tusks(scene)
    add_ears(scene, length=0.9, tilt=15.0)
    add_club(scene, "r")
    scene.poses = humanoid_poses(hunch=6.0)
    scene.active_pose = "idle"
    scene.base["diameter"] = 32.0
    return scene


def make_skeleton() -> Scene:
    bone_col = "#d8cfb6"
    scene = build_humanoid(
        "Skeleton", height=30.0,
        head=1.0, bulk=0.55, shoulders=0.92,
        skin=bone_col, torso_color=bone_col, limb_color=bone_col,
        boot_color=bone_col, slim_torso=True,
    )
    # swap the solid torso for a ribcage
    scene.remove_shape("belly")
    scene.remove_shape("chest")
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
