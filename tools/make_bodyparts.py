"""Generate the base humanoid body parts as OSRS-styled .mmpart files.

Run from the repo root:  python tools/make_bodyparts.py

These are the atoms of the modular part system: one clean part per body
region (head, neck, torso, pelvis/crotch, upper arm, forearm, hand, thigh,
shin, foot), all in the muted OSRS palette and sized to the ~30 mm figure so
they read as a matching set. Each is authored in its natural on-body upright
orientation with the attachment joint near the local origin and +Z pointing
the way the part grows, so they place cleanly and render well as a catalog.

A byte-lock test keeps the shipped .mmpart JSON in sync with this generator
(thumbnails are not locked; PNG bytes vary).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minimaster.parts import Part, render_part_thumbnail  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "minimaster" / "parts"

# OSRS muted palette (shared with the templates' human look)
SKIN = "#b78a5c"
TUNIC = "#5f4a33"
SLEEVE = "#6f5640"
TROUSER = "#4f4030"
BOOT = "#3f3020"
HAIR = "#4f3b28"
BELT = "#3a2c1d"
EYE = "#241f1a"


def sd(name, kind, pos, scale, rot=(0.0, 0.0, 0.0), params=None, color=SKIN):
    return {
        "name": name,
        "kind": kind,
        "params": dict(params or {}),
        "position": [float(v) for v in pos],
        "rotation": [float(v) for v in rot],
        "scale": [float(v) for v in scale],
        "bone": None,
        "color": color,
    }


# --------------------------------------------------------------------------
# parts (upright, attachment near origin, +Z = growth direction)


def make_head() -> Part:
    hd = 4.9
    return Part(
        name="head", category="body", tags=["head"],
        shapes=[
            sd("skull", "icosphere", (0, 0.15, 2.9), (hd * 0.9, hd * 1.08, hd),
               params={"subdivisions": 1}, color=SKIN),
            sd("jaw", "capsule", (0, -hd * 0.12, 2.9 - hd * 0.36),
               (hd * 0.46, hd * 0.34, hd * 0.5), rot=(80, 0, 0),
               params={"segments": 6}, color=SKIN),
            sd("nose", "box", (0, -hd * 0.5, 2.9 - hd * 0.1),
               (0.22, 0.3, 0.3), rot=(-12, 0, 0), color=SKIN),
            *[sd(f"eye_{s}", "box",
                 (sx * hd * 0.19, 0.15 - hd * 0.47, 2.9 + hd * 0.06),
                 (hd * 0.12, hd * 0.06, hd * 0.14), color=EYE)
              for s, sx in (("l", 1), ("r", -1))],
            sd("hair", "icosphere", (0, 0.15 + hd * 0.12, 2.9 + hd * 0.2),
               (hd * 0.96, hd * 0.92, hd * 0.7), params={"subdivisions": 1},
               color=HAIR),
        ],
    )


def make_neck() -> Part:
    return Part(
        name="neck", category="body", tags=["neck"],
        shapes=[sd("neck", "cylinder", (0, 0, 1.2), (2.0, 2.0, 2.6),
                   params={"segments": 8, "taper": 0.82}, color=SKIN)],
    )


def make_torso() -> Part:
    w, d = 7.0, 5.0
    return Part(
        name="torso", category="body", tags=["torso"],
        shapes=[
            sd("belly", "capsule", (0, 0, 2.4), (w * 0.78, d * 0.86, 4.6),
               params={"segments": 8}, color=TUNIC),
            sd("chest", "capsule", (0, -d * 0.12, 5.6), (w * 1.02, d * 1.12, 5.4),
               params={"segments": 8}, color=TUNIC),
            *[sd(f"pec_{s}", "capsule", (sx * w * 0.3, -d * 0.7, 6.1),
                 (w * 0.6, d * 0.42, 0.8), params={"segments": 6}, color=TUNIC)
              for s, sx in (("l", 1), ("r", -1))],
            sd("collar", "cylinder", (0, 0, 7.6), (2.2, 2.2, 1.0),
               params={"segments": 8, "taper": 0.85}, color=TUNIC),
        ],
    )


def make_pelvis() -> Part:
    w, d = 7.0, 5.0
    return Part(
        name="pelvis", category="body", tags=["crotch"],
        shapes=[
            sd("hips", "capsule", (0, 0, 0), (w, d, 3.6),
               params={"segments": 8}, color=TROUSER),
            sd("glutes", "capsule", (0, d * 0.34, 0.2), (w * 0.76, d * 0.62, 3.0),
               params={"segments": 6}, color=TROUSER),
            sd("groin", "wedge", (0, -0.4, -1.7), (w * 0.5, d * 0.5, 2.2),
               rot=(90, 0, 0), color=TROUSER),
            sd("belt", "cylinder", (0, 0, 1.7), (w * 1.04, d * 1.06, 0.9),
               params={"segments": 8}, color=BELT),
        ],
    )


def make_upper_arm() -> Part:
    t = 2.3
    return Part(
        name="upper_arm", category="body", tags=["arm"],
        shapes=[
            sd("shoulder", "icosphere", (0, 0, -0.2), (t * 1.18, t * 1.18, t * 1.18),
               params={"subdivisions": 1}, color=SLEEVE),
            sd("upper_arm", "cylinder", (0, 0, -2.6), (t, t, 4.6),
               params={"segments": 8, "taper": 0.74}, color=SLEEVE),
            sd("biceps", "capsule", (0, -t * 0.5, -1.7), (t * 0.82, t * 0.82, 2.6),
               params={"segments": 6}, color=SLEEVE),
        ],
    )


def make_forearm() -> Part:
    t = 1.9
    return Part(
        name="forearm", category="body", tags=["arm"],
        shapes=[
            sd("elbow", "icosphere", (0, 0, -0.1), (t * 1.08, t * 1.08, t * 1.08),
               params={"subdivisions": 1}, color=SLEEVE),
            sd("flexor", "capsule", (0, -t * 0.12, -1.3), (t * 1.15, t * 1.15, 2.3),
               params={"segments": 6}, color=SKIN),
            sd("forearm", "cylinder", (0, 0, -2.3), (t * 0.92, t * 0.92, 4.2),
               params={"segments": 8, "taper": 0.5}, color=SKIN),  # thin wrist
        ],
    )


def make_hand() -> Part:
    return Part(
        name="hand", category="body", tags=["hand"],
        shapes=[
            sd("palm", "box", (0, 0, -1.1), (1.5, 1.9, 2.2), color=SKIN),
            sd("thumb", "box", (0.9, -0.5, -0.7), (0.75, 0.95, 1.3),
               rot=(0, 20, 24), color=SKIN),
        ],
    )


def make_thigh() -> Part:
    t = 2.9
    return Part(
        name="thigh", category="body", tags=["leg"],
        shapes=[
            sd("hip", "icosphere", (0, 0, -0.2), (t * 1.05, t * 1.05, t * 1.05),
               params={"subdivisions": 1}, color=TROUSER),
            sd("thigh", "cylinder", (0, 0, -3.0), (t, t, 5.8),
               params={"segments": 8, "taper": 0.82}, color=TROUSER),
        ],
    )


def make_shin() -> Part:
    t = 2.3
    return Part(
        name="shin", category="body", tags=["leg"],
        shapes=[
            sd("knee", "icosphere", (0, 0, -0.1), (t * 1.12, t * 1.12, t * 1.12),
               params={"subdivisions": 1}, color=TROUSER),
            sd("calf", "capsule", (0, t * 0.28, -1.9), (t * 1.02, t * 1.18, 3.2),
               params={"segments": 6}, color=BOOT),
            sd("shin", "cylinder", (0, 0, -2.9), (t * 0.9, t * 0.9, 5.6),
               params={"segments": 8, "taper": 0.62}, color=BOOT),
        ],
    )


def make_foot() -> Part:
    return Part(
        name="foot", category="body", tags=["foot"],
        shapes=[
            sd("heel", "box", (0, 0.7, -0.9), (1.7, 1.1, 1.5), color=BOOT),
            sd("foot", "box", (0, -0.4, -1.3), (2.0, 2.6, 1.4), color=BOOT),
            sd("toe", "wedge", (0, -1.8, -1.6), (1.8, 1.4, 1.1),
               rot=(0, 0, -90), color=BOOT),
        ],
    )


PARTS = {
    "body_head": make_head,
    "body_neck": make_neck,
    "body_torso": make_torso,
    "body_pelvis": make_pelvis,
    "body_upper_arm": make_upper_arm,
    "body_forearm": make_forearm,
    "body_hand": make_hand,
    "body_thigh": make_thigh,
    "body_shin": make_shin,
    "body_foot": make_foot,
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stem, factory in PARTS.items():
        part = factory()
        part.validate()
        path = OUT_DIR / f"{stem}.mmpart"
        part.save(path)
        render_part_thumbnail(part, path.with_suffix(".png"))
        print(f"{path.name}: {part.name} ({part.category}), {len(part.shapes)} shapes")


if __name__ == "__main__":
    main()
