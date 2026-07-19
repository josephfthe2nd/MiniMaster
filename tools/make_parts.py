"""Generate the shipped starter part library in minimaster/parts/.

Run from the repo root:  python tools/make_parts.py

Parts follow the attachment convention: origin (0,0,0) is the point that
lands on the clicked surface, +Z points outward along the surface normal.
Sizes suit the ~30 mm template figures. Gear is authored grip-at-origin with
the business end along +Z, so placing it in an open hand with the normal
pointing up "just works".

A test byte-locks the .mmpart JSON to this generator (thumbnails are not
byte-locked; PNG bytes vary by zlib).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minimaster.parts import Part, render_part_thumbnail  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "minimaster" / "parts"

BONE_COL = "#e8ddc4"
METAL = "#b9c2c9"
WOOD = "#6b543a"
LEATHER = "#7d6a4d"
DARK = "#3a3733"
FLESH = "#b08d57"


def sd(name, kind, pos, scale, rot=(0.0, 0.0, 0.0), params=None, bone=None,
       color=FLESH):
    return {
        "name": name,
        "kind": kind,
        "params": dict(params or {}),
        "position": [float(v) for v in pos],
        "rotation": [float(v) for v in rot],
        "scale": [float(v) for v in scale],
        "bone": bone,
        "color": color,
    }


def jd(name, parent, pos):
    return {"name": name, "parent": parent, "position": [float(v) for v in pos]}


def make_eye() -> Part:
    return Part(
        name="eye",
        category="face",
        shapes=[
            sd("ball", "icosphere", (0, 0, 0.45), (1.8, 1.8, 1.8),
               params={"subdivisions": 1}, color="#ece7da"),
            sd("pupil", "icosphere", (0, 0, 1.2), (0.8, 0.8, 0.8),
               params={"subdivisions": 0}, color=DARK),
        ],
    )


def make_horn() -> Part:
    return Part(
        name="horn",
        category="head",
        shapes=[
            sd("base", "cylinder", (0, 0, 1.3), (1.7, 1.7, 2.8),
               params={"segments": 5, "taper": 0.55}, color=BONE_COL),
            sd("tip", "cylinder", (0, -0.5, 3.5), (1.0, 1.0, 2.6),
               rot=(18, 0, 0), params={"segments": 5, "taper": 0.12},
               color=BONE_COL),
        ],
    )


def make_ear() -> Part:
    return Part(
        name="ear",
        category="face",
        shapes=[
            sd("flap", "cylinder", (0, 0, 1.4), (1.7, 0.9, 3.2),
               rot=(-10, 0, 0), params={"segments": 3, "taper": 0.08}),
        ],
    )


def make_spike() -> Part:
    return Part(
        name="spike",
        category="body",
        shapes=[
            sd("cone", "cylinder", (0, 0, 1.6), (1.9, 1.9, 3.4),
               params={"segments": 6, "taper": 0.0}, color=BONE_COL),
        ],
    )


def make_claw() -> Part:
    return Part(
        name="claw",
        category="body",
        shapes=[
            sd("knuckle", "cylinder", (0, 0, 0.9), (1.5, 1.5, 2.1),
               params={"segments": 4, "taper": 0.55}, color=BONE_COL),
            sd("talon", "cylinder", (0, -0.55, 2.7), (0.9, 0.9, 2.3),
               rot=(24, 0, 0), params={"segments": 4, "taper": 0.1},
               color=BONE_COL),
        ],
    )


def make_wing() -> Part:
    return Part(
        name="wing",
        category="body",
        shapes=[
            sd("membrane", "wedge", (3.4, 0, 3.6), (8.4, 0.7, 6.4),
               rot=(0, 148, 0), color="#8a7256"),
            sd("spar", "cylinder", (2.6, 0, 4.6), (1.0, 1.0, 9.6),
               rot=(0, 38, 0), params={"segments": 5, "taper": 0.55},
               color="#6b543a"),
            sd("shoulder", "icosphere", (0, 0, 0.7), (2.0, 2.0, 2.0),
               params={"subdivisions": 0}, color="#6b543a"),
        ],
    )


def make_tail() -> Part:
    return Part(
        name="tail",
        category="body",
        tags=["posable"],
        joints=[
            jd("root", None, (0, 0, 0)),
            jd("mid", "root", (0, 0, 3.8)),
            jd("tip", "mid", (0, 0, 7.4)),
        ],
        shapes=[
            sd("seg1", "capsule", (0, 0, 2.0), (2.6, 2.6, 5.0),
               params={"segments": 6}, bone="mid"),
            sd("seg2", "capsule", (0, 0, 5.6), (1.9, 1.9, 4.6),
               params={"segments": 6}, bone="tip"),
            sd("seg3", "cylinder", (0, 0, 8.6), (1.3, 1.3, 3.4),
               params={"segments": 5, "taper": 0.1}, bone="tip"),
        ],
    )


def make_sword() -> Part:
    return Part(
        name="sword",
        category="gear",
        shapes=[
            sd("blade", "box", (0, 0, 7.8), (1.35, 0.42, 12.6), color=METAL),
            sd("point", "cylinder", (0, 0, 14.85), (1.35, 0.42, 1.5),
               params={"segments": 4, "taper": 0.0}, color=METAL),
            sd("guard", "box", (0, 0, 1.35), (3.3, 0.9, 0.66), color=WOOD),
            sd("grip", "cylinder", (0, 0, -1.35), (0.9, 0.9, 2.7),
               params={"segments": 6}, color=WOOD),
            sd("pommel", "icosphere", (0, 0, -3.0), (1.35, 1.35, 1.35),
               params={"subdivisions": 0}, color=METAL),
        ],
    )


def make_shield() -> Part:
    return Part(
        name="shield",
        category="gear",
        shapes=[
            sd("disc", "cylinder", (0, 0, 0.45), (10.2, 10.2, 0.85),
               params={"segments": 12}, color=LEATHER),
            sd("boss", "cylinder", (0, 0, 1.0), (3.0, 3.0, 1.05),
               params={"segments": 8, "taper": 0.55}, color=METAL),
        ],
    )


def make_axe() -> Part:
    return Part(
        name="axe",
        category="gear",
        shapes=[
            sd("haft", "cylinder", (0, 0, 4.2), (1.05, 1.05, 15.0),
               params={"segments": 6}, color=WOOD),
            sd("head", "wedge", (0, -2.25, 6.0), (4.8, 1.2, 4.2),
               rot=(0, 0, -90), color=METAL),
        ],
    )


def make_club() -> Part:
    return Part(
        name="club",
        category="gear",
        shapes=[
            sd("shaft", "cylinder", (0, 0, 5.1), (3.0, 3.0, 13.8),
               rot=(180, 0, 0), params={"segments": 6, "taper": 0.45},
               color="#7b5c39"),
            sd("knob", "icosphere", (0, 0, 11.4), (3.6, 3.6, 3.6),
               params={"subdivisions": 0}, color="#7b5c39"),
        ],
    )


def make_dagger() -> Part:
    return Part(
        name="dagger",
        category="gear",
        shapes=[
            sd("blade", "cylinder", (0, 0, 3.9), (1.5, 0.6, 6.6),
               params={"segments": 4, "taper": 0.08}, color=METAL),
            sd("guard", "box", (0, 0, 1.2), (2.7, 0.84, 0.6), color="#5d4a33"),
        ],
    )


PARTS = {
    "eye": make_eye,
    "horn": make_horn,
    "ear": make_ear,
    "spike": make_spike,
    "claw": make_claw,
    "wing": make_wing,
    "tail": make_tail,
    "sword": make_sword,
    "shield": make_shield,
    "axe": make_axe,
    "club": make_club,
    "dagger": make_dagger,
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, factory in PARTS.items():
        part = factory()
        part.validate()
        path = OUT_DIR / f"{name}.mmpart"
        part.save(path)
        render_part_thumbnail(part, path.with_suffix(".png"))
        posable = " (posable)" if part.joints else ""
        print(f"{path.name}: {part.category}, {len(part.shapes)} shapes"
              f"{posable}")


if __name__ == "__main__":
    main()
