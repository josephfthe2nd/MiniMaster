"""Render an evaluation contact sheet: all templates (front + back) and all
shipped parts, for the iterative art-improvement loop.

Usage: python tools/eval_sheet.py OUT.png [--back] [--parts]
"""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minimaster.parts import Part, part_preview_scene  # noqa: E402
from minimaster.render import render_scene, write_png  # noqa: E402
from minimaster.templates import list_templates, load_template  # noqa: E402

TEMPLATE_ORDER = ["human_fighter", "dwarf", "goblin", "orc", "skeleton"]
PART_ORDER = ["eye", "horn", "ear", "spike", "claw", "wing", "tail",
              "sword", "shield", "axe", "club", "dagger"]


def render_row(scenes, size, azimuth, elevation=16):
    return np.hstack([
        render_scene(s, pose_name="__active__", size=(size, size),
                     azimuth=azimuth, elevation=elevation)
        for s in scenes
    ])


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "eval.png"
    flags = set(sys.argv[2:])
    size = 400
    scenes = [load_template(n) for n in TEMPLATE_ORDER if n in list_templates()]
    rows = [render_row(scenes, size, azimuth=335)]
    if "--back" in flags:
        rows.append(render_row(scenes, size, azimuth=155))
    if "--parts" in flags:
        parts_dir = Path(__file__).resolve().parent.parent / "minimaster" / "parts"
        imgs = []
        for name in PART_ORDER:
            path = parts_dir / f"{name}.mmpart"
            if not path.is_file():
                continue
            scene = part_preview_scene(Part.load(path))
            imgs.append(render_scene(scene, pose_name=None, with_base=False,
                                     size=(size // 2, size // 2), azimuth=335,
                                     elevation=14))
        half = (len(imgs) + 1) // 2
        row1 = np.hstack(imgs[:half])
        row2 = np.hstack(imgs[half:])
        if row2.shape[1] < row1.shape[1]:
            pad = np.zeros((row2.shape[0], row1.shape[1] - row2.shape[1], 4),
                           dtype=np.uint8)
            pad[..., :3] = row2[0, 0, :3]
            pad[..., 3] = 255
            row2 = np.hstack([row2, pad])
        rows.append(np.vstack([row1, row2]))
    width = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < width:
            pad = np.zeros((r.shape[0], width - r.shape[1], 4), dtype=np.uint8)
            pad[..., :3] = r[0, 0, :3]
            pad[..., 3] = 255
            r = np.hstack([r, pad])
        padded.append(r)
    write_png(out, np.vstack(padded))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
