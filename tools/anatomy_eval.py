"""Render an anatomy evaluation sheet for the iterative body-improvement loop.

Usage: python tools/anatomy_eval.py OUT.png

Top row: a BARE male humanoid (no gear) from front / 3-4 / side / back — the
view that exposes proportion and muscle problems. Bottom row: the five shipped
templates (front) so roster-wide regressions show up.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minimaster.render import render_scene, write_png  # noqa: E402
from minimaster.templates import list_templates, load_template  # noqa: E402
from tools.make_templates import build_humanoid, humanoid_poses  # noqa: E402


def bare_male():
    s = build_humanoid("bare", height=30.0, head=1.05,
                       skin="#b78a5c", torso_color="#7a5c3d",
                       limb_color="#8a6a44", boot_color="#5a4630")
    s.poses = humanoid_poses()
    s.active_pose = "idle"
    return s


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "anatomy_eval.png"
    s = bare_male()
    top = [render_scene(s, pose_name="idle", size=(360, 360), azimuth=az,
                        elevation=8)
           for az in (0, 320, 270, 180)]
    order = ["human_fighter", "dwarf", "goblin", "orc", "skeleton"]
    have = list_templates()
    bottom = [render_scene(load_template(n), pose_name="__active__",
                           size=(288, 360), azimuth=335, elevation=14)
              for n in order if n in have]
    top_row = np.hstack(top)
    bot_row = np.hstack(bottom)
    w = max(top_row.shape[1], bot_row.shape[1])

    def pad(row):
        if row.shape[1] < w:
            p = np.zeros((row.shape[0], w - row.shape[1], 4), dtype=np.uint8)
            p[..., :3] = row[0, 0, :3]
            p[..., 3] = 255
            row = np.hstack([row, p])
        return row

    write_png(out, np.vstack([pad(top_row), pad(bot_row)]))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
