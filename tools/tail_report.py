"""Numbers for the tail's anatomy and its junction, on several bodies (docs/anatomy_tail.md).

    python tools/tail_report.py [--bodies neutral,tall,short,child,heavy,female] [--levels 1]

For every body: the placement (the root point's height over the body's, its distance to the landmarks of the
sacrum in base radii R = 0.0265 H, the axis' elevation, the lateral offset of the root and of the axis),
the junction (vertices moved, the largest move in R, the smallest quad area ratio, flipped quads, the angle at
which the skin meets the tail before -> after, cage vertices that cross the midline OUTSIDE the tail), the time
of ``apply_junctions`` and of building the tail, and the printability numbers of the shell. Needs the CC0
assets like every character command. Run from the repository root.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from minimaster.character import junction as J  # noqa: E402
from minimaster.character import kinds  # noqa: E402
from minimaster.character import mhbase as mh  # noqa: E402
from minimaster.character.sliders import Character, SliderCatalog  # noqa: E402

BODIES = {
    "neutral": {"gender": 1.0, "muscle": 0.55}, "tall": {"gender": 1.0, "muscle": 0.55, "height": 1.0},
    "short": {"gender": 1.0, "muscle": 0.55, "height": 0.0}, "child": {"age": 0.1875},
    "heavy": {"weight": 1.0, "gender": 1.0, "muscle": 1.0}, "female": {"gender": 0.0},
    "old": {"age": 1.0},
}
APEX, CLEFT_TOP, EDGE_A, EDGE_B = 4223, 4387, 4215, 4205


def report(names, levels=1) -> list[dict]:
    base, lib = mh.load()
    cat = SliderCatalog(lib.names)
    rows = []
    for name in names:
        c = Character()
        c.macro.update(BODIES[name])
        c.appendages.append(kinds.add_from_text("tail", c.appendages, base=base, lib=lib))
        verts = lib.apply(base.verts, c.effective_target_weights(cat))
        t0 = time.perf_counter()
        placed = c.instantiate_appendages(base, verts, lib)
        t_build = time.perf_counter() - t0
        p = placed[0]
        cv, cq = base.body_cage(verts)
        H = float(np.ptp(cv[:, 1]))
        R = 0.0265 * H
        P = 0.5 * (cv[EDGE_A] + cv[EDGE_B])
        axis = p.matrix[:3, 2] / np.linalg.norm(p.matrix[:3, 2])
        spec = p.junction()
        t0 = time.perf_counter()
        shaped = c.shaped_body(base, verts, placed)
        t_junc = time.perf_counter() - t0
        cv1, _ = base.body_cage(shaped)
        mag = np.linalg.norm(cv1 - cv, axis=1)
        m = J.measure_junction(base, verts, shaped, spec, levels=levels)
        skin = p.mesh()
        rows.append(dict(
            body=name, H=H, R=R, tail_R=p.params["base_radius"] * p.model.scale,
            y_over_H=(P[1] - cv[:, 1].min()) / H,
            arc_R=(np.linalg.norm(cv[EDGE_A] - cv[APEX]) + 0.5 * np.linalg.norm(cv[EDGE_B] - cv[EDGE_A])) / R,
            top_R=np.linalg.norm(cv[CLEFT_TOP] - P) / R,
            elevation=math.degrees(math.asin(axis[1])), lateral_R=abs(p.anchor[0]) / R,
            moved=int((mag > 0).sum()), max_R=float(mag.max() / R), min_area=m["min_area_ratio"],
            flipped=m["flipped"], join_before=m.get("join_angle_deg_before", float("nan")),
            join_after=m.get("join_angle_deg_after", float("nan")),
            cross=int(((cv[:, 0] * cv1[:, 0]) < -1e-12).sum()), ms_junction=1000 * t_junc,
            ms_build=1000 * t_build, volume=skin.volume(), verts=len(skin.vertices)))
    return rows


def main(argv) -> int:
    names, levels = list(BODIES)[:6], 1
    i = 0
    while i < len(argv):
        if argv[i] == "--bodies":
            names, i = argv[i + 1].split(","), i + 2
        elif argv[i] == "--levels":
            levels, i = int(argv[i + 1]), i + 2
        else:
            print(__doc__)
            return 2
    rows = report(names, levels)
    print(f"{'body':8s} {'H':>5s} {'R':>5s} {'y/H':>5s} {'apex':>5s} {'top':>5s} {'elev':>5s} {'lat':>6s} | "
          f"{'moved':>5s} {'max/R':>5s} {'minA':>5s} {'flip':>4s} {'join b->a':>11s} {'cross':>5s} {'ms j':>5s} {'ms b':>5s}")
    for r in rows:
        print(f"{r['body']:8s} {r['H']:5.2f} {r['R']:5.3f} {r['y_over_H']:5.3f} {r['arc_R']:5.2f} {r['top_R']:5.2f} "
              f"{r['elevation']:5.1f} {r['lateral_R']:6.4f} | {r['moved']:5d} {r['max_R']:5.2f} {r['min_area']:5.2f} "
              f"{r['flipped']:4d} {r['join_before']:4.0f}->{r['join_after']:3.0f}   {r['cross']:5d} "
              f"{r['ms_junction']:5.1f} {r['ms_build']:5.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
