"""The numbers of the root collar (``minimaster.character.collar``) on the real body.

    python tools/collar_report.py [--bodies neutral,tall,short,child,heavy,slim_f] [--levels 1,2]
        [--poses rest,curl,droop,lift,lift70,sway,low60,lift90,swing-45,raise60] [--spec tail]

For every body x level x pose (the appendage's pose controls) it builds the character, the shaped skin and the
collar and prints, over the part of the collar that can be seen (a column whose visible reach is under 0.04
units is not counted):

* ``rim`` / ``wall``  the angle between the normal the RENDERER draws the collar with (the skin's own where it
              meets the skin, the wall's where it meets the appendage: ``collar.shading_transfer``) and the skin's /
              the appendage's shaded normal, p95 (0: no edge in the shading);
* ``geo``     the same with the collar's OWN geometric normals, rim / wall p95: the angles the sheet crosses the skin
              (``GAMMA_SKIN_DEG``) and enters the wall (``ALPHA_WALL_DEG``, less where the profile is flatter);
* ``before``  the angle between the skin and the wall where they meet WITHOUT the collar (p50);
* ``crease``  the 99th percentile of the dihedral between neighbouring faces of the visible sheet (degrees);
* ``wwave``   the wobble of the line where the collar enters the wall (root radii, ``collar.contact_lines``);
* ``fold``    columns whose sheet faces away from the skin where it meets it; ``self`` pairs of crossing triangles;
* ``kept``    the share of the ring that has a collar (the rest: the fold side, a leaning wall, a cut column);
* ``ms``      the time of ``collar_pieces`` with the body mesh given (what a pipeline pays per root).

Needs the CC0 assets like every character command. Run from the repository root.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimaster.character import collar as CO  # noqa: E402
from minimaster.character import kinds  # noqa: E402
from minimaster.character import mhbase as mh  # noqa: E402
from minimaster.character.sliders import Character, SliderCatalog  # noqa: E402

BODIES = {
    "neutral": {"gender": 1.0, "muscle": 0.55},
    "tall": {"gender": 1.0, "muscle": 0.55, "height": 1.0},
    "short": {"gender": 1.0, "muscle": 0.55, "height": 0.0},
    "child": {"age": 0.1875},
    "heavy": {"weight": 1.0, "gender": 1.0, "muscle": 1.0},
    "slim_f": {"weight": 0.0, "gender": 0.0},
}
POSES = {"rest": {}, "curl": {"curl": 1.2}, "droop": {"droop": 0.9}, "lift": {"lift": 40.0},
         "lift70": {"lift": 70.0}, "sway": {"sway": 40.0}, "low60": {"lift": -60.0}, "lift90": {"lift": 90.0},
         "swing-45": {"swing": -45.0}, "raise60": {"raise_": 60.0}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bodies", default="neutral,tall,short,child,heavy")
    ap.add_argument("--levels", default="1,2")
    ap.add_argument("--poses", default="rest,curl,droop,lift")
    ap.add_argument("--spec", default="tail", help="the --add text of the appendage (default: tail)")
    args = ap.parse_args(argv)
    base, lib = mh.load()
    cat = SliderCatalog(lib.names)
    print(f"{'body':8s} {'L':>1s} {'pose':8s} {'rim/wall':>9s} {'geo r/w':>11s} {'before':>7s} "
          f"{'crease':>7s} {'wwave':>6s} {'fold':>4s} {'self':>5s} {'kept':>5s} {'ms':>5s}")
    for body in args.bodies.split(","):
        for pose in args.poses.split(","):
            c = Character()
            c.macro.update(BODIES[body])
            c.appendages.append(kinds.add_from_text(args.spec, c.appendages, base=base, lib=lib))
            if POSES[pose]:
                keys = {cs.key for cs in kinds.kind_of(c.appendages[0]).controls}
                ctl = {k: v for k, v in POSES[pose].items() if k in keys}
                if not ctl:
                    continue                                   # a pose this kind does not have
                c.appendages[0].controls = ctl
            verts = lib.apply(base.verts, c.effective_target_weights(cat))
            placed = c.instantiate_appendages(base, verts, lib)
            shaped = c.shaped_body(base, verts, placed)
            for lv in (int(x) for x in args.levels.split(",")):
                skin = CO.skin_from_cage(base, shaped, lv)
                r = CO.join_report(base, shaped, placed[0], None, levels=lv, skin=skin)
                if not r:
                    print(f"{body:8s} {lv:1d} {pose:8s} (no collar)")
                    continue
                print(f"{body:8s} {lv:1d} {pose:8s} {r['rim_p95']:4.1f}/{r['wall_p95']:4.1f} "
                      f"{r['geo_rim_p95']:5.1f}/{r['geo_wall_p95']:5.1f} {r['before_p50']:7.1f} "
                      f"{r['crease_p99']:7.1f} {r['wall_wave']:6.3f} {r['fold_columns']:4d} {r['self_crossings']:5d} "
                      f"{100 * r['kept_share']:4.0f}% {1000 * r['seconds']:5.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
