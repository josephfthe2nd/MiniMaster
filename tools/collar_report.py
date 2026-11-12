"""The numbers of the root collar (``minimaster.character.collar``) on the real body.

    python tools/collar_report.py [--bodies neutral,tall,short,child,heavy] [--levels 1,2]
        [--poses rest,curl,droop,lift,lift70,sway] [--spec tail]

For every body x level x pose (the tail's pose controls) it builds the character, the shaped skin and the
collar and prints, over the part of the collar that can be seen (a collar thinner than 0.04 units is a hair
on the fold side and not counted):

* ``rim``     the angle between the renderer's vertex normal of the collar at its rim and the skin's (p50 / p95);
* ``wall``    the same where the collar meets the tail, against the tail's shaded normal (p50 / p95);
* ``before``  the angle between the skin and the tail wall where they meet WITHOUT the collar (p50): 90 = the skin
              runs into the wall, 0 = tangent -- what the collar replaces;
* ``min h``   the lowest the visible sheet gets over the skin (units, >= -0.002 = never under it);
* ``crease``  the 99th percentile of the dihedral between neighbouring faces of the visible sheet (degrees);
* ``self``    pairs of crossing triangles in the whole shell (the hair-thin fold side has a few);
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
         "lift70": {"lift": 70.0}, "sway": {"sway": 40.0}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bodies", default="neutral,tall,short,child,heavy")
    ap.add_argument("--levels", default="1,2")
    ap.add_argument("--poses", default="rest,curl,droop,lift")
    ap.add_argument("--spec", default="tail", help="the --add text of the appendage (default: tail)")
    args = ap.parse_args(argv)
    base, lib = mh.load()
    cat = SliderCatalog(lib.names)
    print(f"{'body':8s} {'L':>1s} {'pose':7s} {'rim p50/p95':>12s} {'wall p50/p95':>13s} {'before':>7s} "
          f"{'min h':>7s} {'crease':>7s} {'self':>5s} {'seen':>5s} {'ms':>5s}")
    for body in args.bodies.split(","):
        for pose in args.poses.split(","):
            c = Character()
            c.macro.update(BODIES[body])
            c.appendages.append(kinds.add_from_text(args.spec, c.appendages, base=base, lib=lib))
            if POSES[pose]:
                c.appendages[0].controls = dict(POSES[pose])
            verts = lib.apply(base.verts, c.effective_target_weights(cat))
            placed = c.instantiate_appendages(base, verts, lib)
            shaped = c.shaped_body(base, verts, placed)
            for lv in (int(x) for x in args.levels.split(",")):
                skin = CO.skin_from_cage(base, shaped, lv)
                r = CO.join_report(base, shaped, placed[0], None, levels=lv, skin=skin)
                if not r:
                    print(f"{body:8s} {lv:1d} {pose:7s} (no collar)")
                    continue
                print(f"{body:8s} {lv:1d} {pose:7s} {r['rim_p50']:5.1f}/{r['rim_p95']:5.1f}  "
                      f"{r['wall_p50']:5.1f}/{r['wall_p95']:6.1f} {r['before_p50']:7.1f} {r['above_skin_min']:7.4f} "
                      f"{r['crease_p99']:7.1f} {r['self_crossings']:5d} {100 * r['visible_share']:4.0f}% "
                      f"{1000 * r['seconds']:5.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
