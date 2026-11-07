"""Numbers for the skin junctions: how smooth is the skin around each appendage root?

    python tools/junction_report.py [--bodies neutral,tall,short,child,heavy] [--levels 1,2]
        [--kinds tail,horn,...]

For every kind x body it builds the appendage, shapes the skin around it
(``junction.shaped_body``) and prints, on the cage and on the Catmull-Clark subdivided body:

* cost (ms) and how many vertices moved / how far / how many were backed off;
* the smallest quad area ratio and the number of flipped quads on the cage;
* ``join``: the mean angle between the skin and the tube wall where they meet, before -> after
  (90 = the skin runs into the tube, 0 = it is tangent to it);
* ``dih``: the 95th-percentile crease between neighbouring quads in the fillet zone, before ->
  after, and ``inc``: the 99th percentile of how much the junction INCREASED a crease;
* ``curv``: the curvature step across the fillet boundary (0 = continuous);
* ``grow`` / ``lip`` / ``xnew``: the validity numbers of ``junction.junction_quality`` -- the largest
  growth of a cage quad's area, the largest difference of two neighbours' displacements per edge
  length, and the new crossings of the skin with itself outside the appendage's own tube (cage /
  body subdivided once / twice);
* ``weld``: the share of the appendage's volume inside the body, before -> after (the printability
  report's root-weld number), and the print report's blocking findings.

Needs the CC0 assets. Run from the repository root.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from minimaster.character import junction as J  # noqa: E402
from minimaster.character import kinds, print_report, printprep  # noqa: E402
from minimaster.character import mhbase as mh  # noqa: E402
from minimaster.character.sliders import Character  # noqa: E402

BODIES = {
    "neutral": {"gender": 1.0, "muscle": 0.55},
    "tall": {"gender": 1.0, "muscle": 0.55, "height": 1.0},
    "short": {"gender": 1.0, "muscle": 0.55, "height": 0.0},
    "child": {"age": 0.1875},
    "heavy": {"weight": 1.0, "gender": 1.0, "muscle": 1.0},
}
KINDS = {
    "tail": "tail",
    "horn": "horn",
    "bat": "wing:style=membrane",
    "angel": "wing:style=feathered",
    "upper_arms": "harvest:preset=upper_arms",
    "lower_arms": "harvest:preset=lower_arms",
}


def weld(base, lib, verts, shaped, placed, size="medium"):
    """The printability report of ``placed`` on the body ``verts`` / the shaped body: ``(report
    before, report after)``."""
    reports = []
    for body_v in (verts, shaped):
        figure = printprep.character_print_shells(base, body_v)
        app = [[(p.name, p.mesh())] for p in placed if p.rig is None]
        wings = [[(f"{p.name}/{n}", m) for n, m in p.rig.piece_meshes(None, **{
            k: v for k, v in p.controls.items() if k != "preset"})] for p in placed if p.rig is not None]
        app += wings
        shells = figure + printprep.appendage_print_shells(app)
        mesh, rep = printprep.assemble_character(shells, size=size,
                                                 reference=printprep.FIGURE_SHELLS)
        mm = printprep.figure_scale(shells, size=size, reference=printprep.FIGURE_SHELLS)
        reports.append(print_report.appendage_report(app, base, verts, mm, figure_shells=figure,
                                                     assembly=rep, body_verts=body_v))
    return reports


def main(argv) -> int:
    bodies, levels, kind_names = list(BODIES), [1, 2], list(KINDS)
    i = 0
    while i < len(argv):
        if argv[i] == "--bodies":
            bodies = argv[i + 1].split(",")
            i += 2
        elif argv[i] == "--levels":
            levels = [int(x) for x in argv[i + 1].split(",")]
            i += 2
        elif argv[i] == "--kinds":
            kind_names = argv[i + 1].split(",")
            i += 2
        else:
            print(__doc__)
            return 2
    base, lib = mh.load()
    print(f"{'kind':11s}{'body':8s}{'ms':>6s}{'moved':>6s}{'maxmv':>6s}{'lim':>4s}{'minA':>6s}"
          f"{'flip':>5s}{'join b->a':>12s}", end="")
    for lv in levels:
        print(f"{'  L%d dih95 b->a  inc99' % lv:>28s}", end="")
    print(f"{'  curv b->a':>14s}{'  grow':>6s}{'  lip':>5s}{'  xnew c/1/2':>12s}{'  weld b->a':>14s}  blocking")
    for kn in kind_names:
        for bn in bodies:
            c = Character()
            c.macro.update(BODIES[bn])
            c.appendages.append(kinds.add_from_text(KINDS[kn], c.appendages, base=base, lib=lib))
            verts = lib.apply(base.verts, c.effective_target_weights(__import__(
                "minimaster.character.sliders", fromlist=["SliderCatalog"]).SliderCatalog(lib.names)))
            placed = c.instantiate_appendages(base, verts, lib)
            specs = J.junction_specs(placed)
            info: dict = {}
            t = []
            for _ in range(5):
                t0 = time.perf_counter()
                shaped = J.apply_junctions(base, verts, specs, info)
                t.append((time.perf_counter() - t0) * 1000)
            m = J.measure_junction(base, verts, shaped, specs[0], levels=1)
            line = (f"{kn:11s}{bn:8s}{min(t):6.1f}{info['moved']:6d}{info['max_move']:6.2f}"
                    f"{info['limited']:4d}{m['min_area_ratio']:6.2f}{m['flipped']:5d}"
                    f"{m.get('join_angle_deg_before', 0):7.0f}->{m.get('join_angle_deg_after', 0):<3.0f}")
            for lv in levels:
                z = J.measure_junction(base, verts, shaped, specs[0], levels=lv)
                line += (f"   {z.get('p95_dihedral_deg_before', 0):5.1f}->"
                         f"{z.get('p95_dihedral_deg_after', 0):5.1f} "
                         f"{z.get('dihedral_increase_p99_deg', 0):5.1f}")
            line += f"  {m['curvature_step_before']:5.2f}->{m['curvature_step_after']:5.2f}"
            qy = J.junction_quality(base, verts, shaped, specs, levels=(1, 2))
            line += (f"  {qy['area_growth_max']:5.2f}  {qy['lipschitz_max']:4.2f}"
                     f"  {qy['selfx_new_visible']:2d}/{qy['L1_selfx_new_visible']:2d}/{qy['L2_selfx_new_visible']:2d}")
            try:
                rb, ra = weld(base, lib, verts, shaped, placed)
                gb = max(g.root_embed for g in rb.groups.values())
                ga = max(g.root_embed for g in ra.groups.values())
                line += f"  {100 * gb:4.0f}%->{100 * ga:3.0f}%  {len(ra.blocking)}"
            except Exception as exc:                                    # a report is a bonus here
                line += f"  (report: {type(exc).__name__}: {exc})"
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
