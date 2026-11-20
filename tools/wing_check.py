"""Three checks of the feathered wing as GEOMETRY, for comparing it with reference pictures.

    python tools/wing_check.py plain OUT.png [thickness=0.16 primaries=8 ...]
        the wing alone in one plain material, from above and edge-on.  ``tools/wing_view.py`` paints
        every feather darker toward its edges so the rows read apart; a step seen there may be paint.
        Here a step is a step.

    python tools/wing_check.py section [MM]
        the mid-length cross-section of the wing's own primary, secondary and covert, each laid flat:
        its width and how thick it is (median, at the shaft, near the edge), in mm on a figure MM tall
        (default 32).

    python tools/wing_check.py report "wing:style=feathered,thickness=0.16" [MM]
        the print check of one appendage on the default body at that height: passes or not, the
        warnings, and how much of its area is under 0.2 mm and under 0.3 mm.

The numbers in ``docs/appendage_references.md`` (Feathered wing and feathers) come from these.
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

UNIT_HEIGHT = 17.5       # the body is this many units tall; a figure MM tall is MM / 17.5 mm a unit
INTS = ("primaries", "secondaries", "tertials", "scapulars")


def plain(out, args):
    import wing_view as WV
    from minimaster.character import portrait as po

    kw = {}
    for a in args:
        k, v = a.split("=")
        kw[k] = int(v) if k in INTS else float(v)
    rig = WV.make_rig("feathered", **kw)
    mesh = rig.pose_mesh({}).transform(WV.WING_TO_WORLD)
    # framed on the default wing, so two pictures with different arguments line up
    centre, radius = WV.frame([WV.make_rig("feathered").pose_mesh({}).transform(WV.WING_TO_WORLD)])
    shots = [po.render_portrait([po.Shell(mesh, WV.MATS["feathered"])], size=(1500, h), azimuth=0.0,
                                elevation=el, fit=(centre, radius * 0.62), cache=False)
             for el, h in ((88.0, 760), (6.0, 300))]
    po.write_png(out, np.concatenate(shots))
    print(f"wrote {os.path.basename(out)}: {len(rig.pieces)} pieces, {len(mesh.faces)} triangles")


def section(mm=32.0):
    import feather_view as FV
    from minimaster.character import wings as W

    rng = np.random.default_rng(0)
    scale = mm / UNIT_HEIGHT
    for name, (a, k) in (("primary", FV.PRIMARY), ("secondary", FV.SECONDARY), ("covert", FV.COVERT)):
        k = {**k, "curl": 0.0, "cup": 0.0, "sweep": 0.0}          # flat: the section is the feather's own
        k.pop("splits", None)
        m = W._blade(*a, **k)
        tri = np.asarray(m.vertices)[np.asarray(m.faces)]
        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        i = rng.choice(len(tri), 600000, p=area / area.sum())     # points spread evenly over the surface
        r, s = np.sqrt(rng.random(len(i))), rng.random(len(i))
        p = tri[i, 0] * (1 - r)[:, None] + tri[i, 1] * (r * (1 - s))[:, None] + tri[i, 2] * (r * s)[:, None]
        cut = p[np.abs(p[:, 0] - 0.5 * a[0]) < 0.01 * a[0]]       # +X along the feather, +Y up, +Z across
        width = np.ptp(cut[:, 2])
        at = cut[:, 2].min() + width * np.linspace(0.03, 0.97, 33)
        th = np.array([np.ptp(cut[np.abs(cut[:, 2] - c) < 0.02 * width, 1]) for c in at])
        edge = min(th[1], th[-2])
        print(f"{name}: {width * scale:.2f} mm wide; thick {np.median(th) * scale:.2f} mm (median), "
              f"{th.max() * scale:.2f} mm at the shaft, {edge * scale:.2f} mm near the edge "
              f"= {np.median(th) / width:.2f} / {th.max() / width:.2f} / {edge / width:.2f} of its width")


def report(text, mm=32.0):
    from minimaster.character import kinds as K, mhbase as mh, print_report as pr
    from minimaster.character.sliders import Character, SliderCatalog

    warnings.simplefilter("ignore")
    base, lib = mh.load()
    c = Character()
    c.macro.update(gender=1.0, muscle=0.55)
    c.appendages.append(K.add_from_text(text, c.appendages, base=base, lib=lib))
    v = lib.apply(base.verts, c.effective_target_weights(SliderCatalog(lib.names)))
    placed = c.instantiate_appendages(base, v, lib)
    rep = pr.appendage_report([p.shells() for p in placed], base, v, mm / UNIT_HEIGHT,
                              body_verts=c.shaped_body(base, v, placed))
    g = next(iter(rep.groups.values()))
    found = sorted({f"{f.level}:{f.code}" for f in rep.findings}) or ["nothing"]
    print(f"{text} at {mm:g} mm: {'passes' if rep.ok else 'BLOCKED'}; findings {', '.join(found)}; "
          f"{g.thin_below_min:.1%} of the area under 0.2 mm, {g.thin_below_thin:.1%} under 0.3 mm")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "plain":
        plain(sys.argv[2], sys.argv[3:])
    elif what == "section":
        section(float(sys.argv[2]) if len(sys.argv) > 2 else 32.0)
    elif what == "report":
        report(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 32.0)
    else:
        print(__doc__)
