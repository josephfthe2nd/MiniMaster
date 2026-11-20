"""Sheets of the kinds that cover the skin (hair, scatter): one row per character, one column per view.

    python tools/surface_view.py OUT.png "ROW|ROW|..." [az,el ...] [--at X,Y,Z | --head] [--r R]
        [--size WxH] [--levels N] [--mono]

A ``ROW`` is appendage specs separated by ``;`` (``hair:style=mohawk``, ``wing:style=feathered;scatter``),
optionally followed by ``@`` and the body's macro values (``hair@gender=0,age=0.3``).  ``--head`` frames the
head of each row's own body (the default frames the trunk); ``--at`` a point of the neutral body (MakeHuman
frame: Y up, +Z front).  ``--mono`` draws everything in the skin's material (shape apart from colour).
It prints, per row, the pieces, the triangles and the seconds the appendages took to build.
"""

import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from minimaster.character import hair as HR  # noqa: E402
from minimaster.character import kinds, mhbase as mh, portrait as po  # noqa: E402
from minimaster.character import weld as W  # noqa: E402
from minimaster.character.sliders import Character, SliderCatalog  # noqa: E402


def main(argv):
    out, rows = argv[0], [r for r in argv[1].split("|") if r]
    views, size, rad, levels, at, head, mono = [], (300, 340), None, 1, None, False, False
    i = 2
    while i < len(argv):
        a = argv[i]
        if a == "--size":
            size = tuple(int(x) for x in argv[i + 1].split("x"))
            i += 2
        elif a == "--r":
            rad = float(argv[i + 1])
            i += 2
        elif a == "--levels":
            levels = int(argv[i + 1])
            i += 2
        elif a == "--at":
            at = np.array([float(x) for x in argv[i + 1].split(",")])
            i += 2
        elif a == "--head":
            head = True
            i += 1
        elif a == "--mono":
            mono = True
            i += 1
        else:
            views.append(tuple(float(x) for x in a.split(",")))
            i += 1
    views = views or ([(0, 5), (50, 10), (100, 8), (180, 8)] if head else [(0, 8), (180, 10), (225, 18)])
    base, lib = mh.load()
    cat = SliderCatalog(lib.names)
    sheet = []
    for row in rows:
        text, _, macro = row.partition("@")
        c = Character()
        if macro:
            c.macro.update({k: float(v) for k, v in (kv.split("=") for kv in macro.split(","))})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for t in (s for s in text.split(";") if s):
                c.appendages.append(kinds.add_from_text(t, c.appendages, base=base, lib=lib))
            verts = lib.apply(base.verts, c.effective_target_weights(cat))
            t0 = time.perf_counter()
            placed = c.instantiate_appendages(base, verts, lib)
            dt = time.perf_counter() - t0
            body_verts = c.shaped_body(base, verts, placed)
            shells = po.character_shells(base, body_verts, levels=levels, with_eyes=True)
            if W.seam(c) == "weld" and any(p.kind.welds(p.params) for p in placed):
                b0 = shells[0]
                shells[0] = po.Shell(W.welded_shell_mesh(b0.mesh, placed), b0.material, b0.ao, b0.ao_strength,
                                     b0.base_colors)
        skin = shells[0].material
        shells += [po.Shell(p.mesh().transform(po.MH_TO_Z_UP), skin if mono else p.material) for p in placed]
        if head:
            h = HR.head_landmarks(base, verts)
            centre, r = np.array([0.0, h["eye"] + 0.05 * h["span"], h["z0"]]), rad or 1.9
        else:
            bv, _ = base.body_cage(verts)
            centre = at if at is not None else np.array([0.0, 0.42 * bv[:, 1].max() + 0.0, -0.1])
            r = rad or 3.6
        fit = (po.MH_TO_Z_UP[:3, :3] @ centre, r)
        sheet.append(np.concatenate([po.render_portrait(shells, size=size, azimuth=az, elevation=el, fit=fit,
                                                        cache=False) for az, el in views], axis=1))
        print(f"{row}: {sum(len(p.parts or [0]) for p in placed)} pieces, "
              f"{sum(len(p.mesh().faces) for p in placed)} triangles, built in {dt:.2f} s")
    po.write_png(out, np.concatenate(sheet, axis=0))
    print("wrote", os.path.basename(out))


if __name__ == "__main__":
    main(sys.argv[1:])
