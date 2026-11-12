"""Close-ups of appendage roots as the pipeline draws them (round 6: the body's skin welded onto the roots,
``minimaster.character.weld``), with geometry-only modes, for judging the seam as GEOMETRY.

    python tools/weld_view.py OUT.png "<spec>[;<spec>]" FOCUS [az,el ...] [--levels N] [--macro k=v,...]
        [--pose ID:control=value] [--seam weld|none|collar] [--mono] [--flat] [--size WxH] [--r R]

``FOCUS`` is the start of an appendage name (``tail``, ``wing``, ``horn``, ``arm``, ``leg``) or ``all``.
``--seam`` overrides the document's (``none``: the bare junction, ``collar``: the round-4 overlay shell);
``--mono`` draws every shell in the skin's material (colour apart from geometry), ``--flat`` every triangle
flat-shaded in one material (what a slicer or an STL viewer shows).  It prints, per root, whether the weld was
applied, the guard's rounds and the seconds it took.
"""

import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from minimaster.character import collar as CO  # noqa: E402
from minimaster.character import kinds, mhbase as mh, portrait as po  # noqa: E402
from minimaster.character import weld as W  # noqa: E402
from minimaster.character.sliders import Character, SliderCatalog  # noqa: E402
from minimaster.core.mesh import Mesh  # noqa: E402


def main(argv):
    out, specs, focus = argv[0], [s for s in argv[1].split(";") if s], argv[2]
    views, macro, size, rad, levels, seam = [], {"gender": 1.0, "muscle": 0.55}, (380, 380), None, 1, None
    mono = flat = False
    poses = []
    i = 3
    while i < len(argv):
        a = argv[i]
        if a == "--macro":
            macro = {k: float(v) for k, v in (kv.split("=") for kv in argv[i + 1].split(","))}
            i += 2
        elif a == "--size":
            size = tuple(int(x) for x in argv[i + 1].split("x"))
            i += 2
        elif a == "--r":
            rad = float(argv[i + 1])
            i += 2
        elif a == "--levels":
            levels = int(argv[i + 1])
            i += 2
        elif a == "--pose":
            poses.append(argv[i + 1])
            i += 2
        elif a == "--seam":
            seam = argv[i + 1]
            i += 2
        elif a == "--mono":
            mono = True
            i += 1
        elif a == "--flat":
            flat = True
            i += 1
        else:
            views.append(tuple(float(x) for x in a.split(",")))
            i += 1
    views = views or [(180, 15), (130, 20), (90, 10), (225, 25)]
    base, lib = mh.load()
    cat = SliderCatalog(lib.names)
    c = Character()
    c.macro.update(macro)
    if seam == "none":
        c.anatomy["collar"] = False
    elif seam == "collar":
        c.anatomy["collar"] = "shell"
    notes = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for t in specs:
            c.appendages.append(kinds.add_from_text(t, c.appendages, base=base, lib=lib))
        for t in poses:
            kinds.apply_pose(*kinds.pose_from_text(t, c.appendages))
        verts = lib.apply(base.verts, c.effective_target_weights(cat))
        placed = c.instantiate_appendages(base, verts, lib) if c.appendages else []
        body_verts = c.shaped_body(base, verts, placed) if placed else verts
        body = po.character_shells(base, body_verts, levels=levels, with_eyes=True)
        how = W.seam(c)
        diags: list = []
        t0 = time.perf_counter()
        if how == "weld" and placed:
            b0 = body[0]
            body[0] = po.Shell(W.welded_shell_mesh(b0.mesh, placed, diags=diags), b0.material, b0.ao,
                               b0.ao_strength, b0.base_colors)
        dt = time.perf_counter() - t0
        shells = list(body) + [po.Shell(p.mesh().transform(po.MH_TO_Z_UP), p.material) for p in placed]
        if how == "collar" and placed:
            for _, _, m, mat in CO.collar_shells(base, body_verts, placed, levels=levels,
                                                 skin=CO.skin_from_shell(body[0].mesh)):
                m = m.transform(po.MH_TO_Z_UP)
                shells.append(po.Shell(Mesh(np.asarray(m.vertices), np.asarray(m.faces)), mat))
        notes.append(f"seam={how} weld_s={dt:.2f}")
        for d in diags:
            notes.append(f"{d.get('name')}:applied={d.get('applied')},guard={d.get('guard')},"
                         f"x={d.get('crossings')},s={d.get('seconds', 0.0):.2f}")
        notes += [str(w.message)[:160] for w in caught]
    skin_mat = body[0].material
    if flat:
        new = []
        for s in shells:
            v, f = np.asarray(s.mesh.vertices), np.asarray(s.mesh.faces)
            vv = v[f].reshape(-1, 3)
            new.append(po.Shell(Mesh(vv, np.arange(len(vv)).reshape(-1, 3)), skin_mat, ao=False))
        shells = new
    elif mono:
        shells = [po.Shell(s.mesh, skin_mat, s.ao, s.ao_strength, None) for s in shells]
    sel = [p for p in placed if focus == "all" or p.name.startswith(focus)]
    if sel:
        ctr = np.vstack([p.anchor[None, :] for p in sel]).mean(axis=0)
        ctr_z = (po.MH_TO_Z_UP[:3, :3] @ ctr) + po.MH_TO_Z_UP[:3, 3]
    else:
        lo, hi = body[0].mesh.bounds
        ctr_z = (lo + hi) / 2
    fit = (ctr_z, rad or 3.0)
    rows = [po.render_portrait(shells, size=size, azimuth=az, elevation=el, fit=fit, cache=False) for az, el in views]
    po.write_png(out, np.concatenate(rows, axis=1))
    print("wrote", os.path.basename(out), [p.name for p in placed], *notes)


if __name__ == "__main__":
    main(sys.argv[1:])
