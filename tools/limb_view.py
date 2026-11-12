"""Close-ups of an extra limb and its girdle, with the anatomical layers (ecorche).

    python tools/limb_view.py OUT.png "harvest:preset=lower_arms" [FOCUS] [az,el ...]
        [--layers bone,muscle,skin] [--cut coronal|sagittal|none] [--macro neutral|tall|short|child|heavy|slim_f]
        [--pose arm-1:raise_=40,swing=10] [--r 3.0] [--size 420x520] [--levels 1] [--no-junction] [--no-body]

``--layers`` picks what is drawn of every limb (``skin`` is the shell that is printed, ``muscle`` the
masses that shape it, ``bone`` the chain inside); ``--cut`` slices the body and the outer layers away
on one side of the plane through the girdle's pivot (``coronal``: the front half goes, look from the
front; ``sagittal``: the lateral half goes, look from the side) so that the layers inside can be seen.
Needs the CC0 assets. Run from the repository root. Draws with ``minimaster.character.ecorche`` (the
shared helper; the copy this tool carried until round 2b is gone).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import junction_view as jv  # noqa: E402
from minimaster.character import kinds, mhbase as mh, portrait as po  # noqa: E402
from minimaster.character.sliders import SliderCatalog  # noqa: E402

LAYERS = ("bone", "muscle", "skin")


def parse_layers(text) -> tuple:
    names = [t.strip().lower() for t in (text.split(",") if isinstance(text, str) else text) if t]
    for n in names:
        if n not in LAYERS:
            raise ValueError(f"unknown layer {n!r}; the layers are {', '.join(LAYERS)}")
    return tuple(n for n in LAYERS if n in names)


def layer_shells(base, verts, placed, layers=LAYERS, *, cut: str = "none", plane=None, levels: int = 1,
                 body: bool = True):
    """Render shells (Z up): the body and the ``layers`` of every appendage in ``placed``, drawn by the
    shared :mod:`minimaster.character.ecorche` (round 2b: the tool's own copy of it is gone)."""
    from minimaster.character import ecorche
    if cut != "none" and plane is None:
        plane = _plane(placed, cut)
    return ecorche.layer_shells(base, verts, placed, parse_layers(layers), cut=cut != "none", plane=plane,
                                levels=levels, body=body)


def _plane(placed, cut: str):
    """The cutting plane through the pivot of the first limb: ``(point, normal)`` in the MakeHuman frame."""
    p0 = placed[0]
    pivot = np.asarray(p0.limb.joints(p0.controls)["shoulder"])
    normal = {"coronal": (0.0, 0.0, 1.0), "sagittal": (1.0 if pivot[0] >= 0 else -1.0, 0.0, 0.0)}[cut]
    return pivot, np.asarray(normal)


def main(argv) -> int:
    args = list(argv)
    if len(args) < 2:
        print(__doc__)
        return 2
    out, specs = args[0], [s for s in args[1].split(";") if s]
    focus, views, macro, size, rad, levels = "arm", [], jv.BODIES["neutral"], (420, 520), 3.0, 1
    layers, cut, pose, junction, with_body = LAYERS, "none", [], True, True
    i = 2
    while i < len(args):
        a = args[i]
        if a == "--layers":
            layers = parse_layers(args[i + 1]); i += 2
        elif a == "--cut":
            cut = args[i + 1]; i += 2
        elif a == "--macro":
            macro = jv.BODIES[args[i + 1]]; i += 2
        elif a == "--size":
            size = tuple(int(x) for x in args[i + 1].split("x")); i += 2
        elif a == "--r":
            rad = float(args[i + 1]); i += 2
        elif a == "--levels":
            levels = int(args[i + 1]); i += 2
        elif a == "--pose":
            pose.append(args[i + 1]); i += 2
        elif a == "--no-junction":
            junction = False; i += 1
        elif a == "--no-body":
            with_body = False; i += 1
        elif "," in a and all(t.lstrip("-").replace(".", "").isdigit() for t in a.split(",")):
            views.append(tuple(float(x) for x in a.split(","))); i += 1
        else:
            focus = a; i += 1
    views = views or [(0, 8), (50, 8), (90, 8), (140, 8)]
    base, lib = mh.load()
    cat = SliderCatalog(lib.names)
    c, verts, placed, shaped, info = jv.build(specs, macro, base, lib, cat)
    for text in pose:                                       # "arm-1:raise_=40,swing=10"
        ident, _, rest = text.partition(":")
        for spec in c.appendages:
            if spec.id == ident:
                spec.controls.update({k: float(v) for k, v in (kv.split("=") for kv in rest.split(","))})
        placed = c.instantiate_appendages(base, verts, lib)
        shaped = c.shaped_body(base, verts, placed, info)
    body_v = shaped if junction else verts
    shells = layer_shells(base, body_v, placed, layers, cut=cut, levels=levels, body=with_body)
    if junction and with_body and "skin" in layers and cut in (None, "none") and levels >= 1:
        # round 4: the girdle limbs' root collar is on by default (collar.py; it needs a subdivided body)
        from minimaster.character import collar as collar_mod
        shells += [po.Shell(m.transform(po.MH_TO_Z_UP), mat) for _, _, m, mat in collar_mod.collar_shells(
            base, body_v, placed, levels=levels, enabled=c.collar_enabled)]
    sel = [p for p in placed if p.name.startswith(focus)] or placed
    ctr = np.vstack([np.asarray(p.limb.joints(p.controls)["shoulder"])[None, :] for p in sel]).mean(axis=0)
    ctr_z = (po.MH_TO_Z_UP[:3, :3] @ ctr) + po.MH_TO_Z_UP[:3, 3]
    img = np.concatenate([po.render_portrait(shells, size=size, azimuth=az, elevation=el, fit=(ctr_z, rad))
                          for az, el in views], axis=1)
    po.write_png(out, img)
    print("wrote", out, [p.name for p in placed])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
