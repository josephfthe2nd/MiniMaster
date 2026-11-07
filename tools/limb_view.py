"""Close-ups of an extra limb and its girdle, with the anatomical layers (ecorche).

    python tools/limb_view.py OUT.png "harvest:preset=lower_arms" [FOCUS] [az,el ...]
        [--layers bone,muscle,skin] [--cut coronal|sagittal|none] [--macro neutral|tall|short|child|heavy|slim_f]
        [--pose arm-1:raise_=40,swing=10] [--r 3.0] [--size 420x520] [--levels 1] [--no-junction] [--no-body]

``--layers`` picks what is drawn of every limb (``skin`` is the shell that is printed, ``muscle`` the
masses that shape it, ``bone`` the chain inside); ``--cut`` slices the body and the outer layers away
on one side of the plane through the girdle's pivot (``coronal``: the front half goes, look from the
front; ``sagittal``: the lateral half goes, look from the side) so that the layers inside can be seen.
Needs the CC0 assets. Run from the repository root. Uses ``minimaster.character.ecorche`` when it
exists (the tail track's helper) and its own copy of the same drawing otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import junction_view as jv  # noqa: E402
from minimaster.character import kinds, mhbase as mh, portrait as po  # noqa: E402
from minimaster.character import shading as sh  # noqa: E402
from minimaster.character.sliders import SliderCatalog  # noqa: E402
from minimaster.core.mesh import Mesh  # noqa: E402

LAYERS = ("bone", "muscle", "skin")
MATERIALS = {"bone": sh.Material(base_color="#e4d8b8", roughness=0.55),
             "muscle": sh.Material(base_color="#9c2f2b", roughness=0.45, specular=0.5)}


def parse_layers(text) -> tuple:
    names = [t.strip().lower() for t in (text.split(",") if isinstance(text, str) else text) if t]
    for n in names:
        if n not in LAYERS:
            raise ValueError(f"unknown layer {n!r}; the layers are {', '.join(LAYERS)}")
    return tuple(n for n in LAYERS if n in names)


def cut_mesh(mesh: Mesh, point, normal) -> Mesh:
    """``mesh`` without the faces on the positive side of the plane, inside out (so that a viewer
    on the cut side sees the inner face of the open shell)."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    cent = mesh.vertices[mesh.faces].mean(axis=1)
    keep = ((cent - np.asarray(point, dtype=np.float64)) @ n) <= 0.0
    return Mesh(mesh.vertices, mesh.faces[keep][:, ::-1])


def layer_shells(base, verts, placed, layers=LAYERS, *, cut: str = "none", plane=None, levels: int = 1,
                 body: bool = True):
    """Render shells (Z up): the body and the ``layers`` of every appendage in ``placed``."""
    try:                                                    # the shared helper, when the tail track's is merged
        from minimaster.character import ecorche
        if cut != "none" and plane is None:
            plane = _plane(placed, cut)
        return ecorche.layer_shells(base, verts, placed, layers, cut=cut != "none", plane=plane,
                                    levels=levels, body=body)
    except ImportError:
        pass
    layers = parse_layers(layers)
    if cut != "none" and plane is None:
        plane = _plane(placed, cut)
    xf = po.MH_TO_Z_UP
    shells = []
    if body:
        shells = list(po.character_shells(base, verts, levels=levels, with_eyes=cut == "none"))
        if cut != "none":
            pz = xf[:3, :3] @ np.asarray(plane[0]) + xf[:3, 3]
            nz = xf[:3, :3] @ np.asarray(plane[1])
            shells = [po.Shell(cut_mesh(s.mesh, pz, nz), s.material, s.ao, s.ao_strength, s.base_colors)
                      if i == 0 else s for i, s in enumerate(shells)]
    for p in placed:
        meshes = p.layers()
        for name in layers:
            m = meshes.get(name)
            if m is None or not len(m.faces):
                continue
            if cut != "none" and name != layers[0]:
                m = cut_mesh(m, plane[0], plane[1])
            mat = p.material if name == "skin" else MATERIALS[name]
            shells.append(po.Shell(m.transform(xf), mat))
    return shells


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
