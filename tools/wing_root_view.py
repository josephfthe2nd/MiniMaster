"""Close-ups of a wing's root: the skin, and the anatomy under it (bone, muscle, skin).

    python tools/wing_root_view.py OUT.png "wing:style=membrane" [az,el ...] [--macro neutral|k=v,...]
        [--pose fold=0.8,raise_=30 | --pose preset=folded] [--layers bone,muscle,skin] [--cut y|x|z]
        [--cut-at 0.0] [--window 2.4] [--levels 1] [--r 2.6] [--size 420x420] [--pair] [--skin-only]

One ``azimuth,elevation`` pair per view; the views are laid side by side. ``--pair`` puts the body
WITHOUT the junction above and with it below. ``--layers`` draws the layers of every wing
(``bone``, ``muscle``, ``skin``; an ecorche): with ``--cut`` the figure and the outer layers are
sectioned by a plane through the left wing's socket (``y`` a horizontal section, ``x`` a sagittal one,
``z`` a coronal one) so one sees the bone inside the muscle inside the skin. ``--skin-only`` hides
the wings (the skin alone shows what the junction does to the body).

This is the wing's own ecorche view; the generic one (``minimaster character --layers``, S2's
``ecorche.py``) draws the same ``Placed.layers()``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from minimaster.character import kinds, portrait as po  # noqa: E402
from minimaster.character import mhbase as mh  # noqa: E402
from minimaster.character import shading as sh  # noqa: E402
from minimaster.character.sliders import Character, SliderCatalog  # noqa: E402
from minimaster.core.mesh import Mesh  # noqa: E402

BODIES = {
    "neutral": {"gender": 1.0, "muscle": 0.55},
    "tall": {"gender": 1.0, "muscle": 0.55, "height": 1.0},
    "short": {"gender": 1.0, "muscle": 0.55, "height": 0.0},
    "child": {"age": 0.1875},
    "heavy": {"weight": 1.0, "gender": 1.0, "muscle": 1.0},
    "female": {"gender": 0.0, "muscle": 0.4},
    "slim_f": {"weight": 0.0, "gender": 0.0},
}
LAYERS = ("bone", "muscle", "skin")
MATERIALS = {"bone": sh.Material(base_color="#e4d8b8", roughness=0.55),
             "muscle": sh.Material(base_color="#9c2f2b", roughness=0.45, specular=0.5)}


def cut_mesh(mesh: Mesh, point, normal) -> Mesh:
    """``mesh`` without the faces whose centroid is on the positive side of the plane, turned inside
    out so a viewer on the cut side sees the inside of the shell (a section view)."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    cent = mesh.vertices[mesh.faces].mean(axis=1)
    keep = ((cent - np.asarray(point, dtype=np.float64)) @ n) <= 0.0
    return Mesh(mesh.vertices, mesh.faces[keep][:, ::-1])


def window_mesh(mesh: Mesh, center, radius) -> Mesh:
    """``mesh`` without the faces whose centroid is within ``radius`` of ``center``: a window."""
    cent = mesh.vertices[mesh.faces].mean(axis=1)
    keep = np.linalg.norm(cent - np.asarray(center, dtype=np.float64), axis=1) >= radius
    return Mesh(mesh.vertices, mesh.faces[keep])


def parse_pose(text: str) -> dict:
    out = {}
    for kv in (text or "").split(","):
        if not kv:
            continue
        k, v = kv.split("=")
        out[k] = v if k == "preset" else float(v)
    return out


def build(specs, macro, base, lib, cat, pose=None):
    c = Character()
    c.macro.update(macro)
    for t in specs:
        c.appendages.append(kinds.add_from_text(t, c.appendages, base=base, lib=lib))
    verts = lib.apply(base.verts, c.effective_target_weights(cat))
    placed = c.instantiate_appendages(base, verts, lib) if c.appendages else []
    if pose:
        for p in placed:
            p.controls = {**p.controls, **pose}
    info: dict = {}
    shaped = c.shaped_body(base, verts, placed, info)
    return c, verts, placed, shaped, info


def layer_shells(placed, layers, cut, plane, window=None):
    xf = po.MH_TO_Z_UP
    out = []
    for k, p in enumerate(placed):
        meshes = p.layers()
        for name in layers:
            m = meshes.get(name)
            if m is None or not len(m.faces):
                continue
            if cut and plane is not None and name != layers[0]:
                m = cut_mesh(m, plane[0], plane[1])
            if window is not None and name != layers[0]:
                # a dissection: the skin's window is the widest, the muscle's is 0.6 of it
                m = window_mesh(m, window[k][0], window[k][1] * (1.0 if name == "skin" else 0.62))
            out.append(po.Shell(m.transform(xf), p.material if name == "skin" else MATERIALS[name]))
    return out


def render(base, body_verts, placed, views, size, rad, levels, skin_only=False, layers=None,
           cut=None, cut_at=0.0, no_body=False, window=0.0):
    to_z = lambda p: po.MH_TO_Z_UP[:3, :3] @ np.asarray(p) + po.MH_TO_Z_UP[:3, 3]      # noqa: E731
    left = [p for p in placed if p.side != "r"] or list(placed)
    anchor = np.asarray(left[0].anchor) if left else None
    plane = None
    if cut and anchor is not None:
        sign = -1.0 if cut.startswith("-") else 1.0
        axis = cut.lstrip("-")
        normal = sign * np.array({"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[axis])
        pt = anchor.copy()
        pt[{"x": 0, "y": 1, "z": 2}[axis]] += cut_at
        plane = (pt, np.asarray(normal))
    body = list(po.character_shells(base, body_verts, levels=levels, with_eyes=not cut))
    if plane is not None and not no_body:
        pz, nz = to_z(plane[0]), po.MH_TO_Z_UP[:3, :3] @ plane[1]
        body = [po.Shell(cut_mesh(s.mesh, pz, nz), s.material, s.ao, s.ao_strength, s.base_colors)
                if i == 0 else s for i, s in enumerate(body)]
    win = None
    if window > 0.0 and placed:
        # one window per wing, over the skin at the shoulder socket (the placed frame is Z up)
        win = [(to_z(np.asarray(p.anchor)), window) for p in placed]
        for _, (c, r) in enumerate(win):
            body = [po.Shell(window_mesh(s.mesh, c, r), s.material, s.ao, s.ao_strength, s.base_colors)
                    if i == 0 else s for i, s in enumerate(body)]
    shells = [] if no_body else list(body)
    if layers:
        shells += layer_shells(placed, layers, cut, plane, win)
    elif not skin_only:
        shells += [po.Shell(p.mesh().transform(po.MH_TO_Z_UP), p.material) for p in placed]
    ctr = to_z(anchor) if anchor is not None else (body[0].mesh.bounds[0] + body[0].mesh.bounds[1]) / 2
    return np.concatenate(
        [po.render_portrait(shells, size=size, azimuth=az, elevation=el, fit=(ctr, rad))
         for az, el in views], axis=1)


def main(argv) -> int:
    args = list(argv)
    if len(args) < 2:
        print(__doc__)
        return 2
    out, specs = args[0], [s for s in args[1].split(";") if s]
    views, macro, size, rad, levels = [], BODIES["neutral"], (420, 420), 2.6, 1
    pair = skin_only = no_body = False
    layers = cut = pose = None
    cut_at = 0.0
    window = 0.0
    i = 2
    while i < len(args):
        a = args[i]
        if a == "--macro":
            m = args[i + 1]
            macro = (BODIES[m] if m in BODIES else
                     {k: float(v) for k, v in (kv.split("=") for kv in m.split(","))})
            i += 2
        elif a == "--size":
            size = tuple(int(x) for x in args[i + 1].split("x")); i += 2
        elif a == "--r":
            rad = float(args[i + 1]); i += 2
        elif a == "--levels":
            levels = int(args[i + 1]); i += 2
        elif a == "--layers":
            layers = tuple(n for n in LAYERS if n in args[i + 1].split(",")); i += 2
        elif a == "--cut":
            cut = args[i + 1]; i += 2
        elif a == "--cut-at":
            cut_at = float(args[i + 1]); i += 2
        elif a == "--window":
            window = float(args[i + 1]); i += 2
        elif a == "--pose":
            pose = parse_pose(args[i + 1]); i += 2
        elif a == "--pair":
            pair, i = True, i + 1
        elif a == "--skin-only":
            skin_only, i = True, i + 1
        elif a == "--no-body":
            no_body, i = True, i + 1
        else:
            views.append(tuple(float(x) for x in a.split(","))); i += 1
    views = views or [(180, 15), (130, 20), (90, 10)]
    base, lib = mh.load()
    cat = SliderCatalog(lib.names)
    _, verts, placed, shaped, info = build(specs, macro, base, lib, cat, pose)
    rows = []
    if pair:
        rows.append(render(base, verts, placed, views, size, rad, levels, skin_only, layers, cut, cut_at, no_body, window))
    rows.append(render(base, shaped, placed, views, size, rad, levels, skin_only, layers, cut, cut_at, no_body, window))
    po.write_png(out, np.concatenate(rows, axis=0))
    print("wrote", out, [p.name for p in placed], info)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
