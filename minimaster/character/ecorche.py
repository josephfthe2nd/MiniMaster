"""The anatomical layers of the appendages, drawn: an ecorche view.

An appendage that has an anatomy exposes it through ``Placed.layers()``: named meshes in the
MakeHuman frame, ``"bone"`` (the internal chain, hidden), ``"muscle"`` (the masses that shape the
surface) and ``"skin"`` (the shell that is printed and rendered). This module turns them into
render shells so the layers can be shown and checked::

    shells = ecorche.layer_shells(base, shaped_verts, placed, ("bone", "muscle"), cut=True)
    img = portrait.render_portrait(shells, ...)

``cut=True`` removes the half of every layer but the innermost shown (and of the body) on the positive side of a plane
(default: the sagittal plane through the appendage's anchor, so a view from +X looks into the
section), which is how one sees the bone inside the muscle inside the skin. ``tools/tail_view.py``
and ``tools/junction_view.py --layers`` use it.
"""

from __future__ import annotations

import numpy as np

from ..core.mesh import Mesh
from . import portrait as po
from . import shading as sh

LAYERS = ("bone", "muscle", "skin")

BONE = sh.Material(base_color="#e4d8b8", roughness=0.55)
MUSCLE = sh.Material(base_color="#9c2f2b", roughness=0.45, specular=0.5)
MATERIALS = {"bone": BONE, "muscle": MUSCLE}
TENDON = "#d9c4a3"          # where a layer fades to tendon and fascia (a muscle sleeve's distal part)


def parse_layers(text) -> tuple:
    """``"bone,muscle"`` -> ``("bone", "muscle")`` (in the order bone, muscle, skin); a ``ValueError``
    names the unknown layer."""
    names = [t.strip().lower() for t in (text.split(",") if isinstance(text, str) else text) if t]
    for n in names:
        if n not in LAYERS:
            raise ValueError(f"unknown layer {n!r}; the layers are {', '.join(LAYERS)}")
    return tuple(n for n in LAYERS if n in names)


def cut_mesh(mesh: Mesh, point, normal) -> Mesh:
    """``mesh`` without the faces whose centroid lies on the positive side of the plane (an open
    shell: a section view) turned inside out, so that it shows its inner side to a viewer on the cut side."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    cent = mesh.vertices[mesh.faces].mean(axis=1)
    keep = ((cent - np.asarray(point, dtype=np.float64)) @ n) <= 0.0
    # inside out: the renderer culls back faces and the INSIDE of the cut shell is what one looks at
    return Mesh(mesh.vertices, mesh.faces[keep][:, ::-1])


def layer_shells(base, verts, placed, layers=LAYERS, *, cut: bool = False, plane=None,
                 levels: int = 0, body: bool = True, eye_color: str = "#5b7c8d") -> list:
    """Render shells (Z up) for the body (``verts``: the body that is meshed, i.e. with the skin
    moulded around the appendages) and the ``layers`` of every appendage in ``placed``.

    ``plane`` is ``(point, normal)`` in the MakeHuman frame for ``cut`` (default: through the
    first appendage's anchor, normal +X). With ``"skin"`` among the layers the shell that is
    printed is drawn with the appendage's own material, the inner layers with bone / muscle
    colours. ``body=False`` leaves the figure out."""
    layers = parse_layers(layers)
    placed = list(placed)
    if plane is None and placed:
        plane = (np.asarray(placed[0].anchor, dtype=np.float64), np.array([1.0, 0.0, 0.0]))
    xf = po.MH_TO_Z_UP
    shells = []
    if body:
        shells = list(po.character_shells(base, verts, eye_color=eye_color, levels=levels,
                                          with_eyes=not cut))
        if cut and plane is not None:
            pz = xf[:3, :3] @ np.asarray(plane[0]) + xf[:3, 3]
            nz = xf[:3, :3] @ np.asarray(plane[1])
            shells = [po.Shell(cut_mesh(s.mesh, pz, nz), s.material, s.ao, s.ao_strength,
                               s.base_colors) if i == 0 else s for i, s in enumerate(shells)]
    for p in placed:
        meshes = p.layers()
        for name in layers:
            m = meshes.get(name)
            if m is None or not len(m.faces):
                continue
            if cut and plane is not None and name != layers[0]:
                m = cut_mesh(m, plane[0], plane[1])       # the innermost layer shown stays whole
            mat = p.material if name == "skin" else MATERIALS[name]
            w = p.kind.layer_colors(p, name, p.controls)
            colors = None
            if w is not None and len(w) == len(meshes[name].vertices):
                w = np.asarray(w, dtype=np.float64)[:, None]
                colors = w * sh.hex_to_linear(mat.base_color) + (1.0 - w) * sh.hex_to_linear(TENDON)
            shells.append(po.Shell(m.transform(xf), mat, base_colors=colors))
    return shells
