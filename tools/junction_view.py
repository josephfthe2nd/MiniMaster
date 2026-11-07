"""Close-up renders of an appendage junction, with and without the skin moulded around it.

    python tools/junction_view.py OUT.png "tail;horn" FOCUS [az,el ...] [--macro k=v,...]
        [--levels 1] [--r 2.2] [--size 420x420] [--before] [--skin-only] [--pair]

``specs`` is a ``;``-separated list of ``--add`` texts (``tail``, ``wing:style=feathered``,
``harvest:preset=upper_arms`` ...); FOCUS is the kind (tail, horn, wing, arm, leg) or ``all``
(what the camera frames). Each view is one ``azimuth,elevation`` pair; the images are laid
side by side. ``--before`` renders the body WITHOUT the junction (the skin as it was),
``--pair`` renders before (top) and after (bottom), ``--skin-only`` hides the appendages
(the shaped skin alone: shows what the junction does to the body and whether any of it
is left sticking out of the appendage).

Needs the CC0 assets like every character command. Run from the repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from minimaster.character import kinds, portrait as po  # noqa: E402
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


def build(specs, macro, base, lib, cat):
    """``(character, verts, placed, shaped verts, info)`` for the appendage texts ``specs``."""
    c = Character()
    c.macro.update(macro)
    for t in specs:
        c.appendages.append(kinds.add_from_text(t, c.appendages, base=base, lib=lib))
    verts = lib.apply(base.verts, c.effective_target_weights(cat))
    placed = c.instantiate_appendages(base, verts, lib) if c.appendages else []
    info: dict = {}
    shaped = c.shaped_body(base, verts, placed, info)
    return c, verts, placed, shaped, info


def render(base, body_verts, placed, focus, views, size, rad, levels, skin_only=False):
    body = po.character_shells(base, body_verts, levels=levels, with_eyes=True)
    shells = list(body)
    if not skin_only:
        shells += [po.Shell(p.mesh().transform(po.MH_TO_Z_UP), p.material) for p in placed]
    sel = [p for p in placed if focus == "all" or p.name.startswith(focus)]
    if sel:
        ctr = np.vstack([p.anchor[None, :] for p in sel]).mean(axis=0)
        ctr_z = (po.MH_TO_Z_UP[:3, :3] @ ctr) + po.MH_TO_Z_UP[:3, 3]
    else:
        lo, hi = body[0].mesh.bounds
        ctr_z = (lo + hi) / 2
    return np.concatenate(
        [po.render_portrait(shells, size=size, azimuth=az, elevation=el, fit=(ctr_z, rad))
         for az, el in views], axis=1)


def main(argv) -> int:
    args = list(argv)
    if len(args) < 3:
        print(__doc__)
        return 2
    out, specs, focus = args[0], [s for s in args[1].split(";") if s], args[2]
    views, macro, size, rad, levels = [], BODIES["neutral"], (420, 420), 2.2, 1
    before = pair = skin_only = False
    i = 3
    while i < len(args):
        a = args[i]
        if a == "--macro":
            m = args[i + 1]
            macro = (BODIES[m] if m in BODIES else
                     {k: float(v) for k, v in (kv.split("=") for kv in m.split(","))})
            i += 2
        elif a == "--size":
            size = tuple(int(x) for x in args[i + 1].split("x"))
            i += 2
        elif a == "--r":
            rad = float(args[i + 1])
            i += 2
        elif a == "--levels":
            levels = int(args[i + 1])
            i += 2
        elif a == "--before":
            before, i = True, i + 1
        elif a == "--pair":
            pair, i = True, i + 1
        elif a == "--skin-only":
            skin_only, i = True, i + 1
        else:
            views.append(tuple(float(x) for x in a.split(",")))
            i += 1
    views = views or [(180, 15), (130, 20), (90, 10), (225, 25)]
    base, lib = mh.load()
    cat = SliderCatalog(lib.names)
    _, verts, placed, shaped, info = build(specs, macro, base, lib, cat)
    rows = []
    if pair or before:
        rows.append(render(base, verts, placed, focus, views, size, rad, levels, skin_only))
    if pair or not before:
        rows.append(render(base, shaped, placed, focus, views, size, rad, levels, skin_only))
    po.write_png(out, np.concatenate(rows, axis=0))
    print("wrote", out, [p.name for p in placed], info)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
