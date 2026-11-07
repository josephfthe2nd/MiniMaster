"""Close-ups of horns on several heads, with the anatomical layers and next to reference images.

    python tools/horn_view.py OUT.png "horn;horn:shape=ram" [az,el ...] [--bodies neutral,tall,child]
        [--r 1.6] [--size 360x360] [--levels 1] [--layers bone,muscle,skin] [--cut | --section] [--no-body] [--before]
        [--ref DIR/ID,...] [--face]

``specs`` is a ``;``-separated list of ``--add`` texts (``horn``, ``horn:shape=ram`` ...). Each view is an
``azimuth,elevation`` pair (0 = the face, 90 = the left side seen from the face's side, 180 = the back);
the default is front, three-quarter, side and top. ``--bodies`` names the bodies (rows): neutral, tall, short,
child, heavy, old, female, thin, or ``k=v,...`` macro values joined by ``+`` (``gender=1+age=0.1875``).

``--layers`` draws the anatomy instead of the figure alone (``ecorche.py``): ``bone`` (the cornual process), ``muscle`` (the soft
tissue the scalp is tented over) and ``skin`` (the sheath); ``--cut`` removes the half of every layer outside the
innermost one and of the head on the +X side of the horn's sagittal plane, so a view from +X looks into the section.
``--before`` leaves the scalp as the morphs made it (no junction). ``--no-body`` leaves the head out of a layers picture. ``--face`` frames the whole head. ``--section`` is ``--layers bone,muscle,skin --cut`` on the plane through the horn's axis and
bend (a longitudinal section), the first view looking into it.

``--ref`` puts reference pictures beside the renders (one per row, cycling): ``DIR/ID`` is the id of an image
in a reference library (``$MM_REFS/<DIR>/images/<ID>_*``), e.g. ``horn/A22``; set ``MM_REFS`` to the library's directory. The picture is scaled to the row's height.

Needs the CC0 assets like every character command. Run from the repository root.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from minimaster.character import ecorche, kinds, portrait as po  # noqa: E402
from minimaster.character import mhbase as mh  # noqa: E402
from minimaster.character.sliders import Character, SliderCatalog  # noqa: E402

BODIES = {
    "neutral": {"gender": 1.0, "muscle": 0.55},
    "tall": {"gender": 1.0, "muscle": 0.55, "height": 1.0},
    "short": {"gender": 1.0, "muscle": 0.55, "height": 0.0},
    "child": {"age": 0.1875},
    "heavy": {"weight": 1.0, "gender": 1.0, "muscle": 1.0},
    "old": {"gender": 1.0, "muscle": 0.55, "age": 1.0},
    "female": {"gender": 0.0},
    "thin": {"weight": 0.0, "gender": 0.5},
}
DEFAULT_VIEWS = [(0, 10), (50, 15), (90, 8), (0, 80)]
REFS = Path(os.environ["MM_REFS"]) if os.environ.get("MM_REFS") else None     # the reference library, see --ref


def macro_of(name: str) -> dict:
    if name in BODIES:
        return BODIES[name]
    return {k: float(v) for k, v in (kv.split("=") for kv in name.split("+"))}


def build(specs, macro, base, lib, cat, junction=True):
    """``(character, verts, placed, shaped verts, info)`` for the appendage texts ``specs``."""
    c = Character()
    c.macro.update(macro)
    for t in specs:
        c.appendages.append(kinds.add_from_text(t, c.appendages, base=base, lib=lib))
    verts = lib.apply(base.verts, c.effective_target_weights(cat))
    placed = c.instantiate_appendages(base, verts, lib) if c.appendages else []
    info: dict = {}
    shaped = c.shaped_body(base, verts, placed, info) if junction else verts
    return c, verts, placed, shaped, info


def section_plane(placed):
    """``(plane, (az, el))``: the plane through the first horn's anchor that contains its axis and its bend
    direction (a longitudinal section) and the view that looks into it from its +side."""
    p = next(q for q in placed if q.side in ("l", ""))
    n = p.matrix[:3, 1] / max(float(np.linalg.norm(p.matrix[:3, 1])), 1e-12)
    az = float(np.degrees(np.arctan2(n[0], n[2])))
    el = float(np.degrees(np.arcsin(np.clip(n[1], -1, 1))))
    return (np.asarray(p.anchor, dtype=np.float64), n), (az, el)


def render_row(base, body_verts, placed, views, size, rad, levels, layers, cut, face, plane=None, body=True):
    if layers:
        shells = ecorche.layer_shells(base, body_verts, placed, layers, cut=cut, plane=plane,
                                      levels=levels, body=body)
    else:
        shells = list(po.character_shells(base, body_verts, levels=levels, with_eyes=True))
        shells += [po.Shell(p.mesh().transform(po.MH_TO_Z_UP), p.material) for p in placed]
    if face or not placed or (layers and not body and False):
        lo, hi = shells[0].mesh.bounds
        ctr_z = (lo + hi) / 2
    else:
        ctr = np.vstack([p.anchor[None, :] + 0.5 * p.matrix[:3, 2] / max(np.linalg.norm(p.matrix[:3, 2]), 1e-9)
                         for p in placed]).mean(axis=0)
        ctr_z = (po.MH_TO_Z_UP[:3, :3] @ ctr) + po.MH_TO_Z_UP[:3, 3]
    return np.concatenate(
        [po.render_portrait(shells, size=size, azimuth=az, elevation=el, fit=(ctr_z, rad))
         for az, el in views], axis=1)


def load_ref(ref: str, height: int):
    from PIL import Image
    if REFS is None:
        raise SystemExit("--ref needs MM_REFS: the directory of the reference library (<MM_REFS>/horn/images/A29_*.jpg)")
    region, _, ident = ref.partition("/")
    d = REFS / region / "images"
    hits = sorted(d.glob(f"{ident}_*")) + sorted(d.glob(f"{ident}.*"))
    hits = [h for h in hits if "annot" not in h.name]
    if not hits:
        raise FileNotFoundError(f"no reference image {ref!r} in {d}")
    im = Image.open(hits[0]).convert("RGB")
    w = max(int(im.width * height / im.height), 1)
    return np.asarray(im.resize((w, height)))


def label(img, text):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img).convert("RGB")
    ImageDraw.Draw(im).text((6, 4), text, fill=(235, 235, 235))
    return np.asarray(im)


def main(argv) -> int:
    args = list(argv)
    if len(args) < 2:
        print(__doc__)
        return 2
    out, specs = args[0], [s for s in args[1].split(";") if s]
    views, bodies, size, rad, levels, refs = [], ["neutral"], (360, 360), 1.6, 1, []
    layers = None
    cut = before = face = section = False
    body = True
    i = 2
    while i < len(args):
        a = args[i]
        if a == "--bodies":
            bodies = args[i + 1].split(",")
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
        elif a == "--layers":
            layers = ecorche.parse_layers(args[i + 1])
            i += 2
        elif a == "--ref":
            refs = [r for r in args[i + 1].split(",") if r]
            i += 2
        elif a == "--cut":
            cut, i = True, i + 1
        elif a == "--section":
            section, i = True, i + 1
        elif a == "--no-body":
            body, i = False, i + 1
        elif a == "--before":
            before, i = True, i + 1
        elif a == "--face":
            face, i = True, i + 1
        else:
            views.append(tuple(float(x) for x in a.split(",")))
            i += 1
    views = views or DEFAULT_VIEWS
    base, lib = mh.load()
    cat = SliderCatalog(lib.names)
    rows = []
    for r, name in enumerate(bodies):
        _, verts, placed, shaped, info = build(specs, macro_of(name), base, lib, cat, junction=not before)
        plane, vws = None, views
        if section and placed:
            layers = layers or ecorche.LAYERS
            cut = True
            plane, v0 = section_plane(placed)
            vws = [v0] + [v for v in views if v != DEFAULT_VIEWS[0]] if views is not DEFAULT_VIEWS else [v0]
        row = render_row(base, shaped, placed, vws, size, rad, levels, layers, cut, face, plane, body)
        row = label(row, f"{name}  {info if info else ''}"[:110])
        if refs:
            row = np.concatenate([load_ref(refs[r % len(refs)], row.shape[0]), row], axis=1)
        rows.append(row)
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]
    from PIL import Image
    Image.fromarray(np.concatenate(rows, axis=0)).save(out)
    print("wrote", out, [p.name for p in placed])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
