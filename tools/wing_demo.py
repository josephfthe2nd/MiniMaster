"""The wing sheet: both styles on a body, plus spread-vs-folded and planforms.

    PYTHONPATH=. python tools/wing_demo.py
"""
import numpy as np

from minimaster.character import appendages as A
from minimaster.character import creature_anatomy as ca
from minimaster.character import mhbase as mh
from minimaster.character import portrait as po
from minimaster.character import shading as sh
from minimaster.character import wings as W
from minimaster.character.sliders import Character, SliderCatalog
from minimaster.core.mesh import Mesh

# wing-local (X span, Y down, Z chord) -> world Z-up, wing's top upward
WING_TO_WORLD = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0],
                          [0, -1.0, 0, 0], [0, 0, 0, 1.0]])

MEMBRANE = sh.Material(base_color="#6f4f3f", roughness=0.74, specular=0.28,
                       f0=0.035, wrap=0.40, scatter="#8c4a38", rim=0.24)
PLUMAGE = sh.Material(base_color="#cdc4b6", roughness=0.55, specular=0.42,
                      f0=0.040, wrap=0.26, scatter="#9a8f80", rim=0.20)
MATS = {"membrane": MEMBRANE, "feathered": PLUMAGE}

# left scapula; the lateral up_hint aims the span across the back
ANCHOR = (0.84, 4.39, -0.73)
HINT = (1.0, 1.5, 0.0)


def wing_appendage(style, **params):
    return A.Appendage("wing", anchor_point=ANCHOR, mirror=True, up_hint=HINT,
                       scale=1.0, params=dict(style=style, **params))


def panel_planforms():
    """Each style on its own, seen from above."""
    out = []
    for style, fn in (("membrane", W.membrane_wing), ("feathered", W.feathered_wing)):
        mesh = fn().transform(WING_TO_WORLD)
        lo, hi = mesh.bounds
        out.append(po.render_portrait(
            [po.Shell(mesh, MATS[style])], size=(660, 460), azimuth=0.0,
            elevation=86.0, fit=((lo + hi) / 2.0, float(np.linalg.norm(hi - lo)) / 2.0)))
    return np.concatenate(out, axis=1)


def panel_on_body(base, lib, cat, size=(560, 800)):
    """Plain, membrane-winged and feather-winged, all framed identically."""
    panels = []
    variants = [(None, None), ("membrane", 0.0), ("feathered", 0.0),
                ("membrane", 1.0)]
    fit = None
    for style, fold in variants:
        c = Character()
        c.macro.update(gender=1.0, muscle=0.55)
        apps = []
        if style:
            apps = [wing_appendage(style, fold=fold)]
            ca.apply_to(c, apps, cat)
        v = lib.apply(base.verts, c.target_weights(cat))
        shells = po.character_shells(base, v, levels=1, with_eyes=True)
        if fit is None:                       # frame every panel off the body
            lo, hi = shells[0].mesh.bounds
            fit = ((lo + hi) / 2.0, float(np.linalg.norm(hi - lo)) / 2.0 * 1.5)
        for app in apps:
            for _, m in app.build(base, v):
                shells.append(po.Shell(m.transform(po.MH_TO_Z_UP), MATS[style]))
        # straight from behind and slightly above: an off-axis camera turns
        # one wing face-on and the other edge-on and reads as an asymmetry
        panels.append(po.render_portrait(shells, size=size, azimuth=180.0,
                                         elevation=34.0, fit=fit))
    return np.concatenate(panels, axis=1)


def main():
    base, lib = mh.load()
    cat = SliderCatalog(lib.names)
    body = panel_on_body(base, lib, cat)
    plans = panel_planforms()
    # pad the narrower strip so the two rows stack cleanly
    w = max(body.shape[1], plans.shape[1])

    def pad(img):
        if img.shape[1] == w:
            return img
        out = np.zeros((img.shape[0], w, 4), dtype=img.dtype)
        out[:, :, :3] = img[0, 0, :3]
        out[:, :, 3] = 255
        off = (w - img.shape[1]) // 2
        out[:, off:off + img.shape[1]] = img
        return out

    sheet = np.concatenate([pad(body), pad(plans)], axis=0)
    po.write_png("docs/images/wings.png", sheet)
    print("wrote docs/images/wings.png", sheet.shape)

    for style in ("membrane", "feathered"):
        m = W.make_wing(style=style)
        rep = m.integrity_report()
        print(f"  {style:10s} tris={len(m.faces):6d} watertight={rep['watertight']} "
              f"span={float(np.ptp(m.vertices[:, 0])):.1f}")

    v = lib.apply(base.verts, Character().target_weights(cat))
    cv, cq = base.body_cage(v)
    vol = Mesh(cv, np.vstack([cq[:, [0, 1, 2]], cq[:, [0, 2, 3]]])).volume()
    pair = [m for _, m in wing_appendage("membrane").build(base, v)]
    print()
    print(ca.flight_report(vol, ca.wingspan_of(pair)).as_text())


if __name__ == "__main__":
    main()
