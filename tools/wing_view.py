"""Fast wing viewer: one wing (no body), several views, any pose.

    PYTHONPATH=. python tools/wing_view.py out.png [membrane|feathered] [fold=0.5 ...]

Used for iterating on the wing and for the rig-pose strip in the docs; the
full body sheets live in ``wing_demo.py``.
"""
import sys

import numpy as np

from minimaster.character import portrait as po
from minimaster.character import shading as sh
from minimaster.character import wings as W

# wing-local (X span, Y down, Z chord) -> world Z-up, the wing's top upward
WING_TO_WORLD = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, -1.0, 0, 0], [0, 0, 0, 1.0]])

MEMBRANE = sh.Material(base_color="#6f4f3f", roughness=0.74, specular=0.28,
                       f0=0.035, wrap=0.40, scatter="#8c4a38", rim=0.24)
PLUMAGE = sh.Material(base_color="#cdc4b6", roughness=0.55, specular=0.42,
                      f0=0.040, wrap=0.26, scatter="#9a8f80", rim=0.20)
DRAGON_HIDE = sh.Material(base_color="#5a3d36", roughness=0.70, specular=0.30, f0=0.035, wrap=0.40, scatter="#8a3b2c", rim=0.26)
MATS = {"membrane": MEMBRANE, "dragon": DRAGON_HIDE, "feathered": PLUMAGE}

def _smooth(x, a, b):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _jitter(name, amount):
    """A stable +-amount multiplier per piece, so neighbours read apart."""
    h = (sum(ord(c) * (i + 3) for i, c in enumerate(name)) * 2654435761) % 1000 / 1000.0
    return 1.0 + amount * (2.0 * h - 1.0)


def albedo(rig, style, mesh=None):
    """Per-vertex linear albedo that makes the structure legible: feathers are
    darker toward the quill and tinted by row (with a little per-feather
    variation so each shingle reads), bones darker than the membrane that
    stretches between them, the membrane lighter toward its free edge."""
    from minimaster.character import shading as sh
    base = sh.hex_to_linear("#a89b88" if style == "feathered" else ("#6a4638" if style == "dragon" else "#7a5a49"))
    sk = rig.skeleton
    V = rig.rest_mesh.vertices
    out = np.tile(base, (len(V), 1))
    tips = {f["tip"]: f for f in sk.feathers}
    row_tint = {"primary": 0.86, "secondary": 0.93, "tertial": 0.99, "scapular": 1.02, "alula": 0.9}
    for p in rig.pieces:
        a, b = rig.ranges[p.name]
        v = V[a:b]
        if p.bone in tips:                                  # a flight feather
            f = tips[p.bone]
            r0, r1 = sk.joints[f["root"]], sk.joints[f["tip"]]
            d = r1 - r0
            t = np.clip(((v - r0) @ d) / float(d @ d), 0.0, 1.0)
            # across the vane: pale along the shaft, darker toward the edges, so
            # each overlapping feather draws its own outline
            perp = np.array([-d[2], 0.0, d[0]]) / max(float(np.hypot(d[0], d[2])), 1e-9)
            lat = (v - r0) @ perp
            e = np.clip(np.abs(lat - np.median(lat)) / max(np.ptp(lat) / 2.0, 1e-9), 0.0, 1.0)
            m = (0.62 + 0.38 * _smooth(t, 0.04, 0.45)) * (1.0 - 0.42 * e ** 1.3) * row_tint.get(f["row"], 1.0)
            out[a:b] = base * (m * _jitter(p.name, 0.09))[:, None]
        elif style == "feathered" and p.bone is not None or p.name.startswith(
                ("greater", "median", "primary_covert", "lesser", "under_")):
            tier = 1.02 if p.name.startswith(("greater", "primary_covert")) else 1.10
            out[a:b] = base * (tier * _jitter(p.name, 0.08))
        elif p.membrane:
            r = np.linalg.norm(v[:, [0, 2]] - sk.wrist[[0, 2]], axis=1)
            f = _smooth(r / max(r.max(), 1e-9), 0.15, 1.0)
            tint = np.array([1.0, 0.82, 0.70])               # warm, translucent edge
            out[a:b] = base * (0.80 + 0.40 * f)[:, None] * (1.0 - f[:, None] * (1.0 - tint) * 0.6)
        elif style in ("membrane", "dragon"):               # bones, claw, knobs
            out[a:b] = base * 0.52
    return out


# Where the membrane root lands on the body, in wing units (plan x, z): what
# back_wings() produces on the default body. Isolated renders use it so the
# planform matches the one on the figure instead of the standalone default.
BODY_ROOT = {"membrane": (-3.3, 2.3), "dragon": (-3.3, 2.3), "feathered": (-2.2063, 2.9923)}
# ... and where the back is in the wing's own frame (out of the back, down the spine, away
# from the spine): a stowed feathered wing lies against it, exactly as on the figure
BODY_BACK = {"feathered": ((0.253, -0.7761, 0.5776), (-0.7873, 0.1818, 0.5892),
                           (0.5623, 0.6038, 0.565))}


def make_rig(style, **kw):
    root = BODY_ROOT.get(style)
    rig = W.make_wing_rig(style, **({"root": root} if root else {}), **kw)
    if style in BODY_BACK:
        rig.skeleton.stow_against(*BODY_BACK[style])
    return rig


# name -> (azimuth, elevation)
VIEWS = {"plan": (0.0, 88.0), "below": (0.0, -88.0), "front": (0.0, 4.0),
         "three_quarter": (35.0, 26.0), "tip": (90.0, 8.0)}


def frame(meshes):
    lo = np.min([m.bounds[0] for m in meshes], axis=0)
    hi = np.max([m.bounds[1] for m in meshes], axis=0)
    return (lo + hi) / 2.0, float(np.linalg.norm(hi - lo)) / 2.0


def render_views(meshes, style, views=("plan", "three_quarter", "front"),
                 size=(620, 440), fit=None, colors=None):
    """Render posed wing meshes (wing frame) side by side, one row per mesh.
    ``colors`` is an optional per-vertex albedo from :func:`albedo`."""
    meshes = [m.transform(WING_TO_WORLD) for m in meshes]
    fit = fit or frame(meshes)
    rows = []
    for m in meshes:
        row = [po.render_portrait([po.Shell(m, MATS[style], base_colors=colors)], size=size,
                                  azimuth=VIEWS[v][0], elevation=VIEWS[v][1], fit=fit)
               for v in views]
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def main():
    out = sys.argv[1]
    style = sys.argv[2] if len(sys.argv) > 2 else "membrane"
    kw = []
    for a in sys.argv[3:]:
        k, v = a.split("=")
        kw.append((k, float(v)))
    rig = make_rig(style)
    poses = [{}] + [rig.pose_from_controls(**{k: v}) for k, v in kw]
    meshes = [rig.pose_mesh(p) for p in poses]
    img = render_views(meshes, style, colors=albedo(rig, style))
    po.write_png(out, img)
    print("wrote", out, img.shape)


if __name__ == "__main__":
    main()
