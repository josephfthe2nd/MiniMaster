"""Stretch of a membrane wing under each animator control: the table behind ``BAT_SPEC["limits"]``.

    PYTHONPATH=. python tools/wing_ranges.py [membrane|dragon] [free|placed] [fold=0.0]

For every control it prints ``value: p99 stretch / edges over 1.3x / max`` of the membrane's
mid-surface grid (edges whose rest length is over 0.15 units), with the controls UNclamped, so the
ranges in ``wings.BAT_SPEC`` can be re-derived whenever the skin changes: pick the largest span of
each control that keeps p99 near 1.25 with only a handful of edges over 1.3x. ``free`` is the wing
alone (raise about its chord, swing in its plane); ``placed`` is on the MakeHuman body (raise about
the body's fore-aft axis, swing about its vertical one). The stowed wing has stretch of its own
(p99 1.4 at fold 1: pleats and gathers, not a control), so judge a control against its own zero.
"""
import sys

import numpy as np

SWEEPS = {
    "raise_": [-70, -60, -50, -40, -35, -30, -20, -10, 0, 10, 20, 25, 30, 40, 50, 70, 100],
    "swing": [-30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 85],
    "twist": [-45, -35, -30, -25, -15, 0, 15, 22, 25, 35, 45],
    "fan": [0.55, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2],
    "curl": [-10, 0, 10, 20, 30, 40, 50, 60, 80, 100, 150],
    "wrist_flex": [-60, -45, -30, -15, 0, 15, 30, 45],
    "hand_twist": [-30, -22, -10, 0, 10, 22, 30],
}


def grid_stretch(rig, mesh, min_rest=0.15):
    """``(p99, edges over 1.3x, max, edges)`` of the membrane's mid-surface grid under ``mesh``,
    against the rest pose; edges shorter than ``min_rest`` at rest (the sliver at a fingertip)
    are left out."""
    piece = next(p for p in rig.pieces if p.membrane and p.pleat)
    a, b = rig.ranges[piece.name]
    n1, m1 = piece.pleat["shape"]
    nv = n1 * m1
    R = rig.rest_mesh.vertices[a:b]
    V = mesh.vertices[a:b]
    rest = (0.5 * (R[:nv] + R[nv:2 * nv])).reshape(n1, m1, 3)
    pos = (0.5 * (V[:nv] + V[nv:2 * nv])).reshape(n1, m1, 3)
    out = []
    for axis in (0, 1):
        r = np.linalg.norm(np.diff(rest, axis=axis), axis=2)
        q = np.linalg.norm(np.diff(pos, axis=axis), axis=2)
        out.append(q[r > min_rest] / r[r > min_rest])
    s = np.concatenate(out)
    return float(np.percentile(s, 99)), int((s > 1.3).sum()), float(s.max()), int(s.size)


def main(argv):
    from minimaster.character import wings as W
    from minimaster.character.wingrig import PoseControls
    style = argv[0] if argv else "membrane"
    mode = argv[1] if len(argv) > 1 else "free"
    fold = float(argv[2].split("=")[1]) if len(argv) > 2 else 0.0
    if mode == "placed":
        from minimaster.character import appendages as A
        from minimaster.character import mhbase as mh
        base, lib = mh.load()
        v = lib.apply(base.verts, mh.macro_weights(gender=1.0))
        rig = A.back_wings(style).build_rigs(base, v)[0][1].rig
    else:
        rig = W.make_wing_rig(style)
    for name, values in SWEEPS.items():
        row = []
        for x in values:
            pose = rig.skeleton.controls_to_pose(PoseControls(**{name: float(x), "fold": fold}))
            p99, over, top, _ = grid_stretch(rig, rig.pose_mesh(pose))
            row.append(f"{x:g}: {p99:.2f}/{over}/{top:.2f}")
        print(f"{style} {mode} fold {fold:g} {name}\n    " + "  ".join(row))


if __name__ == "__main__":
    main(sys.argv[1:])
