"""Numbers for a wing's root: what the shoulder anatomy does to the body (docs/anatomy_wing.md).

    python tools/wing_root_report.py [--bodies neutral,child,heavy,tall,short,female]
        [--styles membrane,dragon,feathered] [--poses "rest;preset=folded;raise_=45"] [--no-crossings]

One row per (body, style, pose): the vertices moved, the cage quads backed off by the validity pass,
the smallest and largest area ratio of the touched quads and the largest edge growth (1 = untouched),
the angle at which the arm's axis crosses the skin before and after (90 = leaves it square), the volume
the relief adds (litres: a MakeHuman unit is a decimetre) and the number of NEW pairs of crossing cage
triangles near the junction (the skin pushed through itself: the armpit and the shoulder). Needs the CC0
assets. Run from the repository root.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from minimaster.character import junction as J  # noqa: E402
from minimaster.character import kinds, wingroot as WR  # noqa: E402
from minimaster.character import mhbase as mh  # noqa: E402
from minimaster.character.sliders import Character, SliderCatalog  # noqa: E402
from minimaster.core.subdiv import quads_to_mesh  # noqa: E402

BODIES = {
    "neutral": {"gender": 1.0, "muscle": 0.55},
    "tall": {"gender": 1.0, "muscle": 0.55, "height": 1.0},
    "short": {"gender": 1.0, "muscle": 0.55, "height": 0.0},
    "child": {"age": 0.1875},
    "heavy": {"gender": 1.0, "muscle": 1.0, "weight": 1.0},
    "female": {"gender": 0.0, "muscle": 0.4},
}


def _seg_tri(P0, P1, A, B, C):
    e1, e2, D = B - A, C - A, P1 - P0
    h = np.cross(D, e2)
    a = (e1 * h).sum(1)
    ok = np.abs(a) > 1e-14
    f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    s = P0 - A
    u = f * (s * h).sum(1)
    q = np.cross(s, e1)
    v = f * (D * q).sum(1)
    t = f * (e2 * q).sum(1)
    eps = 1e-9
    return ok & (u > eps) & (v > eps) & (u + v < 1 - eps) & (t > eps) & (t < 1 - eps)


def crossings(v, q, region) -> set:
    """Pairs of non-adjacent cage triangles, at least one touching ``region`` (a vertex mask), that cross."""
    tri = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    P = v[tri]
    lo, hi = P.min(1), P.max(1)
    out = set()
    for i in np.nonzero(region[tri].any(1))[0]:
        ov = np.nonzero(np.all(lo <= hi[i] + 1e-9, 1) & np.all(hi >= lo[i] - 1e-9, 1))[0]
        ov = ov[ov != i]
        ov = ov[~(tri[ov][:, :, None] == tri[i][None, None, :]).any((1, 2))]
        if not len(ov):
            continue
        A, B, C = P[i]
        hit = np.zeros(len(ov), bool)
        for X, Y in ((A, B), (B, C), (C, A)):
            hit |= _seg_tri(np.broadcast_to(X, (len(ov), 3)), np.broadcast_to(Y, (len(ov), 3)),
                            P[ov, 0], P[ov, 1], P[ov, 2])
        for k in range(3):
            hit |= _seg_tri(P[ov, k], P[ov, (k + 1) % 3], np.broadcast_to(A, (len(ov), 3)),
                            np.broadcast_to(B, (len(ov), 3)), np.broadcast_to(C, (len(ov), 3)))
        out.update((min(i, j), max(i, j)) for j in ov[hit])
    return out


def parse_pose(text: str):
    if text in ("", "rest"):
        return None
    out = {}
    for kv in text.split(","):
        k, v = kv.split("=")
        out[k] = v if k == "preset" else float(v)
    return out


def measure(base, lib, cat, body: str, style: str, pose, with_crossings: bool = True) -> dict:
    c = Character()
    c.macro.update(BODIES[body])
    c.appendages.append(kinds.add_from_text(f"wing:style={style}", c.appendages, base=base, lib=lib))
    verts = lib.apply(base.verts, c.effective_target_weights(cat))
    placed = c.instantiate_appendages(base, verts, lib)
    if pose:
        for p in placed:
            p.controls = {**p.controls, **pose}
    info: dict = {}
    t0 = time.perf_counter()
    shaped = c.shaped_body(base, verts, placed, info)
    ms = 1000.0 * (time.perf_counter() - t0)
    cv0, cq = base.body_cage(verts)
    cv1, _ = base.body_cage(shaped)
    mv = np.linalg.norm(cv1 - cv0, axis=1)
    moved = mv > 0
    quads = np.nonzero(moved[cq].any(1))[0]

    def area(v):
        p = v[cq[quads]]
        return np.linalg.norm(J._cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1]), axis=1)

    ratio = area(cv1) / np.maximum(area(cv0), 1e-12)
    e = np.concatenate([cq[quads][:, [0, 1]], cq[quads][:, [1, 2]], cq[quads][:, [2, 3]],
                        cq[quads][:, [3, 0]]])
    growth = (np.linalg.norm(cv1[e[:, 0]] - cv1[e[:, 1]], axis=1)
              / np.maximum(np.linalg.norm(cv0[e[:, 0]] - cv0[e[:, 1]], axis=1), 1e-9))
    p = placed[0]
    S, E = p.model.joints(p.controls)
    d = (E - S) / np.linalg.norm(E - S)
    angles = []
    for v in (cv0, cv1):
        h = WR.BodySurface(v, cq).hit(S, d, near=8.0)
        angles.append(float("nan") if h is None else math.degrees(math.asin(min(abs(float(d @ h[2])), 1.0))))
    out = {"body": body, "style": style, "pose": pose or "rest", "ms": ms, "moved": int(moved.sum()),
           "limited": info.get("limited", 0), "area_min": float(ratio.min()), "area_max": float(ratio.max()),
           "edge_max": float(growth.max()), "exit_before": angles[0], "exit_after": angles[1],
           "relief_litres": quads_to_mesh(cv1, cq).volume() - quads_to_mesh(cv0, cq).volume()}
    # the join one SEES: the skin's angle to the arm on a ring 1.3 / 1.8 root radii round the crossing (min: the side
    # the arm leans over; the exit angles above are where the axis meets the skin, inside the arm)
    s0, s1 = WR.BodySurface(cv0, cq), WR.BodySurface(cv1, cq)
    r = p.model.exit_radius()
    for k in (1.3, 1.8):
        for key, sf in (("before", s0), ("after", s1)):
            v = WR.visible_join(s0, sf, S, E, r, k)
            out[f"ring{k}_{key}"] = (float("nan"), float("nan")) if v is None else v
    if with_crossings:
        region = moved.copy()
        region[cq[moved[cq].any(1)].ravel()] = True
        out["new_crossings"] = len(crossings(cv1, cq, region) - crossings(cv0, cq, region))
    return out


def main(argv) -> int:
    args = list(argv)
    opt = {"--bodies": "neutral,child,heavy,tall,short,female", "--styles": "membrane",
           "--poses": "rest;preset=folded;raise_=45"}
    with_crossings = "--no-crossings" not in args
    args = [a for a in args if a != "--no-crossings"]
    for i in range(0, len(args) - 1, 2):
        if args[i] in opt:
            opt[args[i]] = args[i + 1]
    base, lib = mh.load()
    cat = SliderCatalog(lib.names)
    print(f"{'body':8s} {'style':9s} {'pose':18s} {'ms':>5s} {'moved':>5s} {'lim':>4s} {'area':>11s} {'edge':>5s} "
          f"{'exit deg':>10s} {'litres':>6s} {'cross':>5s}  {'seen 1.3r min/mean':>19s} {'seen 1.8r min/mean':>19s}")
    for body in opt["--bodies"].split(","):
        for style in opt["--styles"].split(","):
            for ps in opt["--poses"].split(";"):
                pose = parse_pose(ps)
                r = measure(base, lib, cat, body, style, pose, with_crossings)
                label = "rest" if pose is None else ",".join(f"{k}={v}" for k, v in pose.items())
                print(f"{body:8s} {style:9s} {label:18s} {r['ms']:5.0f} {r['moved']:5d} {r['limited']:4d} "
                      f"{r['area_min']:5.2f}-{r['area_max']:4.2f} {r['edge_max']:5.2f} "
                      f"{r['exit_before']:4.0f}->{r['exit_after']:3.0f} {r['relief_litres']:6.2f} "
                      f"{r.get('new_crossings', '-'):>5}  "
                      + "  ".join(f"{r[f'ring{k}_before'][0]:3.0f}/{r[f'ring{k}_before'][1]:3.0f}->"
                                  f"{r[f'ring{k}_after'][0]:3.0f}/{r[f'ring{k}_after'][1]:3.0f}" for k in (1.3, 1.8)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
