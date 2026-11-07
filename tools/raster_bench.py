"""Timings of the portrait renderer and the draft tier (see docs/renderer.md).

    PYTHONPATH=. python tools/raster_bench.py [--size 320x480] [--repeat 7]

Scenes: the body (``levels=0``, 26,756 triangles) with a pair of bat wings
(2 x 8,274) and with a pair of feathered "angel" wings (2 x 85,624). For each:

* ``final cold`` -- ``render_portrait`` with an empty shading cache (first frame
  after loading a character or editing every shell),
* ``final warm`` -- the same call again (a camera move: normals/AO cached),
* ``final after wing edit`` -- one wing re-posed, the body's terms still cached,
* ``draft``      -- body proxy + the two wings clustered, ``render_draft``, in fps.

The pre-change per-triangle loop is not in the package any more (it lives in
``tests/test_raster.py`` as the golden reference); measure it there.
"""

import argparse
import statistics
import sys
import time

sys.path.insert(0, ".")

from minimaster.character import appendages as A  # noqa: E402
from minimaster.character import draft as D  # noqa: E402
from minimaster.character import mhbase as mh  # noqa: E402
from minimaster.character import portrait as po  # noqa: E402
from minimaster.character import shading as sh  # noqa: E402
from minimaster.core.mesh import Mesh  # noqa: E402

WING = sh.Material(base_color="#6f4f3f", roughness=0.74, specular=0.28, f0=0.035,
                   wrap=0.40, scatter="#8c4a38", rim=0.24)


def timed(fn, repeat):
    out = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return statistics.median(out), min(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="320x480")
    ap.add_argument("--repeat", type=int, default=7)
    args = ap.parse_args()
    w, h = (int(x) for x in args.size.split("x"))
    base, _ = mh.load()
    body = po.character_shells(base, None, levels=0, with_eyes=True)
    proxy = D.BodyProxy(base, 4000)
    print(f"{w}x{h}, median / best of {args.repeat}; body {len(body[0].mesh.faces)} tris, "
          f"proxy {len(proxy.faces)} tris")

    coarse = dict(detail=5.0, min_cells=2, max_cells=10)
    for label, style, extra, pkw in (
            ("bat", "membrane", {}, {}),
            ("angel", "feathered", {}, coarse),
            ("angel (draft rig)", "feathered",
             dict(coverts=False, underwing=False, bones=False), {})):
        app = A.back_wings(style)
        app.params.update(extra)
        rigs = [pw for _, pw in app.build_rigs(base, None)]
        full = [pw.pose().transform(po.MH_TO_Z_UP) for pw in rigs]
        shells = list(body) + [po.Shell(m, WING) for m in full]
        tris = sum(len(s.mesh.faces) for s in shells)

        def render():
            return po.render_portrait(shells, size=(w, h))

        cache = sh.DEFAULT_CACHE
        cold = []
        for _ in range(args.repeat):
            cache.clear()
            t = time.perf_counter()
            render()
            cold.append(time.perf_counter() - t)
        warm = timed(render, args.repeat)

        k = [0]

        def edit():                     # drag a pose control: wing meshes change, body does not
            k[0] += 1
            m = rigs[0].pose(fold=0.1 + 0.05 * (k[0] % 10)).transform(po.MH_TO_Z_UP)
            po.render_portrait(list(body) + [po.Shell(m, WING), shells[-1]], size=(w, h))
        edit()
        edited = timed(edit, args.repeat)

        # draft tier: body proxy + clustered wings (the wing posing itself not included)
        cps = []
        for pw in rigs:
            rig = pw.rig
            piece = D.pieces_from_ranges([rig.ranges[p.name] for p in rig.pieces],
                                         len(rig.rest_mesh.vertices))
            cps.append(D.ClusterProxy.by_pieces(rig.rest_mesh.vertices, rig.rest_mesh.faces,
                                                piece, **pkw))
        dshells = [po.Shell(proxy.mesh(None), sh.SKIN)] + [
            po.Shell(Mesh(cp.update(pw.pose().vertices), cp.faces).transform(po.MH_TO_Z_UP), WING)
            for cp, pw in zip(cps, rigs)]
        dtris = sum(len(s.mesh.faces) for s in dshells)
        az = [0]

        def draft():
            az[0] += 7
            D.render_draft(dshells, size=(w, h), azimuth=az[0], shading=["smooth", "flat", "flat"])
        draft()
        dm = timed(draft, max(args.repeat, 15))
        print(f"{label:18s} {tris:7d} tris | final cold {statistics.median(cold) * 1e3:6.0f} ms "
              f"(best {min(cold) * 1e3:.0f}) | warm {warm[0] * 1e3:5.0f} ms "
              f"({1 / warm[0]:4.1f} fps) | after a wing edit {edited[0] * 1e3:5.0f} ms | "
              f"draft {dtris:6d} tris {dm[0] * 1e3:5.1f} ms ({1 / dm[0]:5.0f} fps)")


if __name__ == "__main__":
    main()
