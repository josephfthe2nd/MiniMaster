"""Figures of the hair system, the particle system and the feathers grown with them.

    python tools/feather_view.py feathers OUT.png [--detail D]   today's sheet (left) beside the grown feather
                                                                  (right): a slotted primary, a secondary, a covert
    python tools/feather_view.py strands OUT.png                  one feather in the making: its barbs as strands,
                                                                  the strands cut at its outline, the cut strands
                                                                  zipped into its mesh
    python tools/feather_view.py systems OUT.png                  the two systems on a bare ball: particles with a
                                                                  template on each, and hair grown from particles

Each prints the triangles and the milliseconds of what it drew.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from minimaster.character import feather as FE  # noqa: E402
from minimaster.character import kinds as K, portrait as po, wings as W  # noqa: E402
from minimaster.core import hair as H, particles as PT  # noqa: E402
from minimaster.core.mesh import Mesh  # noqa: E402
from minimaster.core.subdiv import cube_cage, subdivided_mesh  # noqa: E402

# the arguments the wing gives three of its feathers (ridge up, for the picture)
PRIMARY = ((5.89, 0.764, 0.2), dict(curl=0.152, cup=0.08, lead_frac=0.27, tip=0.84, tip_pow=2.4, emarg=0.72,
                                    emarg_in=0.32, tip_floor=0.5, root_floor=0.07, notch=0.6, rise_at=0.42, n=16,
                                    ridge=1.6, vee=0.1, ridge_dir=1.0))
SECONDARY = ((3.773, 0.877, 0.2), dict(curl=0.089, cup=0.12, lead_frac=0.4, tip=0.8, tip_pow=2.8, tip_floor=0.2,
                                       root_floor=0.07, notch=0.6, rise_at=0.34, n=16, ridge=1.6, vee=0.1,
                                       ridge_dir=1.0, splits=((0.62, 0.3),)))
COVERT = ((1.498, 0.539, 0.2), dict(curl=0.04, cup=0.05, sweep=0.06, lead_frac=0.42, tip=0.712, tip_pow=1.7,
                                    calamus=0.0, rise_at=0.22, root_floor=0.55, tip_floor=0.0, n=10, m=6,
                                    ridge=1.25, vane=0.9, ridge_dir=1.0, butt=0.14, vee=0.08))


def _shot(mesh, centre, radius, size, az, el, material):
    m = Mesh(np.asarray(mesh.vertices) - np.asarray(centre), mesh.faces).transform(po.MH_TO_Z_UP)
    return po.render_portrait([po.Shell(m, material)], size=size, azimuth=az, elevation=el,
                              fit=(np.zeros(3), radius), cache=False)


def _timed(fn, *a, **k):
    t = time.perf_counter()
    for _ in range(5):
        out = fn(*a, **k)
    return out, (time.perf_counter() - t) / 5 * 1000.0


def feathers(out, detail=1.0):
    mat = K.material_named("feathered")
    rows = []
    for name, (a, k) in (("primary", PRIMARY), ("secondary", SECONDARY), ("covert", COVERT)):
        sheet, t_s = _timed(W._blade, *a, **k)
        grown, t_g = _timed(FE.barbed_blade, *a, **{**k, "detail": detail})
        spec = {**FE._DEFAULTS, **k, "length": a[0], "width": a[1], "thickness": a[2], "edge": W.EDGE,
                "cap": W.TIP_CAP, "detail": detail}
        print(f"{name}: sheet {len(sheet.faces)} triangles {t_s:.1f} ms | grown {len(grown.faces)} triangles "
              f"{t_g:.1f} ms ({FE.barb_count(spec)[0]} barbs a side)")
        r = 0.29 * a[0] + 0.25 * a[1]
        cells = [np.concatenate([_shot(m, (a[0] / 2, 0, 0), r, (760, 250), 0, 78, mat),
                                 _shot(m, (a[0] / 2, 0, 0), r, (760, 210), 25, 28, mat)]) for m in (sheet, grown)]
        rows.append(np.concatenate(cells, axis=1))
    po.write_png(out, np.concatenate(rows))


def strands(out):
    a, k = SECONDARY
    spec = {**FE._DEFAULTS, **k, "length": a[0], "width": a[1], "thickness": a[2], "edge": W.EDGE, "cap": W.TIP_CAP}
    B, _ = FE.barb_count(spec)
    R = 2 * B + 1
    s = np.arange(R) / float(R)
    barbs = FE.barb_strands(s, [a[0]], [spec["sweep"]], W.BARB_ANGLE, [1.2 * a[1] + 0.25 * a[0]])
    p = {key: np.array([[float(spec[key])]]) for key in FE._NUMBERS}
    side = barbs.attrs["side"]

    def inside(q):
        t = q[:, 0] / a[0]
        wl, wt = FE._outline(np.clip(t, 0.0, 1.0)[None, :], p)
        off = q[:, 2] - spec["sweep"] * a[0] * t ** 2
        return np.minimum(np.where(np.repeat(side, barbs.steps) < 0, wl[0] + off, wt[0] - off), a[0] - q[:, 0])

    cut = H.trim(barbs, inside, at=np.linspace(0.0, 1.0, 6))
    shaft = H.Strands(np.stack([np.linspace(0, a[0], 12), np.zeros(12), spec["sweep"] * a[0] * np.linspace(0, 1, 12) ** 2],
                               axis=1)[None])
    mat, bone = K.material_named("feathered"), K.material_named("bone")
    grown = FE.barbed_blade(*a, **k)
    panels = []
    for st in (barbs, cut):
        tubes = H.tube_mesh(st, 0.012, sides=5, taper=0.8)
        rach = H.tube_mesh(shaft, 0.035, sides=6, taper=0.4)
        m = Mesh.merge([tubes, rach])
        mm = Mesh(m.vertices - [a[0] / 2, 0, 0], m.faces).transform(po.MH_TO_Z_UP)
        panels.append(po.render_portrait([po.Shell(mm, bone)], size=(700, 300), azimuth=0, elevation=78,
                                         fit=(np.zeros(3), 0.34 * a[0]), cache=False))
        print(f"{len(st)} strands as tubes: {len(tubes.faces)} triangles")
    panels.append(_shot(grown, (a[0] / 2, 0, 0), 0.34 * a[0], (700, 300), 0, 78, mat))
    print(f"the same strands zipped: {len(grown.faces)} triangles")
    po.write_png(out, np.concatenate(panels))


def systems(out):
    ball = subdivided_mesh(*cube_cage(2.0), levels=3)
    skin, bone, hair = K.material_named("skin"), K.material_named("bone"), K.material_named("hair:chestnut")
    t = time.perf_counter()
    ps = PT.emit_surface(ball, 400, seed=1, flow=(0.0, -1.0, 0.0))
    t_emit = (time.perf_counter() - t) * 1000
    from minimaster.character.scatter import item_mesh
    spike = item_mesh("spike", 0.34, 0.12, 0.1, tip_floor=0.03)
    stand = PT.Particles(ps.position, np.stack([ps.up, -ps.heading, ps.frame[:, :, 2]], axis=2), ps.size)   # +X out
    t = time.perf_counter()
    spikes = PT.instance(spike, PT.jitter(stand, 3, yaw=0.0, tilt=14.0, size=0.35))
    t_inst = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    grown = H.grow(ps.position, ps.up, 0.75, steps=9, steer=lambda p, d, s: d + [0.0, -0.33, 0.0])
    tubes = H.tube_mesh(H.roughen(grown, 0.03, seed=2), 0.016, sides=5)
    t_hair = (time.perf_counter() - t) * 1000
    print(f"400 particles on a {len(ball.faces)}-triangle ball: {t_emit:.0f} ms | 400 instances: "
          f"{len(spikes.faces)} triangles in {t_inst:.1f} ms | 400 strands grown and meshed: "
          f"{len(tubes.faces)} triangles in {t_hair:.0f} ms")
    cells = []
    for extra, mat in ((spikes, bone), (tubes, hair)):
        shells = [po.Shell(ball.transform(po.MH_TO_Z_UP), skin), po.Shell(extra.transform(po.MH_TO_Z_UP), mat)]
        cells.append(po.render_portrait(shells, size=(520, 480), azimuth=25, elevation=12, fit=(np.zeros(3), 1.75),
                                        cache=False))
    po.write_png(out, np.concatenate(cells, axis=1))


if __name__ == "__main__":
    what, out = sys.argv[1], sys.argv[2]
    if what == "feathers":
        feathers(out, float(sys.argv[sys.argv.index("--detail") + 1]) if "--detail" in sys.argv else 1.0)
    elif what == "strands":
        strands(out)
    else:
        systems(out)
    print("wrote", os.path.basename(out))
