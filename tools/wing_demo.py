"""The wing sheets: both styles on a body, spread and posed, plus the rig strips.

    PYTHONPATH=. python tools/wing_demo.py            # writes docs/images/wings.png + wing_rig.png

``wings.png``     back and profile views of a plain body, membrane wings and
                  feathered wings, each spread and folded (a pose of the rig).
``wing_rig.png``  one wing, no body: the rig through rest / half fold / folded /
                  raised / lowered / curled, in three views.
"""
import sys

import numpy as np

sys.path.insert(0, ".")

from minimaster.character import appendages as A  # noqa: E402
from minimaster.character import creature_anatomy as ca  # noqa: E402
from minimaster.character import mhbase as mh  # noqa: E402
from minimaster.character import portrait as po  # noqa: E402
from minimaster.character import wings as W  # noqa: E402
from minimaster.character.sliders import Character, SliderCatalog  # noqa: E402
from minimaster.core.mesh import Mesh  # noqa: E402
from tools import wing_view as V  # noqa: E402


class Scene:
    """A body, plus a wing appendage built on it for each style."""

    def __init__(self):
        self.base, self.lib = mh.load()
        self.cat = SliderCatalog(self.lib.names)
        self.cache = {}

    def character(self, style):
        c = Character()
        c.macro.update(gender=1.0, muscle=0.55)
        app = A.back_wings(style) if style else None
        if app:
            ca.apply_to(c, [app], self.cat)
        v = self.lib.apply(self.base.verts, c.target_weights(self.cat))
        return app, v

    def render(self, style, pose, azimuth, elevation, size, fit):
        key = style
        if key not in self.cache:
            app, v = self.character(style)
            body = po.character_shells(self.base, v, levels=1, with_eyes=True)
            rigs = app.build_rigs(self.base, v) if app else []
            self.cache[key] = (body, rigs)
        body, rigs = self.cache[key]
        shells = list(body)
        for _, pw in rigs:
            m = pw.pose(**pose).transform(po.MH_TO_Z_UP)
            shells.append(po.Shell(m, V.MATS[style], base_colors=V.albedo(pw.rig, style)))
        return po.render_portrait(shells, size=size, azimuth=azimuth, elevation=elevation, fit=fit)

    def fit(self, scale):
        body, _ = self.cache.setdefault("body", self._body())
        lo, hi = body[0].mesh.bounds
        return (lo + hi) / 2.0, float(np.linalg.norm(hi - lo)) / 2.0 * scale

    def _body(self):
        _, v = self.character(None)
        return po.character_shells(self.base, v, levels=1, with_eyes=True), []


def panel_row(sc, columns, azimuth, elevation, size, scale):
    fit = sc.fit(scale)
    return np.concatenate([sc.render(style, pose, azimuth, elevation, size, fit)
                           for style, pose in columns], axis=1)


def main():
    sc = Scene()
    columns = [(None, {}), ("membrane", {}), ("feathered", {}),
               ("membrane", {"fold": 1.0}), ("feathered", {"fold": 1.0})]
    # straight from behind and slightly above: an off-axis camera turns one wing
    # face-on and the other edge-on and reads as an asymmetry
    back = panel_row(sc, columns, 180.0, 18.0, (440, 640), 1.55)
    side = panel_row(sc, columns, 270.0, 6.0, (440, 640), 1.30)
    sheet = np.concatenate([back, side], axis=0)
    po.write_png("docs/images/wings.png", sheet)
    print("wrote docs/images/wings.png", sheet.shape)

    # the rig on its own: poses of one wing (no body), three views each
    strips = []
    for style in ("membrane", "feathered"):
        rig = V.make_rig(style)
        poses = [{}, {"fold": 0.5}, {"fold": 1.0}, {"raise_": 45.0}, {"raise_": -40.0},
                 {"curl": 35.0}]
        meshes = [rig.pose_mesh(rig.pose_from_controls(**p)) for p in poses]
        colors = V.albedo(rig, style)
        # frame every pose off the spread wing so scale is comparable
        fit = V.frame([m.transform(V.WING_TO_WORLD) for m in meshes[:1]])
        cols = [np.concatenate([po.render_portrait(
            [po.Shell(m.transform(V.WING_TO_WORLD), V.MATS[style], base_colors=colors)],
            size=(300, 300), azimuth=V.VIEWS[v][0], elevation=V.VIEWS[v][1], fit=fit)
            for m in meshes], axis=1) for v in ("plan", "three_quarter", "front")]
        strips.append(np.concatenate(cols, axis=0))
    rig_sheet = np.concatenate(strips, axis=0)
    po.write_png("docs/images/wing_rig.png", rig_sheet)
    print("wrote docs/images/wing_rig.png", rig_sheet.shape)

    base, lib = sc.base, sc.lib
    v = sc.character(None)[1]
    cv, cq = base.body_cage(v)
    vol = Mesh(cv, np.vstack([cq[:, [0, 1, 2]], cq[:, [0, 2, 3]]])).volume()
    pair = [m for _, m in A.back_wings("membrane").build(base, v)]
    print()
    print(ca.flight_report(vol, ca.wingspan_of(pair)).as_text())


if __name__ == "__main__":
    main()
