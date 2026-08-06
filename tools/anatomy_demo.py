"""Before/after: what carrying a pair of wings does to the body itself.

Front and back, plain vs. adapted, identical framing and lighting so the only
difference in the picture is the morph the appendage imposed.
"""
import numpy as np

from minimaster.character import appendages as A
from minimaster.character import creature_anatomy as ca
from minimaster.character import mhbase as mh
from minimaster.character import portrait as po
from minimaster.character.sliders import Character, SliderCatalog
from minimaster.core.mesh import Mesh

base, lib = mh.load()
cat = SliderCatalog(lib.names)

WINGS = [A.Appendage("wing", anchor_point=(0.84, 4.39, -0.73), mirror=True,
                     up_hint=(1.0, 1.5, 0.0), scale=1.0,
                     params=dict(style="membrane", span=13.0))]

plain = Character(name="plain")
plain.macro.update(gender=1.0, muscle=0.55)
winged = Character(name="winged")
winged.macro.update(gender=1.0, muscle=0.55)
notes = ca.apply_to(winged, WINGS, cat)

SIZE = (520, 760)
meshes = {}
for name, c in (("plain", plain), ("winged", winged)):
    v = lib.apply(base.verts, c.target_weights(cat))
    meshes[name] = (v, po.character_shells(base, v, levels=1, with_eyes=True))

# one frame for every panel, taken from the plain body
lo, hi = meshes["plain"][1][0].mesh.bounds
FIT = ((lo + hi) / 2.0, float(np.linalg.norm(hi - lo)) / 2.0)

rows = []
for az in (34.0, 214.0):                      # front three-quarter, then back
    row = [po.render_portrait(meshes[n][1], size=SIZE, azimuth=az,
                              elevation=4, fit=FIT) for n in ("plain", "winged")]
    rows.append(np.concatenate(row, axis=1))
sheet = np.concatenate(rows, axis=0)
po.write_png("docs/images/anatomy_before_after.png", sheet)
print("wrote docs/images/anatomy_before_after.png", sheet.shape)

# the numbers that go with the picture
v = meshes["winged"][0]
cv, cq = base.body_cage(v)
body = Mesh(cv, np.vstack([cq[:, [0, 1, 2]], cq[:, [0, 2, 3]]]))
wing_meshes = [m for _, m in WINGS[0].build(base, v)]
print()
print(ca.flight_report(body.volume(), ca.wingspan_of(wing_meshes)).as_text())
print()
for k, val in sorted(winged.sliders.items()):
    print(f"  {k:40s} {val:+.3f}")
print(f"  macro weight                             "
      f"{plain.macro['weight']:.2f} -> {winged.macro['weight']:.2f}")
for n in notes:
    print(" *", n)
