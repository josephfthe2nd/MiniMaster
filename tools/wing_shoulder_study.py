"""Compare shoulder pivots on the experimental relief wing, using the reference figure.

    python -m pip install manifold3d==3.5.1 "Pillow>=10.1"
    python tools/wing_shoulder_study.py out/shoulder.png --sweep

The picture is at half fold. The optional sweep measures nine poses. Distances
are in mm on a 32 mm figure. Body contact is the intersection of the complete
wing with the posed body, including the skin moulded at its attachment.
"""

import argparse
from pathlib import Path

import manifold3d as md
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from relief_wing_study import ReliefWing, figure
from minimaster.character import portrait, shading
from minimaster.character.wingrig import WingRig, euler_from_matrix
from minimaster.core.math3d import euler_rotation, transform_points
from minimaster.core.mesh import Mesh


class ShoulderPivotRig(WingRig):
    """Turn the arm about the head in the glenoid on the shoulder blade."""

    def pose_from_controls(self, controls=None, **kw):
        pose = super().pose_from_controls(controls, **kw)
        if "root" in pose:
            # The old shoulder inherits the root rotation. Apply their product
            # at the shoulder so the arm keeps its orientation about the new pivot.
            root_turn = euler_rotation(*pose.pop("root"))
            arm_turn = euler_rotation(*pose["shoulder"])
            pose["shoulder"] = euler_from_matrix(root_turn @ arm_turn)
        return pose


def solid(mesh):
    result = md.Manifold(md.Mesh64(np.ascontiguousarray(mesh.vertices),
                                   np.ascontiguousarray(mesh.faces, dtype=np.uint64)))
    if result.status() != md.Error.NoError:
        raise ValueError(f"Cannot measure this mesh: {result.status()}")
    return result


def measure(placed, fold, body, wing):
    scale = 32 / 17.5
    controls = {"fold": fold}
    posed = placed.mesh(controls)
    bones = placed.model.sleeve_meshes(posed, controls)
    # The humerus is the shaft followed by the head in WingRoot.sleeve_meshes.
    shaft, humerus = bones["shaft"], bones["bone"]
    count = len(shaft.vertices)
    head = Mesh(humerus.vertices[count:], humerus.faces[len(shaft.faces):] - count)
    rim = placed.model.glenoid()
    gap = solid(head).min_gap(solid(rim), 10.) * scale
    body_solid = solid(body)
    contact = solid(wing) ^ body_solid
    wrist = dict(placed.rig.piece_meshes(**controls))["wrist_knob"].transform(portrait.MH_TO_Z_UP)
    clearance = solid(wrist).min_gap(body_solid, 20.) * scale
    print(f"{fold:5.3f}  {gap:8.4f}  {clearance:8.4f}  {len(contact.decompose()):8d}"
          f"  {contact.volume() * scale ** 3:10.4f}")
    return gap, humerus, rim


def half_fold_views(placed, body, wing, humerus, rim):
    xf = portrait.MH_TO_Z_UP
    skin = shading.Material(base_color="#bcb2a6", roughness=.78)
    surface = [portrait.Shell(body, skin), portrait.Shell(wing, skin)]
    bones = [portrait.Shell(rim.transform(xf), shading.Material(base_color="#167675")),
             portrait.Shell(humerus.transform(xf), shading.Material(base_color="#e3bc86"))]
    rest = placed.rig.joint_positions()["shoulder"]
    fits = [(placed.anchor + [1.65, 0, -.6], 3.75), (rest + [.32, -.30, -.03], 1.19)]
    views = []
    for azimuth, elevation in [(180, 9), (90, 4)]:
        for i, (centre, radius) in enumerate(fits):
            camera = dict(size=(470, 315), fit=(transform_points(xf, centre), radius),
                          azimuth=azimuth, elevation=elevation, cache=False)
            rgb = portrait.render_portrait(surface, **camera)[:, :, :3].copy()
            if i == 1:
                deep = portrait.render_portrait(bones, background=None, **camera)
                alpha = deep[:, :, 3:4] / 255.
                rgb[:] = (1 - alpha) * .38 * rgb + alpha * deep[:, :, :3]
            views.append(Image.fromarray(rgb))
    return views


def compare(out, sweep):
    page = Image.new("RGB", (1532, 724), "#f4f1eb")
    draw = ImageDraw.Draw(page)

    def label(x, y, text, size):
        draw.text((x, y), text, fill="#27363b", font=ImageFont.load_default(size=size))

    label(26, 14, "The feathered shoulder at half fold", 34)
    label(27, 64, "The socket rim is teal. The humerus is ivory. Each column uses the same camera for both poses.", 19)
    columns = [26 + 372 * i for i in range(4)]
    for x, title in zip(columns, ["Rear", "Rear shoulder", "Side", "Side shoulder"]):
        label(x, 105, title, 23)
    for i, title in enumerate(["Existing fold", "Pivot at the shoulder"]):
        placed, body_at = figure()
        if i == 1:
            native = placed.rig.rig
            placed.rig.rig = ShoulderPivotRig(native.skeleton, native.pieces)
        relief = ReliefWing(placed)
        print(title, "\n fold   gap(mm) wrist(mm)   patches volume(mm3)")
        for fold in np.linspace(0, 1, 9) if sweep else [.5]:
            body, _ = body_at(fold)
            wing = relief.pose(fold).transform(portrait.MH_TO_Z_UP @ placed.matrix)
            gap, humerus, rim = measure(placed, fold, body, wing)
            if fold == .5:
                y = 151 + 285 * i
                label(26, y, f"{title}     |     Socket gap {gap:.2f} mm", 21)
                for x, view in zip(columns, half_fold_views(placed, body, wing, humerus, rim)):
                    page.paste(view.resize((362, 243), Image.Resampling.LANCZOS), (x, y + 31))
    out.parent.mkdir(parents=True, exist_ok=True)
    page.save(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    compare(args.out, args.sweep)
