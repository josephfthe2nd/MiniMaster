"""Render the experimental elbow and measure its sleeve through the fold.

    python tools/wing_elbow_study.py out/elbow.png --sweep

Uses the reference figure and the seated shoulder from wing_shoulder_study.
The isolated sleeve and the feathered surface use the same elbow envelope.
"""
import argparse
from pathlib import Path

import manifold3d as md
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from relief_wing_study import ReliefWing, elbow_pad, figure, section_hulls
from wing_shoulder_study import ShoulderPivotRig, solid
from minimaster.character import collar, portrait, shading
from minimaster.core.math3d import transform_points
from minimaster.core.mesh import Mesh


def unit(vector):
    return vector / np.linalg.norm(vector)


def sleeve_surface(original, rings, radius):
    pad = elbow_pad(radius, rings[len(rings) // 2])
    if pad is None:
        return original
    shells = section_hulls(rings, pad)
    shells.append(md.Manifold.hull_points(np.vstack((rings[0], original.vertices[-2]))))
    merged = md.Manifold.batch_boolean(shells, md.OpType.Add).simplify(.000002)
    if merged.status() != md.Error.NoError or len(merged.decompose()) != 1:
        raise ValueError("The elbow sleeve must form one connected solid")
    result = merged.to_mesh64()
    return Mesh(result.vert_properties[:, :3].copy(), result.tri_verts)


def elbow_frame(joints):
    shoulder, elbow, wrist = (joints[name] for name in ("shoulder", "elbow", "wrist"))
    upper, lower = unit(shoulder - elbow), unit(wrist - elbow)
    x, z = unit(lower - upper), unit(lower + upper)
    axes = np.stack((x, np.cross(z, x), z))
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = axes, -axes @ elbow
    return matrix


def elbow_view(mesh, frame, crossing):
    view = mesh.transform(frame)
    hit = np.unique(crossing)
    keep = np.ones(len(mesh.faces), dtype=bool)
    keep[hit] = False
    shells = [portrait.Shell(Mesh(view.vertices, view.faces[keep]), CLAY)]
    if len(hit):
        shells.append(portrait.Shell(Mesh(view.vertices, view.faces[hit]),
                                     shading.Material(base_color="#aa453e")))
    return render(shells, [0, 0, .65], 1.15, 10, 11)


CLAY = shading.Material(base_color="#c9bdab", roughness=.74)


def render(shells, centre, radius, azimuth, elevation):
    rgb = portrait.render_portrait(shells, size=(480, 320), fit=(centre, radius),
                                   azimuth=azimuth, elevation=elevation, cache=False)
    return Image.fromarray(rgb[:, :, :3])


def compare(out, sweep):
    p, body_at = figure()
    rig = ShoulderPivotRig(p.rig.rig.skeleton, p.rig.rig.pieces)
    p.rig.rig = rig
    wings = (ReliefWing(p), ReliefWing(p, round_elbow=True))
    page = Image.new("RGB", (1478, 861), "#f4f1eb")
    draw = ImageDraw.Draw(page)

    def label(x, y, text, size):
        draw.text((x, y), text, fill="#26363c", font=ImageFont.load_default(size=size))

    label(24, 16, "The feathered wing at the elbow", 35)
    label(25, 68, "Red marks crossing faces of the isolated arm sleeve. Both rows use the seated shoulder.", 20)
    for x, title in zip((24, 510, 996), ("Half folded", "Fully folded", "Folded on the figure")):
        label(x, 118, title, 23)
    for row, name in enumerate(("Existing sleeve", "Rounded sleeve")):
        label(24, 160 + row * 344, name, 21)

    print("Fold    sleeve    triangle pairs crossing    outward depth (mm)")
    folds = np.linspace(0, 1, 17) if sweep else [.5, 1.]
    for fold in folds:
        pose = rig.pose_from_controls(fold=fold)
        frame = elbow_frame(rig.posed_joints(pose))
        original = rig.pose_piece("arm", pose)
        rings = original.vertices[:-2].reshape(-1, p.model.profile.sides, 3)
        rounded = sleeve_surface(original, rings, wings[1].elbow_radius)
        if sweep:
            controls = {"fold": fold}
            layers = p.model.sleeve_meshes(p.mesh(controls), controls)
            outer = solid(rounded.transform(p.matrix))
            for name in ("muscle", "shaft"):
                if (solid(layers[name]) - outer).volume() > 1e-8:
                    raise ValueError(f"The {name} protrudes at fold {fold:g}")
        centre = rings[len(rings) // 2].mean(axis=0)
        for row, mesh in enumerate((original, rounded)):
            crossing = collar.crossing_pairs(mesh.vertices, mesh.faces)
            # Measure the outer contour in the plane of the bend. The same ray
            # starts at the posed skin ring centre in both versions of this pose.
            hits = solid(mesh).ray_cast(centre, centre - 4 * wings[1].elbow_radius * frame[2, :3])
            depth = np.linalg.norm(hits[0].position - centre) * 32 / 17.5
            name = "rounded" if row else "existing"
            print(f"{fold:.4f}  {name:8}  {len(crossing):8}  {depth:17.3f}")
            y = 195 + row * 344
            if fold in (.5, 1.):
                page.paste(elbow_view(mesh, frame, crossing).resize((456, 304), Image.Resampling.LANCZOS),
                           (24 if fold == .5 else 510, y))
        if fold == 1.:
            body, _ = body_at(fold)
            xf = portrait.MH_TO_Z_UP
            for row, wing in enumerate(wings):
                part = wing.pose(fold).transform(xf @ p.matrix)
                view = render([portrait.Shell(body, CLAY), portrait.Shell(part, CLAY)],
                              transform_points(xf, [1.3, 3.65, -.75]), 5.15, 156, 12)
                page.paste(view.resize((456, 304), Image.Resampling.LANCZOS), (996, 195 + row * 344))
    out.parent.mkdir(parents=True, exist_ok=True)
    page.save(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    compare(args.out, args.sweep)
