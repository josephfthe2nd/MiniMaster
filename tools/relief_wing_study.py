"""Render one relief wing on the reference figure. This study runs outside the character builder.

    python -m pip install manifold3d==3.5.1
    python tools/relief_wing_study.py out/relief

The feather relief keeps its indices through the fold. At each pose the skin over
the native arm and hand joins it into one solid. The boolean is local to this study. The section and the feather
laps are geometry, including the underside. The pictures use one material.
"""

from pathlib import Path
import sys

import manifold3d as md
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import wing_view as viewer
from wing_root_view import MATERIALS
from minimaster.character import kinds, mhbase, portrait, weld
from minimaster.character.wings import ARM_SIDES
from minimaster.character.sliders import Character, SliderCatalog
from minimaster.core.mesh import Mesh
from minimaster.core.math3d import rot_x, rot_y, transform_points


def smooth(x):
    t = np.clip(x, 0, 1)
    return t * t * (3 - 2 * t)


def rail(points, u):
    """Interpolate the authored span stations with local cubic tangents."""
    points = np.asarray(points)
    at, coords = points[:, 0], points[:, 1:]
    slopes = np.gradient(coords, at, axis=0)
    i = np.clip(np.searchsorted(at, u, side="right") - 1, 0, len(at) - 2)
    width = (at[i + 1] - at[i])[:, None]
    t = (u - at[i])[:, None] / width
    return ((1 - smooth(t)) * coords[i] + smooth(t) * coords[i + 1]
            + width * t * (1 - t) * ((1 - t) * slopes[i] - t * slopes[i + 1]))


# The inner stations carry the scapulars onto the shoulder. Beyond the wrist the
# stations rotate through the hand fan. Each pair of numbers is span and chord.
SPREAD_NOSE = [(0, -.85, .84), (.12, -.10, -.35), (.30, 2.10, .03),
               (.49, 4.68, .13), (.66, 6.68, -.17), (.84, 7.90, -.05), (1, 8.68, -.18)]
SPREAD_TAIL = [(0, -1.03, 1.36), (.10, -.93, 3.09), (.25, .88, 4.62),
               (.43, 4.20, 5.26), (.56, 6.69, 4.64), (.68, 8.99, 5.05),
               (.80, 11.51, 4.53), (.92, 13.24, 2.90), (1, 13.05, .18)]

# The folded stations are in the back view, measured sideways and up from the
# socket. Their dorsal surface faces behind the figure. The wrist crowns the shield.
FOLDED_NOSE = [(0, -.54, -.05), (.15, -.27, .21), (.38, .46, .70),
               (.58, 1.12, 1.16), (.72, 1.61, 1.08), (.88, 1.94, .44), (1, 1.92, -.96)]
FOLDED_TAIL = [(0, -.65, -.56), (.13, -.70, -1.57), (.35, -.53, -3.49),
               (.57, -.06, -4.70), (.74, .42, -5.94), (.90, .92, -6.30), (1, 1.25, -6.04)]


def feather_surface(u, t, underside, folded_xy=None):
    """Lay the broad covert tips across the vane and bury their bases in the next row."""
    phase = 10 * u + .13 * np.sin(2 * np.pi * u) * (1 - t) - .19 * np.sin(np.pi * t)
    vane = phase % 1
    shaft = np.exp(-((vane - .66) / .080) ** 2)
    flight = (.061 * np.sin(np.pi * vane) ** 1.4 + .043 * shaft
              - .037 * np.exp(-((vane - .035) / .072) ** 2))
    surface = flight * smooth((t - .28) / .3) * (.35 + .65 * smooth((1.01 - t) / .20))
    if folded_xy is not None:
        # The coverts overlap more deeply when stowed. Sculpt only the tips that
        # stay exposed, in shoulder coordinates so they keep their breadth.
        x, y = folded_xy[:, :, 0], folded_xy[:, :, 1]
        surface *= .32
        for tips, width, length, height in [
            ([(-.37, -1.47), (.12, -2.04), (.65, -2.13), (1.27, -1.72), (1.81, -1.23)], .39, 2.25, .12),
            ([(-.32, -.63), (.21, -.90), (.82, -.99), (1.37, -.71), (1.88, -.42)], .36, 1.65, .11),
            ([(-.18, -.12), (.38, -.13), (.96, -.02), (1.52, .18)], .34, 1.20, .086),
        ]:
            patches = np.zeros_like(x)
            for cx, tip in tips:
                h = (y - tip) / length
                side = (x - cx - .22 * h) / width
                end = 1 - np.sqrt(np.maximum(0, 1 - side ** 2))
                mask = smooth((h - .53 * end) / .13) * smooth((1 - abs(side)) / .29)
                mask *= smooth((1.20 - h) / .38)
                patches = np.maximum(patches, mask * height * (1 - .45 * np.minimum(1, side ** 2)))
            surface += patches
        return surface * (.45 if underside else 1.)
    hand = smooth((u - .57) / .41)
    for count, start, length, offset, height in (
        (7, .09, .62, .28, .125), (8, .03, .41, .73, .103), (9, -.04, .29, .16, .068)
    ):
        if underside:
            length *= .88
        # The shoulder feathers run farther down the back. The outer coverts shorten
        # onto the hand, exposing the broad bases of the primaries.
        reach = length * (1 - .43 * hand + .22 * np.exp(-((u - .16) / .16) ** 2))
        centres = (np.arange(count + 2) - .45 + offset) / count
        patches = np.zeros_like(surface)
        for i, centre in enumerate(centres):
            lean = -.44 * (1 - 2.3 * centre) * t + .12 * np.sin(i * 2.4) * t
            across = ((u - centre) * count + lean) / (.64 + .045 * np.cos(i * 3.1))
            reach_i = reach * (1 + .065 * np.sin(i * 2.3 + offset))
            tip = start + reach_i * np.sqrt(np.maximum(0, 1 - across ** 2))
            mask = smooth((tip - t) / .034) * smooth((1 - np.abs(across)) / .17)
            dome = 1 - .38 * np.minimum(1, across ** 2)
            patches = np.maximum(patches, height * mask * dome)
        surface += patches
    return surface * (.52 if underside else 1.0)


def shell_faces(shape):
    top = np.arange(np.prod(shape)).reshape(shape)
    quads = np.stack((top[:-1, :-1], top[1:, :-1], top[1:, 1:], top[:-1, 1:]), axis=-1).reshape(-1, 4)
    rim = np.r_[top[:, 0], top[-1, 1:], top[-2::-1, -1], top[0, -2:0:-1]]
    adjacent = np.roll(rim, -1)
    walls = np.stack((adjacent, rim, rim + top.size, adjacent + top.size), axis=1)
    quads = np.vstack((quads, quads[:, ::-1] + top.size, walls))
    return np.vstack((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]))


def section(u, t, closed):
    """The leading and trailing rails, the exposed chord, and the heights of its two faces."""
    nose, tail = (FOLDED_NOSE, FOLDED_TAIL) if closed else (SPREAD_NOSE, SPREAD_TAIL)
    front, rear = rail(nose, u[:, 0]), rail(tail, u[:, 0])
    across = abs((u * 10) % 1 - .5) * 2
    slot = .049 * across ** 2.5 + .60 * smooth((u - .54) / .33) * across ** 1.65
    if closed:
        slot *= .12
    along = t * (1 - slot) * (1 + .35 * slot * (1 - t))
    xy = front[:, None] + (rear - front)[:, None] * along[:, :, None]

    muscle = np.sqrt(t / .105) * np.exp(.5 * (1 - t / .105)) * (1 - t) ** 1.4
    arm = .82 * (1 - .72 * smooth((u - .5) / .49))
    thick = .21 + arm * muscle
    camber = (.24 + .13 * (1 - u)) * np.sin(t * np.pi) * (1 - .72 * u)
    if closed:
        depth = .65 - 1.23 * smooth(u / .55) - .12 * np.sin(t * np.pi) + .37 * t + .24 * u ** 2
    else:
        lift = .34 * np.sin(np.pi * along) * smooth((u - .4) / .45) * t
        depth = .0072 * np.maximum(0, xy[:, :, 0]) ** 2 - lift
    relief = smooth(u / .085)
    dorsal = depth - camber - thick / 2 - relief * feather_surface(u, t, False, xy if closed else None)
    ventral = depth - camber + thick / 2 + relief * feather_surface(u, t, True, xy if closed else None)
    # Align the long axes of the primaries before interpolating the rails. This
    # lets the changing slots slide along each chord without it crossing its neighbour.
    if closed:
        turn = rot_y(50)[np.ix_([0, 2], [0, 2])]
        front = (front * [1, -1]) @ turn.T
        rear = (rear * [1, -1]) @ turn.T
    return front[:, None], rear[:, None], along[:, :, None], np.stack((dorsal, ventral))


def elbow_pad(radius, ring):
    """A rounded envelope for the bent elbow, sized from its spread skin section."""
    centre = ring.mean(axis=0)
    radial = ring - centre
    short_radius = np.linalg.norm(radial, axis=1).min()
    # A linearly skinned ring narrows roughly as cos(half the bend). Grow the
    # longitudinal axis of the pad as sin(half the bend) while keeping its girth.
    flex = np.sqrt(np.clip(1 - (short_radius / radius) ** 2, 0, 1))
    if flex < 1e-6:
        return None
    normal = np.linalg.svd(radial, full_matrices=False)[2][-1]
    across = radial[0] - normal * (radial[0] @ normal)
    across /= np.linalg.norm(across)
    sphere = md.Manifold.sphere(radius, 32).to_mesh64().vert_properties[:, :3]
    return centre + sphere @ np.array([across, np.cross(normal, across), flex * normal])


def section_hulls(contours, pad=None):
    """Envelope each pair of skin sections and carry both sides of the elbow over its pad."""
    middle = len(contours) // 2
    hulls = []
    for i, (a, b) in enumerate(zip(contours, contours[1:])):
        points = (a, b, pad) if pad is not None and i in (middle - 1, middle) else (a, b)
        hulls.append(md.Manifold.hull_points(np.vstack(points)))
    return hulls


class ReliefWing:
    """One surface whose span stations gather into the folded pose on the reference shoulder."""

    def __init__(self, placed, round_elbow=False):
        self.arm_sides = placed.model.profile.sides
        self.rig = placed.rig.rig
        self.elbow_radius = None
        if round_elbow:
            start, end = self.rig.ranges["arm"]
            rings = self.rig.rest_mesh.vertices[start:end - 2].reshape(-1, self.arm_sides, 3)
            middle = rings[len(rings) // 2]
            self.elbow_radius = np.linalg.norm(middle - middle.mean(axis=0), axis=1).mean()
        u, t = np.meshgrid(np.linspace(0, 1, 181), .5 - .5 * np.cos(np.linspace(0, np.pi, 89)), indexing="ij")
        self.spread = section(u, t, False)
        self.folded = section(u, t, True)
        self.grid_shape = u.shape
        self.faces = shell_faces(u.shape)
        # Translate the authored back view into this wing's local shoulder frame.
        inv = np.linalg.inv(placed.matrix)
        self.closed_rotation = inv[:3, :3] @ rot_x(90) @ rot_y(-50)
        self.offset = transform_points(inv, placed.anchor)
        self.root_seat = (1 - smooth(u / .36)).ravel()[:, None] * (inv[:3, :3] @ [-.65, 0, .95])
        span = np.tile(u.ravel(), 2)
        boundaries = [0, .20, .39, .61, .77]
        transfer = [smooth((span - a) / (b - a)) for a, b in zip(boundaries, boundaries[1:])]
        self.weights = [1 - transfer[0], *(a - b for a, b in zip(transfer, transfer[1:])), transfer[-1]]
        self.rest = self.pose(0)

    def pose(self, fold, **controls):
        if not 0 <= fold <= 1:
            raise ValueError("fold must lie between zero and one")
        f = smooth(fold)
        front, rear, along, depths = ((1 - f) * a + f * b for a, b in zip(self.spread, self.folded))
        xy = front + (rear - front) * along
        v = np.empty((*depths.shape, 3))
        v[:, :, :, 0], v[:, :, :, 2], v[:, :, :, 1] = xy[:, :, 0], xy[:, :, 1], depths
        # The polar factor turns the entire fan as its plane goes from spread to
        # lying on the back. It preserves the authored thickness during the turn.
        left, _, right = np.linalg.svd((1 - f) * np.eye(3) + f * self.closed_rotation)
        vertices = v.reshape(-1, 3) @ (left @ right).T + f * self.offset
        vertices += (1 - f) * np.tile(self.root_seat, (2, 1))
        if controls:
            base = self.rig.skin_matrices(self.rig.pose_from_controls(fold=fold))
            posed = self.rig.skin_matrices(self.rig.pose_from_controls(fold=fold, **controls))
            moved = self.weights[0][:, None] * vertices
            for joint, weight in zip(("shoulder", "elbow", "wrist", "d3_mcp"), self.weights[1:]):
                moved += weight[:, None] * transform_points(posed[joint] @ np.linalg.inv(base[joint]), vertices)
            vertices = moved
        # The arm must keep the girth and position of the actual muscle sleeve.
        # Shape the volume between its posed rings and the feather bed, including
        # the folded elbow beneath the coverts, then remove the internal surfaces.
        planes = vertices.reshape(2, *self.grid_shape, 3)
        bed = planes.mean(axis=0)
        native = self.rig.pose_mesh(self.rig.pose_from_controls(fold=fold, **controls))
        arm = self.rig.ranges["arm"]
        hand = self.rig.ranges["hand"]
        arm_rings = native.vertices[arm[0]:arm[1] - 2].reshape(-1, self.arm_sides, 3)
        hand_rings = native.vertices[hand[0]:hand[1] - 2].reshape(-1, ARM_SIDES, 3)
        # As the ulna doubles back, the wrist forms the crown of the folded wing.
        # Carry the feather bases over it before joining the underlying sleeve.
        u_axis = np.linspace(0, 1, bed.shape[0])
        t_axis = .5 - .5 * np.cos(np.linspace(0, np.pi, bed.shape[1]))
        stations = [.11, .35, .69, .97]
        arm_path = np.array([arm_rings[0].mean(axis=0), arm_rings[len(arm_rings) // 2].mean(axis=0),
                             hand_rings[0].mean(axis=0), hand_rings[-1].mean(axis=0)])
        arm_chord = [.14, .17 + .45 * f, .13, .14]
        samples = [bed[np.argmin(abs(u_axis - u)), np.argmin(abs(t_axis - t))]
                   for u, t in zip(stations, arm_chord)]
        # In the wing plane a turn and a uniform scale keep neighbouring vanes
        # in order. Depth can follow each bone independently: the elbow tucks
        # behind the coverts while the wrist lifts their rounded crown.
        basis = left @ right
        flat = basis[:, [0, 2]]
        normal = basis[:, 1]
        origin = samples[0]
        source = (samples[2] - origin) @ flat
        target = (arm_path[2] - origin) @ flat
        follow = smooth(fold / .45)
        target = source + follow * .76 * (target - source)
        cosine = (source @ target) / (source @ source)
        sine = (source[0] * target[1] - source[1] * target[0]) / (source @ source)
        gain = min(1, 1.08 / np.hypot(cosine, sine))
        cosine, sine = gain * cosine, gain * sine
        warp = np.array([[cosine - 1, sine], [-sine, cosine - 1]])
        vertices += ((vertices - origin) @ flat) @ warp @ flat.T
        shift = ((arm_path - samples) @ normal) * [.10, .38, .88, .74]
        offsets = np.interp(u_axis, [0, *stations, 1], [0, *shift, shift[-1]])
        carry = (.16 + .84 * (1 - smooth((t_axis - .12) / .76))) * follow
        depth_shift = offsets[:, None] * carry[None, :]
        planes[0] += np.minimum(depth_shift, .22 * depth_shift)[:, :, None] * normal
        planes[1] += np.maximum(depth_shift, 1.20 * depth_shift)[:, :, None] * normal
        bed = planes.mean(axis=0)
        solids = [md.Manifold(md.Mesh64(vertices, self.faces.astype(np.uint64)))]
        pad = elbow_pad(self.elbow_radius, arm_rings[len(arm_rings) // 2]) if self.elbow_radius is not None else None
        for rings, along_span, along_chord, width, joint_pad in [
            (arm_rings, (.10, .35, .69), (.15, .16 + .44 * f, .15), .13, pad),
            (hand_rings, (.69, .84, .98), (.15, .14, .19), .085, None),
        ]:
            frac = np.linspace(0, 1, len(rings))
            rows = np.rint(np.interp(frac, [0, .5, 1], along_span) * (bed.shape[0] - 1)).astype(int)
            chord = np.interp(frac, [0, .5, 1], along_chord)
            # Interior chord samples give the sleeve a broad landing on the fan.
            cols = np.arccos(1 - 2 * np.clip(chord[:, None] + [-width, 0, width], 0, 1))
            cols = np.rint(cols / np.pi * (bed.shape[1] - 1)).astype(int)
            contours = np.concatenate((rings, bed[rows[:, None], cols]), axis=1)
            solids.extend(section_hulls(contours, joint_pad))
        solids.append(md.Manifold.hull_points(np.vstack((arm_rings[0], native.vertices[arm[1] - 2]))))
        fused = md.Manifold.batch_boolean(solids, md.OpType.Add).simplify(.000005)
        if fused.status() != md.Error.NoError or len(fused.decompose()) != 1:
            raise ValueError("The arm and feather bed did not form one solid")
        out = fused.to_mesh64()
        return Mesh(out.vert_properties[:, :3], out.tri_verts)


def figure():
    base, lib = mhbase.load()
    character = Character()
    character.macro.update(gender=1., muscle=.55)
    spec = kinds.add_from_text("wing:style=feathered", [], base=base, lib=lib)
    spec.side = "left"
    character.appendages.append(spec)
    verts = lib.apply(base.verts, character.effective_target_weights(SliderCatalog(lib.names)))
    placed, = character.instantiate_appendages(base, verts, lib)

    def body_at(fold):
        placed.controls = {"fold": fold}
        skin = character.shaped_body(base, verts, [placed])
        body = portrait.character_shells(base, skin, levels=1, with_eyes=False)[0].mesh
        if weld.seam(character) == "weld":
            body = weld.welded_shell_mesh(body, [placed])
        return body, skin

    return placed, body_at


def render(out):
    out.mkdir(parents=True, exist_ok=True)
    placed, body_at = figure()
    wing = ReliefWing(placed)
    material = viewer.PLUMAGE

    def shot(name, shells, centre, r, az, el, size):
        image = portrait.render_portrait(shells, fit=(centre, r), size=size,
                                        azimuth=az, elevation=el, cache=False)
        portrait.write_png(str(out / (name + ".png")), image)

    base_mesh = placed.rig.rig.rest_mesh.transform(viewer.WING_TO_WORLD)
    centre, r = viewer.frame([base_mesh])
    for name, mesh in [("current", base_mesh), ("relief", wing.rest.transform(viewer.WING_TO_WORLD))]:
        shells = [portrait.Shell(mesh, material)]
        shot(name + "-plan", shells, centre, r * .66, 0, 88, (900, 475))
        shot(name + "-section", shells, centre, r * .49, 0, 6, (900, 330))
    for fold, name in [(0, "spread"), (.5, "half"), (1, "folded")]:
        local = wing.pose(fold)
        print(name, len(local.faces), local.integrity_report(), flush=True)
        part = local.transform(placed.matrix).transform(portrait.MH_TO_Z_UP)
        # A single wing and a matching body material leave the join visible.
        body, shaped = body_at(fold)
        # The blend on the back and the deep dissection use this same posed body.
        shells = [portrait.Shell(body, material), portrait.Shell(part, material)]
        centre = portrait.MH_TO_Z_UP[:3, :3] @ np.array([3.8, 7.8, -1.7] if fold < 1 else [1.25, 3.6, -.75])
        shot(name + "-back", shells, centre, 10.6 if fold < 1 else 5.15, 180, 9, (700, 720))
        if fold in (0, 1):
            base_skin = placed.mesh({"fold": fold})
            plain = Mesh(base_skin.vertices, base_skin.faces).transform(portrait.MH_TO_Z_UP)
            current = portrait.Shell(plain, material)
            shot("current-" + name, [shells[0], current], centre, 10.6 if fold == 0 else 5.15,
                 180, 9, (700, 720))
            socket = transform_points(portrait.MH_TO_Z_UP, placed.anchor)
            layers = placed.kind.layers_on(placed, shaped, {"fold": fold})
            fit = (socket, 3.1)
            camera = dict(fit=fit, size=(640, 540), azimuth=162, elevation=15, cache=False)
            image = portrait.render_portrait(shells, **camera)
            image[:, :, :3] = (image[:, :, :3] * .54).astype(np.uint8)
            # Each deep layer is projected through the intact surface with the same
            # camera. A removed-face window also cuts the far side of a narrow torso.
            for layer in ("muscle", "bone"):
                mesh = layers[layer].transform(portrait.MH_TO_Z_UP)
                deep = portrait.render_portrait([portrait.Shell(mesh, MATERIALS[layer])],
                                                background=None, **camera)
                strength = .70 if layer == "muscle" else .95
                alpha = strength * deep[:, :, 3:4] / 255.
                image[:, :, :3] = (1 - alpha) * image[:, :, :3] + alpha * deep[:, :, :3]
            portrait.write_png(str(out / (name + "-anatomy.png")), image)
        if fold == 1:
            shot(name + "-oblique", shells, centre, 5.15, 146, 11, (700, 720))


if __name__ == "__main__":
    render(Path(sys.argv[1]))
