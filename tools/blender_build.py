"""Blender build/refine backend for MiniMaster scenes.

Runs *inside* Blender (its bundled Python has numpy and can import the pure
MiniMaster kernel), so it works from the exact posed per-shape meshes and
colors the scene defines — no lossy STL round-trip:

    blender --background --python tools/blender_build.py -- --scene fig.mmp \
        --size medium --union boolean --stl out.stl --render out.png

It never runs in the tkinter app or the pytest suite (those stay bpy-free);
the pure-Python launcher ``minimaster.blender`` shells out to it.

Pipeline (each stage optional): scene -> colored mesh objects (scaled/placed
to match ``export.assemble``) -> boolean or voxel union into one solid ->
subdivision / smooth shading -> Cycles render and/or STL/GLB export.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy  # noqa: E402  (only present inside Blender)
import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


sys.path.insert(0, str(_repo_root()))

from minimaster.bases import build_base  # noqa: E402
from minimaster.export import SIZE_PRESETS  # noqa: E402
from minimaster.scene import Scene  # noqa: E402


# --------------------------------------------------------------------------
# color


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _hex_to_linear(color: str) -> tuple[float, float, float, float]:
    c = color.lstrip("#")
    rgb = [int(c[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
    return (*[_srgb_to_linear(v) for v in rgb], 1.0)


_MATERIALS: dict[str, "bpy.types.Material"] = {}


def _material(color: str) -> "bpy.types.Material":
    mat = _MATERIALS.get(color)
    if mat is None:
        mat = bpy.data.materials.new(name=f"mm_{color.lstrip('#')}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is None:  # unusual build; fall back to any BSDF node
            bsdf = next(
                (n for n in mat.node_tree.nodes if n.type.startswith("BSDF")), None
            )
        if bsdf is not None and "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = _hex_to_linear(color)
            bsdf.inputs["Roughness"].default_value = 0.62
            # a touch of specular so edges catch the light without looking wet
            spec = bsdf.inputs.get("Specular IOR Level") or bsdf.inputs.get("Specular")
            if spec is not None:
                spec.default_value = 0.25
        _MATERIALS[color] = mat
    return mat


# --------------------------------------------------------------------------
# scene -> objects


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in (bpy.data.meshes, bpy.data.materials):
        for item in list(block):
            block.remove(item)
    _MATERIALS.clear()


def _placement(scene: Scene, pose, height: float | None):
    """Uniform scale + translation matching export.assemble: scale the figure
    to the target height, center on XY, stand on z=0."""
    figure = scene.build_merged_mesh(pose)
    if not len(figure.faces):
        raise SystemExit("error: scene has no shapes to build")
    lo, hi = figure.bounds
    s = 1.0
    if height is not None:
        extent = hi[2] - lo[2]
        if extent <= 1e-9:
            raise SystemExit("error: figure has no height to scale")
        s = height / extent
    lo_s, hi_s = lo * s, hi * s
    center = (lo_s + hi_s) / 2.0
    offset = np.array([-center[0], -center[1], -lo_s[2]])
    return s, offset


def _add_mesh_object(name, verts, faces, color, collection):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    mesh.update()
    mesh.materials.append(_material(color))
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    return obj


def build_objects(scene: Scene, pose, height, with_base) -> list:
    s, offset = _placement(scene, pose, height)
    coll = bpy.context.scene.collection
    objs = []
    for shape, mesh in scene.build_shape_meshes(pose):
        verts = mesh.vertices * s + offset  # match assemble placement
        objs.append(_add_mesh_object(shape.name, verts, mesh.faces, shape.color, coll))
    if with_base:
        base = build_base(scene.base)  # own mm, unscaled (as in assemble)
        if base is not None and len(base.faces):
            col = scene.base.get("color", "#6e6a63")
            objs.append(_add_mesh_object("base", base.vertices, base.faces, col, coll))
    return objs


# --------------------------------------------------------------------------
# refine


def _apply_modifier(obj, mod):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)


def union_boolean(objs: list):
    """Self-union every shell into one solid in a single exact pass, then weld
    coincident verts and recalc normals outward. Preserves hard edges (good
    for gear / simple part sets); heavily interpenetrating bodies may still
    leave a few non-manifold edges — prefer ``voxel`` when watertightness is
    required for printing."""
    if not objs:
        return None
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.intersect_boolean(operation="UNION", solver="EXACT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=1e-4)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    return obj


def union_voxel(objs: list, voxel: float):
    """Join then voxel-remesh into a single watertight organic solid — always
    manifold, auto-smooths (good for bodies/creatures)."""
    if not objs:
        return None
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    mod = obj.modifiers.new(name="remesh", type="REMESH")
    mod.mode = "VOXEL"
    mod.voxel_size = voxel
    _apply_modifier(obj, mod)
    return obj


def add_subdivision(obj, levels: int):
    mod = obj.modifiers.new(name="subsurf", type="SUBSURF")
    mod.levels = levels
    mod.render_levels = levels
    _apply_modifier(obj, mod)


def shade_smooth(objs):
    for obj in objs:
        if obj.type != "MESH":
            continue
        for poly in obj.data.polygons:
            poly.use_smooth = True
        mod = obj.modifiers.new(name="smooth", type="EDGE_SPLIT")
        mod.split_angle = np.radians(40.0)
        _apply_modifier(obj, mod)


# --------------------------------------------------------------------------
# render


def _scene_bounds(objs):
    pts = []
    for obj in objs:
        for corner in obj.bound_box:
            pts.append(obj.matrix_world @ __import__("mathutils").Vector(corner))
    arr = np.array([[p.x, p.y, p.z] for p in pts])
    return arr.min(axis=0), arr.max(axis=0)


def setup_render(objs, azimuth, elevation, resolution, samples, transparent):
    import mathutils

    lo, hi = _scene_bounds(objs)
    center = mathutils.Vector((hi + lo) / 2.0)
    radius = float(np.linalg.norm(hi - lo)) / 2.0 or 1.0

    scn = bpy.context.scene
    scn.render.engine = "CYCLES"
    scn.cycles.device = "CPU"
    scn.cycles.samples = samples
    scn.cycles.use_denoising = False  # this build ships without OIDN
    for vl in scn.view_layers:
        vl.cycles.use_denoising = False
    scn.render.resolution_x, scn.render.resolution_y = resolution
    scn.render.film_transparent = transparent
    scn.world.use_nodes = True
    bg = scn.world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = (0.72, 0.71, 0.68, 1.0)
        bg.inputs["Strength"].default_value = 0.55

    az, el = np.radians(azimuth), np.radians(elevation)
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 80
    cam_data.sensor_fit = "AUTO"
    direction = mathutils.Vector(
        (np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el))
    )
    # bounding-sphere radius about the true center, then back the camera off
    # far enough that it fits the framing cone with margin (lens/aspect aware)
    sphere_r = max(
        float(np.linalg.norm(np.array([p.x, p.y, p.z]) - np.array(center)))
        for obj in objs
        for p in (obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box)
    ) or 1.0
    half_fov = np.arctan((cam_data.sensor_width / 2.0) / cam_data.lens)
    distance = sphere_r * 1.12 / np.sin(half_fov)
    eye = center + direction * distance
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = eye
    cam.rotation_euler = (center - eye).to_track_quat("-Z", "Y").to_euler()
    scn.camera = cam
    radius = sphere_r

    # three-point rig scaled to the subject
    def _light(name, kind, loc, energy, size=None):
        data = bpy.data.lights.new(name, type=kind)
        data.energy = energy
        if size is not None and kind == "AREA":
            data.size = size
        lamp = bpy.data.objects.new(name, data)
        lamp.location = center + mathutils.Vector(loc) * radius
        lamp.rotation_euler = (center - lamp.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.scene.collection.objects.link(lamp)

    _light("key", "AREA", (-2.2, -2.6, 3.2), 900 * radius, size=radius * 2.5)
    _light("fill", "AREA", (3.0, -1.5, 1.2), 300 * radius, size=radius * 3.0)
    _light("rim", "AREA", (0.5, 3.0, 2.0), 500 * radius, size=radius * 2.0)


def render(path):
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


# --------------------------------------------------------------------------
# export


def export_stl(path):
    import addon_utils

    bpy.ops.object.select_all(action="SELECT")
    try:  # Blender 4.x built-in exporter (not always registered on this build)
        bpy.ops.wm.stl_export(filepath=str(path), export_selected_objects=True)
    except (AttributeError, RuntimeError):
        addon_utils.enable("io_mesh_stl", default_set=False)
        bpy.ops.export_mesh.stl(filepath=str(path), use_selection=True)


def export_glb(path):
    import addon_utils

    addon_utils.enable("io_scene_gltf2", default_set=False)
    bpy.ops.export_scene.gltf(
        filepath=str(path), export_format="GLB",
        export_draco_mesh_compression_enable=False,  # optional lib absent here
    )


# --------------------------------------------------------------------------
# main


def parse_args(argv):
    p = argparse.ArgumentParser(prog="blender_build")
    p.add_argument("--scene", required=True)
    p.add_argument("--pose", default="__active__",
                   help="pose name, or 'rest' for the rest pose")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--size", choices=sorted(SIZE_PRESETS))
    grp.add_argument("--height", type=float)
    p.add_argument("--no-base", action="store_true")
    p.add_argument("--union", choices=["none", "boolean", "voxel"], default="none")
    p.add_argument("--voxel", type=float, default=0.6, help="voxel size (mm) for --union voxel")
    p.add_argument("--subdiv", type=int, default=0)
    p.add_argument("--smooth", action="store_true")
    p.add_argument("--render")
    p.add_argument("--stl")
    p.add_argument("--glb")
    p.add_argument("--check-watertight", action="store_true",
                   help="verify the exported STL is watertight+outward, fail if not")
    p.add_argument("--res", type=int, nargs=2, default=[600, 600])
    p.add_argument("--samples", type=int, default=48)
    p.add_argument("--azimuth", type=float, default=335.0)
    p.add_argument("--elevation", type=float, default=16.0)
    p.add_argument("--transparent", action="store_true")
    return p.parse_args(argv)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parse_args(argv)

    scene = Scene.load(args.scene)
    pose = None if args.pose == "rest" else args.pose
    height = SIZE_PRESETS[args.size] if args.size else args.height

    _clear_scene()
    objs = build_objects(scene, pose, height, with_base=not args.no_base)

    if args.union == "boolean":
        objs = [union_boolean(objs)]
    elif args.union == "voxel":
        objs = [union_voxel(objs, args.voxel)]

    if args.subdiv > 0:
        for obj in objs:
            add_subdivision(obj, args.subdiv)
    if args.smooth:
        shade_smooth(objs)

    if args.stl:
        export_stl(args.stl)
        if args.check_watertight:
            # Re-read with MiniMaster's own reader and hold the Blender solid
            # to the same bar as `minimaster export`: watertight + outward.
            from minimaster.core.stl import read_stl

            rep = read_stl(args.stl).integrity_report()
            if not rep["watertight"] or not rep["outward"]:
                print(
                    f"BUILD_FAIL fused STL is not a printable solid "
                    f"(watertight={rep['watertight']}, outward={rep['outward']}, "
                    f"boundary_edges={rep['boundary_edges']}, "
                    f"nonmanifold_edges={rep['nonmanifold_edges']}) — try "
                    f"--method voxel or a smaller --voxel"
                )
                raise SystemExit(1)
        print(f"BUILD_STL {args.stl}")
    if args.glb:
        export_glb(args.glb)
        print(f"BUILD_GLB {args.glb}")
    if args.render:
        setup_render(objs, args.azimuth, args.elevation, tuple(args.res),
                     args.samples, args.transparent)
        render(args.render)
        print(f"BUILD_RENDER {args.render}")
    print("BUILD_OK")


if __name__ == "__main__":
    main()
