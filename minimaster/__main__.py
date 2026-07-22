"""MiniMaster command line: launch the studio or run headless operations.

    minimaster                     launch the GUI
    minimaster templates           list starter templates
    minimaster new NAME -o f.mmp   start a project from a template
    minimaster export f.mmp -o out.stl [--pose P] [--size medium | --height MM]
    minimaster preview f.mmp -o out.png [--pose P]
    minimaster validate f.mmp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .export import SIZE_PRESETS, ExportError, assemble_body, export_stl
from .scene import Scene
from .templates import list_templates, load_template


def _add_pose_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--pose",
        default="__active__",
        help="pose name to apply (default: the scene's active pose); use 'rest' for the rest pose",
    )


def _resolve_pose(arg: str) -> str | None:
    return None if arg == "rest" else arg


def _load_scene(path: str) -> Scene | None:
    """Load a scene, or print the CLI-convention error and return None."""
    try:
        return Scene.load(path)
    except Exception as exc:
        print(f"error: could not load {path}: {exc}", file=sys.stderr)
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minimaster",
        description="Design, rig, pose, and export 3D-printable miniatures.",
    )
    parser.add_argument("--version", action="version", version=f"minimaster {__version__}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("gui", help="launch the studio (default)")

    sub.add_parser("templates", help="list starter templates")

    p_new = sub.add_parser("new", help="create a project from a starter template")
    p_new.add_argument("template", help="template name (see 'minimaster templates')")
    p_new.add_argument("-o", "--output", required=True, help="output .mmp path")

    p_exp = sub.add_parser("export", help="export a project to binary STL")
    p_exp.add_argument("scene", help="input .mmp project")
    p_exp.add_argument("-o", "--output", required=True, help="output .stl path")
    _add_pose_arg(p_exp)
    group = p_exp.add_mutually_exclusive_group()
    group.add_argument(
        "--size", choices=sorted(SIZE_PRESETS), help="scale figure to a size category"
    )
    group.add_argument("--height", type=float, help="scale figure to this height in mm")
    p_exp.add_argument("--no-base", action="store_true", help="export without the base")
    p_exp.add_argument("--png", help="also write a PNG preview to this path")

    p_pre = sub.add_parser("preview", help="render a PNG preview of a project")
    p_pre.add_argument("scene", help="input .mmp project")
    p_pre.add_argument("-o", "--output", required=True, help="output .png path")
    _add_pose_arg(p_pre)
    p_pre.add_argument("--width", type=int, default=800)
    p_pre.add_argument("--height", type=int, default=800)
    p_pre.add_argument("--azimuth", type=float, default=35.0)
    p_pre.add_argument("--elevation", type=float, default=22.0)
    p_pre.add_argument("--no-base", action="store_true")

    p_val = sub.add_parser("validate", help="check a project file and its geometry")
    p_val.add_argument("scene", help="input .mmp project")

    def _add_size(p):
        g = p.add_mutually_exclusive_group()
        g.add_argument("--size", choices=sorted(SIZE_PRESETS))
        g.add_argument("--height", type=float)

    # -- dynamic body mesh (pure-Python isosurface, no Blender) ----------
    p_bake = sub.add_parser(
        "bake",
        help="fuse the shapes into ONE continuous skinned body (STL or PNG)")
    p_bake.add_argument("scene", help="input .mmp project")
    p_bake.add_argument("-o", "--output", required=True,
                        help="output path; .stl bakes a solid, .png renders the body")
    _add_pose_arg(p_bake)
    _add_size(p_bake)
    p_bake.add_argument("--resolution", type=float, default=0.6,
                        help="voxel size in scene units (smaller = smoother, slower)")
    p_bake.add_argument("--blend", type=float, default=0.6,
                        help="smooth-union width; higher fuses limbs more")
    p_bake.add_argument("--all-shapes", action="store_true",
                        help="fuse gear/detail too (default keeps them out of the body)")
    p_bake.add_argument("--no-base", action="store_true")
    p_bake.add_argument("--smooth", action="store_true",
                        help="PNG only: smooth shading (default flat, for the chunky look)")
    p_bake.add_argument("--azimuth", type=float, default=35.0, help="PNG only")
    p_bake.add_argument("--elevation", type=float, default=14.0, help="PNG only")
    p_bake.add_argument("--res", type=int, nargs=2, default=[700, 900], help="PNG size")

    # -- Blender backend (optional) --------------------------------------

    p_fuse = sub.add_parser(
        "fuse", help="[Blender] union the shells into one watertight solid STL")
    p_fuse.add_argument("scene")
    p_fuse.add_argument("-o", "--output", required=True, help="output .stl path")
    _add_pose_arg(p_fuse)
    _add_size(p_fuse)
    p_fuse.add_argument("--method", choices=["voxel", "boolean"], default="voxel",
                        help="voxel = watertight organic solid; boolean = hard edges")
    p_fuse.add_argument("--voxel", type=float, default=0.6,
                        help="voxel size in mm for --method voxel")
    p_fuse.add_argument("--subdiv", type=int, default=0)
    p_fuse.add_argument("--no-base", action="store_true")

    p_hq = sub.add_parser(
        "hq-render", help="[Blender] Cycles render (lit, shadowed, materials)")
    p_hq.add_argument("scene")
    p_hq.add_argument("-o", "--output", required=True, help="output .png path")
    _add_pose_arg(p_hq)
    _add_size(p_hq)
    p_hq.add_argument("--union", choices=["none", "voxel", "boolean"], default="none")
    p_hq.add_argument("--voxel", type=float, default=0.6)
    p_hq.add_argument("--subdiv", type=int, default=0)
    p_hq.add_argument("--smooth", action="store_true", help="smooth shading")
    p_hq.add_argument("--samples", type=int, default=48)
    p_hq.add_argument("--res", type=int, nargs=2, default=[600, 600])
    p_hq.add_argument("--azimuth", type=float, default=335.0)
    p_hq.add_argument("--elevation", type=float, default=16.0)
    p_hq.add_argument("--transparent", action="store_true")
    p_hq.add_argument("--no-base", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command or "gui"

    if command == "gui":
        from .app import run_app  # tkinter imported only when actually needed

        run_app()
        return 0

    if command == "templates":
        names = list_templates()
        if not names:
            print("no templates installed")
        for name in names:
            print(name)
        return 0

    if command == "new":
        try:
            scene = load_template(args.template)
        except KeyError as exc:
            print(f"error: {exc.args[0]}", file=sys.stderr)
            return 2
        out = Path(args.output)
        scene.save(out)
        print(f"created {out} from template {args.template!r}")
        return 0

    if command == "export":
        scene = _load_scene(args.scene)
        if scene is None:
            return 2
        try:
            report = export_stl(
                scene,
                args.output,
                pose_name=_resolve_pose(args.pose),
                size=args.size,
                height=args.height,
                with_base=not args.no_base,
            )
        except (ExportError, KeyError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        sx, sy, sz = report["size_mm"]
        print(
            f"wrote {report['path']}: {report['triangles']} triangles, "
            f"{sx:.1f} x {sy:.1f} x {sz:.1f} mm, "
            f"{report['volume_mm3'] / 1000.0:.1f} cm3, "
            f"watertight={report['watertight']}"
        )
        if args.png:
            from .render import render_scene

            render_scene(
                scene,
                path=args.png,
                pose_name=_resolve_pose(args.pose),
                with_base=not args.no_base,
            )
            print(f"wrote {args.png}")
        return 0

    if command == "preview":
        from .render import render_scene

        scene = _load_scene(args.scene)
        if scene is None:
            return 2
        try:
            render_scene(
                scene,
                path=args.output,
                pose_name=_resolve_pose(args.pose),
                with_base=not args.no_base,
                size=(args.width, args.height),
                azimuth=args.azimuth,
                elevation=args.elevation,
            )
        except (KeyError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"wrote {args.output}")
        return 0

    if command == "validate":
        try:
            scene = Scene.load(args.scene)
        except Exception as exc:
            print(f"invalid scene file: {exc}", file=sys.stderr)
            return 2
        problems = []
        for shape, mesh in scene.build_shape_meshes(None):
            rep = mesh.integrity_report()
            if not rep["watertight"]:
                problems.append(f"shape {shape.name!r}: {rep}")
        for pose_name in scene.poses:
            merged = scene.build_merged_mesh(pose_name)
            rep = merged.integrity_report()
            if not rep["watertight"]:
                problems.append(f"pose {pose_name!r}: merged mesh not watertight")
        if problems:
            print(f"{args.scene}: PROBLEMS FOUND", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 1
        print(
            f"{args.scene}: ok ({len(scene.shapes)} shapes, "
            f"{len(scene.armature)} joints, {len(scene.poses)} poses)"
        )
        return 0

    if command == "bake":
        scene = _load_scene(args.scene)
        if scene is None:
            return 2
        pose = _resolve_pose(args.pose)
        exclude = () if args.all_shapes else None
        out = Path(args.output)
        if out.suffix.lower() == ".png":
            from .core import bodymesh
            from .core.mesh import Mesh
            from .bases import build_base
            from .render import render_meshes, write_png

            kw = {} if exclude is None else {"exclude": exclude}
            body = bodymesh.body_from_scene(
                scene, args.resolution, args.blend, pose, **kw)
            if not len(body.faces):
                print("error: scene baked to an empty body", file=sys.stderr)
                return 2
            colored = [(body, "#6c5560")]
            skins = scene.armature.skin_matrices(scene.resolve_pose(pose))
            for s in scene.shapes:  # overlay eyes so the face still reads
                if not any(k in s.name for k in ("eye", "brow")):
                    continue
                m = s.build_mesh()
                if s.bone in skins:
                    m = m.transform(skins[s.bone])
                colored.append((m, s.color))
            fig = Mesh.merge([m for m, _ in colored])
            lo, hi = fig.bounds
            c = (lo + hi) / 2.0
            colored = [(m.translated([-c[0], -c[1], -lo[2]]), col)
                       for m, col in colored]
            if not args.no_base:
                base = build_base(scene.base)
                if base is not None:
                    colored.append((base, "#6e6a63"))
            write_png(out, render_meshes(
                colored, size=tuple(args.res),
                azimuth=args.azimuth, elevation=args.elevation,
                shading="smooth" if args.smooth else "flat"))
            print(f"wrote {out}")
            return 0

        from .core.stl import write_stl

        try:
            mesh, rep = assemble_body(
                scene, pose_name=pose, resolution=args.resolution,
                blend=args.blend, size=args.size, height=args.height,
                with_base=not args.no_base, exclude=exclude)
        except (ExportError, KeyError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        write_stl(mesh, out, name=scene.name)
        lo, hi = mesh.bounds
        sx, sy, sz = (hi - lo)
        note = "" if rep["watertight"] else "  (NOT watertight — try --blend higher or --resolution finer)"
        print(f"wrote {out}: {rep['triangles']} triangles, "
              f"{sx:.1f} x {sy:.1f} x {sz:.1f} mm, "
              f"blend={rep['blend']:.2f}, watertight={rep['watertight']}{note}")
        return 0

    if command in ("fuse", "hq-render"):
        from .blender import BlenderError, BuildOptions, run_build

        common = dict(
            pose=_resolve_pose(args.pose) or "rest",
            size=args.size,
            height=args.height,
            with_base=not args.no_base,
        )
        if command == "fuse":
            opts = BuildOptions(
                union=args.method, voxel=args.voxel, subdiv=args.subdiv,
                stl=args.output, check_watertight=True, **common,
            )
        else:
            opts = BuildOptions(
                union=args.union, voxel=args.voxel, subdiv=args.subdiv,
                smooth=args.smooth, render=args.output, samples=args.samples,
                resolution=tuple(args.res), azimuth=args.azimuth,
                elevation=args.elevation, transparent=args.transparent, **common,
            )
        try:
            run_build(args.scene, opts)
        except BlenderError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"wrote {args.output}")
        return 0

    raise AssertionError(f"unhandled command {command!r}")


if __name__ == "__main__":
    sys.exit(main())
