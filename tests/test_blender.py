"""Tests for the optional Blender build/refine backend.

The argv-builder tests are pure Python and always run. The integration tests
shell out to a real ``blender`` and are skipped when none is available, so the
core suite stays Blender-free.
"""

import shutil

import pytest

from minimaster.blender import (
    BlenderError,
    BuildOptions,
    build_command,
    find_blender,
    run_build,
)
from minimaster.core.stl import read_stl
from minimaster.templates import template_path

HAVE_BLENDER = find_blender() is not None
needs_blender = pytest.mark.skipif(not HAVE_BLENDER, reason="blender not installed")


# -- pure argv building (no Blender) --------------------------------------


def test_script_args_size_and_render():
    opts = BuildOptions(size="medium", render="out.png", smooth=True, samples=32)
    args = opts.script_args("scene.mmp")
    assert args[:2] == ["--scene", "scene.mmp"]
    assert "--size" in args and args[args.index("--size") + 1] == "medium"
    assert "--render" in args and "--smooth" in args
    assert args[args.index("--samples") + 1] == "32"
    # height omitted when size given
    assert "--height" not in args


def test_script_args_voxel_and_no_base():
    opts = BuildOptions(height=40.0, union="voxel", voxel=0.5, stl="s.stl",
                        with_base=False, subdiv=2)
    args = opts.script_args("s.mmp")
    assert args[args.index("--height") + 1] == "40.0"
    assert args[args.index("--union") + 1] == "voxel"
    assert args[args.index("--voxel") + 1] == "0.5"
    assert "--no-base" in args
    assert args[args.index("--subdiv") + 1] == "2"


def test_voxel_arg_only_for_voxel_union():
    assert "--voxel" not in BuildOptions(union="boolean").script_args("s.mmp")
    assert "--voxel" in BuildOptions(union="voxel").script_args("s.mmp")


def test_build_command_structure():
    cmd = build_command("/usr/bin/blender", "s.mmp", BuildOptions(stl="o.stl"))
    assert cmd[0] == "/usr/bin/blender"
    assert "--background" in cmd and "--python" in cmd
    dashdash = cmd.index("--")
    assert "blender_build.py" in cmd[cmd.index("--python") + 1]
    assert cmd[dashdash + 1] == "--scene"


def test_run_build_without_blender(monkeypatch):
    monkeypatch.setenv("MINIMASTER_BLENDER", "/nonexistent/blender-binary")
    with pytest.raises(BlenderError, match="not found"):
        run_build("s.mmp", BuildOptions(stl="o.stl"))


# -- integration (real Blender) -------------------------------------------


@needs_blender
def test_voxel_fuse_is_watertight(tmp_path):
    out = tmp_path / "fused.stl"
    run_build(template_path("human_fighter"),
              BuildOptions(size="medium", union="voxel", voxel=0.6, stl=str(out)))
    assert out.is_file()
    mesh = read_stl(out)
    rep = mesh.integrity_report()
    assert rep["watertight"], rep
    assert rep["outward"], rep
    lo, hi = mesh.bounds
    assert hi[2] - lo[2] == pytest.approx(35.0, rel=0.05)  # 32 figure + ~3 base


@needs_blender
def test_boolean_fuse_is_watertight(tmp_path):
    out = tmp_path / "fused.stl"
    run_build(template_path("goblin"),
              BuildOptions(size="medium", union="boolean", stl=str(out)))
    mesh = read_stl(out)
    rep = mesh.integrity_report()
    assert rep["watertight"] and rep["outward"], rep
    # boolean preserves the low-poly structure: far fewer tris than voxel
    assert len(mesh.faces) < 5000


@needs_blender
def test_hq_render_writes_png(tmp_path):
    out = tmp_path / "r.png"
    run_build(template_path("dwarf"),
              BuildOptions(size="medium", smooth=True, samples=8,
                           resolution=(200, 200), render=str(out)))
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@needs_blender
def test_fuse_watertight_gate_rejects_bad_solid(tmp_path):
    """A voxel far larger than the figure's features remeshes to garbage; the
    watertight gate must fail loudly rather than write a bad STL."""
    out = tmp_path / "bad.stl"
    with pytest.raises(BlenderError, match="not a printable solid|not watertight"):
        run_build(template_path("human_fighter"),
                  BuildOptions(size="small", union="voxel", voxel=40.0,
                               stl=str(out), check_watertight=True))


def test_check_watertight_flag_emitted():
    args = BuildOptions(stl="o.stl", check_watertight=True).script_args("s.mmp")
    assert "--check-watertight" in args
    assert "--check-watertight" not in BuildOptions(stl="o.stl").script_args("s.mmp")


@needs_blender
def test_cli_fuse_and_error(tmp_path):
    from minimaster.__main__ import main

    out = tmp_path / "c.stl"
    rc = main(["fuse", str(template_path("orc")), "-o", str(out),
               "--method", "voxel", "--size", "small"])
    assert rc == 0
    assert read_stl(out).integrity_report()["watertight"]
