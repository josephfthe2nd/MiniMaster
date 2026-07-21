"""Launcher for the optional Blender build/refine backend.

This module is intentionally **bpy-free** — it only shells out to Blender,
which runs ``tools/blender_build.py`` (the code that imports bpy). Keeping the
subprocess boundary here means the package, the tkinter studio, and the test
suite never depend on Blender being installed; the Blender pipeline is a pure
bonus when a ``blender`` binary is available.

Set ``MINIMASTER_BLENDER`` to point at a specific Blender binary; otherwise the
first ``blender`` on ``PATH`` is used.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

_BUILD_SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "blender_build.py"


class BlenderError(RuntimeError):
    pass


def find_blender() -> str | None:
    """Path to the Blender binary, or None if not found."""
    override = os.environ.get("MINIMASTER_BLENDER")
    if override:
        return override if (Path(override).exists() or shutil.which(override)) else None
    return shutil.which("blender")


def build_script_path() -> Path:
    if not _BUILD_SCRIPT.is_file():
        raise BlenderError(f"blender build script missing: {_BUILD_SCRIPT}")
    return _BUILD_SCRIPT


@dataclass
class BuildOptions:
    pose: str = "__active__"
    size: str | None = None
    height: float | None = None
    with_base: bool = True
    union: str = "none"  # none | boolean | voxel
    voxel: float = 0.6
    subdiv: int = 0
    smooth: bool = False
    render: str | None = None
    stl: str | None = None
    glb: str | None = None
    check_watertight: bool = False
    resolution: tuple[int, int] = (600, 600)
    samples: int = 48
    azimuth: float = 335.0
    elevation: float = 16.0
    transparent: bool = False

    def script_args(self, scene: str) -> list[str]:
        """Args passed after ``--`` to blender_build.py (kept in sync with its
        argparse). Pure — safe to unit-test without Blender."""
        args = ["--scene", str(scene), "--pose", self.pose]
        if self.size:
            args += ["--size", self.size]
        elif self.height is not None:
            args += ["--height", str(self.height)]
        if not self.with_base:
            args.append("--no-base")
        args += ["--union", self.union]
        if self.union == "voxel":
            args += ["--voxel", str(self.voxel)]
        if self.subdiv > 0:
            args += ["--subdiv", str(self.subdiv)]
        if self.smooth:
            args.append("--smooth")
        if self.render:
            args += ["--render", str(self.render)]
        if self.stl:
            args += ["--stl", str(self.stl)]
        if self.glb:
            args += ["--glb", str(self.glb)]
        if self.check_watertight:
            args.append("--check-watertight")
        args += ["--res", str(self.resolution[0]), str(self.resolution[1])]
        args += ["--samples", str(self.samples)]
        args += ["--azimuth", str(self.azimuth), "--elevation", str(self.elevation)]
        if self.transparent:
            args.append("--transparent")
        return args


def build_command(blender: str, scene: str, opts: BuildOptions) -> list[str]:
    """Full argv to invoke Blender headlessly on the build script."""
    return [
        blender, "--background", "--factory-startup",
        "--python", str(build_script_path()),
        "--", *opts.script_args(scene),
    ]


def run_build(scene, opts: BuildOptions, timeout: float = 600.0) -> str:
    """Run the Blender build and return its stdout. Raises BlenderError on a
    missing binary or a non-zero / failed run."""
    blender = find_blender()
    if blender is None:
        raise BlenderError(
            "Blender not found. Install it, or set MINIMASTER_BLENDER to the "
            "blender binary. The Blender backend is optional; 'export' and "
            "'preview' work without it."
        )
    cmd = build_command(blender, str(scene), opts)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise BlenderError(f"Blender build timed out after {timeout}s") from exc
    if proc.returncode != 0 or "BUILD_OK" not in proc.stdout:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
        raise BlenderError(f"Blender build failed (rc={proc.returncode}):\n{tail}")
    return proc.stdout
