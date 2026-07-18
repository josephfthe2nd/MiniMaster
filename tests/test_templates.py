"""Every shipped template must load, be fully rigged, and export watertight
in every pose."""

import subprocess
import sys
from pathlib import Path

import pytest

from minimaster.export import assemble
from minimaster.templates import list_templates, load_template

EXPECTED = {"human_fighter", "dwarf", "goblin", "orc", "skeleton"}


def test_all_expected_templates_ship():
    assert EXPECTED.issubset(set(list_templates()))


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_template_loads_and_is_rigged(name):
    scene = load_template(name)
    assert len(scene.shapes) >= 20
    assert len(scene.armature) == 21
    assert scene.poses, "template should ship poses"
    assert scene.active_pose in scene.poses
    # every shape is bound to a real bone
    bones = set(scene.armature.bone_names())
    for shape in scene.shapes:
        assert shape.bone in bones, f"{shape.name} unbound"
    # every posed joint exists
    for pose_name, pose in scene.poses.items():
        for joint in pose:
            assert joint in scene.armature.joints, f"{pose_name} poses ghost {joint}"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_template_shapes_watertight(name):
    scene = load_template(name)
    for shape, mesh in scene.build_shape_meshes(None):
        rep = mesh.integrity_report()
        assert rep["watertight"], f"{shape.name}: {rep}"
        assert rep["volume"] > 0


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_template_exports_watertight_in_every_pose(name):
    scene = load_template(name)
    for pose_name in [None, *scene.poses]:
        mesh = assemble(scene, pose_name=pose_name, size="medium")
        rep = mesh.integrity_report()
        assert rep["watertight"], f"pose {pose_name}: {rep}"
        lo, hi = mesh.bounds
        assert hi[2] - lo[2] == pytest.approx(35.0)  # 32 figure + 3 base


def test_generator_matches_shipped_files(tmp_path):
    """tools/make_templates.py output must match the committed templates."""
    repo = Path(__file__).resolve().parent.parent
    script = repo / "tools" / "make_templates.py"
    env_script = (
        "import sys, runpy; "
        f"sys.argv=['make_templates']; "
        f"import tools.make_templates as mt; "
        f"mt.OUT_DIR = r'{tmp_path}'; "
        "from pathlib import Path; mt.OUT_DIR = Path(mt.OUT_DIR); "
        "mt.main()"
    )
    subprocess.run(
        [sys.executable, "-c", env_script], cwd=repo, check=True, capture_output=True
    )
    for name in sorted(EXPECTED):
        generated = (tmp_path / f"{name}.mmp").read_text()
        shipped = (repo / "minimaster" / "templates" / f"{name}.mmp").read_text()
        assert generated == shipped, f"{name}: regenerate minimaster/templates"
