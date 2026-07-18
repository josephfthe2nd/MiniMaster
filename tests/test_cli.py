import numpy as np
import pytest

from minimaster.__main__ import main
from minimaster.core.stl import read_stl
from minimaster.scene import Scene


@pytest.fixture()
def project(tmp_path):
    scene = Scene(name="cli-fig")
    scene.armature.add_joint("root", [0, 0, 0])
    scene.armature.add_joint("tip", [0, 0, 20], parent="root")
    scene.add_shape("cylinder", name="body", position=[0, 0, 10],
                    scale=[6, 6, 20], bone="tip")
    scene.save_pose("lean", {"root": (20, 0, 0)})
    path = tmp_path / "fig.mmp"
    scene.save(path)
    return path


def test_export_command(project, tmp_path, capsys):
    out = tmp_path / "fig.stl"
    rc = main(["export", str(project), "-o", str(out), "--size", "medium"])
    assert rc == 0
    assert "watertight=True" in capsys.readouterr().out
    mesh = read_stl(out)
    lo, hi = mesh.bounds
    assert hi[2] - lo[2] == pytest.approx(35.0, rel=1e-5)


def test_export_pose_and_no_base(project, tmp_path):
    out = tmp_path / "fig.stl"
    rc = main(["export", str(project), "-o", str(out), "--pose", "lean", "--no-base",
               "--height", "30"])
    assert rc == 0
    mesh = read_stl(out)
    lo, hi = mesh.bounds
    assert hi[2] - lo[2] == pytest.approx(30.0, rel=1e-5)


def test_export_rest_pose_keyword(project, tmp_path):
    out = tmp_path / "r.stl"
    rc = main(["export", str(project), "-o", str(out), "--pose", "rest", "--no-base"])
    assert rc == 0


def test_export_unknown_pose_fails(project, tmp_path, capsys):
    rc = main(["export", str(project), "-o", str(tmp_path / "x.stl"), "--pose", "nope"])
    assert rc == 2
    assert "nope" in capsys.readouterr().err


def test_export_with_png(project, tmp_path):
    out = tmp_path / "fig.stl"
    png = tmp_path / "fig.png"
    rc = main(["export", str(project), "-o", str(out), "--png", str(png)])
    assert rc == 0
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_preview_command(project, tmp_path):
    png = tmp_path / "p.png"
    rc = main(["preview", str(project), "-o", str(png), "--width", "64", "--height", "64"])
    assert rc == 0
    assert png.is_file()


def test_validate_command(project, capsys):
    rc = main(["validate", str(project)])
    assert rc == 0
    assert "ok" in capsys.readouterr().out


def test_validate_rejects_garbage(tmp_path, capsys):
    bad = tmp_path / "bad.mmp"
    bad.write_text("{not json")
    rc = main(["validate", str(bad)])
    assert rc == 2


def test_templates_command_runs(capsys):
    rc = main(["templates"])
    assert rc == 0


def test_new_unknown_template(tmp_path, capsys):
    rc = main(["new", "not-a-template", "-o", str(tmp_path / "x.mmp")])
    assert rc == 2


def test_export_missing_scene_clean_error(tmp_path, capsys):
    rc = main(["export", str(tmp_path / "nope.mmp"), "-o", str(tmp_path / "x.stl")])
    assert rc == 2
    assert "could not load" in capsys.readouterr().err


def test_preview_corrupt_scene_clean_error(tmp_path, capsys):
    bad = tmp_path / "bad.mmp"
    bad.write_text("{broken")
    rc = main(["preview", str(bad), "-o", str(tmp_path / "x.png")])
    assert rc == 2
    assert "could not load" in capsys.readouterr().err
