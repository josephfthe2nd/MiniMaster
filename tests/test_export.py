import numpy as np
import pytest

from minimaster.bases import build_base, round_base, square_base
from minimaster.export import SIZE_PRESETS, ExportError, assemble, export_stl
from minimaster.core.stl import read_stl
from minimaster.scene import Scene


def figure_scene() -> Scene:
    scene = Scene(name="fig")
    scene.armature.add_joint("root", [0, 0, 0])
    scene.armature.add_joint("top", [0, 0, 20], parent="root")
    scene.add_shape("capsule", name="body", position=[0, 0, 10],
                    scale=[4, 4, 20], bone="top")
    scene.save_pose("lean", {"root": (0, 30, 0)})
    return scene


def test_round_base_watertight_and_top_at_zero():
    b = round_base(25.0, 3.0)
    rep = b.integrity_report()
    assert rep["watertight"] and rep["volume"] > 0
    lo, hi = b.bounds
    assert hi[2] == pytest.approx(0.0)
    assert lo[2] == pytest.approx(-3.0)
    assert hi[0] - lo[0] == pytest.approx(25.0)


def test_square_base_and_none():
    b = square_base(20.0, 3.0)
    assert b.integrity_report()["watertight"]
    assert build_base({"style": "none"}) is None
    assert build_base(None) is None
    with pytest.raises(ValueError):
        build_base({"style": "hex"})
    with pytest.raises(ValueError):
        build_base({"style": "round", "diameter": -1})


def test_assemble_scales_to_size():
    scene = figure_scene()
    for size, target in SIZE_PRESETS.items():
        mesh = assemble(scene, pose_name=None, size=size, with_base=False)
        lo, hi = mesh.bounds
        assert hi[2] - lo[2] == pytest.approx(target, rel=1e-9)
        assert lo[2] == pytest.approx(0.0, abs=1e-9)  # stands on the plate
        assert (lo[:2] + hi[:2]) / 2 == pytest.approx([0.0, 0.0], abs=1e-9)


def test_assemble_with_base_extends_below():
    scene = figure_scene()
    mesh = assemble(scene, pose_name=None, size="medium", with_base=True)
    lo, hi = mesh.bounds
    assert lo[2] == pytest.approx(-3.0)  # base underneath
    assert hi[2] == pytest.approx(32.0)
    assert mesh.integrity_report()["watertight"]


def test_assemble_height_and_conflicts():
    scene = figure_scene()
    mesh = assemble(scene, pose_name=None, height=40.0, with_base=False)
    lo, hi = mesh.bounds
    assert hi[2] - lo[2] == pytest.approx(40.0)
    with pytest.raises(ExportError):
        assemble(scene, height=30, size="medium")
    with pytest.raises(ExportError):
        assemble(scene, size="galactic")
    with pytest.raises(ExportError):
        assemble(Scene())


def test_assemble_posed():
    scene = figure_scene()
    upright = assemble(scene, pose_name=None, with_base=False)
    leaned = assemble(scene, pose_name="lean", with_base=False)
    # Leaning shrinks the height and widens the X footprint.
    up_lo, up_hi = upright.bounds
    ln_lo, ln_hi = leaned.bounds
    assert ln_hi[2] - ln_lo[2] < up_hi[2] - up_lo[2]
    assert ln_hi[0] - ln_lo[0] > up_hi[0] - up_lo[0]
    assert leaned.integrity_report()["watertight"]


def test_export_stl_round_trip(tmp_path):
    scene = figure_scene()
    out = tmp_path / "fig.stl"
    report = export_stl(scene, out, pose_name=None, size="medium")
    assert report["watertight"]
    assert out.stat().st_size == 84 + 50 * report["triangles"]
    loaded = read_stl(out)
    assert loaded.integrity_report()["watertight"]
    lo, hi = loaded.bounds
    assert hi[2] - lo[2] == pytest.approx(35.0, rel=1e-5)  # 32 figure + 3 base


def test_assemble_rejects_inverted_shell(monkeypatch):
    from minimaster.core.mesh import Mesh

    scene = figure_scene()
    orig = scene.build_shape_meshes

    def flipped(pose_name="__active__"):
        return [
            (s, Mesh(m.vertices, m.faces[:, [0, 2, 1]])) for s, m in orig(pose_name)
        ]

    monkeypatch.setattr(scene, "build_shape_meshes", flipped)
    with pytest.raises(ExportError, match="not a printable shell"):
        assemble(scene, pose_name=None)


def test_assemble_wraps_bad_base_as_export_error():
    scene = figure_scene()
    scene.base = {"style": "round", "diameter": 0.0}
    with pytest.raises(ExportError, match="positive"):
        assemble(scene, pose_name=None)
