import numpy as np
import pytest

from minimaster.scene import Scene, Shape, UndoStack


def simple_scene() -> Scene:
    scene = Scene(name="test")
    scene.armature.add_joint("shoulder", [0, 0, 10])
    scene.armature.add_joint("elbow", [5, 0, 10], parent="shoulder")
    scene.armature.add_joint("wrist", [10, 0, 10], parent="elbow")
    scene.add_shape(
        "capsule",
        name="forearm",
        position=[7.5, 0, 10],
        rotation=[0, 90, 0],
        scale=[1, 1, 5],
        bone="wrist",
    )
    scene.save_pose("bend", {"elbow": (0, -90, 0)})
    return scene


def test_add_shape_unique_names():
    scene = Scene()
    a = scene.add_shape("box")
    b = scene.add_shape("box")
    c = scene.add_shape("box")
    assert a.name == "box" and b.name == "box.001" and c.name == "box.002"


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        Shape(name="s", kind="blob")


def test_build_rest_and_posed():
    scene = simple_scene()
    rest = scene.build_merged_mesh(None)
    center_rest = rest.vertices.mean(axis=0)
    assert np.allclose(center_rest, [7.5, 0, 10], atol=1e-9)

    posed = scene.build_merged_mesh("bend")
    center_posed = posed.vertices.mean(axis=0)
    assert np.allclose(center_posed, [5, 0, 12.5], atol=1e-9)
    assert posed.integrity_report()["watertight"]


def test_active_pose_resolution():
    scene = simple_scene()
    scene.active_pose = "bend"
    active = scene.build_merged_mesh()
    explicit = scene.build_merged_mesh("bend")
    assert np.allclose(active.vertices, explicit.vertices)
    with pytest.raises(KeyError):
        scene.build_merged_mesh("nope")


def test_json_round_trip(tmp_path):
    scene = simple_scene()
    scene.active_pose = "bend"
    path = tmp_path / "t.mmp"
    scene.save(path)
    loaded = Scene.load(path)
    assert loaded.name == "test"
    assert loaded.active_pose == "bend"
    assert loaded.get_shape("forearm").bone == "wrist"
    a = scene.build_merged_mesh()
    b = loaded.build_merged_mesh()
    assert np.allclose(a.vertices, b.vertices)
    assert loaded.to_dict() == scene.to_dict()


def test_load_cleans_dangling_references():
    scene = simple_scene()
    data = scene.to_dict()
    data["shapes"][0]["bone"] = "ghost"
    data["active_pose"] = "missing"
    loaded = Scene.from_dict(data)
    assert loaded.get_shape("forearm").bone is None
    assert loaded.active_pose is None


def test_load_rejects_wrong_format():
    with pytest.raises(ValueError):
        Scene.from_dict({"format": "other"})
    with pytest.raises(ValueError):
        Scene.from_dict({"format": "minimaster-scene", "version": 99})


def test_duplicate_and_mirror():
    scene = simple_scene()
    dup = scene.duplicate_shape("forearm")
    assert dup.name == "forearm.001"
    assert np.allclose(dup.position, [9.5, 0, 10])

    scene.armature.add_joint("shoulder_r", [0, 0, 10], parent="shoulder")
    scene.armature.add_joint("wrist_l", [10, 0, 10], parent="shoulder_r")
    scene.armature.add_joint("wrist_r", [-10, 0, 10], parent="shoulder_r")
    arm_l = scene.add_shape("box", name="hand", position=[3, 1, 2], bone="wrist_l")
    mirrored = scene.mirror_shape("hand")
    assert np.allclose(mirrored.position, [-3, 1, 2])
    assert mirrored.bone == "wrist_r"
    rep = mirrored.build_mesh().integrity_report()
    assert rep["watertight"] and rep["volume"] > 0


def test_remove_joint_unbinds_shapes_and_prunes_poses():
    scene = simple_scene()
    scene.remove_joint("wrist")
    assert scene.get_shape("forearm").bone is None
    scene2 = simple_scene()
    scene2.remove_joint("elbow")
    assert scene2.poses["bend"] == {}
    # wrist got reparented to shoulder; binding to it remains valid
    assert scene2.get_shape("forearm").bone == "wrist"


def test_rename_joint_updates_bindings_and_poses():
    scene = simple_scene()
    scene.rename_joint("wrist", "hand")
    assert scene.get_shape("forearm").bone == "hand"
    scene.rename_joint("elbow", "hinge")
    assert scene.poses["bend"] == {"hinge": (0, -90, 0)}


def test_auto_bind():
    scene = simple_scene()
    s = scene.add_shape("box", name="pad", position=[2.5, 0.5, 10])
    scene.auto_bind(only=["pad"])
    assert s.bone == "elbow"


def test_undo_stack():
    undo = UndoStack(limit=3)
    assert not undo.can_undo and not undo.can_redo
    undo.push("v1")
    undo.push("v1")  # dedup
    undo.push("v2")
    assert undo.undo("v3") == "v2"
    assert undo.undo("v2") == "v1"
    assert undo.undo("v1") is None
    assert undo.redo("v1") == "v2"
    undo.push("v4")
    assert not undo.can_redo  # push clears redo


def test_load_filters_ghost_pose_joints():
    scene = simple_scene()
    data = scene.to_dict()
    data["poses"]["bend"]["ghost"] = [90, 0, 0]
    loaded = Scene.from_dict(data)
    assert "ghost" not in loaded.poses["bend"]
    assert loaded.poses["bend"]["elbow"] == (0.0, -90.0, 0.0)
