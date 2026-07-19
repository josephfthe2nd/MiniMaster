import numpy as np
import pytest

from minimaster.core import math3d as m3
from minimaster.scene import Scene


def grouped_scene() -> Scene:
    """A rigged scene with a two-shape, two-joint part instance 'horn'."""
    scene = Scene(name="g")
    scene.armature.add_joint("pelvis", [0, 0, 10])
    scene.armature.add_joint("chest", [0, 0, 20], parent="pelvis")
    scene.armature.add_joint("head", [0, 0, 28], parent="chest")
    scene.add_shape("box", name="body", position=[0, 0, 20], scale=[8, 6, 16],
                    bone="chest")

    scene.armature.add_joint("horn:root", [2, -3, 30], parent="chest")
    scene.armature.add_joint("horn:tip", [2, -3, 34], parent="horn:root")
    scene.add_shape("cylinder", name="horn:base", position=[2, -3, 31],
                    scale=[2, 2, 3], bone="horn:tip", group="horn")
    scene.add_shape("icosphere", name="horn:knob", position=[2, -3, 34],
                    scale=[1.5, 1.5, 1.5], bone="horn:tip", group="horn")
    scene.groups["horn"] = {
        "part": "horn",
        "origin": [2.0, -3.0, 30.0],
        "rotation": [0.0, 0.0, 0.0],
        "scale": 1.0,
        "joints": ["horn:root", "horn:tip"],
    }
    return scene


def local_matrices(scene, group):
    return {s.name: s.local_matrix() for s in scene.group_members(group)}


def test_transform_group_matches_matrix_composition():
    rng = np.random.default_rng(3)
    for _ in range(20):
        scene = grouped_scene()
        # give members interesting rotations and a mirrored (negative) scale
        scene.get_shape("horn:base").rotation = rng.uniform(-90, 90, 3)
        scene.get_shape("horn:knob").scale = np.array([-1.5, 1.5, 1.5])
        before = local_matrices(scene, "horn")

        angles = rng.uniform(-120, 120, 3)
        rot = m3.euler_rotation(*angles)
        t = rng.uniform(-5, 5, 3)
        s = float(rng.uniform(0.4, 2.5))
        pivot = np.array(scene.groups["horn"]["origin"])
        scene.transform_group("horn", rotation=rot, translation=t, scale=s)

        g = (
            m3.translation_mat(pivot + t)
            @ m3.mat4(rotation=rot * s)
            @ m3.translation_mat(-pivot)
        )
        after = local_matrices(scene, "horn")
        for name, m_before in before.items():
            assert np.allclose(after[name], g @ m_before, atol=1e-9), name
        # joints and origin moved by the same map
        expected_root = pivot + t + s * (rot @ (np.array([2, -3, 30.0]) - pivot))
        assert np.allclose(
            scene.armature.joints["horn:root"].position, expected_root, atol=1e-9
        )
        assert np.allclose(scene.groups["horn"]["origin"], expected_root, atol=1e-9)


def test_transform_group_rejects_nonpositive_scale():
    scene = grouped_scene()
    with pytest.raises(ValueError):
        scene.transform_group("horn", scale=0.0)
    with pytest.raises(ValueError):
        scene.transform_group("horn", scale=-1.0)


def test_translate_group_moves_shapes_and_joints():
    scene = grouped_scene()
    scene.translate_group("horn", [1, 2, 3])
    assert np.allclose(scene.get_shape("horn:base").position, [3, -1, 34])
    assert np.allclose(scene.armature.joints["horn:tip"].position, [3, -1, 37])
    assert np.allclose(scene.groups["horn"]["origin"], [3, -1, 33])
    # non-members untouched
    assert np.allclose(scene.get_shape("body").position, [0, 0, 20])


def test_group_outward_axis_tracks_rotation():
    scene = grouped_scene()
    assert np.allclose(scene.group_outward_axis("horn"), [0, 0, 1])
    scene.transform_group("horn", rotation=m3.rot_x(90))
    assert np.allclose(scene.group_outward_axis("horn"), [0, -1, 0], atol=1e-12)


def test_shapes_stay_printable_after_group_ops():
    scene = grouped_scene()
    scene.transform_group("horn", rotation=m3.euler_rotation(33, -20, 140),
                          translation=[4, 1, -2], scale=1.7)
    for _, mesh in scene.build_shape_meshes(None):
        rep = mesh.integrity_report()
        assert rep["watertight"] and rep["outward"]


def test_mirror_group_geometry_and_joints():
    scene = grouped_scene()
    new = scene.mirror_group("horn")
    assert new != "horn" and new in scene.groups
    # mirrored joints exist with reflected positions, parented like the source
    meta = scene.groups[new]
    assert len(meta["joints"]) == 2
    root, tip = meta["joints"]
    assert np.allclose(scene.armature.joints[root].position, [-2, -3, 30])
    assert np.allclose(scene.armature.joints[tip].position, [-2, -3, 34])
    assert scene.armature.joints[root].parent == "chest"
    assert scene.armature.joints[tip].parent == root
    # mirrored shapes bind to the mirrored internal joints and stay printable
    for s in scene.group_members(new):
        assert s.bone == tip
        rep = s.build_mesh().integrity_report()
        assert rep["watertight"] and rep["outward"], s.name
    # geometry is the reflection of the source
    src_mesh = scene.get_shape("horn:base").build_mesh()
    mir_name = [s.name for s in scene.group_members(new) if "base" in s.name][0]
    mir_mesh = scene.get_shape(mir_name).build_mesh()
    lo_s, hi_s = src_mesh.bounds
    lo_m, hi_m = mir_mesh.bounds
    assert np.allclose(lo_m, [-hi_s[0], lo_s[1], lo_s[2]], atol=1e-9)
    assert np.allclose(hi_m, [-lo_s[0], hi_s[1], hi_s[2]], atol=1e-9)


def test_mirror_group_then_transform_original_only():
    scene = grouped_scene()
    new = scene.mirror_group("horn")
    scene.translate_group("horn", [0, 0, 5])
    # the mirrored copy must not move
    meta = scene.groups[new]
    assert np.allclose(scene.armature.joints[meta["joints"][0]].position, [-2, -3, 30])


def test_remove_group_cleans_everything():
    scene = grouped_scene()
    scene.save_pose("flex", {"horn:root": (30, 0, 0), "chest": (5, 0, 0)})
    scene.remove_group("horn")
    assert "horn" not in scene.groups
    assert scene.shape_names() == {"body"}
    assert "horn:root" not in scene.armature.joints
    assert "horn:tip" not in scene.armature.joints
    assert scene.poses["flex"] == {"chest": (5, 0, 0)}


def test_ungroup_keeps_shapes_and_joints():
    scene = grouped_scene()
    scene.ungroup("horn")
    assert "horn" not in scene.groups
    assert scene.get_shape("horn:base").group is None
    assert "horn:root" in scene.armature.joints


def test_joint_rename_and_remove_update_group_lists():
    scene = grouped_scene()
    scene.rename_joint("horn:tip", "horn:end")
    assert scene.groups["horn"]["joints"] == ["horn:root", "horn:end"]
    assert scene.get_shape("horn:base").bone == "horn:end"
    scene.remove_joint("horn:end")
    assert scene.groups["horn"]["joints"] == ["horn:root"]


def test_remove_last_member_prunes_group():
    scene = grouped_scene()
    scene.groups["horn"]["joints"] = []
    scene.remove_shape("horn:base")
    assert "horn" in scene.groups  # knob still there
    scene.remove_shape("horn:knob")
    assert "horn" not in scene.groups


def test_v2_round_trip_and_v1_compat(tmp_path):
    scene = grouped_scene()
    path = tmp_path / "g.mmp"
    scene.save(path)
    loaded = Scene.load(path)
    assert loaded.to_dict() == scene.to_dict()
    assert loaded.groups["horn"]["joints"] == ["horn:root", "horn:tip"]
    assert loaded.get_shape("horn:base").group == "horn"

    # v1 documents (no group fields) load unchanged
    data = scene.to_dict()
    data.pop("groups")
    data["version"] = 1
    for sd in data["shapes"]:
        sd.pop("group", None)
    v1 = Scene.from_dict(data)
    assert v1.groups == {}
    assert all(s.group is None for s in v1.shapes)


def test_load_cleans_dangling_group_data():
    scene = grouped_scene()
    data = scene.to_dict()
    data["shapes"][1]["group"] = "ghost-group"  # horn:base -> nonexistent group
    data["groups"]["horn"]["joints"].append("ghost-joint")
    loaded = Scene.from_dict(data)
    assert loaded.get_shape("horn:base").group is None
    assert loaded.groups["horn"]["joints"] == ["horn:root", "horn:tip"]

    # a group with no members and no joints disappears on load
    data2 = grouped_scene().to_dict()
    data2["groups"]["empty"] = {"part": "x", "joints": []}
    loaded2 = Scene.from_dict(data2)
    assert "empty" not in loaded2.groups


def test_ungrouped_shape_dict_has_no_group_key():
    scene = grouped_scene()
    data = scene.to_dict()
    by_name = {sd["name"]: sd for sd in data["shapes"]}
    assert "group" not in by_name["body"]
    assert by_name["horn:base"]["group"] == "horn"
