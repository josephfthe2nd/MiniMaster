import numpy as np
import pytest

from minimaster.core.armature import Armature


def two_bone_arm() -> Armature:
    arm = Armature()
    arm.add_joint("shoulder", [0, 0, 10])
    arm.add_joint("elbow", [5, 0, 10], parent="shoulder")
    arm.add_joint("wrist", [10, 0, 10], parent="elbow")
    return arm


def test_rest_pose_is_identity():
    arm = two_bone_arm()
    pos = arm.posed_positions({})
    assert np.allclose(pos["wrist"], [10, 0, 10])
    for m in arm.skin_matrices({}).values():
        assert np.allclose(m, np.eye(4))


def test_elbow_bend_moves_wrist_not_shoulder():
    arm = two_bone_arm()
    # Rotate elbow -90 about Y: forearm (+X direction) swings up to +Z.
    pose = {"elbow": (0, -90, 0)}
    pos = arm.posed_positions(pose)
    assert np.allclose(pos["shoulder"], [0, 0, 10])
    assert np.allclose(pos["elbow"], [5, 0, 10])
    assert np.allclose(pos["wrist"], [5, 0, 15], atol=1e-12)


def test_shoulder_rotation_carries_chain():
    arm = two_bone_arm()
    pose = {"shoulder": (0, 0, 90)}  # about Z: +X -> +Y
    pos = arm.posed_positions(pose)
    assert np.allclose(pos["elbow"], [0, 5, 10], atol=1e-12)
    assert np.allclose(pos["wrist"], [0, 10, 10], atol=1e-12)


def test_skin_matrix_moves_bound_point_with_bone():
    arm = two_bone_arm()
    pose = {"elbow": (0, -90, 0)}
    skins = arm.skin_matrices(pose)
    # A point mid-forearm at rest ([7.5, 0, 10]) should swing to [5, 0, 12.5].
    p = np.array([7.5, 0, 10, 1.0])
    moved = skins["wrist"] @ p
    assert np.allclose(moved[:3], [5, 0, 12.5], atol=1e-12)
    # Upper arm (bone "elbow") is unaffected by the elbow's own rotation.
    assert np.allclose(skins["elbow"], np.eye(4))


def test_rotation_at_leaf_moves_nothing():
    arm = two_bone_arm()
    pos = arm.posed_positions({"wrist": (45, 45, 45)})
    assert np.allclose(pos["wrist"], [10, 0, 10])


def test_nearest_bone():
    arm = two_bone_arm()
    assert arm.nearest_bone([2.5, 1, 10]) == "elbow"
    assert arm.nearest_bone([9, -1, 10]) == "wrist"
    empty = Armature()
    assert empty.nearest_bone([0, 0, 0]) is None
    only_root = Armature()
    only_root.add_joint("root", [0, 0, 0])
    assert only_root.nearest_bone([0, 0, 0]) is None


def test_add_remove_rename_reparent():
    arm = two_bone_arm()
    with pytest.raises(ValueError):
        arm.add_joint("elbow", [0, 0, 0])
    with pytest.raises(ValueError):
        arm.add_joint("x", [0, 0, 0], parent="nope")
    arm.remove_joint("elbow")
    assert arm.joints["wrist"].parent == "shoulder"
    arm.rename_joint("wrist", "hand")
    assert "hand" in arm.joints and "wrist" not in arm.joints
    with pytest.raises(ValueError):
        arm.reparent_joint("shoulder", "hand")  # cycle
    arm.add_joint("chest", [0, 0, 12])
    arm.reparent_joint("shoulder", "chest")
    assert arm.joints["shoulder"].parent == "chest"


def test_serialization_round_trip_out_of_order():
    arm = two_bone_arm()
    data = arm.to_dict()
    data["joints"].reverse()  # loader must tolerate child-before-parent
    loaded = Armature.from_dict(data)
    assert set(loaded.joints) == set(arm.joints)
    assert loaded.joints["wrist"].parent == "elbow"
    pos = loaded.posed_positions({"elbow": (0, -90, 0)})
    assert np.allclose(pos["wrist"], [5, 0, 15], atol=1e-12)


def test_orphan_joint_rejected():
    with pytest.raises(ValueError, match="orphan"):
        Armature.from_dict(
            {"joints": [{"name": "a", "parent": "ghost", "position": [0, 0, 0]}]}
        )
