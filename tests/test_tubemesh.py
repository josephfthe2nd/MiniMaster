"""Skeletal tube mesher: RMF-framed quad tubes swept along the bone chains.

The guarantees under test: every region tube is watertight, outward, and
2-manifold *by construction* (and stays so under any pose, since posing only
moves vertices); the reference frame is twist-free; limbs are symmetric; the
build is deterministic; and it drops into the export pipeline.
"""

import numpy as np
import pytest

from minimaster.core import tubemesh as tm
from minimaster.core.mesh import Mesh
from minimaster.export import assemble_body
from minimaster.templates import load_template

TEMPLATES = ["human_fighter", "dwarf", "goblin", "orc", "skeleton", "four_arms"]


def _clean(mesh: Mesh) -> dict:
    return mesh.integrity_report()


def test_straight_tube_is_watertight_outward_manifold():
    mesh, bones, W = tm._build_tube(
        centers=[[0, 0, 0], [0, 0, 3], [0, 0, 6]],
        rx=[1, 1, 1], ry=[1, 1, 1], roll=[0, 0, 0],
        ring_weights=[{"a": 1.0}, {"a": 1.0}, {"a": 1.0}], N=8)
    r = _clean(mesh)
    assert r["watertight"] and r["outward"]
    assert r["boundary_edges"] == 0 and r["nonmanifold_edges"] == 0
    assert r["duplicate_directed_edges"] == 0 and r["degenerate_faces"] == 0
    assert mesh.volume() > 0
    assert W.shape == (len(mesh.vertices), len(bones))
    assert np.allclose(W.sum(axis=1), 1.0)


@pytest.mark.parametrize("name", TEMPLATES)
def test_every_region_tube_is_watertight_at_rest(name):
    scene = load_template(name)
    parts = tm.tube_body_regions(scene, pose_name="rest")
    assert parts
    for region, mesh, color in parts:
        r = _clean(mesh)
        assert r["watertight"], f"{name}/{region}: {r}"
        assert r["outward"], f"{name}/{region} not outward: {r}"
        assert color.startswith("#")


def test_four_arms_has_six_limb_tubes():
    scene = load_template("four_arms")
    regions = {r for r, _, _ in tm.tube_body_regions(scene, pose_name="rest")}
    assert {"shoulder_l", "shoulder_r", "shoulder2_l", "shoulder2_r",
            "hip_l", "hip_r"}.issubset(regions)


@pytest.mark.parametrize("name", ["human_fighter", "orc", "four_arms"])
def test_watertight_is_pose_invariant(name):
    # posing is a per-vertex affine map that never touches faces, so a mesh
    # that is watertight at rest stays watertight in every pose
    scene = load_template(name)
    for pose in [None, *scene.poses]:
        for region, mesh, _ in tm.tube_body_regions(scene, pose_name=pose or "rest"):
            r = _clean(mesh)
            assert r["watertight"], f"{name}/{pose}/{region}: {r}"
            assert r["boundary_edges"] == 0


def test_limbs_are_symmetric():
    # regression: a shield bound to wrist_l once ballooned the whole left arm
    # because gear polluted the radius mining
    for name in ("human_fighter", "orc", "four_arms"):
        scene = load_template(name)
        vols = {r: m.volume() for r, m, _ in tm.tube_body_regions(scene, pose_name="rest")}
        assert vols["shoulder_l"] == pytest.approx(vols["shoulder_r"], rel=1e-6)
        assert vols["hip_l"] == pytest.approx(vols["hip_r"], rel=1e-6)


def test_frame_is_twist_free_on_a_straight_chain():
    P = [[0, 0, z] for z in range(6)]
    T, U, V, s = tm.chain_frames(P, forward=(0, -1, 0))
    # U seeded from 'forward' and parallel-transported stays put on a line
    for i in range(len(U)):
        assert abs(abs(float(U[i] @ np.array([0, -1, 0]))) - 1.0) < 1e-9
    assert np.allclose(s, 1.0)  # no bend -> no miter inflation


def test_frame_is_stable_at_a_sharp_bend():
    # a right-angle chain must still yield a finite, unit, twist-minimal frame
    P = [[0, 0, 0], [0, 0, 3], [3, 0, 3]]
    T, U, V, s = tm.chain_frames(P)
    assert np.all(np.isfinite(U)) and np.all(np.isfinite(V))
    assert np.allclose(np.linalg.norm(U, axis=1), 1.0)
    assert np.allclose(np.einsum("ij,ij->i", U, T), 0.0, atol=1e-9)  # U _|_ T


def test_radius_floor_keeps_a_zero_radius_ring_watertight():
    # a genuine r=0 ring would collapse to a point (degenerate faces); the
    # floor in _build_tube must keep it watertight
    mesh, _, _ = tm._build_tube(
        centers=[[0, 0, 0], [0, 0, 3], [0, 0, 6]],
        rx=[1.0, 1.0, 0.0], ry=[1.0, 1.0, 0.0], roll=[0, 0, 0],
        ring_weights=[{"a": 1.0}] * 3, N=8)
    r = _clean(mesh)
    assert r["watertight"] and r["degenerate_faces"] == 0
    # and through the public profile= override (r=0 at a joint)
    scene = load_template("human_fighter")
    prof = tm.radius_profile_from_scene(scene)
    prof["elbow_l"] = {"rx": 0.0, "ry": 0.0, "roll": 0.0}
    for region, mesh, _ in tm.tube_body_regions(scene, profile=prof, pose_name="rest"):
        assert mesh.integrity_report()["watertight"], region


def test_non_humanoid_rig_falls_back_to_fuse():
    # a renamed/creature rig has no recognized bone chains; tube must not
    # silently produce an empty body — it falls back to the SDF path
    from minimaster.core.armature import Armature
    scene = load_template("human_fighter")  # 54 real body shapes
    a = Armature()
    a.add_joint("root", [0, 0, 0])
    a.add_joint("mid", [0, 0, 3], "root")
    a.add_joint("tip", [0, 0, 6], "mid")
    scene.armature = a
    assert tm.region_chains(a) == {}
    parts = tm.tube_body_regions(scene, pose_name="rest")
    assert parts, "tube should fall back to a non-empty body"
    assert len(tm.tube_body_from_scene(scene, pose_name="rest").faces) > 0


def test_region_colors_are_not_polluted_by_gear_or_head():
    # regression: a shield on wrist_l recolored the arm; the head folded into
    # the torso vote turned the orc's torso head-skin green
    scene = load_template("orc")
    cols = {r: c for r, _, c in tm.tube_body_regions(scene, pose_name="rest")}
    torso = next(s.color for s in scene.shapes if s.name == "chest")
    skin = next(s.color for s in scene.shapes if s.name == "head")
    assert cols["core"] == torso
    assert cols["head"] == skin
    assert cols["core"] != cols["head"]


def test_head_is_its_own_region():
    for name in ("human_fighter", "orc", "four_arms"):
        scene = load_template(name)
        regions = {r for r, _, _ in tm.tube_body_regions(scene, pose_name="rest")}
        assert "head" in regions


def test_forearm_is_not_stripped_from_radius_mining():
    assert tm._is_accessory("forearm_l") is False
    assert tm._is_accessory("forearm_flex_r") is False
    assert tm._is_accessory("ear_l") is True
    assert tm._is_accessory("shield_l") is True


def test_bind_bones_are_real_and_weights_normalized():
    scene = load_template("four_arms")
    valid = set(scene.armature.bone_names())
    for region, chain in tm.region_chains(scene.armature).items():
        profile = tm.radius_profile_from_scene(scene)
        _, bones, W = tm.tube_region(region, chain, scene, profile)
        assert set(bones).issubset(valid), f"{region} binds a non-bone: {set(bones) - valid}"
        assert np.allclose(W.sum(axis=1), 1.0)


def test_build_is_deterministic():
    scene = load_template("orc")
    a = tm.tube_body_from_scene(scene, pose_name="rest")
    b = tm.tube_body_from_scene(scene, pose_name="rest")
    assert np.array_equal(a.vertices, b.vertices)
    assert np.array_equal(a.faces, b.faces)


@pytest.mark.parametrize("name", ["orc", "four_arms"])
def test_assemble_body_tube_method(name):
    scene = load_template(name)
    mesh, rep = assemble_body(scene, pose_name="rest", method="tube",
                              size="medium", with_base=False)
    assert rep["method"] == "tube"
    assert rep["watertight"]
    lo, hi = mesh.bounds
    assert hi[2] - lo[2] == pytest.approx(32.0)


def test_unknown_method_rejected():
    from minimaster.export import ExportError
    scene = load_template("orc")
    with pytest.raises(ExportError):
        assemble_body(scene, method="nope")
