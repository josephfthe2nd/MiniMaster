"""The dynamic body mesh: SDF blend -> Surface Nets isosurface -> skinning.

These lock the two guarantees that matter: the extracted body is a watertight,
outward-oriented solid, and the smooth skin weights are a valid partition of
unity that actually deforms the mesh under a pose.
"""

import numpy as np
import pytest

from minimaster.core import bodymesh as bm
from minimaster.core.armature import Armature
from minimaster.export import assemble_body
from minimaster.scene import Scene
from minimaster.templates import load_template

TEMPLATES = ["human_fighter", "dwarf", "goblin", "orc", "skeleton"]


def _sphere_scene(radius):
    sc = Scene()
    sc.add_shape("icosphere", name="ball", params={"radius": 0.5},
                 scale=[2 * radius] * 3)
    return sc


def test_isosurface_sphere_is_watertight_and_sized():
    mesh = bm.build_field_mesh(_sphere_scene(5.0).shapes, resolution=0.5)
    rep = mesh.integrity_report()
    assert rep["watertight"] and rep["outward"]
    # Surface Nets on a coarse grid under-fills slightly; within a few percent.
    assert mesh.volume() == pytest.approx(4 / 3 * np.pi * 5**3, rel=0.05)


def test_smooth_union_fuses_two_shapes_into_one_solid():
    sc = Scene()
    sc.add_shape("capsule", name="a", params={"radius": 0.5},
                 position=[-3, 0, 0], rotation=[0, 90, 0], scale=[6, 6, 16])
    sc.add_shape("icosphere", name="b", params={"radius": 0.5},
                 position=[5, 0, 0], scale=[9, 9, 9])
    mesh = bm.build_field_mesh(sc.shapes, resolution=0.5, blend=1.5)
    rep = mesh.integrity_report()
    assert rep["watertight"] and rep["outward"]
    assert rep["degenerate_faces"] == 0


def test_vertices_are_clamped_inside_cells_no_degenerate_faces():
    # coincident vertices (zero-area slivers) are what break watertightness;
    # the in-cell clamp must prevent them across resolutions
    for res in (0.3, 0.55, 0.8):
        mesh = bm.build_field_mesh(_sphere_scene(6.0).shapes, resolution=res)
        assert mesh.integrity_report()["degenerate_faces"] == 0


@pytest.mark.parametrize("name", TEMPLATES)
def test_template_bakes_to_watertight_solid(name):
    scene = load_template(name)
    mesh, rep = assemble_body(scene, pose_name="rest", resolution=0.7,
                              blend=0.7, size="medium", with_base=False)
    assert rep["watertight"], f"{name} baked non-watertight (blend {rep['blend']})"
    assert rep["triangles"] > 0
    lo, hi = mesh.bounds
    assert hi[2] - lo[2] == pytest.approx(32.0)  # scaled to the size preset


def test_bake_is_deterministic():
    scene = load_template("orc")
    a = bm.body_from_scene(scene, resolution=0.7, blend=0.6, pose_name="rest")
    b = bm.body_from_scene(scene, resolution=0.7, blend=0.6, pose_name="rest")
    assert np.array_equal(a.vertices, b.vertices)
    assert np.array_equal(a.faces, b.faces)


def test_accessory_shapes_excluded_by_default():
    # the human fighter carries a sword/shield/cape; the default body drops them
    scene = load_template("human_fighter")
    body = bm.body_from_scene(scene, resolution=0.9, blend=0.6, pose_name="rest")
    everything = bm.body_from_scene(scene, resolution=0.9, blend=0.6,
                                    pose_name="rest", exclude=())
    # fusing the gear too reaches noticeably wider in X (the sword/shield)
    assert everything.bounds[1][0] - everything.bounds[0][0] > \
        body.bounds[1][0] - body.bounds[0][0]


# -- skinning --------------------------------------------------------------


def _two_bone_arm():
    arm = Armature()
    arm.add_joint("root", [0, 0, 0])
    arm.add_joint("mid", [0, 0, 5], parent="root")
    arm.add_joint("tip", [0, 0, 10], parent="mid")
    return arm


def test_skin_weights_are_a_partition_of_unity():
    arm = _two_bone_arm()
    verts = np.array([[0, 0, 2.5], [0, 0, 7.5], [1, 0, 5.0]], dtype=float)
    names, W = bm.skin_weights(verts, arm, max_bones=2)
    assert names == ["mid", "tip"]  # bone keys are child-joint names
    assert W.shape == (3, 2)
    assert np.allclose(W.sum(axis=1), 1.0)
    assert (W >= 0).all()
    # the low vertex leans to the lower bone, the high one to the upper bone
    assert W[0, 0] > W[0, 1]
    assert W[1, 1] > W[1, 0]


def test_skin_weights_respect_max_bones():
    arm = Armature()
    arm.add_joint("a", [0, 0, 0])
    for i in range(1, 6):
        arm.add_joint(f"b{i}", [i, 0, i], parent="a" if i == 1 else f"b{i-1}")
    verts = np.random.default_rng(0).uniform(-1, 6, size=(20, 3))
    _, W = bm.skin_weights(verts, arm, max_bones=3)
    assert (np.count_nonzero(W, axis=1) <= 3).all()
    assert np.allclose(W.sum(axis=1), 1.0)


def test_pose_body_mesh_rest_is_identity_and_pose_deforms():
    scene = load_template("human_fighter")
    body = bm.body_from_scene(scene, resolution=0.9, blend=0.6, pose_name="rest")
    names, W = bm.skin_weights(body.vertices, scene.armature)
    rest = bm.pose_body_mesh(body, names, W, scene.armature, {})
    assert np.allclose(rest.vertices, body.vertices)  # empty pose = identity
    posed = bm.pose_body_mesh(body, names, W, scene.armature,
                              {"shoulder_l": (-90.0, 0.0, 0.0)})
    assert posed.faces.shape == body.faces.shape  # topology preserved
    moved = np.linalg.norm(posed.vertices - body.vertices, axis=1)
    assert moved.max() > 1.0  # the raised arm's vertices actually move
