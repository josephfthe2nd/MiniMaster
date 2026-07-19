import numpy as np
import pytest

from minimaster.core import math3d as m3
from minimaster.parts import (
    Part,
    list_parts,
    load_part,
    part_from_selection,
    place_part,
    render_part_thumbnail,
    save_user_part,
)
from minimaster.scene import Scene


def horn_part() -> Part:
    return Part(
        name="horn",
        category="head",
        shapes=[
            {
                "name": "spike",
                "kind": "cylinder",
                "params": {"segments": 4, "taper": 0.1},
                "position": [0, 0, 1.5],
                "rotation": [0, 0, 0],
                "scale": [1.2, 1.2, 3.0],
                "bone": None,
                "color": "#e8ddc4",
            }
        ],
    )


def tail_part() -> Part:
    return Part(
        name="tail",
        category="tail",
        shapes=[
            {
                "name": "seg1",
                "kind": "capsule",
                "params": {"segments": 6},
                "position": [0, 0, 2],
                "rotation": [0, 0, 0],
                "scale": [2, 2, 5],
                "bone": "mid",
                "color": "#7a8f4f",
            },
            {
                "name": "seg2",
                "kind": "capsule",
                "params": {"segments": 6},
                "position": [0, 0, 6],
                "rotation": [0, 0, 0],
                "scale": [1.4, 1.4, 4.5],
                "bone": "tip",
                "color": "#7a8f4f",
            },
        ],
        joints=[
            {"name": "root", "parent": None, "position": [0, 0, 0]},
            {"name": "mid", "parent": "root", "position": [0, 0, 4]},
            {"name": "tip", "parent": "mid", "position": [0, 0, 8]},
        ],
        tags=["posable"],
    )


def rigged_scene() -> Scene:
    scene = Scene(name="host")
    scene.armature.add_joint("pelvis", [0, 0, 10])
    scene.armature.add_joint("spine", [0, 0, 16], parent="pelvis")
    scene.armature.add_joint("chest", [0, 0, 22], parent="spine")
    scene.armature.add_joint("head", [0, 0, 30], parent="chest")
    scene.add_shape("box", name="torso", position=[0, 0, 19], scale=[8, 5, 10],
                    bone="chest")
    scene.add_shape("icosphere", name="skull", position=[0, 0, 28],
                    scale=[6, 6, 6], bone="head")
    return scene


# -- format ----------------------------------------------------------------


def test_part_round_trip(tmp_path):
    part = tail_part()
    path = tmp_path / "tail.mmpart"
    part.save(path)
    loaded = Part.load(path)
    assert loaded.to_dict() == part.to_dict()


def test_part_rejects_bad_format():
    with pytest.raises(ValueError, match="not a MiniMaster part"):
        Part.from_dict({"format": "other"})
    with pytest.raises(ValueError, match="newer"):
        Part.from_dict({"format": "minimaster-part", "version": 99, "name": "x"})


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda p: p.shapes.clear(), "no shapes"),
        (lambda p: p.joints.append({"name": "j2", "parent": None, "position": [0, 0, 0]}),
         "one root"),
        (lambda p: p.joints.append({"name": "j3", "parent": "ghost", "position": [0, 0, 0]}),
         "orphan"),
        (lambda p: p.shapes[0].update(bone="nope"), "unknown part joint"),
        (lambda p: p.shapes[0]["params"].update(radius=-1), "positive"),
    ],
)
def test_validation_failures(mutate, match):
    part = tail_part()
    mutate(part)
    with pytest.raises(ValueError, match=match):
        part.validate()


# -- placement -------------------------------------------------------------


def test_place_rigid_part_on_surface():
    scene = rigged_scene()
    normal = np.array([1.0, 0.0, 0.3])
    normal /= np.linalg.norm(normal)
    group = place_part(scene, horn_part(), point=[4, 0, 20], normal=normal,
                       attach_shape="torso")
    assert group == "horn"
    members = scene.group_members(group)
    assert len(members) == 1
    spike = members[0]
    assert spike.bone == "chest"  # clicked shape's bone
    # local +Z went to the surface normal
    outward = scene.group_outward_axis(group)
    assert np.allclose(outward, normal, atol=1e-9)
    # the shape's base sits at the hit point (origin maps to the point)
    assert np.allclose(scene.groups[group]["origin"], [4, 0, 20])
    rep = spike.build_mesh().integrity_report()
    assert rep["watertight"] and rep["outward"]


def test_place_with_scale_and_spin():
    scene = rigged_scene()
    g1 = place_part(scene, horn_part(), point=[0, 0, 34], scale=2.0)
    assert scene.groups[g1]["scale"] == pytest.approx(2.0)
    s = scene.group_members(g1)[0]
    assert np.allclose(s.scale, [2.4, 2.4, 6.0])
    # spin about the normal keeps the outward axis
    g2 = place_part(scene, horn_part(), point=[2, 0, 34], normal=[0, -1, 0],
                    spin=45.0)
    assert np.allclose(scene.group_outward_axis(g2), [0, -1, 0], atol=1e-9)


def test_place_posable_part_grafts_to_bone_parent():
    scene = rigged_scene()
    group = place_part(scene, tail_part(), point=[0, 2.5, 14],
                       normal=[0, 1, -0.2], attach_shape="torso")
    meta = scene.groups[group]
    assert set(meta["joints"]) == {f"{group}:root", f"{group}:mid", f"{group}:tip"}
    # C1: root grafts to the *parent* of the attachment bone (chest's parent
    # is spine), so the part follows the surface exactly
    assert scene.armature.joints[f"{group}:root"].parent == "spine"
    assert scene.armature.joints[f"{group}:mid"].parent == f"{group}:root"
    # internal bindings got prefixed
    segs = {s.name.split(":", 1)[1]: s for s in scene.group_members(group)}
    assert segs["seg1"].bone == f"{group}:mid"
    assert segs["seg2"].bone == f"{group}:tip"


def test_posing_attach_bone_does_not_move_part():
    """The C1 regression: posing the attachment bone's child joint must leave
    the part where the surface is; posing an ancestor carries it along."""
    scene = rigged_scene()
    group = place_part(scene, tail_part(), point=[0, 2.5, 14],
                       attach_shape="torso")  # torso bone = "chest"
    rest = {s.name: mesh.vertices.copy()
            for s, mesh in scene.build_shape_meshes(None) if s.group == group}

    # rotating the chest joint moves chest's children, not the torso surface
    scene.save_pose("p", {"chest": (40, 0, 0)})
    posed = {s.name: mesh.vertices
             for s, mesh in scene.build_shape_meshes("p") if s.group == group}
    for name in rest:
        assert np.allclose(posed[name], rest[name], atol=1e-9), name

    # rotating the spine (which moves the torso surface) carries the part
    scene.save_pose("q", {"spine": (30, 0, 0)})
    moved = {s.name: mesh.vertices
             for s, mesh in scene.build_shape_meshes("q") if s.group == group}
    assert any(not np.allclose(moved[n], rest[n]) for n in rest)

    # and the grafted joints themselves pose the part
    scene.save_pose("r", {f"{group}:mid": (45, 0, 0)})
    bent = {s.name: mesh.vertices
            for s, mesh in scene.build_shape_meshes("r") if s.group == group}
    seg2 = [n for n in bent if n.endswith("seg2")][0]
    assert not np.allclose(bent[seg2], rest[seg2])
    for _, mesh in scene.build_shape_meshes("r"):
        assert mesh.integrity_report()["watertight"]


def test_place_on_unrigged_scene():
    scene = Scene(name="empty")
    scene.add_shape("box", name="blob", position=[0, 0, 5], scale=[10, 10, 10])
    group = place_part(scene, tail_part(), point=[0, 5, 5], normal=[0, 1, 0])
    # no rig: grafted root is an armature root; seg1/seg2 still bind their
    # internal (non-root) joints
    assert scene.armature.joints[f"{group}:root"].parent is None
    segs = {s.name.split(":", 1)[1]: s for s in scene.group_members(group)}
    assert segs["seg1"].bone == f"{group}:mid"

    # clicking the unbound blob keeps the part unbound (no parasitic
    # attachment to the tail's grafted bones)
    rigid = place_part(scene, horn_part(), point=[0, -5, 5], normal=[0, -1, 0],
                       attach_shape="blob")
    assert scene.group_members(rigid)[0].bone is None
    # purely programmatic placement (no clicked shape) may fall back to the
    # nearest bone — here that's the grafted tail
    prog = place_part(scene, horn_part(), point=[0, 4, 5], normal=[0, 1, 0])
    assert scene.group_members(prog)[0].bone in {f"{group}:mid", f"{group}:tip"}


def test_duplicate_placement_names_unique():
    scene = rigged_scene()
    g1 = place_part(scene, horn_part(), point=[3, 0, 28], attach_shape="skull")
    g2 = place_part(scene, horn_part(), point=[-3, 0, 28], attach_shape="skull")
    assert g1 != g2
    assert len(scene.group_members(g1)) == len(scene.group_members(g2)) == 1


def test_place_invalid_attach_bone():
    scene = rigged_scene()
    with pytest.raises(ValueError, match="attach bone"):
        place_part(scene, horn_part(), point=[0, 0, 0], attach_bone="nope")


# -- authoring -------------------------------------------------------------


def test_part_from_selection_rigid_origin_bottom_center():
    scene = Scene(name="author")
    scene.add_shape("box", name="a", position=[10, 5, 8], scale=[2, 2, 4])
    scene.add_shape("box", name="b", position=[12, 5, 11], scale=[2, 2, 2])
    part = part_from_selection(scene, ["a", "b"], name="stack", category="misc")
    part.validate()
    # origin = bbox bottom-center: x in [9,13] -> 11, y -> 5, z bottom = 6
    by_name = {s["name"]: s for s in part.shapes}
    assert np.allclose(by_name["a"]["position"], [-1, 0, 2])
    assert np.allclose(by_name["b"]["position"], [1, 0, 5])


def test_part_from_selection_posable_and_round_trip():
    scene = rigged_scene()
    group = place_part(scene, tail_part(), point=[0, 2.5, 14],
                       attach_shape="torso")
    names = [s.name for s in scene.group_members(group)]
    saved = part_from_selection(scene, names, name="tail2", category="tail",
                                joint_root=f"{group}:root")
    # prefix stripped, internal structure restored relative to the root joint
    assert {j["name"] for j in saved.joints} == {"root", "mid", "tip"}
    assert {s["name"] for s in saved.shapes} == {"seg1", "seg2"}
    root = [j for j in saved.joints if j["parent"] is None][0]
    assert root["name"] == "root"
    assert np.allclose(root["position"], [0, 0, 0])

    # re-place the saved part elsewhere: same relative geometry
    scene2 = rigged_scene()
    g2 = place_part(scene2, saved, point=[0, 0, 0], normal=[0, 0, 1])
    tip = scene2.armature.joints[f"{g2}:tip"].position
    root_p = scene2.armature.joints[f"{g2}:root"].position
    assert np.linalg.norm(tip - root_p) == pytest.approx(8.0, rel=1e-6)


def test_part_from_selection_external_bones_dropped():
    scene = rigged_scene()
    part = part_from_selection(scene, ["skull"], name="ball", category="misc")
    assert part.shapes[0]["bone"] is None


def test_part_from_selection_empty_rejected():
    with pytest.raises(ValueError):
        part_from_selection(Scene(), [], name="x")


# -- library ---------------------------------------------------------------


def test_save_and_list_user_parts(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMASTER_PARTS", str(tmp_path / "userparts"))
    path = save_user_part(horn_part())
    assert path.is_file()
    assert path.with_suffix(".png").is_file()
    assert path.with_suffix(".png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    infos = [i for i in list_parts() if i.user]
    assert len(infos) == 1
    assert infos[0].name == "horn"
    assert infos[0].thumbnail is not None
    loaded = load_part(infos[0])
    assert loaded.name == "horn"

    # unreadable files are skipped
    (tmp_path / "userparts" / "junk.mmpart").write_text("{broken")
    assert len([i for i in list_parts() if i.user]) == 1


def test_thumbnail_render(tmp_path):
    out = tmp_path / "t.png"
    render_part_thumbnail(tail_part(), out, size=(48, 48))
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
