"""Scene model: shapes + armature + poses, JSON project files, undo.

A scene is the complete document the app edits — the list of primitive shapes
(with their transforms and bone bindings), the user-built armature, named
poses, and the base choice. Scenes serialize to `.mmp` files (JSON).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .core import math3d as m3
from .core import primitives
from .core.armature import Armature, Pose
from .core.mesh import Mesh

FORMAT_NAME = "minimaster-scene"
FORMAT_VERSION = 2  # v2 adds optional shape "group" fields + scene "groups"


def _unique_in(names, base: str) -> str:
    """base, or base.001-style suffixed variant, not present in ``names``."""
    if base not in names:
        return base
    i = 1
    while f"{base}.{i:03d}" in names:
        i += 1
    return f"{base}.{i:03d}"

DEFAULT_COLORS = {
    "box": "#b08d57",
    "wedge": "#a07e4f",
    "cylinder": "#9c8b6d",
    "capsule": "#ad9264",
    "icosphere": "#b39a6b",
    "torus": "#8f7f5f",
}


@dataclass
class Shape:
    name: str
    kind: str
    params: dict = field(default_factory=dict)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3))  # XYZ euler deg
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    bone: str | None = None
    color: str = "#b08d57"
    group: str | None = None  # part-instance membership (see Scene.groups)

    def __post_init__(self):
        # np.array always copies: the shape owns its vectors even when a
        # caller passes a shared ndarray.
        self.position = np.array(self.position, dtype=np.float64)
        self.rotation = np.array(self.rotation, dtype=np.float64)
        self.scale = np.array(self.scale, dtype=np.float64)
        if self.kind not in primitives.PRIMITIVES:
            raise ValueError(f"unknown primitive kind {self.kind!r}")

    def local_matrix(self) -> np.ndarray:
        return m3.compose_trs(self.position, self.rotation, self.scale)

    def build_mesh(self) -> Mesh:
        """Shape mesh in rest scene space."""
        return primitives.build(self.kind, self.params).transform(self.local_matrix())

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "kind": self.kind,
            "params": dict(self.params),
            "position": [float(v) for v in self.position],
            "rotation": [float(v) for v in self.rotation],
            "scale": [float(v) for v in self.scale],
            "bone": self.bone,
            "color": self.color,
        }
        if self.group is not None:  # omit-when-unset keeps v1-era dicts stable
            d["group"] = self.group
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Shape":
        return cls(
            name=d["name"],
            kind=d["kind"],
            params=dict(d.get("params", {})),
            position=d.get("position", [0, 0, 0]),
            rotation=d.get("rotation", [0, 0, 0]),
            scale=d.get("scale", [1, 1, 1]),
            bone=d.get("bone"),
            color=d.get("color", DEFAULT_COLORS.get(d["kind"], "#b08d57")),
            group=d.get("group"),
        )


class Scene:
    def __init__(self, name: str = "untitled"):
        self.name = name
        self.shapes: list[Shape] = []
        self.armature = Armature()
        self.poses: dict[str, Pose] = {}
        self.active_pose: str | None = None
        self.base: dict = {"style": "round", "diameter": 25.0}
        # Part instances: group name -> {"part", "origin", "rotation",
        # "scale", "joints": [grafted joint names]}
        self.groups: dict[str, dict] = {}

    # -- shapes -----------------------------------------------------------

    def shape_names(self) -> set[str]:
        return {s.name for s in self.shapes}

    def _unique_name(self, base: str) -> str:
        return _unique_in(self.shape_names(), base)

    def _unique_group_name(self, base: str) -> str:
        return _unique_in(set(self.groups), base)

    def add_shape(self, kind: str, name: str | None = None, **fields) -> Shape:
        shape = Shape(
            name=self._unique_name(name or kind),
            kind=kind,
            color=fields.pop("color", DEFAULT_COLORS.get(kind, "#b08d57")),
            **fields,
        )
        self.shapes.append(shape)
        return shape

    def get_shape(self, name: str) -> Shape:
        for s in self.shapes:
            if s.name == name:
                return s
        raise KeyError(f"no shape named {name!r}")

    def remove_shape(self, name: str) -> None:
        shape = self.get_shape(name)
        self.shapes.remove(shape)
        if shape.group:
            self._prune_group_if_empty(shape.group)

    def duplicate_shape(self, name: str, offset=(2.0, 0.0, 0.0)) -> Shape:
        src = self.get_shape(name)
        dup = Shape.from_dict(src.to_dict())
        dup.name = self._unique_name(src.name)
        dup.position = src.position + np.asarray(offset, dtype=np.float64)
        self.shapes.append(dup)
        return dup

    @staticmethod
    def _mirrored_fields(src: Shape) -> Shape:
        """Copy of a shape reflected across X=0 (equivalent to composing its
        matrix with diag(-1,1,1); winding stays outward via the negative-scale
        flip in Mesh.transform)."""
        dup = Shape.from_dict(src.to_dict())
        dup.position = src.position * np.array([-1.0, 1.0, 1.0])
        dup.rotation = src.rotation * np.array([1.0, -1.0, -1.0])
        dup.scale = src.scale * np.array([-1.0, 1.0, 1.0])
        return dup

    def mirror_shape(self, name: str) -> Shape:
        """Duplicate a shape mirrored across the X=0 plane (left <-> right).

        Bone binding is swapped between ``_l``/``_r`` suffixed bones when the
        counterpart exists.
        """
        src = self.get_shape(name)
        dup = self._mirrored_fields(src)
        dup.name = self._unique_name(src.name)
        dup.group = None  # a lone mirrored shape leaves its part instance
        if src.bone:
            swapped = _swap_side(src.bone)
            if swapped in self.armature.joints:
                dup.bone = swapped
        self.shapes.append(dup)
        return dup

    # -- part-instance groups ---------------------------------------------

    def group_members(self, name: str) -> list[Shape]:
        return [s for s in self.shapes if s.group == name]

    def _prune_group_if_empty(self, name: str) -> None:
        meta = self.groups.get(name)
        if meta is None:
            return
        has_shapes = any(s.group == name for s in self.shapes)
        has_joints = any(j in self.armature.joints for j in meta.get("joints", []))
        if not has_shapes and not has_joints:
            self.groups.pop(name)

    def transform_group(
        self,
        name: str,
        rotation=None,
        translation=(0.0, 0.0, 0.0),
        scale: float = 1.0,
        pivot=None,
    ) -> None:
        """Move a whole part instance: rigid rotation + translation + uniform
        scale about ``pivot`` (default: the group's stored origin).

        The delta must be rigid + uniform because uniform scale is the only
        scale that commutes with member rotations; a non-uniform delta would
        shear rotated members out of the TRS parameterization. Members' own
        per-axis (even negative/mirrored) scales are preserved.
        """
        meta = self.groups[name]
        if scale <= 0:
            raise ValueError(
                "group scale must be positive (mirroring is a separate operation)"
            )
        if rotation is None:
            rg = np.eye(3)
        else:
            rg = np.asarray(rotation, dtype=np.float64)
            if rg.shape != (3, 3):
                rg = m3.euler_rotation(*rotation)
        t = np.asarray(translation, dtype=np.float64)
        c = np.asarray(
            pivot if pivot is not None else meta["origin"], dtype=np.float64
        )

        def map_point(p):
            return c + t + scale * (rg @ (np.asarray(p, dtype=np.float64) - c))

        for s in self.group_members(name):
            s.position = map_point(s.position)
            s.rotation = np.array(
                m3.matrix_to_euler_xyz(rg @ m3.euler_rotation(*s.rotation))
            )
            s.scale = s.scale * scale
        for jname in meta.get("joints", []):
            if jname in self.armature.joints:
                self.armature.move_joint(
                    jname, map_point(self.armature.joints[jname].position)
                )
        meta["origin"] = [float(v) for v in map_point(meta["origin"])]
        meta["rotation"] = list(
            m3.matrix_to_euler_xyz(rg @ m3.euler_rotation(*meta["rotation"]))
        )
        meta["scale"] = float(meta.get("scale", 1.0) * scale)

    def translate_group(self, name: str, delta) -> None:
        self.transform_group(name, translation=delta)

    def group_outward_axis(self, name: str) -> np.ndarray:
        """The group's current outward (+Z at stamp time) axis."""
        meta = self.groups[name]
        return m3.euler_rotation(*meta["rotation"]) @ np.array([0.0, 0.0, 1.0])

    def mirror_group(self, name: str) -> str:
        """Duplicate a part instance mirrored across X=0, including grafted
        joints (poses on the source joints are not copied). Returns the new
        group name."""
        meta = self.groups[name]
        new_name = self._unique_group_name(meta.get("part", name))
        prefix = name + ":"

        def inner_of(full: str) -> str:
            return full[len(prefix):] if full.startswith(prefix) else full

        joint_map: dict[str, str] = {}
        group_joints = [j for j in meta.get("joints", []) if j in self.armature.joints]
        ordered = [j for j in self.armature._ordered() if j.name in set(group_joints)]
        for j in ordered:
            mirrored = f"{new_name}:{inner_of(j.name)}"
            while mirrored in self.armature.joints:
                mirrored += "_m"
            if j.parent in joint_map:
                parent = joint_map[j.parent]
            elif j.parent is not None:
                swapped = _swap_side(j.parent)
                parent = swapped if swapped in self.armature.joints else j.parent
            else:
                parent = None
            self.armature.add_joint(
                mirrored, j.position * np.array([-1.0, 1.0, 1.0]), parent
            )
            joint_map[j.name] = mirrored

        bones = set(self.armature.bone_names())
        for src in self.group_members(name):
            dup = self._mirrored_fields(src)
            dup.name = self._unique_name(f"{new_name}:{inner_of(src.name)}")
            dup.group = new_name
            if src.bone in joint_map:
                dup.bone = joint_map[src.bone]
            elif src.bone:
                swapped = _swap_side(src.bone)
                if swapped in bones:
                    dup.bone = swapped
            self.shapes.append(dup)

        rx, ry, rz = meta["rotation"]
        self.groups[new_name] = {
            "part": meta.get("part", name),
            "origin": [
                -float(meta["origin"][0]),
                float(meta["origin"][1]),
                float(meta["origin"][2]),
            ],
            "rotation": [float(rx), -float(ry), -float(rz)],
            "scale": float(meta.get("scale", 1.0)),
            "joints": [joint_map[j] for j in group_joints],
        }
        return new_name

    def remove_group(self, name: str) -> None:
        meta = self.groups.pop(name)
        for s in self.group_members(name):
            self.shapes.remove(s)
        for jname in list(meta.get("joints", [])):
            if jname in self.armature.joints:
                self.remove_joint(jname)

    def ungroup(self, name: str) -> None:
        """Dissolve the part instance: members become ordinary shapes and any
        grafted joints become ordinary joints."""
        for s in self.group_members(name):
            s.group = None
        self.groups.pop(name)

    # -- armature interplay ----------------------------------------------

    def remove_joint(self, name: str) -> None:
        """Remove a joint; children reparent to its parent (Armature policy).

        Shapes bound to the removed bone are unbound. Shapes bound to a
        reparented child's bone stay bound — that bone still exists, now
        pivoting at the grandparent.
        """
        self.armature.remove_joint(name)
        valid = set(self.armature.bone_names())
        for s in self.shapes:
            if s.bone is not None and s.bone not in valid:
                s.bone = None
        for pose in self.poses.values():
            pose.pop(name, None)
        for gname, meta in list(self.groups.items()):
            if name in meta.get("joints", []):
                meta["joints"] = [j for j in meta["joints"] if j != name]
                self._prune_group_if_empty(gname)

    def rename_joint(self, old: str, new: str) -> None:
        self.armature.rename_joint(old, new)
        for s in self.shapes:
            if s.bone == old:
                s.bone = new
        for pose in self.poses.values():
            if old in pose:
                pose[new] = pose.pop(old)
        for meta in self.groups.values():
            meta["joints"] = [new if j == old else j for j in meta.get("joints", [])]

    def auto_bind(self, only: list[str] | None = None) -> None:
        """Bind shapes to the nearest bone (by shape origin, rest space)."""
        targets = self.shapes if only is None else [self.get_shape(n) for n in only]
        for s in targets:
            s.bone = self.armature.nearest_bone(s.position)

    # -- poses ------------------------------------------------------------

    def resolve_pose(self, pose_name: str | None = "__active__") -> Pose:
        if pose_name == "__active__":
            pose_name = self.active_pose
        if pose_name is None:
            return {}
        if pose_name not in self.poses:
            raise KeyError(f"no pose named {pose_name!r}")
        return self.poses[pose_name]

    def save_pose(self, name: str, pose: Pose) -> None:
        self.poses[name] = {
            k: tuple(float(a) for a in v) for k, v in pose.items() if any(v)
        }

    # -- building ---------------------------------------------------------

    def build_shape_meshes(
        self, pose_name: str | None = "__active__"
    ) -> list[tuple[Shape, Mesh]]:
        """Posed scene-space mesh for every shape."""
        pose = self.resolve_pose(pose_name)
        skins = self.armature.skin_matrices(pose) if pose else {}
        out = []
        for s in self.shapes:
            mesh = s.build_mesh()
            if s.bone and s.bone in skins:
                mesh = mesh.transform(skins[s.bone])
            out.append((s, mesh))
        return out

    def build_merged_mesh(self, pose_name: str | None = "__active__") -> Mesh:
        return Mesh.merge([mesh for _, mesh in self.build_shape_meshes(pose_name)])

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> dict:
        d = {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "name": self.name,
            "shapes": [s.to_dict() for s in self.shapes],
            "armature": self.armature.to_dict(),
            "poses": {
                name: {j: list(a) for j, a in pose.items()}
                for name, pose in self.poses.items()
            },
            "active_pose": self.active_pose,
            "base": dict(self.base),
        }
        if self.groups:
            d["groups"] = {k: dict(v) for k, v in self.groups.items()}
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Scene":
        if data.get("format") != FORMAT_NAME:
            raise ValueError("not a MiniMaster scene file")
        if int(data.get("version", 0)) > FORMAT_VERSION:
            raise ValueError(
                f"scene file version {data['version']} is newer than this "
                f"MiniMaster (supports up to {FORMAT_VERSION})"
            )
        scene = cls(name=data.get("name", "untitled"))
        scene.armature = Armature.from_dict(data.get("armature", {}))
        names = set()
        for sd in data.get("shapes", []):
            shape = Shape.from_dict(sd)
            if shape.name in names:
                raise ValueError(f"duplicate shape name {shape.name!r}")
            names.add(shape.name)
            scene.shapes.append(shape)
        valid_bones = set(scene.armature.bone_names())
        for s in scene.shapes:
            if s.bone is not None and s.bone not in valid_bones:
                s.bone = None
        joints = set(scene.armature.joints)
        scene.poses = {
            name: {
                j: tuple(float(a) for a in ang)
                for j, ang in pose.items()
                if j in joints  # drop entries for joints that no longer exist
            }
            for name, pose in data.get("poses", {}).items()
        }
        scene.active_pose = data.get("active_pose")
        if scene.active_pose is not None and scene.active_pose not in scene.poses:
            scene.active_pose = None
        scene.base = dict(data.get("base", {"style": "round", "diameter": 25.0}))

        scene.groups = {k: dict(v) for k, v in data.get("groups", {}).items()}
        for s in scene.shapes:
            if s.group is not None and s.group not in scene.groups:
                s.group = None
        for gname, meta in scene.groups.items():
            meta["joints"] = [
                j for j in meta.get("joints", []) if j in scene.armature.joints
            ]
            meta.setdefault("part", gname)
            meta.setdefault("origin", [0.0, 0.0, 0.0])
            meta.setdefault("rotation", [0.0, 0.0, 0.0])
            meta.setdefault("scale", 1.0)
        for gname in list(scene.groups):
            scene._prune_group_if_empty(gname)
        return scene

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Scene":
        return cls.from_dict(json.loads(text))

    def save(self, path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path) -> "Scene":
        return cls.from_json(Path(path).read_text())


def _swap_side(name: str) -> str:
    for a, b in (("_l", "_r"), ("_left", "_right")):
        if name.endswith(a):
            return name[: -len(a)] + b
        if name.endswith(b):
            return name[: -len(b)] + a
    return name


class UndoStack:
    """Snapshot-based undo/redo over scene JSON (scene documents are tiny)."""

    def __init__(self, limit: int = 100):
        self.limit = limit
        self._undo: list[str] = []
        self._redo: list[str] = []

    def push(self, snapshot: str) -> None:
        if self._undo and self._undo[-1] == snapshot:
            return
        self._undo.append(snapshot)
        if len(self._undo) > self.limit:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self, current: str) -> str | None:
        if not self._undo:
            return None
        self._redo.append(current)
        return self._undo.pop()

    def redo(self, current: str) -> str | None:
        if not self._redo:
            return None
        self._undo.append(current)
        return self._redo.pop()

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)
