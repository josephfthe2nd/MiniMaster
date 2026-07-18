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
FORMAT_VERSION = 1

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
        return {
            "name": self.name,
            "kind": self.kind,
            "params": dict(self.params),
            "position": [float(v) for v in self.position],
            "rotation": [float(v) for v in self.rotation],
            "scale": [float(v) for v in self.scale],
            "bone": self.bone,
            "color": self.color,
        }

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
        )


class Scene:
    def __init__(self, name: str = "untitled"):
        self.name = name
        self.shapes: list[Shape] = []
        self.armature = Armature()
        self.poses: dict[str, Pose] = {}
        self.active_pose: str | None = None
        self.base: dict = {"style": "round", "diameter": 25.0}

    # -- shapes -----------------------------------------------------------

    def shape_names(self) -> set[str]:
        return {s.name for s in self.shapes}

    def _unique_name(self, base: str) -> str:
        names = self.shape_names()
        if base not in names:
            return base
        i = 1
        while f"{base}.{i:03d}" in names:
            i += 1
        return f"{base}.{i:03d}"

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
        self.shapes.remove(self.get_shape(name))

    def duplicate_shape(self, name: str, offset=(2.0, 0.0, 0.0)) -> Shape:
        src = self.get_shape(name)
        dup = Shape.from_dict(src.to_dict())
        dup.name = self._unique_name(src.name)
        dup.position = src.position + np.asarray(offset, dtype=np.float64)
        self.shapes.append(dup)
        return dup

    def mirror_shape(self, name: str) -> Shape:
        """Duplicate a shape mirrored across the X=0 plane (left <-> right).

        The mirrored copy keeps a watertight outward orientation (mesh
        transform flips winding for negative determinants). Bone binding is
        swapped between ``_l``/``_r`` suffixed bones when the counterpart
        exists.
        """
        src = self.get_shape(name)
        dup = Shape.from_dict(src.to_dict())
        dup.name = self._unique_name(src.name)
        dup.position = src.position * np.array([-1.0, 1.0, 1.0])
        dup.rotation = src.rotation * np.array([1.0, -1.0, -1.0])
        dup.scale = src.scale * np.array([-1.0, 1.0, 1.0])
        if src.bone:
            swapped = _swap_side(src.bone)
            if swapped in self.armature.joints:
                dup.bone = swapped
        self.shapes.append(dup)
        return dup

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

    def rename_joint(self, old: str, new: str) -> None:
        self.armature.rename_joint(old, new)
        for s in self.shapes:
            if s.bone == old:
                s.bone = new
        for pose in self.poses.values():
            if old in pose:
                pose[new] = pose.pop(old)

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
        return {
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
