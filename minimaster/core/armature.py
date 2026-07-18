"""Editable joint hierarchies with forward-kinematics posing.

An armature is a set of named joints; each non-root joint implicitly defines a
*bone* from its parent joint to itself. Bindings and skinning are keyed by the
bone's child joint name (unique per bone).

Posing semantics: a pose maps joint names to XYZ euler degrees. The rotation
stored at a joint spins that joint's *outgoing* bones (its whole subtree)
about the joint's own position — rotate the elbow to bend the forearm. A shape
bound to bone ``p -> j`` therefore follows the accumulated rotation of ``p``
and its ancestors:

    W_root = T(rest_root) @ R_root
    W_j    = W_parent @ T(rest_j - rest_parent) @ R_j
    skin(bone p -> j) = W_p @ T(-rest_p)

At rest (all rotations zero) every skin matrix is the identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import math3d as m3

Pose = dict[str, tuple[float, float, float]]


@dataclass
class Joint:
    name: str
    position: np.ndarray  # rest position, scene space
    parent: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "parent": self.parent,
            "position": [float(v) for v in self.position],
        }


class Armature:
    def __init__(self):
        self.joints: dict[str, Joint] = {}

    # -- editing ----------------------------------------------------------

    def add_joint(self, name: str, position, parent: str | None = None) -> Joint:
        if not name:
            raise ValueError("joint name must be non-empty")
        if name in self.joints:
            raise ValueError(f"joint {name!r} already exists")
        if parent is not None and parent not in self.joints:
            raise ValueError(f"parent joint {parent!r} does not exist")
        j = Joint(name, np.asarray(position, dtype=np.float64).copy(), parent)
        self.joints[name] = j
        return j

    def remove_joint(self, name: str) -> None:
        """Remove a joint; its children are reparented to its parent (or become
        roots)."""
        j = self.joints.pop(name)
        for child in self.joints.values():
            if child.parent == name:
                child.parent = j.parent

    def rename_joint(self, old: str, new: str) -> None:
        if new in self.joints:
            raise ValueError(f"joint {new!r} already exists")
        j = self.joints.pop(old)
        j.name = new
        # Rebuild preserving order so serialization stays parent-before-child.
        renamed = {}
        for key, joint in self.joints.items():
            if joint.parent == old:
                joint.parent = new
            renamed[key] = joint
        renamed[new] = j
        self.joints = {}
        for joint in renamed.values():
            self.joints[joint.name] = joint

    def move_joint(self, name: str, position) -> None:
        self.joints[name].position = np.asarray(position, dtype=np.float64).copy()

    def reparent_joint(self, name: str, new_parent: str | None) -> None:
        if new_parent is not None:
            if new_parent not in self.joints:
                raise ValueError(f"parent joint {new_parent!r} does not exist")
            if name == new_parent or name in self.ancestors(new_parent):
                raise ValueError("reparenting would create a cycle")
        self.joints[name].parent = new_parent

    def ancestors(self, name: str) -> list[str]:
        out = []
        parent = self.joints[name].parent
        while parent is not None:
            out.append(parent)
            parent = self.joints[parent].parent
        return out

    def children(self, name: str) -> list[str]:
        return [j.name for j in self.joints.values() if j.parent == name]

    def bones(self) -> list[tuple[str, str]]:
        """(parent, child) pairs; the child name is the bone's key."""
        return [(j.parent, j.name) for j in self.joints.values() if j.parent is not None]

    def bone_names(self) -> list[str]:
        return [child for _, child in self.bones()]

    # -- kinematics -------------------------------------------------------

    def _ordered(self) -> list[Joint]:
        """Joints sorted parents-before-children (stable)."""
        seen: dict[str, Joint] = {}

        def visit(j: Joint):
            if j.name in seen:
                return
            if j.parent is not None:
                visit(self.joints[j.parent])
            seen[j.name] = j

        for j in self.joints.values():
            visit(j)
        return list(seen.values())

    def world_matrices(self, pose: Pose | None = None) -> dict[str, np.ndarray]:
        pose = pose or {}
        out: dict[str, np.ndarray] = {}
        for j in self._ordered():
            angles = pose.get(j.name, (0.0, 0.0, 0.0))
            local_rot = m3.mat4(rotation=m3.euler_rotation(*angles))
            if j.parent is None:
                out[j.name] = m3.translation_mat(j.position) @ local_rot
            else:
                offset = j.position - self.joints[j.parent].position
                out[j.name] = out[j.parent] @ m3.translation_mat(offset) @ local_rot
        return out

    def posed_positions(self, pose: Pose | None = None) -> dict[str, np.ndarray]:
        return {name: w[:3, 3].copy() for name, w in self.world_matrices(pose).items()}

    def skin_matrices(self, pose: Pose | None = None) -> dict[str, np.ndarray]:
        """Bone key (child joint name) -> 4x4 mapping rest scene space to posed
        scene space for shapes bound to that bone."""
        world = self.world_matrices(pose)
        out = {}
        for parent, child in self.bones():
            rest_p = self.joints[parent].position
            out[child] = world[parent] @ m3.translation_mat(-rest_p)
        return out

    # -- binding ----------------------------------------------------------

    def nearest_bone(self, point) -> str | None:
        """Child-joint name of the bone segment closest to ``point`` at rest,
        or None when the armature has no bones."""
        best, best_d = None, np.inf
        for parent, child in self.bones():
            d = m3.point_segment_distance(
                point, self.joints[parent].position, self.joints[child].position
            )
            if d < best_d:
                best, best_d = child, d
        return best

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> dict:
        return {"joints": [j.to_dict() for j in self._ordered()]}

    @classmethod
    def from_dict(cls, data: dict) -> "Armature":
        arm = cls()
        pending = list(data.get("joints", []))
        # Insert parents first regardless of stored order.
        progress = True
        while pending and progress:
            progress = False
            remaining = []
            for jd in pending:
                parent = jd.get("parent")
                if parent is None or parent in arm.joints:
                    arm.add_joint(jd["name"], jd["position"], parent)
                    progress = True
                else:
                    remaining.append(jd)
            pending = remaining
        if pending:
            names = [jd["name"] for jd in pending]
            raise ValueError(f"armature has orphaned/cyclic joints: {names}")
        return arm

    def __len__(self) -> int:
        return len(self.joints)
