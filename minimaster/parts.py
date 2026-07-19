"""Custom part library: reusable sub-assemblies you snap onto characters.

A part is a small bundle of shapes (and optionally its own joint chain) with
an attachment convention: local origin (0,0,0) is the attachment point and
local +Z points outward from the surface. Placing a part *flattens* it into
the scene — shapes are copied in under a fresh group name, joints are grafted
onto the armature — so .mmp projects stay fully self-contained and everything
downstream (posing, export gates, rendering, undo) works unchanged.

Grafting semantics: the part's root joint is parented to the *parent* joint of
the attachment bone. A shape bound to bone ``B`` (child joint of ``P``) skins
by ``W_P``, so the graft must follow ``P`` too — parenting to ``B`` itself
would add B's own rotation, shearing the part off its surface when B poses.

Library layout: shipped parts live in ``minimaster/parts/*.mmpart`` (with
``.png`` thumbnails alongside); user parts live in ``~/.minimaster/parts``
(override with the ``MINIMASTER_PARTS`` environment variable).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .core import math3d as m3
from .core.armature import Armature
from .scene import Scene, Shape

PART_FORMAT_NAME = "minimaster-part"
PART_FORMAT_VERSION = 1
PARTS_DIR = Path(__file__).parent / "parts"


def user_parts_dir() -> Path:
    override = os.environ.get("MINIMASTER_PARTS")
    if override:
        return Path(override)
    return Path.home() / ".minimaster" / "parts"


@dataclass
class Part:
    name: str
    category: str = "misc"
    shapes: list[dict] = field(default_factory=list)  # Shape.to_dict schema
    joints: list[dict] = field(default_factory=list)  # Joint.to_dict schema
    tags: list[str] = field(default_factory=list)

    # -- validation -------------------------------------------------------

    def validate(self) -> None:
        if not self.name:
            raise ValueError("part needs a name")
        if not self.shapes:
            raise ValueError("part has no shapes")
        arm = Armature.from_dict({"joints": self.joints})  # orphan/cycle checks
        roots = [j for j in arm.joints.values() if j.parent is None]
        if self.joints and len(roots) != 1:
            raise ValueError(
                f"part joints must form one tree with one root, got {len(roots)} roots"
            )
        joint_names = set(arm.joints)
        for sd in self.shapes:
            shape = Shape.from_dict({**sd, "group": None})
            if shape.bone is not None and shape.bone not in joint_names:
                raise ValueError(
                    f"shape {shape.name!r} binds unknown part joint {shape.bone!r}"
                )
            rep = shape.build_mesh().integrity_report()
            if not rep["watertight"] or not rep["outward"]:
                raise ValueError(
                    f"shape {shape.name!r} is not a printable shell: {rep}"
                )

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format": PART_FORMAT_NAME,
            "version": PART_FORMAT_VERSION,
            "name": self.name,
            "category": self.category,
            "tags": list(self.tags),
            "shapes": [dict(s) for s in self.shapes],
            "joints": [dict(j) for j in self.joints],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Part":
        if data.get("format") != PART_FORMAT_NAME:
            raise ValueError("not a MiniMaster part file")
        if int(data.get("version", 0)) > PART_FORMAT_VERSION:
            raise ValueError(
                f"part file version {data['version']} is newer than this "
                f"MiniMaster (supports up to {PART_FORMAT_VERSION})"
            )
        return cls(
            name=data["name"],
            category=data.get("category", "misc"),
            shapes=[dict(s) for s in data.get("shapes", [])],
            joints=[dict(j) for j in data.get("joints", [])],
            tags=list(data.get("tags", [])),
        )

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path) -> "Part":
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass
class PartInfo:
    name: str
    category: str
    path: Path
    thumbnail: Path | None
    user: bool


def list_parts() -> list[PartInfo]:
    """Shipped + user parts, sorted by (category, name). Unreadable files are
    skipped."""
    out: list[PartInfo] = []
    for base, is_user in ((PARTS_DIR, False), (user_parts_dir(), True)):
        if not base.is_dir():
            continue
        for path in sorted(base.glob("*.mmpart")):
            try:
                data = json.loads(path.read_text())
                if data.get("format") != PART_FORMAT_NAME:
                    continue
            except (OSError, json.JSONDecodeError):
                continue
            thumb = path.with_suffix(".png")
            out.append(
                PartInfo(
                    name=data.get("name", path.stem),
                    category=data.get("category", "misc"),
                    path=path,
                    thumbnail=thumb if thumb.is_file() else None,
                    user=is_user,
                )
            )
    return sorted(out, key=lambda i: (i.category, i.name))


def load_part(info_or_path) -> Part:
    path = info_or_path.path if isinstance(info_or_path, PartInfo) else info_or_path
    return Part.load(path)


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "part"
    return cleaned


def save_user_part(part: Part, thumbnail: bool = True) -> Path:
    part.validate()
    directory = user_parts_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_safe_filename(part.name)}.mmpart"
    part.save(path)
    if thumbnail:
        render_part_thumbnail(part, path.with_suffix(".png"))
    return path


# --------------------------------------------------------------------------
# placement


def place_part(
    scene: Scene,
    part: Part,
    point,
    normal=(0.0, 0.0, 1.0),
    attach_shape: str | None = None,
    attach_bone: str | None = None,
    scale: float = 1.0,
    spin: float = 0.0,
) -> str:
    """Stamp a part into the scene at ``point`` with local +Z aligned to
    ``normal``. Returns the new group name.

    The attachment bone comes from ``attach_bone`` if given, else from the
    clicked shape's binding, else from the nearest bone to the point. Part
    joints are grafted parents-first; the root is parented to the attachment
    bone's parent joint (see module docstring for why).
    """
    part.validate()
    point = np.asarray(point, dtype=np.float64)

    if attach_bone is None:
        if attach_shape is not None:
            # Respect the clicked shape's binding, even when it is unbound —
            # falling back to nearest_bone here could parasitically attach to
            # some other part's grafted bones.
            attach_bone = scene.get_shape(attach_shape).bone
        else:
            attach_bone = scene.armature.nearest_bone(point)
    if attach_bone is not None and attach_bone not in scene.armature.bone_names():
        raise ValueError(f"attach bone {attach_bone!r} does not exist")
    attach_parent = (
        scene.armature.joints[attach_bone].parent if attach_bone else None
    )

    group = scene._unique_group_name(part.name)
    rotation = m3.rotation_between([0.0, 0.0, 1.0], normal) @ m3.rot_z(spin)

    part_arm = Armature.from_dict({"joints": part.joints})
    grafted: dict[str, str] = {}
    for joint in part_arm._ordered():
        new_name = f"{group}:{joint.name}"
        parent = grafted[joint.parent] if joint.parent else attach_parent
        scene.armature.add_joint(new_name, joint.position, parent)
        grafted[joint.name] = new_name

    bone_names = set(scene.armature.bone_names())
    for sd in part.shapes:
        shape = Shape.from_dict({**sd, "group": None})
        shape.name = scene._unique_name(f"{group}:{shape.name}")
        shape.group = group
        if shape.bone is not None and grafted.get(shape.bone) in bone_names:
            shape.bone = grafted[shape.bone]
        else:
            shape.bone = attach_bone
        scene.shapes.append(shape)

    scene.groups[group] = {
        "part": part.name,
        "origin": [0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0],
        "scale": 1.0,
        "joints": [grafted[j] for j in part_arm.joints],
    }
    scene.transform_group(
        group,
        rotation=rotation,
        translation=point,
        scale=scale,
        pivot=(0.0, 0.0, 0.0),
    )
    return group


# --------------------------------------------------------------------------
# authoring


def part_from_selection(
    scene: Scene,
    shape_names: list[str],
    name: str,
    category: str = "custom",
    joint_root: str | None = None,
) -> Part:
    """Capture scene shapes (and optionally a joint subtree) as a reusable
    part.

    Origin: the root joint's position when ``joint_root`` is given (posable
    parts attach at their root), otherwise the bottom-center of the
    selection's bounding box. The selection keeps its current orientation —
    author parts pointing up (+Z = outward) for predictable placement.
    """
    if not shape_names:
        raise ValueError("select at least one shape to save as a part")
    shapes = [scene.get_shape(n) for n in shape_names]

    captured_joints: list[str] = []
    if joint_root is not None:
        if joint_root not in scene.armature.joints:
            raise ValueError(f"no joint named {joint_root!r}")

        def collect(jn: str):
            captured_joints.append(jn)
            for child in scene.armature.children(jn):
                collect(child)

        collect(joint_root)
    joint_set = set(captured_joints)

    if joint_root is not None:
        origin = scene.armature.joints[joint_root].position.copy()
    else:
        meshes = [s.build_mesh() for s in shapes]
        lo = np.min([m.bounds[0] for m in meshes], axis=0)
        hi = np.max([m.bounds[1] for m in meshes], axis=0)
        origin = np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, lo[2]])

    # When the whole selection is one placed part instance, strip its group
    # prefix so the saved part round-trips cleanly.
    groups = {s.group for s in shapes}
    prefix = ""
    if len(groups) == 1 and next(iter(groups)):
        prefix = next(iter(groups)) + ":"

    def inner(full: str) -> str:
        return full[len(prefix):] if prefix and full.startswith(prefix) else full

    shape_dicts = []
    for s in shapes:
        d = s.to_dict()
        d.pop("group", None)
        d["name"] = inner(d["name"])
        d["position"] = [float(v) for v in (s.position - origin)]
        d["bone"] = inner(s.bone) if s.bone in joint_set else None
        shape_dicts.append(d)

    joint_dicts = []
    for jn in captured_joints:
        j = scene.armature.joints[jn]
        joint_dicts.append(
            {
                "name": inner(jn),
                "parent": inner(j.parent) if j.parent in joint_set else None,
                "position": [float(v) for v in (j.position - origin)],
            }
        )

    part = Part(
        name=name,
        category=category,
        shapes=shape_dicts,
        joints=joint_dicts,
        tags=["posable"] if joint_dicts else [],
    )
    part.validate()
    return part


# --------------------------------------------------------------------------
# thumbnails


def part_preview_scene(part: Part) -> Scene:
    scene = Scene(name=part.name)
    scene.base = {"style": "none"}
    for sd in part.shapes:
        shape = Shape.from_dict({**sd, "group": None, "bone": None})
        shape.name = scene._unique_name(shape.name)
        scene.shapes.append(shape)
    return scene


def render_part_thumbnail(part: Part, path, size: tuple[int, int] = (96, 96)) -> None:
    from .render import render_scene

    render_scene(
        part_preview_scene(part),
        path=path,
        pose_name=None,
        with_base=False,
        size=size,
        azimuth=305.0,
        elevation=30.0,  # high 3/4 view keeps flat/swept parts legible
    )
