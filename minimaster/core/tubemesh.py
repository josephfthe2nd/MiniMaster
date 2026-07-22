"""Skeletal tube mesher: build a body from bones, the Skin-Modifier way.

Instead of fusing overlapping primitives (which melts limbs together and drops
thin joints), this sweeps a ring of vertices along each bone chain of the
armature and stitches them into one clean, watertight, quad-dominant **tube per
anatomical region** — head+torso (``core``), each arm, each leg. Rings are
framed with a rotation-minimizing (double-reflection) frame so the edge loops
never twist, and each joint gets a mitered ring so bends pose without creasing.
Limbs plug into the torso by burying their root ring inside the core tube (the
same overlap contract the shape export already uses), so the union prints as one
connected figure. Every region binds to the armature and poses through the
existing :func:`bodymesh.pose_body_mesh` (linear-blend skinning) — no new
skinning math.

The math (per the design spec):

* miter-bisector tangents ``T`` so an interior ring lies in the bend-bisecting
  plane; a double-reflection RMF transports the reference axis ``U`` with no
  accumulated twist; ``V = cross(T, U)`` makes ``(U, V, T)`` right-handed.
* rings run CCW about ``+T`` and are stitched ``(A,B,C),(A,C,D)`` — every
  undirected edge is shared by exactly two faces, so the tube is watertight and
  2-manifold *by construction* (hence pose-invariant, since posing only moves
  vertices), and the fixed winding is outward (a final volume-sign flip guards).
"""

from __future__ import annotations

import re

import numpy as np

from .mesh import Mesh

FORWARD = (0.0, -1.0, 0.0)  # characters face -Y
_R_MIN = 1e-3  # ring-radius floor: a zero-radius ring collapses to degenerate faces


def _norm(v):
    n = float(np.linalg.norm(v))
    return np.asarray(v, dtype=np.float64) / n if n > 1e-12 else np.asarray(v, dtype=np.float64)


def _is_accessory(name: str) -> bool:
    """Gear/detail test on word boundaries, so 'ear' does not swallow
    'forearm' (a substring test would strip the forearm from body mining)."""
    from .bodymesh import ACCESSORY_KEYWORDS
    toks = set(re.split(r"[^a-z0-9]+", name.lower()))
    return any((k in toks) or ("_" in k and k in name.lower())
               for k in ACCESSORY_KEYWORDS)


# --------------------------------------------------------------------------
# region chains: the armature's joints grouped into linear part polylines


def region_chains(armature) -> dict[str, list[str]]:
    """Region label -> ordered joint chain (root -> tip). Labels match
    :func:`bodymesh._region_of`: ``core`` (spine, head appended separately),
    ``shoulder_{s}`` (an arm), ``shoulder2_{s}`` (a lower arm), ``hip_{s}`` (a
    leg). A leg stops at the ankle — the foot stays authored boxes."""
    J = armature.joints
    chains: dict[str, list[str]] = {}

    core = [n for n in ("pelvis", "spine", "chest", "neck") if n in J]
    if len(core) >= 2:
        chains["core"] = core

    for s in ("l", "r"):
        arm = [f"{k}_{s}" for k in ("shoulder", "elbow", "wrist", "hand") if f"{k}_{s}" in J]
        if len(arm) >= 2:
            chains[f"shoulder_{s}"] = arm
        arm2 = [f"{k}2_{s}" for k in ("shoulder", "elbow", "wrist", "hand") if f"{k}2_{s}" in J]
        if len(arm2) >= 2:
            chains[f"shoulder2_{s}"] = arm2
        leg = [f"{k}_{s}" for k in ("hip", "knee", "ankle") if f"{k}_{s}" in J]
        if len(leg) >= 2:
            chains[f"hip_{s}"] = leg
    return chains


# --------------------------------------------------------------------------
# per-joint radius profile mined from the authored shapes


def _cross_radius(shape) -> float:
    """A limb ring radius from a shape: half the mean of its two smallest world
    scales (the length axis is the largest, so the other two are the girth)."""
    sc = np.sort(np.abs(np.asarray(shape.scale, dtype=np.float64)))
    return 0.5 * float(sc[:2].mean())


# the tube ring is the muscle silhouette itself, so trim the mined radii a
# touch (they come from the fattest bound shape) and deepen the torso so it
# isn't a slab from the side
_LIMB_SCALE = 0.8
_CORE_WIDTH = 0.86
_CORE_DEPTH = 1.18


def radius_profile_from_scene(scene) -> dict[str, dict]:
    """Mine a per-joint ``{joint: {"rx","ry","roll"}}`` profile from the shapes
    bound near each joint. ``core`` joints are elliptical (width vs depth); limb
    joints are circular. Missing joints inherit from their chain neighbours."""
    arm = scene.armature
    # gear/detail must not size the body — a shield bound to wrist_l would
    # balloon that whole arm — so mine radii from body shapes only
    body = [s for s in scene.shapes if not _is_accessory(s.name)]
    by_bone: dict[str | None, list] = {}
    for s in body:
        by_bone.setdefault(s.bone, []).append(s)
    by_name = {s.name: s for s in body}

    profile: dict[str, dict] = {}
    for region, chain in region_chains(arm).items():
        core = region == "core"
        for idx, j in enumerate(chain):
            out_bone = chain[idx + 1] if idx + 1 < len(chain) else None
            cands = list(by_bone.get(out_bone, [])) + list(by_bone.get(j, []))
            if j in by_name:  # a shape named after the joint (e.g. "neck")
                cands.append(by_name[j])
            if not cands:
                continue
            if core:
                xs = max(abs(s.scale[0]) for s in cands)  # X = width  -> ry (side)
                ys = max(abs(s.scale[1]) for s in cands)  # Y = depth  -> rx (front)
                profile[j] = {"rx": 0.5 * ys * _CORE_DEPTH,
                              "ry": 0.5 * xs * _CORE_WIDTH, "roll": 0.0}
            else:
                r = _LIMB_SCALE * max(_cross_radius(s) for s in cands)
                profile[j] = {"rx": r, "ry": r, "roll": 0.0}
        # fill any gaps from the nearest neighbour that resolved
        present = [j for j in chain if j in profile]
        if present:
            for idx, j in enumerate(chain):
                if j in profile:
                    continue
                near = min(present, key=lambda p: abs(chain.index(p) - idx))
                profile[j] = dict(profile[near])
    return profile


# --------------------------------------------------------------------------
# frames along a polyline (miter tangents + double-reflection RMF)


def chain_frames(P, forward=FORWARD):
    """Return per-ring ``(T, U, V, s)``: mitered tangents, a twist-free
    reference axis, its perpendicular, and the miter radius-compensation scale."""
    P = np.asarray(P, dtype=np.float64)
    R = len(P)
    d = np.zeros((R - 1, 3))
    prev = np.array([0.0, 0.0, 1.0])
    for i in range(R - 1):
        v = P[i + 1] - P[i]
        L = float(np.linalg.norm(v))
        d[i] = v / L if L > 1e-9 else prev
        prev = d[i]

    T = np.zeros((R, 3))
    T[0], T[-1] = d[0], d[-1]
    for i in range(1, R - 1):
        b = d[i - 1] + d[i]
        T[i] = _norm(b) if np.linalg.norm(b) > 1e-8 else d[i - 1]

    U = np.zeros((R, 3))
    u0 = None
    for hint in (np.asarray(forward, dtype=np.float64), np.array([0.0, 0.0, 1.0]),
                 np.array([1.0, 0.0, 0.0])):
        cand = hint - float(hint @ T[0]) * T[0]
        if np.linalg.norm(cand) > 1e-6:
            u0 = cand
            break
    U[0] = _norm(u0 if u0 is not None else np.array([1.0, 0.0, 0.0]))
    for i in range(R - 1):
        v1 = P[i + 1] - P[i]
        c1 = float(v1 @ v1)
        if c1 > 1e-12:
            rL = U[i] - (2.0 / c1) * (v1 @ U[i]) * v1
            tL = T[i] - (2.0 / c1) * (v1 @ T[i]) * v1
        else:
            rL, tL = U[i], T[i]
        v2 = T[i + 1] - tL
        c2 = float(v2 @ v2)
        rN = rL - (2.0 / c2) * (v2 @ rL) * v2 if c2 > 1e-12 else rL
        U[i + 1] = _norm(rN)
    V = np.cross(T, U)

    s = np.ones(R)
    for i in range(1, R - 1):
        ch = float(np.clip(T[i] @ d[i], 1e-3, 1.0))
        s[i] = min(1.0 / ch, 2.0)
    return T, U, V, s


# --------------------------------------------------------------------------
# build one tube from a ring list


def _build_tube(centers, rx, ry, roll, ring_weights, N, forward=FORWARD):
    """Sweep + stitch + cap a ring list into a watertight tube. ``ring_weights``
    is one ``{bone: weight}`` dict per ring. Returns ``(mesh, bone_names, W)``."""
    P = np.asarray(centers, dtype=np.float64)
    R = len(P)
    # floor the radii: a zero-radius ring collapses its N verts to a point and
    # emits zero-area (degenerate) faces, breaking watertightness
    rx = np.maximum(np.asarray(rx, dtype=np.float64), _R_MIN)
    ry = np.maximum(np.asarray(ry, dtype=np.float64), _R_MIN)
    roll = np.asarray(roll, dtype=np.float64)
    T, U, V, s = chain_frames(P, forward)
    th = 2.0 * np.pi * np.arange(N) / N

    verts = []
    for i in range(R):
        a = roll[i] + th
        ct, st = np.cos(a), np.sin(a)
        ring = (P[i] + (rx[i] * s[i]) * np.outer(ct, U[i])
                + (ry[i] * s[i]) * np.outer(st, V[i]))
        verts.append(ring)
    verts = list(np.vstack(verts))

    faces: list[tuple[int, int, int]] = []
    for i in range(R - 1):
        bi, bn = i * N, (i + 1) * N
        for k in range(N):
            kn = (k + 1) % N
            A, B, C, D = bi + k, bi + kn, bn + kn, bn + k
            faces.append((A, B, C))
            faces.append((A, C, D))
    p_root = len(verts)
    verts.append(P[0].copy())
    for k in range(N):
        faces.append((k, p_root, (k + 1) % N))  # -T0 outward
    p_tip = len(verts)
    verts.append(P[-1].copy())
    b = (R - 1) * N
    for k in range(N):
        faces.append((b + k, b + (k + 1) % N, p_tip))  # +T outward

    per_vertex = []
    for i in range(R):
        per_vertex.extend([ring_weights[i]] * N)
    per_vertex.append(ring_weights[0])
    per_vertex.append(ring_weights[-1])

    bone_names = sorted({bk for w in ring_weights for bk in w})
    col = {bk: i for i, bk in enumerate(bone_names)}
    W = np.zeros((len(verts), len(bone_names)))
    for vi, w in enumerate(per_vertex):
        for bk, val in w.items():
            W[vi, col[bk]] += val
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    W = W / rs

    mesh = Mesh(np.asarray(verts), np.asarray(faces, dtype=np.int64))
    if mesh.volume() < 0.0:  # guaranteed-consistent winding; normalize to outward
        mesh = Mesh(mesh.vertices, mesh.faces[:, [0, 2, 1]])
    return mesh, bone_names, W


# --------------------------------------------------------------------------
# assemble a region's ring list


def build_head(scene, profile, N=8, latitudes=5):
    """The head as its own ovoid tube — a neck ring that buries into the core,
    then latitude rings up to a crown pole. Its own region so it keeps the
    head/skin color instead of the torso's. Returns ``(mesh, bones, W)`` or
    None. All rings bind one-hot to ``head_top``."""
    arm = scene.armature
    if "head_top" not in arm.joints or "neck" not in arm.joints:
        return None
    neck = np.asarray(arm.joints["neck"].position, dtype=np.float64)
    up = _norm(arm.joints["head_top"].position - neck)
    head = next((s for s in scene.shapes if s.name == "head"), None)
    if head is None:
        return None
    hc = np.asarray(head.position, dtype=np.float64)
    sc = np.abs(np.asarray(head.scale, dtype=np.float64))
    ru, rv, a = 0.5 * sc[0], 0.5 * sc[1], 0.5 * sc[2]
    nk = profile.get("neck", {"rx": 0.4 * ru, "ry": 0.4 * rv})

    centers = [neck.copy()]  # buried root ring, overlaps the core's neck
    rx = [float(nk["rx"])]
    ry = [float(nk["ry"])]
    roll = [0.0]
    weights = [{"head_top": 1.0}]
    for l in range(latitudes):
        phi = np.pi * (l + 1) / (latitudes + 1)  # thin -> fat -> thin ovoid
        centers.append(hc - a * np.cos(phi) * up)
        rx.append(ru * np.sin(phi))
        ry.append(rv * np.sin(phi))
        roll.append(0.0)
        weights.append({"head_top": 1.0})
    return _build_tube(centers, rx, ry, roll, weights, N)


def tube_region(region, chain, scene, profile, N=8, sub=1, overlap=0.7,
                forward=FORWARD):
    """Build one region tube. Returns ``(mesh, bone_names, weights)``."""
    J = scene.armature.joints
    jp = {n: np.asarray(J[n].position, dtype=np.float64) for n in chain}

    def rad(j):
        p = profile.get(j, {"rx": 0.5, "ry": 0.5, "roll": 0.0})
        return float(p["rx"]), float(p["ry"]), float(p.get("roll", 0.0))

    centers, rx, ry, roll, weights = [], [], [], [], []

    if region != "core":  # limb: bury a root ring inside the torso
        d0 = _norm(jp[chain[1]] - jp[chain[0]])
        rxr, ryr, rlr = rad(chain[0])
        centers.append(jp[chain[0]] - overlap * max(rxr, ryr) * d0)
        rx.append(1.05 * rxr)
        ry.append(1.05 * ryr)
        roll.append(rlr)
        weights.append({chain[0]: 1.0})  # planted in the torso (root bone)

    for idx, j in enumerate(chain):
        if idx == 0:
            wk = {chain[1]: 1.0}  # region root moves with its first outgoing bone
        elif idx == len(chain) - 1:
            wk = {j: 1.0}  # tip: incoming bone (key = this joint)
        else:
            wk = {j: 0.5, chain[idx + 1]: 0.5}  # joint: crease-free 2-bone blend
        rxj, ryj, rlj = rad(j)
        centers.append(jp[j])
        rx.append(rxj)
        ry.append(ryj)
        roll.append(rlj)
        weights.append(wk)
        if idx + 1 < len(chain) and sub > 0:  # rigid mid-segment rings hold girth
            a, bpt = jp[j], jp[chain[idx + 1]]
            child = chain[idx + 1]
            rxa, rya, _ = rad(j)
            rxb, ryb, _ = rad(child)
            for t in np.linspace(0.0, 1.0, sub + 2)[1:-1]:
                centers.append(a + (bpt - a) * t)
                rx.append(rxa + (rxb - rxa) * t)
                ry.append(rya + (ryb - rya) * t)
                roll.append(0.0)
                weights.append({child: 1.0})

    return _build_tube(centers, rx, ry, roll, weights, N, forward)


# --------------------------------------------------------------------------
# scene -> per-region tube bodies (drop-in for bodymesh.body_regions)

# face/foot shapes the tube doesn't replace but that still layer on top
_FACE_KEEP = ("eye", "nose", "jaw", "mouth", "brow", "hair", "tusk", "ear", "beard")
_FOOT_KEEP = ("foot", "heel", "toe")


def _retained_shells(scene, region, skins):
    """Detail shapes the tube doesn't replace but that still layer on the
    region: face features on the head, boot boxes on a leg."""
    from .bodymesh import _region_of
    out = []
    for s in scene.shapes:
        if s.bone is None or _region_of(s.bone, scene.armature) != region:
            continue
        if region == "head":
            keep = any(k in s.name for k in _FACE_KEEP)
        elif region.startswith("hip"):
            keep = any(k in s.name for k in _FOOT_KEEP)
        else:
            keep = False
        if not keep:
            continue
        m = s.build_mesh()
        if s.bone in skins:
            m = m.transform(skins[s.bone])
        out.append(m)
    return out


def _region_color(scene, region):
    """The region's own color: the most common color among its *body* shapes
    (gear/detail excluded, so face features can't outvote the torso and a
    shield can't recolor an arm)."""
    from .bodymesh import _region_of
    counts: dict[str, int] = {}
    for s in scene.shapes:
        if _is_accessory(s.name) or _region_of(s.bone, scene.armature) != region:
            continue
        counts[s.color] = counts.get(s.color, 0) + 1
    return max(counts, key=counts.get) if counts else "#b08d57"


def tube_body_regions(scene, profile=None, pose_name=None, N=8,
                      subdivisions_per_segment=1, overlap=0.7, forward=FORWARD):
    """Per-region skeletal tube bodies: ``[(region, mesh, color)]``, mirroring
    :func:`bodymesh.body_regions`. Posed if ``pose_name`` is given. Falls back
    to the SDF ``body_regions`` for a scene with no recognized humanoid rig."""
    from . import bodymesh
    arm = scene.armature
    chains = region_chains(arm)
    if not chains:  # non-humanoid / renamed rig: no bone chains to sweep
        return bodymesh.body_regions(scene, pose_name=pose_name)
    if profile is None:
        profile = radius_profile_from_scene(scene)
    pose = None if pose_name in (None, "rest") else scene.resolve_pose(pose_name)
    skins = arm.skin_matrices(pose) if pose else {}

    builds = list(chains.items())
    head = build_head(scene, profile, N)
    if head is not None:
        builds.append(("head", head))

    out = []
    for region, item in builds:
        if region == "head":
            mesh, bones, W = item
        else:
            mesh, bones, W = tube_region(region, item, scene, profile, N,
                                         subdivisions_per_segment, overlap, forward)
        if pose:
            mesh = bodymesh.pose_body_mesh(mesh, bones, W, arm, pose)
        extra = _retained_shells(scene, region, skins)
        if extra:
            mesh = Mesh.merge([mesh, *extra])
        out.append((region, mesh, _region_color(scene, region)))
    return out


def tube_body_from_scene(scene, profile=None, pose_name=None, N=8,
                         subdivisions_per_segment=1, overlap=0.7, forward=FORWARD):
    """One mesh: the union of the per-region tubes (they overlap at joints)."""
    parts = tube_body_regions(scene, profile, pose_name, N,
                              subdivisions_per_segment, overlap, forward)
    return Mesh.merge([m for _, m, _ in parts])
