"""The wing rig seen from outside: a host armature that drives the wing, input validation,
the weights export, and the behaviours that the constants in ``wingrig`` / ``wings`` exist for.

``test_wingrig.py`` pins the rig's own mechanics. These tests pin (1) the host path --
``PlacedWing.attach / body_pose / skin_weights`` -- against the finished mesh, bone by
bone; (2) that a misspelt control, a misspelt joint or a NaN is an error and not a silent
rest pose or a NaN mesh; (3) behaviours (not constants) that a round of mutation testing
showed nothing was checking: the anatomical envelope of the controls, the stowed-wing lift
and V, the fan ranges, the cupped-sheet fade, the finger columns, the flap's phase structure.
"""

import numpy as np
import pytest

from minimaster.character import appendages as A
from minimaster.character import mhbase as mh
from minimaster.character import wingrig as R
from minimaster.character import wings as W
from minimaster.character.wingrig import (PRESETS, PlacedWing, PoseControls, SkinWeights,
                                          euler_from_matrix, pose_matrices)
from minimaster.core import math3d as m3
from minimaster.core.armature import Armature

STYLES = ("membrane", "dragon", "feathered")
MEMBRANE_STYLES = ("membrane", "dragon")
needs_body = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")


@pytest.fixture(scope="module", params=STYLES)
def rig(request):
    return W.make_wing_rig(request.param)


@pytest.fixture(scope="module")
def membrane():
    return W.make_wing_rig("membrane")


@pytest.fixture(scope="module")
def feathered():
    return W.make_wing_rig("feathered")


# ==========================================================================
# the host-armature path

CONTROL_SETS = [
    {},
    {"fold": 0.3},
    {"raise_": 40.0, "twist": 20.0, "swing": 25.0},
    {"fold": 0.6, "raise_": 30.0, "curl": 20.0},
    {"fold": 1.0},
]


def placement(seed, mirror, scale=1.0):
    """A placement with a real rotation (every axis), an optional mirror and a scale."""
    rng = np.random.default_rng(seed)
    Q = m3.euler_rotation(*rng.uniform(-180.0, 180.0, 3))
    if mirror:
        Q = Q @ np.diag([-1.0, 1.0, 1.0])
    M = np.eye(4)
    M[:3, :3] = Q * scale
    M[:3, 3] = rng.uniform(-3.0, 3.0, 3)
    return M


def body_armature():
    host = Armature()
    host.add_joint("pelvis", (0.0, 0.0, 10.0))
    host.add_joint("chest", (0.0, 1.0, 14.0), "pelvis")
    return host


def host_skin(host, weights: SkinWeights, rest_vertices, pose):
    """What a host engine does with the export: blend the host armature's skin matrices by
    the exported weights, on the placed rest vertices. (Independent of ``pose_mesh``.)"""
    skins = host.skin_matrices(pose)
    Rm = np.stack([skins[b][:3, :3] for b in weights.bones])
    t = np.stack([skins[b][:3, 3] for b in weights.bones])
    out = np.zeros_like(rest_vertices)
    for k in range(weights.weights.shape[1]):
        i, w = weights.indices[:, k], weights.weights[:, k]
        out += w[:, None] * (np.einsum("vij,vj->vi", Rm[i], rest_vertices) + t[i])
    return out


def mask_of(sw: SkinWeights, names):
    m = np.zeros(len(sw.indices), dtype=bool)
    for n in names:
        a, b = sw.pieces[n]
        m[a:b] = True
    return m


def test_a_mirrored_pair_attaches_to_one_body_armature_under_side_prefixes(membrane):
    host = body_armature()
    left = PlacedWing(membrane, placement(1, False))
    right = PlacedWing(membrane, placement(2, True))
    assert (left.default_prefix, right.default_prefix) == ("wing_l", "wing_r")
    n0 = len(host)
    names_l = left.attach(host)                  # no prefix: the base build made a second call raise
    names_r = right.attach(host)
    assert names_l == left.joint_names() and names_r == right.joint_names()
    assert names_l[0] == "wing_l:root" and names_r[0] == "wing_r:root"
    assert len(host) == n0 + 2 * len(membrane.armature.joints)
    assert host.joints["wing_l:root"].parent == "chest" == host.joints["wing_r:root"].parent
    # the same side twice is an error, and it adds nothing (no half-attached wing)
    with pytest.raises(ValueError, match="already exists"):
        left.attach(host)
    assert len(host) == n0 + 2 * len(membrane.armature.joints)
    # ... unless it is given a name of its own; and the parent must exist
    assert left.attach(host, prefix="spare")[0] == "spare:root"
    with pytest.raises(ValueError, match="parent joint"):
        left.attach(host, parent="no_such_joint", prefix="x")
    assert not any(n.startswith("x:") for n in host.joints)


def test_a_wing_frame_pose_applied_to_the_attached_joints_as_it_is_would_be_wrong(membrane):
    """Why body_pose exists: the attached joints sit in body space, the pose is in the wing's
    frame. (Pinned so the helper is never mistaken for a convenience.)"""
    pw = PlacedWing(membrane, placement(3, False))
    host = body_armature()
    pw.attach(host, prefix="w")
    pose = membrane.pose_from_controls(fold=0.6, raise_=30.0)
    wrong = host.skin_matrices({f"w:{k}": v for k, v in pose.items()})
    ref = pw.skin_matrices_body(pose)
    assert max(np.abs(wrong[f"w:{k}"] - ref[k]).max() for k in ref) > 1.0
    right = host.skin_matrices(pw.body_pose(pose, prefix="w"))
    assert max(np.abs(right[f"w:{k}"] - ref[k]).max() for k in ref) < 1e-9


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("scale", [1.0, 0.37])
def test_every_joint_of_the_host_armature_follows_the_wing_frame_pose(rig, mirror, scale):
    """Each joint of the wing turned on its own, then all at once: the host's skin matrices for
    EVERY bone and the posed position of EVERY joint equal the wing's own (placed) ones."""
    pw = PlacedWing(rig, placement(11 + mirror, mirror, scale))
    host = body_armature()
    pw.attach(host, prefix="w")
    rng = np.random.default_rng(5)
    poses = [{j: tuple(rng.uniform(-60.0, 60.0, 3))} for j in rig.armature.joints]
    poses.append({j: tuple(rng.uniform(-40.0, 40.0, 3)) for j in rig.armature.joints})
    poses.append(rig.pose_from_controls(fold=0.7, raise_=20.0, curl=30.0))
    sw, rest = pw.skin_weights("w"), pw.mesh().vertices
    exact = ~mask_of(sw, sw.post_processed)
    for n, pose in enumerate(poses):
        hp = pw.body_pose(pose, prefix="w")
        skins, ref = host.skin_matrices(hp), pw.skin_matrices_body(pose)
        assert set(ref) == set(rig.bone_keys)
        for k in ref:
            assert np.abs(skins[f"w:{k}"] - ref[k]).max() < 1e-9, (k, pose)
        pos, rpos = host.posed_positions(hp), pw.joint_positions(pose)
        for k in rpos:
            assert np.abs(pos[f"w:{k}"] - rpos[k]).max() < 1e-9, (k, pose)
        if scale == 1.0:
            # ... and the mesh every piece but the pleated slab: whatever one joint moves, the
            # host's blend of the exported weights lands where pose() puts the vertices
            assert np.abs(host_skin(host, sw, rest, hp) - pw.mesh(pose).vertices)[exact].max() < 1e-9, \
                (n, list(pose)[:1])


def test_the_matrix_form_of_body_pose_drives_a_host_with_its_own_kinematics(rig):
    """form="matrix" is what a host with its own (non-Euler) joints needs; check it with a
    forward-kinematics written here, and that the matrices are proper rotations even for a
    mirrored placement."""
    pw = PlacedWing(rig, placement(7, True, 1.3))
    pose = rig.pose_from_controls(fold=0.8, raise_=25.0, twist=10.0, curl=30.0)
    mats = pw.body_pose(pose, prefix="w", form="matrix")
    assert set(mats) == {f"w:{k}" for k in pose}
    for Rb in mats.values():
        assert np.allclose(Rb @ Rb.T, np.eye(3), atol=1e-12) and np.linalg.det(Rb) == pytest.approx(1.0)
    joints = {n: pw.matrix[:3, :3] @ p + pw.matrix[:3, 3] for n, p in rig.skeleton.joints.items()}
    world = {}
    for name in rig.armature.joints:                      # parents come first
        par = rig.skeleton.parent[name]
        rot = np.eye(4)
        rot[:3, :3] = mats.get(f"w:{name}", np.eye(3))
        if par is None:
            world[name] = m3.translation_mat(joints[name]) @ rot
        else:
            world[name] = world[par] @ m3.translation_mat(joints[name] - joints[par]) @ rot
    ref = pw.skin_matrices_body(pose)
    for name in rig.bone_keys:
        par = rig.skeleton.parent[name]
        mine = world[par] @ m3.translation_mat(-joints[par])
        assert np.abs(mine - ref[name]).max() < 1e-9, name
    with pytest.raises(ValueError, match="form"):
        pw.body_pose(pose, form="quaternion")


def test_body_pose_takes_controls_presets_and_raw_poses_alike(membrane):
    pw = PlacedWing(membrane, placement(4, False))
    raw = membrane.pose_from_controls(fold=0.5, raise_=20.0)
    a = pw.body_pose(raw, prefix="w")
    assert a.keys() == {f"w:{k}" for k in raw}
    assert pw.body_pose(PoseControls(fold=0.5, raise_=20.0), prefix="w") == a
    assert pw.body_pose(prefix="w", fold=0.5, raise_=20.0) == a
    assert pw.body_pose("folded", prefix="w") == pw.body_pose(membrane.pose_from_controls("folded"), prefix="w")
    assert all(k.startswith("wing_l:") for k in pw.body_pose(fold=0.2))        # the default prefix
    with pytest.raises(TypeError, match="raw pose"):
        pw.body_pose(raw, fold=0.5)


def test_a_placement_a_host_cannot_reproduce_is_refused(membrane):
    """A non-uniform scale makes the conjugate of a rotation something that is not a rotation:
    a host armature cannot be told that pose, so body_pose says so instead of returning
    angles that quietly do not match."""
    M = placement(8, False)
    M[:3, :3] = M[:3, :3] @ np.diag([1.0, 2.0, 1.0])
    pw = PlacedWing(membrane, M)
    with pytest.raises(ValueError, match="uniform"):
        pw.body_pose(fold=0.3)
    assert pw.attach(body_armature(), prefix="w")          # attaching (rest) is still fine


@pytest.mark.parametrize("style", STYLES)
def test_a_host_armature_skins_the_wing_to_the_finished_mesh_except_where_the_membrane_is_pleated(style):
    """The exported weights, the host's own skin matrices and a plain linear blend reproduce
    pose() EXACTLY for every piece but the membrane slab, to 1e-9 -- bones, feathers, the
    thumb, knobs and the propatagium included; the slab is within a measured bound (below)."""
    rig = W.make_wing_rig(style)
    host = body_armature()
    left = PlacedWing(rig, placement(21, False, 1.0))
    right = PlacedWing(rig, placement(22, True, 1.0))
    left.attach(host)
    right.attach(host)
    for pw in (left, right):
        sw = pw.skin_weights()
        assert sw.bones == tuple(f"{pw.default_prefix}:{k}" for k in rig.bone_keys)
        rest = pw.mesh().vertices
        exact = ~mask_of(sw, sw.post_processed)
        assert exact.sum() > 0.2 * len(exact) and exact.sum() == sum(
            b - a for n, (a, b) in sw.pieces.items() if n not in sw.post_processed)
        for ctrl in CONTROL_SETS + [R.PoseControls.flap(90.0), R.PoseControls.flap(300.0)]:
            ctrl = ctrl if isinstance(ctrl, PoseControls) else PoseControls(**ctrl)
            got = host_skin(host, sw, rest, pw.body_pose(ctrl))
            want = pw.pose(ctrl).vertices
            assert np.abs(got - want)[exact].max() < 1e-9, (style, ctrl)
            assert np.isfinite(got).all()
            if (~exact).any():                           # the slab, measured through the host path itself
                assert np.abs(got - want)[~exact].max() < POST_PROCESS_MAX_LIMIT, (style, ctrl)
    if style == "feathered":
        assert sw.post_processed == ()                   # a feathered wing is plain skinning all through
    else:
        assert sw.post_processed == ("membrane",)


# How far the membrane slab can be from the finished sheet when a host does the skinning (the
# pleat / gather / stretch-limiter post-process is not run): measured in wing units (the span
# is 13.1) over the presets, twelve flap phases, a fold sweep and curls. docs/wings.md quotes
# the figures; the bounds below keep ~30% headroom.
NEAR_SPREAD = [{}, {"fold": 0.1}, {"curl": 20.0}, "mid_downstroke", "downstroke"]
POST_PROCESS_MEAN_LIMIT = 1.1
POST_PROCESS_MAX_LIMIT = 4.0


def _raw_lbs(rig, pose):
    sw = rig.skin_weights()
    sk = rig.skin_matrices(pose)
    Rm = np.stack([sk[b][:3, :3] for b in sw.bones])
    t = np.stack([sk[b][:3, 3] for b in sw.bones])
    V = rig.rest_mesh.vertices
    out = np.zeros_like(V)
    for k in range(sw.weights.shape[1]):
        i = sw.indices[:, k]
        out += sw.weights[:, k, None] * (np.einsum("vij,vj->vi", Rm[i], V) + t[i])
    return out


@pytest.mark.parametrize("style", MEMBRANE_STYLES)
def test_the_membrane_through_a_host_is_within_a_measured_distance_of_the_finished_sheet(style):
    rig = W.make_wing_rig(style)
    a, b = rig.skin_weights().pieces["membrane"]
    span = float(np.ptp(rig.rest_mesh.vertices[:, 0]))

    def deviation(controls):
        pose = rig.pose_from_controls(controls)
        d = np.linalg.norm(_raw_lbs(rig, pose)[a:b] - rig.pose_mesh(pose).vertices[a:b], axis=1)
        return float(d.max()), float(d.mean())

    assert max(deviation(PoseControls())) < 1e-9                    # the rest pose is exact
    worst_max = worst_mean = 0.0
    for ctrl in (list(PRESETS) + [PoseControls.flap(p) for p in range(0, 360, 30)]
                 + [PoseControls(fold=f) for f in (0.2, 0.4, 0.6, 0.8, 0.9, 1.0)]
                 + [PoseControls(curl=c) for c in (40.0, 100.0)]):
        mx, mean = deviation(ctrl)
        worst_max, worst_mean = max(worst_max, mx), max(worst_mean, mean)
        assert mx < POST_PROCESS_MAX_LIMIT and mean < POST_PROCESS_MEAN_LIMIT, ctrl
    assert worst_max < 0.35 * span
    # near the spread pose a host is practically exact (the pleats only begin to matter
    # once a panel has closed)
    for ctrl in NEAR_SPREAD:
        mx, mean = deviation(PoseControls(**ctrl) if isinstance(ctrl, dict) else ctrl)
        # (0.05 before the interior became the chord at every pose, which differs from raw skinning
        # by a few hundredths even for a gentle curl)
        assert mx < 0.75 and mean < 0.08, ctrl


@needs_body
@pytest.mark.parametrize("style", STYLES)
def test_a_real_pair_on_a_body_armature_is_driven_through_the_chest(style):
    """The whole path on the actual MakeHuman placement: attach both wings, pose them with
    body_pose, rotate the chest, skin with the exported weights; every bone matches the
    wing's own body-space skin matrices (composed with the chest) and every exact piece
    matches the finished mesh carried by the same chest rotation."""
    base, lib = mh.load()
    v = lib.apply(base.verts, mh.macro_weights(gender=1.0))
    rigs = A.back_wings(style).build_rigs(base, v)
    assert [n for n, _ in rigs] == ["wing_l", "wing_r"]
    host = body_armature()
    for name, pw in rigs:
        assert pw.default_prefix == name
        pw.attach(host)
    chest = {"chest": (12.0, -20.0, 25.0)}
    for ctrl in ({}, {"fold": 0.6, "raise_": 30.0, "curl": 20.0}, {"fold": 1.0}):
        pose = {}
        for _, pw in rigs:
            pose.update(pw.body_pose(**ctrl))
        skins = host.skin_matrices({**pose, **chest})
        # what the chest carries everything attached to it by: a turn about the chest joint
        q = host.joints["chest"].position
        C = m3.translation_mat(q) @ m3.mat4(rotation=m3.euler_rotation(*chest["chest"])) @ m3.translation_mat(-q)
        for name, pw in rigs:
            ref = pw.skin_matrices_body(pw.rig.pose_from_controls(**ctrl))
            for k in ref:
                assert np.abs(skins[f"{name}:{k}"] - C @ ref[k]).max() < 1e-8, (name, k, ctrl)
            sw = pw.skin_weights()
            got = host_skin(host, sw, pw.mesh().vertices, {**pose, **chest})
            moved = pw.pose(**ctrl).vertices @ C[:3, :3].T + C[:3, 3]
            exact = ~mask_of(sw, sw.post_processed)
            assert np.abs(got - moved)[exact].max() < 1e-8, (name, ctrl)


# ==========================================================================
# the weights export

def test_the_skin_weights_export_is_the_rigs_weights_by_bone_name(rig):
    sw = rig.skin_weights()
    idx, wt = rig.weights()
    assert isinstance(sw, SkinWeights)
    assert sw.bones == tuple(rig.bone_keys) and set(sw.bones) <= set(rig.armature.joints)
    assert np.array_equal(sw.indices, idx) and np.array_equal(sw.weights, wt)
    assert sw.pieces == rig.ranges
    ends = sorted(sw.pieces.values())
    assert ends[0][0] == 0 and ends[-1][1] == len(rig.rest_mesh.vertices)
    assert all(a[1] == b[0] for a, b in zip(ends, ends[1:]))
    # rebuilt from the vertex groups it is the same dense matrix
    dense = np.zeros((len(wt), len(sw.bones)))
    for name, (v, w) in sw.vertex_groups().items():
        assert len(np.unique(v)) == len(v) and (w > 0).all()
        dense[v, sw.bones.index(name)] = w
    ref = np.zeros_like(dense)
    for k in range(wt.shape[1]):
        np.add.at(ref, (np.arange(len(wt)), idx[:, k]), wt[:, k])
    assert np.abs(dense - ref).max() < 1e-15 and np.allclose(dense.sum(axis=1), 1.0)
    # the per-vertex form agrees on a sample, and carries only names of real bones
    pv = sw.per_vertex()
    assert len(pv) == len(wt)
    for i in range(0, len(pv), max(len(pv) // 40, 1)):
        assert pv[i].keys() <= set(sw.bones)
        assert sum(pv[i].values()) == pytest.approx(1.0)
        assert {sw.bones.index(n): w for n, w in pv[i].items()} == pytest.approx(
            {j: dense[i, j] for j in np.nonzero(dense[i])[0]})
    # the export is a copy: changing it cannot change the rig
    sw.weights[:] = 0.0
    assert rig.weights()[1].sum() > 0.0


def test_the_placed_exports_carry_the_attached_joints_names(membrane):
    pw = PlacedWing(membrane, placement(9, True))
    host = body_armature()
    pw.attach(host)
    sw = pw.skin_weights()
    assert sw.bones == tuple(f"wing_r:{k}" for k in membrane.bone_keys)
    assert all(b in host.joints for b in sw.bones)                 # every bone is a host joint
    assert set(pw.joint_names()) == {n for n in host.joints if n.startswith("wing_r:")}
    assert set(pw.skin_weights("zz").vertex_groups()) == {f"zz:{k}" for k in membrane.skin_weights().vertex_groups()}


# ==========================================================================
# Euler triples versus matrices

def test_euler_from_matrix_round_trips_including_next_to_the_poles():
    rng = np.random.default_rng(0)
    cases = [tuple(rng.uniform(-180.0, 180.0, 3)) for _ in range(200)]
    for ry in (90.0, -90.0, 90.0 - 1e-7, -90.0 + 1e-7, 89.999, 90.0 - 1e-12):     # gimbal and near it
        cases += [(rx, ry, rz) for rx, rz in rng.uniform(-180.0, 180.0, (4, 2))]
    for e in cases:
        Rm = m3.euler_rotation(*e)
        back = euler_from_matrix(Rm)
        assert np.abs(m3.euler_rotation(*back) - Rm).max() < 1e-9, e
        assert -90.0 - 1e-9 <= back[1] <= 90.0 + 1e-9


def test_a_euler_triple_jumps_across_the_pole_but_the_matrix_does_not():
    """The documented gimbal flaw: ry is confined to [-90, 90], so a rotation sweeping through
    ry = 90 comes out with rx and rz jumping by 180. Hence: never key or lerp the triples; the
    matrices (pose_matrices, body_pose(form="matrix")) are continuous."""
    lo, hi = euler_from_matrix(m3.euler_rotation(30.0, 89.5, 20.0)), euler_from_matrix(m3.euler_rotation(30.0, 90.5, 20.0))
    assert abs(lo[0] - hi[0]) > 170.0 and abs(lo[2] - hi[2]) > 170.0
    Ra, Rb = m3.euler_rotation(30.0, 89.5, 20.0), m3.euler_rotation(30.0, 90.5, 20.0)
    assert np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1.0) / 2.0, -1, 1))) < 2.5


def test_pose_matrices_are_continuous_in_the_controls(rig):
    """Over a 0.5% fold sweep no joint's rotation matrix turns by more than a few degrees per
    step (a Euler triple can jump by 180 at the pole; see above)."""
    prev = None
    worst = 0.0
    for f in np.linspace(0.0, 1.0, 201):
        mats = pose_matrices(rig.pose_from_controls(fold=float(f)))
        if prev is not None:
            for j in mats.keys() & prev.keys():
                tr = np.trace(prev[j].T @ mats[j])
                worst = max(worst, float(np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1, 1)))))
        prev = mats
    assert worst < 12.0, worst


# ==========================================================================
# input validation

def test_a_misspelt_control_is_an_error_not_a_silent_rest_pose(rig):
    with pytest.raises(TypeError, match=r"unknown pose control 'foled'.*did you mean 'fold'"):
        rig.pose(foled=0.5)
    with pytest.raises(TypeError, match="unknown pose control 'raise'"):
        rig.pose_from_controls(fold=0.2, **{"raise": 40.0})
    with pytest.raises(TypeError, match="unknown pose control 'flod'"):
        PoseControls.flap(30.0, flod=1.0)
    with pytest.raises(TypeError, match="foled"):
        PoseControls(foled=0.5)
    c = PoseControls()
    with pytest.raises(TypeError, match="unknown pose control"):
        c.flod = 1.0
    assert not hasattr(c, "flod")                                   # no stray attribute is left behind
    with pytest.raises(TypeError, match="controls must be"):
        rig.pose({"fold": 0.5})
    with pytest.raises(KeyError, match=r"unknown pose preset 'upstrok'.*did you mean 'upstroke'"):
        rig.pose("upstrok")


@pytest.mark.parametrize("name", ["fold", "raise_", "swing", "twist", "fan", "curl", "wrist_flex", "hand_twist"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_a_non_finite_control_is_an_error_not_a_nan_mesh(rig, name, bad):
    with pytest.raises(ValueError, match="finite"):
        rig.pose(**{name: bad})
    with pytest.raises(ValueError, match="finite"):
        rig.pose_from_controls(PoseControls(), **{name: bad})
    with pytest.raises(ValueError, match="finite"):
        PoseControls(**{name: bad})
    c = PoseControls()
    with pytest.raises(ValueError, match="finite"):
        setattr(c, name, bad)
    assert getattr(c, name) == getattr(PoseControls(), name)                # the bad value did not land


def test_a_control_that_is_not_a_number_is_a_type_error(membrane):
    for bad in ("0.5", None, [0.1, 0.2], b"1"):
        with pytest.raises(TypeError, match="must be a number"):
            membrane.pose(fold=bad)
    with pytest.raises(ValueError, match="finite"):
        PoseControls.flap(float("nan"))


def test_every_valid_way_of_giving_a_control_still_works(membrane):
    base = membrane.pose_from_controls(fold=0.5, raise_=20.0)
    assert membrane.pose_from_controls(PoseControls(fold=np.float64(0.5), raise_=np.int64(20))) == base
    assert membrane.pose_from_controls(PoseControls(fold=0.5), raise_=20) == base
    assert membrane.pose_from_controls(fold=np.float32(0.5), raise_=np.array(20.0)) == base
    for name in PRESETS:
        assert membrane.pose_from_controls(name) == membrane.pose_from_controls(PoseControls(**PRESETS[name]))
    for ph in (0, 90.0, 270, 1234.5):
        membrane.pose_from_controls(PoseControls.flap(ph, amp=30.0, fold=0.2))
    assert membrane.pose_from_controls(PoseControls(fold=9.0)) == membrane.pose_from_controls(fold=1.0)   # clamped, not refused
    c = PoseControls(raise_=200.0, fold=3.0)
    membrane.pose_from_controls(c)
    assert (c.raise_, c.fold) == (200.0, 3.0)                       # and the caller's object is untouched


def test_a_misspelt_joint_or_a_malformed_rotation_in_a_raw_pose_is_an_error(rig):
    good = rig.pose_from_controls(fold=0.4)
    typo = {**good, "wirst": (10.0, 0.0, 0.0)}
    for call in (rig.pose_mesh, rig.skin_matrices, rig.posed_joints):
        with pytest.raises(KeyError, match=r"no joint 'wirst'.*did you mean 'wrist'"):
            call(typo)
    pw = PlacedWing(rig, placement(6, False))
    for call in (pw.mesh, pw.joint_positions, pw.skin_matrices_body, pw.body_pose):
        with pytest.raises(KeyError, match="no joint 'wirst'"):
            call(typo)
    for bad in ((1.0, 2.0), (float("nan"), 0.0, 0.0), "abc", (0, 0, float("inf")), [[1, 2, 3]]):
        with pytest.raises(ValueError, match="three finite euler angles"):
            rig.pose_mesh({"wrist": bad})
    for ok in ({}, None, {"wrist": (1, 2, 3)}, {"wrist": np.array([1.0, 2.0, 3.0])}, good):
        rig.pose_mesh(ok)


# ==========================================================================
# what the constants are for

def _applied_angle(euler) -> float:
    """The angle (degrees) a joint is actually turned by a pose entry, whatever the axes."""
    Rm = m3.euler_rotation(*euler)
    return float(np.degrees(np.arccos(np.clip((np.trace(Rm) - 1.0) / 2.0, -1.0, 1.0))))


def test_an_impossible_control_is_clamped_to_an_anatomical_envelope(rig):
    """Asked for a million degrees, a control gets what a joint can do. The envelope is stated
    here in degrees of what the joint is actually turned by -- not read back from LIMITS, a test
    that takes its expected value from the table it tests passes whatever the table says."""
    sk = rig.skeleton
    knuckle = sk.chains["III"][1]
    for sign in (1.0, -1.0):
        big = 1e6 * sign
        # the shoulder alone carries swing, raise and twist (spread wing: no fold to confound)
        assert _applied_angle(rig.pose_from_controls(twist=big)["shoulder"]) < 60.0          # roll about the span
        assert _applied_angle(rig.pose_from_controls(swing=big)["shoulder"]) < (100.0 if sign > 0 else 60.0)
        assert _applied_angle(rig.pose_from_controls(raise_=big)["shoulder"]) < (110.0 if sign > 0 else 100.0)
        assert _applied_angle(rig.pose_from_controls(hand_twist=big)["wrist"]) < 45.0        # wash-out
        assert _applied_angle(rig.pose_from_controls(wrist_flex=big)["wrist"]) < 90.0
        gain = sk.curl_gain                         # the bird's hand flexes as a unit, a fraction per knuckle
        turned = _applied_angle(rig.pose_from_controls(curl=big)[knuckle]) / gain
        assert turned < (170.0 if sign > 0 else 20.0), turned          # a fist at most; hardly extends past straight
    # A knuckle cannot turn a full circle: the applied angle must never WRAP. A limit past 180 would
    # make a harder curl come out as a gentler one (400 degrees is 40), so over a sweep far beyond
    # any limit the angle only ever grows or holds (whatever the limit is).
    sweep = [_applied_angle(rig.pose_from_controls(curl=c).get(knuckle, (0.0, 0.0, 0.0))) / gain
             for c in (0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 250.0, 300.0, 360.0, 400.0, 500.0, 1e3)]
    assert all(b >= a - 1e-9 for a, b in zip(sweep, sweep[1:])), np.round(sweep, 1)
    # and the clamp saturates: nothing more is asked of the joint however much more is requested
    for name in ("twist", "swing", "raise_", "hand_twist", "wrist_flex", "curl"):
        for sign in (1.0, -1.0):
            assert rig.pose_from_controls(**{name: sign * 1e3}) == rig.pose_from_controls(**{name: sign * 1e6}), name
    # a legal request is not touched by the clamp
    for name, joint in (("twist", "shoulder"), ("swing", "shoulder"), ("raise_", "shoulder"),
                        ("hand_twist", "wrist"), ("wrist_flex", "wrist")):
        assert _applied_angle(rig.pose_from_controls(**{name: 10.0})[joint]) == pytest.approx(10.0, abs=1e-6), name


@pytest.mark.parametrize("style", MEMBRANE_STYLES)
def test_a_stowed_membrane_wing_is_lifted_off_its_plane(style):
    """The plane a wing is mounted in is sunk into the back so a print is welded; stowed along it
    the wing would be buried in the torso, so a fold also lifts the wing (the elbow rises toward
    the upper surface, -Y; it stays in the plane, to 0.07, without the lift)."""
    rig = W.make_wing_rig(style)
    spread = -rig.posed_joints({})["elbow"][1]
    stowed = -rig.posed_joints(rig.pose_from_controls(fold=1.0))["elbow"][1]
    assert stowed - spread > 0.35, (stowed, spread)
    # and the lift is the stow's, it grows with the fold
    heights = [-rig.posed_joints(rig.pose_from_controls(fold=f))["elbow"][1] for f in (0.0, 0.5, 1.0)]
    assert heights[0] < heights[1] < heights[2]


def test_a_membrane_keeps_its_fingers_apart_under_any_fan_request():
    """Fan 0 would close the fingers onto a sheet that cannot follow, fan 1.5 spread them to
    tearing: the range the skin can take is a property of the geometry, so it is checked as
    one (III-V tip separation against the rest pose), not as a number read from the code."""
    for style in MEMBRANE_STYLES:
        rig = W.make_wing_rig(style)
        gap = lambda fan: np.linalg.norm(                                                   # noqa: E731
            (lambda J: J["d3_tip"] - J["d5_tip"])(rig.posed_joints(rig.pose_from_controls(fan=fan))))
        rest = gap(1.0)
        assert gap(0.0) > 0.4 * rest and gap(1.5) < 1.35 * rest, style
        assert gap(0.0) < 0.97 * rest and gap(1.5) > 1.02 * rest, style    # ... yet the control still moves them


def test_the_birds_fan_has_a_range_that_keeps_the_feathers_distinct_and_uncrossed(feathered):
    """Fan 0 stacked ten primaries on one heading (coplanar shells), fan 1.5 sent P1 across
    the secondaries. The request is clamped to a range in which neither happens."""
    def headings(fan):
        J = feathered.posed_joints(feathered.pose_from_controls(fan=fan))
        return {f["root"][:len(f["row"]) + 2]: float(np.degrees(np.arctan2(*(J[f["tip"]] - J[f["root"]])[[2, 0]])))
                for f in feathered.skeleton.feathers if f["row"] in ("primary", "secondary")}

    closed, rest, wide = headings(0.0), headings(1.0), headings(1.5)
    pri = lambda h: [h[f"primary{i:02d}"] for i in range(1, 11)]       # noqa: E731
    sec = lambda h: [h[f"secondary{i:02d}"] for i in range(1, 15)]     # noqa: E731
    assert np.ptp(pri(closed)) > 8.0 and np.ptp(sec(closed)) > 8.0      # still distinct at the floor
    assert np.ptp(pri(closed)) < 0.4 * np.ptp(pri(rest))                 # ... and still a closed fan
    assert wide["primary01"] < wide["secondary01"]                       # P1 never crosses S1
    assert wide["primary01"] > rest["primary01"]                         # the control still opens it
    assert all(a > b for a, b in zip(pri(wide), pri(wide)[1:]))          # primaries stay ordered P1 > ... > P10


@needs_body
def test_the_stowed_pairs_toe_in_closes_the_hem_toward_the_spine():
    """A feathered pair stows in a V that closes toward the hem. Part of that is the wing's own
    shape (the root's lean, the taper); toe-in (the wing's axis turned toward the spine) is what is
    left to tune, so compare with the same wing without it: the hem is closer to the spine, and by
    a visible amount, and the pair still does not cross."""
    base, lib = mh.load()
    v = lib.apply(base.verts, mh.macro_weights(gender=1.0))

    def hem_and_depth(pw, sign):
        w = pw.pose(fold=1.0).vertices
        low = w[w[:, 1] < w[:, 1].min() + 1.0]
        high = w[w[:, 1] > w[:, 1].max() - 1.5]
        return (sign * low[:, 0]).mean(), (sign * high[:, 0]).mean() - (sign * low[:, 0]).mean(), (sign * w[:, 0]).min()

    for name, pw in A.back_wings("feathered").build_rigs(base, v):
        sign = 1.0 if name.endswith("l") else -1.0
        hem, depth, clearance = hem_and_depth(pw, sign)
        shipped = pw.rig.skeleton.stow_toe_in
        pw.rig.skeleton.stow_toe_in = 0.0
        try:
            hem0, depth0, _ = hem_and_depth(pw, sign)
        finally:
            pw.rig.skeleton.stow_toe_in = shipped
        assert hem0 - hem > 0.15, (name, hem0, hem)                      # the toe-in closes the hem
        assert depth > depth0 + 0.3 and depth > 1.2, (name, depth, depth0)   # a V, not parallel planks
        assert clearance > 0.1, (name, clearance)                         # and it never reaches the spine


def _finger_column_distance(rig, pose, mesh):
    """How far each finger's column of the sheet's mid-surface lies from that finger's bones."""
    piece = next(p for p in rig.pieces if p.pleat)
    a, b = rig.ranges[piece.name]
    n1, m1 = piece.pleat["shape"]
    nv = n1 * m1
    pos = mesh.vertices[a:b]
    mid = (0.5 * (pos[:nv] + pos[nv:2 * nv])).reshape(n1, m1, 3)
    joints = rig.posed_joints(pose)
    out = {}
    for (c0, _), name in zip(piece.pleat["sectors"], piece.pleat["fingers"]):
        path = [joints[j] for j in ["wrist"] + list(rig.skeleton.chains[name][1:])]
        d = []
        for p, q in zip(path[:-1], path[1:]):
            ab = q - p
            t = np.clip(((mid[:, c0] - p) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
            d.append(np.linalg.norm(mid[:, c0] - (p + t[:, None] * ab), axis=1))
        out[name] = float(np.min(d, axis=0).max())
    return out


@pytest.mark.parametrize("style", MEMBRANE_STYLES)
def test_the_sheet_stays_on_its_fingers_while_they_fold_up(style):
    """Until a finger is folded right up its column of the sheet lies along the finger, not on the
    straight line between its two ends: at fold 0.9 (the knuckles mid-concertina) it is within a
    bone's width of the bones, and from the finished sheet's side the stowed fan is exact rays."""
    rig = W.make_wing_rig(style)
    for fold, limit in ((0.9, 0.35), (1.0, 0.3)):
        pose = rig.pose_from_controls(fold=fold)
        d = _finger_column_distance(rig, pose, rig.pose_mesh(pose))
        assert max(d.values()) < limit, (fold, d)


@pytest.mark.parametrize("style", MEMBRANE_STYLES)
def test_the_cupped_sheets_relief_is_gone_once_the_hand_is_a_third_of_the_way_back(style, monkeypatch):
    """A cupped wing (curled, flapping) takes a little more stretch and buckles less; the allowance
    fades out by the time the hand has folded 30% of the way back (docs/wings.md). So from fold 0.45
    on, switching the allowance off changes nothing -- while on a cupped wing it does."""
    rig = W.make_wing_rig(style)
    pose = rig.pose_from_controls(fold=0.45)
    assert rig._hand_fold(pose) > 0.3
    cupped = rig.pose_from_controls(curl=40.0, fold=0.1)
    assert rig._hand_fold(cupped) < 0.2
    before, before_cupped = rig.pose_mesh(pose).vertices.copy(), rig.pose_mesh(cupped).vertices.copy()
    monkeypatch.setattr(R, "ELASTIC_CUPPED", 0.0)
    monkeypatch.setattr(R, "PLEAT_CUPPED", 0.0)
    assert np.abs(rig.pose_mesh(pose).vertices - before).max() < 1e-9
    assert np.abs(rig.pose_mesh(cupped).vertices - before_cupped).max() > 0.02


def vars_of(c):
    return [getattr(c, f) for f in ("fold", "raise_", "swing", "twist", "fan", "curl", "wrist_flex", "hand_twist")]


def _pleat_height_over_tent(rig, controls):
    """For every panel, the tallest pleat on any row as a fraction of the tent that row's own slack
    could raise: the sheet's edges closed from their rest gap L to the posed gap g, an inextensible
    skin buckles to at most sqrt(L^2 - g^2) / 2 over the lobes it splits into. The pleat is
    isolated by posing the same skinned sheet with and without the pleat term (``gain=0``)."""
    import inspect
    if "gain" not in inspect.signature(rig._apply_pleats).parameters:
        pytest.skip("WingRig._apply_pleats lost its gain parameter: re-point this test at the pleat term")
    piece = next(p for p in rig.pieces if p.pleat)
    a, b = rig.ranges[piece.name]
    n1, m1 = piece.pleat["shape"]
    nv = n1 * m1
    pose = rig.pose_from_controls(controls)
    raw = _raw_lbs(rig, pose)
    flat = rig._apply_pleats(raw.copy(), pose, gain=0.0)[a:b]
    pleated = rig._apply_pleats(raw.copy(), pose)[a:b]
    mid = lambda v: (0.5 * (v[:nv] + v[nv:2 * nv])).reshape(n1, m1, 3)             # noqa: E731
    rest, m0, m1_ = mid(rig.rest_mesh.vertices[a:b]), mid(flat), mid(pleated)
    lift = np.linalg.norm(m1_ - m0, axis=2)
    out = []
    for k, (c0, c1) in enumerate(piece.pleat["sectors"]):
        L = np.linalg.norm(rest[:, c1] - rest[:, c0], axis=1)
        g = np.linalg.norm(m1_[:, c1] - m1_[:, c0], axis=1)
        tent = 0.5 * np.sqrt(np.maximum(L ** 2 - g ** 2, 0.0)) / piece.pleat["lobes"][k]
        rows = tent > 0.02
        out.append(float((lift[:, c0:c1 + 1].max(axis=1)[rows] / tent[rows]).max()) if rows.any() else 0.0)
    return out


@pytest.mark.parametrize("style", MEMBRANE_STYLES)
def test_a_pleat_is_never_taller_than_its_own_slack_and_the_inboard_panel_pleats_least(style):
    """The sheet is inextensible: a panel that has closed from L to g cannot buckle higher than the
    tent its slack makes. That holds for every panel of every fold (the pleat gain is a damping of
    it, 0.8 at most), and the long inboard panel -- which hangs as a drape -- pleats clearly
    less than the finger panels (docs/wings.md)."""
    rig = W.make_wing_rig(style)
    for fold in (0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0):
        ratios = _pleat_height_over_tent(rig, PoseControls(fold=fold))
        assert max(ratios) < 1.0, (fold, ratios)
        if 0.3 <= fold <= 0.8:
            assert ratios[-1] < 0.8 * max(ratios[:-1]), (fold, ratios)


def _series(fn, phases=range(0, 360)):
    return np.array([fn(float(p)) for p in phases])


def test_a_flap_cycle_has_the_phase_structure_of_a_wingbeat():
    """The shoulder leads, the wrist lags it and the finger curl lags the wrist (the hand trails),
    and the upstroke half-folds the arm while the downstroke extends it: structure, not the numbers
    the generator happens to use."""
    ph = np.arange(360)
    raise_ = _series(lambda p: PoseControls.flap(p).raise_)
    wrist = _series(lambda p: PoseControls.flap(p).wrist_flex)
    curl = _series(lambda p: PoseControls.flap(p).curl)
    fold = _series(lambda p: PoseControls.flap(p).fold)
    lag = lambda x, y: (int(ph[np.argmax(x)]) - int(ph[np.argmax(y)])) % 360        # noqa: E731
    # raised wing: the wrist droops (flex minimum) some way after the shoulder's peak ...
    wrist_lag = (int(ph[np.argmin(wrist)]) - int(ph[np.argmax(raise_)])) % 360
    assert 15 <= wrist_lag <= 90, wrist_lag
    # ... and the curl peaks later still than the wrist's droop
    curl_lag = lag(curl, raise_)
    assert wrist_lag < curl_lag <= 150, (wrist_lag, curl_lag)
    # the arm half-folds on the upstroke (the rising half of the cycle) and is open on the downstroke
    assert 0.15 <= fold.max() <= 0.6
    assert fold[ph == 270][0] == 0.0 and fold[ph == 200][0] == 0.0
    assert fold[(ph > 0) & (ph < 120)].max() == fold.max()
    # the cycle closes on itself
    assert np.allclose(vars_of(PoseControls.flap(0.0)), vars_of(PoseControls.flap(360.0)), atol=1e-9)
