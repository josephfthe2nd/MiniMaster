"""The wing rig: joints, skin weights and poses.

``test_wings.py`` pins the geometry. These tests pin what makes it a *rig*: a
mesh bound to a joint hierarchy that deforms when joints turn, without the
topology changing, the bones stretching, or the root coming off the body.
"""

import itertools

import numpy as np
import pytest

from minimaster.character import appendages as A
from minimaster.character import mhbase as mh
from minimaster.character import wings as W
from minimaster.character.wingrig import PlacedWing, PoseControls
from minimaster.core.armature import Armature

STYLES = ("membrane", "dragon", "feathered")

# a spread of poses that exercises every control, alone and combined
POSES = [
    {}, {"fold": 0.5}, {"fold": 1.0}, {"raise_": 45.0}, {"raise_": -45.0},
    {"swing": 30.0}, {"twist": 25.0}, {"curl": 30.0}, {"fan": 0.0},
    {"fold": 0.7, "raise_": 20.0, "twist": -10.0},
    {"fold": 0.4, "raise_": 25.0, "curl": 50.0},
]


@pytest.fixture(scope="module", params=STYLES)
def rig(request):
    return W.make_wing_rig(request.param)


@pytest.fixture(scope="module")
def membrane():
    return W.make_wing_rig("membrane")


@pytest.fixture(scope="module")
def feathered():
    return W.make_wing_rig("feathered")


def bone_lengths(rig, pose):
    J = rig.armature.posed_positions(pose)
    return {k: float(np.linalg.norm(J[k] - J[rig.armature.joints[k].parent]))
            for k in rig.bone_keys}


# -- the joint hierarchy --------------------------------------------------

def test_the_skeleton_is_a_real_anatomical_hierarchy(membrane):
    arm = membrane.armature
    assert arm.joints["root"].parent is None
    assert arm.ancestors("d3_tip") == ["d3_ip", "d3_mcp", "d3_cmc", "wrist",
                                       "elbow", "shoulder", "root"]
    # the thumb hangs off the wrist, not off a finger
    assert arm.ancestors("thumb_tip")[:2] == ["thumb_cmc", "wrist"]
    # every digit has its own base joint at the wrist so the fan can spread
    for n in (2, 3, 4, 5):
        assert np.allclose(arm.joints[f"d{n}_cmc"].position, arm.joints["wrist"].position)


def test_digits_are_chains_of_phalanges_with_knuckles(membrane):
    sk = membrane.skeleton
    # joints: base at the wrist, then one per knuckle, then the tip. So digit
    # III is a metacarpal + two phalanges (3 bones), digit II a metacarpal + one.
    assert sk.chains["III"] == ["d3_cmc", "d3_mcp", "d3_ip", "d3_tip"]
    assert sk.chains["II"] == ["d2_cmc", "d2_mcp", "d2_tip"]


def test_every_flight_feather_has_its_own_root_and_tip_joint(feathered):
    sk = feathered.skeleton
    rows = {f["row"] for f in sk.feathers}
    # everything that must move with a fold is a rigged feather, not just the
    # flight feathers: tertials and scapulars drape the root, the alula sits on the thumb
    assert rows == {"primary", "secondary", "tertial", "scapular", "alula"}
    for f in sk.feathers:
        assert feathered.armature.joints[f["tip"]].parent == f["root"]
        assert feathered.armature.joints[f["root"]].parent == f["parent"]


# -- rest pose and weights ------------------------------------------------

def test_the_rest_pose_reproduces_the_mesh_exactly(rig):
    assert np.array_equal(rig.pose_mesh({}).vertices, rig.rest_mesh.vertices)
    zeros = {k: (0.0, 0.0, 0.0) for k in rig.armature.joints}
    assert np.abs(rig.pose_mesh(zeros).vertices - rig.rest_mesh.vertices).max() < 1e-12


def test_every_vertex_has_valid_normalised_weights(rig):
    idx, wt = rig.weights()
    assert (wt >= 0.0).all()
    assert np.allclose(wt.sum(axis=1), 1.0, atol=1e-9)
    assert idx.max() < len(rig.bone_keys)
    assert (wt > 0).sum(axis=1).max() <= 4


def test_rigid_pieces_follow_exactly_one_bone(rig):
    idx, wt = rig.weights()
    rigid = [p for p in rig.pieces if p.bone is not None]
    assert rigid
    for p in rigid:
        a, b = rig.ranges[p.name]
        assert (wt[a:b, 0] == 1.0).all() and (wt[a:b, 1:] == 0.0).all(), p.name
        assert (idx[a:b, 0] == rig.bone_keys.index(p.bone)).all(), p.name


def test_a_finger_is_pulled_only_by_its_own_bones(membrane):
    idx, wt = membrane.weights()
    for name, chain in membrane.skeleton.chains.items():
        a, b = membrane.ranges[f"digit_{name}"]
        used = {membrane.bone_keys[i] for i in np.unique(idx[a:b][wt[a:b] > 0])}
        assert used <= set(chain[1:]), f"digit {name} is pulled by {used - set(chain[1:])}"


def test_each_flight_feather_is_welded_to_its_own_tip_bone(feathered):
    for f in feathered.skeleton.feathers:
        name = f"{f['row']}{f['tip'][len(f['row']):len(f['row']) + 2]}"
        assert feathered.dominant_bone(name) == f["tip"]


def test_coverts_ride_on_the_flight_feather_they_lie_over(feathered):
    """A greater covert that stayed behind when its secondary fanned out would
    leave a hole; it is bound to the same bone instead."""
    for row, src in (("greater", "secondary"), ("primary_covert", "primary")):
        for i in (1, 5):
            assert (feathered.dominant_bone(f"{row}{i:02d}")
                    == feathered.dominant_bone(f"{src}{i:02d}"))


def test_the_membrane_root_is_bound_to_the_body_bone(membrane):
    """The root edge must follow the torso, not the arm."""
    a, b = membrane.ranges["membrane"]
    v = membrane.rest_mesh.vertices[a:b]
    idx, wt = membrane.weights()
    root = (v[:, 0] < 0.02 * membrane.skeleton.span) & (v[:, 2] > 0.5 * membrane.skeleton.root[2])
    assert root.sum() > 10
    carried = {membrane.bone_keys[i] for i in idx[a:b][root][:, 0]}
    assert carried == {"shoulder"}, carried                     # root -> shoulder bone


# -- posing ----------------------------------------------------------------

@pytest.mark.parametrize("pose", POSES, ids=lambda p: "+".join(p) or "rest")
def test_bones_never_stretch(rig, pose):
    rest = bone_lengths(rig, {})
    posed = bone_lengths(rig, rig.pose_from_controls(**pose))
    for k, length in rest.items():
        assert posed[k] == pytest.approx(length, abs=1e-9), k


@pytest.mark.parametrize("pose", POSES, ids=lambda p: "+".join(p) or "rest")
def test_every_pose_stays_a_watertight_solid_per_piece(rig, pose):
    """Skinning moves vertices but never edges, so closed shells stay closed."""
    posed = rig.pose_mesh(rig.pose_from_controls(**pose))
    assert np.isfinite(posed.vertices).all()
    for p in rig.pieces:
        a, b = rig.ranges[p.name]
        assert p.mesh.integrity_report()["watertight"], p.name
        assert posed.faces[(posed.faces >= a).all(axis=1) & (posed.faces < b).all(axis=1)].size \
            == p.mesh.faces.size, p.name


def test_posing_is_deterministic(rig):
    pose = rig.pose_from_controls(fold=0.4, raise_=15.0)
    assert np.array_equal(rig.pose_mesh(pose).vertices, rig.pose_mesh(pose).vertices)


def reach(rig, mesh):
    """Farthest any vertex gets from the shoulder. (Not the x-extent: a folding
    wing's tips swing back across the shoulder, so its x-range is not monotonic.)"""
    return float(np.linalg.norm(mesh.vertices - rig.skeleton.shoulder, axis=1).max())


def test_folding_makes_the_wing_compact(rig):
    assert reach(rig, rig.pose(fold=1.0)) < 0.6 * reach(rig, rig.rest_mesh)


def test_fold_is_progressive(rig):
    reaches = [reach(rig, rig.pose(fold=f)) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    # never grows as the wing closes (it may level off once stowed, when the
    # farthest point is simply the thumb or an elbow knob)
    assert all(b <= a + 0.02 * reaches[0] for a, b in zip(reaches, reaches[1:])), reaches
    assert reaches[0] > 1.5 * reaches[-1]


def test_raising_lifts_the_wingtip_toward_the_upper_surface(rig):
    """+Y is the underside, so a raised tip has a smaller y."""
    tip0 = rig.posed_joints({})["d3_tip"]
    tip_up = rig.posed_joints(rig.pose_from_controls(raise_=40.0))["d3_tip"]
    tip_dn = rig.posed_joints(rig.pose_from_controls(raise_=-40.0))["d3_tip"]
    assert tip_up[1] < tip0[1] - 2.0
    assert tip_dn[1] > tip0[1] + 2.0
    reach = np.linalg.norm(tip0 - rig.armature.joints["shoulder"].position)
    assert tip0[1] - tip_up[1] == pytest.approx(reach * np.sin(np.radians(40.0)), rel=0.35)


def test_the_root_stays_on_the_body_while_the_wing_flaps(membrane):
    a, b = membrane.ranges["membrane"]
    v = membrane.rest_mesh.vertices[a:b]
    root = (v[:, 0] < 0.02 * membrane.skeleton.span) & (v[:, 2] > 0.5 * membrane.skeleton.root[2])
    tip = v[:, 0] > 0.9 * v[:, 0].max()
    for pose in ({"raise_": 50.0}, {"raise_": -30.0}, {"fold": 1.0}):
        moved = np.linalg.norm(membrane.pose(**pose).vertices[a:b] - v, axis=1)
        assert moved[root].max() < 0.05 * moved[tip].mean(), pose


def test_closing_the_fan_brings_the_fingertips_together(membrane):
    def spread(pose):
        J = membrane.posed_joints(membrane.pose_from_controls(**pose))
        return np.linalg.norm(J["d3_tip"] - J["d5_tip"])
    lo = membrane.skeleton.fan_range[0]
    assert spread({"fan": lo}) < 0.75 * spread({})


def test_curl_bends_the_fingers_toward_the_underside(membrane):
    J0 = membrane.posed_joints({})
    J1 = membrane.posed_joints(membrane.pose_from_controls(curl=35.0))
    for n in (3, 4, 5):
        assert J1[f"d{n}_tip"][1] > J0[f"d{n}_tip"][1] + 0.2


def test_finger_bones_stay_on_the_sheet_when_the_wing_moves(membrane):
    """A finger tube is skinned over its own bones and the membrane over all of
    them; if the two ever disagreed the bone would peel off the skin."""
    a, b = membrane.ranges["digit_III"]
    ma, mb = membrane.ranges["membrane"]

    def gap(pose):
        m = membrane.pose_mesh(pose)
        tube = m.vertices[a:b][::3]
        sheet = m.vertices[ma:mb]
        d = np.linalg.norm(tube[:, None, :] - sheet[None, ::2, :], axis=2).min(axis=1)
        return float(d.max())

    rest = gap({})
    for pose in ({"fold": 0.5}, {"raise_": 40.0}, {"curl": 30.0}, {"fan": 0.3}):
        assert gap(membrane.pose_from_controls(**pose)) < max(4.0 * rest, 0.6), pose


def test_a_folding_membrane_pleats_instead_of_exploding(membrane):
    """The slack corrective has to stay bounded: pleats are fabric folds, not
    spikes, so nothing may leave the wing's own reach."""
    spread = float(np.ptp(membrane.rest_mesh.vertices[:, 0]))
    for f in (0.3, 0.6, 1.0):
        v = membrane.pose(fold=f).vertices
        assert np.abs(v - membrane.rest_mesh.vertices.mean(axis=0)).max() < 1.5 * spread


# -- placement on a body ---------------------------------------------------

def _rot(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


@pytest.mark.parametrize("mirror", [False, True])
def test_body_space_skin_matrices_reproduce_the_placed_mesh(membrane, mirror):
    """A host armature that only knows body space must be able to pose the wing:
    M S M^-1 applied to the placed rest vertices equals the placed posed ones,
    for a mirrored (det < 0) placement too."""
    M = np.eye(4)
    M[:3, :3] = _rot(37.0) * (np.diag([-1.0, 1.0, 1.0]) if mirror else np.eye(3)) * 0.8
    M[:3, 3] = [1.0, -2.0, 3.0]
    pw = PlacedWing(membrane, M)
    pose = membrane.pose_from_controls(fold=0.4, raise_=25.0)
    placed_rest = membrane.rest_mesh.transform(M)
    want = pw.mesh(pose).vertices
    skins = pw.skin_matrices_body(pose)
    a, b = membrane.ranges["thumb"]
    key = "thumb_tip"
    S = skins[key]
    got = placed_rest.vertices[a:b] @ S[:3, :3].T + S[:3, 3]
    assert np.allclose(got, want[a:b], atol=1e-9)


def test_wing_joints_attach_to_a_body_armature(membrane):
    body = Armature()
    body.add_joint("chest", (0.0, 0.0, 14.0))
    M = np.eye(4)
    M[:3, 3] = [2.0, 0.0, 15.0]
    names = PlacedWing(membrane, M).attach(body, "chest", prefix="wing_l")
    assert "wing_l:shoulder" in names and "wing_l:d3_tip" in names
    assert body.joints["wing_l:root"].parent == "chest"
    assert np.allclose(body.joints["wing_l:shoulder"].position,
                       membrane.armature.joints["shoulder"].position + [2.0, 0.0, 15.0])
    assert body.ancestors("wing_l:d3_tip")[-2:] == ["wing_l:root", "chest"]
    # the body armature can now pose the wing through the chest
    posed = body.posed_positions({"chest": (0.0, 0.0, 30.0)})
    assert not np.allclose(posed["wing_l:d3_tip"], body.joints["wing_l:d3_tip"].position)


@pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
def test_the_membrane_root_lies_in_the_wings_own_plane_on_the_back():
    """Regression: with a fixed roll the wing plane sloped forward as it
    descended the back, so the waist target projected into it landed on the
    FRONT of the belly. The roll is now derived so the plane contains the line
    to the root: the target is in-plane (local y = 0) and the joint is on the back."""
    base, lib = mh.load()
    v = lib.apply(base.verts, mh.macro_weights(gender=1.0))
    cv, _ = base.body_cage(v)
    for name, pw in A.back_wings("membrane").build_rigs(base, v):
        sign = 1.0 if name.endswith("l") else -1.0
        target = np.array([sign * A.WING_ROOT[0], A.WING_ROOT[1], A.WING_ROOT[2]])
        local = np.linalg.inv(pw.matrix) @ np.append(target, 1.0)
        assert abs(local[1]) < 1e-6, f"{name}: the root target is {local[1]:.3f} out of the wing plane"
        root_local = pw.rig.skeleton.root
        # joints are lifted onto the membrane surface (a few hundredths off the
        # plane), so compare the in-plane position
        assert np.allclose([root_local[0], root_local[2]], [local[0], local[2]], atol=1e-6), name
        root = pw.matrix[:3, :3] @ root_local + pw.matrix[:3, 3]
        assert np.linalg.norm(root - target) < 0.15, name
        assert root[2] < 0.0, f"{name}: the root is on the belly side (z={root[2]:.2f})"


@pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("fold", [0.0, 1.0])
def test_no_wing_pokes_out_in_front_of_the_torso(style, fold):
    """A stowed wing lies along the back; nothing of it may appear in front of
    the body's own front surface at the same height."""
    base, lib = mh.load()
    v = lib.apply(base.verts, mh.macro_weights(gender=1.0))
    cv, _ = base.body_cage(v)
    for name, pw in A.back_wings(style).build_rigs(base, v):
        w = pw.pose(fold=fold).vertices
        for x, y, z in w[::7]:
            near = (np.abs(cv[:, 1] - y) < 0.3) & (np.abs(cv[:, 0] - x) < 1.2)
            if near.any():
                assert z <= cv[near, 2].max() + 0.25, (name, fold, (x, y, z))


class TestOnTheBody:
    @pytest.mark.parametrize("style", STYLES)
    def test_a_posed_pair_is_still_an_exact_mirror_image(self, style):
        base, lib = mh.load()
        v = lib.apply(base.verts, mh.macro_weights(gender=1.0))
        rigs = A.back_wings(style).build_rigs(base, v)
        for pose in ({"fold": 0.6}, {"raise_": 35.0, "twist": 10.0}, {"curl": 20.0},
                     {"fold": 1.0, "curl": 40.0}, {"fold": 0.5, "raise_": 30.0}):
            (_, left), (_, right) = rigs
            mirrored = left.pose(**pose).vertices @ np.diag([-1.0, 1.0, 1.0]).T
            assert np.abs(mirrored - right.pose(**pose).vertices).max() < 1e-9, pose

    @pytest.mark.parametrize("style", STYLES)
    def test_posed_wings_on_the_body_are_watertight_and_follow_the_pose(self, style):
        base, lib = mh.load()
        v = lib.apply(base.verts, mh.macro_weights(gender=1.0))
        app = A.back_wings(style)
        rest = dict(app.build(base, v))
        for name, pw in app.build_rigs(base, v):
            folded = pw.pose(fold=1.0)
            assert folded.integrity_report()["watertight"], name
            assert np.ptp(folded.vertices, axis=0).max() < np.ptp(rest[name].vertices, axis=0).max()

    def test_a_fold_given_in_the_appendage_params_is_a_pose_not_a_rebuild(self):
        base, lib = mh.load()
        v = lib.apply(base.verts, mh.macro_weights(gender=1.0))
        folded = A.back_wings("membrane", fold=1.0).build(base, v)
        rigs = A.back_wings("membrane").build_rigs(base, v)
        for (name, mesh), (_, pw) in zip(folded, rigs):
            assert np.allclose(mesh.vertices, pw.pose(fold=1.0).vertices)

    def test_only_wings_are_rigged(self):
        base = mh.BaseMesh()
        with pytest.raises(ValueError, match="only wings are rigged"):
            A.Appendage("tail", anchor_point=(0, 0, 0)).build_rigs(base)


def test_pose_controls_default_is_the_rest_pose():
    assert PoseControls().fold == 0.0 and PoseControls().fan == 1.0


def test_unknown_style_has_no_rig():
    with pytest.raises(KeyError, match="unknown wing style"):
        W.make_wing_rig("jetpack")


# -- named poses and joint limits -------------------------------------------

from minimaster.character.wingrig import LIMITS, PRESETS  # noqa: E402


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_is_a_valid_pose_of_both_rigs(rig, name):
    posed = rig.pose_mesh(rig.pose_from_controls(name))
    assert np.isfinite(posed.vertices).all()
    assert posed.integrity_report()["watertight"]


def test_a_preset_can_be_overridden_by_keywords(membrane):
    a = membrane.pose_from_controls("upstroke")
    b = membrane.pose_from_controls("upstroke", raise_=10.0)
    assert a["shoulder"] != b["shoulder"]


def test_folded_preset_is_fold_one(membrane):
    assert np.allclose(membrane.pose("folded").vertices, membrane.pose(fold=1.0).vertices)


def test_upstroke_lifts_the_tip_and_downstroke_lowers_it(membrane):
    tip = lambda name: membrane.posed_joints(membrane.pose_from_controls(name))["d3_tip"][1]  # noqa: E731
    rest = membrane.posed_joints({})["d3_tip"][1]
    assert tip("upstroke") < rest < tip("downstroke")


def test_controls_are_clamped_to_what_the_joints_allow(rig):
    """Asking for an impossible pose gets the nearest possible one, not a wing
    wrenched through its own body."""
    lo, hi = LIMITS["raise_"]
    assert np.allclose(rig.pose_mesh(rig.pose_from_controls(raise_=hi + 500)).vertices,
                       rig.pose_mesh(rig.pose_from_controls(raise_=hi)).vertices)
    assert np.allclose(rig.pose_mesh(rig.pose_from_controls(fold=7.0)).vertices,
                       rig.pose_mesh(rig.pose_from_controls(fold=1.0)).vertices)


@pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
def test_a_placed_wing_accepts_a_named_pose():
    base, lib = mh.load()
    v = lib.apply(base.verts, mh.macro_weights(gender=1.0))
    (_, left), _ = A.back_wings("membrane").build_rigs(base, v)
    assert np.allclose(left.pose("folded").vertices, left.pose(fold=1.0).vertices)


# -- the folded wing lies along the body -------------------------------------

def test_a_stowed_wing_lies_along_the_body_line_not_the_frames_z(feathered):
    """On a real placement the frame's +Z points down-and-OUTWARD, so folding
    feathers toward +Z splays them sideways. They must fold toward the line from
    shoulder to root, which is the body."""
    sk = feathered.skeleton
    root = sk.root
    assert abs(sk.body_angle - np.degrees(np.arctan2(root[2], root[0]))) < 1e-9
    posed = feathered.posed_joints(feathered.pose_from_controls(fold=1.0))
    tip = lambda f: posed[f["tip"]] - posed[f["root"]]  # noqa: E731
    want = np.radians(sk.body_angle)
    for f in sk.feathers:
        d = tip(f)
        heading = np.arctan2(d[2], d[0])
        assert abs(np.degrees(heading) - np.degrees(want)) <= 20.0, f["root"]   # the alula rests ~15 off, a little more once lifted out of plane


@pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
def test_the_membrane_root_is_aimed_at_the_waist_of_the_actual_body():
    base, lib = mh.load()
    v = lib.apply(base.verts, mh.macro_weights(gender=1.0))
    app = A.back_wings("membrane")
    for name, pw in app.build_rigs(base, v):
        root_body = pw.matrix[:3, :3] @ pw.rig.skeleton.root + pw.matrix[:3, 3]
        target = np.asarray(A.WING_ROOT) * (1.0 if name.endswith("l") else -1.0)
        target = np.array([abs(A.WING_ROOT[0]) * (1 if name.endswith("l") else -1),
                           A.WING_ROOT[1], A.WING_ROOT[2]])
        # the wing plane carries the root to the target's projection into it
        assert abs(root_body[1] - target[1]) < 1.2, name
        assert abs(root_body[0] - target[0]) < 1.2, name


# -- the pose controls round 2 added ------------------------------------------

def test_the_birds_fan_control_actually_spreads_the_primaries(feathered):
    """Regression: the feather compensation cancelled the fan swing exactly, so
    fan=0..1.5 left every primary where it was."""
    sk = feathered.skeleton

    def spread(fan):
        posed = feathered.posed_joints(feathered.pose_from_controls(fan=fan))
        yaws = []
        for f in sk.feathers:
            if f["row"] == "primary":
                d = posed[f["tip"]] - posed[f["root"]]
                yaws.append(np.degrees(np.arctan2(d[2], d[0])))
        return max(yaws) - min(yaws)

    s = [spread(f) for f in (0.0, 0.5, 1.0, 1.5)]
    assert s == sorted(s) and s[0] < 3.0 and s[2] > 40.0, s


def test_a_flap_leads_with_the_wrist_and_the_hand_trails(membrane):
    """At the top of the upstroke the hand hangs BELOW the wrist; a plank swung
    from the shoulder (raise only) holds the tip above it."""
    from minimaster.character.wingrig import PoseControls as PC

    def hand_over_wrist(ctrl):
        J = membrane.posed_joints(membrane.pose_from_controls(ctrl))
        return J["d3_tip"][1] - J["wrist"][1]          # +Y is down

    plank = hand_over_wrist(PC(raise_=45.0))
    flap = hand_over_wrist(PC.flap(90.0, amp=45.0))
    assert flap > plank + 1.0, (flap, plank)


def test_wrist_flex_cocks_the_hand_out_of_the_wing_plane(rig):
    J0 = rig.posed_joints({})
    up = rig.posed_joints(rig.pose_from_controls(wrist_flex=30.0))
    down = rig.posed_joints(rig.pose_from_controls(wrist_flex=-30.0))
    assert up["d3_tip"][1] < J0["d3_tip"][1] - 0.5        # cocked up = smaller y
    assert down["d3_tip"][1] > J0["d3_tip"][1] + 0.5      # drooping = larger y
    # the arm itself does not move
    assert np.allclose(up["wrist"], J0["wrist"])


def test_a_stowed_wing_can_still_be_raised(rig):
    """Regression: raise rotated about the frame's Z, which is parallel to a
    stowed humerus, so a folded wing could not be lifted."""
    def height(raise_):
        # the ELBOW: in a Z-fold the forearm folds back along the humerus, so the
        # wrist's lift cancels against the humerus's
        J = rig.posed_joints(rig.pose_from_controls(fold=1.0, raise_=raise_))
        return -J["elbow"][1]
    # (a stowed wing has used up most of the raise range -- see FOLD_LIMITS)
    assert height(25.0) - height(0.0) > 0.8


def test_the_folded_hand_lies_back_along_the_forearm(membrane):
    J = membrane.posed_joints(membrane.pose_from_controls(fold=1.0))
    fore = J["wrist"] - J["elbow"]
    mc = J["d3_mcp"] - J["wrist"]
    c = float(fore @ mc / (np.linalg.norm(fore) * np.linalg.norm(mc)))
    assert c < -0.7, f"the hand is not folded back along the forearm (cos {c:.2f})"


def test_the_thumb_tucks_along_the_forearm_when_stowed(membrane):
    J0 = membrane.posed_joints({})
    J1 = membrane.posed_joints(membrane.pose_from_controls(fold=1.0))
    thumb = J1["thumb_tip"] - J1["wrist"]
    fore = J1["wrist"] - J1["elbow"]
    # lies roughly parallel to the forearm, not standing out sideways
    assert abs(float(thumb @ fore)) / (np.linalg.norm(thumb) * np.linalg.norm(fore)) > 0.5


def test_the_membrane_is_thick_enough_to_print(membrane):
    """~0.3 mm at the 1.79 mm/unit of a 32 mm figure: the free edge included."""
    a, b = membrane.ranges["membrane"]
    n1, m1 = next(p for p in membrane.pieces if p.name == "membrane").pleat["shape"]
    nv = n1 * m1
    v = membrane.rest_mesh.vertices[a:b]
    thick = np.linalg.norm(v[:nv] - v[nv:2 * nv], axis=1)
    assert thick.min() > 0.15, f"thinnest membrane is {thick.min():.3f} units"


@pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
def test_stowed_wings_rest_on_the_back_instead_of_being_buried_in_it():
    """The wing plane is sunk into the back so a print is welded; a stowed wing
    lying along that plane was 57% inside the torso. The feathered wing now stows
    FLAT on the back about a root that stands off the surface (``stow_against``),
    which took the stowed pair from 28% inside the torso to ~4%, and it still touches
    (the under-coverts lie against the back), so a folded print is attached."""
    from tests.test_wings import _inside_body
    base, lib = mh.load()
    v = lib.apply(base.verts, mh.macro_weights(gender=1.0))
    cv, cq = base.body_cage(v)
    tris = cv[np.vstack([cq[:, [0, 1, 2]], cq[:, [0, 2, 3]]])]
    for style, limit in (("feathered", 0.10), ("membrane", 0.36)):
        for name, pw in A.back_wings(style).build_rigs(base, v):
            m = pw.pose(fold=1.0)
            w = m.vertices[::9]
            sign = 1.0 if name.endswith("l") else -1.0
            inside = _inside_body(w, tris).mean()
            assert inside < limit, (style, name)
            assert (sign * w[:, 0] < -0.1).mean() < 0.02, (style, name)
            if style == "feathered":
                assert inside > 0.01, f"{name}: the stowed wing no longer touches the back at all"
                flight = np.concatenate([m.vertices[a:b] for p in pw.rig.pieces
                                         if p.name.startswith(("primary0", "primary10", "secondary"))
                                         for a, b in [pw.rig.ranges[p.name]]])[::3]
                assert _inside_body(flight, tris).mean() < 0.08, (style, name)


# -- orientation must not depend on the body's sliders -------------------------

@pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
def test_the_wings_orientation_does_not_depend_on_body_morphs():
    """Regression: the span axis came from projecting a hint into the tangent
    plane of ONE quad of the morphed body, so a shoulder or pectoral slider
    re-aimed the wing -- 19.5 degrees between the plain and the anatomy-adjusted
    body, with the spread membrane anywhere from flat to 6 units behind the back.
    The rake is now authored (the hint, in body space); only the origin follows
    the body."""
    from minimaster.character import creature_anatomy as ca
    from minimaster.character.sliders import Character, SliderCatalog
    base, lib = mh.load()
    cat = SliderCatalog(lib.names)

    def body(**macro):
        c = Character()
        c.macro.update(**macro)
        return lib.apply(base.verts, c.target_weights(cat))

    adj = Character()
    adj.macro.update(gender=1.0, muscle=0.55)
    ca.apply_to(adj, [A.back_wings("membrane")], cat)
    bodies = [body(gender=1.0), lib.apply(base.verts, adj.target_weights(cat)),
              body(gender=0.0), body(gender=1.0, weight=1.0, muscle=1.0),
              body(gender=1.0, weight=0.0, muscle=0.0)]
    axes = []
    for v in bodies:
        app = A.back_wings("membrane")
        for name, pw in app.build_rigs(base, v):
            ax = pw.matrix[:3, 0] / np.linalg.norm(pw.matrix[:3, 0])
            axes.append(ax if name.endswith("l") else ax * np.array([-1.0, 1.0, 1.0]))
        (_, L), (_, R) = app.build(base, v)
        assert np.abs(L.vertices @ np.diag([-1.0, 1.0, 1.0]).T - R.vertices).max() < 1e-9
    axes = np.array(axes)
    worst = max(np.degrees(np.arccos(np.clip(a @ b, -1, 1))) for a in axes for b in axes)
    assert worst < 3.0, f"the span axis moves {worst:.1f} degrees between bodies"


def test_the_primary_fan_does_not_change_when_the_wing_is_raised_or_twisted(feathered):
    """Regression: feather headings were read from the plan projection of a tilted
    frame, so raise_ 90 made P1-P10 span 123 degrees instead of 62. The fan is a
    property of the plumage, not of how the wing is held: its 3D angle must not
    change under any rigid motion of the arm."""
    sk = feathered.skeleton

    def fan_angle(**kw):
        posed = feathered.posed_joints(feathered.pose_from_controls(**kw))
        d = {}
        for f in sk.feathers:
            if f["row"] == "primary":
                v = posed[f["tip"]] - posed[f["root"]]
                d[f["root"]] = v / np.linalg.norm(v)
        a, b = d["primary01_r"], d["primary10_r"]
        return np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0)))

    rest = fan_angle()
    for kw in ({"raise_": 90.0}, {"raise_": -60.0}, {"twist": 45.0}, {"twist": -45.0},
               {"wrist_flex": -60.0}, {"hand_twist": 30.0}):
        assert fan_angle(**kw) == pytest.approx(rest, abs=0.5), kw
    # and a stowed wing stays packed even when it is raised
    assert fan_angle(fold=1.0, raise_=45.0) < 12.0


def test_a_folding_membrane_never_turns_inside_out(membrane):
    """Regression: the slab was re-oriented per cell toward the skinned 'up'
    vector, which turned it inside out wherever a panel doubled back (signed
    volume fell from +0.75 to -0.03 as the wing stowed)."""
    rest = membrane.rest_mesh.integrity_report()["volume"]
    for fold in (0.25, 0.5, 0.75, 1.0):
        vol = membrane.pose(fold=fold).integrity_report()["volume"]
        assert vol > 0.4 * rest, (fold, vol, rest)
    assert membrane.pose(fold=1.0, raise_=45.0).integrity_report()["volume"] > 0.4 * rest


def test_controls_are_not_mutated_by_posing(membrane):
    from minimaster.character.wingrig import PoseControls as PC
    c = PC(raise_=200.0, fold=3.0)
    membrane.pose_from_controls(c)
    assert (c.raise_, c.fold) == (200.0, 3.0)


def test_a_membrane_refuses_a_fan_its_skin_cannot_take(membrane):
    """fan=0 closed the fingers on a sheet that cannot follow (stretch up to 2.9x)."""
    closed = membrane.pose_from_controls(fan=0.0)
    limit = membrane.pose_from_controls(fan=membrane.skeleton.fan_range[0])
    assert closed == limit


@pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
def test_every_pose_control_can_ride_in_a_wing_definition():
    """Regression: POSE_KEYS was hand-written, so back_wings(..., wrist_flex=20)
    leaked a pose control into the wing builder and crashed."""
    from dataclasses import fields
    from minimaster.character.wingrig import PoseControls as PC
    assert set(A.Appendage.POSE_KEYS) == {f.name for f in fields(PC)}
    base, lib = mh.load()
    v = lib.apply(base.verts, mh.macro_weights(gender=1.0))
    for key, val in (("wrist_flex", -30.0), ("hand_twist", 20.0), ("curl", 20.0), ("fan", 0.8)):
        built = A.back_wings("membrane", **{key: val}).build(base, v)
        rest = A.back_wings("membrane").build(base, v)
        assert not np.allclose(built[0][1].vertices, rest[0][1].vertices), key


# -- round-3 rig fixes ----------------------------------------------------

def _membrane_slab(rig):
    piece = next(p for p in rig.pieces if p.pleat)
    a, b = rig.ranges[piece.name]
    n1, m1 = piece.pleat["shape"]
    return piece, a, n1, m1


@pytest.mark.parametrize("style", ["membrane", "dragon"])
def test_the_slab_thickness_axis_is_consistent_across_the_whole_sheet(style):
    """The dragon's deep scallops leave each fingertip along the grid's own rays,
    where du x dv collapses; one column of the slab then had its thickness lying
    in the sheet (and the pleats, which stand along that axis, ballooned)."""
    rig = W.make_wing_rig(style)
    piece, a, n1, m1 = _membrane_slab(rig)
    nv = n1 * m1
    V = rig.rest_mesh.vertices[a:a + 2 * nv]
    up = V[:nv] - V[nv:]
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    # the sheet is a gently cambered plane facing -y (up): no vertex may lie on
    # the wrong side, and none may be tilted far off it (the root edge is steepest)
    assert (up[:, 1] < -0.65).all(), float(up[:, 1].max())


def test_grid_normals_are_untouched_where_the_grid_is_well_conditioned():
    from minimaster.character.wingrig import grid_normals
    u, v = np.meshgrid(np.linspace(0, 3, 7), np.linspace(0, 2, 5), indexing="ij")
    grid = np.stack([u, 0.1 * u * v, v], axis=2)
    du, dv = np.gradient(grid, axis=0), np.gradient(grid, axis=1)
    want = np.cross(du, dv)
    want /= np.linalg.norm(want, axis=2, keepdims=True)
    assert np.allclose(grid_normals(grid), want)


def test_grid_normals_survive_a_degenerate_column():
    from minimaster.character.wingrig import grid_normals
    u, v = np.meshgrid(np.linspace(0, 3, 9), np.linspace(0, 2, 7), indexing="ij")
    grid = np.stack([u, 0 * u, v], axis=2)
    # fold column 3 back onto its neighbours' ray: du x dv ~ 0 there
    grid[:, 3] = grid[:, 2] + 1e-4 * (grid[:, 4] - grid[:, 2])
    n = grid_normals(grid)
    assert np.isfinite(n).all()
    assert (np.abs(n[..., 1]) > 0.95).all()          # all still face +-y, none in-plane
    assert (np.sign(n[..., 1]) == np.sign(n[0, 0, 1])).all()


@pytest.mark.parametrize("style", ["membrane", "dragon"])
def test_a_stowed_membrane_is_a_flat_bundle_not_a_tent(style):
    """The pleats stand along the slab's up axis and used to rise a few bone
    thicknesses off the sheet. A stowed membrane must lie among its bones: the sheet
    may stand at most 0.5 units above the topmost finger/arm bone and 0.35 below the
    lowest (uncapped it reaches 1.28 above and 0.54 below). Measured against the
    bones, not as a bounding box, so it does not depend on the rest camber."""
    rig = W.make_wing_rig(style)
    mem = next(p for p in rig.pieces if p.pleat)
    bones = [p.name for p in rig.pieces if p.name.startswith("digit_") or p.name == "arm"]
    for fold in (0.3, 0.5, 0.7, 0.8, 0.9, 1.0):
        m = rig.pose_mesh(rig.pose_from_controls(fold=fold))
        ym = m.vertices[slice(*rig.ranges[mem.name]), 1]
        yb = np.concatenate([m.vertices[slice(*rig.ranges[n]), 1] for n in bones])
        assert yb.min() - ym.min() < 0.5, (style, fold, "above", yb.min() - ym.min())
        assert ym.max() - yb.max() < 0.35, (style, fold, "below", ym.max() - yb.max())
        assert m.volume() > 0.0, (style, fold)


def test_the_pleat_cap_is_what_keeps_a_stowed_membrane_flat():
    from minimaster.character import wingrig as R
    rig = W.make_wing_rig("membrane")
    pose = rig.pose_from_controls(fold=1.0)
    capped = np.ptp(rig.pose_mesh(pose).vertices[:, 1])
    old, R.PLEAT_CAP = R.PLEAT_CAP, 1e6
    try:
        uncapped = np.ptp(rig.pose_mesh(pose).vertices[:, 1])
    finally:
        R.PLEAT_CAP = old
    assert uncapped > capped + 0.8, (capped, uncapped)


def test_a_knuckle_curl_and_the_fold_concertina_multiply_instead_of_adding(membrane):
    from minimaster.core import math3d as m3
    sk = membrane.skeleton
    c = PoseControls(fold=0.8, curl=30.0)
    pose = sk.controls_to_pose(c)
    k = W.knuckle_phase(0.8)
    assert 0.0 < k < 1.0                         # mid-way through the concertina, so the test bites
    for name, chain in sk.chains.items():
        for jn, bend in zip(chain[1:-1], sk.bends[name]):
            a = np.radians(sk.out_angle[jn])
            axis = np.array([-np.sin(a), 0.0, np.cos(a)])
            curl = W._rotation_about(axis, c.curl * sk.curl_gain)
            fold = m3.rot_y(-(180.0 - bend) * sk.knuckle_fold * k)
            assert np.allclose(m3.euler_rotation(*pose[jn]), fold @ curl, atol=1e-6), jn


def test_the_fold_dependent_limits_leave_only_the_unused_range():
    from minimaster.character.wingrig import clamp_controls
    assert clamp_controls(PoseControls(fold=0.0, swing=85.0, raise_=100.0)).swing == 85.0
    assert clamp_controls(PoseControls(fold=0.0, swing=85.0, raise_=100.0)).raise_ == 100.0
    stowed = clamp_controls(PoseControls(fold=1.0, swing=85.0, raise_=100.0))
    assert stowed.swing <= 5.0 + 1e-9 and stowed.raise_ <= 25.0 + 1e-9
    # monotone: the more folded, the less is left
    prev = (1e9, 1e9)
    for f in np.linspace(0.0, 1.0, 6):
        c = clamp_controls(PoseControls(fold=float(f), swing=85.0, raise_=100.0))
        assert c.swing <= prev[0] + 1e-9 and c.raise_ <= prev[1] + 1e-9
        prev = (c.swing, c.raise_)
    # negative ranges do not shrink
    low = clamp_controls(PoseControls(fold=1.0, swing=-30.0, raise_=-70.0))
    assert (low.swing, low.raise_) == (-30.0, -70.0)


@pytest.mark.parametrize("fold", [0.0, 0.5, 1.0])
def test_extreme_controls_cannot_wrench_the_humerus_past_what_a_shoulder_does(rig, fold):
    J0 = rig.posed_joints({})
    h0 = J0["elbow"] - J0["shoulder"]
    J = rig.posed_joints(rig.pose_from_controls(fold=fold, swing=85.0, raise_=100.0))
    h = J["elbow"] - J["shoulder"]
    ang = np.degrees(np.arccos(np.clip(h @ h0 / np.linalg.norm(h) / np.linalg.norm(h0), -1, 1)))
    assert ang < 125.0, ang


@pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
def test_a_named_pose_can_ride_along_in_the_appendage_params():
    base, lib = mh.load()
    v = lib.apply(base.verts, mh.macro_weights(gender=1.0))
    plain = dict(A.back_wings("membrane").build(base, v))
    folded = dict(A.back_wings("membrane", preset="folded").build(base, v))
    explicit = dict(A.back_wings("membrane", fold=1.0).build(base, v))
    for name in plain:
        assert np.allclose(folded[name].vertices, explicit[name].vertices)
        assert not np.allclose(folded[name].vertices, plain[name].vertices)
    # explicit controls override the preset
    over = dict(A.back_wings("membrane", preset="folded", fold=0.0).build(base, v))
    for name in plain:
        assert np.allclose(over[name].vertices, plain[name].vertices)


# -- the stowed and mid-fold membrane wing ---------------------------------

MEMBRANE_STYLES = ("membrane", "dragon")


def _digit_bones(rig, J):
    """Every finger bone as (digit, start, end) in the posed wing."""
    out = []
    for name, chain in rig.skeleton.chains.items():
        pts = [J["wrist"]] + [J[j] for j in chain[1:]]
        out += [(name, a, b) for a, b in zip(pts[:-1], pts[1:])]
    return out


def _worst_off_axis(rig, fold=1.0):
    """Widest angle (degrees) between any finger bone and the forearm's axis."""
    J = rig.posed_joints(rig.pose_from_controls(fold=fold))
    fore = J["wrist"] - J["elbow"]
    worst = 0.0
    for _, a, b in _digit_bones(rig, J):
        d = b - a
        c = abs(float(d @ fore)) / (np.linalg.norm(d) * np.linalg.norm(fore))
        worst = max(worst, float(np.degrees(np.arccos(min(c, 1.0)))))
    return worst


@pytest.mark.parametrize("style", MEMBRANE_STYLES)
def test_stowed_finger_bones_lie_along_the_forearm_not_criss_crossing(style):
    """Each phalanx folds all the way back over its metacarpal (a full concertina), so the
    three bones of a finger end up one rod along the forearm. At 90% of a hairpin every
    knuckle left ~17 degrees and the long Z of digit III crossed the short Z of digit V
    in an X (bones were up to 49 degrees off the forearm)."""
    rig = W.make_wing_rig(style)
    assert _worst_off_axis(rig) < 30.0
    rig.skeleton.knuckle_fold = 0.9                  # the old value: the check has to be able to fail
    assert _worst_off_axis(rig) > 40.0


def _rod_gaps(rig):
    """Centreline distance between adjacent stowed fingers (and the last finger and the
    forearm), sampled along the metacarpals from 1.4 to 2.6 units out of the wrist."""
    J = rig.posed_joints(rig.pose_from_controls(fold=1.0))
    sk = rig.skeleton
    axes = []
    for n in ("II", "III", "IV", "V"):
        v = J[sk.chains[n][1]] - J["wrist"]
        axes.append(v / np.linalg.norm(v))
    f = J["elbow"] - J["wrist"]
    axes.append(f / np.linalg.norm(f))
    ds = np.linspace(1.4, 2.6, 7)
    return [min(float(np.linalg.norm((a - b) * d)) for d in ds) for a, b in zip(axes[:-1], axes[1:])]


@pytest.mark.parametrize("style", MEMBRANE_STYLES)
def test_stowed_fingers_are_separate_ordered_rods_with_daylight_between_them(style):
    """Collapsing the fan to a fixed 8% of its rest angle left the four metacarpals 1-2
    degrees apart (0.05-0.1 units between rod centrelines against a rod 0.3 across): one fused bundle. The stowed spread is deliberate (a fan residual and an
    alternating tilt out of the plane), keeps the digits in order, and leaves the rods
    clear of each other and of the forearm."""
    rig = W.make_wing_rig(style)
    sk = rig.skeleton
    J = rig.posed_joints(rig.pose_from_controls(fold=1.0))
    mc = [J[sk.chains[n][1]] - J["wrist"] for n in ("II", "III", "IV", "V")]
    up = np.array([0.0, -1.0, 0.0])
    # in-plane order is preserved (II < III < IV < V going round the wrist)
    plan = [float(np.arctan2(m[2], m[0])) for m in mc]
    steps = np.diff(np.unwrap(plan))
    assert (steps > np.radians(0.5)).all() or (steps < -np.radians(0.5)).all(), np.degrees(steps)
    # alternate fingers sit above and below the wing plane
    tilt = np.array([float(m @ up) / np.linalg.norm(m) for m in mc])
    tilt -= tilt.mean()                       # the whole stowed wing is lifted off its plane
    assert tilt[0] * tilt[1] < 0.0 and tilt[1] * tilt[2] < 0.0 and tilt[2] * tilt[3] < 0.0, tilt
    # and a heavier rod (the dragon's) gets proportionally more room
    scale = 1.0 if style == "membrane" else W.DRAGON["bone_scale"]
    assert min(_rod_gaps(rig)) > 0.22 * scale, _rod_gaps(rig)
    # without the deliberate spread the same rods are fused (5-10 hundredths of a unit apart)
    sk.stow_fan, sk.stow_tilt = {}, {}
    assert min(_rod_gaps(rig)) < 0.5 * 0.22 * scale


def _finger_columns(rig, pose):
    """The sheet's mid-surface column along each digit, posed."""
    piece = next(p for p in rig.pieces if p.pleat)
    a, b = rig.ranges[piece.name]
    n1, m1 = piece.pleat["shape"]
    nv = n1 * m1
    v = rig.pose_mesh(pose).vertices[a:b]
    mid = (0.5 * (v[:nv] + v[nv:2 * nv])).reshape(n1, m1, 3)
    return [mid[:, c0] for c0, _ in piece.pleat["sectors"]]


def _backtrack_and_bow(col):
    """How far a column doubles back along its own chord and how far it bows off it, both
    as a fraction of the chord."""
    chord = col[-1] - col[0]
    length = float(np.linalg.norm(chord))
    u = chord / length
    proj = (col - col[0]) @ u
    off = np.linalg.norm((col - col[0]) - np.outer(proj, u), axis=1)
    return float(np.maximum(0.0, -np.diff(proj)).sum() / length), float(off.max() / length)


@pytest.mark.parametrize("style", MEMBRANE_STYLES)
def test_a_stowed_sheet_runs_along_its_fingers_as_a_closed_fan_not_a_zigzag(style, monkeypatch):
    """Skinned bone by bone, the sheet's column along a stowed finger ran out, back and out
    again (it doubled back along its chord by 25-60% and bowed 17-33% off it) and the panels
    between two of them crumpled. Gathered, every column is a straight ray from the wrist to
    the fingertip."""
    rig = W.make_wing_rig(style)
    for fold in (1.0, 0.97):
        for col in _finger_columns(rig, rig.pose_from_controls(fold=fold)):
            back, bow = _backtrack_and_bow(col)
            assert back < 0.02, (fold, back)
            assert bow < 0.1, (fold, bow)
    # the check can fail: with the gather switched off the columns zigzag
    monkeypatch.setattr(type(rig), "_gather_fingers", lambda self, mid, r_mid, pleat, pose: mid)
    cols = _finger_columns(rig, rig.pose_from_controls(fold=1.0))
    assert max(_backtrack_and_bow(c)[0] for c in cols) > 0.2


@pytest.mark.parametrize("pose", [{"curl": 40.0}, {"curl": 40.0, "fan": 0.6}, {"fold": 0.3, "curl": 30.0},
                                  {"fold": 0.5}, {"fold": 0.7}, {"raise_": 40.0, "twist": 15.0}])
def test_a_curl_or_an_open_half_fold_keeps_the_skinned_sheet_taut(pose, monkeypatch):
    """The finger columns are only gathered once a finger has folded up: a modest curl is
    a smooth arc (a taut sheet) and a closing fan has straight fingers. In those poses
    the gather must be a no-op."""
    rig = W.make_wing_rig("membrane")
    p = rig.pose_from_controls(**pose)
    with_gather = rig.pose_mesh(p).vertices.copy()
    monkeypatch.setattr(type(rig), "_gather_fingers", lambda self, mid, r_mid, pleat, pose: mid)
    assert np.array_equal(with_gather, rig.pose_mesh(p).vertices)


def test_the_gather_engages_once_the_fingers_fold_up(monkeypatch):
    rig = W.make_wing_rig("membrane")
    p = rig.pose_from_controls(fold=1.0)
    with_gather = rig.pose_mesh(p).vertices.copy()
    monkeypatch.setattr(type(rig), "_gather_fingers", lambda self, mid, r_mid, pleat, pose: mid)
    assert np.abs(with_gather - rig.pose_mesh(p).vertices).max() > 1.0


def test_the_knuckles_fold_over_the_last_stretch_not_through_the_whole_fold():
    """A finger folding in the wing plane goes through an L, a U and a spiral on its way to
    the Z, and no sheet can follow that: the fingers stay straight while the fan closes and
    the hand folds back, then fold up quickly."""
    sk = W.make_wing_rig("membrane").skeleton
    ks = [W.knuckle_phase(f) for f in np.linspace(0.0, 1.0, 21)]
    assert ks[0] == 0.0 and ks[-1] == pytest.approx(1.0)
    assert all(b >= a for a, b in zip(ks, ks[1:]))
    assert W.knuckle_phase(W.KNUCKLE_FOLD_START) == 0.0
    for f in (0.3, 0.5, W.KNUCKLE_FOLD_START):                       # straight: no knuckle is posed at all
        pose = sk.controls_to_pose(PoseControls(fold=f))
        for chain in sk.chains.values():
            for jn in chain[1:-1]:
                assert pose.get(jn, (0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0), (f, jn)
    pose = sk.controls_to_pose(PoseControls(fold=0.9))
    assert any(abs(v) > 20.0 for jn in sk.chains["III"][1:-1] for v in pose[jn])


@pytest.mark.parametrize("style", MEMBRANE_STYLES)
def test_a_cupped_wing_keeps_its_skin_taut_and_a_folded_one_pleats(style, monkeypatch):
    """A curl leans the fingers in together; that is not the wing closing, and its slack
    must not be gathered into the pleats of a stowed wing (rucked bulges at the tip)."""
    from minimaster.character import wingrig as R
    rig = W.make_wing_rig(style)
    assert rig._hand_fold(rig.pose_from_controls(curl=40.0)) == pytest.approx(0.0, abs=1e-6)
    assert rig._hand_fold(rig.pose_from_controls(fold=1.0)) > 0.85
    a, b = rig.ranges["membrane"]
    curl = rig.pose_from_controls(curl=30.0)
    stowed = rig.pose_from_controls(fold=1.0)
    taut = rig.pose_mesh(curl).vertices[a:b].copy()
    folded = rig.pose_mesh(stowed).vertices.copy()
    with monkeypatch.context() as m:
        m.setattr(R, "ELASTIC_CUPPED", 0.0)
        m.setattr(R, "PLEAT_CUPPED", 0.0)
        ruched = rig.pose_mesh(curl).vertices[a:b]
        folded_again = rig.pose_mesh(stowed).vertices
    with monkeypatch.context() as m:                                # pure skinning, no pleats at all
        m.setattr(R.WingRig, "_apply_pleats", lambda self, posed, pose=None, gain=0.8: posed)
        skinned = rig.pose_mesh(curl).vertices[a:b]
    moved = lambda v: float(np.linalg.norm(v - skinned, axis=1).mean())        # noqa: E731
    # how far the pleat pass moves the sheet from where skinning leaves it: far less when taut
    assert moved(taut) < 0.5 * moved(ruched), (moved(taut), moved(ruched))
    # ... and the stowed wing is the same either way: it is never "cupped"
    assert np.array_equal(folded, folded_again)


@pytest.mark.parametrize("style", MEMBRANE_STYLES)
@pytest.mark.parametrize("fold", [0.0, 0.5, 0.8, 1.0])
def test_the_arm_wing_stays_taut_while_the_wing_folds(style, fold):
    """The propatagium (the skin in front of the arm) is a flat triangle hinged on the elbow:
    it never strays farther from the arm than at rest and never swells into a lobe."""
    rig = W.make_wing_rig(style)

    def stray(pose):
        J = rig.posed_joints(pose)
        a, b = rig.ranges["propatagium"]
        v = rig.pose_mesh(pose).vertices[a:b]
        d = []
        for p, q in (("shoulder", "elbow"), ("elbow", "wrist")):
            ab = J[q] - J[p]
            t = np.clip(((v - J[p]) @ ab) / (ab @ ab), 0.0, 1.0)
            d.append(np.linalg.norm(v - (J[p] + t[:, None] * ab), axis=1))
        return float(np.min(d, axis=0).max()), float(np.ptp(v[:, 1]))

    d0, _ = stray(rig.pose_from_controls())
    d, thick = stray(rig.pose_from_controls(fold=fold))
    assert d <= d0 + 1e-6
    assert thick < 1.2


# -- the feathered wing's stowed and intermediate poses ----------------------------

def _feathered_on_body(verts=None):
    base, lib = mh.load()
    v = lib.apply(base.verts, mh.macro_weights(gender=1.0)) if verts is None else verts
    return base, lib, v, A.back_wings("feathered").build_rigs(base, v)


def _plane_tilt_from_back(pw, pose, bone="secondary07_t"):
    """Degrees between a stowed feather's plane and the back's own plane (body space)."""
    sm = pw.skin_matrices_body(pose)[bone]
    n = sm[:3, :3] @ (pw.matrix[:3, :3] @ np.array([0.0, 1.0, 0.0]))     # the feather's under-normal
    n /= np.linalg.norm(n)
    return float(np.degrees(np.arccos(abs(n[2]))))                       # the back faces +-Z


@pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
def test_a_stowed_feathered_wing_lies_flat_on_the_back_not_leaning_off_it():
    """The wing is mounted at an angle so the SPREAD wing sweeps behind the shoulder; stowed
    along that plane it was a plank leaning ~55 degrees off the back. ``stow_against`` pivots
    the wing about its root until the plane lies on the back (within the small lean that
    clears the buttocks)."""
    base, lib, v, rigs = _feathered_on_body()
    (_, pw), _ = rigs
    pose = pw.rig.pose_from_controls(fold=1.0)
    assert _plane_tilt_from_back(pw, pose) < 22.0
    # the same wing without knowing its back keeps the old attitude (this is what the test is about)
    bare = PlacedWing(W.make_wing_rig("feathered", root=tuple(pw.rig.skeleton.joints["root"][[0, 2]])),
                      pw.matrix)
    assert _plane_tilt_from_back(bare, bare.rig.pose_from_controls(fold=1.0)) > 40.0
    # progressive: half way folded the plane is half way there
    mid = _plane_tilt_from_back(pw, pw.rig.pose_from_controls(fold=0.5))
    spread = _plane_tilt_from_back(pw, pw.rig.pose_from_controls(fold=0.0))
    assert _plane_tilt_from_back(pw, pose) < mid < spread


def test_stow_rotation_carries_the_upper_side_onto_the_back_by_the_lean():
    sk = W.make_wing_rig("feathered").skeleton
    out = np.array([0.3, -0.8, 0.5]); out /= np.linalg.norm(out)       # the back, in the wing's frame
    down = np.array([-0.8, 0.1, 0.6]); down -= (down @ out) * out; down /= np.linalg.norm(down)
    side = np.cross(down, out)
    sk.stow_against(out, down, side)
    sk.stow_lean, sk.stow_toe_in = 6.0, 0.0
    ang = lambda a, b: float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))   # noqa: E731
    assert sk._stow_rotation(0.0) is None
    Q = sk._stow_rotation(1.0)
    assert np.allclose(Q @ Q.T, np.eye(3)) and np.linalg.det(Q) == pytest.approx(1.0)
    # the wing's upper side ends up on the back's normal, tilted by the lean ...
    assert ang(Q @ np.array([0.0, -1.0, 0.0]), out) == pytest.approx(6.0, abs=1e-6)
    # ... and its axis (shoulder -> root) runs down the back, its lower end standing off by the lean
    b = sk.joints["root"] - sk.joints["shoulder"]
    b /= np.linalg.norm(b)
    assert ang(Q @ b, down) == pytest.approx(6.0, abs=1e-6)
    assert (Q @ b) @ out > 0.0
    # toe-in turns the axis toward the spine (-side) as well
    sk.stow_toe_in = 8.0
    assert (sk._stow_rotation(1.0) @ b) @ side < -0.1
    # progressive: half way is half the rotation
    angle = lambda R: np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))   # noqa: E731
    assert angle(sk._stow_rotation(0.5)) == pytest.approx(0.5 * angle(sk._stow_rotation(1.0)), rel=1e-6)
    # a wing that is not told where its back is does not stow flat at all
    assert W.make_wing_rig("feathered").skeleton._stow_rotation(1.0) is None


def test_the_flat_stow_is_only_for_styles_that_ask_for_it():
    """Placing a membrane wing must not change how it folds (the bat and dragon keep their
    own stow)."""
    for style in ("membrane", "dragon"):
        plain = W.make_wing_rig(style)
        told = W.make_wing_rig(style)
        told.skeleton.stow_against((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        assert not told.skeleton.stows_flat
        for kw in ({"fold": 1.0}, {"fold": 0.5, "raise_": 20.0}):
            assert np.array_equal(plain.pose(**kw).vertices, told.pose(**kw).vertices), (style, kw)


def _morphed_bodies(base, lib):
    from minimaster.character import creature_anatomy as ca
    from minimaster.character.sliders import Character, SliderCatalog
    cat = SliderCatalog(lib.names)

    def body(**macro):
        c = Character()
        c.macro.update(**macro)
        return lib.apply(base.verts, c.target_weights(cat))

    adj = Character()
    adj.macro.update(gender=1.0, muscle=0.55)
    ca.apply_to(adj, [A.back_wings("feathered")], cat)
    return [body(gender=1.0), lib.apply(base.verts, adj.target_weights(cat)),
            body(gender=1.0, weight=1.0, muscle=1.0), body(gender=1.0, weight=0.0, muscle=0.0)]


@pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
def test_the_stowed_pair_clears_the_spine_and_stays_off_the_back_on_every_body():
    """Slim bodies once showed the folded pair touching the spine, and a wing stowed about a
    fixed body point is buried on a heavy back and floats off a slim one. The root is relative
    to the anchor, so the stowed wing keeps its standoff and its clearance across morphs."""
    from tests.test_wings import _inside_body
    base, lib = mh.load()
    for i, v in enumerate(_morphed_bodies(base, lib)):
        cv, cq = base.body_cage(v)
        tris = cv[np.vstack([cq[:, [0, 1, 2]], cq[:, [0, 2, 3]]])]
        for name, pw in A.back_wings("feathered").build_rigs(base, v):
            sign = 1.0 if name.endswith("l") else -1.0
            w = pw.pose(fold=1.0).vertices
            assert (sign * w[:, 0]).min() > 0.1, (i, name, "touches the spine")
            assert _inside_body(w[::12], tris).mean() < 0.12, (i, name, "buried")
            # they meet in a V: the tips are nearer the spine than the shoulders
            low, high = w[w[:, 1] < w[:, 1].min() + 1.0], w[w[:, 1] > w[:, 1].max() - 1.5]
            assert (sign * low[:, 0]).mean() < (sign * high[:, 0]).mean() - 0.4, (i, name)


@pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
def test_the_feathered_wings_frame_does_not_depend_on_body_morphs():
    """The root is relative to the anchor, so the whole frame (span axis AND chord) is the same on
    every body; an absolute root would tilt the plane by the difference in back depth."""
    base, lib = mh.load()
    frames = []
    for v in _morphed_bodies(base, lib):
        (_, pw), _ = A.back_wings("feathered").build_rigs(base, v)
        frames.append(pw.matrix[:3, :3] / np.linalg.norm(pw.matrix[:3, 0]))
    worst = max(np.degrees(np.arccos(np.clip(f[:, k] @ g[:, k], -1, 1)))
                for f in frames for g in frames for k in (0, 2))
    assert worst < 1.0, f"the wing frame moves {worst:.2f} degrees between bodies"


@pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
def test_a_relative_root_follows_the_anchor_and_an_absolute_one_does_not():
    base, lib = mh.load()
    plain, _, heavy, slim = _morphed_bodies(base, lib)

    def root_world(app, v):
        (_, pw), _ = app.build_rigs(base, v)
        r = pw.rig.skeleton.joints["root"]
        return pw.matrix[:3, :3] @ r + pw.matrix[:3, 3]

    rel = A.back_wings("feathered")
    assert rel.root_relative
    assert np.linalg.norm(root_world(rel, slim) - root_world(rel, heavy)) > 0.3
    absolute = A.back_wings("membrane")
    assert not absolute.root_relative
    for v in (slim, heavy):
        # the membrane's root stays where it was put (joints are lifted a little off the plane)
        assert np.linalg.norm(root_world(absolute, v) - np.asarray(A.WING_ROOT)) < 0.15


def test_primaries_close_their_fan_progressively_as_the_wing_folds(feathered):
    """The primaries close up toward the hand on the way to the stow -- the fan is mostly shut
    by half way -- instead of staying open until the end (or snapping shut)."""
    sk = feathered.skeleton

    def spread(rig, f):
        posed = rig.posed_joints(rig.pose_from_controls(fold=f))
        d = {q["root"]: posed[q["tip"]] - posed[q["root"]] for q in rig.skeleton.feathers
             if q["row"] == "primary"}
        a, b = d["primary01_r"], d["primary10_r"]
        return np.degrees(np.arccos(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))

    s = [spread(feathered, f) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    # never reopens (0.1 deg of slack: the stow taper turns the edge feathers a few degrees
    # toward the middle of the bundle, which nudges the P1-P10 angle by ~0.03 deg)
    assert all(a >= b - 0.1 for a, b in zip(s, s[1:])), s
    assert s[2] < 0.45 * s[0], f"half folded the fan is still {s[2]:.0f} of {s[0]:.0f} degrees"
    assert s[4] < 8.0
    # a feather that simply chased the stowed heading in a straight line was ~half open here
    lin = W.make_wing_rig("feathered")
    lin.skeleton.stow_follow = 0.0
    assert spread(lin, 0.5) > 0.45 * s[0]


def test_feathers_follow_their_bone_until_late_in_the_fold(feathered):
    """At fold 0 and fold 1 following changes nothing; half way, the secondaries lie along the
    forearm they grow from (they are perpendicular to it) instead of having already swung to
    the stowed heading."""
    free = W.make_wing_rig("feathered")
    free.skeleton.stow_follow = 0.0

    def heading(rig, name, f):
        J = rig.posed_joints(rig.pose_from_controls(fold=f))
        d = J[name + "_t"] - J[name + "_r"]
        return np.degrees(np.arctan2(d[2], d[0]))

    for f in (0.0, 1.0):
        assert heading(feathered, "secondary07", f) == pytest.approx(heading(free, "secondary07", f), abs=1e-6)
    J = feathered.posed_joints(feathered.pose_from_controls(fold=0.5))
    fore = J["wrist"] - J["elbow"]
    t = J["secondary07_t"] - J["secondary07_r"]
    cos = lambda a, b: abs(float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))  # noqa: E731
    Jf = free.posed_joints(free.pose_from_controls(fold=0.5))
    tf = Jf["secondary07_t"] - Jf["secondary07_r"]
    forf = Jf["wrist"] - Jf["elbow"]
    assert cos(fore, t) < cos(forf, tf) - 0.1, (cos(fore, t), cos(forf, tf))


@pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
def test_the_stowed_wing_tapers_toward_its_tips():
    """Broad covert mass at the shoulder, narrow primaries at the hem: the edge feathers turn
    toward the middle, so the stowed bundle is narrower at the tips than parallel feathers
    would leave it."""
    base, lib, v, rigs = _feathered_on_body()
    (_, pw), _ = rigs

    def widths():
        w = pw.pose(fold=1.0).vertices
        y0, h = w[:, 1].min(), np.ptp(w[:, 1])
        return (np.ptp(w[w[:, 1] < y0 + 0.2 * h][:, 0]),
                np.ptp(w[(w[:, 1] > y0 + 0.4 * h) & (w[:, 1] < y0 + 0.7 * h)][:, 0]))

    tip, mid = widths()
    # (0.55 before the spread plumage got slotted outer primaries, whose wider gaps widen the
    # tip band a little; the assertion below is the one that proves the taper is doing the work)
    assert tip < 0.62 * mid, (tip, mid)
    pw.rig.skeleton.stow_taper = 0.0
    parallel_tip, _ = widths()
    assert tip < 0.92 * parallel_tip, (tip, parallel_tip)


def test_a_hand_curl_cups_the_outer_primaries(feathered):
    """The curl used to be nearly invisible on the angel (the hand flexes as a unit). It now
    droops the primaries toward the underside, the outermost most."""
    def drop(rig, name):
        a = rig.posed_joints(rig.pose_from_controls(curl=60.0))[name]
        b = rig.posed_joints(rig.pose_from_controls())[name]
        return float(a[1] - b[1])                                  # +Y is down
    assert drop(feathered, "primary10_t") > 2.0
    assert abs(drop(feathered, "primary01_t")) < 0.3
    flat = W.make_wing_rig("feathered")
    flat.skeleton.curl_cup = 0.0
    assert drop(flat, "primary10_t") < 2.0
    # no curl, no cup
    assert np.allclose(feathered.pose_mesh(feathered.pose_from_controls()).vertices,
                       feathered.rest_mesh.vertices)


def test_a_raised_wing_arches_instead_of_swinging_as_one_plank(feathered):
    """The hand lags the wrist: raising the shoulder 40 degrees bends the wrist ~12 degrees (and
    lowering it bends the other way), so raise/lower is not a rigid rotation of the whole limb.
    The wingtip is where it shows: a rigid swing and the arched one are over a unit apart."""
    rigid = W.make_wing_rig("feathered")
    rigid.skeleton.raise_arch = 0.0

    def tip_gap(**kw):
        a = feathered.posed_joints(feathered.pose_from_controls(**kw))["primary10_t"]
        b = rigid.posed_joints(rigid.pose_from_controls(**kw))["primary10_t"]
        return float(np.linalg.norm(a - b))

    assert tip_gap(raise_=40.0) > 0.8
    assert tip_gap(raise_=-30.0) > 0.6
    assert tip_gap() < 1e-9                                       # at rest there is nothing to arch
    # and the hand really is bent at the wrist, out of the plane the forearm lies in
    def hand_lift(rig, **kw):
        J = rig.posed_joints(rig.pose_from_controls(**kw))
        fore = (J["wrist"] - J["elbow"]) / np.linalg.norm(J["wrist"] - J["elbow"])
        hand = (J["d3_tip"] - J["wrist"]) / np.linalg.norm(J["d3_tip"] - J["wrist"])
        return float(np.degrees(np.arcsin(np.clip((hand - (hand @ fore) * fore) @ np.array([0.0, 1.0, 0.0]), -1, 1))))
    assert abs(hand_lift(feathered, raise_=40.0) - hand_lift(rigid, raise_=40.0)) > 6.0


# -- mutation guards for the stowed-membrane mechanisms (each constant was found to be
#    changeable without a single test noticing) -----------------------------------

def _pose_verts(style, **kw):
    rig = W.make_wing_rig(style)
    return rig.pose_mesh(rig.pose_from_controls(**kw)).vertices


@pytest.mark.parametrize("style", ["membrane", "dragon"])
def test_the_cupped_relief_changes_a_curled_wing_and_fades_out_when_stowed(style):
    from minimaster.character import wingrig as R
    old = R.PLEAT_CUPPED
    try:
        curled = _pose_verts(style, curl=40.0, fold=0.2)
        stowed = _pose_verts(style, fold=1.0)
        R.PLEAT_CUPPED = 0.0
        curled0 = _pose_verts(style, curl=40.0, fold=0.2)
        stowed0 = _pose_verts(style, fold=1.0)
    finally:
        R.PLEAT_CUPPED = old
    assert np.abs(curled - curled0).max() > 0.02          # it does something to a cupped wing ...
    assert np.abs(stowed - stowed0).max() < 1e-9          # ... and nothing to a folded one


@pytest.mark.parametrize("style", ["membrane", "dragon"])
def test_the_sheet_follows_a_bent_finger_from_the_bend_threshold(style):
    from minimaster.character import wingrig as R
    old = R.BEND_FROM
    try:
        base = _pose_verts(style, fold=0.85)
        R.BEND_FROM = 110.0
        late = _pose_verts(style, fold=0.85)
    finally:
        R.BEND_FROM = old
    assert np.abs(base - late).max() > 0.05, "BEND_FROM does nothing at fold 0.85"


def test_the_knuckles_stay_straight_until_the_start_of_the_last_quarter():
    assert W.knuckle_phase(0.70) == 0.0 and W.knuckle_phase(0.75) == 0.0
    assert 0.0 < W.knuckle_phase(0.80) < W.knuckle_phase(0.90) < W.knuckle_phase(1.0) == 1.0
