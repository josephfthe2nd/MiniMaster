"""T3: harvested limbs that make sense and print, their pose rig, and the tail / horn controls.

The first half needs no MakeHuman assets: the rig and the root taper on a synthetic limb,
the control ranges, the builders (the first version's tail and horn are still bit-for-bit
what ``make_tail`` / ``make_horn`` return, the rounded tip and the new bends are new),
presets as data. The second half builds the extra arms, legs, tail and horns on the real
morphed body (the neutral one and the tall, short, child and heavy bodies the earlier
reviews used) and is skipped when the CC0 assets are not fetched.
"""

import itertools
import math
import warnings

import numpy as np
import pytest

from minimaster.__main__ import main
from minimaster.character import appendages as A
from minimaster.character import kinds as K
from minimaster.character import limbrig as L
from minimaster.character import mhbase as mh
from minimaster.character import print_report as pr
from minimaster.character import printprep as pp
from minimaster.character.sliders import Character
from minimaster.core import math3d as m3
from minimaster.core.mesh import Mesh
from minimaster.core.subdiv import quads_to_mesh

assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")

BODIES = {
    "neutral": {}, "tall": dict(height=1.0), "short": dict(height=0.0),
    "child": dict(age=0.1875), "heavy": dict(weight=1.0, gender=1.0, muscle=1.0)}
LIMB_OF = {"lower_arms": "arm", "upper_arms": "arm", "extra_legs": "leg"}


# ==============================================================================================
# a synthetic limb: a tube along shoulder -> elbow -> hand, closed, with a pole on the cut end


def _tube(joints, radius=0.5, ring_verts=12, per_bone=6):
    """A closed tube along the polyline ``joints``: ``(Mesh, root_loop, root_pole)``."""
    pts = [np.asarray(j, dtype=float) for j in joints]
    centres = []
    for a, b in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0.0, 1.0, per_bone, endpoint=False):
            centres.append(a + (b - a) * t)
    centres.append(pts[-1])
    rings = []
    for i, c in enumerate(centres):
        d = (centres[min(i + 1, len(centres) - 1)] - centres[max(i - 1, 0)])
        d = d / np.linalg.norm(d)
        basis = A._frame(d)
        rings.append(A._ring(c, radius, radius, basis, ring_verts))
    mesh = A._loft(rings)
    root_loop = np.arange(ring_verts)
    centre = mesh.vertices[root_loop].mean(axis=0)
    pole = int(np.argmin(np.linalg.norm(mesh.vertices[ring_verts * len(rings):] - centre, axis=1))
               + ring_verts * len(rings))
    return mesh, root_loop, pole


# a shoulder, elbow and hand: the forearm lies in the plane through the upper bone and the
# forward axis (the plane an elbow bends in), so a flexion turns it by exactly its angle
_UP = np.array([1.6, -1.8, 0.0]) / np.linalg.norm([1.6, -1.8, 0.0])
ARM = ((0.0, 0.0, 0.0), (1.6, -1.8, 0.0), tuple(np.array([1.6, -1.8, 0.0]) + 1.5 * _UP
                                                + np.array([0.0, 0.0, 1.6])))
LEG = ((0.0, 0.0, 0.0), (0.4, -4.0, 0.3), (0.8, -8.0, 0.0))


def _rig(joints=ARM, limb="arm", mirrored=False):
    """A rig on the synthetic limb; ``mirrored`` is the same limb on the other side of the
    body (the mesh and the markers reflected, winding flipped), as a right-hand harvest is."""
    pts = [np.asarray(j, dtype=float) for j in joints]
    mesh, loop, pole = _tube(pts)
    if mirrored:
        pts = [p * [-1.0, 1.0, 1.0] for p in pts]
        mesh = Mesh(mesh.vertices * [-1.0, 1.0, 1.0], mesh.faces[:, ::-1])
    return L.LimbRig(mesh, *pts, limb=limb, mirrored=mirrored, root_loop=loop, root_pole=pole)


def _as_set(points, decimals=9):
    return {tuple(p) for p in np.round(np.asarray(points), decimals)}


def _set_distance(a, b):
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return max(d.min(axis=1).max(), d.min(axis=0).max())


# -- ranges --------------------------------------------------------------------------------------


def test_the_ranges_are_anatomical_and_depend_on_the_other_controls():
    arm = L.typical_geometry("arm")
    lo, hi = arm.control_range("raise_")
    assert lo == pytest.approx(-arm.rest_abduction) and hi == pytest.approx(180 - arm.rest_abduction)
    # the elbow cannot straighten past straight nor bend past ~145 degrees in total
    lo, hi = arm.control_range("elbow")
    assert lo == pytest.approx(-arm.rest_flex) and hi == pytest.approx(145 - arm.rest_flex)
    # a limb held out sideways swings over a narrower arc than one hanging or held up
    hang = arm.control_range("swing", {"raise_": -arm.rest_abduction})
    out = arm.control_range("swing", {"raise_": 90 - arm.rest_abduction})
    assert hang[1] - hang[0] > out[1] - out[0] + 30
    leg = L.typical_geometry("leg")
    assert leg.control_range("twist")[1] < arm.control_range("twist")[1]      # a hip turns less
    assert leg.control_range("knee") == leg.control_range("elbow")            # one control, two names
    assert arm.control_range("wobble") is None


def test_a_rig_answers_to_the_flex_control_of_its_limb():
    arm, leg = _rig(ARM, "arm"), _rig(LEG, "leg")
    assert arm.control_names == ("swing", "raise_", "twist", "elbow")
    assert leg.control_names == ("swing", "raise_", "twist", "knee")
    assert arm.control_range("knee") is None and leg.control_range("elbow") is None
    # the other limb's name is not silently an alias: it does nothing
    assert np.array_equal(arm.pose({"knee": 60}).vertices, arm.mesh.vertices)
    assert np.array_equal(leg.pose({"elbow": 60}).vertices, leg.mesh.vertices)
    assert not np.array_equal(leg.pose({"knee": 60}).vertices, leg.mesh.vertices)


# -- the rig on a synthetic limb --------------------------------------------------------------------


def test_zero_controls_are_the_harvested_pose_exactly():
    rig = _rig()
    assert np.array_equal(rig.pose({}).vertices, rig.mesh.vertices)
    assert np.allclose(rig.pose({"elbow": 0.0, "swing": 0.0}).vertices, rig.mesh.vertices,
                       atol=1e-12)


def test_weights_are_a_partition_of_unity_with_the_cut_end_pinned():
    rig = _rig()
    w = rig.weights
    assert w.shape == (len(rig.mesh.vertices), len(rig.bone_names))
    assert (w >= 0).all() and np.allclose(w.sum(axis=1), 1.0)
    pin = rig.bone_names.index("shoulder")                   # the bone that never moves
    assert (w[rig.root_vertices, pin] == 1.0).all()
    far = np.argmax(np.linalg.norm(rig.mesh.vertices - rig.joints_rest["shoulder"], axis=1))
    assert w[far, pin] == 0.0                                # ... and the hand follows the pose


def test_the_pinned_end_never_moves_and_bones_keep_their_lengths():
    rig = _rig()
    j0 = rig.joints_rest
    lengths = (np.linalg.norm(j0["elbow"] - j0["shoulder"]), np.linalg.norm(j0["hand"] - j0["elbow"]))
    limits = {k: rig.control_range(k) for k in rig.control_names}
    for corner in itertools.product(*[limits[k] for k in rig.control_names]):
        controls = dict(zip(rig.control_names, corner))
        m = rig.pose(controls)
        rep = m.integrity_report()
        assert rep["watertight"] and rep["outward"] and np.isfinite(m.vertices).all(), controls
        assert np.array_equal(m.vertices[rig.root_vertices], rig.mesh.vertices[rig.root_vertices])
        j = rig.joints(controls)
        assert np.allclose([np.linalg.norm(j["elbow"] - j["shoulder"]),
                            np.linalg.norm(j["hand"] - j["elbow"])], lengths, atol=1e-9), controls
        assert np.allclose(j["shoulder"], j0["shoulder"])        # the pivot stays where it is
        assert 0.3 < m.volume() / rig.mesh.volume() < 1.1, (controls, m.volume())


def test_each_control_moves_the_limb_the_way_it_is_named():
    rig = _rig()
    j0 = rig.joints_rest
    up = L._unit(j0["elbow"] - j0["shoulder"])
    # raise: the upper bone turns away from the body by exactly that angle, in the coronal plane
    j = rig.joints({"raise_": 30.0})
    new_up = L._unit(j["elbow"] - j["shoulder"])
    assert math.degrees(math.acos(up @ new_up)) == pytest.approx(30.0, abs=1e-6)
    assert new_up[0] > up[0] and new_up[2] == pytest.approx(up[2], abs=1e-9)
    # swing: toward the front (+Z) for positive values, back for negative
    assert (L._unit(rig.joints({"swing": 25.0})["elbow"] - j0["shoulder"]))[2] > up[2] + 0.1
    assert (L._unit(rig.joints({"swing": -25.0})["elbow"] - j0["shoulder"]))[2] < up[2] - 0.1
    # twist: about the bone's own axis, so the elbow does not move and the hand goes round it
    j = rig.joints({"twist": 40.0})
    assert np.allclose(j["elbow"], j0["elbow"], atol=1e-12)
    r0 = (j0["hand"] - j0["shoulder"]) - ((j0["hand"] - j0["shoulder"]) @ up) * up
    r1 = (j["hand"] - j["shoulder"]) - ((j["hand"] - j["shoulder"]) @ up) * up
    assert math.degrees(math.acos(r0 @ r1 / np.linalg.norm(r0) / np.linalg.norm(r1))) \
        == pytest.approx(40.0, abs=1e-6)
    # elbow: the forearm bends further by exactly that angle, the upper bone stays
    j = rig.joints({"elbow": 35.0})
    assert np.allclose(j["elbow"], j0["elbow"], atol=1e-12)
    f0, f1 = L._unit(j0["hand"] - j0["elbow"]), L._unit(j["hand"] - j["elbow"])
    assert math.degrees(math.acos(f0 @ f1)) == pytest.approx(35.0, abs=1e-6)
    # ... toward the front (an elbow flexes forward) and the further it is bent the nearer the
    # hand comes to the shoulder
    assert f1[2] > f0[2]
    assert np.linalg.norm(j["hand"] - j["shoulder"]) < np.linalg.norm(j0["hand"] - j0["shoulder"])
    # the leg's knee flexes the other way: the shank goes BACK
    leg = _rig(LEG, "leg")
    g0 = L._unit(leg.joints_rest["hand"] - leg.joints_rest["elbow"])
    jl = leg.joints({"knee": 40.0})
    assert L._unit(jl["hand"] - jl["elbow"])[2] < g0[2] - 0.1


def test_values_outside_the_range_are_held_at_the_range():
    rig = _rig()
    lo, hi = rig.control_range("elbow")
    assert np.array_equal(rig.pose({"elbow": hi + 500}).vertices, rig.pose({"elbow": hi}).vertices)
    assert np.array_equal(rig.pose({"elbow": lo - 500}).vertices, rig.pose({"elbow": lo}).vertices)
    r_lo, _ = rig.control_range("raise_")
    assert np.array_equal(rig.pose({"raise_": r_lo - 90}).vertices,
                          rig.pose({"raise_": r_lo}).vertices)


@pytest.mark.parametrize("controls", [
    {"raise_": 40.0}, {"swing": 30.0, "twist": 25.0}, {"elbow": 50.0, "raise_": -20.0},
    {"raise_": 60.0, "swing": -15.0, "twist": -30.0, "elbow": 70.0}])
def test_the_right_limb_is_the_exact_mirror_of_the_left_one(controls):
    left, right = _rig(mirrored=False), _rig(mirrored=True)
    a = left.pose(controls).vertices * [-1.0, 1.0, 1.0]
    b = right.pose(controls).vertices
    assert np.array_equal(a.shape, b.shape)
    assert np.abs(a - b).max() < 1e-12                       # vertex for vertex: it IS the mirror
    assert right.pose(controls).integrity_report()["outward"]     # and the winding came with it


# -- the root taper ------------------------------------------------------------------------------


def _rim_radius(mesh, loop):
    r = mesh.vertices[loop]
    return float(np.linalg.norm(r - r.mean(axis=0), axis=1).mean())


def test_the_root_taper_narrows_the_cut_end_and_nothing_beyond_its_length():
    mesh, loop, pole = _tube(ARM)
    s, e = (np.asarray(p, dtype=float) for p in ARM[:2])
    kw = dict(root_loop=loop, root_pole=pole)
    assert np.array_equal(L.taper_root(mesh, s, e, fraction=1.0, **kw).vertices, mesh.vertices)
    half = L.taper_root(mesh, s, e, fraction=0.5, length=1.5, **kw)
    third = L.taper_root(mesh, s, e, fraction=0.3, length=1.5, **kw)
    assert _rim_radius(half, loop) == pytest.approx(0.5 * _rim_radius(mesh, loop), rel=1e-6)
    assert _rim_radius(third, loop) == pytest.approx(0.3 * _rim_radius(mesh, loop), rel=1e-6)
    assert third.volume() < half.volume() < mesh.volume()
    for m in (half, third):
        rep = m.integrity_report()
        assert rep["watertight"] and rep["outward"]
    along = (mesh.vertices - s) @ L._unit(e - s)
    far = along > 1.5 + 1e-9
    assert far.any() and np.array_equal(half.vertices[far], mesh.vertices[far])   # beyond the neck


# ==============================================================================================
# builders and kinds without the body


def test_the_first_versions_tail_and_horn_are_unchanged():
    """``tip_radius=None`` (the builders' default) is the pointed shape the T1 tests pin: the
    new keyword arguments add a rounded tip and more bends, they do not move a vertex of the
    old default. Frozen values from the base commit."""
    tail, horn = A.make_tail(), A.make_horn()
    assert tail.vertices.shape == (132, 3) and tail.volume() == pytest.approx(1.2544715928, rel=1e-9)
    assert np.allclose(tail.vertices[[0, 50, 120]], [
        [0.0, -0.47, 0.0], [0.4332730717, -0.2049451339, 2.4495217062],
        [1.2239528097, -2.9232845005, 2.9197310618]], atol=1e-9)
    assert horn.vertices.shape == (74, 3) and horn.volume() == pytest.approx(0.3149454097, rel=1e-9)
    assert np.allclose(horn.vertices[[0, 40, 70]], [
        [0.0, -0.432, 0.0], [0.1352504107, -0.2049052884, 1.3465624161],
        [0.0, -0.7910099354, 1.8942659935]], atol=1e-9)
    t2 = A.make_tail(length=9.0, curl=1.2, droop=-0.4, segments=20, ring_verts=14)
    assert t2.vertices.shape == (296, 3) and t2.volume() == pytest.approx(1.9958503708, rel=1e-9)
    assert np.allclose(t2.vertices[[100, 250]], [[0.2010894640, 0.3113709388, 3.2036289959],
                                                 [3.4837800494, 1.7805516944, 3.1839850666]],
                       atol=1e-9)
    # an explicit zero of every new knob is the same mesh
    assert np.array_equal(A.make_tail(lift=0.0, sway=0.0).vertices, tail.vertices)
    assert np.array_equal(A.make_horn(turns=0.0).vertices, horn.vertices)
    assert not np.array_equal(A.make_tail(tip_radius=0.09).vertices[:100], tail.vertices[:100])


def _centreline(mesh, rings, ring_verts):
    return np.array([mesh.vertices[k * ring_verts:(k + 1) * ring_verts].mean(axis=0)
                     for k in range(rings)])


def test_lift_and_sway_bend_the_tail_in_the_vertical_and_the_lateral_plane():
    rest = _centreline(A.make_tail(curl=0.0, droop=0.0), 13, 10)
    assert np.allclose(rest[:, :2], 0.0)                         # a straight tail along +Z
    up = _centreline(A.make_tail(curl=0.0, droop=0.0, lift=40.0), 13, 10)
    down = _centreline(A.make_tail(curl=0.0, droop=0.0, lift=-40.0), 13, 10)
    left = _centreline(A.make_tail(curl=0.0, droop=0.0, sway=40.0), 13, 10)
    right = _centreline(A.make_tail(curl=0.0, droop=0.0, sway=-40.0), 13, 10)
    assert up[-1, 0] > 0.5 > -0.5 > down[-1, 0] and np.allclose(up[:, 1], 0.0)
    assert left[-1, 1] > 0.5 > -0.5 > right[-1, 1] and np.allclose(left[:, 0], 0.0)
    # the total turn is what was asked for: 40 degrees over the 13 rings (each one a turn of
    # 40/13), so the last of the 12 segments points 11 turns off the first
    seg = np.diff(up, axis=0)
    first, last = seg[0] / np.linalg.norm(seg[0]), seg[-1] / np.linalg.norm(seg[-1])
    assert math.degrees(math.acos(first @ last)) == pytest.approx(40.0 * 11 / 13, abs=1e-6)
    # bends add: lift and sway together are two planes at once
    both = _centreline(A.make_tail(curl=0.0, droop=0.0, lift=30.0, sway=30.0), 13, 10)
    assert both[-1, 0] > 0.3 and both[-1, 1] > 0.3


def test_a_rounded_tip_ends_at_the_tip_radius_and_closes_the_shell():
    tip, length = 0.1, 3.0
    for builder, kw in ((A.make_tail, dict(curl=0.0, droop=0.0)), (A.make_horn, dict(curve=0.0))):
        mesh = builder(length=length, tip_radius=tip, **kw)
        rep = mesh.integrity_report()
        assert rep["watertight"] and rep["outward"] and rep["volume"] > 0
        # a straight tube: the dome adds ``tip`` beyond the last ring; the end is round, not
        # a disc or a needle
        assert mesh.vertices[:, 2].max() == pytest.approx(length + tip, abs=0.02)
        end = mesh.vertices[mesh.vertices[:, 2] > length - 1e-9]
        assert np.linalg.norm(end[:, :2], axis=1).max() == pytest.approx(tip, rel=0.02)
        dome = mesh.vertices[mesh.vertices[:, 2] > length + 0.5 * tip]
        assert 0 < np.linalg.norm(dome[:, :2], axis=1).max() < 0.9 * tip      # narrowing, a cap
        # and the legacy point is still a point
        pointed = builder(length=length, **kw)
        pole = pointed.vertices[:, 2].max()
        assert np.linalg.norm(pointed.vertices[pointed.vertices[:, 2] > length - 1e-9][:, :2],
                              axis=1).max() < 0.05 and pole == pytest.approx(length, abs=0.02)


@pytest.mark.parametrize("kind,builder,rv,nseg", [("tail", A.make_tail, 10, 12),
                                                  ("horn", A.make_horn, 8, 8)])
def test_the_report_tube_law_follows_the_rounded_tip(kind, builder, rv, nseg):
    """print_report derives a tail's tip from its radius law; with ``tip_radius`` the law is
    ``tip + (base - tip)(1 - t)^e``. Ring k of the loft is station k / segments."""
    params = {"base_radius": 0.5, "tip_radius": 0.12}
    mesh = builder(**params)
    t = np.linspace(0.0, 1.0, 2001)
    d = pr.tube_diameters(kind, params)
    for k in range(nseg + 1):
        ring = mesh.vertices[k * rv:(k + 1) * rv]
        h = rv // 2
        diameter = float(min(np.linalg.norm(ring[i] - ring[i + h]) for i in range(h)))
        assert diameter == pytest.approx(float(np.interp(k / nseg, t, d)), rel=0.04), (kind, k)
    assert d[0] == pytest.approx(2 * (0.92 if kind == "tail" else 1.0) * 0.5)    # base unchanged
    assert d[-1] == pytest.approx(2 * (0.92 if kind == "tail" else 1.0) * 0.12)  # the tip, not ~0


def _ring_twist(mesh, rings, rv):
    """The angle (degrees) each ring is turned about the path relative to the one before it:
    the angle between vertex 0's offset from the axis in consecutive rings."""
    V = mesh.vertices
    out = []
    for k in range(rings - 1):
        r0, r1 = V[k * rv:(k + 1) * rv], V[(k + 1) * rv:(k + 2) * rv]
        c0, c1 = r0.mean(axis=0), r1.mean(axis=0)
        d = (c1 - c0) / np.linalg.norm(c1 - c0)
        a, b = r0[0] - c0, r1[0] - c1
        a, b = a - (a @ d) * d, b - (b @ d) * d
        out.append(math.degrees(math.acos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b),
                                                  -1, 1))))
    return out


def test_the_rounded_tube_carries_its_rings_along_the_path_without_a_twist_band():
    """The first version's ring frame switches its reference axis at a threshold, so a bent
    tube twists its rings by 90-160 degrees in one segment (a skewed band; on a lifted tail
    the thickness probe read 0.02 mm there). A number for ``tip_radius`` carries the rings
    along by parallel transport instead; ``None`` stays bit-for-bit the first version."""
    for kw in ({}, {"lift": 25.0}, {"lift": 25.0, "curl": 1.0}, {"sway": -60.0}):
        new = A.make_tail(tip_radius=0.09, **kw)
        old = A.make_tail(**kw)
        assert max(_ring_twist(new, 13, 10)[:-1]) < 8.0, kw
        assert max(_ring_twist(old, 13, 10)) > 90.0, kw
    for kw in (dict(), dict(curve=2.1, twist=0.6), dict(curve=0.7, turns=3.0)):
        horn = A.make_horn(length=4.0, segments=24, ring_verts=12, tip_radius=0.08, **kw)
        assert max(_ring_twist(horn, 25, 12)[:-1]) < 12.0, kw


def test_a_horn_turns_into_a_helix_and_otherwise_stays_in_a_plane():
    flat = _centreline(A.make_horn(length=5.0, curve=1.0, segments=24, ring_verts=8), 25, 8)
    helix = _centreline(A.make_horn(length=5.0, curve=1.0, turns=2.0, segments=24, ring_verts=8),
                        25, 8)
    off_plane = lambda c: np.linalg.svd(c - c.mean(axis=0))[1][2]                   # noqa: E731
    assert off_plane(flat) < 1e-9 < 0.05 < off_plane(helix)
    # the bend of a helix is about the same all along (the turning per step stays within 25%:
    # the bend is applied about fixed axes, so it is not exactly constant); a spiral's grows
    # eight-fold from the base to the tip
    def turn(c):
        d = np.diff(c, axis=0)
        d = d / np.linalg.norm(d, axis=1)[:, None]
        return np.array([math.acos(np.clip(u @ w, -1, 1)) for u, w in zip(d[:-1], d[1:])])

    th, tf = turn(helix), turn(flat)
    assert th[2:].min() / th[2:].max() > 0.7
    assert tf[-1] / tf[2] > 6.0


# -- the registry: horn shapes, tail controls, presets as data -----------------------------------


def test_the_horn_shapes_have_their_own_defaults_through_when():
    horn = K.KINDS["horn"]
    assert horn.param("shape").choices == ("curved", "ram", "spiral", "antelope")
    base = horn.defaults()
    assert base["shape"] == "curved" and (base["length"], base["curve"], base["segments"]) == (2.2, 1.1, 8)
    seen = set()
    for shape in horn.param("shape").choices:
        d = horn.defaults(shape=shape)
        assert d["shape"] == shape and d["tip_radius"] == K.HORN_TIP
        seen.add((d["length"], d["base_radius"], d["curve"], d["twist"], d["turns"]))
        # every shape's own limits hold: a value is clamped to the range of its shape
        values, notes = horn.clean({"shape": shape, "length": 99.0})
        assert values["length"] == 12.0 and any("outside" in n for n in notes)
    assert len(seen) == 4                                         # four different horns
    assert horn.defaults(shape="spiral")["turns"] == 3.0 and horn.defaults()["turns"] == 0.0
    bad, notes = horn.clean({"shape": "unicorn"})
    assert bad["shape"] == "curved" and any("one of" in n for n in notes)


def test_the_tail_controls_are_added_to_the_shape_and_their_ranges_follow_it():
    tail = K.KINDS["tail"]
    assert [c.key for c in tail.controls] == ["curl", "droop", "lift", "sway"]
    assert tail.control_range("curl", {}, {"curl": 0.55}) == pytest.approx((-2.55, 1.45))
    assert tail.control_range("droop", {}, {"droop": -1.0}) == pytest.approx((-1.0, 3.0))
    assert tail.control_range("lift") == (-60.0, 90.0) and tail.control_range("sway") == (-90.0, 90.0)
    assert tail.control_range("fold") is None
    args = tail.builder_args(tail.defaults(), {"curl": 0.5, "droop": -0.2, "lift": 12.0})
    assert args["curl"] == pytest.approx(1.05) and args["droop"] == pytest.approx(0.7)
    assert args["lift"] == 12.0 and args["sway"] == 0.0 and args["tip_radius"] == K.TAIL_TIP
    assert tail.builder_args({**tail.defaults(), "tip_radius": 0.0}, {})["tip_radius"] is None


def test_a_harvest_record_starts_in_its_presets_pose_and_ids_name_the_limb():
    arm = K.new_spec("harvest")
    assert (arm.id, arm.placement["preset"], arm.params) == ("arm-1", "lower_arms", {})
    assert arm.controls == K.KINDS["harvest"].presets["lower_arms"]["controls"]
    arm.controls["elbow"] = 99.0                                       # a copy, not the preset
    assert K.KINDS["harvest"].presets["lower_arms"]["controls"]["elbow"] != 99.0
    assert K.new_spec("harvest", [arm]).id == "arm-2"
    leg = K.new_spec("harvest", params={"limb": "leg"})
    assert (leg.id, leg.placement["preset"]) == ("leg-1", "extra_legs")
    assert K.new_spec("harvest", preset="upper_arms").controls["raise_"] > 45
    assert K.new_spec("harvest", controls={}).controls == {}           # "no controls" is a choice
    assert K.new_spec("tail").controls == {}                           # kinds without start poses
    for name, preset in K.KINDS["harvest"].presets.items():            # every preset names a face
        assert preset["face"] in K.PRESET_FACES, name


def test_the_old_harvest_presets_are_still_known():
    harvest = K.KINDS["harvest"]
    assert {"shoulder_socket", "hip_socket"} <= set(harvest.presets)
    assert "controls" not in harvest.presets["shoulder_socket"]        # the old straight copy
    assert set(harvest.presets) >= {"lower_arms", "upper_arms", "extra_legs"}


# ==============================================================================================
# on the real body


@pytest.fixture(scope="module")
def world():
    base, lib = mh.load()
    return base, lib, lib.apply(base.verts, mh.macro_weights())


@pytest.fixture(scope="module")
def bodies(world):
    base, lib, _ = world
    return {name: lib.apply(base.verts, mh.macro_weights(**kw)) for name, kw in BODIES.items()}


def _body_solid(base, v):
    return pr.Solid(quads_to_mesh(*base.body_cage(v)))


def _original_limb(base, v, limb):
    """The character's own limb as a closed solid, where it is (what a copy must not overlap)."""
    radius = K.KINDS["harvest"].param("radius", {"limb": limb}).default
    return pr.Solid(A.harvest_region(base, list(K.LIMB_CHAINS[limb]), v, radius=radius, trim=0.0))


def _harvest(base, v, preset, **spec_kw):
    spec = K.new_spec("harvest", preset=preset, base=base, **spec_kw)
    return K.instantiate_all([spec], base, v)


@assets
@pytest.mark.parametrize("bad", [-1.0, 0.0])
def test_a_harvest_scale_that_is_not_positive_is_skipped_not_inverted(world, bad):
    base, _, v0 = world
    spec = K.new_spec("harvest", base=base)
    spec.placement["scale"] = bad
    with pytest.warns(K.AppendageWarning, match="must be positive"):
        assert K.instantiate_all([spec, K.new_spec("horn", base=base)], base, v0)[0].name == "horn-1.l"


# -- (1) limbs that sit on the body, are welded into it and stay clear of the original --------------


@assets
@pytest.mark.parametrize("preset", ["lower_arms", "upper_arms", "extra_legs"])
def test_every_preset_buries_its_cut_end_in_the_torso_on_every_body(world, bodies, preset):
    """The cut end of the copy (its rim, a ragged ring) lies INSIDE the body, so what a
    slicer unions is a limb growing out of the torso and not a flat cut stuck on its side;
    and the share of the limb inside the body passes the printability rule (>= 8%: no
    root-weld finding) on the neutral, tall, short, child and heavy bodies."""
    base, _, _ = world
    limb = LIMB_OF[preset]
    for name, v in bodies.items():
        solid = _body_solid(base, v)
        for p in _harvest(base, v, preset)[:2]:
            m = p.mesh()
            rim = m.vertices[p.limb.root_vertices]
            assert solid.contains(rim).mean() >= 0.95, (preset, name, p.name)
            embedded = pr.embedded_volume_share(m, solid, 8192)
            assert 0.12 <= embedded <= 0.45, (preset, name, p.name, embedded)
            # the weld is into the TORSO: the copy does not count the original limb it sits next to
            overlap = pr.embedded_volume_share(m, _original_limb(base, v, limb), 8192)
            assert overlap <= (0.22 if preset == "upper_arms" else 0.03), (preset, name, overlap)


@assets
def test_the_old_placement_was_welded_to_the_original_limb_not_to_the_torso(world):
    """What this replaces: the first version moved the copy only ~1 unit, so it lay beside the
    original arm; the 8-13% it 'embedded' was mostly that arm (the weld to the TORSO was
    ~1.5%), and its cut end hung outside the body. The lower arms touch the original arm
    nowhere and are 20% buried in the torso with the cut end inside."""
    base, _, v0 = world
    spec = K.new_spec("harvest", preset="shoulder_socket", side="left", base=base, controls={},
                      params={"neck": 1.0})
    old = K.instantiate_all([spec], base, v0)[0]
    arm = _original_limb(base, v0, "arm")
    solid = _body_solid(base, v0)
    body_old = pr.embedded_volume_share(old.mesh(), solid, 16384)
    arm_old = pr.embedded_volume_share(old.mesh(), arm, 16384)
    assert body_old > 0.08 and arm_old > 0.07 and body_old - arm_old < 0.04     # "welded" to the arm
    assert solid.contains(old.mesh().vertices[old.limb.root_vertices]).mean() < 0.7   # cut end hung out
    new = _harvest(base, v0, "lower_arms", side="left")[0]
    body_new = pr.embedded_volume_share(new.mesh(), solid, 16384)
    arm_new = pr.embedded_volume_share(new.mesh(), arm, 16384)
    assert arm_new < 0.01 and body_new - arm_new > 0.15                        # welded to the torso
    assert solid.contains(new.mesh().vertices[new.limb.root_vertices]).mean() == 1.0


@assets
def test_a_limb_is_cut_whole_and_follows_the_body_whatever_its_size(world, bodies):
    """The same record on every body: lengths (radius, trim, the sink) are the neutral
    limb's and scale with the character's own limb, so a child's copy is the child's limb with
    the hand on it, welded as deep as the neutral one."""
    base, _, _ = world
    depth = []
    for name, v in bodies.items():
        p = _harvest(base, v, "lower_arms", side="left")[0]
        rig = p.limb
        s, e, h = (rig.joints_rest[k] for k in ("shoulder", "elbow", "hand"))
        reach = np.linalg.norm(rig.mesh.vertices - s, axis=1).max()
        assert reach > 0.9 * np.linalg.norm(h - s), name              # the hand is on it
        assert len(rig.weights) == len(rig.mesh.vertices) and np.allclose(rig.weights.sum(1), 1.0)
        solid = _body_solid(base, v)
        depth.append(pr.embedded_volume_share(rig.mesh, solid, 8192))
    assert max(depth) - min(depth) < 0.15 and min(depth) > 0.15        # not 20% on one, 2% on another


@assets
def test_the_weights_are_recomputed_for_every_re_harvest(world, bodies):
    """The cut differs from body to body (the first version's faces: 1276 tall, 1632 heavy,
    2004 neutral); the rig is built from the geometry it is given every time, so the weights
    always fit it, and a pose of any body works."""
    base, _, _ = world
    counts = set()
    for name, v in bodies.items():
        p = _harvest(base, v, "lower_arms", side="left")[0]
        counts.add(len(p.limb.mesh.vertices))
        assert p.limb.weights.shape == (len(p.limb.mesh.vertices), 3)
        m = p.mesh({"elbow": 40.0})
        assert m.integrity_report()["watertight"] and m.vertices.shape == p.limb.mesh.vertices.shape
    assert len(counts) >= 3                                          # the topology IS body-dependent
    # a body slider move re-harvests: another cut, other weights, same record
    child = bodies["child"]
    a = _harvest(base, bodies["neutral"], "lower_arms", side="left")[0].limb
    b = _harvest(base, child, "lower_arms", side="left")[0].limb
    assert a.weights.shape != b.weights.shape or not np.allclose(a.weights, b.weights)


# -- (2) pose: ranges, watertight, pinned, mirrored, from the document and the CLI ----------------


@assets
@pytest.mark.parametrize("preset", ["lower_arms", "upper_arms", "extra_legs"])
def test_every_control_at_its_limits_is_a_watertight_pinned_limb(world, bodies, preset):
    base, _, _ = world
    for name in ("neutral", "tall", "child"):
        v = bodies[name]
        for p in _harvest(base, v, preset)[:2]:
            rig = p.limb
            names = p.control_names
            ranges = {k: p.control_range(k) for k in names}
            assert all(r is not None and r[0] < r[1] for r in ranges.values())
            j0 = rig.joints_rest
            lengths = [np.linalg.norm(j0["elbow"] - j0["shoulder"]),
                       np.linalg.norm(j0["hand"] - j0["elbow"])]
            for corner in itertools.product(*[ranges[k] for k in names]):
                controls = dict(zip(names, corner))
                m = p.mesh(controls)
                rep = m.integrity_report()
                assert rep["watertight"] and rep["outward"], (preset, name, controls)
                assert np.isfinite(m.vertices).all()
                assert np.array_equal(m.vertices[rig.root_vertices],
                                      rig.mesh.vertices[rig.root_vertices])      # the weld stays put
                j = rig.joints(controls)
                assert np.allclose([np.linalg.norm(j["elbow"] - j["shoulder"]),
                                    np.linalg.norm(j["hand"] - j["elbow"])], lengths, atol=1e-9)
                assert 0.25 < m.volume() / rig.mesh.volume() < 1.1


@assets
@pytest.mark.parametrize("preset", ["lower_arms", "upper_arms", "extra_legs"])
@pytest.mark.parametrize("controls", [
    {}, {"raise_": 30.0}, {"swing": 40.0, "twist": -30.0}, {"elbow": 60.0, "knee": 60.0},
    {"raise_": 20.0, "swing": 15.0, "twist": 25.0, "elbow": 45.0, "knee": 45.0}])
def test_a_harvested_pair_is_an_exact_mirror_in_every_pose(world, preset, controls):
    """The right limb is cut from the right arm (leg) with the l-/r- markers swapped and
    posed with the mirrored rotations: on the symmetric body it is the left one's mirror image
    to 1e-9, for the preset pose and for any control on top of it. (The first version's
    mirrored-leg harvest raised KeyError on 21 of the 55 markers.)"""
    base, _, v0 = world
    left, right = _harvest(base, v0, preset)[:2]
    assert (left.side, right.side) == ("l", "r")
    own = {k: val for k, val in controls.items() if k in left.control_names}
    mine = {**left.controls, **own}
    a = left.mesh(mine).vertices * [-1.0, 1.0, 1.0]
    b = right.mesh(mine).vertices
    assert a.shape == b.shape
    assert _set_distance(a, b) < 1e-9
    assert left.mesh(mine).volume() == pytest.approx(right.mesh(mine).volume(), rel=1e-9)
    assert np.allclose(left.anchor * [-1.0, 1.0, 1.0], right.anchor, atol=1e-9)


@assets
@pytest.mark.parametrize("preset", ["lower_arms", "extra_legs"])
def test_a_lone_right_limb_is_the_right_limb_of_the_pair_bit_for_bit(world, preset):
    base, _, v0 = world
    pair = _harvest(base, v0, preset)
    right = _harvest(base, v0, preset, side="right")
    left = _harvest(base, v0, preset, side="left")
    assert [p.name for p in right] == [pair[1].name] and [p.name for p in left] == [pair[0].name]
    assert np.array_equal(right[0].mesh().vertices, pair[1].mesh().vertices)
    assert np.array_equal(right[0].mesh().faces, pair[1].mesh().faces)
    assert np.array_equal(left[0].mesh().vertices, pair[0].mesh().vertices)
    pose = {"elbow": 33.0, "raise_": 12.0} if preset == "lower_arms" else {"knee": 33.0}
    assert np.array_equal(right[0].mesh(pose).vertices, pair[1].mesh(pose).vertices)


@assets
def test_an_asymmetric_body_keeps_each_sides_own_limb(world):
    """A sided slider makes the left arm different: the right copy is then cut from the right
    arm, not reflected from the left one (the mirror reuse is only for a symmetric body)."""
    base, lib, _ = world
    names = [n for n in lib.names if "l-arm" in n or "l-upperarm" in n or "l-hand" in n]
    if not names:
        pytest.skip("no sided arm target in this library")
    w = mh.macro_weights()
    w[names[0]] = 1.0
    v = lib.apply(base.verts, w)
    left, right = _harvest(base, v, "lower_arms")[:2]
    a = left.limb.mesh.vertices * [-1.0, 1.0, 1.0]
    assert not (a.shape == right.limb.mesh.vertices.shape and _set_distance(a, right.limb.mesh.vertices) < 1e-6)


@assets
def test_a_pose_is_reachable_from_the_document_and_saved_with_it(world, tmp_path):
    base, lib, v0 = world
    doc = Character()
    spec = K.new_spec("harvest", doc.appendages, base=base, spec_id="arm-1")
    doc.appendages.append(spec)
    K.apply_pose(*K.pose_from_text("arm-1:elbow=60,raise_=25", doc.appendages))
    K.apply_pose(*K.pose_from_text("arm-1.r:twist=-20", doc.appendages))
    assert doc.appendages[0].controls["*"] == {"elbow": 60.0, "raise_": 25.0}
    path = tmp_path / "c.mmchar"
    doc.save(path)
    again = Character.load(path)
    assert again.appendages[0].controls == doc.appendages[0].controls
    left, right = again.instantiate_appendages(base, v0, lib)[:2]
    assert left.controls["elbow"] == 60 and right.controls["twist"] == -20 and "twist" not in left.controls
    # the pose is in the mesh: more flexion than the start pose brings the hand to the shoulder
    first = K.instantiate_all([K.new_spec("harvest", base=base)], base, v0)[0]
    assert np.linalg.norm(left.limb.joints(left.controls)["hand"] - left.limb.joints_rest["shoulder"]) \
        < np.linalg.norm(first.limb.joints(first.controls)["hand"] - first.limb.joints_rest["shoulder"])
    lo, hi = left.control_range("elbow")
    assert lo < 0 < hi and left.control_range("knee") is None and left.control_range("fold") is None
    # a range that depends on another control: swing narrows as the arm is raised to the side
    w_hang = left.control_range("swing", {"raise_": lo and left.control_range("raise_")[0]})
    w_out = left.control_range("swing", {"raise_": 50.0})
    assert w_hang[1] - w_hang[0] > w_out[1] - w_out[0]


@assets
def test_bad_controls_are_named_not_hidden(world):
    base, _, v0 = world
    spec = K.new_spec("harvest", base=base, side="left", controls={})
    spec.controls.update({"elbow": 400.0, "knee": 10.0, "wobble": 1.0, "twist": float("nan")})
    with pytest.warns(K.AppendageWarning) as caught:
        p = K.instantiate_all([spec], base, v0)[0]
    text = " | ".join(str(w.message) for w in caught)
    assert "elbow = 400 is outside" in text and "using 10" in text and \
        "'knee' does not apply to an arm" in text and "wobble" in text and "twist" in text
    assert p.controls == {"elbow": 400.0}                         # what could be used; the record keeps all
    assert np.array_equal(p.mesh().vertices, p.mesh({"elbow": p.control_range("elbow")[1]}).vertices)
    assert spec.controls["wobble"] == 1.0                         # nothing is deleted


@assets
def test_a_legacy_harvest_record_still_builds(world):
    """A document written with the first version's presets (a straight copy at the shoulder
    socket, no rig) loads and builds; nothing about it is rewritten."""
    base, lib, v0 = world
    doc = Character.from_dict({"appendages": [{
        "id": "harvest-1", "kind": "harvest", "side": "both", "params": {"limb": "arm"},
        "placement": {"preset": "shoulder_socket", "anchor": {"face": 7954}, "offset": -0.5}}]})
    placed = doc.instantiate_appendages(base, v0, lib)
    assert [p.name for p in placed] == ["harvest-1.l", "harvest-1.r"]
    assert all(p.mesh().integrity_report()["watertight"] for p in placed)


# -- (3) tail and horn: controls, tip, shapes, and the report -------------------------------------


@assets
def test_the_tail_controls_re_evaluate_the_loft_where_the_tail_is_placed(world):
    base, _, v0 = world
    spec = K.new_spec("tail", base=base)
    p = K.instantiate_all([spec], base, v0)[0]
    rest = p.mesh()
    assert np.array_equal(rest.vertices, p.static.vertices) and p.control_names == (
        "curl", "droop", "lift", "sway")
    tip = lambda m: m.vertices[np.argmax(np.linalg.norm(m.vertices - p.anchor, axis=1))]   # noqa: E731
    base_tip = tip(rest)
    assert tip(p.mesh({"lift": 40.0}))[1] > base_tip[1] + 0.4          # up is +Y (MakeHuman)
    assert tip(p.mesh({"lift": -40.0}))[1] < base_tip[1] - 0.4
    assert tip(p.mesh({"sway": 40.0}))[0] > base_tip[0] + 0.3          # toward +X
    assert tip(p.mesh({"sway": -40.0}))[0] < base_tip[0] - 0.3
    assert tip(p.mesh({"droop": 0.5}))[1] < base_tip[1] - 0.3          # droop pulls the end down
    assert tip(p.mesh({"curl": 0.5}))[1] > base_tip[1] + 0.1           # curl rolls it up
    # the base stays where it is bound: only the end moves
    base_ring = lambda m: m.vertices[:10].mean(axis=0)                  # noqa: E731
    assert np.allclose(base_ring(p.mesh({"lift": 60.0, "sway": -60.0})), base_ring(rest), atol=1e-9)
    # the controls come with the record
    spec.controls = {"lift": 30.0}
    moved = K.instantiate_all([spec], base, v0)[0]
    assert np.array_equal(moved.mesh().vertices, p.mesh({"lift": 30.0}).vertices)
    assert not np.array_equal(moved.mesh().vertices, rest.vertices)


@assets
def test_every_tail_control_at_its_limits_is_a_watertight_shell(world):
    base, _, v0 = world
    for params in ({}, {"curl": 2.0, "droop": -2.0}, {"tip_radius": 0.0}, {"length": 20.0}):
        spec = K.new_spec("tail", params=params, base=base)
        p = K.instantiate_all([spec], base, v0)[0]
        ranges = {k: p.control_range(k) for k in p.control_names}
        for corner in itertools.product(*ranges.values()):
            m = p.mesh(dict(zip(ranges, corner)))
            rep = m.integrity_report()
            assert rep["watertight"] and rep["outward"] and np.isfinite(m.vertices).all(), (
                params, corner)
        # a parameter at its limit leaves no room beyond it: the sum stays in the range
        lo, hi = p.control_range("curl")
        assert lo + p.params["curl"] == pytest.approx(-2.0) and hi + p.params["curl"] == pytest.approx(2.0)


@assets
@pytest.mark.parametrize("shape", ["curved", "ram", "spiral", "antelope"])
def test_every_horn_shape_at_the_limits_of_its_parameters_is_watertight(world, shape):
    base, _, v0 = world
    horn = K.KINDS["horn"]
    values = horn.defaults(shape=shape)
    for p in horn.params:
        if p.type in ("float", "int") and p.applies(values):
            for which in ("lo", "hi"):
                spec = K.new_spec("horn", params={"shape": shape, p.key: getattr(p, which)},
                                  side="left", base=base)
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    placed = K.instantiate_all([spec], base, v0)
                assert len(placed) == 1
                rep = placed[0].mesh().integrity_report()
                assert rep["watertight"] and rep["outward"] and rep["volume"] > 0, (shape, p.key, which)


@assets
def test_the_horn_shapes_are_four_different_horns_on_the_same_sunk_base(world):
    base, _, v0 = world
    tips, bases = [], []
    for shape in ("curved", "ram", "spiral", "antelope"):
        p = K.instantiate_all([K.new_spec("horn", params={"shape": shape}, side="left",
                                          base=base)], base, v0)[0]
        m = p.mesh()
        tips.append(m.vertices[np.argmax(np.linalg.norm(m.vertices - p.anchor, axis=1))])
        bases.append(m.vertices[:p.params["ring_verts"]].mean(axis=0))      # ring 0
    assert np.allclose(bases, bases[0], atol=1e-9)                     # all grow from the same place
    for a, b in itertools.combinations(tips, 2):
        assert np.linalg.norm(a - b) > 1.0


def _tube_report(placed_list, base, v, kinds_params=None):
    """The report of a set of placed appendages exactly as the CLI makes it (ready-made
    shells), at 'medium'."""
    app_shells = [[(p.name, p.mesh())] for p in placed_list]
    figure = pp.character_print_shells(base, v)
    shells = figure + pp.appendage_print_shells(app_shells)
    mm = pp.figure_scale(shells, size="medium", reference=pp.FIGURE_SHELLS)
    return pr.appendage_report(app_shells, base, v, mm, figure_shells=figure), mm


@assets
@pytest.mark.parametrize("name", list(BODIES))
def test_the_default_arms_legs_tail_and_horns_print_clean_at_medium(world, bodies, name):
    """Acceptance (4): no BLOCKING finding for the default arm, leg, tail and horn pairs at
    'medium' on the neutral and the tall, short, child and heavy bodies (no root-weld block:
    the weld into the torso is real; no thin-part block). The findings that remain are
    warnings, listed in the next test."""
    base, _, _ = world
    v = bodies[name]
    for kind, params in (("harvest", {}), ("harvest", {"limb": "leg"}), ("tail", {}), ("horn", {})):
        placed = K.instantiate_all([K.new_spec(kind, params=params, base=base)], base, v)
        rep, mm = _tube_report(placed, base, v)
        assert rep.ok and not rep.blocking, (name, kind, params, rep.as_text())
        for g in rep.groups.values():
            assert g.root_embed >= 0.08, (name, kind, g.name, g.root_embed)      # not even a warning
            assert g.detached == []


@assets
def test_the_rounded_tip_removes_the_tube_tip_warning_at_medium(world, bodies):
    """Before: the default tail's tip is 0.07 mm across and the horn's 0.04 mm at 'medium'
    (the printing review), below the 0.2 mm a printer resolves, and the report warns. After:
    tip_radius 0.09 / 0.08 gives 0.3 / 0.3 mm, no warning, on the neutral body and on the
    tall one (the smallest scale, 1.33 mm per unit)."""
    base, _, _ = world

    def warns(v, kind, tip):
        params = {"tip_radius": tip} if tip is not None else {}
        spec = K.new_spec(kind, params=params, base=base)
        values, _ = K.KINDS[kind].clean(spec.params)
        legacy_params = {k: val for k, val in values.items()
                         if k in ("length", "base_radius", "tip_radius", "curve", "curl", "droop",
                                  "twist", "turns", "segments", "ring_verts")}
        if legacy_params.get("tip_radius", 0) == 0:
            legacy_params.pop("tip_radius", None)
        face = K.PRESET_FACES["tail_base" if kind == "tail" else "temples"]
        app = A.Appendage(kind, anchor_face=face, up_hint=(0.0, 1.0, 0.0), params=legacy_params,
                          offset=K.TAIL_SINK if kind == "tail" else K.HORN_SINK,
                          side="center" if kind == "tail" else "both")
        shells = pp.character_print_shells(base, v)
        mm = pp.figure_scale(shells + pp.appendage_print_shells([app], base, v), size="medium",
                             reference=pp.FIGURE_SHELLS)
        rep = pr.appendage_report([app], base, v, mm)
        tips = [g.tube["tip_mm"] for g in rep.groups.values() if g.tube]
        return [f for f in rep.findings if f.code == "tube-tip"], tips, rep

    for name in ("neutral", "tall"):
        v = bodies[name]
        for kind in ("tail", "horn"):
            before, tips_before, _ = warns(v, kind, 0.0)
            after, tips_after, rep = warns(v, kind, None)
            assert before and max(tips_before) < 0.1, (name, kind, tips_before)
            assert not after and min(tips_after) >= 0.2, (name, kind, tips_after)
            assert rep.ok


@assets
def test_the_default_tail_and_horn_look_is_close_to_the_first_versions(world):
    """'Without changing the default look much': the rounded tail / horn follow the pointed
    ones to within a tip radius everywhere (the base radius and the length are untouched)."""
    base, _, v0 = world
    for kind in ("tail", "horn"):
        new = K.instantiate_all([K.new_spec(kind, side="left" if kind == "horn" else None,
                                            base=base)], base, v0)[0].mesh()
        old = K.instantiate_all([K.new_spec(kind, params={"tip_radius": 0.0},
                                            side="left" if kind == "horn" else None,
                                            base=base)], base, v0)[0].mesh()
        (lo_n, hi_n), (lo_o, hi_o) = new.bounds, old.bounds
        assert np.abs(hi_n - hi_o).max() < 0.35 and np.abs(lo_n - lo_o).max() < 0.35, kind
        assert new.volume() / old.volume() < 1.35


# -- placement constants ---------------------------------------------------------------------------


@assets
def test_the_neutral_limb_lengths_and_the_new_preset_faces_still_describe_the_base(world):
    """The harvest lengths scale by the ratio to the NEUTRAL upper arm / thigh, and the new
    presets are quads of the neutral body: if the base mesh ever changes, this says so."""
    base, lib, v0 = world
    for limb, chain in K.LIMB_CHAINS.items():
        s, e, _ = (A.joint_position(base, n, v0) for n in chain)
        assert float(np.linalg.norm(e - s)) == pytest.approx(A.LIMB_REFERENCE_LENGTH[limb], rel=1e-12)
    nc = K.neutral_cage(base, lib)
    for name in ("lower_arms", "upper_arms", "extra_legs"):
        assert K.nearest_face(nc, K.PRESET_POINTS[name]) == K.PRESET_FACES[name], name


@assets
def test_new_presets_follow_the_body(world, bodies):
    """Bound to a quad, a limb's shoulder lands on the same part of the torso on a tall, short,
    child or heavy body: its height over the pelvis, as a share of the torso, barely moves."""
    base, _, _ = world
    share = []
    for name, v in bodies.items():
        p = _harvest(base, v, "lower_arms", side="left")[0]
        pelvis = A.joint_position(base, "pelvis", v)[1]
        neck = A.joint_position(base, "l-shoulder", v)[1]
        share.append((p.anchor[1] - pelvis) / (neck - pelvis))
    assert max(share) - min(share) < 0.12, share


# -- the command line ---------------------------------------------------------------------------------


@assets
def test_the_command_line_poses_a_harvested_arm_and_prints_it(tmp_path, capsys):
    """``--add harvest --pose arm-1:elbow=60 -o x.stl``: the figure is 32.0 mm tall with the
    arms on it, the report is printed and does not block, the posed arm is not the unposed
    one, and the PNG shows it."""
    plain, posed = tmp_path / "plain.stl", tmp_path / "posed.stl"
    assert main(["character", "-o", str(plain), "--add", "harvest", "--size", "medium"]) == 0
    out = capsys.readouterr().out
    assert "appendages: arm-1.l, arm-1.r" in out and "printability  OK" in out
    assert "height       32.0 mm" in out
    rc = main(["character", "-o", str(posed), "--add", "harvest", "--pose",
               "arm-1:elbow=70,raise_=15", "--size", "medium"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "printability  OK" in out and "height       32.0 mm" in out
    assert plain.read_bytes() != posed.read_bytes()
    png_plain, png_posed = tmp_path / "a.png", tmp_path / "b.png"
    assert main(["character", "-o", str(png_plain), "--add", "harvest", "--res", "120", "180"]) == 0
    assert main(["character", "-o", str(png_posed), "--add", "harvest", "--pose",
                 "arm-1:elbow=70,raise_=15", "--res", "120", "180"]) == 0
    capsys.readouterr()
    assert png_plain.read_bytes() != png_posed.read_bytes()


@assets
def test_the_command_line_adds_legs_by_limb_or_by_preset_and_refuses_the_wrong_control(tmp_path, capsys):
    out = tmp_path / "x.stl"
    assert main(["character", "-o", str(out), "--add", "harvest:limb=leg", "--add",
                 "harvest:preset=upper_arms", "--pose", "leg-1:knee=40", "--size", "medium"]) == 0
    cap = capsys.readouterr()
    assert "appendages: leg-1.l, leg-1.r, arm-1.l, arm-1.r" in cap.out
    assert main(["character", "-o", str(out), "--add", "harvest", "--pose", "arm-1:knee=40",
                 "--size", "medium"]) == 0               # a knee on an arm: named, then ignored
    assert "does not apply to an arm" in capsys.readouterr().err
    assert main(["character", "-o", str(out), "--add", "harvest:preset=extra_legs,limb=arm"]) == 2
    assert "is for limb='leg'" in capsys.readouterr().err
    assert main(["character", "-o", str(out), "--add", "harvest", "--pose", "arm-1:elbo=1"]) == 2
    assert "did you mean 'elbow'" in capsys.readouterr().err


@assets
def test_the_command_line_poses_a_tail_and_picks_a_horn_shape(tmp_path, capsys):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    assert main(["character", "-o", str(a), "--add", "tail", "--add", "horn:shape=ram",
                 "--res", "120", "180"]) == 0
    assert main(["character", "-o", str(b), "--add", "tail", "--pose", "tail-1:lift=50,sway=-30",
                 "--add", "horn:shape=ram", "--res", "120", "180"]) == 0
    assert a.read_bytes() != b.read_bytes()
    assert "horn-1.l" in capsys.readouterr().out
    assert main(["character", "-o", str(a), "--add", "horn:shape=unicorn"]) == 2
    assert "one of" in capsys.readouterr().err


@assets
def test_the_figure_with_arms_a_tail_and_horns_exports_clean(tmp_path, capsys):
    """The task's end-to-end: extra arms + tail + horns, body 32.0 mm, the report printed."""
    out = tmp_path / "figure.stl"
    rc = main(["character", "-o", str(out), "--add", "harvest", "--add", "tail", "--add",
               "horn:shape=ram", "--pose", "arm-1:elbow=50", "--pose", "tail-1:lift=25",
               "--size", "medium"])
    cap = capsys.readouterr()
    assert rc == 0, cap.err
    assert "height       32.0 mm" in cap.out and "watertight   True" in cap.out
    assert "printability  OK" in cap.out and "0 blocking" in cap.out
    assert out.stat().st_size > 1000
