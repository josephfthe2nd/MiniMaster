"""Skin quality of the bat / dragon wing: continuity under a sweeping control, the stretch
limiter, rigid finger tubes and the posed slab's orientation.

These were found by sweeping a control in small steps and by measuring the *posed* shells
(the watertight gate is edge-count topology and cannot see a bow-tie or a flipped cell).
Every test here fails on the code it was written against (the pleat/gather post-process
that switched on at a closure threshold, a Jacobi limiter that could not move a chain
between two pinned ends, finger tubes skinned across their knuckles).
"""

from __future__ import annotations

import numpy as np
import pytest

from minimaster.character import wings as W
from minimaster.character import wingrig as R
from minimaster.character.wingrig import PoseControls, WingRig

STYLES = ("membrane", "dragon")


@pytest.fixture(scope="module", params=STYLES)
def rig(request):
    return W.make_wing_rig(request.param)


def _mid(rig, vertices):
    piece = next(p for p in rig.pieces if p.pleat)
    a, _ = rig.ranges[piece.name]
    n1, m1 = piece.pleat["shape"]
    nv = n1 * m1
    v = vertices[a:a + 2 * nv]
    return (0.5 * (v[:nv] + v[nv:])).reshape(n1, m1, 3), (n1, m1)


def _finger_joints(rig, pose):
    J = rig.posed_joints(pose)
    return np.array([J[j] for chain in rig.skeleton.chains.values() for j in chain[1:]])


def _sweep(rig, control, values, **fixed):
    """Posed sheet mid-surface (T, n, 3), sheet vertices (T, nv, 3) and finger joints (T, J, 3)."""
    mids, sheets, joints = [], [], []
    for x in values:
        pose = rig.pose_from_controls(**{**fixed, control: float(x)})
        verts = rig.pose_mesh(pose).vertices
        mid, _ = _mid(rig, verts)
        a, b = rig.ranges["membrane"]
        mids.append(mid)
        sheets.append(verts[a:b])
        joints.append(_finger_joints(rig, pose))
    return np.array(mids), np.array(sheets), np.array(joints)


def _pops(mid, ratio=2.5, floor=0.15):
    """Steps in which some vertex moves far more than it did just before and just after."""
    s = np.linalg.norm(np.diff(mid, axis=0), axis=-1)
    worst, count = 0.0, 0
    for i in range(1, len(s) - 1):
        excess = s[i] - ratio * np.maximum(s[i - 1], s[i + 1]) - floor
        if (excess > 0).any():
            count += 1
            worst = max(worst, float(excess.max()))
    return count, worst


SWEEPS = {"fold": np.arange(0.0, 1.0001, 0.005), "curl": np.arange(0.0, 150.0001, 1.0)}


@pytest.mark.parametrize("control", ["fold", "curl"])
def test_no_sheet_vertex_pops_as_a_control_is_swept(rig, control):
    """Stepping ``fold`` by 0.5% or ``curl`` by a degree, one mid-surface vertex moved 0.04 per
    step and then 1.1 in a single step (the interior switching from skinned to chord-based as
    a panel closed past its elastic limit; 70-108 of 200 fold steps held such a pop, and
    curl 127 -> 128). A vertex may not move more than 2.5x as far as it did the step before or
    after, plus a sixth of a unit."""
    mid, _, _ = _sweep(rig, control, SWEEPS[control])
    count, worst = _pops(mid)
    assert count == 0, (control, count, worst)


@pytest.mark.parametrize("control", ["fold", "curl"])
def test_the_sheet_never_outruns_the_bones_it_hangs_from(rig, control):
    """The fastest sheet vertex in a sweep is at most 3x the fastest finger joint (rigid bone
    vertices move with their joint; the skin may only follow). On the old code a curl sweep
    peaked at 4.7x (bat) and 5.4x (dragon); plain skinning gives 1x."""
    _, sheet, joints = _sweep(rig, control, SWEEPS[control])
    sheet_step = np.linalg.norm(np.diff(sheet, axis=0), axis=2).max()
    bone_step = np.linalg.norm(np.diff(joints, axis=0), axis=2).max()
    assert sheet_step <= 3.0 * bone_step, (control, sheet_step, bone_step, sheet_step / bone_step)


def test_the_sheet_is_continuous_while_folding_a_curled_wing(rig):
    mid, _, _ = _sweep(rig, "fold", np.arange(0.0, 1.0001, 0.005), curl=30.0)
    count, worst = _pops(mid)
    assert count <= 1 and worst < 0.2, (count, worst)


# -- the stretch limiter ---------------------------------------------------------------------

def _chain(n, total, xs=None):
    """A free chain of ``n`` edges between two pinned ends ``total`` apart (a row of the sheet
    between two fingers; the grid has two such rows). The rest length of the whole chain is 1;
    ``xs`` gives the free vertices' positions along it (default evenly spaced)."""
    rest = np.zeros((2, n + 1, 3))
    rest[:, :, 0] = np.linspace(0.0, 1.0, n + 1)[None, :]
    rest[1, :, 2] = 1.0
    x = np.linspace(0.0, total, n + 1) if xs is None else np.asarray(xs, dtype=float)
    mid = rest.copy()
    mid[:, :, 0] = x[None, :]
    pinned = np.zeros((2, n + 1), dtype=bool)
    pinned[:, 0], pinned[:, -1] = True, True
    return mid, rest, pinned


def _worst_stretch(mid, rest):
    cur = np.linalg.norm(np.diff(mid, axis=1), axis=2)
    ref = np.linalg.norm(np.diff(rest, axis=1), axis=2)
    return float((cur / ref).max())


def _pushed_chain():
    """Six cells between two pinned fingers 1.10 apart (the skin can reach 1.15), two free
    vertices pushed along so one edge is stretched to 1.8x and its neighbour crushed."""
    xs = np.linspace(0.0, 1.10, 7)
    xs[2:4] += 0.12
    return _chain(6, 1.10, xs)


def test_the_limiter_wins_a_free_chain_between_two_pinned_ends():
    """Jacobi passes with a count of active edges per vertex (16 of them) left this chain at
    1.22x; Gauss-Seidel projection brings every edge to the limit. A feasible chain between
    pinned ends can always be shortened: only the free vertices move."""
    mid, rest, pinned = _pushed_chain()
    assert _worst_stretch(mid, rest) > 1.7                      # the setup is violent
    out = WingRig._limit_stretch(mid, rest, pinned)
    assert _worst_stretch(out, rest) < 1.15 + 2e-3, _worst_stretch(out, rest)
    assert (out[pinned] == mid[pinned]).all()                   # pinned vertices stay put


def test_the_limiter_leaves_a_slack_sheet_alone():
    mid, rest, pinned = _chain(14, 0.9)
    out = WingRig._limit_stretch(mid, rest, pinned)
    assert np.array_equal(out, mid)


def test_the_limiter_is_a_continuous_function_of_the_positions():
    """It used to divide each vertex's correction by the number of edges that were over the
    limit, so a second edge crossing it by a hair halved the first one's pull: a vertex jumped
    by a sixth of a cell when another edge was one part in a million over the limit."""
    r = 1.0 / 6.0
    xs = np.array([0.0, r, 2 * r, 2 * r + 1.15 * r, 2 * r + 1.15 * r + 1.5 * r, 1.10 - r, 1.10])
    outs = []
    for eps in (-1e-7, 1e-7):                                   # the edge before vertex 3 just under / over its limit
        moved = xs.copy()
        moved[3] += eps
        mid, rest, pinned = _chain(6, 1.10, moved)
        outs.append(WingRig._limit_stretch(mid, rest, pinned))
    assert np.abs(outs[1] - outs[0]).max() < 1e-4, np.abs(outs[1] - outs[0]).max()


def test_an_unreachable_chain_is_stretched_evenly_not_piled_on_one_edge():
    """Two pinned ends farther apart than the chain can reach have no solution; the strain is
    then spread along the row (it is what its ends dictate) instead of landing on the last edge
    the sweep visits."""
    mid, rest, pinned = _chain(14, 1.62)
    mid[:, 1:-1, 0] += 0.02 * np.sin(np.arange(1, 14))[None, :]      # a little uneven to start with
    limit = np.full((2, 14), 1.02 * 1.62)                            # what the ends dictate
    out = WingRig._limit_stretch(mid, rest, pinned, limit_v=limit)
    cur = np.linalg.norm(np.diff(out, axis=1), axis=2)[0]
    assert cur.max() / cur.min() < 1.9, (cur.min(), cur.max())
    # held to 1.15 instead, projection alone leaves the row alternating between long and short edges
    held = np.linalg.norm(np.diff(WingRig._limit_stretch(mid, rest, pinned), axis=1), axis=2)[0]
    assert held.max() / held.min() > 1.5


def _stretch_stats(rig, kw):
    """Worst and count (>1.3) of mid-surface edge stretch (rest length over 0.15) in a pose."""
    piece = next(p for p in rig.pieces if p.pleat)
    a, _ = rig.ranges[piece.name]
    rm, _ = _mid(rig, rig.rest_mesh.vertices)
    # (raw controls, not clamped to the wing's ranges: this tests the limiter, not the ranges)
    pm, _ = _mid(rig, rig.pose_mesh(rig.skeleton.controls_to_pose(PoseControls(**kw))).vertices)
    out = []
    for axis in (0, 1):
        r = np.linalg.norm(np.diff(rm, axis=axis), axis=2)
        p = np.linalg.norm(np.diff(pm, axis=axis), axis=2)
        out.append((p / np.maximum(r, 1e-9))[r > 0.15])
    st = np.concatenate(out)
    return float(st.max()), int((st > 1.3).sum()), float(np.percentile(st, 99))


LIMITED_POSES = [{"fold": 0.8}, {"fold": 0.5, "curl": 60.0}, {"fold": 0.8, "curl": 30.0}, {"curl": 100.0}]


@pytest.mark.parametrize("kw", LIMITED_POSES, ids=str)
def test_the_limiter_keeps_the_skin_from_stretching(rig, kw, monkeypatch):
    """With the limiter the skin has at most a handful of edges over 1.3x its rest length in
    these poses; with it switched off 26-99 (and curl 100 reaches 4.1x on the bat, 7.5x on the
    dragon)."""
    _, on, _ = _stretch_stats(rig, kw)
    with monkeypatch.context() as m:
        m.setattr(WingRig, "_limit_stretch", staticmethod(lambda mid, rest, pinned, iters=0, limit_v=None: mid))
        _, off, _ = _stretch_stats(rig, kw)
    assert on <= 10, (kw, on, off)
    assert off >= on + 15, (kw, on, off)


def test_the_limiter_works_at_curl_100_where_the_skin_overshoots(rig):
    worst, count, p99 = _stretch_stats(rig, {"curl": 100.0})
    assert worst < 2.2 and count <= 10 and p99 < 1.4, (worst, count, p99)


def test_presets_and_a_flap_cycle_do_not_over_stretch_the_skin(rig):
    """The upstroke preset had 64 edges over 1.3x (1.54 at worst) and flap(90) 56; now 1 and 2.
    The 99th percentile edge stays within a quarter of its rest length everywhere."""
    from minimaster.character.wingrig import PRESETS, PoseControls
    for name, controls in [(n, PoseControls(**c)) for n, c in PRESETS.items()] + [
            (f"flap{ph}", PoseControls.flap(float(ph))) for ph in range(0, 360, 45)]:
        piece = rig.pose(controls)
        mid, _ = _mid(rig, piece.vertices)
        rmid, _ = _mid(rig, rig.rest_mesh.vertices)
        st = []
        for axis in (0, 1):
            r = np.linalg.norm(np.diff(rmid, axis=axis), axis=2)
            p = np.linalg.norm(np.diff(mid, axis=axis), axis=2)
            st.append((p / np.maximum(r, 1e-9))[r > 0.15])
        st = np.concatenate(st)
        assert np.percentile(st, 99) < 1.25 and (st > 1.3).sum() <= 12, (
            name, float(np.percentile(st, 99)), int((st > 1.3).sum()))


# -- finger tubes -----------------------------------------------------------------------------

def _ring_area(vertices, sides):
    n = (len(vertices) - 2) // sides
    ring = vertices[:n * sides].reshape(n, sides, 3)
    c = ring.mean(axis=1, keepdims=True)
    p = ring - c
    return 0.5 * np.linalg.norm(np.cross(p, np.roll(p, -1, axis=1)).sum(axis=1), axis=1)


def _finger_pieces(rig):
    return [p.name for p in rig.pieces if p.name.startswith("digit_")]


POSES_THAT_FOLD_FINGERS = [{"fold": 0.9}, {"fold": 1.0}, {"curl": 60.0}, {"curl": 150.0},
                           {"fold": 1.0, "curl": 60.0}, {"fold": 1.0, "curl": 150.0}]


@pytest.mark.parametrize("kw", POSES_THAT_FOLD_FINGERS, ids=str)
def test_finger_tube_cross_sections_do_not_collapse(rig, kw):
    """Skinned as one tube across its knuckles, a finger's rings were 5-12% of their rest area
    at a fully folded knuckle (digit III 0.12, IV 0.07, V 0.05), 0.55 at fold 0.9 and 0.25 at
    curl 150: bow-ties the watertight gate cannot see. One rigid tube per phalanx keeps every
    ring at its rest area."""
    rest = rig.rest_mesh.vertices
    posed = rig.pose(**kw).vertices
    worst = 1e9
    for name in _finger_pieces(rig):
        a, b = rig.ranges[name]
        r0, r1 = _ring_area(rest[a:b], 7), _ring_area(posed[a:b], 7)
        worst = min(worst, float((r1 / np.maximum(r0, 1e-12)).min()))
    assert worst > 0.98, (kw, worst)


def test_every_phalanx_is_a_rigid_tube_and_every_knuckle_a_ball(rig):
    sk = rig.skeleton
    pieces = {p.name: p for p in rig.pieces}
    for digit, chain in sk.chains.items():
        bones = chain[1:]                                      # one bone per phalanx (and the metacarpal)
        for i, key in enumerate(bones):
            name = f"digit_{digit}" if i == 0 else f"digit_{digit}_p{i}"
            assert pieces[name].bone == key, name              # welded to its own bone
        for i in range(1, len(bones)):
            ball = pieces[f"knuckle_{digit}_{i}"]
            assert ball.bone is not None
    idx, wt = rig.weights()
    for name in _finger_pieces(rig):
        a, b = rig.ranges[name]
        assert (wt[a:b, 0] == 1.0).all()


def test_a_finger_still_reads_as_one_tapering_rod_at_rest(rig):
    """The phalanx tubes are slices of one taper: the radius never jumps at a knuckle, so the
    spread finger is as smooth as the single tube it replaced."""
    for digit, chain in rig.skeleton.chains.items():
        names = [f"digit_{digit}"] + [f"digit_{digit}_p{i}" for i in range(1, len(chain) - 1)]
        radii = []
        for n in names:
            a, b = rig.ranges[n]
            v = rig.rest_mesh.vertices[a:b]
            rings = v[:(len(v) - 2) // 7 * 7].reshape(-1, 7, 3)
            radii.append(np.linalg.norm(rings - rings.mean(axis=1, keepdims=True), axis=2).mean(axis=1))
        for r0, r1 in zip(radii[:-1], radii[1:]):
            assert abs(r1[0] - r0[-1]) < 1e-9, digit             # the next tube starts at the radius the last ended at
        assert radii[-1][min(6, len(radii[-1]) - 1)] <= radii[0][0] + 1e-9


# -- the posed slab ---------------------------------------------------------------------------

def _cell_volumes(rig, pose_kw):
    """Signed volume of each grid cell of the posed slab (its oriented area times the thickness
    vector along that orientation) and its area."""
    piece = next(p for p in rig.pieces if p.pleat)
    a, _ = rig.ranges[piece.name]
    n1, m1 = piece.pleat["shape"]
    nv = n1 * m1
    v = rig.pose(**pose_kw).vertices[a:a + 2 * nv]
    mid = (0.5 * (v[:nv] + v[nv:])).reshape(n1, m1, 3)
    up = (v[:nv] - v[nv:]).reshape(n1, m1, 3)
    p00, p01, p10, p11 = mid[:-1, :-1], mid[:-1, 1:], mid[1:, :-1], mid[1:, 1:]
    area = 0.5 * np.cross(p11 - p00, p01 - p10)
    uc = 0.25 * (up[:-1, :-1] + up[:-1, 1:] + up[1:, :-1] + up[1:, 1:])
    return np.einsum("ijk,ijk->ij", area, uc), np.linalg.norm(area, axis=2)


@pytest.mark.parametrize("fold,limit", [(0.5, 0.006), (0.75, 0.006), (0.9, 0.006), (1.0, 0.012)])
def test_a_folding_slab_has_little_inverted_area(rig, fold, limit):
    """The slab's thickness vector must agree with the way its cells are wound: a cell whose
    oriented area points against it is turned inside out. By cell count 10% of the grid was
    (116 of 1122 at fold 1 -- most of them slivers with no area); by area 2% of the sheet at
    fold 0.5-0.9 and 5-6% stowed, now 0.1-0.3%. The area that remains is the collapsed panels of
    the closed fan, where chord and ray are parallel and the sheet has no extent to turn over."""
    vol, area = _cell_volumes(rig, {"fold": fold})
    inverted = float(area[vol < 0].sum() / area.sum())
    assert inverted < limit, (fold, inverted)


@pytest.mark.parametrize("kw,limit", [({"fold": 0.5}, 0.004), ({"curl": 100.0}, 0.005), ({"fold": 0.8, "curl": 30.0}, 0.01)])
def test_a_posed_slab_is_not_inside_out(rig, kw, limit):
    vol, area = _cell_volumes(rig, kw)
    assert float(area[vol < 0].sum() / area.sum()) < limit
    assert float(vol.sum() / np.abs(vol).sum()) > 0.99


# -- the post-process and the rest pose -------------------------------------------------------

def test_the_rest_pose_is_still_the_rest_mesh(rig):
    out = rig.pose_mesh(rig.pose_from_controls()).vertices
    assert np.abs(out - rig.rest_mesh.vertices).max() < 5e-3     # slab normals are re-derived, nothing else moves
    mid0, _ = _mid(rig, rig.rest_mesh.vertices)
    mid1, _ = _mid(rig, out)
    assert np.abs(mid0 - mid1).max() < 1e-9
