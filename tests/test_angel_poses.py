"""The feathered wing's intermediate poses, its folded weld and its feather details.

``test_wingrig.py`` pins the rig and the flat stow; these tests pin what makes a half-folded
angel wing read as a wing (the plumage fans across the elbow instead of hinging there), a curled
hand read as curled (a graded cup, not an L), a stowed wing a print that is actually joined to the
back, and the outer primaries a tip a printer can hold.
"""

import numpy as np
import pytest

from minimaster.character import appendages as A
from minimaster.character import mhbase as mh
from minimaster.character import wings as W
from tests.test_wingrig import _morphed_bodies
from tests.test_wings import _piece_extent, _plan_frame


@pytest.fixture(scope="module")
def feathered():
    return W.make_wing_rig("feathered")


def _heading(rig, J, name):
    d = J[f"{name}_t"] - J[f"{name}_r"]
    return float(np.degrees(np.arctan2(d[2], d[0])))


def _length(rig, name):
    j = rig.skeleton.joints
    return float(np.linalg.norm(j[f"{name}_t"] - j[f"{name}_r"]))


# -- half fold: the plumage fans across the elbow ----------------------------------------------

def _trailing_edge_jump(rig, fold):
    """The biggest gap between the tips of neighbouring feathers along the arm (the secondaries from
    the wrist to the elbow, then the tertials from the elbow to the shoulder: the trailing edge of the
    inner wing), as a share of a secondary's length."""
    J = rig.posed_joints(rig.pose_from_controls(fold=fold))
    names = [f"secondary{i:02d}" for i in range(1, 15)] + [f"tertial{i:02d}" for i in range(1, 8)]
    tips = np.array([J[f"{n}_t"] for n in names])
    gaps = np.linalg.norm(np.diff(tips, axis=0), axis=1)
    return float(gaps.max() / np.mean([_length(rig, n) for n in names[:14]]))


@pytest.mark.parametrize("fold", [0.25, 0.5, 0.75])
def test_a_half_folded_wing_has_a_continuous_trailing_edge_over_the_elbow(feathered, fold):
    """Carried rigidly, the secondaries turn through the whole elbow angle against the tertials: the
    two blocks were separate rectangles with a step between them (the tips of the last secondary and
    the first tertial 0.5-0.8 of a secondary apart at fold .25-.75). Fanned across the elbow, no two
    neighbouring tips are further apart than a third of a secondary."""
    assert _trailing_edge_jump(feathered, fold) < 0.30, fold
    hinged = W.make_wing_rig("feathered")
    hinged.skeleton.elbow_fan = 0.0
    assert _trailing_edge_jump(hinged, fold) > 0.45, fold


def test_the_last_secondary_and_the_first_tertial_meet_at_one_heading_at_the_elbow(feathered):
    """The fan: at the elbow each feather points half way between the headings of the two arm bones,
    so the secondaries and the tertials telescope out of the same point. Hinged, they differ by the
    whole elbow angle."""
    def gap(rig):
        J = rig.posed_joints(rig.pose_from_controls(fold=0.5))
        return abs((_heading(rig, J, "secondary14") - _heading(rig, J, "tertial01") + 180.0) % 360.0 - 180.0)
    hinged = W.make_wing_rig("feathered")
    hinged.skeleton.elbow_fan = 0.0
    assert gap(feathered) < 12.0, gap(feathered)
    assert gap(hinged) > 40.0, gap(hinged)
    # and the turn runs out: a secondary well along the forearm keeps the heading of its bone
    far = feathered.skeleton._elbow_fan_weights()
    assert far["secondary14_r"][1] == pytest.approx(0.5, abs=0.02)
    assert far["tertial01_r"][1] == pytest.approx(0.5, abs=0.03)
    assert "secondary01_r" not in far and "tertial07_r" not in far
    assert "primary05_r" not in far                                  # the hand is not part of it


def test_the_fan_across_the_elbow_is_gone_at_rest_and_when_stowed(feathered):
    """It must not move the spread wing or the stowed one (the stowed heading takes over)."""
    hinged = W.make_wing_rig("feathered")
    hinged.skeleton.elbow_fan = 0.0
    for f in (0.0, 1.0):
        a = feathered.pose_mesh(feathered.pose_from_controls(fold=f)).vertices
        b = hinged.pose_mesh(hinged.pose_from_controls(fold=f)).vertices
        assert np.abs(a - b).max() < 1e-9, f
    a = feathered.pose_mesh(feathered.pose_from_controls(fold=0.5)).vertices
    b = hinged.pose_mesh(hinged.pose_from_controls(fold=0.5)).vertices
    assert np.abs(a - b).max() > 0.5


def test_the_tertials_start_as_long_as_the_last_secondary_and_shorten_into_the_back(feathered):
    """The trailing edge ran vertically down at the elbow because the first tertial (0.52 of the
    longest primary) was 13% shorter than the last secondary (0.60): a step. Now they match within
    5% and the tertials shorten smoothly toward the shoulder."""
    sec = _length(feathered, "secondary14")
    ter = [_length(feathered, f"tertial{i:02d}") for i in range(1, 8)]
    assert abs(ter[0] - sec) < 0.05 * sec
    assert all(a > b for a, b in zip(ter, ter[1:]))
    assert ter[-1] > 0.62 * ter[0]       # (0.60 down to 0.40: the innermost, shortest tertial must not swing past the spine when half folded)


# -- the curled hand --------------------------------------------------------------------------

def _droop(rig, **kw):
    J = rig.posed_joints(rig.pose_from_controls(**kw))
    out = []
    for i in range(1, 11):
        d = J[f"primary{i:02d}_t"] - J[f"primary{i:02d}_r"]
        out.append(np.degrees(np.arcsin(d[1] / np.linalg.norm(d))))
    return np.array(out)


def test_a_curled_hand_is_a_graded_cup_and_not_a_hinge(feathered):
    """At curl 100 the whole hand pivoted ~80 degrees about its base (P5 21 deg, P6 53 deg: a 32 degree
    step) with every primary straight and parallel to the next, as wrist_flex -45 does. Now the droop
    grows smoothly from P1 (level) to P10 (steeply down), with no step between neighbours of more than
    a dozen degrees."""
    d = _droop(feathered, curl=100.0)
    assert d[0] < 5.0 and d[-1] > 50.0
    assert np.all(np.diff(d) > 0.0)
    assert np.diff(d).max() < 14.0, np.round(d, 0)
    # the old hand: linear cup, a big knuckle hinge
    hinge = W.make_wing_rig("feathered")
    hinge.skeleton.curl_gain, hinge.skeleton.curl_cup = 0.35, 0.35
    hinge.skeleton.curl_cup_pow, hinge.skeleton.curl_roll = 1.0, 0.0
    assert np.diff(_droop(hinge, curl=100.0)).max() > 25.0
    # and it is not the same pose as drooping the wrist
    assert np.diff(_droop(feathered, wrist_flex=-45.0)).max() < 6.0


def test_the_outer_primaries_roll_about_their_shaft_as_the_hand_curls(feathered):
    """The vane planes follow the cup: P10 turns by curl_roll x curl about its own root against the same
    pose without the roll, P1 not at all. (The relative rotation is read off the posed vertices: the best
    rigid fit of one set onto the other about the root, and the axis of that fit is the shaft.)"""
    flat = W.make_wing_rig("feathered")
    flat.skeleton.curl_roll = 0.0

    def roll_of(name, **kw):
        a, b = feathered.ranges[name]
        va = feathered.pose_mesh(feathered.pose_from_controls(**kw)).vertices[a:b]
        vb = flat.pose_mesh(flat.pose_from_controls(**kw)).vertices[a:b]
        J = flat.posed_joints(flat.pose_from_controls(**kw))
        r, shaft = J[f"{name}_r"], J[f"{name}_t"] - J[f"{name}_r"]
        shaft = shaft / np.linalg.norm(shaft)
        u, _, vt = np.linalg.svd((vb - r).T @ (va - r))
        R = (u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt).T      # takes vb onto va
        angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
        k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        about = abs(float(k @ shaft)) / max(float(np.linalg.norm(k)), 1e-12)
        return angle, about

    angle, about = roll_of("primary10", curl=100.0)
    assert angle == pytest.approx(30.0, abs=1.0)
    assert about > 0.99                                                # ... about its own shaft
    assert roll_of("primary01", curl=100.0)[0] < 0.5
    assert roll_of("primary10")[0] < 1e-6                               # no curl, no roll


# -- feather details ---------------------------------------------------------------------------

def _tip_width(rig, name, lo, hi):
    """Width across the vane (plan) of the part of a feather between two fractions of its length."""
    a, b = rig.ranges[name]
    v = rig.rest_mesh.vertices[a:b]
    j = rig.skeleton.joints
    r, t = j[f"{name}_r"], j[f"{name}_t"]
    d = t - r
    length = float(np.linalg.norm(d))
    u = d / length
    perp = np.array([-u[2], 0.0, u[0]])
    s = ((v - r) @ u) / length
    sel = (s >= lo) & (s < hi)
    lat = (v[sel] - r) @ perp
    return float(lat.max() - lat.min())


@pytest.mark.parametrize("name", ["primary07", "primary08", "primary09", "primary10"])
def test_the_outer_primaries_end_in_a_tip_a_printer_can_hold(feathered, name):
    """P7-P10 tapered from 70% of their length down to 7% of their width: the last 3% was 0.09-0.17
    units wide (0.17-0.32 mm at 32 mm), a fifth of the vane -- a needle that snaps. They now keep their
    distal vane to 84% and close onto a rounded tip at half the width: a third of the vane over the last
    3% of the length (0.22-0.25 before) and half of it just before that (0.29-0.35 before)."""
    mid = _tip_width(feathered, name, 0.50, 0.60)
    assert _tip_width(feathered, name, 0.97, 1.0) > 0.30 * mid
    assert _tip_width(feathered, name, 0.93, 0.96) > 0.40 * mid
    assert _tip_width(feathered, name, 0.97, 1.0) > 0.12             # and never thinner than 0.2 mm
    # and the slots between the outer primaries are still open
    assert mid < 0.8


def test_the_covert_rows_are_not_a_lattice(feathered):
    """A stowed wing lays every covert of a tier parallel, and equal ones (length jitter 3%) read as a
    lattice of identical round tips -- armour. They now vary by ~6% (and in aim and tilt)."""
    sk = feathered.skeleton
    pieces = {p.name: p for p in feathered.pieces}
    ratio = []
    for i in range(1, 15):
        r, d, n, length = _plan_frame(sk, f"secondary{i:02d}")
        ratio.append(_piece_extent(pieces[f"greater{i:02d}"].mesh, d)[0] / length)
    ratio = np.array(ratio)
    assert ratio.std() / ratio.mean() > 0.04
    assert 0.44 < ratio.mean() < 0.58                                  # still half a secondary


# -- on the body: the folded weld ---------------------------------------------------------------

def _back_depth(pts, tris):
    """(inside, depth) of points against the body: how far behind the back's skin each one is, measured
    along the -Z ray (the back faces -Z); a point is inside when an odd number of skin triangles
    lie behind it."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    v0, v1 = c[:, :2] - a[:, :2], b[:, :2] - a[:, :2]
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    ok = np.abs(den) > 1e-14
    safe = np.where(ok, den, 1.0)
    lo = np.minimum(np.minimum(a[:, :2], b[:, :2]), c[:, :2])
    hi = np.maximum(np.maximum(a[:, :2], b[:, :2]), c[:, :2])
    inside = np.zeros(len(pts), dtype=bool)
    depth = np.zeros(len(pts))
    for i, p in enumerate(pts):
        idx = np.nonzero(ok & (p[0] >= lo[:, 0]) & (p[0] <= hi[:, 0])
                         & (p[1] >= lo[:, 1]) & (p[1] <= hi[:, 1]))[0]
        if not len(idx):
            continue
        v2 = p[:2] - a[idx, :2]
        d20, d21 = (v2 * v0[idx]).sum(1), (v2 * v1[idx]).sum(1)
        u = (d11[idx] * d20 - d01[idx] * d21) / safe[idx]
        w = (d00[idx] * d21 - d01[idx] * d20) / safe[idx]
        m = (u >= 0) & (w >= 0) & (u + w <= 1)
        if not m.any():
            continue
        h = idx[m]
        z = a[h, 2] + u[m] * (c[h, 2] - a[h, 2]) + w[m] * (b[h, 2] - a[h, 2])
        behind = z[z < p[2]]
        if len(behind) % 2 == 1:
            inside[i] = True
            depth[i] = p[2] - behind.max()
    return inside, depth


@pytest.fixture(scope="module")
def bodies():
    base, lib = mh.load()
    out = []
    for v in _morphed_bodies(base, lib):
        cv, cq = base.body_cage(v)
        tris = cv[np.vstack([cq[:, [0, 1, 2]], cq[:, [0, 2, 3]]])]
        (_, pw), _ = A.back_wings("feathered").build_rigs(base, v)
        out.append((pw, tris))
    return out


@pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
def test_a_stowed_angel_is_welded_to_the_back_on_every_body(bodies):
    """A stowed wing is a plank on a curved back and was joined along a hairline: 0.4-4.6% of its vertices
    inside the torso, the deepest 0.34-0.61 units (0.6-1.1 mm) and almost none beyond 0.3. Leaning it 11
    degrees about its pivot (the top sinks into the shoulder blade, the hem stands off the buttocks) puts
    1.8-9% inside and the deepest vertices 0.61-0.98 units in, on the plain, the anatomy-adjusted, the
    heavy and the slim body alike."""
    for i, (pw, tris) in enumerate(bodies):
        w = pw.pose(fold=1.0).vertices[::6]
        inside, depth = _back_depth(w, tris)
        assert inside.mean() > 0.012, (i, inside.mean())
        assert depth.max() > 0.55, (i, depth.max())
        assert (depth > 0.3).mean() > 0.002, (i, (depth > 0.3).mean())
        assert inside.mean() < 0.12, (i, "buried")


@pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
def test_no_covert_is_built_wholly_inside_the_torso(bodies):
    """The first three marginal and two under-marginal coverts at the shoulder lay wholly inside the torso
    on every body (the root is sunk 0.9 into the back): ~1.4k triangles nobody sees. They are no longer
    built (see ROOT_BURIED), and no piece of the spread wing is buried wholesale on all four morphs."""
    buried = None
    for pw, tris in bodies:
        m = pw.pose().vertices
        here = set()
        for p in pw.rig.pieces:
            a, b = pw.rig.ranges[p.name]
            inside, _ = _back_depth(m[a:b][::2], tris)
            if inside.mean() >= 0.95:
                here.add(p.name)
        buried = here if buried is None else buried & here
    assert not buried, sorted(buried)
