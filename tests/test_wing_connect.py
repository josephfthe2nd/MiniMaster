"""Round 8: the wings grow out of the body.

The owner's finding: "wings themselves don't seem to connect to the body".  Measured on the tree that was shown
(a7f84cd): the bat's sheet stood on the back as a 0.22-thick blade, its inner edge a straight line from the shoulder
to the waist that hung 0.40-0.57 off the skin at its lower corner and met the back at 68-82 degrees; the feathered
wing left the back as a bare arm, 27-62% of its top and leading side uncovered for the first 1.2 units, the nose row
of coverts inside the muscle.

A wing is built flat in its own frame and then placed, so its builder never knew where the back was.  It does now
(``appendages.SkinProbe``): the membrane's inner edge follows the back's own profile under the skin, each face of
the sheet leaves it in a fillet sized by its own height over the skin, and the sheet is the body's skin where it
grows out of it; the feathered arm leaves the back under rows of small coverts.  These tests measure the FIGURE
(the placed wing against the moulded, rounded skin the pipeline draws), on four bodies.
"""

import collections
import inspect

import numpy as np
import pytest

from minimaster.character import appendages as A
from minimaster.character import collar as CO
from minimaster.character import kinds as K
from minimaster.character import mhbase as mh
from minimaster.character import weld as WE
from minimaster.character import wingrig as WRIG
from minimaster.character import wings as W

assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")

BODIES = {
    "neutral": dict(gender=1.0, muscle=0.55),
    "heavy": dict(gender=1.0, muscle=1.0, weight=1.0),
    "child": dict(age=0.1875),
    "slim_f": dict(gender=0.0, weight=0.0),
}


@pytest.fixture(scope="module")
def world():
    base, lib = mh.load()
    from minimaster.character.sliders import SliderCatalog
    return base, lib, SliderCatalog(lib.names)


_FIGURES: dict = {}


class Figure:
    """A wing pair on a body as the pipeline draws it: the left wing's placed mesh and the body's skin (moulded by
    the wing's root, subdivided once), with the signed distance of points to that skin."""

    def __init__(self, world, body, style, controls=None):
        from minimaster.character.sliders import Character
        base, lib, cat = world
        c = Character()
        c.macro.update(BODIES[body])
        spec = K.add_from_text(f"wing:style={style}", c.appendages, base=base, lib=lib)
        if controls:
            spec.controls["*"] = dict(controls)
        c.appendages.append(spec)
        self.verts = lib.apply(base.verts, c.effective_target_weights(cat))
        self.placed = c.instantiate_appendages(base, self.verts, lib)
        self.left = next(p for p in self.placed if p.side == "l")
        self.wing = self.left.rig                                    # the PlacedWing
        sv, sf = CO.shaped_skin(base, c.shaped_body(base, self.verts, self.placed), 1)
        self.skin = WE.Surface(np.asarray(sv), np.asarray(sf), self.left.anchor, 9.0, walk=1)
        self.V = np.asarray(self.left.mesh().vertices)
        self.F = np.asarray(self.left.mesh().faces)
        self.ranges = self.wing.rig.ranges

    def distance(self, points):
        """Signed distance to the figure's skin, positive outside."""
        P = np.atleast_2d(np.asarray(points, dtype=np.float64))
        q, _, b, _ = self.skin.closest(P)
        return np.linalg.norm(P - q, axis=1) * np.where(np.asarray(b) < 0.0, -1.0, 1.0)

    def skin_normal(self, points):
        return np.asarray(self.skin.closest(np.atleast_2d(points))[1])

    def piece(self, name):
        a, b = self.ranges[name]
        return self.V[a:b]


def figure(world, body="neutral", style="membrane", controls=None) -> Figure:
    key = (body, style, tuple(sorted((controls or {}).items())))
    if key not in _FIGURES:
        _FIGURES[key] = Figure(world, body, style, controls)
    return _FIGURES[key]


# -- the body's skin in the wing's frame -------------------------------------------------------------


@assets
def test_the_probe_reads_the_bodys_skin_in_the_wings_own_frame(world):
    base, _, _ = world
    body_v, body_q = base.body_cage(base.verts)
    wing = A.back_wings("membrane")
    (_, placed), _ = wing.build_rigs(base, base.verts)
    M = placed.matrix
    probe = A.SkinProbe(M, body_v, body_q, M[:3, 3], 6.0)
    # a point of the back (the one nearest the wing's own origin, which is sunk into it), and the points a fifth of
    # a unit out of and into it along the skin's own normal
    origin = np.zeros((1, 3))
    assert probe.distance(origin)[0] < 0.0                      # the joint is inside the body
    n = probe.normal(origin)[0]
    assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-6)
    local = np.append(origin[0] - probe.distance(origin)[0] * n, 1.0)          # on the skin
    assert probe.distance(local[None, :3])[0] == pytest.approx(0.0, abs=0.02)
    n = probe.normal(local[None, :3])[0]
    out, into = probe.distance(np.array([local[:3] + 0.2 * n, local[:3] - 0.2 * n]))
    assert out == pytest.approx(0.2, abs=0.03) and into == pytest.approx(-0.2, abs=0.03)
    # ``within``: the far points are not searched (inf), the near ones read the same
    pts = np.array([local[:3] + 0.3 * n, local[:3] + 40.0 * n])
    near, far = probe.distance(pts, within=1.0)
    assert near == pytest.approx(probe.distance(pts[:1])[0], abs=1e-12) and np.isinf(far)
    d, nrm = probe.distance(pts, within=1.0, normals=True)
    assert np.allclose(nrm[0], probe.normal(pts[:1])[0]) and not nrm[1].any()
    # the body's size against the reference man's
    assert probe.size == pytest.approx(1.0, abs=0.08)


@assets
def test_a_childs_back_takes_a_childs_fillet(world):
    man, child = figure(world, "neutral"), figure(world, "child")
    thick = []
    for f in (man, child):
        M = f.piece("membrane")
        nv = len(M) // 2
        thick.append(float(np.linalg.norm(M[:nv] - M[nv:], axis=1).max()))
    assert 0.6 * thick[0] < thick[1] < 0.9 * thick[0]


@assets
def test_the_right_wing_reads_the_skin_through_the_mirror(world):
    """A cage quad and its mirror image are split along different diagonals, so the two sides' triangles are not
    mirror images of one another (1e-6 apart): a root fitted to each side's own was 0.05 off its mirror image.  The
    right wing reads the LEFT side's skin through the mirror when the body is its own mirror image -- and its own
    side's when it is not."""
    base, _, _ = world
    body_v, _ = base.body_cage(base.verts)
    assert A._is_symmetric(base, body_v)
    f = figure(world, "neutral", "membrane")
    left, right = (next(p for p in f.placed if p.side == s) for s in "lr")
    lv, rv = np.asarray(left.mesh().vertices), np.asarray(right.mesh().vertices)
    assert A._is_mirror_image(lv, rv, tol=1e-7)
    bent = base.verts.copy()
    side = np.nonzero(base.verts[:, 0] > 1.0)[0]
    bent[side] += np.array([0.0, 0.05, 0.0])                   # one side of the body a little higher
    assert not A._is_symmetric(base, base.body_cage(bent)[0])
    (_, lw), (_, rw) = A.back_wings("membrane").build_rigs(base, bent)       # ... and the pair is still built
    assert len(lw.mesh().vertices) == len(rw.mesh().vertices)


@assets
@pytest.mark.parametrize("style", ["membrane", "dragon", "feathered"])
def test_the_skeleton_is_the_plans_whatever_the_body(world, style):
    """What is built against the body is PIECES.  The joints are the plan's: lifted onto the sheet as the plan has
    it, not onto the sheet fitted to the back (which moved the elbow of a rolled wing by 0.07, and with it where an
    old document's arm leaves the back and the pose limits).  A draft, a wing alone and the figure's wing pose
    alike."""
    base, _, _ = world
    for roll in (None, -95.0):
        fitted, plain = A.back_wings(style), A.back_wings(style, skin=None)
        if roll is not None:
            fitted.roll = plain.roll = roll
        for (_, a), (_, b) in zip(fitted.place_rigs(base, base.verts), plain.place_rigs(base, base.verts)):
            ja, jb = a.rig.skeleton.joints, b.rig.skeleton.joints
            assert set(ja) == set(jb)
            for name in ja:
                assert np.array_equal(np.asarray(ja[name]), np.asarray(jb[name])), (style, roll, name)
            assert np.array_equal(a.matrix, b.matrix)
            # ... and so is every piece but the sheets (the bones are laid on the plan's sheet)
            pa, pb = {p.name: p for p in a.rig.pieces}, {p.name: p for p in b.rig.pieces}
            if style != "feathered":
                assert list(pa) == list(pb)
            changed = [n for n in pa if n in pb and not (
                pa[n].mesh.vertices.shape == pb[n].mesh.vertices.shape
                and np.array_equal(pa[n].mesh.vertices, pb[n].mesh.vertices))]
            assert set(changed) <= {"membrane", "propatagium"}, (style, roll, changed)
            if style != "feathered":
                assert "membrane" in changed


@assets
def test_a_wing_alone_is_what_it_was(world):
    """No body, nothing built against one: no skin on the sheet, no fillet, no coverts fitted to a sleeve."""
    for style in ("membrane", "dragon", "feathered"):
        rig = W.make_wing_rig(style)
        assert all(p.skin_share is None for p in rig.pieces)
        assert all((p.pleat or {}).get("swell") is None for p in rig.pieces)
        assert not [p.name for p in rig.pieces if "shoulder" in p.name]


# -- the membrane ------------------------------------------------------------------------------------


def _root_columns(style):
    """``(rows, columns per row, root columns)`` of the sheet's grid: the columns whose outer end is on the plan's
    root line (the straight run of edge points that ends at the shoulder), from the waist corner to the shoulder."""
    rig = W.make_wing_rig(style, root=(-1.6, 3.0))
    V = np.asarray(next(q for q in rig.pieces if q.name == "membrane").mesh.vertices)
    nv = len(V) // 2
    rows = inspect.signature(W.membrane_pieces).parameters["rows"].default
    nu = nv // (rows + 1)
    edge = 0.5 * (V[rows * nu:(rows + 1) * nu] + V[nv + rows * nu:nv + (rows + 1) * nu])[:, [0, 2]]
    ax = (edge[-2] - edge[-1]) / np.linalg.norm(edge[-2] - edge[-1])
    rel = edge - edge[-1]
    off = np.abs(rel[:, 0] * ax[1] - rel[:, 1] * ax[0])
    k = len(edge) - 1
    while k > 0 and off[k - 1] < 1e-6:
        k -= 1
    return rows, nu, np.arange(k, len(edge))


def _sheet(f: Figure, style):
    rows, nu, cols = _root_columns(style)
    M = f.piece("membrane")
    nv = len(M) // 2
    assert nv == (rows + 1) * nu and len(cols) >= 6
    return M[:nv].reshape(rows + 1, nu, 3), M[nv:].reshape(rows + 1, nu, 3), cols


@assets
@pytest.mark.parametrize("style", ["membrane", "dragon"])
@pytest.mark.parametrize("body", list(BODIES))
def test_the_sheets_inner_edge_is_under_the_skin_all_along(world, body, style):
    """The edge was a straight line from the shoulder to the waist: its lower corner hung 0.40-0.57 off the skin of
    the four bodies (daylight between the sheet and the flank) and the line lay up to 0.3 under the shoulder blade.
    It follows the back now, every point of it under the skin (0.17-0.42 on the four), from the corner to the
    shoulder, evenly along its line."""
    f = figure(world, body, style)
    top, bottom, cols = _sheet(f, style)
    edge = 0.5 * (top[-1, cols] + bottom[-1, cols])
    d = f.distance(edge)
    assert d.max() < -0.04, np.round(d, 2)                     # a7f84cd: +0.44 .. +0.57
    assert d.min() > -0.7, np.round(d, 2)
    step = np.linalg.norm(np.diff(edge, axis=0), axis=1)
    assert step.max() < 2.2 * step.min(), np.round(step, 2)    # (along each column's own ray: 5 of 8 at the corner)


@assets
@pytest.mark.parametrize("body", list(BODIES))
def test_the_sheet_leaves_the_back_in_a_fillet(world, body):
    """A blade stood on the back, 0.17-0.21 thick where it met the skin.  Each face leaves the sheet's plane along
    a quarter circle now: the sheet is 0.22-0.40 thick where its middle meets the skin on the four middle columns
    of the edge (their median 0.32-0.36; the fillet runs out toward the waist corner and the shoulder), and under
    the skin it is four to five times its own thickness (the lip, which is the figure's own flesh once the shells are joined).  (A small
    fillet: the faces still cross the skin at 55-65 degrees, against the blade's 68-82.  A larger one, 0.14 of the
    wing's scale for 0.10, notched the line where the sheet meets the back.)"""
    f = figure(world, body, "membrane")
    top, bottom, cols = _sheet(f, "membrane")
    thick = np.linalg.norm(top - bottom, axis=2)
    mid = 0.5 * (top + bottom)
    at_skin = []
    for c in cols[2:6]:
        d = f.distance(mid[:, c])
        i = int(np.argmax(d < 0.0))
        assert i > 0 and d[i] < 0.0, (c, np.round(d, 2))
        t = d[i - 1] / (d[i - 1] - d[i])
        at_skin.append(thick[i - 1, c] + t * (thick[i, c] - thick[i - 1, c]))
    assert np.median(at_skin) > 0.29, np.round(at_skin, 2)       # a7f84cd: 0.17 .. 0.21
    assert thick[:, cols].max() > 3.0 * 0.22 * (0.8 if body == "child" else 1.0)


@assets
@pytest.mark.parametrize("style", ["membrane", "dragon"])
@pytest.mark.parametrize("body", list(BODIES))
def test_no_lip_of_the_fillet_comes_back_out_of_the_skin(world, body, style):
    """What is under the skin stays under it.  A sheet meets the back at a slant, so the face that leans away from
    it is higher over the skin than the sheet's middle: sized by the middle's distance, its lip stood 0.14 over the
    heavy man's skin and came out of his back in torn flaps (22 vertices of it and more, up to 0.2 out).  Each face
    is sized by its own height, and a lip the plane-under-each-point estimate still leaves out is drawn in."""
    f = figure(world, body, style)
    top, bottom, cols = _sheet(f, style)
    near = np.arange(cols[0] - 5, cols[-1] + 1)                    # the root's columns and the corner's
    mid = 0.5 * (top[:, near] + bottom[:, near]).reshape(-1, 3)
    under = f.distance(mid) < -0.02
    out = np.concatenate([f.distance(top[:, near].reshape(-1, 3))[under],
                          f.distance(bottom[:, near].reshape(-1, 3))[under]])
    assert (out > 0.01).sum() <= 8 and out.max() < 0.09, ((out > 0.01).sum(), out.max())


@assets
def test_the_sheet_is_the_bodys_skin_where_it_grows_out_of_it(world):
    """``Piece.skin_share``: 1 at the skin, 0 in the wing's own integument, a blend between; the colour only (the
    arm's own blend is ``test_wing_root``'s)."""
    f = figure(world, "neutral", "membrane")
    piece = next(p for p in f.wing.rig.pieces if p.name == "membrane")
    share = np.asarray(piece.skin_share)
    M = f.piece("membrane")
    assert share.shape == (len(M),)
    nv = len(M) // 2
    d = f.distance(0.5 * (M[:nv] + M[nv:]))
    s = share[:nv]
    assert s[d < 0.0].min() > 0.9 and s[d > 2.5].max() == 0.0
    assert ((s > 0.1) & (s < 0.9)).sum() > 20
    assert np.array_equal(share[:nv], share[nv:])                  # both faces of the sheet
    assert 0.02 < (s > 0.5).mean() < 0.25                          # a strip along the root, not the wing


@assets
def test_the_fillet_goes_as_the_arm_swings(world):
    """The fillet is built against the back as the wing rests.  A folding wing drags the sheet across the back: the
    lips that were drawn in under the skin came out of it along the flank as a jagged flap.  The fillet is whole
    while the upper arm is within ``SWELL_SWING[0]`` degrees of its rest and gone by ``SWELL_SWING[1]``."""
    rest = figure(world, "neutral", "membrane")
    rig = rest.wing.rig
    assert rig._arm_swing({}) == 0.0
    half = figure(world, "neutral", "membrane", {"fold": 0.5})
    top0, bottom0, cols = _sheet(rest, "membrane")
    top1, bottom1, _ = _sheet(half, "membrane")
    t0 = np.linalg.norm(top0[:, cols] - bottom0[:, cols], axis=2)
    t1 = np.linalg.norm(top1[:, cols] - bottom1[:, cols], axis=2)
    assert t0.max() > 0.7 and t1.max() < 0.3                       # the sheet's own 0.22 again
    pose = rig.pose_from_controls(None, fold=0.5)
    assert rig._arm_swing(pose) == 1.0
    assert WRIG.SWELL_SWING[0] < WRIG.SWELL_SWING[1]


@assets
@pytest.mark.parametrize("controls", [None, {"fold": 0.5}, {"preset": "folded"}])
def test_the_sheet_is_one_piece_with_the_arm_in_every_pose(world, controls):
    from minimaster.character import print_report as pr
    f = figure(world, "neutral", "membrane", controls)
    ctl = dict(f.left.controls)
    pieces = dict(f.wing.piece_meshes(ctl.pop("preset", None), **ctl))
    labels, _ = pr.overlap_components([pr.Solid(m) for m in pieces.values()])
    assert len(set(labels)) == 1


# -- the feathered wing ------------------------------------------------------------------------------


def _bare_share(f: Figure, upto=1.2):
    """Share of the arm's surface on its top and leading side, over the first ``upto`` past where it leaves the skin,
    with no feather over it."""
    a, b = f.ranges["arm"]
    arm_f = f.F[(f.F >= a).all(axis=1) & (f.F < b).all(axis=1)]
    cover = f.F[~(((f.F >= a) & (f.F < b)).any(axis=1))]
    for name in ("hand", "wrist_knob", "alula_mount"):
        if name in f.ranges:
            x, y = f.ranges[name]
            cover = cover[~(((cover >= x) & (cover < y)).any(axis=1))]
    P = f.V[a:b]
    tri = f.V[arm_f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.zeros_like(f.V)
    for k in range(3):
        np.add.at(vn, arm_f[:, k], fn)
    vn = vn[a:b] / np.maximum(np.linalg.norm(vn[a:b], axis=1, keepdims=True), 1e-12)
    feathers = WE.Surface(f.V, cover, f.left.anchor, 9.0, walk=1)
    q = np.asarray(feathers.closest(P + 0.03 * vn)[0])
    covered = (np.linalg.norm(q - P, axis=1) < 0.12) & (((q - P) * vn).sum(axis=1) > -0.03)
    inv = np.linalg.inv(f.wing.matrix)
    local = P @ inv[:3, :3].T + inv[:3, 3]
    J = f.wing.rig.skeleton.joints
    S, E = np.asarray(J["shoulder"]), np.asarray(J["elbow"])
    ax = (E - S) / np.linalg.norm(E - S)
    along = (local - S) @ ax
    rad = local - S - along[:, None] * ax
    angle = np.degrees(np.arctan2(-rad[:, 1], rad[:, 2]))          # 90: the top (dorsal); +-180: the leading side
    outside = f.distance(P) > 0.02
    leaves = along[outside & (along < np.linalg.norm(E - S))].min()
    sel = (outside & (along >= leaves) & (along < leaves + upto)
           & (((angle >= 45) & (angle < 135)) | (np.abs(angle) > 135)))
    assert sel.sum() >= 20
    return float(1.0 - covered[sel].mean())


@assets
@pytest.mark.parametrize("body", list(BODIES))
def test_a_feathered_arm_leaves_the_back_under_coverts(world, body):
    """It left it bare: 27-62% of its top and leading side uncovered over the first 1.2 units on the four bodies
    (the nose row of coverts was rooted inside the muscular sleeve, 76-99% of each feather in it, and the first 0.8
    were never built).  A bird's wing meets its body under a mound of coverts (wing/A11), a sculpted angel's in rows
    of ever smaller feathers (wing/H22)."""
    assert _bare_share(figure(world, body, "feathered")) < 0.06   # a7f84cd: 0.27 .. 0.62


@assets
@pytest.mark.parametrize("body", list(BODIES))
def test_the_shoulder_coverts_are_seated_on_the_arm_and_out_of_the_body(world, body):
    """Each of the small coverts round the sleeve is rooted IN the arm (it dips at least 0.03 under its surface),
    lies along it (nowhere more than 0.25 off it: a long vane seated on a fat sleeve was a chord through it with its
    end in the air) and shows (none is wholly under the skin; one rooted inside the body is not built)."""
    f = figure(world, body, "feathered")
    a, b = f.ranges["arm"]
    arm = WE.Surface(f.V, f.F[(f.F >= a).all(axis=1) & (f.F < b).all(axis=1)], f.left.anchor, 9.0, walk=1)
    names = [n for n in f.ranges if "shoulder" in n]
    assert len(names) >= 30, len(names)                            # 36 (the child) .. 76
    for name in names:
        P = f.piece(name)
        q, _, inside, _ = arm.closest(P)
        d = np.linalg.norm(P - q, axis=1) * np.where(np.asarray(inside) < 0.0, -1.0, 1.0)
        assert d.min() < -0.03 and d.max() < 0.25, (name, d.min(), d.max())
        assert f.distance(P).max() > 0.0, name


@assets
@pytest.mark.parametrize("controls", [None, {"fold": 0.5}, {"preset": "folded"}])
def test_every_feather_is_one_piece_with_the_wing_in_every_pose(world, controls):
    """None of the new shells floats: spread, half folded and folded, the wing's 340 solids are one body."""
    from minimaster.character import print_report as pr
    f = figure(world, "neutral", "feathered", controls)
    ctl = dict(f.left.controls)
    pieces = dict(f.wing.piece_meshes(ctl.pop("preset", None), **ctl))
    assert sum("shoulder" in n for n in pieces) >= 60
    labels, _ = pr.overlap_components([pr.Solid(m) for m in pieces.values()])
    counts = collections.Counter(labels)
    assert len(counts) == 1, [n for n, l in zip(pieces, labels) if l != counts.most_common(1)[0][0]]
