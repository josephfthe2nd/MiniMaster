"""Wings as modified forelimbs.

The contract is the same one every appendage keeps: a wing is a set of closed
solids merged into one mesh, overlapping freely, so the slicer unions them and
the per-shell print gate applies unchanged. On top of that these tests pin the
things that actually went wrong while building it -- a planform that came out
square, feathers that left gaps instead of shingling, and a mirrored pair that
was a 180-degree roll rather than a mirror.
"""

import numpy as np
import pytest

from minimaster.character import appendages as A
from minimaster.character import mhbase as mh
from minimaster.character import wings as W
from minimaster.core.mesh import Mesh

STYLES = ("membrane", "feathered")


def span_of(mesh):
    return float(np.ptp(mesh.vertices[:, 0]))


def chord_of(mesh):
    return float(np.ptp(mesh.vertices[:, 2]))


# -- the mesh primitives ----------------------------------------------------

def test_tube_is_a_capped_solid():
    pts = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0.3, 0]])
    mesh = W._tube(pts, np.array([0.2, 0.15, 0.05]), sides=8)
    rep = mesh.integrity_report()
    assert rep["watertight"] and rep["outward"] and rep["volume"] > 0


def test_slab_from_a_flat_grid_is_a_box_of_the_right_volume():
    u, v = np.meshgrid(np.linspace(0, 4, 9), np.linspace(0, 2, 5), indexing="ij")
    grid = np.stack([u, np.zeros_like(u), v], axis=2)
    mesh = W._slab_from_grid(grid, np.full(u.shape, 0.25))
    rep = mesh.integrity_report()
    assert rep["watertight"] and rep["outward"]
    assert rep["volume"] == pytest.approx(4 * 2 * 0.25, rel=0.02)


def test_slab_survives_a_curved_grid():
    """Camber and droop bend the mid-surface; the rim must still close."""
    u, v = np.meshgrid(np.linspace(0, 5, 21), np.linspace(0, 3, 11), indexing="ij")
    grid = np.stack([u, -0.4 * np.sin(np.pi * v / 3) + 0.1 * u ** 2, v], axis=2)
    mesh = W._slab_from_grid(grid, np.full(u.shape, 0.2))
    rep = mesh.integrity_report()
    assert rep["watertight"], rep
    assert rep["nonmanifold_edges"] == 0


def test_resample_is_even_and_hits_the_ends():
    line = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 0, 3.0]])
    out = W._resample(line, 9)
    assert np.allclose(out[0], line[0]) and np.allclose(out[-1], line[-1])
    steps = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert steps.std() < 1e-9


# -- both styles are printable solids ---------------------------------------

@pytest.mark.parametrize("style", STYLES)
def test_wing_is_a_printable_solid(style):
    mesh = W.make_wing(style=style)
    rep = mesh.integrity_report()
    assert rep["watertight"], f"{style}: {rep}"
    assert rep["outward"] and rep["volume"] > 0
    assert rep["nonmanifold_edges"] == 0


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("kw", [
    {}, {"span": 6.0}, {"span": 22.0}, {"fold": 1.0}, {"fold": 0.5},
    {"droop": 0.0}, {"droop": 0.4}, {"sweep": 0.0}, {"sweep": 30.0},
    {"thickness": 0.6}, {"bones": False}])
def test_wing_stays_watertight_across_the_parameter_range(style, kw):
    rep = W.make_wing(style=style, **kw).integrity_report()
    assert rep["watertight"], f"{style} {kw}: {rep}"
    assert rep["volume"] > 0


def test_unknown_style_is_rejected():
    with pytest.raises(KeyError, match="unknown wing style"):
        W.make_wing(style="jetpack")


# -- the planform -----------------------------------------------------------

def planform_area(mesh):
    """Area seen from above. A closed solid projects twice, hence the half."""
    t = mesh.vertices[mesh.faces][:, :, [0, 2]]
    u, v = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    return float((0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])).sum() / 2.0)


def test_membrane_planform_has_a_realistic_aspect_ratio():
    """Regression: fanning the digits off an already-swept forearm compounded
    the angles until digit III sat 62 degrees off the span axis, the whole
    chain's length went into chord and the wing came out square. The root now
    runs down to the hip, so a bounding-box ratio no longer says anything; the
    planform-area aspect ratio does."""
    mesh = W.make_wing(style="membrane")
    ar = 2.0 * span_of(mesh) ** 2 / planform_area(mesh)       # for the pair
    # literature gives 6-7 for flying foxes; the roost photograph, foreshortened
    # and with the wrist flexed, gives 4.7-5.8. The wrist is a visible corner
    # here, so the build sits between them. A paddle is below 3.
    assert 4.5 < ar < 8.0, f"aspect ratio {ar:.2f} is outside what a real bat shows"


# -- the spread membrane wing: scallops, billow, arm, claw ------------------------
#
# These read the finished mesh (the free-edge row, the piece rings) rather than the
# constants that produced it, and use the plan position of the root that back_wings()
# really lands on the body: with the default root straight behind the shoulder the
# trailing-edge rays never come near the arm, which hid a fold-over for several rounds.

BODY_ROOT = (-3.3, 2.3)
MEMBRANE_STYLES = ("membrane", "dragon")


@pytest.fixture(scope="module", params=MEMBRANE_STYLES)
def spread(request):
    return request.param, W.make_wing_rig(request.param, root=BODY_ROOT)


def _grid(rig, name="membrane"):
    """(piece, mid-surface grid) of the slab named ``name``."""
    piece = next(p for p in rig.pieces if p.name == name)
    a, b = rig.ranges[name]
    n1, m1 = piece.pleat["shape"]
    nv = n1 * m1
    v = rig.rest_mesh.vertices[a:a + 2 * nv]
    return piece, (0.5 * (v[:nv] + v[nv:])).reshape(n1, m1, 3)


def _ring_radii(rig, name, sides):
    """Mean radius of every ring of a ``_tube`` piece (rings first, two cap centres last)."""
    a, b = rig.ranges[name]
    v = rig.rest_mesh.vertices[a:b]
    rings = v[:(len(v) - 2) // sides * sides].reshape(-1, sides, 3)
    return np.linalg.norm(rings - rings.mean(axis=1, keepdims=True), axis=2).mean(axis=1)


def _edge_sag(grid, c0, c1):
    """Perpendicular sag of the free edge between two fingertips, over their chord."""
    e = grid[-1, c0:c1 + 1][:, [0, 2]]
    ch = e[-1] - e[0]
    n = np.array([ch[1], -ch[0]]) / np.linalg.norm(ch)
    return float(np.abs((e - e[0]) @ n).max() / np.linalg.norm(ch))


def _pillow(grid, c0, c1):
    """How far the panel between two bone columns bulges above the line joining them
    (-Y is up), at its most, as a fraction of the gap between the two bones."""
    n1 = grid.shape[0]
    best = 0.0
    for i in range(int(0.4 * (n1 - 1)), n1):
        y = grid[i, c0:c1 + 1, 1]
        line = y[0] + (y[-1] - y[0]) * np.linspace(0.0, 1.0, c1 - c0 + 1)
        best = max(best, float((line - y).max()))
    return best / float(np.linalg.norm(grid[-1, c1] - grid[-1, c0]))


def test_the_outer_scallops_are_visibly_concave_and_the_fingertips_stay_pointed():
    """The bat's III-IV edge was a nearly straight chord (sag 0.03 of the tip-to-tip
    distance); a real dactylopatagium is a shallow concave arc. The edge must also leave
    each fingertip steeply enough that it reads as a point: the IV tip was a 135 degree
    corner, it is now ~100 (the deeper arc is what sharpens it)."""
    rig = W.make_wing_rig("membrane", root=BODY_ROOT)
    piece, g = _grid(rig)
    assert _edge_sag(g, *piece.pleat["sectors"][1]) > 0.08, "III-IV is a ruled line again"
    assert _edge_sag(g, *piece.pleat["sectors"][2]) > 0.20, "IV-V lost its deep scallop"
    assert _edge_sag(g, *piece.pleat["sectors"][3]) > 0.07, "the long edge to the body is straight"
    e = g[-1][:, [0, 2]]
    for col in (piece.pleat["sectors"][1][1], piece.pleat["sectors"][2][1]):     # IV and V tips
        a, b = e[col - 1] - e[col], e[col + 1] - e[col]
        ang = np.degrees(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))
        assert ang < 112.0, f"fingertip at column {col} is a {ang:.0f} degree corner, not a point"


def test_the_dragons_edge_to_the_body_stays_clear_of_the_arm(spread):
    """At 0.30 the V-to-root arc swung in to 0.3 units of the humerus, and the thin
    web between the arm and the free edge read as a tear."""
    style, rig = spread
    piece, g = _grid(rig)
    c0, c1 = piece.pleat["sectors"][3]
    J = rig.skeleton.joints

    def dist(p, a, b):
        t = np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0.0, 1.0)
        return float(np.linalg.norm(p - (a + t * (b - a))))

    S, E, Wr = (J[k][[0, 2]] for k in ("shoulder", "elbow", "wrist"))
    clear = min(min(dist(p, S, E), dist(p, E, Wr)) for p in g[-1, c0:c1 + 1][:, [0, 2]])
    assert clear > 0.5, f"the {style} edge comes within {clear:.2f} units of the arm"
    if style == "dragon":
        assert _edge_sag(g, c0, c1) < 0.22


def test_the_membrane_billows_between_the_fingers(spread):
    """A flat developable umbrella showed no sag between the digits (the hand panels
    bulged 3-4% of the finger gap, which a render reads as flat)."""
    style, rig = spread
    piece, g = _grid(rig)
    for k in (1, 2):                                   # III-IV and IV-V
        c0, c1 = piece.pleat["sectors"][k]
        p = _pillow(g, c0, c1)
        assert 0.045 < p < 0.12, f"{style} panel {k}: pillow {p:.3f} of the gap"


def test_the_trailing_edge_rays_never_cross_the_arm(spread):
    """With the root where the body puts it (behind and below the shoulder) the elbow
    hides part of the edge from the forearm. The fan used to run its rays from the
    forearm to those points anyway, so they cut across the humerus: the arm-to-body
    sheet folded over itself and creased across the elbow."""
    style, rig = spread
    piece, g = _grid(rig)
    J = rig.skeleton.joints
    S, E, Wr = (J[k][[0, 2]] for k in ("shoulder", "elbow", "wrist"))

    def crosses(p, q, a, b):
        d1, d2 = q - p, b - a
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 1e-12:
            return False
        t = ((a[0] - p[0]) * d2[1] - (a[1] - p[1]) * d2[0]) / den
        u = ((a[0] - p[0]) * d1[1] - (a[1] - p[1]) * d1[0]) / den
        return 0.02 < t < 0.98 and 0.02 < u < 0.98

    bad = [c for c in range(piece.pleat["sectors"][3][0], g.shape[1])
           if crosses(g[0, c][[0, 2]], g[-1, c][[0, 2]], S, E)
           or crosses(g[0, c][[0, 2]], g[-1, c][[0, 2]], E, Wr)]
    assert not bad, f"rays of columns {bad} cut across the arm"


def test_the_leading_edge_has_no_step_at_the_tip_of_digit_two(spread):
    """Digit II ended 0.4 units in front of digit III's line, so the leading edge jogged
    sideways there (a 46 degree turn in one segment) before running on down digit III."""
    style, rig = spread
    J = rig.skeleton.joints
    p = J["d2_tip"][[0, 2]]
    chain = [J[k][[0, 2]] for k in ("wrist", "d3_mcp", "d3_ip", "d3_tip")]

    def dist(a, b):
        t = np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0.0, 1.0)
        return float(np.linalg.norm(p - (a + t * (b - a))))

    assert min(dist(a, b) for a, b in zip(chain[:-1], chain[1:])) < 0.25
    piece, g = _grid(rig)
    e = g[-1][:piece.pleat["sectors"][0][1] + 1][:, [0, 2]]
    head = np.degrees(np.arctan2(*np.diff(e, axis=0).T[::-1]))
    assert np.abs(np.diff(head)).max() < 38.0, np.round(head)


def test_the_humerus_tapers_into_the_shoulder_instead_of_ending_like_a_bolt(spread):
    """The arm tube was thickest at the shoulder, so a constant-looking rod ended
    abruptly against the torso. The shoulder end now tapers toward the scapula -- but never
    below the printable bone floor (0.72 x the membrane's thickness): the bone is buried 40% of
    its length and a spindle thinner than that leaves a hairline where it exits the back."""
    style, rig = spread
    r = _ring_radii(rig, "arm", 9)
    elbow = len(r) // 2
    floor = 0.72 * 0.22 * (1.35 if style == "dragon" else 1.0)
    assert r[0] < 0.88 * r[elbow], (r[0], r[elbow])
    assert r[:elbow + 1].min() >= floor - 1e-6, (r[:elbow + 1].min(), floor)
    assert r[-1] < r[elbow]


def test_the_humerus_swells_into_a_fillet_where_it_leaves_the_back(spread):
    """The taper stopped it ending like a bolt but it still entered the back as a bare rod, a
    constant 0.2-unit tube. Where it leaves the scapula (0.33-0.43 of its length from the
    shoulder, on every body morph) it swells into a pad of ~1.4 x the rod, and the swelling is a
    fairing, not a lump: never more than 1.5 x the bone, back to the rod by mid-humerus."""
    style, rig = spread
    r = _ring_radii(rig, "arm", 9)
    elbow = len(r) // 2
    t = np.linspace(0.0, 1.0, elbow + 1)
    h = r[:elbow + 1]
    assert h[(t > 0.25) & (t < 0.5)].max() > 1.25 * r[elbow], "the humerus is still a bare rod where it leaves the back"
    assert h.max() <= 1.5 * r[elbow] + 1e-9, "the fillet is a lump, not a fairing"
    assert h[(t > 0.65) & (t < 0.9)].max() < 1.05 * r[elbow], "the swelling runs on down the whole humerus"
    # the bolt fix is intact: the buried end is no thicker than the free arm
    assert r[0] < 0.88 * r[elbow]


def test_the_elbow_and_wrist_knobs_swell_the_bone_instead_of_standing_out_of_it(spread):
    """The joint balls were 1.25-1.3 x the bone's own radius, twice its diameter at figure scale:
    a rivet on a rod from above, a pendant at the hem when stowed. They are a swelling now."""
    style, rig = spread
    r = _ring_radii(rig, "arm", 9)
    for knob, ring in (("elbow_knob", r[len(r) // 2]), ("wrist_knob", r[-1])):
        a, b = rig.ranges[knob]
        v = rig.rest_mesh.vertices[a:b]
        radius = float(np.linalg.norm(v - v.mean(axis=0), axis=1).mean())
        assert 1.0 < radius / ring <= 1.2, (style, knob, radius / ring)


def test_the_dragon_is_more_than_a_bat_with_deeper_scallops():
    """Same skeleton and rig, but a dragon keeps what dragon art keeps and what survives at
    figure scale: a thorn at the wrist, a thumb ending in a claw nearly twice a bat's, and
    longer finger spikes."""
    bat = W.make_wing_rig("membrane", root=BODY_ROOT)
    drg = W.make_wing_rig("dragon", root=BODY_ROOT)
    assert "wrist_spur" not in bat.ranges and "wrist_spur" in drg.ranges

    def claw_len(rig):
        a, b = rig.ranges["claw"]
        v = rig.rest_mesh.vertices[a:b]
        return float(np.linalg.norm(v[-1] - v[-2])) / rig.skeleton.scale        # in forearms

    def spike_len(rig):
        # how far the digit's tube reaches past its tip joint, along the last phalanx
        a, b = rig.ranges["digit_III"]
        v = rig.rest_mesh.vertices[a:b]
        J = rig.skeleton.joints
        d = J["d3_tip"] - J["d3_ip"]
        d = d / np.linalg.norm(d)
        return float(((v - J["d3_tip"]) @ d).max()) / rig.skeleton.scale

    assert claw_len(drg) > 1.6 * claw_len(bat), (claw_len(drg), claw_len(bat))
    assert spike_len(drg) > 1.6 * spike_len(bat), (spike_len(drg), spike_len(bat))
    # the thorn stands out of the wrist, up out of the wing plane, and is a real spike, not a bump
    a, b = drg.ranges["wrist_spur"]
    v = drg.rest_mesh.vertices[a:b]
    wrist = drg.skeleton.joints["wrist"]
    reach = v[np.argmax(np.linalg.norm(v - wrist, axis=1))] - wrist
    assert np.linalg.norm(reach) > 0.15 * drg.skeleton.scale
    assert -reach[1] > 0.3 * np.linalg.norm(reach), "the spur lies in the wing plane instead of standing up out of it"
    # the spur is a rigid part of the forearm, like the knobs
    assert drg.dominant_bone("wrist_spur") == "wrist"


def test_the_free_edge_meets_the_body_above_the_waist_joint():
    """A full-width panel ending in a corner at the hip hangs as a flat rectangular tab beside
    the buttock at half fold. The plagiopatagium's free edge now stops short on the root line, so
    its lowest corner is a taper coming up into the flank."""
    for style in ("membrane", "dragon"):
        rig = W.make_wing_rig(style, root=BODY_ROOT)
        piece, g = _grid(rig)
        c1 = piece.pleat["sectors"][3][1]                    # the last trailing-edge column
        corner = g[-1, c1][[0, 2]]
        root = rig.skeleton.joints["root"][[0, 2]]
        shoulder = rig.skeleton.joints["shoulder"][[0, 2]]
        up = float((corner - root) @ (shoulder - root)) / float(np.linalg.norm(shoulder - root))
        assert up > 0.15 * float(np.linalg.norm(shoulder - root)), (style, up)


def test_the_thumb_claw_is_a_blunt_peg_exempt_from_the_bone_floor(spread):
    """A needle thinner than the sheet cannot print; a peg of 0.5-0.6 of its base radius,
    ending flat, can -- and it is deliberately finer than the bones' own floor."""
    style, rig = spread
    thickness = 0.22
    bone_scale = W.DRAGON["bone_scale"] if style == "dragon" else 1.0
    r = _ring_radii(rig, "claw", 8)
    a, b = rig.ranges["claw"]
    v = rig.rest_mesh.vertices[a:b]
    length = float(np.linalg.norm(v[-1] - v[-2]))                 # between the two cap centres
    fa = rig.skeleton.scale
    assert r[0] < 0.72 * thickness * bone_scale, "the claw is held to the bone floor again"
    # it grows out of the thumb, so the dragon's heavier bones carry a heavier claw too
    thumb = _ring_radii(rig, "thumb", 7)
    assert 0.5 * thumb[-1] < r[0] < 1.2 * thumb[-1], (r[0], thumb[-1])
    assert r[-1] > 0.5 * r[0], "the claw ends in a needle, not a peg"
    assert r[-1] > 0.05
    assert length > 0.10 * fa, f"the claw is a {length / fa:.2f} FA stub"


def test_finger_ridges_stay_proud_of_the_sheet_to_the_last_joint(spread):
    """The ridges tapered to half their base radius, which for IV and V is below half the
    membrane's thickness: the bone sank into the sheet and the digit read as a slit. A finger
    is one rigid tube per phalanx now; the last joint is where the last phalanx's tube
    starts its needle (ring ``FINGER_RINGS``)."""
    style, rig = spread
    thickness = 0.22
    for digit in ("IV", "V"):
        last = f"digit_{digit}_p{len(rig.skeleton.chains[digit]) - 2}"
        r = _ring_radii(rig, last, 7)
        r0 = _ring_radii(rig, f"digit_{digit}", 7)
        assert r[W.FINGER_RINGS] >= 0.5 * thickness, f"digit {digit} ridge is only {r[W.FINGER_RINGS]:.3f} thick at its last joint"
        assert r[W.FINGER_RINGS] < r0[0], "the ridge no longer tapers"


def test_the_tip_of_the_wing_is_washed_out(monkeypatch):
    """The outer wing is twisted leading-edge-down: toward the tip the chord tilts, the
    trailing edge standing higher than the leading edge it trails; inboard it does not."""
    def tilt(frac):
        rig = W.make_wing_rig("membrane", root=BODY_ROOT)
        piece, g = _grid(rig)
        J = rig.skeleton.joints
        x0 = J["wrist"][0] + frac * (J["d3_tip"][0] - J["wrist"][0])
        pts = g.reshape(-1, 3)
        pts = pts[np.abs(pts[:, 0] - x0) < 0.3]
        return float(np.degrees(np.arctan(-np.polyfit(pts[:, 2], pts[:, 1], 1)[0])))

    with_washout = [tilt(f) for f in (0.15, 0.5, 0.9)]
    monkeypatch.setattr(W, "WASHOUT", 0.0)
    without = [tilt(f) for f in (0.15, 0.5, 0.9)]
    assert with_washout[2] - without[2] > 3.0
    assert with_washout[1] - without[1] > 1.0
    assert abs(with_washout[0] - without[0]) < 0.6, "inboard of the hand nothing should twist"
    assert with_washout[1] > 1.0, "mid-wing chord should tilt trailing-up"


def test_the_sheet_has_no_ledge_between_neighbouring_columns(spread):
    """A whole-sheet guard against steps in the surface height from one grid column to
    the next. The camber weight once stepped from 0.25 to 1.0 at digit V (a 0.13-unit
    ledge down the finger; it is smoothed over nine columns now), and the single
    elbow-wedge cell of a fan that ran every ray from one arm parameter was another. A
    ledge shows as a spike in the second difference of the height across columns:
    0.154 on the old sheet, ~0.11 now."""
    style, rig = spread
    piece, g = _grid(rig)
    second = np.abs(np.diff(g[..., 1], n=2, axis=1)).max()
    assert second < 0.125, f"height steps by {second:.3f} between neighbouring columns"


def test_feathered_planform_is_long_not_square():
    mesh = W.make_wing(style="feathered")
    assert span_of(mesh) / chord_of(mesh) > 1.8, "planform is too stubby"


def test_the_hand_wing_continues_outward_instead_of_sideways():
    """Same regression, stated on the skeleton: wrist-to-tip of digit III must
    stay close to the span axis."""
    sk = W.WingSkeleton()
    d = sk.digits["III"] - sk.wrist
    assert np.degrees(np.arctan2(d[2], d[0])) < 50.0


def test_the_arm_is_a_z_with_the_wrist_forward_of_the_shoulder():
    """Every skeleton and photograph shows it: humerus aft, forearm forward, so
    the wrist is the most forward point of the planform."""
    sk = W.WingSkeleton()
    assert sk.elbow[2] > sk.shoulder[2], "the elbow sits aft of the shoulder"
    assert sk.wrist[2] < sk.shoulder[2], "the wrist sits forward of the shoulder"
    assert sk.wrist[2] == min(v[2] for k, v in sk.joints.items()
                              if k in ("shoulder", "elbow", "wrist", "d3_tip", "d4_tip", "d5_tip"))


def test_digit_three_makes_the_wingtip():
    sk = W.WingSkeleton(span=13.0)
    tips = sk.digits
    assert tips["III"][0] == max(t[0] for t in tips.values())
    assert np.allclose(sk.tip, tips["III"])


def test_digits_fan_from_the_leading_edge_round_to_the_body():
    sk = W.WingSkeleton(span=13.0)
    d = sk.digits
    # II rides the leading edge: least swept back of the four
    assert d["II"][2] == min(t[2] for t in d.values())
    # and the outboard reach falls away from III toward the body
    assert d["III"][0] > d["IV"][0] > d["V"][0]


def test_span_parameter_sets_the_actual_span():
    for want in (6.0, 13.0, 21.0):
        assert span_of(W.membrane_wing(span=want)) == pytest.approx(want, rel=0.02)


def test_the_trailing_edge_is_scalloped_not_straight():
    plain = W.membrane_wing(scallop=0.0, bones=False, droop=0.0)
    cut = W.membrane_wing(scallop=0.3, bones=False, droop=0.0)
    assert cut.volume() < plain.volume(), "scallops must remove membrane area"


def test_camber_lifts_the_membrane():
    flat = W.membrane_wing(camber=0.0, droop=0.0, gull=0.0, bones=False)
    arched = W.membrane_wing(camber=0.2, droop=0.0, gull=0.0, bones=False)
    # camber bulges toward -Y (the wing's upper surface)
    assert arched.vertices[:, 1].min() < flat.vertices[:, 1].min() - 0.2


def test_droop_bends_the_wing_down_without_changing_span():
    flat = W.membrane_wing(droop=0.0)
    bent = W.membrane_wing(droop=0.4)
    assert bent.vertices[:, 1].max() > flat.vertices[:, 1].max() + 1.0
    assert span_of(bent) == pytest.approx(span_of(flat), rel=0.02)


def test_folding_makes_the_wing_compact():
    """A folded wing swings the forearm and hand back on themselves, so it
    stops reaching sideways -- that is what makes it stowable on a mini."""
    spread = W.membrane_wing(fold=0.0)
    folded = W.membrane_wing(fold=1.0)
    assert span_of(folded) < 0.75 * span_of(spread)


def test_bones_add_volume_and_are_optional():
    with_bones = W.membrane_wing(bones=True)
    without = W.membrane_wing(bones=False)
    assert with_bones.volume() > without.volume()
    assert len(with_bones.faces) > len(without.faces)


def test_thickness_scales_the_solid():
    thin = W.membrane_wing(thickness=0.2, bones=False)
    thick = W.membrane_wing(thickness=0.5, bones=False)
    assert thick.volume() > 2.0 * thin.volume()


# -- feathers ---------------------------------------------------------------

def test_one_feather_is_a_solid_with_an_asymmetric_vane():
    f = W._blade(5.0, 1.2, 0.2)
    rep = f.integrity_report()
    assert rep["watertight"] and rep["volume"] > 0
    # lead_frac=0.34 puts more vane on the trailing side of the shaft
    z = f.vertices[:, 2]
    assert abs(z.max()) > abs(z.min())


def test_feather_vane_stays_broad_instead_of_tapering_to_a_point():
    """Regression: a vane shaped like sin(pi*t) is widest only at its middle,
    so neighbours touch at one point and leave V-shaped gaps. Real vanes run
    nearly parallel-sided, which is what makes a wing read as shingled."""
    f = W._blade(5.0, 1.0, 0.15, curl=0.0, sweep=0.0)
    v = f.vertices
    mid = v[(v[:, 0] > 1.5) & (v[:, 0] < 3.5)]
    far = v[(v[:, 0] > 3.5) & (v[:, 0] < 4.4)]
    assert np.ptp(far[:, 2]) > 0.7 * np.ptp(mid[:, 2])


def test_emargination_notches_the_leading_vane_near_the_tip():
    """Soaring primaries are slotted: the leading vane narrows sharply toward
    the tip, which is what lets the feathers spread into fingers."""
    plain = W._blade(5.0, 1.0, 0.15, curl=0.0, sweep=0.0, emarg=0.0)
    cut = W._blade(5.0, 1.0, 0.15, curl=0.0, sweep=0.0, emarg=0.7)
    def lead_width(mesh):
        v = mesh.vertices
        sel = v[(v[:, 0] > 3.0) & (v[:, 0] < 3.8)]      # the notched stretch, to near the tip
        return -sel[:, 2].min()
    assert lead_width(cut) < 0.6 * lead_width(plain)
    assert cut.integrity_report()["watertight"]


def test_the_longest_flight_feather_is_a_primary():
    sk, parts = W.feathered_pieces()
    length = {}
    for p in parts:
        if p.name[:7] in ("primary", "seconda") and p.name[-2:].isdigit() and "covert" not in p.name:
            length[p.name] = float(np.ptp(p.mesh.vertices @ np.array([1.0, 0, 0])))
    longest = max(length, key=length.get)
    assert longest.startswith("primary"), longest


def test_feather_counts_change_the_mesh():
    few = W.feathered_wing(primaries=6, secondaries=6, coverts=False)
    many = W.feathered_wing(primaries=14, secondaries=16, coverts=False)
    assert len(many.faces) > len(few.faces)
    assert many.integrity_report()["watertight"]


def test_coverts_are_optional_and_add_material():
    bare = W.feathered_wing(coverts=False)
    full = W.feathered_wing(coverts=True)
    assert full.volume() > bare.volume()


# -- the feather vocabulary ---------------------------------------------------
#
# A feather was a lens: a thin ellipsoid cross-section with a blunt cut tip, so a
# wing of them read as a heap of rice grains. These tests pin what replaced it --
# a flat printable sheet with a raised shaft, slotted outer primaries, covert tiers
# that step down geometrically, and an alula that is a visible group of its own.

@pytest.fixture(scope="module")
def plumage():
    sk, parts = W.feathered_pieces()
    return sk, {p.name: p for p in parts}


def _plan_frame(sk, name):
    """Origin, axis, perpendicular and length of a rigged feather, in plan."""
    r, t = sk.joints[f"{name}_r"], sk.joints[f"{name}_t"]
    d = (t - r)[[0, 2]]
    length = float(np.linalg.norm(d))
    d = d / length
    return r[[0, 2]], d, np.array([-d[1], d[0]]), length


def _piece_extent(mesh, axis):
    """(length, width) of a piece in plan along its own long axis (principal axis)."""
    xz = mesh.vertices[:, [0, 2]] - mesh.vertices[:, [0, 2]].mean(axis=0)
    u = xz @ axis
    w = xz @ np.array([-axis[1], axis[0]])
    return float(np.ptp(u)), float(np.ptp(w))


def _sheet_thickness(mesh, n_cols):
    """Top-to-bottom distance at every grid point of a ``_blade`` slab."""
    nv = len(mesh.vertices) // 2
    return np.linalg.norm(mesh.vertices[:nv] - mesh.vertices[nv:], axis=1).reshape(-1, n_cols)


def test_a_feather_is_a_flat_printable_sheet_with_a_shaft_ridge():
    """Regression: the vane was a lens (full thickness only on the shaft, 0.4x at the
    edge and tip), so the outer half of every vane was 0.08 units -- thinner than a
    printer resolves -- and the whole feather read as a rice grain. The vane is now a
    sheet of nearly constant thickness with a ridge standing proud of it."""
    t = 0.2
    blade = W._blade(5.0, 1.0, t, n=16)
    th = _sheet_thickness(blade, 9)
    along = blade.vertices[: len(blade.vertices) // 2].reshape(-1, 9, 3)[:, 4, 0] / 5.0
    assert th[along < 0.96].min() >= 0.75 * t             # printable all along, edge included
    assert th.min() >= 0.45 * t                           # only the last 3% is bevelled, to 60%
    mid = th[len(th) // 2]
    assert mid[0] >= 0.85 * mid[2]                        # the edge is a sheet, not a taper to nothing
    assert mid[4] >= 1.15 * mid[2]                        # ... and the shaft is a ridge on it


def test_the_shaft_ridge_stands_off_one_face_only():
    def faces(ridge_dir):
        b = W._blade(5.0, 1.0, 0.2, curl=0.0, cup=0.0, sweep=0.0, n=16, ridge_dir=ridge_dir)
        v = b.vertices.reshape(2, -1, 9, 3)
        top, bot = (v[0], v[1]) if v[0][..., 1].mean() < v[1][..., 1].mean() else (v[1], v[0])
        row = len(top) // 2
        return top[row, :, 1], bot[row, :, 1]              # y is DOWN: top is the dorsal face
    top, bot = faces(-1.0)                                  # dorsal feather: ridge on top
    assert np.ptp(bot) < 0.02 and top[4] < top[0] - 0.05
    top, bot = faces(+1.0)                                  # ventral feather: ridge below
    assert np.ptp(top) < 0.02 and bot[4] > bot[0] + 0.05


def test_every_feather_in_the_wing_is_a_printable_sheet(plumage):
    """Every flight feather and covert (not the arm tube or the wrist knob) keeps the
    sheet thickness the shaft is made from: nowhere under 0.45x (the bevelled end of a
    tip) and, over most of its area, at least 0.75x. The old lens went down to 0.03."""
    sk, pieces = plumage
    for n, p in pieces.items():
        if n in ("arm", "hand", "wrist_knob"):
            continue
        v = p.mesh.vertices
        nv = len(v) // 2
        th = np.linalg.norm(v[:nv] - v[nv:], axis=1)
        assert th.min() >= 0.45 * 0.20, (n, th.min())
        assert np.percentile(th, 30) >= 0.75 * 0.20, (n, np.percentile(th, 30))


def test_the_outer_primaries_are_slotted_and_the_inner_ones_a_closed_sheet(plumage):
    """P1-P4 overlap into one sheet; from P6 outward the leading vane is cut back at
    the notch and the gaps widen, so the last five are separate fingers with light
    showing between them. (Before: 6.9 deg between every pair and a gentle notch, so
    even P8-P9 only just touched at 85% of their length.)"""
    sk, pieces = plumage

    def gap(i, frac):
        r, d, n, length = _plan_frame(sk, f"primary{i:02d}")
        lat = []
        for k in (i, i + 1):
            v = pieces[f"primary{k:02d}"].mesh.vertices[:, [0, 2]] - r
            sel = np.abs(v @ d - frac * length) < 0.04 * length
            lat.append((v @ n)[sel])
        return float(lat[0].min() - lat[1].max())          # trailing edge of P_i - leading edge of P_i+1

    for i in (1, 2, 3):
        assert gap(i, 0.7) < -0.2, i                       # closed
    for i in (6, 7, 8, 9):
        assert gap(i, 0.85) > 0.25, i                      # slotted


def test_the_notch_cuts_the_leading_vane_back_sharply(plumage):
    """Read straight off the slab's columns (0 = leading edge, 4 = shaft): behind the
    notch the leading vane of P8-P10 is under a third of what it was in front of it."""
    sk, pieces = plumage
    for i in (8, 9, 10):
        r, d, n, length = _plan_frame(sk, f"primary{i:02d}")
        v = pieces[f"primary{i:02d}"].mesh.vertices
        top = (v[: len(v) // 2][:, [0, 2]] - r).reshape(-1, 9, 2)
        s, l = top @ d, top @ n

        def lead_half(frac):
            row = int(np.argmin(np.abs(s[:, 4] - frac * length)))
            return float(l[row, 4] - l[row, 0])
        assert lead_half(0.72) < 0.3 * lead_half(0.40), i


def test_the_covert_tiers_step_down_geometrically(plumage):
    """Greater coverts are about half a secondary, medians 30%, lessers ~15%, each a
    narrower-than-lozenge rounded strip. (The old greater coverts were 0.55 as wide as
    long and every row used the same lozenge.)"""
    sk, pieces = plumage
    ratios = {"greater": [], "median": []}
    widths = []
    for i in range(1, 15):
        r, d, n, length = _plan_frame(sk, f"secondary{i:02d}")
        for row in ratios:
            ln, wd = _piece_extent(pieces[f"{row}{i:02d}"].mesh, d)
            ratios[row].append(ln / length)
            if row == "greater":
                widths.append(wd / ln)
    assert 0.44 < np.mean(ratios["greater"]) < 0.58
    assert 0.25 < np.mean(ratios["median"]) < 0.36
    assert max(widths) < 0.46                               # a strip, not a lozenge
    sec = np.mean([_plan_frame(sk, f"secondary{i:02d}")[3] for i in range(1, 15)])
    along = np.array([0.0, 1.0])
    lesser = [np.mean([_piece_extent(pieces[n].mesh, along)[0] for n in pieces
                       if n.startswith(f"lesser{r}_")]) / sec for r in (0, 1)]
    assert 0.10 < lesser[1] < lesser[0] < 0.22
    assert np.mean(ratios["median"]) > lesser[0] + 0.05      # a visible step between every tier
    assert np.mean(ratios["greater"]) > np.mean(ratios["median"]) + 0.12


def test_the_alula_is_a_small_stiff_group_at_the_wrist_standing_off_the_edge(plumage):
    """Two or three small feathers, 25-35% of the longest primary, rooted at the wrist on
    the leading edge, angled ~25 deg off it and riding on top of the primary bases."""
    sk, pieces = plumage
    alula = [f for f in sk.feathers if f["row"] == "alula"]
    assert 2 <= len(alula) <= 3
    prim = max(float(np.linalg.norm((sk.joints[f["tip"]] - sk.joints[f["root"]])))
               for f in sk.feathers if f["row"] == "primary")
    hv = sk.joints["d3_tip"] - sk.joints["wrist"]
    hand_yaw = np.degrees(np.arctan2(hv[2], hv[0]))
    hand_len = float(np.linalg.norm(hv))
    off = []
    for f in alula:
        v = sk.joints[f["tip"]] - sk.joints[f["root"]]
        ln = float(np.linalg.norm(v))
        assert 0.25 <= ln / prim <= 0.36, ln / prim
        # rooted at the wrist end of the hand, not out along the thumb or the hand
        assert np.linalg.norm(sk.joints[f["root"]] - sk.joints["wrist"]) < 0.45 * hand_len
        off.append(hand_yaw - np.degrees(np.arctan2(v[2], v[0])))
    assert 18.0 <= max(off) <= 35.0 and min(off) >= 5.0, off   # cocked off the edge, fanned apart
    assert max(off) - min(off) >= 8.0                          # three readable feathers, not a stack
    # on top of the primary bases: higher (smaller y, y is down) than every primary root
    top_primary = min(sk.joints[f["root"]][1] for f in sk.feathers if f["row"] == "primary")
    assert all(sk.joints[f["root"]][1] < top_primary for f in alula)


def test_the_plumage_stays_inside_its_piece_and_face_budget():
    """The finer vocabulary must not be bought with a heavier mesh: the previous wing
    was 306 pieces and 98k triangles."""
    sk, parts = W.feathered_pieces()
    assert len(parts) <= 380
    assert sum(len(p.mesh.faces) for p in parts) <= 1.25 * 98156


# -- on the real body -------------------------------------------------------

def _inside_body(pts, tris):
    """Even-odd ray test along +X against a triangle soup."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    v0, v1 = c[:, 1:] - a[:, 1:], b[:, 1:] - a[:, 1:]
    d00 = (v0 * v0).sum(1)
    d01 = (v0 * v1).sum(1)
    d11 = (v1 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    ok = np.abs(den) > 1e-14
    safe = np.where(ok, den, 1.0)
    out = np.zeros(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        v2 = p[1:] - a[:, 1:]
        d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
        u = np.where(ok, (d11 * d20 - d01 * d21) / safe, -1.0)
        w = np.where(ok, (d00 * d21 - d01 * d20) / safe, -1.0)
        hit = ok & (u >= 0) & (w >= 0) & (u + w <= 1)
        if not hit.any():
            continue
        bx = (a[hit, 0] + u[hit] * (c[hit, 0] - a[hit, 0])
              + w[hit] * (b[hit, 0] - a[hit, 0]))
        out[i] = int((bx > p[0]).sum()) % 2 == 1
    return out


@pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
class TestOnRealBody:
    ANCHOR = A.WING_ANCHOR                # left scapula
    HINT = A.WING_UP_HINT                 # aims the span up and out

    def _app(self, style, **params):
        return A.Appendage("wing", anchor_point=self.ANCHOR, mirror=True,
                           up_hint=self.HINT, roll=A.WING_ROLL,
                           offset=A.WING_OFFSET,
                           params=dict(style=style, **params))

    @staticmethod
    def _root_verts(mesh, proto):
        lx = proto.vertices[:, 0]
        return mesh.vertices[lx < 0.12 * lx.max()]

    def test_roll_lays_the_root_along_the_back_instead_of_out_behind_it(self):
        """Regression: place() maps the appendage's local +Z to the surface
        normal, so without a roll a wing's chord points straight out of the
        back. The root chord is longer than the body is deep, so it hangs in
        the air behind the figure and only the anchor point touches."""
        base, lib = mh.load()
        v = lib.apply(base.verts, mh.macro_weights(gender=1.0))
        cv, _ = base.body_cage(v)
        proto = W.membrane_wing()

        def root_gap(roll):
            app = A.Appendage("wing", anchor_point=self.ANCHOR, mirror=False,
                              up_hint=self.HINT, roll=roll,
                              params=dict(style="membrane"))
            root = self._root_verts(app.build(base, v)[0][1], proto)
            d = np.linalg.norm(root[:, None, :] - cv[None, :, :], axis=2)
            return float(d.min(axis=1).max())

        assert root_gap(A.WING_ROLL) < 0.4 * root_gap(0.0)

    def test_the_root_is_welded_into_the_body_not_grazing_it(self):
        """Overlapping shells are what the slicer unions. A wing that only
        touches the skin is joined along a hairline and prints detached, so
        the root has to be genuinely buried in the torso."""
        base, lib = mh.load()
        v = lib.apply(base.verts, mh.macro_weights(gender=1.0))
        cv, cq = base.body_cage(v)
        tris = cv[np.vstack([cq[:, [0, 1, 2]], cq[:, [0, 2, 3]]])]
        proto = W.membrane_wing()
        root = self._root_verts(A.back_wings("membrane").build(base, v)[0][1],
                                proto)
        buried = _inside_body(root[::4], tris)
        assert buried.mean() > 0.2, (
            f"only {100 * buried.mean():.0f}% of the wing root is inside the body")

    @pytest.mark.parametrize("style", STYLES)
    def test_placed_wings_are_printable(self, style):
        base, lib = mh.load()
        v = lib.apply(base.verts, mh.macro_weights(gender=1.0))
        built = self._app(style).build(base, v)
        assert len(built) == 2
        for name, mesh in built:
            rep = mesh.integrity_report()
            assert rep["watertight"], f"{name}: {rep}"
            assert rep["volume"] > 0

    @pytest.mark.parametrize("style", STYLES)
    def test_a_mirrored_pair_is_a_true_mirror_image(self, style):
        """Regression: with a mirrored up_hint, negating local X on top of the
        already-reflected frame cancels into a 180-degree roll. The anchor also
        has to bind to the WHOLE quad -- its first three corners differ between
        a face and its mirror, which tilts one wing by a degree or two."""
        base = mh.BaseMesh()
        (_, left), (_, right) = self._app(style).build(base)
        mirrored = left.vertices @ np.diag([-1.0, 1.0, 1.0]).T
        assert np.abs(mirrored - right.vertices).max() < 1e-9

    def test_wings_follow_the_body_through_a_morph(self):
        base, lib = mh.load()
        app = self._app("membrane")
        small = app.build(base, lib.apply(base.verts,
                                          mh.macro_weights(gender=0.0, height=0.0)))
        large = app.build(base, lib.apply(base.verts,
                                          mh.macro_weights(gender=1.0, height=1.0,
                                                           muscle=1.0)))
        a = small[0][1].vertices.mean(axis=0)
        b = large[0][1].vertices.mean(axis=0)
        assert not np.allclose(a, b), "the wing root must track the shoulder"

    @pytest.mark.parametrize("style", STYLES)
    def test_a_winged_character_exports_through_the_print_gate(self, style):
        from minimaster.character import printprep as pp
        base, lib = mh.load()
        v = lib.apply(base.verts, mh.macro_weights(gender=1.0))
        shells = pp.character_print_shells(base, v)
        shells += self._app(style).build(base, v)
        mesh, rep = pp.assemble_character(shells, size="medium")
        assert rep.watertight
        assert rep.shells >= 6
        assert len(mesh.faces) > 0


def test_the_hand_panel_has_visible_camber(spread):
    """Mutation check: HAND_CAMBER 0 or the old 0.25 left every other test green. The mean
    arch of the hand panel's rays (offset from their own chord, over the ray length) is
    ~1.9% on the bat and ~1.6% on the dragon; 0.25 gave 1.0% / 0.9%, 0 gave 0.65%."""
    style, rig = spread
    piece, g = _grid(rig)
    offs = []
    for j in range(piece.pleat["sectors"][1][1]):
        ray = g[:, j]
        chord = ray[0] + (ray[-1] - ray[0]) * np.linspace(0.0, 1.0, len(ray))[:, None]
        offs.append(np.abs((ray - chord)[:, 1]).max() / np.linalg.norm(ray[-1] - ray[0]))
    assert np.mean(offs) > 0.0135, (style, np.mean(offs))


@pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
def test_the_tests_BODY_ROOT_is_near_where_back_wings_really_puts_it():
    """The isolated-wing tests use a fixed plan root; it must stay roughly where the real
    placement lands, or they would keep checking a planform nobody builds."""
    from minimaster.character import appendages as A
    base, lib = mh.load()
    v = lib.apply(base.verts, mh.macro_weights(gender=1.0))
    for style in MEMBRANE_STYLES:
        app = A.back_wings(style)
        (_, left), _ = app.build_rigs(base, v)
        r = np.linalg.inv(left.matrix) @ np.append(np.asarray(app.root_point, dtype=np.float64), 1.0)
        assert np.hypot(r[0] - BODY_ROOT[0], r[2] - BODY_ROOT[1]) < 1.0, (style, r)
