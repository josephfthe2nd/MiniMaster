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

@pytest.mark.parametrize("style", STYLES)
def test_planform_is_a_wing_not_a_paddle(style):
    """Regression: fanning the digits off an already-swept forearm compounded
    the angles until digit III sat 62 degrees off the span axis. The whole
    chain's length went into chord and the wing came out square."""
    mesh = W.make_wing(style=style)
    assert span_of(mesh) / chord_of(mesh) > 1.8, "planform is too stubby"


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
    flat = W.membrane_wing(camber=0.0, droop=0.0, bones=False)
    arched = W.membrane_wing(camber=0.2, droop=0.0, bones=False)
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
    f = W._feather(5.0, 1.2, 0.2)
    rep = f.integrity_report()
    assert rep["watertight"] and rep["volume"] > 0
    # lead_frac=0.34 puts more vane on the trailing side of the shaft
    z = f.vertices[:, 2]
    assert abs(z.max()) > abs(z.min())


def test_feather_vane_stays_broad_instead_of_tapering_to_a_point():
    """Regression: a vane shaped like sin(pi*t) is widest only at its middle,
    so neighbours touch at one point and leave V-shaped gaps. Real vanes run
    nearly parallel-sided, which is what makes a wing read as shingled."""
    f = W._feather(5.0, 1.0, 0.15, curl=0.0, sweep=0.0)
    v = f.vertices
    mid = v[(v[:, 0] > 1.5) & (v[:, 0] < 3.5)]
    far = v[(v[:, 0] > 3.5) & (v[:, 0] < 4.4)]
    assert np.ptp(far[:, 2]) > 0.7 * np.ptp(mid[:, 2])


def test_feather_counts_change_the_mesh():
    few = W.feathered_wing(primaries=6, secondaries=6, coverts=False)
    many = W.feathered_wing(primaries=14, secondaries=16, coverts=False)
    assert len(many.faces) > len(few.faces)
    assert many.integrity_report()["watertight"]


def test_coverts_are_optional_and_add_material():
    bare = W.feathered_wing(coverts=False)
    full = W.feathered_wing(coverts=True)
    assert full.volume() > bare.volume()


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
