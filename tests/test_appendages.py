"""Extra arms, wings, tails and horns.

The contract: every appendage is its own CLOSED watertight shell that overlaps
the body, so the base mesh's vertex numbering (which all 1,280 morph targets
depend on) is never disturbed, and the existing per-shell print gate applies
unchanged.
"""

import numpy as np
import pytest

from minimaster.character import appendages as A
from minimaster.character import mhbase as mh
from minimaster.core.mesh import Mesh


# -- boundary tracing / capping (no assets needed) -------------------------

def _open_box_quads():
    """A cube missing its top face: one boundary loop of 4."""
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    q = np.array([[3, 2, 1, 0], [0, 1, 5, 4], [1, 2, 6, 5],
                  [2, 3, 7, 6], [3, 0, 4, 7]], dtype=np.int64)
    return v, q


def test_boundary_loop_found_and_capped():
    v, q = _open_box_quads()
    loops = A.boundary_loops(q)
    assert len(loops) == 1 and len(loops[0]) == 4
    cv, cf = A.cap_boundaries(v, q)
    mesh = Mesh(cv, cf)
    rep = mesh.integrity_report()
    assert rep["watertight"], rep
    assert rep["volume"] > 0        # cap winding matches the surrounding faces


def test_closed_mesh_has_no_boundary():
    v, q = _open_box_quads()
    cv, cf = A.cap_boundaries(v, q)
    assert A.boundary_loops(cf) == []


def test_two_separate_holes_both_get_capped():
    """A tube open at both ends -> two loops, both closed."""
    th = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    rings = [np.stack([np.cos(th), np.sin(th), np.full(8, z)], axis=1)
             for z in (0.0, 1.0, 2.0)]
    v = np.vstack(rings)
    q = []
    for i in range(2):
        b0, b1 = i * 8, (i + 1) * 8
        for k in range(8):
            kn = (k + 1) % 8
            q.append([b0 + k, b0 + kn, b1 + kn, b1 + k])
    q = np.asarray(q, dtype=np.int64)
    assert len(A.boundary_loops(q)) == 2
    cv, cf = A.cap_boundaries(v, q)
    assert Mesh(cv, cf).integrity_report()["watertight"]


def test_largest_component_drops_stray_islands():
    v, q = _open_box_quads()
    stray = q[:1] + 100          # a disconnected face using far-away indices
    both = np.vstack([q, stray])
    kept = A.largest_component(both)
    assert len(kept) == len(q)


# -- procedural appendages --------------------------------------------------

@pytest.mark.parametrize("name,builder", [
    ("tail", A.make_tail), ("horn", A.make_horn), ("wing", A.make_wing)])
def test_procedural_appendage_is_a_printable_solid(name, builder):
    mesh = builder()
    rep = mesh.integrity_report()
    assert rep["watertight"], f"{name}: {rep}"
    assert rep["outward"] and rep["volume"] > 0


def test_tail_length_and_taper_respond_to_params():
    short = A.make_tail(length=3.0)
    long = A.make_tail(length=9.0)
    span = lambda m: np.linalg.norm(m.bounds[1] - m.bounds[0])
    assert span(long) > span(short)
    assert long.integrity_report()["watertight"]


def test_unknown_kind_is_rejected():
    app = A.Appendage("gizzard")
    with pytest.raises(KeyError, match="unknown appendage kind"):
        app.build(_FakeBase())


class _FakeBase:
    """Minimal stand-in so the error path needs no assets."""
    verts = np.zeros((3, 3))

    def body_cage(self, verts=None):
        return np.zeros((3, 3)), np.array([[0, 1, 2, 0]], dtype=np.int64)


# -- the surface anchor -----------------------------------------------------

def test_anchor_follows_the_surface_it_is_bound_to():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    f = np.array([[0, 1, 2]], dtype=np.int64)
    a = A.SurfaceAnchor.from_point(v, f, [0.3, 0.3, 0.0])
    o1, b1 = a.resolve(v)
    # move the whole triangle: the anchor must move with it
    o2, b2 = a.resolve(v + np.array([5.0, 0.0, 0.0]))
    assert np.allclose(o2 - o1, [5.0, 0.0, 0.0])
    assert np.allclose(b1, b2)
    # scale it: the anchor tracks the new geometry rather than staying put
    o3, _ = a.resolve(v * 2.0)
    assert not np.allclose(o3, o1)


def test_anchor_basis_is_orthonormal_and_outward():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    f = np.array([[0, 1, 2]], dtype=np.int64)
    _, basis = A.SurfaceAnchor.from_point(v, f, [0.3, 0.3, 0.0]).resolve(v)
    assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-9)
    assert np.isclose(abs(np.linalg.det(basis)), 1.0)


def test_place_positions_and_scales():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    f = np.array([[0, 1, 2]], dtype=np.int64)
    a = A.SurfaceAnchor.from_point(v, f, [0.3, 0.3, 0.0])
    horn = A.make_horn()
    placed = A.place(horn, a, v, scale=2.0)
    assert placed.integrity_report()["watertight"]
    big = np.linalg.norm(placed.bounds[1] - placed.bounds[0])
    small = np.linalg.norm(A.place(horn, a, v, scale=1.0).bounds[1]
                           - A.place(horn, a, v, scale=1.0).bounds[0])
    assert big == pytest.approx(2 * small, rel=1e-6)


# -- against the real base mesh --------------------------------------------

@pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
class TestOnRealBody:
    CHAIN = ["l-shoulder", "l-elbow", "l-hand"]

    def test_joint_markers_move_with_the_morphs(self):
        base, lib = mh.load()
        a = A.joint_position(base, "l-shoulder")
        big = lib.apply(base.verts, mh.macro_weights(gender=1.0, height=1.0))
        b = A.joint_position(base, "l-shoulder", big)
        assert not np.allclose(a, b), "markers must track the body"

    @pytest.mark.parametrize("kw", [
        {}, {"gender": 0.0}, {"gender": 1.0, "weight": 1.0},
        {"age": 0.1875}, {"gender": 1.0, "muscle": 0.85, "weight": 0.45},
        {"gender": 1.0, "muscle": 1.0, "weight": 1.0, "height": 1.0}])
    def test_harvested_arm_is_watertight_under_every_morph(self, kw):
        base, lib = mh.load()
        v = lib.apply(base.verts, mh.macro_weights(**kw))
        mesh = A.harvest_region(base, self.CHAIN, v, radius=1.15, trim=0.6)
        rep = mesh.integrity_report()
        assert rep["watertight"], f"{kw}: {rep}"
        assert rep["outward"]
        assert rep["nonmanifold_edges"] == 0

    def test_harvested_leg_is_watertight(self):
        base, lib = mh.load()
        v = lib.apply(base.verts, mh.macro_weights(gender=1.0))
        mesh = A.harvest_region(
            base, ["l-upper-leg", "l-knee", "l-ankle"], v, radius=1.5, trim=0.6)
        assert mesh.integrity_report()["watertight"]

    def test_harvest_tracks_the_body_size(self):
        base, lib = mh.load()
        small = lib.apply(base.verts, mh.macro_weights(gender=0.0, height=0.0))
        large = lib.apply(base.verts, mh.macro_weights(gender=1.0, height=1.0,
                                                       muscle=1.0))
        a = A.harvest_region(base, self.CHAIN, small, radius=1.3, trim=0.6)
        b = A.harvest_region(base, self.CHAIN, large, radius=1.3, trim=0.6)
        assert b.volume() > a.volume()

    def test_too_small_a_radius_fails_loudly(self):
        base = mh.BaseMesh()
        with pytest.raises(ValueError, match="selected no faces"):
            A.harvest_region(base, self.CHAIN, radius=0.001)

    def test_full_creature_every_shell_printable(self):
        base, lib = mh.load()
        v = lib.apply(base.verts, mh.macro_weights(gender=1.0, muscle=0.8))
        apps = [
            A.Appendage("wing", anchor_point=(1.5, 5.4, -1.0), mirror=True),
            A.Appendage("tail", anchor_point=(0.0, 1.6, -1.15)),
            A.Appendage("horn", anchor_point=(0.85, 8.55, 0.05), mirror=True),
            A.Appendage("harvest", anchor_point=(2.0, 4.2, 0.15), mirror=True,
                        chain=tuple(CHAIN_L := ("l-shoulder", "l-elbow", "l-hand")),
                        params=dict(radius=1.15, trim=0.6)),
        ]
        built = [(n, m) for a in apps for n, m in a.build(base, v)]
        assert len(built) == 7          # 2 wings + tail + 2 horns + 2 arms
        for name, mesh in built:
            rep = mesh.integrity_report()
            assert rep["watertight"], f"{name}: {rep}"
            assert rep["volume"] > 0, name

    def test_creature_exports_through_the_print_gate(self):
        from minimaster.character import printprep as pp
        base, lib = mh.load()
        v = lib.apply(base.verts, mh.macro_weights(gender=1.0))
        shells = pp.character_print_shells(base, v)
        for app in (A.Appendage("tail", anchor_point=(0.0, 1.6, -1.15)),
                    A.Appendage("horn", anchor_point=(0.85, 8.55, 0.05),
                                mirror=True)):
            shells += app.build(base, v)
        mesh, rep = pp.assemble_character(shells, size="medium")
        assert rep.watertight
        assert rep.shells >= 6
