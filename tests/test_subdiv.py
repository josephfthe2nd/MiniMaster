"""Catmull-Clark kernel: closure, winding, counts, and limit behaviour."""

import numpy as np
import pytest

from minimaster.core import subdiv


def test_cube_one_level_counts_and_integrity():
    v, f = subdiv.cube_cage(2.0)
    v1, f1 = subdiv.subdivide_quads(v, f, 1)
    # V' = V + F + E, F' = 4F for an all-quad cage
    assert len(v1) == 8 + 6 + 12
    assert len(f1) == 24
    rep = subdiv.quads_to_mesh(v1, f1).integrity_report()
    assert rep["watertight"] and rep["outward"]


@pytest.mark.parametrize("levels", [1, 2, 3])
def test_cube_stays_watertight_outward_at_depth(levels):
    v, f = subdiv.cube_cage(2.0)
    mesh = subdiv.subdivided_mesh(v, f, levels)
    rep = mesh.integrity_report()
    assert rep["watertight"] and rep["outward"], rep
    # subdivision shrinks a convex cage toward its limit surface: volume
    # decreases but stays well above zero
    assert 0.0 < mesh.volume() < 8.0


def test_limit_volume_converges():
    v, f = subdiv.cube_cage(2.0)
    vols = [subdiv.subdivided_mesh(v, f, n).volume() for n in (1, 2, 3, 4)]
    diffs = [abs(vols[i + 1] - vols[i]) for i in range(3)]
    assert diffs[2] < diffs[0]  # converging


def test_open_cage_is_rejected():
    v, f = subdiv.cube_cage(1.0)
    with pytest.raises(ValueError, match="closed 2-manifold"):
        subdiv.subdivide_quads(v, f[:-1], 1)  # drop one face -> boundary


def test_loft_rings_closed_tube():
    th = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)
    rings = [np.stack([np.cos(th), np.sin(th), np.full(8, z)], axis=1)
             for z in (0.0, 1.0, 2.0, 3.0)]
    v, f = subdiv.loft_rings(rings)
    rep = subdiv.quads_to_mesh(v, f).integrity_report()
    assert rep["watertight"] and rep["outward"], rep
    mesh = subdiv.subdivided_mesh(v, f, 2)
    rep2 = mesh.integrity_report()
    assert rep2["watertight"] and rep2["outward"], rep2


def test_loft_rings_odd_count_rejected():
    th = np.linspace(0.0, 2 * np.pi, 7, endpoint=False)
    rings = [np.stack([np.cos(th), np.sin(th), np.full(7, z)], axis=1)
             for z in (0.0, 1.0)]
    with pytest.raises(ValueError, match="even"):
        subdiv.loft_rings(rings)


def test_subdivision_is_deterministic():
    v, f = subdiv.cube_cage(2.0)
    a = subdiv.subdivided_mesh(v, f, 2)
    b = subdiv.subdivided_mesh(v, f, 2)
    assert np.array_equal(a.vertices, b.vertices)
    assert np.array_equal(a.faces, b.faces)
