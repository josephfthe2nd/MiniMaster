import numpy as np
import pytest

from minimaster.core import math3d as m3
from minimaster.core.mesh import Mesh
from minimaster.core.primitives import box, icosphere


def test_box_watertight_and_volume():
    b = box(2.0, 3.0, 4.0)
    rep = b.integrity_report()
    assert rep["watertight"], rep
    assert rep["volume"] == pytest.approx(24.0)
    lo, hi = b.bounds
    assert np.allclose(lo, [-1.0, -1.5, -2.0])
    assert np.allclose(hi, [1.0, 1.5, 2.0])


def test_convex_faces_point_outward():
    b = box(1.0, 1.0, 1.0)
    normals = b.face_normals()
    centers = b.triangles().mean(axis=1)
    assert (np.einsum("ij,ij->i", normals, centers) > 0).all()


def test_transform_preserves_orientation_under_mirror():
    b = box(1.0, 1.0, 1.0)
    mirrored = b.transform(m3.scale_mat([-1.0, 1.0, 1.0]))
    rep = mirrored.integrity_report()
    assert rep["watertight"]
    assert rep["volume"] == pytest.approx(1.0)

    also = b.scaled([-1.0, 1.0, 1.0])
    assert also.integrity_report()["volume"] == pytest.approx(1.0)


def test_nonuniform_scale_keeps_watertight():
    s = icosphere(0.5, 1).transform(m3.scale_mat([3.0, 1.0, 0.25]))
    rep = s.integrity_report()
    assert rep["watertight"]
    assert rep["volume"] > 0


def test_merge_of_shells_stays_watertight():
    a = box(1, 1, 1)
    b = box(1, 1, 1).translated([0.5, 0.0, 0.0])
    merged = Mesh.merge([a, b])
    rep = merged.integrity_report()
    assert rep["watertight"]  # two disjoint index ranges, each closed
    assert rep["volume"] == pytest.approx(2.0)
    assert len(merged.faces) == 24


def test_open_mesh_detected():
    b = box(1, 1, 1)
    opened = Mesh(b.vertices, b.faces[:-1])
    rep = opened.integrity_report()
    assert not rep["watertight"]
    assert rep["boundary_edges"] == 3


def test_degenerate_face_detected():
    b = box(1, 1, 1)
    faces = np.vstack([b.faces, [[0, 0, 1]]])
    bad = Mesh(b.vertices, faces)
    assert len(bad.degenerate_faces()) == 1
    assert not bad.integrity_report()["watertight"]


def test_face_index_out_of_range_rejected():
    with pytest.raises(ValueError):
        Mesh(np.zeros((3, 3)), np.array([[0, 1, 3]]))


def test_empty_merge():
    m = Mesh.merge([])
    assert len(m.faces) == 0
    assert not m.integrity_report()["watertight"]
