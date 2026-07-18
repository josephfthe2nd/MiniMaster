import numpy as np
import pytest

from minimaster.core import math3d as m3


def test_rot_z_maps_x_to_y():
    p = m3.rot_z(90) @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(p, [0.0, 1.0, 0.0], atol=1e-12)


def test_rot_x_maps_y_to_z():
    p = m3.rot_x(90) @ np.array([0.0, 1.0, 0.0])
    assert np.allclose(p, [0.0, 0.0, 1.0], atol=1e-12)


def test_rot_y_maps_z_to_x():
    p = m3.rot_y(90) @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(p, [1.0, 0.0, 0.0], atol=1e-12)


def test_euler_order_x_first():
    r = m3.euler_rotation(90, 0, 90)
    expected = m3.rot_z(90) @ m3.rot_x(90)
    assert np.allclose(r, expected)


def test_compose_trs_matches_manual():
    t, r, s = [1, 2, 3], [10, 20, 30], [2, 3, 4]
    m = m3.compose_trs(t, r, s)
    manual = (
        m3.translation_mat(t)
        @ m3.mat4(rotation=m3.euler_rotation(*r))
        @ m3.scale_mat(s)
    )
    assert np.allclose(m, manual)
    pts = np.array([[1.0, 1.0, 1.0]])
    out = m3.transform_points(m, pts)
    manual_pt = m3.euler_rotation(*r) @ (np.array([1.0, 1.0, 1.0]) * [2, 3, 4]) + [1, 2, 3]
    assert np.allclose(out[0], manual_pt)


def test_point_segment_distance():
    a, b = [0, 0, 0], [10, 0, 0]
    assert m3.point_segment_distance([5, 3, 0], a, b) == pytest.approx(3.0)
    assert m3.point_segment_distance([-4, 0, 0], a, b) == pytest.approx(4.0)
    assert m3.point_segment_distance([13, 0, 4], a, b) == pytest.approx(5.0)
    # degenerate segment
    assert m3.point_segment_distance([3, 4, 0], a, a) == pytest.approx(5.0)


def test_look_at_centers_target():
    view = m3.look_at([10, 10, 10], [0, 0, 0])
    p = m3.transform_points(view, np.array([[0.0, 0.0, 0.0]]))[0]
    assert p[0] == pytest.approx(0.0, abs=1e-12)
    assert p[1] == pytest.approx(0.0, abs=1e-12)
    assert p[2] == pytest.approx(-np.sqrt(300), rel=1e-12)  # in front (-Z)
