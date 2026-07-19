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


def test_axis_angle_matches_primitive_rotations():
    assert np.allclose(m3.axis_angle([1, 0, 0], 37), m3.rot_x(37))
    assert np.allclose(m3.axis_angle([0, 1, 0], -71), m3.rot_y(-71))
    assert np.allclose(m3.axis_angle([0, 0, 2], 123), m3.rot_z(123))
    with pytest.raises(ValueError):
        m3.axis_angle([0, 0, 0], 10)


def test_rotation_between_cases():
    rng = np.random.default_rng(11)
    for _ in range(100):
        a = rng.normal(size=3)
        b = rng.normal(size=3)
        if np.linalg.norm(a) < 1e-3 or np.linalg.norm(b) < 1e-3:
            continue
        r = m3.rotation_between(a, b)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(r) == pytest.approx(1.0)
        mapped = r @ (a / np.linalg.norm(a))
        assert np.allclose(mapped, b / np.linalg.norm(b), atol=1e-9)
    # parallel and anti-parallel
    assert np.allclose(m3.rotation_between([0, 0, 1], [0, 0, 2]), np.eye(3))
    r = m3.rotation_between([0, 0, 1], [0, 0, -1])
    assert np.allclose(r @ np.array([0, 0, 1.0]), [0, 0, -1.0], atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)
    # near-anti-parallel stays stable
    r = m3.rotation_between([0, 0, 1], [1e-8, 0, -1])
    assert np.allclose(r @ np.array([0, 0, 1.0]), [1e-8, 0, -1] / np.linalg.norm([1e-8, 0, -1]), atol=1e-6)


def test_matrix_to_euler_round_trip():
    rng = np.random.default_rng(23)
    for _ in range(300):
        angles = rng.uniform(-180, 180, 3)
        r = m3.euler_rotation(*angles)
        back = m3.euler_rotation(*m3.matrix_to_euler_xyz(r))
        assert np.allclose(back, r, atol=1e-9)


@pytest.mark.parametrize("ry", [90.0, -90.0])
def test_matrix_to_euler_gimbal_lock(ry):
    for rx in (0.0, 30.0, -120.0):
        for rz in (0.0, 45.0):
            r = m3.euler_rotation(rx, ry, rz)
            back = m3.euler_rotation(*m3.matrix_to_euler_xyz(r))
            assert np.allclose(back, r, atol=1e-9)
