import numpy as np
import pytest

from minimaster.core import math3d as m3
from minimaster.core.primitives import box, icosphere
from minimaster.core.raycast import ray_mesh_hit, raycast_meshes, screen_ray


def test_hit_front_face_of_box():
    b = box(2, 2, 2)
    hit = ray_mesh_hit([0, -10, 0], [0, 1, 0], b)
    assert hit is not None
    assert hit.t == pytest.approx(9.0)
    assert np.allclose(hit.point, [0, -1, 0])
    assert np.allclose(hit.normal, [0, -1, 0])  # outward normal of the face hit


def test_miss():
    b = box(1, 1, 1)
    assert ray_mesh_hit([0, -10, 5], [0, 1, 0], b) is None
    # ray pointing away
    assert ray_mesh_hit([0, -10, 0], [0, -1, 0], b) is None


def test_nearest_of_two_shells():
    near = box(1, 1, 1)
    far = box(1, 1, 1).translated([0, 5, 0])
    key, hit = raycast_meshes([("far", far), ("near", near)], [0, -10, 0], [0, 1, 0])
    assert key == "near"
    assert hit.t == pytest.approx(9.5)
    assert raycast_meshes([("a", near)], [0, -10, 5], [0, 1, 0]) is None


def test_hit_sphere_normal_points_out():
    s = icosphere(1.0, 2)
    hit = ray_mesh_hit([0, 0, 10], [0, 0, -1], s)
    assert hit is not None
    assert hit.point[2] == pytest.approx(1.0, abs=0.02)  # faceted sphere
    assert hit.normal @ np.array([0, 0, 1.0]) > 0.9


def test_zero_direction_rejected():
    with pytest.raises(ValueError):
        ray_mesh_hit([0, 0, 0], [0, 0, 0], box(1, 1, 1))


def test_screen_ray_inverts_projection():
    """Project a world point exactly as viewport._project does, then unproject
    the pixel: the resulting ray must pass through the original point."""
    rng = np.random.default_rng(7)
    w, h = 800, 600
    for _ in range(50):
        eye = rng.uniform(-50, 50, 3)
        target = rng.uniform(-10, 10, 3)
        if np.linalg.norm(eye - target) < 1.0:
            continue
        view = m3.look_at(eye, target)
        focal = (h / 2.0) / np.tan(np.radians(30.0) / 2.0)
        point = target + rng.uniform(-5, 5, 3)
        cam = m3.transform_points(view, point.reshape(1, 3))[0]
        depth = -cam[2]
        if depth < 1.0:
            continue
        sx = w / 2.0 + focal * cam[0] / depth
        sy = h / 2.0 - focal * cam[1] / depth
        origin, direction = screen_ray(sx, sy, w, h, view, focal)
        assert np.allclose(origin, eye, atol=1e-9)
        to_point = point - origin
        dist = np.linalg.norm(np.cross(direction, to_point))
        assert dist < 1e-6, f"ray misses projected point by {dist}"
