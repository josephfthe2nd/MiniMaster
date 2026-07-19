"""Small 3D math helpers on numpy arrays.

Matrices are 4x4 homogeneous (row-vector-free: points transform as M @ [x,y,z,1]).
Rotations passed as euler angles use degrees and XYZ order (X applied first):
``R = Rz @ Ry @ Rx``.
"""

from __future__ import annotations

import numpy as np


def vec3(x: float, y: float, z: float) -> np.ndarray:
    return np.array([x, y, z], dtype=np.float64)


def rot_x(degrees: float) -> np.ndarray:
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(degrees: float) -> np.ndarray:
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(degrees: float) -> np.ndarray:
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def euler_rotation(rx: float, ry: float, rz: float) -> np.ndarray:
    """3x3 rotation from XYZ euler degrees (X applied first)."""
    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)


def mat4(rotation: np.ndarray | None = None, translation=None) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    if rotation is not None:
        m[:3, :3] = rotation
    if translation is not None:
        m[:3, 3] = translation
    return m


def translation_mat(v) -> np.ndarray:
    return mat4(translation=np.asarray(v, dtype=np.float64))


def scale_mat(s) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    if s.ndim == 0:
        s = np.array([s, s, s])
    return mat4(rotation=np.diag(s))


def compose_trs(translation, rotation_deg, scale) -> np.ndarray:
    """T @ R @ S from position, XYZ euler degrees, and per-axis scale."""
    return (
        translation_mat(translation)
        @ mat4(rotation=euler_rotation(*np.asarray(rotation_deg, dtype=np.float64)))
        @ scale_mat(scale)
    )


def transform_points(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 to an (n, 3) array of points."""
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ m[:3, :3].T + m[:3, 3]


def point_segment_distance(p, a, b) -> float:
    """Distance from point p to segment a-b."""
    p, a, b = (np.asarray(v, dtype=np.float64) for v in (p, a, b))
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-18:
        return float(np.linalg.norm(p - a))
    t = np.clip((p - a) @ ab / denom, 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def axis_angle(axis, degrees: float) -> np.ndarray:
    """3x3 rotation of ``degrees`` about an arbitrary axis (Rodrigues)."""
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        raise ValueError("axis must be non-zero")
    x, y, z = axis / n
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    cc = 1.0 - c
    return np.array(
        [
            [c + x * x * cc, x * y * cc - z * s, x * z * cc + y * s],
            [y * x * cc + z * s, c + y * y * cc, y * z * cc - x * s],
            [z * x * cc - y * s, z * y * cc + x * s, c + z * z * cc],
        ]
    )


def rotation_between(a, b) -> np.ndarray:
    """3x3 rotation taking direction a onto direction b (Rodrigues).

    Degenerate cases: parallel vectors give the identity; anti-parallel
    vectors give a 180-degree flip about an arbitrary perpendicular axis.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    s2 = float(v @ v)
    if s2 < 1e-16:
        if c > 0:
            return np.eye(3)
        # anti-parallel: rotate 180 degrees about any axis perpendicular to a
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if axis @ axis < 1e-12:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64
    )
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / s2)


def matrix_to_euler_xyz(r: np.ndarray) -> tuple[float, float, float]:
    """Inverse of :func:`euler_rotation`: R = Rz(rz) @ Ry(ry) @ Rx(rx),
    angles in degrees."""
    r = np.asarray(r, dtype=np.float64)
    sy = float(np.clip(-r[2, 0], -1.0, 1.0))
    if abs(sy) < 1.0 - 1e-9:
        ry = np.arcsin(sy)
        rx = np.arctan2(r[2, 1], r[2, 2])
        rz = np.arctan2(r[1, 0], r[0, 0])
    else:  # gimbal lock: only rx +- rz is determined, pick rz = 0
        ry = np.pi / 2.0 * np.sign(sy)
        rz = 0.0
        if sy > 0:
            rx = np.arctan2(r[0, 1], r[0, 2])
        else:
            rx = np.arctan2(-r[0, 1], -r[0, 2])
    return (float(np.degrees(rx)), float(np.degrees(ry)), float(np.degrees(rz)))


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """World-to-camera 4x4. Camera looks down -Z, +X right, +Y up."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    fwd = target - eye
    n = np.linalg.norm(fwd)
    if n < 1e-12:
        raise ValueError("look_at: eye and target coincide")
    fwd = fwd / n
    right = np.cross(fwd, up)
    rn = np.linalg.norm(right)
    if rn < 1e-12:  # looking straight along up: pick any perpendicular
        right = np.cross(fwd, np.array([1.0, 0.0, 0.0]))
        rn = np.linalg.norm(right)
        if rn < 1e-12:
            right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
            rn = np.linalg.norm(right)
    right = right / rn
    cam_up = np.cross(right, fwd)
    m = np.eye(4, dtype=np.float64)
    m[0, :3] = right
    m[1, :3] = cam_up
    m[2, :3] = -fwd
    m[:3, 3] = -m[:3, :3] @ eye
    return m
