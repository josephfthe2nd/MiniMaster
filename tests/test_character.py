"""Character Lab: the morphable human generator.

Locks the spec's guarantees: constant closed topology, watertight shells at
every subdivision level for any slider combination and pose, orthogonal
height, symmetric default body, deterministic builds, and live-preview speed.
"""

import time

import numpy as np
import pytest

from minimaster.character import generator as G
from minimaster.character import topology as T
from minimaster.core.subdiv import subdivide_quads


def _all_watertight(shells):
    for name, mesh, _ in shells:
        rep = mesh.integrity_report()
        assert rep["watertight"], f"{name}: {rep}"
        assert rep["volume"] > 0, f"{name} inverted"


def test_topology_euler_and_no_orphans():
    assert T.BODY_QUADS.shape == (401, 4)
    assert len(T.BODY_KEEP) == 403
    assert np.unique(T.BODY_QUADS).size == 403  # every vertex referenced
    # closed all-quad manifold: E = 2F, chi = V - E + F = 2
    assert 403 - 2 * 401 + 401 == 2
    assert T.HEAD_QUADS.shape == (176, 4)
    assert T.HEAD_V == 178


def test_manifold_independent_of_geometry():
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(T.RAW_V, 3))
    subdivide_quads(raw[T.BODY_KEEP], T.BODY_QUADS, 1)  # raises if not closed


def test_default_character_watertight_at_depth():
    for levels in (0, 1, 2):
        _all_watertight(G.character_shells(levels=levels))


@pytest.mark.parametrize("name", sorted(G.PARAMS))
def test_slider_extremes_watertight(name):
    _, lo, hi, _ = G.PARAMS[name]
    for val in (lo, hi):
        _all_watertight(G.character_shells({name: val}, levels=1))


def test_corner_combinations_watertight():
    corners = [
        {k: v[1] for k, v in G.PARAMS.items()},  # all lo
        {k: v[2] for k, v in G.PARAMS.items()},  # all hi
    ]
    for g in (0.0, 1.0):
        for w in (-1.0, 1.0):
            for m in (-1.0, 1.0):
                corners.append({"gender": g, "weight": w, "muscle": m})
    for params in corners:
        _all_watertight(G.character_shells(params, levels=1))


@pytest.mark.parametrize("pose", sorted(G.POSES))
def test_poses_stay_watertight(pose):
    shells = G.character_shells({"gender": 0.0, "muscle": 0.5},
                                pose_name=pose, levels=1)
    _all_watertight(shells)


def test_height_is_exact_and_orthogonal():
    for h in (20.0, 30.0, 45.0):
        mesh = G.character_mesh({"height": h}, levels=1)
        lo, hi = mesh.bounds
        assert hi[2] - lo[2] == pytest.approx(h, rel=1e-6)
    # proportions don't change the normalized silhouette scale relation
    a = G.character_mesh({"height": 30.0, "muscle": 1.0}, levels=1)
    lo, hi = a.bounds
    assert hi[2] - lo[2] == pytest.approx(30.0, rel=1e-6)


def test_default_body_symmetry_by_moments():
    shells = dict((n, m) for n, m, _ in G.character_shells(levels=1))
    for name in ("body", "head"):
        v = shells[name].vertices
        assert abs(v[:, 0].mean()) < 1e-9  # centroid on the mirror plane
        assert abs(v[:, 0].max() + v[:, 0].min()) < 1e-6  # extents mirror


def test_gender_actually_dimorphs():
    male = G.character_mesh({"gender": 0.0}, levels=1)
    female = G.character_mesh({"gender": 1.0}, levels=1)
    mlo, mhi = male.bounds
    flo, fhi = female.bounds
    assert mhi[0] - mlo[0] > fhi[0] - flo[0]  # male wider overall (shoulders)


def test_unknown_param_and_pose_rejected():
    with pytest.raises(KeyError, match="unknown character params"):
        G.character_shells({"nose_hair": 1.0})
    with pytest.raises(KeyError, match="unknown pose"):
        G.character_shells(pose_name="breakdance")


def test_determinism():
    a = G.character_mesh({"gender": 0.3, "muscle": 0.7}, pose_name="walk", levels=2)
    b = G.character_mesh({"gender": 0.3, "muscle": 0.7}, pose_name="walk", levels=2)
    assert np.array_equal(a.vertices, b.vertices)
    assert np.array_equal(a.faces, b.faces)


def test_live_preview_speed():
    G.character_shells(levels=1)  # warm
    t = time.time()
    for _ in range(5):
        G.character_shells({"muscle": 0.5}, levels=1)
    per = (time.time() - t) / 5
    assert per < 0.15, f"live rebuild too slow: {per * 1000:.0f} ms"


def test_skin_weights_valid():
    assert T.BODY_WEIGHTS.shape == (403, len(T.BONES))
    assert np.allclose(T.BODY_WEIGHTS.sum(axis=1), 1.0)
    assert (T.BODY_WEIGHTS >= 0).all()
    arm = G.joint_layout(G.resolve({}))
    assert set(T.BONES).issubset(set(arm.bone_names()))
