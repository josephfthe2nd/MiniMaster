"""The CC0 MakeHuman base mesh + morph-target path.

Skipped entirely when the assets are absent (they are fetched, not committed —
see tools/fetch_makehuman_assets.py), so the suite stays green on a bare clone.
"""

import numpy as np
import pytest

from minimaster.character import mhbase as M

pytestmark = pytest.mark.skipif(
    not M.available(), reason="CC0 MakeHuman assets not fetched")


@pytest.fixture(scope="module")
def base():
    return M.BaseMesh()


@pytest.fixture(scope="module")
def lib():
    return M.TargetLibrary()


def test_base_mesh_shape(base):
    assert base.verts.shape == (19158, 3)
    assert base.quads.shape == (18486, 4)
    assert base.body_quads.shape == (13378, 4)
    assert len(base.body_verts_idx) == 13380


def test_body_is_a_closed_manifold(base):
    mesh = M.body_mesh(base)
    rep = mesh.integrity_report()
    assert rep["watertight"] and rep["outward"], rep
    assert rep["boundary_edges"] == 0
    assert rep["nonmanifold_edges"] == 0


def test_body_cage_is_all_quad_and_euler_2(base):
    v, q = base.body_cage()
    e = np.sort(np.stack([q.ravel(), np.roll(q, -1, axis=1).ravel()], axis=1), axis=1)
    uq, cnt = np.unique(e, axis=0, return_counts=True)
    assert (cnt == 2).all()
    assert len(v) - len(uq) + len(q) == 2  # genus 0


def test_target_library_loads(lib):
    assert len(lib) == 1280
    cats = lib.categories()
    for expected in ("macrodetails", "nose", "eyes", "mouth", "cheek", "chin", "ears"):
        assert expected in cats, cats
    assert cats["nose"] >= 40 and cats["eyes"] >= 60


def test_targets_are_sparse_deltas(lib):
    idx, vec = lib.get("nose/nose-curve-convex")
    assert idx.ndim == 1 and vec.shape == (len(idx), 3)
    assert idx.max() < 19158
    assert np.isfinite(vec).all()


def test_morph_changes_geometry_and_keeps_it_watertight(base, lib):
    v = lib.apply(base.verts, {"macrodetails/caucasian-male-young": 1.0})
    assert not np.allclose(v, base.verts)
    mesh = M.body_mesh(base, v)
    assert mesh.integrity_report()["watertight"]


def test_missing_target_is_neutral_unless_strict(base, lib):
    v = lib.apply(base.verts, {"no/such-target": 1.0})
    assert np.allclose(v, base.verts)
    with pytest.raises(KeyError):
        lib.apply(base.verts, {"no/such-target": 1.0}, strict=True)


def test_macro_weights_axis_partition():
    # every axis interpolates between two nodes and sums to 1
    w = M.macro_weights(gender=1.0, age=0.5, muscle=0.5, weight=0.5, height=0.5)
    eth = sum(x for k, x in w.items()
              if k.startswith("macrodetails/caucasian"))
    assert eth == pytest.approx(1.0)
    assert all(x > 0 for x in w.values())


def test_macro_sliders_are_continuous_and_dimorphic(base, lib):
    def build(**kw):
        return M.body_mesh(base, lib.apply(base.verts, M.macro_weights(**kw)))
    man = build(gender=1.0, age=0.5)
    woman = build(gender=0.0, age=0.5)
    mid = build(gender=0.5, age=0.5)
    for m in (man, woman, mid):
        assert m.integrity_report()["watertight"]
    span = lambda m: m.bounds[1][0] - m.bounds[0][0]
    assert span(man) > span(woman)          # broader male shoulders
    assert span(woman) < span(mid) < span(man)  # the slider interpolates
    child = build(gender=1.0, age=0.1875)
    assert child.bounds[1][1] - child.bounds[0][1] < man.bounds[1][1] - man.bounds[0][1]


def test_paired_slider():
    assert M.paired_weights("a", "b", 0.5) == {"b": 0.5}
    assert M.paired_weights("a", "b", -0.25) == {"a": 0.25}
    assert M.paired_weights("a", "b", 0.0) == {}


def test_facial_detail_targets_exist_and_move_the_face(base, lib):
    for name in ("nose/nose-curve-convex", "nose/nose-curve-concave",
                 "cheek/l-cheek-bones-incr", "mouth/mouth-cupidsbow-incr",
                 "chin/chin-bones-incr", "eyes/l-eye-corner1-up"):
        assert name in lib, name
    v = lib.apply(base.verts, {"cheek/l-cheek-bones-incr": 1.0})
    moved = np.linalg.norm(v - base.verts, axis=1)
    assert moved.max() > 1e-4
    assert (moved > 1e-9).sum() < 2000  # a LOCAL edit, not a global one


def test_deterministic(base, lib):
    w = M.macro_weights(gender=0.7, age=0.6, muscle=0.8)
    a = lib.apply(base.verts, w)
    b = lib.apply(base.verts, w)
    assert np.array_equal(a, b)
