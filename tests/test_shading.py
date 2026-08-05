"""Skin shading model and the portrait renderer."""

import numpy as np
import pytest

from minimaster.character import mhbase as mh
from minimaster.character import portrait as P
from minimaster.character import shading as sh
from minimaster.core.subdiv import cube_cage, quads_to_mesh


def test_srgb_roundtrip():
    x = np.linspace(0.0, 1.0, 32)
    assert np.allclose(sh.linear_to_srgb(sh.srgb_to_linear(x)), x, atol=1e-9)


def test_hex_to_linear_is_darker_than_srgb():
    # the whole point of the linear pipeline: mid grey is ~0.216, not 0.5
    lin = sh.hex_to_linear("#808080")
    assert lin == pytest.approx([0.2158, 0.2158, 0.2158], abs=1e-3)
    assert np.allclose(sh.hex_to_linear("#ffffff"), 1.0)
    assert np.allclose(sh.hex_to_linear("#000000"), 0.0)


def test_hex_validation():
    with pytest.raises(ValueError):
        sh.hex_to_linear("#fff")


def test_tonemap_is_monotone_and_bounded():
    x = np.linspace(0.0, 12.0, 64)[:, None] * np.ones(3)
    y = sh.tonemap(x)
    assert (y >= 0).all() and (y <= 1).all()
    assert (np.diff(y[:, 0]) >= -1e-12).all()  # never decreases


def _sphere(n_sub=1):
    from minimaster.core import primitives
    return primitives.icosphere(radius=1.0, subdivisions=n_sub)


def test_shade_vertices_shape_and_positivity():
    mesh = _sphere()
    n = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
    v = np.tile([0.0, 0.0, 1.0], (len(n), 1))
    out = sh.shade_vertices(n, v, sh.SKIN, sh.LightRig())
    assert out.shape == (len(n), 3)
    assert np.isfinite(out).all() and (out >= 0).all()


def test_wrap_diffuse_lights_the_terminator():
    """Wrapped diffuse must lift the region just past N.L = 0 (that is the
    subsurface cue); plain Lambert leaves it black."""
    # just PAST the terminator: N.L is slightly negative (about -0.24), the
    # band where real skin still glows because light scattered in from nearby
    n = np.array([[1.0, -0.25, 0.0]])
    n = n / np.linalg.norm(n)
    v = np.array([[0.0, 0.0, 1.0]])
    rig = sh.LightRig(key=sh.Light((0.0, 1.0, 0.0), "#ffffff", 1.0),
                      fill=sh.Light((0.0, 1.0, 0.0), "#000000", 0.0),
                      rim=sh.Light((0.0, 1.0, 0.0), "#000000", 0.0),
                      ambient="#000000", ambient_intensity=0.0)
    lam = sh.shade_vertices(n, v, sh.Material(wrap=0.0), rig).sum()
    wrapped = sh.shade_vertices(n, v, sh.Material(wrap=0.6), rig).sum()
    assert lam == pytest.approx(0.0, abs=1e-9)
    assert wrapped > 0.01


def test_per_vertex_base_colors_override_material():
    mesh = _sphere()
    n = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
    v = np.tile([0.0, 0.0, 1.0], (len(n), 1))
    cols = np.tile(sh.hex_to_linear("#ff0000"), (len(n), 1))
    out = sh.shade_vertices(n, v, sh.SKIN, sh.LightRig(), base_colors=cols)
    assert out[:, 0].sum() > out[:, 1].sum() * 2


def test_cavity_ao_darkens_creases_not_ridges():
    """A dented sphere: the dent must come back darker than the rest."""
    mesh = _sphere(2)
    v = mesh.vertices.copy()
    n = v / np.linalg.norm(v, axis=1, keepdims=True)
    dent = n[:, 2] > 0.85
    v[dent] *= 0.72  # push a cap inward -> a concave crater rim
    ao = sh.cavity_ao(v, mesh.faces, n)
    assert ao.shape == (len(v),)
    assert ((ao >= 0) & (ao <= 1)).all()
    assert ao.min() < 0.999  # something got occluded


def test_ao_is_scale_invariant():
    mesh = _sphere(2)
    v = mesh.vertices
    n = v / np.linalg.norm(v, axis=1, keepdims=True)
    a = sh.cavity_ao(v, mesh.faces, n)
    b = sh.cavity_ao(v * 100.0, mesh.faces, n)
    assert np.allclose(a, b, atol=1e-6)


def test_smoothing_reduces_variance():
    mesh = _sphere(2)
    rng = np.random.default_rng(0)
    noisy = rng.normal(size=len(mesh.vertices))
    smoothed = sh.smooth_vertex_scalar(noisy, mesh.faces, 3)
    assert smoothed.var() < noisy.var()


def test_eye_vertex_colors_paints_sclera_iris_pupil():
    mesh = _sphere(2)
    cols = sh.eye_vertex_colors(mesh.vertices, [0.0, -1.0, 0.0])
    assert cols.shape == (len(mesh.vertices), 3)
    uniq = np.unique(np.round(cols, 6), axis=0)
    assert len(uniq) == 3          # sclera + iris + pupil
    lum = cols.sum(axis=1)
    assert lum.max() > lum.min() * 5   # the pupil is much darker than sclera


def test_render_portrait_produces_an_image():
    mesh = _sphere(2)
    img = P.render_portrait([P.Shell(mesh, sh.SKIN)], size=(64, 64))
    assert img.shape == (64, 64, 4)
    assert img.dtype == np.uint8
    assert (img[:, :, 3] == 255).all()
    # the sphere must actually cover the middle of the frame
    assert not np.array_equal(img[32, 32, :3], img[0, 0, :3])


def test_render_portrait_empty_is_background():
    from minimaster.core.mesh import Mesh
    empty = Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    img = P.render_portrait([P.Shell(empty)], size=(16, 16))
    assert img.shape == (16, 16, 4)


def test_flat_cage_still_renders():
    v, q = cube_cage(2.0)
    img = P.render_portrait([P.Shell(quads_to_mesh(v, q))], size=(48, 48))
    assert img[:, :, 3].any()


@pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
def test_character_shells_include_eyes():
    base = mh.BaseMesh()
    shells = P.character_shells(base)
    assert len(shells) == 3               # body + two eyes
    assert shells[0].material is sh.SKIN
    for s in shells[1:]:
        assert s.base_colors is not None  # irises painted
        assert len(s.base_colors) == len(s.mesh.vertices)
    # body arrives Z-up: taller than it is wide
    lo, hi = shells[0].mesh.bounds
    assert (hi[2] - lo[2]) > (hi[0] - lo[0])


@pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")
def test_full_character_portrait_renders():
    base, lib = mh.load()
    v = lib.apply(base.verts, mh.macro_weights(gender=1.0))
    img = P.render_portrait(P.character_shells(base, v), size=(96, 140))
    assert img.shape == (140, 96, 4)
    assert len(np.unique(img[:, :, :3].reshape(-1, 3), axis=0)) > 50  # real shading
