import numpy as np
import pytest

from minimaster.core import primitives as prim


def check_solid(mesh, name=""):
    rep = mesh.integrity_report()
    assert rep["watertight"], f"{name}: {rep}"
    assert rep["volume"] > 0, f"{name}: volume {rep['volume']}"


@pytest.mark.parametrize("kind", sorted(prim.PRIMITIVES))
def test_defaults_watertight(kind):
    check_solid(prim.build(kind), kind)


@pytest.mark.parametrize("segments", [3, 4, 5, 6, 8, 12, 24])
@pytest.mark.parametrize("taper", [0.0, 0.25, 0.5, 1.0])
def test_cylinder_family(segments, taper):
    m = prim.cylinder(radius=0.7, height=2.0, segments=segments, taper=taper)
    check_solid(m, f"cylinder s={segments} t={taper}")
    lo, hi = m.bounds
    assert hi[2] - lo[2] == pytest.approx(2.0)


def test_cylinder_volume_converges():
    m = prim.cylinder(radius=1.0, height=1.0, segments=64)
    assert m.volume() == pytest.approx(np.pi, rel=0.01)


def test_cone_apex_single_vertex():
    m = prim.cylinder(radius=1.0, height=1.0, segments=6, taper=0.0)
    top = m.vertices[np.isclose(m.vertices[:, 2], 0.5)]
    assert len(top) == 1


@pytest.mark.parametrize("segments", [4, 6, 8, 12])
def test_capsule(segments):
    m = prim.capsule(radius=0.5, height=2.0, segments=segments)
    check_solid(m, f"capsule s={segments}")
    lo, hi = m.bounds
    assert hi[2] - lo[2] == pytest.approx(2.0)


def test_capsule_height_clamped_to_sphere():
    m = prim.capsule(radius=1.0, height=0.5, segments=8)
    check_solid(m, "capsule clamped")
    lo, hi = m.bounds
    assert hi[2] - lo[2] == pytest.approx(2.0)


@pytest.mark.parametrize("subdiv", [0, 1, 2, 3])
def test_icosphere(subdiv):
    m = prim.icosphere(radius=0.5, subdivisions=subdiv)
    check_solid(m, f"icosphere {subdiv}")
    r = np.linalg.norm(m.vertices, axis=1)
    assert np.allclose(r, 0.5)
    assert len(m.faces) == 20 * 4**subdiv


def test_icosphere_volume_converges():
    m = prim.icosphere(radius=1.0, subdivisions=3)
    assert m.volume() == pytest.approx(4.0 / 3.0 * np.pi, rel=0.02)


@pytest.mark.parametrize("segments,minor", [(3, 3), (8, 6), (16, 8)])
def test_torus(segments, minor):
    m = prim.torus(radius=0.5, thickness=0.2, segments=segments, minor_segments=minor)
    check_solid(m, f"torus {segments}x{minor}")


def test_wedge_watertight_and_convex_orientation():
    m = prim.wedge(2.0, 1.0, 3.0)
    check_solid(m, "wedge")
    assert m.volume() == pytest.approx(0.5 * 2.0 * 3.0 * 1.0)
    normals = m.face_normals()
    centers = m.triangles().mean(axis=1)
    centroid = m.vertices.mean(axis=0)
    assert (np.einsum("ij,ij->i", normals, centers - centroid) > 0).all()


def test_box_dimensions():
    m = prim.box(2.0, 4.0, 6.0)
    lo, hi = m.bounds
    assert np.allclose(hi - lo, [2.0, 4.0, 6.0])


def test_build_rejects_unknown():
    with pytest.raises(KeyError):
        prim.build("dodecahedron")
    with pytest.raises(KeyError):
        prim.build("box", {"radius": 1.0})


def test_build_merges_defaults():
    m = prim.build("cylinder", {"segments": 5})
    # 5-gon prism: 5*2 side tris + 5 bottom + 5 top fan tris
    assert len(m.faces) == 20


def test_revolve_rejects_bad_profiles():
    with pytest.raises(ValueError):
        prim._revolve([(0, 0), (1, 1)], 8)  # too short
    with pytest.raises(ValueError):
        prim._revolve([(0.5, 0), (1, 0.5), (0, 1)], 8)  # doesn't start at axis
    with pytest.raises(ValueError):
        prim.cylinder(segments=2)


@pytest.mark.parametrize(
    "kind,params",
    [
        ("box", {"width": -1.0}),
        ("box", {"height": 0.0}),
        ("wedge", {"depth": -2.0}),
        ("cylinder", {"radius": -0.5}),
        ("cylinder", {"height": -1.0}),
        ("cylinder", {"taper": -0.5}),
        ("capsule", {"radius": 0.0}),
        ("icosphere", {"radius": -0.5}),
        ("torus", {"radius": -0.5}),
        ("torus", {"thickness": -0.1}),
    ],
)
def test_bad_dimensions_rejected(kind, params):
    """Negative/zero dimensions would build inverted shells that slice as
    cavities — they must be rejected, not silently accepted."""
    with pytest.raises(ValueError):
        prim.build(kind, params)


def test_torus_self_intersection_rejected():
    with pytest.raises(ValueError, match="thickness"):
        prim.torus(radius=0.5, thickness=1.2)
    with pytest.raises(ValueError, match="thickness"):
        prim.torus(radius=0.5, thickness=1.0)  # tube touching the axis
