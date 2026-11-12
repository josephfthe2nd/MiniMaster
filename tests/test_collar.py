"""The root collar (``minimaster.character.collar``): the seam between the shaped skin and a tail.

The junction moves cage vertices and so gives the skin MASS around a root, but a cage quad is 0.28 units
and a root 0.4-0.6 across: the skin meets the root 53-100 degrees from tangent and a crisp ring survives in
every close-up. The collar is a closed shell on the appendage's own geometry whose outer rim lies on the
shaped skin, tangent to it, and that rises along the wall to meet it tangentially.

The first half of this file needs no assets: a synthetic sphere is the body and a 16-gon tube leaves its
pole, so the geometry is pinned down exactly (rim on the surface, tangent to it, joined to the wall,
closed, hidden parts inside, mirror symmetric, narrower on the folded side). The second half builds real
tails on the real body (five bodies, poses, two subdivision levels) and runs the print report, the CLI and the lab.
"""

import json
import subprocess
import sys
import time
import warnings

import numpy as np
import pytest

from minimaster.character import collar as CO
from minimaster.character import kinds as K
from minimaster.character import mhbase as mh
from minimaster.character import print_report as PR
from minimaster.character import printprep as pp
from minimaster.character import shading as sh
from minimaster.character.sliders import Character, SliderCatalog
from minimaster.core.mesh import Mesh
from minimaster.core.subdiv import cube_cage, quads_to_mesh, subdivide_quads

assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")

SPHERE_R = 10.0
TUBE_R = 1.0


# -- a synthetic body: a sphere, a 16-gon tube leaving its +Z pole ---------------------------------------------


def _sphere_mesh(levels=6, radius=SPHERE_R, x_shift=0.0):
    v, q = cube_cage(2.0)
    v, q = subdivide_quads(v, q, levels)
    v = v / np.linalg.norm(v, axis=1, keepdims=True) * radius
    m = quads_to_mesh(v, q)
    return m.vertices + np.array([x_shift, 0.0, 0.0]), m.faces


def _tube_mesh(x=0.0, r=TUBE_R, z0=8.0, z1=14.0, ring=16, rings=24, tilt=0.0):
    """A closed 16-gon tube along +Z through ``(x, 0)`` (leaning toward +X by ``tilt`` degrees)."""
    zs = np.linspace(z0, z1, rings)
    ang = np.arange(ring) * 2 * np.pi / ring
    t = np.tan(np.radians(tilt))
    pts = np.array([[x + t * (z - z0) + r * np.cos(a), r * np.sin(a), z] for z in zs for a in ang])
    faces = []
    for i in range(rings - 1):
        for j in range(ring):
            a, b = i * ring + j, i * ring + (j + 1) % ring
            faces += [[a, b, b + ring], [a, b + ring, a + ring]]
    c0 = np.array([x, 0.0, z0])
    c1 = np.array([x + t * (z1 - z0), 0.0, z1])
    pts = np.vstack([pts, c0, c1])
    n0, n1 = len(pts) - 2, len(pts) - 1
    for j in range(ring):
        faces.append([n0, (j + 1) % ring, j])
        last = (rings - 1) * ring
        faces.append([n1, last + j, last + (j + 1) % ring])
    m = Mesh(pts, np.array(faces))
    return m if m.volume() > 0 else Mesh(pts, np.array(faces)[:, [0, 2, 1]])


def _axis(x=0.0, r=TUBE_R, z0=8.0, z1=14.0, tilt=0.0):
    t = np.tan(np.radians(tilt))
    zs = np.linspace(z0, z1, 12)
    return [(x + t * (z - z0), 0.0, z) for z in zs], [r * np.cos(np.pi / 16)] * 12      # the inscribed circle


@pytest.fixture(scope="module")
def sphere():
    v, f = _sphere_mesh()
    return v, f


def _collar(sphere, profile=None, x=0.0, tilt=0.0, diag=None, centre=None):
    v, f = sphere
    pts, rad = _axis(x=x, tilt=tilt)
    wall = _tube_mesh(x=x, tilt=tilt)
    patch = CO.SkinPatch(v, f, np.array([x, 0.0, SPHERE_R]) if centre is None else centre, 6.0)
    prof = profile or CO.CollarProfile()
    m = CO.build_collar(patch, pts, rad, wall, prof, diag)
    if m is not None:
        CO.shading_transfer(m, patch, wall)                 # what collar_pieces does: the shading the renderer uses
    return m, patch, wall


def _rings(diag):
    n = diag["top"].shape[1]
    A = len(CO.ARC_ANGLES)
    return n, A


# -- the profile ------------------------------------------------------------------------------------------------


def test_a_profile_is_validated_hashable_data():
    p = CO.CollarProfile()
    assert p.active and p.segments >= 48 and p.rings >= 12 and p.up == (0.0, 1.0, 0.0)
    assert p.with_(strength=0.0).active is False and p.with_(height=0.0).active is False
    assert p.digest() == CO.CollarProfile().digest() != p.with_(width=2.0).digest()
    assert json.loads(json.dumps(p.to_dict()))["material"] == "skin"
    for bad in (dict(width=-1), dict(falloff=1.0), dict(material="stone"), dict(up=(0, 0, 0)),
                dict(segments=8), dict(open_deg=100, fold_deg=90), dict(sink=float("nan"))):
        with pytest.raises(ValueError):
            CO.CollarProfile(**bad)


def test_the_default_kind_has_no_collar():
    assert K.Kind().collar_profile(None) is None
    for kid in ("horn", "harvest"):
        assert K.KINDS[kid].collar_profile(None) is None        # the next stage enables them (the wing has one:
                                                                # tests/test_wing_root.py)


# -- the pieces of the geometry ------------------------------------------------------------------------------------


def test_the_closest_point_on_a_triangle_is_exact_in_every_region():
    rng = np.random.default_rng(3)
    a, b, c = (np.array(x, dtype=float) for x in ([0, 0, 0], [1, 0, 0], [0, 1, 0]))
    pts = rng.uniform(-1, 2, (400, 3))
    q, bary = CO.closest_on_triangles(pts, np.tile(a, (400, 1)), np.tile(b, (400, 1)), np.tile(c, (400, 1)))
    # brute force over a fine sampling of the triangle
    u, v = np.meshgrid(np.linspace(0, 1, 201), np.linspace(0, 1, 201))
    keep = u + v <= 1
    samples = np.stack([u[keep], v[keep], np.zeros(keep.sum())], axis=1)
    for i in range(0, 400, 8):
        best = samples[((samples - pts[i]) ** 2).sum(1).argmin()]
        assert np.linalg.norm(q[i] - pts[i]) <= np.linalg.norm(best - pts[i]) + 1e-9
    assert np.allclose(bary.sum(1), 1.0) and (bary >= -1e-12).all()
    assert np.allclose(q, a * bary[:, :1] + b * bary[:, 1:2] + c * bary[:, 2:])


def test_a_skin_patch_finds_the_surface_normal_and_signed_distance_of_a_sphere(sphere):
    v, f = sphere
    patch = CO.SkinPatch(v, f, np.array([0.0, 0.0, SPHERE_R]), 5.0)
    rng = np.random.default_rng(1)
    d = rng.uniform(-0.6, 0.8, 300)
    dirs = rng.normal(size=(300, 3))
    dirs[:, 2] = np.abs(dirs[:, 2]) * 6 + 3
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    q = dirs * (SPHERE_R + d)[:, None]
    q = q[np.linalg.norm(q - np.array([0, 0, SPHERE_R]), axis=1) < 3.0]
    pt, nrm, sd, _ = patch.closest(q)
    r = np.linalg.norm(q, axis=1)
    assert np.allclose(sd, r - SPHERE_R, atol=0.03)                    # signed, positive outside
    assert np.allclose(np.linalg.norm(pt, axis=1), SPHERE_R, atol=0.03)
    cosang = (nrm * pt / np.linalg.norm(pt, axis=1, keepdims=True)).sum(1)
    assert np.degrees(np.arccos(np.clip(cosang, -1, 1))).max() < 1.0   # the smooth vertex normal is the sphere's there


def test_the_skin_patch_agrees_with_brute_force_and_the_walk_with_the_grid(sphere):
    v, f = sphere
    patch = CO.SkinPatch(v, f, np.array([0.0, 0.0, SPHERE_R]), 4.0)
    rng = np.random.default_rng(5)
    base = patch.V[rng.integers(0, patch.n_vertices, 200)]
    q = base + patch.N[rng.integers(0, patch.n_vertices, 200)] * rng.uniform(-0.05, 0.2, (200, 1))
    pt, _, _, vtx = patch.closest(q)
    a, b, c = (patch.V[patch.F[:, k]] for k in range(3))
    for i in range(0, 200, 10):
        brute, _ = CO.closest_on_triangles(np.tile(q[i], (len(a), 1)), a, b, c)
        assert np.linalg.norm(pt[i] - q[i]) == pytest.approx(np.linalg.norm(brute - q[i], axis=1).min(), abs=1e-9)
    # warm start from the neighbouring answer: the same nearest vertex
    q2 = q + 0.01
    assert (patch.nearest_vertex(q2, guess=vtx) == patch.nearest_vertex(q2)).mean() > 0.95


def test_rays_find_the_first_wall_of_a_closed_mesh():
    tube = _tube_mesh()
    o = np.tile(np.array([0.0, 0.0, 10.0]), (8, 1))
    ang = np.arange(8) * np.pi / 4
    d = np.stack([np.cos(ang), np.sin(ang), np.zeros(8)], axis=1)
    t = CO.ray_mesh_first_hit(o, d, tube.triangles())
    assert (t > TUBE_R * np.cos(np.pi / 16) - 1e-6).all() and (t < TUBE_R + 1e-6).all()
    miss = CO.ray_mesh_first_hit(o + np.array([5, 0, 0]), d[[0]].repeat(8, 0), tube.triangles())
    assert np.isinf(miss).all()


def test_the_axis_frame_is_orthonormal_and_follows_up():
    ax = CO.Axis([(0, 0, 0), (0, 0, 1), (0.3, 0, 2), (0.9, 0, 2.6)], [0.5] * 4, (0, 1, 0))
    s = np.linspace(0, ax.length, 30)
    pt, T, u, v, r = ax.at(s)
    for a, b in ((T, u), (T, v), (u, v)):
        assert np.abs((a * b).sum(1)).max() < 1e-9
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0) and (u[:, 1] > 0.99).all()      # a bend in the XZ plane keeps u up


# -- the collar on a sphere: the geometry exactly ------------------------------------------------------------------------


def test_the_collar_is_a_closed_outward_shell_with_the_asked_sampling(sphere):
    d = {}
    m, _, _ = _collar(sphere, diag=d)
    rep = m.integrity_report()
    assert rep["watertight"] and rep["outward"] and rep["degenerate_faces"] == 0 and rep["nonmanifold_edges"] == 0
    assert d["top"].shape[1] >= 48 and d["n_vis"] >= 12 + 1                               # >= 48 around, >= 12 radial


def _ang(a, b):
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def _ids(d, station):
    """The vertex of every kept column at a station of the sheet (``vid_top``; -1 where the column is cut)."""
    return d["vid_top"][station][d["keep"]]


def test_the_sheet_crosses_the_skin_at_the_crossing_angle_and_is_drawn_with_the_skins_normals(sphere):
    """Round 4: the sheet is no longer tangent to the skin where they meet. A tangent sheet lay within the facets'
    error of the skin over a band and z-fought there (the serrated outline of the round-3 'sticker'); it now goes
    INTO the skin at GAMMA_SKIN_DEG (a clean depth crossing) and that kink is never seen, because the renderer draws
    the collar with the skin's own normals there (CollarMesh.shading_normals)."""
    d = {}
    m, patch, _ = _collar(sphere, diag=d)
    n_vis = d["n_vis"]
    rim = d["top"][n_vis - 1][d["keep"]]
    assert np.abs(np.linalg.norm(rim, axis=1) - SPHERE_R).max() < 0.03          # on the sphere (the mesh's facets)
    sphere_n = rim / np.linalg.norm(rim, axis=1, keepdims=True)
    own = sh.vertex_normals(m.vertices, m.faces)[_ids(d, n_vis - 1)]             # the geometry: the crossing angle
    shade = m.shading_normals[_ids(d, n_vis - 1)]                                 # what the renderer draws with
    wide = (d["W"] > 0.3)[d["keep"]]
    g = _ang(own, sphere_n)[wide]
    assert wide.sum() > 20 and 0.5 * CO.GAMMA_SKIN_DEG < np.median(g) < 2.0 * CO.GAMMA_SKIN_DEG
    # the last visible chord itself (the vertex normal also sees the dive past the crossing): GAMMA_SKIN_DEG into the skin
    chord = (rim - d["top"][n_vis - 2][d["keep"]])[wide]
    into = np.degrees(np.arcsin(np.abs((chord * sphere_n[wide]).sum(1)) / np.linalg.norm(chord, axis=1)))
    assert 0.7 * CO.GAMMA_SKIN_DEG < np.median(into) < 1.3 * CO.GAMMA_SKIN_DEG, np.median(into)
    assert np.percentile(_ang(shade, sphere_n)[wide], 95) < 1.0                  # no edge in the shading


def test_the_visible_sheet_is_above_the_skin_and_the_dive_and_the_skirt_are_under_it(sphere):
    d = {}
    m, patch, _ = _collar(sphere, diag=d)
    top, bottom, arcs, keep = d["top"], d["bottom"], d["arcs"], d["keep"]
    n_vis = d["n_vis"]
    for i in range(1, n_vis - 1):                                                 # the visible sheet: never under the skin
        r = np.linalg.norm(top[i][keep], axis=1)
        assert (r - SPHERE_R).min() > -CO.SKIN_TOL, i
    for i in range(n_vis, len(top)):                                              # past the crossing: under it
        assert (np.linalg.norm(top[i][keep], axis=1) - SPHERE_R).max() < 0.0, i
    assert (np.linalg.norm(bottom[:-1][:, keep], axis=2) < SPHERE_R + 0.01).all()  # the underside: under it
    far = bottom[n_vis - 1][keep]
    assert (SPHERE_R - np.linalg.norm(far, axis=1)).min() > 0.01                  # sunk, so the weld overlaps the body
    # the carried-on sheet is inside the tube: its distance to the axis is below the wall's
    rad = np.linalg.norm(arcs[:, keep, :2], axis=2)
    assert (rad < TUBE_R * 1.001).all() and (rad[-1] < 0.95 * TUBE_R).all()


def test_the_sheet_reaches_the_faceted_wall_and_goes_in_at_the_wall_angle(sphere):
    """The top edge lies ON the polygon of the wall (not on a smoothed circle a hair outside it, which the facets
    crossed in scallops) and arrives at about ALPHA_WALL_DEG; the renderer draws it with the wall's own normals."""
    d = {}
    m, _, wall = _collar(sphere, diag=d)
    keep = d["keep"]
    E = d["E"][keep]
    rad = np.linalg.norm(E[:, :2], axis=1)
    assert (rad <= TUBE_R + 1e-6).all() and (rad >= TUBE_R * np.cos(np.pi / 16) - 1e-6).all()   # on the 16-gon
    t = CO._unit(E - d["top"][1][keep])                                           # the last chord, into the wall
    radial = np.stack([E[:, 0], E[:, 1], np.zeros(len(E))], axis=1) / rad[:, None]
    arrive = 90.0 - _ang(-t, radial)                                              # its angle to the wall's plane
    assert 0.4 * CO.ALPHA_WALL_DEG < np.median(arrive) < 1.3 * CO.ALPHA_WALL_DEG
    vw = sh.vertex_normals(wall.vertices, wall.faces)
    wn = CO._shaded_normals(E, wall, vw)
    assert np.percentile(_ang(m.shading_normals[_ids(d, 0)], wn), 95) < 1.0


def test_the_columns_sit_on_the_corners_of_the_walls_polygon(sphere):
    """Between two columns the collar's top edge is a straight chord; a corner of the 16-gon between two columns
    poked through it and the contact line dipped there (a scallop per facet, the round-3 horn finding). The
    columns are a multiple of the sides, the first ON a corner, so every chord lies on a facet."""
    d = {}
    m, patch, wall = _collar(sphere, diag=d)
    assert d["sides"] == 16 and len(d["theta"]) % 16 == 0
    per = len(d["theta"]) // 16
    corners = np.linalg.norm(d["E"][::per, :2], axis=1)
    assert np.abs(corners - TUBE_R).max() < 2e-3                                   # the polygon's vertices
    rep = CO.contact_lines(d, wall, patch)
    assert rep["wall_wave"] < 0.01 and rep["wall_contact_missing"] == 0.0
    # the wall IS faceted: its radius round the ring varies by the sagitta
    sag = TUBE_R * (1.0 - np.cos(np.pi / 16))
    w = CO.Wall(CO.Axis(*_axis(), (0, 1, 0)), wall)
    raw = w.radius(np.arange(96) * 2 * np.pi / 96, np.full(96, 3.0))[0]
    assert np.ptp(raw) == pytest.approx(sag, rel=0.2)


def test_the_skin_alone_meets_the_wall_far_from_tangent(sphere):
    """What the collar replaces: the sphere meets a tube leaving it at 90 degrees; the visible surface now
    meets the wall at a few degrees (the number the junction alone cannot reach: 53-100 on the real body)."""
    d = {}
    _collar(sphere, diag=d)
    skin_n = d["N"][0]                                    # the skin normal at the exit: the pole's, along +Z
    wall_n = np.stack([d["X"][:, 0], d["X"][:, 1], np.zeros(len(d["X"]))], axis=1)
    wall_n /= np.linalg.norm(wall_n, axis=1, keepdims=True)
    before = np.degrees(np.arccos(np.clip(np.abs((skin_n * wall_n).sum(1)), 0, 1)))
    assert before.min() > 75.0


def test_the_collar_is_wide_on_the_open_side_and_a_crease_on_the_folded_side(sphere):
    d = {}
    _collar(sphere, diag=d)
    n = len(d["W"])
    assert d["W"][0] > 0.8 * CO.CollarProfile().width * TUBE_R * np.cos(np.pi / 16)       # up: the full width
    assert not d["keep"][n // 2] and d["W"][n // 2] == 0.0                                  # down: no collar (an open ring)
    # no asymmetry asked: the same all round
    d2 = {}
    _collar(sphere, profile=CO.CollarProfile(ventral=CO.CollarProfile().width), diag=d2)
    assert d2["W"].min() > 0.9 * d2["W"].max()


def test_a_mirrored_pair_is_mirror_symmetric():
    v, f = _sphere_mesh()
    # two off-centre tubes on a symmetric body: the collar of the left one is the mirror image of the right's
    out = []
    for x in (2.0, -2.0):
        pts, rad = _axis(x=x)
        wall = _tube_mesh(x=x)
        patch = CO.SkinPatch(v, f, np.array([x, 0.0, np.sqrt(SPHERE_R ** 2 - x * x)]), 6.0)
        out.append(CO.build_collar(patch, pts, rad, wall, CO.CollarProfile()))
    a, b = (m.vertices for m in out)
    mirrored = a * np.array([-1.0, 1.0, 1.0])
    # compare as point sets (the vertex order of the mirrored ring is reversed)
    da = np.sort(np.linalg.norm(mirrored[:, None, :] - b[None, ::7, :], axis=2).min(axis=0))
    assert da.max() < 2e-3


def test_a_lean_over_the_skin_takes_the_collar_away_on_that_side():
    """A tube that leans over the skin (a lifted tail) has no room for a fillet under its leaning side."""
    v, f = _sphere_mesh()
    d0, d1 = {}, {}
    # lean the tube toward +X: the +X side (theta = 90 degrees from up) is the acute side
    for tilt, dd in ((0.0, d0), (55.0, d1)):
        pts, rad = _axis(tilt=tilt)
        wall = _tube_mesh(tilt=tilt)
        patch = CO.SkinPatch(v, f, np.array([0.0, 0.0, SPHERE_R]), 6.0)
        CO.build_collar(patch, pts, rad, wall, CO.CollarProfile(open_deg=170, fold_deg=179), dd)
    lean_side = int(np.argmin(d1["beta"]))
    assert d1["beta"].min() < 70.0 < d0["beta"].min()
    assert d1["W"][lean_side] < 0.5 * d0["W"][lean_side]


# -- what must fail without the fixes (the mutations the tests above and below are meant to catch) -------------------------


def test_the_sheet_goes_into_the_skin_at_the_asked_angle_and_a_steeper_one_shows_less_of_it(sphere):
    """``ramp`` is the tangent of the angle the sheet crosses the skin at: the near-flat outer part of the profile (less
    than a hair of height, the wide tinted 'patch' of round 3) is under the skin, more of it the steeper the crossing."""
    reach, angle = {}, {}
    for ramp in (0.05, 0.15):
        d = {}
        m, _, _ = _collar(sphere, profile=CO.CollarProfile(ramp=ramp), diag=d)
        keep = d["keep"] & (d["W"] > 0.3)
        n_vis = d["n_vis"]
        reach[ramp] = np.median((d["u_x"] * d["W"])[keep])
        rim = d["top"][n_vis - 1][keep]
        own = sh.vertex_normals(m.vertices, m.faces)[d["vid_top"][n_vis - 1][keep]]
        angle[ramp] = np.median(_ang(own, rim / np.linalg.norm(rim, axis=1, keepdims=True)))
    assert reach[0.15] < 0.9 * reach[0.05]
    for ramp, a in angle.items():
        assert a == pytest.approx(np.degrees(np.arctan(ramp)), abs=3.0), (ramp, a)


def test_the_collar_lifts_off_the_skin_only_by_the_profile(sphere):
    d = {}
    _collar(sphere, profile=CO.CollarProfile(height=0.5), diag=d)
    n_vis = d["n_vis"]
    h = np.linalg.norm(d["top"][1:n_vis - 1][:, d["keep"]], axis=2) - SPHERE_R
    assert h.max() > 0.2 and h.min() > -CO.SKIN_TOL
    d2 = {}
    _collar(sphere, profile=CO.CollarProfile(height=0.5, strength=0.5), diag=d2)
    assert np.linalg.norm(d2["top"][1:n_vis - 1][:, d2["keep"]], axis=2).max() - SPHERE_R < 0.7 * h.max()


def test_the_collar_is_sunk_under_the_skin_by_the_asked_amount_up_to_the_cap(sphere):
    """The underside need only be under the skin for the slicer's union: it is sunk by ``sink`` root radii up to
    ``SINK_MAX_REL`` (a deep underside under a curved skin crossed itself, round 3)."""
    depth = {}
    for sink in (0.06, 0.6, 0.0):
        d = {}
        _collar(sphere, profile=CO.CollarProfile(sink=sink), diag=d)
        depth[sink] = SPHERE_R - np.linalg.norm(d["bottom"][d["n_vis"] - 1][0])
        r_root = d["r_root"]
    assert depth[0.06] == pytest.approx(0.06 * r_root, rel=0.15)
    assert depth[0.6] == pytest.approx(CO.SINK_MAX_REL * r_root, rel=0.15)
    assert depth[0.0] < 0.2 * depth[0.06]


def test_a_tube_that_misses_the_skin_has_no_collar(sphere):
    v, f = sphere
    pts, rad = _axis(z0=11.0, z1=15.0)                       # floating above the pole
    patch = CO.SkinPatch(v, f, np.array([0.0, 0.0, SPHERE_R]), 6.0)
    assert CO.build_collar(patch, pts, rad, _tube_mesh(z0=11.0, z1=15.0), CO.CollarProfile()) is None
    pts, rad = _axis(z0=3.0, z1=8.0)                          # buried
    assert CO.build_collar(patch, pts, rad, _tube_mesh(z0=3.0, z1=8.0), CO.CollarProfile()) is None


def test_the_collar_survives_an_off_pole_tube_and_never_crosses_itself(sphere):
    """Round 4: zero, not 'few' (round 3 allowed 2 % of the faces: the hair-thin fold side; that side is now no
    collar at all, an open ring closed by two caps, and a guard cuts any column that still crosses)."""
    d = {}
    m, patch, _ = _collar(sphere, x=3.0, tilt=10.0, diag=d, centre=np.array([3.0, 0.0, np.sqrt(SPHERE_R ** 2 - 9)]))
    rep = m.integrity_report()
    assert rep["watertight"] and rep["outward"] and rep["degenerate_faces"] == 0
    assert CO.count_crossings(m.vertices, m.faces) == 0 and not d["keep"].all()


def test_the_crossing_counter_finds_a_pair_and_ignores_neighbours():
    a = Mesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.0]]), np.array([[0, 1, 2]]))
    cross = Mesh(np.array([[0.2, 0.2, -1], [0.2, 0.2, 1], [0.8, 0.8, 0.0]]), np.array([[0, 1, 2]]))
    assert CO.count_crossings(a.vertices, a.faces, cross.vertices, cross.faces) == 1
    assert CO.count_crossings(*(Mesh.merge([a, cross]).vertices, Mesh.merge([a, cross]).faces)) == 1
    quad = Mesh(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.0]]), np.array([[0, 1, 2], [0, 2, 3]]))
    assert CO.count_crossings(quad.vertices, quad.faces) == 0


# -- the real body ----------------------------------------------------------------------------------------------------------------


BODIES = {
    "neutral": dict(gender=1.0, muscle=0.55), "tall": dict(height=1.0, gender=1.0, muscle=0.55),
    "short": dict(height=0.0, gender=1.0, muscle=0.55), "child": dict(age=0.1875),
    "heavy": dict(weight=1.0, gender=1.0, muscle=1.0),
}


@pytest.fixture(scope="module")
def world():
    base, lib = mh.load()
    return base, lib, SliderCatalog(lib.names)


_BUILT: dict = {}


def _tail(world, body="neutral", controls=None, text="tail"):
    key = (body, json.dumps(controls, sort_keys=True), text)
    if key not in _BUILT:
        base, lib, cat = world
        c = Character()
        c.macro.update(BODIES[body])
        c.appendages.append(K.add_from_text(text, c.appendages, base=base, lib=lib))
        if controls:
            c.appendages[0].controls = dict(controls)
        verts = lib.apply(base.verts, c.effective_target_weights(cat))
        placed = c.instantiate_appendages(base, verts, lib)
        shaped = c.shaped_body(base, verts, placed)
        _BUILT[key] = (c, verts, placed[0], shaped)
    return _BUILT[key]


_SKIN: dict = {}


def _skin(world, body, controls, levels):
    key = (body, json.dumps(controls, sort_keys=True), levels)
    if key not in _SKIN:
        _, _, _, shaped = _tail(world, body, controls)
        _SKIN[key] = CO.skin_from_cage(world[0], shaped, levels)
    return _SKIN[key]


def _report(world, body="neutral", controls=None, levels=2, text="tail"):
    c, verts, p, shaped = _tail(world, body, controls, text)
    return CO.join_report(world[0], shaped, p, None, levels=levels, skin=_skin(world, body, controls, levels))


@assets
def test_the_tail_has_a_collar_of_its_own_and_other_kinds_do_not(world):
    assert K.KINDS["tail"].collar_profile(_tail(world)[2]).active
    base, lib, cat = world
    c = Character()
    for t in ("horn",):            # (the wing's collar: tests/test_wing_root.py)
        c.appendages.append(K.add_from_text(t, c.appendages, base=base, lib=lib))
    verts = lib.apply(base.verts, c.effective_target_weights(cat))
    for p in c.instantiate_appendages(base, verts, lib):
        if p.kind.id == "horn":
            continue                                    # (the horn's collar: tests/test_horn_collar.py, S6b horn)
        assert p.kind.collar_profile(p) is None and CO.collar_pieces(base, verts, p) == []


@assets
@pytest.mark.parametrize("body", list(BODIES))
def test_the_collar_meets_the_skin_and_the_tail_tangentially_on_every_body(world, body):
    """The join angle the junction alone leaves is 55-83 degrees (the skin runs into the wall); the collar's
    visible surface meets the skin and the wall at a few: p95 <= 10 degrees on the rim and the wall."""
    r = _report(world, body)
    assert r and r["watertight"] and r["outward"]
    assert r["rim_p95"] <= 10.0 and r["wall_p95"] <= 10.0, r
    assert r["before_p50"] > 35.0                                       # what it replaces (S7 tail masses: child 39.3)
    assert r["above_skin_min"] > -CO.SKIN_TOL and r["below_skin_vertices"] <= 20
    assert r["crease_p99"] < 45.0                                       # no crease of its own
    assert r["visible_share"] > 0.3


@assets
@pytest.mark.parametrize("body", ["neutral", "heavy"])
def test_the_collar_follows_the_subdivision_level_it_is_fitted_to(world, body):
    for lv in (1, 2):
        r = _report(world, body, levels=lv)
        assert r["rim_p95"] <= 10.0 and r["wall_p95"] <= 10.0 and r["above_skin_min"] > -CO.SKIN_TOL, (lv, r)
    # its rim lies on the surface it was fitted to, not on the other level's (CC shrinks every level)
    base = world[0]
    c, v, p, shaped = _tail(world, body)
    d1, d2 = {}, {}
    CO.collar_pieces(base, shaped, p, None, levels=1, skin=_skin(world, body, None, 1), diag=d1)
    CO.collar_pieces(base, shaped, p, None, levels=2, skin=_skin(world, body, None, 2), diag=d2)
    n_vis = d1["n_vis"]
    p1 = CO.SkinPatch(*_skin(world, body, None, 1), p.anchor, 3.0)
    p2 = CO.SkinPatch(*_skin(world, body, None, 2), p.anchor, 3.0)
    wide = d1["W"] > 0.3
    rim1 = d1["P"][n_vis - 1][wide]
    on1 = np.abs(p1.closest(rim1)[2]).max()
    on2 = np.abs(p2.closest(rim1)[2]).max()
    assert on1 < 1e-6 < on2                                                   # exactly on L1, off L2 by the CC shrink


@assets
def test_the_collar_follows_the_posed_tail(world):
    base = world[0]
    rest = _tail(world)
    lifted = _tail(world, controls={"lift": 40.0})
    wall_rest = rest[2].mesh()
    wall_lift = lifted[2].mesh()
    assert np.abs(wall_rest.vertices - wall_lift.vertices).max() > 0.5            # the pose moved the tail
    d_rest, d_lift = {}, {}
    CO.collar_pieces(base, rest[3], rest[2], None, levels=2, skin=_skin(world, "neutral", None, 2), diag=d_rest)
    CO.collar_pieces(base, lifted[3], lifted[2], None, levels=2, skin=_skin(world, "neutral", {"lift": 40.0}, 2),
                     diag=d_lift)
    assert np.abs(d_rest["E"] - d_lift["E"]).max() > 0.1                         # it moved with the tail
    # the join ring is on the POSED wall
    tris = wall_lift.triangles()
    for E in d_lift["E"][::6]:
        pt, _ = CO.closest_on_triangles(np.tile(E, (len(tris), 1)), tris[:, 0], tris[:, 1], tris[:, 2])
        assert np.sqrt(((pt - E) ** 2).sum(1).min()) < 0.02
    # and a pose that moves only the free tail (curl, droop, sway) leaves the root collar where it was
    curled = _tail(world, controls={"curl": 1.0, "droop": 0.5})
    d_curl = {}
    CO.collar_pieces(base, curled[3], curled[2], None, levels=2, skin=_skin(world, "neutral", {"curl": 1.0, "droop": 0.5}, 2),
                     diag=d_curl)
    assert np.abs(d_rest["E"] - d_curl["E"]).max() < 0.05


@assets
def test_at_level_zero_the_cage_is_drawn_and_there_is_no_collar(world):
    """Level 0 is the 0.28-unit cage, a coarse polyhedron: a collar fitted to it stands off in flaps (it was
    rendered and looked worse than none), so the pipelines draw none there and say so."""
    c, v, p, shaped = _tail(world)
    assert CO.collar_pieces(world[0], shaped, p, None, levels=0) == []
    assert CO.collar_pieces(world[0], shaped, p, None, levels=1, skin=_skin(world, "neutral", None, 1))
    assert CO.MIN_LEVEL == 1


@assets
def test_the_pose_controls_keep_the_collar_valid_on_every_body(world):
    for body in BODIES:
        for ctl in ({"lift": 40.0}, {"lift": 70.0}, {"sway": 40.0}):
            r = _report(world, body, ctl)
            assert r and r["watertight"] and r["outward"] and r["rim_p95"] <= 12.0, (body, ctl, r)


@assets
def test_a_thin_collar_on_the_underside_keeps_the_cleft_open(world):
    c, v, p, shaped = _tail(world)
    d = {}
    CO.collar_pieces(world[0], shaped, p, None, levels=2, skin=_skin(world, "neutral", None, 2), diag=d)
    n = len(d["W"])
    assert d["W"][0] > 0.5                                                                  # wide on the back ...
    # ... and none underneath (round 4: an open ring; round 3 had a 0.006-wide hair there, which crossed itself):
    # the cleft is not filled
    assert not d["keep"][n // 2] and d["W"][n // 2] == 0.0


@assets
def test_the_collar_strength_parameter_scales_it_and_zero_switches_it_off(world):
    base = world[0]
    sizes = {}
    for k in (0.5, 1.0, 1.6):
        c, v, p, shaped = _tail(world, text=f"tail:collar={k}")
        d = {}
        pieces = CO.collar_pieces(base, shaped, p, None, levels=1, skin=_skin_for(world, c, shaped), diag=d)
        assert len(pieces) == 1 and p.params["collar"] == k
        sizes[k] = d["W"].max()
    assert sizes[0.5] < sizes[1.0] < sizes[1.6] and sizes[1.0] == pytest.approx(2 * sizes[0.5], rel=0.15)
    c, v, p, shaped = _tail(world, text="tail:collar=0")
    assert K.KINDS["tail"].collar_profile(p) is None and CO.collar_pieces(base, shaped, p) == []
    # a record written before the parameter existed has the default
    values, notes = K.KINDS["tail"].clean({})
    assert values["collar"] == 1.0 and not notes


def _skin_for(world, c, shaped, levels=1):
    key = ("custom", id(shaped), levels)
    if key not in _SKIN:
        _SKIN[key] = CO.skin_from_cage(world[0], shaped, levels)
    return _SKIN[key]


@assets
def test_collar_pieces_names_the_shell_and_gives_it_the_skin_material(world):
    c, v, p, shaped = _tail(world)
    (name, mesh, mat), = CO.collar_pieces(world[0], shaped, p, None, levels=2, skin=_skin(world, "neutral", None, 2))
    assert name == "tail-1/collar" and mat is sh.SKIN and mesh.is_watertight
    (_, _, mat2), = CO.collar_pieces(world[0], shaped, p, None, levels=2, skin=_skin(world, "neutral", None, 2),
                                     profile=CO.CollarProfile(material="appendage"))
    assert mat2 is p.material


@assets
def test_a_collar_is_built_in_a_hundred_odd_milliseconds_warm(world):
    c, v, p, shaped = _tail(world)
    skin = _skin(world, "neutral", None, 2)
    CO.collar_pieces(world[0], shaped, p, None, levels=2, skin=skin)
    best = min(_timed(lambda: CO.collar_pieces(world[0], shaped, p, None, levels=2, skin=skin)) for _ in range(4))
    assert best < 0.45, best                                            # ~0.12 s on the dev machine; CI is slower


def _timed(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


@assets
def test_the_shaped_skin_cache_subdivides_a_body_once(world):
    base = world[0]
    c, v, p, shaped = _tail(world)
    CO._SKIN_CACHE.clear()
    a = CO.shaped_skin(base, shaped, 1)
    b = CO.shaped_skin(base, shaped, 1)
    assert a is b and len(CO._SKIN_CACHE) == 1
    CO.shaped_skin(base, shaped, 2)
    CO.shaped_skin(base, v, 1)
    assert len(CO._SKIN_CACHE) <= CO.SKIN_CACHE_ENTRIES                 # bounded


@assets
def test_skin_from_shell_is_the_pipelines_own_mesh_back_in_the_makehuman_frame(world):
    from minimaster.character import portrait as po
    base = world[0]
    c, v, p, shaped = _tail(world)
    shell = po.character_shells(base, shaped, levels=1, with_eyes=False)[0]
    V, F = CO.skin_from_shell(shell.mesh)
    V0, F0 = CO.skin_from_cage(base, shaped, 1)
    assert np.allclose(V, V0, atol=1e-9) and (F == F0).all()


@assets
def test_a_collar_that_cannot_be_built_is_skipped_with_a_warning(world, monkeypatch):
    c, v, p, shaped = _tail(world)

    def boom(*a, **k):
        raise ValueError("the skin is not where it was")

    monkeypatch.setattr(CO, "build_collar", boom)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = CO.collar_shells(world[0], shaped, [p], levels=1, skin=_skin_for(world, c, shaped))
    assert out == [] and any("no root collar" in str(w.message) for w in caught)
    assert CO.collar_shells(world[0], shaped, [p], enabled=False) == []


# -- the document -------------------------------------------------------------------------------------------------------------


def test_the_collar_switch_is_saved_only_when_it_is_off_and_the_junction_off_disables_it():
    c = Character()
    c.appendages.append(K.new_spec("tail"))                                  # the anatomy block is saved with appendages
    assert c.collar_enabled and "collar" not in json.dumps(c.to_dict())     # a document that never asked for it is unchanged
    c.anatomy["collar"] = False
    assert not c.collar_enabled
    back = Character.from_dict(json.loads(json.dumps(c.to_dict())))
    assert not back.collar_enabled and back.anatomy["collar"] is False
    c2 = Character()
    c2.anatomy["junction"] = False
    assert not c2.collar_enabled                                           # --no-junction leaves the skin as it was
    assert Character().to_dict() == json.loads(json.dumps(Character().to_dict()))      # a plain character is unchanged


# -- the print report -------------------------------------------------------------------------------------------------------------


def _tail_shells(world, offset=0.0, collar=True):
    base = world[0]
    c, v, p, shaped = _tail(world)
    skin = _skin(world, "neutral", None, 1)
    tail = p.mesh()
    if offset:
        tail = Mesh(tail.vertices + offset * np.array(p.matrix[:3, 2]) / np.linalg.norm(p.matrix[:3, 2]), tail.faces)
    group = [(p.name, tail)]
    if collar:
        (name, mesh, _), = CO.collar_pieces(base, shaped, p, None, levels=1, skin=skin)
        group.append((name, mesh))
    figure = pp.character_print_shells(base, shaped, levels=1)
    return base, shaped, figure, [group]


@assets
def test_the_report_counts_the_collar_in_the_group_but_not_in_the_weld_or_the_thickness(world):
    base, shaped, figure, apps = _tail_shells(world)
    with_c = PR.appendage_report(apps, base, shaped, 1.9, levels=1, figure_shells=figure, body_verts=shaped)
    without = PR.appendage_report([[apps[0][0]]], base, shaped, 1.9, levels=1, figure_shells=figure, body_verts=shaped)
    g1, g0 = with_c.groups["tail-1"], without.groups["tail-1"]
    assert g1.shells == 2 and g0.shells == 1 and not with_c.blocking
    assert g1.root_embed == pytest.approx(g0.root_embed, abs=0.03)        # the collar is no root: the tail is the weld
    assert g1.thin_p5_mm == pytest.approx(g0.thin_p5_mm, rel=1e-6)         # ... and no thickness measure of its own
    assert g1.volume_mm3 > g0.volume_mm3                                   # but it is part of the appendage's volume


@assets
def test_a_tail_that_only_touches_the_skin_still_blocks_with_a_collar_sunk_in_the_body(world):
    """The collar is embedded by construction: if it counted as a root, a hairline-welded tail would pass."""
    base, shaped, figure, apps = _tail_shells(world, offset=1.2)
    rep = PR.appendage_report(apps, base, shaped, 1.9, levels=1, figure_shells=figure, body_verts=shaped)
    assert any(f.level == "block" and f.code == "root-weld" for f in rep.findings), rep.as_text()


@assets
def test_a_collar_that_is_not_a_closed_shell_blocks_the_print(world):
    base, shaped, figure, apps = _tail_shells(world)
    (name, mesh), = [apps[0][1]]
    broken = Mesh(mesh.vertices, mesh.faces[:-3])
    rep = PR.appendage_report([[apps[0][0], (name, broken)]], base, shaped, 1.9, levels=1, figure_shells=figure,
                              body_verts=shaped)
    assert any(f.level == "block" and "collar" in f.where for f in rep.findings)


# -- the command line -------------------------------------------------------------------------------------------------------------


def _cli(tmp_path, *args):
    return subprocess.run([sys.executable, "-m", "minimaster", "character", *args], capture_output=True,
                          text=True, cwd=str(tmp_path.parent.parent) if False else None)


@assets
def test_the_cli_prints_the_collar_as_a_second_shell_and_no_collar_removes_it(tmp_path):
    out = tmp_path / "a.stl"
    done = subprocess.run([sys.executable, "-m", "minimaster", "character", "--add", "tail", "--subdivide", "1",
                           "-o", str(out)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "tail-1           2 shells" in done.stdout and "printability  OK" in done.stdout
    out2 = tmp_path / "b.stl"
    done = subprocess.run([sys.executable, "-m", "minimaster", "character", "--add", "tail", "--subdivide", "1",
                           "--no-collar", "-o", str(out2)], capture_output=True, text=True)
    assert done.returncode == 0 and "tail-1           1 shell" in done.stdout
    out3 = tmp_path / "c.stl"
    done = subprocess.run([sys.executable, "-m", "minimaster", "character", "--add", "tail", "--subdivide", "1",
                           "--no-junction", "-o", str(out3)], capture_output=True, text=True)
    assert done.returncode == 0 and "tail-1           1 shell" in done.stdout          # no junction: no collar either
    assert out.stat().st_size > out2.stat().st_size


@assets
def test_the_cli_at_level_zero_draws_no_collar_and_says_so(tmp_path):
    """Round 4: the DEFAULT level of a character with a collar is 1 (tests/test_collar_seam.py); an explicit
    ``--subdivide 0`` is honoured and says what it costs."""
    done = subprocess.run([sys.executable, "-m", "minimaster", "character", "--add", "tail", "--subdivide", "0",
                           "--res", "120", "160", "-o", str(tmp_path / "x.png")], capture_output=True, text=True)
    assert done.returncode == 0 and "needs a subdivided body" in done.stdout
    done = subprocess.run([sys.executable, "-m", "minimaster", "character", "--add", "tail", "--no-collar",
                           "--subdivide", "0", "--res", "120", "160", "-o", str(tmp_path / "y.png")],
                          capture_output=True, text=True)
    assert "needs a subdivided body" not in done.stdout                    # switched off: nothing to say
    done = subprocess.run([sys.executable, "-m", "minimaster", "character", "--add", "tail", "--subdivide", "0", "-o",
                           str(tmp_path / "z.stl")], capture_output=True, text=True)
    assert done.returncode == 0 and "tail-1           1 shell" in done.stdout


@assets
def test_the_cli_picture_has_the_collar_and_no_collar_changes_it(tmp_path):
    imgs = {}
    for tag, extra in (("on", []), ("off", ["--no-collar"])):
        out = tmp_path / f"{tag}.png"
        done = subprocess.run([sys.executable, "-m", "minimaster", "character", "--add", "tail", "--subdivide", "1",
                               "--azimuth", "150", "--elevation", "20", "--res", "300", "420", *extra, "-o", str(out)],
                              capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        imgs[tag] = out.read_bytes()
    assert imgs["on"] != imgs["off"]


@assets
def test_no_collar_is_saved_with_the_character_and_collar_switches_it_back(tmp_path):
    f = tmp_path / "c.mmchar"
    done = subprocess.run([sys.executable, "-m", "minimaster", "character", "--add", "tail", "--no-collar",
                           "--save-character", str(f), "-o", str(tmp_path / "x.png"), "--res", "100", "140"],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert json.loads(f.read_text())["anatomy"]["collar"] is False
    done = subprocess.run([sys.executable, "-m", "minimaster", "character", "--character", str(f), "--collar",
                           "--save-character", str(f), "-o", str(tmp_path / "x.png"), "--res", "100", "140"],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "collar" not in json.loads(f.read_text()).get("anatomy", {})


# -- the lab ----------------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lab_res():
    from minimaster.character import lab as L
    return L.Resources.shared()


@assets
def test_the_lab_draws_the_collar_in_the_final_render_only_and_caches_it(lab_res):
    from minimaster.character import lab as L
    d = L.LabDocument(lab_res)
    d.add_appendage("tail")
    eng = d.engine
    draft = eng.draft_scene(d.character)
    assert any("root collar" in n for n in draft.notes)                    # the draft says it skips it
    n_draft = len(draft.shells)
    final = eng.final_scene(d.character, levels=1)
    owners = [o for o in final.owners if o[0] == "tail-1"]
    assert len(owners) == 2                                                  # the tail and its collar
    assert eng.stats["collar"] == 1
    eng.final_scene(d.character, levels=1)
    assert eng.stats["collar"] == 1                                          # cached by body + spec + pose + level
    level0 = eng.final_scene(d.character, levels=0)
    assert len([o for o in level0.owners if o[0] == "tail-1"]) == 1 and any("needs a subdivided body" in n for n in level0.notes)
    d.set_anatomy(collar=False)
    assert len([o for o in eng.final_scene(d.character, levels=1).owners if o[0] == "tail-1"]) == 1
    assert "collar" in d.character.anatomy and d.character.anatomy["collar"] is False
    d.set_anatomy(collar=True)
    assert "collar" not in d.character.anatomy
    d.set_anatomy(junction=False)
    assert len([o for o in eng.final_scene(d.character, levels=1).owners if o[0] == "tail-1"]) == 1


@assets
def test_the_lab_refits_the_collar_when_the_pose_or_the_level_changes(lab_res):
    from minimaster.character import lab as L
    d = L.LabDocument(lab_res)
    spec = d.add_appendage("tail")
    eng = d.engine
    eng.final_scene(d.character, levels=1)
    eng.final_scene(d.character, levels=2)
    assert eng.stats["collar"] == 2                                          # another level, another fit
    d.set_control(spec.id, "lift", 40.0)
    eng.final_scene(d.character, levels=1)
    assert eng.stats["collar"] == 3                                          # another pose, another fit


@assets
def test_the_lab_export_carries_the_collar_in_the_group_and_reports_it(lab_res, tmp_path):
    from minimaster.character import lab as L
    d = L.LabDocument(lab_res)
    d.add_appendage("tail")
    res = d.engine.export(d.character, tmp_path / "x.stl", levels=1)
    assert res.written and not res.blocked
    assert res.printability.groups["tail-1"].shells == 2
    d.set_anatomy(collar=False)
    res2 = d.engine.export(d.character, None, levels=1)
    assert res2.printability.groups["tail-1"].shells == 1
