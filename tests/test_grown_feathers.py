"""The hair system and the particle system (``minimaster/core``), and the feathers grown with them.

The cores are tested on bare polylines and meshes: they know nothing about bodies.  The feather tests compare a
feather grown from its shaft with the sheet ``wings._blade`` draws from the same arguments; the wing tests
need the CC0 MakeHuman assets only where a body is involved.
"""

import inspect
import warnings

import numpy as np
import pytest

from minimaster.character import feather as FE
from minimaster.character import mhbase as mh
from minimaster.character import wings as W
from minimaster.core import hair as H
from minimaster.core import particles as PT
from minimaster.core.mesh import Mesh
from minimaster.core.subdiv import cube_cage, subdivided_mesh

assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")

# the arguments the wing gives four of its feathers (a slotted primary, a secondary, a greater and a lesser covert)
PRIMARY = ((5.89, 0.764, 0.2), dict(curl=0.152, cup=0.08, lead_frac=0.27, tip=0.84, tip_pow=2.4, emarg=0.72,
                                    emarg_in=0.32, tip_floor=0.5, root_floor=0.07, notch=0.6, rise_at=0.42, n=16,
                                    ridge=1.6, vee=0.1))
SECONDARY = ((3.773, 0.877, 0.2), dict(curl=0.089, cup=0.12, lead_frac=0.4, tip=0.8, tip_pow=2.8, tip_floor=0.2,
                                       root_floor=0.07, notch=0.6, rise_at=0.34, n=16, ridge=1.6, vee=0.1))
GREATER = ((1.498, 0.539, 0.2), dict(curl=0.04, cup=0.05, sweep=0.06, lead_frac=0.42, tip=0.712, tip_pow=1.7,
                                     calamus=0.0, rise_at=0.22, root_floor=0.55, tip_floor=0.0, n=10, m=6,
                                     ridge=1.25, vane=0.9, ridge_dir=1.0, butt=0.14, vee=0.08))
LESSER = ((0.594, 0.594, 0.2), dict(curl=0.3, cup=0.05, sweep=0.06, lead_frac=0.45, tip=0.4, tip_pow=1.7,
                                    calamus=0.0, rise_at=0.22, root_floor=0.55, tip_floor=0.0, n=10, m=4,
                                    ridge=1.25, vane=0.9, ridge_dir=-1.0, butt=0.14, vee=0.08))
FEATHERS = {"primary": PRIMARY, "secondary": SECONDARY, "greater": GREATER, "lesser": LESSER}


def _solid(m: Mesh) -> bool:
    rep = m.integrity_report()
    return rep["watertight"] and rep["outward"] and rep["degenerate_faces"] == 0 and rep["volume"] > 0


# -- the cores know nothing about bodies --------------------------------------------------------------

def test_the_two_systems_are_core_modules_with_no_body_in_them():
    for mod in (H, PT):
        src = inspect.getsource(mod)
        assert mod.__name__.startswith("minimaster.core.")
        assert "mhbase" not in src and "from ..character" not in src and "minimaster.character" not in src


# -- the hair system ----------------------------------------------------------------------------------

def test_strands_grow_from_their_roots_as_long_as_they_are_told_and_bend_as_they_go():
    roots = np.zeros((4, 3))
    s = H.grow(roots, [[1.0, 0.0, 0.0]] * 4, [1.0, 2.0, 3.0, 4.0], steps=12)
    assert s.points.shape == (4, 13, 3) and np.allclose(s.points[:, 0], 0.0)
    assert np.allclose(s.lengths(), [1, 2, 3, 4]) and np.allclose(s.points[:, -1, 0], [1, 2, 3, 4])
    arc = H.grow(roots, [[1.0, 0.0, 0.0]] * 4, [1.0, 2.0, 3.0, 4.0], steps=48, bend=np.pi / 2)
    assert np.allclose(arc.lengths(), [1, 2, 3, 4])
    end = arc.points[:, -1] - arc.points[:, -2]
    assert np.allclose(end / np.linalg.norm(end, axis=1, keepdims=True), [0.0, 0.0, -1.0], atol=0.04)   # a quarter turn
    # a quarter circle of length L has radius 2 L / pi: its end is (r, 0, -r)
    assert np.allclose(arc.points[:, -1, 0], 2 * np.array([1, 2, 3, 4]) / np.pi, rtol=2e-3)
    with pytest.raises(ValueError, match="N, K, 3"):
        H.Strands(np.zeros((3, 3)))


def test_steering_is_how_forces_and_collision_come_in():
    roots = np.zeros((2, 3))
    free = H.grow(roots, [[1.0, 0.0, 0.0]] * 2, 2.0, steps=10)
    same = H.grow(roots, [[1.0, 0.0, 0.0]] * 2, 2.0, steps=10, steer=lambda p, d, s: d)
    assert np.allclose(free.points, same.points)                        # no force: the same strands
    fall = H.grow(roots, [[1.0, 0.0, 0.0]] * 2, 2.0, steps=10, steer=lambda p, d, s: d + [0.0, -0.3, 0.0])
    assert np.allclose(fall.lengths(), 2.0) and fall.points[0, -1, 1] < -0.8

    def floor(p, d, s):                                                  # nothing goes under y = -0.2
        low = p[:, 1] < -0.2
        p[low, 1] = -0.2
        d = d + [0.0, -0.3, 0.0]
        d[low, 1] = np.maximum(d[low, 1], 0.0)
        return d
    held = H.grow(roots, [[1.0, 0.0, 0.0]] * 2, 4.0, steps=20, steer=floor)
    assert held.points[:, :, 1].min() > -0.2 - 0.3 and held.points[0, -1, 0] > 2.5      # it slid along the floor


def test_a_haircut_cuts_every_strand_where_it_leaves_the_region():
    s = H.grow(np.zeros((3, 3)), [[1.0, 0.0, 0.0]] * 3, [1.0, 2.0, 3.0], steps=16)
    inside = lambda p: 1.5 - p[:, 0]                                     # the region: x < 1.5
    assert np.allclose(H.cut_length(s, inside), [1.0, 1.5, 1.5])
    t = H.trim(s, inside, at=(0.0, 0.5, 1.0))
    assert t.points.shape == (3, 3, 3) and np.allclose(t.points[:, -1, 0], [1.0, 1.5, 1.5])
    assert np.allclose(t.points[:, 1, 0], [0.5, 0.75, 0.75]) and np.allclose(t.attrs["kept"], [1.0, 1.5, 1.5])
    shorter = H.trim(s, inside, at=(1.0,), scale=[1.0, 0.5, 1.0])
    assert np.allclose(shorter.points[:, 0, 0], [1.0, 0.75, 1.5])
    own = H.trim(s, inside, at=np.array([[1.0], [0.2], [0.4]]))          # each strand its own fractions
    assert np.allclose(own.points[:, 0, 0], [1.0, 0.3, 0.6])
    assert np.allclose(H.cut_length(s, lambda p: -np.ones(len(p))), 0.0)  # all outside: nothing is left


def test_children_clumps_and_roughness():
    g = H.grow(np.array([[0.0, 0, 0], [0, 0, 1.0], [0, 0, 2.0]]), [[1.0, 0, 0]] * 3, [1.0, 2.0, 1.0], steps=4)
    kids = H.between(g, 3)
    assert len(kids) == 9 and np.allclose(kids.points[[0, 4, 8]], g.points)       # the guides stay, in order
    assert np.allclose(kids.points[2, 0], [0.0, 0.0, 0.5]) and np.allclose(kids.lengths()[2], 1.5)
    assert list(kids.attrs["guide"]) == [0, 0, 0, 0, 1, 1, 1, 1, 2]
    fan = H.grow(np.zeros((4, 3)), [[1, 0, 0.3], [1, 0, 0.1], [1, 0, -0.1], [1, 0, -0.3]], 1.0, steps=6)
    lock = H.clump(fan, [0, 0, 0, 0], amount=0.8)
    assert np.allclose(lock.points[:, 0], 0.0)                                    # not at the root
    assert np.ptp(lock.points[:, -1, 2]) < 0.25 * np.ptp(fan.points[:, -1, 2])    # drawn together at the tip
    two = H.clump(fan, [0, 0, 1, 1], amount=1.0)
    assert np.allclose(two.points[0, -1], two.points[1, -1]) and not np.allclose(two.points[1, -1], two.points[2, -1])
    r1, r2 = H.roughen(fan, 0.05, seed=3), H.roughen(fan, 0.05, seed=3)
    assert np.array_equal(r1.points, r2.points) and np.allclose(r1.points[:, 0], 0.0)
    assert 1e-3 < np.abs(r1.points - fan.points).max() <= 0.05 + 1e-9
    assert not np.array_equal(H.roughen(fan, 0.05, seed=4).points, r1.points)
    h = H.hash01(np.arange(1000), 7)
    assert h.min() >= 0.0 and h.max() < 1.0 and 0.45 < h.mean() < 0.55 and np.array_equal(h, H.hash01(np.arange(1000), 7))


def _chevrons(R=9, C=7):
    x, z = np.meshgrid(np.linspace(0.0, 2.0, R), np.linspace(-0.5, 0.5, C), indexing="ij")
    return np.stack([x + 0.4 * np.abs(z), np.zeros_like(x), z], axis=2)


def test_zipped_strands_are_one_closed_slab_with_a_coarse_far_face():
    g = _chevrons()
    R, C = g.shape[:2]
    fine = np.full((R, C), 0.05)
    fine[1::2] += 0.02                                                    # every second strand a ridge
    coarse = np.full((R, C), 0.05)
    wall = H.sheet_mesh(g, fine, coarse)
    tipped = H.sheet_mesh(g, fine, coarse, tip=(2.6, 0.0, 0.0), tip_fine=0.03, tip_coarse=0.03)
    for m in (wall, tipped):
        assert _solid(m)
    # the fine face has every row and column, the coarse one the even rows and three columns
    fine_tris, coarse_tris = 2 * (R - 1) * (C - 1), 2 * ((R - 1) // 2) * 2
    assert coarse_tris * 6 == fine_tris
    assert len(wall.faces) == fine_tris + coarse_tris + 2 * 3 * ((R - 1) // 2) + 2 * (C - 1 + 2)
    top = wall.vertices[:R * C].reshape(R, C, 3)
    assert np.allclose(top[1::2, :, 1], 0.07) and np.allclose(top[0::2, :, 1], 0.05)      # the ridges, on this face
    assert np.allclose(wall.vertices[R * C:, 1], -0.05) and len(wall.vertices) == R * C + ((R + 1) // 2) * 3
    # the same slab whichever way the normals are given (the fine face follows them)
    turned = H.sheet_mesh(g, fine, coarse, normals=-H._grid_normals(g))
    assert _solid(turned) and np.allclose(turned.vertices[:R * C].reshape(R, C, 3)[1::2, :, 1], -0.07)
    assert np.isclose(turned.volume(), wall.volume())
    with pytest.raises(ValueError, match="odd number of rows"):
        H.sheet_mesh(g[:8], fine[:8], coarse[:8])


def test_many_sheets_at_once_are_the_same_sheets():
    g = np.stack([_chevrons(), _chevrons() * [1.5, 1.0, 0.7] + [0, 0.2, 0]])
    fine = np.full(g.shape[:3], 0.04)
    fine[:, 1::2] += 0.015
    coarse = np.full(g.shape[:3], 0.04)
    tips = np.array([[2.6, 0.0, 0.0], [3.9, 0.2, 0.0]])
    batch = H.sheet_meshes(g, fine, coarse, tips, 0.02, 0.02)
    for i, m in enumerate(batch):
        one = H.sheet_mesh(g[i], fine[i], coarse[i], tip=tips[i], tip_fine=0.02, tip_coarse=0.02)
        assert np.array_equal(m.vertices, one.vertices) and np.array_equal(m.faces, one.faces) and _solid(m)
    assert batch[0].faces is batch[1].faces or np.array_equal(batch[0].faces, batch[1].faces)   # one set of triangles


def test_a_tube_a_strand_is_closed_and_costs_six_times_a_zipped_segment():
    s = H.grow(np.array([[0.0, 0, 0], [0, 0, 1.0]]), [[1.0, 0.2, 0]] * 2, 2.0, steps=8, bend=0.5)
    tubes = H.tube_mesh(s, 0.05, sides=6)
    assert _solid(tubes) and len(tubes.faces) == 2 * (8 * 12 + 12)
    assert 12 == 6 * 2                                                    # (triangles a segment: tube 12, sheet 2)


# -- the particle system ------------------------------------------------------------------------------

def test_particles_along_a_curve_head_along_it():
    c = PT.emit_curve(np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 1.0, 0]]), count=4)
    assert np.allclose(c.position, [[0.25, 0, 0], [0.75, 0, 0], [1, 0.25, 0], [1, 0.75, 0]])
    assert np.allclose(c.heading, [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]])
    assert np.allclose(np.einsum("nij,nik->njk", c.frame, c.frame), np.eye(3))
    assert np.allclose(c.attrs["at"], [0.125, 0.375, 0.625, 0.875])
    at = PT.emit_curve(np.array([[0.0, 0, 0], [2.0, 0, 0]]), at=[0.0, 0.5, 1.0], size=[1.0, 2.0, 3.0])
    assert np.allclose(at.position[:, 0], [0.0, 1.0, 2.0]) and list(at.size) == [1.0, 2.0, 3.0]


def test_particles_on_a_bare_mesh_are_even_repeatable_and_follow_it():
    ball = subdivided_mesh(*cube_cage(2.0), levels=3)
    a, b = PT.emit_surface(ball, 200, seed=1), PT.emit_surface(ball, 200, seed=1)
    assert len(a) == 200 and np.array_equal(a.position, b.position)
    assert not np.array_equal(PT.emit_surface(ball, 200, seed=2).position, a.position)
    assert np.allclose(a.attrs["bary"].sum(axis=1), 1.0) and (a.attrs["bary"] >= 0).all()
    tri = ball.vertices[ball.faces[a.attrs["face"]]]
    assert np.allclose((tri * a.attrs["bary"][:, :, None]).sum(axis=1), a.position)      # on its triangle
    assert np.allclose(np.einsum("nij,nik->njk", a.frame, a.frame), np.eye(3), atol=1e-9)
    assert ((a.up * a.position).sum(axis=1) > 0).all()                                    # up is out of the ball
    d = np.linalg.norm(a.position[:, None] - a.position[None], axis=2) + 9 * np.eye(200)
    near = d.min(axis=1)
    assert near.min() > 0.6 * np.median(near)                                             # no two crowd
    half = d[:100, :100].min(axis=1)
    assert half.min() > 1.2 * near.min()                                                  # the first half: wider
    # a triangle whose density is 0 gets none
    upper = (ball.vertices[ball.faces].mean(axis=1)[:, 1] > 0).astype(float)
    top = PT.emit_surface(ball, 100, density=upper)
    assert len(top) == 100 and (ball.vertices[ball.faces[top.attrs["face"]]].mean(axis=1)[:, 1] > 0).all()
    assert len(PT.emit_surface(ball, 50, density=np.zeros(len(ball.faces)))) == 0
    # the mesh moves: the particles go with it
    moved = Mesh(ball.vertices * 1.5 + [1.0, 0.0, 0.0], ball.faces)
    assert np.allclose(a.follow(moved).position, a.position * 1.5 + [1.0, 0.0, 0.0])
    assert np.allclose(a.follow(moved).up, a.up)


def test_instancing_is_one_template_and_n_frames_realised_at_once():
    ball = subdivided_mesh(*cube_cage(2.0), levels=2)
    ps = PT.jitter(PT.emit_surface(ball, 60, seed=5), seed=2, yaw=25.0, tilt=10.0, size=0.3)
    assert 0.7 - 1e-9 <= ps.size.min() and ps.size.max() <= 1.3 + 1e-9 and np.ptp(ps.size) > 0.3
    assert np.allclose(np.einsum("nij,nik->njk", ps.frame, ps.frame), np.eye(3), atol=1e-9)
    spike = Mesh(np.array([[0, 0, .1], [0, 0, -.1], [0, .1, 0], [.5, 0, 0], [0, -.05, 0]], float),
                 np.array([[0, 3, 2], [2, 3, 1], [1, 3, 4], [4, 3, 0], [0, 2, 1], [0, 1, 4]]))
    assert _solid(spike)
    one = PT.instance(spike, ps)
    assert len(one.faces) == 60 * 6 and len(one.vertices) == 60 * 5 and _solid(one)
    assert np.isclose(one.volume(), spike.volume() * (ps.size ** 3).sum())
    many = PT.instance(spike, ps, merge=False)
    assert len(many) == 60 and np.allclose(np.concatenate([m.vertices for m in many]), one.vertices)
    tip = many[7].vertices[3]                                              # the template's +X end
    assert np.allclose(tip, ps.position[7] + 0.5 * ps.size[7] * ps.heading[7])
    # a left-handed frame places the mirror image, and it is still a solid
    mirrored = PT.Particles(ps.position, ps.frame * [1.0, 1.0, -1.0], ps.size)
    assert _solid(PT.instance(spike, mirrored)) and all(_solid(m) for m in PT.instance(spike, mirrored, merge=False))
    m = ps.matrices()
    assert m.shape == (60, 4, 4) and np.allclose(m[:, :3, 3], ps.position)


def test_hair_grows_from_particles():
    ball = subdivided_mesh(*cube_cage(2.0), levels=2)
    ps = PT.emit_surface(ball, 80, seed=1)
    hair = H.grow(ps.position, ps.up, 0.6, steps=6, steer=lambda p, d, s: d + [0.0, -0.4, 0.0])
    assert len(hair) == 80 and np.allclose(hair.points[:, 0], ps.position)
    assert np.allclose(hair.lengths(), 0.6)
    assert (hair.points[:, -1, 1] < hair.points[:, 0, 1] + 0.6 * ps.up[:, 1] + 1e-9).all()   # it droops
    assert _solid(H.tube_mesh(hair, 0.02))


# -- a feather grown from its shaft ---------------------------------------------------------------------

def _widths(m: Mesh, length, stations, sweep=0.08):
    """The (lead, trail) reach of a feather's outline from its shaft at ``stations`` of its length, from its
    vertices (the shaft bows ``sweep * length * t**2`` to the trailing side)."""
    v = m.vertices
    out = []
    for t in stations:
        near = np.abs(v[:, 0] / length - t) < 0.04
        z = v[near, 2] - sweep * length * (v[near, 0] / length) ** 2
        out.append((-z.min(), z.max()))
    return np.array(out)


def _spec(a, k):
    from minimaster.character import wings as W_
    d = {**FE._DEFAULTS, **k, "length": a[0], "width": a[1], "thickness": a[2]}
    d["edge"] = W_.EDGE if d["edge"] is None else d["edge"]
    d["cap"] = W_.TIP_CAP if d["cap"] is None else d["cap"]
    return d


def _edges(m: Mesh, spec):
    """The two edges of a grown feather: ``(t, lead reach, trail reach)`` per row of its barbed face."""
    B, C = FE.barb_count(spec)
    R = 2 * B + 1
    g = m.vertices[:R * C].reshape(R, C, 3)
    L, sw = spec["length"], spec["sweep"]
    lead, trail = g[:, 0], g[:, -1]
    return (lead[:, 0] / L, sw * L * (lead[:, 0] / L) ** 2 - lead[:, 2],
            trail[:, 0] / L, trail[:, 2] - sw * L * (trail[:, 0] / L) ** 2)


@pytest.mark.parametrize("name", list(FEATHERS))
def test_a_grown_feather_is_the_same_feather_as_the_sheet(name):
    a, k = FEATHERS[name]
    spec = _spec(a, k)
    sheet, grown = W._blade(*a, **k), FE.barbed_blade(*a, **k)
    assert _solid(grown) and len(grown.faces) <= len(sheet.faces)             # no more triangles than the sheet
    assert 0.9 < grown.volume() / sheet.volume() < 1.12
    lo_s, hi_s = sheet.bounds
    lo_g, hi_g = grown.bounds
    assert np.allclose(lo_g[[0, 2]], lo_s[[0, 2]], atol=0.05 * a[1] + 0.03)   # the same plan: length and both edges
    assert np.allclose(hi_g[[0, 2]], hi_s[[0, 2]], atol=0.05 * a[1] + 0.03)
    # every barb ends ON the sheet's outline (or short of it by its fray), and the outline here IS the sheet's
    p = {key: np.array([[float(spec[key])]]) for key in FE._NUMBERS}
    tl, lead, tt, trail = _edges(grown, spec)
    want_l = FE._outline(np.clip(tl, 0, 1)[None, :], p)[0][0]
    want_t = FE._outline(np.clip(tt, 0, 1)[None, :], p)[1][0]
    body = slice(2, -2)                                                       # (not the rows of the very ends)
    slack = 0.5 * a[2]                                                        # the sheet's own thickness at its edge
    assert (lead[body] < want_l[body] + slack).all() and (trail[body] < want_t[body] + slack).all()
    assert (lead[body] > (1.0 - FE.BARB_FRAY) * want_l[body] - slack).all()
    assert (trail[body] > (1.0 - FE.BARB_FRAY) * want_t[body] - slack).all()
    # ... the sheet's own edge columns lie on that outline too
    n_rows = len(sheet.vertices) // 2 // (int(spec["m"]) + 1)
    top = sheet.vertices[:n_rows * (int(spec["m"]) + 1)].reshape(n_rows, -1, 3)
    t = top[:, 0, 0] / a[0]
    off = spec["sweep"] * a[0] * t ** 2
    wl, wt = FE._outline(np.clip(t, 0, 1)[None, :], p)
    assert np.abs((off - top[:, 0, 2]) - wl[0]).max() < slack and np.abs((top[:, -1, 2] - off) - wt[0]).max() < slack


def test_the_notch_of_a_slotted_primary_is_cut_by_its_barbs():
    a, k = PRIMARY
    g = FE.barbed_blade(*a, **k)
    w = _widths(g, a[0], [0.45, 0.8])
    assert w[1, 0] < 0.45 * w[0, 0]                       # past the notch the leading vane is under half as wide
    assert w[1, 1] < 0.85 * w[0, 1]                       # and the trailing one is cut in too
    plain = FE.barbed_blade(*a, **{**k, "emarg": 0.0, "emarg_in": 0.0})
    assert _widths(plain, a[0], [0.8])[0, 0] > 1.8 * w[1, 0]


def test_every_barb_is_a_ridge_on_the_face_the_shaft_stands_off():
    a, k = SECONDARY
    B, C = FE.barb_count(_spec(a, k))
    assert (B, C) == (20, 7)
    R = 2 * B + 1
    g = FE.barbed_blade(*a, **k)
    smooth = FE.barbed_blade(*a, **{**k, "relief": 0.0})
    assert len(g.vertices) == R * C + (B + 1) * 3 + 2 and len(g.faces) == 34 * B + 20
    top, flat = g.vertices[:R * C].reshape(R, C, 3), smooth.vertices[:R * C].reshape(R, C, 3)
    rise = np.linalg.norm(top - flat, axis=2)             # how far each vertex stands over the smooth vane
    mid = slice(B // 4, 3 * B // 4)                       # (barbs of the middle of the vane)
    assert np.allclose(rise[0::2], 0.0)                   # the rows between barbs: where the smooth vane is
    assert rise[1::2][mid, 1].min() > 0.15 * 0.2          # a barb's ridge, at the bevel line of the leading vane
    assert rise[1::2][mid, 5].min() > 0.15 * 0.2          # ... and of the trailing one
    assert np.allclose(rise[:, C // 2], 0.0)              # none on the shaft itself
    assert np.array_equal(g.vertices[R * C:], smooth.vertices[R * C:])        # the far face is the same sheet
    # the ridges are on the ridge_dir face (-Y here), the coarse face on the other
    assert top[R // 2, :, 1].mean() < g.vertices[R * C:R * C + (B + 1) * 3, 1].mean()
    up = FE.barbed_blade(*a, **{**k, "ridge_dir": 1.0})
    assert _solid(up) and up.vertices[:R * C, 1].mean() > up.vertices[R * C:-2, 1].mean()


def test_barbs_leave_the_shaft_for_the_tip_and_sweep_towards_it():
    s = np.linspace(0.0, 0.95, 20)
    st = FE.barb_strands(s, [3.0], [0.0], W.BARB_ANGLE, [1.0])
    assert len(st) == 40 and list(st.attrs["side"][:20]) == [-1.0] * 20 and list(st.attrs["side"][20:]) == [1.0] * 20
    P = st.points
    assert np.allclose(P[:20, 0], P[20:, 0]) and np.allclose(P[:20, 0, 0], 3.0 * s)      # pairs leave together
    assert (P[:20, -1, 2] < 0).all() and (P[20:, -1, 2] > 0).all()                        # leading vane on -Z
    first = np.degrees(np.arctan2(np.abs(P[:, 1, 2] - P[:, 0, 2]), P[:, 1, 0] - P[:, 0, 0]))
    last = np.degrees(np.arctan2(np.abs(P[:, -1, 2] - P[:, -2, 2]), P[:, -1, 0] - P[:, -2, 0]))
    assert (last < first).all()                                                            # it sweeps to the tip
    assert first[0] > 80.0 and abs(first[8] - W.BARB_ANGLE) < 6.0 and first[19] < first[8]   # square at the base,
    #                                                                 at BARB_ANGLE along the vane, acute at the tip


def test_detail_buys_barbs_and_a_seed_frays_the_edge_its_own_way():
    a, k = SECONDARY
    base = FE.barbed_blade(*a, **k)
    fine = FE.barbed_blade(*a, **{**k, "detail": 2.0})
    assert _solid(fine) and 1.9 < len(fine.faces) / len(base.faces) < 2.1
    assert FE.barb_count({**_spec(a, k), "detail": 2.0})[0] == 40
    six = FE.barbed_blade(*a, **{**k, "count": 6})
    assert _solid(six) and len(six.faces) == 34 * 6 + 20
    again, other = FE.barbed_blade(*a, **{**k, "seed": 3}), FE.barbed_blade(*a, **{**k, "seed": 4})
    assert np.array_equal(again.vertices, FE.barbed_blade(*a, **{**k, "seed": 3}).vertices)
    assert not np.array_equal(again.vertices, other.vertices)
    assert np.abs(again.vertices - other.vertices).max() < 0.05 * a[1] + 0.02             # only the fray differs


def test_a_split_is_a_barb_cut_short():
    a, k = SECONDARY
    whole = FE.barbed_blade(*a, **k)
    split = FE.barbed_blade(*a, **{**k, "splits": ((0.6, 0.35),)})
    assert _solid(split) and len(split.faces) == len(whole.faces)
    R, C = 41, 7
    w, s = whole.vertices[:R * C].reshape(R, C, 3), split.vertices[:R * C].reshape(R, C, 3)
    moved = np.linalg.norm(w - s, axis=2).max(axis=1)
    r = int(np.argmax(moved))
    assert r % 2 == 0 and abs(r / R - 0.6) < 0.05 and moved[r] > 0.1        # a row between two barbs, at the split
    others = np.delete(moved, r)
    assert others.max() < 0.02 and (others > 1e-9).sum() <= 2               # (its two neighbours' normals turn a hair)
    assert np.abs(w[r, :3] - s[r, :3]).max() < 0.02                          # the leading vane is whole
    reach = lambda g: np.linalg.norm(g[r, 6] - g[r, 3])                      # shaft to trailing edge along the row
    assert np.isclose(reach(s), (1.0 - 0.35) * reach(w), rtol=0.03)


def test_a_wings_feathers_are_grown_together_as_a_handful_of_strand_systems():
    names = ["primary", "greater", "secondary", "lesser", "greater", "secondary"]
    specs = [dict(FEATHERS[n][1], length=FEATHERS[n][0][0], width=FEATHERS[n][0][1], thickness=0.2, seed=i)
             for i, n in enumerate(names)]
    batch = FE.barbed_blades(specs)
    assert len(batch) == 6
    for sp, m in zip(specs, batch):                                          # in order, and the same feather
        one = FE.barbed_blade(sp["length"], sp["width"], sp["thickness"],
                              **{k: v for k, v in sp.items() if k not in ("length", "width", "thickness")})
        assert np.allclose(m.vertices, one.vertices, atol=1e-12) and np.array_equal(m.faces, one.faces)
    sizes = {FE.barb_count({**FE._DEFAULTS, **sp, "edge": W.EDGE, "cap": W.TIP_CAP}) for sp in specs}
    assert sizes == {(25, 7), (20, 7), (11, 5), (7, 5)}


def test_a_grown_feather_never_has_more_triangles_than_its_sheet():
    """Its barbs are counted from what ``_blade`` spends on the same feather (the sheet's own rows and
    columns), whatever its shape."""
    k0 = dict(SECONDARY[1])
    for length in (1.2, 3.0, 6.5):
        for width in (0.3, 0.9):
            for n in (8, 11, 16, 24):
                for m in (4, 6, 8):
                    for extra in ({}, {"emarg": 0.7, "emarg_in": 0.3}, {"tip_floor": 0.5}, {"tip_floor": 0.0},
                                  {"splits": ((0.6, 0.3),)}):
                        k = {**k0, "n": n, "m": m, **extra}
                        sheet, grown = W._blade(length, width, 0.2, **k), FE.barbed_blade(length, width, 0.2, **k)
                        assert len(grown.faces) <= len(sheet.faces), (length, width, n, m, extra)
                        assert _solid(grown)


def test_barbs_on_both_faces_for_a_feather_seen_from_underneath():
    a, k = SECONDARY
    B, C = FE.barb_count(_spec(a, k))
    R = 2 * B + 1
    one, two = FE.barbed_blade(*a, **k), FE.barbed_blade(*a, **{**k, "both": True})
    smooth = FE.barbed_blade(*a, **{**k, "both": True, "relief": 0.0})
    assert _solid(two) and len(two.vertices) == 2 * R * C + 2
    assert 1.55 < len(two.faces) / len(one.faces) < 1.75                      # two thirds more
    assert np.array_equal(two.vertices[:R * C], one.vertices[:R * C])         # the near face is the same
    far, flat = two.vertices[R * C:2 * R * C].reshape(R, C, 3), smooth.vertices[R * C:2 * R * C].reshape(R, C, 3)
    rise = np.linalg.norm(far - flat, axis=2)
    assert np.allclose(rise[0::2], 0.0) and rise[1::2][B // 4:3 * B // 4, 1].min() > 0.15 * 0.2   # ridges there too
    assert np.allclose(rise[:, C // 2], 0.0)                                  # the shaft's ridge is on one face only
    # mixed in one batch, each comes back as asked
    batch = FE.barbed_blades([dict(k, length=a[0], width=a[1], thickness=a[2], both=b) for b in (True, False, True)])
    assert [len(m.faces) for m in batch] == [len(two.faces), len(one.faces), len(two.faces)]


# -- on the wing ----------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rigs():
    return W.feathered_rig(), W.feathered_rig(grown=1.0, underside=False)


def test_the_wing_keeps_its_sheets_unless_asked(rigs):
    sheets, _ = rigs
    assert len(sheets.pieces) == 275 and sum(len(p.mesh.faces) for p in sheets.pieces) == 91880
    again = W.feathered_rig(grown=0.0)
    assert all(np.array_equal(a.mesh.vertices, b.mesh.vertices) for a, b in zip(sheets.pieces, again.pieces))


def test_a_grown_wing_is_the_same_wing_with_barbs_for_fewer_triangles(rigs):
    """(With the flight feathers' far faces plain: ``underside=False``.  Barbed underneath as well, the default,
    is the next test.)"""
    sheets, grown = rigs
    assert [p.name for p in grown.pieces] == [p.name for p in sheets.pieces]
    assert [p.bone for p in grown.pieces] == [p.bone for p in sheets.pieces]
    assert list(grown.skeleton.joints) == list(sheets.skeleton.joints)
    assert all(np.allclose(grown.skeleton.joints[j], sheets.skeleton.joints[j]) for j in sheets.skeleton.joints)
    assert all(isinstance(p.mesh, Mesh) and _solid(p.mesh) for p in grown.pieces)
    t_s, t_g = (sum(len(p.mesh.faces) for p in r.pieces) for r in (sheets, grown))
    assert t_g < t_s and t_g > 0.9 * t_s
    changed = 0
    for a, b in zip(sheets.pieces, grown.pieces):
        if np.array_equal(a.mesh.faces, b.mesh.faces) and np.array_equal(a.mesh.vertices, b.mesh.vertices):
            continue                                                         # (a bone, the alula's mount)
        changed += 1
        lo_a, hi_a = a.mesh.bounds
        lo_b, hi_b = b.mesh.bounds
        size = float(np.linalg.norm(hi_a - lo_a))
        assert np.abs(lo_a - lo_b).max() < 0.08 * size + 0.03 and np.abs(hi_a - hi_b).max() < 0.08 * size + 0.03, a.name
        assert 0.85 < b.mesh.volume() / a.mesh.volume() < 1.15, a.name
        assert len(b.mesh.faces) <= len(a.mesh.faces), a.name                # feather for feather, no more triangles
    assert changed == 271                                                    # every feather, nothing else


def test_by_default_a_grown_wings_flight_feathers_are_barbed_underneath_too(rigs):
    sheets, plain = rigs
    both = W.feathered_rig(grown=1.0)
    assert [p.name for p in both.pieces] == [p.name for p in plain.pieces]
    more = [a.name for a, b in zip(plain.pieces, both.pieces) if len(b.mesh.faces) > len(a.mesh.faces)]
    rows = {"".join(ch for ch in n if not ch.isdigit()) for n in more}
    assert rows == {"primary", "secondary", "tertial", "scapular", "alula"} and len(more) == 41
    assert all(_solid(p.mesh) for p in both.pieces)
    t_s, t_p, t_b = (sum(len(p.mesh.faces) for p in r.pieces) for r in (sheets, plain, both))
    assert t_p < t_s < t_b < 1.2 * t_s                                       # a fifth more than the sheets, at most


def test_a_grown_wing_folds_and_finer_barbs_are_finer(rigs):
    grown = W.feathered_rig(grown=1.0)
    spread, folded = grown.pose(None), grown.pose("folded")
    assert len(folded.faces) == len(spread.faces) and np.isfinite(folded.vertices).all()
    assert np.ptp(folded.vertices[:, 0]) < 0.75 * np.ptp(spread.vertices[:, 0])
    finer = W.feathered_rig(grown=2.0)
    t1, t2 = (sum(len(p.mesh.faces) for p in r.pieces) for r in (grown, finer))
    assert 1.8 < t2 / t1 < 2.1 and all(_solid(p.mesh) for p in finer.pieces)


@assets
def test_a_grown_wing_on_a_figure_prints_better_than_the_sheets_at_26_mm():
    from minimaster.character import kinds as K
    from minimaster.character import print_report as PR
    from minimaster.character.sliders import Character, SliderCatalog
    base, lib = mh.load()

    def report(text, mm):
        c = Character()
        c.macro.update(gender=1.0, muscle=0.55)
        c.appendages.append(K.add_from_text(text, c.appendages, base=base, lib=lib))
        verts = lib.apply(base.verts, c.effective_target_weights(SliderCatalog(lib.names)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            placed = c.instantiate_appendages(base, verts, lib)
        left, right = placed
        twin = right.mesh().vertices * [-1.0, 1.0, 1.0]
        assert np.abs(np.sort(left.mesh().vertices, axis=0) - np.sort(twin, axis=0)).max() < 1e-6    # mirror images
        return PR.appendage_report([p.shells() for p in placed], base, verts, mm / 17.5,
                                   body_verts=c.shaped_body(base, verts, placed))

    assert K.KINDS["wing"].defaults(style="feathered")["grown"] == 0.0
    grown32, grown26, sheet26 = report("wing:style=feathered,grown=1", 32.0), report("wing:style=feathered,grown=1", 26.0), \
        report("wing:style=feathered", 26.0)
    for rep in (grown32, grown26):
        g = rep.groups["wing-1.l"]
        assert rep.ok and g.shells == 340 and g.components == 1 and not g.detached
        assert not any(f.code == "thin" for f in rep.findings)
    assert grown32.groups["wing-1.l"].thin_below_thin < 0.08
    assert any(f.code == "thin" for f in sheet26.findings)                   # the sheets warn at this size
    assert grown26.groups["wing-1.l"].thin_below_thin < 0.5 * sheet26.groups["wing-1.l"].thin_below_thin
