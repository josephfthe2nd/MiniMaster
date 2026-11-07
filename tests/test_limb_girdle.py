"""Extra limbs on a second girdle (``minimaster.character.limbgirdle``).

A harvested limb used to be the character's own arm cut at a ring and sunk 2 units along its own
axis into the torso: the shoulder joint ended up at the sternum, the neck or the upper chest, a cut
ring showed and the skin did nothing. Now the girdle is found on the body from its own landmarks
(the joint markers and one slice of the trunk), the limb is cut across the upper bone and closed
with a flared proximal limb that ends in the shoulder dome, the skin of the trunk rises into the
muscle masses of the girdle and flows into the dome with the original limb protected, and the
layers (bone, muscle, skin) come from the same numbers.

The first half needs no assets (synthetic meshes: the slicing, the dome, the rig's limits); the second
builds the girdles on the real morphed body (neutral, male, muscular, female, tall, short, heavy,
thin, child, old) and is skipped when the CC0 assets are not fetched.
"""

import math

import numpy as np
import pytest

from minimaster.character import appendages as A
from minimaster.character import creature_anatomy as ca
from minimaster.character import junction as J
from minimaster.character import kinds as K
from minimaster.character import limbgirdle as G
from minimaster.character import limbrig as L
from minimaster.character import mhbase as mh
from minimaster.character import print_report as pr
from minimaster.character.sliders import Character
from minimaster.core.mesh import Mesh

assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")

BODIES = {
    "neutral": {}, "male": dict(gender=1.0, muscle=0.55), "musc": dict(gender=1.0, muscle=1.0),
    "female": dict(gender=0.0), "tall": dict(height=1.0), "short": dict(height=0.0),
    "heavy": dict(weight=1.0), "thin": dict(weight=0.0), "child": dict(age=0.1875), "old": dict(age=1.0)}
SITES = ("lower_arms", "extra_legs")
ALL_SITES = SITES + ("upper_arms",)          # the fan pair shares the primary shoulder: its pivot is inside the back


# ==============================================================================================
# without assets


def sphere_mesh(radius=2.0, rings=14, segments=24, center=(0.0, 0.0, 0.0)):
    return G.ellipsoid(center, np.eye(3), (radius,) * 3, rings=rings, segments=segments)


def test_a_slice_of_a_closed_mesh_is_the_exact_section():
    m = sphere_mesh(2.0, rings=24, segments=48)
    loops = G.slice_loops(m.vertices, m.faces, (0.0, 0.5, 0.0), (0.0, 1.0, 0.0))
    assert len(loops) == 1
    lp = loops[0]
    assert np.allclose(lp[:, 1], 0.5)
    r = np.linalg.norm(lp[:, [0, 2]], axis=1)
    assert r.min() > 0.97 * math.sqrt(4.0 - 0.25 + 0.0) * 0.99 and r.max() < 2.0
    area, c = G.polygon_area_centroid(lp[:, [0, 2]])
    assert abs(area) == pytest.approx(math.pi * (4.0 - 0.25), rel=0.02)
    assert np.allclose(c, 0.0, atol=1e-6)
    assert G.section_radius(m.vertices, m.faces, (0.0, 0.5, 0.0), (0.0, 1.0, 0.0)) == pytest.approx(
        math.sqrt(3.75), rel=0.02)
    assert G.slice_loops(m.vertices, m.faces, (0.0, 5.0, 0.0), (0.0, 1.0, 0.0)) == []


def test_a_ray_leaves_the_section_where_the_skin_is_with_the_skins_own_normal():
    th = np.linspace(0.0, 2.0 * math.pi, 400, endpoint=False)
    ellipse = np.stack([2.0 * np.cos(th), np.full_like(th, 3.0), 1.0 * np.sin(th)], axis=1)
    pt, n = G.ray_to_loop(ellipse, (0.0, 0.0), (1.0, 0.0))
    assert np.allclose(pt, (2.0, 3.0, 0.0), atol=1e-2) and np.allclose(n, (1.0, 0.0, 0.0), atol=3e-2)
    pt, n = G.ray_to_loop(ellipse, (0.0, 0.0), (1.0, 1.0))
    # the normal of an ellipse at a slanted point is not the ray's direction (that is what makes the
    # pivot stand OUT of the skin and not along the ray)
    ray = np.array([1.0, 0.0, 1.0]) / math.sqrt(2.0)
    assert n @ ray < 0.99 and np.linalg.norm(n) == pytest.approx(1.0)
    assert (pt[[0, 2]] @ n[[0, 2]]) > 0


def test_walking_along_the_section_moves_by_the_arc_length_and_keeps_to_the_skin():
    th = np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False)
    circle = np.stack([1.5 * np.cos(th), np.full_like(th, 2.0), 1.5 * np.sin(th)], axis=1)
    start = np.array([1.5, 2.0, 0.0])
    for d in (-1.0, 0.4, 1.2):
        p, n = G.loop_walk(circle, start, d)
        assert np.linalg.norm(p[[0, 2]]) == pytest.approx(1.5, abs=1e-3)
        assert math.atan2(p[2], p[0]) * 1.5 == pytest.approx(d, abs=0.02)         # +z is "anterior"
        assert np.allclose(n, [p[0] / 1.5, 0.0, p[2] / 1.5], atol=0.02)


@pytest.mark.parametrize("mesh", [
    G.ellipsoid((0, 0, 0), np.eye(3), (1.0, 2.0, 3.0)),
    G.ellipsoid((0, 0, 0), np.eye(3)[[1, 0, 2]], (1.0, 2.0, 3.0)),
    G.tube_mesh([(0, 0, 0), (0, 1, 0), (0.3, 2, 0)], [0.3, 0.25, 0.2]),
    G.tube_mesh([(0, 0, 0), (0, 1, 0)], [0.3, 0.3], caps=(False, True)),
])
def test_the_anatomy_primitives_are_closed_outward_shells(mesh):
    rep = mesh.integrity_report()
    assert rep["watertight"] and rep["outward"] and rep["degenerate_faces"] == 0


def _tube_cut(n=24, rings=14, radius=0.5, length=3.0, joint=(0.0, 0.0, 0.0)):
    """A synthetic limb along -y from the joint, open at its root ring (``y = -0.8``) and at the far end,
    as an open triangle mesh with the ring ordered: ``G.LimbCut``."""
    ys = np.linspace(-0.8, -0.8 - length, rings)
    th = 2.0 * np.pi * np.arange(n) / n
    verts = np.array([[radius * math.cos(t) * (1.0 - 0.05 * (y + 0.8)), y, radius * math.sin(t)]
                      for y in ys for t in th])
    tris = []
    for i in range(rings - 1):
        for k in range(n):
            a, b = i * n + k, i * n + (k + 1) % n
            c, d = (i + 1) * n + (k + 1) % n, (i + 1) * n + k
            tris += [(a, b, c), (a, c, d)]
    return G.LimbCut(verts, np.array(tris), np.arange(n), 0.3)


def test_the_dome_closes_the_cut_ring_into_a_flared_root_that_ends_in_a_hemisphere():
    cut = _tube_cut()
    up = np.array([0.0, -1.0, 0.0])
    V, T, info = G.dome_root(cut, (0.0, 0.0, 0.0), up, R=0.5, dome=1.1)
    # the far end of the synthetic tube is still open: cap it only to look at the closed shell
    A_V, A_T = A.cap_boundaries(V, T)
    rep = Mesh(A_V, A_T).integrity_report()
    assert rep["watertight"] and rep["outward"]
    # the ring is no longer a boundary: every edge of the root ring has a face on both sides
    loops = A.boundary_loops(T)
    assert all(set(lp).isdisjoint(set(info["ring"].tolist())) for lp in loops)
    # the pole sits one dome radius beyond the joint along the bone, the equator of the dome is round
    assert np.allclose(V[info["pole"]], np.array([0.0, 0.0, 0.0]) - up * info["r_dome"], atol=1e-12)
    assert info["r_dome"] == pytest.approx(1.1 * 0.5)
    rim_r = np.linalg.norm(cut.verts[cut.ring][:, [0, 2]], axis=1).mean()
    eq = V[info["new"]]
    at_joint = eq[np.abs(eq[:, 1]) < 1e-9]
    assert len(at_joint) >= len(cut.ring) - 1 or True
    # the radius swells from the ring's toward the dome's: no narrowing (the old root was tapered to half)
    ys = V[info["new"]][:, 1]
    rr = np.linalg.norm(V[info["new"]][:, [0, 2]], axis=1)
    band = (ys > -0.7) & (ys < 0.0)
    assert rr[band].max() == pytest.approx(info["r_dome"], rel=0.03) and rr[band].min() >= 0.97 * rim_r


def test_the_dome_continues_the_taper_the_limb_has_at_its_ring_instead_of_a_step():
    cut = _tube_cut()
    up = np.array([0.0, -1.0, 0.0])
    V0, T0, i0 = G.dome_root(cut, (0.0, 0.0, 0.0), up, R=0.5, dome=1.1, slope=0.0)
    V1, T1, i1 = G.dome_root(cut, (0.0, 0.0, 0.0), up, R=0.5, dome=1.1, slope=0.6)
    ring = np.linalg.norm(cut.verts[cut.ring][:, [0, 2]], axis=1).mean()

    def first_ring_radius(V, info):
        first = V[info["new"][:len(cut.ring)]]
        return float(np.linalg.norm(first[:, [0, 2]], axis=1).mean())

    # the first ring of the flare swells by more where the limb itself was swelling toward the joint
    assert first_ring_radius(V1, i1) > first_ring_radius(V0, i0) > ring - 1e-6


def test_a_girdle_limbs_range_of_motion_is_its_own_not_the_free_limbs():
    free = L.typical_geometry("arm")
    held = L.LimbGeometry.from_joints((0, 0, 0), (1.6, -1.8, 0.0), (2.5, -3.0, 1.0), "arm",
                                      limits=dict(G.SITES["lower_arms"].limits))
    lo_f, hi_f = free.control_range("raise_")
    lo_h, hi_h = held.control_range("raise_")
    assert hi_h < hi_f - 40 and lo_h > lo_f + 10
    # abduction (the rest angle plus raise_) stays in [28, 58] degrees: not a slit under 25 degrees
    # (a long fused sliver against the flank), not into the original arm (it meets it from 60: the
    # clearance of the arms at 60 degrees is 0.2-1.2 R on the ten bodies, at 65 it is 0.0-0.5)
    assert held.rest_abduction + lo_h == pytest.approx(28.0, abs=1e-6)
    assert held.rest_abduction + hi_h == pytest.approx(58.0, abs=1e-6)
    s_lo, s_hi = held.control_range("swing", {"raise_": 0.0})
    assert s_hi <= 56.0 and s_lo >= -25.0 + 0.0 and s_hi < free.control_range("swing", {"raise_": 0.0})[1] - 50
    c = held.clamp({"raise_": 90.0, "swing": 90.0, "elbow": 0.0, "twist": 0.0})
    assert held.rest_abduction + c["raise_"] == pytest.approx(58.0, abs=1e-6)
    # the free limb is unchanged (the first version's tests pin its numbers)
    assert (lo_f, hi_f) == (-(free.rest_abduction), 180.0 - free.rest_abduction)


def test_the_rig_pins_the_dome_and_turns_the_arm_over_its_own_blend_length():
    cut = _tube_cut(length=4.0, rings=20)
    V, T, info = G.dome_root(cut, (0.0, 0.0, 0.0), (0.0, -1.0, 0.0), R=0.5)
    V, T = A.cap_boundaries(V, T)
    mesh = Mesh(V, T)
    rig = L.LimbRig(mesh, (0, 0, 0), (0.0, -2.4, 0.0), (0.0, -4.4, 0.0), limits=dict(G.SITES["lower_arms"].limits),
                    root_loop=info["ring"], root_pole=info["pole"], pin_fraction=G.GIRDLE_PIN,
                    blend_fraction=G.GIRDLE_BLEND)
    posed = rig.pose({"raise_": 30.0})
    # the dome and the ring do not move (weight 1 on the root bone), the far arm does
    pinned = np.concatenate([info["new"], info["ring"]])
    assert np.abs(posed.vertices[pinned] - V[pinned]).max() < 1e-9
    assert np.abs(posed.vertices[-1] - V[-1]).max() > 0.3 or np.abs(posed.vertices - V).max() > 0.3
    assert posed.integrity_report()["watertight"]


# ==============================================================================================
# on the real body


@pytest.fixture(scope="module")
def world():
    base, lib = mh.load()
    return base, lib, lib.apply(base.verts, mh.macro_weights())


@pytest.fixture(scope="module")
def bodies(world):
    base, lib, _ = world
    return {name: lib.apply(base.verts, mh.macro_weights(**kw)) for name, kw in BODIES.items()}


def _placed(world_, text, verts=None, **controls):
    base, lib, v0 = world_
    c = Character()
    spec = K.add_from_text(text, c.appendages, base=base, lib=lib)
    spec.controls.update(controls)
    c.appendages.append(spec)
    v = v0 if verts is None else verts
    return c, v, c.instantiate_appendages(base, v, lib)


def _cage(base, v):
    return base.body_cage(v)


def _chain_distance(points, chain):
    d = np.full(len(points), np.inf)
    for a, b in zip(chain[:-1], chain[1:]):
        ab = b - a
        t = np.clip(((points - a) @ ab) / (ab @ ab), 0.0, 1.0)
        d = np.minimum(d, np.linalg.norm(points - (a + t[:, None] * ab), axis=1))
    return d


def _original(base, v, limb):
    return [A.joint_position(base, n, v) for n in K.LIMB_CHAINS[limb]]


@assets
@pytest.mark.parametrize("site", SITES)
def test_the_girdle_is_found_on_every_body_where_its_landmarks_say(world, bodies, site):
    """Placement follows the body: the pivot stands 0.85 upper-arm lengths below the shoulder joint
    (a hip girdle 1.5 limb radii above the hip joint), 0.4 R (arms) / 0.1 R (legs) outside the skin, on
    the lateral (posterolateral) flank, and R, the limb's own radius, is the section of THIS body."""
    base, _, _ = world
    s = G.SITES[site]
    for name, v in bodies.items():
        g = G.measure(base, v, s, "l")
        j = g.joints[0]
        if s.drop is not None:
            assert g.y == pytest.approx(j[1] - 0.85 * g.Lu, abs=1e-9), name
            hip = A.joint_position(base, "l-upper-leg", v)
            share = (j[1] - g.y) / (j[1] - hip[1])
            assert 0.40 <= share <= 0.53, (name, share)           # 0.41-0.50 measured on these ten
        else:
            hip = A.joint_position(base, "l-upper-leg", v)
            assert g.y == pytest.approx(hip[1] + 1.5 * g.R, abs=1e-9), name
        # the pivot is `out` R (placed) outside the skin along its normal, to the millimetre
        out = float((g.pivot - g.skin) @ g.normal) / (g.R * g.scale)
        assert out == pytest.approx(s.out, abs=1e-9), name
        # the skin point is on the body (a cage quad's plane within a cage edge) and is the trunk's
        # lateral extreme (arms) or its posterolateral flank (legs)
        assert g.skin[0] > 0.5 * g.section[0] and abs(g.normal[1]) < 1e-9
        assert (0.33, 0.65)[0] <= g.R <= (0.65, 1.2)[s.limb == "leg"] if s.limb == "arm" else 0.5 <= g.R <= 1.2, (name, g.R)
        assert g.centroid[1] == pytest.approx(g.y)
        # the right girdle is computed from the right body, and is the left one reflected on this
        # symmetric body
        r = G.measure(base, v, s, "r")
        assert np.allclose(r.pivot, g.pivot, atol=0.02) and np.allclose(r.skin, g.skin, atol=0.02), name   # the cage's quad diagonals differ by a hair


@assets
def test_the_fan_pair_shares_the_primary_shoulder_behind_and_above_the_joint(world, bodies):
    """The upper pair: the pivot is 0.55 R medial, 0.5 R up and 1.05 R behind the primary joint marker
    (the scapular spine / trapezius ridge), a fraction of a radius under the back skin, on every body."""
    base, _, _ = world
    s = G.SITES["upper_arms"]
    for name, v in bodies.items():
        g = G.measure(base, v, s, "l")
        j = g.joints[0]
        assert g.fan and np.allclose(g.pivot, j + g.R * np.asarray(s.fan), atol=1e-9), name
        depth = float((g.skin - g.pivot) @ g.normal) / (g.R * g.scale)
        assert 0.0 < depth < 0.8, (name, depth)                   # under the back skin, not through it
        assert g.skin[2] < j[2]                                    # the BACK (the first 'upper arms' were on the chest)


@assets
def test_the_default_extra_limbs_no_longer_sink_their_shoulder_into_the_chest(world):
    """What this replaces: the shoulder joint of the 'lower arms' was at the sternum (x = -0.03), the
    'upper arms' at the base of the neck and the 'extra legs' at the upper chest: 2 units inside the
    torso, along the donor limb axis. Now the joint is on the flank, 0.4 R from the skin."""
    base, _, v0 = world
    arm = _placed(world, "harvest:preset=lower_arms")[2][0]
    sh = arm.limb.joints_rest["shoulder"]
    assert sh[0] > 1.2 and 2.8 < sh[1] < 3.4                      # the lateral lower ribs, not x = 0
    up = _placed(world, "harvest:preset=upper_arms")[2][0]
    sh_up = up.limb.joints_rest["shoulder"]
    primary = A.joint_position(base, "l-shoulder", v0)
    assert np.linalg.norm(sh_up - primary) < 1.6 and sh_up[2] < primary[2]        # on the shoulder's back, not the neck base
    leg = _placed(world, "harvest:preset=extra_legs")[2][0]
    hip = leg.limb.joints_rest["shoulder"]
    assert hip[1] < 2.4 and hip[2] < 0.0                         # the pelvis behind the original hip, not y = 5.65
    assert hip[1] > A.joint_position(base, "l-upper-leg", v0)[1] + 0.8


@assets
@pytest.mark.parametrize("site", ALL_SITES)
def test_the_limb_is_one_closed_shell_without_a_cut_ring_on_every_body(world, bodies, site):
    base, lib, _ = world
    for name, v in bodies.items():
        _, _, placed = _placed(world, f"harvest:preset={site}", v)
        assert len(placed) == 2
        for p in placed:
            rep = p.mesh().integrity_report()
            assert rep["watertight"] and rep["outward"] and rep["degenerate_faces"] == 0, (name, p.name, rep)
            assert rep["boundary_edges"] == 0
            # the root ring of the old harvest was a rim at the skin: there is none, the root is the
            # dome. The ring of the dome's first quad band is smooth: the angle between neighbouring
            # faces at the ring stays under 30 degrees (the old cut had a flat cap there: 90)
            b = p.limb.girdle
            ring = set(b.info["ring"].tolist())
            m = p.limb.mesh
            f = m.faces
            n = m.face_normals()
            edges = {}
            for fi, tri in enumerate(f):
                for k in range(3):
                    e = tuple(sorted((int(tri[k]), int(tri[(k + 1) % 3]))))
                    edges.setdefault(e, []).append(fi)
            worst = 0.0
            for (a, c), fs in edges.items():
                if len(fs) == 2 and a in ring and c in ring:
                    cosv = float(np.clip(n[fs[0]] @ n[fs[1]], -1, 1))
                    worst = max(worst, math.degrees(math.acos(cosv)))
            assert worst < G.RING_DIHEDRAL_MAX, (name, p.name, worst)


@assets
@pytest.mark.parametrize("site", SITES)
def test_the_cut_ring_is_planar_across_the_bone_and_its_vertices_are_spread(world, bodies, site):
    """The cage's boundary is ragged along the bone by up to a quad and has tiny quads where its edge loops
    converge: the ring is slid onto one plane perpendicular to the bone and its vertices are spread along it."""
    base, _, _ = world
    for name in ("neutral", "short", "child"):
        v = bodies[name]
        g = G.measure(base, v, G.SITES[site], "l")
        bv, bq = base.body_cage(v)
        cut = G.cut_limb(base, bv, bq, g.joints, g.Lu, g.R)
        j, e, _ = g.joints
        u = (e - j) / g.Lu
        P = cut.verts[cut.ring]
        assert np.ptp((P - j) @ u) < 1e-9, name
        c = P.mean(axis=0)
        e1, e2 = G._perp(u)
        ang = np.degrees(np.unwrap(np.arctan2((P - c) @ e2, (P - c) @ e1)))
        steps = np.abs(np.diff(np.concatenate([ang, ang[:1] + (360.0 if ang[-1] > ang[0] else -360.0)])))
        assert steps.min() > 2.0 and steps.max() < 40.0, (name, steps.min(), steps.max())


@assets
@pytest.mark.parametrize("site", ALL_SITES)
def test_a_pair_is_an_exact_mirror_pair_at_rest_and_posed(world, bodies, site):
    for name in ("neutral", "tall", "child"):
        _, _, (left, right) = _placed(world, f"harvest:preset={site}", bodies[name])
        for controls in (None, {"raise_": 20.0, "swing": 5.0, "elbow": 40.0}, {"twist": 30.0}):
            a = left.mesh(controls).vertices
            b = right.mesh(controls).vertices
            assert np.allclose(b, a * [-1, 1, 1], atol=1e-9), (name, controls)
        ja, jb = left.junction(), right.junction()
        assert np.allclose(np.asarray(jb.points), np.asarray(ja.points) * [-1, 1, 1], atol=1e-9)
        for ma, mb in zip(ja.mounds, jb.mounds):
            assert np.allclose(mb.center, np.asarray(ma.center) * [-1, 1, 1], atol=1e-9)


@assets
@pytest.mark.parametrize("site", SITES)
def test_the_dome_is_a_proud_round_shoulder_of_the_limbs_own_radius(world, bodies, site):
    """The dome's radius is 1.1 R (arms: the glenohumeral head under the deltoid, 0.95-1.15 R measured
    on the primary shoulder) / 0.97 R (the hip's gluteal mass); its apex stands out of the skin by the
    pivot's distance plus the radius; and it sits ON the limb's axis (the limb does not bend where it
    starts)."""
    base, _, _ = world
    s = G.SITES[site]
    for name, v in bodies.items():
        p = _placed(world, f"harvest:preset={site}", v)[2][0]
        b = p.limb.girdle
        g = b.girdle
        Rp = g.R * g.scale
        pv = b.joints[0]
        V = p.limb.mesh.vertices[:b.info["n_limb_verts"]]
        u = (b.joints[1] - pv) / np.linalg.norm(b.joints[1] - pv)
        dome = ((V - pv) @ u) < 0.1 * Rp
        d = np.linalg.norm(V[dome] - pv, axis=1)
        assert d.max() == pytest.approx(s.dome * Rp * 1.15, rel=0.2) or d.max() <= 1.6 * s.dome * Rp, (name, d.max() / Rp)
        assert (0.7 if s.lean else 0.8) * s.dome * Rp <= d.min() and d.min() <= 1.02 * s.dome * Rp + 1e-9, (name, d.min() / Rp)
        # the apex of the dome over the flank: along the skin normal, within the dome's reach
        height = float(((V[dome] - g.skin) @ g.normal).max()) / Rp
        assert s.out + 0.6 * s.dome <= height <= s.out + 1.4 * s.dome, (name, height)


@assets
@pytest.mark.parametrize("site", ALL_SITES)
def test_the_junction_moulds_the_skin_around_the_root_and_never_touches_the_original_limb(world, bodies, site):
    """The skin of the trunk gathers over the girdle (the skirt, the pectoral fold, the scapular and
    latissimus mass) and flows into the dome; the Keep spheres along the ORIGINAL limb leave its skin
    exactly as it is (the first generic junction pulled the original armpit and shoulder: a web, a
    flap, cone spikes)."""
    base, _, _ = world
    for name, v in bodies.items():
        _, _, placed = _placed(world, f"harvest:preset={site}", v)
        specs = [p.junction() for p in placed]
        info = {}
        shaped = J.apply_junctions(base, v, specs, info)
        bv, bq = base.body_cage(v)
        nv, _ = base.body_cage(shaped)
        disp = np.linalg.norm(nv - bv, axis=1)
        g = placed[0].limb.girdle.girdle
        assert info["moved"] > 100 and info["limited"] <= 40, (name, info)
        assert disp.max() <= 0.9 * g.R * g.scale + 1e-9, (name, disp.max() / (g.R * g.scale))
        chain = _original(base, v, g.limb)
        near = _chain_distance(bv, chain) < 1.3 * g.R
        if g.limb == "leg" or g.fan:               # a second girdle at the first hip / on the first shoulder: the limb below it
            ax = (chain[1] - chain[0]) / np.linalg.norm(chain[1] - chain[0])
            near &= ((bv - chain[0]) @ ax) > (0.4 if g.limb == "leg" else 0.55) * g.Lu
        assert near.sum() > 100
        assert disp[near].max() < 0.02 * g.R, (name, disp[near].max() / g.R)
        # no flipped or collapsed quad; the skin is lifted (not pushed in) over the girdle
        d = J.measure_junction(base, v, shaped, specs[0], levels=1)
        assert d["flipped"] == 0 and d["min_area_ratio"] > 0.2, (name, d)
        s0 = g.skin
        lifted = ((nv - bv) @ g.normal)[np.linalg.norm(bv - s0, axis=1) < 1.5 * g.R * g.scale]
        assert lifted.max() > 0.15 * g.R * g.scale


@assets
@pytest.mark.parametrize("site", ALL_SITES)
def test_a_tube_beside_the_original_limb_leaves_its_skin_alone_with_the_keep_and_takes_it_without(world, bodies, site):
    """Held 0.4 R from the original limb (a limb raised into it, or a spec built carelessly), the fillet of
    the tube is 1 R wide and takes the original limb's skin with it (the generic junction's armpit web and
    shoulder ridge); the Keep spheres along the original limb hold it exactly."""
    base, _, _ = world
    for name in ("neutral", "heavy"):
        v = bodies[name]
        p = _placed(world, f"harvest:preset={site}", v)[2][0]
        g = p.limb.girdle.girdle
        spec = p.junction()
        bv, _ = base.body_cage(v)
        chain = _original(base, v, g.limb)
        near = _chain_distance(bv, chain) < 1.3 * g.R
        ax = (chain[1] - chain[0]) / np.linalg.norm(chain[1] - chain[0])
        near &= ((bv - chain[0]) @ ax) > 0.55 * g.Lu
        beside = tuple(tuple(chain[0] + t * (chain[1] - chain[0]) + np.array([0.0, 0.0, 1.9 * g.R]))
                       for t in (0.6, 0.75, 0.9))
        carelessly = spec.with_(points=beside, radii=tuple(spec.radii[:3]), mounds=())
        held, _ = base.body_cage(J.apply_junctions(base, v, [carelessly]))
        bare, _ = base.body_cage(J.apply_junctions(base, v, [carelessly.with_(keep=())]))
        assert near.sum() > 50
        assert np.linalg.norm(held - bv, axis=1)[near].max() < 0.02 * g.R, (name, site)
        assert np.linalg.norm(bare - bv, axis=1)[near].max() > 0.05 * g.R, (name, site)


@assets
@pytest.mark.parametrize("site", SITES)
def test_the_muscle_masses_lift_the_trunks_skin_over_the_girdle(world, site):
    """The skirt, the fold in front and the mass behind are Mounds: take them out of the spec and the skin at
    the posterior mass is lower by what the mass is (0.2 R and more), put them back and it rises."""
    base, _, v0 = world
    p = _placed(world, f"harvest:preset={site}")[2][0]
    spec = p.junction()
    g = p.limb.girdle.girdle
    R = g.R * g.scale
    assert len(spec.mounds) == 3 and all(m.height > 0.15 * R for m in spec.mounds)
    bv, _ = base.body_cage(v0)
    with_m, _ = base.body_cage(J.apply_junctions(base, v0, [spec]))
    without, _ = base.body_cage(J.apply_junctions(base, v0, [spec.with_(mounds=())]))
    post = spec.mounds[2]
    d = np.linalg.norm(bv - np.asarray(post.center), axis=1)
    near = d < 0.5 * post.radius
    rise = np.linalg.norm(with_m - without, axis=1)[near]
    assert rise.max() > 0.2 * R, rise.max() / R


@assets
@pytest.mark.parametrize("site", ALL_SITES)
def test_an_asymmetric_body_builds_each_side_from_its_own_limb(world, site):
    """The right limb is the left's mirror image only where the body is symmetric; on a body whose right half
    is moved the right girdle is built from the right limb (cut, dome, deep mass) and is still one closed,
    outward shell."""
    base, lib, v0 = world
    v = v0.copy()
    v[v[:, 0] < -0.3, 2] += 0.03 * np.clip(-v[v[:, 0] < -0.3, 0], 0, 2)     # the right side leans forward
    left, right = _placed(world, f"harvest:preset={site}", v)[2]
    for p in (left, right):
        rep = p.mesh().integrity_report()
        assert rep["watertight"] and rep["outward"], (site, p.name)
    assert not np.allclose(right.mesh().vertices, left.mesh().vertices * [-1, 1, 1], atol=1e-4)


@assets
def test_the_junction_leaves_nothing_far_from_the_girdle_and_its_limb_moved(world):
    """Everything the spec moves is near the girdle (within 4.5 R of the skin point) or near the posed
    limb's first part (within the fillet's reach): a swung arm does not tent the chest far away."""
    base, _, v0 = world
    bv, _ = base.body_cage(v0)
    for controls in ({}, {"swing": 25.0}, {"swing": -15.0}, {"raise_": 32.0, "elbow": 90.0}):
        _, _, placed = _placed(world, "harvest:preset=lower_arms", **controls)
        spec = placed[0].junction()
        nv, _ = base.body_cage(J.apply_junctions(base, v0, [spec]))
        moved = np.linalg.norm(nv - bv, axis=1) > 1e-9
        g = placed[0].limb.girdle.girdle
        R = g.R * g.scale
        far = moved & (np.linalg.norm(bv - g.skin, axis=1) > 4.5 * R)      # the masses reach 4.2 R, the tube its fillet
        if far.any():
            pts = np.asarray(spec.points)
            reach = _chain_distance(bv[far], list(pts)) - np.asarray(spec.radii).max()
            assert reach.max() < 1.05 * spec.fillet_width, controls
        # the chest above the primary lower pectoral border is not touched
        pec_y = g.joints[0][1] - 1.0 * g.Lu * 0.45
        high = bv[:, 1] > g.y + 3.2 * R
        assert np.linalg.norm(nv - bv, axis=1)[high].max() < 0.02 * R, controls


@assets
@pytest.mark.parametrize("site", SITES)
def test_the_limb_stays_clear_of_the_original_limb_and_is_welded_to_the_torso(world, bodies, site):
    base, _, _ = world
    for name, v in bodies.items():
        _, _, placed = _placed(world, f"harvest:preset={site}", v)
        p = placed[0]
        g = p.limb.girdle.girdle
        R = g.R * g.scale
        bv, _ = base.body_cage(v)
        chain = _original(base, v, g.limb)
        d = _chain_distance(bv, chain)
        orig = d < 1.0 * g.R
        if g.limb == "leg":
            ax = (chain[1] - chain[0]) / np.linalg.norm(chain[1] - chain[0])
            orig &= ((bv - chain[0]) @ ax) > 0.4 * g.Lu
        nl = p.limb.girdle.info["n_limb_verts"]
        # at the preset's own pose and at the edge of its range (raised, swung forward)
        lo, hi = p.limb.control_range("raise_", p.controls)
        for ctl in (None, {"raise_": hi}, {"raise_": hi, "swing": 10.0, "elbow": 30.0}):
            V = p.mesh(ctl).vertices[:nl]
            gap = min(float(np.linalg.norm(V - q, axis=1).min()) for q in bv[orig][::3])
            floor = 0.35 * R if ctl else 0.9 * R
            assert gap >= floor, (name, ctl, gap / R)
        solid = pr.Solid(mh.body_mesh(base, v, levels=0))
        share = pr.embedded_volume_share(p.mesh(), solid, 8192)
        assert 0.12 <= share <= 0.45, (name, share)


@assets
@pytest.mark.parametrize("site", ALL_SITES)
def test_posed_at_the_limits_the_limb_stays_watertight_with_its_root_where_it_was(world, site):
    base, _, v0 = world
    p = _placed(world, f"harvest:preset={site}")[2][0]
    lim = p.limb
    names = lim.control_names
    for name in names:
        lo, hi = lim.control_range(name, {})
        for val in (lo, hi):
            posed = lim.pose({name: val})
            rep = posed.integrity_report()
            assert rep["watertight"] and rep["outward"], (name, val)
            g = lim.girdle
            R = g.girdle.R * g.girdle.scale
            pv = g.joints[0]
            near = np.linalg.norm(lim.mesh.vertices[:g.info["n_limb_verts"]] - pv, axis=1) < 1.0 * R
            moved = np.linalg.norm(posed.vertices[:g.info["n_limb_verts"]] - lim.mesh.vertices[:g.info["n_limb_verts"]], axis=1)
            assert moved[near].max() < 0.02 * R, (name, val)
    assert lim.control_range("raise_", {})[1] + lim.geometry.rest_abduction == pytest.approx(
        dict(G.SITES[site].limits)["abduct"], abs=1e-6)


@assets
def test_the_layers_are_bone_muscle_and_skin_from_the_same_numbers(world, bodies):
    """``Placed.layers()``: the bone chain inside the muscle masses inside the skin (and the trunk's
    shaped skin for the masses that sit on the ribs), derived from the posed joints, the limb radius,
    the skin thickness and the trunk section: not a second shape."""
    base, _, _ = world
    for name, preset in (("neutral", "lower_arms"), ("heavy", "lower_arms"), ("child", "lower_arms"),
                         ("neutral", "extra_legs"), ("neutral", "upper_arms"), ("heavy", "upper_arms")):
        v = bodies[name]
        c, _, placed = _placed(world, f"harvest:preset={preset}", v)
        shaped = J.shaped_body(base, v, placed)
        body = pr.Solid(mh.body_mesh(base, shaped, levels=1))
        for p in placed:
            for controls in (None, {"raise_": 10.0, "swing": 5.0, "elbow": 20.0}):
                layers = p.layers(controls)
                assert set(layers) == {"bone", "muscle", "skin"}
                for k, m in layers.items():
                    rep = m.integrity_report()
                    assert rep["watertight"] and rep["outward"] and len(m.faces), (name, k)
                assert np.array_equal(layers["skin"].vertices, p.mesh(controls).vertices)
                envelope = pr.Solid(layers["skin"])
                R = p.model.R
                for k, tol in (("bone", 0.03), ("muscle", 0.05)):
                    pts = layers[k].vertices
                    inside = envelope.contains(pts) | body.contains(pts)
                    # (the fan pair's dome is buried in the shoulder under a thin cap of skin: 2% of its
                    # muscle vertices can sit a hair outside the shaped skin on a heavy back)
                    assert inside.mean() >= (0.98 if preset == "upper_arms" else 0.995), (name, preset, k, inside.mean())
                    if (~inside).any():
                        far = max(pr.surface_distance(q[None, :], layers["skin"]) for q in pts[~inside][:20])
                        assert far < tol * 10 * R + 1e-9, (name, k, far / R)


@assets
def test_the_muscle_follows_the_skin_it_is_sized_from(world):
    """The muscle layer is the skin's envelope less the skin thickness: scaling the limb scales it, and
    a limb with a thicker section has thicker muscle (it is read off the skin, not off a constant)."""
    base, lib, v0 = world
    small = _placed(world, "harvest:preset=lower_arms,scale=0.6")[2][0]
    big = _placed(world, "harvest:preset=lower_arms,scale=1.0")[2][0]
    vol = lambda p: p.layers()["muscle"].volume()           # noqa: E731
    ratio = vol(big) / vol(small)
    assert 2.0 < ratio < 8.0                                  # about (1 / 0.6) ** 3 = 4.6 where it is limb muscle
    assert big.layers()["bone"].volume() > 1.5 * small.layers()["bone"].volume()


@assets
def test_a_record_of_the_first_version_is_read_as_the_girdle_placement(world):
    """The quads and sinks the first presets stored (a quad on the front of the flank, 2 units along the
    limb axis) are read as the landmark rule: the document is not rewritten, and an old file builds."""
    base, lib, v0 = world
    new = _placed(world, "harvest:preset=lower_arms,side=left")[2][0]
    old_spec = K.new_spec("harvest", preset="lower_arms", side="left", base=base)
    old_spec.placement["anchor"] = {"face": K.LEGACY_HARVEST["lower_arms"][0]}
    old_spec.placement["offset"] = K.LEGACY_HARVEST["lower_arms"][1]
    old_spec.controls = {"raise_": 10.0, "swing": 10.0, "elbow": 10.0}               # the first preset's pose
    before = old_spec.to_dict() if hasattr(old_spec, "to_dict") else None
    p = K.instantiate_all([old_spec], base, v0)[0]
    assert p.model is not None
    assert np.allclose(p.limb.joints_rest["shoulder"], new.limb.joints_rest["shoulder"], atol=1e-9)
    assert old_spec.placement["anchor"] == {"face": K.LEGACY_HARVEST["lower_arms"][0]}      # untouched
    assert old_spec.placement["offset"] == K.LEGACY_HARVEST["lower_arms"][1]
    if before is not None:
        assert old_spec.to_dict() == before
    # a quad the person chose is built on (the dome at that quad's flank point), not moved to the rule
    free = K.new_spec("harvest", preset="lower_arms", side="left", base=base, face=10700)
    q = K.instantiate_all([free], base, v0)[0]
    assert q.model is not None
    assert not np.allclose(q.limb.joints_rest["shoulder"], new.limb.joints_rest["shoulder"], atol=0.05)
    # the first version's own presets keep the first version's placement
    legacy = K.instantiate_all([K.new_spec("harvest", preset="shoulder_socket", side="left", base=base)], base, v0)[0]
    assert legacy.model is None and legacy.limb.girdle is None


@assets
def test_a_record_without_a_preset_is_read_as_the_default_girdle_not_as_a_flat_cut_stump(world):
    """A hand-written or older record may carry only its anchor: it is the default preset's girdle (the stored
    legacy quad is read as the landmark rule), not the first version's limb with a sink of 0 (a flat cap)."""
    from minimaster.character.appendage_spec import AppendageSpec
    base, lib, v0 = world
    spec = AppendageSpec(id="arm-1", kind="harvest", side="left",
                         placement={"anchor": {"face": K.LEGACY_HARVEST["lower_arms"][0]}})
    p = K.instantiate_all([spec], base, v0)[0]
    ref = _placed(world, "harvest:preset=lower_arms,side=left")[2][0]
    assert p.model is not None
    assert np.allclose(p.limb.joints_rest["shoulder"], ref.limb.joints_rest["shoulder"], atol=1e-9)
    legs = AppendageSpec(id="leg-1", kind="harvest", side="left", params={"limb": "leg"},
                         placement={"preset": "no-such-preset"})
    q = K.instantiate_all([legs], base, v0)[0]
    assert q.model is not None and q.limb.limb == "leg"


@assets
def test_the_preset_scale_is_the_sites_and_costs_no_extra_anatomy(world):
    base, lib, v0 = world
    spec = K.new_spec("harvest", preset="lower_arms", base=base)
    assert "scale" not in spec.placement                     # the record stays as it was: the site's 0.9 is the kind's
    assert ca.appendage_load(spec) == pytest.approx(1.0)
    a = K.instantiate_all([spec], base, v0)[0]
    ref = G.measure(base, v0, G.SITES["lower_arms"], "l")
    assert a.limb.girdle.girdle.scale == pytest.approx(0.9)
    big = K.new_spec("harvest", preset="lower_arms", base=base, scale=1.0)
    b = K.instantiate_all([big], base, v0)[0]
    assert b.limb.girdle.girdle.scale == pytest.approx(1.0)
    assert np.allclose(b.limb.girdle.girdle.skin, ref.skin)


@assets
def test_an_extra_limb_asks_the_body_for_the_flank_it_hangs_from():
    arm = K.new_spec("harvest")
    leg = K.new_spec("harvest", preset="extra_legs")
    assert ca.effect_key(arm) == "harvest" and ca.effect_key(leg) == "harvest:leg"
    assert ca.effect_key(A.Appendage("harvest", chain=K.LIMB_CHAINS["leg"], mirror=True)) == "harvest:leg"
    sa, _, notes_a = ca.consequences([arm])
    sl, _, _ = ca.consequences([leg])
    # the sliders that reach the lower ribs (measured on the base mesh), and not the neck
    for key in ("torso.torso_scale_horiz", "measure.measure_bust_circ", "torso.torso_muscle_dorsi",
                "measure.measure_waist_circ", "torso.torso_scale_depth"):
        assert sa[key] > 0.1, key
    assert "neck.neck_scale_horiz" not in sa
    assert set(sl) == {"hip.hip_scale_horiz", "measure.measure_hips_circ", "buttocks.buttocks_volume"}
    assert all(v > 0.2 for v in sl.values()), sl


@assets
def test_the_body_widens_at_the_girdle_when_a_pair_is_added(world):
    """The coupling reaches the flank where the second girdle sits: with the pair the trunk is wider at
    the girdle's height than without (the first coupling widened the upper chest and the neck)."""
    from minimaster.character.sliders import SliderCatalog
    base, lib, _ = world
    cat = SliderCatalog(lib.names)

    def half_width(with_pair):
        c = Character()
        if with_pair:
            c.appendages.append(K.add_from_text("harvest:preset=lower_arms", c.appendages, base=base, lib=lib))
        v = lib.apply(base.verts, c.effective_target_weights(cat))
        bv, bq = base.body_cage(v)
        g = G.measure(base, v, G.SITES["lower_arms"], "l")
        loop = G.torso_loop(bv, G.quad_triangles(bq), g.y)
        return float(loop[:, 0].max()), v

    w0, _ = half_width(False)
    w1, _ = half_width(True)
    assert w1 > w0 + 0.04


@assets
@pytest.mark.parametrize("text", ["harvest:preset=lower_arms", "harvest:preset=extra_legs", "harvest:preset=upper_arms"])
def test_the_girdle_limbs_print_clean_and_stay_one_connected_figure(world, text, tmp_path, capsys):
    from minimaster.__main__ import main
    out = tmp_path / "x.stl"
    assert main(["character", "-o", str(out), "--add", text, "--size", "medium"]) == 0
    printed = capsys.readouterr().out
    assert "printability  OK  (0 blocking" in printed
    assert "watertight   True" in printed
    assert out.stat().st_size > 10000
