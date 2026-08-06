"""Wings built the way wings are actually built.

A wing is not a slab. In both bats and birds it is a **modified forelimb**:
humerus, then radius/ulna, then a wrist, and then an enormously elongated hand
that carries the flight surface. Everything a viewer reads as "wing" -- the
elbow crook, the finger struts fanning through a membrane, the scalloped
trailing edge, the layered feathers -- comes from that skeleton. So this module
builds the skeleton first and hangs the surface on it.

Two styles share one arm:

* **membrane** (bat / dragon). The patagium is stretched over digits II-V. Real
  bats divide it into named regions: the *propatagium* from shoulder to wrist
  along the leading edge, the *dactylopatagium* between the fingers, and the
  *plagiopatagium* from the body flank out to digit V. Digit I (the thumb) stays
  free and clawed at the wrist -- the single most recognisable bat/dragon
  detail. The membrane carries ~12% camber, the value measured on real bat
  armwings (11.8-13.1% of chord).
* **feathered** (bird / angel). Primaries grow from the hand, secondaries from
  the forearm, and two rows of coverts shingle over their bases, each feather
  overlapping the next like real flight feathers. An alula sits on the thumb.

Output contract: every piece is its own closed solid, merged into one mesh.
Overlapping closed shells are exactly what the rest of MiniMaster expects --
the slicer unions them -- so bones can sit *inside* the membrane and read as
ridges without any boolean work.

Authored frame (the wing is built at the origin, then :func:`appendages.place`
puts it on the body)::

    +X  spanwise, shoulder -> wingtip
    +Y  down (the wing's underside); camber bulges toward -Y
    +Z  chordwise, leading edge -> trailing edge

That is why :class:`~minimaster.character.appendages.Appendage` takes an
``up_hint``: it aims the anchor's tangent, which is where local +X ends up. A
lateral hint puts the span across the character's back, the chord trailing
behind, and the thickness vertical -- which is what a wing does.
"""

from __future__ import annotations

import numpy as np

from ..core.mesh import Mesh

# --------------------------------------------------------------------------
# proportions
#
# Bat wing bones are quoted relative to FOREARM length, the standard measure in
# chiropteran morphometrics. Digits II-V are the ones that support the
# membrane; their metacarpals and phalanges are the elongated elements, and
# their proportions have been conserved for ~50 million years.

FOREARM = 1.0
HUMERUS = 0.72

# digit -> (length from the wrist, fan angle off the forearm axis, degrees)
#
# Digit III effectively CONTINUES the forearm -- that is what gives a wing its
# span. Fan it wide and the hand becomes a backward-pointing paddle instead:
# the whole chain's length goes into chord and the planform comes out square.
DIGITS = {
    "II": (1.15, -2.0),     # leading edge of the hand, just forward of III
    "III": (2.32, 8.0),     # longest -- it makes the wingtip
    "IV": (2.05, 36.0),
    "V": (1.52, 72.0),      # sweeps back to meet the body
}
THUMB = (0.24, -46.0)       # free and clawed, forward of the leading edge

DEFAULT_SPAN = 13.0         # wing units; one MakeHuman unit is a decimetre
CAMBER = 0.12               # fraction of chord; real bat armwings run 0.118-0.131
CAMBER_PEAK = 0.72          # exponent placing max camber near 30% chord


def _dir(deg: float) -> np.ndarray:
    """Unit vector in the wing plane, measured from +X toward +Z."""
    a = np.radians(deg)
    return np.array([np.cos(a), 0.0, np.sin(a)])


# --------------------------------------------------------------------------
# mesh primitives (self-contained so this module never imports appendages)


def _orient(mesh: Mesh) -> Mesh:
    """Flip the winding if the solid came out inside-out."""
    if mesh.volume() < 0:
        return Mesh(mesh.vertices, mesh.faces[:, [0, 2, 1]])
    return mesh


def _quads_to_tris(quads) -> np.ndarray:
    q = np.asarray(quads, dtype=np.int64)
    return np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])


def _tube(points, radii, sides: int = 8) -> Mesh:
    """A closed tapered tube through ``points``, capped with fans at both ends."""
    pts = np.asarray(points, dtype=np.float64)
    rad = np.asarray(radii, dtype=np.float64)
    n = len(pts)
    tangents = np.zeros_like(pts)
    tangents[1:-1] = pts[2:] - pts[:-2]
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(norms, 1e-12)

    # one reference frame carried along the tube; bones are short, so a single
    # seed direction is enough to stay twist-free
    ref = np.array([0.0, 1.0, 0.0])
    if abs(float(tangents[0] @ ref)) > 0.95:
        ref = np.array([0.0, 0.0, 1.0])
    rings = []
    th = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    for i in range(n):
        t = tangents[i]
        u = ref - float(ref @ t) * t
        lu = np.linalg.norm(u)
        u = u / lu if lu > 1e-9 else np.array([0.0, 0.0, 1.0])
        w = np.cross(t, u)
        ref = u                       # carry the frame forward
        rings.append(pts[i] + rad[i] * (np.cos(th)[:, None] * u[None, :]
                                        + np.sin(th)[:, None] * w[None, :]))
    verts = np.vstack(rings)
    quads = []
    for i in range(n - 1):
        b0, b1 = i * sides, (i + 1) * sides
        for k in range(sides):
            kn = (k + 1) % sides
            quads.append([b0 + k, b0 + kn, b1 + kn, b1 + k])
    tris = list(_quads_to_tris(quads))

    # fan caps: one centre vertex per end
    c0 = len(verts)
    verts = np.vstack([verts, pts[0][None, :], pts[-1][None, :]])
    c1 = c0 + 1
    last = (n - 1) * sides
    for k in range(sides):
        kn = (k + 1) % sides
        tris.append([c0, kn, k])
        tris.append([c1, last + k, last + kn])
    return _orient(Mesh(verts, np.asarray(tris, dtype=np.int64)))


def _grid_normals(grid: np.ndarray) -> np.ndarray:
    """Per-vertex normals of an (n+1, m+1, 3) surface grid."""
    du = np.gradient(grid, axis=0)
    dv = np.gradient(grid, axis=1)
    nrm = np.cross(du, dv)
    ln = np.linalg.norm(nrm, axis=2, keepdims=True)
    return nrm / np.maximum(ln, 1e-12)


def _slab_from_grid(grid: np.ndarray, thickness: np.ndarray) -> Mesh:
    """Thicken a surface grid into a closed solid.

    ``grid`` is (n+1, m+1, 3) mid-surface points, ``thickness`` an (n+1, m+1)
    field. Emits a top sheet, a bottom sheet, and a rim stitching their shared
    perimeter, wound so every edge is used exactly twice.
    """
    n1, m1, _ = grid.shape
    n, m = n1 - 1, m1 - 1
    nrm = _grid_normals(grid)
    half = (thickness / 2.0)[:, :, None]
    top = (grid + nrm * half).reshape(-1, 3)
    bot = (grid - nrm * half).reshape(-1, 3)
    verts = np.vstack([top, bot])
    nv = n1 * m1

    def T(i, j):
        return i * m1 + j

    def B(i, j):
        return nv + i * m1 + j

    quads = []
    for i in range(n):
        for j in range(m):
            quads.append([T(i, j), T(i, j + 1), T(i + 1, j + 1), T(i + 1, j)])
            quads.append([B(i, j), B(i + 1, j), B(i + 1, j + 1), B(i, j + 1)])

    # Perimeter in the same rotational sense the top sheet is wound in, so each
    # rim quad can re-use its boundary edge backwards.
    per = ([(0, j) for j in range(m1)]
           + [(i, m) for i in range(1, n1)]
           + [(n, j) for j in range(m - 1, -1, -1)]
           + [(i, 0) for i in range(n - 1, 0, -1)])
    for k in range(len(per)):
        a = per[k]
        b = per[(k + 1) % len(per)]
        quads.append([T(*b), T(*a), B(*a), B(*b)])

    return _orient(Mesh(verts, _quads_to_tris(quads)))


def _resample(polyline, count: int) -> np.ndarray:
    """Even arc-length resampling of a polyline to ``count`` points."""
    p = np.asarray(polyline, dtype=np.float64)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        return np.repeat(p[:1], count, axis=0)
    want = np.linspace(0.0, total, count)
    out = np.empty((count, 3))
    for k in range(3):
        out[:, k] = np.interp(want, s, p[:, k])
    return out


def _lift(flat_xz, flat_y, query_xz, sigma):
    """Gaussian-weighted height lookup, to sit bones on the membrane surface.

    The membrane is a parametric sweep, so a bone drawn in the flat plan has no
    closed-form (u, v). Weighting the surrounding grid heights is smooth, needs
    no solver, and is exact wherever a bone lies on a grid line.
    """
    d2 = ((query_xz[:, None, :] - flat_xz[None, :, :]) ** 2).sum(axis=2)
    w = np.exp(-d2 / (2.0 * sigma * sigma))
    tot = w.sum(axis=1)
    out = (w @ flat_y) / np.maximum(tot, 1e-30)
    # far outside the membrane the weights underflow; fall back to the nearest
    lost = tot < 1e-12
    if lost.any():
        out[lost] = flat_y[np.argmin(d2[lost], axis=1)]
    return out


# --------------------------------------------------------------------------
# the shared arm


class WingSkeleton:
    """Joint positions of a wing's forelimb, in the flat wing plane.

    ``fold`` closes the wing the way a real one folds: the forearm swings back
    against the humerus about the elbow, and the hand swings back against the
    forearm about the wrist. At ``fold=1`` the wing is stowed along the back.
    """

    def __init__(self, span=13.0, sweep=8.0, fold=0.0, elbow=4.0,
                 digits=DIGITS):
        self.fold = float(np.clip(fold, 0.0, 1.0))

        def joints(fold):
            a_hum = elbow
            a_fore = a_hum + sweep + fold * 150.0
            shoulder = np.zeros(3)
            elb = shoulder + HUMERUS * _dir(a_hum)
            wrist = elb + FOREARM * _dir(a_fore)
            # the hand folds back on the forearm, so digit fan angles are
            # measured off the forearm axis and swept further as it closes
            a_hand = a_fore + fold * 155.0
            tips = {n: wrist + L * _dir(a_hand + fan) for n, (L, fan) in digits.items()}
            L, fan = THUMB
            return shoulder, elb, wrist, wrist + L * _dir(a_hand + fan), tips

        # ``span`` describes the wing SPREAD. Measuring reach after folding
        # would rescale a folded wing back up to the requested span, so folding
        # would change the pose but never make the wing any smaller.
        spread = joints(0.0)
        pts = np.array([spread[0], spread[1], spread[2], spread[3]]
                       + list(spread[4].values()))
        reach = float(pts[:, 0].max() - pts[:, 0].min())
        self.scale = span / max(reach, 1e-9)

        s, e, w, th, tips = joints(self.fold)
        self.shoulder, self.elbow = s * self.scale, e * self.scale
        self.wrist, self.thumb = w * self.scale, th * self.scale
        self.digits = {k: v * self.scale for k, v in tips.items()}
        self.span = span

    @property
    def tip(self) -> np.ndarray:
        return self.digits["III"]


# --------------------------------------------------------------------------
# membrane (bat / dragon) wing


def membrane_wing(span=13.0, thickness=0.30, sweep=8.0, droop=0.14,
                  fold=0.0, camber=CAMBER, scallop=0.16, root_chord=1.7,
                  bones=True, samples=(44, 22)) -> Mesh:
    """A patagium stretched over a forelimb skeleton.

    ``thickness`` is in wing units. A 32 mm figure makes one unit ~1.9 mm, so
    the 0.30 default is a ~0.57 mm membrane -- thin, but above the ~0.3 mm a
    resin printer needs. Scaling the whole appendage down scales this too.
    """
    sk = WingSkeleton(span=span, sweep=sweep, fold=fold)
    D = sk.digits
    root_back = sk.shoulder + np.array([0.0, 0.0, root_chord * sk.scale])

    # Leading edge: the propatagium runs shoulder -> wrist, then the hand
    # carries on out along digit II to the tip at digit III.
    lead = [sk.shoulder, sk.elbow, sk.wrist, D["II"], D["III"]]

    # Trailing edge: from the body, out through the digit tips. Between tips
    # the free edge sags inward -- that is the scalloped silhouette.
    def scallop_arc(a, b, k=7):
        """A smooth concave arc from ``a`` to ``b``, bowed toward the interior.

        Bowing only the midpoint leaves a V-notch: the tips stay sharp (right,
        they are finger ends) but the sag between them has to be an arc, which
        is what a stretched membrane actually forms. Depth is proportional to
        the gap so every scallop stays in scale.
        """
        d = b - a
        perp = np.array([d[2], 0.0, -d[0]])
        ln = float(np.linalg.norm(perp))
        if ln < 1e-9:
            return []
        perp = perp / ln
        if float(perp @ (sk.wrist - (a + b) / 2.0)) < 0:
            perp = -perp
        depth = scallop * float(np.linalg.norm(d))
        t = np.linspace(0.0, 1.0, k + 2)[1:-1]
        return [a + d * s + perp * depth * np.sin(np.pi * s) for s in t]

    trail = ([root_back] + scallop_arc(root_back, D["V"]) + [D["V"]]
             + scallop_arc(D["V"], D["IV"]) + [D["IV"]]
             + scallop_arc(D["IV"], D["III"]) + [D["III"]])

    n, m = samples
    LE = _resample(lead, n + 1)
    TE = _resample(trail, n + 1)
    # keep the tip blunt so the grid never collapses into degenerate quads
    TE[-1] = LE[-1] + (TE[-1] - LE[-1]) * 0.06 + np.array([0.0, 0.0, 0.045 * span])

    v = np.linspace(0.0, 1.0, m + 1)
    grid = LE[:, None, :] * (1.0 - v)[None, :, None] + TE[:, None, :] * v[None, :, None]

    chord = np.linalg.norm(TE - LE, axis=1)
    u = np.linspace(0.0, 1.0, n + 1)
    # camber lifts the mid-surface toward -Y (up), peaking near quarter chord
    grid[:, :, 1] -= (camber * chord)[:, None] * np.sin(np.pi * v ** CAMBER_PEAK)[None, :]
    # and the whole wing arcs downward toward the tip
    grid[:, :, 1] += (droop * span * u ** 2)[:, None]

    # thicker where the arm and fingers run, thinning to the free edge
    thick = thickness * (1.0 - 0.45 * u)[:, None] * (1.0 - 0.4 * v)[None, :]
    thick = np.maximum(thick, thickness * 0.25)

    parts = [_slab_from_grid(grid, thick)]
    if bones:
        parts += _membrane_bones(sk, grid, span)
    return Mesh.merge(parts)


def _membrane_bones(sk: WingSkeleton, grid: np.ndarray, span: float):
    """Arm and finger bones, lifted onto the membrane so they read as ridges."""
    flat_xz = grid[:, :, [0, 2]].reshape(-1, 2)
    flat_y = grid[:, :, 1].reshape(-1)
    sigma = 0.045 * span

    def lay(a, b, r0, r1, steps=7, sides=7):
        t = np.linspace(0.0, 1.0, steps)[:, None]
        pts = a[None, :] * (1 - t) + b[None, :] * t
        pts[:, 1] = _lift(flat_xz, flat_y, pts[:, [0, 2]], sigma)
        radii = r0 + (r1 - r0) * t[:, 0]
        return _tube(pts, radii, sides)

    s = span / 13.0                    # radii quoted for a 13-unit wing
    out = [lay(sk.shoulder, sk.elbow, 0.30 * s, 0.24 * s, sides=9),
           lay(sk.elbow, sk.wrist, 0.24 * s, 0.17 * s, sides=9)]
    for name in ("II", "III", "IV", "V"):
        out.append(lay(sk.wrist, sk.digits[name], 0.15 * s, 0.05 * s))

    # digit I: free, clawed, and the detail that says "bat" at a glance
    claw_base = sk.wrist + (sk.thumb - sk.wrist) * 0.55
    out.append(lay(sk.wrist, claw_base, 0.13 * s, 0.09 * s, steps=4, sides=7))
    tip = sk.thumb.copy()
    tip[1] = _lift(flat_xz, flat_y, tip[None, [0, 2]], sigma)[0] - 0.12 * s
    mid = (claw_base + tip) / 2.0
    mid[1] = _lift(flat_xz, flat_y, mid[None, [0, 2]], sigma)[0] - 0.05 * s
    base = claw_base.copy()
    base[1] = _lift(flat_xz, flat_y, base[None, [0, 2]], sigma)[0]
    out.append(_tube(np.array([base, mid, tip]),
                     np.array([0.09, 0.055, 0.012]) * s, 7))
    return out


# --------------------------------------------------------------------------
# feathered (bird / angel) wing


def _feather(length, width, thickness, curl=0.35, sweep=0.30, lead_frac=0.34,
             samples=(9, 5)) -> Mesh:
    """One flight feather: an asymmetric, curved blade.

    Real flight feathers have a narrow leading vane and a wide trailing vane --
    that asymmetry is what makes them aerodynamic rather than decorative, and
    it is visible in silhouette.
    """
    n, m = samples
    t = np.linspace(0.0, 1.0, n + 1)
    # The vane flares off the quill, then runs nearly parallel-sided before
    # rounding off. A feather that tapers to a point from its widest place
    # leaves V-shaped gaps between neighbours instead of shingling.
    w = width * np.minimum(1.0, t / 0.16) * (1.0 - t ** 6) ** 0.35
    w = np.maximum(w, width * 0.06)
    s = np.linspace(-lead_frac, 1.0 - lead_frac, m + 1)

    grid = np.zeros((n + 1, m + 1, 3))
    grid[:, :, 0] = (length * t)[:, None]
    grid[:, :, 2] = w[:, None] * s[None, :] + (sweep * length * t ** 2)[:, None]
    # the shaft droops along its length, and the vanes cup downward
    grid[:, :, 1] = ((curl * length * t ** 2)[:, None]
                     + 0.18 * w[:, None] * (s ** 2)[None, :])

    # thickest at the rachis, thin at the vane edges
    prof = 1.0 - 0.62 * np.abs(s + lead_frac - 0.5) / 0.5
    thick = thickness * np.clip(prof, 0.35, 1.0)[None, :] * (1.0 - 0.45 * t)[:, None]
    return _slab_from_grid(grid, np.maximum(thick, thickness * 0.3))


def _place_feather(mesh: Mesh, root, yaw_deg, roll_deg=0.0) -> Mesh:
    """Rotate a feather about its root: yaw in the wing plane, then roll."""
    a = np.radians(yaw_deg)
    ry = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
    b = np.radians(roll_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    m = np.eye(4)
    m[:3, :3] = ry @ rx
    m[:3, 3] = root
    return mesh.transform(m)


def feathered_wing(span=13.0, thickness=0.20, sweep=8.0, droop=0.14, fold=0.0,
                   primaries=10, secondaries=12, coverts=True,
                   bones=True) -> Mesh:
    """A bird wing: primaries off the hand, secondaries off the forearm.

    Feathers overlap the way real ones do -- each slides under its inboard
    neighbour -- so the wing reads as shingled plates rather than a fan.
    """
    sk = WingSkeleton(span=span, sweep=sweep, fold=fold)
    s = span / 13.0
    parts = []

    def height(x):
        """The wing arcs downward toward the tip."""
        return droop * span * (x / max(span, 1e-9)) ** 2

    def row(a, b, count, length0, length1, yaw0, yaw1, width, thick,
            roll0=0.0, roll1=0.0, lift=0.0):
        for i in range(count):
            f = i / max(count - 1, 1)
            root = a + (b - a) * f
            root = root.copy()
            root[1] = height(root[0]) + lift
            blade = _feather(length0 + (length1 - length0) * f, width, thick,
                             curl=0.22, sweep=0.26)
            parts.append(_place_feather(blade, root,
                                        yaw0 + (yaw1 - yaw0) * f,
                                        roll0 + (roll1 - roll0) * f))

    # Primaries: rooted along the hand, longest and most swept at the tip.
    # Width is set well above the root spacing so consecutive feathers overlap
    # -- real ones shingle, each sliding under its inboard neighbour.
    hand_a, hand_b = sk.wrist, sk.digits["III"]
    row(hand_a, hand_b, primaries, 4.4 * s, 6.2 * s, 58.0, 26.0,
        1.9 * s, thickness, roll0=-4.0, roll1=-14.0)
    # secondaries: rooted along the forearm, pointing almost straight back
    row(sk.elbow, sk.wrist, secondaries, 3.6 * s, 3.9 * s, 98.0, 74.0,
        2.1 * s, thickness, roll0=-2.0, roll1=-6.0)

    if coverts:
        # two shingled rows over the bases, plus the alula on the thumb
        row(sk.elbow, hand_b, max(primaries, secondaries), 2.4 * s, 2.0 * s,
            92.0, 40.0, 1.6 * s, thickness * 0.9, lift=-0.12 * s)
        row(sk.shoulder, sk.wrist, max(primaries, secondaries) - 2,
            1.7 * s, 1.4 * s, 96.0, 66.0, 1.4 * s, thickness * 0.8,
            lift=-0.24 * s)
        row(sk.wrist, sk.thumb, 3, 1.6 * s, 1.1 * s, 26.0, 4.0,
            0.7 * s, thickness * 0.8, lift=-0.34 * s)

    if bones:
        def lay(a, b, r0, r1, sides=9):
            pts = np.array([a, (a + b) / 2.0, b])
            pts[:, 1] = [height(p[0]) for p in pts]
            return _tube(pts, np.array([r0, (r0 + r1) / 2, r1]), sides)
        parts.append(lay(sk.shoulder, sk.elbow, 0.34 * s, 0.27 * s))
        parts.append(lay(sk.elbow, sk.wrist, 0.27 * s, 0.19 * s))
        # The hand bone must stop short of where the primaries root, and follow
        # the same line they do. Run it out to digit II while the primaries
        # follow digit III and it emerges past the plumage as a bare rod.
        parts.append(lay(sk.wrist, sk.wrist + (hand_b - sk.wrist) * 0.72,
                         0.17 * s, 0.07 * s))

    return Mesh.merge(parts)


# --------------------------------------------------------------------------

STYLES = {"membrane": membrane_wing, "feathered": feathered_wing}


def make_wing(style: str = "membrane", **kw) -> Mesh:
    """Build a wing. ``style`` is ``"membrane"`` (bat/dragon) or ``"feathered"``."""
    builder = STYLES.get(style)
    if builder is None:
        raise KeyError(f"unknown wing style {style!r}; have {sorted(STYLES)}")
    return builder(**kw)
