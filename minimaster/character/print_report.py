"""What the watertight gate cannot see: is each appendage a printable PART of the figure?

``printprep.check_shells`` is a topology test. A wing floating behind the figure, a tail
that merely touches the skin, a sheet 0.1 mm thick and three feathers that hang in the air
near the wrist all pass it. This module measures the things that decide whether a print
holds together, per appendage, in millimetres at the chosen print size, and sorts what it
finds into **blocking** (the print will fail or fall apart: do not export) and **warning**
(it will print, check it). Design: ``docs/appendages_system_design.md`` D8.

Everything is measured in scene units on the Z-up shells :func:`printprep.appendage_print_shells`
returns (a wing split into one shell per rig piece, so the offender has a name) and scaled to
mm with ``mm_per_unit`` (:func:`printprep.figure_scale`); nothing here depends on the print
size except the thresholds, so a size change re-uses every cached measurement.

Blocking
    * a shell that is empty, open, non-manifold, degenerate or inside-out (ALL of them);
    * a part that is not overlap-connected to the body and is big enough to matter (a
      detached tail or arm; three tiny feathers only warn);
    * a root that is not buried in the body: the largest share of the VOLUME of the named
      root piece(s) (wing arm / hand / membrane, or the whole tail or horn) that lies inside
      the body is below ``min_root_embed`` (a shallow weld, below ``warn_root_embed``, warns).
      A tail or horn that only touches the skin is welded along a hairline and prints
      detached;
    * more than ``block_area_thin`` of an appendage's area thinner than the finest feature
      the printer resolves (0.2 mm) at this size; the same share of a tail or horn's LENGTH.

Warning
    * the same appendage when it is merely thin (< 0.3 mm), or thin only in the posed
      (folded) mesh, where pleats lay sheet on sheet; a tail or horn tip below the finest
      feature; two appendages that intersect; anything below the feet plane; a base that
      the appendages overhang.

Assumption, stated in every report: the slicer unions overlapping shells.

The expensive measurements (thickness, connectivity) are cached per appendage in the dict
the caller passes as ``cache``; a size change or a different threshold costs nothing.

What an "appendage" is here (so any registry can feed it): an object with ``build(base,
verts) -> [(name, Mesh)]`` in the MakeHuman frame, and for rigged wings ``build_rigs``; its
``kind`` / ``params`` / ``scale`` (a tail or horn's profile, the thickness floor); optionally
``cache_key()`` (default: its repr) and ``at_rest()`` (default: ``params`` without the pose
keys of a dataclass with ``POSE_KEYS``). A ready-made ``[(name, Mesh)]`` list works too. A
wing must expose ``build_rigs`` so the report sees its pieces: the weld is measured on the
named root pieces and connectivity among all of them, and a wing merged into one shell has
only its whole-wing embedding (~5%) and would be flagged for it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
from dataclasses import dataclass, field

import numpy as np

from ..core.mesh import Mesh
from . import printprep as pp

ASSUMPTION = "assumes the slicer unions overlapping shells (every shell is checked alone)"

# Pieces of a rigged wing that carry the weld to the body. A wing is welded through whichever
# of them is buried deepest: a bat through its arm / propatagium / membrane root, an angel
# through its arm (its hand starts outboard of the torso and is 0% embedded).
# (in the order they are measured: the arm first, because a firm arm settles the question).
ROOT_PIECES = ("arm", "hand", "propatagium", "membrane")
# The root collar (collar.py) is a thin transition sheet that the slicer unions with the body and the
# appendage: it is checked as a shell (closed, outward) and counted in the connection and the volume,
# but it is NOT a root (it is embedded by construction, so a tail that only touches the skin would
# otherwise pass the weld rule through it) and its own thickness is not a printable feature (the
# union is as thick as the body and the tail).
COLLAR_PIECE = "collar"


@dataclass(frozen=True)
class PrintRules:
    """Every threshold of the report, in one place. Lengths in mm, shares as fractions."""

    min_feature_mm: float = pp.RESIN_MIN_FEATURE_MM        # 0.2: below this nothing resolves
    thin_feature_mm: float = 0.30                          # below this a part is fragile
    block_area_thin: float = 0.30      # of an appendage's area under min_feature_mm -> block
    warn_area_thin: float = 0.20       # of its area under thin_feature_mm -> warn
    warn_posed_area_thin: float = 0.10  # of a deformed (folded) sheet under min_feature_mm
    # Share of the root piece's VOLUME inside the body (Monte Carlo, below). Measured on the
    # reference body: a tail at offset 0 is 2-4% and a horn 0.1-0.2%; sunk by 0.3 / 0.6 units
    # the tail is 14% / 27% and the horn 40% / 70%; the default wings are 11-14% (bat arm /
    # propatagium, angel arm) and, because placement does not follow the morphs yet (design
    # bug 1), 3-17% across body morphs and poses. Below `min` it is tangent contact (a
    # hairline weld) and blocks; below `warn` the weld is shallow.
    min_root_embed: float = 0.05
    warn_root_embed: float = 0.08
    embed_samples: int = 16384         # Halton points per root piece (a big sheet gets an eighth)
    block_part_fraction: float = 0.02  # a detached part of this share of its appendage's volume
    block_part_volume_mm3: float = 2.0  # ... or this big, blocks; smaller floating bits warn
    block_tube_thin: float = 0.25      # of a tube's LENGTH under min_feature_mm -> block
    warn_tube_thin: float = 0.15       # of its length under 2x min_feature_mm -> warn
    below_feet_mm: float = 0.05        # tolerance for "below the standing plane"
    rays_per_shell: int = 200          # thickness probe budget per shell


DEFAULT_RULES = PrintRules()


@dataclass
class Finding:
    level: str                  # "block" | "warn" | "info"
    code: str                   # stable id: "shell", "detached", "root-weld", "thin", ...
    where: str                  # the shell / appendage group the finding is about
    message: str
    value: float | None = None
    limit: float | None = None

    def as_text(self) -> str:
        return f"{self.level.upper():5s} {self.where}: {self.message}"


@dataclass
class GroupReport:
    """What was measured for one placed appendage (``wing_l``, ``tail``, ``horn_r``)."""

    name: str
    shells: int = 0
    volume_mm3: float = 0.0              # sum of its shells' volumes
    root_embed: float | None = None      # best root piece, share of its volume inside the body
    root_pieces: dict = field(default_factory=dict)       # piece -> share inside
    root_gap_mm: float | None = None
    components: int = 1                  # overlap-connected components among its shells
    detached: list = field(default_factory=list)          # [(shells, volume_mm3)]
    thin_p5_mm: float | None = None      # 5th percentile (area weighted) of the thickness
    thin_below_min: float | None = None  # share of the area under min_feature_mm
    thin_below_thin: float | None = None
    posed_below_min: float | None = None
    tube: dict | None = None             # {"tip_mm", "below_min", "below_2x", "base_mm"}


@dataclass
class PrintabilityReport:
    mm_per_unit: float
    rules: PrintRules
    groups: dict
    findings: list
    assumption: str = ASSUMPTION

    @property
    def blocking(self) -> list:
        return [f for f in self.findings if f.level == "block"]

    @property
    def warnings(self) -> list:
        return [f for f in self.findings if f.level == "warn"]

    @property
    def ok(self) -> bool:
        return not self.blocking

    def as_dict(self) -> dict:
        return {
            "ok": self.ok, "mm_per_unit": self.mm_per_unit,
            "assumption": self.assumption,
            "findings": [dataclasses.asdict(f) for f in self.findings],
            "groups": {k: dataclasses.asdict(g) for k, g in self.groups.items()},
        }

    def as_text(self) -> str:
        head = (f"printability  {'OK' if self.ok else 'BLOCKED'}  "
                f"({len(self.blocking)} blocking, {len(self.warnings)} warnings)")
        lines = [head]
        for g in self.groups.values():
            bits = []
            if g.root_embed is not None:
                best = max(g.root_pieces, key=g.root_pieces.get) if g.root_pieces else "root"
                bits.append(f"{best} {100 * g.root_embed:.0f}% buried")
            if g.thin_p5_mm is not None:
                bits.append(f"p5 {g.thin_p5_mm:.2f} mm, "
                            f"{100 * g.thin_below_min:.1f}% < {self.rules.min_feature_mm:g} mm")
            if g.components > 1:
                bits.append(f"{g.components} components")
            if g.tube is not None:
                bits.append(f"tip {g.tube['tip_mm']:.2f} mm")
            lines.append(f"  {g.name:14s} {g.shells:3d} shells  " + ", ".join(bits))
        lines += [f.as_text() for f in self.findings if f.level != "info"]
        lines += [f"INFO  {f.where}: {f.message}" for f in self.findings if f.level == "info"]
        lines.append(f"note  {self.assumption}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# geometry: cheap inside / overlap tests on closed shells


class Solid:
    """A closed shell prepared for many "is this point inside" questions.

    Even-odd parity of a +X ray against the triangles, in barycentric coordinates of the
    (y, z) projection. Triangles are cropped to the query points' bounding box (and to
    ``x >= min x``), which is exact for a +X ray and turns a 26k-triangle body into a few
    hundred for a root piece: the difference between 0.1 s and 30 s.
    """

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        V = mesh.vertices
        tri = V[mesh.faces]
        self.lo, self.hi = (V.min(axis=0), V.max(axis=0)) if len(V) else (np.zeros(3), np.zeros(3))
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        self._a = a
        v0, v1 = c[:, 1:] - a[:, 1:], b[:, 1:] - a[:, 1:]
        self._v0, self._v1 = v0, v1
        self._d00 = (v0 * v0).sum(1)
        self._d01 = (v0 * v1).sum(1)
        self._d11 = (v1 * v1).sum(1)
        den = self._d00 * self._d11 - self._d01 ** 2
        self._ok = np.abs(den) > 1e-18
        self._den = np.where(self._ok, den, 1.0)
        self._dxc = c[:, 0] - a[:, 0]
        self._dxb = b[:, 0] - a[:, 0]
        self._tmin = tri[:, :, 1:].min(axis=1)
        self._tmax = tri[:, :, 1:].max(axis=1)
        self._xmax = tri[:, :, 0].max(axis=1)

    def contains(self, pts, chunk: int = 256) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        out = np.zeros(len(pts), dtype=bool)
        if not len(pts):
            return out
        # A ray through a shared edge or vertex crosses two triangles at once and the parity
        # flips. Axis-aligned or symmetric shapes do that on purpose; shifting every query a
        # hair off the grid (a ten-millionth of the shell) makes the ray generic.
        pts = pts + np.array([0.0, 1.31e-7, 0.77e-7]) * max(float((self.hi - self.lo).max()), 1e-9)
        yz_lo, yz_hi = pts[:, 1:].min(axis=0) - 1e-12, pts[:, 1:].max(axis=0) + 1e-12
        keep = (self._ok & (self._tmax >= yz_lo).all(axis=1) & (self._tmin <= yz_hi).all(axis=1)
                & (self._xmax >= pts[:, 0].min() - 1e-12))
        if not keep.any():
            return out
        a, v0, v1 = self._a[keep], self._v0[keep], self._v1[keep]
        d00, d01, d11, den = (self._d00[keep], self._d01[keep], self._d11[keep], self._den[keep])
        ax = a[:, 0]
        dxc, dxb = self._dxc[keep], self._dxb[keep]
        tmin, tmax, xmax = self._tmin[keep], self._tmax[keep], self._xmax[keep]
        # Walk the points in spatial order so each chunk covers a small patch of the (y, z)
        # plane and only has to look at the triangles over that patch: a long root piece
        # against the 26k-triangle body is ~8x cheaper this way, at the same answer.
        g = max(1, int(np.ceil(np.sqrt(len(pts) / chunk))))
        cell = np.minimum(((pts[:, 1:] - yz_lo) / np.maximum(yz_hi - yz_lo, 1e-300) * g).astype(int),
                          g - 1)
        order = np.argsort(cell[:, 1] * g + cell[:, 0], kind="stable")
        for s in range(0, len(pts), chunk):
            idx = order[s:s + chunk]
            P = pts[idx]
            lo, hi = P[:, 1:].min(axis=0) - 1e-12, P[:, 1:].max(axis=0) + 1e-12
            sel = (tmax >= lo).all(axis=1) & (tmin <= hi).all(axis=1) & (xmax >= P[:, 0].min())
            if not sel.any():
                continue
            v2 = P[:, None, 1:] - a[None, sel, 1:]
            d20 = (v2 * v0[None, sel]).sum(2)
            d21 = (v2 * v1[None, sel]).sum(2)
            u = (d11[sel] * d20 - d01[sel] * d21) / den[sel]
            w = (d00[sel] * d21 - d01[sel] * d20) / den[sel]
            hit = (u >= 0.0) & (w >= 0.0) & (u + w <= 1.0)
            bx = ax[None, sel] + u * dxc[None, sel] + w * dxb[None, sel]
            out[idx] = ((hit & (bx > P[:, None, 0])).sum(axis=1) % 2) == 1
        return out

    def boxes_overlap(self, other: "Solid") -> bool:
        return bool((self.lo <= other.hi).all() and (other.lo <= self.hi).all())

    def in_box(self, pts) -> np.ndarray:
        return ((pts >= self.lo) & (pts <= self.hi)).all(axis=1)


def _halton(n: int, bases=(2, 3, 5)) -> np.ndarray:
    """First ``n`` points of the Halton sequence in the unit cube: deterministic, and far
    more even than random points, so a coarse volume estimate does not jitter."""
    out = np.empty((n, len(bases)))
    idx = np.arange(1, n + 1)
    for k, b in enumerate(bases):
        r, f, i = np.zeros(n), 1.0 / b, idx.copy()
        while i.max() > 0:
            r += f * (i % b)
            i //= b
            f /= b
        out[:, k] = r
    return out


def embedded_volume_share(mesh: Mesh, body: Solid, samples: int = 4096) -> float:
    """Share of ``mesh``'s volume that lies inside the closed ``body`` shell.

    Halton points in the box where the two shells' boxes meet; the estimate is the share of
    them inside both, times the box volume, over the shell's exact volume. The tail's
    3.8% / horn's 0.2% embedding and a wing arm's 14% come out of this, and so do T1's sunk
    tails (27% at -0.6 units), within about a percentage point.
    """
    vol = mesh.volume()
    if vol <= 0 or not len(mesh.vertices):
        return 0.0
    lo = np.maximum(mesh.vertices.min(axis=0), body.lo)
    hi = np.minimum(mesh.vertices.max(axis=0), body.hi)
    if (hi <= lo).any():
        return 0.0
    P = lo + _halton(samples) * (hi - lo)
    inside = Solid(mesh).contains(P)
    if not inside.any():
        return 0.0
    both = int(body.contains(P[inside]).sum())
    return min(1.0, both * float(np.prod(hi - lo)) / samples / vol)


def shells_overlap(a: Solid, b: Solid, first: int = 48) -> bool:
    """Do two closed shells share volume? True when a vertex of either lies inside the other.

    Vertices are tried in two stages (an even sample first), because most pairs that overlap
    show it at once and most pairs that do not would otherwise pay for every vertex. A
    crossing that no vertex of either shell lies inside of is missed; at the mesh densities
    here that needs two knife edges to cross exactly, and the alula gap that matters
    (0.03 units) is far larger than a vertex spacing.
    """
    if not a.boxes_overlap(b):
        return False
    for p, q in ((a, b), (b, a)):
        V = p.mesh.vertices
        idx = np.nonzero(q.in_box(V))[0]
        if not len(idx):
            continue
        if len(idx) > first:
            head = idx[np.linspace(0, len(idx) - 1, first).astype(int)]
            if q.contains(V[head]).any():
                return True
            rest = np.setdiff1d(idx, head, assume_unique=True)
        else:
            rest = idx
        if len(rest) and q.contains(V[rest]).any():
            return True
    return False


def overlap_components(solids, links=None) -> list:
    """Union-find over shells: ``labels[i]`` is the component of ``solids[i]``.

    ``links`` is an optional set of ``(i, j)`` pairs already known to overlap (re-used from a
    congruent wing). The set is extended with every link found, so the caller can cache it.
    """
    n = len(solids)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if links:
        for i, j in links:
            parent[find(i)] = find(j)
    found = set(links or ())
    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j) or not solids[i].boxes_overlap(solids[j]):
                continue
            if shells_overlap(solids[i], solids[j]):
                parent[find(i)] = find(j)
                found.add((i, j))
    return [find(i) for i in range(n)], found


# --------------------------------------------------------------------------
# thickness


def _van_der_corput(n: int) -> np.ndarray:
    """The first ``n`` points (1..n) of the base-2 van der Corput sequence in [0, 1)."""
    out = np.zeros(n)
    i = np.arange(1, n + 1)
    f = 0.5
    while i.max() > 0:
        out += f * (i % 2)
        i = i // 2
        f /= 2.0
    return out


def _sample_faces(area: np.ndarray, k: int):
    """``k`` faces chosen at low-discrepancy quantiles of the cumulative area: a deterministic,
    area-proportional sample, each carrying weight ``total / k`` (all faces when k >= n).

    Evenly spaced quantiles ALIAS with a grid-ordered mesh (every 24th face of a slab is a
    rim column, or never is): on the bat's 4,780-face membrane they put 1.7% of the area under
    0.2 mm where the full probe says 0.44%. The van der Corput sequence is just as even and
    has no period (0.41% for the same 200 rays).
    """
    n = len(area)
    if k >= n:
        return np.arange(n), area.copy()
    cum = np.cumsum(area)
    return np.minimum(np.searchsorted(cum, _van_der_corput(k) * cum[-1]), n - 1), \
        np.full(k, cum[-1] / k)


def thickness_samples(mesh: Mesh, rays: int = 200, chunk: int = 64):
    """``(t, w)``: local thickness of a closed shell at ``rays`` face centroids, and the
    surface area each sample stands for.

    The thickness at a face is the distance from its CENTROID, along the inward face normal,
    to the first other triangle of the shell. (Per-vertex normals report 0.000 at every rim
    and bevel -- the ray hits the neighbouring rim quad at once -- and are not a thickness.)
    A face that sees nothing gets ``inf``.
    """
    V, F = mesh.vertices, mesh.faces
    if not len(F):
        return np.zeros(0), np.zeros(0)
    tri = V[F]
    a = tri[:, 0]
    e1, e2 = tri[:, 1] - a, tri[:, 2] - a
    fn = np.cross(e1, e2)
    ln = np.linalg.norm(fn, axis=1)
    nrm = fn / np.maximum(ln, 1e-300)[:, None]
    idx, w = _sample_faces(ln / 2.0, rays)
    cen = tri.mean(axis=1)
    ax, ay, az = a[:, 0], a[:, 1], a[:, 2]
    e1x, e1y, e1z = e1[:, 0], e1[:, 1], e1[:, 2]
    e2x, e2y, e2z = e2[:, 0], e2[:, 1], e2[:, 2]
    out = np.full(len(idx), np.inf)
    for s in range(0, len(idx), chunk):
        ii = idx[s:s + chunk]
        ox, oy, oz = (cen[ii, k][:, None] for k in range(3))
        dx, dy, dz = (-nrm[ii, k][:, None] for k in range(3))
        # Moller-Trumbore with the components spelled out: np.cross on (rays, faces, 3) is
        # several times slower than the same arithmetic on flat arrays
        hx, hy, hz = dy * e2z - dz * e2y, dz * e2x - dx * e2z, dx * e2y - dy * e2x
        det = e1x * hx + e1y * hy + e1z * hz
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        sx, sy, sz = ox - ax, oy - ay, oz - az
        u = (sx * hx + sy * hy + sz * hz) * inv
        qx, qy, qz = sy * e1z - sz * e1y, sz * e1x - sx * e1z, sx * e1y - sy * e1x
        v = (dx * qx + dy * qy + dz * qz) * inv
        t = (e2x * qx + e2y * qy + e2z * qz) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
        hit[np.arange(len(ii)), ii] = False
        out[s:s + chunk] = np.where(hit, t, np.inf).min(axis=1)
    return out, w


def _share_below(t, w, limit: float) -> float:
    tot = float(w.sum())
    return float(w[t < limit].sum()) / tot if tot > 0 else 0.0


def _percentile(t, w, p: float) -> float:
    if not len(t):
        return float("nan")
    o = np.argsort(t)
    cum = np.cumsum(w[o]) / w.sum()
    return float(t[o][min(int(np.searchsorted(cum, p)), len(t) - 1)])


def _deformed(rest: Mesh, posed: Mesh) -> bool:
    """Did the pose stretch this piece? A rigid piece keeps every edge length."""
    f = rest.faces
    if not len(f):
        return False
    le_r = np.linalg.norm(rest.vertices[f[:, 1]] - rest.vertices[f[:, 0]], axis=1)
    le_p = np.linalg.norm(posed.vertices[f[:, 1]] - posed.vertices[f[:, 0]], axis=1)
    return bool(np.abs(le_p - le_r).max() > 1e-6 * max(float(le_r.max()), 1e-12))


# --------------------------------------------------------------------------
# tails and horns: diameter along the length, analytically

# kind -> (builder, exponent, floor, minor-axis factor). ``radius(t) = base_radius * (1 - t)
# ** exponent + floor``; with a ``tip_radius`` param (the registry's rounded tip) it is
# ``tip + (base_radius - tip) * (1 - t) ** exponent`` instead. The cross-section is an ellipse
# whose minor semi-axis is ``minor`` x the radius. The law is the builder's own
# (appendages.make_tail / make_horn); the tests compare it with the built mesh, so a changed
# builder cannot drift unnoticed. A new tube kind registers its law here.
TUBE_LAWS = {"tail": ("make_tail", 0.85, 0.02, 0.92), "horn": ("make_horn", 1.25, 0.012, 1.0)}


def tube_diameters(kind: str, params: dict, scale: float = 1.0, n: int = 2001):
    """Thinnest cross-section diameter of a tail / horn (scene units) at ``n`` stations
    along its length, base to tip, or ``None`` for a kind without a registered law."""
    law = TUBE_LAWS.get(kind)
    if law is None:
        return None
    from . import appendages
    builder, expo, floor, minor = law
    base_radius = params.get("base_radius")
    if base_radius is None:
        base_radius = inspect.signature(getattr(appendages, builder)).parameters[
            "base_radius"].default
    t = np.linspace(0.0, 1.0, n)
    tip = params.get("tip_radius")
    if tip:       # a registry tail / horn with a rounded tip: radius = tip + (base - tip) (1-t)^e
        return (2.0 * minor * (float(tip) + (float(base_radius) - float(tip)) * (1.0 - t) ** expo)
                * float(scale))
    return 2.0 * minor * (float(base_radius) * (1.0 - t) ** expo + floor) * float(scale)


# --------------------------------------------------------------------------
# helpers on names


def group_of(name: str) -> str:
    """``"wing_l/primary05"`` -> ``"wing_l"``; ``"tail"`` -> ``"tail"``."""
    return name.split("/", 1)[0]


def piece_of(name: str):
    return name.split("/", 1)[1] if "/" in name else None


def _is_figure(name: str) -> bool:
    return name in pp.FIGURE_SHELLS


def _verts_key(verts) -> str:
    if verts is None:
        return "base"
    return hashlib.blake2b(np.ascontiguousarray(verts, dtype=np.float64).tobytes(),
                           digest_size=8).hexdigest()


def spec_key(app) -> str:
    """What identifies an appendage for caching: its own ``cache_key()`` if it has one,
    else its repr (a dataclass's repr lists every field)."""
    ck = getattr(app, "cache_key", None)
    return str(ck()) if callable(ck) else repr(app)


def rest_version(app):
    """The appendage without its pose controls: the unposed shape whose thickness is cached."""
    at_rest = getattr(app, "at_rest", None)
    if callable(at_rest):
        return at_rest()
    params = getattr(app, "params", None)
    if dataclasses.is_dataclass(app) and isinstance(params, dict):
        keys = set(getattr(app, "POSE_KEYS", ())) | {getattr(app, "PRESET_KEY", "preset")}
        if any(k in params for k in keys):
            return dataclasses.replace(app, params={k: x for k, x in params.items()
                                                    if k not in keys})
    return app


# --------------------------------------------------------------------------
# distance from an appendage root to the body surface (the frame check)


def _point_triangle_distance(p, tri) -> float:
    """Distance from one point to the nearest of ``tri`` (T, 3, 3): Ericson's closest point
    on a triangle, vectorised over the triangles."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b - a, c - a
    ap = p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    with np.errstate(divide="ignore", invalid="ignore"):
        # interior of the face
        den = va + vb + vc
        v_in = vb / den
        w_in = vc / den
        q = a + ab * v_in[:, None] + ac * w_in[:, None]
        # edge ab, ac, bc
        t_ab = np.clip(d1 / (d1 - d3), 0, 1)
        t_ac = np.clip(d2 / (d2 - d6), 0, 1)
        t_bc = np.clip((d4 - d3) / ((d4 - d3) + (d5 - d6)), 0, 1)
    cand = [
        q,
        a + ab * t_ab[:, None],
        a + ac * t_ac[:, None],
        b + (c - b) * t_bc[:, None],
        a, b, c,
    ]
    inside = (va >= 0) & (vb >= 0) & (vc >= 0)
    best = np.full(len(tri), np.inf)
    for k, pt in enumerate(cand):
        d = np.linalg.norm(p - pt, axis=1)
        d = np.where(np.isfinite(d) & (inside if k == 0 else True), d, np.inf)
        best = np.minimum(best, d)
    return float(best.min())


def surface_distance(points, mesh: Mesh, candidates: int = 48) -> float:
    """Smallest distance from any of ``points`` to the surface of ``mesh`` (0 for a point on it).

    The ``candidates`` points nearest to a mesh VERTEX are measured against every triangle,
    which is exact whenever the nearest point of the surface lies within a triangle's reach of
    the nearest vertex; for the roots of an appendage on a body that is the case with room to
    spare."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(P):
        return float("inf")
    V = mesh.vertices
    near = np.empty(len(P))
    vv = (V * V).sum(axis=1)
    for s in range(0, len(P), 1024):
        c = P[s:s + 1024]
        # |p - v|^2 = |p|^2 - 2 p.v + |v|^2: one matrix product instead of a (n, m, 3) temp
        near[s:s + 1024] = np.maximum(
            (c * c).sum(axis=1) + (vv[None, :] - 2.0 * c @ V.T).min(axis=1), 0.0)
    pick = np.argsort(near)[:candidates]
    tri = mesh.triangles()
    lo, hi = tri.min(axis=1), tri.max(axis=1)
    best = float("inf")
    for i in pick:
        r = min(float(np.sqrt(near[i])), best)          # the nearest surface point is within r
        near_tri = (lo <= P[i] + r).all(axis=1) & (hi >= P[i] - r).all(axis=1)
        if near_tri.any():                    # none: this point is farther than the best so far
            best = min(best, _point_triangle_distance(P[i], tri[near_tri]))
    return best


def root_gaps(shells, body: str = "body") -> dict:
    """``{appendage group: gap}``: how far the root of each appendage is from the body surface,
    in the units of ``shells`` (mm after :func:`printprep.assemble_character`).

    0.0 when any vertex of a root piece lies inside the body (it is buried); otherwise the
    distance from the nearest root vertex to the surface. The root is the named root pieces
    of a split wing (arm, hand, membrane, propatagium) or the whole shell of a tail / horn /
    arm. This is the post-assembly check of the FRAME: an appendage that was never turned
    from MakeHuman's Y up to the figure's Z up is a few millimetres away from the body (or
    inside nothing) and fails it, which the watertight gate cannot notice.
    """
    named = dict(shells)
    if body not in named:
        raise KeyError(f"no body shell {body!r} among {list(named)[:6]}")
    solid = Solid(named[body])
    groups: dict = {}
    for n, m in shells:
        if n == body or _is_figure(n):
            continue
        groups.setdefault(group_of(n), []).append((n, m))
    out = {}
    for g, items in groups.items():
        roots = [m for n, m in items if piece_of(n) in ROOT_PIECES] or [m for _, m in items]
        pts = np.vstack([m.vertices for m in roots])
        cand = solid.in_box(pts)
        if cand.any() and solid.contains(pts[cand]).any():
            out[g] = 0.0
        else:
            out[g] = surface_distance(pts, named[body])
    return out


def detached_appendages(shells, eps: float, body: str = "body") -> dict:
    """The groups of :func:`root_gaps` farther than ``eps`` (same units) from the body."""
    return {g: d for g, d in root_gaps(shells, body).items() if d > eps}


# --------------------------------------------------------------------------
# the analysis of ONE appendage, in scene units (cached)


@dataclass
class _Geometry:
    """What one appendage measures against the body (depends on the body, not on the size)."""

    shells: list                       # [(name, Mesh)] Z-up units, posed, split into pieces
    failures: list                     # [(name, message)] from the gate
    volume: dict                       # shell -> volume (units^3)
    boxes: dict                        # shell -> (lo, hi)
    components: dict                   # group -> [(attached to the body, [shell names])]
    root_share: dict                   # group -> {root shell: share of its vertices inside}
    zmin: dict                         # group -> lowest z (units)
    crossings: dict = field(default_factory=dict)   # collar shell -> pairs of its own triangles that cross


@dataclass
class _Thickness:
    """Thickness samples in scene units (depend on the appendage only, not body or size)."""

    rest: dict                         # shell -> (t, w)
    posed: dict                        # shell -> (t, w), only the shells the pose stretched


def _group_components(solids, names, body: Solid, link_cache: dict, sig) -> list:
    """Overlap-connected components of one group's shells and whether each reaches the body.

    The links among a wing's own pieces are shape-only, so they are found once and re-used
    for its congruent (mirrored) twin through ``link_cache``.
    """
    labels, found = overlap_components(solids, links=link_cache.get(sig))
    link_cache.setdefault(sig, found)
    comps: dict = {}
    for k, lab in enumerate(labels):
        comps.setdefault(lab, []).append(k)
    out = []
    for members in comps.values():
        # root pieces first: the arm reaches the body at once, the main body of a wing
        # needs one test, and only a detached part pays for all of them
        members = sorted(members, key=lambda k: piece_of(names[k]) not in ROOT_PIECES)
        attached = any(body.boxes_overlap(solids[k]) and shells_overlap(body, solids[k])
                       for k in members)
        out.append((attached, [names[k] for k in members]))
    return out


def _root_embedding(items, roots, body: Solid, rules: PrintRules) -> dict:
    """``{shell: share of its volume inside the body}`` for EVERY root shell.

    Nothing stops early: a mirrored pair must report the same piece and the same number, and
    an early exit at a threshold lets the Monte Carlo noise (about a point at these sample
    counts) decide which piece a side is judged by. The cost is kept down instead: a big sheet
    (the bat's 4,800-face membrane, 1% buried) gets an eighth of the points, the small
    tubes that carry the weld (arm, hand, propatagium) the full count.
    """
    out = {}
    for k in roots:
        name, mesh = items[k]
        n = rules.embed_samples if len(mesh.faces) <= 1500 else max(512, rules.embed_samples // 8)
        out[name] = embedded_volume_share(mesh, body, n)
    return out


def _geometry(shells, body: Solid, link_cache: dict, rules: PrintRules) -> _Geometry:
    by_group: dict = {}
    for n, m in shells:
        by_group.setdefault(group_of(n), []).append((n, m))
    volume = {n: m.volume() for n, m in shells}
    boxes = {n: (m.vertices.min(axis=0), m.vertices.max(axis=0)) for n, m in shells if len(m.vertices)}
    components, root_share, zmin = {}, {}, {}
    for g, items in by_group.items():
        names = [n for n, _ in items]
        solids = [Solid(m) for _, m in items]
        zmin[g] = min(float(m.vertices[:, 2].min()) for _, m in items)
        roots = [k for k, n in enumerate(names) if piece_of(n) in ROOT_PIECES] \
            or [k for k, n in enumerate(names) if piece_of(n) != COLLAR_PIECE] \
            or list(range(len(names)))
        root_share[g] = _root_embedding(items, roots, body, rules)
        # the links of a group are shape-only, so a mirrored twin may share them -- but a
        # different pose or placement of the same pieces must not: the signature carries the
        # pieces' pairwise centroid distances (invariant under a rigid motion or a mirror,
        # unlike the volumes alone, which a move leaves untouched)
        cen = np.array([m.vertices.mean(axis=0) for _, m in items])
        dist = np.round(np.linalg.norm(cen[:, None, :] - cen[None, :, :], axis=2), 5)
        sig = (tuple((piece_of(n), round(float(volume[n]), 9)) for n in names),
               hash(dist.tobytes()))
        components[g] = _group_components(solids, names, body, link_cache, sig)
    # a root collar is a closed shell whose own triangles must not cross: the report checks every shell
    # alone, and a slicer that does not union a shell with itself prints a crossing as a hole or a lip
    from .collar import count_crossings
    crossings = {n: count_crossings(m.vertices, m.faces) for n, m in shells if piece_of(n) == COLLAR_PIECE}
    return _Geometry(list(shells), pp.shell_failures(shells), volume, boxes, components,
                     root_share, zmin, crossings)


def _thickness(shells, rest, rules: PrintRules) -> _Thickness:
    rest_by = dict(rest)
    seen: dict = {}                     # a mirrored twin has the same samples
    out_rest, out_posed = {}, {}

    def probe(m: Mesh, rays: int):
        key = (len(m.faces), round(m.volume(), 9), round(m.area(), 9), rays)
        if key not in seen:
            seen[key] = thickness_samples(m, rays)
        return seen[key]

    for n, m in shells:
        if piece_of(n) == COLLAR_PIECE:
            continue
        r = rest_by.get(n, m)
        out_rest[n] = probe(r, rules.rays_per_shell)
        if r is not m and _deformed(r, m):
            # a pleated sheet is thin in patches, so it gets four times the rays (800 rays:
            # 22%, the full probe's 23%; 200 rays read 19%)
            out_posed[n] = probe(m, 4 * rules.rays_per_shell)
    return _Thickness(out_rest, out_posed)


def _memo(cache: dict, key, make):
    if key not in cache:
        cache[key] = make()
    return cache[key]


def _measure(app, base, verts, body_solid: Solid, bkey: str, rules: PrintRules,
             cache: dict, link_cache: dict):
    """``(geometry, thickness, kind, params, scale)`` of one appendage, from the cache when
    it has been measured on this body (geometry) / in this shape (thickness)."""
    if pp.ready_made_shells(app) is not None:        # ready-made meshes, MakeHuman frame
        shells = pp.appendage_print_shells([app])
        return (_geometry(shells, body_solid, link_cache, rules), _thickness(shells, [], rules),
                None, {}, 1.0)
    key = spec_key(app)
    rest_app = rest_version(app)
    built: dict = {}

    def shells_of(a):
        k = id(a)
        if k not in built:
            built[k] = pp.appendage_print_shells([a], base, verts, split_pieces=True)
        return built[k]

    geo = _memo(cache, ("geom", key, bkey, rules.embed_samples),
                lambda: _geometry(shells_of(app), body_solid, link_cache, rules))
    th = _memo(cache, ("thick", spec_key(rest_app), key, rules.rays_per_shell),
               lambda: _thickness(shells_of(app), shells_of(rest_app), rules))
    return (geo, th, getattr(app, "kind", None), getattr(app, "params", {}),
            float(getattr(app, "scale", 1.0)))


def appendage_report(appendages, base=None, verts=None, mm_per_unit: float = 1.0, *,
                     body=None, levels: int = 0, figure_shells=None,
                     rules: PrintRules | None = None, cache: dict | None = None,
                     min_thickness_mm=None, assembly=None, body_verts=None) -> PrintabilityReport:
    """The printability report of ``appendages`` on the body ``verts``, at ``mm_per_unit``.

    ``body_verts`` (default ``verts``) are the vertices the printed body is MESHED from: the
    skin moulded around the appendages (``junction.shaped_body``) -- the root weld is measured
    against the body that prints. ``verts`` stays the body the appendages are PLACED on.

    ``appendages`` are what :func:`printprep.appendage_print_shells` takes. ``body`` is the
    PRINTED body shell (Z-up, scene units); by default the cage, or the cage subdivided
    ``levels`` times as the print will be. ``figure_shells`` (``character_print_shells``) adds
    the figure's own shells to the gate. ``min_thickness_mm`` applies the thickness floor
    (:func:`printprep.apply_thickness_floor`) to the wings first -- pass the same value the
    export uses -- and the report names what changed. ``assembly`` is the
    :class:`printprep.PrintReport` of the export, for the base overhang.

    ``cache`` is a dict the caller keeps between calls. Thickness is measured once per
    appendage (on the unposed shape, in scene units), connectivity once per appendage and
    body; a different print size or different thresholds re-use both. Cost: a bat wing pair
    and a tail about 0.4 s uncached (folded 0.9 s), an angel pair about 2 s (folded 2.7 s),
    under 0.1 s cached.
    """
    rules = DEFAULT_RULES if rules is None else rules
    cache = {} if cache is None else cache
    apps = list(appendages)
    adjustments = []
    if min_thickness_mm is not None:
        apps, adjustments = pp.apply_thickness_floor(apps, mm_per_unit, min_thickness_mm)
    if body is None:
        body = pp.character_print_shells(base, verts if body_verts is None else body_verts,
                                         levels=levels, with_eyes=False,
                                         with_teeth=False)[0][1]
    bkey = _verts_key(body.vertices)
    body_solid = _memo(cache, ("solid", bkey), lambda: Solid(body))
    link_cache = cache.setdefault("links", {})

    findings: list = []
    groups: dict = {}
    for fail_name, msg in (pp.shell_failures(figure_shells) if figure_shells else []):
        findings.append(Finding("block", "shell", fail_name, msg))

    used: set = set()
    analyses = []
    for app in apps:
        geo, th, kind, params, scale = _measure(app, base, verts, body_solid, bkey, rules,
                                                cache, link_cache)
        mapping = {}
        for g in dict.fromkeys(group_of(n) for n, _ in geo.shells):
            new, k = g, 2
            while new in used:
                new, k = f"{g}#{k}", k + 1
            mapping[g] = new
            used.add(new)
        analyses.append((app, geo, th, kind, params, scale, mapping))

    body_zmin = float(body.vertices[:, 2].min())
    group_boxes: dict = {}
    for app, geo, th, kind, params, scale, mapping in analyses:
        for name, msg in geo.failures:
            findings.append(Finding("block", "shell", _rename(name, mapping), msg))
        for name, nx in geo.crossings.items():
            if nx:
                findings.append(Finding(
                    "warn", "self-crossing", _rename(name, mapping),
                    f"{name} crosses itself ({nx} pairs of triangles): a slicer that does not union a shell "
                    "with itself may print the crossing as a hole or a lip", float(nx), 0.0))
        by_group: dict = {}
        for n in (n for n, _ in geo.shells):
            by_group.setdefault(group_of(n), []).append(n)
        for g, members in by_group.items():
            gname = mapping[g]
            rep = GroupReport(gname, shells=len(members))
            groups[gname] = rep
            vol_u = sum(geo.volume[n] for n in members)
            rep.volume_mm3 = vol_u * mm_per_unit ** 3
            lo = np.min([geo.boxes[n][0] for n in members], axis=0)
            hi = np.max([geo.boxes[n][1] for n in members], axis=0)
            group_boxes[gname] = (lo, hi, app, g)

            # -- the weld: how deep the root is buried ----------------------------------
            shares = geo.root_share[g]
            rep.root_pieces = {piece_of(n) or n: s for n, s in shares.items()}
            rep.root_embed = max(shares.values())
            best = max(shares, key=shares.get)
            if rep.root_embed < rules.min_root_embed:
                findings.append(Finding(
                    "block", "root-weld", gname,
                    f"only {100 * rep.root_embed:.1f}% of the volume of {best} lies in the body "
                    f"(needs {100 * rules.min_root_embed:.0f}%): it touches the skin along a "
                    "hairline and prints detached; sink it into the body",
                    rep.root_embed, rules.min_root_embed))
            elif rep.root_embed < rules.warn_root_embed:
                findings.append(Finding(
                    "warn", "root-weld", gname,
                    f"{best} is only {100 * rep.root_embed:.1f}% buried in the body "
                    f"(a firm weld is {100 * rules.warn_root_embed:.0f}% or more): the join is "
                    "shallow", rep.root_embed, rules.warn_root_embed))

            # -- connection: parts floating free of the body ----------------------------
            comps = geo.components[g]
            rep.components = len(comps)
            for attached, mem in comps:
                if attached:
                    continue
                vol = sum(geo.volume[n] for n in mem)
                frac = vol / vol_u if vol_u > 0 else 1.0
                vmm = vol * mm_per_unit ** 3
                rep.detached.append((mem, vmm))
                big = frac >= rules.block_part_fraction or vmm >= rules.block_part_volume_mm3
                shown = ", ".join(mem[:4]) + (f" +{len(mem) - 4} more" if len(mem) > 4 else "")
                findings.append(Finding(
                    "block" if big else "warn", "detached", gname,
                    f"{shown} ({vmm:.2f} mm3, {100 * frac:.1f}% of its appendage) is not "
                    "connected to the body: a slicer prints it as a floating island",
                    frac, rules.block_part_fraction))

            # -- thickness at this size -------------------------------------------------
            solid_members = [n for n in members if n in th.rest]
            t_all = (np.concatenate([th.rest[n][0] for n in solid_members]) * mm_per_unit
                     if solid_members else np.zeros(0))
            w_all = (np.concatenate([th.rest[n][1] for n in solid_members])
                     if solid_members else np.zeros(0))
            if len(t_all):
                rep.thin_p5_mm = _percentile(t_all, w_all, 0.05)
                rep.thin_below_min = _share_below(t_all, w_all, rules.min_feature_mm)
                rep.thin_below_thin = _share_below(t_all, w_all, rules.thin_feature_mm)
                if rep.thin_below_min > rules.block_area_thin:
                    findings.append(Finding(
                        "block", "thin", gname,
                        f"{100 * rep.thin_below_min:.0f}% of its area is thinner than "
                        f"{rules.min_feature_mm:g} mm at this size (p5 {rep.thin_p5_mm:.2f} mm): "
                        "print larger, or raise the thickness floor",
                        rep.thin_below_min, rules.block_area_thin))
                elif rep.thin_below_thin > rules.warn_area_thin:
                    findings.append(Finding(
                        "warn", "thin", gname,
                        f"{100 * rep.thin_below_thin:.0f}% of its area is thinner than "
                        f"{rules.thin_feature_mm:g} mm at this size: fragile",
                        rep.thin_below_thin, rules.warn_area_thin))
            posed = [n for n in solid_members if n in th.posed]
            if posed:
                tp = np.concatenate([(th.posed[n] if n in th.posed else th.rest[n])[0]
                                     for n in solid_members]) * mm_per_unit
                wp = np.concatenate([(th.posed[n] if n in th.posed else th.rest[n])[1]
                                     for n in solid_members])
                rep.posed_below_min = _share_below(tp, wp, rules.min_feature_mm)
                if rep.posed_below_min > rules.warn_posed_area_thin:
                    findings.append(Finding(
                        "warn", "thin-posed", gname,
                        f"posed, {100 * rep.posed_below_min:.0f}% of its area is thinner than "
                        f"{rules.min_feature_mm:g} mm (folds lay sheet on sheet; print it "
                        "spread, or check the fold in the slicer)",
                        rep.posed_below_min, rules.warn_posed_area_thin))

            # -- a tail or horn tapers to a point ---------------------------------------
            diam = tube_diameters(kind, params, scale) if kind else None
            if diam is not None:
                d = diam * mm_per_unit
                rep.tube = {"base_mm": float(d[0]), "tip_mm": float(d[-1]),
                            "below_min": float(np.mean(d < rules.min_feature_mm)),
                            "below_2x": float(np.mean(d < 2 * rules.min_feature_mm))}
                if rep.tube["below_min"] > rules.block_tube_thin:
                    findings.append(Finding(
                        "block", "tube-thin", gname,
                        f"{100 * rep.tube['below_min']:.0f}% of its length is thinner than "
                        f"{rules.min_feature_mm:g} mm at this size: it will not print as drawn",
                        rep.tube["below_min"], rules.block_tube_thin))
                elif rep.tube["below_2x"] > rules.warn_tube_thin:
                    findings.append(Finding(
                        "warn", "tube-thin", gname,
                        f"{100 * rep.tube['below_2x']:.0f}% of its length is thinner than "
                        f"{2 * rules.min_feature_mm:g} mm: the end is fragile",
                        rep.tube["below_2x"], rules.warn_tube_thin))
                if rep.tube["tip_mm"] < rules.min_feature_mm:
                    findings.append(Finding(
                        "warn", "tube-tip", gname,
                        f"the tip is {rep.tube['tip_mm']:.2f} mm across, below what a printer "
                        "resolves: it rounds off",
                        rep.tube["tip_mm"], rules.min_feature_mm))

            # -- the standing plane -----------------------------------------------------
            if (geo.zmin[g] - body_zmin) * mm_per_unit < -rules.below_feet_mm:
                findings.append(Finding(
                    "warn", "below-feet", gname,
                    f"reaches {(body_zmin - geo.zmin[g]) * mm_per_unit:.1f} mm below the feet: "
                    "it would sink into the base"))

    findings += _appendage_overlaps(analyses, group_boxes)

    for a in adjustments:
        findings.append(Finding("info", "floor", a.appendage, a.as_text()))
    if assembly is not None and getattr(assembly, "overhang_mm", 0.0) > 0.5:
        findings.append(Finding(
            "warn", "overhang", "base",
            f"appendages overhang the base rim by {assembly.overhang_mm:.1f} mm "
            "(fit_base=True widens it)", assembly.overhang_mm))
    findings.sort(key=lambda f: {"block": 0, "warn": 1, "info": 2}[f.level])
    return PrintabilityReport(mm_per_unit, rules, groups, findings)


def _rename(name: str, mapping: dict) -> str:
    g = group_of(name)
    return mapping.get(g, g) + name[len(g):]


def _appendage_overlaps(analyses, group_boxes) -> list:
    """Warn for every pair of placed appendages that share volume (a folded wing against the
    tail, a pair of folded wings against each other). Not a print failure: the slicer unions
    them. Tests only the shells whose boxes overlap, and stops at the first hit per pair."""
    out = []
    names = list(group_boxes)
    shells_of = {}
    mapping_of = {}
    for app, geo, th, kind, params, scale, mapping in analyses:
        for g, new in mapping.items():
            mapping_of[new] = (geo, g)
    solid_of: dict = {}

    def members(gname):
        geo, g = mapping_of[gname]
        return [(n, m) for n, m in geo.shells if group_of(n) == g]

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            lo_a, hi_a = group_boxes[a][:2]
            lo_b, hi_b = group_boxes[b][:2]
            if (lo_a > hi_b).any() or (lo_b > hi_a).any():
                continue
            hit = None
            for na, ma in members(a):
                for nb, mb in members(b):
                    sa = solid_of.setdefault((a, na), None)
                    if sa is None:
                        sa = solid_of[(a, na)] = Solid(ma)
                    sb = solid_of.get((b, nb))
                    if sb is None:
                        sb = solid_of[(b, nb)] = Solid(mb)
                    if sa.boxes_overlap(sb) and shells_overlap(sa, sb):
                        hit = (na, nb)
                        break
                if hit:
                    break
            if hit:
                out.append(Finding("warn", "overlap", f"{a} x {b}",
                                   f"{hit[0]} intersects {hit[1]}: the slicer unions them, "
                                   "but check the join"))
    return out
