"""A feather made the way a feather is made: a shaft, and barbs that grow from it.

``wings._blade`` draws a feather as an outline filled with a sheet.  Here the same outline is filled by a hair
system (``core/hair.py``): the shaft (the rachis) is a curve, a barb leaves it on either side every so often,
heading for the tip at ``BARB_ANGLE`` and sweeping further towards it as it goes, and each is cut where it
reaches the feather's outline, which is exactly ``_blade``'s (the same parameters mean the same silhouette:
the narrow leading vane, the notch of a slotted primary, the rounded end).  A split in the vane is a barb's
neighbour cut short; the frayed edge is every barb's own length.

The mesh is the barbs ZIPPED, as barbules zip them on a bird: each barb is a row of ONE surface, alternate
rows raised so every barb is a ridge, the other face a coarse sheet of its own (``hair.sheet_mesh``).  A flight
feather has about twenty barbs a side for the triangles ``_blade`` spends on a sheet with none; ``detail``
buys more.

Feathers are grown in batches: ``barbed_blades`` takes the arguments of many and grows all that share a mesh
size as ONE strand system (a wing's 271 feathers are ten sizes), which is what makes a wing of them build
in half the time a wing of sheets does.
"""

from __future__ import annotations

import numpy as np

from ..core import hair as H
from ..core.mesh import Mesh

__all__ = ["barbed_blade", "barbed_blades", "barb_strands", "barb_count", "BARB_HOOK", "BARB_RELIEF",
           "BARB_FRAY"]

BARB_HOOK = 0.30         # how much of its angle a barb loses by its end: it leaves the shaft at BARB_ANGLE and
                         # sweeps towards the feather's tip
BARB_ACUTE = 0.45        # ... and the barbs near the tip leave the shaft this much more acutely than at the root
BARB_BASE = 0.26         # the share of the length, from the root, over which the barbs come round from square
BARB_RELIEF = 0.22       # how far a barb's ridge stands over the vane, in sheet thicknesses
BARB_FRAY = 0.035        # the share of its length a barb may fall short of the outline by (the edge's fray)
SHAFT_COLUMN = 1.35      # where the column beside the shaft lies, in shaft half-widths
BARB_STEPS = 10          # segments a barb is grown in before it is cut

# ``_blade``'s parameters and their defaults (``edge`` and ``cap`` default to wings.EDGE and wings.TIP_CAP)
_DEFAULTS = dict(curl=0.10, cup=0.16, sweep=0.08, lead_frac=0.30, tip=0.74, tip_pow=1.8, emarg=0.0,
                 emarg_in=0.0, calamus=0.10, tip_floor=0.07, root_floor=0.07, n=11, m=8, notch=0.55,
                 rise_at=0.42, ridge=1.35, vane=0.90, ridge_dir=-1.0, butt=0.0, edge=None, cap=None, vee=0.0,
                 splits=(), barbs=0.0, barb_from=0.0, detail=1.0, count=None, relief=BARB_RELIEF, seed=0,
                 triangles=None, both=False)
_NUMBERS = ("length", "width", "thickness", "curl", "cup", "sweep", "lead_frac", "tip", "tip_pow", "emarg",
            "emarg_in", "calamus", "tip_floor", "root_floor", "notch", "rise_at", "ridge", "vane", "ridge_dir",
            "butt", "edge", "cap", "vee", "relief")


def _outline(t, p):
    """``(lead, trail)`` half-widths at ``t`` (F, X) along ``F`` feathers whose parameters ``p`` are (F, 1)
    arrays: ``_blade``'s outline, less its splits (here a split is a barb cut short)."""
    from . import wings as W
    rise = np.clip((t - p["calamus"]) / (p["rise_at"] - p["calamus"]), 0.0, 1.0) ** 0.8
    ogive = np.clip(1.0 - (np.clip(t - p["tip"], 0.0, None) / (1.0 - p["tip"])) ** p["tip_pow"], 0.0, 1.0) ** 0.5
    floor = np.where(t > 0.5, p["tip_floor"], p["root_floor"])
    wd = p["width"] * np.maximum(rise * ogive, floor)
    # the rounded end, where there is one (a cap and a floor to round)
    end_w = np.maximum(p["tip_floor"] * p["width"], 0.5 * p["thickness"])
    cap_t = np.where((p["cap"] > 0) & (p["tip_floor"] > 0),
                     np.clip(p["cap"] * 0.5 * end_w / np.maximum(p["length"], 1e-9), 0.0, W.TIP_CAP_MAX), 0.0)
    capped = np.sqrt(np.clip(1.0 - (np.clip(t - (1.0 - cap_t), 0.0, None) / np.maximum(cap_t, 1e-12)) ** 2, 0.0, 1.0))
    wd = wd * np.where(cap_t > 0.0, capped, 1.0)
    butt = np.sqrt(np.clip(1.0 - (1.0 - np.clip(t / np.maximum(p["butt"], 1e-12), 0.0, 1.0)) ** 2, 0.0, 1.0))
    wd = wd * np.where(p["butt"] > 0.0, butt, 1.0)
    wd = np.maximum(wd, 0.5 * p["thickness"])
    lead = p["lead_frac"] * wd * (1.0 - p["emarg"] * W._step(t, p["notch"] - 0.05, p["notch"] + 0.04))
    trail = (1.0 - p["lead_frac"]) * wd * (1.0 - p["emarg_in"] * W._step(t, p["notch"] - 0.12, p["notch"] - 0.03))
    return lead, trail


def barb_count(spec: dict) -> tuple:
    """``(barbs a side, columns)`` of the mesh one feather is grown as: ``count`` if it says, else as many
    barbs as ``triangles`` pay for (default: what ``_blade`` spends on the same feather), times ``detail``."""
    from . import wings as W
    n, m = int(spec["n"]), int(spec["m"])
    fine_cols = m >= 8                                   # a flight feather: a column either side of the shaft
    if spec["count"] is not None:
        return int(spec["count"]), 7 if fine_cols else 5
    triangles = spec["triangles"]
    if triangles is None:
        # exactly what ``_blade`` spends on the same feather: its own rows (the body and the tip, the
        # notch's, the cap's), times its columns.  (Not the rows it adds for a split: a split does not change
        # how many barbs a feather has, and the sheet with a split only has more.)
        cap = W.TIP_CAP if spec["cap"] is None else spec["cap"]
        rows = len(W._blade_rows(spec["length"], spec["width"], spec["thickness"], spec["lead_frac"], spec["tip"],
                                 spec["emarg"], spec["emarg_in"], spec["tip_floor"], n, spec["notch"], cap,
                                 (), 0.0, spec["barb_from"])[0]) - 1
        triangles = 4 * rows * m + 4 * (rows + m)
    per_barb, fixed = (34, 20) if fine_cols else (26, 16)
    return int(max(4, np.floor(spec["detail"] * (triangles - fixed) / per_barb))), 7 if fine_cols else 5


def barb_strands(s, length, sweep, angle_deg: float, reach, steps: int = BARB_STEPS) -> H.Strands:
    """The barbs of ``F`` feathers before they are cut, as ONE hair system of ``F * 2R`` strands: feather
    after feather, the ``R`` barbs of its leading vane and then the ``R`` of its trailing one
    (``attrs["side"]``: -1 / +1, ``attrs["feather"]``, ``attrs["at"]``: where along the shaft it leaves).

    ``s`` (R,) is where along the shaft each pair leaves it (0..1); ``length``, ``sweep`` and ``reach`` are
    (F,).  A shaft runs along +X and bows ``sweep * length * s**2`` towards +Z; its leading vane is on -Z.
    A barb is ``reach`` long: longer than any outline, the cut comes after."""
    s = np.asarray(s, dtype=np.float64)
    L = np.atleast_1d(np.asarray(length, dtype=np.float64))[:, None]
    sw = np.atleast_1d(np.asarray(sweep, dtype=np.float64))[:, None]
    F_, R = len(L), len(s)
    roots = np.stack([L * s, np.zeros((F_, R)), sw * L * s ** 2], axis=2)             # (F, R, 3)
    tangent = np.stack([np.ones((F_, R)), np.zeros((F_, R)), 2.0 * sw * s[None, :]], axis=2)
    tangent /= np.linalg.norm(tangent, axis=2, keepdims=True)
    # at BARB_ANGLE to the shaft, more acutely towards the tip, and squarely at the very base (the loose barbs
    # of the base stand out from the quill; and a covert's broad butt is filled, not an arrowhead)
    base = 1.0 - np.clip(s / BARB_BASE, 0.0, 1.0)
    alpha = np.radians(angle_deg) * (1.0 - BARB_ACUTE * s ** 2)
    alpha = alpha + (np.radians(88.0) - alpha) * base * base * (3.0 - 2.0 * base)
    side = np.tile(np.repeat([-1.0, 1.0], R), F_)                                      # (F * 2R,)
    a2 = np.tile(alpha, 2 * F_)
    up = np.array([0.0, 1.0, 0.0])
    roots2 = np.concatenate([roots, roots], axis=1).reshape(-1, 3)
    heading = H._turn(np.concatenate([tangent, tangent], axis=1).reshape(-1, 3), up, -side * a2)
    return H.grow(roots2, heading, np.repeat(np.atleast_1d(reach), 2 * R), steps, bend=side * BARB_HOOK * a2,
                  bend_axis=up, attrs={"side": side, "feather": np.repeat(np.arange(F_), 2 * R),
                                       "at": np.tile(s, 2 * F_)})


def _grow_group(specs: list, B: int, C: int, both: bool = False) -> list:
    """``F`` feathers of one mesh size (``B`` barbs a side, ``C`` columns; ``both``: barbs on the far face
    too), grown together."""
    from . import wings as W
    F_ = len(specs)
    p = {k: np.array([float(sp[k]) for sp in specs])[:, None] for k in _NUMBERS}        # (F, 1) each
    L, thick = p["length"], p["thickness"]
    R = 2 * B + 1
    fine_cols = C == 7
    s = np.arange(R) / float(R)                          # the last row stops short of the tip: the tip is a point
    barbs = barb_strands(s, L[:, 0], p["sweep"][:, 0], W.BARB_ANGLE, (1.2 * p["width"] + 0.25 * L)[:, 0])
    K = barbs.steps
    side = barbs.attrs["side"].reshape(F_, 2 * R)
    roots = barbs.points.reshape(F_, 2 * R, K, 3)[:, :R, 0]                              # (F, R, 3)

    def inside(pts):
        # in the feather: this side of the outline, and short of the feather's end
        q = pts.reshape(F_, 2 * R * K, 3)
        t = q[:, :, 0] / L
        wl, wt = _outline(np.clip(t, 0.0, 1.0), p)
        off = q[:, :, 2] - p["sweep"] * L * t ** 2
        return np.minimum(np.where(np.repeat(side, K, axis=1) < 0, wl + off, wt - off), L - q[:, :, 0]).ravel()

    # every barb its own length (the edge's fray); a split is the barb BETWEEN two barbs cut short
    odd = (np.arange(R) % 2 == 1)
    seed = np.array([int(sp["seed"]) for sp in specs])[:, None]
    cut = 1.0 - BARB_FRAY * H.hash01(np.arange(2 * R)[None, :], seed, 5) * np.tile(odd, 2)[None, :]
    for f, sp in enumerate(specs):
        for at, depth in sp["splits"]:
            r = int(np.clip(2 * round(0.5 * at * R), 2, R - 3))        # the valley row nearest ``at``
            cut[f, R + r] = 1.0 - float(depth)                         # (of the trailing vane)
    kept = H.cut_length(barbs, inside) * cut.ravel()

    # columns along each barb: beside the shaft (a flight feather), where the edge's bevel starts, the edge
    wl, wt = _outline(np.broadcast_to(s, (F_, R)), p)
    shaft_l = np.maximum(W.SHAFT_WIDTH * wl, W.SHAFT_MIN * thick)
    shaft_t = np.maximum(W.SHAFT_WIDTH * wt, W.SHAFT_MIN * thick)
    half, shaft2 = np.concatenate([wl, wt], axis=1), np.concatenate([shaft_l, shaft_t], axis=1)    # (F, 2R)
    edge_col = np.full((F_, 2 * R), W.EDGE_AT)
    if fine_cols:
        u_s = np.clip(SHAFT_COLUMN * shaft2 / np.maximum(half, 1e-9), 0.10, 0.5 * W.EDGE_AT)
        u2 = np.stack([u_s, edge_col, np.ones((F_, 2 * R))], axis=2)
    else:
        u2 = np.stack([edge_col, np.ones((F_, 2 * R))], axis=2)
    cols = u2.shape[2]
    pts = H.sample(barbs.points, kept[:, None] * u2.reshape(-1, cols)).reshape(F_, 2 * R, cols, 3)
    pl, pt, ul, ut = pts[:, :R], pts[:, R:], u2[:, :R], u2[:, R:]
    grid = np.concatenate([pl[:, :, ::-1], roots[:, :, None], pt], axis=2)             # lead edge .. shaft .. trailing
    u = np.concatenate([ul[:, :, ::-1], np.zeros((F_, R, 1)), ut], axis=2)             # 0 at the shaft, 1 at an edge
    Lc = L[:, :, None]
    t = np.clip(grid[..., 0] / Lc, 0.0, 1.0)
    off = np.abs(grid[..., 2] - p["sweep"][:, :, None] * Lc * t ** 2)                  # how far from the shaft, across
    rd = p["ridge_dir"][:, :, None]
    # the shaft droops and the vanes cup and slope away from it, as ``_blade``'s do
    grid[..., 1] = p["curl"][:, :, None] * Lc * t ** 2 - rd * (p["cup"][:, :, None] * off * u ** 0.5
                                                                 + p["vee"][:, :, None] * off)

    # thickness: the vane's sheet bevelled to its edge, the shaft's ridge, and every second row a barb's ridge
    col_side = np.concatenate([np.full(cols, -1), [0], np.full(cols, 1)])
    shaft = np.where(col_side[None, None, :] < 0, shaft_l[:, :, None], shaft_t[:, :, None])
    vane, edge = p["vane"][:, :, None], np.minimum(p["edge"], p["vane"])[:, :, None]
    flat = vane - (vane - edge) * W._step((u - W.EDGE_AT) / (1.0 - W.EDGE_AT), 0.0, 1.0)
    extra = (p["ridge"][:, :, None] - vane) * np.exp(-(off / shaft) ** 2) * (1.0 - W.SHAFT_FADE * t)
    cal = p["calamus"][:, :, None]
    on = (W._step(t, cal, cal + 0.08) * (1.0 - W._step(t, 0.94, 0.995))
          * W._step(u, 0.0, 0.25) * (1.0 - W._step(u, 0.88, 1.0)))
    rib = p["relief"][:, :, None] * on * odd[None, :, None]
    bevel = 1.0 - 0.4 * W._step(t, 0.965, 1.0)
    th = thick[:, :, None]
    fine = th * bevel * (0.5 * flat + extra + rib)
    coarse = th * bevel * (0.5 * flat + (rib if both else 0.0))          # (the far face: barbs, no shaft ridge)

    normals = H._grid_normals(grid)
    wrong = np.sign(normals[..., 1].mean(axis=(1, 2))) != np.sign(p["ridge_dir"][:, 0])
    normals[wrong] *= -1.0                               # the fine face is the one the shaft stands off
    tips = np.concatenate([L, p["curl"] * L, p["sweep"] * L], axis=1)
    end = (0.6 * 0.5 * thick * np.minimum(p["edge"], p["vane"]))[:, 0]
    return H.sheet_meshes(grid, fine, coarse, tips, end, end, normals, both=both)


def barbed_blades(specs) -> list:
    """Many feathers at once: ``specs`` are dicts of :func:`barbed_blade`'s arguments (``length``, ``width``
    and ``thickness`` among them).  Those that share a mesh size are grown as one strand system.  The meshes
    come back in the order of ``specs``."""
    from . import wings as W
    full = []
    for sp in specs:
        d = {**_DEFAULTS, **sp}
        d["edge"] = W.EDGE if d["edge"] is None else d["edge"]
        d["cap"] = W.TIP_CAP if d["cap"] is None else d["cap"]
        full.append(d)
    groups: dict = {}
    for i, d in enumerate(full):
        groups.setdefault(barb_count(d) + (bool(d["both"]),), []).append(i)
    out: list = [None] * len(full)
    for (B, C, both), idx in groups.items():
        for i, m in zip(idx, _grow_group([full[i] for i in idx], B, C, both)):
            out[i] = m
    return out


def barbed_blade(length, width, thickness, **kw) -> Mesh:
    """One feather grown from its shaft: ``_blade``'s parameters, frame (+X its length, +Z its trailing vane,
    the shaft standing off the ``ridge_dir`` face) and outline.

    ``count`` is the barbs on each side of the shaft; by default as many as ``triangles`` pay for (default:
    what ``_blade`` spends on the same feather, from ``n`` and ``m``), times ``detail``.  ``relief`` is how far
    a barb stands over the vane (0: a smooth vane on a barbed mesh).  ``seed`` frays the edge differently.
    ``both``: barbs on the far face too (as many triangles again on that face: a feather that is seen from
    underneath).  ``barbs`` / ``barb_from`` (``_blade``'s grooves) mean nothing here: the barbs are the mesh."""
    return barbed_blades([dict(kw, length=length, width=width, thickness=thickness)])[0]
