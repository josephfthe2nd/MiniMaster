"""Head hair: the scalp's strands combed into a style (``strands.py`` grows and meshes them).

A style is a COMB PATTERN, the direction the hair is laid in over the scalp; how long it is, how much it
stands off the head, whether it waves or curls and how the fringe is cut are separate numbers, so "long and
swept back" and "cropped and swept back" are the same pattern at two lengths.

``natural``      falls from the whorl at the back of the crown: forwards over the forehead (the fringe),
                 down the sides and the back
``swept_back``   combed straight back off the face
``centre_part``  parted down the middle, falling to each side
``side_part``    parted over the left eye
``spiky``        stands off the head all over
``mohawk``       a crest along the midline, the rest of the scalp bare

The fringe is cut where the forward-combed hair reaches the brow (``fringe`` 1.0), higher (below 1) or lower.
Roots are emitted on the neutral body and ride the morphs (``scatter.py``), so a head of hair keeps its locks
when the body under it changes.
"""

from __future__ import annotations

import numpy as np

from . import scatter as SC
from . import strands as ST

__all__ = ["STYLES", "build_hair", "head_landmarks"]

SUNK = 0.05          # how deep a lock's underside lies under the CAGE where it lies on the skin (units): the
                     # weld.  (The skin that is drawn and printed lies 0.01-0.02 inside the cage on a head.)
STAND_SINK = 0.25    # a lock that stands on its root (a spike, a crest) is sunk this share of its length, at
                     # its full width and thickness: EVERY lock of a crest has 6-7% or more of its volume in the
                     # head.  (At 0.16 with a root as slim as a lying lock's some had under 5%, and the print
                     # report, which judges an appendage by its best-buried piece, did not see it.)

STYLES = ("natural", "swept_back", "centre_part", "side_part", "spiky", "mohawk")

# per style: lift (degrees off the skin), droop scale, the share of the scalp's width the roots keep to
# (mohawk), and whether forward-combed roots are cut as a fringe
_STYLE = {
    "natural":     dict(lift=14.0, droop=1.0, strip=None, fringe=True),
    "swept_back":  dict(lift=14.0, droop=0.8, strip=None, fringe=False),
    "centre_part": dict(lift=12.0, droop=1.0, strip=None, fringe=False),
    "side_part":   dict(lift=14.0, droop=1.0, strip=None, fringe=False),
    "spiky":       dict(lift=68.0, droop=0.10, strip=None, fringe=False),
    "mohawk":      dict(lift=84.0, droop=0.04, strip=0.20, fringe=False),
}


def _hash01(*key) -> np.ndarray:
    """Deterministic numbers in 0..1 from integer arrays (no random state: the same on every machine)."""
    x = np.zeros(np.broadcast(*[np.asarray(k) for k in key]).shape, dtype=np.uint64)
    for k in key:
        x = (x ^ (np.asarray(k).astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
                  + (x << np.uint64(6)) + (x >> np.uint64(2))))
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def head_landmarks(base, verts) -> dict:
    """The head's measure on a morphed body (``verts``: all of the base mesh's vertices)."""
    V = np.asarray(verts, dtype=np.float64)
    quads = np.asarray(base.quads)

    def joint(name):
        return V[np.unique(quads[base.groups["joint-" + name]])].mean(axis=0)

    bv, _ = base.body_cage(verts)
    h = SC.head_frame(bv, joint("head"), joint("head-2"), joint("l-eye"))
    h["neck"] = joint("neck")
    return h


def _flow(style: str, P, N, h) -> np.ndarray:
    """The comb direction at each root (unit, along the skin)."""
    x = P[:, 0]
    if style in ("natural", "spiky"):
        whorl = np.array([0.0, h["crown"] - 0.02 * h["span"], h["z0"] - 0.45 * h["half"]])
        d = P - whorl
        d[:, 1] -= 0.35 * np.linalg.norm(d, axis=1)             # (and a little down: a whorl is not a star)
    elif style == "swept_back":
        d = np.tile(np.array([0.0, -0.30, -1.0]), (len(P), 1))
        d[:, 0] = 0.25 * x / max(h["half"], 1e-9)               # fanning out a little towards the sides
    elif style in ("centre_part", "side_part"):
        part = 0.0 if style == "centre_part" else 0.42 * h["half"]
        s = np.where(x >= part, 1.0, -1.0)
        # to each side over the top and the front; the back of the head just falls (no parting at the nape)
        w = np.clip((P[:, 2] - (h["z0"] - 0.75 * h["half"])) / max(0.9 * h["half"], 1e-9), 0.0, 1.0)
        d = np.stack([s * w, -1.0 + 0.45 * w, -0.35 * np.ones(len(P))], axis=1)
    else:                                                        # mohawk: up and a little back
        d = np.tile(np.array([0.0, 0.2, -1.0]), (len(P), 1))
    return SC.tangent_field(N, d)


def build_hair(base, verts, *, style: str = "natural", length: float = 1.2, locks: int = 80,
               width: float = 0.30, thickness: float = 0.17, volume: float = 0.3, droop: float = 2.5,
               wave: float = 0.0, curl: float = 0.0, fringe: float = 0.8, seed: int = 0, cap: bool = True,
               lib=None, region: str = "scalp", tip_floor: float = 0.14, probe=None) -> dict:
    """The pieces of one head of hair on the body ``verts`` (all the base mesh's vertices, morphed).

    Returns ``{"pieces": [(name, Mesh)], "anchor": point, "roots": (N, 3), "paths": (N, S + 1, 3)}`` in the
    MakeHuman frame.  The pieces are closed shells, each sunk into the body at its root."""
    from .appendages import SkinProbe
    if style not in _STYLE:
        raise ValueError(f"unknown hair style {style!r}; the styles are {', '.join(STYLES)}")
    st = _STYLE[style]
    bv, bq = base.body_cage(verts)
    bv = np.asarray(bv, dtype=np.float64)
    h = head_landmarks(base, verts)
    weight = None
    if st["strip"] is not None:
        # a crest: the quads within a strip of the midline, measured on the neutral body
        nb = SC._body(base, lib)
        nh = nb.head()
        weight = (np.abs(nb.centroid[:, 0]) < st["strip"] * nh["half"]).astype(np.float64)
    samples = SC.emit(base, region, int(locks), seed=int(seed), lib=lib, weight=weight)
    vn = SC.vertex_normals(bv, bq)
    P, N = samples.positions(bv), samples.normals(vn)
    idx = np.arange(len(P))
    T = _flow(style, P, N, h)
    L = float(length) * (0.86 + 0.28 * _hash01(idx, 11 + int(seed)))
    if st["fringe"]:
        # forward-combed roots are cut where they reach the brow (times ``fringe``)
        az = np.degrees(np.arctan2(np.abs(P[:, 0]), P[:, 2] - h["z0"]))
        forward = (T[:, 2] > 0.25) & (az < 62.0)
        brow = h["eye"] + 0.16 * h["span"]
        reach = np.maximum(P[:, 1] - brow, 0.0) * 1.25 + 0.12 * h["span"]
        L = np.where(forward, np.minimum(L, np.maximum(float(fringe) * reach, 0.12)), L)
    if st["strip"] is not None:
        # a crest stands tallest over the crown and falls away to the forehead and the nape
        t = (P[:, 2] - (h["z0"] - h["half"])) / max(2.0 * h["half"], 1e-9)
        L = L * (0.55 + 0.45 * np.sin(np.pi * np.clip(t, 0.0, 1.0)) ** 0.7)
    growth = ST.Growth(segments=max(6, min(20, int(round(6 + 5.0 * float(length))))),
                       lift=st["lift"] + 22.0 * float(volume) * (st["lift"] < 60.0),
                       droop=float(droop) * st["droop"],
                       clearance=max(ST.UNDER * float(thickness) - SUNK, 0.0), rise=0.04 * float(volume),
                       wave=float(wave), wave_length=max(0.5, 0.9 + 0.5 * float(length) ** 0.5),
                       curl=float(curl), curl_pitch=0.55 + 2.2 * float(curl))
    if probe is None:
        reach = float(L.max()) if len(L) else 0.0
        centre = np.array([0.0, 0.5 * (h["eye"] + h["crown"]) - 0.45 * reach, h["z0"]])
        probe = SkinProbe(np.eye(4), bv, bq, centre, 1.6 * h["span"] + 0.75 * reach)
    standing = growth.lift > 45.0
    shape = ST.LockShape(width=float(width), thickness=float(thickness), tip_floor=float(tip_floor),
                         sink=max(0.05, 0.35 * float(thickness)), root=1.0 if standing else 0.55,
                         root_thick=1.0 if standing else 0.6)
    pieces = []
    paths = np.zeros((0, growth.segments + 1, 3))
    if len(P):
        stand = N
        if st["strip"] is not None:
            # a crest stands in the head's middle plane: the skin's normal with most of its sideways part
            # taken out (a root a little off the midline would otherwise lean out to its side)
            stand = N * np.array([0.2, 1.0, 1.0])
            stand = stand / np.maximum(np.linalg.norm(stand, axis=1, keepdims=True), 1e-12)
        paths, ups, height = ST.grow(P, stand, T, L, growth, probe, phase=_hash01(idx, 23 + int(seed)))
        wid = float(width) * (0.82 + 0.36 * _hash01(idx, 37 + int(seed)))
        if st["fringe"]:
            wid = np.where(L < 0.5 * float(length), wid * 0.85, wid)
        sink = np.full(len(P), shape.sink)
        if standing:
            sink = np.maximum(sink, STAND_SINK * L)             # (a standing lock's whole weld is its root)
        meshes = ST.lock_meshes(paths, ups, shape, width=wid, root_normals=N,
                                free=ST.free_of_skin(height, float(thickness)), sink=sink)
        pieces += [(f"lock{i:03d}", m) for i, m in enumerate(meshes)]
    if cap and st["strip"] is None:
        m = ST.cap_mesh(bv, bq, SC.region_mask(base, region, lib),
                        thickness=(0.5 + 0.9 * float(volume)) * float(thickness),
                        sink=max(0.04, 0.3 * float(thickness)))
        if m is not None:
            pieces.insert(0, ("cap", m))
    anchor = np.array([0.0, h["crown"], h["z0"]])
    return {"pieces": pieces, "anchor": anchor, "roots": P, "paths": paths, "landmarks": h}
