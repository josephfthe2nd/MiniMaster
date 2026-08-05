"""Load the CC0 MakeHuman base mesh + morph-target library.

This is the "realistic" path for Character Lab. Instead of a hand-authored
178-vertex cage, it stands on the MakeHuman Community base mesh — 13,380
vertices / 13,378 quads for the body proper, a closed all-quad 2-manifold with
professional edge flow (concentric loops around the eyes and mouth, proper
joint loops) — deformed by the 1,280 shipped morph targets.

Everything loaded here is **CC0 1.0** (public domain dedication):

* base mesh + assets: github.com/makehumancommunity/makehuman-assets
  (LICENSE.txt = CC0 1.0 Universal) and the base.obj shipped in the
  makehuman repo's data/3dobjs.
* morph targets: the ``targets.npz`` bundled in the ``makehuman`` PyPI
  distribution, whose embedded license record reads
  ``author=MakeHuman Team, license=CC0``.

Only MakeHuman's *Python source* is AGPL3; none of it is used or vendored
here — this module reads the data formats directly.

A target is stored sparsely as (uint16 vertex indices, int16 deltas); the
deltas are in 1/1000 of a MakeHuman unit, so the applied offset is
``delta * scale / 1000``. Morphs sum linearly, which is exactly the
"one slider drives many vertices" model every commercial creator uses.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..core.mesh import Mesh

# Where the CC0 data lives. Override with MINIMASTER_MH_BASE / _MH_TARGETS,
# else look in a few conventional spots.
_BASE_CANDIDATES = [
    "/workspace/makehumancommunity/makehuman/makehuman/data/3dobjs/base.obj",
    "assets/makehuman/base.obj",
]
_TARGET_CANDIDATES = [
    "assets/makehuman/targets.npz",
]

TARGET_SCALE = 1.0 / 1000.0  # int16 deltas are in milli-units


class AssetsMissing(RuntimeError):
    """The CC0 MakeHuman data could not be located."""


def _find(env: str, candidates) -> Path | None:
    val = os.environ.get(env)
    if val and Path(val).exists():
        return Path(val)
    for c in candidates:
        p = Path(c)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[2] / c
        if p.exists():
            return p
    return None


def base_path() -> Path:
    p = _find("MINIMASTER_MH_BASE", _BASE_CANDIDATES)
    if p is None:
        raise AssetsMissing(
            "MakeHuman base.obj not found. Set MINIMASTER_MH_BASE, or place it at "
            "assets/makehuman/base.obj (CC0, from the makehuman repo's "
            "data/3dobjs/base.obj).")
    return p


def targets_path() -> Path:
    p = _find("MINIMASTER_MH_TARGETS", _TARGET_CANDIDATES)
    if p is None:
        raise AssetsMissing(
            "MakeHuman targets.npz not found. Set MINIMASTER_MH_TARGETS, or place "
            "it at assets/makehuman/targets.npz (CC0; ships inside the 'makehuman' "
            "PyPI distribution at makehuman/data/targets.npz).")
    return p


def available() -> bool:
    try:
        base_path()
        targets_path()
    except AssetsMissing:
        return False
    return True


# --------------------------------------------------------------------------
# base mesh


def load_base_obj(path=None):
    """Parse base.obj -> (verts (n,3), quads (m,4), groups {name: face idx}).

    Only ``v``/``f``/``g`` are needed; the file is all-quad.
    """
    path = Path(path) if path is not None else base_path()
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    groups: dict[str, list[int]] = {}
    current = ""
    with open(path) as fh:
        for line in fh:
            if not line or line[0] not in "vgf":
                continue
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "v":
                verts.append([float(x) for x in parts[1:4]])
            elif tag == "g":
                current = parts[1] if len(parts) > 1 else ""
            elif tag == "f":
                idx = [int(t.split("/")[0]) - 1 for t in parts[1:]]
                if len(idx) != 4:
                    continue  # base mesh is all-quad; ignore anything else
                faces.append(idx)
                groups.setdefault(current, []).append(len(faces) - 1)
    return (np.asarray(verts, dtype=np.float64),
            np.asarray(faces, dtype=np.int64),
            {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()})


class BaseMesh:
    """The MakeHuman base mesh, with the body extracted as a closed cage.

    ``verts`` are all 19,158 control vertices (body + helper geometry) so that
    target indices apply directly; ``body_quads`` indexes the 13,378 quads of
    the ``body`` group, remapped onto the 13,380 vertices it actually uses.
    """

    def __init__(self, path=None):
        self.verts, self.quads, self.groups = load_base_obj(path)
        body = self.groups["body"]
        bq = self.quads[body]
        self.body_verts_idx = np.unique(bq)
        remap = -np.ones(len(self.verts), dtype=np.int64)
        remap[self.body_verts_idx] = np.arange(len(self.body_verts_idx))
        self.body_quads = remap[bq]

    def body_cage(self, verts=None) -> tuple[np.ndarray, np.ndarray]:
        """(verts, quads) for the closed body cage, optionally from morphed
        control vertices."""
        v = self.verts if verts is None else verts
        return v[self.body_verts_idx], self.body_quads

    def helper_group(self, name: str, verts=None):
        """A helper island (e.g. ``helper-l-eye``) as (verts, quads)."""
        if name not in self.groups:
            raise KeyError(f"no group {name!r}; have {len(self.groups)} groups")
        v = self.verts if verts is None else verts
        q = self.quads[self.groups[name]]
        used = np.unique(q)
        remap = -np.ones(len(v), dtype=np.int64)
        remap[used] = np.arange(len(used))
        return v[used], remap[q]


# --------------------------------------------------------------------------
# morph targets


class TargetLibrary:
    """The 1,280 CC0 morph targets, loaded lazily from ``targets.npz``."""

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else targets_path()
        self._z = np.load(self.path, allow_pickle=True)
        self.names = sorted(
            k[len("targets/"):-len(".index")]
            for k in self._z.files if k.endswith(".index"))
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: str) -> bool:
        return f"targets/{name}.index" in self._z

    def categories(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for n in self.names:
            out[n.split("/")[0]] = out.get(n.split("/")[0], 0) + 1
        return out

    def find(self, *fragments: str) -> list[str]:
        """Target names containing every fragment (case-insensitive)."""
        frags = [f.lower() for f in fragments]
        return [n for n in self.names if all(f in n.lower() for f in frags)]

    def get(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(indices, deltas) for one target; deltas already scaled to units."""
        if name not in self._cache:
            key = f"targets/{name}"
            idx = np.asarray(self._z[key + ".index"], dtype=np.int64)
            vec = np.asarray(self._z[key + ".vector"], dtype=np.float64)
            self._cache[name] = (idx, vec * TARGET_SCALE)
        return self._cache[name]

    def apply(self, verts: np.ndarray, weights: dict[str, float],
              out: np.ndarray | None = None, strict: bool = False) -> np.ndarray:
        """verts + sum(w * target) — the linear morph blend.

        A name with no stored target contributes nothing: MakeHuman omits the
        file for any combination that equals the neutral base (e.g. the
        'averageheight' corner), so a missing target legitimately means a zero
        delta. Pass ``strict=True`` to raise on unknown names instead.
        """
        res = verts.copy() if out is None else out
        for name, w in weights.items():
            if not w:
                continue
            if name not in self:
                if strict:
                    raise KeyError(f"unknown morph target {name!r}")
                continue
            idx, vec = self.get(name)
            res[idx] += vec * float(w)
        return res


# --------------------------------------------------------------------------
# convenience


def load(base=None, targets=None) -> tuple[BaseMesh, TargetLibrary]:
    return BaseMesh(base), TargetLibrary(targets)


# --------------------------------------------------------------------------
# the macro system: continuous sliders -> a blend over the target hypercube
#
# MakeHuman stores its "macro" morphs as the CORNERS of a grid — every
# combination of ethnicity x gender x age (and separately gender x age x
# muscle x weight, and again x height). A continuous slider is realised by
# interpolating between the two corners that bracket it, and the total weight
# of any axis sums to 1, so the whole blend is a product of per-axis weights.

AGE_NODES = ("baby", "child", "young", "old")
# MakeHuman's age slider is piecewise: 0=baby, 0.1875=child, 0.5=young, 1=old
AGE_POS = (0.0, 0.1875, 0.5, 1.0)
LEVEL_NODES = ("min", "average", "max")
ETHNICITIES = ("african", "asian", "caucasian")


def _axis_weights(value: float, nodes, positions=None) -> list[tuple[str, float]]:
    """Interpolate ``value`` across ordered nodes -> [(node, weight)] summing 1."""
    pos = list(positions) if positions is not None else list(
        np.linspace(0.0, 1.0, len(nodes)))
    v = float(np.clip(value, pos[0], pos[-1]))
    for i in range(len(pos) - 1):
        lo, hi = pos[i], pos[i + 1]
        if lo <= v <= hi:
            t = 0.0 if hi == lo else (v - lo) / (hi - lo)
            out = []
            if 1.0 - t > 1e-9:
                out.append((nodes[i], 1.0 - t))
            if t > 1e-9:
                out.append((nodes[i + 1], t))
            return out
    return [(nodes[-1], 1.0)]


def macro_weights(gender=0.5, age=0.5, muscle=0.5, weight=0.5, height=0.5,
                  african=0.0, asian=0.0, caucasian=1.0) -> dict[str, float]:
    """Continuous macro sliders (all 0..1) -> target weights.

    ``gender`` 0=female 1=male; ``age`` 0=baby .1875=child .5=young 1=old;
    ``muscle``/``weight``/``height`` 0=min .5=average 1=max; the three
    ethnicity values are normalised to sum to 1.
    """
    eth = np.array([african, asian, caucasian], dtype=np.float64)
    if eth.sum() <= 0:
        eth = np.array([0.0, 0.0, 1.0])
    eth = eth / eth.sum()
    g = _axis_weights(gender, ("female", "male"))
    a = _axis_weights(age, AGE_NODES, AGE_POS)
    m = _axis_weights(muscle, LEVEL_NODES)
    w = _axis_weights(weight, LEVEL_NODES)
    h = _axis_weights(height, LEVEL_NODES)

    out: dict[str, float] = {}

    def add(name, val):
        if val > 1e-9:
            out[name] = out.get(name, 0.0) + val

    # ethnicity x gender x age  (the base body shape)
    for ei, ename in enumerate(ETHNICITIES):
        if eth[ei] <= 1e-9:
            continue
        for gname, gw in g:
            for aname, aw in a:
                add(f"macrodetails/{ename}-{gname}-{aname}", eth[ei] * gw * aw)
    # gender x age x muscle x weight  (universal build)
    for gname, gw in g:
        for aname, aw in a:
            for mname, mw in m:
                for wname, ww in w:
                    add(f"macrodetails/universal-{gname}-{aname}-"
                        f"{mname}muscle-{wname}weight", gw * aw * mw * ww)
    # ... x height
    for gname, gw in g:
        for aname, aw in a:
            for mname, mw in m:
                for wname, ww in w:
                    for hname, hw in h:
                        add(f"macrodetails/height/{gname}-{aname}-{mname}muscle-"
                            f"{wname}weight-{hname}height",
                            gw * aw * mw * ww * hw)
    return out


def paired_weights(name_decr: str, name_incr: str, value: float) -> dict[str, float]:
    """A -1..+1 slider across a decr/incr target pair."""
    v = float(np.clip(value, -1.0, 1.0))
    if v >= 0:
        return {name_incr: v} if v > 1e-9 else {}
    return {name_decr: -v}


def body_mesh(base: BaseMesh, verts=None, levels: int = 0) -> Mesh:
    """The morphed body as a triangle :class:`Mesh`, optionally subdivided."""
    from ..core.subdiv import quads_to_mesh, subdivide_quads
    v, q = base.body_cage(verts)
    if levels:
        v, q = subdivide_quads(v, q, levels)
    return quads_to_mesh(v, q)
