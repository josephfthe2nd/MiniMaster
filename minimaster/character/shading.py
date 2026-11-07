"""Skin-aware shading for the character renderer.

The base renderer multiplies an sRGB colour by ``ambient + k*max(N.L, 0)``.
That is wrong in three ways at once — it shades in sRGB space, it has no
specular, and it has no occlusion — and the result is the flat, chalky "unfired
clay" look. This module fixes the light transport while staying pure numpy:

* **Linear pipeline.** sRGB in -> shade in linear light -> tonemap ->
  sRGB out. Shading in gamma space is what crushes the midtones.
* **Wrapped diffuse with a scatter tint** approximates subsurface scattering:
  light bends around the terminator and reddens as it does, because skin is
  translucent and red light scatters furthest. This single term is most of
  what separates skin from plaster.
* **Fresnel-weighted specular** (Schlick, F0 = 0.028 for skin) over a
  normalised Blinn-Phong lobe, so grazing angles catch a sheen.
* **Cavity ambient occlusion** darkens creases — eyelids, nostrils, lips,
  the neck — which is where a face reads.
* **Per-material parameters**, so eyes are glossy and dark, teeth bright and
  hard, skin soft and translucent.

Shading is evaluated **per vertex** and interpolated across triangles: on a
13k-vertex mesh that is visually close to per-pixel and vastly cheaper in a
Python rasteriser.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace

import numpy as np


# --------------------------------------------------------------------------
# colour space


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def hex_to_linear(color: str) -> np.ndarray:
    h = color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"expected #rrggbb, got {color!r}")
    srgb = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0
    return srgb_to_linear(srgb)


def tonemap(rgb: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    """Filmic-ish shoulder (ACES approximation) — keeps highlights from
    clipping to flat white, which is another thing that reads as plastic."""
    x = np.asarray(rgb, dtype=np.float64) * exposure
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


# --------------------------------------------------------------------------
# materials


@dataclass
class Material:
    """A shading description. ``base_color`` is an sRGB hex string."""

    base_color: str = "#c08a6a"
    roughness: float = 0.55        # 0 = mirror, 1 = fully diffuse
    specular: float = 0.5          # scales F0
    f0: float = 0.04               # normal-incidence reflectance
    wrap: float = 0.0              # diffuse wrap (fake SSS); 0 = Lambert
    scatter: str | None = None     # terminator tint (sRGB hex), skin -> red
    ao_strength: float = 1.0
    rim: float = 0.0               # rim/backlight strength
    emissive: float = 0.0

    def linear_color(self) -> np.ndarray:
        return hex_to_linear(self.base_color)

    def scatter_color(self) -> np.ndarray:
        return hex_to_linear(self.scatter) if self.scatter else np.ones(3)


# Skin: soft, translucent, slightly glossy. F0 0.028 is the measured value for
# skin; the wrap + red scatter is what makes the terminator look alive.
SKIN = Material(base_color="#a87f68", roughness=0.62,
                specular=0.45, f0=0.028, wrap=0.42, scatter="#b06a55",
                ao_strength=1.0, rim=0.18)
SKIN_PALE = replace(SKIN, base_color="#e0b49a")
SKIN_TAN = replace(SKIN, base_color="#b57f55")
SKIN_DARK = replace(SKIN, base_color="#6b452e", scatter="#7a2418")

# The eye is the strongest realism cue: a wet, very glossy sphere. Low
# roughness gives the tight catchlight that makes a face look alive.
EYE = Material(base_color="#e8e4dc", roughness=0.08, specular=1.0, f0=0.05,
               wrap=0.0, ao_strength=1.4, rim=0.0)
IRIS = replace(EYE, base_color="#4a3520")
TEETH = Material(base_color="#e6e0d2", roughness=0.30, specular=0.7, f0=0.05,
                 ao_strength=1.2)
HAIR = Material(base_color="#2e2119", roughness=0.35, specular=0.6, f0=0.05,
                ao_strength=1.0, rim=0.35)
GENERIC = Material()


# --------------------------------------------------------------------------
# lights


@dataclass
class Light:
    direction: tuple[float, float, float]  # points FROM the surface TO the light
    color: str = "#ffffff"
    intensity: float = 1.0

    def vec(self) -> np.ndarray:
        d = np.asarray(self.direction, dtype=np.float64)
        n = np.linalg.norm(d)
        return d / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])

    def linear(self) -> np.ndarray:
        return hex_to_linear(self.color) * self.intensity


@dataclass
class LightRig:
    """A classic three-point portrait rig, in CAMERA space (+X right, +Y up,
    +Z toward the viewer) so it follows the turntable."""

    key: Light = field(default_factory=lambda: Light(
        (-0.45, 0.55, 0.70), "#fff2e2", 1.9))
    fill: Light = field(default_factory=lambda: Light(
        (0.75, 0.05, 0.45), "#c4d6f0", 0.42))
    rim: Light = field(default_factory=lambda: Light(
        (0.35, 0.35, -0.85), "#ffffff", 0.9))
    ambient: str = "#5a6172"
    ambient_intensity: float = 0.22

    def lights(self):
        return (self.key, self.fill, self.rim)

    def ambient_linear(self) -> np.ndarray:
        return hex_to_linear(self.ambient) * self.ambient_intensity


# --------------------------------------------------------------------------
# ambient occlusion


def _neighbour_sums(faces, n):
    """Directed edge endpoints (both ways) for scatter-add neighbourhoods."""
    a = faces
    src = np.concatenate([a[:, i] for i in range(a.shape[1])])
    dst = np.concatenate([a[:, (i + 1) % a.shape[1]] for i in range(a.shape[1])])
    return np.concatenate([src, dst]), np.concatenate([dst, src])


# The scatter-adds below use np.bincount, not np.add.at. bincount adds its
# weights one by one in index order, exactly like add.at, so the sums (and
# everything computed from them) are bit-for-bit what add.at gave; but add.at
# is ~4x slower before numpy 1.25 (the declared floor is 1.24) and bincount is
# not.


def _scatter_sum(idx, values, n):
    """``out[idx[k]] += values[k]`` for (m,) or (m, c) ``values``, in index order."""
    if values.ndim == 1:
        return np.bincount(idx, weights=values, minlength=n).astype(np.float64, copy=False)
    return np.stack([np.bincount(idx, weights=values[:, a], minlength=n)
                     for a in range(values.shape[1])], axis=1).astype(np.float64, copy=False)


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals (zero where a vertex has no area).

    The face normals are accumulated corner 0 for every face, then corner 1,
    then corner 2 (the order ``np.add.at`` was called in), so the result is
    identical to the add.at version it replaces.
    """
    f = faces
    fn = np.cross(verts[f[:, 1]] - verts[f[:, 0]], verts[f[:, 2]] - verts[f[:, 0]])
    out = _scatter_sum(f.T.reshape(-1), np.tile(fn, (3, 1)), len(verts))
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, n, out=np.zeros_like(out), where=n > 1e-12)


def _smooth(values, src, dst, cnt, iterations):
    v = np.asarray(values, dtype=np.float64).copy()
    for _ in range(int(iterations)):
        acc = np.bincount(src, weights=v[dst], minlength=len(v)).astype(np.float64, copy=False)
        v = 0.5 * v + 0.5 * (acc / cnt)
    return v


def smooth_vertex_scalar(values, faces, iterations: int = 2):
    """Average a per-vertex scalar over its neighbours a few times.

    Raw cavity AO follows the *tessellation* as much as the shape, which shows
    up as dirt streaks along edge loops. Smoothing keeps the real creases (they
    are many vertices wide) and removes the per-quad noise.
    """
    n = len(values)
    src, dst = _neighbour_sums(faces, n)
    cnt = np.maximum(np.bincount(src, minlength=n).astype(np.float64), 1.0)
    return _smooth(values, src, dst, cnt, iterations)


def cavity_term(verts: np.ndarray, faces: np.ndarray, normals: np.ndarray,
                radius_scale: float = 1.0, smooth: int = 2) -> np.ndarray:
    """The strength-independent part of :func:`cavity_ao`: per-vertex concavity
    in ``0..1`` (0 on flat and convex surface, towards 1 in a crease).

    It depends only on the mesh, not on the camera or the AO strength, so a
    cache can keep it (see :class:`ShadingCache`)."""
    n = len(verts)
    src, dst = _neighbour_sums(faces, n)
    cnt = np.maximum(np.bincount(src, minlength=n).astype(np.float64), 1.0)
    vd = verts[dst]
    mean_nb = _scatter_sum(src, vd, n) / cnt[:, None]
    delta = mean_nb - verts
    # normalise by each vertex's OWN neighbourhood size, not the global mean:
    # the mesh has wildly varying quad density (dense face, coarse limbs) and a
    # global scale turns that density variation into fake occlusion.
    edge = _scatter_sum(src, np.linalg.norm(vd - verts[src], axis=1), n)
    edge = np.maximum(edge / cnt, 1e-9) * radius_scale
    proj = np.einsum("ij,ij->i", delta, normals) / edge

    proj = _smooth(proj, src, dst, cnt, smooth)
    # proj > 0 -> concave (occluded). Map through a smooth curve to 0..1.
    occ = 1.0 / (1.0 + np.exp(-3.0 * proj))          # 0.5 at flat
    return np.clip((occ - 0.5) * 2.0, 0.0, 1.0)


def ao_from_term(term: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Per-vertex AO (1 = open, 0 = fully occluded) from a :func:`cavity_term`."""
    return np.clip(1.0 - strength * term, 0.0, 1.0)


def cavity_ao(verts: np.ndarray, faces: np.ndarray, normals: np.ndarray,
              strength: float = 1.0, radius_scale: float = 1.0,
              smooth: int = 2) -> np.ndarray:
    """Per-vertex cavity AO from local concavity.

    For each vertex, look at the mean offset to its edge-neighbours. If the
    neighbourhood sits *in front of* the tangent plane (positive projection on
    the normal) the vertex is in a crease and is occluded; if it sits behind,
    the vertex is on a ridge and is exposed. Normalising by the local edge
    length makes it scale-invariant.

    This is what darkens eyelid folds, nostrils, the lip line and the neck —
    exactly the features that make a face legible — for a couple of
    scatter-adds rather than a ray cast per vertex.
    """
    return ao_from_term(cavity_term(verts, faces, normals, radius_scale, smooth), strength)


# --------------------------------------------------------------------------
# view-independent terms, cached


def _digest(a: np.ndarray) -> bytes:
    a = np.ascontiguousarray(a)
    h = hashlib.blake2b(digest_size=16)
    h.update(repr((a.shape, a.dtype.str)).encode())
    if a.size:
        h.update(memoryview(a).cast("B"))
    return h.digest()


class ShadingCache:
    """LRU cache of the view-independent shading terms of a mesh: vertex normals
    and :func:`cavity_term`.

    Entries are keyed by the *content* of the vertex and face arrays (a 128-bit
    hash, about 1 ms per megabyte), not by object identity, so an edited mesh
    can never be served stale terms, while the body of a character whose wing
    was just dragged still hits: its AO is computed once, not once per frame.
    The cache is bounded by ``max_entries`` and ``max_bytes``; a posed wing
    produces a new key per pose and simply ages out. It is safe to share
    between a UI thread and a render worker (entries are inserted whole, so a
    cancelled render never leaves a half-built one).
    """

    def __init__(self, max_entries: int = 64, max_bytes: int = 96 << 20):
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self._items: OrderedDict = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.normal_builds = 0      # vertex-normal computations (cache misses)
        self.cavity_builds = 0      # cavity-term computations (cache misses)

    def __len__(self):
        return len(self._items)

    @property
    def nbytes(self) -> int:
        return self._bytes

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0

    def terms(self, verts: np.ndarray, faces: np.ndarray, ao: bool = True,
              radius_scale: float = 1.0, smooth: int = 2):
        """``(normals, cavity_term)`` of the mesh; the term is ``None`` if ``ao``
        is false. Do not modify the returned arrays."""
        key = (_digest(verts), _digest(faces))
        akey = (float(radius_scale), int(smooth))
        with self._lock:
            ent = self._items.get(key)
            if ent is not None:
                self._items.move_to_end(key)
        hit = ent is not None and (not ao or akey in ent["ao"])
        if ent is None:
            normals = vertex_normals(verts, faces)
            ent = {"n": normals, "ao": {}}
            with self._lock:
                self.normal_builds += 1
        if ao and akey not in ent["ao"]:
            term = cavity_term(verts, faces, ent["n"], radius_scale, smooth)
            with self._lock:
                self.cavity_builds += 1
            ent["ao"][akey] = term
        with self._lock:
            if hit:
                self.hits += 1
            old = self._items.get(key)
            if old is not None and old is not ent:
                self._bytes -= self._size(old)
            self._items[key] = ent
            self._items.move_to_end(key)
            self._bytes = sum(self._size(e) for e in self._items.values())
            while len(self._items) > 1 and (len(self._items) > self.max_entries
                                            or self._bytes > self.max_bytes):
                self._bytes -= self._size(self._items.popitem(last=False)[1])
        return ent["n"], (ent["ao"][akey] if ao else None)

    @staticmethod
    def _size(ent) -> int:
        return ent["n"].nbytes + sum(t.nbytes for t in ent["ao"].values())


DEFAULT_CACHE = ShadingCache()


# --------------------------------------------------------------------------
# the shader


def eye_vertex_colors(verts: np.ndarray, forward: np.ndarray,
                      sclera="#e9e5dd", iris="#5b7c8d", pupil="#0a0a0c",
                      iris_deg: float = 34.0, pupil_deg: float = 13.0):
    """Paint an eyeball sphere: white sclera, coloured iris, black pupil.

    The eyeball helper mesh is a plain sphere with no material of its own, so
    the iris is placed geometrically — by angle from the gaze direction about
    the eye's centre. Without this the eyes render as blank white balls, which
    is far worse than the dark sockets they replaced.
    """
    verts = np.asarray(verts, dtype=np.float64)
    c = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
    d = verts - c
    n = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    f = np.asarray(forward, dtype=np.float64)
    f = f / max(float(np.linalg.norm(f)), 1e-12)
    cosang = (d / n) @ f
    col = np.tile(hex_to_linear(sclera), (len(verts), 1))
    ci, cp = np.cos(np.radians(iris_deg)), np.cos(np.radians(pupil_deg))
    col[cosang >= ci] = hex_to_linear(iris)
    col[cosang >= cp] = hex_to_linear(pupil)
    return col


def shade_vertices(normals_cam: np.ndarray, view_dirs: np.ndarray,
                   material: Material, rig: LightRig,
                   ao: np.ndarray | None = None,
                   base_colors: np.ndarray | None = None) -> np.ndarray:
    """Linear-space radiance per vertex, (n, 3).

    ``normals_cam`` and ``view_dirs`` are unit vectors in camera space.
    ``base_colors`` optionally overrides the material albedo per vertex.
    """
    N = normals_cam
    V = view_dirs
    base = (material.linear_color() if base_colors is None
            else np.asarray(base_colors, dtype=np.float64))
    if base.ndim == 1:
        base = np.broadcast_to(base, (len(N), 3))
    scatter = material.scatter_color()
    occ = np.ones(len(N)) if ao is None else np.clip(ao, 0.0, 1.0)
    occ = 1.0 - material.ao_strength * (1.0 - occ)
    occ = np.clip(occ, 0.0, 1.0)

    # Blinn-Phong exponent from roughness (perceptual mapping)
    rough = float(np.clip(material.roughness, 0.02, 1.0))
    shininess = 2.0 / (rough ** 4 + 1e-6) - 2.0
    norm_spec = (shininess + 8.0) / (8.0 * np.pi)
    f0 = material.f0 * material.specular * 2.0

    # AO occludes AMBIENT fully and direct diffuse only partially (a crease
    # still sees the key light). It must NOT touch the specular lobe, and it
    # must not be applied twice — both were bugs in the first version.
    direct_occ = (0.55 + 0.45 * occ)[:, None]

    out = np.zeros_like(N)
    for light in rig.lights():
        L = light.vec()
        radiance = light.linear()
        ndl = N @ L

        # Wrapped diffuse: light leaks past the terminator (subsurface).
        # The (1+w)**2 divisor is what keeps it ENERGY CONSERVING — dividing
        # by (1+w) integrates to ~1.42x Lambert at w=0.42, silently
        # brightening every skin surface.
        w = float(np.clip(material.wrap, 0.0, 1.0))
        diff = np.clip((ndl + w) / ((1.0 + w) ** 2), 0.0, 1.0)
        if w > 0.0:
            # redden as we approach and pass the terminator
            t = np.clip(ndl, 0.0, 1.0)[:, None]
            tint = scatter + (1.0 - scatter) * t
        else:
            tint = 1.0
        out += diff[:, None] * base * tint * radiance[None, :] * direct_occ

        # Fresnel-weighted Blinn-Phong specular (no AO term here)
        H = L[None, :] + V
        H /= np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-9)
        ndh = np.clip(np.einsum("ij,ij->i", N, H), 0.0, 1.0)
        vdh = np.clip(np.einsum("ij,ij->i", V, H), 0.0, 1.0)
        fres = f0 + (1.0 - f0) * (1.0 - vdh) ** 5
        spec = norm_spec * ndh ** shininess * fres * np.clip(ndl, 0.0, 1.0)
        out += spec[:, None] * radiance[None, :]

    # ambient, occluded once
    out += rig.ambient_linear()[None, :] * base * occ[:, None]

    # Rim/backlight: must depend on the rim LIGHT, otherwise the silhouette
    # glows just as brightly on the unlit side, which reads as fake.
    if material.rim > 0.0:
        Lr = rig.rim.vec()
        facing_rim = np.clip(N @ Lr, 0.0, 1.0)
        ndv = np.clip(np.einsum("ij,ij->i", N, V), 0.0, 1.0)
        rim = (1.0 - ndv) ** 3 * material.rim * occ * facing_rim
        out += rim[:, None] * base * rig.rim.linear()[None, :]
    if material.emissive:
        out += base * material.emissive
    return out
