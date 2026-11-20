"""A particle system: points with a frame each, where they come from, and what is put on them.

A PARTICLE is a place and a way round: a position, a frame (which way it heads, which way is up) and a size.
A system of them is three arrays (``Particles``), so whatever is done to particles is done to all at once.
Nothing here knows about bodies or wings: an emitter takes a bare polyline or a bare ``Mesh``.

``emit_curve``    particles along a polyline, heading along it (barbs along a feather's shaft, spikes down
                  a spine, feathers along an arm)
``emit_surface``  particles spread evenly over a mesh, each bound to the triangle it sits on, so the system
                  can ``follow`` the mesh when it moves (hair roots, scales, a feather tract)
``jitter``        repeatable variation of aim and size
``instance``      one template mesh on every particle, realised in ONE pass: the template's vertices times N
                  frames, its triangles N times with an offset.  A thousand feathers are one template and a
                  thousand frames until the moment they are needed as triangles.

A strand of a hair system (``core/hair.py``) is what a particle leaves behind when it moves: ``hair.grow``
takes a particle system's positions and headings as its roots.

Emission is static (the particles are placed, not simulated over time), and repeatable: no random state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hair import _unit, hash01
from .mesh import Mesh

__all__ = ["Particles", "emit_curve", "emit_surface", "jitter", "instance", "halton"]


@dataclass
class Particles:
    """``N`` particles: ``position`` (N, 3), ``frame`` (N, 3, 3) whose columns are the heading (+X of what
    is put there), up (+Y) and side (+Z), ``size`` (N,).  ``attrs`` holds anything else per particle (the
    triangle and the weights a surface particle is bound by, where along its curve a curve particle is)."""

    position: np.ndarray
    frame: np.ndarray
    size: np.ndarray
    attrs: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.position)

    @property
    def heading(self) -> np.ndarray:
        return self.frame[:, :, 0]

    @property
    def up(self) -> np.ndarray:
        return self.frame[:, :, 1]

    def take(self, index) -> "Particles":
        return Particles(self.position[index], self.frame[index], self.size[index],
                         {k: np.asarray(v)[index] for k, v in self.attrs.items()})

    def matrices(self) -> np.ndarray:
        """``(N, 4, 4)``: the transform of each particle (its frame scaled by its size, then its position)."""
        m = np.tile(np.eye(4), (len(self), 1, 1))
        m[:, :3, :3] = self.frame * self.size[:, None, None]
        m[:, :3, 3] = self.position
        return m

    def follow(self, mesh: Mesh, flow=None) -> "Particles":
        """The same particles on ``mesh``, a moved copy of the mesh they were emitted on (the same
        triangles): each where its triangle and weights put it now, up along the surface there."""
        tri = np.asarray(mesh.vertices)[np.asarray(mesh.faces)[self.attrs["face"]]]
        pos = (tri * self.attrs["bary"][:, :, None]).sum(axis=1)
        up = _unit(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))
        head = self.heading if flow is None else np.broadcast_to(np.asarray(flow, dtype=np.float64), up.shape)
        return Particles(pos, _frames(up, head), self.size.copy(), dict(self.attrs))


def _frames(up, heading) -> np.ndarray:
    """Frames (N, 3, 3) whose +Y is ``up`` and whose +X is ``heading`` laid into the plane across ``up``
    (any direction in it where ``heading`` stands along ``up``)."""
    up = _unit(np.asarray(up, dtype=np.float64))
    h = np.asarray(heading, dtype=np.float64)
    x = h - (h * up).sum(axis=1, keepdims=True) * up
    bad = np.linalg.norm(x, axis=1) < 1e-9
    if bad.any():
        x = x.copy()
        x[bad] = np.cross(up[bad], np.eye(3)[np.argmin(np.abs(up[bad]), axis=1)])
    x = _unit(x)
    return np.stack([x, up, np.cross(x, up)], axis=2)


def halton(n: int, bases=(2, 3, 5), skip: int = 0) -> np.ndarray:
    """``n`` points of the Halton sequence in the unit cube: even, and the same on every run."""
    out = np.empty((n, len(bases)))
    idx = np.arange(1 + skip, n + 1 + skip)
    for k, b in enumerate(bases):
        r, f, i = np.zeros(n), 1.0 / b, idx.copy()
        while i.max() > 0:
            r += f * (i % b)
            i //= b
            f /= b
        out[:, k] = r
    return out


def emit_curve(curve, count: int | None = None, at=None, up=(0.0, 1.0, 0.0), size=1.0) -> Particles:
    """Particles along a polyline ``curve`` (M, 3): ``count`` of them evenly by length, or at the fractions
    ``at`` of its length.  Each heads along the curve; ``up`` is made square to that.  ``attrs["at"]`` is
    where along the curve each one is (0..1)."""
    P = np.asarray(curve, dtype=np.float64)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    at = (np.arange(int(count)) + 0.5) / int(count) if at is None else np.asarray(at, dtype=np.float64)
    d = at * s[-1]
    k = np.clip(np.searchsorted(s, d, side="right") - 1, 0, len(seg) - 1)
    f = ((d - s[k]) / np.maximum(seg[k], 1e-300))[:, None]
    pos = P[k] * (1.0 - f) + P[k + 1] * f
    tan = _unit(P[k + 1] - P[k])
    u = np.broadcast_to(np.asarray(up, dtype=np.float64), tan.shape)
    u = u - (u * tan).sum(axis=1, keepdims=True) * tan
    bad = np.linalg.norm(u, axis=1) < 1e-9
    if bad.any():
        u = u.copy()
        u[bad] = np.cross(tan[bad], np.eye(3)[np.argmin(np.abs(tan[bad]), axis=1)])
    u = _unit(u)
    frame = np.stack([tan, u, np.cross(tan, u)], axis=2)
    return Particles(pos, frame, np.broadcast_to(np.asarray(size, dtype=np.float64), (len(pos),)).copy(), {"at": at})


def emit_surface(mesh: Mesh, count: int, seed: int = 0, density=None, flow=(0.0, -1.0, 0.0), size=1.0,
                 oversample: int = 24) -> Particles:
    """``count`` particles spread evenly over ``mesh``: up along the surface's normal, heading along ``flow``
    laid into the surface (one direction, or a function of the positions).

    Candidates are drawn area for area from a Halton sequence and chosen farthest-first, so no two crowd one
    another, the first ``n`` of them are an even cover on their own, and the same ``seed`` gives the same
    particles everywhere.  ``density`` (a number per triangle) thins or thickens parts of the surface; a
    triangle whose density is 0 gets none.  ``attrs["face"]`` and ``attrs["bary"]`` bind each particle to
    its triangle (see :meth:`Particles.follow`)."""
    V, F = np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces)
    tri = V[F]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    if density is not None:
        area = area * np.asarray(density, dtype=np.float64)
    live = np.nonzero(area > 0.0)[0]
    count = int(count)
    if count <= 0 or not len(live):
        z = np.zeros((0, 3))
        return Particles(z, np.zeros((0, 3, 3)), np.zeros(0), {"face": np.zeros(0, dtype=np.int64), "bary": z})
    cdf = np.cumsum(area[live])
    n_c = max(oversample * count, 400)
    h = halton(n_c, skip=97 * int(seed))
    face = live[np.minimum(np.searchsorted(cdf, h[:, 0] * cdf[-1], side="right"), len(live) - 1)]
    r1 = np.sqrt(h[:, 1])
    bary = np.stack([1.0 - r1, r1 * (1.0 - h[:, 2]), r1 * h[:, 2]], axis=1)       # even over a triangle
    pos = (tri[face] * bary[:, :, None]).sum(axis=1)
    d = np.full(n_c, np.inf)
    pick = np.empty(min(count, n_c), dtype=np.int64)
    k = 0
    for i in range(len(pick)):                       # farthest-first
        pick[i] = k
        d = np.minimum(d, np.linalg.norm(pos - pos[k], axis=1))
        d[k] = -1.0
        k = int(np.argmax(d))
    pos, face, bary = pos[pick], face[pick], bary[pick]
    up = _unit(nrm[face])
    head = flow(pos) if callable(flow) else np.broadcast_to(np.asarray(flow, dtype=np.float64), up.shape)
    return Particles(pos, _frames(up, head), np.broadcast_to(np.asarray(size, dtype=np.float64), (len(pos),)).copy(),
                     {"face": face, "bary": bary})


def jitter(particles: Particles, seed: int = 0, yaw: float = 0.0, tilt: float = 0.0, size: float = 0.0) -> Particles:
    """Repeatable variation: each particle turned up to ``yaw`` degrees either way about its up, tipped up
    to ``tilt`` degrees about its side, and scaled by up to ``size`` (a fraction) either way."""
    n = len(particles)
    i = np.arange(n)
    X, Y, Z = (particles.frame[:, :, k] for k in range(3))
    a = np.radians(yaw) * (2.0 * hash01(i, seed, 11) - 1.0)[:, None]
    X, Z = np.cos(a) * X + np.sin(a) * Z, -np.sin(a) * X + np.cos(a) * Z
    b = np.radians(tilt) * (2.0 * hash01(i, seed, 12) - 1.0)[:, None]
    X, Y = np.cos(b) * X + np.sin(b) * Y, -np.sin(b) * X + np.cos(b) * Y
    s = particles.size * (1.0 + size * (2.0 * hash01(i, seed, 13) - 1.0))
    return Particles(particles.position.copy(), np.stack([X, Y, Z], axis=2), s, dict(particles.attrs))


def instance(template: Mesh, particles: Particles, merge: bool = True):
    """``template`` on every particle: its +X along the particle's heading, +Y up, scaled by its size.

    One pass over all of them: ``(N, V, 3)`` vertices from one matrix product, the template's triangles
    repeated with an offset.  ``merge``: one mesh of everything; else a list of ``N`` (each a slice of the
    same array).  A particle whose frame is left-handed (a mirrored one) gets the mirror image, its
    triangles turned over."""
    V = np.asarray(template.vertices, dtype=np.float64)
    F = np.asarray(template.faces)
    n = len(particles)
    W = (V[None, :, :] * particles.size[:, None, None]) @ np.transpose(particles.frame, (0, 2, 1)) \
        + particles.position[:, None, :]
    mirrored = np.linalg.det(particles.frame) < 0 if n else np.zeros(0, dtype=bool)
    if not merge:
        flipped = F[:, ::-1].copy()
        return [Mesh(W[i], flipped if mirrored[i] else F) for i in range(n)]
    faces = np.broadcast_to(F[None], (n,) + F.shape).copy()
    faces[mirrored] = faces[mirrored][:, :, ::-1]
    faces += (len(V) * np.arange(n))[:, None, None]
    return Mesh(W.reshape(-1, 3), faces.reshape(-1, 3))
