# The portrait renderer: rasteriser, caches, picking, draft tier

`render_portrait` (the CLI, every figure in the docs, and the interactive preview
the appendage design asks for) used to rasterise with a Python loop over
triangles, one `np.meshgrid` per triangle. That loop, not the shading, was the
cost: 95,000 `meshgrid` calls for a feathered wing pair. This note records what
replaced it, how equality with the old output is guarded, what was measured, and
the building blocks for the draft tier. Design context:
[appendages_system_design.md](appendages_system_design.md) (D7, track T2). The consumer of the draft tier,
the picking and the cancellation is the Character tab ([character_tab.md](character_tab.md)).

| Module | Contents |
| --- | --- |
| `minimaster/character/raster.py` | the vectorised visibility-buffer rasteriser, `Camera`, culling, deferred Gouraud, `Cancelled` / `CancelToken` (numpy only, no tkinter) |
| `minimaster/character/portrait.py` | `render_portrait` (same signature plus `check=`, `cache=`), `visibility_buffer`, `pick` |
| `minimaster/character/shading.py` | `bincount` normals and cavity AO, `ShadingCache` |
| `minimaster/character/draft.py` | `ClusterProxy`, `BodyProxy`, `render_draft` (no tkinter, does not touch `viewport.py`) |
| `tests/test_raster.py`, `tests/test_draft.py` | the golden arrangement and everything below |
| `tools/raster_bench.py` | the timings in this note |

## How the rasteriser works

1. **Setup.** Back faces and near-plane triangles are culled with the old loop's
   own test (spelled identically, so borderline triangles are decided
   identically); survivors get an integer pixel bounding box (pixel centres at
   `i + 0.5`, plus a 1e-6 px guard band).
2. **Bucket.** Triangles are grouped by bounding-box size class (1, 2, 3, 4, 6, 8,
   12 ... 128 px). A class is rasterised on a fixed `B x B` candidate grid per
   triangle, and a whole chunk (about 1.5 M candidates) is evaluated at once with
   the *same barycentric arithmetic as the loop*. A triangle wider than 128 px
   takes a per-triangle path (the loop's method): for those, the call overhead is
   small next to the pixels covered, and a zoomed-in view only has a few.
3. **Resolve.** Fragments fold into a visibility buffer, `(depth, triangle id)`
   per pixel, by `np.minimum.at`. The rule is *nearest depth, then lowest triangle
   id*: that is exactly the old draw order with its strict `<` (shells in order,
   triangles in index order), and it makes the result independent of bucketing and
   chunking. (An earlier spike resolved ties by "last write wins", which is
   undefined for repeated indices.)
4. **Shade.** Colour is Gouraud-interpolated for the *visible* pixels only, from
   the per-vertex colours of `shading.shade_vertices`, with the loop's expressions
   in the loop's order.

Everything is float64 and reproduces the old output **bit for bit**.
`rasterize(..., dtype=np.float32)` exists for draft rendering and is not exact.

### Equality arrangement (the golden test)

`tests/test_raster.py` keeps the pre-change renderer verbatim as
`_reference_render_portrait` (the per-triangle loop *and* the `np.add.at` normals
and cavity AO it used), and `_reference_ids` (the loop cut down to the visibility
buffer). Against them:

* **24 golden renders** on real scenes, `test_golden_scenes_...`: bat spread,
  bat fold 0.7, angel fold 1.0 and angel raise 35 (the angel at reduced feather
  detail so the slow reference stays affordable; the bat-spread scene includes the
  body and its eyes), each at 4 azimuths (0, 90, 200, 315) and 2 zoomed-in fits, at three
  image sizes. RGB **and alpha** must be equal in every pixel (about 12 s).
* **The same 24 on the full scenes** (`MM_GOLDEN_FULL=1`, about 1 min): body +
  two full-detail wings (43k / 198k triangles) at 320x480, 400x300 and 500x500;
  4,188,800 pixels, 0 differing.
* **Adversarial soups** attack the rasteriser itself: vertices and edges exactly
  on pixel centres, exact depth ties, all-coplanar soups, triangles hundreds of
  pixels wide and entirely off-screen, depths beyond 1e12; the id and depth
  buffers must be equal to the loop's, and independent of chunk size.
* **Extreme zoom** on the body until single quads exceed 128 px (the per-triangle
  path), coincident shells (every pixel a depth tie: the first shell must win),
  transparent background, empty and zero-area input.
* Bit-identity of the `bincount` normals and AO against the `add.at` versions
  (below), on a sphere, a cube cage, a lumpy mesh, an empty mesh and the body.

The suite also passes unchanged on **numpy 1.24.4 / Python 3.11**, and the two
bat/angel renders there equal the old loop's output on numpy 2.5.3 pixel for
pixel.

## Speed

320x480, body (`levels=0`, 26,756 triangles) + two wings, median of 7, Python 3.13 /
numpy 2.5.3, 4 cores, other jobs running on the machine (`tools/raster_bench.py`):

| scene | triangles | before | after, cold | after, warm | speedup (cold / warm) |
| --- | --- | --- | --- | --- | --- |
| bat | 43,584 | 1.24 s | 0.065 s | 0.036 s | 19x / 34x |
| angel | 198,284 | 4.29 s | 0.28 s | 0.14 s | 15x / 30x |

*Cold* is the first frame, with the shading cache empty (every shell's normals and
cavity AO computed); *warm* is a camera move on the same shells (those terms are
cached). Re-posing one wing and rendering again ("after a wing edit") costs about
0.05 s (bat) and 0.21 s (angel): only the edited shell's terms are recomputed.
At 640x960 the picture is the same (the cost is mostly per shell and per
triangle, not per pixel). `tools/wing_demo.py` (all the wing figures, 440x640
panels on a `levels=1` body) went from 90 s to 6.4 s, and the three figures it writes are byte-identical to the
committed ones (and to a run of the old code).

## Shading: caches and numpy portability

* **`ShadingCache`** keeps the view-independent terms, vertex normals and the
  strength-independent part of the cavity AO (`cavity_term`; the AO strength is
  applied afterwards, so shells with different `ao_strength` share an entry). It
  is keyed by the *content* of the vertex and face arrays (128-bit blake2b, about
  1 ms per MB), not by object identity: editing a mesh in place can never be
  served stale terms, and the body of a character whose wing was just edited hits
  the cache, so its AO is computed once. LRU, bounded in entries (64) and bytes
  (96 MB), thread-safe (a render worker and the UI thread may share it; entries
  are inserted whole, so a cancelled render never leaves a half-built one).
  `render_portrait(cache=None)` uses the shared `shading.DEFAULT_CACHE`,
  `cache=False` bypasses it.
* **`bincount` instead of `np.add.at`** in the vertex normals, `cavity_ao` and
  `smooth_vertex_scalar`. `np.add.at` is about 4x slower before numpy 1.25 (the
  declared floor is 1.24). `bincount` adds its weights in index order exactly as
  `add.at` did, and the normals accumulate corner 0 of every face, then corner 1,
  then corner 2 (the old call order), so the outputs are **bit-identical**
  (`np.array_equal`, max diff 0.0, tested against the `add.at` originals). Public
  functions keep their behaviour; `cavity_ao` is now `cavity_term` + strength.
* **numpy floor: kept at >= 1.24.** The one remaining `ufunc.at` is the z-resolve
  (`np.minimum.at`); it is slower on 1.24, but on numpy 1.24.4 the whole render
  measures the same as on 2.5.3 (bat 0.07 s, angel 0.27 s cold, 0.14 s warm), so
  neither a version check nor a sort-based fallback is worth its code, and
  `pyproject.toml` is unchanged.

## Cancellation

`render_portrait(..., check=fn)` (and `rasterize`, `visibility_buffer`,
`render_draft`) calls `fn()` with no arguments before each shell is shaded and
before each chunk of the rasteriser (about 10-20 calls for a character, never per
triangle or per pixel). Whatever `fn` raises propagates and no image is returned;
raise `portrait.Cancelled`. `portrait.CancelToken` is a ready-made thread-safe
`check`:

```python
token = CancelToken()
worker:   img = render_portrait(shells, size=(900, 800), check=token)   # raises Cancelled
UI:       token.cancel()          # on the next edit; the worker stops at its next checkpoint
```

The worst-case latency is the longest uninterruptible numpy call between two
checkpoints (a wing's cavity AO, about 60 ms for a feathered wing on a cold
cache); the draft tier does not need it.

## Picking

`visibility_buffer(shells, size, azimuth, elevation, fov, fit)` rasterises only
the ids (no shading; about 5 ms for the body at 320x480) with the camera
`render_portrait` uses for the same arguments, and returns a `VisBuffer`:
`shell` / `face` / `depth` arrays and `pick(x, y)` -> `(shell index, triangle
index)` or `None` (shell indices refer to the list you passed, empty shells keep
their slot), `pick_many` for arrays, `world_point(x, y)` for the 3-D position of
the surface under a pixel (for `SurfaceAnchor.from_point`). `pick(shells, size,
azimuth, elevation, fit, x, y)` is the one-call form. Pixels are columns/rows of the
image, origin top left, sampled at their centres. On the body the id buffer agrees
with `raycast.ray_mesh_hit` through the same pixel centre (note `screen_ray` has no
+0.5 offset) on the same face, or on one sharing a vertex.

## Draft tier building blocks

For orbiting, slider drags and pose drags the full meshes are too heavy even for
the new rasteriser's *shading* path; the preview draws proxies and refines on idle.

* **`ClusterProxy`** decimates a mesh by vertex clustering: vertices in the same
  grid cell merge. The clustering is computed **once from the rest mesh**
  (`by_cell`, `by_target(n_triangles)`, `by_pieces`); after that the proxy of any
  deformed version is `proxy.update(vertices)`, a `bincount` average, with no
  decimation per frame. `by_pieces` clusters every piece (feather, membrane, bone)
  on its own grid in one pass: a single `np.unique` over `(piece, cell)` keys, so
  the 542 pieces of a feathered wing cost the same as one (15-18 ms to build,
  under 1 ms to update). `pieces_from_ranges(rig.ranges...)` gives the piece ids
  of a wing rig. `proxy.rep` is one source vertex per proxy vertex, for callers
  that can deform only those (skinning). Nothing knows about wings; posing is the
  caller's job.
* **`BodyProxy(base, 4000)`** is the body cage as 4,029 triangles, built once from
  the base mesh (28 ms); `mesh(verts)` / `points(verts)` take the same morphed
  control vertices `character_shells` takes and return the Z-up proxy in 0.2 ms.
  Silhouette IoU against the full body over five views (az/el 0/0, 90/0, 180/10,
  35/20, 270/5): 0.957-0.976 (worst 0.957) for the neutral body, a tall heavy man
  and a child.
* **`render_draft(shells, ..., shading="flat"|"smooth"|[per shell])`** is the same
  camera, lights, materials and output as `render_portrait`, minus cavity AO, with
  a float32 rasteriser; flat shading is a per-triangle colour table (the right
  choice for clustered thin parts, whose averaged vertex normals shade badly),
  smooth is Gouraud. Per-vertex albedo overrides are averaged per face.

Measured at 320x480 (`tools/raster_bench.py`, orbit frames, body proxy smooth +
wings flat, wing posing not included): body + two bat-wing proxies (6.3k triangles)
**11 ms = 90 fps**; body + two angel-wing proxies built on the draft rig
(`coverts=False, underwing=False, bones=False`; 10k triangles) **13 ms = 76 fps**;
the full feathered rig clustered at `detail=5, min_cells=2, max_cells=10` (17.5k
triangles) 20 ms = 49 fps (below the 60 fps bar: use the draft rig or a coarser
`detail`). The default detail on the full feathered rig is 46k triangles, too heavy
(about 21 fps). The proxies are drafts: bat-wing
silhouette IoU with the full posed wing is 0.80-0.83 spread and 0.59-0.70 at fold
0.8 (no membrane pleats). Pose time (`PlacedWing.pose`, 13 ms bat / 34 ms angel for
the pair) is not part of those frame times.

## Limits

* The per-triangle path for triangles wider than 128 px is as slow per triangle as
  the old loop; a view zoomed so far that thousands of triangles exceed 128 px is
  not accelerated.
* Peak memory of a chunk is about 100 MB at the default 1.5 M candidates
  (`raster.MAX_CANDIDATES`).
* `float32` rasterisation (draft) can differ from float64 by a pixel at edges.
* The cache keys on content, so a freshly posed wing always misses once: pose
  drags at final quality still pay the wing's normals and AO per frame (about
  60 ms per feathered wing); that is what the draft tier is for.
