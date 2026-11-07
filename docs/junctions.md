# Skin junctions: the body's skin flows into an appendage root

Design context: [appendages_system_design.md](appendages_system_design.md) (the S1 status in section 6e);
the appendage kinds it plugs into: `kinds.py`; the body it moves: `mhbase.py` (the 19,158-vertex MakeHuman
base, the 13,378-quad body cage). **S1b** (this revision) made the core robust: a validity pass that bounds
stretch and forbids new self-intersections, skin ownership (a root only moulds the skin of its own part of the
body), body-relative sizes, tubes that follow the real mesh, and a public quality helper
([S1b: what changed and what it cost](#s1b-hardening-the-core)).

An appendage is its own closed shell that overlaps the body, and the slicer unions overlapping shells. That
is what the first versions did and it is what read as "slapped over the skin": where a tail, a horn, a wing
arm or an extra limb enters the body there was a hard intersection curve and nothing else, the skin did not
rise to meet the root and no flesh gathered around it. The body mesh cannot get new vertices (the 1,280
morph targets address it BY VERTEX INDEX) but its vertices can be MOVED after the morphs are applied. That
is the whole mechanism: `junction.apply_junctions(base, verts, specs) -> verts'`, a pure, deterministic,
topology-preserving displacement of body cage vertices around each appendage root.

![a tail on five bodies, before (top) and after](images/junction_tail.png)

The same on horns, a bat wing's arm and an extra pair of arms (before on top, after below; neutral, tall,
short, child, heavy):

![horns](images/junction_horn.png)
![bat wing](images/junction_wing.png)
![upper arms](images/junction_arms.png)

## The API (what other tracks build on)

Everything is in the **MakeHuman frame (Y up)**, the frame of the morphed body and of `Placed.mesh()`.

```python
from minimaster.character import junction as J

spec = J.JunctionSpec(
    name="tail-1",
    points=[(x, y, z), ...],     # the centreline, ROOT FIRST: the first points are inside the body
    radii=[r0, r1, ...],         # radius of the surface the skin should hug at each point
    width=1.5,                   # fillet width W in units of base_radius   (W = width * base_radius)
    base_radius=0.45,            # None: max(radii)
    height=0.8,                  # how high the skin climbs the root, over W (1 = a quarter circle)
    sharpness=3.0,               # k of the profile: 2 = quarter ellipse (C1 at W), 3 = C2 at W
    pull=1.0,                    # share of the in-plane redistribution (0 = the skin only climbs)
    strength=1.0,                # scales the whole spec; 0 switches it off
    lead=0.25,                   # the skin starts to rise this many W before the tube's axis reaches it
    mounds=(J.Mound(center, radius, height, end=None, axis=None, stretch=1.0, direction=None,
                    scale=None),),   # scale="trunk": radius / height are FRACTIONS of the body's own trunk radius
    keep=(J.Keep(center, radius, fade=0.5),),   # spheres where the skin must stay as it is
    max_move=None,               # cap on one vertex (default 1.6 W, at least 1.25 x the tallest mound)
    owner="trunk")               # S1b: the part of the body the root belongs to (None: from the skin it exits)

new_verts = J.apply_junctions(base, verts, [spec, ...], info=None)   # pure; verts if nothing to do
q = J.junction_quality(base, verts, new_verts, specs, levels=(1, 2))   # S1b: the validity numbers
assert J.quality_violations(q) == []                                  # S1b: every bound holds
body_verts = J.shaped_body(base, verts, placed, enabled=True)         # from built appendages
specs = J.junction_specs(placed)                                       # [Placed.junction()]
numbers = J.measure_junction(base, verts, new_verts, spec, levels=1)  # see "Numbers"
```

* **`JunctionSpec`** is immutable data (tuples, hashable, `to_dict()` / `from_dict()`, `digest()`,
  `mirrored()` for the right-hand twin of a left-hand spec, `transformed(4x4)` for a rigid motion with
  uniform scale). `J.specs_digest(list)` is a cache key that does not depend on the order. A spec without
  `points` is mounds only (a sacral rise with no tube). A **`Mound`** is a smooth bulge of the skin along its
  own normal (`direction=` a fixed direction instead), `height` < 0 a hollow, `end=` turns it into a ridge or
  a groove along a segment, `axis=` with `stretch=` elongates it. A **`Keep`** is a sphere inside which the
  spec's displacement is zero, growing back to full strength over `fade * radius`: lips, eyelids, a hand.
* **`apply_junctions(base, verts, specs, info=None)`**: `verts` are the **UNDEFORMED morphed control
  vertices** (19,158 x 3) the appendages were placed on; only body cage vertices move (eyes, teeth, joint
  markers do not), the topology and the numbering are untouched so every morph target still addresses the
  same vertices. Returns `verts` ITSELF when no spec is active (a character without appendages is
  bit-identical), never modifies its input, never fed its own result (that would be a second junction on
  the first). Each spec is computed from the INPUT body and the displacements are summed (so the result
  does not depend on the order of the specs and a mirrored pair gives a mirror-symmetric result on a
  symmetric body, to 1e-12), each spec is capped by its own `max_move`, the sum by the largest, and a
  **validity pass** backs off the displacement of the corners of any cage quad that would flip (normal
  turned past 78 degrees) or collapse (area under 12% of its own). `info` receives `moved`, `max_move`,
  `limited` (vertices backed off), `rounds`, `specs`.
* **`Placed.junction(controls=None)`** (`kinds.py`) returns the spec of one placed appendage under its pose
  controls (cached per controls), or `None`. `Kind.junction(placed, controls)` is the hook a kind overrides;
  the base class returns `None`. Every kind has a generic default (below). The spec is built from the placed
  geometry (the loft's own rings, the wing's posed joints, the limb rig's posed joints), never from the
  body, so it is the same on every body.
* **Parameters** of the four kinds (a `ParamSpec`, in the property grid, saved in the record):
  `fillet` (width in root radii; **0 = no junction**; the spec's `width`) and, for tail and horn, `flare`
  (radius multiplier at the base, easing to 1 over the first quarter of the length: `make_tail(...,
  flare=, flare_length=)`, `make_horn` likewise; 0 changes nothing, bit for bit).
* **Pipeline** (`verts -> instantiate -> shaped_body -> everything that meshes the body`):
  `Character.shaped_body(base, verts, placed)` (off when `character.anatomy["junction"]` is false, written
  to the document only then; `Character.junction_enabled`); the CLI (`--no-junction` / `--junction`, both
  saved with `--save-character`); `lab.Engine.shaped_state(...)` (draft, final, pick and export; the
  panel's "Skin flows into the appendages" check box, `LabDocument.set_anatomy(junction=)`); the print path
  (`printprep.character_print_shells(base, shaped_verts, levels)` and `print_report.appendage_report(...,
  verts, body_verts=shaped_verts)`, so the root weld is measured against the body that prints). **Placement
  (`SurfaceAnchor`, the mirror-face map, `kinds.instantiate_all`, `Engine.anchor_points`) keeps using the
  undeformed `verts`**: otherwise an anchor chases its own displacement.
* **`measure_junction(base, verts, shaped, spec, levels)`** returns the numbers in the table below
  (`min_area_ratio`, `flipped`, `join_angle_deg_*`, `p95_dihedral_deg_*`, `dihedral_increase_*`,
  `curvature_step_*`); tests and `tools/junction_report.py` use it.

### Supplying anatomy (what a later track writes)

A track that knows what a tail root really is overrides `Kind.junction` (or builds specs itself and passes
them to `apply_junctions`): the centreline of the first vertebrae with the real radii, a sacral **mound**
over the sacrum, a hollow either side, `keep` around anything that must stay sharp, its own `width` and
`height`. Nothing else changes: the pipeline, the cache keys (a content digest of the specs), the print
report, the checks (no flip, no collapse, cap, mirror symmetry) are generic.

## The mechanism

For a body vertex `p` at distance `delta` outside the tube (`0 < delta < W`; `delta` is measured to the
round-cone chain of the centreline, `|p - c(t)| - r(t)`):

1. **Remaining distance** `rho = W f(delta / W)` with `f(u) = 2u^2 - u^3` (`f(0) = 0`, `f(1) = 1`,
   `f'(1) = 1`: C1 at both ends, the skin is unchanged at `W` and tangent to the tube at the join). The
   vertex is pulled toward the tube **in the skin plane** by `delta - rho`. The cage is only 2 to 4 quads
   wide across a fillet, so this does not make the fillet, it puts the vertices ALONG it where the profile
   is steep.
2. **Climb** along the tube wall (the tangent part of the skin normal at the closest point, so a tube
   leaving the skin at any angle, from perpendicular to lying along it, climbs the way it should) to the
   height of the profile `h(rho) = H (1 - sqrt(1 - (1 - rho / W)^k))`, `H = height * W`: vertical at the wall
   (`rho = 0`), horizontal at `rho = W`, curvature-continuous at `W` for `k >= 3`. Every vertex ends on the
   profile.
3. **Only the exposed tube is moulded around.** The displacement is weighted by how far the closest
   centreline point is above the skin: a tube buried more than a root radius (at most `W`) and `lead * W`
   under the skin leaves it alone, the skin starts to rise `lead * W` before the axis reaches it, and (S1b) it
   never climbs higher than the tube stands above it.
4. Vertices **inside** the tube climb with it (they are hidden) and are kept inside it (a bent or tapered
   tube would otherwise leave them outside once they climbed).
5. (S1b) The tube displacement is weighted by the **ownership** of the vertex (0 on another part's skin),
   capped in width by the part's size, **smoothed** over the cage, then mounds (relative sizes resolved) and
   keep spheres are added, the result is faded where two sheets of skin touch, capped, and the **validity
   pass** of [S1b](#s1b-hardening-the-core) backs vertices off until the skin is valid.

Why not just "move toward the appendage surface so the remaining distance is `W f(delta/W)`": for a tube
leaving the skin at a right angle that is an in-plane compression: the vertices bunch up and nothing rises.
The profile (2) is what makes the fillet; the remap (1) is what makes the cage resolve it.

## Defaults per kind (generic; the anatomy tracks replace them)

| kind | tube | `fillet` (W in root radii) | height | mound |
| --- | --- | --- | --- | --- |
| tail | superseded by the tail's own anatomy spec (`tailrig.junction_spec`, [anatomy_tail.md](anatomy_tail.md)): the flared skin tube from the stem to 2 R beyond the skin point, a sacral rise, two cheek mounds, a keep along the cleft | 1.5 | 1.0 | ridge up the back + cheeks, cleft kept |
| horn | the same | 1.0 | 0.6 | none |
| wing | shoulder -> elbow of the posed arm, radius = the arm bone's | 3.0 | 0.4 | scapular mass at the anchor: **radius 0.5, height 0.055 of the body's trunk radius** (`Mound(scale="trunk")`: 0.78 x 0.086 units on the neutral body, 0.47 x 0.05 on the child; S1: 1.74 x 0.22 units on everyone) |
| extra limb | pivot -> elbow -> into the forearm, posed; radii = the neutral body's own limb profile (`kinds.LIMB_RADII`, fractions of the upper bone) narrowed like the cut end (`neck`), then **capped by the radius of the placed limb's own mesh** (`fit_radii_to_mesh`) | 1.0 | 0.7 | none |

Owners (S1b): tail, wing and extra limbs `trunk`, horn `head` (`Kind.junction_owner`, `JunctionSpec.owner`).

`ring_verts` of a tail and a horn are now **16 by default** (they were 10 and 8 to 12): the skin meets a tube
along a seam, and against a 10-gon the seam is a zigzag between the facets that no displacement of the body
can hide (24 facets: a clean curve). It changes the mesh of every existing tail and horn, with or without the
junction (say `tail:ring_verts=10` to get the old one back).

## Numbers

`python tools/junction_report.py` (5 bodies x 6 kinds; neutral = gender 1, muscle 0.55; tall/short =
height 1 / 0; child = age 0.1875; heavy = weight 1, muscle 1). Columns: `ms` the time of
`apply_junctions` (min of 5, the skeleton cached); `moved`, `maxmv` vertices moved and the largest move (units; 1 unit = 10 cm of
the character); `lim` vertices backed off by the validity pass; `minA` the smallest cage quad area ratio;
`flip` flipped quads; `join` the angle between the skin normal and the tube wall normal near the wall,
before -> after (90 = the skin runs into the tube, 0 = it is tangent); `dih95` the 95th-percentile crease
between neighbouring quads in the fillet zone on the body subdivided once (L1) and twice (L2), before ->
after, and `inc99` the 99th percentile of what the junction ADDED to the crease of an edge; `curv` the
curvature step across the fillet boundary (0 = continuous); `grow` the largest growth of a cage quad's area,
`lip` the largest difference of two neighbours' displacements per edge length, `xnew` the new crossings of the
skin with itself outside the appendage's own tube (cage / subdivided once / twice); `weld` the printability
report's root weld (the share of the appendage's volume inside the body) before -> after; `blocking` its blocking
findings. (The first version's table, without `grow`, `lip` and `xnew`, is in git history; the S1 -> S1b comparison
is [below](#numbers-s1---s1b).)

```
kind       body        ms moved maxmv lim  minA flip   join b->a        L1 dih95 b->a  inc99        L2 dih95 b->a  inc99     curv b->a  grow  lip  xnew c/1/2     weld b->a  blocking
tail       neutral   21.4    52  0.29   0  0.82    0     91->88     11.9-> 12.5  10.1     5.9->  6.5   5.3   0.27-> 0.11   1.32  0.92   0/ 0/ 0    21%-> 25%  0
tail       tall      17.4    42  0.54   3  0.48    0     93->66     11.5-> 15.5  12.7     6.2->  8.1   7.2   0.12-> 0.36   1.50  1.18   0/ 0/ 0    21%-> 34%  0
tail       short     23.2    87  0.30   0  0.76    0     88->87     12.9-> 13.6  10.0     6.4->  6.7   5.1   0.06-> 0.01   1.40  1.03   0/ 0/ 0    21%-> 25%  0
tail       child     35.5   118  0.49   8  0.91    0     89->82     12.6-> 14.7  16.0     6.1->  7.2   8.7   0.17-> 0.38   1.54  1.24   0/ 0/ 0    21%-> 27%  0
tail       heavy     21.5    65  0.17   0  0.65    0     94->102    19.5-> 19.6   5.1     9.6->  9.8   2.6   0.16-> 0.19   1.26  0.55   0/ 0/ 0    22%-> 23%  0
horn       neutral   40.6   122  0.25   0  0.94    0     56->40     20.0-> 20.8   5.9     9.8-> 10.0   3.0   0.41-> 0.38   1.46  0.66   0/ 0/ 0    52%-> 66%  0
horn       tall      28.4    62  0.25   0  0.96    0     66->43     10.5-> 12.1   4.6     5.4->  6.1   2.8   0.01-> 0.17   1.32  0.66   0/ 0/ 0    53%-> 67%  0
horn       short     55.0   163  0.25   0  0.93    0     51->38     24.0-> 24.2   3.7    11.7-> 11.8   1.9   0.74-> 0.83   1.43  0.67   0/ 0/ 0    51%-> 66%  0
horn       child     55.8   156  0.25   0  0.90    0     58->38     22.5-> 22.5   3.2    11.0-> 11.0   1.7   3.26-> 3.64   1.43  0.68   0/ 0/ 0    52%-> 67%  0
horn       heavy     54.9   122  0.25   0  0.94    0     56->40     19.9-> 20.8   5.9     9.8-> 10.0   3.0   0.47-> 0.40   1.46  0.66   0/ 0/ 0    52%-> 66%  0
bat        neutral   33.5   196  0.26   8  0.79    0     93->73     12.8-> 14.0   6.1     6.4->  6.8   3.3   0.35-> 0.24   1.98  0.82   0/ 0/ 0    13%-> 20%  0
bat        tall      26.1   178  0.29   0  0.88    0     96->76     12.9-> 14.2   6.0     6.2->  6.9   3.4   0.27-> 0.37   1.42  0.71   0/ 0/ 0    17%-> 23%  0
bat        short    116.0   274  0.29  64  0.68    0    100->90     12.8-> 12.1   6.9     6.2->  6.0   3.8   0.40-> 0.38   1.98  1.19   0/ 0/ 0    12%-> 16%  0
bat        child     59.1   182  0.18  16  0.70    0     98->84     11.7-> 11.6   3.5     5.8->  5.8   1.9   0.34-> 0.29   1.92  1.24   0/ 0/ 0    10%-> 14%  0
bat        heavy     28.6   168  0.24  12  0.74    0    102->93     17.6-> 17.9   5.2     9.1->  9.8   2.9   0.12-> 0.23   1.93  0.88   0/ 0/ 0    19%-> 22%  0
angel      neutral  116.2   242  0.27   8  0.73    0     81->67     11.6-> 11.4   2.7     5.5->  5.5   1.4   0.09-> 0.09   1.93  0.75   0/ 0/ 0    13%-> 18%  0
angel      tall      41.0   278  0.43  16  0.43    0     83->67     11.4-> 11.5   4.0     5.6->  5.8   2.3   0.08-> 0.03   1.88  0.87   0/ 0/ 0    15%-> 23%  0
angel      short    264.4   342  0.30  84  0.68    0     80->71     13.9-> 14.1   3.0     6.8->  6.9   1.6   0.08-> 0.06   1.93  1.25   0/ 0/ 0    13%-> 17%  0
angel      child    150.5   260  0.18  20  0.16    0     71->64     15.9-> 16.0   3.1     7.8->  7.8   1.5   0.46-> 0.49   1.90  0.87   0/ 0/ 0    12%-> 14%  0
angel      heavy     43.9   224  0.34  24  0.80    0     81->71     14.8-> 14.8   2.9     7.3->  7.3   1.6   0.17-> 0.11   1.96  1.20   0/ 0/ 0    17%-> 24%  0
upper_arms neutral  113.8   256  0.32  32  0.32    0     68->48     11.5-> 12.4   6.2     5.5->  6.2   4.3   0.18-> 0.29   1.98  1.25   0/ 0/ 0    25%-> 25%  0
upper_arms tall     153.2   401  0.49  60  0.34    0     65->49     20.6-> 21.3  11.2    10.1-> 10.6   5.5   0.15-> 0.33   1.94  1.25   0/ 0/ 0    19%-> 19%  0
upper_arms short     54.3   216  0.26  26  0.29    0     70->49     11.5-> 11.9   6.3     5.6->  6.2   4.1   0.15-> 0.10   1.89  1.17   0/ 0/ 0    22%-> 24%  0
upper_arms child     69.4   242  0.23  24  0.31    0     67->46      9.3-> 10.9   9.1     4.6->  5.5   5.4   0.01-> 0.14   1.93  1.14   0/ 0/ 0    23%-> 23%  0
upper_arms heavy     99.5   164  0.31   8  0.56    0     63->51     13.9-> 14.3   3.9     6.8->  7.1   2.5   0.50-> 0.35   1.95  0.74   0/ 0/ 0    37%-> 38%  0
lower_arms neutral   30.7    82  0.32   0  0.72    0     88->58      8.9-> 10.3   7.9     4.4->  5.1   4.4   0.42-> 0.02   1.71  0.87   0/ 0/ 0    25%-> 30%  0
lower_arms tall      43.3   102  0.49   8  0.88    0     84->56      9.0->  9.3   6.8     4.4->  4.7   3.6   0.34-> 0.30   1.91  0.78   0/ 0/ 0    22%-> 28%  0
lower_arms short     32.3    80  0.26   0  0.91    0     86->62      9.3-> 10.6   8.3     4.6->  5.2   4.8   0.15-> 0.12   1.44  0.87   0/ 0/ 0    27%-> 33%  0
lower_arms child     31.9    70  0.24   0  0.85    0     87->61      8.1->  9.9   8.2     4.0->  5.1   5.0   0.23-> 0.27   1.44  0.80   0/ 0/ 0    26%-> 29%  0
lower_arms heavy     34.5    74  0.31   0  0.83    0     95->71     14.2-> 15.1   6.8     7.0->  7.5   3.8   0.19-> 0.26   1.70  0.76   0/ 0/ 0    32%-> 34%  0
```

Reading it:

* No flipped quad, no blocking finding and **no new crossing of the skin with itself** anywhere; the largest quad
  growth is 1.98, the largest neighbour difference 1.25 edge lengths (guards: 2 and 1.25).
* The skin meets the root at **38-102 degrees from tangent** (41-104 without a junction; 32-81 in S1, with the
  stretch of the next section): 38-43 on the horn, 46-76 on the arms and the wings, 66-102 on the tail. The
  fillet is two to three cage quads wide (the cage is about 0.28 units a quad and a root is 0.4-0.5 across), which
  is the ceiling inside a 2x stretch cap; a Catmull-Clark level or two rounds it (the horn's creases are the body's own at L2)
  but does not close the gap to 0. The child and the wings on short bodies improve least.
* The 99th percentile of the crease the junction adds is 1.4-16 degrees on the twice-subdivided skin (mostly under 8); the worst single
  edge no longer rises by 60-95 degrees where the body already has a crease (S1: the cleft, the armpit): that is what the
  crossing and Lipschitz guards and the tight-gap hold are for.
* The weld (the best root piece's share inside the body) rises for the tail (21% -> 23-34%), the horn
  (52% -> 66%), the bat (10-19% -> 14-23%) and the angel (12-17% -> 14-24%); a lower arm gains 3-8 points, an upper
  arm 0-2 (its weld is the buried root, not the skin).
* The backing-off pass is quiet on most kinds (`lim` 0-24) and busiest where the cage is dense or the root is next to the original limb (the
  bat and the angel on a short body, the upper arms on a tall one: 60-84 vertices): `minA` 0.16-0.96.
* Cost: see [S1b](#numbers-s1---s1b): mostly 20-60 ms for one kind on one body (17-260 in the table), 8 ms of it the skeleton the first time.
* The figure stays one overlap-connected component
  (`tests/test_junction.py::test_the_figure_stays_one_connected_component_and_the_report_has_no_new_block`).

## S1b: hardening the core

The first version (S1, above) raised a collar of skin around a root, but the reviewer found that it did
so by stretching the cage (quad areas up to 2.5-11x, edges 2-5x, neighbours moving 2-4 edge lengths
apart), by pushing the skin through itself in the gluteal cleft and the armpits (12-94 new pairs of
crossing triangles on the cage, 25-166 on the body subdivided once, 55-334 twice: the printed body), by
moulding the skin of other body parts (the original arm's armpit, the neck), by tenting the skin with
a root that was buried or fatter than the mesh it stood for, and by a body-independent scapular mound that
puffed a child's back. S1b fixes the mechanism, not the numbers: nothing is tuned to a body.

### What changed, finding by finding

| finding (S1 review) | mechanism now | where |
| --- | --- | --- |
| stretch unbounded (area up to 6.5x, edges 4.7x, neighbours 3.7 edge lengths apart) | `limit_displacement` backs vertices off in 10% steps until no quad area grows past `MAX_AREA_GROWTH` = 2, no edge past `MAX_EDGE_GROWTH` = 2, no two neighbours differ by more than `MAX_LIPSCHITZ` = 1.25 edge lengths (a vertical wall one edge high is 1.41), no quad flips or shrinks under 12%. Before that the raw displacement is **smoothed** over the cage (`diffuse_displacement`: the pinned inside-the-tube vertices keep their displacement, the rim stays zero, the rest takes the 1/edge-weighted mean of its neighbours, 20 Jacobi rounds) and built on **averaged skin normals** (`smooth_normals`): the normals jump across a crease where the cage is also 4x denser, which was the source of the 3-4x neighbour differences | `junction.py` |
| the skin passes through itself in a cleft or armpit (12-52 pairs on the cage, 30-64 subdivided) | a **triangle-triangle crossing test** on the region (`junction_checks.IntersectionTester`: a grid broad phase and Moller-Trumbore, ~10 ms) in the limiter: a pair that crosses only AFTER the move backs its neighbourhood off, unless both triangles are inside the appendage's own tube (the slicer unions them). A crossing on the SUBDIVIDED body appears where two sheets already touch (the closed cleft: 0.02 apart; the Catmull-Clark limit surface crosses at the slightest move of the cage around it): `tight_gap` = 0.04 holds the skin within two rings of such a place (`facing_pairs`) and fades the displacement back in over two more | `junction.py`, `junction_checks.py` |
| a big or flared horn root swells the head (2,803 vertices moved, 1,424 backed off) | the fillet is capped by the size of the part it lies on (`WIDTH_CAP` = 0.6 radii of the part), and a root thicker than that part (`FIT_NONE` = 1.1 radii) moulds nothing: there is no skin for it to flow into | `junction.py` |
| the extra limbs pull the original arm's armpit; the swung arms make cone spikes | **ownership**: `JunctionSpec.owner` (kinds set `head` / `trunk`; `None`: the part the tube exits in) and `bodyframe.BodyFrame` (the skeleton from the `joint-*` markers, parts head / trunk / arm / leg, the radius of each bone's skin measured on this body) give every skin vertex a weight 0..1 for "my part"; the armpit and a hand get 0. Then only the **exposed** tube moulds (the weight falls as the closest point of the centreline sinks under the skin, with a tolerance of one root radius capped by the fillet width), and the skin never climbs higher than the tube stands above it (`stand`): a root that grazes the skin from inside is a hair high, not a spike of the fillet's height | `junction.py`, `bodyframe.py` |
| the cause of the swung-arm spikes | the harvest spec's tube came from a model of the neutral limb (`LIMB_RADII`) and was up to 2x FATTER than the placed copy's own mesh (a limb tapered to fit the torso, a pose): the skin "hidden inside the tube" stuck out of it. `fit_radii_to_mesh` caps the radii by the real mesh (the 20th percentile of the vertices' distance to the axis, per segment of the centreline) | `junction.py`, `kinds.py` |
| the wing's body-independent mound puffs a child's back and makes lobes beside a folded wing | `Mound.scale`: `radius` / `height` are fractions of a body dimension the core resolves from the skeleton (`"head"`, `"trunk"`, `"arm"`, `"leg"`, `"local"`): the generic wing is now 0.5 x 0.055 of the trunk's radius (0.78 x 0.086 units on the neutral body, 0.47 x 0.05 on the child) instead of 1.74 x 0.22 on everyone | `junction.py`, `kinds.py` |
| no way to measure any of this | `junction_quality(base, verts, shaped, specs, levels=(1, 2))` and `quality_violations(q)` | `junction.py` |

### Additions to the API (all backward compatible)

* `apply_junctions(..., max_area_growth=2, max_edge_growth=2, max_lipschitz=1.25, intersections=True,
  smooth=20, tight_gap=0.04, frame=None)`: the S1 call is unchanged and gets the guards. Each guard is
  switched off by `None` / 0 (`frame=False`: no skeleton, so no ownership and absolute mound sizes; the
  synthetic bodies of the tests have no skeleton and behave that way).
  `limit_displacement(..., max_area_growth=, max_edge_growth=, max_lipschitz=, intersections=, benign=)` has
  the same guards, opt-in. `junction_displacement(spec, cage_v, normals, candidates, frame, quads, smooth)` is
  the raw law of one spec (plus ownership and smoothing when given a frame / the cage).
* `JunctionSpec.owner` (`None`, a part, or `"arm"` / `"leg"`) and `Mound.scale`: serialised only when set, so
  the digest of an old spec is the one it always had (`.mmchar` files do not store specs: they are rebuilt from
  the appendage's parameters, so nothing to migrate); `transformed()` does not scale a relative size;
  `mirrored()` keeps both. `Kind.junction_owner` is the class attribute the loft kinds read.
* `resolve_spec(spec, frame)` (relative mounds -> absolute), `fit_radii_to_mesh(points, radii, vertices)`,
  `diffuse_displacement`, `smooth_normals`, `facing_pairs`, `tube_closest` (distance + arclength),
  `tube_emergence` (where the centreline comes out of the skin).
* `bodyframe.BodyFrame(base, verts)`, `frame_or_none(base, verts)` (cached by the vertices' hash):
  `part_scores`, `owner`, `resolve_owner`, `owner_weights`, `size`, `part_size`.
* `junction_checks`: `IntersectionTester`, `self_intersection_pairs`, `stretch_report`,
  `cage_edges`, `neighbour_keys`, `share_a_quad`, `ring_distance`, `pairs_cross`, `vertex_ring`, `overlapping_pairs`.
* `junction_quality(...)` returns `moved`, `max_move`, `flipped`, `area_growth_max`, `area_ratio_min`,
  `edge_growth_max`, `edge_ratio_min`, `lipschitz_max` / `_p99`, the crossings (`selfx_before`, `selfx_after`,
  `selfx_new`, `selfx_new_visible` = outside every tube, and the same per subdivision level as `L1_...`),
  and `by_part` (vertices moved and the largest move per body part); `quality_violations(q)` names every bound
  that is broken. `tools/junction_report.py` prints `grow`, `lip` and `xnew` from it.

### Numbers: S1 -> S1b

Cage quads and edges the junction touched; `join angle` before-after (90 = the skin runs into the tube, 0 =
tangent), measured on the body subdivided twice; "crossings" are NEW pairs of crossing triangles outside the
appendage's own tube on the cage / subdivided once / twice. Neutral, child and heavy bodies (`tools/junction_report.py`
has all five); S1 measured with the same code on the S1 output. `ms` is S1b's cost of `apply_junctions`
(S1: 4-12 ms).

| kind | body | moved | max move | area growth x | edge growth x | neighbour Lipschitz | new crossings cage / L1 / L2 | join angle | L2 crease +p99 | ms |
|---|---|---|---|---|---|---|---|---|---|---|
| tail | neutral | 74 -> 52 | 0.54 -> 0.29 | 2.55 -> 1.32 | 3.38 -> 1.31 | 3.11 -> 0.92 | 12/25/55 -> 0/0/0 | 91-62 -> 91-89 | 9.6 -> 5.3 | 69 |
| tail | child | 159 -> 118 | 0.54 -> 0.49 | 4.27 -> 1.54 | 5.01 -> 1.62 | 4.57 -> 1.24 | 28/53/99 -> 0/0/0 | 89-53 -> 89-81 | 10.0 -> 8.7 | 51 |
| tail | heavy | 106 -> 65 | 0.54 -> 0.17 | 4.63 -> 1.26 | 5.20 -> 1.27 | 4.56 -> 0.55 | 27/43/85 -> 0/0/0 | 95-65 -> 95-100 | 10.6 -> 2.6 | 30 |
| horn | neutral | 118 -> 122 | 0.24 -> 0.25 | 1.74 -> 1.46 | 1.81 -> 1.45 | 1.09 -> 0.66 | 0/0/0 -> 0/0/0 | 56-35 -> 56-40 | 4.3 -> 3.0 | 52 |
| horn | child | 176 -> 156 | 0.25 -> 0.25 | 1.69 -> 1.43 | 1.82 -> 1.46 | 1.13 -> 0.68 | 0/0/0 -> 0/0/0 | 58-34 -> 58-38 | 3.0 -> 1.7 | 70 |
| horn | heavy | 118 -> 122 | 0.24 -> 0.25 | 1.74 -> 1.46 | 1.81 -> 1.45 | 1.09 -> 0.66 | 0/0/0 -> 0/0/0 | 56-35 -> 56-40 | 4.3 -> 3.0 | 53 |
| bat | neutral | 397 -> 196 | 0.37 -> 0.26 | 2.47 -> 1.98 | 2.07 -> 1.78 | 1.37 -> 0.82 | 20/0/0 -> 0/0/0 | 93-56 -> 93-73 | 8.6 -> 3.3 | 45 |
| bat | child | 721 -> 182 | 0.40 -> 0.18 | 2.96 -> 1.92 | 3.02 -> 1.79 | 2.37 -> 1.24 | 66/122/264 -> 0/0/0 | 97-81 -> 97-84 | 9.6 -> 1.9 | 67 |
| bat | heavy | 359 -> 168 | 0.34 -> 0.24 | 2.52 -> 1.93 | 2.22 -> 1.61 | 1.59 -> 0.88 | 42/72/146 -> 0/0/0 | 104-79 -> 104-92 | 9.3 -> 2.9 | 39 |
| angel | neutral | 592 -> 242 | 0.61 -> 0.27 | 5.49 -> 1.93 | 2.81 -> 1.64 | 2.26 -> 0.75 | 52/102/226 -> 0/0/0 | 81-66 -> 81-67 | 5.0 -> 1.4 | 126 |
| angel | child | 1368 -> 260 | 0.63 -> 0.18 | 11.38 -> 1.90 | 4.36 -> 1.63 | 3.71 -> 0.87 | 94/166/334 -> 0/0/0 | 71-60 -> 71-64 | 8.7 -> 1.5 | 198 |
| angel | heavy | 530 -> 224 | 0.56 -> 0.34 | 5.46 -> 1.96 | 3.06 -> 1.56 | 2.08 -> 1.20 | 94/152/268 -> 0/0/0 | 82-59 -> 82-70 | 7.3 -> 1.6 | 55 |
| upper_arms | neutral | 264 -> 256 | 0.32 -> 0.32 | 3.95 -> 1.98 | 2.84 -> 1.72 | 1.85 -> 1.25 | 0/0/0 -> 0/0/0 | 67-38 -> 67-48 | 6.1 -> 4.3 | 151 |
| upper_arms | child | 246 -> 242 | 0.24 -> 0.23 | 2.71 -> 1.93 | 2.57 -> 1.94 | 1.58 -> 1.14 | 0/0/0 -> 0/0/0 | 68-38 -> 68-48 | 7.7 -> 5.4 | 80 |
| upper_arms | heavy | 170 -> 164 | 0.31 -> 0.31 | 2.52 -> 1.95 | 1.94 -> 1.64 | 1.06 -> 0.74 | 0/0/0 -> 0/0/0 | 62-45 -> 62-51 | 4.4 -> 2.5 | 137 |
| lower_arms | neutral | 134 -> 82 | 0.32 -> 0.32 | 1.64 -> 1.71 | 1.63 -> 1.53 | 1.09 -> 0.87 | 0/0/0 -> 0/0/0 | 86-64 -> 86-56 | 8.3 -> 4.4 | 33 |
| lower_arms | child | 124 -> 70 | 0.24 -> 0.24 | 1.65 -> 1.44 | 1.67 -> 1.37 | 1.25 -> 0.80 | 0/0/0 -> 0/0/0 | 88-65 -> 88-59 | 8.7 -> 5.0 | 36 |
| lower_arms | heavy | 118 -> 74 | 0.32 -> 0.31 | 1.81 -> 1.70 | 2.14 -> 1.72 | 1.21 -> 0.76 | 0/0/0 -> 0/0/0 | 98-80 -> 98-71 | 9.5 -> 3.8 | 38 |
| extra_legs | neutral | 1094 -> 526 | 0.39 -> 0.30 | 6.46 -> 1.99 | 4.70 -> 1.73 | 3.71 -> 1.15 | 0/0/0 -> 0/0/0 | 51-39 -> 51-47 | 11.1 -> 3.0 | 88 |
| extra_legs | child | 1158 -> 656 | 0.29 -> 0.25 | 4.34 -> 1.91 | 3.50 -> 1.64 | 2.58 -> 1.17 | 0/0/0 -> 0/0/0 | 55-34 -> 55-48 | 13.3 -> 3.2 | 97 |
| extra_legs | heavy | 840 -> 206 | 0.39 -> 0.12 | 4.19 -> 1.24 | 3.40 -> 1.18 | 2.66 -> 0.37 | 82/126/254 -> 0/0/0 | 41-35 -> 41-41 | 10.2 -> 0.9 | 56 |

Reading it, honestly:

* **Validity is now exact and measured.** Every kind on every body: no flipped quad, area growth <= 2.0
  (S1: up to 11.4), edge growth <= 1.94, neighbours <= 1.25 edge lengths apart (S1: up to 4.6) and **no new
  crossing of the skin with itself on the cage or on the body subdivided once or twice** (S1: up to 94 / 166 / 334).
  `tests/test_junction.py::test_the_junction_is_a_valid_skin_on_every_kind_and_body` asserts it for every kind
  on five bodies.
* **The join angle got worse for the tail and the wing, and that is the price.** S1 bought its 91 -> 62 on the
  tail and 93 -> 56 on the bat with 2.5-3.4x stretched edges and crossing cheeks. On the 0.28-unit cage a
  root 0.4-0.5 wide is two to three quads across, so a fillet that is tangent to the wall needs an edge to
  stretch past 2x or to turn a corner of more than a quarter turn. Within the caps the cage gives 35-50 degrees
  on the horn and the arms (horn 56 -> 40, upper arms 67 -> 48, lower arms 86 -> 56), 73 on the bat, and almost
  nothing on a tail whose root sits in the cleft (91 -> 89; the heavy body 95 -> 100, a slight groove: the two cheeks
  already touch there, so the skin around the root is held, see `tight_gap`). The creases it makes are
  smaller than S1's (the 99th percentile of what the junction adds to an edge's crease on the twice-subdivided
  body: 1-9 degrees, S1 3-13).
* **What limits the tangency** is the smoothing and the area cap, not the Lipschitz bound (a sweep of
  `max_lipschitz` 1.0 - 2.0 - off gives the same join angles to the degree on every kind: the smoothing already
  brings the neighbours within 1). Joins after the guards, for the neutral body; the number in brackets is the
  largest quad growth, `lip` the largest neighbour difference, `p95+` the 99th percentile of the crease the junction adds:

smoothing rounds (join angle after, area growth, Lipschitz, p99 crease increase)

| kind | smooth=0 | smooth=10 | smooth=20 | smooth=40 |
|---|---|---|---|---|
| tail | 78 (x1.35, lip 1.08, p95+6) | 88 (x1.32, lip 0.90, p95+5) | 89 (x1.32, lip 0.92, p95+5) | 90 (x1.32, lip 0.93, p95+5) |
| horn | 35 (x1.72, lip 1.10, p95+4) | 40 (x1.48, lip 0.67, p95+3) | 40 (x1.46, lip 0.66, p95+3) | 40 (x1.45, lip 0.66, p95+3) |
| bat | 70 (x1.90, lip 0.97, p95+7) | 73 (x1.95, lip 0.82, p95+4) | 73 (x1.98, lip 0.82, p95+3) | 73 (x1.99, lip 0.82, p95+3) |
| upper_arms | 41 (x1.98, lip 1.25, p95+6) | 48 (x1.98, lip 1.25, p95+4) | 48 (x1.98, lip 1.25, p95+4) | 48 (x1.98, lip 1.25, p95+4) |
| lower_arms | 47 (x1.71, lip 1.17, p95+7) | 56 (x1.71, lip 0.91, p95+4) | 56 (x1.71, lip 0.87, p95+4) | 56 (x1.71, lip 0.87, p95+4) |

area / edge growth cap

| kind | cap=1.5 | cap=2.0 | cap=3.0 | cap=None |
|---|---|---|---|---|
| tail | 89 (x1.32, lip 0.92, p95+5) | 89 (x1.32, lip 0.92, p95+5) | 89 (x1.32, lip 0.92, p95+5) | 89 (x1.32, lip 0.92, p95+5) |
| horn | 40 (x1.46, lip 0.66, p95+3) | 40 (x1.46, lip 0.66, p95+3) | 40 (x1.46, lip 0.66, p95+3) | 40 (x1.46, lip 0.66, p95+3) |
| bat | 80 (x1.46, lip 0.90, p95+4) | 73 (x1.98, lip 0.82, p95+3) | 72 (x2.11, lip 0.91, p95+3) | 72 (x2.11, lip 0.91, p95+3) |
| upper_arms | 51 (x1.49, lip 1.25, p95+5) | 48 (x1.98, lip 1.25, p95+4) | 46 (x2.84, lip 1.25, p95+4) | 46 (x3.14, lip 1.25, p95+4) |
| lower_arms | 67 (x1.45, lip 0.62, p95+3) | 56 (x1.71, lip 0.87, p95+4) | 56 (x1.71, lip 0.87, p95+4) | 56 (x1.71, lip 0.87, p95+4) |

  Smoothing gives up 4-11 degrees of tangency (smooth=0 would be 35 / 70 / 41 / 47) for creases and neighbour
  differences that S1 did not have; 10 rounds are as good as 20. Allowing a 3x quad (or no cap) buys the bat and the
  upper arms 1-2 degrees and one of them a 2.8-3.1x quad: not worth it. The next step up would be to apply the
  law on the Catmull-Clark subdivided body (4x the vertices, 0.14 units a quad) instead of the cage, or to solve for a
  cage whose limit surface follows the law: both change what `shaped_body` returns and are the pipeline owners' call.
* **The tail is the weakest result and its cause is not the skin.** It is placed off the midline at the side of
  the left buttock (placement is the tail track's), where the body has a closed cleft (the two cheeks 0.02
  apart on a heavy body): the core now refuses to move that skin, which is right, and the collar around the root is
  small there.
* **Cost**: 17-260 ms per `apply_junctions` (mostly under 60; a tail and a horn pair 50-70 ms; the wings on a short body, whose root sits
  by a closed armpit and backs off 60-80 vertices, the slowest) against 4-12 ms: the crossing
  test is ~10 ms a round, the ownership frame 8 ms the first time (cached afterwards), the smoothing 3 ms a spec.
  The lab caches the shaped skin by the body and the specs' content and a drag draws the unshaped skin, so the
  cost is paid once per edit, 150 ms after the last change. `test_a_pair_and_a_tail_take_well_under_150_milliseconds`.

### Poses and the oversized cases the reviewer found

* **Folded wing**: S1's two pillow-like lobes of back skin beside the roots (up to 0.3 units) were the mound
  (1.74 wide), not the tube: beyond the fillet of the arm the skin now moves by at most the mound's 0.086
  (`test_a_wing_pose_does_not_puff_the_back_into_lobes...`, folded, swung and at rest, neutral and child).
* **Swung lower arms** (`arm-1:swing=50`): the cone-shaped spikes were the tube fatter than the arm, and a root
  grazing the skin from inside; the displacement keeps within 1.25 edge lengths and 2x, and the raw law (S1) is shown
  to violate both on this pose in the test.
* **Extra legs / upper arms**: the extra legs moved 1,094 vertices and dragged the original arm's armpit into a
  web; now the original arms and the head move by at most 0.02-0.05 units (the rim of the blend: `by_part` in
  `junction_quality`, `test_an_extra_limb_does_not_mould_the_skin_of_the_original_arm_or_the_head`).
* **`horn:flare=2` and `horn:length=8,base_radius=1.0`**: a root thicker than 1.1 head radii moulds nothing
  (0 vertices moved; S1 moved 1,000-2,800); a sane large one (`flare=1.5,base_radius=0.8`) moves only head skin,
  at most 700 vertices by at most 0.3.

### References used

The skin reference library (`refs/skin`, see its `DIGEST.md`) measured how the skin meets a protrusion, and
what the generic core can and cannot say about it:

* `wm_leonardo_shoulder_photo` (a bare shoulder, arm raised): the arm underside meets the chest wall at an inside
  angle of about 122 degrees with a rounding of 0.06-0.11 root radii: a near-sharp crease where a limb folds
  against the trunk; on the upper side the deltoid and trapezius flow into the neck slope over more than a
  radius. This is why the core does not fillet an armpit (it holds the skin where two sheets touch and gives the
  skin of another part weight 0) and why a single symmetric fillet profile is only the generic part: the
  asymmetric one belongs to the anatomy tracks that know which side of the root is the fold.
* `wm_giraffe_head`, `ov_giraffe_ossicones` (skin-covered horn cores): the cores widen toward the base by 4-8
  degrees a side and their outer edges sweep into the skull without a seam, with a concave crotch between a pair: the
  flare is the appendage's own geometry (`flare`), not something the skin can fake; a hair line, not a fillet, is
  the seam.
* `wm_cassowary_casque` and `wm_antlers_season_start`: hard caps meet the skin along a sharp rim of ~0.05-0.11 root
  radii (casque) and the skin dome of a pedicle is 0.44-0.88 radii high: the fillet widths of the kinds
  (horn 1.0, tail 1.5, harvest 1.0, wing arm 3.0 root radii) are at the generous end for hard roots and about right
  for the soft ones; the digest's reading ("hard caps and creases blend over about 0.1 r, soft muscle masses over more than 1 r")
  is the per-kind decision a later track makes with the `fillet` parameter.
* `wc_ecorche_side_V0008062`, `wc_horse_ecorche_V0016913`, `url_cuyer_fig14_15`, `wm_gray_410`: the visible
  "fillet" around a limb root is mostly a muscle mass bigger than the shaft (x1.3 on a human arm, far more on a
  quadruped limb), the skin a thin shell over it: the core's `Mound` with a relative size is the machinery for that
  mass; the numbers per kind are the tracks' (a generic wing mound of 0.5 x 0.055 trunk radii is deliberately small).
* `tail/wc_dwgghrqs`, `tail/wc_bayw69tw` (the library's takeaways, "human sacral surface"): two gluteal masses meet at a
  midline cleft and "a junction reproduces both, it does not fill the cleft": the core's `tight_gap` hold keeps a closed
  cleft closed instead of raising the skin across it. `horn/A29` (ossicone): a valley fillet of 0.5 root radii between a pair
  and flush skin on the outer flank: the numbers behind the horn's 1.0 r fillet being generous, not short.
* `url_bridgman_armpit_front`, `wm_winged_scapula`: the axilla as an interlocking of masses with a hollow between
  them and a scapular border that tents the skin with a short fillet: what the ownership and the tight-gap hold
  protect (the hollow), and what the relative mound is for (the border).

A side-by-side of a giraffe head with a horn pair, and of the Leonardo shoulder with the extra lower arms, makes the
gap plain (the generic fillet is a faint collar where the references have a flared base and a muscle mass): that
gap is the anatomy tracks' work and the reason the core provides ownership, relative sizes and a validity guard
that a mound or a flare can be built on without breaking the skin.

### Tests that can fail

`tests/test_junction.py` grew from 49 to 115 tests. The guards are tested on synthetic bodies first (a quad that would
grow 3x, an edge 3x, neighbours 3 edge lengths apart, a cap pushed through the opposite cap, a closed cleft; the law
itself with the guards off) and then on the real body: every kind on five bodies must give an empty
`quality_violations`, on the cage and on the body subdivided once (the tail and the wing twice), the poses and the
oversized horns above, the original arm and head untouched by an extra limb, the owners, the skeleton's map of the
body, the tube's radii against the mesh, the lab's mirrored draft and export. A **mutation run** of 40 single-line
mutants (the 20 the S1 reviewer used, plus the guards, the caps, the smoothing, the ownership, the relative sizes, the
fit to the mesh, the neck as a head segment, the exposure tolerance, the 0.6 back-off step that makes a groove): 4 survived
the first version of the new tests (the `stand` weight, the smoothing, the width cap, a mutant that replaced a comment),
each got a test that fails on it, and all 40 are now caught (`tools`-free: the runner is a 60-line script of
`(file, old, new)` triples; the five survivors of the S1 review - `LIMB_RADII` x0.5, the tail's `reach`, the lab export
without `body_verts`, the mirrored twin of a draft, the `sps` factor - are among them and caught).
The two tests of the fillet's law (`..._law_holds_inside_it`, `..._climbs_the_tube...`) now call `apply_junctions` with
the guards off (`LAW_ONLY`) and check the guarded call separately: the law is one thing, what the cage can follow another.
The S1 test that the skin meets every root at least 15 degrees closer to tangent is now 5 degrees for the tail
(S1's 27 was bought with 2.5-3.4x stretched edges and 12 new crossings), and the budget of the pair-and-a-tail test is
150 ms (it was 50 ms for 8 ms of work).

### Caveats

* **The generic tail got valid, not better looking.** Next to the baseline the shaped tail is barely distinguishable on
  the neutral body, and on the heavy body (a closed cleft) the join is 94 -> 102 degrees, a hair worse than none. The tail's
  look comes from where it is placed (the midline anchor) and from the sacral mass behind it (a `Mound(scale="trunk")`, a
  hollow either side, `keep` around the cleft): the tail track's. What the core gives it is that none of that can break the skin.
* **A root thicker than 1.1 radii of the part it grows from moulds nothing.** `horn:flare=2` and `horn:length=8,base_radius=1.0`
  (both reachable in the property grid) switch the junction off for that horn (0 vertices moved; S1 moved 1,000-2,800 and swelled
  the head). Better than a swollen head, but a behaviour change: `FIT_NONE`, `FIT_FULL`.
* **No per-spec opt-out** of the cleft hold (`tight_gap`) or of ownership: `shaped_body` passes no keywords and a
  `JunctionSpec` has no field to say "this spec may move skin next to a closed cleft" or "this root may mould the arm's
  skin". A track that needs it today has only `apply_junctions(..., tight_gap=None, frame=False)` for everything. A
  per-spec switch is the next API addition; the integrator should know.
* **`kinds.py` hunks that other tracks will also rewrite**: `WingKind.junction` (the mound line and `owner`) and
  `HarvestKind.junction` (the `fit_radii_to_mesh` call and `owner`); `Kind.junction_owner` and the two constants
  `WING_MOUND_RADIUS` / `WING_MOUND_HEIGHT` (now fractions of the trunk's radius, not arm lengths).
* Cost is 3-20x S1's (17-260 ms against 4-12), see above.
* Tangency is limited by the cage (above); the table is the honest number, not a target met.
* The crossing test is exact on the cage and the guarantee on the subdivided body (levels 1 and 2) is
  **measured, not proved**: it holds for the kinds, bodies and poses in the tests, through the `tight_gap` hold, which is
  conservative (it can cost a collar next to a closed cleft). A body with another closed fold would be covered by the
  same rule but is not in the tests.
* Ownership is only as good as the skeleton: the deltoid and the shoulder's top are shared between the trunk and the arm
  by the bones' radii, so an extra arm rooted at the shoulder joint (the `upper_arms` preset) has its root skin weighted
  down where it overlaps the original arm. `JunctionSpec.owner` overrides the guess per kind.
* The skin is moulded in the rest pose of the body (the junction is computed on the morphed, unposed body).
* `Mound.scale` needs the skeleton markers (the real base mesh has them); without them (a synthetic body) it stays
  absolute. The skeleton is cached by the hash of the morphed vertices: a body that has not changed costs nothing.

## Subdivision

The body is subdivided (Catmull-Clark, levels 1-2) for render and print. The junction moves cage vertices
and CC rounds the result, so `levels >= 1` renders and prints the fillet smoothly (the 95th-percentile crease
in the zone falls by about half with each level: horn 20 -> 10 degrees from L1 to L2, `tests/test_junction.py::
test_the_subdivided_shaped_body_is_smooth_where_the_cage_is_coarse`); at level 0 (the CLI's default for a
picture) it is as coarse as the cage. `minimaster character --subdivide 1 -o x.stl` used to die with "cage
is not a closed 2-manifold: 38 edges" even without appendages: the two TEETH helper islands are open (38
boundary edges) and Catmull-Clark of an open cage raises. At level 0 they were dropped by the watertight
check AFTER building; `printprep.character_print_shells` now leaves them out BEFORE subdividing
(`subdiv.cage_is_closed`). Subdivided prints work, with and without appendages. S1b checks the crossings of the
skin with itself on the subdivided body too (`junction_quality(..., levels=(1, 2))`): a cage that does not cross
itself can still make a subdivided body that does where two sheets of skin touch (the closed gluteal cleft), which is
what `tight_gap` holds.

## A kind that knows its anatomy: the tail

The tail is the first kind to replace the generic default (`Kind.junction` returns its own spec, nothing else
changes): [anatomy_tail.md](anatomy_tail.md). Its root is on the midline of the sacrum (a midline anchor: the bound
quad paired with its mirror image), its shell is the envelope of a caudal vertebra chain and a muscle sleeve
(`Placed.layers()`: `bone`, `muscle`, `skin`; `ecorche.py` draws them), and its spec is the tail's flared skin tube, a
low sacral rise up the back, two cheek mounds and a row of `Keep` spheres down the gluteal cleft, which is what stops
the two cleft walls from being pushed through each other (the generic spec added 12 self-intersecting pairs there).
The spec uses only what is documented above (`mounds`, `keep`, `width`, `height`): a kind that needs more than a sphere
of `Keep` can chain several, as the tail does.

![the tail's skin flowing into its root](images/tail_before_after.png)

## Seen from the lab, and its limits

* The Character tab draws, picks and exports the shaped skin. During a **body-slider or shape drag the draft
  keeps the unshaped skin** (the 4k-triangle proxy averages a fillet away and reshaping on every frame costs
  30-120 ms); the final render 150 ms later has the fillet. A pose drag keeps the cached skin of the rest pose.
* The body proxy of the draft is cluster-averaged, so the fillet is only faintly there in the draft.
* Appendages that overlap the original limb (the extra upper arms start at the same shoulder) used to pull the
  original limb's skin; since S1b the skin of another part of the body is left alone (ownership), so the root skin
  where the two overlap is moulded less (`JunctionSpec.owner`).
* `junction_specs` skips an appendage whose junction cannot be worked out, with an `AppendageWarning`.
* Not done: the right-hand spec of an asymmetric body is not reflected from the left; per-appendage
  junction parameters beyond `fillet` and `flare` (a tail's `height` / `sharpness` are the kind's); a real
  slicer was not run (the report assumes the union, as before).

## Looking at it

```
python tools/junction_view.py OUT.png "tail;horn" tail 200,12 160,25 --pair --r 1.5
python tools/junction_view.py OUT.png "wing" wing --macro heavy --levels 2 --skin-only
python tools/junction_report.py --kinds tail,horn --bodies neutral,child --levels 1,2
```

and from code (what the tests and a reviewer use):

```python
q = J.junction_quality(base, verts, shaped, J.junction_specs(placed), levels=(1, 2))
print(J.quality_violations(q) or "valid", q["area_growth_max"], q["by_part"])
```

`--pair` puts the body without the junction above and with it below; `--skin-only` hides the appendages
(the shaped skin alone shows whether any of it is left sticking out of the appendage); `--before` is the
old look. The tests are `tests/test_junction.py` (the mechanism on a synthetic sphere, then the real body).
