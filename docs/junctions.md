# Skin junctions: the body's skin flows into an appendage root

Design context: [appendages_system_design.md](appendages_system_design.md) (the S1 status in section 6e);
the appendage kinds it plugs into: `kinds.py`; the body it moves: `mhbase.py` (the 19,158-vertex MakeHuman
base, the 13,378-quad body cage).

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
    mounds=(J.Mound(center, radius, height, end=None, axis=None, stretch=1.0, direction=None),),
    keep=(J.Keep(center, radius, fade=0.5),),   # spheres where the skin must stay as it is
    max_move=None)               # cap on one vertex (default 1.6 W, at least 1.25 x the tallest mound)

new_verts = J.apply_junctions(base, verts, [spec, ...], info=None)   # pure; verts if nothing to do
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
   centreline point is above the skin: a tube buried a radius and a bit under the scalp leaves it alone,
   and the skin starts to rise `lead * W` before the axis reaches it.
4. Vertices **inside** the tube climb with it (they are hidden) and are kept inside it (a bent or tapered
   tube would otherwise leave them outside once they climbed).
5. Mounds and keep spheres, then the caps and the flip/collapse pass described above.

Why not just "move toward the appendage surface so the remaining distance is `W f(delta/W)`": for a tube
leaving the skin at a right angle that is an in-plane compression: the vertices bunch up and nothing rises.
The profile (2) is what makes the fillet; the remap (1) is what makes the cage resolve it.

## Defaults per kind (generic; the anatomy tracks replace them)

| kind | tube | `fillet` (W in root radii) | height | mound |
| --- | --- | --- | --- | --- |
| tail | the first rings of the loft from the sunk base to `1.5 x sink + 1.5 W`, the inscribed circle of each ring times 0.97 | 1.5 | 0.8 | none |
| horn | the same | 1.0 | 0.6 | none |
| wing | shoulder -> elbow of the posed arm, radius = the arm bone's | 3.0 | 0.4 | scapular mass: radius 0.55, height 0.07 arm lengths, at the anchor |
| extra limb | pivot -> elbow -> into the forearm, posed; radii = the neutral body's own limb profile (`kinds.LIMB_RADII`, fractions of the upper bone) narrowed like the cut end (`neck`) | 1.0 | 0.7 | none |

`ring_verts` of a tail and a horn are now **16 by default** (they were 10 and 8 to 12): the skin meets a tube
along a seam, and against a 10-gon the seam is a zigzag between the facets that no displacement of the body
can hide (24 facets: a clean curve). It changes the mesh of every existing tail and horn, with or without the
junction (say `tail:ring_verts=10` to get the old one back).

## Numbers

`python tools/junction_report.py` (5 bodies x 6 kinds; neutral = gender 1, muscle 0.55; tall/short =
height 1 / 0; child = age 0.1875; heavy = weight 1, muscle 1). Columns: `ms` the time of
`apply_junctions` (min of 5); `moved`, `maxmv` vertices moved and the largest move (units; 1 unit = 10 cm of
the character); `lim` vertices backed off by the validity pass; `minA` the smallest cage quad area ratio;
`flip` flipped quads; `join` the angle between the skin normal and the tube wall normal near the wall,
before -> after (90 = the skin runs into the tube, 0 = it is tangent); `dih95` the 95th-percentile crease
between neighbouring quads in the fillet zone on the body subdivided once (L1) and twice (L2), before ->
after, and `inc99` the 99th percentile of what the junction ADDED to the crease of an edge; `curv` the
curvature step across the fillet boundary (0 = continuous); `weld` the printability report's root weld (the
share of the appendage's volume inside the body) before -> after; `blocking` its blocking findings.

```
kind       body        ms moved maxmv lim  minA flip   join b->a        L1 dih95 b->a  inc99        L2 dih95 b->a  inc99     curv b->a     weld b->a  blocking
tail       neutral    5.8    74  0.54   9  0.49    0     91->64     11.9-> 16.3  17.2     5.9->  8.4   9.6   0.27-> 0.56    21%-> 33%  0
tail       tall       4.8    42  0.54   7  0.32    0     93->61     11.3-> 19.2  20.9     6.2->  9.9  10.6   0.12-> 0.68    21%-> 34%  0
tail       short      6.1   116  0.54  11  0.32    0     88->56     13.0-> 16.0  15.6     6.4->  8.0   9.0   0.06-> 0.33    21%-> 34%  0
tail       child      4.6   159  0.54  20  0.36    0     89->53     12.6-> 15.7  17.3     6.1->  7.8  10.0   0.17-> 0.25    21%-> 34%  0
tail       heavy      4.8   106  0.54  11  0.27    0     94->66     19.6-> 24.3  20.6     9.6-> 11.9  10.6   0.16-> 0.14    22%-> 34%  0
horn       neutral    4.8   118  0.24   0  0.93    0     56->36     20.0-> 20.9   8.9     9.8-> 10.2   4.3   0.41-> 0.40    52%-> 66%  0
horn       tall       5.3    64  0.24   0  0.95    0     66->41     10.7-> 12.1   6.6     5.4->  6.1   3.4   0.01-> 0.09    53%-> 67%  0
horn       short      5.2   180  0.25   0  0.91    0     51->32     24.0-> 24.2   6.6    11.8-> 11.8   3.3   0.74-> 0.66    51%-> 65%  0
horn       child      5.8   176  0.25   0  0.93    0     58->34     22.5-> 22.5   6.3    11.0-> 11.0   3.0   3.26-> 3.48    52%-> 66%  0
horn       heavy      5.1   118  0.24   0  0.93    0     56->36     19.9-> 20.9   8.9     9.8-> 10.2   4.3   0.47-> 0.46    52%-> 66%  0
bat        neutral    5.1   397  0.37  12  0.51    0     93->56     12.8-> 16.4  17.3     6.4->  8.2   8.6   0.35-> 0.22    13%-> 22%  0
bat        tall       4.0   280  0.34   0  0.67    0     96->63     12.3-> 15.3   8.8     6.2->  7.4   4.6   0.27-> 0.20    17%-> 25%  0
bat        short      6.1   509  0.38  24  0.31    0    100->68     12.8-> 15.3  17.1     6.2->  7.5   8.9   0.40-> 0.32    12%-> 21%  0
bat        child      7.6   721  0.40  60  0.26    0     98->81     11.8-> 13.0  18.6     5.9->  6.7   9.6   0.34-> 0.20    10%-> 17%  0
bat        heavy      5.5   359  0.34  18  0.31    0    102->79     17.2-> 20.2  14.6     8.8-> 10.4   9.3   0.12-> 0.04    19%-> 26%  0
angel      neutral    8.8   592  0.61  46  0.43    0     81->65     11.4-> 12.1   9.7     5.5->  6.0   5.0   0.09-> 0.02    13%-> 26%  0
angel      tall       6.3   380  0.59  24  0.30    0     83->56     11.3-> 12.3  11.3     5.6->  6.1   5.8   0.08-> 0.06    15%-> 28%  0
angel      short     10.7   883  0.62  76  0.27    0     80->74     13.9-> 14.9   9.4     6.8->  7.4   5.0   0.08-> 0.05    13%-> 23%  0
angel      child     10.6  1368  0.63 125  0.36    0     71->60     15.9-> 18.8  17.6     7.8->  9.2   8.7   0.46-> 0.57    12%-> 22%  0
angel      heavy      6.0   530  0.56  44  0.44    0     81->58     14.9-> 15.5  12.8     7.3->  7.8   7.3   0.17-> 0.10    17%-> 31%  0
upper_arms neutral    6.9   264  0.32   8  0.23    0     68->38     11.4-> 13.1  10.8     5.5->  6.7   6.1   0.18-> 0.30    25%-> 25%  0
upper_arms tall      10.0   461  0.50  28  0.26    0     65->41     20.6-> 22.1  15.2    10.1-> 10.9   7.7   0.15-> 0.21    19%-> 19%  0
upper_arms short      5.5   220  0.26   8  0.32    0     70->42     11.5-> 13.4  11.8     5.6->  6.9   6.5   0.19-> 0.57    22%-> 24%  0
upper_arms child      6.7   246  0.24   8  0.23    0     67->38      9.3-> 12.2  13.9     4.6->  6.2   7.7   0.01-> 0.61    23%-> 23%  0
upper_arms heavy      7.1   170  0.31   0  0.50    0     63->44     13.9-> 14.3   6.6     6.8->  7.1   4.4   0.50-> 1.15    37%-> 37%  0
lower_arms neutral    5.1   134  0.32   0  0.71    0     82->47      9.0-> 11.3  11.2     4.5->  5.6   5.7   0.32-> 0.45    25%-> 31%  0
lower_arms tall       5.9   186  0.50   0  0.68    0     79->44      8.7->  9.7   9.9     4.3->  4.8   5.1   0.34-> 0.68    22%-> 30%  0
lower_arms short      5.1   122  0.26   0  0.81    0     88->51      9.3-> 11.2  13.2     4.6->  5.6   6.3   0.14-> 0.89    27%-> 34%  0
lower_arms child      4.5   124  0.24   0  0.68    0     82->46      8.1-> 11.8  14.9     4.0->  6.0   7.2   0.04-> 1.26    26%-> 34%  0
lower_arms heavy      5.3   118  0.32   0  0.47    0     98->61     14.9-> 16.1  11.7     7.1->  8.0   5.7   0.41-> 0.38    32%-> 35%  0
```

Reading it, honestly:

* Cost: 4-12 ms per body for one kind, 8 ms for a tail and a horn pair (the 50 ms budget). A repeated build
  is a cache hit in the lab (`Engine.shaped_state`, keyed by the body and the specs' content).
* No flipped quad anywhere; the smallest cage quad area ratio is 0.23 (the cap is 0.12).
* The skin meets the root at **32-81 degrees from tangent** (it was 51-102), not tangent (the horn is the closest, 32-41; the bat on a child the farthest, 81). The fillet is two
  to three cage quads wide (the cage is about 0.28 units a quad and a root is 0.4-0.5 across), which is the
  ceiling; a Catmull-Clark level or two rounds it (the horn's creases are the body's own at L2) but does not
  close the gap to 0. The child and the angel/bat on short bodies improve least (the wing is not scaled to
  the body).
* `curv` AFTER is larger than BEFORE for the tail and the extra arms (0.27 -> 0.56, 0.18 -> 0.30): the
  curvature step at `rho = W` is the one thing the cage cannot resolve (it is exactly zero analytically for
  `k = 3`), and the body's own curvature varies by about as much. It is reported, not hidden.
* The 99th percentile of the crease the junction adds is 3-11 degrees on the twice-subdivided skin. The
  WORST single edge can rise by 60-95 degrees where the body already has a crease the junction pushes on
  (the tail sits in the gluteal cleft, the bat arm at the armpit): measured on the neutral body at L1/L2.
* The validity pass backs off up to 125 vertices (angel on a child, bat on a child or short body: `minA`
  0.26-0.31): the fillet is being clamped there, which is what the pass is for, and the place a later
  anatomy track will meet a wall first.
* The weld (the best root piece's share inside the body) rises for the tail (21% -> 33-34%), the horn
  (52% -> 66%), the bat (10-19% -> 17-26%) and the angel (12-17% -> 22-31%); a lower arm gains 3-8 points,
  an upper arm 0-2 (its weld is the buried root, not the skin). A tail at `offset = 0`, which only touches
  the skin (a hairline weld, blocked by the report without the junction), is welded by it.
* No new blocking finding anywhere, and the figure stays one overlap-connected component
  (`tests/test_junction.py::test_the_figure_stays_one_connected_component_and_the_report_has_no_new_block`).

## Subdivision

The body is subdivided (Catmull-Clark, levels 1-2) for render and print. The junction moves cage vertices
and CC rounds the result, so `levels >= 1` renders and prints the fillet smoothly (the 95th-percentile crease
in the zone falls by about half with each level: horn 20 -> 10 degrees from L1 to L2, `tests/test_junction.py::
test_the_subdivided_shaped_body_is_smooth_where_the_cage_is_coarse`); at level 0 (the CLI's default for a
picture) it is as coarse as the cage. `minimaster character --subdivide 1 -o x.stl` used to die with "cage
is not a closed 2-manifold: 38 edges" even without appendages: the two TEETH helper islands are open (38
boundary edges) and Catmull-Clark of an open cage raises. At level 0 they were dropped by the watertight
check AFTER building; `printprep.character_print_shells` now leaves them out BEFORE subdividing
(`subdiv.cage_is_closed`). Subdivided prints work, with and without appendages.

## Seen from the lab, and its limits

* The Character tab draws, picks and exports the shaped skin. During a **body-slider or shape drag the draft
  keeps the unshaped skin** (the 4k-triangle proxy averages a fillet away and reshaping on every frame costs
  12-26 ms); the final render 150 ms later has the fillet. A pose drag keeps the cached skin of the rest pose.
* The body proxy of the draft is cluster-averaged, so the fillet is only faintly there in the draft.
* Appendages that overlap the original limb (the extra upper arms start at the same shoulder) also pull the
  original limb's skin a little.
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

`--pair` puts the body without the junction above and with it below; `--skin-only` hides the appendages
(the shaped skin alone shows whether any of it is left sticking out of the appendage); `--before` is the
old look. The tests are `tests/test_junction.py` (the mechanism on a synthetic sphere, then the real body).
