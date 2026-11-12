# The root collar: where the skin meets an appendage

Design context: [junctions.md](junctions.md) (the skin junction: the cage displacement that gives the skin MASS
around a root), [anatomy_tail.md](anatomy_tail.md) (the first kind to use the collar), [appendages_system_design.md](appendages_system_design.md).
Code: `minimaster/character/collar.py`, tests `tests/test_collar.py`, numbers `tools/collar_report.py`.

![the tail root before (top) and with the collar (bottom): neutral, tall, heavy, child; rear, three-quarter, low, side](images/collar_before_after.png)

## Why a collar

The round-1 reviewers all named the same thing: the skin and the appendage meet along a ring. The cause is
geometric, not a tuning problem. The junction moves the vertices of the 13,378-quad body cage, which gives the skin
mass around a root, but

* a cage quad is about 0.28 units across and a root is 0.4-0.6 across, so a fillet is two to three quads wide;
* the body can never get new vertices (1,280 morph targets address it by index);
* so the skin meets the root **53-100 degrees from tangent** (measured here: median 56-80 degrees between the skin
  and the tail wall at the exit, `before` in the table below) and every close-up shows a crisp ring, a flange.

The collar does not depend on the cage. It is a closed shell on the APPENDAGE's own geometry (72 around x about 30 stations,
16 of them the visible sheet, whatever the body is subdivided to) whose outer rim lies on the shaped skin, tangent to it, and that rises
along the wall to meet the wall tangentially. The slicer unions it with the body and the appendage like every other
overlapping shell (the print report counts it, see below).

## The construction

For every angle theta around the appendage's axis (`collar.build_collar`):

1. **The exit curve.** `X(theta)` is where the wall's generator pierces the skin: a ray from the axis against the REAL
   appendage mesh (not an idealised tube: the tail's wall is an ellipse, flared, a 16-gon), a coarse scan, then
   secant steps on a closest-point signed distance to the subdivided body. The wall's radius is corrected for the
   polygon of the rings (a low pass over the ring at one station: the 16 facets' normals are up to 11 degrees from the
   shading normals, and a collar that follows them shows a scalloped seam).
2. **A geodesic-ish walk over the skin.** From `X` outward in the direction away from the axis (a step along the
   tangent, a closest-point projection back onto the surface, the direction carried by projection): the ring stations
   `P(rho)` lie ON the subdivided skin and follow the curved sacrum. The first march at the full width also tells where
   the surface goes (see "the asymmetry").
3. **The profile** `top(rho) = P(rho) + h(rho) D(rho)`. `h = H (1 - sqrt(1 - x^k))`, `x = 1 - rho / W`: vertical at the
   wall, so the sheet leaves the wall along the wall (G1), and flat at the rim with `h' = 0`, so it leaves the skin
   along the skin (G1). `D` is the skin normal at the rim and turns to the wall's generator at the wall. Stations are
   spaced half equally in arc length and half equally in turning angle, with a short chord next to the wall: the renderer
   shades with area weighted vertex normals computed per shell, and the vertex at the join averages the faces on both
   sides, so the first chord next to the wall must be nearly parallel to it (with equal arc length the first chord
   was 18 degrees off).
4. **No side wall.** The sheet is carried ON past both visible edges: past the rim it dips under the skin (`sink`,
   graded so its first faces are 1-3 degrees off the skin), past the wall it curves INTO the appendage (arcs, then a fan).
   Only then is it closed to the underside (the skin pushed down by `sink`). Every vertex whose normal is ever seen has
   only top-sheet neighbours; a vertical closing wall at the rim would tilt its normal by 20 degrees and draw the ring
   this module exists to remove. The shell is a topological ball: watertight, outward, no degenerate faces.
5. **A small ramp.** Exactly tangent at the rim means coplanar with the skin's facets over a band of 0.1-0.2 units
   (the collar's chords bridge four skin triangles): in the visibility-buffer rasteriser the two surfaces z-fight there and
   where their shading differs (a fat body's lateral skin) the edge is ragged. The sheet leaves the skin at
   `ramp = 0.035` (2.0 degrees), 0.003 clear of the facets 0.1 units in: invisible as an angle, and the band is gone.
   A positive `lift` (a step) was tried and shows as a hairline; `lift = 0` and the ramp are what ship.

### The asymmetry (the anatomy)

A symmetric collar fills the natal cleft and is the blister the critics named. The references do not do that:

* `tail/cm_Superficial_muscles_of_a_cat`, `tail/cm_PSM_V08_D430_Skeleton_of_the_kangaroo` (3 % deviation measured in the
  digest): the croup and the dorsal line run into the tail tangent-continuously, no step, no collar;
* the same plates, `ov_964703a3` (rhinoceros iguana), `ov_a9455ce0` (monitor): the underside meets the thigh in a crease;
* `skin/wm_leonardo_shoulder_photo` (digest 1b): 0.11 root radii on the fold side, more than 1 radius on the open side.

So `CollarProfile.width` (1.6 root radii) is the open side (`up`, the back and the flanks) and `ventral` (0) the fold side,
blended between `open_deg` (60) and `fold_deg` (115) from `up`: on the open side the sheet is 0.87 units wide and 0.28
high on the neutral body, on the underside a hair (0.006 units: no collar, the cleft stays open). Besides the profile
the width is cut by what the first march finds: where the wall leans over the skin (the underside of a lifted tail: no
room for a fillet; `LEAN_FROM_DEG`..`LEAN_FULL_DEG`), where the skin is a crease (consecutive normals turning more than
18 degrees or a step leaving the tangent plane by 28: the cleft, a fat fold), where the exit curve could not be fitted
to the skin (`FIT_TOL`), and where the finished rim is more than 15 degrees off the skin's tangent plane (the sheet is
built again, narrower).

![the collar in red, the tail in blue: rear, three-quarter, low](images/collar_parts.png)

## The API

```python
from minimaster.character import collar as CO

CO.CollarProfile(width=1.6, ventral=0.0, height=0.5, sink=0.2, falloff=2.5, lift=0.0, ramp=0.035,
                 open_deg=60, fold_deg=115, up=(0, 1, 0), material="skin", strength=1.0,
                 segments=72, rings=14)            # lengths in ROOT RADII; immutable, hashable (digest())

pieces = CO.collar_pieces(base, shaped_verts, placed, controls=None, levels=2, skin=None, profile=None)
# -> [(name, Mesh, material)] in the MakeHuman frame like Placed.mesh(); [] = no collar
shells = CO.collar_shells(base, shaped_verts, placed_list, levels=2, skin=None, controls=None, enabled=True)
# -> [(placed, name, Mesh, material)]: what the three pipeline sites call; a collar that fails is a warning, not an error
report = CO.join_report(base, shaped_verts, placed, ...)       # the numbers below
mesh   = CO.build_collar(skin_patch, points, radii, wall_mesh, profile)    # the geometry alone, any skin, any wall
```

* `Kind.collar_profile(placed)` (`kinds.py`, default `None`: no collar) and `Kind.collar_axis(placed, controls)` (default: the
  junction spec's centreline and radii). `TailKind` returns `CollarProfile(strength=params["collar"], up=(0, 1, 0))`.
* `shaped_verts` are the body AFTER `junction.apply_junctions`; `levels` is the subdivision the body is MESHED with. The collar
  is fitted to that surface (a collar fitted to level 2 stands off a level-1 body by the Catmull-Clark shrink, which is
  largest exactly in a fillet): `skin=` takes the body mesh a pipeline has already built (`collar.skin_from_shell(mesh)` turns
  the Z-up shell back into the MakeHuman frame), else `shaped_skin` subdivides the cage once per (body, level), cached.
* `controls` is the pose: the collar is built on the POSED wall (a lifted tail, a swung arm), while the cage junction of a
  pose drag stays the rest pose's (lab). The tail's stiff root means curl, droop and sway leave the root collar as it was.
* The tail's parameter `collar` (0 = off, default 1.0, up to 2) scales width, height, sink and the radius it is measured in.
  The collar grows around the junction spec's centreline, so a tail whose skin `fillet` is 0 (no junction) has no collar either.

### The pipeline

| where | what |
| --- | --- |
| CLI PNG (`minimaster character --add tail --subdivide 1 -o x.png`) | the collar is one more `Shell` with the skin material, fitted to the body at `--subdivide` (level 0: none, a note) |
| CLI STL | the collar is one more member of the appendage's group (`tail-1/collar`), unioned by the slicer, counted in the report |
| `--no-collar` / `--collar` | `anatomy["collar"] = false` (saved with `--save-character`, written only then); `--no-junction` disables the collar too (`Character.collar_enabled`) |
| lab `Engine.final_scene` / `export` | `Engine.collars(...)`: cached by (shaped body, spec, pose, level, profile), final quality and export only; `LabDocument.set_anatomy(collar=)` |
| lab draft (orbit, drags) | skipped, and says so in the frame's notes ("the root collar ... is drawn in the final render only") |
| print report | a `.../collar` piece of the appendage's group: checked as a shell (closed, outward) and in the connection and the volume; **not a root** (it is embedded by construction: a hairline-welded tail would pass the weld rule through it) and **not thickness-probed** (its own thickness is not a printable feature: the union is as thick as the body and the tail) |

## Numbers

`python tools/collar_report.py --bodies neutral,tall,short,child,heavy --levels 1,2 --poses rest,curl,droop,lift`. Neutral = gender 1,
muscle 0.55; tall / short = height 1 / 0; child = age 0.1875; heavy = weight 1, muscle 1. `rim` / `wall`: the angle between the
renderer's vertex normal of the collar and the skin's (rim) or the tail's shaded normal (wall), degrees, over the part of the
collar that can be seen. `before`: the angle between the skin and the wall where they meet without the collar (S1b's "join
angle": 90 = the skin runs into the wall). `min h`: the lowest the visible sheet gets over the skin.

```
body     L pose   rim p50/p95  wall p50/p95  before   min h  crease  self  seen    ms
neutral  1 rest  3.4/ 7.0 2.9/ 4.9 63.6 -0.0000 22.9 138 61% 140
neutral  2 rest  2.7/ 4.8 3.0/ 4.8 63.9 -0.0000 17.4 152 62% 149
neutral  1 lift  3.3/ 6.3 5.2/ 13.2 67.8 -0.0000 24.4 128 61% 100
neutral  2 lift  2.6/ 6.0 5.4/ 13.0 71.0 -0.0000 20.7 314 62% 155
tall     1 rest  3.1/ 7.2 2.7/ 4.9 56.1 -0.0000 22.0 84 64% 127
tall     2 rest  2.8/ 4.6 2.8/ 4.9 56.1 -0.0000 16.2 62 62% 119
tall     1 lift  2.7/ 5.3 5.3/ 12.4 65.6 -0.0000 33.2 621 62% 118
tall     2 lift  2.9/ 5.0 5.4/ 11.7 63.9 -0.0000 27.7 493 62% 166
short    1 rest  3.1/ 7.5 2.7/ 4.7 76.5 -0.0000 24.8 151 60% 105
short    2 rest  2.1/ 4.7 2.8/ 4.6 76.7 -0.0000 21.6 143 62% 192
short    1 lift  2.8/ 7.8 4.6/ 8.9 79.3 -0.0000 28.2 292 61% 98
short    2 lift  2.0/ 5.2 4.5/ 8.0 79.8 -0.0000 20.7 138 62% 161
child    1 rest  3.2/ 4.7 3.1/ 5.3 57.8 -0.0000 22.1 149 60% 157
child    2 rest  3.3/ 7.4 2.8/ 5.4 58.6 -0.0000 18.7 147 62% 186
child    1 lift  2.6/ 3.8 4.8/ 13.8 66.0 -0.0000 29.7 145 60% 87
child    2 lift  2.6/ 3.8 4.9/ 11.8 64.8 -0.0000 21.1 141 60% 112
heavy    1 rest  2.8/ 9.3 3.9/ 7.4 74.2 -0.0000 32.2 158 60% 144
heavy    2 rest  3.3/ 7.2 4.0/ 7.5 73.6 -0.0000 25.9 207 54% 149
heavy    1 lift  3.6/ 9.5 5.2/ 12.6 78.3 -0.0000 36.4 384 60% 117
heavy    2 lift  3.6/ 10.7 5.6/ 12.9 78.8 -0.0000 28.8 392 54% 144
```

`curl` and `droop` rows are identical to `rest` (the stiff root: the collar is the root's, a bend starts 2.5 radii out; `sway` likewise); `lift 70` is in
`tools/collar_report.py --poses lift70`.

![the tail at rest, lifted 40 and 70 degrees, swayed 40: three-quarter and side](images/collar_poses.png)

Reading it honestly:

* **The join is tangent where it can be seen.** Median 2-4 degrees on the rim and the wall, p95 3.8-9.3 on the rim and 4.6-7.5 on the wall
  at rest (a lifted tail: rim 3.8-10.7, wall 8-14), on every body and both levels (target 10), where the skin met the wall at 56-80 degrees (median) before. The
  visible sheet is never under the skin (`min h` 0.0000: the lowest vertex is on it).
* **Only the part that can be seen is measured.** `seen` is the share of the ring wider than 0.04 units (0.08 mm at medium): 54-64 %.
  The rest is the fold side, a hair 0.006 wide that hugs the wall (and, on the heavy body, the fit failures below).
* **Curl, droop and sway cannot change the root** (the first 2.5 radii of the tail are stiff): identical to rest. A lift changes the wall's
  angle to the skin and the join at the wall (p95 8-14 vs 4.6-5.3 at rest on the neutral, tall, short and child bodies).
* **Self-crossings are not zero.** 62-621 pairs of crossing triangles (of ~10,000) in the hair-thin fold zone at the apex of the cleft,
  where the exit curve jumps 0.2 units along the wall between neighbouring columns of a grid whose bands are 0.001 thick. They are hidden
  inside the union (the visible sheet is clean); a strict "no self-intersection" claim cannot be made for the shell. A real fix is to make the
  fold side an open sector closed by caps instead of a thin ring.
* **Cost**: 87-192 ms per root for `collar_pieces` with the body mesh given, median about 125 ms on the development machine (**above the
  100 ms asked for**; a quarter of it is the 72 x 25 grid's closest-point queries, a quarter the rays against the wall, 28 ms the first
  patch of the body). The first collar of a body also pays the Catmull-Clark subdivision of the cage (0.05 s at level 1, 0.26 s at level 2),
  shared by every appendage and cached per body and level. The lab pays it once per final-quality edit, never in a drag.

## What does not work yet (the honest list)

* **A faint rim is still visible in some views.** At the flanks of the collar, from the rear and the low view, a thin line
  where the sheet leaves the skin (the 2-degree ramp): faint on the neutral and child bodies, more visible on the tall body (a
  wavy line where the tail joins the collar at the dorsal join) and on the heavy body, whose sacral dimples the rim crosses and
  whose upper edge looks nibbled from the rear. These are the remaining "ring" the user would see in a close-up.
* **The collar covers, it does not model.** The round-1 critics' second finding (the "muscle" is a skin tube, the skin is not the
  envelope of a mass) is not touched: the collar hides the junction's cage fillet under a smooth flare, which makes the tail look
  grown rather than plugged but does not give it the cat's croup or the human sacral triangle. Together with the junction's sacral
  mound the root is a bulge; its height is the sum of the mound and the collar (up to 0.5 units on the neutral body).
* **The fold side is a hair, not a crease.** On the underside the collar is 0.006 units wide: the cleft stays open and the
  crease is the body's own. In a very close low view the apex of the cleft shows a few tiny tabs at the wall.
* **The heavy body is the weakest result.** Its closed cleft defeats the exit fit on a few columns of the underside (the signed distance cannot be trusted between
  two touching cheeks: no collar there, `FIT_TOL`); its sacral dimples and skin folds make the rim wander (the visible sheet's rim p95 is 7-11 degrees, the
  highest of the five bodies), and from the rear its upper rim looks nibbled and its lateral base crinkled (`images/collar_before_after.png`, row 6).
  The tall body shows a wavy line where the tail joins the collar at the dorsal-lateral side in the three-quarter view.
* **Horns, wings and extra limbs** do not have a collar (the next stage enables them: `collar_profile` and, where the tube is not
  the skin tube, `collar_axis`). The generator-along-the-wall construction is exact for a tube leaving the skin within about 60 degrees
  of its normal; a wing arm or an extra arm that leaves at a shallow angle has an acute side where the wall leans over the skin
  and the profile cannot be a graph over it (the width is cut to the floor there: `lean`).
* **There is no collar at level 0, and level 0 is the default.** The CLI's `--subdivide` defaults to 0 and the Character tab's final preview and
  export run at `levels=0` today (`character_panel.py`: `Lab(..., levels=0)`): the body there is the 0.28-unit cage, a coarse polyhedron, and a collar fitted
  to its facets stands off it in pink flaps (rendered: `--add tail` at the default level with the collar looked WORSE than without, a ragged join
  and a floating nub). `collar_pieces(levels=0)` therefore returns nothing (`MIN_LEVEL = 1`), the CLI prints "the root collar needs a subdivided body
  (--subdivide 1 or 2)" and the lab's final scene carries the same note. So a user who types `minimaster character --add tail -o x.png` still sees the
  old seam: the collar shows with `--subdivide 1` or `2`, in the STL at those levels, and in the tab once its `levels` is raised (one line in
  `character_panel.py`, not changed here: it also costs final-render time).

## Disclosure: what changes in a saved document

* **Every saved tail now renders with a collar at `--subdivide` 1 or 2** (not at level 0) (`params["collar"]` is absent in old records and defaults to 1.0): a close-up of a
  tail root looks different (the flange is gone, replaced by a flare up to 0.87 units wide on the back and 0.28 high), in the picture,
  the STL (a second shell, `tail-N/collar`, +5 % of the tail group's triangles at level 1) and the print report (two shells in the group).
  `--no-collar` (saved as `anatomy["collar"] = false`) or `collar=0` on the spec gives the old look; `--no-junction` does too.
* A document that does not use it is byte-identical (`anatomy["collar"]` is written only when it is off; the parameter is not written
  into records that never set it). No `.mmchar` migration is needed: v1 and v2 files load as before.
* The print report: the group of a tail with a collar has 2 shells, the weld (`root_embed`) and the thickness numbers are the tail's own
  (unchanged), the volume counts both.

## Findings along the way (what had to be found, for the next stage)

* **z-fight and shading at a tangent rim.** Coplanar surfaces of the same material do not show a band if their shading agrees; a
  step (`lift`) shows as a hairline, an angle of 3 degrees does not (above).
* **Per-shell normals pollute a joint.** The renderer's vertex normals are area weighted over the shell alone (and so is the cavity
  AO): a closing wall at the rim, or a chord next to the wall that is 18 degrees off it, tilts the normal of a visible vertex.
  Both closing parts are hidden and the stations are graded.
* **A tube leaving a fillet-climbed skin.** The cage junction's skin climbs the wall by up to 0.5 units, so the wall's exit curve is
  0.8 units long along the axis (the tail leaves 11 degrees up and the skin around it is a dome): the collar must start where the wall
  meets THAT skin, not the original one.
* **Resolution independence has a limit:** the closest-point surface is the body at the subdivision it is drawn at (level 1 and level 2
  differ by up to 0.02 in a fillet); `collar_pieces(levels=)` takes it.

## References used

| id | what it gave |
| --- | --- |
| `tail/cm_Superficial_muscles_of_a_cat` (`collar_refs.png`, top left) | the croup (gluteus medius, "prominence of the hip bone") running tangent into the tail; the tail base soft and tapering: the open side of the collar is wide and flush, there is no separate collar ring |
| `tail/cm_PSM_V08_D430_Skeleton_of_the_kangaroo` | the dorsal contour of the rump continues into the tail with 3 % deviation (digest), the underside meets the thigh in a crease: `width` vs `ventral` |
| `tail/ov_964703a3`, `tail/ov_a9455ce0` | the same one-sided pattern on the iguana and the monitor |
| `tail/wc_dwgghrqs` (`collar_refs.png`, bottom left) | the human sacral triangle and the two gluteal masses that meet at the cleft: the fold side keeps the cleft open |
| `skin/wm_leonardo_shoulder_photo`, DIGEST 1b | 0.11 root radii of rounding on the fold side, more than one radius on the open side: the widths |
| `skin/wm_giraffe_head`, `skin/ov_giraffe_ossicones` | skin-covered cores flare into the head with no seam: the join is tangent, with the appendage's own flare, not a ring (for the horn stage) |

![the references beside the render of the same view](images/collar_refs.png)

## Tests

`tests/test_collar.py` (51 tests, fail-first): a synthetic sphere with a 16-gon tube (rim on the surface, tangent to it within 8 degrees,
joined to the wall on the wall and tangent to it, sunk by the asked amount, closed and outward, the hidden parts inside the tube, mirror
symmetric, narrower on the folded side, none where the tube misses the skin or is buried, a lean takes the width away); then real tails
on five bodies x poses x levels 1 and 2 (p95 <= 10 on the rim and the wall, never under the skin), the pose follows the wall, the level
follows the surface, the print report (the collar is in the group but not the weld or the thickness; a hairline-welded tail still
blocks), the CLI (`--no-collar`, `--no-junction`, saved in the document), the lab (final only, cached, refits on pose and level, the
draft says so). 17 single-line mutants of the collar, the report, the document and the pipeline were run against them
(`scratchpad/junction/S6_out/mut6.py`): 16 caught (the facet low-pass needed a test of its own: it survived the first run); one mutant, the pose taken out of
the lab's collar cache key, survives and is an equivalent mutant (the shaped body's key already carries the pose through the junction spec, so the pose
cannot reach a stale collar that way).
