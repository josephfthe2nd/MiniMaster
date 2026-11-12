# The tail as an anatomy

The first tails were a tapered tube pushed through the skin of the side of the left buttock: the anchor was
one cage quad (`3921`) whose centre is 0.36 R off the midline and whose normal is yawed ~10 degrees off the
sagittal plane, the pipe was sunk 0.6 along that tilted normal, it bent sideways (the `droop` control rotated
about the tangent that points UP), and the skin did not mould around it. A sculptor's verdict was "a shell
placed on the skin". This track rebuilds the tail the way the animal is built: **bone, muscle, skin**, rooted
on the **midline of the sacrum**, with the skin of the sacrum flowing into it.

![before (top, the S1 state) and after](images/tail_before_after.png)

Code: `minimaster/character/tailrig.py` (the model and the junction spec), `kinds.TailKind` (placement and
build), `appendages.SurfaceAnchor` / `face_anchor` (the midline anchor), `creature_anatomy.TailEffect` (what a
big tail asks of the pelvis), `ecorche.py` (the layers drawn). Tests: `tests/test_tail_anchor.py`,
`test_tail_anatomy.py`, `test_tail_junction.py`, `test_tail_ecorche.py`. Numbers: `tools/tail_report.py`.
Pictures: `tools/junction_view.py ... --layers bone,muscle,skin --cut`, `minimaster character --add tail
--layers bone,muscle --cut -o x.png --azimuth 90`.

![the default tail on six bodies: back, side, three-quarter, low](images/tail_bodies.png)

## 1. Where the tail is rooted: the midline

**The landmark rule** (vertex ids of the 13,380-vertex body cage, verified on eight morphs by
`tests/test_tail_anchor.py`): vertices `4223` (the apex, the most posterior midline point), `4215` and
`4205` (the midline edge the tail roots on) and `4387` (the top of the gluteal cleft) are exactly on `x = 0`
on every morph. The root point P is the middle of the edge `4215-4205`. In units of R = 0.0265 H (H the height of
the morphed cage) it measures, over neutral / tall / short / child / heavy / female / old / male bodies:

| | measured | rule |
| --- | --- | --- |
| P over the body's height | 0.522 - 0.581 | 0.50 - 0.585 |
| apex 4223 to P, along the midline | 0.75 - 1.12 R | 0.7 - 1.15 R |
| P to the cleft top 4387 | 0.86 - 1.34 R | 0.8 - 1.4 R |
| axis elevation above horizontal | 5.8 - 17.3 degrees | clamped to 5 - 18 |
| lateral offset of the root point | 0 to 1e-4 R (the base mesh itself is 5e-5 asymmetric) | 0 |

**The anchor.** A single quad cannot carry a midline normal, so `SurfaceAnchor` gained an optional mirror
partner (`mirror_corners`, `mirror_weights`): the bound quad PAIRED with its mirror image. The origin is the
mean of the two weighted points and the normal the sum of the two quads' Newell normals, so both lie exactly
on the sagittal plane of a symmetric body (the polygon, not an arithmetic `x = 0`: a body shifted sideways
carries its anchor with it). `appendages.face_anchor(base, quads, face, weights, hint, midline=True)` builds it;
`Appendage.anchor_midline` (default `False`: the legacy engine is untouched) switches it on. Weights over the
stored (+x) quad slide the root along the spine and stay on the plane.

**Who is midline.** `TailKind.resolve_placement` binds a `center` tail to the quad AND its mirror image when
the stored quad has an edge on the midline (`appendages.is_midline_face`); `anchor.midline: false` in the
record switches it off, and a tail clicked onto the side of the body (a quad that is not a midline quad) or a
`both`/`left`/`right` pair stays on its single quad exactly as before. The lab's anchor markers
(`Engine.anchor_points`) read the same anchor.

**Documents.** Nothing is rewritten. A record written by the first tails has the preset `tail_base`, the
anchor face `3921` (with its corners) and `offset -0.6`: `resolve_placement` reads face `3921` as the new
`tail_base` quad (`10562`, the +x quad around the midline edge) and the first preset's sink `-0.6` as the new
default `0` (`kinds.LEGACY_TAIL_BASE_FACE`, `LEGACY_TAIL_SINK`); a v1 world-point record that resolved to
`3921` is read the same way. An explicit user anchor or offset is honoured. `PRESET_FACES["tail_base"]` is `10562` and
`PRESET_POINTS["tail_base"]` is a point on the neutral cage whose nearest quad is that one
(`test_kinds` recomputes every preset from its point). The old `curl` / `droop` / `lift` / `sway` / `length` /
`base_radius` / `segments` / `ring_verts` / `tip_radius` / `flare` / `fillet` keys all still load; what they MEAN
changed (below), and that is stated, not hidden.

**Direction.** The tail's axis is the sacral surface normal at P, pitched in the sagittal plane into 5 - 18
degrees above horizontal (the first part of a tail is the continuation of the sacrum's axis, not of whatever way
one cage quad faces). This is done in `TailKind._root_frame`, not through the document's `roll`.

**The stem.** The sink is no longer an offset along a tilted normal: the tail's own geometry starts
`STEM_R = 1.75` base radii BELOW the skin point, along its own axis (the hidden bone root: the sacral apex,
Cd1 at 1.0 R). The first ring is inside the body on every body (`test_the_sink_is_the_bone_root_not_an_offset`),
the default tail is 36 % embedded (the first one: 20 %), and `placement.offset` slides the tail along its
axis (0.3 -> 24 %, 0.6 -> 11 %, +0.9 units -> tangent: a hairline weld that the printability report blocks).

**Size follows the body.** `length` (6.0) and `base_radius` (0.45) are the values on the NEUTRAL body
(`kinds.neutral_cage`, height 16.76); the model is scaled by `H / H_neutral`
(`TailKind.body_scale`), so R is 0.0268 H on every body (research: 0.0265 H; the document's own
`placement.scale` multiplies on top). A short body's tail was 0.27 of its hip span and a tall one's 0.16; now the
ratio of the tail to the pelvis is the same on every morph.

## 2. The three layers

`TailModel(TailParams)` builds all three from one parameter set, in a local frame (x up in the sagittal
plane, y lateral, z along the axis, `s` along the axis from the skin point: negative inside the body). `Placed.layers()`
returns `{"bone", "muscle", "skin"}` as meshes in the MakeHuman frame (a kind without an anatomy has the skin
only); the skin IS `Placed.mesh()`, the shell that is printed and rendered.

![bone, muscle (belly to tendon) and skin, cut on the sagittal plane](images/tail_layers.png)

**Bone** (hidden): the sacral apex, then `n = round(48 L / H_ref)` (6 - 30, 17 on the default) caudal vertebrae
from Cd1 (rostral end 1.0 R below the skin point) to the tip. Vertebra lengths rise from 0.55 of the longest at Cd1
to 1.0 at 29 % of the chain and fall to 0.35 at the tip (the longest is Cd7-8 of 26 in the macaque data,
Wakamori & Hamada 2019). Each centrum is a waisted spool of radius 0.30 of the local skin radius at the root
and 0.33 beyond a quarter of the length; the first six carry transverse processes (0.65 R long on Cd1,
shorter after), the first four a neural spine (0.25 R, none by Cd5), and chevron bones hang ventrally from Cd4 to 55 % of the chain,
tapering to nothing. The bone radius is calibrated on the cat skeleton plate (below): the research had guessed
0.40 / 0.55 of the local radius distally, which makes the bone 1.4 - 1.9 times too thick there.

**Muscle** (the sleeve): its outer surface is the skin's inward by the skin thickness, lobed where the bellies are
(`LOBE` 6 %: a dorsal median groove, two dorsal and two ventral bellies, lateral bands). It is muscle belly on the proximal
part (full to 12 % of the length, a cosine fall to nothing at 55 %: bellies arise on the sacrum, ilium and last
lumbar vertebra and act through long tendons) and tendon, fascia and fat beyond. The ecorche colours it so
(`Kind.layer_colors`): red to pale along the length, as in the kangaroo tail model. Seen at print scale the lobes are
under 0.1 mm; they justify the root width and the mound, they are not surface modelling.

**Skin**: `skin = muscle surface + skin thickness` (0.08 R at the root rising to 0.12 of the local radius, floor
0.02 units), with 40 % of the muscle lobing kept. It is the envelope, not a second shape: `test_the_skin_is_the_muscle_surface_plus_the_skin_thickness` fails if either thickness changes alone, and `test_every_layer_is_inside_the_next_one`
checks bone inside muscle inside skin, point by point, in six poses.

* **Radius law** (the registry tail's, unchanged so the print report's analytic law stays valid): `r(t) = tip + (R - tip)
  (1 - t)^0.85`, tip 0.2 R (`tip_radius` 0.09), times a **root flare** `1 + F exp(-s / 0.9 R)` held at `1 + F` inside the
  body, `F = flare = 0.35`: 1.35 R at the skin, 1.13 R at 0.9 R, within 5 % of the law by 2 R. Against the
  references (width over base width at 25 / 50 / 75 %): mine 0.83 / 0.64 / 0.45; kangaroo outline 0.79 / 0.47 / 0.34,
  rhinoceros iguana 0.87 / 0.61 / 0.40, spider monkey 0.9 / 0.8 at 50 / 75 %: between the cone-shaped tails and the cylinders.
* **Section**: an ellipse measured in the transported ring frame, wider than tall at the root (lateral : vertical
  1.08 at the skin, 1.0 at 2 R) and slightly taller than wide beyond 0.3 L (0.92). (The research's GUESS was
  1.25 at the root; with it the flared section stuck out laterally past the circle the skin can be moulded around,
  as two ears either side of the root.) `crest` adds a dorsal ridge (a lizard's) along the first 60 %.
* **Rings**: one every 0.4 R from the buried end to 4 R, then `segments` evenly; 16 vertices a ring; a rounded
  tip of three rings on a quarter sphere.

## 3. Pose: a chain, not a rubber tube

The centreline is a smooth curve whose tangent is the sum of a few angle laws (`TailModel._path`), the rings
are carried along it by parallel transport:

* **stiff root**: the first 2.5 R (the sacrococcygeal joint and Cd1-3 are muscle-bound; the cat plate's first fifth
  of the tail is within 5 degrees of the lumbar line) is straight, whatever the controls (`lift` and `sway`
  start after it, `lift` through a smooth 3 R);
* `droop` bends it DOWN in the sagittal plane by `85 degrees x droop x u^1.2` (u: 0 at the end of the stiff part, 1 at the tip);
  `curl` curls the distal 45 % UP by `45 x curl x v^2` degrees. Both are sagittal now: the first builder bent `droop`
  sideways. The defaults (0.9, 0.55) leave the tip pointing 52 degrees below the axis (24 degrees at mid-length), a relaxed tail (the research: -50..-70 degrees), 2.35 units below the root over 6 units of length;
* `lift` (degrees) rotates the whole free tail about the sacrococcygeal joint (a raised lemur tail bends about 100
  degrees there), the part inside the body does not move; `sway` flexes it sideways (+ toward +X) over the free length;
* the curvature is limited (radius x curvature <= 0.8, `BEND_LIMIT`), so consecutive rings never cross whatever the
  controls: `test_no_pose_folds_the_shell_on_its_inner_side` runs the extremes. The default tail's minimum bend radius is
  8.8 R over the first 40 % and 8 R overall (research: >= 3 R).

![lift 60, lift -30, sway 50, curl up, droop 1.8](images/tail_poses.png)

Controls keep their names and ranges (`curl`, `droop`: the sum stays in [-2, 2]; `lift` -60..90; `sway` -90..90) and
are clamped on the document path. The legacy tube (`appendages.make_tail`, the legacy `Appendage("tail")`,
`print_report.TUBE_LAWS`) is untouched; `TailKind.builder_args` still returns its arguments.

## 4. The skin of the sacrum flows into the tail

The generic junction (`junction.py`, S1) raises a collar of skin around a tube. For a tail it tented the skin
around a pipe and folded the two walls of the gluteal cleft through each other (12 new self-intersecting pairs in
the cage, 9 of them outside the tail; 30 / 64 on the subdivided body). `tailrig.junction_spec` is the tail's
own, built in the tail's local frame from the model (so it is the same on every body) and carried by the frame:

* the **skin tube**: the flared section from the buried end to 2 R beyond the skin point, inscribed radius
  (1.35 R at the root); fillet width `fillet = 1.5` base radii and height 1.0 of that (`TAIL_FILLET`, `TAIL_FILLET_HEIGHT`:
  the soft rump is the mound; the research and the references want a dorsal / lateral fillet of at least 1 R, no step, no collar);
* the **sacral rise**: a ridge `Mound` up the back from 0.2 R to 2.6 R above the root, 1.5 R wide, 0.3 R high (the erector
  spinae funnel into the dorsal tail; the dorsal contour of the rump runs into the tail line);
* two **cheek mounds** that draw the medial cheeks 0.18 R toward the axis (the gluteal masses abut the root);
* the **cleft kept**: four `Keep` spheres (0.9 R, fading over 0.6) down the midline below the root, along the cleft
  (kangaroo, cat, iguana: the ventral side meets the thigh in a crease; the natal cleft below the tail is not filled).

![the tail's skin tube, mound, cheeks and cleft on neutral, heavy and child](images/tail_bodies.png)

**Numbers** (`tools/tail_report.py`, junction on the cage, `measure_junction` at one subdivision level):

| body | H | y/H | elev. | moved | largest move | min quad area | flipped | skin meets the tail at (degrees from tangent) | junction ms | tail build ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neutral | 17.48 | 0.557 | 11.7 | 140 | 1.75 R | 0.19 | 0 | 73 -> 46 | 6.1 | 3 - 6 |
| tall | 24.71 | 0.581 | 11.3 | 164 | 1.75 R | 0.16 | 0 | 72 -> 43 | 5.3 | 3.5 |
| short | 13.82 | 0.523 | 12.7 | 118 | 1.74 R | 0.29 | 0 | 74 -> 49 | 4.8 | 3.4 |
| child | 13.11 | 0.537 | 7.8 | 150 | 1.75 R | 0.14 | 0 | 75 -> 44 | 5.1 | 3.5 |
| heavy | 17.48 | 0.555 | 17.3 | 158 | 1.62 R | 0.35 | 0 | 67 -> 53 | 6.5 | 4.1 |
| female | 16.05 | 0.540 | 9.8 | 143 | 1.72 R | 0.39 | 0 | 73 -> 42 | 6.4 | 5.5 |

(the S1 generic spec on the same tail: 91 -> 64 degrees, 12 new self-intersections). The cleft: cage triangles that cross each
other after the junction and did not before, counted outside the tail's own skin (a pair inside the tail is hidden
by the union): **0** below the root on neutral / child / heavy / short at the cage, **0** outside the tail on all six
bodies on the once-subdivided body (the S1 spec: 9 of 12 and 30). Above the root, beside the tail's wall at the dorsal midline, the
heavy body has 3 tiny ones at the cage (`test_the_gluteal_cleft_is_not_folded_through_itself`). Without the beads the child's cleft
folds (`test_without_the_cleft_beads_the_walls_do_cross`, the fail-first for it).

## 5. What a big tail asks of the pelvis

`creature_anatomy.TailEffect` replaces the flat coefficients (`hip_scale_horiz +0.45`, `hip_scale_depth +0.35`,
`buttocks_volume +0.55`, `torso_muscle_dorsi +0.30`, `measure_hips_circ +0.35`, applied even to the default tail)
by a function of size: the default tail's base radius is 0.21 of the hip-joint span, a size the pelvis already carries, so
at load 1 it asks for `buttocks_volume +0.15` (and 0.10 / 0.05 / 0.10 / 0.10 for the others), and each unit of load beyond 1 adds
`buttocks +0.35` (cap 0.6), `hip_scale_depth +0.15` (cap 0.3), `hip_scale_horiz +0.2`, `dorsi +0.2`, `hips_circ +0.15`. `TailKind.load` is
`length / 6 x base_radius / 0.45` (x the placement scale): a thicker tail costs the pelvis too. The cleft is never narrowed by a
slider: the junction keeps it.

## 6. References used, and what each one gave

Compiled library (outside the repository): `refs/` (`tail/DIGEST.md`, `TAKEAWAYS.md`); the sheets `images/tail_refs_human.png` and
`images/tail_refs_animal.png` put the public-domain / CC0 references beside the render of the same view. Two consulted images are NOT reproduced:
the Wellcome photograph of a child with a tail (its catalogue lists it as in copyright; research use only) and the CC BY-SA photograph of the
kangaroo tail model.

| id | what it gave |
| --- | --- |
| `tail/cm_Anatomical_technology_as_applied_to_the_domestic` (cat skeleton) | the tail is the continuation of the lumbar line for the first fifth of its length, then droops (the stiff 2.5 R and the sagittal droop); the bone widths 0.98 / 0.59 / 0.50 / 0.32 / 0.19 of the first free caudal at 25 / 50 / 75 / 95 / 100 % (`BONE_FRAC`, `test_the_bone_and_the_skin_taper_like_the_reference_animals`); 19 - 21 caudals; the proximal caudals short and blocky |
| `tail/cm_PSM_V08_D430_Skeleton_of_the_kangaroo` | a thin bone core in a soft outline 3 - 4 times wider along the proximal third; the soft taper 0.79 / 0.47 / 0.34; the dorsal contour runs into the tail with no step and the underside meets the thigh in a crease (the junction's asymmetry: wide dorsal / lateral fillet, the cleft kept) |
| `tail/cm_Kangaroo_tail_anatomical_model_6727490457` | muscle bellies on the proximal third giving way to long tendons under a smooth skin envelope (`belly()`, the belly -> tendon colouring) |
| `tail/cm_Superficial_muscles_of_a_cat` | the rump (gluteals) is the mound the tail grows out of; "muscles concerned in the movements of the tail" at the root: no separate collar, a low wide mound |
| `tail/ov_964703a3` (rhinoceros iguana) | a thick base, a long taper (0.87 / 0.61 / 0.40) with a dorsal crest running on: the radius law's neighbour, the `crest` parameter |
| `tail/cm_A_child_with_a_tail_side_and_posterior_view_Repr` | where a human tail really is: the sacrococcygeal level at the top of the cleft, pointing back and slightly up in profile, hanging in the cleft from behind, the skin continuous with the buttock (research use only: the licence record says so) |
| `tail/wc_dwgghrqs`, `tail/met_250897_2` | the human sacral surface: the spinal furrow into the sacral triangle, the two gluteal masses meeting at the cleft: the root sits on the furrow's end and the cleft below it is kept |
| `tail/cm_Lemur_catta_skelet_1782_Print_Iconographia_Zoolo` | a raised tail bends at the sacrocaudal joint (about 100 degrees): `lift` rotates about it, the buried part does not move |
| Wakamori & Hamada 2019 (Sci Rep 9:1473) | vertebra counts, the longest vertebra at Cd7-8 of 26 (29 % of the chain) and the proximal region of 5 - 8 |
| `TAKEAWAYS.md` section 3 | cone vs fur-cylinder taper profiles; the base 0.18 - 0.37 x hip width (mine is 0.31 x the bitrochanteric width); the tail checklist (midline, first fifth straight, no dorsal collar, one ventral crease, cleft and gluteal mounds visible) |

## 7. Tests, and what existing tests changed

92 new tests (`test_tail_anchor.py` 29, `test_tail_anatomy.py` 39, `test_tail_junction.py` 14, `test_tail_ecorche.py` 6,
`test_tail_coupling.py` 4). 28 single-line mutants of the new code (a stem that is too short, no flare, a thin muscle, bellies that
never end, no curvature limit, no stiff root, a flipped sway, uniform vertebrae, half the vertebrae, the research's bone taper, a lift that
moves the buried part, coplanar caps, a thick root skin, a cheek mound that pushes out, one cleft bead, no sacral rise, a wide root aspect,
no pitch clamp, no face / offset migration, no body scale, every quad a midline quad, unpermuted mirror weights, no mirror normal, the
wrong side cut, no sleeve colours, a flat coupling, a load that ignores thickness): 25 were caught first time and the 3 that survived
(the curvature limit, the stiff root, the aspect) were tightened until caught (a test that compared a constant with itself, a case that
never reached the limit).

Existing tests that had to change, and why (no assertion was loosened to make something pass; each states a premise the mandate overturned):

* `test_kinds`: the tail's registry-only parameters gain `vertebrae` and `crest`; the embedded share of the default tail is 36 %
  (windows 33 - 39 %, and its slides along the axis) instead of 18 - 21 % at `offset = -0.6`; the tail half of "the presets equal the
  legacy placement plus the sink" is replaced (the premise, one quad on a buttock, is the bug), the horn half is unchanged.
* `test_junction`: the tail's generic spec (the loft's rings, no mounds) is now the anatomical spec (stem first, 1.35 R flared radius,
  mounds and keep, scaled with the body); the "tangent tail welded by the junction" test slides the tail out by +1.0 instead of `offset = 0`.
* `test_character_cli`: the blocking-report test needs a tail that only touches the skin: `offset=1.0` (with `--no-junction`) instead of 0.
* `test_limbs`: the rounded-tip report test builds a LEGACY tube through the legacy engine, which needs the legacy sink
  (`LEGACY_TAIL_SINK`) now that `TAIL_SINK` is 0.

## 8. Limits, honestly

* **Guesses** (research-marked and still guesses): the vertebral spool dimensions, the processes and chevrons, the
  belly / tendon fall-off (12 % -> 55 %), the lobing, the skin thickness, the aspect, the root flare (1.35 R: `F = 0.35`); the bone
  radius is the only bone number calibrated (n = 1: one plate). No cross-section or MRI of an animal tail exists in the library.
  At print scale (1 unit = 1.9 mm) every layer above is under 0.1 mm; what shows is the placement, the axis, the flare, the
  taper, the bend and the sacral rise.
* **The seam** (now bridged by the root collar, [collar.md](collar.md): the tail's `collar` parameter, 1 by default) between the tail's shell and the moulded skin was a seam: the cage is 0.28 units a quad and the root is 0.6
  across, so the fillet is 2 - 3 quads wide; subdividing twice rounds it but the intersection of the two meshes wanders by
  about 0.04 units (a few pixels in a close-up: `images/tail_bodies.png` is at two levels). The skin meets the tail at 42 - 53 degrees
  from tangent (it was 64 - 91), not at 0. Making that seam invisible needs vertices the cage does not have (or normal blending across the
  union); it is out of this track.
* **Displacement**: the largest vertex move is 1.6 - 1.75 R (0.7 - 0.8 units at the neutral size), the climb of the collar to its
  full height at the tail wall, mostly vertices the tail covers; the research asked for 0.5 R from a smooth union, which this
  mechanism cannot do and still reach the wall. The junction touches 118 - 164 cage vertices.
* **Stretch** (the S1 review asked for it to be bounded; the limiter still only checks flips and collapse): of the 156 - 216 cage
  quads the junction touches, those outside the tail grow in area by at most 1.9x (neutral), 2.0x (child), 2.5x (short), 3.1x
  (heavy), 95th percentile 1.5 - 1.7x, and their longest edge by 2.5 - 4.0x; the S1 spec on this tail was 2.55x and 3.4x. Not
  worse, not bounded.
* The dorsal midline above the root can fold three tiny cage triangle pairs on the heaviest body (beside the tail's wall).
* `length` and `base_radius` are values on the neutral body; a document that sets them gets them scaled by the body height
  (the same tail on a taller body is longer): `placement.scale` is the absolute knob.
* The Character tab's draft proxy does not show the fillet during a drag (S1); the ecorche is a render option
  (`--layers`), not a lab view.
* Mirrored tail pairs (`side: both`) are two single-quad tails 1 - 2 R off the midline, not midline tails: they get the
  anatomical shell but not the midline frame. The research did not spec them.
* **Every existing tail document looks different** (midline root, sagittal droop, 36 % embedded, a 1.35 R flared root, a taper that
  ends in a bone-and-tendon tail): there is no switch back (S1's `ring_verts=10` restored the old mesh; here the legacy tube is
  only reachable as `appendages.make_tail` through the legacy `Appendage("tail")`). The values are migrated, not preserved.
* The first preset's sink is read as the new default by its VALUE: a record with `offset = -0.6` gets `0` (it does not matter whether
  the record is old or someone typed it); every other offset is honoured as a slide along the axis.
* `params["length"]` / `["base_radius"]` are the values on the NEUTRAL body, not the built size (that is x `Placed.model.scale`). Anything
  that reads the parameters analytically (the print report's `TUBE_LAWS` for a legacy `Appendage`) is unaffected, but would be off by
  the body scale if it were ever fed a registry tail (the registry path hands it meshes and does not).
* Merging: the shared hooks this track added (`Placed.layers()`, `Placed.model`, `Kind.layers()`, `Kind.layer_colors()` in `kinds.py`, the
  `--layers` / `--cut` hunk in `__main__.py`, `tools/junction_view.py --layers`, five lines in `lab.py`) are the likeliest to conflict with the other tracks;
  `ecorche.py` and `tools/tail_report.py` are new files and stand alone.
