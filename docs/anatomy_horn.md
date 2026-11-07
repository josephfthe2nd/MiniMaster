# The horn as an anatomy

Track S4 of the appendage junctions ([junctions.md](junctions.md) is the generic core this builds on, the
design is [appendages_system_design.md](appendages_system_design.md)). The complaint was that an appendage
"seems placed onto the model": for the horn that was literally true. The first horn was a 0.42-radius cone
pushed 0.55 into the scalp on the **parietal** eminence 0.6 behind the bregma (`temples`, quad 12116), its
root was a flat fan polygon (the "faceted cut"), nothing like a boss, and with the generic fillet a big or
flared root swelled the head and dragged the ears. The anatomy was wrong, not only the seam.

![before and after, the same horn, three close-ups](images/horn_root_before_after.png)

(Top row: `bfaaa27`; bottom row: this track. Same body, same quad count, subdivided twice.)

The horn is now built the way a real cranial protrusion is, as three layers from one set of numbers, placed on
the frontal bone from landmarks of the morphed head:

![the default pair on six bodies](images/horn_bodies.png)

| layer | what it is | where it is in the code |
| --- | --- | --- |
| `bone` | the cornual process of the frontal bone: a cone 0.82 R wide at the skin, tapering over the first 0.62 of the length (the last 0.4 of a bovid horn is solid keratin), widening into the skull below the skin, on a plate of the skull's outer table | `HornModel.bone`, `skull_plate` |
| `muscle` | the soft tissue between the skull and the skin that the pedestal tents (scalp, galea, periosteum, the germinal cuff): a lens around the cornual process, its upper surface the MOULDED skin less the skin thickness, its lower surface the outer table | `soft_tissue` |
| `skin` | the keratin sheath, the shell that is printed and rendered, with its foot (below); the scalp around it is the body itself, moulded by the junction spec | `HornModel.skin`, `Placed.mesh()` |

`Placed.layers()` returns them as named closed meshes in the MakeHuman frame; `ecorche.layer_shells` and
`tools/horn_view.py --layers bone,muscle,skin --cut` (or `--section`) draw them:

![skin cut open, the soft tissue and the bone, from the same side](images/horn_layers.png)

(Left: the sheath cut along the horn's axis, seen from inside; middle: the soft tissue lens (red) around the
core; right: the cornual process on the skull plate.)

## Where it sits (placement)

Three sites, each a quad of the cage found on the neutral head by the research and verified here on the morphed
head of six bodies (neutral, tall, short, child, old, female; the quad follows every morph, only the numbers below
are re-measured per body). They are presets of the
horn kind; `new_spec("horn", params={"shape": "ram"})` picks the shape's own site.

| preset | quad (left) | centroid (neutral) | what it is on the skull | default for |
| --- | --- | --- | --- | --- |
| `horn_crown` | 11861 | (0.473, 8.283, 0.980) | the frontal bone just in front of the coronal suture, where the superior temporal line arches over the vault; 0.19 in front of the bregma plane, 0.55 of the half-width out | curved, spiral, antelope, ossicone |
| `horn_brow` | 7314 | (0.354, 8.215, 1.216) | the frontal eminence (tuber frontale), 63% of the way glabella to bregma | short brow horns |
| `horn_temple_crest` | 11860 | (0.616, 8.011, 0.809) | the frontotemporal crest above the temporal fossa | ram |

Measured on the six bodies (`tests/test_horn_anatomy.py::test_every_preset_sits_on_the_frontal_bone_on_every_body`):
the centre is 0.41-0.82 of the head's half-width (0.861 x the eye distance over the neutral one) out, at least
0.40 of the way from the eye line to the crown, at least 0.35 head sizes in front of the ear, and the tent
never reaches within 0.075 of the midline. The first preset's quad (12116, `temples`) is on the parietal bone,
0.6 behind the bregma and behind the ear plane: a record that still names it (without anchor weights) is read as
the shape's own site, so old documents build the new horn instead of one out of the back of the skull.

**Head scale.** The eye distance over the neutral head's (0.582): 1.00 neutral, 1.29 tall, 0.88 short, 0.90
child. It follows every morph (the eye joint markers move with them) and matches the head's half-width within
3% on adults (the skull half-width over the eye distance is 0.705 on every adult body, 0.76 on a child, whose
head is relatively wide). `length`, `base_radius` and `tip_radius` are the values on the neutral head and
are multiplied by it; the base radius is never more than 0.40 of the head's half-width at the site's height and
the tent stays off the midline, so a record that says 0.6 builds a horn that fits the head.

**Axis.** Not a hint in the record: the skull's surface normal (Gaussian-smoothed over about 1.2 base radii:
a cage quad's own normal is a few degrees off) turned by `tilt` degrees toward the shape's lean direction, which
is written in the head's own axes (x out, y up, z forward, from the eye, crown and skull-base markers). The
research's rule is 15 to 45 degrees between the axis and the normal for a horn that reads as grown out of the
bone; every shape is inside it and the test measures it on six bodies (curved 27, ram 41, spiral 40, antelope 45,
ossicone 20). The bend plane contains the axis and the shape's bend direction (also in head axes), turned about
the axis by the placement `roll`. The right horn is `diag(-1, 1, 1)` of the left frame, so the pair is an exact
mirror (the mesh to 1e-12, the junction specs to 1e-9).

## The sheath

* **Radius** `r(s) = tip + (R - tip)(1 - s/L)^1.05` (the first horn's 1.25 made a thorn: its mid-shaft was 6%
  too thin), the section an ellipse with the major axis in the bend plane (minor : major 0.85 curved, 0.80 ram
  with a 0.06 three-lobe keel, 0.90 antelope, 1.0 spiral and ossicone).
* **Curve** the centreline's curvature is that of an equiangular spiral, `kappa(s) = k / (S0 - s)`: it is not zero at
  the root (the first version's `curve * t` was straight and then hooked) and rises toward the tip; the total turn
  is `curve x L / 2`, which is what the first versions' `curve` meant, so every saved value keeps its meaning
  (curved 69 -> 60 degrees, ram 252 -> 275). A lyre changes sign (out 45 degrees, then back 30); a corkscrew is
  a helix wound around the planar axis (a displacement, not a curvature: added to the curvature it made the pair
  converge and cross the midline), the ring frames carried along by parallel transport.
* **Growth rings** (annuli) ridges of amplitude `rings` x radius (default 0.03-0.06), pitch 0.8 R at the root
  falling to 0.55 of that at the tip, none on the last 16% of the length; the loft has at least six stations a
  period. They continue down to the skin on the side the horn leans away from.
* **Tip** rounded, at least 0.012 units, default 0.04-0.07 (x the head scale): 0.23 mm across at "medium" on every
  body. The old registry tip was tied to the legacy tube law of `print_report`, which a registry horn (a
  ready-made shell) no longer goes through.
* **The foot** replaces the flat cap. Below the first shaft ring the sheath is a stack of rings that FOLLOW THE
  SKIN it comes out of: they are lowered to the moulded cage along the horn's axis (`SkinField`: rays at the
  body's triangles), start 0.25 root radii under it and carry a flare that decays over 0.35 radii along the skin
  (`flare`, 1 + flare at the skin line, limited to 0.6), a groove 0.05 deep at 0.10 radii and a bead 0.07
  high at 0.34 radii (the keratin's crisp rim). A rounded bowl closes them under the scalp. The result is that
  on a tilted skull (the side the horn leans away from falls away by R tan(tilt)) the sheath emerges from the
  skin all round with no cut and no gap.

![the five shapes against references](images/horn_shapes_refs.png)

(From the top: curved against the ibex sheath A02; ram against the bighorn skull A04; spiral against the
Siberian ibex A01; antelope against the gemsbok A22; ossicone against the live giraffe A29.)

| shape | L, R (neutral head) | tilt | curve law | section | site | material |
| --- | --- | --- | --- | --- | --- | --- |
| curved | 1.6, 0.17 | 27 | spiral, Psi 60 deg, q 1.8 | 0.85 | crown | bone |
| ram | 3.0, 0.28 | 41 | spiral, Psi 275 deg, q 1.35, bends back, down and out | 0.80 + keel | temple crest | bone |
| spiral | 4.0, 0.13 | 40 | spiral Psi 40 deg + 2.25 turns of helix | 1.0 | crown | bone |
| antelope | 4.0, 0.13 | 45 | lyre: +23 deg out, then back | 0.90 | crown | bone |
| ossicone | 0.45, 0.15 | 20 | spiral Psi 3 deg | 1.0 | crown | skin |

The **ossicone** is the one shape that is skin-covered bone (a giraffe's): the whole surface has the skin
material, so its foot is the concave skin fillet of A29 (flare 0.45 ~ the measured 0.5 radii), it tapers by only a
quarter (tip 0.7 R) and ends in a dome. It is also the one that shows the mechanism best: on the same head the
skin flows into it without a colour change.

## The skin around the root (the junction spec)

`HornKind.junction` -> `hornrig.junction_spec`: the sheath's first part as the tube the scalp flows into (fillet
1 base radius, height 0.6) and the tent as a mound of height `H = max(0.35 R, 0.10)` (0.5 R for a ram) and radius
`2.2 R` (at least 0.36, three mean cage edges; at most the distance from the midline less 0.075), whose
direction is **the horn's own axis**, not the skin normal (a normal-directed mound on a skull of radius 0.6 made
the ledge across the forehead the research found), plus `Keep` spheres on the ear and the eye. Measured per body
(the table below, `tests/test_horn_anatomy.py`): the displacement outside the sheath's wall is at most 0.056 units (a quarter of a base radius) and falls to
zero by 1.7 base radii past it; no cage quad flips or is backed off; nothing farther than the tent plus a radius from the
anchor moves; the ear, the eye line and the midline are untouched. The reviewer's extremes (`flare` at its limit, `length` 8 with a base radius of 0.6, a ram at 0.6, an ossicone
at 0.6) move under 300 vertices and back off fewer than 10 (his `flare` 2 and `length` 8, radius 1.0 were 2803
vertices and 1424 backed off, 2375 and 1308).

The foot of the sheath is built against the cage **as this spec moulds it** (`hornrig.moulded_cage`, the same
displacement and validity pass as `apply_junctions` for that one spec), so its rim meets the skin the body will
really have; the soft-tissue layer is built from the same two surfaces, which is the test that the layers and
the tent agree (`test_the_soft_tissue_is_a_lens_under_the_moulded_skin_and_over_the_skull`).

**The human-skin decision** (references H15, H16, H17 against the animal ones; TAKEAWAYS section 2 warns against
borrowing the animal hair-line seam on bald skin). A bald head has three real behaviours at a horn root: skin
flush against the keratin with a faint shadow (H15 an ear horn, H17 a scalp horn), skin mounded warty around the
base (H16), and the root hidden in hair (H19, H24, H25: the Moses horns). With no hair the third is out. The skin
is therefore **mounded and meets the keratin at a crisp edge**: a low tent (0.35-0.6 R) under a foot whose groove
and bead are the rim. That is the bovid mechanism (A06: the hatched skin tucked around the sheath's rim, no
fillet, A22: the keratin meets the skin with no collar) with the human skin's mound (H16) under it. (The bovid checklist
of the reference takeaways says no fillet: the foot's flare, 1.15 R at the skin line relaxing over 0.35 radii, is the
sheath's own periople-like base, not a skin fillet, and `flare` 0 removes it; the skin's own fillet is the junction's
and is a mound, not a collar.)

## Documents

Nothing in a saved document is rewritten and every old record builds (`tests/test_horn_anatomy.py::test_the_first_versions_temples_quad_...`).
The parameter names are the old ones (`shape`, `length`, `base_radius`, `segments`, `ring_verts`, `curve`, `twist`,
`turns`, `tip_radius`, `flare`, `fillet`) plus `rings`, `tilt` and `boss`; `shape` has a fifth value, `ossicone`.
Their numbers are the values on the NEUTRAL HEAD and are scaled by the head: a record that stored the first
version's defaults explicitly (`base_radius` 0.42) builds a much bigger horn, limited to 0.40 of the head's
half-width and to what keeps its tent off the midline (about 0.24 on the neutral head). A record without
parameters builds the new defaults. `placement.offset` is a slide along the axis now (the first preset's -0.55
means "the anatomical root", read as 0), `placement.roll` turns the bend plane, `up_hint` is not used (the axis comes
from the skull; the key stays in the file). `HornKind.placement_keys` is `anchor, scale, roll, offset`.
The legacy engine (`appendages.make_horn`, `Appendage("horn")`) is untouched and bit for bit what it was.

## Numbers

`HornKind` on six bodies (made by a measurement script of this track; neutral = default macro;
weld = share of the horn's volume inside the body from the printability report at "medium"; build = a pair,
including the junction spec and the moulded cage; the CLI path adds the 7-10 ms of `apply_junctions`):

```
shape     body     k      R      R/hw tilt  tent r x H   moved maxmv  limited  weld  blocking  build
curved    neutral  1.000  0.170  0.28 27.0  0.37 x 0.100   46  0.177     0   0.32     0     ~90 ms
curved    tall     1.288  0.219  0.29 27.0  0.48 x 0.129   46  0.230     0   0.32     0
curved    short    0.876  0.149  0.28 27.0  0.33 x 0.088   46  0.156     0   0.32     0
curved    child    0.902  0.153  0.26 27.0  0.34 x 0.090   44  0.160     0   0.32     0
curved    old      0.999  0.170  0.28 27.0  0.37 x 0.100   48  0.178     0   0.32     0
curved    female   0.991  0.169  0.28 27.0  0.37 x 0.099   48  0.176     0   0.31     0
ram       neutral  0.999  0.246  0.35 41.0  0.54 x 0.123  168  0.251     0   0.30     0    ~110 ms
ram       tall     1.288  0.329  0.37 41.0  0.72 x 0.165  190  0.337     0   0.32     0
ram       child    0.902  0.247  0.36 41.0  0.54 x 0.123  194  0.251     0   0.35     0
spiral    neutral  0.999  0.130  0.21 40.0  0.36 x 0.100   44  0.149     0   0.14     0    ~150 ms
antelope  neutral  0.999  0.130  0.21 45.0  0.36 x 0.100   46  0.147     0   0.17     0
ossicone  neutral  0.999  0.150  0.25 20.0  0.36 x 0.100   48  0.164     0   0.63     0     ~80 ms
```

(`moved` and `maxmv` count a PAIR, and the largest move is of the vertices INSIDE the sheath's tube, which are
hidden; outside the wall it is at most 0.056. The ram's tent is on the lateral head, where the cage is coarser,
so more vertices move; `R/hw` is limited to 0.40.) Clearance, from the sheath outside its foot to the head: at
least 0.5 head units on every shape and body, and 0.9 from the ear for the ram (the research asks 0.08 and 0.15).
The embedded share of the default horn's volume (Monte Carlo against the UNmoulded body, 60000 samples) is 15.7%
(the first horn's was 65%); sunk 0.1 deeper 18.5%, slid 0.1 out 13.5%. Against the moulded body, which is the one
that prints, the printability report's weld is 32%.

## What was taken from which reference

Ids are those of the reference library (`refs/horn/`, [appendage_references.md](appendage_references.md)); a number
is MEASURED by the gatherers on that image (one image each, 5-15% error) unless marked.

* **A05** (goat skull, scale bar): the core is a linear cone, 0.75/0.50/0.25 of its base at 25/50/75% of its
  length, base diameter 0.11 of the skull length, d/L 0.20 -> the taper exponent 1.05, `CORE_BASE` 0.82 and
  the 0.62 core fraction (the research's guess for the core, the cone itself is measured), the base radius as
  a share of the head.
* **A06** (the only layer diagram): stippled sheath over a cross-hatched core, skin tucked around the rim as a
  collar -> the layer order and the crisp keratin/skin edge (groove and bead of the foot).
* **A22** (gemsbok, live): the horns rise side by side from the top of the head, the annuli start right at the
  base, the keratin meets the short skin with no collar -> rings from the skin line, the crown site, the antelope
  shape (kept more splayed than a gemsbok's: a lyre).
* **A29** (giraffe ossicones): width 1.0/0.84/0.78/0.73 over the height, dome tip, concave fillet 0.5 base radii
  -> the ossicone shape, its tip 0.7 R and flare 0.45.
* **A20, A21, A18** (deer): the pedicles sit above the eyes, beams lean 23-26 degrees off vertical, the burr 1.3
  x the neck -> the crown site and the lean; `offset` slides a horn out onto a taller foot (a pedicle), but the antler
  itself (burr, beams, tines, velvet-to-bone colour) is not built.
* **A04, A31** (bighorn skull and sheath): massive base, the curl back, down past the ear and forward -> the
  ram's 275 degrees of turn and its bend direction (the research's own bend vector curled in over the cheek; the
  one used bends outward, 0.45 out, -0.65 down, -0.55 back).
* **A02, A01, A11** (sheaths): the ridged sheath, the open flared mouth, L/base 3.2 (A11, measured) -> the
  rings, the foot flare, the proportions (curved 4.7, ram 5.4).
* **H01-H03** (the human skull): glabella-bregma 0.53 of the skull length, the superior temporal line over the
  vault -> the three sites. **H05, H10**: the temporalis fills the fossa BELOW that line, so no muscle belongs at the
  crown or the brow site (the `muscle` layer is soft tissue, honestly not a muscle). **H13**: the order skin,
  subcutis, galea, periosteum, skull -> the soft-tissue lens.
* **H15, H16, H17, H19, H24, H25**: the human skin behaviours (above). **H24**: Moses' horns are about a fifth
  of the head height: the short end of `length`.

## Tags: measured, derived, guessed

MEASURED on the mesh (this track): the sites and their stability over six bodies, the head scale, the tilts,
R/hw, the tent and the displacements, clearances, welds, build times, the ear position (found as the widest part
of the head below the eye line: (0.81, 0.18, -0.06) head sizes from the skull base, the same on every adult).
MEASURED on references (single images): the core taper (A05), L/base (A11), ossicone taper (A29), burr/neck (A18).
FROM THE RESEARCH (derived there from sources and sphere fits): the sites' normals, the 15-45 degree rule, the
equiangular spiral and its Psi per shape, the scalp thicknesses (3-8 mm, 6.7 mm), the tent numbers.
GUESSES (no source): `CORE_BASE` 0.82 and the keratin wall (0.12 R minimum), `FOOT_HEIGHTS`, the groove and the
bead, the ring amplitudes, the 0.04-0.07 tips, `HELIX_RHO` 2.0, the ram's bend vector (changed from the research's
by looking at it against A04), the skin thickness 0.06 and `PLATE_GAP`, `TENT_MIN` 0.10 (it is what makes the tent
bigger than 0.35 R on a head: 0.6 R at the default).

## Decisions the references disagree on

* **Size.** The reference takeaways give a realistic human cutaneous horn a base of 0.10-0.15 of the cranial width
  (H17) and a stylised print 0.30 (H22). The default (curved) is 0.34 across on a 1.22 wide skull at the crown, 0.28
  of its width and 0.20 of the head's width with the ears (the research's number for devil / tiefling horns); the
  ram is 0.57, the antelope and spiral 0.21 (R/hw in the table). `base_radius` goes down to 0.05 (0.07 of the
  width), so a realistic horn is one parameter away; the default is the fantasy size the kind is for.
* **Ossicone proportions.** H/W 1.25 (A29, foreshortened) against 1.75 (a silhouette): the default is 1.5 (0.45
  long, 0.3 wide) with the taper and the dome of A29. The pair is 3.1 widths apart, not the giraffe's 1.2-1.5: the
  crown site is 0.47 out, and the tent's 0.075 margin from the midline stops a pair from standing closer (a `poll`
  preset next to the midline would need a smaller tent there).
* **Site on the temple.** The checklist says a root on the temple needs the skin to bulge over a muscle mound
  (H05, H10, H14). The ram's `horn_temple_crest` is on the frontotemporal crest, just above the temporal fossa (the
  research's footprint rule), and has no temporalis mass: the tent is the only bulge.

## Limits (stated plainly)

* **One material per shell.** The lab, the CLI and the print path draw `Shell(mesh, placed.material)`, so the
  foot, which is skin-coloured in life, is ivory beside the skin; a hairless collar and the lighter, glossier
  periople band cannot be shown. The boss is geometry. (The ossicone, with the skin material, shows what it would
  be.)
* The foot's contact line with the skin is only as smooth as the skin: the horn is high resolution, the body's
  cage is 0.125 an edge; at level 0 it is the cage's facets, at 1-2 it wavers by the sagitta (0.01).
* `base_radius` above 0.40 of the half-width is silently limited (saved records with the first version's 0.42 build at
  about 0.24 on the neutral head); a warning would break the "every parameter at its limits" test.
* The pair is an exact mirror of the left horn: on a body with an asymmetric head the right horn does not follow
  its quad (the base mesh is 5e-4 asymmetric on a child).
* A record with explicit `anchor.weights` on quad 12116 is not migrated and stays on the parietal bone.
* No antler / pedicle shape, no temporalis mass (the research puts none at these sites), no hair.
* `print_report.TUBE_LAWS["horn"]` no longer applies to a registry horn (a ready-made shell): the thin-tip and
  thin-length findings come from the area probe, the tip's width is fixed by the model.
* Build is 80-230 ms for a pair (the loft kinds were 1-7 ms): the foot casts rays at the cage; the lab caches it.

## Looking at it

```
python tools/horn_view.py OUT.png "horn;horn:shape=ram" 0,10 50,15 90,8 --bodies neutral,tall,child --levels 1
python tools/horn_view.py OUT.png "horn:side=left" --section --r 0.6        # the layers, cut along the horn's axis
python tools/horn_view.py OUT.png "horn:shape=ossicone" 0,10 50,15 --ref horn/A29    # next to a reference image
```

`--bodies` takes neutral, tall, short, child, heavy, old, female, thin or macro values; `--ref DIR/ID` puts a
reference image (`$MM_REFS/horn/images/A29_*`) beside each row. The tests are `tests/test_horn_anatomy.py` (the
model needs no assets; the rest builds horns on six morphed bodies).
