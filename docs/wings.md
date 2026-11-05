# Wings

A wing is not a shape. In both bats and birds it is a **modified forelimb** —
humerus, radius/ulna, wrist, and then a hand that carries the flight surface.
Everything a viewer reads as "wing" comes from that skeleton: the Z-shaped arm,
the finger struts fanning through a membrane, the scalloped free edge, the
shingled feathers. So `minimaster.character.wings` builds the skeleton first and
hangs the surface on it, and `minimaster.character.wingrig` turns skeleton plus
surface into a **rig**: named joints, skin weights, and poses. Folding, raising
or flapping a wing rotates joints; it never rebuilds a mesh.

![wings](images/wings.png)

*A plain body, then bat/dragon (membrane) wings, angel (feathered) wings, and
the same two folded — a pose of the rig, not a different mesh. Top: from behind.
Bottom: profile.*

![the rig](images/wing_rig.png)

*One wing, no body, through rest / half fold / folded / raised 45° / lowered 40° /
fingers curled 35°, in plan, 3/4 and front. Top: membrane. Bottom: feathered.*

```python
from minimaster.character import appendages as A

app = A.back_wings("membrane")                 # or "feathered"
shells = app.build(base, verts)                # [("wing_l", Mesh), ("wing_r", Mesh)]  (spread)

rigs = app.build_rigs(base, verts)             # [("wing_l", PlacedWing), ("wing_r", PlacedWing)]
left = rigs[0][1]
left.pose("upstroke")                          # a named pose ...
left.pose(fold=0.6, raise_=20, curl=10)        # ... or the controls
left.rig.armature                              # the joints, as a core Armature
```

## Where the numbers come from

The proportions are fitted to measurements of real animals, not remembered: CC-licensed
photographs of Indian and grey-headed flying foxes (*Pteropus*) — a mounted skeleton
from two angles, gliding and spread-from-below flight shots, a roosting wrap — and of
a bald eagle with wings half raised, a galah and a flamingo with folded wings, plus a
gargoyle and a dragon for the stylised end. Joint positions and silhouettes were read off
the photographs; things a photograph cannot show (flap amplitudes, joint limits, the
stowed pose) come from the literature and are marked as such. The photographs are
used as measurement references only and are not redistributed in this repository.

All bone lengths are quoted **relative to the forearm (FA)**, the standard measure in
chiropteran morphometrics; angles are in the wing plane, from the span axis (+X)
toward the trailing edge (+Z).

An independent check on the result: the planform-area aspect ratio of the membrane
wing comes out at **6.1**, inside the 6–7 reported for flying foxes, even though that
figure was never an input (`test_membrane_planform_has_a_realistic_aspect_ratio`).

## The membrane wing (bat / dragon)

| | measured | |
| --- | --- | --- |
| humerus | 0.72 FA at +25° (aft) | four chord measurements: 0.66, 0.71, 0.74, 0.75 |
| forearm | 1.00 FA at −30° (forward) | **the arm is a Z**: the wrist is the most forward point of the planform |
| digit II | 0.60 + 0.13 FA | short, hugging III |
| digit III | 0.71 + 0.57 + 0.70 FA | longest; makes the wingtip |
| digit IV | 0.70 + 0.43 + 0.33 FA | |
| digit V | 0.73 + 0.36 + 0.30 FA | sweeps back toward the hip |
| hooked tip | +24…+36° at the last knuckle | the falcate wingtip |
| thumb | 0.28 FA + 0.12 FA claw | hangs ventrally, hooked |
| fan off the forearm | II +18°, III +33°, IV +61°, V +90° | mounted skeletons splay 35–90° depending on the mount; this is the compromise that also matches the flight silhouettes |
| root | ~1.0 FA, down the back to the waist | the plagiopatagium runs shoulder → hip |

The earlier version had a nearly straight arm, single-bone digits, a 0.4 FA root and a
uniform scallop. The previous failure mode, still pinned by a regression test, was fanning
the digits off an already-swept forearm so the angles compounded and digit III ended up
62° off the span axis — a paddle, not a wing.

**Construction.** The sheet is a *fan*: every grid line runs from the arm out to the free
edge, and the lines that carry the digits run exactly down the bones, so a panel's border
and its bone coincide and the membrane can be bound to the fingers it is stretched
between. It is divided as real bats divide it: the *propatagium* fills the elbow notch in
front of the arm (a triangle between the straight shoulder–wrist leading edge and the
bent arm), the *dactylopatagium* spans the fingers, and the *plagiopatagium* runs from
digit V to the body. The leading edge follows digit III's own knuckles out to the tip so
the hooked bone never pokes out in front of the sheet.

**Scallops** differ gap by gap — depth is perpendicular sag over the tip-to-tip chord:

| gap | bat | dragon |
| --- | --- | --- |
| III–IV | 0.03 (taut) | 0.40 |
| IV–V | 0.26 (the deep one) | 0.50 |
| V → body | 0.09 (one long shallow arc) | 0.30 |

`SCALLOP_DRAGON` is a drop-in: `A.back_wings("membrane", scallop=W.SCALLOP_DRAGON)`. The
digit tips are **needles** that poke 0.14 FA past the membrane edge, and the scallop arcs
meet them at a cusp.

**Surface.** Camber is 6% of chord at rest (a gliding bat reaches 12%); the panels billow
a few percent between the bones; the hand rises ~9° off the arm (`gull`), so the wrist is
the low point of the wing. Bones are tubes laid on the sheet so they read as ridges, with
radii taken from the skeleton photographs (forearm ~0.053 FA across, digit bases ~0.034 FA
tapering to half) — but never thinner than 0.45× the membrane thickness, because a printed
bone has to stay thicker than the sheet it holds up.

## The feathered wing (bird / angel)

Primaries root along the hand, secondaries along the ulna, tertials and scapulars on the
humerus, and five tiers of coverts shingle over their bases.

| row | count | length (× longest primary) | width / length | direction |
| --- | --- | --- | --- | --- |
| primaries P1–P10 | 10 | 0.60 … **1.00** (P7/P8) … 0.70 (P10) | 0.20 (P1) … 0.13 … 0.105 (P10) | **−15° (P10) to +75° (P1), ~10° apart** |
| secondaries | 14 | 0.62 … 0.70 | 0.23 | +78° … +100° |
| tertials | 4 | 0.55 … 0.62 | 0.30 | +100° … +115° |
| primary coverts | 10 | 0.30 … 0.15 of the primary | 0.26 | follow their primary |
| greater / median coverts | 14 + 14 | 0.375 / 0.22 | 0.38 / 0.50 | follow their secondary |
| lesser coverts | 3 rows of 18–14 | 0.10 … 0.07 | 0.9 | along the arm's leading edge |
| alula | 3 | 0.18 / 0.25 / 0.30 | 0.22 | on the thumb |
| scapulars | 9 | ~0.45 | 0.25 | +100° … +130° |

The longest primary is ~1.65 FA, so a primary is 7–8× as long as it is wide: a long narrow
blade, not a paddle. The hand is only ~0.58 FA (the first build gave it 0.94), which is what
makes the primaries root at the right spacing (~0.065 FA) and overlap ~65% at the base.

**One feather.** A bare quill for the first tenth of the length, then a vane that widens to
its maximum by ~40% and holds nearly parallel-sided before closing in an ogive tip. The
rachis is off-centre — the leading vane is 25–30% of the width on the outer primaries, 35–40%
on the inner ones — and the outer primaries are **emarginated**: the leading vane narrows
sharply near the tip so the feathers separate into slotted fingers. The shaft droops and the
vanes cup away from it.

**Overlap.** On the upper surface each feather lies over its outboard neighbour (P1 over P2,
… P9 over P10), and the stack runs tertials > inner secondaries > outer secondaries > P1 …
P10. That is achieved with **roll**, not height: each feather's leading vane tilts up and its
trailing vane down, so it rides over the next one while every root stays on the arm. (An
earlier version stacked feathers vertically; the roots climbed 1.6 units off the arm and the
bone floated beneath the plumage.) The coverts lie over the flight-feather bases, the
leading-edge row on top.

## The rig

Everything below is `wingrig.py`, built on `core/armature.py`.

**Joints.** `root` (the end of the attachment line on the body, the only parentless joint) →
`shoulder` → `elbow` → `wrist` → per-digit base joints `d2_cmc … d5_cmc` (coincident with
the wrist, so the fan can open one finger at a time) → knuckles `dN_mcp`, `dN_ip` → `dN_tip`;
and `thumb_cmc` → `thumb_tip`. A bone is keyed by its child joint, exactly as in `Armature`.
The feathered wing adds a **root and a tip joint for every flight feather**, parented to the
arm joint it grows from, so the fan can open and close.

**Skinning** is linear blend, the same convention as `bodymesh.pose_body_mesh`, so topology
never changes: a shell that is watertight at rest is watertight in every pose.

| piece | binding |
| --- | --- |
| a feather, the thumb, a claw, a joint knob | **rigid**, weight 1 on one bone — a feather is stiff, and smearing it across two bones shears the shingles apart |
| a covert | rigid to the bone of the flight feather it lies over, so it fans with it |
| a finger or the arm | blended **only over its own chain's bones**, by proximity — it bends smoothly at its knuckles and is never pulled by the finger next door |
| the membrane | blended over the arm and finger bones by distance in the wing plane; the **root edge is bound to a body bone**, so the sheet stays glued to the flank however the arm moves |

**Controls** (all clamped to anatomical limits, `wingrig.LIMITS`):

| control | meaning | range |
| --- | --- | --- |
| `fold` | 0 spread … 1 stowed | 0 … 1 |
| `raise_` | shoulder elevation (flap) | −70° … +85° |
| `swing` | humerus swung back (+) or forward | −30° … +85° |
| `twist` | roll about the span axis | ±45° |
| `fan` | finger spread, 0 together … 1 rest | 0 … 1.5 |
| `curl` | finger knuckle flexion | −10° … 150° |

**Presets** (`wingrig.PRESETS`): `glide`, `half_fold`, `folded`, `upstroke` (wrist up, arm
half-folded, leading edge edge-on), `downstroke`, `mid_downstroke`.

**Folding.** Stowing closes the Z of the arm: the humerus tucks onto the **body line** (the
direction from shoulder to root — *not* the frame's +Z, which on a real placement points
down-and-outward, so folding toward it splays the wing sideways), the forearm swings forward
onto it to an elbow interior of ~25°, the hand folds back along the forearm, the fan collapses,
and each finger hairpins at its knuckles. Feathers fold like a closing fan, each turning about
its own root until it lies along the body line.

**The membrane in a fold.** Skinning moves the vertices that sit on a finger with that finger
but leaves everything between two fingers wherever blended transforms drop it, and a closing
sheet crumples. A patagium is continuous and inextensible, so each panel is **re-evaluated
between its two posed bounding fingers**: the interior is the chord between them plus the
panel's own rest curvature (scallop sag, billow) scaled by how far the gap has closed, and
the slack the sheet cannot lose buckles into a pleat — a tent of slant L/2 over a base g,
height √(L²−g²)/2, standing along the bones' own up direction, alternating panel to panel the
way a closing fan zig-zags. The long inboard panel pleats less: it hangs as a drape.

**Validated** (`tests/test_wingrig.py`): the rest pose reproduces the mesh to 1e-12; weights are
a partition of unity with ≤4 influences; rigid pieces use exactly one bone; a finger is pulled
only by its own bones; bones never change length under any pose; every pose of both styles is
watertight per piece; the root stays on the body while the wing flaps (≤0.16 units of drift
against 8–12 at the tip); finger tubes stay on the sheet; a posed mirrored pair on the body is
an exact mirror image (1e-15); body-space skin matrices (`M S M⁻¹`) reproduce the placed mesh
for mirrored placements; and the joints attach to a body `Armature` through the chest.

## Placement

Appendages are authored at the origin; the wing's frame is

```
+X  spanwise, shoulder -> wingtip
+Y  down (the underside); camber bulges toward -Y
+Z  chordwise, leading edge -> trailing edge
```

`back_wings()` wraps the placement: a scapula **anchor**, an `up_hint` that aims +X up and out,
a `roll`, and an `offset` that sinks the root into the back so a 32 mm print is welded, not
hairline-joined (at offset 0 only 2% of the root lies inside the torso; at −0.9, ~40%).

On a real body the frame's +Z does not run down the spine — it points down-and-*outward*
(≈ (0.80, −0.55, 0.24) in MakeHuman coordinates) — so a membrane root laid along it hangs in the
air beside the figure. `WING_ROOT` is therefore a **body point** (the lower back at the waist);
the builder converts it into the wing's plan coordinates (`Appendage.root_point`), so the root edge
runs from the scapula down the real back for any body shape, and `fold` stows the wing along the
same line.

A mirrored pair is an exact mirror image, which took two fixes: the mirrored anchor already
carries a mirrored tangent, so negating local Y (not X) leaves a true mirror; and the anchor
binds to the whole face, not `face[:3]`, because a quad and its mirror are stored with different
corner orders.

## Printing

`thickness` is in wing units. A 32 mm figure makes one unit ≈ 1.9 mm: the 0.16 membrane is
≈ 0.30 mm (right at what a resin printer resolves), feathers 0.20 ≈ 0.38 mm. Bones are inflated
to stay thicker than the sheet. Every piece — membrane, each bone, each feather — is its own
closed solid, merged into one mesh; overlapping closed shells still use every edge exactly twice,
so the mesh passes the watertight gate as-is and the slicer unions them. Print a wing spread (or
posed); a fully folded membrane is a thicket of self-overlapping sheet that no slicer will love.

## Regenerating the figures

```
PYTHONPATH=. python tools/wing_demo.py          # docs/images/wings.png + wing_rig.png  (~100 s)
PYTHONPATH=. python tools/wing_view.py out.png membrane fold=0.5 raise_=40
```

The per-feather and per-row tinting in the renders is vertex colour applied by
`tools/wing_view.py`; the geometry carries none.
