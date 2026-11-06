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
wing comes out at **5.0** (the dragon 5.3). The literature gives 6–7 for flying foxes; the roost photograph —
foreshortened, wrist flexed — gives 4.7–5.8; the build sits inside the photographs
(`test_membrane_planform_has_a_realistic_aspect_ratio` accepts 4.5–8, a paddle being below 3).
The hand was widened for the box-shaped hand panel (IV and V longer, IV fanned further) and the
deeper scallops give a little of the area back; the free edge that now stops short of the root
joint (below) takes a corner of the hip's area out again, which is where most of the move from 4.6
to 5.0 comes from. The fingers are still short for a flying fox (digit III 1.98 FA against 2.3–2.5 in
*Pteropus*) and were not lengthened: that would have moved every angle the earlier rounds fitted.

## The membrane wing (bat / dragon)

| | measured | |
| --- | --- | --- |
| humerus | 0.72 FA at +25° (aft) | four chord measurements: 0.66, 0.71, 0.74, 0.75 |
| forearm | 1.00 FA at −30° (forward) | **the arm is a Z**: the wrist is the most forward point of the planform |
| digit II | 0.60 + 0.13 FA | short, hugging III |
| digit III | 0.71 + 0.57 + 0.70 FA | longest; makes the wingtip |
| digit IV | 0.70 + 0.44 + 0.46 FA | |
| digit V | 0.73 + 0.40 + 0.34 FA | sweeps back toward the hip |
| hooked tip | III +14° then +34°, IV +8° / +16°, V +6° / +24° at the knuckles | the falcate wingtip, spread over two knuckles instead of one 46° elbow; IV is hooked less than V so the IV–V pocket is not a slot |
| thumb | 0.22 FA + 0.14 FA claw | a blunt peg that curls forward and up, so it reads from above; **exempt from the bone floor** (radius 0.028 FA at the base, 60% of that at the flat tip; ×1.35 on the dragon with the rest of its bones) |
| fan off the forearm | II +46°, III +50°, IV +86°, V +106° | the skeleton and the roost silhouette give 32/56/87/127°; the literature aspect ratio (6–7) wants less. This sits between them (AR 4.6) and makes the wrist a visible forward corner, as in the photographs; II is held so close to III that its tip is 0.1 units from digit III's line: the leading edge runs on down III without the sideways jog a free II tip made |
| root | ~1.0 FA, down the back to the waist | the plagiopatagium runs shoulder → hip; its free edge meets the root line 35% of the way up it from the waist joint (`ROOT_TAPER`; more makes the slab's thickness axis tilt past what `test_the_slab_thickness_axis_is_consistent_across_the_whole_sheet` allows at the root corner), so the lowest corner is a taper coming up into the flank, not a rectangle hanging beside the buttock when the arm closes (at half fold it is still a drape beside the hip, but a smaller, triangular one) |

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

The back panels' rays leave the **forearm** (trailing edge) and the **humerus** (root line) and
meet at the elbow, where the ray to the root corner is the seam. A trailing-edge point that lies
forward of the forearm's line is hidden behind the elbow (it happens when the root sits behind and
below the shoulder, as on the body), so its ray comes from the elbow itself; the visible points
share the rest of the forearm. Running every ray from one parameter along the whole arm had sent
the rays of six columns across the humerus: the arm-to-body fan folded over itself and the sheet
creased over the elbow (`test_the_trailing_edge_rays_never_cross_the_arm`, built on the real body
root: the default root never exposed it).

**Scallops** differ gap by gap — depth is perpendicular sag over the tip-to-tip chord:

| gap | bat | dragon |
| --- | --- | --- |
| III–IV | 0.12 (the old 0.03 was a ruled line) | 0.40 |
| IV–V | 0.28 (the deep one) | 0.50 |
| V → body | 0.10 (a long, shallow arc) | 0.15 (0.30 swung in to 0.3 units of the humerus; 0.18 came within 0.4 once the free edge was tapered up the root line, which the clearance test does not allow) |

The dragon is its own registered style, `A.back_wings("dragon")` / `W.make_wing_rig("dragon")`: the same
skeleton and rig with `SCALLOP_DRAGON`, a flatter sheet and ribs 1.35× heavier, and what dragon art keeps
that a bat does not (`DRAGON`, `DRAGON_SPEC`): a **thumb** 0.30 FA long (the bat's 0.22) ending in a **claw** 0.26 FA
(the bat's 0.14, 1.9×), a **thorn at the wrist** (`wrist_spur`, 0.22 FA, standing 40° up out of the wing plane, a rigid
part of the forearm like the knobs, tapering to a needle) and **longer finger spikes**: III 0.30 FA, IV and V 0.45× that,
II none, against the bat's 0.10 (the span still counts them, so the dragon's bones are a little shorter for the same
span). The scallop arcs meet the tips at a cusp. At fold 1 the thumb claw and the thorn stand up out of
the two wrists as a pair of horns at the shoulder blades.

**Surface.** Camber is 6% of chord at rest on the arm panel (a gliding bat reaches 12%) and
`HAND_CAMBER` = 0.42 of that, ~2.5% of the ray, on the taut hand panels (the dragon 5% / 2%). Each
finger panel **billows** between its bones by 6% of the gap between their tips (`BILLOW`; the loose arm
panel 0.62× that), up toward the dorsal face, so seen from below the fingers stand as ridges on a
domed sheet. Sector 0 (the sliver along digit III) does not billow: that would be a lump on the
bone. The hand rises ~9° off the arm (`gull`), so the wrist is the low point of the wing, and the
outer wing is **washed out** (`WASHOUT` = 5°): toward the tip the trailing edge stands higher than the
digit-III leading edge it trails. The old surface was a flat developable umbrella: the finger panels
bulged 3–4% of the gap, which a render reads as flat. Bones are tubes laid on the sheet so they
read as ridges, with radii taken from the skeleton photographs (forearm ~0.053 FA across, digit
bases ~0.034 FA) — but a bone is never thinner than 0.72× the membrane thickness at its base. The elbow and wrist knobs are a
**swelling of the bone** (1.12× its radius; they were 1.25–1.3×, twice the bone's diameter at figure scale: a rivet
on the rod from above, a pendant at the hem when stowed), and
a finger **tapers only to `RIDGE_FLOOR` = 0.55× the thickness** (half the thickness is where it
would sink into the sheet; IV and V used to taper to 0.36 and read as a dark slit). The
humerus keeps the elbow's radius over its outer part and **tapers to half of it at the shoulder**
(`HUMERUS_ROOT_TAPER`; the printable floor holds it at 0.86×): it used to be thickest there, so a constant rod ended abruptly against the
torso like a bolt. Where it leaves the back (0.33–0.43 of its length from the shoulder, on every body morph, because the shoulder is sunk
0.9 units into the scapula) it **swells into a fillet** of 1.4× the elbow's radius (`HUMERUS_FILLET`, a gaussian 0.19 of the
humerus wide centred 0.36 from the shoulder, merged with the spindle by a soft maximum), so the rod grows out of the shoulder
blade like the muscle mass of an upper arm. It is a fairing in the middle of the bone, not a lump: never more than 1.5× the rod, and back
to the rod by mid-humerus.

## The feathered wing (bird / angel)

Primaries root along the hand, secondaries along the ulna, tertials and scapulars on the
humerus, and five tiers of coverts shingle over their bases.

| row | count | length | width / length | direction |
| --- | --- | --- | --- | --- |
| primaries P1–P10 | 10 | 0.60 … **1.00** (P7) … 0.80 (P10) × the longest primary | 0.20 (P1) … 0.13 … 0.105 (P10) | **+70° (P1) to +8° (P10)**; P10 continues the hand axis. Gaps widen from ~4° (P1–P2) to ~9° (P9–P10; measured 4.2, 4.9, 5.5, 5.4, 7.4, 8.5, 8.3, 8.2, 8.8), so the inner wing is a closed sheet and the outer five are separate fingers |
| secondaries | 14 | 0.62 … 0.70 | 0.23 | +78° … +100° |
| tertials | 7 | **0.60 … 0.48** (the first is as long as the last secondary) | 0.26 | +92° … +110° |
| primary coverts | 10 | 0.46 (P1) … 0.32 (P10) **of their own primary** | 0.36 | follow their primary |
| greater coverts | 14 | **0.50** of their secondary | 0.40 | follow their secondary |
| median coverts | 14 | **0.30** of their secondary | 0.52 | follow their secondary |
| tertial coverts | 7 + 7 | 0.50 / 0.30 of their tertial | 0.42 / 0.55 | follow their tertial |
| lesser coverts | 2 rows of 22 and 26 | 0.17 / 0.13 of a secondary (**~15%**) | 0.75 | straight back from the arm |
| marginal coverts | 1 row of 36 on the nose of the arm, 1 of 28 below (**the 12% nearest the shoulder are not built**: 3 and 2 of them lay wholly inside the torso on every body) | 0.15 / 0.12 of a secondary | 1.0 / 0.95 | root on the front of the arm and wrap over (under) it |
| alula | 3 | 0.32 / 0.29 / 0.26 of the longest primary | 0.20 | rooted on the leading edge at the wrist, **26° / 18° / 10° off the hand axis**, on top of the primary bases |
| scapulars | 7 | 0.70 … 0.56 FA | 0.26 | +98° … +112° |

The longest primary is ~1.65 FA, so a primary is 5–10× as long as it is wide: a long narrow
blade, not a paddle. The hand is only ~0.58 FA (the first build gave it 0.94), which is what
makes the primaries root at the right spacing (~0.065 FA) and overlap ~65% at the base.

**One feather.** A bare quill for the first tenth of the length, then a vane that widens to
its maximum by 35–40% and holds parallel-sided before closing onto a **rounded point**
(`tip_floor`, never a bar: the last 3% is bevelled to 60% of the sheet so the end does not
stand up as a flat wall, which read as a spike on every tip). The outer primaries (P7-P10) used to
taper from 70% of their length down to 7% of the vane width: the last 3% was 0.09-0.17 units wide
(0.17-0.32 mm at 32 mm), a fifth of the vane, a needle that snaps. They now keep the distal vane to
84% and close onto a rounded tip at half the width (`OUTER_TIP`): a third of the vane's width over
the last 3% of the length (0.14-0.28 units) and half of it just before (0.20-0.39); the notch and
the slots are untouched. The rachis is off-centre —
the leading vane is 27% of the width on the outer primaries, 37% on the inner ones — and
the outer primaries are **emarginated**: from P4 outward the leading vane is cut back
sharply at the notch (a step over ~9% of the length, to 55–85% less) and kept narrow to the
tip, and from P5 the trailing vane is cut a little nearer the root, so the feathers separate
into slotted fingers (the slot between P6 and P10 opens to 0.3 units at 85% of the length;
the old fan left 0.1). The vane is a **sheet** of near-constant thickness (0.9 × `thickness`,
0.79 at its edge) with a **shaft ridge** standing 0.45 × `thickness` proud of one face only —
the upper on a dorsal feather, the lower on a ventral one — fading toward the tip; the
other face stays flat. That is what a single lit ridge down each feather reads as, and it
keeps every part of the vane at least 0.75 × `thickness` (0.15 units, ~0.27 mm), where the
old lens went down to 0.4 × at the edge and tip and 0.2 × at a covert's rim. Vanes curl
down in proportion to their own width and the shaft droops along its length.

The slab's normals are computed from the *un-ridged* grid and the ridge is added as a
vertical offset afterwards: folding it into the grid gave adjacent columns of a narrow
(notched) vane a hair's breadth apart and a steep slope, which tilted the normals sideways
and twisted the sheet there.

**Overlap.** On the upper surface each feather lies over its outboard neighbour (P1 over P2,
… P9 over P10), and the stack runs tertials > inner secondaries > outer secondaries > P1 …
P10. That is achieved with **roll**, not height: each feather's leading vane tilts up and its
trailing vane down, so it rides over the next one while every root stays on the arm. (An
earlier version stacked feathers vertically; the roots climbed 1.6 units off the arm and the
bone floated beneath the plumage.) The coverts lie over the flight-feather bases, the
leading-edge row on top. Each covert tier is **sized as a share of the feather it lies
over** — greater 50%, median 30%, lesser 17% and 13% — so the steps down are a visible
geometric progression, each tier is **staggered** half a pitch against the one behind it
(each covert straddles two flight feathers, so rows brick instead of stacking into
columns), and a covert is a *strip* with a **U-shaped end** carrying nearly full width from
the quill (a lozenge narrow at both ends is what a row of rice looks like): greater coverts
are 0.40 as wide as long, medians 0.52. Their quill ends are buried under the row in front. Each covert differs a little from its
neighbours (length +-9%, aim +-1 degree, tilt and roll a couple of degrees: `COVERT_JITTER`): with
3% a stowed wing, which lays every covert of a tier parallel, read as a lattice of identical round
tips, armour.

The **leading edge** is a solid round nose: the arm tube (radius 0.085 FA at the shoulder,
0.066 at the wrist, its axis 0.07 FA above the wing plane) with two lesser rows behind it and
one **marginal row that roots on the front of the tube and wraps over it** (pitched up 58°
and curled back to lie flat over the lesser rows), a second wrapping row underneath, and a
single lesser row under that. The old edge — 38 rounded pods per face on a thin rod — read
as a string of beads from the front (the three-quarter view) with the bare rod showing
between them; with rounded quill ends overlapping at ~55% it is a feathered roll with no
rod to see. The covert field runs on over the tertials to the back instead of stopping at the
elbow; an **underwing covert field** covers the underside, which is what an angel seen from
the front shows. Scapulars (lanceolate, in a near-parallel rank, each a little longer than
the one outboard of it) and the alula are rigged feathers like the flight feathers, so a fold
re-aims all the plumage.

The **alula** is two or three small, stiff feathers on the leading edge at the wrist, 26–32%
of the longest primary long (it was 12–19%, rooted out along the thumb), standing 10–26° off
the hand axis in a small fan and rolled up 12°, stacked longest and lowest: it is a group of
its own, not a continuation of the primaries. It sits on top of the primary bases (each
feather 0.08 units higher than the one below).

Total: 274 pieces, ~86k triangles (was 306 pieces, 98k). The humerus's tube ends in a rounded
blunt tip at the shoulder (`blunt` in `_feather_bones`), and the first marginals are not built
(`ROOT_BURIED`): stowed, the shoulder end of the arm stood up beside the wrist as a second dome, and
spread it is buried in the back.

The tertials (0.60 down to 0.48 of the longest primary, `TERTIAL_RATIO`; they were 0.52 down to 0.38,
so the first was 13% shorter than the last secondary and the inner trailing edge fell vertically at the
elbow in a step) and scapulars trail at +92…+112° rather than hanging straight down the body line (+131°
on the placed wing), so the spread wing's inner trailing edge rises into the back at the lower
scapula instead of the two wings' edges nearly meeting at buttock height (that edge sat at body
height 1.9 and now sits at 2.4; raising the root alone moves it by only 0.2-0.3, because the wing plane is
fixed mostly by the span axis).

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
| a membrane finger | **one rigid tube per phalanx**, welded to its own bone (`digit_III`, `digit_III_p1`, `digit_III_p2`; the last owns the needle), and a **ball on every knuckle** (`knuckle_III_1` …), as the elbow and wrist have: a finger blended across a knuckle collapsed where the concertina folds it back (its rings were 5–12% of their rest area at fold 1, 25% at curl 150: bow-ties that the watertight gate, which counts edges, cannot see) |
| the arm | blended **only over its own chain's bones** (`elbow`, `wrist`), by proximity — it bends smoothly at the elbow and is never pulled by a finger |
| the membrane | blended over the arm and finger bones by distance in the wing plane; the **root edge is bound to a body bone**, so the sheet stays glued to the flank however the arm moves |

**Controls** (clamped to anatomical limits, `wingrig.LIMITS`, and then to the narrower per-skeleton ranges
below; the caller's object is never modified):

| control | meaning | anatomical range (`LIMITS`) |
| --- | --- | --- |
| `fold` | 0 spread … 1 stowed | 0 … 1 |
| `raise_` | shoulder elevation (flap). Free wing: about the frame's chord axis Z. **Placed on a body: about the body's fore-aft axis through the shoulder, + lifts the tip on either side.** The top shrinks with `fold` (100° spread, 25° stowed) and so does the bottom (−70° spread, 0° stowed: a stowed wing already lies on the back). The feathered wing **arches**: the hand lags the wrist by `raise_arch` (0.30) × raise (free wing: out of the plane; on a body: in the plane of the flap, so a raised wing's hand hangs lower), instead of the limb swinging as one plank | −70° … +100° |
| `swing` | humerus swung back (+) or forward. Free wing: in the wing's plane. **Placed: about the body's vertical axis, + sweeps the tip back on either side** (so it changes the azimuth and never the elevation). Top 85° spread → 5° stowed, bottom −30° spread → 0° stowed | −30° … +85° |
| `twist` | roll about the span axis | ±45° |
| `fan` | finger spread, 0 together … 1 rest. A bird's primaries spread about the hand axis and the request is clamped to 0.2 … 1.1 (at 0 all ten primaries share one heading and the secondaries stack coplanar; past ~1.1 P1 swings across S1); a membrane's skin takes only 0.9 … 1.1 (see below) | 0 … 1.5 |
| `curl` | finger knuckle flexion; on the feathered wing it **cups the hand** instead: each primary droops toward the underside by `curl_cup` (0.50) × curl × (its rank P1 … P10 / 9)^`curl_cup_pow` (1.5), capped at 80°, and rolls about its own shaft by `curl_roll` (−0.30) × curl × the same weight, so the fan curves down from level P1 to steeply drooping P10 (curl 100: 2, 4, 9, 14, 20, 31, 39, 47, 55, 62° for P1 … P10) and the vane planes follow the cup. The hand's own knuckle flexes only `curl_gain` (0.05) × curl: a 0.35 hinge with a linear cup made curl 100 an L-shaped hand drop, every primary straight and parallel (a 32° step between P5 and P6), indistinguishable from `wrist_flex` −45. Curl 60 drops the tip of P10 2.4 units, P1's none. On a membrane each knuckle follows the control exactly up to `curl_knee` (30°) and at `curl_slope` (0.4) of it beyond (curl 60 → 42°, 100 → 58°, 150 → 78° a knuckle, two knuckles a finger): a sheet stretched between fingers that spiral through 100–150° each had nowhere to go but a clump at the fingertips with the rods poking out | −10° … 150° |
| `wrist_flex` | cocks the hand out of the wing plane at the wrist, tip up (+) or drooping (−) | −60° … +45° |
| `hand_twist` | rolls the hand about its own axis (wash-out) | ±30° |

**Bad input is an error, not a rest pose.** A control the rig does not have (`rig.pose(foled=0.5)`,
`PoseControls.flap(30, flod=1.0)`, `c.flod = 1`) is a `TypeError` that names the controls and suggests the near
miss; an unknown preset name is a `KeyError`; a control that is not a number is a `TypeError` and one that is NaN
or infinite is a `ValueError` (NaN used to pass `np.clip` and come out as a NaN mesh); a joint the wing does not
have in a raw pose dict (`{"wirst": ...}`, handed to `pose_mesh`, `skin_matrices`, `posed_joints`, `PlacedWing.*`)
is a `KeyError`, and a rotation that is not three finite angles a `ValueError` (`WingRig.check_pose`). Every valid call is unchanged:
out-of-range values are still *clamped*, not refused.

**Presets** (`wingrig.PRESETS`): `glide`, `half_fold`, `folded`, `upstroke` (wrist up, arm
half-folded, leading edge edge-on: raise 25, fold 0.35, twist −15, fan 0.85, wrist −30), `downstroke` (raise −20,
twist 8, curl 12, wrist 20), `mid_downstroke`. (The upstroke used to ask for raise 55 and fan 0.7: on a body that
put 15% of the sheet across the spine and 374 of the bat's 1872 grid edges over 1.15× stretch.)
`PoseControls.flap(phase)` generates a cycle in which the shoulder leads while the wrist (40°) and the finger curl (70°) lag it,
so the hand trails the wrist — hanging below it on the upstroke — instead of the whole wing swinging like a plank.
The upstroke is not the downstroke played backwards: the arm half-folds and the fingers
close to shed the air, then the arm extends and the hand pronates for the power stroke. `flap(..., up_reach=, down_reach=)`
sizes the two halves of the stroke apart, and `WingRig.flap` / `PlacedWing.flap` fill them in from the wing's own range:
a wing on a body rests ~50° above horizontal, so it can rise little and drop a lot, and a cycle of ±45° would sit clamped at the top.

**Control ranges per skeleton.** The anatomical ranges are what the joints allow; a skeleton narrows them to what
its skin takes (`WingSkeleton.limits`, in the specs) and, once placed, to what the body leaves (`placed_limits`, which
**replace** the entry of the same control: raise and swing mean a different rotation on a body). A bound may be a function of
`fold` (`wings.by_fold`); `WingRig.control_range(name, fold)` returns what is left. At fold 0, in degrees (fan and fold as is):

| control | free bat / dragon | bat / dragon on a body | angel on a body |
| --- | --- | --- | --- |
| `raise_` | −50 … 50 | −26 … 22 (22 → 26 by fold 0.3; the floor closes to −10 at 0.5, 0 stowed) | −28 … 38 (floor −10 at 0.3, 0 from 0.6; 50 from 0.3) |
| `swing` | −5 … 55 (35 from fold 0.5) | −20 … 35 (25 at 0.25, 16 at 0.5) | −15 … 15 (28 at 0.3, 40 at 0.6) |
| `twist` | −30 … 22 | −30 … 22 | −25 … 25 (hi 8 at 0.6; lo −8 stowed) |
| `fan` | 0.9 … 1.1 (0.85 at fold 0.3) | same | 0 … 1.1 |
| `curl` | −10 … 30 (12 from fold 0.5) | same | −10 … 150 |
| `wrist_flex` | −60 … 45 (floor −45 at 0.3, −35 at 0.5) | same | −60 … 45 |
| `hand_twist` | ±22 | same | ±30 |

*Why these numbers.* Membrane (bat, dragon): the stretch limiter is a Jacobi pass that cannot shorten a chain between two pinned
ends, so it could not keep a sheet inside what skin does at the legal extremes. Measured on the mid-surface grid
(edges of rest length > 0.15; `tools/wing_ranges.py` prints the sweeps), p99 stretch / edges over 1.3×, before → after the range:
fan 0.55 1.94 / 191 → 0.9 1.23 / 0; raise 100 1.50 / 150 → 50 1.21 / 0 (free); swing −30 1.75 / 193 → −5 1.15 / 0; twist 45 1.52 / 119 → 22 1.19 / 0;
curl 100 1.36 / 50 → 30 1.15 / 0 (the dragon's heavier ribs set the curl for both). At every end of every range, spread, the p99 is ≤ 1.25 with at most one edge
over 1.3× (`test_asking_for_any_extreme_control_keeps_the_sheet_inside_the_stretch_bound`, which also runs a third folded). On a body the root edge is glued to the waist,
so a raise lengthens the sheet there: the cliff is raise −32 (1.00 → 1.29 / 17 edges) and +25 (1.26 / 2; 1.31 / 25 at +30), and the
spine (the tips cross the midline at about +25) agrees. Angel: no skin to tear, so the ranges are the body's — the
inner edge of the spread angel is only ~0.3 units from the midline, so a raise below −32 or a swing past +22 puts the scapulars across the
spine, and twisting a stowed wing toward the body presses it into the torso (fold 1, twist −15: 10% of the vertices inside it against 2% unposed).
Every corner of the legal raise × swing × twist box, spread, half folded and stowed, is a test on the plain body
(`test_combined_extremes_of_the_shoulder_controls_...`). The fan of a bird opens to 1.1: past it P1 crosses the secondaries.

Feather headings are computed from the **in-plane** rotations only, so lifting, twisting or cocking a
wing tilts its plane without changing where the feathers point within it: the 3D angle between the
first and last primary stays 61° under any raise, twist or wrist flex (a first version read the plan
projection of the tilted frame and spread the fan to 166° at raise 100).

**Folding.** Stowing closes the Z of the arm: the humerus tucks onto the **body line** (the
direction from shoulder to root — *not* the frame's +Z, which on a real placement points
down-and-outward, so folding toward it splays the wing sideways), the forearm swings forward
onto it to an elbow interior of ~15° (the bat's was 25° before; the bird's was already 14°), the hand folds back along the forearm (wrist 172°), the fan collapses, and the thumb
tucks along the forearm. The stowed wing is also
**lifted off its plane** (bird 20° — a feathered wing is deep): that plane is sunk into the back so a print is
welded, and a wing lying along it would be buried in the torso. Feathers fold like a closing fan,
each turning about its own root until it lies along the body line; scapulars, the alula and the
lesser coverts fold with them.

**Stowing the bat and the dragon on the back.** Lifting the membrane arm as a whole (`fold_lift`) does not
work: the stowed wrist is *above* the shoulder (the Z closed up on itself), so the arm is a see-saw about the
shoulder and a lift that raises the elbow pushes the wrist into the back. The old stow (`fold_lift` 12°) left the
wrist knob, the thumb and the claw 100% inside the torso on the default body, the sheet 25% (the dragon 29%), the propatagium 51% and the arm 39%,
and only the elbow (hanging at the hem like a pendant) showed. It now has four separate spec parameters
(`BAT_SPEC`, in `controls_to_pose`, none of them touches the spread wing or the shoulder lift/swing mapping):
`fold_lift` 6° (the humerus), **`stow_elbow_lift` 12°** (the folded forearm is tilted out of the humerus's plane at the elbow, so the
wrist and the hand folded along the forearm stand off the back and the wrist is the visible apex of the roosting wing),
**`stow_root_lift` 2°** (the whole wing pivots about its root joint, which stays on the waist, so its upper end stands off the
back; nothing in the wing's own frame changes, so the Z-fold and the order of the fingers are untouched; small because the sheet
beyond the root swings the other way and a stowed membrane must lie among its bones) and **`stow_thumb_lift` 55°** with **`stow_thumb_swing` 55°** (the thumb
turns up out of the wrist and swings back 55° along the forearm instead of 70°, so the claw hooks out of the wrist instead of hanging toward the
back or reaching across the spine: the dragon's longer claw crossed the midline by 0.3 units at 70°/60°). Measured as the share of vertices inside the torso on the anatomy-adjusted, plain, slim, heavy and lean bodies (before → now):
the bat's whole stowed wing 13–34% → 7–10%, its wrist knob 100% (0% on the slim body) → 0–3%, thumb and claw up to 100% → 0%, membrane sheet 12–30% → 6–12%,
propatagium 39–58% → 23–28%; the dragon's whole wing 8–12%, wrist knob 0–21% (the lean body), thumb 0–5%, claw 0%, sheet 6–13%, propatagium 24–34%.
The arm is still 16–25% buried: that is the weld at the shoulder, deliberate.

The membrane's **fingers** fold in two phases (`wings.knuckle_phase`). Until the wing is 75% stowed
they stay straight: the fan closes, the hand folds back, and the sheet pleats between straight
fingers like a closing fan. A finger that folds *in the wing plane* passes through an L, a U and a
spiral on its way to the Z, which no sheet can follow, so the concertina is kept to the last
quarter of the fold. Then each knuckle turns the **full** 180° less its rest bend
(`knuckle_fold` 1.0; 0.90 left every knuckle ~17° off and the Z of digit III crossed the Z of
digit V in an X, 49° off the forearm), so metacarpal, first and second phalanx lie back along
each other as one rod. The closed hand is not left to chance: each metacarpal closes to a
deliberate angle (`stow_fan`, −12/−9/−6/−3° for II–V, spread ×`bone_scale` for the heavier dragon
ribs) and alternate fingers tilt ±8° above and below the plane (`stow_tilt`), so the stowed hand is
four separate rods fanned alongside the forearm with 0.3–0.5 units of daylight between their
centrelines (the old residual, 8% of the rest fan, left them 0.05–0.1 apart: one fused
bundle). They end up within ~25° of the forearm axis (the plain 180° concertina is ~20°).

**Stowing the feathered wing against the back.** A feathered wing placed on a body does not stow
along the plane it was mounted in (that plane leans ~55° off the back, so the stowed wing was a plank standing
off the back, buried at the top and floating at the hem). `WingSkeleton.stow_against(out, down, side)` is told where
the back is, in the wing's own frame, and a fold then **pivots the whole wing about its root** (the root joint rotates; every
bone length is preserved) until the plane lies on the back: `stow_lean` (11°, it was 6°) leaves the lower end standing off it, which is what
clears the buttocks and sinks the top of the wing into the shoulder blade (the weld: the top sinks ~5 units × sin 11°
about the pivot, the hem stands ~3 units × sin 11° off the back);
`stow_toe_in` (8°) turns the wing's axis toward the spine, so a pair stows in a **V** that closes toward the hem
(`x` clearance of the stowed wing 0.3 on the plain body, never below 0.1 on any morph tested); `stow_pitch` (8°) lifts every
flight feather about its root, because a primary droops 0.16 of its length toward the body (its camber) and a flat-stowed tip
would dig into the buttocks; `stow_taper` (8°) turns the edge feathers toward the middle of the bundle, so it narrows from the broad
covert mass to the primary tips (tip band 16% narrower than parallel feathers leave it). The wrist, the top of the forearm's
leading-edge covert rope, ends outboard and above the shoulder, a rounded bow clear of the back.
On the plain body 4.6% of the stowed wing's vertices are inside the torso (it was 28% before the flat stow, 1.7% at lean 6°) and
2.4% of the flight feathers; measured along the back's normal on the four body morphs (plain, anatomy-adjusted, heavy, slim;
`tests/test_angel_poses.py`) 1.8-9.3% of the vertices are inside (it was 0.4-4.6%), the deepest 0.61-0.98 units
(1.1-1.8 mm; it was 0.34-0.61) and 0.3-4.6% are deeper than 0.3 units (0.01-1.2% before); none cross the midline. The wing is a flat
plank laid on a back that curves away to the flank, so the contact is a strip down the medial edge, the shoulder blade and the
lumbar, not the whole underside: slices of the stowed pair show the lateral half floating 0.2-1.0 units off the skin.

**Feathers follow their bone.** The feathers no longer swing to the stowed heading in a straight line: until `stow_follow`
(30%) of the fold is done each follows the bone it grows from, then eases to the stowed heading; the primaries close their fan on the
way (smoothstep: 65% closed by half fold, 90% by three-quarters) so a half-folded wing fans out of its arm rather than hinging
like a plank.

**The fan across the elbow.** Carried rigidly, the secondaries (on the forearm) and the tertials and scapulars (on the
humerus) turn through the whole elbow angle against each other. The trailing side is the outside of that bend, so a wedge opened
there and the two blocks stood as separate rectangles with a step between them: at fold .25 / .5 / .75 the tips of the last
secondary and the first tertial were 0.49 / 0.83 / 0.55 of a secondary's length apart. A real wing's feathers fan there. With
`elbow_fan` (0.9) each secondary, tertial and scapular near the elbow half-follows the *other* arm bone's heading as well, weighted
0.5 at the elbow and running out over 90% of each bone (`WingSkeleton._elbow_fan_weights`): at the elbow the last secondary and the
first tertial point the same way (50° apart hinged, under 12° fanned) and telescope out of the one point, so the trailing edge goes
round the bend as one curve (the largest gap between neighbouring tips along the arm is 0.12-0.25 of a secondary at .25-.75). It is
applied to the heading the feather *carries* before the blend to the stowed heading, so it is exactly absent at fold 0 and at
fold 1. The arm itself still folds linearly (the Z is unchanged); only the plumage no longer hinges with it.

**The membrane in a fold.** Skinning moves the vertices that sit on a finger with that finger
but leaves everything between two fingers wherever blended transforms drop it, and a closing
sheet crumples. A patagium is continuous and inextensible, so each panel is **re-evaluated
between its two bounding fingers**: the interior is the chord between them plus the
panel's own rest curvature (scallop sag, billow) scaled by how far the gap has closed, and
the slack the sheet cannot lose buckles into a pleat — a tent of slant L/2 over a base g,
height √(L²−g²)/2, standing along the bones' own up direction, alternating panel to panel the
way a closing fan zig-zags.

Those two bounding **columns** are the weak point: skinned bone by bone, the column along a
finger folded into a concertina runs out, back and out again (it doubled back along its own
chord by 25–60% and bowed 17–33% off it), and two such columns crumple the panel between
them into a heap. `WingRig._gather_fingers` therefore puts each finger's column where the skin
actually is: along the finger (each row at the distance along the bones it had at rest) once a
knuckle is bent past ~75–105° (a rest finger's hooked tip is bent 47°, a curl of 40° gives 67°,
so a curl keeps its skinned arc), and along the **straight ray from the posed wrist to the posed
fingertip** once it is folded up (120–165°): the closed fan, each row at the same fraction of the
way it was at rest. The column behind the last finger (arm to body) is straightened with it.
In the stowed pose the four columns are exact rays, the panels between them are ruled strips,
and the pleats run along them like the ribs of a closed fan.

Skin is elastic, so nothing pleats until a panel has closed by ~10%, and then the pleats come in over
a further 15% of closure (`PLEAT_RAMP`; a smoothstep, not a switch). A wing that is only
*cupped* (curled or flapping, the hand not yet folded back: `WingRig._hand_fold`, read off the
angle between forearm and hand) gives a further 8% and buckles half as hard, fading out by the
time the hand is 30% of the way back, so a curl stays a smooth finger arc under a taut sheet
instead of rucking at the tip. Each panel pleats in several fine lobes (1/2/2/3) rather than one
bladder, and a pleat is softly capped twice: by `PLEAT_CAP` (0.30 units per lobe, about one bone
radius: a stowed membrane is a bundle lying among the fingers, not a tent three finger-thicknesses
tall; the folded bat stood 4.0 units off the plane before the cap was introduced, 0.45 per lobe gave
2.3 and left lobes proud of the rods) and by `PLEAT_SLOPE` times its own base (the gap over the
lobes), so a pleat dies away with the gap it stands over instead of flipping from one face of the
sheet to the other when two fingers pass each other (the chord between them reverses). The long
inboard panel pleats less: it hangs as a drape. The propatagium (the skin in front of the arm) is a separate piece skinned to the arm **alone** (the humerus
and the forearm, `chain=("elbow", "wrist")`: it used to be weighted over every arm and finger bone by plan distance, so closing
the fan pulled it by 0.1 units): it hinges with the elbow as a flat triangle and never leaves the arm farther than at rest. It does
not close up with the arm (a stowed propatagium is a flap 0.6–0.9 units wide beside the arm, about half of its rest volume); blending its edge
from the humerus into the forearm by position along it looks as if it should, but at a fold the two frames are 165° apart and half
of its cells turn inside out (signed volume fell to zero), so it is left as a hinge.

**The interior is re-evaluated at every pose.** Earlier the chord-based interior was blended in by
`fade = clip(slack / (0.02 g_rest^2))`, which goes from 0 to 1 over 1-2% of closure: a panel that
closed past its elastic limit snapped from the skinned sheet to the chord-based one, and a mid-surface
vertex moved 0.04 per step and then 1.1 in one step (fold 0.920 -> 0.925), curl popped at 127 -> 128,
and 70-108 of 200 fold steps held such a pop. Linear-blend skinning of a vertex between two fingers
is `(1-s) T_A p + s T_B p`, which differs from the chord between the two fingers' own posed points
by `s(1-s)(R_A - R_B)(b - a)` (up to 2-3 units mid-fold), so it was the skinned interior that was
wrong, not the chord. The interior is now the chord, plus the panel's rest curvature carried in an
orthonormal frame (which reproduces the rest mesh exactly), at **every** pose including a raise, a
twist or a swing; skin and chord agree where the fingers move together, and elsewhere the chord
strains each row evenly (by exactly `g_pose / arc`) where skinning piled the strain onto a few
edges (raise 100: 150 edges over 1.3x became 34-40, twist 45: 119 became 21). What is left of the
old threshold is the pleat's onset, which is continuous.

The panel's frame needs care in a stowed bundle: two rods stacked one above the other leave the
panel between them standing on edge, its chord vertical, so "the wing's up axis made perpendicular to
the chord" is noise and reverses on the far side of vertical. The panel's normal is therefore built
from the quad's own orientation (`chord x ray`, the ray being the mean direction of the two bounding
fingers' metacarpals, which a concertina never folds back; sign fixed at rest and carried into the
pose) and from the up axis, each weighted by how well defined it is; a curl turns the finger tip about
the chord, so while the wing is only cupped the up axis leads. The sliver along digit III (where chord
and ray are nearly parallel) uses the up axis alone. Pleat direction, the transport of the rest
curvature and the slab's thickness hint all use this frame.

**Stretch limiter.** After the interior is placed, `WingRig._limit_stretch` pulls any edge longer
than `MAX_STRETCH` (1.15) times its rest length back. It is Gauss-Seidel projection in four
colour classes (so every class is a set of disjoint edges and the sweep is vectorised), 24 sweeps
(`STRETCH_ITERS`; halving it, or 6, fails the limiter tests), moving the free end of an edge (both
ends if both are free): the whole finger column of each panel is pinned (the sheet rides its bone).
A pose costs 11-20 ms (7-13 ms before), of which the limiter is a third to a half. The old Jacobi pass divided each vertex's
correction by the number of edges that were over the limit, which is discontinuous in the pose (a
second edge crossing the limit by a hair halved the first one's pull) and could not shorten a chain
between pinned ends. A row of a panel whose two fingers are held farther apart than the skin can
reach has no solution; projection would pile the whole excess on the last edge visited, so such a row
is allowed the even strain its ends dictate (`limit_v`) and the limiter leaves it alone. The effect
is measured on the mid-surface, edges of rest length over 0.15:

| pose | edges over 1.3x, limiter off | on |
| --- | --- | --- |
| fold 0.8 | 26 (bat) / 43 (dragon) | 1 |
| fold 0.5 + curl 60 | 26 / 46 | 4 / 9 |
| curl 100 | 36 / 99, worst 4.1x / 7.5x | 4 / 7, worst 2.0x |
| `upstroke` preset, flap(90) | 64, 56 (old limiter) | 1, 2 |

The strain that remains is where the controls ask for the impossible (`fan` 0.55, `swing` -30: the
row between the last finger and the body is held 1.4-1.6x too far apart; the legal control ranges,
not the limiter, are the protection there).

The slab keeps its grid's own orientation (re-orienting cells toward a skinned "up" vector turned the
whole shell inside out where a panel doubled back: signed volume fell from +0.75 to -0.03).

**Composition and limits.** A knuckle curl (about the digit's own flexion axis) and the fold
concertina (in the wing plane) are *multiplied* as matrices — curl first, then the swing — rather
than having their Euler angles summed, which is not a rotation of anything once two axes are
involved (`test_a_knuckle_curl_and_the_fold_concertina_multiply_instead_of_adding`; the schedule `knuckle_phase` is read from `wings.py`, not restated). Controls are
offsets on top of what `fold` already does, so the shoulder's `swing` and `raise_` ranges shrink
as the wing stows (`FOLD_LIMITS`): fold 1 with swing 85 once asked the humerus for 195° and put
half the membrane across the midline.

**Slab normals.** A scalloped patagium leaves each fingertip along the grid's own rays, where the
cross product of the two grid tangents collapses and its direction — even its sign — is noise. On
the dragon (scallops 0.30–0.50) 205 vertices were within 20° of degenerate and one column of the
slab had its thickness lying *in* the sheet; the pleats, which stand along that axis, ballooned.
`grid_normals` therefore blends toward a reference normal wherever the tangents are within ~20° of
parallel, and is bit-identical elsewhere (it builds the rest slab). For a *posed* sheet that was
still a switch: it picked the sign of an unreliable normal by comparing it with a reference, so a
normal nearly perpendicular to the reference changed sides between two poses and the top and bottom
faces of that vertex swapped places (a one-thickness jump; with 0.2-unit-thick skin, 25 such vertices
in a 200-step fold sweep). `_slab_normals` replaces it: a vertex's thickness direction is the **sum of
the area vectors of the cells around it** (so it always follows the grid's own winding, and a cell
that has collapsed counts for nothing because it is small) plus a small weight (`SLAB_HINT` x
thickness^2) times the panel's frame normal, which decides only where every cell around the vertex has
collapsed. It is a continuous function of the positions, and the posed slab's oriented volume stays
positive: by cell count 116 (bat) / 119 (dragon) of 1122 cells were inverted against the slab's own up
vector at fold 1 (and 25-48 at fold 0.5-0.9); now 21 / 16 (3-14 at fold 0.5-0.9). By *area* the old
slab had 5.0% / 6.3% of its sheet inverted stowed and ~2% at fold 0.5-0.9; now 0.3% / 0.2% and
0.1-0.3%. What remains is the collapsed panels of the closed fan (chord parallel to ray, no extent
left to turn over: zero-area slivers whose sign is noise). The oriented slab volume is 99.9%
positive. slab's thickness along a steep pleat is thinner than nominal (the weight of the hint pulls it toward
the panel normal): print a bat spread. Separately, the camber weight stepped from 0.25 to 1.0 at digit V,
leaving a 0.13-unit ledge down the finger (more than half the membrane's thickness); it is now smoothed
over nine columns. The dragon is part of the rig test matrix.

**Validated** (`tests/test_wingrig.py`, `tests/test_wingrig_api.py`): the rest pose reproduces the mesh to 1e-12; weights are
a partition of unity with ≤4 influences; rigid pieces use exactly one bone; a finger is pulled
only by its own bones; bones never change length under any pose; every pose of every style is
watertight per piece and a posed piece has positive volume; a rigid piece keeps every edge length; the root line
stays on the body while the wing flaps (0.0003 units of drift for a raise of 50° or -30°, against 5–11 at the tip),
0.02 at fold 0.5 and 0.27 at fold 1 (the test allows 5% of the tip's travel, ~0.5); the membrane root lies in the wing's plane on the back and no folded
wing pokes out in front of the torso (it would have failed on the old placement); finger tubes
stay on the sheet; a posed mirrored pair on the body is
an exact mirror image (1e-15) under every control and every preset; body-space skin matrices (`M S M⁻¹`)
reproduce the placed mesh for mirrored placements; and the joints attach to a body `Armature` through the chest and
that armature, driven through `PlacedWing.body_pose`, skins the wing to the finished mesh (next section).

**Driving the wing from a host armature.** A game or DCC pipeline that already has a skeleton for the
body can carry the wing on it instead of re-meshing: the wing's joints are added to the body's `Armature`, the
host skins the placed rest mesh, and the wing follows the torso. Nothing is rebuilt per frame.

```python
host = Armature()                                          # the body's own joints ("chest", ...)
rigs = app.build_rigs(base, verts)                         # [("wing_l", PlacedWing), ("wing_r", PlacedWing)]
for name, pw in rigs:
    pw.attach(host, "chest")                               # joints "wing_l:root", "wing_l:shoulder", ... (body space)
pose = {"chest": (0, 0, 10)}                               # the body's pose ...
for name, pw in rigs:
    pose.update(pw.body_pose(fold=0.6, raise_=20))         # ... plus each wing's: controls, a preset or a raw pose dict
skins = host.skin_matrices(pose)                           # bone name -> 4x4, as for any other shape of the body
sw = rigs[0][1].skin_weights()                             # bone names "wing_l:d3_tip", ..., per-vertex indices + weights
# sw.vertex_groups() -> {bone: (vertex indices, weights)}, sw.per_vertex() -> [{bone: weight}]; skin pw.mesh().vertices
```

`attach` puts the joints at their body-space rest positions, so a rotation stored on one of them turns about
the *host's* axes. A pose from `pose_from_controls` is in the wing's frame; applied as it stands it puts the wing
up to a span away (joints 13.5 units off and skin matrices 16.6 off on a 13-unit wing). `PlacedWing.body_pose` conjugates each joint's rotation by the
placement's rotation `Q` (`Q R Qᵀ`: the uniform scale cancels, a mirrored right wing still gives a proper
rotation) and prefixes the names, after which the host's skin matrices equal `skin_matrices_body` for **every**
bone and its posed positions equal `joint_positions` for every joint (1e-9; measured 1e-14), for both wings of a
pair, three styles. `form="matrix"` returns the 3x3 rotations instead of Euler triples, for a host with its own
(quaternion, local-axis) joints; the placement must be a rotation or mirror times a *uniform* scale, or `body_pose`
refuses (a shear's conjugate is not a rotation). The default prefix is the side, `wing_l` or `wing_r` (a right
wing is the placement with a negative determinant), so a pair attaches one after the other; a name that is taken is
a `ValueError` and nothing is added. `WingRig.skin_weights()` / `PlacedWing.skin_weights()` export the weights by
bone name (`SkinWeights`; `rig.weights()` is the same data by index) and list in `post_processed` the pieces a host
cannot finish.

What the host path reproduces and what it cannot, measured on the pose set of the test (presets, twelve flap
phases, a fold sweep, curls) against `PlacedWing.pose`:

| piece | host skinning vs `pose()` |
| --- | --- |
| every bone, feather, covert, thumb, claw, knob; the propatagium | **exact**, 1e-9 (measured 1e-14), all three styles, any pose |
| the membrane slab (bat, dragon) | **not exact**: mean 0.00–0.85, max 0–3.0 wing units (the span is 13.1; 3.0 at fold 0.9, 2.0–2.2 at fold 1); within 0.45 of the finished sheet (mean ≤ 0.015) near the spread pose: fold ≤ 0.1, curl ≤ 20, the downstroke and the lower half of a flap |
| a feathered wing's every piece | exact, nothing is post-processed |

The slab differs because its pleats, finger-column gather and stretch limiter (`WingRig._apply_pleats`) are a
re-evaluation of the sheet's vertices *after* skinning, which an external skinning engine does not run: through a
host armature a stowed bat/dragon is the raw skinned sheet (at fold 1, 219 grid edges over 1.3× their rest length,
longest 4.2×, against 7 and 2.3× finished). The bones and the fingers are right, the sheet between them is a
crumpled approximation. For the finished shape use `PlacedWing.pose` / `WingRig.pose_mesh`; to keep a
host-driven bat usable, stay near spread (the table) or bake the sheet per pose.

**Euler triples versus matrices.** Most joints of a pose are the Euler form `(rx, ry, rz)` (`R = Rz Ry Rx`) of a rotation
matrix `controls_to_pose` composed, with `ry` confined to [-90°, 90°] (a few, the elbow and the thumb, are stored as the plain
in-plane swing `(0, -deg, 0)`). Where such a joint's true rotation passes `|ry| = 90°` the triple jumps by 180° in `rx` and
`rz` while the rotation turns by under a degree: the feathered wrist does it in one 0.5% fold step at fold 0.52 (180° of
triple, 0.9° of rotation), and 21 of 201 bat fold poses and 54 of 201 angel fold poses have a joint within 2° of the pole.
`body_pose(form="euler")` is affected too, and more: conjugating by the placement turns the axes, so a joint that is smooth in
the wing frame can cross the pole in the host's. Never key or lerp triples. `wingrig.pose_matrices(pose)` and
`body_pose(form="matrix")` give the rotation matrices, continuous in the controls (under 12° per 0.5% of fold, tested), which a
host can blend as quaternions; `wingrig.euler_from_matrix` is the (pole-accurate) conversion the other way.

`tests/test_angel_poses.py` pins the feathered wing's intermediate poses and details: the trailing edge is continuous over the elbow
at fold .25 / .5 / .75 (no two neighbouring tips along the arm further apart than 0.30 of a secondary's length; hinged they are
0.49-0.83; a plan-view "concave depth" was tried first and does not tell the two apart, 2.6% against 0.4% of a secondary), a
curled hand is a graded cup, the outer primaries end in a printable tip, and on four body morphs a stowed wing is joined to the back
by a minimum contact depth and no piece is built wholly inside the torso.

**Validated** (`tests/test_wing_membrane_skin.py`, the membrane's skin): no mid-surface vertex moves
more than 2.5x as far as it did the step before or after (plus 0.15) while `fold` is stepped by 0.5%
or `curl` by a degree, bat and dragon (the old code: 52-108 such steps in 200); the fastest sheet
vertex of a sweep is at most 3x the fastest finger joint (the old curl sweep peaked at 4.7x / 5.4x);
the limiter wins a free chain between two pinned ends that the old one left at 1.22x, is continuous in
the positions, and is what keeps the skin to a handful of edges over 1.3x in the fold, curl and
combined poses above (switching it off, or halving its iteration count, or setting `MAX_STRETCH` to 1.6,
each fails a test); every ring of every finger tube keeps 98% of its rest area under fold 0.9/1.0,
curl 60/150 and fold 1 + curl 150 (the old worst was 0.00-0.12); the slab's inverted area stays
under 0.6% of the sheet for fold 0.5-0.9, 1.2% stowed, and its oriented volume is 99% positive.

**Limits of the rig, stated plainly.**

- Linear blend skinning through a host `Armature` (`PlacedWing.attach`, `body_pose`, `skin_weights`) is exact for
  every piece but the membrane slab; the slab's pleats, gather and stretch limiter are a post-process on the
  vertices that an external skinning engine does not run, so a stowed bat/dragon is its raw skinned sheet (numbers
  above). Use `PlacedWing.pose` / `WingRig.pose_mesh` for the finished shape.
- Pose dictionaries (joint → Euler) are not interpolable: Euler triples do not blend (and jump by 180° across
  `|ry| = 90°`, above), and the fold/pleat state is non-linear in them. Interpolate the **controls**
  (`PoseControls`, or the `flap(phase)` generator) and rebuild the pose per frame, or blend `pose_matrices`.
- The stretch limiter is a fixed-iteration Gauss-Seidel projection (24 sweeps), not an exact
  solver, and it cannot move a finger: where the controls hold two fingers (or the last finger and
  the body) farther apart than the skin can reach, the row between them is stretched evenly by what
  its ends dictate (`fan` 0.55: 1.6x on the row from digit V to the body; `swing` -30: 1.4x; `raise_` 100 and
  `twist` 45: 1.15-1.2x), and the along-ray edges of the root column next to the elbow stay over 1.3x
  (2-2.8x at `swing` 90): the legal control ranges are the protection there, not the limiter. In the
  poses it can fix it brings every edge to the limit; some 40-200 edges sit within a hair above 1.15x
  after 24 sweeps (the tail of the Gauss-Seidel convergence), and edges between two pinned vertices
  are out of its reach.
- Control ranges are measured on the plain body and the anatomy-adjusted one with the default placement; a very different back (a
  slider that moves the shoulder blades by a unit) moves the spine clearance of the angel, whose inner edge is ~0.3 units from the midline.
- The membrane's skin *weights* are unchanged: inverse-distance^4 in plan to the arm and finger
  bones with a top-4 truncation (a mean 5% / worst 36% of the weight is dropped before renormalising, and
  120 of 1173 across-ray grid edges change weights by more than 0.5 in L1). Smoothing them (power 3,
  eps 0.10: 88 such edges) moved the stowed pose enough to take the total folded volume under the
  `0.4 x rest` bound of `test_a_folding_membrane_never_turns_inside_out`, and a soft top-4 made it
  worse (146), so they stay. They matter only through a host armature (the post-process is what
  places the sheet in `pose_mesh`).
- The feathered wing's elbow and wrist still fold linearly in `fold` (a reciprocal-apparatus coupling is modelled as the
  shared fraction, not as a separate linkage), so the half-folded arm is still a Z; the plumage now fans across the elbow
  (see above) instead of hinging with it, and the trailing edge is continuous at .25-.75. What is still visible: at .75 the
  humerus block and the forearm block are two lobes with a V between them on the leading side (the arm hairpin, ~49° at .75),
  and the primaries lie over the secondaries at the wrist with a 0.3-0.5 secondary-length gap between the tips of P1 and S1
  (they overlap in the plane, there is no hole). The wrist junction is not fanned.
- A stowed angel is joined to the back only along the strip where the plank is tangent to it: 1.8-9.3% of the vertices inside the
  torso, the deepest 0.61-0.98 units, on the four morphs (the anatomy-adjusted body is the weakest, 1.8% / 0.61). That is a joint a
  resin print can survive but not a generous one; a wing posed with a little fold, or spread, is still the sturdier print.
- The stowed wing's silhouette is a long taper over a flat plank (lean 11°), not a sheath that follows the lumbar hollow: the hem stands
  off the back by a few tenths where the spine curves in. A rigid feather cannot bend along its length; that is the cost of clearing the buttocks.
- The stowed pair meets in a V but does not cross: tips crossing low on the back would put vertices on the wrong side of the midline.
- The stowed bat/dragon is still a bundle of rods with a ragged sheet among them, not a wrapped sleeve: the per-vertex pleats and
  the stretch limiter leave jagged edges at the hem and the finger needles splay a little (the dragon's are longer now). The stow only puts it where it can be seen: wrist apex and claw clear of the back, the sheet
  and the arm out of the torso, the elbow a joint and no longer a pendant.
- A hard seam crease still runs from the elbow back toward the root corner in the plan and below views of the spread wing: the
  trailing-edge rays that leave the forearm and those that fan out of the elbow meet there (the normals across one column differ by up
  to 55°, in thin cells where the rays are parallel to the forearm). Smoothing the camber across it did not remove it, and re-spacing the
  ray origins did not move it; it needs a different construction of the arm-to-body fan.
- A folded membrane at print thickness cannot be a physically pleated bundle (the folded sheet
  has more volume than the space between the fingers), so a stowed bat/dragon is a damped bundle,
  capped in height, that reads as folded rather than a faithful accordion. Print them spread.
  At fold 1 the three bones of a finger are collinear: what reads is one rod per finger (the
  second phalanx of digit III overhangs its metacarpal by half a unit), not three.
- Fold timing is not linear: the fingers stay straight until fold 0.75, so between 0.3 and 0.75 the
  wing is more open than the same fold value was before (z-extent plateaus, then collapses between
  0.75 and 0.9), and fold 0.3-0.5 poses and the flap phases differ from the previous build on ~80% of
  vertices (mean 0.3-0.5 units): the elbow closes to 15°, the fan law changed and the cupped-wing
  relief applies to every unfolded pose. Sweeping fold in 0.5% or curl in 1-degree steps is now
  free of isolated pops of the mid-surface (it was 52-108 steps in 200); the thickness axis of the slab
  still turns up to 90 degrees in one 0.5% step at a handful of vertices while the fingers are mid-spiral
  (fold 0.85-0.93), and a combined fold + curl 30 sweep has 1-2 such steps of 0.1-0.5 units at the
  free edge of the sliver along digit III.
- Stowed poses are display poses. The wing's weld to the body is best at rest (the plane is
  sunk into the back: 1.8-6.9% of the vertices inside, the deepest 0.9-1.1 units); a stowed feathered wing keeps 1.8-9.3%
  (the lean of 11° took it from 0.4-4.6%), see above.
- The stowed wing still reads partly as scale mail: the top 60% is the covert field, whose round tips are now varied (length,
  aim, tilt) but are still rows of rounded ends, and the lower 40% is the long flight feathers lying parallel under the
  secondaries with only a low ladder of primary tips at the hem. The long scapular/tertial rank that drapes over the coverts of a
  real folded wing is not modelled (the coverts ride on the feathers they lie over, so a pose cannot lay a tertial over its own
  coverts without re-layering the rest pose).
- The leading edge of the underside ends in a sawtooth of small quill tips along the rim (under-marginal row). Rounding the butt
  and sinking the roots into the tube were tried and did not help (the second exposed the bare tube); it is left.
- The spread wing has no display arch (wrist high, hand drooping). Lifting the plane at the wrist moves the feather roots off the
  skeleton's wrist joint (the alula test pins them there), and lifting the joints with it tilts the stowed bones off the back; a
  rest wrist-flex would change the default-controls-equals-rest-mesh contract. `raise_arch` still arches a raised wing.
- The last quarter of the fold (0.75 -> 1.0) is the knuckle concertina, and the in-plane fold of
  a three-bone finger is a spiral on the way to its Z: around fold 0.9 the fingers are mid-spiral
  and the stack of rods and sheet there is the least tidy state of the wing. The sheet follows
  the bent fingers through it; it does not hide the spiral.

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
air beside the figure. `WING_ROOT` is therefore a **body point** (the lower back at the waist).
The wing plane has to contain it: with a fixed roll the plane slopes forward as it descends the
back, and the projected target landed on the *front of the belly* (a folded wing then poked out
of the abdomen — the old placement measured 28 sampled vertices up to 0.85 units in front of the
torso). When `Appendage.root_point` is set the frame is therefore **built from body-space vectors**:
the span axis is the hint itself (about 18° back from vertical in profile), the chord points at the
root, and only the origin follows the body, sunk a fixed distance straight into the back. The
root is exactly in-plane and on the back, the builder passes its plan coordinates to the wing, and
`fold` stows the wing along the same line. (An earlier version projected the hint into the tangent
plane of one quad of the morphed body, so a shoulder or pectoral slider re-aimed the whole wing —
19.5° between the plain and the anatomy-adjusted body, with the spread membrane anywhere from
flat to 6 units behind the back. `test_the_wings_orientation_does_not_depend_on_body_morphs`
checks five bodies.)

The feathered wing overrides the root (`WING_PLACEMENT`): it has no membrane to glue to the waist, so its root only aims
the stowed wing and is the **pivot** it stows about. A stowed wing is a plank that has to clear the shoulder blades *and* the
buttocks (the back is at depth −0.9 at the blades, −0.5 in the lumbar hollow, −0.95 at the buttocks on the plain body), so
the pivot sits 0.27 *behind* the anchor's surface point, 3.5 down and 0.6 toward the shoulder side, **relative to the anchor**
(`Appendage.root_relative`): on an absolute point a heavy back buried the wing (14% inside the torso) and a slim one floated it, and the
frame depended on the body's depth. Now the whole frame is the same on every morph (`test_the_feathered_wings_frame_does_not_depend_on_body_morphs`).
Because the pivot is the root, **the shoulder joint translates** (out of the back) during a fold: 0.35 units at fold 0.25, 0.7 at 0.5 and
1.4 at fold 1 (about 2.6 mm on a 32 mm figure) on the plain body; the root stays put, and the membrane wing's shoulder does not move at all. A host
animator driving the shoulder joint of a feathered wing directly will not reproduce a flat stow: drive the controls and let `body_pose` do it.

A mirrored pair is an exact mirror image, which took two fixes: the mirrored anchor already
carries a mirrored tangent, so negating local Y (not X) leaves a true mirror; and the anchor
binds to the whole face, not `face[:3]`, because a quad and its mirror are stored with different
corner orders.

**Shoulder controls on the body.** The frame's axes are not the body's: its chord axis Z, which `raise_` turns about on a free wing,
maps to body (0.25, −0.88, 0.40) on the back (left wing; the plane normal to (0.47, −0.06, 0.88)), so "raising" swept the span back behind
the figure instead of lifting it — the tip's elevation never rose above ~46°, at `raise_` 100 it was 3.9 units across the midline, a flap cycle moved the
angel's tip 9.9 units fore-aft and 3.6 up and down, and the top of the upstroke was its *lowest* point. `WingRig.placed(matrix)` (what `build_rigs` calls)
now aims `WingSkeleton.flap_axis` / `swing_axis` at the body's fore-aft and vertical axes (`WingSkeleton.place_on_body`; the sign is chosen so
+`raise_` lifts and +`swing` sweeps back on both sides — a mirrored wing's axes are the same wing-frame vectors, so a pair stays an exact mirror).
Raise and swing then act on the whole limb: what the fold alone does (the humerus onto the body line, the stowed lift) and the twist,
raised about the fore-aft axis and swung about the vertical one, with the stow pivot of the feathered wing undone around them. Elevation is
fixed by `raise_` alone (the tip's elevation climbs monotonically over the whole range: bat from 10° to 51°, rest 33°; angel 20° to 70°, rest 43°)
and azimuth by `swing` alone. With both at zero the pose is exactly what it was: fold, stowed lift, twist and the stowed pose are unchanged. A free
rig (and `PlacedWing(rig, matrix)` built directly) keeps the frame as the world, bit for bit; **`rig.placed(...)` makes the rig that placement's**,
so a mirrored pair is two rigs. On a body the angel's hand lag (`raise_arch`) is a turn about the same axis (the hand drops in the plane of the flap)
instead of the out-of-plane flex of a free wing, which tipped it forward.

A flap cycle on the body: `PlacedWing.flap(phase)` sizes it to the wing's range (bat up to 26 at the top of the stroke, down 26; angel up to 45, down 28; `PoseControls.flap`
alone would ask for ±45 and sit clamped). The wrist's elevation over the 8 phases runs 35° → 70° (bat) and 21° → 73° (angel), highest at phase 90-135,
lowest at 225-315, and every vertex of both wings stays ≥ 0.2 units off the midline in every preset and every phase
(`test_every_preset_and_flap_phase_keeps_the_wings_off_the_midline`; the pure folds are held to 0.1: the half-folded angel's inner edge is
0.19 from the spine with no control touched).

## Printing

`thickness` is in wing units. A 32 mm figure makes one unit ≈ 1.9 mm: the 0.16 membrane is
≈ 0.30 mm (right at what a resin printer resolves), feathers 0.27 ≈ 0.51 mm at the shaft ridge and 0.18 ≈ 0.34 mm across the vane; the measured minimum vane thickness is 0.095 ≈ 0.18 mm (at the bevelled edge of an underwing covert); 16% of feather vertices are under 0.15 ≈ 0.29 mm because the tip rows are dense. Bones are inflated
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
