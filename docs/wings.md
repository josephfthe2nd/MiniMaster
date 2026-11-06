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
wing comes out at **4.6**. The literature gives 6–7 for flying foxes; the roost photograph —
foreshortened, wrist flexed — gives 4.7–5.8; the build sits just under the photographs
(`test_membrane_planform_has_a_realistic_aspect_ratio` accepts 4.5–8, a paddle being below 3).
The hand was widened for the box-shaped hand panel (IV and V longer, IV fanned further) and the
deeper scallops give a little of the area back; there is only ~0.08 of margin to the test floor.

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
| V → body | 0.10 (a long, shallow arc) | 0.18 (0.30 swung in to 0.3 units of the humerus) |

The dragon is its own registered style, `A.back_wings("dragon")` / `W.make_wing_rig("dragon")`: the same
skeleton and rig with `SCALLOP_DRAGON`, longer spiny tips, a flatter sheet and ribs 1.35× heavier
(dragon art exaggerates what a bat wing already is). The
digit tips end in short needles — III 0.10 FA, IV and V about half that, II none — and the
scallop arcs meet them at a cusp.

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
bases ~0.034 FA) — but a bone is never thinner than 0.72× the membrane thickness at its base, and
a finger **tapers only to `RIDGE_FLOOR` = 0.55× the thickness** (half the thickness is where it
would sink into the sheet; IV and V used to taper to 0.36 and read as a dark slit). The
humerus keeps the elbow's radius over its outer part and **tapers to half of it at the shoulder**
(`HUMERUS_ROOT_TAPER`): it used to be thickest there, so a constant rod ended abruptly against the
torso like a bolt. It still enters the back as a rod; there is no blended fairing.

## The feathered wing (bird / angel)

Primaries root along the hand, secondaries along the ulna, tertials and scapulars on the
humerus, and five tiers of coverts shingle over their bases.

| row | count | length | width / length | direction |
| --- | --- | --- | --- | --- |
| primaries P1–P10 | 10 | 0.60 … **1.00** (P7) … 0.80 (P10) × the longest primary | 0.20 (P1) … 0.13 … 0.105 (P10) | **+70° (P1) to +8° (P10)**; P10 continues the hand axis. Gaps widen from ~4° (P1–P2) to ~9° (P9–P10; measured 4.2, 4.9, 5.5, 5.4, 7.4, 8.5, 8.3, 8.2, 8.8), so the inner wing is a closed sheet and the outer five are separate fingers |
| secondaries | 14 | 0.62 … 0.70 | 0.23 | +78° … +100° |
| tertials | 7 | 0.52 … 0.38 | 0.26 | +92° … +110° |
| primary coverts | 10 | 0.46 (P1) … 0.32 (P10) **of their own primary** | 0.36 | follow their primary |
| greater coverts | 14 | **0.50** of their secondary | 0.40 | follow their secondary |
| median coverts | 14 | **0.30** of their secondary | 0.52 | follow their secondary |
| tertial coverts | 7 + 7 | 0.50 / 0.30 of their tertial | 0.42 / 0.55 | follow their tertial |
| lesser coverts | 2 rows of 22 and 26 | 0.17 / 0.13 of a secondary (**~15%**) | 0.75 | straight back from the arm |
| marginal coverts | 1 row of 36 on the nose of the arm, 1 of 28 below | 0.15 / 0.12 of a secondary | 1.0 / 0.95 | root on the front of the arm and wrap over (under) it |
| alula | 3 | 0.32 / 0.29 / 0.26 of the longest primary | 0.20 | rooted on the leading edge at the wrist, **26° / 18° / 10° off the hand axis**, on top of the primary bases |
| scapulars | 7 | 0.70 … 0.56 FA | 0.26 | +98° … +112° |

The longest primary is ~1.65 FA, so a primary is 5–10× as long as it is wide: a long narrow
blade, not a paddle. The hand is only ~0.58 FA (the first build gave it 0.94), which is what
makes the primaries root at the right spacing (~0.065 FA) and overlap ~65% at the base.

**One feather.** A bare quill for the first tenth of the length, then a vane that widens to
its maximum by 35–40% and holds parallel-sided before closing onto a **rounded point**
(`tip_floor`, never a bar: the last 3% is bevelled to 60% of the sheet so the end does not
stand up as a flat wall, which read as a spike on every tip). The rachis is off-centre —
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
are 0.40 as wide as long, medians 0.52. Their quill ends are buried under the row in front.

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

Total: 282 pieces, ~88k triangles (was 306 pieces, 98k).

The tertials and scapulars trail at +92…+112° rather than hanging straight down the body line (+131°
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
| a finger or the arm | blended **only over its own chain's bones**, by proximity — it bends smoothly at its knuckles and is never pulled by the finger next door |
| the membrane | blended over the arm and finger bones by distance in the wing plane; the **root edge is bound to a body bone**, so the sheet stays glued to the flank however the arm moves |

**Controls** (clamped to anatomical limits, `wingrig.LIMITS`; the caller's object is never
modified):

| control | meaning | range |
| --- | --- | --- |
| `fold` | 0 spread … 1 stowed | 0 … 1 |
| `raise_` | shoulder elevation (flap); the top shrinks with `fold` (100° spread, 25° stowed). The feathered wing **arches**: the wrist flexes `raise_arch` (0.30) × raise the other way, so a raised wing's hand hangs below the wrist line and a lowered one's rides up, instead of the limb swinging as one plank | −70° … +100° |
| `swing` | humerus swung back (+) or forward; the top shrinks with `fold` (90° spread, 5° stowed) | −30° … +85° |
| `twist` | roll about the span axis | ±45° |
| `fan` | finger spread, 0 together … 1 rest. A bird's primaries spread about the hand axis; a membrane clamps it to 0.55 … 1.2 (a sheet tears long before the fingers close) | 0 … 1.5 |
| `curl` | finger knuckle flexion; on the feathered wing it also **cups the hand**: each primary droops toward the underside by `curl_cup` (0.35) × curl × its rank P1 … P10, so the outer primaries bend most (curl 60 drops P10's tip 2.7 units, P1's none) | −10° … 150° |
| `wrist_flex` | cocks the hand out of the wing plane at the wrist, tip up (+) or drooping (−) | −60° … +45° |
| `hand_twist` | rolls the hand about its own axis (wash-out) | ±30° |

**Presets** (`wingrig.PRESETS`): `glide`, `half_fold`, `folded`, `upstroke` (wrist up, arm
half-folded, leading edge edge-on), `downstroke`, `mid_downstroke`. `PoseControls.flap(phase)`
generates a cycle in which the shoulder leads while the wrist (40°) and the finger curl (70°) lag it,
so the hand trails the wrist — hanging below it on the upstroke — instead of the whole wing swinging
like a plank. The upstroke is not the downstroke played backwards: the arm half-folds and the fingers
close to shed the air, then the arm extends and the hand pronates for the power stroke.

Feather headings are computed from the **in-plane** rotations only, so lifting, twisting or cocking a
wing tilts its plane without changing where the feathers point within it: the 3D angle between the
first and last primary stays 61° under any raise, twist or wrist flex (a first version read the plan
projection of the tilted frame and spread the fan to 166° at raise 100).

**Folding.** Stowing closes the Z of the arm: the humerus tucks onto the **body line** (the
direction from shoulder to root — *not* the frame's +Z, which on a real placement points
down-and-outward, so folding toward it splays the wing sideways), the forearm swings forward
onto it to an elbow interior of ~15° (the bat's was 25° before; the bird's was already 14°), the hand folds back along the forearm (wrist 172°), the fan collapses, and the thumb
tucks along the forearm. The stowed wing is also
**lifted off its plane** (bat 12°, bird 20° — a feathered wing is deep): that plane is sunk into the back so a print is
welded, and a wing lying along it would be buried in the torso. Feathers fold like a closing fan,
each turning about its own root until it lies along the body line; scapulars, the alula and the
lesser coverts fold with them.

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
bone length is preserved) until the plane lies on the back: `stow_lean` (6°) leaves the lower end standing off it, which is what
clears the buttocks and sinks the top of the wing into the shoulder blade (the weld);
`stow_toe_in` (8°) turns the wing's axis toward the spine, so a pair stows in a **V** that closes toward the hem
(`x` clearance of the stowed wing 0.3 on the plain body, never below 0.1 on any morph tested); `stow_pitch` (8°) lifts every
flight feather about its root, because a primary droops 0.16 of its length toward the body (its camber) and a flat-stowed tip
would dig into the buttocks; `stow_taper` (6°) turns the edge feathers toward the middle of the bundle, so it narrows from the broad
covert mass to the primary tips (tip band 16% narrower than parallel feathers leave it). The wrist, the top of the forearm's
leading-edge covert rope, ends outboard and above the shoulder, a rounded bow clear of the back.
On the plain body 3% of the stowed wing's vertices are inside the torso (it was 28%) and 2% of the flight feathers (6% before); the
slim and heavy morphs stay between 0.3% and 7%; none cross the midline.

**Feathers follow their bone.** The feathers no longer swing to the stowed heading in a straight line: until `stow_follow`
(30%) of the fold is done each follows the bone it grows from, then eases to the stowed heading; the primaries close their fan on the
way (smoothstep: 65% closed by half fold, 90% by three-quarters) so a half-folded wing fans out of its arm rather than hinging
like a plank.

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

Skin is elastic, so nothing pleats until a panel has closed by ~10% — and a wing that is only
*cupped* (curled or flapping, the hand not yet folded back: `WingRig._hand_fold`, read off the
angle between forearm and hand) gives a further 8% and buckles half as hard, fading out by the
time the hand is 30% of the way back, so a curl stays a smooth finger arc under a taut sheet
instead of rucking at the tip. Each panel pleats in several fine lobes (1/2/2/3) rather than one
bladder, and a pleat is softly capped (`PLEAT_CAP`, 0.30 units per lobe, about one bone radius): a
stowed membrane is a bundle lying among the fingers, not a tent three finger-thicknesses tall
(the folded bat stood 4.0 units off the plane before the cap was introduced; 0.45 per lobe gave 2.3
and left lobes proud of the rods). The slab keeps its grid's own orientation (re-orienting cells toward a skinned "up" vector turned the whole shell inside
out where a panel doubled back: signed volume fell from +0.75 to -0.03), and a stretch limiter pulls
any edge back toward 1.15× its rest length. The long inboard panel pleats less: it hangs as a drape.
The propatagium (the skin in front of the arm) is a separate piece skinned to the arm: it hinges
with the elbow as a flat triangle and never leaves the arm farther than at rest.

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
`grid_normals` now blends toward a reference normal (the sheet's mean normal at rest, the skinned
"up" vector when posed) wherever the tangents are within ~20° of parallel, and is bit-identical
elsewhere. Separately, the camber weight stepped from 0.25 to 1.0 at digit V, leaving a 0.13-unit
ledge down the finger (more than half the membrane's thickness); it is now smoothed over nine
columns. The dragon is part of the rig test matrix.

**Validated** (`tests/test_wingrig.py`): the rest pose reproduces the mesh to 1e-12; weights are
a partition of unity with ≤4 influences; rigid pieces use exactly one bone; a finger is pulled
only by its own bones; bones never change length under any pose; every pose of both styles is
watertight per piece; the root stays on the body while the wing flaps (≤0.16 units of drift
against 8–12 at the tip); the membrane root lies in the wing's plane on the back and no folded
wing pokes out in front of the torso (it would have failed on the old placement); finger tubes
stay on the sheet; a posed mirrored pair on the body is
an exact mirror image (1e-15); body-space skin matrices (`M S M⁻¹`) reproduce the placed mesh
for mirrored placements; and the joints attach to a body `Armature` through the chest.

**Limits of the rig, stated plainly.**

- Linear blend skinning through a host `Armature` (`PlacedWing.attach`, `skin_matrices_body`)
  moves the bones, feathers and the part of the membrane that rides a finger. The membrane's
  between-finger re-evaluation (pleats, stretch limiter) is a *post-process on the vertices* that
  an external skinning engine does not run: through a host armature a stowed membrane is the raw
  skinned sheet. Use `PlacedWing.pose` / `WingRig.pose_mesh` for the finished shape.
- Pose dictionaries (joint → Euler) are not interpolable: Euler triples do not blend, and the
  fold/pleat state is non-linear in them. Interpolate the **controls** (`PoseControls`, or the
  `flap(phase)` generator) and rebuild the pose per frame.
- The stretch limiter is a fixed-iteration Jacobi pass, not a solver: under extreme controls a
  few edges stay a little over 1.15x.
- The feathered wing's elbow and wrist still fold linearly in `fold` (a reciprocal-apparatus coupling is modelled as the
  shared fraction, not as a separate linkage), so the half-folded arm is still a Z; what changed is that the plumage no longer
  hinges with it. A half-folded wing reads as a bird's M-shaped arm with the fan closing, not a rigid plank, but it is not smooth.
- A stowed angel touches the back rather than fusing with it: the under-coverts lie against the blade (3% of vertices inside on the
  plain body, 1% on a slim one). Print it posed with a little fold, or spread.
- The stowed wing's silhouette is a long taper over a flat plank (lean 6°), not a sheath that follows the lumbar hollow: the hem stands
  off the back by a few tenths where the spine curves in. A rigid feather cannot bend along its length; that is the cost of clearing the buttocks.
- The stowed pair meets in a V but does not cross: tips crossing low on the back would put vertices on the wrong side of the midline.
- A folded membrane at print thickness cannot be a physically pleated bundle (the folded sheet
  has more volume than the space between the fingers), so a stowed bat/dragon is a damped bundle,
  capped in height, that reads as folded rather than a faithful accordion. Print them spread.
  At fold 1 the three bones of a finger are collinear: what reads is one rod per finger (the
  second phalanx of digit III overhangs its metacarpal by half a unit), not three.
- Fold timing is not linear: the fingers stay straight until fold 0.75, so between 0.3 and 0.75 the
  wing is more open than the same fold value was before (z-extent plateaus, then collapses between
  0.75 and 0.9), and fold 0.3-0.5 poses and the flap phases differ from the previous build on ~80% of
  vertices (mean 0.3-0.5 units): the elbow closes to 15°, the fan law changed and the cupped-wing
  relief applies to every unfolded pose. Sweeping fold or curl in 1% steps can move a few membrane
  vertices by 1-1.8 units in one step (a gather switching columns); a full flap cycle does not.
- Stowed poses are display poses. The wing's weld to the body is guaranteed at rest (the plane is
  sunk into the back); a stowed feathered wing keeps only 0.4-4.6% of its vertices inside the
  torso (plain 1.8%, adjusted body 0.4%, heavy 4.6%), i.e. barely joined: print wings spread.
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
Because the pivot is the root, **the shoulder joint translates about one unit** (out of the back) during a fold; the root stays put.

A mirrored pair is an exact mirror image, which took two fixes: the mirrored anchor already
carries a mirrored tangent, so negating local Y (not X) leaves a true mirror; and the anchor
binds to the whole face, not `face[:3]`, because a quad and its mirror are stored with different
corner orders.

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
