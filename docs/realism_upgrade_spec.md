# Realism upgrade spec (synthesized + critiqued)


## whats_wrong_now

NOTE ON INPUTS: the eight research reports referenced in the brief did NOT arrive in my prompt (no JSON payload was attached). I therefore synthesized from (a) the actual source, (b) the prior-session research artifacts still on disk (MakeHuman algos3d.py/human.py/targets.py, a scraped modifier index mh_mods.json with 249 modifiers, MetaHuman OpenRigLogic/DNA docs), and (c) NEW MEASUREMENTS I ran against the live code. Every number below is either read out of the repo or measured, not recalled. Web search budget for this session was exhausted, so external citations are limited to the local research artifacts.

THE ONE-LINE DIAGNOSIS: the cage is not too coarse to be *smooth* — it is too coarse to be *legible*. Catmull-Clark is a low-pass filter, and every facial feature in the current head is authored at or below the cage's Nyquist limit, so the subdivider deletes it. You are spending 22,528 triangles to render an egg.

=== EVIDENCE 1: I rendered the current head. ===
/tmp/claude-0/-home-user-MiniMaster/d099e56a-afda-5e64-b63b-1d4150e4051f/scratchpad/face_front.png, face_3q.png, face_side.png (subdiv 3, 22,528 tris, smooth-shaded). What renders: an egg with two black dots. NO mouth is visible at all. The "protrude-recess-protrude 3-ring lip sandwich" that docs/character_lab_spec.md describes at length produces ZERO visible lip line. No nostrils, no alae, no lid, no ear, no jawline, no brow shadow. In profile the nose is a soft swelling, not a nose. This is not a tuning problem.

=== EVIDENCE 2: I measured the Catmull-Clark attenuation directly. ===
Displacing head cage verts by 0.100 Hh and measuring peak displacement on the limit surface:
  support = 1 vertex        -> 0.0451  (45% survives)   [subdiv 3; 56% at subdiv 1]
  support = 3-vertex row    -> 0.0672  (67% survives)
  support = 2x3 vertex block-> 0.0961  (96% survives)
THE RULE THIS IMPLIES: a feature needs at least a 2x2 block of cage vertices to survive subdivision. Count the current head's feature support in topology.py NEUTRAL_OFFSETS: nose tip = 1 vertex. Nose dorsum = 1. Mid-bridge = 1. Glabella = 1. Chin ball = 1. Nose wings = 2 (one per side, isolated). Mouth corners = 2. Inner canthus = 2. Every single one is at or below the 45-67% band, and they are competing against a rear-biased skull ellipsoid whose own radii are 0.26-0.33 Hh. A 0.021 Hh nose tip on a 0.30 Hh head is a 7% perturbation that arrives at the surface as 3%. It is invisible by construction. The generator's comment already admits this ("Deltas are authored ~2x their target read... subtle offsets vanish") — the author saw the symptom and treated it with gain instead of topology.

=== EVIDENCE 3: I ran the decisive experiment. ===
I took the SAME head cage, refined it once with Catmull-Clark to get a 706-vert/704-quad "dense cage" proxy, machine-selected 4 crude feature bands by coordinate predicate (lip seam 11 verts, lid crease 10, nasolabial 34, alar crease 4 — no artistry whatsoever), pushed them in 0.022-0.045 Hh, and subdivided ONCE.
  dense cage @ subdiv 1 =  5,632 tris  -> exp_dense_L1_front.png / exp_dense_L1_3q.png
  current cage @ subdiv 3 = 22,528 tris -> exp_orig_L3_front.png / exp_orig_L3_3q.png
The 5,632-triangle version has a visible mouth, visible lid creases, a defined brow, and a nose with alar definition. The 22,528-triangle version is an egg. FOUR TIMES CHEAPER AND VASTLY MORE FACE. This is the single most important finding in this document: you are not resource-constrained, you are topology-constrained, and each additional subdivision level actively DESTROYS information while quadrupling cost. Extra subdiv levels are not a detail knob; they are a blur knob.

=== EVIDENCE 4: the counts, versus what the job requires. ===
Current head cage: 178 verts / 176 quads, 11 rings x 16 columns. Column spacing at the front is 13-15 degrees; the mouth corner (c2, 28 deg) sits at x=0.122 Hh while the mouth aperture needs to be ~0.20 Hh wide — the mouth is literally wider than the vertices allotted to describe it. Vertical ring spacing across the lips is 0.05 Hh; the vermilion border needs a discontinuity finer than 0.02 Hh. There is no loop that follows any anatomical boundary: the eye aperture, the lid crease, the vermilion border, the alar crease, and the nasolabial fold are the five curves that make a face read, and the cage contains a loop for exactly ZERO of them. Every ring is a horizontal slice of a rotated ellipse.
Reference scale for what "customizable realistic face" costs elsewhere: MakeHuman (mh_mods.json, scraped in a prior session) ships 249 modifiers of which 146 are head/face — 34 for eyes alone, 22 for mouth, 21 for nose, 22 for ears. Our 22 face sliders are 15% of that, and they drive a cage with 4% of the loops needed to express them.
Body cage: 403 verts / 401 quads. Specific failures: 8-vertex limb rings (an arm cross-section octagon); SINGLE pinch rings at elbow (S3), knee (L2), wrist (S5), ankle (L5) instead of joint LOOP PAIRS, so linear blend skinning collapses the joint volume on any real bend; deltoid/biceps/triceps/calf are single-vertex nudges at 45% survival; MITTEN hands with zero fingers and a 2-vertex "thumb bump"; feet with a ball ring and a toe pole and no toes, no malleoli, no arch; no clavicle, no sternum, no scapula, no linea alba, no navel, no spine groove, no iliac crest loop, no costal margin loop.

=== EVIDENCE 5: the shading model is from 1978 and is half the problem. ===
render.py line ~150: shade = 0.42 + 0.58 * max(N.L, 0). That is it. No specular. No Fresnel. No ambient occlusion. No subsurface term. No shadows. ONE flat color per shell (render_meshes takes list[(Mesh, "#rrggbb")]), so the eyeball is a solid #2e2b28 ball — there is no sclera, which is why the eyes read as two drilled holes. No sRGB->linear conversion and no tone map: base_color is multiplied by shade in sRGB space and written straight to the PNG, which is exactly the transfer function that makes everything look like unfired clay. Ambient 0.42 is enormous — it crushes every normal variation by 42% before it starts, so even the creases you DO have lose nearly half their contrast. And _smooth_corner_normals is a Python loop over faces x corners with a dict incidence map, which is why smooth shading costs 1.92s vs flat 0.88s at subdiv 2 (512x512, measured).

=== EVIDENCE 6: a real bug. ===
docs/character_lab_spec.md specifies "head r0 ring a,f,b x(1+0.3*(neck_thick-1)) so the throat overlap never opens a gap." generator.py head_verts_local() reads only face_width_eff and face_length. The neck_thick coupling was never implemented, so at neck_thick=0.75 the body's neck tube (radius 0.042*0.75 = 0.0315 H) can shrink inside the head's fixed throat ring and open a visible hole at the seam. Fix this regardless of the rest of the plan.

=== MEASURED BASELINE (for the gates below) ===
build: L1 9ms / L2 31ms / L3 124ms.  verts,tris: L1 2,396/4,776 | L2 9,320/18,624 | L3 37,016/74,016.
render 512x512: L1 flat 0.23s smooth 0.51s | L2 flat 0.88s smooth 1.92s.

## target_head_topology

TARGET: 618 verts / 616 quads for the head shell, plus 2 separate ear shells (62v/60q each) and 2 eyeballs. That is 3.5x the current 176 quads. Euler characteristic verified below at every step: for a closed all-quad mesh E = 2F, so chi = V - E + F = V - F, and V - F must equal 2 at every stage. I checked each island's arithmetic against that invariant.

=== BASE SKULL GRID: 24 columns x 19 rings ===
Keep the loft-of-rings structure (it is what makes the code index-computable), keep the TRUE FRONT MIDLINE column, but double the column count and nearly double the rings, with NON-UNIFORM column spacing biased to the face. Columns are dense where curvature is high (midline, eye, mouth) and sparse across the occiput where nothing happens.

COLUMN AZIMUTH TABLE (psi from front -Y toward +X; c13..c23 mirror as c <-> (24-c)%24):
  c0  =   0 deg  front midline: nose ridge, philtrum, chin point
  c1  =   8 deg  philtrum wall / nasal dorsum side / cupid's bow peak
  c2  =  17 deg  alar wall / lip body
  c3  =  26 deg  inner canthus / alar crease / mid-lip  [was c2 at 28 deg = the ONLY eye column before]
  c4  =  36 deg  mid-eye / mouth corner
  c5  =  48 deg  outer canthus / nasolabial fold
  c6  =  62 deg  cheekbone crest (malar) / masseter
  c7  =  78 deg  temple / gonial angle
  c8  =  96 deg  ear line (ear shell roots here)
  c9  = 118 deg
  c10 = 145 deg
  c11 = 165 deg
  c12 = 180 deg  back midline (occipital)
The gain that matters: the eye region now owns FOUR columns (c3,c4,c5 + midline neighbours) instead of one, and the mouth owns c0..c4 per side = 9 columns across the aperture instead of 5.

RING STACK, bottom to top, 19 rings (z in Hh; each named for the anatomical boundary it must follow, not for a slice height):
  r0  throat tuck (sits inside the body neck tube)      z 0.02
  r1  mandible underside / submental                    z 0.07
  r2  jaw base / chin base                              z 0.12
  r3  labiomental crease (chin-lip fold)                z 0.17   <- crease loop
  r4  lower lip lower margin                            z 0.20
  r5  lower lip body (vermilion)                        z 0.225
  r6  MOUTH SEAM                                        z 0.245  <- aperture, island rim
  r7  upper lip body (vermilion)                        z 0.265
  r8  cupid's bow / vermilion border                    z 0.285  <- crease loop
  r9  philtrum mid                                      z 0.32
  r10 subnasale / alar crease / nose base               z 0.37   <- crease loop
  r11 mid-nose / lower malar                            z 0.43
  r12 lower orbital rim / cheekbone crest               z 0.49
  r13 EYE LINE / palpebral fissure                      z 0.53   <- aperture, island rim
  r14 upper lid crease                                  z 0.575  <- crease loop
  r15 brow ridge / supraorbital torus                   z 0.63
  r16 forehead mid                                      z 0.71
  r17 frontal eminence / hairline                       z 0.82
  r18 crown ring                                        z 0.93
  + top pole (crown), + bottom pole (throat)
Design rule made explicit: FIVE of these rings exist only to be creases (r3, r8, r10, r14, plus the two aperture rings). A crease is a ring pair with SMALL z-spacing (0.02-0.03 Hh) and an out-of-plane offset between them; a single displaced ring at 0.05 Hh spacing is what the current cage does and it produces a swell, not a crease.

BASE GRID COUNT: 19 x 24 = 456 verts; bands 18 x 24 = 432 quads; crown fan 12 quads + 1 pole; throat fan 12 quads + 1 pole. V=458, F=456, V-F=2. OK.

=== FEATURE ISLANDS (cut a patch, bridge to concentric loops) ===
This is the same machinery the body already uses for armholes (HOLE_QUADS + _bridge in topology.py), so it is proven code, not new risk. For a cut of a x b quads: rim = 2(a+b) verts, orphaned interior = (a-1)(b-1) verts.

EYE ISLAND, one per side. Cut 3 cols x 2 rows = 6 quads spanning c3..c5 / r12..r14. Rim = 10 verts, orphans = 2.
  L1 orbital rim loop   (10 verts) - the bony socket edge; drives eye_depth, socket recession
  L2 lid crease loop    (10 verts) - the supratarsal fold; drives hooding, crease height
  L3 LID MARGIN loop    (10 verts) - the actual eyelid edge, the silhouette against the eyeball
  L4 inner funnel loop  (10 verts) - pushed 0.06 Hh back into the skull
  funnel cap: 5-quad fan to 1 pole, deep inside the head
  quads: 10(bridge) + 10 + 10 + 10 + 5 = 45.  net V = 41-2 = +39, net F = 45-6 = +39. OK.
Three concentric loops is the minimum for an eye that reads: L1->L2 is the orbital hollow, L2->L3 is the eyelid PLANE (a lid is a surface, not a line), L3 is the margin. The eyeball sphere protrudes through L3/L4 and the shell stays closed and watertight, so the existing CC kernel (which raises on any edge not shared by exactly 2 quads) and the STL gate both keep working unchanged.

MOUTH ISLAND, one, spanning the midline. Cut 6 cols x 4 rows = 24 quads (c3-left through c3-right, r4..r8). Rim = 20 verts, orphans = 15.
  L1 nasolabial / mentolabial ring (20 verts) - the fold that frames the mouth mass
  L2 VERMILION BORDER   (20 verts) - skin-to-lip transition; the hard line
  L3 lip body           (20 verts) - the fleshy roll
  L4 mouth seam         (20 verts) - collapses toward the slit
  seal: 10-quad fan to 1 pole, pushed back into the oral cavity
  quads: 20 x 4 + 10 = 90.  net V = 81-15 = +66, net F = 90-24 = +66. OK.
The three loops L2/L3/L4 replace the current r2/r3/r4 "sandwich" that provably does not read. The difference is that these loops are CONCENTRIC (they wrap the aperture) rather than horizontal slices, so the lip corner is described by loop geometry instead of by two lonely vertices at (3,2) and (3,14).

NOSTRIL ISLAND, one per side. Cut 2x2 = 4 quads at c1..c2 / r9..r10. Rim = 8, orphans = 1.
  L1 alar rim loop (8 verts) + 4-quad cap fan to 1 pole diving up into the nostril.
  net V = 9-1 = +8, net F = 12-4 = +8. OK. x2 = +16/+16.
A nostril is a hole in shadow. It costs 8 quads and it is the difference between "nose" and "lump".

=== HEAD SHELL TOTALS ===
V = 458 + 78(eyes) + 66(mouth) + 16(nostrils) = 618
F = 456 + 78 + 66 + 16 = 616
chi = 618 - 616 = 2. Watertight, all-quad, CC-legal.
  subdiv 1: 2,464 quads / 4,928 tris  <- SHIP AT THIS LEVEL for preview AND for most prints
  subdiv 2: 9,856 quads / 19,712 tris <- beauty renders and >54mm prints
  subdiv 3: 39,424 quads / 78,848 tris <- never; it only blurs

=== EARS: SEPARATE SHELLS, NOT GRID INSERTS ===
6 rings x 10 cols + 2 poles = 62 verts / 60 quads each, an oval dish with a concha depression (rings 2-3 pulled in 0.05 Hh), a helix rim (outer ring rolled forward), a tragus bump and a lobe. Rooted at c8, sunk 0.03 Hh INSIDE the skull surface so the intersection line hides behind the ear's own silhouette. ear_size / ear_protrusion / ear_rotation / ear_height become a 4x4 transform on the shell — free sliders, zero topology risk. subdiv 1: 480 tris each.
Rejected alternative: grid-inserting the ear. An anatomical ear needs a spiral (helix into antihelix into concha), which forces 3 extra poles into the skull grid at c7-c9 and a non-planar cut. It is the one feature whose topology genuinely does not want to be part of the skull.

=== POLE INVENTORY (all CC-safe, all hidden) ===
crown pole (valence 12, under hair/at the top where nothing is looked at); throat pole (inside the neck); 2 eye funnel poles (inside the skull); 1 mouth pole (inside the oral cavity); 2 nostril poles (inside the nose); 2 ear poles per ear. NOT ONE POLE IS ON A VISIBLE, CURVATURE-CRITICAL SURFACE. That is the rule: poles go in cavities. Extraordinary vertices of valence 3 and 5 appear along every island rim, which is normal and is what a hand-modeled head looks like.

## target_body_topology

TARGET: 973 verts / 971 quads with grooved hands and feet (2.4x current), or 1,133 / 1,131 with true finger tubes. chi = V - F = 2 verified.

=== DECISION FIRST: keep the body MIDLINE-FREE. ===
The head needs a front midline column because the nose ridge, philtrum and chin are RIDGES and points — they want a vertex on the centre. The torso's midline features (sternum, linea alba, navel, spine groove, gluteal cleft) are VALLEYS, and on a subdivision surface a valley is made correctly by pulling two adjacent columns inward, not by displacing one centre column. So the phase-offset azimuth trick ((k+0.5)*15 deg for 24 columns) survives, the proven 1-quad crotch survives, and we still get a sternum groove and a spine groove for free from column pairs c11/c12 (front, +/-7.5 deg) and c3/c4 (back). This resolves the "we need a midline for the torso" pressure without touching the crotch construction that the spec spent a paragraph verifying edge-by-edge. Do NOT attempt a column-transition band to weld head and body; that was the right call before and it is still the right call.

=== TORSO: 24 columns x 14 rings ===
Rings, each named for an anatomical boundary:
  r0  pelvic floor / crotch          z 0.485
  r1  hip / greater trochanter        z 0.520
  r2  ILIAC CREST                     z 0.565   <- new; the ridge that makes a pelvis read
  r3  navel / waist                   z 0.615
  r4  lower ribs (floating)           z 0.655
  r5  COSTAL MARGIN (rib arch)        z 0.685   <- new; crease loop, the ribcage's bottom edge
  r6  nipple / pectoral line          z 0.720
  r7  upper chest / sternum mid       z 0.750
  r8  ARMPIT FLOOR                    z 0.768   <- armhole island bottom
  r9  CLAVICLE / acromion             z 0.792   <- new; clavicle is a column-pair ridge here
  r10 shoulder top / trap base        z 0.812
  r11 neck base (trap slope)          z 0.832
  r12 mid neck                        z 0.855
  r13 neck top                        z 0.878
14 x 24 = 336 verts. Bands 13 x 24 = 312 quads. Neck cap fan 12 quads + 1 pole.
The R7->R8 "big step subdivides into the trapezius slope" hack in the current code is replaced by real rings r9/r10/r11: a trapezius is a slope from the acromion to C7 and it needs three loops to describe, not a gap.

ARMHOLE ISLANDS: cut 4 cols x 2 rows = 8 quads per side (r8..r10), rim = 12 verts, orphans = 3. Rim 12 matches the 12-vertex arm ring, so the bridge is 12 quads and there is no vertex-count transition. Removing 8 quads per side = -16, orphans -6 verts.

CROTCH: unchanged, 1 quad, splitting r0's 24 verts into two 12-vertex leg loops. Verified in the existing spec by directed-edge cancellation; it still cancels at 24 columns.

=== LIMBS: 12-VERTEX RINGS, WITH JOINT LOOP PAIRS ===
The 8-vertex ring is the second-worst decision in the body cage (after mitten hands): an octagon cannot describe a cross-section that is simultaneously flat on one side (tibia, ulna) and round on the other, and 8 verts leaves 2 verts per quadrant which is not enough to place a muscle belly.
THE JOINT LOOP PAIR RULE (this is the load-bearing change): every joint that bends gets TWO rings straddling it at ~0.02 H spacing, not one pinch ring. Under linear blend skinning a single ring at the joint is weighted between two bones and collapses to a hinge crease at 90 degrees; a pair lets the outer ring follow the extending bone and the inner ring follow the flexing one, preserving volume. This costs 12 verts per joint and fixes the elbow/knee collapse for free, before you consider dual quaternions.

ARM, 11 rings x 12 verts = 132 verts per arm:
  S0 shoulder root (at the armhole rim)
  S1 deltoid upper
  S2 deltoid lower / biceps top
  S3 biceps belly / triceps belly
  S4 ABOVE ELBOW  <- pair
  S5 BELOW ELBOW  <- pair
  S6 forearm flexor bulk (high, just below the elbow)
  S7 forearm mid
  S8 wrist
  S9 palm base
  S10 knuckle line (metacarpal heads)
Bands 10 x 12 = 120 quads per arm.

HAND (default "grooved" tier): after S10 add F0 finger-mass ring (12) + F1 (12) + tip pole = 25 verts; 12+12+6 = 30 quads. Inter-finger grooves are made by pulling 3 alternating vertex COLUMNS of the F0/F1 band inward — three grooves, four finger lobes, on a 2-ring support block so they actually survive subdivision. THUMB IS A REAL TUBE: cut 2 quads on the palm (rim 6 verts, 0 orphans), bridge to a 3-ring x 6-vert tube + pole = 19 verts, 6+12+3 = 21 quads minus the 2 cut = 19. The thumb is the single highest-value hand feature: it is what makes a hand read as a hand in silhouette, and it is the one digit big enough to print at 30 mm.
  per hand: 25 + 19 = 44 verts, 30 + 19 = 49 quads.

LEG, 11 rings x 12 verts = 132 verts per leg:
  L0 thigh top (at the pelvic loop)
  L1 upper thigh
  L2 mid thigh
  L3 ABOVE KNEE  <- pair
  L4 BELOW KNEE  <- pair
  L5 calf belly (high and posterior - gastrocnemius)
  L6 lower calf / achilles taper
  L7 ankle + MALLEOLI (the two ankle knobs: 2 vertex bumps, medial higher than lateral)
  L8 instep
  L9 ball of foot
  L10 toe base
Bands 10 x 12 = 120 quads per leg.
FOOT: T0 toe ring (12) + tip pole = 13 verts, 12 + 6 = 18 quads. Toes are GROOVED, not tubed — 4 grooves on the T0 band. Add an arch by pulling the medial verts of L8/L9 up 0.008 H.

=== BODY TOTALS (grooved tier) ===
V = 336(torso) + 1(neck pole) - 6(armhole orphans) + 264(arms) + 88(hands) + 264(legs) + 26(feet) = 973
F = 312 + 12 - 16 + 1(crotch) + 24(arm bridges) + 240(arm bands) + 98(hands) + 24(leg bridges) + 240(leg bands) + 36(feet) = 971
chi = 2. OK.
  subdiv 1: 3,884 quads / 7,768 tris
  subdiv 2: 15,536 quads / 31,072 tris
FINGERS TIER (optional): 4 finger tubes per hand, each 3 rings x 6 verts + pole = 19 verts / ~19 net quads, cutting into the F1 band. +80 V / +80 F per hand -> 1,133 verts / 1,131 quads total.

=== THE HEADLINE BUDGET NUMBER ===
Head 616 + body 971 = 1,587 cage quads.
  v2 @ subdiv 1 = 6,348 quads / 12,696 tris (+ ears 960 + eyes 640) = ~14,300 tris
  v1 @ subdiv 2 (today's shipping setting) = 18,624 tris (MEASURED)
THE NEW CAGE AT SUBDIV 1 IS CHEAPER THAN THE CURRENT CAGE AT SUBDIV 2, and it carries eyelids, a mouth, nostrils, a thumb, malleoli, a clavicle and joint loop pairs. Print/beauty tier is v2 @ subdiv 2 = ~55,300 tris, roughly 3x today's cost for something categorically different. Build time extrapolates to ~90ms at subdiv 1, ~350ms at subdiv 2 (current: 9ms / 31ms) — still interactive for the cage, not for the beauty render.

=== RIG ===
The 16-bone armature must go to ~24: add clavicle_l/r (the shoulder currently pivots at the chest, which is why the deltoid tears), split spine into spine1/spine2 (one spine joint cannot do both lordosis and kyphosis), add forearm_twist_l/r (a wrist rotation currently candy-wraps the whole forearm), add ball_l/r for the foot. Skin weights get re-authored per RING rather than per raw id; with joint loop pairs the weight ramp is 1.0 / 0.75 / 0.25 / 0.0 across four rings instead of a hard 0.5 at a single pinch ring.

## new_cage_features

Ranked by realism gained per cage quad spent. "Worth it" verdicts are for the stated goal (realistic human, print 28-32mm but also close-up renders).

1. EYELIDS — 3 concentric loops per eye (orbital rim, lid crease, lid margin). Cost 39 quads/eye. VERDICT: MANDATORY, HIGHEST VALUE ITEM IN THE ENTIRE PLAN. A bare sphere in a socket is the single strongest "doll" signal a face can emit, and it is precisely what face_front.png shows. The lid margin creates a hard silhouette edge against the eyeball; the lid crease creates the shadow that gives the eye depth; the orbital rim gives you eye_depth and hooding sliders. Nothing else you can do to the head buys this much.

2. LIP LOOPS WITH VERMILION — 3 concentric loops (nasolabial ring, vermilion border, lip body) + seam. Cost 66 quads. VERDICT: MANDATORY. The vermilion border is a hard C1 discontinuity in real anatomy; the current 3-horizontal-ring "sandwich" tries to fake it with a 0.02 Hh ripple between rings spaced 0.05 Hh apart and produces literally nothing (verified in face_front.png). Concentric loops also give the mouth CORNER real geometry, which is where all mouth-shape identity lives.

3. AMBIENT OCCLUSION IN THE SHADER — zero cage cost. VERDICT: MANDATORY, AND IT IS THE MULTIPLIER ON EVERYTHING ABOVE. Every feature in this list is a crease, and a crease is only visible because it is occluded. Without AO you build the geometry and still cannot see it. Do not build items 1-2 without item 3.

4. NOSTRILS / ALAE — 2 nostril islands (8 quads each) + the alar crease loop r10. Cost ~30 quads. VERDICT: MANDATORY. A nostril is a black hole; black holes read at any resolution, including at 28mm. This is the cheapest strong feature on the list.

5. BROW RIDGE — ring pair r14/r15 with the supraorbital torus as a 2x3 block per side rather than 3 lonely vertices. Cost: already in the ring stack, ~0 extra. VERDICT: MANDATORY. The brow is the shadow-caster that makes eyes read as recessed. It is also the primary male/female and primary "human vs. not" cue.

6. THUMB AS A REAL TUBE — 19 quads/hand. VERDICT: MANDATORY. Highest silhouette-per-quad on the body. A mitten with a thumb reads as a hand; a mitten without one reads as a mitten. Prints fine at 30mm (~0.8mm diameter).

7. JOINT LOOP PAIRS at elbow/knee/shoulder/hip — 12 verts each. VERDICT: MANDATORY, and it is a RIGGING fix disguised as a topology fix. Without it every posed render collapses at the joints, which no amount of shading hides.

8. NASOLABIAL FOLD — the L1 loop of the mouth island already provides it. VERDICT: WORTH IT, FREE. But keep its default depth near zero and drive it from an `age` slider; a nasolabial on a 20-year-old is the classic tell of a bad face model.

9. EAR STRUCTURE as separate shells (helix, antihelix, concha, tragus, lobe) — 60 quads each, off-shell. VERDICT: WORTH IT. Ears are low-attention in front views but they define the silhouette in 3/4 and profile, which is how miniatures are actually looked at. Separate shells make ear_protrusion/rotation free. Do NOT grid-insert them.

10. CLAVICLE + STERNUM + SPINE GROOVE as column-pair valleys — ~0 extra quads, comes with the r9 ring and 24 columns. VERDICT: WORTH IT, ESSENTIALLY FREE. The clavicle is the most legible bone landmark on a clothed-or-nude torso and the current cage has no loop anywhere near it.

11. LABIOMENTAL CREASE (r3) + COSTAL MARGIN (r5) + ILIAC CREST (r2) — 3 rings. VERDICT: WORTH IT. These are the "read as a body not a mannequin" landmarks, and they cost one ring each.

12. MALLEOLI + ARCH — 4 vertex bumps and a lift. VERDICT: WORTH IT, TRIVIAL.

13. TRUE FINGER TUBES — 4 tubes/hand, ~80 quads/hand. VERDICT: CONDITIONAL, PUT IT BEHIND A FLAG. At 30mm total height a finger is ~0.4mm diameter: printable in resin, fragile, impossible in FDM, and it adds 8 bones to the rig. At 54mm+ or for renders it is required for realism. Ship `hand_detail = mitten | grooved | fingers`, default `grooved`, auto-downgrade below 40mm with a warning. This is the honest answer to "the user wants realism but also 28mm prints."

14. TRUE TOE TUBES — VERDICT: NOT WORTH IT. Ever. Grooved toe block only. Real sculptors do the same thing at this scale, feet are usually in boots, and 5 separate 0.3mm toes will snap off the support raft.

15. TEETH / TONGUE / ORAL CAVITY — VERDICT: NOT WORTH IT unless you add an open-mouth expression. The mouth island's seam cap already seals the shell.

16. EYEBROWS / EYELASHES as thin shells — VERDICT: DEFER to a polish phase. Eyebrows are a huge identity signal but as geometry they read as caterpillars; better as a vertex-colour darkening on the r15 band, which costs nothing.

17. HAIR — VERDICT: NOT ACHIEVABLE WELL. See the ceiling statement. A scalp-cap shell with a few volume slabs is the maximum; it will read as a helmet. Ship bald + hood/helmet accessories and be honest about it.

18. GENITALS / NIPPLES — VERDICT: nipples yes (2 vertex bumps on r6, free, and their absence is uncanny on a nude torso); genitals as an optional shell only if the product needs it.

## slider_upgrade

TARGET: ~88 sliders (48 face / 34 body / 6 global), against MakeHuman's 249 (146 head/face) and Fallout 4's ~50 face controls. But the count is NOT the point — the ARCHITECTURE is. Three architectural changes have to land before any new slider is worth adding.

=== ARCHITECTURAL CHANGE 1: MIN/MAX MORPH PAIRS, NOT SIGNED VECTORS ===
All 22 current face sliders are a single signed vector: v += s * delta. Look at what MakeHuman actually ships (mh_mods.json, scraped): every modifier carries explicit endpoint NAMES — decr/incr, concave/convex, in/out, down/up, compress/uncompress. Those are two independently sculpted targets, not one vector and its negation. This matters anatomically: a gaunt cheek is a hollow under the zygomatic arch with the arch itself becoming MORE prominent; a full cheek is a rounded mass that BURIES the arch. Those are not negatives of each other. Same for eyefold concave/convex, nose curve concave/convex, chin prominent/recessed.
IMPLEMENT: morph registry entry = (name, D_neg, D_pos), each a sparse (idx, delta) pair; v += max(-s,0)*D_neg + max(s,0)*D_pos. This one change roughly doubles the expressive range of the sliders you already have, for zero new topology.

=== ARCHITECTURAL CHANGE 2: MORPHS FROM FALLOFF FIELDS, NOT HAND-PICKED VERTEX LISTS ===
topology.py GROUPS is 14 hand-typed vertex lists with hand-typed weights. That does not scale to 88 sliders and it is why every slider has 1-6 verts of support (the thing that measurably fails). Replace with a generator:
    def falloff_morph(anchor_xyz, radius, direction, profile='smoothstep', mask=None):
        g = bfs_geodesic_distance(cage_adjacency, nearest_vertex(anchor_xyz))   # hops, not Euclidean
        w = profile(1 - g/radius); w[w<0]=0
        return sparse(idx=nonzero(w), delta=w[:,None]*direction)
Use GEODESIC distance over cage edges, not Euclidean: on a 618-vert cage a BFS is microseconds, and it stops a "nose" morph from leaking across the air gap onto the cheek or through the lip seam into the other lip. That leak is exactly why MakeHuman ships per-vertex target FILES. With this primitive a new slider is 6 numbers instead of a vertex list, and every morph automatically has multi-vertex support — which is the fix for the 45%-survival problem, enforced by construction.
AUTOMATED GATE: every morph must touch >= 40 cage verts. Assert it at import.

=== ARCHITECTURAL CHANGE 3: "ONE SLIDER AFFECTS SEVERAL MUSCLES" = COMPOSITES + CORRECTIVES ===
The user's exact ask ("depth of cheekbones and where they are, with it affecting facial muscles") is a request for two mechanisms MetaHuman's RigLogic and MakeHuman's macro-modifiers both implement:
  (a) COMPOSITE / macro layer. Today resolve() is ~16 hand-written couplings in Python. Replace with an explicit matrix:
        s_eff = clip(M @ s_user + b, lo, hi)
      M is a named, inspectable (n_drive x n_slider) coupling matrix stored as data. `cheek_depth` gets a column that also drives masseter_volume (+0.35), nasolabial_depth (+0.25), submalar_hollow (+0.40), lower_lid_bag (-0.20) — i.e. raising the cheekbone tents the whole midface, which is the anatomy the user is describing. Being a matrix means it is auditable, invertible-ish, and unit-testable; being Python if-statements means it is not.
      MakeHuman precedent found in targets.py: its macros blend across gender[2] x age[4] x race[3] x muscle[3] x weight[3] x height[3] x breastsize[3] x breastfirmness[3] x proportions[3] — a full combinatorial target space resolved by weighted sum. That is the reference architecture.
  (b) CORRECTIVE / COMBINATION morphs. v += sum_ij f(s_i, s_j) * C_ij, with f = max(s_i,0)*max(s_j,0) (a "both are high" gate) or a smoothstep. This is how you stop slider stacking from leaving the human manifold. You need maybe 20 of these, not MetaHuman's 800: (cheek_depth x weight) so a gaunt fat face is impossible; (jaw_width x muscle) for the masseter; (brow_ridge x gender); (mouth_width x lip_volume) so a wide full mouth does not tear the corners; (muscle x weight) so shredded-and-obese cannot coexist; (eye_size x eye_spacing) to keep the intercanthal distance sane; (nose_length x nose_tip_angle).
  (c) DEPENDENT ANCHORS. Some sliders must MOVE OTHER SLIDERS' anchors, not just add deltas: moving cheekbone_position_v must relocate the anchor that cheekbone_projection pushes along, or the two sliders fight. Implement by resolving anchor positions in a first pass, then evaluating morphs in a second — the same two-stage order the spec already uses for ring-params-then-atoms.

=== THE 38, TRIAGED ===
KEEP AS-IS (these work because they scale ring parameters, not individual verts): height, head_size, neck_len, arm_len, leg_len, torso_len, shoulders, hips, waist, hand_size, foot_size. 11 sliders.
KEEP BUT FIX: neck_thick (the head-side coupling specified in the spec was never implemented — see the bug in "what's wrong"); face_width, face_length (fine mechanically, but they currently scale the ellipse BEFORE offsets, which is right — preserve that ordering).
FIX (currently inert: <=4 verts of support, ~45% survival): nose_bridge_height (2 verts), nose_bridge_depth (2), nose_tip (1), chin_width (2), ear_size (4), brow_height/brow_depth (5+5). All become falloff morphs on the new cage. Nothing about their SEMANTICS is wrong; their support is.
SPLIT (one knob is doing several anatomical jobs):
  eye_size -> eye_width, palpebral_height (aperture), eyeball_size. [MakeHuman ships eye-scale + height1/2/3 separately]
  nose_length -> nose_length (subnasale drop) + nose_tip_angle (septum rotation). [MH: nose-septumangle, nose-point up/down]
  nose_width -> nasal_root_width + nose_bridge_width + nostril_width + alar_flare. [MH ships FIVE: width1/2/3, nostrils-width, point-width, flaring]
  cheek_depth -> cheekbone_projection + cheekbone_position_vert + cheekbone_position_horiz. THIS IS THE USER'S EXPLICIT REQUEST ("depth of cheekbones AND WHERE THEY ARE"). Currently one slider pushes 6 verts along a fixed malar direction; it must become projection along the surface normal plus a 2-axis relocation of the anchor.
  jaw_width -> jaw_width (bigonial) + gonial_angle (flare) + masseter_volume.
  lip_full -> upper_lip_volume + lower_lip_volume + philtrum_depth + cupids_bow.
  mouth_width -> mouth_width + mouth_corner_angle (the smile/frown resting set). [MH: mouth-angles]
  brow_depth -> brow_ridge_projection + glabella_projection (separate bones, separate reads).
  weight -> weight + fat_distribution (android/gynoid).
  muscle -> muscle_mass + muscle_definition (bulk vs. cut — mass is radial, definition is crease depth; conflating them is why "muscle" currently just inflates).
MERGE/RETIRE: chin_length and chin_width stay but gain chin_projection; ear_size becomes an ear-shell transform group (4 sliders) now that the ear is a separate shell.

=== NEW SLIDERS, GROUPED, WITH THE ANATOMY THAT JUSTIFIES THEM ===
CRANIUM (6): skull_width, skull_length (dolichocephalic<->brachycephalic — the top-level "head shape" axis MakeHuman spends 7 shape targets on), occipital_projection, temple_width [MH forehead-temple], forehead_slope [MH forehead-nubian], forehead_height.
BROW/EYE (11): brow_ridge_projection, glabella_projection, brow_arch_height, brow_tail_angle, canthal_tilt (inner-vs-outer corner height — arguably the single most identity-defining eye parameter and completely absent today), upper_lid_crease_height, upper_lid_hood, epicanthic_fold [MH r-eye-epicanthus], lower_lid_bag [MH r-eye-bag + bag-height], sclera_show, interpupillary (rename of eye_spacing).
NOSE (9): nose_hump (dorsal convexity) [MH nose-hump, nose-curve, nose-greek], nasal_root_width, nose_bridge_width, nose_bridge_height, nose_bridge_depth, nose_length, nose_tip_angle, nostril_width, alar_flare, columella_show.
MOUTH (9): upper_lip_volume, lower_lip_volume, vermilion_height_upper, vermilion_height_lower, cupids_bow, philtrum_depth, mouth_width, mouth_corner_angle, mouth_protrusion (prognathism — a whole-midface axis nothing currently touches), nasolabial_depth.
CHEEK/JAW/CHIN (9): cheekbone_projection, cheekbone_position_vert, cheekbone_position_horiz, submalar_hollow, masseter_volume, jaw_width, gonial_angle, chin_projection, chin_length, chin_width, chin_cleft, labiomental_depth, submental_fullness (double chin) [MH neck-double].
EARS (4): ear_size, ear_protrusion (the read that actually matters and that ear_size cannot fake), ear_rotation, lobe_attachment [MH r-ear-lobe].
BODY (16 new): clavicle_prominence, trap_mass, deltoid_mass, pec_mass, lat_spread, ab_definition [MH stomach-tone], navel_depth, spine_groove, ribcage_width, ribcage_depth, waist_position, glute_mass, quad_sweep, calf_mass, forearm_mass, shoulder_slope, posture (lordosis<->kyphosis), hip_tilt.
GLOBAL (4): age (a MacroModifier in MakeHuman's sense — must drive ~15 couplings: skin sag, nasolabial, submental, brow descent, lip thinning, ear/nose growth), body_proportions (heroic 8-head <-> realistic 7.5-head <-> stylized), asymmetry (a low-amplitude randomized left/right bias; PERFECT SYMMETRY IS THE #2 UNCANNY TELL AFTER DEAD EYES and costs almost nothing — a single seeded scalar that scales a fixed random per-vertex field), randomize_seed.

=== AUTOMATED GATES ON THE SLIDER SET (build these, they are cheap and they will save you) ===
G1. Every morph touches >= 40 cage verts.
G2. At s = +/-1, every slider produces >= 3% change in either silhouette area or peak luminance in a standard 3-view render. A slider you cannot see is a lie in the UI.
G3. No two morph vectors may have cosine similarity > 0.90. This catches duplicate sliders objectively and is the test that would have caught nose_bridge_height vs nose_bridge_depth being near-identical today (both push (7,0) in -Y).
G4. Randomized slider draws (1000 samples) must all pass the watertight + no-self-intersection gate. This is what the corrective morphs are FOR.

## shading_upgrade

IS THIS THE HIGHEST REALISM-PER-EFFORT ITEM? Honest answer: IT IS THE HIGHEST RATIO, BUT IT IS NOT SUFFICIENT, AND IT MUST GO FIRST ANYWAY. Roughly 2-3 days of work buys maybe 40% of the perceived realism gap for renders, versus ~2 weeks for the topology. But an ACES-tone-mapped, subsurface-lit, ambient-occluded EGG is still an egg — face_front.png is an egg. The reason to do it FIRST is different and more important: YOU CANNOT ART-DIRECT A FACE YOU CANNOT SEE. Every crease you are about to build in Phase 3 is invisible under a 0.42-ambient N.L shader, so without the shading pass you will be authoring topology blind and tuning by guesswork, exactly as the current NEUTRAL_OFFSETS table was ("authored ~2x their target read" is a comment written by someone who could not see what they were doing). It is also the only part of the plan with zero risk to the STL gate. So: highest ratio, mandatory first, insufficient alone.

=== STEP 0: RESTRUCTURE TO A DEFERRED G-BUFFER. Do this before any lighting math. ===
Today render_meshes computes shading per triangle-corner inside a Python loop over triangles, then interpolates a SCALAR. That forecloses every effect below and is why smooth shading costs 1.92s vs flat 0.88s at subdiv 2.
Rasterize into whole-image float buffers instead: N (h,w,3 world normal), P (h,w,3 world position), Z (h,w), ALBEDO (h,w,3), GLOSS (h,w), ID (h,w int), AO (h,w). Interpolate per-vertex attributes with the barycentrics you already compute. Then do ALL lighting as vectorized whole-image numpy. Two wins: shading cost becomes O(pixels) in C instead of O(triangles) in Python, and screen-space effects become possible.
Also replace _smooth_corner_normals entirely: accumulate area-weighted per-VERTEX normals with np.add.at over mesh.faces (fully vectorized, ~50x faster) and interpolate them per pixel = true Phong normals instead of Gouraud. On a subdivision surface the crease-angle splitting logic is not needed; the cage already encodes where creases are.
ADD PER-VERTEX COLOR. render_meshes currently takes one hex color per mesh, which is why the eyeball is a solid black ball with no sclera. This is a required change, not a nicety.

=== STEP 1: COLOR PIPELINE (biggest single-line win, ~20 lines) ===
Today: linear multiply on sRGB values, written straight to PNG. That transfer function IS a large part of "unfired clay."
  albedo_lin = albedo_srgb ** 2.2
  ... all lighting in linear ...
  x = color_lin * exposure
  aces = (x*(2.51*x + 0.03)) / (x*(2.43*x + 0.59) + 0.14)      # Narkowicz ACES fit
  out = clip(aces, 0, 1) ** (1/2.2)
Drop ambient from 0.42 to ~0.06 and replace the lost fill with the hemisphere/rim terms below. The 0.42 ambient is currently destroying 42% of every normal variation before it starts.

=== STEP 2: WRAPPED DIFFUSE WITH PER-CHANNEL SCATTER (the fake SSS) ===
Skin's tell is that the terminator goes RED, because red light scatters furthest under the surface. Costs three multiplies:
  w_rgb = (0.90, 0.40, 0.25)                                   # per-channel wrap; red wraps most
  NdL   = dot(N, L)
  diff_rgb = clip((NdL + w_rgb) / (1 + w_rgb), 0, 1)
  diffuse = albedo_lin * diff_rgb * light_color * light_intensity
Optionally raise to a power ~1.5 to tighten. This single expression is the highest-value shading line in the whole plan: it turns the plastic terminator into a fleshy one, and it is why current renders look like clay even where the form is right.

=== STEP 3: SPECULAR WITH FRESNEL ===
Skin has a thin oily dielectric layer. No specular at all is the second-largest tell.
  H = normalize(L + V)
  F = F0 + (1 - F0) * (1 - dot(V,H))**5                        # Schlick; F0 = 0.028 for skin
  D = (N.H)**n                                                 # Blinn-Phong, n ~ 40 broad / ~120 tight
  spec = F * D * gloss_mask * light_color
Two specular lobes look markedly better than one (a broad n=25 sheen + a tight n=150 highlight at ~0.3 weight) and cost one extra line.
GLOSS MASK WITHOUT TEXTURES: you know the cage's named vertex groups, so bake a per-cage-vertex gloss scalar and interpolate it — T-zone (forehead, nose dorsum, nose tip, cheekbone crest, chin ball, lower lip) high ~1.0; cheeks/temple mid ~0.5; eyelid, nasolabial, under-jaw low ~0.2; lips WET ~1.3. This is the texture-free substitute for a specular map and it works because the map is a function of anatomy, which you have.

=== STEP 4: AMBIENT OCCLUSION — DO BOTH, MULTIPLY THEM ===
This is the term that makes the new topology visible. Without it, Phase 3 is wasted.
(a) BAKED CURVATURE AO, computed once per mesh build, view-independent, essentially free:
      for each vertex i: c_i = sum_j w_ij * dot(normalize(v_j - v_i), n_i) / sum_j w_ij   over 1-ring neighbours j
      ao_curv = clip(0.5 - k * c_i, 0, 1)**g          # k ~ 1.6, g ~ 1.5
    Concave regions (neighbours sitting above the tangent plane) darken. It is one np.add.at scatter over the mesh edges. It darkens EXACTLY the lid crease, nasolabial fold, lip seam, alar crease, ear concha and inter-finger grooves you are about to build. Widen the kernel by running it on a 2-ring neighbourhood or by smoothing the result to get a softer, more SSS-like falloff.
(b) SCREEN-SPACE AO in the deferred pass, for occlusion curvature cannot see (chin-to-neck, arm-to-torso, nostril interior):
      for each of K~12 poisson offsets d: sample Z at (px+d); occluded += (Z_sample < Z_center - bias) weighted by 1/(1+dist)
      ao_ss = 1 - strength * occluded/K
    Whole-image numpy, ~12 shifted array compares. Blur the result with a 5x5 box (two separable passes).
    ao = ao_curv * ao_ss, applied to the diffuse and ambient terms only, NOT to specular.

=== STEP 5: THREE-POINT RIG + RIM ===
Key (warm, up-left, camera-relative as today), fill (cool, opposite, 0.25 intensity, no specular), rim/back (from behind, +30 elevation):
  rim = (1 - clip(dot(N,V),0,1))**3 * clip(dot(N, L_back), 0, 1) * rim_color
Plus a hemisphere ambient instead of a constant: amb = lerp(ground_color, sky_color, 0.5 + 0.5*N.z) * ao. The rim is what separates a head from a background without an outline, and it is the cheapest "photographed" cue there is.

=== STEP 6: EYES GET THEIR OWN MATERIAL. THIS IS NOT OPTIONAL. ===
Currently both eyeballs are one flat #2e2b28 sphere: two drilled holes. With per-vertex color:
  - sclera off-white (0.85, 0.82, 0.80) linear, NOT pure white; iris disc as a colored ring with a darker limbal ring at the outer 12%; pupil near-black.
  - cornea specular: F0 = 0.05, n = 400, i.e. a tiny hard CATCHLIGHT. In portraiture the catchlight is the single highest "alive" signal per pixel that exists. If you implement one thing from this section, implement the catchlight.
  - FAKE LID SHADOW: our renderer has no shadow casting, so the upper eyeball would be unnaturally bright. Multiply the eyeball's shade by smoothstep on its own local +z: shade *= lerp(0.45, 1.0, smoothstep(0.15, 0.75, local_z_normalized_inverted)). Every real eye is darker at the top. This one hack does more for eye realism than the eyelid geometry does at small render sizes.

=== STEP 7: TEXTURE-FREE ALBEDO VARIATION (vertex colors from the named groups) ===
Roughly 8 rules, all free, all read as skin: lips +0.10 R / -0.04 G / -0.04 B; cheeks and nose tip +0.04 R saturation; ears +0.06 R at the rim (they are translucent and full of capillaries); eyelids slightly darker and redder; brow band r15 darkened (eyebrow, no geometry needed); chin/jaw/upper-lip band cooled and darkened by a `beard_shadow` slider gated on gender; under-chin and neck slightly darker; forehead very slightly lighter. Also add a low-amplitude 3D value noise over the whole skin (a hash of the vertex position, +/-2%) so the surface is not mathematically uniform — uniform albedo is a strong CG tell.

=== STEP 8: 2x SSAA ===
Render at 2x and box-downsample. Four lines. The hard aliased silhouette is one of the loudest "software renderer" signals and this removes it completely. Cost is 4x raster; with the deferred restructure making shading O(pixels) in numpy, you will still come out ahead of today's 1.92s.

=== STEP 9 (OPTIONAL, BEAUTY ONLY): ONE SHADOW MAP ===
Render depth from the key light with the same rasterizer, project and compare with a slope-scaled bias, 3x3 PCF. Doubles raster cost. Buys the under-brow, under-nose and under-chin cast shadows that define a face more than any other single lighting element. Gate it behind `quality='beauty'`; skip it for the interactive viewport.

ORDER OF IMPLEMENTATION BY VALUE/EFFORT: tone map + gamma + drop ambient (1 hr, huge) > curvature AO (2 hr, huge) > per-channel wrap diffuse (30 min, huge) > eye material + catchlight (2 hr, huge) > specular + Fresnel (2 hr, large) > deferred restructure (1 day, enabler) > SSAA (1 hr) > rim light (1 hr) > SSAO (3 hr) > vertex-color skin variation (3 hr) > shadow map (1 day, beauty only).

## honest_ceiling

WHAT PURE NUMPY WITH NO EXTERNAL ASSETS WILL NEVER GIVE YOU, ranked by how badly it hurts:

1. SKIN MICRODETAIL. Pores, fine wrinkles, crow's feet, lip texture, stubble. These are texture and displacement-map phenomena at ~0.1mm scale. Putting them in geometry would need ~10^6 cage quads; putting them in a texture needs authored maps, i.e. assets. This is the #1 remaining uncanny tell for any close-up render and there is NO workaround. Best available substitute: the 2% position-hash albedo noise, which reads as "slightly imperfect surface", not as skin.

2. HAIR. Not achievable to any standard a viewer would call realistic. Strand hair is out of reach; hair cards need authored assets; a shell-based scalp cap WILL read as a helmet no matter how you shade it. Ship bald + hoods/helmets and say so in the README. This is the largest single realism gap after item 1 and it is the one users notice first.

3. POPULATION-REALISTIC IDENTITY. MetaHuman and Eldritch Foundry look real partly because their bases are statistical fits to SCANNED heads - MetaHuman ships a DNA file per character encoding a scan-derived rig (see the OpenRigLogic/DNA docs in the scratchpad). Our cage is a hand-authored idealization, so no matter how many sliders you add the output will feel like "one face family with knobs on it." Without a scan-derived PCA basis - which is an asset - you cannot synthesize the non-linear shape covariance of a real population. Partial mitigation: encode published anthropometric correlations (ANSUR II means/SDs) into the coupling matrix M so slider combinations at least stay inside the human manifold. NOTE: the prior session's ansur_f.csv fetch returned a 14-byte "404: Not Found" - that data acquisition FAILED and must be redone, because the composite matrix depends on it.

4. REAL SUBSURFACE SCATTERING. The per-channel wrap is a good cheap fake and will fool most viewers at mini scale. What it cannot do: translucent ears in backlight, light bleeding through the nose wings, the correct diffusion falloff at silhouette edges. A separable screen-space diffusion blur (Jimenez) IS implementable in numpy as two depth-weighted 1D convolutions on the diffuse buffer, so this is partial-credit rather than a hard wall - but it needs correct world-space units and it will cost ~0.5s/frame.

5. EYES AT PORTRAIT SCALE. No corneal refraction, no iris parallax, no wet meniscus at the lid margin, no caustic on the iris floor. A flat colored iris + limbal ring + catchlight reads convincingly at 28mm and in a 500px render; it will not survive a 2000px crop.

6. EXPRESSIONS. You can hand-author maybe 10 expression morphs. MetaHuman's RigLogic evaluates hundreds of raw controls into thousands of corrective blendshape deltas per frame. Do not promise expressions.

7. PERFORMANCE. A pure-numpy rasterizer at ~55,000 triangles with deferred lighting, SSAO and 2x SSAA will land around 1-3 s/frame at 512x512. Fallout-4-style live slider feedback at beauty quality is NOT achievable. You need a two-tier viewport: cage-only or subdiv-1 flat for dragging, and a "render" button. Say this in the UI.

8. CLOTHING, ARMOR, CLOTH SIM, FITTING. Shells only, manually placed, no draping, no collision.

=== WHERE THE CROSSOVER IS - THE ACTUAL DECISION ===
IF THE DELIVERABLE STAYS 3D-PRINTED 28-32mm MINIATURES: pure numpy plus this plan is genuinely sufficient, and I would not switch. At 30mm the print is one color of resin - texture, SSS, skin albedo and hair color are ALL irrelevant to the artifact. The entire game is silhouette plus crease depth plus feature legibility at ~0.2mm, and Phases 3-4 deliver exactly that. Your renderer is then a preview tool, not the product, and Phase 1 is worth doing purely so you can see what you are sculpting.

IF THE DELIVERABLE IS RENDERS THAT A STRANGER WOULD CALL A PHOTOGRAPH OF A PERSON: SWITCH NOW, DO NOT FINISH THIS PLAN. You will spend 3 weeks to arrive at a very good stylized clay figure, and stylized clay is the ceiling. The crossover is not gradual - it is items 1, 2 and 3 above, all three of which are asset problems and none of which more numpy solves.

TRIGGER CONDITIONS - if ANY ONE of these becomes a requirement, the Godot/MakeHuman-asset route wins immediately: textures or normal maps; hair; expressions or animation; >5fps interactive with a real face; population-realistic identity variety; UV-mapped materials of any kind.

THE HYBRID THAT IS ACTUALLY BEST, AND WHICH YOU ARE ALREADY HALFWAY TO: keep this generator as the DRIVER and stop authoring the base geometry by hand. MakeHuman's base mesh plus its 249 modifiers plus its target library represents person-decades of professional sculpting; you will not out-sculpt it in numpy in any number of weeks. Use MakeHuman offline to emit a base mesh and a target set, bake them into a numpy-loadable .npz, and keep your slider/composite/corrective layer, your subdivision kernel, your STL gate and your renderer on top. The scratchpad ALREADY CONTAINS algos3d.py and targets.py - the exact target-blending and macro-resolution machinery - which strongly suggests a previous research pass reached the same conclusion. LICENSING WARNING BEFORE YOU DO THIS: MakeHuman's Python source is explicitly AGPL3 (stated in the file headers of algos3d.py, human.py and targets.py in the scratchpad), so PORTING their code makes MiniMaster AGPL3. The mesh and target ASSETS are separately licensed (CC0 for MH 1.x as I recall) - VERIFY THAT INDEPENDENTLY before shipping, do not take my word for it. Loading CC0 target data is fine; copying AGPL code is a licensing decision, not a technical one.

FINAL HONEST FRAMING: the gap between where you are and "realistic" is not one gap, it is two. Gap one is legibility - your features do not survive subdivision - and this plan closes it completely and provably (a 704-quad cage at subdiv 1, 5,632 tris, already beats your 176-quad cage at subdiv 3, 22,528 tris, using machine-picked features and zero artistry: see exp_dense_L1_front.png versus exp_orig_L3_front.png). Gap two is authored detail - texture, hair, scan-derived identity - and this plan does not close it and cannot. Close gap one. Then look at the result and decide honestly whether gap two matters for what you are actually shipping.

## Phased plan

1. PHASE 0 - INSTRUMENTATION AND TRUTH (0.5 day). Build the feature-legibility harness BEFORE changing anything: render a fixed head at 5 views x 3 lighting rigs and measure the local luminance step across each named feature (lip seam, upper-lid crease, alar crease, nasolabial, jawline, brow shadow). Also freeze the Catmull-Clark attenuation measurement as a unit test (1 vert -> 45%, 3-vert row -> 67%, 2x3 block -> 96%) so nobody ever authors a single-vertex feature again. Add a chi == V - F == 2 assertion helper. GATE: harness runs and correctly scores the CURRENT head 0 of 6 features legible (it will - see face_front.png). If the harness says the current head is fine, the harness is wrong; fix it before proceeding.
2. PHASE 1 - RENDERER: DEFERRED G-BUFFER AND SKIN SHADING (2-3 days). Rasterize to N/P/Z/albedo/gloss/id buffers; vectorized per-vertex normals via np.add.at replacing the O(F) Python loop in _smooth_corner_normals; per-vertex color support (required, the eyeball has no sclera today); sRGB->linear in, ACES tonemap + gamma out, ambient 0.42 -> 0.06; per-channel wrap diffuse w_rgb=(0.90,0.40,0.25); Blinn-Phong + Schlick Fresnel with an anatomy-derived gloss mask; baked curvature AO x 12-tap SSAO; three-point rig with Fresnel rim; eye material with sclera, limbal ring, catchlight and the fake lid-shadow gradient; 2x SSAA. DO THIS FIRST EVEN THOUGH THE TOPOLOGY IS THE REAL PROBLEM - you cannot art-direct geometry you cannot see. GATE: the UNCHANGED 178-vert head must now show a visible nose shadow, a visible brow shadow and a catchlight in each eye; smooth-shaded 512x512 at subdiv 2 must be <= 1.5x the current 1.92s; ship a clay-vs-skin A/B pair into docs/gallery. Bank the win before touching topology.
3. PHASE 2 - CAGE AUTHORING INFRASTRUCTURE (2 days). This is the phase people skip and then regret. Generalize loft_rings into a CageBuilder with: add_ring_tube(rings, cols), cut_patch(r0,r1,c0,c1) -> rim loop + orphan list, bridge(rim, loop), insert_concentric_island(rim, n_loops, cap_kind), automatic orphan compaction, and a chi assertion after every operation. Separately build the morph primitive: BFS geodesic distance over cage adjacency + falloff_morph(anchor, radius, direction, profile, mask) returning a sparse delta. GATE: rebuild the CURRENT 403/401 body and 178/176 head through the new builder and get byte-identical vertex arrays out; a synthetic test (cut 3x2, insert 4 loops + pole) must produce chi == 2; a geodesic falloff seeded at the nose tip must NOT leak onto the cheek across the air gap.
4. PHASE 3 - HEAD CAGE v2 (3-4 days). 24 columns x 19 rings base skull (456v) + 2 eye islands (+39/+39 each) + 1 mouth island (+66/+66) + 2 nostril islands (+8/+8 each) = 618 verts / 616 quads. 2 separate ear shells at 62v/60q. Re-author every neutral offset as a >=2x2 support block. Switch the default ship level from subdiv 2 to subdiv 1 (2,464 quads / 4,928 tris - cheaper than today and vastly better). Fix the neck_thick head-coupling bug while you are in there. GATE: chi == 2 and watertight; 6 of 6 features pass the Phase 0 legibility gate at SUBDIV 1; the profile silhouette must show the full sequence brow -> nasion depression -> dorsum -> subnasale -> upper lip -> lip step -> lower lip -> labiomental crease -> chin; a side-by-side against face_side.png must be unambiguous to a non-technical viewer.
5. PHASE 4 - BODY CAGE v2 AND RIG (3-4 days). 24-column phase-offset torso (NO midline - grooves come from column pairs, the 1-quad crotch survives) x 14 rings, 12-vertex limb rings, JOINT LOOP PAIRS at elbow/knee/shoulder/hip, clavicle/costal-margin/iliac-crest rings, grooved hands with a real thumb tube, grooved toe block, malleoli, arch. 973 verts / 971 quads. Rig 16 -> ~24 bones (clavicles, split spine, forearm twist, ball-of-foot) with per-ring weight ramps 1.0/0.75/0.25/0.0 replacing hard 0.5 pinch-ring weights. GATE: chi == 2; bend elbow to 120 deg and knee to 110 deg and verify the joint cross-sectional area stays >= 70% of rest (this is the objective test that the loop pairs worked); no self-intersection at the armpit or crotch in any of the 4 shipped poses; rotate the wrist 90 deg and verify the forearm does not candy-wrap.
6. PHASE 5 - SLIDER SYSTEM v2 (3-4 days). Min/max morph PAIRS replacing signed vectors. All morphs generated by falloff_morph from anatomical anchors, none hand-typed. resolve() becomes an explicit, data-driven coupling matrix s_eff = clip(M @ s_user + b) instead of ~16 Python if-statements. ~20 corrective/combination morphs gated on drive products. Two-stage evaluation so position sliders relocate other sliders' anchors. Ship ~88 sliders (48 face / 34 body / 6 global) including the user's explicit asks: nose_bridge_height, nose_bridge_depth, nose_hump, nose_length, cheekbone_projection, cheekbone_position_vert, cheekbone_position_horiz, and the masseter/nasolabial/submalar couplings that make cheekbone_depth 'affect facial muscles'. GATE: G1 every morph touches >= 40 cage verts; G2 every slider at +/-1 changes silhouette area or peak luminance by >= 3% in a 3-view render; G3 no two morph vectors have cosine similarity > 0.90; G4 1000 randomized slider draws all pass watertight + no-self-intersection. G3 is the one that will expose your duplicate sliders honestly.
7. PHASE 6 - PRINT REALITY PASS (1-2 days). Minimum-feature-size analysis at the target height: at 30mm the head is 4.14mm tall, an eye aperture is ~0.58mm wide, a palpebral fissure ~0.2mm, a finger ~0.42mm diameter. Resin LCD (35-50um XY) resolves ~0.15-0.2mm reliably; FDM with a 0.4mm nozzle resolves nothing below ~0.8mm. Implement: a local-thickness scan that warns below 0.25mm (resin) / 0.8mm (FDM); auto-suppression of sub-resolution morphs and auto-downgrade of hand_detail below 40mm; a 'print scale' preview mode that shows the figure at true size. GATE: STL at 30mm is watertight with a clean min-wall report; STL at 54mm and 75mm show the face features; the same params rendered vs printed do not disagree about what is visible.
8. PHASE 7 - POLISH, THEN STOP (ongoing). Vertex-color skin variation (lips, ears, eyelids, brow band, beard shadow, 2% position-hash noise). Asymmetry slider (a seeded low-amplitude field - perfect symmetry is the #2 uncanny tell after dead eyes and costs nothing to fix). Optional shadow map for beauty renders. Scalp-cap hair shell with the explicit caveat below. ~10 hand-authored expression morphs if wanted. GATE: none - this is where you stop and decide whether the ceiling statement means you should have switched engines.

## Key sources

- MEASUREMENT (mine, this session): Catmull-Clark feature attenuation on the live head cage. 0.100 Hh cage displacement -> limit surface peak: 1 vertex = 0.0451 (45%), 3-vertex row = 0.0672 (67%), 2x3 block = 0.0961 (96%) at subdiv 3 (56%/75%/100% at subdiv 1). This is the numeric basis for the '>= 2x2 support block' rule.
- MEASUREMENT (mine, this session): decisive experiment. 706v/704q cage (one CC refinement of the existing head) + 4 machine-selected feature bands, subdivided ONCE = 5,632 tris and shows a mouth, lid creases and alar definition. Current 176-quad cage at subdiv 3 = 22,528 tris and shows an egg. Evidence: /tmp/claude-0/-home-user-MiniMaster/d099e56a-afda-5e64-b63b-1d4150e4051f/scratchpad/exp_dense_L1_front.png, exp_dense_L1_3q.png, exp_orig_L3_front.png, exp_orig_L3_3q.png
- MEASUREMENT (mine, this session): current baseline renders proving the head reads as an egg with two dots - /tmp/claude-0/-home-user-MiniMaster/d099e56a-afda-5e64-b63b-1d4150e4051f/scratchpad/face_front.png, face_3q.png, face_side.png (subdiv 3, smooth shaded)
- MEASUREMENT (mine, this session): performance baseline. Build 9ms/31ms/124ms at subdiv 1/2/3; verts,tris 2,396/4,776 | 9,320/18,624 | 37,016/74,016; render 512x512 flat 0.23s/0.88s and smooth 0.51s/1.92s at subdiv 1/2.
- /home/user/MiniMaster/minimaster/character/topology.py - the 403/401 body cage, the 178/176 head cage, the 14 hand-typed GROUPS vertex lists, and the NEUTRAL_OFFSETS table where every facial feature has 1-4 vertices of support
- /home/user/MiniMaster/minimaster/character/generator.py - the 38 PARAMS registry, resolve() with its ~16 hand-written Python composite couplings, head_verts_local() (which is MISSING the neck_thick coupling that docs/character_lab_spec.md specifies), and eyeball_meshes()
- /home/user/MiniMaster/minimaster/render.py - shade = 0.42 + 0.58*max(N.L,0), no specular, no AO, no gamma/tonemap, one hex color per mesh (hence the sclera-less black eyeball), and the O(F) Python _smooth_corner_normals loop
- /home/user/MiniMaster/minimaster/core/subdiv.py - the Catmull-Clark kernel; note _cc_once raises on any edge not shared by exactly 2 quads, which is why eye and mouth apertures must be SEALED funnels rather than open holes
- /home/user/MiniMaster/docs/character_lab_spec.md - the original merged design spec (D1-D4 rationale, crotch-quad edge cancellation proof, armhole rim derivation, the 'authored ~2x their target read' admission, and the unimplemented head r0 neck_thick coupling)
- /home/user/MiniMaster/docs/anatomy.md - the Loomis/Richer/Bridgman proportion canon already in the repo, with its own honest note that web sourcing was blocked when it was written
- PRIOR-SESSION RESEARCH ARTIFACT: /tmp/claude-0/.../scratchpad/mh_mods.json - scraped MakeHuman modifier index. 249 modifiers across 22 groups; 146 are head/face (eyes 34, mouth 22, ears 22, nose 21, head 17, cheek 8, neck 8, chin 7, forehead 4, eyebrows 3); armslegs 60, torso 9, hip 7. Every modifier carries explicit min/max endpoint NAMES (decr/incr, concave/convex, in/out) - the evidence for the min/max morph-pair architecture.
- PRIOR-SESSION RESEARCH ARTIFACT: /tmp/claude-0/.../scratchpad/targets.py (MakeHuman) - the macro-modifier combinatorial target space: gender[2] x age[4] x race[3] x muscle[3] x weight[3] x height[3] x breastsize[3] x breastfirmness[3] x bodyproportions[3], resolved by weighted sum. The reference architecture for our composite matrix. AGPL3 per its file header.
- PRIOR-SESSION RESEARCH ARTIFACT: /tmp/claude-0/.../scratchpad/algos3d.py, human.py, hm.py (MakeHuman) - target loading and blending machinery; all AGPL3 per file headers ('basemesh hm08' referenced at algos3d.py:459)
- PRIOR-SESSION RESEARCH ARTIFACT: /tmp/claude-0/.../scratchpad/dna.md and orl.md (Epic OpenRigLogic / MetaHuman DNA docs) - the Descriptor/Definition/Behavior/Geometry layer model, and confirmation that MetaHuman rigs are scan-derived per-character DNA with corrective-expression evaluation. Basis for the corrective-morph recommendation and for the identity-ceiling argument.
- FAILED SOURCE, MUST BE REDONE: /tmp/claude-0/.../scratchpad/ansur_f.csv is 14 bytes containing '404: Not Found'. The ANSUR II anthropometric data was never actually retrieved. The Phase 5 composite matrix depends on it.
- INPUT GAP: the eight research reports referenced in the brief were NOT present in my prompt - no JSON payload arrived. This synthesis is built from the repo, the prior-session research artifacts listed above, and my own measurements. Web search budget for this session was already exhausted (200/200), so I could not independently re-source external topology references.

# CRITIC


## Missing

- PROPORTIONS: the new 19-ring z-table reproduces the exact error it was meant to fix. Normalizing the spec's own numbers (chin base r2=0.12, crown pole 1.03): eye line r13 lands at 0.451 of head height (canon 0.500), mouth seam r6 at 0.137 (canon 0.198), subnasale r10 at 0.275 (canon 0.300). The current cage measures 0.457/0.152/0.283 - i.e. the rebuild carries the defect forward. This is a free data fix and the loudest single non-realism cue; it is nowhere in the plan.
- FIGURE PROPORTION: HEAD_UNIT=0.138 in generator.py:130 gives a 6.77-head figure (chibi). 7.5 heads needs 0.1224. Hands measure 0.077 H against a 0.108 H anthropometric target (~30% small), feet 0.175 H against 0.152 H, and foot(0.164) != forearm(0.120) violates the canon. The plan adds a thumb tube to an undersized mitten and never fixes scale. `proportion_heads` should be a slider, not a constant.
- SEMI-SHARP CREASES in core/subdiv.py. `_cc_once` is uniform smooth Catmull-Clark with no per-edge sharpness. DeRose et al. sigma creases are ~40 lines (sharp edge point = plain midpoint; 1/8-6/8-1/8 crease vertex rule; decrement sigma per level) and are exactly the mechanism for lid margin, vermilion border, nostril rim, alar crease and ear helix. Two independent research reports recommend it; the plan spends 5 extra rings and 5 islands trying to buy the same thing with topology alone.
- JOINT POSITIONS AS A FUNCTION OF SHAPE. MakeHuman defines every joint as the centroid of an 8-vertex cube embedded in the mesh (shared/skeleton.py: verts.mean(axis=0)); SMPL writes J(beta) explicitly. The plan grows the rig 16->24 bones but leaves joint_layout(d) as hand-written arithmetic. With ~88 sliders that becomes unmaintainable and joints will drift. ~18 joint cubes, ~144 cage verts, ~60 lines.
- POSE-SPACE CORRECTIVES (Daz JCM / SMPL Bp(theta)). Joint loop pairs give the weight ramp somewhere to live; they do not restore volume. LBS at 120 degrees collapses regardless. 6 channels keyed on rotation-matrix entries (not Euler), applied to the rest cage before skinning, is the researched fix and shares its machinery with the slider correctives.
- ARCHETYPE / PCA IDENTITY BASIS. The plan's own ceiling statement concedes the output will be 'one face family with knobs on it' and then does nothing about it. 8-12 same-topology archetype neutral cages blended barycentrically (MetaHuman's 3-slot influence triangle) with per-region masks costs no external assets, guarantees plausibility by convex combination, and removes the need for the G4 clamping heuristics. This is the largest strategic omission.
- PRINT-SCALE EXAGGERATION. At 32 mm the head is 4.42 mm, so every current facial slider's full throw is under the 0.2 mm SLA design floor (eye_spacing 0.177 mm, eye_height 0.133, eye_depth 0.124, ear bump 0.053). Phase 6's answer is 'auto-suppression of sub-resolution morphs' - which deletes the product at its stated primary size. The sculptor's answer is heroic exaggeration: head x1.3, feature relief x2-3 at 32 mm ramping to 1.0 by 75 mm, applied on the export path only.
- RAY-TRACED AO OVER THE MERGED FIGURE. core/raycast.py already has vectorized Moller-Trumbore. Per-mesh curvature AO is blind to head->neck, lid->eyeball, chin->throat, arm->torso and crotch occlusion - precisely the contact shadows that make the new geometry read - and a 12-tap SSAO barely sees them. A batched any-hit variant gives AO and the translucency thickness bake from one function, at cage resolution (581 verts x 64 rays).
- BODY ANTHROPOMETRY DATA FIXES, all cheap, none in the plan: female shoulder-ring/hip-ring measures 0.86 (no dataset puts it below 1.0; target 1.20 female / 1.40 male); gender couples shoulders at +/-14% where real skeletal biacromial dimorphism is ~2% (move it to deltoid/trap mass); the widest hip ring sits at iliac-crest height when the widest point is the greater trochanter, level with the crotch; limbs are straight columns with no elbow carrying angle (8 vs 15 deg) or knee Q-angle (13 vs 16 deg) - a strong mannequin tell; `weight` inflates uniformly instead of routing through an android/gynoid deposition mask; bust is one ring at fixed columns instead of upper-pole/apex/inframammary.
- EAR PLACEMENT RULE and EYEBALL SIZE. Ears must span brow line to nose base (~0.231 Hh) tilted 15-20 deg with the top rearward; the spec makes them separate shells rooted at c8 with no vertical span or tilt rule. Eyeball radius 0.056*Hh works out to ~28 mm diameter; anatomical is ~24 mm (0.048).
- EXPORT SCALE CONVENTION. export.py:19 SIZE_PRESETS scale TOTAL height, but tabletop '28mm/32mm scale' is conventionally sole-to-eye. Every figure ships ~7% undersized next to Hero Forge / Eldritch Foundry output. One-line fix, and it shifts every mm threshold in Phase 6.
- SLIDERS AS DATA. falloff_morph generates morphs procedurally but never serializes them. MakeHuman's sparse `vertIdx dx dy dz` format (uint16 + int16*1e-3 compiled, 8 bytes/vert) makes every future slider a data file rather than code, and their whole 243-target facial set compiles to ~0.56 MB. Also unstated: mirror-lock policy (MakeHuman ships eyes/ears/cheek per-side), and measurement rulers (vertex polylines -> waist in cm), which a print tool wants anyway.
- HAIR. 'Not achievable well, ship bald' is the wrong call for the stated deliverable. At 28-32 mm hair is a MASS, not strands; chunky ribbons/lobes >=0.4 mm thick with >=0.3 mm grooves is what every commercial mini does. It is also the feature users notice first. A bald crown additionally exposes the valence-12 pole the spec justifies by saying it hides under hair.

## Wrong or infeasible

- 'Extra subdiv levels are not a detail knob; they are a blur knob' / 'each additional subdivision level actively DESTROYS information' is FALSE. Catmull-Clark converges to a fixed limit surface. The 45% at L3 IS the limit value; the 56% at L1 is simply non-convergence, not higher fidelity. L1 is a coarser approximation that happens to sit further from the limit. EVIDENCE 3's conclusion (denser cage wins) is correct; its stated cause is not - and the plan uses that wrong cause to justify shipping at subdiv 1, where a 616-quad head is 4,928 tris and visibly faceted (facet edges ~0.1 mm at 32 mm print, at the resin pixel).
- EYEBALL PROTRUSION BREAKS THE PLAN'S OWN GATE. 'The eyeball sphere protrudes through L3/L4' means the eyeball shell and the head shell intersect. MiniMaster has no boolean union; character_mesh() gates each shell with integrity_report() then Mesh.merge()s them, so intersecting shells pass manifoldness but the exported solid is self-intersecting by construction. Gate G4 ('1000 randomized draws pass watertight + no-self-intersection') is therefore unsatisfiable as written. Either accept intersecting shells and say so, or budget for a union - which is a large new subsystem, not a line item.
- CROWN POLE GETS WORSE, NOT BETTER. Measured on the live cage: the head has exactly two valence-8 poles (crown and throat); histogram is {3:16, 4:160, 8:2}. Going to 24 columns makes the crown fan 12 quads = a VALENCE-12 pole, on the visible top of a bald skull. The spec dismisses it as 'under hair / at the top where nothing is looked at' while item 17 ships bald. Industry rule is never above valence 5. Cap the crown with a grid patch (or column-reduce toward the pole) instead of a fan.
- CREASE LOOPS WITHOUT SEMI-SHARP CREASES PRODUCE NARROW SWELLS, NOT CREASES. The plan's own attenuation data (a lone displaced ring reads at 45-67%) applies equally to a 0.02 Hh ring pair - it is a tighter rounded transition, not a C1 discontinuity. The claim that r3/r8/r10/r14 'exist only to be creases' overstates what uniform Catmull-Clark can deliver. Pair them with sigma creases or expect the vermilion border to fail the Phase 3 gate.
- PHASE 1 GATE IS UNREACHABLE AS SCOPED. 'The UNCHANGED 178-vert head must now show a visible nose shadow' - a convex nose is not darkened by curvature AO (it is convex) and barely by a 12-tap SSAO; cast shadow under the nose needs the shadow map, which Phase 1 explicitly defers to Step 9 / beauty only.
- PHASE 1 PERFORMANCE GATE IS ARITHMETICALLY IMPOSSIBLE. It requires <=1.5x of the measured 1.92s while adding 2x SSAA. The deferred restructure moves SHADING out of Python but the raster loop in render.py stays a Python `for i in np.nonzero(valid)` over triangles; SSAA multiplies exactly that loop by 4, and the new cage at subdiv 2 is ~3x the triangles. That is ~12x the raster work against a 1.5x budget.
- ARCHITECTURE CHANGES 1 AND 2 CONTRADICT EACH OTHER. falloff_morph(anchor, radius, direction, ...) returns a single signed displacement field by construction, so v += s*D. The min/max pair architecture requires two independently authored endpoints (gaunt cheek = hollow WITH a more prominent arch; full cheek = mass that BURIES the arch - explicitly not negatives). The plan never says where D_neg and D_pos come from once all morphs are machine-generated. As written, generated morphs are exactly the signed vectors change 1 says to abolish.
- G3 (no two morph vectors with cosine similarity > 0.90) WILL REJECT LEGITIMATE SLIDERS. Once every morph is a smooth geodesic falloff from an anchor, any two sliders sharing an anchor - cheekbone_projection vs cheekbone_position_horiz, nose_bridge_height vs nose_bridge_depth, upper_lip_volume vs vermilion_height_upper - will exceed 0.90 by construction of the falloff, not because they are duplicates. The gate needs to compare direction-normalized fields or be scoped per-region.
- THE ANSUR BLOCKER IS FALSE. The plan states the Phase 5 coupling matrix 'depends on' re-fetching ANSUR II. The research already supplies measured targets sufficient to author M: biacromial/stature 23.01 vs 22.57, shoulder:hip 1.40/1.20, waist:hip 0.86/0.74, brachial index 0.74-0.80, crural 0.80-0.86, carrying angle 10.97+/-4.27 vs 15.07+/-4.95, Q-angle 12-13.5 vs 15.9-17. Phase 5 is not blocked.
- MAKEHUMAN ASSET LICENSING IS ALREADY VERIFIED, NOT UNCERTAIN. LICENSE.md section C releases base mesh, proxies, all 1,280 targets, textures and poses as CC0 1.0; every .target file carries the inline header 'explicitly released as CC0 in september 2020'; default_weights.mhw contains "license": "CC0". Only the Python is AGPL3. The plan's hedge ('verify that independently, do not take my word for it') wrongly leaves the single largest unlock in the ceiling section flagged as risky.
- PHASE 2's 'byte-identical rebuild' GATE IS EXPENSIVE AND DISPOSABLE. The body cage is not lofted - it is hand-indexed (tid/lleg/larm) with compaction via np.unique(q), so BODY_KEEP ordering is an emergent property of face-emission order. Reproducing it byte-identically through a general CageBuilder is real work that Phase 4 immediately discards. Assert chi==2, watertightness and a vertex-set match under sorting instead.
- 616 HEAD QUADS IS BELOW EVERY RESEARCHED MINIMUM (600-900, 1,000-1,400, 1,100-1,400 quads; 1,200-2,000 verts). Specifically the eye island's 10-vertex rim gives 5 verts per lid - not enough to carry a lid curve, a canthal tilt, an inner canthus and a caruncle. Research puts aperture loops at 12-16 verts. Expect to need ~900-1,100 quads, which is still cheap; do not lock the column table at a number chosen to make the budget headline work.

## Over-promises

- '~90 ms at subdiv 1 - still interactive for the cage.' That is the subdivision cost alone. It excludes 88 sliders through a two-stage resolve, a coupling matrix, ~20 correctives, LBS, and (per Phase 1) an AO bake that must be redone on every rebuild.
- '6 of 6 features pass the legibility gate at SUBDIV 1.' At 4,928 tris a head shows facet edges of the same order as the crease depths being measured; the harness will partly be scoring tessellation.
- 'Bend elbow to 120 deg and verify joint cross-sectional area stays >= 70% of rest.' Joint loop pairs plus a 1.0/0.75/0.25/0.0 weight ramp do not conserve volume under linear blend skinning - that is what pose correctives are for. This gate will fail and the plan has no mechanism to make it pass.
- 'This plan closes gap one completely and provably.' It closes legibility for the features it builds. It does not touch the proportion errors (eye line 0.451 vs 0.500, 6.77 heads, hands 30% small) that are themselves a legibility failure and are cheaper to fix than any island.
- '3.5x the current 176 quads' is presented as sufficient in the same document that reports MakeHuman spending 741 control verts on the nose region and 768 on the cheeks. Both cannot be true; the budget headline is doing rhetorical work the numbers do not support.
- Phase durations. Phase 3 at 3-4 days covers a 24x19 base grid, 5 islands, 2 ear shells, a full re-authoring of every neutral offset as a >=2x2 block, and the neck_thick fix. Phase 5 at 3-4 days covers a morph-pair registry, geodesic falloff generation, a data-driven coupling matrix, ~20 correctives, two-stage anchor resolution, ~88 sliders and four automated gates. Both read ~2x optimistic.
- 'Ship at subdiv 1 for preview AND for most prints.' Cheaper than today, yes - but see the convergence point: L1 is under-converged, and its facet size at 32 mm is at the resin pixel.
- 'Not one pole is on a visible, curvature-critical surface.' Measured: the crown pole is on the visible top of the skull today at valence 8, and the plan raises it to valence 12.

## Highest leverage

"Build the ISLAND PRIMITIVE and use it to ship EYELIDS. Concretely: cut_patch(r0,r1,c0,c1) -> rim loop + orphan list, bridge(rim, loop), insert_concentric_island(rim, n_loops, cap_kind), automatic orphan compaction, and a chi == V - F == 2 assertion after every operation - then apply it to cut the eye region and insert orbital rim / lid crease / lid margin / funnel + pole, on a base skull widened enough at the eye columns to hold a >=12-vertex aperture.\n\nWhy this and not the shading pass or the mouth: (1) A bare sphere in a socket is the strongest doll signal a face emits and the plan itself ranks it #1. (2) It is the ONE feature whose payoff needs no renderer work to evaluate - a lid margin is a SILHOUETTE edge against the eyeball, so it reads under the existing flat/gouraud shader. That directly refutes the plan's 'you cannot art-direct a face you cannot see' sequencing argument for this specific item, and it means the visible win banks before any of the deferred-G-buffer risk. (3) It also reads at 28-32 mm print scale, where the whole shading stack is irrelevant to the artifact. (4) The machinery is the same machinery the mouth, nostrils, armholes and the thumb tube all need, and it is the phase the plan correctly says people skip and then regret - so building it here means Phases 3 and 4 become configuration rather than another hand-indexed topology.py.\n\nDo it with one addition the plan omits: land per-edge semi-sharp crease support in core/subdiv.py at the same time (~40 lines) and crease the lid margin at sigma ~2.0. Without it the lid margin is a rounded swell and the island's cost is only partly repaid."