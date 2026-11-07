# Appendage anatomy: consolidated takeaways from the reference library

Consolidated from the six region DIGESTs (tail, horn, wing, limb, skin, general), 357 manifest entries / 330 unique images. Ids are always `region/id` (horn and wing both use A01.., H01..). Companion files: `INDEX.md` (every image with licence and measurements), `sheets/SHEETS.md`, `appendage_references.md` (sources-only, for the repo).

## 0. How much to trust a number

- **[M id]** measured by a gatherer on that image from pixel landmarks (method and landmarks in the manifest `measurements` field and the `_annot.png` copy). Typical error 5-10 % on flat plates and scale-bar photos, 10-20 % on perspective photos and engravings, 25-30 % on stylised prints. Every number rests on **one image (n = 1)**. I opened six of the annotated copies (`limb/limb_071`, `skin/wm_giraffe_head`, `horn/A29`, `tail/cm_PSM_V08_D430_Skeleton_of_the_kangaroo`, `wing/H03`, the cat tail axis) and the landmarks sit where the text says; I did not re-measure.
- **[S id]** seen on the image, no number. **[T]** textbook recall by a gatherer, not verifiable from this library; treat as a hint. **[D]** derived by arithmetic from measured values. **[I]** inference (design reading of the pictures).
- **Not independent**: `wing/H03` and `limb/limb_042` are the same plate (Gray 1918 fig. 204), so the "agreement" between the wing and limb digests on scapula and interscapular ratios is one plate measured twice. The independent checks are `wing/H01` (Albinus) and `limb/limb_054` (Vesalius).
- Ratios are in the unit named in each line (base diameter, base radius r, body width). "r" is always the **base radius of the protrusion** unless stated.

## 1. The model every reference supports

An appendage is never a tube plugged into the skin. It is three nested layers: **a bone strut that is part of the skeleton** (sacrocaudal column, skull site, scapula/glenoid girdle), **a muscle or soft mound that is much larger than the shaft and spreads over the trunk**, and **one continuous thin skin sheet** over both. Most of the "fillet" people see around a limb or tail root is the muscle mound, not skin stretching (skin/wc_ford_shoulder_V0008626: skin and fascia flayed back as a thin sheet over the shoulder [S]; skin/wm_leonardo_shoulder_photo: arm diameter grows x1.31 over the last 0.57 root diameters [M]). The skin is C1-continuous (no seam) on the side where the contour runs along the muscle and has at most **one crease on the side where the part folds against the trunk** (axilla, tail underside, elbow-chest). What differs between appendages is the **seam type**: soft appendages stay continuous; hard ones (keratin, bone, ivory) meet skin along a hair line, a keratin rim or a collar of folds.

## 2. Skin transition: fillet width by junction type (in base radii r)

| junction | seam / fillet | number | id |
|---|---|---|---|
| soft muscle mass to neck/trunk (human deltoid-trapezius) | long smooth fillet, no crease on the upper side | dome height **1.23 arm radii** above the neck-base-to-arm chord; mean fillet radius **1.40 arm radii**; tangent turns 83 deg from trapezius slope (9 deg) to the near-vertical arm edge [M] | limb/limb_071 |
| arm underside to chest (axilla) | crease, small radius | rounding **0.11 r**, inside angle about 122 deg; torso side almost straight [M] | skin/wm_leonardo_shoulder_photo |
| quadruped chest to forelimb | smooth S ramp + one crease where the ramp ends | swell **1.14 forearm radii** beyond the forearm line, returning over **3.1 radii** at 21 deg from vertical [M] | limb/limb_012 |
| limb root flare | first quarter of the limb carries the mound | width x**1.26** (Rodin bronze, root/mid-arm) and x**1.31** (live shoulder) over the first 0.25-0.57 diameters; then near-cylinder [M] | limb/limb_059, skin/wm_leonardo_shoulder_photo |
| tail root, dorsal and lateral sides | no step: the rump IS the mound | upper-tail contour deviates only **3 %** from the rump line (kangaroo outline); horse tail top line within 2 px of the croup summit [M]; so fillet radius is the whole rump curvature, **>= 1 r** [I] | tail/cm_PSM_V08_D430_Skeleton_of_the_kangaroo, general/gen_025 |
| tail root, ventral side | crease where the tail underside meets thigh/buttock | not measured; seen in cat, iguana, monitor [S]; use about 0.1-0.3 r [I] | tail/cm_Superficial_muscles_of_a_cat, tail/ov_964703a3, tail/ov_a9455ce0 |
| ossicone (skin-covered bone) | hair continuous; concave valley between a pair; outer flank flows into head | valley fillet radius **0.25 x base width = 0.5 r** [M] | horn/A29 |
| deer pedicle (velvet dome) | no geometric fillet: a fur whorl at the base | dome height **0.88 r** (H/W 0.44), seam about 0 r [M] | skin/wm_antlers_season_start |
| bovid horn sheath on core | skin ends in a collar at the sheath rim | about **0** [S] | horn/A22, horn/A06 |
| keratin cap / casque | sharp seam, strong texture change | rim **0.11 r** (0.054 x base chord) [M] | skin/wm_cassowary_casque |
| rhino horn on rugose pad | skin folds radiate to a slightly raised cuff | not measured [S] | horn/A15 |
| tusk through lip or pad | collar of radiating wrinkles (elephant) or hairy pad (walrus) | no smooth fillet [S]; tusk visible length 10.7 x diameter [M] | skin/ov_elephant_tusk, skin/wm_walrus_noaa |
| fin or sail on a trunk | leading edge almost straight into the back; trailing edge concave flare | flare **0.6 r wide x 0.5-0.6 r tall**; fin H/base chord 1.59 [M] | skin/wm_orca_noaa |
| knife-edge bone under skin (withers, winged scapula) | skin rides on bone; very short fillet | skin 0.06 x thorax width above the spinous tip (**0.12 r**) [M]; bone ridge tents thin skin [S] | skin/url_cuyer_fig14_15, skin/wm_winged_scapula |
| subdermal dome / transdermal anchor (human) | smooth domes, gradual fall-off, hair continues; anchor leaves only a tiny tented collar | not measured [S] | skin/wm_implant_ball, skin/wm_dermal_anchor |

**Rule of thumb that fits all rows:** soft muscle-mass transitions blend over **>= 1 r**; fins about **0.6 r**; ossicones **0.5 r**; hard caps, creases and bone ridges **about 0.1 r**; bovid horn and antler seams **about 0** (the transition is a hair line or colour/texture change, not geometry).

**Animal vs human disagreement that matters for this project:** the animal "hair line" seam (deer pedicle, giraffe ossicone, horse ear root hidden in a ring of hair) has no equivalent on bald human skin. On a human body a horn root therefore has only the real human behaviours: **flush skin with a faint shadow** (horn/H15 ear horn, horn/H17 scalp horn), **mounded warty skin tenting round the base** (horn/H16, the only photograph of a soft mound), or **root hidden in hair** (horn/H19, H24, H25). Pick one deliberately; do not borrow the animal hair-line seam without hair.

## 3. Tail

**Where it emerges.** On the dorsal midline directly above/behind the pelvic ring at the end of the sacrum; the first free caudal sits above the ischial tuberosities [M tail/cm_Anatomical_technology_as_applied_to_the_domestic]; in straight rear views the root is centred between two symmetric haunches [S general/gen_038, general/gen_039, tail/ov_279c0519, tail/ov_c91d0138]. It never starts on the outer rump surface above the gluteal mass. In humans the (rare) real caudal appendage sat at the sacrococcygeal level, hanging in the top of the gluteal cleft [M tail/cm_A_child_with_a_tail_side_and_posterior_view_Repr, research-only licence]; satyr art roots a thin tail on the lumbosacral midline [S tail/met_912467_2, tail/met_255385_0].

**Emergence angle is a species parameter, not a constant.**

| species | first caudal segment vs the lumbar line | id |
|---|---|---|
| horse | 13 deg below, nearly straight out, then drooping [M] | general/gen_019 |
| cat | within about 5 deg for the first 19 % of tail length, then bends about 80 deg downward over the next 20 % [M] | tail/cm_Anatomical_technology_as_applied_to_the_domestic |
| dog | about 48 deg below horizontal [M] | general/gen_020 |
| lion | hangs down along the thigh to about ground level [S] | general/gen_028 |
| ring-tailed lemur, raised | sacrocaudal bend about 100 deg (first caudal 95 deg up, lumbar 14 deg down) [M] | tail/cm_Lemur_catta_skelet_1782_Print_Iconographia_Zoolo |
| kangaroo | thick continuation curving to the ground as a third support [S] | general/gen_043, tail/cm_PSM_V08_D430_Skeleton_of_the_kangaroo |
| human coccyx | turns ventrally about 57 deg from the dorsal sacral chord [M] | tail/cm_Gray97 |

**What to use for a human body:** because the human sacrum/coccyx points downward and the tail must leave on the midline at the sacrococcygeal point, start the tail axis within about 0-20 deg of the local sacral tangent (cat-like), keep the first fifth of its length nearly straight, then let it curve; do not leave at the quadruped's horizontal lumbar line.

**Length.** Caudal column as a fraction of the cervical+trunk column: horse 0.38 of atlas-to-sacrum, 0.62 of thoracolumbar (general/gen_019); dog 0.60 (general/gen_020); lion 0.98 (general/gen_028) [M]. Soft tail: leopard 0.45 of nose-to-root (general/gen_037) [M]. Tail/head-body: cat about 0.74 (rough), ring-tailed lemur 1.54, mongoose lemur 1.23 [M tail/...]. Lizards: 1.18 x snout-to-pelvis (water monitor, tail/cm_Varanus_salvator_5514) and 1.61 x snout-vent (Nile monitor, general/gen_040) [M]; the two differ in species and reference point, so use **1.2-1.6**. Vertebra counts: 12 horse, 16 ox, 14 pig, 15 dog (Chauveau table, general/gen_024); cat 19-21 countable on the plate [S]; kangaroo 20-26, lemur about 30, human coccyx 3-5 [T].

**Bone layer.** Bone taper (cat, width / first free caudal): 0.98 @25 %, 0.59 @50 %, 0.50 @75 %, 0.32 @95 %, 0.19 tip [M tail/cm_Anatomical_technology...]; macaque span falls to 0.55 of the first caudal within five vertebrae, first five caudals 8.4 cm, pelvis 11.06 cm [M tail/si_a151e192, scale bar]; horse base:tip depth 3.5:1 [M general/gen_019]. The proximal 5-6 caudals are short and blocky, the middle ones long and slender [S tail/cm_A_House_Cat_s_Bones]. Reptile caudals: tall neural spines and long transverse processes at the root, chevrons from about the 4th caudal [S tail/cm_The_Osteology_of_the_Reptiles_p108]. **The bone is a thin core**: the soft-tissue outline is about 3-4 x wider than the bone along the proximal third (kangaroo, [S] tail/cm_PSM_V08_D430_Skeleton_of_the_kangaroo), so do not use the bone taper as the skin taper.

**Soft outline, width / base width, two default profiles** (tails are bimodal, not contradictory):

| profile | 25 % | 50 % | 75 % | tip | source |
|---|---|---|---|---|---|
| **cone** (kangaroo, horse, lizards): default **1.00 / 0.76 / 0.52 / 0.38 / 0.10** (mean of five) | 0.79 kangaroo outline, 0.78 Stubbs horse, 0.87 iguana, 0.68 Nile monitor, 0.69 kangaroo litho | 0.47, 0.57, 0.61, 0.44, 0.52 | 0.34, 0.45, 0.40, 0.33, 0.40 | 0.11 @90 % kangaroo, 0.10 horse, 0.30 @95 % iguana | tail/cm_PSM_V08_D430_Skeleton_of_the_kangaroo, general/gen_025, tail/ov_964703a3, general/gen_040, tail/wc_csumm3s8 [M] |
| **fur cylinder** (cats, lemur, spider monkey, leopard): default **1.0 / 0.9 / 0.9 / 0.85, taper only in the last 20 % to 0.4-0.7 with a rounded tip** | 0.87 leopard | 0.92 leopard, about 1.0 lemur, 0.9 spider monkey | 0.87 leopard, 0.8 spider monkey | 0.79 leopard (to tuft), 0.55 spider monkey | general/gen_037, tail/cm_Ring_tailed_lemur_Lemur_catta, tail/cm_Spider_Monkey_Stretched_6251624427 [M, by eye +-15 %] |

**Base size.** Base about 0.37 x hip width (spider monkey, by eye +-15 %); about 0.24 x trunk depth at the haunch (kangaroo litho, rough); 0.61 x mid-trunk width (Nile monitor, general/gen_040); cat root about 0.4 x hip width (tail/cm_Cat_in_the_streets_of_Tangeri_110, by eye); human caudal appendage 0.18 x hip dimension with length 0.32 x hip depth (lateral) / 0.41 x hip width (posterior), length/base 1.8-2.3 [M +-20 %]; satyr statue tail diameter 0.06-0.12 x back width (thin, sculptural, tail/met_912467_2). **Use** about 0.25 x bitrochanteric width for a mammal-style tail on a human body (inside the 0.18-0.37 bracket), 0.12 for a slim sculptural tail. Human reference widths: bi-iliac crest 0.166, bitrochanteric 0.173 of stature (Albinus, general/gen_005, +-5 px).

**Muscle layer.** The tail muscles are slender strands along the vertebrae ending in long tendons (kangaroo model, tail/cm_Kangaroo_tail_anatomical_model_6727490457, general/gen_044); gluteus medius and maximus form the rump mass beside and in front of the root, and the cat plate labels "muscles concerned in the movements of the tail" at the root (tail/cm_Superficial_muscles_of_a_cat). [T] mammal: sacrocaudalis dorsalis/ventralis, intertransversarii, coccygeus/levator ani; reptile: caudofemoralis from the caudal transverse processes and chevrons to the femur, which is why lizard and crocodile tail bases are a deep fusiform mass continuous with the thigh (tail/ov_a9455ce0, tail/ov_964703a3 show the mass [S]).

**Human sacral surface (what the skin must keep).** Spinal furrow descending into a sacral triangle, two gluteal masses meeting at a midline cleft that starts near the sacral apex/coccyx and ends at the gluteal fold (tail/wc_dwgghrqs, tail/wc_bayw69tw, tail/ov_a77ce19c, tail/met_360773_0). Cleft length about 0.45 x hip width, cleft apex about 95 px below the lumbosacral landmark against hip width 180 px [M, +-15 %, tail/wc_dwgghrqs]. Dissections show thin skin and fat over a broad gluteus maximus sheet fanning from sacrum and iliac crest (tail/wc_j42auhe2, tail/wc_kpq8pswh). Do not fill the cleft or hide the mounds; a junction reproduces both. Sacrum+coccyx 0.082 of stature, coccyx tip at about trochanter level, trochanters at 0.51 of stature, shoulder:iliac width 1.24 (Albinus, [M] general/gen_005). Heavy bodies lose the waist and bone contours (general/gen_018).

**Tail checklist for a render:** (1) root on the midline in the cleft/sacral triangle, not on the outer glute surface; (2) first fifth of the tail nearly straight (0-20 deg off the sacral tangent) then curving; (3) base 0.18-0.37 x hip width; (4) taper follows the cone or cylinder profile above; (5) upper/lateral skin tangent-continuous with the rump (no collar), a crease only on the underside; (6) cleft and both gluteal mounds still visible; (7) if furred or scaled, the texture runs onto the tail with no seam.

## 4. Horn and cranial protrusions

**Five real mechanisms; a horn must use one** (all [S] unless numbered):

| mechanism | layers under the visible horn | skin at the root | ids |
|---|---|---|---|
| bovid horn | frontal-bone core continuous with the skull + larger hollow keratin sleeve | skin ends in a collar at the sheath rim, **no fillet** | horn/A01, A02, A05, A06, A07, A11, A12, A22 |
| cervid antler | skull, short skin-covered **pedicle**, rough **burr ring**, bare bone antler (velvet while growing) | hairy skin runs up the pedicle and stops at the burr | horn/A18, A19, A20, A21, skin/wm_antlers_season_start |
| giraffe ossicone | bone column fully sleeved in skin and hair | hair continuous over the horn, concave valley fillet | horn/A29, A30, A16, skin/wm_giraffe_head, skin/ov_giraffe_ossicones |
| rhino horn | rugose bone pad, **no core**, keratin cap normal to the pad | skin folds radiate to a raised cuff | horn/A13, A14, A15 |
| human (pathological/sculptural) | keratin cone from the epidermis, or carved cone merged into hair | flush, or mounded warty skin, or hidden in hair | horn/H15, H16, H17, H18, H22, H19, H24, H25 |

**Where it roots.** Bovid cores leave the top-rear corner of the skull (poll) in cattle/aurochs [S horn/A07, A08, A09, A10] and sit above and behind the orbit in goat and sheep [S horn/A05, A31]. Ossicones sit on the poll behind the ear line close to the midline, about 1/5 of head length tall [S horn/A30]. Antler pedicles sit directly above the eyes: inter-pedicle / inter-ocular **0.97**, eye line to burr height **0.71** x inter-ocular, beams diverging **23-26 deg each** off vertical (V about 49 deg) [M sika, horn/A20]. Rhino front horn on the nasal-tip pad, second on the frontal [S horn/A13]. **Human skull** [M horn/H01, length glabella to opisthocranion = 1.00]: calvarium height above the zygomatic arch 0.58, glabella-bregma chord 0.53, temporal-fossa span 0.62. Layer order skin, subcutaneous tissue, galea aponeurotica, loose connective tissue, periosteum, skull (schematic horn/H13, **order only**). A forehead or vertex horn sits on bone with only skin, galea and periosteum, so only a thin skin cuff is plausible; a temple horn sits on the temporalis fan (horn/H05, H10, H11, H20) and the skin must bulge over it (horn/H14 shows how far muscle bulk reshapes the temple).

**Dimensions.**

- **Bovid bone core** (goat, scale bar): linear cone, thickness 22/31/39/59 px at x = 140/180/220/300, half-angle 5.8 deg, length 13.8 cm, base 2.8 cm, d/L **0.20**, width **0.75 / 0.50 / 0.25** of base at 25/50/75 % length; base about 0.11 x skull length [M +-10 %, horn/A05].
- **Keratin sheath** is stouter and larger than the core: water buffalo base opening 10.6 cm, width 5.1 cm at 46 % (0.48 x base), centre line 34 cm, L/base **3.2** [M +-15 %, horn/A11]; cow without boss beside bull with fused boss on one skull plan (horn/A12). Rhino keratin: front horn L/base **2.8**, base/skull length 0.21, axis **94 deg** to its own (tilted) pad, so the backward lean is pad orientation not horn bending; frontal horn h/base 0.97, base/skull 0.27 [M +-10 %, horn/A13]. So a keratin horn is a stout cone about **3 : 1**, a bare bovid core a slender cone about 5 : 1.
- **Antler junction** (fallow shed antler, 10 cm scale): burr ring 5.5 cm, constricted neck 4.2 cm (burr/neck **1.3**), pedicle stump 5.5 cm, antler flares to 8.3 cm (1.5 x burr) only 2.2 cm above the burr; sika burr/beam 1.5 [M +-15 %, horn/A18, horn/A20]. Young pedicle dome H/W **0.44**, W/centre-spacing 0.67 [M skin/wm_antlers_season_start].
- **Ossicone**: width 1.00 / 0.84 / 0.78 / 0.73 of base at 0/25/50/75 % height, base W/H about 0.8 [M +-10 %, foreshortened close-up, horn/A29]; silhouette against sky: height about 145 px, base width 79-87 px (**H/W 1.75**), tuft width 0.63-0.80 x base, taper 4-8 deg per side, centre spacing 1.2-1.5 x local width, inner edges meet in a concave crotch [M skin/wm_giraffe_head]. Rounded dome with dark tuft, not a point. **Conflict and choice:** H/W 1.25 (horn/A29) vs 1.75 (skin/wm_giraffe_head). A29 is foreshortened and measured from a higher base line, the sky silhouette is not; use **H/W about 1.5-1.75** and take the taper profile and valley fillet from A29.
- **Human horns** [low confidence]: ear cutaneous horn about 4.2 x 2.8 cm using ear length 6.3 cm as an assumed scale (+-25 %, horn/H15); scalp horn base about 0.10 x cranial width, h/base 0.9 (horn/H17, visual); stylised spiral horn base about 0.30 x head width, coil height 0.32 x head height (+-30 %, horn/H22); Moses horns about 1/5 of head height (horn/H24). **Conflict:** base 0.10 (H17) vs 0.30 (H22) of head width; H22 is a stylised print. Use **0.10-0.15** for a realistic horn, larger only when the root is hidden by hair as in sculpture.

**Horn checklist:** (1) mechanism chosen and its layers present (core + larger sleeve; pedicle + burr ring; skin-covered column + dome; pad + cap); (2) root at a bone site (poll/above orbit/above the eyes), not on the temple unless the skin bulges over a muscle mound; (3) bovid: no fillet, skin ends at a collar; antler: ring about 1.3 x neck with texture change at the burr; ossicone: taper only to 0.73 of base, dome tip, concave valley about 0.5 r; (4) pair spacing and divergence match the species (pedicle V about 49 deg, ossicone spacing 1.2-1.5 x width); (5) on bald human skin choose flush, mounded or hair-hidden, deliberately.

## 5. Wing root, shoulder girdle and back

**Principle.** In every real winged body the wing root is a skeletal joint on a girdle with a muscle mass around it, and the skin or feather coat closes over it without a seam. Where the muscle is differs: birds carry pectoralis and supracoracoideus on the chest and keel (15-30 % of body mass in strong fliers [T]; dorsal wing-root muscles are thin straps, wing/A04, A05, A06), bats a fur-covered shoulder with a membrane root along the whole flank, Draco a membrane held by 5-6 elongated ribs [T] (wing/A20), colugo and flying squirrel a furred skin fold with no skeletal root (wing/A21, A22).

**Birds and bats [M].**

| quantity | value | id |
|---|---|---|
| bird wing-root edge height / body width at the shoulders | **1.31** (165 and 156 px vs 123 px); root runs from the shoulder top to the flank, lower end about **0.4** of body length below the head | wing/A12 |
| raised bird wing leading edge | 3.2 x body width at shoulders; **30-35 deg** above horizontal (55-60 deg from the body axis) | wing/A12 |
| bat radius / humerus | **1.44** (flat plate, +-5 %) vs **1.7** (photo, right wing cut) | wing/A17, general/gen_041 |
| bat humerus / radius / scapula, as fractions of trunk C7-pelvis | 0.84 / 1.21 / 0.28 | wing/A17 |
| bat glenoid span / ribcage width | **1.21** (shoulder joints just outside the ribs) | wing/A17 |
| bat longest digit / radius; arm+hand / head-body; span / head-body | 1.17; 2.4; about 3.7 (estimate) | general/gen_041 |

Bat radius/humerus conflict: two species and two plates; use **1.4-1.7**. A bird's mantle and scapular feathers make a smooth saddle between the two roots with no gap (wing/A12), the humeral feather tract is a narrow band over the shoulder joining wing and back tracts (wing/A03), and a folded wing is a Z of humerus (back) and radius/ulna (forward) lying low and behind the shoulder joint (wing/A07, A10, A11). A bat membrane begins at the fur edge along the flank and runs to the hind limb; folded it lies in soft folds over the back (wing/A14, A15, A16, A19).

**Human back under a back-mounted wing** (Gray 204 = wing/H03 = limb/limb_042; Albinus wing/H01; Vesalius limb/limb_054).

| quantity | value | id |
|---|---|---|
| biacromial / widest rib span | **1.40** and **1.38** | limb/limb_042, limb/limb_054 [M] |
| shoulder joint outside the ribcage | glenoid at 1.18 x rib half-width, so about **0.19-0.20 rib spans** outside each side | limb/limb_042 [D] |
| interscapular gap / biacromial | **0.37-0.38** at the spine roots (0.36 Albinus), **0.52-0.55** at the inferior angles; medial borders about 10 deg off vertical | wing/H03, wing/H01 [M] |
| scapula height / biacromial | **0.42-0.47** (0.436 derived from limb_042); about 54 % of thoracic back height | wing/H01, wing/H03, limb/limb_042 [M] |
| scapular spine rise | **22.7-24.9 deg** above horizontal toward the acromion | wing/H03 [M] |
| lower-rib width / biacromial | 0.72 | wing/H03 [M] |
| clavicle length / biacromial (each side) | 0.49 | limb/limb_054 [D] |
| trapezius slope neck to shoulder tip | **27-33 deg** stocky muscular man; **16 deg** lean swimmer (9 deg along the muscle itself) | wing/H15 [M], limb/limb_071 [M] |
| shoulder width / waist | 1.38 stocky man (photo); bi-deltoid / navel waist 1.67 (Houdon ecorche), 1.61 (Shiva), 1.77 (Vishnu) | wing/H15, limb/limb_078, limb/limb_030, limb/limb_032 [M] |
| neck width / shoulder width | 0.30 | wing/H15 [M] |
| back soft tissue, lumbar section | subcutaneous fat 0.07 x half-width, fascia 0.02, deep back muscle 0.27 at the posterior midline (fat thinner over the scapula) | wing/H08 [M] |

Muscle layers over the scapula, outside in: skin, fat, trapezius and latissimus, rhomboids and levator scapulae, serratus anterior, ribs; deltoid caps the shoulder, infraspinatus and teres major raise the mound on the blade (wing/H02, H05, H06, H07, H12). The scapula is a plate that slides under the skin (arms overhead wing/H10, H14; "winged" blade wing/H18, H19, skin/wm_winged_scapula where the bone edge tents thin skin into a sharp ridge with a very short fillet).

**How sculptors join wings to a human back** [M wing/H22, H24]. Canova Cupid: flight feathers narrow to a root, around the base a collar of progressively smaller overlapping feathers about **0.20 x the root footprint** thick, the marble back swelling into the root with no seam; the two roots sit high on the back flanking the spine between the scapulae (wing/H22, H23). Winged Victory: root footprint is a diagonal band from the shoulder top, **0.84 x torso depth** long (about 0.23 x visible torso axis), wing about 2.9 x the footprint (2.4 x torso depth), leading edge about 96 deg to the trunk axis, a rounded shoulder mound rises behind and above the root; the wing was carved as a separate piece with a large flat root plane hidden by drapery (wing/H24, H25, H26). Miniature scale: wings spring between the shoulder blades with a thick root (wing/H28, internal reference only).

**Wing checklist:** (1) root centred on the scapular region between the shoulder blades (gap 0.37 x biacromial at the spine roots), shoulder joint outside the ribcage line; (2) a rounded muscle mound behind and above the root (trapezius/deltoid/infraspinatus), not a flat slab on the skin; (3) feather or membrane coat closes over it with a collar of small overlapping feathers (about 0.20 x root footprint) or a furred membrane edge along the flank; (4) folded wing lies low and behind the shoulder joint, layered; (5) membrane root is a line along the flank, not a point; (6) root footprint about 0.84 x torso depth for a human-scale angel.

## 6. Extra limbs (arms and legs)

**Principle.** A limb hangs from a girdle, wrapped in a muscle sleeve that spreads far beyond the shaft on the trunk, under one continuous skin dome with at most one shallow crease underneath [S limb/limb_047, 049, 067, 069, 071]. Real supernumerary limbs root on the nearest girdle and share its skin: extra hind legs of a goat kid hang from the pelvis/rump on the same side as the normal hind legs [S limb/limb_023]; a steer's extra forelimb hangs from the front of the shoulder, its first segment a rounded skin-covered lump with skin continuous with the shoulder and no fillet or edge [S limb/limb_022]; radiographs show the extra bones alongside the trunk articulating with or near the existing pelvis (limb/limb_024, limb/limb_025; limb_024 is hard to read, relying on the source caption).

**Girdle geometry.** The shoulder joint stands about 0.2 rib spans outside the ribcage each side (human biacromial / rib span 1.38-1.40, D). The same overhang appears in quadrupeds: scapula outer span **1.44 x** (horse) and **1.50 x** (dog) thorax width, thorax W/H 0.52-0.53 [M skin/url_cuyer_fig14_15]. In humans the scapula sits on the dorsal plane of a wide flat thorax; in quadrupeds on the flanks of a narrow deep one (skin/url_cuyer_fig10_11 [S]); horse and dog have no functional clavicle, the limb hangs from the scapula in a muscle sling (limb/limb_001-003, 015, 017). **An extra arm pair flush against the ribcage with no outrigger will look glued on.**

**Human arm numbers [M].** Arm root thickness 0.345 x chest depth; elbow 0.71, forearm bulge 0.88, wrist 0.41 of root; arm length (glenohumeral to wrist) 2.28 x chest depth (skin/wc_ecorche_side_V0008062). Rodin bronze upper arm 1.00 / 0.82 / 0.79 / 0.77 / 0.79 (armpit to elbow, root width 1.26 x mid-arm), forearm 1.00 / 0.89 / 0.70 / 0.62 / 0.57 (elbow to wrist), forearm elbow end 0.66 and wrist 0.38 of upper-arm mid width (limb/limb_059). Hanging upper-arm width 0.18 x bi-deltoid width, bi-deltoid / waist 1.67 (limb/limb_078 Houdon). Arm width at axilla level 0.65 x head width, neck width 0.80 x head width, trapezius slope 16 deg (limb/limb_071). Acromion at 0.84 of stature, torso (acromion to hip joint) 0.31, humerus 0.19 of stature (limb/limb_054, a stylised engraving).

**Multi-armed iconography (the usable solution)** [M limb/limb_030, 031, 032]: each side gets **one** slightly larger shoulder dome and 2 or more slim arms fan off its lateral and posterior faces at different heights and angles; raised arm diameter **0.14-0.17 x bi-deltoid width**; bi-deltoid / belt width **1.61** (Shiva) and **1.77** (Vishnu); raised hands span 1.52 x bi-deltoid width (Vishnu); arms staggered in elbow height so they never intersect; from behind each shoulder is one continuous rounded mound. Same pattern in limb/limb_033, 034, 036 (18 arms fan like a peacock tail from one pair of shoulders), 037. Offset the arm axes **20-40 deg** in the frontal plane [I, not measured]. Where a second pair must sit lower on the flank, the real visible landmark is the toothed serratus/oblique edge about halfway down the ribcage below the scapula, with the latissimus edge behind (limb/limb_041, 048, 067, 079; 9 serratus slips of which 3-4 show, from the Duval text [T]), not another deltoid.

**Mound for the extra root** [M limb/limb_071, limb/limb_059]: dome 1.2 arm radii above the neck-to-arm chord, fillet 1.4 radii, flare x1.26 over the first quarter; footprint of the soft mass beyond the ribcage line about one upper-arm width (0.18 x bi-deltoid, limb_078) plus the bony outrigger (about 0.2 rib spans).

**Hip and extra legs.** Gluteus maximus is the largest, thickest muscle with the fold of the nates running diagonally across its lower border [S limb/limb_064, 065, 066; text Duval pp.252-254]; the analogue of the deltoid mound for an extra leg is the gluteal/hip mass; iliac width 0.65 x biacromial in the Vesalius engraving (see conflicts).

**Quadruped limb roots (centaur-type joins and extra legs).** Horse (real mounted skeleton): scapula / humerus **1.77**, scapula axis 48 deg above horizontal, humerus 66 deg below, joint angle **114 deg**, joint at **0.63** of withers-to-sternum depth below the withers (limb/limb_006). Dog: scapula/humerus 0.74, scapula 40-45 deg, humerus 66 deg, joint 106 deg, joint at 0.56 of depth, skin point of shoulder 0.38 scapula-lengths ahead of the glenoid (limb/limb_016, single diagram; **0.74 looks low against the textbook ratio of about 1, T**, so treat the dog scapula/humerus value as unreliable). Centaur: horse barrel is much deeper than a human chest, so the human waist must blend into the neck-withers slope in front of the scapula tops over a long ramp [I from limb/limb_013, 014, 008].

**Limb checklist:** (1) girdle plate set outside the ribs (about 0.2 rib spans); (2) one shoulder dome per side with arms fanned off it, arm diameter 0.14-0.18 x bi-deltoid width; (3) skin dome height about 1.2 arm radii, fillet about 1.4 radii, root flare x1.2-1.3; (4) a single axillary crease on the fold side, nothing else; (5) second pair staggered in elbow height and offset 20-40 deg; (6) no separate torso, extra limbs share the nearest girdle's skin.

## 7. Where numbers conflict, and what to use

| # | quantity | values and ids | why they differ | USE |
|---|---|---|---|---|
| 1 | giraffe ossicone height / width | 1.25 (horn/A29, foreshortened, base line above the valley) vs 1.75 (skin/wm_giraffe_head, silhouette) | viewing angle and where height starts | **1.5-1.75**; taper 1/0.84/0.78/0.73 and valley 0.5 r from A29 |
| 2 | bat radius / humerus | 1.44 (wing/A17 plate) vs 1.7 (general/gen_041 photo) | different species; photo is rough with the right wing cut | **1.4-1.7** |
| 3 | tail emergence | tail DIGEST: tail continues the spine on the midline; general/gen_020: dog first caudal 48 deg below horizontal vs horse 13 deg | continuity of position and dorsal skin tangent is true for all; direction of the first segment is species-dependent | position on midline always; **angle as a parameter** (0-20 deg human default, section 3) |
| 4 | human coccyx / sacrum | 0.28 (tail/wc_w3snb3r5, oblique drawing, equal scale assumed) vs 0.47 (tail/cm_Gray97, lateral, along dorsal path) | projection and path-vs-chord | **0.47** from the orthographic lateral plate; textbook 3-5 coccygeal vertebrae [T] fits either |
| 5 | tail taper | cone 1/0.76/0.52/0.38 vs fur cylinder 1/0.9/0.9/0.85 | two body plans, not an error | choose by species; cat **bone** taper (0.98/0.59/0.50) is not a skin taper |
| 6 | lizard tail / body | 1.18 x snout-pelvis (tail/cm_Varanus_salvator_5514) vs 1.61 x snout-vent (general/gen_040) | species and reference points | **1.2-1.6** |
| 7 | trapezius slope | 16 deg lean swimmer (limb/limb_071) vs 27-33 deg stocky man (wing/H15) | body type and definition (neck base to cap vs neck to tip) | **by body type**: 16-20 lean/female, 27-33 muscular |
| 8 | shoulder / waist | 1.24 shoulder:iliac (Albinus skeleton), 1.38 stocky man in a photo, 1.61-1.77 bi-deltoid / waist in ecorche and icons | different quantities (bone vs skin, hips vs waist), heroic idealisation in art | realistic **1.4**, heroic **1.6-1.8**; for female add waist and trochanter width rather than narrowing shoulders (repo docs/realism_research.md) |
| 9 | biacromial / stature | 0.207 (Albinus, 8.7 heads tall, general/gen_005), 0.26 (Vesalius, stylised, limb/limb_054), about 0.23 real adults (repo docs/realism_research.md, not from this library) | engravings exaggerate stature or shoulders | **0.23**; use the images only for relative proportions |
| 10 | iliac width / biacromial | 0.65 (limb/limb_054) vs 0.81 (1 / 1.24, general/gen_005) | Vesalius shoulders are wide | **0.7-0.8** |
| 11 | wing root footprint | bird 1.31 x shoulder-width tall (wing/A12) vs angel 0.84 x torso depth (wing/H24) | a bird's wing is its forelimb, so its root spans the whole shoulder; sculptors scale the root to the visible back | not comparable; **0.84 x torso depth** for a human-scale angel |
| 12 | scapula/interscapular agreement | wing digest and limb digest agree (0.37-0.38, 0.44) | **same plate** (Gray 204); independent confirmation only from Albinus 0.36/0.42 and Vesalius 1.38 | state as one plate plus two engravings |
| 13 | fillet widths | 0.1 r (hard cap, axilla, withers) / 0.5 r (ossicone) / 0.6 r (fin) / 1.4 radii (deltoid) | different junction types | use the type table in section 2, never a single value |
| 14 | thorax overhang | human about 0.20 rib spans each side (D); horse 0.22, dog 0.25 x thorax width each side (D from 1.44 and 1.50) | same idea in two body plans | **agreement**: shoulders stand about 0.2 outside the ribs |
| 15 | arm root flare | x1.26 (Rodin bronze) vs x1.31 (live shoulder) | sculpture vs photo | **agreement**, use 1.25-1.3 |
| 16 | human horn base | 0.10 x cranial width (horn/H17) vs 0.30 x head width (horn/H22, stylised) | realism vs stylised print | **0.10-0.15** realistic |
| 17 | Shiva vs Vishnu shoulder / torso | shoulder 1.59 x torso length (Shiva) vs torso 0.77 x bi-deltoid, i.e. 1.30 (Vishnu) | stylistic | **1.3-1.6** |

### Animal vs human, in one place
- **Seam**: animals use hair lines, keratin rims and wrinkle collars; bald human skin has only flush, mounded or hair-hidden roots (section 2).
- **Tail posture**: a quadruped tail continues a horizontal lumbar line; the human sacrum/coccyx turns ventrally (about 57 deg) in an upright body, so a human tail root must start from the sacrococcygeal midline and use a small outward bend, and the cleft and gluteal mounds are part of the surface.
- **Mound**: in quadrupeds the rump/haunch musculature is the tail's mound, in humans it is the gluteal mass plus fat; neither has a separate "collar".
- **Wing**: bird flight muscle is on the chest and keel, a human back-mounted wing has none, so the believable proxy is trapezius/deltoid/infraspinatus swelling around a scapular root, i.e. a design inference [I].
- **Horn site**: animal horns grow from bone processes at the poll, above the orbit or on the nasal tip; the human skull has no pedicle, so the root site is chosen by what lies over the bone (galea and periosteum on forehead/vertex, temporalis at the temple).
- **Girdle**: quadruped scapulae sit on narrow deep flanks, human scapulae on a wide flat back; an extra arm pair needs a human-style dorsal/lateral girdle plate.

## 8. Gaps (what this library does not have)

Imaging and sections
- No CT, MRI or 3D scan of any junction (shoulder, sacrum, skull); only three plain radiographs (cat lumbar spine tail/cm_Cat_Spine_Xray_Img_1, polymelia limb/limb_024, limb/limb_025). No cross-section of an animal tail, horn base (sheath/core/skin) or ossicone: horn/A06 is a 259x331 px diagram, horn/H13 a schematic with no thickness.
- Caudofemoralis and the caudal muscle plates (Chauveau, Ellenberger) were not retrieved; those facts are [T]. No Gray's gluteal-region plates.

Surface photographs
- No clean high-resolution live photograph of a cat or dog tail root from behind; no live crocodile in profile; no close-ups of the olecranon, knee, iliac crest or clavicle transition; no photograph of a rhino or bovid horn base, a big cat's shoulder, or a bird's plucked back or folded-wing base (only a vulture girdle diagram); no bat shoulder close-up with the membrane attachment; no elephant trunk-base cross-section.
- Human body types are thin: lean athletic and stocky muscular men, a few statues, Muybridge frames (male, female, obese female). No heavier, older, or child bodies as modern photographs (children only as drawings and one 1900s medical photo); one sacral/gluteal surface close-up; no bald-head series in varied head types (only ecorche plates and sculpture); no bodybuilder lat-spread back photo; Loomis and Bammes plates are not free.
- No photographs of a healed horn or spike implant or transdermal horn (only subdermal domes and one dermal anchor); no frontal-osteoma images; no photograph of a real multi-armed fantasy miniature (searches returned nothing); no close-up of a polymelia limb root.

Measurement weakness
- Skin fillet width and mound height could not be measured as separate features for tails (only the 3 % contour continuity and the ventral crease, see section 2); the tail fillet numbers are inferences.
- Some values are eyeballed (cat tail/head-body 0.74, kangaroo base, lemur and spider-monkey widths, human horn dimensions); dog scapula/humerus 0.74 is suspect; every number is n = 1.
- Sparse manifest coverage: only 64 of 357 manifest rows carry a `measurements` object (the DIGESTs cite a few more); no image outside those has numbers.

Licensing and sources
- `tail/cm_A_child_with_a_tail_side_and_posterior_view_Repr` (the only real human caudal-appendage photograph) is CC-BY-4.0 on Commons but "In Copyright" on Wellcome: research reference only, excluded from `appendage_references.md`. `wing/H28` is a modern commercial figurine (internal only), `horn/A12` is GFDL-1.2.
- Blocked or throttled hosts (what the gatherers hit): biodiversitylibrary.org, pixabay.com, pexels.com, si.edu / api.si.edu / openaccess-api.si.edu, images.nga.gov and artic.edu IIIF (403/502/429); Met `/public/collection/v1/search` retired (410, `/v1.1/search` works); Wikimedia Commons API, upload.wikimedia.org originals and Wikipedia API throttled (429, retry-after 20-600 s; standard thumbnail widths 500/960/1280 worked); Openverse thumbnail endpoint for Wikimedia items returned 424; anonymous Openverse limit 20/min burst. Reachable and used: api.openverse.org, api.wellcomecollection.org and iiif.wellcomecollection.org, archive.org, Met v1.1 and images.metmuseum.org, rijksmuseum.nl, api.europeana.eu, api.artic.edu (metadata), ids.si.edu (via Openverse), live.staticflickr.com, thumb.wikimedia.org; collections.nlm.nih.gov and images.unsplash.com were reachable but unused.

Next best additions (if more collecting is done): a Chauveau/Ellenberger tail-myology plate and a Gray gluteal plate (both public domain, on archive.org); BHL bird-skeleton plates once the host is reachable; modern CC photos of a horse or dog rump from behind and a cat tail root; CC photos of bighorn or ibex with the horn base visible; a CT scan of a skull or shoulder under a CC licence (for example from NIH/NLM or Wikimedia Commons "CT" categories).
