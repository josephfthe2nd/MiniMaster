# MiniMaster anatomy & proportion spec

A practical reference for **building body parts** and **placing rig joints**,
expressed as fractions of total height `H` and of each segment's length so it
drops straight into the parametric generator (`tools/make_templates.py`,
`tools/make_bodyparts.py`) and the armature layout.

> **Sourcing note.** Automated web-fetching was blocked (proxy 403s) when this
> was compiled, so the numbers below are the **standard artistic-anatomy
> canon** — Loomis (*Figure Drawing for All It's Worth*, the 8-head "ideal"),
> Richer / Vitruvian (7.5-head realistic), and Bridgman (*Constructive
> Anatomy*, muscle masses). These are stable, widely-taught figures; treat the
> ranges as canonical rather than freshly-cited, and we can attach specific
> citations if source fetching is restored.

Coordinate reminder: MiniMaster is Z-up, mm, character faces **−Y**; `+X` is
the figure's left. Heights below are measured **from the sole (z=0) up**.

---

## 1. Proportion canon (vertical landmarks)

Two references. The **8-head heroic** canon is the target for player heroes;
**7.5-head realistic** is the grounded baseline. MiniMaster's current figure
is intentionally **stylised at ≈ 6.25 heads** (chunky, OSRS-ish) — see §6.

Landmark heights as a **fraction of total height** (sole = 0.0, crown = 1.0):

| Landmark                    | 8-head heroic | 7.5-head real | Notes |
|-----------------------------|:-------------:|:-------------:|-------|
| Crown (top of head)         | 1.000 | 1.000 | |
| Chin (base of head)         | 0.875 | 0.867 | head = 1 unit ≈ 0.125–0.133 H |
| Shoulder line (acromion)    | 0.820 | 0.810 | ~1 head + a bit below crown |
| Nipple / pectoral line      | 0.720 | 0.710 | ≈ 2 heads down |
| Bottom of ribcage           | 0.660 | 0.650 | solar plexus |
| Navel                       | 0.630 | 0.620 | just above the ½-way third |
| Iliac crest (top of pelvis) | 0.590 | 0.580 | |
| **Crotch / pubic symphysis**| **0.500** | **0.480** | **the mid-height point** ✔ |
| Mid-thigh (fingertips)      | 0.370 | 0.360 | relaxed fingertips reach here |
| Knee (joint line)           | 0.260 | 0.255 | ≈ 2 heads up from ground |
| Mid-calf (widest)           | 0.180 | 0.175 | |
| Ankle (malleoli)            | 0.055 | 0.050 | |
| Sole                        | 0.000 | 0.000 | |

Key fact used everywhere: **the pubic bone is the halfway point of the
figure** — legs and torso+head each occupy ~half the height.

**Widths** (in head-*heights*; head width ≈ 0.66 × head height):

| Dimension            | Male     | Female   |
|----------------------|:--------:|:--------:|
| Shoulder (biacromial)| 2.0–2.3  | 1.5–1.75 |
| Hip (bitrochanteric) | 1.5–1.7  | 1.6–1.9  |
| Shoulder ÷ hip       | ~1.4     | ~0.95    |
| Waist                | ~1.1     | ~1.0     |

Men: shoulders clearly wider than hips (inverted trapezoid). Women: hips ≥
shoulders, higher & narrower waist.

---

## 2. Segment length ratios (for rigging)

**Arm** (relaxed, hanging): upper arm : forearm : hand ≈ **1.5 : 1.2 : 0.75**
head-units. When the arm hangs:
- **Elbow** falls at the **navel/waist** (~0.62 H)
- **Wrist** falls at the **crotch** (~0.50 H)
- **Fingertips** reach **mid-thigh** (~0.37 H)
- **Arm span ≈ height** (Vitruvian).

**Leg**: thigh : shin : foot-height ≈ **2.0 : 1.7 : 0.5** head-units.
- Foot **length** ≈ 1 head (≈ 0.13 H); foot height ≈ 0.4 head.
- Thigh (crotch 0.50 → knee 0.26) ≈ 0.24 H; shin (knee 0.26 → ankle 0.055) ≈
  0.20 H.

**Head**: length : width : depth ≈ **1 : 0.66 : 0.85**. Hand length ≈ face
length (chin→brow); foot length ≈ head height.

---

## 3. Joint rotation-centres (rig-critical)

The armature pivot is **not** the visible surface bulge. Place joints at the
**bone rotation centre**:

| Joint            | Height (frac H) | Placement notes |
|------------------|:---------------:|-----------------|
| Neck base        | ~0.80 | pivot at the pit of the neck / C7, tilts slightly forward |
| Shoulder (GH)    | ~0.80 | **below & slightly forward** of the acromion; medial to the deltoid cap, not at its outer edge |
| Elbow            | ~0.62 | at the joint line; pivot sits at the back of the arm mass |
| Wrist            | ~0.50 | mid-wrist, between the malleoli of the forearm |
| **Hip (femoral head)** | **~0.50** | **medial and higher** than the fleshy hip/greater-trochanter bulge — set it *inward* toward the midline and up near the pubic line, NOT at the wide part of the hip. This is the single most-common rig error. |
| Knee             | ~0.26 | at the joint line behind the kneecap |
| Ankle            | ~0.055 | at the malleoli; **set posterior** in the foot — the foot extends forward (−Y) from it, heel behind |

Consequences for the model: the **thigh angles inward** from the hip pivot
(inner-knee), so upper legs converge slightly toward the knees; the **foot
pivots at the back**, so `toe`/`foot` shapes should extend forward of the
ankle joint, with a short heel behind it (MiniMaster already does this).

---

## 4. Surface muscle masses → primitive placement

One or two primitives per mass; the goal is the **silhouette**, not detail.
"Along segment" is fraction from the proximal (body-side) joint = 0 to distal
= 1. Taper = thick→thin direction.

| Mass | Primitive | Along segment | Offset | Taper |
|------|-----------|:-------------:|--------|-------|
| **Deltoid** (shoulder cap) | icosphere/half-capsule | 0.0 of upper arm, wrapping onto torso | slightly outward+forward | rounded, blends down into arm ~0.35 |
| **Pectoral** (chest) | two flattened slabs / one wide capsule | upper torso | front, high | tapers down to the sternum notch |
| **Trapezius** (neck slope) | wedge/tapered capsule | neck base → shoulder | back+top | fills the neck-to-shoulder diagonal |
| **Latissimus** (back V) | tapered slab | mid-back → waist | back, sides | wide at ribs → narrow into waist |
| **Biceps** | short capsule | 0.15–0.6 of upper arm | **front** | peak upper-third, fades to elbow |
| **Triceps** | capsule | 0.1–0.8 of upper arm | **back** | fuller than biceps, to the elbow |
| **Forearm flexor/extensor** | tapered cylinder | **bulk at 0.0–0.35**, thin by 0.8 | mass on the **outer/upper** (extensor) side | **thick at elbow → thin wrist** |
| **Gluteus** (buttock) | capsule/ico pair | back of pelvis | **back**, slightly down | rounded |
| **Quadriceps** (front thigh) | tapered cylinder | 0.0–0.85 of thigh | front | **thick at hip → thin at knee** |
| **Hamstring** (back thigh) | subtle capsule | 0.1–0.8 | back | thick→thin to knee |
| **Gastrocnemius** (calf) | capsule | **bulge HIGH, 0.05–0.45 of shin** | **back**, inner head lower than outer | **thick just below knee → thin Achilles/ankle** |
| **Tibia** (shin bone) | (leave flat) | front of lower leg | front | subcutaneous — a flat, hard front edge |

**Limb taper summary** (thick end → thin end): upper arm **shoulder→elbow**;
forearm **elbow→wrist**; thigh **hip→knee**; shin **knee→ankle**. Every limb
is fat at the body, thin at the extremity. Joints (elbow, knee, wrist, ankle)
are the **narrow pinch points**; the muscle bellies sit just proximal to them.

---

## 5. Variant builds (multipliers on the male-heroic baseline)

Expressed as rough multipliers / deltas for the generator's proportion knobs
(`head`, `legs`, `arms`, `bulk`, `shoulders`).

| Build | Heads tall | head | shoulders | bulk | legs | Signature |
|-------|:----------:|:----:|:---------:|:----:|:----:|-----------|
| **Male heroic** | 8 (styl. 6.25) | 1.0 | 1.0 | 1.0 | 1.0 | inverted-trapezoid torso |
| **Female** | 7.5 | 1.0 | 0.8 | 0.9 | 1.05 | narrow shoulders, wide high hips, narrow waist, softer tapers, less deltoid |
| **Dwarf** | ~4 | 1.5 | 1.15 | 1.35 | 0.6 | huge head-ratio, broad, long torso / short thick limbs, big beard mass |
| **Goblin** | ~4.3 | 1.5 | 0.85 | 0.75 | 0.95 | oversized head+ears, spindly limbs, hunched spine, big hands/feet, pot belly optional |
| **Orc** | ~7.5 | 0.95 | 1.4 | 1.45 | 0.9 | massive trapezius/deltoids, long arms (fingertips past knees), short thick legs, forward neck hunch |
| **Skeleton** | 8 | 1.0 | 0.9 | 0.5 | 1.0 | **no muscle bellies** — bone only; joint knobs (epiphyses) are the WIDEST points; ribcage barrel, pelvis basin, visible spine |

Notes:
- **Female**: raise the waist/navel ~0.03 H, widen hips, drop shoulder width,
  remove the deltoid emphasis, smooth all limb tapers.
- **Goblin/Dwarf**: the big-head look comes from a large `head` multiplier
  (head ≈ 1/4 of height, not 1/8).
- **Orc**: the hunch is a forward `spine`/`neck` pose offset plus oversized
  shoulder/trap masses; give it the longest arms.
- **Skeleton**: invert the taper logic — the shapes should **bulge at the
  joints** (ball epiphyses at knee/elbow/wrist/ankle/shoulder) and be thin in
  the shafts.

---

## 6. Mapping to MiniMaster (`build_humanoid` & the rig)

The generator lays joints out as fractions of `H`. Current vs. canon-aligned
(stylised ~6.25-head target, so the head stays chunky):

| Joint z | Current | Canon-aligned (frac H) | Change |
|---------|:-------:|:----------------------:|--------|
| ankle   | 0.07    | 0.055 | slightly lower |
| knee    | ~0.26*  | 0.26  | ok |
| hip     | ~0.49*  | 0.50, **pull X inward** ~15% | hip pivot medial (§3) |
| pelvis/crotch | 0.50 | 0.50 | ok — confirms the mid-height rule |
| navel/waist (spine) | ~0.60 | 0.62 | raise slightly |
| chest/nipple | ~0.72 | 0.72 | ok |
| shoulder | ~0.80 | 0.80, **pivot inward** of deltoid | §3 |
| chin     | ~0.84  | 0.84  | ok |

\* derived from the current `legs`/`bulk` formulas.

**Concrete, high-value changes** (for the next modelling pass):
1. **Hip pivot inward** — move `hip_x` toward the midline (~0.045 H instead of
   0.054) and let the thigh top flare outward to the fleshy hip, so legs
   converge to the knees (canon inner-knee line). Biggest realism win, and it
   fixes rig rotation.
2. **Muscle bellies just proximal to joints** — bias the existing calf/biceps
   masses toward the *upper* third of their segment (§4) and pinch the wrist
   and ankle thinner.
3. **Forearm & shin taper** — increase taper so both end thin at the
   wrist/ankle (already partly done; push further).
4. **Deltoid vs shoulder pivot** — keep the deltoid cap wrapping outward but
   keep the *armature* `shoulder` joint inboard of it.
5. **Per-build knobs** — wire the §5 multiplier table into the species
   presets (esp. female shoulders/hips, orc hunch+arms, skeleton joint-knobs).

These are expressed as parameters, so each is a small change to
`build_humanoid` / `make_bodyparts.py` rather than per-shape hand-tuning.

---

## 7. Extra arm pairs (polymelia / four-armed builds)

Adding a second pair of arms is **not** "duplicate the arms lower down." An
arm hangs off a **shoulder girdle** — a clavicle (front strut to the sternum)
and a scapula (floating plate on the back) — and a whole set of muscles
anchors that girdle to the axial skeleton. A second pair needs a **second
girdle**, and the muscles that serve it change accordingly.

### Skeleton — two girdles, a longer thorax

The two believable layouts:

- **Tandem (recommended).** Stack the girdles vertically and **lengthen the
  thorax** by ~1 head so there's room. Upper arms at the normal shoulder line
  (~0.80 H); lower arms near the **bottom of the ribcage / upper lumbar**
  (~0.62–0.66 H). This is the standard creature convention (marilith, Goro)
  and reads cleanly. The sternum lengthens; add rib pairs to fill the taller
  chest.
- **Broadened (side-by-side).** Both girdles near the same height, shoulders
  widened dramatically. Reads as "very broad," less clean; avoid unless the
  creature is meant to be a wall of shoulders.

Either way the **torso stretches or thickens** — a four-armed figure carries a
longer/deeper thorax and a thicker core, because four arms' worth of leverage
needs a stronger spine to stabilise.

### Muscles that change (this is the "muscle accounting")

| Muscle | Single pair | With a 2nd pair |
|--------|-------------|-----------------|
| **Deltoid** | one cap per shoulder | **four caps** — a second deltoid on each lower shoulder |
| **Pectoralis** | one chest sheet anchoring the arms | a **second, lower pec band** for the lower arms — the chest becomes a two-tier stack of slabs |
| **Trapezius** | suspends one girdle | **enlarged / two-tiered** — an upper trap for the top shoulders plus a second band suspending the lower girdle; the neck-to-shoulder mass grows |
| **Latissimus / back** | one V into the waist | **broadened, taller** back sheet (or two V's) anchoring both arm sets — the back becomes very wide |
| **Serratus / rhomboids** | hold one scapula pair | **duplicated** down a taller ribcage to hold the second scapulae |
| **Rotator cuff** | per shoulder | duplicated per shoulder (internal, not silhouette) |
| **Spinal erectors / core** | baseline | **thicker** — the mid/lower back and obliques bulk up to brace four arms |

Net silhouette change: **broader, deeper, longer torso; a stacked chest;
a much wider, more muscular back; a thicker neck.** The lower arms are often
drawn slightly **smaller (~0.85–0.95)** than the uppers (primary vs secondary
limbs), or equal for a pure brute.

### Parametric rule for the generator / rig

Model an extra arm pair as an **additive graft** (which is exactly how the
part & rigging system will do it):

1. **Joints.** Add `shoulder2_{l,r} → elbow2 → wrist2 → hand2`, parented to a
   **lower chest bone** (or the spine) at z ≈ 0.62–0.66 H, x ≈ the upper
   shoulder's x (optionally 5–10 % narrower). Scale the whole chain ~0.9.
2. **Thorax.** Lengthen/deepen the chest ~0.6–1.0 head and widen the torso
   ~12 % so the lower girdle has an anchor.
3. **Muscle primitives to add:** a lower deltoid cap per lower shoulder; a
   lower pec band on the front at the lower-shoulder height; a second
   trapezius/rhomboid band on the back between the two shoulder rows; a
   broadened lat/erector mass. These are the same primitive types as the
   single-pair masses in §4, just instanced for the second girdle.

Because it's additive, it composes: `arm_pairs = 2` (or a shipped
"four-armed" creature) just runs the arm-and-girdle builder twice at two
shoulder heights and adds the second set of anchoring muscles. The rule
generalises to **N** pairs (centipede-of-arms) — each pair is another girdle
down a proportionally longer thorax.

> **Implemented.** `tools/make_templates.py:add_arm_pair()` is exactly this
> additive builder (grafts `shoulder2→elbow2→wrist2→hand2` per side plus the
> four-deltoid / lower-pec / trap-yoke / lat muscles), and the shipped
> `four_arms` template (`make_four_arms`) uses it. Call it after
> `build_humanoid` on any base to make that creature four-armed.
