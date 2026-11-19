# The weld: the body's own skin moulded onto every root (round 6)

Code: `minimaster/character/weld.py`; per kind `Kind.weld_profile` / `weld_root` / `weld_target` / `welds`
(`minimaster/character/kinds.py`: `TAIL_WELD`, `HORN_WELD`, `WING_WELD`, `GIRDLE_WELD`); tests
`tests/test_weld.py`; viewer `tools/weld_view.py` (the pipeline's own body, with `--mono` / `--flat` geometry modes).
Design context: [junctions.md](junctions.md) (the cage junction, which still gives the skin its mass) and
[collar.md](collar.md) (the round-4 overlay collar this replaces as the default).

## Why

The user: the appendages "seem placed onto the model", "slapped over skin"; the skin does not "mold or conform to
connect around" the root. Two ways failed before:

1. **Moving the body's cage** (`junction.py`): a cage quad is ~0.28 units and a root 0.4-0.6 across, so the cage
   cannot make a tangent fillet: the skin meets a root 55-76 degrees from tangent (the table below, "bare").
2. **An overlay collar shell** (`collar.py`): a second, thin surface on the skin. As geometry (a print, an STL
   viewer) its outline stayed on the skin, and it was absent at level 0, in the lab draft and wherever its guard
   dropped it.

## The idea

The subdivided body (levels 1-2) is derived (morph targets address the cage only), so its vertices may be moved at
mesh time. Around each root the skin's own vertices are moved onto a fillet that leaves the appendage's wall
tangentially at height `k` and lands on the original skin about `region * k` away. One surface climbs the root:
there is no second shell and no rim on the skin. The vertices that were under the root become a plug inside it,
so the skin stays one closed surface (the slicer's union is unchanged, and the printed body is still watertight).
The weld also runs on the level-0 cage's own mesh (coarser, but cheap and better than the bare step).

## The fields and the construction

* `a`: height over the ORIGINAL skin; `b`: distance outside the root's wall (both signed distances along the
  smooth normal; `weld.Surface`, whose closest point is the closest TRIANGLE's: on the tail's tube the nearest
  vertex's triangles missed the closest one for 6.5 % of the points within 0.6 units of the root, by up to 0.2 units).
* The fillet is the zero set of the smooth union `smin(a, b / R; k)`; its profile is `(a, b) = (k (1-h)^2, R k h^2)`,
  `h` in 0..1, tangent to the wall at `h = 0` and to the skin at `h = 1`.
* `k` per vertex (`_reach`): the kind's shape (`WeldProfile.up / open / ventral`), cut where the wall leans over the
  skin (`acute`), at most the skin's concave radius (`valley`) and the room over the skin (a probe up the normal
  that comes near another part of the skin: the facing cheek across the natal cleft, `CLEAR_*`; on the kind's tight
  side in full, not on its open side, where a probe from the floor of the spinal furrow over a heavy body's tail
  found the furrow's own walls), fading with the distance from the root's exit point, zero where the closest wall
  point is beyond the root (`wall_clip`: a drooping tail is not welded to the thighs).
* A cut to the reach stays as deep as it is and fades out at a bounded slope (`REACH_SLOPE` per unit of length over
  the skin outside the wall, `RING_SLOPE` along the last ring; `envelope`). The first pass smoothed by repeated
  min-averaging, which carried the crease under a tail round to its back, and let the hidden vertices under the root
  drag the last ring's reach down.
* The sets: the cap (under the wall; holes filled), the last ring (`top`, thinned so no triangle has three
  corners on it), the rim (fixed), the sleeve.
* The last ring climbs the wall along the root's generator (its axis laid on the wall) to height `k`, at most
  `CLIMB_MAX * k` (2 reaches) of travel: on an oblique wall (the back above a tail) one reach of travel gains only
  three quarters of the height; where the wall runs along the skin the height it reaches is its reach (a crease),
  and the sleeve next to it takes that reach.
* The sleeve follows the last ring as a membrane does (`PLACEMENT = "harmonic"`): the DISPLACEMENT is harmonic over
  the sleeve between the ring (on the wall) and the rim (fixed), so the skin's own relief rides along and the
  vertices keep their order and spacing round the root; it is then laid onto the zero set by Newton steps, evened
  along the fillet, and its rings are packed toward the wall (`grade`, in full only where the reach is healthy).
  The first pass sent every vertex to a preset place on its own profile (`"profile"`): where the reach changes
  round the ring (a tail's shoulders) that left rays of sliver triangles, and the guard cut the reach of the whole
  back for them (0.52 to 0.23 on the neutral tail).
* The cap's first ring moves in proportion to the local reach (`CAP1_FULL`): pushed on by a fixed step it crossed
  its mirror image in the cleft under a tail at level 2.
* The cap's first ring sits just under its neighbours on the last ring; the rest of the cap is a harmonic membrane
  spanning it (conjugate gradients); the whole plug is pushed under the wall (signed distance, then ray parity
  against the wall's shells).
* The guard: after every construction the exact triangle-crossing count of the faces around the root (and around
  every earlier root's weld) is compared with the bare skin's, and faces turned over (normal against the blend's
  gradient by more than about 107 degrees, `FLIP_COS`: an edge-on sliver is not a fold) are found; the reach is
  cut there by `GUARD_SHRINK` and the root solved again (`guard` soft rounds), then, if that did not clear it,
  `GUARD_HARD` rounds take the offending patch out of the weld (the fold and one ring round it: reach 0 there,
  fading in around it, everywhere else the reach the root started with) and keep the rest of the root welded. If the skin would still cross itself more than the junction left it, or a triangle would
  be degenerate, the root is left as the junction made it and a warning says so: torn skin never ships.
* `diag["kept"]`: the share of the root's ring that is welded in the end (of the first construction's last ring,
  the vertices still on the ring with a reach of at least `KEPT_REACH` edges). Under the profile's `kept_min`
  (`KEPT_MIN`, 0.75; a tail's weld is a horseshoe by design and takes 0.5) the root is not welded at all: close up, a
  weld that stops part of the way round reads no better than the bare crease and leaves a ragged fringe where it
  stops. The command line warns and the lab notes it (`diag["refused"]`).
* A mirrored pair (`NAME.l` / `NAME.r`): each side's weld keeps to its own half of the body and its reach ends
  before the midline (`half`). Without it the side welded first took the skin between a pair of legs and its mirror
  image got what was left.
* Several roots: a later root never moves an earlier root's weld (its region is locked and the reach ends before
  it) and never moves skin that lies inside another root's wall (that root's plug).

## The switch and the level

* `weld.seam(character)`: `"weld"` (the default; nothing is written to the document), `"none"`
  (`anatomy["collar"] = false`: `--no-collar`, the bare junction, as before), `"collar"`
  (`anatomy["collar"] = "shell"`: `--collar-shell`, `Lab.doc.set_anatomy(collar="shell")`: the round-4 overlay
  collar instead of the weld). Never both. The junction off means no seam.
* A record's `collar` parameter (0..2, its old meaning: the size of the root seam) scales the weld's reach; 0 is
  the bare root.
* `weld.default_levels(character)` decides the default subdivision from the RECORDS (nothing built) for the command
  line, the lab's final render and its export (they disagreed before: the command line asked the built appendages
  through `collar.default_levels(placed)`, the lab the records): 1 when a root is welded, else 0.
* What is welded: a tail (its base radius stops at 0.9 since round 6; an old document's thicker one is built at
  0.9 and welded), a keratin horn, a wing's arm (ramping in as the arm rises off the back; the folded preset has
  none: the arm lies along the back in a crease), a girdle limb (lower arms, upper arms, extra legs). Not: an
  ossicone (its foot is the skin's own fillet), a first-version limb.

## Where it runs

`__main__.py` (PNG, `--layers` via `ecorche.layer_shells(weld=...)`, STL and its print report, which is now given
the welded body), `Lab.final_scene` and `Lab.export` (one cached weld per shaped body, level, specs and poses:
`Engine.welded_body`, `stats["weld"]`), `tools/limb_view.py`, `tools/weld_view.py`. The lab DRAFT (orbit, drags) is
the 4k-triangle proxy as before, without a weld, and says so in its notes.

## The measures (`weld.arrival_report`, geometry only)

The angle between the normal of every triangle that reaches the wall from outside and the wall's normal there
(0 = the skin arrives tangentially, 90 = a pipe in a plate), split by the kind's open side; exact crossings around
the root before and after; plug vertices that left the wall (ray parity against the wall's shells). Bare = the
junction's skin without the weld (what e7e16b8 draws under its collar and at level 0); p50 / p90 of the open side,
the round-6 measurement on the shared 4-core box:

| kind | level | bare (all) | welded (open side) | welded (all) | guard rounds | s / root |
| --- | --- | --- | --- | --- | --- | --- |
| tail, neutral | 1 | 70 / 82 | 9 / 32 | 22 / 60 | 0 | 0.7 |
| tail, neutral | 2 | 70 / 79 | 12 / 18 | 14 / 39 | 0 | 3.7 |
| tail, heavy / child / slim_f / tall / short | 1 | 63 / 58 / 51 | 11 / 19 / 7 / 9 / 9 | 20 / 27 / 24 / 22 / 28 | 1 / 1 / 0 / 0 / 0 | 0.5-1.1 |
| tail, lift 40 / -40 / 60 | 1 | 65-73 | 17 / 7 / 9 | 21 / 22 / 19 | 0 | 0.6-0.8 |
| tail, base radius 0.85 | 1 | 45 | 9 / 18 | 16 / 47 | 4 | 10 |
| wing (membrane), neutral | 1 / 2 | 70 / 73 | 14 / 9 | 16 / 10 | 0 | 0.4 / 1.9 |
| wing, heavy / child / slim_f | 1 | 58 / 74 / 88 | 17 / 10 / 10 | 22 / 13 / 15 | 0 | 0.3-0.5 |
| half-folded / dragon / feathered, neutral | 1 | 62 / 71 / 47 | 13 / 15 / 13 | 21 / 15 / 18 | 0 / 0 / 1 | 0.4-0.5 |
| horn / ram / spiral, neutral | 1 | 50 / 49 | 28 / 26 / 32 | 28 / 26 / 32 | 0 | 0.15 / 1.2 / 0.2 |
| horn / ram, neutral | 2 | 49 / 56 | 13 / 8 | 13 / 8 | 0 | 0.6 / 5.6 |
| lower arms, neutral | 1 / 2 | 76 / 75 | 5 / 5 | 13 / 6 | 0 | 0.5 / 2.5 |
| lower arms, heavy / child / slim_f | 1 | 76-78 / 75 / 76 | 6 / 6 / 7 | 12 / 13 / 13 | 0 | 0.3-1.7 |
| upper (fan) arms, neutral / heavy | 1 | 56 / 52 | 21 / 15 | 21 / 15 | 3 | 2 / 4 |
| extra legs, neutral / child / slim_f (both sides alike) | 1 | 54-60 | 12 / 15-16 / 10-11 | 13 / 16 / 11-13 | 1 / 0 / 1 | 0.8-1.2 |
| tail / wing / lower arms, neutral | 0 | 75 / 66 / 71 | 13 / 14 / 8 | 40 / 27 / 21 | 0 | 0.2 |

No welded root crosses itself more than the bare skin did and no plug vertex leaves its wall (exact counts over
this matrix). Measured on the tree with the round-6 tail and wing anatomy merged, on the shared 4-core box.

Left bare (the weld could cover less than `kept_min` of the ring; the warning says so). The share is what the
guard could keep:

| root | kept share of the ring |
| --- | --- |
| upper (fan) arms, slim_f / child | 0.50 / 0.64 |
| upper arms beside feathered wings, neutral | 0.72 |
| extra legs, heavy | 0.59 / 0.58 |
| extra legs beside a tail, neutral | 0.73 |

Welded with part of the ring left out: upper (fan) arms on the heavy body (0.79), a 0.85-radius tail (0.55, its
back and flanks; arrival 9 degrees on the open side).

## Round 7 (the session alone, no critic yet)

* **A clean line where the root is another colour.** The last ring stood exactly on the wall; the skin arrives nearly
  flat and the wall is faceted, so the wall came through the last strip between the ring's vertices: a sawtooth line
  round every horn and wing arm. The ring stands `TOP_LIFT` (0.1) edges off the wall now, and the strip from it to
  the plug goes into the wall steeply. A welded horn's base is drawn in the skin's colour to just past the welded
  skin and blends to keratin above it (`kinds.HORN_TINT_*`, a `TintedMesh`, as the wing arm's base already was).
* **The measure changed with it.** `arrival_report` measures the skin's first strip OUTSIDE the wall (the strip that
  crosses the wall is steep by design now), and it queries only the skin inside the wall's own box (a horn's report
  took 18 s, the whole face; now about 1.5 s). The table below is re-measured with it; the round-6 table above was
  measured on the triangles that reach the wall, so its numbers are not comparable line by line.
* **The fillet is as wide as the mesh can draw** (`WIDTH_EDGES`): at least 3.2 of the skin's longer edges, at most
  `WIDTH_ASPECT` (3) reaches. A cuff narrower than that held a ring of skin between the wall and the rim only by
  luck, and without one the root was left bare (horns of radius 0.13 and 0.15 between welded ones of 0.11 and 0.17).
  A profile can give a least reach (`reach_min`; a horn's is one level-1 edge, 0.14). A root the mesh still cannot
  draw a ring round is left bare with a warning and a lab note (`diag["coarse"]`): a tail of radius 0.10 at level 1.
* **A wall lying on the skin is not welded, for a kind that says so** (`WeldProfile.lid`; the fan arms: `LID_COS`,
  21 degrees, within `LID_NEAR` reaches of the wall). Where the fan arm's dome lies on the shoulder the two surfaces
  are tangent already; the skin there lies a hair over the wall here and a hair under it there, and welding that
  patchwork made slivers that crossed (the plug's first ring through the visible strip). The guard cut them and their
  surroundings out and the root ended under `kept_min`: every fan arm that was left bare folded at that same place.
  The reach is 0 there and fades in beside it, and the guard's crossing count takes in the skin left alone. The share
  of the skin beside the wall this touches (`diag["lid"]`): 9-31 % on the fan arms. It is not a rule for every kind:
  as one it took the back's ramp off a tail on the slim female body (a tail leaves the back that flat there, and the
  weld is what makes the ramp) and let a thick tail's weld cross skin the guard did not look at.

Re-measured (level 1 unless said; open side p50 / p90 in degrees; the neutral body unless named):

| root | bare | welded |
| --- | --- | --- |
| tail | 37 | 7 / 34 (level 2: 10 / 24) |
| tail, heavy / child / slim_f / tall / short | | 8 / 10 / 7 / 8 / 12 |
| wing (membrane) | 46 | 18 / 34 (level 2: 14 / 29) |
| wing, heavy / child / slim_f; feathered; dragon | | 18 / 13 / 13; 12; 18 |
| horn / ram / spiral | 49 | 28 / 25 / 28 (level 2: 13 / 9) |
| horn of radius 0.05 / 0.09 / 0.13 | 68 / 64 / 50 | 48 / 28 / 29 (0.05 at level 2: 26) |
| lower arms | 80 | 10 / 21 (level 2: 9 / 16) |
| lower arms, heavy / child / slim_f | | 9 / 8 / 8 |
| upper (fan) arms, neutral / heavy / child / tall / short | 56 / 54 / 58 | 14 / 17 / 17 / 14 / 19 |
| extra legs, neutral / child / slim_f | 54-60 | 12 / 15 / 13-15 |

Fan arms before this round: neutral 21 degrees, heavy kept 0.79 of its ring (and the lifted ring alone tipped it into
being refused), child bare. They are still the weakest root: close up the welded skin wrinkles on the medial side of
the neutral body's dome, and the slim body's pair is still left bare (kept 0.59). The cause is the girdle's shape (a
cup laid on the shoulder), not the weld: the round-5 critic's finding on the fan pair stays open.

Tried and dropped: a ring lift that follows the wall's facets and curvature (it did not make the heavy fan arm any
safer), a wider tangent threshold for the fan arms with a reach that comes down as the wall lies down (no notch, but
the weld all but switched off: 32-36 degrees), and moving the last ring only in proportion to its reach where the
reach runs out (it left the sleeve standing over a half-carried ring: folds on the heavy body's lower arms).
The folded wing was tried again with the weld forced on, with and without the tangent rule: refused on the neutral
and the heavy body (0.69 of the ring), so it stays without one; its bare arrival is 54-61 degrees.

On the final round-7 tree: the full suite 2705 passed, 30 skipped; the Tk tests (panel and lab) 183 passed.

Still to do after round 7: the fan pair's and the extra legs' girdle shapes (the slim body's fan arms, the heavy
body's extra legs, legs beside a tail and fan arms beside feathered wings are left bare; welded fan arms still
wrinkle close up), the folded wing, the coarse starry cuff of a thin root at level 1 (a tail under about 0.3, a horn
of 0.05), the facets of the last strip in flat shading at level 1, the cost, and a review by an independent critic.

## Not done (round 6)

* Cost: 0.15-1.7 s per root at level 1 for most roots (0.5-0.7 s for the default tail, wing and lower arms), 2-6 s
  at level 2, and 10 s for a 0.85-radius tail at level 1 (four guard rounds over a large region); the full
  creature (wings, tail, horns, lower arms) takes about 4 s to weld at level 1 in the final render. The plan's budget
  was 0.15 / 0.5 s. The closest-point queries dominate; every guard round is a whole new construction.
* Roots left bare (the table above; since round 7 the child's fan arms weld): the fan arms on the slim body and
  beside feathered wings, the extra legs on the heavy body and beside a tail. Their girdles (the fan pair's cup, the legs' cushion: round 5's
  limb findings) are untouched; the weld cannot blend what is not shaped to be blended.
* A tail beside extra legs: the tail's open-side arrival is 22 degrees against 9 alone (the legs' masses crowd the
  root).
* By design, no weld: the folded wing (the arm lies along the back; forcing the weld on gave a reach of 0.1 after
  six guard rounds and no visible gain), an ossicone, a first-version limb.
* In flat shading at level 1 the last strip shows a row of small facets ("teeth") round a root where the reach is
  small (under a tail, under a wing's arm); level 2 is clean. Small horns have a cuff about one level-1 edge wide.
* The second pass (the membrane placement, the bounded-slope cuts, the guard's hard rounds) was written and judged
  by the session alone, on one-colour, flat-shaded and real-colour renders; no independent critic has reviewed it.
