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
  that comes near another part of the skin: the facing cheek across the natal cleft, `CLEAR_*`), smoothed without
  ever rising, fading with the distance from the root's exit point, zero where the closest wall point is beyond the
  root (`wall_clip`: a drooping tail is not welded to the thighs).
* The sets: the cap (under the wall; holes filled), the last ring (`top`, thinned so no triangle has three
  corners on it), the rim (fixed), the sleeve.
* The last ring climbs the wall along the root's generator (its axis laid on the wall) to height `k`, at most
  `CLIMB_MAX * k` of travel: where the wall runs along the skin the height it reaches is its reach (a crease), and
  the sleeve next to it takes that reach.
* Every sleeve vertex gets a share `u` of the way from the last ring to the rim (graph distances, smoothed,
  `grade` 1.5 packs the rings toward the wall) and lands where its own path meets the rim (each vertex its own
  aspect `R`); it is placed analytically on its profile, laid onto the zero set by Newton steps, its `h` restored
  along the surface, and evened along its own ring only.
* The cap's first ring sits just under its neighbours on the last ring; the rest of the cap is a harmonic membrane
  spanning it (conjugate gradients); the whole plug is pushed under the wall (signed distance, then ray parity
  against the wall's shells).
* The guard: after every construction the exact triangle-crossing count of the faces around the root (and around
  every earlier root's weld) is compared with the bare skin's, and faces turned over (normal against the blend's
  gradient) are found; the reach is cut there by `GUARD_SHRINK` and the root solved again (`guard` rounds). If the
  skin would still cross itself more than the junction left it, or a triangle would be degenerate, the root is left
  as the junction made it and a warning says so: torn skin never ships.
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
* What is welded: a tail of base radius up to `TAIL_WELD_MAX_R` (1.0), a keratin horn, a wing's arm (ramping in as
  the arm rises off the back; the folded preset has none: the arm lies along the back in a crease), a girdle limb
  (lower arms, upper arms, extra legs). Not: an ossicone (its foot is the skin's own fillet), a first-version limb, a
  tail thicker than 1.0.

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

| kind | level | bare (all) | welded (open side) | welded (all) | s / root |
| --- | --- | --- | --- | --- | --- |
| tail, neutral | 1 | 70 / 82 | 20 / 46 | 33 / 64 | 1.9 |
| tail, neutral | 2 | 70 / 79 | 19 / 66 | 33 / 72 | 7.2 |
| tail, heavy / child / slim_f | 1 | 63 / 59 / 51 | 26 / 19 / 20 | 39 / 30 / 29 | 1-2 |
| wing (membrane), neutral | 1 / 2 | 70 / 73 | 14 / 13 | 18 / 15 | 0.6 / 6 |
| wing, heavy / child / slim_f | 1 | 58 / 74 / 88 | 37 / 20 / 16 | 38 / 18 / 21 | 0.5-1.5 |
| dragon / feathered, neutral | 1 | 71 / 47 | 17 / 24 | 19 / 27 | 0.3-1.4 |
| horn / ram, neutral | 1 | 50 / 49 | 31 / 34 | 31 / 34 | 0.2 / 1.7 |
| horn / ram, neutral | 2 | 49 / 56 | 20 / 15 | 20 / 16 | 1 / 10 |
| lower arms, neutral | 1 / 2 | 76 / 75 | 7 / 7 | 13 / 7 | 0.9 / 4 |
| lower arms, heavy / child / slim_f | 1 | 76-78 / 75 / 76 | 12 / 8 / 8 | 20-25 / 17 / 16 | 0.4-3 |
| extra legs, neutral | 1 | 54-56 | 8 / 25 | 9 / 28 | 1.4-2.6 |
| tail / wing / lower arms / extra legs, neutral | 0 | 75 / 66 / 71 / 56 | 27 / 25 / 23 / 14-25 | 46 / 33 / 35 / 16-30 | 0.1-0.3 |

No welded root crosses itself more than the bare skin did and no plug vertex leaves its wall (exact counts, in the
round-6 report).

## Not done (round 6)

* The plan's targets are NOT all met: the tail's open-side p50 is ~19-26 degrees at level 1 (target 12) and ~19 at
  level 2 (target 8); the lower arms meet them (7-12 at level 1, 7 at level 2).
* Cost: 0.2-3 s per root at level 1 and 1-11 s at level 2 on the shared box (the plan's budget: 0.15 / 0.5 s). The
  closest-point queries and the guard's re-solves dominate (most roots need 1-3 guard rounds).
* The guard leaves these roots unwelded (the bare junction, with a warning in the command line and a note in the
  lab): the upper (fan) arms on the neutral, heavy and child bodies (welded on slim_f); the extra legs on the heavy
  and child bodies at level 1 and on the neutral body at level 2; the 0.85-radius tail; the half-folded wing on the
  child; horns at level 0. By design, none: the folded wing, an ossicone, a tail thicker than 1.0.
* The tail's lower flanks (4 and 8 o'clock, where the reach falls from the back's to the crease's) show thin facet
  slivers in flat shading at level 1, and pleats on the upper flank at level 2; small horns show a radial fan of
  facets round the root in flat shading; a horn's cuff is about one level-1 edge wide.
