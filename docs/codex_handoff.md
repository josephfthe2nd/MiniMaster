# Handoff: the feathered wing and its feathers

Written 2026-11-20 for whoever picks this branch up next. It says where the work stands, what the owner
asked for, what was found, and what to do first. Everything it points at is in the repository; nothing
depends on the session that wrote it.

- Branch: `claude/minimaster-stl-design-a60puv` (pushed, tree clean). No pull request exists and the owner
  has not asked for one.
- The wing and feather **code** is as it was at `ba659ad`. The two commits after it change only
  `docs/appendage_references.md`; this handoff adds `tools/wing_check.py`.

## 1. Where things stand

The owner's last three requests, in order:

1. "I think we need a hair/strand system and particle system." This was first misread as two new body
   coverings (`hair` and `scatter` kinds, which exist and work). What the owner meant came next:
2. "I meant actual hair and particle systems, we would be able to better create feathers with a clever
   mesh to keep it efficient." Done: general systems in `minimaster/core/hair.py` and
   `minimaster/core/particles.py`, and feathers grown with the hair system (`minimaster/character/feather.py`),
   wired into the wing as the opt-in parameter `grown`. See `docs/hair_particles.md`.
3. "Those wings still look wrong, feathers too but I don't know how. Get reference images for wings and
   feathers." Done: 75 references gathered and compared with our wing (section 3 below). **Nothing has been
   rebuilt yet.**

The owner was shown eleven sheets (five of references, six comparing our renders with them) and a ranked
list of what is wrong, with a recommendation and this question, which is **still unanswered**:

> Rebuild the wing the way sculptors do it (one fused body with the feathers as shallow steps on it), and
> show one wing built that way before wiring anything in?

Also unanswered from earlier rounds: whether `grown` feathers should become the default (the evidence now
says no, see finding 5), and whether the barbed undersides should default on.

**So the first thing to do is not to rebuild.** Either get the owner's answer, or do the small thing the
recommendation already promises: build ONE wing the proposed way, as an experiment outside the shipped
path, and show it beside the references. The owner cannot always say what is wrong; pictures side by side
are how they decide.

## 2. How the owner works (standing instructions)

- The goal that started all this: appendages must look grown, not stuck on. Skin, bone and muscle
  junctions have to read as anatomy. Keep iterating until it does.
- Judge anything visual from **rendered images**, and show the owner the images. Do not report a visual
  result from numbers alone.
- Show one small thing before building a big thing. Request 1 above cost a whole round because a large
  feature was built on a guess.
- Gather references rather than work from memory, and record them (`docs/appendage_references.md`).
- The owner asked that the work be done by the main agent itself. Ask before spending on parallel
  reviewers or sub-agents.
- Do not open a pull request unless asked. Commit and push to the branch above.
- Treat the owner's name, email and project details as private: never put them in a request to an outside
  service (a User-Agent string, a URL, a form).
- Commit messages in this repository are plain sentences that say what changed and why
  (`git log --oneline` shows the style). Docs are written the same way: full sentences, no shorthand.

## 3. What is wrong with the wing, most important first

Ids like `plume/S01` are in `docs/appendage_references.md` under "Feathered wings and single feathers";
the measured numbers are in its "Feathered wing and feathers" table. "Ours" is the default
`wing:style=feathered` on the default body.

1. **Every feather is thick, so every overlap is a step.** A flat secondary on a 32 mm figure is 1.63 mm
   wide and 0.37 mm thick (0.51 mm at the shaft, 0.30 mm at the edges): a quarter to a third of its width.
   A real vane is under a hundredth of its width, and a real wing reads as one surface on which each
   feather is a faint line (`plume/P01`, `plume/W03`, `plume/W06`). Thinner feathers help but run into the
   print floor: `thickness=0.16` (0.30 mm) passes with the thin warning, `thickness=0.11` (0.20 mm) is
   blocked. So this cannot be fixed by a slider while each feather is printed as its own free piece.
2. **Coverts and the front edge.** Real coverts are a dense field of broad scales of which only the rounded
   tips show (`plume/P03`, `plume/W06`, `plume/W25`); the front edge is one smooth roll. Ours are narrow
   tongues hanging in rows, each a separate thick piece, and a row of pegs along the front.
3. **The root.** On birds the long feathers of the back lie over the wing's root, so wing and back are one
   surface (`plume/W06`, `plume/W26`). Sculptors grow the wing out of the shoulder blades as a thick
   feathered root (`plume/S01`, `plume/S02`). Ours stops short of the back with a squared-off inner corner.
4. **Too many feathers, too small.** A carved wing shows about 30 forms: 6 to 10 long broad feathers and
   three or four rows of scales (`plume/S01`). Ours has 41 long feathers and about 230 small ones, each
   standing free.
5. **Grown barbs are the wrong kind of detail.** A pigeon's 15 cm wing feather has about 600 barbs a side;
   at any distance they read as a smooth sheen with the shaft as one crisp line (`plume/F15`). Ours has
   20 to 25 ridges a side: on screen they read as corduroy, and at 32 mm they stand about 0.08 mm high,
   0.3 mm apart, finer than a printer resolves. The marble angel appears to carry barb hatching, but as very
   shallow incised lines on its large feathers. This pulls against the direction the owner chose last round;
   it is their call, and the recommendation is not to make `grown` the default.
6. **The section is nearly flat.** A wing is thick and rounded along the front and thins to one feather at
   the back (`plume/W30`). Ours, edge-on, is close to a plate with the arm as a low ridge.
7. **Folded.** Real: a compact shield shingled into the back (`plume/W25`, `plume/W26`, `plume/S01`).
   Ours: a lumpy bundle standing off the back with the arm showing at the top.
8. **Smaller points.** Primaries are straight, flat and all alike (real ones curve up and back and taper).
   Feathers lack the bow of the shaft, the loose down where the vane starts and natural splits. Body
   feathers are solid pills where real ones are downy for half their length and show only a rounded tip.

**Already close to real; leave these alone:** feather length against width and their lopsidedness, how
much the flight feathers overlap (0.26 to 0.45 of each secondary shows), the number of separate primaries
and how far they separate, and the wing's outline at the size a small figure is looked at.

## 4. The recommended next step (not yet approved)

Build the feathered wing as **one fused body with the feathers as relief**:

- a wing body with a real section: thick and rounded at the arm, thinning to the trailing edge;
- flight feathers and coverts as shallow overlapping steps on that body (a step along each feather's
  edge and one shaft line), free only where a real wing is free: the outer primaries' ends;
- fewer and broader forms, in the direction of `plume/S01` (the counts are already parameters);
- scapulars carried from the back over the root, so the wing and the back are one surface;
- barbs, if any, as a few shallow lines on the largest feathers, not ridges.

What it must keep, whichever way it is built:

- **Print:** nothing under 0.2 mm, little under 0.3 mm (`tools/wing_check.py report ...`). A fused body
  makes this easier, since the body carries the thickness and the feathers need none of their own. Note
  that the report measures a wing piece by piece today; a wing made of relief on one body should be
  measured as the solid it prints as (see `Kind.fused` and the `fused=` argument of `appendage_report`).
- **The rig:** the wing folds (`--pose wing-1:preset=folded`, controls such as `fold`, `raise_`). Today
  each feather is a rigid piece on its own bone, which is what lets it fold. A single surface has to fold
  too; decide early whether the relief wing is skinned across bones or stays pieces that fuse only when
  spread. Fold it in the first experiment, not after.
- **Saved documents and tests:** the default wing is pinned by tests (275 pieces and 91,880 triangles in
  `tests/test_grown_feathers.py`, `test_print_report.py` and `test_wings.py`) and saved characters carry
  wing parameters. Add the new construction as an opt-in first
  (the way `grown` was), and change the default only when the owner has seen it and agreed.
- **Cost:** the default wing is 275 pieces, 91,880 triangles, built in about 0.16 s. The owner cares
  that feather detail stays cheap ("a clever mesh to keep it efficient").

Suggested order: (a) one spread wing built the new way, rendered with `tools/wing_check.py plain` and on
the figure, beside `plume/P01` and `plume/S01`; (b) the same wing folded; (c) show the owner; (d) only
then parameters, tests, docs and the print report.

## 5. Set-up

- Python 3.10 or later, numpy only: `pip install -e .`
- The body mesh is not committed. Fetch it once (needs PyPI and GitHub):
  `python tools/fetch_makehuman_assets.py`
- Frame and scale: Y up, +Z front, one unit is about 1 dm, the body is 17.5 units tall. A 32 mm figure is
  32 / 17.5 = 1.83 mm a unit. The print floor is 0.2 mm (blocks), and under 0.3 mm is fragile (warns).
- Tests: `python -m pytest tests -q`. At `ba659ad` the suite was 2,742 passed, 29 skipped; it is slow, so
  run the files you touch first: `tests/test_wings.py`, `test_wingrig.py`, `test_wingrig_api.py`,
  `test_wing_root.py`, `test_wing_connect.py`, `test_grown_feathers.py`, `test_kinds.py`,
  `test_print_report.py`, `test_print_appendages.py`, `test_weld.py`.
- `tests/test_character_panel.py` and `tests/test_lab.py` need Python with tkinter and a display (for
  example under `xvfb-run`). One of them (`test_a_sequence_through_the_panel_equals_the_headless_core`)
  is sensitive to timing and has failed once on a busy machine; it passes when run alone.

## 6. Looking at the wing

```
# the wing alone, plain material, from above and edge-on (any builder argument as k=v)
python tools/wing_check.py plain out.png
python tools/wing_check.py plain thin.png thickness=0.16

# the section of the wing's own feathers, in mm on a 32 mm figure
python tools/wing_check.py section

# the print check of an appendage on the default body
python tools/wing_check.py report "wing:style=feathered,thickness=0.16" 32

# on the figure: one row per character ("|"), one column per view (azimuth,elevation)
python tools/surface_view.py out.png "wing:style=feathered@gender=1,muscle=0.55|wing:style=feathered,grown=1@gender=1,muscle=0.55" --at 0,4.6,-0.6 --r 8.6 --size 900x560 180,10 180,75

# the root and a pose, close
python tools/weld_view.py out.png "wing:style=feathered" all 180,8 215,15 --at 0,4.6,-0.6 --r 5.2 --size 900x900 --pose wing-1:preset=folded

# single feathers: today's sheet beside the grown one
python tools/feather_view.py feathers out.png
```

**A trap:** `tools/wing_view.py` paints every feather darker toward its edges so the rows read apart. A
step seen in its pictures may be paint, not shape. An early version of the comparison made that mistake.
Use `wing_check.py plain`, `surface_view.py` or `weld_view.py` to judge geometry.

## 7. Where the code is

| what | where |
|---|---|
| the wing's bones, feathers' roots and tips, poses | `minimaster/character/wings.py`: `WingSkeleton` (its `feathers` list), `make_wing_rig`, the rig's `pose_mesh` and `pose_from_controls` |
| the feathered wing's pieces | `wings.py`: `feathered_pieces` (flight feathers, then `_coverts`), one feather is `_blade` / `_covert_blade`; `_blade_rows` is the outline both feather builders share |
| grown feathers | `minimaster/character/feather.py` (`barbed_blades`, `barb_count`); `wings.py` `_Grown` and `_grow_pieces` batch them |
| strands and their meshes | `minimaster/core/hair.py` (`grow`, `trim`, `tube_mesh`, `sheet_mesh`) |
| particles and instancing | `minimaster/core/particles.py` (`emit_curve`, `emit_surface`, `instance`); nothing shipped uses it yet |
| the wing as a kind, its parameters | `minimaster/character/kinds.py`: `WingKind` (`thickness`, `primaries`, `secondaries`, `tertials`, `scapulars`, `coverts`, `underwing`, `barbs`, `grown`, `underside`, ...) |
| where the wing leaves the back | `minimaster/character/wingroot.py`, `minimaster/character/weld.py`; `docs/anatomy_wing.md`, `docs/weld.md` |
| the print check | `minimaster/character/print_report.py` (`appendage_report`). A wing's pieces are measured one by one; only the kinds marked `fused = True` (hair, scatter) are measured through the solid their overlapping pieces print as (the `fused=` argument) |
| docs | `docs/wings.md` (the wing), `docs/hair_particles.md` (the two systems and grown feathers), `docs/hair_scatter.md` (the two covering kinds) |

## 8. The references

- `docs/appendage_references.md` lists every picture: source page, creator, licence, what it shows, and
  the numbers read from it. The `plume` region is this pass. Licences are public domain, CC0, CC-BY and
  CC-BY-SA only; the twelve book plates are public domain in the United States (marked `PD-US`).
- **The pictures are not in the repository and must not be committed**, and neither may any sheet that
  embeds them (the comparison sheets do). `docs/images/` holds only our own renders. The owner has the
  eleven sheets from the session that made them; they can be rebuilt from the page links.
- How they were found, for another pass: Openverse's image search (`license=cc0,pdm,by,by-sa`), which
  indexes Flickr and museum collections; iNaturalist's observation search limited to free photo licences
  (`photo_license=cc0,cc-by,cc-by-sa`, and the "feather" evidence annotation for found feathers); the
  open-access collections of the Cleveland Museum of Art and the Art Institute of Chicago; and page images
  of a scanned 1906 book at archive.org.
- Wikimedia Commons refused nearly every download from the network that pass ran on. Twelve files the
  searches listed, among them the labelled wing-topography diagrams, are named at the end of the `plume`
  section as **not yet seen**. Fetch and look at them if the network allows; they would fill the one gap
  in the set (labelled diagrams of the feather groups).

## 9. Older open items

These predate the reference pass. The rebuild in section 4 would absorb the first three.

- The folded feathered wing is a lumpy pack; the feathers nearest the root sit poorly.
- No camber across the wing; no down at a feather's base.
- The coverts are laid out by the wing builder, not emitted from the particle system.
- The `hair` and `scatter` kinds still use their own emitter and strand code
  (`minimaster/character/scatter.py`, `strands.py`, `hair.py`), not the systems in `minimaster/core`.
- Known limits of the covering kinds are listed in `docs/hair_scatter.md`.
