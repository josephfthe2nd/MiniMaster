# Dynamic appendages — design (v2, after independent review)

Status: **plan; implementation starts with tracks T1 and T2 below.** Goal: the user adds, removes
and tunes appendages (wings, tails, horns, extra arms, anything later) on a realistic character in
the running app, sees the result, and prints it.

v1 of this document was reviewed by three independent engineers who each measured things in a
scratch copy (architecture/persistence, interactive preview, printing/validation). Their evidence
changed the plan substantially; the corrections are folded in below and the bugs they found in the
*existing* code are listed in section 2.

## 1. What exists today (verified)

| Fact | Consequence |
| --- | --- |
| The realistic character ("Character Lab", MakeHuman base) has **no GUI**; `minimaster character …` is CLI only. The tkinter studio only knows the primitive-shape pipeline; its Parts tab is that pipeline's add/remove/tune. | The wanted "add/remove/tune in the running app" is a **new front end** for the realistic path. |
| `Appendage` is a declarative dataclass but is **not in `.mmchar`**, has **no CLI flags**, and **is not part of the STL path**: `printprep.character_print_shells` returns body, eyes and teeth only. | Wings, tails and horns cannot be saved, requested or printed from outside Python today. |
| Only wings are rigged. Tails/horns are static lofts (~1–7 ms to build); harvested extra arms have no rig. No app code consumes `attach`/`body_pose`/`skin_weights`. | A joint-rig protocol for every kind is cost with no user-visible benefit yet. |
| Geometry is fast: morph apply 4–15 ms (not 0.18 s), wing build 0.05 s (bat) / 0.36 s (angel), pose 0.013–0.034 s, ray pick 6–9 ms, build + gate + STL write < 0.5 s. | Interactivity is not limited by geometry. |
| `render_portrait` costs 1.3 s (bat scene) – 5.6 s (angel scene) per frame, **dominated by a per-triangle Python loop** (`portrait.py:147-182`, 95k `np.meshgrid` calls), not by shading. The tk painter `Viewport` draws 3 fps (bat) / 0.7 fps (angel) on the real meshes. | Preview needs a new renderer, but not necessarily proxies. A vectorised visibility-buffer rasteriser reproduces the old loop pixel-for-pixel (0 differing pixels in 24 renders) at 0.09 s (bat) / 0.35 s (angel): **15×**. |

## 2. Bugs the review found in the existing code (all to be fixed, with tests that fail first)

1. **Placement does not follow the morphs**, contrary to the docstrings. `Appendage` stores a world
   point and re-snaps it to whatever face is nearest on every build. On a tall body the wing
   shoulder lands 7.7 units *below* the scapula; on a child 1.9 above (neutral −1.2). `WING_ROOT`
   is also a world point (wing-local shoulder-to-root distance 4.1 neutral, 9.9 tall, 1.4 child).
2. **The frame is unspecified.** `Appendage.build` returns MakeHuman Y-up; the print shells are Z-up.
   Three "exports through the print gate" tests add Y-up wings to Z-up shells and pass (0% of the
   wing inside the body; wings flat behind the figure) because they only assert watertightness.
3. **`creature_anatomy.apply_to` is additive, in place and non-idempotent** (pectoral 0.85 → 1.00 on
   a second call; body volume +7.5%), with no inverse, so add/remove/undo cannot round-trip.
4. **Mirror is not a side.** There is no way to express a single right wing/horn/arm; two wing specs
   collide on shell and joint names (`wing_l`, `wing:root`); both sides always get the same controls.
5. **The l-/r- chain swap for mirrored harvests is broken** (`'l-finger'` → `'r-fingel'`; 21 of 55
   joint markers mangle).
6. **Harvest** ignores `scale`, `roll` and `up_hint`, keeps the donor arm's pose, never sinks into the
   body (floats 0.08–0.5 mm off it), and its topology depends on the body.
7. **Tails and horns are tangent to the skin** (`offset = 0`): 3.8% / 0.2% of their volume inside.
8. **The spread feathered wing's alula (three feathers) is a floating island** (closest approach
   0.03–0.06 units); the watertight gate cannot see it.
9. **Print size and base are set by the wing tips**, not the body: bat wings drop the body from
   32.0 to 26.9 mm at "medium"; spread wings overhang the base by 11–15 mm. `PrintReport.volume_mm3`
   double-counts overlaps (+18% bat, +82% angel).
10. **Thickness has no floor against scale or print size**: `Appendage.scale` scales thickness linearly
    (scale 0.5 → ~0.17 mm at medium); a tiny-size angel wing is 69% under 0.2 mm and still passes the
    gate. Tail/horn tips are 0.04–0.07 mm. A folded membrane puts 22% of its area under 0.2 mm.
11. **numpy ≥ 1.24 is the declared floor, but `np.add.at` is ~4× slower before 1.25**: the shading
    path costs 3–4× more there.

## 3. Decisions (revised)

**D1. UI home.** A new "Character" tab in the studio (the only GUI). Everything below lives in
`minimaster/character/` and is testable without tkinter; `viewport.py` imports tkinter at import
time, so tk-dependent code is isolated and tested with `importorskip`.

**D2. Pure-data document model.** `AppendageSpec` is a **numpy-free** module (`sliders.py` is
numpy-free and loading a `.mmchar` must stay cheap): `{id, kind, side, params, controls, placement,
enabled, extra}`. `Character` gains `appendages` and an `extra` dict that re-emits unknown top-level
keys. Persistence rules: version 2 is written **only when appendages exist**; version 1 files load
unchanged and an old reader never silently loses data it cannot understand; unknown kinds, params,
controls and styles are kept, warned about and skipped at build time, never deleted; ids are stable
and every shell/joint name derives from the id (`wing-1/primary05`).

**D3. Kind registry (the heavy side).** A `Kind` declares `params: [ParamSpec]` (float|int|choice|bool,
`lo`, `hi`, `default`, `step`, plus `when=(other_key, allowed)` for style-dependent params such as
wing style), dynamic `control_range(name, state)`, `instantiate(spec, ctx) -> [Placed]` (`ctx` carries
base, morphed verts, cage, mirror map; **not** a body-free `build(params, scale)`), placement
presets, an optional `draft_params` dict, and `load(params)`/`paired` for body adaptation. **`Appendage`
stays the engine** (additive change only: an optional stored anchor); the kind id `harvest` stays;
`root_point`/`up_hint`/`roll`/`offset` stay on `Appendage` (pinned by ~50 tests). The pose-control
protocol is just `mesh(controls)`, `control_range`, control names; `attach`/`body_pose`/`skin_weights`
stay an optional export capability that only wings implement.

**D4. Placement follows the body.** Persist the topological binding, not a point:
`placement = {anchor: {face}, side: center|left|right|both, scale, roll, offset, up_hint, aim:
{face | landmark+offset}}`. The left-hand definition is stored once; the right side uses an exact
mirror-face map of the cage (max error 1.9e-15, 0.39 s once per base). Presets resolve to a face and
landmark offsets on the neutral body at instantiation (`WING_ANCHOR`/`WING_ROOT` become body-relative,
as the feathered wing's `root_relative` already is). Click-to-place: `ray_mesh_hit` on the cage
(6–9 ms; `quad = tri % 13378`). v1 points fall back to a body-relative conversion. Tail/horn presets
get a default sink (≈ −0.45 to −0.6 units: 22–28% / 58–70% embedded); harvested arms sink along the
limb axis and are treated as **unprintable until placed body-relative and sunk**.

**D5. One frame seam.** `Character.shells()` returns **Z-up** shells (body, eyes, teeth, appendages),
used by preview and print alike; appendage objects stay in the MakeHuman frame. A post-assembly check
(each appendage root within ε of the body surface) is part of every kind's acceptance; the three
vacuous export tests are replaced.

**D6. Body adaptation is a derived layer.** `effective_sliders = user sliders + Σ consequences of
enabled appendages`, recomputed from scratch, clamped on the sum, **never written back**; only the
toggle and cap are stored. `apply_to` returns the delta and the notes without mutating (kept for the
legacy tools). Tests: save → load → build leaves weights unchanged; removing a spec restores the body
exactly.

**D7. Preview = one image-based numpy renderer.** New tk-free `minimaster/character/raster.py`, a
vectorised visibility-buffer rasteriser (triangle bucketing by bbox class, fixed candidate grids,
z-resolve, id buffer, deferred Gouraud on visible pixels only), shown through a `tk.PhotoImage` in a
Canvas; grid, joints and selection outlines stay ordinary canvas items on top; the id buffer also gives
O(1) picking. It replaces the per-triangle loop **behind `render_portrait`'s signature**, so the CLI
and the doc figures get the 15× too. View-independent shading terms (vertex normals, cavity AO) are
cached per shell. Draft tier (orbit, pose drag, body drag, and shape-parameter drag via the kind's
`draft_params`, e.g. feathered `coverts=False, underwing=False, bones=False`: 350 → 65 ms) uses a
vertex-clustered body proxy (~4k tris, 0.3 ms per morph update) and cluster proxies of appendages
(1–20 ms); the final tier runs on idle in a **cancellable** worker thread (latest-wins slot, generation
counter, `check()` between shells and buckets, frozen spec in, bytes out, main thread polls a queue with
`after(15)`; debounce 150 ms). Display: draft at half resolution when the pane is large. The numpy floor
is either raised to ≥ 1.25 or the shading uses `bincount` (same numbers on both).

**D8. Printability is defined by measurements, not by the gate.** The watertight gate is topological
and blind to placement, connection, scale and thickness. Per-appendage report (computed per spec and
cached in units, rescaled by mm/unit; run on idle and at export; per-piece meshes via a new
`PlacedWing.piece_meshes` that handles mirror winding in one place):
- **Blocking:** non-watertight / inside-out / degenerate shells (report all, not the first); any shell
  group above a volume threshold not overlap-connected to the body (a detached tail or arm blocks; tiny
  floating feathers warn); root embedding of the named root pieces below threshold (tail 3.8% and horn
  0.2% block; wing arm ~21–25% passes); more than ~25–40% of an appendage's area under 0.2 mm at the
  chosen size; tube tips (analytic) under the printable minimum.
- **Warning:** posed-mesh thickness (folded membranes), appendage–appendage overlap, base overhang.
- Thickness floors become a function of print scale: `thickness = max(thickness, floor_mm /
  (mm_per_unit · appendage_scale))` applied at export with the report naming what changed.
- **Size contract (decision for the user): "medium / 32 mm" means body height** (tabletop convention);
  a winged figure is then ~38 mm tall and spread wings overhang a 25 mm base by 11–15 mm. Height, XY
  centring and base size come from `reference=` body shells in `assemble_character`; the base can
  auto-widen or warn. `PrintReport.volume_mm3` is renamed "sum of shell volumes" or estimates the union.
- The "slicers union overlapping shells" premise is documented as an assumption and verified with a real
  slicer before any print claim (51% of the angel wing's union volume is covered by two or more feather
  shells; an even-odd slicer would lose ~35%).

**D9. Undo/redo.** Snapshot the document with the existing JSON-string `UndoStack` (a Character with one
tail spec is 497 bytes, 0.012 ms to serialise).

## 4. Tracks and acceptance criteria

| Track | Work | Done when |
| --- | --- | --- |
| **T1 — document, placement, print (headless)** | D2–D6 and D8: pure-data specs + `.mmchar` v2, registry + `ParamSpec`, face-id placement with `side` and the mirror-face map, id-derived names, the Z-up `Character.shells()` seam, anatomy overlay, `appendage_print_shells` + `assemble_character(reference=)`, per-appendage report and blocking rules, CLI (`--add KIND[:k=v,…]`, `--remove ID`, `--list-kinds`, `--pose ID:k=v`), tail/horn sink presets, fix of the l-/r- swap, alula weld fix. | `minimaster character --add wing:style=feathered --add tail -o out.stl` prints a gate-clean STL **whose body height equals the preset and whose appendages are connected to the body**; wings stay on the scapula on tall/short/child/heavy bodies; two wing specs coexist; a v1 → v2 → reload round trip preserves an unknown kind; save → load leaves the body unchanged; all current tests (1036) pass; every new test is shown to fail without its fix. |
| **T2 — renderer** | D7 steps 1–2: vectorised visibility-buffer `raster.py` behind `render_portrait`, per-shell shading cache, `check()` cancellation hooks, `bincount` normals/AO, id-buffer picking. | Pixel-equal to the old loop in float64 on a golden set (old loop kept as the reference on small scenes); bat/angel scenes ≥ 10× faster; picking agrees with `ray_mesh_hit` to one face. |
| **T3 — kinds** | Tail, horn variants, wing registry adapter; harvested arms made body-relative, sunk, rig via `Armature` + `bodymesh.skin_weights` (re-weighted per re-harvest, `scale`/`roll` honoured or hidden). | Property test at every `ParamSpec` lo/hi (watertight); mirrored harvest passes over all 55 joint markers. |
| **T4 — Character tab** | D1, D7 steps 3–4, D9: body sliders, appendage list (Add by kind/preset, Remove, Duplicate), property grid generated from `ParamSpec`, controls, click-to-place, draft/final preview with cancellable worker, undo/redo, export with the printability report. | Add a wing and a tail, tune, drag, undo, save, reload and export entirely from the tab; the same document operations are driven headless in a test; draft ≥ 20 fps at 900×800 including `PhotoImage`, shape-param drag ≥ 10 fps, final < 1 s cold at 900×800 for the angel, cancel latency ≤ 100 ms. |

Order: T1 and T2 are independent (different files) and run in parallel; T3 follows T1; T4 needs all
three.

## 5. Open items and risks

- **Print claims need a real slicer** (union assumption); until then the report says "assumes the
  slicer unions overlapping shells".
- Appendage–appendage collisions are warned about, not solved.
- A single right wing must be mirror-exact against the pair's right wing (test).
- Clothes are a separate subsystem; the registry must not assume an appendage is a limb.
- What was not measured: native Windows/macOS Tk, HiDPI, the dragon style's proxy timings (same class
  as the bat), numpy 1.24 for the Tk paths.
