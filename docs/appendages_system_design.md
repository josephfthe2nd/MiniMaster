# Dynamic appendages — design

Status: **proposal, no code yet.** Goal: the user adds, removes and tunes appendages
(wings, tails, horns, extra arms, anything later) on a realistic character in the running
app, sees the result, and prints it.

## 1. What exists today (verified in the repo, not assumed)

| Fact | Consequence |
| --- | --- |
| The realistic character ("Character Lab", MakeHuman base) has **no GUI**. `minimaster character …` is a CLI command. The tkinter studio (`app.py`, `panels.py`, `viewport.py`) only knows the primitive-shape pipeline (Model / Rig / Pose / Parts / Export tabs). | "Add / remove / tune in the running app" means a **new interactive front end** for the realistic path. The studio's **Parts** tab already does add/remove/tune, but for primitive minis, not for this body. |
| `Appendage` (`appendages.py`) is a declarative dataclass (kind, anchor, mirror, params…). It is **not stored in `.mmchar`** (`Character` has macro / sliders / asymmetry / materials only) and the CLI has **no appendage flags**. | An appendage cannot be saved, loaded or requested from outside Python. |
| `printprep.character_print_shells` returns **body, eyes and teeth only**. | **Wings cannot be exported to STL through the CLI or any app path today.** They exist as an API and as renders (`tools/wing_demo.py`). This must be fixed before any UI is worth building. |
| Only wings are rigged (`wingrig.PlacedWing`: pose controls, host-armature export). Tails and horns are static lofts (`make_tail`, `make_horn`); extra arms are "harvested" from the body with no rig. | The rig API has to be generalised, not copied. |
| `creature_anatomy.apply_to` already folds an appendage set into body slider adjustments (broader back for wings, etc.). | Body consequences exist; the UI only needs to expose them. |
| Measured on this machine (4 cores, numpy): morph apply 0.18 s; wing build 0.07 s (bat) / 0.34 s (angel); pose 0.02 s. | **Geometry is interactive-fast.** |
| Measured: `portrait.render_portrait` 3–9 s per frame (bat 3 s, angel 8 s); flat z-buffer `render.render_meshes` 2.7–6.8 s at 320×480; the tk painter's-algorithm `Viewport` is built for hundreds of polygons. Body 26.7k tris, bat 16.5k, angel 171k. | **Preview, not geometry, is the bottleneck.** A live viewport needs level-of-detail proxies; nothing in the repo can draw these meshes at interactive rates. |

## 2. Decisions

**D1. The UI is a new "Character" tab in the existing studio**, because it is the only GUI.
All logic lives in `minimaster/character/` and is testable headless; the tab is thin.

**D2. One appendage contract (a kind registry).** Each kind declares:
- `params`: a list of `ParamSpec(key, label, kind=float|int|choice|bool, lo, hi, default, step)`.
  This single description drives validation, the auto-generated property grid, CLI flags and
  file I/O, so adding a kind never needs new UI code.
- `build(params, scale) -> Piece`, where a piece is shells plus, optionally, a rig.
- `placement` defaults (anchor, hint, roll, offset, mirror, root point) and named
  **placement presets** ("shoulder blades", "tail base", "temples", "shoulder socket").
- `controls`: pose controls with their limits (wings: today's `PoseControls`; tail: curl, sway,
  lift; horn: none or a curl parameter).
- `preview(lod)`: a cheap proxy mesh (see D6).
- `load` for `creature_anatomy` (replaces the hard-coded size heuristic).

`PlacedWing` becomes the first implementation of a `PlacedAppendage` protocol
(`pose(**controls) -> Mesh`, `control_range(name)`, `joints`, `attach(...)`, `body_pose(...)`).
Wings must not regress: all current wing tests stay green and the wing API stays source
compatible.

**D3. Document model and persistence.** `Character` gains
`appendages: list[AppendageSpec]` (id, kind, params, placement, mirror, pose controls, enabled).
`.mmchar` goes to version 2; version 1 files load unchanged. CLI: `--add KIND[:k=v,…]`,
`--remove ID`, `--list-kinds`, `--pose ID:k=v`. Specs are plain JSON. Unknown kinds or params
load with a warning instead of failing, the way unknown sliders already do.

**D4. Placement.** Click the body in the viewport: the existing ray cast gives a face, then
`SurfaceAnchor.from_point` binds the appendage to it, so it follows every morph. Placement
presets cover the common cases; offset, roll, scale and mirror are tunable. Wings keep their
constants as the "shoulder blades" preset.

**D5. Rigs and poses.** Wings as today. Tail: a chain of joints (curl, sway, lift) with the same
body-space `attach` / `skin_weights` contract. Horns: static, with curl and twist as build
parameters. Extra arms: harvested limb (re-harvested when the body changes), rigged to its
own chain; this is the riskiest kind and comes last.

**D6. Preview strategy (the main technical risk).**
- Keep two tiers. **Draft** (while a slider moves or the camera orbits): proxies, a handful of
  thousand polygons, flat shaded. **Final** (on idle, in a worker thread): the existing
  `render_portrait`, shown as an image, cancelled when a new edit arrives.
- Proxies: the body at cage level or vertex-clustered to ~3–5k tris; per-kind proxies. A
  feathered wing previews as one quad per feather plus the arm (about 600 tris instead of
  171k); a membrane previews as its grid at reduced resolution.
- Cache by key: body weights → morphed vertices; spec hash → built appendage; (spec, controls)
  → posed mesh. A slider drag on an appendage then rebuilds only that appendage.
- Phase 0 is a spike that measures this before any UI is written.

**D7. Body consequences.** `creature_anatomy` runs automatically with a visible on/off toggle
and a cap, and the notes it already produces ("broadened back, thinner thighs") are shown in
the panel. Manual slider edits always win.

**D8. Printability gate, per appendage.** Reuse the per-shell watertight gate and add checks
for weld depth (inside-the-torso fraction at the root), minimum feature thickness at the
chosen print scale, and a warning when two appendages overlap. A failed check blocks export
with a specific message; warnings are shown but do not block. The report reads "stowed wings
are display poses; print wings spread".

**D9. Undo/redo.** Snapshot the document (macro, sliders, appendages) per edit; the studio
already has undo/redo for the primitive path and the pattern is reused.

## 3. Phases and acceptance criteria

| Phase | Work | Done when |
| --- | --- | --- |
| **P0 spike** | Measure draft previews: a decimated body plus per-kind proxies, drawn by the existing viewport and by a numpy flat rasteriser at 320×480. | Orbit and slider drag reach ≥ 10 fps for body + two wings, or the design changes. Numbers go in this doc. |
| **P1 headless foundation** | Kind registry and `ParamSpec`; wings, tail, horn adapters; `AppendageSpec` in `Character`; `.mmchar` v2; CLI flags; **appendages included in `character_print_shells` / `assemble_character`**; per-appendage printability report. | `minimaster character --add wing:style=feathered --add tail -o out.stl` prints a gate-clean STL; v1 files still load; all 1036 current tests pass; each new piece has tests that fail without it. |
| **P2 rigs** | Tail chain rig with pose controls and limits; horn variants; common `PlacedAppendage` protocol with the host-armature path. | Tail poses are watertight and bone-length preserving; mirrored pairs exact; documented limits like `docs/wings.md`. |
| **P3 Character tab** | Body macro sliders; appendage list with Add (kind or preset) / Remove / Duplicate; a property grid generated from `ParamSpec`; pose sliders from `controls`; click-to-place; draft/final preview; undo/redo; export with the printability report. | Add a wing and a tail, tune them, drag, undo, save, reload and export, entirely from the tab, with a headless test driving the same document operations. |
| **P4 arms and polish** | Harvested extra arms with a rig; presets; docs and gallery. | A four-armed character exports and re-harvests after a body slider change. |

## 4. Risks and open questions

- **Preview speed** (D6) can force a different viewport (an image-based one) instead of
  extending the tk painter. P0 decides.
- **Appendage collisions** (wing against arm against tail) are not modelled; the plan is to
  warn, not to solve them.
- **Sided parameters and mirrors** must stay exact mirrors under every control (the wing work
  proved this is easy to break).
- **File format growth:** version the spec, and keep unknown-key tolerance.
- **Kind-specific assumptions in `Appendage`** (`root_point`, wing-only fields) must move into
  the wing kind's placement preset so other kinds are not burdened with them.
- **Scope creep toward clothes:** the registry should not assume an appendage is a limb, but
  clothes (a fitted, body-following shell) are a separate subsystem and not part of this plan.
