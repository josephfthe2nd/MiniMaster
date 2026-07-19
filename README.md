# MiniMaster

Design, rig, pose, and export **3D-printable tabletop miniatures** — a
low-poly modeling and rigging studio in pure Python. The third program in the
Forge family (alongside DiceForge and DungeonForge): where those make dice and
dungeon terrain, MiniMaster makes the minis that stand on them, working both
as a character creator and an enemy creator.

Build a figure from stretched primitive shapes, place bones and joints for
articulation, pose the skeleton, and export a watertight binary STL scaled to
tabletop size — chunky, faceted minis that look like little carvings and
print cleanly without fussy detail.

| Human Fighter | Dwarf | Goblin | Orc | Skeleton |
| --- | --- | --- | --- | --- |
| ![human fighter](docs/gallery/human_fighter.png) | ![dwarf](docs/gallery/dwarf.png) | ![goblin](docs/gallery/goblin.png) | ![orc](docs/gallery/orc.png) | ![skeleton](docs/gallery/skeleton.png) |

One model, four of its shipped poses (idle / walk / attack / guard) — every
figure is articulated by its bones, so posing never breaks the mesh:

![poses](docs/gallery/poses.png)

## Install & run

Requires Python ≥ 3.10 with tkinter (Linux: `apt install python3-tk`). The
only third-party dependency is numpy.

```bash
pip install -e .
minimaster                 # launch the studio
# or without installing:
python -m minimaster
```

## The studio

The window is a 3D viewport plus four mode tabs:

- **Model** — add primitive shapes (box, wedge, cylinder/cone, capsule,
  icosphere, torus), then stretch and place them: position / rotation /
  per-axis scale, primitive parameters (segment counts, taper, subdivisions),
  color, duplicate, and one-click **Mirror X** for symmetric limbs.
- **Rig** — build the armature: add child joints, drag their positions,
  rename, delete. Bind each shape to a bone by hand or hit **Auto-bind**
  (nearest bone wins). A joint's rotation moves everything below it.
- **Pose** — create named poses, click a joint, drag the X/Y/Z rotation
  sliders and watch the figure move in real time. Poses are stored in the
  project, so one mini can ship with idle, walk, attack, guard…
- **Export** — pick a size category (tiny 15 mm → huge 60 mm, or a custom
  height), a base (round / square / none), and a pose; check printability
  (triangle count, dimensions, watertightness, resin volume); export STL or a
  PNG turnaround.

Viewport controls: **drag** orbit · **middle/right-drag** pan · **wheel**
zoom · **click** select · **g** grab (move the selected shape in the view
plane, click to confirm, Esc to cancel) · **Ctrl+D** duplicate · **m**
mirror · **x** delete · **f** frame · **Ctrl+Z / Ctrl+Y** undo/redo.

## Part library (Spore-style)

The **Parts** tab is a snap-on part palette: pick a part, click anywhere on
the mini, and it lands on that surface point oriented outward along the
surface normal, auto-bound to the bone of the body part you clicked. Placed
parts edit as a unit — grab-move, spin about their outward axis, uniform
scale, **Mirror** for symmetric pairs (wings, horns), ungroup, delete — all
undoable.

Parts can be **posable**: a part may carry its own joint chain (the shipped
tail does), which grafts onto the character's skeleton on placement and shows
up in the Pose tab like any other joints. Removing the part removes its
joints; every pose still exports watertight.

Twelve parts ship in the box (eyes, horns, ears, claws, spikes, wings, a
posable tail, and sword/shield/axe/club/dagger), and the library is yours to
grow: model something in the Model tab (origin = attachment point, +Z =
outward), then **Save selection as part…** writes it to `~/.minimaster/parts`
with a thumbnail, ready to snap onto any future mini. Placed parts flatten
into the project, so `.mmp` files stay self-contained and shareable. See
`examples/goblin_gargoyle.mmp` for a goblin with snapped-on horns, a third
eye, mirrored wings, and a pose-curled tail.

## Starter templates

`File → New from template` (or `minimaster new <name> -o my.mmp`) opens a
fully rigged, fully posed figure to reshape instead of a blank scene:
`human_fighter`, `dwarf`, `goblin`, `orc`, `skeleton`. All five come from one
parameterized humanoid (21 joints) with proportion multipliers and feature
add-ons (beard, ears, tusks, ribcage, gear), built by
`tools/make_templates.py` — add your own factory there and re-run it to grow
the roster.

## Headless CLI

Everything the Export tab does works without a display:

```bash
minimaster templates                                  # list starter templates
minimaster new orc -o my_orc.mmp                      # start from a template
minimaster export my_orc.mmp -o orc.stl --size large --pose attack
minimaster export my_orc.mmp -o orc.stl --height 40 --no-base
minimaster preview my_orc.mmp -o orc.png --azimuth 335
minimaster validate my_orc.mmp                        # watertightness gate
```

## How it prints

Every shape is an independently **watertight, outward-oriented shell**; the
exported STL is the union of overlapping shells, which every slicer merges at
slice time. Rigid per-bone binding means posing transforms whole shapes —
joints stay covered by design (limb segments overlap ball pads), so any pose
exports watertight. The `validate` command and the test suite enforce this:
every edge shared by exactly two faces, consistent winding, positive volume.

Figures are modeled at roughly 28 mm-heroic scale and rescaled on export;
size categories put the figure at 15 / 24 / 32 / 45 / 60 mm total height on a
25 mm (or custom) base.

## Project files

`.mmp` files are plain JSON: shapes (kind + params + transform + bone +
color), joints (name/parent/position), named poses (joint → XYZ euler
degrees), active pose, and base spec. They diff cleanly and are easy to
generate from scripts — the starter templates are built exactly that way.

## Architecture

```
minimaster/
  core/        geometry kernel: math3d, Mesh (+integrity checks),
               watertight primitive builders, armature FK, binary STL I/O
  scene.py     the document: shapes + armature + poses + undo snapshots
  export.py    pose → scale → base → merged STL (shared by GUI, CLI, tests)
  render.py    headless z-buffer flat-shade renderer → PNG (no imaging deps)
  templates/   shipped .mmp starter scenes (regenerable)
  viewport.py  tk Canvas painter's-algorithm viewport with picking
  panels.py    Model / Rig / Pose / Export tabs
  app.py       the studio window: menus, undo, mode-aware refresh
```

## Development

```bash
pip install -e .[dev]
python -m pytest                       # 139 tests, all headless
python tools/make_templates.py         # regenerate starter templates
xvfb-run -a python tools/gui_smoke_test.py   # drive the GUI end-to-end
```

`examples/` holds a ready-to-print goblin scout (`.mmp` + STL) and an
attack-posed human fighter STL straight from the templates.
