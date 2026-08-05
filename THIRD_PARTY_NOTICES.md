# Third-party notices

MiniMaster's own source is original work. This file records everything it
depends on or optionally consumes, and the licence each is under.

> This is a factual record of licence terms, not legal advice. If MiniMaster is
> going to be distributed or sold, have a lawyer confirm the conclusions below.

## Runtime dependency

| Component | Licence | How it is used |
| --- | --- | --- |
| numpy | BSD-3-Clause | imported as a library |

## Optional external tool

| Component | Licence | How it is used |
| --- | --- | --- |
| Blender | GPL-2.0-or-later | invoked as a **separate process** (`blender --background`) via `minimaster/blender.py`. No Blender code is linked, copied, or redistributed; MiniMaster runs fully without it. |

## Optional CC0 assets (Character Lab "realistic" path)

`minimaster/character/mhbase.py` can load two data files from the MakeHuman
Community project. **They are not committed to this repository** — they are
downloaded on demand by `tools/fetch_makehuman_assets.py` into
`assets/makehuman/`, which is gitignored.

| File | Contents | Licence |
| --- | --- | --- |
| `base.obj` | base mesh, 19,158 verts / 18,486 quads | CC0 1.0 Universal |
| `targets.npz` | 1,280 morph targets | CC0 1.0 Universal |

MakeHuman deliberately splits its licensing into two documents:

* **`LICENSE.CODE.md` — AGPL-3.0** covers "files that contain program logic…
  python files, bat scripts, shell scripts and glsl shaders".
  **None of this code is used, copied, vendored, or redistributed by
  MiniMaster.**
* **`LICENSE.ASSETS.md` — CC0 1.0** covers the data, explicitly listing
  "The base mesh and proxies" and "Targets and modifiers". These are the only
  MakeHuman files MiniMaster touches.

The CC0 dedication is also recorded *inside* the data: the licence record
embedded in `targets.npz` reads `author = MakeHuman Team`, `license = CC0`,
`copyright (c) www.makehumancommunity.org 2001-2020`.

### On generated output

MakeHuman's `LICENSE.md` §D states the project "makes no claim whatsoever" over
output including "Exports to files (FBX, OBJ, DAE, MHX2…)", "Graphical data
generated via scripting or plugins", "Renderings" and "Saved model files", and
that "As the assets have been released under CC0, there is no limitation on
what you can do with this combined output. … We regard these things as your
data, which is yours to handle as you see fit."

Meshes, STLs, and renders produced by MiniMaster are therefore unencumbered,
including for commercial use.

### How the implementation was written

`mhbase.py` is original code: an OBJ parser, a sparse-delta `.npz` reader, and
a macro-blending function written against the observed **data formats**. The
macro system's *design* (interpolating a slider across the corners of a target
hypercube) follows MakeHuman's approach — an idea/method, which copyright does
not protect — but no MakeHuman source was copied or consulted line-by-line to
produce it.

### Caveats worth knowing

* CC0 waives copyright; it does **not** grant trademark rights. Do not imply
  endorsement by or affiliation with the MakeHuman project.
* CC0 requires no attribution. This notice is kept as good practice.
* Only the assets **bundled with MakeHuman** are CC0. Third-party assets from
  the community asset repository carry their own licences and are not used
  here.
* Users who print or sell figures are responsible for any content they add.

## Reference material

`docs/anatomy.md`, `docs/realism_research.md` and the design specs summarise
publicly documented anatomical canon and modelling technique gathered from
research. Facts, measurements, and techniques are not copyrightable; no
substantial text was reproduced from any source.
