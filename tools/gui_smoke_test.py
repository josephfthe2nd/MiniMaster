"""GUI smoke test: boots the studio under a real (or virtual) display and
drives it through every mode.

Run headless:  xvfb-run -a python3 tools/gui_smoke_test.py

Exercises: template load, tab switching, shape add/select/transform, grab,
undo, joint selection, pose slider, export STL + integrity check, and window
screenshots (written next to --out).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minimaster.app import MiniMasterApp  # noqa: E402
from minimaster.core.stl import read_stl  # noqa: E402
from minimaster.export import export_stl  # noqa: E402
from minimaster.templates import load_template  # noqa: E402


def screenshot(app, path: Path) -> bool:
    try:
        from PIL import ImageGrab

        app.update()
        x, y = app.winfo_rootx(), app.winfo_rooty()
        w, h = app.winfo_width(), app.winfo_height()
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h), xdisplay=None)
        img.save(path)
        return True
    except Exception as exc:  # pillow missing or no XCB — non-fatal
        print(f"  (screenshot skipped: {exc})")
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, help="directory for screenshots")
    args = parser.parse_args()
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="mm-smoke-"))
    out.mkdir(parents=True, exist_ok=True)

    checks = 0

    def ok(cond, what):
        nonlocal checks
        if not cond:
            raise SystemExit(f"FAIL: {what}")
        checks += 1
        print(f"  ok: {what}")

    print("booting app with human_fighter template...")
    app = MiniMasterApp(load_template("human_fighter"))
    app.update_idletasks()
    app.update()
    ok(app.winfo_exists(), "window exists")
    ok(len(app.viewport.find_all()) > 200, "viewport drew polygons")
    screenshot(app, out / "01_model.png")

    print("model tab: add + select + move a shape...")
    n_before = len(app.scene.shapes)
    app.add_shape("box")
    ok(len(app.scene.shapes) == n_before + 1, "shape added")
    ok(app.selected_shape == app.scene.shapes[-1].name, "new shape selected")
    shape = app.selected_shape_obj()
    pos_before = shape.position.copy()
    app.start_grab()
    ok(app.viewport.grabbing, "grab started")
    app.viewport._motion_free(type("E", (), {"x": 100, "y": 100})())
    app.viewport._motion_free(type("E", (), {"x": 140, "y": 100})())
    app.viewport._confirm_grab()
    ok(not app.viewport.grabbing, "grab confirmed")
    moved = app.selected_shape_obj().position
    ok(abs(moved[0] - pos_before[0]) > 0.01 or abs(moved[2] - pos_before[2]) > 0.01,
       "grab moved the shape")
    app.undo()
    app.undo()
    ok(len(app.scene.shapes) == n_before, "undo removed the added shape")
    app.update()

    print("rig tab: joint selection...")
    app.notebook.select(1)
    app.update()
    app.select_joint("elbow_l")
    ok(app.rig_panel.tree.selection() == ("elbow_l",), "joint selected in tree")
    screenshot(app, out / "02_rig.png")

    print("pose tab: slider drives the pose...")
    app.notebook.select(2)
    app.update()
    ok(app.scene.active_pose == "idle", "template's idle pose active")
    app.select_joint("shoulder_r")
    app.pose_panel.scale_vars[0].set(-120.0)
    app.pose_panel._slider_moved(0)
    ok(app.scene.poses["idle"]["shoulder_r"][0] == -120.0, "slider wrote pose")
    app.update()
    screenshot(app, out / "03_pose.png")

    print("export tab: STL round trip...")
    app.notebook.select(3)
    app.update()
    opts = app.export_panel.export_options()
    stl_path = out / "smoke.stl"
    report = export_stl(app.scene, stl_path, **opts)
    ok(report["watertight"], "export watertight")
    mesh = read_stl(stl_path)
    ok(mesh.integrity_report()["watertight"], "re-read STL watertight")
    app.export_panel._check()
    ok("watertight" in app.export_panel.stats.cget("text"), "printability check ran")
    screenshot(app, out / "04_export.png")

    app.destroy()
    print(f"\nsmoke test passed ({checks} checks); artifacts in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
