"""GUI smoke test and timings of the Character tab: boots the studio under a real (or virtual)
display, opens the tab and drives it like a person would.

Run headless:  xvfb-run -a python3 tools/character_smoke.py [--out DIR] [--measure]

Checks: the tab swaps the left pane and back, adds a wing and a tail, tunes parameters and poses
with the sliders, places the tail by clicking the body, undoes and redoes, saves and reloads a
``.mmchar``, exports an STL with the printability report shown in the panel. Screenshots go next to
``--out`` (PIL ``ImageGrab``).

``--measure`` adds the acceptance timings that need a display: the draft tier INCLUDING
``PhotoImage`` (orbit, pose drag, body drag, shape-parameter drag), the final render of the angel
reaching the screen, and the gap of a 5 ms Tk timer while the worker thread renders.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minimaster.app import MiniMasterApp  # noqa: E402
from minimaster.character import mhbase  # noqa: E402
from minimaster.core.stl import read_stl  # noqa: E402
from minimaster.templates import load_template  # noqa: E402


def screenshot(app, path: Path) -> bool:
    try:
        from PIL import ImageGrab

        app.update()
        x, y = app.winfo_rootx(), app.winfo_rooty()
        ImageGrab.grab(bbox=(x, y, x + app.winfo_width(), y + app.winfo_height()),
                       xdisplay=None).save(path)
        return True
    except Exception as exc:  # pillow missing or no display grab: non-fatal
        print(f"  (screenshot skipped: {exc})")
        return False


def pump(app, seconds: float = 0.0, until=None, timeout: float = 60.0) -> bool:
    """Run the event loop for ``seconds``, or until ``until()`` holds (the real UI does the same
    through its own ``after`` poll)."""
    end = time.monotonic() + (timeout if until else seconds)
    while time.monotonic() < end:
        app.update()
        if until is not None and until():
            return True
        time.sleep(0.004)
    return until is None


def settle(app, panel) -> None:
    """Wait for the final render of the last edit to be on screen."""
    pump(app, until=lambda: panel.lab.scheduler.idle and panel.view.frame is not None
         and panel.view.frame.tier == "final")


def drag(app, row, value: float) -> None:
    """Press, move and release a slider the way the pointer does (one undo step)."""
    row.scale.event_generate("<ButtonPress-1>", x=1, y=1)
    row.scale.set(value)
    app.update()
    row.scale.event_generate("<ButtonRelease-1>", x=1, y=1)
    app.update()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, help="directory for screenshots")
    parser.add_argument("--measure", action="store_true", help="also run the timings")
    parser.add_argument("--docs-shot", default=None, metavar="PNG",
                        help="write the screenshot used in docs/character_tab.md and stop")
    args = parser.parse_args()
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="mm-char-"))
    out.mkdir(parents=True, exist_ok=True)
    if not mhbase.available():
        print("the CC0 MakeHuman assets are not installed: nothing to test")
        return 0

    checks = 0

    def ok(cond, what):
        nonlocal checks
        if not cond:
            raise SystemExit(f"FAIL: {what}")
        checks += 1
        print(f"  ok: {what}")

    print("booting the studio ...")
    app = MiniMasterApp(load_template("human_fighter"))
    app.geometry("1260x830")
    app.update()
    panel = app.character_panel
    if args.docs_shot:
        docs_shot(app, panel, Path(args.docs_shot))
        app.destroy()
        return 0
    ok(app.viewport.winfo_manager() == "pack", "the studio viewport is shown at start")
    ok(panel.lab is None, "the character lab is not loaded until the tab is opened")

    print("opening the Character tab ...")
    t0 = time.perf_counter()
    app.notebook.select(5)
    app.update()
    print(f"  tab opened in {1e3 * (time.perf_counter() - t0):.0f} ms")
    ok(app.mode() == "character" and panel.lab is not None, "the tab built the lab")
    ok(not app.viewport.winfo_manager() and panel.view.winfo_manager() == "pack",
       "the left pane shows the image view")
    pump(app, until=lambda: panel.view.frames_shown > 0)
    settle(app, panel)
    ok(panel.view.frame.tier == "final", "the body is drawn and refined to the final tier")
    screenshot(app, out / "10_character_body.png")

    print("add a wing and a tail ...")
    panel.add(panel.entry("feathered"))
    panel.add(kind="tail")
    ok([s.id for s in panel.lab.doc.specs] == ["wing-1", "tail-1"], "two appendages in the list")
    ok(set(panel.tree.get_children()) == {"wing-1", "tail-1"}, "the list shows them")
    settle(app, panel)
    screenshot(app, out / "11_wing_and_tail.png")

    print("tune: sliders, drags, undo ...")
    panel.lab.doc.select("wing-1")
    app.update()
    span = panel.prop_rows[("param", "span")]
    ok(span.value == 13.0, "the property grid shows the span parameter of the wing")
    steps = len(panel.lab.doc.undo_stack._undo)
    drag(app, span, 16.0)
    ok(panel.lab.doc.spec("wing-1").params["span"] == 16.0, "dragging the span slider edits it")
    ok(len(panel.lab.doc.undo_stack._undo) == steps + 1, "the drag is ONE undo step")
    fold = panel.control_rows["fold"]
    before = panel.control_rows["raise_"].scale.cget("to")
    drag(app, fold, 0.6)
    ok(panel.lab.doc.spec("wing-1").controls["*"]["fold"] == 0.6, "the fold slider poses the wing")
    ok(panel.control_rows["raise_"].scale.cget("to") != before,
       "the raise range follows the fold (live range of the PLACED wing)")
    settle(app, panel)
    screenshot(app, out / "12_tuned.png")
    panel.undo()
    panel.undo()
    ok(panel.lab.doc.spec("wing-1").params.get("span", 13.0) == 13.0
       and not panel.lab.doc.spec("wing-1").controls, "Ctrl+Z twice puts both edits back")
    app.event_generate("<Control-y>")
    app.update()
    ok(panel.lab.doc.spec("wing-1").params["span"] == 16.0, "Ctrl+Y redoes the span")
    panel.redo()

    print("place the tail by clicking the body ...")
    panel.lab.doc.select("tail-1")
    app.update()
    before_face = panel.lab.doc.spec("tail-1").placement["anchor"]["face"]
    panel.place_btn.invoke()
    ok(panel.lab.placing == "tail-1", "'place by clicking' arms the placement")
    panel.lab.set_view("back")
    pump(app, 0.1)
    w, h = panel.view.size
    cx, cy = w // 2, int(h * 0.48)
    panel.view.event_generate("<ButtonPress-1>", x=cx, y=cy)
    panel.view.event_generate("<ButtonRelease-1>", x=cx, y=cy)
    app.update()
    after_face = panel.lab.doc.spec("tail-1").placement["anchor"]["face"]
    ok(panel.lab.placing is None and after_face != before_face,
       f"clicking the body moved the tail anchor (quad {before_face} -> {after_face})")
    settle(app, panel)
    screenshot(app, out / "13_placed.png")

    print("select by clicking an appendage; orbit, pan, zoom ...")
    az = panel.lab.view.azimuth
    panel.view.event_generate("<ButtonPress-1>", x=300, y=300)
    panel.view.event_generate("<B1-Motion>", x=340, y=310)
    panel.view.event_generate("<ButtonRelease-1>", x=340, y=310)
    app.update()
    ok(panel.lab.view.azimuth != az, "a left drag orbits")
    c = panel.lab.view.center
    panel.view.event_generate("<ButtonPress-3>", x=300, y=300)
    panel.view.event_generate("<B3-Motion>", x=320, y=300)
    panel.view.event_generate("<ButtonRelease-3>", x=320, y=300)
    app.update()
    ok(panel.lab.view.center != c, "a right drag pans")
    r = panel.lab.view.radius
    panel.view.event_generate("<Button-4>", x=300, y=300)
    app.update()
    ok(panel.lab.view.radius < r, "the wheel zooms")
    panel.lab.set_view("back")
    panel.lab.frame_content()
    settle(app, panel)

    print("save, reload, export ...")
    mm = out / "smoke.mmchar"
    snap = panel.lab.doc.snapshot()
    ok(panel.save(mm) == mm and mm.exists(), "Save writes a .mmchar")
    panel.new_dialog = lambda: None                      # (the dialog is not for a script)
    panel.lab.doc.new()
    ok(panel.lab.doc.specs == [] and not panel.tree.get_children(), "New empties the list")
    ok(panel.load(mm), "Open reads it back")
    ok(panel.lab.doc.snapshot() == snap, "the reloaded document equals the saved one")
    ok(set(panel.tree.get_children()) == {"wing-1", "tail-1"}, "the list shows the reloaded ones")
    stl = out / "smoke.stl"
    panel.export(stl, force=True)
    ok(panel.wait_task(), "the export finished")
    ok(stl.exists() and len(read_stl(stl).faces) > 10000, "the STL exists")
    shown = panel.report.get("1.0", "end")
    ok("printability" in shown, "the printability report is shown in the panel")
    print("  report:", " | ".join(x for x in shown.splitlines() if x.strip())[:240])
    screenshot(app, out / "14_exported.png")

    print("a blocking report stops the export until 'Export anyway' ...")
    panel.lab.doc.set_placement("tail-1", "offset", 3.0)
    blocked = out / "blocked.stl"
    blocked.unlink(missing_ok=True)                       # (a previous run may have forced it)
    panel.export(blocked)
    panel.wait_task()
    ok(not blocked.exists(), "the blocked export wrote nothing")
    ok(str(panel.force_btn.cget("state")) == "normal", "'Export anyway' is offered")
    ok("BLOCK" in panel.report.get("1.0", "end"), "the blocking finding is listed")
    screenshot(app, out / "15_blocked.png")
    panel.force_btn.invoke()
    panel.wait_task()
    ok(blocked.exists(), "'Export anyway' wrote it")
    panel.lab.doc.undo()

    print("back to the studio ...")
    app.notebook.select(0)
    app.update()
    ok(app.viewport.winfo_manager() == "pack" and not panel.view.winfo_manager(),
       "the Model tab shows the studio viewport again")
    ok(len(app.viewport.find_all()) > 200, "and it still draws the scene")
    app.notebook.select(5)
    app.update()
    ok(panel.view.winfo_manager() == "pack" and panel._poll_job is not None,
       "coming back resumes the tab")

    if args.measure:
        measure(app, panel, out)

    app.destroy()
    print(f"\ncharacter smoke test passed ({checks} checks); artifacts in {out}")
    return 0


# --------------------------------------------------------------------------------------------


def docs_shot(app, panel, path: Path) -> None:
    """The picture in docs/character_tab.md: an angel with a tail and horns, one wing selected."""
    app.notebook.select(5)
    app.update()
    doc = panel.lab.doc
    panel.add(panel.entry("feathered"))
    panel.add(kind="tail", params={"length": 8.0, "curl": 0.9})
    panel.add(kind="horn")
    doc.set_control("wing-1", "raise_", 14.0)
    doc.set_control("wing-1", "fold", 0.12)
    doc.select("wing-1")
    panel.lab.view.azimuth, panel.lab.view.elevation = 152.0, 9.0
    panel.lab.frame_content()
    app.update()
    settle(app, panel)
    pump(app, 0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    screenshot(app, path)
    print(f"wrote {path}")


def measure(app, panel, out: Path) -> None:
    """Acceptance timings with the real Tk: frames per second of each drag INCLUDING PhotoImage,
    the final render arriving on screen, and the Tk timer's gap while the worker renders."""
    lab, view, doc = panel.lab, panel.view, panel.lab.doc
    print(f"\nmeasurements (pane {view.size[0]}x{view.size[1]}, draft scale {lab.draft_scale}):")
    lab.scheduler.debounce = 1e9                          # no final during a drag measurement

    def fps(label, step, n=24):
        times = []
        for i in range(n + 2):
            t = time.perf_counter()
            step(i)
            app.update()                                   # the draw: PhotoImage + canvas items
            times.append(time.perf_counter() - t)
        med = statistics.median(times[2:])
        print(f"  {label:50s} median {1e3 * med:6.1f} ms = {1 / med:5.1f} fps  "
              f"(worst {1e3 * max(times[2:]):.0f} ms)")
        return 1 / med

    def u(i):
        return (i * 0.0731) % 1.0

    def snap(row, frac):
        """A value inside the row's current range (rows keep their range up to date)."""
        lo, hi = float(row.scale.cget("from")), float(row.scale.cget("to"))
        return lo + (hi - lo) * (0.05 + 0.9 * frac)

    for style in ("membrane", "feathered"):
        doc.new()
        doc.add_appendage("wing", params={"style": style})
        lab.frame_content()
        pump(app, 0.3)
        name = "bat" if style == "membrane" else "angel"
        print(f" body + 2 {name} wings (through the panel's sliders):")
        fps("orbit (the drag gesture)", lambda i: lab.orbit(7, 0))
        raise_row = panel.control_rows["raise_"]
        fps("pose drag (raise slider)", lambda i: raise_row.scale.set(snap(raise_row, u(i))))
        height = panel.body_rows[("macro", "height")]
        fps("body drag (height slider)", lambda i: height.scale.set(0.3 + 0.4 * u(i)))
        span = panel.prop_rows[("param", "span")]
        fps("shape-parameter drag (span slider, draft_params)",
            lambda i: span.scale.set(4.0 + 0.5 * i))
    lab.scheduler.debounce = 0.15

    print(" the final render of the angel reaching the screen (900x800 class pane):")
    doc.new()
    doc.add_appendage("wing", params={"style": "feathered"})
    lab.engine._built.clear()
    lab.engine._bodies.clear()
    lab.engine._states.clear()
    lab.engine._shells.clear()
    lab.engine.shading.clear()
    lab.engine.report_cache.clear()
    t_edit = time.perf_counter()
    lab.refresh()
    pump(app, until=lambda: view.frame is not None and view.frame.tier == "final"
         and view.frame.gen == lab.scheduler.gen)
    total = time.perf_counter() - t_edit
    print(f"  edit -> final on screen, all caches cold: {1e3 * total:.0f} ms "
          f"(includes the {1e3 * lab.scheduler.debounce:.0f} ms debounce; "
          f"render itself {1e3 * view.frame.seconds:.0f} ms)")

    print(" Tk timer gap while the worker renders the angel (5 ms timer, in the real mainloop):")
    stamps: list = []
    running = [True]

    def timer():
        stamps.append(time.perf_counter())
        if running[0]:
            app.after(5, timer)

    app.after(5, timer)
    app.after(400, app.quit)                       # an idle phase first
    app.mainloop()
    idle_gaps = [b - a for a, b in zip(stamps, stamps[1:])]

    window: list = []

    def start():
        lab.refresh()                              # the draft is the UI thread's own work ...
        stamps.clear()                             # ... the gaps from here on are the worker's
        window.append(time.perf_counter())
        watch()

    def watch():
        if lab.scheduler.idle and not lab.scheduler._worker.busy:
            window.append(time.perf_counter())
            app.quit()
        else:
            app.after(5, watch)

    # the UI thread's own work is not contention: showing a frame is ~15 ms of PhotoImage, so the
    # gaps that contain a show_frame call are left out (the first frame is the draft of the edit)
    shown: list = []
    inner = view.show_frame

    def timed_show(frame):
        t = time.perf_counter()
        inner(frame)
        shown.append((t, time.perf_counter()))

    view.show_frame = timed_show
    lab.on_frame = timed_show
    worst = []
    for trial in range(3):                         # a cold angel render each time
        lab.engine._built.clear()
        lab.engine._shells.clear()
        lab.engine.shading.clear()
        lab.scheduler.debounce = 0.0
        window.clear()
        shown.clear()
        app.after(10, start)
        app.mainloop()
        busy_gaps = [b - a for a, b, in zip(stamps, stamps[1:])
                     if not any(a <= t1 and t0 <= b for t0, t1 in shown)]
        worst.append(max(busy_gaps))
        print(f"  worker, cold angel render #{trial + 1}: median {1e3 * statistics.median(busy_gaps):.1f} ms, "
              f"p99 {1e3 * sorted(busy_gaps)[int(0.99 * len(busy_gaps))]:.1f} ms, "
              f"max {1e3 * max(busy_gaps):.1f} ms over {1e3 * (window[1] - window[0]):.0f} ms of rendering")
    running[0] = False
    lab.scheduler.debounce = 0.15
    lab.on_frame = inner
    view.show_frame = inner
    print(f"  idle:   median {1e3 * statistics.median(idle_gaps):.1f} ms, "
          f"max {1e3 * max(idle_gaps):.1f} ms   (nominal 5 ms)")
    print(f"  worst gap above the idle median over the 3 renders: "
          f"{1e3 * (max(worst) - statistics.median(idle_gaps)):.1f} ms")


if __name__ == "__main__":
    sys.exit(main())
