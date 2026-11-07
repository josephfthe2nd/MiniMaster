"""Timings of the Character tab's preview pipeline (no tkinter, no display needed).

    python tools/lab_bench.py [--size 900x800] [--frames 20]

Measures what the tab spends per frame, without the one thing that needs a display
(``PhotoImage.configure``, measured by ``tools/character_smoke.py``): the draft tier for a body
and two bat wings (orbit, pose drag, body drag) and a shape-parameter drag of a feathered wing
(``draft_params``), the final render of the angel cold and warm, and the latency of cancelling a
final render in flight. The draft numbers include building the PPM bytes of the frame.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minimaster.character import lab as L  # noqa: E402


def ms(seconds: float) -> str:
    return f"{seconds * 1e3:7.1f} ms"


def timed(fn, n: int, warm: int = 2):
    """Median / worst seconds of ``fn(i)`` over ``n`` calls; callers pass a DIFFERENT value
    each call, because a repeated value is (rightly) served from the caches."""
    times = []
    for i in range(n + warm):
        t = time.perf_counter()
        fn(i)
        times.append(time.perf_counter() - t)
    return statistics.median(times[warm:]), max(times[warm:])


def report(name: str, median: float, worst: float) -> None:
    print(f"  {name:44s} median {ms(median)} = {1 / median:5.1f} fps   worst {ms(worst)}")


def make_lab(size, style: str, res) -> L.Lab:
    lab = L.Lab(res, size=size, threaded=False)
    lab.scheduler.debounce = 1e9                      # the final never starts during a drag
    lab.on_frame = lambda f: f.ppm()                  # bytes for tk are part of the frame cost
    lab.doc.add_appendage("wing", params={"style": style} if style != "membrane" else None)
    lab.frame_content()
    return lab


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="900x800")
    ap.add_argument("--frames", type=int, default=20)
    args = ap.parse_args()
    w, h = (int(x) for x in args.size.split("x"))
    n = args.frames
    t0 = time.perf_counter()
    res = L.Resources.shared()
    print(f"assets + catalog loaded in {ms(time.perf_counter() - t0)}; pane {w}x{h}, "
          f"draft scale {2 if w * h > L.DRAFT_FULL_PIXELS else 1}")

    print("draft tier, body + 2 bat wings (frame + PPM bytes):")
    lab = make_lab((w, h), "membrane", res)
    spec = lab.doc.spec("wing-1")
    report("orbit", *timed(lambda i: lab.orbit(7, 0), n))
    lo, hi = lab.doc.control_range("wing-1", "raise_")
    report("pose drag (raise_)",
           *timed(lambda i: lab.doc.set_control("wing-1", "raise_", lo + (hi - lo) * (0.05 + 0.9 * ((i * 0.0731) % 1.0))), n))
    report("body drag (height slider, wings re-placed)",
           *timed(lambda i: lab.doc.set_macro("height", 0.3 + 0.4 * ((i * 0.0731) % 1.0)), n))

    def rebuilt(i):                                   # the exact path: what a body drag would cost
        lab.doc.set_macro("height", 0.3 + 0.4 * ((i * 0.0731) % 1.0))
        lab.engine.render_draft(lab.doc.character, lab.view, lab.size, scale=lab.draft_scale,
                                mode="exact").ppm()

    report("body drag, wings REBUILT (no re-place)", *timed(rebuilt, n))
    report("shape drag (span), bat",
           *timed(lambda i: lab.doc.set_param(spec.id, "span", 10 + 6.0 * ((i * 0.0731) % 1.0)), n))
    lab.close()

    print("draft tier, feathered wings (the angel, draft rig):")
    lab = make_lab((w, h), "feathered", res)
    report("orbit", *timed(lambda i: lab.orbit(7, 0), n))
    lo, hi = lab.doc.control_range("wing-1", "raise_")
    report("pose drag (raise_)",
           *timed(lambda i: lab.doc.set_control("wing-1", "raise_", lo + (hi - lo) * (0.05 + 0.9 * ((i * 0.0731) % 1.0))), n))
    report("body drag (height slider, wings re-placed)",
           *timed(lambda i: lab.doc.set_macro("height", 0.3 + 0.4 * ((i * 0.0731) % 1.0)), n))
    report("shape drag (span) with draft_params",
           *timed(lambda i: lab.doc.set_param("wing-1", "span", 10 + 6.0 * ((i * 0.0731) % 1.0)), n))
    lab.close()

    print("final tier, the angel (feathered pair) at the pane size:")
    cold = L.Lab(res, size=(w, h), threaded=False)
    cold.doc.add_appendage("wing", params={"style": "feathered"})
    t = time.perf_counter()
    f = cold.engine.render_final(cold.doc.character, cold.view, cold.size)
    print(f"  cold  (fresh engine: morph + build + pose + shading caches) {ms(time.perf_counter() - t)}")
    t = time.perf_counter()
    cold.engine.render_final(cold.doc.character, cold.view, cold.size)
    print(f"  warm  (everything cached, camera move)                      {ms(time.perf_counter() - t)}")
    cold.doc.set_control("wing-1", "fold", 0.4)
    t = time.perf_counter()
    cold.engine.render_final(cold.doc.character, cold.view, cold.size)
    print(f"  after a pose edit (re-pose + wing normals/AO)               {ms(time.perf_counter() - t)}")
    del f

    print("cancel latency (final of the angel running on the worker thread):")
    lab = L.Lab(res, size=(w, h), threaded=True, debounce=0.0)
    lab.doc.add_appendage("wing", params={"style": "feathered"})
    lab.scheduler.wait_idle(60)                      # populate the caches, like a settled session
    worker = lab.scheduler._worker
    lats = []
    for k in range(10):
        lab.doc.set_control("wing-1", "fold", 0.2 + 0.03 * k)      # draft now, final armed
        lab.scheduler.tick()                                       # the final starts
        time.sleep(0.01 + 0.04 * k)                                # cancel at different points
        worker.last_cancel_latency = None
        lab.doc.set_control("wing-1", "fold", 0.25 + 0.03 * k)     # cancels, then drafts
        end = time.perf_counter() + 3
        while time.perf_counter() < end and worker.last_cancel_latency is None and worker._running:
            time.sleep(0.001)
        lats.append(worker.last_cancel_latency)                    # None: it had already finished
        lab.scheduler.wait_idle(30)
    got = [x for x in lats if x is not None]
    print(f"  cancel -> worker aborted: {', '.join(ms(x).strip() for x in got) or 'n/a'}"
          f"   (max {ms(max(got)).strip() if got else 'n/a'})")
    lab.close()
    print(f"threads alive at the end: {threading.active_count()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
