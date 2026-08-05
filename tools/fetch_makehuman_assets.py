"""Fetch the CC0 MakeHuman base mesh + morph targets into assets/makehuman/.

Both files are **CC0 1.0 (public domain dedication)** — see the license notes
in ``minimaster/character/mhbase.py``. Only MakeHuman's *Python source* is
AGPL3, and none of it is used or vendored by MiniMaster; we read the data
formats directly.

    python tools/fetch_makehuman_assets.py

* ``targets.npz`` (~30 MB, 1,280 morph targets) is bundled inside the
  ``makehuman`` distribution on PyPI, so it comes down with ``pip download``.
* ``base.obj`` (19,158 verts / 18,486 quads) lives in the makehuman git
  repository under ``makehuman/data/3dobjs/``.

The files are deliberately NOT committed — they are large, and this script
reproduces them exactly.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "assets" / "makehuman"

TARGETS_MEMBER = "makehuman/data/targets.npz"
BASE_IN_REPO = "makehuman/data/3dobjs/base.obj"
MH_GIT = "https://github.com/makehumancommunity/makehuman"


def fetch_targets(dest: Path) -> bool:
    out = dest / "targets.npz"
    if out.exists():
        print(f"targets.npz already present ({out.stat().st_size / 1e6:.1f} MB)")
        return True
    with tempfile.TemporaryDirectory() as tmp:
        print("downloading the makehuman sdist from PyPI ...")
        r = subprocess.run(
            [sys.executable, "-m", "pip", "download", "makehuman==1.3.2",
             "--no-deps", "--no-binary", ":all:", "-d", tmp],
            capture_output=True, text=True)
        if r.returncode != 0:
            print("  pip download failed:\n", r.stderr[-800:], file=sys.stderr)
            return False
        sdist = next(Path(tmp).glob("makehuman-*.tar.gz"), None)
        if sdist is None:
            print("  no sdist produced", file=sys.stderr)
            return False
        with tarfile.open(sdist) as tf:
            member = next((m for m in tf.getmembers()
                           if m.name.endswith(TARGETS_MEMBER)), None)
            if member is None:
                print("  targets.npz not found in the sdist", file=sys.stderr)
                return False
            src = tf.extractfile(member)
            dest.mkdir(parents=True, exist_ok=True)
            with open(out, "wb") as fh:
                shutil.copyfileobj(src, fh)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return True


def fetch_base(dest: Path, repo: Path | None) -> bool:
    out = dest / "base.obj"
    if out.exists():
        print(f"base.obj already present ({out.stat().st_size / 1e6:.1f} MB)")
        return True
    if repo is not None:
        src = repo / BASE_IN_REPO
        if not src.exists():
            src = repo / "data" / "3dobjs" / "base.obj"
        if src.exists():
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, out)
            print(f"wrote {out} (from {src})")
            return True
        print(f"  {src} not found in --makehuman-repo", file=sys.stderr)
        return False
    with tempfile.TemporaryDirectory() as tmp:
        print("shallow-cloning the makehuman repository ...")
        r = subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none",
             "--sparse", MH_GIT, tmp + "/mh"],
            capture_output=True, text=True,
            env={"GIT_LFS_SKIP_SMUDGE": "1", "PATH": "/usr/bin:/bin:/usr/local/bin"})
        if r.returncode != 0:
            print("  clone failed:\n", r.stderr[-800:], file=sys.stderr)
            return False
        subprocess.run(["git", "-C", tmp + "/mh", "sparse-checkout", "set",
                        "makehuman/data/3dobjs"], capture_output=True)
        src = Path(tmp) / "mh" / BASE_IN_REPO
        if not src.exists():
            print(f"  {BASE_IN_REPO} missing after clone", file=sys.stderr)
            return False
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=str(DEST))
    ap.add_argument("--makehuman-repo", default=None,
                    help="path to an existing makehuman checkout (skips cloning)")
    args = ap.parse_args()
    dest = Path(args.dest)
    repo = Path(args.makehuman_repo) if args.makehuman_repo else None
    ok = fetch_targets(dest) & fetch_base(dest, repo)
    if ok:
        print("\nCC0 assets ready. Character Lab's realistic path is enabled.")
    else:
        print("\nSome assets could not be fetched; see errors above.",
              file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
