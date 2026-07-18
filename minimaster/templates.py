"""Access to shipped starter templates (.mmp scenes bundled with the package)."""

from __future__ import annotations

from pathlib import Path

from .scene import Scene

TEMPLATE_DIR = Path(__file__).parent / "templates"


def list_templates() -> list[str]:
    if not TEMPLATE_DIR.is_dir():
        return []
    return sorted(p.stem for p in TEMPLATE_DIR.glob("*.mmp"))


def template_path(name: str) -> Path:
    path = TEMPLATE_DIR / f"{name}.mmp"
    if not path.is_file():
        known = ", ".join(list_templates()) or "(none installed)"
        raise KeyError(f"no template named {name!r}; available: {known}")
    return path


def load_template(name: str) -> Scene:
    return Scene.load(template_path(name))
