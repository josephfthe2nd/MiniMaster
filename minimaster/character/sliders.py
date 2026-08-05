"""Turn the raw morph-target names into a usable slider catalog.

The CC0 library ships 1,280 targets, but they are *files*, not controls: a
single conceptual slider like "nose curve" exists as the two separate targets
``nose/nose-curve-concave`` and ``nose/nose-curve-convex``, and anything on the
face exists twice more as ``l-`` and ``r-`` variants. Exposing the raw names
would mean ~900 checkboxes; exposing only the handful the macro system reaches
left roughly 600 hand-sculpted targets — most of the shipped art — unreachable.

This module derives the catalog from the names themselves, so it stays correct
if the asset set changes. The grammar, verified against the shipped data:

* **Bipolar pairs** — 259 stems carry exactly two targets whose final tokens
  are antonyms (``decr``/``incr``, ``down``/``up``, ``in``/``out``,
  ``backward``/``forward``, ``concave``/``convex``, …). Each becomes one
  -1..+1 slider.
* **Sided pairs** — a leaf beginning ``l-``/``r-`` (132 each) is one side of a
  symmetric feature. Both sides fold into a single slider, with an optional
  asymmetry offset that drives them apart.
* **Translation groups** — stems like ``l-foot-trans`` carry up to three
  antonym pairs at once and split into one slider per axis.
* **Exclusive sets** — ``*-shape`` (pointed/round/square/triangle) and
  ``bodyshapes-elvs-*`` (apple/pear/hourglass/…) are choices, not sliders.
* **Unipolar** — 40 lone stems become one 0..1 slider.

Nothing here imports the mesh or numpy: it is string manipulation over a list
of names, so it is cheap and independently testable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# negative token -> positive token. Order defines which way the slider runs.
# Handled by their own systems rather than the shape catalog: expression
# targets are pose UNITS referenced by the .mhpose presets, and they ship one
# variant per ethnicity under a nested path (expression/units/african/...).
SEPARATE_SYSTEMS = frozenset({"expression"})

ANTONYMS: dict[str, str] = {
    "decr": "incr",
    "down": "up",
    "in": "out",
    "backward": "forward",
    "compress": "uncompress",
    "concave": "convex",
    "closure": "slit",
    "compression": "depression",
    "dilatation": "elevation",
    "l": "r",
}
_NEG = set(ANTONYMS)
_POS = set(ANTONYMS.values())

# category -> (group, region label)
REGIONS: dict[str, tuple[str, str]] = {
    "eyes": ("Face", "Eyes"),
    "eyebrows": ("Face", "Brows"),
    "nose": ("Face", "Nose"),
    "mouth": ("Face", "Mouth"),
    "cheek": ("Face", "Cheeks"),
    "chin": ("Face", "Chin & jaw"),
    "ears": ("Face", "Ears"),
    "forehead": ("Face", "Forehead"),
    "head": ("Face", "Head shape"),
    "neck": ("Body", "Neck"),
    "torso": ("Body", "Torso"),
    "stomach": ("Body", "Stomach"),
    "hip": ("Body", "Hips"),
    "buttocks": ("Body", "Buttocks"),
    "pelvis": ("Body", "Pelvis"),
    "breast": ("Body", "Chest"),
    "genitals": ("Body", "Genitals"),
    "armslegs": ("Body", "Arms & legs"),
    "measure": ("Measurements", "Measurements"),
    "bodyshapes": ("Body", "Body shape"),
    "asym": ("Asymmetry", "Asymmetry"),
    "expression": ("Expression", "Expression"),
}

_WORD_FIX = {"vert": "vertical", "horiz": "horizontal", "circ": "circumference",
             "trans": "move", "scale": "scale", "incr": "", "decr": ""}


def _prettify(stem: str) -> str:
    words = [w for w in stem.replace("_", "-").split("-") if w]
    out = [_WORD_FIX.get(w, w) for w in words]
    out = [w for w in out if w]
    return " ".join(out).capitalize() if out else stem


@dataclass(frozen=True)
class Slider:
    """One user-facing control over one or more morph targets."""

    key: str
    label: str
    group: str
    region: str
    kind: str                                  # bipolar | unipolar | choice
    minus: tuple[str, ...] = ()
    plus: tuple[str, ...] = ()
    options: tuple[tuple[str, tuple[str, ...]], ...] = ()  # (label, targets)
    sided: bool = False
    default: float = 0.0

    @property
    def lo(self) -> float:
        return -1.0 if self.kind == "bipolar" else 0.0

    @property
    def hi(self) -> float:
        return 1.0

    def targets(self) -> tuple[str, ...]:
        opt = tuple(t for _, names in self.options for t in names)
        return tuple(self.minus) + tuple(self.plus) + opt

    def weights(self, value, asymmetry: float = 0.0) -> dict[str, float]:
        """Slider value -> {target: weight}.

        ``asymmetry`` (-1..1) biases a sided slider toward one side; real faces
        are never perfectly symmetric and a little of this reads as realism.
        """
        out: dict[str, float] = {}
        if self.kind == "choice":
            idx = int(round(value))
            if 0 <= idx < len(self.options):
                for nm in self.options[idx][1]:
                    out[nm] = 1.0
            return out

        v = float(value)
        if self.kind == "unipolar":
            v = max(0.0, min(1.0, v))
        else:
            v = max(-1.0, min(1.0, v))

        def emit(names, amount):
            if amount <= 1e-9:
                return
            if self.sided and len(names) == 2 and abs(asymmetry) > 1e-9:
                # names are ordered (left, right)
                bias = max(-1.0, min(1.0, asymmetry))
                lw = max(0.0, min(1.0, amount * (1.0 + bias)))
                rw = max(0.0, min(1.0, amount * (1.0 - bias)))
                for nm, wt in zip(names, (lw, rw)):
                    if wt > 1e-9:
                        out[nm] = out.get(nm, 0.0) + wt
                return
            for nm in names:
                out[nm] = out.get(nm, 0.0) + amount

        if v >= 0:
            emit(self.plus, v)
        else:
            emit(self.minus, -v)
        return out


# --------------------------------------------------------------------------
# parsing


def _split_side(leaf: str) -> tuple[str | None, str]:
    m = re.match(r"^([lr])-(.+)$", leaf)
    if m:
        return m.group(1), m.group(2)
    return None, leaf


def _category(name: str) -> str:
    return name.split("/")[0]


def build_catalog(names) -> list[Slider]:
    """Derive the slider catalog from raw target names."""
    # bucket: (category, side, stem) -> {token: full name}
    buckets: dict[tuple[str, str | None, str], dict[str, str]] = {}
    for name in names:
        cat = _category(name)
        if cat.startswith("macrodetails") or cat in SEPARATE_SYSTEMS:
            continue  # driven elsewhere, not by individual shape sliders
        leaf = name.split("/")[-1]
        side, rest = _split_side(leaf)
        stem, _, token = rest.rpartition("-")
        if not stem:                       # single-word leaf
            stem, token = rest, ""
        buckets.setdefault((cat, side, stem), {})[token] = name

    # merge left/right buckets that share a stem
    merged: dict[tuple[str, str], dict[str, list[str]]] = {}
    sided: set[tuple[str, str]] = set()
    for (cat, side, stem), toks in buckets.items():
        entry = merged.setdefault((cat, stem), {})
        for tok, full in toks.items():
            entry.setdefault(tok, [])
            # keep left before right for a stable asymmetry order
            if side == "r":
                entry[tok].append(full)
            else:
                entry[tok].insert(0, full)
        if side is not None:
            sided.add((cat, stem))

    sliders: list[Slider] = []
    for (cat, stem), toks in sorted(merged.items()):
        group, region = REGIONS.get(cat, ("Other", cat.capitalize()))
        is_sided = (cat, stem) in sided
        base_key = f"{cat}.{stem}".replace("-", "_")

        pairs = [(n, p) for n, p in ANTONYMS.items() if n in toks and p in toks]
        leftovers = set(toks) - {t for pair in pairs for t in pair}

        for neg, pos in pairs:
            suffix = "" if len(pairs) == 1 else f".{neg}_{pos}"
            label = _prettify(stem)
            if len(pairs) > 1:
                label = f"{label} ({pos})"
            sliders.append(Slider(
                key=base_key + suffix.replace("-", "_"),
                label=label, group=group, region=region, kind="bipolar",
                minus=tuple(toks[neg]), plus=tuple(toks[pos]), sided=is_sided))

        if len(leftovers) >= 3:  # an exclusive set (ear shape, body shapes)
            # every side of the option, not just the first — a one-sided ear
            # shape would leave the other ear un-morphed
            options = tuple(sorted(
                (_prettify(t), tuple(toks[t])) for t in leftovers))
            sliders.append(Slider(
                key=base_key + ".choice", label=_prettify(stem) + " type",
                group=group, region=region, kind="choice", options=options,
                sided=is_sided))
        else:
            for tok in sorted(leftovers):
                name = f"{base_key}.{tok}".rstrip(".").replace("-", "_")
                lbl = _prettify(f"{stem}-{tok}" if tok else stem)
                sliders.append(Slider(
                    key=name, label=lbl, group=group, region=region,
                    kind="unipolar", plus=tuple(toks[tok]), sided=is_sided))
    return sorted(sliders, key=lambda s: (s.group, s.region, s.key))


# --------------------------------------------------------------------------
# catalog + character document


class SliderCatalog:
    """Indexed access to the derived sliders."""

    def __init__(self, names):
        self.sliders = build_catalog(names)
        self._by_key = {s.key: s for s in self.sliders}

    def __len__(self) -> int:
        return len(self.sliders)

    def __contains__(self, key: str) -> bool:
        return key in self._by_key

    def __getitem__(self, key: str) -> Slider:
        return self._by_key[key]

    def get(self, key: str):
        return self._by_key.get(key)

    def groups(self) -> dict[str, dict[str, list[Slider]]]:
        out: dict[str, dict[str, list[Slider]]] = {}
        for s in self.sliders:
            out.setdefault(s.group, {}).setdefault(s.region, []).append(s)
        return out

    def find(self, *fragments: str) -> list[Slider]:
        frags = [f.lower() for f in fragments]
        return [s for s in self.sliders
                if all(f in s.key.lower() or f in s.label.lower() for f in frags)]

    def coverage(self, names) -> tuple[int, int]:
        """(targets reachable through sliders, total non-macro targets)."""
        reachable = {t for s in self.sliders for t in s.targets()}
        total = {n for n in names
                 if not n.startswith("macrodetails")
                 and n.split("/")[0] not in SEPARATE_SYSTEMS}
        return len(reachable & total), len(total)

    def weights(self, values: dict[str, float],
                asymmetry: dict[str, float] | None = None) -> dict[str, float]:
        out: dict[str, float] = {}
        asymmetry = asymmetry or {}
        for key, val in values.items():
            s = self._by_key.get(key)
            if s is None:
                continue
            for name, w in s.weights(val, asymmetry.get(key, 0.0)).items():
                out[name] = out.get(name, 0.0) + w
        return out


@dataclass
class Character:
    """A saved character: macro values + slider values + choices."""

    name: str = "unnamed"
    macro: dict = field(default_factory=lambda: {
        "gender": 0.5, "age": 0.5, "muscle": 0.5, "weight": 0.5,
        "height": 0.5, "african": 0.0, "asian": 0.0, "caucasian": 1.0})
    sliders: dict = field(default_factory=dict)
    asymmetry: dict = field(default_factory=dict)
    materials: dict = field(default_factory=lambda: {
        "skin": "SKIN", "eye_color": "#5b7c8d"})

    def to_dict(self) -> dict:
        return {"format": "minimaster-character", "version": 1,
                "name": self.name, "macro": dict(self.macro),
                "sliders": {k: float(v) for k, v in self.sliders.items()},
                "asymmetry": {k: float(v) for k, v in self.asymmetry.items()},
                "materials": dict(self.materials)}

    @classmethod
    def from_dict(cls, d: dict) -> "Character":
        c = cls(name=d.get("name", "unnamed"))
        c.macro.update(d.get("macro", {}))
        c.sliders = {k: float(v) for k, v in d.get("sliders", {}).items()}
        c.asymmetry = {k: float(v) for k, v in d.get("asymmetry", {}).items()}
        c.materials.update(d.get("materials", {}))
        return c

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2,
                                         sort_keys=True) + "\n")

    @classmethod
    def load(cls, path) -> "Character":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def target_weights(self, catalog: SliderCatalog) -> dict[str, float]:
        """Everything this character asks for, as target weights.

        Macro targets come from :func:`mhbase.macro_weights`; slider targets
        from the catalog. Unknown slider keys are ignored, so a character saved
        against a different asset set still loads.
        """
        from .mhbase import macro_weights
        out = dict(macro_weights(**self.macro))
        for name, w in catalog.weights(self.sliders, self.asymmetry).items():
            out[name] = out.get(name, 0.0) + w
        return out


def randomize(catalog: SliderCatalog, seed=None, amount: float = 0.35,
              groups=("Face",)) -> dict[str, float]:
    """Plausible random slider values: a gaussian around neutral, not uniform
    noise — real faces cluster near the mean, and uniform draws look deformed.
    """
    import random as _random
    rng = _random.Random(seed)
    out: dict[str, float] = {}
    for s in catalog.sliders:
        if groups and s.group not in groups:
            continue
        if s.kind == "choice":
            continue
        v = rng.gauss(0.0, amount)
        v = max(s.lo, min(s.hi, v))
        if abs(v) > 0.02:
            out[s.key] = round(v, 3)
    return out
