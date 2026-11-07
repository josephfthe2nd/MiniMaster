"""The appendage records a saved character carries (pure data, no numpy).

``sliders.py`` is numpy-free on purpose: opening a ``.mmchar`` must stay cheap, and a
character without appendages must not pay for the wing modules. So the *document* side of
an appendage is this module: a plain record with an id, a kind name, a side, the
kind's parameters, its pose controls, where it is placed and a few flags. Everything
that needs geometry (the kind registry, the placement engine, the wing rigs) lives in
:mod:`minimaster.character.kinds` and is imported only when something is actually built.

Persistence rules (design D2):

* the record is written verbatim: parameters, controls and placement are never coerced,
  validated or dropped when the document is loaded or saved;
* a key this version does not know (a new field, a parameter of a newer kind, a kind that
  does not exist here) is kept in the record and written back, and is only *skipped, with
  a warning,* when something is built from it;
* every shell and joint name derives from the ``id`` (``wing-1.l``), so ids are stable and
  never renumbered.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

SIDES = ("center", "left", "right", "both")
"""``center`` on the midline, ``left`` / ``right`` a single one, ``both`` a mirrored pair.
The left-hand definition is stored once; the right side is its exact mirror."""

# keys of a record this version understands; everything else rides along in ``extra``
KNOWN_KEYS = ("id", "kind", "side", "params", "controls", "placement", "enabled", "name")

# an id becomes part of shell and joint names ("wing-1.l") and of command lines
# ("--pose wing-1.r:fold=1"), so it may not contain the separators those use
_ID_OK = re.compile(r"^[A-Za-z0-9_-]+$")


def valid_id(spec_id) -> bool:
    return isinstance(spec_id, str) and bool(_ID_OK.match(spec_id))


def next_id(kind: str, existing) -> str:
    """The smallest unused ``"<kind>-<n>"``. Ids are never renumbered, so the shell and
    joint names derived from them stay valid for as long as the spec exists."""
    taken = {getattr(e, "id", e) for e in existing}
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", str(kind)).strip("_") or "appendage"
    n = 1
    while f"{stem}-{n}" in taken:
        n += 1
    return f"{stem}-{n}"


@dataclass
class AppendageSpec:
    """One appendage of a character: wing, tail, horn, extra limb, or a kind from the future.

    ``params``     what the kind builds (``{"style": "feathered", "span": 13.0}``)
    ``controls``   how it is posed: shared controls and/or per-side overrides,
                   ``{"fold": 0.5}`` or ``{"*": {"fold": 0.5}, "r": {"raise_": 20}}``
    ``placement``  where it sits, as a *topological* binding that follows the body's
                   morphs: ``{"anchor": {"face": 10296}, "scale": 1.0, "roll": -95.0,
                   "offset": -0.9, "up_hint": [1.0, 1.4, -0.45], "aim": {...}}``
    """

    id: str
    kind: str
    side: str = "center"
    params: dict = field(default_factory=dict)
    controls: dict = field(default_factory=dict)
    placement: dict = field(default_factory=dict)
    enabled: bool = True
    name: str = ""
    extra: dict = field(default_factory=dict)       # unknown keys, written back verbatim

    # -- what creature_anatomy reads (it duck-types .kind .params .scale .mirror) --

    @property
    def scale(self) -> float:
        try:
            return float(self.placement.get("scale", 1.0))
        except (TypeError, ValueError):
            return 1.0

    @property
    def mirror(self) -> bool:
        return self.side == "both"

    # -- persistence ----------------------------------------------------------

    def to_dict(self) -> dict:
        out = copy.deepcopy(self.extra)
        out.update(id=self.id, kind=self.kind, side=self.side,
                   params=copy.deepcopy(self.params),
                   controls=copy.deepcopy(self.controls),
                   placement=copy.deepcopy(self.placement),
                   enabled=bool(self.enabled))
        if self.name:
            out["name"] = self.name
        return out

    @classmethod
    def from_dict(cls, d) -> "AppendageSpec":
        if not isinstance(d, Mapping):
            raise ValueError(f"an appendage record must be an object, not {type(d).__name__}")
        for key in ("params", "controls", "placement"):
            if key in d and not isinstance(d[key], Mapping):
                raise ValueError(f"appendage {d.get('id', '?')!r}: {key!r} must be an object")
        d = copy.deepcopy(dict(d))
        placement = dict(d.get("placement") or {})
        side = d.get("side")
        if side is None:                      # a record written with the side inside placement
            side = placement.pop("side", "center")
        return cls(
            id=str(d.get("id", "")), kind=str(d.get("kind", "")), side=str(side),
            params=dict(d.get("params") or {}), controls=dict(d.get("controls") or {}),
            placement=placement, enabled=bool(d.get("enabled", True)),
            name=str(d.get("name", "")),
            extra={k: v for k, v in d.items() if k not in KNOWN_KEYS})

    def copy(self) -> "AppendageSpec":
        return AppendageSpec.from_dict(self.to_dict())

    def label(self) -> str:
        return self.name or self.id


def find(specs, spec_id: str):
    """The spec called ``spec_id``, or ``None``."""
    for s in specs:
        if s.id == spec_id:
            return s
    return None
