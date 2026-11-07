"""The Character document with appendages: pure data, versioned, forward compatible.

No MakeHuman assets needed: this file is about the file format. Placement and the kinds
that read it are in test_kinds.py; the derived body layer in test_creature_anatomy.py.
"""

import copy
import json
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from minimaster.character.appendage_spec import (
    AppendageSpec, KNOWN_KEYS, SIDES, find, next_id, valid_id)
from minimaster.character.sliders import Character

FIXTURES = Path(__file__).parent / "fixtures"


# -- layering -----------------------------------------------------------------

def test_loading_a_character_does_not_import_numpy_or_the_wing_modules():
    """sliders.py and appendage_spec.py are the cheap, numpy-free document side. A
    character with appendages must load, save and round-trip without the geometry."""
    code = (
        "import sys, json\n"
        "from minimaster.character.sliders import Character\n"
        "d = {'format': 'minimaster-character', 'version': 2, 'name': 'x',\n"
        "     'appendages': [{'id': 'wing-1', 'kind': 'wing', 'side': 'both',\n"
        "                     'params': {'style': 'feathered'}}]}\n"
        "c = Character.from_dict(d)\n"
        "assert json.dumps(c.to_dict())\n"
        "bad = [m for m in sys.modules if m == 'numpy' or m.startswith('numpy.')\n"
        "       or m.endswith(('character.wings', 'character.wingrig',\n"
        "                      'character.appendages', 'character.kinds'))]\n"
        "assert not bad, bad\n")
    root = Path(__file__).resolve().parents[1]
    res = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True,
                         text=True, env={"PYTHONPATH": str(root), "PATH": "/usr/bin:/bin"})
    assert res.returncode == 0, res.stderr


# -- version 1 is untouched -------------------------------------------------------

def test_golden_v1_file_loads_and_resaves_byte_for_byte():
    """The fixture was written by the code BEFORE this change (bcdae81)."""
    text = (FIXTURES / "character_v1.mmchar").read_text()
    c = Character.load(FIXTURES / "character_v1.mmchar")
    assert c.appendages == [] and c.extra == {}
    assert c.name == "golden-v1" and c.macro["gender"] == 0.8
    assert c.sliders["nose.nose_curve"] == 0.8 and c.asymmetry["cheek.cheek_bones"] == 0.15
    assert c.materials["eye_color"] == "#3c6e47"
    assert json.dumps(c.to_dict(), indent=2, sort_keys=True) + "\n" == text


def test_default_character_still_writes_version_1_byte_for_byte(tmp_path):
    out = tmp_path / "c.mmchar"
    Character().save(out)
    assert out.read_text() == (FIXTURES / "character_v1_default.mmchar").read_text()
    d = json.loads(out.read_text())
    assert d["version"] == 1 and "appendages" not in d and "anatomy" not in d


def test_version_2_is_written_only_when_appendages_exist():
    c = Character()
    assert c.to_dict()["version"] == 1 and "appendages" not in c.to_dict()
    c.appendages.append(AppendageSpec(id="tail-1", kind="tail"))
    d = c.to_dict()
    assert d["version"] == 2 and d["appendages"][0]["id"] == "tail-1"
    assert d["anatomy"] == {"enabled": True, "cap": 1.0}
    assert c.remove_appendage("tail-1") and not c.remove_appendage("tail-1")
    assert c.to_dict() == Character().to_dict()      # back to the plain v1 document


# -- the old reader -----------------------------------------------------------------

@dataclass
class _HeadCharacter:
    """``Character`` exactly as it was at bcdae81 (the reader a user may still have)."""

    name: str = "unnamed"
    macro: dict = field(default_factory=lambda: {
        "gender": 0.5, "age": 0.5, "muscle": 0.5, "weight": 0.5,
        "height": 0.5, "african": 0.0, "asian": 0.0, "caucasian": 1.0})
    sliders: dict = field(default_factory=dict)
    asymmetry: dict = field(default_factory=dict)
    materials: dict = field(default_factory=lambda: {
        "skin": "SKIN", "eye_color": "#5b7c8d"})

    @classmethod
    def from_dict(cls, d):
        c = cls(name=d.get("name", "unnamed"))
        c.macro.update(d.get("macro", {}))
        c.sliders = {k: float(v) for k, v in d.get("sliders", {}).items()}
        c.asymmetry = {k: float(v) for k, v in d.get("asymmetry", {}).items()}
        c.materials.update(d.get("materials", {}))
        return c


def test_an_old_reader_loads_a_v2_file_without_crashing():
    c = Character(name="winged")
    c.macro["height"] = 0.9
    c.sliders["nose.nose_curve"] = 0.4
    c.appendages += [AppendageSpec(id="wing-1", kind="wing", side="both",
                                   params={"style": "feathered"},
                                   placement={"anchor": {"face": 10296}}),
                     AppendageSpec(id="gizzard-1", kind="gizzard", extra={"future": [1, 2]})]
    text = json.dumps(c.to_dict())
    old = _HeadCharacter.from_dict(json.loads(text))     # must not raise
    assert old.name == "winged" and old.macro["height"] == 0.9
    assert old.sliders == {"nose.nose_curve": 0.4}


# -- the spec record --------------------------------------------------------------

def test_spec_round_trip_is_verbatim():
    d = {"id": "wing-1", "kind": "wing", "side": "both", "enabled": False, "name": "Left & right",
         "params": {"style": "feathered", "span": 13.0, "coverts": True},
         "controls": {"*": {"fold": 0.5}, "r": {"preset": "folded"}},
         "placement": {"anchor": {"face": 10296, "weights": [0.25] * 4},
                       "scale": 1.5, "up_hint": [1.0, 1.4, -0.45]}}
    spec = AppendageSpec.from_dict(d)
    assert spec.to_dict() == d
    assert spec.scale == 1.5 and spec.mirror is True and spec.label() == "Left & right"
    # a copy is independent of the original
    cp = spec.copy()
    cp.params["span"] = 99
    cp.placement["anchor"]["face"] = 1
    assert spec.params["span"] == 13.0 and spec.placement["anchor"]["face"] == 10296


def test_spec_defaults_and_duck_typing_for_creature_anatomy():
    s = AppendageSpec(id="tail-1", kind="tail")
    assert s.side == "center" and s.scale == 1.0 and s.mirror is False and s.enabled
    assert "name" not in s.to_dict()
    assert AppendageSpec(id="h", kind="horn", side="both").mirror
    assert not AppendageSpec(id="h", kind="horn", side="left").mirror
    assert AppendageSpec(id="x", kind="tail", placement={"scale": "oops"}).scale == 1.0
    assert set(SIDES) == {"center", "left", "right", "both"}


def test_a_side_inside_placement_is_read_as_the_side():
    s = AppendageSpec.from_dict({"id": "h-1", "kind": "horn",
                                 "placement": {"side": "both", "scale": 2.0}})
    assert s.side == "both" and "side" not in s.placement


def test_malformed_records_are_refused_clearly():
    with pytest.raises(ValueError, match="object"):
        AppendageSpec.from_dict(["not", "a", "dict"])
    with pytest.raises(ValueError, match="params"):
        AppendageSpec.from_dict({"id": "x", "kind": "tail", "params": [1]})
    with pytest.raises(ValueError, match="appendages"):
        Character.from_dict({"appendages": {"wing-1": {}}})


def test_ids_are_short_stable_and_never_renumbered():
    specs = []
    for _ in range(3):
        specs.append(AppendageSpec(id=next_id("wing", specs), kind="wing"))
    assert [s.id for s in specs] == ["wing-1", "wing-2", "wing-3"]
    gone = specs.pop(0)                                  # removing wing-1 ...
    assert [s.id for s in specs] == ["wing-2", "wing-3"]   # ... renames nobody
    assert next_id("wing", specs) == gone.id             # the freed id is the smallest unused
    assert next_id("tail", specs) == "tail-1"
    assert next_id("weird kind!", []) == "weird_kind-1"
    assert find(specs, "wing-3") is specs[1] and find(specs, "nope") is None
    assert valid_id("wing-1") and not valid_id("wing.1") and not valid_id("a:b")
    assert not valid_id("") and not valid_id(None)


def test_a_record_without_an_id_gets_one_on_load():
    c = Character.from_dict({"appendages": [{"kind": "tail"}, {"kind": "tail"}]})
    assert [a.id for a in c.appendages] == ["tail-1", "tail-2"]


# -- nothing unknown is ever lost -------------------------------------------------

def _future_document():
    return {
        "format": "minimaster-character", "version": 2, "name": "from the future",
        "macro": {"gender": 0.5}, "sliders": {"no.such_slider": 0.3},
        "appendages": [
            {"id": "wing-1", "kind": "wing", "side": "both", "enabled": True,
             "params": {"style": "gryphon", "span": 13.0, "wobble": 0.3},
             "controls": {"*": {"fold": 0.5, "levitate": 1.0}, "l": {"preset": "hover"}},
             "placement": {"anchor": {"face": 5, "shiny": True}, "scale": 1.0},
             "provenance": {"made_by": "v9"}},
            {"id": "gizzard-1", "kind": "gizzard", "params": {"x": 1}, "future_field": [1, 2]},
        ],
        "anatomy": {"enabled": False, "cap": 0.4, "mood": "grumpy"},
        "future_top_level": {"a": 1}, "another": [1, 2, 3],
    }


def test_unknown_kinds_params_controls_styles_and_keys_survive_load_and_save(tmp_path):
    original = _future_document()
    c = Character.from_dict(copy.deepcopy(original))
    out = tmp_path / "x.mmchar"
    c.save(out)
    again = json.loads(out.read_text())
    for key in ("future_top_level", "another"):
        assert again[key] == original[key]
    assert again["anatomy"] == original["anatomy"]
    wing, gizzard = again["appendages"]
    assert wing["params"] == original["appendages"][0]["params"]       # style + wobble kept
    assert wing["controls"] == original["appendages"][0]["controls"]   # levitate + preset kept
    assert wing["placement"] == original["appendages"][0]["placement"]
    assert wing["provenance"] == {"made_by": "v9"}
    assert gizzard["kind"] == "gizzard" and gizzard["future_field"] == [1, 2]
    # and a second generation is stable
    c2 = Character.load(out)
    assert c2.to_dict() == again


def test_extra_keys_cannot_override_the_documents_own_keys():
    c = Character()
    c.extra = {"name": "hijack", "version": 99, "extra_ok": 1}
    d = c.to_dict()
    assert d["name"] == "unnamed" and d["version"] == 1 and d["extra_ok"] == 1


def test_a_newer_version_is_not_relabelled_and_warns():
    doc = {"format": "minimaster-character", "version": 3, "name": "n", "newthing": {"k": 1}}
    with pytest.warns(UserWarning, match="version 3"):
        c = Character.from_dict(doc)
    out = c.to_dict()
    assert out["version"] == 3 and out["newthing"] == {"k": 1}


def test_a_v1_file_with_unknown_top_level_keys_keeps_them():
    d = {"format": "minimaster-character", "version": 1, "name": "n", "mystery": [1]}
    c = Character.from_dict(d)
    assert c.extra == {"mystery": [1]}
    assert c.to_dict()["mystery"] == [1] and c.to_dict()["version"] == 1


def test_known_keys_constant_matches_what_the_record_writes():
    s = AppendageSpec(id="a", kind="k", name="n")
    assert set(s.to_dict()) == set(KNOWN_KEYS)


def test_loading_a_document_never_warns_for_unknown_kinds():
    """Unknown kinds are skipped with a warning only when something is BUILT from them."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Character.from_dict(_future_document())
