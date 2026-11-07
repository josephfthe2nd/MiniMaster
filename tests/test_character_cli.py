"""`minimaster character --add / --remove / --pose / --list-kinds` and the document they edit.

Rendering goes through the existing software portrait path, which costs seconds per wing
scene, so these tests save documents with an .stl output (body export, quick) and render a
PNG only with a tail and horns.
"""

import json
from pathlib import Path

import pytest

from minimaster.__main__ import main
from minimaster.character import mhbase as mh

FIXTURES = Path(__file__).parent / "fixtures"
needs_assets = pytest.mark.skipif(not mh.available(), reason="CC0 MakeHuman assets not fetched")


def _doc(path):
    return json.loads(Path(path).read_text())


def test_list_kinds_needs_neither_assets_nor_an_output(capsys):
    assert main(["character", "--list-kinds"]) == 0
    out = capsys.readouterr().out
    for word in ("wing", "tail", "horn", "harvest", "style", "feathered", "shoulder_blades",
                 "tail_base", "temples", "shoulder_socket", "pose controls"):
        assert word in out


def test_an_output_is_still_required_to_build_something(capsys):
    assert main(["character", "--add", "tail"]) == 2
    assert "-o/--output" in capsys.readouterr().err


@needs_assets
def test_add_pose_and_save_a_document(tmp_path, capsys):
    stl, doc = tmp_path / "c.stl", tmp_path / "c.mmchar"
    rc = main(["character", "-o", str(stl), "--save-character", str(doc),
               "--add", "wing:style=membrane,span=11", "--add", "tail:length=7",
               "--add", "horn:side=left", "--pose", "wing-1:fold=0.5",
               "--pose", "wing-1.r:raise_=10", "--force"])     # (the document, not the print rules)
    assert rc == 0
    cap = capsys.readouterr()
    assert "appendages: wing-1.l, wing-1.r, tail-1, horn-1.l" in cap.out
    assert "printability" in cap.out
    d = _doc(doc)
    assert d["version"] == 2 and d["anatomy"] == {"cap": 1.0, "enabled": True}
    wing, tail, horn = d["appendages"]
    assert (wing["id"], wing["kind"], wing["side"]) == ("wing-1", "wing", "both")
    assert wing["params"] == {"style": "membrane", "span": 11}
    assert wing["controls"] == {"*": {"fold": 0.5}, "r": {"raise_": 10}}
    assert wing["placement"]["anchor"]["face"] == 10296
    assert (tail["id"], tail["side"], tail["params"]) == ("tail-1", "center", {"length": 7})
    assert (horn["id"], horn["side"]) == ("horn-1", "left")
    assert "sliders" in d and not d["sliders"]                   # the body carries no anatomy


@needs_assets
def test_edit_a_saved_character(tmp_path, capsys):
    first, second = tmp_path / "a.mmchar", tmp_path / "b.mmchar"
    assert main(["character", "-o", str(tmp_path / "a.stl"), "--save-character", str(first),
                 "--add", "wing", "--add", "tail", "--add", "horn"]) == 0
    rc = main(["character", "-o", str(tmp_path / "b.stl"), "--character", str(first),
               "--remove", "wing-1", "--add", "wing:style=feathered,id=angel",
               "--add", "horn:id=crown,side=center", "--no-anatomy", "--anatomy-cap", "0.5",
               "--save-character", str(second)])
    assert rc == 0
    d = _doc(second)
    assert [a["id"] for a in d["appendages"]] == ["tail-1", "horn-1", "angel", "crown"]
    assert d["anatomy"] == {"cap": 0.5, "enabled": False}
    assert "body adapted" not in capsys.readouterr().out.split("wrote")[-1]
    # the first file is untouched
    assert [a["id"] for a in _doc(first)["appendages"]] == ["wing-1", "tail-1", "horn-1"]


@needs_assets
def test_the_body_adapts_unless_told_not_to(tmp_path, capsys):
    args = ["character", "-o", str(tmp_path / "c.stl"), "--add", "wing"]
    assert main(args) == 0
    assert "body adapted: wing (load 1.00)" in capsys.readouterr().out
    assert main(args + ["--no-anatomy"]) == 0
    assert "body adapted" not in capsys.readouterr().out


@needs_assets
def test_the_body_is_built_from_the_derived_weights(tmp_path, monkeypatch):
    """Not just the note: the morph weights the body is built from include the adaptation
    unless it is switched off, and the user's own sliders are never what changes."""
    calls = []
    real = mh.TargetLibrary.apply

    def spy(self, verts, weights, *a, **k):
        calls.append(dict(weights))
        return real(self, verts, weights, *a, **k)

    monkeypatch.setattr(mh.TargetLibrary, "apply", spy)
    out = str(tmp_path / "x.stl")

    def weights(*extra):
        calls.clear()
        assert main(["character", "-o", out, *extra]) == 0
        return calls[-1]

    plain = weights()
    adapted = weights("--add", "wing")
    switched_off = weights("--add", "wing", "--no-anatomy")
    capped = weights("--add", "wing", "--anatomy-cap", "0.1")
    assert switched_off == plain and adapted != plain
    assert any("pectoral" in name for name in set(adapted) - set(plain))
    assert capped != adapted and capped != plain


@needs_assets
def test_a_body_only_character_is_written_exactly_as_before(tmp_path):
    out = tmp_path / "same.mmchar"
    rc = main(["character", "-o", str(tmp_path / "c.stl"), "--character",
               str(FIXTURES / "character_v1.mmchar"), "--save-character", str(out)])
    assert rc == 0
    assert out.read_text() == (FIXTURES / "character_v1.mmchar").read_text()


@needs_assets
@pytest.mark.parametrize("argv,message", [
    (["--add", "wng"], "did you mean 'wing'"),
    (["--add", "tail:lenght=3"], "did you mean 'length'"),
    (["--add", "wing:style=feathered,gull=0.2"], "only applies when style"),
    (["--add", "wing:span=99"], "outside the range"),
    (["--add", "tail:side=up"], "unknown side"),
    (["--add", "harvest:roll=30"], "does not use 'roll'"),
    (["--add", "tail:oops"], "key=value"),
    (["--remove", "wing-9"], "no appendage 'wing-9' to remove"),
    (["--add", "tail", "--pose", "wing-1:fold=1"], "no appendage 'wing-1' to pose"),
    (["--add", "wing", "--pose", "wing-1:flod=1"], "did you mean 'fold'"),
    (["--add", "horn", "--pose", "horn-1:fold=1"], "no controls"),
    (["--add", "tail", "--pose", "tail-1:fold=1"], "has no control 'fold'"),
])
def test_mistakes_are_refused_clearly(tmp_path, capsys, argv, message):
    rc = main(["character", "-o", str(tmp_path / "x.stl"), *argv])
    assert rc == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "x.stl").exists()


@needs_assets
def test_a_png_shows_the_appendages(tmp_path, capsys):
    plain, horned = tmp_path / "plain.png", tmp_path / "horned.png"
    common = ["--res", "160", "240", "--azimuth", "160"]
    assert main(["character", "-o", str(plain), *common]) == 0
    # --no-anatomy keeps the body identical, so any difference is the appendages themselves
    assert main(["character", "-o", str(horned), "--add", "tail", "--add", "horn",
                 "--no-anatomy", *common]) == 0
    assert horned.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert "appendages: tail-1, horn-1.l, horn-1.r" in capsys.readouterr().out
    assert plain.read_bytes() != horned.read_bytes()


@needs_assets
def test_unknown_kinds_in_a_saved_file_are_warned_about_and_kept(tmp_path, capsys):
    src = tmp_path / "future.mmchar"
    src.write_text(json.dumps({
        "format": "minimaster-character", "version": 2, "name": "future",
        "appendages": [{"id": "gizzard-1", "kind": "gizzard", "params": {"x": 1}},
                       {"id": "tail-1", "kind": "tail", "side": "center", "params": {}}],
        "from_the_future": {"k": 1}}))
    out = tmp_path / "again.mmchar"
    rc = main(["character", "-o", str(tmp_path / "c.png"), "--res", "120", "180",
               "--character", str(src), "--save-character", str(out)])
    assert rc == 0
    cap = capsys.readouterr()
    assert "unknown kind 'gizzard'" in cap.err and "appendages: tail-1" in cap.out
    d = _doc(out)
    assert d["from_the_future"] == {"k": 1}
    assert d["appendages"][0] == {"id": "gizzard-1", "kind": "gizzard", "params": {"x": 1},
                                  "side": "center", "controls": {}, "placement": {},
                                  "enabled": True}


# -- the STL export of appendages (the print integration reached through the CLI) -----------

@needs_assets
def test_an_stl_with_appendages_is_scaled_by_the_body_and_carries_a_report(tmp_path, capsys):
    from minimaster.core.stl import read_stl
    stl = tmp_path / "winged.stl"
    rc = main(["character", "--gender", "1", "--add", "wing:style=membrane", "--add", "tail",
               "-o", str(stl), "--size", "medium", "--fit-base"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "printability" in out and "wing-1.l" in out and "tail-1" in out
    assert "32.0 mm is the body" in out or "height       32.0 mm" in out
    mesh = read_stl(stl)
    assert mesh.integrity_report()["watertight"]
    # the body, not the wing tips, sets the size: much more than 32 mm of figure in the file
    lo, hi = mesh.bounds
    assert hi[2] - lo[2] > 33.0


@needs_assets
def test_a_blocking_report_stops_the_export_unless_forced(tmp_path, capsys):
    stl = tmp_path / "hairline.stl"
    # ``--no-junction``: with it the skin rises around even a tail that only touches it and the
    # weld is real (tests/test_junction.py), so the hairline needs the skin left as it is. A tail's
    # bone root is its sink now (offset 0 is 36% embedded): +1.0 slides it out until it only touches
    argv = ["character", "--add", "tail:offset=1.0", "--no-junction", "-o", str(stl),
            "--size", "medium"]
    assert main(argv) == 3
    err = capsys.readouterr().err
    assert "blocks this export" in err and "BLOCKED" in err and "tail-1" in err and not stl.exists()
    assert main(argv + ["--force"]) == 0
    assert stl.exists()


def test_stale_connectivity_links_are_not_replayed_on_another_placement():
    """The overlap links of a group were cached by (piece names, volumes) only, and a rigid
    move leaves volumes alone: the cached 'connected' of a touching placement was replayed onto
    a placement 5 units away."""
    from minimaster.character import print_report as pr
    from minimaster.core import primitives as prim

    body = pr.Solid(prim.box(2.0, 2.0, 2.0))
    cache: dict = {}

    def components(offset):
        a = prim.box(1.0, 1.0, 1.0).translated((0.0, 0.0, 0.0))
        b = prim.box(1.0, 1.0, 1.0).translated((offset, 0.0, 0.0))
        names = ["g/arm", "g/tip"]
        solids = [pr.Solid(a), pr.Solid(b)]
        items = [(n, m) for n, m in zip(names, (a, b))]
        # the signature exactly as _geometry builds it
        import numpy as np
        cen = np.array([m.vertices.mean(axis=0) for _, m in items])
        dist = np.round(np.linalg.norm(cen[:, None, :] - cen[None, :, :], axis=2), 5)
        sig = (tuple((pr.piece_of(n), round(float(m.volume()), 9)) for n, m in items),
               hash(dist.tobytes()))
        return pr._group_components(solids, names, body, cache, sig)

    touching = components(0.5)
    apart = components(5.0)
    assert len(touching) == 1 and len(apart) == 2
