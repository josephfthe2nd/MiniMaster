import struct

import numpy as np
import pytest

from minimaster.core import primitives as prim
from minimaster.core.stl import read_stl, write_stl


@pytest.mark.parametrize("kind", ["box", "icosphere", "cylinder", "torus"])
def test_round_trip(tmp_path, kind):
    original = prim.build(kind)
    path = tmp_path / f"{kind}.stl"
    write_stl(original, path, name=kind)
    loaded = read_stl(path)
    assert len(loaded.faces) == len(original.faces)
    rep = loaded.integrity_report()
    assert rep["watertight"], rep
    assert loaded.volume() == pytest.approx(original.volume(), rel=1e-5)


def test_file_size_matches_format(tmp_path):
    m = prim.box()
    path = tmp_path / "b.stl"
    write_stl(m, path)
    assert path.stat().st_size == 84 + 50 * len(m.faces)
    with open(path, "rb") as fh:
        header = fh.read(80)
        (count,) = struct.unpack("<I", fh.read(4))
    assert count == 12
    assert header.startswith(b"MiniMaster STL")


def test_read_rejects_truncated(tmp_path):
    m = prim.box()
    path = tmp_path / "b.stl"
    write_stl(m, path)
    data = path.read_bytes()
    path.write_bytes(data[:-10])
    with pytest.raises(ValueError, match="corrupt"):
        read_stl(path)


def test_read_rejects_ascii(tmp_path):
    path = tmp_path / "a.stl"
    path.write_text("solid thing\n" + "x" * 200)
    with pytest.raises(ValueError, match="ASCII"):
        read_stl(path)


def test_read_rejects_tiny_file(tmp_path):
    path = tmp_path / "t.stl"
    path.write_bytes(b"hi")
    with pytest.raises(ValueError, match="too small"):
        read_stl(path)


def test_normals_written_outward(tmp_path):
    m = prim.icosphere(1.0, 1)
    path = tmp_path / "s.stl"
    write_stl(m, path)
    raw = np.frombuffer(
        path.read_bytes()[84:],
        dtype=np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")]),
    )
    centers = raw["v"].mean(axis=1)
    dots = np.einsum("ij,ij->i", raw["n"].astype(np.float64), centers.astype(np.float64))
    assert (dots > 0).all()
