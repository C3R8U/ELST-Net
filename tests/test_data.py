"""Indexes should keep video sequences and duplicate artwork files together."""

from pathlib import Path

import numpy as np
from PIL import Image

from elstnet.data import StyleVideoDataset, index_davis, index_mose, index_wikiart, read_manifest


def _write_image(path, value=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((32, 32, 3), value, dtype=np.uint8)).save(path)


def test_wikiart_duplicate_groups(tmp_path):
    root = tmp_path / "art"
    _write_image(root / "style-a" / "a.png", 33)
    _write_image(root / "style-b" / "b.png", 33)
    _write_image(root / "style-a" / "c.png", 220)
    out = tmp_path / "wiki.csv"
    index_wikiart(root, out)
    rows = read_manifest(out)
    same = [row for row in rows if row["sha256"] == rows[0]["sha256"]]
    assert len(same) == 2
    assert len({row["split"] for row in same}) == 1


def test_davis_sequence_structure(tmp_path):
    root = tmp_path / "DAVIS"
    _write_image(root / "JPEGImages" / "480p" / "rider" / "00000.jpg", 10)
    _write_image(root / "JPEGImages" / "480p" / "rider" / "00001.jpg", 20)
    sets = root / "ImageSets" / "2017"
    sets.mkdir(parents=True)
    (sets / "train.txt").write_text("rider\n")
    (sets / "val.txt").write_text("")
    out = tmp_path / "davis.csv"
    index_davis(root, out)
    assert len(read_manifest(out)) == 2


def test_mose_split_structure(tmp_path):
    root = tmp_path / "MOSE"
    _write_image(root / "valid" / "JPEGImages" / "scene-a" / "00000.jpg")
    out = tmp_path / "mose.csv"
    index_mose(root, out, splits=("valid",))
    rows = read_manifest(out)
    assert len(rows) == 1
    assert rows[0]["sequence"] == "scene-a"
