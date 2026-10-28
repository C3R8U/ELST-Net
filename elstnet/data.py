"""Dataset indexes and lazy image loading for the artistic-video pipeline."""

import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .utils import rgb_tensor, sha256_file

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def write_manifest(path: str | Path, rows: list[dict], fields: list[str]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _stable_value(value: str) -> float:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16) / 16**16


def index_wikiart(root: str | Path, destination: str | Path, seed: int = 42,
                  artist_map: str | None = None) -> dict:
    """Split by exact-image identity and optional artist identity within each movement."""
    root = Path(root).resolve()
    mapped = {}
    if artist_map:
        for row in read_manifest(artist_map):
            mapped[row["path"]] = row["artist_id"]
    groups = defaultdict(list)
    for file in sorted(root.rglob("*")):
        if file.suffix.lower() not in IMAGE_EXT or not file.is_file():
            continue
        relative = file.relative_to(root).as_posix()
        genre = file.relative_to(root).parts[0] if len(file.relative_to(root).parts) > 1 else "unknown"
        try:
            with Image.open(file) as image:
                image.verify()
        except (OSError, ValueError):
            continue
        digest = sha256_file(file)
        # An artist mapping keeps multiple paintings by one artist together.
        group_id = ("artist:" + mapped[relative]) if mapped.get(relative) else ("hash:" + digest)
        groups[group_id].append({"path": relative, "genre": genre, "sha256": digest,
                                 "artist_id": mapped.get(relative, "")})
    # Duplicate files always share a split, even when they have different labels.
    split_groups = defaultdict(list)
    for group_id, rows in groups.items():
        for row in rows:
            split_groups[row["sha256"]].append(group_id)
    parent = {key: key for key in groups}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for linked in split_groups.values():
        for other in linked[1:]:
            parent[find(other)] = find(linked[0])
    merged = defaultdict(list)
    for group_id, rows in groups.items():
        merged[find(group_id)].extend(rows)
    selected = {}
    for group_id, rows in merged.items():
        score = _stable_value(f"{seed}:{group_id}")
        split = "train" if score < 0.8 else "val" if score < 0.9 else "test"
        selected[group_id] = split
    all_rows = []
    for group_id, rows in merged.items():
        for row in rows:
            all_rows.append({**row, "split": selected[group_id]})
    write_manifest(destination, sorted(all_rows, key=lambda row: row["path"]),
                   ["path", "genre", "sha256", "artist_id", "split"])
    counts = {split: sum(row["split"] == split for row in all_rows)
              for split in ("train", "val", "test")}
    return {"total": len(all_rows), "counts": counts, "unique_hashes": len(split_groups)}


def _sequence_files(images: Path, sequence: str) -> list[Path]:
    folder = images / sequence
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXT)


def index_davis(root: str | Path, destination: str | Path) -> dict:
    """Respect the official DAVIS 2017 train and val sequence lists."""
    root = Path(root).resolve()
    image_root = root / "JPEGImages" / "480p"
    split_root = root / "ImageSets" / "2017"
    if not image_root.is_dir() or not split_root.is_dir():
        raise FileNotFoundError("Expected JPEGImages/480p and ImageSets/2017")
    rows = []
    for split in ("train", "val"):
        split_file = split_root / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(split_file)
        sequence_names = split_file.read_text(encoding="utf-8").splitlines()
        for sequence in (line.strip() for line in sequence_names if line.strip()):
            for frame in _sequence_files(image_root, sequence):
                rows.append({"dataset": "davis", "split": split,
                             "sequence": sequence, "path": frame.relative_to(root).as_posix(),
                             "frame": frame.stem})
    write_manifest(destination, rows, ["dataset", "split", "sequence", "path", "frame"])
    return {"frames": len(rows), "sequences": len(set((r["split"], r["sequence"]) for r in rows))}


def index_mose(root: str | Path, destination: str | Path, splits=("train", "valid")) -> dict:
    """Look for the JPEGImages folder in each extracted MOSEv2 split archive."""
    root = Path(root).resolve()
    rows = []
    for split in splits:
        candidates = [root / split / "JPEGImages", root / split / "JPEGImages" / "480p",
                      root / "JPEGImages"]
        images = next((folder for folder in candidates if folder.is_dir()), None)
        if images is None:
            if split == "valid" and (root / "val" / "JPEGImages").is_dir():
                images = root / "val" / "JPEGImages"
            else:
                raise FileNotFoundError(f"No extracted JPEGImages folder for {split} under {root}")
        for folder in sorted(p for p in images.iterdir() if p.is_dir()):
            for frame in sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT):
                rows.append({"dataset": "mosev2", "split": split,
                             "sequence": folder.name, "path": frame.relative_to(root).as_posix(),
                             "frame": frame.stem})
        if images == root / "JPEGImages":
            break
    write_manifest(destination, rows, ["dataset", "split", "sequence", "path", "frame"])
    return {"frames": len(rows), "sequences": len(set((r["split"], r["sequence"]) for r in rows))}


class StyleVideoDataset(Dataset):
    """Return temporal clips paired with independently selected artistic references."""

    def __init__(self, video_manifest, video_root, wiki_manifest, wiki_root,
                 split="train", length=8, size=256, stride=1, seed=42,
                 wiki_split=None):
        self.video_root = Path(video_root)
        self.wiki_root = Path(wiki_root)
        self.size, self.length, self.seed = int(size), int(length), int(seed)
        self.video_rows = defaultdict(list)
        for row in read_manifest(video_manifest):
            if row["split"] == split:
                self.video_rows[(row["dataset"], row["sequence"])].append(row)
        self.clips = []
        for key, rows in self.video_rows.items():
            rows.sort(key=lambda r: r["frame"])
            if len(rows) < length:
                continue
            # Full clips stay within the original video; no neighboring split crosses over.
            for start in range(0, len(rows) - length + 1, stride):
                self.clips.append((key, start))
        style_split = wiki_split or split
        self.styles = [row for row in read_manifest(wiki_manifest) if row["split"] == style_split]
        if not self.styles and split != "train":
            raise ValueError(f"No WikiArt images in {style_split} split")
        if not self.clips:
            raise ValueError(f"No {length}-frame clips in {split} split")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        key, start = self.clips[index]
        video = self.video_rows[key][start:start + self.length]
        frames = torch.stack([rgb_tensor(self.video_root / row["path"], self.size)
                              for row in video])
        # DataLoader workers receive independent random streams from PyTorch.
        style = random.choice(self.styles)
        image = rgb_tensor(self.wiki_root / style["path"], self.size)
        if key and self.video_rows[key][start]["split"] == "train" and random.random() < 0.5:
            frames, image = frames.flip(-1), image.flip(-1)
        return {"video": frames, "style": image,
                "sequence": key[1], "style_path": style["path"]}


class VideoSequences:
    """Iterate over whole sequences without changing their official split."""

    def __init__(self, manifest, root, split="val", size=512):
        self.root, self.size = Path(root), int(size)
        self.rows = defaultdict(list)
        for row in read_manifest(manifest):
            if row["split"] == split:
                self.rows[(row["dataset"], row["sequence"])].append(row)
        for rows in self.rows.values():
            rows.sort(key=lambda r: r["frame"])

    def __iter__(self):
        for key, rows in sorted(self.rows.items()):
            frames = torch.stack([rgb_tensor(self.root / row["path"], self.size)
                                  for row in rows])
            yield key, frames
