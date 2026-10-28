#!/usr/bin/env python3
"""Fetch original dataset archives without storing third-party images in the source tree."""

import argparse
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ART_MOVEMENTS = (
    "abstract_expressionism", "baroque", "ecole_de_paris", "expressionism",
    "impressionism", "naive_art_primitivism", "neo_impressionism", "neoclassicism",
    "post_impressionism", "pre_raphaelite_brotherhood", "realism", "rococo",
    "romanticism", "surrealism", "symbolism"
)

DAVIS_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
WIKIART_RELEASE = "https://github.com/asahi417/wikiart-crawler/releases/download/v0.0.0"


def _fetch(url: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Using existing archive: {destination}")
        return
    print(f"Fetching {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        with destination.open("wb") as target:
            shutil.copyfileobj(response, target)


def _safe_extract_zip(path: Path, root: Path):
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            target = (root / item.filename).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError(f"Unsafe path in {path}: {item.filename}")
        archive.extractall(root)


def _safe_extract_tar(path: Path, root: Path):
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (root / member.name).resolve()
            if not target.is_relative_to(root.resolve()) or member.issym() or member.islnk():
                raise ValueError(f"Unsafe archive entry: {member.name}")
        archive.extractall(root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["wikiart", "davis", "mosev2"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.dataset == "wikiart":
        for genre in ART_MOVEMENTS:
            archive = root / "archives" / f"{genre}.zip"
            if not args.extract_only:
                _fetch(f"{WIKIART_RELEASE}/{genre}.zip", archive)
            if not archive.exists():
                raise FileNotFoundError(archive)
            # Keep each movement in its own folder for class-aware splits.
            folder = root / genre.replace("_", "-")
            folder.mkdir(parents=True, exist_ok=True)
            _safe_extract_zip(archive, folder)
    elif args.dataset == "davis":
        archive = root / "DAVIS-2017-trainval-480p.zip"
        if not args.extract_only:
            _fetch(DAVIS_URL, archive)
        if not archive.exists():
            raise FileNotFoundError(archive)
        _safe_extract_zip(archive, root)
    else:
        # The hub hosts large compressed split archives. Only fetch with explicit intent.
        if not args.extract_only:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="FudanCVL/MOSEv2", repo_type="dataset",
                              local_dir=str(root), allow_patterns=["*.tar.gz"])
        for archive in sorted(root.glob("*.tar.gz")):
            split = archive.name.split(".")[0]
            folder = root / split
            folder.mkdir(parents=True, exist_ok=True)
            _safe_extract_tar(archive, folder)
    print(f"Finished preparing {args.dataset} under {root}")


if __name__ == "__main__":
    main()
