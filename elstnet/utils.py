"""Small utilities shared by training, inference, and evaluation."""

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Keep random sampling under control across workers and hardware."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_json(path: str | Path, obj: dict | list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rgb_tensor(path: str | Path, size: int, device=None) -> torch.Tensor:
    """Read RGB bytes and resize without cropping the subject out of frame."""
    with Image.open(path) as source:
        image = source.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    data = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(data).permute(2, 0, 1).to(device=device)


def save_rgb(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (tensor.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).round()
    Image.fromarray(data.astype(np.uint8), mode="RGB").save(path)


def default_device(value: str = "auto") -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def set_worker_environment() -> None:
    os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(1 << 32))
