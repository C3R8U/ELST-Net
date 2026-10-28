#!/usr/bin/env python3
"""Apply input shifts to videos and measure the model without preset score values."""

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from elstnet.cli import load_checkpoint, make_model
from elstnet.flow import MotionEstimator
from elstnet.metrics import temporal_warp_error
from elstnet.utils import default_device, load_config, rgb_tensor, save_json


def perturb(video, kind, intensity, seed=42):
    if kind == "none":
        return video
    if kind == "noise":
        generator = torch.Generator(device=video.device).manual_seed(seed)
        return (video + torch.randn(video.shape, generator=generator,
                                    device=video.device, dtype=video.dtype) * intensity).clamp(0, 1)
    if kind == "frame_stride":
        stride = int(intensity)
        if stride < 1:
            raise ValueError("Frame stride must be positive")
        return video[:, ::stride]
    if kind == "jpeg":
        result = []
        for frame in video[0]:
            pixels = (frame.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            buff = io.BytesIO()
            Image.fromarray(pixels).save(buff, format="JPEG", quality=int(intensity))
            buff.seek(0)
            image = np.asarray(Image.open(buff).convert("RGB"), dtype=np.float32).copy() / 255
            result.append(torch.from_numpy(image).permute(2, 0, 1))
        return torch.stack(result)[None].to(video.device)
    raise ValueError(f"Unsupported shift: {kind}")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--style", required=True)
    parser.add_argument("--kind", choices=["none", "noise", "jpeg", "frame_stride"], required=True)
    parser.add_argument("--intensity", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = default_device(args.device)
    model = make_model(load_config(args.config), device).eval()
    load_checkpoint(model, args.checkpoint, device)
    paths = sorted((p for p in Path(args.frames).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}), key=lambda p: p.name)
    if len(paths) < 2:
        raise ValueError("At least two input frames are required")
    source = torch.stack([rgb_tensor(path, args.size, device) for path in paths])[None]
    moved = perturb(source, args.kind, args.intensity)
    if moved.shape[1] < 2:
        raise ValueError("The time stride left too few frames")
    style = rgb_tensor(args.style, args.size, device)[None]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    output = model(moved, style, return_details=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - start) * 1000 / moved.shape[1]
    estimator = MotionEstimator(model.flow.backend, pretrained=True).to(device)
    temporal = temporal_warp_error(output.images, moved, estimator)
    active = [float(value) for frame in output.active_ratios for value in frame if value is not None]
    report = {"condition": args.kind, "intensity": args.intensity,
              "mean_frame_ms": elapsed, "warp_error": temporal["warp_error"],
              "mean_update_ratio": float(np.mean(active)) if active else None,
              "frames": moved.shape[1]}
    save_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
