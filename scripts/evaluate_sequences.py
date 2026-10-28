#!/usr/bin/env python3
"""Run one art reference across complete videos and export per-sequence measurements."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from elstnet.cli import load_checkpoint, make_model
from elstnet.data import VideoSequences, read_manifest, write_manifest
from elstnet.metrics import temporal_warp_error
from elstnet.utils import default_device, load_config, rgb_tensor, save_json, save_rgb


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video-manifest", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--style", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", default="full")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.chunk < 1:
        raise ValueError("Chunk size must be positive")
    device = default_device(args.device)
    model = make_model(load_config(args.config), device, args.mode).eval()
    load_checkpoint(model, args.checkpoint, device)
    style = rgb_tensor(args.style, args.size, device)[None]
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    pair_rows, sequence_rows = [], []
    sequences = VideoSequences(args.video_manifest, args.video_root, args.split, args.size)
    for (dataset_name, name), original in sequences:
        original = original.to(device)[None]
        output_frames, probabilities = [], []
        carry = None
        sync(device)
        start = time.perf_counter()
        for offset in range(0, original.shape[1], args.chunk):
            packet = original[:, offset:offset + args.chunk]
            out = model(packet, style, return_details=True, carry=carry)
            carry = out.carry
            output_frames.append(out.images)
            probabilities.extend([float(value) for frame in out.active_ratios
                                  for value in frame if value is not None])
        sync(device)
        latency = (time.perf_counter() - start) * 1000 / original.shape[1]
        styled = torch.cat(output_frames, dim=1)
        score = temporal_warp_error(styled, original, model.flow)
        sequence_rows.append({"dataset": dataset_name, "sequence": name, "split": args.split,
                              "model": args.mode, "frames": original.shape[1],
                              "mean_frame_ms": latency, "warp_error": score["warp_error"],
                              "mean_update_ratio": float(np.mean(probabilities))
                              if probabilities else ""})
        rows = [row for row in read_manifest(args.video_manifest)
                if row["dataset"] == dataset_name and row["sequence"] == name
                and row["split"] == args.split]
        rows.sort(key=lambda r: r["frame"])
        for row, stylized in zip(rows, styled[0]):
            file = folder / "frames" / dataset_name / name / (row["frame"] + ".png")
            save_rgb(stylized, file)
            pair_rows.append({"content": str(Path(args.video_root) / row["path"]),
                              "styled": str(file), "style": str(Path(args.style).resolve()),
                              "dataset": dataset_name, "sequence": name})
        print(json.dumps(sequence_rows[-1]))
    if not sequence_rows:
        raise ValueError(f"No sequences for split {args.split}")
    write_manifest(folder / "sequence_metrics.csv", sequence_rows,
                   ["dataset", "sequence", "split", "model", "frames",
                    "mean_frame_ms", "warp_error", "mean_update_ratio"])
    write_manifest(folder / "artfid_pairs.csv", pair_rows,
                   ["content", "styled", "style", "dataset", "sequence"])
    save_json(folder / "summary.json", {
        "video_count": len(sequence_rows),
        "mean_sequence_frame_ms": float(np.mean([r["mean_frame_ms"] for r in sequence_rows])),
        "mean_sequence_warp_error": float(np.mean([r["warp_error"] for r in sequence_rows])),
        "evaluation_split": args.split,
    })


if __name__ == "__main__":
    main()
