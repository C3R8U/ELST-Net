"""Command line tools for data preparation, training, rendering, and scoring."""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import (
    StyleVideoDataset, VideoSequences, index_davis, index_mose,
    index_wikiart, read_manifest,
)
from .flow import MotionEstimator
from .losses import PerceptualFeatures, full_loss
from .metrics import ArtFIDMetric, paired_statistics, temporal_warp_error
from .model import ELSTNet
from .utils import default_device, load_config, rgb_tensor, save_json, save_rgb, seed_everything


def make_model(config, device, mode=None):
    settings = dict(config["model"])
    settings["mode"] = mode or settings.get("mode", "full")
    return ELSTNet(**settings).to(device)


def load_checkpoint(model, path, device):
    # File access is explicit; never fetch untrusted checkpoints automatically.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return checkpoint


def _config(args):
    return load_config(args.config)


def cmd_index(args):
    if args.dataset == "wikiart":
        report = index_wikiart(args.root, args.output, args.seed, args.artist_map)
    elif args.dataset == "davis":
        report = index_davis(args.root, args.output)
    else:
        report = index_mose(args.root, args.output, splits=tuple(args.splits))
    print(json.dumps(report, indent=2))


def _get_loader(config, split="train"):
    dataset = StyleVideoDataset(
        config["data"]["video_manifest"], config["data"]["video_root"],
        config["data"]["wiki_manifest"], config["data"]["wiki_root"],
        split=split, length=config["data"]["clip_length"],
        size=config["data"]["size"], stride=config["data"].get("stride", 1),
        wiki_split=config["data"].get("style_split") if split == "train" else None,
    )
    return DataLoader(dataset, batch_size=config["train"]["batch_size"],
                      shuffle=(split == "train"), num_workers=config["train"].get("workers", 0),
                      pin_memory=torch.cuda.is_available(), drop_last=False)


def cmd_train(args):
    config = _config(args)
    seed_everything(args.seed, config["train"].get("deterministic", False))
    device = default_device(args.device)
    model = make_model(config, device, mode=args.mode)
    extractor = PerceptualFeatures(pretrained=not args.no_pretrained_vgg).to(device).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"],
                                  weight_decay=config["train"].get("weight_decay", 0.01))
    max_steps = int(config["train"]["max_steps"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
    step = 0
    if args.resume:
        checkpoint = load_checkpoint(model, args.resume, device)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        step = int(checkpoint["step"])
    loader = _get_loader(config)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / "train.jsonl"
    model.train()
    # Every run writes its own metadata so different seeds stay separate.
    with log.open("a", encoding="utf-8") as sink:
        iterator = iter(loader)
        while step < max_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            video = batch["video"].to(device)
            style = batch["style"].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(video, style, return_details=True)
            enabled = {"content", "style", "temporal", "phase", "cycle", "gate"}
            if args.mode == "no_phase":
                enabled.remove("phase")
            if args.mode == "no_cycle":
                enabled.remove("cycle")
            loss, components = full_loss(output, video, style, extractor,
                                         config["loss_weights"], active_losses=enabled)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training objective at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           config["train"].get("gradient_clip", 1.0))
            optimizer.step()
            scheduler.step()
            step += 1
            if step % args.log_every == 0 or step == 1:
                record = {"step": step, "seed": args.seed, "mode": args.mode,
                          "loss": float(loss.detach()), **components}
                sink.write(json.dumps(record) + "\n")
                sink.flush()
                print(json.dumps(record))
            if step % args.save_every == 0 or step == max_steps:
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "step": step,
                            "seed": args.seed, "config": config}, directory / "latest.pt")


def _open_images(folder, size, device):
    images = sorted(p for p in Path(folder).iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        raise FileNotFoundError(f"No image frames found in {folder}")
    return images, torch.stack([rgb_tensor(p, size, device) for p in images])[None]


@torch.no_grad()
def cmd_render(args):
    config = _config(args)
    device = default_device(args.device)
    model = make_model(config, device, mode=args.mode).eval()
    load_checkpoint(model, args.checkpoint, device)
    paths, frames = _open_images(args.frames, args.size, device)
    style = rgb_tensor(args.style, args.size, device)[None]
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    # A carried state keeps the same keyframe schedule across neighboring chunks.
    carry = None
    chunk = args.chunk_size or len(paths)
    for start in range(0, len(paths), chunk):
        batch = frames[:, start:start + chunk]
        output = model(batch, style, return_details=True, carry=carry)
        carry = output.carry
        for path, tensor in zip(paths[start:start + chunk], output.images[0]):
            save_rgb(tensor, folder / (path.stem + ".png"))
    print(f"Saved {len(paths)} frames to {folder}")


@torch.no_grad()
def cmd_benchmark(args):
    config = _config(args)
    device = default_device(args.device)
    model = make_model(config, device, mode=args.mode).eval()
    load_checkpoint(model, args.checkpoint, device)
    _, video = _open_images(args.frames, args.size, device)
    style = rgb_tensor(args.style, args.size, device)[None]
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    for _ in range(args.warmup):
        model(video, style)
        synchronize()
    observations = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.repeats):
        synchronize()
        begin = time.perf_counter()
        model(video, style)
        synchronize()
        observations.append((time.perf_counter() - begin) * 1000 / video.shape[1])
    output = {"mode": args.mode, "height": args.size, "width": args.size,
              "frames": video.shape[1], "mean_frame_ms": sum(observations) / len(observations),
              "per_run_frame_ms": observations,
              "peak_gpu_gb": (torch.cuda.max_memory_allocated(device) / 1024**3)
              if device.type == "cuda" else None,
              "timing_scope": "all model operations including optical flow"}
    save_json(args.output, output)
    print(json.dumps(output, indent=2))


@torch.no_grad()
def cmd_temporal(args):
    device = default_device(args.device)
    estimator = MotionEstimator(args.flow_backend, pretrained=True).to(device).eval()
    source_paths, original = _open_images(args.original, args.size, device)
    generated_paths, styled = _open_images(args.styled, args.size, device)
    if [p.stem for p in source_paths] != [p.stem for p in generated_paths]:
        raise ValueError("Original and stylized frame names do not match")
    output = temporal_warp_error(styled, original, estimator)
    save_json(args.output, output)
    print(json.dumps({key: value for key, value in output.items() if key != "per_pair"}, indent=2))


def cmd_artfid(args):
    entries = read_manifest(args.pairs)
    metric = ArtFIDMetric(args.device)
    output = metric.evaluate(entries, args.size)
    save_json(args.output, output)
    print(json.dumps(output, indent=2))


def cmd_stats(args):
    entries = read_manifest(args.pairs)
    score = paired_statistics([float(row[args.baseline]) for row in entries],
                              [float(row[args.candidate]) for row in entries])
    save_json(args.output, score)
    print(json.dumps(score, indent=2))


def build_parser():
    p = argparse.ArgumentParser(prog="elst", description="Motion-aware artistic video renderer")
    sub = p.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index", help="Build an index for a local dataset")
    index.add_argument("dataset", choices=["wikiart", "davis", "mosev2"])
    index.add_argument("--root", required=True)
    index.add_argument("--output", required=True)
    index.add_argument("--artist-map")
    index.add_argument("--seed", type=int, default=42)
    index.add_argument("--splits", nargs="+", default=["train", "valid"])
    index.set_defaults(func=cmd_index)
    train = sub.add_parser("train", help="Fit model parameters from paired video/style clips")
    train.add_argument("--config", default="configs/default.yaml")
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--mode", default="full", choices=[
        "full", "full_scan", "flow_full", "fixed_sparse", "no_gate",
        "no_transport", "no_spectral", "no_phase", "no_cycle"])
    train.add_argument("--resume")
    train.add_argument("--output", default="runs/default")
    train.add_argument("--log-every", type=int, default=10)
    train.add_argument("--save-every", type=int, default=500)
    train.add_argument("--no-pretrained-vgg", action="store_true")
    train.set_defaults(func=cmd_train)
    render = sub.add_parser("render", help="Stylize a folder of ordered image frames")
    render.add_argument("--config", default="configs/default.yaml")
    render.add_argument("--checkpoint", required=True)
    render.add_argument("--frames", required=True)
    render.add_argument("--style", required=True)
    render.add_argument("--output", required=True)
    render.add_argument("--size", type=int, default=512)
    render.add_argument("--chunk-size", type=int, default=0)
    render.add_argument("--device", default="auto")
    render.add_argument("--mode", default="full")
    render.set_defaults(func=cmd_render)
    bench = sub.add_parser("benchmark", help="Time a whole input sequence")
    bench.add_argument("--config", default="configs/default.yaml")
    bench.add_argument("--checkpoint", required=True)
    bench.add_argument("--frames", required=True)
    bench.add_argument("--style", required=True)
    bench.add_argument("--size", type=int, default=512)
    bench.add_argument("--mode", default="full")
    bench.add_argument("--device", default="auto")
    bench.add_argument("--warmup", type=int, default=2)
    bench.add_argument("--repeats", type=int, default=10)
    bench.add_argument("--output", required=True)
    bench.set_defaults(func=cmd_benchmark)
    temp = sub.add_parser("temporal", help="Compute original-flow-aligned temporal error")
    temp.add_argument("--original", required=True)
    temp.add_argument("--styled", required=True)
    temp.add_argument("--flow-backend", default="raft", choices=["raft", "farneback"])
    temp.add_argument("--size", type=int, default=512)
    temp.add_argument("--device", default="auto")
    temp.add_argument("--output", required=True)
    temp.set_defaults(func=cmd_temporal)
    art = sub.add_parser("artfid", help="Score paired content, style, and generated images")
    art.add_argument("--pairs", required=True)
    art.add_argument("--size", type=int, default=512)
    art.add_argument("--device", default="auto")
    art.add_argument("--output", required=True)
    art.set_defaults(func=cmd_artfid)
    stats = sub.add_parser("stats", help="Confidence intervals for paired sequence scores")
    stats.add_argument("--pairs", required=True)
    stats.add_argument("--baseline", required=True)
    stats.add_argument("--candidate", required=True)
    stats.add_argument("--output", required=True)
    stats.set_defaults(func=cmd_stats)
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
