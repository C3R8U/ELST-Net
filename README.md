# ELST-Net

**Event-Triggered Lagrangian State Transport for Artistic Video Stylization**

ELST-Net pairs a multiscale selective state-space backbone with optical-flow-aligned feature transport and sparse local updates. The renderer accepts an RGB video and a single reference painting. It returns a stylized video with the original frame order preserved.

The repository contains PyTorch model code, data indexes, training and inference entry points, metric implementations, parameter configurations, and automated tests. Dataset archives, pretrained checkpoints, generated images, and experiment logs are intentionally not included.

## Architecture

- A three-level pyramid extracts content features at 1/4, 1/8, and 1/16 resolution. A separate style encoder supplies conditioning vectors.
- Four spatial scan directions aggregate long-range spatial context. `scan_backend: mamba` uses the optional CUDA implementation; `scan_backend: torch` uses a readable diagonal selective recurrence.
- Optical flow maps previous-frame feature states into current-frame coordinates. Forward/backward agreement downweights occlusions and invalid samples.
- A style-conditioned matrix exponential with a negative semidefinite generator limits growth during state transport. A low-rank term couples feature channels.
- Prediction residual, motion curvature, and directional frequency entropy form an event gate. Only selected tiles enter the local state-space scan.
- Perceptual content/style loss, temporal warping, oriented complex wavelet phase, hidden-state cycle consistency, and sparse-update regularization guide optimization.

The propagated state is a **spatial latent feature map produced by the selective scan**. It is not an internal recurrent cache of the optional Mamba CUDA kernel. Phase loss uses compact complex oriented wavelet filters; this implementation does not require an external dual-tree wavelet package.

## Requirements

Python 3.10 or newer, PyTorch, TorchVision, OpenCV, NumPy, SciPy, Pillow, and PyYAML are required. Install with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,metrics,download]'
```

For the CUDA configuration in `configs/default.yaml`, install `mamba-ssm` using the instructions for your PyTorch and CUDA combination. The default optical-flow backend is pretrained RAFT-Small; its weights are downloaded by TorchVision on first use. A CPU-only debug configuration is available in `configs/cpu_debug.yaml` using OpenCV Farneback flow and a pure PyTorch scan.

```bash
pytest -q
elst --help
```

**Runtime note:** the pure PyTorch diagonal scan intentionally emphasizes transparent recurrence behavior and is not optimized for large high-resolution sequences. No real-time throughput is claimed without measuring the chosen backend and device.

## Datasets

All images and videos remain under the licensing terms of their original providers. Do not commit the downloaded datasets into this repository.

| Dataset | Original source | Role |
| --- | --- | --- |
| WikiArt General | https://github.com/asahi417/wikiart-image-dataset | Style images and evaluation references |
| DAVIS 2017 Semi-supervised TrainVal 480p | https://davischallenge.org/davis2017/code.html | Training videos and held-out sequence evaluation |
| MOSEv2 | https://huggingface.co/datasets/FudanCVL/MOSEv2 | Independent difficult-scene evaluation |

WikiArt **General** contains 15 art movements. The Face variant is not used. DAVIS should retain the `ImageSets/2017/train.txt` and `val.txt` lists; the official evaluation sequences must not enter optimization. MOSEv2 data should remain outside training and model selection when it is used as a held-out external test.

The MOSEv2 data card states that data use is limited to non-commercial research. Check the latest dataset card and upstream usage conditions before downloading, publishing derived data, or distributing any model trained on restricted data.

### Fetch the originals

```bash
python scripts/download_data.py wikiart --root data/wikiart-general
python scripts/download_data.py davis --root data/DAVIS
python scripts/download_data.py mosev2 --root data/MOSEv2
```

These commands fetch large upstream archives, and MOSEv2 is particularly large. Prepare sufficient disk space and review the source licenses. Downloads are never attempted during project installation or tests. Existing archives can be extracted with `--extract-only`.

Expected data layout:

```text
data/
  wikiart-general/
    impressionism/*.jpg
    expressionism/*.jpg
    ...
  DAVIS/
    JPEGImages/480p/<sequence>/*.jpg
    Annotations/480p/<sequence>/*.png
    ImageSets/2017/train.txt
    ImageSets/2017/val.txt
  MOSEv2/
    train/JPEGImages/<sequence>/*.jpg
    valid/JPEGImages/<sequence>/*.jpg
```

### Index local files

```bash
elst index wikiart --root data/wikiart-general --output data/manifests/wikiart.csv --seed 42
elst index davis --root data/DAVIS --output data/manifests/davis.csv
elst index mosev2 --root data/MOSEv2 --output data/manifests/mosev2.csv
```

The WikiArt index removes cross-split collisions caused by byte-identical paintings. An optional CSV with `path,artist_id` columns also groups paintings by their known artist. Without artist metadata, artist-disjoint splitting cannot be guaranteed. The split is deterministic by group identity; preserve the resulting CSV for all later runs.

The DAVIS manifest follows official train/val sequences. The MOSEv2 index expects extracted `train` and `valid` archives; use `--splits valid` to index the held-out portion only.

## Train

```bash
elst train --config configs/default.yaml --device cuda --seed 42 --output runs/seed_42
```

Outputs are written to the selected run directory, including `latest.pt` and `train.jsonl`. A checkpoint stores optimizer and scheduler states, random seed, and config. An interrupted run can be resumed with `--resume runs/seed_42/latest.pt`.

For a short CPU test with already indexed data:

```bash
elst train --config configs/cpu_debug.yaml --no-pretrained-vgg \
  --device cpu --output runs/cpu_debug
```

The CPU configuration is for functional checks, not for any final timing or visual-quality comparison. The usual training configuration uses pretrained VGG-19 perceptual features and pretrained RAFT-Small optical flow. The first use of these models requires their publicly hosted weights.

Train separate model variants when comparing module contributions:

```bash
bash scripts/run_ablations.sh configs/default.yaml
bash scripts/run_seeds.sh configs/default.yaml
```

`full_scan` independently processes each frame; `no_gate` scans every motion-aligned local tile; `fixed_sparse` scans a fixed tile lattice; `no_transport` removes optical-flow state alignment; `no_spectral` removes the style-conditioned spectral operator. `no_phase` and `no_cycle` disable the corresponding objectives during training. These are controlled **internal** comparisons. External architectures such as CAP-VSTNet require their original authors' implementations and checkpoints; they are not silently substituted with the internal baselines.

## Render videos

Input frames are sorted by filename. Keep frame names zero-padded so the order is chronological. The style reference is an RGB painting image.

```bash
elst render --config configs/default.yaml --checkpoint runs/seed_42/latest.pt \
  --frames data/DAVIS/JPEGImages/480p/<sequence> \
  --style data/wikiart-general/impressionism/<painting>.jpg \
  --size 512 --chunk-size 8 --device cuda --output outputs/<sequence>
```

Chunking carries the previous feature maps and global keyframe index forward. It does not reset state at chunk boundaries. Rendering saves only the generated PNG frames in the chosen output directory; no outputs are shipped with the project.

## Measure latency and visual quality

```bash
elst benchmark --config configs/default.yaml --checkpoint runs/seed_42/latest.pt \
  --frames data/DAVIS/JPEGImages/480p/<sequence> --style <style_image> \
  --size 512 --device cuda --warmup 10 --repeats 50 --output results/timing.json

elst temporal --original <original_frames> --styled <generated_frames> \
  --flow-backend raft --size 512 --device cuda --output results/temporal.json
```

The benchmark reports the full model time, including optical flow. GPU synchronization surrounds each timed run. GPU peak allocation is reported separately and is not confused with total process or reserved memory. External capture, video decoding, and display are not included.

To evaluate complete datasets, render entire sequences and export a per-sequence metrics CSV along with content/style/generated image pairs:

```bash
python scripts/evaluate_sequences.py --config configs/default.yaml \
  --checkpoint runs/seed_42/latest.pt \
  --video-manifest data/manifests/davis.csv --video-root data/DAVIS \
  --style <style_image> --split val --chunk 8 --size 512 \
  --output results/davis_eval

python scripts/evaluate_sequences.py --config configs/default.yaml \
  --checkpoint runs/seed_42/latest.pt \
  --video-manifest data/manifests/mosev2.csv --video-root data/MOSEv2 \
  --style <style_image> --split valid --chunk 8 --size 512 \
  --output results/mose_eval
```

ArtFID combines content LPIPS and FID between generated and reference painting features. The `artfid_pairs.csv` file provides `content,styled,style` columns, with one row per image pair. For a representative ArtFID, collect a sufficiently large set of independent paintings and generated images; a few repeated reference artworks are not enough to estimate a reliable high-dimensional covariance.

```bash
elst artfid --pairs results/davis_eval/artfid_pairs.csv --device cuda \
  --output results/artfid.json

elst stats --pairs results/paired_scores.csv \
  --baseline baseline_warp --candidate elst_warp \
  --output results/paired_statistics.json

python scripts/sequence_analysis.py --pairs results/paired_scores.csv \
  --baseline baseline_warp --candidate elst_warp \
  --resamples 10000 --output results/sequence_analysis.json
```

`artfid` requires the optional LPIPS and pytorch-fid packages. Feature FID estimates are sample-size dependent. Report the exact pairing policy, pretrained feature weights, number of reference artworks, and style-wise aggregation rule. The `stats` command treats each row as one independent paired unit; use per-video or per-training-run scores rather than thousands of correlated neighboring frames.

## Input robustness checks

```bash
python scripts/robustness.py --config configs/default.yaml \
  --checkpoint runs/seed_42/latest.pt --frames <frames> --style <style_image> \
  --kind noise --intensity 0.02 --output results/noise.json

python scripts/robustness.py --config configs/default.yaml \
  --checkpoint runs/seed_42/latest.pt --frames <frames> --style <style_image> \
  --kind jpeg --intensity 50 --output results/jpeg.json
```

The script also supports `frame_stride` and `none`. It computes motion-compensated temporal error on generated output, not a preset score. It does not infer cross-dataset generalization from a single input sequence.

## Run artifacts and reporting

Keep these items together for each trained model:

- Checkpoint and frozen environment versions.
- Seed, configuration file, code commit, and dataset index CSVs.
- Whole-sequence timing traces and score files.
- Exact visual-quality pairing lists, full split manifests, and aggregation scripts.
- Statistical summary with confidence intervals and number of independent sequences.

Only report results generated by an executed model. No performance numbers, pretrained ELST-Net weights, or external baseline outputs are supplied here. Model performance depends on training, flow backend, scan implementation, data selection, and GPU hardware.

## Implementation boundaries

- The PyTorch selective scan fallback uses sequential updates and is slow on large feature maps; the optional Mamba CUDA backend is intended for GPU workloads.
- The state transport applies to spatial latent features rather than the private CUDA kernel cache.
- The current flow estimator is frozen and can fail with strong occlusion, motion blur, or large displacements.
- The wavelet phase regularizer uses compact complex oriented filters rather than a full dual-tree transform.
- Full-size inputs are resized to a square canvas. A production system can add aspect-preserving padding and coordinate restoration.
- External model comparisons need separately licensed implementations and a matched measurement protocol.
- The model must be trained before use. No accuracy or speed is guaranteed by the source tree alone.

## License

Code: MIT. Original dataset licenses and pretrained third-party model licenses remain with their respective publishers.
