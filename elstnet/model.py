"""A motion-aligned selective state-space network for artistic video rendering."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .flow import MotionEstimator
from .ops import (
    flow_confidence, gradient_curvature, local_frequency_entropy,
    scale_flow, standardize_map, warp,
)


def _groups(channels: int) -> int:
    for count in (8, 4, 2, 1):
        if channels % count == 0:
            return count
    return 1


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1), nn.GroupNorm(_groups(cout), cout),
            nn.SiLU(), nn.Conv2d(cout, cout, 3, 1, 1),
            nn.GroupNorm(_groups(cout), cout)
        )
        self.short = nn.Conv2d(cin, cout, 1, stride) if (cin != cout or stride != 1) else nn.Identity()

    def forward(self, x):
        return F.silu(self.net(x) + self.short(x))


class PyramidEncoder(nn.Module):
    def __init__(self, widths: tuple[int, int, int]):
        super().__init__()
        a, b, c = widths
        self.stem = nn.Sequential(ConvBlock(3, a // 2, 2), ConvBlock(a // 2, a, 2))
        self.down1 = ConvBlock(a, b, 2)
        self.down2 = ConvBlock(b, c, 2)

    def forward(self, x):
        a = self.stem(x)
        b = self.down1(a)
        c = self.down2(b)
        return [a, b, c]


class SelectiveSequence(nn.Module):
    """A readable diagonal state-space scan with input-conditioned B, C, and step size."""

    def __init__(self, channels: int, state_size: int = 16):
        super().__init__()
        self.state_size = state_size
        self.channels = channels
        self.linear = nn.Linear(channels, state_size * 3)
        self.decay = nn.Parameter(torch.zeros(state_size))
        self.out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        dt_raw, b_raw, c_raw = self.linear(x).chunk(3, -1)
        dt = F.softplus(dt_raw).clamp(1e-4, 1.0)
        b = torch.tanh(b_raw)
        c = torch.tanh(c_raw)
        attenuation = F.softplus(self.decay) + 1e-4
        state = x.new_zeros(batch, channels, self.state_size)
        outputs = []
        # The short recurrence is useful on CPU and for debugging CUDA kernels.
        for i in range(length):
            a_i = torch.exp(-dt[:, i] * attenuation).unsqueeze(1)
            u_i = (dt[:, i] * b[:, i]).unsqueeze(1) * x[:, i].unsqueeze(-1)
            state = a_i * state + u_i
            outputs.append((state * c[:, i].unsqueeze(1)).sum(-1))
        return self.out(torch.stack(outputs, dim=1))


class SpatialScan(nn.Module):
    def __init__(self, channels: int, state_size: int = 16, backend: str = "torch"):
        super().__init__()
        if backend == "mamba":
            try:
                from mamba_ssm import Mamba
            except ImportError as exc:
                raise ImportError("Install mamba-ssm for the optimized scan backend") from exc
            self.sequence = Mamba(d_model=channels, d_state=state_size, d_conv=4, expand=2)
        elif backend == "torch":
            self.sequence = SelectiveSequence(channels, state_size)
        else:
            raise ValueError(f"Unknown scan backend: {backend}")
        self.weights = nn.Parameter(torch.zeros(4))
        self.norm = nn.GroupNorm(_groups(channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        horizontal = x.flatten(2).transpose(1, 2)
        vertical = x.transpose(-2, -1).flatten(2).transpose(1, 2)
        outs = []
        for tokens, flip, transpose in (
            (horizontal, False, False), (horizontal, True, False),
            (vertical, False, True), (vertical, True, True),
        ):
            sequence = tokens.flip(1) if flip else tokens
            output = self.sequence(sequence)
            if flip:
                output = output.flip(1)
            restored = output.transpose(1, 2).reshape(
                batch, channels, width, height
            ).transpose(-2, -1) if transpose else output.transpose(1, 2).reshape(
                batch, channels, height, width
            )
            outs.append(restored)
        weighted = sum(w * value for w, value in zip(self.weights.softmax(0), outs))
        return self.norm(weighted + x)


class SpectralTransport(nn.Module):
    """A stable style-conditioned matrix exponential with a low-rank correction."""

    def __init__(self, style_dim: int, channels: int, rank: int = 4):
        super().__init__()
        self.channels = channels
        self.rank = rank
        self.transition = nn.Linear(style_dim, channels + channels * rank)
        self.anchor = nn.Linear(style_dim, channels)
        nn.init.normal_(self.transition.weight, std=0.005)
        nn.init.zeros_(self.transition.bias)

    def forward(self, z: torch.Tensor, state: torch.Tensor, delta: float = 1.0):
        batch, channels, _, _ = state.shape
        params = self.transition(z)
        diagonal = F.softplus(params[:, :channels]) + 1e-4
        lowrank = params[:, channels:].reshape(batch, channels, self.rank)
        matrix = torch.diag_embed(diagonal) + lowrank @ lowrank.transpose(1, 2)
        # A negative symmetric generator keeps the transition norm below one.
        operator = torch.matrix_exp(-float(delta) * matrix.float()).to(state.dtype)
        anchor = self.anchor(z).to(state.dtype)
        transformed = torch.einsum("bij,bjhw->bihw", operator, state)
        eye = torch.eye(channels, device=state.device, dtype=state.dtype).unsqueeze(0)
        correction = torch.bmm(eye - operator, anchor.unsqueeze(-1)).view(batch, channels, 1, 1)
        return transformed + correction


class EventGate(nn.Module):
    def __init__(self, tile_size: int = 8, threshold: float = 0.5):
        super().__init__()
        self.tile_size = tile_size
        self.threshold = threshold
        self.raw_weights = nn.Parameter(torch.tensor([0.8, 0.5, 0.5, 0.8]))
        self.bias = nn.Parameter(torch.tensor(-1.0))

    def forward(self, current, previous, flow, confidence):
        aligned = warp(previous.detach(), flow)
        residual = (current - aligned).abs().mean(1, keepdim=True) / (
            current.abs().mean(1, keepdim=True) + 1e-4
        )
        curv = gradient_curvature(flow)
        entropy = local_frequency_entropy(current)
        weights = F.softplus(self.raw_weights)
        logit = (
            weights[0] * standardize_map(residual) +
            weights[1] * standardize_map(curv) +
            weights[2] * standardize_map(entropy) +
            weights[3] * (1 - confidence) + self.bias
        )
        prob = logit.sigmoid()
        tile = self.tile_size
        h, w = prob.shape[-2:]
        pooled = F.avg_pool2d(prob, tile, stride=tile, ceil_mode=True)
        hard_tiles = pooled > self.threshold
        # The packed boolean tiles control actual computation in the local scan.
        # A soft counterpart lets the gate receive gradients from its output blend.
        hard = F.interpolate(hard_tiles.float(), size=(h, w), mode="nearest")
        mask = hard.detach() - prob.detach() + prob
        return mask, hard_tiles, prob, residual


class PackedTileScan(nn.Module):
    """Scan selected blocks in a batch rather than scanning the entire image."""

    def __init__(self, channels: int, state_size: int, tile: int, halo: int,
                 backend: str):
        super().__init__()
        self.scan = SpatialScan(channels, state_size, backend)
        self.tile, self.halo = tile, halo

    def forward(self, x: torch.Tensor, predicted: torch.Tensor,
                active: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tile, halo = self.tile, self.halo
        hp = (height + tile - 1) // tile * tile
        wp = (width + tile - 1) // tile * tile
        # Each patch includes a small neighborhood; only the center is written back.
        padded = F.pad(x + predicted, (halo, halo + wp - width, halo, halo + hp - height))
        windows = F.unfold(padded, kernel_size=tile + 2 * halo, stride=tile)
        number = windows.shape[-1]
        windows = windows.transpose(1, 2).reshape(
            batch * number, channels, tile + 2 * halo, tile + 2 * halo
        )
        selections = active.flatten().bool()
        if selections.numel() != batch * number:
            raise RuntimeError("Tile selection dimensions do not match the feature map")
        result = predicted.new_zeros((batch * number, channels, tile, tile))
        if selections.any():
            scanned = self.scan(windows[selections])
            center = scanned[:, :, halo:halo + tile, halo:halo + tile]
            result[selections] = center
        result = result.reshape(batch, number, channels * tile * tile).transpose(1, 2)
        dense = F.fold(result, output_size=(hp, wp), kernel_size=tile, stride=tile)
        return dense[:, :, :height, :width]


@dataclass
class RenderOutput:
    images: torch.Tensor
    gate_probabilities: list
    gate_residuals: list
    frame_states: list
    motion: list
    confidence: list
    active_ratios: list
    carry: dict


class ELSTNet(nn.Module):
    def __init__(self, widths=(64, 96, 128), state_size=16, rank=4,
                 keyframe_interval=8, threshold=0.5, tile_size=8, halo=2,
                 flow_backend="farneback", pretrained_flow=True,
                 scan_backend="torch", mode="full"):
        super().__init__()
        self.widths = tuple(widths)
        self.interval = int(keyframe_interval)
        self.mode = mode
        self.content = PyramidEncoder(self.widths)
        self.style = PyramidEncoder(self.widths)
        self.flow = MotionEstimator(flow_backend, pretrained_flow)
        self.gamma = nn.ModuleList([nn.Linear(c, c) for c in widths])
        self.beta = nn.ModuleList([nn.Linear(c, c) for c in widths])
        self.scans = nn.ModuleList([SpatialScan(c, state_size, scan_backend) for c in widths])
        self.local = nn.ModuleList([
            PackedTileScan(c, state_size, tile_size, halo, scan_backend) for c in widths
        ])
        self.transports = nn.ModuleList([SpectralTransport(c, c, rank) for c in widths])
        self.gates = nn.ModuleList([EventGate(tile_size, threshold) for _ in widths])
        self.projections = nn.ModuleList([nn.Conv2d(c, 64, 1) for c in widths])
        self.decoder = nn.Sequential(
            ConvBlock(64 * 3, 96), ConvBlock(96, 64),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def _decode(self, states, size):
        base_size = states[0].shape[-2:]
        unified = [F.interpolate(project(s), size=base_size, mode="bilinear",
                                 align_corners=False)
                   for project, s in zip(self.projections, states)]
        output = self.decoder(torch.cat(unified, 1))
        return torch.sigmoid(F.interpolate(output, size=size, mode="bilinear",
                                           align_corners=False))

    def forward(self, video: torch.Tensor, style: torch.Tensor,
                return_details: bool = False, carry: dict | None = None) -> torch.Tensor | RenderOutput:
        if video.ndim != 5 or style.ndim != 4:
            raise ValueError("Expected video [B,T,3,H,W] and style [B,3,H,W]")
        batch, length, channels, height, width = video.shape
        if length < 1 or channels != 3 or min(height, width) < 32:
            raise ValueError("Video needs at least one RGB frame of at least 32 pixels")
        if style.shape[0] == 1 and batch > 1:
            style = style.expand(batch, -1, -1, -1)
        if style.shape[0] != batch:
            raise ValueError("Style batch must match the video batch")
        z = [feature.mean((-2, -1)) for feature in self.style(style)]
        frames, all_states, gate_probs, gate_residuals = [], [], [], []
        flow_items, confidences, active_ratios = [], [], []
        previous_features = carry["features"] if carry else None
        previous_states = carry["states"] if carry else None
        previous_frame = carry["frame"] if carry else None
        start_index = int(carry["next_index"]) if carry else 0
        for t in range(length):
            frame = video[:, t]
            features = self.content(frame)
            conditioned = [
                f * (1 + g(zz)[:, :, None, None]) + b(zz)[:, :, None, None]
                for f, zz, g, b in zip(features, z, self.gamma, self.beta)
            ]
            keyframe = (start_index + t) % self.interval == 0 or self.mode == "full_scan"
            back = forward = conf_full = None
            if not keyframe:
                back, forward = self.flow.bidirectional(frame, previous_frame)
                conf_full = flow_confidence(back, forward)
            states, probs, residuals, frame_active = [], [], [], []
            for level, (x, previous, prev_f, zz) in enumerate(
                zip(conditioned, previous_states or [None] * 3,
                    previous_features or [None] * 3, z)
            ):
                if keyframe:
                    state = self.scans[level](x)
                    probs.append(None)
                    residuals.append(None)
                    frame_active.append(None)
                else:
                    if self.mode == "no_transport":
                        # Start from zero when temporal alignment is disabled.
                        predicted = torch.zeros_like(previous)
                        flow = torch.zeros(batch, 2, *x.shape[-2:], device=x.device, dtype=x.dtype)
                        confidence = torch.zeros(batch, 1, *x.shape[-2:], device=x.device, dtype=x.dtype)
                    else:
                        flow = scale_flow(back, x.shape[-2:])
                        confidence = F.interpolate(conf_full, size=x.shape[-2:], mode="bilinear",
                                                    align_corners=False)
                        aligned_state = warp(previous, flow)
                        if self.mode == "no_spectral":
                            predicted = confidence * aligned_state
                        else:
                            predicted = confidence * self.transports[level](zz, aligned_state)
                    mask, active, probability, residual = self.gates[level](
                        x, prev_f, flow, confidence
                    )
                    if self.mode in ("no_gate", "flow_full"):
                        active = torch.ones_like(active)
                        mask = torch.ones_like(mask)
                    elif self.mode == "fixed_sparse":
                        # Use a deterministic spatial lattice with the same tile layout.
                        ids = torch.arange(active.numel(), device=active.device).reshape(active.shape)
                        active = ids.remainder(4) == 0
                        mask = F.interpolate(active.float(), size=x.shape[-2:], mode="nearest")
                    elif self.mode == "no_transport":
                        active = torch.ones_like(active)
                        mask = torch.ones_like(mask)
                    frame_active.append(active.float().mean())
                    update = self.local[level](x, predicted, active)
                    state = predicted * (1 - mask) + update * mask
                    probs.append(probability)
                    residuals.append(residual)
                states.append(state)
            frames.append(self._decode(states, (height, width)))
            if return_details:
                all_states.append(states)
                gate_probs.append(probs)
                gate_residuals.append(residuals)
                flow_items.append((back, forward))
                confidences.append(conf_full)
                active_ratios.append(frame_active)
            previous_features, previous_states, previous_frame = conditioned, states, frame
        image_sequence = torch.stack(frames, dim=1)
        if not return_details:
            return image_sequence
        context = {"features": previous_features, "states": previous_states,
                   "frame": previous_frame, "next_index": start_index + length}
        return RenderOutput(image_sequence, gate_probs, gate_residuals,
                            all_states, flow_items, confidences, active_ratios, context)
