"""Content, style, temporal, phase, and state consistency objectives."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .ops import scale_flow, warp


class PerceptualFeatures(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19
        weights = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = vgg19(weights=weights).features.eval()
        self.features = nn.ModuleList([
            backbone[:4], backbone[4:9], backbone[9:18], backbone[18:27]
        ])
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, x):
        x = (x - self.mean) / self.std
        outputs = []
        for layer in self.features:
            x = layer(x)
            outputs.append(x)
        return outputs


def _statistics(x):
    mean = x.mean((-2, -1), keepdim=True)
    std = (x.var((-2, -1), keepdim=True, unbiased=False) + 1e-6).sqrt()
    return mean, std


def content_style_loss(styled, content, style, extractor: PerceptualFeatures):
    """Match content activations and low-order style statistics."""
    generated = extractor(styled)
    with torch.no_grad():
        targets = extractor(content)
        references = extractor(style)
    content_loss = sum((g - c).abs().mean() for g, c in zip(generated, targets))
    style_loss = styled.new_zeros(())
    for gen, ref in zip(generated, references):
        gm, gs = _statistics(gen)
        rm, rs = _statistics(ref)
        style_loss = style_loss + (gm - rm).abs().mean() + (gs - rs).abs().mean()
    return content_loss, style_loss


def _complex_wavelet_responses(images, orientations=(0.0, math.pi / 3, 2 * math.pi / 3)):
    """Complex oriented wavelet pairs provide local phase without an extra package."""
    positions = torch.arange(-3, 4, device=images.device, dtype=images.dtype)
    yy, xx = torch.meshgrid(positions, positions, indexing="ij")
    real, imag = [], []
    for angle in orientations:
        projected = xx * math.cos(angle) + yy * math.sin(angle)
        envelope = torch.exp(-(xx.square() + yy.square()) / (2 * 2.0 ** 2))
        real.append(envelope * torch.cos(projected * math.pi / 2))
        imag.append(envelope * torch.sin(projected * math.pi / 2))
    kernels_r = torch.stack(real)[:, None]
    kernels_i = torch.stack(imag)[:, None]
    gray = images.mean(1, keepdim=True)
    return F.conv2d(gray, kernels_r, padding=3), F.conv2d(gray, kernels_i, padding=3)


def phase_loss(current, previous, backward_flow, confidence):
    """Compare oriented complex responses after motion alignment."""
    aligned = warp(previous, backward_flow)
    cr, ci = _complex_wavelet_responses(current)
    pr, pi = _complex_wavelet_responses(aligned)
    cosine = (cr * pr + ci * pi) / (
        torch.sqrt(cr.square() + ci.square() + 1e-6) *
        torch.sqrt(pr.square() + pi.square() + 1e-6)
    )
    return (confidence * (1 - cosine.clamp(-1, 1))).mean()


def temporal_consistency(current, previous, backward_flow, confidence):
    delta = (current - warp(previous, backward_flow)).abs().mean(1, keepdim=True)
    return (confidence * delta).sum() / confidence.sum().clamp_min(1e-6)


def state_cycle_loss(previous_states, current_states, forward, confidence):
    total = current_states[0].new_zeros(())
    for old, current in zip(previous_states, current_states):
        flow = scale_flow(forward, old.shape[-2:])
        conf = F.interpolate(confidence, size=old.shape[-2:], mode="bilinear",
                             align_corners=False)
        total = total + (conf * (old - warp(current, flow)).abs().mean(1, keepdim=True)).mean()
    return total / len(previous_states)


def full_loss(out, video, style, extractor, weights, active_losses=None):
    """Keep losses independently switchable for controlled training runs."""
    active_losses = set(active_losses or ["content", "style", "temporal", "phase", "cycle", "gate"])
    batch, length, channels, height, width = video.shape
    stylized = out.images.reshape(batch * length, 3, height, width)
    contents = video.reshape(batch * length, 3, height, width)
    styles = style[:, None].expand(batch, length, 3, height, width).reshape_as(contents)
    content, style_term = content_style_loss(stylized, contents, styles, extractor)
    terms = {"content": content, "style": style_term}
    terms.update({name: video.new_zeros(()) for name in ("temporal", "phase", "cycle", "gate")})
    count = 0
    for t in range(1, length):
        back, forward = out.motion[t]
        if back is None:
            continue
        from .ops import flow_confidence
        conf = flow_confidence(back, forward)
        if "temporal" in active_losses:
            terms["temporal"] = terms["temporal"] + temporal_consistency(
                out.images[:, t], out.images[:, t - 1], back, conf
            )
        if "phase" in active_losses:
            terms["phase"] = terms["phase"] + phase_loss(
                out.images[:, t], out.images[:, t - 1], back, conf
            )
        if "cycle" in active_losses:
            terms["cycle"] = terms["cycle"] + state_cycle_loss(
                out.frame_states[t - 1], out.frame_states[t], forward, conf
            )
        if "gate" in active_losses:
            for probability, residual in zip(out.gate_probabilities[t], out.gate_residuals[t]):
                if probability is not None:
                    # Sparse updates cost compute; missed high-error blocks cost quality.
                    terms["gate"] = terms["gate"] + (
                        probability.mean() + (1 - probability).mul(residual.detach()).mean()
                    ) / len(out.gate_probabilities[t])
        count += 1
    if count:
        for name in ("temporal", "phase", "cycle", "gate"):
            terms[name] = terms[name] / count
    objective = sum(weights.get(key, 0.0) * term for key, term in terms.items()
                    if key in active_losses)
    return objective, {key: value.detach().item() for key, value in terms.items()}
