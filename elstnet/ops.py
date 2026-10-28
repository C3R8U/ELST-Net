"""Geometric alignment and compact spatial frequency features."""

import torch
import torch.nn.functional as F


def warp(x: torch.Tensor, flow: torch.Tensor, return_valid: bool = False):
    """Sample source x at the coordinates pointed to by target-to-source flow."""
    b, _, h, w = x.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=x.device, dtype=x.dtype),
        torch.arange(w, device=x.device, dtype=x.dtype), indexing="ij"
    )
    source_x = xx[None] + flow[:, 0]
    source_y = yy[None] + flow[:, 1]
    valid = ((source_x >= 0) & (source_x <= w - 1) &
             (source_y >= 0) & (source_y <= h - 1)).unsqueeze(1)
    nx = (2 * source_x / max(w - 1, 1)) - 1 if w > 1 else source_x * 0
    ny = (2 * source_y / max(h - 1, 1)) - 1 if h > 1 else source_y * 0
    grid = torch.stack((nx, ny), dim=-1)
    result = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return (result, valid.to(x.dtype)) if return_valid else result


def scale_flow(flow: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resample a motion field and keep displacement in pixel coordinates."""
    h, w = flow.shape[-2:]
    oh, ow = size
    out = F.interpolate(flow, size=size, mode="bilinear", align_corners=True)
    out = out.clone()
    out[:, 0] *= ow / max(w, 1)
    out[:, 1] *= oh / max(h, 1)
    return out


def flow_confidence(backward: torch.Tensor, forward: torch.Tensor,
                    sigma: float = 3.0) -> torch.Tensor:
    """Reject points where opposite flow directions disagree or leave the image."""
    f_at_b, valid = warp(forward, backward, return_valid=True)
    error = (backward + f_at_b).square().sum(dim=1, keepdim=True)
    return valid * torch.exp(-error / (sigma * sigma))


def gradient_curvature(flow: torch.Tensor) -> torch.Tensor:
    """A smooth turning-rate cue highlights fast-changing motion boundaries."""
    speed = torch.linalg.vector_norm(flow, dim=1, keepdim=True)
    speed = F.avg_pool2d(speed, 3, stride=1, padding=1)
    dx = F.pad(speed[:, :, :, 2:] - speed[:, :, :, :-2], (1, 1, 0, 0)) * 0.5
    dy = F.pad(speed[:, :, 2:, :] - speed[:, :, :-2, :], (0, 0, 1, 1)) * 0.5
    magnitude = torch.sqrt(dx.square() + dy.square() + 1e-6)
    nx, ny = dx / magnitude, dy / magnitude
    ddx = F.pad(nx[:, :, :, 2:] - nx[:, :, :, :-2], (1, 1, 0, 0)) * 0.5
    ddy = F.pad(ny[:, :, 2:, :] - ny[:, :, :-2, :], (0, 0, 1, 1)) * 0.5
    return (ddx + ddy).abs()


def local_frequency_entropy(feature: torch.Tensor) -> torch.Tensor:
    """Three directional high-pass responses capture coarse texture complexity."""
    gray = feature.mean(dim=1, keepdim=True)
    filters = gray.new_tensor([
        [[-1, -1], [1, 1]], [[-1, 1], [-1, 1]], [[1, -1], [-1, 1]]
    ]).unsqueeze(1) * 0.5
    padded = F.pad(gray, (0, 1, 0, 1), mode="replicate")
    response = F.conv2d(padded, filters).square()
    energy = F.avg_pool2d(response, 3, 1, 1) + 1e-8
    q = energy / energy.sum(dim=1, keepdim=True)
    return -(q * q.log()).sum(dim=1, keepdim=True) / 1.0986122886681098


def standardize_map(x: torch.Tensor) -> torch.Tensor:
    """A per-frame scale keeps motion and texture cues on comparable ranges."""
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-4)
    return ((x - mean) / std).clamp(-5, 5)
