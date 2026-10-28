"""Inter-frame flow estimators with an explicit choice of backend."""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class MotionEstimator(nn.Module):
    def __init__(self, backend: str = "farneback", pretrained: bool = True):
        super().__init__()
        self.backend = backend
        if backend == "raft":
            from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
            weights = Raft_Small_Weights.DEFAULT if pretrained else None
            self.network = raft_small(weights=weights, progress=True).eval()
            for param in self.network.parameters():
                param.requires_grad_(False)
        elif backend != "farneback":
            raise ValueError(f"Unknown flow backend: {backend}")

    @torch.no_grad()
    def pair(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return displacement from source coordinates to target coordinates."""
        if self.backend == "raft":
            h, w = source.shape[-2:]
            hp, wp = (h + 7) // 8 * 8, (w + 7) // 8 * 8
            # Padding is applied on the same side for both frames to preserve alignment.
            a = F.pad(source * 2 - 1, (0, wp - w, 0, hp - h), mode="replicate")
            b = F.pad(target * 2 - 1, (0, wp - w, 0, hp - h), mode="replicate")
            flow = self.network(a, b)[-1]
            return flow[:, :, :h, :w]
        flows = []
        for a, b in zip(source.detach(), target.detach()):
            aa = (a.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            bb = (b.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            gray_a = cv2.cvtColor(aa, cv2.COLOR_RGB2GRAY)
            gray_b = cv2.cvtColor(bb, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                gray_a, gray_b, None, pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            flows.append(torch.from_numpy(flow.copy()).permute(2, 0, 1))
        return torch.stack(flows, dim=0).to(device=source.device, dtype=source.dtype)

    @torch.no_grad()
    def bidirectional(self, current: torch.Tensor, previous: torch.Tensor):
        # The backward field is the one we need to sample the previous state.
        return self.pair(current, previous), self.pair(previous, current)
