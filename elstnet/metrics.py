"""Perceptual scores, motion-compensated error, and statistical summaries."""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
from scipy.stats import t

from .flow import MotionEstimator
from .ops import flow_confidence, warp
from .utils import rgb_tensor


def feature_fid(generated: np.ndarray, references: np.ndarray) -> float:
    """A Frechet distance between two feature sets, with robust covariance handling."""
    if len(generated) < 2 or len(references) < 2:
        raise ValueError("FID requires at least two observations in each set")
    means_a, means_b = generated.mean(0), references.mean(0)
    cov_a = np.cov(generated, rowvar=False)
    cov_b = np.cov(references, rowvar=False)
    product = sqrtm(cov_a @ cov_b)
    if np.iscomplexobj(product):
        if not np.allclose(product.imag, 0, atol=1e-3):
            raise ValueError("Covariance square root contains large imaginary values")
        product = product.real
    score = np.sum((means_a - means_b) ** 2) + np.trace(cov_a + cov_b - 2 * product)
    return float(max(score, 0.0))


class ArtFIDMetric:
    """ArtFID is (1 + content LPIPS) times (1 + style feature FID)."""

    def __init__(self, device="cpu"):
        try:
            import lpips
            from pytorch_fid.inception import InceptionV3
        except ImportError as exc:
            raise ImportError("Install the metrics extras: pip install -e '.[metrics]'") from exc
        self.device = torch.device(device)
        self.lpips = lpips.LPIPS(net="alex").to(self.device).eval()
        self.inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(self.device).eval()

    @torch.no_grad()
    def _features(self, image):
        feats = self.inception(F.interpolate(image, (299, 299), mode="bilinear",
                                             align_corners=False))[0]
        return feats.reshape(feats.shape[0], -1).cpu().numpy()

    @torch.no_grad()
    def evaluate(self, pairs: list[dict], size: int = 512):
        """Each pair names source content, stylized output, and reference artwork."""
        gen, reference, distances = [], [], []
        for pair in pairs:
            content = rgb_tensor(pair["content"], size, self.device)[None]
            styled = rgb_tensor(pair["styled"], size, self.device)[None]
            art = rgb_tensor(pair["style"], size, self.device)[None]
            distances.append(float(self.lpips(styled * 2 - 1, content * 2 - 1).mean()))
            gen.append(self._features(styled)[0])
            reference.append(self._features(art)[0])
        lpips_value = float(np.mean(distances))
        fid = feature_fid(np.asarray(gen), np.asarray(reference))
        return {"artfid": (1 + lpips_value) * (1 + fid), "lpips": lpips_value,
                "style_fid": fid, "pair_count": len(pairs)}


@torch.no_grad()
def temporal_warp_error(stylized: torch.Tensor, original: torch.Tensor,
                        estimator: MotionEstimator) -> dict:
    """Use original-frame flow to compare neighboring stylized frames fairly."""
    if stylized.shape != original.shape or stylized.ndim != 5:
        raise ValueError("Original and stylized videos must have matching [B,T,C,H,W] shapes")
    errors = []
    for index in range(1, stylized.shape[1]):
        backward, forward = estimator.bidirectional(original[:, index], original[:, index - 1])
        confidence = flow_confidence(backward, forward)
        delta = (stylized[:, index] - warp(stylized[:, index - 1], backward)).abs()
        score = (delta * confidence).sum() / (3 * confidence.sum()).clamp_min(1e-6)
        errors.append(float(score))
    if not errors:
        raise ValueError("Temporal error needs at least two frames")
    return {"warp_error": float(np.mean(errors)), "frame_pair_count": len(errors),
            "per_pair": errors}


def paired_statistics(values_a, values_b, confidence=0.95) -> dict:
    """Keep the pairing when computing uncertainty across independent sequences."""
    a, b = np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2:
        raise ValueError("Expected equal-length paired 1D arrays with at least two samples")
    delta = a - b
    mean = float(np.mean(delta))
    sd = float(np.std(delta, ddof=1))
    margin = float(t.ppf((1 + confidence) / 2, len(delta) - 1) * sd / math.sqrt(len(delta)))
    tstat = mean / (sd / math.sqrt(len(delta))) if sd > 0 else (math.inf if mean else 0)
    pvalue = float(2 * t.sf(abs(tstat), len(delta) - 1)) if sd > 0 else (0.0 if mean else 1.0)
    return {"n": len(delta), "mean_difference": mean, "std_difference": sd,
            "ci_low": mean - margin, "ci_high": mean + margin, "pvalue": pvalue}
