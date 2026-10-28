"""Checks for coordinate alignment and statistical conventions."""

import numpy as np
import torch

from elstnet.metrics import feature_fid, paired_statistics
from elstnet.ops import flow_confidence, local_frequency_entropy, scale_flow, warp


def test_identity_motion():
    x = torch.rand(2, 3, 9, 11)
    zero = torch.zeros(2, 2, 9, 11)
    result, valid = warp(x, zero, return_valid=True)
    assert torch.allclose(x, result, atol=1e-6)
    assert bool(valid.all())
    assert torch.allclose(flow_confidence(zero, zero), valid)


def test_flow_rescaling():
    flow = torch.ones(1, 2, 8, 8)
    smaller = scale_flow(flow, (4, 4))
    assert smaller.shape == (1, 2, 4, 4)
    assert torch.allclose(smaller, torch.full_like(smaller, 0.5))


def test_entropy_finite():
    scores = local_frequency_entropy(torch.rand(1, 4, 9, 9))
    assert torch.isfinite(scores).all()
    assert scores.shape == (1, 1, 9, 9)


def test_statistics_and_distance():
    result = paired_statistics([1, 2, 3, 4], [0, 0, 0, 0])
    assert result["n"] == 4
    assert abs(result["mean_difference"] - 2.5) < 1e-6
    x = np.random.default_rng(1).normal(size=(15, 4))
    assert feature_fid(x, x) < 1e-6
