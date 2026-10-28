"""Shape, gradient, and temporal state checks on small inputs."""

import torch

from elstnet.model import ELSTNet, SpectralTransport


def tiny_model(**kwargs):
    return ELSTNet(widths=(8, 12, 16), state_size=4, rank=2,
                   keyframe_interval=2, tile_size=2, halo=1,
                   scan_backend="torch", flow_backend="farneback",
                   pretrained_flow=False, **kwargs)


def test_keyframe_output_and_gradients():
    torch.set_num_threads(1)
    model = tiny_model()
    frames = torch.rand(1, 1, 3, 32, 32)
    style = torch.rand(1, 3, 32, 32)
    output = model(frames, style)
    assert output.shape == frames.shape
    assert torch.isfinite(output).all()
    output.mean().backward()
    assert model.content.stem[0].net[0].weight.grad is not None


def test_motion_update_and_modes():
    torch.set_num_threads(1)
    frames = torch.rand(1, 2, 3, 32, 32)
    style = torch.rand(1, 3, 32, 32)
    for mode in ("full", "full_scan", "no_gate", "no_transport", "fixed_sparse", "no_spectral"):
        model = tiny_model(mode=mode)
        with torch.no_grad():
            result = model(frames, style, return_details=True)
        assert result.images.shape == frames.shape
        assert torch.isfinite(result.images).all(), mode
        assert len(result.frame_states) == 2


def test_spectral_transition_is_finite():
    model = SpectralTransport(8, 8, rank=2)
    out = model(torch.randn(2, 8), torch.randn(2, 8, 4, 4))
    assert out.shape == (2, 8, 4, 4)
    assert torch.isfinite(out).all()


def test_chunked_render_keeps_temporal_context():
    torch.set_num_threads(1)
    torch.manual_seed(13)
    model = tiny_model().eval()
    video = torch.rand(1, 3, 3, 32, 32)
    style = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        complete = model(video, style)
        first = model(video[:, :1], style, return_details=True)
        rest = model(video[:, 1:], style, return_details=True, carry=first.carry)
    together = torch.cat((first.images, rest.images), dim=1)
    assert torch.allclose(complete, together, atol=1e-5)
