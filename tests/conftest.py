"""Shared fixtures: a small synthetic sensor task with a source and a shifted target."""

from __future__ import annotations

import pytest
import torch

from calproj import protocol, upstream

upstream.enable()

from src.data.splits import SplitTensors  # noqa: E402


def synthetic_windows(n: int, channels: int, length: int, classes: int, gain: torch.Tensor, seed: int):
    """Windows whose class sets the amplitude pattern of a few latent oscillations.

    The channels mix eight latent signals, so the inputs are low-rank plus
    noise, as sensor windows are. ``gain`` rescales the channels: the shift.
    """
    g = torch.Generator().manual_seed(seed)
    mixing = torch.randn(channels, 8, generator=torch.Generator().manual_seed(7))
    patterns = torch.randn(classes, 8, generator=torch.Generator().manual_seed(11))
    t = torch.linspace(0, 6.28, length)
    basis = torch.stack([torch.sin((k + 1) * t) for k in range(8)])  # 8 x T
    y = torch.randint(0, classes, (n,), generator=g)
    latent = patterns[y][:, :, None] * basis[None] + 0.3 * torch.randn(n, 8, length, generator=g)
    x = torch.einsum("cl,nlt->nct", mixing, latent) + 0.1 * torch.randn(n, channels, length, generator=g)
    return (x * gain[None, :, None]).float(), y


@pytest.fixture(scope="session")
def args():
    """The Opportunity/T-ResNet setting: 97 channels, 17 classes, windows of 60."""
    return protocol.make_args("opportunity", "t_resnet", "/nonexistent")


@pytest.fixture(scope="session")
def split(args):
    c, t, k = args.expected_input_channels, args.window_size, args.expected_num_classes
    ones = torch.ones(c)
    shifted = 1.0 + 0.8 * torch.rand(c, generator=torch.Generator().manual_seed(3))
    return SplitTensors(
        pretrain=synthetic_windows(512, c, t, k, ones, 1),
        pretest=synthetic_windows(128, c, t, k, ones, 2),
        shift_train=synthetic_windows(256, c, t, k, shifted, 3),
        shift_test=synthetic_windows(128, c, t, k, shifted, 4),
    )


@pytest.fixture(scope="session")
def source_state(args, split):
    """A source model trained for one epoch on the synthetic source windows."""
    from calproj import adapt
    from experiments.benchmark_training import fit_epochs_best
    from src.data.benchmark import make_loaders

    adapt.seed_everything(1)
    loaders = make_loaders(split, args.batch_size, 1)
    model = adapt.build_backbone(args)
    fit_epochs_best(
        model, loaders["Pretrain_train"], loaders["Pretrain_val"], 1,
        args.pretrain_lr, args.pretrain_weight_decay, torch.device("cpu"), args.backbone,
    )
    return {k: v.detach().clone() for k, v in model.state_dict().items()}
