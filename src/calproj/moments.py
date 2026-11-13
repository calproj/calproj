"""Calibration moments, measured on a plain backbone without adapters.

An adapter starts at zero, so at the first adaptation step the adapted network
computes exactly what the frozen backbone computes. The statistics a
projection is calibrated on (the inputs of each adapted convolution, and the
gradient of its weight) can therefore be measured on the backbone itself, with
its BatchNorm layers in evaluation mode, as they are during adaptation.

A *site* is an adapted convolution, named by its module path in the backbone
(the name the MemFLoRA release gives the adapter it puts there).
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import torch
from torch import nn

Forward = Callable[[torch.Tensor], torch.Tensor]


def site_convolutions(model: nn.Module, site_names: Sequence[str]) -> dict[str, nn.Conv2d]:
    """The convolutions behind the site names; pointwise projections need ``groups == 1``."""
    convs: dict[str, nn.Conv2d] = {}
    for name in site_names:
        module = model.get_submodule(name)
        if not isinstance(module, nn.Conv2d):
            raise TypeError(f"site {name} is a {type(module).__name__}, not a Conv2d")
        if module.groups != 1:
            raise ValueError(f"site {name} is a grouped convolution (groups={module.groups})")
        convs[name] = module
    return convs


@torch.no_grad()
def activation_moments(
    model: nn.Module, site_names: Sequence[str], forward: Forward, batches: Iterable[torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Second moment ``E[x x^T]`` and mean ``E[x]`` of the input of each site, over batch and positions.

    ``forward(x)`` runs the model on one batch (it holds any input reshaping).
    Returns ``(moments, means)``: per site, a ``C_in x C_in`` and a ``C_in``
    float64 CPU tensor. Forward-only: this is what a device can accumulate in
    the pass that calibrates AdaBN.
    """
    convs = site_convolutions(model, site_names)
    sums = {name: None for name in convs}
    firsts = {name: None for name in convs}
    counts = {name: 0 for name in convs}

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            x = inputs[0]
            flat = x.detach().transpose(0, 1).reshape(x.shape[1], -1).to(torch.float64)
            gram = flat @ flat.T
            first = flat.sum(dim=1)
            sums[name] = gram if sums[name] is None else sums[name] + gram
            firsts[name] = first if firsts[name] is None else firsts[name] + first
            counts[name] += flat.shape[1]

        return hook

    handles = [conv.register_forward_pre_hook(make_hook(name)) for name, conv in convs.items()]
    was_training = model.training
    model.eval()
    try:
        for x in batches:
            forward(x)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    out, means = {}, {}
    for name in convs:
        if counts[name] == 0:
            raise RuntimeError(f"site {name} saw no input: no calibration batch reached it")
        moment = (sums[name] / counts[name]).cpu()
        if not torch.isfinite(moment).all():
            raise FloatingPointError(f"non-finite activation moment at site {name}")
        out[name] = moment
        means[name] = (firsts[name] / counts[name]).cpu()
    return out, means


def weight_gradients(
    model: nn.Module, site_names: Sequence[str], forward: Forward, x: torch.Tensor, y: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Cross-entropy gradient of each site's convolution weight on one labelled batch.

    BatchNorm stays in evaluation mode, so the gradient is that of the fused
    Conv-BN site the adapter perturbs. Returns ``C_out x C_in x k_h x k_w``
    float64 CPU tensors. A device does not form these: it sketches them (see
    :func:`calproj.projection.sketch_gradient_moment`).
    """
    convs = site_convolutions(model, site_names)
    weights = [conv.weight for conv in convs.values()]
    flags = [w.requires_grad for w in weights]
    was_training = model.training
    model.eval()
    try:
        for w in weights:
            w.requires_grad_(True)
        with torch.enable_grad():
            loss = nn.functional.cross_entropy(forward(x), y)
            grads = torch.autograd.grad(loss, weights)
    finally:
        for w, flag in zip(weights, flags):
            w.requires_grad_(flag)
        model.train(was_training)
    out = {}
    for name, grad in zip(convs, grads):
        grad = grad.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(grad).all():
            raise FloatingPointError(f"non-finite weight gradient at site {name}")
        out[name] = grad
    return out
