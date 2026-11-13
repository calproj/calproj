#!/usr/bin/env python3
"""The calibrated projection on synthetic layer inputs.

    python examples/calibrated_projection.py

Draws inputs that follow a rectifier (so their mean is far from zero), the
frozen Gaussian projection of a memory-floor adapter, and the calibrated
projection built from it. Prints, for each bottleneck channel, the share of
its energy that is a constant, and its energy relative to tr(Sigma)/r.
"""

import torch

from calproj import projection

C, r, n = 64, 4, 8192
generator = torch.Generator().manual_seed(0)

# layer inputs after a rectifier: n positions, C channels
mixing = torch.randn(C, C, dtype=torch.float64, generator=generator) / C ** 0.5
x = torch.relu(torch.randn(n, C, dtype=torch.float64, generator=generator) @ mixing + 0.5)

sigma = x.T @ x / n                          # second moment of the inputs
mu = x.mean(0)                               # their mean
energy = projection.total_energy(sigma)      # tr(Sigma)
print(f"mean share of the input energy: s = {float(mu @ mu) / energy:.2f}")

# the frozen projection of MemFLoRA: entries N(0, 1/r)
P = projection.gaussian_rows(C, r, generator)
q = x @ P.T

# the calibrated projection: the same draw, each bottleneck channel centred and
# given the energy tr(Sigma)/r (rows D P, shift -D m)
covariance = sigma - torch.outer(mu, mu)
rows = projection.balanced_rows(P.T, covariance, energy=energy)
shift = -(rows @ mu)
q_cal = x @ rows.T + shift


def describe(name: str, bottleneck: torch.Tensor) -> None:
    second_moment = (bottleneck ** 2).mean(0)
    constant = bottleneck.mean(0) ** 2 / second_moment
    relative = second_moment / (energy / r)
    print(f"{name:10s} constant share per channel: " + " ".join(f"{v:.2f}" for v in constant)
          + "   energy / (tr(Sigma)/r): " + " ".join(f"{v:.2f}" for v in relative))


describe("Gaussian", q)
describe("calibrated", q_cal)
