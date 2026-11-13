"""Rescaling a frozen projection is a change of step size, under Adam, exactly.

Training ``U`` behind the projection ``D P`` with Adam ``(lr, weight_decay, eps)``
gives the same function as training behind ``P`` with, for bottleneck channel
``j``, ``(lr d_j, weight_decay / d_j^2, eps / d_j)``.
"""

from __future__ import annotations

import torch


def train(columns, projection, x, y, groups, steps=30):
    """Fit ``y ~ U (projection x)`` with Adam; ``columns`` are the columns of ``U``."""
    optimizer = torch.optim.Adam(groups, betas=(0.9, 0.999))
    for _ in range(steps):
        optimizer.zero_grad()
        u = torch.stack(columns, dim=1)
        loss = ((x @ projection.T @ u.T - y) ** 2).mean()
        loss.backward()
        optimizer.step()
    return torch.stack(columns, dim=1).detach()


def test_rescaled_projection_equals_rescaled_step_sizes():
    g = torch.Generator().manual_seed(0)
    c, r, out, n = 10, 3, 4, 64
    p = torch.randn(r, c, dtype=torch.float64, generator=g)
    d = torch.tensor([0.5, 2.0, 7.0], dtype=torch.float64)
    x = torch.randn(n, c, dtype=torch.float64, generator=g)
    y = torch.randn(n, out, dtype=torch.float64, generator=g)
    lr, wd, eps = 1e-2, 5e-4, 1e-8

    def columns():
        return [torch.zeros(out, dtype=torch.float64, requires_grad=True) for _ in range(r)]

    a = columns()  # behind D P, one setting for all channels
    u_scaled = train(a, d[:, None] * p, x, y, [{"params": a, "lr": lr, "weight_decay": wd, "eps": eps}])
    b = columns()  # behind P, per-channel settings
    groups = [
        {"params": [b[j]], "lr": lr * float(d[j]), "weight_decay": wd / float(d[j]) ** 2, "eps": eps / float(d[j])}
        for j in range(r)
    ]
    u_plain = train(b, p, x, y, groups)
    # the same function: U' D = U
    assert torch.allclose(u_scaled * d[None, :], u_plain, rtol=1e-9, atol=1e-12)
    assert float(u_plain.abs().max()) > 1e-3  # and it did move


def test_a_uniform_factor_is_a_learning_rate():
    g = torch.Generator().manual_seed(1)
    c, r, out, n, factor = 12, 2, 3, 48, 5.0
    p = torch.randn(r, c, dtype=torch.float64, generator=g)
    x = torch.randn(n, c, dtype=torch.float64, generator=g)
    y = torch.randn(n, out, dtype=torch.float64, generator=g)
    a = [torch.zeros(out, dtype=torch.float64, requires_grad=True) for _ in range(r)]
    u_scaled = train(a, factor * p, x, y, [{"params": a, "lr": 1e-3, "weight_decay": 0.0, "eps": 0.0}])
    b = [torch.zeros(out, dtype=torch.float64, requires_grad=True) for _ in range(r)]
    u_plain = train(b, p, x, y, [{"params": b, "lr": 1e-3 * factor, "weight_decay": 0.0, "eps": 0.0}])
    assert torch.allclose(u_scaled * factor, u_plain, rtol=1e-9, atol=1e-12)
