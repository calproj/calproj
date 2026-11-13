"""The frozen down-projection of a memory-floor adapter: its subspace and its scale.

Notation: a layer input has ``C`` channels and the adapter rank is ``r``. A
projection is an ``r x C`` matrix ``P`` (the weight of a pointwise convolution).
A *moment* is a ``C x C`` symmetric positive semi-definite matrix: the second
moment of the layer inputs, ``Sigma = E[x x^T]`` (not centred: the gradient of
a weight is built from ``x``, not from ``x - mean``), or the input-side moment
of a weight gradient, ``M = sum_taps G_tap^T G_tap``.

Everything here runs in float64 on the CPU: the matrices are small, and the
eigenvalues of a moment span many orders of magnitude.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable

import torch

#: floor on the energy of a direction, relative to the total energy, when
#: dividing by it (a direction no input excites would otherwise scale to infinity)
ENERGY_FLOOR = 1e-10


def _f64(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(device="cpu", dtype=torch.float64)


def _check_finite(name: str, *tensors: torch.Tensor) -> None:
    for t in tensors:
        if not torch.isfinite(t).all():
            raise FloatingPointError(f"{name}: non-finite values")


def top_eigenspace(moment: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Leading eigenvectors of a moment.

    Returns ``(V, lam)``: ``V`` is ``C x rank`` with orthonormal columns, ``lam``
    the matching eigenvalues, largest first.
    """
    m = _f64(moment)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"a moment is square, got {tuple(m.shape)}")
    c = m.shape[0]
    if not 0 < rank <= c:
        raise ValueError(f"rank {rank} is not in 1..{c}")
    _check_finite("moment", m)
    evals, evecs = torch.linalg.eigh(0.5 * (m + m.T))
    _check_finite("eigendecomposition", evals, evecs)  # verify the output, not the status
    order = torch.argsort(evals, descending=True)[:rank]
    return evecs[:, order].contiguous(), evals[order].clamp_min(0.0)


def direction_energy(directions: torch.Tensor, moment: torch.Tensor) -> torch.Tensor:
    """``v_j^T M v_j`` for each column ``v_j`` of ``directions`` (``C x r``)."""
    v, m = _f64(directions), _f64(moment)
    return torch.einsum("cj,cd,dj->j", v, m, v)


def total_energy(moment: torch.Tensor) -> float:
    """``tr(M)``: for an input moment, the mean squared norm of a layer input."""
    return float(torch.trace(_f64(moment)))


def balanced_rows(
    directions: torch.Tensor, activation_moment: torch.Tensor, energy: float | None = None
) -> torch.Tensor:
    """Projection along ``directions`` whose bottleneck channels all carry ``energy / r``.

    A Gaussian projection with entries ``N(0, 1/r)`` gives each of its ``r``
    bottleneck channels the second moment ``tr(Sigma)/r`` in expectation. This
    scales each direction to exactly that value (``energy`` defaults to
    ``tr(Sigma)`` of the moment given), so the bottleneck has the magnitude the
    Gaussian one has. The directions need not be orthonormal. Passing a
    covariance as the moment and the uncentred ``tr(Sigma)`` as ``energy``
    balances a mean-centred bottleneck to the same magnitude.
    """
    v = _f64(directions)
    rank = v.shape[1]
    total = total_energy(activation_moment) if energy is None else float(energy)
    along = direction_energy(v, activation_moment).clamp_min(ENERGY_FLOOR * max(total, 1e-300))
    scale = torch.sqrt(total / (rank * along))
    return (v * scale).T.contiguous()


def energy_matched_rows(directions: torch.Tensor, activation_moment: torch.Tensor) -> torch.Tensor:
    """Orthonormal directions times one factor, so that the bottleneck carries ``tr(Sigma)`` in total.

    The channels keep their natural, unequal energies: the first principal
    direction dominates. The counterpart of :func:`balanced_rows` without the
    per-channel balancing.
    """
    v = _f64(directions)
    energy = total_energy(activation_moment)
    along = direction_energy(v, activation_moment).sum().clamp_min(ENERGY_FLOOR * max(energy, 1e-300))
    return (v * math.sqrt(energy / float(along))).T.contiguous()


def norm_matched_rows(directions: torch.Tensor) -> torch.Tensor:
    """Orthonormal directions times ``sqrt(C/r)``: the expected row norm of ``N(0, 1/r)``."""
    v = _f64(directions)
    c, rank = v.shape
    return (v * math.sqrt(c / rank)).T.contiguous()


def random_directions(channels: int, rank: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """A uniformly random ``rank``-dimensional subspace: ``C x rank``, orthonormal columns."""
    if not 0 < rank <= channels:
        raise ValueError(f"rank {rank} is not in 1..{channels}")
    g = torch.randn(channels, rank, dtype=torch.float64, generator=generator)
    q, _ = torch.linalg.qr(g, mode="reduced")
    return q


def gaussian_rows(channels: int, rank: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """The MemFLoRA initialisation: i.i.d. ``N(0, 1/r)`` entries, ``rank x C``."""
    return torch.randn(rank, channels, dtype=torch.float64, generator=generator) / math.sqrt(rank)


def rademacher_rows(channels: int, rank: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """I.i.d. ``+-1/sqrt(r)`` entries, ``rank x C``: the second moments of ``N(0, 1/r)``, applied by additions."""
    signs = torch.randint(0, 2, (rank, channels), generator=generator, dtype=torch.int64)
    return (2.0 * signs.to(torch.float64) - 1.0) / math.sqrt(rank)


def selection_directions(indices: torch.Tensor, channels: int) -> torch.Tensor:
    """One-hot directions (``C x r``) that pick the input channels ``indices``.

    A selection projection makes the adapter a channel-sparse update of the
    frozen convolution: ``U`` is added to the kernel slices of those channels.
    """
    idx = indices.to(torch.int64).flatten()
    if idx.numel() != torch.unique(idx).numel() or idx.min() < 0 or idx.max() >= channels:
        raise ValueError("selection indices must be distinct and within the layer's channels")
    out = torch.zeros(channels, idx.numel(), dtype=torch.float64)
    out[idx, torch.arange(idx.numel())] = 1.0
    return out


def active_channels(activation_moment: torch.Tensor, floor: float = 1e-3) -> torch.Tensor:
    """Indices of the channels whose energy is at least ``floor`` times the mean channel energy."""
    energy = torch.diagonal(_f64(activation_moment))
    return torch.nonzero(energy >= floor * energy.mean(), as_tuple=False).flatten()


def random_channels(
    channels: int, rank: int, generator: torch.Generator | None = None, among: torch.Tensor | None = None
) -> torch.Tensor:
    """``rank`` distinct channel indices drawn uniformly, from ``among`` when given and large enough."""
    pool = torch.arange(channels) if among is None or among.numel() < rank else among.to(torch.int64)
    if rank > pool.numel():
        raise ValueError(f"rank {rank} exceeds the {pool.numel()} channels to choose from")
    return pool[torch.randperm(pool.numel(), generator=generator)[:rank]]


def top_energy_channels(activation_moment: torch.Tensor, rank: int) -> torch.Tensor:
    """The ``rank`` channels with the largest second moment."""
    energy = torch.diagonal(_f64(activation_moment))
    return torch.argsort(energy, descending=True)[:rank]


def captured_fraction(projection: torch.Tensor, moment: torch.Tensor) -> float:
    """``tr(Pi M) / tr(M)``, with ``Pi`` the orthogonal projector on the row space of ``projection``.

    With ``M`` the input-side moment of a weight gradient ``G``, this is the
    share of ``||G||_F^2`` that an update confined to the row space of the
    projection can follow. It depends on the subspace only, not on the scale.
    """
    p, m = _f64(projection), _f64(moment)
    if p.ndim != 2 or p.shape[1] != m.shape[0]:
        raise ValueError(f"projection {tuple(p.shape)} does not match moment {tuple(m.shape)}")
    total = float(torch.trace(m))
    if not total > 0.0:
        return float("nan")
    basis, _ = torch.linalg.qr(p.T, mode="reduced")
    return float((basis * (m @ basis)).sum() / total)


def reachable_moment(projection: torch.Tensor, activation_moment: torch.Tensor) -> torch.Tensor:
    """``K = Sigma P^T (P Sigma P^T)^+ P Sigma``: the part of the inputs a linear map of ``P x`` can reproduce.

    For any matrix ``A``, ``min_U E||A x - U P x||^2 = tr(A (Sigma - K) A^T)``:
    ``K`` is to the function space of the layer what the projector on the row
    space of ``P`` is to its weight space. It does not depend on the scale of
    ``P``, nor on any invertible mixing of its rows.
    """
    p, sigma = _f64(projection), _f64(activation_moment)
    ps = p @ sigma
    gram = ps @ p.T
    return ps.T @ torch.linalg.pinv(0.5 * (gram + gram.T), hermitian=True) @ ps


def explained_share(projection: torch.Tensor, activation_moment: torch.Tensor) -> float:
    """``tr(K) / tr(Sigma)``: share of the input energy the bottleneck ``P x`` linearly explains.

    The principal subspace maximises it; a random projection reaches a
    comparable value when a few directions hold most of the input energy.
    """
    total = total_energy(activation_moment)
    if not total > 0.0:
        return float("nan")
    return float(torch.trace(reachable_moment(projection, activation_moment)) / total)


def functional_fraction(
    projection: torch.Tensor, activation_moment: torch.Tensor, gradient_moment: torch.Tensor
) -> float:
    """``tr(K M) / tr(Sigma M)``: share of the *effect* of a gradient step the adapter can reproduce.

    A step along the weight gradient ``G`` changes the layer output by
    ``-eta G x``. With ``M = sum_taps G^T G``, the mean squared size of that
    change is ``eta^2 tr(Sigma M)``, and the best ``U P x`` reproduces
    ``eta^2 tr(K M)`` of it. :func:`captured_fraction` measures the same in
    weight space, where every input direction counts alike whether or not the
    data excite it.
    """
    sigma, m = _f64(activation_moment), _f64(gradient_moment)
    total = float((sigma * m).sum())
    if not total > 0.0:
        return float("nan")
    return float((reachable_moment(projection, sigma) * m).sum() / total)


def remove_direction(rows: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Project the rows of a projection on the orthogonal complement of one input direction."""
    r, d = _f64(rows), _f64(direction)
    norm = float(torch.linalg.norm(d))
    if not norm > 0.0:
        return r
    d = d / norm
    return r - (r @ d)[:, None] * d[None, :]


def deflate(moment: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """``Q M Q`` with ``Q`` the projector on the orthogonal complement of ``direction``."""
    m, d = _f64(moment), _f64(direction)
    norm = float(torch.linalg.norm(d))
    if not norm > 0.0:
        return m
    d = d / norm
    q = torch.eye(m.shape[0], dtype=torch.float64) - torch.outer(d, d)
    return q @ m @ q


def optimal_fraction(moment: torch.Tensor, rank: int) -> float:
    """The largest :func:`captured_fraction` any ``rank``-dimensional subspace reaches."""
    _, lam = top_eigenspace(moment, rank)
    total = total_energy(moment)
    return float(lam.sum() / total) if total > 0.0 else float("nan")


def sketch_gradient_moment(
    weight_grad: torch.Tensor, sketch: int | None, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Input-side moment of a convolution weight gradient, optionally through an output sketch.

    ``weight_grad`` is ``C_out x C_in x k_h x k_w``. Exact (``sketch=None``):
    ``M = sum_taps G_tap^T G_tap``. Sketched: the same with ``S G`` in place of
    ``G``, where ``S`` is ``sketch x C_out`` with ``N(0, 1/sketch)`` entries,
    so that its expectation is ``M``. A device forms ``S G`` without the
    full-width input ever being saved: it keeps ``S g`` (``sketch`` channels)
    in the backward pass and contracts it with the input in one forward replay.
    """
    g = _f64(weight_grad)
    if g.ndim != 4:
        raise ValueError(f"expected a 4-D convolution weight gradient, got {tuple(g.shape)}")
    if sketch is not None:
        s = torch.randn(sketch, g.shape[0], dtype=torch.float64, generator=generator) / math.sqrt(sketch)
        g = torch.einsum("so,oikl->sikl", s, g)
    return torch.einsum("oikl,ojkl->ij", g, g)


def randomized_eigenspace(
    moment: torch.Tensor, rank: int, oversample: int = 8, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """What :func:`streaming_eigenspace` computes, given the moment it never forms.

    ``Sigma Omega`` and ``Q^T Sigma Q`` are linear in ``Sigma``, so accumulating
    them over a stream equals applying them to the accumulated moment. A
    simulation that already holds the moment uses this; a device streams.
    """
    m = _f64(moment)
    channels = m.shape[0]
    width = min(channels, rank + oversample)
    omega = torch.randn(channels, width, dtype=torch.float64, generator=generator)
    basis, _ = torch.linalg.qr(m @ omega, mode="reduced")
    w, lam = top_eigenspace(basis.T @ m @ basis, min(rank, width))
    return (basis @ w).contiguous(), lam


def streaming_eigenspace(
    batches: Callable[[], Iterable[torch.Tensor]],
    channels: int,
    rank: int,
    oversample: int = 8,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Leading eigenspace of ``Sigma = sum_n x_n x_n^T / N`` in ``O(C (rank + oversample))`` memory.

    A randomized range finder followed by a Rayleigh-Ritz step, in two passes
    over the data, for a device that cannot hold the ``C x C`` moment.
    ``batches()`` yields ``N_i x C`` tensors and is called once per pass.
    Returns ``(V, lam, energy)`` as :func:`top_eigenspace` does, plus
    ``energy = tr(Sigma)``; ``lam`` holds the exact energies ``v_j^T Sigma v_j``.
    """
    width = min(channels, rank + oversample)
    omega = torch.randn(channels, width, dtype=torch.float64, generator=generator)
    y = torch.zeros(channels, width, dtype=torch.float64)
    count = 0
    for x in batches():  # pass 1: Y = Sigma Omega, never forming Sigma
        x = _f64(x)
        y += x.T @ (x @ omega)
        count += x.shape[0]
    if count == 0:
        raise ValueError("no calibration data")
    basis, _ = torch.linalg.qr(y, mode="reduced")
    small = torch.zeros(width, width, dtype=torch.float64)
    energy = 0.0
    for x in batches():  # pass 2: B = Q^T Sigma Q and tr(Sigma)
        x = _f64(x)
        z = x @ basis
        small += z.T @ z
        energy += float((x * x).sum())
    small /= count
    energy /= count
    w, lam = top_eigenspace(small, min(rank, width))
    return (basis @ w).contiguous(), lam, energy
