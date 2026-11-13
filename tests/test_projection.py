"""The mathematics of the projection: subspace, scale, captured share, estimators."""

from __future__ import annotations

import math

import pytest
import torch

from calproj import projection as pj


def spd(c: int, decay: float = 0.6, seed: int = 0) -> torch.Tensor:
    """A moment with a geometric spectrum in a random basis."""
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(c, c, dtype=torch.float64, generator=g))
    lam = decay ** torch.arange(c, dtype=torch.float64)
    return (q * lam) @ q.T


def test_top_eigenspace_orders_and_orthonormalises():
    m = torch.diag(torch.tensor([1.0, 5.0, 3.0, 0.5], dtype=torch.float64))
    v, lam = pj.top_eigenspace(m, 2)
    assert torch.allclose(lam, torch.tensor([5.0, 3.0], dtype=torch.float64))
    assert torch.allclose(v.T @ v, torch.eye(2, dtype=torch.float64), atol=1e-12)
    assert abs(abs(float(v[1, 0])) - 1.0) < 1e-12 and abs(abs(float(v[2, 1])) - 1.0) < 1e-12


def test_top_eigenspace_rejects_bad_input():
    with pytest.raises(ValueError):
        pj.top_eigenspace(torch.eye(3), 4)
    with pytest.raises(FloatingPointError):
        pj.top_eigenspace(torch.full((3, 3), float("nan")), 1)


@pytest.mark.parametrize("rank", [1, 2, 4, 8])
def test_balanced_rows_give_every_channel_the_gaussian_energy(rank):
    sigma = spd(24)
    v, _ = pj.top_eigenspace(sigma, rank)
    p = pj.balanced_rows(v, sigma)
    channel_energy = torch.diagonal(p @ sigma @ p.T)
    assert p.shape == (rank, 24)
    assert torch.allclose(channel_energy, torch.full((rank,), pj.total_energy(sigma) / rank, dtype=torch.float64))


def test_gaussian_rows_have_that_energy_in_expectation():
    sigma, rank = spd(24), 4
    g = torch.Generator().manual_seed(0)
    energies = torch.stack(
        [torch.diagonal(p @ sigma @ p.T) for p in (pj.gaussian_rows(24, rank, g) for _ in range(4000))]
    )
    expected = pj.total_energy(sigma) / rank
    assert abs(float(energies.mean()) / expected - 1.0) < 0.05


def test_energy_matched_rows_carry_the_total_energy():
    sigma = spd(24)
    v, _ = pj.top_eigenspace(sigma, 3)
    p = pj.energy_matched_rows(v, sigma)
    assert math.isclose(float(torch.trace(p @ sigma @ p.T)), pj.total_energy(sigma), rel_tol=1e-10)
    rows = torch.linalg.norm(p, dim=1)
    assert torch.allclose(rows, rows[0].expand(3))  # one factor for all rows


def test_norm_matched_rows_have_the_gaussian_row_norm():
    p = pj.norm_matched_rows(pj.random_directions(64, 2, torch.Generator().manual_seed(0)))
    assert torch.allclose(torch.linalg.norm(p, dim=1), torch.full((2,), math.sqrt(64 / 2), dtype=torch.float64))


def test_balanced_rows_survive_a_direction_without_energy():
    sigma = torch.diag(torch.tensor([4.0, 1.0, 0.0, 0.0], dtype=torch.float64))
    v, _ = pj.top_eigenspace(sigma, 3)
    assert torch.isfinite(pj.balanced_rows(v, sigma)).all()


def test_captured_fraction_is_r_over_c_for_an_isotropic_moment():
    eye = torch.eye(40, dtype=torch.float64)
    for rank in (1, 2, 8):
        p = pj.random_directions(40, rank, torch.Generator().manual_seed(rank)).T
        assert math.isclose(pj.captured_fraction(p, eye), rank / 40, rel_tol=1e-10)


def test_captured_fraction_ignores_scale_and_peaks_at_the_top_eigenspace():
    m = spd(32, decay=0.5)
    v, lam = pj.top_eigenspace(m, 3)
    best = pj.captured_fraction(v.T, m)
    assert math.isclose(best, float(lam.sum()) / pj.total_energy(m), rel_tol=1e-10)
    assert math.isclose(best, pj.optimal_fraction(m, 3), rel_tol=1e-10)
    assert math.isclose(pj.captured_fraction(pj.balanced_rows(v, m), m), best, rel_tol=1e-10)
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        assert pj.captured_fraction(pj.gaussian_rows(32, 3, g), m) <= best + 1e-12


def test_random_subspace_captures_r_over_c_on_average():
    m = spd(32, decay=0.5)
    g = torch.Generator().manual_seed(1)
    mean = sum(pj.captured_fraction(pj.random_directions(32, 4, g).T, m) for _ in range(3000)) / 3000
    assert abs(mean - 4 / 32) < 0.01


def test_gradient_moment_exact_and_sketched():
    g = torch.Generator().manual_seed(0)
    grad = torch.randn(12, 10, 5, 1, dtype=torch.float64, generator=g)
    exact = pj.sketch_gradient_moment(grad, None)
    by_hand = sum(grad[:, :, k, 0].T @ grad[:, :, k, 0] for k in range(5))
    assert torch.allclose(exact, by_hand)
    mean = sum(pj.sketch_gradient_moment(grad, 4, g) for _ in range(3000)) / 3000
    assert float(torch.linalg.norm(mean - exact) / torch.linalg.norm(exact)) < 0.05  # unbiased


def test_streaming_estimator_matches_its_moment_form_and_the_exact_eigenspace():
    g = torch.Generator().manual_seed(0)
    mixing = torch.randn(48, 5, dtype=torch.float64, generator=g)
    data = [torch.randn(200, 5, dtype=torch.float64, generator=g) @ mixing.T
            + 0.01 * torch.randn(200, 48, dtype=torch.float64, generator=g) for _ in range(4)]
    stacked = torch.cat(data)
    sigma = stacked.T @ stacked / stacked.shape[0]
    v_stream, lam_stream, energy = pj.streaming_eigenspace(lambda: iter(data), 48, 3, 8, torch.Generator().manual_seed(5))
    v_moment, lam_moment = pj.randomized_eigenspace(sigma, 3, 8, torch.Generator().manual_seed(5))
    assert math.isclose(energy, pj.total_energy(sigma), rel_tol=1e-10)
    assert torch.allclose(lam_stream, lam_moment, rtol=1e-8)
    assert torch.allclose(v_stream @ v_stream.T, v_moment @ v_moment.T, atol=1e-8)  # the same subspace
    assert torch.allclose(lam_stream, pj.direction_energy(v_stream, sigma), rtol=1e-8)
    assert pj.captured_fraction(v_stream.T, sigma) > 0.999 * pj.optimal_fraction(sigma, 3)


def test_function_space_shares():
    g = torch.Generator().manual_seed(0)
    sigma = spd(30, decay=0.5)
    grad = torch.randn(7, 30, dtype=torch.float64, generator=g)
    m = grad.T @ grad
    v, lam = pj.top_eigenspace(sigma, 3)
    # the principal subspace explains the top eigenvalues, and nothing explains more
    assert math.isclose(pj.explained_share(v.T, sigma), float(lam.sum()) / pj.total_energy(sigma), rel_tol=1e-9)
    rnd = pj.gaussian_rows(30, 3, g)
    assert pj.explained_share(rnd, sigma) <= pj.explained_share(v.T, sigma) + 1e-12
    # with a steep spectrum a random projection explains far more than r/C of the inputs
    assert pj.explained_share(rnd, sigma) > 3 * (3 / 30)
    # scale and row mixing do not matter
    mix = torch.randn(3, 3, dtype=torch.float64, generator=g)
    assert math.isclose(pj.explained_share(mix @ rnd, sigma), pj.explained_share(rnd, sigma), rel_tol=1e-8)
    assert math.isclose(pj.functional_fraction(5.0 * rnd, sigma, m), pj.functional_fraction(rnd, sigma, m), rel_tol=1e-8)
    # the definition: the best U P x reproduces tr(K M) of the step's effect G x
    k = pj.reachable_moment(rnd, sigma)
    u = grad @ sigma @ rnd.T @ torch.linalg.inv(rnd @ sigma @ rnd.T)
    residual = torch.trace((grad - u @ rnd) @ sigma @ (grad - u @ rnd).T)
    assert math.isclose(float(residual), float(torch.trace(grad @ (sigma - k) @ grad.T)), rel_tol=1e-8)
    assert 0.0 <= pj.functional_fraction(rnd, sigma, m) <= 1.0
    # full rank reproduces everything
    assert math.isclose(pj.functional_fraction(torch.eye(30, dtype=torch.float64), sigma, m), 1.0, rel_tol=1e-8)


def test_remove_direction_and_deflate():
    g = torch.Generator().manual_seed(0)
    rows = torch.randn(3, 12, dtype=torch.float64, generator=g)
    d = torch.randn(12, dtype=torch.float64, generator=g)
    out = pj.remove_direction(rows, d)
    assert float((out @ d).abs().max()) < 1e-10
    m = spd(12)
    deflated = pj.deflate(m, d)
    assert float((deflated @ d).abs().max()) < 1e-10
    v, _ = pj.top_eigenspace(deflated, 2)
    assert float((v.T @ d).abs().max()) < 1e-8


def test_rademacher_and_selection_rows():
    g = torch.Generator().manual_seed(0)
    signs = pj.rademacher_rows(50, 4, g)
    assert torch.allclose(signs.abs(), torch.full((4, 50), 0.5, dtype=torch.float64))
    assert torch.allclose(torch.linalg.norm(signs, dim=1), torch.full((4,), math.sqrt(50 / 4), dtype=torch.float64))
    sigma = torch.diag(torch.tensor([9.0, 0.0, 4.0, 1.0, 0.25], dtype=torch.float64))
    assert pj.top_energy_channels(sigma, 2).tolist() == [0, 2]
    assert pj.active_channels(sigma).tolist() == [0, 2, 3, 4]  # the dead channel is left out
    picked = pj.random_channels(5, 3, g, among=pj.active_channels(sigma))
    assert len(set(picked.tolist())) == 3 and 1 not in picked.tolist()
    d = pj.selection_directions(torch.tensor([2, 0]), 5)
    assert d.shape == (5, 2) and d[2, 0] == 1.0 and d[0, 1] == 1.0 and float(d.sum()) == 2.0
    rows = pj.balanced_rows(d, sigma)  # each selected channel rescaled to tr(Sigma)/r
    assert torch.allclose(torch.diagonal(rows @ sigma @ rows.T), torch.full((2,), 14.25 / 2, dtype=torch.float64))
    with pytest.raises(ValueError):
        pj.selection_directions(torch.tensor([1, 1]), 5)
