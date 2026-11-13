"""The trial runner against the MemFLoRA release, on a synthetic task."""

from __future__ import annotations

import math

import pytest
import torch

from calproj import adapt, projection as pj, protocol, upstream
from calproj.variants import VARIANTS, get

upstream.enable()

from experiments.benchmark_adabn import calibrate_minimal_adabn_if_needed  # noqa: E402
from experiments.benchmark_common import method_forces_bn_eval  # noqa: E402
from experiments.benchmark_training import evaluate, fit_steps  # noqa: E402
from experiments.minimal_methods_benchmark import configure_method  # noqa: E402
from src.adapters.bnpa_sg import BNPASGConfig  # noqa: E402
from src.data.benchmark import make_loaders  # noqa: E402

CPU = torch.device("cpu")
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def sites(args):
    return adapt.site_names(args)


@pytest.fixture(scope="module")
def calib(args, split, source_state, sites):
    c = adapt.target_calibration(args, split, 1, source_state, sites, CPU)
    c.source, c.source_mean = adapt.source_moments(args, source_state, sites, split.pretrain[0][:256], CPU)
    return c


def test_sites_are_the_convolutions_of_the_backbone(args, sites):
    assert len(sites) == 11  # three blocks of three convolutions, two projection shortcuts
    plain = adapt.new_backbone(args, CPU)
    assert all(isinstance(plain.get_submodule(s), torch.nn.Conv2d) for s in sites)


def test_calibration_covers_every_site(args, calib, sites):
    plain = adapt.new_backbone(args, CPU)
    for s in sites:
        c_in = plain.get_submodule(s).in_channels
        for table in (calib.source, calib.target, calib.probe, calib.probe_act):
            assert table[s].shape == (c_in, c_in)
            assert torch.allclose(table[s], table[s].T, atol=1e-9 * float(table[s].abs().max()))
        assert calib.target_grads[s].shape == plain.get_submodule(s).weight.shape


def test_calibration_uses_the_batches_the_trial_draws(args, split, source_state):
    """The AdaBN batch and the first adaptation batch are the two `target_batches` returns."""
    (x0, _), (x1, _) = adapt.target_batches(args, split, 5)
    seen = []
    loaders = make_loaders(split, args.batch_size, 5)
    model = adapt.new_backbone(args, CPU)
    model.load_state_dict(source_state)
    targs = protocol.trial_args(args, method=upstream.MEMFLORA, rank=2)
    configure_method(model, targs)
    handle = model.register_forward_pre_hook(lambda _m, inputs: seen.append(inputs[0].detach().clone()))
    calibrate_minimal_adabn_if_needed(model, upstream.MEMFLORA, loaders["Shift_train"], targs, CPU)
    adabn_batch = seen[0]
    seen.clear()
    adapt.fit_logged(model, loaders["Shift_train"], loaders["Shift_test"], [1], 1e-3, 0.0, CPU, args.backbone, True)
    handle.remove()
    assert torch.equal(adabn_batch, x0)
    assert torch.equal(seen[0], x1)


@pytest.mark.parametrize("name", ["gauss", "sg"])
def test_fit_logged_ends_where_the_release_does(args, split, source_state, name):
    """Our loop and the release's `fit_steps` give the same metrics from the same start."""
    variant = get(name)
    results = []
    for ours in (True, False):
        loaders = make_loaders(split, args.batch_size, 3)
        model = adapt.new_backbone(args, CPU)
        model.load_state_dict(source_state)
        targs = protocol.trial_args(args, method=variant.method, rank=2)
        adapt.seed_everything(11)
        configure_method(model, targs)
        calibrate_minimal_adabn_if_needed(model, variant.method, loaders["Shift_train"], targs, CPU)
        sg = BNPASGConfig(protocol.SG_P_LR, protocol.SG_P_WEIGHT_DECAY) if variant.sg else None
        common = dict(force_bn_eval=method_forces_bn_eval(variant.method), sg_config=sg)
        if ours:
            metrics = adapt.fit_logged(model, loaders["Shift_train"], loaders["Shift_test"], [3, 6], 1e-3,
                                       args.adapt_weight_decay, CPU, args.backbone, **common)[-1]
        else:
            metrics = fit_steps(model, loaders["Shift_train"], loaders["Shift_test"], 6, 1e-3,
                                args.adapt_weight_decay, 3, CPU, args.backbone, **common)
        results.append(metrics)
    for key in ("loss", "accuracy", "macro_f1"):
        assert math.isclose(results[0][key], results[1][key], rel_tol=1e-6, abs_tol=1e-9), key


def test_trials_of_one_seed_are_reproducible_and_paired(args, split, source_state, calib):
    a = adapt.run_trial(args, split, 1, 101, source_state, get("gauss"), 2, [2], CPU, calib)
    b = adapt.run_trial(args, split, 1, 101, source_state, get("gauss"), 2, [2], CPU, calib)
    assert a["history"][0]["macro_f1"] == b["history"][0]["macro_f1"]
    assert a["capture"] == b["capture"]
    sg = adapt.run_trial(args, split, 1, 101, source_state, get("sg"), 2, [2], CPU, calib)
    assert sg["capture"] == a["capture"]  # the same Gaussian draw, before replay trains it


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_every_variant_adapts(args, split, source_state, calib, name):
    out = adapt.run_trial(args, split, 1, 101, source_state, get(name), 4, [2, 4], CPU, calib)
    assert [h["step"] for h in out["history"]] == [2, 4]
    assert all(math.isfinite(h["loss"]) and 0.0 <= h["macro_f1"] <= 1.0 for h in out["history"])
    assert out["n_sites"] == 11 and out["trainable_params"] > 0
    assert 0.0 <= out["capture"]["total"] <= 1.0


def _adapted(args, source_state, variant, rank, calib, seed=7):
    model = adapt.new_backbone(args, CPU)
    model.load_state_dict(source_state)
    adapt.seed_everything(seed)
    configure_method(model, protocol.trial_args(args, method=variant.method, rank=rank))
    adapt.apply_variant(model, variant, rank, calib, torch.Generator().manual_seed(seed))
    return model


def test_calibrated_bottleneck_has_the_gaussian_energy(args, source_state, calib):
    for name, table in (("pca-src", calib.source), ("pca-tgt", calib.target), ("grad", calib.target)):
        for site, block in adapt.adapter_blocks(_adapted(args, source_state, get(name), 4, calib)):
            p = adapt.get_projection(block).double()
            energy = torch.diagonal(p @ table[site] @ p.T)
            want = pj.total_energy(table[site]) / 4
            assert torch.allclose(energy, torch.full((4,), want, dtype=torch.float64), rtol=1e-3), (name, site)


def test_rows_without_the_mean_direction(args, source_state, calib):
    for name, means, table in (("gauss-nomean", calib.target_mean, calib.target),
                               ("pca-tgt-nomean", calib.target_mean, calib.target),
                               ("gauss-nomean-src", calib.source_mean, calib.source)):
        for site, block in adapt.adapter_blocks(_adapted(args, source_state, get(name), 2, calib)):
            p = adapt.get_projection(block).double()
            mean = means[site]
            scale = float(torch.linalg.norm(p)) * float(torch.linalg.norm(mean))
            assert float((p @ mean).abs().max()) < 1e-4 * max(scale, 1e-30), (name, site)
            energy = torch.diagonal(p @ table[site] @ p.T)
            want = pj.total_energy(table[site]) / 2
            assert torch.allclose(energy, torch.full((2,), want, dtype=torch.float64), rtol=1e-3), (name, site)


def test_mean_removal_can_keep_the_drawn_scale(args, source_state, calib):
    kept = dict(adapt.adapter_blocks(_adapted(args, source_state, get("gauss-nomean-keep"), 2, calib)))
    drawn = dict(adapt.adapter_blocks(_adapted(args, source_state, get("gauss"), 2, calib)))
    for site in kept:
        a, b = adapt.get_projection(kept[site]).double(), adapt.get_projection(drawn[site]).double()
        assert torch.allclose(a, pj.remove_direction(b, calib.target_mean[site]).to(a), rtol=1e-4, atol=1e-6)


def test_centred_bottleneck_has_zero_mean_and_the_gaussian_energy(args, source_state, calib):
    for name in ("cal", "select-cal"):
        for site, block in adapt.adapter_blocks(_adapted(args, source_state, get(name), 2, calib)):
            p = adapt.get_projection(block).double()
            mean, sigma = calib.target_mean[site].double(), calib.target[site].double()
            assert block.proj_mean_center is True
            shift = block.projection_shift.double()
            assert torch.allclose(p @ mean + shift, torch.zeros(2, dtype=torch.float64),
                                  atol=1e-4 * float((p @ mean).abs().max() + 1e-12)), (name, site)
            centred = torch.diagonal(p @ (sigma - torch.outer(mean, mean)) @ p.T)
            want = pj.total_energy(sigma) / 2
            assert torch.allclose(centred, torch.full((2,), want, dtype=torch.float64), rtol=1e-3), (name, site)


def test_bias_variants_train_a_bias(args, split, source_state, calib):
    model = _adapted(args, source_state, get("gauss-nomean-bias"), 2, calib)
    blocks = adapt.adapter_blocks(model)
    assert all(block.U.bias is not None and block.U.bias.requires_grad for _s, block in blocks)
    out = adapt.run_trial(args, split, 1, 101, source_state, get("gauss-nomean-bias"), 2, [3], CPU, calib)
    plain = adapt.run_trial(args, split, 1, 101, source_state, get("gauss-nomean"), 2, [3], CPU, calib)
    assert out["trainable_params"] > plain["trainable_params"]


def test_shares_are_ordered_and_bounded(args, source_state, calib):
    """The principal subspace explains the most input energy; every share lies in [0, 1]."""
    out = {name: adapt.captured(_adapted(args, source_state, get(name), 2, calib), calib)
           for name in ("gauss", "pca-tgt", "grad")}
    for shares in out.values():
        for key in ("total", "functional", "explained"):
            assert -1e-9 <= shares[key] <= 1.0 + 1e-9
        assert all(len(v) == 5 for v in shares["sites"].values())
        assert 0.0 <= shares["input_mean"] <= 1.0 and 0.0 <= shares["gradient_mean"] <= 1.0
    assert out["gauss"]["input_mean"] == out["pca-tgt"]["input_mean"]  # a property of the batch, not of P
    assert out["pca-tgt"]["explained"] >= out["gauss"]["explained"] - 1e-9
    assert out["gauss"]["explained"] > out["gauss"]["total"]  # low-rank inputs: explained far above r/C


def test_selection_is_a_channel_sparse_weight_update(args, source_state, calib):
    """With a selection projection, U P is nonzero only in the kernel slices of the selected channels."""
    for name in ("select-norm", "select", "select-top", "select-top-src"):
        for _site, block in adapt.adapter_blocks(_adapted(args, source_state, get(name), 2, calib)):
            p = adapt.get_projection(block)
            assert int((p != 0).sum()) == 2 and int((p != 0).any(dim=0).sum()) == 2, name
            assert int((p != 0).sum(dim=1).max()) == 1


def test_gaussian_at_unit_norm_and_signs_at_the_gaussian_norm(args, source_state, calib):
    unit = dict(adapt.adapter_blocks(_adapted(args, source_state, get("gauss-unit"), 2, calib)))
    full = dict(adapt.adapter_blocks(_adapted(args, source_state, get("gauss"), 2, calib)))
    for site in unit:
        a, b = adapt.get_projection(unit[site]), adapt.get_projection(full[site])
        assert torch.allclose(a * (a.shape[1] / 2) ** 0.5, b, rtol=1e-5, atol=1e-7)  # the same draw, rescaled
    for _site, block in adapt.adapter_blocks(_adapted(args, source_state, get("signs"), 2, calib)):
        p = adapt.get_projection(block)
        assert torch.allclose(p.abs(), torch.full_like(p, 0.5 ** 0.5))


def test_rotation_keeps_the_subspace(args, source_state, calib):
    plain = dict(adapt.adapter_blocks(_adapted(args, source_state, get("pca-src"), 4, calib)))
    rotated = dict(adapt.adapter_blocks(_adapted(args, source_state, get("pca-src-rot"), 4, calib)))
    for site in plain:
        a = torch.linalg.qr(adapt.get_projection(plain[site]).double().T).Q
        b = torch.linalg.qr(adapt.get_projection(rotated[site]).double().T).Q
        assert torch.allclose(a @ a.T, b @ b.T, atol=1e-5), site  # the same row space
        assert not torch.allclose(adapt.get_projection(plain[site]), adapt.get_projection(rotated[site]), atol=1e-3)


def test_norm_matched_orthogonal_rows(args, source_state, calib):
    for _site, block in adapt.adapter_blocks(_adapted(args, source_state, get("orth-norm"), 2, calib)):
        p = adapt.get_projection(block).double()
        c = p.shape[1]
        assert torch.allclose(p @ p.T, (c / 2) * torch.eye(2, dtype=torch.float64), atol=1e-3 * c)


def test_calibration_captures_more_of_the_gradient_than_a_random_subspace(args, source_state, calib):
    share = {
        name: adapt.captured(_adapted(args, source_state, get(name), 2, calib), calib)["total"]
        for name in ("gauss", "orth", "pca-src", "pca-tgt", "grad")
    }
    assert share["pca-src"] > 2 * share["gauss"]
    assert share["pca-tgt"] > 2 * share["gauss"]
    assert share["grad"] > 2 * share["gauss"]
    assert abs(share["gauss"] - share["orth"]) < 0.2


def test_untouched_adapter_is_the_source_model(args, split, source_state, calib):
    """With U at zero, any projection leaves the predictions of the source model unchanged."""
    loaders = make_loaders(split, args.batch_size, 1)
    plain = adapt.new_backbone(args, CPU)
    plain.load_state_dict(source_state)
    reference = evaluate(plain, loaders["Shift_test"], CPU, args.backbone)
    for name in ("gauss", "pca-src"):
        model = _adapted(args, source_state, get(name), 2, calib)
        got = evaluate(model, loaders["Shift_test"], CPU, args.backbone)
        assert math.isclose(got["loss"], reference["loss"], rel_tol=1e-4)


def test_calibration_changes_neither_the_saved_state_nor_the_operations(args, split, source_state, calib):
    """The release's profilers count the same bytes and MACs for a calibrated projection."""
    from experiments.benchmark_common import PreparedInputModel
    from src.profiling.performance import profile_one_batch_performance
    from src.profiling.sram import profile_full_sram_for_step
    from src.train import freeze_bn_eval

    x, y = split.shift_train[0][:16], split.shift_train[1][:16]
    memory, macs = {}, {}
    for name in ("gauss", "pca-src", "grad"):
        model = _adapted(args, source_state, get(name), 2, calib)
        model.train()
        freeze_bn_eval(model)
        memory[name] = profile_full_sram_for_step(PreparedInputModel(model, args.backbone), x, y,
                                                  metadata={}, force_bn_eval=True)
        macs[name] = profile_one_batch_performance(PreparedInputModel(model, args.backbone), x, {})
    for name in ("pca-src", "grad"):
        for key in ("saved_backward_state_bytes", "activation_saved_bytes", "bitmask_saved_bytes",
                    "peak_sram_estimated_bytes", "optimizer_state_bytes_estimated"):
            assert memory[name][key] == memory["gauss"][key], (name, key)
        assert macs[name]["total_train_step_macs_estimated"] == macs["gauss"]["total_train_step_macs_estimated"]
    assert memory["gauss"]["activation_saved_bytes"] > 0 and memory["gauss"]["bitmask_saved_bytes"] > 0


@pytest.mark.parametrize("name", ["gauss", "cal", "select-cal", "cal-bias"])
def test_adapted_site_folds_into_a_kernel_and_a_bias(args, split, source_state, calib, name):
    """After adaptation every site equals its frozen convolution with a merged kernel and bias."""
    import torch.nn.functional as F

    model = _adapted(args, source_state, get(name), 4, calib)
    model.eval()
    generator = torch.Generator().manual_seed(0)
    for _site, block in adapt.adapter_blocks(model):
        with torch.no_grad():  # a non-trivial adapter: random U (and bias)
            block.U.weight.copy_(0.05 * torch.randn(block.U.weight.shape, generator=generator))
            if block.U.bias is not None:
                block.U.bias.copy_(0.05 * torch.randn(block.U.bias.shape, generator=generator))
        conv = block.base_conv
        x = torch.randn(3, conv.in_channels, 24, 1, generator=generator)
        kernel, bias = adapt.merged(block)
        scale, shift = block._bn_scale_shift()
        z = F.conv2d(x, kernel, bias, stride=conv.stride, padding=conv.padding, dilation=conv.dilation)
        want = block._apply_activation(z * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1))
        with torch.no_grad():
            got = block(x)
        assert torch.allclose(got, want, rtol=1e-4, atol=1e-4 * float(want.abs().max() + 1e-12)), name
