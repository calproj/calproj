"""One adaptation trial: the MemFLoRA release, with the projection of our choosing.

The release builds the model, injects the adapters, calibrates AdaBN and
evaluates; this module only replaces the frozen projection of each adapter
before adaptation starts, and logs the evaluation at several step counts.

Trials of one (target, seed) are *paired*: every variant and rank starts from
the same source model, sees the same calibration batch and the same sequence
of adaptation batches, and the random projections of the same rank are the
same draw. The variants differ in the projection only.
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from torch import nn

from . import moments, projection, protocol, upstream
from .variants import Variant

upstream.enable()

from experiments.benchmark_adabn import calibrate_minimal_adabn_if_needed  # noqa: E402
from experiments.benchmark_common import method_forces_bn_eval, prepare_batch_inputs  # noqa: E402
from experiments.benchmark_training import SGSession, evaluate  # noqa: E402
from experiments.minimal_methods_benchmark import build_backbone, configure_method  # noqa: E402
from src.adapters.bnpa_conv_bn_act_2d import BNPAConvBNAct2D  # noqa: E402
from src.adapters.bnpa_sg import BNPASGConfig  # noqa: E402
from src.data.benchmark import make_loaders  # noqa: E402
from src.train import freeze_bn_eval  # noqa: E402

#: sketch width of the gradient calibration, in multiples of the rank
SKETCH_FACTOR = 2
#: oversampling of the streaming estimator
OVERSAMPLE = 8


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def new_backbone(args: argparse.Namespace, device: torch.device) -> nn.Module:
    """The setting's architecture, randomly initialised: its weights come from a state dict."""
    return build_backbone(protocol.trial_args(args, mobilenet_v2_pretrained=False)).to(device)


def adapter_blocks(model: nn.Module) -> list[tuple[str, BNPAConvBNAct2D]]:
    """The MemFLoRA adapters of a model, with the site each one sits on, in injection order."""
    return [
        (module.adapter_site_name, module)
        for module in model.modules()
        if isinstance(module, BNPAConvBNAct2D) and hasattr(module, "adapter_site_name")
    ]


def site_names(args: argparse.Namespace) -> list[str]:
    """The convolutions the setting adapts, by module path in the plain backbone."""
    model = new_backbone(args, torch.device("cpu"))
    configure_method(model, protocol.trial_args(args, method=upstream.MEMFLORA, rank=1))
    return [name for name, _ in adapter_blocks(model)]


def get_projection(block: BNPAConvBNAct2D) -> torch.Tensor:
    weight = block.P.weight
    return weight.detach().reshape(weight.shape[0], weight.shape[1])


def set_projection(block: BNPAConvBNAct2D, rows: torch.Tensor) -> None:
    weight = block.P.weight
    if tuple(weight.shape[2:]) != (1, 1) or tuple(rows.shape) != tuple(weight.shape[:2]):
        raise ValueError(f"projection {tuple(rows.shape)} does not fit P {tuple(weight.shape)}")
    if not torch.isfinite(rows).all():
        raise FloatingPointError("non-finite projection")
    with torch.no_grad():
        weight.copy_(rows.to(device=weight.device, dtype=weight.dtype).reshape(weight.shape))


def merged(block: BNPAConvBNAct2D) -> tuple[torch.Tensor, torch.Tensor]:
    """The kernel and the pre-normalisation bias that an adapted site is equivalent to.

    ``U * (P x + shift)`` is a convolution of ``x`` with the kernel ``U P`` (tap
    by tap) plus a constant per output channel, so after adaptation the branch
    folds into the frozen convolution: the site computes
    ``act(a * (conv(x, kernel) + bias) + b)`` with the returned ``kernel`` and
    ``bias``, and inference costs what it did before the adapter was added.
    """
    u = block.U.weight.detach()
    p = get_projection(block).to(u)
    kernel = block.base_conv.weight.detach() + block.scale * torch.einsum("ojhw,ji->oihw", u, p)
    bias = torch.zeros(u.shape[0], device=u.device, dtype=u.dtype)
    if block.proj_mean_center:
        bias = bias + block.scale * torch.einsum("ojhw,j->o", u, block.projection_shift.to(u))
    if block.U.bias is not None:
        bias = bias + block.scale * block.U.bias.detach()
    if block.base_conv.bias is not None:
        bias = bias + block.base_conv.bias.detach()
    return kernel, bias


@dataclass
class Calibration:
    """What the projections of one (target, seed) are calibrated on, and probed with.

    Each field maps a site to a float64 CPU tensor. ``source``, ``source_mean``:
    input moment and mean on source windows (offline). ``target``,
    ``target_mean``: the same on the batch that calibrates AdaBN.
    ``target_grads``: weight gradients on that batch. ``probe``, ``probe_act``:
    exact gradient moment and input moment on the first adaptation batch, which
    no calibration sees; the captured shares are measured against them.
    """

    sites: list[str]
    source: dict[str, torch.Tensor] = field(default_factory=dict)
    source_mean: dict[str, torch.Tensor] = field(default_factory=dict)
    target: dict[str, torch.Tensor] = field(default_factory=dict)
    target_mean: dict[str, torch.Tensor] = field(default_factory=dict)
    target_grads: dict[str, torch.Tensor] = field(default_factory=dict)
    probe: dict[str, torch.Tensor] = field(default_factory=dict)
    probe_act: dict[str, torch.Tensor] = field(default_factory=dict)
    probe_mean: dict[str, torch.Tensor] = field(default_factory=dict)


def _forward(model: nn.Module, args: argparse.Namespace, device: torch.device):
    return lambda x: model(prepare_batch_inputs(x.to(device), args.backbone))


def source_moments(
    args: argparse.Namespace,
    source_state: dict,
    sites: Sequence[str],
    windows: torch.Tensor,
    device: torch.device,
    chunk: int = 256,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Input moments and means of the source model on source windows: the offline calibration."""
    model = new_backbone(args, device)
    model.load_state_dict(source_state)
    batches = (windows[i : i + chunk] for i in range(0, windows.shape[0], chunk))
    return moments.activation_moments(model, sites, _forward(model, args, device), batches)


def target_batches(args: argparse.Namespace, split, seed: int):
    """The first two batches a trial draws: the AdaBN batch, then the first adaptation batch.

    A trial builds its loaders with ``make_loaders(split, batch, seed)``; AdaBN
    opens the first iterator on the target loader and the adaptation loop the
    second. The same calls on a fresh set of loaders give the same batches.
    """
    loader = make_loaders(split, args.batch_size, seed)["Shift_train"]
    first = next(iter(loader))
    second = next(iter(loader))
    return first, second


def target_calibration(
    args: argparse.Namespace,
    split,
    seed: int,
    source_state: dict,
    sites: Sequence[str],
    device: torch.device,
) -> Calibration:
    """Target-side statistics of one (target, seed), on the batches the trial will draw.

    A fresh set of loaders with the trial's seed yields the trial's batches:
    the first iterator gives the batch AdaBN calibrates on, the second one
    starts the adaptation.
    """
    model = new_backbone(args, device)
    model.load_state_dict(source_state)
    forward = _forward(model, args, device)
    (x0, y0), (x1, y1) = target_batches(args, split, seed)
    calib = Calibration(sites=list(sites))
    calib.target, calib.target_mean = moments.activation_moments(model, sites, forward, [x0])
    calib.target_grads = moments.weight_gradients(model, sites, forward, x0, y0.to(device))
    calib.probe_act, calib.probe_mean = moments.activation_moments(model, sites, forward, [x1])
    probe_grads = moments.weight_gradients(model, sites, forward, x1, y1.to(device))
    calib.probe = {s: projection.sketch_gradient_moment(g, None) for s, g in probe_grads.items()}
    return calib


def _reference(variant: Variant, site: str, calib: Calibration) -> tuple[torch.Tensor, torch.Tensor]:
    """The input moment and mean a variant measures its scale (and ranks channels) on."""
    if variant.uses_source:
        return calib.source[site], calib.source_mean[site]
    return calib.target[site], calib.target_mean[site]


def _directions(
    variant: Variant, site: str, rank: int, calib: Calibration, generator: torch.Generator
) -> torch.Tensor:
    """Directions (``C x rank``) of a calibrated or selection variant at a site."""
    reference = None if variant.data_free else _reference(variant, site, calib)[0]
    if variant.subspace == "select-random":
        channels = calib.target[site].shape[0] if reference is None else reference.shape[0]
        among = None if reference is None else projection.active_channels(reference)
        return projection.selection_directions(projection.random_channels(channels, rank, generator, among), channels)
    if variant.subspace == "select-top":
        return projection.selection_directions(projection.top_energy_channels(reference, rank), reference.shape[0])
    if variant.subspace == "activation-source":
        moment = calib.source[site]
    elif variant.subspace in ("activation-target", "activation-target-stream"):
        moment = calib.target[site]
    elif variant.subspace == "gradient-target":
        moment = projection.sketch_gradient_moment(calib.target_grads[site], SKETCH_FACTOR * rank, generator)
    else:  # pragma: no cover - Variant validates its fields
        raise ValueError(variant.subspace)
    if variant.nomean:
        moment = projection.deflate(moment, calib.target_mean[site])
    if variant.subspace == "activation-target-stream":
        directions, _ = projection.randomized_eigenspace(moment, rank, OVERSAMPLE, generator)
    else:
        directions, _ = projection.top_eigenspace(moment, rank)
    if variant.rotate:  # the same subspace, spanned by another orthonormal basis
        directions = directions @ projection.random_directions(directions.shape[1], directions.shape[1], generator)
    return directions


def _balanced(variant: Variant, directions: torch.Tensor, site: str, calib: Calibration) -> torch.Tensor:
    """Rows along ``directions`` whose bottleneck channels carry ``tr(Sigma)/r`` each.

    For a centred bottleneck the energy of a channel is measured on the
    covariance, and still set to the uncentred ``tr(Sigma)/r``: what the
    Gaussian projection carries, mean included.
    """
    moment, mean = _reference(variant, site, calib)
    if not variant.center:
        return projection.balanced_rows(directions, moment)
    covariance = moment.double() - torch.outer(mean.double(), mean.double())
    return projection.balanced_rows(directions, covariance, projection.total_energy(moment))


def build_projection(
    variant: Variant,
    site: str,
    rank: int,
    calib: Calibration | None,
    generator: torch.Generator,
    drawn: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """The ``rank x C`` projection of a variant at a site; ``None`` keeps the release's draw.

    ``drawn`` is the release's own projection at the site, which the variants
    with the ``upstream`` subspace rescale.
    """
    if variant.subspace == "upstream" and variant.scale == "upstream" and not variant.nomean:
        return None
    if variant.subspace in ("upstream", "signs"):
        if drawn is None:
            raise ValueError(f"variant {variant.name} starts from the drawn projection")
        if variant.subspace == "signs":
            rows = projection.rademacher_rows(drawn.shape[1], drawn.shape[0], generator)
        else:
            rows = drawn.detach().to(device="cpu", dtype=torch.float64)
        channels_per_rank = (rows.shape[1] / rows.shape[0]) ** 0.5
        if variant.scale == "norm":  # the release's orthonormal rows, at the Gaussian's norm
            return rows * channels_per_rank
        if variant.scale == "unit":  # the release's Gaussian rows, at the orthonormal norm
            return rows / channels_per_rank
        if variant.nomean:
            if calib is None:
                raise ValueError(f"variant {variant.name} needs calibration data")
            rows = projection.remove_direction(rows, _reference(variant, site, calib)[1])
        if variant.scale == "upstream":
            return rows
        if calib is None:
            raise ValueError(f"variant {variant.name} needs calibration data")
        return _balanced(variant, rows.T, site, calib)
    if calib is None:
        raise ValueError(f"variant {variant.name} needs calibration data")
    directions = _directions(variant, site, rank, calib, generator)
    if variant.scale == "norm":
        return projection.norm_matched_rows(directions)
    if variant.scale == "balanced":
        return _balanced(variant, directions, site, calib)
    if variant.scale == "energy":
        return projection.energy_matched_rows(directions, _reference(variant, site, calib)[0])
    raise ValueError(variant.scale)  # pragma: no cover


def apply_variant(
    model: nn.Module, variant: Variant, rank: int, calib: Calibration | None, generator: torch.Generator
) -> None:
    """Set the projection of every adapter as the variant says, with its bottleneck shift and bias."""
    for site, block in adapter_blocks(model):
        rows = build_projection(variant, site, rank, calib, generator, get_projection(block))
        if rows is not None:
            set_projection(block, rows)
        if variant.center:  # q = P (x - mean): the release adds `projection_shift` to the bottleneck
            mean = _reference(variant, site, calib)[1]
            shift = -(get_projection(block).double().cpu() @ mean.double())
            block.proj_mean_center = True
            with torch.no_grad():
                block.projection_shift.copy_(shift.to(block.projection_shift))
        if variant.bias:  # the release's backward already returns the gradient of a bias of U
            weight = block.U.weight
            block.U.bias = nn.Parameter(torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype))


def captured(model: nn.Module, calib: Calibration | None) -> dict:
    """What each adapter's projection captures of the probe batch, per site and overall.

    Three shares, each of them independent of the scale of the projection:
    ``weight``, of the squared norm of the weight gradient (its row space);
    ``functional``, of the mean squared change a gradient step makes to the
    layer outputs; ``explained``, of the input energy. Two more describe the
    probe batch itself, whatever the projection: ``input_mean``, the share of
    the input energy along the mean input, and ``gradient_mean``, the share of
    the weight-gradient energy along that direction. ``sites`` lists the five
    per site in that order; the overall figures weigh the sites by the
    denominators, so ``total`` (the weight-space share) is the share of the
    full gradient of all adapted weights.
    """
    if calib is None or not calib.probe:
        return {}
    names = ("weight", "functional", "explained", "input_mean", "gradient_mean")
    sums = {name: [0.0, 0.0] for name in names}
    sites = {}
    for site, block in adapter_blocks(model):
        rows = get_projection(block)
        moment, sigma = calib.probe[site].double(), calib.probe_act[site].double()
        mean = calib.probe_mean[site].double()
        reach = projection.reachable_moment(rows, sigma)
        gradient_energy, input_energy = projection.total_energy(moment), projection.total_energy(sigma)
        mean_energy = float(mean @ mean)
        along_mean = float(mean @ moment @ mean) / mean_energy if mean_energy > 0.0 else 0.0
        parts = {
            "weight": (projection.captured_fraction(rows, moment) * gradient_energy, gradient_energy),
            "functional": (float((reach * moment).sum()), float((sigma * moment).sum())),
            "explained": (float(torch.trace(reach)), input_energy),
            "input_mean": (mean_energy, input_energy),
            "gradient_mean": (along_mean, gradient_energy),
        }
        sites[site] = [num / den if den > 0.0 else float("nan") for num, den in parts.values()]
        for name, (num, den) in parts.items():
            if den > 0.0 and num == num:
                sums[name][0] += num
                sums[name][1] += den
    out = {name: (num / den if den > 0.0 else float("nan")) for name, (num, den) in sums.items()}
    return {"total": out["weight"], **{k: out[k] for k in names[1:]}, "sites": sites}


def fit_logged(
    model: nn.Module,
    train_loader,
    eval_loader,
    eval_steps: Sequence[int],
    lr: float,
    weight_decay: float,
    device: torch.device,
    backbone: str,
    force_bn_eval: bool = False,
    sg_config: BNPASGConfig | None = None,
    beta2: float = 0.999,
) -> list[dict]:
    """The release's ``fit_steps`` loop, keeping the evaluation at each of ``eval_steps``.

    The update is the release's, statement for statement (``reference`` runtime
    mode); ``tests/test_adapt.py`` checks that both end on the same metrics.
    ``beta2`` is Adam's second-moment decay, 0.999 in the release; it is a
    parameter here only for the experiment that varies it.
    """
    wanted = sorted(set(int(s) for s in eval_steps))
    sg = SGSession(model, sg_config) if sg_config is not None else None
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, beta2),
        eps=1e-8,
    )
    criterion = nn.CrossEntropyLoss()
    history: list[dict] = []
    iterator = iter(train_loader)
    for step in range(1, wanted[-1] + 1):
        model.train()
        if force_bn_eval:
            freeze_bn_eval(model)
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y = next(iterator)
        x = prepare_batch_inputs(x.to(device, non_blocking=True), backbone)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with sg.begin() if sg else nullcontext():
            loss = criterion(model(x), y)
            loss.backward()
        sg_stats = sg.accumulate(x, force_bn_eval) if sg else None
        optimizer.step()
        if sg:
            sg.finish(sg_stats)
        if step in wanted:
            metrics = evaluate(model, eval_loader, device, backbone)
            metrics["step"] = step
            metrics["train_loss"] = float(loss.detach())
            history.append(metrics)
    if sg:
        sg.close()
    return history


def run_trial(
    args: argparse.Namespace,
    split,
    seed: int,
    trial_seed: int,
    source_state: dict,
    variant: Variant,
    rank: int,
    eval_steps: Sequence[int],
    device: torch.device,
    calib: Calibration | None = None,
    lr: float | None = None,
    beta2: float = 0.999,
) -> dict:
    """Adapt the source model to the target split with one variant and rank.

    ``seed`` seeds the loaders (hence the batches); ``trial_seed`` seeds the
    random projection draw and everything else. Returns the evaluation history,
    the captured gradient shares at initialisation and the trainable count.
    """
    start = time.perf_counter()
    lr = float(args.adapt_lr if lr is None else lr)
    targs = protocol.trial_args(args, method=variant.method, rank=rank, adapt_lr=lr, seed=seed)
    loaders = make_loaders(split, args.batch_size, seed)
    model = new_backbone(args, device)
    model.load_state_dict(source_state)
    seed_everything(trial_seed)  # the projection draw depends on the trial seed only
    layers = configure_method(model, targs)
    apply_variant(model, variant, rank, calib, torch.Generator().manual_seed(trial_seed))
    capture = captured(model, calib)
    calibrate_minimal_adabn_if_needed(model, variant.method, loaders["Shift_train"], targs, device)
    history = fit_logged(
        model,
        loaders["Shift_train"],
        loaders["Shift_test"],
        eval_steps,
        lr,
        args.adapt_weight_decay,
        device,
        args.backbone,
        force_bn_eval=method_forces_bn_eval(variant.method),
        sg_config=BNPASGConfig(protocol.SG_P_LR, protocol.SG_P_WEIGHT_DECAY) if variant.sg else None,
        beta2=beta2,
    )
    return {
        "variant": variant.name,
        "rank": int(rank),
        "lr": lr,
        "beta2": beta2,
        "history": history,
        "capture": capture,
        "n_sites": len(layers),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in model.parameters()),
        "seconds": time.perf_counter() - start,
    }
