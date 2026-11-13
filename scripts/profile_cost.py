#!/usr/bin/env python3
"""Training-state memory and operation counts of one adaptation step, per projection.

    python scripts/profile_cost.py --dataset opportunity --backbone t_resnet \
        --run-dir out/runs/cost-opportunity-t_resnet

Both figures come from the MemFLoRA release's own profilers
(``src/profiling/sram.py``, ``src/profiling/performance.py``), applied to the
adapter with each projection: the saved tensors of one forward and backward
pass, split into rank-``r`` activations and bit-packed ReLU masks, and the
multiply-accumulate count of one training step. A calibrated projection
changes neither, since only the values of the frozen ``P`` differ; this script
measures that rather than assuming it.

It also counts what a calibration needs on the device: the full ``C x C``
moments, against the two ``C x (r + oversample)`` buffers of the streaming
estimator, and the ``r x C`` projection that must be stored when it cannot be
regenerated from a seed.
"""

from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

from calproj import adapt, data, protocol, runlog, upstream, variants  # noqa: E402

upstream.enable()

from experiments.benchmark_adabn import calibrate_minimal_adabn_if_needed  # noqa: E402
from experiments.benchmark_common import PreparedInputModel, method_forces_bn_eval  # noqa: E402
from experiments.minimal_methods_benchmark import configure_method  # noqa: E402
from src.data.benchmark import make_loaders  # noqa: E402
from src.profiling.performance import profile_one_batch_performance  # noqa: E402
from src.profiling.sram import profile_full_sram_for_step  # noqa: E402
from src.train import freeze_bn_eval  # noqa: E402

KEY = ("variant", "rank", "batch")
BYTES = 4  # float32


def adapter_macs(model: torch.nn.Module, args, x: torch.Tensor) -> tuple[int, int]:
    """Multiply-accumulates of one forward pass through the projections ``P`` and through the ``U``.

    The release's profiler counts both inside its forward total; they are
    separated here because a selection projection needs no arithmetic at all,
    and a sign projection only additions.
    """
    totals = [0, 0]
    handles = []

    def hook(block, inputs, output):
        batch, _c, height, width = inputs[0].shape
        rank, c_in = block.P.weight.shape[:2]
        c_out, _r, k_h, k_w = block.U.weight.shape
        totals[0] += batch * rank * c_in * height * width
        totals[1] += batch * c_out * rank * k_h * k_w * output.shape[2] * output.shape[3]

    for _site, block in adapt.adapter_blocks(model):
        handles.append(block.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            PreparedInputModel(model, args.backbone)(x)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return totals[0], totals[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=protocol.DATASETS)
    ap.add_argument("--backbone", required=True, choices=protocol.BACKBONES)
    ap.add_argument("--data-root", default=os.environ.get("HAR_DATA_ROOT"))
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--source-dir", default=os.environ.get("CALPROJ_SOURCE_DIR", "out/source"))
    ap.add_argument("--variants", nargs="+", default=["gauss", "pca-src", "pca-tgt", "sg"], choices=sorted(variants.VARIANTS))
    ap.add_argument("--ranks", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 64])
    ap.add_argument("--n-source-cal", type=int, default=1024)
    ap.add_argument("--run-dir", required=True)
    cli = ap.parse_args()
    if not cli.data_root:
        ap.error("--data-root or $HAR_DATA_ROOT is required")
    device = torch.device("cpu")  # byte and operation counts do not depend on the device
    args = protocol.make_args(cli.dataset, cli.backbone, cli.data_root)
    run_dir = Path(cli.run_dir)
    runlog.write_json(run_dir / "config.json", {"argv": sys.argv, "cli": vars(cli)})
    runlog.write_json(run_dir / "env.json", runlog.env_info(ROOT))
    log = runlog.UnitLog(run_dir, KEY)
    arrays = data.load_arrays(args, cli.cache_dir or Path(cli.data_root) / "cache", run_dir)
    target = data.targets(args, arrays)[0]
    seed = 1
    state = data.load_source(cli.source_dir, args, target, seed)["state_dict"]
    split = data.split_for(args, arrays, target, seed)
    sites = adapt.site_names(args)
    calib = adapt.target_calibration(args, split, seed, state, sites, device)
    order = torch.randperm(split.pretrain[0].shape[0], generator=torch.Generator().manual_seed(20261110))
    calib.source, calib.source_mean = adapt.source_moments(args, state, sites, split.pretrain[0][order[: cli.n_source_cal]], device)
    plain = adapt.new_backbone(args, device)
    channels = {s: plain.get_submodule(s).in_channels for s in sites}
    outputs = {s: plain.get_submodule(s).out_channels for s in sites}

    for name in cli.variants:
        variant = variants.get(name)
        for rank in cli.ranks:
            for batch in cli.batches:
                if log.done(variant=name, rank=rank, batch=batch):
                    continue
                targs = protocol.trial_args(args, method=variant.method, rank=rank, batch_size=batch)
                loaders = make_loaders(split, batch, seed)
                model = adapt.new_backbone(args, device)
                model.load_state_dict(state)
                adapt.seed_everything(seed)
                configure_method(model, targs)
                adapt.apply_variant(model, variant, rank, calib, torch.Generator().manual_seed(seed))
                calibrate_minimal_adabn_if_needed(model, variant.method, loaders["Shift_train"], targs, device)
                x, y = next(iter(loaders["Shift_train"]))
                force = method_forces_bn_eval(variant.method)
                meta = {"dataset": cli.dataset, "backbone": cli.backbone, "method": variant.method,
                        "target_domain": str(target), "rank": rank, "seed": seed, "batch_size": batch}
                memory_model = deepcopy(model)
                memory_model.train()
                if force:
                    freeze_bn_eval(memory_model)
                memory = profile_full_sram_for_step(PreparedInputModel(memory_model, args.backbone), x, y,
                                                    metadata=meta, force_bn_eval=force)
                macs = profile_one_batch_performance(PreparedInputModel(deepcopy(model), args.backbone), x, meta)
                parts = adapter_macs(model, args, x)
                width = rank + adapt.OVERSAMPLE
                log.add({
                    "dataset": cli.dataset, "backbone": cli.backbone, "variant": name, "rank": rank, "batch": batch,
                    "n_sites": len(sites),
                    "memory": {k: v for k, v in memory.items() if k.endswith("_bytes") or k.endswith("_estimated")},
                    "macs": {k: v for k, v in macs.items() if "macs" in k or k.endswith("params")},
                    "projection_macs": parts[0], "up_projection_macs": parts[1],
                    "projection_params": sum(rank * c for c in channels.values()),
                    "up_projection_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "calibration_bytes_full_all_sites": BYTES * sum(c * c for c in channels.values()),
                    "calibration_bytes_full_largest_site": BYTES * max(c * c for c in channels.values()),
                    "calibration_bytes_streaming_all_sites": BYTES * sum(2 * c * min(c, width) + min(c, width) ** 2
                                                                         for c in channels.values()),
                    "site_in_channels": channels, "site_out_channels": outputs,
                })
                unit = log.units[-1]
                print(f"[done] {name} r={rank} B={batch}: saved {unit['memory']['saved_backward_state_bytes'] / 2**20:.3f} MB "
                      f"(activations {unit['memory']['activation_saved_bytes'] / 2**20:.3f}, masks "
                      f"{unit['memory']['bitmask_saved_bytes'] / 2**20:.3f}), peak "
                      f"{unit['memory']['peak_sram_estimated_bytes'] / 2**20:.2f} MB, step "
                      f"{unit['macs']['total_train_step_macs_estimated'] / 1e6:.1f} M MACs", flush=True)
    log.write_metrics(dataset=cli.dataset, backbone=cli.backbone, kind="cost")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
