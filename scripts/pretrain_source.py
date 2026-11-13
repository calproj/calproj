#!/usr/bin/env python3
"""Pretrain the source models of one setting, one per target domain.

    python scripts/pretrain_source.py --dataset opportunity --backbone t_resnet \
        --run-dir out/runs/source-opportunity-t_resnet-s1

The protocol is the MemFLoRA release's: 20 epochs of Adam on the source
domains, keeping the epoch with the lowest validation loss. Its scripts train
one source model per target and reuse it for every seed; ``--source-seeds 1``
does the same, and more seeds give independent source models. A checkpoint
that exists is not trained again.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

from calproj import adapt, data, protocol, runlog, upstream  # noqa: E402

upstream.enable()

from experiments.benchmark_training import evaluate, fit_epochs_best  # noqa: E402
from src.data.benchmark import make_loaders  # noqa: E402


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=protocol.DATASETS)
    ap.add_argument("--backbone", required=True, choices=protocol.BACKBONES)
    ap.add_argument("--data-root", default=os.environ.get("HAR_DATA_ROOT"), help="holds opportunity/, realdisp/, realworld/")
    ap.add_argument("--cache-dir", default=None, help="windowed arrays; default <data-root>/cache")
    ap.add_argument("--source-dir", default=os.environ.get("CALPROJ_SOURCE_DIR", "out/source"))
    ap.add_argument("--source-seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--targets", nargs="+", default=None, help="default: every target of the dataset")
    ap.add_argument("--epochs", type=int, default=None, help="default: the protocol's 20")
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    if not args.data_root:
        ap.error("--data-root or $HAR_DATA_ROOT is required")
    return args


def main() -> int:
    cli = parse()
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    if threads:
        torch.set_num_threads(threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = protocol.make_args(cli.dataset, cli.backbone, cli.data_root)
    epochs = cli.epochs or args.pretrain_epochs
    run_dir = Path(cli.run_dir)
    runlog.write_json(run_dir / "config.json", {"argv": sys.argv, "cli": vars(cli), "epochs": epochs})
    runlog.write_json(run_dir / "env.json", runlog.env_info(ROOT))
    log = runlog.UnitLog(run_dir, ("target", "source_seed"))
    arrays = data.load_arrays(args, cli.cache_dir or Path(cli.data_root) / "cache", run_dir)
    wanted = None if cli.targets is None else {str(t) for t in cli.targets}
    for seed in cli.source_seeds:
        for target in data.targets(args, arrays):
            if wanted is not None and str(target) not in wanted:
                continue
            path = data.source_path(cli.source_dir, args, target, seed)
            if path.is_file() and log.done(target=str(target), source_seed=seed):
                print(f"[skip] {path}", flush=True)
                continue
            start = time.perf_counter()
            adapt.seed_everything(seed)
            split = data.split_for(args, arrays, target, seed)
            loaders = make_loaders(split, args.batch_size, seed)
            model = adapt.build_backbone(args).to(device)  # MobileNetV2 starts from ImageNet weights
            source_metrics = fit_epochs_best(
                model, loaders["Pretrain_train"], loaders["Pretrain_val"], epochs,
                args.pretrain_lr, args.pretrain_weight_decay, device, args.backbone,
            )
            zero_shot = evaluate(model, loaders["Shift_test"], device, args.backbone)
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if not all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):
                raise FloatingPointError(f"non-finite weights in the source model of {target}")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            torch.save({"state_dict": state, "source_metrics": source_metrics, "zero_shot": zero_shot,
                        "target": str(target), "source_seed": seed, "epochs": epochs}, tmp)
            os.replace(tmp, path)
            unit = {
                "dataset": cli.dataset, "backbone": cli.backbone, "target": str(target), "source_seed": seed,
                "epochs": epochs, "source_val": source_metrics, "zero_shot": zero_shot,
                "n_source_train": int(split.pretrain[0].shape[0]), "n_source_val": int(split.pretest[0].shape[0]),
                "n_target_adapt": int(split.shift_train[0].shape[0]), "n_target_test": int(split.shift_test[0].shape[0]),
                "input_shape": list(split.pretrain[0].shape[1:]), "seconds": time.perf_counter() - start,
            }
            log.add(unit)
            log.write_metrics(dataset=cli.dataset, backbone=cli.backbone, kind="source")
            print(f"[done] {cli.dataset}/{cli.backbone} target={target} seed={seed} "
                  f"val_f1={source_metrics['macro_f1']:.3f} zero_shot_f1={zero_shot['macro_f1']:.3f} "
                  f"({unit['seconds']:.0f} s)", flush=True)
    log.write_metrics(dataset=cli.dataset, backbone=cli.backbone, kind="source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
