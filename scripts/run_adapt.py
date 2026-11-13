#!/usr/bin/env python3
"""Adapt the source models of one setting to their targets, one unit per trial.

    python scripts/run_adapt.py --dataset opportunity --backbone t_resnet \
        --seeds 1-20 --variants gauss pca-src pca-tgt --ranks 2 4 8 \
        --run-dir out/runs/main-opportunity-t_resnet

A unit is one (target, seed, variant, rank, learning rate): the source model of
the target is adapted for the largest of ``--eval-steps`` and evaluated on the
target test split at each of them. The units of one (target, seed) are paired
(``calproj.adapt``). Finished units are appended to ``units.jsonl``; a job that
is resubmitted skips them.

The source models come from ``scripts/pretrain_source.py``. ``--source-seed 1``
(the default) is the release's protocol, one source model per target reused
for every seed; ``--source-seed seed`` pairs seed ``s`` with source model ``s``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

from calproj import adapt, data, protocol, runlog, upstream, variants  # noqa: E402

upstream.enable()

from experiments.benchmark_training import evaluate  # noqa: E402
from src.data.benchmark import make_loaders  # noqa: E402

KEY = ("target", "seed", "source_seed", "variant", "rank", "lr")
ZERO_SHOT = "zero-shot"


def int_list(values: list[str]) -> list[int]:
    out: list[int] = []
    for value in values:
        for part in value.split(","):
            if "-" in part:
                lo, hi = part.split("-", 1)
                out.extend(range(int(lo), int(hi) + 1))
            elif part:
                out.append(int(part))
    return out


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=protocol.DATASETS)
    ap.add_argument("--backbone", required=True, choices=protocol.BACKBONES)
    ap.add_argument("--data-root", default=os.environ.get("HAR_DATA_ROOT"), help="holds opportunity/, realdisp/, realworld/")
    ap.add_argument("--cache-dir", default=None, help="windowed arrays; default <data-root>/cache")
    ap.add_argument("--source-dir", default=os.environ.get("CALPROJ_SOURCE_DIR", "out/source"))
    ap.add_argument("--source-seed", default="1", help="an integer, or 'seed' for one source model per seed")
    ap.add_argument("--seeds", nargs="+", default=["1"], help="e.g. 1-20, or 1 2 3")
    ap.add_argument("--targets", nargs="+", default=None, help="default: every target of the dataset")
    ap.add_argument("--variants", nargs="+", default=["gauss", "pca-src"], choices=sorted(variants.VARIANTS))
    ap.add_argument("--ranks", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--eval-steps", type=int, nargs="+", default=[50])
    ap.add_argument("--lrs", type=float, nargs="+", default=None, help="default: the protocol's 1e-3")
    ap.add_argument("--n-source-cal", type=int, default=1024, help="source windows the offline calibration reads")
    ap.add_argument("--beta2", type=float, default=0.999, help="Adam's second-moment decay (the release: 0.999)")
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    if not args.data_root:
        ap.error("--data-root or $HAR_DATA_ROOT is required")
    return args


def calibration_windows(windows: torch.Tensor, count: int) -> torch.Tensor:
    """A fixed random subset of the source training windows."""
    order = torch.randperm(windows.shape[0], generator=torch.Generator().manual_seed(20261110))
    return windows[order[:count]]


def main() -> int:
    cli = parse()
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    if threads:
        torch.set_num_threads(threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = protocol.make_args(cli.dataset, cli.backbone, cli.data_root)
    seeds = int_list(cli.seeds)
    lrs = cli.lrs or [float(args.adapt_lr)]
    chosen = [variants.get(name) for name in cli.variants]
    needs_source = any(v.uses_source for v in chosen)
    run_dir = Path(cli.run_dir)
    runlog.write_json(run_dir / "config.json", {
        "argv": sys.argv, "cli": vars(cli), "seeds": seeds, "lrs": lrs,
        "protocol": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, list, type(None)))},
        "sketch_factor": adapt.SKETCH_FACTOR, "oversample": adapt.OVERSAMPLE,
        "upstream_commit": upstream.UPSTREAM_COMMIT,
    })
    runlog.write_json(run_dir / "env.json", runlog.env_info(ROOT))
    log = runlog.UnitLog(run_dir, KEY)
    summary = dict(dataset=cli.dataset, backbone=cli.backbone, kind="adapt")
    arrays = data.load_arrays(args, cli.cache_dir or Path(cli.data_root) / "cache", run_dir)
    sites = adapt.site_names(args)
    wanted = None if cli.targets is None else {str(t) for t in cli.targets}
    since_write = 0

    for target_index, target in enumerate(data.targets(args, arrays)):
        if wanted is not None and str(target) not in wanted:
            continue
        sources: dict[int, tuple[dict, dict, dict]] = {}
        for seed in seeds:
            source_seed = seed if cli.source_seed == "seed" else int(cli.source_seed)
            base = dict(target=str(target), seed=seed, source_seed=source_seed)
            todo = [(v, r, lr) for lr in lrs for v in chosen for r in cli.ranks
                    if not log.done(**base, variant=v.name, rank=r, lr=lr)]
            zero_done = log.done(**base, variant=ZERO_SHOT, rank=0, lr=0.0)
            if not todo and zero_done:
                continue
            if source_seed not in sources:
                sources.clear()  # one source model in memory at a time
                checkpoint = data.load_source(cli.source_dir, args, target, source_seed)
                state = checkpoint["state_dict"]
                source_table = ({}, {})
                if needs_source:
                    source_split = data.split_for(args, arrays, target, source_seed)
                    cal = calibration_windows(source_split.pretrain[0], cli.n_source_cal)
                    source_table = adapt.source_moments(args, state, sites, cal, device)
                    del source_split
                sources[source_seed] = (state, source_table, checkpoint["source_metrics"])
            state, source_table, source_metrics = sources[source_seed]
            split = data.split_for(args, arrays, target, seed)
            common = dict(dataset=cli.dataset, backbone=cli.backbone, target_index=target_index, **base,
                          source_val_macro_f1=float(source_metrics.get("macro_f1", float("nan"))),
                          n_target_adapt=int(split.shift_train[0].shape[0]),
                          n_target_test=int(split.shift_test[0].shape[0]))
            trial_seed = 100_000 * seed + 100 * target_index + 7

            plain = adapt.new_backbone(args, device)
            plain.load_state_dict(state)
            zero = evaluate(plain, make_loaders(split, args.batch_size, seed)["Shift_test"], device, args.backbone)
            del plain
            if not zero_done:
                log.add({**common, "variant": ZERO_SHOT, "rank": 0, "lr": 0.0, "history": [{**zero, "step": 0}]})
            if not todo:
                continue
            calib = adapt.target_calibration(args, split, seed, state, sites, device)
            calib.source, calib.source_mean = source_table
            for variant, rank, lr in todo:
                out = adapt.run_trial(args, split, seed, trial_seed, state, variant, rank,
                                      cli.eval_steps, device, calib, lr, cli.beta2)
                log.add({**common, **out, "zero_shot_macro_f1": zero["macro_f1"]})
                last = out["history"][-1]
                print(f"[done] {cli.dataset}/{cli.backbone} target={target} seed={seed} {variant.name} r={rank} "
                      f"lr={lr:g} step={last['step']} f1={last['macro_f1']:.4f} (zero-shot {zero['macro_f1']:.4f}) "
                      f"capture={out['capture'].get('total', float('nan')):.3f} {out['seconds']:.1f}s", flush=True)
                since_write += 1
                if since_write >= 25:
                    log.write_metrics(**summary)
                    since_write = 0
        log.write_metrics(**summary)
    log.write_metrics(**summary, finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    print(f"finished: {len(log.units)} units in {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
