"""The evaluation protocol of the MemFLoRA release, as its scripts set it.

Each of the six dataset-backbone settings is the argument list of the matching
``third_party/memflora/scripts/0[1-6]_main_accuracy_*.sh``, parsed by the
release's own parser, so that defaults and derived fields are theirs. What this
project varies (method, rank, seeds, steps) is set per trial, not here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import upstream

upstream.enable()

from experiments.benchmark_cli import apply_dataset_defaults, parse_args  # noqa: E402
from experiments.benchmark_common import parse_float_list, parse_int_list  # noqa: E402

DATASETS = ("opportunity", "realdisp", "realworld")
BACKBONES = ("t_resnet", "mobilenet_v2")

#: scripts/common.sh: the MemFLoRA-SG projection optimiser
SG_P_LR = 1e-2
SG_P_WEIGHT_DECAY = 0.0

_COMMON = [
    "--pretrain-epochs", "20", "--steps-adapt", "50", "--batch-size", "64",
    "--reuse-first-source-model", "true", "--adapt-eval-every-steps", "50",
    "--adabn-calibration-mode", "ema_no_reset", "--adabn-calib-batches", "1",
    "--proj-init", "random", "--bnpa-bottleneck-bn", "on",
    "--profile-full-sram", "false",
    "--bnpa-sg-p-lr", "1e-2", "--bnpa-sg-p-weight-decay", "0", "--bnpa-sg-p-optimizer", "adam",
    "--bnpa-sg-p-update-every", "1", "--bnpa-sg-p-warmup-steps", "0", "--bnpa-sg-renorm-p", "none",
    "--adabn-stat-source", "target", "--train_mode_adaBN", "off", "--adapt-lr", "1e-3",
    "--rank", "2", "4", "8",
]
_DATASET = {
    "opportunity": [
        "--dataset", "opportunity", "--target-subjects", "all",
        "--window-size", "60", "--window-stride", "30",
        "--label-column", "ml_both_arms", "--window-label-rule", "majority",
    ],
    "realdisp": [
        "--dataset", "realdisp", "--target-subjects", "all",
        "--source-scenario", "ideal", "--target-scenario", "self",
        "--realdisp-feature-set", "imu13",
        "--window-size", "250", "--window-stride", "125", "--window-label-rule", "majority",
    ],
    "realworld": [
        "--dataset", "realworld", "--target-locations", "all",
        "--realworld-feature-set", "imu9", "--realworld-split-strategy", "chronological",
        "--window-size", "500", "--window-stride", "250", "--window-label-rule", "majority",
    ],
}
_BACKBONE = {
    "t_resnet": ["--backbone", "t_resnet_official", "--t-resnet-feature-maps", "64", "--adapter-layers", "all"],
    "mobilenet_v2": [
        "--backbone", "mobilenet_v2", "--mobilenet-v2-pretrained", "true", "--width-mult", "1.0",
        "--adapter-layers", "pointwise_only",
    ],
}


def make_args(dataset: str, backbone: str, data_root: str | Path) -> argparse.Namespace:
    """The release's parsed arguments for one setting; ``data_root`` holds the three dataset folders."""
    if dataset not in _DATASET or backbone not in _BACKBONE:
        raise ValueError(f"unknown setting {dataset}/{backbone}")
    argv = [
        *_DATASET[dataset], "--data-root", str(Path(data_root) / dataset), *_BACKBONE[backbone],
        *_COMMON, "--method", upstream.MEMFLORA, "--seed", "1",
    ]
    args = parse_args(argv)
    apply_dataset_defaults(args)
    # the conversions the release's main() applies after parsing
    args.steps_adapt = parse_int_list(args.steps_adapt, "--steps-adapt")
    args.adapt_lr = parse_float_list(args.adapt_lr, "--adapt-lr")[0]
    args.batch_size_values = parse_int_list(args.batch_size, "--batch-size")
    args.batch_size = args.batch_size_values[0]
    args.adabn_calib_batches = int(args.adabn_calib_batches[0])
    args.bnpa_bottleneck_bn = args.bnpa_bottleneck_bn[0]
    args.proj_init = args.proj_init[0]
    args.method = upstream.MEMFLORA
    args.rank = 2
    return args


def trial_args(args: argparse.Namespace, **overrides) -> argparse.Namespace:
    """A copy of the setting's arguments with per-trial fields replaced."""
    return argparse.Namespace(**{**vars(args), **overrides})
