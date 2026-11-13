"""Datasets, splits and source checkpoints, through the MemFLoRA release's loaders.

The release parses the raw recordings at every start, which takes minutes for
the larger datasets; the windowed arrays are cached once per setting instead.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from . import upstream

upstream.enable()

from experiments.benchmark_cli import resolve_minimal_targets  # noqa: E402
from src.data.benchmark import (  # noqa: E402
    build_minimal_split,
    load_minimal_dataset_arrays,
    minimal_target_key,
)


def cache_name(args: argparse.Namespace) -> str:
    extra = {
        "opportunity": f"{args.label_column}-{args.window_label_rule}",
        "realworld": args.realworld_feature_set,
        "realdisp": args.realdisp_feature_set,
    }[args.dataset]
    return f"{args.dataset}-w{args.window_size}-s{args.window_stride}-{extra}.pt"


def load_arrays(args: argparse.Namespace, cache_dir: Path | None, meta_dir: Path):
    """The windowed dataset of a setting, from the cache or from the raw files."""
    meta_dir = Path(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    cached = Path(cache_dir) / cache_name(args) if cache_dir else None
    if cached is not None and cached.is_file():
        return torch.load(cached, weights_only=False)
    arrays = load_minimal_dataset_arrays(args, meta_dir)
    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_name(f"{cached.name}.{os.getpid()}.tmp")
        torch.save(arrays, tmp)
        os.replace(tmp, cached)  # atomic: a concurrent job reads the old file or the new one
    return arrays


def targets(args: argparse.Namespace, arrays) -> list:
    """The target domains of the setting: subjects, body locations, or the one placement."""
    return list(resolve_minimal_targets(args, arrays))


def target_key(args: argparse.Namespace, target) -> str:
    return minimal_target_key(args, target)


def split_for(args: argparse.Namespace, arrays, target, seed: int):
    """Source train/validation and target adaptation/test tensors of one target and seed."""
    return build_minimal_split(args, arrays, target, seed)


def source_path(source_dir: Path, args: argparse.Namespace, target, source_seed: int) -> Path:
    backbone = "t_resnet" if args.backbone == "t_resnet_official" else args.backbone
    return Path(source_dir) / args.dataset / backbone / f"{target_key(args, target)}-s{source_seed}.pt"


def load_source(source_dir: Path, args: argparse.Namespace, target, source_seed: int) -> dict:
    path = source_path(source_dir, args, target, source_seed)
    if not path.is_file():
        raise FileNotFoundError(f"no source model at {path}: run scripts/pretrain_source.py first")
    return torch.load(path, map_location="cpu", weights_only=False)
