#!/usr/bin/env python3
"""Window one dataset once and cache the arrays every other script reads.

    python scripts/prepare_data.py --dataset realdisp --run-dir out/runs/data-realdisp

The MemFLoRA release parses the raw recordings at every start (minutes for
RealWorld, far longer for the 7 GB of RealDisp logs). This runs its loader
once per dataset and stores the windowed arrays under ``<data-root>/cache``;
``metrics.json`` records how many windows each target domain holds.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

from calproj import data, protocol, runlog  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=protocol.DATASETS)
    ap.add_argument("--data-root", default=os.environ.get("HAR_DATA_ROOT"), help="holds opportunity/, realdisp/, realworld/")
    ap.add_argument("--cache-dir", default=None, help="default <data-root>/cache")
    ap.add_argument("--run-dir", required=True)
    cli = ap.parse_args()
    if not cli.data_root:
        ap.error("--data-root or $HAR_DATA_ROOT is required")
    run_dir = Path(cli.run_dir)
    runlog.write_json(run_dir / "config.json", {"argv": sys.argv, "cli": vars(cli)})
    runlog.write_json(run_dir / "env.json", runlog.env_info(ROOT))
    args = protocol.make_args(cli.dataset, "t_resnet", cli.data_root)  # the windows do not depend on the backbone
    cache_dir = Path(cli.cache_dir or Path(cli.data_root) / "cache")
    start = time.perf_counter()
    arrays = data.load_arrays(args, cache_dir, run_dir)
    seconds = time.perf_counter() - start
    targets = data.targets(args, arrays)
    per_target = {}
    for target in targets:
        split = data.split_for(args, arrays, target, 1)
        per_target[str(target)] = {
            "source_train": int(split.pretrain[0].shape[0]), "source_val": int(split.pretest[0].shape[0]),
            "target_adapt": int(split.shift_train[0].shape[0]), "target_test": int(split.shift_test[0].shape[0]),
        }
    windows = arrays.windows
    runlog.write_json(run_dir / "metrics.json", {
        "dataset": cli.dataset, "kind": "data", "cache": str(cache_dir / data.cache_name(args)),
        "n_windows": int(windows.shape[0]), "channels": int(windows.shape[1]), "window": int(windows.shape[2]),
        "n_classes": int(torch.unique(arrays.labels).numel()), "targets": [str(t) for t in targets],
        "per_target": per_target, "finite": bool(torch.isfinite(windows).all()),
        "abs_max": float(windows.abs().max()), "seconds": seconds,
    })
    print(f"{cli.dataset}: {tuple(windows.shape)} windows, {len(targets)} targets, {seconds:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
