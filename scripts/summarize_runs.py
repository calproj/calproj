#!/usr/bin/env python3
"""Print a quick summary of adaptation runs: mean macro-F1 and captured share per variant and rank.

    python3 scripts/summarize_runs.py out/runs/main-opportunity-t_resnet-*

Reads ``units.jsonl`` (or ``metrics.json`` where only that was pulled) of each
run directory; standard library only, so it also runs on a login node. With
``--paired NAME`` it adds the mean difference to that variant over the trials
both have, with a standard error over those trials.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def load_units(run_dirs: list[str]) -> list[dict]:
    units, seen = [], set()
    for run in run_dirs:
        run = Path(run)
        rows: list[dict] = []
        if (run / "units.jsonl").is_file():
            for line in (run / "units.jsonl").read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        elif (run / "metrics.json").is_file():
            rows = json.loads((run / "metrics.json").read_text()).get("units", [])
        for row in rows:
            if "variant" not in row:
                continue
            key = (row["dataset"], row["backbone"], row["target"], row["seed"], row.get("source_seed"),
                   row["variant"], row["rank"], row.get("lr"))
            if key not in seen:
                seen.add(key)
                units.append(row)
    return units


def mean_se(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(values) / n
    if n < 2:
        return mean, float("nan")
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--step", type=int, default=None, help="evaluation step (default: the last one)")
    ap.add_argument("--paired", default="gauss", help="variant the differences are taken to")
    ap.add_argument("--lr", type=float, default=None, help="keep one learning rate")
    args = ap.parse_args()
    units = load_units(args.runs)
    if args.lr is not None:
        units = [u for u in units if u["variant"] == "zero-shot" or abs(u.get("lr", 0.0) - args.lr) < 1e-12]
    settings = sorted({(u["dataset"], u["backbone"]) for u in units})
    for dataset, backbone in settings:
        sub = [u for u in units if (u["dataset"], u["backbone"]) == (dataset, backbone)]
        trials = sorted({(u["target"], u["seed"]) for u in sub})
        print(f"\n== {dataset} / {backbone}: {len(sub)} units, {len(trials)} (target, seed) trials")
        score: dict[tuple, dict[tuple, float]] = defaultdict(dict)
        capture: dict[tuple, list[float]] = defaultdict(list)
        for u in sub:
            history = u["history"]
            point = history[-1] if args.step is None else next((h for h in history if h["step"] == args.step), None)
            if point is None:
                continue
            key = (u["variant"], u["rank"], u.get("lr", 0.0))
            score[key][(u["target"], u["seed"])] = point["macro_f1"]
            cap = u.get("capture", {})
            if cap.get("total") is not None:
                capture[key].append((cap["total"], cap.get("functional", float("nan")),
                                     cap.get("explained", float("nan"))))
        lrs = sorted({k[2] for k in score if k[0] != "zero-shot"})
        ranks = sorted({k[1] for k in score if k[0] != "zero-shot"})
        zero = [v for k, d in score.items() if k[0] == "zero-shot" for v in d.values()]
        if zero:
            print(f"zero-shot: {mean_se(zero)[0]:.4f} (n={len(zero)})")
        for lr in lrs:
            names = sorted({k[0] for k in score if k[0] != "zero-shot" and k[2] == lr})
            print(f"-- lr {lr:g}: macro-F1 [difference to {args.paired} +- s.e.] "
                  "(shares captured: weight / functional / input)")
            print(f"{'variant':16s}" + "".join(f"{'r=' + str(r):>46s}" for r in ranks))
            for name in names:
                line = f"{name:16s}"
                for rank in ranks:
                    mine = score.get((name, rank, lr), {})
                    ref = score.get((args.paired, rank, lr), {})
                    if not mine:
                        line += f"{'.':>46s}"
                        continue
                    mean, _ = mean_se(list(mine.values()))
                    common = sorted(set(mine) & set(ref))
                    diff, se = mean_se([mine[t] - ref[t] for t in common])
                    cap = capture.get((name, rank, lr), [])
                    cell = f"{mean:.4f} [{diff:+.4f}+-{se:.4f}]"
                    if cap:
                        shares = [sum(c[i] for c in cap) / len(cap) for i in range(3)]
                        cell += " (" + "/".join(f"{x:.2f}" for x in shares) + ")"
                    line += f"{cell:>46s}"
                print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
