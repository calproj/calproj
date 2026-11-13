#!/usr/bin/env python3
"""Build the tables of the paper from the run summaries.

    python3 scripts/make_tables.py            # reads out/runs (or $CALPROJ_RUNS)

Writes the ``tabular`` bodies to ``<paper>/tables/*.tex`` and every number the
text quotes to ``<paper>/tables/numbers.tex`` as LaTeX macros, so that no
number of the manuscript is typed by hand; ``results/tables/*.csv`` holds the
same aggregates in machine-readable form. ``<paper>`` is the directory of a
manuscript the tree holds, else ``paper``. The published
baselines come from ``results/published/memflora.json``
(``scripts/extract_published.py``).

A table is skipped, with a message, while the runs it needs are missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calproj import results as R  # noqa: E402

RANKS = (2, 4, 8)
LABEL = {
    "gauss": "Gaussian (MemFLoRA)",
    "gauss-unit": "Gaussian, unit-norm rows",
    "orth": "Orthogonal",
    "orth-norm": "Orthogonal, matched norm",
    "signs": "Random signs",
    "select-norm": "Channels, matched norm",
    "select": "Channels, balanced",
    "select-top-src": "Top-energy channels",
    "pca-src": "Principal, source data",
    "pca-tgt": "Principal, target batch",
    "grad": "Gradient sketch",
    "pca-tgt-nomean": "Principal, mean removed",
    "gauss-balanced": "Balanced",
    "gauss-nomean-keep": "Mean removed, scale kept",
    "gauss-nomean": "Mean removed, rebalanced",
    "cal": "Calibrated, target batch",
    "cal-src": "Calibrated, source data",
    "signs-cal": "Calibrated random signs",
    "select-cal": "Calibrated random channels",
    "gauss-bias": "Gaussian, with a bias",
    "cal-bias": "Calibrated, with a bias",
    "sg": "Replay (MemFLoRA-SG)",
    "sg-cal": "Calibrated, replay",
    "sg-nomean": "Mean removed, replay",
}
SETTING_SHORT = {
    ("opportunity", "t_resnet"): "Opp./T", ("opportunity", "mobilenet_v2"): "Opp./M",
    ("realdisp", "t_resnet"): "RD/T", ("realdisp", "mobilenet_v2"): "RD/M",
    ("realworld", "t_resnet"): "RW/T", ("realworld", "mobilenet_v2"): "RW/M",
}
SETTINGS = list(SETTING_SHORT)


def f3(x: float) -> str:
    return f"{x:.3f}"


def signed(x: float) -> str:
    """A signed difference in math mode; a value that rounds to zero carries no sign."""
    if round(x, 3) == 0:
        return "$0.000$"
    return f"${'+' if x > 0 else '-'}{abs(x):.3f}$"


def pm(mean: float, half: float) -> str:
    return f"{signed(mean)}\\,{{\\scriptsize$\\pm$\\,{half:.3f}}}"


def mean_std(mean: float, std: float) -> str:
    return f"{mean:.3f}\\,{{\\scriptsize$\\pm$\\,{std:.3f}}}"


def tex_power(x: float) -> str:
    """A learning rate as LaTeX math: 0.002 -> 2 \\times 10^{-3}, 0.01 -> 10^{-2}."""
    mantissa, exponent = f"{x:.0e}".split("e")
    power = f"10^{{{int(exponent)}}}"
    return power if mantissa == "1" else f"{mantissa} \\times {power}"


def shown(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {shown(path)}")


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {shown(path)}")


PARTIAL = False  # --partial: build from an unfinished campaign, to check the layout


def complete(table: R.Table, variant: str, settings=SETTINGS, ranks=RANKS, reference: str = "gauss") -> bool:
    """Whether the variant has run on every trial the reference has, in every setting and rank."""
    for dataset, backbone in settings:
        for rank in ranks:
            ref = table.trials(dataset, backbone, reference, rank)
            mine = table.trials(dataset, backbone, variant, rank)
            if not ref or not mine or (not PARTIAL and set(mine) != set(ref)):
                return False
    return True


def setting_mean(table: R.Table, dataset: str, backbone: str, variant: str, ranks=RANKS) -> float:
    """Macro-F1 of a variant in a setting, averaged over the ranks (each rank: mean over its trials)."""
    return R.mean([table.pooled(dataset, backbone, variant, rank)[0] for rank in ranks])


def overall_mean(table: R.Table, variant: str, settings=SETTINGS, ranks=RANKS) -> float:
    """The mean over the setting-rank configurations, as the source paper averages its methods."""
    return R.mean([table.pooled(d, b, variant, rank)[0] for d, b in settings for rank in ranks])


def setting_difference(table: R.Table, dataset: str, backbone: str, variant: str, reference: str,
                       ranks=RANKS) -> tuple[float, float, int]:
    """Paired difference to the reference in one setting, averaged over the ranks, with seeds as replicates."""
    m, half, seeds, _ = table.overall_difference(variant, reference, ranks, [(dataset, backbone)])
    return m, half, seeds


# --------------------------------------------------------------------------- tables

def table_main(table: R.Table, published: dict, out: Path, csv_dir: Path) -> None:
    ours = ["gauss", "cal", "sg", "sg-cal"]
    if not all(complete(table, v) for v in ours):
        print("main table: runs incomplete, skipped")
        return
    pub = {(r["dataset"], r["backbone"], r["method"], r["rank"]): r for r in published["headline"]}
    lines = [
        r"\begin{tabular}{@{}lc cccc cccc@{}}", r"\toprule",
        r" & & \multicolumn{4}{c}{Published~\cite{akbulut2026memflora}} & \multicolumn{4}{c}{This work} \\",
        r"\cmidrule(lr){3-6}\cmidrule(l){7-10}",
        r"Setting & $r$ & LoRA-C & LoRA-Edge & MemFLoRA & MemFLoRA-SG & Gaussian & Calibrated & "
        r"Gaussian, replay & Calibrated, replay \\", r"\midrule",
    ]
    rows = []
    for index, (dataset, backbone) in enumerate(SETTINGS):
        for rank in RANKS:
            cells = [SETTING_SHORT[(dataset, backbone)] if rank == RANKS[0] else "", str(rank)]
            record = [dataset, backbone, rank]
            for method in ("lora-c", "lora-edge", "memflora", "memflora-sg"):
                p = pub[(dataset, backbone, method, rank)]
                cells.append(mean_std(p["mean"], p["std"]))
                record += [p["mean"], p["std"]]
            for variant in ours:
                m, s, n = table.pooled(dataset, backbone, variant, rank)
                cells.append(mean_std(m, s))
                record += [round(m, 4), round(s, 4), n]
            lines.append(" & ".join(cells) + r" \\")
            rows.append(record)
        if index < len(SETTINGS) - 1:
            lines.append(r"\midrule" if backbone == "mobilenet_v2" else r"\cmidrule(l){1-10}")
    lines.append(r"\midrule")
    means = ["Mean", ""]
    for method in ("lora-c", "lora-edge", "memflora", "memflora-sg"):
        means.append(f3(R.mean([pub[(d, b, method, r)]["mean"] for d, b in SETTINGS for r in RANKS])))
    means += [f3(overall_mean(table, v)) for v in ours]
    lines += [" & ".join(means) + r" \\", r"\botrule", r"\end{tabular}"]
    write(out / "main.tex", lines)
    header = ["dataset", "backbone", "rank"]
    for method in ("lora_c", "lora_edge", "memflora", "memflora_sg"):
        header += [f"published_{method}_mean", f"published_{method}_std"]
    for variant in ours:
        header += [f"{variant}_mean", f"{variant}_std", f"{variant}_trials"]
    write_csv(csv_dir / "main.csv", header, rows)


def table_rankfree(table: R.Table, published: dict, out: Path, csv_dir: Path) -> None:
    pub = {(r["dataset"], r["backbone"], r["method"]): r for r in published["headline"] if r["rank"] == 0}
    zero = {s: list(table.trials(s[0], s[1], R.ZERO_SHOT, 0).values()) for s in SETTINGS}
    if not all(zero.values()):
        print("rank-free table: runs incomplete, skipped")
        return
    lines = [r"\begin{tabular}{@{}l ccccc@{}}", r"\toprule",
             r" & \multicolumn{4}{c}{Published~\cite{akbulut2026memflora}} & This work \\",
             r"\cmidrule(lr){2-5}\cmidrule(l){6-6}",
             r"Setting & Zero-shot & Bias tuning & BN tuning & Full fine-tuning & Zero-shot \\", r"\midrule"]
    rows = []
    for dataset, backbone in SETTINGS:
        cells = [SETTING_SHORT[(dataset, backbone)]]
        record = [dataset, backbone]
        for method in ("zero-shot", "bias-tuning", "bn-tuning", "full-ft"):
            p = pub[(dataset, backbone, method)]
            cells.append(mean_std(p["mean"], p["std"]))
            record += [p["mean"], p["std"]]
        m, s = R.mean(zero[(dataset, backbone)]), R.sample_std(zero[(dataset, backbone)])
        cells.append(mean_std(m, s))
        record += [round(m, 4), round(s, 4), len(zero[(dataset, backbone)])]
        lines.append(" & ".join(cells) + r" \\")
        rows.append(record)
    lines += [r"\botrule", r"\end{tabular}"]
    write(out / "rankfree.tex", lines)
    write_csv(csv_dir / "rankfree.csv",
              ["dataset", "backbone", "published_zero_shot_mean", "published_zero_shot_std", "published_bias_mean",
               "published_bias_std", "published_bn_mean", "published_bn_std", "published_full_ft_mean",
               "published_full_ft_std", "zero_shot_mean", "zero_shot_std", "trials"], rows)


def table_scale(table: R.Table, out: Path, csv_dir: Path) -> None:
    variants = ["gauss", "orth-norm", "orth", "gauss-unit"]
    if not all(complete(table, v) for v in variants):
        print("scale table: runs incomplete, skipped")
        return
    lines = [r"\begin{tabular}{@{}ll cc cc cc@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{Row norm $\sqrt{C/r}$} & \multicolumn{2}{c}{Row norm 1} & "
             r"\multicolumn{2}{c}{Difference at equal norm} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}",
             r"Dataset & Backbone & Gauss. & Orth. & Orth. & Gauss. & $\sqrt{C/r}$ & 1 \\", r"\midrule"]
    rows = []
    for dataset, backbone in SETTINGS:
        cells = [R.DATASET_LABEL[dataset] if backbone == "t_resnet" else "", R.BACKBONE_LABEL[backbone]]
        values = [setting_mean(table, dataset, backbone, v) for v in variants]
        cells += [f3(v) for v in values]
        big = setting_difference(table, dataset, backbone, "orth-norm", "gauss")
        small = setting_difference(table, dataset, backbone, "gauss-unit", "orth")
        cells += [pm(big[0], big[1]), pm(small[0], small[1])]
        lines.append(" & ".join(cells) + r" \\")
        rows.append([dataset, backbone, *[round(v, 4) for v in values], round(big[0], 4), round(big[1], 4),
                     round(small[0], 4), round(small[1], 4)])
    lines.append(r"\midrule")
    big = table.overall_difference("orth-norm", "gauss", RANKS, SETTINGS)
    small = table.overall_difference("gauss-unit", "orth", RANKS, SETTINGS)
    lines.append(" & ".join(["Mean", "", *[f3(overall_mean(table, v)) for v in variants],
                             pm(big[0], big[1]), pm(small[0], small[1])]) + r" \\")
    lines += [r"\botrule", r"\end{tabular}"]
    write(out / "scale.tex", lines)
    write_csv(csv_dir / "scale.csv",
              ["dataset", "backbone", "gauss", "orth_norm", "orth", "gauss_unit", "orth_norm_minus_gauss",
               "half_width", "gauss_unit_minus_orth", "half_width_unit"], rows)


def share(table: R.Table, variant: str, which: str) -> float:
    """A captured share averaged over the setting-rank configurations."""
    return R.mean([table.share(d, b, variant, rank, which) for d, b in SETTINGS for rank in RANKS])


def table_differences(table: R.Table, variants: list[str], name: str, out: Path, csv_dir: Path,
                      shares: bool, reference: str = "gauss") -> None:
    variants = [v for v in variants if complete(table, v)]
    if not variants:
        print(f"{name} table: runs incomplete, skipped")
        return
    columns = "c" * len(SETTINGS)
    head = " & ".join(SETTING_SHORT[s] for s in SETTINGS)
    if shares:
        lines = [rf"\begin{{tabular}}{{@{{}}l cc {columns} c@{{}}}}", r"\toprule",
                 rf"Projection & $\rho$ & $\psi$ & {head} & All \\", r"\midrule"]
    else:
        lines = [rf"\begin{{tabular}}{{@{{}}l {columns} c@{{}}}}", r"\toprule",
                 rf"Projection & {head} & All \\", r"\midrule"]
    rows = []
    if shares:
        cells = [LABEL[reference], f"{share(table, reference, 'total'):.2f}", f"{share(table, reference, 'explained'):.2f}"]
        cells += [f3(setting_mean(table, d, b, reference)) for d, b in SETTINGS] + [f3(overall_mean(table, reference))]
        lines += [" & ".join(cells) + r" \\", r"\midrule"]
    else:
        cells = [LABEL[reference]] + [f3(setting_mean(table, d, b, reference)) for d, b in SETTINGS]
        lines += [" & ".join(cells + [f3(overall_mean(table, reference))]) + r" \\", r"\midrule"]
    for variant in variants:
        cells = [LABEL[variant]]
        record = [variant]
        if shares:
            cells += [f"{share(table, variant, 'total'):.2f}", f"{share(table, variant, 'explained'):.2f}"]
            record += [round(share(table, variant, "total"), 4), round(share(table, variant, "explained"), 4)]
        for dataset, backbone in SETTINGS:
            m, _half, _n = setting_difference(table, dataset, backbone, variant, reference)
            cells.append(signed(m))
            record.append(round(m, 4))
        m, half, seeds, configs = table.overall_difference(variant, reference, RANKS, SETTINGS)
        cells.append(pm(m, half))
        record += [round(m, 4), round(half, 4), seeds, configs]
        lines.append(" & ".join(cells) + r" \\")
        rows.append(record)
    lines += [r"\botrule", r"\end{tabular}"]
    write(out / f"{name}.tex", lines)
    header = ["variant"] + (["weight_share", "explained_share"] if shares else []) \
        + [f"{d}_{b}" for d, b in SETTINGS] + ["overall", "half_width", "seeds", "configurations"]
    write_csv(csv_dir / f"{name}.csv", header, rows)


def site_shares(units: list[dict], backbone: str) -> dict[str, tuple[float, float]]:
    """Per site of a backbone: the share of the input energy, and of the gradient energy, along the mean input.

    Both are properties of the probe batch, not of the projection: they are
    read from the Gaussian trials at rank 2 and averaged over datasets, targets
    and seeds.
    """
    sums: dict[str, list[float]] = {}
    for u in units:
        if u["backbone"] != backbone or u["variant"] != "gauss" or u["rank"] != 2:
            continue
        for site, values in u.get("capture", {}).get("sites", {}).items():
            if len(values) >= 5:
                acc = sums.setdefault(site, [0.0, 0.0, 0])
                acc[0] += values[3]
                acc[1] += values[4]
                acc[2] += 1
    return {site: (a / n, b / n) for site, (a, b, n) in sums.items() if n}


def after_rectifier(site: str) -> bool:
    """Whether a MobileNetV2 pointwise site reads the output of a rectifier (its projection convolutions do)."""
    return site.endswith(".conv.2") or site == "features.1.conv.1"


def table_sites(units: list[dict], out: Path, csv_dir: Path) -> dict[str, str]:
    """Mean shares per group of sites; returns the macros that quote them."""
    macros: dict[str, str] = {}
    rows = []
    for backbone in R.BACKBONES:
        shares = site_shares(units, backbone)
        if not shares:
            continue
        for site, (inp, grad) in sorted(shares.items()):
            group = "all" if backbone == "t_resnet" else ("rectified" if after_rectifier(site) else "linear")
            rows.append([backbone, site, group, round(inp, 4), round(grad, 4)])
    if not rows:
        print("site table: runs incomplete, skipped")
        return macros
    write_csv(csv_dir / "sites.csv", ["backbone", "site", "input", "input_mean_share", "gradient_mean_share"], rows)
    groups = {"Tres": [r for r in rows if r[0] == "t_resnet"],
              "MobRect": [r for r in rows if r[0] == "mobilenet_v2" and r[2] == "rectified"],
              "MobLin": [r for r in rows if r[0] == "mobilenet_v2" and r[2] == "linear"]}
    lines = [r"\begin{tabular}{@{}llc cc cc@{}}", r"\toprule",
             r" & & & \multicolumn{2}{c}{Input energy} & \multicolumn{2}{c}{Gradient energy} \\",
             r"\cmidrule(lr){4-5}\cmidrule(l){6-7}",
             r"Backbone & Input of the site & Sites & range & mean & range & mean \\", r"\midrule"]
    names = {"Tres": ("T-ResNet", "all sites"), "MobRect": ("MobileNetV2", "after a rectifier"),
             "MobLin": ("", "after a linear bottleneck")}
    for key, members in groups.items():
        if not members:
            continue
        inp, grad = [m[3] for m in members], [m[4] for m in members]
        macros[f"MeanShare{key}Min"], macros[f"MeanShare{key}Max"] = f"{min(inp):.2f}", f"{max(inp):.2f}"
        macros[f"MeanShare{key}Mean"] = f"{R.mean(inp):.2f}"
        macros[f"GradShare{key}Min"], macros[f"GradShare{key}Max"] = f"{min(grad):.2f}", f"{max(grad):.2f}"
        macros[f"GradShare{key}Mean"] = f"{R.mean(grad):.2f}"
        macros[f"Sites{key}"] = str(len(members))
        lines.append(" & ".join([names[key][0], names[key][1], str(len(members)),
                                 f"{min(inp):.2f} to {max(inp):.2f}", f"{R.mean(inp):.2f}",
                                 f"{min(grad):.2f} to {max(grad):.2f}", f"{R.mean(grad):.2f}"]) + r" \\")
    lines += [r"\botrule", r"\end{tabular}"]
    write(out / "sites.tex", lines)
    return macros


def trial_seconds(units: list[dict]) -> dict[str, str]:
    """Wall-clock time of a trial of the Gaussian projection (its evaluation included): the fastest and slowest setting."""
    by: dict[tuple, list[float]] = {}
    for u in units:
        if u["variant"] == "gauss" and "seconds" in u:
            by.setdefault((u["dataset"], u["backbone"]), []).append(u["seconds"])
    if len(by) < len(SETTINGS):
        return {}
    medians = [sorted(v)[len(v) // 2] for v in by.values()]
    return {"TrialSecondsMin": f"{min(medians):.0f}", "TrialSecondsMax": f"{max(medians):.0f}"}


LR_VARIANTS = [("gauss", "Gaussian"), ("gauss-nomean-keep", "Mean removed"), ("cal", "Calibrated"),
               ("sg", "Replay")]
DEFAULT_LR = 1e-3


def lr_cells(units: list[dict]) -> tuple[dict, list[float]]:
    """Macro-F1 by (dataset, backbone, variant, rank, learning rate) and trial, and the rates swept."""
    by: dict[tuple, dict[tuple, float]] = {}
    for u in units:
        if u["variant"] == R.ZERO_SHOT:
            continue
        key = (u["dataset"], u["backbone"], u["variant"], u["rank"], u["lr"])
        by.setdefault(key, {})[(u["target"], u["seed"])] = u["history"][-1]["macro_f1"]
    return by, sorted({k[4] for k in by})


def table_lr(units: list[dict], out: Path, csv_dir: Path) -> dict[str, str]:
    """Each projection at the protocol's learning rate and at the best rate of the sweep."""
    macros: dict[str, str] = {}
    by, lrs = lr_cells(units)
    ranks = (2, 8)
    needed = [(d, b, v, r, lr) for d, b in SETTINGS for v, _ in LR_VARIANTS for r in ranks for lr in lrs]
    if not lrs or not all(by.get(k) for k in needed):
        print("learning-rate table: runs incomplete, skipped")
        return macros
    lines = [r"\begin{tabular}{@{}lc cc cc cc cc@{}}", r"\toprule",
             " & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{label}}}" for _v, label in LR_VARIANTS) + r" \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(l){9-10}",
             r"Setting & $r$" + r" & $10^{-3}$ & best" * len(LR_VARIANTS) + r" \\", r"\midrule"]
    rows = []
    gaps: dict[str, list[float]] = {v: [] for v, _ in LR_VARIANTS}
    best_values: dict[str, list[float]] = {v: [] for v, _ in LR_VARIANTS}
    default_values: dict[str, list[float]] = {v: [] for v, _ in LR_VARIANTS}
    for dataset, backbone in SETTINGS:
        for rank in ranks:
            cells = [SETTING_SHORT[(dataset, backbone)] if rank == ranks[0] else "", str(rank)]
            record = [dataset, backbone, rank]
            common = set.intersection(*[set(by[(dataset, backbone, v, rank, lr)]) for v, _ in LR_VARIANTS for lr in lrs])
            for variant, _label in LR_VARIANTS:
                curve = {lr: R.mean([by[(dataset, backbone, variant, rank, lr)][t] for t in sorted(common)]) for lr in lrs}
                best_lr = max(curve, key=curve.get)
                cells += [f3(curve[DEFAULT_LR]), f"{curve[best_lr]:.3f}\\,{{\\scriptsize({best_lr * 1e3:g})}}"]
                record += [round(curve[DEFAULT_LR], 4), round(curve[best_lr], 4), best_lr]
                gaps[variant].append(curve[best_lr] - curve[DEFAULT_LR])
                best_values[variant].append(curve[best_lr])
                default_values[variant].append(curve[DEFAULT_LR])
            lines.append(" & ".join(cells) + r" \\")
            rows.append(record)
    lines.append(r"\midrule")
    means = ["Mean", ""]
    for variant, _label in LR_VARIANTS:
        means += [f3(R.mean(default_values[variant])), f3(R.mean(best_values[variant]))]
    lines += [" & ".join(means) + r" \\", r"\botrule", r"\end{tabular}"]
    write(out / "lr.tex", lines)
    header = ["dataset", "backbone", "rank"]
    for variant, _label in LR_VARIANTS:
        header += [f"{variant}_default", f"{variant}_best", f"{variant}_best_lr"]
    write_csv(csv_dir / "lr.csv", header, rows)
    key = {"gauss": "Gauss", "gauss-nomean-keep": "NoMeanKeep", "cal": "Cal", "sg": "SG"}
    for variant, _label in LR_VARIANTS:
        macros[f"Lr{key[variant]}Default"] = f3(R.mean(default_values[variant]))
        macros[f"Lr{key[variant]}Best"] = f3(R.mean(best_values[variant]))
        macros[f"Lr{key[variant]}Gap"] = f3(R.mean(gaps[variant]))
    macros["LrCalOverGaussBest"] = signed(R.mean(best_values["cal"]) - R.mean(best_values["gauss"]))
    macros["LrKeepOverGaussBest"] = signed(R.mean(best_values["gauss-nomean-keep"]) - R.mean(best_values["gauss"]))
    macros["LrTrials"] = str(sum(len(by[(d, b, "gauss", 2, DEFAULT_LR)]) for d, b in SETTINGS))
    # the two-by-two design, each projection at its own best rate
    best_rates: dict[str, list[float]] = {}
    for variant, name in (("gauss", "Gauss"), ("orth", "Orth"), ("orth-norm", "OrthNorm"), ("gauss-unit", "GaussUnit")):
        if all(by.get((d, b, variant, r, lr)) for d, b in SETTINGS for r in ranks for lr in lrs):
            for dataset, backbone in SETTINGS:
                for rank in ranks:
                    curve = {lr: R.mean(list(by[(dataset, backbone, variant, rank, lr)].values())) for lr in lrs}
                    best_rates.setdefault(variant, []).append(max(curve, key=curve.get))
    if len(best_rates) == 4:
        for key, members in (("Norm", ("gauss", "orth-norm")), ("Unit", ("orth", "gauss-unit"))):
            rates = sorted(rate for v in members for rate in best_rates[v])
            macros[f"LrBestMedian{key}"] = tex_power(rates[(len(rates) - 1) // 2])
            macros[f"LrBestMin{key}"] = tex_power(rates[0])
            macros[f"LrBestMax{key}"] = tex_power(rates[-1])
    for variant, name in (("orth", "Orth"), ("orth-norm", "OrthNorm"), ("gauss-unit", "GaussUnit")):
        if all(by.get((d, b, variant, r, lr)) for d, b in SETTINGS for r in ranks for lr in lrs):
            best, default = [], []
            for dataset, backbone in SETTINGS:
                for rank in ranks:
                    common = set(by[(dataset, backbone, variant, rank, lrs[0])])
                    for lr in lrs:
                        common &= set(by[(dataset, backbone, variant, rank, lr)])
                    curve = {lr: R.mean([by[(dataset, backbone, variant, rank, lr)][t] for t in sorted(common)])
                             for lr in lrs}
                    best.append(max(curve.values()))
                    default.append(curve[DEFAULT_LR])
            macros[f"Lr{name}Best"] = f3(R.mean(best))
            macros[f"Lr{name}Default"] = f3(R.mean(default))
    return macros


BUDGET_VARIANTS = [("cal", "Calibrated", "Cal"), ("select", "Channels, balanced", "Select"),
                   ("pca-src", "Principal, source data", "PcaSrc"),
                   ("sg", "Replay (MemFLoRA-SG)", "SG"), ("sg-cal", "Calibrated, replay", "SGCal")]
BUDGET_STEPS = {5: "Five", 10: "Ten", 20: "Twenty", 50: "Fifty", 100: "Hundred", 250: "TwoFifty"}


def table_budget(units: list[dict], out: Path, csv_dir: Path) -> dict[str, str]:
    """Macro-F1 after 5 to 250 steps of one run: the Gaussian projection, and each paired difference to it."""
    macros: dict[str, str] = {}
    ranks = (2, 8)
    steps = sorted({h["step"] for u in units if u["variant"] != R.ZERO_SHOT for h in u["history"]})
    tables = {step: R.Table(units, step=step) for step in steps}
    names = [v for v, _label, _key in BUDGET_VARIANTS]
    if not steps or not all(step in tables for step in BUDGET_STEPS) or not all(
            complete(tables[steps[-1]], v, ranks=ranks) for v in names):
        print("budget table: runs incomplete, skipped")
        return macros
    shown_steps = list(BUDGET_STEPS)
    lines = [rf"\begin{{tabular}}{{@{{}}l {'c' * len(shown_steps)}@{{}}}}", r"\toprule",
             rf" & \multicolumn{{{len(shown_steps)}}}{{c}}{{Adaptation steps}} \\",
             rf"\cmidrule(l){{2-{len(shown_steps) + 1}}}",
             "Projection & " + " & ".join(str(step) for step in shown_steps) + r" \\", r"\midrule",
             " & ".join([LABEL["gauss"]] + [f3(overall_mean(tables[step], "gauss", ranks=ranks))
                                            for step in shown_steps]) + r" \\", r"\midrule"]
    rows = [["gauss", "mean"] + [round(overall_mean(tables[step], "gauss", ranks=ranks), 4) for step in steps]]
    seeds, widest = 0, 0.0
    for variant, label, key in BUDGET_VARIANTS:
        cells, record, halves = [label], [variant, "difference"], [variant, "half_width"]
        for step in steps:
            m, half, seeds, _configs = tables[step].overall_difference(variant, "gauss", ranks, SETTINGS)
            record.append(round(m, 4))
            halves.append(round(half, 4))
            if step in BUDGET_STEPS:
                cells.append(signed(m))
                widest = max(widest, half)
                macros[f"Steps{key}Diff{BUDGET_STEPS[step]}"] = signed(m)
                macros[f"Steps{key}Half{BUDGET_STEPS[step]}"] = f"{half:.3f}"
                macros[f"Steps{key}DiffAbs{BUDGET_STEPS[step]}"] = f"{abs(m):.3f}"
        lines.append(" & ".join(cells) + r" \\")
        rows += [record, halves]
    lines += [r"\botrule", r"\end{tabular}"]
    write(out / "budget.tex", lines)
    write_csv(csv_dir / "budget.csv", ["variant", "quantity"] + [f"steps_{step}" for step in steps], rows)
    for step, name in BUDGET_STEPS.items():
        macros[f"StepsGauss{name}"] = f3(overall_mean(tables[step], "gauss", ranks=ranks))
        macros[f"StepsCal{name}"] = f3(overall_mean(tables[step], "cal", ranks=ranks))
    macros["StepsSeeds"] = str(seeds)
    macros["StepsMaxHalf"] = f3(widest)  # the widest 95 % half-width among the cells of the table
    macros["StepsSelectMaxAbs"] = f3(max(abs(tables[step].overall_difference("select", "gauss", ranks, SETTINGS)[0])
                                         for step in steps))
    macros["StepsTrials"] = str(sum(len(tables[50].trials(d, b, "gauss", 2)) for d, b in SETTINGS))
    return macros


MB = 10 ** 6  # decimal megabytes, the unit of the published memory figures


def table_cost(runs: Path | None, published: dict, out: Path, csv_dir: Path) -> dict[str, str]:
    """Saved state, peak training-state memory and operations of a step, per projection (batch 64)."""
    macros: dict[str, str] = {}
    root = runs if runs is not None else R.runs_root()
    units = []
    for run in sorted(root.glob("cost-*")):
        units += (R.read_metrics(run) or {}).get("units", [])
    cell = {(u["backbone"], u["variant"], u["rank"], u["batch"]): u for u in units}
    variants = [("gauss", "Gaussian"), ("cal", "Calibrated"), ("signs", "Random signs"),
                ("select", "Random channels"), ("pca-src", "Principal subspace"), ("sg", "Gaussian, replay")]
    if not all((b, v, r, 64) in cell for b in R.BACKBONES for v, _ in variants for r in RANKS):
        print("cost table: runs incomplete, skipped")
        return macros
    lines = [r"\begin{tabular}{@{}ll ccc c c c@{}}", r"\toprule",
             r" & & \multicolumn{3}{c}{Saved state [MB]} & Peak & Step & Stored \\", r"\cmidrule(lr){3-5}",
             r"Backbone & Projection & bottleneck & masks & total & [MB] & [GMAC] & [kB] \\", r"\midrule"]
    rows = []
    for backbone in R.BACKBONES:
        for index, (variant, label) in enumerate(variants):
            u = cell[(backbone, variant, 2, 64)]
            m = u["memory"]
            stored = 0.0 if variant in ("gauss", "signs", "select", "sg") else 4 * u["projection_params"] / 1e3
            if variant == "cal":  # the drawn projection is regenerated from its seed; 2r numbers per site are kept
                stored = 4 * 2 * 2 * u["n_sites"] / 1e3
            macs = u["macs"]["total_train_step_macs_estimated"]
            if variant == "select":  # a selection needs no arithmetic: forward and backward counts of P removed
                macs -= 2 * u["projection_macs"]
            cells = [R.BACKBONE_LABEL[backbone] if index == 0 else "", label,
                     f"{m['activation_saved_bytes'] / MB:.2f}", f"{m['bitmask_saved_bytes'] / MB:.2f}",
                     f"{m['saved_backward_state_bytes'] / MB:.2f}", f"{m['peak_sram_estimated_bytes'] / MB:.2f}",
                     f"{macs / 1e9:.2f}", f"{stored:.1f}"]
            lines.append(" & ".join(cells) + r" \\")
            rows.append([backbone, variant, m["activation_saved_bytes"], m["bitmask_saved_bytes"],
                         m["saved_backward_state_bytes"], m["peak_sram_estimated_bytes"], macs, stored])
        if backbone != R.BACKBONES[-1]:
            lines.append(r"\midrule")
    lines += [r"\botrule", r"\end{tabular}"]
    write(out / "cost.tex", lines)
    write_csv(csv_dir / "cost.csv", ["backbone", "variant", "bottleneck_bytes", "mask_bytes", "saved_bytes",
                                     "peak_bytes", "step_macs", "stored_kb"], rows)
    for backbone, tag in (("t_resnet", "Tres"), ("mobilenet_v2", "Mob")):
        g, sg = cell[(backbone, "gauss", 2, 64)], cell[(backbone, "sg", 2, 64)]
        macros[f"CostSaved{tag}"] = f"{g['memory']['saved_backward_state_bytes'] / MB:.2f}"
        macros[f"CostPeak{tag}"] = f"{g['memory']['peak_sram_estimated_bytes'] / MB:.2f}"
        macros[f"CostMaskShare{tag}"] = f"{100 * g['memory']['bitmask_saved_bytes'] / g['memory']['saved_backward_state_bytes']:.0f}"
        macros[f"CostPeakReplay{tag}"] = f"{sg['memory']['peak_sram_estimated_bytes'] / MB:.2f}"
        macros[f"CostCalKB{tag}"] = f"{4 * 2 * 2 * g['n_sites'] / 1e3:.1f}"  # 2r numbers per site at rank 2
        macros[f"CostReplayRatio{tag}"] = f"{sg['macs']['total_train_step_macs_estimated'] / g['macs']['total_train_step_macs_estimated']:.2f}"
        macros[f"CostProjShare{tag}"] = f"{100 * 2 * g['projection_macs'] / g['macs']['total_train_step_macs_estimated']:.1f}"
        for rank in RANKS:
            name = {2: "Two", 4: "Four", 8: "Eight"}[rank]
            u = cell[(backbone, "gauss", rank, 64)]
            macros[f"CostProjKB{tag}{name}"] = f"{4 * u['projection_params'] / 1e3:.0f}"
            macros[f"CostSaved{tag}{name}"] = f"{u['memory']['saved_backward_state_bytes'] / MB:.2f}"
        pub = {(r["backbone"], r["method"]): r for r in published["memory"]}
        macros[f"PubSaved{tag}"] = f"{pub[(backbone, 'MemFLoRA')]['saved_mb']['64']:.2f}"
        macros[f"PubPeak{tag}"] = f"{pub[(backbone, 'MemFLoRA')]['peak_mb']['64']:.2f}"
        same = all(cell[(backbone, v, r, 64)]["memory"]["saved_backward_state_bytes"]
                   == cell[(backbone, "gauss", r, 64)]["memory"]["saved_backward_state_bytes"]
                   for v, _ in variants if v != "sg" for r in RANKS)
        if not same:
            raise SystemExit(f"cost: the saved state of {backbone} depends on the projection; the text says it does not")
    return macros


def numbers(table: R.Table, published: dict, out: Path, extra: dict[str, str] | None = None) -> None:
    """Macros for the numbers the text quotes."""
    macros: dict[str, str] = dict(extra or {})

    def put(name: str, value: str) -> None:
        macros[name] = value

    pub = {(r["dataset"], r["backbone"], r["method"], r["rank"]): r for r in published["headline"]}
    for method, key in (("memflora", "PubMemflora"), ("memflora-sg", "PubMemfloraSG"), ("lora-c", "PubLoraC"),
                        ("lora-edge", "PubLoraEdge")):
        put(key + "Mean", f3(R.mean([pub[(d, b, method, r)]["mean"] for d, b in SETTINGS for r in RANKS])))
    for method, key in (("full-ft", "PubFullFT"), ("bias-tuning", "PubBias"), ("bn-tuning", "PubBN"),
                        ("zero-shot", "PubZeroShot")):
        put(key + "Mean", f3(R.mean([pub[(d, b, method, 0)]["mean"] for d, b in SETTINGS])))
    zero = [R.mean(list(table.trials(d, b, R.ZERO_SHOT, 0).values())) for d, b in SETTINGS
            if table.trials(d, b, R.ZERO_SHOT, 0)]
    if len(zero) == len(SETTINGS):
        put("ZeroShotMean", f3(R.mean(zero)))
    for row in published["ablation"]:
        if row["factor"] == "Init":
            key = "PubInitNormal" if row["variant"].startswith("Normal") else "PubInitOrth"
            for rank in RANKS:
                put(f"{key}R{'two' if rank == 2 else 'four' if rank == 4 else 'eight'}", f3(row[f"r{rank}"]))
    names = {"gauss": "Gauss", "cal": "Cal", "cal-src": "CalSrc", "sg": "SG", "sg-cal": "SGCal", "orth": "Orth",
             "orth-norm": "OrthNorm", "gauss-unit": "GaussUnit", "signs": "Signs", "select": "Select",
             "select-norm": "SelectNorm", "select-top-src": "SelectTop", "pca-src": "PcaSrc", "pca-tgt": "PcaTgt",
             "grad": "Grad", "pca-tgt-nomean": "PcaNoMean", "gauss-balanced": "Balanced",
             "gauss-nomean-keep": "NoMeanKeep", "gauss-nomean": "NoMean", "signs-cal": "SignsCal",
             "select-cal": "SelectCal", "gauss-bias": "GaussBias", "cal-bias": "CalBias", "sg-nomean": "SGNoMean"}
    for variant, key in names.items():
        if not complete(table, variant):
            continue
        put(key + "Mean", f3(overall_mean(table, variant)))
        put(key + "Rho", f"{share(table, variant, 'total'):.2f}")
        put(key + "Psi", f"{share(table, variant, 'explained'):.2f}")
        if variant != "gauss":
            m, half, _seeds, _configs = table.overall_difference(variant, "gauss", RANKS, SETTINGS)
            put(key + "Diff", signed(m))
            put(key + "DiffAbs", f"{abs(m):.3f}")
            put(key + "Half", f"{half:.3f}")
            per_setting = [setting_difference(table, d, b, variant, "gauss")[0] for d, b in SETTINGS]
            put(key + "DiffMin", signed(min(per_setting)))
            put(key + "DiffMax", signed(max(per_setting)))
            put(key + "DiffMaxAbs", f"{max(abs(v) for v in per_setting):.3f}")  # the largest effect on a setting
    rank_name = {2: "Rtwo", 4: "Rfour", 8: "Reight"}
    for variant, key in (("gauss", "Gauss"), ("orth", "Orth"), ("orth-norm", "OrthNorm"), ("gauss-unit", "GaussUnit"),
                         ("sg", "SG"), ("cal", "Cal")):
        if complete(table, variant):  # the setting of the published ablation
            for rank in RANKS:
                put(f"{key}OppT{rank_name[rank]}", f3(table.pooled("opportunity", "t_resnet", variant, rank)[0]))
    for variant in ("cal", "cal-src", "sg", "sg-cal", "pca-src", "pca-tgt", "grad", "gauss-nomean-keep",
                    "gauss-balanced", "select", "signs"):
        if not complete(table, variant):
            continue
        for (dataset, backbone), short in SETTING_SHORT.items():
            tag = short.replace(".", "").replace("/", "")
            m, half, _n = setting_difference(table, dataset, backbone, variant, "gauss")
            put(f"{names[variant]}Diff{tag}", signed(m))
            put(f"{names[variant]}Half{tag}", f"{half:.3f}")
        for rank in RANKS:  # the difference at one rank, over all settings
            m, half, _s, _c = table.overall_difference(variant, "gauss", [rank], SETTINGS)
            put(f"{names[variant]}Diff{rank_name[rank]}", signed(m))
    if all(complete(table, v) for v in ("gauss-unit", "orth", "orth-norm")):
        equal = [abs(setting_difference(table, d, b, v, ref)[0])
                 for d, b in SETTINGS for v, ref in (("orth-norm", "gauss"), ("gauss-unit", "orth"))]
        put("EqualNormMaxAbs", f3(max(equal)))  # largest difference at equal row norm, over the settings
    init = {("Normal" if row["variant"].startswith("Normal") else "Orth"): row
            for row in published["ablation"] if row["factor"] == "Init"}
    if len(init) == 2:
        gaps = [init["Normal"][f"r{rank}"] - init["Orth"][f"r{rank}"] for rank in RANKS]
        put("PubInitGapMin", f3(min(gaps)))
        put("PubInitGapMax", f3(max(gaps)))
    if complete(table, "gauss-unit") and complete(table, "orth"):
        m, half, _s, _c = table.overall_difference("gauss-unit", "orth", RANKS, SETTINGS)
        put("GaussUnitOverOrthDiff", signed(m))
        put("GaussUnitOverOrthHalf", f"{half:.3f}")
    if complete(table, "select-top-src") and complete(table, "select"):
        m, half, _s, _c = table.overall_difference("select-top-src", "select", RANKS, SETTINGS)
        put("SelectTopOverSelectDiff", signed(m))
        put("SelectTopOverSelectHalf", f"{half:.3f}")
    if complete(table, "pca-tgt-nomean") and complete(table, "gauss-nomean"):
        m, half, _s, _c = table.overall_difference("pca-tgt-nomean", "gauss-nomean", RANKS, SETTINGS)
        put("PcaNoMeanOverNoMeanDiff", signed(m))
        put("PcaNoMeanOverNoMeanHalf", f"{half:.3f}")
    if complete(table, "cal") and complete(table, "cal-src"):
        m, half, _s, _c = table.overall_difference("cal-src", "cal", RANKS, SETTINGS)
        put("CalSrcOverCalDiff", signed(m))
        put("CalSrcOverCalHalf", f"{half:.3f}")
    if complete(table, "cal") and complete(table, "gauss-nomean"):
        m, half, _s, _c = table.overall_difference("gauss-nomean", "cal", RANKS, SETTINGS)
        put("NoMeanOverCalDiff", signed(m))
        put("NoMeanOverCalHalf", f"{half:.3f}")
    if complete(table, "gauss"):
        put("InputMeanShare", f"{share(table, 'gauss', 'input_mean'):.2f}")
        put("GradMeanShare", f"{share(table, 'gauss', 'gradient_mean'):.2f}")
        for (dataset, backbone), short in SETTING_SHORT.items():
            tag = short.replace(".", "").replace("/", "")
            put("InputMeanShare" + tag, f"{R.mean([table.share(dataset, backbone, 'gauss', r, 'input_mean') for r in RANKS]):.2f}")
        put("Trials", str(sum(len(table.trials(d, b, "gauss", 2)) for d, b in SETTINGS)))
    for variant, reference, key in (("cal", "gauss", "CalWins"), ("sg-cal", "sg", "SGCalWinsOverSG"),
                                    ("cal", "sg", "CalWinsOverSG")):
        if complete(table, variant) and complete(table, reference):  # configurations with a positive difference
            wins = sum(table.overall_difference(variant, reference, [rank], [setting])[0] > 0
                       for setting in SETTINGS for rank in RANKS)
            put(key, str(wins))
    put("Configurations", str(len(SETTINGS) * len(RANKS)))
    if complete(table, "sg") and complete(table, "sg-cal"):
        m, half, _s, _c = table.overall_difference("sg-cal", "sg", RANKS, SETTINGS)
        put("SGCalOverSGDiff", signed(m))
        put("SGCalOverSGHalf", f"{half:.3f}")
    if complete(table, "cal") and complete(table, "sg"):
        m, half, _s, _c = table.overall_difference("cal", "sg", RANKS, SETTINGS)
        put("CalOverSGDiff", signed(m))
        put("CalOverSGHalf", f"{half:.3f}")
    lines = ["% Generated by scripts/make_tables.py: every number the text quotes."]
    lines += [f"\\newcommand{{\\num{name}}}{{{value}}}" for name, value in sorted(macros.items())]
    write(out / "numbers.tex", lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None, help="run directories (default: $CALPROJ_RUNS or out/runs)")
    ap.add_argument("--prefix", nargs="+", default=["main-"], help="run-name prefixes of the main campaign")
    ap.add_argument("--out", default=None, help="the paper directory (tables/ is created in it)")
    ap.add_argument("--partial", action="store_true", help="accept an unfinished campaign (layout check only)")
    args = ap.parse_args()
    global PARTIAL
    PARTIAL = args.partial
    paper = Path(args.out) if args.out else R.paper_dir(ROOT)
    out, csv_dir = paper / "tables", ROOT / "results" / "tables"
    published = json.loads((ROOT / "results" / "published" / "memflora.json").read_text(encoding="utf-8"))
    units = R.load_units(args.prefix, Path(args.runs) if args.runs else None)
    table = R.Table(units, step=50)
    print(f"{len(units)} units from the main campaign")
    table_main(table, published, out, csv_dir)
    table_rankfree(table, published, out, csv_dir)
    table_scale(table, out, csv_dir)
    table_differences(table, ["orth-norm", "signs", "select-norm", "select", "select-top-src", "pca-src", "pca-tgt",
                              "grad", "pca-tgt-nomean"], "subspace", out, csv_dir, shares=True)
    table_differences(table, ["gauss-balanced", "gauss-nomean-keep", "gauss-nomean", "cal", "cal-src", "signs-cal",
                              "select-cal", "gauss-bias", "cal-bias", "sg", "sg-nomean", "sg-cal"],
                      "calibration", out, csv_dir, shares=False)
    extra = table_sites(units, out, csv_dir)
    extra.update(trial_seconds(units))
    runs = Path(args.runs) if args.runs else None
    extra.update(table_lr(R.load_units(["lr-"], runs), out, csv_dir))
    extra.update(table_budget(R.load_units(["steps-"], runs), out, csv_dir))
    extra.update(table_cost(runs, published, out, csv_dir))
    independent = R.Table(R.load_units(["indep-"], runs), step=50)
    if independent.cells:
        table_differences(independent, ["orth-norm", "pca-src", "select", "cal", "cal-src", "sg", "sg-cal"],
                          "sources", out, csv_dir, shares=False)
        for variant, key in (("cal", "Cal"), ("pca-src", "PcaSrc"), ("orth-norm", "OrthNorm"), ("sg", "SG"),
                             ("sg-cal", "SGCal"), ("select", "Select"), ("cal-src", "CalSrc")):
            if complete(independent, variant):
                m, half, seeds, _configs = independent.overall_difference(variant, "gauss", RANKS, SETTINGS)
                extra["IndepSeeds"] = str(seeds)  # the seeds every setting has, over which the interval is taken
                extra[f"Indep{key}Diff"] = signed(m)
                extra[f"Indep{key}Half"] = f"{half:.3f}"
    if independent.cells:
        extra["IndepTrials"] = str(sum(len(independent.trials(d, b, "gauss", 2)) for d, b in SETTINGS))
    numbers(table, published, out, extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
