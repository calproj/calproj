#!/usr/bin/env python3
"""Build the figures of the paper from the run summaries.

    python3 scripts/make_figures.py           # reads out/runs (or $CALPROJ_RUNS)

Writes vector PDFs to ``<paper>/figures/``; ``<paper>`` is the directory of a
manuscript the tree holds, else ``paper``. A figure is
skipped, with a message, while the runs it needs are missing.

Design: one y-axis per panel, a legend whenever a panel holds several series,
colour for the identity of a projection and line style for a second attribute
(its row norm, or whether it is trained by replay), so that no series is told
apart by colour alone. The three hues are the first slots of a categorical
palette validated for colour-vision deficiency on a white surface.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calproj import results as R  # noqa: E402

WIDTH = 5.17            # width of the figures in inches: a text width of 372 pt
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1a1a1a", "#555555", "#e3e3e0"
RANKS = (2, 4, 8)
SETTINGS = [(d, b) for b in R.BACKBONES for d in R.DATASETS]  # a row of panels per backbone
ALL_SETTINGS = [(d, b) for d in R.DATASETS for b in R.BACKBONES]


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True, "lines.linewidth": 1.5, "lines.markersize": 5.5,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "figure.dpi": 150,
    })


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"CreationDate": None})   # no date: the same bytes at every build
    plt.close(fig)
    try:
        shown = path.resolve().relative_to(ROOT)
    except ValueError:
        shown = path
    print(f"wrote {shown}")


def has(table: R.Table, variants, settings, ranks) -> bool:
    return all(table.trials(d, b, v, r) for v in variants for d, b in settings for r in ranks)


def panel_title(dataset: str, backbone: str) -> str:
    return f"{R.DATASET_LABEL[dataset]}, {R.BACKBONE_LABEL[backbone]}"


# --------------------------------------------------------------------------- figures

# The projections of unit row norm are left to the scale table: their difference
# is five times the range of every other and would flatten the axis.
FOREST = [
    ("Random", [("orth-norm", "Orthogonal, matched norm"), ("signs", "Random signs"),
                ("select", "Random channels")]),
    ("Data-aware subspace", [("select-top-src", "Top-energy channels"), ("pca-src", "Principal subspace, source"),
                             ("pca-tgt", "Principal subspace, target"), ("grad", "Gradient sketch"),
                             ("pca-tgt-nomean", "Principal subspace, mean removed")]),
    ("Mean", [("gauss-balanced", "Balanced"), ("gauss-nomean-keep", "Mean removed, scale kept"),
              ("gauss-nomean", "Mean removed, rebalanced"), ("cal", "Calibrated, target batch"),
              ("cal-src", "Calibrated, source data"), ("signs-cal", "Calibrated random signs"),
              ("select-cal", "Calibrated random channels")]),
    ("Trained by replay", [("sg", "Gaussian (MemFLoRA-SG)"), ("sg-cal", "Calibrated")]),
]


def figure_forest(table: R.Table, out: Path) -> None:
    """Paired difference of every projection to the Gaussian one, over all settings and ranks."""
    names = [v for _group, items in FOREST for v, _label in items]
    if not has(table, ["gauss", *names], ALL_SETTINGS, RANKS):
        print("forest figure: runs incomplete, skipped")
        return
    rows = sum(len(items) for _g, items in FOREST) + len(FOREST)
    fig, ax = plt.subplots(figsize=(WIDTH, 0.17 * rows + 0.55))
    y, ticks, labels = 0.0, [], []
    for group, items in FOREST:
        ax.text(-0.01, y, group, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontweight="bold", color=INK)
        y += 1
        for variant, label in items:
            mean, half, _seeds, _configs = table.overall_difference(variant, "gauss", RANKS, ALL_SETTINGS)
            ax.plot([mean - half, mean + half], [y, y], color=BLUE, linewidth=1.5, solid_capstyle="round", zorder=2)
            ax.plot([mean], [y], marker="o", color=BLUE, markeredgecolor="white", markeredgewidth=1.0,
                    markersize=6, linestyle="none", zorder=3)
            ticks.append(y)
            labels.append(label)
            y += 1
    ax.axvline(0.0, color=MUTED, linewidth=0.8, zorder=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, color=INK)
    ax.set_ylim(y - 0.4, -0.6)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Difference in macro-F1 to the Gaussian projection")
    save(fig, out / "forest.pdf")


LR_SERIES = [
    ("gauss", "Gaussian, row norm $\\sqrt{C/r}$", BLUE, "-", "o", True),
    ("gauss-unit", "Gaussian, row norm 1", BLUE, "--", "o", False),
    ("orth-norm", "Orthogonal, row norm $\\sqrt{C/r}$", ORANGE, "-", "s", True),
    ("orth", "Orthogonal, row norm 1", ORANGE, "--", "s", False),
]
LR_CAL_SERIES = [
    ("gauss", "Gaussian", BLUE, "-", "o", True),
    ("gauss-nomean-keep", "Mean removed, scale kept", AQUA, "-", "^", True),
    ("cal", "Calibrated", ORANGE, "-", "s", True),
    ("sg", "Gaussian, trained by replay", BLUE, "--", "o", False),
]


def lr_table(units: list[dict]) -> tuple[dict, list[float]]:
    by: dict[tuple, dict[tuple, float]] = {}
    for u in units:
        if u["variant"] == R.ZERO_SHOT:
            continue
        key = (u["dataset"], u["backbone"], u["variant"], u["rank"], u["lr"])
        by.setdefault(key, {})[(u["target"], u["seed"])] = u["history"][-1]["macro_f1"]
    return by, sorted({k[4] for k in by})


def legend_below(fig, axes, columns: int, rows: int) -> None:
    """A shared legend under the panels, in a band reserved for it."""
    band = (0.19 * rows + 0.12) / fig.get_figheight()
    fig.tight_layout(w_pad=0.6, h_pad=0.8, rect=(0.0, band, 1.0, 1.0))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=columns, bbox_to_anchor=(0.5, 0.0), handlelength=3.2,
               columnspacing=1.6, borderaxespad=0.0)


def figure_lr(units: list[dict], series, rank: int, out: Path, name: str, window: float) -> None:
    """Macro-F1 against the learning rate; each panel shows the top ``window`` of its range.

    A diverged run falls to chance, far below every other point: the y-axis
    keeps the window under the best value of the panel, and a curve that
    diverges leaves the panel through its floor.
    """
    by, lrs = lr_table(units)
    if not all(by.get((d, b, s[0], rank, lr)) for d, b in SETTINGS for s in series for lr in lrs) or not lrs:
        print(f"{name} figure: runs incomplete, skipped")
        return
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 3.75), sharex=True)
    for ax, (dataset, backbone) in zip(axes.flat, SETTINGS):
        top = 0.0
        for variant, label, colour, line, marker, filled in series:
            cells = [by[(dataset, backbone, variant, rank, lr)] for lr in lrs]
            common = set.intersection(*[set(c) for c in cells])
            ys = [R.mean([c[t] for t in sorted(common)]) for c in cells]
            top = max(top, *ys)
            ax.plot(lrs, ys, color=colour, linestyle=line, marker=marker, label=label, markersize=4.5,
                    markerfacecolor=colour if filled else "white", markeredgecolor=colour if not filled else "white",
                    markeredgewidth=0.9)
        ax.set_xscale("log")
        ax.set_title(panel_title(dataset, backbone), color=INK)
        ceiling = top + 0.08 * window
        ax.set_ylim(ceiling - window, ceiling)
    for ax in axes[-1]:
        ax.set_xlabel("Learning rate")
    for ax in axes[:, 0]:
        ax.set_ylabel("Macro-F1")
    legend_below(fig, axes, 2, (len(series) + 1) // 2)
    save(fig, out / f"{name}.pdf")


STEP_SERIES = [
    ("gauss", "Gaussian", BLUE, "-", "o", True),
    ("cal", "Calibrated", ORANGE, "-", "s", True),
    ("pca-src", "Principal subspace", AQUA, "-", "^", True),
    ("sg", "Gaussian, trained by replay", BLUE, "--", "o", False),
    ("sg-cal", "Calibrated, trained by replay", ORANGE, "--", "s", False),
]


def figure_steps(units: list[dict], rank: int, out: Path, name: str) -> None:
    steps = sorted({h["step"] for u in units if u["variant"] != R.ZERO_SHOT for h in u["history"]})
    tables = {step: R.Table(units, step=step) for step in steps}
    if not steps or not has(tables[steps[-1]], [s[0] for s in STEP_SERIES], SETTINGS, [rank]):
        print(f"{name} figure: runs incomplete, skipped")
        return
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 3.95), sharex=True)
    for ax, (dataset, backbone) in zip(axes.flat, SETTINGS):
        for variant, label, colour, line, marker, filled in STEP_SERIES:
            ys = [tables[step].pooled(dataset, backbone, variant, rank)[0] for step in steps]
            ax.plot(steps, ys, color=colour, linestyle=line, marker=marker, label=label, markersize=4.5,
                    markerfacecolor=colour if filled else "white", markeredgecolor=colour if not filled else "white",
                    markeredgewidth=0.9)
        ax.set_xscale("log")
        ax.set_xticks(steps)
        ax.set_xticklabels([str(step) if step in (5, 10, 20, 50, 100, 250) else "" for step in steps])
        ax.minorticks_off()
        ax.set_title(panel_title(dataset, backbone), color=INK)
    for ax in axes[-1]:
        ax.set_xlabel("Adaptation steps")
    for ax in axes[:, 0]:
        ax.set_ylabel("Macro-F1")
    legend_below(fig, axes, 2, (len(STEP_SERIES) + 1) // 2)
    save(fig, out / f"{name}.pdf")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None, help="run directories (default: $CALPROJ_RUNS or out/runs)")
    ap.add_argument("--out", default=None, help="the paper directory (figures/ is created in it)")
    args = ap.parse_args()
    paper = Path(args.out) if args.out else R.paper_dir(ROOT)
    out = paper / "figures"
    runs = Path(args.runs) if args.runs else None
    style()
    figure_forest(R.Table(R.load_units(["main-"], runs), step=50), out)
    lr_units = R.load_units(["lr-"], runs)
    figure_lr(lr_units, LR_SERIES, 2, out, "lr_scale", 0.40)
    figure_lr(lr_units, LR_SERIES, 8, out, "lr_scale_rank8", 0.40)
    figure_lr(lr_units, LR_CAL_SERIES, 2, out, "lr_calibrated", 0.12)
    figure_lr(lr_units, LR_CAL_SERIES, 8, out, "lr_calibrated_rank8", 0.12)
    step_units = R.load_units(["steps-"], runs)
    figure_steps(step_units, 2, out, "steps")
    figure_steps(step_units, 8, out, "steps_rank8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
