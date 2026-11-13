"""Read run directories and aggregate their units into the numbers the tables and figures print.

A *unit* is one adaptation trial (``scripts/run_adapt.py``). A *trial* here is
a (target, seed) pair of one setting; the units of a trial are paired, so a
difference between two variants is taken trial by trial. Aggregates follow
the source paper where they stand beside its numbers (mean and sample standard
deviation over the pooled trials) and use seeds as replicates for intervals:
the difference is first averaged over the targets of each seed.
"""

from __future__ import annotations

import gzip
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ZERO_SHOT = "zero-shot"
DATASETS = ("opportunity", "realdisp", "realworld")
BACKBONES = ("t_resnet", "mobilenet_v2")
DATASET_LABEL = {"opportunity": "Opportunity", "realdisp": "RealDisp", "realworld": "RealWorld"}
BACKBONE_LABEL = {"t_resnet": "T-ResNet", "mobilenet_v2": "MobileNetV2"}

# two-sided 95 % Student quantiles, by degrees of freedom (the standard library has none)
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
         10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
         18: 2.101, 19: 2.093, 20: 2.086, 24: 2.064, 29: 2.045, 30: 2.042, 39: 2.023, 40: 2.021,
         59: 2.001, 60: 2.000, 120: 1.980}


def t975(dof: int) -> float:
    if dof < 1:
        return float("nan")
    if dof in _T975:
        return _T975[dof]
    below = max(k for k in _T975 if k <= dof)
    return _T975[below] if dof < 120 else 1.960


def runs_root() -> Path:
    """Where the run directories are.

    ``$CALPROJ_RUNS`` when set; else the run outputs shipped with the
    repository, ``results/runs``, when they are there; else ``out/runs``, where
    the campaigns write. A rerun therefore never mixes with the shipped
    outputs: point ``CALPROJ_RUNS`` at ``out/runs`` to build from it.
    """
    env = os.environ.get("CALPROJ_RUNS")
    if env:
        return Path(env)
    root = Path(__file__).resolve().parents[2]
    shipped = root / "results" / "runs"
    return shipped if shipped.is_dir() else root / "out" / "runs"


def paper_dir(root: Path) -> Path:
    """Where the generated tables and figures go: beside a manuscript the tree holds, else ``paper/``."""
    manuscripts = sorted(Path(root).glob("*/*/manuscript.tex"))
    return manuscripts[0].parent if manuscripts else Path(root) / "paper"


def read_metrics(run: Path) -> dict | None:
    """The summary of a run directory: ``metrics.json``, or its gzip-compressed copy."""
    plain, packed = Path(run) / "metrics.json", Path(run) / "metrics.json.gz"
    if plain.is_file():
        return json.loads(plain.read_text(encoding="utf-8"))
    if packed.is_file():
        with gzip.open(packed, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return None


def load_units(prefixes: Iterable[str], root: Path | None = None) -> list[dict]:
    """Units of every run directory whose name starts with one of ``prefixes``, without duplicates."""
    root = Path(root) if root is not None else runs_root()
    prefixes = tuple(prefixes)
    units, seen = [], set()
    for run in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefixes)):
        metrics = read_metrics(run)
        if metrics is None:
            continue
        for unit in metrics.get("units", []):
            if "variant" not in unit:
                continue
            key = (unit["dataset"], unit["backbone"], unit["target"], unit["seed"], unit.get("source_seed"),
                   unit["variant"], unit["rank"], unit.get("lr", 0.0), unit.get("beta2", 0.999))
            if key not in seen:
                seen.add(key)
                units.append({**unit, "run": run.name})
    return units


def score(unit: dict, step: int | None = None, metric: str = "macro_f1") -> float | None:
    """The metric of a unit at an evaluation step (default: its last one)."""
    history = unit["history"]
    if step is None:
        return history[-1][metric]
    for point in history:
        if point["step"] == step:
            return point[metric]
    return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def interval(values: list[float]) -> tuple[float, float]:
    """Mean and half-width of the two-sided 95 % Student interval of the mean."""
    n = len(values)
    if n < 2:
        return mean(values), float("nan")
    return mean(values), t975(n - 1) * sample_std(values) / math.sqrt(n)


class Table:
    """Scores indexed by setting, variant, rank and trial."""

    def __init__(self, units: list[dict], step: int | None = None, lr: float | None = None,
                 metric: str = "macro_f1") -> None:
        self.cells: dict[tuple, dict[tuple, float]] = defaultdict(dict)
        self.capture: dict[tuple, dict[tuple, dict]] = defaultdict(dict)
        self.params: dict[tuple, int] = {}
        for unit in units:
            if unit["variant"] != ZERO_SHOT and lr is not None and abs(unit.get("lr", 0.0) - lr) > 1e-12:
                continue
            value = score(unit, None if unit["variant"] == ZERO_SHOT else step, metric)
            if value is None:
                continue
            key = (unit["dataset"], unit["backbone"], unit["variant"], unit["rank"])
            trial = (unit["target"], unit["seed"])
            self.cells[key][trial] = value
            if unit.get("capture"):
                self.capture[key][trial] = unit["capture"]
            if "trainable_params" in unit:
                self.params[key] = unit["trainable_params"]

    def trials(self, dataset: str, backbone: str, variant: str, rank: int) -> dict[tuple, float]:
        return self.cells.get((dataset, backbone, variant, rank), {})

    def variants(self) -> list[str]:
        return sorted({k[2] for k in self.cells})

    def settings(self) -> list[tuple[str, str]]:
        present = {(k[0], k[1]) for k in self.cells}
        return [(d, b) for d in DATASETS for b in BACKBONES if (d, b) in present]

    def pooled(self, dataset: str, backbone: str, variant: str, rank: int) -> tuple[float, float, int]:
        """Mean, sample standard deviation and count over the pooled trials: the source paper's summary."""
        values = list(self.trials(dataset, backbone, variant, rank).values())
        return mean(values), sample_std(values), len(values)

    def seed_differences(self, dataset: str, backbone: str, variant: str, reference: str, rank: int,
                         reference_rank: int | None = None) -> list[float]:
        """Per seed, the difference to ``reference`` averaged over the targets both variants ran."""
        mine = self.trials(dataset, backbone, variant, rank)
        ref = self.trials(dataset, backbone, reference, rank if reference_rank is None else reference_rank)
        by_seed: dict[int, list[float]] = defaultdict(list)
        for trial in sorted(set(mine) & set(ref)):
            by_seed[trial[1]].append(mine[trial] - ref[trial])
        return [mean(v) for _seed, v in sorted(by_seed.items())]

    def difference(self, dataset: str, backbone: str, variant: str, reference: str, rank: int) -> tuple[float, float, int]:
        """Mean paired difference to ``reference``, 95 % half-width over seeds, number of seeds."""
        values = self.seed_differences(dataset, backbone, variant, reference, rank)
        m, half = interval(values)
        return m, half, len(values)

    def overall_difference(self, variant: str, reference: str, ranks: Iterable[int],
                           settings: Iterable[tuple[str, str]] | None = None) -> tuple[float, float, int, int]:
        """The paired difference averaged over settings and ranks, with seeds as replicates.

        Returns mean, 95 % half-width, number of seeds and number of
        configurations (setting-rank pairs) that entered every seed's average.
        """
        per_seed: dict[int, list[float]] = defaultdict(list)
        configs = 0
        for dataset, backbone in (settings or self.settings()):
            for rank in ranks:
                mine = self.trials(dataset, backbone, variant, rank)
                ref = self.trials(dataset, backbone, reference, rank)
                common = sorted(set(mine) & set(ref))
                if not common:
                    continue
                configs += 1
                by_seed: dict[int, list[float]] = defaultdict(list)
                for trial in common:
                    by_seed[trial[1]].append(mine[trial] - ref[trial])
                for seed, values in by_seed.items():
                    per_seed[seed].append(mean(values))
        complete = [mean(v) for _seed, v in sorted(per_seed.items()) if len(v) == configs]
        m, half = interval(complete)
        return m, half, len(complete), configs

    def share(self, dataset: str, backbone: str, variant: str, rank: int, which: str = "total") -> float:
        """Mean captured share (``total``: weight space, ``functional``, ``explained``) over the trials."""
        values = [c[which] for c in self.capture.get((dataset, backbone, variant, rank), {}).values()
                  if c.get(which) is not None and c[which] == c[which]]
        return mean(values)
