#!/usr/bin/env python3
"""Read the published numbers this work compares with out of the source paper's LaTeX.

    python3 scripts/extract_published.py <arXiv source of 2610.08669>/sample-sigconf.tex

Writes results/published/memflora.json: the macro-F1 table of MemFLoRA
(arXiv:2610.08669v1, Table 1: mean and sample standard deviation at 50
adaptation steps), its initialisation ablation (Table 4), its memory table
(Table 3) and its cost table (Table 5). No number of another paper is typed by
hand anywhere in this repository: the tables of the manuscript read this file.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PM = re.compile(r"(\d+\.\d+)\s*\$\\pm\$\s*(\d+\.\d+)")


def table_body(tex: str, label: str) -> str:
    start = tex.index(r"\label{" + label + "}")
    begin = tex.index(r"\begin{tabular}", start)
    end = tex.index(r"\end{tabular}", begin)
    return tex[begin:end]


def rows(body: str) -> list[list[str]]:
    body = re.sub(r"(?m)^\s*%.*$", "", body)          # comment lines
    body = re.sub(r"(?<!\\)%.*", "", body)             # trailing comments
    out = []
    for raw in body.split(r"\\"):
        cells = [" ".join(c.split()) for c in raw.split("&")]
        if len(cells) > 1:
            out.append(cells)
    return out


def headline(tex: str) -> list[dict]:
    """Table 1: one record per dataset, backbone, method and rank (rank 0: rank-free methods)."""
    records, dataset, backbone = [], None, None
    for cells in rows(table_body(tex, "tab:headline_50_steps_std")):
        joined = " ".join(cells)
        for name in ("Opportunity", "RealDisp", "RealWorld"):
            if name in cells[0]:
                dataset = name.lower()
        for name, key in (("MobileNetV2", "mobilenet_v2"), ("T-ResNet", "t_resnet")):
            if name in cells[1]:
                backbone = key
        if dataset is None or backbone is None or len(cells) < 11 or not cells[6].strip().isdigit():
            continue
        rank = int(cells[6])
        for column, method in ((2, "zero-shot"), (3, "full-ft"), (4, "bias-tuning"), (5, "bn-tuning")):
            m = PM.search(cells[column])
            if m:
                records.append(dict(dataset=dataset, backbone=backbone, method=method, rank=0,
                                    mean=float(m.group(1)), std=float(m.group(2))))
        for column, method in ((7, "lora-c"), (8, "lora-edge"), (9, "memflora"), (10, "memflora-sg")):
            m = PM.search(cells[column])
            if not m:
                raise SystemExit(f"no value for {method} in row: {joined[:80]}")
            records.append(dict(dataset=dataset, backbone=backbone, method=method, rank=rank,
                                mean=float(m.group(1)), std=float(m.group(2))))
    return records


def ablation(tex: str) -> list[dict]:
    """Table 4: Opportunity, T-ResNet, 50 steps; factor, variant and macro-F1 at ranks 2, 4, 8."""
    records, factor = [], None
    for cells in rows(table_body(tex, "tab:ablation_design_choices")):
        if len(cells) < 5:
            continue
        head = re.sub(r"\\(midrule|toprule|bottomrule)", "", cells[0]).strip()
        if head:
            factor = head.rstrip(".")
        values = [re.sub(r"[^0-9.]", "", c) for c in cells[2:5]]
        if factor is None or "Factor" in cells[0] or "Variant" in cells[1] or not all(values):
            factor = None if "Factor" in cells[0] else factor
            continue
        variant = re.sub(r"\$|\\|\{|\}|times", " ", cells[1])
        records.append(dict(factor=factor, variant=" ".join(variant.split()),
                            **{f"r{r}": float(v) for r, v in zip((2, 4, 8), values)}))
    return records


def memory(tex: str) -> list[dict]:
    """Table 3: peak / saved-activation memory (MB) at rank 2 for batch sizes 1, 8, 32, 64, and optimiser state."""
    records, backbone = [], None
    for cells in rows(table_body(tex, "tab:memory_rank_batch")):
        joined = " ".join(cells)
        if "T-ResNet" in cells[0]:
            backbone = "t_resnet"
        if "MobileNetV2" in cells[0]:
            backbone = "mobilenet_v2"
        pairs = re.findall(r"(\d+\.\d+)\s*/\s*(\d+\.\d+)", joined)
        if backbone is None or len(pairs) != 4:
            continue
        method = re.sub(r"\\(midrule|cmidrule\(l\)\{2-7\})", "", cells[1]).strip()
        opt = re.sub(r"[^0-9.]", "", cells[-1])
        records.append(dict(backbone=backbone, method=method,
                            peak_mb={str(b): float(p) for b, (p, _) in zip((1, 8, 32, 64), pairs)},
                            saved_mb={str(b): float(s) for b, (_, s) in zip((1, 8, 32, 64), pairs)},
                            optimizer_mb=float(opt)))
    return records


def cost(tex: str) -> list[dict]:
    """Table 5: trainable parameters (thousands), MACs (millions) of AdaBN calibration plus 50 steps, at rank 2."""
    records, backbone = [], None
    for cells in rows(table_body(tex, "tab:training_perf_rank2_compact_adapt")):
        if "T-ResNet" in cells[0]:
            backbone = "t_resnet"
        if "MobileNetV2" in cells[0]:
            backbone = "mobilenet_v2"
        if backbone is None or len(cells) < 6:
            continue
        numbers = [re.sub(r"[^0-9.]", "", c) for c in cells[2:4]]
        if not all(numbers):
            continue
        method = re.sub(r"\\(midrule|toprule)", "", cells[1]).strip()
        records.append(dict(backbone=backbone, method=method, params_k=float(numbers[0]), macs_m=float(numbers[1])))
    return records


def main() -> int:
    source = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "sources/arXiv-2610.08669v1/sample-sigconf.tex")
    tex = source.read_text(encoding="utf-8")
    payload = {
        "source": "M. E. Akbulut, J. Geier, U. Schlichtmann, MemFLoRA: Memory-Floor LoRA for CNN Adaptation "
                  "at the Edge, arXiv:2610.08669v1",
        "note": "extracted by scripts/extract_published.py from the paper's LaTeX source; macro-F1 at 50 "
                "adaptation steps, mean and sample standard deviation over the pooled seed-by-target trials",
        "headline": headline(tex),
        "ablation": ablation(tex),
        "memory": memory(tex),
        "cost": cost(tex),
    }
    out = ROOT / "results" / "published" / "memflora.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"{out}: {len(payload['headline'])} headline, {len(payload['ablation'])} ablation, "
          f"{len(payload['memory'])} memory, {len(payload['cost'])} cost records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
