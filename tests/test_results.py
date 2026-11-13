"""Aggregation of units into the tables' numbers."""

from __future__ import annotations

import json
import math

from calproj import results


def unit(target, seed, variant, rank, f1, dataset="opportunity", backbone="t_resnet", lr=1e-3, steps=(50,)):
    history = [{"step": s, "macro_f1": f1 + 0.001 * i} for i, s in enumerate(steps)]
    return {"dataset": dataset, "backbone": backbone, "target": str(target), "seed": seed, "source_seed": 1,
            "variant": variant, "rank": rank, "lr": lr, "history": history,
            "capture": {"total": 0.1, "functional": 0.5, "explained": 0.3}, "trainable_params": 10}


def test_pooled_and_paired(tmp_path):
    units = []
    for seed in (1, 2, 3):
        for target in (1, 2):
            units.append(unit(target, seed, "gauss", 2, 0.70 + 0.01 * seed + 0.1 * target))
            units.append(unit(target, seed, "x", 2, 0.72 + 0.01 * seed + 0.1 * target + 0.01 * (seed == 3)))
    run = tmp_path / "main-a"
    run.mkdir()
    (run / "metrics.json").write_text(json.dumps({"units": units}))
    (tmp_path / "other-b").mkdir()
    loaded = results.load_units(["main-"], tmp_path)
    assert len(loaded) == 12 and all(u["run"] == "main-a" for u in loaded)
    table = results.Table(loaded)
    m, s, n = table.pooled("opportunity", "t_resnet", "gauss", 2)
    values = [0.70 + 0.01 * seed + 0.1 * target for seed in (1, 2, 3) for target in (1, 2)]
    assert n == 6 and math.isclose(m, sum(values) / 6) and math.isclose(s, results.sample_std(values))
    diffs = table.seed_differences("opportunity", "t_resnet", "x", "gauss", 2)
    assert len(diffs) == 3 and all(math.isclose(d, w, abs_tol=1e-12) for d, w in zip(diffs, (0.02, 0.02, 0.03)))
    mean, half, seeds = table.difference("opportunity", "t_resnet", "x", "gauss", 2)
    assert seeds == 3 and math.isclose(mean, 0.07 / 3)
    assert math.isclose(half, results.t975(2) * results.sample_std(diffs) / math.sqrt(3))
    overall = table.overall_difference("x", "gauss", [2])
    assert math.isclose(overall[0], 0.07 / 3) and overall[2:] == (3, 1)
    assert math.isclose(table.share("opportunity", "t_resnet", "x", 2, "functional"), 0.5)


def test_steps_and_duplicates(tmp_path):
    a = unit(1, 1, "gauss", 2, 0.5, steps=(50, 100))
    for name in ("main-a", "main-b"):  # the same unit pulled from two runs counts once
        (tmp_path / name).mkdir()
        (tmp_path / name / "metrics.json").write_text(json.dumps({"units": [a]}))
    loaded = results.load_units(["main-"], tmp_path)
    assert len(loaded) == 1
    assert results.score(loaded[0], 100) == 0.501 and results.score(loaded[0]) == 0.501
    assert results.score(loaded[0], 75) is None
    assert results.Table(loaded, step=50).pooled("opportunity", "t_resnet", "gauss", 2)[0] == 0.5


def test_student_quantiles():
    assert results.t975(19) == 2.093 and results.t975(1000) == 1.960
    assert results.t975(25) == results.t975(24)  # conservative: the next tabulated value below


def test_compressed_summary_reads_like_the_plain_one(tmp_path):
    import gzip

    unit = {"dataset": "opportunity", "backbone": "t_resnet", "target": "1", "seed": 1, "variant": "gauss",
            "rank": 2, "history": [{"step": 50, "macro_f1": 0.5}]}
    for name, packed in (("main-plain", False), ("main-packed", True)):
        run = tmp_path / name
        run.mkdir()
        text = json.dumps({"units": [{**unit, "seed": 2 if packed else 1}]})
        if packed:
            with gzip.open(run / "metrics.json.gz", "wt", encoding="utf-8") as handle:
                handle.write(text)
        else:
            (run / "metrics.json").write_text(text)
    loaded = results.load_units(["main-"], tmp_path)
    assert sorted(u["seed"] for u in loaded) == [1, 2]
    assert results.read_metrics(tmp_path / "missing") is None


def test_runs_root_follows_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CALPROJ_RUNS", str(tmp_path))
    assert results.runs_root() == tmp_path
