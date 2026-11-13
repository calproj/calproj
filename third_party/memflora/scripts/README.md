# Experiment scripts

One script per paper experiment, plus a smoke test. Run them from the repository
root; all share the settings in `common.sh`.

```bash
bash scripts/00_smoke_test.sh                            # quick pipeline check
bash scripts/01_main_accuracy_opportunity_t_resnet.sh    # one experiment
bash scripts/run_all.sh main                             # a group
```

| Variable | Default | Meaning |
|---|---|---|
| `PYTHON` | `python3` | interpreter to use |
| `RESULTS_ROOT` | `runs` | where output is written |

| # | Scripts | Paper |
|---|---|---|
| 01–06 | `*_main_accuracy_*` | Table 1 |
| 07–08 | `*_convergence_*` | Figure 3 |
| 09–10 | `*_memory_profiling_opportunity_*` | Table 5 (cost) |
| 11–12 | `*_memory_profiling_batch_sweep_*` | Table 3, Figure 2 |
| 14–17 | `*_ablations_*` | Table 4 |
| 20–22 | `*_tinytl_comparison_*` | TinyTL comparison |
| 23–24 | `*_time_spec_*` | Table 5 (steps to 85%) |

Groups for `run_all.sh`: `main`, `convergence`, `memory`, `ablations`, `tinytl`,
`time_spec`, `all`. Scripts 01–08 are the long ones and need a GPU.
