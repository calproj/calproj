# calproj: calibrated projections for memory-floor adapters

Memory-floor adapters adapt a convolutional network on the device it runs on
without saving full-width activations: each convolution receives a low-rank
branch `U * (P x)` whose down-projection `P` is frozen, so the backward pass
needs only a bottleneck of `r` channels
([MemFLoRA](https://arxiv.org/abs/2610.08669)). This repository studies that
frozen projection, the one part of the adapter that adaptation never changes,
by separating three of its properties:

- **its scale**, which is exactly a step size of the optimiser for the
  trainable factor (a Gaussian projection is an orthogonal one trained with a
  step `sqrt(C/r)` times larger);
- **its subspace**, which does not matter: principal subspaces of the
  activations and gradient sketches adapt no better than a random subspace,
  and random signs or a random selection of input channels do as well as a
  Gaussian matrix;
- **its mean**, which does: after a rectifier a large share of the energy of a
  random bottleneck is a constant. The **calibrated projection** centres each
  bottleneck channel and gives it the energy `tr(Sigma)/r`, from `2r + 1`
  statistics per layer, at unchanged memory and operations.

It holds the code of the experiments, the outputs of every run, and the
scripts that rebuild every table and figure from them.

## Installation

Python 3.11 to 3.13.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests -q          # 83 tests; add -m "not slow" for the quick ones
```

The pinned versions (`torch==2.6.0`, `numpy==2.4.6`, `scikit-learn==1.8.0`) are
those the shipped results were produced with. Everything ran on CPUs.

## A short example

`examples/calibrated_projection.py` builds the calibrated projection on
synthetic layer inputs that follow a rectifier:

```bash
python examples/calibrated_projection.py
```

```
mean share of the input energy: s = 0.47
Gaussian   constant share per channel: 0.52 0.34 0.02 0.17   energy / (tr(Sigma)/r): 0.86 0.95 0.55 0.64
calibrated constant share per channel: 0.00 0.00 0.00 0.00   energy / (tr(Sigma)/r): 1.00 1.00 1.00 1.00
```

The Gaussian bottleneck spends an uneven part of its energy on a constant; the
calibrated one has none, and every channel carries `tr(Sigma)/r`. In an
adapter, `src/calproj/adapt.py` (`apply_variant`) installs the projection and
its shift in the fused operator of the MemFLoRA release, and
`merged` folds an adapted layer back into one convolution.

## What is here

| path | what |
|---|---|
| `src/calproj/` | the projections (`projection.py`, `variants.py`), their installation in the adapter and the paired trial (`adapt.py`), the protocol (`protocol.py`), the aggregation of results (`results.py`) |
| `tests/` | unit tests, among them the exact equivalence between the scale of the projection and the step size of Adam |
| `scripts/` | data preparation, source pretraining, adaptation, cost profiling, and the table and figure builders |
| `campaigns/` | the lists of runs of each experiment, for Slurm or for one machine |
| `third_party/memflora/` | the MemFLoRA release this code runs (see below) |
| `results/runs/` | the outputs of every run: `config.json`, `env.json`, `metrics.json.gz` |
| `results/published/` | the published numbers the tables quote, extracted from the MemFLoRA paper's source |
| `results/tables/` | the aggregates as CSV |
| `paper/tables/`, `paper/figures/` | the tables (LaTeX) and figures (PDF) as generated |

## Rebuilding the tables and figures

```bash
python scripts/make_tables.py      # paper/tables/*.tex, results/tables/*.csv
python scripts/make_figures.py     # paper/figures/*.pdf
```

Both read the shipped run outputs in `results/runs/` and reproduce the shipped
files exactly. Which file holds what:

| file | content |
|---|---|
| `tables/scale` | Gaussian and orthogonal projections at the two row norms |
| `tables/subspace` | random and data-aware subspaces: captured gradient share, explained input energy, paired differences |
| `tables/sites` | share of the input and gradient energy along the mean input, per group of layers |
| `tables/calibration` | the calibrated projection, its ingredients and its combinations |
| `tables/lr` | each projection at the default and at its best learning rate |
| `tables/budget` | paired differences after 5 to 250 adaptation steps |
| `tables/main`, `tables/rankfree` | the projections beside the published results of other methods |
| `tables/cost` | memory and operations of an adaptation step |
| `tables/sources` | the comparison with one source model per seed |
| `tables/numbers` | every number quoted in the text, as LaTeX macros |
| `figures/forest` | paired difference of every projection to the Gaussian one |
| `figures/lr_scale*`, `figures/lr_calibrated*` | macro-F1 against the learning rate |
| `figures/steps*` | macro-F1 against the number of adaptation steps |

## Rerunning the experiments

**Data.** Three public datasets for human activity recognition, to be placed
under one directory (`HAR_DATA_ROOT`) as the MemFLoRA release expects:

| dataset | source | license |
|---|---|---|
| Opportunity | [UCI repository, dataset 226](https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition), doi:10.24432/C5M027 | CC BY 4.0 |
| RealDisp | [UCI repository, dataset 305](https://archive.ics.uci.edu/dataset/305/realdisp+activity+recognition+dataset), doi:10.24432/C5GP6D | CC BY 4.0 |
| RealWorld (2016) | [University of Mannheim](https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/) | none stated on its page; cite Sztyler and Stuckenschmidt, PerCom 2016 |

```
$HAR_DATA_ROOT/opportunity/OpportunityUCIDataset/dataset/S1-ADL1.dat ...
$HAR_DATA_ROOT/realdisp/subject1_ideal.log, subject1_self.log ...
$HAR_DATA_ROOT/realworld/realworld2016_dataset/proband1/data/acc_walking_csv.zip ...
```

**Campaigns.** Each script of `campaigns/` lists the runs of one experiment
and submits them with `sbatch` on any Slurm cluster, or runs them one after
another with `RUNNER=local`:

```bash
export HAR_DATA_ROOT=/path/to/data
bash campaigns/00_data.sh                  # window each dataset once, cache the arrays
bash campaigns/01_sources.sh               # one source model per target domain
bash campaigns/02_main.sh                  # 24 projections, 3 ranks, 20 seeds
bash campaigns/03_learning_rate.sh         # 7 learning rates
bash campaigns/04_steps.sh                 # 5 to 250 adaptation steps
SOURCE_SEEDS="2 3 4 5" bash campaigns/01_sources.sh
bash campaigns/05_independent_sources.sh   # one source model per seed
bash campaigns/06_cost.sh                  # memory and operation profiles
CALPROJ_RUNS=out/runs python scripts/make_tables.py
```

A run writes `out/runs/<name>/` and, started again, skips the units it has
finished. The tables are built from the shipped outputs unless `CALPROJ_RUNS`
points elsewhere, so a rerun never mixes with them. One trial is seconds to a
minute on 8 CPU cores; in all, the shipped runs took about 2,400 core-hours on
96-core AMD EPYC 9655 nodes (source models 550, main campaign 1,070, learning
rates 240, budgets 470, independent source models 70), 8 cores per job.

A single run, by hand:

```bash
python scripts/run_adapt.py --dataset opportunity --backbone t_resnet --targets 1 \
    --seeds 1-2 --ranks 2 --eval-steps 50 --variants gauss cal sg --run-dir out/runs/try
```

`src/calproj/variants.py` lists the projections by name (`gauss`: the Gaussian
projection of MemFLoRA; `cal`: the calibrated projection; `sg`: the projection
trained by replay; ...).

## Third-party code

`third_party/memflora/` is the release of *MemFLoRA: Memory-Floor LoRA for CNN
Adaptation at the Edge* (M. E. Akbulut, J. Geier, U. Schlichtmann),
<https://github.com/mehmetemreakbulut/MemFLoRA>, at commit
`1efcf1ff0c55a7ddcda07e3b1b684452e1bcbd7d`, unmodified, under its MIT license
(`third_party/memflora/LICENSE`). The adapters, the training loop and the
memory and operation profilers are its own; this repository adds the
construction of the projections and the paired evaluation.
`results/published/memflora.json` holds the numbers of that paper's tables,
extracted from its LaTeX source by `scripts/extract_published.py`.

## License

MIT (`LICENSE`).
