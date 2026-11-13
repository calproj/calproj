# MemFLoRA

Memory-Floor LoRA: low-rank adapters for fine-tuning CNNs on edge devices, where
the limit is activation memory, not the number of trainable parameters.

MemFLoRA keeps the state saved for backward at rank `r` by

- freezing the down-projection and training only the up-projection,
- running the backbone BatchNorm in eval mode (a fixed affine map),
- storing the ReLU mask as packed bits, and
- scaling the adapter output by the frozen BatchNorm scale.

MemFLoRA-SG additionally updates the projection by replaying the forward pass.

| Paper name | Method name in the code |
|---|---|
| MemFLoRA | `bnpa_fa_postbn_bnr_off_scaled` |
| MemFLoRA-SG | `bnpa_q3_sg` |

Baselines: `zero_shot`, `full`, `bias_tuning`, `bn_tuning`, `lora_c`,
`lora_edge_optimized`, `tinytl_lite_residual_bias*` (all in `src/methods.py`).

## Setup

Python 3.11 or newer:

```bash
pip install -r requirements.txt     # or: conda env create -f environment.yml
```

The datasets are not included. Put them under `data/` (or pass `--data-root`):

```text
data/opportunity/OpportunityUCIDataset/dataset/S1-ADL1.dat ...
data/realdisp/subject1_ideal.log, subject1_self.log ...
data/realworld/realworld2016_dataset/proband1/data/acc_walking_csv.zip ...
```

## Reproducing the paper

Every experiment has a ready script in `scripts/` with all arguments set.
Run them from the repository root:

```bash
bash scripts/00_smoke_test.sh                              # quick pipeline check
bash scripts/01_main_accuracy_opportunity_t_resnet.sh      # one experiment
bash scripts/run_all.sh main                               # a whole group
```

Groups: `main`, `convergence`, `memory`, `ablations`, `tinytl`, `time_spec`, `all`.
Set `PYTHON=/path/to/python` to choose the interpreter and `RESULTS_ROOT` to
change the output folder (default `runs/`). Each trial adds a row to
`minimal_benchmark_results.csv`; `minimal_benchmark_summary.csv` is written at the end.

| Paper | Scripts |
|---|---|
| Table 1 (accuracy) | `01`–`06` |
| Figure 3 (adaptation steps) | `07`, `08` |
| Table 3, Figure 2 (memory vs. batch size) | `11`, `12` |
| Table 4 (ablations) | `14`–`17` |
| Table 5 (cost, steps to 85%) | `09`, `10`, `23`, `24` |
| TinyTL comparison | `20`–`22` |
| Tables 6, 7 (Jetson) | `tools/jetson_profile.py` |
| Figure 4 (Jetson memory trace) | `tools/jetson_memory.py` |

The memory scripts add `--profile-full-sram true` (and `09`, `10` also
`--profile-performance true`), which write `full_sram_summary.csv` and
`performance_summary.csv` next to the results.

The scripts use `--reuse-first-source-model true` for faster experiments: the
source model is pretrained once per target and shared across seeds. Pretraining
a new source model for every seed (`false`) gives overall the same results.

## Running a single experiment

Everything goes through `experiments/minimal_methods_benchmark.py`. For example,
MemFLoRA at rank 4 on Opportunity with T-ResNet, every subject held out in turn:

```bash
python experiments/minimal_methods_benchmark.py \
  --dataset opportunity --backbone t_resnet_official \
  --method bnpa_fa_postbn_bnr_off_scaled \
  --rank 4 --target-subjects all --seed 1 2 3 4 5 \
  --window-size 60 --window-stride 30 \
  --label-column ml_both_arms --window-label-rule majority \
  --pretrain-epochs 20 --steps-adapt 50 --batch-size 64 \
  --reuse-first-source-model true --adapt-eval-every-steps 50 \
  --adapter-layers all --adabn-calibration-mode ema_no_reset \
  --adabn-calib-batches 1 --proj-init random --bnpa-bottleneck-bn on \
  --adabn-stat-source target --train_mode_adaBN off --adapt-lr 1e-3 \
  --results-root runs
```

Arguments that take several values are swept, e.g. `--method zero_shot full
bnpa_fa_postbn_bnr_off_scaled --rank 2 4 8`. Use `--adapter-layers pointwise_only`
for MobileNetV2.

## Jetson Orin Nano

```bash
python tools/jetson_profile.py                         # Tables 6 and 7 (rank 2, batches 1/8/32/64)
python tools/jetson_memory.py --model mobilenetv2 --rank 2 --batch 64 \
  --bitpack-backend cuda                               # Figure 4
```

`jetson_memory.py` serves its report at `http://127.0.0.1:8000`; Ctrl+C saves
`report.html` under `runs/jetson_memory_mobilenetv2_r2/`.

The optional CUDA kernel packs eight ReLU gates per byte. It needs a CUDA-enabled
PyTorch, `nvcc`, a C++ compiler and `ninja`, and builds on first use.
`--bitpack-backend cuda` requires it, `auto` (default) falls back to torch if the
build fails, and `torch` never compiles. For the benchmark runner, use
`MEMFLORA_BITPACK_BACKEND=cuda|auto|torch`.

## Code layout

```text
src/adapters/   MemFLoRA blocks with hand-written backward passes; LoRA-C, LoRA-Edge, TinyTL
src/models/     T-ResNet, MobileNetV2 and adapter injection
src/data/       dataset loaders and splits
src/profiling/  saved-tensor memory accounting and MAC counting
experiments/    benchmark runner
scripts/        one script per paper experiment
tools/          Jetson profilers
```

The method itself is in `src/adapters/bnpa_conv_bn_act_2d.py`.
