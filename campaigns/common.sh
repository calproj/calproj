#!/usr/bin/env bash
# Shared by the campaign scripts. Run them from the repository root:
#
#   HAR_DATA_ROOT=/path/to/data bash campaigns/<script>.sh
#
#   HAR_DATA_ROOT  holds opportunity/, realdisp/, realworld/ and cache/ (required)
#   RUNNER         "slurm" (default) submits each run with sbatch and
#                  campaigns/job.sbatch; "local" runs them one after another
#   TAG            optional suffix of every run name, to tell machines apart
#   MEM_PER_CPU    MB of memory per CPU of a Slurm job (default 1900)
#   SBATCH_EXTRA   extra sbatch flags, e.g. "-p <partition> -A <account>"
#
# Runs are written to out/runs/<name>. A run that is started again skips the
# units it has already finished. To build the tables and figures from a rerun
# instead of the shipped outputs: CALPROJ_RUNS=out/runs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${HAR_DATA_ROOT:?set HAR_DATA_ROOT to the directory that holds the datasets}"
: "${RUNNER:=slurm}"
: "${TAG:=}"
: "${MEM_PER_CPU:=1900}"
: "${SBATCH_EXTRA:=}"
: "${PYTHON:=python}"

DATASETS=(opportunity realdisp realworld)
BACKBONES=(t_resnet mobilenet_v2)
# ONLY_DATASETS / ONLY_BACKBONES restrict a campaign, e.g. ONLY_DATASETS="realworld"
# shellcheck disable=SC2206
[[ -n "${ONLY_DATASETS:-}" ]] && DATASETS=($ONLY_DATASETS)
# shellcheck disable=SC2206
[[ -n "${ONLY_BACKBONES:-}" ]] && BACKBONES=($ONLY_BACKBONES)

# submit NAME CPUS TIME DEPENDENCY SCRIPT [ARGS...]
#   DEPENDENCY is "-" or Slurm job ids joined by ":" that must end well first.
#   Prints the job id (Slurm) or the run name (local).
submit() {
  local name="$1${TAG:+-$TAG}" cpus="$2" time="$3" dep="$4" script="$5"
  shift 5
  if [[ "$RUNNER" == local ]]; then
    HAR_DATA_ROOT="$HAR_DATA_ROOT" OMP_NUM_THREADS="$cpus" "$PYTHON" "$script" "$@" --run-dir="out/runs/$name" >&2
    echo "$name"
    return
  fi
  local opts=(--parsable --job-name="$name" -c "$cpus" --mem="$((MEM_PER_CPU * cpus))M"
              --time="$time" --export="ALL,HAR_DATA_ROOT=$HAR_DATA_ROOT,PYTHON=$PYTHON")
  [[ "$dep" != - ]] && opts+=(--dependency="afterok:$dep")
  mkdir -p logs
  # shellcheck disable=SC2086
  sbatch "${opts[@]}" $SBATCH_EXTRA campaigns/job.sbatch "$script" "$@" --run-dir="out/runs/$name"
}
