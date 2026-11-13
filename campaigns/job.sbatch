#!/usr/bin/env bash
#SBATCH --output=logs/%x-%j.out
# One run: a script of scripts/ with its arguments, on the CPUs of the job.
# The resources (CPUs, memory, time) are set on the sbatch line by
# campaigns/common.sh.
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
echo "started: $(date -Is)"
echo "command: ${PYTHON:-python} $*"
"${PYTHON:-python}" "$@"
echo "finished: $(date -Is)"
