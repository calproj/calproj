#!/usr/bin/env bash
# Training-state memory and operation counts of one step, with the MemFLoRA
# release's profilers, on Opportunity-shaped inputs as in the source paper.
source "$(dirname "$0")/common.sh"
: "${VARIANTS:=gauss cal signs select pca-src sg}"

for backbone in "${BACKBONES[@]}"; do
  # shellcheck disable=SC2086
  submit "cost-opportunity-$backbone" 8 0-02:00:00 - \
    scripts/profile_cost.py --dataset opportunity --backbone "$backbone" \
    --variants $VARIANTS --ranks 2 4 8 --batches 1 8 32 64
done
