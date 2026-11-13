#!/usr/bin/env bash
# Learning-rate sweep at 50 steps: the two-by-two design of projection family
# and row norm, the mean-free projection at the Gaussian scale, the calibrated
# projection and projection training, at ranks 2
# and 8. Five seeds (three on RealWorld, which has seven targets).
source "$(dirname "$0")/common.sh"
: "${VARIANTS:=gauss gauss-unit orth orth-norm gauss-nomean-keep cal sg}"
: "${LRS:=3e-4 1e-3 2e-3 3e-3 5e-3 1e-2 3e-2}"

plan() {
  case "$1/$2" in
    opportunity/t_resnet)     TARGETS=("1 2 3 4"); CHUNKS="1-5" ;;
    opportunity/mobilenet_v2) TARGETS=(1 2 3 4);   CHUNKS="1-5" ;;
    realdisp/t_resnet)        TARGETS=(self);      CHUNKS="1-5" ;;
    realdisp/mobilenet_v2)    TARGETS=(self);      CHUNKS="1 2 3 4 5" ;;
    realworld/*)              TARGETS=(chest forearm head shin thigh upperarm waist); CHUNKS="1-3" ;;
  esac
}

for dataset in "${DATASETS[@]}"; do
  for backbone in "${BACKBONES[@]}"; do
    plan "$dataset" "$backbone"
    for targets in "${TARGETS[@]}"; do
      for chunk in $CHUNKS; do
        # shellcheck disable=SC2086
        submit "lr-$dataset-$backbone-${targets// /_}-s$chunk" 8 0-12:00:00 - \
          scripts/run_adapt.py --dataset "$dataset" --backbone "$backbone" --targets $targets \
          --seeds "$chunk" --ranks 2 8 --eval-steps 50 --lrs $LRS --variants $VARIANTS
      done
    done
  done
done
