#!/usr/bin/env bash
# Adaptation budget: the evaluation after 5 to 250 steps of one run, at ranks 2
# and 8, on ten seeds. The seeds are split into jobs of about an hour:
#   SEED_CHUNKS     seed ranges of a job on Opportunity and RealDisp
#   SEED_CHUNKS_RW  seed ranges of a job on RealWorld (one job per body location)
#   SEED_SINGLES    seeds of RealDisp/MobileNetV2, one job each
source "$(dirname "$0")/common.sh"
: "${VARIANTS:=gauss cal pca-src select sg sg-cal}"
: "${STEPS:=5 10 20 50 100 150 200 250}"
: "${SEED_CHUNKS:=1-5 6-10}"
: "${SEED_CHUNKS_RW:=1-3 4-6 7-10}"
: "${SEED_SINGLES:=1 2 3 4 5 6 7 8 9 10}"

plan() {
  case "$1/$2" in
    opportunity/t_resnet)     TARGETS=("1 2 3 4"); CHUNKS="$SEED_CHUNKS" ;;
    opportunity/mobilenet_v2) TARGETS=(1 2 3 4);   CHUNKS="$SEED_CHUNKS" ;;
    realdisp/t_resnet)        TARGETS=(self);      CHUNKS="$SEED_CHUNKS" ;;
    realdisp/mobilenet_v2)    TARGETS=(self);      CHUNKS="$SEED_SINGLES" ;;
    realworld/*)              TARGETS=(chest forearm head shin thigh upperarm waist); CHUNKS="$SEED_CHUNKS_RW" ;;
  esac
}

for dataset in "${DATASETS[@]}"; do
  for backbone in "${BACKBONES[@]}"; do
    plan "$dataset" "$backbone"
    for targets in "${TARGETS[@]}"; do
      for chunk in $CHUNKS; do
        # shellcheck disable=SC2086
        submit "steps-$dataset-$backbone-${targets// /_}-s$chunk" 8 0-12:00:00 - \
          scripts/run_adapt.py --dataset "$dataset" --backbone "$backbone" --targets $targets \
          --seeds "$chunk" --ranks 2 8 --eval-steps $STEPS --variants $VARIANTS
      done
    done
  done
done
