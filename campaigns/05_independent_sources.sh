#!/usr/bin/env bash
# The main comparison with one source model per seed, instead of one per target
# reused for every seed. Needs the source models of
#   SOURCE_SEEDS="2 3 4 5" bash campaigns/01_sources.sh     (2 and 3 on RealWorld)
# besides those of seed 1. Five seeds (three on RealWorld).
source "$(dirname "$0")/common.sh"
: "${VARIANTS:=gauss orth-norm pca-src select cal cal-src sg sg-cal}"

# One job per setting and target; RealDisp with MobileNetV2, whose trials are
# the longest, gets one job per seed.
plan() {
  case "$1/$2" in
    opportunity/*)         TARGETS=("1 2 3 4"); CHUNKS="1-5" ;;
    realdisp/mobilenet_v2) TARGETS=(self);      CHUNKS="1 2 3 4 5" ;;
    realdisp/*)            TARGETS=(self);      CHUNKS="1-5" ;;
    realworld/*)           TARGETS=(chest forearm head shin thigh upperarm waist); CHUNKS="1-3" ;;
  esac
}

for dataset in "${DATASETS[@]}"; do
  for backbone in "${BACKBONES[@]}"; do
    plan "$dataset" "$backbone"
    for targets in "${TARGETS[@]}"; do
      for chunk in $CHUNKS; do
        name="indep-$dataset-$backbone-${targets// /_}"
        [[ "$CHUNKS" == *" "* ]] && name="$name-s$chunk"
        # shellcheck disable=SC2086
        submit "$name" 8 0-12:00:00 "${DEP:--}" \
          scripts/run_adapt.py --dataset "$dataset" --backbone "$backbone" --targets $targets \
          --seeds "$chunk" --source-seed seed --ranks 2 4 8 --eval-steps 50 --variants $VARIANTS
      done
    done
  done
done
