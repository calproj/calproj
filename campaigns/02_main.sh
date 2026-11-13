#!/usr/bin/env bash
# The main campaign: every projection, on the six settings, at 50 adaptation
# steps and ranks 2, 4 and 8, seeds 1 to 20, with the source models of
# 01_sources.sh. A job covers one dataset, backbone, group of targets and
# range of seeds, sized to a few hours on 8 cores (46 jobs in all, under the
# 384-core cap of the cluster they ran on); a job that is resubmitted skips
# the units it has finished.
#   VARIANTS  default: the list below
source "$(dirname "$0")/common.sh"
: "${VARIANTS:=gauss gauss-unit orth orth-norm signs select-norm select select-top-src pca-src pca-tgt grad pca-tgt-nomean gauss-balanced gauss-nomean-keep gauss-nomean cal cal-src signs-cal select-cal gauss-bias cal-bias sg sg-nomean sg-cal}"

# setting -> "targets per job" and "seed ranges"
plan() {
  case "$1/$2" in
    opportunity/t_resnet)     TARGETS=("1 2 3 4"); CHUNKS="1-20" ;;
    opportunity/mobilenet_v2) TARGETS=(1 2 3 4);   CHUNKS="1-10 11-20" ;;
    realdisp/t_resnet)        TARGETS=(self);      CHUNKS="1-10 11-20" ;;
    realdisp/mobilenet_v2)    TARGETS=(self);      CHUNKS="1-3 4-6 7-9 10-12 13-15 16-18 19-20" ;;
    realworld/*)              TARGETS=(chest forearm head shin thigh upperarm waist); CHUNKS="1-10 11-20" ;;
  esac
}

for dataset in "${DATASETS[@]}"; do
  for backbone in "${BACKBONES[@]}"; do
    plan "$dataset" "$backbone"
    for targets in "${TARGETS[@]}"; do
      for chunk in $CHUNKS; do
        # shellcheck disable=SC2086
        submit "main-$dataset-$backbone-${targets// /_}-s$chunk" 8 1-00:00:00 - \
          scripts/run_adapt.py --dataset "$dataset" --backbone "$backbone" --targets $targets \
          --seeds "$chunk" --ranks 2 4 8 --eval-steps 50 --variants $VARIANTS
      done
    done
  done
done
