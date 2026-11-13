#!/usr/bin/env bash
# Pretrain the source models: one job per dataset, backbone and target domain.
#   SOURCE_SEEDS   default "1": one source model per target, reused for every
#                  seed, as the MemFLoRA release's scripts do
#   DEP_<dataset>  job id of that dataset's 00_data.sh job, to wait on it
source "$(dirname "$0")/common.sh"
: "${SOURCE_SEEDS:=1}"

targets_of() {
  case "$1" in
    opportunity) echo "1 2 3 4" ;;
    realdisp)    echo "self" ;;
    realworld)   echo "chest forearm head shin thigh upperarm waist" ;;
  esac
}

for dataset in "${DATASETS[@]}"; do
  dep_var="DEP_$dataset"; dep="${!dep_var:--}"
  for backbone in "${BACKBONES[@]}"; do
    for target in $(targets_of "$dataset"); do
      # shellcheck disable=SC2086
      submit "source-$dataset-$backbone-$target-s${SOURCE_SEEDS// /_}" 8 0-12:00:00 "$dep" \
        scripts/pretrain_source.py --dataset "$dataset" --backbone "$backbone" \
        --targets "$target" --source-seeds $SOURCE_SEEDS
    done
  done
done
