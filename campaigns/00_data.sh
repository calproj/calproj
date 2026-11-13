#!/usr/bin/env bash
# Window each dataset once and cache the arrays (scripts/prepare_data.py).
# Prints "dataset jobid" lines; the source campaign waits on these jobs.
source "$(dirname "$0")/common.sh"

for dataset in "${DATASETS[@]}"; do
  case "$dataset" in
    realdisp) cpus=16; time=0-12:00:00 ;;   # 7 GB of text logs
    *)        cpus=8;  time=0-04:00:00 ;;
  esac
  echo "$dataset $(submit "data-$dataset" "$cpus" "$time" - scripts/prepare_data.py --dataset "$dataset")"
done
