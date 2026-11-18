#!/usr/bin/env bash
# Usage: bash scripts/nsys_profile.sh out_name -- python bench/serving_bench.py ...
set -euo pipefail
out="$1"; shift; [ "$1" = "--" ] && shift
nsys profile -o "results/$out" --trace=cuda,nvtx,nvlink --force-overwrite=true "$@"
