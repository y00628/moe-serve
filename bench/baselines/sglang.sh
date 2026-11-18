#!/usr/bin/env bash
# Baseline: SGLang. Usage: MODEL=allenai/OLMoE-1B-7B-0924 TP=2 bash bench/baselines/sglang.sh
set -euo pipefail
python -m sglang.launch_server --model-path "${MODEL:-allenai/OLMoE-1B-7B-0924}" --tp "${TP:-1}" ${EXTRA:-}
