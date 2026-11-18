#!/usr/bin/env bash
# Baseline: vLLM. Usage: MODEL=allenai/OLMoE-1B-7B-0924 TP=2 bash bench/baselines/vllm.sh
set -euo pipefail
vllm serve "${MODEL:-allenai/OLMoE-1B-7B-0924}" --tensor-parallel-size "${TP:-1}" ${EXTRA:-}
