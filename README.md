# moe-serve

A mini MoE inference engine (PyTorch + Triton) that serves OLMoE-1B-7B across 2–4 GPUs, benchmarked against vLLM, SGLang and TensorRT-LLM.

## Research question

Speculative decoding helps much less on MoE models than on dense ones. Why? And how does the best parallelism setup (tensor vs. expert parallel) change with batch size and speculation length k?

Verifying k draft tokens at once routes those tokens to more distinct experts. A decode step that was cheap at batch 1 (few experts' weights read) becomes memory-bound on many more experts, so the dense-model speedup shrinks. We measure this, explain it, and fit a simple analytical model (`analysis/`).

## Scope

Hand-written hot paths only; PyTorch for control flow.

1. Paged decode attention (Triton), with a tree-mask variant for verification
2. MoE permute + grouped GEMM (Triton)

| Area | Where |
|---|---|
| Model forward (attention, router, top-k, experts) | `src/moe_serve/model/` |
| Paged KV, block tables, rollback | `src/moe_serve/kv/` |
| Kernels | `src/moe_serve/kernels/` |
| TP / EP / PP / DP-attention | `src/moe_serve/parallel/` |
| Spec decoding (draft, EAGLE head, tree verify) | `src/moe_serve/spec/`, `train/` |
| Scheduler / continuous batching | `src/moe_serve/engine/` |
| Baselines, sweeps, profiling | `bench/`, `scripts/` |
| Analytical model, roofline, Nsight | `analysis/` |

## Hardware

Needs ≥2 NVLink-connected GPUs for meaningful TP/EP results (PCIe all-to-all drowns everything). Line up access first (lab machines or Bridges-2 via ACCESS).

## Models

- Dev: OLMoE-1B-7B (64 experts)
- Stretch: Qwen3-30B-A3B (4×80GB), DeepSeek-V2-Lite (MLA)

## Plan

See [docs/ROADMAP.md](docs/ROADMAP.md).

## Setup

```bash
pip install -e ".[dev]"
pytest tests/
```
