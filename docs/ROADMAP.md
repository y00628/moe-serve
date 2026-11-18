# Roadmap (~6 weeks)

- [ ] **Week 1** Single-GPU forward matching HF (`tests/test_hf_parity.py`), basic generate loop. vLLM + SGLang baselines running (`bench/baselines/`).
- [ ] **Week 2** Paged KV cache, continuous batching, Triton paged decode kernel. Throughput/latency vs batch size.
- [ ] **Week 3** TP attention + EP MoE (all-to-all, grouped GEMM). Benchmark TP-only, EP-only, hybrid.
- [ ] **Week 4 (milestone)** Draft-model spec decoding, then tree verification. Start EAGLE head training with DDP (`train/`).
- [ ] **Week 5** Pipeline parallel, DP-attention. TRT-LLM baseline. Full sweep: batch × k × parallelism.
- [ ] **Week 6** Analytical model, roofline, Nsight comm/compute/memory breakdown. Poster.

## Risks
- GPU access / NVLink availability (decide before committing scope).
- TRT-LLM setup time: baseline only, do last.
- Don't write everything in CUDA: only the two hot paths.

## Stretch
- Radix-tree prefix caching (`kv/radix_cache.py`)
- Qwen3-30B-A3B, DeepSeek-V2-Lite (MLA)
