# moe-serve checklist

Legend: `[ ]` todo, `[x]` done. "Learn" items are things to understand *before* the stage's build items. Each stage ends with a **gate**: don't move on until it passes.

## Big picture

Pipeline of a request in an LLM server:
**prefill** (process the whole prompt, compute-bound, builds the KV cache) → **decode** (one token per step, memory-bound, reads weights + KV cache) → repeat.
In an MoE model each layer's FFN is replaced by `E` experts, a router picks top-`r` per token. Decode at small batch only reads a few experts' weights, so it is cheap. Speculative decoding verifies `k+1` tokens at once, which touches many more experts, so the step is no longer cheap and the speedup shrinks. **That is the project's research question.** Everything else is infrastructure to measure it.

---

## Stage 0: Setup and access (before Week 1)
**Learn:** GPU basics (SMs, HBM bandwidth, NVLink vs PCIe), `nvidia-smi topo -m`, CUDA/PyTorch/Triton versions.
- [ ] Get ≥2 NVLink GPUs (lab machines or Bridges-2 via ACCESS). Verify with `nvidia-smi topo -m`
- [ ] Python env with torch, triton, transformers; `pip install -e ".[dev]"`
- [ ] Download OLMoE-1B-7B (`allenai/OLMoE-1B-7B-0924`)
- [ ] Decide hardware scope (2 vs 4 GPUs, 40GB vs 80GB) and write it in README
- **Gate:** can run HF OLMoE generate on a GPU.

## Stage 1: Single-GPU model + baselines (Week 1)
**Learn:** transformer decoder anatomy (RMSNorm, RoPE, GQA attention), MoE (router, softmax top-k gating, load balancing, shared vs routed experts), prefill vs decode, HF `modeling_olmoe.py` as reference.
- [ ] `layers.py`: RMSNorm, RoPE, embeddings
- [ ] `attention.py`: attention with a plain contiguous KV cache
- [ ] `moe.py`: router + top-k + naive per-expert loop (correct first, fast later)
- [ ] `olmoe.py`: full forward + HF weight loading
- [ ] Greedy generate loop
- [ ] Get vLLM serving OLMoE; run `bench/baselines/vllm.sh`
- [ ] Get SGLang serving OLMoE; run `bench/baselines/sglang.sh`
- [ ] `serving_bench.py`: fixed workload (prompt/output lengths, request rate), reports throughput, TTFT, TPOT
- **Gate:** `tests/test_hf_parity.py` passes (logits match HF within tolerance); baseline numbers recorded in `results/`.

## Stage 2: Paged KV cache + continuous batching (Week 2)
**Learn:** PagedAttention paper (vLLM), block tables and fragmentation, continuous (iteration-level) batching, chunked prefill, Triton programming model (programs, blocks, masks), online softmax / FlashAttention idea, KV-cache size math (`2·layers·kv_heads·head_dim·bytes` per token).
- [ ] `block_manager.py`: allocator, free list, per-sequence block tables
- [ ] `paged_cache.py`: write new K/V into blocks
- [ ] `kernels/paged_attention.py`: Triton decode kernel (one query token, gather KV via block table)
- [ ] `tests/test_paged_attention.py` vs PyTorch SDPA reference
- [ ] `scheduler.py`: continuous batching (admit/finish requests every step, prefill+decode mix)
- [ ] `engine.py`: end-to-end loop
- [ ] Sweep throughput/latency vs batch size; compare to vLLM
- **Gate:** engine serves many concurrent requests; kernel matches reference; one throughput-vs-batch plot.

## Stage 3: Tensor + expert parallel (Week 3)
**Learn:** Megatron column/row-parallel linears, NCCL all-reduce / all-to-all, `torch.distributed` process groups, expert parallelism (dispatch/combine), token permutation, grouped GEMM, load imbalance across experts/ranks, DeepSpeed-MoE / Megatron-MoE / SGLang EP design.
- [ ] `comm.py`: init, groups, wrappers
- [ ] `tp.py`: head-split attention (col QKV, row O + all-reduce)
- [ ] `moe_permute.py`: sort tokens by expert id, unpermute
- [ ] `grouped_gemm.py`: Triton grouped GEMM over ragged expert groups
- [ ] `tests/test_moe_dispatch.py` vs naive loop
- [ ] `ep.py`: all-to-all dispatch → local experts → all-to-all combine
- [ ] Also implement TP-for-experts (split each expert's FFN) for comparison
- [ ] Benchmark TP-only, EP-only, hybrid (attn TP + MoE EP) vs batch size
- **Gate:** 2-GPU outputs match 1-GPU; results table for the 3 configs.

## Stage 4: Speculative decoding (Week 4, milestone)
**Learn:** speculative sampling (accept/reject math, lossless guarantee), draft model vs EAGLE/Medusa heads, tree attention + tree masks, KV rollback, acceptance length vs k, DDP/FSDP basics.
- [ ] `draft.py`: small draft model proposes k tokens
- [ ] `verify.py`: verify k+1 tokens in one target pass (chain first)
- [ ] `rollback.py`: free/rewind KV blocks for rejected tokens; `tests/test_spec_rollback.py`
- [ ] Tree verification: tree mask in the paged attention kernel
- [ ] Log per-step: accepted length, **number of distinct experts activated**, step time
- [ ] `eagle.py` + `train/eagle_train.py`: collect target hidden states, train head with DDP
- **Gate:** spec output identical to non-spec greedy output; measured speedup vs k on dense-like vs MoE step cost. First evidence for the research question.

## Stage 5: PP, DP-attention, full sweep (Week 5)
**Learn:** pipeline parallelism (micro-batches, bubble), DP-attention + EP-MoE (SGLang), TensorRT-LLM engine build flow, Nsight Systems basics.
- [ ] `pp.py`: 2-stage pipeline, micro-batches; measure idle time vs TP
- [ ] `dp_attn.py`: replicated attention per GPU, EP MoE
- [ ] TRT-LLM running as baseline only (container; do last, time-box it)
- [ ] `sweep.py`: batch size × k × {TP, EP, hybrid, PP, DP-attn}; save CSVs to `results/`
- [ ] Same sweep (where supported) on vLLM / SGLang / TRT-LLM
- **Gate:** one complete results CSV with all configs.

## Stage 6: Model, profiling, poster (Week 6)
**Learn:** roofline model, arithmetic intensity, Nsight Systems/Compute, reading vLLM/SGLang MoE + spec-decode code.
- [ ] Fit `analysis/model.py` (expected active experts → step time → speedup) to measured data; report error
- [ ] Predict best parallelism (TP vs EP) as function of batch and k; check vs sweep
- [ ] Roofline plots (`roofline.py`)
- [ ] Nsight breakdown: communication vs compute vs memory (`nsight_breakdown.py`, `scripts/nsys_profile.sh`)
- [ ] Explain gaps vs vLLM/SGLang/TRT-LLM (kernels, scheduling, CUDA graphs, overlap)
- [ ] Poster + write-up
- **Gate:** poster answers: why spec decoding helps less on MoE, and when TP vs EP wins.

## Stretch
- [ ] Radix-tree prefix cache (`radix_cache.py`)
- [ ] Qwen3-30B-A3B on 4×80GB
- [ ] DeepSeek-V2-Lite with MLA
- [ ] CUDA graphs for decode; overlap comm with compute

---

## Skills map (what to learn, why, where)

| Skill | Why it matters here | Stage | Resource to start |
|---|---|---|---|
| Prefill vs decode | Compute-bound vs memory-bound regimes; drives everything | 1 | vLLM / "LLM inference" blog posts |
| Transformer + GQA + RoPE | You hand-write the forward pass | 1 | HF `modeling_olmoe.py` |
| MoE routing (top-k, gating) | Core of the research question | 1, 3 | OLMoE paper, Mixtral paper |
| KV cache + memory math | Sizing blocks, batch limits | 2 | PagedAttention paper |
| PagedAttention | Your first hot-path kernel | 2 | vLLM paper + source |
| Continuous batching | Throughput under load | 2 | Orca paper, vLLM scheduler |
| Triton | Both kernels | 2, 3 | Triton tutorials (fused softmax, matmul) |
| Online softmax / FlashAttention | Needed inside paged decode kernel | 2 | FlashAttention paper |
| NCCL collectives (all-reduce, all-to-all) | TP and EP comms | 3 | `torch.distributed` docs, your DLRM work |
| Tensor parallel (Megatron) | Attention/FFN splits | 3 | Megatron-LM paper |
| Expert parallel + grouped GEMM | Second hot-path kernel | 3 | DeepSpeed-MoE, Megablocks, SGLang EP |
| Speculative decoding | Verify k tokens, accept/reject | 4 | Leviathan et al., Chen et al. |
| Tree attention, EAGLE/Medusa | Tree verify + draft head | 4 | EAGLE, Medusa, SpecInfer |
| KV rollback | Correctness of spec decode | 4 | vLLM/SGLang spec-decode code |
| DDP / FSDP | EAGLE head training | 4 | PyTorch DDP tutorial |
| Pipeline parallel | Bubble measurement | 5 | GPipe, 1F1B |
| DP-attention | SGLang's MoE layout | 5 | SGLang blog / source |
| vLLM / SGLang | Baselines + code reading | 1, 5, 6 | Run, profile, read MoE + spec paths |
| TensorRT-LLM | Third baseline only | 5 | Container + `trtllm-build` docs |
| Nsight Systems / Compute, roofline | Evidence for the analysis | 6 | NVIDIA docs |
