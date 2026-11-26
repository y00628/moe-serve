# Learning path

For each stage: what gets built, what to understand before moving on, and a self-check question. Update the Progress column as stages finish.

Legend: ⬜ not started, 🟨 partial, ✅ done.

| Stage | Progress | Result (what gets built) | What to understand before the next stage | Self-check |
|---|---|---|---|---|
| **0. Setup** | ⬜ Not started. Repo is on GitHub; GPU access, env and model download unchecked | Working GPU env, OLMoE downloaded, `nvidia-smi topo -m` output recorded | GPU memory hierarchy (HBM, SRAM, SMs). Memory bandwidth vs FLOPs. NVLink vs PCIe bandwidth. | What is the bandwidth between your GPUs, and why does that decide whether EP is viable? |
| **1. Model + baselines** | 🟨 Scaffold only: stubs for `layers`, `attention`, `moe`, `olmoe`, baseline launch scripts, empty test. No working code | Hand-written OLMoE forward that matches HF logits. A generate loop. vLLM and SGLang numbers. A benchmark harness. | Decoder block anatomy: RMSNorm, RoPE, GQA, residuals. Router, softmax, top-k, and why experts are sparse. **Prefill vs decode**: compute-bound vs memory-bound. Metrics: TTFT, TPOT, throughput. | Why is decode at batch 1 limited by bandwidth and not FLOPs? |
| **2. Paged KV + continuous batching** | ⬜ Not started (stubs only) | Block manager, Triton paged-decode kernel, scheduler, throughput-vs-batch curve | KV-cache size math. Why paging fixes fragmentation. Block tables as indirection. Online softmax. Triton basics. Static vs continuous batching. | What does continuous batching do that static batching can't? |
| **3. TP + EP** | ⬜ Not started (stubs only) | Column/row-parallel attention, permute + grouped GEMM, all-to-all EP, TP vs EP vs hybrid benchmarks | All-reduce vs all-to-all cost. Megatron column→row split. Token permutation and grouped GEMM. Expert load imbalance. | In TP-for-MoE vs EP, what does each GPU hold and what is communicated per layer? |
| **4. Speculative decoding** | ⬜ Not started (stubs only) | Draft+verify, tree mask, KV rollback, EAGLE head trained with DDP | Accept/reject rule and why output is unchanged. Acceptance length vs `k`. Tree attention. Rollback as block rewind. DDP. Expected distinct experts `E(1-(1-r/E)^T)`. | Why does verifying 5 tokens cost almost the same as 1 for a dense model, but not for MoE? |
| **5. PP, DP-attention, sweep** | ⬜ Not started (stubs only) | PP bubble measurement, DP-attention variant, TRT-LLM baseline, full sweep CSV | Bubble fraction `(p-1)/(m+p-1)`. Why DP-attention avoids head-splitting. Sweep design and noise control. | When would PP beat TP for decode, and when would it lose? |
| **6. Analysis + poster** | 🟨 Started: `analysis/model.py` has the expected-experts and speedup formulas, 2 passing tests. Not yet fitted to measurements | Fitted model, roofline plots, Nsight breakdown, write-up | Roofline and arithmetic intensity. Reading Nsight timelines. Explaining model-vs-measurement residuals. How vLLM/SGLang close the gaps. | Where does your model predict wrong, and which errors are real physics vs engine gaps? |

See also [../checklist.md](../checklist.md) for the detailed task list.
