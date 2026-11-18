"""Analytical model of speculative decoding on MoE (starter).

Core idea: with T = B*(k+1) tokens in a verify step and uniform top-r routing over
E experts, the expected number of distinct experts touched per layer is
    E_act(T) = E * (1 - (1 - r/E)^T)
Decode is weight-read bound, so MoE-layer time ~ E_act(T) * expert_bytes / BW.
Dense models pay the same weight read for T=1 and T=k+1; MoE does not.
"""
from __future__ import annotations


def expected_active_experts(tokens: int, num_experts: int = 64, top_k: int = 8) -> float:
    """Expected distinct experts hit by `tokens` tokens under uniform routing."""
    return num_experts * (1.0 - (1.0 - top_k / num_experts) ** tokens)


def moe_step_time(
    tokens: int,
    num_experts: int,
    top_k: int,
    expert_bytes: float,
    attn_bytes: float,
    bandwidth: float,
) -> float:
    """Memory-bound step time (s): attention weights + active expert weights."""
    e_act = expected_active_experts(tokens, num_experts, top_k)
    return (attn_bytes + e_act * expert_bytes) / bandwidth


def spec_speedup(
    batch: int,
    k: int,
    accept_len: float,
    num_experts: int,
    top_k: int,
    expert_bytes: float,
    attn_bytes: float,
    bandwidth: float,
    draft_cost: float = 0.0,
) -> float:
    """Speedup vs. plain decode: tokens/step gained over relative step-time increase."""
    t1 = moe_step_time(batch, num_experts, top_k, expert_bytes, attn_bytes, bandwidth)
    tk = moe_step_time(batch * (k + 1), num_experts, top_k, expert_bytes, attn_bytes, bandwidth)
    return accept_len / ((tk + draft_cost) / t1)
