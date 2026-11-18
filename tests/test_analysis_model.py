from analysis.model import expected_active_experts, spec_speedup


def test_active_experts_bounds():
    assert abs(expected_active_experts(1, 64, 8) - 8) < 1e-9
    assert expected_active_experts(1000, 64, 8) > 63.9


def test_moe_gains_less_than_dense():
    # Dense-like (1 expert, always active) vs MoE, same accept length.
    kw = dict(batch=1, k=4, accept_len=3.0, expert_bytes=1e8, attn_bytes=1e8, bandwidth=1e12)
    dense = spec_speedup(num_experts=1, top_k=1, **kw)
    moe = spec_speedup(num_experts=64, top_k=8, **kw)
    assert moe < dense
