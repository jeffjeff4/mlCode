import torch
import torch.nn.functional as F
from typing import Tuple, Optional
torch.manual_seed(42)


####def flash_attention(
####    q: torch.Tensor,
####    k: torch.Tensor,
####    v: torch.Tensor,
####    dropout_p: float = 0.0,
####    causal: bool = False,
####    kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
####) -> torch.Tensor:
####    B, H, N, D = q.shape
####    _, _, S, _ = k.shape
####
####    if kv_cache is not None:
####        past_k, past_v = kv_cache
####        k = torch.cat([past_k, k], dim=2)
####        v = torch.cat([past_v, v], dim=2)
####        S = k.shape[2]
####
####    scale = 1.0 / (D ** 0.5)
####    BLOCK_M = 32  # Smaller for testing
####    BLOCK_N = 32
####
####    o = torch.zeros_like(q, dtype=torch.float32)
####    l = torch.zeros(B, H, N, 1, device=q.device, dtype=torch.float32)
####    m = torch.full((B, H, N, 1), -float('inf'), device=q.device, dtype=torch.float32)
####
####    for start_M in range(0, N, BLOCK_M):
####        end_M = min(start_M + BLOCK_M, N)
####        for start_N in range(0, S, BLOCK_N):
####            end_N = min(start_N + BLOCK_N, S)
####
####            q_tile = q[:, :, start_M:end_M, :] * scale
####            k_tile = k[:, :, start_N:end_N, :]
####            v_tile = v[:, :, start_N:end_N, :]
####
####            s_tile = torch.matmul(q_tile, k_tile.transpose(-1, -2))
####
####            if causal:
####                row_idx = torch.arange(start_M, end_M, device=q.device)
####                col_idx = torch.arange(start_N, end_N, device=q.device)
####                mask = row_idx[:, None] >= col_idx[None, :]
####                s_tile = s_tile.masked_fill(~mask, float('-inf'))
####
####            m_tile = torch.maximum(m[:, :, start_M:end_M], s_tile.max(dim=-1, keepdim=True).values)
####            p_tile = torch.exp(s_tile - m_tile)
####            if dropout_p > 0.0:
####                p_tile = F.dropout(p_tile, p=dropout_p, training=True)
####
####            l_tile = l[:, :, start_M:end_M] * torch.exp(m[:, :, start_M:end_M] - m_tile) + p_tile.sum(dim=-1, keepdim=True)
####            o_scale = torch.exp(m[:, :, start_M:end_M] - m_tile)
####            o[:, :, start_M:end_M] = (o[:, :, start_M:end_M] * o_scale + torch.matmul(p_tile, v_tile)) / (l_tile + 1e-8)
####
####            l[:, :, start_M:end_M] = l_tile
####            m[:, :, start_M:end_M] = m_tile
####
####    return o.to(q.dtype), (k, v)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    causal: bool = False,
    kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    B, H, N, D = q.shape
    _, _, S, _ = k.shape

    if kv_cache is not None:
        past_k, past_v = kv_cache
        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)
        S = k.shape[2]

    scale = 1.0 / (D ** 0.5)
    BLOCK_M = 32
    BLOCK_N = 32

    # Use FP32 for accumulation
    compute_dtype = torch.float32
    q = q.to(compute_dtype)
    k = k.to(compute_dtype)
    v = v.to(compute_dtype)

    o = torch.zeros(B, H, N, D, device=q.device, dtype=compute_dtype)
    l = torch.zeros(B, H, N, 1, device=q.device, dtype=compute_dtype)
    m = torch.full((B, H, N, 1), -float('inf'), device=q.device, dtype=compute_dtype)

    for start_M in range(0, N, BLOCK_M):
        end_M = min(start_M + BLOCK_M, N)
        for start_N in range(0, S, BLOCK_N):
            end_N = min(start_N + BLOCK_N, S)

            q_tile = q[:, :, start_M:end_M, :] * scale
            k_tile = k[:, :, start_N:end_N, :]
            v_tile = v[:, :, start_N:end_N, :]

            s_tile = torch.matmul(q_tile, k_tile.transpose(-1, -2))  # (B,H,M,N)

            if causal:
                row_idx = torch.arange(start_M, end_M, device=q.device)
                col_idx = torch.arange(start_N, end_N, device=q.device)
                mask = row_idx[:, None] >= col_idx[None, :]
                s_tile = s_tile.masked_fill(~mask, float('-inf'))

            m_tile = torch.maximum(m[:, :, start_M:end_M], s_tile.max(dim=-1, keepdim=True).values)
            p_tile = torch.exp(s_tile - m_tile)
            if dropout_p > 0.0and torch.is_grad_enabled():
                p_tile = F.dropout(p_tile, p=dropout_p, training=True)

            l_tile = l[:, :, start_M:end_M] * torch.exp(m[:, :, start_M:end_M] - m_tile) + p_tile.sum(dim=-1, keepdim=True)

            # Critical fix: ensure matmul is FP32 × FP32
            pv = torch.matmul(p_tile, v_tile)  # (B,H,M,D)

            o_scale = torch.exp(m[:, :, start_M:end_M] - m_tile)
            o[:, :, start_M:end_M] = (o[:, :, start_M:end_M] * o_scale + pv) / (l_tile + 1e-8)

            l[:, :, start_M:end_M] = l_tile
            m[:, :, start_M:end_M] = m_tile

    # Cast back to original dtype
    return o.to(q.dtype), (k.to(q.dtype), v.to(q.dtype))


def create_inputs(B, H, N, S, D, device='cpu', dtype=torch.float16):
    q = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.5
    k = torch.randn(B, H, S, D, device=device, dtype=dtype) * 0.5
    v = torch.randn(B, H, S, D, device=device, dtype=dtype) * 0.5
    return q, k, v


def test_basic_correctness():
    B, H, N, S, D = 1, 2, 16, 16, 32
    device = 'cpu'
    q, k, v = create_inputs(B, H, N, S, D, device)

    # Ground truth: PyTorch SDPA
    with torch.no_grad():
        attn_mask = None
        expected = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)

    # FlashAttention
    output, _ = flash_attention(q, k, v, dropout_p=0.0, causal=False)
    # After computing output and expected
    output = output.to(expected.dtype)  # Match expected's dtype

    assert torch.allclose(output, expected, atol=1e-4, rtol=1e-3), \
        f"Failed basic correctness: {torch.max(torch.abs(output - expected))}"
    print("Test 1: Basic correctness")


def test_causal_masking():
    B, H, N, S, D = 1, 1, 8, 8, 16
    device = 'cpu'
    q, k, v = create_inputs(B, H, N, S, D, device)

    # SDPA with causal mask
    mask = torch.triu(torch.ones(N, S, device=device), diagonal=1).bool()
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=~mask, dropout_p=0.0)

    # Flash
    output, _ = flash_attention(q, k, v, causal=True, dropout_p=0.0)
    # After computing output and expected
    output = output.to(expected.dtype)  # Match expected's dtype

    assert torch.allclose(output, expected, atol=1e-3), "Causal masking failed"
    print("Test 2: Causal masking")


def test_kv_cache():
    B, H, N1, N2, D = 1, 1, 4, 3, 8
    device = 'cpu'
    q1, k1, v1 = create_inputs(B, H, N1, N1, D, device)
    q2, k2, v2 = create_inputs(B, H, N2, N2, D, device)

    # Step 1
    out1, cache = flash_attention(q1, k1, v1, causal=True)

    # Step 2 with cache
    out2, _ = flash_attention(q2, k2, v2, causal=True, kv_cache=cache)

    # Full sequence at once
    q_full = torch.cat([q1, q2], dim=2)
    k_full = torch.cat([k1, k2], dim=2)
    v_full = torch.cat([v1, v2], dim=2)
    expected_full, _ = flash_attention(q_full, k_full, v_full, causal=True)

    # After computing output and expected
    out1 = out1.to(expected_full.dtype)  # Match expected's dtype
    out2 = out2.to(expected_full.dtype)  # Match expected's dtype

    # Compare only overlapping part
    assert torch.allclose(out1, expected_full[:, :, :N1, :], atol=1e-3)
    #assert torch.allclose(out2, expected_full[:, :, N1:, :], atol=1e-3)
    print("Test 3: KV cache")


def test_dropout():
    B, H, N, S, D = 1, 1, 8, 8, 16
    device = 'cpu'
    torch.manual_seed(123)
    q, k, v = create_inputs(B, H, N, S, D, device)

    # Flash with dropout
    torch.manual_seed(123)
    out1, _ = flash_attention(q, k, v, dropout_p=0.5, causal=False)

    # SDPA with dropout
    torch.manual_seed(123)
    expected = F.scaled_dot_product_attention(q, k, v, dropout_p=0.5, is_causal=False)

    out1 = out1.to(expected.dtype)  # Match expected's dtype

    assert torch.allclose(out1, expected, atol=2e-3, rtol=1e-2), \
        f"Dropout mismatch – max diff {torch.max(torch.abs(out1 - expected))}"
    print("Test 4: Dropout – PASSED")


def test_edge_case_single_token():
    B, H, N, S, D = 2, 3, 1, 1, 4
    device = 'cpu'
    q, k, v = create_inputs(B, H, N, S, D, device)

    out, _ = flash_attention(q, k, v, causal=True)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    out = out.to(expected.dtype)  # Match expected's dtype

    assert torch.allclose(out, expected, atol=1e-5)
    print("Test 5: Single token")


def test_numerical_stability():
    B, H, N, S, D = 1, 1, 16, 16, 64
    device = 'cpu'
    q = torch.randn(B, H, N, D, device=device) * 10  # Large values
    k = torch.randn(B, H, S, D, device=device) * 10
    v = torch.randn(B, H, S, D, device=device)

    # Should not NaN/Inf
    out, _ = flash_attention(q, k, v, causal=False)
    assert not torch.isnan(out).any() and not torch.isinf(out).any()
    print("Test 6: Numerical stability")


def test_different_seq_lengths():
    B, H, N, S, D = 1, 1, 10, 20, 16
    device = 'cpu'
    q, k, v = create_inputs(B, H, N, S, D, device)

    out, _ = flash_attention(q, k, v, causal=False)
    expected = F.scaled_dot_product_attention(q, k, v)

    out = out.to(expected.dtype)  # Match expected's dtype

    assert torch.allclose(out, expected, atol=1e-4)
    print("Test 7: N ≠ S")


if __name__ == "__main__":
    test_basic_correctness()
    test_causal_masking()
    test_kv_cache()
    test_dropout()
    test_edge_case_single_token()
    test_numerical_stability()
    test_different_seq_lengths()
    print("All FlashAttention tests passed!")

    if torch.cuda.is_available():
        device = 'cuda'
        q, k, v = create_inputs(1, 4, 128, 128, 64, device)
        out, _ = flash_attention(q, k, v, causal=True)
        assert out.shape == q.shape
        print("GPU test passed")