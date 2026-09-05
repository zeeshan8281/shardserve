"""Adapted MIT ragged kernel from cloud-inference-from-scratch 1747158.
BF16 extension is UNVERIFIED on CUDA. Host bounds validated in coordinator.
"""
from typing import Any
import triton
import triton.language as tl
SUPPORTED_RAGGED_HEAD_DIMS = (64, 128)
SUPPORTED_BLOCK_SIZE = 16
class KernelUnsupported(ValueError): pass

def ragged_attention_direct(
    q: Any,
    k_pool_layer: Any,
    v_pool_layer: Any,
    block_tables: Any,
    query_seq_ids: Any,
    positions: Any,
    sm_scale: float,
) -> Any:
    """Attention for a flat mixed-query batch over physical KV blocks.

    Each query token carries its sequence row and absolute position. The kernel
    therefore handles decode and chunked prefill in one launch without building
    logical K/V tensors.
    """
    import torch

    if q.device.type != "cuda" or q.dtype != torch.bfloat16:
        raise KernelUnsupported("ragged kernel requires BF16 CUDA queries")
    if q.dim() != 3 or q.shape[-1] not in SUPPORTED_RAGGED_HEAD_DIMS:
        raise KernelUnsupported(
            f"ragged q must be [tokens, heads, 64|128], got {tuple(q.shape)}"
        )
    if k_pool_layer.shape != v_pool_layer.shape or k_pool_layer.dim() != 4:
        raise KernelUnsupported("ragged K/V pools must have identical rank-4 shapes")
    if k_pool_layer.dtype != torch.bfloat16 or k_pool_layer.shape[1] != SUPPORTED_BLOCK_SIZE:
        raise KernelUnsupported("ragged KV pools must be BF16 with block_size=16")
    if k_pool_layer.shape[-1] != q.shape[-1]:
        raise KernelUnsupported("query and KV head dimensions differ")
    if block_tables.device != q.device or block_tables.dim() != 2:
        raise KernelUnsupported("block_tables must be a device-resident rank-2 tensor")
    if query_seq_ids.numel() != q.shape[0] or positions.numel() != q.shape[0]:
        raise KernelUnsupported("one sequence id and absolute position are required per query")
    if q.shape[0] < 1:
        raise KernelUnsupported("ragged batch cannot be empty")
    tokens, num_heads, head_dim = q.shape
    kv_heads = k_pool_layer.shape[2]
    if num_heads % kv_heads:
        raise KernelUnsupported("query heads must be divisible by KV heads")
    out = torch.empty_like(q)
    _ragged_paged_attention_kernel[(tokens, num_heads)](
        q,
        block_tables,
        query_seq_ids,
        positions,
        k_pool_layer,
        v_pool_layer,
        out,
        sm_scale,
        num_heads // kv_heads,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        block_tables.stride(0),
        block_tables.stride(1),
        k_pool_layer.stride(0),
        k_pool_layer.stride(1),
        k_pool_layer.stride(2),
        k_pool_layer.stride(3),
        v_pool_layer.stride(0),
        v_pool_layer.stride(1),
        v_pool_layer.stride(2),
        v_pool_layer.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK=SUPPORTED_BLOCK_SIZE,
        HEAD_DIM=head_dim,
        num_warps=4 if head_dim == 64 else 8,
        num_stages=2,
    )
    return out


@triton.jit
def _ragged_paged_attention_kernel(
    Q,
    BT,
    Q_SEQ,
    POS,
    KP,
    VP,
    OUT,
    sm_scale,
    GROUP,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_btb,
    stride_btn,
    stride_kn,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_ot,
    stride_oh,
    stride_od,
    BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    seq = tl.load(Q_SEQ + pid_t)
    absolute_position = tl.load(POS + pid_t)
    seq_len = absolute_position + 1
    kv_h = pid_h // GROUP

    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + pid_t * stride_qt + pid_h * stride_qh + offs_d * stride_qd).to(
        tl.float32
    )
    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for block_index in range(tl.cdiv(seq_len, BLOCK)):
        phys = tl.load(BT + seq * stride_btb + block_index * stride_btn)
        slots = tl.arange(0, BLOCK)
        key_positions = block_index * BLOCK + slots
        valid = key_positions < seq_len
        k_ptrs = (
            KP
            + phys.to(tl.int64) * stride_kn
            + slots[:, None] * stride_ks
            + kv_h * stride_kh
            + offs_d[None, :] * stride_kd
        )
        k_tile = tl.load(k_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(k_tile * q[None, :], axis=1) * sm_scale
        scores = tl.where(valid, scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        probs = tl.exp(scores - m_new)
        l_i = l_i * alpha + tl.sum(probs, axis=0)

        v_ptrs = (
            VP
            + phys.to(tl.int64) * stride_vn
            + slots[:, None] * stride_vs
            + kv_h * stride_vh
            + offs_d[None, :] * stride_vd
        )
        v_tile = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(probs[:, None] * v_tile, axis=0)
        m_i = m_new

    result = acc / l_i
    tl.store(
        OUT + pid_t * stride_ot + pid_h * stride_oh + offs_d * stride_od,
        result.to(OUT.dtype.element_ty),
    )
