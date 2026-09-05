"""Adapted from cloud-inference-from-scratch 1747158 (MIT; see LICENSE)."""
from typing import Any
import torch as _torch

def rms_norm(hidden: Any, weight: Any, eps: float) -> Any:
    dtype = hidden.dtype
    x = hidden.to(_torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * _torch.rsqrt(variance + eps)
    return weight * x.to(dtype)


def build_rope_cache(head_dim: int, max_len: int, theta: float, device: str) -> tuple[Any, Any]:
    inv_freq = 1.0 / (
        theta ** (_torch.arange(0, head_dim, 2, dtype=_torch.float32, device=device) / head_dim)
    )
    positions = _torch.arange(max_len, dtype=_torch.float32, device=device)
    angles = positions[:, None] * inv_freq[None, :]
    emb = _torch.cat((angles, angles), dim=-1)
    return emb.cos(), emb.sin()  # [max_len, head_dim], float32


def rotate_half(x: Any) -> Any:
    half = x.shape[-1] // 2
    return _torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(
    q: Any,
    k: Any,
    cos: Any,
    sin: Any,
    positions: Any,
) -> tuple[Any, Any]:
    """Rotate q/k of shape [seq, heads, head_dim] with RoPE at given positions."""
    cos_row = cos[positions][:, None, :].to(q.dtype)  # [seq, 1, head_dim]
    sin_row = sin[positions][:, None, :].to(q.dtype)
    return (q * cos_row) + (rotate_half(q) * sin_row), (k * cos_row) + (rotate_half(k) * sin_row)


def causal_attention(
    q: Any,
    keys: Any,
    values: Any,
    sm_scale: float,
    past_len: int = 0,
) -> Any:
    """Scaled dot-product causal attention.

    q: [Tq, num_heads, head_dim]; keys/values: [Tk, num_heads, head_dim].
    Returns context [Tq, num_heads * head_dim]. Softmax accumulates in fp32;
    the tensor-core matmuls below accumulate in fp32 internally as well.
    """
    query_len, num_heads, head_dim = q.shape
    key_len = keys.shape[0]
    q_heads = q.transpose(0, 1)
    key_heads = keys.transpose(0, 1)
    value_heads = values.transpose(0, 1)
    scores = _torch.matmul(q_heads, key_heads.transpose(1, 2)) * sm_scale
    positions_q = _torch.arange(query_len, device=q.device) + past_len
    allowed = positions_q[:, None] >= _torch.arange(key_len, device=q.device)[None, :]
    scores = scores.to(_torch.float32).masked_fill(~allowed[None, :, :], float("-inf"))
    probs = _torch.softmax(scores, dim=-1).to(values.dtype)
    context = _torch.matmul(probs, value_heads).transpose(0, 1).contiguous()
    return context.reshape(query_len, num_heads * head_dim)
