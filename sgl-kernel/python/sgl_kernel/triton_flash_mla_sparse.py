"""
Triton-based Flash MLA Sparse Attention Kernel for AMD GPUs (MI355, CDNA4)

This module implements sparse attention prefill kernel compatible with FlashMLA's
flash_mla_sparse_fwd interface.

Algorithm (from FlashMLA README):
    P = (Q @ focused_kv.T) * sm_scale * log2(e)  # [s_q, h_q, topk]
    max_logits = P.max(dim=-1)  # [s_q, h_q]
    lse = log2sumexp2(P, dim=-1)  # [s_q, h_q], base-2
    S = exp2(P - lse)  # [s_q, h_q, topk]
    out = S @ focused_kv[:, :, :d_v]  # [s_q, h_q, d_v]
"""

import math
from typing import Tuple

import torch
import triton
import triton.language as tl


# Constants
LOG2E = math.log2(math.e)  # 1.44269504


@triton.autotune(
    configs=[
        # Configs for small h_q (tensor parallelism, e.g., h_q=16)
        triton.Config({'BLOCK_TOPK': 64, 'BLOCK_H': 16, 'TILE_D': 64}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_TOPK': 64, 'BLOCK_H': 16, 'TILE_D': 64}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_TOPK': 128, 'BLOCK_H': 16, 'TILE_D': 64}, num_warps=8, num_stages=1),
        # Configs for medium h_q (h_q=32-64)
        triton.Config({'BLOCK_TOPK': 64, 'BLOCK_H': 32, 'TILE_D': 64}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_TOPK': 128, 'BLOCK_H': 32, 'TILE_D': 64}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_TOPK': 64, 'BLOCK_H': 64, 'TILE_D': 64}, num_warps=8, num_stages=1),
        # Configs for large h_q (h_q>=64)
        triton.Config({'BLOCK_TOPK': 64, 'BLOCK_H': 128, 'TILE_D': 64}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_TOPK': 128, 'BLOCK_H': 64, 'TILE_D': 64}, num_warps=8, num_stages=1),
    ],
    key=['topk'],  # Only key on topk to avoid recompilation for different s_q/h_q
)
@triton.jit
def _sparse_attention_fwd_kernel_optimized(
    # Pointers
    Q_ptr,
    KV_ptr,
    Indices_ptr,
    Out_ptr,
    MaxLogits_ptr,
    LSE_ptr,
    # Dimensions
    batch,
    s_q,
    s_kv,
    h_q,
    topk,
    # Scales
    sm_scale_log2,
    # Strides for Q [batch, s_q, h_q, d_qk]
    stride_q_b,
    stride_q_s,
    stride_q_h,
    # Strides for KV [batch, s_kv, h_kv, d_qk]
    stride_kv_b,
    stride_kv_s,
    # Strides for indices [batch, s_q, h_kv, topk]
    stride_idx_b,
    stride_idx_s,
    # Strides for output [batch, s_q, h_q, d_v]
    stride_o_b,
    stride_o_s,
    stride_o_h,
    # Block sizes (autotuned)
    BLOCK_TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TILE_D: tl.constexpr,
):
    """
    Optimized sparse attention kernel for AMD GPUs with batch support.
    
    Each program handles BLOCK_H heads for one query position in one batch.
    Uses tiled computation with online softmax.
    
    Optimization: Pre-load Q tiles outside the topk loop to reduce redundant loads.
    
    Following tilelang convention:
    - dtype (storage): bfloat16
    - accum_dtype (computation): float32
    """
    pid_b = tl.program_id(0)  # Batch index
    pid_s = tl.program_id(1)  # Query position
    pid_h_block = tl.program_id(2)  # Head block
    
    h_start = pid_h_block * BLOCK_H
    h_offs = h_start + tl.arange(0, BLOCK_H)
    h_mask = h_offs < h_q
    
    # Initialize online softmax state [BLOCK_H] - fp32 for numerical stability
    m_prev = tl.full([BLOCK_H], -1e30, dtype=tl.float32)
    l_prev = tl.zeros([BLOCK_H], dtype=tl.float32)
    
    # Output accumulators [BLOCK_H, 512] split into tiles - fp32 (same as tilelang)
    acc_0 = tl.zeros([BLOCK_H, TILE_D], dtype=tl.float32)
    acc_1 = tl.zeros([BLOCK_H, TILE_D], dtype=tl.float32)
    acc_2 = tl.zeros([BLOCK_H, TILE_D], dtype=tl.float32)
    acc_3 = tl.zeros([BLOCK_H, TILE_D], dtype=tl.float32)
    acc_4 = tl.zeros([BLOCK_H, TILE_D], dtype=tl.float32)
    acc_5 = tl.zeros([BLOCK_H, TILE_D], dtype=tl.float32)
    acc_6 = tl.zeros([BLOCK_H, TILE_D], dtype=tl.float32)
    acc_7 = tl.zeros([BLOCK_H, TILE_D], dtype=tl.float32)
    
    # Base pointers (with batch offset)
    q_base = Q_ptr + pid_b * stride_q_b + pid_s * stride_q_s
    kv_base = KV_ptr + pid_b * stride_kv_b
    idx_base = Indices_ptr + pid_b * stride_idx_b + pid_s * stride_idx_s
    
    # Pre-load Q tiles outside the topk loop (9 tiles of [BLOCK_H, TILE_D])
    # This avoids loading Q 9 times per topk block iteration
    d_offs_0 = tl.arange(0, TILE_D)
    q_ptrs_0 = q_base + h_offs[:, None] * stride_q_h + d_offs_0[None, :]
    q_tile_0 = tl.load(q_ptrs_0, mask=h_mask[:, None], other=0.0)
    
    d_offs_1 = 64 + tl.arange(0, TILE_D)
    q_ptrs_1 = q_base + h_offs[:, None] * stride_q_h + d_offs_1[None, :]
    q_tile_1 = tl.load(q_ptrs_1, mask=h_mask[:, None], other=0.0)
    
    d_offs_2 = 128 + tl.arange(0, TILE_D)
    q_ptrs_2 = q_base + h_offs[:, None] * stride_q_h + d_offs_2[None, :]
    q_tile_2 = tl.load(q_ptrs_2, mask=h_mask[:, None], other=0.0)
    
    d_offs_3 = 192 + tl.arange(0, TILE_D)
    q_ptrs_3 = q_base + h_offs[:, None] * stride_q_h + d_offs_3[None, :]
    q_tile_3 = tl.load(q_ptrs_3, mask=h_mask[:, None], other=0.0)
    
    d_offs_4 = 256 + tl.arange(0, TILE_D)
    q_ptrs_4 = q_base + h_offs[:, None] * stride_q_h + d_offs_4[None, :]
    q_tile_4 = tl.load(q_ptrs_4, mask=h_mask[:, None], other=0.0)
    
    d_offs_5 = 320 + tl.arange(0, TILE_D)
    q_ptrs_5 = q_base + h_offs[:, None] * stride_q_h + d_offs_5[None, :]
    q_tile_5 = tl.load(q_ptrs_5, mask=h_mask[:, None], other=0.0)
    
    d_offs_6 = 384 + tl.arange(0, TILE_D)
    q_ptrs_6 = q_base + h_offs[:, None] * stride_q_h + d_offs_6[None, :]
    q_tile_6 = tl.load(q_ptrs_6, mask=h_mask[:, None], other=0.0)
    
    d_offs_7 = 448 + tl.arange(0, TILE_D)
    q_ptrs_7 = q_base + h_offs[:, None] * stride_q_h + d_offs_7[None, :]
    q_tile_7 = tl.load(q_ptrs_7, mask=h_mask[:, None], other=0.0)
    
    d_offs_8 = 512 + tl.arange(0, TILE_D)
    q_ptrs_8 = q_base + h_offs[:, None] * stride_q_h + d_offs_8[None, :]
    q_tile_8 = tl.load(q_ptrs_8, mask=h_mask[:, None] & (d_offs_8[None, :] < 576), other=0.0)
    
    # Process topk in blocks
    for t_block in range(0, topk, BLOCK_TOPK):
        t_offs = t_block + tl.arange(0, BLOCK_TOPK)
        t_mask = t_offs < topk
        
        # Load indices [BLOCK_TOPK]
        indices = tl.load(idx_base + t_offs, mask=t_mask, other=-1)
        valid = (indices >= 0) & (indices < s_kv)
        safe_idx = tl.where(valid, indices, 0)
        
        # Compute Q @ K^T using pre-loaded Q tiles
        scores = tl.zeros([BLOCK_H, BLOCK_TOPK], dtype=tl.float32)
        
        # Tile 0: 0-64
        k_ptrs = kv_base + safe_idx[:, None] * stride_kv_s + d_offs_0[None, :]
        k_tile = tl.load(k_ptrs, mask=valid[:, None], other=0.0)
        scores += tl.dot(q_tile_0, tl.trans(k_tile))
        
        # Tile 1: 64-128
        k_ptrs = kv_base + safe_idx[:, None] * stride_kv_s + d_offs_1[None, :]
        k_tile = tl.load(k_ptrs, mask=valid[:, None], other=0.0)
        scores += tl.dot(q_tile_1, tl.trans(k_tile))
        
        # Tile 2: 128-192
        k_ptrs = kv_base + safe_idx[:, None] * stride_kv_s + d_offs_2[None, :]
        k_tile = tl.load(k_ptrs, mask=valid[:, None], other=0.0)
        scores += tl.dot(q_tile_2, tl.trans(k_tile))
        
        # Tile 3: 192-256
        k_ptrs = kv_base + safe_idx[:, None] * stride_kv_s + d_offs_3[None, :]
        k_tile = tl.load(k_ptrs, mask=valid[:, None], other=0.0)
        scores += tl.dot(q_tile_3, tl.trans(k_tile))
        
        # Tile 4: 256-320
        k_ptrs = kv_base + safe_idx[:, None] * stride_kv_s + d_offs_4[None, :]
        k_tile = tl.load(k_ptrs, mask=valid[:, None], other=0.0)
        scores += tl.dot(q_tile_4, tl.trans(k_tile))
        
        # Tile 5: 320-384
        k_ptrs = kv_base + safe_idx[:, None] * stride_kv_s + d_offs_5[None, :]
        k_tile = tl.load(k_ptrs, mask=valid[:, None], other=0.0)
        scores += tl.dot(q_tile_5, tl.trans(k_tile))
        
        # Tile 6: 384-448
        k_ptrs = kv_base + safe_idx[:, None] * stride_kv_s + d_offs_6[None, :]
        k_tile = tl.load(k_ptrs, mask=valid[:, None], other=0.0)
        scores += tl.dot(q_tile_6, tl.trans(k_tile))
        
        # Tile 7: 448-512
        k_ptrs = kv_base + safe_idx[:, None] * stride_kv_s + d_offs_7[None, :]
        k_tile = tl.load(k_ptrs, mask=valid[:, None], other=0.0)
        scores += tl.dot(q_tile_7, tl.trans(k_tile))
        
        # Tile 8: 512-576
        k_ptrs = kv_base + safe_idx[:, None] * stride_kv_s + d_offs_8[None, :]
        k_tile = tl.load(k_ptrs, mask=valid[:, None] & (d_offs_8[None, :] < 576), other=0.0)
        scores += tl.dot(q_tile_8, tl.trans(k_tile))
        
        # Scale and mask
        scores = scores * sm_scale_log2
        scores = tl.where(valid[None, :], scores, -1e30)
        
        # Online softmax - fp32
        m_cur = tl.max(scores, axis=1)  # [BLOCK_H]
        m_new = tl.maximum(m_prev, m_cur)
        
        alpha = tl.exp2(m_prev - m_new)
        p = tl.exp2(scores - m_new[:, None])  # [BLOCK_H, BLOCK_TOPK]
        l_cur = tl.sum(p, axis=1)  # [BLOCK_H]
        l_new = l_prev * alpha + l_cur
        
        m_prev = m_new
        l_prev = l_new
        
        # Rescale accumulators - fp32
        acc_0 = acc_0 * alpha[:, None]
        acc_1 = acc_1 * alpha[:, None]
        acc_2 = acc_2 * alpha[:, None]
        acc_3 = acc_3 * alpha[:, None]
        acc_4 = acc_4 * alpha[:, None]
        acc_5 = acc_5 * alpha[:, None]
        acc_6 = acc_6 * alpha[:, None]
        acc_7 = acc_7 * alpha[:, None]
        
        # Accumulate P @ V: [BLOCK_H, BLOCK_TOPK] @ [BLOCK_TOPK, d_v] -> [BLOCK_H, d_v]
        # bf16 input, fp32 accumulate (matches hardware behavior)
        p_bf16 = p.to(tl.bfloat16)
        
        # Tile 0: 0-64
        v_ptrs = kv_base + safe_idx[:, None] * stride_kv_s + d_offs_0[None, :]
        v_tile = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.bfloat16)
        acc_0 += tl.dot(p_bf16, v_tile)
        
        # Tile 1: 64-128
        v_ptrs = kv_base + safe_idx[:, None] * stride_kv_s + d_offs_1[None, :]
        v_tile = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.bfloat16)
        acc_1 += tl.dot(p_bf16, v_tile)
        
        # Tile 2: 128-192
        v_ptrs = kv_base + safe_idx[:, None] * stride_kv_s + d_offs_2[None, :]
        v_tile = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.bfloat16)
        acc_2 += tl.dot(p_bf16, v_tile)
        
        # Tile 3: 192-256
        v_ptrs = kv_base + safe_idx[:, None] * stride_kv_s + d_offs_3[None, :]
        v_tile = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.bfloat16)
        acc_3 += tl.dot(p_bf16, v_tile)
        
        # Tile 4: 256-320
        v_ptrs = kv_base + safe_idx[:, None] * stride_kv_s + d_offs_4[None, :]
        v_tile = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.bfloat16)
        acc_4 += tl.dot(p_bf16, v_tile)
        
        # Tile 5: 320-384
        v_ptrs = kv_base + safe_idx[:, None] * stride_kv_s + d_offs_5[None, :]
        v_tile = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.bfloat16)
        acc_5 += tl.dot(p_bf16, v_tile)
        
        # Tile 6: 384-448
        v_ptrs = kv_base + safe_idx[:, None] * stride_kv_s + d_offs_6[None, :]
        v_tile = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.bfloat16)
        acc_6 += tl.dot(p_bf16, v_tile)
        
        # Tile 7: 448-512
        v_ptrs = kv_base + safe_idx[:, None] * stride_kv_s + d_offs_7[None, :]
        v_tile = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.bfloat16)
        acc_7 += tl.dot(p_bf16, v_tile)
    
    # Finalize output
    safe_l = tl.where(l_prev > 0, l_prev, 1.0)
    inv_l = 1.0 / safe_l  # fp32
    
    # Store output [BLOCK_H, 512] - convert from fp32 accumulator to bf16 storage
    out_base = Out_ptr + pid_b * stride_o_b + pid_s * stride_o_s
    
    # Tile 0
    out_ptrs = out_base + h_offs[:, None] * stride_o_h + d_offs_0[None, :]
    tl.store(out_ptrs, (acc_0 * inv_l[:, None]).to(tl.bfloat16), mask=h_mask[:, None])
    
    # Tile 1
    out_ptrs = out_base + h_offs[:, None] * stride_o_h + d_offs_1[None, :]
    tl.store(out_ptrs, (acc_1 * inv_l[:, None]).to(tl.bfloat16), mask=h_mask[:, None])
    
    # Tile 2
    out_ptrs = out_base + h_offs[:, None] * stride_o_h + d_offs_2[None, :]
    tl.store(out_ptrs, (acc_2 * inv_l[:, None]).to(tl.bfloat16), mask=h_mask[:, None])
    
    # Tile 3
    out_ptrs = out_base + h_offs[:, None] * stride_o_h + d_offs_3[None, :]
    tl.store(out_ptrs, (acc_3 * inv_l[:, None]).to(tl.bfloat16), mask=h_mask[:, None])
    
    # Tile 4
    out_ptrs = out_base + h_offs[:, None] * stride_o_h + d_offs_4[None, :]
    tl.store(out_ptrs, (acc_4 * inv_l[:, None]).to(tl.bfloat16), mask=h_mask[:, None])
    
    # Tile 5
    out_ptrs = out_base + h_offs[:, None] * stride_o_h + d_offs_5[None, :]
    tl.store(out_ptrs, (acc_5 * inv_l[:, None]).to(tl.bfloat16), mask=h_mask[:, None])
    
    # Tile 6
    out_ptrs = out_base + h_offs[:, None] * stride_o_h + d_offs_6[None, :]
    tl.store(out_ptrs, (acc_6 * inv_l[:, None]).to(tl.bfloat16), mask=h_mask[:, None])
    
    # Tile 7
    out_ptrs = out_base + h_offs[:, None] * stride_o_h + d_offs_7[None, :]
    tl.store(out_ptrs, (acc_7 * inv_l[:, None]).to(tl.bfloat16), mask=h_mask[:, None])
    
    # Store max_logits and lse (with batch offset)
    max_logits = tl.where(l_prev > 0, m_prev, -float("inf"))
    lse = tl.where(l_prev > 0, tl.log2(safe_l) + m_prev, -float("inf"))
    
    ml_base = MaxLogits_ptr + pid_b * s_q * h_q + pid_s * h_q
    lse_base = LSE_ptr + pid_b * s_q * h_q + pid_s * h_q
    tl.store(ml_base + h_offs, max_logits, mask=h_mask)
    tl.store(lse_base + h_offs, lse, mask=h_mask)


@triton.jit
def _sparse_attention_fwd_kernel_simple(
    # Pointers
    Q_ptr,
    KV_ptr,
    Indices_ptr,
    Out_ptr,
    MaxLogits_ptr,
    LSE_ptr,
    # Dimensions
    batch,
    s_q,
    s_kv,
    h_q,
    topk,
    # Scales
    sm_scale_log2,
    # Strides for Q [batch, s_q, h_q, d_qk]
    stride_q_b,
    stride_q_s,
    stride_q_h,
    # Strides for KV [batch, s_kv, h_kv, d_qk]
    stride_kv_b,
    stride_kv_s,
    # Strides for indices [batch, s_q, h_kv, topk]
    stride_idx_b,
    stride_idx_s,
    # Strides for output [batch, s_q, h_q, d_v]
    stride_o_b,
    stride_o_s,
    stride_o_h,
    # Block sizes
    BLOCK_TOPK: tl.constexpr,
    TILE_D: tl.constexpr,
):
    """
    Simple sparse attention kernel with batch support - one head per program.
    """
    pid_b = tl.program_id(0)  # Batch index
    pid_s = tl.program_id(1)  # Query position
    pid_h = tl.program_id(2)  # Head index
    
    if pid_h >= h_q:
        return
    
    q_base = Q_ptr + pid_b * stride_q_b + pid_s * stride_q_s + pid_h * stride_q_h
    kv_base = KV_ptr + pid_b * stride_kv_b
    idx_base = Indices_ptr + pid_b * stride_idx_b + pid_s * stride_idx_s
    out_base = Out_ptr + pid_b * stride_o_b + pid_s * stride_o_s + pid_h * stride_o_h
    
    m_prev = -1e30
    l_prev = 0.0
    
    # Output accumulators
    acc_0 = tl.zeros([TILE_D], dtype=tl.float32)
    acc_1 = tl.zeros([TILE_D], dtype=tl.float32)
    acc_2 = tl.zeros([TILE_D], dtype=tl.float32)
    acc_3 = tl.zeros([TILE_D], dtype=tl.float32)
    acc_4 = tl.zeros([TILE_D], dtype=tl.float32)
    acc_5 = tl.zeros([TILE_D], dtype=tl.float32)
    acc_6 = tl.zeros([TILE_D], dtype=tl.float32)
    acc_7 = tl.zeros([TILE_D], dtype=tl.float32)
    
    for t_block in range(0, topk, BLOCK_TOPK):
        t_offs = t_block + tl.arange(0, BLOCK_TOPK)
        t_mask = t_offs < topk
        
        indices = tl.load(idx_base + t_offs, mask=t_mask, other=-1)
        valid = (indices >= 0) & (indices < s_kv)
        safe_idx = tl.where(valid, indices, 0)
        
        # Compute scores
        scores = tl.zeros([BLOCK_TOPK], dtype=tl.float32)
        
        for d_tile in range(9):  # 576 / 64
            d_offs = d_tile * TILE_D + tl.arange(0, TILE_D)
            d_mask = d_offs < 576
            
            q_tile = tl.load(q_base + d_offs, mask=d_mask, other=0.0).to(tl.float32)
            k_ptrs = kv_base + safe_idx[:, None] * stride_kv_s + d_offs[None, :]
            k_tiles = tl.load(k_ptrs, mask=valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            
            scores += tl.sum(k_tiles * q_tile[None, :], axis=1)
        
        scores = scores * sm_scale_log2
        scores = tl.where(valid, scores, -1e30)
        
        m_cur = tl.max(scores)
        m_new = tl.maximum(m_prev, m_cur)
        alpha = tl.exp2(m_prev - m_new)
        p = tl.exp2(scores - m_new)
        l_cur = tl.sum(p)
        l_new = l_prev * alpha + l_cur
        
        m_prev = m_new
        l_prev = l_new
        
        acc_0 = acc_0 * alpha
        acc_1 = acc_1 * alpha
        acc_2 = acc_2 * alpha
        acc_3 = acc_3 * alpha
        acc_4 = acc_4 * alpha
        acc_5 = acc_5 * alpha
        acc_6 = acc_6 * alpha
        acc_7 = acc_7 * alpha
        
        # Accumulate V
        d_offs = tl.arange(0, TILE_D)
        v_ptrs = kv_base + safe_idx[:, None] * stride_kv_s + d_offs[None, :]
        v_tiles = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
        acc_0 += tl.sum(p[:, None] * v_tiles, axis=0)
        
        d_offs = 64 + tl.arange(0, TILE_D)
        v_ptrs = kv_base + safe_idx[:, None] * stride_kv_s + d_offs[None, :]
        v_tiles = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
        acc_1 += tl.sum(p[:, None] * v_tiles, axis=0)
        
        d_offs = 128 + tl.arange(0, TILE_D)
        v_ptrs = kv_base + safe_idx[:, None] * stride_kv_s + d_offs[None, :]
        v_tiles = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
        acc_2 += tl.sum(p[:, None] * v_tiles, axis=0)
        
        d_offs = 192 + tl.arange(0, TILE_D)
        v_ptrs = kv_base + safe_idx[:, None] * stride_kv_s + d_offs[None, :]
        v_tiles = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
        acc_3 += tl.sum(p[:, None] * v_tiles, axis=0)
        
        d_offs = 256 + tl.arange(0, TILE_D)
        v_ptrs = kv_base + safe_idx[:, None] * stride_kv_s + d_offs[None, :]
        v_tiles = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
        acc_4 += tl.sum(p[:, None] * v_tiles, axis=0)
        
        d_offs = 320 + tl.arange(0, TILE_D)
        v_ptrs = kv_base + safe_idx[:, None] * stride_kv_s + d_offs[None, :]
        v_tiles = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
        acc_5 += tl.sum(p[:, None] * v_tiles, axis=0)
        
        d_offs = 384 + tl.arange(0, TILE_D)
        v_ptrs = kv_base + safe_idx[:, None] * stride_kv_s + d_offs[None, :]
        v_tiles = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
        acc_6 += tl.sum(p[:, None] * v_tiles, axis=0)
        
        d_offs = 448 + tl.arange(0, TILE_D)
        v_ptrs = kv_base + safe_idx[:, None] * stride_kv_s + d_offs[None, :]
        v_tiles = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
        acc_7 += tl.sum(p[:, None] * v_tiles, axis=0)
    
    safe_l = tl.where(l_prev > 0, l_prev, 1.0)
    
    tl.store(out_base + tl.arange(0, TILE_D), (acc_0 / safe_l).to(tl.bfloat16))
    tl.store(out_base + 64 + tl.arange(0, TILE_D), (acc_1 / safe_l).to(tl.bfloat16))
    tl.store(out_base + 128 + tl.arange(0, TILE_D), (acc_2 / safe_l).to(tl.bfloat16))
    tl.store(out_base + 192 + tl.arange(0, TILE_D), (acc_3 / safe_l).to(tl.bfloat16))
    tl.store(out_base + 256 + tl.arange(0, TILE_D), (acc_4 / safe_l).to(tl.bfloat16))
    tl.store(out_base + 320 + tl.arange(0, TILE_D), (acc_5 / safe_l).to(tl.bfloat16))
    tl.store(out_base + 384 + tl.arange(0, TILE_D), (acc_6 / safe_l).to(tl.bfloat16))
    tl.store(out_base + 448 + tl.arange(0, TILE_D), (acc_7 / safe_l).to(tl.bfloat16))
    
    max_logits = tl.where(l_prev > 0, m_prev, -float("inf"))
    lse = tl.where(l_prev > 0, tl.log2(safe_l) + m_prev, -float("inf"))
    
    # Store max_logits and lse (with batch offset)
    ml_base = MaxLogits_ptr + pid_b * s_q * h_q + pid_s * h_q
    lse_base = LSE_ptr + pid_b * s_q * h_q + pid_s * h_q
    tl.store(ml_base + pid_h, max_logits)
    tl.store(lse_base + pid_h, lse)


def triton_flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Triton-based sparse attention prefill kernel for AMD GPUs (MI355, CDNA4).
    
    Following tilelang convention:
    - dtype (storage): bfloat16 for Q, KV, Output
    - accum_dtype (computation): float32 for accumulators
    
    Args:
        q: [batch, s_q, h_q, d_qk], bfloat16
        kv: [batch, s_kv, h_kv, d_qk], bfloat16
        indices: [batch, s_q, h_kv, topk], int32. Invalid indices should be set to -1 or >= s_kv
        sm_scale: float
        d_v: The dimension of value vectors. Default 512.
    
    Returns:
        (output, max_logits, lse)
        - output: [batch, s_q, h_q, d_v], bfloat16
        - max_logits: [batch, s_q, h_q], float
        - lse: [batch, s_q, h_q], float, 2-based log-sum-exp
    """
    assert q.dtype == torch.bfloat16, f"q must be bfloat16, got {q.dtype}"
    assert kv.dtype == torch.bfloat16, f"kv must be bfloat16, got {kv.dtype}"
    assert indices.dtype == torch.int32, f"indices must be int32, got {indices.dtype}"
    
    assert q.dim() == 4, f"q must be 4D [batch, s_q, h_q, d_qk], got {q.dim()}D"
    assert kv.dim() == 4, f"kv must be 4D [batch, s_kv, h_kv, d_qk], got {kv.dim()}D"
    assert indices.dim() == 4, f"indices must be 4D [batch, s_q, h_kv, topk], got {indices.dim()}D"
    
    batch, s_q, h_q, d_qk = q.shape
    _, s_kv, h_kv, _ = kv.shape
    _, _, _, topk = indices.shape
    
    assert h_kv == 1, "Currently only h_kv=1 (MQA) is supported"
    assert d_qk == 576, f"d_qk must be 576, got {d_qk}"
    assert d_v == 512, f"d_v must be 512, got {d_v}"
    
    # Flatten indices: [batch, s_q, h_kv, topk] -> [batch, s_q, topk]
    indices_flat = indices.squeeze(2).contiguous()
    
    # Allocate outputs
    out = torch.empty((batch, s_q, h_q, d_v), dtype=torch.bfloat16, device=q.device)
    max_logits = torch.empty((batch, s_q, h_q), dtype=torch.float32, device=q.device)
    lse = torch.empty((batch, s_q, h_q), dtype=torch.float32, device=q.device)
    
    sm_scale_log2 = sm_scale * LOG2E
    
    # Use optimized kernel for large topk (common case with TP parallelism)
    # Note: h_q can be small (e.g., 16) when using tensor parallelism
    if topk >= 256:
        # Grid: (batch, s_q, num_head_blocks)
        grid = lambda meta: (batch, s_q, (h_q + meta['BLOCK_H'] - 1) // meta['BLOCK_H'])
        _sparse_attention_fwd_kernel_optimized[grid](
            q, kv, indices_flat, out, max_logits, lse,
            batch, s_q, s_kv, h_q, topk,
            sm_scale_log2,
            # Q strides: [batch, s_q, h_q, d_qk]
            q.stride(0), q.stride(1), q.stride(2),
            # KV strides: [batch, s_kv, h_kv, d_qk]
            kv.stride(0), kv.stride(1),
            # indices strides: [batch, s_q, topk]
            indices_flat.stride(0), indices_flat.stride(1),
            # out strides: [batch, s_q, h_q, d_v]
            out.stride(0), out.stride(1), out.stride(2),
        )
    else:
        BLOCK_TOPK = 64 if topk >= 64 else 32
        TILE_D = 64
        # Grid: (batch, s_q, h_q)
        grid = (batch, s_q, h_q)
        _sparse_attention_fwd_kernel_simple[grid](
            q, kv, indices_flat, out, max_logits, lse,
            batch, s_q, s_kv, h_q, topk,
            sm_scale_log2,
            # Q strides: [batch, s_q, h_q, d_qk]
            q.stride(0), q.stride(1), q.stride(2),
            # KV strides: [batch, s_kv, h_kv, d_qk]
            kv.stride(0), kv.stride(1),
            # indices strides: [batch, s_q, topk]
            indices_flat.stride(0), indices_flat.stride(1),
            # out strides: [batch, s_q, h_q, d_v]
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_TOPK, TILE_D,
        )
    
    return out, max_logits, lse

