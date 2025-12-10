#!/usr/bin/env python3
"""
Benchmark script for Triton SparseMLA kernel on AMD MI355 GPUs.

This script compares the performance of:
1. Triton-based SparseMLA kernel
2. TileLang-based SparseMLA kernel (baseline)

Usage:
    python bench_triton_sparse_mla.py [--seq-len SEQ_LEN] [--topk TOPK]

Example:
    python bench_triton_sparse_mla.py --seq-len 8192 --topk 2048
    python bench_triton_sparse_mla.py --all  # Run all configurations
"""

import argparse
import math
import os
import sys
import time
from typing import Tuple

import torch

# Add paths for imports (relative to this script's location)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SGL_KERNEL_PYTHON = os.path.join(_SCRIPT_DIR, '..', 'python')
_SGL_KERNEL_DIR = os.path.join(_SCRIPT_DIR, '..', 'python', 'sgl_kernel')
_SGLANG_PYTHON = os.path.join(_SCRIPT_DIR, '..', '..', '..', 'python')

if _SGL_KERNEL_DIR not in sys.path:
    sys.path.insert(0, _SGL_KERNEL_DIR)
if _SGL_KERNEL_PYTHON not in sys.path:
    sys.path.insert(0, _SGL_KERNEL_PYTHON)
if _SGLANG_PYTHON not in sys.path:
    sys.path.insert(0, _SGLANG_PYTHON)


def get_triton_kernel():
    """Import Triton kernel directly (avoid loading sgl_kernel which requires common_ops)."""
    from triton_flash_mla_sparse import triton_flash_mla_sparse_fwd
    return triton_flash_mla_sparse_fwd


def get_tilelang_kernel():
    """Import TileLang kernel that supports 4D batch input."""
    try:
        from sglang.srt.layers.attention.nsa.tilelang_kernel import (
            sparse_attention_fwd_kernel_v1,
            _is_hip,
        )
        return sparse_attention_fwd_kernel_v1, _is_hip
    except ImportError:
        print("Warning: TileLang kernel not available")
        return None, None


def tilelang_sparse_fwd_4d(kernel_factory, q, kv, indices, sm_scale, d_v=512):
    """TileLang kernel wrapper that accepts 4D batch input directly."""
    # q: [batch, seq_q, num_heads, d_qk]
    # kv: [batch, seq_kv, h_kv, d_qk]  
    # indices: [batch, seq_q, h_kv, topk]
    num_heads = q.shape[2]
    dim = q.shape[3]
    tail_dim = dim - d_v
    topk = indices.shape[-1]
    
    # Create kernel with same parameters
    kernel = kernel_factory(
        num_heads, d_v, tail_dim, topk, sm_scale=sm_scale, num_stages=1
    )
    # Directly pass 4D tensors
    return kernel(q, kv, indices)


def generate_test_data(
    batch: int,
    seq_q: int,
    seq_kv: int,
    num_heads: int = 128,
    d_qk: int = 576,
    topk: int = 2048,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Generate test data for benchmarking with batch dimension."""
    torch.manual_seed(42)
    
    # 4D tensors with batch dimension
    q = torch.randn(batch, seq_q, num_heads, d_qk, dtype=torch.bfloat16, device=device) / 10
    kv = torch.randn(batch, seq_kv, 1, d_qk, dtype=torch.bfloat16, device=device) / 10
    indices = torch.randint(0, seq_kv, (batch, seq_q, 1, topk), dtype=torch.int32, device=device)
    sm_scale = 1.0 / math.sqrt(d_qk)
    
    return q, kv, indices, sm_scale


def benchmark_kernel(
    kernel_fn,
    args: tuple,
    kwargs: dict = None,
    warmup: int = 5,
    repeat: int = 20,
) -> float:
    """Benchmark a kernel function and return average time in ms."""
    if kwargs is None:
        kwargs = {}
    
    # Warmup
    for _ in range(warmup):
        _ = kernel_fn(*args, **kwargs)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(repeat):
        _ = kernel_fn(*args, **kwargs)
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / repeat * 1000


def run_benchmark(
    seq_len: int,
    topk: int = 2048,
    batch: int = 1,
    num_heads: int = 128,
    d_qk: int = 576,
    d_v: int = 512,
    check_correct: bool = True,
    verbose: bool = True,
) -> dict:
    """Run benchmark for a single configuration using TileLang as baseline."""
    device = 'cuda'
    
    # Generate data with batch dimension (4D tensors)
    q, kv, indices, sm_scale = generate_test_data(
        batch=batch,
        seq_q=seq_len,
        seq_kv=seq_len,
        num_heads=num_heads,
        d_qk=d_qk,
        topk=topk,
        device=device,
    )
    
    # Get kernels
    triton_fwd = get_triton_kernel()
    tilelang_kernel_factory, _is_hip = get_tilelang_kernel()
    
    results = {
        'seq_len': seq_len,
        'topk': topk,
        'batch': batch,
        'num_heads': num_heads,
    }
    
    if verbose:
        print(f"\nBenchmarking batch={batch}, seq_len={seq_len}, topk={topk}")
        print("-" * 60)
    
    # Benchmark Triton (uses 4D tensors natively)
    triton_time = benchmark_kernel(
        triton_fwd,
        (q, kv, indices, sm_scale),
    )
    results['triton_ms'] = triton_time
    
    if verbose:
        print(f"Triton:   {triton_time:8.3f} ms")
    
    # Benchmark TileLang (also uses 4D tensors - fair comparison)
    if tilelang_kernel_factory is not None:
        tilelang_time = benchmark_kernel(
            tilelang_sparse_fwd_4d,
            (tilelang_kernel_factory, q, kv, indices, sm_scale),
            kwargs={'d_v': d_v},
        )
        results['tilelang_ms'] = tilelang_time
        results['speedup'] = tilelang_time / triton_time
        
        if verbose:
            print(f"TileLang: {tilelang_time:8.3f} ms")
            print(f"Speedup:  {results['speedup']:8.2f}x (Triton vs TileLang)")
        
        # Check correctness against TileLang baseline
        if check_correct:
            triton_out, _, _ = triton_fwd(q, kv, indices, sm_scale)
            tilelang_out = tilelang_sparse_fwd_4d(tilelang_kernel_factory, q, kv, indices, sm_scale, d_v)
            max_diff = (triton_out.float() - tilelang_out.float()).abs().max().item()
            passed = max_diff < 1e-3
            results['correct'] = passed
            results['max_diff'] = max_diff
            
            if verbose:
                status = "PASS" if passed else "FAIL"
                print(f"Correct:  {status} (max_diff={max_diff:.2e})")
    
    return results


def run_all_benchmarks():
    """Run benchmarks for all configurations."""
    # Only use topk=2048 which TileLang supports
    configs = [
        # (seq_len, topk)
        (4096, 2048),
        (8192, 2048),
        (16384, 2048),
    ]
    
    # Test both batch=1 and batch=4
    batch_sizes = [1, 4]
    
    print("=" * 80)
    print("Triton SparseMLA Benchmark on AMD MI355")
    print("=" * 80)
    
    all_results = []
    for batch in batch_sizes:
        print(f"\n{'='*80}")
        print(f"Batch Size = {batch}")
        print(f"{'='*80}")
        
        for seq_len, topk in configs:
            torch.cuda.empty_cache()
            try:
                results = run_benchmark(seq_len, topk, batch=batch, verbose=True)
                all_results.append(results)
            except Exception as e:
                print(f"Error for batch={batch}, seq_len={seq_len}, topk={topk}: {e}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'batch':>6} {'seq_len':>10} {'topk':>8} {'Triton(ms)':>12} {'TileLang(ms)':>14} {'Speedup':>10} {'Status':>8}")
    print("-" * 80)
    
    for r in all_results:
        tilelang_str = f"{r.get('tilelang_ms', 0):12.2f}" if 'tilelang_ms' in r else "N/A".rjust(12)
        speedup_str = f"{r.get('speedup', 0):10.2f}x" if 'speedup' in r else "N/A".rjust(10)
        status = "PASS" if r.get('correct', False) else "FAIL"
        print(f"{r['batch']:>6} {r['seq_len']:>10} {r['topk']:>8} {r['triton_ms']:>12.2f} {tilelang_str:>14} {speedup_str:>10} {status:>8}")


def run_profiling_benchmark(seq_len: int = 8192, topk: int = 2048, batch: int = 1):
    """Run benchmark suitable for rocprof profiling."""
    print(f"Running profiling benchmark: batch={batch}, seq_len={seq_len}, topk={topk}")
    print("Use with: rocprof -i metrics.txt python bench_triton_sparse_mla.py --profile")
    
    device = 'cuda'
    q, kv, indices, sm_scale = generate_test_data(
        batch=batch,
        seq_q=seq_len,
        seq_kv=seq_len,
        topk=topk,
        device=device,
    )
    
    triton_fwd, _ = get_triton_kernel()
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        _ = triton_fwd(q, kv, indices, sm_scale)
    torch.cuda.synchronize()
    
    # Profile runs
    print("Running profiled iterations...")
    for _ in range(5):
        _ = triton_fwd(q, kv, indices, sm_scale)
    torch.cuda.synchronize()
    
    print("Profiling complete!")


def main():
    parser = argparse.ArgumentParser(description='Benchmark Triton SparseMLA kernel')
    parser.add_argument('--seq-len', type=int, default=8192, help='Sequence length')
    parser.add_argument('--topk', type=int, default=2048, help='Top-k value')
    parser.add_argument('--all', action='store_true', help='Run all configurations')
    parser.add_argument('--profile', action='store_true', help='Run in profiling mode')
    parser.add_argument('--no-check', action='store_true', help='Skip correctness check')
    
    args = parser.parse_args()
    
    if args.profile:
        run_profiling_benchmark(args.seq_len, args.topk)
    elif args.all:
        run_all_benchmarks()
    else:
        run_benchmark(
            seq_len=args.seq_len,
            topk=args.topk,
            check_correct=not args.no_check,
            verbose=True,
        )


if __name__ == '__main__':
    main()

