# PR: Add triton_sparse backend for NSA (Native Sparse Attention)

## Summary
This PR adds a Triton-based sparse attention kernel (`triton_sparse`) as an alternative backend for NSA in sglang. The implementation is optimized for AMD GPUs (MI355, CDNA4) and provides:
- **1.3x speedup** in offline kernel benchmarks
- **1.04-1.10x TTFT speedup** for 8K-16K input in end-to-end serving

Additionally, this PR includes a **decode optimization** that batches topk calls in `nsa_indexer.py`:
- Reduces topk calls from 976 to 61 per decode step (16x reduction)
- **Up to 17% OTPS improvement** in end-to-end serving

## Performance Results

### Offline Kernel Benchmark (TP=8, h_q=16, topk=2048)

| s_q   | TileLang | Triton  | Speedup |
|-------|----------|---------|---------|
| 1024  | 1.41 ms  | 1.09 ms | **1.29x** |
| 2048  | 2.80 ms  | 2.15 ms | **1.30x** |
| 4096  | 5.57 ms  | 4.29 ms | **1.30x** |
| 8192  | 11.13 ms | 8.57 ms | **1.30x** |
| 16384 | 22.29 ms | 17.24 ms| **1.29x** |

### End-to-End Serving Benchmark (DeepSeek-V3.2-Exp, TP=8, 100 prompts, 128 output tokens, rate=32)

**Time to First Token (TTFT) - Lower is better:**

| Input Length | TileLang TTFT | Triton TTFT | Speedup |
|--------------|---------------|-------------|---------|
| 8K(rate=32)            | 32633 ms      | 31273 ms    | **1.04x** |
| 16K(rate=32)          | 35650 ms      | 32546 ms    | **1.10x** |
| 32K(rate=2)          | 27091 ms     | 22594 ms   | **1.19x**  |

**Output Token Throughput (OTPS) - Higher is better:**

| Input Length | TileLang OTPS | Triton OTPS | Triton+BatchedTopK | Speedup vs TileLang |
|--------------|---------------|-------------|---------------------|---------------------|
| 8K(rate=32)  | 32.07 tok/s   | 32.90 tok/s | **37.53 tok/s**     | **1.17x** |
| 16K(rate=32) | 29.64 tok/s   | 30.67 tok/s | **34.12 tok/s**     | **1.15x** |

**Inter-Token Latency (ITL) - Lower is better:**

| Input Length | TileLang ITL | Triton ITL | Triton+BatchedTopK | Improvement |
|--------------|--------------|------------|---------------------|-------------|
| 8K(rate=32)  | 1826 ms      | 1788 ms    | **1530 ms**         | **1.19x** |
| 16K(rate=32) | 2067 ms      | 1997 ms    | **1782 ms**         | **1.16x** |

Note: Performance tested with `SGLANG_NSA_FUSE_TOPK=false`.

## Decode Optimization: Batched TopK

Profiling showed `aten::topk` was the largest decode bottleneck (30.5% of CUDA time, 976 calls/step). 

**Solution**: Batch topk calls per layer instead of per batch item.
- **Before**: 976 topk calls/step (61 layers × 16 batch)
- **After**: 61 topk calls/step (1 batched call per layer)
- **Result**: ~16x fewer kernel launches, 15-19% ITL improvement

## How to Reproduce

### 1. Clone the repository

```bash
git clone https://github.com/vivienfanghuagood/sglang.git
cd sglang
git checkout dsk_v32
```

### 2. Install dependencies

```bash
pip install -e "python[all]"
pip install -e "sgl-kernel"
```

### 3. Start the server with TileLang backend

```bash
export SGLANG_NSA_FUSE_TOPK=false
export SGLANG_NSA_KV_CACHE_STORE_FP8=false
export SGLANG_NSA_USE_REAL_INDEXER=true
export SGLANG_NSA_USE_TILELANG_PREFILL=true

python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.2-Exp \
  --disable-cuda-graph \
  --tp 8 \
  --mem-fraction-static 0.85 \
  --page-size 64 \
  --nsa-prefill tilelang \
  --nsa-decode tilelang \
  --port 30000
```

### 4. Run benchmark for TileLang

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model deepseek-ai/DeepSeek-V3.2-Exp \
  --dataset-name random \
  --random-input-len 16384 \
  --random-output-len 128 \
  --num-prompts 10 \
  --request-rate inf
```

### 5. Start the server with Triton backend

```bash
# Stop previous server first
pkill -f sglang

export SGLANG_NSA_FUSE_TOPK=false
export SGLANG_NSA_KV_CACHE_STORE_FP8=false
export SGLANG_NSA_USE_REAL_INDEXER=true
export SGLANG_NSA_USE_TILELANG_PREFILL=true

python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.2-Exp \
  --disable-cuda-graph \
  --tp 8 \
  --mem-fraction-static 0.85 \
  --page-size 64 \
  --nsa-prefill triton_sparse \
  --nsa-decode triton_sparse \
  --port 30000
```

### 6. Run benchmark for Triton

```bash
# Run twice - first run includes autotune compilation
python -m sglang.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model deepseek-ai/DeepSeek-V3.2-Exp \
  --dataset-name random \
  --random-input-len 16384 \
  --random-output-len 128 \
  --num-prompts 10 \
  --request-rate inf

# Second run shows actual performance (autotune cached)
python -m sglang.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model deepseek-ai/DeepSeek-V3.2-Exp \
  --dataset-name random \
  --random-input-len 16384 \
  --random-output-len 128 \
  --num-prompts 10 \
  --request-rate inf
```

## Files Changed

### 1. NEW: `sgl-kernel/python/sgl_kernel/triton_flash_mla_sparse.py`
- Triton-based sparse attention kernel implementation
- Online softmax with flash attention algorithm
- Optimized autotune configs for different head sizes (h_q=16 to 128+)
- Added BLOCK_TOPK=128 configs for better performance on large topk

### 2. MODIFIED: `sgl-kernel/python/sgl_kernel/__init__.py`
```python
from sgl_kernel.triton_flash_mla_sparse import triton_flash_mla_sparse_fwd
```

### 3. MODIFIED: `python/sglang/srt/layers/attention/nsa_backend.py`
- Add `"triton_sparse"` to `_NSA_IMPL_T` type alias
- Add `_forward_triton_sparse()` method
- Add dispatch branches for `triton_sparse` in `forward_extend` and `forward_decode`

### 4. MODIFIED: `python/sglang/srt/server_args.py`
```python
NSA_CHOICES = ["flashmla_prefill", "flashmla_decode", "fa3", "tilelang", "aiter", "triton_sparse"]
```

### 5. BUGFIX: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`
```python
# Before (bug - returns 4D tensor instead of 3D):
return kernel(q.unsqueeze(0), kv.unsqueeze(0), indices.unsqueeze(0))

# After (fixed - returns correct 3D tensor):
out = kernel(q.unsqueeze(0), kv.unsqueeze(0), indices.unsqueeze(0))
return out.squeeze(0)  # Remove batch dimension to return [s_q, h_q, d_v]
```

### 6. OPTIMIZED: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`
- Added `_forward_indexer_batched_topk()` method for decode mode
- Batches topk calls: 976 → 61 calls per decode step
- Expected savings: ~72ms per decode step (~14x reduction in topk overhead)

```python
def _forward_indexer_batched_topk(self, ...):
    """
    Optimized decode path: collect scores then do ONE batched topk.
    Reduces topk calls from batch_size to 1 per layer.
    """
    # Pre-allocate scores tensor
    all_scores = torch.full((batch_size, max_seq_len), -inf, ...)
    
    # Compute scores (fp8_index per-item due to kernel constraint)
    for i in range(batch_size):
        all_scores[i, :seq_len] = fp8_index(...).view(-1)[:seq_len]
    
    # Single batched topk call instead of batch_size calls
    topk_indices = all_scores.topk(topk_actual, dim=-1)[1]
```

## Key Implementation Details

1. **Kernel Selection**: Uses optimized kernel for `topk >= 256` (removed h_q >= 64 restriction to support TP parallelism)

2. **Autotune Key**: Uses only `['topk']` to avoid kernel recompilation when s_q or h_q varies

3. **Memory Efficiency**: No `.contiguous()` calls - Triton uses strides for memory access

4. **Tensor Handling**: Matches tilelang's approach by adding batch dimension via `unsqueeze(0)` without memory copy

5. **Autotune Configs**:
   - BLOCK_TOPK: 64, 128
   - BLOCK_H: 16, 32, 64, 128
   - num_stages: 1 (optimized for AMD)
   - num_warps: 4, 8

6. **Batched TopK**: Collects scores from all batch items, then performs single batched topk per layer

## Environment Variables

| Variable | Value | Description |
|----------|-------|-------------|
| `SGLANG_NSA_FUSE_TOPK` | `false` | **Required for correct inference accuracy**. Disables fused topk. |
| `SGLANG_NSA_KV_CACHE_STORE_FP8` | `false` | Disables FP8 KV cache storage for better precision. |
| `SGLANG_NSA_USE_REAL_INDEXER` | `true` | Uses real indexer for NSA. |
| `SGLANG_NSA_USE_TILELANG_PREFILL` | `true` | Enables TileLang prefill optimization. |

## Technical Notes

### Decode Stage Bottleneck Analysis

Profiling with PyTorch profiler on AMD HIP revealed:
1. **aten::topk**: 30.5% of CUDA time (74.7ms, 976 calls)
2. **MoE Triton kernels**: ~18% of CUDA time
3. **Sparse Attention**: ~6.5% of CUDA time

The batched topk optimization targets the largest single bottleneck.

### fp8_index Kernel Limitation

The tilelang `fp8_index` kernel requires `h >= 32` due to warp configuration constraints. With TP=8, `h = 64/8 = 8`, which prevents batching the fp8_index call. This is a known limitation that could be addressed in future tilelang kernel updates.
