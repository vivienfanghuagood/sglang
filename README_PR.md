# PR: Add triton_sparse backend for NSA (Native Sparse Attention)

## Summary
This PR adds a Triton-based sparse attention kernel (`triton_sparse`) as an alternative backend for NSA in sglang. The implementation is optimized for AMD GPUs (MI355, CDNA4) and provides:
- **1.3x speedup** in offline kernel benchmarks
- **1.04-1.10x TTFT speedup** for 8K-16K input in end-to-end serving
- **1.03x output throughput** improvement for 8K-16K input

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

| Input Length | TileLang OTPS | Triton OTPS | Speedup |
|--------------|---------------|-------------|---------|
| 8K(rate=32)           | 32.07 tok/s   | 32.90 tok/s | **1.03x** |
| 16K(rate=32)          | 29.64 tok/s   | 30.67 tok/s | **1.03x** |
| 32K(rate=2)          | 12.07 tok/s   | 13.28 tok/s | **1.1x**   |

**Inter-Token Latency (ITL) - Lower is better:**

| Input Length | TileLang ITL | Triton ITL | Improvement |
|--------------|--------------|------------|-------------|
| 8K(rate=32)           | 1826 ms      | 1788 ms    | **1.02x** |
| 16K(rate=32)          | 2067 ms      | 1997 ms    | **1.04x** |
| 32K(rate=2)          | 905 ms      | 835 ms    | **1.08x**   |

Note: Performance tested with `SGLANG_NSA_FUSE_TOPK=false`.

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

## Environment Variables

| Variable | Value | Description |
|----------|-------|-------------|
| `SGLANG_NSA_FUSE_TOPK` | `false` | **Required for correct inference accuracy**. Disables fused topk. |
| `SGLANG_NSA_KV_CACHE_STORE_FP8` | `false` | Disables FP8 KV cache storage for better precision. |
| `SGLANG_NSA_USE_REAL_INDEXER` | `true` | Uses real indexer for NSA. |
| `SGLANG_NSA_USE_TILELANG_PREFILL` | `true` | Enables TileLang prefill optimization. |

