## Beating a Hand‑Tuned TileLang Kernel on AMD MI355 with Triton SparseMLA

*How we turned a 40% slower Triton kernel into a 1.2× faster one by fixing data reuse*

---

### 1. TL;DR

We optimized a Triton implementation of **Sparse Multi‑head Latent Attention (SparseMLA)** on **AMD MI355 (CDNA4)** and made it **1.2× faster than a hand‑tuned TileLang kernel**.

- **Workload**: long‑context prefill, `seq = 8K–16K`, `topk = 2048`, `num_heads = 128`, `d_qk = 576`, `d_v = 512`
- **Baseline (TileLang)**: 30.8 ms @ 8K, 62.5 ms @ 16K
- **Triton (before)**: 50.3 ms / 100.8 ms → **~40% slower**
- **Triton (after)**: 27.5 ms / 51.9 ms → **up to 1.2× faster than TileLang**

The key insight: **we were not reusing K/V data across heads enough**. Once we fixed the **heads‑per‑block** configuration and reused Q tiles properly, performance flipped.

---

### 2. Background: SparseMLA and the Target Workload

SparseMLA is a core component of DeepSeek’s **Native Sparse Attention (NSA)**. Instead of attending to every position, it focuses on a pre‑selected **top‑k** subset:

```python
# Dense attention: O(seq_q × seq_kv)
scores = Q @ K.T

# Sparse attention: O(seq_q × topk)
indices = select_topk(Q, K)       # [seq_q, topk]
focused_kv = gather(KV, indices)  # [seq_q, topk, d]
scores = Q @ focused_kv.T
```

Our typical prefill setting:

- **Sequence length**: `seq_q = seq_kv = 8,192 – 16,384`
- **Heads**: `num_heads = 128`
- **Dimensions**: `d_qk = 576`, `d_v = 512`
- **Sparsity**: `topk = 2,048`

This is exactly the kind of memory‑bound, structured workload where Triton should shine—if we get the tiling right.

---

### 3. Initial Triton Kernel: Clean Design, Disappointing Performance

We started from a fairly standard Triton attention kernel:

- **bfloat16** storage, **float32** accumulators
- **Online softmax** to avoid materializing the full attention matrix
- **Autotune** over reasonable block sizes

The simplified structure looked like this:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_TOPK': 64, 'BLOCK_H': 16, 'TILE_D': 64}, num_warps=8),
        triton.Config({'BLOCK_TOPK': 64, 'BLOCK_H': 32, 'TILE_D': 64}, num_warps=8),
    ],
    key=['s_q', 'topk', 'h_q'],
)
@triton.jit
def sparse_mla_kernel(..., BLOCK_TOPK: tl.constexpr, BLOCK_H: tl.constexpr):
    pid_q = tl.program_id(0)      # query index
    pid_h_blk = tl.program_id(1)  # head block index
    # Each program processes BLOCK_H heads for one query
    # Loop over top-k blocks and do online softmax + P @ V
```

On paper this looked fine, but the performance was not:

| Config                     | Triton (before) | TileLang | Triton / TileLang |
|---------------------------|-----------------|----------|-------------------|
| `seq=8192, topk=2048`     | 50.3 ms         | 30.8 ms  | 0.61×             |
| `seq=16384, topk=2048`    | 100.8 ms        | 62.5 ms  | 0.62×             |

So where was the 40% gap coming from?

---

### 4. Profiling on AMD MI355: What the Counters Told Us

Instead of guessing, we used **`rocprof`** to capture a few key hardware counters for both kernels:

- `SQ_WAVES` – total waves launched (like “warps” on NVIDIA)
- `SQ_INSTS_VALU` – vector ALU instructions
- `SQ_INSTS_SMEM` – scalar memory instructions

Comparing Triton vs TileLang revealed:

- **4× more grid blocks and waves** in Triton
- **~1.5× more ALU instructions**
- **~3× more scalar memory instructions**

In words:

> Our Triton kernel was doing **the same logical work**, but with **many more waves** and **more instructions per wave**.

That strongly suggested a **tiling / data‑reuse** problem, not a “Triton vs TileLang” problem.

---

### 5. Root Cause: Too Few Heads per Block → Poor K/V Reuse

The difference boiled down to **how many attention heads one block processes**.

- **TileLang kernel**: processes **64 heads per block**
- **Our Triton kernel**: autotune was choosing **16 heads per block (`BLOCK_H = 16`)**

Why does this matter?

Each iteration over top‑k does roughly:

1. Load `BLOCK_TOPK` indices
2. Gather `BLOCK_TOPK` K vectors
3. Compute `Q @ K^T` for `BLOCK_H` heads
4. Gather V vectors
5. Compute `P @ V` for `BLOCK_H` heads

If `BLOCK_H` is small:

- The **same K/V data** is loaded multiple times by different blocks
- You pay for global memory and instructions repeatedly

For our workload (`num_heads = 128`, `BLOCK_TOPK = 64`):

- With **`BLOCK_H = 16`**: 8 head blocks → **4× more K/V loads**
- With **`BLOCK_H = 64`**: 2 head blocks → much better data reuse

This exactly matched what we saw in the counters: **4× more waves**, more instructions, and worse performance.

---

### 6. The Fix: Expand Autotune Space and Reuse Q Tiles

#### 6.1 Increase `BLOCK_H` in Autotune

The first fix was simple: **let Triton try larger `BLOCK_H` values** so that one block can reuse K/V across many heads.

```python
@triton.autotune(
    configs=[
        # Original (too few heads per block)
        triton.Config({'BLOCK_TOPK': 64, 'BLOCK_H': 16,  'TILE_D': 64}, num_warps=8),
        triton.Config({'BLOCK_TOPK': 64, 'BLOCK_H': 32,  'TILE_D': 64}, num_warps=8),

        # New: better K/V reuse
        triton.Config({'BLOCK_TOPK': 64, 'BLOCK_H': 64,  'TILE_D': 64}, num_warps=8),
        triton.Config({'BLOCK_TOPK': 64, 'BLOCK_H': 128, 'TILE_D': 64}, num_warps=8),
    ],
    key=['s_q', 'topk', 'h_q'],
)
```

Once we added these options, autotune naturally picked the larger `BLOCK_H` on MI355, which:

- **Reduced grid size and waves by ~4×**
- **Cut both ALU and memory instructions significantly**

Lesson: **autotune can’t explore tilings you never propose.**

#### 6.2 Hoist Q Loads out of the Top‑k Loop

We also removed unnecessary Q reloads:

- **Before**: Q tiles were loaded **inside** the top‑k loop → reloaded every iteration
- **After**: Q tiles are **loaded once** per (query, head block) and reused across all top‑k blocks

Conceptually:

```python
# Load Q tiles once
q_tiles = [tl.load(Q_ptr + offs_d) for offs_d in d_tiles]

for t_block in range(0, topk, BLOCK_TOPK):
    k_tiles = ...
    # Reuse q_tiles here for all dot products
```

This is a smaller win than fixing `BLOCK_H`, but it further reduced memory traffic.

---

### 7. Final Results: Triton > TileLang

After these changes, we reran the same benchmarks on AMD MI355:

| Config                  | Triton (before) | Triton (after) | TileLang | Triton vs TileLang |
|------------------------|-----------------|----------------|----------|--------------------|
| `seq=8192, topk=2048`  | 50.3 ms         | **27.5 ms**    | 30.8 ms  | **1.12× faster**   |
| `seq=16384, topk=2048` | 100.8 ms        | **51.9 ms**    | 62.6 ms  | **1.21× faster**   |

Hardware counters confirmed the story:

- **Waves**: ~4× fewer than before, now on par with TileLang
- **VALU instructions**: ~4× fewer
- **Memory instructions**: ~2× fewer

Correctness was validated against a PyTorch reference implementation with **max error \< 1e‑3** in bfloat16.

---

### 8. How to Reproduce

If you want to try this yourself on an AMD MI300‑series GPU:

1. **Use the SGLang ROCm image** (already configured with Triton and ROCm 7.0)
2. **Clone and install SGLang** (the kernel lives in `sgl-kernel`)
3. Run the provided **SparseMLA benchmark** in the repo

The key pieces you’ll want to inspect are:

- The **Triton SparseMLA kernel** and its `BLOCK_H` autotune configs
- The **benchmark script** that exercises long‑sequence prefill with `topk = 2048`

---

### 9. Takeaways

1. **The biggest win was not a trick, but data reuse.** Increasing heads per block (`BLOCK_H`) so that K/V are shared across many heads removed massive redundancy.
2. **Autotune is only as good as your search space.** You must explicitly include tilings that match your algorithm’s reuse patterns.
3. **Hardware counters are your compass.** Metrics like waves and instruction counts quickly pointed us to “too many blocks, too little reuse” instead of vague “AMD vs NVIDIA” guesses.
4. **Triton can match and even beat hand‑tuned kernels.** With the right tiling decisions, the higher‑level abstraction is not the bottleneck.

If you’re porting kernels to AMD or writing new Triton kernels, a good rule of thumb is:

> **First design for data reuse (across heads, tiles, and iterations), then let autotune search within those patterns.**

---

### References

- [Triton Language Documentation](https://triton-lang.org/)
- [AMD ROCm Profiling Tools](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/)
- [SGLang Project](https://github.com/sgl-project/sglang)

This work was done as part of optimizing SGLang’s Native Sparse Attention backend for AMD GPUs.
