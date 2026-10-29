# Radeon (RDNA) CI for SGLang

[![rdna-ci](https://github.com/vivienfanghuagood/sglang/actions/workflows/rdna-ci.yml/badge.svg?branch=rdna-ci)](https://github.com/vivienfanghuagood/sglang/actions/workflows/rdna-ci.yml)

This branch only holds a daily test of upstream [sgl-project/sglang](https://github.com/sgl-project/sglang) on Radeon GPUs, run on self-hosted runners. It's the RDNA CI shape proposed in sgl-project/sglang#46900. Results are on the [Actions page](https://github.com/vivienfanghuagood/sglang/actions/workflows/rdna-ci.yml): each job's summary has a table, and its `result-*` artifact has the JSON and log tails.

## What runs, every day at 18:00 UTC

| arch | hardware class |
|---|---|
| gfx1201 | RDNA4 |
| gfx1100 | RDNA3, W7900-class (being added) |

Three lines are tested on each arch, reported separately:

| line | tree |
|---|---|
| `main` | upstream `main` as is |
| `amd_march` | upstream `amd_march` as is |
| `amd_march-plus-prs` | `amd_march` with the open RDNA fixes layered on top (#46910, #46914, #46915, #46967, #46909). Shows what the branch looks like once they merge. |

Each job runs in a fresh stock `rocm/pytorch` container with one GPU and without aiter:

1. Build sgl-kernel for the arch (`AMDGPU_TARGET=<arch> python setup_rocm.py install`).
2. Run the ROCm AOT kernel unit tests: activation, token bitmask, MoE top-k softmax/sigmoid, top-k, dsv4 norm-rope, MoE align. Each file is its own pytest.
3. Install SGLang with the documented source install (`SGLANG_BUILD_RUST_EXTS=none pip install -e "python[all_hip]"`).
4. Serve Qwen/Qwen3-8B with default flags:
   - two requests;
   - `sgl-eval` gsm8k, 100 questions, `enable_thinking=false`, temperature 0.7, top_p 0.8;
   - `bench_serving` random-ids 512/256 at concurrency 1 and 8.
5. Serve allenai/OLMoE-1B-7B-0125-Instruct (MoE) with default flags; `sgl-eval` gsm8k, 50 questions.

A step that fails is reported, and the job carries on with what it can still run.

## Isolation

- The workflow is triggered only by `schedule` and `workflow_dispatch`. There is no `pull_request`, `pull_request_target` or `issue_comment` trigger, so code from outside this fork never reaches the runners.
- `permissions: contents: read`, and no secrets.
- Runners are ephemeral (one job per registration) and registered to this repository only.
- Each job uses a throwaway container. Models are mounted read-only, and core dumps are off.
- Logs and artifacts are scrubbed of host-specific details before upload.
