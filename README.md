# Radeon (RDNA) CI for SGLang

| line | gfx1201 (RDNA4) and gfx1100 (RDNA3), one job each |
|---|---|
| upstream `main` | [![main](https://github.com/vivienfanghuagood/sglang/actions/workflows/rdna-ci-main.yml/badge.svg?branch=rdna-ci)](https://github.com/vivienfanghuagood/sglang/actions/workflows/rdna-ci-main.yml) |
| upstream `amd_march` | [![amd_march](https://github.com/vivienfanghuagood/sglang/actions/workflows/rdna-ci-amd-march.yml/badge.svg?branch=rdna-ci)](https://github.com/vivienfanghuagood/sglang/actions/workflows/rdna-ci-amd-march.yml) |
| `amd_march` + open RDNA PRs (separate, not upstream) | [![amd_march + PRs](https://github.com/vivienfanghuagood/sglang/actions/workflows/rdna-ci-amd-march-plus-prs.yml/badge.svg?branch=rdna-ci)](https://github.com/vivienfanghuagood/sglang/actions/workflows/rdna-ci-amd-march-plus-prs.yml) |

A badge is red when any step of that line fails: the kernel build, a kernel test, the install, a server failing to start, or no gsm8k score. A red `main` therefore means "RDNA does not work on upstream `main` today", not that the CI is broken.

This branch only holds a daily test of upstream [sgl-project/sglang](https://github.com/sgl-project/sglang) on Radeon GPUs, run on self-hosted runners. It's the RDNA CI shape proposed in sgl-project/sglang#46900. Results are on the Actions page of each workflow: each job's summary has a table, and its `result-*` artifact has the JSON and log tails.

## What runs, every day (dispatched around 18:15, 19:15, 20:15 UTC)

| arch | hardware class |
|---|---|
| gfx1201 | RDNA4 |
| gfx1100 | RDNA3, W7900-class. This host has no general internet access, so the job runs in a pre-provisioned ROCm 7.2 / torch 2.13 environment (PyPI only) rather than the stock image, and is skipped when no GPU is idle. |

Three lines are tested on each arch, reported separately:

| line | tree |
|---|---|
| `main` | upstream `main` as is |
| `amd_march` | upstream `amd_march` as is |
| `amd_march-plus-prs` | `amd_march` with the open RDNA fixes layered on top (#46910, #46914, #46915, #46967, #46909, #47299). Shows what the branch looks like once they merge. |

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

- The workflows are triggered only by `workflow_dispatch`, sent at fixed times by the runner owner (GitHub cron schedules on this fork fired hours late). There is no `pull_request`, `pull_request_target` or `issue_comment` trigger, so code from outside this fork never reaches the runners.
- `permissions: contents: read`, and no secrets.
- Runners are ephemeral (one job per registration) and registered to this repository only.
- Each job uses a throwaway container. Models are mounted read-only, and core dumps are off.
- Logs and artifacts are scrubbed of host-specific details before upload.
