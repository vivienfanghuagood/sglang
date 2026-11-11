"""Local repro for sglang#49657: HIP custom all-reduce 1-stage on two PCIe GPUs.

Each rank drives the custom all-reduce ops directly (the HIP branch of CustomAllreduce.__init__),
fills fresh random inputs, syncs + barriers, reduces, and compares bitwise against the exact
two-rank sum and against the other rank's output. Also times it against RCCL.
Writes a summary to $CAR_OUT (default /ci/out/car_result.log).
"""

import os
import socket
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

COLS = 5120
ROWS = (8, 16)
ITERS = int(os.environ.get("CAR_ITERS", "2000"))
GRAPH_ITERS = int(os.environ.get("CAR_GRAPH_ITERS", "0"))
TIME_CALLS = 500


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _gather(obj, group, ws, rank):
    data = [[None] for _ in range(ws)]
    data[rank][0] = obj
    for i in range(ws):
        dist.broadcast_object_list(data[i], src=i, group=group, device="cpu")
    return [d[0] for d in data]


def _worker(rank, ws, port, q):
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=ws)
    cpu = dist.group.WORLD
    nccl = dist.new_group(backend="nccl")
    import sglang.srt.distributed.device_communicators.custom_all_reduce_ops as ops

    assert ops.IS_CUSTOM_AR_AVAILABLE, "custom all-reduce ops not built"
    max_size = 2 * 8192 * 1024
    dev = torch.device(f"cuda:{rank}")
    meta = ops.allocate_meta_buffer(ops.meta_size() + max_size)
    buf = torch.empty(max_size, dtype=torch.uint8, device=dev)
    hs = _gather((bytes(ops.get_meta_buffer_ipc_handle(meta)), 0), cpu, ws, rank)
    rank_data = torch.empty(max_size, dtype=torch.uint8, device=dev)
    ptr = ops.init_custom_ar(meta, rank_data, [h for h, _ in hs], [o for _, o in hs], rank, False)
    hb = _gather((bytes(ops.get_meta_buffer_ipc_handle(buf)), 0), cpu, ws, rank)
    ops.register_buffer(ptr, buf, [h for h, _ in hb], [o for _, o in hb])

    res = {}
    for rows in ROWS:
        bad = div = 0
        gen = torch.Generator(device=dev).manual_seed(1000 * rows + rank)
        for it in range(ITERS):
            x = torch.randn(rows, COLS, dtype=torch.bfloat16, device=dev, generator=gen)
            torch.cuda.synchronize()
            dist.barrier(group=cpu)
            out = torch.empty_like(x)
            ops.all_reduce_unreg(ptr, x, buf, out)
            torch.cuda.synchronize()
            xs = [torch.empty_like(x) for _ in range(ws)]
            dist.all_gather(xs, x, group=nccl)
            ref = torch.stack([t.float() for t in xs]).sum(0).to(torch.bfloat16)
            outs = [torch.empty_like(out) for _ in range(ws)]
            dist.all_gather(outs, out, group=nccl)
            bad += int((out != ref).sum())
            div += int((outs[0] != outs[1]).any())
        # timing: custom AR vs RCCL, same input, eager, per call
        x = torch.randn(rows, COLS, dtype=torch.bfloat16, device=dev)
        out, y = torch.empty_like(x), x.clone()
        t = {}
        for name, fn in (("car", lambda: ops.all_reduce_unreg(ptr, x, buf, out)),
                         ("rccl", lambda: dist.all_reduce(y, group=nccl))):
            for _ in range(50):
                fn()
            torch.cuda.synchronize(); dist.barrier(group=cpu)
            s = time.perf_counter()
            for _ in range(TIME_CALLS):
                fn()
            torch.cuda.synchronize()
            t[name] = (time.perf_counter() - s) / TIME_CALLS * 1e6
        res[rows] = dict(bad=bad, div=div, car_us=t["car"], rccl_us=t["rccl"])
    q.put((rank, res))
    dist.barrier(group=cpu)
    dist.destroy_process_group()


def test_car_pcie():
    assert torch.cuda.device_count() >= 2
    port, ctx = _port(), mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_worker, args=(r, 2, port, q)) for r in range(2)]
    [p.start() for p in ps]
    got = dict(q.get(timeout=3000) for _ in ps)
    [p.join() for p in ps]
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    lines = [f"arch={arch} iters/rows={ITERS} cols={COLS}"]
    for rows in ROWS:
        r0, r1 = got[0][rows], got[1][rows]
        lines.append(f"rows={rows}: wrong elems rank0={r0['bad']} rank1={r1['bad']} "
                     f"rank-divergent iters={r0['div']} | car {r0['car_us']:.1f} us vs rccl {r0['rccl_us']:.1f} us")
    s = "\n".join(lines)
    print(s)
    with open(os.environ.get("CAR_OUT", "/ci/out/car_result.log"), "w") as f:
        f.write(s + "\n")
