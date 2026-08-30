import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32, const_expr
from cutlass.cute.runtime import from_dlpack

TILE_M, TILE_N = 128, 128
THREADS = 128


class SchedDemo:
    def __init__(self, swizzle_size: int, raster_along_m: bool, use_lpt: bool,
                 num_sms: int):
        self.swizzle_size = swizzle_size
        self.raster_along_m = raster_along_m
        self.use_lpt = use_lpt
        self.num_sms = num_sms

    @cute.jit
    def __call__(self, mX, mO, mWork, mPerm, mTrace, stream):

        ntile_mnl = (
            cute.ceil_div(mX.shape[0], TILE_M),
            cute.ceil_div(mX.shape[1], TILE_N),
            1,
        )
        params = utils.PersistentTileSchedulerParams(
            ntile_mnl,
            (1, 1, 1),
            swizzle_size=self.swizzle_size,
            raster_along_m=self.raster_along_m,
        )

        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            params, self.num_sms
        )
        self.kernel(mX, mO, mWork, mPerm, mTrace, params).launch(
            grid=grid, block=[THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, mX, mO, mWork, mPerm, mTrace, params):

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        ntile_n = cute.ceil_div(mX.shape[1], TILE_N)

        smem = cutlass.utils.SmemAllocator()
        red_buf = smem.allocate_array(Float32, THREADS // 32)

        sched = utils.StaticPersistentTileScheduler.create(
            params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work = sched.initial_work_tile_info()
        slot = Int32(0)
        while work.is_valid_tile:
            tm, tn, _ = work.tile_idx


            if const_expr(self.use_lpt):
                lin = mPerm[tm * ntile_n + tn]
                tile_m = lin // ntile_n
                tile_n = lin % ntile_n
            else:
                tile_m, tile_n = tm, tn

            if tidx == 0 and slot < mTrace.shape[1]:
                mTrace[bidx, slot] = tile_m * ntile_n + tile_n


            gX = cute.local_tile(mX, (TILE_M, TILE_N), (tile_m, tile_n))
            w = mWork[tile_m]
            acc = Float32(0.0)


            for _ in cutlass.range(0, w, 1, unroll=1):
                part = Float32(0.0)
                for i in cutlass.range_constexpr(TILE_M * TILE_N // THREADS):
                    off = i * THREADS + tidx
                    part += gX[off // TILE_N, off % TILE_N]
                acc += part


            acc = cute.arch.warp_reduction(
                acc, lambda p, q: p + q, threads_in_group=32
            )
            lane = tidx % 32
            warp = tidx // 32
            if lane == 0:
                red_buf[warp] = acc
            cute.arch.sync_threads()
            if warp == 0:
                v = red_buf[lane] if lane < THREADS // 32 else Float32(0.0)
                v = cute.arch.warp_reduction(
                    v, lambda p, q: p + q, threads_in_group=THREADS // 32
                )
                if lane == 0:
                    mO[tile_m, tile_n] = v
            cute.arch.sync_threads()

            sched.advance_to_next_work()
            work = sched.get_current_work()
            slot += 1


def build_work(ntile_m):

    w = [1 + (i % 4) * 3 for i in range(ntile_m)]
    return torch.tensor(w, dtype=torch.int32, device="cuda")


def run_variant(M, N, swizzle_size, raster_along_m, use_lpt, ref=None):
    torch.manual_seed(3)
    x = torch.randn(M, N, dtype=torch.float32, device="cuda")
    ntile_m, ntile_n = M // TILE_M, N // TILE_N
    o = torch.zeros(ntile_m, ntile_n, dtype=torch.float32, device="cuda")
    work = build_work(ntile_m)


    order = torch.argsort(work.repeat_interleave(ntile_n), descending=True)
    perm = order.to(torch.int32).contiguous()

    torch.cuda.synchronize()
    num_sms = utils.HardwareInfo().get_device_multiprocessor_count()
    trace = torch.full((num_sms, 64), -1, dtype=torch.int32, device="cuda")

    import cuda.bindings.driver as cuda_drv
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

    op = SchedDemo(swizzle_size, raster_along_m, use_lpt, num_sms)
    args = (
        from_dlpack(x, assumed_align=16),
        from_dlpack(o, assumed_align=16),
        from_dlpack(work, assumed_align=16),
        from_dlpack(perm, assumed_align=16),
        from_dlpack(trace, assumed_align=16),
        stream,
    )
    compiled = cute.compile(op, *args)
    compiled(*args)
    torch.cuda.synchronize()

    if ref is None:

        blocks = x.reshape(ntile_m, TILE_M, ntile_n, TILE_N).sum(dim=(1, 3))
        ref = blocks * work.float().unsqueeze(1)
    err = (o - ref).abs().max().item() / max(1.0, ref.abs().max().item())

    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(5):
        compiled(*args)
    start.record()
    for _ in range(20):
        compiled(*args)
    stop.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(stop) / 20 * 1e3

    tag = f"swz={swizzle_size} rasterM={int(raster_along_m)} lpt={int(use_lpt)}"
    ok = err < 1e-4
    print(f"{'OK  ' if ok else 'FAIL'} {tag:28s} {us:9.2f} us  rel_err={err:.2e}"
          f"  cta0_tiles={trace[0, :6].tolist()}")
    return ok, us, ref


def main():
    M, N = 4096, 4096
    variants = [
        (1, True, False),
        (4, True, False),
        (8, False, False),
        (1, True, True),
    ]
    ref, all_ok, times = None, True, {}
    for (s, r, l) in variants:
        ok, us, ref = run_variant(M, N, s, r, l, ref)
        all_ok &= ok
        times[(s, r, l)] = us
    base = times[(1, True, False)]
    for k, v in times.items():
        print(f"     {k}: {base / v:.2f}x vs plain")
    print("PASS" if all_ok else "FAIL")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
