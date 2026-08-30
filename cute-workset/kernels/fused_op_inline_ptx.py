import math

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass import Float32
from cutlass._mlir.dialects import llvm

TILE_M = 8
THREADS = 128
VEC = 4
LOG2_E = math.log2(math.e)


class FusedSilu:
    def __init__(self, mode: str):
        assert mode in ("math", "tanh_ptx", "ex2_ptx")
        self.mode = mode

    @cute.jit
    def __call__(self, mX: cute.Tensor, mO: cute.Tensor, stream):
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128)
        thr_layout = cute.make_ordered_layout(
            (TILE_M, THREADS // TILE_M), order=(1, 0))
        val_layout = cute.make_layout((1, VEC))
        tiled_copy = cute.make_tiled_copy_tv(atom, thr_layout, val_layout)
        tiler_mn = (TILE_M, (THREADS // TILE_M) * VEC)

        self.kernel(mX, mO, tiled_copy, tiler_mn).launch(
            grid=[mX.shape[0] // tiler_mn[0], mX.shape[1] // tiler_mn[1], 1],
            block=[THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mO: cute.Tensor,
               tiled_copy: cute.TiledCopy, tiler_mn: cute.Shape):
        tidx, _, _ = cute.arch.thread_idx()
        bm, bn, _ = cute.arch.block_idx()

        gX = cute.local_tile(mX, tiler_mn, (bm, bn))
        gO = cute.local_tile(mO, tiler_mn, (bm, bn))
        thr_copy = tiled_copy.get_slice(tidx)
        tXgX = thr_copy.partition_S(gX)
        tXgO = thr_copy.partition_D(gO)
        frg = cute.make_fragment_like(tXgX)
        frgO = cute.make_fragment_like(tXgO)
        cute.copy(tiled_copy, tXgX, frg)

        if cutlass.const_expr(self.mode == "math"):

            x = frg.load()
            y = x / (cute.math.exp2(-(x * LOG2_E), fastmath=True) + 1.0)
            frgO.store(y)
        elif cutlass.const_expr(self.mode == "tanh_ptx"):


            t = frg.load() * 0.5
            frg.store(t)
            for i in cutlass.range_constexpr(cute.size(frg)):
                frg[i] = llvm.inline_asm(
                    Float32.mlir_type,
                    [frg[i].ir_value()],
                    "tanh.approx.f32 $0, $1;",
                    "=f,f",
                    has_side_effects=False,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
            frgO.store(t * frg.load() + t)

        else:


            x = frg.load()
            t = x * (-LOG2_E)
            frg.store(t)
            for i in cutlass.range_constexpr(cute.size(frg)):
                frg[i] = llvm.inline_asm(
                    Float32.mlir_type,
                    [frg[i].ir_value()],
                    "ex2.approx.f32 $0, $1;",
                    "=f,f",
                    has_side_effects=False,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
            frgO.store(x / (frg.load() + 1.0))


        cute.copy(tiled_copy, frgO, tXgO)


def bench(fn, mX, mO, stream, iters=100):
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(10):
        fn(mX, mO, stream)
    start.record()
    for _ in range(iters):
        fn(mX, mO, stream)
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters * 1e3


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 2048, 2048
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    ref = torch.nn.functional.silu(x)

    mX = from_dlpack(x, assumed_align=16)
    stream_t = torch.cuda.current_stream().cuda_stream
    import cuda.bindings.driver as cuda_drv
    stream = cuda_drv.CUstream(stream_t)

    ok = True
    for mode in ("math", "tanh_ptx", "ex2_ptx"):
        op = FusedSilu(mode)
        o = torch.zeros_like(x)
        mO = from_dlpack(o, assumed_align=16)
        compiled = cute.compile(op, mX, mO, stream)
        compiled(mX, mO, stream)
        torch.cuda.synchronize()
        err = (o - ref).abs().max().item()
        good = err < 2e-3
        ok &= good
        us = bench(compiled, mX, mO, stream)
        gbps = 2 * x.numel() * 4 / (us * 1e-6) / 1e9
        print(f"{'OK  ' if good else 'FAIL'} {mode:9s} {us:8.2f} us  "
              f"{gbps:8.1f} GB/s  max_err={err:.2e}")
    print("PASS" if ok else "FAIL")
