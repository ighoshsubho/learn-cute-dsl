import math

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

TILE_M, TILE_N = 8, 128
THREADS = 128
VEC = 4
STAGES = 3


class StageModifyStore:
    def __init__(self, stages: int):
        self.stages = stages

    @cute.jit
    def __call__(self, mX: cute.Tensor, mO: cute.Tensor, stream):

        atom_load = cute.make_copy_atom(
            cpasync.CopyG2SOp(), Float32, num_bits_per_copy=128
        )
        atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
        )
        thr_layout = cute.make_ordered_layout(
            (TILE_M, THREADS // TILE_M), order=(1, 0)
        )
        val_layout = cute.make_layout((1, VEC))
        tiled_copy_load = cute.make_tiled_copy_tv(atom_load, thr_layout, val_layout)
        tiled_copy_store = cute.make_tiled_copy_tv(atom_store, thr_layout, val_layout)
        # (TILE_M, TILE_N, stages): row-major inside a stage, stages outermost
        sX_layout = cute.make_ordered_layout(
            (TILE_M, TILE_N, self.stages), order=(1, 0, 2)
        )
        self.kernel(
            mX, mO, tiled_copy_load, tiled_copy_store, sX_layout
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], TILE_M), 1, 1],
            block=[THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mX, mO, tiled_copy_load, tiled_copy_store, sX_layout):

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(Float32, sX_layout, byte_alignment=16)


        gX = cute.local_tile(mX, (TILE_M, TILE_N), (bidx, None))
        gO = cute.local_tile(mO, (TILE_M, TILE_N), (bidx, None))
        cX = cute.local_tile(
            cute.make_identity_tensor(mX.shape), (TILE_M, TILE_N), (bidx, None)
        )

        thr_load = tiled_copy_load.get_slice(tidx)
        thr_store = tiled_copy_store.get_slice(tidx)
        tXgX = thr_load.partition_S(gX)     # ((v,rest),1,1,k_tiles)
        tXsX = thr_load.partition_D(sX)     # ((v,rest),1,1,stages)
        tXgO = thr_store.partition_D(gO)    # ((v,rest),1,1,k_tiles)
        tXcX = thr_load.partition_S(cX)

        tXrX = cute.make_rmem_tensor(tXsX[None, None, None, 0].shape, Float32)
        tXrO = cute.make_rmem_tensor(tXsX[None, None, None, 0].shape, Float32)

        num_k_tiles = cute.size(gX, mode=[2])
        is_even_N = const_expr(mX.shape[1] % TILE_N == 0)


        pred_layout = cute.make_layout(
            (
                cute.size(tXcX, mode=[0, 1]),
                cute.size(tXcX, mode=[1]),
                cute.size(tXcX, mode=[2]),
            ),
            stride=(cute.size(tXcX, mode=[2]), 0, 1),
        )


        for s in range(self.stages - 1):
            if s < num_k_tiles:
                self._load_tile(
                    tiled_copy_load, tXgX, tXsX, tXcX, pred_layout,
                    mX.shape[1], s, s, is_even_N,
                )
            cute.arch.cp_async_commit_group()

        for k in range(num_k_tiles):
            nxt = k + self.stages - 1
            if nxt < num_k_tiles:
                self._load_tile(
                    tiled_copy_load, tXgX, tXsX, tXcX, pred_layout,
                    mX.shape[1], nxt, nxt % self.stages, is_even_N,
                )

            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(
                self.stages - 2 if const_expr(self.stages > 1) else 0
            )
            cute.arch.sync_threads()


            cute.autovec_copy(tXsX[None, None, None, k % self.stages], tXrX)
            v = tXrX.load()
            tXrO.store(v * (1.0 + cute.math.tanh(v, fastmath=True)))


            if const_expr(is_even_N):
                cute.copy(tiled_copy_store, tXrO, tXgO[None, None, None, k])
            else:
                pred = cute.make_rmem_tensor(pred_layout, cutlass.Boolean)
                for rv in cutlass.range_constexpr(pred_layout.shape[0]):
                    for rk in cutlass.range_constexpr(pred_layout.shape[2]):
                        pred[rv, 0, rk] = cute.elem_less(
                            tXcX[(0, rv), 0, rk, k][1], mX.shape[1]
                        )
                cute.copy(
                    tiled_copy_store, tXrO, tXgO[None, None, None, k], pred=pred
                )


            cute.arch.sync_threads()

    @cute.jit
    def _load_tile(self, tiled_copy, tXgX, tXsX, tXcX, pred_layout,
                   n_limit, tile_idx, stage_idx, is_even_N: cutlass.Constexpr):
        if const_expr(is_even_N):
            cute.copy(
                tiled_copy,
                tXgX[None, None, None, tile_idx],
                tXsX[None, None, None, stage_idx],
            )
        else:
            pred = cute.make_rmem_tensor(pred_layout, cutlass.Boolean)
            for rv in cutlass.range_constexpr(pred_layout.shape[0]):
                for rk in cutlass.range_constexpr(pred_layout.shape[2]):
                    pred[rv, 0, rk] = cute.elem_less(
                        tXcX[(0, rv), 0, rk, tile_idx][1], n_limit
                    )
            cute.copy(
                tiled_copy,
                tXgX[None, None, None, tile_idx],
                tXsX[None, None, None, stage_idx],
                pred=pred,
            )


            fill = cute.make_rmem_tensor_like(
                tXsX[(None, 0), None, 0, stage_idx]
            )
            fill.fill(0.0)
            for rv in cutlass.range_constexpr(tXsX.shape[0][1]):
                for rk in cutlass.range_constexpr(tXsX.shape[2]):
                    if not pred[rv, 0, rk]:
                        cute.autovec_copy(
                            fill, tXsX[(None, rv), None, rk, stage_idx]
                        )


def run_case(M: int, N: int, stages: int, verbose=True) -> float:
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=torch.float32, device="cuda")
    o = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    op = StageModifyStore(stages)
    import cuda.bindings.driver as cuda_drv
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)


    mX = from_dlpack(x, assumed_align=16)
    mO = from_dlpack(o, assumed_align=16)
    compiled = cute.compile(op, mX, mO, stream)

    compiled(mX, mO, stream)
    torch.cuda.synchronize()

    ref = x * (1.0 + torch.tanh(x))
    err = (o - ref).abs().max().item()

    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(10):
        compiled(mX, mO, stream)
    start.record()
    for _ in range(50):
        compiled(mX, mO, stream)
    stop.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(stop) / 50 * 1e3
    gbs = 2 * M * N * 4 / (us * 1e-6) / 1e9

    ok = err < 1e-5
    if verbose:
        print(f"{'OK  ' if ok else 'FAIL'} S={stages} M={M:5d} N={N:5d} "
              f"{us:7.2f} us  {gbs:7.1f} GB/s  max_err={err:.2e}")
    return us if ok else -1.0


def main():
    results = {}
    for stages in (1, STAGES):
        ok = True
        for (M, N) in [(2048, 4096), (1024, 4000)]:
            t = run_case(M, N, stages)
            if t < 0:
                ok = False
            if (M, N) == (2048, 4096):
                results[stages] = t
        if not ok:
            print("FAIL")
            raise SystemExit(1)
    if results.get(1, 0) > 0 and results.get(STAGES, 0) > 0:
        print(f"speedup {STAGES}-stage vs 1-stage: "
              f"{results[1] / results[STAGES]:.2f}x")
    print("PASS")


if __name__ == "__main__":
    main()
