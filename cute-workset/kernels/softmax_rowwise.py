import math

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass import Float32, const_expr


class RowSoftmax:
    def __init__(self, dtype, N: int):
        self.dtype = dtype
        self.N = N


        self.vecsize = math.gcd(N, 128 // dtype.width)


        if N <= 64:
            tpr = 8
        elif N <= 128:
            tpr = 16
        else:
            tpr = 32
        self.threads_per_row = tpr
        self.num_threads = 128

        cols_per_tile = self.vecsize * tpr
        self.blocks_n = max(1, (N + cols_per_tile - 1) // cols_per_tile)
        self.tiler_mn = (self.num_threads // tpr, cols_per_tile * self.blocks_n)

    @cute.jit
    def __call__(self, mX: cute.Tensor, mO: cute.Tensor, stream):

        atom_g2s = cute.make_copy_atom(
            cpasync.CopyG2SOp(), self.dtype,
            num_bits_per_copy=self.vecsize * self.dtype.width)
        atom_r2g = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.dtype,
            num_bits_per_copy=self.vecsize * self.dtype.width)
        thr_layout = cute.make_ordered_layout(
            (self.num_threads // self.threads_per_row, self.threads_per_row),
            order=(1, 0))
        val_layout = cute.make_layout((1, self.vecsize))
        tiled_copy_g2s = cute.make_tiled_copy_tv(atom_g2s, thr_layout, val_layout)
        tiled_copy_r2g = cute.make_tiled_copy_tv(atom_r2g, thr_layout, val_layout)

        num_blocks = cute.ceil_div(mX.shape[0], self.tiler_mn[0])
        self.kernel(mX, mO, tiled_copy_g2s, tiled_copy_r2g).launch(
            grid=[num_blocks, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mO: cute.Tensor,
               tiled_copy_g2s: cute.TiledCopy, tiled_copy_r2g: cute.TiledCopy):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(self.tiler_mn, order=(1, 0)),
            byte_alignment=16)


        gX = cute.local_tile(mX, self.tiler_mn, (bidx, 0))
        gO = cute.local_tile(mO, self.tiler_mn, (bidx, 0))
        cX = cute.local_tile(cute.make_identity_tensor(mX.shape),
                             self.tiler_mn, (bidx, 0))

        thr_copy = tiled_copy_g2s.get_slice(tidx)
        tXgX = thr_copy.partition_S(gX)
        tXsX = thr_copy.partition_D(sX)
        tXcX = thr_copy.partition_S(cX)
        thr_copy_out = tiled_copy_r2g.get_slice(tidx)
        tXgO = thr_copy_out.partition_D(gO)
        tXrX = cute.make_fragment_like(tXgX)
        tXrO = cute.make_fragment_like(tXgO)


        is_even_N = const_expr(mX.shape[1] % self.tiler_mn[1] == 0)
        tXpX = cute.make_rmem_tensor(
            cute.make_layout(
                (cute.size(tXcX, mode=[0, 1]), cute.size(tXcX, mode=[1]),
                 cute.size(tXcX, mode=[2])),
                stride=(cute.size(tXcX, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tXpX.shape[0]):
            for rest_k in cutlass.range_constexpr(tXpX.shape[2]):
                tXpX[rest_v, 0, rest_k] = cute.elem_less(
                    tXcX[(0, rest_v), 0, rest_k][1], mX.shape[1])

        row = tXcX[(0, 0), 0, 0][0]
        if row < mX.shape[0]:
            cute.copy(tiled_copy_g2s, tXgX, tXsX, pred=tXpX)


        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()


        if const_expr(not is_even_N):
            fill = cute.make_rmem_tensor_like(tXsX[(None, 0), None, 0])
            fill.fill(-Float32.inf)
            for rest_v in cutlass.range_constexpr(tXsX.shape[0][1]):
                for rest_k in cutlass.range_constexpr(tXsX.shape[2]):
                    if not tXpX[rest_v, 0, rest_k]:
                        cute.autovec_copy(fill, tXsX[(None, rest_v), None, rest_k])


        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(Float32)


        m = x.reduce(cute.ReductionOp.MAX, -Float32.inf, 0)
        m = cute.arch.warp_reduction(
            m, lambda a, b: cute.arch.fmax(a, b),
            threads_in_group=self.threads_per_row)
        if m == -Float32.inf:
            m = Float32(0.0)
        log2_e = math.log2(math.e)
        e = cute.math.exp2(x * log2_e - (m * log2_e), fastmath=True)
        s = e.reduce(cute.ReductionOp.ADD, 0.0, 0)
        s = cute.arch.warp_reduction(
            s, lambda a, b: a + b, threads_in_group=self.threads_per_row)
        y = e * (1.0 / s)


        tXrO.store(y.to(mO.element_type))
        if row < mX.shape[0]:
            cute.copy(tiled_copy_r2g, tXrO, tXgO, pred=tXpX)


def run_case(M, N, dtype=torch.float32, masked_rows=0):
    cute_dtype = {torch.float32: cutlass.Float32,
                  torch.bfloat16: cutlass.BFloat16}[dtype]
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    if masked_rows:
        x[:masked_rows] = float("-inf")
    o = torch.empty_like(x)
    op = RowSoftmax(cute_dtype, N)
    mX = from_dlpack(x, assumed_align=16)
    mO = from_dlpack(o, assumed_align=16)
    stream = torch.cuda.current_stream().cuda_stream
    import cuda.bindings.driver as cuda_drv
    op(mX, mO, cuda_drv.CUstream(stream))
    torch.cuda.synchronize()
    ref = torch.softmax(x.float(), dim=-1).to(dtype)
    tol = 1e-5 if dtype == torch.float32 else 2e-2
    diff = (o.float() - ref.float()).abs()
    diff = torch.where(torch.isnan(ref.float()) & torch.isnan(o.float()),
                       torch.zeros_like(diff), diff)
    err = diff.max().item()
    ok = err < tol and not math.isnan(err)
    print(f"{'OK  ' if ok else 'FAIL'} M={M:5d} N={N:5d} {str(dtype):15s} "
          f"masked={masked_rows} max_err={err:.2e}")
    return ok


if __name__ == "__main__":
    torch.manual_seed(0)
    ok = True
    ok &= run_case(256, 1024, torch.float32)
    ok &= run_case(256, 1000, torch.float32)
    ok &= run_case(1024, 2048, torch.bfloat16)
    ok &= run_case(333, 768, torch.bfloat16)
    ok &= run_case(64, 1000, torch.float32, masked_rows=3)
    print("PASS" if ok else "FAIL")
