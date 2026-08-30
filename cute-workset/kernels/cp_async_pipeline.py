import math

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass import Float32


class SiluRowSum:
    NUM_THREADS = 128
    TPR = 32
    VEC = 4

    def __init__(self, num_stages: int):
        self.num_stages = num_stages
        self.TM = self.NUM_THREADS // self.TPR
        self.TK = self.TPR * self.VEC

    @cute.jit
    def __call__(self, mX: cute.Tensor, mO: cute.Tensor, stream):
        atom_g2s = cute.make_copy_atom(
            cpasync.CopyG2SOp(), Float32, num_bits_per_copy=self.VEC * 32)
        thr_layout = cute.make_ordered_layout((self.TM, self.TPR), order=(1, 0))
        val_layout = cute.make_layout((1, self.VEC))
        tiled_copy = cute.make_tiled_copy_tv(atom_g2s, thr_layout, val_layout)

        self.kernel(mX, mO, tiled_copy).launch(
            grid=[mX.shape[0] // self.TM, 1, 1],
            block=[self.NUM_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mO: cute.Tensor,
               tiled_copy: cute.TiledCopy):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        S = self.num_stages
        num_k_tiles = mX.shape[1] // self.TK

        smem = cutlass.utils.SmemAllocator()
        # (TM, TK, S): S independent stages, each a row-major (TM, TK) tile.
        sX = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.TM, self.TK, S),
                             stride=(self.TK, 1, self.TM * self.TK)),
            byte_alignment=16)

        # (TM, TK, num_k_tiles): coord None keeps the k-tile mode iterable.
        gX = cute.local_tile(mX, (self.TM, self.TK), (bidx, None))
        cX = cute.local_tile(cute.make_identity_tensor(mX.shape),
                             (self.TM, self.TK), (bidx, 0))

        thr_copy = tiled_copy.get_slice(tidx)
        tXgX = thr_copy.partition_S(gX)   # (V, rest_m, rest_k, num_k_tiles)
        tXsX = thr_copy.partition_D(sX)   # (V, rest_m, rest_k, S)
        tXcX = thr_copy.partition_S(cX)
        tXrX = cute.make_fragment_like(tXsX[None, None, None, 0])

        acc = Float32(0.0)
        log2_e = math.log2(math.e)


        for s in range(S - 1):
            cute.copy(tiled_copy, tXgX[None, None, None, s],
                      tXsX[None, None, None, s])
            cute.arch.cp_async_commit_group()

        for k in range(num_k_tiles):
            nxt = k + S - 1
            if nxt < num_k_tiles:
                cute.copy(tiled_copy, tXgX[None, None, None, nxt],
                          tXsX[None, None, None, nxt % S])
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(S - 2 if S > 1 else 0)
            cute.arch.sync_threads()

            cute.autovec_copy(tXsX[None, None, None, k % S], tXrX)
            x = tXrX.load().to(Float32)
            sig = 1.0 / (cute.math.exp2(-(x * log2_e), fastmath=True) + 1.0)
            acc = acc + (x * sig).reduce(cute.ReductionOp.ADD, 0.0, 0)

            cute.arch.sync_threads()


        acc = cute.arch.warp_reduction(
            acc, lambda a, b: a + b, threads_in_group=self.TPR)


        coord = tXcX[(0, 0), 0, 0]
        if coord[1] == 0:
            mO[coord[0]] = acc


def bench(fn, mX, mO, stream, iters=50):
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(5):
        fn(mX, mO, stream)
    start.record()
    for _ in range(iters):
        fn(mX, mO, stream)
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


if __name__ == "__main__":
    torch.manual_seed(0)
    M, K = 2048, 16384
    op3, op1 = SiluRowSum(num_stages=3), SiluRowSum(num_stages=1)
    assert M % op3.TM == 0 and K % op3.TK == 0, "shapes kept tile-even"

    x = torch.randn(M, K, device="cuda", dtype=torch.float32)
    ref = (x * torch.sigmoid(x)).sum(dim=1)

    mX = from_dlpack(x, assumed_align=16)
    stream_t = torch.cuda.current_stream().cuda_stream
    import cuda.bindings.driver as cuda_drv
    stream = cuda_drv.CUstream(stream_t)

    ok = True
    results = {}
    for name, op in (("1-stage", op1), ("3-stage", op3)):
        o = torch.zeros(M, device="cuda", dtype=torch.float32)
        mO = from_dlpack(o, assumed_align=16)
        compiled = cute.compile(op, mX, mO, stream)
        compiled(mX, mO, stream)
        torch.cuda.synchronize()
        err = (o - ref).abs().max().item()
        good = err < 1e-2
        ok &= good
        ms = bench(compiled, mX, mO, stream)
        gbps = x.numel() * 4 / (ms * 1e-3) / 1e9
        results[name] = ms
        print(f"{'OK  ' if good else 'FAIL'} {name:8s} {ms:7.3f} ms  "
              f"{gbps:8.1f} GB/s  max_err={err:.2e}")
    print(f"speedup 3-stage vs 1-stage: "
          f"{results['1-stage'] / results['3-stage']:.2f}x")
    print("PASS" if ok else "FAIL")
