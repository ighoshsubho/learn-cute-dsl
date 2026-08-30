import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass import Float32

TILE = 32          # (TILE, TILE) tile per CTA (smem version)
THREADS = 256
NAIVE_TM = 8


def make_copy128():

    atom128 = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128)
    thr_layout = cute.make_ordered_layout((TILE, THREADS // TILE), order=(1, 0))
    val_layout = cute.make_layout((1, 4))
    return cute.make_tiled_copy_tv(atom128, thr_layout, val_layout)


def make_copy_naive():

    atom32 = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=32)
    thr_layout = cute.make_ordered_layout((NAIVE_TM, 32), order=(1, 0))
    val_layout = cute.make_layout((1, 1))
    return cute.make_tiled_copy_tv(atom32, thr_layout, val_layout)


@cute.kernel
def transpose_naive_kernel(mX: cute.Tensor, mO: cute.Tensor,
                           copy32: cute.TiledCopy):

    tidx, _, _ = cute.arch.thread_idx()
    bm, bn, _ = cute.arch.block_idx()

    gX = cute.local_tile(mX, (NAIVE_TM, 32), (bm, bn))       # (8, 32)
    gO = cute.local_tile(mO, (32, NAIVE_TM), (bn, bm))       # (32, 8)


    gOt = cute.composition(gO, cute.make_layout((NAIVE_TM, 32), stride=(32, 1)))

    thr32 = copy32.get_slice(tidx)
    tXgX = thr32.partition_S(gX)
    tXgOt = thr32.partition_D(gOt)
    frg = cute.make_fragment_like(tXgX)
    cute.copy(copy32, tXgX, frg)
    cute.copy(copy32, frg, tXgOt)


@cute.kernel
def transpose_smem_kernel(mX: cute.Tensor, mO: cute.Tensor,
                          copy128: cute.TiledCopy):
    tidx, _, _ = cute.arch.thread_idx()
    bm, bn, _ = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()


    sX = smem.allocate_tensor(
        Float32,
        cute.make_layout((TILE, TILE), stride=(TILE + 1, 1)),
        byte_alignment=16)


    gX = cute.local_tile(mX, (TILE, TILE), (bm, bn))
    gO = cute.local_tile(mO, (TILE, TILE), (bn, bm))

    thr128 = copy128.get_slice(tidx)


    tXgX = thr128.partition_S(gX)
    tXsX = thr128.partition_D(sX)
    frg_in = cute.make_fragment_like(tXgX)
    cute.copy(copy128, tXgX, frg_in)


    cute.autovec_copy(frg_in, tXsX)

    cute.arch.sync_threads()


    sXt = cute.composition(
        sX, cute.make_layout((TILE, TILE), stride=(TILE, 1)))
    tOsXt = thr128.partition_S(sXt)
    tOgO = thr128.partition_D(gO)
    frg_out = cute.make_fragment_like(tOgO)
    cute.autovec_copy(tOsXt, frg_out)
    cute.copy(copy128, frg_out, tOgO)


@cute.jit
def transpose_naive(mX: cute.Tensor, mO: cute.Tensor):
    copy32 = make_copy_naive()
    transpose_naive_kernel(mX, mO, copy32).launch(
        grid=[mX.shape[0] // NAIVE_TM, mX.shape[1] // 32, 1],
        block=[THREADS, 1, 1])


@cute.jit
def transpose_smem(mX: cute.Tensor, mO: cute.Tensor):
    copy128 = make_copy128()
    transpose_smem_kernel(mX, mO, copy128).launch(
        grid=[mX.shape[0] // TILE, mX.shape[1] // TILE, 1],
        block=[THREADS, 1, 1])


def bench(fn, mX, mO, iters=50):
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(5):
        fn(mX, mO)
    start.record()
    for _ in range(iters):
        fn(mX, mO)
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


if __name__ == "__main__":
    torch.manual_seed(0)
    M = N = 4096
    assert M % TILE == 0 and N % TILE == 0, "exercise keeps shapes tile-even"
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    ref = x.t().contiguous()

    mX = from_dlpack(x, assumed_align=16)
    ok = True
    results = {}
    for name, fn in (("naive", transpose_naive), ("smem", transpose_smem)):
        o = torch.zeros(N, M, device="cuda", dtype=torch.float32)
        mO = from_dlpack(o, assumed_align=16)
        compiled = cute.compile(fn, mX, mO)
        compiled(mX, mO)
        torch.cuda.synchronize()
        good = torch.equal(o, ref)
        ok &= good
        ms = bench(compiled, mX, mO)
        gbps = 2 * x.numel() * 4 / (ms * 1e-3) / 1e9
        results[name] = ms
        print(f"{'OK  ' if good else 'FAIL'} {name:6s} {ms:7.3f} ms   "
              f"{gbps:8.1f} GB/s")
    print(f"speedup smem vs naive: {results['naive'] / results['smem']:.2f}x")
    print("PASS" if ok else "FAIL")
