import math

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass import Float32


@cute.kernel
def vector_add_kernel(
    gA: cute.Tensor,   # ((TileM,TileN),(RestM,RestN))
    gB: cute.Tensor,
    gC: cute.Tensor,
    cC: cute.Tensor,
    shape: cute.Shape,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
    copy_bits: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()


    blk_coord = ((None, None), bidx)
    blkA = gA[blk_coord]      # (TileM, TileN)
    blkB = gB[blk_coord]
    blkC = gC[blk_coord]
    blkCrd = cC[blk_coord]


    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), gA.element_type, num_bits_per_copy=copy_bits
    )
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
    thr_copy = tiled_copy.get_slice(tidx)

    thrA = thr_copy.partition_S(blkA)     # ((atom_v, rest_v), rest_m, rest_n)
    thrB = thr_copy.partition_S(blkB)
    thrC = thr_copy.partition_D(blkC)
    thrCrd = thr_copy.partition_S(blkCrd)

    frgA = cute.make_fragment_like(thrA)
    frgB = cute.make_fragment_like(thrB)
    frgC = cute.make_fragment_like(thrC)


    # (rest_v, rest_m, rest_n) — NOT per element. cute.copy broadcasts each


    frgPred = cute.make_rmem_tensor(
        cute.make_layout(
            (
                cute.size(thrCrd, mode=[0, 1]),
                cute.size(thrCrd, mode=[1]),
                cute.size(thrCrd, mode=[2]),
            )
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(frgPred.shape[0]):
        for m in cutlass.range_constexpr(frgPred.shape[1]):
            for n in cutlass.range_constexpr(frgPred.shape[2]):
                frgPred[rest_v, m, n] = cute.elem_less(
                    thrCrd[(0, rest_v), m, n], shape
                )


    cute.copy(tiled_copy, thrA, frgA, pred=frgPred)
    cute.copy(tiled_copy, thrB, frgB, pred=frgPred)


    result = frgA.load().to(Float32) + frgB.load().to(Float32)
    frgC.store(result.to(gC.element_type))


    cute.copy(tiled_copy, frgC, thrC, pred=frgPred)


@cute.jit
def vector_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
               vecsize: cutlass.Constexpr):
    dtype = mA.element_type
    copy_bits = vecsize * dtype.width


    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, vecsize), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)


    # ((TileM,TileN),(RestM,RestN)) — rounds up, edge tiles are ragged.
    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)
    cC = cute.zipped_divide(cute.make_identity_tensor(mC.shape), tiler=tiler_mn)

    vector_add_kernel(
        gA, gB, gC, cC, mC.shape, thr_layout, val_layout, copy_bits
    ).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


def run_case(M, N, check_ptx=False):
    a = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    c = torch.zeros_like(a)

    mA = from_dlpack(a, assumed_align=16)
    mB = from_dlpack(b, assumed_align=16)
    mC = from_dlpack(c, assumed_align=16)


    vecsize = math.gcd(N, 128 // cutlass.BFloat16.width)

    compiled = cute.compile[cute.KeepPTX()](vector_add, mA, mB, mC, vecsize)
    compiled(mA, mB, mC)
    torch.cuda.synchronize()

    ref = (a.float() + b.float()).bfloat16()
    ok = torch.equal(c, ref)
    print(f"{'OK  ' if ok else 'FAIL'} M={M:5d} N={N:5d} vecsize={vecsize}")

    if check_ptx:


        ptx = compiled.__ptx__
        vectorized = ("v4" in ptx) or ("b128" in ptx)
        print(f"{'OK  ' if vectorized else 'FAIL'} PTX contains a 128-bit "
              f"vectorized access (v4/b128)")
        ok = ok and vectorized
    return ok


if __name__ == "__main__":
    torch.manual_seed(0)
    ok = True
    ok &= run_case(1024, 1024, check_ptx=True)
    ok &= run_case(513, 1000)
    ok &= run_case(100, 257)
    print("PASS" if ok else "FAIL")
