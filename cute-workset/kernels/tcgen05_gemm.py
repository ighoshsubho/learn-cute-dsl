import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack
import torch


io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32
mma_inst_shape_mnk = (128, 256, 16)
mma_tiler_mnk = (128, 256, 64)
THREADS_PER_CTA = 128
TMEM_COLS = 512
SUBTILE_CNT = 4


@cute.struct
class SharedStorage:
    tma_mbar: cute.struct.MemRange[cutlass.Int64, 1]
    mma_mbar: cute.struct.MemRange[cutlass.Int64, 1]
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mk: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nk: cute.Tensor,
    mC_mn: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx, bidy, _ = cute.arch.block_idx()
    mma_coord_mnk = (bidx, bidy, None)


    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(
        element_type=io_dtype,
        layout=a_smem_layout.outer,
        byte_alignment=128,
        swizzle=a_smem_layout.inner,
    )
    sB = smem.allocate_tensor(
        element_type=io_dtype,
        layout=b_smem_layout.outer,
        byte_alignment=128,
        swizzle=b_smem_layout.inner,
    )
    tma_mbar = storage.tma_mbar.data_ptr()
    mma_mbar = storage.mma_mbar.data_ptr()


    tma_bytes = cute.size_in_bytes(io_dtype, a_smem_layout) + cute.size_in_bytes(
        io_dtype, b_smem_layout
    )

    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(tma_mbar, 1)
            cute.arch.mbarrier_init(mma_mbar, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()


    if warp_idx == 0:
        cute.arch.alloc_tmem(TMEM_COLS, storage.tmem_holding_buf.ptr)
    cute.arch.barrier()
    tmem_ptr = cute.arch.retrieve_tmem_ptr(
        acc_dtype,
        alignment=16,
        ptr_to_buffer_holding_addr=storage.tmem_holding_buf.ptr,
    )


    # (bM, bK, RestK) / (bN, bK, RestK) / (bM, bN)
    gA = cute.local_tile(mA_mk, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    gB = cute.local_tile(mB_nk, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
    gC = cute.local_tile(mC_mn, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))

    thr_mma = tiled_mma.get_slice(0)
    tCgA = thr_mma.partition_A(gA)            # (MMA, MMA_M, MMA_K, RestK)
    tCgB = thr_mma.partition_B(gB)            # (MMA, MMA_N, MMA_K, RestK)
    tCgC = thr_mma.partition_C(gC)            # (MMA, MMA_M, MMA_N)

    tCrA = tiled_mma.make_fragment_A(sA)      # (MMA, MMA_M, MMA_K)
    tCrB = tiled_mma.make_fragment_B(sB)      # (MMA, MMA_N, MMA_K)


    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)  # (MMA, MMA_M, MMA_N)


    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        0,
        cute.make_layout(1),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b,
        0,
        cute.make_layout(1),
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )


    num_k_tiles = cute.size(gA, mode=[2])
    if warp_idx == 0:
        for k_tile in cutlass.range(num_k_tiles):
            phase = k_tile % 2


            if k_tile > 0:


                cute.arch.mbarrier_wait(mma_mbar, 1 - phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(tma_mbar, tma_bytes)
            cute.copy(
                tma_atom_a, tAgA[(None, k_tile)], tAsA, tma_bar_ptr=tma_mbar
            )
            cute.copy(
                tma_atom_b, tBgB[(None, k_tile)], tBsB, tma_bar_ptr=tma_mbar
            )


            cute.arch.mbarrier_wait(tma_mbar, phase)


            num_k_blocks = cute.size(tCrA, mode=[2])
            for kb in cutlass.range_constexpr(num_k_blocks):
                cute.gemm(
                    tiled_mma,
                    tCtAcc,
                    tCrA[None, None, kb],
                    tCrB[None, None, kb],
                    tCtAcc,
                )

                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)


            with cute.arch.elect_one():
                tcgen05.commit(mma_mbar)


        cute.arch.mbarrier_wait(mma_mbar, (num_k_tiles - 1) % 2)


    cute.arch.barrier()


    epi_tiler = (
        (
            cute.size(tCtAcc, mode=[0, 0]),
            cute.size(tCtAcc, mode=[0, 1]) // SUBTILE_CNT,
        ),
    )
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)   # (EpiTile, NumTiles)
    gC_epi = cute.zipped_divide(tCgC, epi_tiler)         # (EpiTile, NumTiles)


    tmem_atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition(64)), acc_dtype
    )
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)

    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)   # (TmemCpy, NumCpy, NumTiles)
    tDgC = tmem_thr_copy.partition_D(gC_epi)       # (TmemCpy, NumCpy, NumTiles)

    tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, acc_dtype)
    tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, io_dtype)

    for i in cutlass.range(cute.size(tDtC, mode=[2])):
        cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
        tCrC.store(tCrAcc.load().to(io_dtype))
        cute.autovec_copy(tCrC, tDgC[None, None, i])


    cute.arch.barrier()
    if warp_idx == 0:
        cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.dealloc_tmem(tmem_ptr, TMEM_COLS)


@cute.jit
def host(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):


    op = tcgen05.MmaF16BF16Op(
        io_dtype,
        acc_dtype,
        mma_inst_shape_mnk,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)


    a_smem_layout_staged = sm100_utils.make_smem_layout_a(
        tiled_mma, mma_tiler_mnk, io_dtype, 1
    )
    b_smem_layout_staged = sm100_utils.make_smem_layout_b(
        tiled_mma, mma_tiler_mnk, io_dtype, 1
    )

    a_smem_layout = cute.select(a_smem_layout_staged, mode=[0, 1, 2])
    b_smem_layout = cute.select(b_smem_layout_staged, mode=[0, 1, 2])


    tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        tma_op, a, a_smem_layout, mma_tiler_mnk, tiled_mma
    )
    tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
        tma_op, b, b_smem_layout, mma_tiler_mnk, tiled_mma
    )

    grid_shape = cute.ceil_div((*c.layout.shape, 1), mma_tiler_mnk[:2])
    kernel(
        tiled_mma,
        tma_atom_a,
        tma_tensor_a,
        tma_atom_b,
        tma_tensor_b,
        c,
        a_smem_layout,
        b_smem_layout,
    ).launch(
        grid=grid_shape,
        block=(THREADS_PER_CTA, 1, 1),
    )


def run_one(m: int, n: int, k: int) -> bool:

    assert m % mma_tiler_mnk[0] == 0, "M must be divisible by 128"
    assert n % mma_tiler_mnk[1] == 0, "N must be divisible by 256"
    assert k % mma_tiler_mnk[2] == 0, "K must be divisible by 64"


    a = torch.empty(m, k, dtype=torch.int32).random_(-2, 2).to(torch.float16).cuda()
    b = torch.empty(n, k, dtype=torch.int32).random_(-2, 2).to(torch.float16).cuda()
    c = torch.zeros(m, n, dtype=torch.float16, device="cuda")

    host(
        from_dlpack(a, assumed_align=32),
        from_dlpack(b, assumed_align=32),
        from_dlpack(c, assumed_align=32),
    )
    torch.cuda.synchronize()

    ref = torch.einsum("mk,nk->mn", a.float(), b.float())
    ok = torch.allclose(c.float(), ref, atol=1e-2, rtol=0)
    print(f"  ({m:5d},{n:5d},{k:5d}) -> {'ok' if ok else 'MISMATCH'}")
    return ok


def main():
    torch.manual_seed(2025)
    shapes = [(512, 512, 512), (1024, 768, 256), (2048, 512, 1024)]
    if all(run_one(*s) for s in shapes):
        print("PASS")
    else:
        print("FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
