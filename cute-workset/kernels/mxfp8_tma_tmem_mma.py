import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack

ab_dtype = cutlass.Float8E4M3FN
sf_dtype = cutlass.Float8E8M0FNU
acc_dtype = cutlass.Float32
SF_VEC = 32
MMA_INST_TILE_K = 4
mma_tiler_mn = (128, 128)
THREADS_PER_CTA = 128
TMEM_COLS = 512
SUBTILE_CNT = 4

SF_ATOM_MN = 32
NUM_SFA_COLS = (mma_tiler_mn[0] // SF_ATOM_MN) * MMA_INST_TILE_K
NUM_SFB_COLS = (mma_tiler_mn[1] // SF_ATOM_MN) * MMA_INST_TILE_K
NUM_ACC_COLS = mma_tiler_mn[1]


def make_reordered_sf(values: torch.Tensor):

    mn, sf_k = values.shape
    assert mn % 128 == 0 and sf_k % 4 == 0
    rest_m = (mn + 127) // 128
    rest_k = (sf_k + 3) // 4


    #   (rest_m, rest_k, m32, m4, k4).


    encoded = values.to(torch.float8_e8m0fnu)
    storage = (
        encoded.reshape(rest_m, 4, 32, rest_k, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .view(torch.uint8)
    )
    raw = from_dlpack(storage, assumed_align=32)
    raw.element_type = sf_dtype
    return raw


@cute.struct
class SharedStorage:
    tma_mbar: cute.struct.MemRange[cutlass.Int64, 1]
    mma_mbar: cute.struct.MemRange[cutlass.Int64, 1]
    tmem_holding_buf: cutlass.Int32


def s2t_copy_and_partition(sSF, tSF, sf_dt):

    tCsSF_compact = cute.filter_zeros(sSF)
    tCtSF_compact = cute.filter_zeros(tSF)
    copy_atom_s2t = cute.make_copy_atom(
        tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), sf_dt
    )
    tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
    thr_copy_s2t = tiled_copy_s2t.get_slice(0)
    src_ = thr_copy_s2t.partition_S(tCsSF_compact)

    src = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, src_)
    dst = thr_copy_s2t.partition_D(tCtSF_compact)
    return tiled_copy_s2t, src, dst


@cute.kernel
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB: cute.Tensor,
    tma_atom_sfa: cute.CopyAtom,
    mSFA: cute.Tensor,
    tma_atom_sfb: cute.CopyAtom,
    mSFB: cute.Tensor,
    mC: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
    sfa_smem_layout: cute.Layout,
    sfb_smem_layout: cute.Layout,
    mma_tiler_mnk: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx, bidy, _ = cute.arch.block_idx()
    mma_coord_mnk = (bidx, bidy, None)


    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(ab_dtype, a_smem_layout.outer, 128, a_smem_layout.inner)
    sB = smem.allocate_tensor(ab_dtype, b_smem_layout.outer, 128, b_smem_layout.inner)
    sSFA = smem.allocate_tensor(sf_dtype, sfa_smem_layout, 128)
    sSFB = smem.allocate_tensor(sf_dtype, sfb_smem_layout, 128)

    tma_mbar = storage.tma_mbar.data_ptr()
    mma_mbar = storage.mma_mbar.data_ptr()


    tma_bytes = (
        cute.size_in_bytes(ab_dtype, a_smem_layout)
        + cute.size_in_bytes(ab_dtype, b_smem_layout)
        + cute.size_in_bytes(sf_dtype, sfa_smem_layout)
        + cute.size_in_bytes(sf_dtype, sfb_smem_layout)
    )

    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        cpasync.prefetch_descriptor(tma_atom_sfa)
        cpasync.prefetch_descriptor(tma_atom_sfb)
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(tma_mbar, 1)
            cute.arch.mbarrier_init(mma_mbar, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()


    if warp_idx == 0:
        cute.arch.alloc_tmem(TMEM_COLS, storage.tmem_holding_buf.ptr)
    cute.arch.barrier()
    acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
        acc_dtype, alignment=16,
        ptr_to_buffer_holding_addr=storage.tmem_holding_buf.ptr,
    )


    gA = cute.local_tile(mA, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    gB = cute.local_tile(mB, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
    gC = cute.local_tile(mC, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))
    gSFA = cute.local_tile(mSFA, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    gSFB = cute.local_tile(mSFB, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))

    thr_mma = tiled_mma.get_slice(0)
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)
    tCgC = thr_mma.partition_C(gC)
    tCgSFA = thr_mma.partition_A(gSFA)
    tCgSFB = thr_mma.partition_B(gSFB)

    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)

    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc = cute.make_tensor(acc_tmem_ptr, tiled_mma.make_fragment_C(acc_shape).layout)


    sfa_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + NUM_ACC_COLS, dtype=sf_dtype)
    tCtSFA = cute.make_tensor(
        sfa_tmem_ptr,
        blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma, mma_tiler_mnk, SF_VEC, sfa_smem_layout
        ),
    )
    sfb_tmem_ptr = cute.recast_ptr(
        acc_tmem_ptr + NUM_ACC_COLS + NUM_SFA_COLS, dtype=sf_dtype
    )
    tCtSFB = cute.make_tensor(
        sfb_tmem_ptr,
        blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma, mma_tiler_mnk, SF_VEC, sfb_smem_layout
        ),
    )


    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a, 0, cute.make_layout(1),
        cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3),
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b, 0, cute.make_layout(1),
        cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3),
    )
    tSFAsSFA, tSFAgSFA = cpasync.tma_partition(
        tma_atom_sfa, 0, cute.make_layout(1),
        cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3),
    )


    tSFAsSFA = cute.filter_zeros(tSFAsSFA)
    tSFAgSFA = cute.filter_zeros(tSFAgSFA)
    tSFBsSFB, tSFBgSFB = cpasync.tma_partition(
        tma_atom_sfb, 0, cute.make_layout(1),
        cute.group_modes(sSFB, 0, 3), cute.group_modes(tCgSFB, 0, 3),
    )
    tSFBsSFB = cute.filter_zeros(tSFBsSFB)
    tSFBgSFB = cute.filter_zeros(tSFBgSFB)


    s2t_sfa, sfa_s2t_src, sfa_s2t_dst = s2t_copy_and_partition(sSFA, tCtSFA, sf_dtype)
    s2t_sfb, sfb_s2t_src, sfb_s2t_dst = s2t_copy_and_partition(sSFB, tCtSFB, sf_dtype)


    num_k_tiles = cute.size(gA, mode=[2])
    if warp_idx == 0:
        for k_tile in cutlass.range(num_k_tiles):
            phase = k_tile % 2
            if k_tile > 0:
                cute.arch.mbarrier_wait(mma_mbar, 1 - phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(tma_mbar, tma_bytes)
            cute.copy(tma_atom_a, tAgA[(None, k_tile)], tAsA, tma_bar_ptr=tma_mbar)
            cute.copy(tma_atom_b, tBgB[(None, k_tile)], tBsB, tma_bar_ptr=tma_mbar)
            cute.copy(tma_atom_sfa, tSFAgSFA[(None, k_tile)], tSFAsSFA,
                      tma_bar_ptr=tma_mbar)
            cute.copy(tma_atom_sfb, tSFBgSFB[(None, k_tile)], tSFBsSFB,
                      tma_bar_ptr=tma_mbar)
            cute.arch.mbarrier_wait(tma_mbar, phase)


            cute.copy(s2t_sfa, sfa_s2t_src, sfa_s2t_dst)
            cute.copy(s2t_sfb, sfb_s2t_src, sfb_s2t_dst)
            cute.arch.fence_view_async_tmem_store()

            tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)


            cute.gemm(
                tiled_mma,
                tCtAcc,
                [tCrA, tCtSFA],
                [tCrB, tCtSFB],
                tCtAcc,
            )
            with cute.arch.elect_one():
                tcgen05.commit(mma_mbar)
        cute.arch.mbarrier_wait(mma_mbar, (num_k_tiles - 1) % 2)
    cute.arch.barrier()


    epi_tiler = (
        (cute.size(tCtAcc, mode=[0, 0]),
         cute.size(tCtAcc, mode=[0, 1]) // SUBTILE_CNT),
    )
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    gC_epi = cute.zipped_divide(tCgC, epi_tiler)

    tmem_atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), acc_dtype
    )
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
    tDgC = tmem_thr_copy.partition_D(gC_epi)

    frg = cute.make_rmem_tensor(tDgC[None, None, 0].shape, acc_dtype)
    for i in cutlass.range(cute.size(tDtC, mode=[2])):
        cute.copy(tmem_tiled_copy, tDtC[None, None, i], frg)
        cute.autovec_copy(frg, tDgC[None, None, i])

    cute.arch.barrier()
    if warp_idx == 0:
        cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.dealloc_tmem(acc_tmem_ptr, TMEM_COLS)


@cute.jit
def host(a: cute.Tensor, b: cute.Tensor, sfa_raw: cute.Tensor,
         sfb_raw: cute.Tensor, c: cute.Tensor):
    tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
        ab_dtype, ab_dtype,
        cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
        sf_dtype, SF_VEC, tcgen05.CtaGroup.ONE, mma_tiler_mn,
    )
    mma_inst_k = cute.size(tiled_mma.shape_mnk, mode=[2])
    mma_tiler_mnk = (mma_tiler_mn[0], mma_tiler_mn[1], mma_inst_k * MMA_INST_TILE_K)


    sfa = cute.make_tensor(
        sfa_raw.iterator,
        blockscaled_utils.tile_atom_to_shape_SF((*a.shape, 1), SF_VEC),
    )[None, None, 0]
    sfb = cute.make_tensor(
        sfb_raw.iterator,
        blockscaled_utils.tile_atom_to_shape_SF((*b.shape, 1), SF_VEC),
    )[None, None, 0]

    a_smem_layout = cute.select(
        sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, ab_dtype, 1),
        mode=[0, 1, 2])
    b_smem_layout = cute.select(
        sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, ab_dtype, 1),
        mode=[0, 1, 2])
    sfa_smem_layout = cute.slice_(
        blockscaled_utils.make_smem_layout_sfa(tiled_mma, mma_tiler_mnk, SF_VEC, 1),
        (None, None, None, 0))
    sfb_smem_layout = cute.slice_(
        blockscaled_utils.make_smem_layout_sfb(tiled_mma, mma_tiler_mnk, SF_VEC, 1),
        (None, None, None, 0))

    tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        tma_op, a, a_smem_layout, mma_tiler_mnk, tiled_mma)
    tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
        tma_op, b, b_smem_layout, mma_tiler_mnk, tiled_mma)
    tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
        tma_op, sfa, sfa_smem_layout, mma_tiler_mnk, tiled_mma,
        internal_type=cutlass.Int16)
    tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
        tma_op, sfb, sfb_smem_layout, mma_tiler_mnk, tiled_mma,
        internal_type=cutlass.Int16)

    grid = cute.ceil_div((*c.layout.shape, 1), mma_tiler_mn)
    kernel(
        tiled_mma,
        tma_atom_a, tma_tensor_a,
        tma_atom_b, tma_tensor_b,
        tma_atom_sfa, tma_tensor_sfa,
        tma_atom_sfb, tma_tensor_sfb,
        c,
        a_smem_layout, b_smem_layout, sfa_smem_layout, sfb_smem_layout,
        mma_tiler_mnk,
    ).launch(grid=grid, block=(THREADS_PER_CTA, 1, 1))


TILE_M_CHECK, TILE_N_CHECK, TILE_K_CHECK = 128, 128, 128


def make_inputs(M, N, K):

    import cutlass.torch as cutlass_torch

    torch.manual_seed(7)
    a_f32 = (torch.randn(M, K, device="cuda") * 0.5)
    b_f32 = (torch.randn(N, K, device="cuda") * 0.5)

    a_cute, _ = cutlass_torch.cute_tensor_like(
        a_f32.cpu(), ab_dtype, is_dynamic_layout=False, assumed_align=32)
    a_cute = cutlass_torch.convert_cute_tensor(
        a_f32, a_cute, ab_dtype, is_dynamic_layout=False)
    b_cute, _ = cutlass_torch.cute_tensor_like(
        b_f32.cpu(), ab_dtype, is_dynamic_layout=False, assumed_align=32)
    b_cute = cutlass_torch.convert_cute_tensor(
        b_f32, b_cute, ab_dtype, is_dynamic_layout=False)


    ea = torch.randint(125, 130, (M, K // SF_VEC), device="cuda").float()
    eb = torch.randint(125, 130, (N, K // SF_VEC), device="cuda").float()


    sfa_cute = make_reordered_sf(torch.pow(2.0, ea - 127.0))
    sfb_cute = make_reordered_sf(torch.pow(2.0, eb - 127.0))


    a_q = a_f32.to(torch.float8_e4m3fn).float()
    b_q = b_f32.to(torch.float8_e4m3fn).float()
    sa = torch.pow(2.0, ea - 127.0).repeat_interleave(SF_VEC, dim=1)
    sb = torch.pow(2.0, eb - 127.0).repeat_interleave(SF_VEC, dim=1)
    ref = torch.einsum("mk,nk->mn", a_q * sa, b_q * sb)
    return a_cute, b_cute, sfa_cute, sfb_cute, ref


def run_one(M, N, K):
    assert M % TILE_M_CHECK == 0 and N % TILE_N_CHECK == 0
    assert K % TILE_K_CHECK == 0
    a_cute, b_cute, sfa_cute, sfb_cute, ref = make_inputs(M, N, K)
    c = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    host(a_cute, b_cute, sfa_cute, sfb_cute, from_dlpack(c, assumed_align=32))
    torch.cuda.synchronize()

    err = (c - ref).abs().max().item() / max(1.0, ref.abs().max().item())
    ok = err < 5e-2
    print(f"  ({M:5d},{N:5d},{K:5d}) -> {'ok' if ok else 'MISMATCH'}  rel_err={err:.3e}")
    return ok


def main():
    if all(run_one(*s) for s in [(256, 256, 256), (512, 256, 512)]):
        print("PASS")
    else:
        print("FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
