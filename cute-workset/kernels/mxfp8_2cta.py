import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

ab_dtype = cutlass.Float8E4M3FN
sf_dtype = cutlass.Float8E8M0FNU
acc_dtype = cutlass.Float32
SF_VEC = 32
MMA_INST_TILE_K = 4
mma_tiler_mn = (256, 128)
mma_tiler_sfb_mn = (128, 128)
CLUSTER_SHAPE_MN = (2, 1)
THREADS_PER_CTA = 128
TMEM_COLS = 512
SUBTILE_CNT = 4

SF_ATOM_MN = 32


NUM_SFA_COLS = ((mma_tiler_mn[0] // 2) // SF_ATOM_MN) * MMA_INST_TILE_K
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

    ab_mbars: cute.struct.MemRange[cutlass.Int64, 2]
    acc_mbars: cute.struct.MemRange[cutlass.Int64, 2]
    tmem_dealloc_mbar: cutlass.Int64
    tmem_holding_buf: cutlass.Int32


def s2t_copy_and_partition(sSF, tSF, sf_dt):

    tCsSF_compact = cute.filter_zeros(sSF)
    tCtSF_compact = cute.filter_zeros(tSF)
    copy_atom_s2t = cute.make_copy_atom(
        tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.TWO), sf_dt
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
    tiled_mma_sfb: cute.TiledMma,
    tma_atom_a: cute.CopyAtom, mA: cute.Tensor,
    tma_atom_b: cute.CopyAtom, mB: cute.Tensor,
    tma_atom_sfa: cute.CopyAtom, mSFA: cute.Tensor,
    tma_atom_sfb: cute.CopyAtom, mSFB: cute.Tensor,
    mC: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
    sfa_smem_layout: cute.Layout,
    sfb_smem_layout: cute.Layout,
    mma_tiler_mnk: cutlass.Constexpr,
    mma_tiler_sfb: cutlass.Constexpr,
    cluster_layout_vmnk: cute.Layout,
    cluster_layout_sfb_vmnk: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx, bidy, _ = cute.arch.block_idx()


    cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank)
    vmnk_sfb = cluster_layout_sfb_vmnk.get_flat_coord(cta_rank)
    mma_rank = bidx % cute.size(tiled_mma.thr_id.shape)
    is_leader = mma_rank == 0
    mma_coord_mnk = (bidx // 2, bidy, None)

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(ab_dtype, a_smem_layout.outer, 128, a_smem_layout.inner)
    sB = smem.allocate_tensor(ab_dtype, b_smem_layout.outer, 128, b_smem_layout.inner)
    sSFA = smem.allocate_tensor(sf_dtype, sfa_smem_layout, 128)
    sSFB = smem.allocate_tensor(sf_dtype, sfb_smem_layout, 128)

    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        cpasync.prefetch_descriptor(tma_atom_sfa)
        cpasync.prefetch_descriptor(tma_atom_sfb)


    tma_bytes = (
        cute.size_in_bytes(ab_dtype, a_smem_layout)
        + cute.size_in_bytes(ab_dtype, b_smem_layout)
        + cute.size_in_bytes(sf_dtype, sfa_smem_layout)
        + cute.size_in_bytes(sf_dtype, sfb_smem_layout)
    ) * cute.size(cluster_layout_vmnk, mode=[0])
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.ab_mbars.data_ptr(),
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        tx_count=tma_bytes,
        cta_layout_vmnk=cluster_layout_vmnk,
    ).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_mbars.data_ptr(),
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),

        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 8),
        cta_layout_vmnk=cluster_layout_vmnk,
    ).make_participants()


    alloc_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS_PER_CTA)
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr,
        barrier_for_retrieve=alloc_bar,
        allocator_warp_id=0,
        is_two_cta=True,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
    )


    pipeline_init_arrive(cluster_shape_mn=CLUSTER_SHAPE_MN, is_relaxed=True)
    tmem.allocate(TMEM_COLS)
    tmem.wait_for_alloc()
    acc_tmem_ptr = tmem.retrieve_ptr(acc_dtype)

    gA = cute.local_tile(mA, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    gB = cute.local_tile(mB, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
    gC = cute.local_tile(mC, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))
    gSFA = cute.local_tile(mSFA, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    gSFB = cute.local_tile(mSFB, mma_tiler_sfb, mma_coord_mnk, proj=(None, 1, 1))

    thr_mma = tiled_mma.get_slice(mma_rank)
    thr_mma_sfb = tiled_mma_sfb.get_slice(mma_rank)
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)
    tCgC = thr_mma.partition_C(gC)
    tCgSFA = thr_mma.partition_A(gSFA)
    tCgSFB = thr_mma_sfb.partition_B(gSFB)
    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)

    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc = cute.make_tensor(acc_tmem_ptr, tiled_mma.make_fragment_C(acc_shape).layout)
    tCtSFA = cute.make_tensor(
        cute.recast_ptr(acc_tmem_ptr + NUM_ACC_COLS, dtype=sf_dtype),
        blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma, mma_tiler_mnk, SF_VEC, sfa_smem_layout),
    )
    tCtSFB = cute.make_tensor(
        cute.recast_ptr(acc_tmem_ptr + NUM_ACC_COLS + NUM_SFA_COLS, dtype=sf_dtype),
        blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma, mma_tiler_mnk, SF_VEC, sfb_smem_layout),
    )


    # (2,1) cluster the B-side masks include both V peers, which is the shared

    a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
    b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
    sfb_cta_layout = cute.make_layout(cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape)
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a, vmnk[2], a_cta_layout,
        cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3))
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b, vmnk[1], b_cta_layout,
        cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3))
    tSFAsSFA, tSFAgSFA = cpasync.tma_partition(
        tma_atom_sfa, vmnk[2], a_cta_layout,
        cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3))
    tSFBsSFB, tSFBgSFB = cpasync.tma_partition(
        tma_atom_sfb, vmnk_sfb[1], sfb_cta_layout,
        cute.group_modes(sSFB, 0, 3), cute.group_modes(tCgSFB, 0, 3))
    tSFAsSFA, tSFAgSFA = cute.filter_zeros(tSFAsSFA), cute.filter_zeros(tSFAgSFA)
    tSFBsSFB, tSFBgSFB = cute.filter_zeros(tSFBsSFB), cute.filter_zeros(tSFBgSFB)

    mask_a = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, vmnk, mcast_mode=2)
    mask_b = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, vmnk, mcast_mode=1)
    mask_sfa = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, vmnk, mcast_mode=2)
    mask_sfb = cpasync.create_tma_multicast_mask(cluster_layout_sfb_vmnk, vmnk_sfb, mcast_mode=1)
    s2t_sfa, sfa_src, sfa_dst = s2t_copy_and_partition(sSFA, tCtSFA, sf_dtype)
    s2t_sfb, sfb_src, sfb_dst = s2t_copy_and_partition(sSFB, tCtSFB, sf_dtype)

    pipeline_init_wait(cluster_shape_mn=CLUSTER_SHAPE_MN)
    num_k_tiles = cute.size(gA, mode=[2])
    if warp_idx == 0:
        acc_empty = acc_producer.acquire_and_advance()
        for k_tile in cutlass.range(num_k_tiles):
            empty = ab_producer.acquire_and_advance()
            cute.copy(tma_atom_a, tAgA[(None, k_tile)], tAsA,
                      tma_bar_ptr=empty.barrier, mcast_mask=mask_a)
            cute.copy(tma_atom_b, tBgB[(None, k_tile)], tBsB,
                      tma_bar_ptr=empty.barrier, mcast_mask=mask_b)
            cute.copy(tma_atom_sfa, tSFAgSFA[(None, k_tile)], tSFAsSFA,
                      tma_bar_ptr=empty.barrier, mcast_mask=mask_sfa)
            cute.copy(tma_atom_sfb, tSFBgSFB[(None, k_tile)], tSFBsSFB,
                      tma_bar_ptr=empty.barrier, mcast_mask=mask_sfb)


            if is_leader:
                full = ab_consumer.wait_and_advance()
                cute.copy(s2t_sfa, sfa_src, sfa_dst)
                cute.copy(s2t_sfb, sfb_src, sfb_dst)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                cute.gemm(tiled_mma, tCtAcc,
                          [tCrA, tCtSFA], [tCrB, tCtSFB], tCtAcc)
                full.release()
        ab_producer.tail()
        if is_leader:
            acc_empty.commit()


    acc_full = acc_consumer.wait_and_advance()
    epi_tiler = ((cute.size(tCtAcc, mode=[0, 0]),
                  cute.size(tCtAcc, mode=[0, 1]) // SUBTILE_CNT),)
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    gC_epi = cute.zipped_divide(tCgC, epi_tiler)
    tmem_atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), acc_dtype)
    tmem_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    thr_copy = tmem_copy.get_slice(tidx)
    tDtC = thr_copy.partition_S(tCtAcc_epi)
    tDgC = thr_copy.partition_D(gC_epi)
    frg = cute.make_rmem_tensor(tDgC[None, None, 0].shape, acc_dtype)
    for i in cutlass.range(cute.size(tDtC, mode=[2])):
        cute.copy(tmem_copy, tDtC[None, None, i], frg)
        cute.autovec_copy(frg, tDgC[None, None, i])
    with cute.arch.elect_one():
        acc_full.release()

    cute.arch.barrier()
    if warp_idx == 0 and is_leader:
        acc_producer.tail()
    tmem.relinquish_alloc_permit()
    tmem.free(acc_tmem_ptr)
    cute.arch.cluster_arrive_relaxed()
    cute.arch.cluster_wait()


@cute.jit
def host(a: cute.Tensor, b: cute.Tensor, sfa_raw: cute.Tensor,
         sfb_raw: cute.Tensor, c: cute.Tensor):
    tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
        ab_dtype, ab_dtype,
        cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
        sf_dtype, SF_VEC, tcgen05.CtaGroup.TWO, mma_tiler_mn)
    tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
        ab_dtype, ab_dtype,
        cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
        sf_dtype, SF_VEC, tcgen05.CtaGroup.ONE, mma_tiler_sfb_mn)
    mma_inst_k = cute.size(tiled_mma.shape_mnk, mode=[2])
    mma_tiler_mnk = (*mma_tiler_mn, mma_inst_k * MMA_INST_TILE_K)
    mma_tiler_sfb = (*mma_tiler_sfb_mn, mma_inst_k * MMA_INST_TILE_K)

    sfa = cute.make_tensor(
        sfa_raw.iterator, blockscaled_utils.tile_atom_to_shape_SF((*a.shape, 1), SF_VEC)
    )[None, None, 0]
    sfb = cute.make_tensor(
        sfb_raw.iterator, blockscaled_utils.tile_atom_to_shape_SF((*b.shape, 1), SF_VEC)
    )[None, None, 0]
    a_smem = cute.select(sm100_utils.make_smem_layout_a(
        tiled_mma, mma_tiler_mnk, ab_dtype, 1), mode=[0, 1, 2])
    b_smem = cute.select(sm100_utils.make_smem_layout_b(
        tiled_mma, mma_tiler_mnk, ab_dtype, 1), mode=[0, 1, 2])
    sfa_smem = cute.slice_(blockscaled_utils.make_smem_layout_sfa(
        tiled_mma, mma_tiler_mnk, SF_VEC, 1), (None, None, None, 0))
    sfb_smem = cute.slice_(blockscaled_utils.make_smem_layout_sfb(
        tiled_mma, mma_tiler_mnk, SF_VEC, 1), (None, None, None, 0))

    cluster_layout = cute.tiled_divide(
        cute.make_layout((*CLUSTER_SHAPE_MN, 1)), (tiled_mma.thr_id.shape,))
    cluster_layout_sfb = cute.tiled_divide(
        cute.make_layout((*CLUSTER_SHAPE_MN, 1)), (tiled_mma_sfb.thr_id.shape,))
    op_a = sm100_utils.cluster_shape_to_tma_atom_A(CLUSTER_SHAPE_MN, tiled_mma.thr_id)
    op_b = sm100_utils.cluster_shape_to_tma_atom_B(CLUSTER_SHAPE_MN, tiled_mma.thr_id)
    op_sfb = sm100_utils.cluster_shape_to_tma_atom_SFB(CLUSTER_SHAPE_MN, tiled_mma.thr_id)
    ta, ga = cute.nvgpu.make_tiled_tma_atom_A(
        op_a, a, a_smem, mma_tiler_mnk, tiled_mma, cluster_layout.shape)
    tb, gb = cute.nvgpu.make_tiled_tma_atom_B(
        op_b, b, b_smem, mma_tiler_mnk, tiled_mma, cluster_layout.shape)
    tsa, gsa = cute.nvgpu.make_tiled_tma_atom_A(
        op_a, sfa, sfa_smem, mma_tiler_mnk, tiled_mma, cluster_layout.shape,
        internal_type=cutlass.Int16)
    tsb, gsb = cute.nvgpu.make_tiled_tma_atom_B(
        op_sfb, sfb, sfb_smem, mma_tiler_sfb, tiled_mma_sfb,
        cluster_layout_sfb.shape, internal_type=cutlass.Int16)

    grid = (cute.ceil_div(c.shape[0], mma_tiler_mn[0]) * 2,
            cute.ceil_div(c.shape[1], mma_tiler_mn[1]), 1)
    kernel(tiled_mma, tiled_mma_sfb,
           ta, ga, tb, gb, tsa, gsa, tsb, gsb, c,
           a_smem, b_smem, sfa_smem, sfb_smem,
           mma_tiler_mnk, mma_tiler_sfb, cluster_layout, cluster_layout_sfb).launch(
        grid=grid, block=(THREADS_PER_CTA, 1, 1), cluster=(*CLUSTER_SHAPE_MN, 1))


TILE_M_CHECK, TILE_N_CHECK, TILE_K_CHECK = 256, 128, 128


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


    top_written = c[:M // 2].abs().sum().item() > 0
    bottom_written = c[M // 2:].abs().sum().item() > 0
    ok = err < 5e-2 and top_written and bottom_written
    print(f"  ({M:5d},{N:5d},{K:5d}) -> {'ok' if ok else 'MISMATCH'}  "
          f"rel_err={err:.3e} halves_written=({top_written},{bottom_written})")
    return ok


def main():
    if all(run_one(*s) for s in [(256, 256, 256), (512, 256, 512)]):
        print("PASS")
    else:
        print("FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
