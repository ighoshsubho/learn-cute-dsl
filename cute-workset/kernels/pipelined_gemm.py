from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait


io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32

cluster_shape_mnk = (1, 1, 1)
mma_inst_shape_mnk = (128, 128, 16)
mma_tiler_mnk = (128, 128, 64)
threads_in_epilogue = 128


ab_stages = 4
epi_stages = 2
acc_stages = 2

scheduler_type = utils.StaticPersistentTileScheduler


@cute.struct
class SharedStorage:

    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stages * 2]
    tmem_holding_buffer: cutlass.Int32


@cute.kernel()
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    tma_atom_c: cute.CopyAtom,
    mC_mnl: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
    c_smem_layout_kind: cutlass.Constexpr,
    epi_smem_layout_staged: cute.ComposedLayout,
    epi_tile: cute.Tile,
    cta_layout_vmnk: cute.Layout,
    tile_sched_params: utils.PersistentTileSchedulerParams,
):
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster)


    mma_tile_coord_v = bidx % cute.size(cta_layout_vmnk, mode=[0])
    is_leader_cta = mma_tile_coord_v == 0


    epilogue_warp_ids = (0, 1, 2, 3)
    mma_warp_id = 4
    tma_warp_id = 5

    epilog_sync_bar_id = 1
    tmem_alloc_sync_bar_id = 2


    if warp_idx == tma_warp_id:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        cpasync.prefetch_descriptor(tma_atom_c)


    num_mcast_participants = (
        cute.size(cta_layout_vmnk, mode=[1]) + cute.size(cta_layout_vmnk, mode=[2]) - 1
    )
    tma_mcast_mask_a = cute.nvgpu.cpasync.create_tma_multicast_mask(
        cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=2
    )
    tma_mcast_mask_b = cute.nvgpu.cpasync.create_tma_multicast_mask(
        cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=1
    )


    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)


    epilogue_sync_barrier = pipeline.NamedBarrier(
        barrier_id=epilog_sync_bar_id,
        num_threads=threads_in_epilogue,
    )


    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=tmem_alloc_sync_bar_id,
        num_threads=32 * len((mma_warp_id, *epilogue_warp_ids)),
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buffer.ptr,
        barrier_for_retrieve=tmem_alloc_barrier,
        allocator_warp_id=epilogue_warp_ids[0],
        is_two_cta=False,
    )


    num_tma_copy_bytes = (
        cute.size_in_bytes(io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2]))
        + cute.size_in_bytes(io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))
    ) * cute.size(cta_layout_vmnk, mode=[0])


    mainloop_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread, size=num_mcast_participants
    )

    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
        num_stages=ab_stages,
        producer_group=mainloop_pipeline_producer_group,
        consumer_group=mainloop_pipeline_consumer_group,
        tx_count=num_tma_copy_bytes,
        cta_layout_vmnk=cta_layout_vmnk,
    ).make_participants()


    acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    acc_pipeline_consumer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread,
        size=cute.size(cta_layout_vmnk, mode=[0]) * len(epilogue_warp_ids),
    )
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
        num_stages=acc_stages,
        producer_group=acc_pipeline_producer_group,
        consumer_group=acc_pipeline_consumer_group,
        cta_layout_vmnk=cta_layout_vmnk,
    ).make_participants()


    pipeline_init_arrive(cluster_shape_mn=cluster_shape_mnk, is_relaxed=True)


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
    sC = smem.allocate_tensor(
        element_type=io_dtype,
        layout=epi_smem_layout_staged.outer,
        byte_alignment=128,
        swizzle=epi_smem_layout_staged.inner,
    )


    # (bM, bK, RestM, RestK)
    gA = cute.local_tile(
        mA_mkl, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None)
    )
    # (bN, bK, RestN, RestK)
    gB = cute.local_tile(
        mB_nkl, cute.slice_(mma_tiler_mnk, (0, None, None)), (None, None)
    )
    # (bM, bN, RestM, RestN)
    gC = cute.local_tile(
        mC_mnl, cute.slice_(mma_tiler_mnk, (None, None, 0)), (None, None)
    )

    thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
    tCgA = thr_mma.partition_A(gA)  # (MMA, MMA_M, MMA_K, RestM, RestK)
    tCgB = thr_mma.partition_B(gB)  # (MMA, MMA_N, MMA_K, RestN, RestK)
    tCgC = thr_mma.partition_C(gC)  # (MMA, MMA_M, MMA_N, RestM, RestN)


    tCrA = tiled_mma.make_fragment_A(sA)  # (MMA, MMA_M, MMA_K, STAGE)
    tCrB = tiled_mma.make_fragment_B(sB)  # (MMA, MMA_N, MMA_K, STAGE)


    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, acc_stages))


    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a,
        cta_in_cluster_coord_vmnk[2],
        cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )  # ((atom_v, rest_v), STAGE), ((atom_v, rest_v), RestM, RestK)
    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
        tma_atom_b,
        cta_in_cluster_coord_vmnk[1],
        cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )  # ((atom_v, rest_v), STAGE), ((atom_v, rest_v), RestN, RestK)

    gC_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None)], epi_tile)
    tCsC, tCgC_tma = cute.nvgpu.cpasync.tma_partition(
        tma_atom_c,
        0,
        cute.make_layout(1),
        cute.group_modes(sC, 0, 2),
        cute.group_modes(gC_epi, 0, 2),
    )

    pipeline_init_wait(cluster_shape_mn=cluster_shape_mnk)


    tile_sched = scheduler_type.create(
        tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
    )
    work_tile = tile_sched.initial_work_tile_info()

    num_k_tiles = cute.size(gA, mode=[3])


    if warp_idx == tma_warp_id:
        while work_tile.is_valid_tile:
            cur_tile_coord = work_tile.tile_idx
            mma_tile_coord_mnl = (
                cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                cur_tile_coord[1],
                cur_tile_coord[2],
            )
            # ((atom_v, rest_v), RestK) for this output tile's row / column
            tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None)]
            tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None)]


            for k_tile_idx in range(num_k_tiles):
                handle = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a,
                    tAgA_slice[(None, k_tile_idx)],
                    tAsA[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                    mcast_mask=tma_mcast_mask_a,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_slice[(None, k_tile_idx)],
                    tBsB[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                    mcast_mask=tma_mcast_mask_b,
                )


            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()


        ab_producer.tail()


    elif warp_idx == mma_warp_id:
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(acc_dtype)
        tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        while work_tile.is_valid_tile:
            if is_leader_cta:

                acc_empty = acc_producer.acquire_and_advance()
                tCtAcc = tCtAcc_base[(None, None, None, acc_empty.index)]


                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_tile_idx in range(num_k_tiles):
                    handle = ab_consumer.wait_and_advance()

                    num_k_blocks = cute.size(tCrA, mode=[2])
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (None, None, k_block_idx, handle.index)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            tCtAcc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                    handle.release()


                acc_empty.commit()


            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()


        acc_producer.tail()


    elif warp_idx < mma_warp_id:

        num_tmem_cols = 512
        tmem.allocate(num_tmem_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(acc_dtype)
        tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        epilogue_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            size=128,
        )
        epilogue_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=epi_stages,
            producer_group=epilogue_pipeline_producer_group,
        )

        copy_atom_t2r = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x32, tcgen05.Pack.NONE),
            cutlass.Float32,
        )

        while work_tile.is_valid_tile:
            cur_tile_coord = work_tile.tile_idx
            mma_tile_coord_mnl = (
                cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                cur_tile_coord[1],
                cur_tile_coord[2],
            )


            acc_full = acc_consumer.wait_and_advance()
            tCtAcc = tCtAcc_base[(None, None, None, acc_full.index)]


            tCtAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tile)
            mma_tile_coord_mn = cute.slice_(mma_tile_coord_mnl, (None, None, 0))
            tCgC_epi = cute.flat_divide(
                tCgC[((None, None), 0, 0, *mma_tile_coord_mn)], epi_tile
            )
            tCgC_tma_cur_tile = tCgC_tma[(None, None, None, *mma_tile_coord_mn)]


            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tCtAcc_epi[(None, None, 0, 0)]
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc_epi)
            tTR_gC = thr_copy_t2r.partition_D(tCgC_epi)
            tTR_rAcc = cute.make_rmem_tensor(
                tTR_gC[(None, None, None, 0, 0)].shape, cutlass.Float32
            )
            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))


            copy_atom_r2s = cutlass.utils.blackwell_helpers.get_smem_store_op(
                c_smem_layout_kind, cutlass.Float32, cutlass.Float32, tiled_copy_t2r
            )
            tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
            thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
            tRS_sC = thr_copy_r2s.partition_D(sC)

            tRS_rAcc = tiled_copy_r2s.retile(tTR_rAcc)
            tRS_rC = cute.make_rmem_tensor(tRS_rAcc.shape, io_dtype)
            tCgC_grouped = cute.group_modes(
                tCgC_tma_cur_tile, 1, cute.rank(tCgC_tma_cur_tile)
            )

            subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])

            for subtile_idx in cutlass.range(subtile_cnt):

                tTR_tAcc_slice = tTR_tAcc[(None, None, None, subtile_idx)]
                cute.copy(tiled_copy_t2r, tTR_tAcc_slice, tTR_rAcc)


                c_buffer = subtile_idx % epi_stages
                tRS_sC_slice = tRS_sC[(None, None, None, c_buffer)]
                tRS_rC.store(tRS_rAcc.load().to(io_dtype))
                cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC_slice)


                cute.arch.fence_view_async_shared()
                epilogue_sync_barrier.arrive_and_wait()
                if warp_idx == epilogue_warp_ids[0]:
                    cute.copy(
                        tma_atom_c,
                        tCsC[(None, c_buffer)],
                        tCgC_grouped[(None, subtile_idx)],
                    )
                    epilogue_pipeline.producer_commit()
                    epilogue_pipeline.producer_acquire()
                epilogue_sync_barrier.arrive_and_wait()


            with cute.arch.elect_one():
                acc_full.release()

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()


        epilogue_pipeline.producer_tail()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)


def compute_grid(
    c: cute.Tensor,
    cta_tiler_mnk: Tuple[int, int, int],
    cluster_shape_mnk: Tuple[int, int, int],
    max_active_clusters: cutlass.Constexpr,
):
    c_shape = cute.slice_(cta_tiler_mnk, (None, None, 0))
    gc = cute.zipped_divide(c, tiler=c_shape)
    num_ctas_mn = gc[(0, (None, None))].shape

    tile_sched_params = utils.PersistentTileSchedulerParams(
        (*num_ctas_mn, 1), cluster_shape_mnk
    )
    grid = utils.StaticPersistentTileScheduler.get_grid_shape(
        tile_sched_params, max_active_clusters
    )
    return tile_sched_params, grid


@cute.jit
def host_function(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    max_active_clusters: cutlass.Constexpr,
):
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


    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma, mma_tiler_mnk, a.element_type, ab_stages
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma, mma_tiler_mnk, b.element_type, ab_stages
    )
    c_smem_layout_kind = utils.LayoutEnum.from_tensor(c)

    cta_layout_mnk = cute.make_layout(cluster_shape_mnk)
    cta_layout_vmnk = cute.tiled_divide(cta_layout_mnk, (tiled_mma.thr_id,))


    op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.ONE)
    a_smem_layout_slice = cute.slice_(a_smem_layout, (None, None, None, 0))
    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        op, a, a_smem_layout_slice, mma_tiler_mnk, tiled_mma, cta_layout_vmnk.shape
    )
    b_smem_layout_slice = cute.slice_(b_smem_layout, (None, None, None, 0))
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        op, b, b_smem_layout_slice, mma_tiler_mnk, tiled_mma, cta_layout_vmnk.shape
    )

    cta_tile_shape_mnk = (
        mma_tiler_mnk[0] // cute.size(tiled_mma.thr_id),
        mma_tiler_mnk[1],
        mma_tiler_mnk[2],
    )

    epi_tile = utils.compute_epilogue_tile_shape(
        cta_tile_shape_mnk, False, c_smem_layout_kind, io_dtype
    )
    epi_smem_layout_staged = cutlass.utils.blackwell_helpers.make_smem_layout_epi(
        io_dtype, c_smem_layout_kind, epi_tile, epi_stages
    )
    epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
    c_tma_atom, c_tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, epi_tile
    )

    tile_sched_params, grid_shape = compute_grid(
        c, cta_tile_shape_mnk, cluster_shape_mnk, max_active_clusters
    )

    kernel(
        tiled_mma,
        a_tma_atom,
        a_tma_tensor,
        b_tma_atom,
        b_tma_tensor,
        c_tma_atom,
        c_tma_tensor,
        a_smem_layout,
        b_smem_layout,
        c_smem_layout_kind,
        epi_smem_layout_staged,
        epi_tile,
        cta_layout_vmnk,
        tile_sched_params,
    ).launch(
        grid=grid_shape,
        block=[192, 1, 1],


    )


def main():
    import torch
    import cutlass.torch as cutlass_torch

    torch.manual_seed(2024)
    torch.zeros(1, device="cuda")
    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
        cluster_shape_mnk[0] * cluster_shape_mnk[1]
    )

    def make_kmajor(rows, k):
        return (
            torch.empty(rows, k, dtype=torch.int32)
            .random_(-2, 2)
            .to(device="cuda", dtype=cutlass_torch.dtype(io_dtype))
        )


    shapes = [(256, 512, 256), (1024, 768, 512), (2048, 2048, 1024)]
    all_ok = True
    compiled = None
    for m, n, k in shapes:
        a = make_kmajor(m, k)
        b = make_kmajor(n, k)
        c = torch.zeros(m, n, device="cuda", dtype=cutlass_torch.dtype(io_dtype))
        a_memref = from_dlpack(a).mark_layout_dynamic()
        b_memref = from_dlpack(b).mark_layout_dynamic()
        c_memref = from_dlpack(c).mark_layout_dynamic()
        if compiled is None:


            compiled = cute.compile(
                host_function, a_memref, b_memref, c_memref, max_active_clusters
            )
        compiled(a_memref, b_memref, c_memref)
        torch.cuda.synchronize()

        ref = torch.einsum("mk,nk->mn", a.float(), b.float())
        err = (c.float() - ref).abs().max().item()
        ok = err <= 1e-1
        all_ok &= ok
        print(f"  mnk=({m:5d},{n:5d},{k:5d})  max_abs_err={err:.3e}  "
              f"{'OK' if ok else 'FAIL'}")


    m, n, k = shapes[-1]
    a = make_kmajor(m, k)
    b = make_kmajor(n, k)
    c = torch.zeros(m, n, device="cuda", dtype=cutlass_torch.dtype(io_dtype))
    a_memref = from_dlpack(a).mark_layout_dynamic()
    b_memref = from_dlpack(b).mark_layout_dynamic()
    c_memref = from_dlpack(c).mark_layout_dynamic()
    for _ in range(5):
        compiled(a_memref, b_memref, c_memref)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    iters = 50
    start.record()
    for _ in range(iters):
        compiled(a_memref, b_memref, c_memref)
    end.record()
    torch.cuda.synchronize()
    t = start.elapsed_time(end) / iters / 1e3
    tflops = 2 * m * n * k / t / 1e12
    print(f"  perf mnk=({m},{n},{k}): {t * 1e6:.1f} us  ->  {tflops:.1f} TFLOPS")

    print("PASS" if all_ok else "FAIL")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
