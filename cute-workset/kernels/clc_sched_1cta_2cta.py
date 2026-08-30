# Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import argparse
import os
import subprocess
import sys
from typing import Tuple, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait


io_dtype = cutlass.BFloat16
acc_dtype = cutlass.Float32


use_2cta_instrs = os.environ.get("CUTE_ADV06_2CTA", "1") == "1"
cluster_shape_mnk = (2, 1, 1) if use_2cta_instrs else (1, 1, 1)
mma_inst_shape_mnk = (256 if use_2cta_instrs else 128, 128, 16)
mma_tiler_mnk = (256 if use_2cta_instrs else 128, 128, 64)
c_dtype = cutlass.Float32
threads_in_epilogue = 128


ab_stages = 6
epi_stages = 2
acc_stages = 2
num_clc_stage = 1


use_clc_dynamic_scheduler = True
scheduler_type = (
    utils.ClcDynamicPersistentTileScheduler
    if use_clc_dynamic_scheduler
    else utils.StaticPersistentTileScheduler
)

num_clc_response_bytes = 16


@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stages * 2]
    tmem_dealloc_mbar: cutlass.Int64
    tmem_holding_buffer: cutlass.Int32

    clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    clc_response_align_bytes = num_clc_response_bytes
    clc_response: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int32, 4],
        clc_response_align_bytes,
    ]


@cute.kernel()
def kernel(
    tiled_mma: cute.TiledMma,
    tma_a: cpasync.TmaInfo,
    tma_b: cpasync.TmaInfo,
    tma_c: cpasync.TmaInfo,
    c_smem_layout_kind: cutlass.Constexpr,
    epi_tile: cute.Tile,
    cta_layout_vmnk: cute.Layout,
    tile_sched_params: Union[
        utils.ClcDynamicPersistentTileSchedulerParams,
        utils.PersistentTileSchedulerParams,
    ],
    mTileCount: cute.Tensor,
):

    mA_mkl = tma_a.tma_tensor
    mB_nkl = tma_b.tma_tensor
    mC_mnl = tma_c.tma_tensor

    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()

    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

    mma_tile_coord_v = bidx % cute.size(cta_layout_vmnk, mode=[0])
    is_leader_cta = mma_tile_coord_v == 0

    epilogue_warp_ids = (
        0,
        1,
        2,
        3,
    )
    mma_warp_id = 4
    tma_warp_id = 5

    sched_warp_id = 6

    epilog_sync_bar_id = 1
    tmem_alloc_sync_bar_id = 2


    if warp_idx == tma_warp_id:
        cpasync.prefetch_descriptor(tma_a.atom)
        cpasync.prefetch_descriptor(tma_b.atom)
        cpasync.prefetch_descriptor(tma_c.atom)


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
        num_threads=32
        * len((mma_warp_id, *epilogue_warp_ids)),
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buffer.ptr,
        barrier_for_retrieve=tmem_alloc_barrier,
        allocator_warp_id=epilogue_warp_ids[0],
        is_two_cta=use_2cta_instrs,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
    )

    num_tma_copy_bytes = (
        cute.size_in_bytes(io_dtype, cute.select(tma_a.smem_layout, mode=[0, 1, 2]))
        + cute.size_in_bytes(io_dtype, cute.select(tma_b.smem_layout, mode=[0, 1, 2]))
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


    if cutlass.const_expr(use_clc_dynamic_scheduler):
        clc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        cluster_size = cute.size(cluster_shape_mnk)
        num_clc_consumer_threads = 32 * len(
            (
                sched_warp_id,
                *(
                    cluster_size
                    * (
                        mma_warp_id,
                        tma_warp_id,
                        *epilogue_warp_ids,
                    )
                ),
            )
        )
        clc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_clc_consumer_threads
        )
        clc_pipeline = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=storage.clc_mbar_ptr.data_ptr(),
            num_stages=num_clc_stage,
            producer_group=clc_pipeline_producer_group,
            consumer_group=clc_pipeline_consumer_group,
            tx_count=num_clc_response_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

        clc_response_ptr = storage.clc_response.data_ptr()

        clc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, num_clc_stage
        )
    else:
        clc_pipeline = None
        clc_response_ptr = None
        clc_consumer_state = None


    pipeline_init_arrive(cluster_shape_mn=cluster_shape_mnk, is_relaxed=True)


    sA = smem.allocate_tensor(
        element_type=io_dtype,
        layout=tma_a.smem_layout.outer,
        byte_alignment=128,
        swizzle=tma_a.smem_layout.inner,
    )
    sB = smem.allocate_tensor(
        element_type=io_dtype,
        layout=tma_b.smem_layout.outer,
        byte_alignment=128,
        swizzle=tma_b.smem_layout.inner,
    )
    sC = smem.allocate_tensor(
        element_type=c_dtype,
        layout=tma_c.smem_layout.outer,
        byte_alignment=128,
        swizzle=tma_c.smem_layout.inner,
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
    # (MMA, MMA_M, MMA_K, RestM, RestK)
    tCgA = thr_mma.partition_A(gA)
    # (MMA, MMA_N, MMA_K, RestN, RestK)
    tCgB = thr_mma.partition_B(gB)
    # (MMA, MMA_M, MMA_N, RestM, RestN)
    tCgC = thr_mma.partition_C(gC)

    # (MMA, MMA_M, MMA_K, STAGE)
    tCrA = tiled_mma.make_fragment_A(sA)
    # (MMA, MMA_N, MMA_K, STAGE)
    tCrB = tiled_mma.make_fragment_B(sB)

    # (MMA, MMA_M, MMA_N)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    # (MMA, MMA_M, MMA_N, STAGE)
    tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, acc_stages))


    # ((atom_v, rest_v), STAGE)
    # ((atom_v, rest_v), RestM, RestK)
    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_a.atom,
        cta_in_cluster_coord_vmnk[2],
        cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )

    # ((atom_v, rest_v), STAGE)
    # ((atom_v, rest_v), RestN, RestK)
    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
        tma_b.atom,
        cta_in_cluster_coord_vmnk[1],
        cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )

    gC_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None)], epi_tile)

    tCsC, tCgC_tma = cute.nvgpu.cpasync.tma_partition(
        tma_c.atom,
        0,
        cute.make_layout(1),
        cute.group_modes(sC, 0, 2),
        cute.group_modes(gC_epi, 0, 2),
    )


    pipeline_init_wait(cluster_shape_mn=cluster_shape_mnk)


    if cutlass.const_expr(use_clc_dynamic_scheduler):
        tile_sched = scheduler_type.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            clc_response_ptr,
        )
    else:
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


            # ((atom_v, rest_v), RestK)
            tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None)]

            # ((atom_v, rest_v), RestK)
            tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None)]


            for k_tile_idx in range(num_k_tiles):

                handle = ab_producer.acquire_and_advance()


                cute.copy(
                    tma_a.atom,
                    tAgA_slice[(None, k_tile_idx)],
                    tAsA[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                    mcast_mask=tma_mcast_mask_a,
                )
                cute.copy(
                    tma_b.atom,
                    tBgB_slice[(None, k_tile_idx)],
                    tBsB[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                    mcast_mask=tma_mcast_mask_b,
                )


            if cutlass.const_expr(use_clc_dynamic_scheduler):
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            else:
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()


        ab_producer.tail()


    if cutlass.const_expr(use_clc_dynamic_scheduler):
        is_first_cta_in_cluster = cta_rank_in_cluster == 0

        if warp_idx == sched_warp_id and is_first_cta_in_cluster:

            clc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.ProducerConsumer, num_clc_stage
            )
            while work_tile.is_valid_tile:

                clc_pipeline.producer_acquire(clc_producer_state)
                mbarrier_addr = clc_pipeline.producer_get_barrier(clc_producer_state)
                tile_sched.advance_to_next_work(mbarrier_addr)
                clc_producer_state.advance()

                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            clc_pipeline.producer_tail(clc_producer_state)


            if tidx % 32 == 0:
                gridx, gridy, _ = cute.arch.grid_dim()
                linear_cta = bidx + gridx * (bidy + gridy * bidz)


                processed = tile_sched.num_tiles_executed
                mTileCount[linear_cta] = processed
                if cutlass.const_expr(use_2cta_instrs):
                    mTileCount[linear_cta + 1] = processed


    if warp_idx == mma_warp_id:

        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(acc_dtype)
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        while work_tile.is_valid_tile:
            if is_leader_cta:

                acc_empty = acc_producer.acquire_and_advance()


                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_empty.index)]

                for k_tile_idx in range(num_k_tiles):

                    handle = ab_consumer.wait_and_advance()


                    tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile_idx != 0)
                    tile_crd = (None, None, None, handle.index)
                    cute.gemm(tiled_mma, tCtAcc, tCrA[tile_crd], tCrB[tile_crd], tCtAcc)


                    handle.release()


                acc_empty.commit()


            if cutlass.const_expr(use_clc_dynamic_scheduler):
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            else:
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()


        acc_producer.tail()


    if warp_idx < mma_warp_id:

        num_tmem_cols = 512
        tmem.allocate(num_tmem_cols)


        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(acc_dtype)
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)


        epilogue_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            size=threads_in_epilogue,
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


            # (MMA, MMA_M, MMA_N)
            tCtAcc = tCtAcc_base[(None, None, None, acc_full.index)]

            # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N)
            tCtAcc_epi = cute.flat_divide(
                tCtAcc[((None, None), 0, 0)],
                epi_tile,
            )

            mma_tile_coord_mn = cute.slice_(mma_tile_coord_mnl, (None, None, 0))
            # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN)
            tCgC_epi = cute.flat_divide(
                tCgC[((None, None), 0, 0, *mma_tile_coord_mn)], epi_tile
            )

            tCgC_tma_cur_tile = tCgC_tma[(None, None, None, *mma_tile_coord_mn)]


            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tCtAcc_epi[(None, None, 0, 0)]
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
            tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc_epi)
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
            tTR_gC = thr_copy_t2r.partition_D(tCgC_epi)
            # (T2R, T2R_M, T2R_N)
            tTR_rAcc = cute.make_rmem_tensor(
                tTR_gC[(None, None, None, 0, 0)].shape, cutlass.Float32
            )
            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))


            copy_atom_r2s = cutlass.utils.blackwell_helpers.get_smem_store_op(
                c_smem_layout_kind, cutlass.Float32, cutlass.Float32, tiled_copy_t2r
            )
            tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
            # (R2S, R2S_M, R2S_N, PIPE_D)
            thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
            tRS_sC = thr_copy_r2s.partition_D(sC)

            tRS_rAcc = tiled_copy_r2s.retile(tTR_rAcc)
            tRS_rC = cute.make_rmem_tensor(tRS_rAcc.shape, c_dtype)
            tCgC_grouped = cute.group_modes(
                tCgC_tma_cur_tile, 1, cute.rank(tCgC_tma_cur_tile)
            )

            subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])


            for subtile_idx in cutlass.range(subtile_cnt):

                tTR_tAcc_slice = tTR_tAcc[(None, None, None, subtile_idx)]
                cute.copy(tiled_copy_t2r, tTR_tAcc_slice, tTR_rAcc)


                c_buffer = subtile_idx % epi_stages
                tRS_sC_slice = tRS_sC[(None, None, None, c_buffer)]


                tRS_rC.store(tRS_rAcc.load().to(c_dtype))

                cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC_slice)


                cute.arch.fence_view_async_shared()
                epilogue_sync_barrier.arrive_and_wait()

                if warp_idx == epilogue_warp_ids[0]:
                    cute.copy(
                        tma_c.atom,
                        tCsC[(None, c_buffer)],
                        tCgC_grouped[(None, subtile_idx)],
                    )

                    epilogue_pipeline.producer_commit()
                    epilogue_pipeline.producer_acquire()
                epilogue_sync_barrier.arrive_and_wait()


            with cute.arch.elect_one():
                acc_full.release()


            if cutlass.const_expr(use_clc_dynamic_scheduler):
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            else:
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()


        epilogue_pipeline.producer_tail()


        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

def compute_grid(
    c: cute.Tensor,
    mma_tiler_mnk: Tuple[int, int, int],
    cluster_shape_mnk: Tuple[int, int, int],
    scheduler_type: Union[
        utils.StaticPersistentTileScheduler, utils.ClcDynamicPersistentTileScheduler
    ],
    max_active_clusters: cutlass.Constexpr,
) -> Tuple[
    Union[
        utils.ClcDynamicPersistentTileSchedulerParams,
        utils.PersistentTileSchedulerParams,
    ],
    Tuple[int, int, int],
]:
    c_shape = cute.slice_(mma_tiler_mnk, (None, None, 0))
    gc = cute.zipped_divide(c, tiler=c_shape)
    num_ctas_mn = gc[(0, (None, None))].shape

    if cutlass.const_expr(
        issubclass(scheduler_type, utils.ClcDynamicPersistentTileScheduler)
    ):
        tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
            (*num_ctas_mn, 1), cluster_shape_mnk
        )
        grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(tile_sched_params)
    else:
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
    tile_count: cute.Tensor,
    max_active_clusters: cutlass.Constexpr,
    stream: cuda.CUstream,
):


    op = tcgen05.MmaF16BF16Op(
        io_dtype,
        acc_dtype,
        mma_inst_shape_mnk,
        tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)


    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma,
        mma_tiler_mnk,
        a.element_type,
        ab_stages,
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma,
        mma_tiler_mnk,
        b.element_type,
        ab_stages,
    )


    c_smem_layout_kind = utils.LayoutEnum.from_tensor(c)


    cta_layout_mnk = cute.make_layout(cluster_shape_mnk)
    cta_layout_vmnk = cute.tiled_divide(cta_layout_mnk, (tiled_mma.thr_id,))


    op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp(
        tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
    )
    tma_a = cute.nvgpu.make_tiled_tma_atom_A(
        op,
        a,
        a_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cta_layout_vmnk.shape,
    )
    tma_b = cute.nvgpu.make_tiled_tma_atom_B(
        op,
        b,
        b_smem_layout,
        mma_tiler_mnk,
        tiled_mma,
        cta_layout_vmnk.shape,
    )

    cta_tile_shape_mnk = (
        mma_tiler_mnk[0] // cute.size(tiled_mma.thr_id),
        mma_tiler_mnk[1],
        mma_tiler_mnk[2],
    )

    epi_tile = utils.compute_epilogue_tile_shape(
        cta_tile_shape_mnk,
        use_2cta_instrs,
        c_smem_layout_kind,
        c_dtype,
    )

    epi_smem_layout_staged = cutlass.utils.blackwell_helpers.make_smem_layout_epi(
        c_dtype,
        c_smem_layout_kind,
        epi_tile,
        epi_stages,
    )

    tma_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
        c,
        epi_smem_layout_staged,
        epi_tile,
    )


    tile_sched_params, grid_shape = compute_grid(
        c,
        cta_tile_shape_mnk,
        cluster_shape_mnk,
        scheduler_type,
        max_active_clusters,
    )

    kernel(
        tiled_mma,
        tma_a,
        tma_b,
        tma_c,
        c_smem_layout_kind,
        epi_tile,
        cta_layout_vmnk,
        tile_sched_params,
        tile_count,
    ).launch(
        grid=grid_shape,
        block=[224, 1, 1] if use_clc_dynamic_scheduler else [192, 1, 1],
        stream=stream,
        **({"cluster": cluster_shape_mnk} if use_2cta_instrs else {}),
    )


def run_one(mnk: Tuple[int, int, int]) -> bool:

    import torch

    m, n, k = mnk
    torch.manual_seed(5)
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    c = torch.zeros(m, n, dtype=torch.float32, device="cuda")


    launched_ctas = (m // 128) * (n // 128)
    counts = torch.zeros(launched_ctas, dtype=torch.int32, device="cuda")
    am = from_dlpack(a, assumed_align=32).mark_layout_dynamic()
    bm = from_dlpack(b, assumed_align=32).mark_layout_dynamic()
    cm = from_dlpack(c, assumed_align=32).mark_layout_dynamic()
    countm = from_dlpack(counts, assumed_align=16)
    import cuda.bindings.driver as cuda_drv
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    max_clusters = utils.HardwareInfo().get_max_active_clusters(
        cluster_shape_mnk[0] * cluster_shape_mnk[1])


    compiled = cute.compile(
        host_function, am, bm, cm, countm, max_clusters, stream)
    compiled(am, bm, cm, countm, stream)
    torch.cuda.synchronize()
    ref = torch.einsum("mk,nk->mn", a.float(), b.float())
    rel_err = (c - ref).abs().max().item() / max(1.0, ref.abs().max().item())

    for _ in range(3):
        compiled(am, bm, cm, countm, stream)
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(10):
        compiled(am, bm, cm, countm, stream)
    stop.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(stop) / 10 * 1e3
    tflops = 2 * m * n * k / (us * 1e-6) / 1e12

    active = counts[counts > 0].cpu()
    hist = {}
    for value in active.unique(sorted=True).tolist():
        hist[int(value)] = int((active == value).sum())
    ok = rel_err < 5e-2
    mode = "2cta" if use_2cta_instrs else "1cta"
    print(f"{'OK  ' if ok else 'FAIL'} {mode} ({m:5d},{n:5d},{k:5d}) "
          f"{us:9.2f} us {tflops:7.1f} TFLOPS rel_err={rel_err:.2e} "
          f"tiles/CTA={hist}")
    return ok


def worker_main() -> int:
    tile_m = 256 if use_2cta_instrs else 128
    shapes = [
        (1024, 1024, 512),
        (2048, 2048, 1024),


        (tile_m * 37, 640, 512),
    ]
    return 0 if all(run_one(shape) for shape in shapes) else 1


def main() -> None:
    if os.environ.get("CUTE_ADV06_WORKER") == "1":
        raise SystemExit(worker_main())


    for use_two in (False, True):
        env = os.environ.copy()
        env["CUTE_ADV06_WORKER"] = "1"
        env["CUTE_ADV06_2CTA"] = "1" if use_two else "0"
        result = subprocess.run([sys.executable, "-u", __file__], env=env)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    print("PASS")


if __name__ == "__main__":
    main()
