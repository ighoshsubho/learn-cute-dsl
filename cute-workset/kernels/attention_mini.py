from types import SimpleNamespace
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack
import cutlass.pipeline as pipeline


io_dtype = cutlass.BFloat16
m_block_size = 128
n_block_size = 128
num_threads = 128


class AttentionMini:


    def __init__(self, head_dim: int):
        assert head_dim in (64, 128)
        self._head_dim = head_dim
        self._num_threads = num_threads
        self._m_block_size = m_block_size
        self._n_block_size = n_block_size
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=num_threads
        )


    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self._dtype: Type[cutlass.Numeric] = mQ.element_type


        sQ_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3),
            0,
            cute.make_layout((8, 64), stride=(64, 1)),
        )
        sQ_layout = cute.tile_to_shape(
            sQ_layout_atom, (self._m_block_size, self._head_dim), (0, 1)
        )
        sKV_layout = cute.tile_to_shape(
            sQ_layout_atom, (self._n_block_size, self._head_dim), (0, 1)
        )
        sO_layout = sQ_layout

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]


        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self._dtype.width
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )

        tQKV_shape_dim_1 = 64 // async_copy_elems
        tQKV_layout = cute.make_layout(
            (self._num_threads // tQKV_shape_dim_1, tQKV_shape_dim_1),
            stride=(tQKV_shape_dim_1, 1),
        )
        vQKV_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_QKV = cute.make_tiled_copy_tv(
            atom_async_copy, tQKV_layout, vQKV_layout
        )
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy, tQKV_layout, vQKV_layout
        )


        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self._num_threads // 32, 1, 1),
            permutation_mnk=(self._num_threads // 32 * 16, 16, 16),
        )

        grid_dim = (
            cute.ceil_div(mQ.shape[1], self._m_block_size),
            cute.size(mQ.shape[0]),
            cute.size(mQ.shape[2]),
        )
        LOG2_E = 1.4426950408889634074
        softmax_scale_log2 = softmax_scale * LOG2_E
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            softmax_scale_log2,
            sQ_layout,
            sKV_layout,
            sO_layout,
            gmem_tiled_copy_QKV,
            gmem_tiled_copy_O,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self._num_threads, 1, 1],
            stream=stream,
        )


    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sKV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_QKV: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, batch_idx, head_idx = cute.arch.block_idx()

        n_block_max = cute.ceil_div(mK.shape[1], self._n_block_size)
        n_block = n_block_max - 1


        # (m_block_size, head_dim)
        gQ = cute.local_tile(
            mQ[batch_idx, None, head_idx, None],
            (self._m_block_size, self._head_dim),
            (m_block, 0),
        )
        # (n_block_size, head_dim, n_block)
        gK = cute.local_tile(
            mK[batch_idx, None, head_idx, None],
            (self._n_block_size, self._head_dim),
            (None, 0),
        )
        gV = cute.local_tile(
            mV[batch_idx, None, head_idx, None],
            (self._n_block_size, self._head_dim),
            (None, 0),
        )


        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sKV_layout)
        sV = storage.sV.get_tensor(sKV_layout)


        sVt = cute.composition(
            sV,
            cute.make_layout(
                (self._head_dim, self._n_block_size),
                stride=(self._n_block_size, 1),
            ),
        )

        gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(tidx)
        tQgQ = gmem_thr_copy_QKV.partition_S(gQ)
        tQsQ = gmem_thr_copy_QKV.partition_D(sQ)
        tKgK = gmem_thr_copy_QKV.partition_S(gK)  # (CPY, CPY_N, CPY_K, n_block)
        tKsK = gmem_thr_copy_QKV.partition_D(sK)
        tVgV = gmem_thr_copy_QKV.partition_S(gV)
        tVsV = gmem_thr_copy_QKV.partition_D(sV)


        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
        acc_shape_O = thr_mma.partition_shape_C(
            (self._m_block_size, self._head_dim)
        )
        acc_O = cute.make_rmem_tensor(acc_shape_O, cutlass.Float32)
        acc_O.fill(0.0)


        smem_copy_atom_QK = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self._dtype
        )
        smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_QK, tiled_mma)
        smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_QK, tiled_mma)
        smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)

        smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
        smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
        smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx)

        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tSrK_copy_view = smem_thr_copy_K.retile(tSrK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)
        tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)


        mcQ = cute.make_identity_tensor(mQ.layout.shape)
        mcKV = cute.make_identity_tensor(mK.layout.shape)
        cQ = cute.local_tile(
            mcQ[batch_idx, None, head_idx, None],
            (self._m_block_size, self._head_dim),
            (m_block, 0),
        )
        cKV = cute.local_tile(
            mcKV[batch_idx, None, head_idx, None],
            (self._n_block_size, self._head_dim),
            (n_block, 0),
        )
        tQcQ = gmem_thr_copy_QKV.partition_S(cQ)
        tKVcKV = gmem_thr_copy_QKV.partition_S(cKV)
        tQpQ = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tQsQ.shape[0][1],
                    cute.size(tQsQ, mode=[1]),
                    cute.size(tQsQ, mode=[2]),
                ),
                stride=(cute.size(tQsQ, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        tKVpKV = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tKsK.shape[0][1],
                    cute.size(tKsK, mode=[1]),
                    cute.size(tKsK, mode=[2]),
                ),
                stride=(cute.size(tKsK, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tQpQ.shape[0]):
            for rest_k in cutlass.range_constexpr(tQpQ.shape[2]):
                tQpQ[rest_v, 0, rest_k] = cute.elem_less(
                    tQcQ[(0, rest_v), 0, rest_k][3], mQ.layout.shape[3]
                )
        for rest_v in cutlass.range_constexpr(tKVpKV.shape[0]):
            for rest_k in cutlass.range_constexpr(tKVpKV.shape[2]):
                tKVpKV[rest_v, 0, rest_k] = cute.elem_less(
                    tKVcKV[(0, rest_v), 0, rest_k][3], mK.layout.shape[3]
                )


        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            if cute.elem_less(tQcQ[0, m, 0][1], mQ.layout.shape[1]):
                cute.copy(
                    gmem_tiled_copy_QKV,
                    tQgQ[None, m, None],
                    tQsQ[None, m, None],
                    pred=tQpQ[None, m, None],
                )
            else:
                tQsQ[None, m, None].fill(0)
        for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
            if cute.elem_less(tKVcKV[0, n, 0][1], mK.layout.shape[1]):
                cute.copy(
                    gmem_tiled_copy_QKV,
                    tKgK[None, n, None, n_block],
                    tKsK[None, n, None],
                    pred=tKVpKV[None, n, None],
                )
            else:
                tKsK[None, n, None].fill(0)
        cute.arch.cp_async_commit_group()


        row_max = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_sum = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        basic_params = SimpleNamespace(
            m_block=m_block,
            n_block=n_block,
            mQ=mQ,
            mK=mK,
            batch_idx=batch_idx,
            head_idx=head_idx,
        )
        mma_params = SimpleNamespace(
            thr_mma=thr_mma,
            tiled_mma=tiled_mma,
            tSrQ=tSrQ,
            tSrK=tSrK,
            tOrVt=tOrVt,
            acc_O=acc_O,
        )
        gmem_copy_params = SimpleNamespace(
            gmem_tiled_copy_QKV=gmem_tiled_copy_QKV,
            tKVcKV=tKVcKV,
            tKgK=tKgK,
            tKsK=tKsK,
            tVgV=tVgV,
            tVsV=tVsV,
            tKVpKV=tKVpKV,
        )
        smem_copy_params = SimpleNamespace(
            smem_tiled_copy_Q=smem_tiled_copy_Q,
            smem_tiled_copy_K=smem_tiled_copy_K,
            smem_tiled_copy_V=smem_tiled_copy_V,
            tSsQ=tSsQ,
            tSrQ_copy_view=tSrQ_copy_view,
            tSsK=tSsK,
            tSrK_copy_view=tSrK_copy_view,
            tOsVt=tOsVt,
            tOrVt_copy_view=tOrVt_copy_view,
        )
        softmax_params = SimpleNamespace(
            row_max=row_max,
            row_sum=row_sum,
            softmax_scale_log2=softmax_scale_log2,
        )


        self.compute_one_n_block(
            basic_params,
            mma_params,
            gmem_copy_params,
            smem_copy_params,
            softmax_params,
            is_first_n_block=True,
            apply_padding_mask=True,
        )
        for n_tile in range(1, n_block_max, 1):
            basic_params.n_block = n_block_max - n_tile - 1
            self.compute_one_n_block(
                basic_params,
                mma_params,
                gmem_copy_params,
                smem_copy_params,
                softmax_params,
                is_first_n_block=False,
                apply_padding_mask=False,
            )


        self.normalize_softmax(acc_O, row_sum)
        rO = cute.make_fragment_like(acc_O, self._dtype)
        rO.store(acc_O.load().to(self._dtype))
        sO = cute.make_tensor(sQ.iterator, sO_layout)

        smem_copy_atom_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self._dtype
        )
        smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
        smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        gO = cute.local_tile(
            mO[batch_idx, None, head_idx, None],
            (self._m_block_size, self._head_dim),
            (m_block, 0),
        )
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO)
        tOgO = gmem_thr_copy_O.partition_D(gO)
        tOrO = cute.make_fragment_like(tOgO, self._dtype)

        self.cta_sync_barrier.arrive_and_wait()
        cute.copy(gmem_tiled_copy_O, tOsO, tOrO)

        mcO = cute.make_identity_tensor(mO.layout.shape)
        cO = cute.local_tile(
            mcO[batch_idx, None, head_idx, None],
            (self._m_block_size, self._head_dim),
            (m_block, 0),
        )
        tOcO = gmem_thr_copy_O.partition_D(cO)
        tOpO = cute.make_rmem_tensor(
            cute.make_layout(
                (tOgO.shape[0][1], tOgO.shape[1], tOgO.shape[2]),
                stride=(tOgO.shape[2], 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tOpO.shape[0]):
            for rest_n in cutlass.range_constexpr(cute.size(tOpO.shape[2])):
                tOpO[rest_v, 0, rest_n] = cute.elem_less(
                    tOcO[(0, rest_v), 0, rest_n][3], mO.layout.shape[3]
                )
        for rest_m in cutlass.range_constexpr(cute.size(tOpO.shape[1])):
            if cute.elem_less(tOcO[0, rest_m, 0][1], mO.layout.shape[1]):
                cute.copy(
                    gmem_tiled_copy_O,
                    tOrO[None, rest_m, None],
                    tOgO[None, rest_m, None],
                    pred=tOpO[None, rest_m, None],
                )


    @cute.jit
    def compute_one_n_block(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        gmem_copy_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        is_first_n_block: cutlass.Constexpr,
        apply_padding_mask: cutlass.Constexpr,
    ):
        acc_shape_S = mma_params.thr_mma.partition_shape_C(
            (self._m_block_size, self._n_block_size)
        )
        acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
        acc_S.fill(0.0)


        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()
        if is_first_n_block:


            for n in cutlass.range_constexpr(cute.size(gmem_copy_params.tVsV.shape[1])):
                if cute.elem_less(
                    gmem_copy_params.tKVcKV[0, n, 0][1],
                    basic_params.mK.layout.shape[1],
                ):
                    cute.copy(
                        gmem_copy_params.gmem_tiled_copy_QKV,
                        gmem_copy_params.tVgV[None, n, None, basic_params.n_block],
                        gmem_copy_params.tVsV[None, n, None],
                        pred=gmem_copy_params.tKVpKV[None, n, None],
                    )
                else:
                    gmem_copy_params.tVsV[None, n, None].fill(0.0)
        else:
            cute.copy(
                gmem_copy_params.gmem_tiled_copy_QKV,
                gmem_copy_params.tVgV[None, None, None, basic_params.n_block],
                gmem_copy_params.tVsV,
                pred=gmem_copy_params.tKVpKV,
            )
        cute.arch.cp_async_commit_group()


        cute.copy(
            smem_copy_params.smem_tiled_copy_Q,
            smem_copy_params.tSsQ[None, None, 0],
            smem_copy_params.tSrQ_copy_view[None, None, 0],
        )
        cute.copy(
            smem_copy_params.smem_tiled_copy_K,
            smem_copy_params.tSsK[None, None, 0],
            smem_copy_params.tSrK_copy_view[None, None, 0],
        )
        for k in cutlass.range_constexpr(cute.size(smem_copy_params.tSsQ.shape[2])):
            k_next = (k + 1) % cute.size(smem_copy_params.tSsQ.shape[2])
            cute.copy(
                smem_copy_params.smem_tiled_copy_Q,
                smem_copy_params.tSsQ[None, None, k_next],
                smem_copy_params.tSrQ_copy_view[None, None, k_next],
            )
            cute.copy(
                smem_copy_params.smem_tiled_copy_K,
                smem_copy_params.tSsK[None, None, k_next],
                smem_copy_params.tSrK_copy_view[None, None, k_next],
            )
            cute.gemm(
                mma_params.tiled_mma,
                acc_S,
                mma_params.tSrQ[None, None, k],
                mma_params.tSrK[None, None, k],
                acc_S,
            )


        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()
        if basic_params.n_block > 0:
            cute.copy(
                gmem_copy_params.gmem_tiled_copy_QKV,
                gmem_copy_params.tKgK[None, None, None, basic_params.n_block - 1],
                gmem_copy_params.tKsK,
                pred=gmem_copy_params.tKVpKV,
            )
            cute.arch.cp_async_commit_group()


        self.softmax_rescale_O(
            basic_params,
            mma_params,
            softmax_params,
            acc_S,
            is_first_n_block,
            apply_padding_mask,
        )


        rP = cute.make_fragment_like(acc_S, self._dtype)
        rP.store(acc_S.load().to(self._dtype))


        # (4, MMA_M, MMA_N): 4 values = 2 rows x 2 adjacent columns of an


        rP_layout_divided = cute.logical_divide(rP.layout, (None, None, 2))
        rP_mma_view = cute.make_layout(
            (
                (rP_layout_divided.shape[0], rP_layout_divided.shape[2][0]),
                rP_layout_divided.shape[1],
                rP_layout_divided.shape[2][1],
            ),
            stride=(
                (rP_layout_divided.stride[0], rP_layout_divided.stride[2][0]),
                rP_layout_divided.stride[1],
                rP_layout_divided.stride[2][1],
            ),
        )
        tOrS = cute.make_tensor(rP.iterator, rP_mma_view)


        cute.copy(
            smem_copy_params.smem_tiled_copy_V,
            smem_copy_params.tOsVt[None, None, 0],
            smem_copy_params.tOrVt_copy_view[None, None, 0],
        )
        for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
            k_next = (k + 1) % cute.size(tOrS.shape[2])
            cute.copy(
                smem_copy_params.smem_tiled_copy_V,
                smem_copy_params.tOsVt[None, None, k_next],
                smem_copy_params.tOrVt_copy_view[None, None, k_next],
            )
            cute.gemm(
                mma_params.tiled_mma,
                mma_params.acc_O,
                tOrS[None, None, k],
                mma_params.tOrVt[None, None, k],
                mma_params.acc_O,
            )


    @cute.jit
    def softmax_rescale_O(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        acc_S: cute.Tensor,
        is_first_n_block: cutlass.Constexpr,
        apply_padding_mask: cutlass.Constexpr,
    ):
        # (M, N) views of the fragments: row index = (2-row mode, MMA_M),

        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
        acc_O_mn = self._make_acc_tensor_mn_view(mma_params.acc_O)


        if cutlass.const_expr(apply_padding_mask):
            mcS = cute.make_identity_tensor(
                (
                    basic_params.mQ.shape[0],
                    basic_params.mQ.shape[1],
                    basic_params.mQ.shape[2],
                    basic_params.mK.shape[1],
                )
            )
            cS = cute.local_tile(
                mcS[basic_params.batch_idx, None, basic_params.head_idx, None],
                (self._m_block_size, self._n_block_size),
                (basic_params.m_block, basic_params.n_block),
            )
            tScS = mma_params.thr_mma.partition_C(cS)
            tScS_mn = self._make_acc_tensor_mn_view(tScS)
            for r in cutlass.range_constexpr(cute.size(acc_S_mn.shape[0])):
                for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                    if cute.elem_less(basic_params.mK.shape[1], tScS_mn[0, c][3] + 1):
                        acc_S_mn[r, c] = -cutlass.Float32.inf


        row_max_prev = None
        if cutlass.const_expr(not is_first_n_block):
            row_max_prev = cute.make_fragment_like(
                softmax_params.row_max, cutlass.Float32
            )
            cute.basic_copy(softmax_params.row_max, row_max_prev)


        for r in cutlass.range_constexpr(cute.size(softmax_params.row_max)):
            acc_S_row = acc_S_mn[r, None].load()
            row_max_cur = acc_S_row.reduce(
                cute.ReductionOp.MAX, -cutlass.Float32.inf, 0
            )

            row_max_cur = self._threadquad_reduce_max(row_max_cur)

            row_max_prev_row = None
            if cutlass.const_expr(not is_first_n_block):
                row_max_prev_row = row_max_prev[r]
                row_max_cur = cute.arch.fmax(row_max_prev_row, row_max_cur)


            acc_S_row_exp = cute.math.exp2(
                acc_S_row * softmax_params.softmax_scale_log2
                - row_max_cur * softmax_params.softmax_scale_log2,
                fastmath=True,
            )


            acc_S_row_sum = acc_S_row_exp.reduce(
                cute.ReductionOp.ADD, cutlass.Float32.zero, 0
            )

            if cutlass.const_expr(not is_first_n_block):

                correction = cute.math.exp2(
                    row_max_prev_row * softmax_params.softmax_scale_log2
                    - row_max_cur * softmax_params.softmax_scale_log2,
                    fastmath=True,
                )
                acc_S_row_sum = (
                    acc_S_row_sum + softmax_params.row_sum[r] * correction
                )

                acc_O_mn[r, None] = acc_O_mn[r, None].load() * correction

            softmax_params.row_max[r] = row_max_cur
            softmax_params.row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None] = acc_S_row_exp


    @cute.jit
    def normalize_softmax(self, acc_O: cute.Tensor, row_sum: cute.Tensor):

        acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
        for r in cutlass.range_constexpr(cute.size(row_sum)):
            row_sum[r] = self._threadquad_reduce_sum(row_sum[r])

            is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
            scale = (
                1.0 if is_zero_or_nan else cute.arch.rcp_approx(row_sum[r])
            )
            acc_O_mn[r, None] = acc_O_mn[r, None].load() * scale


    def _make_acc_tensor_mn_view(self, acc: cute.Tensor) -> cute.Tensor:

        acc_layout_col_major = cute.make_layout(acc.layout.shape)
        acc_layout_mn = cute.make_layout(
            (
                (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),
                (acc_layout_col_major.shape[0][0], acc_layout_col_major.shape[2]),
            ),
            stride=(
                (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),
                (acc_layout_col_major.stride[0][0], acc_layout_col_major.stride[2]),
            ),
        )
        acc_layout_mn = cute.composition(acc.layout, acc_layout_mn)
        return cute.make_tensor(acc.iterator, acc_layout_mn)

    def _threadquad_reduce(self, val, op):


        val = op(
            val, cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31)
        )
        val = op(
            val, cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31)
        )
        return val

    def _threadquad_reduce_max(self, val):
        return self._threadquad_reduce(val, lambda x, y: cute.arch.fmax(x, y))

    def _threadquad_reduce_sum(self, val):
        return self._threadquad_reduce(val, lambda x, y: x + y)


def main():
    import torch
    import cutlass.torch as cutlass_torch

    torch.manual_seed(7)

    def make_qkvo(b, s, d):
        # (batch, seqlen, num_head=1, head_dim), bf16, small ints
        t = (
            torch.empty(b, s, 1, d, dtype=torch.int32)
            .random_(-2, 2)
            .to(dtype=cutlass_torch.dtype(io_dtype))
            .cuda()
        )
        m = (
            from_dlpack(t, assumed_align=16)
            .mark_layout_dynamic(leading_dim=3)
            .mark_compact_shape_dynamic(
                mode=3,
                stride_order=t.dim_order(),
                divisibility=(128 // io_dtype.width),
            )
        )
        return m, t

    def attention_ref(q, k, v, scale, dtype):
        # (b, s, 1, d) -> (b, 1, s, d), matmuls in `dtype`, softmax in fp32
        qh = q.permute(0, 2, 1, 3).to(dtype)
        kh = k.permute(0, 2, 1, 3).to(dtype)
        vh = v.permute(0, 2, 1, 3).to(dtype)
        s = torch.matmul(qh, kh.transpose(-1, -2)).float() * scale
        p = s.softmax(dim=-1).to(dtype)
        o = torch.matmul(p, vh)
        return o.permute(0, 2, 1, 3).float()

    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    # (batch, seqlen_q, seqlen_k, head_dim); seqlen_k=1000 exercises the

    cases = [
        (2, 512, 512, 64),
        (1, 1024, 1000, 128),
        (1, 2048, 2048, 64),
    ]
    compiled_by_dim = {}
    all_ok = True
    for b, sq, sk, d in cases:
        scale = 1.0 / (d ** 0.5)
        q, q_t = make_qkvo(b, sq, d)
        k, k_t = make_qkvo(b, sk, d)
        v, v_t = make_qkvo(b, sk, d)
        o, o_t = make_qkvo(b, sq, d)

        if d not in compiled_by_dim:
            fa = AttentionMini(d)
            compiled_by_dim[d] = cute.compile(fa, q, k, v, o, scale, stream)
        compiled_by_dim[d](q, k, v, o, scale, stream)
        torch.cuda.synchronize()


        ref_fp32 = attention_ref(q_t, k_t, v_t, scale, torch.float32).cpu()
        ref_bf16 = attention_ref(q_t, k_t, v_t, scale, torch.bfloat16).cpu()
        kernel_err = (o_t.float().cpu() - ref_fp32).abs().max().item()
        bf16_err = (ref_bf16 - ref_fp32).abs().max().item()
        ok = kernel_err <= 4 * bf16_err
        all_ok &= ok
        print(
            f"  b={b} sq={sq:4d} sk={sk:4d} d={d:3d}  "
            f"kernel_err={kernel_err:.3e}  4*bf16_err={4 * bf16_err:.3e}  "
            f"{'OK' if ok else 'FAIL'}"
        )

    print("PASS" if all_ok else "FAIL")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
