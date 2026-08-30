import operator

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.runtime import from_dlpack
import torch


M, N = 256, 8192
CLUSTER_N = 2
COLS_PER_CTA = N // CLUSTER_N
THREADS_PER_CTA = 256
WARPS_PER_ROW = THREADS_PER_CTA // 32
ELEMS_PER_THREAD = COLS_PER_CTA // THREADS_PER_CTA
DTYPE = cutlass.Float32


EXCHANGE_BYTES = WARPS_PER_ROW * CLUSTER_N * 4

assert N % CLUSTER_N == 0 and COLS_PER_CTA % THREADS_PER_CTA == 0


@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: Int32, *, loc=None, ip=None
) -> Int32:

    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def store_shared_remote(
    val: Float32,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc=None,
    ip=None,
) -> None:

    remote_smem_ptr_i32 = set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    llvm.inline_asm(
        None,
        [remote_smem_ptr_i32, val.ir_value(loc=loc, ip=ip), remote_mbar_ptr_i32],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [$0], $1, [$2];",
        "r,f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord, *, loc=None, ip=None) -> cute.Pointer:

    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@cute.kernel
def kernel(mX: cute.Tensor, mO: cute.Tensor, tiler_mn: cute.Shape):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, cluster_y, _ = cute.arch.block_idx()
    cta_rank = cute.arch.block_idx_in_cluster()
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())


    smem = utils.SmemAllocator()
    reduction_buffer = smem.allocate_tensor(
        Float32,
        cute.make_layout((1, (WARPS_PER_ROW, CLUSTER_N))),
        byte_alignment=8,
    )
    mbar_ptr = smem.allocate_array(Int64, num_elems=1)


    if tidx == 0:
        cute.arch.mbarrier_init(mbar_ptr, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.cluster_arrive_relaxed()
    cute.arch.cluster_wait()


    gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
    gO = cute.local_tile(mO, tiler_mn, (bidx, cluster_y))

    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), DTYPE, num_bits_per_copy=128
    )
    thr_layout = cute.make_ordered_layout((1, THREADS_PER_CTA), order=(1, 0))
    val_layout = cute.make_layout((1, ELEMS_PER_THREAD))
    tiled_copy = cute.make_tiled_copy_tv(atom, thr_layout, val_layout)
    thr_copy = tiled_copy.get_slice(tidx)

    tXgX = thr_copy.partition_S(gX)
    tXgO = thr_copy.partition_D(gO)
    tXrX = cute.make_fragment_like(tXgX)
    cute.copy(tiled_copy, tXgX, tXrX)


    x = tXrX.load().to(Float32)
    local_sum = x.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0)
    warp_sum = cute.arch.warp_reduction(local_sum, operator.add)


    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, EXCHANGE_BYTES)


    if lane_idx < CLUSTER_N:
        store_shared_remote(
            warp_sum,
            elem_pointer(reduction_buffer, (0, (warp_idx, cta_rank))),
            mbar_ptr,
            peer_cta_rank_in_cluster=lane_idx,
        )


    cute.arch.mbarrier_wait(mbar_ptr, 0)


    num_partials = WARPS_PER_ROW * CLUSTER_N
    total = Float32(0.0)
    if lane_idx < num_partials:
        total = reduction_buffer[0, lane_idx]
    total = cute.arch.warp_reduction(total, operator.add)


    y = x * cute.arch.rcp_approx(total)
    tXrX.store(y.to(DTYPE))
    cute.copy(tiled_copy, tXrX, tXgO)


    cute.arch.cluster_arrive_relaxed()
    cute.arch.cluster_wait()


@cute.jit
def host(mX: cute.Tensor, mO: cute.Tensor):
    tiler_mn = (1, COLS_PER_CTA)
    kernel(mX, mO, tiler_mn).launch(
        grid=(M, CLUSTER_N, 1),
        block=(THREADS_PER_CTA, 1, 1),
        cluster=(1, CLUSTER_N, 1),
    )


def main():
    torch.manual_seed(2025)

    x = torch.rand((M, N), dtype=torch.float32, device="cuda") + 0.5
    o = torch.zeros_like(x)

    host(from_dlpack(x, assumed_align=16), from_dlpack(o, assumed_align=16))
    torch.cuda.synchronize()

    ref = x / x.sum(dim=1, keepdim=True)

    if torch.allclose(o, ref, rtol=1e-4, atol=1e-7):
        print("PASS")
    else:
        err = (o - ref).abs().max().item()
        print(f"FAIL: max abs err {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
