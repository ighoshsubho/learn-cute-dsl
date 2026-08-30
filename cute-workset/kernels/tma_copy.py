import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
import torch


M, N = 1024, 1024
TILE_M, TILE_N = 128, 128
N_TILES = N // TILE_N
THREADS_PER_CTA = 128
DTYPE = cutlass.BFloat16
SCALE = 2.0

TMA_BYTES = TILE_M * TILE_N * (DTYPE.width // 8)

assert M % TILE_M == 0 and N % TILE_N == 0, "this variant assumes divisible shapes"


@cute.struct
class SharedStorage:

    tma_mbar: cute.struct.MemRange[cutlass.Int64, 1]

    data: cute.struct.Align[
        cute.struct.MemRange[DTYPE, TILE_M * TILE_N], 1024
    ]


@cute.kernel
def kernel(
    tma_atom_src: cute.CopyAtom,
    tma_tensor_src: cute.Tensor,
    tma_atom_dst: cute.CopyAtom,
    tma_tensor_dst: cute.Tensor,
    smem_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sT = storage.data.get_tensor(smem_layout)          # (TILE_M, TILE_N) smem view
    mbar = storage.tma_mbar.data_ptr()


    # (None, None) keeps ALL tile coordinates — we index per iteration below.

    gSrc = cute.local_tile(tma_tensor_src, (TILE_M, TILE_N), (None, None))
    gDst = cute.local_tile(tma_tensor_dst, (TILE_M, TILE_N), (None, None))


    with cute.arch.elect_one():


        cute.arch.mbarrier_init(mbar, 1)

    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()


    tAsS, tAgS = cpasync.tma_partition(
        tma_atom_src,
        0,
        cute.make_layout(1),
        cute.group_modes(sT, 0, 2),
        cute.group_modes(gSrc, 0, 2),
    )
    tBsS, tBgD = cpasync.tma_partition(
        tma_atom_dst,
        0,
        cute.make_layout(1),
        cute.group_modes(sT, 0, 2),
        cute.group_modes(gDst, 0, 2),
    )


    atom_univ = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), DTYPE, num_bits_per_copy=128
    )
    thr_layout = cute.make_ordered_layout((THREADS_PER_CTA, 1), order=(1, 0))
    val_layout = cute.make_layout((1, TILE_N))
    tiled_copy = cute.make_tiled_copy_tv(atom_univ, thr_layout, val_layout)
    thr_copy = tiled_copy.get_slice(tidx)
    tXsS = thr_copy.partition_S(sT)
    tXrS = cute.make_fragment_like(tXsS)


    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    for j in cutlass.range_constexpr(N_TILES):
        if cutlass.const_expr(j > 0):


            if warp_idx == 0:
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.barrier()


        if warp_idx == 0:
            with cute.arch.elect_one():


                cute.arch.mbarrier_arrive_and_expect_tx(mbar, TMA_BYTES)
            cute.copy(
                tma_atom_src,
                tAgS[(None, bidx, j)],
                tAsS,
                tma_bar_ptr=mbar,
            )


        cute.arch.mbarrier_wait(mbar, j % 2)


        cute.autovec_copy(tXsS, tXrS)
        x = tXrS.load().to(cutlass.Float32)
        tXrS.store((x * SCALE).to(DTYPE))
        cute.autovec_copy(tXrS, tXsS)


        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier()


        if warp_idx == 0:
            cute.copy(tma_atom_dst, tBsS, tBgD[(None, bidx, j)])


            cute.arch.cp_async_bulk_commit_group()


    if warp_idx == 0:
        cute.arch.cp_async_bulk_wait_group(0, read=True)


@cute.jit
def host(src: cute.Tensor, dst: cute.Tensor):


    smem_layout = cute.make_layout((TILE_M, TILE_N), stride=(TILE_N, 1))


    tma_atom_src, tma_tensor_src = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), src, smem_layout, (TILE_M, TILE_N)
    )
    tma_atom_dst, tma_tensor_dst = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), dst, smem_layout, (TILE_M, TILE_N)
    )

    kernel(tma_atom_src, tma_tensor_src, tma_atom_dst, tma_tensor_dst, smem_layout).launch(
        grid=(M // TILE_M, 1, 1),
        block=(THREADS_PER_CTA, 1, 1),
    )


def main():
    torch.manual_seed(2025)
    a = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")
    b = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")

    a_cute = from_dlpack(a, assumed_align=16)
    b_cute = from_dlpack(b, assumed_align=16)

    host(a_cute, b_cute)
    torch.cuda.synchronize()

    ref = a * SCALE
    if torch.equal(b, ref):
        print("PASS")
    else:
        bad = (b != ref).sum().item()
        print(f"FAIL: {bad}/{b.numel()} mismatched elements")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
