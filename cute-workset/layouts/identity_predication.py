import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

M, N = 64, 88
TILE_M, TILE_N = 32, 128
NUM_THREADS = 128
VEC = 8

assert N % VEC == 0, "vector-granularity predication needs N % VEC == 0"


@cute.kernel
def copy_kernel(mX: cute.Tensor, mO: cute.Tensor, tiled_copy: cute.TiledCopy):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    gX = cute.local_tile(mX, (TILE_M, TILE_N), (bidx, 0))
    gO = cute.local_tile(mO, (TILE_M, TILE_N), (bidx, 0))


    cX = cute.local_tile(cute.make_identity_tensor(mX.shape), (TILE_M, TILE_N), (bidx, 0))

    thr_copy = tiled_copy.get_slice(tidx)
    tXgX = thr_copy.partition_S(gX)   # ((8,1), 2, 2)
    tXgO = thr_copy.partition_D(gO)
    tXcX = thr_copy.partition_S(cX)
    tXrX = cute.make_fragment_like(tXgX)


    tXpX = cute.make_rmem_tensor(
        cute.make_layout(
            (
                cute.size(tXcX, mode=[0, 1]),
                cute.size(tXcX, mode=[1]),
                cute.size(tXcX, mode=[2]),
            ),
            stride=(cute.size(tXcX, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tXpX.shape[0]):
        for rest_n in cutlass.range_constexpr(tXpX.shape[2]):


            tXpX[rest_v, 0, rest_n] = cute.elem_less(
                tXcX[(0, rest_v), 0, rest_n][1], mX.shape[1]
            )


    cute.copy(tiled_copy, tXgX, tXrX, pred=tXpX)
    cute.copy(tiled_copy, tXrX, tXgO, pred=tXpX)


@cute.jit
def run(mX: cute.Tensor, mO: cute.Tensor):
    thr_layout = cute.make_ordered_layout((16, 8), order=(1, 0))
    val_layout = cute.make_layout((1, VEC))
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
    )
    tiled_copy = cute.make_tiled_copy_tv(atom, thr_layout, val_layout)


    gX = cute.local_tile(mX, (TILE_M, TILE_N), (0, 0))
    cX = cute.local_tile(cute.make_identity_tensor(mX.shape), (TILE_M, TILE_N), (0, 0))
    thr_copy = tiled_copy.get_slice(0)
    print("data  partition:", thr_copy.partition_S(gX).shape)
    print("coord partition:", thr_copy.partition_S(cX).shape)
    assert tuple(thr_copy.partition_S(cX).shape) == tuple(thr_copy.partition_S(gX).shape)

    copy_kernel(mX, mO, tiled_copy).launch(
        grid=[M // TILE_M, 1, 1], block=[NUM_THREADS, 1, 1]
    )


if __name__ == "__main__":
    torch.manual_seed(0)


    SENTINEL = -777.0
    x_buf = torch.randn(M, TILE_N, device="cuda", dtype=torch.bfloat16)
    o_buf = torch.full((M, TILE_N), SENTINEL, device="cuda", dtype=torch.bfloat16)
    x, o = x_buf[:, :N], o_buf[:, :N]
    run(from_dlpack(x, assumed_align=16), from_dlpack(o, assumed_align=16))
    torch.cuda.synchronize()
    assert torch.equal(o, x), "ragged copy corrupted data"
    assert torch.all(o_buf[:, N:] == SENTINEL),\
        "predication leaked writes into the padding columns"
    print("PASS")
