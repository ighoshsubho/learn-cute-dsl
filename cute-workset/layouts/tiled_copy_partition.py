import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


M, N = 64, 128
TILE_M, TILE_N = 32, 128
NUM_THREADS = 128


@cute.kernel
def copy_kernel(mX: cute.Tensor, mO: cute.Tensor, tiled_copy: cute.TiledCopy):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()


    gX = cute.local_tile(mX, (TILE_M, TILE_N), (bidx, 0))
    gO = cute.local_tile(mO, (TILE_M, TILE_N), (bidx, 0))


    thr_copy = tiled_copy.get_slice(tidx)
    tXgX = thr_copy.partition_S(gX)
    tXgO = thr_copy.partition_D(gO)


    tXrX = cute.make_fragment_like(tXgX)

    cute.copy(tiled_copy, tXgX, tXrX)
    cute.copy(tiled_copy, tXrX, tXgO)


@cute.jit
def run(mX: cute.Tensor, mO: cute.Tensor):


    thr_layout = cute.make_ordered_layout((16, 8), order=(1, 0))

    val_layout = cute.make_layout((1, 8))
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
    )
    tiled_copy = cute.make_tiled_copy_tv(atom, thr_layout, val_layout)
    print("thr_layout       :", thr_layout)
    print("val_layout       :", val_layout)
    print("tiled copy tiler :", tiled_copy.tiler_mn)   # (16, 64)
    print("tiled copy TV    :", tiled_copy.layout_tv_tiled)


    #        ((atom_v, rest_v), m_iters, n_iters)
    predicted = ((8, 1), 2, 2)

    gX = cute.local_tile(mX, (TILE_M, TILE_N), (0, 0))
    tXgX = tiled_copy.get_slice(0).partition_S(gX)
    print("partition_S shape:", tXgX.shape, " layout:", tXgX.layout)
    assert tuple(tXgX.shape) == predicted, f"{tXgX.shape} != {predicted}"


    val_layout_4 = cute.make_layout((1, 4))
    atom_64 = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=64
    )
    tiled_copy_4 = cute.make_tiled_copy_tv(atom_64, thr_layout, val_layout_4)
    tXgX_4 = tiled_copy_4.get_slice(0).partition_S(gX)
    predicted_4 = ((4, 1), 2, 4)
    print("val=(1,4) tiler  :", tiled_copy_4.tiler_mn)
    print("val=(1,4) shape  :", tXgX_4.shape)
    assert tuple(tXgX_4.shape) == predicted_4, f"{tXgX_4.shape} != {predicted_4}"


    copy_kernel(mX, mO, tiled_copy).launch(
        grid=[M // TILE_M, 1, 1], block=[NUM_THREADS, 1, 1]
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    o = torch.zeros_like(x)
    run(from_dlpack(x, assumed_align=16), from_dlpack(o, assumed_align=16))
    torch.cuda.synchronize()
    assert torch.equal(o, x), "output != input"
    print("PASS")
