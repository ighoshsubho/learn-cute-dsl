import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

TILE = (32, 32)
THREADS = 64


@cute.kernel
def block_shuffle_kernel(mA: cute.Tensor, mO: cute.Tensor,
                         tiler: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()


    gA = cute.local_tile(mA, tiler, (bidx, bidy))

    gO = cute.local_tile(mO, tiler, (bidy, bidx))
    print("[trace] gA:", gA)     # (32,32) view into gmem, strides (128,1)


    for i in cutlass.range_constexpr(cute.size(gA) // THREADS):
        idx = tidx + i * THREADS
        gO[idx] = gA[idx]


@cute.kernel
def ktile_loop_kernel(mA: cute.Tensor, mO: cute.Tensor,
                      tiler: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()


    gA = cute.local_tile(mA, tiler, (bidx, None))
    gO = cute.local_tile(mO, tiler, (bidx, None))
    print("[trace] gA with None mode:", gA)   # (32,32,4)


    for k in cutlass.range_constexpr(cute.size(gA, mode=[2])):
        blkA = gA[None, None, k]              # (32,32) tile k
        blkO = gO[None, None, k]
        for i in cutlass.range_constexpr(cute.size(blkA) // THREADS):
            idx = tidx + i * THREADS
            blkO[idx] = blkA[idx]


@cute.jit
def run(mA: cute.Tensor, mO1: cute.Tensor, mO2: cute.Tensor):

    gCols = cute.local_tile(mA, TILE, (None, 0))
    print("local_tile(mA, (32,32), (None, 0)):", gCols)
    assert cute.shape(gCols) == (32, 32, 4), "None mode should keep 4 row-tiles"

    block_shuffle_kernel(mA, mO1, TILE).launch(
        grid=[4, 4, 1], block=[THREADS, 1, 1])
    ktile_loop_kernel(mA, mO2, TILE).launch(
        grid=[4, 1, 1], block=[THREADS, 1, 1])


if __name__ == "__main__":
    a = torch.arange(128 * 128, dtype=torch.float32, device="cuda").reshape(128, 128)
    o1 = torch.full_like(a, -1.0)
    o2 = torch.full_like(a, -1.0)
    run(from_dlpack(a), from_dlpack(o1), from_dlpack(o2))
    torch.cuda.synchronize()


    for i in range(4):
        for j in range(4):
            src = a[32 * i:32 * (i + 1), 32 * j:32 * (j + 1)]
            dst = o1[32 * j:32 * (j + 1), 32 * i:32 * (i + 1)]
            assert torch.equal(src, dst), f"block ({i},{j}) landed wrong"
    print("  ok  all 16 blocks landed at transposed tile positions")


    assert torch.equal(a, o2), "k-tile loop copy mismatch"
    print("  ok  None-mode k-loop copied every column tile")
    print("PASS")
