import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

M, N = 64, 64
NUM_THREADS = 128


@cute.kernel
def stage_kernel(mX: cute.Tensor, mO: cute.Tensor, tiled_copy: cute.TiledCopy,
                 sX_layout: cute.ComposedLayout):
    tidx, _, _ = cute.arch.thread_idx()

    @cute.struct
    class SharedStorage:
        sX: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, M * N], 16]

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)


    sX = storage.sX.get_tensor(sX_layout)

    thr_copy = tiled_copy.get_slice(tidx)
    tXgX = thr_copy.partition_S(mX)
    tXsX = thr_copy.partition_D(sX)
    tXgO = thr_copy.partition_D(mO)

    cute.copy(tiled_copy, tXgX, tXsX)
    cute.arch.sync_threads()
    tXsX_src = thr_copy.partition_S(sX)
    cute.copy(tiled_copy, tXsX_src, tXgO)


@cute.jit
def run(mX: cute.Tensor, mO: cute.Tensor):


    sX_layout_atom = cute.make_composed_layout(
        cute.make_swizzle(3, 3, 3),
        0,
        cute.make_layout((8, 64), stride=(64, 1)),
    )

    sX_layout = cute.tile_to_shape(sX_layout_atom, (M, N), (0, 1))
    print("swizzle atom :", sX_layout_atom)
    print("smem layout  :", sX_layout)


    plain = cute.make_layout((M, N), stride=(N, 1))
    plain_slots = [(2 * cute.crd2idx((r, 0), plain)) % 128 // 16 for r in range(8)]
    swz_slots = [(2 * cute.crd2idx((r, 0), sX_layout)) % 128 // 16 for r in range(8)]
    print("column 0, rows 0..7 -> 16B bank-group slots")
    print("  plain row-major :", plain_slots, " (all identical: 8-way conflict)")
    print("  swizzled        :", swz_slots, " (a permutation of 0..7: conflict-free)")
    assert plain_slots == [0] * 8
    assert sorted(swz_slots) == list(range(8)), "swizzle must spread all 8 slots"


    row_offs = [[cute.crd2idx((r, c), sX_layout) for c in range(8)] for r in range(8)]
    for offs in row_offs:
        assert offs == list(range(offs[0], offs[0] + 8)), "swizzle broke 8-elem contiguity"


    thr_layout = cute.make_ordered_layout((16, 8), order=(1, 0))
    val_layout = cute.make_layout((1, 8))
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
    )
    tiled_copy = cute.make_tiled_copy_tv(atom, thr_layout, val_layout)

    stage_kernel(mX, mO, tiled_copy, sX_layout).launch(
        grid=[1, 1, 1], block=[NUM_THREADS, 1, 1]
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    o = torch.zeros_like(x)
    run(from_dlpack(x, assumed_align=16), from_dlpack(o, assumed_align=16))
    torch.cuda.synchronize()
    assert torch.equal(o, x), "swizzled staging corrupted data"
    print(
        "\nWhy this matters: without the swizzle, every element of a column\n"
        "lives in the same bank group, so column-wise smem reads (transposes,\n"
        "ldmatrix feeding tensor cores) serialize 8-way. The XOR rotates each\n"
        "row's chunks so any 8-row column slice touches all 8 bank groups —\n"
        "same capacity, same vectorization, zero conflicts."
    )
    print("PASS")
