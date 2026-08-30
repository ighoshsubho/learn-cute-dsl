import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warp
from cutlass.cute.runtime import from_dlpack


TILE_M, TILE_N, TILE_K = 64, 32, 32
NUM_THREADS = 128


def make_mma():


    return cute.make_tiled_mma(
        warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)),
        (NUM_THREADS // 32, 1, 1),
        permutation_mnk=(NUM_THREADS // 32 * 16, 16, 16),
    )


@cute.kernel
def gemm_kernel(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
                tiled_mma: cute.TiledMma, g2s_copy: cute.TiledCopy):
    tidx, _, _ = cute.arch.thread_idx()


    @cute.struct
    class SharedStorage:
        sA: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, TILE_M * TILE_K], 16]
        sB: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, TILE_N * TILE_K], 16]

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = storage.sA.get_tensor(cute.make_layout((TILE_M, TILE_K), stride=(TILE_K, 1)))
    sB = storage.sB.get_tensor(cute.make_layout((TILE_N, TILE_K), stride=(TILE_K, 1)))

    thr_g2s = g2s_copy.get_slice(tidx)
    cute.copy(g2s_copy, thr_g2s.partition_S(mA), thr_g2s.partition_D(sA))
    cute.copy(g2s_copy, thr_g2s.partition_S(mB), thr_g2s.partition_D(sB))
    cute.arch.sync_threads()


    thr_mma = tiled_mma.get_slice(tidx)
    tCsA = thr_mma.partition_A(sA)                 # ((2,2,2), MMA_M=1, MMA_K=2)
    tCsB = thr_mma.partition_B(sB)                 # ((2,2),   MMA_N=4, MMA_K=2)
    tCrA = tiled_mma.make_fragment_A(tCsA)
    tCrB = tiled_mma.make_fragment_B(tCsB)
    acc = thr_mma.make_fragment_C(thr_mma.partition_shape_C((TILE_M, TILE_N)))
    acc.fill(0.0)


    ldm_atom = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.BFloat16
    )
    s2r_A = cute.make_tiled_copy_A(ldm_atom, tiled_mma)
    s2r_B = cute.make_tiled_copy_B(ldm_atom, tiled_mma)
    thr_s2r_A = s2r_A.get_slice(tidx)
    thr_s2r_B = s2r_B.get_slice(tidx)
    tXsA = thr_s2r_A.partition_S(sA)               # ((8,1), 1, 2) — copy's view
    tXsB = thr_s2r_B.partition_S(sB)
    tXrA = thr_s2r_A.retile(tCrA)
    tXrB = thr_s2r_B.retile(tCrB)


    for k in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
        cute.copy(s2r_A, tXsA[None, None, k], tXrA[None, None, k])
        cute.copy(s2r_B, tXsB[None, None, k], tXrB[None, None, k])
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)


    tCgC = thr_mma.partition_C(mC)
    cute.autovec_copy(acc, tCgC)


@cute.jit
def run(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    tiled_mma = make_mma()
    thr_mma = tiled_mma.get_slice(0)


    tCsA = thr_mma.partition_A(mA)
    tCsB = thr_mma.partition_B(mB)
    tCrA = tiled_mma.make_fragment_A(tCsA)
    tCrB = tiled_mma.make_fragment_B(tCsB)
    acc_shape = thr_mma.partition_shape_C((TILE_M, TILE_N))
    print("partition_A     :", tCsA.shape, "  frag layout:", tCrA.layout)
    print("partition_B     :", tCsB.shape, "  frag layout:", tCrB.layout)
    print("partition_shape_C:", acc_shape)

    predicted_A = ((2, 2, 2), 1, 2)

    predicted_B = ((2, 2), 4, 2)

    predicted_C = ((2, 2), 1, 4)
    assert tuple(tCsA.shape) == predicted_A, f"{tCsA.shape} != {predicted_A}"
    assert tuple(tCsB.shape) == predicted_B, f"{tCsB.shape} != {predicted_B}"
    assert tuple(acc_shape) == predicted_C, f"{acc_shape} != {predicted_C}"


    ldm_atom = cute.make_copy_atom(
        warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.BFloat16
    )
    s2r_A = cute.make_tiled_copy_A(ldm_atom, tiled_mma)
    tXsA = s2r_A.get_slice(0).partition_S(mA)
    tXrA = s2r_A.get_slice(0).retile(tCrA)
    print("s2r partition_S :", tXsA.shape)
    print("retiled frag A  :", tXrA.shape, " layout:", tXrA.layout)


    assert tuple(tXsA.shape) == ((8, 1), 1, 2)
    assert cute.size(tXrA) == cute.size(tCrA)


    g2s_copy = cute.make_tiled_copy_tv(
        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16,
                            num_bits_per_copy=128),
        cute.make_ordered_layout((32, 4), order=(1, 0)),
        cute.make_layout((1, 8)),
    )

    gemm_kernel(mA, mB, mC, tiled_mma, g2s_copy).launch(
        grid=[1, 1, 1], block=[NUM_THREADS, 1, 1]
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.randn(TILE_M, TILE_K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(TILE_N, TILE_K, device="cuda", dtype=torch.bfloat16)  # (N, K)
    c = torch.zeros(TILE_M, TILE_N, device="cuda", dtype=torch.float32)
    run(
        from_dlpack(a, assumed_align=16),
        from_dlpack(b, assumed_align=16),
        from_dlpack(c, assumed_align=16),
    )
    torch.cuda.synchronize()
    ref = a.float() @ b.float().T
    max_err = (c - ref).abs().max().item()
    print(f"max |err| vs torch: {max_err:.3e}")
    assert torch.allclose(c, ref, atol=1e-2, rtol=1e-3), "MMA result mismatch"
    print("PASS")
