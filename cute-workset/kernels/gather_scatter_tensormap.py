from typing import Sequence

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32, const_expr
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

TILE_N = 128
THREADS_CP = 32
THREADS_G4 = 128
VEC = 4


@cute.struct
class TmaSharedStorage:
    load_mbar: cutlass.Int64


@dsl_user_op
def get_tma_desc_addr(tma_atom: cute.CopyAtom, *, loc=None, ip=None):

    exec_atom = _cute_nvgpu_ir.atom_make_exec_tma(
        tma_atom._trait.value, loc=loc, ip=ip)
    desc_ptr_ty = ir.Type.parse(
        "!cute.ptr<!cute_nvgpu.tma_descriptor_tiled, generic, align<128>>")
    return _cute_nvgpu_ir.get_tma_desc_addr(
        desc_ptr_ty, exec_atom, loc=loc, ip=ip)


@dsl_user_op
def tma_gather4_load(
    tma_desc_ptr: cute.Pointer,
    dst_smem_ptr: cute.Pointer,
    mbarrier_ptr: cute.Pointer,
    col: Int32,
    rows: Sequence[Int32],
    *,
    loc=None,
    ip=None,
):

    assert len(rows) == 4
    desc = tma_desc_ptr.toint(loc=loc, ip=ip).ir_value()
    dst = dst_smem_ptr.toint(loc=loc, ip=ip).ir_value()
    mbar = mbarrier_ptr.toint(loc=loc, ip=ip).ir_value()
    crd = Int32(col).ir_value()
    idx = [Int32(row).ir_value() for row in rows]
    llvm.inline_asm(
        None,
        [dst, desc, crd, idx[0], idx[1], idx[2], idx[3], mbar],
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4."
        "mbarrier::complete_tx::bytes.cta_group::1 "
        "[$0], [$1, {$2, $3, $4, $5, $6}], [$7];",
        "r,l,r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_scatter4_store(
    tma_desc_ptr: cute.Pointer,
    src_smem_ptr: cute.Pointer,
    col: Int32,
    rows: Sequence[Int32],
    *,
    loc=None,
    ip=None,
):

    assert len(rows) == 4
    desc = tma_desc_ptr.toint(loc=loc, ip=ip).ir_value()
    src = src_smem_ptr.toint(loc=loc, ip=ip).ir_value()
    crd = Int32(col).ir_value()
    idx = [Int32(row).ir_value() for row in rows]
    llvm.inline_asm(
        None,
        [desc, src, crd, idx[0], idx[1], idx[2], idx[3]],
        "cp.async.bulk.tensor.2d.global.shared::cta.tile::scatter4.bulk_group "
        "[$0, {$2, $3, $4, $5, $6}], [$1];",
        "l,r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@cute.jit
def aligned_row_view(tensor, row, n):

    ptr = tensor.iterator + row * tensor.stride[0]
    ptr = cute.make_ptr(
        tensor.element_type, ptr.toint(), tensor.memspace,
        assumed_align=tensor.iterator.alignment)
    return cute.make_tensor(
        ptr, cute.make_layout(
            (1, n), stride=(tensor.stride[0], tensor.stride[1])))


class GatherScatter:
    def __init__(self, mode: str):
        assert mode in ("cpasync", "tma_gather4")
        self.mode = mode

    @cute.jit
    def __call__(self, mX, mY, mIdx, alpha, stream):


        atom_cp = cute.make_copy_atom(
            cpasync.CopyG2SOp(), Float32, num_bits_per_copy=128)
        atom_uni = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128)
        val_layout = cute.make_layout((1, VEC))

        if const_expr(self.mode == "cpasync"):
            thr_layout = cute.make_layout((1, THREADS_CP))
            cp_copy = cute.make_tiled_copy_tv(atom_cp, thr_layout, val_layout)
            uni_copy = cute.make_tiled_copy_tv(atom_uni, thr_layout, val_layout)
            smem_layout = cute.make_ordered_layout((1, TILE_N), order=(1, 0))
            self.kernel_cpasync(
                mX, mY, mIdx, alpha, cp_copy, uni_copy, smem_layout
            ).launch(grid=(mIdx.shape[0], 1, 1),
                     block=(THREADS_CP, 1, 1), stream=stream)
        else:


            desc_layout = cute.make_ordered_layout((1, TILE_N), order=(1, 0))
            s4_layout = cute.make_ordered_layout((4, TILE_N), order=(1, 0))
            load_atom, _ = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(), mX, desc_layout,
                (1, TILE_N))
            store_atom, _ = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(), mY, desc_layout,
                (1, TILE_N))
            thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
            uni_copy = cute.make_tiled_copy_tv(atom_uni, thr_layout, val_layout)
            self.kernel_tma_gather4(
                mX, mY, mIdx, alpha, load_atom, store_atom,
                uni_copy, s4_layout,
            ).launch(grid=(cute.ceil_div(mIdx.shape[0], 4), 1, 1),
                     block=(THREADS_G4, 1, 1), stream=stream)

    @cute.kernel
    def kernel_cpasync(self, mX, mY, mIdx, alpha, cp_copy, uni_copy, smem_layout):
        tidx, _, _ = cute.arch.thread_idx()
        g, _, _ = cute.arch.block_idx()
        row = mIdx[g]
        n = mX.shape[1]

        row_x = aligned_row_view(mX, row, n)
        row_y = aligned_row_view(mY, row, n)
        gx = cute.local_tile(row_x, (1, TILE_N), (0, None))
        gy = cute.local_tile(row_y, (1, TILE_N), (0, None))
        coords = cute.local_tile(
            cute.make_identity_tensor(row_x.shape), (1, TILE_N), (0, None))

        smem = utils.SmemAllocator()
        sx = smem.allocate_tensor(Float32, smem_layout, 16)
        tc = cp_copy.get_slice(tidx)
        ts = uni_copy.get_slice(tidx)
        tcgx, tcsx = tc.partition_S(gx), tc.partition_D(sx)
        tsgy = ts.partition_D(gy)
        tcoord = tc.partition_S(coords)
        rx = cute.make_rmem_tensor(tcsx.shape, Float32)
        ry = cute.make_rmem_tensor(tcsx.shape, Float32)


        pred = cute.make_rmem_tensor(
            cute.make_layout((cute.size(tcoord, mode=[0, 1]), 1, 1)),
            cutlass.Boolean)
        num_tiles = cute.size(gx, mode=[2])
        for nt in cutlass.range(num_tiles):
            for v in cutlass.range_constexpr(pred.shape[0]):
                pred[v, 0, 0] = cute.elem_less(
                    tcoord[(0, v), 0, 0, nt][1], n)
            cute.copy(cp_copy, tcgx[None, None, None, nt], tcsx, pred=pred)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            cute.autovec_copy(tcsx, rx)
            ry.store(cute.math.exp2(
                rx.load() * alpha * Float32(1.4426950408889634), fastmath=True))
            cute.copy(uni_copy, ry, tsgy[None, None, None, nt], pred=pred)
            cute.arch.sync_threads()

    @cute.kernel
    def kernel_tma_gather4(
        self, mX, mY, mIdx, alpha,
        load_atom, store_atom, uni_copy, smem_layout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        group, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        n = mX.shape[1]


        g0 = group * 4
        rows = [mIdx[g0 + i] for i in range(4)]

        smem = utils.SmemAllocator()
        storage = smem.allocate(TmaSharedStorage)
        sx = smem.allocate_tensor(Float32, smem_layout, 128)
        sy = smem.allocate_tensor(Float32, smem_layout, 128)
        load_mbar = storage.load_mbar.ptr
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(load_mbar, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()


        load_desc = get_tma_desc_addr(load_atom)
        store_desc = get_tma_desc_addr(store_atom)
        tc = uni_copy.get_slice(tidx)
        tcsx = tc.partition_S(sx)
        tcsy = tc.partition_D(sy)
        rx = cute.make_rmem_tensor(tcsx.shape, Float32)
        ry = cute.make_rmem_tensor(tcsx.shape, Float32)

        num_tiles = cute.ceil_div(n, TILE_N)
        for nt in cutlass.range(num_tiles):
            phase = nt % 2
            if warp_idx == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        load_mbar, 4 * TILE_N * Float32.width // 8)
                    tma_gather4_load(
                        load_desc, sx.iterator, load_mbar,
                        Int32(nt * TILE_N), rows)
            cute.arch.mbarrier_wait(load_mbar, phase)
            cute.arch.sync_threads()

            cute.autovec_copy(tcsx, rx)
            ry.store(cute.math.exp2(
                rx.load() * alpha * Float32(1.4426950408889634), fastmath=True))
            cute.copy(uni_copy, ry, tcsy)
            cute.arch.fence_view_async_shared()
            cute.arch.sync_threads()
            if warp_idx == 0:
                with cute.arch.elect_one():
                    tma_scatter4_store(
                        store_desc, sy.iterator, Int32(nt * TILE_N), rows)
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.sync_threads()


def run_one(R, N, G, mode, verbose=True):
    assert G % 4 == 0, "Gather4/Scatter4 requires groups of exactly four rows"
    torch.manual_seed(11)
    x = torch.randn(R, N, dtype=torch.float32, device="cuda")
    y = torch.zeros(R, N, dtype=torch.float32, device="cuda")
    idx = torch.randperm(R, device="cuda", dtype=torch.int32)[:G].contiguous()
    alpha = 0.5

    import cuda.bindings.driver as cuda_drv
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

    op = GatherScatter(mode)
    args = (
        from_dlpack(x, assumed_align=16),
        from_dlpack(y, assumed_align=16),
        from_dlpack(idx, assumed_align=16),
        Float32(alpha),
        stream,
    )
    compiled = cute.compile(op, *args)
    compiled(*args)
    torch.cuda.synchronize()

    ref = torch.zeros_like(y)
    rows = idx.long()
    ref[rows] = torch.exp(alpha * x[rows])
    err = (y - ref).abs().max().item()

    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(5):
        compiled(*args)
    start.record()
    for _ in range(20):
        compiled(*args)
    stop.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(stop) / 20 * 1e3

    ok = err < 1e-5
    if verbose:
        print(f"{'OK  ' if ok else 'FAIL'} {mode:9s} R={R:5d} N={N:5d} G={G:5d} "
              f"{us:8.2f} us  max_err={err:.2e}")
    return ok, us, y


def main():
    all_ok = True
    for (R, N, G) in [(4096, 512, 64), (4096, 512, 4096)]:
        ts = {}
        outputs = {}
        for mode in ("cpasync", "tma_gather4"):
            ok, us, outputs[mode] = run_one(R, N, G, mode)
            all_ok &= ok
            ts[mode] = us
        identical = torch.equal(outputs["cpasync"], outputs["tma_gather4"])
        all_ok &= identical
        print(f"     paths bit-identical: {identical}")
        if all(v > 0 for v in ts.values()):
            print(f"     G={G}: tma_gather4 / cpasync = "
                  f"{ts['tma_gather4'] / ts['cpasync']:.2f}x")
    print("PASS" if all_ok else "FAIL")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
