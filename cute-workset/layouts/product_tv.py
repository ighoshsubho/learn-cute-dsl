import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

THREADS = 64


def check(name, got, want):
    assert got == want, f"{name}: got {got}, want {want}"
    print(f"  ok  {name} = {got}")


@cute.kernel
def stamp_owner_kernel(mO: cute.Tensor, tiler_mn: cutlass.Constexpr,
                       tv_layout: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()

    # ((TileM,TileN),(RestM,RestN)) — the (16,64) tensor as 2 stacked tiles.
    gT = cute.zipped_divide(mO, tiler_mn)
    print("[trace] gT:", gT)

    for t in cutlass.range_constexpr(cute.size(gT, mode=[1])):
        blk = gT[(None, t)]


        tidfrg = cute.composition(blk, tv_layout)  # ((8,8),8): (thr, val)
        thr = tidfrg[(tidx, None)]
        print("[trace] per-thread view:", thr)
        for v in cutlass.range_constexpr(cute.size(thr)):
            thr[v] = tidx


@cute.jit
def run(mO: cute.Tensor):


    thr = cute.make_layout(4)
    val = cute.make_layout(2)
    print("logical_product(thr, val):", cute.logical_product(thr, val))

    B = cute.blocked_product(thr, val)
    R = cute.raked_product(thr, val)
    print("blocked_product(thr, val):", B)
    print("raked_product(thr, val):  ", R)


    owners_blocked = [B(i) % 4 for i in range(8)]
    owners_raked = [R(i) % 4 for i in range(8)]
    print("blocked_product owner per position:", owners_blocked)
    print("raked_product owner per position:  ", owners_raked)


    expected_blocked = [0, 1, 2, 3, 0, 1, 2, 3]
    expected_raked = [0, 0, 1, 1, 2, 2, 3, 3]
    check("blocked_product => cyclic ownership", owners_blocked, expected_blocked)
    check("raked_product => contiguous ownership", owners_raked, expected_raked)


    thr_layout = cute.make_ordered_layout((8, 8), order=(1, 0))
    val_layout = cute.make_layout((1, 8))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    print("tiler_mn:", tiler_mn)
    print("tv_layout:", tv_layout)
    check("tiler", tiler_mn, (8, 64))


    for t in [0, 9, 26, 63]:
        for v in [0, 7]:
            pos = tv_layout((t, v))
            row, col = pos % 8, pos // 8
            check(f"thread {t} value {v} -> (row,col)", (row, col),
                  (t // 8, 8 * (t % 8) + v))


    stamp_owner_kernel(mO, tiler_mn, tv_layout).launch(
        grid=[1, 1, 1], block=[THREADS, 1, 1])


if __name__ == "__main__":
    out = torch.full((16, 64), -1, dtype=torch.int32, device="cuda")
    run(from_dlpack(out))
    torch.cuda.synchronize()


    rows = torch.arange(16).view(16, 1)
    cols = torch.arange(64).view(1, 64)
    expected = (8 * (rows % 8) + cols // 8).to(torch.int32)
    assert torch.equal(out.cpu(), expected), "ownership pattern mismatch"
    assert int(out.min()) == 0 and int(out.max()) == THREADS - 1
    print("  ok  GPU ownership matches the TV layout prediction")
    print("PASS")
