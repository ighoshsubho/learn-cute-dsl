import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack


def check(name, got, want):
    assert got == want, f"{name}: got {got}, want {want}"
    print(f"  ok  {name} = {got}")


@cute.jit
def exercise(mX: cute.Tensor):


    A = cute.make_layout((8, 16), stride=(16, 1))
    tiler = (4, 4)
    print("A:", A)

    ld = cute.logical_divide(A, tiler)
    print("logical_divide:", ld)

    expected_ld = ((4, 2), (4, 4))
    check("logical_divide shape", cute.shape(ld), expected_ld)

    zd = cute.zipped_divide(A, tiler)
    print("zipped_divide: ", zd)
    # ((TileShape), (RestShape)) — one whole tile first, tile grid second
    expected_zd = ((4, 4), (2, 4))
    check("zipped_divide shape", cute.shape(zd), expected_zd)
    check("elements per tile", cute.size(zd, mode=[0]), 16)
    check("number of tiles", cute.size(zd, mode=[1]), 8)

    fd = cute.flat_divide(A, tiler)
    print("flat_divide:   ", fd)
    expected_fd = (4, 4, 2, 4)
    check("flat_divide shape", cute.shape(fd), expected_fd)


    # (bi,bj), i.e. global coord (bi*4+ti, bj*4+tj):
    check("zd(((1,2),(1,3)))", zd(((1, 2), (1, 3))), A((1 * 4 + 1, 3 * 4 + 2)))


    print("mX:", mX)
    tX = cute.zipped_divide(mX, (8, 32))
    print("zipped_divide(mX, (8,32)):", tX)
    # (16,64) cut into (8,32) tiles -> 2x2 tile grid
    expected_tX = ((8, 32), (2, 2))
    check("tensor zipped_divide shape", cute.shape(tX), expected_tX)

    fX = cute.flat_divide(mX, (8, 32))
    expected_fX = (8, 32, 2, 2)
    check("tensor flat_divide shape", cute.shape(fX), expected_fX)


if __name__ == "__main__":
    x = torch.zeros(16, 64, dtype=torch.float32, device="cuda")
    exercise(from_dlpack(x))
    print("PASS")
