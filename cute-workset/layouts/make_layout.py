import cutlass
import cutlass.cute as cute


def check(name, got, want):
    assert got == want, f"{name}: got {got}, want {want}"
    print(f"  ok  {name} = {got}")


@cute.jit
def exercise():


    A = cute.make_layout((4, 8), stride=(8, 1))
    print("A (row-major):", A)


    coords_and_offsets = [((0, 0), 0), ((1, 2), 10), ((3, 7), 31)]
    for coord, want in coords_and_offsets:
        check(f"A{coord}", A(coord), want)

        check(f"crd2idx({coord}, A)", cute.crd2idx(coord, A), want)


    B = cute.make_layout((4, 8))
    print("B (default = col-major):", B)

    check("B((1,2))", B((1, 2)), 9)


    H = cute.make_layout(((2, 64), 8), stride=((1, 16), 2))
    print("H (hierarchical):", H)
    check("size(H)", cute.size(H), 2 * 64 * 8)


    h_expected = 59
    check("H(((1,3),5))", H(((1, 3), 5)), h_expected)


    O_row = cute.make_ordered_layout((3, 5), order=(1, 0))
    print("ordered, order=(1,0):", O_row)
    check("O_row((1,2))", O_row((1, 2)), 1 * 5 + 2)


    O_col = cute.make_ordered_layout((3, 5), order=(0, 1))
    print("ordered, order=(0,1):", O_col)
    check("O_col((1,2))", O_col((1, 2)), 1 + 2 * 3)


    check("size(A)", cute.size(A), 32)
    check("cosize(A)", cute.cosize(A), 32)


    P = cute.make_layout((4, 8), stride=(16, 1))
    print("P (padded, ld=16):", P)
    p_cosize = 56
    check("size(P)", cute.size(P), 32)
    check("cosize(P)", cute.cosize(P), p_cosize)


if __name__ == "__main__":
    exercise()
    print("PASS")
