import cutlass
import cutlass.cute as cute


def check(name, got, want):
    assert got == want, f"{name}: got {got}, want {want}"
    print(f"  ok  {name} = {got}")


@cute.jit
def exercise():


    L_flat = cute.make_layout((2, 4), stride=(1, 2))
    C_flat = cute.coalesce(L_flat)
    print("coalesce((2,4):(1,2)) ->", C_flat)
    check("flattened shape", cute.shape(C_flat), 8)
    check("flattened stride-1 run", [C_flat(i) for i in range(8)],
          [0, 1, 2, 3, 4, 5, 6, 7])


    L_rm = cute.make_layout((2, 4), stride=(4, 1))
    C_rm = cute.coalesce(L_rm)
    print("coalesce((2,4):(4,1)) ->", C_rm)
    check("cannot flatten", cute.shape(C_rm), (2, 4))


    for i in [0, 3, 5, 7]:
        check(f"C_flat({i})==L_flat({i})", C_flat(i), L_flat(i))
        check(f"C_rm({i})==L_rm({i})", C_rm(i), L_rm(i))


    R = cute.make_layout((4, 8), stride=(8, 1))
    S = cute.make_layout((8, 4), stride=(4, 1))
    RS = cute.composition(R, S)
    print("R:", R)
    print("S:", S)
    print("R o S:", RS)


    rs_at_1 = 1
    check("(RoS)(1) by hand", RS(1), rs_at_1)

    for i in [0, 1, 5, 13, 31]:
        check(f"RS({i}) == R(S({i}))", RS(i), R(S(i)))


    check("transposed shape", cute.shape(RS), (8, 4))
    for i, j in [(0, 0), (2, 1), (7, 3), (5, 2)]:
        check(f"RS(({i},{j})) == R(({j},{i}))", RS((i, j)), R((j, i)))


if __name__ == "__main__":
    exercise()
    print("PASS")
