import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.jit
def run(mA: cute.Tensor, mB: cute.Tensor):


    # (16,8):(8,1): its output 8*j + i is a 1-D index into mA's domain, and


    mAt = cute.composition(mA, cute.make_layout((16, 8), stride=(8, 1)))
    print("(a) mA :", mA.layout)
    print("    mAt:", mAt.layout)
    assert mAt.layout == cute.make_layout((16, 8), stride=(1, 16))

    ok = all(
        cute.crd2idx((j, i), mAt.layout) == cute.crd2idx((i, j), mA.layout)
        for i in range(8)
        for j in range(16)
    )
    assert ok, "transpose view must alias mA[i, j] at mAt[j, i]"


    mB_f32 = cute.recast_tensor(mB, cutlass.Float32)
    print("(b) bf16  :", mB.layout)
    print("    as f32:", mB_f32.layout, mB_f32.element_type)
    assert mB_f32.layout == cute.make_layout((8, 32), stride=(32, 1))


    mB_i128 = cute.recast_tensor(mB, cutlass.Int128)
    print("    as i128:", mB_i128.layout)
    assert mB_i128.layout == cute.make_layout((8, 8), stride=(8, 1))

    assert cute.recast_tensor(mB_f32, cutlass.BFloat16).layout == mB.layout


    r4 = cute.make_layout((2, 3, 4, 5), stride=(60, 20, 5, 1))
    picked = cute.select(r4, mode=[0, 2])
    grouped = cute.group_modes(r4, 1, 3)
    print("(c) rank-4 :", r4)
    print("    select [0,2] :", picked)
    print("    group [1,3)  :", grouped)
    assert picked == cute.make_layout((2, 4), stride=(60, 5))
    assert grouped == cute.make_layout((2, (3, 4), 5), stride=(60, (20, 5), 1))

    assert cute.crd2idx((1, (2, 3), 4), grouped) == cute.crd2idx((1, 2, 3, 4), r4)


    bcast = cute.make_layout((8, 4), stride=(1, 0))
    filt = cute.filter_zeros(bcast)
    print("(d) broadcast    :", bcast, " size", cute.size(bcast), " cosize", cute.cosize(bcast))
    print("    filter_zeros :", filt, " size", cute.size(filt))


    assert filt == cute.make_layout((8, 1), stride=(1, 0))
    assert cute.size(filt) == cute.cosize(bcast) == 8


if __name__ == "__main__":


    a = torch.arange(8 * 16, dtype=torch.float32, device="cuda").reshape(8, 16)
    b = torch.zeros(8, 64, dtype=torch.bfloat16, device="cuda")
    run(from_dlpack(a, assumed_align=16), from_dlpack(b, assumed_align=16))
    print("PASS")
