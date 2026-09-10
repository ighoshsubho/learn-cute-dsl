link - https://www.aryatschand.com/blog/blog-tpu-gpu-comparison.html
nvidia arches -
- ampere
    - tensor cores unavoidable
    - software pipelining through smem and solving bank conflicts
    - low occupancy by design due to register pressure and shared mem
    - trade parallel for register reuse and instruction throughput
    - resource usage vs instruction schedule

- hopper
    - introduce tma
    - poorly choosen tiles cap performance
    - warpgroup async mma and dsmem and several other advancements

- blackwell
    - tcgen mma ops
    - tmem ops

amd arch -
- mi300 and 350 cdna 3 and 4 -
    - SIMD execution model instead of SIMT
    - vectorization and predication is explicit
    - tensorcores low level than nvidia
    - prone to layouts data alignment and shapes
    - various cache heirarchy
    - profile else cant understand if its mem bandwidth, sync overhead or instruction throughput
    - they are made of 8 XCDs or dies compared to 2 in blackwell/rubin
    - data locality hence becomes important

tpu arch -
- tpuv7 ironwood
    - you need to compile for one specific shape
    - instruction free fetch of different loads while current compute is happening
    - tile sizes mem layout loop structure are autotune performance more imp
    - systolic array fully utilized for perf
