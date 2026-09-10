Pranjal NVFP4 KERNEL - https://cudaforfun.substack.com/p/outperforming-cublas-on-nvfp4
Warps  Role
-----  --------------------------------------------------------------------------------
0-3    Read FP32 accumulators from TMEM, convert to FP16, and store
4      Copy scale factors from SMEM to TMEM and issue every MMA (which outputs to TMEM)
5      Load A and B into shared memory with TMA
6      Load A and B scale factors into shared memory with TMA

- K=96 mma since gb300 allows this for same cycles as 64
- 128x192 packed for NVFP4 for a CTA for both A and B load and for SFA, SFB 
its 128x12 logically but SFB gets replicated as 256x12
- Since 128x12 we need to use warpX4 or cp4 in cute dsl
- 128 rows x 12 bytes - 1536/512 which is 3 and same for SFB with 256 is 6
- occupy 36 TMEM cols and packs 2 mma to use full cp instruction
- warps 0-3 handle drain since 32x128 ld from tmem earch warp ows 32 roes
- uniform registers like %laneid allowe fewer instructions
- K=256 with pipeline, few things to keep in mind like non 128 multiple
- need to wait extra time until mma is done and options like free tma or scale
- Overlap the accumulator like if 2 mma fill 2*256 then we left with nothing
- but we can always overlap once its done reading 36 cols 2 tile and this
- int8 or 256 bit loads optimization
- for one tile use L1:no_allocate and L2:evict_first
- compiler hints like cluster dims, launch bounds, block size, fallback cluster
- Dont issue tail mma which are 0s
- ALso may fold tail with mma k=64
- L2 cache hilbert curve/swizzle
- GB300 2 die arch where near means more bandwidth far means less (through
interconnect)
- 4 kb memory segment fall in 1 side and 4 other for 8kb span
- L2 ownership where "If A[m,:] is going to be reused many times, make 
the same group of SMs perform all those reuses rather than splitting them 
across both GPU sides." + 8x8 sub grouping for L2 reuse/swizzle