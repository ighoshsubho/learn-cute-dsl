NVFP4 blog
block scaled mma notes -
"quads" stored by each thread see docs
SFA or SFB stored in half or one of the quads in that thread
for SFA it is ((2, 2, 8),(16, 4)):((32, 0, 4),(0, 1))
16 threads if you calculate and 2 is 0 since we ignore the rest
for SFB ((4, 8), (16, 4)):((0, 4), (0, 1))
since 1 scale factor due to 1 quad compared to 4 in SFA(see nvidia docs)
producer 1 dma and 8 mma (since sync warp mma)
warp layout 4,2,1 in total 8 warps with mma m16n8k64 so becomes m64n16k64
do TMA copy from gmem to smem with helpers
smem layout - (((32,4), REST_M/N), ((16,4), 1, REST_K)) : (((16, 4), 512 * REST_K), ((0, 1), 4, 512))
64/16=4 32*4 rows and for every row 16,4 or basically 4 scales for 16 values and rest k
SMEM to RMEM copy, tiled copy -> thr,val layout from helpers -> make_fragment -> retile
filter zeros and copy
do mma and stuff

NVFP4 OPTIM blog
1. Version 1 same kernel as previous
2. version 2 check performance graph on ncu like sm sampling, dram throughput, l2 cache, l1 cache
gemm is compute bound, so ideally you want to keep data in fast stuff possible
like l2/l1 or registers but l2 hit was low hence we need tb swizzle
3. Improve epiligue - previously ->
producer acquire and wait ->
stage register work like alpha load and compute ->
barrier ->
then copy from RMEM to SMEM ->
fence and barrier ->
tma copy from SMEM -> GMEM ->
commit pipeline
we can rather producer acquire after register work
4. Warp specialized like 1 load warp, 8 mma warps, 1 store warp
5. SFA bank conflict which is 2 way since SMEM layout
basically broadcasts it (2,2,8) <- middle 2 is stride 0
basically use the complement layout locations so we dont get
conflicts like generally for bank m=m0+32m1
6. 12 mma warps and warp layout (6,2,1) so we dont have wave
quantization at 2k ctx
7. Autotuning stuff and clc with custom layout since by default
it has understanding of 2 cta stuff
clc did not perform well though