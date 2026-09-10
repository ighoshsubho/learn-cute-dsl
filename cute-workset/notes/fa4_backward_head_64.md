blog - https://research.colfax-intl.com/optimization-diaries-improving-flashattention-4-backward-for-head-dimension-64/
- for head dim 128 its fairly optimized backward due to tmem full utilization, relay warps in use, dsmem collab
- for 64 not that optimized the tmem left empty while still doing aliasing
- 256 thread barrier syncs with warp syncs if we de alias
- s=q@kt, p=exp(s-l), dp=do@vt, dv=pt@do, dkt=qt@ds or dk=dst@q, dq=ds@k
| role         | warps | description                                                                                                   |
|--------------|-------|----------------------------------------------------------------------------------------------------------------|
| load         | 1     | issues the TMA loads K, V, Q and dO                                                                            |
| MMA          | 1     | issues every `tcgen05.mma` instruction                                                                         |
| compute      | 8     | computes P from S, and computes dS from dP                                                                     |
| reduce       | 4     | reads each partial dQ from tensor memory and adds it into the global accumulator                               |
| relay/empty  | 2     | idle at hdim 64; the relay warp is used only in the hdim 128 case, where 2-CTA MMA instructions are used       |
- compute warps and mma warps focus, compute do pointwise ops and mma does 5 mmas
- gemm flops scale with head dimension but at 64 it halves tensor core work
- FLOPs by compute warps are independent so it becomes difficult since compute warps becomes bottlenecked hence compute warps are 8
- baseline pre optimization - S(0-128) with aliased P(0-64) | dv (128-192) | dp (192-320) with ds/dq | dk (320-384) | rest free
- TMEM allows warps to access 0-32 lanes
- since 8 warps, warps w+4 in the first 0-3 can hinder the next warp loading hence compute wide barrier
- cute.arch.fence_view_async_tmem_load() -> self.compute_sync_barrier.arrive_and_wait()
- tell s read and p written at same time 
"""
cute.arch.fence_view_async_tmem_store()
cute.arch.fence_view_async_shared()
self.compute_sync_barrier.arrive_and_wait()
with cute.arch.elect_one():
    pipeline_S_P.consumer_release(consumer_state) # "S read and P written"
    pipeline_LSE.consumer_release(consumer_state_LSE)
"""
- so issue order qk(t+1) dk(t) dq(t) dp(t+1) pdo(t+1)
- IKET trace shows this and how softmax overlaps very less mmas
- optimization 1 - store P and ds in tmem slot for 64 head dim
- no alias guards required
- we can now release S earlier and dont need to wait for dV computation
- no 8 warp wide sync rather sync warp due to tmem load its a weak sync but across threads
- so it becomes qk(t+1) dp(t) dk(t) dq(t) pdo(t+1)
- non deterministic good speedup but at determ due to ordering of dq accum we see difference