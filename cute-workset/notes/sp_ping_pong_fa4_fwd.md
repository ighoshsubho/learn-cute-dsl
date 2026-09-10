blog - https://research.colfax-intl.com/optimization-diaries-s-p-ping-pong-for-flashattention-4-decode/
- only 128 block q available
- the other slot (since fa4 try to parallel sq*2 tiles) is free in tmem
- we can do qk(i+1) at the other slot while the softmax is still running for qk(i)
- so this becomes s/p pingpong since softmax produces p so p(i) overlaps with s(i+1)
- then pv is done without anything in parallel since it writes o and o is shared by all
- softmax is also overlapped somewhat with pv due to split p arrival
- correction warps correct o accumulator
- prologue since we need to make this pipeline working at beginning which is prologue since -
    - we want s0 to be ready while we do gemm of s1
    - s1 should have its loads arrived
    - we also have different barriers for different tmem slots 
    - bit 0 tracts barrier bit 1 determines how many times the slot has flipped
- for a recap - consumer does -> consumer_wait and consumer_release and producer does -> producer_acquire and producer_commit
"""
if const_expr(self.use_s_ping_pong):
   # Wait for Q
   pipeline_q.consumer_wait_w_index_phase(0, mma_q_consumer_phase)
   # Wait for K(0)
   pipeline_kv.consumer_wait(mma_kv_consumer_state)
   Ki_index, Ki_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
   sK_cur = sK[None, None, None, Ki_index]
   if const_expr(self.uneven_kv_smem):
       sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
   # Issue QK(0).
   if (mma_pv_count & 1) == 0:
       gemm_Si[0](smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator))
       pipeline_s_p_o.producer_commit_w_index(0)
   else:
       gemm_Si[1](smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator))
       pipeline_s_p_o.producer_commit_w_index(1)
   mma_q_consumer_phase ^= 1 # we use q only for 1 slot not multiple hence manual phase flip
   # Release K(0)
   pipeline_kv.consumer_release(mma_kv_consumer_state)
   # Advance to K(1)
   mma_kv_consumer_state.advance() 
   O_should_accumulate = False
"""
- main loop code -
"""
for i in cutlass.range(block_iter_count - 1, unroll=1):
   # Wait for K(i+1)
   pipeline_kv.consumer_wait(mma_kv_consumer_state)
   Ki_index, Ki_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
   sK_cur = sK[None, None, None, Ki_index]
   if const_expr(self.uneven_kv_smem):
       sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
   # Issue QK(i+1). Even global block count writes to TMEM slot 0 (0-127) and odd writes to TMEM slot 1 (128-255)
   if ((mma_pv_count + 1) & 1) == 0:
       gemm_Si[0](smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator))
       pipeline_s_p_o.producer_commit_w_index(0)
   else:
       gemm_Si[1](smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator))
       pipeline_s_p_o.producer_commit_w_index(1)
   # Release K(i+1)
   pipeline_kv.consumer_release(mma_kv_consumer_state)
   # Advance to V(i)
   mma_kv_consumer_state.advance()
   # Wait for V(i)
   pipeline_kv.consumer_wait(mma_kv_consumer_state)
   Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
   tOrVi = tOrV[None, None, None, Vi_index]
   sV_cur = sV[None, None, None, Vi_index]
   if const_expr(self.uneven_kv_smem):
       sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
   # Phase for this block's slot. Each slot is reused every second block, and its barrier phase flips on each reuse.
   pv_phase = (mma_pv_count >> 1) & 1
   # Issue PV(i)
   if (mma_pv_count & 1) == 0:
       pipeline_s_p_o.producer_acquire_w_index_phase(0, pv_phase)
       gemm_Pi[0](
           tCrB=tOrVi,
           sB=sV_cur,
           zero_init=not O_should_accumulate,
           mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(0) if self.split_P_arrive > 0 else None,
           mbar_phase=pv_phase,
       )
   else:
       pipeline_s_p_o.producer_acquire_w_index_phase(1, pv_phase)
       gemm_Pi[1](
           tCrB=tOrVi,
           sB=sV_cur,
           zero_init=not O_should_accumulate,
           mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(1) if self.split_P_arrive > 0 else None,
           mbar_phase=pv_phase,
       )
   pipeline_o_acc.producer_commit_w_index(mma_pv_count & 1)
   mma_pv_count += 1
   # Release V(i)
   pipeline_kv.consumer_release(mma_kv_consumer_state)
   # Advance to K(i+2)
   mma_kv_consumer_state.advance() 
   O_should_accumulate = True
"""
- here is a pic idea is simple - after prologue is done where you issue k(i+1) and complete gemm s0 for mainloop -
    - kv consumer wait for k i+1
    - issue qk i+1 for 2nd slot
    - release k i+1 slot
    - advance to vi and wait for load
    - issue pv
    - rease vi
    - issue k i + 2
- iket profiling for base shows no overlap while with this optimization shows qk next mma overlap
- iket question asked about split p arrival here - https://research.colfax-intl.com/optimization-diaries-s-p-ping-pong-for-flashattention-4-decode/#comment-2304