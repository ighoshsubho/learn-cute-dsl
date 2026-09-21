blog - https://iaroslavelistratov.github.io/b200-attention/
notes -
- due to assymetric hardware scaling in b200 the softmax throughput became bottleneck compared to tensor cores
- as many cycles as the beefy mmas
- make the softmax cheaper and overlapinng with matmuls
- we copy q tile from gemem to smem in slices
- slices along k or reduction dim
- load in micro tiles like 8 tile for block k of 128 so 16 its
- and mma consumes 16 k so it becomes 8 its
- the slice we use for tma load that is 8x8 is a single swizzle atom mma expects
- mma needs info on how to step between atoms
- lbo along reduction and sbo along non reduction
- lbo is step over one slice sbo is step over one atom
- 3 pass over tmem for all fwd stuff
- one over s to produce max, one to produce p with scale and stuff and one to rescale o
- now tmem can be accessed by 4 warps with all 32 lanes seperately
- 32,128 leads to register spills hence tile it
- 128,8 chunk so read 16 its
- tmem address contains 16 upper bit row | 16 lower bit col
- information is like this to access data from every lane -
row_base << 16:  [row_base][  0   ]
column:          [   0    ][column]
sum:             [row_base][column]

(here, column = taddr_s + score_n8 * 8)
- in first iteration get the max dont change s
- 2nd iteration change s by writing p one tile at a time 128x8
- before we do 2nd iteraton on tmem to produce s we want to rescale o
- since we cant start pv before it otherwise accum gets wrong
- v needs to be copied via tma in 8,8 unswizzled due to mn major reduction, we cant copy whole slice
- lbo and sbo becomes mn major
- we do mma iterating on slices and tma copy from gmem to smem from block k in 8 chunks and block n in 8 chunks
- in first kernel we use smem for p but for tmem we need to sync after tcgen05.st
- for 128b swizzle 64 wide atom head dim is adapted
- we need only sbo here not lbo, sbo will determine atom movement and an atom is 8,64 but mma is 16 at k
- hence we loop over 4 micro tiles but we dont specify lbo
- add warp specialization without any independant stuff
- next kernel loads 2 indep q tiles and loops over same kv 
- now since we have 2 q tiles we also have 2 indep softmax warps for indep work
- even if there are 2 q tiles or indep loads in load warp we dont need extra warps
- since these are async they dont block the work they just need inputs ready
- load pipeline is slot 3 buffer with kv in same buffer
- prologue pre loads the k0 v0 k1 and first iter waits on k0 v0
- pipeline the load with this single buffer probably to stage loads without tma mem fragmentaiton?
- so here in compute pipeline we do a prologue prefill of q0 and q1 with ki
- then the next iterations in middle it does p0v q0ki+1 p1v q1ki+1
- just qki+1 issue after pvi for better overlap than p1vi
- inline ptx instead of using wrapper since it does log2 conversion in hot loop
- then we have software approximations - 75% mufu and 25% alu
- hardware uses mufu and software uses other approximations but we inteligently move resources
- software emulaition is proofed to be correct by flash attention check once
- now cache s in registers since we have 2 s passes but spills heavily for full s
- so cache last 96 cols and not first 32
- if we instead do first 96 we clobber by writing p
- since this 96 is cached we release them with split p arrival before we read first 32
- conditional rescale for o in softmax computation
- seperate correction warps applied shared by both q tile (1 warp group correction warp)
- slicing p for split p arrival also slices v so mind that slice 1 is later 96 and slice 2 is first 32
- persistent work items for l2 cache reuse
- persistent helps not reinitialize or invalidate the work items that are done
- now small optims -
    - overlap tmem to regs with softmax
    - cached 96 cols and 32 as seperate partial sums and then compute final scalar row sum
    - phase bitmasks is integer packed in registers removing spills
    - TMA l2 promotion
