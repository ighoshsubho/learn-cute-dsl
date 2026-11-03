blog - https://drisspg.github.io/nuggets/TorchInductor-Walkthrough
- at pregrad both nn.relu and x.relu exists not yet normalized
- aliasing and mutations remain
- fwd and backward not yet traced
- tensor subclass not yet been traced
- automated mixed precision which affect dispatching will not be present
- graph manipulation not recommended at this stage
- at jointgraph -
- compisite operators were broken down
- decompositions were also applied here
- used when we need to modify both fwd and bkwd
- recomputing allow to fuse and also interestingly broadcast stuff
- scheduler node which decides how to organize for efficient computation
- 1. with same buffer access group stuff for fusions for less HBM access
- 2. rank fusions with amount of global mem saved
- 3 things checked here ranking, legality and dynamic updating
- they do speedup by fusion with triton template to finalize based on profitability
- they async execute on different templates and at the end decide for fusion
- the above are verticak fusion and the combo ones are horizontal
- horizontal means they dont fuse readily but async execute due to no data dependency
- it also does loop reordering after fusion
- loop reodering is different from loop reindexing
- mix order reductions basically needs one partial and one full reduction
- nested order reductions does reductions normally in phase 1 and in phase 2 basically reorders in registers and do reductions
- some more reductions -
Reorder for Peak Memory - on by default

Cudagraph Partioning (soon to be enabled)

Reorder Compute And Comm (off by default, though being worked on)
