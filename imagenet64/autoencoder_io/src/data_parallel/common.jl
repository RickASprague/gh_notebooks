# Data-parallel primitives shared by data_parallel/verify.jl and data_parallel/train.jl.

include(joinpath(@__DIR__, "..", "shared.jl"))
using Functors

const NREP         = 5                     # simulated replicas
const MICRO        = BATCH                 # per-replica batch: the normal batch (shared.jl)
const GLOBAL_BATCH = NREP * MICRO          # 5120 images per optimizer step

clone_params(p) = fmap(copy, p)

# All-reduce (mean) over N gradient trees, leaf by leaf. Param-less layers
# (relu, sigmoid) get a `nothing` gradient from Zygote; it must pass through
# untouched -- `sum((nothing, nothing, ...))` has no method.
_leafmean(xs...) = xs[1] === nothing ? nothing : sum(xs) ./ length(xs)
avg_grads(gs) = fmap(_leafmean, gs...)

# CUDA kernels are async: without a sync we would time the launch, not the
# work. In data_parallel/train.jl the syncs are also load-bearing for CORRECTNESS: all
# replicas stage through one xb_gpu, so replica r's backward pass must finish
# before replica r+1 overwrites the buffer.
function timed(f)
    CUDA.synchronize()
    t0 = time_ns()
    r = f()
    CUDA.synchronize()
    return r, (time_ns() - t0) / 1e9
end

tree_leaves(x)     = [l for l in Functors.fleaves(x) if l isa AbstractArray]
tree_absmax(t)     = maximum(maximum(abs.(l)) for l in tree_leaves(t))
tree_maxdiff(a, b) = maximum(maximum(abs.(x .- y)) for (x, y) in zip(tree_leaves(a), tree_leaves(b)))
