# Two arithmetic checks on a single global batch, before timing anything.
#
#   julia --project src/data_parallel/verify.jl
#
# 1. 5 replicas x 1024 (gradients averaged) vs one batch of 5120.
#    Expected to DIFFER: BatchNorm statistics are per replica.
# 2. Sequential replicas in forward vs reversed visit order.
#    Expected to be BIT-IDENTICAL; reduction order may move the last bits.

include(joinpath(@__DIR__, "common.jl"))

N, ys, Xs, μs = load_train()
model = build_model()
dev = gpu_device()
@printf("parameters: %d\n", Lux.parameterlength(model))

function init()
    rng = Random.default_rng(); Random.seed!(rng, 0)
    ps, st = Lux.setup(rng, model)
    return ps |> dev, st |> dev
end

# Micro-batches are materialized contiguously, not @view'd: cuDNN wants dense
# input, and a strided view would change what is being compared.
micro(r) = CuArray(Xs[1][:, :, :, (r-1)*MICRO+1:r*MICRO])

println("\n── 1. averaged micro-batch gradient vs one large batch ──")
let
    ps0, st0 = init()
    gs = []; ls = Float32[]
    for r in 1:NREP
        g, ℓ, _ = mse_grad(model, ps0, st0, micro(r))
        push!(gs, g); push!(ls, ℓ)
    end
    ḡ = avg_grads(gs)
    gbig, ℓ_big, _ = mse_grad(model, ps0, st0, CuArray(Xs[1][:, :, :, 1:GLOBAL_BATCH]))

    # avg_grads must be the leafwise MEAN (not the sum), checked independently
    lav, lrep = tree_leaves(ḡ), [tree_leaves(g) for g in gs]
    err = maximum(maximum(abs.(lav[k] .- sum(lrep[r][k] for r in 1:NREP) ./ NREP)) for k in eachindex(lav))
    @assert err == 0 "avg_grads is not the leafwise mean (err = $err)"

    @printf("per-replica losses  : %s\n", join((@sprintf("%.6f", x) for x in ls), "  "))
    @printf("mean of replicas    : %.6f\n", mean(ls))
    @printf("single 5120 batch   : %.6f\n", ℓ_big)
    @printf("max rel |ḡ - g_big| : %.3e\n", tree_maxdiff(ḡ, gbig) / tree_absmax(gbig))
end

println("\n── 2. visit order and reduction order ──")
let
    function dp_grads(order)
        ps, st = init()
        replicas = [clone_params(ps) for _ in 1:NREP]
        drift = maximum(tree_maxdiff(replicas[r], ps) for r in 1:NREP)
        grads = Vector{Any}(undef, NREP)
        for r in order
            grads[r] = mse_grad(model, replicas[r], st, micro(r))[1]   # stored by replica id
        end
        return ps, grads, drift
    end
    step(ps, ḡ) = Optimisers.update(Optimisers.setup(Optimisers.Adam(1f-3), ps), ps, ḡ)[2]

    ps_f,  gf,  drift = dp_grads(1:NREP)
    ps_f2, gf2, _     = dp_grads(1:NREP)             # control: same order, run again
    ps_r,  gr,  _     = dp_grads(reverse(1:NREP))
    ḡ_f, ḡ_f2, ḡ_r, ḡ_rr = avg_grads(gf), avg_grads(gf2), avg_grads(gr), avg_grads(reverse(gr))

    # Gradient-level first: Adam's first step is m̂/√v̂ ≈ g/|g| per coordinate,
    # which amplifies a last-bit change in a near-zero coordinate to ~η.
    g = tree_absmax(ḡ_f)
    @printf("replica drift at step start  : %.3e  (must be exactly 0)\n", drift)
    @printf("max |ḡ_fwd − ḡ_fwd'|  / |ḡ|  : %.3e  (same order, re-run: kernel nondeterminism)\n", tree_maxdiff(ḡ_f, ḡ_f2) / g)
    @printf("max |ḡ_fwd − ḡ_rev|   / |ḡ|  : %.3e  (compute order permuted)\n", tree_maxdiff(ḡ_f, ḡ_r) / g)
    @printf("max |ḡ_rev − ḡ_rev,rr| / |ḡ| : %.3e  (reduction order permuted)\n", tree_maxdiff(ḡ_r, ḡ_rr) / g)
    @printf("per-replica: max_r |g_r,fwd − g_r,rev| / |ḡ| : %.3e\n",
            maximum(tree_maxdiff(gf[r], gr[r]) for r in 1:NREP) / g)

    θ_f, θ_f2, θ_r = step(ps_f, ḡ_f), step(ps_f2, ḡ_f2), step(ps_r, ḡ_r)
    @printf("max |θ_fwd − θ_fwd'|         : %.3e  (after one Adam step, η = 1e-3)\n", tree_maxdiff(θ_f, θ_f2))
    @printf("max |θ_fwd − θ_rev|          : %.3e\n", tree_maxdiff(θ_f, θ_r))

    # Same experiment with cuDNN pinned to deterministic algorithms.
    CUDA.math_mode!(CUDA.PEDANTIC_MATH)
    _, hf, _  = dp_grads(1:NREP)
    _, hf2, _ = dp_grads(1:NREP)
    _, hr, _  = dp_grads(reverse(1:NREP))
    CUDA.math_mode!(CUDA.DEFAULT_MATH)
    @printf("\nPEDANTIC_MATH  re-run       : %.3e\n", tree_maxdiff(avg_grads(hf), avg_grads(hf2)) / g)
    @printf("PEDANTIC_MATH  order swap   : %.3e\n", tree_maxdiff(avg_grads(hf), avg_grads(hr)) / g)
end
