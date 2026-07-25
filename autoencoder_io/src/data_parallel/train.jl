# The data-parallel run (5 simulated replicas, instrumented per phase), its
# reassembly into a modelled multi-GPU step, the baseline run (single model in
# the same process), and the comparison.
#
#   julia --project src/data_parallel/train.jl            # full: data-parallel 10 epochs, baseline 2, ~8 min
#   julia --project src/data_parallel/train.jl 20         # smoke test: cap at 20 DP steps/epoch
#
# The data-parallel model goes to build/checkpoint_dp/model_10.jld2 and its losses to
# results/dp_train_{loss,epochs}.csv, for src/data_parallel/reconstruct.jl.

include(joinpath(@__DIR__, "common.jl"))
using JLD2, DelimitedFiles

N, ys, Xs, μs = load_train()
model = build_model()

const STEPS_FULL_DP   = sum(size(X, 4) ÷ GLOBAL_BATCH for X in Xs)
const STEPS_FULL_BASE = sum(size(X, 4) ÷ MICRO        for X in Xs)
const MAX_STEPS = isempty(ARGS) ? typemax(Int) : parse(Int, ARGS[1])
const NEPOCHS_A = 10       # same epoch count as normal/train.jl, so the model is comparable
const NEPOCHS_B = 2        # baseline only needs a steady-state epoch for timing
# Timing excludes epoch 1 (JIT + first touch of the mmap) in both runs.

# ── all-reduce model ──────────────────────────────────────────────────────────
const PARAM_BYTES     = Lux.parameterlength(model) * sizeof(Float32)
const LINK_GBPS       = 64.0       # PCIe 5.0 x16 raw unidirectional
const LINK_LATENCY_US = 5.0        # per hop: launch + collective handshake

# Ring all-reduce: 2(N-1) hops, 2(N-1)/N · P·s bytes moved per GPU.
allreduce_bytes(n)   = 2 * PARAM_BYTES * (n - 1) / n
allreduce_seconds(n) = n <= 1 ? 0.0 :
    2 * (n - 1) * LINK_LATENCY_US * 1e-6 + allreduce_bytes(n) / (LINK_GBPS * 1e9)

@printf("parameters      : %d  (%.2f MB Float32)\n", Lux.parameterlength(model), PARAM_BYTES / 1e6)
@printf("full epoch      : %d DP steps vs %d baseline steps, %d images each\n",
        STEPS_FULL_DP, STEPS_FULL_BASE, STEPS_FULL_DP * GLOBAL_BATCH)
@printf("ring all-reduce : %.3f ms = %.0f µs latency (%d hops) + %.0f µs transfer (%.2f MB at %.0f GB/s)\n",
        allreduce_seconds(NREP) * 1e3, 2 * (NREP - 1) * LINK_LATENCY_US, 2 * (NREP - 1),
        allreduce_bytes(NREP) / (LINK_GBPS * 1e9) * 1e6, allreduce_bytes(NREP) / 1e6, LINK_GBPS)

function init()
    rng = Random.default_rng(); Random.seed!(rng, 0)
    ps, st = Lux.setup(rng, model)
    return ps |> gpu_device(), st |> gpu_device()
end

# ── data-parallel run ──────────────────────────────────────────────────────────
function run_dp()
    ps, st = init()
    opt_state = Optimisers.setup(Optimisers.Adam(1f-3), ps)
    replicas  = [clone_params(ps) for _ in 1:NREP]
    # SPEED: one host and one device buffer, reused by every replica and step (see normal/train.jl).
    # Slicing Xk[:,:,:,lo:hi] would allocate 50 MB per replica per step, and tcopy would then
    # time page faults, not the copy a real GPU would do.
    xb_host   = Array{Float32}(undef, 64, 64, 3, MICRO)
    xb_gpu    = CUDA.zeros(Float32, 64, 64, 3, MICRO)     # shared staging buffer; see `timed`
    acc = (copy = Float64[], grad = Float64[], gradmax = Float64[],
           reduce = Float64[], update = Float64[], bcast = Float64[], step = Float64[])
    out = Dict{Symbol, Any}()
    step_loss = Float32[]
    epochs = Matrix{Float64}(undef, 0, 4)              # epoch, loss, seconds, img/s
    warm_seconds = Float64[]

    for epoch in 1:NEPOCHS_A
        eloss = 0f0; nsteps = 0
        t_epoch = time()
        warm = epoch > 1
        drift_checked = false
        for k in randperm(length(Xs))
            nsteps >= MAX_STEPS && break
            Xk = Xs[k]; Nk = size(Xk, 4)
            for i in shuffle(collect(1:GLOBAL_BATCH:Nk))
                i + GLOBAL_BATCH - 1 > Nk && continue
                nsteps >= MAX_STEPS && break
                t_step = time_ns()

                # The invariant everything rests on: every replica holds master θ.
                if !drift_checked
                    drift = maximum(tree_maxdiff(replicas[r], ps) for r in 1:NREP)
                    @assert drift == 0 "replicas drifted from master by $drift"
                    drift_checked = true
                end

                grads  = Vector{Any}(undef, NREP)
                losses = zeros(Float32, NREP)
                tcopy  = zeros(NREP); tgrad = zeros(NREP)
                st_next = st

                # scatter + local gradients: sequential here, concurrent on N GPUs
                for r in 1:NREP
                    lo = i + (r - 1) * MICRO
                    _, tcopy[r] = timed() do                              # SPEED: view → reused buffer
                        copyto!(xb_host, view(Xk, :, :, :, lo:lo+MICRO-1))
                        copyto!(xb_gpu, xb_host)
                    end
                    (g, ℓ, st_new), tgrad[r] = timed(() -> mse_grad(model, replicas[r], st, xb_gpu))
                    grads[r] = g; losses[r] = ℓ
                    r == 1 && (st_next = st_new)        # BN buffers follow replica 1 (DDP broadcast_buffers)
                end

                ḡ, tred = timed(() -> avg_grads(grads))                                   # all-reduce
                (opt_state, ps), tupd = timed(() -> Optimisers.update(opt_state, ps, ḡ))  # master update
                _, tbc = timed() do                                                       # broadcast
                    for r in 1:NREP; replicas[r] = clone_params(ps); end
                end

                st = st_next
                push!(step_loss, mean(losses))
                eloss += mean(losses); nsteps += 1
                if warm
                    push!(acc.copy, sum(tcopy));  push!(acc.grad, sum(tgrad))
                    push!(acc.gradmax, maximum(tcopy .+ tgrad))
                    push!(acc.reduce, tred); push!(acc.update, tupd); push!(acc.bcast, tbc)
                    push!(acc.step, (time_ns() - t_step) / 1e9)
                end
            end
        end
        dt = time() - t_epoch
        epochs = vcat(epochs, [epoch eloss/nsteps dt nsteps*GLOBAL_BATCH/dt])
        @printf("  DP epoch %2d/%d | loss %.5f | %6.1fs | %4d steps | %6d img/s%s\n",
                epoch, NEPOCHS_A, eloss / nsteps, dt, nsteps,
                round(Int, nsteps * GLOBAL_BATCH / dt), warm ? "" : "  (warm-up, excluded)")
        warm && push!(warm_seconds, dt)
        out[:steps] = nsteps; out[:loss] = eloss / nsteps
    end
    out[:epoch_seconds] = mean(warm_seconds)
    out[:t] = acc

    # The trained master model, for src/data_parallel/reconstruct.jl.
    ispath(CHKPT_DIR_DP) && rm(CHKPT_DIR_DP, force = true, recursive = true)
    mkpath(CHKPT_DIR_DP)
    cpu = cpu_device()
    jldsave(joinpath(CHKPT_DIR_DP, "model_$NEPOCHS_A.jld2"); parameters = ps |> cpu, state = st |> cpu)
    writedlm(joinpath(RESULTS, "dp_train_loss.csv"), step_loss, ',')
    writedlm(joinpath(RESULTS, "dp_train_epochs.csv"),
             vcat(["epoch" "loss" "seconds" "img_per_s"], epochs), ',')
    return out
end

# ── baseline run ────────────────────────────────────────────────────────────────
# The normal training loop itself (`train_normal`, src/shared.jl), for NEPOCHS_B epochs in this
# process, so the comparison shares warm-up, page cache and GPU clocks. Its last epoch is the
# steady-state baseline.
function run_baseline()
    max_b = MAX_STEPS > typemax(Int) ÷ NREP ? typemax(Int) : MAX_STEPS * NREP   # same images as a capped DP run
    _, epochs, _, _ = train_normal(model, Xs; nepochs = NEPOCHS_B, max_steps = max_b)
    last = epochs[end, :]                               # [epoch, loss, seconds, img/s]
    steps = round(Int, last[4] * last[3] / BATCH)
    return Dict{Symbol, Any}(:epoch_seconds => last[3], :steps => steps, :loss => last[2])
end

println("\n── data-parallel run: simulated (N=$NREP, global batch $GLOBAL_BATCH) ──")
a = run_dp()
println("\n── baseline run: single model (batch $MICRO) ──")
b = run_baseline()

# ── reassembly ────────────────────────────────────────────────────────────────
t = a[:t]
@assert !isempty(t.step) "no steady-state steps recorded"
ms(x) = 1e3 * mean(x)
seq_step = mean(t.step)
par_step = mean(t.gradmax) + allreduce_seconds(NREP) + mean(t.update)

println("\n── data-parallel run, per optimizer step (steady state, n=$(length(t.step))) ──")
@printf("  H→D copy   x%d : %7.2f ms   (%.2f ms each)\n", NREP, ms(t.copy), ms(t.copy) / NREP)
@printf("  fwd+bwd    x%d : %7.2f ms   (%.2f ms each)\n", NREP, ms(t.grad), ms(t.grad) / NREP)
@printf("  reduce (dev)    : %7.2f ms\n", ms(t.reduce))
@printf("  master update   : %7.2f ms\n", ms(t.update))
@printf("  broadcast  x%d : %7.2f ms   (free in real DDP)\n", NREP, ms(t.bcast))
@printf("  total step      : %7.2f ms\n", 1e3 * seq_step)
println("── reassembled as $NREP GPUs ──")
@printf("  slowest replica : %7.2f ms   (max_r copy+grad)\n", ms(t.gradmax))
@printf("  ring all-reduce : %7.2f ms   (modelled)\n", 1e3 * allreduce_seconds(NREP))
@printf("  master update   : %7.2f ms\n", ms(t.update))
@printf("  total step      : %7.2f ms\n", 1e3 * par_step)

# ── comparison ────────────────────────────────────────────────────────────────
b_step = b[:epoch_seconds] / b[:steps]
d_step = a[:epoch_seconds] / a[:steps]
b_epoch = b_step * STEPS_FULL_BASE
d_epoch = d_step * STEPS_FULL_DP
d_theor = par_step * STEPS_FULL_DP

println("\n── comparison ──")
@printf("%-32s %12s %12s\n", "", "baseline", "DP (N=5)")
@printf("%-32s %12d %12d\n", "images / optimizer step", MICRO, GLOBAL_BATCH)
@printf("%-32s %12d %12d\n", "optimizer steps / epoch", STEPS_FULL_BASE, STEPS_FULL_DP)
@printf("%-32s %12d %12d\n", "steps measured", b[:steps], a[:steps])
@printf("%-32s %12.2f %12.2f\n", "ms / step, 1 GPU", 1e3 * b_step, 1e3 * d_step)
@printf("%-32s %12.1f %12.1f\n", "s / epoch, 1 GPU", b_epoch, d_epoch)
@printf("%-32s %12s %12.1f\n", "s / epoch, modelled 5 GPUs", "—", d_theor)
@printf("%-32s %12.5f %12s\n", "loss, epoch $NEPOCHS_B", b[:loss], "")
@printf("%-32s %12s %12.5f\n", "loss, epoch $NEPOCHS_A", "", a[:loss])
@printf("\nsequential DP vs baseline : %+.1f%%\n", 100 * (d_epoch / b_epoch - 1))
@printf("modelled 5-GPU vs baseline: %.2fx faster per epoch (ceiling)\n", b_epoch / d_theor)
@printf("modelled 5-GPU vs sequential DP: %.2fx\n", seq_step / par_step)
MAX_STEPS < typemax(Int) && println("NOTE: capped run -- per-epoch rows are per-step cost scaled to a full epoch.")
