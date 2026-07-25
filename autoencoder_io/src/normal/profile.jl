# Nsight Systems / Compute capture of the normal training loop, with NVTX labels.
#
#   nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
#        --trace=cuda,nvtx,osrt -o results/train_profile \
#        julia --project src/normal/profile.jl
#
# Warms up (JIT, first touch of the pages) outside the capture, then records
# PROFILE_STEPS steps inside CUDA.@profile external=true. The loop body is
# that of `train_normal` (src/shared.jl), unchanged, so the profile describes the code that produced the
# numbers in Auto Encoder.md. No CairoMakie here: loading it alongside Zygote
# stalls the first gradient's compile for >9 min on Julia 1.12.

include(joinpath(@__DIR__, "..", "shared.jl"))
using NVTX

const WARMUP_STEPS  = 60
const PROFILE_STEPS = parse(Int, get(ENV, "PROFILE_STEPS", "200"))   # ncu: 1 step is enough

N, ys, Xs, μs = load_train()
model = build_model()

dev = gpu_device()
rng = Random.default_rng(); Random.seed!(rng, 0)
ps, st = Lux.setup(rng, model)
ps = ps |> dev; st = st |> dev
opt_state = Optimisers.setup(Optimisers.Adam(1f-3), ps)
xb_host = Array{Float32}(undef, 64, 64, 3, BATCH)
xb_gpu  = CUDA.zeros(Float32, 64, 64, 3, BATCH)

# Same batch order policy as train_normal: shuffled shards, shuffled batch starts.
batches = [(k, i) for k in randperm(length(Xs)) for i in shuffle(collect(1:BATCH:size(Xs[k], 4)))
           if i + BATCH - 1 <= size(Xs[k], 4)]

function step!(k, i)
    global ps, st, opt_state
    NVTX.@range "step" begin
        NVTX.@range "slice" copyto!(xb_host, view(Xs[k], :, :, :, i:i+BATCH-1))  # SPEED: view into reused buffer (see train_normal)
        NVTX.@range "H2D" copyto!(xb_gpu, xb_host)                                # pageable → device
        ∇, ℓ, st = NVTX.@range "fwd+bwd" mse_grad(model, ps, st, xb_gpu)
        opt_state, ps = NVTX.@range "adam" Optimisers.update(opt_state, ps, ∇)
        return ℓ
    end
end

@info "warm-up" steps = WARMUP_STEPS
for b in 1:WARMUP_STEPS
    step!(batches[b]...)
end
CUDA.synchronize()

# Touch every batch the capture will use, so the profile sees the steady state of training
# epochs 2+ (data resident in RAM) and not NVMe reads. The ZFS ARC can be smaller than the
# 63 GB file, e.g. right after a reboot.
@info "prefetch" batches = PROFILE_STEPS
for b in WARMUP_STEPS+1:WARMUP_STEPS+PROFILE_STEPS
    k, i = batches[b]
    copyto!(xb_host, view(Xs[k], :, :, :, i:i+BATCH-1))
end

@info "profiling" steps = PROFILE_STEPS
t0 = time()
CUDA.@profile external = true begin
    for b in WARMUP_STEPS+1:WARMUP_STEPS+PROFILE_STEPS
        step!(batches[b]...)
    end
    CUDA.synchronize()
end
dt = time() - t0
@printf("profiled %d steps in %.2f s = %.1f ms/step (%.0f img/s)\n",
        PROFILE_STEPS, dt, 1e3 * dt / PROFILE_STEPS, PROFILE_STEPS * BATCH / dt)
