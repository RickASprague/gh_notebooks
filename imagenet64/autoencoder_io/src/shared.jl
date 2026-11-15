# Shared by every autoencoder script: the model, its loss and training loop. Data loading is common to
# all imagenet64 projects (../utils/data.jl, shards in ../build).
#
# All paths are relative to the project root (autoencoder_io/), so scripts can
# be run from anywhere:  julia --project src/<script>.jl

using Lux, LuxCUDA, CUDA, Optimisers, Zygote
using Random, Statistics, Printf

const ROOT      = dirname(@__DIR__)
const BUILD     = joinpath(ROOT, "build")
const RESULTS   = joinpath(ROOT, "results")
const RESOURCES = joinpath(ROOT, "resources")
const CHKPT_DIR    = joinpath(BUILD, "checkpoint")      # normal/train.jl
const CHKPT_DIR_DP = joinpath(BUILD, "checkpoint_dp")   # data_parallel/train.jl, data-parallel run

include(joinpath(dirname(ROOT), "utils", "data.jl"))   # common: shards + load_train / load_val

# Symmetric conv / deconv, stride-2 at every stage.
#   encoder: 64x64x3 → 32x32x32 → 16x16x64 → 8x8x128 → 4x4x256
#   decoder: 4x4x256 → 8x8x128 → 16x16x64 → 32x32x32 → 64x64x3
function build_model()
    encoder = Chain(
        Conv((3,3),   3 =>  32, stride=2, pad=1), BatchNorm(32),  relu,
        Conv((3,3),  32 =>  64, stride=2, pad=1), BatchNorm(64),  relu,
        Conv((3,3),  64 => 128, stride=2, pad=1), BatchNorm(128), relu,
        Conv((3,3), 128 => 256, stride=2, pad=1), BatchNorm(256), relu,
    )
    decoder = Chain(
        ConvTranspose((3,3), 256 => 128, stride=2, pad=1, outpad=1), BatchNorm(128), relu,
        ConvTranspose((3,3), 128 =>  64, stride=2, pad=1, outpad=1), BatchNorm(64),  relu,
        ConvTranspose((3,3),  64 =>  32, stride=2, pad=1, outpad=1), BatchNorm(32),  relu,
        ConvTranspose((3,3),  32 =>   3, stride=2, pad=1, outpad=1),
        sigmoid,                                            # pixels in [0,1]
    )
    return Chain(encoder, decoder)
end

# One training-mode forward + backward pass, MSE reconstruction loss.
#
#   pullback runs the forward pass now and returns its value plus `back`, which runs the
#   backward pass (chain rule) when called. The seed (1, nothing) asks for ∂ℓ/∂ps and
#   no gradient through the BatchNorm state.
#
# Returns (∇ = ∂ℓ/∂ps, ℓ, st1), where ℓ is the batch loss. st0 is the state going in, st1 the
# state coming out of the forward pass (BatchNorm running mean/var updated from this batch).
# Lux is functional: st0 is not mutated; the caller keeps st1 for the next step.
# `local` inside the closure is required: without it, assigning st1 there would capture the
# outer st1 in a Core.Box (slow, and awkward for Zygote) instead of making a new variable.
function mse_grad(model, ps, st0, xb)
    (ℓ, st1), back = Zygote.pullback(ps) do p
        local x̂, st1 = Lux.apply(model, xb, p, st0)    # forward: reconstruction + new state
        mean((x̂ .- xb) .^ 2), st1                      # (ℓ, st1) is what `back` differentiates
    end
    ∇, = back((one(ℓ), nothing))                       # backward: one gradient per argument of
                                                       # the closure; only p, so a 1-tuple (∂ℓ/∂ps,)
    return ∇, ℓ, st1
end

# Larger batches do not raise throughput here: cuDNN's NCHW BatchNorm launches one
# block per channel, so its grid does not grow with the batch, and every kernel's
# cost per image stays the same. They do cost optimizer steps (and, at 16384, 10.5%
# of each shard as dropped tail). 1024 is the batch to keep.
const BATCH = 1024

# The normal training loop: one model, batch 1024, Adam 1e-3, MSE. Used by
# normal/train.jl (10 epochs, checkpoints via `on_epoch`) and by data_parallel/train.jl as its
# baseline (2 epochs, timing only), so both run exactly the same code.
#
# Returns (per-step losses, epoch table [epoch loss seconds img/s], ps, st).
function train_normal(model, Xs; nepochs, max_steps = typemax(Int),
                      on_epoch = (epoch, ps, st) -> nothing)
    dev = gpu_device()
    rng = Random.default_rng(); Random.seed!(rng, 0)
    ps, st = Lux.setup(rng, model)
    ps = ps |> dev; st = st |> dev
    opt_state = Optimisers.setup(Optimisers.Adam(1f-3), ps)

    # SPEED: one host and one device buffer for the whole run; nothing is
    # allocated per step. Do NOT take the batch as Xk[:,:,:,i:j]: that copies
    # into a fresh 50 MB array every step, ~20 ms of page faults + GC while the
    # GPU sits idle, and it cuts throughput from 52k to 30k img/s.
    # (Auto Encoder.md, "Batch extraction")
    xb_host = Array{Float32}(undef, 64, 64, 3, BATCH)
    xb_gpu  = CUDA.zeros(Float32, 64, 64, 3, BATCH)

    batch_loss = Float32[]
    epochs = Matrix{Float64}(undef, 0, 4)              # epoch, loss, seconds, img/s
    for epoch ∈ 1:nepochs
        eloss = 0f0; nb = 0
        t0 = time()
        for k ∈ randperm(length(Xs))                   # shard order shuffled
            nb >= max_steps && break
            Xk = Xs[k]; Nk = size(Xk, 4)               # mmap'd (64,64,3,Nk)
            for i ∈ shuffle(collect(1:BATCH:Nk))       # batch order shuffled
                j = i + BATCH - 1
                j > Nk && continue                     # drop ragged tail
                nb >= max_steps && break
                copyto!(xb_host, view(Xk, :, :, :, i:j))   # SPEED: view, not Xk[:,:,:,i:j]
                copyto!(xb_gpu, xb_host)   # Array → CuArray = one cudaMemcpy. A view can't go
                                           # straight to xb_gpu: CUDA.jl doesn't see through the
                                           # mmap wrappers and errors on scalar indexing.
                ∇, ℓ, st = mse_grad(model, ps, st, xb_gpu)   # st ← new BatchNorm state
                opt_state, ps = Optimisers.update(opt_state, ps, ∇)
                push!(batch_loss, ℓ)
                eloss += ℓ; nb += 1
            end
        end
        dt = time() - t0
        ips = nb * BATCH / dt
        epochs = vcat(epochs, [epoch eloss/nb dt ips])
        @printf("Epoch %2d/%d | avg loss %.5f | %6.1fs | %6d img/s | %d batches\n",
                epoch, nepochs, eloss/nb, dt, round(Int, ips), nb)
        on_epoch(epoch, ps, st)
    end
    return batch_loss, epochs, ps, st
end
