# Train the classifier on ImageNet-64: Adam 1e-3, cross-entropy, val top-1/top-5 after every epoch.
#
#   julia -t 4 --project src/train.jl            # 10 epochs
#   julia -t 4 --project src/train.jl 10 20      # smoke test: 10 epochs capped at 20 steps each
#   BATCH=512 julia -t 4 --project src/train.jl  # batch size (default 1024; 512 fits a 12 GB GPU)
#   MODEL=resnet18 julia -t 4 --project src/train.jl   # network: vgg (default) or resnet18
#
# Data and loading are common to all imagenet64 projects (../utils/data.jl, ../utils/dataloader.jl,
# shards in ../build); the same code runs on the Threadripper and the NAS. -t 4 lets the loader's
# reader thread overlap with training.
#
# Writes build/checkpoint/model_<epoch>.jld2 and results/train_{epochs,loss}.csv; models other
# than the default get a suffix (build/checkpoint-resnet18/, results/train_epochs-resnet18.csv).
# TensorBoard logs go to build/tensorboard/<model>-b<batch>-<time>/ while training runs:
#
#   tensorboard --logdir build/tensorboard          # then http://localhost:6006

using Lux, LuxCUDA, CUDA, Optimisers, Zygote
using Random, Statistics, Printf, JLD2, DelimitedFiles, Dates
using TensorBoardLogger

const ROOT      = dirname(@__DIR__)
const MODEL     = get(ENV, "MODEL", "vgg")
const TAG       = MODEL == "vgg" ? "" : "-$MODEL"
const RESULTS   = joinpath(ROOT, "results")
const CHKPT_DIR = joinpath(ROOT, "build", "checkpoint$TAG")
const TB_DIR    = joinpath(ROOT, "build", "tensorboard")
const NCLASS    = 1000
const BATCH     = parse(Int, get(ENV, "BATCH", "1024"))

include(joinpath(dirname(ROOT), "utils", "data.jl"))        # load_train / load_val
include(joinpath(dirname(ROOT), "utils", "dataloader.jl"))  # DataLoader
include(joinpath(@__DIR__, "model.jl"))                      # build_model

# Cross-entropy from logits with a one-hot target Y (1000 × B):
#   ℓ = −(1/B) Σᵢ Σ_c Y_ci log softmax(z)_ci
# One-hot times logsoftmax instead of indexing: a dense broadcast, no gather/scatter on the GPU.
crossentropy(z, Y) = -sum(Y .* logsoftmax(z; dims=1)) / size(z, 2)

# One training-mode forward + backward pass. st0 in, st1 (updated BatchNorm statistics) out;
# `local` keeps st1 out of a Core.Box.
function ce_grad(model, ps, st0, xb, Yb)
    (ℓ, st1), back = Zygote.pullback(ps) do p
        local z, st1 = Lux.apply(model, xb, p, st0)
        crossentropy(z, Yb), st1
    end
    ∇, = back((one(ℓ), nothing))
    return ∇, ℓ, st1
end

# Fill a preallocated one-hot host buffer for labels y (1..NCLASS).
function onehot!(Y::Matrix{Float32}, y)
    fill!(Y, 0f0)
    @inbounds for (i, c) ∈ enumerate(y)
        Y[c, i] = 1f0
    end
    return Y
end

# Top-1 and top-5 accuracy on (y, X), inference mode (BatchNorm uses running statistics).
function evaluate(model, ps, st, y, X; batch = BATCH)
    st = Lux.testmode(st)
    N = size(X, 4)
    top1 = 0; top5 = 0
    for i ∈ 1:batch:N
        j = min(i + batch - 1, N)
        xb = CuArray(Array(view(X, :, :, :, i:j)))          # copy out of the mmap, then to the GPU
        z, _ = Lux.apply(model, xb, ps, st)
        z = Array(z)                                        # 1000 × b logits back on the host
        for (k, c) ∈ enumerate(y[i:j])
            col = view(z, :, k)
            top1 += argmax(col) == c
            top5 += c ∈ partialsortperm(col, 1:5; rev=true)
        end
    end
    return top1 / N, top5 / N
end

# The training loop. Returns (per-step losses, epoch table [epoch loss top1 top5 seconds img/s], ps, st).
# With a TBLogger `tb`, logs train/loss every step, train/img_per_s every 50 steps, and
# val/top1, val/top5, train/epoch_loss, time/epoch_s every epoch, against the global step.
function train(model, loader, yv, Xv; nepochs, lr = 1f-3, on_epoch = (epoch, ps, st) -> nothing,
               tb = nothing)
    dev = gpu_device()
    rng = Random.default_rng(); Random.seed!(rng, 0)
    ps, st = Lux.setup(rng, model)
    ps = ps |> dev; st = st |> dev
    opt_state = Optimisers.setup(Optimisers.Adam(lr), ps)

    xb_gpu  = CUDA.zeros(Float32, 64, 64, 3, loader.batch)
    Yb_host = zeros(Float32, NCLASS, loader.batch)
    Yb_gpu  = CUDA.zeros(Float32, NCLASS, loader.batch)

    batch_loss = Float32[]
    epochs = Matrix{Float64}(undef, 0, 6)
    step = 0                                     # global optimizer step, the TensorBoard x-axis
    t_rate = time(); n_rate = 0                  # throughput window for train/img_per_s
    for epoch ∈ 1:nepochs
        eloss = 0f0; nb = 0
        t0 = time()
        for (x, y) ∈ loader
            copyto!(xb_gpu, x)
            copyto!(Yb_gpu, onehot!(Yb_host, y))
            ∇, ℓ, st = ce_grad(model, ps, st, xb_gpu, Yb_gpu)
            opt_state, ps = Optimisers.update(opt_state, ps, ∇)
            push!(batch_loss, ℓ)
            eloss += ℓ; nb += 1; step += 1; n_rate += 1
            if tb !== nothing
                log_value(tb, "train/loss", ℓ; step)
                if n_rate == 50
                    log_value(tb, "train/img_per_s", n_rate * loader.batch / (time() - t_rate); step)
                    t_rate = time(); n_rate = 0
                end
            end
        end
        dt = time() - t0
        ips = nb * loader.batch / dt
        top1, top5 = evaluate(model, ps, st, yv, Xv)
        epochs = vcat(epochs, [epoch eloss/nb top1 top5 dt ips])
        if tb !== nothing
            log_value(tb, "val/top1", 100top1; step)
            log_value(tb, "val/top5", 100top5; step)
            log_value(tb, "train/epoch_loss", eloss/nb; step)
            log_value(tb, "time/epoch_s", dt; step)
            t_rate = time(); n_rate = 0              # don't count evaluation time as training
        end
        @printf("Epoch %2d/%d | loss %.4f | val top-1 %5.2f%% top-5 %5.2f%% | %6.1fs | %6d img/s | %d batches\n",
                epoch, nepochs, eloss/nb, 100top1, 100top5, dt, round(Int, ips), nb)
        on_epoch(epoch, ps, st)
    end
    return batch_loss, epochs, ps, st
end

if abspath(PROGRAM_FILE) == @__FILE__
    nepochs   = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 10
    max_steps = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : typemax(Int)

    @info "Loading"
    @time N, ys, Xs, _ = load_train()
    @time yv, Xv = load_val()
    loader = DataLoader(Xs, ys; batch = BATCH, limit = max_steps)
    @info "images" train = N shards = length(Xs) val = length(yv) batch = BATCH steps = length(loader) threads = Threads.nthreads()

    model = build_model(MODEL; nclass = NCLASS)
    @info "model" name = MODEL params = Lux.parameterlength(model)

    ispath(CHKPT_DIR) && rm(CHKPT_DIR, force = true, recursive = true)
    mkpath(CHKPT_DIR); mkpath(RESULTS)
    save_checkpoint(epoch, ps, st) =
        jldsave(joinpath(CHKPT_DIR, "model_$epoch.jld2");
                parameters = ps |> cpu_device(), state = st |> cpu_device())

    runname = "$MODEL-b$BATCH-" * Dates.format(now(), "yyyymmdd-HHMM")
    tb      = TBLogger(joinpath(TB_DIR, runname), tb_overwrite)
    @info "TensorBoard" logdir = joinpath(TB_DIR, runname)

    @info "Training"
    @time batch_loss, epochs, _, _ = train(model, loader, yv, Xv; nepochs, on_epoch = save_checkpoint, tb)

    writedlm(joinpath(RESULTS, "train_loss$TAG.csv"), batch_loss, ',')
    writedlm(joinpath(RESULTS, "train_epochs$TAG.csv"),
             vcat(["epoch" "loss" "val_top1" "val_top5" "seconds" "img_per_s"], epochs), ',')
    @printf("best val top-1 %.2f%%  top-5 %.2f%%\n", 100maximum(epochs[:, 3]), 100maximum(epochs[:, 4]))
end
