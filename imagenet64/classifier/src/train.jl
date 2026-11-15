# Train the classifier on ImageNet-64: Adam 1e-3, cross-entropy, val top-1/top-5 after every epoch.
#
#   julia -t 4 --project src/train.jl            # 10 epochs
#   julia -t 4 --project src/train.jl 10 20      # smoke test: 10 epochs capped at 20 steps each
#   BATCH=512 julia -t 4 --project src/train.jl  # batch size (default 1024; 512 fits a 12 GB GPU)
#   MODEL=resnet18 julia -t 4 --project src/train.jl   # network: vgg (default) or resnet18
#   PRECISION=bf16 julia -t 4 --project src/train.jl   # BF16 mixed precision (default fp32)
#
# Data and loading are common to all imagenet64 projects (../utils/data.jl, ../utils/dataloader.jl,
# shards in ../build); the same code runs on the Threadripper and the NAS. -t 4 lets the loader's
# reader thread overlap with training.
#
# Writes build/checkpoint/model_<epoch>.jld2 and results/train_{epochs,loss}.csv; models other
# than the default get a suffix (build/checkpoint-resnet18/, results/train_epochs-resnet18.csv),
# and so does a non-default precision (-bf16).
# TensorBoard logs go to build/tensorboard/<model>[-bf16]-b<batch>-<time>/ while training runs:
#
#   tensorboard --logdir build/tensorboard          # then http://localhost:6006

using Lux, LuxCUDA, CUDA, Optimisers, Zygote
using Random, Statistics, Printf, JLD2, DelimitedFiles, Dates
using TensorBoardLogger

const ROOT      = dirname(@__DIR__)
const MODEL     = get(ENV, "MODEL", "vgg")
const PRECISION = get(ENV, "PRECISION", "fp32")             # fp32 or bf16
const PTAG      = PRECISION == "fp32" ? "" : "-$PRECISION"
# A run limited to a few steps (second argument) is a smoke test: its outputs get a -smoke tag
# so they never overwrite a full run's results, checkpoints or TensorBoard log.
const STAG      = length(ARGS) ≥ 2 ? "-smoke" : ""
const TAG       = (MODEL == "vgg" ? "" : "-$MODEL") * PTAG * STAG
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

# BF16 mixed precision: the step's forward and backward run on BF16 copies, the weights, Adam's
# moments and the BatchNorm statistics stay FP32.
#
#   ps32 → bf16 → forward/backward (tensor cores, half the activation bytes) → ∇ in BF16 → FP32
#   → Adam on ps32.
#
# The weights are the FP32 part because they accumulate many tiny updates: near |w| = 0.5 one BF16
# ulp is 2⁻⁸ ≈ 4e-3, so a 1e-3 Adam step would round away. All conversions run on the GPU: Julia's
# CPU BF16 conversion crashes LLVM on this machine (Zen 4).
#
# Only the convolutional body runs in BF16. The head (global pool, dropout, Dense, and the softmax
# after it) runs in FP32 on FP32 weights: it is ~0.5 M multiply-adds per image against ~2.2 G for
# the body, the logits and softmax want FP32 anyway, and LuxLib's fused BF16 Dense (cuBLASLt matmul
# with a bias epilogue) fails its algorithm search with CUBLAS_STATUS_INVALID_VALUE.
to_bf16(x) = Lux.bf16(x)
to_f32(t) = Lux.fmap(x -> x isa AbstractArray ? Float32.(x) : x, t)

# Split a model's top-level Chain into body (all layers but the last) and head (the last layer).
# Both networks in model.jl end in `head(...)`. Cached per model, since Chain construction is cheap
# but needn't run every step.
const _SPLIT = IdDict{Any, Any}()
function split_head(model)
    get!(_SPLIT, model) do
        ks = keys(model.layers)
        bk = ks[1:end-1]
        (; body = Chain(NamedTuple{bk}(map(k -> model.layers[k], bk))), bk, hk = ks[end])
    end
end
pick(nt, ks) = NamedTuple{ks}(map(k -> nt[k], ks))

function ce_grad_bf16(model, ps32, st32, xb, Yb)
    (; body, bk, hk) = split_head(model)
    head = model.layers[hk]
    psb, stb = to_bf16(pick(ps32, bk)), to_bf16(pick(st32, bk))   # body: BF16 copies
    psh, sth = ps32[hk], st32[hk]                                  # head: FP32 as is
    x16 = to_bf16(xb)
    (ℓ, (stb1, sth1)), back = Zygote.pullback(psb, psh) do pb, ph
        local f, stb1 = Lux.apply(body, x16, pb, stb)              # BF16 features
        local z, sth1 = Lux.apply(head, Float32.(f), ph, sth)      # FP32 head → FP32 logits
        crossentropy(z, Yb), (stb1, sth1)
    end
    ∇b, ∇h = back((one(ℓ), nothing))
    ks = (bk..., hk)                                               # same key order as ps32
    ∇   = NamedTuple{ks}((map(k -> to_f32(∇b[k]), bk)..., ∇h))
    st1 = NamedTuple{ks}((map(k -> to_f32(stb1[k]), bk)..., sth1))
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
# With a TBLogger `tb`, logs against the global step:
#   every step       train/loss
#   every 50 steps   train/img_per_s
#   every 500 steps  val_subset/top1, val_subset/top5 on a fixed 5 000-image slice of val (cheap,
#                    dense; its time is excluded from epoch time and throughput)
#   every epoch      val/top1, val/top5 (all 50 000), train/epoch_loss, time/epoch_s
const SUBSET_EVERY = 500
const SUBSET_N     = 5_000

function train(model, loader, yv, Xv; nepochs, lr = 1f-3, on_epoch = (epoch, ps, st) -> nothing,
               tb = nothing, grad = ce_grad)
    ys_sub, Xs_sub = yv[1:SUBSET_N], view(Xv, :, :, :, 1:SUBSET_N)
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
        t0 = time(); t_sub = 0.0                 # t_sub: subset-evaluation time, not training
        for (x, y) ∈ loader
            copyto!(xb_gpu, x)
            copyto!(Yb_gpu, onehot!(Yb_host, y))
            ∇, ℓ, st = grad(model, ps, st, xb_gpu, Yb_gpu)   # ce_grad, or ce_grad_bf16
            opt_state, ps = Optimisers.update(opt_state, ps, ∇)
            push!(batch_loss, ℓ)
            eloss += ℓ; nb += 1; step += 1; n_rate += 1
            if tb !== nothing
                log_value(tb, "train/loss", ℓ; step)
                if n_rate == 50
                    log_value(tb, "train/img_per_s", n_rate * loader.batch / (time() - t_rate); step)
                    t_rate = time(); n_rate = 0
                end
                if step % SUBSET_EVERY == 0
                    te = time()
                    s1, s5 = evaluate(model, ps, st, ys_sub, Xs_sub)
                    log_value(tb, "val_subset/top1", 100s1; step)
                    log_value(tb, "val_subset/top5", 100s5; step)
                    de = time() - te
                    t_sub += de; t_rate += de            # keep it out of epoch time and img/s
                end
            end
        end
        dt = time() - t0 - t_sub
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

    grad = PRECISION == "bf16" ? ce_grad_bf16 :
           PRECISION == "fp32" ? ce_grad : error("PRECISION must be fp32 or bf16, got $PRECISION")
    @info "precision" PRECISION
    runname = "$MODEL$PTAG$STAG-b$BATCH-" * Dates.format(now(), "yyyymmdd-HHMM")
    tb      = TBLogger(joinpath(TB_DIR, runname), tb_overwrite)
    @info "TensorBoard" logdir = joinpath(TB_DIR, runname)

    @info "Training"
    @time batch_loss, epochs, _, _ = train(model, loader, yv, Xv; nepochs, on_epoch = save_checkpoint, tb, grad)

    writedlm(joinpath(RESULTS, "train_loss$TAG.csv"), batch_loss, ',')
    writedlm(joinpath(RESULTS, "train_epochs$TAG.csv"),
             vcat(["epoch" "loss" "val_top1" "val_top5" "seconds" "img_per_s"], epochs), ',')
    @printf("best val top-1 %.2f%%  top-5 %.2f%%\n", 100maximum(epochs[:, 3]), 100maximum(epochs[:, 4]))
end
