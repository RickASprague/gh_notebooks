# Train the classifier with Reactant + Enzyme: the whole training step is traced, compiled through
# MLIR to XLA (fused kernels), and differentiated by Enzyme. Same model, data, loss, optimizer and
# TensorBoard tags as src/train.jl, which runs the same recipe eagerly on CUDA.jl with Zygote.
#
#   MODEL=resnet18 BATCH=512 julia -t 4 --project src/train_reactant.jl          # 10 epochs
#   MODEL=resnet18 BATCH=512 julia -t 4 --project src/train_reactant.jl 2 30     # smoke test
#   PRECISION=bf16 MODEL=resnet18 BATCH=512 julia -t 4 --project src/train_reactant.jl   # BF16
#
# Writes build/checkpoint-<model>-reactant[-bf16]/model_<epoch>.jld2,
# results/train_{epochs,loss}-<model>-reactant[-bf16].csv and
# build/tensorboard/<model>-reactant[-bf16]-b<batch>-<time>/.
#
# BF16 mixed precision here casts inside the compiled objective: the TrainState keeps FP32
# weights and Adam moments, the forward/backward run on BF16 copies, and Enzyme differentiates
# through the casts, so the gradient arrives in FP32 for the update. Logits go back to FP32 before
# the softmax, and the BatchNorm state is returned in FP32 so its type is the same every step.
#
# Compiled XLA programs have fixed shapes: the DataLoader already drops ragged batches, and
# evaluation uses batches of EVAL_BATCH = 1000, which divides the 50 000 val images and the
# 5 000-image subset exactly.

using Lux, Reactant, Enzyme, Optimisers
using Random, Statistics, Printf, JLD2, DelimitedFiles, Dates
using TensorBoardLogger

const ROOT       = dirname(@__DIR__)
const MODEL      = get(ENV, "MODEL", "resnet18")
const PRECISION  = get(ENV, "PRECISION", "fp32")            # fp32 or bf16
const PTAG       = PRECISION == "fp32" ? "" : "-$PRECISION"
# A run limited to a few steps (second argument) is a smoke test: its outputs get a -smoke tag
# so they never overwrite a full run's results, checkpoints or TensorBoard log.
const STAG      = length(ARGS) ≥ 2 ? "-smoke" : ""
const TAG        = "-$MODEL-reactant$PTAG" * STAG
const RESULTS    = joinpath(ROOT, "results")
const CHKPT_DIR  = joinpath(ROOT, "build", "checkpoint$TAG")
const TB_DIR     = joinpath(ROOT, "build", "tensorboard")
const NCLASS     = 1000
const BATCH      = parse(Int, get(ENV, "BATCH", "1024"))
const EVAL_BATCH = 1000
const SUBSET_EVERY = 500
const SUBSET_N     = 5_000

include(joinpath(dirname(ROOT), "utils", "data.jl"))        # load_train / load_val
include(joinpath(dirname(ROOT), "utils", "dataloader.jl"))  # DataLoader
include(joinpath(@__DIR__, "model.jl"))                      # build_model

# Same loss as src/train.jl: cross-entropy from logits with one-hot targets.
crossentropy(z, Y) = -sum(Y .* logsoftmax(z; dims=1)) / size(z, 2)

# The objective in the shape Lux.Training expects: (model, ps, st, data) → (loss, st1, stats).
function objective(model, ps, st, (x, Y))
    z, st1 = model(x, ps, st)
    return crossentropy(z, Y), st1, (;)
end

# BF16 mixed precision, traced and compiled like the FP32 objective (see the header).
# The casts test for TracedRArray{Float32}: inside the trace an array's eltype is
# TracedRNumber{Float32}, not an AbstractFloat, so Lux.bf16 would leave it FP32, and Conv would
# then promote a BF16 input back to the FP32 weight type (no BF16 kernel anywhere).
const BF16 = Core.BFloat16
to_bf16(t) = Lux.fmap(a -> a isa Reactant.TracedRArray{Float32} ? BF16.(a) : a, t)
to_f32(t)  = Lux.fmap(a -> a isa Reactant.TracedRArray{BF16} ? Float32.(a) : a, t)

function objective_bf16(model, ps, st, (x, Y))
    z, st1 = model(BF16.(x), to_bf16(ps), to_bf16(st))
    return crossentropy(Float32.(z), Y), to_f32(st1), (;)
end

function onehot!(Y::Matrix{Float32}, y)
    fill!(Y, 0f0)
    @inbounds for (i, c) ∈ enumerate(y)
        Y[c, i] = 1f0
    end
    return Y
end

# Top-1 / top-5 with a compiled forward pass (test mode), batches of EVAL_BATCH.
function evaluate(fwd, ps, st_test, y, X, dev)
    N = size(X, 4)
    top1 = 0; top5 = 0
    for i ∈ 1:EVAL_BATCH:N
        j = i + EVAL_BATCH - 1
        xb = dev(Array(view(X, :, :, :, i:j)))
        z  = Array(first(fwd(xb, ps, st_test)))          # 1000 × EVAL_BATCH logits on the host
        for (k, c) ∈ enumerate(y[i:j])
            col = view(z, :, k)
            top1 += argmax(col) == c
            top5 += c ∈ partialsortperm(col, 1:5; rev=true)
        end
    end
    return top1 / N, top5 / N
end

function train_reactant(model, loader, yv, Xv; nepochs, lr = 1f-3, tb = nothing,
                        on_epoch = (epoch, ps, st) -> nothing, obj = objective)
    dev = reactant_device()
    rng = Random.default_rng(); Random.seed!(rng, 0)
    ps, st = Lux.setup(rng, model) |> dev
    ts = Training.TrainState(model, ps, st, Optimisers.Adam(lr))

    Yb_host = zeros(Float32, NCLASS, loader.batch)
    ys_sub, Xs_sub = yv[1:SUBSET_N], view(Xv, :, :, :, 1:SUBSET_N)

    # Compiled inference forward pass, reused for every evaluation (fixed EVAL_BATCH shape).
    x_eval = dev(zeros(Float32, 64, 64, 3, EVAL_BATCH))
    t = time()
    fwd = @compile model(x_eval, ts.parameters, Lux.testmode(ts.states))
    @info "compiled evaluation forward pass" seconds = round(time() - t; digits = 1)

    batch_loss = Float32[]
    epochs = Matrix{Float64}(undef, 0, 6)
    step = 0
    t_rate = time(); n_rate = 0
    for epoch ∈ 1:nepochs
        eloss = 0f0; nb = 0
        t0 = time(); t_sub = 0.0
        ℓ_prev = nothing                       # device loss of the previous step, not yet read
        record(ℓr) = begin
            ℓ = Float32(ℓr)                    # host copy; waits for that step
            push!(batch_loss, ℓ)
            eloss += ℓ
            tb === nothing || log_value(tb, "train/loss", ℓ; step = length(batch_loss))
        end
        for (x, y) ∈ loader
            xr = dev(x)
            Yr = dev(onehot!(Yb_host, y))
            ts_step = time()
            _, ℓr, _, ts = Training.single_train_step!(AutoEnzyme(), obj, (xr, Yr), ts)
            # Reading the loss blocks until its step has finished. Reading the previous step's
            # loss instead keeps one step queued on the GPU while the host loads, copies and
            # dispatches the next, so the GPU does not idle for the host's part of every step.
            step == 0 && (Float32(ℓr); @info "first training step (includes XLA compile)" seconds = round(time() - ts_step; digits = 1))
            ℓ_prev === nothing || record(ℓ_prev)
            ℓ_prev = ℓr
            nb += 1; step += 1; n_rate += 1
            if tb !== nothing
                if n_rate == 50
                    log_value(tb, "train/img_per_s", n_rate * loader.batch / (time() - t_rate); step)
                    t_rate = time(); n_rate = 0
                end
                if step % SUBSET_EVERY == 0
                    te = time()
                    s1, s5 = evaluate(fwd, ts.parameters, Lux.testmode(ts.states), ys_sub, Xs_sub, dev)
                    log_value(tb, "val_subset/top1", 100s1; step)
                    log_value(tb, "val_subset/top5", 100s5; step)
                    de = time() - te
                    t_sub += de; t_rate += de
                end
            end
        end
        ℓ_prev === nothing || record(ℓ_prev)
        dt = time() - t0 - t_sub
        ips = nb * loader.batch / dt
        top1, top5 = evaluate(fwd, ts.parameters, Lux.testmode(ts.states), yv, Xv, dev)
        epochs = vcat(epochs, [epoch eloss/nb top1 top5 dt ips])
        if tb !== nothing
            log_value(tb, "val/top1", 100top1; step)
            log_value(tb, "val/top5", 100top5; step)
            log_value(tb, "train/epoch_loss", eloss/nb; step)
            log_value(tb, "time/epoch_s", dt; step)
            t_rate = time(); n_rate = 0
        end
        @printf("Epoch %2d/%d | loss %.4f | val top-1 %5.2f%% top-5 %5.2f%% | %6.1fs | %6d img/s | %d batches\n",
                epoch, nepochs, eloss/nb, 100top1, 100top5, dt, round(Int, ips), nb)
        flush(stdout)
        on_epoch(epoch, ts.parameters, ts.states)
    end
    return batch_loss, epochs
end

if abspath(PROGRAM_FILE) == @__FILE__
    nepochs   = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 10
    max_steps = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : typemax(Int)

    @info "Loading"
    @time N, ys, Xs, _ = load_train()
    @time yv, Xv = load_val()
    loader = DataLoader(Xs, ys; batch = BATCH, limit = max_steps)
    @info "images" train = N val = length(yv) batch = BATCH steps = length(loader) threads = Threads.nthreads()

    model = build_model(MODEL; nclass = NCLASS)
    @info "model" name = MODEL params = Lux.parameterlength(model) backend = "Reactant + Enzyme"

    ispath(CHKPT_DIR) && rm(CHKPT_DIR, force = true, recursive = true)
    mkpath(CHKPT_DIR); mkpath(RESULTS)
    cdev = cpu_device()
    save_checkpoint(epoch, ps, st) =
        jldsave(joinpath(CHKPT_DIR, "model_$epoch.jld2"); parameters = ps |> cdev, state = st |> cdev)

    obj = PRECISION == "bf16" ? objective_bf16 :
          PRECISION == "fp32" ? objective : error("PRECISION must be fp32 or bf16, got $PRECISION")
    @info "precision" PRECISION

    runname = "$MODEL-reactant$PTAG$STAG-b$BATCH-" * Dates.format(now(), "yyyymmdd-HHMM")
    tb      = TBLogger(joinpath(TB_DIR, runname), tb_overwrite)
    @info "TensorBoard" logdir = joinpath(TB_DIR, runname)

    @info "Training"
    @time batch_loss, epochs = train_reactant(model, loader, yv, Xv; nepochs, tb, on_epoch = save_checkpoint, obj)

    writedlm(joinpath(RESULTS, "train_loss$TAG.csv"), batch_loss, ',')
    writedlm(joinpath(RESULTS, "train_epochs$TAG.csv"),
             vcat(["epoch" "loss" "val_top1" "val_top5" "seconds" "img_per_s"], epochs), ',')
    @printf("best val top-1 %.2f%%  top-5 %.2f%%\n", 100maximum(epochs[:, 3]), 100maximum(epochs[:, 4]))
end
