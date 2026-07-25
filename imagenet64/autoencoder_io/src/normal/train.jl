# Single-GPU training: 10 epochs over ImageNet-64, batch 1024, Adam 1e-3, MSE.
#
#   julia --project src/normal/train.jl         # full: 10 epochs, ~5 min
#   julia --project src/normal/train.jl 20      # smoke test: cap at 20 steps/epoch
#
# The loop itself is `train_normal` in src/shared.jl (also the baseline of data_parallel/train.jl).
# Writes build/checkpoint/model_<epoch>.jld2 and results/train_{epochs,loss}.csv.
# Plotting lives in reconstruct.jl: loading CairoMakie here made Julia 1.12 spend
# >9 min in type inference on the first Zygote pullback.

include(joinpath(@__DIR__, "..", "shared.jl"))
using JLD2, DelimitedFiles

@info "Loading"
@time N, ys, Xs, μs = load_train()
@info "images" N shards = length(Xs)

model = build_model()

const NEPOCHS   = 10
const MAX_STEPS = isempty(ARGS) ? typemax(Int) : parse(Int, ARGS[1])

ispath(CHKPT_DIR) && rm(CHKPT_DIR, force = true, recursive = true)
mkpath(CHKPT_DIR); mkpath(RESULTS); mkpath(RESOURCES)

save_checkpoint(epoch, ps, st) =
    jldsave(joinpath(CHKPT_DIR, "model_$epoch.jld2");
            parameters = ps |> cpu_device(), state = st |> cpu_device())

@info "Training"
@time batch_loss, epochs, _, _ = train_normal(model, Xs; nepochs = NEPOCHS, max_steps = MAX_STEPS,
                                              on_epoch = save_checkpoint)

writedlm(joinpath(RESULTS, "train_loss.csv"), batch_loss, ',')
writedlm(joinpath(RESULTS, "train_epochs.csv"),
         vcat(["epoch" "loss" "seconds" "img_per_s"], epochs), ',')

# IO accounting: bytes that crossed host → device over the run.
bytes = 64 * 64 * 3 * sizeof(Float32) * length(batch_loss) * BATCH
@printf("H→D bytes   : %.1f GB over %d epochs\n", bytes / 1e9, NEPOCHS)
@printf("total wall  : %.1f s   (%.2f GB/s average)\n", sum(epochs[:, 3]), bytes / 1e9 / sum(epochs[:, 3]))
@printf("steady wall : %.1f s/epoch (epochs 2..%d)\n", mean(epochs[2:end, 3]), NEPOCHS)
